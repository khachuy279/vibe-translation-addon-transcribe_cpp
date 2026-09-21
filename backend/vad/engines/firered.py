"""Engine FireRed-VAD (DFSMN SOTA Streaming VAD — Xiaohongshu / FireRedTeam).

Bản viết lại dùng **đúng API công khai trong docs** của gói `fireredvad`:

    from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

    vad_config = FireRedStreamVadConfig(
        use_gpu=False, smooth_window_size=5, speech_threshold=0.4,
        pad_start_frame=5, min_speech_frame=8, max_speech_frame=2000,
        min_silence_frame=20, chunk_max_frame=30000)
    stream_vad = FireRedStreamVad.from_pretrained("pretrained_models/FireRedVAD/Stream-VAD", vad_config)

Hình học frame (upstream `fireredvad/core/constants.py`, không đổi được):
    FRAME_LENGTH_SAMPLE = 400 (cửa sổ 25 ms) · FRAME_SHIFT_SAMPLE = 160 (hop 10 ms)

Vì `AudioFeat.extract()` tạo `OnlineFbank` MỚI mỗi lần gọi với `snip_edges=True`, một
cửa sổ 400 mẫu cho ĐÚNG 1 frame — nên engine tự gom cửa sổ trượt 400 mẫu (KHÔNG zero-pad
như bản cũ: frame đầu tiên chỉ được phát khi đã có đủ 400 mẫu THẬT, giống hệt
`vad_framewise` của upstream) rồi gọi `detect_frame()` mỗi 160 mẫu.

State (`FireRedStreamVad`) gồm model caches + postprocessor nên **phải tạo mới cho mỗi
phiên**; trọng số chỉ 2,3 MB nên chi phí chấp nhận được (đo trong
`backend/tests/test_41_vad_engine_contract.py`).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

from backend.config import config, MODELS_DIR
from backend.vad.base import (
    EVENT_END,
    EVENT_START,
    BaseVADEngine,
    VADResult,
    VADStreamState,
)
from backend.utils.logger import logger

#: Hình học frame của upstream (`fireredvad/core/constants.py`).
FRAME_LENGTH_SAMPLE = 400   # cửa sổ 25 ms
FRAME_SHIFT_SAMPLE = 160    # hop 10 ms


@dataclass
class _Session:
    """State riêng của FireRed cho 1 phiên: model VAD + bộ gom cửa sổ 400 mẫu."""

    vad: Any
    pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))

    def push(self, frame_int16: np.ndarray) -> List[np.ndarray]:
        """Nạp thêm `frame_int16` và trả các cửa sổ 400 mẫu MỚI đã đủ (0 hoặc 1 mỗi lần).

        `pending` luôn ngắn hơn `window + hop` mẫu nên chi phí cấp phát ở đây không đáng
        kể (~1 KB/frame, 100 frame/giây).
        """
        self.pending = (
            np.concatenate((self.pending, frame_int16))
            if self.pending.size
            else np.array(frame_int16, dtype=np.int16, copy=True)
        )
        windows: List[np.ndarray] = []
        while self.pending.size >= FRAME_LENGTH_SAMPLE:
            windows.append(self.pending[:FRAME_LENGTH_SAMPLE])
            self.pending = self.pending[FRAME_SHIFT_SAMPLE:]
        return windows


class FireRedVADEngine(BaseVADEngine):
    """Engine VAD FireRed sử dụng gói `fireredvad` theo đúng docs."""

    name = "firered-vad"
    default_threshold = 0.4
    #: Bước nhảy processor phải cắt = FRAME_SHIFT_SAMPLE (10 ms).
    frame_samples = FRAME_SHIFT_SAMPLE

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self.model_dir = self.resolve_model_dir(model_dir)
        self._ensure_model_files()

        cfg = config.vad.firered
        # Pre-roll tối đa mà VAD có thể yêu cầu xả lại khi báo START:
        # các frame thuộc `pad_start_frame` + số frame cần để xác nhận (`min_speech_frame`).
        self.max_lookback_frames = max(0, int(cfg.pad_start_frame) + int(cfg.min_speech_frame))

        logger.info(
            f"FireRed-VAD sẵn sàng (cửa sổ {FRAME_LENGTH_SAMPLE} mẫu, hop {FRAME_SHIFT_SAMPLE} mẫu, "
            f"model_dir={self.model_dir})",
            extra={"module_tag": "VAD"},
        )

    # ------------------------------------------------------------------ model files
    @staticmethod
    def resolve_model_dir(explicit_dir: Optional[Union[str, Path]] = None) -> Path:
        if explicit_dir and Path(explicit_dir).exists():
            return Path(explicit_dir)
        return MODELS_DIR / "firered_stream" / "Stream-VAD"

    @classmethod
    def prepare_files(cls) -> None:
        """QWEN-Q2: tải model NGOÀI lock cấp lớp. Xem `BaseVADEngine.prepare_files`."""
        cls._ensure_files_in(cls.resolve_model_dir())

    @staticmethod
    def _ensure_files_in(model_dir: Path) -> None:
        """Tự động tải model từ HuggingFace nếu chưa tồn tại cục bộ."""
        model_dir.mkdir(parents=True, exist_ok=True)
        cmvn_file = model_dir / "cmvn.ark"
        model_file = model_dir / "model.pth.tar"

        if not cmvn_file.exists() or not model_file.exists():
            from huggingface_hub import hf_hub_download
            logger.info("Đang tải model FireRed Stream-VAD từ HuggingFace...", extra={"module_tag": "VAD"})
            parent_dir = model_dir.parent
            hf_hub_download("FireRedTeam/FireRedVAD", "Stream-VAD/cmvn.ark", local_dir=str(parent_dir))
            hf_hub_download("FireRedTeam/FireRedVAD", "Stream-VAD/model.pth.tar", local_dir=str(parent_dir))

    def _ensure_model_files(self) -> None:
        """Giữ lại cho tương thích: uỷ quyền cho `_ensure_files_in`."""
        self._ensure_files_in(self.model_dir)

    # ------------------------------------------------------------------ config
    def _resolve_effective(self, threshold: Optional[float], silence_ms: Optional[int]):
        """Giải giá trị đang có hiệu lực: knob phiên (nếu có) hay mặc định config engine."""
        cfg = config.vad.firered
        eff_threshold = float(threshold) if threshold is not None else float(cfg.speech_threshold)
        if silence_ms is None:
            eff_silence_frames = int(cfg.min_silence_frame)
        else:
            # 1 frame = 10 ms ⇒ ms -> frame, tối thiểu 1 frame.
            eff_silence_frames = max(1, int(round(float(silence_ms) / 10.0)))
        return eff_threshold, eff_silence_frames

    def _build_config(self, threshold: Optional[float], silence_ms: Optional[int]):
        """Dựng `FireRedStreamVadConfig` từ config backend (khớp 1-1 field của docs)."""
        from fireredvad import FireRedStreamVadConfig

        cfg = config.vad.firered
        eff_threshold, eff_silence_frames = self._resolve_effective(threshold, silence_ms)
        return FireRedStreamVadConfig(
            use_gpu=bool(cfg.use_gpu),
            smooth_window_size=int(cfg.smooth_window_size),
            speech_threshold=eff_threshold,
            pad_start_frame=int(cfg.pad_start_frame),
            min_speech_frame=int(cfg.min_speech_frame),
            max_speech_frame=int(cfg.max_speech_frame),
            min_silence_frame=eff_silence_frames,
            chunk_max_frame=int(cfg.chunk_max_frame),
        )

    def create_initial_state(
        self,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADStreamState:
        from fireredvad import FireRedStreamVad

        vad_config = self._build_config(threshold, silence_ms)
        stream_vad = FireRedStreamVad.from_pretrained(str(self.model_dir), vad_config)

        state = VADStreamState()
        state.engine_state = _Session(vad=stream_vad)
        return state

    # ------------------------------------------------------------------ streaming
    def is_speech(
        self,
        frame_int16: np.ndarray,
        state: VADStreamState,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADResult:
        session: Optional[_Session] = state.engine_state
        if session is None:
            session = _Session(vad=self.create_initial_state(threshold, silence_ms).engine_state.vad)
            state.engine_state = session

        eff_threshold, eff_silence_frames = self._resolve_effective(threshold, silence_ms)
        self._sync_runtime_config(session.vad, eff_threshold, eff_silence_frames)

        result = VADResult()
        for window in session.push(frame_int16):
            frame_result = session.vad.detect_frame(window)
            result.probability = float(frame_result.smoothed_prob)
            result.is_speech = bool(frame_result.is_speech)
            if frame_result.is_speech_end:
                result.event = EVENT_END
                result.lookback_frames = 0
            elif frame_result.is_speech_start:
                # `speech_start_frame` là frame 1-based mà VAD coi là mép đầu đoạn nói
                # (đã trừ `pad_start_frame`). Số frame cần xả lại = frame hiện tại - mép đầu.
                result.event = EVENT_START
                result.lookback_frames = max(0, int(frame_result.frame_idx) - int(frame_result.speech_start_frame))
        return result

    @staticmethod
    def _sync_runtime_config(vad: Any, threshold: float, silence_frames: int) -> None:
        """Áp cấu hình runtime lên postprocessor đang chạy (đổi ngưỡng/silence giữa phiên)."""
        post = getattr(vad, "postprocessor", None)
        if post is None:
            return
        if float(post.speech_threshold) != float(threshold):
            post.speech_threshold = float(threshold)
            if getattr(vad, "config", None) is not None:
                vad.config.speech_threshold = float(threshold)
        if int(post.min_silence_frame) != int(silence_frames):
            post.min_silence_frame = int(silence_frames)
            if getattr(vad, "config", None) is not None:
                vad.config.min_silence_frame = int(silence_frames)
