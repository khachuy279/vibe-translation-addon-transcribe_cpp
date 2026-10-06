"""Engine FireRed-VAD (DFSMN SOTA Streaming VAD — Xiaohongshu / FireRedTeam) — **ONNX, không torch**.

Trước đây engine này dùng gói `fireredvad`, mà gói đó `import torch` **ngay ở cấp module**
(`fireredvad/core/detect_model.py`, `audio_feat.py`, `stream_vad.py`) ⇒ nó là lý do chính khiến
PyTorch buộc phải có mặt trong môi trường runtime. Nay toàn bộ đường suy luận chạy bằng
onnxruntime với model ONNX **chính thức** của upstream
(`fireredvad_stream_vad_with_cache.onnx`); xem `backend/vad/engines/firered_onnx.py`.

Hình học frame (upstream `fireredvad/core/constants.py`, không đổi được):
    FRAME_LENGTH_SAMPLE = 400 (cửa sổ 25 ms) · FRAME_SHIFT_SAMPLE = 160 (hop 10 ms)

Vì `KaldifeatFbank` tạo `OnlineFbank` MỚI mỗi lần gọi với `snip_edges=True`, một cửa sổ 400 mẫu
cho ĐÚNG 1 frame — nên engine tự gom cửa sổ trượt 400 mẫu (KHÔNG zero-pad: frame đầu tiên chỉ
được phát khi đã có đủ 400 mẫu THẬT, giống hệt `vad_framewise` của upstream) rồi gọi
`detect_frame()` mỗi 160 mẫu.

Tương đương với bản torch: đo trên 12.621 frame của 4 file audio cho `max|Δp| ≤ 1e-6` và
**0 frame lệch quyết định** tại ngưỡng 0.30/0.40/0.50 (`backend/tests/test_62_vad_onnx_parity.py`).
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
from backend.vad.engines.firered_onnx import (
    FRAME_LENGTH_SAMPLE,
    FRAME_SHIFT_SAMPLE,
    FireRedStreamVadOnnx,
    ensure_firered_onnx_files,
)
from backend.utils.logger import logger


@dataclass
class _Session:
    """State riêng của FireRed cho 1 phiên: model ONNX + bộ gom cửa sổ 400 mẫu."""

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
    """Engine VAD FireRed sử dụng model ONNX (onnxruntime)."""

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
        #: Mép đoạn nói được VAD báo SỚM hơn thực tế ngần này ms (`speech_start_frame` đã trừ
        #: `pad_start_frame`). Pipeline B (OFFLINE_BATCH) KHÔNG chạy VAD để cắt câu, nhưng vẫn
        #: dùng con số này để canh mốc phụ đề/lồng tiếng (xem
        #: `LookaheadSessionState._refresh_start_pad`).
        self.start_pad_ms = int(round(int(cfg.pad_start_frame) * self.frame_samples / 16000.0 * 1000.0))

        logger.info(
            f"FireRed-VAD sẵn sàng (ONNX, cửa sổ {FRAME_LENGTH_SAMPLE} mẫu, hop {FRAME_SHIFT_SAMPLE} mẫu, "
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
        """Đảm bảo `cmvn.ark` + `fireredvad_stream_vad_with_cache.onnx` tồn tại."""
        model_dir.mkdir(parents=True, exist_ok=True)
        ensure_firered_onnx_files(model_dir)

    def _ensure_model_files(self) -> None:
        """Giữ lại cho tương thích: uỷ quyền cho `_ensure_files_in`."""
        self._ensure_files_in(self.model_dir)

    # ------------------------------------------------------------------ config
    def _resolve_effective(self, threshold: Optional[float], silence_ms: Optional[int]):
        """Giải giá trị đang có hiệu lực: knob phiên (hệ số) hay mặc định config engine."""
        cfg = config.vad.firered
        eff_threshold = float(threshold) if threshold is not None else float(cfg.speech_threshold)
        if silence_ms is None:
            eff_silence_frames = int(cfg.min_silence_frame)
        else:
            # 1 frame = 10 ms ⇒ ms -> frame, tối thiểu 1 frame.
            eff_silence_frames = max(1, int(round(float(silence_ms) / 10.0)))
        return eff_threshold, eff_silence_frames

    def _build_vad(self, threshold: Optional[float], silence_ms: Optional[int]) -> FireRedStreamVadOnnx:
        """Dựng `FireRedStreamVadOnnx` từ config backend (khớp 1-1 field của upstream)."""
        cfg = config.vad.firered
        eff_threshold, eff_silence_frames = self._resolve_effective(threshold, silence_ms)
        return FireRedStreamVadOnnx(
            self.model_dir,
            smooth_window_size=int(cfg.smooth_window_size),
            speech_threshold=eff_threshold,
            pad_start_frame=int(cfg.pad_start_frame),
            min_speech_frame=int(cfg.min_speech_frame),
            max_speech_frame=int(cfg.max_speech_frame),
            min_silence_frame=eff_silence_frames,
        )

    def create_initial_state(
        self,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADStreamState:
        state = VADStreamState()
        state.engine_state = _Session(vad=self._build_vad(threshold, silence_ms))
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
            session = self.create_initial_state(threshold, silence_ms).engine_state
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
