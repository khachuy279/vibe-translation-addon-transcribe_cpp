"""Engine FireRed-VAD (DFSMN SOTA Streaming VAD - Xiaohongshu / FireRedTeam)."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch

from backend.config import config, MODELS_DIR
from backend.vad.base import BaseVADEngine, VADResult, VADStreamState
from backend.utils.logger import logger


class FireRedVADEngine(BaseVADEngine):
    """Engine VAD FireRed chính thức sử dụng gói fireredvad."""

    name = "firered-vad"
    default_threshold = config.vad.firered.threshold or config.vad.threshold
    #: Cửa sổ phân tích của model = 25 ms (upstream `FRAME_LENGTH_SAMPLE`). CỐ ĐỊNH.
    frame_window_samples = 400
    #: Bước nhảy mặc định = 10 ms (upstream `FRAME_SHIFT_SAMPLE`). `__init__` ghi đè
    #: theo `config.vad.firered.frame_hop_ms`. Bản cũ để 400 (25 ms) ⇒ sai nhịp 2,5×.
    native_frame_samples = 160  # 10ms @ 16kHz

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        # Bước nhảy theo cấu hình (mặc định 10 ms). VADProcessor đọc
        # `native_frame_samples` để quyết định cắt frame đầu vào, nên đặt ở instance.
        hop_ms = int(getattr(config.vad.firered, "frame_hop_ms", 10) or 10)
        self.native_frame_samples = max(1, int(16000 * hop_ms / 1000))
        self.model_dir = self._resolve_model_dir(model_dir)
        self._ensure_model_files()

        from fireredvad.core.audio_feat import AudioFeat
        from fireredvad.core.detect_model import DetectModel

        cmvn_path = str(self.model_dir / "cmvn.ark")
        self.audio_feat = AudioFeat(cmvn_path)
        self.vad_model = DetectModel.from_pretrained(str(self.model_dir))
        self.vad_model.eval()
        self.vad_model.cpu()

        # Giới hạn PyTorch CPU threads xuống 2 để tránh đánh thức toàn bộ nhân CPU 40 lần/giây
        if torch.get_num_threads() > 2:
            torch.set_num_threads(2)

        logger.info("Model FireRed Stream sẵn sàng", extra={"module_tag": "VAD"})

    def _resolve_model_dir(self, explicit_dir: Optional[Union[str, Path]]) -> Path:
        return self.resolve_model_dir(explicit_dir)

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

    def create_initial_state(self, threshold: Optional[float] = None) -> VADStreamState:
        from fireredvad.core.stream_vad_postprocessor import StreamVadPostprocessor

        cfg = config.vad.firered
        active_thresh = threshold if threshold is not None else (cfg.threshold or config.vad.threshold)

        state = VADStreamState()
        state.firered_postprocessor = StreamVadPostprocessor(
            smooth_window_size=cfg.smooth_window_size,
            speech_threshold=active_thresh,
            pad_start_frame=cfg.pad_start_frame,
            min_speech_frame=cfg.min_speech_frame,
            max_speech_frame=2000,
            min_silence_frame=cfg.min_silence_frame,
        )
        state.firered_caches = None
        return state

    def is_speech(
        self,
        chunk_float32: Optional[np.ndarray],
        state: VADStreamState,
        threshold: float,
        chunk_raw: Optional[bytes] = None,
    ) -> VADResult:
        if state.firered_postprocessor is None:
            init_s = self.create_initial_state(threshold)
            state.firered_postprocessor = init_s.firered_postprocessor
            state.firered_caches = init_s.firered_caches

        # Cập nhật threshold động nếu có thay đổi
        if threshold is not None and state.firered_postprocessor.speech_threshold != threshold:
            state.firered_postprocessor.speech_threshold = float(threshold)

        # Tránh roundtrip F32 -> I16: ưu tiên dùng trực tiếp chunk_raw Int16 nếu có
        if chunk_raw is not None:
            chunk_int16 = np.frombuffer(chunk_raw, dtype=np.int16)
        elif chunk_float32 is not None:
            arr = np.asarray(chunk_float32, dtype=np.float32)
            chunk_int16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            raise ValueError("Cần cung cấp ít nhất chunk_raw hoặc chunk_float32 cho FireRed VAD")

        # ── CỬA SỔ TRƯỢT 25 ms, BƯỚC NHẢY 10 ms ─────────────────────────────────
        # `AudioFeat.extract()` dùng `OnlineFbank` MỚI mỗi lần gọi với `frame_shift=10ms`
        # và `snip_edges=True` ⇒ input 400 samples cho ĐÚNG 1 frame. Vì vậy để có 1 frame
        # mỗi 10 ms ta phải tự duy trì cửa sổ 400 samples và trượt đi 160 samples mỗi lần,
        # thay vì đưa thẳng 160 samples (sẽ ra 0 frame).
        win_len = self.frame_window_samples
        prev = state.firered_window
        if prev is None:
            prev = np.zeros(win_len, dtype=np.int16)
        n = chunk_int16.shape[0]
        if n >= win_len:
            window = chunk_int16[-win_len:]
        else:
            # Giữ `win_len - n` mẫu CUỐI của cửa sổ trước rồi nối phần mới.
            # ⚠️ Phải là `prev[-(win_len - n):]` (lấy từ CUỐI). Viết `prev[win_len - n:]`
            # là sai: trên mảng 400 phần tử, `prev[240:]` chỉ còn 160 phần tử ⇒ cửa sổ
            # ngắn dần (400 → 320) và model nhận 0 frame ⇒ RuntimeError.
            window = np.concatenate((prev[-(win_len - n):], chunk_int16))
        state.firered_window = window

        feat, _ = self.audio_feat.extract(window)
        with torch.no_grad():
            probs, state.firered_caches = self.vad_model.forward(
                feat.unsqueeze(0), caches=state.firered_caches
            )

        raw_prob = probs.squeeze().tolist()
        if isinstance(raw_prob, list):
            prob_val = float(raw_prob[-1]) if raw_prob else 0.0
        else:
            prob_val = float(raw_prob)

        frame_result = state.firered_postprocessor.process_one_frame(prob_val)

        event = None
        if frame_result.is_speech_start:
            event = "START"
        elif frame_result.is_speech_end:
            event = "END"

        return VADResult(
            is_speech=bool(frame_result.is_speech),
            probability=float(frame_result.smoothed_prob),
            event=event,
        )
