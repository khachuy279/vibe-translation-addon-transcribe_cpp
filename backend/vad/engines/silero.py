"""Engine Silero VAD v5 — chạy bằng **onnxruntime** (không còn PyTorch).

    from backend.vad.silero_onnx import SileroVadOnnx, SileroVADIterator

    model = SileroVadOnnx(Path("backend/models/silero_vad.onnx"))
    it = SileroVADIterator(model, threshold=0.5, sampling_rate=16000,
                           min_silence_duration_ms=100, speech_pad_ms=30)
    for frame in audio:            # 512 mẫu @ 16 kHz
        res = it(frame)            # {'start': …} | {'end': …} | None

Ghi chú thiết kế:

* `SileroVADIterator` là **bản port chính xác** `silero_vad.VADIterator` (xem
  `backend/vad/silero_onnx.py`) — giữ nguyên mọi ngưỡng, kể cả hằng số trễ `threshold - 0.15`
  mà upstream hardcode. Đổi bất kỳ nhánh nào là đổi hành vi cắt câu.
* Model ONNX cho **cùng xác suất từng frame** với model JIT cũ (`max|Δp| ≤ 2e-6`, 0 frame lệch
  quyết định trên 3.797 cửa sổ — `backend/tests/test_62_vad_onnx_parity.py`).
* State do caller giữ (không giấu trong model) nên `create_initial_state()` **không phải nạp
  model riêng cho từng phiên** như bản JIT (trước đây ~93 ms/phiên, 2,3 MB/phiên).
* Input phải là float32 trong [-1, 1] đúng `frame_samples = 512` mẫu ⇒
  `int16.astype(float32) / 32768.0` tạo **bản sao**, KHÔNG đụng vào buffer của processor
  (audio tới ASR phải nguyên byte — xem hợp đồng ở `backend/vad/base.py`).
"""

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from backend.config import config, MODELS_DIR
from backend.vad.base import (
    EVENT_END,
    EVENT_START,
    BaseVADEngine,
    VADResult,
    VADStreamState,
)
from backend.vad.silero_onnx import (
    WINDOW_SAMPLES,
    SileroVADIterator,
    SileroVadOnnx,
    ensure_silero_onnx_model,
)
from backend.utils.logger import logger

#: VADIterator chỉ nhận 512 mẫu @ 16 kHz (32 ms).
WINDOW_MS = 32


@dataclass
class _Session:
    """State riêng của Silero cho 1 phiên: iterator ONNX + bộ đếm frame."""

    iterator: SileroVADIterator
    frames: int = 0


class SileroVADEngine(BaseVADEngine):
    """Engine Silero VAD chính thức (ONNX Runtime)."""

    name = "silero-vad"
    default_threshold = 0.5
    frame_samples = WINDOW_SAMPLES

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = self._resolve_model_path(model_path)
        self._ensure_model_file()

        cfg = config.vad.silero
        # Pre-roll tối đa: `speech_pad_ms` + đúng 1 cửa sổ (VADIterator báo 'start' ở
        # chính frame vượt ngưỡng đầu tiên, lùi lại `speech_pad_samples`).
        self.max_lookback_frames = max(
            1, int(ceil((float(cfg.speech_pad_ms) / 1000.0 * 16000 + WINDOW_SAMPLES) / WINDOW_SAMPLES))
        )
        #: `res["start"]` của VADIterator đã lùi lại `speech_pad_samples` ⇒ mép đoạn nói được
        #: báo SỚM hơn thực tế ngần này ms. Pipeline Lookahead dùng để canh lại mốc phụ đề/TTS.
        self.start_pad_ms = int(cfg.speech_pad_ms)

        logger.info(
            f"Silero-VAD sẵn sàng (ONNX, cửa sổ {WINDOW_SAMPLES} mẫu, model={self.model_path})",
            extra={"module_tag": "VAD"},
        )

    # ------------------------------------------------------------------ model files
    def _resolve_model_path(self, explicit_path: Optional[Union[str, Path]]) -> Path:
        if explicit_path and Path(explicit_path).exists():
            return Path(explicit_path)
        return MODELS_DIR / "silero_vad.onnx"

    @classmethod
    def prepare_files(cls) -> None:
        """Đảm bảo `silero_vad.onnx` có trong `backend/models` (ưu tiên offline, rồi mới tải)."""
        try:
            ensure_silero_onnx_model(MODELS_DIR / "silero_vad.onnx")
        except FileNotFoundError as exc:
            # Không ném ở đây: `VADEngineFactory` log cảnh báo rồi vẫn thử dựng engine để
            # thông báo lỗi cụ thể hơn (xem `get_engine`).
            logger.warning(str(exc), extra={"module_tag": "VAD"})

    def _ensure_model_file(self) -> None:
        """Đảm bảo file `silero_vad.onnx` có trong `backend/models` (ưu tiên offline)."""
        self.prepare_files()

    def _load_model(self) -> SileroVadOnnx:
        """Nạp model ONNX: CHỈ nạp từ file cục bộ trong backend/models."""
        if not self.model_path.is_file():
            self._ensure_model_file()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy file model silero_vad.onnx tại {self.model_path}! "
                "Hãy chạy `python -m backend.main` một lần để tự tải, hoặc copy thủ công vào "
                f"{self.model_path.parent}."
            )
        return SileroVadOnnx(self.model_path)

    # ------------------------------------------------------------------ config
    def _resolve_effective(self, threshold: Optional[float], silence_ms: Optional[int]):
        """Giá trị đang có hiệu lực: knob phiên (nếu có) hay mặc định config engine."""
        cfg = config.vad.silero
        eff_threshold = float(threshold) if threshold is not None else float(cfg.threshold)
        eff_silence_ms = int(cfg.min_silence_duration_ms if silence_ms is None else silence_ms)
        return eff_threshold, eff_silence_ms

    def create_initial_state(
        self,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADStreamState:
        cfg = config.vad.silero
        eff_threshold, eff_silence_ms = self._resolve_effective(threshold, silence_ms)

        model = self._load_model()
        iterator = SileroVADIterator(
            model=model,
            threshold=eff_threshold,
            sampling_rate=16000,
            min_silence_duration_ms=eff_silence_ms,
            speech_pad_ms=int(cfg.speech_pad_ms),
        )

        state = VADStreamState()
        state.engine_state = _Session(iterator=iterator)
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

        eff_threshold, eff_silence_ms = self._resolve_effective(threshold, silence_ms)
        self._sync_runtime_config(session.iterator, eff_threshold, eff_silence_ms)

        # Bản sao float32 [-1, 1] chuẩn docs; KHÔNG đụng vào mảng Int16 của processor.
        # (Hợp đồng: audio tới ASR phải nguyên byte ⇒ mọi chuyển đổi đều tạo bản sao.)
        samples = frame_int16.astype(np.float32) / 32768.0

        res = session.iterator(samples)

        session.frames += 1
        # `is_speech` = BẰNG CHỨNG của riêng frame này (xác suất ≥ ngưỡng), KHÔNG phải
        # `iterator.triggered` (triggered giữ nguyên True suốt `min_silence_samples` nên
        # không dùng để đếm im lặng được).
        last_prob = float(session.iterator.last_prob)
        result = VADResult(
            probability=last_prob,
            is_speech=last_prob >= eff_threshold,
        )
        if res:
            if "start" in res:
                # `start` là chỉ số mẫu (tính từ đầu phiên) của mép đầu đoạn nói, đã trừ
                # `speech_pad_samples`. Số frame cần xả lại = (mẫu hiện tại - mép đầu)/512.
                consumed = session.frames * WINDOW_SAMPLES
                lookback_samples = max(0, consumed - int(res["start"]))
                result.event = EVENT_START
                result.lookback_frames = int(ceil(lookback_samples / WINDOW_SAMPLES))
                result.is_speech = True
            elif "end" in res:
                result.event = EVENT_END
        return result

    @staticmethod
    def _sync_runtime_config(iterator: Any, threshold: float, silence_ms: int) -> None:
        """Áp cấu hình runtime lên iterator đang chạy."""
        SileroVADIterator._sync_runtime_config(iterator, threshold, silence_ms)
