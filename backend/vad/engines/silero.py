"""Engine Silero VAD v5/v6 (TorchScript JIT) — dùng đúng API trong docs.

    from silero_vad import load_silero_vad, VADIterator

    model = load_silero_vad()
    vad_iterator = VADIterator(model, threshold=0.5, sampling_rate=16000,
                               min_silence_duration_ms=100, speech_pad_ms=30)
    for chunk in audio:            # 512 mẫu @ 16 kHz
        speech_dict = vad_iterator(chunk)

Ghi chú thiết kế:

* `VADIterator` **không** trả xác suất (chỉ trả `{'start': …}`, `{'end': …}` hoặc
  `None`), nên cần một wrapper mỏng ghi lại giá trị mà model trả về. Wrapper chỉ ĐỌC và
  uỷ quyền mọi thuộc tính khác (`reset_states`, tham số…) cho model thật ⇒ không đổi hành
  vi của thư viện.
* Model Silero **có state nội bộ** (`reset_states()` ghi vào chính model), nên mỗi phiên
  phải có model riêng: `create_initial_state()` nạp JIT (~93 ms, 2,3 MB) cho từng phiên.
* Input phải là float32 trong [-1, 1] đúng `frame_samples = 512` mẫu ⇒
  `int16.astype(float32) / 32768.0` tạo **bản sao** (không đụng vào buffer của processor).
"""

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import shutil
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
from backend.utils.logger import logger

#: VADIterator chỉ nhận 512 mẫu @ 16 kHz (32 ms).
WINDOW_SAMPLES = 512
WINDOW_MS = 32


class _ProbProbe:
    """Wrapper CHỈ-ĐỌC quanh model Silero để ghi lại xác suất của frame vừa xử lý.

    `VADIterator` không trả xác suất nên đây là cách duy nhất để có `probability` cho
    log/metric mà KHÔNG tự tính lại model. Mọi truy cập khác được chuyển tiếp nguyên vẹn
    cho model thật (`__getattr__`), nên `VADIterator.reset_states()` vẫn gọi đúng
    `model.reset_states()`.
    """

    def __init__(self, model: Any):
        self._model = model
        self.last_prob: float = 0.0

    def __call__(self, x, sr: int):
        out = self._model(x, sr)
        try:
            self.last_prob = float(out.item())
        except Exception:  # noqa: BLE001 - chỉ là thông tin phụ
            self.last_prob = 0.0
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


@dataclass
class _Session:
    """State riêng của Silero cho 1 phiên: model JIT + `VADIterator` + bộ đếm frame."""

    iterator: Any
    probe: _ProbProbe
    frames: int = 0


class SileroVADEngine(BaseVADEngine):
    """Engine Silero VAD chính thức (JIT)."""

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

        logger.info(
            f"Silero-VAD sẵn sàng (cửa sổ {WINDOW_SAMPLES} mẫu, model={self.model_path})",
            extra={"module_tag": "VAD"},
        )

    # ------------------------------------------------------------------ model files
    def _resolve_model_path(self, explicit_path: Optional[Union[str, Path]]) -> Path:
        if explicit_path and Path(explicit_path).exists():
            return Path(explicit_path)
        return MODELS_DIR / "silero_vad.jit"

    def _ensure_model_file(self) -> None:
        """Đảm bảo file `silero_vad.jit` có trong `backend/models` (ưu tiên offline)."""
        if self.model_path.exists():
            return
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            import importlib.resources as impresources

            src = str(impresources.files("silero_vad.data").joinpath("silero_vad.jit"))
            shutil.copy(src, self.model_path)
            logger.info(f"Đã sao chép silero_vad.jit từ package vào {self.model_path}",
                        extra={"module_tag": "VAD"})
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Không thể copy bundled silero_vad.jit ({exc}); sẽ dùng model của package.",
                extra={"module_tag": "VAD"},
            )

    def _load_model(self) -> Any:
        """Nạp model JIT: dùng file cục bộ nếu có, ngược lại dùng loader chuẩn của docs."""
        from silero_vad.utils_vad import init_jit_model

        if self.model_path.exists():
            return init_jit_model(str(self.model_path))
        from silero_vad import load_silero_vad

        return load_silero_vad()

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
        from silero_vad import VADIterator

        cfg = config.vad.silero
        eff_threshold, eff_silence_ms = self._resolve_effective(threshold, silence_ms)

        model = self._load_model()
        probe = _ProbProbe(model)
        iterator = VADIterator(
            model=probe,
            threshold=eff_threshold,
            sampling_rate=16000,
            min_silence_duration_ms=eff_silence_ms,
            speech_pad_ms=int(cfg.speech_pad_ms),
        )

        state = VADStreamState()
        state.engine_state = _Session(iterator=iterator, probe=probe)
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
        samples = frame_int16.astype(np.float32) / 32768.0

        import torch

        with torch.no_grad():
            res = session.iterator(torch.from_numpy(samples))

        session.frames += 1
        result = VADResult(
            probability=float(session.probe.last_prob),
            is_speech=bool(getattr(session.iterator, "triggered", False)),
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
                result.is_speech = False
        return result

    @staticmethod
    def _sync_runtime_config(iterator: Any, threshold: float, silence_ms: int) -> None:
        """Áp cấu hình runtime lên `VADIterator` đang chạy."""
        if float(getattr(iterator, "threshold", threshold)) != float(threshold):
            iterator.threshold = float(threshold)
        min_silence_samples = 16000 * float(silence_ms) / 1000.0
        if float(getattr(iterator, "min_silence_samples", min_silence_samples)) != min_silence_samples:
            iterator.min_silence_samples = min_silence_samples
