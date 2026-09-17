"""Engine Silero VAD v5 (PyTorch / TorchScript JIT VAD)."""

import importlib.resources as impresources
from pathlib import Path
import shutil
from typing import Any, Optional, Union
import numpy as np
import torch

from backend.config import config, MODELS_DIR
from backend.vad.base import BaseVADEngine, VADResult, VADStreamState
from backend.utils.logger import logger


class SileroModelProbe:
    """Proxy ghi nhận xác suất liên tục chính xác từ Silero model."""

    def __init__(self, model: Any):
        self._model = model
        self.last_prob: float = 0.0

    def __call__(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        out = self._model(x, sr)
        try:
            self.last_prob = float(out.item())
        except Exception:
            self.last_prob = 0.0
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class SileroVADEngine(BaseVADEngine):
    """Engine Silero VAD v5 chính thức."""

    name = "silero-vad"
    default_threshold = config.vad.silero.threshold or config.vad.threshold
    native_frame_samples = 512  # 32ms @ 16kHz

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = self._resolve_model_path(model_path)
        self._ensure_model_file()

        from silero_vad.utils_vad import init_jit_model
        self._prototype_model = init_jit_model(str(self.model_path))

        if torch.get_num_threads() > 2:
            torch.set_num_threads(2)

        logger.info("Model Silero (JIT) sẵn sàng", extra={"module_tag": "VAD"})

    def _resolve_model_path(self, explicit_path: Optional[Union[str, Path]]) -> Path:
        if explicit_path and Path(explicit_path).exists():
            return Path(explicit_path)
        return MODELS_DIR / "silero_vad.jit"

    def _ensure_model_file(self) -> None:
        """Đảm bảo file silero_vad.jit tồn tại."""
        if not self.model_path.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                src = str(impresources.files("silero_vad.data").joinpath("silero_vad.jit"))
                shutil.copy(src, self.model_path)
            except Exception as e:
                logger.warning(f"Không thể copy bundled silero_vad.jit: {e}. Đang tải từ GitHub...", extra={"module_tag": "VAD"})
                torch.hub.download_url_to_file(
                    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit",
                    str(self.model_path),
                )

    def create_initial_state(self, threshold: Optional[float] = None) -> VADStreamState:
        from silero_vad.utils_vad import init_jit_model
        from silero_vad import VADIterator

        cfg = config.vad.silero
        active_thresh = threshold if threshold is not None else (cfg.threshold or config.vad.threshold)

        state = VADStreamState()
        state.silero_model = init_jit_model(str(self.model_path))
        state.silero_probe = SileroModelProbe(state.silero_model)
        state.silero_iterator = VADIterator(
            model=state.silero_probe,
            threshold=active_thresh,
            sampling_rate=16000,
            min_silence_duration_ms=cfg.min_silence_duration_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        )
        return state

    def is_speech(
        self,
        chunk_float32: Optional[np.ndarray],
        state: VADStreamState,
        threshold: float,
        chunk_raw: Optional[bytes] = None,
    ) -> VADResult:
        if state.silero_iterator is None or state.silero_model is None:
            init_s = self.create_initial_state(threshold)
            state.silero_model = init_s.silero_model
            state.silero_probe = init_s.silero_probe
            state.silero_iterator = init_s.silero_iterator

        if chunk_float32 is None:
            if chunk_raw is not None:
                chunk_float32 = np.frombuffer(chunk_raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                raise ValueError("Cần cung cấp ít nhất chunk_raw hoặc chunk_float32 cho Silero VAD")

        # Căn chỉnh kích thước frame 512 samples
        if len(chunk_float32) != self.native_frame_samples:
            if len(chunk_float32) < self.native_frame_samples:
                chunk_float32 = np.pad(chunk_float32, (0, self.native_frame_samples - len(chunk_float32)))
            else:
                chunk_float32 = chunk_float32[: self.native_frame_samples]

        tensor_chunk = torch.from_numpy(chunk_float32).float()

        if threshold is not None and state.silero_iterator.threshold != threshold:
            state.silero_iterator.threshold = float(threshold)

        with torch.no_grad():
            res = state.silero_iterator(tensor_chunk)

        event = None
        if res:
            if "start" in res:
                event = "START"
            elif "end" in res:
                event = "END"

        is_speech_active = bool(state.silero_iterator.triggered)
        prob = state.silero_probe.last_prob if state.silero_probe is not None else (0.9 if is_speech_active else 0.1)

        return VADResult(is_speech=is_speech_active, probability=prob, event=event)
