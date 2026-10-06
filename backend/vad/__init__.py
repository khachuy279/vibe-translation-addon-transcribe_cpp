"""Package VAD (Voice Activity Detection) cho Backend.

Cả hai engine (`firered-vad` mặc định, `silero-vad`) chạy bằng **onnxruntime** — không cần
PyTorch. Engine `fsmn-vad` đã bị xoá ở Giai đoạn 3, xem `backend/vad/base.py`.
"""

from backend.vad.base import BaseVADEngine, VADResult, VADStreamState
from backend.vad.processor import VADStreamProcessor, VADProcessor
from backend.vad.engines import (
    FireRedVADEngine,
    SileroVADEngine,
    VADEngineFactory,
    SUPPORTED_VAD_ENGINES,
)

__all__ = [
    "BaseVADEngine",
    "VADResult",
    "VADStreamState",
    "VADStreamProcessor",
    "VADProcessor",
    "FireRedVADEngine",
    "SileroVADEngine",
    "VADEngineFactory",
    "SUPPORTED_VAD_ENGINES",
]

