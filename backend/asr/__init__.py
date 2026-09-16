"""Package ASR (Speech-to-Text) kết nối transcribe.cpp cho Backend."""

# PHẢI chạy trước mọi import chạm tới `transcribe_cpp`: `backend/asr/adapters.py` và
# `backend/asr/engine.py` đều import nó ở cấp module, và `import transcribe_cpp` sẽ dlopen
# native ngay. Xem docstring `backend/asr/native.py` để biết vì sao thứ tự này quan trọng.
from backend.asr.native import bootstrap as _bootstrap_native

_bootstrap_native()

from backend.asr.base import BaseASREngine
from backend.asr.registry import ModelRegistry
from backend.asr.adapters import build_family_options, normalize_language_for_family
from backend.asr.text_cleaner import clean_transcript_text
from backend.asr.engine import TranscribeEngine
from backend.asr import native

__all__ = [
    "BaseASREngine",
    "ModelRegistry",
    "build_family_options",
    "normalize_language_for_family",
    "clean_transcript_text",
    "TranscribeEngine",
    "native",
]
