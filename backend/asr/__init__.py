"""Package ASR (Speech-to-Text) kết nối transcribe.cpp cho Backend."""

import os
import sys

# Đảm bảo thư mục backend/asr nằm trong sys.path để vendor package qwen_asr có thể import trực tiếp
_ASR_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_ASR_DIR, "qwen_asr")) and _ASR_DIR not in sys.path:
    sys.path.insert(0, _ASR_DIR)

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
from backend.asr import lifecycle
from backend.asr import lifecycle as hotswap  # Alias tương thích ngược

__all__ = [
    "lifecycle",
    "hotswap",
    "BaseASREngine",
    "ModelRegistry",
    "build_family_options",
    "normalize_language_for_family",
    "clean_transcript_text",
    "TranscribeEngine",
    "native",
]
