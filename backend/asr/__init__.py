"""Package ASR (Speech-to-Text) kết nối transcribe.cpp cho Backend."""

import os
import sys

# ⚠️ CẢNH BÁO DLL: `bootstrap()` dưới đây nạp `transcribe.dll`, kéo theo `backend/bin/ggml*.dll`
# vào TIẾN TRÌNH NÀY. Windows phân giải DLL theo TÊN MODULE, nên bất kỳ thư viện nào khác cũng
# mang `ggml.dll`/`ggml-base.dll` (llama.cpp, CrispASR) sẽ bind nhầm nếu nạp chung tiến trình.
# Đó là lý do phần CrispASR nằm ở `backend/utils/crispasr_native.py` và chạy trong TIẾN TRÌNH CON
# — xem docstring đầu file đó.

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
