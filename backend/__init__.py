"""Bilingual Subtitle Backend Package."""

import os

# ── A1-1: giới hạn thread CPU — phải nằm Ở ĐÂY, không phải trong `main.py` ──────────
# `backend/__init__.py` chạy TRƯỚC mọi submodule của MỌI đường vào: app thật, harness
# (`backend/tests/test_09_wer_ab.py`), benchmark, `pytest`. Trước đây guard chỉ có trong
# `main.py` nên mọi đường khác chạy bằng mặc định của OpenMP/OpenBLAS = số nhân logic (12)
# + chế độ busy-spin.
#
# Đo được trên máy 12 luồng khi chạy harness WER: 11 thread `python/34..44` ở ~85% CPU mỗi
# thread, GIỮ LIÊN TỤC kể cả khi `nvidia-smi` báo GPU util = 0%. Tức CPU bị đốt bởi vòng
# QUAY CHỜ VIỆC của thread pool, không phải tính toán thật. Chỉ cần VAD FireRed chạy
# 40 frame/s là đủ giữ CPU 100%.
_DEFAULT_CPU_THREADS = "2"
_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": _DEFAULT_CPU_THREADS,
    # ⚠️ OpenBLAS đọc biến NÀY TRƯỚC `OMP_NUM_THREADS` ⇒ thiếu nó là KHÔNG chặn được numpy
    # (numpy ở đây build trên `scipy-openblas`). Đây chính là lỗ hổng của bản cũ.
    "OPENBLAS_NUM_THREADS": _DEFAULT_CPU_THREADS,
    "MKL_NUM_THREADS": _DEFAULT_CPU_THREADS,
    "NUMEXPR_NUM_THREADS": _DEFAULT_CPU_THREADS,
    "VECLIB_MAXIMUM_THREADS": _DEFAULT_CPU_THREADS,
    # Tắt busy-spin: luồng NGỦ khi hết việc thay vì quay nóng chờ việc mới.
    # Thiếu hai dòng này thì việc giới hạn số luồng cũng không cứu được CPU.
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}


def apply_thread_limits() -> None:
    """Đặt giới hạn thread CPU. Idempotent; KHÔNG ghi đè biến môi trường người dùng đã đặt.

    Dùng `setdefault` (không gán thẳng) để ai muốn ép số luồng khác vẫn làm được:
    `$env:OPENBLAS_NUM_THREADS=8; python backend\\main.py`.
    """
    for key, value in _THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


apply_thread_limits()

# ── DLL llama.cpp: chọn thư mục DLL NGAY khi import package ─────────────────────────
# `llama_cpp/llama_cpp.py` đọc env `LLAMA_CPP_LIB_PATH` NGAY LÚC IMPORT để quyết định nạp
# `llama.dll` từ đâu. Binding ta cài là wheel **CPU ~7 MB** (xem `backend/requirements.txt`);
# nếu env chưa được đặt thì nó nạp bản CPU trong `site-packages/llama_cpp/lib` — và mọi thứ
# vẫn "chạy", chỉ chậm **~10×**, còn log thì vẫn ghi "(GPU, …)". Đo được: dịch EN→VI 589 ms
# (đúng) so với 6027 ms (sai thứ tự import).
#
# Đặt Ở ĐÂY, cùng lý do như `apply_thread_limits` ở trên: `backend/__init__.py` chạy TRƯỚC mọi
# submodule của MỌI đường vào, nên KHÔNG phụ thuộc thứ tự `import`. `setdefault` để người dùng
# vẫn ép được thư mục khác. `backend/utils/cuda.py::setup_llama_cpp_dll_path` vẫn đặt lại
# (idempotent) cho đường vào không qua package — ví dụ script gọi thẳng `import llama_cpp`.
_LLAMA_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "llama")
if os.path.isfile(os.path.join(_LLAMA_LIB_DIR, "llama.dll")):
    os.environ.setdefault("LLAMA_CPP_LIB_PATH", _LLAMA_LIB_DIR)

from backend.config import config, load_config
from backend.utils.logger import logger, get_logger

__version__ = "2.0.0"
__all__ = ["config", "load_config", "logger", "get_logger", "apply_thread_limits"]
