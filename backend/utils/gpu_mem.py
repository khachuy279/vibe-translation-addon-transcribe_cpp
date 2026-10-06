"""Đo VRAM GPU KHÔNG cần PyTorch — gọi thẳng NVML (`nvml.dll` / `libnvidia-ml.so.1`) qua ctypes.

VÌ SAO KHÔNG DÙNG `torch.cuda.mem_get_info()`:
- Buộc import torch (~1-2 s, vài trăm MB RAM) chỉ để đọc một con số.
- Lần gọi đầu tạo **CUDA context** trong tiến trình chính (~300-500 MB VRAM) — đúng thứ ta
  đang muốn tiết kiệm cho TTS/ASR native.
- Chỉ thấy VRAM theo góc nhìn CUDA runtime của torch; NVML đọc từ driver nên thấy cả VRAM do
  các tiến trình con native (TTS worker omnivoice.cpp, llama.cpp, transcribe.dll) chiếm.

NVML đi kèm driver NVIDIA (Windows: `C:\\Windows\\System32\\nvml.dll`), không cần cài thêm gói
Python nào. Máy không có GPU NVIDIA ⇒ mọi hàm trả `None` (caller coi như "không đo được").
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from typing import Optional, Tuple

from backend.utils.logger import get_logger

logger = get_logger("utils.gpu_mem")

_NVML_SUCCESS = 0


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


_lock = threading.Lock()
_nvml: Optional[ctypes.CDLL] = None
_init_failed = False


def _load_nvml() -> Optional[ctypes.CDLL]:
    """Nạp + `nvmlInit_v2` đúng MỘT lần (thread-safe). Lỗi ⇒ nhớ lại, không thử lại liên tục."""
    global _nvml, _init_failed
    if _nvml is not None or _init_failed:
        return _nvml
    with _lock:
        if _nvml is not None or _init_failed:
            return _nvml
        names = ["nvml.dll"] if sys.platform == "win32" else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        if sys.platform == "win32":
            pf = os.environ.get("ProgramFiles", r"C:\Program Files")
            names.append(os.path.join(pf, "NVIDIA Corporation", "NVSMI", "nvml.dll"))
        lib = None
        for name in names:
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if lib is None:
            _init_failed = True
            logger.debug("Không tìm thấy NVML — bỏ qua đo VRAM.", extra={"module_tag": "CORE"})
            return None
        try:
            if lib.nvmlInit_v2() != _NVML_SUCCESS:
                raise RuntimeError("nvmlInit_v2 thất bại")
        except Exception as exc:  # noqa: BLE001
            _init_failed = True
            logger.debug(f"NVML init lỗi: {exc}", extra={"module_tag": "CORE"})
            return None
        _nvml = lib
        return _nvml


def _device_index() -> int:
    """Chỉ số GPU vật lý: lấy phần tử đầu của `CUDA_VISIBLE_DEVICES` nếu là số, mặc định 0."""
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    return int(visible) if visible.isdigit() else 0


def vram_info_mb(index: Optional[int] = None) -> Optional[Tuple[float, float]]:
    """`(free_mb, total_mb)` của GPU `index` (mặc định theo `CUDA_VISIBLE_DEVICES`), hoặc None."""
    lib = _load_nvml()
    if lib is None:
        return None
    try:
        handle = ctypes.c_void_p()
        idx = _device_index() if index is None else int(index)
        if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(idx), ctypes.byref(handle)) != _NVML_SUCCESS:
            return None
        mem = _NvmlMemory()
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != _NVML_SUCCESS:
            return None
        mb = 1024.0 * 1024.0
        return float(mem.free) / mb, float(mem.total) / mb
    except Exception:  # noqa: BLE001
        return None


def free_vram_mb(index: Optional[int] = None) -> Optional[float]:
    """VRAM trống (MB), hoặc None nếu không đo được (không có GPU NVIDIA / thiếu driver)."""
    info = vram_info_mb(index)
    return info[0] if info else None
