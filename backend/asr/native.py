"""Chọn & nạp thư viện native transcribe.cpp, và quyết định backend ASR.

Tách riêng khỏi `engine.py` vì hai lý do về **THỨ TỰ NẠP**:

1. `import transcribe_cpp` sẽ `dlopen` `transcribe.dll` NGAY LẬP TỨC, và ggml nạp các
   module backend (`ggml-cuda.dll`, `ggml-vulkan.dll`) ngay sau đó. Mọi thư mục chứa DLL
   phụ thuộc (cudart/cublas của torch) phải được đăng ký **TRƯỚC** — xem
   `backend/utils/cuda.py::setup_cuda_dll_paths()`.
2. `backend/asr/adapters.py` cũng import `transcribe_cpp`, nên việc chuẩn bị phải chạy
   trước cả module đó ⇒ hàm `bootstrap()` được gọi ở dòng đầu `backend/asr/__init__.py`.

## Bundle native cục bộ (`backend/bin/`)

Bản dựng CUDA cho Windows **chưa được phát hành** trên PyPI: gói `transcribe-cpp-native-cu12`
chỉ là name-reservation (wheel 1380 byte, không có native code) — xem
`report/audit/KE_HOACH_FIX_LOI_Hy3.md` §4.1.1. Vì vậy dự án tự dựng lấy và đặt bundle tại
`<project_root>/backend/bin/`:

    backend/bin/transcribe.dll      libtranscribe (bản dựng cục bộ)
    backend/bin/ggml.dll            ggml dispatcher
    backend/bin/ggml-base.dll       ggml core
    backend/bin/ggml-cpu.dll        backend CPU
    backend/bin/ggml-cuda.dll       backend CUDA   (nếu có)
    backend/bin/ggml-vulkan.dll     backend Vulkan (nếu có)

Bundle nằm trong `backend/` (không phải thư mục gốc repo) để **toàn bộ tài nguyên native của
backend nằm chung một cây** — `backend/` trở thành đơn vị duy nhất cần sao chép khi triển khai,
và DLL của ASR không còn lẫn với các thư mục build ở gốc repo. Vị trí CŨ `<project_root>/bin/`
vẫn được chấp nhận như **fallback** (xem `_resolve_bundle_dir`) để bản cài cũ không mất CUDA.

Khi bundle tồn tại, ta đặt `TRANSCRIBE_LIBRARY` trỏ vào `transcribe.dll` **trước khi**
import `transcribe_cpp` ⇒ binding đi theo "dev-tree / explicit path" và nạp đúng bundle này,
thay vì provider `transcribe-cpp-native` đã cài trong site-packages.

Nếu bundle KHÔNG tồn tại (ví dụ bản cài cho người dùng cuối chỉ có wheel), ta **không** đặt
gì cả và binding tự tìm provider đã cài — hành vi cũ được giữ nguyên.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

from backend.utils.logger import logger

#: Tag logging chuẩn (xem QUY ƯỚC LOGGING trong backend/utils/logger.py).
#: Dùng hằng chuỗi cấp module — cùng kiểu với `hotswap.py` (`_TAG = "TRANSLATE"`) — để
#: `test_20_logging_convention.py` phân tích AST và xác nhận được `module_tag`.
_TAG = "ASR"

#: Env var mà Python binding dùng làm "developer escape hatch" (xem
#: `bindings/python/src/transcribe_cpp/_library.py`).
_LIBRARY_ENV = "TRANSCRIBE_LIBRARY"

_bootstrap_lock = threading.RLock()
_bootstrapped = False
_bundle_dir: Optional[Path] = None
_bundle_source: str = "unset"

# Cache kết quả phân giải backend: khoá = giá trị config đã chuẩn hoá.
_resolve_lock = threading.RLock()
_resolved: dict[str, str] = {}


def _project_root() -> Path:
    # backend/asr/native.py -> backend/asr -> backend -> <root>
    return Path(__file__).resolve().parent.parent.parent


def _backend_dir() -> Path:
    # backend/asr/native.py -> backend/asr -> backend
    return Path(__file__).resolve().parent.parent


def native_bundle_dir() -> Optional[Path]:
    """Thư mục bundle native đang dùng, hoặc None nếu dùng provider đã cài."""
    return _bundle_dir


def native_bundle_source() -> str:
    """Nguồn bundle: 'config' | 'default' | 'env' | 'installed'."""
    return _bundle_source


def _resolve_bundle_dir() -> Tuple[Optional[Path], str]:
    """Tìm thư mục bundle native theo thứ tự ưu tiên."""
    from backend.config import config

    # 1. Biến môi trường đã đặt sẵn (người dùng/dev chủ động) — tôn trọng tuyệt đối.
    env_path = os.environ.get(_LIBRARY_ENV)
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p.parent, "env"
        if (p / "transcribe.dll").is_file():
            return p, "env"

    if not getattr(config.asr, "use_local_native", True):
        return None, "installed"

    # 2. Đường dẫn cấu hình; 3. mặc định `backend/bin`; 4. vị trí cũ `<root>/bin`.
    configured = (getattr(config.asr, "native_dir", "") or "").strip()
    candidates = []
    if configured:
        candidates.append((Path(configured), "config"))
    candidates.append((_backend_dir() / "bin", "default"))
    # Vị trí CŨ trước khi bundle được gom vào `backend/bin`. Giữ làm fallback để bản cài đã có
    # sẵn `<root>/bin` không mất CUDA sau khi nâng cấp. Vẫn báo nguồn là "default" — tập giá
    # trị của `native_bundle_source()` giữ nguyên: config | default | env | installed.
    candidates.append((_project_root() / "bin", "default"))

    for cand, source in candidates:
        if (cand / "transcribe.dll").is_file():
            return cand, source
    return None, "installed"


def bootstrap() -> None:
    """Chuẩn bị môi trường nạp native. Idempotent; phải gọi TRƯỚC `import transcribe_cpp`."""
    global _bootstrapped, _bundle_dir, _bundle_source
    if _bootstrapped:
        return
    with _bootstrap_lock:
        if _bootstrapped:
            return

        # Nếu `transcribe_cpp` đã nằm trong sys.modules thì native ĐÃ được dlopen từ nguồn
        # khác, và mọi thứ ta đặt dưới đây sẽ KHÔNG có tác dụng (DLL đã nạp vào tiến trình).
        # Ghi cảnh báo rõ ràng thay vì log "đang dùng backend/bin" gây hiểu sai — đây là lỗi
        # thứ tự nạp, thường do import `transcribe_cpp` trực tiếp trước `backend.asr`.
        too_late = "transcribe_cpp" in sys.modules
        if too_late:
            logger.warning(
                "ASR native: `transcribe_cpp` đã được import TRƯỚC khi bootstrap chạy — "
                "bundle trong backend/bin KHÔNG được áp dụng (native đã dlopen từ nguồn khác). "
                "Hãy import `backend.asr` (hoặc để backend/asr/__init__.py chạy) trước.",
                extra={"module_tag": _TAG},
            )

        # DLL phụ thuộc (cudart/cublas của torch) phải đăng ký trước khi dlopen native.
        try:
            from backend.utils.cuda import setup_cuda_dll_paths

            setup_cuda_dll_paths()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Bỏ qua setup_cuda_dll_paths: {exc}", extra={"module_tag": _TAG})

        bundle, source = _resolve_bundle_dir()
        if bundle is not None:
            lib = bundle / "transcribe.dll"
            os.environ[_LIBRARY_ENV] = str(lib)
            # Bundle nằm ngoài PATH mặc định ⇒ thêm vào để loader tìm được ggml*.dll.
            try:
                os.add_dll_directory(str(bundle))
            except Exception:  # noqa: BLE001
                pass
            path_env = os.environ.get("PATH", "")
            if str(bundle) not in path_env.split(os.pathsep):
                os.environ["PATH"] = os.pathsep.join([str(bundle), path_env])
            _bundle_dir, _bundle_source = bundle, source
            if not too_late:
                logger.info(
                    f"Native bundle: {bundle.name}/ (nguồn: {source})",
                    extra={"module_tag": _TAG},
                )
        else:
            _bundle_dir, _bundle_source = None, "installed"
            if not too_late:
                logger.info(
                    "Native bundle: không tìm thấy trong backend/bin (dùng provider cài đặt)",
                    extra={"module_tag": _TAG},
                )

        _bootstrapped = True


# --------------------------------------------------------------------------- backend

#: Thứ tự ưu tiên khi backend yêu cầu không khả dụng.
#:
#: `auto` = "tốt nhất đang có": **CUDA trước** (đo được nhanh hơn Vulkan ~1,53×), rồi **fallback về
#: Vulkan**. Vulkan được giữ làm **đích fallback** (và vẫn là lựa chọn hợp lệ) vì đó là đường
#: `transcribe.cpp` **hỗ trợ chính thức** qua wheel trên PyPI ⇒ **chắc chắn chạy trên mọi máy**;
#: bản CUDA là do dự án **tự build** nên không đảm bảo có mặt ở mọi cấu hình — khi thiếu,
#: `resolve_backend()` tự chuyển về Vulkan kèm log WARNING.
#:
#:  * "auto"   -> cuda trước, rồi vulkan.
#:  * "cuda"   -> cuda trước; KHÔNG có thì **fallback về Vulkan**.
#:  * "vulkan" -> Vulkan trước (chỉ định rõ thì không tự chuyển sang CUDA trừ khi Vulkan không có).
#:
#: KHÔNG có "cpu" trong danh sách. Backend CPU vẫn tồn tại trong thư viện native
#: (`backend/bin/ggml-cpu.dll`) và **phải giữ lại** vì ggml dùng nó làm backend mặc định cho
#: các op không offload được lên GPU (`ggml_backend_sched` cần một CPU backend cho graph) —
#: nhưng nó KHÔNG phải lựa chọn hợp lệ cho ASR.
#:
#: Đo thật (external/build-tmp/probe_cpu_backend.py, qwen3-asr-0.6b, clip 5,08 s):
#:   CPU    : 8 100 ms  -> RTF 1,60  (CHẬM HƠN thời gian thực)
#:   Vulkan :   156 ms  -> RTF 0,031
#:   CUDA   :   100 ms  -> RTF 0,020
#: CPU chạy *đúng* nhưng ở RTF 1,6 thì phụ đề luôn trễ dần và không bao giờ đuổi kịp video
#: => không dùng làm lựa chọn, cũng không dùng làm đích fallback.
_PREFERENCE: dict[str, Tuple[str, ...]] = {
    "auto": ("cuda", "vulkan"),
    "cuda": ("cuda", "vulkan"),
    "vulkan": ("vulkan", "cuda"),
}

_ALIASES = {"": "auto", "none": "auto", "default": "auto", "gpu": "auto"}


#: Các backend kind ĐƯỢC PHÉP chọn cho ASR. KHÔNG gồm "cpu" — xem `_PREFERENCE`.
_SELECTABLE = ("cuda", "vulkan")


def _available_kinds() -> set[str]:
    """Các backend kind mà thư viện native đang nạp thực sự hỗ trợ VÀ được phép chọn.

    Chỉ xét `_SELECTABLE` (cuda, vulkan). `cpu` luôn hiện diện trong thư viện native
    nhưng không phải lựa chọn hợp lệ cho ASR nên không được tính ở đây; nó vẫn xuất hiện
    trong `backend_devices()` để phục vụ chẩn đoán.
    """
    try:
        import transcribe_cpp

        if transcribe_cpp is None:
            return set()
        out = set()
        for kind in _SELECTABLE:
            try:
                if transcribe_cpp.backend_available(kind):
                    out.add(kind)
            except Exception:  # noqa: BLE001
                pass
        return out
    except Exception:  # noqa: BLE001
        return set()


def backend_devices() -> str:
    """Mô tả ngắn các device backend đang thấy (cho log)."""
    try:
        import transcribe_cpp

        if transcribe_cpp is None:
            return "n/a"
        return ", ".join(
            f"{d.kind}={d.name}" for d in transcribe_cpp.backends()
        ) or "n/a"
    except Exception:  # noqa: BLE001
        return "n/a"


def resolve_backend(requested: Optional[str], *, force_log: bool = False) -> str:
    """Quyết định backend thực tế sẽ truyền cho `transcribe_cpp.Model`.

    Trả về một trong "cuda" / "vulkan". Khi backend yêu cầu không khả dụng, tự
    động fallback theo `_PREFERENCE` và ghi log **WARNING** nêu rõ lý do + các backend
    thực có. Fallback chỉ chạy khi `config.asr.backend_fallback` bật.
    """
    from backend.config import config

    raw = (requested or "auto").strip().lower()
    key = _ALIASES.get(raw, raw)
    if key not in _PREFERENCE:
        logger.warning(
            f"ASR backend '{requested}' không hợp lệ — coi như 'auto'. "
            f"Hợp lệ: {', '.join(sorted(_PREFERENCE))}.",
            extra={"module_tag": _TAG},
        )
        key = "auto"

    with _resolve_lock:
        cached = _resolved.get(key)
    if cached is not None and not force_log:
        return cached

    avail = _available_kinds()
    prefs = _PREFERENCE[key]

    if not getattr(config.asr, "backend_fallback", True):
        chosen = prefs[0]
        _resolved[key] = chosen
        logger.info(
            f"ASR backend: dùng '{chosen}' (fallback TẮT theo cấu hình; "
            f"yêu cầu='{key}', hiện có: {sorted(avail) or 'không rõ'}).",
            extra={"module_tag": _TAG},
        )
        return chosen

    chosen = next((p for p in prefs if p in avail), None)

    if chosen is None:
        # Không backend nào báo khả dụng (native thiếu/hỏng) — trả ứng viên đầu để lỗi
        # native nổi lên rõ ràng thay vì che mất.
        chosen = prefs[0]
        logger.warning(
            f"ASR backend: không backend nào trong {prefs} khả dụng "
            f"(yêu cầu='{key}'). Vẫn thử '{chosen}'; nếu nạp thất bại, xem log native ở trên.",
            extra={"module_tag": _TAG},
        )
    elif key != "auto" and chosen != prefs[0]:
        # Người dùng chỉ định rõ một backend mà nó không có ⇒ đây là FALLBACK, phải WARNING.
        logger.warning(
            f"Backend '{prefs[0]}' không khả dụng -> Fallback '{chosen}' (yêu cầu='{key}', devices: {backend_devices()})",
            extra={"module_tag": _TAG},
        )
    else:
        # 'auto' chọn theo thứ tự ưu tiên — đó là hành vi bình thường, không phải fallback.
        logger.debug(
            f"Backend active: '{chosen}' (yêu cầu='{key}', devices: {backend_devices()})",
            extra={"module_tag": _TAG},
        )

    _resolved[key] = chosen
    return chosen


def reset_backend_cache() -> None:
    """Xoá cache phân giải backend (dùng khi đổi cấu hình lúc chạy / trong test)."""
    with _resolve_lock:
        _resolved.clear()


def runtime_info() -> dict:
    """Thông tin cho `/health`: backend nào thực có, bundle nào đang dùng."""
    return {
        "native_bundle_dir": str(_bundle_dir) if _bundle_dir else None,
        "native_bundle_source": _bundle_source,
        "library_path": os.environ.get(_LIBRARY_ENV),
        "available_backends": sorted(_available_kinds()),
        "devices": backend_devices(),
    }
