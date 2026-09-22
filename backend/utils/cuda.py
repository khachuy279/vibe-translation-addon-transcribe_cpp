"""Thiết lập đường dẫn nạp DLL cho CUDA/cuBLAS/cuDNN và Vulkan trên môi trường Windows."""

import os
import sys
import threading
from typing import Set

from backend.utils.logger import logger

def _configure_alloc_conf() -> None:
    """Đặt biến cấu hình allocator của torch ĐÚNG TÊN theo phiên bản.

    Torch đổi tên `PYTORCH_CUDA_ALLOC_CONF` → `PYTORCH_ALLOC_CONF` và cảnh báo deprecation
    khi thấy biến CŨ còn được đặt:
        `[W...] Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF instead`
    (ĐO THỰC: cảnh báo này xuất hiện với **torch 2.9.1** ⇒ mốc đổi tên là 2.9, KHÔNG phải 2.12.)

    Cách chọn: torch ≥ 2.9 → CHỈ đặt tên mới (torch cũ hơn không biết biến này nên sẽ bỏ qua);
    torch < 2.9 → đặt tên cũ. Dùng `importlib.metadata` để biết phiên bản mà KHÔNG import torch.
    """
    name = "PYTORCH_CUDA_ALLOC_CONF"
    try:
        from importlib.metadata import version as _pkg_version

        parts = _pkg_version("torch").split("+")[0].split(".")
        if (int(parts[0]), int(parts[1])) >= (2, 9):
            name = "PYTORCH_ALLOC_CONF"
    except Exception:  # noqa: BLE001 - thiếu torch/chuỗi lạ → giữ tên cũ
        pass
    os.environ.setdefault(name, "expandable_segments:True")


_configure_alloc_conf()

_cuda_paths_initialized: bool = False
_init_lock = threading.RLock()


def setup_cuda_dll_paths() -> None:
    """Đăng ký đường dẫn chứa CUDA/Vulkan DLLs cho Windows dynamic loader."""
    global _cuda_paths_initialized
    if _cuda_paths_initialized:
        return

    with _init_lock:
        if _cuda_paths_initialized:
            return

        if sys.platform != "win32":
            _cuda_paths_initialized = True
            return

        dll_dirs: Set[str] = set()

        # 1. Tìm trong site-packages (torch, llama_cpp, nvidia)
        try:
            import site
            site_packages_dirs = [p for p in (site.getsitepackages() + [site.USER_SITE]) if p]
            for sp in site_packages_dirs:
                if not os.path.exists(sp):
                    continue

                torch_lib = os.path.join(sp, "torch", "lib")
                if os.path.isdir(torch_lib):
                    dll_dirs.add(os.path.abspath(torch_lib))

                llama_lib = os.path.join(sp, "llama_cpp", "lib")
                if os.path.isdir(llama_lib):
                    dll_dirs.add(os.path.abspath(llama_lib))

                nvidia_dir = os.path.join(sp, "nvidia")
                if os.path.isdir(nvidia_dir):
                    for item in os.listdir(nvidia_dir):
                        if item.lower() == "cudnn":
                            continue
                        sub_bin = os.path.join(nvidia_dir, item, "bin")
                        if os.path.isdir(sub_bin):
                            dll_dirs.add(os.path.abspath(sub_bin))
                        sub_lib = os.path.join(nvidia_dir, item, "lib")
                        if os.path.isdir(sub_lib):
                            dll_dirs.add(os.path.abspath(sub_lib))
        except Exception as e:
            logger.debug(f"Lỗi khi quét site-packages cho CUDA DLLs: {e}", extra={"module_tag": "CORE"})

        # 2. Tìm trong biến môi trường CUDA_PATH
        for env_key, env_val in os.environ.items():
            if env_key.startswith("CUDA_PATH") and env_val and os.path.exists(env_val):
                bin_dir = os.path.join(env_val, "bin")
                if os.path.isdir(bin_dir):
                    dll_dirs.add(os.path.abspath(bin_dir))

        # 2b. CUDA toolkit 13 đi kèm repo (`external/cuda-toolkit` hoặc `.cuda-toolkit`).
        # VÌ SAO CẦN: bundle ASR CUDA trong `bin/` (`ggml-cuda.dll`) link với
        # `cudart64_13.dll` / `cublas64_13.dll` / `cublasLt64_13.dll`. Trước đây các DLL này
        # đến từ `torch/lib` (torch bản CUDA), nhưng môi trường có thể dùng torch CPU-only
        # (đã gặp: torch 2.12.0+cu130 -> 2.9.1 CPU) ⇒ CUDA backend **không nạp được** dù
        # `bin/ggml-cuda.dll` còn nguyên. Toolkit đã có sẵn trong repo (pip wheel dùng để
        # build) nên đăng ký luôn ⇒ bundle CUDA tự đủ, không phụ thuộc bản torch nào.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        for toolkit_root in (
            os.path.join(project_root, "external", "cuda-toolkit"),
            os.path.join(project_root, ".cuda-toolkit"),
        ):
            candidates = [
                os.path.join(toolkit_root, "nvidia", "cu13", "bin", "x86_64"),
                os.path.join(toolkit_root, "nvidia", "cu13", "bin"),
            ]
            # Glob nhẹ nhàng: chấp nhận mọi phiên bản cu* nếu sau này nâng toolkit.
            nvidia_root = os.path.join(toolkit_root, "nvidia")
            if os.path.isdir(nvidia_root):
                try:
                    for item in os.listdir(nvidia_root):
                        candidates.append(os.path.join(nvidia_root, item, "bin", "x86_64"))
                        candidates.append(os.path.join(nvidia_root, item, "bin"))
                except OSError:
                    pass
            for cand in candidates:
                if os.path.isdir(cand) and any(
                    f.startswith("cudart64") for f in os.listdir(cand)
                ):
                    dll_dirs.add(os.path.abspath(cand))

        # 3. Tìm trong Program Files NVIDIA GPU Computing Toolkit
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        cuda_toolkit_base = os.path.join(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        if os.path.isdir(cuda_toolkit_base):
            try:
                for ver_dir in os.listdir(cuda_toolkit_base):
                    bin_dir = os.path.join(cuda_toolkit_base, ver_dir, "bin")
                    if os.path.isdir(bin_dir):
                        dll_dirs.add(os.path.abspath(bin_dir))
            except Exception:
                pass

        # Đăng ký với Windows loader
        path_env = os.environ.get("PATH", "")
        path_list = path_env.split(os.path.pathsep)

        for d in dll_dirs:
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

            if d not in path_list:
                path_list.insert(0, d)

        os.environ["PATH"] = os.path.pathsep.join(path_list)
        _cuda_paths_initialized = True
        logger.debug(f"Đã đăng ký các thư mục CUDA DLL: {list(dll_dirs)}", extra={"module_tag": "CORE"})
