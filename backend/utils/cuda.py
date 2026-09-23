"""Thiết lập đường dẫn nạp DLL cho CUDA/cuBLAS/cuDNN và Vulkan trên môi trường Windows."""

import os
import sys
import threading
from typing import Optional, Set

from backend.utils.logger import logger

#: Env var mà `llama_cpp/llama_cpp.py` đọc để chọn thư mục chứa `llama.dll`.
#: Nhờ nó, binding (wheel CPU 7 MB từ PyPI) tách rời khỏi DLL CUDA (nằm trong repo).
_LLAMA_LIB_ENV = "LLAMA_CPP_LIB_PATH"


def llama_cpp_lib_dir() -> Optional[str]:
    """Thư mục DLL llama.cpp đi kèm repo (`backend/bin/llama/`), hoặc None nếu chưa có.

    ⚠️ PHẢI là thư mục RIÊNG, KHÔNG dùng chung với bundle ASR (`backend/bin/`): cả hai đều
    chứa `ggml.dll` / `ggml-base.dll` / `ggml-cpu.dll` / `ggml-cuda.dll` nhưng là **hai bản
    ggml khác nhau**. Windows phân giải DLL phụ thuộc theo TÊN MODULE, nên để chung một thư
    mục thì `transcribe.dll` sẽ bind nhầm `ggml-base.dll` của llama.cpp và cả tiến trình chết
    với `0xc0000139` (STATUS_ENTRYPOINT_NOT_FOUND) — xem cảnh báo trong `backend/tests/conftest.py`.
    """
    candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin", "llama"))
    return candidate if os.path.isfile(os.path.join(candidate, "llama.dll")) else None


def setup_llama_cpp_dll_path() -> Optional[str]:
    """Ép `llama_cpp` nạp `llama.dll` từ `backend/bin/llama/` thay vì `site-packages/llama_cpp/lib`.

    VÌ SAO: wheel Windows của `llama-cpp-python` trên **PyPI chỉ có sdist** (phải tự build, cần
    MSVC + nvcc ~10 phút và vẫn ra bản CPU-only nếu thiếu `CMAKE_ARGS`), còn wheel dựng sẵn của
    abetlen thì chỉ có kernel tới `sm_90` (GPU Blackwell phải chạy qua PTX JIT, chậm hơn ~43%)
    và nhiều bản còn dính AVX-512. Cách nay: cài **binding CPU 7 MB** từ index `whl/cpu`
    (binding chỉ là lớp ctypes, không chứa native code) rồi để DLL CUDA đi kèm repo.

    Không đặt gì nếu `backend/bin/llama/` chưa có ⇒ rơi về hành vi cũ (DLL trong site-packages).

    Gọi TRƯỚC `from llama_cpp import Llama`.
    """
    lib_dir = llama_cpp_lib_dir()
    if not lib_dir:
        return None
    if _LLAMA_LIB_ENV not in os.environ:
        os.environ[_LLAMA_LIB_ENV] = lib_dir
        logger.debug(
            f"llama.cpp DLL: dùng {lib_dir} (qua {_LLAMA_LIB_ENV})", extra={"module_tag": "CORE"}
        )
    if sys.platform == "win32":
        try:
            os.add_dll_directory(lib_dir)
        except Exception:  # noqa: BLE001
            pass
    return lib_dir


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

        # BƯỚC 0 (làm TRƯỚC khi quét): chọn thư mục DLL llama.cpp ⇒ đặt `LLAMA_CPP_LIB_PATH`.
        # Phải chạy trước vòng quét bên dưới để nó biết mà bỏ qua `site-packages/llama_cpp/lib`
        # (bản ggml **CPU** của wheel). Gộp vào đây để MỌI caller cũ chỉ cần gọi
        # `setup_cuda_dll_paths()` là đủ — không phải nhớ thêm một hàm nữa.
        llama_repo_dir = setup_llama_cpp_dll_path()

        if sys.platform != "win32":
            _cuda_paths_initialized = True
            return

        dll_dirs: Set[str] = set()

        # 0. Thư mục native cục bộ của backend: `<repo>/backend/bin/`.
        # Đây là chỗ GOM DLL của dự án: bundle ASR (`transcribe.dll` + `ggml*.dll`, gồm
        # `ggml-cuda.dll`) nằm sẵn ở đây, và thư mục này được đăng ký cho CẢ ASR lẫn engine
        # dịch (llama.cpp). Muốn chạy độc lập với `torch/lib` thì chỉ cần chép thêm
        # `cudart64_13.dll` / `cublas64_13.dll` / `cublasLt64_13.dll` vào cùng thư mục này.
        try:
            backend_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin"))
            if os.path.isdir(backend_bin):
                dll_dirs.add(backend_bin)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Bỏ qua backend/bin cho CUDA DLLs: {e}", extra={"module_tag": "CORE"})

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

                # `backend/bin/llama/` là kho DLL llama.cpp (engine dịch).
                llama_repo_dir = llama_cpp_lib_dir()
                if llama_repo_dir:
                    dll_dirs.add(llama_repo_dir)

                # ⚠️ Chỉ đăng ký `llama_cpp/lib` khi CHƯA trỏ DLL sang `backend/bin/llama/`:
                # thư mục đó chứa bản ggml **CPU** của wheel, đăng ký nó cùng lúc có thể khiến
                # `llama.dll` (bản CUDA của ta) bind nhầm `ggml.dll` CPU.
                llama_lib = os.path.join(sp, "llama_cpp", "lib")
                if os.path.isdir(llama_lib) and not os.environ.get(_LLAMA_LIB_ENV):
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
        # VÌ SAO CẦN: bundle ASR CUDA trong `backend/bin/` (`ggml-cuda.dll`) link với
        # `cudart64_13.dll` / `cublas64_13.dll` / `cublasLt64_13.dll`. Trước đây các DLL này
        # đến từ `torch/lib` (torch bản CUDA), nhưng môi trường có thể dùng torch CPU-only
        # (đã gặp: torch 2.12.0+cu130 -> 2.9.1 CPU) ⇒ CUDA backend **không nạp được** dù
        # `backend/bin/ggml-cuda.dll` còn nguyên. Toolkit đã có sẵn trong repo (pip wheel dùng
        # để build) nên đăng ký luôn ⇒ bundle CUDA tự đủ, không phụ thuộc bản torch nào.
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
