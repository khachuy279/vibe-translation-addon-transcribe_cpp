"""Binding native + TIẾN TRÌNH CON cho Qwen3-ForcedAligner-0.6B bản GGUF (CrispASR).

VÌ SAO MODULE NÀY NẰM Ở `backend/utils/` CHỨ KHÔNG Ở `backend/asr/`
=================================================================
Đây KHÔNG phải lựa chọn cho gọn. `backend/asr/__init__.py` gọi `bootstrap()` **ngay khi import**
(`backend/asr/__init__.py:14-16`), mà `bootstrap()` nạp `transcribe.dll` — kéo theo
`backend/bin/ggml.dll` + `ggml-base.dll` + `ggml-cpu.dll` + `ggml-cuda.dll` vào tiến trình.

Hệ quả ĐÃ GẶP THẬT (2026-10-06): nếu tiến trình con import `backend.asr.*` thì bộ `ggml*.dll` của
transcribe.cpp được nạp TRƯỚC; sau đó `ctypes.CDLL("crispasr.dll")` phân giải `ggml-base.dll`
theo **TÊN MODULE** và bind vào bản đã nạp ⇒ lệch ABI ⇒
`OSError: [WinError 127] The specified procedure could not be found`
(đúng loại lỗi `0xc0000139`/`STATUS_ENTRYPOINT_NOT_FOUND` mà `backend/utils/cuda.py` cảnh báo).

Đã kiểm chứng bằng bisect (`.research/bisect_dll.py`): nạp `crispasr.dll` trong tiến trình SẠCH
thành công với mọi tổ hợp thư mục; chỉ hỏng khi tiến trình đã nạp bộ ggml của transcribe.cpp.

⇒ `backend/utils/__init__.py` chỉ import `logger` (an toàn), nên tiến trình con khởi động qua
module này KHÔNG bao giờ chạm tới `backend.asr`.

ĐO THẬT (RTX 5060 Ti, `.research/bench_cuda13.py`)
=================================================
| Build | DLL mang theo | `Chinese_noise_28s` (28,26 s) | RTF |
|---|---|---|---|
| CPU | 56 MB | 3 956 ms | 0,140 |
| Vulkan | 87 MB | ~8 200 ms | 1,62 (không dùng được) |
| CUDA 12 (wheel `+cuda`) | 988 MB | 194 ms | 0,0069 |
| **CUDA 13 non-cuda + toolkit trong repo** | **167 MB** | **195 ms** | **0,0069** |

⇒ Chọn bản CUDA 13 `-non-cuda`: tốc độ Y HỆT bản CUDA 12 nhưng nhỏ hơn 6 lần, vì dùng đúng
`cudart64_13.dll`/`cublas64_13.dll`/`cublasLt64_13.dll` mà repo ĐÃ có trong `external/cuda-toolkit`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import BACKEND_DIR, MODELS_DIR
from backend.utils.logger import logger

_TAG = "ASR"

#: Thư mục DLL CrispASR đi kèm repo (`backend/bin/crispasr/`). PHẢI RIÊNG BIỆT.
_DLL_DIR_NAME = "crispasr"
#: Tên file model GGUF trong `backend/models/`.
MODEL_FILENAME = "Qwen3-ForcedAligner-0.6B-q4_k.gguf"
#: Repo HuggingFace phát hành bản GGUF (đúng nguồn registry của CrispASR).
MODEL_REPO_ID = "cstr/qwen3-forced-aligner-0.6b-GGUF"

#: Override để thử build khác mà không sửa mã.
ENV_LIB_DIR = "CRISPASR_LIB_DIR"
ENV_MODEL = "CRISPASR_ALIGNER_MODEL"
ENV_WORKER_DEBUG = "CRISPASR_WORKER_DEBUG"
#: Số luồng CPU cho nhánh aligner (không ảnh hưởng khi chạy CUDA).
ENV_THREADS = "CRISPASR_ALIGN_THREADS"

DEFAULT_N_THREADS = 8

# ══════════════════════════════════════════════════════════════════════════════════════════
# TẢI DLL CRISPASR KHI THIẾU (GitHub giới hạn 100 MB/file)
# ══════════════════════════════════════════════════════════════════════════════════════════
# `backend/bin/crispasr/` gồm 5 DLL (~167 MB). Trong đó **`ggml-cuda.dll` = 149,63 MB > 100 MB**
# nên GitHub TỪ CHỐI push file đó ở chế độ thường (các DLL cũ trong repo đều < 68 MB nên chưa
# từng gặp). Hai cách xử lý:
#
#   (A) Git LFS — thêm `.gitattributes`:
#           backend/bin/crispasr/ggml-cuda.dll filter=lfs diff=lfs merge=lfs -text
#       rồi `git lfs track` + commit. Tốn ~150 MB quota LFS và mỗi clone phải tải 150 MB.
#
#   (B) KHÔNG commit DLL, TẢI KHI THIẾU — mặc định ở đây. `ensure_lib_files()` tải gói
#       `crispasr-windows-x86_64-cuda13-non-cuda.zip` (157 MB, ĐÃ GHIM v0.8.41) từ GitHub
#       Releases rồi giải nén ĐÚNG 5 file cần thiết, có xác thực SHA-256. Nhất quán với cách
#       repo xử lý model (~500 MB GGUF cũng tự tải), và giữ clone nhẹ.
#
# Chọn bản `-cuda13-non-cuda` vì nó KHÔNG kèm cuBLAS/cudart (đó là nghĩa của "non-cuda"): nó
# dùng đúng runtime CUDA 13 mà repo đã có trong `external/cuda-toolkit`. Bản wheel `+cuda` phải
# mang thêm 800 MB cuBLAS 12 mà tốc độ y hệt (đo: cùng RTF 0,0069).
RELEASE_TAG = "v0.8.41"
BUNDLE_ASSET = "crispasr-windows-x86_64-cuda13-non-cuda.zip"
BUNDLE_URL = (
    f"https://github.com/CrispStrobe/CrispASR/releases/download/{RELEASE_TAG}/{BUNDLE_ASSET}"
)
#: SHA-256 của file zip phát hành (đã tải và kiểm ngày 2026-10-06).
BUNDLE_SHA256 = "91c11b44834e274fa0db87b114a0427102a35797d8c218ac6ab44203bc7e9a45"

#: Các DLL BẮT BUỘC (đủ để chạy aligner trên CUDA 13 + fallback CPU).
REQUIRED_DLLS: Tuple[str, ...] = (
    "crispasr.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-cpu.dll",
    "ggml-cuda.dll",
)
#: SHA-256 của từng DLL, xác thực SAU khi giải nén (bắt cả trường hợp zip đúng nhưng giải nén hỏng).
DLL_SHA256: Dict[str, str] = {
    "crispasr.dll": "53576958552562261630210fa7e956a1defd6be84d3cb533bf13505c2bf5032b",
    "ggml-base.dll": "317ed2595ab76d51a535d755e907bfd4c978a47d5836c8e85914b41c71102b7f",
    "ggml-cpu.dll": "8bd644182eb7a41b4940759c33e7bc584d63c79c7881c51c6b0e6990a6be83e4",
    "ggml-cuda.dll": "2c5310e2189ee0eedbbe6abad48633d04c013593f13b5bae979826aec19cd156",
    "ggml.dll": "1f46a3c3845e711406c7cd35a85fe4ac97199ec40ca81dcd232c64c657a28d3b",
}


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _dlls_present(lib_dir: Path, *, verify: bool = False) -> bool:
    """Đủ 5 DLL chưa (và khớp hash nếu `verify=True`)."""
    for name in REQUIRED_DLLS:
        p = lib_dir / name
        if not p.is_file() or p.stat().st_size == 0:
            return False
    if verify:
        for name in REQUIRED_DLLS:
            expected = DLL_SHA256.get(name)
            if expected and _sha256_file(lib_dir / name) != expected:
                return False
    return True


def ensure_lib_files(
    lib_dir: Optional[Path] = None,
    *,
    allow_download: bool = True,
    progress: bool = True,
) -> Path:
    """Đảm bảo `backend/bin/crispasr/` có đủ 5 DLL; tải từ GitHub Releases nếu thiếu.

    VÌ SAO KHÔNG COMMIT SẴN: `ggml-cuda.dll` (149,63 MB) vượt giới hạn **100 MB/file** của GitHub
    nên không push được ở chế độ thường. Xem khối ghi chú phía trên `RELEASE_TAG`.

    Trả về thư mục DLL. Ném `FileNotFoundError` nếu thiếu và không được phép/khoong tải được.
    """
    import hashlib
    import urllib.error
    import urllib.request
    import zipfile

    target = Path(lib_dir) if lib_dir else (Path(BACKEND_DIR) / "bin" / _DLL_DIR_NAME)
    # Đường NHANH: chỉ kiểm sự tồn tại. Băm 167 MB mỗi lần spawn worker là lãng phí; hash chỉ
    # được xác thực ngay SAU khi giải nén (lúc dữ liệu vừa từ mạng về — đó mới là lúc cần).
    if _dlls_present(target, verify=False):
        return target

    if not allow_download:
        raise FileNotFoundError(
            f"Thiếu DLL CrispASR trong {target} (cần {', '.join(REQUIRED_DLLS)}) và "
            f"allow_download=False. Tải thủ công: {BUNDLE_URL}"
        )

    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / ".bundle.download.zip"
    try:
        logger.info(
            f"Thiếu DLL CrispASR — đang tải {BUNDLE_ASSET} (~157 MB) từ GitHub Releases...",
            extra={"module_tag": _TAG},
        )
        with urllib.request.urlopen(BUNDLE_URL, timeout=300) as resp:  # noqa: S310 - URL hằng số
            data = resp.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != BUNDLE_SHA256:
            raise RuntimeError(
                f"Gói DLL tải về có SHA-256 {digest} khác giá trị mong đợi {BUNDLE_SHA256} — TỪ CHỐI."
            )
        zip_path.write_bytes(data)

        with zipfile.ZipFile(zip_path) as zf:
            names = {n.split("/")[-1]: n for n in zf.namelist()}
            for name in REQUIRED_DLLS:
                src = names.get(name)
                if src is None:
                    raise RuntimeError(f"Gói DLL không chứa `{name}` — cấu trúc phát hành đã đổi?")
                with zf.open(src) as fin, (target / name).open("wb") as fout:
                    fout.write(fin.read())

        if not _dlls_present(target, verify=True):
            raise RuntimeError(
                f"Đã giải nén nhưng DLL không khớp SHA-256 mong đợi (thư mục {target}) — TỪ CHỐI dùng."
            )
        logger.info(
            f"Đã cài DLL CrispASR vào {target} ({sum((target / n).stat().st_size for n in REQUIRED_DLLS) / 1e6:.0f} MB)",
            extra={"module_tag": _TAG},
        )
        return target
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        raise FileNotFoundError(
            f"Không tải được DLL CrispASR: {type(exc).__name__}: {exc}\n"
            f"  • Tải thủ công `{BUNDLE_ASSET}` từ {BUNDLE_URL},\n"
            f"    giải nén và chép {', '.join(REQUIRED_DLLS)} vào {target}.\n"
            f"  • Hoặc đặt `{ENV_LIB_DIR}` trỏ tới thư mục đã có `crispasr.dll`."
        ) from exc
    finally:
        if zip_path.exists():
            try:
                zip_path.unlink()
            except OSError:
                pass


# =========================================================================== đường dẫn
def aligner_lib_dir() -> Optional[Path]:
    """Thư mục chứa `crispasr.dll` (+ `ggml*.dll`), hoặc None nếu chưa có."""
    override = os.environ.get(ENV_LIB_DIR)
    if override:
        p = Path(override)
        return p if (p / "crispasr.dll").is_file() else None
    candidate = Path(BACKEND_DIR) / "bin" / _DLL_DIR_NAME
    return candidate if (candidate / "crispasr.dll").is_file() else None


def aligner_model_path() -> Path:
    """Đường dẫn file GGUF của aligner trong `backend/models/`."""
    override = os.environ.get(ENV_MODEL)
    return Path(override) if override else Path(MODELS_DIR) / MODEL_FILENAME


def available() -> Tuple[bool, str]:
    """Aligner GGUF có dùng được NGAY không (chưa tính việc tải): `(ok, lý do)`.

    Lưu ý: trả `False` KHÔNG có nghĩa là hỏng — DLL và model đều tự tải được ở lần dùng đầu
    (`ensure_lib_files()` / `ensure_model_file()`). Hàm này chỉ kiểm tra hiện trạng trên đĩa nên
    dùng được ở `/health` và pre-flight mà không kích hoạt tải 157 MB + 500 MB.
    """
    lib = aligner_lib_dir()
    if lib is None or not _dlls_present(lib, verify=False):
        return False, (
            f"chưa có DLL CrispASR trong {Path(BACKEND_DIR) / 'bin' / _DLL_DIR_NAME} "
            f"(sẽ tự tải ~157 MB ở lần dùng đầu, hoặc đặt {ENV_LIB_DIR})"
        )
    model = aligner_model_path()
    if not model.is_file() or model.stat().st_size == 0:
        return False, f"chưa có model GGUF aligner ({model}) — sẽ tự tải ~500 MB ở lần dùng đầu"
    return True, ""


def ensure_model_file(*, allow_download: bool = True) -> Path:
    """Đảm bảo file GGUF aligner có trong `backend/models/` (tải từ HuggingFace nếu thiếu)."""
    model = aligner_model_path()
    if model.is_file() and model.stat().st_size > 0:
        return model
    if not allow_download:
        raise FileNotFoundError(f"Chưa có model aligner cục bộ: {model}")
    from backend.utils.model_download import ensure_model_file as _ensure

    logger.info(
        f"Chưa có aligner GGUF tại {model.name} — bắt đầu tải từ {MODEL_REPO_ID} (~500 MB)...",
        extra={"module_tag": _TAG},
    )
    _ensure(MODEL_REPO_ID, model.name, local_dir=model.parent, stage="ASR")
    return model


# =========================================================================== nạp DLL an toàn
#: Giữ THAM CHIẾU tới handle của `os.add_dll_directory`.
#:
#: ⚠️ BẪY ĐÃ SẬP THẬT: `os.add_dll_directory()` trả về một object; **thư mục chỉ nằm trong đường
#: tìm kiếm DLL khi object đó còn sống**. Bỏ qua giá trị trả về ⇒ object bị GC ⇒ thư mục biến mất
#: khỏi search path. Phải giữ handle tới hết vòng đời tiến trình.
_DLL_DIR_HANDLES: List[Any] = []


def _add_dll_dir(path: Path) -> None:
    """`os.add_dll_directory` + GIỮ handle (xem `_DLL_DIR_HANDLES`)."""
    if sys.platform != "win32":
        return
    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(path)))
    except OSError as exc:
        logger.warning(f"Không đăng ký được thư mục DLL {path}: {exc}", extra={"module_tag": _TAG})


def _cuda_runtime_dirs() -> List[Path]:
    """Thư mục chứa runtime CUDA 13 đi kèm repo (`cudart64_13.dll`, `cublas64_13.dll`…).

    Bản CrispASR `-cuda13-non-cuda` KHÔNG mang cuBLAS/cudart (đó là nghĩa của "non-cuda"); nó
    dùng đúng runtime CUDA 13 mà repo đã có sẵn — cùng bộ mà `setup_cuda_dll_paths()` đang đăng
    ký cho ASR/llama.cpp. Nhờ vậy chỉ mang ~167 MB thay vì ~988 MB (cuBLAS 12 của wheel `+cuda`)
    mà tốc độ y hệt (RTF 0,0069).

    Chỉ nhận thư mục THỰC SỰ có `cudart64_13.dll` — thư mục `cuda_runtime/bin` (chỉ có bản CUDA 12)
    bị loại để không trộn hai major CUDA vào cùng search path.
    """
    root = Path(__file__).resolve().parent.parent.parent
    out: List[Path] = []
    seen: set[str] = set()
    for toolkit_root in (root / "external" / "cuda-toolkit", root / ".cuda-toolkit"):
        nvidia_root = toolkit_root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        try:
            versions = list(nvidia_root.iterdir())
        except OSError:
            continue
        for ver in versions:
            for cand in (ver / "bin" / "x86_64", ver / "bin"):
                if not cand.is_dir():
                    continue
                try:
                    has_13 = any(f.startswith("cudart64_13") for f in os.listdir(cand))
                except OSError:
                    continue
                key = str(cand.resolve()).lower()
                if has_13 and key not in seen:
                    seen.add(key)
                    out.append(cand)
    return out


def _loaded_module_path(name: str) -> Optional[str]:
    """Đường dẫn THẬT của một DLL đã nạp trong tiến trình hiện tại (Windows)."""
    if sys.platform != "win32":
        return None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        k32.GetModuleHandleW.restype = ctypes.c_void_p
        k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        k32.GetModuleFileNameW.restype = ctypes.c_uint32
        handle = k32.GetModuleHandleW(name)
        if not handle:
            return None
        buf = ctypes.create_unicode_buffer(32768)
        if k32.GetModuleFileNameW(handle, buf, len(buf)) == 0:
            return None
        return buf.value or None
    except Exception:  # noqa: BLE001
        return None


# =========================================================================== binding
class CrispASRAlignLib:
    """Binding ctypes tối thiểu cho ABI căn chỉnh của CrispASR (chỉ dùng ở TIẾN TRÌNH CON)."""

    def __init__(self, lib_dir: Path):
        self.lib_dir = Path(lib_dir)
        # THỨ TỰ/PHẠM VI ĐĂNG KÝ QUAN TRỌNG: chỉ (a) runtime CUDA 13 của repo và (b) thư mục DLL
        # CrispASR. TUYỆT ĐỐI không đăng ký `backend/bin`, `backend/bin/llama` hay
        # `site-packages/torch/lib` — chúng cũng chứa `ggml*.dll` và Windows phân giải theo TÊN
        # MODULE, nên đăng ký thêm là tự tạo ra lỗi bind nhầm.
        for d in _cuda_runtime_dirs():
            _add_dll_dir(d)
        _add_dll_dir(self.lib_dir)

        self.lib = ctypes.CDLL(str(self.lib_dir / "crispasr.dll"))

        # ── KIỂM CHỨNG BIND ĐÚNG BỘ ggml ─────────────────────────────────────────
        # Nếu `ggml-base.dll` được nạp từ thư mục khác thì mọi kết quả phía sau đều không đáng tin
        # và tiến trình có thể chết bất cứ lúc nào. Thà chết NGAY, với thông báo rõ ràng.
        bound = _loaded_module_path("ggml-base.dll")
        if bound is not None:
            lib_res = str(self.lib_dir.resolve()).lower()
            if not str(Path(bound).resolve()).lower().startswith(lib_res):
                raise RuntimeError(
                    f"crispasr.dll đã bind NHẦM `ggml-base.dll` từ {bound} (mong đợi trong "
                    f"{self.lib_dir}) — tiến trình sẽ chết với 0xc0000139. Nguyên nhân thường gặp: "
                    f"tiến trình đã import `backend.asr` (nạp ggml của transcribe.cpp) hoặc PATH "
                    f"chứa `backend/bin`/`torch/lib`."
                )
            logger.info(f"Đã bind đúng bộ ggml: {bound}", extra={"module_tag": _TAG})

        f = self.lib.crispasr_align_words_abi
        f.argtypes = [
            ctypes.c_char_p,                    # aligner_model
            ctypes.c_char_p,                    # transcript
            ctypes.POINTER(ctypes.c_float),     # samples
            ctypes.c_int32,                     # n_samples
            ctypes.c_int64,                     # t_offset_cs
            ctypes.c_int32,                     # n_threads
        ]
        f.restype = ctypes.c_void_p
        self._align = f

        self.lib.crispasr_align_result_n_words.argtypes = [ctypes.c_void_p]
        self.lib.crispasr_align_result_n_words.restype = ctypes.c_int
        self.lib.crispasr_align_result_word_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.crispasr_align_result_word_text.restype = ctypes.c_char_p
        for name in ("crispasr_align_result_word_t0", "crispasr_align_result_word_t1"):
            fn = getattr(self.lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
            fn.restype = ctypes.c_int64
        self.lib.crispasr_align_result_free.argtypes = [ctypes.c_void_p]
        self.lib.crispasr_align_result_free.restype = None
        # `crispasr_aligner_free_cache` KHÔNG được export ở mọi build (header khai báo nó không
        # kèm macro `CRISPASR_API`) ⇒ phải kiểm tra tồn tại trước khi dùng.
        self._has_free_cache = hasattr(self.lib, "crispasr_aligner_free_cache")
        if self._has_free_cache:
            self.lib.crispasr_aligner_free_cache.restype = None

    def align_words(
        self,
        model_path: str,
        text: str,
        pcm: np.ndarray,
        t_offset: float = 0.0,
        n_threads: int = DEFAULT_N_THREADS,
    ) -> List[Dict[str, float]]:
        """Gọi ABI. Trả danh sách `{text, t0, t1}` (giây, đã cộng `t_offset`)."""
        samples = np.ascontiguousarray(pcm, dtype=np.float32)
        res = self._align(
            str(model_path).encode("utf-8"),
            text.encode("utf-8"),
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(samples.size),
            int(round(float(t_offset) * 100.0)),
            int(n_threads),
        )
        if not res:
            raise RuntimeError("crispasr_align_words_abi trả về NULL (nạp model hoặc căn chỉnh lỗi)")
        try:
            n = self.lib.crispasr_align_result_n_words(res)
            out: List[Dict[str, float]] = []
            for i in range(n):
                raw = self.lib.crispasr_align_result_word_text(res, i)
                out.append(
                    {
                        "text": raw.decode("utf-8") if raw else "",
                        "t0": self.lib.crispasr_align_result_word_t0(res, i) / 100.0,
                        "t1": self.lib.crispasr_align_result_word_t1(res, i) / 100.0,
                    }
                )
            return out
        finally:
            self.lib.crispasr_align_result_free(res)

    def free_cache(self) -> None:
        """Giải phóng model aligner đang cache trong DLL (trả VRAM).

        No-op nếu build không export `crispasr_aligner_free_cache` — khi đó VRAM được trả bằng
        cách tắt hẳn tiến trình con (`CrispASRAlignerClient.shutdown()`).
        """
        if getattr(self, "_has_free_cache", False):
            self.lib.crispasr_aligner_free_cache()


# =========================================================================== worker
def worker_loop(conn: Any, lib_dir: str, model_path: str, n_threads: int) -> None:
    """Vòng lặp tiến trình con: nạp DLL một lần, phục vụ nhiều yêu cầu căn chỉnh."""
    try:
        # Chốt chặn cuối: nếu vì lý do nào đó bộ ggml của transcribe.cpp đã vào tiến trình này,
        # dừng NGAY thay vì chạy với DLL bind nhầm.
        if _loaded_module_path("ggml-base.dll") is not None:
            raise RuntimeError(
                "tiến trình worker aligner đã có sẵn ggml-base.dll (bind nhầm bộ của "
                "transcribe.cpp) — sai thiết kế cách ly"
            )
        lib = CrispASRAlignLib(Path(lib_dir))
        conn.send(("ready", "crispasr_aligner"))
    except Exception as exc:  # noqa: BLE001
        conn.send(("init_error", f"{type(exc).__name__}: {exc}"))
        conn.close()
        return

    while True:
        try:
            if not conn.poll(None):
                continue
            msg = conn.recv()
            if not isinstance(msg, tuple) or len(msg) < 1:
                continue
            cmd = msg[0]

            if cmd == "ping":
                conn.send(("ok", "pong"))

            elif cmd == "align":
                text, pcm, t_offset, threads = msg[1]
                words = lib.align_words(
                    model_path, text, np.asarray(pcm, dtype=np.float32), t_offset, threads or n_threads
                )
                conn.send(("ok", words))

            elif cmd == "free":
                lib.free_cache()
                conn.send(("ok", True))

            elif cmd == "shutdown":
                try:
                    lib.free_cache()
                except Exception:  # noqa: BLE001
                    pass
                conn.send(("ok", True))
                conn.close()
                return

            else:
                conn.send(("error", f"lệnh không hỗ trợ: {cmd!r}"))

        except Exception as exc:  # noqa: BLE001
            try:
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
            except Exception:  # noqa: BLE001
                return


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point tiến trình con (KHÔNG import `backend.asr` — xem docstring đầu file)."""
    from multiprocessing.connection import Client

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "--worker":
        print("dùng: python -m backend.utils.crispasr_native --worker <lib_dir> <model> <port> <authkey> <threads>")
        return 2
    lib_dir, model_path, port, authkey, n_threads = args[1], args[2], int(args[3]), args[4], int(args[5])
    conn = Client(("127.0.0.1", port), authkey=authkey.encode())
    try:
        worker_loop(conn, lib_dir, model_path, n_threads)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


__all__ = [
    "BUNDLE_ASSET",
    "BUNDLE_SHA256",
    "BUNDLE_URL",
    "DEFAULT_N_THREADS",
    "DLL_SHA256",
    "ENV_LIB_DIR",
    "ENV_MODEL",
    "ENV_THREADS",
    "ENV_WORKER_DEBUG",
    "MODEL_FILENAME",
    "MODEL_REPO_ID",
    "RELEASE_TAG",
    "REQUIRED_DLLS",
    "CrispASRAlignLib",
    "aligner_lib_dir",
    "aligner_model_path",
    "available",
    "ensure_lib_files",
    "ensure_model_file",
    "main",
    "worker_loop",
]


if __name__ == "__main__":  # pragma: no cover - tiến trình con
    raise SystemExit(main())
