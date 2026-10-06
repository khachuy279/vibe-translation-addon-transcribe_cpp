"""Client căn chỉnh mốc thời gian bằng **Qwen3-ForcedAligner-0.6B bản GGUF (CrispASR)**.

MỤC TIÊU: thay đường `transformers` + PyTorch của `ForcedAlignerService` bằng runtime GGML thuần
C++ ⇒ bỏ được `torch`, `torchaudio`, `transformers`, cây vendored `backend/asr/qwen_asr/` và hai
dependency `nagisa`/`soynlp`.

KIẾN TRÚC: module này CHỈ là client. Toàn bộ phần nạp DLL và vòng lặp tiến trình con nằm ở
`backend/utils/crispasr_native.py` — **cố ý**, vì tiến trình con không được import `backend.asr`
(`backend/asr/__init__.py` nạp `transcribe.dll` ⇒ kéo `backend/bin/ggml*.dll` vào ⇒ `crispasr.dll`
bind nhầm `ggml-base.dll` ⇒ `WinError 127`). Xem docstring đầu file đó để có bằng chứng đầy đủ.

Giao tiếp dùng `multiprocessing.connection` (localhost socket) theo ĐÚNG khuôn mẫu của
`backend/tts/worker.py`.

ĐO THẬT (RTX 5060 Ti):
    nạp model + lần gọi đầu : 127–839 ms (so với torch: 1,6 s nạp + 0,5 s prewarm)
    khối 28,26 s (warm)     : 195 ms  → RTF 0,0069  (torch GPU: RTF 0,0080)
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.utils.crispasr_native import (
    DEFAULT_N_THREADS,
    ENV_LIB_DIR,
    ENV_MODEL,
    ENV_THREADS,
    ENV_WORKER_DEBUG,
    MODEL_FILENAME,
    MODEL_REPO_ID,
    aligner_lib_dir,
    aligner_model_path,
    available,
    ensure_lib_files,
    ensure_model_file,
)
from backend.utils.logger import logger

_TAG = "ASR"

#: Timeout: lần gọi đầu phải nạp model + warmup CUDA graph nên rộng hơn nhiều.
_TIMEOUT_FIRST = 300.0
_TIMEOUT_ALIGN = 120.0
_AUTHKEY_LEN = 32


def _sanitized_env(lib_dir: Path) -> Dict[str, str]:
    """Bản sao `os.environ` đã LOẠI mọi thư mục chứa `ggml*.dll` khác khỏi `PATH`.

    VÌ SAO CẦN: tiến trình chính đã gọi `setup_cuda_dll_paths()`, hàm này chèn `backend/bin`
    (bộ ggml của transcribe.cpp), `backend/bin/llama`, `backend/bin/omnivoice` và
    `site-packages/torch/lib` vào `PATH`. Tiến trình con thừa hưởng `PATH` đó, và Windows có thể
    phân giải `ggml-base.dll` của `crispasr.dll` sang một trong các bản kia ⇒ `WinError 127` /
    `0xc0000139`. Loại bỏ tại đây biến rủi ro đó thành không thể xảy ra.
    """
    env = os.environ.copy()
    ggml_names = ("ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml-vulkan.dll")
    lib_resolved = str(lib_dir.resolve()).lower()

    def _is_conflicting(entry: str) -> bool:
        if not entry:
            return False
        try:
            p = Path(entry)
            if not p.is_dir():
                return False
            if str(p.resolve()).lower() == lib_resolved:
                return False
            return any((p / n).is_file() for n in ggml_names)
        except OSError:
            return False

    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if not _is_conflicting(p)
    )
    env.pop(ENV_LIB_DIR, None)  # worker nhận thư mục DLL qua argv, không qua env
    return env


class CrispASRAlignerClient:
    """Singleton client của tiến trình con căn chỉnh (thread-safe)."""

    _instance: Optional["CrispASRAlignerClient"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "CrispASRAlignerClient":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = CrispASRAlignerClient()
            return cls._instance

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conn: Any = None
        self._proc: Optional[subprocess.Popen] = None
        self._model_path: Optional[str] = None
        self._loaded = False
        self._n_threads = int(os.environ.get(ENV_THREADS, str(DEFAULT_N_THREADS)))

    # ------------------------------------------------------------------ vòng đời
    @property
    def is_worker_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def is_loaded(self) -> bool:
        """Model đã được nạp trong worker chưa (đã chạy ít nhất 1 lần align)."""
        return self._loaded and self.is_worker_alive

    def _ensure_worker_started(self) -> None:
        """Khởi động tiến trình con nếu chưa có (caller phải giữ `self._lock`)."""
        if self.is_worker_alive and self._conn is not None:
            return
        # `ensure_lib_files()` tự tải DLL nếu thiếu (`ggml-cuda.dll` 149,63 MB vượt giới hạn
        # 100 MB/file của GitHub nên KHÔNG commit được — xem `backend/utils/crispasr_native.py`).
        from backend.utils.crispasr_native import ensure_lib_files

        lib_dir = aligner_lib_dir()
        try:
            lib_dir = ensure_lib_files(lib_dir)
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        model = ensure_model_file()

        from multiprocessing.connection import Listener

        authkey = secrets.token_hex(_AUTHKEY_LEN // 2)
        listener = Listener(("127.0.0.1", 0), authkey=authkey.encode())
        port = listener.address[1]

        repo_root = Path(__file__).resolve().parent.parent.parent
        env = _sanitized_env(lib_dir)
        env["PYTHONPATH"] = str(repo_root)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "backend.utils.crispasr_native",
                "--worker",
                str(lib_dir),
                str(model),
                str(port),
                authkey,
                str(self._n_threads),
            ],
            cwd=str(repo_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None if os.environ.get(ENV_WORKER_DEBUG) else subprocess.DEVNULL,
            env=env,
        )
        try:
            conn = listener.accept()
            status, msg = conn.recv()
            if status != "ready":
                raise RuntimeError(f"worker aligner khởi động lỗi: {msg}")
        except Exception as exc:  # noqa: BLE001
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"Không khởi động được worker aligner CrispASR: {exc}") from exc
        finally:
            listener.close()

        self._proc = proc
        self._conn = conn
        self._model_path = str(model)
        logger.info(
            f"Worker căn chỉnh CrispASR đã kết nối (PID {proc.pid}, dll={lib_dir}, model={model.name})",
            extra={"module_tag": _TAG},
        )

    def _terminate_worker(self) -> None:
        """Tắt tiến trình con và thu hồi VRAM (caller phải giữ `self._lock`)."""
        if self._conn is not None:
            try:
                self._conn.send(("shutdown", None))
            except Exception:  # noqa: BLE001
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3.0)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._proc = None
        self._loaded = False

    def shutdown(self) -> None:
        with self._lock:
            self._terminate_worker()

    # ------------------------------------------------------------------ API
    def prewarm(self, sample_rate: int = 16000) -> None:
        """Nạp model + warmup CUDA graph bằng một đoạn audio ngắn (BLOCKING)."""
        dummy = np.zeros(sample_rate, dtype=np.float32)
        dummy[1000:6000] = (0.05 * np.sin(np.arange(5000) * 0.1)).astype(np.float32)
        self.align(dummy, "hello world", 0.0)

    def free(self) -> bool:
        """Giải phóng model aligner trong worker (trả VRAM)."""
        with self._lock:
            if not (self.is_worker_alive and self._conn is not None):
                return False
            try:
                self._conn.send(("free", None))
                if not self._conn.poll(30.0):
                    raise TimeoutError("free quá thời gian")
                status, _data = self._conn.recv()
                self._loaded = False
                return status == "ok"
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"free aligner CrispASR lỗi: {exc}", extra={"module_tag": _TAG})
                self._terminate_worker()
                return False

    def align(
        self,
        pcm: np.ndarray,
        text: str,
        t_offset: float = 0.0,
    ) -> List[Dict[str, float]]:
        """Căn chỉnh `text` với `pcm` (float32 @16 kHz). Trả `[{text, t0, t1}]` (giây).

        `t_offset` (giây) được cộng vào mọi mốc ⇒ mốc TUYỆT ĐỐI theo timeline video.
        """
        clean = (text or "").strip()
        if not clean or pcm is None or len(pcm) == 0:
            return []
        arr = np.ascontiguousarray(pcm, dtype=np.float32)

        with self._lock:
            self._ensure_worker_started()
            assert self._conn is not None
            timeout = _TIMEOUT_FIRST if not self._loaded else _TIMEOUT_ALIGN
            t0 = time.perf_counter()
            try:
                self._conn.send(("align", (clean, arr, float(t_offset), self._n_threads)))
                if not self._conn.poll(timeout):
                    raise TimeoutError(f"align quá thời gian ({timeout}s)")
                status, data = self._conn.recv()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"Worker căn chỉnh CrispASR lỗi ({type(exc).__name__}: {exc}) — khởi động lại.",
                    extra={"module_tag": _TAG},
                )
                self._terminate_worker()
                raise
            if status != "ok":
                raise RuntimeError(f"lỗi từ worker aligner: {data}")
            self._loaded = True
            logger.debug(
                f"Căn chỉnh CrispASR: {len(data)} đơn vị trong {(time.perf_counter() - t0) * 1000:.1f}ms",
                extra={"module_tag": _TAG},
            )
            return list(data)


__all__ = [
    "ENV_LIB_DIR",
    "ENV_MODEL",
    "MODEL_FILENAME",
    "MODEL_REPO_ID",
    "CrispASRAlignerClient",
    "aligner_lib_dir",
    "aligner_model_path",
    "available",
    "ensure_lib_files",
    "ensure_model_file",
]
