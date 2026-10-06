"""Tiến trình Worker chuyên biệt chạy OmniVoice C++ native (Process Isolation).

Mục tiêu cốt lõi:
1. Cách ly hoàn toàn DLL: omnivoice.dll chạy trong tiến trình Python con độc lập
   qua `subprocess.Popen([sys.executable, "-m", "backend.tts.worker", ...])`.
   Hoàn toàn không re-import `backend.main` hay nạp `ggml.dll` của ASR.
2. Thu hồi VRAM triệt để: Khi tắt TTS, tiến trình con được giải phóng hoàn toàn và
   HĐH/NVIDIA Driver lập tức thu hồi 100% VRAM GPU.
3. Chống crash lan truyền: Bất kỳ ngoại lệ native nào cũng không làm sập server chính.
4. Giao tiếp cực nhanh: Sử dụng IPC Connection chuẩn của Python (độ trễ < 0.2ms).
"""

import os
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener, Client, Connection
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

from backend.config import BACKEND_DIR
from backend.utils.logger import logger

_TAG = "TTS"
_AUTHKEY_LEN = 32


def _worker_loop(conn: Connection, dll_dir: str) -> None:
    """Vòng lặp lắng nghe lệnh từ tiến trình chính trong TTS Worker subprocess."""
    try:
        from backend.utils.cuda import setup_cuda_dll_paths

        setup_cuda_dll_paths()
        from backend.tts.native import OmniVoiceNative

        engine = OmniVoiceNative(dll_dir=Path(dll_dir))
    except Exception as exc:
        conn.send(("init_error", f"{type(exc).__name__}: {exc}"))
        conn.close()
        return

    conn.send(("ready", "worker_initialized"))

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

            elif cmd == "load":
                model_path, codec_path, use_fa, clamp_fp16 = msg[1]
                engine.load_model(model_path, codec_path, use_fa=use_fa, clamp_fp16=clamp_fp16)
                conn.send(("ok", True))

            elif cmd == "is_loaded":
                conn.send(("ok", engine.is_loaded))

            elif cmd == "extract_voice":
                voice_key, audio_24k = msg[1]
                ok = engine.extract_voice_ref(voice_key, audio_24k)
                conn.send(("ok", ok))

            elif cmd == "has_voice":
                voice_key = msg[1]
                conn.send(("ok", voice_key in engine._cached_voice_refs))

            elif cmd == "synthesize":
                text, voice_key, ref_text, raw_audio, num_steps, seed, lang, instruct = msg[1]
                audio_np = engine.synthesize(
                    text=text,
                    voice_key=voice_key,
                    ref_text=ref_text,
                    raw_ref_audio_24k=raw_audio,
                    num_steps=num_steps,
                    seed=seed,
                    lang=lang,
                    instruct=instruct,
                )
                conn.send(("ok", audio_np))

            elif cmd == "unload":
                engine.unload_model()
                conn.send(("ok", True))

            elif cmd == "shutdown":
                engine.unload_model()
                conn.send(("ok", True))
                break

            else:
                conn.send(("error", f"Lệnh không xác định: {cmd}"))

        except EOFError:
            break
        except Exception as exc:
            try:
                conn.send(("error", f"{type(exc).__name__}: {exc}"))
            except Exception:
                break

    try:
        conn.close()
    except Exception:
        pass


class TTSWorkerClient:
    """Client giao tiếp với TTS Worker Subprocess từ tiến trình chính FastAPI."""

    _instance: Optional["TTSWorkerClient"] = None
    _client_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "TTSWorkerClient":
        with cls._client_lock:
            if cls._instance is None:
                cls._instance = TTSWorkerClient()
            return cls._instance

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._conn: Optional[Connection] = None
        self._lock = threading.RLock()
        self._is_loaded = False
        self._dll_dir = str((BACKEND_DIR / "bin" / "omnivoice").resolve())

    def _ensure_worker_started(self) -> None:
        """Khởi động Worker Process nếu chưa chạy hoặc bị crash."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._conn is not None:
                return

            self._terminate_worker()

            authkey_str = secrets.token_hex(_AUTHKEY_LEN)
            authkey_bytes = authkey_str.encode("ascii")

            listener = Listener(("127.0.0.1", 0), authkey=authkey_bytes)
            port = listener.address[1]

            cmd = [
                sys.executable,
                "-m",
                "backend.tts.worker",
                str(port),
                authkey_str,
                self._dll_dir,
            ]

            env = os.environ.copy()
            cuda_dll = str(Path(self._dll_dir) / "ggml-cuda.dll")
            if os.path.isfile(cuda_dll):
                env["GGML_BACKEND_PATH"] = cuda_dll

            # KHÔNG import torch ở đây (chỉ để lấy `torch/lib` thì quá đắt). cudart/cublas cho
            # `ggml-cuda.dll` đã có qua: (1) PATH kế thừa từ tiến trình chính sau
            # `setup_cuda_dll_paths()`, (2) chính worker gọi lại `setup_cuda_dll_paths()` trong
            # `_worker_loop` — hàm này quét `torch/lib`, `nvidia/*/bin`, toolkit CUDA trong repo
            # và CUDA_PATH theo ĐƯỜNG DẪN, không import gói nào.
            extra_paths = [self._dll_dir]
            env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PATH", "")

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )

            try:
                # Chờ subprocess kết nối ngược lại
                conn = listener.accept()
                status, msg = conn.recv()
                if status == "ready":
                    self._proc = proc
                    self._conn = conn
                    logger.info(f"TTS Worker Subprocess đã kết nối thành công (PID {proc.pid})", extra={"module_tag": _TAG})
                    return
                raise RuntimeError(f"TTS Worker khởi động lỗi: {msg}")
            except Exception as exc:
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
                raise RuntimeError(f"Không thể khởi động TTS Worker: {exc}") from exc
            finally:
                listener.close()

    def _terminate_worker(self) -> None:
        """Tắt hoàn toàn tiến trình con."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(("shutdown", None))
                except Exception:
                    pass
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

            if self._proc is not None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None

            self._is_loaded = False

    def is_worker_alive(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.is_worker_alive()

    def _send_cmd(self, cmd: str, payload: Any = None, timeout: float = 60.0) -> Any:
        with self._lock:
            self._ensure_worker_started()
            assert self._conn is not None

            self._conn.send((cmd, payload))
            if not self._conn.poll(timeout):
                logger.error(f"Lệnh TTS '{cmd}' quá thời gian ({timeout}s). Đang khởi động lại worker...", extra={"module_tag": _TAG})
                self._terminate_worker()
                raise TimeoutError(f"TTS Worker timeout khi thực thi lệnh '{cmd}'")

            status, data = self._conn.recv()
            if status == "ok":
                return data
            raise RuntimeError(f"Lỗi từ TTS Worker ({cmd}): {data}")

    def load_model(
        self,
        model_path: str,
        codec_path: str,
        use_fa: bool = True,
        clamp_fp16: bool = False,
    ) -> None:
        """Nạp model GGUF vào Worker."""
        with self._lock:
            t0 = time.perf_counter()
            self._send_cmd("load", (model_path, codec_path, use_fa, clamp_fp16), timeout=30.0)
            self._is_loaded = True
            elapsed = time.perf_counter() - t0
            logger.info(f"Đã nạp model OmniVoice GGUF trong {elapsed:.2f}s", extra={"module_tag": _TAG})

    def extract_voice(self, voice_key: str, audio_24k: np.ndarray) -> bool:
        """Trích xuất mã giọng tham chiếu và lưu vào cache của Worker."""
        with self._lock:
            return bool(self._send_cmd("extract_voice", (voice_key, audio_24k), timeout=15.0))

    def has_voice(self, voice_key: str) -> bool:
        """Kiểm tra xem voice_key đã có trong cache Worker chưa."""
        with self._lock:
            if not self.is_loaded:
                return False
            return bool(self._send_cmd("has_voice", voice_key, timeout=5.0))

    def synthesize(
        self,
        text: str,
        voice_key: Optional[str] = None,
        ref_text: Optional[str] = None,
        raw_ref_audio: Optional[np.ndarray] = None,
        num_steps: int = 16,
        seed: int = 42,
        lang: str = "",
        instruct: str = "",
        timeout: float = 60.0,
    ) -> np.ndarray:
        """Tổng hợp âm thanh float32 24kHz từ Worker."""
        with self._lock:
            payload = (text, voice_key, ref_text, raw_ref_audio, num_steps, seed, lang, instruct)
            result = self._send_cmd("synthesize", payload, timeout=timeout)
            if isinstance(result, np.ndarray):
                return result
            return np.zeros((0,), dtype=np.float32)

    def unload_model(self) -> None:
        """Giải phóng hoàn toàn model và tắt worker để trả 100% VRAM."""
        with self._lock:
            self._terminate_worker()
            logger.info("Đã tắt TTS Worker (toàn bộ VRAM đã thu hồi)", extra={"module_tag": _TAG})


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        port = int(sys.argv[1])
        authkey_str = sys.argv[2]
        dll_dir = sys.argv[3]
        authkey_bytes = authkey_str.encode("ascii")

        # Đảm bảo PYTHONPATH trỏ tới thư mục dự án
        _here = Path(__file__).resolve().parent.parent.parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))

        # Cấu hình CUDA DLL và thư mục DLL cho worker
        cuda_dll = os.path.join(dll_dir, "ggml-cuda.dll")
        if os.path.isfile(cuda_dll):
            os.environ["GGML_BACKEND_PATH"] = cuda_dll
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        if sys.platform == "win32":
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass

        try:
            conn = Client(("127.0.0.1", port), authkey=authkey_bytes)
            _worker_loop(conn, dll_dir)
        except Exception as e:
            sys.stderr.write(f"Worker process crash: {e}\n")
            sys.exit(1)
