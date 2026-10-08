"""Silero VAD v5 chạy bằng **onnxruntime** — bỏ hẳn PyTorch runtime.

VÌ SAO FILE NÀY TỒN TẠI
=======================
Trước đây VAD Silero dùng model TorchScript (`silero_vad.jit`) qua gói `silero-vad`, kéo theo
`torch` + `torchaudio` vào tiến trình. Model ONNX (`silero_vad.onnx`) **cho CÙNG xác suất từng
frame** với bản JIT — đo thật trên 3.797 cửa sổ của 3 file audio (`Japanese_5s`,
`English_multiple_kinds_of_noise_88s`, `Chinese_noise_28s`): `max|Δp| ≤ 2e-6`, **0 cửa sổ lệch
quyết định** tại ngưỡng 0.30 / 0.40 / 0.50. Xem `backend/tests/test_62_vad_onnx_parity.py`.

HỢP ĐỒNG API ONNX (đúng như `OnnxWrapper` của gói `silero-vad`):
    input  float32 [batch, 576]   # 576 = 64 mẫu CONTEXT + 512 mẫu cửa sổ MỚI
    state  float32 [2, batch, 128]
    sr     int64   scalar (16000)
    output float32 [batch, 1]     # xác suất tiếng nói của cửa sổ
    stateN float32 [2, batch, 128]

⚠️ Phải ghép 64 mẫu context vào trước cửa sổ, nếu không model trả xác suất **sai hoàn toàn**
(đo được: ~83% cửa sổ lệch quyết định). Đây là cái bẫy dễ mắc nhất khi tự viết đường ONNX.

KHÁC BIỆT CÓ LỢI SO VỚI BẢN JIT
==============================
State (`state`, `context`) do **caller sở hữu** thay vì giấu trong model ⇒

* Một `InferenceSession` **dùng chung** cho mọi phiên (bản JIT phải nạp model riêng cho từng
  phiên vì `reset_states()` ghi vào chính model, tốn ~93 ms/phiên).
* Không cần `torch.no_grad()`, không cần chuyển tensor, không có GIL contention của torch.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from backend.utils.logger import logger

#: Cửa sổ bắt buộc của Silero @16 kHz (32 ms) — KHÔNG đổi được.
WINDOW_SAMPLES = 512
#: Số mẫu "context" phải ghép trước mỗi cửa sổ (OnnxWrapper dùng 64 @16 kHz).
CONTEXT_SAMPLES = 64
#: Hình dạng state LSTM của Silero v5.
STATE_SHAPE: Tuple[int, int, int] = (2, 1, 128)

#: Nguồn tải model ONNX chính thức (silero-vad lưu model trong chính repo GitHub).
#: Ghi chú: đây là nguồn duy nhất KHÔNG phụ thuộc PyTorch; `torch.hub` của gói `silero-vad`
#: cũng trỏ về cùng file.
SILERO_ONNX_URLS: Tuple[str, ...] = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
)
#: Kích thước hợp lệ tối thiểu (file thật ~2,33 MB). Dùng để phát hiện tải hỏng/HTML lỗi.
_MIN_VALID_BYTES = 1_000_000
#: SHA-256 của `silero_vad.onnx` chính thức (khớp file trong wheel `silero-vad` 6.2.2 và
#: file tại `src/silero_vad/data/silero_vad.onnx` của repo snakers4/silero-vad).
#: Dùng để xác thực file TẢI VỀ (nguồn duy nhất có thể bị can thiệp trên đường truyền).
EXPECTED_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

_SESSION_LOCK = threading.RLock()
_SESSIONS: Dict[Tuple[str, int], Any] = {}


# --------------------------------------------------------------------------- model file
def _site_packages_candidates() -> list[Path]:
    """Các thư mục `site-packages` khả dĩ — quét thẳng đĩa, KHÔNG import package.

    Cố ý không dùng `importlib.resources.files("silero_vad.data")`: nó sẽ chạy
    `silero_vad/__init__.py` → `import torch`, đúng thứ ta đang muốn loại bỏ.
    """
    import site
    import sysconfig

    dirs: list[Path] = []
    for getter in (
        lambda: site.getsitepackages(),
        lambda: [site.getusersitepackages()],
        lambda: [sysconfig.get_paths().get("purelib", "")],
    ):
        try:
            for raw in getter():
                if raw:
                    p = Path(raw)
                    if p.is_dir():
                        dirs.append(p)
        except Exception:  # noqa: BLE001
            continue
    seen: set[Path] = set()
    out: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def find_bundled_onnx() -> Optional[Path]:
    """Tìm `silero_vad.onnx` trong các gói đã cài (đường OFFLINE, không cần mạng).

    Dùng khi gói `silero-vad` vẫn còn trong môi trường: file model nằm sẵn ở
    `<site-packages>/silero_vad/data/silero_vad.onnx`. Chỉ đọc đĩa nên không kéo torch.
    """
    for root in _site_packages_candidates():
        candidate = root / "silero_vad" / "data" / "silero_vad.onnx"
        if candidate.is_file() and candidate.stat().st_size >= _MIN_VALID_BYTES:
            return candidate
    return None


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _download_onnx(target: Path) -> bool:
    """Tải model ONNX từ nguồn chính thức + xác thực SHA-256. Trả `False` nếu mọi URL hỏng."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".onnx.part")
    for url in SILERO_ONNX_URLS:
        try:
            logger.info(f"Đang tải Silero VAD ONNX từ {url} ...", extra={"module_tag": "VAD"})
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - URL hằng số
                data = resp.read()
            if len(data) < _MIN_VALID_BYTES:
                logger.warning(
                    f"Tải Silero ONNX từ {url} về file quá nhỏ ({len(data)} byte) — bỏ qua.",
                    extra={"module_tag": "VAD"},
                )
                continue
            digest = _sha256(data)
            if digest != EXPECTED_SHA256:
                logger.warning(
                    f"Silero ONNX tải từ {url} có SHA-256 {digest} khác giá trị mong đợi "
                    f"{EXPECTED_SHA256} — TỪ CHỐI dùng file này.",
                    extra={"module_tag": "VAD"},
                )
                continue
            tmp.write_bytes(data)
            tmp.replace(target)
            logger.info(
                f"Đã tải Silero VAD ONNX về {target} ({len(data) / 1e6:.2f} MB, sha256 khớp)",
                extra={"module_tag": "VAD"},
            )
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning(
                f"Không tải được Silero ONNX từ {url}: {type(exc).__name__}: {exc}",
                extra={"module_tag": "VAD"},
            )
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return False


def ensure_silero_onnx_model(target: Optional[Path] = None, *, allow_download: bool = True) -> Path:
    """Đảm bảo `backend/models/silero_vad.onnx` tồn tại; trả đường dẫn cục bộ.

    Thứ tự ưu tiên (KHÔNG bao giờ import torch ở bất kỳ bước nào):
      1. File đã có trong `backend/models/` → dùng ngay, không gọi mạng.
      2. File model đi kèm gói `silero-vad` đang cài → sao chép (đường offline).
      3. Tải từ nguồn chính thức (`SILERO_ONNX_URLS`), có xác thực SHA-256.
    """
    from backend.config import MODELS_DIR

    dest = Path(target) if target else MODELS_DIR / "silero_vad.onnx"
    if dest.is_file() and dest.stat().st_size >= _MIN_VALID_BYTES:
        return dest

    bundled = find_bundled_onnx()
    if bundled is not None:
        data = bundled.read_bytes()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        digest = _sha256(data)
        if digest != EXPECTED_SHA256:
            logger.warning(
                f"silero_vad.onnx lấy từ gói cài đặt ({bundled}) có SHA-256 {digest} khác bản "
                f"đã kiểm chứng ({EXPECTED_SHA256}) — vẫn dùng, nhưng hành vi VAD có thể khác "
                f"bản đã đo parity.",
                extra={"module_tag": "VAD"},
            )
        else:
            logger.info(
                f"Đã sao chép silero_vad.onnx từ gói cài đặt ({bundled}) vào {dest} (sha256 khớp)",
                extra={"module_tag": "VAD"},
            )
        return dest

    if allow_download and _download_onnx(dest):
        return dest

    raise FileNotFoundError(
        f"Không tìm thấy model Silero VAD ONNX tại {dest}.\n"
        f"  • Đã thử sao chép từ gói `silero-vad` (không có) và tải từ: "
        f"{', '.join(SILERO_ONNX_URLS)}\n"
        f"  • Cách khắc phục: tải thủ công `silero_vad.onnx` (model v5, ~2,33 MB, "
        f"sha256={EXPECTED_SHA256}) rồi đặt vào {dest.parent}."
    )


# --------------------------------------------------------------------------- session
def get_onnx_session(model_path: Path, intra_op_threads: int = 1) -> Any:
    """`InferenceSession` dùng chung, cache theo (đường dẫn, số luồng).

    onnxruntime cho phép nhiều luồng gọi `run()` đồng thời trên cùng session; state của
    Silero nằm NGOÀI session nên chia sẻ session là an toàn.
    """
    import onnxruntime as ort

    key = (str(model_path), int(intra_op_threads))
    with _SESSION_LOCK:
        cached = _SESSIONS.get(key)
        if cached is not None:
            return cached

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = max(1, int(intra_op_threads))
        opts.log_severity_level = 3  # tắt cảnh báo "Removing initializer ..."
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])

        inputs = {i.name for i in session.get_inputs()}
        missing = {"input", "state", "sr"} - inputs
        if missing:
            raise RuntimeError(
                f"Model ONNX không đúng chuẩn Silero VAD (thiếu input {sorted(missing)}): {model_path}"
            )
        _SESSIONS[key] = session
        logger.info(
            f"Silero VAD ONNX sẵn sàng (onnxruntime {ort.__version__}, "
            f"intra_op_threads={opts.intra_op_num_threads}, model={model_path})",
            extra={"module_tag": "VAD"},
        )
        return session


class SileroVadOnnx:
    """Model Silero VAD ONNX + state RIÊNG của một phiên quét.

    Khác bản JIT ở chỗ state không giấu trong model: hai `SileroVadOnnx` cùng dùng một
    `InferenceSession` nhưng hoàn toàn độc lập về state.
    """

    def __init__(self, model_path: Path, *, intra_op_threads: int = 1):
        self.model_path = Path(model_path)
        self._session = get_onnx_session(self.model_path, intra_op_threads)
        self._sr = np.asarray(16000, dtype=np.int64)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Xoá state về 0 — tương đương `model.reset_states()` của bản JIT."""
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    # ------------------------------------------------------------------ inference
    def probability(self, window: np.ndarray) -> float:
        """Xác suất tiếng nói của ĐÚNG một cửa sổ 512 mẫu float32 trong [-1, 1]."""
        w = np.asarray(window, dtype=np.float32).reshape(1, -1)
        if w.shape[1] != WINDOW_SAMPLES:
            raise ValueError(f"Silero cần đúng {WINDOW_SAMPLES} mẫu, nhận {w.shape[1]}")
        x = np.concatenate([self._context, w], axis=1)  # (1, 576)
        out, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(np.asarray(out).reshape(-1)[0])

    def probabilities(self, windows: np.ndarray) -> np.ndarray:
        """Chạy TUẦN TỰ nhiều cửa sổ `(n, 512)` — đúng ngữ nghĩa streaming.

        ⚠️ KHÔNG gộp thành một lô: state LSTM của Silero được xử lý khác khi batch > 1 và
        kết quả ĐỔI HẲN (đo trong `core/vad_silence.py`: batch=8 cho khoảng lặng dài nhất
        1,41 s so với 5,50 s của batch=1).
        """
        arr = np.asarray(windows, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != WINDOW_SAMPLES:
            raise ValueError(f"`windows` phải có hình dạng (n, {WINDOW_SAMPLES}), nhận {arr.shape}")
        return np.fromiter((self.probability(arr[i]) for i in range(arr.shape[0])), dtype=np.float32, count=arr.shape[0])


class SileroVADIterator:
    """Bản port **chính xác** `silero_vad.VADIterator` sang ONNX.

    Logic dưới đây sao nguyên thuật toán upstream (silero-vad 6.2.x, `utils_vad.py`), chỉ đổi
    nguồn xác suất từ TorchScript sang ONNX:

        threshold            : ngưỡng MỞ đoạn (`>=`)
        threshold - 0.15     : ngưỡng ĐÓNG đoạn (`<`) — hằng số trễ do upstream hardcode
        min_silence_samples  : số mẫu im lặng tối thiểu trước khi chốt END
        speech_pad_samples   : đệm hai mép đoạn nói

    Không đổi bất kỳ ngưỡng/nhánh nào — nếu đổi, hành vi cắt câu sẽ lệch so với bản torch.
    """

    def __init__(
        self,
        model: SileroVadOnnx,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.sampling_rate = int(sampling_rate)
        if self.sampling_rate not in (8000, 16000):
            raise ValueError("VADIterator chỉ hỗ trợ 8000 hoặc 16000 Hz")

        self.min_silence_samples = self.sampling_rate * float(min_silence_duration_ms) / 1000
        self.speech_pad_samples = self.sampling_rate * float(speech_pad_ms) / 1000
        self.reset_states()

    def reset_states(self) -> None:
        self.model.reset()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        #: Xác suất của cửa sổ VỪA xử lý (thay cho `_ProbProbe` bọc model JIT trước đây).
        self.last_prob = 0.0

    def __call__(self, x: np.ndarray) -> Optional[dict]:
        """Trả `{'start': mẫu}` / `{'end': mẫu}` / `None` — y hệt upstream."""
        window_size_samples = int(np.asarray(x).shape[0])
        self.current_sample += window_size_samples

        speech_prob = self.model.probability(x)
        self.last_prob = speech_prob

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {"start": int(speech_start)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
            self.temp_end = 0
            self.triggered = False
            return {"end": int(speech_end)}

        return None

    # ------------------------------------------------------------------ tương thích
    @staticmethod
    def _sync_runtime_config(iterator: "SileroVADIterator", threshold: float, silence_ms: int) -> None:
        """Áp cấu hình runtime (đổi ngưỡng/silence giữa phiên) — như bản JIT."""
        if float(iterator.threshold) != float(threshold):
            iterator.threshold = float(threshold)
        min_silence_samples = 16000 * float(silence_ms) / 1000.0
        if float(iterator.min_silence_samples) != min_silence_samples:
            iterator.min_silence_samples = min_silence_samples


__all__ = [
    "CONTEXT_SAMPLES",
    "SILERO_ONNX_URLS",
    "STATE_SHAPE",
    "WINDOW_SAMPLES",
    "SileroVADIterator",
    "SileroVadOnnx",
    "ensure_silero_onnx_model",
    "find_bundled_onnx",
    "get_onnx_session",
]
