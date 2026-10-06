"""FireRed Stream-VAD chạy bằng **onnxruntime** — bỏ hẳn PyTorch runtime.

NGUỒN GỐC & GIẤY PHÉP (Apache-2.0)
==================================
Các lớp `KaldifeatFbank`, `CMVN`, `StreamVadPostprocessor`, `StreamVadFrameResult`, `VadState`
dưới đây là **bản port nguyên thuật toán** từ gói `fireredvad` của Xiaohongshu FireRedTeam
(https://github.com/FireRedTeam/FireRedVAD, Apache-2.0), tác giả Kaituo Xu, Wenpeng Li,
Kai Huang, Kun Liu. Điểm khác duy nhất: **bỏ PyTorch** — phần suy luận dùng model ONNX chính
thức `fireredvad_stream_vad_with_cache.onnx` qua onnxruntime; `audio_feat.py` gốc chỉ dùng torch
ở đúng một dòng (`torch.from_numpy(fbank)`) nên bỏ được mà không đổi số học.

VÌ SAO CHUYỂN ĐƯỢC (đo thật, `backend/tests/test_62_vad_onnx_parity.py`)
=======================================================================
So từng frame với đường PyTorch (`fireredvad.core.detect_model.DetectModel`) trên 12.621 frame
của 4 file (`Japanese_5s`, `English_multiple_kinds_of_noise_88s`, `Chinese_noise_28s`,
`Russian_4s`): `max|Δp| ≤ 1e-6`, **0 frame lệch quyết định** tại ngưỡng 0.30 / 0.40 / 0.50.

HỢP ĐỒNG API ONNX (`fireredvad_stream_vad_with_cache.onnx`):
    feat       float32 [1, T, 80]        # fbank 80 chiều, 25 ms/10 ms, ĐÃ qua CMVN
    caches_in  float32 [8, 1, 128, 19]   # cache DFSMN — zeros ở frame đầu
    probs      float32 [1, T, 1]
    caches_out float32 [8, 1, 128, 19]

KHÁC BIỆT SO VỚI BẢN TORCH
==========================
* Cache do caller giữ ⇒ cùng một `InferenceSession` phục vụ nhiều phiên (bản torch nạp model
  riêng cho từng phiên vì `FireRedStreamVad` giữ cả model lẫn postprocessor).
* Frontend fbank dùng `kaldi-native-fbank` (không kéo torch) + CMVN đọc bằng `kaldiio`.
"""

from __future__ import annotations

import enum
import math
import threading
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from backend.utils.logger import logger

#: Hình học frame của upstream (`fireredvad/core/constants.py`) — KHÔNG đổi được.
FRAME_LENGTH_SAMPLE = 400   # cửa sổ 25 ms
FRAME_SHIFT_SAMPLE = 160    # bước nhảy 10 ms
FRAME_PER_SECONDS = 100

#: Hình dạng cache DFSMN rỗng của frame đầu tiên (đã kiểm chứng bit-exact với `caches=None`).
CACHE_SHAPE: Tuple[int, int, int, int] = (8, 1, 128, 19)

ONNX_FILENAME = "fireredvad_stream_vad_with_cache.onnx"
CMVN_FILENAME = "cmvn.ark"

ONNX_URL = (
    "https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/main/"
    "pretrained_models/onnx_models/fireredvad_stream_vad_with_cache.onnx"
)
CMVN_URL = (
    "https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/main/"
    "pretrained_models/onnx_models/cmvn.ark"
)
#: SHA-256 của hai file chính thức — xác thực mọi lượt tải.
ONNX_SHA256 = "b3c97836130dc34fc32d56fab551e88cf9454511de2b3c250a5e6578dee74b93"
CMVN_SHA256 = "c87f6f13edf0f0ec7535ddfc9cc3387d9268cb234b70182d566c5e2edf3ca473"

_SESSION_LOCK = threading.RLock()
_SESSIONS: Dict[str, Any] = {}


# =========================================================================== frontend
class KaldifeatFbank:
    """Port `fireredvad.core.audio_feat.KaldifeatFbank` — bỏ torch.

    Fbank 80 chiều @16 kHz, cửa sổ 25 ms, bước nhảy 10 ms, `dither=0`, `snip_edges=True`
    (đúng tham số upstream). Với một cửa sổ 400 mẫu, `snip_edges=True` cho ĐÚNG 1 frame.
    """

    def __init__(self, num_mel_bins: int = 80, frame_length: int = 25, frame_shift: int = 10, dither: float = 0):
        import kaldi_native_fbank as knf

        self._knf = knf
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_length_ms = frame_length
        opts.frame_opts.frame_shift_ms = frame_shift
        opts.frame_opts.dither = dither
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = num_mel_bins
        opts.mel_opts.debug_mel = False
        self.opts = opts
        self.num_mel_bins = num_mel_bins

    def __call__(self, wav_int16: np.ndarray) -> np.ndarray:
        """`wav_int16`: mảng int16 (giá trị -32768..32767) → `(n_frames, 80)` float32."""
        fbank = self._knf.OnlineFbank(self.opts)
        fbank.accept_waveform(16000, np.asarray(wav_int16).tolist())
        frames = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
        if not frames:
            return np.zeros((0, self.num_mel_bins), dtype=np.float32)
        return np.vstack(frames).astype(np.float32)


class CMVN:
    """Port `fireredvad.core.audio_feat.CMVN` — đọc `cmvn.ark` bằng `kaldiio`, không cần torch."""

    def __init__(self, kaldi_cmvn_file: str | Path):
        self.dim, self.means, self.inverse_std_variances = self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] != self.dim:
            raise ValueError(f"CMVN dim mismatch: {x.shape[-1]} != {self.dim}")
        return (x - self.means) * self.inverse_std_variances

    @staticmethod
    def read_kaldi_cmvn(kaldi_cmvn_file: str | Path):
        import kaldiio

        path = Path(kaldi_cmvn_file)
        if not path.exists():
            raise FileNotFoundError(f"Không thấy file CMVN: {path}")
        stats = kaldiio.load_mat(str(path))
        if stats.shape[0] != 2:
            raise ValueError(f"cmvn.ark không hợp lệ (shape={stats.shape})")
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        if count < 1:
            raise ValueError("cmvn.ark: count < 1")
        floor = 1e-20
        means, istd = [], []
        for d in range(dim):
            mean = stats[0, d] / count
            variance = (stats[1, d] / count) - mean * mean
            if variance < floor:
                variance = floor
            means.append(float(mean))
            istd.append(1.0 / math.sqrt(float(variance)))
        return dim, np.array(means, dtype=np.float32), np.array(istd, dtype=np.float32)


# =========================================================================== postprocessor
@dataclass
class StreamVadFrameResult:
    """Port `fireredvad.core.stream_vad_postprocessor.StreamVadFrameResult`."""

    frame_idx: int  # 1-based
    is_speech: bool
    raw_prob: float
    smoothed_prob: float
    is_speech_start: bool = False
    is_speech_end: bool = False
    speech_start_frame: int = -1  # 1-based
    speech_end_frame: int = -1    # 1-based


@enum.unique
class VadState(enum.Enum):
    SILENCE = 0
    POSSIBLE_SPEECH = 1
    SPEECH = 2
    POSSIBLE_SILENCE = 3


class StreamVadPostprocessor:
    """Port `fireredvad.core.stream_vad_postprocessor.StreamVadPostprocessor`.

    ⚠️ `process_one_frame` là bản sao **nguyên văn** máy trạng thái 4 pha của upstream. Mọi
    ngưỡng (`speech_threshold`, `min_speech_frame`, `pad_start_frame`, `max_speech_frame`,
    `min_silence_frame`) và thứ tự chuyển trạng thái phải giữ y hệt — đổi một nhánh là đổi
    thời điểm START/END, tức đổi cách cắt câu của cả pipeline.
    """

    def __init__(
        self,
        smooth_window_size: int,
        speech_threshold: float,
        pad_start_frame: int,
        min_speech_frame: int,
        max_speech_frame: int,
        min_silence_frame: int,
    ):
        self.smooth_window_size = max(1, smooth_window_size)
        self.speech_threshold = speech_threshold
        self.pad_start_frame = max(self.smooth_window_size, pad_start_frame)
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.reset()

    def reset(self) -> None:
        self.frame_cnt = 0
        # smooth window
        self.smooth_window = deque()
        self.smooth_window_sum = 0.0
        # state transition
        self.state = VadState.SILENCE
        self.speech_cnt = 0
        self.silence_cnt = 0
        self.hit_max_speech = False
        self.last_speech_start_frame = -1
        self.last_speech_end_frame = -1

    def process_one_frame(self, raw_prob: float) -> StreamVadFrameResult:
        raw_prob = float(raw_prob)
        self.frame_cnt += 1

        smoothed_prob = self.smooth_prob(raw_prob)
        is_speech = self.apply_threshold(smoothed_prob)

        result = StreamVadFrameResult(
            frame_idx=self.frame_cnt,
            is_speech=is_speech,
            raw_prob=round(raw_prob, 3),
            smoothed_prob=round(smoothed_prob, 3),
        )
        return self.state_transition(is_speech, result)

    def smooth_prob(self, prob: float) -> float:
        if self.smooth_window_size <= 1:
            return prob
        self.smooth_window.append(prob)
        self.smooth_window_sum += prob
        if len(self.smooth_window) > self.smooth_window_size:
            left = self.smooth_window.popleft()
            self.smooth_window_sum -= left
        return self.smooth_window_sum / len(self.smooth_window)

    def apply_threshold(self, prob: float) -> int:
        return int(prob >= self.speech_threshold)

    def state_transition(self, is_speech: int, result: StreamVadFrameResult) -> StreamVadFrameResult:
        if self.hit_max_speech:
            result.is_speech_start = True
            result.speech_start_frame = self.frame_cnt
            self.last_speech_start_frame = result.speech_start_frame
            self.hit_max_speech = False

        if self.state == VadState.SILENCE:
            if is_speech:
                self.state = VadState.POSSIBLE_SPEECH
                self.speech_cnt += 1
            else:
                self.silence_cnt += 1
                self.speech_cnt = 0

        elif self.state == VadState.POSSIBLE_SPEECH:
            if is_speech:
                self.speech_cnt += 1
                if self.speech_cnt >= self.min_speech_frame:
                    self.state = VadState.SPEECH
                    result.is_speech_start = True
                    result.speech_start_frame = max(
                        1,
                        self.frame_cnt - self.speech_cnt + 1 - self.pad_start_frame,
                        self.last_speech_end_frame + 1,
                    )
                    self.last_speech_start_frame = result.speech_start_frame
                    self.silence_cnt = 0
            else:
                self.state = VadState.SILENCE
                self.silence_cnt = 1
                self.speech_cnt = 0

        elif self.state == VadState.SPEECH:
            self.speech_cnt += 1
            if is_speech:
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame
            else:
                self.state = VadState.POSSIBLE_SILENCE
                self.silence_cnt += 1

        elif self.state == VadState.POSSIBLE_SILENCE:
            self.speech_cnt += 1
            if is_speech:
                self.state = VadState.SPEECH
                self.silence_cnt = 0
                if self.speech_cnt >= self.max_speech_frame:
                    self.hit_max_speech = True
                    self.speech_cnt = 0
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_start_frame = -1
                    self.last_speech_end_frame = result.speech_end_frame
            else:
                self.silence_cnt += 1
                if self.silence_cnt >= self.min_silence_frame:
                    self.state = VadState.SILENCE
                    result.is_speech_end = True
                    result.speech_end_frame = self.frame_cnt
                    result.speech_start_frame = self.last_speech_start_frame
                    self.last_speech_end_frame = result.speech_end_frame
                    self.last_speech_start_frame = -1
                    self.speech_cnt = 0

        return result


# =========================================================================== model files
def _sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, target: Path, expected_sha256: str) -> bool:
    """Tải 1 file + xác thực SHA-256. Trả `False` (không ném) nếu hỏng."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        logger.info(f"Đang tải {target.name} từ {url} ...", extra={"module_tag": "VAD"})
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - URL hằng số
            data = resp.read()
        digest = _sha256_of_bytes(data)
        if digest != expected_sha256:
            logger.warning(
                f"{target.name} tải về có SHA-256 {digest} khác giá trị mong đợi "
                f"{expected_sha256} — TỪ CHỐI dùng file này.",
                extra={"module_tag": "VAD"},
            )
            return False
        tmp.write_bytes(data)
        tmp.replace(target)
        logger.info(
            f"Đã tải {target.name} ({len(data) / 1e6:.2f} MB, sha256 khớp)",
            extra={"module_tag": "VAD"},
        )
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning(
            f"Không tải được {target.name} từ {url}: {type(exc).__name__}: {exc}",
            extra={"module_tag": "VAD"},
        )
        return False
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _sha256_of_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def ensure_firered_onnx_files(model_dir: Path, *, allow_download: bool = True) -> None:
    """Đảm bảo `cmvn.ark` + `fireredvad_stream_vad_with_cache.onnx` có trong `model_dir`."""
    onnx_path = model_dir / ONNX_FILENAME
    cmvn_path = model_dir / CMVN_FILENAME

    if not (cmvn_path.is_file() and cmvn_path.stat().st_size > 0):
        if not allow_download or not _download(CMVN_URL, cmvn_path, CMVN_SHA256):
            raise FileNotFoundError(
                f"Thiếu {CMVN_FILENAME} trong {model_dir} và không tải được từ {CMVN_URL}"
            )

    if not (onnx_path.is_file() and onnx_path.stat().st_size > 1_000_000):
        if not allow_download or not _download(ONNX_URL, onnx_path, ONNX_SHA256):
            raise FileNotFoundError(
                f"Thiếu {ONNX_FILENAME} trong {model_dir} và không tải được từ {ONNX_URL}"
            )


# =========================================================================== engine model
def get_firered_session(model_path: Path, intra_op_threads: int = 1) -> Any:
    """`InferenceSession` FireRed dùng chung, cache theo (đường dẫn, số luồng)."""
    import onnxruntime as ort

    key = str(model_path)
    with _SESSION_LOCK:
        cached = _SESSIONS.get(key)
        if cached is not None:
            return cached
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = max(1, int(intra_op_threads))
        opts.log_severity_level = 3
        session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        names = {i.name for i in session.get_inputs()}
        if {"feat", "caches_in"} - names:
            raise RuntimeError(f"Model ONNX không đúng chuẩn FireRed Stream-VAD: {model_path}")
        _SESSIONS[key] = session
        logger.info(
            f"FireRed-VAD ONNX sẵn sàng (onnxruntime {ort.__version__}, model={model_path})",
            extra={"module_tag": "VAD"},
        )
        return session


class FireRedStreamVadOnnx:
    """Tương đương `fireredvad.FireRedStreamVad` nhưng chạy ONNX, không cần torch.

    State (`caches`) do instance này giữ ⇒ mỗi phiên một instance, model thì dùng chung.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        smooth_window_size: int = 5,
        speech_threshold: float = 0.4,
        pad_start_frame: int = 5,
        min_speech_frame: int = 8,
        max_speech_frame: int = 2000,
        min_silence_frame: int = 20,
        intra_op_threads: int = 1,
    ):
        self.model_dir = Path(model_dir)
        ensure_firered_onnx_files(self.model_dir)
        self.onnx_path = self.model_dir / ONNX_FILENAME

        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25, frame_shift=10, dither=0)
        self.cmvn = CMVN(self.model_dir / CMVN_FILENAME)
        self._session = get_firered_session(self.onnx_path, intra_op_threads)

        self.postprocessor = StreamVadPostprocessor(
            smooth_window_size,
            float(speech_threshold),
            pad_start_frame,
            min_speech_frame,
            max_speech_frame,
            min_silence_frame,
        )
        #: Giữ API `config` để `FireRedVADEngine._sync_runtime_config` hoạt động y như trước
        #: (nó ghi ngược `speech_threshold` / `min_silence_frame` vào cả postprocessor và config).
        self.config = SimpleNamespace(
            smooth_window_size=int(smooth_window_size),
            speech_threshold=float(speech_threshold),
            pad_start_frame=int(pad_start_frame),
            min_speech_frame=int(min_speech_frame),
            max_speech_frame=int(max_speech_frame),
            min_silence_frame=int(min_silence_frame),
        )
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._caches = np.zeros(CACHE_SHAPE, dtype=np.float32)
        self.postprocessor.reset()

    # ------------------------------------------------------------------ inference
    def _feature(self, window_int16: np.ndarray) -> np.ndarray:
        """1 cửa sổ 400 mẫu int16 → `(1, 1, 80)` fbank ĐÃ qua CMVN."""
        feats = self.fbank(window_int16)
        if feats.shape[0] == 0:
            raise ValueError("Cửa sổ audio quá ngắn để trích fbank")
        return self.cmvn(feats)[:1].reshape(1, 1, -1)

    def frame_probability(self, window_int16: np.ndarray) -> float:
        """Xác suất tiếng nói THÔ của một cửa sổ 400 mẫu (chưa làm mượt/ngưỡng)."""
        feat = self._feature(window_int16)
        probs, self._caches = self._session.run(None, {"feat": feat, "caches_in": self._caches})
        return float(np.asarray(probs).reshape(-1)[0])

    def detect_frame(self, window_int16: np.ndarray) -> StreamVadFrameResult:
        """Tương đương `FireRedStreamVad.detect_frame()` — 1 cửa sổ 400 mẫu vào, 1 kết quả ra."""
        if len(window_int16) != FRAME_LENGTH_SAMPLE:
            raise ValueError(
                f"FireRed cần đúng {FRAME_LENGTH_SAMPLE} mẫu, nhận {len(window_int16)}"
            )
        return self.postprocessor.process_one_frame(self.frame_probability(window_int16))


__all__ = [
    "CACHE_SHAPE",
    "CMVN_FILENAME",
    "CMVN_SHA256",
    "CMVN_URL",
    "FRAME_LENGTH_SAMPLE",
    "FRAME_PER_SECONDS",
    "FRAME_SHIFT_SAMPLE",
    "ONNX_FILENAME",
    "ONNX_SHA256",
    "ONNX_URL",
    "CMVN",
    "FireRedStreamVadOnnx",
    "KaldifeatFbank",
    "StreamVadFrameResult",
    "StreamVadPostprocessor",
    "VadState",
    "ensure_firered_onnx_files",
    "get_firered_session",
]
