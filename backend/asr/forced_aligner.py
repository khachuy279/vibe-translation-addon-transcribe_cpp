"""Qwen3ForcedAligner: Engine căn chỉnh mốc thời gian từ/ký tự (Word Timestamps) cho Pipeline B.

Đặc tính kỹ thuật:
- Non-Autoregressive (NAR) single-forward pass siêu tốc (~30-100ms trên GPU).
- Độ chính xác AAS 32-52ms, hỗ trợ 11 ngôn ngữ (Anh, Nhật, Trung, Pháp, Đức, Tây Ban Nha, Ý, Nga, Hàn...).
- Thread-safe singleton với `_infer_lock`.
- Hỗ trợ prewarm lúc khởi động để triệt tiêu độ trễ 2.9s JIT kernel ban đầu.
- Tích hợp bộ gom câu phụ đề theo dấu câu và độ dài (Punctuation-guided Sentence Grouping).
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from backend.utils.logger import get_logger

logger = get_logger("asr.forced_aligner")

# Tương thích transformers v5: đăng ký lại default RoPE nếu thiếu
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _default_rope_init(config=None, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
except Exception:
    pass

# Đảm bảo đường dẫn tới qwen3-asr được nạp nếu cần
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QWEN_ASR_DIR = REPO_ROOT / "qwen3-asr" / "Qwen3-ASR"
if QWEN_ASR_DIR.exists() and str(QWEN_ASR_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_ASR_DIR))

try:
    from qwen_asr import Qwen3ForcedAligner
except ImportError:
    try:
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
    except ImportError:
        Qwen3ForcedAligner = None

_ALIGNER_LANG_MAP: Dict[str, str] = {
    "en": "English",
    "english": "English",
    "ja": "Japanese",
    "japanese": "Japanese",
    "zh": "Chinese",
    "chinese": "Chinese",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "ko": "Korean",
    "korean": "Korean",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spanish": "Spanish",
}


def resolve_aligner_language(lang_code: Optional[str]) -> Optional[str]:
    """Chuyển mã ngôn ngữ sang tên ngôn ngữ chuẩn của Qwen3-ForcedAligner.

    Trả về None nếu không thuộc 11 ngôn ngữ được hỗ trợ hoặc là 'auto'.
    """
    if not lang_code:
        return None
    clean = str(lang_code).strip().lower()
    return _ALIGNER_LANG_MAP.get(clean)


@dataclass
class AlignedWord:
    text: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class SubtitleSentence:
    text: str
    start_time: float
    end_time: float
    words: List[AlignedWord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def __repr__(self) -> str:
        return f"SubtitleSentence([{self.start_time:.3f}s -> {self.end_time:.3f}s] \"{self.text}\")"


MODELS_DIR = REPO_ROOT / "backend" / "models"
ALIGNER_LOCAL_DIR = MODELS_DIR / "Qwen3-ForcedAligner-0.6B"
ALIGNER_DEFAULT_REPO_ID = "Qwen/Qwen3-ForcedAligner-0.6B"


def ensure_forced_aligner_model(
    local_dir: Optional[Union[str, Path]] = None,
    repo_id: str = ALIGNER_DEFAULT_REPO_ID,
    allow_download: bool = True,
) -> Path:
    """Đảm bảo thư mục model Qwen3-ForcedAligner-0.6B tồn tại trong `backend/models`.

    Nếu chưa có, tự động tải snapshot từ Hugging Face về `backend/models/Qwen3-ForcedAligner-0.6B`.
    Luôn trả về đường dẫn cục bộ để `from_pretrained` nạp offline.
    """
    target_path = Path(local_dir) if local_dir else ALIGNER_LOCAL_DIR
    config_file = target_path / "config.json"
    model_file = target_path / "model.safetensors"

    if target_path.exists() and config_file.exists() and (model_file.exists() or any(target_path.glob("*.safetensors"))):
        return target_path

    if not allow_download:
        raise FileNotFoundError(
            f"Chưa có thư mục model ForcedAligner cục bộ tại {target_path} và allow_download=False"
        )

    logger.info(
        f"Chưa có model Qwen3-ForcedAligner-0.6B tại {target_path}. Bắt đầu tự động tải từ Hugging Face ({repo_id})...",
        extra={"module_tag": "ASR"},
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_path),
        )
        logger.info(
            f"Tải Qwen3-ForcedAligner-0.6B về {target_path} hoàn tất.",
            extra={"module_tag": "ASR"},
        )
    except Exception as exc:
        logger.error(
            f"Lỗi khi tải Qwen3-ForcedAligner-0.6B về {target_path}: {exc}",
            extra={"module_tag": "ASR"},
        )
        raise

    return target_path


class ForcedAlignerService:
    """Singleton quản lý mô hình Qwen3-ForcedAligner-0.6B trên GPU, luôn đọc từ backend/models."""

    _instance: Optional["ForcedAlignerService"] = None
    _shared_aligner: Optional[Any] = None
    _shared_lock = threading.RLock()
    _infer_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "ForcedAlignerService":
        with cls._shared_lock:
            if cls._instance is None:
                cls._instance = ForcedAlignerService()
            return cls._instance

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        repo_id: str = ALIGNER_DEFAULT_REPO_ID,
    ):
        self.model_path = Path(model_path) if model_path else ALIGNER_LOCAL_DIR
        self.repo_id = repo_id
        self._is_warmed = False

    def load_model(self) -> Any:
        """Nạp mô hình lên GPU từ backend/models (idempotent, thread-safe)."""
        with self._shared_lock:
            if self.__class__._shared_aligner is not None:
                return self.__class__._shared_aligner

            if Qwen3ForcedAligner is None:
                raise RuntimeError("qwen_asr hoặc Qwen3ForcedAligner chưa được cài đặt trong môi trường!")

            # Đảm bảo model đã có trong backend/models
            resolved_path = ensure_forced_aligner_model(self.model_path, repo_id=self.repo_id)

            t0 = time.perf_counter()
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

            logger.info(
                f"Bắt đầu nạp ForcedAligner từ '{resolved_path}' (device={device}, dtype={dtype})...",
                extra={"module_tag": "ASR"},
            )
            aligner = Qwen3ForcedAligner.from_pretrained(
                str(resolved_path),
                dtype=dtype,
                device_map=device,
            )
            self.__class__._shared_aligner = aligner
            logger.info(
                f"Nạp ForcedAligner hoàn tất trong {time.perf_counter() - t0:.2f}s.",
                extra={"module_tag": "ASR"},
            )
            return self.__class__._shared_aligner

    def prewarm(self) -> None:
        """Chạy prewarm một audio ngắn để loại bỏ 2.9s JIT compilation của PyTorch/CUDA."""
        with self._shared_lock:
            if self._is_warmed:
                return
            try:
                aligner = self.load_model()
                t0 = time.perf_counter()
                t_arr = np.linspace(0, 1.0, 16000, endpoint=False)
                dummy_pcm = (0.2 * np.sin(2 * np.pi * 220.0 * t_arr)).astype(np.float32)
                dummy_text = "Hello world."
                with self._infer_lock:
                    _ = aligner.align(
                        audio=(dummy_pcm, 16000),
                        text=dummy_text,
                        language="English",
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._is_warmed = True
                logger.info(
                    f"Prewarm ForcedAligner hoàn tất trong {time.perf_counter() - t0:.2f}s.",
                    extra={"module_tag": "ASR"},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Prewarm ForcedAligner lỗi (bỏ qua): {exc}", extra={"module_tag": "ASR"})

    def align(
        self,
        audio: Union[np.ndarray, Tuple[np.ndarray, int], str],
        text: str,
        language: str = "English",
    ) -> List[AlignedWord]:
        """Căn chỉnh thời gian từng từ.
        
        Args:
            audio: numpy array 16kHz float32, hoặc tuple (pcm, sr), hoặc file path.
            text: Văn bản nhận dạng từ ASR.
            language: Ngôn ngữ (e.g. 'English', 'Japanese', 'Chinese'...).
        
        Returns:
            Danh sách AlignedWord(text, start_time, end_time).
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return []

        aligner = self.load_model()
        if not self._is_warmed:
            self.prewarm()

        # Chuẩn hóa audio input
        if isinstance(audio, np.ndarray):
            audio_input: Any = (audio.astype(np.float32), 16000)
        else:
            audio_input = audio

        with self._infer_lock:
            t0 = time.perf_counter()
            results = aligner.align(
                audio=audio_input,
                text=clean_text,
                language=language,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_items = results[0].items if results else []
        words = [
            AlignedWord(
                text=str(getattr(item, "text", "")),
                start_time=float(getattr(item, "start_time", 0.0)),
                end_time=float(getattr(item, "end_time", 0.0)),
            )
            for item in raw_items
        ]
        logger.debug(
            f"Forced alignment ({language}): {len(words)} từ trong {elapsed_ms:.1f}ms",
            extra={"module_tag": "ASR"},
        )
        return words

    @staticmethod
    def group_words_to_subtitles(
        aligned_items: List[AlignedWord],
        language: str = "English",
        max_duration_sec: float = 4.5,
        max_words: int = 10,
        max_chars: int = 45,
    ) -> List[SubtitleSentence]:
        """Gom danh sách từ/ký tự từ Forced Aligner thành các câu phụ đề ngắn vừa mắt."""
        if not aligned_items:
            return []

        is_cjk = language.lower() in ("chinese", "japanese", "korean")
        sentence_end_punct = {".", "?", "!", "。", "？", "！", "\n"}
        clause_punct = {",", "、", ";", "；", "—", "-"}

        subtitles: List[SubtitleSentence] = []
        current_words: List[AlignedWord] = []
        curr_start: Optional[float] = None

        for i, it in enumerate(aligned_items):
            w_text = it.text
            t0 = it.start_time
            t1 = it.end_time

            if curr_start is None:
                curr_start = t0
            current_words.append(it)

            curr_dur = t1 - curr_start
            clean_w = w_text.strip()
            last_char = clean_w[-1] if clean_w else ""

            is_sent_end = last_char in sentence_end_punct
            is_clause_end = (last_char in clause_punct) and (len(current_words) >= 4 or curr_dur >= 2.0)
            is_over_length = (
                curr_dur >= max_duration_sec
                or (not is_cjk and len(current_words) >= max_words)
                or (is_cjk and sum(len(w.text) for w in current_words) >= max_chars)
            )

            # Ưu tiên ngắt tại khoảng lặng âm học giữa 2 từ liên tiếp nếu đang quá dài
            has_acoustic_gap = False
            if i + 1 < len(aligned_items):
                gap = aligned_items[i + 1].start_time - t1
                if gap >= 0.18 and (len(current_words) >= 3 or curr_dur >= 1.8):
                    has_acoustic_gap = True

            should_split = is_sent_end or is_clause_end or is_over_length or (has_acoustic_gap and curr_dur >= 2.5)

            if should_split or i == len(aligned_items) - 1:
                if is_cjk:
                    seg_text = "".join(w.text for w in current_words).strip()
                else:
                    seg_text = " ".join(w.text for w in current_words).strip()

                subtitles.append(
                    SubtitleSentence(
                        text=seg_text,
                        start_time=curr_start,
                        end_time=t1,
                        words=list(current_words),
                    )
                )
                current_words = []
                curr_start = None

        return subtitles

    def unload_model(self) -> None:
        """Giải phóng ForcedAligner khỏi GPU."""
        with self._shared_lock:
            with self._infer_lock:
                self.__class__._shared_aligner = None
                self._is_warmed = False
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Đã giải phóng ForcedAligner khỏi GPU.", extra={"module_tag": "ASR"})
