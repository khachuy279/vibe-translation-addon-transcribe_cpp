"""Qwen3ForcedAligner: Engine căn chỉnh mốc thời gian từ/ký tự (Word Timestamps) cho Pipeline B.

Runtime: **Qwen3-ForcedAligner-0.6B bản GGUF (CrispASR)** — C++/ggml, chạy trong tiến trình con,
**KHÔNG cần PyTorch**. Xem `backend/utils/crispasr_native.py` để biết vì sao phải cách ly tiến
trình (xung đột `ggml*.dll` theo tên module trên Windows).

Đặc tính kỹ thuật:
- Non-Autoregressive (NAR) single-forward pass siêu tốc — đo được RTF 0,0062–0,0109 trên RTX 5060 Ti
  (khối 28 s mất ~195–307 ms).
- Hỗ trợ 11 ngôn ngữ (Anh, Nhật, Trung, Pháp, Đức, Tây Ban Nha, Ý, Nga, Hàn...). Cách tách đơn vị
  do CrispASR quyết định theo CHỮ VIẾT (CJK/Hangul từng ký tự, Latin theo khoảng trắng).
- Thread-safe singleton.
- Prewarm lúc khởi động để triệt tiêu độ trễ nạp model + warmup CUDA graph (~0,13–0,84 s).
- Tích hợp bộ gom câu phụ đề theo dấu câu và độ dài (Punctuation-guided Sentence Grouping).
"""

from __future__ import annotations

import gc
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger("asr.forced_aligner")

# ══════════════════════════════════════════════════════════════════════════════════════════
# KHÔNG còn `torch` / `transformers` / `qwen_asr` trong file này.
#
# LỊCH SỬ: tới Giai đoạn 2, aligner có hai đường — `transformers` + PyTorch (đọc snapshot
# safetensors) và CrispASR GGUF. Giai đoạn 3 (2026-10-06) đã XOÁ hẳn đường torch cùng cây
# vendored `backend/asr/qwen_asr/`, vì VAD đã chuyển sang onnxruntime nên đây là thứ cuối cùng
# còn kéo PyTorch (~2,7 GB đĩa + 1–2 s khởi động) trong khi bản GGUF cho kết quả tương đương.
# ══════════════════════════════════════════════════════════════════════════════════════════

# Đảm bảo đường dẫn tới `backend/asr` được nạp (một số công cụ import trực tiếp module con).
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

REPO_ROOT = CURRENT_DIR.parent.parent

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


def detect_aligner_language_from_text(text: str) -> Optional[str]:
    """Đoán ngôn ngữ cho ForcedAligner từ CHÍNH văn bản ASR (dùng khi `source_lang = "auto"`).

    VÌ SAO CẦN (sự cố thật 2026-10-03, "Pipeline B không ngắt câu dài" trên trang tiếng Nhật):
    `source_lang = "auto"` ⇒ `resolve_aligner_language("auto")` trả `None` ⇒ tầng gọi rơi về
    `"English"`. Upstream `encode_timestamp()` CHỈ tách từ riêng cho `japanese`
    (`tokenize_japanese`) và `korean`; mọi ngôn ngữ khác dùng `tokenize_space_lang` — mà tiếng
    Nhật/Trung **không có khoảng trắng**, nên CẢ CÂU thành **MỘT token** ⇒ aligner trả về một
    item duy nhất ⇒ tầng gom câu không có gì để ngắt ⇒ **cả khối 3 câu hiện thành một phụ đề
    dài**. YouTube tiếng Anh không dính lỗi này vì `tokenize_space_lang` đúng cho tiếng Anh.

    Đây chỉ là phép đoán theo CHỮ VIẾT (đủ để chọn nhánh tokenize của aligner, không phải nhận
    diện ngôn ngữ đầy đủ): có Hangul ⇒ Korean, có Kana ⇒ Japanese, chỉ có Hán tự ⇒ Chinese.
    """
    if not text:
        return None
    kana = hangul = han = 0
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
            kana += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            hangul += 1
        elif 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
            han += 1
    if hangul:
        return "Korean"
    if kana:
        return "Japanese"
    if han:
        return "Chinese"
    return None


#: Dấu KẾT CÂU dùng cho tầng ngắt phụ đề bằng VĂN BẢN (khớp `group_words_to_subtitles`).
_SENTENCE_END_CHARS = "。！？!?…；;"
#: Ký tự ĐÓNG đi liền sau dấu kết câu phải ở lại cùng câu (ngoặc/nháy Nhật–Trung–Latin).
_SENTENCE_CLOSERS = "」』）)]}〉》\"'”’"
#: Ngôn ngữ viết KHÔNG có khoảng trắng ⇒ đo độ dài bằng KÝ TỰ thay vì TỪ.
_NO_SPACE_LANGS = ("japanese", "chinese", "korean", "cantonese")

#: TỪ VIẾT TẮT thường gặp: dấu `.` đi kèm KHÔNG phải kết câu.
#:
#: SỰ CỐ THẬT 2026-10-05 (log người dùng): ASR trả về
#:   'Fine, Dr. Kuthrapali. Thank you, sir. … I present Dr. Milstone from MIT.'
#: và tầng "Ngắt câu khối" chẻ thành 15 phụ đề vụn: "Fine, Dr." ⏐ "Kuthrapali." ⏐ … ⏐ "I present Dr."
#: ⏐ "Milstone from MIT." — vừa sai phụ đề vừa sai bản dịch.
#: Danh sách dưới đây CỐ Ý bỏ các từ mơ hồ hay gặp ở dạng từ thường ("no", "am", "pm", "us").
_ABBREVIATIONS_CI = frozenset({
    "dr", "mr", "mrs", "ms", "miss", "prof", "sr", "jr", "st", "mt", "rev", "hon",
    "gen", "col", "capt", "lt", "sgt", "cmdr", "adm", "gov", "sen", "rep", "pres",
    "vs", "etc", "eg", "ie", "al", "cf", "approx", "dept", "est", "inc", "ltd",
    "co", "corp", "univ", "fig", "vol", "pp", "ed", "eds",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "phd", "md", "ba", "ma", "bsc", "msc",
})


def _word_before_trailing_period(text: str) -> str:
    """Từ NGAY TRƯỚC dấu `.` cuối cùng của `text` (rỗng nếu `text` không kết thúc bằng `.`)."""
    t = (text or "").rstrip()
    if not t.endswith("."):
        return ""
    out = ""
    for ch in reversed(t[:-1]):
        if ch.isalnum() or ch in "&'’-":
            out = ch + out
        else:
            break
    return out


def _ends_with_abbreviation(text: str) -> bool:
    """`text` kết thúc bằng TỪ VIẾT TẮT kèm dấu chấm ⇒ dấu chấm đó KHÔNG kết câu.

    Nhận hai dạng:
      * từ viết tắt thông dụng: `Dr.` `Mr.` `Prof.` `St.` `Ph.D.`…
      * chữ cái đầu / viết tắt từng chữ: `J.` `A.` `U.S.` `A.B.`
        (dạng này nhận qua quy tắc "từ cuối chỉ có MỘT ký tự và là chữ HOA").
    """
    word = _word_before_trailing_period(text)
    if not word:
        return False
    if word.lower() in _ABBREVIATIONS_CI:
        return True
    return len(word) == 1 and word.isupper()


def _split_text_by_sentence(text: str) -> List[str]:
    """Cắt văn bản thành các CÂU theo dấu kết câu (giữ dấu ở cuối mảnh, gộp ngoặc/nháy đóng)."""
    if not text:
        return []
    pieces: List[str] = []
    buf = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf += ch
        i += 1
        if ch in _SENTENCE_END_CHARS or ch == ".":
            # Không tách nếu là số thập phân: ví dụ 3.14 hay 152.39s
            if ch == "." and i < n and text[i].isdigit() and len(buf) >= 2 and buf[-2].isdigit():
                continue
            # Không tách nếu là từ viết liền / tên miền / viết tắt chưa hết: domain.com
            if ch == "." and i < n and (text[i].isalpha() and not text[i].isspace()):
                continue
            # Không tách tại TỪ VIẾT TẮT khi phía sau CÒN chữ: "Dr. Kuthrapali", "J. K. Rowling",
            # "U.S. Navy"… (sự cố thật 2026-10-05: "Fine, Dr." | "Kuthrapali." — xem
            # `_ABBREVIATIONS_CI`). Nếu viết tắt nằm ở CUỐI văn bản thì dấu chấm đó vẫn kết câu.
            if ch == "." and text[i:].strip() and _ends_with_abbreviation(buf):
                continue
            while i < n and text[i] in _SENTENCE_CLOSERS:
                buf += text[i]
                i += 1
            while i < n and (text[i] in _SENTENCE_END_CHARS or text[i] == "."):
                buf += text[i]
                i += 1
            if buf.strip():
                pieces.append(buf.strip())
            buf = ""
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def _hard_split_piece(text: str, language: str, max_chars: int, max_words: int) -> List[str]:
    """Cắt cứng một mảnh KHÔNG có dấu kết câu nào (ASR không sinh dấu câu)."""
    if not text:
        return []
    if str(language or "").strip().lower() in _NO_SPACE_LANGS:
        limit = int(max_chars or 0)
        if limit <= 0 or len(text) <= limit:
            return [text]
        return [text[i:i + limit] for i in range(0, len(text), limit)]
    limit = int(max_words or 0)
    words = text.split()
    if limit <= 0 or len(words) <= limit:
        return [text]
    return [" ".join(words[i:i + limit]) for i in range(0, len(words), limit)]


def _allocate_spans(
    pieces: List[str],
    start_time: float,
    end_time: float,
) -> List[tuple]:
    """Chia khoảng [start, end] cho các mảnh theo TỈ LỆ ĐỘ DÀI VĂN BẢN.

    Tỉ lệ ký tự ≈ tỉ lệ thời gian nói (đủ tốt cho một lưới an toàn khi KHÔNG có mốc từ); mốc
    cuối luôn đúng bằng `end_time` để không lệch dần.
    """
    weights = [max(1, len(p)) for p in pieces]
    total = float(sum(weights)) or 1.0
    base = float(start_time)
    dur = float(end_time) - base
    if dur <= 0.2 * len(pieces):
        dur = 0.2 * len(pieces)
    spans = []
    t = base
    acc = 0.0
    for idx, (piece, w) in enumerate(zip(pieces, weights)):
        acc += w
        en = base + dur if idx == len(pieces) - 1 else base + dur * (acc / total)
        if en <= t:
            en = t + 0.2
        spans.append((piece, t, en))
        t = en
    return spans


def is_aligner_kept_char(ch: str) -> bool:
    """Ký tự mà `Qwen3ForcedAligner` GIỮ LẠI khi tokenize (xem `is_kept_char` của upstream).

    Upstream (`qwen_asr/inference/qwen3_forced_aligner.py`) chỉ giữ category Unicode `L` (chữ) và
    `N` (số), cộng thêm dấu nháy đơn `'`. **Mọi dấu câu bị loại bỏ** — đây là lý do `AlignedWord.text`
    không bao giờ có `.`/`,`/`?`/`。` và tầng gom câu không thể ngắt theo dấu câu nếu không gắn lại.
    """
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


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


class ForcedAlignerService:
    """Singleton căn chỉnh mốc thời gian bằng **Qwen3-ForcedAligner-0.6B bản GGUF (CrispASR)**.

    ⚠️ Sau Giai đoạn 3 (2026-10-06) chỉ còn MỘT runtime: GGUF Q4_K chạy trên CrispASR (C++/ggml)
    trong TIẾN TRÌNH CON — xem `backend/utils/crispasr_native.py`. Đường `transformers` + PyTorch
    đã bị XOÁ hoàn toàn cùng cây vendored `backend/asr/qwen_asr/`; lý do: VAD đã chuyển sang
    onnxruntime nên aligner là thứ cuối cùng còn kéo PyTorch (~2,7 GB đĩa + 1–2 s khởi động) trong
    khi bản GGUF cho kết quả tương đương (cùng số câu, cùng nội dung phụ đề) và tốc độ ngang GPU.

    Tầng này giữ nguyên: `AlignedWord`, `SubtitleSentence`, `merge_source_text`,
    `group_words_to_subtitles`, `split_oversized_sentences`, `split_subtitles_by_clause_comma`.
    """

    _instance: Optional["ForcedAlignerService"] = None
    _shared_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "ForcedAlignerService":
        with cls._shared_lock:
            if cls._instance is None:
                cls._instance = ForcedAlignerService()
            return cls._instance

    def __init__(self, model_path: Optional[Union[str, Path]] = None, repo_id: str = ""):
        # `model_path`/`repo_id` giữ lại chỉ để tương thích chữ ký cũ (test/công cụ). Đường GGUF
        # lấy vị trí model từ `backend/utils/crispasr_native.py` (có env override riêng).
        self.model_path = Path(model_path) if model_path else None
        self.repo_id = repo_id
        self._is_warmed = False

    # ------------------------------------------------------------------ vòng đời
    @staticmethod
    def _client():
        from backend.asr.crispasr_aligner import CrispASRAlignerClient

        return CrispASRAlignerClient.get_instance()

    @staticmethod
    def is_ready() -> Tuple[bool, str]:
        """Runtime căn chỉnh đã dùng được chưa: `(ok, lý do)`."""
        from backend.utils.crispasr_native import available

        return available()

    def prewarm(self) -> None:
        """Nạp model GGUF + warmup CUDA graph trong tiến trình con (blocking).

        Đo được 127–839 ms — nhanh hơn 1,6 s nạp + 0,5 s prewarm của đường torch trước đây.
        """
        try:
            t0 = time.perf_counter()
            self._client().prewarm()
            self._is_warmed = True
            logger.info(
                f"Prewarm ForcedAligner (CrispASR GGUF) hoàn tất trong {time.perf_counter() - t0:.2f}s.",
                extra={"module_tag": "ASR"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Prewarm ForcedAligner lỗi (bỏ qua): {exc}", extra={"module_tag": "ASR"})

    def align(
        self,
        audio: Union[np.ndarray, Tuple[np.ndarray, int], str],
        text: str,
        language: str = "English",  # noqa: ARG002 - giữ chữ ký cũ; GGUF tự nhận diện theo chữ viết
        attach_source_punctuation: bool = True,
    ) -> List[AlignedWord]:
        """Căn chỉnh thời gian từng đơn vị.

        Args:
            audio: numpy array 16 kHz float32, hoặc tuple `(pcm, sr)`, hoặc đường dẫn file.
            text: Văn bản nhận dạng từ ASR.
            language: GIỮ LẠI CHO TƯƠNG THÍCH. Bản GGUF của CrispASR tự quyết định cách tách đơn vị
                theo **chữ viết** (CJK/Hangul tách từng ký tự, hệ Latin tách theo khoảng trắng) chứ
                không nhận tên ngôn ngữ như upstream dùng `nagisa`/`soynlp`.
            attach_source_punctuation: Gắn DẤU CÂU của `text` trở lại từng đơn vị (MẶC ĐỊNH BẬT).

        Returns:
            Danh sách `AlignedWord(text, start_time, end_time)`.

        Raises:
            RuntimeError: runtime CrispASR chưa sẵn sàng (thiếu DLL hoặc model GGUF).
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return []

        ok, reason = self.is_ready()
        if not ok:
            raise RuntimeError(
                f"ForcedAligner chưa sẵn sàng: {reason}\n"
                f"  • Model GGUF (~500 MB) sẽ được tự tải ở lần chạy đầu; hoặc tải thủ công từ "
                f"`https://huggingface.co/cstr/qwen3-forced-aligner-0.6b-GGUF`.\n"
                f"  • DLL CrispASR nằm ở `backend/bin/crispasr/` (xem `backend/utils/crispasr_native.py`)."
            )

        pcm = self._normalize_audio(audio)

        t0 = time.perf_counter()
        raw = self._client().align(pcm, clean_text, 0.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        words = [
            AlignedWord(
                text=str(it.get("text", "")),
                start_time=float(it.get("t0", 0.0)),
                end_time=float(it.get("t1", 0.0)),
            )
            for it in raw
        ]
        # ⚠️ BẮT BUỘC: model BỎ HẾT dấu câu khi tokenize (xem `is_aligner_kept_char`), nên nếu
        # không gắn lại thì tầng gom câu không thấy dấu câu nào ⇒ phụ đề bị chẻ theo trần từ và
        # thiếu cả dấu câu (sự cố 2026-10-02). CrispASR ĐÃ tự gắn dấu câu vào đơn vị hiển thị
        # (`tokenise_display_words` + `restore_text`), nhưng `merge_source_text` khớp theo chuỗi ký
        # tự "được giữ" nên idempotent — giữ lại để hành vi phụ đề không đổi.
        if attach_source_punctuation:
            words = ForcedAlignerService.merge_source_text(words, clean_text)
        logger.debug(
            f"Căn chỉnh (CrispASR GGUF): {len(words)} đơn vị trong {elapsed_ms:.1f}ms",
            extra={"module_tag": "ASR"},
        )
        return words

    @staticmethod
    def _normalize_audio(audio: Union[np.ndarray, Tuple[np.ndarray, int], str]) -> np.ndarray:
        """Đưa đầu vào về PCM float32 mono 16 kHz (đường tiện lợi cho công cụ/CLI)."""
        if isinstance(audio, str):
            import soundfile as sf

            data, sr = sf.read(audio, dtype="float32", always_2d=True)
            pcm = data[:, 0]
            if sr != 16000:
                from math import gcd

                from scipy.signal import resample_poly

                g = gcd(int(sr), 16000)
                pcm = resample_poly(pcm, 16000 // g, int(sr) // g).astype(np.float32)
        elif isinstance(audio, tuple):
            pcm, sr = audio[0], int(audio[1])
            if sr != 16000:
                raise ValueError(f"ForcedAligner cần 16 kHz, nhận {sr} Hz")
        else:
            pcm = audio
        arr = np.asarray(pcm, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[:, 0]
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def merge_source_text(
        aligned_items: List[AlignedWord],
        source_text: str,
    ) -> List[AlignedWord]:
        """Gắn lại DẤU CÂU (và chính tả gốc) của văn bản ASR vào từng từ đã căn chỉnh.

        VÌ SAO CẦN: `Qwen3-ForcedAligner` loại bỏ mọi ký tự không phải chữ/số khi tokenize
        (`is_kept_char` của upstream chỉ giữ category `L`/`N` + dấu nháy đơn) ⇒ `item.text` KHÔNG
        BAO GIỜ có `.`/`,`/`?`/`。`. Bằng chứng: `report/09_lookahead_offline_batch/phase1_poc_results.md`
        — ASR trả `"私の人生の中心で最も魅力的だ。"` nhưng item cuối là `"魅力的だ"` và phụ đề gom ra
        `"私の人生の中心で最も魅力的だ"` (mất `。`).

        Hệ quả nếu không gắn lại: tầng gom câu chỉ còn cách cắt theo trần `max_words`/`max_duration`
        ⇒ chẻ giữa câu, phụ đề thiếu dấu câu, và **bản dịch nhận được mảnh vụn** thay vì câu trọn vẹn.

        CÁCH LÀM: duyệt văn bản gốc và danh sách từ SONG SONG, tiêu thụ đúng số ký tự "được giữ" của
        từng từ, rồi lấy ĐOẠN VĂN BẢN GỐC từ ký tự đầu của từ này tới ký tự đầu của từ kế tiếp. Nhờ
        vậy: dấu câu dính sau từ được giữ, dấu nháy/gạch BÊN TRONG từ (`It’s`) cũng được giữ, và
        từ/câu trả về khớp **nguyên văn** văn bản ASR.

        Hoạt động cho cả tiếng Latin (tách theo khoảng trắng) lẫn CJK/Nhật/Hàn (tách theo ký tự/từ),
        vì việc khớp chỉ dựa trên chuỗi ký tự "được giữ".
        """
        if not aligned_items or not source_text:
            return aligned_items

        kept_positions = [i for i, ch in enumerate(source_text) if is_aligner_kept_char(ch)]
        if not kept_positions:
            return aligned_items
        kept_chars = [source_text[i] for i in kept_positions]

        spans: List[Optional[Tuple[int, int]]] = []
        pos = 0
        for item in aligned_items:
            need = "".join(ch for ch in (item.text or "") if is_aligner_kept_char(ch))
            if not need:
                spans.append(None)
                continue
            window_len = len(need) + 16  # cho phép aligner bỏ sót/thêm token
            window = "".join(kept_chars[pos : pos + window_len])
            found = window.find(need)
            if found < 0:
                spans.append(None)
                continue
            start_k = pos + found
            end_k = start_k + len(need) - 1
            if end_k >= len(kept_positions):
                spans.append(None)
                continue
            spans.append((kept_positions[start_k], kept_positions[end_k]))
            pos = end_k + 1

        total = len(source_text)
        raw_pieces: List[Tuple[AlignedWord, str]] = []
        for i, item in enumerate(aligned_items):
            span = spans[i]
            if span is None:
                raw_pieces.append((item, ""))
                continue
            # Lấy ĐOẠN GỐC từ ký tự đầu của từ này tới hết các ký tự KHÔNG-được-giữ ngay sau ký tự
            # cuối của chính nó (dấu câu + khoảng trắng). Dừng ngay khi gặp ký tự chữ/số kế tiếp ⇒
            # KHÔNG nuốt các từ mà aligner bỏ sót (nếu nuốt, từ cuối sẽ ôm trọn phần văn bản còn lại
            # trong khi mốc thời gian của nó chỉ phủ một từ).
            end_char = span[1] + 1
            while end_char < total and not is_aligner_kept_char(source_text[end_char]):
                end_char += 1
            raw_slice = source_text[span[0] : end_char]
            text = raw_slice.strip()
            raw_pieces.append((
                AlignedWord(
                    text=text or item.text,
                    start_time=item.start_time,
                    end_time=item.end_time,
                ),
                raw_slice,
            ))

        # Gộp các token số bị chẻ bởi dấu chấm thập phân hoặc dấu phẩy hàng nghìn không có khoảng trắng
        # Ví dụ: "3." và "9" trong "星3.9でさ" -> "3.9" (raw_slice là "3.", không có khoảng trắng sau dấu chấm)
        # Và gộp các từ ghép có dấu gạch nối (hyphen) không có khoảng trắng:
        # Ví dụ: "deep-" và "cycle" -> "deep-cycle", "sixty-" và "one" -> "sixty-one"
        merged: List[AlignedWord] = []
        for idx, (item, raw_slice) in enumerate(raw_pieces):
            if not merged:
                merged.append(item)
                continue
            prev = merged[-1]
            prev_t = prev.text or ""
            cur_t = item.text or ""
            prev_raw = raw_pieces[idx - 1][1] if idx - 1 < len(raw_pieces) else ""
            if (
                len(prev_t) >= 2
                and prev_t[-2].isdigit()
                and prev_t[-1] in (".", ",")
                and len(cur_t) >= 1
                and cur_t[0].isdigit()
                and not any(c.isspace() for c in prev_raw)
            ):
                merged[-1] = AlignedWord(
                    text=prev_t + cur_t,
                    start_time=prev.start_time,
                    end_time=item.end_time,
                )
            elif (
                len(prev_t) > 1
                and prev_t.endswith("-")
                and not prev_t.endswith("--")
                and len(cur_t) > 0
                and not any(c.isspace() for c in prev_raw)
            ):
                merged[-1] = AlignedWord(
                    text=prev_t + cur_t,
                    start_time=prev.start_time,
                    end_time=item.end_time,
                )
            else:
                merged.append(item)

        return merged

    @staticmethod
    def group_words_to_subtitles(
        aligned_items: List[AlignedWord],
        language: str = "English",
        max_duration_sec: float = 4.5,
        max_words: int = 10,
        max_chars: int = 45,
        min_words: int = 2,
        defer_words: int = 3,
        merge_continuation: bool = True,
        sentence_max_words: int = 24,
        sentence_max_duration_sec: float = 8.0,
        sentence_max_chars: int = 160,
    ) -> List[SubtitleSentence]:
        """Gom danh sách từ/ký tự từ Forced Aligner thành các câu phụ đề ngắn vừa mắt.

        ĐÂY LÀ TẦNG NGẮT CÂU HIỂN THỊ của tuyến OFFLINE_BATCH — nó chạy TRÊN bản phiên âm đã có
        trọn ngữ cảnh của cả khối (khác hẳn Pipeline A, nơi ranh giới câu quyết định luôn cửa sổ
        audio đưa vào model). Vì vậy mọi quy tắc ở đây chỉ được ảnh hưởng tới việc CHIA HIỂN THỊ,
        tuyệt đối không được cắt bớt ngữ cảnh âm học mà ASR đã nghe.

        Quy tắc (theo thứ tự ưu tiên):
          1. Dấu KẾT CÂU (`.?!。？！`) — **cả câu là một phụ đề**, miễn câu không vượt
             `sentence_max_*` (24 từ / 8 s). Đây là mặc định mong muốn: người xem thấy câu trọn vẹn
             và bản dịch cũng nhận được câu trọn vẹn.
          2. Chỉ khi câu QUÁ DÀI mới cắt bên trong: ưu tiên dấu PHẨY cuối cùng trong cửa sổ
             `max_words`/`max_duration_sec`, nếu không có thì tại KHE ÂM HỌC lớn nhất (>= 120 ms),
             cuối cùng mới cắt tại trần.
          3. GỘP các mảnh bị chẻ: mảnh kết thúc KHÔNG có dấu kết câu mà mảnh sau bắt đầu bằng chữ
             thường (hoặc quá ngắn) ⇒ hai mảnh vốn là MỘT câu ⇒ gộp, miễn không vượt trần cứng.

        ⚠️ ĐIỀU KIỆN TIÊN QUYẾT: dấu câu phải TỒN TẠI trong `aligned_items`. `Qwen3-ForcedAligner`
        đã bỏ hết dấu câu khi tokenize, nên BẮT BUỘC gọi `ForcedAlignerService.merge_source_text`
        (được `align()` gọi tự động) trước khi gom câu. Không có bước đó thì hàm này chỉ còn cách
        cắt theo trần và phụ đề sẽ là những mảnh cụt không dấu câu — đúng sự cố 2026-10-02.
        """
        if not aligned_items:
            return []

        n = len(aligned_items)
        is_cjk = language.lower() in ("chinese", "japanese", "korean")
        sentence_end_punct = {".", "?", "!", "。", "？", "！", "\n"}
        clause_punct = {",", "、", ";", "；", "—", "-"}

        min_words = max(1, int(min_words))
        defer = max(0, int(defer_words))
        #: Trần cho một CÂU TRỌN VẸN (ưu tiên số 1). Chỉ khi câu dài hơn mức này mới phải cắt trong.
        sentence_max_words = max(int(max_words), int(sentence_max_words))
        sentence_max_duration_sec = max(float(max_duration_sec), float(sentence_max_duration_sec))
        sentence_max_chars = max(int(max_chars), int(sentence_max_chars))
        #: Trần CỨNG cho bước gộp: chỉ để CỨU một mảnh bị chẻ (không dựng lại "siêu câu" và không
        #: phá ý định của caller khi họ đặt trần mềm nhỏ). Luôn <= trần câu.
        hard_words = min(max_words + defer, sentence_max_words)
        hard_chars = min(int(max_chars * 1.5), int(sentence_max_chars * 1.25))
        hard_dur = min(max_duration_sec * 1.5, sentence_max_duration_sec * 1.25)
        #: Khe âm học tối thiểu để coi là ranh giới tự nhiên khi buộc phải cắt.
        min_acoustic_gap_sec = 0.12

        def _text_at(i: int) -> str:
            return (aligned_items[i].text or "").strip()

        def _last_char(i: int) -> str:
            t = _text_at(i)
            return t[-1] if t else ""

        def _is_sent_end(i: int) -> bool:
            ch = _last_char(i)
            if ch not in sentence_end_punct:
                return False
            # Không coi là kết câu nếu là số thập phân (ví dụ: "3." và token sau bắt đầu bằng số "9")
            if ch == ".":
                t = _text_at(i)
                if len(t) >= 2 and t[-2].isdigit():
                    if i + 1 < n and _text_at(i + 1)[:1].isdigit():
                        return False
                elif t == ".":
                    if i > 0 and _text_at(i - 1)[-1:].isdigit() and i + 1 < n and _text_at(i + 1)[:1].isdigit():
                        return False
                # ── TỪ VIẾT TẮT / CHỮ CÁI ĐẦU KHÔNG KẾT CÂU ────────────────────────
                # SỰ CỐ THẬT 2026-10-05: 'Fine, Dr. Kuthrapali. … I present Dr. Milstone from MIT.'
                # bị chẻ thành "Fine, Dr." ⏐ "Kuthrapali." ⏐ … ⏐ "I present Dr." ⏐ "Milstone from MIT."
                # Chỉ bỏ qua khi PHÍA SAU còn token: viết tắt nằm ở cuối khối thì vẫn là kết câu.
                if i + 1 < n:
                    window = t if len(t) > 1 else (
                        (_text_at(i - 1) if i > 0 else "") + "."
                    )
                    if _ends_with_abbreviation(window):
                        return False
            return True

        def _is_clause_end(i: int) -> bool:
            ch = _last_char(i)
            if ch not in clause_punct:
                return False
            # Không coi là ngắt vế nếu là dấu gạch nối trong từ ghép (ví dụ: "deep-", "sixty-")
            if ch == "-":
                t = _text_at(i)
                if not (t == "-" or t.endswith("--")):
                    return False
            # Không coi là ngắt vế nếu là dấu phẩy phân cách hàng nghìn / số thập phân (ví dụ: "1,000")
            if ch == ",":
                t = _text_at(i)
                if len(t) >= 2 and t[-2].isdigit():
                    if i + 1 < n and _text_at(i + 1)[:1].isdigit():
                        return False
                elif t == ",":
                    if i > 0 and _text_at(i - 1)[-1:].isdigit() and i + 1 < n and _text_at(i + 1)[:1].isdigit():
                        return False
            return True

        def _gap_after(i: int) -> float:
            """Khe âm học giữa từ `i` và từ `i + 1`."""
            if i + 1 >= n:
                return 0.0
            return max(0.0, aligned_items[i + 1].start_time - aligned_items[i].end_time)

        def _chars(lo: int, hi: int) -> int:
            return sum(len(aligned_items[k].text) for k in range(lo, hi + 1))

        def _count(lo: int, hi: int) -> int:
            return hi - lo + 1

        def _dur(lo: int, hi: int) -> float:
            return max(0.0, aligned_items[hi].end_time - aligned_items[lo].start_time)

        def _over(lo: int, hi: int) -> bool:
            """Nhóm [lo, hi] đã vượt trần MỀM chưa (trần dùng khi phải cắt BÊN TRONG câu)."""
            if _dur(lo, hi) >= max_duration_sec:
                return True
            if is_cjk:
                return _chars(lo, hi) >= max_chars
            return _count(lo, hi) >= max_words

        def _over_sentence(lo: int, hi: int) -> bool:
            """Nhóm [lo, hi] đã vượt trần của một CÂU TRỌN VẸN chưa."""
            if _dur(lo, hi) >= sentence_max_duration_sec:
                return True
            if is_cjk:
                return _chars(lo, hi) >= sentence_max_chars
            return _count(lo, hi) >= sentence_max_words

        def _too_big(lo: int, hi: int) -> bool:
            """Nhóm [lo, hi] vượt trần CỨNG (dùng cho bước gộp) chưa."""
            if _dur(lo, hi) > hard_dur:
                return True
            if is_cjk:
                return _chars(lo, hi) > hard_chars
            return _count(lo, hi) > hard_words

        def _choose_cut(lo: int, cap_end: int) -> int:
            """Chọn từ kết thúc nhóm khi không có dấu kết câu nào trong tầm với."""
            # Sàn cắt trong câu: tối thiểu 2 từ nếu cửa sổ cho phép (tránh cắt mảnh mồ côi 1 từ như "I")
            floor = min(cap_end, lo + max(2, min_words) - 1)
            # (a) Dấu phẩy CUỐI CÙNG trong cửa sổ (giữ cụm từ dài nhất có thể mà vẫn dưới trần)
            for k in range(cap_end, floor - 1, -1):
                if _is_clause_end(k):
                    return k
            # (b) Khe âm học LỚN NHẤT (người nói lấy hơi) — không cắt giữa cụm từ
            best_k: Optional[int] = None
            best_gap = 0.0
            for k in range(floor, cap_end):
                gap = _gap_after(k)
                if gap > best_gap:
                    best_k, best_gap = k, gap
            if best_k is not None and best_gap >= min_acoustic_gap_sec:
                return best_k
            # (c) Trần
            return cap_end

        # ── BƯỚC 1: chọn ranh giới từng nhóm
        groups: List[Tuple[int, int]] = []
        start = 0
        while start < n:
            hard_first: Optional[int] = None
            for i in range(start, n):
                if _is_sent_end(i):
                    hard_first = i
                    break

            # Trần MỀM: mốc phải cắt nếu không tìm được dấu kết câu nào.
            cap_end = n - 1
            for i in range(start, n):
                if _over(start, i):
                    cap_end = i
                    break

            # Trần CÂU: một câu trọn vẹn được phép dài tới đây (ưu tiên số 1).
            sentence_cap_end = n - 1
            for i in range(start, n):
                if _over_sentence(start, i):
                    sentence_cap_end = i
                    break

            if hard_first is not None and hard_first <= sentence_cap_end:
                # Có dấu kết câu trong phạm vi một câu bình thường ⇒ lấy TRỌN CÂU.
                end = hard_first
            elif hard_first is not None and hard_first <= min(n - 1, cap_end + defer):
                # Câu hơi dài nhưng sắp có dấu kết câu ⇒ nới thêm cho trọn câu.
                end = hard_first
            else:
                # Không có dấu câu (hoặc câu quá dài) ⇒ cắt bên trong câu tại điểm đẹp nhất.
                end = _choose_cut(start, cap_end)
            groups.append((start, end))
            start = max(end + 1, start + 1)

        # ── BƯỚC 2: gộp các mảnh bị chẻ (câu bị cắt làm hai phụ đề)
        if merge_continuation and len(groups) > 1:
            merged: List[Tuple[int, int]] = [groups[0]]
            for lo, hi in groups[1:]:
                p_lo, p_hi = merged[-1]
                prev_open = not _is_sent_end(p_hi)
                first_txt = _text_at(lo)
                cur_short = _count(lo, hi) < min_words
                cur_cont = cur_short or (not is_cjk and first_txt[:1].islower())
                if prev_open and cur_cont and not _too_big(p_lo, hi):
                    merged[-1] = (p_lo, hi)
                else:
                    merged.append((lo, hi))
            groups = merged

        # ── BƯỚC 3: dựng SubtitleSentence
        subtitles: List[SubtitleSentence] = []
        for lo, hi in groups:
            words = list(aligned_items[lo : hi + 1])
            if is_cjk:
                seg_text = "".join(w.text for w in words).strip()
            else:
                seg_text = " ".join(w.text for w in words).strip()
            if not seg_text:
                continue
            subtitles.append(
                SubtitleSentence(
                    text=seg_text,
                    start_time=float(aligned_items[lo].start_time),
                    end_time=float(aligned_items[hi].end_time),
                    words=words,
                )
            )
        return subtitles

    @staticmethod
    def split_oversized_sentences(
        subtitles: List[SubtitleSentence],
        language: str = "English",
        max_chars: int = 0,
        max_words: int = 0,
    ) -> List[SubtitleSentence]:
        """Lưới AN TOÀN: tách mọi phụ đề đang CHỨA NHIỀU CÂU thành từng câu.

        VÌ SAO CẦN (sự cố thật 2026-10-03 — "Pipeline B không ngắt câu dài", trang tiếng Nhật):
        tầng gom câu (`group_words_to_subtitles`) chỉ ngắt được theo dấu câu khi `aligned_items`
        có NHIỀU item. Hai đường làm nó mất khả năng đó:
          1. Aligner trả về ĐÚNG MỘT item cho cả khối (chọn sai ngôn ngữ — xem
             `detect_aligner_language_from_text`), hoặc aligner LỖI ⇒ nhánh dự phòng lấy cả khối
             làm một câu.
          2. ASR không sinh dấu câu ⇒ không có ranh giới nào để ngắt.
        Khi đó cả khối 3 câu hiện thành MỘT phụ đề dài, tràn 2 dòng và bản dịch cũng thành một
        khối dài. Hàm này không cần mốc từ: ngắt theo dấu kết câu rồi phân bổ thời gian theo tỉ lệ
        độ dài văn bản; mảnh nào vẫn quá dài (không có dấu câu) thì cắt cứng theo trần ký tự/từ.
        """
        if not subtitles:
            return subtitles
        out: List[SubtitleSentence] = []
        for sub in subtitles:
            text = (sub.text or "").strip()
            if not text:
                continue
            expanded: List[str] = []
            for piece in _split_text_by_sentence(text):
                expanded.extend(_hard_split_piece(piece, language, max_chars, max_words))
            if len(expanded) <= 1:
                out.append(sub)
                continue
            for piece, st, en in _allocate_spans(expanded, sub.start_time, sub.end_time):
                out.append(SubtitleSentence(text=piece, start_time=st, end_time=en))
        return out

    @staticmethod
    def split_subtitles_by_clause_comma(
        subtitles: List[SubtitleSentence],
        language: str = "English",
        min_words: int = 4,
    ) -> List[SubtitleSentence]:
        """Tách các câu phụ đề dài tại dấu ngắt vế câu (、 hoặc , hoặc ;) khi số từ phía trước >= min_words.

        Quy tắc:
        1. Duyệt qua từng SubtitleSentence: nếu có dấu ngắt vế (、, ,, ;) và tích lũy >= min_words
           từ/ký tự, đồng thời phần còn lại phía sau có ít nhất 2 từ/token thì tách thành phụ đề riêng.
        2. Nếu câu có `words` (từ Forced Aligner): mốc thời gian start_time và end_time được
           lấy chính xác 100% từ mốc âm học của các từ trong vế đó.
        3. Nếu không có `words`: phân bổ thời gian theo tỉ lệ độ dài ký tự qua `_allocate_spans`.
        """
        if not subtitles or min_words <= 0:
            return subtitles

        is_cjk = str(language or "").strip().lower() in ("japanese", "chinese", "korean", "cantonese")
        clause_delims = {",", "、", ";", "；"}

        def count_tokens(s: str) -> int:
            clean = re.sub(r"[,、;；\.?!。？！\s]", "", s)
            if not is_cjk and " " in s:
                return len(s.split())
            return len(re.findall(r"[a-zA-Z0-9]+|[^\s\W\d_]", clean))

        out: List[SubtitleSentence] = []
        for sub in subtitles:
            text = (sub.text or "").strip()
            if not text:
                continue

            # Kiểm tra nhanh: nếu câu không chứa bất kỳ dấu ngắt vế nào thì giữ nguyên
            if not any(d in text for d in clause_delims):
                out.append(sub)
                continue

            # ── Nhánh 1: Có mốc từ chính xác từ Forced Aligner
            if sub.words and len(sub.words) > 1:
                res = []
                buf: List[AlignedWord] = []
                buf_tokens = 0
                for i, w in enumerate(sub.words):
                    buf.append(w)
                    buf_tokens += count_tokens(w.text)

                    w_text = (w.text or "").strip()
                    ends_with_comma = any(w_text.endswith(d) for d in clause_delims)
                    # Bỏ qua nếu dấu phẩy nằm trong số (ví dụ: "1," theo sau là "000")
                    if ends_with_comma and w_text.endswith(","):
                        if len(w_text) >= 2 and w_text[-2].isdigit():
                            if i + 1 < len(sub.words) and (sub.words[i + 1].text or "").strip()[:1].isdigit():
                                ends_with_comma = False
                    remaining_words = sub.words[i + 1:]
                    remaining_tokens = sum(count_tokens(rw.text) for rw in remaining_words)

                    if ends_with_comma and buf_tokens >= min_words and remaining_tokens >= 2:
                        joined_text = ("" if is_cjk else " ").join(bw.text for bw in buf).strip()
                        if joined_text:
                            res.append(
                                SubtitleSentence(
                                    text=joined_text,
                                    start_time=float(buf[0].start_time),
                                    end_time=float(buf[-1].end_time),
                                    words=list(buf),
                                )
                            )
                        buf = []
                        buf_tokens = 0

                if buf:
                    joined_text = ("" if is_cjk else " ").join(bw.text for bw in buf).strip()
                    if joined_text:
                        res.append(
                            SubtitleSentence(
                                text=joined_text,
                                start_time=float(buf[0].start_time),
                                end_time=float(buf[-1].end_time),
                                words=list(buf),
                            )
                        )

                if len(res) > 1:
                    out.extend(res)
                else:
                    out.append(sub)
                continue

            # ── Nhánh 2: Dự phòng chỉ có văn bản (không có mốc từ)
            # Dùng regex không tách dấu phẩy nằm giữa hai chữ số (như 1,000)
            parts = re.split(r"((?<!\d),(?!\d)|[、;；])", text)
            clauses = []
            i = 0
            while i < len(parts):
                chunk = parts[i]
                delim = parts[i + 1] if i + 1 < len(parts) else ""
                c = chunk + delim
                if c.strip():
                    clauses.append(c.strip())
                i += 2

            res_texts = []
            buf_texts: List[str] = []
            buf_tokens = 0
            for idx, c in enumerate(clauses):
                buf_texts.append(c)
                buf_tokens += count_tokens(c)
                remaining_clauses = clauses[idx + 1:]
                remaining_tokens = sum(count_tokens(rc) for rc in remaining_clauses)
                ends_with_comma = any(c.endswith(d) for d in clause_delims)

                if ends_with_comma and buf_tokens >= min_words and remaining_tokens >= 2:
                    res_texts.append(("" if is_cjk else " ").join(buf_texts).strip())
                    buf_texts = []
                    buf_tokens = 0

            if buf_texts:
                res_texts.append(("" if is_cjk else " ").join(buf_texts).strip())

            if len(res_texts) <= 1:
                out.append(sub)
                continue

            for piece, st, en in _allocate_spans(res_texts, sub.start_time, sub.end_time):
                out.append(SubtitleSentence(text=piece, start_time=st, end_time=en))

        return out

    def unload_model(self) -> None:
        """Giải phóng model aligner (trả VRAM).

        Sau Giai đoạn 3 chỉ còn đường GGUF: model nằm trong TIẾN TRÌNH CON của CrispASR. Gọi
        `free()` để DLL giải phóng context đang cache; muốn trả sạch 100% VRAM thì tắt hẳn worker
        (`CrispASRAlignerClient.get_instance().shutdown()`).
        """
        try:
            from backend.asr.crispasr_aligner import CrispASRAlignerClient

            CrispASRAlignerClient.get_instance().free()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Giải phóng aligner lỗi: {exc}", extra={"module_tag": "ASR"})
        self._is_warmed = False
        logger.info("Đã giải phóng ForcedAligner khỏi VRAM.", extra={"module_tag": "ASR"})
