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
    kana = hangul = han = cyrillic = 0
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
            kana += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            hangul += 1
        elif 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF:
            han += 1
        elif 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
    if kana:
        return "Japanese"
    if hangul:
        return "Korean"
    if han:
        return "Chinese"
    if cyrillic:
        return "Russian"
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


#: Ngôn ngữ KHÔNG có dấu cách giữa từ ⇒ mỗi ký tự là một token, và KHÔNG được nối bằng `" "`.
_CJK_LANGUAGES = ("japanese", "chinese", "korean", "cantonese")


def _is_cjk_language(language: str) -> bool:
    """True nếu ngôn ngữ không dùng dấu cách phân tách từ (Nhật/Trung/Hàn/Quảng)."""
    return str(language or "").strip().lower() in _CJK_LANGUAGES


def _count_language_tokens(text: str, is_cjk: bool) -> int:
    """Đếm token của một mẩu văn bản: CJK ⇒ đếm ký tự; Latinh ⇒ đếm từ.

    Dùng chung cho các bộ tách câu để mọi lớp ngắt câu có cùng định nghĩa "token".
    """
    clean = re.sub(r"[,、;；\.?!。？！\s]", "", text or "")
    if not is_cjk and " " in (text or ""):
        return len(text.split())
    return len(re.findall(r"[a-zA-Z0-9]+|[^\s\W\d_]", clean))


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


def _has_internal_punctuation(text: str) -> bool:
    """Kiểm tra câu có chứa dấu ngắt câu / dấu câu ở giữa hay không.

    Bỏ qua:
      - Dấu mở ngoặc/nháy ở đầu câu (`「『（([{〈《"'“‘`).
      - Dấu kết câu và ngoặc/nháy đóng ở đuôi câu (`。！？!?…；;.,:：!?-—」』）)]}〉》"'”’\n `).
      - Dấu chấm/phẩy/hai chấm giữa các chữ số (ví dụ: `3.9`, `1,000`, `10:30`).
      - Dấu chấm đi liền sau từ viết tắt (ví dụ: `Dr.`, `Mr.`).
    """
    if not text:
        return False
    _closing_punct = "。！？!?…；;.,:：!?-—」』）)]}〉》\"'”’\n "
    _opening_punct = "「『（([{〈《\"'“‘\n "
    inner = text.strip().rstrip(_closing_punct).lstrip(_opening_punct).strip()
    if not inner:
        return False
    pattern = r"[、，;；:：—–。！？!?…\n]|(?<!\d),(?!\d)|(?<!\d)\.(?!\d)|(?<!\d):(?!\d)"
    for match in re.finditer(pattern, inner):
        if match.group() == ".":
            prefix = inner[: match.start() + 1]
            if _ends_with_abbreviation(prefix):
                continue
    return False


def _is_sentence_end_word(words: List[AlignedWord], i: int, is_cjk: bool) -> bool:
    """Xác định token thứ i trong words có phải là điểm KẾT THÚC CÂU hay không.

    Quy tắc:
      1. Bỏ qua các ký tự đóng nháy/ngoặc ở đuôi: "well." hoặc "said," hay '」'
      2. Ký tự kết thúc câu: 。！？!?…
      3. Nếu là dấu `.`:
         - Bỏ qua số thập phân (ví dụ 3.14).
         - Bỏ qua từ viết tắt theo `_ends_with_abbreviation` (Dr., Mr., MIT...).
         - Với Latinh: nếu từ kế tiếp (i+1) bắt đầu bằng chữ thường (và khe âm học < 0.4s) -> không kết câu.
    """
    if i < 0 or i >= len(words):
        return False
    w_text = (words[i].text or "").strip()
    if not w_text:
        return False
    core = w_text.rstrip(_SENTENCE_CLOSERS).strip()
    if not core:
        return False
    last_ch = core[-1]

    # CJK kết câu bằng 。 hoặc ！ hoặc ？ hoặc …
    if last_ch in {"。", "！", "？", "…", "!", "?"}:
        return True

    if last_ch == ".":
        # 1. Số thập phân
        if len(core) >= 2 and core[-2].isdigit():
            if i + 1 < len(words) and (words[i + 1].text or "").strip()[:1].isdigit():
                return False
        elif core == ".":
            if (
                i > 0
                and (words[i - 1].text or "").strip()[-1:].isdigit()
                and i + 1 < len(words)
                and (words[i + 1].text or "").strip()[:1].isdigit()
            ):
                return False

        # 2. Từ viết tắt: Dr., Mr., vs., etc.
        if i + 1 < len(words):
            if _ends_with_abbreviation(core):
                return False

        # 3. Với Latinh: nếu từ kế tiếp bắt đầu bằng chữ thường và không có khoảng lặng lớn
        if not is_cjk and i + 1 < len(words):
            next_t = (words[i + 1].text or "").strip().lstrip("\"'“‘([{<")
            if next_t and next_t[0].islower():
                gap = words[i + 1].start_time - words[i].end_time
                if gap < 0.4:
                    return False

        return True

    return False


def _is_clause_end_word(w_text: str) -> bool:
    """True nếu từ kết thúc bằng dấu ngắt vế câu (,;、；—)."""
    core = (w_text or "").strip().rstrip(_SENTENCE_CLOSERS).strip()
    if not core:
        return False
    ch = core[-1]
    if ch in {",", "、", ";", "；", "—"}:
        # Bỏ qua nếu là số, ví dụ 1,000
        if ch == "," and len(core) >= 2 and core[-2].isdigit():
            return False
        return True
    return False


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
    def segment_aligned_words(
        aligned_items: List[AlignedWord],
        language: str = "English",
        silence_gap_sec: float = 0.4,
        comma_min_words: int = 4,
        comma_min_chars: int = 8,
        min_words: int = 3,
        min_chars: int = 6,
        **kwargs: Any,  # Bỏ qua mọi giới hạn cũ: max_words, max_chars, max_duration_sec, etc.
    ) -> List[SubtitleSentence]:
        """Đại tu ngắt câu SEG_BATCH (2026-10-13): Chỉ áp dụng đúng 2 trường hợp ngắt câu.

        Xóa bỏ toàn bộ các giới hạn khác (không giới hạn 10 từ, không giới hạn giây, không chẻ vụn):
          - Trường hợp 1: Có dấu kết thúc câu (.?!。！？…).
            1. Tách theo dấu kết thúc câu.
            2. Với mỗi candidate, nếu có dấu phẩy/chấm phẩy (,;、；) và phía trước >= comma_min_words
               (hoặc comma_min_chars) và phía sau >= 2 từ (hoặc 4 ký tự) -> tách tiếp theo dấu phẩy.
            3. Nếu không có dấu phẩy: giữ nguyên trọn vẹn cả câu (dù bao nhiêu từ).
          - Trường hợp 2: Không có dấu kết thúc câu (hoặc đoạn lửng).
            1. Dùng word timestamps để tìm khoảng lặng / gap lớn giữa các từ (gap >= silence_gap_sec).
            2. Tách tại các gap đó (mỗi đoạn >= min_units).
            3. Với mỗi đoạn sau tách khoảng lặng, nếu có dấu phẩy thỏa mãn -> tách tiếp theo dấu phẩy.
            4. Tuyệt đối không cắt vụn thêm bằng trần từ hay trần giây.
        """
        if not aligned_items:
            return []

        n = len(aligned_items)
        is_cjk = _is_cjk_language(language)

        def count_units(lo_idx: int, hi_idx: int) -> int:
            if is_cjk:
                return sum(len(aligned_items[k].text or "") for k in range(lo_idx, hi_idx + 1))
            return hi_idx - lo_idx + 1

        min_units = min_chars if is_cjk else min_words
        comma_thresh = comma_min_chars if is_cjk else comma_min_words
        comma_min_remain = 4 if is_cjk else 2

        def split_span_by_commas(lo: int, hi: int) -> List[Tuple[int, int]]:
            """Tách đoạn [lo, hi] tại dấu phẩy/chấm phẩy (,;、；)."""
            if lo > hi:
                return []
            if comma_thresh <= 0:
                return [(lo, hi)]
            spans: List[Tuple[int, int]] = []
            seg_start = lo
            for k in range(lo, hi):
                w_txt = aligned_items[k].text
                if _is_clause_end_word(w_txt):
                    left_u = count_units(seg_start, k)
                    right_u = count_units(k + 1, hi)
                    if left_u >= comma_thresh and right_u >= comma_min_remain:
                        spans.append((seg_start, k))
                        seg_start = k + 1
            spans.append((seg_start, hi))
            return spans

        # ── BƯỚC 1: Quét Candidate blocks theo dấu kết câu (Case 1)
        candidates: List[Tuple[int, int, bool]] = []
        cur_lo = 0
        for i in range(n):
            if _is_sentence_end_word(aligned_items, i, is_cjk):
                candidates.append((cur_lo, i, True))
                cur_lo = i + 1
        if cur_lo < n:
            candidates.append((cur_lo, n - 1, False))

        # ── BƯỚC 2: Xử lý đúng 2 trường hợp
        final_spans: List[Tuple[int, int]] = []
        for c_lo, c_hi, has_sent_end in candidates:
            if has_sent_end:
                # Trường hợp 1: Có dấu kết câu -> tách dấu phẩy nếu có, giữ trọn vẹn câu
                final_spans.extend(split_span_by_commas(c_lo, c_hi))
            else:
                # Trường hợp 2: Không có dấu kết câu -> tách bằng khoảng lặng giữa các từ
                pause_spans: List[Tuple[int, int]] = []
                p_lo = c_lo
                if silence_gap_sec > 0:
                    for k in range(c_lo, c_hi):
                        gap = aligned_items[k + 1].start_time - aligned_items[k].end_time
                        if gap >= silence_gap_sec:
                            left_u = count_units(p_lo, k)
                            right_u = count_units(k + 1, c_hi)
                            if left_u >= min_units and right_u >= min_units:
                                pause_spans.append((p_lo, k))
                                p_lo = k + 1
                pause_spans.append((p_lo, c_hi))

                # Tách tiếp theo dấu phẩy trên từng đoạn sau khoảng lặng
                for p_start, p_end in pause_spans:
                    final_spans.extend(split_span_by_commas(p_start, p_end))

        # ── BƯỚC 3: Dựng SubtitleSentence
        subtitles: List[SubtitleSentence] = []
        for s_lo, s_hi in final_spans:
            sub_words = list(aligned_items[s_lo : s_hi + 1])
            if not sub_words:
                continue
            if is_cjk:
                text = "".join(w.text for w in sub_words).strip()
            else:
                text = " ".join(w.text for w in sub_words).strip()
            if not text:
                continue
            valid_starts = [float(w.start_time) for w in sub_words if float(w.start_time) > 0.001]
            valid_ends = [float(w.end_time) for w in sub_words if float(w.end_time) > 0.001]
            st = valid_starts[0] if valid_starts else float(sub_words[0].start_time)
            en = valid_ends[-1] if valid_ends else float(sub_words[-1].end_time)
            if en <= st:
                en = st + 0.5

            subtitles.append(
                SubtitleSentence(
                    text=text,
                    start_time=st,
                    end_time=en,
                    words=sub_words,
                )
            )

        chunk_dur = float(kwargs.get("chunk_duration_sec", 0.0) or kwargs.get("chunk_duration", 0.0) or 0.0)
        return ForcedAlignerService.sanitize_subtitle_timestamps(subtitles, chunk_duration_sec=chunk_dur)

    @staticmethod
    def sanitize_subtitle_timestamps(
        subtitles: List[SubtitleSentence],
        chunk_duration_sec: float = 0.0,
    ) -> List[SubtitleSentence]:
        """Căn chỉnh thời gian hiển thị phụ đề = đúng với thời gian của đoạn audio đó.

        Quy tắc:
          1. start_time: đúng lúc câu bắt đầu nói trong audio (từ đầu tiên có mốc hợp lệ).
          2. end_time: bao trọn toàn bộ thời gian nói của câu trong audio:
             - Nếu câu có các từ sau chưa được align (mốc 0.0), tự bù thời lượng ước lượng của các từ đó
               hoặc kéo dài đến sát lúc câu tiếp theo bắt đầu (next_start - 0.05s).
             - Giữ phụ đề hiển thị xuyên suốt đoạn audio của câu + đệm nghe/đọc tự nhiên (~0.4s - 0.6s)
               nếu có khoảng lặng trước câu kế tiếp.
             - Tuyệt đối không cắt cụt câu khi người nói chưa dứt lời (không dùng công thức trần cứng).
             - Đảm bảo tính đơn điệu (monotonic): câu sau không bao giờ bắt đầu trước câu trước.
        """
        if not subtitles:
            return subtitles

        n = len(subtitles)
        chunk_dur = float(chunk_duration_sec) if chunk_duration_sec > 0 else 0.0
        if chunk_dur <= 0.0:
            for s in subtitles:
                for w in (s.words or []):
                    chunk_dur = max(chunk_dur, float(w.end_time or 0.0))
            if chunk_dur <= 0.0:
                chunk_dur = max(5.0, n * 3.5)

        # Bước 1: Trích xuất mốc thực tế của từng câu từ các từ đã căn chỉnh
        extracted: List[Dict[str, Any]] = []
        for s in subtitles:
            words = getattr(s, "words", None) or []
            valid_words = [
                w for w in words
                if float(w.end_time or 0.0) > float(w.start_time or 0.0) and float(w.end_time or 0.0) > 0.001
            ]

            if valid_words:
                first_w = valid_words[0]
                last_w = valid_words[-1]
                st = float(first_w.start_time)
                en = float(last_w.end_time)
                trailing_unaligned = len(words) - 1 - words.index(last_w)
            else:
                st = float(s.start_time or 0.0)
                en = float(s.end_time or 0.0)
                trailing_unaligned = 0

            extracted.append({
                "sub": s,
                "st": st,
                "en": en,
                "has_valid": len(valid_words) > 0,
                "trailing_words": trailing_unaligned,
                "word_count": len(words) if words else len(s.text.split()),
            })

        # Bước 2: Tìm các mốc neo (anchor) chắc chắn từ âm học thực tế
        anchors: List[int] = []
        last_t = 0.0
        for i, item in enumerate(extracted):
            if item["has_valid"] and item["st"] >= last_t and item["en"] > item["st"]:
                anchors.append(i)
                last_t = item["en"]

        # Bước 3: Tính toán start_time và end_time cho từng câu bám sát audio
        final_subs: List[SubtitleSentence] = []
        prev_end = 0.0

        for i, item in enumerate(extracted):
            sub = item["sub"]
            st = item["st"]
            en = item["en"]
            has_valid = item["has_valid"]

            # Nếu mốc bị nhảy lùi về quá khứ hoặc = 0
            if not has_valid or st < prev_end or en <= st:
                next_anchor_idx = next((k for k in anchors if k > i), None)
                if next_anchor_idx is not None:
                    next_anchor_st = extracted[next_anchor_idx]["st"]
                    span_available = max(0.6, next_anchor_st - prev_end)
                    total_words = sum(extracted[k]["word_count"] for k in range(i, next_anchor_idx))
                    my_ratio = item["word_count"] / max(1, total_words)
                    alloc = span_available * my_ratio
                    st = prev_end + 0.02
                    en = min(next_anchor_st - 0.02, st + alloc)
                else:
                    span_available = max(1.0, chunk_dur - prev_end)
                    total_words = sum(extracted[k]["word_count"] for k in range(i, n))
                    my_ratio = item["word_count"] / max(1, total_words)
                    alloc = span_available * my_ratio
                    st = prev_end + 0.02
                    en = min(chunk_dur, st + alloc)
            else:
                # Câu có mốc hợp lệ: đảm bảo st không lùi về trước câu trước
                if st < prev_end:
                    st = prev_end + 0.02

                # Nếu có từ chưa align ở đuôi: bù thời gian cho các từ chưa align
                if item["trailing_words"] > 0:
                    en = en + item["trailing_words"] * 0.35

            # Xem xét mốc bắt đầu của câu kế tiếp để phụ đề KHÔNG bị tắt sớm giữa chừng
            if i < n - 1:
                next_item = extracted[i + 1]
                next_st_raw = next_item["st"] if next_item["has_valid"] else None
                if next_st_raw is not None and next_st_raw > en:
                    gap = next_st_raw - en
                    if gap < 0.8:
                        # Khoảng nghỉ ngắn giữa 2 câu: kéo phụ đề hiện tại đến sát câu sau (-0.05s)
                        # để người xem nghe trọn vẹn câu mà không bị chớp tắt phụ đề!
                        en = next_st_raw - 0.05
                    else:
                        # Khoảng lặng lớn hơn: giữ đệm nghe tự nhiên 0.4s sau khi dứt từ cuối
                        en = min(next_st_raw - 0.05, en + 0.4)
                elif next_st_raw is not None and next_st_raw <= en:
                    en = max(st + 0.3, next_st_raw - 0.02)
            else:
                # Câu cuối cùng của khối: cho phép đệm nghe tự nhiên 0.5s đến hết khối
                en = min(chunk_dur, en + 0.5)

            # Đảm bảo thời lượng tối thiểu 0.5s
            if en <= st:
                en = st + 0.5

            final_subs.append(
                SubtitleSentence(
                    text=sub.text,
                    start_time=round(st, 3),
                    end_time=round(en, 3),
                    words=sub.words,
                )
            )
            prev_end = en

        return final_subs

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
        is_cjk = _is_cjk_language(language)
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

        is_cjk = _is_cjk_language(language)
        clause_delims = {",", "\u3001", ";", "\uff1b"}

        def count_tokens(s: str) -> int:
            return _count_language_tokens(s, is_cjk)

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

    @staticmethod
    def split_subtitles_by_pause(
        subtitles: List[SubtitleSentence],
        silences: Optional[List[Tuple[float, float]]] = None,
        *,
        min_pause_ms: float = 350.0,
        min_tokens: int = 3,
        language: str = "English",
        tolerance_sec: float = 0.12,
        min_chars: int = 20,
        min_words: int = 6,
        stats: Optional[Dict[str, int]] = None,
        dry_run: bool = False,
    ) -> List[SubtitleSentence]:
        """Lớp ngắt câu bằng KHOẢNG LẶNG: tách câu dài KHÔNG có dấu câu tại chỗ người nói ngắt hơi.

        VÌ SAO CẦN (log thật 2026-10-11 14:51, trang tiếng Nhật): ASR trả về một câu dài **không
        có dấu câu** gộp lời của 2–3 người::

            "じゃステージランドよよし次こそ勝つぞあらあら楽しそうねこれいただき物だけどよかったらどうぞうまそう。"

        RÀNG BUỘC KÍCH HOẠT (chống chẻ vụn câu, cập nhật 2026-10-12):
        Lớp này CHỈ kích hoạt cho câu khi thỏa mãn đồng thời:
          1. ASR trả về CÂU RẤT DÀI: CJK >= min_chars (mặc định 20 ký tự), Latinh >= min_words (mặc định 6 từ).
             Các câu ngắn/vừa phải được giữ nguyên vẹn.
          2. KHÔNG CÓ DẤU CÂU Ở GIỮA (`not _has_internal_punctuation`): nếu câu đã có dấu ngắt câu/vế
             ở giữa (như `、`, `,`), hoặc đã kết câu ở cuối, thì giữ nguyên câu cho các lớp dấu câu
             xử lý, không chẻ vụn tại các khe ngập ngừng nói.

        CÁCH LÀM: dựa trên mốc từ của Forced Aligner, tách tại KHE giữa hai từ liên tiếp
        (`gap = words[i].start_time - words[i-1].end_time`). Khi có `silences` (danh sách khoảng
        lặng VAD xác nhận, cùng hệ thời gian với `words`) thì khe còn phải NẰM TRONG một khoảng
        lặng — VAD là bên xác nhận, Aligner chỉ đề xuất. Không có `silences` (VAD tắt/lỗi) thì dùng
        khe từ đơn thuần; hành vi vẫn tốt hơn cắt cứng theo ký tự.

        Ràng buộc chất lượng: mỗi mảnh — kể cả mảnh ĐẦU và mảnh CUỐI — phải đạt `min_tokens`, nếu
        không thì BỎ QUA điểm tách đó (thà giữ một câu hơi dài còn hơn tạo mảnh cụt 1–2 ký tự).

        Chỉ áp dụng cho câu CÒN `words` (chưa qua `split_oversized_sentences` — hàm đó bỏ mốc từ).
        """
        if stats is not None:
            stats.update(
                sentences_in=len(subtitles or []),
                sentences_out=len(subtitles or []),
                candidates=0,
                vad_confirmed=0,
                accepted=0,
            )
        if not subtitles or min_pause_ms <= 0:
            return subtitles

        is_cjk = _is_cjk_language(language)

        def _is_eligible(s: SubtitleSentence) -> bool:
            """Chỉ xét tách câu khi câu RẤT DÀI và KHÔNG CÓ DẤU CÂU Ở GIỮA."""
            t = (s.text or "").strip()
            if not t:
                return False
            if _has_internal_punctuation(t):
                return False
            tokens = _count_language_tokens(t, is_cjk)
            threshold = min_chars if is_cjk else min_words
            if threshold > 0 and tokens < threshold:
                return False
            return True

        if dry_run:
            # Chỉ đếm khe đủ dài của các câu đủ điều kiện (không xét VAD, không tách).
            pause_sec = float(min_pause_ms) / 1000.0
            for sub in subtitles:
                if not _is_eligible(sub):
                    continue
                ws = list(sub.words or [])
                for i in range(1, len(ws)):
                    if float(ws[i].start_time) - float(ws[i - 1].end_time) >= pause_sec:
                        if stats is not None:
                            stats["candidates"] += 1
            return subtitles

        is_cjk = _is_cjk_language(language)
        pause_sec = float(min_pause_ms) / 1000.0
        min_tokens = max(1, int(min_tokens))

        def tokens_of(ws: List[AlignedWord]) -> int:
            # Đếm TỪNG từ rồi cộng — KHÔNG nối chuỗi rồi mới đếm: với CJK, nối không dấu cách sẽ
            # gộp các từ Latinh cạnh nhau thành một run `[a-zA-Z0-9]+` (ví dụ "w1 w2 w3" → 1 token),
            # làm bộ đếm token hụt và bỏ qua oan mọi điểm tách. Cách đếm này khớp với
            # `split_subtitles_by_clause_comma`.
            return sum(_count_language_tokens(w.text, is_cjk) for w in ws)

        def in_silence(t_mid: float) -> bool:
            if silences is None:
                return True  # không có VAD ⇒ tin khe từ của aligner
            return any(
                (float(s0) - tolerance_sec) <= t_mid <= (float(s1) + tolerance_sec)
                for s0, s1 in silences
            )

        out: List[SubtitleSentence] = []
        for sub in subtitles:
            words = list(sub.words or [])
            if len(words) < 2 or not _is_eligible(sub):
                out.append(sub)
                continue

            pieces: List[List[AlignedWord]] = []
            buf: List[AlignedWord] = [words[0]]
            for i in range(1, len(words)):
                prev_w, cur_w = words[i - 1], words[i]
                gap = float(cur_w.start_time) - float(prev_w.end_time)
                if gap >= pause_sec:
                    if stats is not None:
                        stats["candidates"] += 1
                    t_mid = (float(prev_w.end_time) + float(cur_w.start_time)) / 2.0
                    rest = words[i:]
                    if not in_silence(t_mid):
                        continue
                    if stats is not None:
                        stats["vad_confirmed"] += 1
                    if tokens_of(buf) >= min_tokens and tokens_of(rest) >= min_tokens:
                        if stats is not None:
                            stats["accepted"] += 1
                        pieces.append(buf)
                        buf = []
                buf.append(cur_w)
            pieces.append(buf)

            if len(pieces) <= 1:
                out.append(sub)
                continue

            joiner = "" if is_cjk else " "
            for piece in pieces:
                text = joiner.join(w.text for w in piece).strip()
                if not text:
                    continue
                out.append(
                    SubtitleSentence(
                        text=text,
                        start_time=float(piece[0].start_time),
                        end_time=float(piece[-1].end_time),
                        words=list(piece),
                    )
                )
        if stats is not None:
            stats["sentences_out"] = len(out)
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
