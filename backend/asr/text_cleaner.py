"""Bộ công cụ làm sạch văn bản và loại bỏ special tokens từ Audio-LLMs."""

import re

from backend.utils.text_repetition import collapse_repetitions

# Biểu thức chính quy loại bỏ token đặc biệt dạng <|tag|> hoặc <tag>
_RE_SPECIAL_TAGS = re.compile(r"<\|.*?\|>|<[^>]+>")
_RE_SYSTEM_PREFIX = re.compile(r"(?m)^(?:system|user|assistant|language\s+\w+)\s*[:]?\s*", flags=re.IGNORECASE)


def clean_transcript_text(raw_text: str) -> str:
    """Làm sạch văn bản đầu ra từ ASR model.

    - Loại bỏ các special tokens như <|transcribe|>, <|en|>, <|notimestamps|>.
    - Loại bỏ tiền tố role chat system/user/assistant.
    - **Gộp cụm lặp do decoder kẹt vòng** (xem `backend/utils/text_repetition.py`): ca thật
      đã gặp là 256 token `ha ha ha ...` từ tiếng cười nền — tốn 1,5 s GPU ở ASR rồi 2,4 s
      GPU ở dịch và tạo một dòng phụ đề khổng lồ. Binding transcribe.cpp không expose
      `max_tokens`/repetition-penalty cho family `qwen3_asr` nên phải chặn ở đây.
    - Cắt bỏ khoảng trắng thừa hai đầu.
    """
    if not raw_text:
        return ""

    text = _RE_SPECIAL_TAGS.sub("", raw_text)
    text = _RE_SYSTEM_PREFIX.sub("", text)
    text = collapse_repetitions(text)
    return text.strip()
