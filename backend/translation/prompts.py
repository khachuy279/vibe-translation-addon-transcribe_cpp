"""Bộ mẫu Prompt chuyên biệt cho Index-Translate (Pipeline A & Pipeline B).

Chỉ giữ lại:
1. Pipeline A: Dịch từng câu realtime (kèm ngữ cảnh lịch sử nếu bật use_context).
2. Pipeline B: Dịch gộp toàn khối qua định dạng JSON (Bilibili instTrans).
"""

import json
from typing import Dict, List, Optional


_LANG_NAME_MAP = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "th": "Thai",
    "id": "Indonesian",
}


def resolve_lang_name(code: str) -> str:
    """Chuyển mã ngôn ngữ 2 ký tự sang tên tiếng Anh."""
    clean = (code or "auto").strip().lower()
    return _LANG_NAME_MAP.get(clean, "Vietnamese" if clean == "vi" else "English")


class PromptStrategy:
    """Base Strategy cho việc tạo Prompt và Stop Tokens."""

    _NO_THINK_PREFIX = "<think>\n\n</think>\n\n"

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = False,
    ) -> str:
        raise NotImplementedError

    def build_batch_prompt(
        self,
        sentences: List[str],
        source_lang: str,
        target_lang: str,
    ) -> str:
        raise NotImplementedError

    def get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"]


class PipelineAPromptStrategy(PromptStrategy):
    """Prompt chuyên dụng cho Pipeline A (Realtime Single Sentence Streaming).

    Model dùng Qwen3.5 làm base — có thinking token `<think>`. Để dịch nhanh (no-think mode)
    ta ép output bắt đầu bằng `<think>\\n\\n</think>\\n\\n` ngay trong prefix assistant.
    Hỗ trợ bổ sung câu ngữ cảnh từ ContextManager (khi use_context: True).
    """

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = False,
    ) -> str:
        src = resolve_lang_name(source_lang) if source_lang != "auto" else "auto"
        tgt = resolve_lang_name(target_lang)

        if src == "auto":
            system = (
                f"You are a professional translator. "
                f"Translate the user's text into {tgt}. "
                f"Output ONLY the translation, no explanations."
            )
        else:
            system = (
                f"You are a professional translator. "
                f"Translate the user's text from {src} into {tgt}. "
                f"Output ONLY the translation, no explanations."
            )

        if context and use_context:
            user_content = (
                f"Reference translations for context: {context.strip()}\n\n"
                f"{text.strip()}"
            )
        else:
            user_content = text.strip()

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{self._NO_THINK_PREFIX}"
        )

    def build_batch_prompt(
        self,
        sentences: List[str],
        source_lang: str,
        target_lang: str,
    ) -> str:
        return PipelineBPromptStrategy().build_batch_prompt(sentences, source_lang, target_lang)


class PipelineBPromptStrategy(PromptStrategy):
    """Prompt chuyên dụng cho Pipeline B (Lookahead Batch Context Translation qua JSON).

    Tuân thủ chuẩn Bilibili instTrans (Hard Constraint: JSON format & key preservation).
    Dịch gộp toàn bộ các câu trong một khối ASR thành 1 lần gọi LLM duy nhất,
    tối đa hóa tính liền mạch ngữ cảnh hội thoại và tiết kiệm 60-70% thời gian GPU.
    """

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = False,
    ) -> str:
        return PipelineAPromptStrategy().build_prompt(text, source_lang, target_lang, context, use_context)

    def build_batch_prompt(
        self,
        sentences: List[str],
        source_lang: str,
        target_lang: str,
    ) -> str:
        tgt = resolve_lang_name(target_lang)
        data = {str(i + 1): s.strip() for i, s in enumerate(sentences)}
        json_text = json.dumps(data, ensure_ascii=False, indent=2)

        system = (
            f"You are a professional translator. "
            f"Translate the user's text into {tgt}. "
            f"Output ONLY valid JSON, no explanations, no markdown formatting."
        )
        user_content = (
            f"Translate the following subtitle JSON data into {tgt}: "
            f"translate only user-facing text values; strictly preserve the exact numeric keys, structure, and JSON format:\n"
            f"{json_text}"
        )

        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{self._NO_THINK_PREFIX}"
        )


# Unified Strategy cho Index-Translate hỗ trợ cả Pipeline A (đơn câu) và Pipeline B (batch JSON)
IndexTranslatePromptStrategy = PipelineBPromptStrategy


def get_prompt_strategy(style: Optional[str] = None) -> PromptStrategy:
    """Lấy Prompt Strategy phù hợp (Index-Translate cho Pipeline A & Pipeline B)."""
    st = (style or "index").lower().strip()
    if st in ("pipeline_a", "single"):
        return PipelineAPromptStrategy()
    if st in ("pipeline_b", "batch"):
        return PipelineBPromptStrategy()
    return IndexTranslatePromptStrategy()
