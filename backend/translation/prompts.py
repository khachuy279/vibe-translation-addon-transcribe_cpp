"""Bộ mẫu Prompt chuyên biệt cho từng mô hình dịch (Hunyuan-MT2, Xiaomi MiLM, GemmaX2, Index-Translate-2B)."""

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

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = False,
    ) -> str:
        raise NotImplementedError

    def get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|endoftext|>", "</s>", "\n\n"]


class TencentPromptStrategy(PromptStrategy):
    """Prompt chuyên dụng cho Tencent Hunyuan-MT2 (ChatML template)."""

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

        if context and use_context:
            user_content = (
                f"Reference the following translations: {context.strip()}\n\n"
                f"Translate the following text into Vietnamese. Note that you must ONLY output the translated result without any additional explanation:\n{text.strip()}"
            )
        else:
            user_content = f"Translate the following text into Vietnamese. Note that you must ONLY output the translated result without any additional explanation:\n{text.strip()}"

        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    def get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "\n\n"]


class MiLMMPromptStrategy(PromptStrategy):
    """Prompt chuyên dụng cho Xiaomi MiLM-MT."""

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = False,
    ) -> str:
        src = resolve_lang_name(source_lang)
        tgt = resolve_lang_name(target_lang)

        if context and use_context:
            return (
                f"Background: {context.strip()}\n"
                f"Translate this from {src} to {tgt}:\n"
                f"{src}: {text.strip()}\n"
                f"{tgt}:"
            )
        return (
            f"Translate this from {src} to {tgt}:\n"
            f"{src}: {text.strip()}\n"
            f"{tgt}:"
        )

    def get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|endoftext|>", "</s>", "\n", "<end_of_turn>", "<eos>"]


class GemmaXPromptStrategy(PromptStrategy):
    """Prompt chuyên dụng cho GemmaX2."""

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = False,
    ) -> str:
        src = resolve_lang_name(source_lang)
        tgt = resolve_lang_name(target_lang)

        if context and use_context:
            return (
                f"Background: {context.strip()}\n"
                f"Translate this from {src} to {tgt}:\n"
                f"{src}: {text.strip()}\n"
                f"{tgt}:"
            )
        return (
            f"Translate this from {src} to {tgt}:\n"
            f"{src}: {text.strip()}\n"
            f"{tgt}:"
        )

    def get_stop_tokens(self) -> List[str]:
        return ["<end_of_turn>", "<eos>", "<|endoftext|>", "</s>", "<|im_end|>", "\n"]


class IndexTranslatePromptStrategy(PromptStrategy):
    """Prompt chuyên dụng cho IndexTeam/Index-Translate-2B (Qwen3.5-base, ChatML).

    Model dùng Qwen3.5 làm base — có thinking token `<think>`. Để dịch nhanh (no-think mode)
    ta ép output bắt đầu bằng `<think>\n\n</think>\n\n` ngay trong prefix assistant,
    tương đương với việc truyền `enable_thinking=False` trong chat template chính thức.

    Prompt reference: https://huggingface.co/IndexTeam/Index-Translate-2B
    (通用翻译模型 — en/ja/zh互译, Qwen3.5系列)
    """

    # Prefix cố định để tắt thinking mode và bắt đầu output thẳng.
    _NO_THINK_PREFIX = "<think>\n\n</think>\n\n"

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

        # System prompt mô tả nhiệm vụ dịch thuật.
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

    def get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>", "\n\n"]


def get_prompt_strategy(style: Optional[str]) -> PromptStrategy:
    """Lấy Prompt Strategy phù hợp với kiểu mô hình."""
    st = (style or "tencent").lower().strip()
    if st in ("milmmt", "xiaomi"):
        return MiLMMPromptStrategy()
    elif st in ("gemmax", "gemmax2", "gemma"):
        return GemmaXPromptStrategy()
    elif st in ("index", "index-translate", "index_translate"):
        return IndexTranslatePromptStrategy()
    return TencentPromptStrategy()
