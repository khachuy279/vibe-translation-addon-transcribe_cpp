"""Bộ mẫu Prompt chuyên dụng cho model Translate-Gemma-4-Sub (17slever17/translate-gemma-4-sub-e4b-GGUF).

Model source: https://huggingface.co/17slever17/translate-gemma-4-sub-e4b-GGUF
Base architecture: Google Gemma 4 (gemma-4-E4B-it fine-tuned).

Đặc điểm kiến trúc & Prompt:
- Được huấn luyện đặc biệt cho dịch phụ đề trực tiếp / video (live subtitle translation).
- Tách biệt rõ ràng giữa SYSTEM MESSAGE và USER MESSAGE.
- Hỗ trợ ngữ cảnh liền kề 2 chiều: [PREVIOUS_SOURCE] và [PREVIOUS_TRANSLATION]
  giúp duy trì kính ngữ, ngôi xưng, ngữ điệu và mạch hội thoại nhất quán.
- Giữ nguyên tiếng lóng (slang), tục ngữ (profanity), khẩu ngữ (spoken tone), cảm xúc và
  câu ngập ngừng / lặp lại — đúng danh sách model card liệt kê.
- Dùng Chat Template chuẩn của Gemma 4 (<|turn>system, <|turn>user, <|turn>model, <turn|>).
- Pipeline B (dịch gộp) dùng CÙNG giao thức JSON khoá số như Index-Translate
  (``{"1": "...", "2": "..."}``) để đi đường một-lần-gọi-LLM, KHÔNG dịch tuần tự dây chuyền.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.translation.prompts import PromptStrategy, resolve_lang_name
from backend.utils.logger import logger

# --------------------------------------------------------------------------- #
# Template System Message & User Message cho Gemma Sub
# --------------------------------------------------------------------------- #

SYSTEM_TEMPLATE_SUBTITLE = (
    "TASK: Translate {source_language} subtitles into {target_language}.\n"
    "RULES: {rules}\n"
    "STYLE: {style}.\n"
    "Translate only CURRENT_SOURCE. PREVIOUS_SOURCE and PREVIOUS_TRANSLATION are context only. "
    "Preserve meaning, tone, slang, profanity, uncertainty, repetitions and incomplete speech. "
    "Return only the final translation without labels or commentary."
)

USER_TEMPLATE_WITH_CONTEXT = (
    "[PREVIOUS_SOURCE]\n"
    "{previous_source}\n\n"
    "[PREVIOUS_TRANSLATION]\n"
    "{previous_translation}\n\n"
    "[CURRENT_SOURCE]\n"
    "{current_source}"
)

USER_TEMPLATE_NO_CONTEXT = (
    "[CURRENT_SOURCE]\n"
    "{current_source}"
)

# System Message cho chế độ dịch GỘP (Pipeline B) — định dạng chuẩn Gemma-Sub kết hợp giao thức JSON:
# - Tuân thủ format huấn luyện của model: TASK: ... RULES: ... STYLE: ...
# - Thêm câu chỉ định CURRENT_SOURCE / PREVIOUS_* tránh model dịch lại ngữ cảnh.
# - Ràng buộc cấu trúc JSON khoá số giống Index-Translate.
SYSTEM_TEMPLATE_BATCH_JSON = (
    "TASK: Translate {source_language} subtitles into {target_language}.\n"
    "RULES: {rules}\n"
    "STYLE: {style}.\n"
    "Translate only CURRENT_SOURCE. PREVIOUS_SOURCE and PREVIOUS_TRANSLATION are context only. "
    "Preserve meaning, tone, slang, profanity, uncertainty, repetitions and incomplete speech. "
    "Return only the final translation without labels or commentary.\n\n"
    "INPUT FORMAT: the user message contains ONE JSON object whose numeric keys map to subtitle segments "
    "(a value may be a nested object mapping a speaker label to the sentence). "
    "Translate only the sentence strings.\n"
    "OUTPUT FORMAT: return ONE JSON object with EXACTLY the same keys, the same nesting and the same order. "
    "Never alter the keys, the structure or the JSON syntax, and never merge, split, drop or renumber segments. "
    "Return the raw JSON object only — no markdown fence, no labels, no commentary."
)

# Template cho dịch văn bản thông thường (General Translation) khi không phải phụ đề
SYSTEM_TEMPLATE_GENERAL = (
    "TASK: Translate {source_language} into {target_language}. "
    "Follow all demonstrations, glossary mappings, partial-translation constraints and formatting instructions in the user prompt. "
    "Preserve meaning, names, numbers, terminology, register and document structure. "
    "Return only the requested final translation without commentary."
)


def _wrap_gemma4_chat(system_content: str, user_content: str) -> str:
    """Bọc System và User prompt vào Chat Template của Gemma 4 (<|turn>)."""
    # logger.info(
    #     f"system\n{system_content.strip()}\nuser\n{user_content.strip()}",
    #     extra={"module_tag": "TRANSLATE"},
    # )
    return (
        f"<|turn>system\n{system_content.strip()}\n<turn|>\n"
        f"<|turn>user\n{user_content.strip()}\n<turn|>\n"
        f"<|turn>model\n"
    )


#: Dấu phân tách giữa các cặp ngữ cảnh. Hai nguồn trong repo dùng HAI kiểu khác nhau:
#: `ContextManager.get_context_str()` nối bằng ``" | "`` (Pipeline A), còn
#: `lookahead_handler` nối bằng ``"\n"`` (Pipeline B). Trước đây chỉ tách ``"|"`` nên chuỗi
#: kiểu xuống dòng bị coi là MỘT cặp ⇒ `[PREVIOUS_TRANSLATION]` chứa lẫn cả câu nguồn kèm
#: mũi tên của cặp thứ hai (lỗi thật gặp trong log 2026-10-10).
_CONTEXT_PAIR_SEPARATOR = re.compile(r"\s*\|\s*|\n+")


def _parse_context_pairs(context_str: str) -> List[Tuple[str, str]]:
    """Tách chuỗi context thành các CẶP ``(nguồn, bản dịch)`` HOÀN CHỈNH.

    CHỈ nhận segment có đủ hai phía. Segment không có ``->`` — ví dụ
    ``lookahead_handler`` đẩy vào ``f"{orig} -> {trans}" if trans else orig`` khi câu trước
    chưa có bản dịch — hoặc có ``->`` mà một phía rỗng, sẽ bị **bỏ qua**: nhận nó sẽ làm
    ``[PREVIOUS_SOURCE]`` và ``[PREVIOUS_TRANSLATION]`` lệch số dòng, tức một dạng prompt
    model chưa từng thấy khi train (model card chỉ cho phép bỏ **cả hai** khối context).
    """
    if not context_str or not context_str.strip():
        return []

    pairs: List[Tuple[str, str]] = []
    dropped = 0
    for seg in _CONTEXT_PAIR_SEPARATOR.split(context_str):
        seg = seg.strip()
        if not seg:
            continue
        src, sep, tgt = seg.partition("->")
        src, tgt = src.strip(), tgt.strip()
        if not sep or not src or not tgt:
            dropped += 1
            continue
        pairs.append((src, tgt))

    if dropped:
        logger.debug(
            f"parse_context_history: bỏ {dropped} segment ngữ cảnh không đủ cặp nguồn->dịch",
            extra={"module_tag": "TRANSLATE"},
        )
    return pairs


def parse_context_history(context_str: str) -> Tuple[str, str]:
    """Trả 2 khối ``previous_source`` / ``previous_translation`` SONG SONG theo dòng.

    Bảo đảm bất biến: hai khối luôn có CÙNG số dòng (hoặc cùng rỗng).
    """
    pairs = _parse_context_pairs(context_str)
    return (
        "\n".join(src for src, _ in pairs),
        "\n".join(tgt for _, tgt in pairs),
    )


class GemmaSubPromptStrategy(PromptStrategy):
    """Chiến lược Prompt cho model 17slever17/translate-gemma-4-sub-e4b-GGUF.

    Tương thích với kiến trúc Prompt của bộ dịch SubWave và fine-tuning Gemma 4.
    """

    #: Model dùng chat template Gemma-4 nhưng ở Pipeline B vẫn gửi payload JSON khoá số
    #: (giống Index-Translate) ⇒ engine đi đường JSON một-lần-gọi, KHÔNG dịch tuần tự.
    supports_json_batch: bool = True

    def __init__(
        self,
        style: str = "neutral",
        default_rules: Optional[str] = None,
    ):
        self.style = style  # friendly / official / neutral
        self.default_rules = (
            default_rules
            or "keep the conversational tone and the speaker's pronouns consistent across segments; if a segment omits the subject, do not invent one"
        )

    #: Ghi chú: KHÔNG có `clean_output` riêng cho Gemma. Việc dọn artifact tag
    #: (`<|turn>`, `<turn|>`, `<think>`, `[CURRENT_SOURCE]`…) do MỘT hàm duy nhất đảm nhiệm:
    #: `backend.translation.engine.clean_translation_tags` — engine gọi nó trên mọi đầu ra
    #: (batch JSON, batch tuần tự, đơn câu). Bản sao trùng lặp trước đây trong file này đã bị
    #: xoá vì không có caller nào (`clean_output` không nằm trong interface `PromptStrategy`).

    def batch_generation_params(self) -> Optional[Dict[str, Any]]:
        """Gemma-Sub dùng ĐÚNG thông số khai trong ``translation_models.yaml``.

        Trả ``None`` ⇒ engine lấy ``temperature/top_p/top_k/repetition_penalty`` của chính
        model này từ catalog (greedy decoding 0.0 / 1.0 / 1 / 1.0 theo khuyến nghị tác giả).
        """
        return None

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = False,
        term: Optional[str] = None,
    ) -> str:
        """Dựng prompt cho dịch từng câu (Pipeline A / Realtime Streaming)."""
        src_name = resolve_lang_name(source_lang)
        tgt_name = resolve_lang_name(target_lang)
        current_text = (text or "").strip()

        # Xây dựng RULES
        rules_list: List[str] = []
        if self.default_rules:
            rules_list.append(self.default_rules)
        if term and str(term).strip():
            rules_list.append(f"fixed terminology rendering: {str(term).strip()}")

        rules_str = "; ".join(rules_list) if rules_list else "none"
        if rules_str != "none" and not rules_str.endswith("."):
            rules_str += "."

        # 1. System Message
        system_msg = SYSTEM_TEMPLATE_SUBTITLE.format(
            source_language=src_name,
            target_language=tgt_name,
            rules=rules_str,
            style=self.style,
        )

        # 2. User Message
        # `parse_context_history` chỉ trả các CẶP hoàn chỉnh ⇒ hai khối luôn cùng số dòng,
        # và rỗng cả hai khi context không có cặp nào dùng được.
        prev_src, prev_tgt = parse_context_history(context) if (use_context and context) else ("", "")

        if prev_src and prev_tgt:
            user_msg = USER_TEMPLATE_WITH_CONTEXT.format(
                previous_source=prev_src,
                previous_translation=prev_tgt,
                current_source=current_text,
            )
        else:
            user_msg = USER_TEMPLATE_NO_CONTEXT.format(
                current_source=current_text,
            )

        return _wrap_gemma4_chat(system_msg, user_msg)

    def build_batch_prompt(
        self,
        sentences: List[str],
        source_lang: str,
        target_lang: str,
        format_type: str = "JSON",
        context: str = "",
        use_context: bool = False,
        terms: str = "",
        speaker_tags: Optional[List[str]] = None,
    ) -> str:
        """Dựng prompt dịch GỘP (Pipeline B) dưới dạng JSON — GIỐNG Index-Translate.

        `format_type` chỉ giữ cho tương thích chữ ký (payload luôn là JSON object).
        """
        src_name = resolve_lang_name(source_lang)
        tgt_name = resolve_lang_name(target_lang)

        rules_list: List[str] = [
            "keep the original segment count and order",
            "translate every segment, never merge, split or drop segments",
            "keep the conversational tone and the speaker's pronouns consistent across segments",
            "if a segment omits the subject, do not invent one",
        ]
        # Gợi ý glossary (tên riêng/thuật ngữ) — mỗi dòng thành một ràng buộc riêng.
        if terms and terms.strip():
            rules_list.extend(line.strip() for line in terms.splitlines() if line.strip())

        rules_str = "; ".join(rules_list)
        if rules_str and not rules_str.endswith("."):
            rules_str += "."

        system_msg = SYSTEM_TEMPLATE_BATCH_JSON.format(
            source_language=src_name,
            target_language=tgt_name,
            rules=rules_str,
            style=self.style,
        )

        # Payload JSON giống Pipeline B của Index-Translate: khoá số, lồng speaker khi có.
        if speaker_tags and len(speaker_tags) == len(sentences):
            data: Dict[str, Any] = {
                str(i + 1): {(speaker_tags[i] or f"speaker {i + 1}").strip(): (s or "").strip()}
                for i, s in enumerate(sentences)
            }
        else:
            data = {str(i + 1): (s or "").strip() for i, s in enumerate(sentences)}
        json_text = json.dumps(data, ensure_ascii=False, indent=2)

        # Cặp ngữ cảnh luôn song song (xem `_parse_context_pairs`).
        prev_src, prev_tgt = parse_context_history(context) if (use_context and context) else ("", "")

        parts: List[str] = []
        if prev_src and prev_tgt:
            parts.append(f"[PREVIOUS_SOURCE]\n{prev_src}")
            parts.append(f"[PREVIOUS_TRANSLATION]\n{prev_tgt}")
        parts.append(f"[CURRENT_SOURCE]\n{json_text}")

        return _wrap_gemma4_chat(system_msg, "\n\n".join(parts))

    def get_stop_tokens(self) -> List[str]:
        """Stop tokens chuẩn cho Gemma 4 và model subtitle."""
        return [
            "<turn|>",
            "<|turn>",
            "<eos>",
            "<end_of_turn>",
            "<|end_of_turn|>",
            "</output>",
            "\n[CURRENT_SOURCE]",
            "\n[PREVIOUS_SOURCE]",
        ]
