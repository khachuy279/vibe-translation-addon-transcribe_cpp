"""Bộ mẫu Prompt CHUẨN Index-Translate (Pipeline A & Pipeline B).

Nguồn chuẩn (nguyên văn): https://github.com/bilibili/Index-Translate/blob/main/docs/prompts.md

Index-Translate định nghĩa ĐÚNG 3 loại prompt chính thức:

1. ``default``      — Dịch mặc định (dùng cho Pipeline A: dịch từng câu realtime).
2. ``terminology``  — Dịch có ràng buộc thuật ngữ (Pipeline A khi có glossary bắt buộc).
3. ``structured``   — Dịch dữ liệu có cấu trúc (Pipeline B: batch JSON / Bilibili instTrans).

.. warning::

   Model Index-Translate được **fine-tune trên đúng câu chữ** của 3 template này.
   Mọi sai lệch so với upstream đều làm giảm chất lượng dịch. Vì vậy module này
   giữ **NGUYÊN VĂN** câu chữ của ``docs/prompts.md`` và tuân thủ 2 quy tắc:

   * Template nằm **TRỌN VẸN trong MỘT lượt ``user``** — KHÔNG bọc thêm system role
     kiểu ``"You are a professional translator..."`` (upstream không có system role).
   * ``{source_text}`` đứng **NGAY SAU** dòng hướng dẫn, cách nhau đúng một dòng trống.

Các lỗi đã sửa so với bản cũ:

* Bỏ system role tự chế ⇒ template khớp 100% với lúc fine-tune.
* ``"Translate the user's text into {tgt}. Output ONLY the translation, no explanations."``
  ⇒ ``"Translate the following text into {tgt}. Output the translation directly, without any explanation:"``
  (đúng câu chữ upstream, và ``{source_text}`` nằm chung lượt ``user``).
* ``"Translate the following subtitle JSON data ... preserve the exact numeric keys, structure, and JSON format"``
  ⇒ template ``structured`` chuẩn: ``{format_type}`` + ``"never alter the structure, keys, or placeholders"``.
* Bổ sung loại ``terminology`` (ràng buộc thuật ngữ) trước đây **hoàn toàn thiếu**.

Phần mở rộng của repo này (KHÔNG thuộc upstream) — bổ sung P4.1/P4.2 ngày 2026-10-07:

* ``_PRONOUN_GUIDANCE_VI`` được siết: nêu đích danh ``chúng mình``/``bọn mình``/``tụi mình``
  và ánh xạ đại từ nguồn tiếng Nhật, vì bản cũ bị Index-Translate-9B bỏ qua trong log thật.
* Pipeline B có thêm khối **ràng buộc cứng** đặt SAU payload JSON (``_build_batch_constraints``)
  kèm gợi ý glossary tên riêng/thuật ngữ từ ``backend/translation/glossary.py``.
  Đây là sai lệch CHỦ Ý so với upstream, đánh đổi lấy việc tuân thủ ràng buộc; đã đo lại
  bằng A/B trước khi bật (xem ``report/13_pronoun_and_glossary``).

Ghi chú thiết kế: ``source_lang`` vẫn được nhận trong signature để giữ tương thích
ngược cho ``engine.py``, nhưng **không** được nhúng vào prompt — cả 3 template chính
thức đều không có placeholder ngôn ngữ nguồn (model tự nhận diện), thêm vào sẽ lệch spec.
"""

import json
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Template CHÍNH THỨC (bản EN) — NGUYÊN VĂN từ docs/prompts.md
# --------------------------------------------------------------------------- #

# Hướng dẫn đại từ nhân xưng khi dịch sang tiếng Việt (khi không rõ giới tính/tuổi tác)
#
# P4.1 (2026-10-07): siết ràng buộc. Bản cũ chỉ nói "never use the word mình anywhere" và
# Index-Translate-9B **vẫn vi phạm** (`"tôi còn nói là mình chắc chắn…"` — log thật). Bản mới
# (a) gọi tên đích danh các dạng ghép ("chúng mình", "bọn mình", "tụi mình") vì model rất hay
# coi đó là ngoại lệ, (b) nêu ví dụ đại từ NGUỒN tiếng Nhật để model ánh xạ đúng, thay vì chỉ
# mô tả trừu tượng bằng tiếng Anh.
#
# Ràng buộc ở prompt vẫn chỉ là HY VỌNG — bảo đảm nằm ở `pronoun_guard.enforce_no_minh()`,
# được engine gọi trên mọi đầu ra.
_PRONOUN_GUIDANCE_VI = (
    "Requirements: NEVER output the Vietnamese word \"mình\" in any form — not as a standalone "
    "pronoun, and not inside \"chúng mình\", \"bọn mình\" or \"tụi mình\"; "
    "translate first-person pronouns (私, 僕, 俺, 自分) strictly as \"tôi\", "
    "second-person pronouns (あなた, 君, お前) strictly as \"bạn\", and \"we\" as \"chúng ta\". "
    "If the source sentence lacks an explicit subject, do NOT invent or assume one (keep it impersonal without arbitrarily adding pronouns). "
)

# Loại 1: Dịch mặc định (Index-Translate)
TEMPLATE_DEFAULT = (
    "Translate the following text into {target_lang}. "
    "{pronoun_guidance}"
    "Output the translation directly, without any explanation:\n\n"
    "{source_text}"
)

# Loại 2: Dịch có ràng buộc thuật ngữ (Index-Translate)
TEMPLATE_TERMINOLOGY = (
    "Translate the following subtitles into {target_lang}. "
    "Requirements: keep terminology consistent across sentences "
    "(fixed rendering for {term}), {pronoun_guidance}preserve structure and placeholders:\n\n"
    "{source_text}"
)

# Loại 3: Dữ liệu có cấu trúc (Index-Translate - tối ưu phụ đề hội thoại video)
TEMPLATE_STRUCTURED = (
    "Translate the following {format_type} data into {target_lang}: "
    "translate user-facing text fields into natural, spoken {target_lang} suitable for video subtitles (conversational tone, contextual nuance); "
    "{pronoun_guidance}"
    "never alter the structure, keys, or placeholders:\n\n"
    "{source_text}"
)

#: Tiêu đề khối ngữ cảnh tham chiếu (phần mở rộng của repo này, KHÔNG thuộc upstream).
_CONTEXT_HEADER = "Reference translations for context:"

#: Khối ràng buộc CỨNG cho Pipeline B, chèn **SAU** payload JSON.
#:
#: Vì sao đặt SAU payload chứ không nhét thêm vào câu lệnh phía trên: model sinh token theo
#: thứ tự, phần nằm gần lượt `assistant` nhất có trọng số cao nhất (recency). Ràng buộc đại từ
#: ở bản cũ nằm lọt giữa câu lệnh tiếng Anh và đã bị model bỏ qua trong log thật.
#:
#: Chỉ áp cho Pipeline B. Pipeline A (dịch từng câu, ưu tiên độ trễ) đã được siết ngay trong
#: `_PRONOUN_GUIDANCE_VI` và được `pronoun_guard` hậu kiểm ở tầng engine.
_BATCH_CONSTRAINTS_HEADER = (
    "\n\nHard constraints — the answer is REJECTED if any of these is violated:\n"
    "- Reply with the JSON object ONLY: no markdown fence, no commentary, no extra keys.\n"
    "- NEVER write the Vietnamese word \"mình\" — not standalone, nor in \"chúng mình\", "
    "\"bọn mình\", \"tụi mình\". First person = \"tôi\", second person = \"bạn\", we = \"chúng tôi\".\n"
    "- If the source sentence does not specify a subject, DO NOT add or invent a subject in the translation (keep it impersonal).\n"
    "- Keep the SAME speaker and the SAME pronouns as the reference context above."
)


def _build_batch_constraints(terms: str = "") -> str:
    """Khối ràng buộc cuối prompt batch; `terms` là gợi ý glossary (có thể rỗng)."""
    block = _BATCH_CONSTRAINTS_HEADER
    for line in (terms or "").splitlines():
        stripped = line.strip()
        if stripped:
            block += f"\n- {stripped}"
    return block


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


def _wrap_chat(user_content: str, no_think_prefix: str = "") -> str:
    """Bọc nội dung vào chat template Qwen CHỈ VỚI lượt ``user`` + ``assistant``.

    KHÔNG thêm lượt ``system``: cả 3 template chính thức của Index-Translate đều là
    nội dung của MỘT lượt người dùng, thêm system role sẽ lệch khỏi lúc fine-tune.
    """
    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{no_think_prefix}"
    )


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
        term: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def build_batch_prompt(
        self,
        sentences: List[str],
        source_lang: str,
        target_lang: str,
        format_type: str = "JSON",
        context: str = "",
        use_context: bool = False,
        terms: str = "",
    ) -> str:
        raise NotImplementedError

    def get_stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"]


class PipelineAPromptStrategy(PromptStrategy):
    """Prompt cho Pipeline A (Realtime Single Sentence Streaming).

    Dùng 2 trong 3 template chính thức của Index-Translate:

    * ``TEMPLATE_DEFAULT``     — mặc định.
    * ``TEMPLATE_TERMINOLOGY`` — khi truyền ``term`` (ràng buộc thuật ngữ).

    Model dùng Qwen làm base — có thinking token ``<think>``. Để dịch nhanh (no-think mode)
    ta ép output bắt đầu bằng ``<think>\\n\\n</think>\\n\\n`` ngay trong lượt assistant.
    Hỗ trợ bổ sung câu ngữ cảnh từ ContextManager (khi ``use_context=True``); khối ngữ cảnh
    được đặt TRƯỚC template để dòng hướng dẫn vẫn dính liền ``{source_text}`` như upstream.
    """

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = False,
        term: Optional[str] = None,
    ) -> str:
        # `source_lang` được giữ để tương thích API nhưng KHÔNG nhúng vào prompt:
        # không template chính thức nào có placeholder ngôn ngữ nguồn.
        tgt = resolve_lang_name(target_lang)
        source_text = (text or "").strip()

        pronoun_guide = _PRONOUN_GUIDANCE_VI if tgt == "Vietnamese" else ""
        if term and str(term).strip():
            instruction = TEMPLATE_TERMINOLOGY.format(
                target_lang=tgt,
                term=str(term).strip(),
                pronoun_guidance=pronoun_guide,
                source_text=source_text,
            )
        else:
            instruction = TEMPLATE_DEFAULT.format(
                target_lang=tgt,
                pronoun_guidance=pronoun_guide,
                source_text=source_text,
            )

        user_content = instruction
        if context and use_context:
            user_content = f"{_CONTEXT_HEADER}\n{context.strip()}\n\n{instruction}"

        return _wrap_chat(user_content, self._NO_THINK_PREFIX)

    def build_batch_prompt(
        self,
        sentences: List[str],
        source_lang: str,
        target_lang: str,
        format_type: str = "JSON",
        context: str = "",
        use_context: bool = False,
        terms: str = "",
    ) -> str:
        return PipelineBPromptStrategy().build_batch_prompt(
            sentences, source_lang, target_lang, format_type, context, use_context, terms
        )


class PipelineBPromptStrategy(PromptStrategy):
    """Prompt cho Pipeline B (Lookahead Batch Context Translation qua JSON).

    Dùng template chính thức loại 3 (``TEMPLATE_STRUCTURED``) với ``format_type="JSON"``,
    tuân thủ chuẩn Bilibili instTrans: giữ nguyên key số, cấu trúc và placeholder.
    Dịch gộp toàn bộ các câu trong một khối ASR thành 1 lần gọi LLM duy nhất,
    tối đa hóa tính liền mạch ngữ cảnh hội thoại và tiết kiệm 60-70% thời gian GPU.
    Hỗ trợ kế thừa khối ngữ cảnh hội thoại gần nhất (cross-block context injection).

    P4.1/P4.2 (2026-10-07): sau payload JSON có thêm khối **ràng buộc cứng** (`terms` là gợi ý
    glossary tên riêng/thuật ngữ). Đặt SAU payload vì lý do recency — xem
    `_BATCH_CONSTRAINTS_HEADER`.
    """

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = False,
        term: Optional[str] = None,
    ) -> str:
        return PipelineAPromptStrategy().build_prompt(
            text, source_lang, target_lang, context, use_context, term
        )

    def build_batch_prompt(
        self,
        sentences: List[str],
        source_lang: str,
        target_lang: str,
        format_type: str = "JSON",
        context: str = "",
        use_context: bool = False,
        terms: str = "",
    ) -> str:
        tgt = resolve_lang_name(target_lang)
        data: Dict[str, Any] = {str(i + 1): (s or "").strip() for i, s in enumerate(sentences)}
        json_text = json.dumps(data, ensure_ascii=False, indent=2)

        pronoun_guide = _PRONOUN_GUIDANCE_VI if tgt == "Vietnamese" else ""
        instruction = TEMPLATE_STRUCTURED.format(
            format_type=format_type,
            target_lang=tgt,
            pronoun_guidance=pronoun_guide,
            source_text=json_text,
        )

        # Khối ràng buộc đứng SAU payload (recency). `terms` rỗng ⇒ khối vẫn có, chỉ thiếu
        # dòng glossary — ràng buộc đại từ và "chỉ trả JSON" luôn được áp.
        instruction += _build_batch_constraints(terms)

        user_content = instruction
        if context and use_context:
            user_content = f"{_CONTEXT_HEADER}\n{context.strip()}\n\n{instruction}"

        return _wrap_chat(user_content, self._NO_THINK_PREFIX)


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
