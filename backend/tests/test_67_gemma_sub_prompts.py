"""Unit test kiểm thử module prompts_gemma.py dành cho model 17slever17/translate-gemma-4-sub-e4b-GGUF."""

import pytest
from backend.translation.prompts_gemma import (
    GemmaSubPromptStrategy,
    parse_context_history,
    _wrap_gemma4_chat,
)


def test_parse_context_history():
    ctx = "Hello -> Xin chào | How are you? -> Bạn khỏe không?"
    src, tgt = parse_context_history(ctx)
    assert src == "Hello\nHow are you?"
    assert tgt == "Xin chào\nBạn khỏe không?"

    # Rỗng
    empty_src, empty_tgt = parse_context_history("")
    assert empty_src == ""
    assert empty_tgt == ""


def test_parse_context_history_kieu_xuong_dong():
    """Hồi quy log 2026-10-10: `lookahead_handler` nối ngữ cảnh bằng `\\n`, không phải `|`.

    Bản cũ tách CHỈ theo `|` nên cả 2 cặp bị coi là 1 ⇒ `[PREVIOUS_TRANSLATION]` chứa lẫn
    câu nguồn kèm mũi tên của cặp thứ hai.
    """
    ctx = "新しいグリル買っちゃったんだよね。 -> Tôi vừa mua cái vỉ nướng mới đấy.\n本当好きですよね。 -> Thật sự rất thích món đó nhỉ."
    src, tgt = parse_context_history(ctx)

    assert src == "新しいグリル買っちゃったんだよね。\n本当好きですよね。"
    assert tgt == "Tôi vừa mua cái vỉ nướng mới đấy.\nThật sự rất thích món đó nhỉ."
    # KHÔNG được còn mũi tên hay câu nguồn lẫn trong khối bản dịch
    assert "->" not in tgt
    assert "新しいグリル" not in tgt


def test_parse_context_history_tron_hai_kieu_va_dong_trong():
    ctx = "  A -> B  \n\n  |  C -> D \n"
    src, tgt = parse_context_history(ctx)
    assert src == "A\nC"
    assert tgt == "B\nD"


def test_gemma_prompt_no_context():
    strategy = GemmaSubPromptStrategy(style="friendly")
    prompt = strategy.build_prompt("Yeah, well... I changed my mind.", "en", "vi")

    # Kiểm tra cấu trúc chat template Gemma 4
    assert "<|turn>system\n" in prompt
    assert "<turn|>\n<|turn>user\n" in prompt
    assert "<turn|>\n<|turn>model\n" in prompt

    # Kiểm tra nội dung system message
    assert "TASK: Translate English subtitles into Vietnamese." in prompt
    assert "STYLE: friendly." in prompt
    assert "Translate only CURRENT_SOURCE." in prompt

    # Kiểm tra user message không context
    assert "[CURRENT_SOURCE]\nYeah, well... I changed my mind." in prompt
    assert "[PREVIOUS_SOURCE]" not in prompt


def test_gemma_prompt_with_context():
    strategy = GemmaSubPromptStrategy(style="friendly")
    context_str = "I thought you said you weren't coming. -> Tôi tưởng bạn nói không đến."
    prompt = strategy.build_prompt(
        text="Yeah, well... I changed my mind.",
        source_lang="en",
        target_lang="vi",
        context=context_str,
        use_context=True,
    )

    # Có context
    assert "[PREVIOUS_SOURCE]\nI thought you said you weren't coming." in prompt
    assert "[PREVIOUS_TRANSLATION]\nTôi tưởng bạn nói không đến." in prompt
    assert "[CURRENT_SOURCE]\nYeah, well... I changed my mind." in prompt


def test_gemma_prompt_with_term():
    strategy = GemmaSubPromptStrategy(style="official")
    prompt = strategy.build_prompt(
        text="Misaki is here.",
        source_lang="ja",
        target_lang="vi",
        term="Misaki -> Mỹ Kỳ",
    )
    assert "STYLE: official." in prompt
    assert "fixed terminology rendering: Misaki -> Mỹ Kỳ" in prompt


def test_gemma_stop_tokens():
    strategy = GemmaSubPromptStrategy()
    tokens = strategy.get_stop_tokens()
    assert "<turn|>" in tokens
    assert "<|turn>" in tokens
    assert "<eos>" in tokens
    assert "</output>" in tokens
    assert "\n[CURRENT_SOURCE]" in tokens


def test_clean_gemma_output():
    from backend.translation.engine import clean_translation_tags

    raw1 = "<output> Không sao, cứ dùng đi. </output>"
    assert clean_translation_tags(raw1) == "Không sao, cứ dùng đi."

    raw2 = "羽田さんは私に下心を向けることはなかった。\n\n</think>"
    assert clean_translation_tags(raw2) == "羽田さんは私に下心を向けることはなかった。"

    raw3 = "<|turn>model\nXin chào bạn<turn|>"
    assert clean_translation_tags(raw3) == "Xin chào bạn"


def test_gemma_strategy_supports_json_batch():
    """Pipeline B của Gemma-Sub dùng JSON giống Index-Translate (không tuần tự dây chuyền)."""
    strategy = GemmaSubPromptStrategy()
    assert strategy.supports_json_batch is True


def test_gemma_batch_prompt_gui_json_khoa_so():
    """Prompt batch phải chứa payload JSON khoá số và yêu cầu trả lại đúng JSON đó."""
    strategy = GemmaSubPromptStrategy(style="friendly")
    prompt = strategy.build_batch_prompt(
        ["Yeah, well... I changed my mind.", "Are you serious?"], "en", "vi"
    )

    # Vẫn giữ chat template Gemma 4
    assert "<|turn>system\n" in prompt
    assert "<turn|>\n<|turn>model\n" in prompt

    # Payload JSON giống Index-Translate: khoá số + giữ nguyên câu nguồn
    assert "[CURRENT_SOURCE]\n{" in prompt
    assert '"1": "Yeah, well... I changed my mind."' in prompt
    assert '"2": "Are you serious?"' in prompt

    # Ràng buộc định dạng đầu ra
    assert "same keys" in prompt
    assert "raw JSON object only" in prompt
    # KHÔNG còn định dạng đánh số [1]/[2] của bản tuần tự cũ
    assert "[1] Yeah" not in prompt


def test_gemma_batch_prompt_voi_context_va_speaker_tags():
    strategy = GemmaSubPromptStrategy()
    prompt = strategy.build_batch_prompt(
        ["Hello", "Bye"],
        "en",
        "vi",
        context="Hi there -> Chào bạn",
        use_context=True,
        terms="Misaki -> Mỹ Kỳ",
        speaker_tags=["speaker 1", "speaker 2"],
    )

    assert "[PREVIOUS_SOURCE]\nHi there" in prompt
    assert "[PREVIOUS_TRANSLATION]\nChào bạn" in prompt
    assert '"speaker 1": "Hello"' in prompt
    assert "Misaki -> Mỹ Kỳ" in prompt


def test_gemma_quy_tac_nhat_quan_dai_tu_va_chu_ngu():
    """Quy tắc SubWave chuẩn: giữ ngữ điệu hội thoại, ngôi xưng nhất quán, không tự bịa chủ ngữ."""
    strategy = GemmaSubPromptStrategy()

    single = strategy.build_prompt("Yeah, well... I changed my mind.", "en", "vi")
    assert "keep the conversational tone and the speaker's pronouns consistent across segments" in single
    assert "if a segment omits the subject, do not invent one" in single
    assert "RULES:" in single
    assert "STYLE: neutral." in single

    batch = strategy.build_batch_prompt(["Hello", "World"], "ja", "vi")
    assert (
        "RULES: keep the original segment count and order; "
        "translate every segment, never merge, split or drop segments; "
        "keep the conversational tone and the speaker's pronouns consistent across segments; "
        "if a segment omits the subject, do not invent one."
    ) in batch
    assert "STYLE: neutral." in batch
    assert "Translate only CURRENT_SOURCE. PREVIOUS_SOURCE and PREVIOUS_TRANSLATION are context only." in batch
    assert "INPUT FORMAT:" in batch
    assert "OUTPUT FORMAT:" in batch


def test_gemma_batch_prompt_ngu_canh_kieu_xuong_dong():
    """Ngữ cảnh THẬT của `lookahead_handler` (nối bằng `\\n`) phải ra 2 khối song song."""
    strategy = GemmaSubPromptStrategy()
    ctx = (
        "新しいグリル買っちゃったんだよね。 -> Tôi vừa mua cái vỉ nướng mới đấy.\n"
        "本当好きですよね。 -> Thật sự rất thích món đó nhỉ."
    )
    prompt = strategy.build_batch_prompt(
        ["ちなみに誰か連れてきてもいいんだぞ。", "いないのか？"],
        "ja",
        "vi",
        context=ctx,
        use_context=True,
    )

    assert (
        "[PREVIOUS_SOURCE]\n新しいグリル買っちゃったんだよね。\n本当好きですよね。\n\n"
        "[PREVIOUS_TRANSLATION]\nTôi vừa mua cái vỉ nướng mới đấy.\nThật sự rất thích món đó nhỉ."
    ) in prompt

    # Khối bản dịch tham chiếu KHÔNG được lẫn câu nguồn hay mũi tên
    block = prompt.split("[PREVIOUS_TRANSLATION]")[1].split("[CURRENT_SOURCE]")[0]
    assert "->" not in block
    assert "新しいグリル" not in block


def test_gemma_glossary_co_mat_o_ca_hai_pipeline(allow_real_translation_methods, monkeypatch, tmp_path):
    """Glossary phải chảy vào prompt THẬT của cả Pipeline A (term) lẫn Pipeline B (terms)."""
    from backend.translation.engine import GGUFTranslator

    path = tmp_path / "glossary.yaml"
    path.write_text("terms:\n  美咲: Mỹ Kỳ\n", encoding="utf-8")

    from backend.translation.glossary import TranslationGlossary

    TranslationGlossary.reset_instance()
    llm = _CapturingLlama('{"1": "Mỹ Kỳ đây.", "2": "Chào."}')
    t = _wire_engine(monkeypatch, llm, "translate-gemma-4-sub", GemmaSubPromptStrategy())
    t.cfg.glossary_file = str(path)
    try:
        t.translate_batch_sync(["美咲さん、おはよう", "はい。"], "ja", "vi")
        assert "美咲 -> Mỹ Kỳ" in llm.calls[0][0], "Pipeline B thiếu glossary"

        llm.calls.clear()
        t.translate_batch_sync(["美咲さん、おはよう"], "ja", "vi")  # 1 câu ⇒ Pipeline A
        assert "美咲->Mỹ Kỳ" in llm.calls[0][0], "Pipeline A thiếu glossary"
    finally:
        TranslationGlossary.reset_instance()


def test_gemma_batch_output_parse_duoc_bang_json_parser():
    """Đầu ra JSON của model phải parse được bằng `_parse_batch_json` của engine."""
    from backend.translation.engine import _parse_batch_json

    raw = '{"1": "Ừ, thì... tôi đổi ý rồi.", "2": "Bạn nghiêm túc đấy à?"}'
    parsed = _parse_batch_json(raw, 2)
    assert parsed == ["Ừ, thì... tôi đổi ý rồi.", "Bạn nghiêm túc đấy à?"]


class _CapturingLlama:
    """LLM giả: ghi lại prompt của từng lượt gọi."""

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.calls.append((prompt, kwargs))
        return {
            "choices": [{"text": self.payload, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10},
        }

    def close(self):
        pass


def _wire_engine(monkeypatch, llm, model_key: str, strategy):
    from types import SimpleNamespace

    from backend.translation.engine import GGUFTranslator

    t = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", llm)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", model_key)
    t.canonical_key = model_key
    t.prompt_strategy = strategy
    t.cfg = SimpleNamespace(
        max_tokens=128, temperature=None, top_p=None, top_k=None, repetition_penalty=None,
        use_context=False, glossary_file="", glossary={}, glossary_derive_names=True,
    )
    return t


def test_engine_goi_gemma_batch_dung_mot_lan(allow_real_translation_methods, monkeypatch):
    """Chốt hành vi: batch của Gemma-Sub đi đường JSON 1 lần gọi, KHÔNG tuần tự dây chuyền."""
    llm = _CapturingLlama('{"1": "Xin chào", "2": "Tạm biệt"}')
    t = _wire_engine(monkeypatch, llm, "gemma-test", GemmaSubPromptStrategy())

    out = t.translate_batch_sync(["Hello", "Bye"], "en", "vi")

    assert out == ["Xin chào", "Tạm biệt"]
    assert len(llm.prompts) == 1, "phải gọi LLM đúng MỘT lần (JSON batch), không dịch tuần tự"
    assert '"1": "Hello"' in llm.prompts[0]

