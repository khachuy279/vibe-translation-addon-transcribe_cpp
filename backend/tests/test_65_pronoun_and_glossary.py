"""test_65 — P4.1 (ép ràng buộc đại từ) & P4.2 (glossary tên riêng/thuật ngữ).

Bối cảnh (log thật 2026-10-07, xem `report/12_qwen35_translation_ab`):

1. Prompt đã ghi rõ ``never use the word mình anywhere`` nhưng Index-Translate-9B vẫn trả
   ``"tôi còn nói là **mình** chắc chắn không thể quay lại…"``.
   ⇒ Ràng buộc ở prompt là *hy vọng*; cần hậu kiểm ở tầng văn bản.
2. ``美咲さん`` bị dịch sai tên (``Meisaka`` / ``Misa`` / ``Miaki-san`` tuỳ model).
   ⇒ Cần bảng "cách viết cố định" — đúng cơ chế ``TEMPLATE_TERMINOLOGY`` của upstream nhưng
   chưa từng được nối vào Pipeline B.

Bộ test này chốt 4 thứ:
  * `pronoun_guard` KHÔNG phá các cụm hợp lệ chứa "mình" (một mình, chính mình, mình ơi…);
  * `glossary` nạp file + ghi đè runtime + phát hiện tên riêng;
  * prompt Pipeline B có khối ràng buộc cứng và dòng glossary;
  * engine THẬT sửa đầu ra và vẫn chạy được với prompt strategy cũ (không có tham số `terms`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.translation.glossary import TranslationGlossary
from backend.translation.prompts import get_prompt_strategy
from backend.translation.pronoun_guard import (
    contains_replaceable_minh,
    count_occurrences,
    enforce_no_minh,
)

# ─────────────────────────────────────────────────────────── P4.1: pronoun guard

#: (đầu vào, kỳ vọng) — nửa đầu là ca PHẢI sửa, nửa sau là ca PHẢI GIỮ NGUYÊN.
PRONOUN_CASES = [
    # ── phải sửa ──
    ("tôi còn nói là mình chắc chắn không thể quay lại", "tôi còn nói là tôi chắc chắn không thể quay lại"),
    ("Mình nghĩ là nên đi.", "Tôi nghĩ là nên đi."),
    ("chúng mình đi thôi", "chúng ta đi thôi"),
    ("bọn mình đã bàn rồi", "chúng ta đã bàn rồi"),
    ("tụi mình về nhé", "chúng ta về nhé"),
    ("để mình làm cho", "để tôi làm cho"),
    ("nếu mình không đi thì sao?", "nếu tôi không đi thì sao?"),
    # ── phải giữ nguyên (thay bừa sẽ thành câu sai ngữ pháp) ──
    ("anh ấy tự mình làm", "anh ấy tự mình làm"),
    ("chính mình cũng không biết", "chính mình cũng không biết"),
    ("nhà mình hôm nay có khách", "nhà mình hôm nay có khách"),
    ("mình ơi, về nhà thôi", "mình ơi, về nhà thôi"),
    ("bản thân mình phải cố gắng", "bản thân mình phải cố gắng"),
    ("để mình một mình", "để tôi một mình"),  # cái đầu sửa, cái sau giữ
    ("mỗi mình nó biết", "mỗi mình nó biết"),
    ("không có gì đâu", "không có gì đâu"),
    ("", ""),
]


@pytest.mark.parametrize("src,want", PRONOUN_CASES)
def test_enforce_no_minh(src, want):
    assert enforce_no_minh(src) == want


def test_enforce_no_minh_tra_nguyen_van_khi_khong_co_gi():
    """Không có gì để sửa ⇒ trả ĐÚNG object cũ (tầng gọi so sánh `is`/`!=` rẻ)."""
    text = "Câu này không có đại từ cấm."
    assert enforce_no_minh(text) is text


def test_contains_replaceable_minh_va_count():
    assert contains_replaceable_minh("tôi nói là mình sẽ đi") is True
    assert contains_replaceable_minh("một mình thôi") is False
    assert contains_replaceable_minh("") is False
    # `count_occurrences` đếm MỌI dạng (kể cả dạng bị chặn) — dùng cho log/metrics.
    assert count_occurrences("mình và mình") == 2
    assert count_occurrences("một mình") == 1


# ───────────────────────────────────────────────────────────── P4.2: glossary

@pytest.fixture(autouse=True)
def _fresh_glossary():
    """Mỗi test dùng cache glossary sạch (cache theo mtime của file)."""
    TranslationGlossary.reset_instance()
    yield
    TranslationGlossary.reset_instance()


def _write_glossary(tmp_path, body: str):
    path = tmp_path / "glossary.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_glossary_doc_file_va_khop_chuoi_con(tmp_path):
    path = _write_glossary(tmp_path, "terms:\n  美咲: Misaki\n  田中: Tanaka\n")
    cfg = SimpleNamespace(glossary_file=str(path), glossary={})
    g = TranslationGlossary.get_instance()

    assert "美咲 -> Misaki" in g.build_hint(["美咲さんとはうまくいってるみたいですね。"], cfg)
    # Khớp CHUỖI CON: `美咲` phải khớp trong `美咲さん` (chỉ cần ghép kính ngữ).
    assert g.matched_terms(["美咲さん、おはよう"], cfg) == {"美咲": "Misaki"}
    # Mục không xuất hiện thì KHÔNG được gửi vào prompt (tiết kiệm token).
    assert "田中" not in g.build_hint(["美咲さんだけ"], cfg)
    # Pipeline A: chuỗi `nguồn->đích` cho TEMPLATE_TERMINOLOGY.
    assert g.terms_for_single("美咲さん、おはよう", cfg) == "美咲->Misaki"
    assert g.terms_for_single("こんにちは", cfg) == ""


def test_glossary_runtime_ghi_de_file(tmp_path):
    path = _write_glossary(tmp_path, "terms:\n  美咲: Misaki\n")
    cfg = SimpleNamespace(glossary_file=str(path), glossary={"美咲": "Mỹ Sako"})
    g = TranslationGlossary.get_instance()
    assert g.matched_terms(["美咲さん"], cfg) == {"美咲": "Mỹ Sako"}


def test_glossary_doc_lai_khi_file_doi(tmp_path):
    """Sửa file là có hiệu lực ngay, KHÔNG cần restart backend."""
    import os
    import time

    path = _write_glossary(tmp_path, "terms:\n  美咲: Misaki\n")
    cfg = SimpleNamespace(glossary_file=str(path), glossary={})
    g = TranslationGlossary.get_instance()
    assert g.matched_terms(["美咲さん"], cfg) == {"美咲": "Misaki"}

    path.write_text("terms:\n  美咲: Misa\n", encoding="utf-8")
    os.utime(path, (time.time() + 5, time.time() + 5))  # ép mtime đổi (FS độ phân giải thô)
    assert g.matched_terms(["美咲さん"], cfg) == {"美咲": "Misa"}


def test_glossary_file_hong_khong_lam_chet_dich(tmp_path):
    """YAML hỏng ⇒ bỏ qua nguồn file, KHÔNG ném exception ra đường dịch."""
    path = _write_glossary(tmp_path, "terms: [khong-phai-mapping\n")
    cfg = SimpleNamespace(glossary_file=str(path), glossary={"美咲": "Misaki"})
    g = TranslationGlossary.get_instance()
    assert g.matched_terms(["美咲さん"], cfg) == {"美咲": "Misaki"}


def test_glossary_file_thieu_la_binh_thuong(tmp_path):
    cfg = SimpleNamespace(glossary_file=str(tmp_path / "khong-ton-tai.yaml"), glossary={},
                          glossary_derive_names=False)
    g = TranslationGlossary.get_instance()
    assert g.load_terms(cfg) == {}
    assert g.build_hint(["美咲さん"], cfg) == ""


def test_detect_names_katakana_va_kinh_ngu():
    g = TranslationGlossary.get_instance()
    assert "ミサキ" in g.detect_names(["ミサキちゃんが来た"])
    assert g.detect_names(["美咲さんとはうまくいってるみたいですね。"]) == ["美咲"]
    assert g.detect_names(["田中くんと林さん"]) == ["田中"]

    # Hồi quy: `今日美咲さん` có run kanji dài 4 ⇒ phải cắt còn 2 kanji CUỐI, không lấy
    # cả cụm `今日美咲` (bug rất dễ mắc khi viết regex tham lam).
    assert g.detect_names(["今日美咲さんが来た"]) == ["美咲"]


def test_build_hint_chi_them_dong_nhat_quan_khi_co_tu_2_cau(tmp_path):
    """Một câu thì không có gì để mâu thuẫn ⇒ đừng tốn token cho ràng buộc nhất quán.

    Tắt suy romaji để cô lập đúng nhánh "chỉ ràng buộc nhất quán" (P4.2a) — nhánh suy romaji
    (P4.2b) có test riêng ở `test_66_name_romaji.py`.
    """
    cfg = SimpleNamespace(glossary_file="", glossary={}, glossary_derive_names=False)
    g = TranslationGlossary.get_instance()
    assert g.build_hint(["ミサキちゃんが来た。"], cfg) == ""
    hint = g.build_hint(["ミサキちゃんが来た。", "ミサキちゃんは元気だ。"], cfg)
    assert "Proper nouns in the source (ミサキ)" in hint


def test_build_hint_khong_lap_khi_glossary_da_an_dinh(tmp_path):
    path = _write_glossary(tmp_path, "terms:\n  美咲: Misaki\n")
    cfg = SimpleNamespace(glossary_file=str(path), glossary={})
    g = TranslationGlossary.get_instance()
    texts = ["美咲さんとはうまくいってるみたいですね。", "もう絶対会社員にも戻れないって言ってるよ。"]
    hint = g.build_hint(texts, cfg)
    assert "Fixed renderings" in hint
    assert "Proper nouns" not in hint, "tên đã có trong glossary thì không cần dòng nhất quán"


# ─────────────────────────────────────────────────────────── Prompt Pipeline B

def test_batch_prompt_co_khoi_rang_buoc_cung():
    prompt = get_prompt_strategy("pipeline_b").build_batch_prompt(["Hello", "World"], "en", "vi")
    # Không được phá các assertion của test_05 (template upstream giữ nguyên).
    assert "Translate the following JSON data into Vietnamese" in prompt
    assert '"1": "Hello"' in prompt
    # Ràng buộc cứng nằm SAU payload (recency) và trước lượt assistant.
    assert "Hard constraints" in prompt
    assert prompt.index('"2": "World"') < prompt.index("Hard constraints")
    assert "NEVER write the Vietnamese word" in prompt
    # Ràng buộc đại từ đã được siết: nêu đích danh các dạng ghép.
    assert "chúng mình" in prompt and "bọn mình" in prompt
    assert "<|im_start|>assistant" in prompt


def test_batch_prompt_chen_glossary_terms():
    terms = "Fixed renderings — use these EXACTLY: 美咲 -> Misaki."
    prompt = get_prompt_strategy("pipeline_b").build_batch_prompt(
        ["美咲さん、おはよう", "はい"], "ja", "vi", terms=terms
    )
    assert "美咲 -> Misaki" in prompt
    assert prompt.index("美咲 -> Misaki") > prompt.index("Hard constraints")


def test_batch_prompt_khong_terms_van_chay():
    """`terms` rỗng (chưa khai báo glossary) không được làm vỡ prompt hay thêm rác."""
    prompt = get_prompt_strategy("pipeline_b").build_batch_prompt(["Hello", "World"], "en", "vi")
    assert "Fixed renderings" not in prompt
    assert "Hard constraints" in prompt


def test_single_prompt_khong_bi_doi_hinh_dang_them():
    """Pipeline A chỉ siết câu chữ đại từ — KHÔNG thêm khối ràng buộc (ưu tiên độ trễ)."""
    prompt = get_prompt_strategy("pipeline_a").build_prompt("Hello world", "en", "vi")
    assert "Translate the following text into Vietnamese" in prompt
    assert "Hard constraints" not in prompt
    assert "chúng mình" in prompt  # ràng buộc đại từ đã siết vẫn có mặt


# ─────────────────────────────────────────────────────── Tích hợp vào engine

def _batch_translator(monkeypatch, llm, cfg):
    from backend.translation.engine import GGUFTranslator

    t = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", llm)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", "m-test")
    monkeypatch.setattr(t.registry, "get_model", lambda k: {})
    t.canonical_key = "m-test"
    t.cfg = cfg
    t.prompt_strategy = get_prompt_strategy("pipeline_b")
    return t


def _cfg(tmp_path, **over):
    base = dict(
        max_tokens=128, temperature=None, top_p=None, top_k=None, repetition_penalty=None,
        use_context=False, enforce_pronoun_policy=True, glossary_file="", glossary={},
        glossary_derive_names=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _CapturingLlama:
    """LLM giả: ghi lại prompt và trả về JSON cố định."""

    closed = False

    def __init__(self, payload: str):
        self.payload = payload
        self.prompts: list[str] = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return {"choices": [{"text": self.payload, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 10}}

    def close(self):
        pass


BATCH_WITH_MINH = (
    '{\n  "1": "Có vẻ như mối quan hệ của anh với Misaki-san đang tốt đẹp nhỉ.",\n'
    '  "2": "tôi còn nói là mình chắc chắn không thể quay lại làm nhân viên nữa."\n}'
)


def test_batch_engine_ep_rang_buoc_dai_tu(allow_real_translation_methods, monkeypatch, tmp_path):
    """Đầu ra batch còn `mình` ⇒ engine PHẢI sửa trước khi trả về."""
    llm = _CapturingLlama(BATCH_WITH_MINH)
    t = _batch_translator(monkeypatch, llm, _cfg(tmp_path))

    out = t.translate_batch_sync(["美咲さんとはうまくいってるみたいですね。", "もう絶対会社員にも戻れないって言ってるよ。"], "ja", "vi")
    assert len(out) == 2
    assert "mình" not in out[1], out[1]
    assert "tôi chắc chắn" in out[1]


def test_batch_engine_tat_duoc_rang_buoc(allow_real_translation_methods, monkeypatch, tmp_path):
    """`enforce_pronoun_policy=False` ⇒ giữ nguyên đầu ra của model (để A/B)."""
    llm = _CapturingLlama(BATCH_WITH_MINH)
    t = _batch_translator(monkeypatch, llm, _cfg(tmp_path, enforce_pronoun_policy=False))

    out = t.translate_batch_sync(["あ", "い"], "ja", "vi")
    assert "mình" in out[1]


def test_batch_engine_bom_glossary_vao_prompt(allow_real_translation_methods, monkeypatch, tmp_path):
    """Glossary khớp được đưa vào prompt THẬT gửi cho model (không chỉ tồn tại trong module)."""
    path = tmp_path / "glossary.yaml"
    path.write_text("terms:\n  美咲: Misaki\n", encoding="utf-8")
    llm = _CapturingLlama(BATCH_WITH_MINH)
    t = _batch_translator(monkeypatch, llm, _cfg(tmp_path, glossary_file=str(path)))

    t.translate_batch_sync(
        ["美咲さんとはうまくいってるみたいですね。", "もう絶対会社員にも戻れないって言ってるよ。"], "ja", "vi"
    )
    assert llm.prompts, "model chưa được gọi"
    assert "美咲 -> Misaki" in llm.prompts[0]


def test_engine_chiu_duoc_strategy_cu_khong_co_terms(allow_real_translation_methods, monkeypatch, tmp_path):
    """Prompt strategy bên thứ ba chưa có tham số `terms`/`term` ⇒ fallback, KHÔNG nổ.

    Đây là hồi quy thật: `_FakeStrategy` trong `test_32` có chữ ký 5 tham số cố định.
    """
    from backend.translation.engine import GGUFTranslator

    class _LegacyStrategy:
        def build_prompt(self, text, source_lang, target_lang, context, use_context):
            return f"legacy:{text}"

        def build_batch_prompt(self, sentences, source_lang, target_lang, context="", use_context=False):
            return "legacy-batch"

        def get_stop_tokens(self):
            return ["</s>"]

    path = tmp_path / "glossary.yaml"
    path.write_text("terms:\n  美咲: Misaki\n", encoding="utf-8")
    llm = _CapturingLlama(BATCH_WITH_MINH)

    t = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", llm)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", "m-test")
    monkeypatch.setattr(t.registry, "get_model", lambda k: {})
    t.canonical_key = "m-test"
    t.cfg = _cfg(tmp_path, glossary_file=str(path))
    t.prompt_strategy = _LegacyStrategy()

    out = t.translate_batch_sync(["美咲さん、おはよう", "もう絶対会社員にも戻れないって言ってるよ。"], "ja", "vi")
    assert llm.prompts == ["legacy-batch"]
    assert "mình" not in out[1]


def test_single_engine_ep_rang_buoc_dai_tu(allow_real_translation_methods, monkeypatch, tmp_path):
    from backend.translation.engine import GGUFTranslator

    llm = _CapturingLlama("tôi còn nói là mình chắc chắn không thể quay lại")
    t = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", llm)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", "m-test")
    monkeypatch.setattr(t.registry, "get_model", lambda k: {})
    t.canonical_key = "m-test"
    t.cfg = _cfg(tmp_path)
    t.prompt_strategy = get_prompt_strategy("pipeline_a")

    res = t._translate_sync("もう絶対会社員にも戻れないって言ってるよ。", "ja", "vi")
    assert res["translated_text"] == "tôi còn nói là tôi chắc chắn không thể quay lại"


def test_single_engine_truyen_glossary_qua_tem(allow_real_translation_methods, monkeypatch, tmp_path):
    """Pipeline A dùng TEMPLATE_TERMINOLOGY khi có mục glossary khớp."""
    from backend.translation.engine import GGUFTranslator

    path = tmp_path / "glossary.yaml"
    path.write_text("terms:\n  美咲: Misaki\n", encoding="utf-8")
    llm = _CapturingLlama("Misaki-san")
    t = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", llm)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", "m-test")
    monkeypatch.setattr(t.registry, "get_model", lambda k: {})
    t.canonical_key = "m-test"
    t.cfg = _cfg(tmp_path, glossary_file=str(path))
    t.prompt_strategy = get_prompt_strategy("pipeline_a")

    t._translate_sync("美咲さんとはうまくいってるみたいですね。", "ja", "vi")
    assert "fixed rendering for 美咲->Misaki" in llm.prompts[0]
