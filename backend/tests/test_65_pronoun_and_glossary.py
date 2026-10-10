"""test_65 — P4.2 (glossary tên riêng/thuật ngữ) & hợp đồng prompt của Pipeline A/B.

Bối cảnh (log thật 2026-10-07, xem `report/12_qwen35_translation_ab`):
``美咲さん`` bị dịch sai tên (``Meisaka`` / ``Misa`` / ``Miaki-san`` tuỳ model).
⇒ Cần bảng "cách viết cố định" — đúng cơ chế ``TEMPLATE_TERMINOLOGY`` của upstream nhưng
chưa từng được nối vào Pipeline B.

LỊCH SỬ P4.1 — ĐÃ GỠ (quyết định 2026-10-10): bản cũ ép đại từ ("tôi"/"bạn") ở HAI lớp —
nhét ``_PRONOUN_GUIDANCE_VI`` vào cả 3 template Index, và hậu kiểm bằng ``pronoun_guard.py``
ở tầng engine. Cả hai lớp đều bị xoá: ràng buộc chỉ nằm ở prompt thì model **vẫn bỏ qua**
(log thật trong ``report/13_pronoun_and_glossary``), còn hậu kiểm văn bản thì **phá câu hợp
lệ** (``một mình``, ``chính mình``, ``nhà mình``…). Việc chọn đại từ do model quyết định
theo ngữ cảnh. Test dưới đây chốt việc **KHÔNG** ép đại từ ở bất kỳ pipeline nào.

Bộ test này chốt 3 thứ:
  * `glossary` nạp file + ghi đè runtime + phát hiện tên riêng;
  * prompt Pipeline B có khối ràng buộc cứng + dòng glossary, Pipeline A KHÔNG thêm khối nào;
  * engine giữ NGUYÊN đầu ra của model và vẫn chạy được với prompt strategy cũ (không có `terms`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.translation.glossary import TranslationGlossary
from backend.translation.prompts import get_prompt_strategy, resolve_lang_name


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
    """Pipeline A: câu chữ ĐÚNG template upstream, KHÔNG thêm khối ràng buộc nào."""
    prompt = get_prompt_strategy("pipeline_a").build_prompt("Hello world", "en", "vi")
    assert "Translate the following text into Vietnamese" in prompt
    assert "Hard constraints" not in prompt
    assert "Style constraints" not in prompt


# ──────────────────────────── KHÔNG ép đại từ tôi/bạn (quyết định 2026-10-10)

def test_khong_ep_dai_tu_o_ca_hai_ho_model():
    """Ràng buộc đại từ đã bị gỡ khỏi MỌI pipeline và không được tái xuất hiện.

    Hồi quy: bản cũ nhét ``Use ONLY "tôi" for the first person`` (Pipeline A) và
    ``NEVER write the Vietnamese word "mình"`` + ``chúng mình``/``bọn mình``/``tụi mình``
    (Pipeline B) vào template Index.
    """
    prompts = [
        get_prompt_strategy("pipeline_a").build_prompt("Hello", "en", "vi"),
        get_prompt_strategy("pipeline_b").build_batch_prompt(["Hello", "World"], "en", "vi"),
        get_prompt_strategy("gemma-sub").build_prompt("Hello", "en", "vi"),
        get_prompt_strategy("gemma-sub").build_batch_prompt(["Hello", "World"], "en", "vi"),
    ]
    for prompt in prompts:
        assert "Style constraints" not in prompt
        assert "Use ONLY" not in prompt
        assert '"tôi"' not in prompt
        assert '"bạn"' not in prompt
        assert "chúng mình" not in prompt
        assert "NEVER write the Vietnamese word" not in prompt


# ─────────────────────────────────────── resolve_lang_name (mã vùng / tên đầy đủ)

def test_resolve_lang_name_chuan_hoa_ma_vung_va_ten_day_du():
    """``vi-VN`` / ``VI`` / ``Vietnamese`` phải CÙNG ra "Vietnamese" (trước đây ra "English")."""
    for code in ("vi", "VI", "vi-VN", "vi_VN", "Vietnamese", "vietnamese"):
        assert resolve_lang_name(code) == "Vietnamese"
    assert resolve_lang_name("zh-CN") == "Chinese"
    assert resolve_lang_name("ja") == "Japanese"
    assert resolve_lang_name("en") == "English"
    # Mã lạ vẫn rơi về mặc định cũ — hành vi không đổi.
    assert resolve_lang_name("auto") == "English"


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
        use_context=False, glossary_file="", glossary={},
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


def test_batch_engine_giu_nguyen_dau_ra(allow_real_translation_methods, monkeypatch, tmp_path):
    """Không còn ép ràng buộc đại từ: đầu ra của model được giữ nguyên."""
    llm = _CapturingLlama(BATCH_WITH_MINH)
    t = _batch_translator(monkeypatch, llm, _cfg(tmp_path))

    out = t.translate_batch_sync(["美咲さんとはうまくいってるみたいですね。", "もう絶対会社員にも戻れないって言ってるよ。"], "ja", "vi")
    assert len(out) == 2
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
    assert len(out) == 2


def test_single_engine_giu_nguyen_dau_ra(allow_real_translation_methods, monkeypatch, tmp_path):
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
    assert res["translated_text"] == "tôi còn nói là mình chắc chắn không thể quay lại"


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
