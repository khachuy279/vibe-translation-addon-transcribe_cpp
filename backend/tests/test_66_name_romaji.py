"""test_66 — P4.2b: romaji hoá tên riêng Nhật **từ chuỗi NGUỒN**.

Vì sao KHÔNG lấy cách viết tên từ bản dịch (ý tưởng "tự học glossary từ `orig -> trans`"):

1. **Vòng lặp tự tham chiếu** — cặp `orig -> trans` do chính model cần ràng buộc sinh ra; lấy nó
   làm chuẩn là khoá cứng sai số (`美咲` bị bịa thành `Meisaka` một lần ⇒ lỗi hệ thống mãi).
2. **Không xác định được ánh xạ** — không biết token đích nào ứng với token nguồn nào.
3. **Sai số tương quan** — hỏi lại chính model đó không mang thêm thông tin.

Cách viết một tên Nhật là **hàm xác định của chuỗi nguồn**: `美咲` đọc là *misaki* bất kể model
dịch nó thành gì. Nhờ vậy cùng một tên luôn ra cùng cách viết ở mọi khối — nhất quán mà
**không cần lưu trạng thái, không cần ghi file, không có gì để "học"**.

Số đo thật (Qwen3.5-4B, glossary RỖNG, `debug/romaji_verify.py`):

    nhóm tên riêng   suy tên TẮT: 2/4   (田中 -> "Thầy Trung", 美咲 -> "cô Mai")
                     suy tên BẬT: 4/4   (Tanaka, Misaki, Sato, Sakurai)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.translation import romaji
from backend.translation.glossary import TranslationGlossary
from backend.translation.romaji import romanize


@pytest.fixture(autouse=True)
def _fresh_glossary():
    TranslationGlossary.reset_instance()
    yield
    TranslationGlossary.reset_instance()


def _cfg(**over):
    base = dict(
        max_tokens=128, temperature=None, top_p=None, top_k=None, repetition_penalty=None,
        use_context=False, enforce_pronoun_policy=True,
        glossary_file="", glossary={}, glossary_derive_names=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ─────────────────────────────────────────────────────── romaji hoá cơ bản

#: Tên kanji — phải khớp cách đọc phổ biến, và dùng kiểu `passport` (Sato, KHÔNG phải Satou).
KANJI_NAMES = [
    ("美咲", "Misaki"),
    ("田中", "Tanaka"),
    ("佐藤", "Sato"),          # passport: Sato; hepburn sẽ ra Satou
    ("桜井", "Sakurai"),
    ("山田", "Yamada"),
    ("長谷川", "Hasegawa"),
    ("中村", "Nakamura"),
    ("高橋", "Takahashi"),
]

KATAKANA_NAMES = [
    ("ミサキ", "Misaki"),
    ("タナカ", "Tanaka"),
    ("ジョン", "Jon"),
    ("アメリア", "Ameria"),
]


@pytest.mark.parametrize("src,want", KANJI_NAMES + KATAKANA_NAMES)
def test_romanize_ten_nhat(src, want):
    assert romanize(src) == want


@pytest.mark.parametrize("bad", ["", "   ", "Misaki", "hello world", "123", "ニューヨーク州", None])
def test_romanize_tu_choi_khi_khong_ap_dung_duoc(bad):
    """Không áp dụng được thì trả `None` ⇒ tầng gọi giữ ràng buộc "nhất quán", KHÔNG ấn định bừa."""
    result = romanize(bad)
    assert result is None or result.isalpha()


def test_romanize_tra_none_cho_chu_latinh():
    """Tên đã ở dạng Latinh thì không cần (và không được) suy lại."""
    assert romanize("Misaki") is None
    assert romanize("Tanaka-san") is None


def test_passport_khong_phai_hepburn():
    """Hồi quy: `hepburn` cho `Satou`/`Toukyou`, `passport` cho `Sato` — phải dùng passport."""
    assert romanize("佐藤") == "Sato"
    assert romanize("佐藤") != "Satou"


# ───────────────────────────────────── fallback thuần Python (không pykakasi)

def test_fallback_kana_khi_thieu_pykakasi(monkeypatch):
    """Thiếu `pykakasi` vẫn phải romaji hoá được katakana (chỉ chịu thua kanji)."""
    monkeypatch.setattr(romaji, "_KAKASI", None)
    monkeypatch.setattr(romaji, "_KAKASI_TRIED", True)  # chặn nạp lại kiểu lười
    assert romanize("ミサキ") == "Misaki"
    assert romanize("タナカ") == "Tanaka"
    assert romanize("ジョン") == "Jon"
    # Kanji là bó tay ⇒ trả None để rơi về ràng buộc "nhất quán", không ấn định bừa.
    assert romanize("美咲") is None


def test_pykakasi_duoc_nap_kieu_luoi():
    """Hồi quy hiệu năng: `import backend.translation` KHÔNG được nạp pykakasi.

    Đo thực: `import pykakasi` 98 ms + `kakasi()` 156 ms ⇒ ~254 ms cộng vào MỖI tiến trình
    (backend, mỗi tiến trình pytest) nếu làm ở cấp module. `backend.translation` mất 453 ms để
    import, trong đó 254 ms là pykakasi — không đáng trả khi phần lớn phiên không có tên nào.

    Kiểm bằng AST (không dùng `sys.modules`): test khác trong cùng tiến trình đã có thể gọi
    `romanize()` và nạp pykakasi rồi, nên `sys.modules` không còn là bằng chứng.
    """
    import ast
    from pathlib import Path

    src = Path(romaji.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:  # chỉ duyệt CẤP MODULE
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "pykakasi" for a in node.names), (
                "`import pykakasi` ở cấp module — phải để trong hàm (nạp kiểu lười)"
            )
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "pykakasi", (
                "`from pykakasi import …` ở cấp module — phải để trong hàm (nạp kiểu lười)"
            )
    # Và API công khai vẫn phải hoạt động.
    assert romaji.pykakasi_available() is True


def test_bang_kana_thuan_python():
    """Bảng kana phải xử lý đúng âm ghép, sokuon (っ) và trường âm (ー)."""
    f = romaji._kana_to_romaji
    assert f("きゃく") == "kyaku"
    assert f("しゃしん") == "shashin"
    assert f("ちゃ") == "cha"
    assert f("がっこう") == "gakkou"       # sokuon gấp đôi phụ âm
    assert f("コーヒー") == "koohii"        # trường âm lặp nguyên âm
    assert f("しんぶん") == "shinbun"


# ─────────────────────────────────────── phát hiện tên: kính ngữ + chức danh

def test_phat_hien_ten_kem_chuc_danh():
    """`佐藤部長` trước đây bị bỏ sót hoàn toàn vì 部長 không nằm trong danh sách kính ngữ."""
    g = TranslationGlossary.get_instance()
    assert "佐藤" in g.detect_names(["佐藤部長に報告しておきます。"])
    assert "田中" in g.detect_names(["田中課長に確認します。"])


def test_khong_de_xuat_chuc_danh_lam_ten_rieng():
    """`社長さん` KHÔNG được thành `社長 -> Shacho` — đó là chức danh, không phải tên."""
    g = TranslationGlossary.get_instance()
    assert "社長" not in g.detect_names(["社長さん、お疲れさまです。"])
    assert "社長" not in g.derived_renderings(["社長さん、お疲れさまです。"], _cfg())


# ─────────────── HỒI QUY THẬT (log 2026-10-08): katakana từ ngoại lai ≠ tên riêng

#: Katakana trần trong tiếng Nhật **chủ yếu là từ ngoại lai**. Bản đầu tiên coi mọi chạy
#: katakana là tên riêng rồi ấn định romaji, **thay mất bản dịch đúng**:
#:     チーム -> "Chiimu" (đúng: đội)            ホテル -> "Hoteru" (khách sạn)
#:     センス -> "Sensu" (gu)                    サッカー -> "Sakkaa" (bóng đá)
#:     バーベキュー -> "Baabekyuu" (tiệc nướng)  グリル -> "Guriru" (vỉ nướng)
#:     カスタム -> "Kasutamu" (tuỳ chỉnh)        イビキ -> "Ibiki" (tiếng ngáy)
LOANWORDS = [
    "チーム", "イビキ", "カスタム", "ホテル", "トラベルダウ", "センス",
    "サッカー", "バーベキュー", "グリル", "コーヒー", "ドア", "カメラ",
]


@pytest.mark.parametrize("word", LOANWORDS)
def test_katakana_tran_khong_bi_coi_la_ten(word):
    """Katakana KHÔNG có kính ngữ ⇒ không phát hiện, không suy, không ấn định."""
    g = TranslationGlossary.get_instance()
    text = f"{word}はどうですか。"
    assert word not in g.detect_names([text])
    assert word not in g.derived_renderings([text], _cfg())


def test_katakana_co_kinh_ngu_van_la_ten():
    """Có kính ngữ thì tín hiệu đủ mạnh — tên phiên âm vẫn phải nhận được."""
    g = TranslationGlossary.get_instance()
    assert g.detect_names(["ミサキちゃんが来た。"]) == ["ミサキ"]
    assert g.detect_names(["カミちゃんは。"]) == ["カミ"]
    assert g.derived_renderings(["ミサキちゃんが来た。"], _cfg()) == {"ミサキ": "Misaki"}


def test_so_dem_truoc_kinh_ngu_khong_phai_ten():
    """Hồi quy: `五年先輩` cho run `五年` ("5 năm") — KHÔNG được thành `五年 -> Gonen`."""
    g = TranslationGlossary.get_instance()
    assert "五年" not in g.detect_names(["五年先輩の花田さんとは。"])
    assert "花田" in g.detect_names(["五年先輩の花田さんとは。"])  # tên thật vẫn phải còn


def test_hoi_quy_log_that_2026_10_08():
    """Đúng các dòng log đã gây lỗi: `チーム`, `イビキ`, `カスタム`, `ホテル`, `センス`."""
    g = TranslationGlossary.get_instance()
    cfg = _cfg()

    # "入社してすぐ同じチームになり、"
    assert g.derived_renderings(
        ["入社してすぐ同じチームになり、", "二人で出張に来ることもしょっちゅうだった。"], cfg) == {}

    # "あいつが俺のイビキうるさかったらごめんって。"
    assert g.derived_renderings(
        ["何いじってんの？", "あいつが俺のイビキうるさかったらごめんって。"], cfg) == {}

    # "最高のカスタム。まるでカスタムね。"
    assert g.derived_renderings(["最高のカスタム。", "まるでカスタムね。"], cfg) == {}

    # Khách sạn — CHỈ 羽田 là tên, phần còn lại là từ ngoại lai.
    hotel = [
        "このホテル初めてですよね。",
        "あ、トラベルダウで星3.9でさ、",
        "羽田さんのホテル選びのセンス信用してますか。",
    ]
    derived = g.derived_renderings(hotel, cfg)
    assert derived == {"羽田": "Haneda"}, derived
    hint = g.build_hint(hotel, cfg)
    assert "ホテル" not in hint and "センス" not in hint and "トラベルダウ" not in hint
    assert "羽田 -> Haneda" in hint


def test_hoi_quy_khoi_barbecue():
    """`バーベキュー`/`グリル` là từ ngoại lai; khối này KHÔNG có tên riêng nào."""
    g = TranslationGlossary.get_instance()
    texts = ["今度の連休さ、", "いざきとまたバーベキューしに行かない？", "新しいグリル買っちゃったんだよね。"]
    assert g.derived_renderings(texts, _cfg()) == {}
    assert g.build_hint(texts, _cfg()) == ""


# ───────────────────────────────────────── suy tự động + thứ tự ưu tiên

def test_derived_renderings_glossary_rong():
    g = TranslationGlossary.get_instance()
    texts = ["美咲さんとはうまくいってるみたいですね。", "佐藤部長に報告しておきます。"]
    assert g.derived_renderings(texts, _cfg()) == {"美咲": "Misaki", "佐藤": "Sato"}


def test_khai_tay_de_len_suy_tu_dong():
    """Glossary thủ công LUÔN thắng romaji suy tự động."""
    g = TranslationGlossary.get_instance()
    texts = ["美咲さんとはうまくいってるみたいですね。", "佐藤部長に報告しておきます。"]
    hint = g.build_hint(texts, _cfg(glossary={"美咲": "Mỹ Sako"}))
    assert "美咲 -> Mỹ Sako" in hint
    assert "美咲 -> Misaki" not in hint
    assert "佐藤 -> Sato" in hint          # tên không khai tay vẫn được suy


def test_tat_suy_tu_dong_thi_quay_ve_rang_buoc_nhat_quan():
    g = TranslationGlossary.get_instance()
    texts = ["美咲さんとはうまくいってるみたいですね。", "もう絶対会社員にも戻れないって言ってるよ。"]
    hint = g.build_hint(texts, _cfg(glossary_derive_names=False))
    assert "Fixed renderings" not in hint
    assert "Proper nouns in the source (美咲)" in hint


def test_suy_tu_dong_khong_lap_lai_dong_nhat_quan():
    """Tên đã được ấn định (suy hoặc khai tay) thì không xuất hiện lại ở dòng "nhất quán"."""
    g = TranslationGlossary.get_instance()
    texts = ["美咲さんとはうまくいってるみたいですね。", "美咲さんは元気ですか。"]
    hint = g.build_hint(texts, _cfg())
    assert "美咲 -> Misaki" in hint
    assert "Proper nouns in the source" not in hint


def test_terms_for_single_co_ca_ten_suy_tu_dong():
    """Pipeline A dịch TỪNG CÂU nên càng cần ấn định tên, nếu không mỗi câu bịa một kiểu."""
    g = TranslationGlossary.get_instance()
    assert g.terms_for_single("美咲さんとはうまくいってるみたいですね。", _cfg()) == "美咲->Misaki"


# ─────────────────────────────────────────────────── tích hợp vào engine

def test_engine_bom_ten_suy_tu_dong_vao_prompt(allow_real_translation_methods, monkeypatch, tmp_path):
    """Glossary RỖNG mà prompt gửi cho model vẫn phải chứa `美咲 -> Misaki`."""
    from backend.translation.engine import GGUFTranslator
    from backend.translation.prompts import get_prompt_strategy

    class _Llama:
        closed = False

        def __init__(self):
            self.prompts = []

        def __call__(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return {"choices": [{"text": '{"1": "Misaki-san", "2": "Vâng."}'}],
                    "usage": {"completion_tokens": 12}}

        def close(self):
            pass

    llm = _Llama()
    t = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", llm)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", "m-test")
    monkeypatch.setattr(t.registry, "get_model", lambda k: {})
    t.canonical_key = "m-test"
    t.cfg = _cfg()
    t.prompt_strategy = get_prompt_strategy("pipeline_b")

    t.translate_batch_sync(["美咲さんとはうまくいってるみたいですね。", "はい。"], "ja", "vi")
    assert llm.prompts, "model chưa được gọi"
    assert "美咲 -> Misaki" in llm.prompts[0]
