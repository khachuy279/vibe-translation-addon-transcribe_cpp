"""Gộp cụm lặp do decoder "kẹt vòng" — ca thật từ phiên video 2026-09-18 00:04.

Log thật:
    [ASR_COMMIT] [MAX_DURATION] (infer=1494.4ms): 'Ha ha ha ... (≈250 lần)'
    qwen3_asr run: output truncated at 256 tokens — decode reached the generation budget
    [TRANSLATE]  (en -> vi in 2395ms): 'Ha ha ha ... (≈150 lần)'

Tức: 1,5 s GPU cho ASR + 2,4 s GPU cho dịch + một dòng phụ đề khổng lồ, tất cả cho một
tiếng cười nền. Binding transcribe.cpp chỉ expose `n_threads`/`kv_type`/`n_ctx` cho family
`qwen3_asr` (không có `max_tokens` hay repetition-penalty) nên phải chặn ở tầng văn bản.

Nguyên tắc quan trọng: **lặp ngắn là nội dung THẬT** — `ha ha ha` (cười), `no no no`,
`very very good` phải giữ nguyên. Chỉ gộp khi lặp đủ nhiều để chắc chắn là lỗi decoder.
"""

import pytest

from backend.asr.text_cleaner import clean_transcript_text
from backend.utils.text_repetition import collapse_ratio, collapse_repetitions


# ─────────────────────────────────────────────── ca thật

def test_tieng_cuoi_250_lan_gop_con_3():
    """Đúng ca trong log: 250 lần `ha` ⇒ `ha ha ha` (vẫn là nội dung thật)."""
    raw = " ".join(["ha"] * 250)
    out = collapse_repetitions(raw)
    assert out == "ha ha ha", out
    assert collapse_ratio(raw, out) > 0.98


def test_cum_lap_verbatim_gop_con_1():
    """Cụm lặp verbatim (decoder kẹt) ⇒ giữ 1 lần."""
    raw = "I don't know " * 6
    assert collapse_repetitions(raw.strip()) == "I don't know"


def test_khong_phan_biet_hoa_thuong_va_dau_cau():
    """`Ha!` / `ha,` / `HA` là cùng một token khi so khớp."""
    raw = "Ha! ha, HA. ha… " + " ".join(["ha"] * 200)
    out = collapse_repetitions(raw)
    assert out.split() == ["Ha!", "ha,", "HA."], out


# ─────────────────────────────────────────────── KHÔNG được cắt bừa

@pytest.mark.parametrize("text", [
    "ha ha ha",                      # cười thật — dưới ngưỡng
    "no no no",                      # nhấn mạnh thật
    "very very good",                # láy thật
    "no no no no",                   # 4 lần vẫn dưới ngưỡng 5
    "I love you",                    # câu bình thường
    "Xin chào, tôi tên là Nam.",     # tiếng Việt bình thường
    "",                              # rỗng
])
def test_cau_binh_thuong_giu_nguyen(text):
    assert collapse_repetitions(text) == text


def test_cjk_khong_dau_cach_tra_ve_nguyen_ven():
    """Tiếng Trung/Nhật không tách được thành token ⇒ KHÔNG được đụng vào."""
    text = "今天天气很好我们一起去公园散步吧"
    assert collapse_repetitions(text) == text


def test_khong_lam_mat_tu_khi_lap_ngan_nam_trong_cau_dai():
    text = "Well, no no no, that is not what I meant at all."
    assert collapse_repetitions(text) == text


# ─────────────────────────────────────────────── tích hợp vào ASR

def test_clean_transcript_text_gop_lap_va_van_bo_special_token():
    raw = "<|transcribe|><|en|> " + " ".join(["ha"] * 256) + " <|notimestamps|>"
    out = clean_transcript_text(raw)
    assert out == "ha ha ha", out
    assert "<|" not in out


def test_clean_transcript_text_giu_cau_binh_thuong():
    raw = "<|en|> What? Are you sleeping?"
    assert clean_transcript_text(raw) == "What? Are you sleeping?"


# ─────────────────────────────────────────────── tích hợp vào DỊCH

def test_dau_ra_dich_cung_bi_gop(allow_real_translation_methods, monkeypatch):
    """Model dịch cũng kẹt vòng — đầu ra của `_translate_sync` phải được gộp."""
    from types import SimpleNamespace

    from backend.translation.engine import GGUFTranslator

    class _LoopingLlama:
        closed = False

        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": " ".join(["Ha"] * 150)}],
                    "usage": {"completion_tokens": 150}}

        def close(self):
            pass

    t = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", _LoopingLlama())
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", "m-test")
    monkeypatch.setattr(t.registry, "get_model", lambda k: {})
    t.canonical_key = "m-test"
    t.cfg = SimpleNamespace(max_tokens=128, temperature=None, top_p=None, top_k=None,
                            repetition_penalty=None, use_context=False)
    t.prompt_strategy = SimpleNamespace(
        build_prompt=lambda **kw: "p", get_stop_tokens=lambda: ["</s>"]
    )

    out = t._translate_sync("ha ha ha", "en", "vi")
    assert out["translated_text"] == "Ha Ha Ha", out["translated_text"]


def test_streaming_dich_khong_phinh_thanh_buc_tuong_chu(allow_real_translation_methods, monkeypatch):
    """Bản streaming cũng phải gộp để phụ đề không phình."""
    from types import SimpleNamespace

    from backend.translation.engine import GGUFTranslator

    class _LoopingStreamLlama:
        closed = False

        def __call__(self, prompt, **kwargs):
            return iter([{"choices": [{"text": "ha "}]} for _ in range(200)])

        def close(self):
            pass

    t = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", _LoopingStreamLlama())
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", "m-test")
    monkeypatch.setattr(t.registry, "get_model", lambda k: {})
    t.canonical_key = "m-test"
    t.cfg = SimpleNamespace(max_tokens=128, temperature=None, top_p=None, top_k=None,
                            repetition_penalty=None, use_context=False)
    t.prompt_strategy = SimpleNamespace(
        build_prompt=lambda **kw: "p", get_stop_tokens=lambda: ["</s>"]
    )

    chunks = list(t._translate_stream_sync("ha ha ha", "en", "vi", ""))
    assert chunks, "phải có ít nhất một chunk"
    assert len(chunks[-1]) < 40, f"phụ đề vẫn phình: {chunks[-1]!r}"
    assert chunks[-1] == "ha ha ha"
