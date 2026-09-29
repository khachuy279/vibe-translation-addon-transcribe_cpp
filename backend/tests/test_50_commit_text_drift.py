"""Test tầng A — chẩn đoán "ASR viết lại cả câu" ([DRIFT]).

Bối cảnh (log thật 2026-09-29 21:29, phim tiếng Nhật): cùng một đoạn audio, Qwen3-ASR đổi cả
phần ĐẦU của câu khi đuôi audio dài thêm — `寄り行ってくる。` (nhịp +0,7 s, câu đúng) rồi bị các
nhịp sau ghi đè bằng `泳いてくるね。` (6 nhịp). Không tầng nào phát hiện ra vì commit chỉ phát văn
bản mới nhất.

`backend/core/hypothesis.py` đếm "nhịp nào đọc ra câu gì" và `TranscribeEngine` ghi WARNING
`[DRIFT]` khi câu phát đi CHƯA TỪNG xuất hiện trong lịch sử preview — thuần CHẨN ĐOÁN, KHÔNG đổi
văn bản phát đi.

Tầng A: không nạp model.
"""

import logging

from backend.core.hypothesis import HypothesisConsensus, normalize_hypothesis

#: 9 nhịp preview của `utt=acb8c6cf` (đúng thứ tự trong log thật).
PREVIEWS = [
    "焼いて。",
    "寄り行ってくる。",
    "泳いてくるね。",
    "寄って来るね。",
    "泳いてくるね。",
    "泳いてくるね。",
    "泳いてくるね。",
    "泳いてくるね。",
    "泳いてくるね。",
]


def _consensus(previews) -> HypothesisConsensus:
    tally = HypothesisConsensus()
    for text in previews:
        tally.observe(text)
    return tally


# ------------------------------------------------------------------ chuẩn hoá / tách câu
def test_chuan_hoa_bo_dau_cau_va_khoang_trang():
    assert normalize_hypothesis("荷物持つんでお母さん上がってください 。") == (
        "荷物持つんでお母さん上がってください"
    )
    assert normalize_hypothesis("Hello, world!") == "Helloworld"


def test_tach_cau_giu_ca_phan_duoi_dang_viet_do():
    """Nhịp preview có thể chứa câu đã xong + phần đuôi đang viết ⇒ phải đếm cả hai."""
    tally = _consensus(["今日はいい天気ですね。明日は雨"])
    assert tally.support("今日はいい天気ですね。") == 1
    assert tally.support("明日は雨") == 1


def test_tach_cau_latin_theo_dau_cham():
    tally = _consensus(["Hello world. Good bye."])
    assert tally.support("Hello world.") == 1
    assert tally.support("Good bye.") == 1


# ------------------------------------------------------------------ bộ đếm
def test_dem_so_nhip_ung_ho_moi_cau():
    tally = _consensus(PREVIEWS)
    assert tally.scans == len(PREVIEWS)
    assert tally.support("泳いてくるね。") == 6
    assert tally.support("寄り行ってくる。") == 1
    assert tally.tally(1) == [("泳いてくるね。", 6)]


def test_khop_mem_khi_khac_dau_cau():
    tally = _consensus(["荷物持つんでお母さん上がってください。"] * 3)
    assert tally.support("荷物持つんでお母さん上がってください") == 3
    assert tally.support("荷物持つんでお母さん上がってくださぃ。") == 3


def test_manh_dang_viet_do_khong_thoi_phong_cau_dai():
    """Mảnh ngắn lặp nhiều nhịp KHÔNG được cộng dồn cho câu dài (chốt an toàn độ dài)."""
    tally = _consensus(["荷物持つんで。"] * 5)
    assert tally.support("荷物持つんで。") == 5
    assert tally.support("荷物持つんでお母さん上がってください。") == 0


def test_reset_quen_lich_su():
    tally = _consensus(PREVIEWS)
    tally.reset()
    assert tally.scans == 0 and tally.support("泳いてくるね。") == 0 and tally.tally() == []


def test_tally_khong_phinh_vo_han():
    tally = HypothesisConsensus(max_entries=4)
    for i in range(50):
        tally.observe(f"câu số {i}。")
    assert len(tally.tally(99)) <= 4


# ------------------------------------------------------------------ nối vào engine
def _make_engine():
    from backend.asr.engine import TranscribeEngine
    from backend.config import config

    return TranscribeEngine(model_key=config.asr.active_model)


def test_canh_bao_drift_khi_cau_chot_chua_tung_xuat_hien(restore_config, caplog):
    eng = _make_engine()
    for text in PREVIEWS:
        eng._hypothesis_consensus.observe(text)

    with caplog.at_level(logging.WARNING):
        eng._warn_unseen_commit_text("まったく別の新しい文です。", "u9")

    assert "[DRIFT]" in caplog.text, caplog.text
    assert "泳いてくるね。" in caplog.text and "×6" in caplog.text, caplog.text


def test_khong_canh_bao_khi_cau_da_tung_xuat_hien(restore_config, caplog):
    eng = _make_engine()
    for text in PREVIEWS:
        eng._hypothesis_consensus.observe(text)

    with caplog.at_level(logging.WARNING):
        eng._warn_unseen_commit_text("泳いてくるね。", "u9")

    assert "[DRIFT]" not in caplog.text, caplog.text


def test_khong_canh_bao_khi_chua_du_bang_chung(restore_config, caplog):
    """Câu khác chỉ xuất hiện 1 nhịp ⇒ chưa đủ bằng chứng để coi là "trôi"."""
    eng = _make_engine()
    eng._hypothesis_consensus.observe("焼いて。")

    with caplog.at_level(logging.WARNING):
        eng._warn_unseen_commit_text("まったく別の文です。", "u9")

    assert "[DRIFT]" not in caplog.text, caplog.text


def test_drift_khong_doi_van_ban_phat_di(restore_config, caplog):
    """Chẩn đoán KHÔNG được sửa văn bản: nguồn đáng tin vẫn là bản chạy lại."""
    eng = _make_engine()
    for text in PREVIEWS:
        eng._hypothesis_consensus.observe(text)

    with caplog.at_level(logging.WARNING):
        out = eng._warn_unseen_commit_text("まったく別の新しい文です。", "u9")

    assert out is None
