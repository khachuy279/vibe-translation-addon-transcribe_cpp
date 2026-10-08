"""test_72 — Ba bộ dedup PHẢI dùng CÙNG một quy tắc chuẩn hoá.

BỐI CẢNH (đã hợp nhất 2026-10-08)

Repo có 3 bộ lọc trùng ở 3 tầng khác nhau, và mỗi bộ tự chuẩn hoá một kiểu:

| Module | Quy tắc cũ | Hệ quả |
|---|---|---|
| `core/dedup.py`        | bỏ MỌI dấu câu + lower + gộp space | (chuẩn) |
| `translation/dedup.py` | chỉ `strip().lower()` | "Xin chào các bạn!" KHÁC "Xin chào các bạn" |
| `tts/dedup.py`         | bỏ dấu câu ở HAI ĐẦU | "Xin chào, các bạn" KHÁC "Xin chào các bạn" |

⇒ Cùng một câu, ba tầng cho ba kết luận khác nhau. Dấu câu từ ASR vốn không đáng tin
(aligner bỏ hết khi tokenize), nên quy tắc đúng là bỏ dấu câu khi so trùng — và phải là
MỘT quy tắc duy nhất.

**Chiến lược khớp thì CỐ Ý khác nhau** (chúng phục vụ mục đích khác nhau, không gộp):
  * `core`  — 3 tầng exact + substring + Jaccard, có cửa sổ thời gian (chống hallucination).
  * `translation` — exact + TTL, kèm cache bản dịch để tái dùng.
  * `tts`   — exact + tiền tố/hậu tố, lịch sử trượt theo SỐ câu.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.dedup import CommitDeduplicator, normalize_for_dedup
from backend.translation.dedup import TranslationDedupState
from backend.tts.dedup import TTSDedupState

ROOT = Path(__file__).resolve().parents[2]

#: Mẫu đa dạng: dấu câu cuối, dấu câu GIỮA, CJK, khoảng trắng thừa, gạch nối, rỗng.
SAMPLES = (
    "Xin chào các bạn!",
    "Xin chào, các bạn",
    "  Hello   World.  ",
    "日本語。",
    "[utt=1] a",
    "...",
    "",
    "A-B",
    "state-of-the-art",
    "It's fine.",
    "3.14 seconds",
)


@pytest.mark.parametrize("text", SAMPLES)
def test_ba_bo_chuan_hoa_dong_nhat(text: str) -> None:
    """Đây là bất biến CHÍNH: ba lối vào chuẩn hoá phải cho cùng một kết quả."""
    expected = normalize_for_dedup(text)
    assert TranslationDedupState._key(text) == expected, (
        f"translation/dedup chuẩn hoá LỆCH với core/dedup cho {text!r}"
    )
    assert TTSDedupState()._normalize(text) == expected, (
        f"tts/dedup chuẩn hoá LỆCH với core/dedup cho {text!r}"
    )


def test_khac_dau_cau_la_trung_o_CA_BA_tang() -> None:
    """Cùng câu, khác dấu câu ⇒ cả 3 tầng phải coi là trùng (trước đây chỉ 2/3)."""
    base = "Xin chào các bạn"
    variant = "Xin chào, các bạn!"

    core = CommitDeduplicator(window_sec=60.0)
    core.record_commit(base)
    assert core.is_duplicate(variant) is True, "core/dedup không nhận ra câu khác dấu câu"

    trans = TranslationDedupState()
    assert trans.is_duplicate(base) is False
    assert trans.is_duplicate(variant) is True, "translation/dedup không nhận ra câu khác dấu câu"

    tts = TTSDedupState()
    assert tts.is_duplicate(base) is False
    assert tts.is_duplicate(variant) is True, "tts/dedup không nhận ra câu khác dấu câu"


def test_cau_khac_that_su_van_duoc_chap_nhan() -> None:
    """Chuẩn hoá mạnh hơn KHÔNG được biến câu khác nghĩa thành trùng."""
    trans = TranslationDedupState()
    assert trans.is_duplicate("Xin chào các bạn") is False
    assert trans.is_duplicate("Chúc một ngày tốt lành") is False

    tts = TTSDedupState()
    assert tts.is_duplicate("Xin chào các bạn") is False
    assert tts.is_duplicate("Chúc một ngày tốt lành") is False


@pytest.mark.parametrize("rel", ["translation/dedup.py", "tts/dedup.py"])
def test_khong_tu_cai_lai_chuan_hoa(rel: str) -> None:
    """Chốt mã nguồn: hai module này phải UỶ QUYỀN, không tự băm chuỗi trở lại."""
    src = (ROOT / "backend" / rel).read_text(encoding="utf-8")
    assert "normalize_for_dedup" in src, (
        f"{rel} phải dùng `core.dedup.normalize_for_dedup` thay vì tự chuẩn hoá"
    )
    assert "re.sub" not in src, (
        f"{rel} còn tự regex chuẩn hoá ⇒ nguy cơ lệch quy tắc với `core/dedup` trở lại"
    )
