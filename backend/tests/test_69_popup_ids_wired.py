"""test_69 — Mọi `id` trong popup.html PHẢI được tham chiếu (JS hoặc CSS).

BỐI CẢNH (lỗi đã sửa 2026-10-08)

`popup.html` từng có **6 id "mồ côi"**: có trong HTML nhưng không `popup.js` lẫn `popup.css`
nhắc tới. Hậu quả thật:

  * `stableToggleRow`, `stableTraceToggleRow`, `translationOnceToggleRow` — người dùng bấm vào
    HÀNG (theo thói quen, vì 2 hàng khác `showOriginalToggleRow`/`ttsToggleRow` bấm được) thì
    KHÔNG có gì xảy ra. Cùng một UI, hai hành vi khác nhau.
  * `duckingSliderRow` — slider "Original audio %" vẫn kéo được khi Auto-Ducking = Off,
    tức điều khiển không có tác dụng.
  * `lookaheadSyncGroup` — slider đồng bộ vẫn chỉnh được khi đang chạy Pipeline A, nhưng
    `lookaheadSyncOffsetMs` CHỈ được `ws/lookahead_handler.py` đọc (Pipeline B).

Cách xử lý đã áp: đấu dây 5 id (4 hàng bấm-để-bật/tắt + 1 nhóm ẩn/hiện), và **xoá** id
`lookaheadStatusDot` vì màu chấm do CSS quyết định qua `.lookahead-status.is-*` — không cần JS.

Bộ test này khoá lại: một id mới thêm vào `popup.html` mà không ai dùng sẽ bị bắt ngay, buộc
người viết chọn **đấu dây** hoặc **bỏ id**.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXT_DIRS = ("extension_firefox", "extension_chrome_edge")

#: Id được phép không xuất hiện trong JS/CSS. Hiện KHÔNG có ngoại lệ nào — thêm vào đây kèm
#: lý do nếu thật sự cần một id chỉ để neo (anchor) hay cho công cụ ngoài.
ALLOWED_ORPHAN_IDS: frozenset[str] = frozenset()


def _read(ext: str, rel: str) -> str:
    return (ROOT / ext / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("ext", EXT_DIRS)
def test_moi_id_trong_popup_html_deu_duoc_tham_chieu(ext: str) -> None:
    """Id không ai dùng ⇒ hoặc là điều khiển ma, hoặc là markup thừa. Cả hai đều phải sửa."""
    html = _read(ext, "popup/popup.html")
    js = _read(ext, "popup/popup.js")
    css = _read(ext, "popup/popup.css")

    ids = sorted(set(re.findall(r'id="([^"]+)"', html)))
    assert ids, f"{ext}: không tìm thấy id nào trong popup.html — test có vấn đề?"

    orphan = [
        i
        for i in ids
        if i not in ALLOWED_ORPHAN_IDS
        and not re.search(rf"\b{re.escape(i)}\b", js)
        and not re.search(rf"\b{re.escape(i)}\b", css)
    ]
    assert not orphan, (
        f"{ext}: {len(orphan)} id trong popup.html không được popup.js lẫn popup.css tham chiếu:\n  "
        + "\n  ".join(orphan)
        + "\n\nMỗi id phải HOẶC được đấu dây (JS), HOẶC được style (CSS), HOẶC bị xoá khỏi HTML.\n"
        "Đừng để lại id mồ côi: nó thường là dấu hiệu một điều khiển hiện ra mà không làm gì."
    )


@pytest.mark.parametrize("ext", EXT_DIRS)
def test_cac_hang_cong_tac_deu_bam_duoc(ext: str) -> None:
    """Mọi `.toggle-row` phải được đấu dây qua `wireToggleRow` — không hàng nào bấm mà không có gì."""
    html = _read(ext, "popup/popup.html")
    js = _read(ext, "popup/popup.js")

    assert "function wireToggleRow(" in js, f"{ext}: thiếu helper `wireToggleRow`"

    row_ids = re.findall(r'class="toggle-row"\s+id="([^"]+)"', html)
    assert row_ids, f"{ext}: không tìm thấy .toggle-row nào — kiểm tra lại cấu trúc HTML"

    wired = re.findall(r"wireToggleRow\(\s*([A-Za-z_$][\w$]*)", js) + re.findall(
        r'wireToggleRow\(\s*document\.getElementById\("([^"]+)"\)', js
    )
    # Tên biến JS ↔ id HTML: `stableToggleRow` giữ nguyên tên, nên so khớp trực tiếp được.
    missing = [r for r in row_ids if r not in wired and r not in js]
    assert not missing, (
        f"{ext}: các hàng công tắc sau chưa được đấu dây: {missing}\n"
        "Thêm `wireToggleRow(<row>, <checkbox>)` để bấm cả hàng cũng bật/tắt."
    )
