"""QWEN-E1/E3/E5/E6 — chạy các harness Node cho extension (P1).

Mỗi harness là một bài kiểm thử CHỨC NĂNG thật (chạy code extension trong scope giả), không
phải "có chuỗi này trong file". Tất cả đều đã được chứng minh là FAIL trên code trước khi
sửa:

* `overlay_attach_guard_test.js` (E1) — đếm THẬT số `new ResizeObserver` và
  `getBoundingClientRect` khi gọi `attachToVideo` 20 lần liên tiếp. Bản cũ: 20 observer mới
  + 40 lần đo layout (forced synchronous layout mỗi sự kiện phụ đề).
* `sw_broadcast_frame_target_test.js` (E3) — bản cũ `tabs.sendMessage` KHÔNG có `frameId`
  ⇒ mọi iframe trong tab đều nhận và parse mỗi sự kiện phụ đề.
* `resample_alias_test.js` (E5) — đo RMS đầu ra với sine trên Nyquist đích. Bản cũ để
  9 kHz và 12 kHz đi qua **100%** (gập phổ nguyên biên độ vào dải thoại); có boxcar còn
  58,8% (−4,6 dB) và 33,3% (−9,5 dB), còn 440 Hz giữ 99,9%.
* `ws_reconnect_on_port_death_test.js` (E6) — port chết SAU khi đã nối: bản cũ không hẹn
  nối lại ("phiên sống sót ảo").

Kèm theo: chạy luôn `overlay_manager_scale_test.js` (đã có từ trước nhưng **chưa từng được
pytest gọi**) để thay đổi ở E1 không làm hỏng logic auto-scale phụ đề.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
JS_DIR = ROOT / "backend" / "tests" / "js"

HARNESSES = [
    ("E1 — attachToVideo không forced-layout mỗi sự kiện", "overlay_attach_guard_test.js", 60),
    ("E3 — broadcast phụ đề chỉ tới top frame", "sw_broadcast_frame_target_test.js", 60),
    ("E5 — resample 48k→16k có lọc chống aliasing", "resample_alias_test.js", 60),
    ("E6 — port chết sau khi nối ⇒ tự nối lại", "ws_reconnect_on_port_death_test.js", 60),
    ("hồi quy — OverlayManager auto-scale vẫn đúng", "overlay_manager_scale_test.js", 60),
]


def _node() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("không có node để chạy harness JS của extension")
    return node


@pytest.mark.parametrize("label,script,timeout", HARNESSES, ids=[h[1] for h in HARNESSES])
def test_harness_js_extension(label, script, timeout):
    path = JS_DIR / script
    assert path.exists(), f"thiếu harness {script}"
    proc = subprocess.run(
        [_node(), str(path)],
        capture_output=True,
        text=True,
        # Node in tiếng Việt (UTF-8); mặc định trên Windows là cp1252 ⇒ UnicodeDecodeError
        # trong luồng đọc của subprocess (đã gặp thật).
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        # In cả stdout (báo cáo PASS/FAIL chi tiết) để đọc được ngay trong pytest.
        print(proc.stdout[-4000:])
        print(proc.stderr[-2000:], file=sys.stderr)
    assert proc.returncode == 0, f"{label}: harness {script} thất bại"


def test_e1_source_guard_host_in_place():
    """Chốt mã nguồn bổ sung: `attachToVideo` phải có guard identity + `_hostInPlace`."""
    src = (ROOT / "extension_firefox" / "content" / "overlay-manager.js").read_text(encoding="utf-8")
    assert "_hostInPlace()" in src, "thiếu guard `_hostInPlace` (E1)"
    assert "sameTarget" in src, "`attachToVideo` phải so target cũ (E1)"
    # Tìm ĐỊNH NGHĨA hàm (có dấu `{`), không phải lời gọi `this.attachToVideo(video);`
    # trong `init()`.
    idx = src.index("attachToVideo(video) {")
    body = src[idx:idx + 1600]
    assert "this._hostInPlace()" in body, "guard phải nằm TRONG `attachToVideo`"
    # `_setupResizeObserver` phải nhớ đối tượng đang theo dõi để không dựng lại vô ích.
    assert "_roVideo" in src and "_roHost" in src


def test_e5_source_guard_boxcar_bat_theo_ti_le_nguyen():
    """Chốt mã nguồn: boxcar chỉ bật cho tỉ lệ NGUYÊN ≥ 2 (48k→16k), không đụng 44,1k."""
    src = (ROOT / "extension_firefox" / "lib" / "audio-processor.js").read_text(encoding="utf-8")
    assert "Number.isInteger(this.ratio)" in src, "phải phát hiện tỉ lệ nguyên (E5)"
    assert "_boxcarHist" in src, "phải mang lịch sử mẫu qua biên block (tránh artifact 2,7 ms)"
    assert "BOXCAR_MIN_RATIO" in src


def test_e6_source_guard_reconnect_trong_on_disconnect():
    """Chốt mã nguồn: nhánh `else` của `port.onDisconnect` phải hẹn nối lại."""
    src = (ROOT / "extension_firefox" / "lib" / "ws-client.js").read_text(encoding="utf-8")
    idx = src.index("port.onDisconnect.addListener")
    body = src[idx:idx + 900]
    assert "_scheduleReconnect()" in body, (
        "`port.onDisconnect` không hẹn nối lại khi port chết SAU khi đã nối (E6)"
    )
