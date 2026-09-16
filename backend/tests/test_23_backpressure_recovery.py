"""Test tầng A — FIX-01: backpressure HARD PHẢI có đường tự phục hồi.

Bug gốc (đã xác nhận, xem `report/audit/17_XAC_NHAN_AUDIT_VA_KE_HOACH_SUA.md`):
`service-worker.js` đặt cờ `pausedByBackpressure = true` khi `bufferedAmount` vượt HARD,
nhưng cờ đó CHỈ được xoá bên trong `sendOrQueue()` — mà hàm này chỉ chạy khi có
`SEND_BINARY`. Khi content script nhận "paused" thì nó tắt `audioCapture.isCapturing`,
worklet ngừng phát chunk ⇒ không còn `SEND_BINARY` ⇒ **capture treo vĩnh viễn**.
`FLUSH_PENDING` có implement nhưng KHÔNG có producer nào trong toàn bộ extension.

Test này là rào chắn ở mức mã nguồn (giống `test_20_logging_convention.py`): nó không thay
thế test hành vi (`extension_firefox/tests/backpressure-gate.test.js`, chạy bằng node), mà
chốt rằng ĐƯỜNG PHỤC HỒI vẫn tồn tại và vẫn được nối vào cả hai đầu.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
EXT = ROOT / "extension_firefox"

GATE_JS = EXT / "lib" / "backpressure-gate.js"
SW_JS = EXT / "background" / "service-worker.js"
CS_JS = EXT / "content" / "content-script.js"
WSCLIENT_JS = EXT / "lib" / "ws-client.js"
MANIFEST_JSON = EXT / "manifest.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _code_only(src: str) -> str:
    """Bỏ dòng comment `//` để không bắt nhầm tên biến được nhắc trong chú thích."""
    return "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("//"))


def test_gate_module_exists():
    """Máy trạng thái thuần phải tồn tại và export được cho cả browser lẫn node."""
    src = _read(GATE_JS)
    assert "createGate" in src
    assert "module.exports" in src, "phải export được cho node --test"
    assert "BackpressureGate" in src, "phải gắn vào global cho service worker (classic script)"


def test_manifest_loads_gate_before_service_worker():
    """`background.scripts` là classic script dùng chung scope ⇒ thứ tự nạp quan trọng."""
    import json

    manifest = json.loads(_read(MANIFEST_JSON))
    scripts = manifest["background"]["scripts"]
    assert "lib/backpressure-gate.js" in scripts, "manifest phải nạp gate"
    assert scripts.index("lib/backpressure-gate.js") < scripts.index("background/service-worker.js"), (
        "gate phải được nạp TRƯỚC service-worker để `BackpressureGate` có mặt"
    )


def test_service_worker_has_auto_resume_path():
    """Service worker phải có timer tự kiểm tra lại, không phụ thuộc frame mới."""
    src = _read(SW_JS)
    assert "attemptResume" in src, "thiếu hàm phục hồi"
    assert "setInterval" in src, "phải có timer tự kiểm tra lại khi đang tạm dừng"
    assert "stopResumeTimer" in src, "timer phải huỷ được (tránh timer zombie)"
    # Timer phải được huỷ ở đường dọn dẹp.
    assert "stopResumeTimer();" in src
    # Cờ cũ (chỉ được xoá khi có frame mới) không được quay lại dưới dạng mã chạy.
    assert "pausedByBackpressure" not in _code_only(src), (
        "cờ `pausedByBackpressure` cũ chính là nguyên nhân treo — không được quay lại"
    )


def test_gate_has_hysteresis():
    """Vào pause ở HARD, chỉ nhả khi xuống DƯỚI SOFT — tránh rung pause/resume."""
    src = _read(GATE_JS)
    assert "buffered < soft" in src, "điều kiện nhả phải là '< soft' (hysteresis)"
    assert "buffered >= hard" in src, "điều kiện vào pause phải là '>= hard'"


def test_flush_pending_has_a_producer():
    """`FLUSH_PENDING` trước đây không có ai gọi ⇒ phải có producer ở content script."""
    assert "FLUSH_PENDING" in _read(SW_JS), "service worker phải xử lý FLUSH_PENDING"
    assert "flushPending" in _read(WSCLIENT_JS), "WSClient phải có hàm gửi probe"
    cs = _read(CS_JS)
    assert "flushPending()" in cs, "content script phải gọi probe khi đang tạm dừng"
    assert "startBackpressureProbe" in cs, "probe phải có timer riêng"


def test_probe_is_cleaned_up_on_stop():
    """Probe không được sống sót qua Stop/cleanup."""
    cs = _read(CS_JS)
    cleanup_idx = cs.index("async function cleanup()")
    cleanup_body = cs[cleanup_idx:cleanup_idx + 600]
    assert "stopBackpressureProbe()" in cleanup_body, "cleanup() phải dừng probe"


@pytest.mark.skipif(not GATE_JS.exists(), reason="chưa có gate module")
def test_gate_limits_match_service_worker_constants():
    """Ngưỡng trong gate mặc định và hằng số service worker phải khớp nhau."""
    gate = _read(GATE_JS)
    sw = _read(SW_JS)
    assert "128 * 1024" in gate and "128 * 1024" in sw
    assert "512 * 1024" in gate and "512 * 1024" in sw
