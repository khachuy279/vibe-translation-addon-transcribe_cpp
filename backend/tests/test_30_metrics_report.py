"""Test tầng A — metrics report PHẢI thực sự được ghi ra file.

Vì sao có file test này: `config.metrics.dump_report_on_disconnect` / `report_file` /
`enabled` từng là **config CHẾT** — không dòng nào trong backend đọc chúng — và
`MetricsCollector.dump_json()` **không có caller nào**. Hệ quả thực tế: người dùng chạy
một phiên 5 phút rồi tắt server và **không tìm thấy `metrics_report.json` ở đâu cả**.

Các test dưới đây chốt 3 mảnh đó lại với nhau:
  1. `dump_metrics_report()` ghi ra file thật, đúng nội dung.
  2. Tên tương đối được neo vào `PROJECT_ROOT` (không phụ thuộc CWD lúc khởi chạy).
  3. Tắt `metrics.enabled` thì KHÔNG ghi (và không raise).
"""

import json

import pytest

from backend.config import PROJECT_ROOT, config
from backend.core.metrics import (
    MetricsCollector,
    dump_metrics_report,
    metrics_collector,
    resolve_report_path,
)


@pytest.fixture()
def report_tmp(tmp_path, monkeypatch, restore_config):
    """Trỏ report vào thư mục tạm và bật metrics."""
    target = tmp_path / "metrics_report.json"
    monkeypatch.setattr(config.metrics, "report_file", str(target), raising=False)
    monkeypatch.setattr(config.metrics, "enabled", True, raising=False)
    monkeypatch.setattr(config.metrics, "dump_report_on_disconnect", True, raising=False)
    return target


def test_dump_writes_real_file_with_expected_keys(report_tmp):
    """Ghi file thật + có đủ stages/counters/gauges."""
    metrics_collector.record_metric("asr", "commit_ms", 111.0)
    metrics_collector.record_metric("asr", "commit_ms", 222.0)
    metrics_collector.increment_counter("selftest.counter")
    metrics_collector.record_gauge("asr", "preview_recompute_ratio", 1.67)

    path = dump_metrics_report("test")
    assert path is not None, "phải trả về đường dẫn khi metrics bật"
    assert report_tmp.exists(), f"file report không được tạo: {report_tmp}"

    data = json.loads(report_tmp.read_text(encoding="utf-8"))
    assert "asr.commit_ms" in data["stages"]
    assert data["stages"]["asr.commit_ms"]["count"] >= 2
    assert data["counters"].get("selftest.counter", 0) >= 1
    assert data["gauges"].get("asr.preview_recompute_ratio") == pytest.approx(1.67)
    assert "uptime_sec" in data


def test_dump_returns_none_when_metrics_disabled(report_tmp, monkeypatch):
    """`metrics.enabled = False` ⇒ không ghi, trả None, KHÔNG raise."""
    monkeypatch.setattr(config.metrics, "enabled", False, raising=False)
    assert dump_metrics_report("test") is None
    assert not report_tmp.exists()


def test_relative_report_file_is_anchored_to_project_root(restore_config, monkeypatch):
    """Tên trần như `metrics_report.json` phải rơi vào PROJECT_ROOT, không phải CWD."""
    monkeypatch.setattr(config.metrics, "report_file", "metrics_report.json", raising=False)
    resolved = resolve_report_path()
    assert resolved == str(PROJECT_ROOT / "metrics_report.json")
    assert resolved.endswith("metrics_report.json")


def test_resolve_keeps_absolute_path(restore_config, monkeypatch, tmp_path):
    """Đường dẫn tuyệt đối phải được giữ nguyên (không ghép thêm PROJECT_ROOT)."""
    monkeypatch.setattr(config.metrics, "report_file", str(tmp_path), raising=False)
    # report_file trỏ tới thư mục: hàm chỉ ghép tên, không kiểm tra kiểu
    assert resolve_report_path("D:/x/y.json") == str(__import__("pathlib").Path("D:/x/y.json"))


def test_dump_never_raises_on_bad_path(restore_config, monkeypatch):
    """Đường dẫn không ghi được ⇒ trả None chứ không làm hỏng đường cleanup phiên."""
    monkeypatch.setattr(config.metrics, "enabled", True, raising=False)
    # Ký tự không hợp lệ trên Windows ⇒ `open()` sẽ lỗi
    monkeypatch.setattr(config.metrics, "report_file", "Z:/khong/ton/tai/x?.json", raising=False)
    assert dump_metrics_report("bad-path") is None


def test_disconnect_path_is_wired_in_handler():
    """Chốt ở mức mã nguồn: cleanup phiên PHẢI gọi dump khi config bật.

    Đây chính là mắt xích từng bị thiếu — nếu ai xoá nó, test này đổ.
    """
    src = (PROJECT_ROOT / "backend" / "ws" / "handler.py").read_text(encoding="utf-8")
    assert "dump_report_on_disconnect" in src, "cleanup phiên phải đọc config này"
    assert "dump_metrics_report(" in src, "cleanup phiên phải gọi dump_metrics_report()"


def test_shutdown_path_is_wired_in_main():
    """Chốt ở mức mã nguồn: shutdown PHẢI có lưới an toàn ghi report."""
    src = (PROJECT_ROOT / "backend" / "main.py").read_text(encoding="utf-8")
    assert 'dump_metrics_report("shutdown")' in src


def test_metrics_class_still_exposes_dump_json():
    """`dump_json` phải còn tồn tại — nếu không, `dump_metrics_report` sẽ vỡ."""
    assert hasattr(MetricsCollector, "dump_json")
    assert hasattr(MetricsCollector, "generate_report")
