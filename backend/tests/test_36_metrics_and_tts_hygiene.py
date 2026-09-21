"""QWEN-Q10 / Q15 / Q14 / Q11 — nhóm P2 "vệ sinh" nhưng có hậu quả thật.

* **Q10**: `config.metrics.enabled=False` là **config NÓI DỐI** — cờ chỉ được đọc ở
  `dump_metrics_report`, còn mọi `record_*` chạy vô điều kiện ⇒ "tắt metrics cho nhanh"
  không có tác dụng.
* **Q15**: `_get_voice_clone_prompt` check-then-act KHÔNG lock ⇒ hai phiên cùng dùng một
  giọng mới tính prompt hai lần (hàng trăm ms) và `move_to_end`/`popitem` chạy song song.
* **Q14**: `dump_metrics_report` gọi ĐỒNG BỘ trên event loop lúc cleanup phiên.
* **Q11**: comment `min_silence_frame` ghi "20 frames" nhưng giá trị là 60, và QWEN-Q11 đã
  chứng minh đây là config CHẾT với đường FireRed.
"""

import ast
import threading
import time
from pathlib import Path

import pytest

from backend.config import config
from backend.core.metrics import MetricsCollector, metrics_collector

ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────── Q10

def test_metrics_enabled_mac_dinh_la_TRUE(monkeypatch):
    """Mặc định phải BẬT — nếu không, gate thật ở `record_*` sẽ tắt metrics ở mọi nơi.

    Bản gốc để `enabled=False` nhưng việc ghi vẫn luôn chạy (cờ chỉ chặn `dump_metrics_report`).
    Khi gate được làm cho thật, giữ mặc định `False` sẽ **âm thầm mất toàn bộ metrics** ở
    `main.py`, harness và test — đã tái hiện: 22 test fail. Xem `MetricsConfig`.
    """
    from backend.config import MetricsConfig

    assert MetricsConfig().enabled is True, (
        "mặc định `metrics.enabled` phải là True: việc ghi vốn luôn chạy, đổi sang gate thật "
        "mà giữ False là mất metrics toàn hệ thống"
    )


def test_metrics_enabled_false_thuc_su_tat_viec_do(monkeypatch):
    """Tắt metrics ⇒ KHÔNG ghi gì nữa (latency, counter, gauge, checkpoint)."""
    m = MetricsCollector()
    monkeypatch.setattr(config.metrics, "enabled", False)

    m.record_latency("stage", 12.0)
    m.record_metric("stage", "x", 5.0)
    m.increment_counter("c", 3)
    m.record_gauge("q", "depth", 7.0)
    m.record_checkpoint("cp")

    assert m.get_stage_stats("stage")["count"] == 0, "record_latency vẫn chạy khi metrics TẮT"
    assert m.get_counter("c") == 0, "increment_counter vẫn chạy khi metrics TẮT"
    assert m.get_gauge("q", "depth", -1.0) == -1.0, "record_gauge vẫn chạy khi metrics TẮT"
    assert m.get_checkpoint("cp") is None, "record_checkpoint vẫn chạy khi metrics TẮT"


def test_metrics_enabled_true_van_ghi_binh_thuong(monkeypatch):
    """Bật lại ⇒ ghi đủ (không được vô tình tắt vĩnh viễn)."""
    m = MetricsCollector()
    monkeypatch.setattr(config.metrics, "enabled", True)

    m.record_latency("stage", 12.0)
    m.increment_counter("c", 3)
    m.record_gauge("q", "depth", 7.0)

    assert m.get_stage_stats("stage")["count"] == 1
    assert m.get_counter("c") == 3
    assert m.get_gauge("q", "depth") == 7.0


def test_metrics_gate_khong_lay_lock_khi_tat(monkeypatch):
    """Khi TẮT, đường ghi phải thoát bằng 1 `if` — KHÔNG được chạm `_lock` (QWEN-Q10)."""
    m = MetricsCollector()
    monkeypatch.setattr(config.metrics, "enabled", False)

    class _BoomLock:
        def __enter__(self):
            raise AssertionError("đã lấy lock khi metrics TẮT — chi phí vẫn còn nguyên")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(m, "_lock", _BoomLock())
    m.record_latency("stage", 1.0)      # không được raise
    m.increment_counter("c")
    m.record_gauge("q", "d", 1.0)
    m.record_checkpoint("cp")


def test_dump_metrics_report_van_ton_trong_config_tat(monkeypatch, tmp_path):
    """Giữ hợp đồng cũ: `dump_metrics_report` trả None khi metrics tắt."""
    from backend.core.metrics import dump_metrics_report

    monkeypatch.setattr(config.metrics, "enabled", False)
    assert dump_metrics_report("test", str(tmp_path / "x.json")) is None


# ─────────────────────────────────────────────── Q15

class _FakeModel:
    """Model giả: đếm số lần tính prompt và cố tình chậm để lộ race."""

    def __init__(self, delay: float = 0.15):
        self.delay = delay
        self.calls = 0
        self._lock = threading.Lock()

    def create_voice_clone_prompt(self, ref_audio=None, ref_text=None):
        with self._lock:
            self.calls += 1
        time.sleep(self.delay)
        return {"prompt": f"{ref_audio}|{ref_text}"}


def _tts():
    from backend.tts.engine import OmniVoiceTTS

    return OmniVoiceTTS()


def test_voice_prompt_chi_tinh_MOT_lan_khi_nhieu_luong(monkeypatch):
    """Hai luồng cùng một giọng mới ⇒ chỉ MỘT lần tính prompt (QWEN-Q15)."""
    engine = _tts()
    engine.model = _FakeModel(delay=0.2)
    engine._voice_prompt_cache.clear()

    results = []

    def _worker():
        results.append(engine._get_voice_clone_prompt("voice.wav", "hello"))

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(results) == 4
    assert all(r is not None for r in results)
    assert engine.model.calls == 1, (
        f"tính prompt {engine.model.calls} lần cho CÙNG một giọng — check-then-act không lock "
        f"(QWEN-Q15). Mỗi lần tốn hàng trăm ms."
    )
    assert len(engine._voice_prompt_cache) == 1


def test_voice_prompt_khac_giong_van_tinh_rieng(monkeypatch):
    """Lock không được làm hỏng ngữ nghĩa cache theo key."""
    engine = _tts()
    engine.model = _FakeModel(delay=0.0)
    engine._voice_prompt_cache.clear()

    a = engine._get_voice_clone_prompt("a.wav", "x")
    b = engine._get_voice_clone_prompt("b.wav", "y")
    a2 = engine._get_voice_clone_prompt("a.wav", "x")

    assert a != b
    assert a2 == a, "lần hai phải lấy từ cache"
    assert engine.model.calls == 2
    assert engine._voice_prompt_cache_max == 8


# ─────────────────────────────────────────────── Q14

def test_dump_metrics_report_o_cleanup_di_qua_thread():
    """Chốt mã nguồn: cleanup phiên phải `asyncio.to_thread(dump_metrics_report, ...)`."""
    src = (ROOT / "backend" / "ws" / "handler.py").read_text(encoding="utf-8")
    assert "asyncio.to_thread(" in src and "dump_metrics_report" in src
    idx = src.index("dump_report_on_disconnect")
    body = src[idx:idx + 900]
    assert "asyncio.to_thread(" in body, (
        "`dump_metrics_report` vẫn gọi ĐỒNG BỘ trên event loop lúc cleanup (QWEN-Q14)"
    )
    assert "await asyncio.to_thread(" in body
    assert "dump_metrics_report(reason=" not in body, "còn call-site đồng bộ"


# ─────────────────────────────────────────────── Q11

def test_comment_min_silence_frame_khop_docs_va_khong_con_sai():
    """Q11 (cập nhật ở lượt viết lại VAD): giá trị phải ĐÚNG mặc định docs (20 frame) và
    comment phải nói rõ đây là knob CHỐT CÂU thật của kiến trúc mới.

    Bối cảnh: `min_silence_frame` từng bị ghi là "config CHẾT" vì processor bản cũ tự đếm
    im lặng bằng `silence_duration_ms`. Kiến trúc mới chỉ nghe event START/END của engine
    nên tham số này là knob thật ⇒ giá trị phải quay về đúng mặc định của
    `FireRedStreamVadConfig` (20 frame = 200 ms). Xem
    `report/audit/19_KE_HOACH_VIET_LAI_VAD.md`.
    """
    src = (ROOT / "backend" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    value = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FireRedVADConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and getattr(stmt.target, "id", "") == "min_silence_frame":
                    value = ast.literal_eval(stmt.value)
    assert value == 20, (
        f"`min_silence_frame` phải là 20 theo đúng mặc định docs "
        f"(20 frame × 10 ms = 200 ms), đang là {value}"
    )

    idx = src.index("min_silence_frame")
    comment_block = src[max(0, idx - 1400):idx]
    stale = "# 20 frames * 25 ms tối thiểu xác nhận kết thúc nói"
    assert stale not in comment_block, f"comment cũ vẫn còn nguyên: {stale!r}"
    assert "20 frame" in comment_block and "200 ms" in comment_block, (
        "comment phải nêu đúng 20 frame = 200 ms (mặc định docs)"
    )
    assert "10 ms" in comment_block, (
        "comment phải nêu đơn vị frame của upstream: 1 frame = FRAME_SHIFT_SAMPLE = 10 ms"
    )
    assert "chốt câu" in comment_block, (
        "phải ghi rõ đây là knob CHỐT CÂU thật của kiến trúc mới (không còn là config chết)"
    )
    assert "silence_duration_ms" in comment_block, (
        "comment phải chỉ người đọc sang `VADConfig.silence_duration_ms` (đường ghi đè)"
    )
    assert "CHẾT" not in comment_block, (
        "không được giữ lại khẳng định cũ 'config CHẾT' — kiến trúc mới chỉ nghe event "
        "START/END của engine nên `min_silence_frame` LÀ knob chốt câu"
    )
