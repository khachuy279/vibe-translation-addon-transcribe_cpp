"""Test tầng A — FIX-09: gauge độ sâu hàng đợi không được ghi ở MỌI item.

`MetricsCollector` dùng mutex toàn cục cho mọi metric, mà `_record_queue_gauges()` được
gọi theo từng item ở vòng stream ASR và trong cả 2 worker. Ghi mỗi item ⇒ 2 lần tranh mutex
trong hot path realtime mà gần như không mang thêm thông tin.
"""

import asyncio

from backend.ws.handler import _record_queue_gauges


class _CountingMetrics:
    """Đếm số lần `record_gauge` được gọi (thay cho collector thật)."""

    def __init__(self):
        self.calls = []

    def record_gauge(self, stage, name, value):
        self.calls.append((stage, name, value))


def _session_with_queues(translation=None, tts=None):
    from types import SimpleNamespace

    q_t = asyncio.Queue(maxsize=64)
    q_s = asyncio.Queue(maxsize=64)
    if translation:
        for i in range(translation):
            q_t.put_nowait({"i": i})
    if tts:
        for i in range(tts):
            q_s.put_nowait({"i": i})
    return SimpleNamespace(translation_queue=q_t, tts_queue=q_s)


def _depths(counter, metric="translation_depth"):
    return [value for _stage, name, value in counter.calls if name == metric]


def test_queue_dung_yen_thi_chi_ghi_mot_lan(monkeypatch):
    """Queue không đổi qua 200 lần gọi ⇒ chỉ ghi 1 lần, không phải 200."""
    from backend.ws import handler as handler_mod

    counter = _CountingMetrics()
    monkeypatch.setattr(handler_mod, "metrics_collector", counter)

    session = _session_with_queues(translation=3, tts=0)

    for _ in range(200):
        _record_queue_gauges(session)

    # Lần đầu: last_depth = -1 nên "changed" ⇒ ghi. Các lần sau không đổi và
    # `time.perf_counter()` không nhảy 100ms trong vòng lặp này ⇒ không ghi thêm.
    assert _depths(counter) == [3], f"ghi quá nhiều lần: {counter.calls}"
    assert len(counter.calls) <= 2, f"tổng số lần ghi quá nhiều: {counter.calls}"


def test_queue_doi_thi_ghi_ngay(monkeypatch):
    """Độ sâu thay đổi ⇒ phải ghi ngay (tín hiệu backlog chính)."""
    from backend.ws import handler as handler_mod

    counter = _CountingMetrics()
    monkeypatch.setattr(handler_mod, "metrics_collector", counter)

    session = _session_with_queues(translation=0)
    _record_queue_gauges(session)                 # 0 -> ghi (changed)
    session.translation_queue.put_nowait({"x": 1})
    _record_queue_gauges(session)                 # 1 -> ghi (changed)
    session.translation_queue.get_nowait()
    _record_queue_gauges(session)                 # 0 -> ghi (changed)

    assert _depths(counter) == [0, 1, 0], counter.calls


def test_queue_ket_lau_thi_co_nhip_tim(monkeypatch):
    """Queue kẹt ở mức > 0 lâu ⇒ vẫn phải ghi định kỳ để gauge không bị 'cũ'."""
    from backend.ws import handler as handler_mod

    counter = _CountingMetrics()
    monkeypatch.setattr(handler_mod, "metrics_collector", counter)

    session = _session_with_queues(translation=2)
    _record_queue_gauges(session)   # lần đầu
    assert _depths(counter) == [2]

    # Ép đồng hồ nhảy qua ngưỡng nhịp tim.
    real_perf = handler_mod.time.perf_counter
    monkeypatch.setattr(handler_mod.time, "perf_counter", lambda: real_perf() + 10.0)

    _record_queue_gauges(session)
    assert _depths(counter) == [2, 2], "queue > 0 kẹt lâu phải có nhịp tim"


def test_khong_co_queue_thi_khong_lam_gi(monkeypatch):
    from types import SimpleNamespace

    from backend.ws import handler as handler_mod

    counter = _CountingMetrics()
    monkeypatch.setattr(handler_mod, "metrics_collector", counter)

    _record_queue_gauges(SimpleNamespace(translation_queue=None, tts_queue=None))
    assert counter.calls == []
