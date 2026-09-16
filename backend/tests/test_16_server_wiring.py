"""Test tầng A — smoke test wiring của server (không khởi động mạng, không nạp model).

Mục đích: bắt lỗi tích hợp ở `backend/main.py` và registry phiên trước khi chạy server thật.
Tất cả thay đổi của đợt triển khai này (endpoint mới, /health mở rộng, registry phiên,
cảnh báo CUDA, khối `streaming` trong /api/config) đều được kiểm tra ở đây.
"""

import asyncio
import inspect
import threading

import pytest


def test_backend_main_imports_and_app_builds():
    """`backend.main` phải import được và dựng FastAPI app (bắt lỗi import vòng/thiếu tên)."""
    import backend.main as main_mod

    assert main_mod.app is not None
    paths = {getattr(r, "path", None) for r in main_mod.app.routes}
    for expected in ("/health", "/api/config", "/api/metrics", "/api/metrics/pipeline", "/api/voices", "/ws"):
        assert expected in paths, f"thiếu route {expected}. Có: {sorted(p for p in paths if p)}"


def test_health_payload_has_runtime_info():
    """P1.1: /health phải phơi backend ASR thật để xác nhận bằng mắt."""
    import backend.main as main_mod

    payload = asyncio.run(main_mod.health_check())
    assert payload["status"] == "ok"
    assert "asr_runtime" in payload
    rt = payload["asr_runtime"]
    for key in ("model", "backend", "provider", "cuda_backend_available"):
        assert key in rt, f"asr_runtime thiếu '{key}'"
    assert "protocol_version" in payload
    assert "active_sessions" in payload


def test_health_exposes_loop_stall_metric(monkeypatch):
    """F-40: /health phải phơi `loop_stall_ms` — dấu hiệu duy nhất nhìn thấy loop bị chặn."""
    import backend.core.heartbeat as hb
    import backend.main as main_mod

    payload = asyncio.run(main_mod.health_check())
    assert "loop_stall_ms" in payload
    assert payload["loop_stall_ms"] >= 0.0

    # Giả lập event loop đã đứng 5 giây.
    monkeypatch.setattr(hb, "_last_tick", hb.time.monotonic() - 5.0, raising=False)
    payload = asyncio.run(main_mod.health_check())
    assert payload["loop_stall_ms"] >= 4500.0, "phải báo được loop đã đứng lâu"


def test_heartbeat_tick_resets_stall(monkeypatch):
    """`tick()` phải đưa `stall_sec()` về ~0."""
    import backend.core.heartbeat as hb

    monkeypatch.setattr(hb, "_last_tick", hb.time.monotonic() - 10.0, raising=False)
    assert hb.stall_sec() >= 9.0
    hb.tick()
    assert hb.stall_sec() < 0.5


def test_heartbeat_loop_ticks_and_is_cancellable():
    """Task nền phải tick rồi thoát sạch khi bị huỷ (không rò task khi shutdown)."""
    import backend.core.heartbeat as hb

    async def _run():
        hb._last_tick = hb.time.monotonic() - 30.0
        task = asyncio.create_task(hb.heartbeat_loop())
        await asyncio.sleep(0.4)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return hb.stall_sec()

    assert asyncio.run(_run()) < 1.0


# ------------------------------------------------- rào chắn RAM (bảo vệ máy, §19)
def test_mem_guard_threshold_logic():
    """`exceeds()` phải đúng, và trần <= 0 nghĩa là TẮT."""
    from backend.utils.mem_guard import exceeds, start_guard

    assert exceeds(5000.0, 4000.0) is True
    assert exceeds(1000.0, 4000.0) is False
    assert exceeds(5000.0, 0.0) is False, "trần 0 phải là TẮT"
    assert start_guard(limit_mb=0.0) is None


def test_mem_guard_default_limit_is_sane():
    """Mặc định phải chặn được kiểu phình hàng chục GB nhưng không chặn oan máy bình thường."""
    from backend.utils.mem_guard import DEFAULT_LIMIT_MB

    assert 1000.0 <= DEFAULT_LIMIT_MB <= 16000.0


def test_mem_guard_rss_reader_returns_value():
    """Đọc RSS phải hoạt động (nếu không rào chắn sẽ vô hiệu trong im lặng)."""
    from backend.utils.mem_guard import rss_mb

    value = rss_mb()
    assert value > 0.0, "không đọc được RSS — rào chắn RAM sẽ không hoạt động"


# ------------------------------------------------- watchdog treo (F-46)
def test_stall_watchdog_decision_and_toggle(monkeypatch):
    """Ngưỡng quyết định dump stack phải đúng và tắt được bằng 0."""
    from backend.utils import stall_watchdog as sw

    assert sw.should_report(11.0, 10.0) is True
    assert sw.should_report(3.0, 10.0) is False
    assert sw.should_report(99.0, 0.0) is False, "ngưỡng 0 = TẮT"

    monkeypatch.setenv("STALL_WATCHDOG_SEC", "0")
    assert sw.start() is None, "=0 thì không được bật luồng nào"
    sw.stop()


def test_stall_watchdog_thread_runs_and_stops():
    """Luồng watchdog phải chạy được và dừng sạch (không rò luồng khi shutdown)."""
    import time

    from backend.utils import stall_watchdog as sw

    t = sw.start(threshold=60.0)
    assert t is not None and t.is_alive()
    sw.stop()
    time.sleep(0.7)
    assert not t.is_alive(), "watchdog phải dừng khi được yêu cầu"


def test_stall_watchdog_dumps_threads_without_raising(capsys):
    """`dump_all_threads` phải ghi ra stderr và không ném lỗi (dùng được trong lúc treo)."""
    from backend.utils.stall_watchdog import dump_all_threads

    dump_all_threads("test")
    captured = capsys.readouterr()
    assert "STALL WATCHDOG" in captured.err


def test_asr_backend_log_helper_never_raises():
    """Log backend ASR lúc khởi động phải an toàn kể cả khi không có backend CUDA.

    `provider` có thể là `None` khi chạy bằng bundle cục bộ trong `bin/`
    (`TRANSCRIBE_LIBRARY`) — binding đi đường "dev-tree" nên không có tên provider PyPI.
    `_asr_runtime_info()` quy đổi trường hợp đó thành "local-bin" và luôn kèm
    `native_source`/`native_bundle_dir` để biết thư viện native đến từ đâu.
    """
    import backend.main as main_mod

    main_mod._log_asr_backend_at_startup()
    info = main_mod._asr_runtime_info()
    assert isinstance(info, dict)
    assert info.get("provider"), "phải xác định được nguồn thư viện native"
    assert info.get("native_source") in ("config", "default", "env", "installed", "unset")
    assert isinstance(info.get("available_backends"), list)


def test_config_response_exposes_streaming_and_protocol():
    """P2.x/P3.0: khối `streaming` và `protocol_version` phải có trong /api/config."""
    import backend.main as main_mod

    resp = main_mod._build_config_response(include_catalog=True)
    assert resp["protocol_version"] >= 2
    stream = resp["streaming"]
    for key in (
        "preview_window_sec",
        "poll_interval_ms",
        "max_duration_sec",
        "stability_duration_sec",
        "enable_tier234",
        "reuse_preview_for_commit",
        "stream_translation",
        "native_backend",
    ):
        assert key in stream, f"khối streaming thiếu '{key}'"
    assert "available_vad_engines" in resp
    assert "available_asr_engines" in resp


def test_session_registry_register_unregister():
    """P1.10: registry phiên đang hoạt động phải đăng ký/huỷ/truy vấn đúng."""
    from backend.ws.handler import (
        count_active_sessions,
        get_active_sessions,
        register_session,
        unregister_session,
    )

    class _FakeSession:
        def __init__(self, sid):
            self.session_id = sid

    before = count_active_sessions()
    a, b = _FakeSession("s-a"), _FakeSession("s-b")
    register_session(a)
    register_session(b)
    assert count_active_sessions() == before + 2
    ids = {s.session_id for s in get_active_sessions()}
    assert {"s-a", "s-b"} <= ids

    unregister_session(a)
    assert count_active_sessions() == before + 1
    unregister_session(b)
    assert count_active_sessions() == before

    # Huỷ 2 lần không được lỗi
    unregister_session(b)
    assert count_active_sessions() == before


def test_session_registry_is_thread_safe():
    from backend.ws.handler import count_active_sessions, register_session, unregister_session

    class _S:
        def __init__(self, sid):
            self.session_id = sid

    before = count_active_sessions()

    def _worker(n):
        for i in range(200):
            s = _S(f"t{n}-{i}")
            register_session(s)
            unregister_session(s)

    threads = [threading.Thread(target=_worker, args=(n,), daemon=True) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert count_active_sessions() == before


def test_track_background_task_discards_and_logs():
    """`track_background_task` phải dọn khỏi set khi xong (không rò tham chiếu)."""
    import backend.main as main_mod

    async def _run():
        async def _ok():
            return 1

        async def _boom():
            raise RuntimeError("boom")

        t1 = main_mod.track_background_task(_ok(), name="ok")
        t2 = main_mod.track_background_task(_boom(), name="boom")
        await asyncio.gather(t1, t2, return_exceptions=True)
        await asyncio.sleep(0)
        return set(main_mod._background_tasks)

    remaining = asyncio.run(_run())
    assert all(t.done() for t in remaining), "task đã xong phải được gỡ khỏi _background_tasks"


def test_lifespan_prewarm_helpers_exist_and_are_callable():
    """Các hàm prewarm trong lifespan phải tồn tại (bắt lỗi refactor làm mất tên)."""
    import backend.main as main_mod

    src = inspect.getsource(main_mod.lifespan)
    for name in ("_prewarm_asr", "_prewarm_translation", "_prewarm_vad_default", "_prewarm_vad_others"):
        assert name in src, f"lifespan thiếu bước prewarm '{name}'"
    # P1.9: engine VAD khác phải được nạp Ở NỀN, không chặn khởi động
    assert "track_background_task" in src
    assert "prewarm_engines" in inspect.getsource(main_mod._prewarm_vad_others)


def test_vad_factory_prewarm_reports_status_without_raising():
    """`prewarm_engines` phải trả dict trạng thái thay vì ném lỗi (dùng cho log khởi động)."""
    from backend.vad.engines import VADEngineFactory

    # Không nạp gì thật: chỉ gọi với tên không hợp lệ
    res = VADEngineFactory.prewarm_engines(["khong-ton-tai"])
    assert res == {}
    assert VADEngineFactory.is_cached("khong-ton-tai") is False
    assert VADEngineFactory.peek_engine("khong-ton-tai") is None


def test_metrics_endpoints_present_in_openapi():
    """/api/metrics/pipeline phải xuất hiện trong OpenAPI (endpoint mới, P0.1)."""
    import backend.main as main_mod

    schema = main_mod.app.openapi()
    assert "/api/metrics/pipeline" in schema["paths"]
    assert "/health" in schema["paths"]
