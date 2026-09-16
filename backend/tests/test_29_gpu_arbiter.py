"""Test tầng A — A2-1: GpuArbiter điều phối tranh chấp GPU.

Không cần GPU: chỉ kiểm tra logic trì hoãn/ nhường/ trần chống starvation, và quan trọng
nhất là **khi tắt thì không đổi hành vi gì**.
"""

import asyncio
import time

import pytest

from backend.config import config
from backend.core import gpu_scheduler as gs
from backend.core.gpu_scheduler import (
    PRIORITY_COMMIT,
    PRIORITY_PREVIEW,
    PRIORITY_TRANSLATION,
    PRIORITY_TTS,
    GpuArbiter,
)


@pytest.fixture()
def arb(restore_config):
    a = GpuArbiter()
    yield a
    a.reset()


# --------------------------------------------------------------- khi TẮT: no-op tuyệt đối
def test_disabled_is_a_noop(arb):
    """Mặc định TẮT ⇒ admit() không chờ, commit_begin/end không giữ trạng thái."""
    config.gpu.scheduler_enabled = False
    assert arb.enabled() is False

    t0 = time.monotonic()
    arb.commit_begin()
    waited = asyncio.run(arb.admit(PRIORITY_TTS))
    arb.commit_end()
    dt = (time.monotonic() - t0) * 1000.0

    assert waited == 0.0
    assert dt < 20.0, "tắt scheduler thì không được thêm trễ"
    assert arb.stats()["commit_active"] == 0
    assert arb.stats()["blocked_total"] == 0


def test_scheduler_state_is_observable_not_hardcoded():
    """`scheduler_enabled` là QUYẾT ĐỊNH VẬN HÀNH, không phải hằng số để test khoá lại.

    Bản trước của test này assert `GpuConfig().scheduler_enabled is False` để chốt rằng
    "bật scheduler phải là quyết định có đo". Cách chốt đó sai chỗ: nó khoá một **giá trị
    cấu hình** (đổi được bất cứ lúc nào, và người dùng đã đổi sang `True`), trong khi điều
    thực sự cần bảo vệ là **khả năng quan sát** — phải luôn biết một file report được đo với
    cờ nào, nếu không thì số liệu A/B vô nghĩa.

    Vì vậy test này chốt: (1) cờ tồn tại và là bool, (2) cờ được ghi vào report.
    """
    from backend.config import GpuConfig
    from backend.core.metrics import MetricsCollector

    assert isinstance(GpuConfig().scheduler_enabled, bool)

    snap = MetricsCollector._run_config_snapshot()
    assert "gpu.scheduler_enabled" in snap, "report phải ghi lại cờ này để đọc được A/B"
    assert isinstance(snap["gpu.scheduler_enabled"], bool)


# ------------------------------------------------------------------- khi BẬT: nhường GPU
def test_low_priority_waits_while_commit_active(arb):
    """TTS/dịch phải chờ khi commit đang chạy; commit xong thì chạy được ngay."""
    config.gpu.scheduler_enabled = True
    config.gpu.commit_reserve_ms = 5000
    config.gpu.admit_max_wait_ms = 5000
    config.gpu.admit_poll_ms = 5

    async def scenario():
        arb.commit_begin()

        async def release_later():
            await asyncio.sleep(0.2)
            arb.commit_end()

        task = asyncio.create_task(release_later())
        t0 = time.monotonic()
        waited = await arb.admit(PRIORITY_TTS)
        elapsed = (time.monotonic() - t0) * 1000.0
        await task
        return waited, elapsed

    waited, elapsed = asyncio.run(scenario())
    assert waited >= 150.0, f"phải chờ commit xong, chỉ chờ {waited:.0f} ms"
    assert elapsed >= 150.0
    assert arb.stats()["blocked_total"] == 1


def test_high_priority_never_waits(arb):
    """P0 commit và P1 preview đi thẳng, không bao giờ bị hoãn."""
    config.gpu.scheduler_enabled = True
    config.gpu.commit_reserve_ms = 5000
    config.gpu.admit_max_wait_ms = 5000

    async def scenario():
        arb.commit_begin()
        try:
            return await asyncio.wait_for(
                asyncio.gather(
                    arb.admit(PRIORITY_COMMIT),
                    arb.admit(PRIORITY_PREVIEW),
                ),
                timeout=0.5,
            )
        finally:
            arb.commit_end()

    got = asyncio.run(scenario())
    assert got == [0.0, 0.0], f"P0/P1 không được chờ, nhận {got}"


def test_waits_until_commit_ends_not_until_full_reserve(arb):
    """commit_end() phải ĐÓNG cửa sổ ngay, không bắt chờ hết commit_reserve_ms."""
    config.gpu.scheduler_enabled = True
    config.gpu.commit_reserve_ms = 60_000  # rất dài
    config.gpu.admit_max_wait_ms = 60_000
    config.gpu.admit_poll_ms = 5

    async def scenario():
        arb.commit_begin()

        async def release_soon():
            await asyncio.sleep(0.1)
            arb.commit_end()

        t = asyncio.create_task(release_soon())
        t0 = time.monotonic()
        await arb.admit(PRIORITY_TRANSLATION)
        dt = (time.monotonic() - t0) * 1000.0
        await t
        return dt

    dt = asyncio.run(scenario())
    assert dt < 1500.0, f"phải thoát ngay khi commit_end(), đã chờ {dt:.0f} ms"


def test_starvation_guard_bounds_wait(arb):
    """Commit không bao giờ kết thúc ⇒ job ưu tiên thấp vẫn phải được chạy sau trần chờ."""
    config.gpu.scheduler_enabled = True
    config.gpu.commit_reserve_ms = 60_000
    config.gpu.admit_max_wait_ms = 150
    config.gpu.admit_poll_ms = 5

    async def scenario():
        arb.commit_begin()  # cố tình KHÔNG gọi commit_end
        t0 = time.monotonic()
        waited = await arb.admit(PRIORITY_TTS)
        return waited, (time.monotonic() - t0) * 1000.0

    waited, elapsed = asyncio.run(scenario())
    assert 100.0 <= elapsed < 900.0, f"phải thoát quanh trần 150 ms, thực tế {elapsed:.0f} ms"
    assert arb.stats()["commit_active"] == 1  # vẫn còn commit treo


def test_nested_commits_keep_window_open(arb):
    """Hai commit chồng lấn: chỉ đóng cửa sổ khi commit CUỐI cùng xong."""
    config.gpu.scheduler_enabled = True
    config.gpu.commit_reserve_ms = 5000
    config.gpu.admit_max_wait_ms = 5000
    config.gpu.admit_poll_ms = 5

    async def scenario():
        arb.commit_begin()
        arb.commit_begin()
        arb.commit_end()  # còn 1 commit
        assert arb.stats()["commit_active"] == 1

        async def finish():
            await asyncio.sleep(0.15)
            arb.commit_end()

        t = asyncio.create_task(finish())
        t0 = time.monotonic()
        await arb.admit(PRIORITY_TTS)
        dt = (time.monotonic() - t0) * 1000.0
        await t
        return dt

    dt = asyncio.run(scenario())
    assert dt >= 100.0, "vẫn còn commit thì chưa được chạy"
    assert arb.stats()["commit_active"] == 0


def test_commit_scope_closes_window_on_exception(arb):
    """`commit_scope()` phải đóng cửa sổ dù thân có exception (nếu không, dịch/TTS bị hoãn vô ích)."""
    config.gpu.scheduler_enabled = True
    config.gpu.commit_reserve_ms = 60_000

    with pytest.raises(ValueError):
        with gs.commit_scope():
            raise ValueError("boom")

    assert arb.stats()["commit_active"] == 0 or gs.gpu_arbiter.stats()["commit_active"] == 0
    assert gs.gpu_arbiter.stats()["commit_active"] == 0


def test_slot_context_manager_runs_body(arb):
    config.gpu.scheduler_enabled = False
    ran = []

    async def body():
        async with arb.slot(PRIORITY_TRANSLATION):
            ran.append(True)

    asyncio.run(body())
    assert ran == [True]


# ------------------------------------------------- A3: đánh thức bằng Event (không poll)
def test_event_wakes_waiter_immediately_from_another_thread(arb):
    """A3: `commit_end()` gọi từ THREAD khác phải đánh thức waiter NGAY.

    Chứng minh bằng cách đặt `admit_poll_ms` rất lớn (500 ms) rồi đo: nếu vẫn dùng polling
    thì `admit()` phải chờ ~500 ms; nếu Event hoạt động thì trả về gần như tức thì sau khi
    `commit_end()` chạy ở thread khác.
    """
    config.gpu.scheduler_enabled = True
    config.gpu.commit_reserve_ms = 30_000
    config.gpu.admit_max_wait_ms = 30_000
    config.gpu.admit_poll_ms = 500          # poll rất thô ⇒ chỉ Event mới cứu được

    async def scenario():
        arb.commit_begin()
        loop = asyncio.get_running_loop()

        async def end_from_thread_later():
            await asyncio.sleep(0.15)
            # chạy `commit_end` trong THREAD POOL, không phải event loop
            await loop.run_in_executor(None, arb.commit_end)

        t = asyncio.create_task(end_from_thread_later())
        t0 = time.monotonic()
        waited = await arb.admit(PRIORITY_TTS)
        dt = (time.monotonic() - t0) * 1000.0
        await t
        return waited, dt

    waited, dt = asyncio.run(scenario())
    assert dt < 350.0, (
        f"Event không đánh thức được (chờ {dt:.0f} ms, trong khi poll=500 ms) "
        f"— có thể đang rơi về đường polling"
    )
    assert waited >= 100.0, "vẫn phải thực sự chờ commit xong"


def test_event_notify_is_safe_without_waiters(arb):
    """`commit_end()` không được raise khi chưa từng có waiter / loop đã đóng."""
    config.gpu.scheduler_enabled = True
    arb.commit_begin()
    arb.commit_end()          # chưa có waiter nào ⇒ `_wake` là None
    arb.commit_end()          # giảm quá số lần begin cũng không được âm
    assert arb.stats()["commit_active"] == 0


def test_waiter_on_new_loop_gets_fresh_event(arb):
    """Hai `asyncio.run()` liên tiếp (loop khác nhau) đều phải được đánh thức."""
    config.gpu.scheduler_enabled = True
    config.gpu.commit_reserve_ms = 30_000
    config.gpu.admit_max_wait_ms = 30_000
    config.gpu.admit_poll_ms = 500

    for _ in range(2):
        arb.reset()

        async def scenario():
            arb.commit_begin()
            loop = asyncio.get_running_loop()

            async def ender():
                await asyncio.sleep(0.1)
                await loop.run_in_executor(None, arb.commit_end)

            t = asyncio.create_task(ender())
            t0 = time.monotonic()
            await arb.admit(PRIORITY_TTS)
            dt = (time.monotonic() - t0) * 1000.0
            await t
            return dt

        dt = asyncio.run(scenario())
        assert dt < 350.0, f"loop mới không được đánh thức (chờ {dt:.0f} ms)"
