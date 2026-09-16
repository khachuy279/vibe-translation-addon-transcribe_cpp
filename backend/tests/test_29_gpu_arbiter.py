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


def test_default_config_has_scheduler_disabled():
    """Chốt: bật scheduler là quyết định phải đo, không được bật ngầm."""
    from backend.config import GpuConfig

    assert GpuConfig().scheduler_enabled is False


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
