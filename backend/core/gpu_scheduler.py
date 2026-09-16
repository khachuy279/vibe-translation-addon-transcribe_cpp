"""A2-1 — GpuArbiter: điều phối tranh chấp GPU giữa ASR / dịch / TTS trên MỘT card.

## Vấn đề

Cả ba stage dùng chung một GPU và không có scheduler (audit A2-1). Khi một câu ASR
`commit` tới đúng lúc dịch/TTS đang chiếm GPU, commit phải xếp sau ⇒ phụ đề spike.

## Điều DUY NHẤT module này làm được

**Trì hoãn việc KHỞI ĐỘNG job ưu tiên thấp** trong khi ASR đang chạy, để nhường GPU cho
commit. Nó **không** preempt được một CUDA kernel đang chạy — driver không cho, và mọi
cách "hủy giữa chừng" đều phải bỏ dở lượt sinh token (không an toàn). Vì vậy đây là bài
toán **thời điểm**, không phải bài toán lock.

Hệ quả thiết kế:
  * KHÔNG có "GPU lock" toàn cục. Khóa toàn cục sẽ ép ASR + dịch + TTS chạy tuần tự và
    làm e2e latency TỆ HƠN (mất phần chồng lấn vốn có).
  * Job ưu tiên thấp đang chạy thì **được chạy cho xong**; chỉ job *mới* bị hoãn.
  * Mọi lần chờ đều có **trần thời gian** (`admit_max_wait_ms`) ⇒ không thể treo pipeline.

## Cơ chế

Đếm số commit đang chờ/chạy bằng một `threading.Lock` (vì commit ASR chạy trong
`ThreadPoolExecutor`, không phải trên event loop), cộng thêm mốc thời gian hạn dành riêng.
`admit()` là async và chỉ *poll* trạng thái đó — không cần `call_soon_threadsafe`, không
phụ thuộc event loop nào cụ thể, nên an toàn cho cả test đa loop.

## Ưu tiên

    P0 commit ASR      — deadline ngắn nhất, KHÔNG được trễ (mất chữ)
    P1 preview ASR     — best-effort, đã có backoff riêng
    P2 dịch            — trễ được trong giới hạn hàng đợi
    P3 TTS             — trễ được

Chỉ P2/P3 bị hoãn. P0/P1 đi thẳng.

## Mặc định TẮT

`config.gpu.scheduler_enabled = False`. Khi tắt, `admit()` trả về ngay (chỉ tốn một phép
so sánh bool) ⇒ hành vi y hệt trước khi có module này.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Iterator, Optional

from backend.core.metrics import metrics_collector
from backend.utils.logger import logger

_TAG = "CORE"

#: Lớp ưu tiên (số nhỏ = ưu tiên cao).
PRIORITY_COMMIT = 0
PRIORITY_PREVIEW = 1
PRIORITY_TRANSLATION = 2
PRIORITY_TTS = 3

#: Từ lớp này trở xuống mới bị hoãn khi ASR đang chạy.
DEFERRABLE_FROM = PRIORITY_TRANSLATION

_STAGE_NAMES = {
    PRIORITY_COMMIT: "asr_commit",
    PRIORITY_PREVIEW: "asr_preview",
    PRIORITY_TRANSLATION: "translation",
    PRIORITY_TTS: "tts",
}


def stage_name(priority: int) -> str:
    return _STAGE_NAMES.get(priority, f"p{priority}")


class GpuArbiter:
    """Trì hoãn job GPU ưu tiên thấp khi ASR đang bận. An toàn khi tắt."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._commit_active: int = 0
        self._reserved_until: float = 0.0
        self._last_commit_ms: float = 0.0
        # Thống kê (đọc không cần lock — chỉ để chẩn đoán/log)
        self._blocked_count: int = 0
        self._waited_ms_total: float = 0.0

    # ---------------------------------------------------------------- cấu hình hiệu dụng
    @staticmethod
    def _cfg():
        from backend.config import config

        return config.gpu

    def enabled(self) -> bool:
        try:
            return bool(self._cfg().scheduler_enabled)
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ phía ASR (sync)
    def commit_begin(self) -> None:
        """Báo có một commit ASR bắt đầu. Mở cửa sổ dành riêng GPU.

        Gọi từ luồng ASR (không phải event loop) nên chỉ dùng `threading.Lock`.
        Rẻ khi scheduler tắt: một phép đọc bool rồi thoát.
        """
        if not self.enabled():
            return
        reserve_s = max(0.0, float(getattr(self._cfg(), "commit_reserve_ms", 1500) or 0) / 1000.0)
        now = time.monotonic()
        with self._lock:
            self._commit_active += 1
            self._reserved_until = now + reserve_s
            self._last_commit_ms = reserve_s * 1000.0

    def commit_end(self) -> None:
        """Báo commit ASR đã xong."""
        if not self.enabled():
            return
        with self._lock:
            self._commit_active = max(0, self._commit_active - 1)
            if self._commit_active == 0:
                # Hết commit ⇒ đóng cửa sổ ngay, không bắt dịch/TTS chờ hết hạn vô ích.
                self._reserved_until = 0.0

    # ------------------------------------------------------------------ phía async
    def _defer_state(self) -> tuple[bool, float]:
        """Trả (có_nên_hoãn, số_giây_còn_lại)."""
        with self._lock:
            if self._commit_active <= 0:
                return False, 0.0
            remaining = self._reserved_until - time.monotonic()
        return (remaining > 0.0), max(0.0, remaining)

    async def admit(self, priority: int) -> float:
        """Chờ (nếu cần) trước khi KHỞI ĐỘNG một job GPU. Trả số ms đã chờ.

        Luôn quay về trong `admit_max_wait_ms` bất kể commit có xong hay không ⇒ không
        thể treo pipeline. Khi scheduler tắt hoặc priority không bị hoãn, trả về ngay.
        """
        if priority < DEFERRABLE_FROM or not self.enabled():
            return 0.0

        cfg = self._cfg()
        max_wait_s = max(0.0, float(getattr(cfg, "admit_max_wait_ms", 1200) or 0) / 1000.0)
        poll_s = max(0.005, float(getattr(cfg, "admit_poll_ms", 15) or 15) / 1000.0)

        should_defer, _ = self._defer_state()
        if not should_defer:
            return 0.0

        t0 = time.monotonic()
        deadline = t0 + max_wait_s
        name = stage_name(priority)
        metrics_collector.increment_counter("gpu.admit_blocked")
        with self._lock:
            self._blocked_count += 1

        while True:
            should_defer, remaining = self._defer_state()
            now = time.monotonic()
            if not should_defer or now >= deadline:
                break
            await asyncio.sleep(min(poll_s, remaining, max(0.0, deadline - now)))

        waited_ms = (time.monotonic() - t0) * 1000.0
        metrics_collector.record_metric("gpu", "admit_wait_ms", waited_ms)
        if waited_ms > 0:
            with self._lock:
                self._waited_ms_total += waited_ms
            if time.monotonic() >= deadline and should_defer:
                # Chạm trần chống starvation: chạy dù commit còn bận. Ghi lại để biết
                # cửa sổ dành riêng có đang quá ngắn hay không.
                metrics_collector.increment_counter("gpu.reserve_extended")
                logger.warning(
                    f"GpuArbiter: '{name}' chạm trần chờ {max_wait_s * 1000:.0f} ms khi ASR đang bận "
                    f"— cho chạy để tránh đói hàng đợi. Cân nhắc tăng gpu.commit_reserve_ms "
                    f"hoặc giảm tải.",
                    extra={"module_tag": _TAG},
                )
        return waited_ms

    @contextlib.asynccontextmanager
    async def slot(self, priority: int):
        """`async with arbiter.slot(PRIORITY_TTS): ...` — chờ nếu cần rồi chạy."""
        await self.admit(priority)
        yield

    # ------------------------------------------------------------------ chẩn đoán
    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled(),
                "commit_active": self._commit_active,
                "blocked_total": self._blocked_count,
                "waited_ms_total": round(self._waited_ms_total, 1),
            }

    def reset(self) -> None:
        """Xoá trạng thái (dùng trong test)."""
        with self._lock:
            self._commit_active = 0
            self._reserved_until = 0.0
            self._blocked_count = 0
            self._waited_ms_total = 0.0


#: Singleton dùng chung toàn tiến trình.
gpu_arbiter = GpuArbiter()


@contextlib.contextmanager
def commit_scope() -> Iterator[None]:
    """Bọc phần việc commit ASR để mở/đóng cửa sổ dành riêng GPU.

    Luôn đóng cửa sổ dù có exception — nếu không, dịch/TTS sẽ bị hoãn vô ích cho tới khi
    hết trần chờ.
    """
    gpu_arbiter.commit_begin()
    try:
        yield
    finally:
        gpu_arbiter.commit_end()


def arbiter_status() -> Optional[dict]:
    """Trạng thái cho `/health` (None nếu không đọc được)."""
    try:
        return gpu_arbiter.stats()
    except Exception:  # noqa: BLE001
        return None
