"""QWEN-Q5 + QWEN-Q4 — hai lỗi ở `VADProcessor` (đều P1).

**Q5 — `reset()` phá hợp đồng identity.** `feed_chunk` bỏ kết quả của batch đang bay bằng
so sánh **IDENTITY**: `if self._state is not state or self._stopped`. Bản cũ `reset()` sửa
field **in-place** (`self._state.reset()`) nên identity KHÔNG đổi ⇒ batch đang chạy model
khi `reset()` được gọi vẫn ghi kết quả vào state vừa reset ⇒ trộn trạng thái câu cũ/mới
(đếm im lặng sai, START/END lệch). Fix: gán `VADStreamState` MỚI (rỗng).

**Q4 — `force_end()` gọi callback TRONG `self._lock`.** `feed_chunk` đã được thiết kế
đúng (gom callback rồi gọi NGOÀI lock, comment "Kích hoạt callbacks bên ngoài lock để
tránh deadlock"), nhưng `force_end()` gọi `on_speech_end()` **khi đang giữ lock**. Chuỗi
ngược dòng `on_speech_end` → ASR (`asr._lock` + `call_soon_threadsafe`) tạo thứ tự lock
VAD→ASR từ trong lock; chỉ cần một callback gọi ngược vào VAD là deadlock. Ngoài ra giữ
lock suốt callback còn chặn `feed_chunk` của chính phiên đó (head-of-line).
"""

import ast
import threading
import time
from collections import deque
from pathlib import Path

import pytest

from backend.core.metrics import metrics_collector
from backend.tests.fakes import FakeVADEngine, make_silence_pcm, make_speech_pcm, pcm_to_int16_bytes
from backend.vad.processor import VADProcessor

ROOT = Path(__file__).resolve().parent.parent.parent


class _SlowVADEngine(FakeVADEngine):
    """Engine giả có độ trễ, để một batch còn 'đang bay' khi reset/force_end."""

    def __init__(self, delay: float = 0.2):
        super().__init__()
        self.delay = delay

    def is_speech(self, chunk_float32, state, threshold, chunk_raw=None):
        time.sleep(self.delay)
        return super().is_speech(chunk_float32, state, threshold, chunk_raw)


def _proc(monkeypatch, engine=None, **cb):
    """VADProcessor với engine giả đã sẵn sàng (không đụng pool toàn cục)."""
    proc = VADProcessor(vad_engine="firered-vad", **cb)
    proc._engine = engine or FakeVADEngine()
    proc._state = proc._new_state(proc._engine, proc.threshold)
    proc._stopped = False
    return proc


# ─────────────────────────────────────────────── Q5

def test_reset_tao_state_MOI_khong_sua_in_place(monkeypatch):
    """`reset()` phải đổi identity của `_state` (đó là hợp đồng của `feed_chunk`)."""
    proc = _proc(monkeypatch)
    old_state = proc._state

    proc.reset()

    assert proc._state is not old_state, (
        "`reset()` sửa state in-place ⇒ batch đang bay không bị nhận diện là cũ (QWEN-Q5)"
    )


def test_reset_lam_batch_dang_bay_bi_bo(monkeypatch):
    """Batch chạy qua `reset()` KHÔNG được ghi vào state mới (đây là hậu quả thật của Q5)."""
    proc = _proc(monkeypatch, engine=_SlowVADEngine(delay=0.25))
    speech = (b"\x10\x27" * 400) * 4     # biên độ ~0.3 ⇒ vượt ngưỡng RMS
    before = metrics_collector.get_counter("vad.batch_discarded_engine_switch")

    t = threading.Thread(target=lambda: proc.feed_chunk(speech), daemon=True)
    t.start()
    time.sleep(0.05)          # để batch kịp lấy `state` rồi chạy model
    proc.reset()              # "tua video" giữa lúc batch đang bay
    new_state = proc._state
    t.join(timeout=5.0)
    assert not t.is_alive()

    assert metrics_collector.get_counter("vad.batch_discarded_engine_switch") > before, (
        "batch qua `reset()` phải bị BỎ (identity đổi) — nếu không, trạng thái câu cũ "
        "được ghi vào state mới (QWEN-Q5)"
    )
    assert new_state.total_samples_processed == 0, (
        "state mới không được nhận kết quả của batch cũ"
    )


def test_state_sau_reset_van_co_pre_speech_ring_gioi_han(monkeypatch):
    """State rỗng phải được gắn `pre_speech_ring` có `maxlen`, nếu không sẽ rò RAM."""
    proc = _proc(monkeypatch)
    proc.reset()
    ring = proc._state.pre_speech_ring
    assert isinstance(ring, deque)
    assert ring.maxlen == proc._max_pre_frames(proc._frame_samples), (
        "`VADStreamState.pre_speech_ring` mặc định là deque KHÔNG giới hạn ⇒ phải gán lại maxlen"
    )


def test_reset_khong_goi_engine_create_initial_state(monkeypatch):
    """`reset()` chạy trên event loop ⇒ KHÔNG được nạp lại JIT model của Silero."""
    calls = []
    engine = FakeVADEngine()
    engine.create_initial_state = lambda threshold=None: (calls.append(1), FakeVADEngine().create_initial_state(threshold))[1]
    proc = _proc(monkeypatch, engine=engine)
    calls.clear()

    t0 = time.perf_counter()
    proc.reset()
    elapsed = time.perf_counter() - t0

    assert calls == [], "`reset()` không được gọi `create_initial_state` (Silero nạp JIT model)"
    assert elapsed < 0.05, f"`reset()` chậm {elapsed * 1000:.0f}ms — không được làm việc nặng"


def test_reset_van_mo_lai_duong_audio(monkeypatch):
    """Giữ nguyên hợp đồng cũ: `reset()` xoá cờ `_stopped` để nhận audio trở lại."""
    proc = _proc(monkeypatch)
    proc.force_end()
    assert proc._stopped is True
    proc.reset()
    assert proc._stopped is False
    # Và phân câu phải chạy lại bình thường sau reset.
    proc.feed_chunk(pcm_to_int16_bytes(make_speech_pcm(0.5)))
    assert proc._state.total_samples_processed > 0


# ─────────────────────────────────────────────── Q4

def test_force_end_goi_callback_NGOAI_lock(monkeypatch):
    """Trong lúc `on_speech_end` chạy, `self._lock` phải ĐÃ được nhả."""
    result = {}

    def _on_end():
        got = []

        def _try_lock():
            acquired = proc._lock.acquire(timeout=1.0)
            got.append(acquired)
            if acquired:
                proc._lock.release()

        th = threading.Thread(target=_try_lock, daemon=True)
        th.start()
        th.join(timeout=2.0)
        result["lock_free"] = bool(got and got[0])

    proc = _proc(monkeypatch, on_speech_end=_on_end)
    proc._state.is_speech = True     # giả lập đang trong câu

    proc.force_end()

    assert "lock_free" in result, "on_speech_end không được gọi"
    assert result["lock_free"] is True, (
        "`force_end()` gọi callback khi ĐANG giữ `self._lock` ⇒ deadlock class (QWEN-Q4)"
    )


def test_force_end_van_chot_cau_va_bao_loi_callback(monkeypatch):
    """Hợp đồng cũ giữ nguyên: chốt câu + callback lỗi không làm nổ `force_end`."""
    fired = []
    proc = _proc(
        monkeypatch,
        on_speech_end=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    proc._state.is_speech = True
    proc.on_speech_end = lambda: fired.append(1)
    proc.force_end()
    assert fired == [1]
    assert proc._state.is_speech is False
    assert proc._stopped is True

    # Callback ném lỗi ⇒ chỉ log, KHÔNG raise ra ngoài.
    proc2 = _proc(monkeypatch, on_speech_end=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    proc2._state.is_speech = True
    proc2.force_end()   # không được raise
    assert proc2._state.is_speech is False


def test_force_end_khong_giu_lock_qua_callback_o_muc_ma_nguon():
    """Chốt AST: `self.on_speech_end()` không được nằm trong khối `with self._lock`."""
    tree = ast.parse((ROOT / "backend" / "vad" / "processor.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("force_end", "reset"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.With) and any(
                    "self._lock" in ast.unparse(item.context_expr) for item in sub.items
                ):
                    inside = ast.unparse(sub)
                    assert "on_speech_end(" not in inside, (
                        f"{node.name}: còn gọi `on_speech_end()` TRONG `with self._lock` (QWEN-Q4)"
                    )
                    if node.name == "reset":
                        assert "_state.reset()" not in inside, (
                            "reset: còn sửa state in-place thay vì gán state MỚI (QWEN-Q5)"
                        )
