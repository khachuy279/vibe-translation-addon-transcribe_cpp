"""Test tầng A — FIX-04 / FIX-12 / FIX-13: lock & thứ tự dọn phiên.

- FIX-04: `GGUFTranslator.load_model()` phải build model NGOÀI `_shared_lock`.
- FIX-12: `TranscribeEngine.cancel_inference()` phải snapshot session trong `_shared_lock`.
- FIX-13: dọn phiên phải VỨT hàng đợi SAU khi huỷ worker, và cân lại `_unfinished_tasks`.
"""

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from backend.ws.handler import _discard_queued

ROOT = Path(__file__).resolve().parent.parent.parent


class _FakeLlama:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


# ─────────────────────────────────────────────── FIX-04
def test_load_model_builds_outside_shared_lock(allow_real_translation_methods, monkeypatch):
    """Trong lúc `_build_llm()` chạy (chậm), `get_instance()` KHÔNG được chặn.

    Trước đây `_build_llm()` nằm trong `with _shared_lock:` nên mọi đường chỉ cần đọc
    trạng thái model (get_instance, unload, đọc `_shared_llm`) bị chặn hàng chục giây.
    """
    from backend.translation.engine import GGUFTranslator

    monkeypatch.setattr(GGUFTranslator, "_shared_llm", None)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", None)
    monkeypatch.setattr(GGUFTranslator, "_instance", None)

    build_started = threading.Event()
    build_release = threading.Event()

    def _slow_build(self, key, cfg, allow):
        build_started.set()
        build_release.wait(timeout=5.0)   # giữ "build" đang chạy
        return _FakeLlama("mới"), "prompt"

    monkeypatch.setattr(GGUFTranslator, "_build_llm", _slow_build)

    translator = GGUFTranslator.get_instance()
    errors = []

    def _run_load():
        try:
            translator.load_model()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=_run_load, daemon=True)
    t.start()
    assert build_started.wait(timeout=5.0), "load_model không gọi _build_llm"

    # Đang build: đọc trạng thái phải TRẢ VỀ NGAY.
    t0 = time.perf_counter()
    GGUFTranslator.get_instance()
    with GGUFTranslator._shared_lock:
        pass
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2, f"bị chặn {elapsed * 1000:.0f}ms trong lúc build model"

    build_release.set()
    t.join(timeout=5.0)
    assert not errors, errors
    assert isinstance(GGUFTranslator._shared_llm, _FakeLlama)
    assert GGUFTranslator._shared_model_key == translator.canonical_key


def test_load_model_idempotent_khi_da_co_model(allow_real_translation_methods, monkeypatch):
    """Đã có ĐÚNG model ⇒ không build lại (đường nóng phải rẻ)."""
    from backend.translation.engine import GGUFTranslator

    existing = _FakeLlama("đang chạy")
    translator = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", existing)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", translator.canonical_key)

    calls = []

    def _build(self, key, cfg, allow):  # pragma: no cover - không được gọi
        calls.append(key)
        return _FakeLlama("khác"), "p"

    monkeypatch.setattr(GGUFTranslator, "_build_llm", _build)
    translator.load_model()
    assert calls == [], "không được build lại khi model đã đúng"
    assert GGUFTranslator._shared_llm is existing


# ─────────────────────────────────────────────── FIX-12
def test_cancel_inference_snapshots_session_under_lock():
    """Chốt ở mức mã nguồn: `_shared_session` phải được đọc trong `_shared_lock`."""
    src = (ROOT / "backend" / "asr" / "engine.py").read_text(encoding="utf-8")
    idx = src.index("async def cancel_inference")
    body = src[idx:idx + 1200]
    assert "with self.__class__._shared_lock:" in body, (
        "cancel_inference phải snapshot `_shared_session` dưới `_shared_lock`"
    )
    assert "session = self.__class__._shared_session" in body
    # Không được giữ lock trong lời gọi native.
    assert "session.cancel()" in body


# ─────────────────────────────────────────────── FIX-13
def test_discard_queued_can_bang_unfinished_tasks():
    """Vứt hết item ⇒ `_unfinished_tasks` phải về 0, nếu không `queue.join()` treo."""
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    for i in range(3):
        q.put_nowait({"text": str(i)})

    session = SimpleNamespace(translation_queue=q, tts_queue=None)
    assert _discard_queued(session, "cleared_on_cleanup") == 3
    assert q.empty()

    async def _join():
        await asyncio.wait_for(q.join(), timeout=1.0)

    asyncio.run(_join())   # treo ⇒ test fail


def test_cleanup_discards_after_cancelling_workers():
    """FIX-13: thứ tự phải là huỷ worker TRƯỚC, rồi mới vứt hàng đợi (ngược lại là vô nghĩa)."""
    src = (ROOT / "backend" / "ws" / "handler.py").read_text(encoding="utf-8")
    cancel_idx = src.index("t.cancel()")
    discard_idx = src.index("_discard_queued(session, \"cleared_on_cleanup\")")
    assert cancel_idx < discard_idx, "phải huỷ worker trước khi dọn hàng đợi"
    # Cách cũ (drain sau khi đã huỷ worker) không được quay lại trong `finally`.
    assert "await session.drain_queues(timeout=0.15)" not in src


def test_reset_stream_van_don_hang_doi():
    """Đường tua video vẫn phải dọn hàng đợi (đã tách sang `_discard_queued`)."""
    src = (ROOT / "backend" / "ws" / "handler.py").read_text(encoding="utf-8")
    assert '_discard_queued(session, "cleared_on_seek")' in src
