"""QWEN-Q2 (P0) — nạp VAD engine không được chặn EVENT LOOP và không giữ lock cấp lớp.

**Lỗi gốc (3 tầng cộng dồn):**
1. `VADProcessor.__init__` gọi `_ensure_engine()` → `VADEngineFactory.get_engine()`.
   Mà `__init__` chạy **TRÊN EVENT LOOP** (`ws/handler.py` → `SessionState.
   init_components()` → `ws/session.py`). Vì asyncio không preempt được code đồng bộ,
   một lượt nạp model ở đây làm **toàn bộ backend đứng hình** (mọi phiên WS, `/health`,
   REST, heartbeat).
2. `get_engine()` giữ `cls._lock` (CẤP LỚP, dùng chung toàn tiến trình) **suốt
   constructor**, mà constructor FireRed/FSMN gọi `hf_hub_download`/`snapshot_download`
   khi model chưa cache — **hàng phút** nếu mạng chậm.
3. `is_cached()` / `peek_engine()` lấy **cùng** `cls._lock` đó ⇒ ngay cả đường chỉ ĐỌC
   trạng thái (được event loop gọi ở `VADProcessor.__init__` và `apply_engine_change`)
   cũng bị chặn theo.

**Cách sửa:**
- `BaseVADEngine.prepare_files()` (classmethod) gom việc TẢI file; `get_engine` gọi nó
  **NGOÀI** `_lock`, dưới một `_prepare_lock` riêng chỉ chặn các luồng đang tải.
- `get_engine` dùng double-checked locking.
- `VADProcessor.__init__` chỉ làm việc RẺ: `peek_engine` (đọc biến) nếu engine đã có
  trong pool, còn không thì hẹn nạp ở thread nền.
- `_ensure_engine()` **không bao giờ** tải/nạp: hết engine thì trả `False`, `feed_chunk`
  BỎ chunk thay vì treo đường hot.
- Đường prewarm lúc khởi động dùng `VADProcessor.prewarm()` (đồng bộ, gọi từ thread nền).
"""

import ast
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from backend.core.metrics import metrics_collector
from backend.tests.fakes import (
    FakeVADEngine,
    make_speech_pcm,
    pcm_to_int16_bytes,
)
from backend.vad.engines import VADEngineFactory
from backend.vad.engines.firered import FireRedVADEngine
from backend.vad.processor import VADProcessor

ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────── tiện ích AST: kiểm CODE, không kiểm docstring/comment ───────────────

def _func_node(rel_path: str, name: str) -> ast.AST:
    tree = ast.parse((ROOT / rel_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"không tìm thấy hàm {name} trong {rel_path}")


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _dotted_calls(node: ast.AST):
    """Mọi lời gọi trong `node`, dạng chuỗi `a.b.c(...)` (đã bỏ docstring/comment)."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            out.append(f"{_dotted(n.func)}()")
    return out


def _calls_inside_with(node: ast.AST, ctx_dotted: str):
    """Các call nằm trong `with <ctx_dotted>:` (tìm trong toàn bộ cây con)."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.With):
            if any(_dotted(item.context_expr) == ctx_dotted for item in n.items):
                for sub in ast.walk(n):
                    if isinstance(sub, ast.Call):
                        out.append(f"{_dotted(sub.func)}()")
    return out


def _call_sites_inside_prepare_lock(node):
    return [c for c in _calls_inside_with(node, "cls._prepare_lock") if c.endswith("prepare_files()")]


def _call_sites_inside_pool_lock(node):
    return [c for c in _calls_inside_with(node, "cls._lock") if c.endswith("prepare_files()")]


def _fresh_pool(monkeypatch):
    monkeypatch.setattr(VADEngineFactory, "_engines", {})


# ─────────────────────────────────────────────── 1. __init__ không nạp engine

def test_khoi_tao_vad_processor_khong_nap_engine_tren_loop(monkeypatch):
    """`VADProcessor(...)` phải trả về NGAY, việc nạp engine do thread nền lo."""
    _fresh_pool(monkeypatch)
    monkeypatch.setattr(VADEngineFactory, "peek_engine", classmethod(lambda cls, n: None))

    started = threading.Event()

    def _slow_get_engine(name):          # giả lập tải model mất 0.5 s
        started.set()
        time.sleep(0.5)
        return FakeVADEngine()

    monkeypatch.setattr(VADEngineFactory, "get_engine", _slow_get_engine)

    t0 = time.perf_counter()
    proc = VADProcessor(vad_engine="firered-vad")
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.2, (
        f"`VADProcessor.__init__` chặn {elapsed * 1000:.0f}ms — nó đang nạp engine "
        f"trên event loop (QWEN-Q2)"
    )
    assert proc._engine is None, "chưa được nạp engine ngay trong __init__"
    assert started.wait(timeout=2.0), "phải hẹn nạp engine ở thread nền"


def test_init_khong_goi_thang_get_engine(monkeypatch):
    """Chốt trực tiếp: `__init__` không được chạm `get_engine` (đường có thể tải model)."""
    _fresh_pool(monkeypatch)
    monkeypatch.setattr(VADEngineFactory, "peek_engine", classmethod(lambda cls, n: None))

    calls = []

    def _boom(name):
        calls.append(name)
        raise AssertionError("QWEN-Q2: __init__ gọi get_engine ⇒ có thể tải model trên loop")

    monkeypatch.setattr(VADEngineFactory, "get_engine", _boom)
    monkeypatch.setattr(VADProcessor, "_spawn_engine_load", lambda self, name: None)

    VADProcessor(vad_engine="firered-vad")   # không được raise
    assert calls == [], f"__init__ đã gọi get_engine: {calls}"


# ─────────────────────────────────────────────── 2. lock cấp lớp không bị giữ qua download

def test_peek_engine_va_is_cached_khong_bi_chan_khi_dang_tai_model(monkeypatch):
    """Trong lúc `prepare_files()` đang tải model, đường ĐỌC trạng thái phải tức thì."""
    _fresh_pool(monkeypatch)

    prepare_started = threading.Event()
    prepare_release = threading.Event()

    def _slow_prepare(cls):
        prepare_started.set()
        prepare_release.wait(timeout=5.0)   # giả lập hf_hub_download đang chạy

    monkeypatch.setattr(FireRedVADEngine, "prepare_files", classmethod(_slow_prepare))
    monkeypatch.setattr(FireRedVADEngine, "__init__", lambda self: None)

    loader = threading.Thread(
        target=lambda: VADEngineFactory.get_engine("firered-vad"), daemon=True
    )
    loader.start()
    try:
        assert prepare_started.wait(timeout=2.0), "get_engine không gọi prepare_files"

        # Đang tải model: hai hàm này được event loop gọi ⇒ phải trả về ngay.
        t0 = time.perf_counter()
        assert VADEngineFactory.is_cached("firered-vad") is False
        VADEngineFactory.peek_engine("firered-vad")
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1, (
            f"`is_cached`/`peek_engine` bị chặn {elapsed * 1000:.0f}ms trong lúc tải model — "
            f"chúng đang dùng CHUNG `_lock` với bước tải (QWEN-Q2)"
        )
    finally:
        prepare_release.set()
        loader.join(timeout=5.0)

    assert VADEngineFactory.is_cached("firered-vad") is True


def test_get_engine_goi_prepare_files_ngoai_lock_pool():
    """Chốt ở mức mã nguồn (AST): `prepare_files()` chỉ được gọi NGOÀI `_lock` pool."""
    node = _func_node("backend/vad/engines/__init__.py", "get_engine")
    calls = _dotted_calls(node)
    assert "cls._factory_for(engine).prepare_files()" in calls or any(
        c.endswith("prepare_files()") for c in calls
    ), "phải gọi hook chuẩn bị file"

    # Mọi call-site `prepare_files` phải nằm trong khối `with cls._prepare_lock:`
    sites = _call_sites_inside_prepare_lock(node)
    assert sites, (
        "`prepare_files()` không nằm trong `with cls._prepare_lock:` — tải model vẫn có thể "
        "giữ lock pool (QWEN-Q2)"
    )
    # Và KHÔNG được nằm trong khối `with cls._lock:` đầu tiên (khối peek).
    assert not _call_sites_inside_pool_lock(node), (
        "`prepare_files()` vẫn nằm TRONG `with cls._lock` ⇒ tải model vẫn giữ lock pool"
    )


# ─────────────────────────────────────────────── 3. feed_chunk không tải model

def test_feed_chunk_bo_chunk_thay_vi_tai_model_khi_engine_chua_san_sang(monkeypatch):
    """Engine chưa có trong pool ⇒ BỎ chunk, KHÔNG gọi `get_engine` trên đường hot."""
    _fresh_pool(monkeypatch)
    monkeypatch.setattr(VADEngineFactory, "peek_engine", classmethod(lambda cls, n: None))

    hot_calls = []

    def _boom(name):
        hot_calls.append(name)
        raise AssertionError("QWEN-Q2: feed_chunk gọi get_engine ⇒ tải model trên đường hot")

    monkeypatch.setattr(VADEngineFactory, "get_engine", _boom)

    proc = VADProcessor(vad_engine="firered-vad")
    before = metrics_collector.get_counter("vad.chunks_dropped_engine_not_ready")

    proc.feed_chunk(bytes(800))   # đang có thread nền "nạp" (sẽ nổ, được bắt)
    proc.feed_chunk(bytes(800))

    after = metrics_collector.get_counter("vad.chunks_dropped_engine_not_ready")
    assert after - before == 2, "phải đếm số chunk bị bỏ vì engine chưa sẵn sàng"

    time.sleep(0.1)
    # Worker nền có gọi get_engine (và bị bắt lỗi), nhưng KHÔNG được gọi từ feed_chunk.
    assert proc._state is None, "không được dựng state khi engine chưa có"


def test_feed_chunk_chay_binh_thuong_khi_engine_da_co_trong_pool(monkeypatch):
    """Đường thường gặp (đã prewarm): `peek_engine` có engine ⇒ phân câu hoạt động."""
    _fresh_pool(monkeypatch)
    fake = FakeVADEngine()
    monkeypatch.setattr(VADEngineFactory, "peek_engine", classmethod(lambda cls, n: fake))

    speech_events = []
    proc = VADProcessor(
        vad_engine="firered-vad",
        on_speech_chunk=lambda *a: speech_events.append("chunk"),
        on_speech_start=lambda: speech_events.append("start"),
    )
    assert proc._engine is fake, "engine đã có trong pool thì lấy ngay (rẻ)"

    pcm = make_speech_pcm(1.0)
    raw = pcm_to_int16_bytes(pcm)
    for i in range(0, len(raw) - 800, 800):
        proc.feed_chunk(raw[i:i + 800])

    assert "start" in speech_events, "VAD phải phát hiện bắt đầu nói"
    assert speech_events.count("chunk") > 0, "audio trong câu phải được đẩy sang ASR"


# ─────────────────────────────────────────────── 4. prewarm đồng bộ

def test_prewarm_nap_dong_bo_engine_va_state(monkeypatch):
    """`prewarm()` là đường DUY NHẤT còn nạp đồng bộ — dùng cho thread khởi động."""
    _fresh_pool(monkeypatch)
    fake = FakeVADEngine()
    monkeypatch.setattr(VADEngineFactory, "get_engine", classmethod(lambda cls, n: fake))
    monkeypatch.setattr(VADEngineFactory, "peek_engine", classmethod(lambda cls, n: None))

    proc = VADProcessor(vad_engine="firered-vad")
    assert proc._engine is None

    assert proc.prewarm() is True
    assert proc._engine is fake
    assert proc._state is not None
    assert proc._frame_samples == fake.native_frame_samples


# ─────────────────────────────────────────────── 5. chốt ở mức mã nguồn

def test_ensure_engine_khong_bao_gio_goi_factory_get_engine():
    """`_ensure_engine` phải là non-blocking: chỉ `peek_engine`, không `get_engine`."""
    node = _func_node("backend/vad/processor.py", "_ensure_engine")
    calls = _dotted_calls(node)

    assert any(c.endswith("peek_engine()") for c in calls), "phải đọc pool qua peek_engine"
    assert not any(c.endswith("VADEngineFactory.get_engine()") for c in calls), (
        "`_ensure_engine` còn gọi `get_engine` ⇒ có thể tải model trên đường hot (QWEN-Q2)"
    )
    assert any(isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
               and n.value.value is False for n in ast.walk(node)), (
        "phải `return False` để caller bỏ chunk thay vì treo"
    )


def test_init_khong_goi_ensure_engine():
    """`__init__` (chạy trên event loop) không được gọi `_ensure_engine`."""
    node = _func_node("backend/vad/processor.py", "__init__")
    calls = _dotted_calls(node)

    assert not any(c.endswith("self._ensure_engine()") for c in calls), (
        "`__init__` còn gọi `_ensure_engine()` — bản cũ chính là chỗ treo event loop"
    )
    assert any(c.endswith("peek_engine()") for c in calls), "phải chỉ ĐỌC pool (rẻ)"
    assert any(c.endswith("_spawn_engine_load()") for c in calls), (
        "chưa có engine thì phải hẹn nạp ở thread nền"
    )
