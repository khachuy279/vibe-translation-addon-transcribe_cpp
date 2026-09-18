"""Pytest conftest cho backend/tests.

Bổ sung so với bản cũ:
- Fixture `restore_config` để mỗi test có thể sửa `backend.config.config` (singleton
  toàn cục) mà không làm hỏng test khác.
- Fixture `session_factory` dựng một `SessionState` với engine GIẢ (tầng A) — chạy
  trong vài mili-giây, không cần nạp model.
"""

import faulthandler
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Thêm thư mục gốc dự án vào sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.utils.cuda import setup_cuda_dll_paths
setup_cuda_dll_paths()

# ⚠️ THỨ TỰ NẠP DLL TRÊN WINDOWS LÀ QUAN TRỌNG:
# `transcribe_cpp_native` và `llama_cpp` mỗi bên bundle một `ggml-base.dll` riêng.
# Windows phân giải phụ thuộc DLL theo TÊN MODULE đã nạp, nên nếu `llama_cpp` được nạp
# trước thì `transcribe.dll` sẽ bind nhầm vào `ggml-base.dll` của llama_cpp và thất bại
# với `0xc0000139` (STATUS_ENTRYPOINT_NOT_FOUND) — tiến trình chết cứng khi import.
# `backend/main.py` tình cờ đúng thứ tự này (asr.engine nạp trước translation.engine);
# ở đây ta chốt thứ tự tường minh cho MỌI module test.
#
# ⚠️ NHƯNG phải để `backend.asr` BOOTSTRAP trước: `bootstrap()` đặt `TRANSCRIBE_LIBRARY`
# trỏ vào bundle `bin/` (nếu có). Nếu `import transcribe_cpp` chạy trước, native của WHEEL
# (chỉ Vulkan) đã vào `sys.modules` và bundle `bin/` (CUDA) KHÔNG áp được — cả suite test
# sẽ chạy sai backend so với bản phát hành. Thứ tự đúng vẫn bảo toàn yêu cầu nạp DLL ở
# trên, vì `transcribe_cpp` vẫn được nạp TRƯỚC `llama_cpp`.
try:  # pragma: no cover - phụ thuộc môi trường
    from backend.asr import native as _asr_native

    _asr_native.bootstrap()
    import transcribe_cpp  # noqa: F401
except Exception:
    pass

WAV_TEST_DIR = _PROJECT_ROOT / "wav_test"
REPORT_DIR = _PROJECT_ROOT / "report"

# Bản GỐC của hai method bị guard autouse thay thế (xem `allow_real_translation_methods`).
# Phải lấy trước khi guard chạy — tức ngay lúc import conftest.
try:  # pragma: no cover - phụ thuộc môi trường
    from backend.translation import engine as _trans_engine_mod

    _REAL_LOAD_MODEL = _trans_engine_mod.GGUFTranslator.load_model
    _REAL_RECONFIGURE = _trans_engine_mod.GGUFTranslator.reconfigure
except Exception:  # pragma: no cover
    _REAL_LOAD_MODEL = None
    _REAL_RECONFIGURE = None


def _hang_guard_timeout() -> float:
    """Số giây tối đa cho cả phiên test tầng A (0 = tắt rào chắn)."""
    raw = os.environ.get("PYTEST_HANG_TIMEOUT", "180")
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 180.0
    return timeout if timeout > 0 else 0.0


def pytest_collection_modifyitems(config, items) -> None:
    """Rào chắn chống treo: dump stack MỌI thread nếu vòng lặp test đứng quá lâu.

    Vì sao cần: đã đo được một lần chạy `pytest` (tầng A) treo bên trong vòng lặp
    streaming của `TranscribeEngine`, ngốn **33 GB RAM trong ~10 phút** mà không in
    thêm một dòng nào. Không có rào chắn này thì sự cố im lặng đó chỉ lộ ra dưới
    dạng "máy hết RAM".

    Hai lớp:
    1. `faulthandler.dump_traceback_later` — chốt theo TỔNG thời gian phiên.
    2. `stall_watchdog` (luồng thật) — bắt trường hợp event loop bị chặn (dump được cả
       khi `faulthandler` không kịp). Bật/tắt bằng `PYTEST_STALL_SEC` (mặc định 25s; 0 = tắt).

    Chỉ áp cho tầng A: nếu có bất kỳ test `slow`/`full` nào được chọn thì bỏ rào chắn
    (tầng B/C cố tình chạy lâu với model thật).
    """
    timeout = _hang_guard_timeout()
    heavy = any(
        item.get_closest_marker("slow") or item.get_closest_marker("full") for item in items
    )
    if heavy:
        return
    if timeout > 0:
        faulthandler.dump_traceback_later(timeout, exit=True)

    stall_raw = os.environ.get("PYTEST_STALL_SEC", "25")
    try:
        stall = float(stall_raw)
    except ValueError:
        stall = 25.0
    if stall > 0:
        try:
            from backend.utils import stall_watchdog

            stall_watchdog.start(threshold=stall)
        except Exception:  # pragma: no cover - không được làm hỏng bộ test
            pass


def pytest_unconfigure(config) -> None:
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:  # pragma: no cover
        pass
    try:
        from backend.utils import stall_watchdog

        stall_watchdog.stop()
    except Exception:  # pragma: no cover
        pass


@pytest.fixture
def wav_test_dir() -> Path:
    return WAV_TEST_DIR


@pytest.fixture
def report_dir() -> Path:
    return REPORT_DIR


def _snapshot(obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        out[name] = value
    return out


@pytest.fixture
def restore_config():
    """Khôi phục `backend.config.config` về trạng thái ban đầu sau test.

    `config` là singleton toàn cục nên test nào sửa nó cũng phải dùng fixture này.
    """
    from backend.config import config

    saved_top = {k: v for k, v in config.__dict__.items()}
    saved_sub = {name: _snapshot(getattr(config, name)) for name in
                 ("ws", "vad", "asr", "sentence", "translation", "tts", "audio_buffer", "metrics")}
    try:
        yield config
    finally:
        for k, v in saved_top.items():
            try:
                setattr(config, k, v)
            except Exception:
                pass
        for name, values in saved_sub.items():
            sub = getattr(config, name, None)
            if sub is None:
                continue
            for k, v in values.items():
                try:
                    setattr(sub, k, v)
                except Exception:
                    pass


@pytest.fixture(autouse=True)
def _forbid_heavy_model_loads(request, monkeypatch):
    """CHẶN test tầng A nạp model thật (translation/TTS/ASR).

    Vì sao cần: đã đo được tiến trình `pytest` giữ **4.765 MB VRAM** và chạy chậm, chỉ vì
    MỘT test vô tình nạp model dịch Hy-MT2-7B (4,6 GB) — nguyên nhân là một model key
    không tồn tại được `resolve_key()` ánh xạ về model mặc định `tencent`, mà file GGUF
    của nó CÓ thật trên đĩa.

    Guard này biến loại lỗi đó thành **thất bại ồn ào** ngay tại chỗ, thay vì âm thầm
    ngốn vài GB VRAM và vài phút. Test cần model thật phải được đánh dấu `@pytest.mark.slow`
    (hoặc `full`) — khi đó guard tự bỏ qua.
    """
    if request.node.get_closest_marker("slow") or request.node.get_closest_marker("full"):
        yield
        return

    from backend.asr import engine as asr_mod
    from backend.translation import engine as trans_mod
    from backend.tts import engine as tts_mod

    def _make_boom(what: str):
        def _boom(*args, **kwargs):
            raise AssertionError(
                f"Test tầng A cố NẠP MODEL THẬT ({what}). "
                f"Hãy dùng engine giả (xem backend/tests/fakes.py) hoặc đánh dấu "
                f"@pytest.mark.slow nếu thực sự cần model."
            )
        return _boom

    monkeypatch.setattr(trans_mod.GGUFTranslator, "load_model", _make_boom("translation"), raising=False)
    monkeypatch.setattr(trans_mod.GGUFTranslator, "reconfigure", _make_boom("translation"), raising=False)
    monkeypatch.setattr(tts_mod.OmniVoiceTTS, "load_model", _make_boom("tts"), raising=False)
    monkeypatch.setattr(asr_mod.TranscribeEngine, "_load_model_locked", _make_boom("asr"), raising=False)
    yield


@pytest.fixture
def allow_real_translation_methods(monkeypatch):
    """Khôi phục implementation THẬT của `load_model`/`reconfigure` trong test tầng A.

    Guard autouse `_forbid_heavy_model_loads` thay hai method này bằng bản "nổ" để chặn nạp
    model thật. Test nào cần kiểm tra đúng logic nạp/swap (nhưng đã tự monkeypatch
    `_build_llm`/`Llama` để không nạp gì) thì yêu cầu fixture này.
    """
    from backend.translation import engine as trans_mod

    monkeypatch.setattr(trans_mod.GGUFTranslator, "load_model", _REAL_LOAD_MODEL, raising=False)
    monkeypatch.setattr(trans_mod.GGUFTranslator, "reconfigure", _REAL_RECONFIGURE, raising=False)
    yield


class MockWebSocket:
    """Mock WebSocket để kiểm thử giao thức ASGI mà không cần mạng."""

    def __init__(self):
        import asyncio

        self.incoming_queue: "asyncio.Queue" = asyncio.Queue()
        self.sent_messages: list = []
        self.sent_bytes: list = []
        self._closed = False

    async def accept(self):
        pass

    async def receive(self):
        return await self.incoming_queue.get()

    async def send_text(self, text: str):
        if not self._closed:
            import json

            try:
                self.sent_messages.append(json.loads(text))
            except Exception:
                pass

    async def send_bytes(self, data: bytes):
        if not self._closed:
            self.sent_bytes.append(data)

    async def close(self, code: int = 1000):
        self._closed = True


@pytest.fixture
def session_factory(restore_config):
    """Trả về factory dựng SessionState với engine giả (tầng A)."""
    from backend.tests.fakes import FakeInferenceEngine, FakeVADEngine
    from backend.ws.connection import SafeWebSocketConnection
    from backend.ws.session import SessionState

    created = []

    def _make(
        text_fn=None,
        infer_delay: float = 0.0,
        rms_threshold: float = 0.02,
        initial_config: Optional[Dict[str, Any]] = None,
        engine_cls=None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ):
        ws = MockWebSocket()
        session = SessionState(SafeWebSocketConnection(ws))
        if initial_config:
            session.config.update(initial_config)

        cls = engine_cls or FakeInferenceEngine
        kwargs = dict(engine_kwargs or {})
        if cls is FakeInferenceEngine:
            kwargs.setdefault("text_fn", text_fn)
            kwargs.setdefault("infer_delay", infer_delay)
        engine = cls(**kwargs)
        vad = FakeVADEngine(rms_threshold=rms_threshold)
        session.init_components(asr_engine=engine, vad_engine=vad)
        session.mock_ws = ws  # type: ignore[attr-defined]
        created.append(session)
        return session

    yield _make

    for s in created:
        try:
            s.asr_engine._is_running = False
        except Exception:
            pass
