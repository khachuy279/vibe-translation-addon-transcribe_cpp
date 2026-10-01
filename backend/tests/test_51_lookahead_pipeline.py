"""Kiểm thử giao thức WebSocket `/ws/lookahead` (Pipeline B v2).

Dùng engine GIẢ (xem `backend/tests/fakes.py`) nên chạy trong vài giây, không cần GPU
hay file model: toàn bộ vòng đời phiên (init -> nhận mảnh -> VAD -> ASR -> trạng thái ->
tua video -> đóng) được kiểm chứng thật.
"""

import io
import json
import struct

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app


def create_test_wav_chunk(duration_sec: float = 1.0, sr: int = 16000, silent: bool = False) -> bytes:
    """Tạo byte WAV mẫu để truyền vào binary WebSocket frame."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    if silent:
        data = np.zeros_like(t, dtype=np.float32)
    else:
        data = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


def make_fragment(wav_bytes: bytes, timestamp_offset: float = 0.0, seek_id: str = "init_0",
                  is_init: bool = False, epoch: int = 0) -> bytes:
    """Đóng gói khung nhị phân: [4B header_len][JSON header][audio bytes]."""
    hdr = json.dumps({
        "timestamp_offset": timestamp_offset,
        "mime_type": "audio/wav",
        "seek_id": seek_id,
        "is_init": is_init,
        "epoch": epoch,
    }).encode("utf-8")
    return struct.pack("<I", len(hdr)) + hdr + wav_bytes


@pytest.fixture
def fake_lookahead_runtime(monkeypatch):
    """Thay ASR + VAD + Translation bằng bản GIẢ cho toàn bộ handler Lookahead."""
    from backend.asr import engine as asr_mod
    from backend.tests.fakes import FakeInferenceEngine, FakeTranslator, FakeVADEngine
    from backend.translation import engine as trans_mod
    from backend.vad import processor as vad_proc

    fake_vad = FakeVADEngine()
    fake_translator = FakeTranslator()

    monkeypatch.setattr(asr_mod, "TranscribeEngine", FakeInferenceEngine, raising=False)
    monkeypatch.setattr(trans_mod, "get_translation_engine", lambda: fake_translator, raising=False)
    monkeypatch.setattr(
        vad_proc.VADEngineFactory, "peek_engine", classmethod(lambda cls, name: fake_vad), raising=False
    )
    monkeypatch.setattr(
        vad_proc.VADEngineFactory, "get_engine", classmethod(lambda cls, name: fake_vad), raising=False
    )
    monkeypatch.setattr(
        vad_proc.VADEngineFactory, "is_cached", classmethod(lambda cls, name: True), raising=False
    )
    return fake_vad


def test_lookahead_init_ready_and_ping(fake_lookahead_runtime):
    """`lookahead_init` phải trả `lookahead_ready`; ping/pong vẫn hoạt động."""
    client = TestClient(app)
    with client.websocket_connect("/ws/lookahead") as ws:
        ws.send_text(json.dumps({
            "type": "lookahead_init",
            "source_lang": "en",
            "target_lang": "vi",
            "lead_time": 15,
        }))
        ready = json.loads(ws.receive_text())
        assert ready.get("type") == "lookahead_ready", ready
        assert ready.get("status") == "ready"
        assert ready.get("lead_time") == pytest.approx(15.0)

        ws.send_text(json.dumps({"type": "ping"}))
        while True:
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "pong":
                break
            assert msg.get("type") == "lookahead_status"


def test_lookahead_lead_time_is_clamped(fake_lookahead_runtime):
    """Thời gian dịch trước bị kẹp trong [10, 15] giây theo yêu cầu sản phẩm."""
    client = TestClient(app)
    with client.websocket_connect("/ws/lookahead") as ws:
        ws.send_text(json.dumps({"type": "lookahead_init", "lead_time": 99}))
        ready = json.loads(ws.receive_text())
        assert ready["lead_time"] == pytest.approx(15.0)

        ws.send_text(json.dumps({"type": "lookahead_init", "lead_time": 1}))
        while True:
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "lookahead_ready":
                assert msg["lead_time"] == pytest.approx(10.0)
                break


def test_lookahead_unavailable_when_asr_broken(monkeypatch):
    """Nếu không dựng được ASR, backend phải báo `lookahead_unavailable` để client fallback."""
    from backend.asr import engine as asr_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("native ASR không khả dụng")

    monkeypatch.setattr(asr_mod, "TranscribeEngine", _boom, raising=False)

    client = TestClient(app)
    with client.websocket_connect("/ws/lookahead") as ws:
        ws.send_text(json.dumps({"type": "lookahead_init"}))
        msg = json.loads(ws.receive_text())
        assert msg.get("type") == "lookahead_unavailable"
        assert "ASR" in msg.get("reason", "")


def test_lookahead_binary_fragment_seek_and_status(fake_lookahead_runtime):
    """Mảnh nhị phân + seek_reset + sync_state ⇒ có phụ đề mang seek_id mới."""
    client = TestClient(app)
    with client.websocket_connect("/ws/lookahead") as ws:
        ws.send_text(json.dumps({"type": "lookahead_init", "target_lang": "vi"}))
        assert json.loads(ws.receive_text())["type"] == "lookahead_ready"

        ws.send_text(json.dumps({"type": "seek_reset", "seek_id": "seek_42", "target_time": 0.0}))
        assert json.loads(ws.receive_text())["type"] == "seek_acknowledged"

        # 0,5 s im lặng -> 2 s tiếng nói -> 1 s im lặng (để VAD mở rồi đóng một câu).
        ws.send_bytes(make_fragment(create_test_wav_chunk(0.5, silent=True), 0.0, "seek_42", epoch=0))
        ws.send_bytes(make_fragment(create_test_wav_chunk(2.0), 0.5, "seek_42", epoch=0))
        ws.send_bytes(make_fragment(create_test_wav_chunk(1.0, silent=True), 2.5, "seek_42", epoch=0))

        ws.send_text(json.dumps({
            "type": "sync_state",
            "currentTime": 0.0,
            "playbackRate": 1.0,
            "paused": True,
            "seek_id": "seek_42",
        }))

        got_subtitles = False
        got_status = False
        for _ in range(40):
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "lookahead_subtitles":
                assert msg["seek_id"] == "seek_42"
                for item in msg["items"]:
                    assert item["end_pts"] > item["start_pts"]
                    assert item["original_text"]
                got_subtitles = True
            elif msg.get("type") == "lookahead_status":
                assert msg["seek_id"] == "seek_42"
                got_status = True
            if got_subtitles and got_status:
                break

        assert got_status, "Không nhận được lookahead_status"
        assert got_subtitles, "Không nhận được phụ đề Lookahead nào"


def test_lookahead_stale_seek_fragments_are_ignored(fake_lookahead_runtime):
    """Mảnh mang seek_id cũ (trước khi tua) phải bị bỏ, không làm bẩn timeline."""
    client = TestClient(app)
    with client.websocket_connect("/ws/lookahead") as ws:
        ws.send_text(json.dumps({"type": "lookahead_init"}))
        assert json.loads(ws.receive_text())["type"] == "lookahead_ready"

        ws.send_text(json.dumps({"type": "seek_reset", "seek_id": "seek_new", "target_time": 100.0}))
        assert json.loads(ws.receive_text())["type"] == "seek_acknowledged"

        # Mảnh của seek CŨ gửi tới muộn -> phải bị bỏ qua hoàn toàn.
        ws.send_bytes(make_fragment(create_test_wav_chunk(1.0), 0.0, "seek_old"))

        ws.send_text(json.dumps({"type": "sync_state", "currentTime": 100.0, "seek_id": "seek_new"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "lookahead_status"
        assert msg["seek_id"] == "seek_new"
        assert msg["buffered_ahead"] == pytest.approx(0.0, abs=0.01)


def test_lookahead_stop_command_ends_session(fake_lookahead_runtime):
    """Lệnh `stop` từ Extension phải ĐÓNG phiên ngay (hồi quy: server vẫn chạy sau Stop)."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)
    with client.websocket_connect("/ws/lookahead") as ws:
        ws.send_text(json.dumps({"type": "lookahead_init"}))
        assert json.loads(ws.receive_text())["type"] == "lookahead_ready"

        ws.send_bytes(make_fragment(create_test_wav_chunk(1.0), 0.0, "init_0"))

        # Người dùng bấm Stop ⇒ client gửi `stop` rồi mới đóng socket.
        ws.send_text(json.dumps({"type": "stop", "reason": "client_stop"}))

        # Server phải kết thúc phiên: không còn bản tin nào, kết nối bị đóng.
        closed = False
        try:
            for _ in range(30):
                ws.receive_text()
        except WebSocketDisconnect:
            closed = True
        except Exception:
            closed = True
        assert closed, "Backend vẫn giữ phiên Lookahead sau lệnh stop"


def test_lookahead_rejects_malformed_binary_frame(fake_lookahead_runtime):
    """Khung nhị phân hỏng không được làm sập phiên."""
    client = TestClient(app)
    with client.websocket_connect("/ws/lookahead") as ws:
        ws.send_text(json.dumps({"type": "lookahead_init"}))
        assert json.loads(ws.receive_text())["type"] == "lookahead_ready"

        ws.send_bytes(b"\x00\x01")                      # quá ngắn
        ws.send_bytes(struct.pack("<I", 10 ** 6) + b"x")  # độ dài header vô lý

        ws.send_text(json.dumps({"type": "ping"}))
        while True:
            msg = json.loads(ws.receive_text())
            if msg.get("type") == "pong":
                break
