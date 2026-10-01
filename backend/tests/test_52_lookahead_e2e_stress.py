"""E2E Stress & Edge Cases — Pipeline B (Lookahead Video Buffering) v2.

Kiểm thử:
1. Tua video liên tục (10 lần) — không rò rỉ phụ đề của đoạn cũ.
2. Co giãn cửa sổ nạp theo `playbackRate` (1.0x -> 2.0x).
3. Video dài 1 giờ — bộ nhớ của dòng âm thanh liên tục giữ ở mức O(1).
4. Bắt tay Dual-Engine (Pipeline A `/ws` và Pipeline B `/ws/lookahead`).
"""

import asyncio
import json
from typing import Any, Dict, List

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.core.lookahead_timeline import ContinuousAudioTimeline
from backend.main import app
from backend.tests.fakes import (
    FakeInferenceEngine,
    FakeTranslator,
    FakeVADEngine,
    make_silence_pcm,
    make_speech_pcm,
)
from backend.ws.lookahead_handler import LookaheadSessionState


class MockSafeConnection:
    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        self.sent_messages.append(payload)
        return True

    def of_type(self, msg_type: str) -> List[Dict[str, Any]]:
        return [m for m in self.sent_messages if m.get("type") == msg_type]


def _make_session() -> LookaheadSessionState:
    conn = MockSafeConnection()
    session = LookaheadSessionState(
        ws=conn,
        asr_engine=FakeInferenceEngine(text_fn=lambda n: "stress test sentence"),
        translation_engine=FakeTranslator(),
    )
    session.conn = conn  # type: ignore[attr-defined]
    session.init_components(vad_engine_override=FakeVADEngine())
    return session


@pytest.mark.asyncio
async def test_stress_rapid_seeking_race_condition():
    """STRESS 1: tua liên tục 10 lần — 100% không rò rỉ phụ đề/âm thanh của đoạn cũ."""
    session = _make_session()
    tries = [10.0, 45.0, 120.0, 300.0, 550.0, 800.0, 1200.0, 1500.0, 2000.0, 2500.0]

    for i, target in enumerate(tries):
        seek_id = f"seek_hop_{i}"
        await session.handle_seek(seek_id, target)
        # Mỗi lần tua lại có dữ liệu mới ở vị trí mới.
        session.timeline.append(target, make_speech_pcm(2.0))
        session.timeline.append(target + 2.0, make_silence_pcm(1.0))

    # Tua lần cuối tới vị trí đích và nạp dữ liệu "sạch" cho vị trí đó.
    await session.handle_seek("seek_final_target", 2500.0)
    session.timeline.append(2500.0, make_speech_pcm(2.0))
    session.timeline.append(2502.0, make_silence_pcm(1.0))

    assert session.active_seek_id == "seek_final_target"
    assert session.current_time == pytest.approx(2500.0)
    assert session.timeline.cursor_pts == pytest.approx(2500.0)
    assert session._anchor_samples == []

    await session.start_tasks()
    try:
        for _ in range(60):
            await session.ingest_once()
            if session.conn.of_type("lookahead_subtitles"):  # type: ignore[attr-defined]
                break
            await asyncio.sleep(0.05)

        subs = session.conn.of_type("lookahead_subtitles")  # type: ignore[attr-defined]
        assert subs, "Không có phụ đề nào cho vị trí sau khi tua"
        for msg in subs:
            assert msg["seek_id"] == "seek_final_target", f"Rò rỉ seek cũ: {msg['seek_id']}"
            for item in msg["items"]:
                assert item["start_pts"] >= 2499.0, item
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_adaptive_playback_rate_window_scaling():
    """STRESS 2: cửa sổ nạp audio phải co giãn theo tốc độ phát lại."""
    session = _make_session()
    session.lead_time = 10.0
    session.current_time = 0.0
    await session.handle_seek("rate", 0.0)

    session.playback_rate = 1.0
    target_1x = session._feed_target_pts()
    session.playback_rate = 2.0
    target_2x = session._feed_target_pts()

    assert target_1x == pytest.approx(16.0, abs=0.01)
    assert target_2x == pytest.approx(26.0, abs=0.01)
    assert target_2x - target_1x == pytest.approx(10.0, abs=0.01)

    # Ở 2x, lượng audio nạp vào cũng phải nhiều hơn hẳn.
    session.timeline.append(0.0, make_speech_pcm(40.0))
    await session.ingest_once()
    assert session._feed_pts is not None and session._feed_pts >= 25.0


def test_long_video_memory_stability_1_hour():
    """STRESS 3: 1 giờ audio (3600 s) — dòng âm thanh giữ bộ nhớ O(1) và mốc PTS liên tục."""
    tl = ContinuousAudioTimeline(sample_rate=16000, max_pending_sec=30.0)
    expected_pts = 0.0
    last_read_pts = 0.0

    for i in range(1200):  # 1200 đoạn x 3 s = 3600 s
        start = i * 3.0
        tl.append(start, np.full(16000 * 3, 0.25, dtype=np.float32))
        # Người tiêu thụ luôn đọc theo nhịp phát (không để tồn đọng vô hạn).
        item = tl.read(6.0)
        if item is not None:
            pts, pcm = item
            assert pts == pytest.approx(last_read_pts, abs=0.05), "Mốc PTS bị trôi"
            last_read_pts = pts + len(pcm) / 16000.0

    assert tl.pending_seconds() <= 31.0, f"Tồn đọng {tl.pending_seconds():.1f}s"
    assert last_read_pts >= 3590.0
    assert tl.gap_filled_sec == pytest.approx(0.0, abs=0.01)


def test_continuous_timeline_handles_hour_with_gaps():
    """Khe hở lớn (vùng buffer chưa có) vẫn được lấp silence đúng độ dài."""
    tl = ContinuousAudioTimeline(sample_rate=16000, max_pending_sec=200.0)
    tl.append(0.0, np.ones(16000 * 2, dtype=np.float32))
    tl.append(3600.0, np.ones(16000 * 2, dtype=np.float32))

    tl.read(10.0)
    pts, gap = tl.read(4000.0)
    assert pts == pytest.approx(2.0, abs=1e-6)
    assert len(gap) == pytest.approx(16000 * 3598, abs=16000)
    assert np.allclose(gap, 0.0)

    pts_next, _ = tl.read(10.0)
    assert pts_next == pytest.approx(3600.0, abs=1e-3)


def test_dual_engine_fallback_handshake(monkeypatch):
    """STRESS 4: `/ws/lookahead` và `/ws` realtime cùng tồn tại, không giành tài nguyên."""
    from backend.asr import engine as asr_mod
    from backend.translation import engine as trans_mod
    from backend.tests.fakes import FakeTranslator
    from backend.vad import processor as vad_proc

    fake_vad = FakeVADEngine()
    monkeypatch.setattr(asr_mod, "TranscribeEngine", FakeInferenceEngine, raising=False)
    monkeypatch.setattr(trans_mod, "get_translation_engine", lambda: FakeTranslator(), raising=False)
    monkeypatch.setattr(vad_proc.VADEngineFactory, "peek_engine", classmethod(lambda cls, n: fake_vad), raising=False)
    monkeypatch.setattr(vad_proc.VADEngineFactory, "get_engine", classmethod(lambda cls, n: fake_vad), raising=False)
    monkeypatch.setattr(vad_proc.VADEngineFactory, "is_cached", classmethod(lambda cls, n: True), raising=False)

    client = TestClient(app)
    with client.websocket_connect("/ws/lookahead") as ws:
        ws.send_text(json.dumps({"type": "lookahead_init"}))
        assert json.loads(ws.receive_text())["type"] == "lookahead_ready"

    with client.websocket_connect("/ws") as ws_realtime:
        res_init = json.loads(ws_realtime.receive_text())
        assert res_init.get("type") == "connected"
        ws_realtime.send_text(json.dumps({"type": "ping"}))
        while True:
            res_rt = json.loads(ws_realtime.receive_text())
            if res_rt.get("type") == "pong":
                break
