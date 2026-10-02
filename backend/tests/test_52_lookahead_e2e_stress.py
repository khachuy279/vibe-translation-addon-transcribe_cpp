"""E2E Stress & Edge Cases — Pipeline B (Lookahead Video Buffering).

Pipeline B CHỈ còn tuyến OFFLINE_BATCH (v2 "streaming" đã bị xoá ngày 2026-10-02), nên các test
dưới đây kiểm chứng bất biến qua những gì tuyến batch thật sự dùng: `ContinuousAudioTimeline`
và `LookaheadSessionState.handle_seek()`.

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

from backend.asr.forced_aligner import AlignedWord, SubtitleSentence
from backend.config import config
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
    """Ghi nhận mọi bản tin JSON gửi về client."""

    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        self.sent_messages.append(payload)
        return True

    def of_type(self, msg_type: str) -> List[Dict[str, Any]]:
        return [m for m in self.sent_messages if m.get("type") == msg_type]


class LocalForcedAligner:
    """Aligner giả TỐI GIẢN cho test stress: không nạp model PyTorch.

    Chia đều các từ của văn bản ASR trên độ dài khối rồi gom 4 từ thành một phụ đề — đủ để
    tuyến batch phát ra `lookahead_subtitles` mà không phụ thuộc việc `Qwen3-ForcedAligner` có
    nạp được trên máy chạy test hay không.
    """

    def __init__(self, group: int = 4):
        self.group = max(1, int(group))

    def prewarm(self):
        pass

    def align(self, audio: np.ndarray, text: str, language: str = "English") -> List[AlignedWord]:
        tokens = [t for t in (text or "").split() if t]
        if not tokens:
            return []
        dur = len(audio) / 16000.0
        step = dur / len(tokens)
        return [
            AlignedWord(
                text=tok,
                start_time=round(i * step, 3),
                end_time=round((i + 1) * step, 3),
            )
            for i, tok in enumerate(tokens)
        ]

    def group_words_to_subtitles(
        self, words: List[AlignedWord], language: str = "English", **kwargs
    ) -> List[SubtitleSentence]:
        subs: List[SubtitleSentence] = []
        for i in range(0, len(words), self.group):
            grp = words[i : i + self.group]
            if not grp:
                continue
            subs.append(
                SubtitleSentence(
                    text=" ".join(w.text for w in grp),
                    start_time=grp[0].start_time,
                    end_time=grp[-1].end_time,
                )
            )
        return subs


def _make_session() -> LookaheadSessionState:
    conn = MockSafeConnection()
    session = LookaheadSessionState(
        ws=conn,  # type: ignore[arg-type]
        # Câu CÓ dấu kết câu: tuyến batch GIỮ LẠI mảnh cuối chưa kết câu để ghép vào khối sau,
        # nên text không dấu câu sẽ không bao giờ được phát ra phụ đề.
        asr_engine=FakeInferenceEngine(text_fn=lambda n: "stress test sentence."),
        translation_engine=FakeTranslator(),
    )
    session.conn = conn  # type: ignore[attr-defined]
    session.init_components(vad_engine_override=FakeVADEngine())
    session._aligner_service = LocalForcedAligner()
    return session


def _feed_window_target_pts(session: LookaheadSessionState) -> float:
    """Mốc nạp xa nhất mà vòng lặp batch chấp nhận: `currentTime + lead_time × rate + margin`.

    Công thức này là của `_offline_batch_loop` (không còn hàm `_feed_target_pts()` riêng sau khi
    v2 bị xoá) — test dùng lại đúng công thức để kiểm tra cửa sổ nạp co giãn theo tốc độ phát.
    """
    rate = max(1.0, float(session.playback_rate or 1.0))
    return float(session.current_time) + float(session.lead_time) * rate + float(
        config.lookahead.feed_margin_sec
    )


def _drain_timeline(session: LookaheadSessionState, target_pts: float, block_sec: float = 0.5) -> int:
    """Đọc timeline như tầng xử lý khối vẫn làm; trả về số khối đã đọc."""
    fed = 0
    while True:
        if session.timeline.cursor_pts is None or session.timeline.cursor_pts >= target_pts:
            break
        item = session.timeline.read(block_sec)
        if item is None:
            break
        pts, pcm = item
        if pcm.size:
            session._feed_pts = pts + pcm.size / 16000.0
            fed += 1
    return fed


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
        # Con trỏ đọc bám đúng mốc vừa tua, không giữ mốc của lần tua trước.
        assert session.current_time == pytest.approx(target)
        assert session.timeline.cursor_pts == pytest.approx(target)

    # Tua lần cuối tới vị trí đích và nạp dữ liệu "sạch" cho vị trí đó.
    await session.handle_seek("seek_final_target", 2500.0)
    session.timeline.append(2500.0, make_speech_pcm(2.0))
    session.timeline.append(2502.0, make_silence_pcm(1.0))

    assert session.active_seek_id == "seek_final_target"
    assert session.current_time == pytest.approx(2500.0)
    assert session.timeline.cursor_pts == pytest.approx(2500.0)
    # Audio của các mốc tua trước đó vẫn nằm trong kho RAM (bền vững), nhưng ranh giới
    # chống-trùng-lặp của khối đã bị xoá sạch ⇒ không có gì của đoạn cũ rò sang.
    assert session._batch_prev_emitted_end_audio is None
    assert session._batch_emitted_tail_norm == []
    assert session._sent_items == []

    await session.start_tasks()
    try:
        for _ in range(60):
            _drain_timeline(session, 2600.0)
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
    target_1x = _feed_window_target_pts(session)
    session.playback_rate = 2.0
    target_2x = _feed_window_target_pts(session)

    assert target_1x == pytest.approx(16.0, abs=0.01)
    assert target_2x == pytest.approx(26.0, abs=0.01)
    assert target_2x - target_1x == pytest.approx(10.0, abs=0.01)

    # Ở 2x, lượng audio đọc vào cũng phải nhiều hơn hẳn.
    session.timeline.append(0.0, make_speech_pcm(40.0))
    fed = _drain_timeline(session, target_2x)
    assert fed > 0
    assert session.timeline.cursor_pts is not None
    assert session.timeline.cursor_pts == pytest.approx(target_2x, abs=1.0)
    assert session._feed_pts is not None and session._feed_pts >= 25.0

    await session.close()


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
