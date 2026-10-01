"""Pipeline B v2 — Lookahead Video Buffering: kiểm thử tầng core + tích hợp.

Bộ test này khoá lại ĐÚNG các lỗi đã làm bản v1 "mất phụ đề gốc và bản dịch":

1. Audio rời rạc phải được ghép thành dòng PCM LIÊN TỤC (lấp khe hở bằng silence) để
   ánh xạ `sample -> PTS tuyệt đối` luôn đúng.
2. Phụ đề phải được tạo bởi ĐÚNG chuỗi VAD -> commit câu -> ASR -> dịch như Pipeline A
   (không còn cắt theo lưới 4 giây), và mang mốc PTS của người nói thật.
3. Tua video phải huỷ kết quả đang bay (dịch xong sau khi tua thì KHÔNG được gửi).
"""

import asyncio
import json
import struct
from typing import Any, Dict, List

import numpy as np
import pytest

from backend.core.lookahead_timeline import ContinuousAudioTimeline
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


# ──────────────────────────────────────────────────────────────────────────────
# 1. ContinuousAudioTimeline
# ──────────────────────────────────────────────────────────────────────────────

def test_timeline_stitches_contiguous_chunks():
    tl = ContinuousAudioTimeline(sample_rate=16000)
    tl.append(0.0, np.full(16000 * 2, 0.5, dtype=np.float32))
    tl.append(2.0, np.full(16000 * 2, 0.8, dtype=np.float32))

    pts, pcm = tl.read(10.0)
    assert pts == pytest.approx(0.0, abs=1e-6)
    assert len(pcm) == 16000 * 2
    assert tl.buffered_end_pts() == pytest.approx(4.0, abs=0.05)

    pts2, pcm2 = tl.read(10.0)
    assert pts2 == pytest.approx(2.0, abs=1e-6)
    assert pcm2[0] == pytest.approx(0.8)


def test_timeline_fills_gap_with_silence_preserving_absolute_pts():
    """Khe hở PHẢI được lấp bằng silence, nếu không mọi mốc phụ đề sau đó bị lệch."""
    tl = ContinuousAudioTimeline(sample_rate=16000)
    tl.append(0.0, np.full(16000 * 2, 0.5, dtype=np.float32))
    tl.append(5.0, np.full(16000 * 2, 0.9, dtype=np.float32))

    # Vùng liên tục chỉ tới 2.0s (khe hở 2.0 -> 5.0)
    assert tl.buffered_end_pts() == pytest.approx(2.0, abs=0.05)

    tl.read(10.0)  # đọc 2 s đầu
    pts_gap, gap = tl.read(10.0)
    assert pts_gap == pytest.approx(2.0, abs=1e-6)
    assert len(gap) == pytest.approx(16000 * 3, abs=32)
    assert np.allclose(gap, 0.0)

    pts_next, _ = tl.read(10.0)
    assert pts_next == pytest.approx(5.0, abs=1e-6)
    assert tl.gap_filled_sec == pytest.approx(3.0, abs=0.05)
    assert tl.cursor_pts == pytest.approx(7.0, abs=0.05)


def test_timeline_trims_overlapping_audio_without_losing_it():
    tl = ContinuousAudioTimeline(sample_rate=16000)
    tl.append(0.0, np.full(16000 * 2, 0.5, dtype=np.float32))
    tl.read(10.0)  # tiêu thụ hết [0, 2)

    # Đoạn mới chồng lấn 1 s: chỉ phần từ 2.0s trở đi là mới.
    tl.append(1.0, np.full(16000 * 2, 0.7, dtype=np.float32))
    pts, pcm = tl.read(10.0)
    assert pts == pytest.approx(2.0, abs=1e-6)
    assert len(pcm) == pytest.approx(16000, abs=32)


def test_timeline_drops_fully_consumed_chunk():
    tl = ContinuousAudioTimeline(sample_rate=16000)
    tl.read(1.0)  # chưa có dữ liệu -> None
    assert tl.read(1.0) is None
    tl.append(10.0, np.ones(16000, dtype=np.float32))
    assert tl.cursor_pts == pytest.approx(10.0)
    tl.append(5.0, np.ones(16000 * 5, dtype=np.float32))  # nằm trước con trỏ, đã đọc hết
    pts, pcm = tl.read(10.0)
    assert pts == pytest.approx(10.0, abs=1e-6)
    assert len(pcm) == 16000


def test_timeline_memory_cap_bounds_pending():
    tl = ContinuousAudioTimeline(sample_rate=16000, max_pending_sec=10.0)
    for i in range(50):
        tl.append(float(i * 20), np.ones(16000, dtype=np.float32))
    assert tl.pending_seconds() <= 11.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. Ánh xạ sample ASR -> PTS và tạo phụ đề có mốc thời gian
# ──────────────────────────────────────────────────────────────────────────────

def _make_session(translation_engine=None) -> LookaheadSessionState:
    conn = MockSafeConnection()
    engine = FakeInferenceEngine(text_fn=lambda n: "hello world from lookahead")
    session = LookaheadSessionState(
        ws=conn, asr_engine=engine, translation_engine=translation_engine or FakeTranslator()
    )
    session.conn = conn  # type: ignore[attr-defined]
    session.init_components(vad_engine_override=FakeVADEngine())
    return session


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cấu hình popup (VAD / ASR / phân câu) phải được áp cho Pipeline B
# ──────────────────────────────────────────────────────────────────────────────

def test_lookahead_init_applies_popup_vad_and_sentence_settings():
    """`lookahead_init` mang cấu hình popup ⇒ VAD + phân câu phải theo đúng thông số đó.

    Hồi quy: bản trước chỉ nhận `source_lang`/`target_lang`/`lead_time`, còn VAD engine,
    threshold, silence, độ ổn định cắt câu, số từ tối thiểu… đều dùng mặc định của server.
    """
    from backend.config import config as app_config

    conn = MockSafeConnection()
    session = LookaheadSessionState(
        ws=conn, asr_engine=FakeInferenceEngine(), translation_engine=FakeTranslator()
    )
    session.apply_init({
        "source_lang": "en",
        "target_lang": "vi",
        "lead_time": 12,
        "vadEngine": "silero-vad",
        "vadThreshold": 0.71,
        "silenceDurationMs": 900,
        "minWordsToCommit": 5,
        "splitOnStability": True,
        "stabilityDurationSec": 1.5,
        "stabilityMinDurationSec": 4.0,
        "stabilityMinWords": 6,
        "holdShortSentence": False,
    })
    session.init_components(vad_engine_override=FakeVADEngine())

    assert session.lead_time == pytest.approx(12.0)
    assert session.source_lang == "en"
    assert session.target_lang == "vi"
    assert session._min_words_to_commit == 5

    # VAD
    assert session.vad_processor.vad_engine == "silero-vad"
    assert session.vad_processor.threshold == pytest.approx(0.71)
    assert session.vad_processor.silence_duration_ms == 900

    # Phân câu
    cfg = session.asr_engine.commit_manager.cfg
    assert cfg.stability_duration_sec == pytest.approx(1.5)
    assert cfg.stability_min_duration_sec == pytest.approx(4.0)
    assert cfg.stability_min_words == 6
    assert cfg.min_words_to_commit == 5
    assert cfg.hold_short_sentence is False

    # Override RIÊNG của Lookahead vẫn phải thắng cấu hình popup.
    assert cfg.inactivity_timeout_sec == pytest.approx(app_config.lookahead.inactivity_timeout_sec)
    assert session.asr_engine.preview_requires_new_audio is True
    assert session.asr_engine.preview_enabled == app_config.lookahead.preview_enabled
    # Không được đụng cấu hình toàn cục của Pipeline A.
    assert app_config.sentence.stability_min_duration_sec != 4.0


@pytest.mark.asyncio
async def test_lookahead_live_config_update_changes_vad_and_min_words():
    """Đổi thiết lập trong popup khi ĐANG chạy cũng phải có hiệu lực tức thì."""
    session = _make_session()
    await session.handle_seek("s1", 0.0)

    applied = await session.apply_config({
        "type": "set_config",
        "action": "configure",
        "vadThreshold": 0.33,
        "silenceDurationMs": 700,
        "minWordsToCommit": 7,
        "stabilityMinWords": 9,
    })

    assert session.vad_processor.threshold == pytest.approx(0.33)
    assert session.vad_processor.silence_duration_ms == 700
    assert session._min_words_to_commit == 7
    assert session.asr_engine.commit_manager.cfg.stability_min_words == 9
    assert applied.get("min_words_to_commit") == 7

    # `silenceDurationMs = 0` nghĩa là "để engine dùng mặc định của nó".
    await session.apply_config({"type": "set_config", "silenceDurationMs": 0})
    assert session.vad_processor.silence_duration_ms is None


@pytest.mark.asyncio
async def test_lookahead_filters_short_sentences_by_popup_min_words():
    """`minWordsToCommit` của popup phải lọc được câu quá ngắn trong Pipeline B."""
    session = _make_session()  # text giả: "hello world from lookahead" = 4 token
    await session.handle_seek("s1", 0.0)
    await session.apply_config({"type": "set_config", "minWordsToCommit": 10})
    await session.start_tasks()
    try:
        session.timeline.append(0.0, make_speech_pcm(3.0))
        session.timeline.append(3.0, make_silence_pcm(1.0))
        await session.ingest_once()
        for _ in range(40):
            await asyncio.sleep(0.05)
        assert session.conn.of_type("lookahead_subtitles") == []  # type: ignore[attr-defined]
    finally:
        await session.close()

    # Hạ ngưỡng xuống 2 từ ⇒ câu phải được gửi đi.
    session2 = _make_session()
    await session2.handle_seek("s1", 0.0)
    await session2.apply_config({"type": "set_config", "minWordsToCommit": 2})
    await session2.start_tasks()
    try:
        session2.timeline.append(0.0, make_speech_pcm(3.0))
        session2.timeline.append(3.0, make_silence_pcm(1.0))
        await session2.ingest_once()
        subs = []
        for _ in range(100):
            subs = session2.conn.of_type("lookahead_subtitles")  # type: ignore[attr-defined]
            if subs:
                break
            await asyncio.sleep(0.05)
        assert subs, "Câu ngắn hợp lệ vẫn phải được gửi khi hạ minWordsToCommit"
    finally:
        await session2.close()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Lồng tiếng (TTS) cho Pipeline B
# ──────────────────────────────────────────────────────────────────────────────

def test_tts_fitted_bytes_shrinks_to_budget(monkeypatch):
    """Câu đọc dài hơn cửa sổ phụ đề phải được NÉN (giữ cao độ) để vừa ngân sách.

    Đây là tầng 1 của cơ chế chống trễ dây chuyền: nếu không nén, câu dài sẽ đẩy lùi mọi
    câu lồng tiếng sau nó.
    """
    from backend.tts.engine import OmniVoiceTTS

    tts = OmniVoiceTTS.__new__(OmniVoiceTTS)   # KHÔNG chạy __init__ (tránh nạp model thật)
    tts.sample_rate = 24000

    sr = 24000
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    fake = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    monkeypatch.setattr(tts, "_synthesize_audio", lambda text, voice, speed: (fake.copy(), 4.0), raising=False)

    # 1. Không nén khi đã vừa ngân sách.
    wav, dur = tts.synthesize_fitted_sync("xin chào", max_duration_sec=5.0)
    assert wav and dur == pytest.approx(4.0, abs=0.05)

    # 2. Ngân sách 2 s, trần tốc độ 1.45 ⇒ nén tối đa 1.45x (chưa đủ vừa ⇒ client cắt nốt).
    wav2, dur2 = tts.synthesize_fitted_sync("xin chào", max_duration_sec=2.0, max_speed=1.45)
    assert wav2
    assert dur2 == pytest.approx(4.0 / 1.45, abs=0.25)
    assert dur2 < 4.0

    # 3. Trần tốc độ 2.0 ⇒ vừa đúng ngân sách.
    wav3, dur3 = tts.synthesize_fitted_sync("xin chào", max_duration_sec=2.0, max_speed=2.0)
    assert wav3
    assert dur3 == pytest.approx(2.0, abs=0.25)

    # 4. Tốc độ gốc của người dùng được tính vào trần (speed=1.2, max_speed=1.45 ⇒ còn 1.21x).
    wav4, dur4 = tts.synthesize_fitted_sync("xin chào", speed=1.2, max_duration_sec=2.0, max_speed=1.45)
    assert wav4
    assert dur4 > 2.0   # bị chặn bởi trần tốc độ ⇒ phần dư do client xử lý


@pytest.mark.asyncio
async def test_lookahead_tts_enqueued_only_when_enabled():
    """Chỉ xếp hàng lồng tiếng khi popup bật TTS."""
    session = _make_session()
    session.tts_queue = asyncio.Queue(maxsize=4)

    session.tts_enabled = False
    session._enqueue_tts("xin chào", 1.0, 3.0)
    assert session.tts_queue.qsize() == 0

    session.tts_enabled = True
    session._enqueue_tts("xin chào", 1.0, 3.0)
    assert session.tts_queue.qsize() == 1
    item = session.tts_queue.get_nowait()
    assert item["text"] == "xin chào"
    assert item["start_pts"] == pytest.approx(1.0)
    assert item["end_pts"] == pytest.approx(3.0)


def test_pack_binary_frame_matches_client_format():
    """Khung nhị phân TTS phải đúng bố cục client đọc: [4B hdrlen][JSON][WAV]."""
    from backend.ws.lookahead_handler import _pack_binary_frame

    frame = _pack_binary_frame({"type": "lookahead_tts", "start_pts": 1.5}, b"WAVDATA")
    hdr_len = struct.unpack("<I", frame[:4])[0]
    header = json.loads(frame[4:4 + hdr_len].decode("utf-8"))
    assert header["type"] == "lookahead_tts"
    assert header["start_pts"] == pytest.approx(1.5)
    assert frame[4 + hdr_len:] == b"WAVDATA"


@pytest.mark.asyncio
async def test_lookahead_tts_config_from_popup_creates_worker():
    """`ttsEnabled` từ popup phải tạo hàng đợi + worker cho Pipeline B."""
    session = _make_session()
    await session.apply_config({
        "type": "set_config",
        "ttsEnabled": True,
        "ttsVoice": "speaker_02.wav",
        "ttsSpeed": 1.15,
    })
    assert session.tts_enabled is True
    assert session.tts_voice == "speaker_02.wav"
    assert session.tts_speed == pytest.approx(1.15)
    assert session.tts_queue is not None
    assert any(t.get_name().startswith("la_tts") for t in session._tasks)
    await session.close()

    # Tắt TTS ⇒ không xếp hàng nữa.
    session2 = _make_session()
    session2.tts_queue = asyncio.Queue(maxsize=4)
    await session2.apply_config({"type": "set_config", "ttsEnabled": False})
    assert session2.tts_enabled is False


# ──────────────────────────────────────────────────────────────────────────────
# 6. Canh mốc phụ đề/TTS khớp tiếng nói (bù phần VAD lùi mép câu)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_asr_start_pts_is_shifted_by_vad_start_pad():
    """VAD báo mép đoạn nói SỚM hơn thực tế ⇒ mốc phụ đề/TTS phải được cộng bù.

    Hồi quy: phụ đề (và lồng tiếng) hiện TRƯỚC tiếng nói trong video. Nguyên nhân: mỗi VAD
    engine lùi mép đầu đoạn nói để lấy ngữ cảnh (`pad_start_frame` / `speech_pad_ms` /
    `lookback_time_start_point`), và pipeline Lookahead dùng thẳng mốc đó.
    """
    session = _make_session()
    session._start_pad_ms = 200          # ví dụ: FSMN lookback_time_start_point
    session.sync_offset_ms = 0.0

    # Một mốc neo: đoạn nói [10.0s -> 12.0s] (audio liên tục quanh mốc neo).
    session._anchor_samples = [0]
    session._anchor_pts = [10.0]

    conn = session.conn  # type: ignore[attr-defined]
    await session._handle_asr_message({
        "type": "utterance_update",
        "is_final": True,
        "text": "hello world from lookahead",
        "start_sample": 0,
        "end_sample": 32000,
    })
    subs = conn.of_type("lookahead_subtitles")
    assert subs, "Không có phụ đề nào được gửi"
    item = subs[-1]["items"][0]
    # Trước khi bù sẽ là 10.0s; sau khi bù (200 ms) phải là 10.2s.
    assert item["start_pts"] == pytest.approx(10.2, abs=0.01), item
    assert item["end_pts"] == pytest.approx(12.0, abs=0.01), item


@pytest.mark.asyncio
async def test_user_sync_offset_is_applied_and_clamped():
    """Offset người dùng chỉnh trong popup được cộng thêm và bị kẹp an toàn."""
    session = _make_session()
    session._start_pad_ms = 50
    session.sync_offset_ms = -300        # người dùng thấy phụ đề hiện MUỘN ⇒ kéo lại sớm

    session._anchor_samples = [0]
    session._anchor_pts = [10.0]
    conn = session.conn  # type: ignore[attr-defined]

    await session._handle_asr_message({
        "type": "utterance_update",
        "is_final": True,
        "text": "hello world from lookahead",
        "start_sample": 0,
        "end_sample": 16000,
    })
    item = conn.of_type("lookahead_subtitles")[-1]["items"][0]
    # 10.0 + 0.05 - 0.30 = 9.75
    assert item["start_pts"] == pytest.approx(9.75, abs=0.01), item

    # Không bao giờ để mốc bắt đầu vượt quá mốc kết thúc (câu rất ngắn).
    session2 = _make_session()
    session2._start_pad_ms = 1200
    session2._anchor_samples = [0]
    session2._anchor_pts = [10.0]
    await session2._handle_asr_message({
        "type": "utterance_update",
        "is_final": True,
        "text": "hello world from lookahead",
        "start_sample": 0,
        "end_sample": 3200,   # câu chỉ 0,2 s
    })
    item2 = session2.conn.of_type("lookahead_subtitles")[-1]["items"][0]  # type: ignore[attr-defined]
    assert item2["start_pts"] < item2["end_pts"]


def test_vad_engines_declare_start_pad():
    """Mỗi VAD engine phải khai báo `start_pad_ms` để pipeline canh lại mốc."""
    import inspect
    from backend.vad.engines import firered, fsmn, silero

    for mod, attr in ((firered, "pad_start_frame"), (silero, "speech_pad_ms"), (fsmn, "lookback_time_start_point")):
        src = inspect.getsource(mod)
        assert "self.start_pad_ms" in src, f"{mod.__name__} thiếu start_pad_ms"
        assert attr in src, f"{mod.__name__} thiếu tham số {attr}"


@pytest.mark.asyncio
async def test_tts_queue_is_trimmed_and_stale_jobs_skipped():
    """TTS là best-effort: hàng đợi phải bị chặn trần và câu cũ bị bỏ.

    Hồi quy sự cố TRÀN VRAM: khi một câu TTS mất ~30 s, hàng đợi phình ra và GPU bị chiếm
    liên tục ⇒ ASR đói ⇒ MẤT CẢ phụ đề. Hàng đợi phải giữ câu MỚI NHẤT.
    """
    from backend.config import config as app_config

    session = _make_session()
    session.tts_enabled = True
    session.tts_queue = asyncio.Queue(maxsize=64)

    for i in range(20):
        session._enqueue_tts(f"câu {i}", float(i * 3), float(i * 3 + 2.5))

    limit = int(app_config.lookahead.tts_max_queue)
    assert session.tts_queue.qsize() <= limit, f"Hàng đợi TTS vượt trần: {session.tts_queue.qsize()}"
    # Câu giữ lại phải là những câu MỚI NHẤT.
    remaining = []
    while not session.tts_queue.empty():
        remaining.append(session.tts_queue.get_nowait()["text"])
    assert "câu 19" in remaining
    assert "câu 0" not in remaining


def test_clip_dubbing_text_cuts_at_word_boundary():
    from backend.ws.lookahead_handler import _clip_dubbing_text

    assert _clip_dubbing_text("ngắn", 20) == "ngắn"
    long_text = "đây là một câu rất dài " * 30
    out = _clip_dubbing_text(long_text, 50)
    assert len(out) <= 50
    assert not out.endswith(" ")
    assert long_text.startswith(out)


def test_free_vram_probe_never_raises():
    """Hàm đo VRAM phải an toàn kể cả khi không có GPU/torch."""
    from backend.ws.lookahead_handler import _empty_torch_cache, _free_vram_mb

    value = _free_vram_mb()
    assert value is None or value >= 0
    _empty_torch_cache()   # không được ném lỗi


# ──────────────────────────────────────────────────────────────────────────────
# 4. Stop phiên
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_close_is_idempotent_and_fast():
    """`close()` phải nhanh, không treo, và gọi lại nhiều lần vẫn an toàn."""
    session = _make_session()
    await session.start_tasks()
    t0 = asyncio.get_running_loop().time()
    await session.close()
    await session.close()
    elapsed = asyncio.get_running_loop().time() - t0
    assert elapsed < 8.0, f"close() quá chậm ({elapsed:.1f}s)"
    assert session._closed is True
    # Sau khi đóng: mọi đường vào đều bị chặn.
    await session.handle_audio_fragment(b"x" * 64, 0.0, "auto", "s1")
    assert await session.ingest_once() == 0


def test_anchor_mapping_survives_discontinuity():
    """Engine chỉ nhận frame ĐOẠN NÓI ⇒ chỉ số mẫu không tuyến tính với thời gian video."""
    session = _make_session()
    frame = b"\x00\x00" * 1600  # 0,1 s @16 kHz

    session._on_speech_chunk(frame, 10.0, "SPEECH")
    session._on_speech_chunk(frame, 10.1, "SPEECH")
    # Nhảy sang đoạn nói khác ở 30 s (giữa chừng có 19,8 s im lặng KHÔNG nạp vào ASR).
    session._on_speech_chunk(frame, 30.0, "SPEECH")

    assert session._pts_at(0) == pytest.approx(10.0)
    assert session._pts_at(1600) == pytest.approx(10.1, abs=1e-3)
    assert session._pts_at(3200) == pytest.approx(30.0, abs=1e-3)
    assert session._pts_at(4800) == pytest.approx(30.1, abs=1e-3)


@pytest.mark.asyncio
async def test_lookahead_end_to_end_produces_timed_subtitles():
    """E2E không cần model: PCM rời rạc -> VAD -> commit -> ASR -> dịch -> phụ đề có PTS."""
    session = _make_session()
    await session.handle_seek("seek_test", 0.0)
    await session.start_tasks()
    try:
        # 0,5 s im lặng, 3 s tiếng nói, 1 s im lặng, 3 s tiếng nói, 1 s im lặng chốt câu cuối
        session.timeline.append(0.0, make_silence_pcm(0.5))
        session.timeline.append(0.5, make_speech_pcm(3.0))
        session.timeline.append(3.5, make_silence_pcm(1.0))
        session.timeline.append(4.5, make_speech_pcm(3.0))
        session.timeline.append(7.5, make_silence_pcm(1.0))

        await session.ingest_once()

        subs: List[Dict[str, Any]] = []
        for _ in range(200):
            subs = session.conn.of_type("lookahead_subtitles")  # type: ignore[attr-defined]
            if len(subs) >= 2:
                break
            await asyncio.sleep(0.05)

        assert len(subs) >= 2, f"Không nhận được phụ đề Lookahead (nhận {len(subs)})"

        items = [it for m in subs for it in m["items"]]
        first = items[0]
        # Câu đầu: tiếng nói bắt đầu ~0,5 s; câu có thể được chốt bằng BẬC 3 (text ổn định,
        # sớm nhất ~2,5 s audio) hoặc bằng VAD END sau ~0,45 s im lặng (~4,0 s).
        assert 0.3 <= first["start_pts"] <= 0.8, first
        assert 2.6 <= first["end_pts"] <= 4.6, first
        assert first["original_text"]
        assert first["translated_text"].startswith("[vi]")

        # Mọi mốc phải TĂNG DẦN và không chồng lấn quá nhiều.
        starts = [it["start_pts"] for it in items]
        assert starts == sorted(starts)
        for it in items:
            assert it["end_pts"] > it["start_pts"]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_lookahead_seek_discards_inflight_translation():
    """Tua video trong lúc đang dịch ⇒ kết quả cũ KHÔNG được gửi về client."""

    class SeekDuringTranslate(FakeTranslator):
        def __init__(self):
            super().__init__()
            self.session = None
            self.done = asyncio.Event()

        async def translate(self, text, source_lang="auto", target_lang="vi", context=""):
            if self.session is not None:
                await self.session.handle_seek("seek_after", 500.0)
            self.done.set()
            return {"translated_text": f"[vi] {text}"}

    translator = SeekDuringTranslate()
    session = _make_session(translation_engine=translator)
    translator.session = session
    await session.handle_seek("seek_before", 0.0)
    await session.start_tasks()
    try:
        session.timeline.append(0.0, make_speech_pcm(3.0))
        session.timeline.append(3.0, make_silence_pcm(1.0))
        await session.ingest_once()

        for _ in range(100):
            if translator.done.is_set():
                break
            await asyncio.sleep(0.05)
        assert translator.done.is_set(), "Không chạy tới bước dịch"

        assert session.conn.of_type("lookahead_subtitles") == []  # type: ignore[attr-defined]
        assert session.active_seek_id == "seek_after"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_lookahead_ingest_respects_lead_time_window():
    """Chỉ nạp audio tới `currentTime + lead_time + margin`, không đốt GPU cho tương lai xa."""
    session = _make_session()
    session.lead_time = 10.0
    session.current_time = 0.0
    await session.handle_seek("s1", 0.0)

    # 30 s audio sẵn có nhưng cửa sổ nhu cầu chỉ ~16 s.
    session.timeline.append(0.0, make_speech_pcm(30.0))
    await session.ingest_once()

    fed_pts = session._feed_pts
    assert fed_pts is not None
    assert fed_pts <= 17.0, f"Nạp quá xa: {fed_pts:.2f}s"
    assert fed_pts >= 15.0, f"Nạp quá ít: {fed_pts:.2f}s"

    # Tua tới 100 s -> nhu cầu dịch chuyển theo.
    await session.handle_seek("s2", 100.0)
    assert session._feed_target_pts() == pytest.approx(100.0 + 10.0 + 6.0, abs=0.01)


@pytest.mark.asyncio
async def test_lookahead_cuts_long_speech_by_text_stability():
    """BẬC 3 (STABLE_PREFIX) PHẢI hoạt động trong Pipeline B.

    Đây là hồi quy cho lỗi "câu quá dài": bản đầu tắt `preview_enabled` nên BẬC 3 chết,
    câu chỉ còn được cắt bởi VAD-END / MAX_DURATION ⇒ dài tới 10-15 s dù có nhiều mốc
    ngắt tự nhiên. Với text preview ổn định, câu phải được cắt SỚM HƠN `max_duration_sec`.
    """
    session = _make_session()  # FakeInferenceEngine trả text KHÔNG ĐỔI
    await session.handle_seek("s1", 0.0)
    await session.start_tasks()
    try:
        # Nạp 1 s audio mỗi nhịp (giống luồng chunk thật về tới theo thời gian) để nhịp
        # preview có cơ hội chạy nhiều lần trên cùng một câu.
        subs = []
        for i in range(12):
            session.timeline.append(float(i), make_speech_pcm(1.0))
            await session.ingest_once()
            await asyncio.sleep(0.25)
            subs = session.conn.of_type("lookahead_subtitles")  # type: ignore[attr-defined]
            if subs:
                break

        assert subs, "Không có câu nào được chốt"
        first = subs[0]["items"][0]
        span = first["end_pts"] - first["start_pts"]
        assert span < 10.0, f"Câu bị dài bất thường ({span:.1f}s) — BẬC 3 không chạy?"
        assert first["end_pts"] < 12.0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_lookahead_no_spurious_cut_when_audio_stalls():
    """Audio KHÔNG tiến ⇒ TUYỆT ĐỐI không được cắt câu vì "text đứng im".

    Pipeline B nạp audio theo lô: giữa hai lô, nhịp poll đọc lại ĐÚNG một cửa sổ audio nên
    text không đổi. Nếu BẬC 3 không được chặn (`preview_requires_new_audio`) thì mỗi lần
    chờ dữ liệu sẽ cắt oan một câu giữa chừng.
    """
    session = _make_session()
    await session.handle_seek("s1", 0.0)
    session.asr_engine.preview_min_new_audio_sec = 0.5
    await session.start_tasks()
    try:
        # 6 s tiếng nói (dài hơn `stability_min_duration_sec` 2,5 s) rồi DỪNG nạp.
        session.timeline.append(0.0, make_speech_pcm(6.0))
        await session.ingest_once()
        assert session.conn.of_type("lookahead_subtitles") == []  # type: ignore[attr-defined]

        # Chờ 1,5 s: hàng chục nhịp poll chạy trên CÙNG một cửa sổ audio.
        for _ in range(30):
            await session.ingest_once()
            await asyncio.sleep(0.05)
        assert session.conn.of_type("lookahead_subtitles") == [], (  # type: ignore[attr-defined]
            "Cắt oan câu khi audio không tiến (BẬC 3 đọc text preview cũ)"
        )

        # Có thêm audio (im lặng) ⇒ VAD đóng câu ⇒ phụ đề phải xuất hiện.
        session.timeline.append(6.0, make_silence_pcm(1.5))
        for _ in range(100):
            await session.ingest_once()
            if session.conn.of_type("lookahead_subtitles"):  # type: ignore[attr-defined]
                break
            await asyncio.sleep(0.05)
        assert session.conn.of_type("lookahead_subtitles"), "Không chốt được câu sau khi VAD đóng"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_lookahead_status_reports_prebuffer_readiness():
    session = _make_session()
    await session.handle_seek("s1", 0.0)
    session.lead_time = 10.0
    session.current_time = 0.0

    await session.send_status()
    st = session.conn.of_type("lookahead_status")[-1]  # type: ignore[attr-defined]
    assert st["prebuffer_ready"] is False

    # Buffer trình duyệt đã có 12 s và phụ đề đã phủ tới 10 s ⇒ sẵn sàng phát.
    session.timeline.append(0.0, make_speech_pcm(12.0))
    session._ready_until_pts = 10.0
    await session.send_status()
    st2 = session.conn.of_type("lookahead_status")[-1]  # type: ignore[attr-defined]
    assert st2["buffered_ahead"] == pytest.approx(12.0, abs=0.1)
    assert st2["prebuffer_ready"] is True
    assert st2["seek_id"] == "s1"


@pytest.mark.asyncio
async def test_audio_fragment_decodes_and_feeds_timeline():
    """Mảnh WebM/Opus thật (bị cắt nhỏ) phải đi được vào timeline qua handler."""
    import io

    av = pytest.importorskip("av")
    from backend.core.stream_demuxer import scan_webm_boundaries

    sr, duration = 48000, 4.0
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="webm")
    st = out.add_stream("libopus", rate=sr)
    st.layout = "mono"
    st.bit_rate = 128000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    for i in range(0, len(sig) - 960, 960):
        frame = av.AudioFrame.from_ndarray(sig[i:i + 960].reshape(1, -1), format="fltp", layout="mono")
        frame.sample_rate = sr
        for pkt in st.encode(frame):
            out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()
    webm = buf.getvalue()
    bounds = scan_webm_boundaries(webm)
    init, body = webm[:bounds[0]], webm[bounds[0]:]

    session = _make_session()
    await session.handle_seek("s1", 0.0)
    await session.handle_audio_fragment(init, 0.0, 'audio/webm; codecs="opus"', "s1", is_init=True, epoch=1)
    for i in range(0, len(body), 8192):
        await session.handle_audio_fragment(body[i:i + 8192], 0.0, 'audio/webm; codecs="opus"', "s1", epoch=1)

    assert session.fragments_received > 1
    assert session.chunks_decoded > 0
    assert session.timeline.buffered_end_pts() == pytest.approx(4.0, abs=0.2)


@pytest.mark.asyncio
async def test_new_session_anchors_to_initial_current_time():
    """Khi mở lại session giữa video (ví dụ @45.0s), session phải neo ngay tại 45s và bỏ qua audio cũ."""
    session = _make_session()
    session.apply_init({"current_time": 45.0, "source_lang": "en", "target_lang": "vi"})

    assert session.current_time == 45.0
    assert session._session_start_pts == 45.0
    assert session.timeline.cursor_pts == 45.0
    assert session.demuxer.last_pts == pytest.approx(44.5, abs=0.1)

    # Nạp audio cũ (0s - 10s): timeline phải bỏ qua hoàn toàn
    session.timeline.append(0.0, make_speech_pcm(10.0))
    assert session.timeline.pending_seconds() == 0.0
    assert session.timeline.cursor_pts == 45.0

    # Nạp audio khớp mốc (45s - 50s): timeline chấp nhận
    session.timeline.append(45.0, make_speech_pcm(5.0))
    assert session.timeline.pending_seconds() == pytest.approx(5.0, abs=0.01)
    assert session.timeline.cursor_pts == 45.0


@pytest.mark.asyncio
async def test_sync_state_recovers_timeline_lag():
    """Nếu client báo currentTime đã đi tới 50s nhưng timeline cursor còn ở 10s, backend phải tự neo lại."""
    session = _make_session()
    await session.handle_seek("init_0", 10.0)
    session.timeline.append(10.0, make_speech_pcm(2.0))
    assert session.timeline.cursor_pts == 10.0

    # Giả lập xử lý message sync_state với video đã phát tới 50.0s (lệch 40s)
    new_time = 50.0
    if new_time > 1.0 and (
        session.timeline.cursor_pts is not None
        and session.timeline.cursor_pts < (new_time - 3.0)
    ):
        await session.handle_seek(session.active_seek_id, new_time)

    assert session.current_time == 50.0
    assert session.timeline.cursor_pts == 50.0
    assert session.timeline.pending_seconds() == 0.0


@pytest.mark.asyncio
async def test_dynamic_tts_toggle_in_lookahead():
    """Tắt TTS giữa phiên phải xoá queue; bật lại phải nạp lại các câu sắp phát."""
    session = _make_session()
    session.tts_enabled = True
    session.tts_queue = asyncio.Queue()
    session.current_time = 10.0

    # Giả lập đã dịch 2 câu: 1 câu đã qua (5s-8s), 1 câu sắp phát (12s-15s)
    session._recent_utterances.append({
        "pts_start": 5.0, "pts_end": 8.0, "text": "past", "translated": "quá khứ"
    })
    session._recent_utterances.append({
        "pts_start": 12.0, "pts_end": 15.0, "text": "future", "translated": "tương lai"
    })
    session.tts_queue.put_nowait({"text": "old", "start_pts": 12.0, "end_pts": 15.0})
    assert session.tts_queue.qsize() == 1

    # 1. Tắt TTS giữa phiên
    await session.apply_config({"tts_enabled": False})
    assert session.tts_enabled is False
    assert session.tts_queue.empty()

    # 2. Bật lại TTS giữa phiên
    await session.apply_config({"tts_enabled": True})
    assert session.tts_enabled is True
    assert session.tts_sent == 0
    # Câu 'past' (8.0s <= 10.0s) bị bỏ qua, câu 'future' (15.0s > 10.0s) được nạp lại
    assert session.tts_queue.qsize() == 1
    queued = session.tts_queue.get_nowait()
    assert queued["text"] == "tương lai"
    assert queued["start_pts"] == 12.0


