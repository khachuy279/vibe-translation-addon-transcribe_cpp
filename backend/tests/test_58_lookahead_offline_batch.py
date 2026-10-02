"""Pipeline B v3 — Lookahead Offline Batch Processing & Bi-directional Context: Kiểm thử tích hợp.

Bộ test này xác nhận:
1. Feature flag `processing_mode`:
   - "streaming": chạy dual-path cũ (ingest_loop + asr_loop v2).
   - "offline_batch": chạy `_offline_batch_loop()`.
2. Vòng lặp `_offline_batch_loop()`:
   - Cắt khối audio từ LookaheadChunker.
   - Gọi ASR offline block decoding (`transcribe_block`).
   - Gióng hàng từ mốc thời gian qua ForcedAlignerService.
   - Gom từ thành SubtitleSentence.
   - Dịch thuật với Bi-directional Context (2 câu trước + 1 câu kế tiếp).
   - Gửi bản tin `lookahead_subtitles` đúng chuẩn mốc PTS.
3. Xử lý Seek Reset trong batch mode:
   - Khi tua video, cờ `_fast_bootstrap` bật lên (cắt khối nhanh 3-4s).
   - Vị trí `_batch_from_pts` dịch chuyển tới mốc tua.
   - Bản tin đang dở dang của seek_id cũ bị loại bỏ.
"""

import asyncio
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from backend.asr.forced_aligner import AlignedWord, SubtitleSentence
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


class FakeForcedAligner:
    """Aligner giả lập cho test không cần model PyTorch nặng."""

    def __init__(self):
        self.align_calls = []

    def prewarm(self):
        pass

    def align(self, audio: np.ndarray, text: str, language: str = "English") -> List[AlignedWord]:
        self.align_calls.append((len(audio), text, language))
        words = text.split()
        dur = len(audio) / 16000.0
        step = dur / max(1, len(words))
        items = []
        for i, w in enumerate(words):
            items.append(AlignedWord(
                text=w,
                start_time=round(i * step, 3),
                end_time=round((i + 1) * step, 3),
            ))
        return items

    def group_words_to_subtitles(self, words: List[AlignedWord], language: str = "English") -> List[SubtitleSentence]:
        if not words:
            return []
        # Gom nhóm 3 từ thành 1 câu
        subs = []
        chunk_size = 3
        for i in range(0, len(words), chunk_size):
            group = words[i : i + chunk_size]
            subs.append(SubtitleSentence(
                text=" ".join(w.text for w in group),
                start_time=group[0].start_time,
                end_time=group[-1].end_time,
            ))
        return subs


class ContextAwareFakeTranslator(FakeTranslator):
    """Translator ghi nhận cả context được truyền vào."""

    def __init__(self, prefix: str = "[vi] "):
        super().__init__(prefix=prefix)
        self.context_calls: List[str] = []

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> dict:
        self.calls.append(text)
        self.context_calls.append(context)
        return {"translated_text": f"{self.prefix}{text}", "elapsed_ms": 1.0}

    async def translate_sentence(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> dict:
        return await self.translate(text, source_lang, target_lang, context)


def _make_batch_session() -> LookaheadSessionState:
    conn = MockSafeConnection()
    asr = FakeInferenceEngine(text_fn=lambda n: "the doctor entered the room and smiled.")
    translator = ContextAwareFakeTranslator()
    session = LookaheadSessionState(
        ws=conn, asr_engine=asr, translation_engine=translator
    )
    session.apply_init({
        "source_lang": "en",
        "target_lang": "vi",
        "lead_time": 15,
        "processing_mode": "offline_batch",
    })
    session.init_components(vad_engine_override=FakeVADEngine())
    session._aligner_service = FakeForcedAligner()
    return session


@pytest.mark.asyncio
async def test_lookahead_processing_mode_flag_routing():
    """Kiểm tra cờ processing_mode chuyển hướng task đúng giữa streaming và offline_batch."""
    # 1. Mode streaming
    s_stream = LookaheadSessionState(
        ws=MockSafeConnection(), asr_engine=FakeInferenceEngine(), translation_engine=FakeTranslator()
    )
    s_stream.apply_init({"processing_mode": "streaming"})
    s_stream.init_components(vad_engine_override=FakeVADEngine())
    assert s_stream.processing_mode == "streaming"
    await s_stream.start_tasks()
    task_names = [t.get_name() for t in s_stream._tasks]
    assert any("la_ingest" in n for n in task_names)
    assert any("la_asr" in n for n in task_names)
    assert not any("la_batch" in n for n in task_names)
    await s_stream.close()

    # 2. Mode offline_batch
    s_batch = _make_batch_session()
    assert s_batch.processing_mode == "offline_batch"
    await s_batch.start_tasks()
    task_names = [t.get_name() for t in s_batch._tasks]
    assert any("la_batch" in n for n in task_names)
    assert not any("la_ingest" in n for n in task_names)
    assert not any("la_asr" in n for n in task_names)
    await s_batch.close()


@pytest.mark.asyncio
async def test_offline_batch_loop_end_to_end():
    """Kiểm tra trọn vẹn luồng offline batch: Audio -> Chunker -> ASR -> Aligner -> Context Translate -> Subtitles."""
    session = _make_batch_session()
    conn: MockSafeConnection = session.connection  # type: ignore

    await session.start_tasks()
    try:
        # Nạp 16s audio: 13s tiếng nói + 3s khoảng lặng
        pcm_speech = make_speech_pcm(13.0)
        pcm_silence = make_silence_pcm(3.0)
        session.timeline.append(0.0, pcm_speech)
        session.timeline.append(13.0, pcm_silence)
        session._ingest_event.set()

        # Đợi batch loop hoàn tất xử lý
        subs = []
        for _ in range(30):
            subs = conn.of_type("lookahead_subtitles")
            if subs:
                break
            await asyncio.sleep(0.05)

        assert subs, "Phải nhận được bản tin lookahead_subtitles trong batch mode"
        msg = subs[0]
        items = msg.get("items", [])
        assert len(items) >= 2, "Phải chia thành nhiều câu phụ đề nhỏ"

        # Kiểm tra nội dung phụ đề
        item0 = items[0]
        assert "start_pts" in item0 and "end_pts" in item0
        assert item0["start_pts"] >= 0.0
        assert item0["end_pts"] > item0["start_pts"]
        assert item0["original_text"] != ""
        assert item0["translated_text"].startswith("[vi] ")

        # Kiểm tra Bi-directional context ở translator
        trans: ContextAwareFakeTranslator = session.translation_engine  # type: ignore
        assert len(trans.calls) >= 2
        # Câu 1 phải có 'Next sentence:' trỏ tới câu 2
        ctx_call_0 = trans.context_calls[0]
        assert "Next sentence:" in ctx_call_0

        # Câu 2 phải có 'Previous context:' chứa câu 1
        ctx_call_1 = trans.context_calls[1]
        assert "Previous context:" in ctx_call_1

    finally:
        await session.close()


@pytest.mark.asyncio
async def test_offline_batch_seek_reset():
    """Khi tua video, batch mode phải bật fast bootstrap và reset con trỏ batch."""
    session = _make_batch_session()
    conn: MockSafeConnection = session.connection  # type: ignore

    # Giả lập trước khi seek: đang ở mốc 0.0s
    session._batch_from_pts = 30.0
    session._fast_bootstrap = False

    # Người dùng tua tới 50.0s
    await session.handle_seek("seek_99", 50.0)

    assert session.active_seek_id == "seek_99"
    assert session.current_time == 50.0
    assert session._batch_from_pts == 50.0
    assert session._fast_bootstrap is True, "Tua video phải bật Fast-Bootstrap để phụ đề hiện ngay"

    await session.close()
