"""Pipeline B v3 — Lookahead Offline Batch Processing & Bi-directional Context: Kiểm thử tích hợp.

Pipeline B CHỈ còn MỘT tuyến: OFFLINE_BATCH (v2 "streaming" và cờ `processing_mode` đã bị xoá
ngày 2026-10-02). Bộ test này xác nhận:

1. Nối dây duy nhất của tuyến batch: `start_tasks()` chỉ dựng vòng lặp xử lý khối (`la_batch_*`)
   và vòng báo trạng thái (`la_status_*`) — KHÔNG còn `la_ingest_*`/`la_asr_*` của bản v2.
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

from backend.asr.forced_aligner import AlignedWord, ForcedAlignerService, SubtitleSentence
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

    def group_words_to_subtitles(
        self,
        words: List[AlignedWord],
        language: str = "English",
        **kwargs,
    ) -> List[SubtitleSentence]:
        """Gom nhóm 3 từ thành 1 câu (bỏ qua các tham số trần của bản thật)."""
        if not words:
            return []
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


class GridForcedAligner:
    """Aligner giả: một từ mỗi 0.4s trải đều trên khối, gom 4 từ thành một phụ đề.

    Dùng để kiểm tra việc TRỪ CHỒNG LẤN ở ranh giới khối: mốc thời gian là tương đối trong khối
    nên nếu phía nhận không trừ, phụ đề đầu của khối sau sẽ chồng thời gian lên phụ đề cuối của
    khối trước (và lặp từ).
    """

    def __init__(self, word_sec: float = 0.4, group: int = 4):
        self.word_sec = word_sec
        self.group = group
        self.align_calls = 0

    def prewarm(self):
        pass

    def align(self, audio: np.ndarray, text: str, language: str = "English") -> List[AlignedWord]:
        self.align_calls += 1
        dur = len(audio) / 16000.0
        count = max(1, int(dur / self.word_sec))
        return [
            AlignedWord(
                text=f"w{i}",
                start_time=round(i * self.word_sec, 3),
                end_time=round((i + 1) * self.word_sec, 3),
            )
            for i in range(count)
        ]

    def group_words_to_subtitles(
        self, words: List[AlignedWord], language: str = "English", **kwargs
    ) -> List[SubtitleSentence]:
        subs = []
        for i in range(0, len(words), self.group):
            g = words[i : i + self.group]
            if not g:
                continue
            subs.append(
                SubtitleSentence(
                    text=" ".join(w.text for w in g),
                    start_time=g[0].start_time,
                    end_time=g[-1].end_time,
                )
            )
        return subs


def _all_subtitle_items(conn: MockSafeConnection) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for msg in conn.of_type("lookahead_subtitles"):
        items.extend(msg.get("items", []))
    items.sort(key=lambda it: float(it["start_pts"]))
    return items


class RealisticForcedAligner:
    """Bắt chước ĐÚNG `ForcedAlignerService.align()`: model bỏ hết dấu câu khi tokenize, rồi tầng
    service GẮN LẠI dấu câu từ văn bản ASR (`merge_source_text`).

    Vì sao cần bản giả này: `Qwen3-ForcedAligner` loại mọi ký tự không phải chữ/số
    (`is_kept_char`), nên nếu fake trả về từ CÓ dấu câu thì test không bao giờ chạm tới nhánh gắn
    lại dấu câu — đúng lỗi đã lọt lưới và gây sự cố 2026-10-02.
    """

    def __init__(self):
        self.seen_texts: List[str] = []

    def prewarm(self):
        pass

    def align(self, audio: np.ndarray, text: str, language: str = "English") -> List[AlignedWord]:
        self.seen_texts.append(text)
        tokens = [t for t in (text or "").split() if t]
        dur = len(audio) / 16000.0
        step = dur / max(1, len(tokens))
        items = [
            AlignedWord(
                text="".join(ch for ch in tok if ch.isalnum() or ch == "'"),
                start_time=round(i * step, 3),
                end_time=round((i + 1) * step, 3),
            )
            for i, tok in enumerate(tokens)
        ]
        return ForcedAlignerService.merge_source_text(items, text)

    def group_words_to_subtitles(
        self, words: List[AlignedWord], language: str = "English", **kwargs
    ) -> List[SubtitleSentence]:
        return ForcedAlignerService.group_words_to_subtitles(words, language=language, **kwargs)


class UpstreamLikeAligner:
    """Bắt chước ĐÚNG upstream `encode_timestamp()` để test hồi quy "không ngắt câu dài".

    Upstream chỉ tách từ riêng cho `japanese` (`tokenize_japanese`) và `korean`; mọi ngôn ngữ khác
    dùng `tokenize_space_lang`. Với văn bản tiếng Nhật (KHÔNG có khoảng trắng) mà truyền
    `language="English"` thì cả câu thành MỘT token ⇒ một item ⇒ không thể ngắt câu — đúng sự cố
    thật 2026-10-03: cả khối 3 câu hiện thành một phụ đề dài.
    """

    def __init__(self):
        self.languages: List[str] = []

    def prewarm(self):
        pass

    def align(self, audio: np.ndarray, text: str, language: str = "English") -> List[AlignedWord]:
        self.languages.append(language)
        dur = len(audio) / 16000.0
        if str(language).strip().lower() == "japanese":
            # Xấp xỉ `tokenize_japanese`: gom ~3 ký tự một token, dấu câu dính vào token trước.
            tokens: List[str] = []
            buf = ""
            for ch in text:
                buf += ch
                if not ch.isalnum() or len(buf) >= 3:
                    tokens.append(buf)
                    buf = ""
            if buf:
                tokens.append(buf)
        else:
            tokens = [t for t in text.split() if t]
        step = dur / max(1, len(tokens))
        items = [
            AlignedWord(
                text="".join(c for c in tok if c.isalnum()),
                start_time=round(i * step, 3),
                end_time=round((i + 1) * step, 3),
            )
            for i, tok in enumerate(tokens)
        ]
        return ForcedAlignerService.merge_source_text(items, text)

    def group_words_to_subtitles(
        self, words: List[AlignedWord], language: str = "English", **kwargs
    ) -> List[SubtitleSentence]:
        return ForcedAlignerService.group_words_to_subtitles(words, language=language, **kwargs)


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
    })
    session.init_components(vad_engine_override=FakeVADEngine())
    session._aligner_service = FakeForcedAligner()
    return session


@pytest.mark.asyncio
async def test_lookahead_start_tasks_wires_only_batch_pipeline():
    """`start_tasks()` phải chỉ dựng tuyến OFFLINE_BATCH (không còn nhánh streaming v2).

    Hồi quy: Pipeline B v2 từng chọn vòng lặp theo cờ `processing_mode` ("streaming" ⇒
    `la_ingest_*` + `la_asr_*`, "offline_batch" ⇒ `la_batch_*`). Cờ đó đã bị xoá, nên tuyến batch
    là ĐƯỜNG DUY NHẤT — nếu một task `la_ingest_*`/`la_asr_*` quay trở lại `_tasks` thì phiên sẽ
    chạy hai pipeline tranh nhau trên cùng một kho audio.
    """
    session = _make_batch_session()
    await session.start_tasks()
    try:
        task_names = [t.get_name() for t in session._tasks]
        assert any("la_batch" in n for n in task_names), task_names
        assert any("la_status" in n for n in task_names), task_names
        assert not any("la_ingest" in n for n in task_names), task_names
        assert not any("la_asr" in n for n in task_names), task_names
    finally:
        await session.close()


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
async def test_batch_auto_source_language_splits_long_japanese_utterance():
    """HỒI QUY 2026-10-03: `source_lang="auto"` + nội dung tiếng Nhật phải NGẮT CÂU.

    Ảnh chụp thật của người dùng: cả câu dài 3 câu (ほら…？/ うわ…よ。/ ああ…きた。) hiện thành MỘT
    phụ đề tràn 2 dòng và bản dịch cũng thành một khối dài. Nguyên nhân: với `auto`, aligner bị gọi
    bằng `language="English"` ⇒ upstream `tokenize_space_lang` biến cả câu tiếng Nhật thành MỘT
    token ⇒ một item ⇒ tầng gom câu không có gì để ngắt.
    """
    ja_text = (
        "ほら、あれ温泉街じゃない？"
        "うわ、なんかお土産屋とか書いてあるよ。"
        "ああ、なんか温泉の香りしてきた。"
    )
    conn = MockSafeConnection()
    aligner = UpstreamLikeAligner()
    session = LookaheadSessionState(
        ws=conn,
        asr_engine=FakeInferenceEngine(text_fn=lambda n: ja_text),
        translation_engine=FakeTranslator(),
    )
    session.apply_init({"source_lang": "auto", "target_lang": "vi"})
    session.init_components(vad_engine_override=FakeVADEngine())
    session._aligner_service = aligner

    await session.start_tasks()
    try:
        session.timeline.append(0.0, make_speech_pcm(13.0))
        session.timeline.append(13.0, make_silence_pcm(3.0))
        session._ingest_event.set()

        items: List[Dict[str, Any]] = []
        for _ in range(60):
            session._ingest_event.set()
            items = _all_subtitle_items(conn)
            if items:
                break
            await asyncio.sleep(0.05)

        assert aligner.languages, "Aligner chưa được gọi"
        assert aligner.languages[0] == "Japanese", (
            f"`auto` phải đoán được tiếng Nhật từ văn bản ASR, nhận {aligner.languages[0]!r}"
        )
        texts = [it["original_text"] for it in items]
        assert texts == [
            "ほら、あれ温泉街じゃない？",
            "うわ、なんかお土産屋とか書いてあるよ。",
            "ああ、なんか温泉の香りしてきた。",
        ], texts
        # Mốc tăng dần và không chồng lấn.
        for prev, nxt in zip(items, items[1:]):
            assert float(nxt["start_pts"]) >= float(prev["end_pts"]) - 0.05, (prev, nxt)
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


@pytest.mark.asyncio
async def test_offline_batch_trims_overlap_between_blocks():
    """Khối sau có CHỒNG LẤN với khối trước ⇒ phải TRỪ phần đã phát (không lặp từ / không đè mốc).

    Sự cố 2026-10-02 (log thật): khối `[58.91s -> 76.91s]` rồi khối sau bắt đầu ở `75.91s`
    (= 76.91 − overlap 1.0s) và 1.0s audio đó bị phiên âm LẦN HAI:
        '… From what I saw the other day, I could understand.'
        'Other day, I could understand why he and some people might find you.'
    """
    conn = MockSafeConnection()
    session = LookaheadSessionState(
        ws=conn,
        asr_engine=FakeInferenceEngine(text_fn=lambda n: "alpha beta gamma delta"),
        translation_engine=FakeTranslator(),
    )
    session.apply_init({
        "source_lang": "en",
        "target_lang": "vi",
    })
    session.init_components(vad_engine_override=FakeVADEngine())
    session._aligner_service = GridForcedAligner()

    # 60s "tiếng nói" LIÊN TỤC (sine biên độ lớn) ⇒ không có khoảng lặng nào
    # ⇒ chunker cắt cưỡng bức + 1.0s overlap.
    session.timeline.append(0.0, make_speech_pcm(60.0))
    session._batch_from_pts = 0.0

    await session.start_tasks()
    try:
        items: List[Dict[str, Any]] = []
        # Đợi tới khi khối THỨ HAI được xử lý (mốc đọc vượt qua cuối khối đầu) — nếu chỉ đợi có
        # phụ đề thì test sẽ dừng ngay sau khối đầu và không hề kiểm tra ranh giới.
        for _ in range(300):
            session._ingest_event.set()
            items = _all_subtitle_items(conn)
            if session._batch_from_pts is not None and session._batch_from_pts > 25.0:
                break
            await asyncio.sleep(0.05)

        assert len(items) >= 8, items
        # Có ít nhất hai khối ⇒ có ít nhất một ranh giới khối để kiểm tra
        assert any(float(it["start_pts"]) > 18.0 for it in items), "chưa thấy phụ đề của khối thứ hai"
        # Không cặp phụ đề liền nhau nào được CHỒNG THỜI GIAN lên nhau. Nếu phần chồng lấn không
        # được trừ, phụ đề đầu của khối sau sẽ bắt đầu TRƯỚC mốc kết thúc của khối trước.
        for prev, nxt in zip(items, items[1:]):
            assert float(nxt["start_pts"]) >= float(prev["end_pts"]) - 0.05, (
                f"phụ đề chồng lấn ranh giới khối: {prev} | {nxt}"
            )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_offline_batch_emits_whole_sentences_with_punctuation():
    """Sự cố 2026-10-02: ASR offline trả câu hoàn hảo nhưng phụ đề ra MẢNH CỤT KHÔNG DẤU CÂU.

    Nguyên nhân: `Qwen3-ForcedAligner` bỏ hết dấu câu khi tokenize ⇒ tầng gom câu không thấy dấu
    câu nào nên chỉ còn cắt theo trần (`max_words=10`). Ảnh chụp màn hình thật:
        ASR:      "You took ballet. God, you never listen. I'm excited to meet Emily. ..."
        Phụ đề:   "You took ballet God you never listen I'm"      ← cụt + mất dấu câu
    """
    asr_text = (
        "You took ballet. God, you never listen. I'm excited to meet Emily. "
        "Me too. I just hope he doesn't blow it."
    )
    conn = MockSafeConnection()
    session = LookaheadSessionState(
        ws=conn,
        asr_engine=FakeInferenceEngine(text_fn=lambda n: asr_text),
        translation_engine=FakeTranslator(),
    )
    session.apply_init({
        "source_lang": "en",
        "target_lang": "vi",
    })
    session.init_components(vad_engine_override=FakeVADEngine())
    session._aligner_service = RealisticForcedAligner()
    session.timeline.append(0.0, make_speech_pcm(60.0))
    session._batch_from_pts = 0.0

    await session.start_tasks()
    try:
        items: List[Dict[str, Any]] = []
        for _ in range(200):
            session._ingest_event.set()
            items = _all_subtitle_items(conn)
            if items:
                break
            await asyncio.sleep(0.05)

        assert items, "phải nhận được phụ đề"
        texts = [it["original_text"] for it in items]
        # Phụ đề phải là CÂU TRỌN VẸN có dấu câu, không phải mảnh cụt 10 từ
        assert texts[0] == "You took ballet.", texts
        assert "You took ballet God you never listen I'm" not in " ".join(texts)
        # MỌI phụ đề phát ra đều phải kết câu: mảnh cụt ở ranh giới khối bị GIỮ LẠI để ghép khối sau
        for t in texts:
            assert t.endswith((".", "?", "!")), t
        # Văn bản gửi đi phải khớp nguyên văn phiên bản ASR (không mất dấu câu)
        assert "God, you never listen." in texts
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unfinished_tail_is_held_back_for_next_block():
    """Mảnh cuối CHƯA kết câu của khối bị cắt giữa câu phải được GIỮ LẠI, không phát ra.

    Ảnh chụp màn hình thật 2026-10-02: `"I promise I'll be on my best behavior You better"` —
    mảnh cụt ở ranh giới khối. Nay nó được cất vào `_batch_carry_words` để ghép vào khối sau.
    """
    from types import SimpleNamespace

    session = _make_batch_session()
    try:
        def _sub(text, t0, t1, words):
            return SubtitleSentence(text, t0, t1, words=words)

        complete = _sub(
            "I promise I'll be on my best behavior.",
            0.0, 2.0,
            [AlignedWord("I", 0.0, 0.1), AlignedWord("behavior.", 1.8, 2.0)],
        )
        fragment = _sub(
            "You better",
            2.2, 3.4,
            [AlignedWord("You", 2.2, 2.7), AlignedWord("better", 2.7, 3.4)],
        )
        chunk_mid = SimpleNamespace(
            pts_start=10.0, pts_end=20.0, is_silence_boundary=False, fallback_mode="forced_overlap"
        )

        kept = session._hold_back_unfinished_tail([complete, fragment], chunk_mid, False)
        assert [s.text for s in kept] == ["I promise I'll be on my best behavior."]
        assert [w.text for w in session._batch_carry_words] == ["You", "better"]
        # Mốc cất giữ ở KHÔNG GIAN AUDIO TUYỆT ĐỐI (chưa cộng offset người dùng)
        assert session._batch_carry_words[0].start_time == pytest.approx(12.2)
        assert session._batch_carry_words[1].end_time == pytest.approx(13.4)

        # 1. Khối kết thúc ở KHOẢNG LẶNG THẬT ⇒ im lặng đã là ranh giới câu ⇒ phát bình thường
        chunk_sil = SimpleNamespace(
            pts_start=10.0, pts_end=20.0, is_silence_boundary=True, fallback_mode="silence_gap"
        )
        kept2 = session._hold_back_unfinished_tail([complete, fragment], chunk_sil, False)
        assert len(kept2) == 2 and not session._batch_carry_words

        # 2. Video đã hết ⇒ phải phát nốt, không được giữ lại rồi mất
        kept3 = session._hold_back_unfinished_tail([complete, fragment], chunk_mid, True)
        assert len(kept3) == 2 and not session._batch_carry_words

        # 3. Mảnh cuối đã TRỌN CÂU ⇒ không giữ gì
        kept4 = session._hold_back_unfinished_tail([complete, complete], chunk_mid, False)
        assert len(kept4) == 2 and not session._batch_carry_words
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_batch_leading_overlap_trimmed_by_word_sequence():
    """Trừ chồng lấn ranh giới bằng CHUỖI TỪ, bù sai số mốc giữa hai lần align.

    Đo thật 2026-10-02: lớp trừ THEO MỐC bỏ được 3 từ nhưng "In front of" vẫn lọt ra phụ đề vì
    mốc align của khối mới muộn hơn khối cũ. Lớp trừ theo chuỗi phải bắt được phần còn sót.
    """
    from types import SimpleNamespace

    session = _make_batch_session()
    try:
        session._batch_prev_emitted_end_audio = 29.68
        session._batch_emitted_tail_norm = ["seeing", "in", "front", "of", "me"]

        # Khối mới bắt đầu 28.68s; "in"/"front" nằm trong vùng chồng lấn, "of" bị align muộn
        # (29.70 > ranh giới) nên lớp-theo-mốc để lọt.
        raw = [
            AlignedWord("in", 0.10, 0.45),
            AlignedWord("front", 0.45, 0.85),
            AlignedWord("of", 1.02, 1.30),
            AlignedWord("I", 1.45, 1.65),
            AlignedWord("promise", 1.65, 2.10),
        ]
        kept, dropped_ts, dropped_seq = session._trim_batch_leading_overlap(
            SimpleNamespace(pts_start=28.68), raw
        )
        assert dropped_ts == 2, "lớp theo mốc phải bỏ 'in' và 'front'"
        assert dropped_seq == 1, "lớp theo chuỗi phải bỏ nốt 'of' đã phát ở khối trước"
        assert [w.text for w in kept] == ["I", "promise"]

        # Không có chồng lấn ⇒ không được bỏ gì
        kept2, ts2, seq2 = session._trim_batch_leading_overlap(
            SimpleNamespace(pts_start=30.5), raw
        )
        assert (ts2, seq2) == (0, 0)
        assert len(kept2) == len(raw)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_seek_reset_is_not_overwritten_by_inflight_block():
    """Iteration đang chạy KHÔNG được ghi đè mốc tua — race thật trong log 2026-10-02.

    Log: `Seek … mốc=34.99s` rồi 93 ms sau `ready_ahead=7.42s` (ready_until = 42.41 = cuối khối
    CŨ) và khối kế tiếp chạy từ `42.41s` ⇒ vùng `[34.99s, 42.41s]` **không bao giờ** được phiên âm
    ("seek đến vùng xám: lúc hiện phụ đề lúc không").
    """
    session = _make_batch_session()
    try:
        # Trạng thái trước khi tua: đã xử lý tới 42.41s
        session._batch_from_pts = 42.41
        session._ready_until_pts = 42.41
        session._ready_until_seq = session._seek_seq
        stale_seq = session._seek_seq

        await session.handle_seek("seek_1", 34.99)
        assert session._batch_from_pts == 34.99
        assert session._ready_until_pts == 34.99
        assert session._seek_seq == stale_seq + 1

        # Iteration CŨ (bắt đầu trước khi tua) quay lại ghi tiến độ ⇒ PHẢI bị bỏ qua
        assert session._commit_batch_progress(stale_seq, 42.41, 42.41) is False
        assert session._batch_from_pts == 34.99, "mốc tua bị ghi đè ⇒ mất vùng audio vừa tua tới"
        assert session._ready_until_pts == 34.99

        # Iteration của THẾ HỆ MỚI thì ghi bình thường
        assert session._commit_batch_progress(session._seek_seq, 38.5, 38.5) is True
        assert session._batch_from_pts == 38.5
        assert session._ready_until_pts == 38.5
        assert session._ready_until_seq == session._seek_seq
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_status_does_not_claim_coverage_of_old_position_after_seek():
    """`ready_until_pts` phải thuộc ĐÚNG thế hệ seek — marker vị trí cũ không được báo 'đã dịch'.

    Sự cố 2026-10-02: sau khi tua ngược về 16.89s, status báo `Đã dịch: 232.9s` trong khi playhead
    ở 27s ⇒ client tưởng đã sẵn sàng, phát video qua vùng chưa xử lý, không có phụ đề.
    """
    session = _make_batch_session()
    conn: MockSafeConnection = session.connection  # type: ignore
    try:
        session._ready_until_pts = 260.0  # marker của vị trí CŨ
        session._ready_until_seq = session._seek_seq
        session.timeline.append(0.0, make_speech_pcm(30.0))

        await session.handle_seek("seek_back", 16.89)
        session.current_time = 27.0  # playhead đi tiếp sau khi tua ngược

        # Giả lập một writer CŨ ghi marker của vị trí cũ mà KHÔNG cập nhật thế hệ — đúng hình dạng
        # của race trước khi vá. Marker 260s TUYỆT ĐỐI không được lọt ra ngoài.
        session._ready_until_pts = 260.0
        session._ready_until_seq = session._seek_seq - 1

        await session.send_status()
        st = conn.of_type("lookahead_status")[-1]
        assert st["ready_until_pts"] == pytest.approx(27.0), "marker cũ 260s bị rò ra ngoài"
        assert st["ready_ahead"] == 0.0
        assert st["prebuffer_ready"] is False, "không được báo sẵn sàng cho vùng chưa xử lý"

        # Sau khi thế hệ mới xử lý xong một khối thì marker mới có giá trị
        session._commit_batch_progress(session._seek_seq, 33.0, 33.0)
        await session.send_status()
        st2 = conn.of_type("lookahead_status")[-1]
        assert st2["ready_until_pts"] == pytest.approx(33.0)
        assert st2["ready_ahead"] == pytest.approx(6.0)
        assert st2["prebuffer_ready"] is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_batch_frontier_reanchors_when_stuck_far_ahead():
    """Con trỏ khối kẹt xa vị trí phát (seek ngược) ⇒ NEO LẠI thay vì `continue` mãi mãi.

    Ràng buộc cứng: không bao giờ để video phát mà pipeline không xử lý gì. Log 2026-10-02: sau khi
    tua ngược về 16.89s, `Đã dịch: 232.9s` và 30 s tiếp theo KHÔNG có khối nào được xử lý.
    """
    import time as _time

    from backend.config import config

    session = _make_batch_session()
    try:
        session.current_time = 27.0
        session._batch_from_pts = 259.9
        max_ahead = session.current_time + session.lead_time + config.lookahead.feed_margin_sec

        # Lần đầu chỉ GHI NHẬN mốc bắt đầu kẹt (phải bền vững mới neo)
        assert session._maybe_reanchor_batch_frontier(259.9, max_ahead) is False
        assert session._batch_from_pts == 259.9
        assert session._frontier_stuck_since is not None

        # Kẹt bền vững ⇒ NEO LẠI về vị trí phát
        session._frontier_stuck_since = _time.perf_counter() - 5.0
        assert session._maybe_reanchor_batch_frontier(259.9, max_ahead) is True
        assert session._batch_from_pts == 27.0
        assert session._fast_bootstrap is True
        assert session._ready_until_pts == 27.0
        assert session._frontier_stuck_since is None

        # Frontier chỉ nhô hơn tầm nhìn trong phạm vi MỘT KHỐI (throttle bình thường) ⇒ KHÔNG neo
        session._frontier_stuck_since = _time.perf_counter() - 5.0
        assert session._maybe_reanchor_batch_frontier(max_ahead + 10.0, max_ahead) is False
        assert session._batch_from_pts == 27.0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_offline_batch_flushes_tail_when_playback_stopped():
    """Video đã dừng ở mép cuối vùng audio ⇒ phải xử lý nốt đoạn đuôi < min_window.

    Trước đây `is_stream_end` luôn là False nên đoạn đuôi ngắn hơn `batch_min_sec` (12s) KHÔNG
    bao giờ được phiên âm ⇒ mất phụ đề ở cuối video.
    """
    session = _make_batch_session()
    conn: MockSafeConnection = session.connection  # type: ignore
    session.is_paused = True
    session.timeline.append(0.0, make_speech_pcm(5.0))
    session._batch_from_pts = 0.0

    await session.start_tasks()
    try:
        subs = []
        for _ in range(60):
            subs = conn.of_type("lookahead_subtitles")
            if subs:
                break
            await asyncio.sleep(0.05)
        assert subs, "phải flush được đoạn đuôi ngắn khi trình phát đã dừng"
    finally:
        await session.close()
