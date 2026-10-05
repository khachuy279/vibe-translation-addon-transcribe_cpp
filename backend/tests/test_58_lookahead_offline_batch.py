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
from backend.config import config
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
        self.batch_calls: List[List[str]] = []

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "auto",
        target_lang: str = "vi",
    ) -> List[str]:
        self.batch_calls.append(list(texts))
        for t in texts:
            self.calls.append(t)
        return [f"{self.prefix}{t}" for t in texts]

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

        # Kiểm tra Batch Context Translation ở translator
        trans: ContextAwareFakeTranslator = session.translation_engine  # type: ignore
        assert len(trans.batch_calls) >= 1
        assert len(trans.calls) >= 2

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
    # ASR giả trả văn bản ĐÚNG MỘT LẦN: ASR thật không phiên âm lại cùng một câu ở khối sau khi bộ
    # cắt khối đã tiến con trỏ (`next_read_pts`). Nếu trả mãi, cùng một câu sẽ được gửi lại ở mỗi
    # khối và test sẽ đo nhầm hiện tượng trùng lặp của chính bộ giả, không phải tầng ngắt câu.
    calls = {"n": 0}

    def _ja_once(_n: int) -> str:
        calls["n"] += 1
        return ja_text if calls["n"] == 1 else ""

    session = LookaheadSessionState(
        ws=conn,
        asr_engine=FakeInferenceEngine(text_fn=_ja_once),
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
async def test_vad_silence_cut_replaces_boundary_patching():
    """Bộ cắt khối cắt tại khoảng lặng do VAD xác nhận ⇒ KHÔNG cần hàn gắn ranh giới nữa.

    Đây là cơ chế THAY THẾ cho bộ ba đã bị XOÁ ngày 2026-10-04 (trừ chồng lấn ranh giới + mang mảnh
    cuối sang khối sau + hiệu chỉnh mốc từ). Bộ ba đó chỉ tồn tại để bù cho việc cắt khối ở giữa
    câu, và chính nó tạo ra lỗi mới — log thật 15:23:02:

        [SEG_BATCH] Ghép 11 từ giữ lại từ khối trước vào khối [74.27 -> 86.37]
        [SEG_BATCH] "I can't see me" | "loving nobody but you for all my This is Rebecca." | ...

    Nguyên tắc mới: **tin [ASR] tuyệt đối** — ASR tạo câu nào thì gửi đúng câu đó; Forced Aligner chỉ
    xác định mốc bắt đầu/kết thúc.
    """
    session = _make_batch_session()
    try:
        # Mọi trạng thái/thuật toán "sửa chữa ranh giới" phải KHÔNG còn tồn tại.
        for attr in (
            "_batch_carry_words",
            "_trim_batch_leading_overlap",
            "_trim_repeated_block_prefix",
            "_hold_back_unfinished_tail",
            "_batch_trim_overlap",
        ):
            assert not hasattr(session, attr), f"{attr} phải bị xoá khỏi LookaheadSessionState"

        # Bộ cắt khối dùng bộ dò khoảng lặng VAD và KHÔNG lấy lùi (overlap_sec = 0).
        assert session.chunker.silence_scanner is session.silence_scanner
        assert session.chunker.overlap_sec == 0.0
        assert float(config.lookahead.batch_vad_silence_ms) >= 1000.0
        assert bool(config.lookahead.batch_use_vad_silence) is True
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


@pytest.mark.asyncio
async def test_batch_gate_waits_for_full_block_unless_urgent():
    """CỔNG KIÊN NHẪN (sự cố 2026-10-05, phiên `29d8eb1a`).

    Log thật: `[CHUNKER] from=39.10s frontier=47.11s available=8.02s` ⇒ khối 8 s `forced_overlap`
    trong khi playhead mới ở 8.8s (đã dịch trước 30 s). Vòng lặp phải CHỜ cho tới khi có đủ
    `batch_ready_sec` audio, hoặc tới khi playhead tiến sát phần đã dịch.
    """
    session = _make_batch_session()
    try:
        session._fast_bootstrap = False
        session._batch_proc_ema_sec = 1.0
        session.timeline.append(0.0, make_speech_pcm(47.11))
        want = min(float(config.lookahead.batch_ready_sec), float(session.chunker.max_audio_sec))
        urgent_lead = float(config.lookahead.batch_urgent_lead_sec) + 2.0

        # Đúng hình dạng log thật: 8 s audio phía trước con trỏ, đã dịch trước playhead ~30 s.
        session.current_time = 8.8
        go, reason, urgent = session._batch_gate(39.10, stream_end=False)
        assert (go, reason) == (False, "wait"), "không được cắt khối 8 s khi còn dư lookahead"

        # Playhead tiến sát phần đã dịch ⇒ PHẢI chạy ngay (khẩn cấp) dù chưa đủ khối dài.
        session.current_time = 39.10 - urgent_lead + 0.1
        go, reason, urgent = session._batch_gate(39.10, stream_end=False)
        assert (go, reason, urgent) == (True, "urgent", True)

        # Đủ một khối dài phía trước con trỏ ⇒ cắt ngay, không cần chờ khẩn cấp.
        session.current_time = 0.0
        go, reason, _ = session._batch_gate(47.11 - want, stream_end=False)
        assert (go, reason) == (True, "full")

        # Bootstrap sau khi tua và hết luồng luôn được đi.
        session._fast_bootstrap = True
        assert session._batch_gate(39.10, stream_end=False)[0] is True
        session._fast_bootstrap = False
        assert session._batch_gate(39.10, stream_end=True)[0] is True
    finally:
        await session.close()


def test_chunker_urgent_takes_short_tail_instead_of_waiting():
    """Khẩn cấp + bộ đệm mỏng (< `min_window_sec`) ⇒ lấy trọn phần đang có thay vì chờ."""
    session = LookaheadSessionState(ws=MockSafeConnection(), asr_engine=None, translation_engine=None)
    chunker = session.chunker
    session.timeline.append(0.0, make_speech_pcm(5.0))
    assert chunker.next_chunk(from_pts=0.0) is None
    chunk = chunker.next_chunk(from_pts=0.0, urgent=True)
    assert chunk is not None
    assert chunk.fallback_mode == "urgent_tail"
    assert chunk.duration == pytest.approx(5.0, abs=0.05)


def test_split_subtitles_by_clause_comma():
    """Kiểm tra ngắt câu phụ đề tại dấu phẩy (、 hoặc ,) khi số từ/token phía trước >= 4."""
    # 1. Tiếng Nhật: 'まあそれは、うん実は好きな人とか、まあ、まあ何。'
    words = [
        AlignedWord("まあ", 0.0, 0.5),
        AlignedWord("それ", 0.5, 1.0),
        AlignedWord("は、", 1.0, 1.5),
        AlignedWord("うん", 3.0, 3.5),
        AlignedWord("実は", 3.5, 4.2),
        AlignedWord("好き", 4.2, 4.8),
        AlignedWord("な", 4.8, 5.0),
        AlignedWord("人", 5.0, 5.5),
        AlignedWord("とか、", 5.5, 6.2),
        AlignedWord("まあ、", 10.0, 11.0),
        AlignedWord("まあ", 12.0, 12.5),
        AlignedWord("何。", 12.5, 13.0),
    ]
    sub = SubtitleSentence(
        text="まあそれは、うん実は好きな人とか、まあ、まあ何。",
        start_time=0.0,
        end_time=13.0,
        words=words,
    )
    splits = ForcedAlignerService.split_subtitles_by_clause_comma([sub], language="Japanese", min_words=4)
    assert len(splits) == 3, f"Phải tách thành 3 câu nhỏ: {[s.text for s in splits]}"
    assert splits[0].text == "まあそれは、"
    assert splits[0].start_time == 0.0
    assert splits[0].end_time == 1.5
    assert splits[1].text == "うん実は好きな人とか、"
    assert splits[1].start_time == 3.0
    assert splits[1].end_time == 6.2
    assert splits[2].text == "まあ、まあ何。"
    assert splits[2].start_time == 10.0
    assert splits[2].end_time == 13.0

    # 2. Tiếng Anh: 'Because I was tired and hungry, I decided to go home.'
    en_words = [
        AlignedWord("Because", 0.0, 0.5),
        AlignedWord("I", 0.5, 0.7),
        AlignedWord("was", 0.7, 0.9),
        AlignedWord("tired", 0.9, 1.4),
        AlignedWord("and", 1.4, 1.6),
        AlignedWord("hungry,", 1.6, 2.2),
        AlignedWord("I", 2.5, 2.7),
        AlignedWord("decided", 2.7, 3.1),
        AlignedWord("to", 3.1, 3.3),
        AlignedWord("go", 3.3, 3.6),
        AlignedWord("home.", 3.6, 4.0),
    ]
    en_sub = SubtitleSentence(
        text="Because I was tired and hungry, I decided to go home.",
        start_time=0.0,
        end_time=4.0,
        words=en_words,
    )
    en_splits = ForcedAlignerService.split_subtitles_by_clause_comma([en_sub], language="English", min_words=4)
    assert len(en_splits) == 2, f"Phải tách thành 2 câu: {[s.text for s in en_splits]}"
    assert en_splits[0].text == "Because I was tired and hungry,"
    assert en_splits[1].text == "I decided to go home."

    # 3. Câu ngắn < 4 từ thì KHÔNG tách
    short_words = [AlignedWord("Oh,", 0.0, 0.3), AlignedWord("no.", 0.4, 0.7)]
    short_sub = SubtitleSentence(text="Oh, no.", start_time=0.0, end_time=0.7, words=short_words)
    short_splits = ForcedAlignerService.split_subtitles_by_clause_comma([short_sub], language="English", min_words=4)
    assert len(short_splits) == 1
    assert short_splits[0].text == "Oh, no."


def test_decimal_numbers_not_split_in_forced_aligner():
    """Số thập phân (ví dụ '星3.9でさ') và số hàng nghìn (1,000) không bị chẻ thành câu mới."""
    text = "あ、トラベルダウで星3.9でさ、温泉がいいらしいよ。"
    raw_items = [
        AlignedWord("あ", 0.0, 0.3),
        AlignedWord("トラベルダウ", 0.3, 1.0),
        AlignedWord("で", 1.0, 1.2),
        AlignedWord("星", 1.2, 1.5),
        AlignedWord("3", 1.5, 1.8),
        AlignedWord("9", 1.8, 2.1),
        AlignedWord("でさ", 2.1, 2.5),
        AlignedWord("温泉", 2.8, 3.3),
        AlignedWord("が", 3.3, 3.5),
        AlignedWord("いい", 3.5, 3.8),
        AlignedWord("らしい", 3.8, 4.1),
        AlignedWord("よ", 4.1, 4.5),
    ]

    merged = ForcedAlignerService.merge_source_text(raw_items, text)
    # 3. và 9 phải được gộp thành 3.9
    texts = [m.text for m in merged]
    assert "3.9" in texts, f"Token số phải được gộp thành '3.9', thực tế: {texts}"

    # Gom câu không được chẻ tại '3.'
    subs = ForcedAlignerService.group_words_to_subtitles(merged, language="Japanese")
    sub_texts = [s.text for s in subs]
    assert len(subs) == 1, f"Toàn câu phải giữ nguyên 1 câu trọn vẹn, thực tế: {sub_texts}"
    assert "星3.9でさ" in subs[0].text

    # Khi qua split_subtitles_by_clause_comma, câu được tách tại 'でさ、' (không tách tại 3.9)
    clause_subs = ForcedAlignerService.split_subtitles_by_clause_comma(subs, language="Japanese", min_words=4)
    clause_texts = [s.text for s in clause_subs]
    assert clause_texts == [
        "あ、トラベルダウで星3.9でさ、",
        "温泉がいいらしいよ。",
    ], f"Tách vế câu chuẩn xác, thực tế: {clause_texts}"





# --- Hồi quy 2026-10-05 (av01.media): NGẮT CÂU TRƯỚC, chỉ bỏ CÂU kẹt vòng ---------------
# Log thật: ASR trả về '一時しないと収まんないよこれ。ちょっと入れるだけだから。ちょっと入れちょっとだから。
# ちょっとちょっと。あああああ…' — 4 câu THẬT ở đầu, CHỈ ĐUÔI kẹt vòng. Bản cũ gọi
# `is_repetition_hallucination(clean_text)` cho CẢ KHỐI nên vứt luôn 4 câu đúng.
#
# Hai test dưới đây chạy ĐÚNG vòng lặp batch (`_offline_batch_loop`) chứ không kiểm tra một hàm phụ
# trợ nào — để khoá đúng HÀNH VI người dùng thấy: câu thật phải ra phụ đề.

_MIXED_REPETITION_TEXT = (
    "一時しないと収まんないよこれ。ちょっと入れるだけだから。"
    "ちょっと入れちょっとだから。ちょっとちょっと。あああああああああああああああ"
)
_PURE_REPETITION_TEXT = "あ" * 30


def _make_batch_session_with_asr_text(text: str) -> LookaheadSessionState:
    """Phiên batch tối thiểu với ASR giả luôn trả về `text`, aligner kiểu tiếng Nhật."""
    conn = MockSafeConnection()
    asr = FakeInferenceEngine(text_fn=lambda n: text)
    translator = ContextAwareFakeTranslator()
    session = LookaheadSessionState(ws=conn, asr_engine=asr, translation_engine=translator)
    session.apply_init({"source_lang": "ja", "target_lang": "vi", "lead_time": 15})
    session.init_components(vad_engine_override=FakeVADEngine())
    session._aligner_service = UpstreamLikeAligner()   # tokenize kiểu tiếng Nhật
    return session


async def _run_one_batch_iteration(session: LookaheadSessionState, seconds: float = 20.0):
    """Nạp `seconds` audio rồi chờ vòng batch xử lý xong một lượt."""
    conn: MockSafeConnection = session.connection  # type: ignore
    await session.start_tasks()
    session.timeline.append(0.0, make_speech_pcm(seconds))
    session._ingest_event.set()
    for _ in range(40):
        if conn.of_type("lookahead_subtitles") or session.utterances_sent:
            break
        await asyncio.sleep(0.05)
    return conn


@pytest.mark.asyncio
async def test_mixed_repetition_tail_keeps_the_real_sentences():
    """Khối lẫn lộn (câu thật + đuôi kẹt vòng) ⇒ PHẢI ra phụ đề cho các câu thật."""
    session = _make_batch_session_with_asr_text(_MIXED_REPETITION_TEXT)
    try:
        conn = await _run_one_batch_iteration(session)
        subs = conn.of_type("lookahead_subtitles")
        assert subs, "Khối còn câu dùng được ⇒ KHÔNG được vứt cả khối"
        texts = [it["original_text"] for it in subs[0]["items"]]
        assert any("収まんない" in t for t in texts), f"câu thật phải được giữ, thực tế: {texts}"
        assert not any("あああああ" in t for t in texts), f"câu ảo giác phải bị bỏ: {texts}"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_pure_repetition_block_emits_nothing():
    """Khối rác thuần ⇒ không gửi phụ đề nào (nhưng đã đẩy con trỏ để không kẹt)."""
    session = _make_batch_session_with_asr_text(_PURE_REPETITION_TEXT)
    try:
        conn = await _run_one_batch_iteration(session)
        assert not conn.of_type("lookahead_subtitles"), "khối rác thuần không được ra phụ đề"
        assert session._batch_from_pts is not None and session._batch_from_pts > 0.0, (
            "con trỏ vẫn phải tiến để vòng lặp không kẹt ở khối rác"
        )
    finally:
        await session.close()
