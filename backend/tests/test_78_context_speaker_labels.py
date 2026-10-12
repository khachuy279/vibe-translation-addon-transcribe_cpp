"""test_78 — P0.3b: gắn nhãn NGƯỜI NÓI vào dòng NGỮ CẢNH (`[PREVIOUS_SOURCE]`/`[PREVIOUS_TRANSLATION]`).

BỐI CẢNH: ràng buộc cứng của Pipeline B yêu cầu *"Keep the SAME speaker and the SAME pronouns as
the reference context above"*, nhưng ngữ cảnh cũ không có nhãn nào ⇒ khi khối có nhiều người nói,
model không thể biết dòng ngữ cảnh nào là của `speaker 5`. Prompt thật của người dùng (2026-10-24)
có `[PREVIOUS_SOURCE]` gồm 3 dòng không nhãn, còn `[CURRENT_SOURCE]` có `speaker 2/4/5`.

Bộ test này khoá 4 hành vi:
  1. Bật cờ + P0.1 chạy ⇒ nhãn có ở CẢ HAI phía của dòng ngữ cảnh (`speaker 5: src -> speaker 5: tgt`).
  2. Tắt cờ, HOẶC P0.1 không chạy (`context_sec = 0`) ⇒ định dạng cũ, KHÔNG nhãn (nhãn không bền thì
     gắn vào là chủ động gây sai).
  3. Mẩu ngữ cảnh TRÙNG với khối sắp dịch (cùng chữ + sát thời gian) bị loại; câu lặp ở đoạn khác
     của video (cách xa thời gian) được GIỮ.
  4. `parse_context_history()` của prompt Gemma vẫn tách đúng cặp khi có tiền tố nhãn.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from backend.asr.forced_aligner import AlignedWord, SubtitleSentence
from backend.config import config
from backend.core.dedup import normalize_for_dedup
from backend.translation.prompts_gemma import parse_context_history
from backend.ws.lookahead_handler import LookaheadSessionState


def _session() -> LookaheadSessionState:
    session = LookaheadSessionState(ws=SimpleNamespace(send_json=None))
    session.apply_init({"source_lang": "ja", "target_lang": "vi"})
    return session


def _utt(text: str, translated: str, pts: float, speaker: Optional[str] = None) -> Dict[str, Any]:
    return {
        "pts_start": pts,
        "pts_end": pts + 1.0,
        "text": text,
        "translated": translated,
        "speaker": speaker,
    }


# ────────────────────────────────────────────────────────────────── 1) GẮN NHÃN


def test_context_gan_nhan_hai_phia_khi_bat(monkeypatch):
    """Bật cờ + P0.1 chạy ⇒ mỗi dòng ngữ cảnh mang nhãn ở CẢ nguồn lẫn bản dịch."""
    monkeypatch.setattr(config.translation, "context_speaker_tags", True, raising=False)
    monkeypatch.setattr(config.diarization, "enabled", True, raising=False)
    monkeypatch.setattr(config.diarization, "context_sec", 8.0, raising=False)
    session = _session()
    session._recent_utterances = [
        _utt("えあるよね。", "Hả, có rồi nhỉ.", 100.0, "speaker 5"),
        _utt("ある？", "Có rồi?", 103.0, "speaker 4"),
    ]

    ctx = session._build_batch_context([(200.0, 202.0)], ["ね？"])

    assert "speaker 5: えあるよね。 -> speaker 5: Hả, có rồi nhỉ." in ctx
    assert "speaker 4: ある？ -> speaker 4: Có rồi?" in ctx


def test_context_khong_gan_nhan_khi_co_tat(monkeypatch):
    """Cờ TẮT (mặc định) ⇒ giữ nguyên định dạng cũ, không một chữ 'speaker' nào."""
    monkeypatch.setattr(config.translation, "context_speaker_tags", False, raising=False)
    session = _session()
    session._recent_utterances = [_utt("えあるよね。", "Hả, có rồi nhỉ.", 100.0, "speaker 5")]

    ctx = session._build_batch_context([(200.0, 202.0)], ["ね？"])

    assert ctx == "えあるよね。 -> Hả, có rồi nhỉ."
    assert "speaker" not in ctx


def test_context_khong_gan_nhan_khi_p01_tat(monkeypatch):
    """Cờ BẬT nhưng `context_sec = 0` (P0.1 tắt) ⇒ KHÔNG gắn nhãn: nhãn đánh lại theo khối là SAI người."""
    monkeypatch.setattr(config.translation, "context_speaker_tags", True, raising=False)
    monkeypatch.setattr(config.diarization, "context_sec", 0.0, raising=False)
    session = _session()
    session._recent_utterances = [_utt("えあるよね。", "Hả, có rồi nhỉ.", 100.0, "speaker 5")]

    ctx = session._build_batch_context([(200.0, 202.0)], ["ね？"])

    assert "speaker" not in ctx


def test_context_khong_gan_nhan_khi_khong_biet_nguoi_noi(monkeypatch):
    """Mẩu không có nhãn (diarization lỗi/tắt ở khối đó) ⇒ dòng đó giữ định dạng cũ."""
    monkeypatch.setattr(config.translation, "context_speaker_tags", True, raising=False)
    monkeypatch.setattr(config.diarization, "context_sec", 8.0, raising=False)
    session = _session()
    session._recent_utterances = [
        _utt("ああいや。", "À không.", 100.0, None),
        _utt("さすがに。", "Thật thì không.", 101.0, "speaker 5"),
    ]

    ctx = session._build_batch_context([(200.0, 202.0)], ["ね？"])
    lines = ctx.splitlines()

    assert lines == ["ああいや。 -> À không.", "speaker 5: さすがに。 -> speaker 5: Thật thì không."]


# ────────────────────────────────────────────────────────────────── 2) LỌC TRÙNG


def test_context_bo_manh_lap_that_su(monkeypatch):
    """Cùng chữ + sát mốc thời gian ⇒ là MỘT mẩu bị lặp ở ranh giới khối ⇒ bỏ khỏi ngữ cảnh."""
    monkeypatch.setattr(config.translation, "context_speaker_tags", True, raising=False)
    session = _session()
    session._recent_utterances = [
        _utt("えあるよね。", "Hả, có rồi nhỉ.", 100.0, "speaker 5"),
        _utt("ある？", "Có rồi?", 120.5, "speaker 2"),  # TRÙNG với mục 1 của khối sắp dịch
    ]

    ctx = session._build_batch_context([(120.0, 122.0)], ["ある？", "ね？"])

    assert "ある？" not in ctx
    assert "えあるよね。" in ctx


def test_context_giu_cau_lap_o_thoi_diem_khac(monkeypatch):
    """Cùng chữ nhưng ở đoạn KHÁC của video ⇒ hội thoại thật, phải giữ trong ngữ cảnh."""
    monkeypatch.setattr(config.translation, "context_speaker_tags", True, raising=False)
    session = _session()
    session._recent_utterances = [_utt("ある？", "Có rồi?", 100.0, "speaker 2")]

    ctx = session._build_batch_context([(300.0, 302.0)], ["ある？", "ね？"])

    assert "ある？" in ctx, "câu lặp ở đoạn khác không được coi là trùng"


def test_context_ton_trong_context_window(monkeypatch):
    """Chỉ lấy `context_window` dòng gần nhất (sau khi lọc trùng)."""
    monkeypatch.setattr(config.translation, "context_window", 2, raising=False)
    monkeypatch.setattr(config.translation, "context_speaker_tags", False, raising=False)
    session = _session()
    session._recent_utterances = [
        _utt(f"câu {i}", f"trans {i}", 100.0 + i, "speaker 1") for i in range(5)
    ]

    ctx = session._build_batch_context([(500.0, 502.0)], ["mới"])

    assert ctx.splitlines() == ["câu 3 -> trans 3", "câu 4 -> trans 4"]


# ────────────────────────────────────────────────────────────────── 3) TƯƠNG THÍCH PROMPT


def test_parse_context_history_voi_nhan_speaker():
    """Prompt Gemma phải tách đúng cặp và giữ 2 khối SONG SONG khi dòng có tiền tố nhãn."""
    context = (
        "speaker 5: えあるよね。 -> speaker 5: Hả, có rồi nhỉ.\n"
        "speaker 4: ある？ -> speaker 4: Có rồi?"
    )

    prev_src, prev_tgt = parse_context_history(context)

    assert prev_src.splitlines() == [
        "speaker 5: えあるよね。",
        "speaker 4: ある？",
    ]
    assert prev_tgt.splitlines() == [
        "speaker 5: Hả, có rồi nhỉ.",
        "speaker 4: Có rồi?",
    ]
    assert len(prev_src.splitlines()) == len(prev_tgt.splitlines()), "hai khối phải cùng số dòng"


def test_parse_context_history_tron_lan_co_nhan_va_khong_nhan():
    """Dòng có nhãn và dòng không nhãn trộn lẫn vẫn phải tách đúng cặp (không mất dòng nào)."""
    context = "ああいや。 -> À không.\nspeaker 5: さすがに。 -> speaker 5: Thật thì không."

    prev_src, prev_tgt = parse_context_history(context)

    assert len(prev_src.splitlines()) == 2
    assert len(prev_tgt.splitlines()) == 2
    assert prev_tgt.splitlines()[0] == "À không."


# ────────────────────────────────────────────────────────────────── 4) ĐẦU-CUỐI: VÒNG BATCH


class _Conn:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        self.sent.append(payload)
        return True

    def items(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in self.sent:
            if m.get("type") == "lookahead_subtitles":
                out.extend(m.get("items", []))
        return out


class _RecTranslator:
    """Ghi lại context mỗi lượt dịch gộp."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
        speaker_tags: Optional[List[str]] = None,
        overlap_flags: Optional[List[bool]] = None,
    ) -> List[str]:
        self.calls.append({"texts": list(texts), "context": context, "speaker_tags": speaker_tags})
        return [f"[vi] {t}" for t in texts]


class _GridAligner:
    """Một từ mỗi 0,4 s, gom 4 từ thành một phụ đề (giống `GridForcedAligner` của test_58)."""

    def __init__(self, word_sec: float = 0.4, group: int = 4):
        self.word_sec = word_sec
        self.group = group

    def prewarm(self) -> None:
        pass

    def align(self, audio: np.ndarray, text: str, language: str = "English") -> List[AlignedWord]:
        dur = len(audio) / 16000.0
        count = max(1, int(dur / self.word_sec))
        return [
            AlignedWord(f"w{i}", round(i * self.word_sec, 3), round((i + 1) * self.word_sec, 3))
            for i in range(count)
        ]

    def group_words_to_subtitles(
        self, words: List[AlignedWord], language: str = "English", **kwargs: Any
    ) -> List[SubtitleSentence]:
        subs = []
        for i in range(0, len(words), self.group):
            g = words[i : i + self.group]
            if g:
                subs.append(
                    SubtitleSentence(
                        text=" ".join(w.text for w in g),
                        start_time=g[0].start_time,
                        end_time=g[-1].end_time,
                        # PHẢI giữ `words` như aligner THẬT: P0.2 cần mốc từng từ để cắt câu theo
                        # người nói, và cổng chi phí `total_words < 2` cũng đọc trường này.
                        words=list(g),
                    )
                )
        return subs


@pytest.mark.asyncio
async def test_vong_batch_luu_speaker_va_ngu_canh_khoi_sau_co_nhan(monkeypatch):
    """ĐẦU-CUỐI: khối 2 phải nhận ngữ cảnh CÓ nhãn người nói của khối 1.

    Đây là mắt xích trước đây bị đứt: `_recent_utterances` không lưu speaker nên dù bật cờ thì
    ngữ cảnh vẫn ra dòng không nhãn.
    """
    from backend.diarization import DiarizationService
    from backend.tests.fakes import FakeInferenceEngine, FakeVADEngine, make_speech_pcm

    monkeypatch.setattr(config.translation, "context_speaker_tags", True, raising=False)
    monkeypatch.setattr(config.translation, "context_window", 3, raising=False)
    monkeypatch.setattr(config.diarization, "enabled", True, raising=False)
    monkeypatch.setattr(config.diarization, "context_sec", 8.0, raising=False)

    async def _fake_diarize(_cls, audio, sample_rate: int = 16000):
        # Cả cửa sổ là MỘT người ⇒ mọi mẩu nhận "speaker 1" (đủ để kiểm tra mắt xích dữ liệu).
        return [
            {
                "start_sample": 0,
                "end_sample": int(np.asarray(audio).size),
                "speaker_id": "speaker_0",
                "confidence": 0.9,
            }
        ]

    monkeypatch.setattr(DiarizationService, "diarize_chunk", classmethod(_fake_diarize))

    conn = _Conn()
    translator = _RecTranslator()
    session = LookaheadSessionState(
        ws=conn,
        asr_engine=FakeInferenceEngine(text_fn=lambda n: "alpha beta gamma delta epsilon zeta"),
        translation_engine=translator,
    )
    session.apply_init({"source_lang": "en", "target_lang": "vi"})
    session.init_components(vad_engine_override=FakeVADEngine())
    session._aligner_service = _GridAligner()
    session.timeline.append(0.0, make_speech_pcm(60.0))
    session._batch_from_pts = 0.0

    await session.start_tasks()
    try:
        for _ in range(400):
            session._ingest_event.set()
            # Chờ tới khi khối THỨ HAI được dịch (nếu chỉ chờ khối đầu thì không kiểm tra được gì).
            if len(translator.calls) >= 2 and session._batch_from_pts and session._batch_from_pts > 25.0:
                break
            await asyncio.sleep(0.05)
    finally:
        await session.close()

    assert len(translator.calls) >= 2, f"phải có ít nhất 2 khối, thực tế: {len(translator.calls)}"
    # Mọi mẩu đã phát phải được LƯU nhãn người nói.
    assert session._recent_utterances, "phải có lịch sử phát"
    assert all(u.get("speaker") == "speaker 1" for u in session._recent_utterances), (
        f"thiếu nhãn trong lịch sử: {session._recent_utterances[:3]}"
    )
    # Khối sau phải nhận ngữ cảnh có nhãn ở cả hai phía.
    later = [c for c in translator.calls[1:] if c["context"]]
    assert later, f"khối sau phải có ngữ cảnh, thực tế: {translator.calls}"
    ctx = later[-1]["context"]
    assert "speaker 1:" in ctx, f"ngữ cảnh phải mang nhãn người nói, thực tế: {ctx!r}"
    assert f"speaker 1: " in ctx and " -> speaker 1: " in ctx, "nhãn phải có ở CẢ HAI phía"
