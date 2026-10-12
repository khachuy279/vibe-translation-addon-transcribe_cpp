"""test_77 — P0.2 (tách câu tại ranh giới lượt nói) + P0.3 (cờ chồng tiếng).

BỐI CẢNH (đo thật 2026-10-24, `scratch/diar_stream_probe.py`, `scratch/diar_chunk_split.py`):
Nemotron-3-Diarization trả `turns` cho từng khối, nhưng bản cũ chỉ gán **MỘT nhãn cho MỘT câu**
theo overlap lớn nhất ⇒ câu vắt qua ranh giới đổi người nói nhận sai vai cho nửa sau, và vùng
hai người nói chồng tiếng hoàn toàn bị che (không có cờ nào cho prompt dịch biết).

Bộ test này khoá ba hành vi:
  1. Xen kẽ ⇒ CHẺ câu tại ranh giới lượt nói, mỗi mảnh mang nhãn riêng.
  2. Chồng tiếng ⇒ ĐÁNH CỜ nhưng **không chẻ** (text trong vùng đó là bản trộn, nhãn từng từ
     không đáng tin) — và phần chữ bị bỏ vào prompt chỉ áp cho ĐÚNG mục bị cờ.
  3. Model "nhấp nháy" nhãn (mảnh ngắn) ⇒ gộp lại, không tạo phụ đề vụn.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from backend.asr.forced_aligner import AlignedWord, SubtitleSentence
from backend.diarization import SpeakerRegistry, attribute_sentences, realign_attribution
from backend.diarization.attribution import SpeakerTurn, normalize_turns, overlap_seconds
from backend.tests.fakes import make_speech_pcm
from backend.translation.prompts import PipelineBPromptStrategy


# ────────────────────────────────────────────────────────────────── tiện ích dựng dữ liệu


def _words(n: int, sec_per_word: float = 0.4, start: float = 0.0) -> List[AlignedWord]:
    return [
        AlignedWord(
            text=f"w{i}",
            start_time=start + i * sec_per_word,
            end_time=start + (i + 1) * sec_per_word,
        )
        for i in range(n)
    ]


def _sentence(n_words: int, sec_per_word: float = 0.4, text: Optional[str] = None) -> SubtitleSentence:
    ws = _words(n_words, sec_per_word)
    return SubtitleSentence(
        text=text if text is not None else " ".join(w.text for w in ws),
        start_time=ws[0].start_time,
        end_time=ws[-1].end_time,
        words=ws,
    )


# ────────────────────────────────────────────────────────────────── P0.2 — TÁCH CÂU


def test_xen_ke_tach_cau_theo_nguoi_noi():
    """A nói 0–2,4 s rồi B nói 2,4–4,8 s trong CÙNG một câu ASR ⇒ phải thành HAI mảnh, hai nhãn."""
    sent = _sentence(12)
    turns = [SpeakerTurn(0.0, 2.4, "speaker_0"), SpeakerTurn(2.4, 4.8, "speaker_1")]

    pieces = attribute_sentences([sent], turns, language="English")

    assert len(pieces) == 2, f"phải chẻ tại ranh giới lượt nói, thực tế: {[p.text for p in pieces]}"
    assert pieces[0].speaker == "speaker 1" and pieces[1].speaker == "speaker 2"
    # Mốc thời gian lấy từ mốc TỪ (không chia theo tỉ lệ ký tự).
    assert pieces[0].start_time == pytest.approx(0.0)
    assert pieces[0].end_time == pytest.approx(2.4)
    assert pieces[1].start_time == pytest.approx(2.4)
    assert pieces[1].end_time == pytest.approx(4.8)
    # Không mất chữ nào.
    assert " ".join(p.text for p in pieces).split() == sent.text.split()
    assert all(not p.overlap for p in pieces)


def test_mot_nguoi_noi_thi_giu_nguyen_van_ban_goc():
    """Chỉ một người nói ⇒ KHÔNG chẻ, và phải giữ NGUYÊN văn bản gốc (kể cả dấu câu)."""
    ws = _words(6)
    sent = SubtitleSentence(
        text="Hello, world. How are you?",
        start_time=ws[0].start_time,
        end_time=ws[-1].end_time,
        words=ws,
    )
    turns = [SpeakerTurn(0.0, 5.0, "speaker_0")]

    pieces = attribute_sentences([sent], turns, language="English")

    assert len(pieces) == 1
    assert pieces[0].text == "Hello, world. How are you?"
    assert pieces[0].speaker == "speaker 1"
    assert pieces[0].overlap is False


def test_nhap_nhay_nhan_ngan_thi_gop_lai():
    """Model diarization đôi lúc chớp nhãn 0,3 s giữa một mạch nói của CÙNG một người.

    Ngưỡng `split_min_sec=0,7` phải gộp mảnh vụn đó ⇒ vẫn là MỘT phụ đề.
    """
    sent = _sentence(10)  # 4,0 s
    turns = [
        SpeakerTurn(0.0, 2.0, "speaker_0"),
        SpeakerTurn(2.0, 2.3, "speaker_1"),  # chớp nhãn 0,3 s
        SpeakerTurn(2.3, 4.0, "speaker_0"),
    ]

    pieces = attribute_sentences([sent], turns, language="English", split_min_sec=0.7)

    assert len(pieces) == 1, f"mảnh chớp 0,3 s không được tạo phụ đề riêng: {[p.text for p in pieces]}"


def test_tat_split_van_giu_goc_va_nhan():
    """`split=False` ⇒ giữ nguyên số câu nhưng VẪN gán nhãn (đường thoát khi cần tắt P0.2)."""
    sent = _sentence(12)
    turns = [SpeakerTurn(0.0, 2.4, "speaker_0"), SpeakerTurn(2.4, 4.8, "speaker_1")]

    pieces = attribute_sentences([sent], turns, language="English", split=False)

    assert len(pieces) == 1
    assert pieces[0].text == sent.text
    # Nhãn = người nói NHIỀU HƠN (hành vi cũ, giữ nguyên khi tắt P0.2).
    assert pieces[0].speaker == "speaker 1"


def test_khong_co_turns_thi_khong_nhan_khong_che():
    sent = _sentence(6)
    pieces = attribute_sentences([sent], [], language="English")
    assert len(pieces) == 1
    assert pieces[0].speaker is None
    assert pieces[0].overlap is False


# ────────────────────────────────────────────────────────────────── P0.3 — CHỒNG TIẾNG


def test_chong_tieng_danh_co_va_khong_che():
    """Hai người nói cùng lúc: nhãn từng từ không đáng tin ⇒ ĐÁNH CỜ, KHÔNG chẻ câu.

    Đây là bẫy thật: nếu chẻ theo nhãn dominant trong vùng chồng tiếng thì nửa câu (vốn là bản
    trộn của hai giọng) bị gán cho một người.
    """
    sent = _sentence(12)  # 4,8 s
    turns = [
        SpeakerTurn(0.0, 4.8, "speaker_0"),
        # B nói chồng từ 3,3 s ⇒ 1,5 s chồng lấn trong câu này
        SpeakerTurn(3.3, 4.8, "speaker_1"),
    ]

    pieces = attribute_sentences([sent], turns, language="English", overlap_min_sec=0.3)

    assert len(pieces) == 1, "vùng chồng tiếng KHÔNG được chẻ"
    assert pieces[0].overlap is True
    assert pieces[0].overlap_sec >= 1.4


def test_chong_tieng_nho_hon_nguong_thi_khong_danh_co():
    """Nhiễu ở mép lượt (0,2 s < ngưỡng 0,3 s) không được đánh cờ."""
    sent = _sentence(12)
    turns = [SpeakerTurn(0.0, 4.6, "speaker_0"), SpeakerTurn(4.6, 4.8, "speaker_1")]

    pieces = attribute_sentences([sent], turns, language="English", overlap_min_sec=0.3)

    assert all(not p.overlap for p in pieces)


def test_overlap_seconds_dung_o_ca_hai_mep():
    """Cửa sổ chồng lấn phải tính cả khi có 3 lượt gối nhau."""
    turns = [
        SpeakerTurn(0.0, 10.0, "speaker_0"),
        SpeakerTurn(2.0, 6.0, "speaker_1"),
        SpeakerTurn(4.0, 8.0, "speaker_2"),
    ]
    # 0–2: 1 người | 2–4: 2 người (2 s) | 4–6: 3 người (2 s) | 6–8: 2 người (2 s) | 8–10: 1 người
    assert overlap_seconds((0.0, 10.0), turns) == pytest.approx(6.0)
    assert overlap_seconds((0.0, 1.0), turns) == pytest.approx(0.0)


# ────────────────────────────────────────────────────────────────── GÁN LẠI THEO THỜI GIAN


def test_realign_theo_thoi_gian_sau_khi_bi_che_tiep():
    """Lớp `split_oversized_sentences` chạy sau có thể chẻ thêm ⇒ nhãn phải theo THỜI GIAN."""
    sent = _sentence(12)
    turns = [SpeakerTurn(0.0, 2.4, "speaker_0"), SpeakerTurn(2.4, 4.8, "speaker_1")]
    pieces = attribute_sentences([sent], turns, language="English")
    assert len(pieces) == 2

    # Giả lập mảnh thứ hai bị chẻ tiếp thành hai phụ đề nhỏ hơn.
    after = [
        SubtitleSentence(text="w0 w1 w2", start_time=0.0, end_time=1.2, words=[]),
        SubtitleSentence(text="w6 w7", start_time=2.4, end_time=3.2, words=[]),
        SubtitleSentence(text="w8 w9 w10 w11", start_time=3.2, end_time=4.8, words=[]),
    ]
    tags, overlaps = realign_attribution(after, pieces)

    assert tags == ["speaker 1", "speaker 2", "speaker 2"]
    assert overlaps == [False, False, False]


# ────────────────────────────────────────────────────────────────── PROMPT DỊCH (P0.3)


def test_prompt_chi_them_ghi_chu_cho_dung_muc_bi_co():
    sentences = ["a", "b", "c"]
    tags = ["speaker 1", "speaker 2", "speaker 1"]
    strat = PipelineBPromptStrategy()

    base = strat.build_batch_prompt(sentences, "en", "vi", speaker_tags=tags)
    with_flag = strat.build_batch_prompt(
        sentences, "en", "vi", speaker_tags=tags, overlap_flags=[False, True, False]
    )

    assert "Overlapping speech" not in base, "không có cờ ⇒ prompt phải y hệt bản cũ"
    assert "Overlapping speech" in with_flag
    assert "items 2" in with_flag, "phải nêu ĐÚNG chỉ số 1-based của mục bị cờ"
    assert "items 1" not in with_flag and "items 3" not in with_flag


def test_prompt_khong_doi_khi_co_toan_false():
    strat = PipelineBPromptStrategy()
    base = strat.build_batch_prompt(["a", "b"], "en", "vi")
    flagged = strat.build_batch_prompt(["a", "b"], "en", "vi", overlap_flags=[False, False])
    assert base == flagged


# ────────────────────────────────────────────────────────────────── CHUẨN HOÁ TURNS


def test_normalize_turns_doi_mau_sang_giay_va_bo_luot_rong():
    turns = [
        {"start_sample": 16000, "end_sample": 32000, "speaker_id": "speaker_0"},
        {"start_sample": 32000, "end_sample": 32000, "speaker_id": "speaker_1"},  # rỗng ⇒ bỏ
        {"start_sample": 48000, "end_sample": 64000, "speaker_id": ""},           # thiếu nhãn ⇒ bỏ
    ]
    out = normalize_turns(turns, sample_rate=16000)
    assert [(t.start_sec, t.end_sec, t.speaker_id) for t in out] == [(1.0, 2.0, "speaker_0")]


def test_turns_that_tu_model_nemotron_tach_dung_ba_luot():
    """Dữ liệu THẬT: `turns` do `audiocpp_cli --task diar` trả về cho file thí nghiệm
    `scratch/diar_probe/alternating.wav` (A nói 0–4,0 s, B nói 4,4–8,4 s, A nói lại 8,8–12,8 s):

        speaker_0 0.00-4.02s | speaker_1 5.65-8.14s | speaker_0 8.81-12.83s

    Một câu ASR trải suốt 12,8 s (32 từ × 0,4 s) phải được chẻ thành 3 phụ đề, nhãn 1 → 2 → 1 —
    tức người nói quay lại vẫn giữ ĐÚNG nhãn cũ trong cùng một lượt gọi (điều mà bản cũ làm được
    ở mức cả khối nhưng KHÔNG chẻ được câu).
    """
    turns = [
        {"start_sample": 0, "end_sample": 64320, "speaker_id": "speaker_0", "confidence": 0.99},
        {"start_sample": 90400, "end_sample": 130240, "speaker_id": "speaker_1", "confidence": 0.90},
        {"start_sample": 140960, "end_sample": 205280, "speaker_id": "speaker_0", "confidence": 0.98},
    ]
    sent = _sentence(32)  # 12,8 s
    assert sent.end_time == pytest.approx(12.8)

    pieces = attribute_sentences([sent], turns, language="English")

    assert len(pieces) == 3, f"thực tế: {[(p.speaker, round(p.start_time, 2), round(p.end_time, 2)) for p in pieces]}"
    assert [p.speaker for p in pieces] == ["speaker 1", "speaker 2", "speaker 1"]
    assert [p.raw_speaker for p in pieces] == ["speaker_0", "speaker_1", "speaker_0"]
    assert " ".join(p.text for p in pieces).split() == sent.text.split()

    # Ranh giới mảnh BÁM MỐC TỪ (0,4 s/từ), không bám mốc turn của model — vì cắt giữa từ sẽ làm
    # mất chữ. Từ nằm trong KHOẢNG LẶNG giữa hai lượt (4,02–5,65 s) được gán cho lượt GẦN NHẤT
    # (quy tắc cũ giữ nguyên), nên mảnh đầu phình tới 4,8 s chứ không dừng ở 4,0 s.
    assert [p.end_time for p in pieces] == pytest.approx([4.8, 8.4, 12.8])
    assert all(p.overlap is False for p in pieces)

    # Mỗi mảnh phải giữ `words` của mình để các lớp ngắt câu sau còn dùng.
    assert [len(p.words) for p in pieces] == [12, 9, 11]


# ────────────────────────────────────────────────────────────────── ĐẤU DÂY VÀO HANDLER


def _make_chunk(
    pts_start: float,
    duration: float,
    session: Any = None,
    *,
    feed_timeline: bool = True,
) -> Any:
    """Khối giả có đủ trường mà handler dùng (`pcm`, `pts_start`, `pts_end`, `duration`).

    `feed_timeline=True` ⇒ nạp luôn PCM vào timeline để `_diar_window_pcm()` lấy được ngữ cảnh.
    """
    pcm = make_speech_pcm(duration)
    if session is not None and feed_timeline:
        session.timeline.append(float(pts_start), pcm)
    return SimpleNamespace(
        pcm=pcm,
        pts_start=float(pts_start),
        pts_end=float(pts_start) + float(duration),
        duration=float(duration),
    )


def _fake_diarize_factory(per_call_turns: List[List[Dict[str, Any]]], lengths: List[int]):
    """Trả về hàm `diarize_chunk` giả: mỗi lượt gọi trả một kịch bản turns khác nhau."""

    async def _fake(_cls, audio, sample_rate: int = 16000):
        lengths.append(int(np.asarray(audio).size))
        idx = min(len(lengths) - 1, len(per_call_turns) - 1)
        return per_call_turns[idx]

    return _fake


def _turns(*spans: tuple) -> List[Dict[str, Any]]:
    """`_turns((0.0, 4.0, "speaker_0"), …)` → danh sách turns theo MẪU 16 kHz."""
    return [
        {
            "start_sample": int(a * 16000),
            "end_sample": int(b * 16000),
            "speaker_id": sid,
            "confidence": 0.9,
        }
        for a, b, sid in spans
    ]


class _RecTranslator:
    """Translator giả GHI LẠI cả `speaker_tags` và `overlap_flags` mà handler truyền xuống."""

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
        self.calls.append(
            {
                "texts": list(texts),
                "speaker_tags": list(speaker_tags) if speaker_tags else None,
                "overlap_flags": list(overlap_flags) if overlap_flags else None,
            }
        )
        return [f"[vi] {t}" for t in texts]


@pytest.mark.asyncio
async def test_handler_attribute_speakers_che_cau_va_gan_nhan(monkeypatch):
    """`_attribute_speakers()` (đấu dây thật của handler) phải chẻ câu và trả nhãn tương ứng."""
    from backend.diarization import DiarizationService, service as diar_service
    from backend.ws.lookahead_handler import LookaheadSessionState

    session = LookaheadSessionState(ws=SimpleNamespace(send_json=None))
    session.apply_init({"source_lang": "en", "target_lang": "vi"})

    async def _fake_diarize(_cls, audio, sample_rate=16000):
        # A nói 0–2,4 s, B nói 2,4–4,8 s (mẫu 16 kHz).
        return [
            {"start_sample": 0, "end_sample": int(2.4 * 16000), "speaker_id": "speaker_0", "confidence": 0.9},
            {"start_sample": int(2.4 * 16000), "end_sample": int(4.8 * 16000), "speaker_id": "speaker_1", "confidence": 0.9},
        ]

    monkeypatch.setattr(DiarizationService, "diarize_chunk", classmethod(_fake_diarize))

    subs = [_sentence(12)]
    chunk = _make_chunk(0.0, 5.0, session)
    new_subs, pieces = await session._attribute_speakers(subs, chunk)

    assert len(new_subs) == 2, "handler phải dùng bản ĐÃ CHẺ"
    assert [p.speaker for p in pieces] == ["speaker 1", "speaker 2"]
    assert new_subs[0].text.split() == ["w0", "w1", "w2", "w3", "w4", "w5"]
    assert new_subs[1].text.split() == ["w6", "w7", "w8", "w9", "w10", "w11"]
    assert new_subs[0].words, "phải giữ mốc từ để các lớp sau còn dùng"


@pytest.mark.asyncio
async def test_handler_bo_qua_diarization_khi_tat(monkeypatch):
    """`diarization.enabled=False` ⇒ handler trả nguyên câu, không gọi model."""
    from backend.config import config
    from backend.diarization import DiarizationService
    from backend.ws.lookahead_handler import LookaheadSessionState

    called = {"n": 0}

    async def _fake_diarize(_cls, audio, sample_rate=16000):
        called["n"] += 1
        return []

    monkeypatch.setattr(DiarizationService, "diarize_chunk", classmethod(_fake_diarize))
    monkeypatch.setattr(config.diarization, "enabled", False, raising=False)

    session = LookaheadSessionState(ws=SimpleNamespace(send_json=None))
    session.apply_init({"source_lang": "en", "target_lang": "vi"})
    subs = [_sentence(12)]
    chunk = SimpleNamespace(pcm=np.zeros(16000, dtype=np.float32), pts_start=0.0)

    new_subs, pieces = await session._attribute_speakers(subs, chunk)

    assert pieces == []
    assert new_subs is subs
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_handler_dua_duoc_muc_chong_tieng_vao_translator(monkeypatch):
    """Đường dịch phải nhận được cờ chồng tiếng (P0.3) — nếu không, prompt không bao giờ được bật."""
    from backend.diarization import DiarizationService
    from backend.ws.lookahead_handler import LookaheadSessionState

    session = LookaheadSessionState(ws=SimpleNamespace(send_json=None))
    session.apply_init({"source_lang": "en", "target_lang": "vi"})
    rec = _RecTranslator()
    session.translation_engine = rec

    async def _fake_diarize(_cls, audio, sample_rate=16000):
        return [
            {"start_sample": 0, "end_sample": int(4.8 * 16000), "speaker_id": "speaker_0", "confidence": 0.9},
            {"start_sample": int(3.3 * 16000), "end_sample": int(4.8 * 16000), "speaker_id": "speaker_1", "confidence": 0.9},
        ]

    monkeypatch.setattr(DiarizationService, "diarize_chunk", classmethod(_fake_diarize))

    subs = [_sentence(12)]
    chunk = _make_chunk(0.0, 5.0, session)
    new_subs, pieces = await session._attribute_speakers(subs, chunk)
    tags, overlaps = realign_attribution(new_subs, pieces)

    await session._translate_batch(
        [s.text for s in new_subs], speaker_tags=tags or None, overlap_flags=overlaps or None
    )

    assert rec.calls, "translator phải được gọi"
    assert rec.calls[0]["overlap_flags"] == [True]
    assert rec.calls[0]["speaker_tags"] == ["speaker 1"]


# ────────────────────────────────────────────────────────────────── P0.1 — DANH TÍNH THEO PHIÊN


def test_registry_giu_nhan_khi_model_doi_so():
    """Cùng một người nhưng model đổi id thô giữa hai cửa sổ ⇒ nhãn PHIÊN phải giữ nguyên."""
    reg = SpeakerRegistry()
    # Cửa sổ 1 (0–12 s): A = speaker_0
    turns1 = [SpeakerTurn(0.0, 12.0, "speaker_0")]
    labels1 = reg.label_window_turns(turns1, window_start_pts=0.0)
    reg.observe(turns1, window_start_pts=0.0, labels=labels1, already_covered_until_pts=0.0)
    assert labels1 == {"speaker_0": "speaker 1"}

    # Cửa sổ 2 (4–24 s): model đổi số — A (đang nói tới 12,5 s) thành speaker_1, B thành speaker_0
    turns2 = [SpeakerTurn(0.0, 8.5, "speaker_1"), SpeakerTurn(8.5, 20.0, "speaker_0")]
    labels2 = reg.label_window_turns(turns2, window_start_pts=4.0)

    assert labels2["speaker_1"] == "speaker 1", "A phải giữ ĐÚNG nhãn cũ dù id thô đã đổi"
    assert labels2["speaker_0"] == "speaker 2", "B là người mới ⇒ nhãn mới"


def test_registry_khong_khop_khi_chong_lan_duoi_nguong():
    """Chồng lấn nhỏ hơn `min_match_sec` là nhiễu ở mép lượt, không phải bằng chứng cùng người."""
    reg = SpeakerRegistry(min_match_sec=0.5)
    turns1 = [SpeakerTurn(0.0, 2.0, "speaker_0")]
    labels1 = reg.label_window_turns(turns1, window_start_pts=0.0)
    reg.observe(turns1, window_start_pts=0.0, labels=labels1)

    # Cửa sổ sau chỉ chồng 0,2 s với lượt cũ
    labels2 = reg.label_window_turns([SpeakerTurn(0.0, 0.2, "speaker_7")], window_start_pts=1.8)
    assert labels2["speaker_7"] == "speaker 2", "0,2 s < 0,5 s ⇒ phải coi là người mới"


def test_registry_luat_lien_tuc_tai_moc_khoi_du_lieu_that():
    """DỮ LIỆU THẬT (`scratch/p01_real_probe.py`, model Nemotron-3 trên `alternating.wav`).

    Khối 1 [0–6 s]: `speaker_0` = A 0–4,02 s | `speaker_1` = B 5,67–6,00 s
    Khối 2, cửa sổ [0–13,2 s] (6 s ngữ cảnh): `speaker_0` = A 0–4,02 + 8,81–12,83 | `speaker_1` = B 5,65–8,14

    B chỉ kịp để lại **0,33 s** trước mốc khối — DƯỚI `min_match_sec` 0,35 s. Không có luật liên tục
    thì B bị cấp số mới ("speaker 3") ngay giữa video; đây là lỗi THẬT đã gặp khi chạy model.
    """
    reg = SpeakerRegistry()
    turns1 = _turns((0.0, 4.02, "speaker_0"), (5.67, 6.00, "speaker_1"))
    labels1 = reg.label_window_turns(turns1, window_start_pts=0.0, boundary_pts=0.0)
    reg.observe(turns1, window_start_pts=0.0, labels=labels1, already_covered_until_pts=0.0)
    assert labels1 == {"speaker_0": "speaker 1", "speaker_1": "speaker 2"}

    turns2 = _turns(
        (0.0, 4.02, "speaker_0"),
        (5.65, 8.14, "speaker_1"),
        (8.81, 12.83, "speaker_0"),
    )
    labels2 = reg.label_window_turns(turns2, window_start_pts=0.0, boundary_pts=6.0)

    assert labels2 == {"speaker_0": "speaker 1", "speaker_1": "speaker 2"}, (
        f"B phải giữ 'speaker 2' nhờ luật liên tục tại mốc khối, thực tế: {labels2}"
    )


def test_registry_luat_lien_tuc_khong_ap_khi_luot_cu_da_ket_thuc_truoc_moc():
    """Lượt cũ KHÔNG dính mốc khối (kết thúc từ lâu) ⇒ chồng lấn nhỏ không đủ để nối danh tính."""
    reg = SpeakerRegistry()
    turns1 = _turns((0.0, 4.0, "speaker_0"))
    labels1 = reg.label_window_turns(turns1, window_start_pts=0.0, boundary_pts=0.0)
    reg.observe(turns1, window_start_pts=0.0, labels=labels1)

    # Lượt mới ở 5,9–8,0 (phủ mốc khối 6,0) nhưng chỉ chồng 0,1 s với lượt cũ (0–4,0 làn trước)
    turns2 = _turns((0.0, 2.1, "speaker_9"))
    labels2 = reg.label_window_turns(turns2, window_start_pts=5.9, boundary_pts=6.0)
    assert labels2["speaker_9"] == "speaker 2"


def test_registry_khong_khop_khi_o_xa_nhau():
    """Khớp danh tính dựa trên CHỒNG LẤN THỜI GIAN: vùng khác hẳn thì không nối được."""
    reg = SpeakerRegistry()
    turns1 = [SpeakerTurn(0.0, 4.0, "speaker_0")]
    labels1 = reg.label_window_turns(turns1, window_start_pts=0.0)
    reg.observe(turns1, window_start_pts=0.0, labels=labels1)

    labels2 = reg.label_window_turns([SpeakerTurn(0.0, 4.0, "speaker_5")], window_start_pts=600.0)
    assert labels2["speaker_5"] == "speaker 2"


def test_registry_tran_8_tai_su_dung_nhan_lau_nhat():
    """Quá `max_speakers` ⇒ tái dùng nhãn lâu nhất không thấy, KHÔNG sinh 'speaker 9'."""
    reg = SpeakerRegistry(max_speakers=8)
    for i in range(12):
        turns = [SpeakerTurn(0.0, 2.0, f"raw_{i}")]
        labels = reg.label_window_turns(turns, window_start_pts=float(i) * 100.0)
        reg.observe(turns, window_start_pts=float(i) * 100.0, labels=labels)
        assert labels[f"raw_{i}"] in {f"speaker {k}" for k in range(1, 9)}
    assert reg.labels == [f"speaker {k}" for k in range(1, 9)]


def test_registry_khong_ghi_lai_luot_da_nam_tron_vung_cu():
    """Lượt đã nằm TRỌN trong vùng đã gán nhãn không được ghi lần nữa (lịch sử không phình bản sao).

    Lượt VƯỢT qua mốc cũ thì VẪN ghi: model vừa thấy nó dài hơn nên biên được cập nhật.
    """
    reg = SpeakerRegistry()
    turns1 = [SpeakerTurn(0.0, 8.0, "speaker_0")]
    labels1 = reg.label_window_turns(turns1, window_start_pts=0.0)
    reg.observe(turns1, window_start_pts=0.0, labels=labels1)
    assert len(reg.history()) == 1

    # Cửa sổ sau: đuôi 0–8 s đã gán nhãn ở khối trước (khối trước kết thúc ở 8 s)
    turns2 = [SpeakerTurn(0.0, 8.0, "speaker_0"), SpeakerTurn(8.0, 14.0, "speaker_1")]
    labels2 = reg.label_window_turns(turns2, window_start_pts=0.0)
    reg.observe(turns2, window_start_pts=0.0, labels=labels2, already_covered_until_pts=8.0)

    hist = reg.history()
    assert len(hist) == 2, f"chỉ được thêm lượt MỚI, thực tế: {hist}"
    assert hist[1].start_pts == pytest.approx(8.0)


def test_attribute_sentences_gop_hai_id_tho_cung_mot_nhan():
    """Hai id thô cùng trỏ về MỘT nhãn phiên (model chớp nhãn) không được tạo 2 phụ đề liền kề."""
    sent = _sentence(12)
    # Từ 0–2,4 s: speaker_0; 2,4–4,0 s: speaker_9 (cùng người, model chớp); 4,0–4,8 s: speaker_0
    turns = [
        SpeakerTurn(0.0, 2.4, "speaker_0"),
        SpeakerTurn(2.4, 4.0, "speaker_9"),
        SpeakerTurn(4.0, 4.8, "speaker_0"),
    ]
    label_map = {"speaker_0": "speaker 1", "speaker_9": "speaker 1"}

    pieces = attribute_sentences([sent], turns, language="English", label_map=label_map)

    assert len(pieces) == 1, f"cùng nhãn phiên ⇒ một phụ đề, thực tế: {[p.speaker for p in pieces]}"
    assert pieces[0].speaker == "speaker 1"


@pytest.mark.asyncio
async def test_handler_noi_danh_tinh_qua_hai_khoi(monkeypatch):
    """P0.1 đầu-cuối: model ĐỔI SỐ người nói ở khối 2 nhưng nhãn gửi đi KHÔNG đổi.

    Kịch bản (đúng ca đo được ở `scratch/diar_chunk_split.py`):
      * khối 1 [0–12 s]: A nói suốt ⇒ model gọi A là `speaker_0`.
      * khối 2 [12–24 s]: A nói tới 12,5 s rồi B nói; model ĐÁNH SỐ LẠI ⇒ A = `speaker_1`, B = `speaker_0`.
    Không có P0.1, A sẽ bị đổi từ "speaker 1" sang "speaker 2" ngay giữa video.
    """
    from backend.diarization import DiarizationService
    from backend.ws.lookahead_handler import LookaheadSessionState

    session = LookaheadSessionState(ws=SimpleNamespace(send_json=None))
    session.apply_init({"source_lang": "en", "target_lang": "vi"})

    lengths: List[int] = []
    fake = _fake_diarize_factory(
        [
            # Lượt 1 — cửa sổ = [0–12 s]: chỉ có A
            _turns((0.0, 12.0, "speaker_0")),
            # Lượt 2 — cửa sổ = [4–24 s] (đã cộng 8 s ngữ cảnh): A nói tới 16 s, B từ 16 s
            _turns((0.0, 12.0, "speaker_1"), (12.0, 20.0, "speaker_0")),
        ],
        lengths,
    )
    monkeypatch.setattr(DiarizationService, "diarize_chunk", classmethod(fake))

    # Khối 1
    subs1 = [_sentence(30, sec_per_word=0.4)]  # 12 s
    chunk1 = _make_chunk(0.0, 12.0, session)
    _subs1, pieces1 = await session._attribute_speakers(subs1, chunk1)
    assert [p.speaker for p in pieces1] == ["speaker 1"]

    # Khối 2 — câu trải 12 s (mốc khối = 0), A nói 4 s đầu rồi B nói phần còn lại
    subs2 = [SubtitleSentence(text=" ".join(f"z{i}" for i in range(30)), start_time=0.0,
                              end_time=12.0, words=_words(30))]
    chunk2 = _make_chunk(12.0, 12.0, session)
    _subs2, pieces2 = await session._attribute_speakers(subs2, chunk2)

    labels2 = [p.speaker for p in pieces2]
    assert labels2 == ["speaker 1", "speaker 2"], (
        f"A phải GIỮ 'speaker 1' qua ranh giới khối, thực tế: {labels2}"
    )
    assert pieces2[0].end_time == pytest.approx(4.0), "ranh giới mảnh bám mốc từ (4 s = 10 từ × 0,4 s)"
    # Ngữ cảnh có được dùng thật: cửa sổ lượt 2 dài hơn khối (12 s) đúng bằng 8 s ngữ cảnh.
    assert lengths[0] == 12 * 16000
    assert lengths[1] == 20 * 16000, "cửa sổ phải gồm 8 s đuôi ngữ cảnh của khối trước"


@pytest.mark.asyncio
async def test_handler_khong_dung_ngu_canh_khi_tat_context(monkeypatch):
    """`context_sec=0` ⇒ TẮT P0.1: mỗi khối tự đánh số lại (hành vi cũ), cửa sổ = đúng khối."""
    from backend.config import config
    from backend.diarization import DiarizationService
    from backend.ws.lookahead_handler import LookaheadSessionState

    session = LookaheadSessionState(ws=SimpleNamespace(send_json=None))
    session.apply_init({"source_lang": "en", "target_lang": "vi"})

    lengths: List[int] = []
    fake = _fake_diarize_factory(
        [_turns((0.0, 12.0, "speaker_0")), _turns((0.0, 12.0, "speaker_0"))],
        lengths,
    )
    monkeypatch.setattr(DiarizationService, "diarize_chunk", classmethod(fake))
    monkeypatch.setattr(config.diarization, "context_sec", 0.0, raising=False)

    chunk1 = _make_chunk(0.0, 12.0, session)
    await session._attribute_speakers([_sentence(30)], chunk1)
    chunk2 = _make_chunk(12.0, 12.0, session)
    _subs2, pieces2 = await session._attribute_speakers([_sentence(30)], chunk2)

    assert lengths[1] == 12 * 16000, "tắt ngữ cảnh ⇒ cửa sổ đúng bằng khối"
    # Không có registry ⇒ khối 2 đánh số lại từ đầu: id thô speaker_0 → "speaker 1".
    assert [p.speaker for p in pieces2] == ["speaker 1"]
