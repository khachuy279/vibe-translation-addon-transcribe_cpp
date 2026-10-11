"""test_76 — Quét khoảng lặng bằng engine VAD streaming (FireRed) cho lớp ngắt câu theo khoảng lặng.

Bối cảnh (2026-10-11): lớp ngắt câu theo khoảng lặng cần biết chỗ nào là im lặng THẬT trong khối.
`VADSilenceScanner` gắn cứng model Silero và đo thật cho thấy bỏ sót phần lớn khoảng lặng giữa các
từ (350 ms → 2/8 khe thật của `ja_concat_1min.wav`; FireRed ở 150 ms → 8/8, không nhận nhầm khe nào
— xem `find_silence_gaps_streaming`).

Test ở đây dùng engine GIẢ (khung 160 mẫu = 10 ms) để khoá LOGIC gom khoảng lặng, không cần model.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from backend.asr.forced_aligner import AlignedWord, ForcedAlignerService, SubtitleSentence
from backend.core.vad_silence import find_silence_gaps_streaming

HOP = 160          # 10 ms @16 kHz — đúng `FRAME_SHIFT_SAMPLE` của FireRed
SR = 16000


class ScriptedVADEngine:
    """Engine VAD giả: trả `is_speech` theo KỊCH BẢN đã cho (mỗi frame 10 ms)."""

    frame_samples = HOP

    def __init__(self, flags: List[bool]):
        self.flags = flags
        self.calls = 0

    def create_initial_state(self):
        return object()

    def is_speech(self, frame, state):
        from backend.vad.base import VADResult

        idx = self.calls
        self.calls += 1
        r = VADResult()
        r.is_speech = bool(self.flags[idx]) if idx < len(self.flags) else False
        return r


def _pcm(n_frames: int) -> np.ndarray:
    return np.full(n_frames * HOP, 0.1, dtype=np.float32)


def test_gom_dung_cac_doan_im_lang_du_dai():
    """Im lặng 20 frame (200 ms) ⇒ 1 khoảng lặng; im lặng 10 frame (100 ms) ⇒ bị bỏ qua ở 150 ms."""
    # 5 frame nói | 20 frame lặng | 5 frame nói | 10 frame lặng | 5 frame nói
    flags = [True] * 5 + [False] * 20 + [True] * 5 + [False] * 10 + [True] * 5
    engine = ScriptedVADEngine(flags)

    gaps = find_silence_gaps_streaming(engine, _pcm(len(flags)), base_pts=100.0, min_silence_ms=150.0)

    assert len(gaps) == 1, gaps
    g = gaps[0]
    assert g.vad_confirmed is True
    # Bắt đầu ở frame 5 (50 ms) và kết thúc ở frame 25 (250 ms), cộng `base_pts`.
    assert g.start_pts == pytest.approx(100.0 + 5 * HOP / SR)
    assert g.end_pts == pytest.approx(100.0 + 25 * HOP / SR)
    assert g.duration == pytest.approx(0.2, abs=1e-3)


def test_im_lang_o_cuoi_duoc_tinh():
    """Im lặng chạy tới hết audio cũng phải ra khoảng lặng (ca cắt khối ở mép)."""
    flags = [True] * 4 + [False] * 30
    gaps = find_silence_gaps_streaming(
        ScriptedVADEngine(flags), _pcm(len(flags)), base_pts=0.0, min_silence_ms=150.0
    )
    assert len(gaps) == 1
    assert gaps[0].start_pts == pytest.approx(4 * HOP / SR)
    assert gaps[0].end_pts == pytest.approx(34 * HOP / SR)


def test_nguong_ngan_hon_thi_bat_duoc_khoang_lang_ngan():
    """Cùng dữ liệu, hạ ngưỡng xuống 100 ms ⇒ bắt thêm khoảng lặng 10 frame."""
    flags = [True] * 5 + [False] * 20 + [True] * 5 + [False] * 10 + [True] * 5
    gaps = find_silence_gaps_streaming(
        ScriptedVADEngine(flags), _pcm(len(flags)), base_pts=0.0, min_silence_ms=100.0
    )
    assert len(gaps) == 2, gaps


@pytest.mark.parametrize("pcm", [None, np.zeros(0, dtype=np.float32), np.zeros(10, dtype=np.float32)])
def test_audio_rong_hoac_qua_ngan_tra_rong(pcm):
    """Không đủ một frame ⇒ không có khoảng lặng nào (không được nổ)."""
    assert find_silence_gaps_streaming(ScriptedVADEngine([True]), pcm, min_silence_ms=150.0) == []


def test_engine_none_tra_rong():
    assert find_silence_gaps_streaming(None, _pcm(10), min_silence_ms=150.0) == []


def test_pcm_float32_duoc_doi_sang_int16_dung_cong_thuc_processor():
    """Engine nhận int16; giá trị phải khớp công thức `(clip(x,-1,1) * 32767)` của VADProcessor."""
    seen = {}

    class _Capturing(ScriptedVADEngine):
        def is_speech(self, frame, state):
            seen["dtype"] = frame.dtype
            seen["max"] = int(np.abs(frame).max())
            return super().is_speech(frame, state)

    find_silence_gaps_streaming(_Capturing([True] * 3), _pcm(3), min_silence_ms=150.0)
    assert seen["dtype"] == np.int16
    assert seen["max"] == int(0.1 * 32767)


# ─────────────────────────────── cổng "chỉ quét VAD khi cần" (dry_run)

def _sub_with_gaps(gaps_after: List[int], n_words: int = 12) -> SubtitleSentence:
    """Câu có khe 0.6 s ngay SAU các từ ở chỉ số `gaps_after`."""
    words = []
    t = 0.0
    for i in range(n_words):
        start = t + (0.6 if i in gaps_after and i > 0 else 0.0)
        words.append(AlignedWord(text=f"w{i}", start_time=start, end_time=start + 0.2))
        t = start + 0.2
    return SubtitleSentence(
        text=" ".join(w.text for w in words),
        start_time=words[0].start_time,
        end_time=words[-1].end_time,
        words=words,
    )


def test_dry_run_dem_khe_ma_khong_tach():
    """`dry_run` chỉ đếm: dùng để bỏ qua bước quét FireRed (~2.6 s/khối) khi không có khe nào."""
    sub = _sub_with_gaps([4, 8])
    stats: dict = {}
    out = ForcedAlignerService.split_subtitles_by_pause(
        [sub], min_pause_ms=350, language="English", stats=stats, dry_run=True
    )
    assert out == [sub], "dry_run KHÔNG được tách"
    assert stats["candidates"] == 2
    assert stats["accepted"] == 0


def test_dry_run_khong_co_khe_thi_dem_bang_khong():
    stats: dict = {}
    ForcedAlignerService.split_subtitles_by_pause(
        [_sub_with_gaps([])], min_pause_ms=350, language="English", stats=stats, dry_run=True
    )
    assert stats["candidates"] == 0
