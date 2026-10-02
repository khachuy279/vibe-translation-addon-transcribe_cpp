"""Unit tests cho LookaheadChunker (Phase 2).

Kiểm thử toàn diện các kịch bản:
1. Có khoảng lặng tự nhiên VAD >= 250ms trong [T+12s, T+18s] -> cắt tại midpoint.
2. Nói liên tục không có khoảng lặng -> fallback forced cut + 1.0s overlap.
3. Fast-Bootstrap sau khi tua video (Seek) -> cắt khối ngắn 3.0s - 4.0s.
4. Điều kiện biên: Hết luồng (end_of_stream) và chờ nạp đệm (chưa đủ 12s).
5. Chuỗi đa khối (Multi-chunk sequence) trên dòng audio 40s liên tục.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.lookahead_chunker import LookaheadChunker
from backend.core.lookahead_timeline import ContinuousAudioTimeline


def make_tone(duration_sec: float, freq: float = 440.0, sr: int = 16000, amp: float = 0.3) -> np.ndarray:
    """Tạo sóng âm giả lập tiếng nói (tone)."""
    t = np.linspace(0, duration_sec, int(round(duration_sec * sr)), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def make_silence(duration_sec: float, sr: int = 16000) -> np.ndarray:
    """Tạo khoảng lặng (silence zeros)."""
    return np.zeros(int(round(duration_sec * sr)), dtype=np.float32)


@pytest.fixture
def timeline():
    tl = ContinuousAudioTimeline(sample_rate=16000, max_storage_sec=3600.0)
    yield tl
    tl.clear()


class TestLookaheadChunker:
    """Kiểm thử hoạt động của LookaheadChunker."""

    def test_silence_gap_detection_and_cut(self, timeline):
        """Kịch bản 1: Có khoảng lặng 400ms ở mốc 14.0s - 14.4s -> cắt tại 14.2s (midpoint)."""
        sr = 16000
        # Cấu trúc: 14.0s tone -> 0.4s silence (14.0s - 14.4s) -> 10.0s tone
        part1 = make_tone(14.0, freq=300.0, sr=sr)
        silence = make_silence(0.4, sr=sr)
        part2 = make_tone(10.0, freq=400.0, sr=sr)

        full_pcm = np.concatenate([part1, silence, part2])
        timeline.append(pts_start=0.0, pcm=full_pcm)

        chunker = LookaheadChunker(
            timeline=timeline,
            sample_rate=sr,
            target_window_sec=15.0,
            min_window_sec=12.0,
            max_window_sec=18.0,
            min_silence_ms=250.0,
        )

        chunk = chunker.next_chunk(from_pts=0.0)
        assert chunk is not None
        assert chunk.pts_start == 0.0
        assert chunk.is_silence_boundary is True
        assert chunk.fallback_mode == "silence_gap"

        # Điểm cắt phải nằm trong khoảng lặng [14.0s, 14.4s], gần midpoint 14.2s (sai số <= 30ms)
        assert 14.15 <= chunk.pts_end <= 14.25
        assert chunk.next_read_pts == chunk.pts_end
        assert chunk.silence_gap_ms >= 350.0

    def test_continuous_speech_fallback_overlap(self, timeline):
        """Kịch bản 2: Nói liên tục không có khoảng lặng nào -> cắt tại điểm TRŨNG NHẤT + 1.0s overlap.

        SIẾT LẠI 2026-10-02: bản cũ cắt đúng tại MÉP dải (18.0s) — mép dải chắc chắn rơi vào giữa
        từ đang nói, và 1.0s overlap bị phiên âm lại mà không ai trừ ⇒ "cuối câu này là đầu của
        câu sau". Nay cắt tại frame ít năng lượng nhất trong dải (không phải mép dải).
        """
        sr = 16000
        # 25s tiếng nói liên tục không có khoảng lặng
        part = make_tone(25.0, freq=350.0, sr=sr, amp=0.25)
        timeline.append(pts_start=0.0, pcm=part)

        chunker = LookaheadChunker(
            timeline=timeline,
            sample_rate=sr,
            target_window_sec=15.0,
            min_window_sec=12.0,
            max_window_sec=18.0,
            min_silence_ms=250.0,
            overlap_sec=1.0,
        )

        chunk = chunker.next_chunk(from_pts=0.0)
        assert chunk is not None
        assert chunk.pts_start == 0.0
        assert chunk.is_silence_boundary is False
        assert chunk.fallback_mode == "forced_overlap"

        # Không còn cắt cứng ở mép dải; vẫn nằm trong [min_window, search_max]
        assert 12.0 <= chunk.pts_end <= 18.05
        # Mốc đọc tiếp theo phải lùi 1.0s overlap (phía nhận BẮT BUỘC trừ ở tầng từ)
        assert abs(chunk.next_read_pts - (chunk.pts_end - 1.0)) <= 0.05

    def test_real_silence_found_beyond_old_ceiling(self, timeline):
        """Khoảng lặng THẬT ở 24.3s (ngoài trần cũ 18s) ⇒ cắt ở đó, không cắt cứng ở 18s.

        Đây là điều kiện để giữ mục tiêu của Pipeline B v3: khối DÀI HƠN = ASR có nhiều ngữ cảnh
        hơn (CER/WER thấp hơn), mà mép khối vẫn sạch (không chẻ đôi từ).
        """
        sr = 16000
        part1 = make_tone(24.0, freq=300.0, sr=sr)
        gap = make_silence(0.6, sr=sr)
        part2 = make_tone(8.0, freq=400.0, sr=sr)
        timeline.append(pts_start=0.0, pcm=np.concatenate([part1, gap, part2]))

        # (a) Có nới dải tìm kiếm ⇒ cắt tại khoảng lặng thật, khối dài hơn trần cũ
        wide = LookaheadChunker(
            timeline=timeline, sample_rate=sr, target_window_sec=15.0,
            min_window_sec=12.0, max_window_sec=18.0, min_silence_ms=250.0,
            search_max_sec=30.0,
        )
        chunk = wide.next_chunk(from_pts=0.0)
        assert chunk is not None
        assert chunk.fallback_mode == "silence_gap"
        assert chunk.is_silence_boundary is True
        assert abs(chunk.pts_end - 24.3) <= 0.15, chunk
        assert chunk.duration > 18.5, "khối phải DÀI HƠN trần cũ để ASR có thêm ngữ cảnh"
        assert chunk.boundary_strength == "strong"

        # (b) Không nới dải (hành vi cũ) ⇒ không thấy khoảng lặng đó, phải cưỡng bức cắt
        narrow = LookaheadChunker(
            timeline=timeline, sample_rate=sr, target_window_sec=15.0,
            min_window_sec=12.0, max_window_sec=18.0, min_silence_ms=250.0,
        )
        legacy = narrow.next_chunk(from_pts=0.0)
        assert legacy is not None
        assert legacy.fallback_mode == "forced_overlap"
        assert legacy.pts_end <= 18.05

    def test_quiet_audio_relative_silence_threshold(self, timeline):
        """Ngưỡng lặng TƯƠNG ĐỐI: audio thu nhỏ (RMS ~0.007) vẫn tìm đúng khoảng lặng thật.

        Ngưỡng tuyệt đối cũ (0.015) coi TOÀN BỘ vùng này là "im lặng" ⇒ không có ranh giới nào,
        hoặc cắt ở giữa dải — tức là giữa câu.
        """
        sr = 16000
        part1 = make_tone(14.0, freq=300.0, sr=sr, amp=0.01)   # RMS ~0.007 < 0.015
        silence = make_silence(0.5, sr=sr)
        part2 = make_tone(10.0, freq=400.0, sr=sr, amp=0.01)
        timeline.append(pts_start=0.0, pcm=np.concatenate([part1, silence, part2]))

        chunker = LookaheadChunker(
            timeline=timeline, sample_rate=sr, target_window_sec=15.0,
            min_window_sec=12.0, max_window_sec=18.0, min_silence_ms=250.0,
        )
        chunk = chunker.next_chunk(from_pts=0.0)
        assert chunk is not None
        assert chunk.fallback_mode == "silence_gap", chunk
        assert 14.15 <= chunk.pts_end <= 14.35, chunk

    def test_short_energy_dips_are_not_sentence_boundaries(self, timeline):
        """Khe 60ms giữa hai từ KHÔNG được coi là ranh giới câu (bản cũ thì có).

        Bản cũ chấp nhận `min_rms < 0.015 * 2 = 0.03` cho MỘT frame 50ms bất kỳ ⇒ cắt vào khe
        giữa hai từ, tức là giữa câu. Nay khe phải kéo dài >= `min_rms_hold_ms` (120ms).
        """
        sr = 16000
        pcm = make_tone(25.0, freq=350.0, sr=sr, amp=0.25)
        for i in range(1, 6):
            a = int(i * 3.0 * sr)
            pcm[a : a + int(0.06 * sr)] = 0.001

        timeline.append(pts_start=0.0, pcm=pcm)
        chunker = LookaheadChunker(
            timeline=timeline, sample_rate=sr, target_window_sec=15.0,
            min_window_sec=12.0, max_window_sec=18.0, min_silence_ms=250.0,
            min_rms_hold_ms=120.0,
        )
        chunk = chunker.next_chunk(from_pts=0.0)
        assert chunk is not None
        # Không có khe nào >= 120ms ⇒ không được coi là ranh giới mềm
        assert chunk.fallback_mode == "forced_overlap", chunk

    def test_fast_bootstrap_after_seek(self, timeline):
        """Kịch bản 3: Sau khi Seek -> Fast Bootstrap trích xuất khối ngắn 3.0s - 4.0s."""
        sr = 16000
        part = make_tone(15.0, freq=440.0, sr=sr)
        timeline.append(pts_start=10.0, pcm=part)

        chunker = LookaheadChunker(timeline=timeline, sample_rate=sr)

        # Tua tới mốc 10.0s, bật fast_bootstrap
        chunk = chunker.next_chunk(from_pts=10.0, fast_bootstrap=True)
        assert chunk is not None
        assert chunk.pts_start == 10.0
        assert chunk.fallback_mode == "bootstrap"
        # Khối ngắn trong dải 3.0s - 4.0s
        assert 3.0 <= chunk.duration <= 4.05
        assert chunk.next_read_pts == chunk.pts_end

    def test_insufficient_audio_waits_for_buffer(self, timeline):
        """Kịch bản 4A: Chỉ có 8s audio và chưa hết stream -> trả None để chờ buffer nạp thêm."""
        sr = 16000
        part = make_tone(8.0, freq=440.0, sr=sr)
        timeline.append(pts_start=0.0, pcm=part)

        chunker = LookaheadChunker(timeline=timeline, sample_rate=sr, min_window_sec=12.0)

        chunk = chunker.next_chunk(from_pts=0.0, is_stream_end=False)
        assert chunk is None

    def test_end_of_stream_extracts_remainder(self, timeline):
        """Kịch bản 4B: Chỉ có 8s audio nhưng is_stream_end=True -> trích xuất trọn vẹn 8s còn lại."""
        sr = 16000
        part = make_tone(8.0, freq=440.0, sr=sr)
        timeline.append(pts_start=0.0, pcm=part)

        chunker = LookaheadChunker(timeline=timeline, sample_rate=sr, min_window_sec=12.0)

        chunk = chunker.next_chunk(from_pts=0.0, is_stream_end=True)
        assert chunk is not None
        assert chunk.pts_start == 0.0
        assert abs(chunk.pts_end - 8.0) <= 0.05
        assert chunk.fallback_mode == "end_of_stream"

    def test_multi_chunk_pipeline_progression(self, timeline):
        """Kịch bản 5: Tiến trình cắt nhiều khối liên tiếp trên đoạn audio 40s có nhiều khoảng lặng."""
        sr = 16000
        # Tạo chuỗi audio 40s có 2 khoảng lặng lớn:
        # [0 - 13.5s tone] -> [13.5 - 14.0s silence (500ms)] -> [14.0 - 28.0s tone] -> [28.0 - 28.5s silence (500ms)] -> [28.5 - 40.0s tone]
        t1 = make_tone(13.5, freq=300.0, sr=sr)
        s1 = make_silence(0.5, sr=sr)
        t2 = make_tone(14.0, freq=400.0, sr=sr)
        s2 = make_silence(0.5, sr=sr)
        t3 = make_tone(11.5, freq=500.0, sr=sr)

        full_pcm = np.concatenate([t1, s1, t2, s2, t3])
        timeline.append(pts_start=0.0, pcm=full_pcm)

        chunker = LookaheadChunker(
            timeline=timeline,
            sample_rate=sr,
            target_window_sec=15.0,
            min_window_sec=12.0,
            max_window_sec=18.0,
            min_silence_ms=250.0,
        )

        chunks = []
        curr_pts = 0.0

        for _ in range(5):
            c = chunker.next_chunk(from_pts=curr_pts, is_stream_end=True)
            if c is None:
                break
            chunks.append(c)
            curr_pts = c.next_read_pts
            if curr_pts >= 40.0 - 0.05:
                break

        assert len(chunks) >= 3
        # Chunk 1 cắt tại khoảng lặng 1 (~13.75s)
        assert chunks[0].pts_start == 0.0
        assert 13.6 <= chunks[0].pts_end <= 13.9
        assert chunks[0].is_silence_boundary is True

        # Chunk 2 bắt đầu tại mốc kết thúc của Chunk 1 và cắt tại khoảng lặng 2 (~28.25s)
        assert abs(chunks[1].pts_start - chunks[0].pts_end) <= 0.05
        assert 28.1 <= chunks[1].pts_end <= 28.4
        assert chunks[1].is_silence_boundary is True

        # Chunk 3 trích xuất phần còn lại đến cuối
        assert abs(chunks[2].pts_start - chunks[1].pts_end) <= 0.05
        assert abs(chunks[-1].pts_end - 40.0) <= 0.1
