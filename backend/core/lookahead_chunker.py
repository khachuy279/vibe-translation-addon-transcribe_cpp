"""LookaheadChunker: Bộ cắt khối âm thanh thông minh (VAD Silence Boundary Chunker).

Nhiệm vụ cốt lõi (Phase 2):
1. Quét tìm khoảng lặng VAD (silence gap >= 250ms) trong dải [T + 12s, T + 18s] trên RAM timeline.
2. Cắt tại điểm chính giữa khoảng lặng (midpoint) để cả đuôi khối hiện tại và đầu khối kế tiếp
   đều có biên im lặng bảo vệ, tuyệt đối không bao giờ cắt vào giữa từ.
3. Fallback khi nói liên tục không ngừng nghỉ:
   - Tìm điểm cực tiểu năng lượng âm học (local minimum RMS).
   - Cưỡng bức cắt tại trần thời gian kèm 1.0s overlap (chồng lấn) cho khối sau.
4. Xử lý điều kiện biên (Edge cases):
   - Fast-Bootstrap sau khi tua video (Seek): cửa sổ ngắn 3.0s - 4.0s.
   - Hết luồng (End of stream / Stream ended): trích xuất trọn vẹn phần còn lại.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from backend.core.lookahead_timeline import ContinuousAudioTimeline
from backend.utils.logger import get_logger

logger = get_logger("core.lookahead_chunker")


@dataclass
class ChunkSlice:
    pts_start: float
    pts_end: float
    pcm: np.ndarray
    is_silence_boundary: bool
    fallback_mode: str  # "silence_gap", "min_rms", "forced_overlap", "end_of_stream", "bootstrap"
    next_read_pts: float
    silence_gap_ms: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.pts_end - self.pts_start)

    def __repr__(self) -> str:
        return (
            f"ChunkSlice([{self.pts_start:.3f}s -> {self.pts_end:.3f}s] "
            f"dur={self.duration:.2f}s, mode='{self.fallback_mode}', "
            f"silence={self.silence_gap_ms:.0f}ms, next={self.next_read_pts:.3f}s)"
        )


class LookaheadChunker:
    """Bộ cắt khối âm thanh dựa vào VAD và năng lượng âm học."""

    def __init__(
        self,
        timeline: ContinuousAudioTimeline,
        sample_rate: int = 16000,
        target_window_sec: float = 15.0,
        min_window_sec: float = 12.0,
        max_window_sec: float = 18.0,
        min_silence_ms: float = 250.0,
        silence_rms_threshold: float = 0.015,
        overlap_sec: float = 1.0,
        vad_engine: Optional[Any] = None,
    ):
        self.timeline = timeline
        self.sample_rate = int(sample_rate)
        self.target_window_sec = float(target_window_sec)
        self.min_window_sec = float(min_window_sec)
        self.max_window_sec = float(max_window_sec)
        self.min_silence_ms = float(min_silence_ms)
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.overlap_sec = float(overlap_sec)
        self.vad_engine = vad_engine

    def scan_silence_gaps(
        self,
        pcm: np.ndarray,
        base_pts: float,
        frame_ms: float = 20.0,
    ) -> List[Tuple[float, float, float]]:
        """Quét tìm các khoảng lặng liên tục trong mảng PCM.

        Returns:
            Danh sách các tuple `(gap_start_pts, gap_end_pts, gap_duration_ms)`.
        """
        if pcm is None or len(pcm) == 0:
            return []

        frame_samples = int(round(frame_ms * self.sample_rate / 1000.0))
        if frame_samples <= 0 or len(pcm) < frame_samples:
            return []

        num_frames = len(pcm) // frame_samples
        gaps: List[Tuple[float, float, float]] = []

        curr_gap_start: Optional[float] = None
        min_silence_sec = self.min_silence_ms / 1000.0

        for f_idx in range(num_frames):
            frame = pcm[f_idx * frame_samples : (f_idx + 1) * frame_samples]
            f_start_pts = base_pts + (f_idx * frame_samples) / self.sample_rate
            f_end_pts = f_start_pts + frame_ms / 1000.0

            # Tính năng lượng RMS
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            is_silent = rms < self.silence_rms_threshold

            if is_silent:
                if curr_gap_start is None:
                    curr_gap_start = f_start_pts
            else:
                if curr_gap_start is not None:
                    gap_dur = f_start_pts - curr_gap_start
                    if gap_dur >= min_silence_sec:
                        gaps.append((curr_gap_start, f_start_pts, gap_dur * 1000.0))
                    curr_gap_start = None

        if curr_gap_start is not None:
            gap_dur = (base_pts + (num_frames * frame_samples) / self.sample_rate) - curr_gap_start
            if gap_dur >= min_silence_sec:
                gaps.append((curr_gap_start, curr_gap_start + gap_dur, gap_dur * 1000.0))

        return gaps

    def find_optimal_cut_point(
        self,
        pcm_search: np.ndarray,
        search_start_pts: float,
        ideal_cut_pts: float,
    ) -> Tuple[float, bool, str, float]:
        """Tìm điểm cắt tối ưu trong vùng tìm kiếm [search_start_pts, search_end_pts].

        Returns:
            Tuple `(cut_pts, is_silence_boundary, fallback_mode, silence_gap_ms)`.
        """
        search_dur = len(pcm_search) / self.sample_rate
        search_end_pts = search_start_pts + search_dur

        # 1. Tìm các khoảng lặng VAD >= min_silence_ms
        gaps = self.scan_silence_gaps(pcm_search, base_pts=search_start_pts)
        if gaps:
            # Chọn khoảng lặng gần với ideal_cut_pts nhất (hoặc khoảng lặng dài nhất)
            # Ưu tiên: khoảng cách tới ideal_cut_pts có trọng số kết hợp độ dài khoảng lặng
            best_gap = None
            best_score = float("-inf")

            for g_start, g_end, g_ms in gaps:
                g_mid = (g_start + g_end) / 2.0
                dist = abs(g_mid - ideal_cut_pts)
                # Score cao khi: khoảng lặng dài và gần mốc lý tưởng
                score = (g_ms / 1000.0) * 2.0 - dist
                if score > best_score:
                    best_score = score
                    best_gap = (g_start, g_end, g_ms)

            if best_gap is not None:
                g_start, g_end, g_ms = best_gap
                cut_pts = (g_start + g_end) / 2.0
                return float(cut_pts), True, "silence_gap", float(g_ms)

        # 2. Fallback 1: Tìm điểm cực tiểu RMS (local minimum energy frame)
        frame_samples = int(round(0.050 * self.sample_rate))  # 50ms window
        num_frames = len(pcm_search) // frame_samples
        if num_frames > 2:
            rms_vals = [
                float(np.sqrt(np.mean(pcm_search[i * frame_samples : (i + 1) * frame_samples] ** 2)))
                for i in range(num_frames)
            ]
            min_idx = int(np.argmin(rms_vals))
            min_rms = rms_vals[min_idx]
            cut_pts = search_start_pts + (min_idx + 0.5) * 0.050

            if min_rms < self.silence_rms_threshold * 2.0:
                return float(cut_pts), False, "min_rms", 0.0

        # 3. Fallback 2: Cưỡng bức cắt tại cuối vùng tìm kiếm (forced cut + overlap)
        return float(search_end_pts), False, "forced_overlap", 0.0

    def next_chunk(
        self,
        from_pts: float,
        fast_bootstrap: bool = False,
        is_stream_end: bool = False,
    ) -> Optional[ChunkSlice]:
        """Trích xuất khối audio tiếp theo tính từ `from_pts`.

        Args:
            from_pts: Mốc thời gian tuyệt đối (PTS giây) bắt đầu lấy audio.
            fast_bootstrap: Bật khi vừa Tua video (Seek) - lấy nhanh 3.0s - 4.0s để xuất câu đầu tiên tức thì.
            is_stream_end: Bật khi video đã kết thúc hoặc không còn nhận thêm audio.

        Returns:
            ChunkSlice nếu có đủ audio để xử lý, hoặc None nếu cần chờ buffer nạp thêm.
        """
        buffered_end = self.timeline.buffered_end_from(from_pts)
        if buffered_end is None or buffered_end <= from_pts + 0.05:
            return None

        available_sec = buffered_end - from_pts

        # 1. KỊCH BẢN FAST-BOOTSTRAP (Sau khi Seek)
        if fast_bootstrap:
            bootstrap_min = 2.5
            bootstrap_max = 4.0
            if available_sec < bootstrap_min and not is_stream_end:
                return None
            target_dur = min(available_sec, bootstrap_max)
            audio_res = self.timeline.get_audio_range(from_pts, target_dur)
            if audio_res is None:
                return None
            pts_start, pcm = audio_res
            cut_pts = pts_start + len(pcm) / self.sample_rate
            return ChunkSlice(
                pts_start=pts_start,
                pts_end=cut_pts,
                pcm=pcm,
                is_silence_boundary=False,
                fallback_mode="bootstrap",
                next_read_pts=cut_pts,
                silence_gap_ms=0.0,
            )

        # 2. KIỂM TRA ĐỦ AUDIO ĐỂ CẮT CỬA SỔ CHUẨN
        if available_sec < self.min_window_sec:
            if not is_stream_end:
                # Chưa đủ audio, đợi buffer nạp thêm để đảm bảo ngữ cảnh offline
                return None
            # Hết stream: lấy nốt phần còn lại
            audio_res = self.timeline.get_audio_range(from_pts, available_sec)
            if audio_res is None:
                return None
            pts_start, pcm = audio_res
            cut_pts = pts_start + len(pcm) / self.sample_rate
            return ChunkSlice(
                pts_start=pts_start,
                pts_end=cut_pts,
                pcm=pcm,
                is_silence_boundary=False,
                fallback_mode="end_of_stream",
                next_read_pts=cut_pts,
                silence_gap_ms=0.0,
            )

        # 3. QUÉT TÌM ĐIỂM CẮT TỐI ƯU TRONG VÙNG [T + min_window, T + max_window]
        search_start_pts = from_pts + self.min_window_sec
        search_max_pts = from_pts + min(available_sec, self.max_window_sec)
        search_dur = max(0.1, search_max_pts - search_start_pts)

        # Lấy audio vùng tìm kiếm để quét VAD
        search_res = self.timeline.get_audio_range(search_start_pts, search_dur)
        if search_res is None:
            # Fallback nếu không đọc được vùng tìm kiếm
            cut_pts = from_pts + self.target_window_sec
            is_silence = False
            fallback_mode = "forced_overlap"
            silence_ms = 0.0
        else:
            _, search_pcm = search_res
            ideal_cut = from_pts + min(search_max_pts - from_pts, self.target_window_sec)
            cut_pts, is_silence, fallback_mode, silence_ms = self.find_optimal_cut_point(
                search_pcm,
                search_start_pts=search_start_pts,
                ideal_cut_pts=ideal_cut,
            )

        total_take_sec = max(0.1, cut_pts - from_pts)
        final_res = self.timeline.get_audio_range(from_pts, total_take_sec)
        if final_res is None:
            return None

        actual_start, chunk_pcm = final_res
        actual_end = actual_start + len(chunk_pcm) / self.sample_rate

        # Nếu là forced_overlap, chunk kế tiếp sẽ bắt đầu sớm hơn overlap_sec để bảo toàn từ ngữ
        if fallback_mode == "forced_overlap":
            next_start = max(actual_start + 1.0, actual_end - self.overlap_sec)
        else:
            next_start = actual_end

        return ChunkSlice(
            pts_start=actual_start,
            pts_end=actual_end,
            pcm=chunk_pcm,
            is_silence_boundary=is_silence,
            fallback_mode=fallback_mode,
            next_read_pts=float(next_start),
            silence_gap_ms=float(silence_ms),
        )
