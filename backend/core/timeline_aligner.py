"""Timeline Aligner & Lookahead Window Manager.

Quản lý luồng âm thanh theo dòng thời gian (PTS timeline):
1. Sắp xếp và ghép nối các DecodedAudioChunk theo mốc PTS không ngắt quãng.
2. Phát hiện khoảng trống (Gap) hoặc đoạn gối đầu (Overlap) để xử lý bù/cắt mẫu.
3. Cắt lát âm thanh (Audio Slices) gửi sang ASR theo câu/đoạn hoàn chỉnh.
4. Quản lý cửa sổ trượt (Sliding Window) tự động giải phóng vùng quá khứ.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from backend.core.lookahead_demuxer import DecodedAudioChunk


@dataclass
class AudioTimelineSlice:
    """Đoạn âm thanh hoàn chỉnh trích xuất từ Timeline sẵn sàng cho ASR."""
    pts_start: float
    pts_end: float
    duration: float
    pcm: np.ndarray


class TimelineAligner:
    """Bộ đồng bộ dòng thời gian Lookahead (Presentation Time Stamp Aligner)."""

    def __init__(self, sample_rate: int = 16000, max_future_window_sec: float = 60.0):
        self.sample_rate = sample_rate
        self.max_future_window_sec = max_future_window_sec
        self._chunks: List[DecodedAudioChunk] = []
        self._last_processed_pts: float = 0.0

    def add_chunk(self, chunk: DecodedAudioChunk):
        """Thêm một chunk âm thanh đã giải mã vào timeline theo thứ tự PTS."""
        if chunk is None or chunk.pcm.size == 0:
            return

        # Tránh trùng lặp chunk giống nhau hoàn toàn
        for existing in self._chunks:
            if abs(existing.pts_start - chunk.pts_start) < 0.02 and abs(existing.duration - chunk.duration) < 0.02:
                return

        self._chunks.append(chunk)
        # Sắp xếp lại danh sách chunk theo pts_start tăng dần
        self._chunks.sort(key=lambda c: c.pts_start)

    def get_buffered_range(self) -> Tuple[float, float]:
        """Trả về khoảng thời gian [min_pts, max_pts] hiện đang lưu trữ."""
        if not self._chunks:
            return 0.0, 0.0
        return self._chunks[0].pts_start, self._chunks[-1].pts_end

    def extract_slice(self, start_pts: float, end_pts: float) -> Optional[AudioTimelineSlice]:
        """Trích xuất một đoạn PCM liên tục trong khoảng [start_pts, end_pts]."""
        if end_pts <= start_pts or not self._chunks:
            return None

        # Tìm các chunk nằm trong phạm vi
        relevant = [c for c in self._chunks if c.pts_end > start_pts and c.pts_start < end_pts]
        if not relevant:
            return None

        total_samples = int((end_pts - start_pts) * self.sample_rate)
        if total_samples <= 0:
            return None

        merged_pcm = np.zeros(total_samples, dtype=np.float32)

        for c in relevant:
            # Tính chỉ số mẫu trong merged_pcm
            c_start_rel = max(0.0, c.pts_start - start_pts)
            c_end_rel = min(end_pts - start_pts, c.pts_end - start_pts)

            idx_start = int(c_start_rel * self.sample_rate)
            idx_end = int(c_end_rel * self.sample_rate)

            # Tính vị trí cắt trong chunk.pcm
            chunk_offset = max(0.0, start_pts - c.pts_start)
            c_idx_start = int(chunk_offset * self.sample_rate)
            needed_samples = idx_end - idx_start

            if c_idx_start < len(c.pcm) and needed_samples > 0:
                c_idx_end = min(len(c.pcm), c_idx_start + needed_samples)
                actual_samples = c_idx_end - c_idx_start
                merged_pcm[idx_start : idx_start + actual_samples] = c.pcm[c_idx_start:c_idx_end]

        return AudioTimelineSlice(
            pts_start=start_pts,
            pts_end=end_pts,
            duration=end_pts - start_pts,
            pcm=merged_pcm
        )

    def prune_past(self, current_time: float, keep_behind_sec: float = 30.0):
        """Giải phóng bộ nhớ các chunk nằm trước (current_time - keep_behind_sec)."""
        cutoff_pts = max(0.0, current_time - keep_behind_sec)
        self._chunks = [c for c in self._chunks if c.pts_end > cutoff_pts]

    def reset(self, target_time: float = 0.0):
        """Xóa sạch bộ đệm khi tua video (Seek Reset)."""
        self._chunks.clear()
        self._last_processed_pts = target_time
