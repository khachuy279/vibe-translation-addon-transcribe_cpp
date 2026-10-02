"""ContinuousAudioTimeline — Dòng âm thanh Lookahead LIÊN TỤC & BỀN VỮNG theo PTS tuyệt đối.

Nhiệm vụ: biến các đoạn PCM rời rạc (kết quả giải mã từ các mảnh MSE) thành
kho lưu trữ âm thanh liên tục theo mốc thời gian tuyệt đối trên RAM trong suốt phiên.

Cải tiến v3 (Persistent RAM Audio Storage for Pipeline B):
1. **Lưu trữ bền vững trên RAM (Persistent RAM Storage)**:
   Mọi mẫu PCM (float32 @ 16kHz) từ khi bắt đầu video được lưu liên tục trong RAM
   (đáp ứng video dài 3+ tiếng ~690 MB).
2. **Đọc không huỷ (Non-destructive read)**:
   Các lời gọi `read()` lấy audio nạp cho ASR/VAD không xoá dữ liệu khỏi RAM.
3. **Bảo tồn khi Tua / Seek (Seek Resilient)**:
   Khi người dùng tua video (trước hoặc sau), `seek(target_time)` chỉ đổi mốc con trỏ
   đọc `_cursor`, KHÔNG XOÁ dữ liệu audio đã đệm trong RAM.
   - Seek lùi (về vùng đã phát): ngay lập tức có audio để ASR giải mã với độ trễ 0s.
   - Seek trong vùng đã buffer: ngay lập tức có audio mà không phụ thuộc trình duyệt tải lại.
4. **Tự động ghép nối và xử lý chồng lấn (Stitch & Overlap Resolution)**:
   Chấp nhận các mảnh đến theo bất kỳ thứ tự nào (out-of-order) và tự động gộp/cắt chồng lấn.
5. **Xử lý khe hở thông minh (Smart Gap Handling)**:
   - Khe hở nhỏ (<= 5.0s, khoảng lặng nói): lấp im lặng (silence) để bảo toàn mốc thời gian tuyệt đối.
   - Khe hở lớn (> 5.0s, do tua nhảy cóc): trả None để chờ buffer nạp thay vì tạo im lặng giả.
"""

from __future__ import annotations

import bisect
import threading
from typing import List, Optional, Tuple

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger("core.lookahead_timeline")

#: Hai đoạn lệch nhau dưới mức này coi như LIỀN nhau khi ĐỌC (sai số lượng tử hóa PTS 1 ms
#: + mép cluster). Lớn hơn ⇒ lấp silence để bảo toàn mốc thời gian tuyệt đối.
GAP_EPS_SEC = 0.02
#: Ngưỡng "liền nhau" khi BÁO CÁO vùng đệm liên tục. Rộng hơn `GAP_EPS_SEC` vì các mối nối
#: giữa hai lượt giải mã có thể hụt ~20-40 ms; nếu lấy chặt thì `buffered_ahead` báo sai
#: (thấp hơn thực tế) và client tưởng buffer ngắn.
REPORT_EPS_SEC = 0.15
#: Khe hở tối đa được phép lấp bằng im lặng (silence). Lớn hơn ngưỡng này được coi là
#: khoảng trống do tua (seek hole) và sẽ trả None để chờ dữ liệu thật thay vì lấp hàng chục giây im lặng giả.
MAX_GAP_FILL_SEC = 5.0


class AudioChunk:
    __slots__ = ("pts_start", "pts_end", "pcm")

    def __init__(self, pts_start: float, pts_end: float, pcm: np.ndarray):
        self.pts_start = float(pts_start)
        self.pts_end = float(pts_end)
        self.pcm = pcm

    def __repr__(self) -> str:
        return f"AudioChunk([{self.pts_start:.3f}s -> {self.pts_end:.3f}s], len={len(self.pcm)})"


class ContinuousAudioTimeline:
    """Ghép các đoạn PCM rời rạc thành kho âm thanh liên tục & bền vững trên RAM theo PTS tuyệt đối."""

    def __init__(
        self,
        sample_rate: int = 16000,
        max_pending_sec: float = 90.0,
        max_storage_sec: float = 10800.0,  # 3 tiếng mặc định (khoảng 690 MB float32)
    ):
        self.sample_rate = int(sample_rate)
        self.max_pending_sec = float(max_pending_sec)
        self.max_storage_sec = float(max_storage_sec)

        self._lock = threading.RLock()
        self._chunks: List[AudioChunk] = []
        self._cursor: Optional[float] = None      # PTS tuyệt đối của mẫu kế tiếp sẽ đọc
        self._session_base_pts: Optional[float] = None  # PTS ứng với sample 0 của phiên ASR
        self._gap_filled_sec: float = 0.0
        self._appended_sec: float = 0.0
        self._has_logged_storage_start: bool = False

    # ------------------------------------------------------------------ trạng thái
    @property
    def cursor_pts(self) -> Optional[float]:
        """PTS của mẫu kế tiếp sẽ được đọc (None nếu chưa có dữ liệu)."""
        return self._cursor

    @property
    def session_base_pts(self) -> Optional[float]:
        """PTS tuyệt đối ứng với sample 0 của bộ đệm ASR (mốc neo của phiên)."""
        return self._session_base_pts

    @property
    def gap_filled_sec(self) -> float:
        return self._gap_filled_sec

    def total_stored_seconds(self) -> float:
        """Tổng thời lượng audio (giây) đang lưu trữ trong RAM."""
        with self._lock:
            return sum(len(c.pcm) for c in self._chunks) / self.sample_rate

    def total_stored_bytes(self) -> int:
        """Tổng dung lượng RAM (bytes) đang lưu trữ audio PCM."""
        with self._lock:
            return sum(c.pcm.nbytes for c in self._chunks)

    def pending_seconds(self) -> float:
        """Tổng độ dài audio liên tục đang chờ đọc phía trước con trỏ."""
        end = self.buffered_end_pts()
        if end is None or self._cursor is None:
            return 0.0
        return max(0.0, end - self._cursor)

    def seek(self, target_time: float) -> None:
        """Tua video: chỉ dịch chuyển con trỏ đọc, KHÔNG xoá audio trong RAM."""
        with self._lock:
            self._cursor = float(target_time)
            self._session_base_pts = float(target_time)

    def reset(self, start_pts: Optional[float] = None) -> None:
        """Tương thích ngược:
        - Nếu có `start_pts`: tua (seek) tới mốc mới và GIỮ audio trong RAM.
        - Nếu `start_pts` là None: xoá sạch toàn bộ RAM (đóng phiên).
        """
        with self._lock:
            if start_pts is not None:
                self.seek(start_pts)
            else:
                self.clear()

    def clear(self) -> None:
        """Xoá sạch toàn bộ audio khi đóng phiên."""
        with self._lock:
            freed_sec = sum(len(c.pcm) for c in self._chunks) / self.sample_rate if self._chunks else 0.0
            freed_bytes = sum(c.pcm.nbytes for c in self._chunks) if self._chunks else 0
            self._chunks.clear()
            self._cursor = None
            self._session_base_pts = None
            self._gap_filled_sec = 0.0
            self._appended_sec = 0.0
            if self._has_logged_storage_start:
                self._has_logged_storage_start = False
                logger.info(
                    f"Đã giải phóng bộ nhớ RAM ({freed_sec:.1f}s audio ~{freed_bytes / (1024 * 1024):.1f}MB).",
                    extra={"module_tag": "CORE"},
                )

    # ------------------------------------------------------------------ ingress
    def append(self, pts_start: float, pcm: np.ndarray) -> None:
        """Thêm một đoạn PCM tại mốc `pts_start` (giây, tuyệt đối trên timeline video)."""
        if pcm is None or len(pcm) == 0:
            return
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)

        start = float(pts_start)
        duration = len(pcm) / self.sample_rate
        end = start + duration

        with self._lock:
            if not self._has_logged_storage_start:
                self._has_logged_storage_start = True
                logger.info(
                    f"Bắt đầu nạp luồng audio vào bộ nhớ RAM (dung lượng tối đa: {self.max_storage_sec / 3600.0:.1f}h).",
                    extra={"module_tag": "CORE"},
                )

            if self._session_base_pts is None:
                self._session_base_pts = start
                self._cursor = start
            elif self._cursor is None:
                self._cursor = self._session_base_pts

            self._appended_sec += duration
            self._insert_chunk(start, end, pcm)
            self._enforce_storage_cap()

    def _insert_chunk(self, start: float, end: float, pcm: np.ndarray) -> None:
        """Chèn chunk mới và xử lý khử trùng lặp / cắt chồng lấn với các chunk đã có."""
        if not self._chunks:
            self._chunks.append(AudioChunk(start, end, pcm))
            return

        starts = [c.pts_start for c in self._chunks]
        idx = bisect.bisect_right(starts, start)

        # 1. Kiểm tra overlap với chunk trước đó
        if idx > 0:
            prev = self._chunks[idx - 1]
            if start < prev.pts_end:
                overlap_sec = prev.pts_end - start
                overlap_samples = int(round(overlap_sec * self.sample_rate))
                if overlap_samples >= len(pcm):
                    # Chunk mới nằm trọn trong prev, không cần ghi đè
                    return
                # Cắt bớt phần trùng ở đầu
                pcm = pcm[overlap_samples:]
                start = prev.pts_end
                end = start + len(pcm) / self.sample_rate

        # 2. Kiểm tra overlap với các chunk tiếp theo
        insert_idx = idx
        while insert_idx < len(self._chunks):
            nxt = self._chunks[insert_idx]
            if end <= nxt.pts_start:
                break
            if end >= nxt.pts_end:
                # Chunk mới bao trọn nxt -> xoá nxt
                del self._chunks[insert_idx]
            else:
                # Chunk mới lẹm vào đầu nxt -> cắt phần lẹm ở đuôi chunk mới
                trim_samples = int(round((end - nxt.pts_start) * self.sample_rate))
                if trim_samples < len(pcm):
                    pcm = pcm[:-trim_samples]
                    end = nxt.pts_start
                break

        if len(pcm) > 0:
            starts = [c.pts_start for c in self._chunks]
            insert_pos = bisect.bisect_right(starts, start)
            self._chunks.insert(insert_pos, AudioChunk(start, end, pcm))

    def _enforce_storage_cap(self) -> None:
        """Chặn trên bộ nhớ RAM (mặc định 3 tiếng): bỏ các đoạn xa vị trí phát hiện tại nhất."""
        if self.max_storage_sec <= 0:
            return
        total = self.total_stored_seconds()
        if total <= self.max_storage_sec:
            return

        playhead = self._cursor if self._cursor is not None else 0.0
        while self.total_stored_seconds() > self.max_storage_sec and len(self._chunks) > 1:
            first_dist = abs(playhead - self._chunks[0].pts_start)
            last_dist = abs(self._chunks[-1].pts_end - playhead)
            if first_dist > last_dist:
                self._chunks.pop(0)
            else:
                self._chunks.pop()

    # ------------------------------------------------------------------ egress
    def buffered_end_from(self, pts: Optional[float] = None) -> Optional[float]:
        """Mốc kết thúc của vùng audio LIÊN TỤC tính từ `pts`."""
        with self._lock:
            if not self._chunks:
                return None
            target = float(pts) if pts is not None else self._cursor
            if target is None:
                return None

            frontier: Optional[float] = None
            for c in self._chunks:
                if c.pts_start <= target <= c.pts_end + 1e-4:
                    frontier = c.pts_end
                elif frontier is not None:
                    if c.pts_start <= frontier + REPORT_EPS_SEC:
                        frontier = max(frontier, c.pts_end)
                    else:
                        break
                elif target < c.pts_start:
                    if (c.pts_start - target) <= REPORT_EPS_SEC:
                        frontier = c.pts_end
                    else:
                        break
            return frontier

    def buffered_end_pts(self) -> Optional[float]:
        """Mốc kết thúc của vùng audio LIÊN TỤC kể từ con trỏ hiện tại."""
        return self.buffered_end_from(self._cursor)

    def has_audio_at(self, pts: float, min_sec: float = 0.1) -> bool:
        """Kiểm tra xem tại mốc `pts` đã có sẵn ít nhất `min_sec` audio liên tục trong RAM chưa."""
        end = self.buffered_end_from(pts)
        if end is None:
            return False
        return (end - pts) >= min_sec

    def read(self, max_sec: float) -> Optional[Tuple[float, np.ndarray]]:
        """Đọc tối đa `max_sec` giây audio liên tục kể từ con trỏ.
        
        LƯU Ý: Không xoá audio khỏi RAM, chỉ tịnh tiến con trỏ `_cursor`.

        Returns:
            (pts_start, pcm) hoặc None nếu chưa có dữ liệu để đọc.
        """
        if max_sec <= 0:
            return None
        max_samples = int(round(max_sec * self.sample_rate))

        with self._lock:
            if self._cursor is None or not self._chunks:
                return None

            cursor = self._cursor

            for chunk in self._chunks:
                # 1. Cursor nằm trong chunk này
                if chunk.pts_start <= cursor + 1e-4 < chunk.pts_end:
                    offset_sec = max(0.0, cursor - chunk.pts_start)
                    offset_samples = int(round(offset_sec * self.sample_rate))
                    available = len(chunk.pcm) - offset_samples
                    if available <= 0:
                        continue
                    take = min(available, max_samples)
                    out = chunk.pcm[offset_samples : offset_samples + take]
                    start_pts = cursor
                    self._cursor += take / self.sample_rate
                    return float(start_pts), np.ascontiguousarray(out)

                # 2. Cursor nằm trước chunk này (khe hở): lấp bằng SILENCE để bảo toàn mốc thời gian tuyệt đối
                if chunk.pts_start > cursor + 1e-4:
                    gap_sec = chunk.pts_start - cursor
                    if gap_sec <= GAP_EPS_SEC:
                        # Jitter nhỏ: nhảy thẳng tới đầu chunk
                        self._cursor = chunk.pts_start
                        return self.read(max_sec)
                    gap_samples = max(1, min(int(round(gap_sec * self.sample_rate)), max_samples))
                    silence = np.zeros(gap_samples, dtype=np.float32)
                    start_pts = cursor
                    self._cursor += gap_samples / self.sample_rate
                    self._gap_filled_sec += gap_samples / self.sample_rate
                    return float(start_pts), silence

            return None
