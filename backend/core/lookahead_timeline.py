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
   - Khe hở lớn (> 5.0s, do tua nhảy cóc): DỪNG đọc tại mép khe hở — `get_audio_range` trả về phần
     đã gom được (NGẮN hơn `duration_sec` yêu cầu), và chỉ trả `None` khi chưa gom được gì (khe hở
     nằm ngay đầu vùng đọc). Việc cắt ngắn này được log WARNING + đếm metric
     `lookahead.read_truncated_by_gap`, vì caller (`LookaheadChunker`) tính `pts_end` theo
     `len(pcm)` nên khối sẽ ngắn đi mà không có dấu vết nào khác.
"""

from __future__ import annotations

import bisect
import threading
from typing import List, Optional, Tuple

import numpy as np

from backend.core.metrics import metrics_collector
from backend.utils.logger import get_logger

logger = get_logger("core.lookahead_timeline")

#: Hai đoạn lệch nhau dưới mức này coi như LIỀN nhau khi ĐỌC (sai số lượng tử hóa PTS 1 ms
#: + mép cluster). Lớn hơn ⇒ lấp silence để bảo toàn mốc thời gian tuyệt đối.
GAP_EPS_SEC = 0.02
#: Ngưỡng "liền nhau" khi BÁO CÁO vùng đệm liên tục — dùng cho `pending_seconds`.
#: LƯU Ý: `buffered_end_from` KHÔNG dùng hằng số này mà dùng `MAX_GAP_FILL_SEC`, vì vùng "đọc được"
#: rộng hơn vùng "liền mạch": các khe nhỏ đã được lấp bằng silence khi đọc (xem `get_audio_range`).
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
    def gap_filled_sec(self) -> float:
        return self._gap_filled_sec

    def total_stored_seconds(self) -> float:
        """Tổng thời lượng audio (giây) đang lưu trữ trong RAM."""
        with self._lock:
            return sum(len(c.pcm) for c in self._chunks) / self.sample_rate

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
        """Mốc kết thúc của vùng audio ĐỌC ĐƯỢC LIÊN TỤC tính từ `pts`.

        Khe hở nhỏ hơn `MAX_GAP_FILL_SEC` KHÔNG cắt vùng này, vì `get_audio_range`/`read` đã lấp
        chúng bằng silence (xem `MAX_GAP_FILL_SEC`) ⇒ vẫn đọc ra được audio liên tục.

        SỰ CỐ THẬT 2026-10-04: hàm này trước đây cắt vùng tại khe hở > `REPORT_EPS_SEC` (0.15 s).
        Bộ giải mã MSE tăng dần sinh ra các khe ~0.2 s, nên **chỉ một khe 0.2 s là đủ để hàm báo
        "chỉ có 1 s audio phía trước" trong khi RAM đang giữ 30 s**. Hệ quả nhìn thấy trong log:
        bộ cắt khối bị kẹp `available_sec` xuống ~9–18 s nên chỉ cắt khối **8 s** dù đệm trước 42 s:

            [SEG_BATCH] Chọn khối: dài 8.1s (mục tiêu 6.3s, dải tìm tới 8.3s, bộ đệm trước 43.3s)

        Đo lại bằng `tlprobe`: 30 mảnh 1 s cách nhau 0.2 s ⇒ `buffered_end_from(0)` cũ = **1.0 s**
        trong khi `total_stored_seconds()` = 30.0 s. Sau khi sửa = 30.0 s (khớp phần đọc được thật).
        """
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
                    if c.pts_start <= frontier + MAX_GAP_FILL_SEC:
                        frontier = max(frontier, c.pts_end)
                    else:
                        break
                elif target < c.pts_start:
                    if (c.pts_start - target) <= MAX_GAP_FILL_SEC:
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

    def first_audio_pts_at_or_after(self, pts: float) -> Optional[float]:
        """Mốc BẮT ĐẦU của đoạn audio đầu tiên có chứa `pts` hoặc nằm sau `pts`.

        Dùng để đưa CON TRỎ ĐỌC ra khỏi KHE HỞ (xem `_nudge_cursor_out_of_gap` phía backend):
        khi khe hở lớn hơn `MAX_GAP_FILL_SEC`, `buffered_end_from(pts)` trả `None` và
        `next_chunk()` trả `None` MÃI MÃI — pipeline đứng im dù RAM có hàng chục giây audio ở
        ngay sau khe. Đo thật 2026-10-05 (xvideos.com, sau khi tua): con trỏ 322.07s, audio thật
        bắt đầu muộn hơn ⇒ "Đã dịch: 0.0s" suốt 35 giây.

        Trả `None` nếu không có đoạn nào kết thúc sau `pts`.
        """
        with self._lock:
            for chunk in self._chunks:
                if chunk.pts_end <= pts:
                    continue
                # Đoạn này chứa `pts` ⇒ audio có ngay tại mốc đó (trả về chính `pts`).
                if chunk.pts_start <= pts:
                    return float(pts)
                return float(chunk.pts_start)
            return None

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

    def get_audio_range(
        self,
        start_pts: float,
        duration_sec: float,
    ) -> Optional[Tuple[float, np.ndarray]]:
        """Lấy một khối audio liên tục từ `start_pts` với độ dài tối đa `duration_sec` mà KHÔNG di chuyển `_cursor`.

        Tự động ghép nối qua nhiều chunk trong RAM và lấp các khe hở nhỏ (<= MAX_GAP_FILL_SEC) bằng silence.

        ⚠️ Khe hở LỚN hơn `MAX_GAP_FILL_SEC` làm hàm trả về kết quả NGẮN hơn `duration_sec`
        (dừng ở mép khe hở), không phải `None` — xem chú thích tại nhánh `break` bên dưới.

        Returns:
            (pts_start, pcm) hoặc None nếu không có audio tại `start_pts`.
        """
        if duration_sec <= 0:
            return None

        with self._lock:
            if not self._chunks:
                return None

            target_start = float(start_pts)
            target_end = target_start + float(duration_sec)
            target_samples = int(round(duration_sec * self.sample_rate))

            pieces: List[np.ndarray] = []
            curr_pts = target_start

            for chunk in self._chunks:
                if chunk.pts_end <= curr_pts:
                    continue
                if chunk.pts_start >= target_end:
                    break

                # Nếu có khoảng trống giữa curr_pts và chunk.pts_start
                if chunk.pts_start > curr_pts:
                    gap_sec = chunk.pts_start - curr_pts
                    if gap_sec > MAX_GAP_FILL_SEC:
                        # Khe hở quá lớn (lỗ tua seek hole) -> dừng ở mép khe hở.
                        # ⚠️ KHÔNG trả None ở đây: nếu đã gom được audio thì trả phần đã gom (NGẮN
                        # hơn `duration_sec` yêu cầu) — người gọi (`LookaheadChunker`) tính
                        # `pts_end` theo `len(pcm)` nên khối sẽ ngắn đi. Log + metric để việc ngắn
                        # đi này KHÔNG còn im lặng (trước 2026-10-06 nó hoàn toàn vô hình).
                        if pieces:
                            got_sec = sum(len(p) for p in pieces) / self.sample_rate
                            logger.warning(
                                f"Vùng đọc bị CẮT NGẮN tại khe hở {gap_sec:.2f}s "
                                f"(> MAX_GAP_FILL_SEC {MAX_GAP_FILL_SEC:.1f}s): yêu cầu "
                                f"{duration_sec:.2f}s từ {target_start:.2f}s nhưng chỉ đọc được "
                                f"{got_sec:.2f}s (dừng ở {curr_pts:.2f}s, audio kế tiếp bắt đầu ở "
                                f"{chunk.pts_start:.2f}s).",
                                extra={"module_tag": "CORE"},
                            )
                            metrics_collector.increment_counter("lookahead.read_truncated_by_gap")
                        break
                    gap_samples = int(round(gap_sec * self.sample_rate))
                    if gap_samples > 0:
                        pieces.append(np.zeros(gap_samples, dtype=np.float32))
                        curr_pts += gap_sec

                # Lấy phần PCM của chunk nằm trong [curr_pts, target_end]
                offset_sec = max(0.0, curr_pts - chunk.pts_start)
                offset_samples = int(round(offset_sec * self.sample_rate))
                end_sec = min(chunk.pts_end, target_end)
                take_sec = max(0.0, end_sec - (chunk.pts_start + offset_sec))
                take_samples = int(round(take_sec * self.sample_rate))

                if take_samples > 0 and offset_samples < len(chunk.pcm):
                    actual_take = min(take_samples, len(chunk.pcm) - offset_samples)
                    pieces.append(chunk.pcm[offset_samples : offset_samples + actual_take])
                    curr_pts += actual_take / self.sample_rate

                if curr_pts >= target_end - 1e-4:
                    break

            if not pieces:
                return None

            result_pcm = np.concatenate(pieces)
            if len(result_pcm) > target_samples:
                result_pcm = result_pcm[:target_samples]

            return float(target_start), np.ascontiguousarray(result_pcm)

