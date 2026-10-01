"""ContinuousAudioTimeline — Dòng âm thanh Lookahead LIÊN TỤC theo PTS tuyệt đối.

Nhiệm vụ: biến các đoạn PCM rời rạc (mỗi đoạn là kết quả giải mã một mảnh MSE) thành
MỘT dòng mẫu liên tục, đúng thứ tự thời gian, để đưa vào đúng chuỗi xử lý của Pipeline A
(VAD -> ASR -> Translate).

Nguyên tắc:

1. **Không bao giờ bỏ audio**: đoạn đến muộn/chồng lấn bị CẮT phần đã đọc, không bị vứt.
2. **Khe hở được lấp bằng im lặng (silence)**: MSE append theo đúng thứ tự thời gian nên
   vùng thiếu là thiếu thật. Lấp silence giữ cho ánh xạ
   ``pts_tuyệt_đối = base_pts + sample / sample_rate`` LUÔN đúng — đây là điều kiện sống
   còn để phụ đề khớp chính xác ``video.currentTime``.
3. **Đọc theo nhu cầu (demand-driven)**: `read()` chỉ trả tối đa `max_sec` mỗi lần để tầng
   điều phối không đẩy audio đi quá xa vị trí phát hiện tại (tránh đốt GPU vô ích).
"""

from __future__ import annotations

import threading
from collections import deque
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


class ContinuousAudioTimeline:
    """Ghép các đoạn PCM rời rạc thành dòng mẫu liên tục kèm mốc PTS tuyệt đối."""

    def __init__(self, sample_rate: int = 16000, max_pending_sec: float = 90.0):
        self.sample_rate = int(sample_rate)
        self.max_pending_sec = float(max_pending_sec)

        self._lock = threading.RLock()
        #: deque[[pts_start, np.ndarray, offset_mẫu_đã_đọc]]
        self._pending: deque = deque()
        self._cursor: Optional[float] = None      # PTS tuyệt đối của mẫu kế tiếp sẽ đọc
        self._session_base_pts: Optional[float] = None  # PTS ứng với sample 0 của phiên ASR
        self._gap_filled_sec: float = 0.0
        self._appended_sec: float = 0.0

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

    def pending_seconds(self) -> float:
        """Tổng độ dài audio đang chờ đọc."""
        with self._lock:
            total = 0.0
            for pts, pcm, off in self._pending:
                total += max(0.0, (len(pcm) - off) / self.sample_rate)
            return total

    def reset(self, start_pts: Optional[float] = None) -> None:
        """Xoá sạch (tua video). `start_pts` = mốc video mới, None = neo vào đoạn kế tiếp."""
        with self._lock:
            self._pending.clear()
            self._cursor = float(start_pts) if start_pts is not None else None
            self._session_base_pts = float(start_pts) if start_pts is not None else None
            self._gap_filled_sec = 0.0
            self._appended_sec = 0.0

    # ------------------------------------------------------------------ ingress
    def append(self, pts_start: float, pcm: np.ndarray) -> None:
        """Thêm một đoạn PCM tại mốc `pts_start` (giây, tuyệt đối trên timeline video)."""
        if pcm is None or len(pcm) == 0:
            return
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)

        with self._lock:
            if self._session_base_pts is None:
                self._session_base_pts = float(pts_start)
                self._cursor = float(pts_start)
            elif self._cursor is None:
                self._cursor = self._session_base_pts

            start = float(pts_start)
            end = start + len(pcm) / self.sample_rate

            cursor = self._cursor if self._cursor is not None else start
            if end <= cursor + 1e-6:
                # Đoạn hoàn toàn thuộc phần đã đọc ⇒ bỏ (không phải mất audio).
                return

            if start < cursor:
                trim = int(round((cursor - start) * self.sample_rate))
                trim = max(0, min(trim, len(pcm)))
                pcm = pcm[trim:]
                start = cursor
                if pcm.size == 0:
                    return

            # Chèn theo thứ tự PTS (thực tế gần như luôn là append cuối).
            self._pending.append([start, pcm, 0])
            self._pending = deque(sorted(self._pending, key=lambda item: item[0]))
            self._appended_sec += len(pcm) / self.sample_rate
            self._enforce_cap()

    def _enforce_cap(self) -> None:
        """Chặn trên bộ đệm chờ (RAM) — bỏ các đoạn XA NHẤT về phía trước."""
        limit = self.max_pending_sec
        if limit <= 0:
            return
        while self.pending_seconds() > limit and len(self._pending) > 1:
            dropped = self._pending.pop()
            # logger.warning(
            #     f"ContinuousAudioTimeline: bỏ đoạn chờ xa nhất @{dropped[0]:.2f}s "
            #     f"(vượt trần {limit:.0f}s) — audio gần vị trí phát vẫn nguyên vẹn.",
            #     extra={"module_tag": "WS"},
            # )

    # ------------------------------------------------------------------ egress
    def buffered_end_pts(self) -> Optional[float]:
        """Mốc kết thúc của vùng audio LIÊN TỤC (không tính khe hở)."""
        with self._lock:
            if self._cursor is None or not self._pending:
                return None
            frontier = self._cursor
            for pts, pcm, off in self._pending:
                remaining = (len(pcm) - off) / self.sample_rate
                if remaining <= 0:
                    continue
                if pts > frontier + REPORT_EPS_SEC:
                    break
                frontier = max(frontier, pts + remaining)
            return frontier

    def read(self, max_sec: float) -> Optional[Tuple[float, np.ndarray]]:
        """Đọc tối đa `max_sec` giây audio liên tục kể từ con trỏ.

        Returns:
            (pts_start, pcm) hoặc None nếu chưa có dữ liệu để đọc.
        """
        if max_sec <= 0:
            return None
        max_samples = int(max_sec * self.sample_rate)

        with self._lock:
            if self._cursor is None:
                return None

            # 1. Bỏ các mảnh đã đọc hết.
            while self._pending and self._pending[0][2] >= len(self._pending[0][1]):
                self._pending.popleft()
            if not self._pending:
                return None

            head = self._pending[0]
            pts, pcm, off = head

            # 2. Khe hở: lấp bằng SILENCE để bảo toàn mốc thời gian tuyệt đối.
            if pts > self._cursor + GAP_EPS_SEC:
                gap_sec = pts - self._cursor
                gap_samples = max(1, min(int(gap_sec * self.sample_rate), max_samples))
                self._cursor += gap_samples / self.sample_rate
                self._gap_filled_sec += gap_samples / self.sample_rate
                return float(pts - gap_sec), np.zeros(gap_samples, dtype=np.float32)

            available = len(pcm) - off
            take = min(available, max_samples)
            if take <= 0:
                return None
            out = pcm[off:off + take]
            head[2] = off + take
            start_pts = self._cursor
            self._cursor += take / self.sample_rate
            if head[2] >= len(pcm):
                self._pending.popleft()
            return float(start_pts), np.ascontiguousarray(out)
