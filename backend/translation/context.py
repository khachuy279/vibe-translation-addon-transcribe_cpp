"""ContextManager: Quản lý bộ nhớ ngữ cảnh dịch cho Pipeline A (Sliding Window FIFO).

⚠️ PHẠM VI ÁP DỤNG:
- CHỈ ÁP DỤNG CHO PIPELINE A (Realtime Streaming đơn câu).
- Được bật/tắt và điều chỉnh qua `use_context: bool = False` và `context_window: int = 3`
  trong `backend/config.py` (TranslationConfig).
- Pipeline B (Lookahead) KHÔNG sử dụng ContextManager này vì đã sử dụng tính năng
  Batch Context Translation (dịch gộp toàn bộ các câu trong khối ASR qua JSON instTrans)
  vốn đã có toàn bộ ngữ cảnh liền mạch tự nhiên và tiết kiệm GPU hơn.
"""

from collections import deque
from typing import Deque, List, Tuple


class ContextManager:
    """Quản lý cửa sổ trượt N câu dịch gần nhất để bổ trợ ngữ cảnh cho Pipeline A."""

    def __init__(self, window_size: int = 3):
        self.window_size = int(window_size)
        self._history: Deque[Tuple[str, str]] = deque(maxlen=self.window_size)
        # B6-3: cache chuỗi ngữ cảnh đã format. `get_context_str()` được gọi cho MỖI
        # câu dịch khi `translation.use_context = True`; bản cũ ghép lại chuỗi từ deque mỗi
        # lần. Lịch sử chỉ đổi ở `add()`/`clear()` nên chỉ cần dựng lại khi có thay đổi.
        # Mặc định `use_context = False` (handler không gọi hàm này) nên cache thuần tuý là
        # tối ưu cho cấu hình có bật ngữ cảnh — không đổi hành vi ở cấu hình mặc định.
        self._context_cache: str = ""
        self._context_dirty: bool = False

    def add(self, source_text: str, translated_text: str) -> None:
        """Thêm 1 cặp câu (gốc, dịch) vào lịch sử."""
        if source_text and translated_text:
            self._history.append((source_text.strip(), translated_text.strip()))
            self._context_dirty = True

    def get_context_str(self) -> str:
        """Tạo chuỗi ngữ cảnh kết hợp các câu gần nhất (có cache)."""
        if not self._context_dirty:
            return self._context_cache
        if not self._history:
            self._context_cache = ""
        else:
            self._context_cache = " | ".join(
                f"{src} -> {tgt}" for src, tgt in self._history
            )
        self._context_dirty = False
        return self._context_cache

    def clear(self) -> None:
        """Xóa sạch ngữ cảnh."""
        self._history.clear()
        self._context_dirty = True


# Alias tương thích
TranslationContextTracker = ContextManager
