"""ContextManager: Quản lý bộ nhớ ngữ cảnh dịch (Sliding Window FIFO)."""

from collections import deque
from typing import Deque, List, Tuple


class ContextManager:
    """Quản lý cửa sổ trượt N câu dịch gần nhất để bổ trợ ngữ cảnh."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._history: Deque[Tuple[str, str]] = deque(maxlen=window_size)
        # B6-3 (Hy3): cache chuỗi ngữ cảnh đã format. `get_context_str()` được gọi cho MỖI
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

