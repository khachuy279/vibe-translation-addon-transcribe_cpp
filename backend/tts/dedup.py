"""Module khử trùng lặp (Deduplication) cho TTS Pipeline.

Chức năng:
- Ngăn chặn phát âm lại các câu đã được phát gần đây (Exact Duplicate & Substring Overlap).
- Chuẩn hóa khoảng trắng, dấu câu trước khi so khớp hash.
- Duy trì lịch sử trượt giới hạn (sliding window) bảo toàn bộ nhớ.

QUY TẮC CHUẨN HOÁ dùng CHUNG với `core/dedup.py` và `translation/dedup.py`
(`normalize_for_dedup`). Trước đây bản ở đây chỉ bỏ dấu câu ở HAI ĐẦU, nên
"Xin chào, các bạn" và "Xin chào các bạn" không được coi là trùng.
"""

from typing import List, Set

from backend.core.dedup import normalize_for_dedup


class TTSDedupState:
    """Quản lý trạng thái chống trùng phát âm cho 1 phiên TTS."""

    def __init__(self, max_history: int = 15):
        self.max_history = max_history
        self._history: List[str] = []
        self._history_set: Set[str] = set()

    def _normalize(self, text: str) -> str:
        """Uỷ quyền cho `core.dedup.normalize_for_dedup` (một quy tắc DUY NHẤT)."""
        return normalize_for_dedup(text)

    def is_duplicate(self, text: str) -> bool:
        """Kiểm tra xem câu văn có bị trùng lặp với các câu vừa phát âm hay không."""
        norm = self._normalize(text)
        if not norm or len(norm) < 2:
            return True

        if norm in self._history_set:
            return True

        # Kiểm tra nếu câu này là tiền tố hoặc phần phụ trùng hoàn toàn với câu gần nhất
        if self._history:
            last = self._history[-1]
            if norm == last or (len(norm) >= 8 and (norm in last or last in norm)):
                return True

        # Ghi nhận vào lịch sử
        self._history.append(norm)
        self._history_set.add(norm)
        if len(self._history) > self.max_history:
            oldest = self._history.pop(0)
            if oldest not in self._history:
                self._history_set.discard(oldest)

        return False

    def reset(self) -> None:
        """Xóa sạch lịch sử dedup."""
        self._history.clear()
        self._history_set.clear()
