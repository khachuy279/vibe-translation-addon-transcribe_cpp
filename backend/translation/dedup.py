"""Bộ lọc chống dịch trùng lặp cho luồng Translation Worker.

FIX-05 (2026-02): trước đây câu trùng bị **bỏ qua im lặng** (`return` không gửi gì).
Client đã nhận `utterance_update(is_final=True, translated="...")` ⇒ phụ đề gốc treo
vĩnh viễn ở dấu "・・・" cho tới khi có câu final kế tiếp. Module này nay giữ luôn
**bản dịch của câu vừa dịch** để lần trùng sau có thể TÁI DÙNG thay vì bỏ trắng.
"""

from collections import deque
import time
from typing import Deque, Dict, Optional, Tuple


class TranslationDedupState:
    """Quản lý các câu vừa dịch để ngăn chặn dịch lặp lại nhiều lần.

    Bộ nhớ gồm 2 tầng dùng CÙNG một khoá chuẩn hoá (`_key`):
      * `_history`      — mốc thời gian của các câu đã gặp (dò trùng).
      * `_translations` — bản dịch tương ứng (tái dùng khi trùng).

    Nhờ vậy `is_duplicate()` vẫn tiết kiệm được một lần suy luận GGUF, còn phụ đề
    thì không còn treo ở dấu "...".
    """

    def __init__(self, cache_ttl_sec: float = 10.0, max_items: int = 20):
        self.cache_ttl_sec = cache_ttl_sec
        self._history: Deque[Tuple[float, str]] = deque(maxlen=max_items)
        self._translations: Dict[str, Tuple[float, str]] = {}
        self._max_items = max_items

    # ------------------------------------------------------------------ nội bộ
    @staticmethod
    def _key(text: str) -> str:
        return text.strip().lower()

    def _prune(self, now: float) -> None:
        while self._history and (now - self._history[0][0] > self.cache_ttl_sec):
            self._history.popleft()
        if self._translations:
            expired = [
                k for k, (ts, _) in self._translations.items()
                if now - ts > self.cache_ttl_sec
            ]
            for k in expired:
                self._translations.pop(k, None)

    # ------------------------------------------------------------------- API
    def is_duplicate(self, text: str) -> bool:
        if not text or not text.strip():
            return True

        clean = self._key(text)
        now = time.time()
        self._prune(now)

        for _ts, prev in reversed(self._history):
            if clean == prev:
                return True

        self._history.append((now, clean))
        return False

    def remember_translation(self, text: str, translated: str) -> None:
        """Ghi nhớ bản dịch của `text` để lần trùng sau tái dùng."""
        if not text or not text.strip():
            return
        clean = self._key(text)
        now = time.time()
        # Chặn phình RAM nếu TTL bị đặt quá lớn.
        if len(self._translations) >= self._max_items and clean not in self._translations:
            oldest = min(self._translations.items(), key=lambda kv: kv[1][0])[0]
            self._translations.pop(oldest, None)
        self._translations[clean] = (now, translated)

    def cached_translation(self, text: str) -> Optional[str]:
        """Bản dịch đã có cho `text` (None nếu chưa từng dịch hoặc đã hết hạn)."""
        if not text or not text.strip():
            return None
        now = time.time()
        self._prune(now)
        hit = self._translations.get(self._key(text))
        return hit[1] if hit else None

    def clear(self) -> None:
        self._history.clear()
        self._translations.clear()


# Alias tương thích
TranslationDeduplicator = TranslationDedupState
