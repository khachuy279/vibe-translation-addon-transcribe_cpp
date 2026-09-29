"""CommitManager: Bộ điều phối ngắt câu và chốt câu đa tầng.

Thực thi 4 bậc ưu tiên:
1. BẬC 1: VAD Silence (Khoảng lặng kết thúc câu từ VAD).
2. BẬC 2: Max Duration / Max Chars Safeguard (Giới hạn tối đa 8.0s / 150 ký tự).
3. BẬC 3: Stability Prefix Split (Chốt câu khi preview text ổn định qua nhiều chu kỳ poll).
4. BẬC 4: Inactivity Timeout Force-Commit (Tự động ép chốt câu sau 1.2s tránh treo).

Kèm theo:
- Bộ đếm từ chuẩn hỗ trợ cả chữ Latin và ký tự CJK (Trung, Nhật, Hàn).
- Lọc câu ngắn (min_words_to_commit) loại bỏ âm thanh ậm ừ.
- Chống trùng lặp 3 lớp (CommitDeduplicator).
"""

import logging
import re
import time
from typing import Optional, Tuple

from backend.config import config, SentenceConfig
from backend.core.dedup import CommitDeduplicator
from backend.core.pipeline_events import CommitReason
from backend.utils.logger import logger

_RE_LATIN_WORDS = re.compile(r"\b[\w'-]+\b")


def is_cjk(char: str) -> bool:
    """Kiểm tra ký tự có thuộc bảng chữ cái CJK (Trung, Nhật, Hàn) hay không."""
    if not char:
        return False
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF) or   # CJK Unified Ideographs
        (0x3040 <= cp <= 0x309F) or   # Hiragana
        (0x30A0 <= cp <= 0x30FF) or   # Katakana
        (0x1100 <= cp <= 0x11FF) or   # Hangul Jamo
        (0xAC00 <= cp <= 0xD7AF)      # Hangul Syllables
    )


def count_content_tokens(text: str) -> int:
    """Đếm số lượng từ/token có nghĩa, hỗ trợ chuẩn xác cả tiếng Latin và CJK."""
    if not text or not text.strip():
        return 0

    cjk_count = 0
    non_cjk_chars = []
    for ch in text:
        if is_cjk(ch):
            cjk_count += 1
        else:
            non_cjk_chars.append(ch)

    latin_text = "".join(non_cjk_chars)
    latin_words = len(_RE_LATIN_WORDS.findall(latin_text)) if latin_text else 0

    return cjk_count + latin_words


class CommitManager:
    """Bộ điều phối quản lý quyết định chốt câu của một Session."""

    def __init__(self, sentence_cfg: Optional[SentenceConfig] = None):
        self.cfg = sentence_cfg or config.sentence
        self.deduplicator = CommitDeduplicator()

        # Trạng thái theo dõi độ ổn định của Preview Text (Stability Split)
        self._last_preview_text: str = ""
        self._stability_start_time: Optional[float] = None
        self._stable_poll_count: int = 0

        # Trạng thái thời gian hoạt động gần nhất (Inactivity Timeout)
        self._last_activity_time: float = time.time()

    def record_activity(self) -> None:
        """Ghi nhận có tín hiệu âm thanh hoặc tương tác mới."""
        self._last_activity_time = time.time()

    def reset_stability(self) -> None:
        """Reset bộ đếm ổn định giữa các câu."""
        self._last_preview_text = ""
        self._stability_start_time = None
        self._stable_poll_count = 0

    def is_text_filtered(self, text: str, min_words: Optional[int] = None) -> bool:
        """Kiểm tra nếu câu quá ngắn (dưới ngưỡng min_words_to_commit) để drop."""
        threshold = min_words if min_words is not None else self.cfg.min_words_to_commit
        if threshold <= 0:
            return False
        tokens = count_content_tokens(text)
        return tokens < threshold

    def is_duplicate(self, text: str) -> bool:
        """Kiểm tra câu có bị trùng lặp qua Deduplicator không."""
        return self.deduplicator.is_duplicate(text)

    def record_commit(self, text: str) -> None:
        """Ghi nhận câu đã commit thành công vào Deduplicator."""
        self.deduplicator.record_commit(text)
        self.reset_stability()
        self.record_activity()

    def evaluate_preview_stability(self, preview_text: str) -> bool:
        """Đánh giá xem preview text có đạt trạng thái ổn định để kích hoạt BẬC 3 (STABLE_PREFIX).
        
        Returns:
            True nếu preview text bất biến đủ lâu, cần kích hoạt commit.
        """
        if not self.cfg.split_on_stability or not preview_text or not preview_text.strip():
            self.reset_stability()
            return False

        clean_text = preview_text.strip()
        now = time.time()

        if clean_text == self._last_preview_text:
            self._stable_poll_count += 1
            if self._stability_start_time is None:
                self._stability_start_time = now

            duration_stable = now - self._stability_start_time
            if (
                self._stable_poll_count >= self.cfg.stability_threshold_polls
                and duration_stable >= self.cfg.stability_duration_sec
            ):
                return True
        else:
            self._last_preview_text = clean_text
            self._stability_start_time = now
            self._stable_poll_count = 1

        return False

    def stability_state(self) -> dict:
        """Trạng thái bộ đếm ổn định (cho log trace từng nhịp — xem `[SEG_TRACE]`).

        Trả các khoá dùng để giải thích VÌ SAO chưa chốt câu: số nhịp đã thấy text đứng im,
        số nhịp cần, thời gian đã đứng im (ms) và thời gian cần (ms). Thuần đọc — không đổi
        trạng thái, nên gọi ở đâu cũng an toàn.
        """
        start = self._stability_start_time
        stable_ms = (time.time() - start) * 1000.0 if start else 0.0
        return {
            "polls": int(self._stable_poll_count),
            "required_polls": int(self.cfg.stability_threshold_polls),
            "stable_ms": stable_ms,
            "required_ms": float(self.cfg.stability_duration_sec) * 1000.0,
            "text": self._last_preview_text,
        }

    def evaluate_inactivity_timeout(self, is_speech_active: bool) -> bool:
        """Đánh giá BẬC 4: Inactivity Timeout Force-Commit nếu session đứng yên quá 1.2s.
        
        Returns:
            True nếu cần ép commit dở dang.
        """
        if not is_speech_active:
            return False

        elapsed = time.time() - self._last_activity_time
        if elapsed >= self.cfg.inactivity_timeout_sec:
            return True
        return False

    def check_max_duration(self, utterance_duration_sec: float, text_len: int) -> bool:
        """Đánh giá BẬC 2: Giới hạn tối đa độ dài phát âm liên tục."""
        if utterance_duration_sec >= self.cfg.max_duration_sec:
            return True
        if text_len >= self.cfg.max_chars:
            return True
        return False

    def decide_commit_trigger(
        self,
        vad_silence: bool,
        duration_sec: float,
        preview_text: str,
        is_speech_active: bool,
    ) -> Optional[CommitReason]:
        """Quyết định lý do chốt câu dựa trên thứ tự 4 bậc ưu tiên rõ ràng."""
        # BẬC 1: Ưu tiên cao nhất - VAD Silence
        if vad_silence:
            return CommitReason.VAD_SILENCE

        # BẬC 2: Max duration limit safeguard
        if self.check_max_duration(duration_sec, len(preview_text)):
            return CommitReason.MAX_DURATION

        # BẬC 3: Stability prefix split
        if self.evaluate_preview_stability(preview_text):
            return CommitReason.STABLE_PREFIX

        # BẬC 4: Inactivity timeout force-commit
        if self.evaluate_inactivity_timeout(is_speech_active):
            return CommitReason.TIMEOUT_FORCE

        return None
