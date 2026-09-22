"""SEG — ghép quyết định tách câu (văn bản) với MỐC AUDIO.

Vì sao cần lớp này (khảo sát 2026-02, xem `report/audit/21_...`)
------------------------------------------------------------------
Qwen3-ASR trong `transcribe.cpp` KHÔNG có timestamp:
  `external/transcribe.cpp/src/arch/qwen3_asr/capabilities.cpp` đặt
  `max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE`; family này không đọc
  `params->timestamps`; xin `"segment"/"word"/"token"` bị trả lỗi status 12.
⇒ Không thể map "dấu câu" sang mốc thời gian trực tiếp. Cách duy nhất đúng:
  1. `SentenceCompleter` quyết định CÂU NÀO đã trọn nghĩa (thuần văn bản).
  2. `StreamingSegmenter` ghi lại LỊCH SỬ SCAN `(text, end_sample)` để khoanh vùng
     mẫu audio mà ranh giới câu rơi vào.
  3. Chọn NEO = khoảng lặng dài nhất (VAD) trong vùng đó ⇒ cắt audio ĐÚNG giữa hai
     từ, rồi CHẠY LẠI ASR trên mảnh đã đóng để lấy câu cuối cùng ổn định
     (không còn hiện tượng dấu câu đổi theo từng scan).
"""

from dataclasses import dataclass, field
import time
from typing import Callable, List, Optional, Sequence, Tuple

from backend.segmentation.boundary import SentenceCompleter, SegmentDecision, content_len

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SilenceGap:
    """Một khoảng lặng VAD (mẫu, toạ độ tuyệt đối trong stream)."""

    start_sample: int
    end_sample: int

    @property
    def duration_ms(self) -> float:
        return max(0, self.end_sample - self.start_sample) * 1000.0 / SAMPLE_RATE


@dataclass
class Scan:
    """Ảnh chụp một lần preview: text ASR và mốc audio tương ứng."""

    text: str
    end_sample: int
    at: float


@dataclass
class SegmentedSentence:
    """Câu đã chốt kèm vùng audio chứa ranh giới của nó."""

    text: str
    reason: str
    end_index: int
    confidence: float
    window: Tuple[int, int]              # (from_sample, to_sample) ranh giới rơi vào
    cut_sample: Optional[int] = None     # mốc cắt đã chọn (giữa khoảng lặng)
    anchored: bool = False               # True nếu cắt đúng vào khoảng lặng VAD
    gap_ms: float = 0.0


def choose_anchor(
    window: Tuple[int, int],
    gaps: Sequence[SilenceGap],
    *,
    min_gap_ms: float = 150.0,
    fallback_overlap_ms: float = 600.0,
) -> Tuple[int, bool, float]:
    """Chọn mốc cắt trong `window`.

    Ưu tiên 1 — NEO KHOẢNG LẶNG: khoảng lặng dài nhất (ưu tiên muộn nếu bằng nhau) đủ
    `min_gap_ms`. Cắt giữa khoảng lặng ⇒ câu trước trọn, câu sau bắt đầu từ im lặng.

    Ưu tiên 2 — CHỒNG LẤN DỰ PHÒNG: không có khoảng lặng (người nói liền mạch) thì lùi
    mốc cắt lại `fallback_overlap_ms` để vùng mới BAO GỒM cả phần ranh giới. Đo bằng ASR
    thật (`scratch/seg_ja_probe.py`): cắt đúng ở cuối vùng quét mà không chồng lấn thì
    câu sau MẤT CHỮ ĐẦU (`溶岩が浮上しやすく…` → chỉ còn `しやすくなっていました。`) vì
    vùng mới bắt đầu giữa từ. Chồng lấn rồi để tầng trim cắt phần lặp.

    Trả `(cut_sample, anchored, gap_ms)`.
    """
    lo, hi = window
    if hi <= lo:
        return hi, False, 0.0

    best: Optional[SilenceGap] = None
    for gap in gaps:
        if gap.end_sample < lo or gap.start_sample > hi:
            continue
        if gap.duration_ms < min_gap_ms:
            continue
        if best is None or gap.duration_ms >= best.duration_ms:
            best = gap
    if best is not None:
        # Cắt giữa khoảng lặng: câu trước vẫn trọn, câu sau bắt đầu từ im lặng.
        cut = (max(best.start_sample, lo) + min(best.end_sample, hi)) // 2
        return cut, True, best.duration_ms

    overlap = int(max(0.0, fallback_overlap_ms) * SAMPLE_RATE / 1000.0)
    return max(lo, hi - overlap), False, 0.0


class StreamingSegmenter:
    """Bọc `SentenceCompleter` + lịch sử scan + neo khoảng lặng."""

    def __init__(
        self,
        completer: Optional[SentenceCompleter] = None,
        *,
        history: int = 16,
        min_gap_ms: float = 150.0,
        anchor_lookback_ms: float = 500.0,
        fallback_overlap_ms: float = 600.0,
        turn_scorer: Optional[Callable[[str], Optional[float]]] = None,
        turn_threshold: float = 0.5,
        turn_min_chars: int = 8,
    ):
        self.completer = completer or SentenceCompleter()
        self.history = int(history)
        self.min_gap_ms = float(min_gap_ms)
        self.anchor_lookback_ms = float(anchor_lookback_ms)
        self.fallback_overlap_ms = float(fallback_overlap_ms)
        self.turn_scorer = turn_scorer
        self.turn_threshold = float(turn_threshold)
        self.turn_min_chars = int(turn_min_chars)

        self._scans: List[Scan] = []
        self._gaps: List[SilenceGap] = []
        self._turn_checked_at = 0

    # ------------------------------------------------------------------ nạp dữ liệu
    def note_silence(self, start_sample: int, end_sample: int) -> None:
        """VAD báo một khoảng lặng (mẫu tuyệt đối)."""
        if end_sample > start_sample:
            self._gaps.append(SilenceGap(int(start_sample), int(end_sample)))
            if len(self._gaps) > 128:
                del self._gaps[:-128]

    def recent_gaps(self) -> List[SilenceGap]:
        return list(self._gaps)

    def reset(self) -> None:
        self.completer.reset()
        self._scans.clear()
        self._gaps.clear()
        self._turn_checked_at = 0

    # ------------------------------------------------------------------ vòng lặp
    def observe(
        self,
        preview_text: str,
        end_sample: int,
        now: Optional[float] = None,
    ) -> List[SegmentedSentence]:
        """Nạp một preview mới; trả về các câu đã chốt kèm vùng audio ranh giới."""
        now = time.monotonic() if now is None else now
        text = preview_text or ""
        self._scans.append(Scan(text=text, end_sample=int(end_sample), at=now))
        if len(self._scans) > self.history:
            del self._scans[:-self.history]

        decisions = self.completer.observe(text, now)
        if not decisions and self._should_try_turn(text):
            turn = self._try_turn(text, now)
            if turn is not None:
                decisions = [turn]
        return [self._attach_audio(d) for d in decisions]

    def flush(
        self,
        preview_text: Optional[str] = None,
        end_sample: Optional[int] = None,
    ) -> List[SegmentedSentence]:
        """VAD đóng vùng nói (hoặc hết stream) ⇒ chốt phần còn lại."""
        text = preview_text
        if text is None:
            text = self._scans[-1].text if self._scans else self.completer.committed_text
        if end_sample is not None:
            self._scans.append(Scan(text=text, end_sample=int(end_sample), at=time.monotonic()))
        decisions = self.completer.flush(text)
        return [self._attach_audio(d) for d in decisions]

    # ------------------------------------------------------------------ turn detector
    def _should_try_turn(self, text: str) -> bool:
        scorer = self.turn_scorer
        if scorer is None or not text:
            return False
        tail = text[self.completer._cursor:]  # noqa: SLF001 — cùng package, tránh API thừa
        if content_len(tail) < self.turn_min_chars:
            return False
        now = time.monotonic()
        if now - self._turn_checked_at < 0.25:
            return False
        self._turn_checked_at = now
        return True

    def _try_turn(self, text: str, now: float) -> Optional[SegmentDecision]:
        scorer = self.turn_scorer
        assert scorer is not None
        cursor = self.completer._cursor  # noqa: SLF001
        tail = text[cursor:]
        try:
            score = scorer(tail)
        except Exception:  # noqa: BLE001 — model phụ trợ không được làm chết pipeline
            return None
        if score is None or score < self.turn_threshold:
            return None
        piece = tail.strip()
        if not piece:
            return None
        return self.completer._commit(text, len(text), "turn_detector", float(score))  # noqa: SLF001

    # ------------------------------------------------------------------ audio
    def _attach_audio(self, decision: SegmentDecision) -> SegmentedSentence:
        window = self._boundary_window()
        if window is None:
            start = self._scans[0].end_sample if self._scans else 0
            window = (start, start)
        cut, anchored, gap_ms = choose_anchor(
            window, self._gaps,
            min_gap_ms=self.min_gap_ms,
            fallback_overlap_ms=self.fallback_overlap_ms,
        )
        return SegmentedSentence(
            text=decision.text,
            reason=decision.reason,
            end_index=decision.end_index,
            confidence=decision.confidence,
            window=window,
            cut_sample=cut,
            anchored=anchored,
            gap_ms=gap_ms,
        )

    def _boundary_window(self) -> Optional[Tuple[int, int]]:
        """Khoanh vùng mẫu mà ranh giới câu vừa chốt rơi vào.

        Bằng chứng dùng được: scan CUỐI CÙNG mà ASR chỉ đọc ra đúng phần đã chốt (chưa
        có chữ của câu kế tiếp). Ở scan đó audio đã chứa tới `end_sample` mà ASR vẫn
        không thấy câu mới ⇒ ranh giới nằm quanh mốc đó. Lùi thêm `anchor_lookback_ms`
        để bắt được khoảng lặng ngay trước đó.
        """
        if not self._scans:
            return None
        committed = self.completer.committed_text
        last_no_tail: Optional[Scan] = None
        for scan in self._scans:
            if scan.text == committed:
                last_no_tail = scan
        current = self._scans[-1]
        if last_no_tail is None:
            if len(self._scans) >= 2:
                last_no_tail = self._scans[-2]
            else:
                return (max(0, current.end_sample - int(self.anchor_lookback_ms * SAMPLE_RATE / 1000.0)),
                        current.end_sample)
        lo = max(0, last_no_tail.end_sample - int(self.anchor_lookback_ms * SAMPLE_RATE / 1000.0))
        hi = max(lo, current.end_sample)
        return (lo, hi)
