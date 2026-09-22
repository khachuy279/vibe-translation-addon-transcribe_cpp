"""Tầng SEG — tách câu hậu-ASR cho pipeline VAD > ASR > SEG.

Xuất API chính:
    * `split_complete_sentences` — tách văn bản đã hoàn chỉnh.
    * `SentenceCompleter`        — chốt câu trên preview ASR tăng dần.
    * `StreamingSegmenter`       — ghép quyết định văn bản với mốc audio + neo khoảng lặng.
    * `SilenceGap`, `choose_anchor` — chọn mốc cắt audio.
"""

from backend.segmentation.boundary import (
    SegmentDecision,
    SentenceCompleter,
    content_len,
    first_boundary,
    is_terminal_at,
    split_complete_sentences,
)
from backend.segmentation.segmenter import (
    Scan,
    SegmentedSentence,
    SilenceGap,
    StreamingSegmenter,
    choose_anchor,
)
__all__ = [
    "SegmentDecision",
    "SentenceCompleter",
    "SegmentedSentence",
    "SilenceGap",
    "Scan",
    "StreamingSegmenter",
    "choose_anchor",
    "content_len",
    "first_boundary",
    "is_terminal_at",
    "split_complete_sentences",
]
