"""Package core chứa các thành phần cốt lõi của pipeline Backend.

Phân tầng theo pipeline (2026-10-06):

* **Pipeline A** (`/ws`, streaming VAD + CommitManager) — import NGAY ở đây:
  `CircularAudioBuffer`, `SpeechNormalizer`, `CommitManager`, `CommitDeduplicator`,
  `metrics`/`MetricsCollector`, `pipeline_events`.
* **Pipeline B** (Lookahead offline-batch: Qwen3-ASR + Forced Aligner) — export **LƯỜI**:
  `StreamDemuxer`, `ContinuousAudioTimeline`, `LookaheadChunker`, `VADSilenceScanner`
  (kèm các dataclass `StreamAudioChunk`, `ChunkSlice`, `SilenceGap`).

VÌ SAO LƯỜI (PEP 562): `backend/core/stream_demuxer.py` import `av` ở cấp module. Nếu import ngay
trong `__init__` này thì MỌI `from backend.core.<bất kỳ> import ...` — kể cả của Pipeline A — đều
buộc phải có `av` cài đặt. Tên lười chỉ được nạp khi thật sự truy cập
(`from backend.core import StreamDemuxer`).

Các module chỉ dùng qua đường dẫn đầy đủ (không export ở đây): `gpu_scheduler`, `hypothesis`,
`heartbeat`. Dùng `from backend.core.gpu_scheduler import gpu_arbiter` như trước.
"""

from backend.core.audio_buffer import CircularAudioBuffer
from backend.core.normalizer import SpeechNormalizer, NormalizationResult
from backend.core.metrics import metrics, MetricsCollector
from backend.core.dedup import CommitDeduplicator, normalize_for_dedup
from backend.core.commit_manager import CommitManager, count_content_tokens, is_cjk
from backend.core.pipeline_events import (
    AudioChunk,
    SpeechSegment,
    ASRTranscript,
    TranslationItem,
    TTSItem,
    VADState,
    CommitReason,
)

#: Tên công khai của Pipeline B -> module thật chứa nó (nạp khi truy cập lần đầu).
_LAZY_PIPELINE_B = {
    "StreamDemuxer": "backend.core.stream_demuxer",
    "StreamAudioChunk": "backend.core.stream_demuxer",
    "ContinuousAudioTimeline": "backend.core.lookahead_timeline",
    "LookaheadChunker": "backend.core.lookahead_chunker",
    "ChunkSlice": "backend.core.lookahead_chunker",
    "VADSilenceScanner": "backend.core.vad_silence",
    "SilenceGap": "backend.core.vad_silence",
}

__all__ = [
    "CircularAudioBuffer",
    "SpeechNormalizer",
    "NormalizationResult",
    "metrics",
    "MetricsCollector",
    "CommitDeduplicator",
    "normalize_for_dedup",
    "CommitManager",
    "count_content_tokens",
    "is_cjk",
    "AudioChunk",
    "SpeechSegment",
    "ASRTranscript",
    "TranslationItem",
    "TTSItem",
    "VADState",
    "CommitReason",
    # Pipeline B (nạp lười — xem `_LAZY_PIPELINE_B`).
    *sorted(_LAZY_PIPELINE_B),
]


def __getattr__(name: str):
    """Nạp lười tên của Pipeline B (PEP 562)."""
    module_path = _LAZY_PIPELINE_B.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache: lần sau không đi qua __getattr__ nữa
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_PIPELINE_B))
