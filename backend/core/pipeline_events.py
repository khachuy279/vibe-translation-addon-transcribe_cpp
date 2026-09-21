"""Định nghĩa các Data Classes và Sự kiện (Events) luân chuyển qua Pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np


class VADState(str, Enum):
    """Trạng thái VAD của luồng âm thanh."""
    SILENCE = "SILENCE"
    PRE_ROLL = "PRE_ROLL"
    SPEECH = "SPEECH"


class CommitReason(str, Enum):
    """Lý do chốt câu (Commit Reason) để phân tích và logging."""
    VAD_SILENCE = "VAD_SILENCE"
    MAX_DURATION = "MAX_DURATION"
    STABLE_PREFIX = "STABLE_PREFIX"
    TIMEOUT_FORCE = "TIMEOUT_FORCE"
    MANUAL = "MANUAL"
    # P4.5: hàng đợi commit đầy -> câu cũ nhất được GỘP thay vì bị vứt bỏ.
    MERGED_BACKLOG = "MERGED_BACKLOG"


@dataclass
class AudioChunk:
    """Frame âm thanh thô nhận từ WebSocket."""
    pcm_data: np.ndarray             # Mảng 1D float32 [-1.0, 1.0]
    sample_rate: int = 16000
    capture_timestamp: float = 0.0   # Timestamp từ client (Firefox)
    chunk_index: int = 0
    received_at: float = 0.0         # Timestamp nhận tại backend


@dataclass
class SpeechSegment:
    """Đoạn âm thanh chứa tiếng nói đã được VAD xác định ranh giới."""
    utterance_id: str
    pcm_data: np.ndarray             # Mảng float32 toàn vẹn (đã gồm pre-roll do CHÍNH VAD yêu cầu
                                     # và đuôi im lặng theo `min_silence` của engine)
    sample_rate: int = 16000
    start_sample_idx: int = 0
    end_sample_idx: int = 0
    duration_sec: float = 0.0
    commit_reason: CommitReason = CommitReason.VAD_SILENCE
    is_final: bool = True
    created_at: float = 0.0


@dataclass
class ASRTranscript:
    """Kết quả nhận dạng ASR từ transcribe.cpp."""
    utterance_id: str
    text: str
    is_final: bool = False
    stable_text: str = ""
    unstable_text: str = ""
    language: str = "auto"
    inference_ms: float = 0.0
    commit_reason: Optional[CommitReason] = None
    created_at: float = 0.0


@dataclass
class TranslationItem:
    """Yêu cầu và kết quả dịch câu."""
    utterance_id: str
    source_text: str
    source_lang: str = "auto"
    target_lang: str = "vi"
    translated_text: str = ""
    inference_ms: float = 0.0
    queue_wait_ms: float = 0.0
    queued_at: float = 0.0


@dataclass
class TTSItem:
    """Yêu cầu và kết quả tổng hợp giọng nói."""
    utterance_id: str
    text: str
    voice_id: str = "speaker_01_0039.wav"
    speed: float = 1.0
    audio_base64: Optional[str] = None
    duration_sec: float = 0.0
    sample_rate: int = 24000
    inference_ms: float = 0.0
    rtf: float = 0.0
    queue_wait_ms: float = 0.0
    queued_at: float = 0.0
