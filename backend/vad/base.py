"""Base Classes và Data Structures cho Module Voice Activity Detection (VAD)."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np


SUPPORTED_VAD_ENGINES = ("firered-vad", "silero-vad", "fsmn-vad")


@dataclass(slots=True)
class VADResult:
    """Kết quả phân tích VAD của 1 frame âm thanh."""
    is_speech: bool
    probability: float
    event: Optional[str] = None  # 'START', 'END', hoặc None

    def __iter__(self):
        yield self.is_speech
        yield self.probability
        yield self.event

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        if idx == 0:
            return self.is_speech
        elif idx == 1:
            return self.probability
        elif idx == 2:
            return self.event
        raise IndexError("VADResult index out of range")


@dataclass
class VADStreamState:
    """Trạng thái nội bộ cách ly theo từng session âm thanh."""
    is_speech: bool = False
    silence_samples: int = 0
    total_samples_processed: int = 0

    # Buffer đệm thô cho việc cắt frame
    raw_buffer: bytearray = field(default_factory=bytearray)

    # Ring buffer chứa các frame trước khi nói (pre-speech buffer)
    pre_speech_ring: Deque[Tuple[bytes, float]] = field(default_factory=deque)

    # State riêng cho FireRed-VAD
    firered_postprocessor: Optional[Any] = None
    firered_caches: Optional[Any] = None
    # Cửa sổ trượt 25 ms (400 samples) dùng để tạo ĐÚNG 1 frame mỗi `frame_hop` samples.
    # Xem `backend/vad/engines/firered.py` + scratch/vad_frame_ab.py: model cần 1 frame
    # mỗi 10 ms (FRAME_SHIFT_SAMPLE=160), không phải mỗi 25 ms.
    firered_window: Optional[Any] = None

    # State riêng cho Silero VAD
    silero_iterator: Optional[Any] = None
    silero_model: Optional[Any] = None
    silero_probe: Optional[Any] = None

    # State riêng cho FSMN-VAD
    fsmn_cache: Optional[Dict[str, Any]] = None
    fsmn_in_speech: bool = False

    def reset(self) -> None:
        """Reset toàn bộ bộ đệm và trạng thái."""
        self.is_speech = False
        self.silence_samples = 0
        self.total_samples_processed = 0

        self.raw_buffer.clear()
        self.pre_speech_ring.clear()

        if self.firered_postprocessor is not None:
            self.firered_postprocessor.reset()
        self.firered_caches = None
        self.firered_window = None

        if self.silero_iterator is not None:
            self.silero_iterator.reset_states()
        if self.silero_probe is not None:
            self.silero_probe.last_prob = 0.0

        if self.fsmn_cache is not None:
            self.fsmn_cache.clear()
        self.fsmn_cache = None
        self.fsmn_in_speech = False


class BaseVADEngine(ABC):
    """Abstract Base Class cho các Engine VAD."""

    name: str = ""
    default_threshold: float = 0.45
    native_frame_samples: int = 400  # Số sample cho 1 bước tính (mặc định 25ms @ 16kHz)

    @classmethod
    def prepare_files(cls) -> None:
        """QWEN-Q2: tải các file model cần thiết — **KHÔNG** dựng engine.

        Được `VADEngineFactory.get_engine()` gọi **TRƯỚC** khi lấy lock cấp lớp, vì
        `hf_hub_download`/`snapshot_download` có thể mất hàng phút khi mạng chậm. Bản cũ
        để việc tải nằm *trong* constructor, tức là *trong* lock cấp lớp ⇒ mọi đường chỉ
        cần ĐỌC (`is_cached`, `peek_engine` — có trên event loop) bị chặn theo, treo cả
        backend.

        Mặc định no-op cho engine không cần tải file (ví dụ Silero lấy model từ package).
        """
        return None

    @abstractmethod
    def create_initial_state(self, threshold: Optional[float] = None) -> VADStreamState:
        """Tạo trạng thái ban đầu cho 1 session stream mới."""
        pass

    @abstractmethod
    def is_speech(
        self,
        chunk_float32: Optional[np.ndarray],
        state: VADStreamState,
        threshold: float,
        chunk_raw: Optional[bytes] = None,
    ) -> VADResult:
        """Xử lý 1 frame âm thanh và cập nhật state (hỗ trợ cả float32 và raw Int16 PCM bytes)."""
        pass
