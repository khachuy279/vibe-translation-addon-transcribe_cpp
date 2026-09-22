"""Base Classes và Data Structures cho Module Voice Activity Detection (VAD).

Triết lý (xem `report/audit/19_KE_HOACH_VIET_LAI_VAD.md`):

1. **Engine sở hữu state machine.** Mọi quyết định nói/im do chính VAD (theo docs của
   từng thư viện) đưa ra qua event `START`/`END`. Processor **không** tự đoán bằng
   `is_speech` từng frame và **không** có đồng hồ im lặng riêng ⇒ không còn chuyện
   "VAD nói A, processor hiểu B".
2. **VAD là tap thụ động.** Engine chỉ ĐỌC frame PCM Int16 mà processor đưa; không
   resample, không gain, không clip, không lượng tử hoá lại. Audio tới ASR là **đúng
   byte** client gửi.
3. **Hình học frame nằm trong engine.** `frame_samples` là BƯỚC NHẢY (hop) mà processor
   phải cắt; engine tự lo cửa sổ phân tích (ví dụ FireRed: cửa sổ 400 mẫu, hop 160 mẫu)
   nên ASR không bao giờ nhận mẫu chồng lấn.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional, Tuple
import numpy as np


SUPPORTED_VAD_ENGINES = ("firered-vad", "silero-vad", "fsmn-vad")

#: Giá trị hợp lệ của `VADResult.event`.
EVENT_START = "START"
EVENT_END = "END"


@dataclass(slots=True)
class VADResult:
    """Kết quả VAD của MỘT bước nhảy (hop) âm thanh.

    Attributes:
        probability: xác suất tiếng nói thô (0..1) — chỉ để log/metric, KHÔNG dùng để
            chuyển trạng thái.
        event: `"START"`, `"END"` hoặc `None`. Đây là **nguồn sự thật duy nhất** cho
            việc mở/đóng một đoạn nói.
        lookback_frames: chỉ có ý nghĩa khi `event == "START"` — số frame ĐÃ được tiêu
            thụ trước frame hiện tại nhưng vẫn thuộc đoạn nói (do pre-padding của chính
            VAD, ví dụ `pad_start_frame` của FireRed). Processor dùng con số này để xả
            đúng phần audio đã đệm, tránh mất phụ âm đầu.
        is_speech: **bằng chứng tiếng nói của RIÊNG frame này** (ví dụ: xác suất frame ≥
            ngưỡng), KHÔNG phải trạng thái máy trạng thái của engine. Processor dùng nó
            cho chế độ `silence_duration_ms > 0` (đếm im lặng để chốt câu); ở chế độ mặc
            định docs thì event `START`/`END` mới là nguồn sự thật.
        forced: engine đánh dấu ĐÂY LÀ CẮT CƯỠNG BỨC vì trần cứng của chính nó (ví dụ
            `max_speech_frame`), không phải vì đã hết im lặng. Chỉ khi đó chế độ ghi đè
            mới được phép đóng câu ngay giữa lúc còn bằng chứng nói.
            LƯU Ý: `is_speech=True` trên một frame END **không** có nghĩa là cắt cưỡng
            bức — đó thường chỉ là frame chuyển tiếp (đã gặp thật: FireRed phát END kèm
            `p=1.00` ⇒ processor cũ đóng câu sai giữa câu, xem report/audit/21 §13).
    """
    probability: float = 0.0
    event: Optional[str] = None
    lookback_frames: int = 0
    is_speech: bool = False
    forced: bool = False

    def __iter__(self):
        yield self.probability
        yield self.event
        yield self.lookback_frames
        yield self.is_speech

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        if idx == 0:
            return self.probability
        if idx == 1:
            return self.event
        if idx == 2:
            return self.lookback_frames
        if idx == 3:
            return self.is_speech
        raise IndexError("VADResult index out of range")


@dataclass
class VADStreamState:
    """Trạng thái cách ly theo từng phiên âm thanh (session-safe).

    `engine_state` là state RIÊNG của engine và hoàn toàn opaque với processor (ví dụ:
    `FireRedStreamVad` giữ model caches + postprocessor; Silero giữ `VADIterator`; FSMN
    giữ `cache` dict). `None` = chưa khởi tạo; engine tự dựng ở frame kế tiếp.
    """

    engine_state: Optional[Any] = None

    #: Buffer thô để cắt frame theo hop.
    raw_buffer: bytearray = field(default_factory=bytearray)

    #: Vòng đệm pre-roll: các frame đã tiêu thụ nhưng chưa gửi ASR (chỉ trong lúc im lặng).
    #: `maxlen` do processor đặt = `engine.max_lookback_frames`.
    pre_roll: Deque[Tuple[bytes, float]] = field(default_factory=deque)

    is_speech: bool = False
    total_samples_processed: int = 0

    #: Tổng số mẫu tại frame CUỐI CÙNG có bằng chứng tiếng nói (`VADResult.is_speech`).
    #: Dùng cho chế độ `silence_duration_ms > 0`: processor tự chốt END sau đúng
    #: `silence_duration_ms` im lặng, không phụ thuộc việc engine có tôn trọng tham số hay không.
    last_speech_sample: int = 0

    #: Engine ĐÃ muốn đóng đoạn nhưng bị chế độ ghi đè hoãn lại (chưa đủ `silence_duration_ms`).
    #: Nếu người dùng gạt VAD Silence về 0 giữa câu, frame kế tiếp sẽ tôn trọng quyết định này
    #: thay vì để câu treo.
    engine_end_pending: bool = False

    #: Đã ghi log "END của engine bị hoãn" cho đoạn nói hiện tại chưa (tránh spam mỗi frame).
    engine_end_logged: bool = False

    #: Mốc thời gian (giây, theo đồng hồ client) của BYTE ĐẦU TIÊN đang nằm trong
    #: `raw_buffer`. Cần thiết vì một frame có thể vắt qua nhiều chunk client gửi (ví dụ
    #: hop 60 ms của FSMN với chunk 20 ms) — nếu lấy `capture_timestamp` của chunk hiện
    #: tại thì timestamp của frame bị lệch.
    stream_ts: float = 0.0

    def reset(self) -> None:
        """Reset về trạng thái rỗng (state engine sẽ được dựng lại ở frame kế tiếp)."""
        self.engine_state = None
        self.raw_buffer.clear()
        self.pre_roll.clear()
        self.is_speech = False
        self.total_samples_processed = 0
        self.last_speech_sample = 0
        self.engine_end_pending = False
        self.engine_end_logged = False
        self.stream_ts = 0.0


class BaseVADEngine(ABC):
    """Abstract Base Class cho các Engine VAD.

    Hợp đồng tối thiểu — engine nào cũng phải:
      * công bố `frame_samples` (hop) để processor cắt frame ĐÚNG nhịp của model;
      * công bố `max_lookback_frames` (trần pre-roll suy ra từ config của chính nó);
      * cấp state riêng cho mỗi phiên qua `create_initial_state()`;
      * trả `VADResult.event` khi mở/đóng đoạn nói.
    """

    name: str = ""
    default_threshold: float = 0.5
    #: BƯỚC NHẢY (hop) tính bằng mẫu @16 kHz mà processor phải cắt.
    frame_samples: int = 160
    #: Trần số frame pre-roll (đệm trước START) mà engine có thể yêu cầu xả lại.
    max_lookback_frames: int = 0

    @classmethod
    def prepare_files(cls) -> None:
        """QWEN-Q2: tải các file model cần thiết — **KHÔNG** dựng engine.

        Được `VADEngineFactory.get_engine()` gọi **TRƯỚC** khi lấy lock cấp lớp, vì
        `hf_hub_download`/`snapshot_download` có thể mất hàng phút khi mạng chậm.

        Mặc định no-op cho engine không cần tải file (ví dụ Silero lấy model từ package).
        """
        return None

    @abstractmethod
    def create_initial_state(
        self,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADStreamState:
        """Tạo trạng thái ban đầu cho 1 phiên stream mới.

        `silence_ms = None` ⇒ dùng đúng giá trị im lặng mặc định trong config của engine
        (đúng docs). Ngược lại engine chiếu `silence_ms` xuống field native của mình.
        """
        pass

    @abstractmethod
    def is_speech(
        self,
        frame_int16: np.ndarray,
        state: VADStreamState,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADResult:
        """Xử lý ĐÚNG `frame_samples` mẫu PCM Int16 và cập nhật state.

        ⚠️ `frame_int16` là **view chỉ-đọc** trên buffer của processor: engine TUYỆT ĐỐI
        không được sửa tại chỗ (đó là audio sẽ tới ASR). Mọi chuyển đổi
        (Int16 → Float32/Int16 chuẩn hoá) chỉ được tạo **bản sao** cho model.
        """
        pass
