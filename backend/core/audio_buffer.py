"""Circular Ring Buffer 60s (Single-Writer, Zero-Drop, Bit-Exact).

Quản lý bộ nhớ đệm âm thanh liên tục:
- Cố định dung lượng 60 giây (960,000 samples Float32 @ 16kHz).
- Mô hình Single-Writer: Ghi liên tục không khóa từ luồng WebSocket.
- Multi-Reader: Đọc snapshot lát cắt (slice) an toàn cho VAD và ASR.
- Tự động trượt khung và bảo vệ tràn buffer (Safe Drop Oldest) nếu phiên làm việc kéo dài liên tục.
- Đảm bảo độ toàn vẹn 100% mẫu âm thanh (Bit-Exact Integrity).
"""

import threading
from typing import Optional, Tuple
import numpy as np


class CircularAudioBuffer:
    """Bộ đệm vòng Circular Audio Buffer tối ưu hóa cho 1 session real-time stream."""

    def __init__(self, sample_rate: int = 16000, capacity_sec: float = 60.0):
        self.sample_rate = sample_rate
        self.capacity_samples = int(sample_rate * capacity_sec)
        
        # Mảng bộ đệm vòng float32
        self._buffer = np.zeros(self.capacity_samples, dtype=np.float32)
        
        # Con trỏ mẫu (tính theo tổng số sample đã ghi từ đầu session)
        self._total_written: int = 0
        self._lock = threading.Lock()
        
        # Thống kê
        self._dropped_samples_count: int = 0

    @property
    def total_written(self) -> int:
        """Tổng số mẫu (samples) đã ghi vào bộ đệm từ đầu phiên."""
        return self._total_written

    @property
    def dropped_samples(self) -> int:
        """Số mẫu cũ nhất đã bị đẩy ra ngoài bộ đệm khi đầy."""
        return self._dropped_samples_count

    def write(self, audio_data: np.ndarray) -> int:
        """Ghi dữ liệu float32 vào bộ đệm vòng (Single-Writer).
        
        Args:
            audio_data: Mảng 1D float32 [-1.0, 1.0].
            
        Returns:
            Số sample đã ghi thành công.
        """
        if audio_data is None or len(audio_data) == 0:
            return 0

        # Đảm bảo kiểu float32 1D
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()

        num_samples = len(audio_data)

        with self._lock:
            start_pos = self._total_written % self.capacity_samples
            end_pos = start_pos + num_samples

            if end_pos <= self.capacity_samples:
                # Ghi liên tục không bị vòng qua mép cuối mảng
                self._buffer[start_pos:end_pos] = audio_data
            else:
                # Ghi tràn qua mép cuối mảng -> chia 2 đoạn
                first_chunk_len = self.capacity_samples - start_pos
                second_chunk_len = num_samples - first_chunk_len
                
                self._buffer[start_pos:self.capacity_samples] = audio_data[:first_chunk_len]
                self._buffer[0:second_chunk_len] = audio_data[first_chunk_len:]

            self._total_written += num_samples
            
            # Cập nhật số sample cũ bị ghi đè nếu vượt quá dung lượng
            if self._total_written > self.capacity_samples:
                self._dropped_samples_count = self._total_written - self.capacity_samples

        return num_samples

    def write_bytes(self, pcm_bytes: bytes, dtype: str = "float32") -> int:
        """Ghi trực tiếp từ chuỗi bytes PCM (Float32 hoặc Int16)."""
        if not pcm_bytes:
            return 0
            
        if dtype == "float32":
            data = np.frombuffer(pcm_bytes, dtype=np.float32)
        elif dtype == "int16":
            # FIX-06b: `astype` rồi chia tạo 2 mảng tạm; chia TẠI CHỖ chỉ tạo 1.
            # Kết quả số học giống hệt (cùng phép chia IEEE trên float32).
            data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
            data /= 32768.0
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        return self.write(data)

    def get_slice(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Trích xuất một đoạn âm thanh liên tục giữa [start_sample, end_sample).
        
        Tự động xử lý an toàn nếu start_sample nằm ngoài phạm vi buffer khả dụng.
        
        Args:
            start_sample: Vị trí mẫu bắt đầu (tuyệt đối).
            end_sample: Vị trí mẫu kết thúc (tuyệt đối).
            
        Returns:
            Mảng np.ndarray (float32) liên tục chứa đúng đoạn âm thanh yêu cầu.
        """
        with self._lock:
            current_total = self._total_written
            
            if start_sample >= end_sample or current_total == 0:
                return np.zeros(0, dtype=np.float32)

            # Giới hạn cận trên không vượt quá số sample hiện có
            effective_end = min(end_sample, current_total)
            
            # Điểm bắt đầu khả dụng xa nhất trong buffer
            earliest_available = max(0, current_total - self.capacity_samples)
            effective_start = max(start_sample, earliest_available)
            
            if effective_start >= effective_end:
                return np.zeros(0, dtype=np.float32)

            requested_len = effective_end - effective_start
            result = np.empty(requested_len, dtype=np.float32)
            
            buf_start = effective_start % self.capacity_samples
            buf_end = buf_start + requested_len
            
            if buf_end <= self.capacity_samples:
                result[:] = self._buffer[buf_start:buf_end]
            else:
                first_len = self.capacity_samples - buf_start
                second_len = requested_len - first_len
                result[:first_len] = self._buffer[buf_start:self.capacity_samples]
                result[first_len:] = self._buffer[0:second_len]
                
            return result

    def get_recent(self, duration_sec: float) -> np.ndarray:
        """Lấy nhanh N giây âm thanh gần đây nhất (thường dùng cho VAD/Preview)."""
        samples_needed = int(self.sample_rate * duration_sec)
        with self._lock:
            end_sample = self._total_written
            start_sample = max(0, end_sample - samples_needed)
        return self.get_slice(start_sample, end_sample)

    def extract_speech_segment(
        self,
        start_sample: int,
        end_sample: int,
        pre_roll_ms: int = 300,
        post_roll_ms: int = 400,
    ) -> Tuple[np.ndarray, int, int]:
        """Trích xuất đoạn phát âm kèm tiền đệm (pre-roll) và hậu đệm (post-roll) để không mất phụ âm.
        
        Returns:
            Tuple (pcm_data, actual_start_sample, actual_end_sample)
        """
        pre_roll_samples = int(self.sample_rate * (pre_roll_ms / 1000.0))
        post_roll_samples = int(self.sample_rate * (post_roll_ms / 1000.0))
        
        actual_start = max(0, start_sample - pre_roll_samples)
        actual_end = end_sample + post_roll_samples
        
        data = self.get_slice(actual_start, actual_end)
        return data, actual_start, actual_end

    def clear(self) -> None:
        """Xoá bỏ dữ liệu cũ và reset con trỏ (Fast Cleanup < 200ms).

        FIX-06: KHÔNG zero-fill toàn bộ buffer nữa. Buffer mặc định 60 s float32
        = 960.000 mẫu × 4 byte ≈ **3,84 MB**; mỗi lần tua video / reset phiên lại ghi
        3,84 MB số 0 vào RAM mà không mang lại lợi ích đúng đắn nào.

        An toàn vì MỌI đường đọc đều bị chặn bởi con trỏ `_total_written`:
        - `get_slice()` trả mảng rỗng ngay khi `current_total == 0` và luôn kẹp
          `effective_start >= max(0, current_total - capacity_samples)`;
        - `write()` ghi đè từ `_total_written % capacity_samples`.
        Nên không có mẫu "rác" nào ngoài vùng hợp lệ được trả ra. (Xem
        `test_01_core_audio.py` cho test toàn vẹn bit-exact.)

        Nếu sau này cần xoá dữ liệu vì lý do bảo mật thì phải zero-fill TƯỜNG MINH ở
        đường đó, đừng bật lại ở đây.
        """
        with self._lock:
            self._total_written = 0
            self._dropped_samples_count = 0
