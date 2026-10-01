"""Lookahead Audio Demuxer & Decoder (PyAV 500x Real-Time Pipeline).

Giải mã các chunk audio (WebM Opus / MP4 AAC / WAV / MP3) nạp từ trình duyệt (MSE DASH)
thành PCM 16kHz Mono Float32 chuẩn xác với Presentation Time Stamp (PTS).
"""

import io
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import av
import numpy as np


@dataclass
class DecodedAudioChunk:
    """Đại diện cho một đoạn âm thanh đã giải mã từ container."""
    pts_start: float          # Mốc thời gian bắt đầu (giây) trên timeline video
    pts_end: float            # Mốc thời gian kết thúc (giây)
    duration: float           # Độ dài âm thanh (giây)
    pcm: np.ndarray           # Mảng 1D Float32 @ 16kHz ([-1.0, 1.0])
    sample_rate: int = 16000  # Tần số mẫu chuẩn
    channels: int = 1         # Mono
    decode_time_ms: float = 0.0 # Thời gian giải mã (ms)


class LookaheadDemuxer:
    """Bộ giải mã container audio đa định dạng cho luồng Lookahead VOD."""

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self._init_segment_cache: Optional[bytes] = None
        self._mime_type: str = "auto"
        self._total_decoded_seconds: float = 0.0
        self._total_decode_time_sec: float = 0.0

    def set_init_segment(self, init_bytes: bytes, mime_type: str = "auto"):
        """Lưu lại Header / Init Segment (WebM EBML / MP4 ftyp+moov) để ghép nối cho các media chunk."""
        self._init_segment_cache = init_bytes
        self._mime_type = mime_type

    def decode_chunk(
        self,
        raw_bytes: bytes,
        timestamp_offset: float = 0.0,
        mime_type: str = "auto"
    ) -> Optional[DecodedAudioChunk]:
        """Giải mã một khối byte audio thô thành DecodedAudioChunk PCM 16kHz.
        
        Args:
            raw_bytes: Dữ liệu nhị phân của chunk.
            timestamp_offset: Mốc offset thời gian (sourceBuffer.timestampOffset).
            mime_type: Gợi ý định dạng (webm, mp4, aac, opus, wav).
        
        Returns:
            DecodedAudioChunk nếu giải mã thành công, None nếu chunk là init header hoặc lỗi.
        """
        if not raw_bytes or len(raw_bytes) == 0:
            return None

        t0 = time.perf_counter()

        def try_decode(data_bytes: bytes) -> Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
            try:
                c = av.open(io.BytesIO(data_bytes))
                if not c or not c.streams.audio:
                    return None, None
                st = c.streams.audio[0]
                resamp = av.AudioResampler(format="flt", layout="mono", rate=self.target_sample_rate)
                f_pcm: List[np.ndarray] = []
                p_list: List[float] = []
                for fr in c.decode(st):
                    if fr.pts is not None and st.time_base is not None:
                        p_list.append(float(fr.pts * st.time_base))
                    for r_fr in resamp.resample(fr):
                        f_pcm.append(r_fr.to_ndarray().squeeze())
                for r_fr in resamp.resample(None):
                    f_pcm.append(r_fr.to_ndarray().squeeze())
                return f_pcm, p_list
            except Exception:
                return None, None

        # 1. Thử giải mã trực tiếp
        frames_pcm, pts_list = try_decode(raw_bytes)

        # 2. Nếu thất bại hoặc không có frame nào và có cache init segment, thử ghép init segment
        if (not frames_pcm) and self._init_segment_cache is not None:
            combined = self._init_segment_cache + raw_bytes
            frames_pcm, pts_list = try_decode(combined)

        # 3. Nếu vẫn không có frame nào, kiểm tra xem đây có phải là Init Header Segment không
        if not frames_pcm:
            # Nếu chunk chứa header (EBML \x1a\x45\xdf\xa3 hoặc MP4 ftyp/moov) hoặc độ dài nhỏ
            if raw_bytes.startswith(b"\x1a\x45\xdf\xa3") or b"ftyp" in raw_bytes[:64] or b"moov" in raw_bytes[:1024]:
                self._init_segment_cache = raw_bytes
            elif self._init_segment_cache is None and len(raw_bytes) < 65536:
                # Lưu chunk đầu tiên làm init candidate
                self._init_segment_cache = raw_bytes
            return None

        # Nối các frame PCM thành 1 mảng liên tục
        pcm_data = np.concatenate([f if f.ndim == 1 else f.flatten() for f in frames_pcm if f.size > 0])
        if pcm_data.size == 0:
            return None

        duration = float(len(pcm_data) / self.target_sample_rate)
        t1 = time.perf_counter()
        decode_time_ms = (t1 - t0) * 1000.0

        # Tính PTS chính xác
        pts_start = timestamp_offset
        if pts_list:
            pts_start += pts_list[0]
        pts_end = pts_start + duration

        # Cập nhật thống kê
        self._total_decoded_seconds += duration
        self._total_decode_time_sec += (t1 - t0)

        return DecodedAudioChunk(
            pts_start=pts_start,
            pts_end=pts_end,
            duration=duration,
            pcm=pcm_data,
            sample_rate=self.target_sample_rate,
            channels=1,
            decode_time_ms=decode_time_ms
        )

    @property
    def average_rtf(self) -> float:
        """Real-Time Factor trung bình của bộ giải mã (Càng nhỏ càng nhanh)."""
        if self._total_decoded_seconds <= 0:
            return 0.0
        return self._total_decode_time_sec / self._total_decoded_seconds
