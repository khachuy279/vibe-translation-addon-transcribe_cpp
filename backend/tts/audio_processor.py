"""Module xử lý và chuyển đổi tín hiệu âm thanh cho TTS (SRP - Single Responsibility Principle).

Bao gồm:
- Chuyển đổi tensor/ndarray thành mảng float32 1 chiều.
- Điều chỉnh tốc độ (Time-Stretching) không làm thay đổi cao độ (Phase Vocoder).
- Chuẩn hóa âm lượng (Peak Normalization) chống méo tiếng.
- Mã hóa Base64 chuỗi WAV PCM 16-bit.
"""

import base64
import io
from typing import Any, List, Union
import numpy as np
import soundfile as sf
import torch
from scipy import signal

from backend.utils.logger import get_logger

logger = get_logger("tts.audio_processor")


class AudioProcessor:
    """Tập hợp các hàm thuần túy biến đổi và chuẩn hóa âm thanh đầu ra của TTS."""

    @staticmethod
    def convert_to_numpy(audio_output: Any) -> np.ndarray:
        """Chuyển đổi kết quả sinh âm thanh của model thành mảng 1D float32."""
        if audio_output is None:
            return np.zeros((0,), dtype=np.float32)

        if isinstance(audio_output, (list, tuple)):
            items = list(audio_output)
        else:
            items = [audio_output]

        converted: List[np.ndarray] = []
        for item in items:
            if isinstance(item, np.ndarray):
                # `np.asarray(..., dtype=...)` KHÔNG copy nếu dtype đã khớp; `.astype()`
                # mặc định `copy=True` nên luôn tạo bản sao thừa (G-05 / audit Gemini).
                arr = np.asarray(item, dtype=np.float32)
            elif isinstance(item, torch.Tensor):
                # `.cpu()` đã copy sang host; `.numpy()` là view trên bộ nhớ đó (zero-copy),
                # còn `asarray` chỉ copy nếu dtype khác float32 (thường là float32 ⇒ 0 copy).
                arr = np.asarray(item.detach().cpu().numpy(), dtype=np.float32)
            else:
                logger.debug(f"Bỏ qua phần tử âm thanh không hợp lệ: {type(item)}", extra={"module_tag": "TTS"})
                continue

            arr = np.atleast_1d(np.squeeze(arr))
            if arr.ndim > 0 and len(arr) > 0:
                converted.append(arr)

        if not converted:
            return np.zeros((0,), dtype=np.float32)

        return np.concatenate(converted, axis=0)

    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        sample_rate: int = 24000,
        threshold_db: float = -48.0,
        pad_start_ms: int = 40,
        pad_end_ms: int = 250,
    ) -> np.ndarray:
        """Cắt tỉa khoảng lặng thừa ở đầu và cuối audio do mô hình TTS sinh ra, bảo toàn trọn vẹn từ ngữ."""
        if audio is None or len(audio) == 0:
            return np.zeros((0,), dtype=np.float32)

        peak = float(np.max(np.abs(audio)))
        if peak < 1e-4:
            return audio

        frame_len = max(16, int(sample_rate * 0.01))  # 10ms
        hop_len = max(8, int(sample_rate * 0.005))    # 5ms

        if len(audio) < frame_len:
            return audio

        # Ngưỡng phát hiện tiếng nói an toàn (-48dB tương đương 0.4% biên độ đỉnh)
        threshold = max(peak * (10.0 ** (threshold_db / 20.0)), 0.002)

        # Trượt cửa sổ lấy biên độ đỉnh
        frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len]
        frame_max = np.max(np.abs(frames), axis=1)

        active = np.where(frame_max > threshold)[0]
        if len(active) == 0:
            return audio

        start_sample = max(0, active[0] * hop_len - int(sample_rate * (pad_start_ms / 1000.0)))
        end_sample = min(len(audio), (active[-1] * hop_len + frame_len) + int(sample_rate * (pad_end_ms / 1000.0)))

        return np.ascontiguousarray(audio[start_sample:end_sample], dtype=np.float32)

    @staticmethod
    def apply_time_stretch(
        audio: np.ndarray,
        speed: float,
        sample_rate: int = 24000,
        win_ms: float = 25.0,
    ) -> np.ndarray:
        """Điều chỉnh tốc độ phát (0.5x - 2.0x) bằng thuật toán WSOLA (Waveform Similarity Overlap-Add).

        Ưu điểm vượt trội so với Phase Vocoder:
        - Hoạt động trong miền thời gian (time-domain) bằng cross-correlation.
        - Bảo toàn 100% liên kết pha của các formant và hài âm giọng người.
        - Hoàn toàn KHÔNG bị méo kim loại (phasiness) hay tiếng vang rỗng (tin-can artifact).
        - Giữ nguyên độ nảy và rõ nét của các phụ âm bật / âm xát (transients).
        """
        if audio is None or len(audio) == 0:
            return np.zeros((0,), dtype=np.float32)

        rate = float(speed)
        # Bỏ qua nếu tốc độ chuẩn 1.0x
        if abs(rate - 1.0) < 0.02 or rate <= 0:
            return audio

        # Giới hạn dải tốc độ an toàn
        rate = max(0.5, min(2.0, rate))

        win_size = int(round(sample_rate * (win_ms / 1000.0)))
        if win_size % 2 != 0:
            win_size += 1
        syn_hop = win_size // 2
        delta = win_size // 2

        if len(audio) < win_size * 2:
            return audio

        try:
            win = np.hanning(win_size).astype(np.float32)
            num_syn_frames = int(np.ceil((len(audio) - win_size) / (syn_hop * rate)))
            if num_syn_frames <= 0:
                return audio

            out_len = (num_syn_frames - 1) * syn_hop + win_size
            out = np.zeros(out_len, dtype=np.float32)
            out_weights = np.zeros(out_len, dtype=np.float32)

            # Khung đầu tiên
            out[0:win_size] += audio[0:win_size] * win
            out_weights[0:win_size] += win

            last_ana_pos = 0

            for k in range(1, num_syn_frames):
                nominal_ana_pos = int(round(k * syn_hop * rate))
                natural_pos = last_ana_pos + syn_hop

                search_start = max(0, nominal_ana_pos - delta)
                search_end = min(len(audio) - win_size, nominal_ana_pos + delta)

                if search_start >= search_end:
                    best_pos = max(0, min(len(audio) - win_size, nominal_ana_pos))
                else:
                    template_start = min(len(audio) - win_size, max(0, natural_pos))
                    template = audio[template_start : template_start + win_size]
                    candidates = audio[search_start : search_end + win_size]

                    corrs = signal.correlate(candidates, template, mode="valid", method="direct")
                    best_offset = int(np.argmax(corrs))
                    best_pos = search_start + best_offset

                syn_pos = k * syn_hop
                out[syn_pos : syn_pos + win_size] += audio[best_pos : best_pos + win_size] * win
                out_weights[syn_pos : syn_pos + win_size] += win
                last_ana_pos = best_pos

            out = np.divide(out, np.maximum(out_weights, 1e-4), out=out)
            return np.ascontiguousarray(out, dtype=np.float32)
        except Exception as e:
            logger.warning(f"WSOLA time stretch thất bại ({e}), dùng âm thanh gốc.", extra={"module_tag": "TTS"})
            return audio

    @staticmethod
    def normalize_audio(
        audio: np.ndarray,
        volume: float = 1.0,
        target_peak: float = 0.95,
    ) -> np.ndarray:
        """Chuẩn hóa đỉnh biên độ và khuếch đại âm lượng, chống hiện tượng clipping méo tiếng."""
        if audio is None or len(audio) == 0:
            return np.zeros((0,), dtype=np.float32)

        max_peak = float(np.max(np.abs(audio)))
        if max_peak > 1e-4:
            vol = float(volume if volume is not None else 1.0)
            target = min(0.98, target_peak * vol)
            normalized = (audio / max_peak) * target
            return np.clip(normalized, -0.99, 0.99).astype(np.float32)
        return audio

    @staticmethod
    def apply_fades(
        audio: np.ndarray,
        sample_rate: int = 24000,
        fade_in_ms: float = 10.0,
        fade_out_ms: float = 20.0,
    ) -> np.ndarray:
        """Áp dụng micro fade-in và fade-out mượt (raised-cosine) ở đầu và cuối audio.

        Triệt tiêu 100% tiếng 'bụp' (click/pop transient do DC offset hoặc cắt giữa chu kỳ sóng)
        khi bắt đầu hoặc khi kết thúc âm thanh tự nhiên.
        """
        if audio is None or len(audio) == 0:
            return np.zeros((0,), dtype=np.float32)

        out = np.asarray(audio, dtype=np.float32).copy()
        n = len(out)
        fade_in_samples = max(1, int(sample_rate * (fade_in_ms / 1000.0)))
        fade_out_samples = max(1, int(sample_rate * (fade_out_ms / 1000.0)))

        if n < fade_in_samples + fade_out_samples:
            fade_in_samples = n // 2
            fade_out_samples = n - fade_in_samples

        if fade_in_samples > 0:
            ramp_in = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_in_samples, endpoint=False)))
            out[:fade_in_samples] *= ramp_in

        if fade_out_samples > 0:
            ramp_out = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_out_samples, endpoint=True)))
            out[-fade_out_samples:] *= ramp_out

        return out

    @staticmethod
    def encode_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        """Mã hóa mảng float32 thành bytes WAV PCM 16-bit (KHÔNG base64), có micro fade chống bụp."""
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return b""
        audio_smooth = AudioProcessor.apply_fades(audio, sample_rate)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_smooth, sample_rate, format="WAV", subtype="PCM_16")
        return wav_buffer.getvalue()

    @staticmethod
    def encode_wav_to_base64(audio: np.ndarray, sample_rate: int) -> str:
        """Mã hóa mảng float32 thành chuỗi Base64 định dạng WAV PCM 16-bit."""
        wav_bytes = AudioProcessor.encode_wav_bytes(audio, sample_rate)
        if not wav_bytes:
            return ""
        return base64.b64encode(wav_bytes).decode("ascii")
