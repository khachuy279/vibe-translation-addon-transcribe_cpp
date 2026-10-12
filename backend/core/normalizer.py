"""Module Chuẩn Hóa Âm Lượng Thích Ứng (Speech Normalization & Soft Knee Limiter).

Đặc tính:
- Tự động bù âm lượng cho các đoạn phát âm quá nhỏ hoặc nén các đoạn hét to/tiếng ồn đột ngột.
- Sử dụng ngưỡng mềm (Soft Knee) để tránh hiện tượng kéo tăng nhiễu nền khi im lặng.
- Khống chế đỉnh tín hiệu (Peak Limiter) để không bao giờ bị méo tiếng (clipping).
- Vector hoá numpy, chỉ cấp phát MỘT mảng đích cho mỗi lần normalize.

Lịch sử tối ưu (P2.6):
- Trước đây mỗi lần `normalize()` cấp phát ~7 mảng tạm (RMS, peak, gain, clip, RMS/peak
  của kết quả) và trả về 4 field không ai đọc. Nay:
    * RMS tính bằng `np.dot` (không tạo mảng tạm).
    * Peak tính bằng hai phép rút gọn `max()`/`min()` của C (không tạo mảng tạm).
    * Chỉ cấp phát đúng một mảng đích, dùng `out=` cho cả nhân gain và clip.
    * `NormalizationResult` chỉ còn các field thực sự được dùng.
- Cấu hình được truyền từ `ASRConfig` qua `SpeechNormalizer.from_config()`; trước đây
  engine gọi `SpeechNormalizer()` không tham số nên toàn bộ config `normalize_*` bị bỏ qua.
"""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass(slots=True)
class NormalizationResult:
    """Kết quả chuẩn hóa âm lượng.

    Lưu ý: `audio` có thể là CHÍNH mảng đầu vào (khi không cần biến đổi) để tránh
    cấp phát thừa. Bên gọi không được giả định đây là bản sao.
    """
    audio: np.ndarray
    applied_gain: float
    original_rms: float
    is_modified: bool


class SpeechNormalizer:
    """Bộ chuẩn hóa âm lượng thích ứng cho ASR & VAD."""

    def __init__(
        self,
        target_rms: float = 0.10,        # Mức RMS tiêu chuẩn (~ -20 dBFS)
        target_peak: float = 0.95,       # Giới hạn đỉnh an toàn
        max_gain: float = 3.0,           # Khuếch đại tối đa 3.0x (+9.5 dB)
        min_gain: float = 0.3333,        # Nén tối đa 0.33x (-9.5 dB)
        knee_start: float = 0.025,       # Dưới mức này coi là nhiễu nền, không boost
        knee_end: float = 0.050,         # Vùng chuyển tiếp mượt mà
        log_stats: bool = False,
    ):
        self.target_rms = float(target_rms)
        self.target_peak = float(target_peak)
        self.max_gain = float(max_gain)
        self.min_gain = float(min_gain)
        self.knee_start = float(knee_start)
        self.knee_end = float(knee_end)
        self.log_stats = bool(log_stats)

    @classmethod
    def from_config(cls, asr_cfg: Optional[Any] = None) -> "SpeechNormalizer":
        """Khởi tạo từ `ASRConfig` (hoặc object có cùng tên thuộc tính).

        Nhờ vậy các tham số `normalize_*` trong config có tác dụng thật (P2.6 / G7).
        """
        if asr_cfg is None:
            return cls()
        return cls(
            target_rms=getattr(asr_cfg, "normalize_target_rms", 0.10),
            target_peak=getattr(asr_cfg, "normalize_target_peak", 0.95),
            max_gain=getattr(asr_cfg, "normalize_max_gain", 3.0),
            min_gain=getattr(asr_cfg, "normalize_min_gain", 0.3333),
            knee_start=getattr(asr_cfg, "normalize_knee_start", 0.025),
            knee_end=getattr(asr_cfg, "normalize_knee_end", 0.050),
            log_stats=getattr(asr_cfg, "normalize_log_stats", False),
        )

    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:
        """Tính RMS không cấp phát mảng tạm (dùng np.dot thay vì mean(a*a))."""
        if audio is None or len(audio) == 0:
            return 0.0
        arr = np.asarray(audio, dtype=np.float32)
        # np.dot trả về vô hướng; không tạo mảng trung gian như (a * a).
        return float(np.sqrt(np.dot(arr, arr) / arr.size))

    @staticmethod
    def calculate_peak(audio: np.ndarray) -> float:
        """Tính đỉnh biên độ không cấp phát mảng tạm (max/min của C thay vì np.abs)."""
        if audio is None or len(audio) == 0:
            return 0.0
        arr = np.asarray(audio, dtype=np.float32)
        return float(max(arr.max(), -arr.min()))

    def normalize(self, audio: np.ndarray) -> NormalizationResult:
        """Chuẩn hóa âm lượng cho mảng float32 1D trong [-1.0, 1.0].

        Bất biến (từ 2026-10-12):
          * `orig_rms < knee_start` ⇒ giữ nguyên (coi là im lặng/nhiễu nền).
          * Audio NHỎ hơn `target_rms` ⇒ gain ≥ 1,0 (chỉ kẹp theo trần đỉnh, KHÔNG bao giờ bị nén
            xuống dưới 1,0 — xem khối "Bảo vệ đỉnh" bên dưới).
          * Audio TO hơn `target_rms` ⇒ nén về `target_peak` như cũ, sàn `min_gain`.
          * Đầu ra luôn bị `clip` về [-1, 1] nên không méo tiếng.
        """
        if audio is None or len(audio) == 0:
            return NormalizationResult(
                audio=np.zeros(0, dtype=np.float32),
                applied_gain=1.0,
                original_rms=0.0,
                is_modified=False,
            )

        arr = audio if audio.dtype == np.float32 else audio.astype(np.float32)
        orig_rms = self.calculate_rms(arr)

        # Im lặng / nhiễu nền dưới knee_start -> giữ nguyên, KHÔNG copy (tránh 1 alloc).
        if orig_rms < self.knee_start:
            return NormalizationResult(
                audio=arr,
                applied_gain=1.0,
                original_rms=orig_rms,
                is_modified=False,
            )

        orig_peak = self.calculate_peak(arr)

        # Hệ số khuếch đại lý thuyết
        desired_gain = self.target_rms / orig_rms

        # Soft Knee cho vùng âm lượng nhỏ sát nhiễu nền [knee_start, knee_end]
        if orig_rms < self.knee_end:
            span = self.knee_end - self.knee_start
            alpha = (orig_rms - self.knee_start) / span if span > 0 else 1.0
            desired_gain = 1.0 + alpha * (desired_gain - 1.0)

        # Giới hạn gain trong khoảng an toàn [min_gain, max_gain]
        clamped_gain = max(self.min_gain, min(self.max_gain, desired_gain))

        # ── Bảo vệ đỉnh: KẸP theo trần, KHÔNG nén audio đang nhỏ ────────────────────────────────
        #
        # SỰ CỐ THẬT (đo 2026-10-12, `wav_test/English_low_speech_quality_19s.wav`): bản cũ **GÁN**
        # `clamped_gain = target_peak / orig_peak` cho mọi trường hợp vượt trần. Hệ quả ngược hẳn mục
        # đích của module ("bù âm lượng cho đoạn phát âm quá nhỏ"):
        #   * `peak = 0,95` (file đã chuẩn-hoá-đỉnh về đúng trần) ⇒ gain = 0,95/0,95 = **1,0000**
        #     ⇒ `is_modified=False` ⇒ không boost gì, dù RMS 0,0702 còn thấp hơn mức mục tiêu 0,10.
        #   * `peak = 0,99` + RMS thấp ⇒ gain = 0,9596 ⇒ **NÉN XUỐNG**, tức làm audio đang cần boost
        #     nhỏ đi thêm.
        #
        # NAY: khi audio ĐANG NHỎ hơn mức mục tiêu (`desired_gain > 1`) thì chỉ **kẹp trần**, và không
        # bao giờ kéo gain xuống dưới 1,0. Audio TO hơn mức mục tiêu (`desired_gain <= 1`) vẫn được
        # nén về trần như cũ — đó mới là mục đích của `target_peak`.
        if orig_peak > 0.0 and orig_peak * clamped_gain > self.target_peak:
            peak_cap = self.target_peak / orig_peak
            if desired_gain > 1.0:
                # Kẹp: tối đa tới trần đỉnh, nhưng KHÔNG nén xuống dưới 1,0.
                clamped_gain = max(1.0, min(clamped_gain, peak_cap))
            else:
                clamped_gain = peak_cap

        # Một mảng đích duy nhất + hai phép biến đổi tại chỗ (không mảng tạm).
        out = np.empty_like(arr)
        np.multiply(arr, clamped_gain, out=out)
        np.clip(out, -1.0, 1.0, out=out)

        return NormalizationResult(
            audio=out,
            applied_gain=clamped_gain,
            original_rms=orig_rms,
            is_modified=(abs(clamped_gain - 1.0) > 1e-4),
        )
