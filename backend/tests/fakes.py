"""Fake engines cho test tầng A (không cần nạp model thật).

Mục tiêu: kiểm thử LOGIC điều phối của pipeline (phân câu, cửa sổ preview, gộp commit,
ranh giới pre-roll, áp dụng cấu hình) trong vài giây thay vì vài phút.

Cách tiếp cận:
- `FakeInferenceEngine` KẾ THỪA `TranscribeEngine` và chỉ override `_run_inference_sync`.
  Nhờ vậy toàn bộ logic orchestration thật (`stream_tokens`, commit tiers, cửa sổ
  preview, xử lý pre-roll) được test nguyên vẹn, chỉ thay phần suy luận C++.
- `FakeVADEngine` là một engine VAD xác định (deterministic) dựa trên năng lượng RMS,
  để test có thể điều khiển chính xác thời điểm bắt đầu/kết thúc nói.
"""

from collections import deque
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np

from backend.asr.engine import TranscribeEngine
from backend.vad.base import BaseVADEngine, VADResult, VADStreamState


class FakeInferenceEngine(TranscribeEngine):
    """TranscribeEngine không cần model: `_run_inference_sync` trả text giả lập."""

    def __init__(
        self,
        text_fn: Optional[Callable[[int], str]] = None,
        infer_delay: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.infer_delay = float(infer_delay)
        self.infer_calls: List[int] = []  # độ dài (samples) mỗi lần suy luận
        self.infer_times: List[float] = []
        self.text_fn = text_fn or (lambda n: "alpha beta gamma delta epsilon")

    def _run_inference_sync(self, pcm_audio: np.ndarray) -> str:  # type: ignore[override]
        n = 0 if pcm_audio is None else len(pcm_audio)
        self.infer_calls.append(n)
        self.infer_times.append(time.perf_counter())
        if self.infer_delay > 0:
            time.sleep(self.infer_delay)
        if n == 0:
            return ""
        return self.text_fn(n)

    def _preload_model(self, force_warm: bool = False) -> None:  # type: ignore[override]
        """Tầng A: KHÔNG có model thật để nạp/pre-warm — bỏ qua.

        Nếu không override, `stream_tokens` sẽ gọi `_ensure_model_loaded()` và cố nạp
        GGUF thật, làm test tầng A mất hết ý nghĩa "không cần model".
        Giữ đúng chữ ký `force_warm` để test có thể gọi `prewarm()`.
        """
        return

    # Tiện ích cho test
    @property
    def slice_count(self) -> int:
        return len(self.infer_calls)


class StableTextEngine(FakeInferenceEngine):
    """Trả text chỉ phụ thuộc vào một "cột mốc" để test STABLE_PREFIX.

    `milestones` là danh sách (min_samples, text). Text trả về là text của mốc cao
    nhất mà `n >= min_samples`. Nhờ vậy text "đứng yên" khi audio vẫn đang dài ra,
    mô phỏng việc người nói ngừng nói giữa câu.
    """

    def __init__(self, milestones, **kwargs: Any):
        super().__init__(**kwargs)
        self.milestones = sorted(milestones, key=lambda m: m[0])

    def _run_inference_sync(self, pcm_audio: np.ndarray) -> str:  # type: ignore[override]
        n = 0 if pcm_audio is None else len(pcm_audio)
        self.infer_calls.append(n)
        self.infer_times.append(time.perf_counter())
        if self.infer_delay > 0:
            time.sleep(self.infer_delay)
        text = ""
        for min_samples, value in self.milestones:
            if n >= min_samples:
                text = value
        return text


class FakeVADEngine(BaseVADEngine):
    """VAD xác định theo năng lượng RMS của frame (không cần model).

    Sau khi viết lại tầng VAD, engine **phải tự phát START/END** (processor không còn suy
    diễn từ `is_speech` từng frame). Fake này mô phỏng đúng hành vi cũ để các test tầng A
    giữ nguyên ngữ nghĩa:

    * `START` ở frame vượt ngưỡng đầu tiên, xin **1 frame pre-roll** (giống vòng đệm
      `pre_speech_buffer_ms=0` cũ ⇒ `deque(maxlen=1)`).
    * `END` sau đúng `silence_ms` im lặng (mặc định 450 ms = 18 frame 25 ms, trùng
      `VADConfig.silence_duration_ms` cũ).
    """

    name = "fake-vad"
    frame_samples = 400          # 25 ms @ 16 kHz
    default_threshold = 0.45
    max_lookback_frames = 1      # giống pre-roll 1 frame của bản cũ
    min_speech_frames = 1

    #: Im lặng mặc định khi phiên không cấu hình `silence_duration_ms` (None = "off").
    default_silence_ms = 450

    def __init__(self, rms_threshold: float = 0.02):
        self.rms_threshold = float(rms_threshold)

    def create_initial_state(
        self,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADStreamState:
        state = VADStreamState()
        state.engine_state = _FakeVADSession()
        return state

    def is_speech(
        self,
        frame_int16: np.ndarray,
        state: VADStreamState,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADResult:
        session: _FakeVADSession = state.engine_state
        if session is None:
            session = _FakeVADSession()
            state.engine_state = session

        arr = np.asarray(frame_int16, dtype=np.float32) / 32768.0
        rms = float(np.sqrt(np.dot(arr, arr) / arr.size)) if arr.size else 0.0
        is_speech_frame = rms > self.rms_threshold

        eff_silence_ms = self.default_silence_ms if silence_ms is None else int(silence_ms)
        silence_frames = max(1, int(round(eff_silence_ms / 25.0)))

        res = VADResult(
            probability=min(1.0, rms / max(1e-6, self.rms_threshold)),
            is_speech=is_speech_frame,   # bằng chứng của RIÊNG frame này (không phải state)
        )

        if is_speech_frame:
            session.silence_run = 0
            if not session.in_speech:
                session.speech_run += 1
                if session.speech_run >= self.min_speech_frames:
                    session.in_speech = True
                    res.event = "START"
                    res.lookback_frames = self.max_lookback_frames
        else:
            session.speech_run = 0
            if session.in_speech:
                session.silence_run += 1
                if session.silence_run >= silence_frames:
                    session.in_speech = False
                    res.event = "END"
                    res.lookback_frames = 0

        return res


@dataclass
class _FakeVADSession:
    """Máy trạng thái nhỏ của `FakeVADEngine`."""

    in_speech: bool = False
    speech_run: int = 0
    silence_run: int = 0


class FakeTranslator:
    """Translator giả: trả bản dịch có tiền tố, không nạp model."""

    def __init__(self, prefix: str = "[vi] ", per_char_delay: float = 0.0):
        self.prefix = prefix
        self.calls: List[str] = []
        self.per_char_delay = per_char_delay

    async def translate(self, text: str, source_lang: str = "auto", target_lang: str = "vi", context: str = "") -> dict:
        self.calls.append(text)
        return {"translated_text": f"{self.prefix}{text}", "elapsed_ms": 1.0}

    async def translate_stream(self, text: str, source_lang: str = "auto", target_lang: str = "vi", context: str = ""):
        self.calls.append(text)
        acc = self.prefix
        yield acc
        for ch in text:
            acc += ch
            if self.per_char_delay:
                import asyncio
                await asyncio.sleep(self.per_char_delay)
            yield acc


def make_speech_pcm(duration_sec: float, sample_rate: int = 16000, amplitude: float = 0.3) -> np.ndarray:
    """Tạo audio "có tiếng nói" xác định (sóng sin biên độ lớn hơn ngưỡng RMS)."""
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def make_silence_pcm(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Tạo im lặng tuyệt đối."""
    return np.zeros(int(sample_rate * duration_sec), dtype=np.float32)


def pcm_to_int16_bytes(pcm: np.ndarray) -> bytes:
    """Chuyển float32 [-1,1] sang bytes PCM16 (đúng định dạng client gửi)."""
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
