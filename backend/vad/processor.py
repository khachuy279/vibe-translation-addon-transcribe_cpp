"""VAD Stream Processor: Bộ điều phối luồng VAD thời gian thực.

Tính năng:
- Nhận luồng âm thanh dạng bytes (PCM Int16/Float32) hoặc np.ndarray.
- Chia khung (Frame Slicing) chính xác theo kích thước native của engine.
- Quản lý trạng thái nói/im lặng bằng Đồng hồ Mẫu Âm thanh (Sample Clock).
- Hỗ trợ Pre-Speech Ring Buffer (đệm trước khi nói ~300ms) để không nuốt âm đầu.
- Hỗ trợ Hangover & Hysteresis (bảo vệ ngắt quãng giữa các từ).
- Cơ chế callbacks an toàn cho luồng ASR downstream.

Sửa đổi theo kế hoạch triển khai (Revision 2):
- P1.9: chuyển engine KHÔNG còn nạp model trong lúc giữ `self._lock`, và KHÔNG còn
  gọi `VADEngineFactory.get_engine()` (có thể tải từ HuggingFace) trên đường hot.
  Yêu cầu đổi engine được ghi lại (`_desired_engine`); việc kích hoạt thật diễn ra
  trên VAD worker thread ở `feed_chunk`, ngoài lock. Nhờ vậy `update_config()` chạy
  trên event loop không bao giờ block, và backend không ngừng nhận audio.
- P1.4: `enabled=False` trước đây khiến KHÔNG BAO GIỜ có phụ đề (vì
  `on_speech_start`/`on_speech_end` không bao giờ được gọi). Nay VAD được coi là bắt
  buộc: cảnh báo rõ một lần rồi vẫn chạy VAD.
- Instrumentation: đo `vad.chunk_ms` và `vad.frame_ms` theo lô (không theo từng frame)
  để không tự tạo overhead cho chính phép đo.
"""

from collections import deque
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.config import config
from backend.core.metrics import metrics_collector
from backend.core.pipeline_events import VADState
from backend.vad.base import BaseVADEngine, VADResult, VADStreamState
from backend.vad.engines import VADEngineFactory, SUPPORTED_VAD_ENGINES
from backend.utils.logger import logger


class VADStreamProcessor:
    """Bộ xử lý VAD streaming cách ly theo phiên (Session-Safe)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_engine: str = "firered-vad",
        threshold: Optional[float] = None,
        silence_duration_ms: int = 600,
        hangover_ms: int = 400,
        pre_speech_buffer_ms: int = 300,
        enabled: bool = True,
        on_speech_chunk: Optional[Callable[[bytes, float, str], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        engine_override: Optional[BaseVADEngine] = None,
    ):
        self.sample_rate = sample_rate
        self.vad_engine = (vad_engine or "firered-vad").lower().strip()
        if self.vad_engine not in SUPPORTED_VAD_ENGINES:
            self.vad_engine = "firered-vad"

        self.threshold = threshold if threshold is not None else config.vad.threshold
        self.silence_duration_ms = silence_duration_ms
        self.hangover_ms = hangover_ms
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        self.enabled = enabled

        self.on_speech_chunk = on_speech_chunk
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        self._lock = threading.RLock()
        self._engine: Optional[BaseVADEngine] = engine_override
        self._state: Optional[VADStreamState] = None
        # P1.9: engine mong muốn nhưng chưa kích hoạt được (chưa nạp sẵn).
        self._desired_engine: Optional[str] = None
        self._engine_load_thread: Optional[threading.Thread] = None
        self._vad_enabled_warned: bool = False
        # P4.6: đánh dấu đã force_end/reset. Vì model nay chạy NGOÀI lock nên một batch
        # có thể đang bay khi force_end() được gọi; kết quả của batch đó phải bị BỎ để
        # không sinh ra speech START mới sau khi phiên đã được chốt/đóng.
        self._stopped: bool = False

        self._frame_samples: int = 400
        self._frame_size_bytes: int = self._frame_samples * 2  # 16-bit PCM bytes

        self._overflow_count: int = 0

        if engine_override is not None:
            self._frame_samples = getattr(engine_override, "native_frame_samples", 400)
            self._frame_size_bytes = self._frame_samples * 2
            self._state = self._new_state(engine_override, self.threshold)
        else:
            self._ensure_engine()

    # ------------------------------------------------------------------ helpers
    def _max_pre_frames(self, frame_samples: int) -> int:
        return max(1, int((self.pre_speech_buffer_ms / 1000.0) * self.sample_rate / max(1, frame_samples)))

    def _new_state(self, engine: BaseVADEngine, threshold: Optional[float]) -> VADStreamState:
        """Tạo state cho engine. Có thể tốn thời gian (Silero nạp JIT model)."""
        state = engine.create_initial_state(threshold=threshold)
        frame_samples = getattr(engine, "native_frame_samples", 400)
        state.pre_speech_ring = deque(maxlen=self._max_pre_frames(frame_samples))
        return state

    def _ensure_engine(self) -> None:
        """Đảm bảo engine và session state đã được khởi tạo (off-lock)."""
        if self._engine is None:
            self._engine = VADEngineFactory.get_engine(self.vad_engine)
            self._frame_samples = getattr(self._engine, "native_frame_samples", 400)
            self._frame_size_bytes = self._frame_samples * 2

        if self._state is None:
            self._state = self._new_state(self._engine, self.threshold)

    # ------------------------------------------------------------------ cấu hình
    def apply_engine_change(self, engine_name: str) -> None:
        """Kích hoạt engine mong muốn (off-lock). Trả về im lặng nếu chưa nạp sẵn.

        P1.9: KHÔNG gọi factory nếu engine chưa có trong pool — tránh tải model từ
        HuggingFace trên đường hot. Thay vào đó nạp ở thread nền và thử lại sau.
        """
        engine = VADEngineFactory.peek_engine(engine_name)
        if engine is None:
            self._spawn_engine_load(engine_name)
            return
        try:
            state = self._new_state(engine, self.threshold)  # off-lock
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Không tạo được state cho engine '{engine_name}': {exc}", extra={"module_tag": "VAD"})
            return

        with self._lock:
            self.vad_engine = engine_name
            self._engine = engine
            self._frame_samples = getattr(engine, "native_frame_samples", 400)
            self._frame_size_bytes = self._frame_samples * 2
            self._state = state
            self._desired_engine = None
        logger.info(
            f"Đã kích hoạt engine '{engine_name}' "
            f"(frame={self._frame_samples} samples, {self._frame_samples / self.sample_rate * 1000:.0f}ms)",
            extra={"module_tag": "VAD"},
        )

    def _spawn_engine_load(self, engine_name: str) -> None:
        """Nạp engine ở thread nền. Không giữ lock, không block event loop hay VAD worker."""
        if self._engine_load_thread is not None and self._engine_load_thread.is_alive():
            return

        def _worker() -> None:
            t0 = time.perf_counter()
            try:
                VADEngineFactory.get_engine(engine_name)
                if self._engine is None:
                    self._ensure_engine()
                logger.info(
                    f"Đã nạp xong engine '{engine_name}' trong {time.perf_counter() - t0:.2f}s "
                        f"(sẽ kích hoạt ở chunk kế tiếp).",
                    extra={"module_tag": "VAD"},
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"Nạp engine '{engine_name}' thất bại: {type(exc).__name__}: {exc}. "
                        f"Giữ nguyên engine hiện tại '{self.vad_engine}'.",
                    extra={"module_tag": "VAD"},
                )
                with self._lock:
                    self._desired_engine = None

        self._engine_load_thread = threading.Thread(
            target=_worker, name=f"vad_load_{engine_name}", daemon=True
        )
        self._engine_load_thread.start()

    def update_config(
        self,
        vad_engine: Optional[str] = None,
        threshold: Optional[float] = None,
        silence_duration_ms: Optional[int] = None,
        hangover_ms: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Cập nhật cấu hình runtime. KHÔNG BAO GIỜ block (P1.9).

        Việc đổi engine chỉ ghi lại yêu cầu; kích hoạt thật diễn ra ở `feed_chunk`
        trên VAD worker thread, ngoài lock.
        """
        changes: List[str] = []
        with self._lock:
            if vad_engine is not None:
                eng = vad_engine.lower().strip()
                if eng in SUPPORTED_VAD_ENGINES and eng != self.vad_engine:
                    self._desired_engine = eng
                    if VADEngineFactory.is_cached(eng):
                        changes.append(f"engine: '{self.vad_engine}' -> '{eng}'")
                    else:
                        changes.append(f"engine: '{self.vad_engine}' -> '{eng}' (đang nạp nền)")

            if threshold is not None and float(threshold) != self.threshold:
                changes.append(f"threshold: {self.threshold:.2f} -> {float(threshold):.2f}")
                self.threshold = float(threshold)
            if silence_duration_ms is not None and int(silence_duration_ms) != self.silence_duration_ms:
                changes.append(f"silence: {self.silence_duration_ms}ms -> {int(silence_duration_ms)}ms")
                self.silence_duration_ms = int(silence_duration_ms)
            if hangover_ms is not None and int(hangover_ms) != self.hangover_ms:
                changes.append(f"hangover: {self.hangover_ms}ms -> {int(hangover_ms)}ms")
                self.hangover_ms = int(hangover_ms)
            if enabled is not None and bool(enabled) != self.enabled:
                changes.append(f"enabled: {self.enabled} -> {bool(enabled)}")
                self.enabled = bool(enabled)

        if changes:
            logger.info(
                f"Đồng bộ cấu hình: {', '.join(changes)} "
                    f"(silence_duration_ms={self.silence_duration_ms}ms, threshold={self.threshold:.2f})",
                extra={"module_tag": "VAD"},
            )

    # ------------------------------------------------------------------ ingress
    def feed_chunk(self, audio_data: Union[bytes, np.ndarray], capture_timestamp: float = 0.0) -> None:
        """Xử lý nạp chunk âm thanh thô (bytes Int16/Float32 hoặc np.ndarray)."""
        if audio_data is None or len(audio_data) == 0:
            return

        t_chunk = time.perf_counter()

        # P1.9: kích hoạt engine mong muốn ở đây (VAD worker thread), KHÔNG giữ lock
        # trong lúc tạo state (Silero nạp JIT model, FSMN init cache).
        desired = self._desired_engine
        if desired is not None and desired != self.vad_engine:
            self.apply_engine_change(desired)

        # P1.4: VAD là bắt buộc để phân câu. Tắt VAD => không có
        # on_speech_start/on_speech_end => không bao giờ có phụ đề. Cảnh báo rõ và
        # vẫn chạy VAD thay vì im lặng không làm gì.
        if not self.enabled:
            if not self._vad_enabled_warned:
                self._vad_enabled_warned = True
                logger.warning(
                    "Đã TẮT VAD trong cấu hình, nhưng VAD là bắt buộc để phân câu — ""nếu không chạy VAD thì sẽ KHÔNG có phụ đề nào. Đã tự bật lại VAD. ""(Dùng silence_duration_ms/min_words_to_commit để tinh chỉnh thay vì tắt VAD.)",
                    extra={"module_tag": "VAD"},
                )
                metrics_collector.increment_counter("vad.enabled_flag_forced_on")
            self.enabled = True

        self._ensure_engine()

        # P4.6 / F-13: TÁCH phần chạy model RA NGOÀI `self._lock`.
        # Trước đây `engine.is_speech()` (một forward PyTorch mỗi frame 25 ms) chạy BÊN
        # TRONG lock, nên `force_end()` / `update_config()` từ thread khác (event loop)
        # bị block — report cũ ghi nhận spike cleanup 102 ms. Nay chỉ giữ lock cho phần
        # đọc/ghi state; model chạy ngoài lock.
        # An toàn vì hiện chỉ có MỘT VAD worker thread (xem `_VAD_EXECUTOR`).
        frames: List[Tuple[bytes, float]] = []

        # --- Bước 1 (giữ lock): chỉ rút frame ra khỏi raw_buffer ---
        with self._lock:
            state = self._state
            raw_buf = state.raw_buffer

            # Chuyển đổi đầu vào thành bytes 16-bit PCM cho raw_buffer
            if isinstance(audio_data, bytes):
                raw_buf.extend(audio_data)
            elif isinstance(audio_data, (bytearray, memoryview)):
                raw_buf.extend(bytes(audio_data))
            elif isinstance(audio_data, np.ndarray):
                if audio_data.dtype == np.float32:
                    int16_arr = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
                    raw_buf.extend(int16_arr.tobytes())
                elif audio_data.dtype == np.int16:
                    raw_buf.extend(audio_data.tobytes())

            # Giới hạn buffer tối đa 3 giây để tránh phình bộ nhớ khi nghẽn
            max_buffer_bytes = int(self.sample_rate * 2 * 3.0)
            if len(raw_buf) > max_buffer_bytes:
                overflow_bytes = len(raw_buf) - max_buffer_bytes
                del raw_buf[:overflow_bytes]
                self._overflow_count += 1

            frame_size = self._frame_size_bytes
            buf_len = len(raw_buf)
            offset = 0
            while buf_len - offset >= frame_size:
                frame_end = offset + frame_size
                frame_ts = capture_timestamp + (offset / (self.sample_rate * 2.0))
                frames.append((bytes(raw_buf[offset:frame_end]), frame_ts))
                offset = frame_end

            if offset > 0:
                del raw_buf[:offset]

            engine = self._engine
            engine_name = getattr(engine, "name", "")

        # --- Bước 2 (KHÔNG giữ lock): chạy model cho từng frame ---
        results: List[Tuple[bytes, float, Any]] = []
        for frame_bytes, frame_ts in frames:
            # Với FireRed-VAD, truyền chunk_raw (Int16 PCM) trực tiếp
            # để triệt tiêu chi phí cấp phát và chuyển đổi Float32 -> Int16
            if engine_name == "firered-vad":
                samples_float32 = None
            else:
                samples_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
                samples_float32 = samples_int16.astype(np.float32) / 32768.0

            res = engine.is_speech(
                samples_float32,
                state,
                self.threshold,
                chunk_raw=frame_bytes,
            )
            results.append((frame_bytes, frame_ts, res))

        # --- Bước 3 (giữ lock): áp chuyển trạng thái + thu thập callback ---
        callbacks_to_fire: List[Tuple[Callable, tuple]] = []
        with self._lock:
            if self._state is not state or self._stopped:
                # Engine/state bị đổi trong lúc chạy model (người dùng đổi VAD engine),
                # hoặc phiên đã force_end -> kết quả thuộc state cũ nên phải BỎ.
                metrics_collector.increment_counter("vad.batch_discarded_engine_switch")
            else:
                for frame_bytes, frame_ts, res in results:
                    self._apply_frame_result(state, frame_bytes, frame_ts, res, callbacks_to_fire)

        # Kích hoạt callbacks bên ngoài lock để tránh deadlock
        for fn, args in callbacks_to_fire:
            try:
                fn(*args)
            except Exception as e:
                logger.error(f"VAD callback error: {e}", exc_info=True, extra={"module_tag": "VAD"})

        metrics_collector.record_metric("vad", "chunk_ms", (time.perf_counter() - t_chunk) * 1000.0)
        metrics_collector.record_gauge("vad", "overflow_count", self._overflow_count)

    def _apply_frame_result(
        self,
        state: VADStreamState,
        frame_bytes: bytes,
        frame_ts: float,
        res: VADResult,
        callbacks_to_fire: List[Tuple[Callable, tuple]],
    ) -> None:
        """Áp kết quả VAD của MỘT frame vào state và thu thập callback cần gọi.

        Caller phải giữ `self._lock`. Không chạy model ở đây (xem P4.6).
        """
        frame_samples = self._frame_samples
        is_speech_frame = res.is_speech
        prob = res.probability
        vad_event = res.event

        # Đếm tổng số mẫu đã xử lý (HEAD có dòng này; bị xoá nhầm khi dọn logging ⇒
        # test `test_vad_still_detects_speech_normally` fail vì counter luôn 0).
        state.total_samples_processed += self._frame_samples

        # Chuyển trạng thái: SILENCE -> SPEECH
        if vad_event == "START" or (not state.is_speech and is_speech_frame):
            if not state.is_speech:
                state.is_speech = True
                state.silence_samples = 0
                logger.info(
                    f"VAD: START (p={prob:.2f})",
                    extra={"module_tag": "VAD"},
                )
                if self.on_speech_start:
                    callbacks_to_fire.append((self.on_speech_start, ()))

                # Xả toàn bộ Pre-speech buffer
                while state.pre_speech_ring:
                    pre_bytes, pre_ts = state.pre_speech_ring.popleft()
                    if self.on_speech_chunk:
                        callbacks_to_fire.append(
                            (self.on_speech_chunk, (pre_bytes, pre_ts, VADState.PRE_ROLL.value))
                        )

            if self.on_speech_chunk:
                callbacks_to_fire.append(
                    (self.on_speech_chunk, (frame_bytes, frame_ts, VADState.SPEECH.value))
                )

        elif state.is_speech:
            if vad_event == "END":
                state.is_speech = False
                state.silence_samples = 0
                logger.info(
                    f"VAD: END (p={prob:.2f})",
                    extra={"module_tag": "VAD"},
                )
                if self.on_speech_end:
                    callbacks_to_fire.append((self.on_speech_end, ()))
            elif is_speech_frame:
                state.silence_samples = 0
                if self.on_speech_chunk:
                    callbacks_to_fire.append(
                        (self.on_speech_chunk, (frame_bytes, frame_ts, VADState.SPEECH.value))
                    )
            else:
                # Đang trong câu nói nhưng frame hiện tại là khoảng lặng -> Tích lũy silence
                state.silence_samples += frame_samples
                silence_elapsed_ms = (state.silence_samples / self.sample_rate) * 1000.0
                total_silence_limit_ms = float(self.silence_duration_ms)
                grace_hangover_ms = min(float(self.hangover_ms), total_silence_limit_ms * 0.5)

                if silence_elapsed_ms <= grace_hangover_ms:
                    # Vẫn nằm trong vùng ân hạn Hangover -> Tiếp tục gửi cho ASR
                    if self.on_speech_chunk:
                        callbacks_to_fire.append(
                            (self.on_speech_chunk, (frame_bytes, frame_ts, VADState.SPEECH.value))
                        )

                if silence_elapsed_ms >= total_silence_limit_ms:
                    # Đạt ngưỡng im lặng chốt câu -> SILENCE
                    state.is_speech = False
                    state.silence_samples = 0
                    logger.info(
                        f"VAD: END (timeout {silence_elapsed_ms:.0f}ms, p={prob:.2f})",
                        extra={"module_tag": "VAD"},
                    )
                    if self.on_speech_end:
                        callbacks_to_fire.append((self.on_speech_end, ()))
        else:
            # SILENCE state -> Lưu frame vào Pre-speech ring buffer
            state.pre_speech_ring.append((frame_bytes, frame_ts))

    def force_end(self) -> None:
        """Ép buộc kết thúc câu nói hiện tại và kích hoạt callback chốt câu.

        P4.6: hàm này KHÔNG còn bị block bởi inference VAD (model chạy ngoài lock).
        Đồng thời đánh dấu `_stopped` để batch inference đang bay bị bỏ, tránh sinh
        speech START mới sau khi đã chốt/đóng phiên.
        """
        with self._lock:
            self._stopped = True
            if self._state and self._state.is_speech:
                self._state.is_speech = False
                self._state.silence_samples = 0
                if self.on_speech_end:
                    try:
                        self.on_speech_end()
                    except Exception as e:
                        logger.error(f"VAD on_speech_end callback error: {e}", extra={"module_tag": "VAD"})

    def reset(self) -> None:
        """Reset trạng thái processor (cho phép nhận audio trở lại)."""
        with self._lock:
            self._stopped = False
            if self._state:
                self._state.reset()
            self._overflow_count = 0


# Alias tương thích ngược
VADProcessor = VADStreamProcessor
