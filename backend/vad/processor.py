"""VAD Stream Processor: bộ điều phối luồng VAD thời gian thực.

Nhiệm vụ DUY NHẤT của tầng này:

1. Cắt PCM thô từ client thành frame đúng **bước nhảy** của engine (`engine.frame_samples`).
2. Đưa frame cho engine VAD (engine sở hữu toàn bộ máy trạng thái nói/im).
3. Chuyển tiếp audio **nguyên bản** sang ASR khi engine báo đang trong đoạn nói, và xả
   lại phần pre-roll mà chính VAD yêu cầu khi báo `START` (`VADResult.lookback_frames`).

Điều tầng này **KHÔNG** làm (khác bản cũ):
- Không tự quyết định nói/im bằng `is_speech` từng frame ⇒ không còn chuyện processor
  chốt câu sớm hơn VAD rồi làm mất phụ âm đầu.
- Không có `hangover_ms`, `pre_speech_buffer_ms` (đã tích hợp sẵn trong từng VAD) và
  không có đồng hồ im lặng riêng (`silence_duration_ms` được CHIẾU xuống engine).
- Không resample, không gain, không clip, không chuyển float→int16 trên đường tới ASR:
  byte tới `on_speech_chunk` chính là byte client gửi.

Các bảo đảm giữ nguyên từ bản cũ:
- QWEN-Q2: KHÔNG nạp/tải model trên event loop hay trên đường hot (`peek_engine` +
  thread nền).
- QWEN-Q4: callback luôn được gọi NGOÀI `self._lock`.
- QWEN-Q5: `reset()` gán state MỚI (identity) để batch đang bay bị bỏ.
- P1.9: đổi engine không block event loop.
"""

from collections import deque
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.config import config
from backend.core.metrics import metrics_collector
from backend.core.pipeline_events import VADState
from backend.vad.base import BaseVADEngine, VADResult, VADStreamState
from backend.vad.engines import VADEngineFactory, SUPPORTED_VAD_ENGINES
from backend.utils.logger import logger

#: Sentinel cho `update_config(silence_duration_ms=...)`: phân biệt "không truyền" (giữ
#: nguyên) với "truyền None/0" (xoá cấu hình phiên ⇒ dùng mặc định của engine).
_UNSET: object = object()


class VADStreamProcessor:
    """Bộ xử lý VAD streaming cách ly theo phiên (Session-Safe)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_engine: str = "firered-vad",
        threshold: Optional[float] = None,
        silence_duration_ms: Optional[int] = None,
        enabled: bool = True,
        on_speech_chunk: Optional[Callable[[bytes, float, str], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        engine_override: Optional[BaseVADEngine] = None,
        auto_load: bool = True,
    ):
        """
        Args:
            threshold: ngưỡng VAD của phiên; `None` = dùng mặc định của engine đang chọn.
            silence_duration_ms: độ dài im lặng để chốt câu; `None` (hoặc 0) = để chính
                VAD quyết định theo config riêng của nó.
        """
        self.sample_rate = sample_rate
        self.vad_engine = (vad_engine or "firered-vad").lower().strip()
        if self.vad_engine not in SUPPORTED_VAD_ENGINES:
            self.vad_engine = "firered-vad"

        self.threshold: Optional[float] = None if threshold is None else float(threshold)
        self.silence_duration_ms: Optional[int] = self._normalize_silence(silence_duration_ms)
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
        self._engine_not_ready_warned: bool = False
        # P4.6: đánh dấu đã force_end/reset. Vì model chạy NGOÀI lock nên một batch có thể
        # đang bay khi force_end() được gọi; kết quả của batch đó phải bị BỎ.
        self._stopped: bool = False

        self._frame_samples: int = 400
        self._frame_size_bytes: int = self._frame_samples * 2  # 16-bit PCM
        self._max_lookback_frames: int = 0

        self._overflow_count: int = 0

        if engine_override is not None:
            self._attach_engine(engine_override)
            self._state = self._new_state(engine_override)
        else:
            # ══ QWEN-Q2 (P0) ══════════════════════════════════════════════════════
            # `__init__` chạy TRÊN EVENT LOOP. Bản cũ gọi `_ensure_engine()` ở đây, đường
            # xuống `VADEngineFactory.get_engine()` — mà constructor engine có thể gọi
            # `hf_hub_download` (hàng phút nếu mạng chậm) TRONG lock cấp lớp ⇒ asyncio
            # không preempt được code đồng bộ ⇒ TOÀN BỘ backend đứng hình.
            # Nay `__init__` chỉ làm việc RẺ: engine đã có trong pool ⇒ lấy tham chiếu
            # (state để VAD worker dựng ở chunk đầu); chưa có ⇒ hẹn nạp ở thread nền.
            engine = VADEngineFactory.peek_engine(self.vad_engine)
            if engine is not None:
                self._attach_engine(engine)
            elif auto_load:
                self._desired_engine = self.vad_engine
                self._spawn_engine_load(self.vad_engine)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_silence(value: Optional[int]) -> Optional[int]:
        """0 hoặc None ⇒ `None` = để engine dùng đúng mặc định trong docs của nó."""
        if value is None:
            return None
        try:
            ms = int(value)
        except (TypeError, ValueError):
            return None
        return None if ms <= 0 else ms

    def _attach_engine(self, engine: BaseVADEngine) -> None:
        """Gắn engine và đồng bộ hình học frame của nó (caller phải giữ lock khi cần)."""
        self._engine = engine
        self._frame_samples = max(1, int(getattr(engine, "frame_samples", 400)))
        self._frame_size_bytes = self._frame_samples * 2
        self._max_lookback_frames = max(0, int(getattr(engine, "max_lookback_frames", 0)))

    def _new_state(self, engine: BaseVADEngine) -> VADStreamState:
        """Tạo state cho engine. Có thể tốn thời gian (Silero nạp JIT model).

        ⚠️ `maxlen` của vòng pre-roll lấy TRỰC TIẾP từ engine (không đọc
        `self._max_lookback_frames`) để không phụ thuộc thứ tự gọi `_attach_engine` —
        bản đầu đọc `self._max_lookback_frames` nên `prewarm()` (gắn engine SAU khi dựng
        state) bị chốt `maxlen=1` và mất gần hết pre-roll.
        """
        state = engine.create_initial_state(
            threshold=self.threshold,
            silence_ms=self.silence_duration_ms,
        )
        max_lookback = max(0, int(getattr(engine, "max_lookback_frames", 0)))
        state.pre_roll = deque(maxlen=max(1, max_lookback))
        return state

    def _new_bare_state(self) -> VADStreamState:
        """QWEN-Q5: `VADStreamState` **RỖNG** (chưa có state của engine).

        Dùng cho `reset()`: rẻ O(1) nên gọi được trên event loop, nhưng vẫn cho **identity
        mới** để batch đang bay bị bỏ. Engine tự dựng state ở frame kế tiếp.

        ⚠️ Phải gán `pre_roll` có `maxlen`: `VADStreamState` khai `default_factory=deque`
        **không giới hạn**, nên state rỗng dựng thẳng sẽ rò RAM.
        """
        state = VADStreamState()
        state.pre_roll = deque(maxlen=max(1, self._max_lookback_frames))
        return state

    def _ensure_engine(self) -> bool:
        """Đảm bảo engine VÀ session state sẵn sàng. **KHÔNG BAO GIỜ tải/nạp model.**

        QWEN-Q2: chỉ lấy engine **đã có sẵn trong pool** (`peek_engine`, chỉ đọc biến);
        nếu chưa có thì hẹn nạp ở thread nền và trả `False` để caller BỎ chunk thay vì treo.
        """
        if self._engine is None:
            engine = VADEngineFactory.peek_engine(self.vad_engine)
            if engine is None:
                self._spawn_engine_load(self.vad_engine)
                return False
            self._attach_engine(engine)

        if self._state is None:
            # Chậm với Silero (nạp JIT model) — nhưng đây là VAD worker thread, KHÔNG phải
            # event loop, nên không treo đường mạng.
            self._state = self._new_state(self._engine)
        return True

    def prewarm(self) -> bool:
        """QWEN-Q2: nạp engine + state **ĐỒNG BỘ**. CHỈ gọi từ thread nền."""
        engine = VADEngineFactory.get_engine(self.vad_engine)
        with self._lock:
            # Gắn engine TRƯỚC khi dựng state để hình học frame/pre-roll luôn khớp.
            self._attach_engine(engine)
        state = self._new_state(engine)
        with self._lock:
            self._state = state
            self._desired_engine = None
        return True

    # ------------------------------------------------------------------ cấu hình
    def apply_engine_change(self, engine_name: str) -> None:
        """Kích hoạt engine mong muốn (off-lock). Trả về im lặng nếu chưa nạp sẵn.

        P1.9: KHÔNG gọi factory nếu engine chưa có trong pool — tránh tải model từ
        HuggingFace trên đường hot.

        Nếu đang giữa câu, câu hiện tại được CHỐT (`on_speech_end`) trước khi đổi engine,
        để không có câu nào bị treo ở phía ASR.
        """
        engine = VADEngineFactory.peek_engine(engine_name)
        if engine is None:
            self._spawn_engine_load(engine_name)
            return
        try:
            state = self._new_state(engine)  # off-lock
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Không tạo được state cho engine '{engine_name}': {exc}", extra={"module_tag": "VAD"})
            return

        pending_end: Optional[Callable[[], None]] = None
        with self._lock:
            if self._state is not None and self._state.is_speech:
                pending_end = self.on_speech_end
            self.vad_engine = engine_name
            self._attach_engine(engine)
            self._state = state
            self._desired_engine = None
        if pending_end:
            self._safe_callback(pending_end)
        logger.info(
            f"Đã kích hoạt engine '{engine_name}' "
            f"(hop={self._frame_samples} mẫu, {self._frame_samples / self.sample_rate * 1000:.0f} ms, "
            f"pre-roll tối đa={self._max_lookback_frames} frame)",
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
                if self._engine is None or self._state is None:
                    self.apply_engine_change(engine_name)
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
        silence_duration_ms: Union[int, None, object] = _UNSET,
        enabled: Optional[bool] = None,
    ) -> None:
        """Cập nhật cấu hình runtime. KHÔNG BAO GIỜ block (P1.9).

        Việc đổi engine chỉ ghi lại yêu cầu; kích hoạt thật diễn ra ở `feed_chunk` trên
        VAD worker thread, ngoài lock. `threshold`/`silence_duration_ms` được truyền xuống
        engine ở MỖI frame nên có hiệu lực ngay mà không cần dựng lại state.

        `silence_duration_ms` dùng sentinel `_UNSET`: truyền `None` (hoặc 0) là **xoá cấu
        hình phiên** ⇒ engine quay về đúng mặc định trong docs của nó; không truyền thì giữ
        nguyên giá trị hiện tại.
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
                changes.append(f"threshold: {self.threshold} -> {float(threshold):.2f}")
                self.threshold = float(threshold)
            if silence_duration_ms is not _UNSET:
                new_silence = self._normalize_silence(silence_duration_ms)  # type: ignore[arg-type]
                if new_silence != self.silence_duration_ms:
                    changes.append(
                        f"silence: {self.silence_duration_ms} -> "
                        f"{new_silence if new_silence is not None else 'mặc định của engine'}"
                    )
                    self.silence_duration_ms = new_silence
            if enabled is not None and bool(enabled) != self.enabled:
                changes.append(f"enabled: {self.enabled} -> {bool(enabled)}")
                self.enabled = bool(enabled)

        if changes:
            logger.info(
                f"Đồng bộ cấu hình VAD: {', '.join(changes)} "
                    f"(engine={self.vad_engine}, silence={self.silence_duration_ms}, "
                    f"threshold={self.threshold})",
                extra={"module_tag": "VAD"},
            )

    # ------------------------------------------------------------------ ingress
    def feed_chunk(self, audio_data: Union[bytes, np.ndarray], capture_timestamp: float = 0.0) -> None:
        """Xử lý nạp chunk âm thanh thô (bytes Int16 PCM hoặc np.ndarray)."""
        if audio_data is None or len(audio_data) == 0:
            return

        t_chunk = time.perf_counter()

        # P1.9: kích hoạt engine mong muốn ở đây (VAD worker thread), KHÔNG giữ lock.
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
                    "Đã TẮT VAD trong cấu hình, nhưng VAD là bắt buộc để phân câu — "
                        "nếu không chạy VAD thì sẽ KHÔNG có phụ đề nào. Đã tự bật lại VAD. "
                        "(Dùng min_words_to_commit / config riêng của VAD để tinh chỉnh.)",
                    extra={"module_tag": "VAD"},
                )
                metrics_collector.increment_counter("vad.enabled_flag_forced_on")
            self.enabled = True

        # QWEN-Q2: nếu engine chưa sẵn sàng thì BỎ chunk này thay vì nạp/tải model ngay
        # trên đường hot.
        if not self._ensure_engine():
            metrics_collector.increment_counter("vad.chunks_dropped_engine_not_ready")
            if not self._engine_not_ready_warned:
                self._engine_not_ready_warned = True
                logger.warning(
                    f"Engine VAD '{self.vad_engine}' chưa nạp xong — BỎ các chunk đến sớm "
                        f"(đang nạp ở thread nền, sẽ tự dùng ở chunk sau).",
                    extra={"module_tag": "VAD"},
                )
            return

        # P4.6 / F-13: TÁCH phần chạy model RA NGOÀI `self._lock` để `force_end()` /
        # `update_config()` từ thread khác (event loop) không bị block.
        frames: List[Tuple[bytes, float]] = []

        # --- Bước 1 (giữ lock): chỉ rút frame ra khỏi raw_buffer ---
        with self._lock:
            state = self._state
            raw_buf = state.raw_buffer
            bytes_per_sec = float(self.sample_rate * 2)

            # Đồng hồ của stream: byte đầu tiên của `raw_buffer`. Nếu buffer rỗng thì byte
            # đầu tiên chính là byte đầu của chunk này (⇒ nhận `capture_timestamp` của
            # client). Nếu còn byte thừa từ chunk trước thì GIỮ mốc cũ — nếu không,
            # timestamp của frame vắt qua nhiều chunk sẽ bị lệch.
            if len(raw_buf) == 0:
                state.stream_ts = float(capture_timestamp)

            # Chuyển đầu vào thành bytes 16-bit PCM cho raw_buffer. Đây là BẢN SAO vào
            # buffer của processor; audio gốc của client không bị sửa.
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
            max_buffer_bytes = int(bytes_per_sec * 3.0)
            if len(raw_buf) > max_buffer_bytes:
                overflow_bytes = len(raw_buf) - max_buffer_bytes
                del raw_buf[:overflow_bytes]
                state.stream_ts += overflow_bytes / bytes_per_sec
                self._overflow_count += 1

            frame_size = self._frame_size_bytes
            buf_len = len(raw_buf)
            offset = 0
            while buf_len - offset >= frame_size:
                frame_end = offset + frame_size
                frame_ts = state.stream_ts + (offset / bytes_per_sec)
                frames.append((bytes(raw_buf[offset:frame_end]), frame_ts))
                offset = frame_end

            if offset > 0:
                del raw_buf[:offset]
                state.stream_ts += offset / bytes_per_sec

            engine = self._engine

        # --- Bước 2 (KHÔNG giữ lock): chạy model cho từng frame ---
        results: List[Tuple[bytes, float, VADResult]] = []
        threshold = self.threshold
        silence_ms = self.silence_duration_ms
        for frame_bytes, frame_ts in frames:
            # View int16 CHỈ-ĐỌC trên chính byte sẽ gửi ASR ⇒ không copy, không đổi định dạng.
            frame_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
            res = engine.is_speech(
                frame_int16,
                state,
                threshold=threshold,
                silence_ms=silence_ms,
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

        Hai chế độ chốt câu:

        * **Mặc định docs** (`silence_duration_ms` là None/0): `START`/`END` của engine là
          nguồn sự thật duy nhất — engine dùng đúng tham số im lặng trong docs của nó.
        * **Ghi đè** (`silence_duration_ms > 0`, popup "VAD Silence" > 0): processor là bên
          chốt `END`, sau ĐÚNG `silence_duration_ms` im lặng kể từ frame cuối cùng có bằng
          chứng tiếng nói (`VADResult.is_speech`) — tức `silence_duration_ms` là **điều kiện
          số 1**, ghi đè `min_silence_frame` / `min_silence_duration_ms` /
          `max_end_silence_time`. Engine phát END sớm hơn thì bị bỏ qua; engine phát muộn
          hơn (hoặc không phát) thì processor vẫn chốt đúng hạn.
        """
        state.total_samples_processed += self._frame_samples
        prob = float(res.probability or 0.0)
        if res.is_speech:
            state.last_speech_sample = state.total_samples_processed

        if res.event == "START":
            if not state.is_speech:
                state.is_speech = True
                state.engine_end_pending = False
                state.engine_end_logged = False
                # Mốc đếm im lặng bắt đầu từ chính frame mở đoạn.
                state.last_speech_sample = max(state.last_speech_sample, state.total_samples_processed)
                logger.info(f"START (p={prob:.2f})", extra={"module_tag": "VAD"})
                if self.on_speech_start:
                    callbacks_to_fire.append((self.on_speech_start, ()))

                # Xả pre-roll: `lookback_frames` frame ĐÃ tiêu thụ nhưng thuộc đoạn nói
                # (do pre-padding của chính VAD). Chúng là byte NGUYÊN BẢN của client.
                n_back = min(max(0, int(res.lookback_frames)), len(state.pre_roll))
                if n_back > 0 and self.on_speech_chunk:
                    for pre_bytes, pre_ts in list(state.pre_roll)[-n_back:]:
                        callbacks_to_fire.append(
                            (self.on_speech_chunk, (pre_bytes, pre_ts, VADState.PRE_ROLL.value))
                        )
                state.pre_roll.clear()

            if self.on_speech_chunk:
                callbacks_to_fire.append(
                    (self.on_speech_chunk, (frame_bytes, frame_ts, VADState.SPEECH.value))
                )
            return

        if state.is_speech and self.silence_duration_ms is not None:
            # ── CHẾ ĐỘ GHI ĐÈ: `silence_duration_ms` là điều kiện số 1 ────────────────
            silent_ms = (state.total_samples_processed - state.last_speech_sample) / self.sample_rate * 1000.0
            # FIX-13: chỉ tôn trọng END "cắt cưỡng bức" khi ENGINE NÓI RÕ đó là trần cứng
            # của nó (`res.forced`). Bản cũ suy ra từ `res.is_speech` — SAI: frame END
            # thường là frame CHUYỂN TIẾP nên vẫn có bằng chứng nói (log phim thật
            # 2026-09-22 22:15: `END [engine_end, im lặng 0ms >= 1500ms] (p=1.00)` cắt đôi
            # câu "Don't try and trick me into buying something I don't want.").
            forced_split = bool(res.event == "END" and getattr(res, "forced", False))
            if forced_split or silent_ms >= float(self.silence_duration_ms):
                self._close_utterance(
                    state,
                    callbacks_to_fire,
                    reason=("engine_end" if forced_split else "silence_override"),
                    detail=(
                        f"cắt cưỡng bức bởi engine (im lặng {silent_ms:.0f}ms)"
                        if forced_split
                        else f"im lặng {silent_ms:.0f}ms >= {self.silence_duration_ms}ms"
                    ),
                    prob=prob,
                )
                state.pre_roll.append((frame_bytes, frame_ts))
                return
            if res.event == "END":
                # Engine muốn đóng nhưng CHƯA đủ im lặng theo yêu cầu ⇒ HOÃN. Ghi nhớ để nếu
                # người dùng gạt VAD Silence về 0 thì frame sau tôn trọng ngay (không treo câu).
                state.engine_end_pending = True
                if not state.engine_end_logged:
                    state.engine_end_logged = True
                    logger.info(
                        f"END của engine bị HOÃN: mới im lặng {silent_ms:.0f}ms "
                        f"< {self.silence_duration_ms}ms (p={prob:.2f}, is_speech={bool(res.is_speech)}) "
                        f"— giữ câu mở để không cắt giữa câu.",
                        extra={"module_tag": "VAD"},
                    )
            # Chưa đủ im lặng: giữ câu mở và tiếp tục chuyển tiếp audio.
            if self.on_speech_chunk:
                callbacks_to_fire.append(
                    (self.on_speech_chunk, (frame_bytes, frame_ts, VADState.SPEECH.value))
                )
            return

        if res.event == "END" or (state.engine_end_pending and state.is_speech):
            if state.is_speech:
                self._close_utterance(
                    state, callbacks_to_fire, reason="engine_end", detail="", prob=prob
                )
            # Frame kết thúc đoạn nằm trong phần im lặng ⇒ đệm cho đoạn kế tiếp.
            state.pre_roll.append((frame_bytes, frame_ts))
            return

        if state.is_speech:
            if self.on_speech_chunk:
                callbacks_to_fire.append(
                    (self.on_speech_chunk, (frame_bytes, frame_ts, VADState.SPEECH.value))
                )
        else:
            # SILENCE: giữ lại tối đa `max_lookback_frames` frame để xả khi START.
            state.pre_roll.append((frame_bytes, frame_ts))

    def _close_utterance(
        self,
        state: VADStreamState,
        callbacks_to_fire: List[Tuple[Callable, tuple]],
        *,
        reason: str,
        detail: str,
        prob: float,
    ) -> None:
        """Chốt câu hiện tại (SILENCE) và thu thập callback `on_speech_end`. Caller giữ lock."""
        state.is_speech = False
        suffix = f", {detail}" if detail else ""
        logger.info(f"END [{reason}{suffix}] (p={prob:.2f})", extra={"module_tag": "VAD"})
        if self.on_speech_end:
            callbacks_to_fire.append((self.on_speech_end, ()))

    @staticmethod
    def _safe_callback(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            logger.error(f"VAD callback error: {e}", exc_info=True, extra={"module_tag": "VAD"})

    def force_end(self) -> None:
        """Ép buộc kết thúc câu nói hiện tại và kích hoạt callback chốt câu.

        P4.6: hàm này KHÔNG bị block bởi inference VAD (model chạy ngoài lock). Đồng thời
        đánh dấu `_stopped` để batch inference đang bay bị bỏ.

        QWEN-Q4: callback được gọi **NGOÀI** `self._lock`.
        """
        callback: Optional[Callable[[], None]] = None
        with self._lock:
            self._stopped = True
            if self._state and self._state.is_speech:
                self._state.is_speech = False
                callback = self.on_speech_end
        if callback:
            self._safe_callback(callback)

    def reset(self) -> None:
        """Reset trạng thái processor (cho phép nhận audio trở lại).

        QWEN-Q5: phải gán `VADStreamState` **MỚI**, KHÔNG `reset()` in-place, vì
        `feed_chunk` bỏ kết quả batch đang bay bằng so sánh **IDENTITY**.
        """
        with self._lock:
            self._stopped = False
            self._state = self._new_bare_state()
            self._overflow_count = 0


# Alias tương thích ngược
VADProcessor = VADStreamProcessor
