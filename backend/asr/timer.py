"""Timer cho tầng SEG — lấy MỐC CẮT chính xác bằng model CÓ timestamp.

Vì sao cần
----------
Qwen3-ASR (model nhận dạng chính) **không có timestamp**:
`external/transcribe.cpp/src/arch/qwen3_asr/capabilities.cpp` đặt
`max_timestamp_kind = TRANSCRIBE_TIMESTAMPS_NONE`; xin `segment/word/token` ⇒ lỗi status 12.
Đo thực tế (`scratch/asr_timestamp_probe.py`): **whisper** trả timestamp mức SEGMENT và
**nemotron/parakeet** trả mức TOKEN (từng ký tự). Vì vậy:

    VAD → Qwen3 (văn bản tốt) → SEG (quyết định đã hết câu) → WHISPER (lấy mốc ms) → cắt

Module này gồm 2 phần tách bạch:
  * `build_char_timeline` + `locate_cut_ms` — **căn chỉnh văn bản thuần** (test được không
    cần model): ghép văn bản Qwen3 với văn bản Whisper bằng `difflib`, rồi tra ra mốc ms
    của ký tự ngay SAU ranh giới câu.
  * `WhisperTimer` — nạp model whisper qua binding (session RIÊNG, KHÔNG đụng
    `TranscribeEngine._shared_session`), chạy `timestamps="segment"`, rồi gọi hàm trên.
"""

from dataclasses import dataclass, field
import difflib
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple

from backend.segmentation.boundary import _IGNORED  # dùng chung định nghĩa "ký tự nội dung"

SAMPLE_RATE = 16000


def normalize_for_align(text: str) -> str:
    """Bỏ dấu câu/khoảng trắng để căn chỉnh theo KÝ TỰ (tiếng Nhật không có khoảng trắng)."""
    return "".join(ch for ch in (text or "") if ch not in _IGNORED)


@dataclass
class TimedChar:
    """Một ký tự nội dung kèm mốc thời gian (ms) nội suy trong segment chứa nó."""

    char: str
    start_ms: float
    end_ms: float


@dataclass
class TimerResult:
    """Kết quả định vị ranh giới câu trên trục thời gian."""

    cut_ms: float
    confidence: float
    method: str = "whisper_segment"
    segments: List[Any] = field(default_factory=list)
    matched_chars: int = 0
    elapsed_ms: float = 0.0

    @property
    def cut_sample(self) -> int:
        return int(round(self.cut_ms / 1000.0 * SAMPLE_RATE))


def build_char_timeline(segments: Sequence[Any]) -> List[TimedChar]:
    """Rải đều ký tự của từng segment trên khoảng `[t0_ms, t1_ms]` của nó.

    Whisper trả mốc theo SEGMENT (không theo từng ký tự), nên nội suy tuyến tính theo độ
    dài văn bản là xấp xỉ tốt nhất có được — sai số chỉ trong phạm vi 1 segment (~vài giây)
    và SEG chỉ dùng nó để chọn mốc cắt giữa hai câu.
    """
    out: List[TimedChar] = []
    for seg in segments or ():
        text = normalize_for_align(getattr(seg, "text", "") or "")
        if not text:
            continue
        t0 = float(getattr(seg, "t0_ms", 0.0) or 0.0)
        t1 = float(getattr(seg, "t1_ms", 0.0) or 0.0)
        if t1 <= t0:
            t1 = t0
        span = t1 - t0
        n = len(text)
        for i, ch in enumerate(text):
            out.append(TimedChar(
                char=ch,
                start_ms=t0 + span * (i / n),
                end_ms=t0 + span * ((i + 1) / n),
            ))
    return out


def locate_cut_ms(
    main_text: str,
    boundary_index: int,
    timeline: Sequence[TimedChar],
    *,
    min_confidence: float = 0.0,
) -> Optional[TimerResult]:
    """Tra mốc ms của ranh giới câu trong `main_text` (Qwen3) lên `timeline` (Whisper).

    `boundary_index` = vị trí KÝ TỰ trong `main_text` ngay sau câu đã chốt (tức chỗ câu
    tiếp theo bắt đầu). Trả mốc bắt đầu của ký tự đầu tiên của câu kế tiếp.

    Căn chỉnh bằng `difflib.SequenceMatcher` trên văn bản đã bỏ dấu câu — chịu được việc
    Whisper nhận sai/thiếu vài ký tự so với Qwen3.
    """
    if not main_text or not timeline:
        return None

    norm_main = normalize_for_align(main_text)
    if not norm_main:
        return None
    norm_whisper = "".join(tc.char for tc in timeline)

    # Vị trí ranh giới trong văn bản đã chuẩn hoá: đếm ký tự nội dung trước ranh giới.
    prefix = normalize_for_align(main_text[:boundary_index])
    pos = len(prefix)

    if pos >= len(norm_main):
        # Ranh giới ở CUỐI văn bản: lấy hết mốc cuối cùng.
        return TimerResult(
            cut_ms=float(timeline[-1].end_ms),
            confidence=0.5,
            matched_chars=0,
        )

    matcher = difflib.SequenceMatcher(None, norm_main, norm_whisper, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None

    matched_ratio = sum(b.size for b in blocks) / max(1, len(norm_whisper))
    mapped: Optional[int] = None
    confidence = 0.0

    for block in blocks:
        if block.a <= pos < block.a + block.size:
            mapped = block.b + (pos - block.a)
            left, right = pos - block.a, (block.a + block.size) - pos
            # Đủ ngữ cảnh hai bên ranh giới ⇒ tin cậy hơn hẳn.
            confidence = matched_ratio * (1.0 if left >= 1 and right >= 1 else 0.6)
            break

    if mapped is None:
        # Ranh giới rơi vào vùng Whisper nhận SAI: lấy mép block gần nhất.
        before = [b for b in blocks if b.a + b.size <= pos]
        after = [b for b in blocks if b.a > pos]
        if before:
            best = max(before, key=lambda b: b.a + b.size)
            mapped = best.b + best.size
        elif after:
            best = min(after, key=lambda b: b.a)
            mapped = best.b
        confidence = matched_ratio * 0.4
    if mapped is None:
        return None

    mapped = max(0, min(mapped, len(timeline) - 1))
    if confidence < min_confidence:
        return None
    return TimerResult(
        cut_ms=float(timeline[mapped].start_ms),
        confidence=confidence,
        matched_chars=sum(b.size for b in blocks),
    )


class WhisperTimer:
    """Chạy whisper (timestamp SEGMENT) trên một mảnh audio để lấy mốc cắt.

    Session RIÊNG: tuyệt đối không dùng `TranscribeEngine._shared_session` (đang giữ model
    nhận dạng chính) — đổi model ở đó sẽ làm hỏng phiên đang chạy.
    """

    def __init__(
        self,
        model_key: str = "whisper-large-v3-turbo",
        language: str = "ja",
        backend: str = "",
        threads: int = 4,
    ):
        self.model_key = model_key
        self.language = language or "ja"
        self.backend = backend
        self.threads = int(threads)
        self._model: Any = None
        self._session: Any = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None
        self.last_error: str = ""
        self.load_ms: float = 0.0

    # ------------------------------------------------------------------ vòng đời
    def available(self, registry: Any = None) -> bool:
        """True nếu file GGUF của model timer có sẵn cục bộ (không chạm mạng)."""
        try:
            from backend.asr.registry import ModelRegistry

            reg = registry or ModelRegistry.get_instance()
            import os

            return os.path.isfile(reg.resolve_model_path(self.model_key))
        except Exception:  # noqa: BLE001
            return False

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:
                return self._session
            if self._load_error:
                return None
            try:
                import transcribe_cpp  # noqa: PLC0415

                from backend.asr.native import resolve_backend  # noqa: PLC0415
                from backend.asr.registry import ModelRegistry  # noqa: PLC0415

                reg = ModelRegistry.get_instance()
                path = reg.resolve_model_path(self.model_key)
                t0 = time.perf_counter()
                self._model = transcribe_cpp.Model(path, backend=resolve_backend(self.backend or "cuda"))
                self._session = self._model.session(n_threads=self.threads)
                self.load_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                self._session = None
        return self._session

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:  # noqa: BLE001
                    pass
                self._session = None
            if self._model is not None:
                try:
                    self._model.close()
                except Exception:  # noqa: BLE001
                    pass
                self._model = None

    # ------------------------------------------------------------------ suy luận
    def run_segments(self, pcm_audio: Any, language: Optional[str] = None) -> List[Any]:
        """Chạy whisper với `timestamps="segment"`; trả danh sách segment (rỗng nếu lỗi).

        `language` PHẢI theo ngôn ngữ của phiên (mặc định lấy `self.language`): trước đây
        timer bị hard-code `"ja"` nên phiên tiếng Anh vẫn đưa "ja" cho Whisper ⇒ mốc thời
        gian kém chính xác (`language="auto"` để Whisper tự dò khi phiên ở chế độ auto).
        """
        session = self._ensure_session()
        if session is None:
            self.last_error = self._load_error or "session chưa sẵn sàng"
            return []
        lang = (language or self.language or "auto").strip() or "auto"
        try:
            import transcribe_cpp  # noqa: PLC0415

            res = session.run(
                pcm_audio,
                language=lang,
                family=transcribe_cpp.WhisperRunOptions(),
                timestamps="segment",
            )
        except Exception as exc:  # noqa: BLE001
            # KHÔNG nuốt im lặng: lỗi timer phải nhìn thấy được (đã từng làm timer
            # "miss" mọi câu mà không rõ lý do).
            self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                import logging  # noqa: PLC0415

                logging.getLogger("backend.asr.timer").warning(
                    "Whisper timer lỗi: %s", self.last_error
                )
            except Exception:  # noqa: BLE001
                pass
            return []
        self.last_error = ""
        return list(getattr(res, "segments", ()) or ())

    def locate_boundary(
        self,
        pcm_audio: Any,
        boundary_index: int,
        main_text: str,
        *,
        min_confidence: float = 0.0,
        language: Optional[str] = None,
    ) -> Optional[TimerResult]:
        """Chạy whisper trên `pcm_audio` rồi định vị ranh giới câu của `main_text`."""
        t0 = time.perf_counter()
        segments = self.run_segments(pcm_audio, language=language)
        if not segments:
            return None
        timeline = build_char_timeline(segments)
        res = locate_cut_ms(main_text, boundary_index, timeline, min_confidence=min_confidence)
        if res is not None:
            res.segments = segments
            res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res


_TIMER: Optional[WhisperTimer] = None


def get_seg_timer() -> WhisperTimer:
    """Singleton timer cho tầng SEG (tạo lười; test có thể monkeypatch hàm này)."""
    global _TIMER
    if _TIMER is None:
        from backend.config import config  # noqa: PLC0415

        seg = config.segmentation
        _TIMER = WhisperTimer(
            model_key=getattr(seg, "whisper_model_key", "whisper-large-v3-turbo"),
            language=(getattr(config.asr, "language", "") or "auto"),
        )
    return _TIMER
