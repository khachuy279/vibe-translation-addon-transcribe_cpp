"""TranscribeEngine: Engine ASR Streaming kết nối transcribe.cpp C++ Backend.

Đặc tính kỹ thuật:
- Streaming token generator không độ trễ, nhịp preview cố định (P2.4).
- Cửa sổ preview có giới hạn (P2.3): CHỈ preview bị cửa sổ hoá, commit luôn dùng
  toàn bộ ngữ cảnh câu => chi phí preview bị chặn trên mà không mất độ chính xác.
- Commit Manager 4 bậc được wire thật (P2.1): VAD_SILENCE > MAX_DURATION >
  STABLE_PREFIX > TIMEOUT_FORCE.
- Hot-Switch model an toàn: nạp model mới trước rồi swap, không gián đoạn preview.
- Tích hợp Speech Normalization thích ứng (đọc cấu hình thật từ ASRConfig).
- Định dạng và lọc sạch output văn bản theo thời gian thực.

Ghi chú an toàn luồng (P1.8b):
- Thứ tự lock DUY NHẤT: `_infer_lock` -> `_shared_lock`. Không có đường nào lấy
  ngược lại, nên không thể deadlock.
- `unload_shared_model()` và `prepare_model()` đều PHẢI giữ `_infer_lock` trước khi
  đóng native handle. Trước đây `unload_shared_model` chỉ giữ `_shared_lock`, nên
  `POST /api/config` có thể giải phóng session native trong lúc một inference đang
  chạy -> use-after-free (crash tiến trình).
- Thư viện transcribe.cpp KHÔNG tự bảo vệ: header ghi rõ chỉ một `run()`/stream được
  phép in-flight trên toàn bộ session của một model. Vì vậy `_infer_lock` là bắt buộc,
  không được gỡ.
"""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import difflib
import logging
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
import numpy as np

# `backend/asr/__init__.py` đã gọi `backend.asr.native.bootstrap()` TRƯỚC khi module này
# được nạp (mọi đường vào đều đi qua package `backend.asr`). Lúc này `TRANSCRIBE_LIBRARY`
# đã trỏ đúng bundle trong `backend/bin/` (nếu có) và các thư mục DLL phụ thuộc đã được đăng ký,
# nên `import transcribe_cpp` bên dưới dlopen đúng thư viện mong muốn.
try:
    import transcribe_cpp
except ImportError:
    transcribe_cpp = None

from backend.config import config, SentenceConfig
from backend.core.audio_buffer import CircularAudioBuffer
from backend.core.commit_manager import CommitManager, count_content_tokens
from backend.segmentation import (
    SentenceCompleter,
    StreamingSegmenter,
    split_complete_sentences,
)
from backend.core.metrics import metrics_collector
from backend.core.normalizer import SpeechNormalizer
from backend.core.pipeline_events import CommitReason
from backend.asr.base import BaseASREngine
from backend.asr.registry import ModelRegistry
from backend.asr.adapters import build_family_options, normalize_language_for_family
from backend.asr.text_cleaner import clean_transcript_text
from backend.asr.native import resolve_backend, native_bundle_dir
from backend.core.gpu_scheduler import gpu_arbiter
from backend.utils.logger import logger

# B5-1 (Hy3): số worker khớp với trần inference đồng thời. Trước đây hardcode 2 trong khi
# `config.asr.max_inflight_infer = 1` ⇒ worker thứ hai không bao giờ có việc (một thread nền
# thường trực dư thừa). Nay suy ra từ cấu hình; nếu người dùng nâng `max_inflight_infer`
# thì executor cũng nới theo. Native transcribe.cpp vẫn giới hạn 1 stream in-flight/session
# nên giá trị >1 chỉ hữu ích khi có nhiều session (xem A2-1/T2 trong báo cáo audit).
_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(getattr(config.asr, "max_inflight_infer", 1) or 1)),
    thread_name_prefix="asr_worker",
)

# Trần số commit chờ xử lý trước khi gộp (P4.5). Không còn `maxlen` vứt câu âm thầm.
_MAX_PENDING_COMMITS = 6


def _best_matching_sentence(parts: List[str], target: str) -> Optional[str]:
    """Chọn câu trong `parts` khớp nhất với `target` (câu mà SEG đã chốt).

    Dùng khi văn bản chạy lại trên mảnh đã cắt chứa NHIỀU câu: ta muốn giữ ĐÚNG câu đã
    quyết định, không phải "câu đầu tiên" (câu đầu có thể chỉ là mảnh ngắn như `Okay.`,
    và giữ nó sẽ làm câu bị coi là quá ngắn ⇒ gộp lại ⇒ LẶP).
    """
    if not parts:
        return None
    if not target:
        return max(parts, key=len)
    from backend.asr.timer import normalize_for_align as _norm

    want = _norm(target)
    best, best_score = parts[0], -1.0
    for part in parts:
        score = difflib.SequenceMatcher(None, _norm(part), want, autojunk=False).ratio()
        if score > best_score:
            best, best_score = part, score
    return best


class TranscribeEngine(BaseASREngine):
    """Engine nhận dạng giọng nói ASR streaming qua transcribe.cpp."""

    # Model singleton dùng chung trên GPU
    _shared_model: Optional[Any] = None
    _shared_model_key: Optional[str] = None
    _shared_session: Optional[Any] = None
    _shared_supports_streaming: bool = False
    # P1.5: trần audio (samples) của session dùng chung. 0 = không biết.
    _shared_max_audio_samples: int = 0
    # F-39: RSS ngay sau khi nạp model xong — mốc so sánh để phát hiện phình bộ nhớ
    # phía native (xem `maybe_recycle_native`).
    _shared_rss_baseline_mb: float = 0.0
    # Đánh dấu đã warm tới độ dài lớn nhất chưa. Cần thiết vì `prewarm()` lúc khởi động
    # chỉ warm đoạn ngắn; nếu `_preload_model()` cứ early-return khi model đã nạp thì
    # runtime sẽ KHÔNG bao giờ warm độ dài dài và câu dài đầu tiên vẫn bị "đơ".
    _shared_warmed_long: bool = False
    _shared_lock = threading.RLock()
    _infer_lock = threading.RLock()

    @staticmethod
    def _session_was_truncated(session: Any) -> bool:
        """Đọc cờ truncation. Binding Python 0.2.3 chưa expose `was_truncated`,
        nhưng C API `transcribe_was_truncated` thì có — gọi qua `_lib` nếu cần."""
        flag = getattr(session, "was_truncated", None)
        if isinstance(flag, bool):
            return flag
        try:
            lib = getattr(transcribe_cpp, "_lib", None)
            handle = getattr(session, "_h", None)
            if lib is not None and handle is not None:
                return bool(lib.transcribe_was_truncated(handle))
        except Exception:
            pass
        return False

    @classmethod
    def shutdown_executors(cls, wait: bool = False) -> None:
        """Dừng các luồng worker executor."""
        try:
            _EXECUTOR.shutdown(wait=wait, cancel_futures=True)
        except Exception:
            pass

    @classmethod
    def unload_shared_model(cls) -> None:
        """Giải phóng hoàn toàn model và session khỏi GPU VRAM.

        P1.8b: giữ `_infer_lock` TRƯỚC `_shared_lock` để không bao giờ đóng native
        handle khi một inference đang chạy (tránh use-after-free).
        """
        with cls._infer_lock:
            with cls._shared_lock:
                if cls._shared_session is not None:
                    try:
                        cls._shared_session.close()
                    except Exception:
                        pass
                    cls._shared_session = None

                if cls._shared_model is not None:
                    try:
                        cls._shared_model.close()
                    except Exception:
                        pass
                    cls._shared_model = None
                    cls._shared_model_key = None
                    cls._shared_supports_streaming = False
                    logger.info("Model đã giải phóng khỏi GPU", extra={"module_tag": "ASR"})

    @staticmethod
    def _process_rss_mb() -> float:
        """RSS của tiến trình (MB). Trả 0 nếu không đọc được (khi đó mọi guard tự tắt)."""
        try:
            import psutil  # type: ignore

            return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
        except Exception:  # noqa: BLE001
            pass
        try:  # pragma: no cover - phụ thuộc nền tảng
            import ctypes
            import ctypes.wintypes as wt

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wt.DWORD),
                    ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    async def maybe_recycle_native(self) -> bool:
        """F-39 (phần native): đóng phiên native đã phình để lần sau nạp lại sạch.

        Vì sao: đo được bộ nhớ phình **bên trong** lời gọi native `transcribe_run`
        (Vulkan) — xem `report/audit/05_measurements_and_status.md` §12.6. Python không
        thể giải phóng phần đó, nhưng CÓ THỂ đóng session/model khi phiên đã xong và
        RAM đã vượt nền lúc nạp model một khoảng lớn.

        An toàn: chỉ gọi ở cuối vòng đời phiên (`cleanup()`), khi pipeline đã dừng.
        `unload_shared_model()` tự giữ `_infer_lock` nên không thể đóng handle khi còn
        inference đang chạy. Trả True nếu đã đóng.
        """
        delta_mb = float(getattr(config.asr, "native_recycle_rss_delta_mb", 2048.0) or 0.0)
        if delta_mb <= 0:
            return False
        baseline = self.__class__._shared_rss_baseline_mb
        current = self._process_rss_mb()
        if baseline <= 0 or current <= 0:
            return False
        growth = current - baseline
        if growth < delta_mb:
            return False

        metrics_collector.increment_counter("asr.native_session_recycled")
        metrics_collector.record_gauge("asr", "recycle_growth_mb", growth)
        logger.warning(
            f"Tái nạp native session (RAM delta: +{growth:.0f} MB > ngưỡng {delta_mb:.0f} MB)",
            extra={"module_tag": "ASR"},
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_EXECUTOR, self.__class__.unload_shared_model)
        self.__class__._shared_rss_baseline_mb = 0.0
        return True

    def __init__(
        self,
        model_key: Optional[str] = None,
        language: str = "auto",
        backend: str = "auto",
        threads: int = 4,
        min_transcribe_sec: Optional[float] = None,
        poll_interval_ms: Optional[int] = None,
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.registry = ModelRegistry.get_instance()
        self.model_key = model_key or self.registry.get_active_model_key()
        self.model_info = self.registry.get_model_info(self.model_key) or {}
        self.language = language or config.asr.language
        self.backend = backend or config.asr.backend
        self.threads = threads or config.asr.threads
        self.min_transcribe_sec = (
            float(min_transcribe_sec) if min_transcribe_sec is not None else float(config.asr.min_transcribe_sec)
        )
        self.poll_interval_ms = (
            int(poll_interval_ms) if poll_interval_ms is not None else int(config.asr.poll_interval_ms)
        )

        # FIX-07: đọc từ config — trước đây hard-code `capacity_sec=60.0` nên
        # `AudioBufferConfig.capacity_sec` hoàn toàn không có tác dụng.
        self.audio_buffer = CircularAudioBuffer(
            sample_rate=int(getattr(config.audio_buffer, "sample_rate", 16000) or 16000),
            capacity_sec=float(getattr(config.audio_buffer, "capacity_sec", 60.0) or 60.0),
        )
        self.normalizer = SpeechNormalizer.from_config(config.asr)
        self.commit_manager = CommitManager(sentence_cfg=config.sentence)

        self._active_utterance_id: str = str(uuid.uuid4())[:8]
        self._speech_active: bool = False
        self._speech_start_sample: int = 0
        self._last_committed_sample: int = 0
        # P1.3: pre-roll phải được tính SAU khi VAD flush các frame pre-roll vào buffer.
        self._awaiting_pre_roll: bool = False
        # P4.5: bounded nhưng GỘP thay vì vứt câu cũ.
        self._pending_commits: deque = deque()
        self._commits_merged: int = 0
        # P2.5: kết quả preview gần nhất để có thể tái sử dụng cho commit.
        self._last_preview_end_sample: int = -1
        self._last_preview_text: str = ""
        # K2: mốc đo thời gian tới preview đầu tiên (đặt lại mỗi khi VAD báo bắt đầu nói).
        self._speech_started_at: float = 0.0
        self._first_preview_reported: bool = True
        # K4: mốc VAD báo ngừng nói (để đo E2E tới phụ đề chốt).
        self._speech_ended_at: float = 0.0
        # F-44: thế hệ stream — tăng mỗi lần tua video để vô hiệu hoá kết quả cũ.
        self._stream_generation: int = 0
        # F-47: đếm số commit liên tiếp không phát gì (rào chắn chống quay nóng vòng lặp).
        self._empty_commit_streak: int = 0
        self._commit_mgr_primed: bool = False

        self._commit_event: asyncio.Event = asyncio.Event()  # đánh thức stream_tokens ngay khi có commit
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None  # set khi stream_tokens bắt đầu
        self._is_running: bool = True
        self._lock = threading.RLock()
        # SEG (VAD > ASR > SEG): chốt câu theo dấu câu của ASR + mốc cắt từ timer.
        seg_cfg = getattr(config, "segmentation", None)
        self._seg_cfg = seg_cfg
        self._seg: Optional[StreamingSegmenter] = None
        if seg_cfg is not None and bool(getattr(seg_cfg, "enabled", False)):
            self._seg = self._build_seg(seg_cfg)
        self._seg_decision: Optional[Any] = None
        self._seg_commits: int = 0
        self._seg_timer_ms_total: float = 0.0
        self._seg_timer_prewarmed: bool = False
        #: Mảnh SEG vừa bị "gộp vào câu kế tiếp" — dùng để phát hiện VÒNG LẶP (FIX-12b).
        self._last_carried_text: str = ""
        self._seg_trace: bool = bool(getattr(seg_cfg, "debug_trace", False)) if seg_cfg else False
        # SEG "bóng": khi SEG TẮT vẫn cho biết nó SẼ cắt ở đâu (so sánh A/B trong 1 lần chạy).
        self._seg_shadow: Optional[StreamingSegmenter] = None
        self._refresh_seg_shadow()
        # P1.5: trần audio hợp lệ của session (samples). 0 = không biết.
        self._max_audio_samples: int = 0
        self._window_warned: bool = False
        # P2.4b: nhịp preview hiệu dụng (tự giãn ra khi backend chậm, tự thu lại khi nhanh).
        self._eff_poll_interval: float = self.poll_interval_ms / 1000.0
        self._preview_durations: deque = deque(maxlen=8)
        # FIX-10: đo mức LÃNG PHÍ compute của preview (tích luỹ theo từng câu).
        #   processed = tổng số giây audio đã đưa qua model cho các lượt preview
        #   unique    = số giây audio THẬT của câu (đỉnh của "đã nói được bao nhiêu")
        # ratio = processed / unique. ratio = 8 nghĩa là mỗi giây audio bị encode/decode 8 lần.
        # Đây là số liệu QUYẾT ĐỊNH để biết có đáng tối ưu preview hay không (đo trước, sửa sau).
        self._preview_sec_processed: float = 0.0
        self._preview_sec_unique_peak: float = 0.0
        # Số inference đang chạy trong executor. Dùng để biết pipeline còn việc hay không
        # (`_pending_commits` rỗng KHÔNG có nghĩa là đã xử lý xong — commit bị pop ra
        # ngay khi bắt đầu inference).
        self._inflight: int = 0
        self._inflight_lock = threading.Lock()
        # Callback tuỳ chọn để thông báo trạng thái nạp model ra ngoài (WS).
        self.on_model_status: Optional[Callable[[Dict[str, Any]], None]] = None

    # ------------------------------------------------------------------ cấu hình
    def _build_seg(self, seg_cfg: Any) -> StreamingSegmenter:
        """Tạo một máy trạng thái SEG từ cấu hình (dùng cho cả SEG thật và SEG 'bóng')."""
        # SEG không được cắt câu ngắn hơn ngưỡng lọc của tầng trên: cắt ra rồi bị gộp lại
        # làm vùng audio không tiến ⇒ LẶP (xem log phim thật 2026-09-22). Ngưỡng này ĐI
        # CHUNG với "Min Words" ở popup (`min_words_to_commit`) để hai tầng không lệch nhau.
        min_words = int(getattr(config.sentence, "min_words_to_commit", 2) or 0)
        min_words = max(0, min_words)
        return StreamingSegmenter(
            SentenceCompleter(
                min_chars=int(getattr(seg_cfg, "min_chars", 2)),
                min_words=min_words,
                max_chars=int(getattr(seg_cfg, "max_chars", 60)),
                tail_min_chars=int(getattr(seg_cfg, "tail_min_chars", 4)),
                tail_scans=int(getattr(seg_cfg, "tail_scans", 2)),
                tail_stable_ms=float(getattr(seg_cfg, "tail_stable_ms", 280.0)),
                stable_ms=float(getattr(seg_cfg, "stable_ms", 350.0)),
                stable_scans=int(getattr(seg_cfg, "stable_scans", 2)),
                # Mặc định TẮT cắt-theo-độ-ổn-định: ASR hay thả dấu kết câu sớm giữa câu
                # (tiếng Nhật: `営業回りを終え。`, `夕食を済ませ。`) ⇒ chờ câu mới / im lặng VAD.
                allow_stable_cut=bool(getattr(seg_cfg, "stable_cut", False)),
            ),
            fallback_overlap_ms=float(getattr(seg_cfg, "fallback_overlap_ms", 600.0)),
        )

    def _refresh_seg_shadow(self) -> None:
        cfg = self._seg_cfg
        want = bool(
            cfg is not None
            and self._seg is None
            and self._seg_trace
            and bool(getattr(cfg, "shadow_when_disabled", True))
        )
        self._seg_shadow = self._build_seg(cfg) if want else None

    def update_seg_config(self, **kwargs: Any) -> None:
        """Cập nhật cấu hình tầng SEG runtime (từ popup/REST) và DỰNG LẠI máy trạng thái.

        Không dựng lại `_seg` thì mọi thay đổi `max_chars`/`tail_*` chỉ nằm trong config mà
        không có tác dụng (đúng loại bug "cấu hình vô hiệu" đã gặp ở BẬC 3 trước đây).
        """
        seg_cfg = getattr(config, "segmentation", None)
        if seg_cfg is None:
            return
        for key, value in kwargs.items():
            if value is None or not hasattr(seg_cfg, key):
                continue
            setattr(seg_cfg, key, value)
        self._seg_cfg = seg_cfg
        self._seg_trace = bool(getattr(seg_cfg, "debug_trace", False))
        if bool(getattr(seg_cfg, "enabled", False)):
            self._seg = self._build_seg(seg_cfg)
            self.prewarm_seg_timer()
        else:
            self._seg = None
        self._refresh_seg_shadow()
        self._seg_decision = None
        logger.info(
            f"SEG cấu hình lại: enabled={bool(getattr(seg_cfg, 'enabled', False))}, "
            f"max_chars={getattr(seg_cfg, 'max_chars', None)}, "
            f"tail_min_chars={getattr(seg_cfg, 'tail_min_chars', None)}, "
            f"tail_scans={getattr(seg_cfg, 'tail_scans', None)}, "
            f"timer={bool(getattr(seg_cfg, 'use_whisper_timer', False))}",
            extra={"module_tag": "ASR"},
        )

    def update_sentence_config(self, **kwargs) -> None:
        """Cập nhật cấu hình phân câu và ngắt đoạn runtime (P2.1 / G4).

        Trước đây các tham số này ghi vào `commit_manager.cfg` nhưng `decide_commit_trigger`
        không bao giờ được gọi nên hoàn toàn vô hiệu.
        """
        for k, v in kwargs.items():
            if hasattr(self.commit_manager.cfg, k):
                setattr(self.commit_manager.cfg, k, v)
        # `min_words_to_commit` cũng là ngưỡng lọc của tầng SEG (dùng chung) ⇒ dựng lại SEG.
        if "min_words_to_commit" in kwargs and self._seg is not None:
            self.update_seg_config()

    @property
    def preview_window_sec(self) -> float:
        """Độ dài cửa sổ preview, đảm bảo >= max_duration_sec (nguyên tắc P2)."""
        win = float(getattr(config.asr, "preview_window_sec", 0.0) or 0.0)
        if win <= 0.0:
            return 0.0
        floor = float(self.commit_manager.cfg.max_duration_sec)
        if win < floor:
            if not self._window_warned:
                self._window_warned = True
                logger.warning(
                    f"preview_window_sec={win:.2f}s < max_duration_sec={floor:.2f}s. "
                        f"Nâng cửa sổ lên {floor:.2f}s để preview không mất ngữ cảnh (nguyên tắc P2).",
                    extra={"module_tag": "ASR"},
                )
            return floor
        return win

    # ------------------------------------------------------------------ nạp model
    def _build_session(self, model: Any) -> Tuple[Any, bool, int]:
        """Tạo session + đọc capabilities/limits. Trả (session, supports_streaming, max_audio_samples)."""
        session = model.session(n_threads=self.threads)
        supports_streaming = bool(getattr(model.capabilities, "supports_streaming", False))
        max_audio_samples = 0
        try:
            limits = session.limits  # P1.5: binding exposes effective_max_audio_ms
            max_ms = int(getattr(limits, "effective_max_audio_ms", 0) or 0)
            if max_ms > 0:
                max_audio_samples = int(max_ms / 1000.0 * 16000)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Không đọc được session limits: {exc}", extra={"module_tag": "ASR"})
        return session, supports_streaming, max_audio_samples

    def _load_model_locked(self, model_key: str) -> Any:
        """Nạp model GGUF (giải phóng cái cũ trước). Caller PHẢI giữ `_shared_lock`."""
        if transcribe_cpp is None:
            raise RuntimeError("transcribe_cpp package chưa được cài đặt!")

        model_path = self.registry.resolve_model_path(model_key)
        if not os.path.exists(model_path):
            if getattr(config.asr, "auto_download", True):
                model_path = self.registry.ensure_model_file(model_key, allow_download=True)
            else:
                raise FileNotFoundError(f"File model GGUF không tồn tại: {model_path}")

        info = self.registry.get_model_info(model_key) or {}
        family = info.get("family", "")

        logger.info(
            f"Nạp model '{model_key}' ({Path(model_path).name})",
            extra={"module_tag": "ASR"},
        )

        # Giải phóng session/model cũ
        cls = self.__class__
        if cls._shared_session is not None:
            try:
                cls._shared_session.close()
            except Exception:
                pass
            cls._shared_session = None
        if cls._shared_model is not None:
            try:
                cls._shared_model.close()
            except Exception:
                pass
            cls._shared_model = None

        # Chọn backend CÓ FALLBACK + log rõ ràng (xem backend/asr/native.py).
        # `self.backend` là giá trị truyền lúc tạo engine; nếu rỗng thì lấy từ config.
        load_backend = resolve_backend(self.backend or config.asr.backend)
        loaded_model = transcribe_cpp.Model(model_path, backend=load_backend)

        session, supports_streaming, max_audio_samples = self._build_session(loaded_model)

        cls._shared_model = loaded_model
        cls._shared_model_key = model_key
        cls._shared_supports_streaming = supports_streaming
        cls._shared_session = session
        cls._shared_max_audio_samples = max_audio_samples
        self._max_audio_samples = max_audio_samples
        # F-39: ghi mốc RSS ngay sau khi nạp xong để sau này biết đã phình bao nhiêu.
        cls._shared_rss_baseline_mb = self._process_rss_mb()

        backend_name = getattr(loaded_model, "backend", "unknown")
        logger.info(
            f"Model '{model_key}' sẵn sàng (backend={backend_name}, streaming={supports_streaming})",
            extra={"module_tag": "ASR"},
        )
        return loaded_model

    def _ensure_model_loaded(self) -> Any:
        """Nạp model nếu chưa đúng model hiện tại (idempotent)."""
        cls = self.__class__
        with cls._shared_lock:
            if cls._shared_model is not None and cls._shared_model_key == self.model_key:
                # Đồng bộ trần audio cho engine instance này khi model đã nạp sẵn.
                self._max_audio_samples = cls._shared_max_audio_samples
                return cls._shared_model
            return self._load_model_locked(self.model_key)

    def prepare_model(self, model_key: str, *, allow_download: Optional[bool] = None) -> None:
        """P1.8: nạp model mới rồi SWAP nguyên tử, không gián đoạn inference hiện tại.

        Model mới được nạp **ngoài** lock (chậm, vài giây) nên preview vẫn phục vụ
        bằng model cũ trong lúc nạp. Việc swap chỉ giữ `_infer_lock` trong thời gian
        rất ngắn (không có inference nào đang chạy), nên không có khoảng trống phụ đề.

        Phương thức này BLOCKING — phải gọi từ thread nền (asyncio.to_thread).
        """
        cls = self.__class__
        if transcribe_cpp is None:
            raise RuntimeError("transcribe_cpp package chưa được cài đặt!")
        if cls._shared_model is not None and cls._shared_model_key == model_key:
            self.model_key = model_key
            self.model_info = self.registry.get_model_info(model_key) or {}
            return

        model_path = self.registry.resolve_model_path(model_key)
        if not os.path.exists(model_path):
            can_download = (
                bool(allow_download) if allow_download is not None
                else getattr(config.asr, "auto_download", True)
            )
            if can_download:
                model_path = self.registry.ensure_model_file(model_key, allow_download=True)
            else:
                raise FileNotFoundError(
                    f"File model GGUF không tồn tại cho '{model_key}': {model_path}"
                )

        info = self.registry.get_model_info(model_key) or {}
        logger.info(
            f"Nạp model '{model_key}' trước khi swap (zero-downtime)",
            extra={"module_tag": "ASR"},
        )
        t0 = time.perf_counter()

        # 1) Nạp ngoài lock: model cũ vẫn phục vụ preview bình thường.
        load_backend = resolve_backend(self.backend or config.asr.backend)
        new_model = transcribe_cpp.Model(model_path, backend=load_backend)
        new_session, supports_streaming, max_audio_samples = self._build_session(new_model)

        old_session = old_model = None
        with cls._infer_lock:  # 2) swap nguyên tử — không inference nào đang chạy
            with cls._shared_lock:
                old_session = cls._shared_session
                old_model = cls._shared_model
                cls._shared_model = new_model
                cls._shared_model_key = model_key
                cls._shared_supports_streaming = supports_streaming
                cls._shared_session = new_session
                cls._shared_max_audio_samples = max_audio_samples
                self.model_key = model_key
                self.model_info = info
                self._max_audio_samples = max_audio_samples

        # 3) Giải phóng model cũ SAU khi swap (an toàn: đang giữ _infer_lock).
        if old_session is not None:
            try:
                old_session.close()
            except Exception:
                pass
        if old_model is not None:
            try:
                old_model.close()
            except Exception:
                pass

        # 4) pre-warm model mới TRƯỚC khi báo "ready".
        # Nếu bỏ bước này, lần inference đầu tiên sau khi đổi model sẽ phải trả chi phí
        # warm-up của backend (build graph + compile kernel + cấp workspace) — đo được
        # ~7-10 giây — và preview đầu tiên bị "đơ" đúng lúc người dùng vừa đổi model.
        t_warm = time.perf_counter()
        try:
            self._begin_infer()
            try:
                self._run_inference_sync(np.zeros(int(16000 * 0.5), dtype=np.float32))
            finally:
                self._end_infer()
            warm_ms = (time.perf_counter() - t_warm) * 1000.0
            metrics_collector.record_metric("asr", "model_warmup_ms", warm_ms)
            logger.info(f"Pre-warm ({model_key}, {warm_ms:.0f}ms)",
                        extra={"module_tag": "ASR"})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"pre-warm model '{model_key}' thất bại: {exc}", extra={"module_tag": "ASR"})

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Đã swap sang '{model_key}' ({elapsed:.2f}s, backend={getattr(new_model, 'backend', 'unknown')})",
            extra={"module_tag": "ASR"},
        )

    def prewarm(self) -> None:
        """Prewarm mô hình trên GPU (nạp + pre-warm cả độ dài câu tối đa).

        Dùng CÙNG đường `_preload_model()` như lúc session bắt đầu, để hành vi lúc khởi
        động và lúc chạy không lệch nhau. Trước đây hàm này tự chạy một dummy 0,5 s rồi
        `_preload_model()` early-return vì model đã nạp ⇒ runtime không bao giờ warm độ
        dài dài và câu dài đầu tiên vẫn bị "đơ".
        """
        try:
            self._preload_model(force_warm=True)
        except Exception as e:
            logger.warning(f"Pre-warm ASR warning: {e}", extra={"module_tag": "ASR"})
        self.prewarm_seg_timer()

    def prewarm_seg_timer(self) -> None:
        """Nạp trước model TIMER của tầng SEG (whisper) trong luồng nền.

        Vì sao: timer chỉ được gọi ở lần CHỐT CÂU đầu tiên; nếu nạp lúc đó thì câu đầu
        tiên bị trễ thêm ~1 s (nạp model 845 MB) đúng lúc người dùng đang chờ phụ đề.
        Nạp nền nên không chặn khởi động phiên; nếu chưa kịp thì `_seg_cut_sample()` vẫn
        có đường dự phòng (chồng lấn) nên không mất chữ.
        """
        cfg = self._seg_cfg
        if self._seg is None or cfg is None:
            return
        if not bool(getattr(cfg, "use_whisper_timer", False)):
            return
        if getattr(self, "_seg_timer_prewarmed", False):
            return
        self._seg_timer_prewarmed = True
        try:
            from backend.asr.timer import get_seg_timer

            timer = get_seg_timer()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Không lấy được SEG timer để prewarm: {exc}", extra={"module_tag": "ASR"})
            return
        if timer is None or not timer.available():
            metrics_collector.increment_counter("seg.timer_unavailable")
            logger.info(
                "SEG timer: chưa có file model whisper cục bộ ⇒ dùng chồng lấn dự phòng.",
                extra={"module_tag": "ASR"},
            )
            return

        def _load() -> None:
            t0 = time.perf_counter()
            try:
                session = timer._ensure_session()  # noqa: SLF001 — cùng package, tránh API thừa
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Prewarm SEG timer lỗi: {exc}", extra={"module_tag": "ASR"})
                return
            elapsed = time.perf_counter() - t0
            if session is None:
                metrics_collector.increment_counter("seg.timer_prewarm_failed")
                logger.warning(
                    f"Prewarm SEG timer thất bại: {timer.last_error or 'không rõ lý do'}",
                    extra={"module_tag": "ASR"},
                )
                return
            metrics_collector.record_metric("seg", "timer_load_ms", elapsed * 1000.0)
            metrics_collector.increment_counter("seg.timer_prewarmed")
            logger.info(
                f"SEG timer sẵn sàng ({timer.model_key}, {elapsed:.2f}s) — mốc cắt theo "
                f"timestamp thật.",
                extra={"module_tag": "ASR"},
            )

        threading.Thread(target=_load, name="seg-timer-prewarm", daemon=True).start()

    # ------------------------------------------------------------------ ingress
    def feed_audio(self, audio_data: Any, timestamp: float = 0.0, vad_state: str = "") -> None:
        """Nạp dữ liệu audio (bytes PCM Int16 hoặc ndarray Float32) vào buffer.

        B3-1 (Hy3) — ĐÃ CÂN NHẮC VÀ GIỮ NGUYÊN: báo cáo audit đề xuất đổi ring buffer
        sang Int16 để "bỏ conversion kép". Kiểm tra lại mã gốc cho thấy tiền đề đó không
        đúng: `feed_audio` chỉ convert Int16->Float32 ĐÚNG MỘT LẦN rồi `write()` copy
        Float32 vào buffer (không convert thêm). Đổi buffer sang Int16 sẽ:
          * phá bảo đảm "Bit-Exact Integrity" mà `test_01_core_audio.py` đang chốt
            (float32 ghi vào rồi đọc ra phải khớp tuyệt đối; Int16 gây sai số 1 LSB
            ≈ 3.05e-5),
          * đẩy thêm một phép convert Int16->Float32 lên ĐƯỜNG ĐỌC `get_slice()` (chạy
            mỗi preview/commit), tức là đổi chỗ chứ không giảm việc,
          * chỉ tiết kiệm 1,92 MB RAM / session (3,84 MB -> 1,92 MB) — không đáng kể so
            với VRAM/RAM của pipeline.
        Xem `report/audit/KE_HOACH_FIX_LOI_Hy3.md` mục "B3-1 — bị loại".
        """
        if audio_data is None:
            return

        if isinstance(audio_data, (bytes, bytearray)):
            if len(audio_data) == 0:
                return
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            # FIX-06b: chia TẠI CHỖ (1 mảng tạm thay vì 2), kết quả float32 như cũ.
            audio_float32 = audio_int16.astype(np.float32)
            audio_float32 /= 32768.0
        elif isinstance(audio_data, np.ndarray):
            if len(audio_data) == 0:
                return
            if audio_data.dtype == np.int16:
                audio_float32 = audio_data.astype(np.float32)
                audio_float32 /= 32768.0
            else:
                audio_float32 = audio_data.astype(np.float32)
        else:
            return

        # P1.3: frame ĐẦU TIÊN sau on_speech_start chính là mép đầu của pre-roll.
        # Phải chốt `_speech_start_sample` Ở ĐÂY (trước khi ghi), vì on_speech_start
        # được VAD gọi TRƯỚC khi flush pre-roll — nếu tính ở đó sẽ lùi vào audio của
        # câu trước và làm bẩn ngữ cảnh đầu câu.
        with self._lock:
            if self._awaiting_pre_roll:
                self._speech_start_sample = self.audio_buffer.total_written
                self._awaiting_pre_roll = False

        self.audio_buffer.write(audio_float32)
        # P2.1: đồng hồ inactivity chỉ được nuôi bởi AUDIO ĐẾN (không phải bởi vòng poll),
        # nếu không BẬC 4 (TIMEOUT_FORCE) sẽ không bao giờ kích hoạt đúng nghĩa.
        self.commit_manager.record_activity()

    def on_speech_start(self) -> None:
        """Callback khi VAD phát hiện bắt đầu nói."""
        with self._lock:
            self._speech_active = True
            self._active_utterance_id = str(uuid.uuid4())[:8]
            # Điểm bắt đầu thật sẽ được chốt trong feed_audio (P1.3).
            self._awaiting_pre_roll = True
            self._speech_start_sample = -1
            self._last_preview_end_sample = -1
            self._last_preview_text = ""
            # K2: mốc để đo "thời gian tới preview ĐẦU TIÊN" của câu này.
            self._speech_started_at = time.perf_counter()
            self._first_preview_reported = False
            # FIX-10: reset bộ đếm lãng phí compute cho câu mới.
            self._preview_sec_processed = 0.0
            self._preview_sec_unique_peak = 0.0
        self.commit_manager.reset_stability()
        if self._seg is not None:
            self._seg.reset()
            self._seg_decision = None
        # K2: ĐÁNH THỨC generator ngay. Trước đây nhánh idle ngủ tới `poll_interval_ms`
        # (mặc định 300 ms) nên preview đầu tiên của câu bị trễ thêm tới ~300 ms chỉ vì
        # generator chưa biết là đã có tiếng nói (đo thật: K2 1,29 s -> 0,62 s).
        if bool(getattr(config.asr, "wake_on_speech_start", True)):
            self._wake_stream()
        self.commit_manager.record_activity()
        metrics_collector.increment_counter("asr.speech_start")

    def on_speech_end(self, reason: str = "VAD_SILENCE") -> None:
        """Callback khi VAD phát hiện kết thúc nói (BẬC 1: kích hoạt chốt câu)."""
        with self._lock:
            if not self._speech_active:
                return
            self._speech_active = False
            end_sample = self.audio_buffer.total_written
            # F-49: nếu chưa chốt được mốc bắt đầu (chưa có frame nào) thì đừng lấy 0 —
            # làm vậy commit sẽ bao trọn audio của câu trước. Dùng end_sample ⇒ mảnh rỗng,
            # bị bỏ qua an toàn.
            start_sample = self._speech_start_sample if self._speech_start_sample >= 0 else end_sample
            # K4: mốc "người nói dừng" để đo E2E tới phụ đề gốc chốt.
            self._speech_ended_at = time.perf_counter()

            self._enqueue_commit_locked(
                utterance_id=self._active_utterance_id,
                start_sample=start_sample,
                end_sample=end_sample,
                reason=reason,
            )

        # Đánh thức stream_tokens ngay lập tức (thread-safe, gọi từ sync VAD thread)
        self._wake_stream()

    def _wake_stream(self) -> None:
        """Đánh thức generator stream_tokens từ một thread đồng bộ."""
        loop = self._event_loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._commit_event.set)
            except RuntimeError:
                pass

    def _enqueue_commit_locked(
        self,
        utterance_id: str,
        start_sample: int,
        end_sample: int,
        reason: str,
    ) -> None:
        """Thêm commit request, GỘP thay vì vứt khi quá tải (P4.5).

        Caller phải giữ `self._lock`. Commit tham chiếu khoảng mẫu TUYỆT ĐỐI trên ring
        buffer, nên gộp `end_sample` không làm mất audio — chỉ làm câu dài hơn, và
        BẬC 2 (max_duration) sẽ chặn câu phình.
        """
        if len(self._pending_commits) >= _MAX_PENDING_COMMITS:
            oldest = self._pending_commits[0]
            oldest["end_sample"] = max(oldest["end_sample"], end_sample)
            oldest["reason"] = "MERGED_BACKLOG"
            self._commits_merged += 1
            metrics_collector.increment_counter("asr.commits_merged")
            logger.warning(
                f"Hàng đợi commit đầy ({_MAX_PENDING_COMMITS}) — đã GỘP câu cũ nhất "
                    f"thay vì vứt bỏ (tổng đã gộp: {self._commits_merged}). "
                        f"Audio KHÔNG bị mất.",
                extra={"module_tag": "ASR"},
            )
            metrics_collector.record_gauge("asr", "pending_commits", len(self._pending_commits))
            return

        self._pending_commits.append({
            "utterance_id": utterance_id,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "reason": reason,
            # F-44: gắn thế hệ stream để commit của đoạn CŨ bị bỏ sau khi tua.
            "generation": self._stream_generation,
        })
        metrics_collector.record_gauge("asr", "pending_commits", len(self._pending_commits))
        metrics_collector.record_gauge("asr", "pending_commits_peak", len(self._pending_commits))

    def set_language(self, language: str) -> None:
        """Cập nhật ngôn ngữ nhận dạng."""
        self.language = language or "auto"

    # ------------------------------------------------------------------ suy luận
    def _run_inference_sync(self, pcm_audio: np.ndarray) -> str:
        """Thực hiện suy luận ASR đồng bộ trên luồng worker C++."""
        if pcm_audio is None or len(pcm_audio) < int(16000 * self.min_transcribe_sec):
            return ""

        t_call = time.perf_counter()
        model = self._ensure_model_loaded()
        metrics_collector.record_metric("asr", "ensure_model_ms", (time.perf_counter() - t_call) * 1000.0)

        # P1.5: chặn trên theo trần audio của session để không bị truncate âm thầm.
        if config.asr.enforce_session_limits and self._max_audio_samples > 0:
            if len(pcm_audio) > self._max_audio_samples:
                logger.warning(
                    f"Audio {len(pcm_audio) / 16000.0:.1f}s vượt trần session "
                        f"{self._max_audio_samples / 16000.0:.1f}s — cắt bớt phần đầu.",
                    extra={"module_tag": "ASR"},
                )
                metrics_collector.increment_counter("asr.audio_truncated")
                pcm_audio = pcm_audio[-self._max_audio_samples:]

        # Chuẩn hóa âm lượng nếu bật config
        if config.asr.normalize_speech:
            t_norm = time.perf_counter()
            norm_res = self.normalizer.normalize(pcm_audio)
            audio_to_infer = norm_res.audio
            metrics_collector.record_metric("asr", "normalize_ms", (time.perf_counter() - t_norm) * 1000.0)
        else:
            audio_to_infer = pcm_audio

        info = self.registry.get_model_info(self.model_key) or {}
        family = info.get("family", "")
        lang = normalize_language_for_family(self.language, family)
        # P2.2: không xin timestamps => binding không materialize segments/words/tokens.
        want_timestamps = bool(getattr(config.asr, "request_timestamps", False))
        ts_arg = "auto" if want_timestamps else "none"

        with self.__class__._infer_lock:
            with self.__class__._shared_lock:
                session = self.__class__._shared_session
                supports_streaming = self.__class__._shared_supports_streaming
            if session is None:
                model = self._ensure_model_loaded()
                with self.__class__._shared_lock:
                    session = self.__class__._shared_session
                    supports_streaming = self.__class__._shared_supports_streaming

            raw_text = ""
            stream_success = False
            t_core = time.perf_counter()

            # 1. Nếu model hỗ trợ native streaming
            if supports_streaming:
                try:
                    stream_opts = build_family_options(family, info, model=model, slot="stream")
                    with session.stream(language=lang, family=stream_opts) as stream:
                        stream.feed(audio_to_infer)
                        stream.finalize()
                        raw_text = stream.text().full
                    stream_success = True
                except Exception as stream_err:
                    logger.debug(f"session.stream fallback to session.run: {stream_err}", extra={"module_tag": "ASR"})

            # 2. Non-streaming hoặc fallback: session.run
            if not stream_success:
                try:
                    run_opts = build_family_options(family, info, model=model, slot="run")
                    res = session.run(audio_to_infer, language=lang, family=run_opts, timestamps=ts_arg)
                    raw_text = getattr(res, "text", str(res))
                except Exception as run_err:
                    partial = getattr(run_err, "partial_result", None)
                    if partial is not None and hasattr(partial, "text"):
                        raw_text = str(partial.text)
                    else:
                        logger.error(f"Lỗi suy luận transcribe.cpp: {run_err}", exc_info=True, extra={"module_tag": "ASR"})
                        return ""

            # Thời gian suy luận THUẦN (không gồm nạp model / chuẩn hoá) để độ trễ
            # thật của model hiển thị tách bạch.
            metrics_collector.record_metric("asr", "infer_core_ms", (time.perf_counter() - t_core) * 1000.0)

            # P1.5: phát hiện truncation âm thầm
            try:
                if self._session_was_truncated(session):
                    metrics_collector.increment_counter("asr.output_truncated")
                    logger.warning(
                        "Kết quả bị TRUNCATED theo trần context của model — câu có thể thiếu chữ.",
                        extra={"module_tag": "ASR"},
                    )
            except Exception:
                pass

            cleaned = clean_transcript_text(raw_text)
            # Decoder "kẹt vòng": ca thật 2026-09-18 — tiếng cười nền làm model sinh 256 token
            # toàn `ha` (1,5 s GPU ở ASR, rồi 2,4 s GPU ở dịch, phụ đề một dòng khổng lồ).
            # `clean_transcript_text` đã gộp; ở đây ĐO để biết tần suất và để lại dấu vết.
            if raw_text and len(raw_text) >= 80 and len(cleaned) < len(raw_text) * 0.4:
                metrics_collector.increment_counter("asr.output_repetition_collapsed")
                logger.warning(
                    f"Đầu ra ASR bị LẶP VÒNG và đã gộp: {len(raw_text)} → {len(cleaned)} ký tự "
                        f"({(1 - len(cleaned) / len(raw_text)) * 100:.0f}% bị cắt).",
                    extra={"module_tag": "ASR"},
                )
            return cleaned

    # ------------------------------------------------------------------ commit
    def _pop_commit_request(self) -> Optional[Dict[str, Any]]:
        """Lấy commit request kế tiếp (VAD hoặc do BẬC 2/3/4 sinh ra)."""
        with self._lock:
            if self._pending_commits:
                return self._pending_commits.popleft()
        return None

    def _evaluate_tier234(self, preview_text: str, duration_sec: float) -> Optional[CommitReason]:
        """P2.1: đánh giá BẬC 2/3/4 trong lúc đang nói (VAD chưa báo im lặng).

        Thứ tự ưu tiên đúng như tài liệu CommitManager:
        MAX_DURATION > STABLE_PREFIX > TIMEOUT_FORCE.

        BẬC 3 có thêm hai sàn bảo vệ độ chính xác (C4):
        - `stability_min_duration_sec`: không cắt khi câu còn quá ngắn.
        - `stability_min_words`: không cắt khi preview chưa đủ từ.
        Nếu không có hai sàn này, một text ngắn không đổi trong ~1s đầu câu sẽ bị cắt
        ngay giữa câu => mất chữ.
        """
        if not bool(getattr(config.sentence, "enable_tier234", True)):
            return None

        cfg = self.commit_manager.cfg

        # BẬC 2 — chốt an toàn chống câu phình (bảo vệ chi phí preview, P2.3)
        if self.commit_manager.check_max_duration(duration_sec, len(preview_text)):
            return CommitReason.MAX_DURATION

        # BẬC 3 — cắt ở ranh giới từ (P3). Khi tầng SEG bật và được phép THAY THẾ thì bỏ
        # qua bậc này: SEG đã bao trùm trường hợp "text đứng yên" (`punct_stable`) và cắt
        # đúng ở dấu câu thay vì cắt mò giữa câu.
        seg_replaces = bool(
            self._seg is not None
            and getattr(self._seg_cfg, "replace_stable_prefix", True)
        )
        if self.commit_manager.cfg.split_on_stability and not seg_replaces:
            min_dur = float(getattr(cfg, "stability_min_duration_sec", 0.0) or 0.0)
            min_words = int(getattr(cfg, "stability_min_words", 0) or 0)
            if duration_sec >= min_dur and count_content_tokens(preview_text) >= min_words:
                if self.commit_manager.evaluate_preview_stability(preview_text):
                    return CommitReason.STABLE_PREFIX
            else:
                # Chưa đủ điều kiện: reset để bộ đếm ổn định không tích luỹ sai.
                self.commit_manager.reset_stability()

        # BẬC 4 — không có audio mới (lưới an toàn chống treo)
        if self.commit_manager.evaluate_inactivity_timeout(True):
            return CommitReason.TIMEOUT_FORCE

        return None

    def _evaluate_seg(self, preview_text: str, end_sample: int,
                      region_start: int = 0) -> Optional[CommitReason]:
        """Tầng SEG: chốt câu khi DẤU CÂU của ASR đã trọn câu (xem `backend/segmentation/`).

        Khác BẬC 3 (STABLE_PREFIX) ở chỗ: bằng chứng là *cấu trúc câu* (đã có chữ của câu
        kế tiếp sau dấu kết câu, hoặc dấu kết câu đứng yên đủ lâu), không phải "text không
        đổi". Nhờ vậy câu không bị cắt vụn ở `、` mà cũng không phình tới `max_duration_sec`.

        Khi `config.segmentation.debug_trace` bật: ghi `[SEG_TRACE]` cho TỪNG nhịp preview
        (kèm lý do còn chờ) để người dùng tự chọn mốc ngắt câu; nếu SEG đang TẮT thì ghi
        `[SEG_SHADOW]` cho biết SEG *sẽ* cắt ở đâu (so sánh A/B trong cùng một lần chạy).
        """
        if self._seg is None:
            self._trace_seg_shadow(preview_text, end_sample)
            return None
        if not preview_text:
            return None
        try:
            decisions = self._seg.observe(preview_text, end_sample)
        except Exception as exc:  # noqa: BLE001 — SEG lỗi không được làm chết pipeline
            logger.warning(f"SEG lỗi, bỏ qua vòng này: {exc}", extra={"module_tag": "ASR"})
            return None

        if not decisions:
            self._log_seg_trace("SEG_TRACE", preview_text, end_sample, region_start,
                                self._seg.completer.trace_state())
            return None

        self._seg_decision = decisions[0]
        if getattr(decisions[0], "stale_text", False):
            metrics_collector.increment_counter("seg.stale_text")
        metrics_collector.increment_counter(f"seg.reason.{decisions[0].reason}")
        self._log_seg_trace("SEG_CUT", preview_text, end_sample, region_start,
                            self._seg.completer.trace_state(), decision=decisions[0])
        return CommitReason.SEG_PUNCT

    @staticmethod
    def _elapsed_sec(end_sample: int, region_start: int) -> float:
        return max(0.0, (end_sample - region_start) / 16000.0)

    def _log_seg_trace(self, tag: str, preview_text: str, end_sample: int,
                       region_start: int, state: Dict[str, Any],
                       decision: Optional[Any] = None) -> None:
        """Ghi MỘT dòng trace cho mỗi nhịp ASR — đủ để chỉnh mốc ngắt câu bằng mắt."""
        if not self._seg_trace:
            return
        elapsed = self._elapsed_sec(end_sample, region_start)
        shown = preview_text if len(preview_text) <= 90 else "…" + preview_text[-90:]
        if decision is None:
            logger.info(
                f"[{tag}] [utt={self._active_utterance_id}] t=+{elapsed:.1f}s "
                f"{state.get('hold', '')} c={state.get('committed_len', 0)} | '{shown}'",
                extra={"module_tag": "SEG"},
            )
            return
        logger.info(
            f"[{tag}] [utt={self._active_utterance_id}] t=+{elapsed:.1f}s "
            f"cut={decision.reason} b={decision.end_index} tail='{state.get('tail', '')}' "
            f"rest='{preview_text[decision.end_index:][:40]}' | '{decision.text}'",
            extra={"module_tag": "SEG"},
        )

    def _trace_seg_shadow(self, preview_text: str, end_sample: int) -> None:
        """SEG đang TẮT: vẫn chạy máy trạng thái 'bóng' để log nó SẼ cắt ở đâu."""
        shadow = self._seg_shadow
        if shadow is None or not preview_text:
            return
        try:
            decisions = shadow.observe(preview_text, end_sample)
        except Exception:  # noqa: BLE001
            return
        if decisions:
            d = decisions[0]
            logger.info(
                f"[SEG_SHADOW] [utt={self._active_utterance_id}] SEG đang TẮT — nếu BẬT thì "
                f"đã cắt ({d.reason}) tại đây: '{d.text}'",
                extra={"module_tag": "SEG"},
            )

    def _seg_cut_sample(self, region_start: int, current_total: int, preview_text: str) -> int:
        """Mốc cắt cho commit SEG: mốc ms của timer (Whisper) nếu được, nếu không thì lùi chồng lấn.

        LƯU Ý: đã BỎ tính năng "tinh chỉnh mốc cắt bằng decode lại cửa sổ con" (2026-09-22)
        vì đo thật tốn ~1.25 s GPU cho mỗi câu được cắt — quá đắt so với lợi ích.
        """
        return self._seg_base_cut_sample(region_start, current_total, preview_text)

    def _seg_base_cut_sample(self, region_start: int, current_total: int,
                             preview_text: str) -> int:
        """Mốc cắt GỐC (trước tinh chỉnh): mốc ms của timer (Whisper) nếu được,
        nếu không thì lùi `fallback_overlap_ms` so với cuối vùng nói.

        """
        fallback = getattr(self._seg_decision, "cut_sample", None)
        if fallback is None:
            fallback = max(region_start, current_total)
        cfg = self._seg_cfg
        if not bool(getattr(cfg, "use_whisper_timer", False)) or self._seg_decision is None:
            return int(fallback)
        duration_sec = (current_total - region_start) / 16000.0
        if duration_sec <= 0 or duration_sec > float(getattr(cfg, "timer_max_audio_sec", 30.0)):
            return int(fallback)
        try:
            from backend.asr.timer import get_seg_timer

            timer = get_seg_timer()
        except Exception:  # noqa: BLE001
            return int(fallback)
        if timer is None or not timer.available():
            metrics_collector.increment_counter("seg.timer_unavailable")
            return int(fallback)
        try:
            audio = self.audio_buffer.get_slice(region_start, current_total)
            res = timer.locate_boundary(
                audio,
                int(self._seg_decision.end_index),
                preview_text,
                min_confidence=float(getattr(cfg, "timer_min_confidence", 0.35)),
                language=self.language,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Timer SEG lỗi: {exc}", extra={"module_tag": "ASR"})
            res = None
        if res is None:
            metrics_collector.increment_counter("seg.timer_miss")
            return int(fallback)
        cut = region_start + res.cut_sample
        # Chốt an toàn: mốc cắt phải nằm TRONG vùng nói, không được vượt hiện tại.
        margin = int(0.1 * 16000)
        if cut <= region_start + margin or cut >= current_total:
            metrics_collector.increment_counter("seg.timer_out_of_range")
            return int(fallback)
        self._seg_timer_ms_total += res.elapsed_ms
        metrics_collector.record_metric("seg", "timer_ms", res.elapsed_ms)
        metrics_collector.record_metric("seg", "timer_confidence", res.confidence)
        metrics_collector.increment_counter("seg.timer_used")
        logger.info(
            f"[utt={self._active_utterance_id}] SEG timer: cắt tại "
            f"{(cut - region_start) / 16000.0:.2f}s trong vùng (tin cậy {res.confidence:.2f}, "
            f"{res.elapsed_ms:.0f}ms) — '{self._seg_decision.text[:40]}'",
            extra={"module_tag": "ASR"},
        )
        return int(cut)

    def has_pending_work(self) -> bool:
        """True nếu pipeline còn việc: đang nói, còn commit chờ, hoặc inference đang chạy.

        Dùng cho harness/benchmark để biết khi nào thực sự xử lý xong. Lưu ý:
        `not engine._pending_commits` một mình là SAI vì commit bị pop ra ngay khi bắt
        đầu inference (đã từng khiến benchmark cắt ngang inference đang chạy).
        """
        with self._inflight_lock:
            inflight = self._inflight
        with self._lock:
            pending = len(self._pending_commits)
            speaking = self._speech_active
        return bool(inflight) or bool(pending) or bool(speaking)

    def _begin_infer(self) -> None:
        with self._inflight_lock:
            self._inflight += 1

    def _end_infer(self) -> None:
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)

    def _inflight_count(self) -> int:
        with self._inflight_lock:
            return self._inflight

    async def _infer_with_watchdog(self, audio_slice: np.ndarray, *, is_commit: bool = False) -> str:
        """Chạy inference trong executor kèm 2 ĐỒNG HỒ canh chừng (F-39).

        A2-1: `is_commit=True` mở **cửa sổ dành riêng GPU** (`GpuArbiter`) trong suốt lượt
        inference, để job ưu tiên thấp (dịch/TTS) không KHỞI ĐỘNG chen vào và đẩy commit ra
        sau. Chỉ có tác dụng khi `config.gpu.scheduler_enabled`; mặc định tắt nên không đổi
        hành vi.

        Vì sao cần: `asyncio` huỷ task KHÔNG dừng được lời gọi native — thread C++ vẫn
        chạy và vẫn giữ `_infer_lock`. Đo thật cho thấy một `session.run()` có lúc chạy
        hàng chục giây (mỗi 20 s dump stack đều thấy nó trong `transcribe_cpp.run`), và
        trong lúc đó `_EXECUTOR` (hàng đợi KHÔNG giới hạn) bị xếp thêm việc ⇒ RAM tăng
        ~66-85 MB/s cho tới khi hết bộ nhớ.

        Hai hàng rào:
        1. `inference_watchdog_sec` — quá ngân sách thì `cancel_inference()` để native
           `session.cancel()` trả về sớm (kèm partial result).
        2. `rss_runaway_delta_mb` — RSS tăng quá nhanh NGAY TRONG lúc inference chạy
           (đo được tới +73 GB private!) thì huỷ ngay và bỏ vòng đó. Đây là dấu hiệu
           phình bộ nhớ trong native/Vulkan mà Python không kiểm soát được.

        `asyncio.shield` giữ cho future không bị huỷ để còn lấy kết quả sau khi native thoát.
        """
        loop = asyncio.get_running_loop()
        self._begin_infer()
        if is_commit:
            gpu_arbiter.commit_begin()
        try:
            fut = loop.run_in_executor(_EXECUTOR, self._run_inference_sync, audio_slice)
            budget = float(getattr(config.asr, "inference_watchdog_sec", 8.0) or 0.0)
            runaway_mb = float(getattr(config.asr, "rss_runaway_delta_mb", 1024.0) or 0.0)
            watch_rss = runaway_mb > 0
            if budget <= 0 and not watch_rss:
                return await fut

            grace_s = max(0.0, float(getattr(config.asr, "inference_watchdog_grace_sec", 2.0) or 0.0))
            rss_start = self._process_rss_mb() if watch_rss else 0.0
            t_start = time.monotonic()
            # Sau khi đã yêu cầu native huỷ, chỉ chờ thêm 2 s rồi bỏ vòng này.
            hard_deadline: Optional[float] = None
            fired_reason: Optional[str] = None

            while True:
                now = time.monotonic()
                if hard_deadline is not None and now >= hard_deadline:
                    metrics_collector.increment_counter("asr.inference_watchdog_hard_timeout")
                    logger.error(
                        f"Native không thoát sau khi huỷ ({fired_reason}) — bỏ kết quả "
                            f"vòng này để không xếp thêm việc vào executor.",
                        extra={"module_tag": "ASR"},
                    )
                    return ""
                slice_s = 0.25  # chế độ chỉ canh RSS: kiểm tra 4 lần/giây
                if hard_deadline is not None:
                    # Đã yêu cầu huỷ: kiểm tra sát hạn ân hạn để thoát NGAY khi hết hạn.
                    slice_s = max(0.02, min(0.1, hard_deadline - now))
                elif budget > 0:
                    slice_s = min(1.0, max(0.02, budget - (now - t_start)))
                try:
                    return await asyncio.wait_for(asyncio.shield(fut), timeout=slice_s)
                except asyncio.TimeoutError:
                    pass

                if hard_deadline is not None:
                    continue  # đã huỷ, chờ cho hết ân hạn
                now = time.monotonic()
                if watch_rss:
                    growth = self._process_rss_mb() - rss_start
                    if growth >= runaway_mb:
                        metrics_collector.increment_counter("asr.rss_runaway_detected")
                        metrics_collector.record_gauge("asr", "rss_runaway_mb", growth)
                        logger.error(
                            f"RSS tăng {growth:.0f} MB trong lúc inference đang chạy "
                                f"(audio {len(audio_slice) / 16000.0:.1f}s) — nghi phình bộ nhớ "
                                    f"trong native/Vulkan. Huỷ inference và bỏ vòng này.",
                            extra={"module_tag": "ASR"},
                        )
                        await self.cancel_inference()
                        fired_reason = f"rss +{growth:.0f}MB"
                        hard_deadline = time.monotonic() + grace_s
                        continue
                if budget > 0 and (now - t_start) >= budget:
                    metrics_collector.increment_counter("asr.inference_watchdog_fired")
                    logger.warning(
                        f"Inference vượt {budget:.1f}s — yêu cầu native HUỶ để giải phóng "
                            f"`_infer_lock` (audio {len(audio_slice) / 16000.0:.1f}s).",
                        extra={"module_tag": "ASR"},
                    )
                    await self.cancel_inference()
                    fired_reason = f"quá {budget:.1f}s"
                    hard_deadline = time.monotonic() + grace_s
                    continue
        finally:
            self._end_infer()
            if is_commit:
                gpu_arbiter.commit_end()

    def _publish_recompute_ratio(self) -> None:
        """FIX-10: công bố gauge lãng phí compute của preview (giá trị SỐNG, cập nhật mỗi vòng).

        `ratio` là số lần trung bình mỗi giây audio bị đưa qua model trong câu hiện tại.
        Đây là chỉ số DUY NHẤT trả lời được câu hỏi "có đáng tối ưu preview không" trước khi
        bỏ công sức (và rủi ro hồi quy WER) vào incremental/adaptive preview.
        """
        unique = self._preview_sec_unique_peak
        if unique <= 0:
            return
        metrics_collector.record_gauge(
            "asr", "audio_seconds_processed_preview", self._preview_sec_processed
        )
        metrics_collector.record_gauge("asr", "audio_seconds_unique_preview", unique)
        metrics_collector.record_gauge(
            "asr", "preview_recompute_ratio", self._preview_sec_processed / unique
        )

    def _finalize_recompute_metrics(self) -> None:
        """FIX-10: chốt số liệu lãng phí preview cho câu VỪA chốt rồi reset bộ đếm.

        Ghi qua `record_metric` để `/api/metrics` trả được p50/p95 — nhờ vậy benchmark
        (`test_08_streaming_latency.py`, `test_09_wer_ab.py`) đọc được phân bố chứ không chỉ
        giá trị cuối cùng.
        """
        processed = self._preview_sec_processed
        unique = self._preview_sec_unique_peak
        if processed > 0:
            metrics_collector.record_metric("asr", "audio_seconds_processed_preview", processed)
        if unique > 0:
            metrics_collector.record_metric("asr", "audio_seconds_unique_preview", unique)
        if processed > 0 and unique > 0:
            ratio = processed / unique
            metrics_collector.record_metric("asr", "preview_recompute_ratio", ratio)
            metrics_collector.record_gauge("asr", "preview_recompute_ratio", ratio)
            logger.debug(
                f"Preview lãng phí: xử lý {processed:.2f}s audio cho câu {unique:.2f}s "
                    f"(ratio {ratio:.2f}x).",
                extra={"module_tag": "ASR"},
            )
        self._preview_sec_processed = 0.0
        self._preview_sec_unique_peak = 0.0

    def _preload_model(self, force_warm: bool = False) -> None:
        """Nạp + pre-warm model trước khi dùng (idempotent).

        Ba lý do (đo thật, xem `report/audit/07_gpu_usage_probe.json`):
        1. Không nạp trước thì inference đầu bao gồm cả thời gian nạp model (~1,2 s với
           qwen3-asr-0.6b) và làm hỏng mọi metric độ trễ.
        2. Chỉ nạp mà không chạy thử thì backend còn phải build graph + compile kernel
           Vulkan + cấp workspace: đo được **~7,0 s** cho đoạn 0,5 s.
        3. QUAN TRỌNG NHẤT: chi phí (2) còn lặp lại theo **ĐỘ DÀI** audio. Đo thực tế:
           sau khi warm 0,5 s, một inference 6 s mất **4218 ms**, nhưng lần 6 s kế tiếp chỉ
           **57 ms** (nhanh hơn 74×). Tức là lần đầu chạm một độ dài LỚN HƠN phải trả chi
           phí cấp workspace/graph mới. Vì vậy phải warm tới `max_duration_sec` — đúng độ
           dài câu lớn nhất runtime sẽ dùng. Nếu không, phụ đề ĐẦU TIÊN của người dùng bị
           "đơ" vài giây (đây chính là nguồn spike từng bị quy cho driver).

        Nhờ (3), sau khi warm đủ thì `asr.preview_ms` p95 giảm từ ~1450 ms xuống ~131 ms và
        `asr.preview_skipped` từ 48 xuống 1 (đo trên 20 s audio nói liên tục).
        """
        cls = self.__class__
        already_loaded = cls._shared_model is not None and cls._shared_model_key == self.model_key

        if already_loaded and cls._shared_warmed_long and not force_warm:
            return

        t0 = time.perf_counter()
        try:
            if not already_loaded:
                self._ensure_model_loaded()
                metrics_collector.record_metric("asr", "model_load_ms", (time.perf_counter() - t0) * 1000.0)

            short_sec = max(0.5, float(self.min_transcribe_sec))
            long_sec = float(getattr(self.commit_manager.cfg, "max_duration_sec", 6.0) or 6.0)
            long_sec = max(short_sec, long_sec)

            t_warm = time.perf_counter()
            self._begin_infer()
            try:
                self._run_inference_sync(np.zeros(int(16000 * short_sec), dtype=np.float32))
                short_ms = (time.perf_counter() - t_warm) * 1000.0

                # Warm độ dài LỚN NHẤT mà runtime sẽ dùng (điểm mấu chốt).
                t_long = time.perf_counter()
                self._run_inference_sync(np.zeros(int(16000 * long_sec), dtype=np.float32))
                long_ms = (time.perf_counter() - t_long) * 1000.0
            finally:
                self._end_infer()

            cls._shared_warmed_long = True
            metrics_collector.record_metric("asr", "model_warmup_ms", short_ms + long_ms)
            metrics_collector.record_metric("asr", "warmup_short_ms", short_ms)
            metrics_collector.record_metric("asr", "warmup_long_ms", long_ms)
            logger.info(
                f"Pre-warm ({short_sec:.1f}s={short_ms:.0f}ms + {long_sec:.1f}s={long_ms:.0f}ms)",
                extra={"module_tag": "ASR"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Nạp trước/ pre-warm model thất bại (sẽ thử lại trong inference): {exc}",
                           extra={"module_tag": "ASR"})

    def _preview_window_start(self, seg_start: int, current_total: int) -> int:
        """Điểm bắt đầu cửa sổ preview (P2.3).

        CHỈ preview bị cửa sổ hoá. Commit luôn dùng `[seg_start, end]` đầy đủ.
        Trả về `seg_start` khi tắt cửa sổ (preview_window_sec = 0).
        """
        win = self.preview_window_sec
        if win <= 0:
            return max(0, seg_start)
        return max(max(0, seg_start), current_total - int(16000 * win))

    def _effective_poll_interval(self, base: float) -> float:
        """P2.4b: nhịp preview thích ứng theo trung vị thời gian inference gần đây.

        Lý do: đo thực tế cho thấy `asr.preview_ms` có p50 rất tốt (~87 ms) nhưng thỉnh
        thoảng spike tới ~2.5 s (driver/GPU contention). Với nhịp cố định, các spike này
        làm bỏ nhịp liên tục và vẫn tranh chấp GPU. Giãn nhịp ra khi chậm giúp phụ đề
        cập nhật **đều và đáng tin hơn** thay vì giật; khi nhanh trở lại thì thu hẹp dần.
        """
        self._eff_poll_interval = max(base, self._eff_poll_interval)
        if not bool(getattr(config.asr, "preview_adaptive_backoff", True)):
            return base
        if not self._preview_durations:
            return self._eff_poll_interval

        vals = sorted(self._preview_durations)
        median = vals[len(vals) // 2]
        slow_ms = float(getattr(config.asr, "preview_slow_ms", 350.0))
        fast_ms = float(getattr(config.asr, "preview_fast_ms", 120.0))
        max_interval = float(getattr(config.asr, "preview_max_interval_ms", 1000)) / 1000.0

        if median >= slow_ms:
            self._eff_poll_interval = min(max_interval, self._eff_poll_interval * 1.5)
        elif median <= fast_ms:
            self._eff_poll_interval = max(base, self._eff_poll_interval / 1.25)

        metrics_collector.record_gauge("asr", "effective_poll_ms", self._eff_poll_interval * 1000.0)
        metrics_collector.record_gauge("asr", "preview_median_ms", median)
        return self._eff_poll_interval

    async def _emit_commit(self, commit_req: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """Chạy inference cho một commit và phát ra message `utterance_update` final."""
        self._commit_event.clear()
        utt_id = commit_req["utterance_id"]
        start_s = commit_req["start_sample"]
        end_s = commit_req["end_sample"]
        reason = commit_req["reason"]

        if end_s <= start_s:
            # F-48: không phát gì thì PHẢI xoá cờ đánh thức, nếu không nhánh idle sẽ
            # `wait_for(event.wait())` trả về tức thì ⇒ quay nóng (xem chú thích ở nhánh idle).
            self._commit_event.clear()
            return

        # F-44: commit này thuộc đoạn TRƯỚC khi tua ⇒ bỏ (tránh phụ đề lệch vị trí).
        req_gen = commit_req.get("generation")
        if req_gen is not None and req_gen != self._stream_generation:
            metrics_collector.increment_counter("asr.commit_dropped_stale")
            logger.info(
                f"[utt={utt_id}] Bỏ commit thuộc đoạn cũ (generation {req_gen} != "
                f"{self._stream_generation}) — video đã tua.",
                extra={"module_tag": "ASR_COMMIT"},
            )
            self._commit_event.clear()  # F-48: không phát gì ⇒ xoá cờ đánh thức
            return

        # CHỐNG "BLOB" KHI TUA VIDEO (F-44): câu không bao giờ được dài hơn
        # `max_duration_sec` một cách vô lý. Khi người dùng tua, trình duyệt có thể đẩy
        # một lượng audio cũ sang và VAD không kịp đóng câu ⇒ mảnh commit phình tới cả
        # phút, chứa lại nội dung đã phát (đã gặp: mảnh ~54 s, infer 1341 ms, phụ đề lặp).
        # Đây là chốt an toàn CUỐI: chỉ lấy phần ĐUÔI (audio mới nhất) nếu vượt trần.
        max_slice_sec = float(getattr(config.sentence, "max_duration_sec", 6.0) or 6.0) * 1.5
        max_slice_samples = int(max_slice_sec * 16000)
        if max_slice_samples > 0 and (end_s - start_s) > max_slice_samples:
            clamped_start = end_s - max_slice_samples
            metrics_collector.increment_counter("asr.commit_slice_clamped")
            logger.warning(
                f"[utt={utt_id}] Mảnh commit dài {(end_s - start_s) / 16000.0:.1f}s "
                f"(> trần {max_slice_sec:.1f}s) — chỉ lấy {max_slice_sec:.1f}s cuối "
                f"(nghi tua video / audio cũ lẫn vào).",
                extra={"module_tag": "ASR_COMMIT"},
            )
            start_s = clamped_start

        audio_slice = self.audio_buffer.get_slice(start_s, end_s)
        final_text = ""
        infer_ms = 0.0

        # P2.5: tái sử dụng kết quả preview cuối nếu cửa sổ audio gần như không đổi.
        reuse = bool(getattr(config.asr, "preview_reuse_for_commit", False))
        max_delta = int(float(getattr(config.asr, "preview_reuse_max_delta_sec", 0.4)) * 16000)
        if (
            reuse
            and self._last_preview_text
            and self._last_preview_end_sample > 0
            and abs(end_s - self._last_preview_end_sample) <= max_delta
        ):
            final_text = self._last_preview_text
            metrics_collector.increment_counter("asr.commit_reused_preview")
        else:
            t0 = time.perf_counter()
            final_text = await self._infer_with_watchdog(audio_slice, is_commit=True)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            metrics_collector.record_metric("asr", "commit_ms", infer_ms)
            metrics_collector.record_metric("asr", "commit_audio_sec", len(audio_slice) / 16000.0)

        # Ranh giới cắt giữa câu: nếu mảnh quá ngắn thì GỘP vào câu kế tiếp thay vì
        # để tầng trên lọc bỏ (tránh mất chữ — P2.1/P2.7).
        # P2.7: nếu câu này bắt đầu bằng audio chồng lấn của câu trước, phần từ lặp ở
        # ĐẦU được TRIM (không drop cả câu).
        if final_text and reason != "VAD_SILENCE":
            trimmed = self.commit_manager.deduplicator.trim_boundary_overlap(final_text)
            if trimmed != final_text:
                metrics_collector.increment_counter("asr.boundary_trimmed")
                logger.info(
                    f"[utt={utt_id}] Trim chồng lấn ranh giới: '{final_text}'-> '{trimmed}'",
                    extra={"module_tag": "ASR_COMMIT"},
                )
                final_text = trimmed

        # FIX-12 (bản 2 — sửa lỗi LẶP do bản 1): mốc cắt có thể hơi lệch nên văn bản chạy lại
        # trên mảnh đã cắt thường chứa NHIỀU câu (log thật: `'Okay. Well, … Aquaman statue.
        # Aquaman. This isn't a gag gift, Stewart.'`). Bản 1 luôn lấy CÂU ĐẦU ⇒ phần lớn
        # trường hợp câu đầu chỉ là mảnh ngắn (`'Okay.'`) ⇒ bị coi là "quá ngắn" ⇒ gộp lại ⇒
        # VÙNG KHÔNG TIẾN ⇒ lặp tới `MAX_DURATION` (câu siêu dài).
        # Bản 2: chọn câu KHỚP NHẤT với câu mà SEG đã chốt (`self._seg_decision.text`).
        if final_text and reason == CommitReason.SEG_PUNCT.value and self._seg_decision is not None:
            parts = split_complete_sentences(final_text)
            if len(parts) > 1:
                chosen = _best_matching_sentence(parts, getattr(self._seg_decision, "text", ""))
                if chosen and chosen != final_text.strip():
                    logger.info(
                        f"[utt={utt_id}] SEG: chọn lại câu khớp với câu đã chốt: "
                        f"{len(parts)} câu -> '{chosen}'",
                        extra={"module_tag": "ASR_COMMIT"},
                    )
                    metrics_collector.increment_counter("asr.commit_tail_trimmed")
                    final_text = chosen

        min_words = int(config.sentence.min_words_to_commit)
        carry_over = bool(getattr(config.sentence, "carry_over_short_fragment", True))
        if (
            carry_over
            and reason != "VAD_SILENCE"
            and final_text
            and count_content_tokens(final_text) < min_words
        ):
            # FIX-11 (đo từ log phim thật 2026-09-22): mảnh cắt ra CHỈ CÒN DẤU CÂU (ví dụ
            # `'.'` sau khi trim chồng lấn) mà vẫn "gộp vào câu kế tiếp" thì vùng audio
            # KHÔNG tiến ⇒ nhịp sau lại đọc đúng mảnh đó, lại gọi timer (~105 ms) rồi lại gộp:
            # log thật cho thấy `'Aquaman.' -> '.'` lặp 4 lần liên tiếp.
            # Mảnh không còn ký tự nội dung thì BỎ HẲN và đẩy mốc bắt đầu vùng lên `end_s`.
            if reason == CommitReason.SEG_PUNCT.value and count_content_tokens(final_text) == 0:
                with self._lock:
                    self._speech_start_sample = end_s
                    self._awaiting_pre_roll = False
                metrics_collector.increment_counter("asr.commit_content_free_dropped")
                logger.info(
                    f"[utt={utt_id}] Mảnh cắt chỉ còn dấu câu ({final_text!r}) — bỏ hẳn, "
                    f"đẩy vùng đọc lên {end_s / 16000.0:.2f}s để không lặp lại.",
                    extra={"module_tag": "ASR_COMMIT"},
                )
                return
            # FIX-12b: CHỐT AN TOÀN CHỐNG LẶP. Nếu cùng một mảnh ngắn bị gộp HAI lần liên
            # tiếp thì vùng audio rõ ràng không tiến (đã gặp: `'Okay.'` lặp 6 lần tới
            # MAX_DURATION, câu phình 15 s). Lần thứ hai thì BỎ mảnh và đẩy vùng lên `end_s`.
            if reason == CommitReason.SEG_PUNCT.value and final_text == self._last_carried_text:
                with self._lock:
                    self._speech_start_sample = end_s
                    self._awaiting_pre_roll = False
                metrics_collector.increment_counter("asr.commit_carry_loop_broken")
                logger.warning(
                    f"[utt={utt_id}] Mảnh '{final_text}' bị gộp LẶP LẠI — cắt vòng lặp, đẩy "
                    f"vùng đọc lên {end_s / 16000.0:.2f}s.",
                    extra={"module_tag": "ASR_COMMIT"},
                )
                return
            if reason == CommitReason.SEG_PUNCT.value:
                self._last_carried_text = final_text
            with self._lock:
                self._speech_start_sample = start_s
                self._awaiting_pre_roll = False
            metrics_collector.increment_counter("asr.commit_carried_over")
            logger.info(
                f"[utt={utt_id}] Mảnh cắt quá ngắn ({count_content_tokens(final_text)} < "
                    f"{min_words} từ, reason={reason}) — gộp vào câu kế tiếp: '{final_text}'",
                extra={"module_tag": "ASR_COMMIT"},
            )
            return

        # P2.7: ranh giới cắt giữa câu -> câu kế tiếp lùi lại một đoạn overlap.
        overlap_samples = 0
        if reason != "VAD_SILENCE":
            overlap_samples = int(float(getattr(config.sentence, "boundary_overlap_ms", 0)) / 1000.0 * 16000)
        with self._lock:
            self._speech_start_sample = max(0, end_s - overlap_samples) if self._speech_active else end_s
            self._awaiting_pre_roll = False
            self._last_preview_end_sample = -1
            self._last_preview_text = ""

        self._last_committed_sample = end_s
        self.commit_manager.record_commit(final_text)
        self._last_carried_text = ""
        metrics_collector.increment_counter(f"asr.commit_reason.{reason}")
        # FIX-10: chốt số liệu lãng phí preview của câu này (đo trước khi tối ưu).
        self._finalize_recompute_metrics()

        if final_text:
            logger.info(
                f"[utt={utt_id}] [{reason}] (infer={infer_ms:.1f}ms): '{final_text}'",
                extra={"module_tag": "ASR_COMMIT"},
            )
            # K4: E2E từ lúc VAD báo ngừng nói tới khi phát phụ đề gốc đã chốt.
            if self._speech_ended_at > 0:
                metrics_collector.record_metric(
                    "asr", "e2e_commit_ms",
                    (time.perf_counter() - self._speech_ended_at) * 1000.0,
                )
                self._speech_ended_at = 0.0
            yield {
                "type": "utterance_update",
                "utterance_id": utt_id,
                "text": final_text,
                "is_final": True,
                "language": self.language,
                "inference_ms": infer_ms,
                "commit_reason": reason,
            }

    # ------------------------------------------------------------------ generator
    async def stream_tokens(self) -> AsyncIterator[Dict[str, Any]]:
        """Async generator liên tục thám thính preview và xuất bản kết quả ASR.

        Nhịp: `poll_interval_ms` cố định (P2.4). Chi phí mỗi vòng bị chặn trên bởi
        `preview_window_sec` (P2.3).
        """
        poll_interval = max(0.05, self.poll_interval_ms / 1000.0)
        fixed_rate = bool(getattr(config.asr, "preview_fixed_rate", True))
        self._event_loop = asyncio.get_running_loop()
        # P2.4b: mỗi lần generator chạy lại thì khởi động từ nhịp ĐÃ CẤU HÌNH rồi mới
        # tự điều chỉnh. Không reset thì giá trị giãn ra từ lần chạy trước (hoặc giá trị
        # khởi tạo trong __init__) sẽ đè lên cấu hình mới.
        self._eff_poll_interval = poll_interval
        self._preview_durations.clear()

        window_sec = self.preview_window_sec
        if window_sec > 0:
            logger.info(
                f"Preview dùng cửa sổ {window_sec:.1f}s (= max_duration "
                    f"{self.commit_manager.cfg.max_duration_sec:.1f}s) — chi phí preview bị chặn trên.",
                extra={"module_tag": "ASR"},
            )

        next_deadline = time.perf_counter()

        # Nạp trước model để lần inference ĐẦU TIÊN không bao gồm thời gian nạp model
        # (đo được ~8s với qwen3-asr-0.6b, làm hỏng mọi metric độ trễ).
        preload_loop = asyncio.get_running_loop()
        await preload_loop.run_in_executor(_EXECUTOR, self._preload_model)

        while self._is_running:
            # 1) Commit đang chờ (VAD silence hoặc do BẬC 2/3/4 sinh ra ở vòng trước)
            commit_req = self._pop_commit_request()
            if commit_req:
                metrics_collector.record_gauge("asr", "pending_commits", len(self._pending_commits))
                emitted = 0
                async for msg in self._emit_commit(commit_req):
                    emitted += 1
                    yield msg
                # F-47 (rào chắn cuối): commit bị BỎ mà không phát gì cả. Nếu lặp lại nhiều
                # lần liên tiếp, vòng lặp có thể quay nóng mà KHÔNG nhường event loop (đã
                # từng treo cứng backend). Ép nhường một nhịp + ghi log để không bao giờ treo.
                if emitted == 0:
                    self._empty_commit_streak += 1
                    if self._empty_commit_streak >= 10:
                        metrics_collector.increment_counter("asr.commit_empty_streak_guard")
                        logger.warning(
                            f"{self._empty_commit_streak} commit liên tiếp không phát gì — "
                            f"nhường nhịp để tránh quay nóng vòng lặp stream.",
                            extra={"module_tag": "ASR"},
                        )
                        self._empty_commit_streak = 0
                        await asyncio.sleep(max(0.02, poll_interval))
                else:
                    self._empty_commit_streak = 0
                next_deadline = time.perf_counter()
                continue

            # 2) Đang nói -> thám thính preview
            if self._speech_active:
                # ƯU TIÊN COMMIT (C4/C5): commit mới là thứ tạo ra phụ đề chốt + bản dịch;
                # preview chỉ là best-effort. Nếu có commit đang chờ, nhường vòng này thay
                # vì xếp thêm một inference nữa trước nó (đo được spike lẻ tới ~2.5s).
                with self._lock:
                    has_commit = bool(self._pending_commits)
                if has_commit:
                    metrics_collector.increment_counter("asr.preview_yielded_to_commit")
                    await asyncio.sleep(0)
                    continue

                with self._lock:
                    current_total = self.audio_buffer.total_written
                    seg_start_raw = self._speech_start_sample
                    speech_active = self._speech_active
                # F-49: `_speech_start_sample < 0` = VAD đã báo bắt đầu nói nhưng CHƯA có
                # frame audio nào để chốt mốc (P1.3). Lúc này TUYỆT ĐỐI không được đánh giá
                # preview/cắt câu: nếu lấy `seg_start = 0` thì mốc câu sẽ trỏ về đầu buffer
                # (audio của câu TRƯỚC) ⇒ span phình 20-30 s, BẬC 2 cắt sai và phần đầu bị
                # cắt mất chữ (đã thấy trong log: 9,6s → 12,7s → 22,9s → 31,8s).
                seg_start = seg_start_raw if seg_start_raw >= 0 else current_total
                ready = seg_start_raw >= 0

                if ready and speech_active and (current_total - seg_start) >= int(16000 * self.min_transcribe_sec):
                    # P2.3: CHỈ preview bị cửa sổ hoá. Cửa sổ >= max_duration_sec nên
                    # luôn bao trùm trọn câu hiện tại => giữ nguyên ngữ cảnh.
                    win_start = self._preview_window_start(seg_start, current_total)

                    # F-39: CHẶN XẾP HÀNG VÔ HẠN. `_EXECUTOR` có hàng đợi không giới hạn, nên
                    # nếu một inference native đang chạy lâu mà ta vẫn nộp preview mỗi vòng thì
                    # mỗi mục xếp hàng giữ nguyên một `audio_slice` (tới ~0,4 MB) ⇒ RAM nổ.
                    # Chỉ cho tối đa `max_inflight_infer` inference cùng lúc; vòng nào vượt thì
                    # BỎ vòng đó (preview là best-effort) và ghi counter.
                    max_inflight = int(getattr(config.asr, "max_inflight_infer", 1) or 0)
                    defer_preview = max_inflight > 0 and self._inflight_count() >= max_inflight

                    if defer_preview:
                        metrics_collector.increment_counter("asr.preview_deferred_inflight")
                        audio_slice = None
                        preview_text = ""
                        infer_ms = 0.0
                    else:
                        t_slice = time.perf_counter()
                        audio_slice = self.audio_buffer.get_slice(win_start, current_total)
                        metrics_collector.record_metric(
                            "asr", "slice_ms", (time.perf_counter() - t_slice) * 1000.0
                        )
                        t0 = time.perf_counter()
                        gen_before = self._stream_generation
                        preview_text = await self._infer_with_watchdog(audio_slice)
                        infer_ms = (time.perf_counter() - t0) * 1000.0
                        # F-44: video bị TUA trong lúc inference đang chạy ⇒ kết quả này
                        # thuộc đoạn CŨ, phải bỏ (không hiện phụ đề lệch vị trí).
                        if gen_before != self._stream_generation:
                            metrics_collector.increment_counter("asr.preview_dropped_stale")
                            preview_text = ""
                            infer_ms = 0.0
                        metrics_collector.record_metric("asr", "preview_ms", infer_ms)
                        metrics_collector.record_metric("asr", "preview_audio_sec", len(audio_slice) / 16000.0)
                        # FIX-10: tích luỹ lượng audio ĐÃ đưa qua model so với lượng audio THẬT.
                        self._preview_sec_processed += len(audio_slice) / 16000.0
                        self._preview_sec_unique_peak = max(
                            self._preview_sec_unique_peak, (current_total - seg_start) / 16000.0
                        )
                        self._publish_recompute_ratio()
                        self._preview_durations.append(infer_ms)

                        if preview_text and self._speech_active:
                            self._last_preview_end_sample = current_total
                            self._last_preview_text = preview_text
                            # K2: đo thời gian từ lúc VAD báo bắt đầu nói tới preview đầu tiên.
                            if not self._first_preview_reported and self._speech_started_at > 0:
                                self._first_preview_reported = True
                                metrics_collector.record_metric(
                                    "asr", "first_preview_ms",
                                    (time.perf_counter() - self._speech_started_at) * 1000.0,
                                )
                            yield {
                                "type": "utterance_update",
                                "utterance_id": self._active_utterance_id,
                                "text": preview_text,
                                "is_final": False,
                                "stable_text": preview_text,
                                "unstable_text": "",
                                "language": self.language,
                                "inference_ms": infer_ms,
                            }

                        # 2b) Nội dung preview đã phát xong; việc CẮT CÂU được đánh giá ở
                        # khối ngay dưới, chạy mọi vòng (kể cả vòng không chạy preview).

                    # 2b) P2.1/F-42: ĐÁNH GIÁ CẮT CÂU PHẢI CHẠY MỌI VÒNG, kể cả khi vòng này
                    # KHÔNG chạy preview (preview bị hoãn vì đang có inference khác, hoặc
                    # audio chưa đủ ngưỡng). Trước đây khối này nằm trong nhánh preview nên
                    # khi ASR chậm (1,3 s/lần) câu cứ phình mãi: đã gặp một câu MAX_DURATION
                    # dài ~45 s (infer 1294 ms) dù `max_duration_sec` chỉ 6 s.
                    if self._speech_active:
                        duration_sec = (current_total - seg_start) / 16000.0
                        reason = self._evaluate_seg(self._last_preview_text, current_total, seg_start)
                        if reason is None:
                            reason = self._evaluate_tier234(self._last_preview_text, duration_sec)
                        if reason is not None:
                            overlap = int(
                                float(getattr(config.sentence, "boundary_overlap_ms", 0)) / 1000.0 * 16000
                            )
                            # SEG: mốc cắt có thể nằm TRONG vùng (nhờ timer có timestamp);
                            # phần audio giữa mốc cắt và hiện tại sẽ được nhận dạng lại ở
                            # câu kế tiếp nên KHÔNG mất chữ.
                            effective_end = current_total
                            if reason is CommitReason.SEG_PUNCT:
                                effective_end = self._seg_cut_sample(
                                    seg_start, current_total, self._last_preview_text
                                )
                                self._seg_commits += 1
                            with self._lock:
                                self._enqueue_commit_locked(
                                    utterance_id=self._active_utterance_id,
                                    start_sample=seg_start,
                                    end_sample=effective_end,
                                    reason=reason.value,
                                )
                                # câu mới bắt đầu ngay -> utterance_id mới
                                self._active_utterance_id = str(uuid.uuid4())[:8]
                                # F-47: CHỐT LUÔN mốc bắt đầu câu mới. Trước đây mốc này chỉ
                                # được cập nhật khi `_emit_commit` chạy xong, nên nếu commit
                                # bị BỎ (hết hạn thế hệ / mảnh rỗng) thì `duration_sec` vẫn
                                # lớn hơn `max_duration_sec` ⇒ vòng lặp enqueue lại cùng đoạn
                                # mãi mãi ⇒ quay nóng không nhường event loop (đã gây treo
                                # backend, watchdog F-46 bắt được stack tại `_pop_commit_request`).
                                self._speech_start_sample = max(0, effective_end - overlap)
                                self._last_preview_text = ""
                                self._last_preview_end_sample = -1
                                self._seg_decision = None
                                if self._seg is not None:
                                    self._seg.reset()
                            self.commit_manager.reset_stability()
                            self._wake_stream()

            # 3) Chờ theo nhịp
            if self._speech_active:
                if fixed_rate:
                    # P2.4b: nhịp hiệu dụng tự giãn ra khi backend chậm.
                    interval = self._effective_poll_interval(poll_interval)
                    # P2.4: lịch tuyệt đối; nếu trễ hạn thì BỎ nhịp thay vì trôi.
                    next_deadline += interval
                    now = time.perf_counter()
                    if next_deadline <= now:
                        skipped = 0
                        while next_deadline <= now:
                            next_deadline += interval
                            skipped += 1
                        metrics_collector.increment_counter("asr.preview_skipped", skipped)
                        wait_s = 0.0
                    else:
                        wait_s = next_deadline - now
                    t_wait = time.perf_counter()
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                    metrics_collector.record_metric("asr", "idle_wait_ms", (time.perf_counter() - t_wait) * 1000.0)
                else:
                    await asyncio.sleep(poll_interval)
            else:
                next_deadline = time.perf_counter()
                # Idle: đợi commit event (timeout để không stuck).
                # F-48: PHẢI `clear()` TRƯỚC KHI CHỜ. Nếu sự kiện đang SET mà không còn
                # commit nào (ví dụ `reset_stream`/`_wake_stream` được gọi lúc rảnh, hoặc
                # commit bị bỏ ở đường thoát sớm), thì `event.wait()` trả về NGAY LẬP TỨC,
                # `wait_for` không hề nhường event loop ⇒ vòng lặp quay nóng 100% CPU, nhịp
                # tim đứng im ⇒ watchdog báo "event loop đứng" và backend phải kill
                # (stack đã bắt được: `asyncio/tasks.py:459 in wait_for` ← engine.py:1308).
                if self._commit_event.is_set():
                    self._commit_event.clear()
                try:
                    await asyncio.wait_for(self._commit_event.wait(), timeout=poll_interval)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    raise
                # Bảo hiểm cuối: luôn nhường một nhịp thật để mọi coroutine khác (nhịp tim,
                # worker dịch/TTS, gửi WS) có cơ hội chạy dù nhánh trên trả về tức thì.
                await asyncio.sleep(0)

    async def reset_stream(self, reason: str = "seek") -> None:
        """F-44: XOÁ SẠCH trạng thái stream khi video bị tua/nhảy vị trí.

        Vì sao cần: khi người dùng tua, extension chuyển sang đoạn audio ở vị trí mới
        nhưng backend vẫn còn audio cũ trong buffer vòng và VAD có thể đang giữa câu ⇒
        câu commit trộn audio trước/sau khi tua (đã gặp mảnh ~54 s, phụ đề lặp lại nội
        dung cũ). Hàm này đưa pipeline về trạng thái "vừa bắt đầu":

        - xoá buffer audio (không để audio cũ lọt vào câu kế tiếp),
        - bỏ commit đang chờ + đặt lại trạng thái speech/preview và lịch preview,
        - tăng `_stream_generation`: kết quả inference đang bay của đoạn CŨ sẽ bị BỎ khi
          trả về (xem `stream_tokens`/`_emit_commit`) nên không cần huỷ native.

        LƯU Ý QUAN TRỌNG: hàm này **KHÔNG** gọi `cancel_inference()`. Đã thử và gây hỏng
        phiên: native `session.cancel()` giữa chừng làm các `run()` sau đó lỗi khiến worker
        ASR chết ⇒ WebSocket đóng (popup báo Disconnected, phải Start lại). Bỏ kết quả cũ
        bằng `_stream_generation` là đủ và an toàn.
        """
        self._stream_generation += 1
        gen = self._stream_generation
        with self._lock:
            self._speech_active = False
            self._pending_commits.clear()
            self._speech_start_sample = -1
            self._awaiting_pre_roll = False
            self._last_preview_end_sample = -1
            self._last_preview_text = ""
            self._last_committed_sample = -1
            self._speech_started_at = 0.0
            self._speech_ended_at = 0.0
            self._first_preview_reported = True
        self.audio_buffer.clear()
        self.commit_manager.reset_stability()
        if self._seg is not None:
            self._seg.reset()
            self._seg_decision = None
        self._preview_durations.clear()
        self._eff_poll_interval = max(0.05, self.poll_interval_ms / 1000.0)
        self._wake_stream()
        metrics_collector.increment_counter("asr.stream_reset")
        logger.info(
            f"Đã reset stream (lý do: {reason}) — xoá audio cũ, chờ audio mới. "
            f"generation={gen}",
            extra={"module_tag": "ASR"},
        )

    async def cleanup(self) -> None:
        """Dừng generator và giải phóng tài nguyên session."""
        self._is_running = False
        self._commit_event.set()  # Đánh thức stream_tokens ngay lập tức để thoát generator
        with self._lock:
            self._speech_active = False
            self._pending_commits.clear()
        self.audio_buffer.clear()
        # F-39 (native): nếu RAM đã phình quá xa mốc lúc nạp model thì đóng phiên native
        # để lần sau nạp lại sạch. Lỗi nằm trong native/Vulkan nên đây là cách duy nhất
        # đòi lại bộ nhớ mà không cần patch upstream.
        try:
            await self.maybe_recycle_native()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Bỏ qua recycle native: {exc}", extra={"module_tag": "ASR"})

    async def cancel_inference(self) -> None:
        """P1.8/T2.6: yêu cầu huỷ inference đang chạy (nếu binding hỗ trợ).

        Huỷ task chỉ huỷ future của asyncio; thread C++ vẫn chạy hết và vẫn giữ
        `_infer_lock`, khiến session mới phải chờ. Gọi `Session.cancel()` để native
        side dừng sớm.

        FIX-12: đọc `_shared_session` **trong** `_shared_lock` rồi mới gọi `.cancel()` trên
        tham chiếu cục bộ. Trước đây đọc trực tiếp không lock, nên có thể đua với
        hot-swap/unload (`prepare_model()`/`unload_shared_model()` đóng native handle) ⇒
        gọi `cancel()` lên handle đã giải phóng. KHÔNG giữ lock trong lời gọi native.
        """
        with self.__class__._shared_lock:
            session = self.__class__._shared_session
        if session is None:
            return
        try:
            session.cancel()
            metrics_collector.increment_counter("asr.inference_cancelled")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Không huỷ được inference: {exc}", extra={"module_tag": "ASR"})
