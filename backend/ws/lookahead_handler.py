"""Lookahead WebSocket Handler — Pipeline B (Zero Perceived Latency) v2.

TRIẾT LÝ THIẾT KẾ (v2 — viết lại sau khi bản v1 mất phụ đề gốc + bản dịch)
=========================================================================

Bản v1 cắt audio thành lưới 4 giây cố định rồi gọi ASR cho TỪNG mảnh rời. Cách đó sai ở
ba điểm chí mạng, đã kiểm chứng bằng log thật + đo lường (xem
`report/08_lookahead_video_buffering/pipeline_b_redesign_v2.md` §2 và test
`backend/tests/test_50_lookahead_demuxer.py::test_stream_demuxer_reassembles_fragmented_webm`):

1. **Giải mã từng mảnh MSE độc lập là bất khả thi.** Trình duyệt append các mảnh ~32 KB
   nằm GIỮA cluster WebM; PyAV chỉ đọc được ~35 % lượng audio ⇒ phần lớn câu không bao
   giờ tới được ASR.
2. **Cắt theo lưới 4 giây phá ngữ cảnh câu.** Model nhận 4 giây cụt đầu cụt đuôi nên trả
   về chữ rác/hallucination, và cùng một đoạn bị xử lý LẶP nhiều lần (khi buffer lớn dần,
   `is_pts_processed` không khớp nên một mốc bị dịch 2-3 lần với nội dung khác nhau).
3. **Không có VAD ⇒ không có mốc câu thật.** Mốc gửi về client là mốc lưới, không phải
   mốc người nói, nên `SubtitleTimelineQueue` không bao giờ khớp `video.currentTime`.

Kiến trúc v2 — DÙNG LẠI ĐÚNG PIPELINE A:

    mảnh MSE ──► StreamDemuxer (ghép nối byte liên tục + giải mã tăng dần)
              ──► ContinuousAudioTimeline (dòng PCM LIÊN TỤC, lấp khe hở bằng silence)
              ──► VADProcessor.feed_chunk  (giống hệt /ws realtime)
              ──► TranscribeEngine.stream_tokens (VAD → commit câu, tắt preview)
              ──► ánh xạ sample → PTS tuyệt đối qua "anchor" ghi lúc nạp
              ──► Dịch GGUF
              ──► lookahead_subtitles {start_pts, end_pts} → client hiện đúng lúc phát

Nhờ vậy chất lượng phụ đề/gốc và bản dịch **giống hệt** Pipeline A, nhưng mốc thời gian
được neo vào timeline video nên phụ đề xuất hiện đúng 0.0 s khi video chạy tới.
"""

from __future__ import annotations

import asyncio
import base64
import bisect
import hashlib
import json
import struct
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from backend.config import config
from backend.core.commit_manager import count_content_tokens
from backend.core.lookahead_timeline import ContinuousAudioTimeline
from backend.core.metrics import metrics_collector
from backend.core.stream_demuxer import StreamDemuxer
from backend.ws.connection import SafeWebSocketConnection
from backend.ws.session import SessionConfigPayload
from backend.utils.logger import get_logger

logger = get_logger("ws.lookahead")

#: Số mốc neo (sample → PTS) giữ lại. Mỗi mốc neo = một lần VAD mở đoạn nói mới.
_MAX_ANCHORS = 256
#: Cửa sổ (giây) coi một mảnh giống hệt là "gửi trùng". Rộng hơn mọi khoảng replay cache,
#: nhưng đủ hẹp để mảnh nạp lại sau khi tua vẫn được chấp nhận.
_DUP_WINDOW_SEC = 15.0
#: Lệch quá ngần này giây giữa audio nhận được và vị trí phát ⇒ cảnh báo "đoạn đầu sẽ không
#: có phụ đề" (dấu hiệu cache mảnh phía Extension không bám vị trí phát).
_START_OFFSET_WARN_SEC = 20.0
#: Xê dịch tối đa (giây) giữa mốc neo hiện tại và vị trí báo mới mà vẫn coi là CÙNG một mốc
#: (không reset pipeline). Chống reset lặp khi `sync_state` gửi `seek_id` mới ở cùng vị trí.
_REANCHOR_TOLERANCE_SEC = 1.5
#: Giãn cách tối thiểu giữa hai lần yêu cầu client gửi lại mảnh (chống spam).
_REPLAY_REQUEST_COOLDOWN_SEC = 4.0


def _pack_binary_frame(header: Dict[str, Any], payload: bytes) -> bytes:
    """Đóng gói khung nhị phân `[4B header_len][JSON header][payload]` (cùng bố cục với
    khung audio client gửi lên ⇒ phía Extension chỉ cần một hàm phân tích)."""
    raw = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return struct.pack("<I", len(raw)) + raw + payload


def _clip_dubbing_text(text: str, max_chars: int) -> str:
    """Cắt bớt văn bản lồng tiếng cho vừa trần (cắt ở ranh giới từ)."""
    clean = (text or "").strip()
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return cut.strip()


def _free_vram_mb() -> Optional[float]:
    """VRAM trống (MB) của GPU đang dùng, hoặc None nếu không đo được (CPU/không có torch)."""
    try:
        import torch  # nội bộ, chỉ có khi TTS chạy được

        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info()
        return float(free_bytes) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


def _empty_torch_cache() -> None:
    """Trả các khối đã cache của PyTorch về driver (giảm phân mảnh khi VRAM căng)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class LookaheadSessionState:
    """Phiên Lookahead: biến mảnh audio tương lai thành phụ đề có mốc thời gian chính xác."""

    def __init__(
        self,
        ws: SafeWebSocketConnection,
        asr_engine: Any = None,
        translation_engine: Any = None,
    ):
        self.session_id: str = str(uuid.uuid4())
        self.connection = ws

        la = config.lookahead
        self.demuxer = StreamDemuxer(
            target_sample_rate=16000,
            max_media_bytes=int(la.max_media_bytes),
            keep_boundaries=int(la.keep_boundaries),
        )
        self.timeline = ContinuousAudioTimeline(
            sample_rate=16000, max_pending_sec=float(la.max_pending_sec)
        )

        # Trạng thái phiên
        self.current_time: float = 0.0
        self.playback_rate: float = 1.0
        self.is_paused: bool = False
        self.active_seek_id: str = "init_0"
        self.source_lang: str = "auto"
        self.target_lang: str = "vi"
        self.lead_time: float = float(la.lead_time_sec)
        #: Offset người dùng chỉnh trong popup (ms) để canh phụ đề/TTS khớp tiếng nói.
        self.sync_offset_ms: float = 0.0
        #: Phần bù cố định theo VAD engine (engine báo mép nói sớm hơn thực tế bao nhiêu ms).
        self._start_pad_ms: int = 0

        # Engines
        self.asr_engine = asr_engine
        self.translation_engine = translation_engine
        self.vad_processor = None
        self.unavailable_reason: str = ""

        # Cấu hình phiên (từ popup). `_parsed_config` giữ payload đã validate để áp khi
        # dựng components; các thay đổi sau đó đi qua `apply_config()`.
        self._parsed_config: Optional[SessionConfigPayload] = None
        self._requested_asr_model: str = ""
        #: `min_words_to_commit` hiệu dụng của RIÊNG phiên (popup có thể đổi khác mặc định).
        self._min_words_to_commit: int = int(config.sentence.min_words_to_commit)

        # ── TTS / lồng tiếng ──────────────────────────────────────────────────
        #: Bật/tắt theo popup. Audio TTS được gắn mốc `start_pts` để client phát ĐÚNG lúc
        #: `video.currentTime` đi qua phụ đề, và được NÉN vừa ngân sách thời gian để không
        #: đẩy lùi các câu sau (xem `synthesize_fitted_bytes`).
        self.tts_enabled: bool = bool(getattr(config.tts, "enabled", False))
        self.tts_voice: str = str(getattr(config.tts, "default_voice", "") or "")
        self.tts_speed: float = float(getattr(config.tts, "speed", 1.0) or 1.0)
        self.tts_queue: Optional[asyncio.Queue] = None
        self.tts_sent: int = 0
        self.tts_skipped: int = 0

        # Ánh xạ sample ASR -> PTS tuyệt đối. Engine chỉ nhận frame ĐOẠN NÓI (VAD lọc),
        # nên chỉ số mẫu trong `audio_buffer` KHÔNG tuyến tính với thời gian video. Ta ghi
        # một "anchor" mỗi khi có điểm không liên tục (đầu mỗi đoạn nói / sau khi hụt audio).
        self._anchor_lock = threading.RLock()
        self._anchor_samples: List[int] = []
        self._anchor_pts: List[float] = []

        # Mốc PTS của mẫu kế tiếp sẽ nạp vào VAD
        self._feed_pts: Optional[float] = None
        self._session_start_pts: Optional[float] = None
        #: Mốc kết thúc audio ĐÃ NHẬN & GIẢI MÃ (kể cả phần đã nạp vào VAD). Dùng để báo
        #: `buffered_ahead` trung thực: nếu chỉ nhìn phần còn chờ trong timeline thì sau khi
        #: nạp hết, buffer sẽ báo 0 dù trình duyệt vẫn còn cả chục giây audio phía trước.
        self._decoded_end_pts: float = 0.0
        self._ready_until_pts: float = 0.0
        #: Lần cuối cùng `prebuffer_ready` đổi giá trị (để chỉ log khi có thay đổi).
        self._last_ready_flag: bool = False
        self._seek_seq: int = 0

        # Sản phẩm đã gửi (chống trùng lặp chồng lấn)
        self._sent_items: List[Tuple[float, float, str]] = []
        self._recent_utterances: List[Dict[str, Any]] = []

        # Vòng đời
        self._closed = False
        self._ingest_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._bg_tasks: set = set()

        # Thống kê
        self.fragments_received = 0
        self.chunks_decoded = 0
        self.utterances_sent = 0
        self.started_at = time.time()
        # ── Chẩn đoán NĂNG LỰC GIẢI MÃ (2026-10-01) ──────────────────────────────
        # Sự cố "trang khác chỉ hiện 1/3-5 câu": cần biết tầng nào là nút cổ chai —
        # `decoder` (byte tới nhiều mà PCM giải mã ra ít) hay `VAD` (PCM có sẵn mà nạp chậm).
        # `bytes_in` / `fed_seconds` cho tỷ lệ giải mã và tốc độ nạp tính bằng số THẬT.
        self.bytes_in = 0
        self.decode_seconds = 0.0          # tổng thời lượng PCM giải mã ra (đã lọc)
        self.decode_wall_sec = 0.0         # tổng thời gian wall của các lời gọi demuxer.feed
        self.media_seconds = 0.0           # tổng giây MEDIA giải mã ra (kể cả phần bị bỏ xa)
        self.decode_delta_sec = 0.0        # giây media trong cửa sổ chẩn đoán hiện tại
        self.decode_windows = 0            # số lượt feed CÓ sinh ra PCM
        self.chunks_dropped_far = 0        # số đoạn PCM bị bỏ vì quá xa vị trí phát
        self.start_offset_warned = 0       # số lần cảnh báo lệch đầu phiên (chỉ log vài lần)
        self.replay_requests = 0           # số lần yêu cầu client gửi lại mảnh
        self._last_replay_request_at = 0.0
        #: Chẩn đoán sau khi tua (xem `handle_audio_fragment`).
        self._seek_reset_at = 0.0
        self._seek_diag_left = 0
        self.fragments_deduped = 0         # số mảnh bị bỏ vì trùng nội dung
        self._fragment_digests: Dict[bytes, float] = {}
        self._fragment_digest_order: deque = deque()
        self._max_decode_lead_sec = float(getattr(la, "max_decode_lead_sec", 240.0) or 240.0)
        self._playhead_pts = 0.0
        self.fed_seconds = 0.0             # tổng audio đã nạp vào VAD
        self.feed_wall_sec = 0.0           # tổng thời gian wall của feed_chunk (đo VAD)
        self._feed_wall_max = 0.0
        self._last_decoded_at = 0.0        # lần cuối có PCM mới
        self._starved_since: Optional[float] = None
        self._starved_bytes = 0
        self._starved_max_sec = 0.0
        #: Mốc so sánh cho log chẩn đoán định kỳ (xem `_maybe_log_diagnostics`).
        self._last_diag_at = time.perf_counter()
        self._diag_bytes_in = 0
        self._diag_decode_seconds = 0.0
        self._diag_media_seconds = 0.0
        self._diag_decode_wall = 0.0
        self._diag_fed_seconds = 0.0
        self._diag_feed_wall = 0.0
        #: Lần cuối in cấu trúc container (thưa hơn log chẩn đoán vì tốn CPU).
        self._last_struct_at = time.perf_counter()

    # ────────────────────────────────────────────────────────────── gửi tin
    async def send_json(self, payload: Dict[str, Any]) -> bool:
        if self._closed:
            return False
        return await self.connection.send_json(payload)

    # ────────────────────────────────────────────────────────── khởi tạo engine
    def init_components(
        self,
        vad_engine: Optional[str] = None,
        vad_engine_override: Any = None,
    ) -> None:
        """Dựng VAD + ASR theo ĐÚNG cấu hình popup, giống `SessionState.init_components`.

        `vad_engine_override` cho phép TIÊM engine VAD giả trong test tầng A
        (xem `backend/tests/fakes.py`).
        """
        from backend.asr.engine import TranscribeEngine
        from backend.vad.processor import VADProcessor

        if self.asr_engine is None:
            self.asr_engine = TranscribeEngine(
                model_key=self._requested_asr_model or None,
                session_id=f"la_{self.session_id[:8]}",
                language=self.source_lang or "auto",
            )

        # Phân câu: bản SAO cấu hình để không đụng cấu hình toàn cục của Pipeline A.
        try:
            self.asr_engine.commit_manager.cfg = config.sentence.model_copy(deep=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Không tạo được bản cấu hình phân câu riêng: {exc}", extra={"module_tag": "WS"})

        # Ngôn ngữ nguồn: popup chọn gì thì ASR nhận đúng cái đó.
        try:
            self.asr_engine.set_language(self.source_lang or "auto")
        except Exception:  # noqa: BLE001
            pass

        # VAD: dựng NGAY với thông số popup (engine/threshold/silence) để các chunk đầu
        # tiên đã dùng đúng cấu hình, không phải chờ cơ chế đổi engine ở chunk kế tiếp.
        parsed = self._parsed_config
        vad_name = (parsed.vad_engine if parsed and parsed.vad_engine else None) \
            or vad_engine or config.vad.vad_engine
        vad_threshold = config.vad.effective_threshold
        if parsed is not None:
            _th = parsed.vad_threshold if parsed.vad_threshold is not None else parsed.threshold
            if _th is not None:
                vad_threshold = float(_th)
        vad_silence = config.vad.effective_silence_ms
        if parsed is not None and parsed.silence_duration_ms is not None:
            _raw = int(parsed.silence_duration_ms)
            vad_silence = _raw if _raw > 0 else None
        vad_enabled = True
        if parsed is not None and parsed.vad_enabled is not None:
            vad_enabled = bool(parsed.vad_enabled)

        self.vad_processor = VADProcessor(
            sample_rate=config.vad.sample_rate,
            vad_engine=vad_name,
            threshold=vad_threshold,
            silence_duration_ms=vad_silence,
            enabled=vad_enabled,
            on_speech_chunk=self._on_speech_chunk,
            on_speech_start=self.asr_engine.on_speech_start,
            on_speech_end=self.asr_engine.on_speech_end,
            engine_override=vad_engine_override,
        )

        # Áp cấu hình popup (VAD/threshold/silence/phân câu/từ tối thiểu…) rồi mới chốt các
        # override riêng của Lookahead (chúng phải thắng).
        if self._parsed_config is not None:
            self._apply_config_sync(self._parsed_config)
        self._apply_lookahead_overrides()
        self._refresh_start_pad()

    def _refresh_start_pad(self) -> None:
        """Đọc phần bù mép-nói-sớm của VAD engine đang dùng (để canh phụ đề/TTS)."""
        try:
            self._start_pad_ms = int(getattr(self.vad_processor, "start_pad_ms", 0) or 0)
        except Exception:  # noqa: BLE001
            self._start_pad_ms = 0
        if self._start_pad_ms:
            logger.info(
                f"Canh mốc phụ đề/TTS: VAD '{getattr(self.vad_processor, 'vad_engine', '?')}' báo mép nói "
                f"sớm {self._start_pad_ms}ms — sẽ cộng bù (offset người dùng "
                f"{self.sync_offset_ms:+.0f}ms).",
                extra={"module_tag": "WS"},
            )

    # ────────────────────────────────────────────────── cấu hình từ popup
    def _apply_lookahead_overrides(self) -> None:
        """Override RIÊNG của Lookahead (áp SAU cấu hình popup để luôn thắng)."""
        la = config.lookahead
        engine = self.asr_engine
        if engine is None:
            return
        # BẬC 4 (TIMEOUT_FORCE) dùng đồng hồ thời gian thực; Lookahead nạp audio theo lô nên
        # đồng hồ này dễ chốt oan giữa câu ⇒ nới trần.
        try:
            engine.commit_manager.cfg.inactivity_timeout_sec = float(la.inactivity_timeout_sec)
        except Exception:  # noqa: BLE001
            pass
        # Preview vẫn BẬT (nguồn duy nhất của BẬC 3) nhưng chỉ chạy khi có audio mới.
        engine.preview_enabled = bool(la.preview_enabled)
        engine.preview_requires_new_audio = True
        engine.preview_min_new_audio_sec = float(la.preview_min_new_audio_sec)
        engine.poll_interval_ms = 100
        engine._eff_poll_interval = 0.1

    def _apply_config_sync(self, parsed: "SessionConfigPayload") -> Dict[str, Any]:
        """Áp phần cấu hình KHÔNG cần nạp model (ngôn ngữ, VAD, phân câu, từ tối thiểu)."""
        applied: Dict[str, Any] = {}

        if parsed.source_lang:
            self.source_lang = str(parsed.source_lang)
            applied["source_lang"] = self.source_lang
            if self.asr_engine is not None:
                try:
                    self.asr_engine.set_language(self.source_lang)
                except Exception:  # noqa: BLE001
                    pass
        if parsed.target_lang:
            self.target_lang = str(parsed.target_lang)
            applied["target_lang"] = self.target_lang

        # ── TTS (lồng tiếng) ──────────────────────────────────────────────────
        if parsed.tts_enabled is not None:
            new_tts_enabled = bool(parsed.tts_enabled)
            if self.tts_enabled and not new_tts_enabled:
                # Người dùng vừa TẮT TTS giữa phiên:
                self.tts_enabled = False
                if self.tts_queue is not None:
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                            self.tts_queue.task_done()
                        except Exception:
                            break
                logger.info("Lookahead: TTS đã bị TẮT giữa phiên — huỷ hàng đợi TTS.", extra={"module_tag": "TTS"})
            elif not self.tts_enabled and new_tts_enabled:
                # Người dùng vừa BẬT LẠI TTS giữa phiên:
                self.tts_enabled = True
                self.tts_sent = 0
                self._ensure_tts_worker()
                self._spawn(self._prewarm_tts())
                logger.info("Lookahead: TTS đã được BẬT LẠI giữa phiên — nạp lại các câu sắp tới.", extra={"module_tag": "TTS"})
                if self.tts_queue is not None:
                    now_pts = float(self.current_time)
                    for u in list(self._recent_utterances):
                        if u["pts_end"] > now_pts and u.get("translated"):
                            self._enqueue_tts(u["translated"], u["pts_start"], u["pts_end"])
            else:
                self.tts_enabled = new_tts_enabled
                if self.tts_enabled:
                    self._ensure_tts_worker()
            applied["tts_enabled"] = self.tts_enabled
        if parsed.tts_voice is not None:
            self.tts_voice = str(parsed.tts_voice)
            applied["tts_voice"] = self.tts_voice
        if parsed.tts_speed is not None:
            self.tts_speed = float(parsed.tts_speed)
            applied["tts_speed"] = self.tts_speed

        engine = self.asr_engine
        if engine is not None:
            updates: Dict[str, Any] = {}
            if parsed.split_on_stability is not None:
                updates["split_on_stability"] = bool(parsed.split_on_stability)
            elif parsed.enable_stability_split is not None:
                updates["split_on_stability"] = bool(parsed.enable_stability_split)
            if parsed.stability_duration_sec is not None:
                updates["stability_duration_sec"] = float(parsed.stability_duration_sec)
            if parsed.max_duration_sec is not None:
                updates["max_duration_sec"] = float(parsed.max_duration_sec)
            if parsed.max_chars is not None:
                updates["max_chars"] = int(parsed.max_chars)
            if parsed.min_words_to_commit is not None:
                self._min_words_to_commit = max(0, int(parsed.min_words_to_commit))
                updates["min_words_to_commit"] = self._min_words_to_commit
                applied["min_words_to_commit"] = self._min_words_to_commit
            if parsed.stability_min_duration_sec is not None:
                updates["stability_min_duration_sec"] = float(parsed.stability_min_duration_sec)
            if parsed.stability_min_words is not None:
                updates["stability_min_words"] = int(parsed.stability_min_words)
            if parsed.trace_stability is not None:
                updates["trace_stability"] = bool(parsed.trace_stability)
            if parsed.hold_short_sentence is not None:
                updates["hold_short_sentence"] = bool(parsed.hold_short_sentence)
            if updates:
                try:
                    engine.update_sentence_config(**updates)
                    applied.update(updates)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Áp cấu hình phân câu Lookahead lỗi: {exc}", extra={"module_tag": "WS"})

            # Cửa sổ preview: popup chỉnh được (P2.3).
            if parsed.preview_window_sec is not None:
                config.asr.preview_window_sec = float(parsed.preview_window_sec)
                applied["preview_window_sec"] = config.asr.preview_window_sec
            if parsed.poll_interval_ms is not None:
                v = max(50, int(parsed.poll_interval_ms))
                engine.poll_interval_ms = v
                engine._eff_poll_interval = v / 1000.0
                engine._preview_durations.clear()
                applied["poll_interval_ms"] = v

        if self.vad_processor is not None:
            vad_kwargs: Dict[str, Any] = {}
            if parsed.vad_engine:
                vad_kwargs["vad_engine"] = str(parsed.vad_engine)
            th = parsed.vad_threshold if parsed.vad_threshold is not None else parsed.threshold
            if th is not None:
                vad_kwargs["threshold"] = float(th)
            if parsed.silence_duration_ms is not None:
                raw_ms = int(parsed.silence_duration_ms)
                vad_kwargs["silence_duration_ms"] = raw_ms if raw_ms > 0 else None
            if parsed.vad_enabled is not None:
                vad_kwargs["enabled"] = bool(parsed.vad_enabled)
            if vad_kwargs:
                try:
                    self.vad_processor.update_config(**vad_kwargs)
                    applied.update({f"vad_{k}": v for k, v in vad_kwargs.items()})
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Áp cấu hình VAD Lookahead lỗi: {exc}", extra={"module_tag": "WS"})

        return applied

    def _schedule_asr_model_switch(self, model_key: str) -> None:
        """Đổi model ASR cho phiên Lookahead (tác vụ nền, không chặn vòng nhận tin)."""
        from backend.asr import hotswap as asr_hotswap
        from backend.asr.registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        clean = (model_key or "").strip().lower()
        if not registry.has_model(clean):
            logger.warning(f"Model ASR '{model_key}' không có trong catalog — bỏ qua.", extra={"module_tag": "WS"})
            return
        if asr_hotswap.is_busy() and not asr_hotswap.is_busy(clean):
            logger.info("Đang nạp model ASR khác — bỏ qua yêu cầu đổi model.", extra={"module_tag": "WS"})
            return

        async def _run() -> None:
            await self.send_json({"type": "model_status", "stage": "asr", "state": "loading", "model": clean})
            try:
                if asr_hotswap.needs_download(clean) and not asr_hotswap.auto_download_enabled():
                    raise RuntimeError("Chưa có file GGUF cục bộ và auto_download đang tắt")
                await asr_hotswap.activate_model(clean)
                registry.set_active_model_key(clean)
                # Đồng bộ engine của phiên: nếu không, lần suy luận kế tiếp sẽ nạp LẠI model cũ
                # (vì `_ensure_model_loaded` so `model_key` của instance với model chia sẻ).
                if self.asr_engine is not None:
                    self.asr_engine.model_key = clean
                    self.asr_engine.model_info = registry.get_model_info(clean) or {}
                await self.send_json({"type": "model_status", "stage": "asr", "state": "ready", "model": clean})
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Đổi model ASR Lookahead sang '{clean}' thất bại: {exc}", extra={"module_tag": "WS"})
                await self.send_json({
                    "type": "model_status", "stage": "asr", "state": "error",
                    "model": clean, "message": str(exc),
                })

        self._spawn(_run())

    def _schedule_translation_model_switch(self, model_request: str) -> None:
        """Đổi model dịch cho phiên Lookahead (tác vụ nền)."""
        from backend.translation import hotswap as trans_hotswap
        from backend.translation.registry import TranslationModelRegistry

        registry = TranslationModelRegistry.get_instance()
        if not registry.is_known(model_request):
            logger.warning(f"Model dịch '{model_request}' không có trong catalog — bỏ qua.", extra={"module_tag": "WS"})
            return
        canonical = registry.resolve_key(model_request)
        if canonical == getattr(config.translation, "base", None):
            return
        if trans_hotswap.is_busy() and not trans_hotswap.is_busy(canonical):
            logger.info("Đang nạp model dịch khác — bỏ qua yêu cầu đổi model.", extra={"module_tag": "WS"})
            return

        async def _run() -> None:
            await self.send_json({"type": "model_status", "stage": "translation", "state": "loading", "model": canonical})
            try:
                if trans_hotswap.needs_download(canonical) and not trans_hotswap.auto_download_enabled():
                    raise RuntimeError("Chưa có file GGUF cục bộ và auto_download đang tắt")
                await trans_hotswap.activate_model(canonical)
                await self.send_json({"type": "model_status", "stage": "translation", "state": "ready", "model": canonical})
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Đổi model dịch Lookahead sang '{canonical}' thất bại: {exc}", extra={"module_tag": "WS"})
                await self.send_json({
                    "type": "model_status", "stage": "translation", "state": "error",
                    "model": canonical, "message": str(exc),
                })

        self._spawn(_run())

    def _spawn(self, coro) -> None:
        """Chạy tác vụ nền và giữ tham chiếu (tránh bị GC thu hồi giữa đường)."""
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            return
        self._bg_tasks.add(task)
        task.add_done_callback(lambda t: self._bg_tasks.discard(t))

    async def apply_config(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Áp cấu hình popup cho phiên Lookahead (cả khi đang chạy)."""
        parsed = self._parse_config(raw)
        if parsed is None:
            return {}

        self._parsed_config = parsed
        applied = self._apply_config_sync(parsed)
        self._apply_lookahead_overrides()
        self._refresh_start_pad()

        sync_raw = raw.get("lookaheadSyncOffsetMs", raw.get("lookahead_sync_offset_ms"))
        if sync_raw is not None:
            try:
                self.sync_offset_ms = max(-1500.0, min(1500.0, float(sync_raw)))
                applied["sync_offset_ms"] = self.sync_offset_ms
            except (TypeError, ValueError):
                pass

        requested_asr = parsed.asr_model or parsed.asr_engine or parsed.model_id
        if requested_asr:
            clean = str(requested_asr).strip().lower()
            self._requested_asr_model = clean
            if self.asr_engine is not None and clean != getattr(self.asr_engine, "model_key", None):
                self._schedule_asr_model_switch(clean)
                applied["asr_model"] = clean

        if parsed.translation_model is not None:
            self._schedule_translation_model_switch(str(parsed.translation_model))
            applied["translation_model"] = str(parsed.translation_model)

        if applied:
            logger.info(
                f"Cấu hình Lookahead từ popup: "
                + ", ".join(f"{k}={v}" for k, v in list(applied.items())[:12]),
                extra={"module_tag": "WS"},
            )
        return applied

    def _parse_config(self, data: Dict[str, Any]) -> Optional[SessionConfigPayload]:
        try:
            return SessionConfigPayload.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Payload cấu hình Lookahead không hợp lệ: {exc}", extra={"module_tag": "WS"})
            return None

    async def start_tasks(self) -> None:
        """Khởi động các vòng lặp nền: nạp audio, tiêu thụ ASR, báo trạng thái (+TTS)."""
        loop = asyncio.get_running_loop()
        base_tasks = [
            loop.create_task(self._ingest_loop(), name=f"la_ingest_{self.session_id[:8]}"),
            loop.create_task(self._asr_loop(), name=f"la_asr_{self.session_id[:8]}"),
            loop.create_task(self._status_loop(), name=f"la_status_{self.session_id[:8]}"),
        ]
        self._tasks.extend(base_tasks)
        if self.tts_enabled:
            self._ensure_tts_worker()

    def _ensure_tts_worker(self) -> None:
        """Tạo hàng đợi + worker TTS (IDEMPOTENT — gọi từ `start_tasks()` và từ
        `apply_config()` khi người dùng bật TTS giữa phiên)."""
        if not self.tts_enabled or self.tts_queue is not None or self._closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self.tts_queue = asyncio.Queue(
            maxsize=max(1, int(getattr(config.ws, "tts_queue_maxsize", 32) or 32))
        )
        self._tasks.append(loop.create_task(self._tts_loop(), name=f"la_tts_{self.session_id[:8]}"))
        # Nạp trước model TTS để câu đầu tiên không phải chờ (giống Pipeline A).
        self._spawn(self._prewarm_tts())

    async def _prewarm_tts(self) -> None:
        try:
            from backend.tts import get_tts_engine
            await get_tts_engine().prewarm()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Prewarm TTS cho Lookahead lỗi (bỏ qua): {exc}", extra={"module_tag": "TTS"})

    async def close(self) -> None:
        """Đóng phiên: huỷ mọi tác vụ và giải phóng engine (có trần thời gian)."""
        if self._closed:
            return
        self._closed = True
        self._ingest_event.set()

        for task in list(self._bg_tasks):
            task.cancel()
        for task in self._tasks:
            task.cancel()
        pending = list(self._tasks) + list(self._bg_tasks)
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=3.0
                )
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
        self._tasks = []
        self._bg_tasks.clear()

        engine = self.asr_engine
        if engine is not None:
            # `cancel_inference()` có thể chờ `_shared_lock` (đang nạp model) và
            # `cleanup()` chạy native — PHẢI có trần thời gian, nếu không phiên không bao
            # giờ đóng được và tài nguyên bị giữ lại sau khi người dùng bấm Stop.
            try:
                await asyncio.wait_for(engine.cancel_inference(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(engine.cleanup(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                logger.debug("Lookahead cleanup engine quá hạn (bỏ qua).", extra={"module_tag": "WS"})

    # ────────────────────────────────────────────────────── ánh xạ sample → PTS
    def _pts_at(self, sample_index: Optional[int]) -> Optional[float]:
        """PTS tuyệt đối (giây) của `sample_index` trong bộ đệm ASR."""
        if sample_index is None:
            return None
        with self._anchor_lock:
            if not self._anchor_samples:
                return None
            idx = bisect.bisect_right(self._anchor_samples, int(sample_index)) - 1
            if idx < 0:
                idx = 0
            base_sample = self._anchor_samples[idx]
            base_pts = self._anchor_pts[idx]
        return base_pts + (int(sample_index) - base_sample) / 16000.0

    async def _request_replay(self, reason: str) -> None:
        """Yêu cầu client GỬI LẠI các mảnh quanh vị trí phát hiện tại.

        Dùng khi audio nhận được lệch xa vị trí phát (thường sau khi tua): cache phía client
        còn giữ mảnh của vị trí cũ nên pipeline không có gì để dịch. Có chốt thời gian để
        không spam client mỗi mảnh.
        """
        now = time.perf_counter()
        if now - self._last_replay_request_at < _REPLAY_REQUEST_COOLDOWN_SEC:
            return
        self._last_replay_request_at = now
        self.replay_requests += 1
        logger.info(
            f"Yêu cầu client gửi lại mảnh quanh vị trí phát {self.current_time:.1f}s "
            f"(lý do: {reason}, tổng {self.replay_requests} lần).",
            extra={"module_tag": "WS"},
        )
        await self.send_json({
            "type": "lookahead_request_replay",
            "seek_id": self.active_seek_id,
            "currentTime": round(self.current_time, 2),
            "reason": reason,
        })

    def _record_anchor(self, sample_index: int, pts: float) -> None:
        """Ghi mốc neo nếu PTS không nối tiếp tuyến tính với mốc trước."""
        with self._anchor_lock:
            if self._anchor_samples:
                expected = self._pts_at(sample_index)
                if expected is not None and abs(expected - pts) <= 0.005:
                    return
            self._anchor_samples.append(int(sample_index))
            self._anchor_pts.append(float(pts))
            if len(self._anchor_samples) > _MAX_ANCHORS:
                del self._anchor_samples[0]
                del self._anchor_pts[0]

    def _prune_anchors(self) -> None:
        """Bỏ các mốc neo đã ra khỏi bộ đệm vòng của ASR (không còn commit nào dùng tới)."""
        engine = self.asr_engine
        if engine is None:
            return
        try:
            total = int(engine.audio_buffer.total_written)
            capacity = int(engine.audio_buffer.capacity_samples)
        except Exception:
            return
        cutoff = total - int(capacity * 0.9)
        with self._anchor_lock:
            drop = 0
            while (
                drop + 1 < len(self._anchor_samples)
                and self._anchor_samples[drop + 1] <= cutoff
            ):
                drop += 1
            if drop > 0:
                del self._anchor_samples[:drop]
                del self._anchor_pts[:drop]

    # ─────────────────────────────────────────────────────────────── nạp audio
    def _on_speech_chunk(self, pcm_bytes: bytes, ts: float, vad_state: str = "") -> None:
        """Callback VAD: ghi mốc neo rồi đẩy PCM NGUYÊN BẢN vào ASR (giống Pipeline A)."""
        engine = self.asr_engine
        if engine is None or not pcm_bytes:
            return
        try:
            sample_index = int(engine.audio_buffer.total_written)
            self._record_anchor(sample_index, float(ts))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Không ghi được mốc neo: {exc}", extra={"module_tag": "WS"})
        engine.feed_audio(pcm_bytes, timestamp=float(ts), vad_state=vad_state)

    def _feed_target_pts(self) -> float:
        """Nạp audio tới mốc nào thì dừng (chặn đốt GPU vô ích)."""
        la = config.lookahead
        rate = max(1.0, float(self.playback_rate or 1.0))
        target = self.current_time + float(self.lead_time) * rate + float(la.feed_margin_sec)
        if self._session_start_pts is not None:
            target = max(target, self._session_start_pts + float(la.min_prebuffer_sec))
        return target

    def _next_feed_pts(self) -> Optional[float]:
        if self._feed_pts is not None:
            return self._feed_pts
        return self.timeline.cursor_pts

    def _feed_block(self, pcm: np.ndarray, pts: float) -> None:
        """Đẩy một khối PCM vào VAD (chạy trên thread, KHÔNG block event loop)."""
        if self.vad_processor is None:
            return
        self.vad_processor.feed_chunk(pcm, capture_timestamp=float(pts))

    async def _ingest_loop(self) -> None:
        """Vòng nạp: đọc dòng PCM liên tục và đẩy vào VAD theo đúng nhu cầu."""
        while not self._closed:
            try:
                await asyncio.wait_for(self._ingest_event.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass
            self._ingest_event.clear()
            try:
                await self.ingest_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Lỗi vòng nạp Lookahead: {exc}", exc_info=True, extra={"module_tag": "WS"})

    async def ingest_once(self) -> int:
        """Nạp tối đa tới mốc nhu cầu hiện tại. Trả về số khối đã nạp."""
        if self._closed or self.vad_processor is None:
            return 0
        fed = 0
        block_sec = float(config.lookahead.feed_block_sec)
        while not self._closed:
            next_pts = self._next_feed_pts()
            if next_pts is None:
                break
            if self._session_start_pts is None:
                self._session_start_pts = float(next_pts)
            remaining = self._feed_target_pts() - next_pts
            if remaining <= 0.02:
                break
            item = self.timeline.read(min(block_sec, remaining))
            if item is None:
                break
            pts, pcm = item
            if pcm.size == 0:
                continue
            t_feed = time.perf_counter()
            try:
                await asyncio.to_thread(self._feed_block, pcm, pts)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Nạp khối audio vào VAD lỗi: {exc}", extra={"module_tag": "WS"})
                break
            _feed_wall = time.perf_counter() - t_feed
            self.feed_wall_sec += _feed_wall
            if _feed_wall > self._feed_wall_max:
                self._feed_wall_max = _feed_wall
            self.fed_seconds += pcm.size / 16000.0
            self._feed_pts = pts + pcm.size / 16000.0
            fed += 1
        if fed:
            self._prune_anchors()
        return fed

    # ────────────────────────────────────────────────────────── tiếp nhận mảnh
    async def handle_audio_fragment(
        self,
        chunk_bytes: bytes,
        timestamp_offset: float,
        mime_type: str,
        seek_id: str,
        is_init: bool = False,
        epoch: Optional[int] = None,
        refetched: bool = False,
    ) -> None:
        """Nhận một mảnh MSE, ghép nối + giải mã rồi đưa vào timeline liên tục.

        `refetched=True`: mảnh do extension TẢI LẠI cho vùng đã buffer sẵn. Byte của nó có thể
        trùng mảnh đã nhận trước đó nên phải **miễn dedup** — nếu không, vùng video đã tải trước
        sẽ không bao giờ có phụ đề (đo thật 2026-10-01: "Bỏ mảnh audio TRÙNG … 51 … 53" ngay
        sau khi backend yêu cầu client gửi lại).
        """
        if self._closed:
            return
        if seek_id and seek_id != self.active_seek_id:
            return

        self.fragments_received += 1
        self.bytes_in += len(chunk_bytes)
        #: Chẩn đoán tua: trong 3 giây đầu sau mỗi lần reset, ghi rõ mảnh đầu tiên nhận được
        #: (mốc media, `timestampOffset`, cờ tải lại) để biết vì sao audio có thể vẫn lệch.
        if (time.perf_counter() - self._seek_reset_at) < 3.0 and self._seek_diag_left > 0:
            self._seek_diag_left -= 1
            logger.warning(
                f"CHẨN ĐOÁN TUA: mảnh {len(chunk_bytes)}B, ts_offset={timestamp_offset:.2f}, "
                f"refetched={refetched}, epoch={epoch}, playhead={self.current_time:.2f}s, "
                f"buffer={self.demuxer.buffered_bytes}B",
                extra={"module_tag": "WS"},
            )

        # ── CHỐNG MẢNH TRÙNG (đo thật 2026-10-01) ────────────────────────────────
        # Extension có thể gửi cùng một mảnh nhiều lần (replay cache qua nhiều đường gọi
        # `REQUEST_INITIAL_CHUNKS`, cộng thêm `_pendingChunks`). Đo được: 15.851 frame AAC
        # (368 s audio) chỉ trải trên 70 s timeline ⇒ decoder phải giải mã rồi lọc trùng gấp
        # ~5 lần, đủ để làm đói cả pipeline.
        #
        # Chỉ bỏ mảnh trùng CÒN NÓNG (vừa gửi lại trong vài giây): mảnh giống hệt đến sau một
        # khoảng dài là dữ liệu hợp lệ (trình phát nạp lại cùng phân đoạn sau khi tua).
        now_perf = time.perf_counter()
        digest = hashlib.blake2b(chunk_bytes, digest_size=16).digest()
        last_seen = self._fragment_digests.get(digest)
        if (not refetched) and last_seen is not None and (now_perf - last_seen) <= _DUP_WINDOW_SEC:
            self.fragments_deduped += 1
            if self.fragments_deduped % 50 == 1:
                logger.warning(
                    f"Bỏ mảnh audio TRÙNG (tổng đã bỏ: {self.fragments_deduped}) — extension "
                    f"đang gửi lại cùng dữ liệu; xem lại đường replay cache.",
                    extra={"module_tag": "WS"},
                )
            return
        self._fragment_digests[digest] = now_perf
        self._fragment_digest_order.append(digest)
        while len(self._fragment_digest_order) > 512:
            self._fragment_digests.pop(self._fragment_digest_order.popleft(), None)

        self._playhead_pts = float(self.current_time or 0.0)
        t_dec = time.perf_counter()
        try:
            decoded = await asyncio.to_thread(
                self.demuxer.feed, chunk_bytes, timestamp_offset, mime_type, is_init, epoch
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Lỗi ghép nối/giải mã mảnh audio: {exc}", extra={"module_tag": "WS"})
            return
        self.decode_wall_sec += time.perf_counter() - t_dec

        # Lọc PCM nằm quá xa phía trước vị trí phát (xem `lookahead.max_decode_lead_sec`).
        # Không lọc thì timeline phải lấp hàng trăm giây im lặng để tới đoạn audio đó, và phụ
        # đề sinh ra sẽ mang mốc ở tương lai ⇒ không bao giờ hiện.
        dropped_ahead = 0
        media_sec = 0.0
        first_pts_here: Optional[float] = None
        for chunk in decoded:
            media_sec += float(chunk.duration)
            if first_pts_here is None:
                first_pts_here = float(chunk.pts_start)
            if chunk.pts_start > self._playhead_pts + self._max_decode_lead_sec:
                dropped_ahead += 1
                continue
            self.timeline.append(chunk.pts_start, chunk.pcm)
            self.chunks_decoded += 1
            self.decode_seconds += float(chunk.duration)
            self._decoded_end_pts = max(self._decoded_end_pts, float(chunk.pts_end))
        #: Giây MEDIA (audio thật) đã giải mã ra — khác `bytes_in` (byte thô, gồm cả mảnh
        #: của SourceBuffer khác lọt vào). Tỷ lệ `decode/media` mới là thước đo độ phủ.
        self.media_seconds += media_sec
        self.decode_delta_sec += media_sec

        if dropped_ahead:
            self.chunks_dropped_far += dropped_ahead
            logger.warning(
                f"Bỏ {dropped_ahead} đoạn PCM nằm >{self._max_decode_lead_sec:.0f}s trước vị trí "
                f"phát ({self._playhead_pts:.1f}s) — cache mảnh của interceptor đang lệch khỏi "
                f"vị trí phát; pipeline sẽ nhận mảnh mới đúng vị trí. "
                f"(tổng đã bỏ: {self.chunks_dropped_far})",
                extra={"module_tag": "ASR"},
            )
            # Yêu cầu client gửi lại mảnh quanh vị trí phát. Đây là cách chữa đúng cho ca
            # "seek tới 161,94 s nhưng cache chỉ có mảnh ở 372 s" (đo thật 2026-10-01).
            await self._request_replay("audio-far-ahead")
        elif (
            first_pts_here is not None
            and first_pts_here - self._playhead_pts > _START_OFFSET_WARN_SEC
            and self.start_offset_warned < 1
        ):
            self.start_offset_warned += 1
            logger.warning(
                f"Audio nhận được bắt đầu ở {first_pts_here:.1f}s trong khi vị trí phát là "
                f"{self._playhead_pts:.1f}s (lệch {first_pts_here - self._playhead_pts:.1f}s) — "
                f"đoạn đầu video sẽ KHÔNG có phụ đề. Nguyên nhân: cache mảnh phía Extension "
                f"chưa bám vị trí phát (nạp lại Extension rồi tải lại trang).",
                extra={"module_tag": "ASR"},
            )
            await self._request_replay("audio-offset")

        if decoded:
            if not dropped_ahead:
                # Hết "đói audio": tổng kết khoảng thời gian vừa qua không có PCM mới trong khi
                # byte VẪN về (triệu chứng "không lấy đủ buffer để dịch sẵn").
                if self._starved_since is not None:
                    starved = time.perf_counter() - self._starved_since
                    if starved > float(config.lookahead.decode_starve_warn_sec):
                        self._starved_max_sec = max(self._starved_max_sec, starved)
                        logger.warning(
                            f"Giải mã ĐÓI AUDIO: {starved:.1f}s không sinh ra PCM mới trong khi "
                            f"vẫn nhận {self._starved_bytes} B audio — nút cổ chai ở tầng GIẢI MÃ "
                            f"(demuxer), không phải ở lookahead.",
                            extra={"module_tag": "ASR"},
                        )
                    self._starved_since = None
                    self._starved_bytes = 0
            self._last_decoded_at = time.perf_counter()
            self.decode_windows += 1
            logger.info(
                f"Giải mã liên tục: +{media_sec:.2f}s "
                f"[{decoded[0].pts_start:.2f}s -> {decoded[-1].pts_end:.2f}s] "
                f"(mảnh {len(chunk_bytes)}B, tổng {self.chunks_decoded} đoạn, "
                f"buffer {self.demuxer.buffered_bytes} B, "
                f"bỏ_xa={dropped_ahead}, bytes_in={self.bytes_in})",
                extra={"module_tag": "ASR"},
            )
        else:
            if self._starved_since is None:
                self._starved_since = time.perf_counter()
            self._starved_bytes += len(chunk_bytes)

        self._ingest_event.set()

    # ────────────────────────────────────────────────────────── xử lý tua video
    async def handle_seek(self, seek_id: str, target_time: float) -> None:
        """Tua video: xoá sạch audio/commit cũ và neo lại mốc thời gian.

        ⚠️ Chống VÒNG LẶP RESET: `sync_state` được gửi 2 lần/giây và mang `seek_id` mới trong
        lúc phát; nếu mỗi lần lại xoá sạch pipeline thì phụ đề "lúc hiện lúc không". Chỉ coi là
        tua thật khi vị trí mới KHÁC ĐÁNG KỂ so với mốc neo hiện tại.
        """
        ref = self._session_start_pts if self._session_start_pts is not None else self.current_time
        if ref is not None and abs(float(target_time) - float(ref)) <= _REANCHOR_TOLERANCE_SEC:
            # Cùng một mốc (hoặc xê dịch trong sai số) ⇒ chỉ cập nhật vị trí phát và NHẬN
            # seek_id mới, KHÔNG reset pipeline. Vẫn phải nhận seek_id mới, nếu không các mảnh
            # gửi kèm seek_id đó sẽ bị bộ lọc "mảnh của seek cũ" bỏ sạch.
            self.active_seek_id = seek_id
            self.current_time = float(target_time)
            self._ingest_event.set()
            return

        self.active_seek_id = seek_id
        self.current_time = float(target_time)
        self._seek_seq += 1

        engine = self.asr_engine
        if engine is not None:
            try:
                await engine.reset_stream("lookahead_seek")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"reset ASR stream lỗi: {exc}", extra={"module_tag": "WS"})
        if self.vad_processor is not None:
            try:
                self.vad_processor.reset()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"reset VAD lỗi: {exc}", extra={"module_tag": "WS"})

        # Giữ nguyên epoch và init segment: cùng một SourceBuffer thì header không đổi, chỉ
        # bỏ phần media đã ghép (dữ liệu của vị trí cũ).
        self.demuxer.reset(self.demuxer.epoch, min_pts=max(0.0, float(target_time) - 0.5))
        self.timeline.reset(float(target_time))
        with self._anchor_lock:
            self._anchor_samples.clear()
            self._anchor_pts.clear()
        self._feed_pts = None
        self._session_start_pts = float(target_time)
        self._decoded_end_pts = float(target_time)
        self._ready_until_pts = float(target_time)
        self._sent_items.clear()
        if self.tts_queue is not None:
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except Exception:
                    break
        self.tts_sent = 0
        self._recent_utterances.clear()

        # Reset mốc chẩn đoán năng lực: nếu không, log sau khi tua sẽ so với số của đoạn cũ
        # và báo sai tốc độ giải mã/nạp.
        self._last_diag_at = time.perf_counter()
        self._diag_bytes_in = self.bytes_in
        self._diag_decode_seconds = self.decode_seconds
        self._diag_media_seconds = self.media_seconds
        self._diag_decode_wall = self.decode_wall_sec
        self._diag_fed_seconds = self.fed_seconds
        self._diag_feed_wall = self.feed_wall_sec
        self._starved_since = None
        self._starved_bytes = 0
        # Bật chẩn đoán cho ~8 mảnh đầu sau khi tua.
        self._seek_reset_at = time.perf_counter()
        self._seek_diag_left = 8
        metrics_collector.increment_counter("lookahead.seek_reset")
        logger.info(
            f"Seek reset seek_id={seek_id} target_time={target_time:.2f}s "
            f"(đã xoá toàn bộ audio/commit/TTS; bộ đệm byte={self.demuxer.buffered_bytes}B, "
            f"min_pts={self.demuxer.last_pts:.2f}s)",
            extra={"module_tag": "WS"},
        )

    # ────────────────────────────────────────────────────────── tiêu thụ ASR
    async def _asr_loop(self) -> None:
        engine = self.asr_engine
        if engine is None:
            return
        try:
            async for msg in engine.stream_tokens():
                try:
                    await self._handle_asr_message(msg)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Lỗi xử lý kết quả ASR Lookahead: {exc}", exc_info=True,
                                   extra={"module_tag": "WS"})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Vòng ASR Lookahead dừng: {exc}", exc_info=True, extra={"module_tag": "ASR"})

    async def _handle_asr_message(self, msg: Dict[str, Any]) -> None:
        if not isinstance(msg, dict) or msg.get("type") != "utterance_update":
            return
        if not msg.get("is_final"):
            return  # Lookahead KHÔNG hiện chữ chạy — chỉ hiện câu đã chốt.

        text = (msg.get("text") or "").strip()
        if not text:
            return

        min_words = int(self._min_words_to_commit)
        if count_content_tokens(text) < min_words:
            logger.info(
                f"Lọc bỏ câu quá ngắn (<{min_words} từ): '{text}'", extra={"module_tag": "WS"}
            )
            return

        pts_start = self._pts_at(msg.get("start_sample"))
        pts_end = self._pts_at(msg.get("end_sample"))
        if pts_start is None or pts_end is None or pts_end <= pts_start:
            logger.warning(
                f"Không ánh xạ được mốc thời gian cho câu '{text}' "
                f"(samples {msg.get('start_sample')}->{msg.get('end_sample')}) — bỏ qua.",
                extra={"module_tag": "WS"},
            )
            return

        # ── CANH LẠI MỐC BẮT ĐẦU ──────────────────────────────────────────────
        # VAD báo mép đoạn nói SỚM hơn thực tế (nó lùi lại để lấy ngữ cảnh: FireRed
        # `pad_start_frame`, Silero `speech_pad_ms`, FSMN `lookback_time_start_point`).
        # Nếu dùng thẳng mốc đó thì phụ đề hiện TRƯỚC tiếng nói và lồng tiếng cũng đọc
        # trước. Cộng bù đúng phần đã lùi, kèm offset người dùng chỉnh trong popup.
        pad_sec = max(0.0, self._start_pad_ms) / 1000.0
        user_sec = float(self.sync_offset_ms) / 1000.0
        shift = pad_sec + user_sec
        if shift:
            new_start = pts_start + shift
            # Không để mốc bắt đầu vượt quá mốc kết thúc (câu cực ngắn).
            if new_start < pts_end - 0.15:
                pts_start = new_start
            else:
                pts_start = max(pts_start, pts_end - 0.15)
        if pts_start < 0:
            pts_start = 0.0

        logger.info(
            f"Lookahead ASR ({pts_start:.2f}s-{pts_end:.2f}s): '{text}'", extra={"module_tag": "ASR"}
        )

        seq = self._seek_seq
        translated = await self._translate(text)
        if translated is None:
            return

        # Nếu video bị tua trong lúc dịch thì kết quả này thuộc đoạn cũ ⇒ bỏ.
        if seq != self._seek_seq:
            logger.info(
                f"Bỏ bản dịch thuộc đoạn cũ (đã tua video): '{text}'", extra={"module_tag": "WS"}
            )
            return

        if self._is_duplicate(pts_start, pts_end, text):
            return

        self._sent_items.append((pts_start, pts_end, text))
        if len(self._sent_items) > 512:
            del self._sent_items[:128]
        self._recent_utterances.append({
            "pts_start": pts_start,
            "pts_end": pts_end,
            "text": text,
            "translated": translated,
        })
        if len(self._recent_utterances) > 128:
            del self._recent_utterances[:32]
        self._ready_until_pts = max(self._ready_until_pts, pts_end)
        self.utterances_sent += 1

        await self.send_json({
            "type": "lookahead_subtitles",
            "seek_id": self.active_seek_id,
            "items": [{
                "start_pts": round(pts_start, 3),
                "end_pts": round(pts_end, 3),
                "original_text": text,
                "translated_text": translated,
            }],
        })

        self._enqueue_tts(translated, pts_start, pts_end)

    # ────────────────────────────────────────────────────────── lồng tiếng (TTS)
    def _enqueue_tts(self, text: str, pts_start: float, pts_end: float) -> None:
        """Xếp câu dịch vào hàng đợi lồng tiếng (nếu popup bật TTS)."""
        if not self.tts_enabled or not self.tts_queue or not text:
            return
        try:
            self.tts_queue.put_nowait({
                "text": text,
                "start_pts": float(pts_start),
                "end_pts": float(pts_end),
                "queued_at": time.perf_counter(),
            })
        except asyncio.QueueFull:
            self.tts_skipped += 1
            metrics_collector.increment_counter("tts.lookahead_queue_full")
            logger.warning("Hàng đợi TTS Lookahead đầy — bỏ một câu lồng tiếng.", extra={"module_tag": "TTS"})
        self._trim_tts_queue()

    def _trim_tts_queue(self) -> None:
        """Giữ hàng đợi TTS trong trần: bỏ câu cũ nhất trong hàng đợi khi vượt trần.

        Vì sao: khi GPU quá tải, một câu có thể mất hàng chục giây để tổng hợp; hàng đợi phình
        ra thì vừa tốn VRAM vừa khiến mọi câu đều trễ. Bỏ câu cũ là đánh đổi đúng.
        """
        limit = max(1, int(config.lookahead.tts_max_queue))
        if self.tts_queue is None:
            return
        dropped = 0
        while self.tts_queue.qsize() > limit:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
                dropped += 1
            except Exception:  # noqa: BLE001
                break
        if dropped:
            self.tts_skipped += dropped
            metrics_collector.increment_counter("tts.lookahead_queue_trimmed", dropped)

    async def _yield_gpu_to_asr(self, max_wait_sec: float) -> None:
        """Chờ (có trần) cho tới khi ASR rảnh trước khi chiếm GPU cho TTS."""
        engine = self.asr_engine
        if engine is None or max_wait_sec <= 0:
            return
        deadline = time.perf_counter() + max_wait_sec
        while not self._closed and time.perf_counter() < deadline:
            try:
                if not engine.has_pending_work():
                    return
            except Exception:  # noqa: BLE001
                return
            await asyncio.sleep(0.1)

    async def _tts_loop(self) -> None:
        """Worker TTS tuần tự: tổng hợp VỪA ngân sách thời gian rồi gửi kèm mốc phát.

        ⚠️ TTS là BEST-EFFORT. Nó KHÔNG được phép làm hỏng đường phụ đề:
        * bỏ câu khi đã trôi qua vị trí phát,
        * bỏ câu khi hàng đợi quá dài (giữ câu mới nhất),
        * nhường GPU cho ASR trước mỗi câu,
        * **không chạy khi VRAM gần cạn** — thà mất tiếng lồng còn hơn tràn VRAM khiến ASR
          không chạy được (mất cả phụ đề, đúng sự cố đã gặp với TTS synth 30 s).
        """
        tts = None
        la = config.lookahead
        slow_warned_at = 0.0
        vram_warned_at = 0.0
        while not self._closed:
            if self.tts_queue is None:
                return
            try:
                item = await self.tts_queue.get()
            except asyncio.CancelledError:
                raise
            try:
                if self._closed or not self.tts_enabled:
                    continue

                start_pts = float(item["start_pts"])
                end_pts = float(item["end_pts"])
                # 1. Câu đã trôi qua từ lâu (video đã chạy qua) ⇒ tổng hợp cũng vô ích.
                if end_pts < self.current_time - 0.5:
                    self.tts_skipped += 1
                    metrics_collector.increment_counter("tts.lookahead_stale_skipped")
                    continue

                # 2. Hàng đợi quá dài ⇒ bỏ bớt câu CŨ NHẤT (chúng trôi xa nhất).
                self._trim_tts_queue()

                # 3. NHƯỜNG GPU CHO ASR: phụ đề là đường chính, TTS chỉ là phụ.
                # Khi chưa gửi được câu TTS nào (đang nạp đệm câu đầu): ưu tiên tổng hợp ngay!
                if self.tts_sent > 0:
                    await self._yield_gpu_to_asr(float(la.tts_asr_yield_max_sec))

                # 4. VRAM gần cạn ⇒ BỎ câu lồng tiếng (chống tràn VRAM).
                free_mb = _free_vram_mb()
                if free_mb is not None and free_mb < float(la.tts_min_free_vram_mb):
                    _empty_torch_cache()
                    free_mb = _free_vram_mb()
                if free_mb is not None and free_mb < float(la.tts_min_free_vram_mb):
                    self.tts_skipped += 1
                    metrics_collector.increment_counter("tts.lookahead_vram_skipped")
                    now = time.time()
                    if now - vram_warned_at > 10.0:
                        vram_warned_at = now
                        logger.warning(
                            f"VRAM trống chỉ {free_mb:.0f} MB (< {la.tts_min_free_vram_mb:.0f} MB) — "
                            f"BỎ lồng tiếng để giữ phụ đề. Hãy dùng model dịch/ASR nhẹ hơn hoặc tắt TTS.",
                            extra={"module_tag": "TTS"},
                        )
                    continue

                # ── BƯỚC 1: XÁC ĐỊNH THỜI GIAN PHỤ ĐỀ HIỂN THỊ TRÊN MÀN HÌNH ──────────
                raw_window = max(0.35, end_pts - start_pts)

                # Tra cứu mốc bắt đầu của câu kế tiếp (từ các câu đã chốt hoặc đang xếp hàng)
                next_start_candidates = [
                    s[0] for s in self._sent_items if s[0] > start_pts + 0.05
                ]
                if self.tts_queue is not None:
                    for queued in getattr(self.tts_queue, "_queue", []):
                        q_start = float(queued.get("start_pts", 0.0))
                        if q_start > start_pts + 0.05:
                            next_start_candidates.append(q_start)

                next_start_pts = min(next_start_candidates) if next_start_candidates else None

                # Thời gian phụ đề hiển thị trên màn hình:
                # - Nếu có câu sau: tối đa hiển thị tới khi câu sau bắt đầu
                # - Nếu không có câu sau: hiển thị trọn vẹn raw_window
                if next_start_pts is not None:
                    gap_to_next = max(0.35, next_start_pts - start_pts)
                    display_window = min(raw_window, gap_to_next)
                else:
                    display_window = raw_window

                # ── BƯỚC 2: ĐỒNG BỘ ÂM THANH TTS BẰNG HOẶC NHỎ HƠN 1 CHÚT ──────────────
                # Ngân sách TTS đặt bằng 92% display_window (trừ thêm 60ms an toàn)
                # Đảm bảo TTS LUÔN đọc xong 100% trước khi phụ đề biến mất hoặc câu sau bắt đầu!
                target_budget = max(0.30, display_window * 0.92 - 0.06)
                budget = min(float(la.tts_max_budget_sec), target_budget)

                if tts is None:
                    from backend.tts import get_tts_engine
                    tts = get_tts_engine()

                text = _clip_dubbing_text(item["text"], int(la.tts_max_chars))
                if not text:
                    self.tts_skipped += 1
                    continue

                t0 = time.perf_counter()
                wav_bytes, duration = await tts.synthesize_fitted_bytes(
                    text=text,
                    voice_id=self.tts_voice or None,
                    speed=self.tts_speed,
                    max_duration_sec=budget,
                    max_speed=float(la.tts_max_speed),
                )
                synth_ms = (time.perf_counter() - t0) * 1000.0
                if not wav_bytes:
                    self.tts_skipped += 1
                    continue

                if synth_ms >= float(la.tts_slow_warn_ms):
                    now = time.time()
                    if now - slow_warned_at > 10.0:
                        slow_warned_at = now
                        logger.warning(
                            f"Tổng hợp lồng tiếng quá chậm ({synth_ms:.0f} ms cho "
                            f"{duration:.2f}s audio) — GPU đang quá tải/thiếu VRAM. "
                            f"Cân nhắc model nhẹ hơn hoặc tắt TTS.",
                            extra={"module_tag": "TTS"},
                        )

                # Mốc kết thúc gửi kèm cho client: phụ đề hiển thị ít nhất bằng thời lượng audio
                effective_end_pts = max(end_pts, start_pts + float(duration) + 0.10)
                if next_start_pts is not None:
                    effective_end_pts = min(effective_end_pts, next_start_pts - 0.02)
                effective_end_pts = max(effective_end_pts, start_pts + 0.35)

                frame = _pack_binary_frame(
                    {
                        "type": "lookahead_tts",
                        "seek_id": self.active_seek_id,
                        "start_pts": round(start_pts, 3),
                        "end_pts": round(effective_end_pts, 3),
                        "duration_sec": round(float(duration), 3),
                        "sample_rate": int(getattr(tts, "sample_rate", 24000) or 24000),
                        "window_sec": round(display_window, 3),
                        "budget_sec": round(budget, 3),
                    },
                    wav_bytes,
                )
                sent = await self.connection.send_bytes(frame)
                if sent:
                    self.tts_sent += 1
                    metrics_collector.record_metric("tts", "lookahead_synth_ms", synth_ms)
                    logger.info(
                        f"TTS Lookahead [{start_pts:.2f}s-{end_pts:.2f}s]: đọc {duration:.2f}s / "
                        f"cửa sổ {display_window:.2f}s (synth {synth_ms:.0f}ms, {len(frame)}B)",
                        extra={"module_tag": "TTS"},
                    )
                    # Gửi kèm bản JSON base64: đường JSON là đường ĐÃ ĐƯỢC CHỨNG MINH chạy
                    # (phụ đề đi qua nó), còn khung nhị phân phụ thuộc `ws.binaryType` của
                    # từng trình duyệt. Client ưu tiên bản JSON và bỏ qua bản nhị phân nếu đã
                    # có (chống trùng theo `start_pts`).
                    await self.send_json({
                        "type": "lookahead_tts",
                        "seek_id": self.active_seek_id,
                        "start_pts": round(start_pts, 3),
                        "end_pts": round(effective_end_pts, 3),
                        "duration_sec": round(float(duration), 3),
                        "sample_rate": int(getattr(tts, "sample_rate", 24000) or 24000),
                        "window_sec": round(display_window, 3),
                        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    })
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Lỗi lồng tiếng Lookahead: {exc}", exc_info=True, extra={"module_tag": "TTS"})
            finally:
                try:
                    self.tts_queue.task_done()
                except Exception:  # noqa: BLE001
                    pass

    async def _translate(self, text: str) -> Optional[str]:
        """Dịch câu gốc sang ngôn ngữ đích. None = không gửi (lỗi)."""
        engine = self.translation_engine
        if engine is None:
            return text
        try:
            result = await engine.translate(text, self.source_lang, self.target_lang)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Lỗi dịch Lookahead: {exc}", extra={"module_tag": "TRANSLATE"})
            return text
        if isinstance(result, str):
            return result.strip() or text
        if isinstance(result, dict):
            return (result.get("translated_text") or result.get("text") or text).strip()
        return text

    def _is_duplicate(self, pts_start: float, pts_end: float, text: str) -> bool:
        for s0, s1, prev in self._sent_items[-64:]:
            if prev == text and abs(s0 - pts_start) < 0.35:
                return True
            if abs(s0 - pts_start) < 0.25 and abs(s1 - pts_end) < 0.25:
                return True
        return False

    # ────────────────────────────────────────────────────────── báo trạng thái
    async def _status_loop(self) -> None:
        interval = max(0.1, float(config.lookahead.status_interval_ms) / 1000.0)
        while not self._closed:
            try:
                await asyncio.sleep(interval)
                await self.send_status()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Gửi lookahead_status lỗi: {exc}", extra={"module_tag": "WS"})

    async def send_status(self) -> None:
        la = config.lookahead
        pending_end = self.timeline.buffered_end_pts()
        buffered_end = max(pending_end or 0.0, self._decoded_end_pts)
        buffered_ahead = max(0.0, buffered_end - self.current_time)
        ready_ahead = max(0.0, self._ready_until_pts - self.current_time)

        # Mục tiêu đệm: đủ `lead_time`, nhưng nếu buffer của trình duyệt ngắn hơn thì chỉ cần
        # phủ hết phần đã có (trừ 1 s an toàn) để KHÔNG bao giờ treo trình phát.
        if buffered_ahead > 0:
            target = min(float(self.lead_time), max(float(la.min_ready_ahead_sec), buffered_ahead - 1.0))
        else:
            target = float(la.min_ready_ahead_sec)

        # Sẵn sàng phát khi EITHER:
        #  - đã dịch trước đủ `target` giây, HOẶC
        #  - đã xử lý (VAD+ASR) đủ `target` giây phía trước nhưng vùng đó KHÔNG có tiếng nói
        #    (im lặng/nhạc nền) ⇒ không có phụ đề nào để chờ. Nếu chỉ xét phụ đề thì một
        #    đoạn im lặng dài sẽ khoá trình phát vô ích.
        fed_ahead = max(0.0, (self._feed_pts or self.current_time) - self.current_time)

        # Điều kiện TTS nếu TTS được bật và có hàng đợi xử lý TTS:
        # Nếu có câu nói trong phạm vi prebuffer (ready_ahead > 0), phải đợi ít nhất 1 câu TTS
        # được tổng hợp xong (self.tts_sent >= 1 hoặc tts_skipped >= 1) để câu đầu không bị phát hụt!
        tts_ready = True
        if self.tts_enabled and self.tts_queue is not None and ready_ahead > 0:
            tts_ready = bool(self.tts_sent >= 1 or self.tts_skipped >= 1)

        prebuffer_ready = (
            (bool(ready_ahead >= target) or bool(fed_ahead >= target and ready_ahead >= float(la.min_ready_ahead_sec)))
            and tts_ready
        )
        # Log khi TRẠNG THÁI SẴN SÀNG ĐỔI — đây là mốc quyết định "cho video phát hay chưa",
        # nếu không log thì rất khó chẩn đoán tại sao video cứ chờ hết timeout.
        if prebuffer_ready != self._last_ready_flag:
            self._last_ready_flag = prebuffer_ready
            logger.info(
                f"Lookahead prebuffer_ready={prebuffer_ready} "
                f"(ready_ahead={ready_ahead:.2f}s, fed_ahead={fed_ahead:.2f}s, target={target:.2f}s, "
                f"tts_ready={tts_ready}, tts_sent={self.tts_sent}, "
                f"buffered_ahead={buffered_ahead:.2f}s, currentTime={self.current_time:.2f}s)",
                extra={"module_tag": "WS"},
            )

        await self._maybe_log_diagnostics(buffered_ahead, ready_ahead, fed_ahead)

        await self.send_json({
            "type": "lookahead_status",
            "seek_id": self.active_seek_id,
            "currentTime": round(self.current_time, 2),
            "buffered_ahead": round(buffered_ahead, 2),
            "buffered_end_pts": round(buffered_end, 2),
            "ready_until_pts": round(self._ready_until_pts, 2),
            "ready_ahead": round(ready_ahead, 2),
            "fed_ahead": round(fed_ahead, 2),
            "target_ahead": round(target, 2),
            "prebuffer_ready": bool(prebuffer_ready),
            "tts_ready": bool(tts_ready),
            "tts_sent": self.tts_sent,
            "fragments": self.fragments_received,
            "decoded_chunks": self.chunks_decoded,
            "utterances": self.utterances_sent,
        })

    async def _maybe_log_diagnostics(
        self, buffered_ahead: float, ready_ahead: float, fed_ahead: float
    ) -> None:
        """Log NĂNG LỰC THẬT mỗi `diag_interval_sec`: byte vào, PCM ra, tốc độ nạp VAD.

        Đây là thước đo để biết nút cổ chai nằm ở ĐÂU:
          * ``PCM/byte`` thấp (byte về nhiều mà PCM ra ít)  ⇒ tầng giải mã (demuxer).
          * ``nạp/VAD`` thấp (PCM có sẵn nhưng `feed_chunk` chậm) ⇒ tầng VAD/ASR.
        """
        interval = max(1.0, float(getattr(config.lookahead, "diag_interval_sec", 10.0) or 10.0))
        now = time.perf_counter()
        if now - self._last_diag_at < interval:
            return
        span = max(0.001, now - self._last_diag_at)
        self._last_diag_at = now

        d_bytes = self.bytes_in - self._diag_bytes_in
        d_decode = self.decode_seconds - self._diag_decode_seconds
        d_media = self.media_seconds - self._diag_media_seconds
        d_decode_wall = self.decode_wall_sec - self._diag_decode_wall
        d_fed = self.fed_seconds - self._diag_fed_seconds
        d_feed_wall = self.feed_wall_sec - self._diag_feed_wall
        self._diag_bytes_in = self.bytes_in
        self._diag_decode_seconds = self.decode_seconds
        self._diag_media_seconds = self.media_seconds
        self._diag_decode_wall = self.decode_wall_sec
        self._diag_fed_seconds = self.fed_seconds
        self._diag_feed_wall = self.feed_wall_sec

        logger.info(
            f"CHẨN ĐOÁN NĂNG LỰC {span:.1f}s | audio vào {d_bytes / 1024.0:.0f} KB "
            f"({d_bytes / span / 1000.0:.1f} KB/s) → media giải mã {d_media:.1f}s "
            f"(nạp vào timeline {d_decode:.1f}s, {self.decode_windows} lượt có PCM) | "
            f"nạp VAD {d_fed:.1f}s ({d_fed / span:.2f}x thời gian thực, wall {d_feed_wall:.1f}s, "
            f"max {self._feed_wall_max * 1000.0:.0f}ms/khối) | "
            f"đệm GIẢI MÃ {self._decoded_end_pts - self.current_time:.1f}s trước vị trí phát, "
            f"đệm ĐÃ DỊCH {ready_ahead:.1f}s (fed {fed_ahead:.1f}s), "
            f"bỏ_xa={self.chunks_dropped_far}, mảnh_trùng={self.fragments_deduped} | "
            f"đói giải mã dài nhất {self._starved_max_sec:.1f}s | "
            f"demuxer: {self.demuxer.coverage_report()}",
            extra={"module_tag": "ASR"},
        )

        # Cấu trúc container chỉ cần in thưa (1 lần/phút): nó giải mã lại toàn bộ bộ đệm byte.
        if now - self._last_struct_at >= 60.0:
            self._last_struct_at = now
            logger.info(
                f"CẤU TRÚC NGUỒN: {self.demuxer.structure_report()}",
                extra={"module_tag": "ASR"},
            )

    # ────────────────────────────────────────────────────────── cấu hình phiên
    def apply_init(self, data: Dict[str, Any]) -> None:
        la = config.lookahead
        self.source_lang = str(data.get("source_lang") or self.source_lang or "auto")
        self.target_lang = str(data.get("target_lang") or self.target_lang or "vi")
        try:
            lead = float(data.get("lead_time", la.lead_time_sec))
        except (TypeError, ValueError):
            lead = float(la.lead_time_sec)
        self.lead_time = max(float(la.min_lead_time_sec), min(float(la.max_lead_time_sec), lead))
        if self.asr_engine is not None:
            self.asr_engine.set_language(self.source_lang)

        # Cấu hình ĐẦY ĐỦ từ popup (VAD engine/threshold/silence, phân câu, model ASR/dịch…).
        # `buildWsConfig()` phía Extension gửi kèm trong `lookahead_init`.
        parsed = self._parse_config(data)
        if parsed is not None:
            self._parsed_config = parsed
            if parsed.source_lang:
                self.source_lang = str(parsed.source_lang)
            if parsed.target_lang:
                self.target_lang = str(parsed.target_lang)
            requested = parsed.asr_model or parsed.asr_engine or parsed.model_id
            if requested:
                self._requested_asr_model = str(requested).strip().lower()
            if parsed.min_words_to_commit is not None:
                self._min_words_to_commit = max(0, int(parsed.min_words_to_commit))

        # Offset canh đồng bộ (ms) do người dùng chỉnh trong popup — trường riêng của
        # Lookahead nên không nằm trong `SessionConfigPayload`.
        sync_raw = data.get("lookaheadSyncOffsetMs", data.get("lookahead_sync_offset_ms"))
        if sync_raw is not None:
            try:
                self.sync_offset_ms = max(-1500.0, min(1500.0, float(sync_raw)))
            except (TypeError, ValueError):
                pass

        # Vị trí phát hiện tại của video khi bắt đầu phiên
        cur_time_raw = data.get("current_time", data.get("currentTime"))
        if cur_time_raw is not None:
            try:
                cur_time = float(cur_time_raw)
            except (TypeError, ValueError):
                cur_time = 0.0
            if cur_time > 0:
                self.current_time = cur_time
                self._session_start_pts = cur_time
                self._decoded_end_pts = cur_time
                self._ready_until_pts = cur_time
                self.timeline.reset(cur_time)
                self.demuxer.set_min_pts(max(0.0, cur_time - 0.5))
                logger.info(
                    f"Lookahead init: neo vị trí phát hiện tại @{cur_time:.2f}s",
                    extra={"module_tag": "WS"},
                )
        seek_id = data.get("seek_id")
        if seek_id:
            self.active_seek_id = str(seek_id)


async def handle_lookahead_ws(websocket: WebSocket) -> None:
    """Entry point WebSocket `/ws/lookahead` (Pipeline B)."""
    await websocket.accept()
    safe_conn = SafeWebSocketConnection(websocket)

    asr_eng = None
    trans_eng = None
    unavailable = ""
    try:
        from backend.asr.engine import TranscribeEngine
        asr_eng = TranscribeEngine(session_id=f"lookahead_{uuid.uuid4().hex[:8]}")
    except Exception as exc:  # noqa: BLE001
        unavailable = f"Không khởi tạo được ASR: {exc}"
        logger.warning(unavailable, extra={"module_tag": "WS"})
    try:
        from backend.translation.engine import get_translation_engine
        trans_eng = get_translation_engine()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Không lấy được TranslationEngine cho lookahead: {exc}", extra={"module_tag": "WS"})

    session = LookaheadSessionState(safe_conn, asr_engine=asr_eng, translation_engine=trans_eng)
    session.unavailable_reason = unavailable

    logger.info(
        f"Khởi tạo phiên lookahead mới: {session.session_id}"
        + (f" | CẢNH BÁO: {unavailable}" if unavailable else ""),
        extra={"module_tag": "WS"},
    )

    # ASR native là SINGLETON (một stream in-flight cho mỗi model). Cảnh báo rõ nếu đang có
    # phiên realtime khác chạy song song để người vận hành biết vì sao kết quả có thể lẫn.
    try:
        from backend.ws.handler import count_active_sessions

        active = count_active_sessions()
        if active > 0:
            logger.warning(
                f"Đang có {active} phiên /ws realtime hoạt động song song với Pipeline B — "
                f"ASR native là singleton, nên chỉ nên chạy Lookahead HOẶC realtime cho mỗi model.",
                extra={"module_tag": "WS"},
            )
    except Exception:  # noqa: BLE001
        pass

    started = False
    try:
        while True:
            message = await safe_conn.receive()
            if not message:
                break
            if message.get("type") == "websocket.disconnect":
                break

            text_data = message.get("text")
            bytes_data = message.get("bytes")

            # Nếu socket đã chết (một lần send thất bại ⇒ `is_closed`) thì KHÔNG xử lý thêm
            # audio nào nữa — tránh việc server vẫn giải mã/ASR sau khi client đã Stop.
            if safe_conn.is_closed:
                logger.info(
                    f"Socket lookahead đã đóng — dừng xử lý (session {session.session_id[:8]}).",
                    extra={"module_tag": "WS"},
                )
                break

            if text_data:
                try:
                    data = json.loads(text_data)
                except Exception:
                    continue
                req_type = data.get("type", "")

                if req_type == "lookahead_init":
                    session.apply_init(data)
                    if not started and not session.unavailable_reason:
                        try:
                            session.init_components(data.get("vad_engine"))
                            await session.start_tasks()
                            started = True
                        except Exception as exc:  # noqa: BLE001
                            session.unavailable_reason = f"Không dựng được pipeline: {exc}"
                            logger.error(session.unavailable_reason, exc_info=True, extra={"module_tag": "WS"})
                    logger.info(
                        f"Lookahead init: source={session.source_lang}, target={session.target_lang}, "
                        f"lead_time={session.lead_time}s",
                        extra={"module_tag": "WS"},
                    )
                    if session.unavailable_reason:
                        await session.send_json({
                            "type": "lookahead_unavailable",
                            "reason": session.unavailable_reason,
                        })
                    else:
                        await session.send_json({
                            "type": "lookahead_ready",
                            "session_id": session.session_id,
                            "lead_time": session.lead_time,
                            "status": "ready",
                        })

                elif req_type == "sync_state":
                    new_time = float(data.get("currentTime", session.current_time))
                    session.playback_rate = float(data.get("playbackRate", 1.0) or 1.0)
                    session.is_paused = bool(data.get("paused", False))
                    seek_id = data.get("seek_id")
                    if seek_id and seek_id != session.active_seek_id:
                        await session.handle_seek(seek_id, new_time)
                    elif session._session_start_pts is None:
                        session.current_time = new_time
                        session._session_start_pts = new_time
                        if new_time > 0:
                            session.timeline.reset(new_time)
                            session.demuxer.set_min_pts(max(0.0, new_time - 0.5))
                    else:
                        # KHÔNG tự đoán "lệch timeline" rồi neo lại: mọi lần tua đều đi qua
                        # `handle_seek` (seek_id mới hoặc `seek_reset`) và reset TOÀN BỘ pipeline
                        # như phiên mới. Cơ chế tự đoán trước đây tạo vòng lặp reset mỗi 3 s vì
                        # cursor chưa kịp tiêu thụ audio nào sau mỗi lần reset.
                        session.current_time = new_time
                    session._ingest_event.set()

                elif req_type in ("seek_reset", "reset_stream"):
                    seek_id = data.get("seek_id") or str(uuid.uuid4())
                    target_time = float(data.get("target_time", data.get("currentTime", 0.0)) or 0.0)
                    await session.handle_seek(seek_id, target_time)
                    await session.send_json({
                        "type": "seek_acknowledged",
                        "seek_id": seek_id,
                        "target_time": target_time,
                    })

                elif req_type == "set_config":
                    await session.apply_config(data)

                elif req_type in ("stop", "bye", "close"):
                    # Lệnh DỪNG tường minh từ Extension (người dùng bấm Stop). Không chờ
                    # close-frame của socket — dừng ngay để server không xử lý thêm gì.
                    logger.info(
                        f"Nhận lệnh Stop từ client (reason={data.get('reason', '')}) — đóng phiên lookahead.",
                        extra={"module_tag": "WS"},
                    )
                    break

                elif req_type == "ping":
                    await session.send_json({"type": "pong", "timestamp": time.time()})

            elif bytes_data:
                await _handle_binary_fragment(session, bytes_data)

    except WebSocketDisconnect:
        logger.info(f"Client ngắt kết nối lookahead: {session.session_id}", extra={"module_tag": "WS"})
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Phiên lookahead kết thúc do lỗi ({exc})", exc_info=True, extra={"module_tag": "WS"})
    finally:
        try:
            await asyncio.wait_for(session.close(), timeout=12.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            logger.warning("Đóng phiên lookahead quá hạn — bỏ qua để không treo kết nối.",
                           extra={"module_tag": "WS"})
        # Đóng socket tường minh (giống `/ws`): server không giữ kết nối sau khi Stop.
        try:
            await safe_conn.close()
        except Exception:  # noqa: BLE001
            pass
        logger.info(f"Đã đóng và giải phóng phiên lookahead: {session.session_id[:8]}", extra={"module_tag": "WS"})


async def _handle_binary_fragment(session: LookaheadSessionState, bytes_data: bytes) -> None:
    """Phân tích khung nhị phân: [4B header_len][JSON header][bytes audio]."""
    if len(bytes_data) < 4:
        return
    header_len = struct.unpack("<I", bytes_data[:4])[0]
    if not (0 < header_len < 4096) or (4 + header_len) > len(bytes_data):
        logger.debug(f"Khung nhị phân không hợp lệ ({len(bytes_data)} B)", extra={"module_tag": "WS"})
        return
    try:
        header = json.loads(bytes_data[4:4 + header_len].decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Header JSON của khung audio không hợp lệ: {exc}", extra={"module_tag": "WS"})
        return

    chunk_bytes = bytes_data[4 + header_len:]
    epoch = header.get("epoch")
    await session.handle_audio_fragment(
        chunk_bytes=chunk_bytes,
        timestamp_offset=float(header.get("timestamp_offset", 0.0) or 0.0),
        mime_type=str(header.get("mime_type", "auto")),
        seek_id=str(header.get("seek_id") or session.active_seek_id),
        is_init=bool(header.get("is_init", False)),
        epoch=int(epoch) if epoch is not None else None,
        # Mảnh tải lại cho vùng đã buffer sẵn: byte có thể trùng mảnh cũ nên phải MIỄN dedup.
        refetched=bool(header.get("refetched", False)),
    )
