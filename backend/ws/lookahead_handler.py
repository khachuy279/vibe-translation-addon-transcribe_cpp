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
import hashlib
import json
import struct
import sys
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from backend.config import config
from backend.core.dedup import CommitDeduplicator, normalize_for_dedup
from backend.core.lookahead_chunker import LookaheadChunker
from backend.core.lookahead_timeline import ContinuousAudioTimeline
from backend.core.metrics import metrics_collector
from backend.core.stream_demuxer import StreamDemuxer
from backend.asr.forced_aligner import (
    AlignedWord,
    ForcedAlignerService,
    SubtitleSentence,
    detect_aligner_language_from_text,
    resolve_aligner_language,
)
from backend.core.vad_silence import VADSilenceScanner
from backend.ws.connection import SafeWebSocketConnection
from backend.ws.session import SessionConfigPayload
from backend.utils.logger import get_logger
from backend.utils.text_repetition import is_repetition_hallucination

logger = get_logger("ws.lookahead")

#: Cửa sổ (giây) coi một mảnh giống hệt là "gửi trùng" (mảnh nạp lại sau khi tua vẫn được nhận).
_DUP_WINDOW_SEC = 15.0
#: Lệch quá ngần này giây giữa audio nhận được và vị trí phát ⇒ cảnh báo "đoạn đầu sẽ không
#: có phụ đề" (dấu hiệu cache mảnh phía Extension không bám vị trí phát).
_START_OFFSET_WARN_SEC = 20.0
#: Xê dịch tối đa (giây) giữa mốc neo hiện tại và vị trí báo mới mà vẫn coi là CÙNG một mốc
#: (không reset pipeline). Chống reset lặp khi `sync_state` gửi `seek_id` mới ở cùng vị trí.
_REANCHOR_TOLERANCE_SEC = 1.5
#: Con trỏ khối batch phải kẹt xa vị trí phát BỀN VỮNG ngần này giây mới neo lại (xem
#: `_maybe_reanchor_batch_frontier`) — tránh neo nhầm khi playhead vừa nhảy bình thường.
_FRONTIER_RESYNC_AFTER_SEC = 2.0
#: Con trỏ khối bị coi là "kẹt PHÍA SAU vị trí phát" khi nó chậm hơn playhead ngần này giây
#: (đo thật 2026-10-05, xhamster.com: playhead 967s nhưng con trỏ đứng ở 575s — một mốc KHÔNG
#: có audio trong timeline — nên `next_chunk()` trả None mãi và 'mốc-đã-xử-lý' không bao giờ tiến).
_FRONTIER_BEHIND_RESYNC_SEC = 20.0
#: Lượng audio TỐI THIỂU tại con trỏ để còn cắt được một khối. Dưới mức này thì đứng chờ là vô
#: ích (bộ cắt khối cần ≥ 2,5 s cho cả chế độ bootstrap lẫn nhánh khẩn cấp) ⇒ nhảy tới đoạn kế
#: nếu có (xem `_nudge_cursor_out_of_gap`).
_MIN_USEFUL_AUDIO_SEC = 2.5


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


#: Ký tự kết câu (Latin + CJK) — dùng để biết một phụ đề đã TRỌN CÂU hay chưa.
_SENTENCE_END_CHARS = (".", "?", "!", "。", "？", "！", "…")
#: Ký tự bao có thể nằm SAU dấu kết câu (`anh ấy nói "Đứng lại."`).
_TRAILING_WRAPPERS = "\"'”’»)]}）】」』"


def _ends_sentence(text: str) -> bool:
    """`True` nếu văn bản kết thúc bằng dấu kết câu (bỏ qua ký tự bao phía sau)."""
    clean = (text or "").strip().rstrip(_TRAILING_WRAPPERS).rstrip()
    return bool(clean) and clean[-1] in _SENTENCE_END_CHARS


def _free_vram_mb() -> Optional[float]:
    """VRAM trống (MB) của GPU đang dùng, hoặc None nếu không đo được (không có GPU NVIDIA).

    Đọc qua NVML (driver) — KHÔNG import torch, không tạo CUDA context, và thấy được cả VRAM
    do TTS worker omnivoice.cpp / llama.cpp / transcribe.dll (native) chiếm.
    """
    from backend.utils.gpu_mem import free_vram_mb

    return free_vram_mb()


def _empty_torch_cache() -> None:
    """Trả các khối cache của PyTorch về driver — CHỈ khi torch ĐÃ được nạp sẵn.

    TTS/dịch/ASR đều chạy native (GGML), nên torch chỉ còn trong tiến trình khi có module
    khác dùng (vd. ForcedAligner của Lookahead). Không bao giờ import torch chỉ để dọn cache.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
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
            sample_rate=16000,
            max_pending_sec=float(la.max_pending_sec),
            max_storage_sec=float(getattr(la, "max_storage_sec", 10800.0) or 10800.0),
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

        # Pipeline B CHỈ có một chế độ: OFFLINE_BATCH (Qwen3-ASR + Forced Aligner).
        # v2 "streaming giả lập" (VAD + CommitManager + preview) đã bị XOÁ ngày 2026-10-02 — người
        # dùng muốn cơ chế cắt câu kiểu đó thì chạy Pipeline A (`/ws`), hoặc tắt Lookahead trong
        # popup để extension tự chuyển sang Pipeline A (xem report/09_lookahead_offline_batch).
        # Bộ cắt khối: cắt CHỈ tại khoảng lặng mà VAD xác nhận (xem `backend/core/vad_silence.py`).
        # Vì mép khối nằm giữa khoảng lặng thật nên không từ nào bị chẻ đôi ⇒ KHÔNG cần
        # `overlap_sec`, KHÔNG cần trừ chồng lấn, KHÔNG cần hàn gắn mảnh cuối (đã xoá 2026-10-04).
        self.silence_scanner: Optional[VADSilenceScanner] = (
            VADSilenceScanner(
                engine_name="silero-vad",
                sample_rate=16000,
                min_silence_ms=float(getattr(la, "batch_vad_silence_ms", 1500.0)),
                threshold=float(getattr(la, "batch_vad_silence_threshold", 0.30)),
            )
            if bool(getattr(la, "batch_use_vad_silence", True))
            else None
        )
        self.chunker = LookaheadChunker(
            timeline=self.timeline,
            sample_rate=16000,
            min_window_sec=float(getattr(la, "batch_min_sec", 8.0)),
            min_silence_ms=float(getattr(la, "batch_min_silence_ms", 250.0)),
            strong_silence_ms=float(getattr(la, "batch_strong_silence_ms", 450.0)),
            silence_rel_db=float(getattr(la, "batch_silence_rel_db", 22.0)),
            silence_floor_rms=float(getattr(la, "batch_silence_floor_rms", 0.004)),
            min_rms_ratio=float(getattr(la, "batch_min_rms_ratio", 0.35)),
            min_rms_hold_ms=float(getattr(la, "batch_min_rms_hold_ms", 120.0)),
            dip_search_sec=float(getattr(la, "batch_dip_search_sec", 3.0)),
            silence_scanner=self.silence_scanner,
            vad_silence_ms=float(getattr(la, "batch_vad_silence_ms", 1500.0)),
            max_audio_sec=float(getattr(la, "batch_max_audio_sec", 45.0)),
        )
        #: Trần gom câu phụ đề của tuyến batch (popup có thể thu hẹp qua `min_words_to_commit`).
        self._batch_sub_opts: Dict[str, Any] = {
            "max_words": int(getattr(la, "batch_sub_max_words", 10)),
            "max_duration_sec": float(getattr(la, "batch_sub_max_duration_sec", 4.5)),
            "max_chars": int(getattr(la, "batch_sub_max_chars", 45)),
            "min_words": max(1, int(getattr(la, "batch_sub_min_words", 2))),
            "defer_words": max(0, int(getattr(la, "batch_sub_defer_words", 3))),
            # Trần cho một CÂU TRỌN VẸN: có dấu kết câu trong phạm vi này ⇒ cả câu là một phụ đề.
            "sentence_max_words": int(getattr(la, "batch_sub_sentence_max_words", 24)),
            "sentence_max_duration_sec": float(
                getattr(la, "batch_sub_sentence_max_duration_sec", 14.0)
            ),
            "sentence_max_chars": int(getattr(la, "batch_sub_sentence_max_chars", 160)),
        }
        #: Mốc PTS (trong KHÔNG GIAN AUDIO) của nội dung ĐÃ PHÁT gần nhất — chỉ để chẩn đoán.
        self._batch_prev_emitted_end_audio: Optional[float] = None
        #: Đuôi TỪ (đã chuẩn hoá) của nội dung đã phát gần nhất — chỉ để chẩn đoán.
        self._batch_emitted_tail_norm: List[str] = []
        #: THẾ HỆ SEEK của marker `_ready_until_pts`. Marker chỉ có giá trị cho đúng thế hệ này;
        #: sau khi tua, marker cũ (thuộc vị trí cũ) KHÔNG được phép báo "đã sẵn sàng" — nếu không,
        #: video phát qua vùng chưa hề được xử lý (sự cố 2026-10-02: "Đã dịch: 232.9s" trong khi
        #: playhead ở 27s ⇒ seek ngược về đầu không hiện phụ đề).
        self._ready_until_seq: int = 0
        #: Mốc thời gian (perf_counter) bắt đầu thấy con trỏ khối KẸT xa vị trí phát — dùng cho
        #: watchdog neo lại (xem `_maybe_reanchor_batch_frontier`).
        self._frontier_stuck_since: Optional[float] = None
        #: Mốc thời gian bắt đầu "con trỏ khối kẹt PHÍA SAU vị trí phát" (xem
        #: `_maybe_reanchor_batch_frontier`). Cần bộ đếm RIÊNG vì hai kiểu kẹt loại trừ nhau.
        #: Mốc thời gian lần cuối log "khối ASR trả về rỗng" (xem `_log_empty_block`).
        self._last_empty_block_log_at: float = 0.0
        self._frontier_behind_since: Optional[float] = None
        #: Bộ lọc trùng của Pipeline A — chỉ còn dùng cho trường hợp aligner lỗi hẳn (fallback).
        self._batch_dedup = CommitDeduplicator()
        self._aligner_service: ForcedAlignerService = ForcedAlignerService.get_instance()
        self._batch_from_pts: Optional[float] = None
        self._fast_bootstrap: bool = False
        #: Thời gian xử lý TRỌN một khối (ASR + Aligner + dịch + gửi), trung bình trượt (giây) —
        #: dùng để tính ngưỡng "khẩn cấp" của cổng kiên nhẫn (xem `_batch_gate`).
        self._batch_proc_ema_sec: float = 1.5
        #: Khoá (seek_seq, from_pts) của lần cổng bắt đầu CHỜ gần nhất — chỉ để log một lần.
        self._batch_gate_wait_key: Optional[Tuple[int, float]] = None

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

        #: PTS của audio đã đưa qua ASR (offline block) — cập nhật bởi `_commit_batch_progress`.
        #: Ở tuyến OFFLINE_BATCH, "đã nạp vào ASR" chính là mốc kết thúc khối vừa xử lý.
        self._feed_pts: Optional[float] = None
        self._session_start_pts: Optional[float] = None
        #: Mốc kết thúc audio ĐÃ NHẬN & GIẢI MÃ. Dùng để báo `buffered_ahead` trung thực: nếu chỉ
        #: nhìn phần còn chờ trong timeline thì sau khi nạp hết, buffer sẽ báo 0 dù trình duyệt vẫn
        #: còn cả chục giây audio phía trước.
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
        self._ingest_lock = asyncio.Lock()
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
        # `decoder` (byte tới nhiều mà PCM giải mã ra ít) hay tầng xử lý khối (PCM có sẵn mà lâu
        # mới có phụ đề). `bytes_in` / `fed_ahead` cho tỷ lệ giải mã và độ phủ tính bằng số THẬT.
        self.bytes_in = 0
        self.decode_seconds = 0.0          # tổng thời lượng PCM giải mã ra (đã lọc)
        self.decode_wall_sec = 0.0         # tổng thời gian wall của các lời gọi demuxer.feed
        self.media_seconds = 0.0           # tổng giây MEDIA giải mã ra (kể cả phần bị bỏ xa)
        self.decode_delta_sec = 0.0        # giây media trong cửa sổ chẩn đoán hiện tại
        self.decode_windows = 0            # số lượt feed CÓ sinh ra PCM
        self.chunks_dropped_far = 0        # số đoạn PCM bị bỏ vì quá xa vị trí phát
        self.start_offset_warned = 0       # số lần cảnh báo lệch đầu phiên (chỉ log vài lần)
        #: Chẩn đoán sau khi tua (xem `handle_audio_fragment`).
        self._seek_reset_at = 0.0
        self._seek_diag_left = 0
        self.fragments_deduped = 0         # số mảnh bị bỏ vì trùng nội dung
        self._fragment_digests: Dict[bytes, float] = {}
        self._fragment_digest_order: deque = deque()
        self._max_decode_lead_sec = float(getattr(la, "max_decode_lead_sec", 240.0) or 240.0)
        self._playhead_pts = 0.0
        self._last_decoded_at = 0.0        # lần cuối có PCM mới
        #: Mốc so sánh cho log chẩn đoán định kỳ (xem `_maybe_log_diagnostics`).
        self._last_diag_at = time.perf_counter()
        self._diag_bytes_in = 0
        self._diag_decode_seconds = 0.0
        self._diag_media_seconds = 0.0
        self._diag_decode_wall = 0.0
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
        """Dựng ASR (+ VAD chỉ để đọc phần bù mép-nói) theo ĐÚNG cấu hình popup.

        ⚠️ Tuyến OFFLINE_BATCH **không chạy VAD** để cắt câu (việc đó do Qwen3-ForcedAligner lo).
        VAD ở đây chỉ còn một nhiệm vụ: cho biết `start_pad_ms` — phần bù mà engine VAD vốn lùi mép
        nói — để canh mốc phụ đề/TTS (xem `_refresh_start_pad`).

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

        # Ngôn ngữ nguồn: popup chọn gì thì ASR nhận đúng cái đó.
        try:
            self.asr_engine.set_language(self.source_lang or "auto")
        except Exception:  # noqa: BLE001
            pass

        # VAD: dựng với thông số popup (engine/threshold/silence) — chỉ dùng cho `start_pad_ms`.
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
            engine_override=vad_engine_override,
        )

        # Áp cấu hình popup (ngôn ngữ/TTS/trần gom câu…) rồi đọc phần bù mép nói.
        if self._parsed_config is not None:
            self._apply_config_sync(self._parsed_config)
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

        # ── Trần GOM CÂU PHỤ ĐỀ của tuyến OFFLINE_BATCH ─────────────────────────
        # Tuyến này KHÔNG dùng CommitManager/preview của Pipeline A (câu được cắt bằng dấu câu của
        # bản phiên âm + mốc từ của Forced Aligner), nên các tham số phân câu kiểu A
        # (`stability_*`, `max_duration_sec`, `max_chars`, `split_on_stability`…) KHÔNG được áp ở
        # đây. Chỉ `min_words_to_commit` được dùng lại — với nghĩa GỘP mảnh cụt, không phải lọc bỏ.
        if parsed.min_words_to_commit is not None:
            self._min_words_to_commit = max(0, int(parsed.min_words_to_commit))
            self._batch_sub_opts["min_words"] = max(1, self._min_words_to_commit)
            applied["min_words_to_commit"] = self._min_words_to_commit

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
        from backend.asr import lifecycle as asr_lifecycle
        from backend.asr.registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        clean = (model_key or "").strip().lower()
        if not registry.has_model(clean):
            logger.warning(f"Model ASR '{model_key}' không có trong catalog — bỏ qua.", extra={"module_tag": "WS"})
            return
        if asr_lifecycle.is_busy() and not asr_lifecycle.is_busy(clean):
            logger.info("Đang nạp model ASR khác — bỏ qua yêu cầu đổi model.", extra={"module_tag": "WS"})
            return

        async def _run() -> None:
            await self.send_json({"type": "model_status", "stage": "asr", "state": "loading", "model": clean})
            try:
                if asr_lifecycle.needs_download(clean) and not asr_lifecycle.auto_download_enabled():
                    raise RuntimeError("Chưa có file GGUF cục bộ và auto_download đang tắt")
                await asr_lifecycle.activate_model(clean)
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
        from backend.translation import lifecycle as trans_lifecycle
        from backend.translation.registry import TranslationModelRegistry

        registry = TranslationModelRegistry.get_instance()
        if not registry.is_known(model_request):
            logger.warning(f"Model dịch '{model_request}' không có trong catalog — bỏ qua.", extra={"module_tag": "WS"})
            return
        canonical = registry.resolve_key(model_request)
        if canonical == getattr(config.translation, "base", None):
            return
        if trans_lifecycle.is_busy() and not trans_lifecycle.is_busy(canonical):
            logger.info("Đang nạp model dịch khác — bỏ qua yêu cầu đổi model.", extra={"module_tag": "WS"})
            return

        async def _run() -> None:
            await self.send_json({"type": "model_status", "stage": "translation", "state": "loading", "model": canonical})
            try:
                if trans_lifecycle.needs_download(canonical) and not trans_lifecycle.auto_download_enabled():
                    raise RuntimeError("Chưa có file GGUF cục bộ và auto_download đang tắt")
                await trans_lifecycle.activate_model(canonical)
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
            if self.asr_engine is not None and clean != getattr(self.asr_engine, "model_key", None):
                logger.warning(
                    f"Lookahead: Bỏ qua yêu cầu đổi model ASR sang '{clean}' (chỉ được đổi trước khi bắt đầu session).",
                    extra={"module_tag": "WS"},
                )

        if parsed.translation_model is not None:
            clean_tr = str(parsed.translation_model).strip().lower()
            from backend.translation.registry import TranslationModelRegistry
            canonical = TranslationModelRegistry.get_instance().resolve_key(clean_tr)
            if canonical != getattr(config.translation, "base", None):
                logger.warning(
                    f"Lookahead: Bỏ qua yêu cầu đổi model dịch sang '{canonical}' (chỉ được đổi trước khi bắt đầu session).",
                    extra={"module_tag": "WS"},
                )


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
        """Khởi động các vòng lặp nền: xử lý khối OFFLINE_BATCH, báo trạng thái (+TTS).

        Tuyến OFFLINE_BATCH là tuyến DUY NHẤT của Pipeline B (v2 "streaming giả lập" đã bị xoá
        ngày 2026-10-02): ASR chỉ nhận khối dài 12-30 s kèm ngữ cảnh đầy đủ, còn việc ngắt câu do
        dấu câu của bản phiên âm + mốc từ của Forced Aligner quyết định.
        """
        loop = asyncio.get_running_loop()
        self._tasks.append(
            loop.create_task(self._offline_batch_loop(), name=f"la_batch_{self.session_id[:8]}")
        )
        self._tasks.append(
            loop.create_task(self._status_loop(), name=f"la_status_{self.session_id[:8]}")
        )
        self._spawn(self._prewarm_aligner())
        logger.info(
            "Khởi động Lookahead Pipeline B ở chế độ: OFFLINE_BATCH (Qwen3-ASR + Forced Aligner)",
            extra={"module_tag": "WS"},
        )
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

        # Giải phóng bộ nhớ RAM audio của phiên
        self.timeline.clear()

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
        media_start: Optional[float] = None,
        media_end: Optional[float] = None,
    ) -> None:
        """Nhận một mảnh MSE, ghép nối + giải mã rồi đưa vào timeline liên tục.

        `refetched=True`: mảnh do extension TẢI LẠI cho vùng đã buffer sẵn. Byte của nó có thể
        trùng mảnh đã nhận trước đó nên phải **miễn dedup** — nếu không, vùng video đã tải trước
        sẽ không bao giờ có phụ đề (đo thật 2026-10-01: "Bỏ mảnh audio TRÙNG … 51 … 53" ngay
        sau khi client gửi lại).

        `media_start`/`media_end`: khoảng THẬT (trục thời gian trình duyệt) của mảnh trong
        `SourceBuffer.buffered` — demuxer dùng để neo PCM khi mốc container lệch trục video
        (trang dùng `mode = "sequence"` / `tfdt` tuyệt đối / đổi `timestampOffset` sau append).
        """
        if self._closed:
            return
        if seek_id and seek_id != self.active_seek_id:
            return

        self.fragments_received += 1
        self.bytes_in += len(chunk_bytes)

        # ── CHỐNG MẢNH TRÙNG ──────────────────────────────────────────────────
        now_perf = time.perf_counter()
        digest = hashlib.blake2b(chunk_bytes, digest_size=16).digest()
        last_seen = self._fragment_digests.get(digest)
        if (not refetched) and last_seen is not None and (now_perf - last_seen) <= _DUP_WINDOW_SEC:
            self.fragments_deduped += 1
            return
        self._fragment_digests[digest] = now_perf
        self._fragment_digest_order.append(digest)
        while len(self._fragment_digest_order) > 512:
            self._fragment_digests.pop(self._fragment_digest_order.popleft(), None)

        self._playhead_pts = float(self.current_time or 0.0)

        # ── ĐỔI EPOCH (nguồn/SourceBuffer mới) ⇒ bộ dò khoảng lặng phải QUÊN cache ──────────
        # Cache xác suất VAD khoá theo mốc PTS TUYỆT ĐỐI. Khi epoch đổi, cùng một PTS ứng với nội
        # dung audio KHÁC (video khác trên cùng phiên WebSocket, hoặc nguồn được nạp lại) nên tái
        # dùng xác suất cũ sẽ cắt khối vào chỗ không hề im lặng, hoặc bỏ qua khoảng lặng thật.
        if (
            epoch is not None
            and int(epoch) != int(self.demuxer.epoch)
            and self.silence_scanner is not None
        ):
            self.silence_scanner.invalidate_cache()
            logger.info(
                f"Đổi epoch {self.demuxer.epoch} -> {int(epoch)}: đã xoá cache xác suất VAD.",
                extra={"module_tag": "VAD"},
            )

        t_dec = time.perf_counter()
        try:
            decoded = await asyncio.to_thread(
                self.demuxer.feed,
                chunk_bytes,
                timestamp_offset,
                mime_type,
                is_init,
                epoch,
                media_start,
                media_end,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Lỗi ghép nối/giải mã mảnh audio: {exc}", extra={"module_tag": "WS"})
            return
        self.decode_wall_sec += time.perf_counter() - t_dec

        # Nạp các đoạn PCM giải mã được vào ContinuousAudioTimeline (lưu trữ RAM)
        dropped_ahead = 0
        media_sec = 0.0
        for chunk in decoded:
            media_sec += float(chunk.duration)
            if chunk.pts_start > self._playhead_pts + self._max_decode_lead_sec:
                dropped_ahead += 1
                continue
            self.timeline.append(chunk.pts_start, chunk.pcm)
            self.chunks_decoded += 1
            self.decode_seconds += float(chunk.duration)
            self._decoded_end_pts = max(self._decoded_end_pts, float(chunk.pts_end))
        self.media_seconds += media_sec
        self.decode_delta_sec += media_sec

        if dropped_ahead:
            self.chunks_dropped_far += dropped_ahead
            logger.debug(
                f"Bỏ {dropped_ahead} đoạn PCM nằm >{self._max_decode_lead_sec:.0f}s trước vị trí phát",
                extra={"module_tag": "ASR"},
            )

        if decoded:
            self._last_decoded_at = time.perf_counter()
            self.decode_windows += 1
            logger.debug(
                f"Giải mã: +{media_sec:.2f}s [{decoded[0].pts_start:.2f}s -> {decoded[-1].pts_end:.2f}s] "
                f"(mảnh {len(chunk_bytes)}B, buffer={self.demuxer.buffered_bytes}B)",
                extra={"module_tag": "ASR"},
            )

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

        async with self._ingest_lock:
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
            # Bộ dò khoảng lặng khoá cache theo PTS tuyệt đối; sau khi tua, vị trí đọc nhảy sang
            # đoạn khác nên bỏ cache + audio mồi để lần quét tới tính lại từ dữ liệu thật.
            if self.silence_scanner is not None:
                self.silence_scanner.invalidate_cache()
            # Không xoá audio RAM khi tua: chỉ chuyển con trỏ đọc (read cursor) tới mốc mới
            self.timeline.seek(float(target_time))
            self._feed_pts = float(target_time)
            self._session_start_pts = float(target_time)
            self._batch_from_pts = float(target_time)
            self._fast_bootstrap = True
            # Lịch sử chẩn đoán của vị trí CŨ không còn dùng được.
            self._batch_prev_emitted_end_audio = None
            self._batch_emitted_tail_norm = []
            self._batch_dedup.clear()

            # Kiểm tra xem RAM đã có sẵn audio cho vị trí mới chưa (ví dụ tua lùi hoặc tua trong vùng đã buffer)
            buffered_end = self.timeline.buffered_end_from(float(target_time))
            if buffered_end is not None and buffered_end > float(target_time):
                self._decoded_end_pts = float(buffered_end)
            else:
                self._decoded_end_pts = float(target_time)
            # Phụ đề cho vị trí tua MỚI chưa được dịch -> mốc ready_until luôn bắt đầu từ target_time.
            # GẮN THẾ HỆ SEEK: marker cũ (thuộc vị trí vừa rời) không được phép báo "đã dịch" cho vị
            # trí mới — nếu không, video sẽ phát qua vùng chưa hề được xử lý.
            self._ready_until_pts = float(target_time)
            self._ready_until_seq = self._seek_seq
            self._frontier_stuck_since = None

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

            self._last_diag_at = time.perf_counter()
            self._diag_bytes_in = self.bytes_in
            self._diag_decode_seconds = self.decode_seconds
            self._diag_media_seconds = self.media_seconds
            self._diag_decode_wall = self.decode_wall_sec
            metrics_collector.increment_counter("lookahead.seek_reset")
            self._ingest_event.set()
            logger.info(
                f"Seek seek_id={seek_id} mốc={target_time:.2f}s "
                f"(RAM={self.timeline.total_stored_seconds():.1f}s, "
                f"đệm trước={(self._decoded_end_pts - target_time):.1f}s)",
                extra={"module_tag": "WS"},
            )

    # ── OFFLINE BATCH PIPELINE (v3 Qwen3-ASR + Forced Aligner) ─────────────────
    async def _prewarm_aligner(self) -> None:
        try:
            if self._aligner_service is not None:
                await asyncio.to_thread(self._aligner_service.prewarm)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Prewarm Aligner lỗi (bỏ qua): {exc}", extra={"module_tag": "ASR"})

    def _run_asr_block(self, pcm: np.ndarray, lang: str) -> str:
        if self.asr_engine is None:
            return ""
        if hasattr(self.asr_engine, "transcribe_block"):
            return self.asr_engine.transcribe_block(pcm, language=lang if lang != "auto" else None)
        return ""

    def _commit_batch_progress(
        self,
        seq: int,
        next_read_pts: float,
        chunk_end_pts: float,
    ) -> bool:
        """Ghi tiến độ khối batch — CHỈ khi CHƯA có seek mới hơn (chống race).

        VÌ SAO CẦN: `_offline_batch_loop` có `await self.send_json(...)` ngay trước khi ghi tiến
        độ. Trong lúc chờ `await` đó, cùng event loop có thể xử lý `sync_state`/`seek_reset` ⇒
        `handle_seek()` đã reset `_batch_from_pts`/`_ready_until_pts` về mốc tua, rồi vòng lặp quay
        lại GHI ĐÈ bằng mốc của khối CŨ. Đo thật 2026-10-02 (log thật):

            [WS] Seek seek_id=…_7thc mốc=34.99s     ← reset pipeline về 34.99s
            93 ms sau: prebuffer_ready=True (ready_ahead=7.42s ⇒ ready_until = 42.41s)
            [ASR] Lookahead Batch ASR [42.41s -> 54.83s]   ← chạy tiếp từ mốc CŨ

        ⇒ vùng [34.99s, 42.41s] **không bao giờ** được phiên âm ("lúc hiện phụ đề lúc không").

        Returns:
            True nếu tiến độ được ghi (cùng thế hệ seek), False nếu là iteration cũ (bỏ qua).
        """
        if self._closed or seq != self._seek_seq:
            return False
        self._batch_from_pts = float(next_read_pts)
        #: "Đã đưa qua ASR" = mốc kết thúc khối vừa xử lý (ở tuyến batch không còn tầng nạp VAD
        #: riêng) — nhờ vậy `fed_ahead` trong status/log phản ánh đúng độ phủ thật.
        self._feed_pts = float(chunk_end_pts)
        # Marker cũ thuộc THẾ HỆ KHÁC phải bị THAY, không được `max()` — nếu lấy max thì một marker
        # quá lớn của vị trí cũ sẽ sống sót và được "hợp thức hoá" bằng thế hệ mới (đo thật: 260s
        # của vị trí cũ ⇒ status báo đã dịch 232.9s trong khi playhead ở 27s).
        if self._ready_until_seq == self._seek_seq:
            self._ready_until_pts = max(self._ready_until_pts, float(chunk_end_pts))
        else:
            self._ready_until_pts = float(chunk_end_pts)
        self._ready_until_seq = self._seek_seq
        self._fast_bootstrap = False
        self._frontier_stuck_since = None
        return True

    def _reanchor_batch(self, target_pts: float) -> None:
        """Neo lại con trỏ khối batch về `target_pts` và xoá mọi trạng thái ranh giới của vị trí cũ."""
        self._batch_from_pts = float(target_pts)
        self._feed_pts = float(target_pts)
        self._fast_bootstrap = True
        self._ready_until_pts = float(target_pts)
        self._ready_until_seq = self._seek_seq
        self._batch_prev_emitted_end_audio = None
        self._batch_emitted_tail_norm = []
        self._batch_dedup.clear()
        self._frontier_stuck_since = None
        self._frontier_behind_since = None

    def _maybe_reanchor_batch_frontier(self, from_pts: float, max_ahead_pts: float) -> bool:
        """RÀNG BUỘC CỨNG: con trỏ khối không được KẸT xa vị trí phát.

        Vòng lặp batch bỏ qua (`continue`) khi con trỏ vượt tầm nhìn lookahead — đúng khi nó chỉ
        nhô hơn `current_time + lead + margin` một khối (cơ chế throttle bình thường). Nhưng nếu vì
        lý do nào đó (state lệch sau khi tua, seek ngược) con trỏ nằm CÁCH XA vị trí phát một cách
        BỀN VỮNG thì vòng lặp sẽ `continue` mãi mãi: **video vẫn phát mà không có gì hoạt động**.
        Đo thật 2026-10-02: sau khi tua ngược về 16.89s, `Đã dịch: 232.9s` và không có khối nào được
        xử lý trong 30 s tiếp theo.

        Ở đây chỉ phát hiện + neo lại (không tự ý bỏ qua khối đang xử lý). Ngưỡng gồm cả một khối
        dài nhất (`batch_max_audio_sec`) nên throttle bình thường không bao giờ kích hoạt; cần thêm
        `_FRONTIER_RESYNC_AFTER_SEC` giây BỀN VỮNG để không rung khi playhead vừa nhảy.

        Returns:
            True nếu VỪA neo lại (vòng lặp nên `continue` để chạy lại từ mốc mới).
        """
        limit = float(max_ahead_pts) + float(getattr(config.lookahead, "batch_max_audio_sec", 45.0)) + 1.0
        if from_pts > limit:
            now = time.perf_counter()
            if self._frontier_stuck_since is None:
                self._frontier_stuck_since = now
            elif now - self._frontier_stuck_since >= _FRONTIER_RESYNC_AFTER_SEC:
                logger.warning(
                    f"Con trỏ khối batch kẹt ở {from_pts:.2f}s trong khi vị trí phát chỉ {self.current_time:.2f}s "
                    f"(quá tầm nhìn {limit:.1f}s) — NEO LẠI về vị trí phát để không phát video mà không xử lý gì.",
                    extra={"module_tag": "WS"},
                )
                metrics_collector.increment_counter("lookahead.batch_frontier_reanchored")
                self._reanchor_batch(self.current_time)
                self._frontier_stuck_since = None
                return True
        else:
            self._frontier_stuck_since = None

        # ── KẸT PHÍA SAU vị trí phát (đo thật 2026-10-05, xhamster.com) ─────────────
        # Triệu chứng: `mốc-đã-xử-lý` đứng im ở 575.01s trong khi playhead chạy tới 967s+ và
        # "Đã dịch: 0.0s" mãi. Nguyên nhân: con trỏ khối nằm ở một mốc KHÔNG còn audio (khe hở
        # của timeline sau khi trang tự nạp lại nguồn / sau khi tua mà `seek_reset` không tới
        # backend) ⇒ `_batch_gate` trả "no_audio", `next_chunk()` trả None, vòng lặp quay mãi.
        # Van cũ chỉ xử lý "kẹt QUÁ XA PHÍA TRƯỚC" nên không bao giờ cứu được ca này.
        behind = float(self.current_time) - float(from_pts)
        frontier_at_cursor = self.timeline.buffered_end_from(from_pts)
        available_at_cursor = (
            0.0 if frontier_at_cursor is None else max(0.0, float(frontier_at_cursor) - float(from_pts))
        )
        # Điều kiện "không còn đủ audio để cắt khối": KHÔNG có audio, hoặc chỉ còn một mẩu nhỏ hơn
        # cửa sổ tối thiểu của bộ cắt khối. Đo thật 2026-10-05 (xvideos.com): con trỏ ở 20.04s —
        # ĐÚNG mép một khe hở của timeline — nên `buffered_end_from()` trả về ~20.04 (khác `None`)
        # và `available = 0.0s`: bộ cắt khối không bao giờ cắt được, `next_chunk()` trả `None`
        # mãi trong khi playhead đã ở 41s ("chạy một chút lại dừng").
        min_window = float(getattr(self.chunker, "min_window_sec", 0.0) or 0.0)
        if behind < _FRONTIER_BEHIND_RESYNC_SEC or available_at_cursor >= min_window:
            self._frontier_behind_since = None
            return False
        now = time.perf_counter()
        if self._frontier_behind_since is None:
            self._frontier_behind_since = now
            return False
        if now - self._frontier_behind_since < _FRONTIER_RESYNC_AFTER_SEC:
            return False
        logger.warning(
            f"Con trỏ khối batch KẸT PHÍA SAU vị trí phát: con trỏ {from_pts:.2f}s, playhead "
            f"{self.current_time:.2f}s (chậm {behind:.1f}s) và chỉ còn {available_at_cursor:.1f}s "
            f"audio tại con trỏ (< {min_window:.1f}s tối thiểu để cắt khối) ⇒ NEO LẠI về vị trí "
            f"phát (nếu không, video phát mà không xử lý gì).",
            extra={"module_tag": "WS"},
        )
        metrics_collector.increment_counter("lookahead.batch_frontier_reanchored")
        self._reanchor_batch(self.current_time)
        self._frontier_behind_since = None
        return True

    def _nudge_cursor_out_of_gap(self, from_pts: float) -> bool:
        """Đưa CON TRỎ KHỐI ra khỏi vùng KHÔNG ĐỦ AUDIO ĐỂ CẮT KHỐI.

        Hai dạng đã gặp thật (xvideos.com, HLS tách track):

        1. **Khe hở hoàn toàn** (2026-10-05): `handle_seek` neo con trỏ đúng vào mốc tua
           (313,44 s) nhưng audio thật của vị trí mới chỉ bắt đầu muộn hơn ⇒
           `buffered_end_from(con_trỏ)` trả `None` ⇒ `next_chunk()` trả `None` MÃI MÃI.
        2. **Mẩu vụn rồi khe hở** (đo lại cùng ngày): con trỏ ở 330,03 s — có ĐÚNG 0,0 s audio
           tại đó (mép cuối một đoạn) rồi khe hở mới tới đoạn kế. `buffered_end_from()` trả về
           ~330,03 (KHÁC `None`) nên van theo kiểu (1) không kích hoạt; `next_chunk()` vẫn trả
           `None` vì `buffered_end <= from_pts + 0.05`. Hệ quả: `mốc-đã-xử-lý` đứng im và client
           pause/resume liên tục dù RAM đã có hàng chục giây audio ngay sau khe.

        Cách xử lý chung: nếu lượng audio ĐỌC ĐƯỢC tại con trỏ nhỏ hơn mức tối thiểu để cắt một
        khối (`_MIN_USEFUL_AUDIO_SEC`) VÀ tồn tại một đoạn audio nằm sau khe hở thì nhảy con trỏ
        tới đầu đoạn đó. Không có đoạn nào phía sau (đang ở mép nạp) ⇒ KHÔNG nhảy, chờ nạp thêm
        là đúng.

        Returns:
            True nếu vừa nhảy con trỏ.
        """
        run_end = self.timeline.buffered_end_from(from_pts)
        available = 0.0 if run_end is None else max(0.0, float(run_end) - float(from_pts))
        if available >= _MIN_USEFUL_AUDIO_SEC:
            return False
        # Dò từ NGAY SAU phần audio hiện có: chỉ nhảy khi thật sự có đoạn audio ở phía sau khe.
        probe = float(from_pts) + max(0.05, available) + 0.05
        nxt = self.timeline.first_audio_pts_at_or_after(probe)
        if nxt is None or nxt <= float(from_pts) + 0.2:
            return False
        logger.info(
            f"[SEG_BATCH] Con trỏ khối {from_pts:.2f}s chỉ có {available:.2f}s audio tại chỗ "
            f"(cần ≥ {_MIN_USEFUL_AUDIO_SEC:.1f}s để cắt khối) — nhảy tới đầu đoạn audio kế tiếp "
            f"{nxt:.2f}s (lệch {nxt - float(from_pts):.2f}s) để pipeline chạy tiếp thay vì đứng im.",
            extra={"module_tag": "ASR"},
        )
        metrics_collector.increment_counter("lookahead.batch_cursor_gap_nudged")
        self._batch_from_pts = float(nxt)
        #: KHÔNG đụng `_feed_pts`: nó nghĩa là "audio ĐÃ QUA ASR". Vùng bị nhảy qua không có audio
        #: nên không thể xử lý; nếu ta đẩy `_feed_pts` tới đây thì `fed_ahead` bị thổi phồng và
        #: client tưởng vùng đó đã được quét (resume sớm) — đo thật 2026-10-05: `prebuffer_ready`
        #: bật với `ready_ahead=0.00s, fed_ahead=29.85s` ngay sau khi nhảy 30s.
        return True

    def _log_empty_block(self, chunk) -> None:
        """Log VÌ SAO một khối không ra chữ — tách "audio im lặng" khỏi "model không nhận ra".

        SỰ CỐ THẬT 2026-10-05 (www.av01.media): `mốc-đã-xử-lý` và `fragments/decoded` vẫn tiến,
        `ready_ahead` 30–66 s (trông rất khoẻ) nhưng số câu (`utterances`) ĐỨNG IM ⇒ về sau không
        có phụ đề nào. Đường đi là nhánh "ASR trả về rỗng": khối vẫn được cắt, `_commit_batch_progress`
        vẫn đẩy con trỏ, mà KHÔNG có dòng log nào ⇒ không thể biết là do audio im lặng hay do model.

        Ba chỉ số dưới đây phân biệt ngay:
          * `RMS`/`đỉnh` ≈ 0 và `tỉ lệ mẫu 0` ≈ 100 % ⇒ PCM là IM LẶNG (lỗi tầng giải mã/khe hở).
          * `RMS` bình thường mà vẫn rỗng ⇒ model/VAD, không phải audio.
          * `im lặng do timeline lấp` = số giây silence được `get_audio_range/read` TỰ SINH để bắc
            qua khe hở ⇒ nếu con số này lớn thì audio thật đang thiếu.
        """
        now = time.perf_counter()
        if now - self._last_empty_block_log_at < 5.0:
            return
        self._last_empty_block_log_at = now
        pcm = getattr(chunk, "pcm", None)
        if pcm is None or len(pcm) == 0:
            stats = "PCM RỖNG"
        else:
            arr = np.asarray(pcm, dtype=np.float32)
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
            zero_ratio = float(np.mean(arr == 0.0)) if arr.size else 1.0
            stats = (
                f"PCM {arr.size / 16000.0:.1f}s | RMS={rms:.5f} | đỉnh={peak:.5f} | "
                f"mẫu-0={zero_ratio * 100:.1f}%"
            )
        logger.warning(
            f"[SEG_BATCH] Khối [{float(chunk.pts_start):.2f}s → {float(chunk.pts_end):.2f}s] "
            f"ASR trả về RỖNG ({stats}; mode={getattr(chunk, 'fallback_mode', '?')}; "
            f"silence={float(getattr(chunk, 'silence_gap_ms', 0.0)):.0f}ms; "
            f"im lặng-do-lấp={self.timeline.gap_filled_sec:.1f}s; playhead={float(self.current_time):.1f}s) "
            f"— KHÔNG có phụ đề cho đoạn này.",
            extra={"module_tag": "ASR"},
        )

    def _batch_stream_end(self) -> bool:
        """Video đã dừng/đã hết VÀ không còn audio chờ ⇒ cho phép cắt nốt phần đuôi (< min_window).
        Extension không gửi bản tin "video ended" riêng, nên nhận biết bằng: trình phát đang DỪNG
        và lượng audio còn lại phía trước ít hơn một cửa sổ chuẩn. Chỉ khi đó mới lấy khối ngắn làm
        khối cuối — nếu không, một lần tạm dừng giữa video sẽ khiến phần còn lại bị cắt vụn thành
        các khối cụt (mỗi khối cụt là một cơ hội chẻ đôi từ).
        """
        from_pts = self._batch_from_pts if self._batch_from_pts is not None else self.current_time
        buffered_end = self.timeline.buffered_end_from(from_pts)
        if buffered_end is None:
            return False
        return bool(self.is_paused and (buffered_end - from_pts) < self.chunker.min_window_sec)

    def _batch_gate(self, from_pts: float, stream_end: bool) -> Tuple[bool, str, bool]:
        """CỔNG KIÊN NHẪN: có nên cắt khối mới NGAY BÂY GIỜ không?

        SỰ CỐ THẬT 2026-10-05 (phiên `29d8eb1a`): ASR chạy ~5× thời gian thực nên đuổi kịp mép bộ
        đệm đã giải mã rất nhanh. Sau đó cứ mỗi lần trình duyệt nạp thêm ~10 s, `available` vừa chạm
        `batch_min_sec` (8 s) là vòng lặp cắt ngay — dù phần ĐÃ DỊCH trước playhead còn 30–60 s:

            [CHUNKER] from=39.10s frontier=47.11s available=8.02s
            [SEG_BATCH] Chọn khối: dài 8.0s (... playhead 8.8s, mode=forced_overlap)

        ⇒ khối 8 s cắt cưỡng bức (không có khoảng lặng VAD) lặp đi lặp lại. Không phải bộ cắt khối
        chọn sai: nó chỉ được nhìn thấy 8 s audio.

        Quy tắc (theo yêu cầu người dùng):
          1. audio đã có phía trước con trỏ khối >= `batch_ready_sec` (~30 s)   ⇒ cắt ("full").
          2. phần đã dịch phía trước playhead <= ngưỡng khẩn cấp              ⇒ cắt ("urgent").
             Ngưỡng = (`batch_urgent_lead_sec` + 2 × thời gian xử lý một khối) × tốc độ phát.
          3. bootstrap sau khi tua / hết luồng                                    ⇒ cắt.
          Còn lại ⇒ CHỜ (video vẫn phát, bộ đệm tiếp tục dày lên).

        Returns:
            `(go, reason, urgent)`.
        """
        la = config.lookahead
        if self._fast_bootstrap:
            return True, "bootstrap", True
        if stream_end:
            return True, "stream_end", True
        if not bool(getattr(la, "batch_wait_for_full_block", True)):
            return True, "gate_off", False

        frontier = self.timeline.buffered_end_from(from_pts)
        if frontier is None:
            # Chưa có audio ở con trỏ: để bộ cắt khối tự trả `None` như cũ.
            return True, "no_audio", False
        available = max(0.0, float(frontier) - float(from_pts))
        want = min(
            float(getattr(la, "batch_ready_sec", 30.0)),
            float(self.chunker.max_audio_sec),
        )
        if available >= want - 0.05:
            return True, "full", False

        rate = max(1.0, float(self.playback_rate or 1.0))
        ready_ahead = float(from_pts) - float(self.current_time)
        urgent_lead = (
            float(getattr(la, "batch_urgent_lead_sec", 5.0)) + 2.0 * float(self._batch_proc_ema_sec)
        ) * rate
        if ready_ahead <= urgent_lead:
            return True, "urgent", True

        key = (int(self._seek_seq), round(float(from_pts), 2))
        if key != self._batch_gate_wait_key:
            self._batch_gate_wait_key = key
            logger.info(
                f"[SEG_GATE] Chờ gom khối dài từ {from_pts:.2f}s: có {available:.1f}s/{want:.1f}s audio, "
                f"đã dịch trước playhead {ready_ahead:.1f}s (khẩn cấp khi <= {urgent_lead:.1f}s, "
                f"xử lý ~{self._batch_proc_ema_sec:.1f}s/khối).",
                extra={"module_tag": "ASR"},
            )
        return False, "wait", False

    async def _offline_batch_loop(self) -> None:
        """Vòng lặp xử lý Offline Batch cho Pipeline B v3.

        Quy trình:
        1. Đợi có audio mới trên ContinuousAudioTimeline qua `_ingest_event`.
        2. Dùng LookaheadChunker cắt khối (ưu tiên khoảng lặng VAD trong dải tới
           `batch_max_audio_sec`) hoặc Fast-Bootstrap (3-4s khi seek).
        3. ASR Offline block decoding qua transcribe.cpp (TRỌN khối — giữ ngữ cảnh dài).
        4. Gióng hàng mốc từ siêu tốc qua ForcedAlignerService (+ gắn lại dấu câu từ văn bản ASR).
        5. Gom từ thành các câu phụ đề SubtitleSentence theo dấu câu.
        6. Dịch từng câu phụ đề và gửi lookahead_subtitles về Client (độ trễ hiển thị 0.0s).
        7. Xếp vào hàng đợi TTS OmniVoice.

        ⚠️ KHÔNG còn bước "TRỪ phần chồng lấn ở ranh giới khối" và bước "GHÉP mảnh cuối chưa kết
        câu của khối trước" — cả hai đã bị XOÁ ngày 2026-10-04 (xem khối chú thích
        "ĐƠN GIẢN HOÁ 2026-10-04" bên dưới). Chúng chỉ tồn tại để bù cho việc cắt khối GIỮA CÂU;
        nay `LookaheadChunker` cắt vào GIỮA khoảng lặng do VAD xác nhận
        (`LookaheadChunker._find_vad_cut`) và `overlap_sec` mặc định = 0.0, nên mép khối không bao
        giờ rơi vào giữa từ ⇒ không có gì để trừ và cũng không được ghép.
        ⇒ Nếu ai đó bật `LookaheadChunker(overlap_sec > 0)` thì phải khôi phục tầng trừ chồng lấn
        TRƯỚC, nếu không từ ở ranh giới sẽ bị phát hai lần (sự cố "cuối câu này là đầu của câu
        sau", 2026-10-02).
        """
        la = config.lookahead
        while not self._closed:
            try:
                await asyncio.wait_for(self._ingest_event.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass
            self._ingest_event.clear()

            if self._closed:
                break

            seq = self._seek_seq
            from_pts = self._batch_from_pts if self._batch_from_pts is not None else self.current_time

            # Chặn nạp quá xa vị trí phát hiện tại (tránh đốt GPU vô ích),
            # nhưng tự động nới lỏng khi browser buffer đã có sẵn dồi dào.
            rate = max(1.0, float(self.playback_rate or 1.0))
            buffered_end_timeline = self.timeline.buffered_end_pts() or self.current_time
            available_ahead = max(0.0, buffered_end_timeline - self.current_time)
            adaptive_lead = min(60.0, max(float(self.lead_time), available_ahead * 0.8))
            max_ahead_pts = self.current_time + adaptive_lead * rate + float(la.feed_margin_sec)
            if self._maybe_reanchor_batch_frontier(from_pts, max_ahead_pts):
                continue
            if from_pts > max_ahead_pts:
                continue

            stream_end_block = self._batch_stream_end()
            go, gate_reason, gate_urgent = self._batch_gate(from_pts, stream_end_block)
            if not go:
                continue
            gate_ready_ahead = float(from_pts) - float(self.current_time)
            t_block = time.perf_counter()
            try:
                chunk = await asyncio.to_thread(
                    self.chunker.next_chunk,
                    from_pts=from_pts,
                    fast_bootstrap=self._fast_bootstrap,
                    is_stream_end=stream_end_block,
                    urgent=gate_urgent,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Lỗi LookaheadChunker: {exc}", extra={"module_tag": "WS"})
                chunk = None

            if chunk is None or self._closed or seq != self._seek_seq:
                # Con trỏ rơi vào KHE HỞ ⇒ `next_chunk()` trả None mãi mãi (xem
                # `_nudge_cursor_out_of_gap`). Chỉ nhảy khi vẫn còn thuộc thế hệ seek hiện tại.
                if chunk is None and not self._closed and seq == self._seek_seq:
                    self._nudge_cursor_out_of_gap(from_pts)
                continue

            # Bước 2A: ASR Offline Block Decoding
            t_asr = time.perf_counter()
            try:
                raw_text = await asyncio.to_thread(self._run_asr_block, chunk.pcm, self.source_lang)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Lỗi ASR offline block: {exc}", extra={"module_tag": "ASR"})
                raw_text = ""

            if self._closed or seq != self._seek_seq:
                continue

            asr_ms = (time.perf_counter() - t_asr) * 1000.0
            clean_text = (raw_text or "").strip()

            if not clean_text:
                # Không nhận diện được tiếng nói (im lặng hoặc nhạc nền)
                self._log_empty_block(chunk)
                self._commit_batch_progress(seq, chunk.next_read_pts, chunk.pts_end)
                await self.send_status()
                continue

            # ── ẢO GIÁC LẶP TỪ: chỉ lọc ở TẦNG CÂU, sau khi đã ngắt câu ─────────────
            # NGUYÊN TẮC (người dùng chốt 2026-10-05): "Ngắt câu khối trước, câu nào repetition
            # loop thì bỏ câu đó thôi".
            #
            # Bản cũ gọi `is_repetition_hallucination(clean_text)` cho TOÀN BỘ khối — hàm này trả
            # True khi có BẤT KỲ đoạn lặp ≥ 3 ở đâu trong chuỗi — nên chỉ cần ĐUÔI khối kẹt vòng là
            # vứt cả khối. Log thật:
            #
            #   '一時しないと収まんないよこれ。ちょっと入れるだけだから。ちょっと入れちょっとだから。
            #    ちょっとちょっと。あああああ…'      ← 4 câu THẬT ở đầu, chỉ ĐUÔI bị kẹt vòng
            #
            # Ở đây KHÔNG lọc gì: việc ngắt câu đã có sẵn trong chính tuyến này
            # (`group_words_to_subtitles` → `split_subtitles_by_clause_comma` →
            # `split_oversized_sentences`), rồi bộ lọc theo TỪNG CÂU ở `candidate_subs` bỏ đúng
            # những câu hỏng. Khối rác thuần vẫn ra 0 phụ đề vì câu duy nhất của nó bị bỏ.

            logger.info(
                f"Lookahead Batch ASR [{chunk.pts_start:.2f}s -> {chunk.pts_end:.2f}s] ({asr_ms:.0f}ms): '{clean_text}'",
                extra={"module_tag": "ASR"},
            )
            # CHẨN ĐOÁN CỠ KHỐI: không có dòng này thì không thể biết vì sao khối ngắn hơn mục tiêu
            # — do bộ đệm CHỈ LIÊN TỤC chưa đủ (khe hở trong timeline), do throttle theo vị trí phát,
            # do không có khoảng lặng gần mốc lý tưởng, hay do trần ngân sách token.
            timeline = self.timeline
            frontier = timeline.buffered_end_from(chunk.pts_start)
            logger.info(
                f"[SEG_BATCH] Chọn khối: dài {chunk.duration:.1f}s "
                f"(mục tiêu {getattr(self.chunker, 'last_effective_target_sec', 0.0):.1f}s, "
                f"dải tìm tới {getattr(self.chunker, 'last_effective_search_max_sec', 0.0):.1f}s, "
                f"mốc cắt {chunk.pts_start:.2f}s, "
                f"biên liên tục tới {(frontier if frontier is not None else -1.0):.2f}s, "
                f"RAM {timeline.total_stored_seconds():.1f}s, "
                f"đệm giải mã trước {available_ahead:.1f}s, "
                f"playhead {self.current_time:.1f}s, mode={chunk.fallback_mode}, "
                f"silence={chunk.silence_gap_ms:.0f}ms, cổng={gate_reason}, "
                f"đã dịch trước playhead lúc cắt {gate_ready_ahead:.1f}s).",
                extra={"module_tag": "ASR"},
            )

            # Bước 2B: Forced Alignment
            aligned_words = []
            align_failed = False
            # NGÔN NGỮ CHO ALIGNER: ưu tiên người dùng chọn; với `auto` thì ĐOÁN TỪ VĂN BẢN ASR.
            # ⚠️ Rơi về "English" khi văn bản là tiếng Nhật/Trung sẽ khiến upstream dùng
            # `tokenize_space_lang` ⇒ cả câu (không có khoảng trắng) thành MỘT token ⇒ aligner trả
            # một item duy nhất ⇒ KHÔNG ngắt được câu (sự cố thật 2026-10-03: cả khối 3 câu thành
            # một phụ đề dài trên trang tiếng Nhật, trong khi YouTube tiếng Anh vẫn đúng).
            align_lang = (
                resolve_aligner_language(self.source_lang)
                or detect_aligner_language_from_text(clean_text)
                or "English"
            )
            if self._aligner_service is not None:
                try:
                    aligned_words = await asyncio.to_thread(
                        self._aligner_service.align,
                        chunk.pcm,
                        clean_text,
                        align_lang,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Lỗi ForcedAligner (fallback sang câu đơn): {exc}", extra={"module_tag": "ASR"})
                    aligned_words = []
                    align_failed = True

            if self._closed or seq != self._seek_seq:
                continue

            # ── ĐƠN GIẢN HOÁ 2026-10-04 ──────────────────────────────────────────────
            # ĐÃ XOÁ toàn bộ tầng "sửa chữa" quanh Forced Aligner và quanh ranh giới khối:
            #   * TRỪ CHỒNG LẤN RANH GIỚI (theo mốc thời gian + theo chuỗi từ),
            #   * GIỮ LẠI mảnh cuối chưa kết câu rồi GHÉP vào khối sau (`_batch_carry_words`),
            #   * HIỆU CHỈNH MỐC TỪ (`calibrate_word_times`).
            #
            # VÌ SAO XOÁ: cả ba chỉ tồn tại để bù cho việc **cắt khối ở giữa câu**. Nay bộ cắt khối
            # chỉ cắt tại KHOẢNG LẶNG do VAD xác nhận (`LookaheadChunker._find_vad_cut`,
            # `batch_overlap_sec = 0.0`) nên mép khối không bao giờ rơi vào giữa từ. Giữ chúng lại
            # chỉ tạo ra lỗi MỚI — log thật 15:23:02:
            #     [SEG_BATCH] Ghép 11 từ giữ lại từ khối trước vào khối [74.27 → 86.37]
            #     [SEG_BATCH] Ngắt câu khối: "I can't see me" |
            #                 "loving nobody but you for all my This is Rebecca." | ...
            #     → dán liền hai câu của HAI KHỐI KHÁC NHAU thành một phụ đề vô nghĩa.
            #
            # NGUYÊN TẮC MỚI: **tin [ASR] tuyệt đối** — ASR tạo ra câu nào thì gửi ra phụ đề đúng
            # như vậy; Forced Aligner chỉ xác định mốc bắt đầu/kết thúc. Không cắt xén, không bù
            # đắp, không hàn gắn.

            # Bước 2E: Gom từ thành các câu phụ đề (trần từ/giây/ký tự lấy từ `LookaheadConfig`).
            # Bộ gom câu ưu tiên CÂU TRỌN VẸN; dấu câu đến từ `merge_source_text` vì aligner đã bỏ
            # hết dấu câu khi tokenize.
            if aligned_words and self._aligner_service is not None:
                try:
                    subtitles = self._aligner_service.group_words_to_subtitles(
                        aligned_words, language=align_lang, **self._batch_sub_opts
                    )
                except TypeError:
                    # Triển khai aligner cũ không nhận tham số trần ⇒ gọi kiểu tối thiểu.
                    subtitles = self._aligner_service.group_words_to_subtitles(
                        aligned_words, language=align_lang
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Gom câu phụ đề lỗi (dùng cả khối làm một câu): {exc}",
                        extra={"module_tag": "ASR"},
                    )
                    subtitles = [
                        SubtitleSentence(text=clean_text, start_time=0.0, end_time=chunk.duration)
                    ]
            else:
                # Aligner lỗi (hoặc không có từ nào): dùng trọn khối làm một câu rồi để
                # `split_oversized_sentences` tách lại theo dấu kết câu.
                subtitles = [
                    SubtitleSentence(text=clean_text, start_time=0.0, end_time=chunk.duration)
                ]

            # Bước 2E-bis: Tách các câu dài tại dấu ngắt vế (、 hoặc ,) khi số từ phía trước >= min_words
            min_comma_words = int(getattr(config.lookahead, "batch_sub_comma_min_words", 4) or 4)
            subtitles = ForcedAlignerService.split_subtitles_by_clause_comma(
                subtitles,
                language=align_lang,
                min_words=min_comma_words,
            )

            # LƯỚI AN TOÀN ngắt câu ở tầng VĂN BẢN (sự cố thật 2026-10-03: cả khối 3
            # câu hiện thành MỘT phụ đề dài khi aligner trả một item duy nhất).
            subtitles = ForcedAlignerService.split_oversized_sentences(
                subtitles,
                language=align_lang,
                max_chars=int(self._batch_sub_opts.get("sentence_max_chars", 160) or 160),
                max_words=int(self._batch_sub_opts.get("sentence_max_words", 24) or 24),
            )

            # CHẨN ĐOÁN: đây là text THẬT SỰ được gửi đi (khác `clean_text` của ASR). Không có log
            # này thì không thể biết vì sao phụ đề hiển thị khác bản phiên âm offline — đúng sự cố
            # 2026-10-02 (ASR log hoàn hảo mà phụ đề ra mảnh cụt không dấu câu).
            if subtitles:
                preview = " ⏐ ".join(f'"{s.text}"' for s in subtitles[:10])
                if len(subtitles) > 10:
                    preview += f" ⏐ …(+{len(subtitles) - 10})"
                logger.info(
                    f"[SEG_BATCH] Ngắt câu khối [{chunk.pts_start:.2f}s -> {chunk.pts_end:.2f}s] "
                    f"({len(subtitles)} phụ đề, align={align_lang}): {preview}",
                    extra={"module_tag": "ASR"},
                )

            # Bước 3 & 4: Dịch phụ đề khối (Batch Context Translation qua JSON) và gửi về Client
            pad_sec = max(0.0, self._start_pad_ms) / 1000.0
            user_sec = float(self.sync_offset_ms) / 1000.0
            shift = pad_sec + user_sec

            # Lọc trước các câu không thuộc quá khứ (tiết kiệm GPU),
            # với biên an toàn rộng (-3.0s) để không drop câu đầu khối khi video đang phát.
            candidate_subs: List[Tuple[int, SubtitleSentence, float, float]] = []
            for idx, sub in enumerate(subtitles):
                pts_start = max(0.0, chunk.pts_start + sub.start_time + shift)
                pts_end = max(pts_start + 0.5, chunk.pts_start + sub.end_time + shift)
                if pts_end < self.current_time - 3.0:
                    continue

                # LỌC ẢO GIÁC LẶP TỪ Ở TẦNG CÂU (đúng nguyên tắc: ngắt câu trước, bỏ câu hỏng sau).
                if is_repetition_hallucination(sub.text, min_reps=3):
                    logger.warning(
                        f"[SEG_BATCH] Bỏ 1 CÂU ảo giác lặp từ (các câu khác trong khối vẫn giữ): "
                        f"'{sub.text[:60]}'",
                        extra={"module_tag": "ASR"},
                    )
                    continue

                candidate_subs.append((idx, sub, pts_start, pts_end))

            # DỊCH GỘP TOÀN BỘ KHỐI: 1 lần gọi GPU duy nhất cho N câu
            candidate_texts = [sub.text for _, sub, _, _ in candidate_subs]
            t_tr = time.perf_counter()
            translated_texts = await self._translate_batch(candidate_texts)
            tr_ms = (time.perf_counter() - t_tr) * 1000.0

            if candidate_subs:
                preview_trans = " ⏐ ".join(f'"{t.strip()}"' for t in translated_texts[:10])
                if len(translated_texts) > 10:
                    preview_trans += f" ⏐ …(+{len(translated_texts) - 10})"
                logger.info(
                    f"Lookahead Batch Translation ({len(candidate_subs)} câu, {tr_ms:.0f}ms): {preview_trans}",
                    extra={"module_tag": "TRANSLATE"},
                )

            items_to_send = []
            emitted_end_audio: Optional[float] = None
            for (idx, sub, pts_start, pts_end), translated in zip(candidate_subs, translated_texts):
                if self._closed or seq != self._seek_seq:
                    break

                if not translated:
                    translated = sub.text

                if self._is_duplicate(pts_start, pts_end, sub.text):
                    continue

                self._sent_items.append((pts_start, pts_end, sub.text))
                if len(self._sent_items) > 512:
                    del self._sent_items[:128]

                self._recent_utterances.append({
                    "pts_start": pts_start,
                    "pts_end": pts_end,
                    "text": sub.text,
                    "translated": translated,
                })
                if len(self._recent_utterances) > 128:
                    del self._recent_utterances[:32]

                items_to_send.append({
                    "start_pts": round(pts_start, 3),
                    "end_pts": round(pts_end, 3),
                    "original_text": sub.text,
                    "translated_text": translated,
                })
                # Ghi mốc đã phát (audio space) + lịch sử cho bộ trim ranh giới dự phòng.
                sub_end_audio = chunk.pts_start + sub.end_time
                emitted_end_audio = (
                    sub_end_audio if emitted_end_audio is None
                    else max(emitted_end_audio, sub_end_audio)
                )
                # Đuôi TỪ đã phát (giữ ~40 từ) để trừ chồng lấn theo chuỗi ở khối kế tiếp.
                for tok in (sub.text or "").split():
                    norm_tok = normalize_for_dedup(tok)
                    if norm_tok:
                        self._batch_emitted_tail_norm.append(norm_tok)
                if len(self._batch_emitted_tail_norm) > 40:
                    del self._batch_emitted_tail_norm[:-40]
                self._batch_dedup.record_commit(sub.text)
                self._enqueue_tts(translated, pts_start, pts_end)
                self.utterances_sent += 1

            if items_to_send and not self._closed and seq == self._seek_seq:
                await self.send_json({
                    "type": "lookahead_subtitles",
                    "seek_id": self.active_seek_id,
                    "items": items_to_send,
                })

            # Ghi mốc nội dung đã phát (chỉ để chẩn đoán — không còn dùng để cắt mép khối).
            if seq == self._seek_seq and not self._closed and emitted_end_audio is not None:
                self._batch_prev_emitted_end_audio = emitted_end_audio

            self._commit_batch_progress(seq, chunk.next_read_pts, chunk.pts_end)
            # Đo thời gian xử lý TRỌN khối để cổng kiên nhẫn biết phải bắt đầu sớm bao nhiêu.
            proc_sec = max(0.0, time.perf_counter() - t_block)
            self._batch_proc_ema_sec = 0.5 * float(self._batch_proc_ema_sec) + 0.5 * proc_sec
            await self.send_status()

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

    async def _translate_batch(self, texts: List[str]) -> List[str]:
        """Dịch gộp cả khối câu phụ đề qua Batch Context Translation (Pipeline B)."""
        if not texts:
            return []
        engine = self.translation_engine
        if engine is None:
            return list(texts)
        try:
            if hasattr(engine, "translate_batch"):
                return await engine.translate_batch(texts, self.source_lang, self.target_lang)
            # Fallback nếu engine không có translate_batch
            results = []
            for t in texts:
                res = await self._translate(t)
                results.append(res or t)
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Lỗi dịch Lookahead batch: {exc}", extra={"module_tag": "TRANSLATE"})
            return list(texts)

    async def _translate(self, text: str, context: str = "") -> Optional[str]:
        """Dịch câu gốc đơn lẻ sang ngôn ngữ đích kèm ngữ cảnh (nếu có). None = không gửi (lỗi)."""
        engine = self.translation_engine
        if engine is None:
            return text
        try:
            if hasattr(engine, "translate_sentence"):
                result = await engine.translate_sentence(
                    text, self.source_lang, self.target_lang, context=context
                )
            else:
                try:
                    result = await engine.translate(
                        text, self.source_lang, self.target_lang, context=context
                    )
                except TypeError:
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
        norm = normalize_for_dedup(text)
        if not norm:
            return True
        for s0, s1, prev in self._sent_items[-64:]:
            if normalize_for_dedup(prev) == norm and abs(s0 - pts_start) < 1.0:
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
        # MARKER PHẢI THUỘC ĐÚNG THẾ HỆ SEEK HIỆN TẠI. Sau khi tua, `_ready_until_pts` có thể còn
        # giá trị của vị trí CŨ (do iteration đang chạy hoặc do race) — dùng nó sẽ báo "đã dịch
        # 232.9s" trong khi playhead ở 27s và KHÔNG có gì được xử lý ở vị trí mới, khiến video phát
        # mà không hoạt động gì (sự cố 2026-10-02). Thế hệ lệch ⇒ coi như chưa phủ gì.
        if self._ready_until_seq != self._seek_seq:
            ready_until = float(self.current_time)
        else:
            ready_until = float(self._ready_until_pts)
        ready_ahead = max(0.0, ready_until - self.current_time)

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

        # Sẵn sàng phát lại khi:
        # 1. Đã dịch trước đủ min_ready_ahead_sec (ít nhất 1 câu phủ mốc phát), HOẶC
        # 2. Đã quét VAD qua ít nhất 3.5s im lặng (vùng này không có tiếng nói để dịch).
        min_ready = float(la.min_ready_ahead_sec)
        if buffered_ahead > 0:
            min_ready = min(min_ready, max(1.0, buffered_ahead - 0.5))
        speech_ready = bool(ready_ahead >= min_ready) or bool(fed_ahead >= 3.5 and ready_ahead == 0.0)
        prebuffer_ready = speech_ready and tts_ready
        # Log khi TRẠNG THÁI SẴN SÀNG ĐỔI — đây là mốc quyết định "cho video phát hay chưa",
        # nếu không log thì rất khó chẩn đoán tại sao video cứ chờ hết timeout.
        if prebuffer_ready != self._last_ready_flag:
            self._last_ready_flag = prebuffer_ready
            logger.info(
                f"Lookahead prebuffer_ready={prebuffer_ready} "
                f"(ready_ahead={ready_ahead:.2f}s, fed_ahead={fed_ahead:.2f}s, target={target:.2f}s, "
                f"tts_ready={tts_ready}, tts_sent={self.tts_sent}, "
                f"buffered_ahead={buffered_ahead:.2f}s, currentTime={self.current_time:.2f}s, "
                f"seek_seq={self._seek_seq})",
                extra={"module_tag": "WS"},
            )

        await self._maybe_log_diagnostics(buffered_ahead, ready_ahead, fed_ahead)

        await self.send_json({
            "type": "lookahead_status",
            "seek_id": self.active_seek_id,
            "currentTime": round(self.current_time, 2),
            "buffered_ahead": round(buffered_ahead, 2),
            "buffered_end_pts": round(buffered_end, 2),
            # Mốc backend đã XỬ LÝ XONG (ASR + dịch) cho ĐÚNG thế hệ seek này. Client dùng mốc này
            # làm RÀNG BUỘC CỨNG: playhead không được vượt qua nó.
            "ready_until_pts": round(ready_until, 2),
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
        """Log NĂNG LỰC THẬT mỗi `diag_interval_sec`: byte vào, PCM ra, độ phủ phụ đề.

        Đây là thước đo để biết nút cổ chai nằm ở ĐÂU:
          * ``PCM/byte`` thấp (byte về nhiều mà PCM ra ít)  ⇒ tầng giải mã (demuxer).
          * ``giải mã`` chậm (PCM ra chậm) ⇒ tầng ghép nối/giải mã MSE.
          * ``đã dịch`` không nhích dù PCM có sẵn ⇒ tầng ASR/Aligner/Dịch.
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
        self._diag_bytes_in = self.bytes_in
        self._diag_decode_seconds = self.decode_seconds
        self._diag_media_seconds = self.media_seconds
        self._diag_decode_wall = self.decode_wall_sec

        logger.info(
            f"Trạng thái Lookahead | Playhead: {self.current_time:.1f}s | "
            f"Audio RAM: {self.timeline.total_stored_seconds():.1f}s | "
            f"Đệm trước: {buffered_ahead:.1f}s | "
            f"Đã dịch: {ready_ahead:.1f}s (đã qua ASR {fed_ahead:.1f}s, "
            f"giải mã {d_decode / span:.2f}x, PCM/byte {d_decode / max(1, d_bytes):.3f}, "
            f"im lặng-do-lấp {self.timeline.gap_filled_sec:.1f}s, "
            f"câu {self.utterances_sent})",
            extra={"module_tag": "ASR"},
        )

        if now - self._last_struct_at >= 60.0:
            self._last_struct_at = now
            logger.debug(
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
                self._batch_sub_opts["min_words"] = max(1, self._min_words_to_commit)

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
    media_start = header.get("media_start")
    media_end = header.get("media_end")
    await session.handle_audio_fragment(
        chunk_bytes=chunk_bytes,
        timestamp_offset=float(header.get("timestamp_offset", 0.0) or 0.0),
        mime_type=str(header.get("mime_type", "auto")),
        seek_id=str(header.get("seek_id") or session.active_seek_id),
        is_init=bool(header.get("is_init", False)),
        epoch=int(epoch) if epoch is not None else None,
        # Mảnh tải lại cho vùng đã buffer sẵn: byte có thể trùng mảnh cũ nên phải MIỄN dedup.
        refetched=bool(header.get("refetched", False)),
        #: Khoảng media THẬT trong `SourceBuffer.buffered` (trục trình duyệt) — dùng để neo PCM.
        media_start=float(media_start) if media_start is not None else None,
        media_end=float(media_end) if media_end is not None else None,
    )
