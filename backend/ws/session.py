"""Module quản lý trạng thái phiên kết nối WebSocket (SessionState).

Đặc điểm tối ưu cho 1 Session duy nhất:
- Cấu hình linh hoạt qua Pydantic aliased model.
- Khởi tạo đầy đủ chu trình: VADProcessor -> TranscribeEngine -> CommitManager -> Translation Queue -> TTS Queue.
- Hàng đợi bất đồng bộ có Back-pressure (maxsize=4) ngăn chặn tràn bộ nhớ.
- Fast Cleanup (< 200ms) giải phóng toàn bộ tài nguyên khi Client ngắt kết nối.

Sửa đổi theo kế hoạch triển khai (Revision 2):
- P1.7: đổi `translation_model` qua WS nay CÓ tác dụng (trước đây chỉ ghi vào dict).
- P1.8: đổi ASR model nay nạp trước rồi swap trong task nền, có thông báo trạng thái
  qua WS, không "nạp trộm" bên trong lần inference kế tiếp.
- P1.10: `apply_config()` là đường DUY NHẤT để đổi cấu hình phiên, dùng chung cho cả
  WS `set_config` và REST `/api/config` => REST không còn ghi vào config toàn cục rồi
  bỏ quên session đang chạy.
- Seam cho test: `init_components()` nhận engine tiêm vào để test tầng A không cần model.
"""

import asyncio
import time
import uuid
from typing import Optional, Dict, Any, Union
from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict, Field

from backend.config import config
from backend.asr.engine import TranscribeEngine
from backend.asr.registry import ModelRegistry
from backend.core.metrics import metrics_collector
from backend.vad.processor import VADProcessor
from backend.ws.connection import SafeWebSocketConnection
from backend.ws.serializers import make_model_status_msg
from backend.utils.logger import get_logger

logger = get_logger("ws.session")


class SessionConfigPayload(BaseModel):
    """Mô hình phân tích thông điệp cấu hình từ Extension (hỗ trợ cả camelCase và snake_case)."""
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    model_id: Optional[str] = Field(default=None, alias="modelId")
    asr_engine: Optional[str] = Field(default=None, alias="asrEngine")
    asr_model: Optional[str] = Field(default=None, alias="asrModel")
    source_lang: Optional[str] = Field(default=None, alias="sourceLang")
    target_lang: Optional[str] = Field(default=None, alias="targetLang")
    vad_engine: Optional[str] = Field(default=None, alias="vadEngine")
    vad_threshold: Optional[float] = Field(default=None, alias="vadThreshold")
    threshold: Optional[float] = None
    silence_duration_ms: Optional[int] = Field(default=None, alias="silenceDurationMs")
    vad_enabled: Optional[bool] = Field(default=None, alias="vadEnabled")
    tts_enabled: Optional[bool] = Field(default=None, alias="ttsEnabled")
    tts_voice: Optional[str] = Field(default=None, alias="ttsVoice")
    tts_speed: Optional[float] = Field(default=None, alias="ttsSpeed")
    tts_ref_audio: Optional[str] = Field(default=None, alias="ttsRefAudio")
    tts_ref_text: Optional[str] = Field(default=None, alias="ttsRefText")
    translation_model: Optional[str] = Field(default=None, alias="translationModel")

    # Các thông số phân câu
    split_on_stability: Optional[bool] = Field(default=None, alias="splitOnStability")
    enable_stability_split: Optional[bool] = None
    stability_duration_sec: Optional[float] = Field(default=None, alias="stabilityDurationSec")
    max_duration_sec: Optional[float] = Field(default=None, alias="maxDurationSec")
    max_chars: Optional[int] = Field(default=None, alias="maxChars")
    min_words_to_commit: Optional[int] = Field(default=None, alias="minWordsToCommit")
    # Cắt câu theo ĐỘ ỔN ĐỊNH ("stable_cut"): text đứng im đủ lâu thì chốt câu.
    stability_min_duration_sec: Optional[float] = Field(default=None, alias="stabilityMinDurationSec")
    stability_min_words: Optional[int] = Field(default=None, alias="stabilityMinWords")
    #: Chẩn đoán `[SEG_TRACE]`/`[SEG_CUT]` từng nhịp preview (mặc định TẮT).
    trace_stability: Optional[bool] = Field(default=None, alias="traceStability")
    #: GIỮ CÂU khi VAD báo END mà câu chưa đủ `stability_min_duration_sec` (mặc định BẬT).
    hold_short_sentence: Optional[bool] = Field(default=None, alias="holdShortSentence")

    # P2.3/P2.4: cho phép chỉnh từ popup
    preview_window_sec: Optional[float] = Field(default=None, alias="previewWindowSec")
    poll_interval_ms: Optional[int] = Field(default=None, alias="pollIntervalMs")
    # P3.0: phiên bản giao thức client khai báo (>=2 mới dùng binary frame cho TTS)
    protocol_version: Optional[int] = Field(default=None, alias="protocolVersion")


# ──────────────────────────────────── Chuẩn hoá giá trị config (MỘT chỗ duy nhất)
#: Các quy ước dưới đây từng bị viết lặp ở 2–3 nơi (`main.py`, `session.py`,
#: `lookahead_handler.py`) ⇒ sửa một chỗ là lệch hai chỗ còn lại. Xem
#: `report/audit/26_RA_SOAT_CODE_CHET_VA_CHONG_CHEO.md` §5.9.

#: Trần kẹp offset đồng bộ phụ đề/TTS (ms) — popup cũng dùng đúng con số này.
SYNC_OFFSET_LIMIT_MS = 1500.0


def pick_vad_threshold(parsed: Any) -> Optional[float]:
    """Ngưỡng VAD từ payload: ưu tiên `vad_threshold`, rơi về alias cũ `threshold`.

    Alias `threshold` là tên cũ còn sót lại; popup hiện gửi `vad_threshold`. Giữ nhánh
    fallback để client cũ không bị bỏ qua.
    """
    if parsed is None:
        return None
    raw = parsed.vad_threshold if parsed.vad_threshold is not None else parsed.threshold
    return None if raw is None else float(raw)


def off_means_none_ms(raw_ms: Any) -> Optional[int]:
    """`0` (hoặc âm) nghĩa là "TẮT" ⇒ trả `None` để engine dùng mặc định của chính nó.

    Quy ước này ghi ở `report/audit/19_...`: kéo slider về 0 phải TẮT ngắt câu theo khoảng
    lặng, KHÔNG phải "khoảng lặng dài 0 ms".
    """
    value = int(raw_ms)
    return value if value > 0 else None


def clamp_min_words(raw: Any) -> int:
    """`min_words_to_commit` không bao giờ âm."""
    return max(0, int(raw))


def clamp_sync_offset_ms(raw: Any) -> Optional[float]:
    """Kẹp offset đồng bộ vào ±`SYNC_OFFSET_LIMIT_MS`. Trả `None` nếu không parse được.

    Trả `None` (thay vì ném) vì đây là giá trị người dùng chỉnh trong popup — payload hỏng
    không được làm sập phiên.
    """
    if raw is None:
        return None
    try:
        return max(-SYNC_OFFSET_LIMIT_MS, min(SYNC_OFFSET_LIMIT_MS, float(raw)))
    except (TypeError, ValueError):
        return None


def spawn_background_task(coro: Any, sink: set) -> None:
    """Chạy `coro` thành task nền và giữ tham chiếu trong `sink` để GC không thu hồi giữa đường.

    DÙNG CHUNG cho `SessionState` và `LookaheadSessionState`: hai lớp trước đây có bản `_spawn`
    **giống hệt nhau từng byte** (khác đúng một dòng log). Xem
    `report/audit/26_RA_SOAT_CODE_CHET_VA_CHONG_CHEO.md` §5.7.

    Không có event loop (test đồng bộ) ⇒ bỏ qua an toàn thay vì ném.
    """
    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        logger.debug("Không có event loop đang chạy — bỏ qua task nền.", extra={"module_tag": "WS"})
        return
    sink.add(task)
    task.add_done_callback(sink.discard)


class SessionConfig:
    """Quản lý cấu hình động theo từng phiên làm việc."""

    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = {
            "source_lang": config.asr.language,
            "target_lang": config.translation.target_lang,
            "translation_model": getattr(config.translation, "base", "tencent"),
            "asr_engine": ModelRegistry.get_instance().get_active_model_key(),
            "vad_engine": config.vad.vad_engine,
            # None = dùng đúng mặc định của từng VAD engine (popup hiển thị "0/off").
            "vad_threshold": config.vad.effective_threshold,
            "silence_duration_ms": config.vad.effective_silence_ms,
            "vad_enabled": config.vad.enabled,
            "min_words_to_commit": config.sentence.min_words_to_commit,
            "stability_duration_sec": config.sentence.stability_duration_sec,
            "stability_min_duration_sec": config.sentence.stability_min_duration_sec,
            "stability_min_words": config.sentence.stability_min_words,
            "split_on_stability": config.sentence.split_on_stability,
            "trace_stability": config.sentence.trace_stability,
            "hold_short_sentence": config.sentence.hold_short_sentence,
            "tts_enabled": config.tts.enabled,
            "tts_voice": config.tts.default_voice,
            "tts_speed": config.tts.speed,
            "tts_ref_audio": "",
            "tts_ref_text": "",
        }
        if initial_config:
            self._data.update(initial_config)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def update(self, new_data: Dict[str, Any]) -> None:
        self._data.update(new_data)


class SessionState:
    """Theo dõi trạng thái và các thành phần Pipeline cho 1 kết nối WebSocket."""

    def __init__(self, ws: Union[WebSocket, SafeWebSocketConnection]):
        if isinstance(ws, SafeWebSocketConnection):
            self.connection: SafeWebSocketConnection = ws
        else:
            self.connection: SafeWebSocketConnection = SafeWebSocketConnection(ws)

        self.session_id: str = str(uuid.uuid4())
        self.config: SessionConfig = SessionConfig()
        self.connected_at: float = time.time()
        self.chunk_index: int = 0

        # Các thành phần cốt lõi của Pipeline
        self.asr_engine: Optional[TranscribeEngine] = None
        self.vad_processor: Optional[VADProcessor] = None
        self.translation_queue: Optional[asyncio.Queue] = None
        self.tts_queue: Optional[asyncio.Queue] = None
        self._bg_tasks: set = set()
        # P3.0/P3.1: client khai báo protocol_version >= 2 thì mới gửi binary frame cho
        # TTS. Mặc định 1 => giữ nguyên hành vi JSON+base64 nên extension CŨ vẫn chạy.
        self.protocol_version: int = 1

    @property
    def supports_binary_tts(self) -> bool:
        """Chỉ dùng binary frame khi CẢ HAI phía đều hỗ trợ phiên bản >= 2."""
        return self.protocol_version >= 2 and int(config.ws.protocol_version) >= 2

    @property
    def supports_compact_payload(self) -> bool:
        """F-30: client khai báo >= 3 thì gửi payload GỌN (không field trùng lặp).

        Client cũ (v1/v2) vẫn nhận payload đầy đủ ⇒ không phá vỡ tương thích.
        """
        return self.protocol_version >= 3 and int(config.ws.protocol_version) >= 3

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        """Gửi JSON payload qua WebSocket an toàn.

        ⚠️ Đây là lớp MỎNG trong chuỗi 3 lớp (`SafeWebSocketConnection.send_json` →
        `SessionState.send_json` → `LookaheadSessionState.send_json`). CỐ Ý giữ cả ba, KHÔNG gộp:
        mỗi lớp làm một việc khác nhau —
          * `connection`: tuần tự hoá, đo metric `ws.send_*`, chặn khi socket đã đóng;
          * `SessionState` (đây): API của lớp phiên, để handler không phải chạm `session.connection`;
          * `LookaheadSessionState`: thoát SỚM khi `_closed` (phiên Lookahead đóng trước khi
            socket đóng), tránh ghi thêm frame vào một phiên đã kết thúc.
        Gộp lại sẽ phải sửa mọi call-site và làm mất ý nghĩa "phiên đã đóng" của Pipeline B.
        Xem `report/audit/26_RA_SOAT_CODE_CHET_VA_CHONG_CHEO.md` §5.7.
        """
        return await self.connection.send_json(payload)

    def _spawn(self, coro) -> None:
        """Chạy task nền và giữ tham chiếu (tránh bị GC thu hồi giữa đường).

        Uỷ quyền cho `spawn_background_task` — dùng chung với `LookaheadSessionState`.
        """
        spawn_background_task(coro, self._bg_tasks)

    def init_components(self, asr_engine=None, vad_engine=None) -> None:
        """Khởi tạo toàn bộ các thành phần ASR, VAD và hàng đợi bất đồng bộ.

        `asr_engine` / `vad_engine` cho phép TIÊM engine giả trong test tầng A
        (không cần nạp model thật) — xem report/audit/03_KE_HOACH_TRIEN_KHAI.md §7.2.
        """
        self.asr_engine = asr_engine or TranscribeEngine(session_id=self.session_id)

        # FIX-02/FIX-03: trần hàng đợi đọc từ config (trước đây hardcode 4). Hàng đợi này
        # CHỈ chứa câu final, nên khi đầy thì bản dịch/lồng tiếng của câu đó bị mất hẳn.
        # Trần vẫn bị chặn trên (RAM bound) nhưng rộng hơn để overflow gần như không xảy ra.
        self.translation_queue = asyncio.Queue(
            maxsize=max(1, int(getattr(config.ws, "translation_queue_maxsize", 32) or 32))
        )
        self.tts_queue = asyncio.Queue(
            maxsize=max(1, int(getattr(config.ws, "tts_queue_maxsize", 32) or 32))
        )

        self.vad_processor = VADProcessor(
            sample_rate=config.vad.sample_rate,
            vad_engine=self.config.get("vad_engine", config.vad.vad_engine),
            threshold=self.config["vad_threshold"],
            # None (hoặc 0) = để chính VAD quyết định theo config riêng của nó.
            silence_duration_ms=self.config["silence_duration_ms"],
            enabled=self.config["vad_enabled"],
            on_speech_chunk=self.asr_engine.feed_audio,
            on_speech_start=self.asr_engine.on_speech_start,
            on_speech_end=self.asr_engine.on_speech_end,
            engine_override=vad_engine,
        )

    def record_chunk(self, chunk_idx: Optional[int] = None) -> int:
        """Cập nhật và theo dõi số thứ tự chunk nhận được từ client."""
        if chunk_idx is not None:
            if self.chunk_index > 0 and chunk_idx > self.chunk_index + 1:
                dropped = chunk_idx - (self.chunk_index + 1)
                logger.warning(f"Session {self.session_id}: Phát hiện mất {dropped} chunk âm thanh!", extra={"module_tag": "WS"})
                metrics_collector.increment_counter("ws.audio_gap_chunks", dropped)
            self.chunk_index = chunk_idx
        else:
            self.chunk_index += 1
        return self.chunk_index

    # ------------------------------------------------------------ model switching
    def _schedule_asr_model_switch(self, model_key: str) -> None:
        """P1.8: nạp trước model ASR rồi swap, có thông báo trạng thái qua WS (tự tải nếu thiếu file).

        ⚠️ CỐ Ý KHÁC bản của Pipeline B (`lookahead_handler.py::_schedule_asr_model_switch`):
          * Ở đây MỌI nhánh từ chối đều GỬI `model_status(state="error")` cho client, và phân biệt
            `downloading` vs `loading` theo `needs_download`;
          * Bản Lookahead chỉ LOG rồi `return`, luôn gửi `loading`, và ném lỗi TRONG task.
        Gộp hai bên sẽ đổi thông báo mà người dùng thấy ở một trong hai pipeline ⇒ KHÔNG gộp.
        Xem `report/audit/26_RA_SOAT_CODE_CHET_VA_CHONG_CHEO.md` §5.7.
        """
        from backend.asr import lifecycle as asr_lifecycle
        from backend.asr.registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        clean_key = (model_key or "").strip().lower()
        if not registry.has_model(clean_key):
            logger.warning(f"Model ASR '{model_key}' không có trong catalog — bỏ qua.", extra={"module_tag": "WS"})
            self._spawn(self.send_json(
                make_model_status_msg("asr", "error", model_key, "Không có trong models.yaml")
            ))
            return

        try:
            model_path = registry.resolve_model_path(clean_key)
        except Exception as exc:  # noqa: BLE001
            model_path = ""
            logger.warning(f"Không resolve được GGUF cho ASR '{clean_key}': {exc}", extra={"module_tag": "WS"})

        needs_download = asr_lifecycle.needs_download(clean_key)
        if needs_download and not asr_lifecycle.auto_download_enabled():
            logger.warning(
                f"Model ASR '{clean_key}' chưa có file GGUF cục bộ "
                f"({model_path or 'không xác định'}) và auto_download đang tắt — GIỮ NGUYÊN model "
                f"hiện tại '{config.asr.active_model}'.",
                extra={"module_tag": "WS"},
            )
            self._spawn(self.send_json(
                make_model_status_msg("asr", "error", clean_key, f"Chưa có file GGUF cục bộ: {model_path}")
            ))
            return

        if asr_lifecycle.is_busy() and not asr_lifecycle.is_busy(clean_key):
            self._spawn(self.send_json(
                make_model_status_msg(
                    "asr", "error", clean_key,
                    f"Đang tải/nạp model ASR khác ({asr_lifecycle.status().get('model')})",
                )
            ))
            return

        self.config["asr_engine"] = clean_key

        async def _run() -> None:
            await self.send_json(
                make_model_status_msg(
                    "asr", "downloading" if needs_download else "loading", clean_key
                )
            )
            try:
                await asr_lifecycle.activate_model(clean_key)
                logger.info(f"Session {self.session_id[:8]}: Swapped ASR -> '{clean_key}'", extra={"module_tag": "WS"})
                await self.send_json(make_model_status_msg("asr", "ready", clean_key))
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Chuyển ASR model sang '{clean_key}' thất bại: {exc}", exc_info=True, extra={"module_tag": "WS"})
                await self.send_json(
                    make_model_status_msg("asr", "error", clean_key, str(exc))
                )

        self._spawn(_run())

    def _schedule_translation_model_switch(self, model_request: str) -> None:
        """P1.7 + F-50: đổi model dịch có tác dụng thật, nạp trong task nền.

        Trước đây `translation_model` từ WS chỉ được ghi vào dict và KHÔNG BAO GIỜ
        được áp dụng (G1). Nay dùng chung `translation.lifecycle` với đường REST: thiếu file
        thì tải trước (nếu bật `auto_download`) → nạp model mới → chỉ khi thành công mới
        đổi config; lỗi thì model đang chạy vẫn nguyên vẹn.
        """
        from backend.translation import lifecycle as trans_lifecycle
        from backend.translation.registry import TranslationModelRegistry

        registry = TranslationModelRegistry.get_instance()
        canonical = registry.resolve_key(model_request)
        info = registry.get_model(canonical)
        if info is None or not registry.is_known(model_request):
            logger.warning(f"Model dịch '{model_request}' không có trong catalog — bỏ qua.", extra={"module_tag": "WS"})
            self._spawn(self.send_json(
                make_model_status_msg(
                    "translation", "error", model_request,
                    "Không có trong translation_models.yaml",
                )
            ))
            return

        # Kiểm tra file trước để báo lỗi rõ thay vì nạp thất bại giữa đường (G9).
        try:
            gguf_path = registry.resolve_gguf_path(canonical)
        except Exception as exc:  # noqa: BLE001
            gguf_path = ""
            logger.warning(f"Không resolve được GGUF cho '{canonical}': {exc}", extra={"module_tag": "WS"})

        needs_download = trans_lifecycle.needs_download(canonical)
        if needs_download and not trans_lifecycle.auto_download_enabled():
            logger.warning(
                f"Model dịch '{canonical}' chưa có file GGUF cục bộ "
                    f"({gguf_path or 'không xác định'}) và auto_download đang tắt — GIỮ NGUYÊN model "
                        f"hiện tại '{config.translation.base}'.",
                extra={"module_tag": "WS"},
            )
            self._spawn(self.send_json(
                make_model_status_msg(
                    "translation", "error", canonical, f"Chưa có file GGUF cục bộ: {gguf_path}"
                )
            ))
            return

        if trans_lifecycle.is_busy() and not trans_lifecycle.is_busy(canonical):
            self._spawn(self.send_json(
                make_model_status_msg(
                    "translation", "error", canonical,
                    f"Đang tải/nạp model dịch khác ({trans_lifecycle.status().get('model')})",
                )
            ))
            return

        self.config["translation_model"] = canonical

        async def _run() -> None:
            await self.send_json(
                make_model_status_msg(
                    "translation", "downloading" if needs_download else "loading", canonical
                )
            )
            try:
                await trans_lifecycle.activate_model(canonical)
                logger.info(f"Session {self.session_id[:8]}: Swapped translation -> '{canonical}'", extra={"module_tag": "WS"})
                await self.send_json(
                    make_model_status_msg("translation", "ready", canonical)
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Chuyển model dịch sang '{canonical}' thất bại: {exc}", exc_info=True, extra={"module_tag": "WS"})
                await self.send_json(
                    make_model_status_msg("translation", "error", canonical, str(exc))
                )

        self._spawn(_run())

    # ------------------------------------------------------------ apply config
    def apply_config(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Áp dụng và cập nhật cấu hình runtime ngay lập tức.

        Trả về dict các thay đổi đã thực sự áp dụng (dùng cho log/test P1.11).
        Đây là ĐƯỜNG DUY NHẤT đổi cấu hình phiên — REST `/api/config` gọi lại hàm này
        (P1.10) nên thay đổi qua REST có hiệu lực ngay trên phiên đang chạy.
        """
        applied: Dict[str, Any] = {}
        try:
            parsed = SessionConfigPayload.model_validate(raw_data)
        except Exception as e:
            logger.warning(f"Lỗi validate config payload: {e}", extra={"module_tag": "WS"})
            return applied

        # P3.0: ghi nhận phiên bản giao thức client (không được làm lỗi nếu thiếu)
        if parsed.protocol_version is not None:
            try:
                self.protocol_version = int(parsed.protocol_version)
                applied["protocol_version"] = self.protocol_version
            except (TypeError, ValueError):
                pass

        updates: Dict[str, Any] = {}
        if parsed.source_lang is not None:
            updates["source_lang"] = parsed.source_lang
        if parsed.target_lang is not None:
            updates["target_lang"] = parsed.target_lang
            config.translation.target_lang = parsed.target_lang

        # VAD settings: không cho đổi VAD engine giữa phiên đang phát audio
        if parsed.vad_engine is not None:
            current_vad = self.config.get("vad_engine", config.vad.vad_engine)
            if self.chunk_index > 0 and parsed.vad_engine.lower().strip() != str(current_vad).lower().strip():
                logger.warning(
                    f"Session {self.session_id[:8]}: Bỏ qua yêu cầu đổi VAD sang '{parsed.vad_engine}' khi session đang phát audio (chỉ được đổi trước khi bắt đầu session).",
                    extra={"module_tag": "WS"},
                )
            else:
                updates["vad_engine"] = parsed.vad_engine
        vad_th = pick_vad_threshold(parsed)
        if vad_th is not None:
            updates["vad_threshold"] = vad_th
        if parsed.silence_duration_ms is not None:
            # 0 (hoặc âm) = "off" ⇒ để VAD dùng mặc định trong docs của nó.
            updates["silence_duration_ms"] = off_means_none_ms(parsed.silence_duration_ms)
        if parsed.vad_enabled is not None:
            updates["vad_enabled"] = bool(parsed.vad_enabled)

        # TTS settings
        if parsed.tts_enabled is not None:
            updates["tts_enabled"] = bool(parsed.tts_enabled)
        if parsed.tts_voice is not None:
            updates["tts_voice"] = parsed.tts_voice
        if parsed.tts_speed is not None:
            updates["tts_speed"] = float(parsed.tts_speed)
        if parsed.tts_ref_audio is not None:
            updates["tts_ref_audio"] = parsed.tts_ref_audio
        if parsed.tts_ref_text is not None:
            updates["tts_ref_text"] = parsed.tts_ref_text

        self.config.update(updates)
        applied.update(updates)

        # ---- Không cho phép đổi ASR Model giữa phiên đang phát audio ----
        target_asr = parsed.asr_model or parsed.asr_engine or parsed.model_id
        if target_asr:
            target_asr = target_asr.lower().strip()
            registry = ModelRegistry.get_instance()
            if self.chunk_index > 0 and registry.has_model(target_asr) and target_asr != registry.get_active_model_key():
                logger.warning(
                    f"Session {self.session_id[:8]}: Bỏ qua yêu cầu đổi ASR sang '{target_asr}' khi session đang phát audio (chỉ được đổi trước khi bắt đầu session).",
                    extra={"module_tag": "WS"},
                )
            elif registry.has_model(target_asr) and target_asr != registry.get_active_model_key():
                registry.set_active_model_key(target_asr)
                self.config["asr_engine"] = target_asr
                applied["asr_engine"] = target_asr
                self._schedule_asr_model_switch(target_asr)
                logger.info(f"Session {self.session_id[:8]}: Switching ASR -> '{target_asr}'", extra={"module_tag": "WS"})
            elif not registry.has_model(target_asr):
                logger.warning(f"Session {self.session_id[:8]}: ASR model '{target_asr}' không tồn tại", extra={"module_tag": "WS"})

        if self.asr_engine and "source_lang" in self.config:
            self.asr_engine.set_language(self.config["source_lang"])

        # ---- Không cho phép đổi Translation Model giữa phiên đang phát audio ----
        if parsed.translation_model is not None:
            requested = str(parsed.translation_model).strip().lower()
            from backend.translation.registry import TranslationModelRegistry
            canonical = TranslationModelRegistry.get_instance().resolve_key(requested)
            if self.chunk_index > 0 and canonical != getattr(config.translation, "base", None):
                logger.warning(
                    f"Session {self.session_id[:8]}: Bỏ qua yêu cầu đổi model dịch sang '{canonical}' khi session đang phát audio (chỉ được đổi trước khi bắt đầu session).",
                    extra={"module_tag": "WS"},
                )
            elif canonical != getattr(config.translation, "base", None):
                self._schedule_translation_model_switch(requested)
                applied["translation_model"] = canonical
            else:
                self.config["translation_model"] = canonical



        # ---- Cập nhật phân câu (Sentence segmentation) ----
        if self.asr_engine:
            sentence_updates: Dict[str, Any] = {}
            if parsed.split_on_stability is not None:
                sentence_updates["split_on_stability"] = parsed.split_on_stability
            elif parsed.enable_stability_split is not None:
                sentence_updates["split_on_stability"] = parsed.enable_stability_split
            if parsed.stability_duration_sec is not None:
                sentence_updates["stability_duration_sec"] = parsed.stability_duration_sec
            if parsed.max_duration_sec is not None:
                sentence_updates["max_duration_sec"] = parsed.max_duration_sec
            if parsed.max_chars is not None:
                sentence_updates["max_chars"] = parsed.max_chars
            if parsed.min_words_to_commit is not None:
                self.config["min_words_to_commit"] = parsed.min_words_to_commit
                sentence_updates["min_words_to_commit"] = parsed.min_words_to_commit
            if parsed.stability_min_duration_sec is not None:
                sentence_updates["stability_min_duration_sec"] = parsed.stability_min_duration_sec
            if parsed.stability_min_words is not None:
                sentence_updates["stability_min_words"] = parsed.stability_min_words
            if parsed.trace_stability is not None:
                sentence_updates["trace_stability"] = bool(parsed.trace_stability)
                self.config["trace_stability"] = bool(parsed.trace_stability)
            if parsed.hold_short_sentence is not None:
                sentence_updates["hold_short_sentence"] = bool(parsed.hold_short_sentence)
                self.config["hold_short_sentence"] = bool(parsed.hold_short_sentence)

            if sentence_updates:
                self.asr_engine.update_sentence_config(**sentence_updates)
                applied.update(sentence_updates)

            # P2.3/P2.4: chỉnh cửa sổ preview và nhịp poll runtime
            if parsed.preview_window_sec is not None:
                config.asr.preview_window_sec = float(parsed.preview_window_sec)
                applied["preview_window_sec"] = config.asr.preview_window_sec
            if parsed.poll_interval_ms is not None:
                v = max(50, int(parsed.poll_interval_ms))
                config.asr.poll_interval_ms = v
                self.asr_engine.poll_interval_ms = v
                # P2.4b: reset nhịp hiệu dụng để cấu hình mới có hiệu lực ngay.
                self.asr_engine._eff_poll_interval = v / 1000.0
                self.asr_engine._preview_durations.clear()
                applied["poll_interval_ms"] = v

        if self.vad_processor:
            self.vad_processor.update_config(
                vad_engine=self.config.get("vad_engine"),
                threshold=self.config.get("vad_threshold"),
                silence_duration_ms=self.config.get("silence_duration_ms"),
                enabled=self.config.get("vad_enabled"),
            )

        return applied

    async def drain_queues(self, timeout: float = 0.2) -> None:
        """Cho phép các tác vụ translation và tts đang xử lý dở được hoàn tất nhanh chóng."""
        drain_tasks = []
        if self.translation_queue and not self.translation_queue.empty():
            drain_tasks.append(self.translation_queue.join())
        if self.tts_queue and not self.tts_queue.empty():
            drain_tasks.append(self.tts_queue.join())
        if drain_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*drain_tasks), timeout=timeout)
            except (asyncio.TimeoutError, Exception):
                pass

    async def cancel_background_tasks(self) -> None:
        """Huỷ mọi task nền của phiên (nạp model, prewarm...)."""
        for t in list(self._bg_tasks):
            if not t.done():
                t.cancel()
        if self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)
        self._bg_tasks.clear()

    async def cleanup(self) -> None:
        """Giải phóng triệt để tài nguyên phiên làm việc trong < 200ms."""
        t0 = time.perf_counter()

        await self.cancel_background_tasks()

        if self.vad_processor:
            self.vad_processor.force_end()

        if self.asr_engine:
            # P1.8/T2.6: huỷ inference đang chạy để session sau không phải chờ hết.
            await self.asr_engine.cancel_inference()
            await self.asr_engine.cleanup()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics_collector.record_metric("session", "cleanup_ms", elapsed_ms)
        logger.info(f"Session {self.session_id[:8]}: Cleanup hoàn tất ({elapsed_ms:.1f}ms)", extra={"module_tag": "WS"})
