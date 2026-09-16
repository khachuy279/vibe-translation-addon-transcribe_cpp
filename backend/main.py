"""Điểm khởi chạy chính (Entry Point) của Hệ Thống Backend Real-Time Audio Processing.

Kiến trúc:
- FastAPI Web Framework với Lifespan quản lý nạp/giải phóng mô hình GPU/CPU.
- Hỗ trợ đầy đủ CORS cho Firefox, Chrome, Edge Extension và Localhost.
- WebSocket Streaming Endpoint: /ws (tối ưu hóa độ trễ cho 1 session).
- Quản lý REST API: Hot-switch ASR/Translation/VAD/TTS runtime.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Đảm bảo đường dẫn gốc dự án trong sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Tối ưu hóa Intel OpenMP / PyTorch / MKL để triệt tiêu hiện tượng busy-spin gây 100% CPU trên đa nhân
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Tắt thanh tiến trình tqdm của HuggingFace để không làm rác log console
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Thiết lập giới hạn luồng PyTorch CPU sớm nhất có thể
try:
    import torch
    torch.set_num_threads(2)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(2)
except ImportError:
    pass

# Windows CUDA DLL setup
from backend.utils.cuda import setup_cuda_dll_paths
setup_cuda_dll_paths()

from backend.config import config, SUPPORTED_LANGUAGES
from backend.vad import SUPPORTED_VAD_ENGINES, VADProcessor
from backend.asr.registry import ModelRegistry
from backend.asr.engine import TranscribeEngine
from backend.translation.engine import GGUFTranslationEngine, get_translation_engine, reset_translation_engine
from backend.translation.registry import TranslationModelRegistry
from backend.translation import hotswap as translation_hotswap
from backend.tts import VoiceManager, OmniVoiceTTS, get_tts_engine
from backend.core.metrics import metrics_collector
from backend.ws.handler import handle_ws
from backend.utils.logger import get_logger

logger = get_logger("main")

_background_tasks: set = set()


def _log_asr_backend_at_startup() -> None:
    """Ghi log MỘT DÒNG nêu rõ ASR sẽ chạy backend nào, từ nguồn native nào.

    Lịch sử: trước đây hàm này tên `_warn_if_cuda_provider_missing()` và khẳng định
    *"KHÔNG dùng được CUDA cho ASR trên Windows"*. Kết luận đó **đã hết đúng**: dự án tự
    dựng được bundle CUDA trong `bin/` (xem `report/audit/KE_HOACH_FIX_LOI_Hy3.md` §4.1.2),
    và PyPI `transcribe-cpp-native-cu12` vẫn chỉ là name-reservation.

    Hàm này KHÔNG quyết định backend — `backend/asr/native.py::resolve_backend()` làm việc đó
    (có fallback + log). Ở đây chỉ tóm tắt trạng thái để người vận hành thấy ngay lúc khởi động.
    """
    try:
        from backend.asr import native as asr_native

        info = asr_native.runtime_info()
        requested = (config.asr.backend or "auto").strip().lower()
        avail = info.get("available_backends") or []
        effective = asr_native.resolve_backend(config.asr.backend)

        bundle = info.get("native_bundle_dir")
        source_desc = f"bundle bin/ ({info.get('native_bundle_source')})" if bundle else "wheel đã cài"
        line = (
            f"[STARTUP] ASR backend: yêu cầu='{requested}' → thực tế='{effective}' | "
            f"native: {source_desc} | có sẵn: {', '.join(avail) or 'không rõ'} | "
            f"device: {info.get('devices')}"
        )
        if effective != requested and requested not in ("auto",):
            logger.warning(line, extra={"module_tag": "ASR"})
        else:
            logger.info(line, extra={"module_tag": "ASR"})

        if "cuda" not in avail:
            logger.info(
                "[STARTUP] ASR: không có backend CUDA trong thư viện native đang nạp. "
                "Đặt bundle có `ggml-cuda.dll` vào bin/ để bật CUDA, hoặc đặt "
                "asr.backend='vulkan' để chỉ định rõ.",
                extra={"module_tag": "ASR"},
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Bỏ qua ghi log backend ASR lúc khởi động: {exc}", extra={"module_tag": "MAIN"})


def _asr_runtime_info() -> Dict[str, Any]:
    """Thông tin backend ASR thực tế đang dùng (để /health xác nhận bằng mắt).

    Lưu ý: `provider` là tên provider PyPI (entry point). Khi chạy bằng bundle cục bộ
    trong `bin/` (qua `TRANSCRIBE_LIBRARY`), binding đi theo đường "dev-tree" nên
    `native_provider()` trả `None` — đó KHÔNG phải lỗi. Dùng `native_source`/
    `native_bundle_dir` để biết thư viện native đang nạp thực sự đến từ đâu.
    """
    info: Dict[str, Any] = {
        "model": None,
        "backend": None,
        "backend_requested": (config.asr.backend or "auto"),
        "provider": None,
        "native_source": None,
        "native_bundle_dir": None,
        "library_path": None,
        "available_backends": [],
        "cuda_backend_available": None,
    }
    try:
        import transcribe_cpp
        info["provider"] = transcribe_cpp.native_provider()
        try:
            info["library_path"] = transcribe_cpp.library_path()
        except Exception:  # noqa: BLE001
            pass
        try:
            info["cuda_backend_available"] = bool(transcribe_cpp.backend_available("cuda"))
        except Exception:
            info["cuda_backend_available"] = False
    except Exception:
        pass
    try:
        from backend.asr import native as asr_native

        rt = asr_native.runtime_info()
        info["native_source"] = rt.get("native_bundle_source")
        info["native_bundle_dir"] = rt.get("native_bundle_dir")
        info["available_backends"] = rt.get("available_backends") or []
        info["devices"] = rt.get("devices")
        if not info["provider"]:
            # Đang chạy bundle cục bộ — nêu rõ để không ai tưởng là thiếu provider.
            info["provider"] = "local-bin" if rt.get("native_bundle_dir") else None
    except Exception:
        pass
    try:
        from backend.core.gpu_scheduler import arbiter_status

        info["gpu_arbiter"] = arbiter_status()
    except Exception:
        pass
    try:
        model = TranscribeEngine._shared_model
        if model is not None:
            info["model"] = TranscribeEngine._shared_model_key
            info["backend"] = getattr(model, "backend", "unknown")
            info["supports_streaming"] = bool(
                getattr(getattr(model, "capabilities", None), "supports_streaming", False)
            )
    except Exception:
        pass
    return info


def track_background_task(coro, name: str = "background_task") -> asyncio.Task:
    """Tạo và theo dõi tác vụ bất đồng bộ trong background tránh bị Garbage Collector thu hồi sớm."""
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(
        lambda t: (
            _background_tasks.discard(t),
            logger.error(f"Task '{name}' gặp lỗi: {t.exception()}", exc_info=t.exception(), extra={"module_tag": "MAIN"})
            if not t.cancelled() and t.exception()
            else None,
        )
    )
    return task


async def _activate_translation_model_bg(canonical_key: str, allow_download: bool = True) -> None:
    """Tác vụ nền: tải (nếu cần) + nạp model dịch rồi thông báo cho các phiên đang chạy.

    Trạng thái được `translation.hotswap` giữ lại để popup hỏi tiến độ qua `/api/config`.
    """
    from backend.ws.handler import get_active_sessions

    try:
        await translation_hotswap.run_reserved(canonical_key, allow_download=allow_download)
    except Exception:
        # hotswap đã log + lưu trạng thái "error" (kèm thông báo) cho popup đọc.
        return

    for sess in get_active_sessions():
        try:
            sess.config["translation_model"] = canonical_key
        except Exception as e:  # noqa: BLE001
            logger.debug(
                f"Không cập nhật được translation_model cho session {getattr(sess, 'session_id', '?')}: {e}",
                extra={"module_tag": "MAIN"},
            )


class SwitchModelRequest(BaseModel):
    """Mô hình nhận request cập nhật cấu hình nóng qua REST API."""
    model_id: Optional[str] = None
    asr_engine: Optional[str] = None
    vad_engine: Optional[str] = None
    vad_threshold: Optional[float] = None
    silence_duration_ms: Optional[int] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    translation_model: Optional[str] = None
    min_words_to_commit: Optional[int] = None
    tts_enabled: Optional[bool] = None
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None


def _prewarm_asr() -> None:
    """Nạp + pre-warm ASR (blocking; gọi qua asyncio.to_thread)."""
    try:
        TranscribeEngine().prewarm()
        logger.info("[STARTUP] ASR model đã được nạp & pre-warm thành công!", extra={"module_tag": "ASR"})
    except Exception as e:
        logger.warning(f"[STARTUP] Cảnh báo pre-warm ASR: {e}", exc_info=True, extra={"module_tag": "ASR"})


def _prewarm_translation() -> None:
    """Nạp model dịch (blocking; gọi qua asyncio.to_thread)."""
    try:
        get_translation_engine().load_model()
        logger.info("[STARTUP] Translation model đã được nạp & pre-warm thành công!", extra={"module_tag": "TRANSLATE"})
    except Exception as e:
        logger.warning(f"[STARTUP] Cảnh báo pre-warm Translation: {e}", exc_info=True, extra={"module_tag": "TRANSLATE"})


def _prewarm_vad_default() -> None:
    """P1.9: nạp engine VAD đang dùng NGAY (cần cho audio đầu tiên)."""
    try:
        vad = VADProcessor(vad_engine=config.vad.vad_engine)
        vad.feed_chunk(bytes(800))
        logger.info(
            f"[STARTUP] VAD engine mặc định '{config.vad.vad_engine}' đã sẵn sàng!",
            extra={"module_tag": "VAD"},
        )
    except Exception as e:
        logger.warning(f"[STARTUP] Cảnh báo pre-warm VAD: {e}", exc_info=True, extra={"module_tag": "VAD"})


def _prewarm_vad_others() -> None:
    """P1.9/G10: nạp NỀN các engine VAD khác để đổi trong popup là áp dụng ngay.

    Đặc biệt quan trọng vì extension từng mặc định gửi một VAD engine khác engine mặc
    định của backend — nếu chưa nạp sẵn thì lần đổi đó sẽ phải nạp model (có thể tải từ
    HuggingFace) ngay trên đường hot, làm backend ngừng nhận audio.

    Hàm này BLOCKING và được gọi ở NỀN để không làm chậm khởi động.
    Trả về dict {engine: status} để test/log kiểm chứng được.
    """
    statuses: Dict[str, str] = {}
    try:
        from backend.vad import SUPPORTED_VAD_ENGINES
        from backend.vad.engines import VADEngineFactory
        others = [e for e in SUPPORTED_VAD_ENGINES if e != config.vad.vad_engine]
        statuses = VADEngineFactory.prewarm_engines(others, threshold=config.vad.threshold)
        for name, status in statuses.items():
            if status == "ok":
                logger.info(f"[STARTUP] VAD engine '{name}' nạp nền xong.",
                            extra={"module_tag": "VAD"})
            else:
                logger.warning(f"[STARTUP] VAD engine '{name}' nạp nền thất bại: {status}",
                               extra={"module_tag": "VAD"})
    except Exception as e:
        logger.warning(f"[STARTUP] Cảnh báo nạp nền VAD engines: {e}", extra={"module_tag": "VAD"})
    return statuses


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo và pre-warm trước (Pre-warm) song song toàn bộ các mô hình khi máy chủ khởi động."""
    logger.info("[STARTUP] Đang nạp và pre-warm song song ASR, Translation & VAD...", extra={"module_tag": "MAIN"})

    _log_asr_backend_at_startup()

    results = await asyncio.gather(
        asyncio.to_thread(_prewarm_asr),
        asyncio.to_thread(_prewarm_translation),
        asyncio.to_thread(_prewarm_vad_default),
        return_exceptions=True,
    )
    for res in results:
        if isinstance(res, Exception):
            logger.warning(f"[STARTUP] Lỗi thành phần trong quá trình prewarm: {res}", extra={"module_tag": "MAIN"})

    # Các engine VAD khác nạp ở nền để không làm chậm khởi động.
    track_background_task(asyncio.to_thread(_prewarm_vad_others), name="vad_prewarm_others")

    # F-40: nhịp tim event loop — để `/health` phát hiện được loop bị chặn đứng (treo im lặng).
    from backend.core import heartbeat

    heartbeat.reset()
    track_background_task(heartbeat.heartbeat_loop(), name="heartbeat")

    # F-46: watchdog LUỒNG THẬT — event loop đứng quá lâu thì dump stack MỌI thread ra stderr
    # (kiểu treo do native chặn GIL không thể phát hiện bằng coroutine).
    try:
        from backend.utils import stall_watchdog

        stall_watchdog.start()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Không bật được stall watchdog: {exc}", extra={"module_tag": "MAIN"})

    logger.info("[STARTUP] Toàn bộ mô hình đã được pre-warm song song và sẵn sàng phục vụ!", extra={"module_tag": "MAIN"})
    yield
    logger.info("[SHUTDOWN] Đang giải phóng toàn bộ tài nguyên GPU & RAM...", extra={"module_tag": "MAIN"})
    for t in list(_background_tasks):
        if not t.done():
            t.cancel()

    try:
        TranscribeEngine.shutdown_executors(wait=False)
        TranscribeEngine.unload_shared_model()
    except Exception as e:
        logger.debug(f"ASR cleanup notice: {e}", extra={"module_tag": "MAIN"})

    try:
        from backend.translation.engine import GGUFTranslator
        GGUFTranslator.shutdown_executors(wait=False)
        reset_translation_engine()
    except Exception as e:
        logger.debug(f"Translation cleanup notice: {e}", extra={"module_tag": "MAIN"})

    try:
        OmniVoiceTTS.reset_instance()
    except Exception as e:
        logger.debug(f"TTS cleanup notice: {e}", extra={"module_tag": "MAIN"})

    try:
        from backend.ws.handler import shutdown_vad_executor
        shutdown_vad_executor(wait=False)
    except Exception as e:
        logger.debug(f"VAD cleanup notice: {e}", extra={"module_tag": "MAIN"})

    # Lưới an toàn: nếu phiên không đóng sạch (Ctrl+C, kill, mất điện) thì đường
    # `dump_report_on_disconnect` trong handler đã không chạy ⇒ ghi ở đây để không mất số liệu.
    # Ví dụ đúng của việc này: người dùng chạy phiên 5 phút rồi tắt server và không tìm thấy
    # `metrics_report.json` ở đâu cả.
    try:
        from backend.core.metrics import dump_metrics_report

        dump_metrics_report("shutdown")
    except Exception as e:
        logger.debug(f"Metrics dump notice: {e}", extra={"module_tag": "MAIN"})

    logger.info("[SHUTDOWN] Hoàn tất tắt máy chủ an toàn.", extra={"module_tag": "MAIN"})



app = FastAPI(
    title="Vibe Translation Backend (Refactored Modular)",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Regex cho Web Extensions
_CORS_ORIGIN_REGEX = (
    r"http://localhost(:\d+)?"
    r"|https://localhost(:\d+)?"
    r"|moz-extension://[0-9a-f-]+"
    r"|chrome-extension://[0-9a-z-]+"
    r"|edge-extension://[0-9a-z-]+"
    r"|extension://[0-9a-z-]+"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Endpoint kiểm tra sức khỏe hệ thống."""
    from backend.core import heartbeat
    from backend.ws.handler import count_active_sessions

    return {
        "status": "ok",
        "service": "backend_modular",
        "protocol_version": config.ws.protocol_version,
        "asr_model": ModelRegistry.get_instance().get_active_model_key(),
        "asr_runtime": _asr_runtime_info(),
        "vad_engine": config.vad.vad_engine,
        "translation_model": config.translation.base,
        "tts_model": config.tts.model,
        "active_sessions": count_active_sessions(),
        # F-40: nếu event loop bị chặn, giá trị này tăng đều dù /health vẫn trả lời được
        # (FastAPI chạy trong loop nên thực tế nó chỉ nhảy vọt khi loop vừa thoát ra).
        "loop_stall_ms": round(heartbeat.stall_sec() * 1000.0, 1),
    }


@app.get("/", response_class=HTMLResponse)
async def root_page():
    """Trang xác nhận chứng chỉ SSL/WSS cho trình duyệt."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Vibe Translation Backend Modular</title></head>
    <body style="font-family: system-ui, sans-serif; text-align: center; padding-top: 60px; background: #0f172a; color: #f8fafc;">
        <h1 style="color: #38bdf8;">✅ Vibe Translation Backend Modular v2.0</h1>
        <p style="font-size: 16px;">Backend đang hoạt động ở chế độ bảo mật <strong>HTTPS / WSS</strong>.</p>
        <p style="color: #4ade80; font-weight: bold; font-size: 18px;">Chứng chỉ SSL đã được xác nhận thành công!</p>
        <p style="color: #94a3b8; font-size: 14px;">Bạn có thể đóng tab này và bắt đầu sử dụng Extension trên Firefox.</p>
    </body>
    </html>
    """


def _build_config_response(include_catalog: bool = True) -> Dict[str, Any]:
    """Tạo đối tượng phản hồi cấu hình tập trung cho Client."""
    registry = ModelRegistry.get_instance()
    active_key = registry.get_active_model_key()
    active_info = registry.get_model_info(active_key) or {}
    loaded_model_name = active_info.get("name", active_key)

    resp: Dict[str, Any] = {
        "status": "ok",
        "protocol_version": config.ws.protocol_version,
        "engine": active_key,
        "asr_engine": active_key,
        "active_model": active_key,
        "loaded_model": loaded_model_name,
        "resolved_vad": config.vad.vad_engine,
        "vad_engine": config.vad.vad_engine,
        "available_vad_engines": list(SUPPORTED_VAD_ENGINES),
        "vad_silence_duration_ms": config.vad.silence_duration_ms,
        "silence_duration_ms": config.vad.silence_duration_ms,
        "hangover_ms": config.vad.hangover_ms,
        "vad_threshold": config.vad.threshold,
        "min_words_to_commit": config.sentence.min_words_to_commit,
        "source_lang": config.asr.language,
        "supported_languages": SUPPORTED_LANGUAGES,
        # P2.x: thông số streaming để popup hiển thị/chỉnh được và client biết backend đang làm gì
        "streaming": {
            "preview_window_sec": config.asr.preview_window_sec,
            "poll_interval_ms": config.asr.poll_interval_ms,
            "min_transcribe_sec": config.asr.min_transcribe_sec,
            "max_duration_sec": config.sentence.max_duration_sec,
            "stability_duration_sec": config.sentence.stability_duration_sec,
            "enable_tier234": config.sentence.enable_tier234,
            "reuse_preview_for_commit": config.asr.preview_reuse_for_commit,
            "stream_translation": config.translation.stream_tokens,
            "native_backend": config.asr.backend,
        },
        "translation": {
            "model": config.translation.model,
            "gguf_file": config.translation.gguf_file,
            "target_lang": config.translation.target_lang,
            "source_lang": config.translation.source_lang,
            "base": config.translation.base,
            "translation_model": config.translation.base,
            # Model nào đã có file GGUF cục bộ (cờ `is_downloaded` trong available_models).
            "auto_download": config.translation.auto_download,
            # Tiến trình tải/nạp model dịch: idle | downloading | loading | ready | error.
            "download": translation_hotswap.status(),
            "available_models": TranslationModelRegistry.get_instance().list_models(),
        },
        "tts": {
            "enabled": config.tts.enabled,
            "engine": config.tts.engine,
            "model": config.tts.model,
            "speed": config.tts.speed,
            "default_voice": config.tts.default_voice,
            "voices": VoiceManager.get_available_voices(),
        },
    }

    if include_catalog:
        available_models = registry.list_models()
        resp["available_models"] = available_models
        resp["available_asr_engines"] = [m["id"] for m in available_models]

    return resp


@app.get("/api/config")
async def get_backend_config():
    """Trả về cấu hình hiện tại và danh mục mô hình ASR/Translation/TTS."""
    return _build_config_response(include_catalog=True)


@app.get("/api/voices")
async def get_voices_list():
    """Trả về danh sách các mẫu giọng clone có sẵn."""
    return {"status": "ok", "voices": VoiceManager.get_available_voices()}


@app.post("/api/config")
@app.post("/api/switch-engine")
async def update_backend_config(req: SwitchModelRequest):
    """Cập nhật cấu hình runtime động hoặc hot-swap mô hình.

    P1.8/P1.10: đổi ASR model dùng đường "nạp trước rồi swap" (không giải phóng model
    cũ trước khi model mới sẵn sàng => không có khoảng trống phụ đề và không còn cửa
    sổ use-after-free). Các thay đổi còn lại được ĐẨY vào phiên đang chạy qua
    `SessionState.apply_config()` thay vì chỉ ghi vào config toàn cục.
    """
    registry = ModelRegistry.get_instance()
    target_model = req.model_id or req.asr_engine

    if target_model:
        try:
            registry.set_active_model_key(target_model)
            engine = TranscribeEngine(target_model)
            # prepare_model nạp model mới NGOÀI lock rồi swap nguyên tử dưới _infer_lock.
            await asyncio.to_thread(engine.prepare_model, target_model)
            logger.info(f"Đã chuyển đổi ASR Model sang: '{target_model}'(nạp trước + swap)", extra={"module_tag": "MAIN"})
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if req.vad_engine is not None:
        ve = req.vad_engine.lower().strip()
        if ve in SUPPORTED_VAD_ENGINES:
            config.vad.vad_engine = ve
            logger.info(f"Đã chuyển VAD engine sang: '{ve}'", extra={"module_tag": "MAIN"})

    if req.vad_threshold is not None:
        config.vad.threshold = req.vad_threshold
    if req.silence_duration_ms is not None:
        config.vad.silence_duration_ms = req.silence_duration_ms
    if req.min_words_to_commit is not None:
        config.sentence.min_words_to_commit = max(0, req.min_words_to_commit)
    if req.target_lang is not None:
        config.translation.target_lang = req.target_lang
    if req.source_lang is not None:
        config.asr.language = req.source_lang

    if req.translation_model is not None:
        tm = req.translation_model.lower().strip()
        trans_registry = TranslationModelRegistry.get_instance()
        # F-50: `resolve_key` fallback về `default_model` nên phải kiểm tra key có thật trước,
        # nếu không một tên sai sẽ âm thầm đổi sang model mặc định.
        if not trans_registry.is_known(tm):
            raise HTTPException(
                status_code=400,
                detail=f"Model dịch '{tm}' không có trong translation_models.yaml",
            )
        canonical_key = trans_registry.resolve_key(tm)

        if translation_hotswap.needs_download(canonical_key):
            if not translation_hotswap.auto_download_enabled():
                gguf_path = trans_registry.resolve_gguf_path(canonical_key)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Chưa có file GGUF cục bộ: {gguf_path}. Hãy copy file vào backend/models "
                        f"hoặc bật TranslationConfig.auto_download để backend tự tải."
                    ),
                )
            if translation_hotswap.is_busy() and not translation_hotswap.is_busy(canonical_key):
                raise HTTPException(
                    status_code=409,
                    detail=f"Đang tải/nạp model dịch '{translation_hotswap.status().get('model')}', vui lòng đợi.",
                )
            snapshot = translation_hotswap.reserve(canonical_key)
            if snapshot.get("started"):
                track_background_task(
                    _activate_translation_model_bg(canonical_key, bool(snapshot.get("allow_download"))),
                    name=f"translation_activate_{canonical_key}",
                )
            payload = _build_config_response(include_catalog=False)
            payload.update({
                # Giữ `status: ok` để client cũ không hiểu nhầm thành lỗi; trạng thái tải nằm
                # ở `download_state`/`download` (HTTP code 202 mới là tín hiệu "đang tải nền").
                "download_state": "downloading",
                "detail": (
                    f"Đang tải model dịch '{canonical_key}' về backend/models (chạy nền). "
                    f"Model '{config.translation.base}' hiện tại vẫn hoạt động bình thường."
                ),
                "translation_model": canonical_key,
                "download": translation_hotswap.status(),
            })
            logger.info(
                f"Model dịch '{canonical_key}' chưa có file — đã xếp lịch tải nền; "
                    f"giữ nguyên model đang chạy '{config.translation.base}'.",
                extra={"module_tag": "MAIN"},
            )
            return JSONResponse(status_code=202, content=payload)

        if translation_hotswap.is_busy():
            raise HTTPException(
                status_code=409,
                detail=f"Đang tải/nạp model dịch '{translation_hotswap.status().get('model')}', vui lòng đợi.",
            )
        try:
            # F-50: tải (nếu cần) + nạp model mới TRƯỚC, chỉ ghi config sau khi thành công.
            await translation_hotswap.activate_model(canonical_key, allow_download=True)
            logger.info(f"Đã chuyển mô hình dịch sang: '{canonical_key}'", extra={"module_tag": "MAIN"})
        except Exception as e:
            logger.error(f"Lỗi chuyển mô hình dịch sang '{canonical_key}': {e}", exc_info=True, extra={"module_tag": "MAIN"})
            raise HTTPException(status_code=500, detail=str(e))

    if req.tts_enabled is not None:
        config.tts.enabled = req.tts_enabled
        if req.tts_enabled:
            track_background_task(get_tts_engine().prewarm(), name="tts_prewarm_post_config")
        else:
            # A2-3 (Hy3): tắt TTS => trả VRAM vài GB thay vì giữ model vô ích.
            # Model sẽ được nạp lại (prewarm) khi người dùng bật lại.
            # Chạy trong thread riêng: `unload_model()` có `gc.collect()` + `empty_cache()`,
            # chặn event loop ở đây sẽ tự tạo ra một latency spike.
            try:
                await asyncio.to_thread(get_tts_engine().unload_model)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"TTS unload khi tắt thất bại (bỏ qua): {exc}", extra={"module_tag": "MAIN"})
    if req.tts_voice is not None:
        config.tts.default_voice = req.tts_voice
    if req.tts_speed is not None:
        config.tts.speed = req.tts_speed

    # P1.10: đẩy các thay đổi vào phiên ĐANG CHẠY (trước đây REST chỉ đổi config toàn
    # cục, còn SessionState giữ snapshot cũ nên thay đổi không có hiệu lực).
    session_payload: Dict[str, Any] = {}
    if req.vad_engine is not None:
        session_payload["vad_engine"] = req.vad_engine
    if req.vad_threshold is not None:
        session_payload["vad_threshold"] = req.vad_threshold
    if req.silence_duration_ms is not None:
        session_payload["silence_duration_ms"] = req.silence_duration_ms
    if req.min_words_to_commit is not None:
        session_payload["min_words_to_commit"] = req.min_words_to_commit
    if req.target_lang is not None:
        session_payload["target_lang"] = req.target_lang
    if req.source_lang is not None:
        session_payload["source_lang"] = req.source_lang
    if req.tts_enabled is not None:
        session_payload["tts_enabled"] = req.tts_enabled
    if req.tts_voice is not None:
        session_payload["tts_voice"] = req.tts_voice
    if req.tts_speed is not None:
        session_payload["tts_speed"] = req.tts_speed

    from backend.ws.handler import get_active_sessions
    applied_to_sessions = 0
    for sess in get_active_sessions():
        try:
            if session_payload:
                sess.apply_config(session_payload)
                applied_to_sessions += 1
            # ASR model: nạp nền + thông báo trạng thái cho phiên đang chạy
            if target_model and sess.asr_engine is not None:
                sess.config["asr_engine"] = target_model
                sess.asr_engine.model_key = target_model
                sess.asr_engine.model_info = registry.get_model_info(target_model) or {}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Không áp dụng được cấu hình REST cho session {sess.session_id}: {e}", extra={"module_tag": "MAIN"})
    if applied_to_sessions:
        logger.info(
            f"Cấu hình từ popup đã áp dụng ngay cho {applied_to_sessions} phiên đang chạy.",
            extra={"module_tag": "MAIN"},
        )

    # Thông báo rõ ràng trên console khi popup cập nhật giá trị
    updated_items = []
    if req.model_id or req.asr_engine:
        updated_items.append(f"asr='{target_model}'")
    if req.vad_engine is not None:
        updated_items.append(f"vad='{req.vad_engine}'")
    if req.vad_threshold is not None:
        updated_items.append(f"threshold={req.vad_threshold}")
    if req.silence_duration_ms is not None:
        updated_items.append(f"silence={req.silence_duration_ms}ms")
    if req.min_words_to_commit is not None:
        updated_items.append(f"min_words={req.min_words_to_commit}")
    if req.source_lang is not None:
        updated_items.append(f"src='{req.source_lang}'")
    if req.target_lang is not None:
        updated_items.append(f"tgt='{req.target_lang}'")
    if req.translation_model is not None:
        updated_items.append(f"trans='{req.translation_model}'")
    if req.tts_enabled is not None:
        updated_items.append(f"tts={req.tts_enabled}")
    if req.tts_voice is not None:
        updated_items.append(f"voice='{req.tts_voice}'")
    if req.tts_speed is not None:
        updated_items.append(f"speed={req.tts_speed}")

    if updated_items:
        logger.info(
            f"Thay đổi từ Extension Popup: {', '.join(updated_items)}",
            extra={"module_tag": "WS"},
        )

    return _build_config_response(include_catalog=True)


@app.post("/api/tts/prewarm")
async def prewarm_tts_endpoint():
    """pre-warm mô hình TTS trên GPU."""
    success = await get_tts_engine().prewarm()
    if success:
        config.tts.enabled = True
    return {"status": "ok" if success else "error", "prewarmed": success}


@app.get("/api/metrics")
async def get_metrics():
    """Lấy báo cáo đo lường hiệu năng và cảnh báo điểm nghẽn thời gian thực."""
    from backend.ws.handler import count_active_sessions
    metrics_collector.record_gauge("ws", "active_sessions", count_active_sessions())
    return metrics_collector.generate_report()


@app.post("/api/metrics/dump")
async def dump_metrics():
    """Ghi metrics ra `metrics_report.json` NGAY (không cần kết thúc phiên).

    Dùng khi đang chạy một phiên đo dài và muốn chốt số liệu mà không phải ngắt kết nối:
        curl -k -X POST https://127.0.0.1:8765/api/metrics/dump
    """
    from backend.core.metrics import dump_metrics_report

    path = await asyncio.to_thread(dump_metrics_report, "api_request")
    if not path:
        return {"status": "error", "detail": "metrics bị tắt hoặc ghi thất bại"}
    return {"status": "ok", "path": path}


@app.get("/api/metrics/pipeline")
async def get_pipeline_metrics():
    """P0.1: ảnh chụp gọn các stage hot path (asr/vad/queue) để chẩn đoán nghẽn."""
    return metrics_collector.snapshot_pipeline()


@app.websocket("/ws")
@app.websocket("/")
async def websocket_endpoint(ws: WebSocket):
    """Endpoint kết nối WebSocket trực tiếp từ Firefox Extension."""
    await handle_ws(ws)


def main():
    """Khởi chạy máy chủ Backend kèm hỗ trợ chứng chỉ WSS tự ký."""
    from backend.utils.ssl import ensure_ssl_certificates
    cert_path, key_path = ensure_ssl_certificates()

    ssl_kwargs = {
        "ssl_certfile": cert_path,
        "ssl_keyfile": key_path,
    }
    logger.info(f"Chế độ WSS (SSL) kích hoạt với cert: {cert_path}", extra={"module_tag": "MAIN"})

    try:
        uvicorn.run(
            app,
            host=config.ws.host,
            port=config.ws.port,
            ws_ping_interval=config.ws.ping_interval,
            ws_ping_timeout=config.ws.ping_timeout,
            timeout_graceful_shutdown=1,
            **ssl_kwargs,
        )
    except KeyboardInterrupt:
        logger.info("Nhận tín hiệu ngắt (Ctrl+C). Đã dừng máy chủ an toàn.", extra={"module_tag": "MAIN"})



if __name__ == "__main__":
    main()
