"""Đổi model dịch "nguyên tử" — dùng chung cho REST `/api/config` và WebSocket.

Vì sao cần module riêng (F-50): đường REST cũ ghi `config.translation.base` TRƯỚC khi nạp
và `GGUFTranslator.reconfigure()` giải phóng model cũ trước khi nạp model mới. Khi model mới
thiếu file GGUF (ví dụ `Hy-MT2-1.8B-UD-Q8_K_XL.gguf` chưa tải về), backend vừa mất model
đang chạy tốt vừa ghi sai config ⇒ mọi bản dịch sau đó hỏng cho tới khi restart.

Nguyên tắc ở đây:
1. Kiểm tra catalog + file trước; thiếu file thì **tải trước** (nếu bật `auto_download`).
2. Chỉ ghi `config.translation.base` SAU khi model mới đã nạp xong.
3. Lỗi ở bất kỳ bước nào ⇒ model cũ vẫn đang phục vụ, trạng thái được báo rõ cho popup.

Trạng thái: `idle → downloading → loading → ready | error` (kèm tiến độ tải thật).
"""

import asyncio
import os
import threading
import time
from typing import Any, Dict, Optional

from backend.config import config, TranslationConfig
from backend.translation.engine import get_translation_engine
from backend.translation.registry import TranslationModelRegistry
from backend.utils import model_download
from backend.utils.logger import logger

_TAG = "TRANSLATE"
_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "state": "idle",
    "model": None,
    "error": None,
    "started_at": None,
    "finished_at": None,
    "elapsed_sec": None,
    "downloaded_mb": None,
    "total_mb": None,
    "percent": None,
}


def _mb(num_bytes: Optional[float]) -> Optional[float]:
    if num_bytes is None:
        return None
    try:
        return round(float(num_bytes) / (1024.0 * 1024.0), 1)
    except Exception:  # noqa: BLE001
        return None


def _set_state(**fields: Any) -> Dict[str, Any]:
    with _LOCK:
        _STATE.update(fields)
        _STATE["updated_at"] = time.time()
        return dict(_STATE)


def auto_download_enabled() -> bool:
    """Cờ `config.translation.auto_download` (bật/tắt tải model dịch từ HuggingFace)."""
    return bool(getattr(config.translation, "auto_download", True))


def needs_download(canonical_key: str) -> bool:
    """True nếu file GGUF của model chưa có cục bộ."""
    try:
        return not os.path.isfile(TranslationModelRegistry.get_instance().resolve_gguf_path(canonical_key))
    except Exception:  # noqa: BLE001
        return True


def is_busy(canonical_key: Optional[str] = None) -> bool:
    """True nếu đang tải/nạp model dịch (tuỳ chọn: chỉ xét một model cụ thể)."""
    with _LOCK:
        if _STATE.get("state") not in ("downloading", "loading"):
            return False
        return canonical_key is None or _STATE.get("model") == canonical_key


def status() -> Dict[str, Any]:
    """Trạng thái hiện tại (đang tải thì ghép tiến độ tải trực tiếp từ model_download)."""
    with _LOCK:
        snapshot = dict(_STATE)

    if snapshot.get("state") == "downloading" and snapshot.get("model"):
        info = TranslationModelRegistry.get_instance().get_model(snapshot["model"]) or {}
        live = model_download.get_state(str(info.get("model", "")), str(info.get("gguf_file", "")))
        if live:
            snapshot["downloaded_mb"] = _mb(live.get("downloaded_bytes"))
            snapshot["total_mb"] = _mb(live.get("total_bytes"))
            snapshot["percent"] = live.get("percent")
            if live.get("error"):
                snapshot["error"] = live.get("error")
    return snapshot


def reset_state() -> None:
    """Đưa trạng thái về `idle` (dùng cho test)."""
    _set_state(
        state="idle", model=None, error=None, started_at=None, finished_at=None,
        elapsed_sec=None, downloaded_mb=None, total_mb=None, percent=None,
    )


def reserve(canonical_key: str, *, allow_download: Optional[bool] = None) -> Dict[str, Any]:
    """Đặt trạng thái NGAY (đồng bộ) trước khi tác vụ nền chạy — tránh race với response 202.

    Trả về dict có `started`: False nghĩa là đã có lượt tải/nạp cho model này đang chạy.
    """
    with _LOCK:
        already = _STATE.get("state") in ("downloading", "loading") and _STATE.get("model") == canonical_key
        need = needs_download(canonical_key)
        allow = auto_download_enabled() if allow_download is None else bool(allow_download)
        if already:
            snapshot = dict(_STATE)
            snapshot["started"] = False
            return snapshot
        _STATE.update(
            state="downloading" if need else "loading",
            model=canonical_key,
            error=None,
            started_at=time.time(),
            finished_at=None,
            elapsed_sec=None,
            downloaded_mb=None,
            total_mb=None,
            percent=0.0 if need else None,
            updated_at=time.time(),
            allow_download=allow,
        )
        snapshot = dict(_STATE)
        snapshot["started"] = True
        return snapshot


async def run_reserved(canonical_key: str, *, allow_download: Optional[bool] = None) -> None:
    """Thực hiện tải (nếu cần) + nạp model cho một lượt đã `reserve()`."""
    registry = TranslationModelRegistry.get_instance()
    info = registry.get_model(canonical_key)
    if info is None:
        raise ValueError(f"Model dịch '{canonical_key}' không có trong translation_models.yaml")

    if allow_download is None:
        with _LOCK:
            allow_download = bool(_STATE.get("allow_download", auto_download_enabled()))
    allow_download = bool(allow_download)

    started = time.time()
    try:
        if needs_download(canonical_key):
            logger.info(
                f"Tải model '{canonical_key}' trước khi swap (zero-downtime)",
                extra={"module_tag": _TAG},
            )
            await asyncio.to_thread(
                model_download.ensure_model_file,
                str(info.get("model", "")),
                str(info.get("gguf_file", "")),
                allow_download=allow_download,
                label=canonical_key,
            )

        _set_state(state="loading", percent=100.0 if _STATE.get("state") == "downloading" else None)

        new_cfg = TranslationConfig(
            base=canonical_key,
            target_lang=config.translation.target_lang,
            source_lang=config.translation.source_lang,
            auto_download=allow_download,
        )
        translator = get_translation_engine(new_cfg)
        # reconfigure nạp model mới trước rồi mới swap (F-50) — lỗi thì model cũ còn nguyên.
        await asyncio.to_thread(translator.reconfigure, new_cfg, False)

        # CHỈ đổi config sau khi model mới đã nạp thành công.
        config.translation.base = canonical_key
        elapsed = time.time() - started
        _set_state(state="ready", error=None, finished_at=time.time(), elapsed_sec=round(elapsed, 1))
        logger.info(
            f"Đã swap sang '{canonical_key}' ({elapsed:.1f}s)",
            extra={"module_tag": _TAG},
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - started
        _set_state(state="error", error=str(exc), finished_at=time.time(), elapsed_sec=round(elapsed, 1))
        logger.error(f"Đổi model dịch sang '{canonical_key}' thất bại: {exc}", extra={"module_tag": _TAG})
        raise


async def activate_model(canonical_key: str, *, allow_download: Optional[bool] = None) -> None:
    """Đổi model dịch và CHỜ tới khi xong (dùng cho đường WS cần báo tiến trình)."""
    snapshot = reserve(canonical_key, allow_download=allow_download)
    if not snapshot.get("started"):
        raise RuntimeError(f"Đang tải/nạp model dịch '{canonical_key}' rồi, vui lòng đợi.")
    await run_reserved(canonical_key, allow_download=snapshot.get("allow_download"))
