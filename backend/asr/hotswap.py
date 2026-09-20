"""Đổi model ASR nguyên tử kèm tự động tải nền — dùng chung cho REST `/api/config` và WebSocket.

Nguyên tắc:
1. Kiểm tra catalog + file trước; thiếu file thì tải trước (nếu bật `auto_download`).
2. Tải xong nạp model mới (ngoài lock) -> swap nguyên tử -> cập nhật config ASR.
3. Lỗi ở bất kỳ bước nào => model cũ vẫn đang phục vụ, trạng thái báo rõ cho popup.

Trạng thái: `idle → downloading → loading → ready | error` (kèm tiến độ tải thật).
"""

import asyncio
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional

from backend.config import config
from backend.asr.registry import ModelRegistry
from backend.utils import model_download
from backend.utils.logger import logger

_TAG = "ASR"
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
    """Cờ `config.asr.auto_download` (bật/tắt tự động tải model ASR từ HuggingFace)."""
    return bool(getattr(config.asr, "auto_download", True))


def needs_download(model_key: str) -> bool:
    """True nếu file GGUF của model ASR chưa có cục bộ."""
    try:
        registry = ModelRegistry.get_instance()
        return registry.needs_download(model_key)
    except Exception:  # noqa: BLE001
        return True


def is_busy(model_key: Optional[str] = None) -> bool:
    """True nếu đang tải/nạp model ASR (tuỳ chọn: chỉ xét một model cụ thể)."""
    with _LOCK:
        if _STATE.get("state") not in ("downloading", "loading"):
            return False
        return model_key is None or _STATE.get("model") == model_key


def status() -> Dict[str, Any]:
    """Trạng thái hiện tại (đang tải thì ghép tiến độ tải trực tiếp từ model_download)."""
    with _LOCK:
        snapshot = dict(_STATE)

    if snapshot.get("state") == "downloading" and snapshot.get("model"):
        registry = ModelRegistry.get_instance()
        repo = registry.repo_id(snapshot["model"])
        fname = registry.file_name(snapshot["model"])
        if repo and fname:
            live = model_download.get_state(repo, fname)
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


def reserve(model_key: str, *, allow_download: Optional[bool] = None) -> Dict[str, Any]:
    """Đặt trạng thái NGAY (đồng bộ) trước khi tác vụ nền chạy — tránh race với response 202.

    Trả về dict có `started`: False nghĩa là đã có lượt tải/nạp cho model này đang chạy.
    """
    clean_key = (model_key or "").strip().lower()
    with _LOCK:
        already = _STATE.get("state") in ("downloading", "loading") and _STATE.get("model") == clean_key
        need = needs_download(clean_key)
        allow = auto_download_enabled() if allow_download is None else bool(allow_download)
        if already:
            snapshot = dict(_STATE)
            snapshot["started"] = False
            return snapshot
        _STATE.update(
            state="downloading" if need else "loading",
            model=clean_key,
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


async def run_reserved(model_key: str, *, allow_download: Optional[bool] = None) -> None:
    """Thực hiện tải (nếu cần) + nạp model ASR cho một lượt đã `reserve()`."""
    from backend.asr.engine import TranscribeEngine

    clean_key = (model_key or "").strip().lower()
    registry = ModelRegistry.get_instance()
    if not registry.has_model(clean_key):
        raise ValueError(f"Model ASR '{clean_key}' không có trong models.yaml")

    if allow_download is None:
        with _LOCK:
            allow_download = bool(_STATE.get("allow_download", auto_download_enabled()))
    allow_download = bool(allow_download)

    started = time.time()
    try:
        if needs_download(clean_key):
            logger.info(
                f"Tải model ASR '{clean_key}' trước khi swap (zero-downtime)",
                extra={"module_tag": _TAG},
            )
            await asyncio.to_thread(
                registry.ensure_model_file,
                clean_key,
                allow_download=allow_download,
            )

        _set_state(state="loading", percent=100.0 if _STATE.get("state") == "downloading" else None)

        engine = TranscribeEngine(clean_key)
        # prepare_model nạp model mới NGOÀI lock rồi swap nguyên tử dưới _infer_lock.
        await asyncio.to_thread(engine.prepare_model, clean_key)

        # CHỈ cập nhật registry/config sau khi nạp thành công
        registry.set_active_model_key(clean_key)
        config.asr.active_model = clean_key

        elapsed = time.time() - started
        _set_state(state="ready", error=None, finished_at=time.time(), elapsed_sec=round(elapsed, 1))
        logger.info(
            f"Đã swap ASR sang '{clean_key}' ({elapsed:.1f}s)",
            extra={"module_tag": _TAG},
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - started
        _set_state(state="error", error=str(exc), finished_at=time.time(), elapsed_sec=round(elapsed, 1))
        logger.error(f"Đổi model ASR sang '{clean_key}' thất bại: {exc}", extra={"module_tag": _TAG})
        raise


async def activate_model(model_key: str, *, allow_download: Optional[bool] = None) -> None:
    """Đổi model ASR và CHỜ tới khi hoàn thành (dùng cho WebSocket cần đồng bộ báo trạng thái)."""
    clean_key = (model_key or "").strip().lower()
    snapshot = reserve(clean_key, allow_download=allow_download)
    if not snapshot.get("started"):
        raise RuntimeError(f"Đang tải/nạp model ASR '{clean_key}' rồi, vui lòng đợi.")
    await run_reserved(clean_key, allow_download=snapshot.get("allow_download"))
