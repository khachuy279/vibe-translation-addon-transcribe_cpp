"""Factory quản lý và cache các Singleton VAD Engine."""

import threading
from typing import Dict, Tuple, Any, Optional

from backend.config import config
from backend.utils.logger import logger
from backend.vad.base import BaseVADEngine
from backend.vad.engines.firered import FireRedVADEngine
from backend.vad.engines.fsmn import FsmnVADEngine
from backend.vad.engines.silero import SileroVADEngine

SUPPORTED_VAD_ENGINES: Tuple[str, ...] = ("firered-vad", "fsmn-vad", "silero-vad")


class VADEngineFactory:
    """Thread-safe Factory & Pool cho các VAD Engine."""

    _engines: Dict[str, BaseVADEngine] = {}
    _lock: threading.RLock = threading.RLock()
    # QWEN-Q2: lock RIÊNG cho bước tải file model. Không dùng chung `_lock` vì
    # `_lock` còn bảo vệ pool — mà `is_cached()`/`peek_engine()` (được EVENT LOOP gọi
    # ở `VADProcessor.__init__`/`apply_engine_change`) cũng lấy `_lock`. Giữ `_lock`
    # qua một lượt `hf_hub_download` vài phút = treo cả backend.
    _prepare_lock: threading.Lock = threading.Lock()

    @classmethod
    def _factory_for(cls, engine: str):
        if engine == "firered-vad":
            return FireRedVADEngine
        if engine == "silero-vad":
            return SileroVADEngine
        return FsmnVADEngine

    @classmethod
    def get_engine(cls, engine_name: str) -> BaseVADEngine:
        """Lấy hoặc khởi tạo instance Engine VAD dùng chung (double-checked locking).

        QWEN-Q2: bước TẢI model chạy **ngoài** `_lock`, dưới `_prepare_lock` riêng. Bản
        cũ giữ `_lock` suốt constructor ⇒ trong lúc tải model, mọi `is_cached()` /
        `peek_engine()` từ event loop bị chặn theo.
        """
        engine = (engine_name or "firered-vad").lower().strip()
        if engine not in SUPPORTED_VAD_ENGINES:
            engine = "firered-vad"

        with cls._lock:
            if engine in cls._engines:
                return cls._engines[engine]

        # ── NGOÀI `_lock`: có thể chậm (tải model) ───────────────────────────────
        # `_prepare_lock` chỉ chặn các luồng ĐANG TẢI cùng lúc (tránh hai luồng cùng ghi
        # vào một cache dir), KHÔNG chặn đường đọc trạng thái.
        with cls._prepare_lock:
            try:
                cls._factory_for(engine).prepare_files()
            except Exception as exc:  # noqa: BLE001
                # Không tải được file ⇒ vẫn thử dựng engine (có thể engine tự xử lý hoặc
                # báo lỗi rõ hơn). Không được để lỗi tải che mất nguyên nhân thật.
                logger.warning(
                    f"Chuẩn bị file model cho VAD engine '{engine}' gặp lỗi "
                        f"({type(exc).__name__}: {exc}) — vẫn thử dựng engine.",
                    extra={"module_tag": "VAD"},
                )

        with cls._lock:
            if engine in cls._engines:      # luồng khác đã dựng xong trong lúc ta tải
                return cls._engines[engine]
            instance = cls._factory_for(engine)()
            cls._engines[engine] = instance
            return instance

    @classmethod
    def is_cached(cls, engine_name: str) -> bool:
        """Engine đã được nạp sẵn trong pool chưa (không kích hoạt nạp/tải)."""
        engine = (engine_name or "").lower().strip()
        with cls._lock:
            return engine in cls._engines

    @classmethod
    def peek_engine(cls, engine_name: str) -> Optional[BaseVADEngine]:
        """Lấy engine CHỈ KHI đã có trong pool; trả None nếu chưa (không nạp, không tải)."""
        engine = (engine_name or "").lower().strip()
        with cls._lock:
            return cls._engines.get(engine)

    @classmethod
    def prewarm_engines(cls, names, threshold: Optional[float] = None) -> Dict[str, str]:
        """P1.9: nạp trước nhiều engine để chuyển nóng không phải tải/nạp model.

        Trả về dict {engine_name: "ok" | "error: ..."} để caller log rõ.
        Hàm này BLOCKING — gọi từ thread nền (asyncio.to_thread) khi khởi động.
        """
        results: Dict[str, str] = {}
        for name in names:
            key = (name or "").lower().strip()
            if key not in SUPPORTED_VAD_ENGINES:
                continue
            try:
                engine = cls.get_engine(key)
                # create_initial_state cũng tốn thời gian (Silero load JIT model) nên
                # warm luôn một lần rồi bỏ state — state thật là per-session.
                engine.create_initial_state(threshold=threshold)
                results[key] = "ok"
            except Exception as exc:  # noqa: BLE001
                results[key] = f"error: {type(exc).__name__}: {exc}"
        return results

    @classmethod
    def reset_pool(cls) -> None:
        """Giải phóng các instance engine đang cache."""
        with cls._lock:
            cls._engines.clear()


__all__ = [
    "BaseVADEngine",
    "FireRedVADEngine",
    "SileroVADEngine",
    "FsmnVADEngine",
    "VADEngineFactory",
    "SUPPORTED_VAD_ENGINES",
]
