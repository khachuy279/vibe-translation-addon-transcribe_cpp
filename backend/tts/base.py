"""Interface và Protocol trừu tượng cho các TTS Engine (Dependency Inversion Principle)."""

from typing import Optional, Tuple, Protocol, runtime_checkable


@runtime_checkable
class BaseTTSEngine(Protocol):
    """Protocol chuẩn hóa giao diện cho mọi Text-to-Speech Engine."""

    sample_rate: int

    def load_model(self) -> None:
        """Nạp và warm-up mô hình vào bộ nhớ/GPU."""
        ...

    def unload_model(self) -> None:
        """Giải phóng hoàn toàn mô hình khỏi RAM/VRAM."""
        ...

    def synthesize_sync(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Tổng hợp giọng nói đồng bộ, trả về (audio_base64, duration_seconds)."""
        ...

    async def synthesize_clone(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Tổng hợp giọng nói bất đồng bộ (non-blocking thread pool)."""
        ...
