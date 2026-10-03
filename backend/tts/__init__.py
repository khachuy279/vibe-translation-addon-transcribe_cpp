"""Package Text-to-Speech (TTS) cho Backend - Native C++ GGUF OmniVoice Voice Cloning."""

from backend.tts.base import BaseTTSEngine
from backend.tts.audio_processor import AudioProcessor
from backend.tts.voice_manager import VoiceManager
from backend.tts.dedup import TTSDedupState
from backend.tts.engine import OmniVoiceTTS
from backend.tts.native import OmniVoiceNative
from backend.tts.worker import TTSWorkerClient
from backend.tts.downloader import (
    ensure_tts_models,
    is_model_available,
    get_model_paths,
    get_download_status,
)


def get_tts_engine() -> BaseTTSEngine:
    """Trả về Singleton instance của OmniVoice TTS Engine."""
    return OmniVoiceTTS.get_instance()


__all__ = [
    "BaseTTSEngine",
    "AudioProcessor",
    "VoiceManager",
    "TTSDedupState",
    "OmniVoiceTTS",
    "OmniVoiceNative",
    "TTSWorkerClient",
    "ensure_tts_models",
    "is_model_available",
    "get_model_paths",
    "get_download_status",
    "get_tts_engine",
]
