"""Module quản lý danh sách mẫu giọng Clone cho OmniVoice TTS.

Chức năng:
- Quét và nạp metadata từ voices.json và các file .wav/.txt trong backend/voices hoặc fallback backend_cpp/voices.
- Hỗ trợ đổi mẫu giọng nóng (Hot-swapping) không cần khởi động lại model/server.
- Tự động fallback an toàn nếu mẫu giọng yêu cầu không tồn tại.
"""

import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from backend.config import config, VOICES_DIR
from backend.utils.logger import get_logger

logger = get_logger("tts.voice_manager")


class VoiceManager:
    """Quản lý các tệp âm thanh tham chiếu mẫu giọng (Voice Samples)."""

    _cached_voices: Optional[List[Dict[str, Any]]] = None
    _lock = threading.RLock()

    @classmethod
    def get_voices_dir(cls) -> Path:
        """Lấy đường dẫn thư mục chứa các mẫu giọng (VOICES_DIR)."""
        configured = getattr(getattr(config, "tts", None), "voices_dir", None)
        if configured:
            p = Path(configured)
            if p.exists():
                return p
        return VOICES_DIR

    @classmethod
    def get_available_voices(cls, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Quét và lấy danh sách toàn bộ các giọng nói có sẵn (Thread-safe)."""
        with cls._lock:
            if cls._cached_voices is not None and not force_reload:
                return cls._cached_voices

            voices_dir = cls.get_voices_dir()
            voices_dir.mkdir(parents=True, exist_ok=True)
            voices_config_file = voices_dir / "voices.json"

            voices: List[Dict[str, Any]] = []
            registered_files = set()

            # 1. Nạp từ file cấu hình voices.json nếu có
            if voices_config_file.exists():
                try:
                    with open(voices_config_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and item.get("audio"):
                                    audio_path = item["audio"]
                                    p = Path(audio_path)
                                    full_path = p if p.is_absolute() else (voices_dir / p.name)
                                    if full_path.exists():
                                        item_id = item.get("id") or full_path.name
                                        voices.append({
                                            "id": item_id,
                                            "name": item.get("name") or item_id,
                                            "audio": str(full_path),
                                            "text": item.get("text") or "",
                                        })
                                        registered_files.add(full_path.name.lower())
                except Exception as e:
                    logger.warning(f"Không thể đọc voices.json: {e}", extra={"module_tag": "TTS"})

            # 2. Tự động phát hiện các file .wav trong thư mục chưa được đăng ký
            try:
                for file_path in sorted(voices_dir.glob("*.wav")):
                    if file_path.name.lower() not in registered_files:
                        clean_name = file_path.stem.replace("_", " ").replace("-", " ").title()
                        sibling_txt = file_path.with_suffix(".txt")
                        ref_text = ""
                        if sibling_txt.exists():
                            try:
                                ref_text = sibling_txt.read_text(encoding="utf-8").strip()
                            except Exception:
                                pass

                        voices.append({
                            "id": file_path.name,
                            "name": f"🎙️ {clean_name} ({file_path.name})",
                            "audio": str(file_path),
                            "text": ref_text,
                        })
                        registered_files.add(file_path.name.lower())
            except Exception as e:
                logger.warning(f"Lỗi khi quét thư mục giọng mẫu: {e}", extra={"module_tag": "TTS"})

            cls._cached_voices = voices
            return voices

    @classmethod
    def resolve_voice(cls, voice_id_or_path: Optional[str]) -> Tuple[str, str]:
        """Phân giải voice_id hoặc đường dẫn tệp âm thanh thành (abs_audio_path, ref_text)."""
        voices = cls.get_available_voices()
        if not voices:
            logger.warning("Không tìm thấy mẫu giọng nào trong hệ thống.", extra={"module_tag": "TTS"})
            return "", ""

        if not voice_id_or_path:
            return voices[0]["audio"], voices[0]["text"]

        target = str(voice_id_or_path).strip().lower()
        target_name = Path(target).name.lower()

        # Nếu yêu cầu giọng ElevenLabs, ưu tiên bản 10s đã được chuẩn hoá để không phình VRAM
        if target_name in ("elevenlabs.wav", "elevenlabs", "elevenlabs_10s.wav", "elevenlabs_10s"):
            for v in voices:
                if "elevenlabs_10s" in v["id"].lower() or "elevenlabs_10s" in Path(v["audio"]).name.lower():
                    return v["audio"], v["text"]

        # Kiểm tra danh sách giọng đã đăng ký
        for v in voices:
            if v["id"].lower() == target or Path(v["audio"]).name.lower() == target_name:
                return v["audio"], v["text"]

        # Kiểm tra tệp đường dẫn trực tiếp
        p = Path(voice_id_or_path)
        if not p.is_absolute():
            p = cls.get_voices_dir() / p.name

        if p.exists():
            sibling_txt = p.with_suffix(".txt")
            ref_text = ""
            if sibling_txt.exists():
                try:
                    ref_text = sibling_txt.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            return str(p), ref_text

        # Mặc định lấy giọng đầu tiên
        return voices[0]["audio"], voices[0]["text"]
