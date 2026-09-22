"""Adapter chuẩn hóa tùy chọn và mã ngôn ngữ cho từng họ mô hình trong transcribe.cpp."""

import logging
from typing import Any, Dict, Optional

try:
    import transcribe_cpp
except ImportError:
    transcribe_cpp = None

from backend.utils.logger import logger

# Bảng map ngôn ngữ 2 ký tự sang chuẩn BCP-47 locale cho Nemotron / Parakeet
_NEMOTRON_LOCALE_MAP: Dict[str, str] = {
    "en": "en-US",
    "zh": "zh-CN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "ru": "ru-RU",
    "nl": "nl-NL",
    "hi": "hi-IN",
    "ar": "ar-AR",
    "pl": "pl-PL",
    "sv": "sv-SE",
    "tr": "tr-TR",
    "uk": "uk-UA",
    "cs": "cs-CZ",
    "id": "id-ID",
    "vi": "vi-VN",
    "th": "th-TH",
    "ro": "ro-RO",
    "hu": "hu-HU",
    "el": "el-GR",
    "da": "da-DK",
    "fi": "fi-FI",
    "no": "no-NO",
}

_SENSEVOICE_LANGUAGES = {"zh", "en", "ja", "ko", "yue"}


def normalize_language_for_family(lang: Optional[str], family: str) -> Optional[str]:
    """Chuẩn hóa mã ngôn ngữ theo yêu cầu cụ thể của từng họ mô hình."""
    if not lang or lang.strip().lower() in ("auto", "none"):
        return None

    clean = lang.strip().replace("_", "-")
    family_norm = (family or "").strip().lower()

    if family_norm in ("nemotron", "parakeet"):
        clean_lower = clean.lower()
        if clean_lower in _NEMOTRON_LOCALE_MAP:
            return _NEMOTRON_LOCALE_MAP[clean_lower]
        for k, v in _NEMOTRON_LOCALE_MAP.items():
            if clean_lower == v.lower():
                return v
        if "-" in clean:
            parts = clean.split("-")
            return f"{parts[0].lower()}-{parts[1].upper()}"
        return None

    if family_norm == "sensevoice":
        base_code = clean.split("-")[0].lower()
        return base_code if base_code in _SENSEVOICE_LANGUAGES else None

    base_code = clean.split("-")[0].lower()
    return base_code


def build_family_options(
    family: str,
    model_info: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    slot: Optional[str] = None,
) -> Optional[Any]:
    """Khởi tạo FamilyExtension options phù hợp cho từng model family."""
    if transcribe_cpp is None:
        return None

    info = model_info or {}
    fam = (family or "").lower().strip()

    try:
        if fam in ("nemotron", "parakeet"):
            att_context_right = int(info.get("att_context_right", 1))
            opt = transcribe_cpp.NemotronOptions(att_context_right=att_context_right)
            return opt

        elif fam == "voxtral":
            num_delay_tokens = int(info.get("num_delay_tokens", 2))
            opt = transcribe_cpp.VoxtralOptions(num_delay_tokens=num_delay_tokens)
            return opt

        elif fam == "whisper":
            # FIX: tên lớp THẬT trong binding là `WhisperRunOptions` (không phải
            # `WhisperOptions`). Trước đây `AttributeError` bị `except` nuốt nên luôn trả
            # None ⇒ mọi tuỳ chọn whisper (initial_prompt, temperature…) âm thầm bị bỏ.
            opt = transcribe_cpp.WhisperRunOptions()
            return opt

    except Exception as e:
        logger.debug(f"Không thể khởi tạo FamilyExtension cho {family}: {e}", extra={"module_tag": "ASR"})

    return None
