"""Module quản lý và tự động tải mô hình GGUF cho OmniVoice TTS.

Quy tắc bất biến:
1. CHỈ tải và nạp mô hình từ thư mục `backend/models` (`MODELS_DIR`).
2. Không tải lại nếu file đã tồn tại cục bộ.
3. Đồng bộ tiến trình tải với `backend/utils/model_download.py` để hiển thị trên UI.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.config import config, MODELS_DIR
from backend.utils.model_download import (
    ensure_model_file,
    is_available,
    get_state,
    ModelDownloadError,
    ModelFileMissing,
)
from backend.utils.logger import logger

_TAG = "TTS"


def get_model_filenames() -> Tuple[str, str]:
    """Trả về cặp (tên_file_base, tên_file_tokenizer) được cấu hình."""
    base_name = getattr(config.tts, "model_base", "omnivoice-base-Q8_0.gguf")
    tok_name = getattr(config.tts, "model_tokenizer", "omnivoice-tokenizer-F32.gguf")
    return base_name, tok_name


def get_model_paths() -> Tuple[Path, Path]:
    """Trả về đường dẫn tuyệt đối trong `backend/models/` của hai mô hình GGUF."""
    base_name, tok_name = get_model_filenames()
    return MODELS_DIR / base_name, MODELS_DIR / tok_name


def is_model_available() -> bool:
    """Kiểm tra xem cả 2 file GGUF đã có sẵn trong `backend/models/` hay chưa."""
    repo_id = getattr(config.tts, "repo_id", "Serveurperso/OmniVoice-GGUF")
    base_name, tok_name = get_model_filenames()
    return is_available(repo_id, base_name, MODELS_DIR) and is_available(repo_id, tok_name, MODELS_DIR)


def ensure_tts_models(
    allow_download: Optional[bool] = None,
    progress_interval: float = 10.0,
) -> Tuple[Path, Path]:
    """Đảm bảo cả hai mô hình GGUF đã có trong `backend/models/`.

    Nếu chưa có và `allow_download=True`, tự động tải từ Hugging Face
    repo `Serveurperso/OmniVoice-GGUF` về thẳng `backend/models/`.
    """
    repo_id = getattr(config.tts, "repo_id", "Serveurperso/OmniVoice-GGUF")
    base_name, tok_name = get_model_filenames()

    if allow_download is None:
        allow_download = getattr(config.tts, "auto_download", True)

    logger.debug(
        f"Kiểm tra mô hình GGUF: base={base_name}, tokenizer={tok_name} trong {MODELS_DIR}",
        extra={"module_tag": _TAG},
    )

    base_path_str = ensure_model_file(
        repo_id=repo_id,
        filename=base_name,
        local_dir=MODELS_DIR,
        allow_download=allow_download,
        label=f"OmniVoice Base ({base_name})",
        progress_interval=progress_interval,
        stage="TTS",
    )

    tok_path_str = ensure_model_file(
        repo_id=repo_id,
        filename=tok_name,
        local_dir=MODELS_DIR,
        allow_download=allow_download,
        label=f"OmniVoice Tokenizer ({tok_name})",
        progress_interval=progress_interval,
        stage="TTS",
    )

    return Path(base_path_str), Path(tok_path_str)


def get_download_status() -> Dict[str, Any]:
    """Lấy trạng thái tải hiện tại của cặp mô hình TTS."""
    repo_id = getattr(config.tts, "repo_id", "Serveurperso/OmniVoice-GGUF")
    base_name, tok_name = get_model_filenames()

    base_state = get_state(repo_id, base_name)
    tok_state = get_state(repo_id, tok_name)

    return {
        "is_available": is_model_available(),
        "base_model": {
            "filename": base_name,
            "path": str(MODELS_DIR / base_name),
            "state": (base_state or {}).get("state", "ready" if (MODELS_DIR / base_name).is_file() else "missing"),
            "percent": (base_state or {}).get("percent"),
            "downloaded_bytes": (base_state or {}).get("downloaded_bytes"),
            "total_bytes": (base_state or {}).get("total_bytes"),
        },
        "tokenizer_model": {
            "filename": tok_name,
            "path": str(MODELS_DIR / tok_name),
            "state": (tok_state or {}).get("state", "ready" if (MODELS_DIR / tok_name).is_file() else "missing"),
            "percent": (tok_state or {}).get("percent"),
            "downloaded_bytes": (tok_state or {}).get("downloaded_bytes"),
            "total_bytes": (tok_state or {}).get("total_bytes"),
        },
    }
