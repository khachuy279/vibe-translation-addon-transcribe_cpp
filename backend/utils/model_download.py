"""Tải file GGUF từ HuggingFace về `backend/models` (một lần cho mỗi file).

Bối cảnh: catalog `translation_models.yaml` / `models.yaml` chỉ khai báo **tên file** và
`registry.resolve_*_path()` ghép thẳng vào `MODELS_DIR`. Trước đây nếu file chưa có thì
backend báo lỗi cứng, người dùng phải tự copy GGUF vào thư mục models. Module này bổ sung
đường tải tự động, nhưng có ba ràng buộc cố ý:

1. **Không bao giờ chạy trên hot path** — chỉ `translation.hotswap.activate_model()` (đổi
   model do người dùng yêu cầu) mới bật `allow_download=True`; đường dịch từng câu chỉ kiểm
   tra `os.path.exists()` rồi đi tiếp.
2. **Không tải lại thứ đã có** — file đã tồn tại (kể cả do người dùng copy tay) thì trả về
   ngay, không gọi mạng.
3. **Không tải song song** — mỗi file có một lock riêng, hai lần bấm "đổi model" liên tiếp
   chỉ sinh đúng một lượt tải.

Tiến độ được giữ trong `_STATE` để `/api/config` và popup hiển thị được (state/bytes/percent).
"""

import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional

from backend.config import MODELS_DIR
from backend.utils.logger import logger

try:  # pragma: no cover - phụ thuộc bản cài đặt
    from huggingface_hub import hf_hub_download, list_repo_files

    _HF_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]
    list_repo_files = None  # type: ignore[assignment]
    _HF_IMPORT_ERROR = str(exc)

try:  # pragma: no cover - chỉ để log console sạch
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except Exception:  # noqa: BLE001
    pass

DEFAULT_PROGRESS_INTERVAL_SEC: float = 10.0

_STATE_LOCK = threading.RLock()
_FILE_LOCKS: Dict[str, threading.RLock] = {}
_STATE: Dict[str, Dict[str, Any]] = {}


class ModelDownloadError(RuntimeError):
    """Không tải được model (mạng, repo, tên file...) — thông báo đã đủ rõ cho người dùng."""


class ModelFileMissing(FileNotFoundError):
    """File GGUF chưa có cục bộ và việc tải tự động đang bị tắt."""


def _mb(num_bytes: Optional[float]) -> Optional[float]:
    if num_bytes is None:
        return None
    try:
        return round(float(num_bytes) / (1024.0 * 1024.0), 1)
    except Exception:  # noqa: BLE001
        return None


def state_key(repo_id: str, filename: str) -> str:
    """Khoá trạng thái cho một file cụ thể (repo + tên file)."""
    return f"{repo_id}::{Path(filename).name}"


def local_target_path(filename: str, local_dir: Path = MODELS_DIR) -> Path:
    """Đường dẫn cục bộ mà `registry.resolve_gguf_path()` sẽ trỏ tới."""
    return Path(local_dir) / Path(filename).name


def is_available(repo_id: str, filename: str, local_dir: Path = MODELS_DIR) -> bool:
    """True nếu file đã có cục bộ (không chạm mạng)."""
    try:
        target = local_target_path(filename, local_dir)
    except Exception:  # noqa: BLE001
        return False
    try:
        return target.is_file() and target.stat().st_size > 0
    except OSError:
        return False


def get_state(repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """Trạng thái tải gần nhất của một file (None nếu chưa từng tải trong phiên này)."""
    with _STATE_LOCK:
        item = _STATE.get(state_key(repo_id, filename))
        return dict(item) if item else None


def _set_state(key: str, **fields: Any) -> Dict[str, Any]:
    with _STATE_LOCK:
        item = _STATE.setdefault(key, {"state": "idle"})
        item.update(fields)
        item["updated_at"] = time.time()
        return dict(item)


def _file_lock(key: str) -> threading.RLock:
    with _STATE_LOCK:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _FILE_LOCKS[key] = lock
        return lock


def _resolve_repo_filename(repo_id: str, filename: str) -> str:
    """Tìm đúng đường dẫn file trong repo (catalog có thể chỉ ghi tên file trần)."""
    if list_repo_files is None:  # pragma: no cover
        raise ModelDownloadError(f"Thiếu huggingface_hub ({_HF_IMPORT_ERROR}) nên không tải được model.")
    basename = Path(filename).name
    try:
        files = list(list_repo_files(repo_id))
    except Exception as exc:  # noqa: BLE001
        raise ModelDownloadError(
            f"Không truy cập được repo HuggingFace '{repo_id}' ({type(exc).__name__}: {exc})."
        ) from exc
    if filename in files:
        return filename
    matches = [f for f in files if f.rsplit("/", 1)[-1] == basename]
    if matches:
        return min(matches, key=len)
    raise ModelDownloadError(f"Repo '{repo_id}' không có file '{basename}'.")


def _expected_size_bytes(repo_id: str, repo_filename: str) -> Optional[int]:
    """Kích thước file trên Hub (chỉ để hiển thị % tiến độ). Lỗi mạng ⇒ bỏ qua."""
    try:  # pragma: no cover - phụ thuộc mạng
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
        for sibling in getattr(info, "siblings", None) or []:
            if getattr(sibling, "rfilename", None) == repo_filename:
                size = getattr(sibling, "size", None)
                return int(size) if size else None
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Không đọc được kích thước file trên Hub: {exc}", extra={"module_tag": "TRANSLATE"})
    return None


def _incomplete_bytes(local_dir: Path) -> int:
    """Tổng dung lượng các file `.incomplete` mà huggingface_hub đang ghi (để báo tiến độ)."""
    root = Path(local_dir) / ".cache"
    total = 0
    if not root.exists():
        return 0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(".incomplete"):
                    continue
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    continue
    except Exception:  # noqa: BLE001
        return total
    return total


def _monitor_progress(
    key: str,
    local_dir: Path,
    total_bytes: Optional[int],
    stop: threading.Event,
    interval: float,
    label: str,
    stage: str = "TRANSLATE",
) -> None:
    """Log tiến độ mỗi `interval` giây trong lúc tải (luồng nền, tự dừng khi xong)."""
    started = time.time()
    tag = stage.upper()
    kind = "ASR" if tag == "ASR" else "dịch"
    while not stop.wait(interval):
        done = _incomplete_bytes(local_dir)
        done_mb = _mb(done)
        pct = f"{done / total_bytes * 100:.0f}%" if total_bytes else "?"
        _set_state(key, downloaded_bytes=done, total_bytes=total_bytes, percent=(done / total_bytes * 100.0) if total_bytes else None)
        logger.info(
            f"Đang tải model {kind} '{label}' về máy: {done_mb} MB"
            f"{f'/{_mb(total_bytes)} MB ({pct})' if total_bytes else ''} "
            f"— {time.time() - started:.0f}s",
            extra={"module_tag": tag},
        )


def _move_into_place(downloaded_path: str, target: Path, stage: str = "TRANSLATE") -> Path:
    """Chuẩn hoá vị trí file: `resolve_gguf_path()` chỉ biết `MODELS_DIR/<tên file>`."""
    src = Path(downloaded_path)
    if src.resolve() == target.resolve():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, target)
        return target
    except OSError as exc:  # khác ổ đĩa ⇒ copy rồi xoá
        logger.debug(f"os.replace không dùng được ({exc}); chuyển sang copy.", extra={"module_tag": stage.upper()})
        import shutil

        shutil.copy2(src, target)
        return target


def ensure_model_file(
    repo_id: str,
    filename: str,
    *,
    local_dir: Path = MODELS_DIR,
    allow_download: bool = True,
    label: Optional[str] = None,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL_SEC,
    stage: str = "TRANSLATE",
) -> str:
    """Trả về đường dẫn file GGUF cục bộ, tải từ HuggingFace nếu chưa có.

    - File đã tồn tại ⇒ trả về ngay, KHÔNG gọi mạng.
    - `allow_download=False` mà file thiếu ⇒ `ModelFileMissing` (thông báo có đường dẫn cụ thể).
    - Lỗi mạng/repo ⇒ `ModelDownloadError`.
    """
    if not repo_id or not filename:
        raise ModelDownloadError("Thiếu thông tin repo/file để tải model.")

    target = local_target_path(filename, local_dir)
    if is_available(repo_id, filename, local_dir):
        return str(target)

    tag = stage.upper()
    kind = "ASR" if tag == "ASR" else "dịch"
    cfg_attr = "ASRConfig.auto_download" if tag == "ASR" else "TranslationConfig.auto_download"

    if not allow_download:
        raise ModelFileMissing(
            f"Chưa có file GGUF cục bộ: {target}. "
            f"Có thể bật tự động tải ({cfg_attr}) hoặc tải tay từ "
            f"https://huggingface.co/{repo_id}/blob/main/{filename}."
        )

    if hf_hub_download is None:  # pragma: no cover
        raise ModelDownloadError(f"Thiếu huggingface_hub ({_HF_IMPORT_ERROR}) nên không tải được model.")

    key = state_key(repo_id, filename)
    display = label or Path(filename).name
    with _file_lock(key):
        # Kiểm tra lại sau khi lấy lock: một luồng khác có thể vừa tải xong.
        if is_available(repo_id, filename, local_dir):
            return str(target)

        target.parent.mkdir(parents=True, exist_ok=True)
        repo_filename = _resolve_repo_filename(repo_id, filename)
        total_bytes = _expected_size_bytes(repo_id, repo_filename)
        _set_state(
            key,
            state="downloading",
            model=display,
            stage=stage,
            repo_id=repo_id,
            filename=Path(filename).name,
            downloaded_bytes=0,
            total_bytes=total_bytes,
            percent=0.0 if total_bytes else None,
            error=None,
            started_at=time.time(),
        )
        logger.info(
            f"Model {kind} '{display}' chưa có ở {target} — bắt đầu tải từ '{repo_id}'"
            f"{f' ({_mb(total_bytes)} MB)' if total_bytes else ''}. Model hiện tại vẫn hoạt động bình thường.",
            extra={"module_tag": tag},
        )

        stop = threading.Event()
        monitor = threading.Thread(
            target=_monitor_progress,
            args=(key, Path(local_dir), total_bytes, stop, max(1.0, float(progress_interval)), display, stage),
            name="model_download_progress",
            daemon=True,
        )
        monitor.start()

        t0 = time.perf_counter()
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=repo_filename,
                local_dir=str(local_dir),
            )
        except Exception as exc:  # noqa: BLE001
            stop.set()
            _set_state(key, state="error", error=f"{type(exc).__name__}: {exc}")
            raise ModelDownloadError(
                f"Tải model '{display}' từ '{repo_id}' thất bại: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            stop.set()

        final_path = _move_into_place(str(downloaded), target, stage=stage)
        elapsed = time.perf_counter() - t0
        size = None
        try:
            size = final_path.stat().st_size
        except OSError:
            pass
        _set_state(
            key,
            state="ready",
            downloaded_bytes=size,
            total_bytes=size or total_bytes,
            percent=100.0 if size else None,
            finished_at=time.time(),
            elapsed_sec=round(elapsed, 1),
        )
        logger.info(
            f"Đã tải xong model {kind} '{display}': {final_path} ({_mb(size)} MB) trong {elapsed:.0f}s.",
            extra={"module_tag": tag},
        )
        return str(final_path)
