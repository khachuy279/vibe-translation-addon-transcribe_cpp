"""Dịch vụ Speaker Diarization sử dụng Nemotron-3-Diarization (GGUF BF16) qua audio.cpp native CLI (CUDA).

Hỗ trợ:
- Gán nhãn người nói (speaker_0, speaker_1, ...) cho từng câu phụ đề dựa trên mốc thời gian [start_sec, end_sec].
- Chạy 100% C++ GGML/CUDA native ngoài tiến trình chính — không xung đột DLL và không dùng PyTorch.
- Tự động map nhãn speaker thân thiện ("speaker 1", "speaker 2", ...) gửi sang LLM dịch.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import zipfile

import numpy as np

from backend.config import BACKEND_DIR, config
from backend.utils.logger import logger
from backend.utils.model_download import sha256_file, verify_sha256

_TAG = "DIAR"
_DIAR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemotron_diar")

AUDIOCPP_RELEASE_TAG = "v0.9.1"
AUDIOCPP_BIN_URL = f"https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_RELEASE_TAG}/audio-{AUDIOCPP_RELEASE_TAG}-bin-windows-x64-cuda12.4.zip"
AUDIOCPP_CUDART_URL = f"https://github.com/0xShug0/audio.cpp/releases/download/{AUDIOCPP_RELEASE_TAG}/audio-{AUDIOCPP_RELEASE_TAG}-cudart-windows-x64-cuda12.4.zip"
NEMOTRON_MODEL_URL = "https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF/resolve/main/nemotron-3-diarization-bf16.gguf"

#: SHA-256 của từng artifact, lấy từ NGUỒN CHÍNH THỨC (không tự băm file cục bộ — băm file
#: trên máy chỉ chứng minh "file khớp với chính nó", không chứng minh nguồn gốc):
#:   • 2 file zip  — trường `digest` do GitHub API công bố cho release v0.9.1
#:                   (`GET /repos/0xShug0/audio.cpp/releases/tags/v0.9.1`).
#:   • model GGUF  — LFS `oid` của repo HuggingFace `audio-cpp/Nemotron-3-Diarization-GGUF`;
#:                   với Git LFS, `oid` CHÍNH LÀ SHA-256 của nội dung file.
#: Đã đối chiếu chéo: hash cục bộ của `nemotron-3-diarization-bf16.gguf` khớp đúng LFS oid.
AUDIOCPP_BIN_SHA256 = "f32a40f8fb14ac4772c9c25525f97715db228979178e51654da73aa65005c0ef"
AUDIOCPP_CUDART_SHA256 = "8bfdce7cb00b5a51560b5ab0d444344d86a2f0d7e2727bb7f18c15ef4734451b"
NEMOTRON_MODEL_SHA256 = "84f6f0f12b9ecf2615548427e884534a72fedbca32d00872d95c0f72a078c953"


def _fetch_verified(url: str, target: Path, expected_sha256: str, label: str) -> bool:
    """Tải `url` về `target` rồi TỪ CHỐI file nếu SHA-256 không khớp.

    VÌ SAO CẦN: trước đây `ensure_audiocpp_binaries()` và `ensure_diarization_model()` chỉ kiểm
    tra **kích thước** file (> 100 MB). Một bản tải hỏng, bị cắt cụt, hoặc bị tráo từ nguồn khác
    vẫn qua được vòng kiểm tra đó rồi được giải nén và nạp vào tiến trình.

    CHỈ xác thực ở LƯỢT TẢI (ranh giới supply-chain), KHÔNG băm lại file đã có sẵn lúc khởi
    động: `ggml-cuda.dll` nặng 1,09 GB, băm mỗi lần khởi động sẽ cộng thêm vài giây vô ích.
    """
    urllib.request.urlretrieve(url, target)
    if verify_sha256(target, expected_sha256):
        logger.info(f"[STARTUP] {label}: tải xong, SHA-256 khớp.", extra={"module_tag": _TAG})
        return True

    got = sha256_file(target) if target.is_file() else "<không có file>"
    logger.error(
        f"[STARTUP] {label}: SHA-256 KHÔNG khớp — TỪ CHỐI dùng file vừa tải.\n"
        f"  mong đợi : {expected_sha256}\n"
        f"  nhận được: {got}",
        extra={"module_tag": _TAG},
    )
    return False


def ensure_audiocpp_binaries() -> bool:
    """Tự động tải và giải nén binary audio.cpp (CUDA) nếu chưa có."""
    cli_path = BACKEND_DIR / "bin" / "audiocpp" / "audiocpp_cli.exe"
    cuda_dll = BACKEND_DIR / "bin" / "audiocpp" / "ggml-cuda.dll"
    if cli_path.is_file() and cuda_dll.is_file():
        return True

    dest_dir = BACKEND_DIR / "bin" / "audiocpp"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[STARTUP] Đang tải audio.cpp runtime ({AUDIOCPP_RELEASE_TAG} CUDA) tự động khi thiếu...",
        extra={"module_tag": _TAG},
    )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_bin_zip = Path(tmp_dir) / "bin.zip"
            tmp_cudart_zip = Path(tmp_dir) / "cudart.zip"

            if not cli_path.is_file():
                if not _fetch_verified(
                    AUDIOCPP_BIN_URL, tmp_bin_zip, AUDIOCPP_BIN_SHA256, "runtime audio.cpp"
                ):
                    return False
                with zipfile.ZipFile(tmp_bin_zip, "r") as zf:
                    zf.extractall(dest_dir)

            if not cuda_dll.is_file():
                if not _fetch_verified(
                    AUDIOCPP_CUDART_URL, tmp_cudart_zip, AUDIOCPP_CUDART_SHA256, "cudart audio.cpp"
                ):
                    return False
                with zipfile.ZipFile(tmp_cudart_zip, "r") as zf:
                    zf.extractall(dest_dir)

        logger.info("[STARTUP] Tải runtime audio.cpp hoàn tất thành công!", extra={"module_tag": _TAG})
        return True
    except Exception as exc:
        logger.warning(f"Không thể tự động tải runtime audio.cpp: {exc}", extra={"module_tag": _TAG})
        return False


def ensure_diarization_model(model_filename: str = "nemotron-3-diarization-bf16.gguf") -> bool:
    """Tự động tải model Nemotron-3-Diarization GGUF từ HuggingFace nếu chưa có."""
    model_path = BACKEND_DIR / "models" / model_filename
    if model_path.is_file() and model_path.stat().st_size > 100_000_000:
        return True

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"[STARTUP] Đang tải model {model_filename} (~190MB) từ HuggingFace...",
        extra={"module_tag": _TAG},
    )
    try:
        tmp_model = model_path.with_suffix(".download.tmp")
        if not _fetch_verified(
            NEMOTRON_MODEL_URL, tmp_model, NEMOTRON_MODEL_SHA256, f"model {model_filename}"
        ):
            # Không để lại file rác đã bị từ chối trong `backend/models/`.
            tmp_model.unlink(missing_ok=True)
            return False
        if tmp_model.is_file() and tmp_model.stat().st_size > 100_000_000:
            tmp_model.replace(model_path)
            logger.info(f"[STARTUP] Tải model {model_filename} hoàn tất!", extra={"module_tag": _TAG})
            return True
        return False
    except Exception as exc:
        logger.warning(f"Không thể tải model {model_filename}: {exc}", extra={"module_tag": _TAG})
        return False


def _find_audiocpp_cli() -> Optional[str]:
    """Tìm binary audiocpp_cli.exe trong backend/bin/audiocpp hoặc external/bin."""
    candidates = [
        BACKEND_DIR / "bin" / "audiocpp" / "audiocpp_cli.exe",
        BACKEND_DIR.parent / "external" / "bin" / "audiocpp_cli.exe",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    if ensure_audiocpp_binaries():
        main_p = BACKEND_DIR / "bin" / "audiocpp" / "audiocpp_cli.exe"
        if main_p.is_file():
            return str(main_p)
    return None


def _find_model_path(model_filename: str) -> Optional[str]:
    """Tìm file model GGUF trong backend/models."""
    p = BACKEND_DIR / "models" / model_filename
    if p.is_file() and p.stat().st_size > 100_000_000:
        return str(p)
    if ensure_diarization_model(model_filename):
        if p.is_file():
            return str(p)
    return None


def diarize_audio_sync(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    model_filename: str = "nemotron-3-diarization-bf16.gguf",
    latency_profile: str = "low",
    backend: str = "cuda",
    threads: int = 4,
) -> List[Dict[str, Any]]:
    """Phân tách người nói (Diarization) trên mảng PCM Float32 [-1.0, 1.0].

    Trả về danh sách các turn:
    [
        {"start_sample": int, "end_sample": int, "speaker_id": "speaker_0", "confidence": float},
        ...
    ]
    """
    if audio_data is None or len(audio_data) < sample_rate * 0.5:
        return []

    cli_path = _find_audiocpp_cli()
    if not cli_path:
        logger.warning("Không tìm thấy audiocpp_cli.exe, bỏ qua Diarization.", extra={"module_tag": _TAG})
        return []

    model_path = _find_model_path(model_filename)
    if not model_path:
        logger.warning(f"Không tìm thấy model {model_filename}, bỏ qua Diarization.", extra={"module_tag": _TAG})
        return []

    import soundfile as sf

    # Chuẩn bị file tạm cho audio và file kết quả json
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_wav, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_turns:
        wav_path = f_wav.name
        turns_path = f_turns.name

    try:
        # Ghi WAV 16kHz PCM 16-bit
        sf.write(wav_path, audio_data, sample_rate, format="WAV", subtype="PCM_16")

        cmd = [
            cli_path,
            "--task", "diar",
            "--family", "nemotron_3_diar",
            "--model", model_path,
            "--backend", backend,
            "--threads", str(threads),
            "--session-option", f"nemotron_3_diar.latency_profile={latency_profile}",
            "--audio", wav_path,
            "--turns-out", turns_path,
        ]

        t0 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        if res.returncode != 0:
            logger.warning(f"audiocpp_cli trả về mã lỗi {res.returncode}: {res.stderr.strip()}", extra={"module_tag": _TAG})
            return []

        if not os.path.exists(turns_path) or os.path.getsize(turns_path) == 0:
            return []

        with open(turns_path, "r", encoding="utf-8") as f:
            turns = json.load(f)

        logger.debug(
            f"Diarization hoàn tất trong {dur_ms:.0f}ms: phát hiện {len(turns)} turns.",
            extra={"module_tag": _TAG},
        )
        return turns

    except Exception as exc:
        logger.warning(f"Lỗi khi thực thi Diarization: {exc}", extra={"module_tag": _TAG})
        return []
    finally:
        for p in (wav_path, turns_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def assign_speakers_to_subtitles(
    subtitles_with_times: List[Tuple[float, float]],
    turns: List[Dict[str, Any]],
    sample_rate: int = 16000,
) -> List[str]:
    """Gán nhãn speaker ('speaker 1', 'speaker 2', ...) cho từng câu phụ đề [start_sec, end_sec].

    Thuật toán:
    1. Đo thời gian overlap giữa câu phụ đề và các turn nói của từng speaker.
    2. Speaker có thời lượng overlap lớn nhất trong câu sẽ được chọn.
    3. Nếu câu rơi vào khoảng lặng nhỏ (gap) giữa các turns, chọn speaker của turn gần nhất.
    4. Ánh xạ speaker_id ('speaker_0', 'speaker_1', ...) sang định dạng thân thiện ('speaker 1', 'speaker 2', ...).
    """
    if not subtitles_with_times:
        return []
    if not turns:
        return ["speaker 1"] * len(subtitles_with_times)

    # Bảng ánh xạ: speaker_0 -> "speaker 1", speaker_1 -> "speaker 2", ...
    # Giữ thứ tự xuất hiện nhất quán
    speaker_map: Dict[str, str] = {}
    next_speaker_num = 1

    assigned: List[str] = []
    for sub_start, sub_end in subtitles_with_times:
        start_samp = int(sub_start * sample_rate)
        end_samp = int(sub_end * sample_rate)
        mid_samp = (start_samp + end_samp) // 2

        # Tính overlap với từng speaker
        overlaps: Dict[str, int] = {}
        for t in turns:
            t_start = t["start_sample"]
            t_end = t["end_sample"]
            ov = max(0, min(end_samp, t_end) - max(start_samp, t_start))
            if ov > 0:
                spk = t["speaker_id"]
                overlaps[spk] = overlaps.get(spk, 0) + ov

        if overlaps:
            best_raw_spk = max(overlaps, key=overlaps.get)
        else:
            # Fallback: tìm turn gần mid_samp nhất
            min_dist = float("inf")
            best_raw_spk = "speaker_0"
            for t in turns:
                d = min(abs(t["start_sample"] - mid_samp), abs(t["end_sample"] - mid_samp))
                if d < min_dist:
                    min_dist = d
                    best_raw_spk = t["speaker_id"]

        if best_raw_spk not in speaker_map:
            speaker_map[best_raw_spk] = f"speaker {next_speaker_num}"
            next_speaker_num += 1

        assigned.append(speaker_map[best_raw_spk])

    return assigned


class DiarizationService:
    """Singleton service điều phối Diarization."""

    @classmethod
    async def diarize_chunk(
        cls,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[Dict[str, Any]]:
        """Async wrapper gọi diarize_audio_sync trên executor."""
        diar_cfg = getattr(config, "diarization", None)
        if not diar_cfg or not getattr(diar_cfg, "enabled", True):
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _DIAR_EXECUTOR,
            diarize_audio_sync,
            audio_data,
            sample_rate,
            getattr(diar_cfg, "active_model", "nemotron-3-diarization-bf16.gguf"),
            getattr(diar_cfg, "latency_profile", "low"),
            getattr(diar_cfg, "backend", "cuda"),
            getattr(diar_cfg, "threads", 4),
        )
