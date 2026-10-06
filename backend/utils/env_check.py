"""Kiểm tra môi trường runtime của dự án — KHÔNG còn liên quan tới PyTorch.

BỐI CẢNH: trước Giai đoạn 3 (2026-10-06), module này tồn tại để chống việc pip âm thầm hạ `torch`
xuống bản CPU-only (`silero-vad` chỉ khai `torch` không giới hạn ⇒ pip phân giải lại và lấy bản
CPU từ PyPI ⇒ CUDA của ASR/TTS/dịch biến mất im lặng). Sau khi VAD chuyển sang **onnxruntime** và
ForcedAligner chuyển sang **GGUF/CrispASR**, **không subsystem nào cần PyTorch** nên toàn bộ logic
đó đã bị xoá. Nay module kiểm tra đúng những runtime thật sự có mặt:

* `onnxruntime` + file model ONNX của VAD (Silero / FireRed) — BẮT BUỘC.
* DLL CrispASR + model GGUF của ForcedAligner — cần cho Pipeline B (Lookahead).
* `llama_cpp` (binding + DLL CUDA trong `backend/bin/llama/`) — engine dịch.
* binding `transcribe-cpp` — backend ASR native.

Module này cung cấp:
* `runtime_status()` — ảnh chụp nhanh trạng thái runtime.
* `check_environment()` — danh sách vấn đề (rỗng = OK).
* CLI: `python -m backend.utils.env_check` → in báo cáo, exit code 1 nếu có vấn đề.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import MODELS_DIR

#: Các engine VAD và file model ONNX tương ứng (đều BẮT BUỘC cho engine đang chọn).
_VAD_MODEL_FILES: Dict[str, List[str]] = {
    "firered-vad": ["firered_stream/Stream-VAD/cmvn.ark",
                    "firered_stream/Stream-VAD/fireredvad_stream_vad_with_cache.onnx"],
    "silero-vad": ["silero_vad.onnx"],
}


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def runtime_status() -> Dict[str, Any]:
    """Ảnh chụp nhanh trạng thái runtime. Không ném lỗi khi thiếu gói."""
    status: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "onnxruntime": None,
        "onnxruntime_providers": [],
        "vad_engine": None,
        "vad_models_present": None,
        "aligner_backend": None,
        "aligner_ready": False,
        "aligner_reason": "",
        "llama_cpp": _find_spec("llama_cpp"),
        "transcribe_cpp": _find_spec("transcribe_cpp"),
        #: Chỉ để báo cáo. Sau Giai đoạn 3 nó PHẢI là False.
        "torch_installed": _find_spec("torch"),
        "problems": [],
    }

    try:
        from backend.config import config

        status["vad_engine"] = config.vad.vad_engine
        status["aligner_backend"] = getattr(config.forced_aligner, "backend", None)
    except Exception as exc:  # noqa: BLE001
        status["problems"].append(f"không đọc được config: {type(exc).__name__}: {exc}")

    # ── onnxruntime (runtime của VAD) ────────────────────────────────────────────
    if not _find_spec("onnxruntime"):
        status["problems"].append(
            "thiếu `onnxruntime` ⇒ KHÔNG engine VAD nào chạy được (cả Silero lẫn FireRed đều chạy "
            "ONNX). Sửa: `pip install onnxruntime`."
        )
    else:
        try:
            import onnxruntime as ort  # noqa: PLC0415

            status["onnxruntime"] = ort.__version__
            status["onnxruntime_providers"] = list(ort.get_available_providers())
        except Exception as exc:  # noqa: BLE001
            status["problems"].append(f"onnxruntime cài rồi nhưng lỗi khi nạp: {type(exc).__name__}: {exc}")

    # ── file model VAD của engine đang chọn ──────────────────────────────────────
    engine = (status["vad_engine"] or "firered-vad").lower().strip()
    expected = _VAD_MODEL_FILES.get(engine, [])
    if expected:
        missing = [f for f in expected if not (Path(MODELS_DIR) / f).is_file()]
        status["vad_models_present"] = not missing
        if missing:
            status["problems"].append(
                f"thiếu file model ONNX cho VAD '{engine}': {', '.join(missing)} ⇒ engine sẽ không "
                f"nạp được. Chạy backend một lần để tự tải."
            )

    # ── CrispASR aligner (Pipeline B) ────────────────────────────────────────────
    try:
        from backend.utils.crispasr_native import available as _fa_available  # noqa: PLC0415

        ok, reason = _fa_available()
        status["aligner_ready"] = bool(ok)
        status["aligner_reason"] = reason
        if not ok:
            # KHÔNG phải "problem" cứng: thiếu aligner thì extension tự chuyển sang Pipeline A.
            status["problems"].append(
                f"ForcedAligner (Lookahead) chưa sẵn sàng: {reason} ⇒ Pipeline B không có mốc từ, "
                f"extension sẽ dùng Pipeline A."
            )
    except Exception as exc:  # noqa: BLE001
        status["aligner_reason"] = f"{type(exc).__name__}: {exc}"
        status["problems"].append(f"không kiểm tra được runtime CrispASR: {type(exc).__name__}: {exc}")

    return status


def check_llama_cpp() -> Optional[str]:
    """Kiểm tra `llama_cpp` nạp được không. Trả về thông báo lỗi (kèm gợi ý) hoặc None.

    Hai lớp lỗi đã gặp thực tế:

    1. **Thiếu runtime CUDA**: `llama.dll` → `ggml.dll` → `ggml-cuda.dll` cần
       `cudart64_13.dll`/`cublas64_13.dll`/`cublasLt64_13.dll`. Phải gọi `setup_cuda_dll_paths()`
       TRƯỚC khi import (đúng như `backend/asr/native.py` làm).
    2. **Wheel build bằng AVX-512**: import được nhưng `llama_init_from_model` nổ
       `OSError: [WinError -1073741795] Windows Error 0xc000001d` (STATUS_ILLEGAL_INSTRUCTION)
       trên CPU không có AVX-512. Lớp này KHÔNG thể phát hiện bằng import — xem README mục
       Khắc phục sự cố.
    """
    if not _find_spec("llama_cpp"):
        return None

    # BẮT BUỘC: đăng ký đường dẫn DLL CUDA (và chọn thư mục DLL llama.cpp) trước khi nạp
    # llama.dll — giống hệt `backend/translation/engine.py`.
    try:
        from backend.utils.cuda import setup_cuda_dll_paths  # noqa: PLC0415

        setup_cuda_dll_paths()
    except Exception:  # noqa: BLE001
        pass

    try:
        import llama_cpp  # noqa: F401,PLC0415

        return None
    except Exception as exc:  # noqa: BLE001
        return (
            f"llama.cpp: không import được llama_cpp ({type(exc).__name__}: {exc}). Kiểm tra: "
            "`backend/bin/llama/llama.dll` có tồn tại không (bundle CUDA đi kèm repo), và runtime "
            "CUDA 13 (`cudart64_13.dll`/`cublas64_13.dll`/`cublasLt64_13.dll`) có trong "
            "`external/cuda-toolkit` hoặc `backend/bin/` không. Nếu lỗi xuất hiện ở bước NẠP MODEL "
            "với `0xc000001d` thì đó là STATUS_ILLEGAL_INSTRUCTION do AVX-512 — bản trong "
            "`backend/bin/llama/` đã tắt AVX-512 nên không gặp."
        )


def check_asr_binding() -> Optional[str]:
    """Binding `transcribe-cpp` có mặt không (thiếu ⇒ KHÔNG có backend ASR nào).

    Đã gặp thực tế trong venv sạch: chỉ cài `transcribe-cpp-native` (provider) mà quên binding
    `transcribe-cpp` ⇒ log `transcribe_cpp package chưa được cài đặt!`,
    `/health → asr_runtime.devices = "n/a"`, và cảnh báo "không backend nào trong ('cuda',
    'vulkan') khả dụng" dù `backend/bin/ggml-cuda.dll` còn nguyên.
    """
    if _find_spec("transcribe_cpp"):
        return None
    if not _find_spec("transcribe_cpp_native"):
        return None  # không dùng ASR native → không phải vấn đề của checker này
    return (
        "thiếu binding `transcribe-cpp` (chỉ có `transcribe-cpp-native`) ⇒ backend không có "
        "backend ASR nào (devices=n/a). Sửa: `pip install \"transcribe-cpp>=0.2.3\"`."
    )


def check_environment() -> List[str]:
    """Danh sách vấn đề của môi trường (rỗng = OK)."""
    problems = list(runtime_status().get("problems") or [])
    for extra in (check_llama_cpp(), check_asr_binding()):
        if extra:
            problems.append(extra)
    return problems


def format_report(status: Optional[Dict[str, Any]] = None) -> str:
    """Báo cáo nhiều dòng cho người đọc (dùng cho CLI và log)."""
    st = status or runtime_status()
    lines = [
        "== Môi trường runtime của dự án (không còn PyTorch) ==",
        f"python        : {st['python']}",
        f"onnxruntime   : {st['onnxruntime'] or 'THIẾU'}  ({', '.join(st['onnxruntime_providers']) or 'n/a'})",
        f"VAD engine    : {st['vad_engine']}  (model ONNX đủ: {st['vad_models_present']})",
        f"aligner       : {st['aligner_backend']}  (sẵn sàng: {st['aligner_ready']}"
        + (f" — {st['aligner_reason']}" if st.get("aligner_reason") else "")
        + ")",
        f"llama_cpp     : {'cài rồi' if st['llama_cpp'] else 'THIẾU'}",
        f"transcribe-cpp: {'cài rồi' if st['transcribe_cpp'] else 'THIẾU'}",
        f"torch         : {'CÒN CÀI (không subsystem nào cần)' if st['torch_installed'] else 'không có (đúng)'}",
    ]
    problems = list(st["problems"])
    llama = check_llama_cpp()
    if llama:
        problems.append(llama)
    asr_binding = check_asr_binding()
    if asr_binding:
        problems.append(asr_binding)
    lines.append(f"llama_cpp nạp : {'OK' if llama is None else 'LỖI (xem bên dưới)'}")
    lines.append(f"ASR binding   : {'OK' if asr_binding is None else 'THIẾU transcribe-cpp'}")
    if st["torch_installed"]:
        lines.append(
            "GỢI Ý: gỡ PyTorch để nhẹ môi trường — `pip uninstall torch torchaudio transformers`."
        )
    if problems:
        lines.append("VẤN ĐỀ:")
        lines.extend(f"  - {p}" for p in problems)
    else:
        lines.append("OK: không phát hiện vấn đề.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: in báo cáo; `--json` để lấy máy đọc; exit 1 nếu có vấn đề."""
    args = list(sys.argv[1:] if argv is None else argv)
    status = runtime_status()
    if "--json" in args:
        status["llama_cpp_error"] = check_llama_cpp()
        status["asr_binding_error"] = check_asr_binding()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 1 if (status["problems"] or status["llama_cpp_error"] or status["asr_binding_error"]) else 0
    print(format_report(status))
    return 1 if check_environment() else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
