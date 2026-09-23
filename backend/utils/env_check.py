"""Kiểm tra môi trường Python của dự án — chống việc pip âm thầm hạ torch CUDA.

BỐI CẢNH (đã xảy ra 2026-09-22): interpreter Python dùng CHUNG với nhiều gói khác có ràng
buộc torch xung đột (whisperx 3.8.6 → `torch~=2.8.0`; compressed-tensors → `torch>=2.10.0`;
torchvision cu130 → `torch==2.12.0`). Chỉ cần `pip install -U silero-vad` (metadata chỉ ghi
`torch`, không giới hạn) là pip phân giải lại và lấy **torch CPU-only từ PyPI**
(`2.12.0+cu130` → `2.9.1+cpu`) ⇒ CUDA của ASR/TTS/dịch biến mất mà không có lỗi rõ ràng.

Module này cung cấp:
* `torch_status()` — ảnh chụp nhanh trạng thái torch (version, bản CUDA, khả dụng GPU, các
  cặp version lệch nhau). Dùng cho log lúc khởi động và `/health`.
* `check_environment()` — danh sách vấn đề (rỗng = OK).
* CLI: `python -m backend.utils.env_check` → in báo cáo, exit code 1 nếu có vấn đề.

Không import `torch` ở cấp module (nặng) — chỉ import trong hàm.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

#: Bộ ba torch/torchaudio/torchvision mà dự án đã đo và khoá trong `backend/constraints.txt`.
#: ⚠️ Số phiên bản KHÔNG bằng nhau: index cu130 chỉ có torchaudio tới 2.11.x (torchaudio đi sau
#: torch một bậc) ⇒ torch 2.12.0 + torchaudio 2.11.0 + torchvision 0.27.0 là bộ hợp lệ.
EXPECTED_CUDA_TRIO = ("2.14.0+cu130", "2.11.0+cu130", "0.29.0+cu130")


def cuda_tag(version: Optional[str]) -> Optional[str]:
    """Tag CUDA của một bản phát hành: `'2.12.0+cu130'` → `'+cu130'`; bản CPU → `None`."""
    text = str(version or "")
    idx = text.find("+cu")
    return text[idx:] if idx >= 0 else None


def torch_status() -> Dict[str, Any]:
    """Ảnh chụp nhanh trạng thái torch. Không ném lỗi nếu thiếu torch."""
    status: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_build": None,          # torch.version.cuda (None = bản CPU-only)
        "cuda_available": False,
        "torchaudio": None,
        "torchvision": None,
        "cuda_device_count": 0,
        "problems": [],
    }
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        status["problems"].append(f"không import được torch: {type(exc).__name__}: {exc}")
        return status

    status["installed"] = True
    status["version"] = getattr(torch, "__version__", None)
    status["cuda_build"] = getattr(torch.version, "cuda", None)
    try:
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["cuda_device_count"] = int(torch.cuda.device_count()) if status["cuda_available"] else 0
    except Exception as exc:  # noqa: BLE001
        status["problems"].append(f"torch.cuda lỗi: {type(exc).__name__}: {exc}")

    for mod_name in ("torchaudio", "torchvision"):
        try:
            mod = __import__(mod_name)
            status[mod_name] = getattr(mod, "__version__", None)
        except Exception as exc:  # noqa: BLE001
            status[mod_name] = None
            # Có cài nhưng import hỏng = dấu hiệu lệch bộ (vd torchvision cu130 với torch CPU).
            try:
                import importlib.util  # noqa: PLC0415

                if importlib.util.find_spec(mod_name) is not None:
                    status["problems"].append(
                        f"{mod_name} đã cài nhưng import lỗi ({type(exc).__name__}: {exc}) "
                        "⇒ thường là lệch bộ CUDA/CPU."
                    )
            except Exception:  # noqa: BLE001
                pass

    # ── Các bất thường hay gặp ────────────────────────────────────────────────
    if not status["cuda_build"]:
        status["problems"].append(
            "torch đang là bản CPU-only (torch.version.cuda = None) ⇒ TTS/dịch/ASR-CUDA mất GPU. "
            "Xem README mục 'Nâng cấp gói phụ thuộc mà không phá torch CUDA'."
        )
    elif not status["cuda_available"]:
        status["problems"].append(
            f"torch có bản CUDA {status['cuda_build']} nhưng torch.cuda.is_available() = False "
            "(driver/CUDA runtime không khớp)."
        )

    version = str(status["version"] or "")
    torch_tag = cuda_tag(version)

    # So theo TAG CUDA, KHÔNG so số phiên bản: torchaudio/torchvision hợp lệ khi lệch số
    # (torchaudio 2.11.0 đi cùng torch 2.12.0) nhưng BẮT BUỘC cùng bộ CUDA.
    for name in ("torchaudio", "torchvision"):
        got = status.get(name)
        if not got:
            continue
        tag = cuda_tag(got)
        if torch_tag and not tag:
            status["problems"].append(
                f"{name} {got} là bản CPU trong khi torch là bản CUDA {torch_tag} ⇒ cài lại cùng bộ "
                "(`torch==2.12.0+cu130 torchaudio==2.11.0+cu130 torchvision==0.27.0+cu130`)."
            )
        elif tag and not torch_tag:
            status["problems"].append(
                f"{name} {got} là bản CUDA nhưng torch {version} là bản CPU ⇒ lệch bộ, cài lại cả ba."
            )
        elif tag and torch_tag and tag != torch_tag:
            status["problems"].append(
                f"{name} {got} dùng {tag} còn torch dùng {torch_tag} ⇒ lệch bộ CUDA."
            )
    return status


def check_llama_cpp() -> Optional[str]:
    """Kiểm tra `llama_cpp` nạp được không. Trả về thông báo lỗi (kèm gợi ý) hoặc None.

    Hai lớp lỗi đã gặp thực tế:

    1. **Thiếu runtime CUDA**: `llama.dll` → `ggml.dll` → `ggml-cuda.dll` cần
       `cudart64_1x.dll`/`cublas64_1x.dll` (major theo wheel: cu124 → bản 12, cu130 → bản 13).
       Phải gọi `setup_cuda_dll_paths()` TRƯỚC khi import (đúng như `backend/asr/native.py`
       làm) — nếu không, DLL nằm trong `torch/lib` hoặc `site-packages/nvidia/*/bin` cũng
       không được tìm thấy.
    2. **Wheel build bằng AVX-512**: import được nhưng `llama_init_from_model` nổ
       `OSError: [WinError -1073741795] Windows Error 0xc000001d` (STATUS_ILLEGAL_INSTRUCTION)
       trên CPU không có AVX-512 (đo thực: Ryzen 5 5600X; wheel 0.3.35-cu130 chứa 7.177 lệnh
       `zmm`, wheel 0.3.22-cu124 chứa 0). Lớp này KHÔNG thể phát hiện bằng import — xem
       hướng dẫn trong README mục Khắc phục sự cố.
    """
    try:
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("llama_cpp") is None:
            return None
    except Exception:  # noqa: BLE001
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
            "`backend/bin/llama/llama.dll` có tồn tại không (bundle CUDA đi kèm repo), và "
            "runtime CUDA 13 (`cudart64_13.dll`/`cublas64_13.dll`/`cublasLt64_13.dll`) có "
            "trong `torch/lib` hoặc `backend/bin/` không. Nếu lỗi xuất hiện ở bước NẠP MODEL "
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
    try:
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("transcribe_cpp") is not None:
            return None
        if importlib.util.find_spec("transcribe_cpp_native") is None:
            return None  # không dùng ASR native → không phải vấn đề của checker này
        return (
            "thiếu binding `transcribe-cpp` (chỉ có `transcribe-cpp-native`) ⇒ backend không có "
            "backend ASR nào (devices=n/a). Sửa: `pip install \"transcribe-cpp>=0.2.3\"`."
        )
    except Exception:  # noqa: BLE001
        return None


def check_silero_onnxruntime() -> Optional[str]:
    """`silero-vad` ≥ 6.2 import `onnxruntime` ở cấp module ⇒ thiếu là engine Silero chết.

    Không phải lỗi của dự án mà là thiếu sót metadata của gói (không khai báo onnxruntime dù
    chỉ dùng đường JIT). Đã gặp: `Engine 'silero-vad' nạp nền thất bại: ModuleNotFoundError`.
    """
    try:
        import importlib.util  # noqa: PLC0415

        if importlib.util.find_spec("silero_vad") is None:
            return None
        if importlib.util.find_spec("onnxruntime") is not None:
            return None
        return (
            "có `silero-vad` nhưng thiếu `onnxruntime` (silero-vad ≥ 6.2 import ở cấp module) ⇒ "
            "engine Silero không nạp được. Sửa: `pip install onnxruntime`."
        )
    except Exception:  # noqa: BLE001
        return None


def check_environment() -> List[str]:
    """Danh sách vấn đề của môi trường (rỗng = OK)."""
    problems = list(torch_status().get("problems") or [])
    for extra in (check_llama_cpp(), check_asr_binding(), check_silero_onnxruntime()):
        if extra:
            problems.append(extra)
    return problems


def format_report(status: Optional[Dict[str, Any]] = None) -> str:
    """Báo cáo nhiều dòng cho người đọc (dùng cho CLI và log)."""
    status = status or torch_status()
    lines = [
        "== Môi trường Python của dự án ==",
        f"torch        : {status['version']}  (bản CUDA: {status['cuda_build'] or 'CPU-only'})",
        f"torch.cuda   : available={status['cuda_available']} devices={status['cuda_device_count']}",
        f"torchaudio   : {status['torchaudio']}",
        f"torchvision  : {status['torchvision']}",
        f"bộ khoá      : {' · '.join(EXPECTED_CUDA_TRIO)} (backend/constraints.txt)",
    ]
    problems = list(status["problems"])
    llama = check_llama_cpp()
    if llama:
        problems.append(llama)
    asr_binding = check_asr_binding()
    if asr_binding:
        problems.append(asr_binding)
    silero = check_silero_onnxruntime()
    if silero:
        problems.append(silero)
    lines.append(f"llama_cpp    : {'OK' if llama is None else 'LỖI (xem bên dưới)'}")
    lines.append(f"ASR binding  : {'OK' if asr_binding is None else 'THIẾU transcribe-cpp'}")
    lines.append(f"silero/onnx  : {'OK' if silero is None else 'THIẾU onnxruntime'}")
    if problems:
        lines.append("VẤN ĐỀ:")
        lines.extend(f"  - {p}" for p in problems)
    else:
        lines.append("OK: không phát hiện vấn đề.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: in báo cáo; `--json` để lấy máy đọc; exit 1 nếu có vấn đề."""
    argv = list(sys.argv[1:] if argv is None else argv)
    status = torch_status()
    if "--json" in argv:
        status["llama_cpp_error"] = check_llama_cpp()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 1 if (status["problems"] or status["llama_cpp_error"]) else 0
    print(format_report(status))
    return 1 if check_environment() else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
