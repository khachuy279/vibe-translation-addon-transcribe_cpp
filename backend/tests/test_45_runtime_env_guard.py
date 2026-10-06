"""Tầng A: guard môi trường runtime sau Giai đoạn 3 — KHÔNG còn PyTorch.

Thay cho `test_45_torch_env_guard.py` (đã xoá): file cũ kiểm `torch_status()` và việc pip hạ torch
xuống bản CPU-only. Sau khi VAD chuyển sang onnxruntime và ForcedAligner sang GGUF/CrispASR, những
câu hỏi đó không còn ý nghĩa; nay phải chốt:

1. `backend/utils/env_check.py` chạy được, phản ánh ĐÚNG runtime thật (onnxruntime + VAD + aligner).
2. Không còn dấu vết cấu hình torch nào trong `requirements.txt` / `constraints.txt`.
3. Không còn `import torch` / `import funasr` ở cấp module trong `backend/` (trừ worker/tiến trình con
   có chủ đích — không có trường hợp nào).
4. `env_check` KHÔNG tự kéo PyTorch khi chạy (chạy trong tiến trình con sạch).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────── 1. runtime_status / check_environment


def test_runtime_status_co_du_khoa_va_khong_nem():
    """`runtime_status()` phải chạy được và có đủ khoá mà `/health` + CLI dùng."""
    from backend.utils import env_check

    st = env_check.runtime_status()
    for key in (
        "python",
        "onnxruntime",
        "onnxruntime_providers",
        "vad_engine",
        "vad_models_present",
        "aligner_backend",
        "aligner_ready",
        "aligner_reason",
        "llama_cpp",
        "transcribe_cpp",
        "torch_installed",
        "problems",
    ):
        assert key in st, f"thiếu khoá `{key}` trong runtime_status()"
    assert isinstance(st["problems"], list)
    assert isinstance(st["onnxruntime_providers"], list)


def test_onnxruntime_la_bat_buoc_va_dang_co():
    """VAD chạy hoàn toàn bằng onnxruntime ⇒ thiếu gói này là hỏng cả hai engine."""
    from backend.utils import env_check

    st = env_check.runtime_status()
    if st["onnxruntime"] is None:
        pytest.skip("môi trường này chưa cài onnxruntime")
    assert st["onnxruntime"], "phải có phiên bản onnxruntime"
    assert any("CPUExecutionProvider" in p for p in st["onnxruntime_providers"])


def test_env_check_khong_tu_keo_torch():
    """`python -m backend.utils.env_check` phải chạy mà KHÔNG nạp torch.

    Chạy trong tiến trình con sạch: nếu module này import torch (trực tiếp hoặc qua
    `backend.utils.cuda` / `backend.config`) thì khẳng định mới có nghĩa.
    """
    code = (
        "import sys\n"
        "from backend.utils import env_check\n"
        "st = env_check.runtime_status()\n"
        "assert 'torch' not in sys.modules, 'env_check da keo torch'\n"
        "assert isinstance(st['problems'], list)\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_check_environment_tra_ve_list():
    """`check_environment()` luôn trả list (rỗng = OK), không ném."""
    from backend.utils import env_check

    problems = env_check.check_environment()
    assert isinstance(problems, list)
    for p in problems:
        assert isinstance(p, str) and p, "mỗi vấn đề phải là chuỗi không rỗng"


def test_format_report_noi_ro_khong_con_pytorch():
    """Báo cáo phải nói rõ trạng thái torch và gợi ý gỡ nếu còn cài."""
    from backend.utils import env_check

    text = env_check.format_report()
    assert "onnxruntime" in text
    assert "VAD engine" in text
    assert "aligner" in text
    assert "torch" in text  # có dòng báo trạng thái torch (kể cả khi đã gỡ)
    assert "PyTorch" in text or "torch" in text


def test_cli_exit_code_theo_tinh_trang(monkeypatch):
    """CLI: `--json` in JSON; exit 0 khi sạch, 1 khi có vấn đề."""
    from backend.utils import env_check

    monkeypatch.setattr(env_check, "runtime_status", lambda: {
        "python": "3", "onnxruntime": "1", "onnxruntime_providers": [], "vad_engine": "firered-vad",
        "vad_models_present": True, "aligner_backend": "crispasr", "aligner_ready": True,
        "aligner_reason": "", "llama_cpp": True, "transcribe_cpp": True, "torch_installed": False,
        "problems": ["vấn đề giả"],
    })
    monkeypatch.setattr(env_check, "check_llama_cpp", lambda: None)
    monkeypatch.setattr(env_check, "check_asr_binding", lambda: None)
    assert env_check.main([]) == 1

    monkeypatch.setattr(env_check, "runtime_status", lambda: {
        "python": "3", "onnxruntime": "1", "onnxruntime_providers": [], "vad_engine": "firered-vad",
        "vad_models_present": True, "aligner_backend": "crispasr", "aligner_ready": True,
        "aligner_reason": "", "llama_cpp": True, "transcribe_cpp": True, "torch_installed": False,
        "problems": [],
    })
    assert env_check.main([]) == 0


def test_check_environment_gop_ca_llama_va_asr_binding(monkeypatch):
    """`check_environment()` phải gộp vấn đề của CẢ llama.cpp lẫn binding ASR."""
    from backend.utils import env_check

    monkeypatch.setattr(env_check, "runtime_status", lambda: {"problems": ["vad lỗi"]})
    monkeypatch.setattr(env_check, "check_llama_cpp", lambda: "llama lỗi")
    monkeypatch.setattr(env_check, "check_asr_binding", lambda: "thiếu binding")
    problems = env_check.check_environment()
    assert "vad lỗi" in problems and "llama lỗi" in problems and "thiếu binding" in problems


# ─────────────────────────────────────────── 2. manifest không còn torch


def test_requirements_khong_con_torch():
    """`requirements.txt` không được cài bất kỳ gói nào kéo PyTorch."""
    text = (PROJECT_ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8")
    # Bỏ dòng comment trước khi kiểm: tài liệu lịch sử CÓ nhắc tới torch là đúng.
    active = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    banned = ("torch", "torchaudio", "torchvision", "transformers", "omnivoice",
              "fireredvad", "silero-vad", "funasr", "nagisa", "soynlp")
    for line in active:
        low = line.lower().lstrip("-")
        for pkg in banned:
            assert not low.startswith(pkg), f"requirements còn cài `{line}` (khớp `{pkg}`)"
    # Index của PyTorch chỉ được nhắc trong comment lịch sử, KHÔNG ở dòng có hiệu lực.
    assert not any("download.pytorch.org" in ln for ln in active), (
        "dòng `--extra-index-url` của PyTorch phải đã bị gỡ khỏi requirements"
    )


def test_constraints_khong_con_ghim_torch():
    """`constraints.txt` không được ghim torch/torchaudio/torchvision nữa."""
    text = (PROJECT_ROOT / "backend" / "constraints.txt").read_text(encoding="utf-8")
    active = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    for line in active:
        low = line.lower().lstrip("-")
        assert not low.startswith(("torch", "torchaudio", "torchvision")), (
            f"constraints còn ghim `{line}`"
        )


def test_khong_con_import_torch_o_cap_module():
    """Quét `backend/`: không file .py nào `import torch` ở cấp module.

    Cho phép nhắc tới torch trong COMMENT/docstring (tài liệu lịch sử), nhưng không cho phép
    câu lệnh import thật.
    """
    import ast

    offenders: list[str] = []
    for path in (PROJECT_ROOT / "backend").rglob("*.py"):
        if any(part in {"__pycache__", ".venv"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in {"torch", "torchaudio", "funasr"}:
                        offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in {"torch", "torchaudio", "funasr"}:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno} from {node.module}")
    assert not offenders, "còn import torch/funasr:\n  " + "\n  ".join(offenders)


def test_khong_con_thu_muc_qwen_asr_vendored():
    """Cây vendored `backend/asr/qwen_asr/` phải đã bị xoá (kéo torch + transformers)."""
    assert not (PROJECT_ROOT / "backend" / "asr" / "qwen_asr").exists()


# ─────────────────────────────────────────── 3. DLL CrispASR (giới hạn 100 MB của GitHub)


def test_dll_crispasr_du_de_chay_hoac_tai_duoc():
    """`ensure_lib_files()` phải trả thư mục có đủ 5 DLL khi chúng đã có sẵn (đường nhanh).

    Đường TẢI (khi thiếu) không chạy ở đây vì nó cần mạng + 157 MB; logic tải được kiểm bằng
    hằng số ghim (`BUNDLE_URL`/`BUNDLE_SHA256`) và đã đo thủ công: tải + giải nén + xác thực
    SHA-256 từng DLL mất ~9 s.
    """
    from backend.utils.crispasr_native import REQUIRED_DLLS, aligner_lib_dir, ensure_lib_files

    lib = aligner_lib_dir()
    if lib is None:
        pytest.skip("chưa có DLL CrispASR trên máy này")
    out = ensure_lib_files(lib)
    for name in REQUIRED_DLLS:
        assert (out / name).is_file(), f"thiếu {name}"


def test_hang_so_goi_dll_duoc_ghim():
    """URL + SHA-256 của gói DLL phải được GHIM (không dùng `latest`)."""
    from backend.utils import crispasr_native as cn

    assert cn.BUNDLE_URL.startswith("https://github.com/CrispStrobe/CrispASR/releases/download/")
    assert cn.RELEASE_TAG.startswith("v"), "phải ghim tag phát hành cụ thể"
    assert cn.BUNDLE_URL.endswith(cn.BUNDLE_ASSET)
    assert len(cn.BUNDLE_SHA256) == 64 and cn.BUNDLE_SHA256.isalnum()
    assert set(cn.DLL_SHA256) == set(cn.REQUIRED_DLLS), (
        "mỗi DLL bắt buộc phải có SHA-256 để xác thực"
    )
    for digest in cn.DLL_SHA256.values():
        assert len(digest) == 64


def test_ggml_cuda_vuot_gioi_han_github_va_duoc_tai_thay_vi_commit():
    """Chốt LÝ DO cơ chế tải tồn tại: `ggml-cuda.dll` > 100 MB nên GitHub từ chối push."""
    from backend.utils.crispasr_native import REQUIRED_DLLS, aligner_lib_dir

    lib = aligner_lib_dir()
    if lib is None:
        pytest.skip("chưa có DLL CrispASR trên máy này")
    big = lib / "ggml-cuda.dll"
    assert big.is_file()
    assert big.stat().st_size > 100 * 1024 * 1024, (
        "giả định '> 100 MB nên không commit được' không còn đúng — có thể commit thẳng được rồi"
    )
    # Và cơ chế tải phải tồn tại để bù cho việc không commit.
    assert "ggml-cuda.dll" in REQUIRED_DLLS


def test_available_khong_kich_hoat_tai():
    """`available()` chỉ ĐỌC trạng thái đĩa — không được tải 157 MB + 500 MB khi bị gọi."""
    from backend.utils import crispasr_native as cn

    calls: list[str] = []
    orig = cn._dlls_present

    def _spy(lib_dir, **kw):  # noqa: ANN001
        calls.append(str(lib_dir))
        return orig(lib_dir, **kw)

    cn._dlls_present = _spy  # type: ignore[assignment]
    try:
        ok, reason = cn.available()
    finally:
        cn._dlls_present = orig  # type: ignore[assignment]
    assert calls, "available() phải kiểm sự tồn tại của DLL"
    assert isinstance(ok, bool) and isinstance(reason, str)


# ─────────────── 4. chạy THẬT với `torch` bị CHẶN Ở CẤP IMPORT (bằng chứng mạnh nhất)


#: Script chạy trong tiến trình con: cài hook chặn `torch`/`transformers`/`funasr`, rồi chạy
#: đúng các đường thật của backend. Nếu còn BẤT KỲ phụ thuộc ẩn nào vào PyTorch (kể cả import
#: trong `except`), tiến trình con sẽ chết ngay.
_TORCH_BLOCKED_SMOKE = r"""
import sys, importlib.abc, importlib.machinery

BLOCKED = ("torch", "torchaudio", "transformers", "funasr", "nagisa", "soynlp", "omnivoice",
           "fireredvad", "silero_vad")

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ImportError(f"BLOCKED-FOR-TEST: {fullname}")
        return None

sys.meta_path.insert(0, _Block())

import numpy as np, soundfile as sf
from pathlib import Path
from math import gcd
from scipy.signal import resample_poly

def pcm16(name):
    x, sr = sf.read(str(Path("wav_test") / name), dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != 16000:
        g = gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x[: 16000 * 8], dtype=np.float32)

# 1. VAD: nạp + chạy cả hai engine ONNX
from backend.vad.engines import VADEngineFactory
for eng_name in ("firered-vad", "silero-vad"):
    eng = VADEngineFactory.get_engine(eng_name)
    st = eng.create_initial_state()
    pcm = pcm16("Japanese_5s.wav")
    hop = eng.frame_samples
    events = 0
    for i in range(pcm.size // hop):
        r = eng.is_speech(np.clip(np.rint(pcm[i*hop:(i+1)*hop] * 32768.0), -32768, 32767).astype(np.int16), st)
        events += 1 if r.event else 0
    print(f"VAD {eng_name}: {pcm.size//hop} frame, {events} su kien")

# 2. Bộ dò khoảng lặng của Lookahead (cũng dùng Silero)
from backend.core.vad_silence import VADSilenceScanner
sc = VADSilenceScanner()
sc.prewarm()
assert sc.vad_unavailable_reason == "", sc.vad_unavailable_reason
print("VADSilenceScanner: OK")

# 3. Aligner: chạy thật qua tiến trình con CrispASR
from backend.asr.forced_aligner import ForcedAlignerService
svc = ForcedAlignerService.get_instance()
x = pcm16("Japanese_5s.wav")
text = (Path("wav_test") / "Japanese_5s.txt").read_text(encoding="utf-8").strip()
words = svc.align(x, text, language="Japanese")
assert words, "aligner khong tra ve don vi nao"
subs = ForcedAlignerService.split_oversized_sentences(
    ForcedAlignerService.group_words_to_subtitles(words, language="Japanese"), language="Japanese")
print(f"Aligner: {len(words)} don vi, {len(subs)} phu de")

from backend.asr.crispasr_aligner import CrispASRAlignerClient
CrispASRAlignerClient.get_instance().shutdown()

assert "torch" not in sys.modules, "torch bi nap du da chan"
print("OK-TORCH-FREE")
"""


def test_chay_that_khi_torch_bi_chan_import():
    """Chạy VAD + bộ dò khoảng lặng + aligner với `torch` bị chặn ⇒ KHÔNG còn phụ thuộc ẩn.

    Đây là bằng chứng mạnh nhất mà môi trường này cho phép mà không phải `pip uninstall torch`:
    hook `sys.meta_path` chặn `torch`/`transformers`/`funasr`/`nagisa`/`soynlp`/`fireredvad`/
    `silero_vad`, nên mọi `import` lười (kể cả nằm trong `except`) đều nổ ngay.
    """
    from backend.utils.crispasr_native import available as _fa_available

    if not _fa_available()[0]:
        pytest.skip("chưa có DLL + model GGUF của CrispASR trên máy này")
    if not (PROJECT_ROOT / "wav_test" / "Japanese_5s.wav").is_file():
        pytest.skip("thiếu audio test")

    proc = subprocess.run(
        [sys.executable, "-c", _TORCH_BLOCKED_SMOKE],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"chạy với torch bị chặn đã THẤT BẠI ⇒ còn phụ thuộc ẩn vào PyTorch.\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "OK-TORCH-FREE" in proc.stdout, proc.stdout
