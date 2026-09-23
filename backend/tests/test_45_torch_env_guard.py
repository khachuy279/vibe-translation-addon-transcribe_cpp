"""Test tầng A — chốt lớp bảo vệ chống pip âm thầm hạ torch CUDA.

BỐI CẢNH: 2026-09-22, `pip install -U silero-vad` (6.2.2) làm pip phân giải lại và cài
**torch CPU-only** (`2.12.0+cu130` → `2.9.1+cpu`) vì interpreter dùng chung với các gói có
ràng buộc torch xung đột. Test này khoá lại 3 lớp phòng ngừa:

1. `backend/utils/env_check.py` phát hiện đúng: torch CPU-only, lệch torchaudio, torchvision
   CUDA đi kèm torch CPU, và **không** báo động giả khi bộ ba khớp nhau.
2. CLI `python -m backend.utils.env_check` trả exit code 1 khi có vấn đề, 0 khi sạch.
3. `/health` có khối `torch`, và `lifespan` gọi cảnh báo môi trường lúc khởi động.
4. `backend/constraints.txt` tồn tại và ghim đúng bộ CUDA (để pip báo lỗi thay vì hạ torch).
"""

from pathlib import Path

import pytest

from backend.utils import env_check

ROOT = Path(__file__).resolve().parent.parent.parent


class _FakeTorch:
    def __init__(self, version: str, cuda_build):
        self.__version__ = version
        self.version = type("V", (), {"cuda": cuda_build})()
        self.cuda = type(
            "C",
            (),
            {
                "is_available": staticmethod(lambda: bool(cuda_build)),
                "device_count": staticmethod(lambda: 1 if cuda_build else 0),
            },
        )()


@pytest.fixture
def fake_modules(monkeypatch):
    """Giả lập `torch`/`torchaudio`/`torchvision` để không phụ thuộc môi trường thật."""

    def _install(torch_obj, modules: dict):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                return torch_obj
            if name in modules:
                mod = modules[name]
                if isinstance(mod, Exception):
                    raise mod
                return mod
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        # `find_spec` dùng để phát hiện "đã cài nhưng import hỏng".
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda n: object() if n in modules else None,
        )

    return _install


# ─────────────────────────────────────────────── 1. phát hiện vấn đề


def test_phat_hien_torch_cpu_only(fake_modules):
    """Bản CPU-only phải bị bắt (đúng sự cố đã xảy ra)."""
    fake_modules(_FakeTorch("2.9.1+cpu", None), {"torchaudio": type("M", (), {"__version__": "2.9.1+cpu"})()})

    status = env_check.torch_status()
    assert status["installed"] is True
    assert status["cuda_build"] is None
    assert status["cuda_available"] is False
    assert any("CPU-only" in p for p in status["problems"]), status["problems"]
    assert env_check.check_environment(), "phải trả về danh sách vấn đề khác rỗng"


def test_phat_hien_torchaudio_cpu_di_kem_torch_cuda(fake_modules):
    """torch CUDA nhưng torchaudio là bản CPU ⇒ phải cảnh báo lệch bộ."""
    fake_modules(
        _FakeTorch("2.12.0+cu130", "13.0"),
        {"torchaudio": type("M", (), {"__version__": "2.11.0+cpu"})()},
    )
    problems = env_check.check_environment()
    assert any("torchaudio" in p and "bản CPU" in p for p in problems), problems


def test_phat_hien_lech_tag_cuda(fake_modules):
    """Cùng là bản CUDA nhưng khác tag (+cu130 vs +cu124) ⇒ phải cảnh báo."""
    fake_modules(
        _FakeTorch("2.12.0+cu130", "13.0"),
        {"torchaudio": type("M", (), {"__version__": "2.12.0+cu124"})()},
    )
    problems = env_check.check_environment()
    assert any("torchaudio" in p and "+cu124" in p and "+cu130" in p for p in problems), problems


def test_khac_so_phien_ban_nhung_cung_tag_cuda_thi_khong_bao_loi(fake_modules):
    """Bộ ĐÚNG của dự án: torch 2.12.0 + torchaudio 2.11.0 + torchvision 0.27.0 (đều +cu130).

    Đây là bài test chống hồi quy cho chính lỗi đã gặp: checker bản đầu so *số phiên bản*
    torchaudio với torch nên báo sai bộ hợp lệ (index cu130 không có torchaudio 2.12.x).
    """
    fake_modules(
        _FakeTorch("2.12.0+cu130", "13.0"),
        {
            "torchaudio": type("M", (), {"__version__": "2.11.0+cu130"})(),
            "torchvision": type("M", (), {"__version__": "0.27.0+cu130"})(),
        },
    )
    status = env_check.torch_status()
    assert status["problems"] == [], status["problems"]
    assert env_check.cuda_tag(status["version"]) == "+cu130"
    assert env_check.cuda_tag(status["torchaudio"]) == "+cu130"


def test_phat_hien_torchvision_cuda_di_kem_torch_cpu(fake_modules):
    """torchvision bản CUDA + torch CPU ⇒ lệch bộ (đúng trạng thái sau sự cố)."""
    fake_modules(
        _FakeTorch("2.9.1+cpu", None),
        {
            "torchaudio": type("M", (), {"__version__": "2.9.1+cpu"})(),
            "torchvision": type("M", (), {"__version__": "0.27.0+cu130"})(),
        },
    )
    problems = env_check.check_environment()
    assert any("torchvision" in p for p in problems), problems


def test_bao_import_hong_khi_da_cai(fake_modules):
    """Module đã cài nhưng import lỗi phải được nêu tên, không im lặng bỏ qua."""
    fake_modules(
        _FakeTorch("2.9.1+cpu", None),
        {"torchvision": RuntimeError("undefined symbol")},
    )
    problems = env_check.check_environment()
    assert any("torchvision" in p and "import lỗi" in p for p in problems), problems


def test_khong_bao_dong_gia_khi_bo_ba_khop(fake_modules):
    """Bộ ba CUDA khớp nhau ⇒ KHÔNG được báo vấn đề (tránh cảnh báo rác)."""
    fake_modules(
        _FakeTorch("2.12.0+cu130", "13.0"),
        {
            "torchaudio": type("M", (), {"__version__": "2.12.0+cu130"})(),
            "torchvision": type("M", (), {"__version__": "0.27.0+cu130"})(),
        },
    )
    status = env_check.torch_status()
    assert status["cuda_available"] is True and status["cuda_device_count"] == 1
    assert status["problems"] == [], status["problems"]


# ─────────────────────────────────────────────── 2. CLI


def test_cli_exit_code_theo_tinh_trang(monkeypatch, capsys):
    monkeypatch.setattr(env_check, "torch_status", lambda: {"problems": ["x"], "version": "1",
                                                            "cuda_build": None, "cuda_available": False,
                                                            "cuda_device_count": 0, "torchaudio": None,
                                                            "torchvision": None, "installed": True})
    assert env_check.main([]) == 1
    assert "VẤN ĐỀ" in capsys.readouterr().out

    monkeypatch.setattr(env_check, "torch_status", lambda: {"problems": [], "version": "1",
                                                            "cuda_build": "13.0", "cuda_available": True,
                                                            "cuda_device_count": 1, "torchaudio": "1",
                                                            "torchvision": "1", "installed": True})
    assert env_check.main([]) == 0
    assert "OK" in capsys.readouterr().out


# ─────────────────────────────────────────────── 3. nối dây vào backend


def test_health_co_khoi_torch():
    """`/health` phải trả khối `torch` để phát hiện sớm."""
    import asyncio

    from backend.main import health_check

    payload = asyncio.run(health_check())
    assert "torch" in payload
    assert "cuda_build" in payload["torch"] and "problems" in payload["torch"]


def test_lifespan_goi_canh_bao_moi_truong():
    """Khởi động phải gọi cảnh báo môi trường torch (biến cố im lặng thành log rõ)."""
    import inspect

    import backend.main as main_mod

    assert hasattr(main_mod, "_log_torch_status_at_startup")
    src = inspect.getsource(main_mod.lifespan)
    assert "_log_torch_status_at_startup" in src, "lifespan thiếu bước kiểm tra môi trường torch"


# ─────────────────────────────────────────────── 5. llama.cpp (lệch major CUDA)


def test_check_llama_cpp_bao_loi_kem_goi_y(monkeypatch):
    """Import `llama_cpp` nổ ⇒ thông báo phải chỉ đúng kho DLL và runtime CUDA cần có.

    Bối cảnh mới: binding lấy từ wheel CPU ~7 MB, còn `llama.dll` + `ggml*.dll` do repo cung cấp
    trong `backend/bin/llama/` (trỏ qua `LLAMA_CPP_LIB_PATH`). Nên gợi ý phải nói về hai thứ đó.
    """
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise RuntimeError("Failed to load shared library 'llama.dll'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr("importlib.util.find_spec", lambda n: object())

    msg = env_check.check_llama_cpp()
    assert msg is not None
    assert "llama.cpp" in msg and "llama.dll" in msg
    assert "backend/bin/llama" in msg
    assert "0xc000001d" in msg and "AVX-512" in msg


def test_check_llama_cpp_chua_cai_thi_bo_qua(monkeypatch):
    """Chưa cài llama.cpp ⇒ không phải vấn đề (dự án chạy được không có dịch GGUF)."""
    monkeypatch.setattr("importlib.util.find_spec", lambda n: None)
    assert env_check.check_llama_cpp() is None


def test_check_environment_gop_ca_llama(monkeypatch):
    """`check_environment()` phải gộp vấn đề torch VÀ llama.cpp."""
    monkeypatch.setattr(env_check, "torch_status", lambda: {"problems": ["torch CPU-only"]})
    monkeypatch.setattr(env_check, "check_llama_cpp", lambda: "llama lỗi")
    problems = env_check.check_environment()
    assert problems == ["torch CPU-only", "llama lỗi"]


# ─────────────────────────────────────────────── 6. allocator conf của torch


def test_alloc_conf_dung_ten_theo_phien_ban_torch(monkeypatch):
    """torch ≥ 2.9 dùng `PYTORCH_ALLOC_CONF` (tên cũ bị deprecation-warning — ĐO THỰC ở 2.9.1)."""
    import os

    from backend.utils import cuda as cuda_utils

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    for version in ("2.9.1+cu130", "2.12.0+cu130"):
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.setattr("importlib.metadata.version", lambda name, v=version: v)
        cuda_utils._configure_alloc_conf()
        assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True", version
        assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ, version

    # Torch CŨ (< 2.9) không biết tên mới ⇒ phải dùng tên cũ, nếu không sẽ MẤT tuning.
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2.4.1")
    cuda_utils._configure_alloc_conf()
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"


# ─────────────────────────────────────────────── 7. gói thiếu làm chết ASR / Silero VAD


def test_phat_hien_thieu_binding_transcribe_cpp(monkeypatch):
    """Chỉ có `transcribe-cpp-native` mà thiếu binding `transcribe-cpp` ⇒ ASR chết (devices=n/a)."""
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda n: None if n == "transcribe_cpp" else object(),
    )
    msg = env_check.check_asr_binding()
    assert msg is not None
    assert "transcribe-cpp" in msg and "devices=n/a" in msg

    monkeypatch.setattr("importlib.util.find_spec", lambda n: object())
    assert env_check.check_asr_binding() is None


def test_bo_qua_khi_khong_dung_asr_native(monkeypatch):
    """Không cài provider native nào ⇒ checker này không có ý kiến (dự án khác dùng chung env)."""
    monkeypatch.setattr("importlib.util.find_spec", lambda n: None)
    assert env_check.check_asr_binding() is None


def test_phat_hien_thieu_onnxruntime_cho_silero(monkeypatch):
    """silero-vad ≥ 6.2 import onnxruntime ở cấp module ⇒ thiếu là engine Silero không nạp."""
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda n: None if n == "onnxruntime" else object(),
    )
    msg = env_check.check_silero_onnxruntime()
    assert msg is not None and "onnxruntime" in msg

    monkeypatch.setattr("importlib.util.find_spec", lambda n: object())
    assert env_check.check_silero_onnxruntime() is None


# ─────────────────────────────────────────────── 8. constraints


def test_constraints_ghim_dung_bo_cuda():
    """`backend/constraints.txt` phải tồn tại và ghim bộ ba CUDA (để pip báo lỗi, không hạ torch)."""
    path = ROOT / "backend" / "constraints.txt"
    assert path.is_file(), "thiếu backend/constraints.txt — pip sẽ lại âm thầm hạ torch"
    text = path.read_text(encoding="utf-8")
    for expected in env_check.EXPECTED_CUDA_TRIO:
        assert expected in text, f"constraints thiếu {expected}"
    for name in ("torch", "torchaudio", "torchvision"):
        assert f"{name}==" in text, f"constraints thiếu ghim cho {name}"
