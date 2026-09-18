"""A1-1 (mở rộng) + 2 lỗi harness WER — chống hồi quy cho đúng thứ vừa cắn.

**A1-1:** guard chống busy-spin thread CPU trước đây chỉ nằm trong `backend/main.py`, nên
MỌI đường khác (harness WER, benchmark, pytest) chạy bằng mặc định OpenMP/OpenBLAS = số
nhân logic + chế độ spin. Đo trên máy 12 luồng: harness WER (wheel/Vulkan) giữ **~11 core
ở 85% mỗi thread LIÊN TỤC**, kể cả khi GPU ở 0%. Nay guard nằm trong `backend/__init__.py`
⇒ mọi `import backend.*` đều có.

**Harness `test_09_wer_ab.py`:** bản cũ `import transcribe_cpp` TRỰC TIẾP trước khi
`backend.asr` kịp bootstrap ⇒ nạp native của WHEEL (Vulkan) và bỏ qua bundle `bin/` (CUDA).
Hệ quả kép: (a) mọi số `infer_ms` không đại diện bản phát hành; (b) chính đường Vulkan của
wheel đốt ~11 core CPU. Đo sau khi sửa: **~0,5 core**.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_INIT = ROOT / "backend" / "__init__.py"
MAIN_SRC = ROOT / "backend" / "main.py"
HARNESS = ROOT / "backend" / "tests" / "test_09_wer_ab.py"

# Biến PHẢI được đặt; thiếu `OPENBLAS_NUM_THREADS` là lỗ hổng của bản cũ (OpenBLAS đọc
# biến này TRƯỚC `OMP_NUM_THREADS`, mà numpy ở đây build trên `scipy-openblas`).
REQUIRED_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_WAIT_POLICY",
    "KMP_BLOCKTIME",
)


# ─────────────────────────────────────────────── A1-1

def test_backend_init_dat_guard_thread():
    """`import backend` (chạy trước mọi submodule) phải đặt đủ biến giới hạn thread."""
    src = BACKEND_INIT.read_text(encoding="utf-8")
    for key in REQUIRED_ENV:
        assert key in src, f"`backend/__init__.py` thiếu biến {key} (A1-1)"
    assert "apply_thread_limits()" in src, "guard phải được GỌI, không chỉ định nghĩa"


def test_import_backend_thuc_su_dat_bien_moi_truong():
    """Kiểm chứng THẬT trong tiến trình con sạch (không kế thừa env của pytest)."""
    code = (
        "import backend, os, json;"
        "print(json.dumps({k: os.environ.get(k) for k in %r}))" % (list(REQUIRED_ENV),)
    )
    env = {k: v for k, v in os.environ.items() if k not in REQUIRED_ENV}
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT)
    )
    assert proc.returncode == 0, proc.stderr[-600:]
    import json

    got = json.loads(proc.stdout.strip().splitlines()[-1])
    for key, value in got.items():
        assert value is not None, f"`import backend` KHÔNG đặt {key} ⇒ còn busy-spin (A1-1)"
    assert got["OMP_WAIT_POLICY"] == "PASSIVE", "phải tắt busy-spin bằng OMP_WAIT_POLICY=PASSIVE"
    assert got["OPENBLAS_NUM_THREADS"] == "2", "numpy/OpenBLAS phải bị giới hạn"


def test_guard_khong_ghi_de_bien_nguoi_dung_da_dat():
    """Ai muốn ép số luồng khác vẫn phải làm được (setdefault, không gán thẳng)."""
    code = (
        "import backend, os;"
        "print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['OMP_NUM_THREADS'])"
    )
    env = dict(os.environ)
    env.update({"OPENBLAS_NUM_THREADS": "7", "OMP_NUM_THREADS": "5", "PYTHONPATH": str(ROOT)})
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT)
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout.split() == ["7", "5"], (
        "guard ghi đè biến người dùng đã đặt — phải dùng `setdefault`"
    )


def test_main_khong_con_dat_bien_thread_trung_lap():
    """`main.py` không được giữ bản sao thứ hai của danh sách biến (dễ lệch nhau)."""
    src = MAIN_SRC.read_text(encoding="utf-8")
    for key in REQUIRED_ENV:
        assert f'os.environ["{key}"]' not in src, (
            f"`main.py` còn gán thẳng `os.environ[\"{key}\"]` — guard đã chuyển sang "
            f"`backend/__init__.py`"
        )
    assert "apply_thread_limits" in src, "main.py nên gọi lại guard cho tường minh"


# ─────────────────────────────────────────────── harness WER

def _first_lineno(tree: ast.AST, predicate) -> "int | None":
    best = None
    for node in ast.walk(tree):
        if predicate(node):
            ln = getattr(node, "lineno", None)
            if ln is not None and (best is None or ln < best):
                best = ln
    return best


def _is_transcribe_import(node: ast.AST) -> bool:
    return isinstance(node, ast.Import) and any(a.name == "transcribe_cpp" for a in node.names)


def _is_backend_asr_ref(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("backend.asr"):
        return True
    if isinstance(node, ast.Import) and any(a.name.startswith("backend.asr") for a in node.names):
        return True
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "bootstrap":
        return True
    return False


def _boot_before_plain_import(path: Path):
    """`(dòng bootstrap, dòng import transcribe_cpp)` thật (AST) — bỏ qua comment/docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return _first_lineno(tree, _is_backend_asr_ref), _first_lineno(tree, _is_transcribe_import)


def test_harness_bootstrap_bin_truoc_khi_cham_transcribe_cpp():
    """`test_09_wer_ab.py` phải để `backend.asr` bootstrap TRƯỚC `import transcribe_cpp`.

    Bản cũ import thẳng `transcribe_cpp` ⇒ native của WHEEL (Vulkan) vào `sys.modules`,
    bundle `bin/` (CUDA) không áp được. Hệ quả ĐO ĐƯỢC: harness chạy Vulkan của wheel và
    đốt **~11 core CPU** liên tục (so với ~0,5 core khi chạy `bin/`+CUDA).
    """
    boot, plain = _boot_before_plain_import(HARNESS)
    assert boot is not None, "harness không bootstrap `backend.asr`"
    if plain is not None:
        assert boot < plain, (
            f"`import transcribe_cpp` (dòng {plain}) chạy TRƯỚC bootstrap (dòng {boot}) "
            f"⇒ bundle bin/ bị bỏ qua"
        )


def test_moi_harness_ngoai_package_asr_phai_bootstrap_truoc():
    """Cùng cái bẫy đó, quét MỌI file trong `backend/` (miễn package `backend/asr`).

    File trong package `backend/asr` được miễn vì `backend/asr/__init__.py` đã bootstrap
    trước khi nạp các submodule (`adapters.py`, `engine.py`).
    """
    offenders = []
    for path in (ROOT / "backend").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("backend/asr/"):
            continue
        boot, plain = _boot_before_plain_import(path)
        if plain is None:
            continue
        if boot is None or boot > plain:
            offenders.append(f"{rel} (import dòng {plain}, bootstrap dòng {boot})")
    assert not offenders, (
        "các file này nạp native của WHEEL trước khi bootstrap bin/: " + "; ".join(offenders)
    )
