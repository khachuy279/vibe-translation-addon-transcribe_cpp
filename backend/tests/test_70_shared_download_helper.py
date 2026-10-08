"""test_70 — Mọi lượt tải qua `urllib` phải đi qua helper dùng chung.

BỐI CẢNH (đã hợp nhất 2026-10-08)

Trước đây có **3 bộ tải `urllib` tự viết**, mỗi bộ một kiểu:

  * `utils/crispasr_native.py`  — tải zip 157 MB, xác thực SHA-256 gói + từng DLL
  * `vad/silero_onnx.py`        — thử NHIỀU URL dự phòng, kiểm thêm kích thước tối thiểu
  * `vad/engines/firered_onnx.py` — một URL, xác thực SHA-256

Chúng lệch nhau về timeout (300/60/120 s), về thông báo lỗi, và về cách ghi file. Đây đúng là
loại chồng chéo khiến `diarization/service.py` bị bỏ quên phần xác thực (xem
`report/audit/26_RA_SOAT_CODE_CHET_VA_CHONG_CHEO.md` §5.2).

Nay cả ba đi qua `backend.utils.model_download.fetch_verified_bytes()`, và `urlopen` chỉ còn
xuất hiện ở ĐÚNG MỘT chỗ trong toàn bộ backend.

Bộ test này khoá lại 4 bất biến:
1. `urlopen` chỉ có trong `model_download.py`.
2. Hash sai ⇒ TỪ CHỐI (trả `None`), không ghi file.
3. `min_bytes` chặn được file cụt (hành vi riêng của đường Silero).
4. `write_bytes_atomic` không để lại file `.part`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.utils.model_download import (
    fetch_verified_bytes,
    sha256_bytes,
    write_bytes_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_DIRS = {"bin", "models", "__pycache__", ".venv", ".git", "external", ".research", "tests"}


def _backend_sources() -> list[Path]:
    return sorted(
        p
        for p in (ROOT / "backend").rglob("*.py")
        if not (EXCLUDE_DIRS & set(p.relative_to(ROOT / "backend").parts))
    )


def test_urlopen_chi_con_mot_cho() -> None:
    """Chốt mã nguồn: không được quay lại kiểu mỗi module tự `urlopen`."""
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in _backend_sources()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"\burlopen\s*\(", line)
    ]
    assert len(offenders) == 1, (
        "`urlopen` phải chỉ xuất hiện MỘT lần (trong backend/utils/model_download.py), "
        f"nhưng thấy {len(offenders)} chỗ:\n  " + "\n  ".join(offenders) + "\n"
        "Mọi lượt tải phải đi qua `fetch_verified_bytes()` để thống nhất timeout, "
        "xác thực SHA-256 và cách ghi file."
    )
    assert "model_download.py" in offenders[0], (
        f"`urlopen` phải nằm trong model_download.py, nhưng thấy ở {offenders[0]}"
    )


def test_tu_choi_khi_hash_sai(monkeypatch) -> None:
    """HÀNH VI: nội dung tải về sai hash ⇒ `None`, KHÔNG trả bytes."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=None: _FakeResponse(b"noi dung bi trao"),
    )
    out = fetch_verified_bytes(
        ["https://example.invalid/a.bin"], "00" * 32, label="artifact test"
    )
    assert out is None, "hash sai mà vẫn trả bytes ⇒ lỗ hổng toàn vẹn dữ liệu quay lại"


def test_chap_nhan_khi_hash_dung(monkeypatch) -> None:
    """HÀNH VI: đường lành phải chạy."""
    payload = b"noi dung dung"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=None: _FakeResponse(payload),
    )
    out = fetch_verified_bytes(
        ["https://example.invalid/a.bin"], sha256_bytes(payload), label="artifact test"
    )
    assert out == payload


def test_bo_qua_url_hong_va_dung_url_du_phong(monkeypatch) -> None:
    """Nhiều URL ứng viên: URL đầu lỗi mạng thì thử URL sau (hành vi của đường Silero)."""
    payload = b"ban du phong"
    calls: list[str] = []

    def _fake(url, timeout=None):  # noqa: ARG001
        calls.append(url)
        if "hong" in url:
            raise OSError("mạng hỏng")
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    out = fetch_verified_bytes(
        ["https://example.invalid/hong.bin", "https://example.invalid/tot.bin"],
        sha256_bytes(payload),
        label="artifact test",
    )
    assert out == payload
    assert len(calls) == 2, "phải thử URL dự phòng sau khi URL đầu hỏng"


def test_min_bytes_chan_file_cut(monkeypatch) -> None:
    """`min_bytes` chặn file cụt dù hash có khớp (bảo vệ riêng của đường Silero ONNX)."""
    payload = b"x" * 10
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=None: _FakeResponse(payload),
    )
    out = fetch_verified_bytes(
        ["https://example.invalid/a.bin"],
        sha256_bytes(payload),
        label="artifact test",
        min_bytes=1_000,
    )
    assert out is None, "file nhỏ hơn `min_bytes` phải bị từ chối"


def test_write_bytes_atomic_khong_de_lai_file_tam(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "model.onnx"
    write_bytes_atomic(target, b"noi dung")
    assert target.read_bytes() == b"noi dung"
    leftovers = [p.name for p in target.parent.iterdir() if p.name != target.name]
    assert not leftovers, f"còn file tạm sau khi ghi: {leftovers}"


class _FakeResponse:
    """`urlopen` giả tối thiểu: chỉ cần context manager + `read()`."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False
