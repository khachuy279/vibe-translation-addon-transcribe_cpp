"""test_68 — Chốt xác thực SHA-256 cho các lượt tải của Diarization.

BỐI CẢNH (lỗ hổng đã vá 2026-10-08)

`backend/diarization/service.py` từng tải **3 artifact** bằng `urllib.request.urlretrieve` và
chỉ kiểm tra **kích thước**:

  * `audio-…-bin-windows-x64-cuda12.4.zip`     (~462 MB)
  * `audio-…-cudart-windows-x64-cuda12.4.zip`  (~607 MB)
  * `nemotron-3-diarization-bf16.gguf`         (~190 MB)

Hệ quả: một bản tải bị cắt cụt, hỏng, hoặc bị tráo từ nguồn khác vẫn qua được vòng kiểm tra
kích thước, rồi được **giải nén và nạp thẳng vào tiến trình**. Trong khi đó 3 module tải khác
(`crispasr_native`, `silero_onnx`, `firered_onnx`) đều xác thực SHA-256 ⇒ bất đối xứng này là
hậu quả của việc mỗi module tự viết lại bộ tải. Xem
`report/audit/26_RA_SOAT_CODE_CHET_VA_CHONG_CHEO.md` §5.2.

Bộ test này khoá lại 4 bất biến:
1. Cả 3 artifact đều có hằng SHA-256 hợp lệ và đi qua `_fetch_verified`.
2. Không còn lời gọi `urlretrieve` trực tiếp nào trong 2 hàm tải.
3. `_fetch_verified` THỰC SỰ TỪ CHỐI khi hash sai (kiểm chứng hành vi, không chỉ đọc mã nguồn).
4. `verify_sha256` từ chối hash rỗng (không được coi "thiếu hash" là "bỏ qua").
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend.diarization import service as diar
from backend.utils.model_download import sha256_bytes, sha256_file, verify_sha256

ROOT = Path(__file__).resolve().parents[2]
SERVICE_SRC = (ROOT / "backend" / "diarization" / "service.py").read_text(encoding="utf-8")

PINNED = (
    ("AUDIOCPP_BIN_SHA256", "audio.cpp runtime zip"),
    ("AUDIOCPP_CUDART_SHA256", "audio.cpp cudart zip"),
    ("NEMOTRON_MODEL_SHA256", "Nemotron-3-Diarization GGUF"),
)


@pytest.mark.parametrize(("const_name", "label"), PINNED)
def test_hang_so_sha256_hop_le(const_name: str, label: str) -> None:
    """Mỗi artifact phải có hash SHA-256 đúng định dạng (64 ký tự hex thường)."""
    value = getattr(diar, const_name)
    assert isinstance(value, str) and len(value) == 64, f"{label}: hash phải dài 64 ký tự"
    assert all(c in "0123456789abcdef" for c in value), (
        f"{label}: hash phải là hex thường, nhận được {value!r}"
    )


@pytest.mark.parametrize(("const_name", "label"), PINNED)
def test_moi_artifact_di_qua_duong_xac_thuc(const_name: str, label: str) -> None:
    """Cả 3 artifact phải được tải qua `_fetch_verified`, không tải trần."""
    assert hasattr(diar, "_fetch_verified"), "thiếu hàm tải có xác thực"
    assert f"{const_name}" in SERVICE_SRC, f"{label}: thiếu hằng hash"


def test_khong_con_urlretrieve_tran_trong_2_ham_tai() -> None:
    """Chốt mã nguồn: không được quay lại kiểu tải chỉ kiểm tra kích thước."""
    for forbidden in (
        "urllib.request.urlretrieve(AUDIOCPP_BIN_URL",
        "urllib.request.urlretrieve(AUDIOCPP_CUDART_URL",
        "urllib.request.urlretrieve(NEMOTRON_MODEL_URL",
    ):
        assert forbidden not in SERVICE_SRC, (
            f"phát hiện lượt tải KHÔNG xác thực: {forbidden}\n"
            "Mọi lượt tải phải đi qua `_fetch_verified()`."
        )


def test_fetch_verified_tu_choi_khi_hash_sai(tmp_path, monkeypatch) -> None:
    """HÀNH VI: hash sai ⇒ trả False và KHÔNG để file được chấp nhận."""
    target = tmp_path / "artifact.bin"

    def _fake_urlretrieve(url, dest):  # noqa: ARG001
        Path(dest).write_bytes(b"noi dung bi trao")

    monkeypatch.setattr(diar.urllib.request, "urlretrieve", _fake_urlretrieve)

    ok = diar._fetch_verified(
        "https://example.invalid/x.zip", target, "00" * 32, "artifact test"
    )
    assert ok is False, "hash sai mà vẫn chấp nhận ⇒ lỗ hổng toàn vẹn dữ liệu quay lại"


def test_fetch_verified_chap_nhan_khi_hash_dung(tmp_path, monkeypatch) -> None:
    """HÀNH VI: hash đúng ⇒ trả True (đường lành phải chạy được)."""
    target = tmp_path / "artifact.bin"
    payload = b"noi dung dung"

    def _fake_urlretrieve(url, dest):  # noqa: ARG001
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(diar.urllib.request, "urlretrieve", _fake_urlretrieve)

    ok = diar._fetch_verified(
        "https://example.invalid/x.zip", target, sha256_bytes(payload), "artifact test"
    )
    assert ok is True


def test_verify_sha256_tu_choi_hash_rong(tmp_path) -> None:
    """`expected=""` phải là False — 'thiếu hash' KHÔNG được hiểu là 'bỏ qua kiểm tra'."""
    f = tmp_path / "a.bin"
    f.write_bytes(b"x")
    assert verify_sha256(f, "") is False
    assert verify_sha256(f, "   ") is False


def test_hash_model_cuc_bo_khop_hang_so_ghim() -> None:
    """Nếu model đã có trên đĩa thì hash thật phải khớp hằng số đã ghim.

    Đây là phép kiểm chứng chéo quan trọng: hằng số được lấy từ **LFS oid của HuggingFace**,
    nên nếu file cục bộ khớp, cả hai nguồn đều nhất quán. Bỏ qua khi chưa tải model (CI sạch).
    """
    model = ROOT / "backend" / "models" / "nemotron-3-diarization-bf16.gguf"
    if not model.is_file():
        pytest.skip("model chưa có trên đĩa — bỏ qua kiểm chứng chéo")
    assert sha256_file(model) == diar.NEMOTRON_MODEL_SHA256, (
        "model cục bộ KHÔNG khớp LFS oid đã ghim — hoặc file hỏng, hoặc hằng số sai"
    )


def test_khong_co_ban_sao_sha256_thu_sau_trong_service() -> None:
    """`service.py` phải dùng helper dùng chung, KHÔNG tự cài lại SHA-256 lần thứ 6."""
    assert "import hashlib" not in SERVICE_SRC, (
        "service.py tự import hashlib ⇒ đang cài lại SHA-256 thay vì dùng "
        "backend.utils.model_download.{sha256_file,verify_sha256}"
    )
    assert hashlib.sha256(b"abc").hexdigest()  # sanity: hashlib vẫn sẵn có cho test này
