"""test_71 — Hai định dạng/gói tin dùng chung phải chỉ được dựng ở MỘT chỗ.

BỐI CẢNH (đã hợp nhất 2026-10-08)

1. **Khung audio** `[4B header_len][JSON header][PCM]` từng được dựng ở **4 nơi**
   (`frame-builder.js`, `lookahead-client.js`, và 2 bản fallback không thể chạy trong
   `ws-client.js` / `service-worker.js`). Nay chỉ còn `lib/frame-builder.js`.

2. **Gói `model_status`** từng được viết dạng dict literal ở **18 chỗ** (12 trong `session.py`,
   6 trong `lookahead_handler.py`) dù `serializers.make_model_status_msg()` đã tồn tại — hàm đó
   chỉ được test dùng. Thêm/bớt một khoá là phải sửa 18 nơi.

Bộ test này khoá lại: không được quay về kiểu dựng lại tại chỗ. Xem
`report/audit/26_RA_SOAT_CODE_CHET_VA_CHONG_CHEO.md` §5.8.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.ws.serializers import make_model_status_msg

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
EXT_SRC = ROOT / "extension_src"
#: `tests` bị loại vì chính file này (và các test khác) hợp lệ khi dựng payload MONG ĐỢI để so
#: sánh — bất biến cần chốt là "code SẢN XUẤT không tự dựng lại".
EXCLUDE_DIRS = {"bin", "models", "__pycache__", ".venv", ".git", "external", ".research", "tests"}


def _py_sources() -> list[Path]:
    return sorted(
        p for p in BACKEND.rglob("*.py") if not (EXCLUDE_DIRS & set(p.relative_to(BACKEND).parts))
    )


def test_model_status_chi_dung_o_serializers() -> None:
    """Không được viết tay dict `model_status` ở nơi khác."""
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in _py_sources()
        if p.name != "serializers.py"
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r'"type"\s*:\s*"model_status"', line)
    ]
    assert not offenders, (
        f"còn {len(offenders)} chỗ tự dựng gói `model_status`:\n  " + "\n  ".join(offenders) + "\n"
        "Dùng `backend.ws.serializers.make_model_status_msg(stage, state, model, message)` "
        "để chỉ có MỘT định nghĩa các khoá."
    )


def test_make_model_status_msg_du_khoa() -> None:
    """Hình dạng gói phải ổn định — extension đọc các khoá này trực tiếp."""
    msg = make_model_status_msg("asr", "ready", "qwen3-asr-1.7b")
    assert msg == {
        "type": "model_status",
        "stage": "asr",
        "state": "ready",
        "model": "qwen3-asr-1.7b",
        "message": "",
    }


def test_make_model_status_msg_mang_thong_diep_loi() -> None:
    msg = make_model_status_msg("translation", "error", "index-2b", "hết dung lượng")
    assert msg["message"] == "hết dung lượng"
    assert msg["state"] == "error"


def test_khung_audio_chi_dung_o_frame_builder() -> None:
    """`setUint32(0, …)` = ghi độ dài header ⇒ dấu hiệu tự dựng lại định dạng khung."""
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in sorted(EXT_SRC.rglob("*.js"))
        if p.name != "frame-builder.js"
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "setUint32(0," in line
    ]
    assert not offenders, (
        f"còn {len(offenders)} chỗ tự dựng định dạng khung audio:\n  " + "\n  ".join(offenders) + "\n"
        "Dùng `buildBinaryAudioPacket(header, pcm, encoder)` trong `lib/frame-builder.js`."
    )


def test_frame_builder_tao_duoc_khung_dung_bo_cuc() -> None:
    """Kiểm chứng chính `frame-builder.js` chạy đúng dưới Node (bố cục 4 byte LE + JSON + PCM)."""
    import json
    import subprocess
    import shutil

    node = shutil.which("node")
    if not node:
        pytest.skip("không có node")

    script = (
        'require("'
        + str(EXT_SRC / "lib" / "frame-builder.js").replace("\\", "/")
        + '");'
        'const pcm = new Uint8Array([1,2,3,4]);'
        'const buf = buildBinaryAudioPacket({a:1}, pcm);'
        'const view = new DataView(buf);'
        'const len = view.getUint32(0, true);'
        'const hdr = new TextDecoder().decode(new Uint8Array(buf, 4, len));'
        'const tail = Array.from(new Uint8Array(buf, 4 + len));'
        'console.log(JSON.stringify({JsonLen: len, hdr, tail, total: buf.byteLength}));'
    )
    proc = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60
    )
    assert proc.returncode == 0, f"frame-builder chạy lỗi: {proc.stderr}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["hdr"] == '{"a":1}', f"header JSON sai: {out['hdr']}"
    assert out["tail"] == [1, 2, 3, 4], f"PCM sai: {out['tail']}"
    assert out["total"] == 4 + out["JsonLen"] + 4, "tổng độ dài khung sai"
