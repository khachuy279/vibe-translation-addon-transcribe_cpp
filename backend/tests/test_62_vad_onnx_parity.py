"""Tầng B: VAD ONNX — chống HỒI QUY bằng **kỳ vọng vàng**, không cần PyTorch.

LỊCH SỬ (giữ lại vì sao bộ số này đáng tin)
===========================================
Khi chuyển VAD từ TorchScript sang ONNX, hai engine đã được chứng minh **bit-tương-đương** với
đường torch cũ bằng cách so TỪNG FRAME:

| Engine | Số frame | `max|Δp|` | Frame lệch quyết định @0.30/0.40/0.50 |
|---|---|---|---|
| Silero | 3 797 | ≤ 2e-6 | **0 / 0 / 0** |
| FireRed | 12 621 | ≤ 1e-6 | **0 / 0 / 0** |

Giai đoạn 3 đã XOÁ đường torch (cùng `silero-vad`, `fireredvad`, `torch`) nên không thể so trực
tiếp nữa. Thay vì bỏ mất bảo đảm, kết quả đã kiểm được **GHIM** vào
`tests/fixtures/vad_onnx_golden.json` (`_doc` trong file ghi rõ nguồn gốc). Test này so engine
hiện tại với bộ số đó ⇒ bắt hồi quy mà KHÔNG cần PyTorch.

Ghim những gì:
  * `frames`            — số frame engine tiêu thụ (bắt lỗi đổi hop/hình học frame),
  * `events`            — CHUỖI SỰ KIỆN đầy đủ `[frame_idx, START|END, lookback_frames]`,
  * `prob_sha256`       — hash của vector xác suất từng frame (làm tròn 6 chữ số),
  * `prob_sum_round6`   — tổng xác suất (đọc được khi hash lệch, để biết lệch BAO NHIÊU).

Chạy: `pytest -m slow backend/tests/test_62_vad_onnx_parity.py -q`
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
WAV_DIR = PROJECT_ROOT / "wav_test"
GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "vad_onnx_golden.json"

SILERO_ONNX = MODELS_DIR / "silero_vad.onnx"
FIRERED_DIR = MODELS_DIR / "firered_stream" / "Stream-VAD"
FIRERED_ONNX = FIRERED_DIR / "fireredvad_stream_vad_with_cache.onnx"

#: SHA-256 của hai file model ONNX chính thức (khớp nguồn upstream đã kiểm chứng).
SILERO_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
FIRERED_ONNX_SHA256 = "b3c97836130dc34fc32d56fab551e88cf9454511de2b3c250a5e6578dee74b93"

pytestmark = pytest.mark.slow


def _golden() -> dict:
    if not GOLDEN_PATH.is_file():
        pytest.skip(f"chưa có fixture vàng: {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _load_pcm16k(name: str, max_sec: float) -> np.ndarray:
    """Đọc wav → PCM int16 16 kHz mono (resample nếu file gốc khác 16 kHz)."""
    x, sr = sf.read(str(WAV_DIR / name), dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    pcm = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(pcm[: int(16000 * max_sec)], dtype=np.int16)


def _run_engine(engine_name: str, pcm: np.ndarray) -> dict:
    """Chạy 1 engine và trả đúng cấu trúc đã ghim."""
    from backend.vad.engines import VADEngineFactory

    engine = VADEngineFactory.get_engine(engine_name)
    hop = engine.frame_samples
    state = engine.create_initial_state()
    probs: list[float] = []
    events: list[list] = []
    for i in range(pcm.size // hop):
        res = engine.is_speech(pcm[i * hop : (i + 1) * hop], state)
        probs.append(round(float(res.probability), 6))
        if res.event:
            events.append([i, res.event, int(res.lookback_frames)])
    arr = np.asarray(probs, dtype=np.float64)
    return {
        "hop_samples": int(hop),
        "frames": len(probs),
        "events": events,
        "prob_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "prob_sum_round6": round(float(arr.sum()), 4),
    }


# ─────────────────────────────────────────────── 0. toàn vẹn file model (tầng A)


@pytest.mark.parametrize(
    "path,sha256",
    [(SILERO_ONNX, SILERO_SHA256), (FIRERED_ONNX, FIRERED_ONNX_SHA256)],
)
def test_model_onnx_dung_file_chinh_thuc(path: Path, sha256: str):
    """File ONNX phải tồn tại, đủ lớn và khớp SHA-256 của nguồn chính thức."""
    if not path.is_file():
        pytest.skip(f"chưa có model {path.name} (sẽ được tải ở lần chạy backend đầu tiên)")
    assert path.stat().st_size > 1_000_000, f"{path.name} quá nhỏ — file tải hỏng?"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == sha256, (
        f"{path.name} có SHA-256 {digest}, khác bản đã kiểm chứng {sha256}. "
        "Model khác ⇒ hành vi VAD có thể khác ⇒ phải cập nhật fixture vàng một cách có ý thức."
    )


# ─────────────────────────────────────────────── 1. không kéo PyTorch (tầng A)


def test_vad_onnx_khong_nap_torch():
    """Nạp + chạy VAD ONNX trong tiến trình SẠCH: `torch` không được xuất hiện.

    Đây là bài kiểm chứng cốt lõi của mục tiêu "bỏ PyTorch runtime". Chạy trong TIẾN TRÌNH CON
    để không bị lây `sys.modules` từ các test khác.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import numpy as np\n"
        "from backend.vad.engines import VADEngineFactory\n"
        "assert 'torch' not in sys.modules, 'torch bị nạp NGAY khi import backend.vad.engines'\n"
        "for name in ('silero-vad', 'firered-vad'):\n"
        "    eng = VADEngineFactory.get_engine(name)\n"
        "    st = eng.create_initial_state()\n"
        "    pcm = np.zeros(eng.frame_samples * 40, dtype=np.int16)\n"
        "    for i in range(40):\n"
        "        eng.is_speech(pcm[i * eng.frame_samples:(i + 1) * eng.frame_samples], st)\n"
        "    assert 'torch' not in sys.modules, f'{name} đã kéo torch vào tiến trình'\n"
        "    assert 'transformers' not in sys.modules, f'{name} đã kéo transformers'\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300
    )
    if proc.returncode != 0 and "No module named 'onnxruntime'" in (proc.stderr or ""):
        pytest.skip("chưa cài onnxruntime")
    assert proc.returncode == 0, f"tiến trình con thất bại:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "OK" in proc.stdout


# ─────────────────────────────────────────────── 2. hồi quy theo kỳ vọng vàng


def _cases() -> list[tuple[str, str]]:
    g = _golden()
    return [
        (wav, engine)
        for wav, per_engine in g["engines"].items()
        for engine in per_engine
        if (WAV_DIR / wav).is_file()
    ]


@pytest.mark.parametrize("wav,engine_name", _cases())
def test_vad_khop_ky_vong_vang(wav: str, engine_name: str):
    """Chuỗi sự kiện + xác suất từng frame phải khớp bộ số VÀNG đã chứng minh với torch."""
    g = _golden()
    expected = g["engines"][wav][engine_name]
    pcm = _load_pcm16k(wav, float(g.get("max_seconds", 30.0)))
    got = _run_engine(engine_name, pcm)

    assert got["hop_samples"] == expected["hop_samples"], (
        f"{wav}/{engine_name}: hop đổi {expected['hop_samples']} → {got['hop_samples']}"
    )
    assert got["frames"] == expected["frames"], (
        f"{wav}/{engine_name}: số frame đổi {expected['frames']} → {got['frames']}"
    )
    assert got["events"] == expected["events"], (
        f"{wav}/{engine_name}: CHUỖI SỰ KIỆN đổi.\n"
        f"  vàng: {expected['events']}\n  nay : {got['events']}"
    )
    assert abs(got["prob_sum_round6"] - expected["prob_sum_round6"]) <= 1e-3, (
        f"{wav}/{engine_name}: tổng xác suất lệch "
        f"{expected['prob_sum_round6']} → {got['prob_sum_round6']}"
    )
    assert got["prob_sha256"] == expected["prob_sha256"], (
        f"{wav}/{engine_name}: vector xác suất từng frame ĐÃ ĐỔI "
        f"(tổng {expected['prob_sum_round6']} → {got['prob_sum_round6']}). "
        "Nếu thay đổi là CÓ CHỦ ĐÍCH, chạy lại `.research/gen_vad_golden.py` và ghi rõ lý do."
    )


def test_fixture_ghi_ro_nguon_goc():
    """Fixture vàng phải tự mô tả nguồn gốc, để người sau không 'reset' nó một cách vô ý."""
    g = _golden()
    assert "_doc" in g and "torch" in g["_doc"], "fixture phải ghi rõ nó chụp từ đường torch đã kiểm"
    assert g["engines"], "fixture rỗng"
    for wav, per_engine in g["engines"].items():
        assert per_engine, f"{wav} không có engine nào"
        for engine, data in per_engine.items():
            assert data["frames"] > 0
            assert len(data["prob_sha256"]) == 64
