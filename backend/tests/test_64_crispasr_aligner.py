"""Tầng A/B: aligner CrispASR GGUF (Qwen3-ForcedAligner-0.6B) — runtime DUY NHẤT từ Giai đoạn 3.

Tầng A (không cần model/GPU) — chốt các bảo đảm THIẾT KẾ:
  1. `backend/utils/crispasr_native.py` KHÔNG kéo PyTorch và KHÔNG import `backend.asr` (điều kiện
     sống còn để tiến trình con không bind nhầm `ggml-base.dll` → `WinError 127`).
  2. `_sanitized_env()` loại mọi thư mục chứa `ggml*.dll` khác khỏi `PATH` của tiến trình con.
  3. Đường torch đã bị XOÁ hoàn toàn: `load_model` / `_use_crispasr` / `ensure_forced_aligner_model`
     / cây `backend/asr/qwen_asr/` đều không còn.
  4. `backend/bin/crispasr/` tự tải được khi thiếu (vì `ggml-cuda.dll` 149,63 MB vượt giới hạn
     100 MB/file của GitHub nên không commit được).

Tầng B (`-m slow`, cần DLL + model + audio thật):
  5. Căn chỉnh chạy được, mốc ĐƠN ĐIỆU, nằm trong độ dài audio, số đơn vị > 0.
  6. Tầng PHỤ ĐỀ khớp **kỳ vọng VÀNG** chụp từ đường torch trước khi nó bị xoá — tức bảo đảm
     "đổi runtime không đổi phụ đề" vẫn được kiểm mà không cần PyTorch.
  7. Import + suy luận KHÔNG nạp `torch`/`transformers` (kiểm trong tiến trình con sạch).

Chạy: `pytest -m slow backend/tests/test_64_crispasr_aligner.py -q`
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WAV_DIR = PROJECT_ROOT / "wav_test"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"

MODEL_GGUF = MODELS_DIR / "Qwen3-ForcedAligner-0.6B-q4_k.gguf"
LIB_DLL = PROJECT_ROOT / "backend" / "bin" / "crispasr" / "crispasr.dll"

_HAS_RUNTIME = LIB_DLL.is_file() and MODEL_GGUF.is_file()

#: (wav, ngôn ngữ) — văn bản lấy từ `.txt` cùng tên.
AB_CASES = [
    ("Japanese_5s.wav", "Japanese"),
    ("Chinese_noise_28s.wav", "Chinese"),
]


# ─────────────────────────────────────────── tầng A: thiết kế & cách ly


def test_native_module_khong_keo_torch_va_khong_import_backend_asr():
    """`backend.utils.crispasr_native` phải sạch: không torch, không `backend.asr`.

    Đây là điều kiện SỐNG CÒN của thiết kế cách ly: `backend/asr/__init__.py` gọi `bootstrap()`
    nạp `transcribe.dll` ⇒ kéo `backend/bin/ggml*.dll` vào tiến trình ⇒ `crispasr.dll` bind nhầm
    `ggml-base.dll` ⇒ `OSError [WinError 127] The specified procedure could not be found`.
    Chạy trong TIẾN TRÌNH CON để không bị lây `sys.modules` từ test khác.
    """
    code = (
        "import sys\n"
        "import backend.utils.crispasr_native as m\n"
        "assert 'torch' not in sys.modules, 'crispasr_native đã kéo torch'\n"
        "assert 'backend.asr' not in sys.modules, 'crispasr_native đã kéo backend.asr'\n"
        "assert 'transcribe_cpp' not in sys.modules, 'crispasr_native đã nạp transcribe.cpp'\n"
        "assert callable(m.aligner_lib_dir) and callable(m.available)\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_available_tra_ve_tuple_khong_nem():
    """`available()` phải LUÔN trả `(bool, str)` — dùng để quyết định fallback, không được ném."""
    from backend.utils.crispasr_native import available

    ok, reason = available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    if ok:
        assert reason == ""
    else:
        assert reason, "khi không dùng được phải nói RÕ lý do"


def test_sanitized_env_loai_thu_muc_ggml_khac(tmp_path):
    """`PATH` của tiến trình con không được chứa thư mục nào khác có `ggml*.dll`."""
    import os

    from backend.asr.crispasr_aligner import _sanitized_env

    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    (fake_bin / "ggml-base.dll").write_bytes(b"x")
    keep = tmp_path / "keep"
    keep.mkdir()

    lib_dir = tmp_path / "crispasr"
    lib_dir.mkdir()
    (lib_dir / "ggml-base.dll").write_bytes(b"y")

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join([str(fake_bin), str(keep), str(lib_dir), old_path])
    try:
        env = _sanitized_env(lib_dir)
    finally:
        os.environ["PATH"] = old_path

    parts = [p for p in env["PATH"].split(os.pathsep) if p]
    assert str(fake_bin) not in parts, "thư mục ggml khác phải bị loại khỏi PATH"
    assert str(keep) in parts, "thư mục không chứa ggml phải được giữ"
    assert str(lib_dir) in parts, "thư mục DLL của chính aligner phải được giữ"


def test_import_forced_aligner_khong_keo_torch():
    """`import backend.asr.forced_aligner` KHÔNG được nạp `torch`/`transformers`.

    Đây là hệ quả trực tiếp của việc chuyển mặc định sang `crispasr` + làm LAZY mọi thứ thuộc
    đường torch (`_torch()`, `_load_qwen_asr_class()`, `_cuda_*`). Trước đây file này có
    `import torch` ở cấp module nên MỌI tiến trình chạm tới nó đều kéo PyTorch dù không dùng.
    """
    code = (
        "import sys\n"
        "import backend.asr.forced_aligner as fa\n"
        "bad = [m for m in ('torch', 'transformers', 'qwen_asr') if m in sys.modules]\n"
        "assert not bad, f'import forced_aligner đã kéo: {bad}'\n"
        "assert not hasattr(fa, '_torch'), 'helper torch còn sót lại'\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_config_mac_dinh_la_crispasr():
    """Mặc định phải là `crispasr` — đó là runtime DUY NHẤT còn lại sau Giai đoạn 3.

    Đường `transformers` + PyTorch đã bị xoá hoàn toàn (xem `backend/asr/forced_aligner.py`), nên
    giá trị này không còn là "lựa chọn giữa hai đường" mà là ghi nhận runtime đang dùng.
    """
    from backend.config import config

    assert config.forced_aligner.backend == "crispasr"
    assert hasattr(config.forced_aligner, "auto_download")
    assert int(config.forced_aligner.n_threads) >= 1


def test_khong_con_duong_torch_trong_aligner():
    """Không được còn API/dấu vết của đường torch trong `ForcedAlignerService`.

    Chốt Giai đoạn 3 ở mức mã nguồn: nếu ai đó vô tình khôi phục `load_model()` /
    `ensure_forced_aligner_model` / `_use_crispasr` thì test này báo ngay.
    """
    from backend.asr import forced_aligner as fa

    for gone in ("load_model", "_use_crispasr", "_align_crispasr", "ensure_forced_aligner_model"):
        assert not hasattr(fa.ForcedAlignerService, gone), f"API đường torch còn sót: {gone}"
        assert not hasattr(fa, gone), f"hàm module đường torch còn sót: {gone}"
    for gone in ("ALIGNER_LOCAL_DIR", "ALIGNER_DEFAULT_REPO_ID", "_torch", "_load_qwen_asr_class"):
        assert not hasattr(fa, gone), f"hằng số/helper đường torch còn sót: {gone}"
    # Cây vendored qwen_asr phải đã bị xoá khỏi đĩa.
    assert not (PROJECT_ROOT / "backend" / "asr" / "qwen_asr").exists(), (
        "cây vendored backend/asr/qwen_asr/ phải đã bị xoá"
    )
    # `is_ready()` là API thay thế để biết runtime có dùng được không.
    ok, reason = fa.ForcedAlignerService.is_ready()
    assert isinstance(ok, bool) and isinstance(reason, str)


def test_is_ready_tra_ve_tuple_khong_nem(monkeypatch, tmp_path):
    """`is_ready()` phải LUÔN trả `(bool, str)` — dùng cho pre-flight, không được ném."""
    from backend.asr.forced_aligner import ForcedAlignerService

    ok, reason = ForcedAlignerService.is_ready()
    assert isinstance(ok, bool) and isinstance(reason, str)

    # Trỏ tới thư mục rỗng ⇒ phải báo chưa sẵn sàng, kèm lý do rõ ràng.
    monkeypatch.setenv("CRISPASR_LIB_DIR", str(tmp_path / "khong_ton_tai"))
    ok2, reason2 = ForcedAlignerService.is_ready()
    assert ok2 is False
    assert reason2, "khi chưa sẵn sàng phải nói RÕ lý do"


# ─────────────────────────────────────────── tầng B: chạy thật


def _skip_if_no_runtime() -> None:
    if not LIB_DLL.is_file():
        pytest.skip(f"chưa có DLL CrispASR: {LIB_DLL.parent}")
    if not MODEL_GGUF.is_file():
        pytest.skip(f"chưa có model GGUF: {MODEL_GGUF}")


def _load_pcm16k(path: Path, max_sec: float = 30.0) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x[: int(16000 * max_sec)], dtype=np.float32)


@pytest.fixture(scope="module")
def crispasr_client():
    _skip_if_no_runtime()
    from backend.asr.crispasr_aligner import CrispASRAlignerClient

    client = CrispASRAlignerClient.get_instance()
    client.prewarm()
    yield client
    client.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("wav,lang", AB_CASES)
def test_can_chinh_chay_duoc_va_moc_don_dieu(wav: str, lang: str, crispasr_client):
    """Căn chỉnh thật: mốc đơn điệu, nằm trong audio, số đơn vị > 0."""
    path = WAV_DIR / wav
    if not path.is_file():
        pytest.skip(f"thiếu audio {wav}")
    pcm = _load_pcm16k(path)
    text = path.with_suffix(".txt").read_text(encoding="utf-8").strip()
    audio_sec = pcm.size / 16000

    words = crispasr_client.align(pcm, text)
    assert words, "căn chỉnh không trả về đơn vị nào"
    assert len(words) > 5

    prev_t0 = -1.0
    for w in words:
        assert w["t0"] <= w["t1"] + 1e-6, f"đơn vị âm/ngược: {w}"
        assert w["t0"] >= prev_t0 - 1e-6, f"mốc KHÔNG đơn điệu tại {w}"
        assert -0.01 <= w["t1"] <= audio_sec + 0.5, f"mốc vượt ngoài audio ({audio_sec:.2f}s): {w}"
        prev_t0 = w["t0"]

    # Ghép lại phải phủ gần hết vùng tiếng nói (không được co cụm về một chỗ).
    span = words[-1]["t1"] - words[0]["t0"]
    assert span > 0.5 * audio_sec, f"mốc co cụm bất thường: span={span:.2f}s / audio={audio_sec:.2f}s"


@pytest.mark.slow
def test_align_that_su_khong_nap_torch():
    """Căn chỉnh thật phải chạy mà KHÔNG nạp `torch`/`transformers`.

    Chạy trong TIẾN TRÌNH CON SẠCH: `sys.modules` là trạng thái toàn cục, nên nếu chạy trong
    tiến trình pytest (đã bị các test khác nạp torch) thì khẳng định này vô nghĩa.

    Đây là bằng chứng "đã THAY THẾ đường torch", không chỉ là "có thêm một đường mới".
    """
    _skip_if_no_runtime()
    code = (
        "import sys\n"
        "import numpy as np, soundfile as sf\n"
        "from pathlib import Path\n"
        "from backend.config import config\n"
        "assert config.forced_aligner.backend == 'crispasr', config.forced_aligner.backend\n"
        "from backend.asr.forced_aligner import ForcedAlignerService\n"
        "from backend.asr.crispasr_aligner import CrispASRAlignerClient\n"
        "p = Path('wav_test/Japanese_5s.wav')\n"
        "x, sr = sf.read(str(p), dtype='float32', always_2d=True)\n"
        "x = np.ascontiguousarray(x[:, 0], dtype=np.float32)\n"
        "text = p.with_suffix('.txt').read_text(encoding='utf-8').strip()\n"
        "svc = ForcedAlignerService.get_instance()\n"
        "words = svc.align(x, text, language='Japanese')\n"
        "assert words, 'khong can chinh duoc don vi nao'\n"
        "assert not hasattr(ForcedAlignerService, '_shared_aligner'), 'thuoc tinh duong torch con sot'\n"
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]\n"
        "assert not bad, f'da nap {bad} du chay duong GGUF'\n"
        "CrispASRAlignerClient.get_instance().shutdown()\n"
        "print('OK', len(words))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "OK" in proc.stdout


#: Kỳ vọng VÀNG cho tầng phụ đề, CHỤP TỪ ĐƯỜNG TORCH trước khi nó bị xoá (Giai đoạn 3).
#:
#: Đây là cách giữ lại bảo đảm "đổi runtime không đổi phụ đề" mà KHÔNG cần torch: thay vì so
#: trực tiếp với đường torch (đã bị xoá), ta ghim kết quả mà đường đó đã cho. Số liệu lấy từ phép
#: đo A/B ngày 2026-10-06 (`.research/ab_subtitles.json`); mốc thời gian của đường torch được ghi
#: kèm để thấy độ lệch thực tế.
_GOLDEN_SUBTITLES: Dict[str, Dict[str, Any]] = {
    "Japanese_5s.wav": {
        # torch: 1 phụ đề, (0.480 → 4.720) — GGUF lệch +0,08 s ở cuối.
        "count": 1,
        "texts": ["抜群の運動神経を持ち合わせ、どんな要求にも応えてきた。"],
        "first_t0": 0.48,
        "last_t1": 4.80,
        "torch_last_t1": 4.72,
    },
    "Chinese_noise_28s.wav": {
        # torch: 10 phụ đề, cùng nội dung; chỉ 1 biên lệch 0,08 s (11,04 vs 11,12).
        "count": 10,
        "texts_first": ["在他十一岁的时候，母亲因为发生交通事故撒手人寰。"],
        "first_t0": 0.00,
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("wav,lang", AB_CASES)
def test_tang_phu_de_khop_ky_vong_vang(wav: str, lang: str, crispasr_client):
    """Cấu trúc PHỤ ĐỀ phải khớp kỳ vọng VÀNG chụp từ đường torch trước khi nó bị xoá.

    Đây là phép đo quyết định của Giai đoạn 2: đơn vị căn chỉnh khác nhau (CrispASR tách CJK theo
    KÝ TỰ, đường torch tách tiếng Nhật theo TỪ bằng nagisa) KHÔNG được làm phụ đề chẻ nhỏ hơn hay
    đổi nội dung. Giai đoạn 3 xoá đường torch nên mốc so sánh được GHIM lại ở `_GOLDEN_SUBTITLES`
    thay vì tính trực tiếp — nhờ vậy bảo đảm này vẫn được kiểm mà không cần PyTorch.
    """
    golden = _GOLDEN_SUBTITLES.get(wav)
    if golden is None:
        pytest.skip(f"chưa có kỳ vọng vàng cho {wav}")
    path = WAV_DIR / wav
    if not path.is_file():
        pytest.skip(f"thiếu audio {wav}")

    pcm = _load_pcm16k(path)
    text = path.with_suffix(".txt").read_text(encoding="utf-8").strip()

    from backend.asr.forced_aligner import ForcedAlignerService

    svc = ForcedAlignerService.get_instance()
    words = svc.align(audio=pcm, text=text, language=lang, attach_source_punctuation=True)
    out = ForcedAlignerService.group_words_to_subtitles(words, language=lang)
    subs = ForcedAlignerService.split_oversized_sentences(out, language=lang)

    assert len(subs) == golden["count"], (
        f"{wav}: số phụ đề lệch kỳ vọng vàng ({golden['count']} → {len(subs)}). "
        f"Nội dung: {[s.text for s in subs]}"
    )
    for expected_text in golden.get("texts", []):
        assert any(s.text == expected_text for s in subs), (
            f"{wav}: thiếu phụ đề {expected_text!r} — có {[s.text for s in subs]}"
        )
    for expected_text in golden.get("texts_first", []):
        assert subs[0].text == expected_text, f"{wav}: phụ đề đầu đổi: {subs[0].text!r}"

    assert abs(subs[0].start_time - golden["first_t0"]) <= 0.2, (
        f"{wav}: mốc bắt đầu lệch kỳ vọng vàng ({golden['first_t0']} vs {subs[0].start_time})"
    )
    if "last_t1" in golden:
        tol = 0.2 + abs(golden.get("torch_last_t1", golden["last_t1"]) - golden["last_t1"])
        assert abs(subs[-1].end_time - golden["last_t1"]) <= tol, (
            f"{wav}: mốc kết thúc lệch kỳ vọng vàng ({golden['last_t1']} vs {subs[-1].end_time})"
        )
