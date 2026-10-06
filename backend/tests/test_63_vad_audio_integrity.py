"""Tầng B: **toàn vẹn âm thanh** khi đi qua VAD ONNX — không lệch pha, không đổi mẫu,
không suy giảm chất lượng.

VÌ SAO CẦN BỘ TEST NÀY
======================
VAD là **tap thụ động** (`backend/vad/base.py`): nó chỉ ĐỌC frame PCM để quyết định nói/im, còn
audio gửi cho ASR phải là **đúng byte client gửi**. Một engine VAD vi phạm điều đó (resample,
gain, clip, hay chỉ đơn giản là `int16 → float32 → int16`) sẽ làm hỏng chất lượng nhận dạng mà
KHÔNG hề báo lỗi. Bộ test này khóa hợp đồng đó bằng 4 góc nhìn độc lập:

1. **Engine không sửa buffer đầu vào** — gọi `is_speech()` trên một frame chứa mẫu "bẫy"
   (các giá trị KHÔNG sống sót qua vòng `int16 → float32 → int16`) rồi khẳng định mảng vẫn
   nguyên từng bit. Buffer đọc từ `bytes` là read-only nên mọi ghi tại chỗ sẽ ném lỗi.
2. **Mọi frame chuyển tiếp khớp ĐÚNG mốc thời gian của nó** — engine báo `frame_ts` là thời điểm
   của mẫu đầu tiên trong frame; ta khẳng định `bytes` chuyển tiếp == PCM gốc tại ĐÚNG mốc đó.
   Đây là bằng chứng mạnh nhất: nó loại trừ đồng thời lệch pha, đổi thứ tự, chèn/mất mẫu, và
   mọi phép biến đổi giá trị (gain/clip/lượng tử hoá lại).
3. **Không lệch pha** — dịch khối đã chuyển tiếp đi ±1 mẫu thì nội dung phải KHÁC (nếu bằng
   nhau ở một độ dịch khác 0 thì đã có lệch pha/đệm).
4. **Chuỗi chuyển tiếp là một slice LIỀN MẠCH, không trùng, không hụt** — kể cả phần pre-roll
   xả lại khi START (`lookback_frames`).

Chạy: `pytest -m slow backend/tests/test_63_vad_audio_integrity.py -q`
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WAV_DIR = PROJECT_ROOT / "wav_test"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
SPEECH_WAV = WAV_DIR / "Russian_4s.wav"

#: Chỉ kiểm 2 engine đã chuyển sang ONNX (FSMN là tuỳ chọn, còn cần torch).
ENGINES = ["firered-vad", "silero-vad"]
#: Bước nhảy frame của từng engine (mẫu) — phải khớp `engine.frame_samples`.
_HOP = {"firered-vad": 160, "silero-vad": 512}

_CHUNK_SEC = 0.02  # client gửi từng 20 ms như thật

pytestmark = pytest.mark.slow


def _skip_if_no_model(engine_name: str) -> None:
    if engine_name == "silero-vad" and not (MODELS_DIR / "silero_vad.onnx").is_file():
        pytest.skip("chưa có silero_vad.onnx")
    if engine_name == "firered-vad" and not (
        MODELS_DIR / "firered_stream" / "Stream-VAD" / "fireredvad_stream_vad_with_cache.onnx"
    ).is_file():
        pytest.skip("chưa có fireredvad_stream_vad_with_cache.onnx")


def _speech_pcm() -> np.ndarray:
    """Speech thật 16 kHz, kẹp 0,5 s lặng đầu + 2 s lặng cuối (để có cả START lẫn END)."""
    x, sr = sf.read(str(SPEECH_WAV), dtype="int16", always_2d=True)
    assert sr == 16000
    x = np.ascontiguousarray(x[:, 0], dtype=np.int16)
    return np.concatenate(
        [np.zeros(8000, dtype=np.int16), x, np.zeros(32000, dtype=np.int16)]
    )


class _Recorder:
    """Ghi lại mọi thứ VAD chuyển tiếp, kèm mốc thời gian engine báo."""

    def __init__(self) -> None:
        self.chunks: list[tuple[bytes, float, str]] = []
        self.starts = 0
        self.ends = 0

    def on_chunk(self, data: bytes, ts: float, tag: str) -> None:
        self.chunks.append((bytes(data), float(ts), str(tag)))

    @property
    def forward(self) -> bytes:
        return b"".join(c for c, _, _ in self.chunks)


def _run(engine_name: str, pcm: np.ndarray, *, threshold=None, silence_ms=None) -> _Recorder:
    from backend.vad.processor import VADStreamProcessor

    rec = _Recorder()
    proc = VADStreamProcessor(
        vad_engine=engine_name,
        threshold=threshold,
        silence_duration_ms=silence_ms,
        on_speech_start=lambda: setattr(rec, "starts", rec.starts + 1),
        on_speech_chunk=rec.on_chunk,
        on_speech_end=lambda: setattr(rec, "ends", rec.ends + 1),
    )
    proc.prewarm()

    raw = pcm.tobytes()
    chunk_bytes = int(16000 * _CHUNK_SEC) * 2
    for i in range(0, len(raw), chunk_bytes):
        # `capture_timestamp` = mốc của BYTE ĐẦU chunk ⇒ mốc frame suy trực tiếp từ mẫu.
        proc.feed_chunk(raw[i : i + chunk_bytes], capture_timestamp=(i / 2) / 16000.0)
    return rec


# ─────────────────────────────────────── 1. engine không sửa buffer đầu vào


@pytest.mark.parametrize("engine_name", ENGINES)
def test_engine_khong_sua_buffer_dau_vao(engine_name):
    """`is_speech()` phải là thao tác CHỈ-ĐỌC trên frame PCM.

    Frame thử chứa "mẫu bẫy": các giá trị mà vòng `int16 → float32(/32768) → *32767 → int16`
    sẽ trả về SAI lệch (ví dụ 1 → 0, 3 → 2, -32768 → -32767). Nếu engine lượng tử hoá lại audio
    "cho tiện", test này bắt được ngay.
    """
    from backend.vad.engines import VADEngineFactory

    _skip_if_no_model(engine_name)
    engine = VADEngineFactory.get_engine(engine_name)
    state = engine.create_initial_state()

    hop = engine.frame_samples
    trap = np.array(
        [0, 1, 2, 3, 5, 7, 11, 32767, -32768, -32767, 12345, -12345, 255, -256],
        dtype=np.int16,
    )
    frame = np.resize(trap, hop).astype(np.int16)
    before = frame.copy()
    before_bytes = frame.tobytes()

    for _ in range(5):
        engine.is_speech(frame, state)

    assert frame.tobytes() == before_bytes, (
        f"{engine_name}: engine ĐÃ SỬA buffer PCM đầu vào — hợp đồng 'VAD là tap thụ động' bị vi phạm"
    )
    assert np.array_equal(frame, before), f"{engine_name}: giá trị mẫu bị đổi"


@pytest.mark.parametrize("engine_name", ENGINES)
def test_engine_khong_the_ghi_vao_buffer_chi_doc(engine_name):
    """Frame đọc từ `bytes` của processor là read-only ⇒ engine không thể ghi tại chỗ."""
    from backend.vad.engines import VADEngineFactory

    _skip_if_no_model(engine_name)
    engine = VADEngineFactory.get_engine(engine_name)
    state = engine.create_initial_state()

    hop = engine.frame_samples
    readonly = np.frombuffer(bytes(hop * 2), dtype=np.int16)
    assert not readonly.flags.writeable, "tiền đề sai: buffer phải read-only"
    engine.is_speech(readonly, state)  # không được ném


# ─────────────────────────────────────── 2 + 3 + 4. toàn vẹn luồng chuyển tiếp


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize(
    "threshold,silence_ms",
    [(None, None), (0.4, 1500), (0.5, 800)],
    ids=["mac-dinh-docs", "nguong-0.4-silence-1500", "nguong-0.5-silence-800"],
)
def test_audio_chuyen_tiep_nguyen_byte_dung_moc(engine_name, threshold, silence_ms):
    """Mọi frame chuyển tiếp phải khớp ĐÚNG mốc thời gian engine báo — không lệch pha/mẫu."""
    _skip_if_no_model(engine_name)
    pcm = _speech_pcm()
    rec = _run(engine_name, pcm, threshold=threshold, silence_ms=silence_ms)

    assert rec.starts >= 1, f"{engine_name}: không có START ⇒ không có audio chuyển tiếp để kiểm"
    assert rec.chunks, f"{engine_name}: không chuyển tiếp frame nào"

    hop = _HOP[engine_name]
    previous_end = -1
    for idx, (data, ts, tag) in enumerate(rec.chunks):
        n_samples = len(data) // 2
        assert n_samples == hop, f"chunk #{idx} ({tag}): {n_samples} mẫu, phải đúng {hop}"

        # Mốc thời gian engine báo → chỉ số mẫu tuyệt đối trong PCM gốc.
        pos = int(round(ts * 16000))
        assert pos >= 0, f"chunk #{idx}: mốc âm ({ts})"
        expected = pcm[pos : pos + n_samples]
        assert expected.size == n_samples, f"chunk #{idx}: mốc {ts:.6f}s vượt cuối PCM"

        got = np.frombuffer(data, dtype=np.int16)
        assert np.array_equal(got, expected), (
            f"{engine_name} chunk #{idx} ({tag}) @ {ts:.6f}s: mẫu KHÁC PCM gốc "
            f"(lệch tối đa {int(np.abs(got.astype(np.int32) - expected.astype(np.int32)).max())}) "
            "⇒ audio đã bị biến đổi (gain/clip/lượng tử hoá lại/resample/lệch pha)"
        )

        # Không trùng, không hụt: chunk sau phải bắt đầu ĐÚNG chỗ chunk trước kết thúc.
        # (Từ START tới END, VAD chuyển tiếp MỌI frame liên tiếp nên chuỗi phải liền mạch.)
        if previous_end >= 0:
            assert pos == previous_end, (
                f"{engine_name}: chuỗi chuyển tiếp bị HỤT/CHỒNG tại chunk #{idx}: "
                f"bắt đầu ở mẫu {pos} nhưng chunk trước kết thúc ở {previous_end}"
            )
        previous_end = pos + n_samples

    # ── Chuỗi nối lại phải là MỘT slice liền mạch của PCM gốc ──────────────────
    forwarded = rec.forward
    offset = pcm.tobytes().find(forwarded)
    assert offset >= 0, (
        f"{engine_name}: chuỗi chuyển tiếp KHÔNG phải slice liền mạch của PCM gốc "
        "(đã bị đổi thứ tự / chèn / xáo trộn)"
    )
    assert offset % 2 == 0, "slice phải căn biên mẫu 16-bit"

    # SHA-256 của phần chuyển tiếp phải bằng SHA-256 của ĐÚNG đoạn gốc tương ứng:
    # cùng một chuỗi byte ⇒ chất lượng tín hiệu KHÔNG thể suy giảm.
    assert hashlib.sha256(forwarded).digest() == hashlib.sha256(
        pcm.tobytes()[offset : offset + len(forwarded)]
    ).digest()


@pytest.mark.parametrize("engine_name", ENGINES)
def test_audio_chuyen_tiep_khong_lech_pha(engine_name):
    """Dịch khối chuyển tiếp ±1 mẫu phải cho nội dung KHÁC ⇒ khẳng định độ lệch pha = 0."""
    _skip_if_no_model(engine_name)
    pcm = _speech_pcm()
    rec = _run(engine_name, pcm)
    forwarded = rec.forward
    assert forwarded, f"{engine_name}: không có audio chuyển tiếp"

    full = pcm.tobytes()
    offset = full.find(forwarded)
    assert offset >= 0 and offset % 2 == 0

    fwd = np.frombuffer(forwarded, dtype=np.int16)
    base = offset // 2

    # Cùng vị trí ⇒ BẰNG NHAU (đã khẳng định ở test trên, nhắc lại cho mạch lạc).
    assert np.array_equal(fwd, pcm[base : base + fwd.size])
    # Lệch ±1 mẫu ⇒ PHẢI KHÁC. Nếu bằng, audio đã bị dịch pha/đệm.
    for shift in (-1, 1):
        start = base + shift
        if start < 0 or start + fwd.size > pcm.size:
            continue
        assert not np.array_equal(fwd, pcm[start : start + fwd.size]), (
            f"{engine_name}: audio chuyển tiếp trùng với bản dịch {shift:+d} mẫu ⇒ CÓ LỆCH PHA"
        )


@pytest.mark.parametrize("engine_name", ENGINES)
def test_tong_mau_chuyen_tiep_bang_so_frame_tieng_noi(engine_name):
    """Không resample: tổng số mẫu chuyển tiếp = số frame engine coi là tiếng nói."""
    _skip_if_no_model(engine_name)
    pcm = _speech_pcm()
    rec = _run(engine_name, pcm, threshold=0.4, silence_ms=1500)
    hop = _HOP[engine_name]

    forwarded_samples = len(rec.forward) // 2
    assert forwarded_samples > 0
    assert forwarded_samples % hop == 0, (
        f"{engine_name}: tổng mẫu chuyển tiếp ({forwarded_samples}) không phải bội số của hop "
        f"({hop}) ⇒ audio đã bị resample/cắt lẻ"
    )
    # Không thể chuyển tiếp nhiều hơn toàn bộ PCM đầu vào.
    assert forwarded_samples <= pcm.size
