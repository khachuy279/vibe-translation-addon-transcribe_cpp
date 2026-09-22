"""Benchmark VAD trên **speech ĐỌC SẠCH** — đúng loại dữ liệu của bảng nhà cung cấp.

VÌ SAO CẦN: bảng của FireRedVAD (F1 97,57 · Silero 95,95 · FunASR 90,91) được đo
**non-streaming trên FLEURS-VAD-102** — clip speech đọc ~10 s, nhãn nhị phân
(`external/FireRedVAD/README.md`). AVA-Speech (`test_43`) là audio PHIM (nhạc, hiệu ứng,
nói chồng tiếng) nên điểm thấp hơn hẳn và KHÔNG so trực tiếp được.

Test này dựng một "FLEURS-like" tại chỗ từ các clip speech đọc sạch có sẵn trong repo:
    [lặng 0,5 s] + clip + [lặng 1 s] + clip + … + [lặng 0,5 s]

⚠️ KHÔNG dùng "cả clip là speech" làm nhãn frame-level: speech đọc có khoảng nghỉ bên trong,
mà bất kỳ VAD nào tôn trọng `min_silence` đều phải cắt ở đó — chấm như vậy là phạt oan
(FireRed chỉ đạt recall 0,81 khi thử cách này). Thay vào đó chấm theo **phát hiện từng clip**
(giống cách bảng nhà cung cấp đánh giá trên clip ngắn) + **báo động giả trên khoảng lặng chèn**,
và in thêm F1 đối chiếu **tham chiếu năng lượng** (không phải nhãn người) để tham khảo.

Kỳ vọng: cả 3 engine phát hiện ≥ 95 % số clip, gần như không báo động giả trong khoảng lặng.
Nếu engine nào tụt ở đây thì đó là LỖI TÍCH HỢP/CẤU HÌNH thật, không phải do domain.

Chạy: `pytest -m slow backend/tests/test_44_vad_clean_speech_benchmark.py -q -s`
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.vad.processor import VADStreamProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FRAME_MS = 10
SR = 16000
CHUNK_MS = 20
#: Khoảng lặng chèn vào phải DÀI HƠN hangover lớn nhất trong 3 engine (FSMN 800 ms, đo thực
#: ~1,2 s kể cả hop 60 ms và bước xác nhận) để phần "lõi khoảng lặng" còn thật sự là lặng.
GAP_SEC = 3.0
EDGE_SEC = 0.5

#: Clip speech ĐỌC SẠCH 16 kHz mono có sẵn trong repo (không phải tải gì).
CLIPS = (
    "external/transcribe.cpp/samples/jfk.wav",
    "external/transcribe.cpp/samples/ar-short.wav",
    "external/transcribe.cpp/samples/german.wav",
    "external/transcribe.cpp/samples/cj-swimming-drop.wav",
    "backend/models/fsmn_vad/example/vad_example.wav",
)

ENGINES = ("firered-vad", "silero-vad", "fsmn-vad")

#: Ngưỡng sàn in-domain: đặt thấp hơn số đo để bắt hồi quy, không chặn dao động nhỏ.
MIN_DETECTION_RATE = 0.95     # tỉ lệ clip được phát hiện (>= 50 % vùng năng lượng lõi)
MAX_FALSE_ALARM = 0.20        # tỉ lệ frame báo động giả trong PHẦN LÕI của khoảng lặng chèn
ENERGY_RATIO = 0.12           # ngưỡng năng lượng lõi theo đỉnh của chính clip
MIN_F1_VS_ENERGY = 0.75       # sàn tham khảo cho F1 đối chiếu tham chiếu năng lượng
TAIL_LEAD_FRAMES = 3          # chừa 30 ms trước mép clip kế tiếp (tránh phạt pre-pad)

#: Im lặng mặc định (docs) của từng engine — dùng để chừa phần "hangover hợp lệ" khi tính
#: báo động giả: VAD ĐƯỢC THIẾT KẾ để kéo dài đoạn thêm `min_silence`, nên phần đuôi đó
#: không phải lỗi. (Cùng bảng với `test_41` và `report/02_vad/ava_speech_report.md`.)
DOCS_SILENCE_MS = {"firered-vad": 200, "silero-vad": 100, "fsmn-vad": 800}


def _frame_count(samples: int) -> int:
    return samples * 1000 // (SR * FRAME_MS)


@pytest.fixture(scope="module")
def clean_stream() -> dict:
    """Ghép clip sạch thành 1 stream + vùng lõi năng lượng từng clip + mặt nạ khoảng lặng."""
    pieces = [np.zeros(int(EDGE_SEC * SR), dtype=np.int16)]
    clips = []
    for rel in CLIPS:
        path = PROJECT_ROOT / rel
        if not path.exists():
            pytest.skip(f"thiếu clip sạch: {rel}")
        audio, sr = sf.read(str(path), dtype="int16")
        assert sr == SR, f"{rel}: cần 16 kHz"
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = np.ascontiguousarray(audio.astype(np.int16))
        start = sum(len(p) for p in pieces)
        pieces.append(audio)
        pieces.append(np.zeros(int(GAP_SEC * SR), dtype=np.int16))
        clips.append({"path": rel, "start": start, "samples": len(audio)})

    pcm = np.concatenate(pieces)
    n_frames = _frame_count(len(pcm))

    # Vùng lõi năng lượng của từng clip (10 ms/frame, ngưỡng theo đỉnh của chính clip đó).
    core = np.zeros(n_frames, dtype=bool)
    cores = []
    for clip in clips:
        a = clip["start"] // (SR * FRAME_MS // 1000)
        b = (clip["start"] + clip["samples"]) // (SR * FRAME_MS // 1000)
        seg = pcm[a * 160:(b) * 160].astype(np.float32)
        n_win = len(seg) // 160
        env = np.sqrt((seg[: n_win * 160].reshape(n_win, 160) ** 2).mean(axis=1))
        voiced = np.flatnonzero(env > ENERGY_RATIO * env.max())
        c0, c1 = a + int(voiced[0]), a + int(voiced[-1]) + 1
        core[c0:c1] = True
        cores.append((c0, c1))
        clip["core"] = (c0, c1)

    return {
        "pcm": pcm,
        "n_frames": n_frames,
        "core": core,
        "gap": ~core,
        "clips": clips,
        "cores": cores,
        "duration_sec": len(pcm) / SR,
    }


def _run_engine(engine_name: str, pcm: np.ndarray, n_frames: int) -> np.ndarray:
    """Stream qua VADProcessor và dựng timeline 10 ms từ chính frame gửi ASR."""
    pred = np.zeros(n_frames, dtype=bool)
    proc = VADStreamProcessor(vad_engine=engine_name, on_speech_chunk=lambda *_: None)
    proc.prewarm()

    def _on_chunk(frame_bytes: bytes, ts: float, tag: str) -> None:
        a = int(round(ts * SR))
        b = a + len(frame_bytes) // 2
        f0 = max(0, a * 1000 // (SR * FRAME_MS))
        f1 = min(n_frames, max(f0 + 1, (b * 1000 + SR * FRAME_MS - 1) // (SR * FRAME_MS)))
        pred[f0:f1] = True

    proc.on_speech_chunk = _on_chunk

    chunk_samples = int(SR * CHUNK_MS / 1000)
    for i in range(0, len(pcm), chunk_samples):
        proc.feed_chunk(pcm[i:i + chunk_samples].tobytes(), capture_timestamp=i / SR)
    return pred


def _gap_core_mask(core: np.ndarray, hangover_ms: int) -> np.ndarray:
    """Phần LÕI của các khoảng lặng chèn vào (bỏ phần hangover hợp lệ sau mỗi clip).

    VAD được thiết kế để kéo dài đoạn thêm `min_silence` (hangover tích hợp trong VAD), nên
    tính báo động giả trên toàn bộ khoảng lặng sẽ phạt oan — nhất là FSMN với 800 ms.
    """
    hang_frames = int(hangover_ms / FRAME_MS)
    gap = ~core
    out = np.zeros_like(gap)
    idx = np.flatnonzero(gap)
    if idx.size == 0:
        return out
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        a = int(run[0]) + hang_frames
        b = int(run[-1]) + 1 - TAIL_LEAD_FRAMES
        if b > a:
            out[a:b] = True
    return out


def _metrics(pred: np.ndarray, stream: dict, engine_name: str) -> dict:
    core = stream["core"]
    tp = int((pred & core).sum())
    fp = int((pred & ~core).sum())
    fn = int((~pred & core).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    detected = 0
    for c0, c1 in stream["cores"]:
        covered = int(pred[c0:c1].sum())
        if covered >= 0.5 * max(1, c1 - c0):
            detected += 1

    gap_core = _gap_core_mask(core, DOCS_SILENCE_MS[engine_name])
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
        "detection_rate": round(detected / max(1, len(stream["cores"])), 4),
        "false_alarm": round(float((pred & gap_core).sum()) / max(1, int(gap_core.sum())), 4),
        "false_alarm_raw": round(float((pred & ~core).sum()) / max(1, int((~core).sum())), 4),
    }


@pytest.mark.slow
@pytest.mark.parametrize("engine_name", list(ENGINES))
def test_vad_tren_speech_doc_sach(engine_name, clean_stream):
    """Speech đọc sạch (FLEURS-like): mọi engine phải phát hiện được từng clip."""
    pred = _run_engine(engine_name, clean_stream["pcm"], clean_stream["n_frames"])
    m = _metrics(pred, clean_stream, engine_name)

    print(
        f"\n[{engine_name}] {len(clean_stream['clips'])} clip sạch · "
        f"{clean_stream['duration_sec']:.0f}s audio\n"
        f"   phát hiện {m['detection_rate'] * 100:.0f}% clip · "
        f"báo động giả (lõi khoảng lặng, đã trừ hangover {DOCS_SILENCE_MS[engine_name]} ms) = "
        f"{m['false_alarm']:.4f} (thô: {m['false_alarm_raw']:.4f})\n"
        f"   (tham khảo) F1 so tham chiếu năng lượng: P={m['precision']:.3f} "
        f"R={m['recall']:.3f} F1={m['f1']:.3f}"
    )

    assert m["detection_rate"] >= MIN_DETECTION_RATE, (
        f"{engine_name}: chỉ phát hiện {m['detection_rate'] * 100:.0f}% clip speech ĐỌC SẠCH — "
        f"đây là lỗi tích hợp/cấu hình, không thể là do domain (bảng nhà cung cấp đo trên đúng "
        f"loại dữ liệu này)"
    )
    assert m["false_alarm"] <= MAX_FALSE_ALARM, (
        f"{engine_name}: báo động giả {m['false_alarm']} trên các khoảng lặng chèn vào"
    )
    assert m["f1"] >= MIN_F1_VS_ENERGY, (
        f"{engine_name}: F1 {m['f1']} so tham chiếu năng lượng quá thấp"
    )
