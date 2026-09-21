"""Fixture dùng chung cho kiểm thử VAD trên bộ **AVA-Speech**.

Nguồn dữ liệu (đã có sẵn trong repo):
    wav_test/vad/5BDj0ow5hnA.wav            48 kHz · stereo · 4178 s
    wav_test/vad/ava_speech_labels_v1.csv   nhãn 10 ms theo từng video

Nhãn của video này phủ LIÊN TỤC 900 → 1800 s (đã kiểm chứng), nên đoạn 10 phút
900 → 1500 s là vùng có nhãn đầy đủ. Đã kiểm chứng timeline khớp ở offset 0
(`scratch/ava_align_check.py`: F1 = 0,726 tại offset 0 so với ≤ 0,702 ở mọi offset khác).

Đoạn 10 phút được **dẫn xuất một lần** rồi cache lại:
    wav_test/vad/derived/5BDj0ow5hnA_900s_600s_16k_mono.wav   (19,2 MB, đã gitignore)

⚠️ Resample 48 kHz → 16 kHz ở đây là bước CHUẨN BỊ DỮ LIỆU TEST (offline). Trong pipeline
thật, plugin đã gửi thẳng PCM 16 kHz mono Int16 nên VAD/ASR **không** resample gì cả.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import time

import numpy as np
import soundfile as sf

# ── Vùng test ────────────────────────────────────────────────────────────────
VIDEO_ID = "5BDj0ow5hnA"
SEGMENT_START_SEC = 900.0
SEGMENT_DUR_SEC = 600.0        # 10 phút
TARGET_SR = 16000
FRAME_MS = 10                  # độ phân giải timeline khi so với nhãn
CHUNK_MS = 20                  # kích thước chunk client (giống extension: 20 ms)

SPEECH_LABELS = ("CLEAN_SPEECH", "SPEECH_WITH_NOISE", "SPEECH_WITH_MUSIC")
NON_SPEECH_LABEL = "NO_SPEECH"


@dataclass
class Segment:
    """Đoạn test đã chuẩn bị + nhãn đã quy về mốc thời gian của đoạn."""

    wav_path: Path
    pcm: np.ndarray                       # Int16 mono 16 kHz
    labels: List[Tuple[float, float, str]]  # (start_s, end_s, label) — mốc TƯƠNG ĐỐI
    start_sec: float
    duration_sec: float

    @property
    def pcm_bytes(self) -> bytes:
        return self.pcm.tobytes()


def vad_data_dir(project_root: Path) -> Path:
    return project_root / "wav_test" / "vad"


def derived_wav_path(project_root: Path) -> Path:
    return vad_data_dir(project_root) / "derived" / f"{VIDEO_ID}_{int(SEGMENT_START_SEC)}s_{int(SEGMENT_DUR_SEC)}s_16k_mono.wav"


def source_wav_path(project_root: Path) -> Path:
    return vad_data_dir(project_root) / f"{VIDEO_ID}.wav"


def labels_csv_path(project_root: Path) -> Path:
    return vad_data_dir(project_root) / "ava_speech_labels_v1.csv"


def read_labels(
    csv_path: Path,
    video_id: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> List[Tuple[float, float, str]]:
    """Đọc nhãn của 1 video; nếu có cửa sổ thì quy về mốc TƯƠNG ĐỐI của cửa sổ đó."""
    out: List[Tuple[float, float, str]] = []
    with open(csv_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 4 or parts[0] != video_id:
                continue
            s, e, label = float(parts[1]), float(parts[2]), parts[3]
            if start_sec is not None:
                if end_sec is None or e <= start_sec or s >= end_sec:
                    continue
                s -= start_sec
                e -= start_sec
                s = max(0.0, s)
                if end_sec is not None:
                    e = min(end_sec - start_sec, e)
            out.append((s, e, label))
    out.sort()
    return out


def _build_segment(project_root: Path, dest: Path) -> None:
    """Cắt 10 phút (kênh 0, giống plugin) + resample polyphase 48k → 16k, ghi Int16 mono."""
    src = source_wav_path(project_root)
    dest.parent.mkdir(parents=True, exist_ok=True)

    import scipy.signal as sps

    with sf.SoundFile(str(src)) as fh:
        sr = fh.samplerate
        fh.seek(int(SEGMENT_START_SEC * sr))
        block = fh.read(int(SEGMENT_DUR_SEC * sr), dtype="int16", always_2d=True)

    mono = block[:, 0].astype(np.float32) / 32768.0
    resampled = sps.resample_poly(mono, TARGET_SR, sr)
    pcm = (np.clip(resampled, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    sf.write(str(dest), pcm, TARGET_SR, subtype="PCM_16")


def ensure_segment(project_root: Path, force: bool = False) -> Segment:
    """Trả đoạn test (dựng + cache nếu chưa có). Raise `FileNotFoundError` nếu thiếu nguồn."""
    src = source_wav_path(project_root)
    csv = labels_csv_path(project_root)
    if not src.exists():
        raise FileNotFoundError(f"thiếu wav AVA-Speech: {src}")
    if not csv.exists():
        raise FileNotFoundError(f"thiếu nhãn AVA-Speech: {csv}")

    dest = derived_wav_path(project_root)
    if force or not dest.exists():
        _build_segment(project_root, dest)

    pcm, sr = sf.read(str(dest), dtype="int16")
    assert sr == TARGET_SR, f"đoạn dẫn xuất phải là 16 kHz, đang là {sr}"
    if pcm.ndim > 1:
        pcm = pcm[:, 0]

    labels = read_labels(csv, VIDEO_ID, SEGMENT_START_SEC, SEGMENT_START_SEC + SEGMENT_DUR_SEC)
    return Segment(
        wav_path=dest,
        pcm=np.ascontiguousarray(pcm.astype(np.int16)),
        labels=labels,
        start_sec=SEGMENT_START_SEC,
        duration_sec=len(pcm) / TARGET_SR,
    )


# ── Ground truth & timeline ──────────────────────────────────────────────────

def labels_to_timeline(labels: Sequence[Tuple[float, float, str]], n_frames: int) -> np.ndarray:
    """Mảng bool độ phân giải `FRAME_MS` cho ground truth (nhãn SPEECH_* = True)."""
    gt = np.zeros(n_frames, dtype=bool)
    for start, end, label in labels:
        if label == NON_SPEECH_LABEL:
            continue
        a = max(0, int(round(start * 1000.0 / FRAME_MS)))
        b = min(n_frames, int(round(end * 1000.0 / FRAME_MS)))
        if b > a:
            gt[a:b] = True
    return gt


def smooth_like_vad(gt: np.ndarray, min_gap_frames: int, min_seg_frames: int) -> np.ndarray:
    """Làm mượt GT theo ĐỘ PHÂN GIẢI của VAD (VAD không thể tách khoảng lặng ngắn hơn config).

    Lấp các khoảng lặng < `min_gap_frames` và bỏ các đoạn nói < `min_seg_frames` (đơn vị
    frame `FRAME_MS`). Nhờ vậy phép so sánh không "phạt" VAD vì những chi tiết mà chính
    cấu hình của nó cấm nó biểu diễn.
    """
    out = gt.copy()

    # Bỏ đoạn nói quá ngắn
    idx = np.flatnonzero(out)
    if idx.size:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run) < min_seg_frames:
                out[run] = False

    # Lấp khoảng lặng quá ngắn (không lấp ở hai đầu)
    silence = ~out
    idx = np.flatnonzero(silence)
    if idx.size:
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if run[0] == 0 or run[-1] == len(silence) - 1:
                continue
            if len(run) < min_gap_frames:
                out[run] = True
    return out


def onsets(mask: np.ndarray) -> List[int]:
    """Chỉ số frame của các mép BẮT ĐẦU đoạn nói."""
    diff = np.diff(mask.astype(np.int8))
    return [int(i) + 1 for i in np.flatnonzero(diff == 1)]


def evaluate(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """P/R/F1/accuracy + số segment + độ trễ onset (ms)."""
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    n = max(1, int(pred.size))

    gt_on = onsets(gt)
    pr_on = onsets(pred)
    errs: List[float] = []
    for t in gt_on:
        if not pr_on:
            continue
        errs.append(min(abs(t - u) for u in pr_on) * FRAME_MS)
    median_err = float(np.median(errs)) if errs else float("nan")

    return {
        "frames": int(pred.size),
        "speech_gt_pct": round(float(gt.mean()) * 100, 2),
        "speech_pred_pct": round(float(pred.mean()) * 100, 2),
        "accuracy": round((tp + tn) / n, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gt_segments": len(gt_on),
        "pred_segments": len(pr_on),
        "onset_median_ms": round(median_err, 1) if errs else None,
        "onset_p90_ms": round(float(np.percentile(errs, 90)), 1) if errs else None,
    }


# ── Chạy VAD qua processor và dựng timeline ──────────────────────────────────

def run_engine(
    engine_name: str,
    pcm: np.ndarray,
    *,
    threshold: Optional[float] = None,
    silence_ms: Optional[int] = None,
    chunk_ms: int = CHUNK_MS,
) -> Dict[str, object]:
    """Stream PCM qua `VADStreamProcessor` và dựng timeline 10 ms của audio đã chuyển tiếp.

    Timeline được dựng từ CHÍNH các frame gửi ASR (tag `SPEECH`/`PRE_ROLL`) và timestamp
    của chúng ⇒ đo đúng thứ tự mà ASR thấy.
    """
    from backend.vad.processor import VADStreamProcessor

    n_frames = len(pcm) * 1000 // (TARGET_SR * FRAME_MS)
    pred = np.zeros(n_frames, dtype=bool)
    segments: List[Tuple[float, float]] = []
    starts: List[int] = []
    ends: List[int] = []
    # Đối chiếu NGUYÊN BẢN: mỗi frame VAD chuyển tiếp phải bằng đúng slice PCM gốc tại
    # mốc thời gian của nó (kiểm tra toàn vẹn tín hiệu trên 10 phút audio thật, O(1) RAM).
    src_bytes = pcm.tobytes()
    integrity = {"frames": 0, "mismatches": 0, "bytes": 0}

    def _on_start() -> None:
        starts.append(1)

    def _on_end() -> None:
        ends.append(1)
        if seg_start[0] is not None:
            segments.append((seg_start[0], last_end[0] or seg_start[0]))
            seg_start[0] = None

    seg_start: List[Optional[float]] = [None]
    last_end: List[Optional[float]] = [None]

    def _on_chunk(frame_bytes: bytes, ts: float, tag: str) -> None:
        a = int(round(ts * TARGET_SR))
        b = a + len(frame_bytes) // 2
        f0 = max(0, a * 1000 // (TARGET_SR * FRAME_MS))
        f1 = min(n_frames, max(f0 + 1, (b * 1000 + TARGET_SR * FRAME_MS - 1) // (TARGET_SR * FRAME_MS)))
        pred[f0:f1] = True
        if seg_start[0] is None:
            seg_start[0] = ts
        last_end[0] = ts + (b - a) / TARGET_SR

        integrity["frames"] += 1
        integrity["bytes"] += len(frame_bytes)
        byte_off = a * 2
        if src_bytes[byte_off:byte_off + len(frame_bytes)] != frame_bytes:
            integrity["mismatches"] += 1

    proc = VADStreamProcessor(
        vad_engine=engine_name,
        threshold=threshold,
        silence_duration_ms=silence_ms,
        on_speech_start=_on_start,
        on_speech_chunk=_on_chunk,
        on_speech_end=_on_end,
    )
    proc.prewarm()

    chunk_samples = int(TARGET_SR * chunk_ms / 1000)
    t0 = time.perf_counter()
    for i in range(0, len(pcm), chunk_samples):
        chunk = pcm[i:i + chunk_samples]
        proc.feed_chunk(chunk.tobytes(), capture_timestamp=i / TARGET_SR)
    elapsed = time.perf_counter() - t0

    if seg_start[0] is not None and last_end[0] is not None:
        segments.append((seg_start[0], last_end[0]))

    return {
        "engine": engine_name,
        "pred": pred,
        "starts": len(starts),
        "ends": len(ends),
        "segments": segments,
        "elapsed_sec": elapsed,
        "rtf": elapsed / max(1e-9, len(pcm) / TARGET_SR),
        "hop_samples": proc._frame_samples,
        "pre_roll_frames": proc._max_lookback_frames,
        "integrity": integrity,
    }
