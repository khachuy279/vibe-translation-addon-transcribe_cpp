"""Fixture dữ liệu tiếng Nhật cho tầng SEG (Google FLEURS `ja_jp`).

Mục đích: có một đoạn audio ~60 giây được ghép từ các câu FLEURS (mỗi câu cách nhau
một khoảng lặng), kèm **bản tham chiếu dấu câu** lấy từ `test.tsv`. Nhờ vậy tầng SEG
được kiểm tra đúng tiêu chí: "tách câu đúng theo dấu câu trong test.tsv".

`test.tsv` (không header) gồm 7 cột:
    id, filename, raw_text (CÓ dấu câu), normalized_text, char_phrases, samples, gender
"""

from dataclasses import dataclass
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

FLEURS_REL = Path("wav_test") / "google_fleurs" / "ja_jp"
SAMPLE_RATE = 16000

#: Dấu kết câu dùng để dựng BẢN THAM CHIẾU (độc lập với `backend.segmentation`).
_REF_SPLIT = re.compile(r"(?<=[。！？])")


@dataclass
class FleursRow:
    row_id: str
    filename: str
    raw_text: str
    samples: int

    @property
    def wav_path(self) -> Path:
        raise NotImplementedError  # gán qua `wav_dir` khi dùng


def fleurs_dir(project_root: Path) -> Path:
    return Path(project_root) / FLEURS_REL


def has_data(project_root: Path) -> bool:
    d = fleurs_dir(project_root)
    return (d / "test.tsv").exists() and (d / "test").is_dir()


def load_rows(project_root: Path) -> List[dict]:
    """Đọc `test.tsv` (bỏ qua dòng nào thiếu cột)."""
    tsv = fleurs_dir(project_root) / "test.tsv"
    rows: List[dict] = []
    with tsv.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            # Cột 5 = số mẫu @16 kHz. Cột 0 cũng toàn chữ số (id) nên KHÔNG được
            # `next(... if p.isdigit())` — sẽ lấy nhầm id.
            samples = int(parts[5]) if parts[5].strip().isdigit() else 0
            if samples <= 0:
                samples = max((int(p) for p in parts if p.strip().isdigit()), default=0)
            rows.append({
                "row_id": parts[0],
                "filename": parts[1],
                "raw_text": parts[2].strip(),
                "samples": samples,
            })
    return rows


def reference_sentences(text: str) -> List[str]:
    """Bản tham chiếu: tách theo dấu kết câu (độc lập với code đang được test)."""
    return [s.strip() for s in _REF_SPLIT.split(text) if s.strip()]


def pick_rows(
    project_root: Path,
    *,
    target_sec: float = 60.0,
    require_final_punct: bool = True,
) -> List[dict]:
    """Chọn các câu FLEURS cho tổng độ dài ~`target_sec` giây.

    Chỉ lấy câu KẾT THÚC bằng dấu câu để ranh giới giữa hai file (nơi ta chèn khoảng
    lặng) trùng khớp với ranh giới câu trong bản tham chiếu.
    """
    picked: List[dict] = []
    total = 0.0
    for row in load_rows(project_root):
        text = row["raw_text"]
        if not text:
            continue
        if require_final_punct and text[-1] not in "。！？":
            continue
        if row["samples"] <= 0:
            continue
        picked.append(row)
        total += row["samples"] / SAMPLE_RATE
        if total >= target_sec:
            break
    return picked


def concat_text(rows: Sequence[dict]) -> str:
    return "".join(r["raw_text"] for r in rows)


def stream_incremental(text: str, chunk: int = 3):
    """Sinh preview TĂNG DẦN giống ASR streaming: cắt 1..N ký tự mỗi bước."""
    step = max(1, int(chunk))
    for end in range(step, len(text) + step, step):
        yield text[:min(end, len(text))]


def build_concat_wav(
    project_root: Path,
    rows: Sequence[dict],
    out_path: Optional[Path] = None,
    *,
    gap_ms: float = 500.0,
) -> Tuple[Path, np.ndarray, List[Tuple[int, int]]]:
    """Ghép các wav FLEURS thành 1 file ~1 phút, chèn khoảng lặng giữa các câu.

    Trả `(đường_dẫn, pcm_float32, danh_sách_(start_sample, end_sample) mỗi câu)`.
    """
    import soundfile as sf

    wav_dir = fleurs_dir(project_root) / "test"
    gap = np.zeros(int(SAMPLE_RATE * gap_ms / 1000.0), dtype=np.float32)
    pieces: List[np.ndarray] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0

    for row in rows:
        path = wav_dir / row["filename"]
        if not path.exists():
            continue
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            # Nội suy tuyến tính — đủ tốt cho fixture nghe thử, không dùng cho benchmark.
            n_out = int(len(data) * SAMPLE_RATE / sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, n_out, dtype=np.float64),
                np.arange(len(data), dtype=np.float64),
                data.astype(np.float64),
            ).astype(np.float32)
        pieces.append(data)
        spans.append((cursor, cursor + len(data)))
        cursor += len(data)
        pieces.append(gap)
        cursor += len(gap)

    pcm = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if out_path is None:
        out_path = fleurs_dir(project_root) / "ja_concat_1min.wav"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm, SAMPLE_RATE, subtype="PCM_16")
    return out_path, pcm, spans
