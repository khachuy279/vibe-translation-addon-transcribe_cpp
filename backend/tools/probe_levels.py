"""Đo mức năng lượng (RMS) theo khung để biết vùng nào là TIẾNG NÓI và vùng nào là nhạc/tiếng cười.

Dùng để chẩn đoán vì sao Forced Aligner đặt mốc câu quá sớm: khối batch bắt đầu ở một đoạn
KHÔNG CÓ TIẾNG NÓI (tiếng cười, nhạc nền, tiếng thở) nhưng vẫn có chữ trong bản phiên âm, nên
model phải "nhét" chữ vào khoảng đầu đó.

Cách dùng:
    .venv\\Scripts\\python.exe -m backend.tools.probe_levels --only 0 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.tools.diag_pipeline_b import load_audio_16k  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="wav_test/youtube/The_Big_Bang_Theory.mp3")
    ap.add_argument("--only", nargs=2, type=float, required=True)
    ap.add_argument("--frame-ms", type=float, default=20.0)
    ap.add_argument("--rel-db", type=float, default=22.0)
    args = ap.parse_args()

    start, end = args.only
    pcm = load_audio_16k(REPO_ROOT / args.audio, start=start, end=end)
    if pcm.size == 0:
        print("Không có audio")
        return 2

    sr = 16000
    frame = int(round(args.frame_ms * sr / 1000.0))
    n = len(pcm) // frame
    arr = pcm[: n * frame].reshape(n, frame)
    rms = np.sqrt((arr ** 2).mean(axis=1))

    p90 = float(np.percentile(rms, 90))
    thr = max(0.004, p90 * (10 ** (-args.rel_db / 20.0)))
    print(f"p90(RMS)={p90:.5f}  ngưỡng lặng={thr:.5f}  khung={args.frame_ms:.0f}ms  "
          f"dải=[{start:.2f}s → {start + len(pcm) / sr:.2f}s]")

    # In dạng thanh để nhìn nhanh
    bar_chars = " .:-=+*#%@"
    peak = float(rms.max()) or 1.0
    print("mốc(s)  RMS      mức")
    for i in range(n):
        t = start + i * args.frame_ms / 1000.0
        lvl = int(min(9, rms[i] / peak * 9))
        flag = "SILENT" if rms[i] < thr else ""
        if flag or i % 1 == 0:
            print(f"{t:7.2f}  {rms[i]:.5f}  {bar_chars[lvl] * 6:<8} {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
