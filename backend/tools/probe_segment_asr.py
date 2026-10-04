"""Nghe thử (bằng ASR) một đoạn audio cụ thể để biết chỗ đó THẬT SỰ có tiếng nói hay không.

Dùng khi cần phân xử: "aligner đặt từ vào đoạn 3.0–8.0 s là sai, hay đoạn đó thật sự có tiếng
nói mà tai người không tách được?".

Cách dùng:
    .venv\\Scripts\\python.exe -m backend.tools.probe_segment_asr --start 2.9 --end 8.6
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
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    args = ap.parse_args()

    from backend.asr.engine import TranscribeEngine

    pcm = load_audio_16k(REPO_ROOT / args.audio, start=args.start, end=args.end)
    rms = float(np.sqrt(np.mean(np.asarray(pcm, dtype=np.float32) ** 2))) if pcm.size else 0.0
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    print(f"Đoạn [{args.start:.2f}–{args.end:.2f}]s: {len(pcm) / 16000.0:.2f}s, "
          f"RMS={rms:.5f}, peak={peak:.5f}")

    eng = TranscribeEngine(session_id="probe_seg")
    print("ASR:", repr(eng.transcribe_block(pcm, language="en")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
