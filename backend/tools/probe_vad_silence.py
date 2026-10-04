"""Kiểm chứng bộ dò khoảng lặng (Silero VAD, chạy theo lô) trên audio THẬT.

In ra khoảng lặng VAD xác nhận, chi phí thời gian, và khoảng lặng dài nhất trong các cửa sổ tìm
kiếm điển hình của bộ cắt khối.

Cách dùng:
    .venv\\Scripts\\python.exe -m backend.tools.probe_vad_silence --start 0 --end 180
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.vad_silence import VADSilenceScanner  # noqa: E402
from backend.tools.diag_pipeline_b import load_audio_16k  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="wav_test/youtube/The_Big_Bang_Theory.mp3")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=180.0)
    ap.add_argument("--engine", default="silero-vad")
    ap.add_argument("--min-silence-ms", type=float, default=1500.0)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--windows", nargs="*", default=None,
                    help="Các cặp 'lo,hi' mô phỏng vùng tìm kiếm của bộ cắt khối.")
    args = ap.parse_args()

    pcm = load_audio_16k(REPO_ROOT / args.audio, start=args.start, end=args.end)
    dur = len(pcm) / 16000.0
    print(f"Audio [{args.start:.2f} → {args.end:.2f}]s = {dur:.2f}s")

    scanner = VADSilenceScanner(
        engine_name=args.engine,
        min_silence_ms=args.min_silence_ms,
        threshold=args.threshold,
    )
    t0 = time.perf_counter()
    scanner.prewarm()
    print(f"prewarm: {(time.perf_counter() - t0) * 1000:.0f}ms "
          f"(VAD dùng được: {not scanner.vad_unavailable_reason})")

    t0 = time.perf_counter()
    gaps = scanner.find_silence_gaps(pcm, args.start)
    wall = (time.perf_counter() - t0) * 1000.0
    print(f"quét {dur:.1f}s: wall {wall:.0f}ms (VAD {scanner.last_scan_ms:.0f}ms) "
          f"→ {len(gaps)} khoảng lặng >= {args.min_silence_ms:.0f}ms")
    for g in gaps:
        print(f"   ✓ [{g.start_pts:8.3f} → {g.end_pts:8.3f}]  {g.duration:6.2f}s")

    windows = args.windows or ["10,28", "32,48", "48,64", "62,78", "74,90"]
    print("\nMô phỏng vùng tìm kiếm của bộ cắt khối (chọn khoảng lặng CUỐI CÙNG >= ngưỡng):")
    for spec in windows:
        lo, hi = (float(x) for x in spec.split(","))
        sub = pcm[int(max(0.0, lo - args.start) * 16000):]
        t0 = time.perf_counter()
        best = scanner.find_last_silence(
            sub, base_pts=max(args.start, lo), region_start_pts=lo, region_end_pts=hi
        )
        wall = (time.perf_counter() - t0) * 1000.0
        print(f"   [{lo:5.1f},{hi:5.1f}] → {best}   ({wall:.0f}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
