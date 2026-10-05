"""Chẩn đoán Pipeline B trên file audio THẬT (wav_test/youtube/*.mp3).

Chạy ĐÚNG chuỗi của `LookaheadSessionState._offline_batch_loop` nhưng không cần WebSocket/Extension:

    MP3 ──► PCM 16k mono ──► ContinuousAudioTimeline
        ──► LookaheadChunker.next_chunk   (cắt tại khoảng lặng do VAD xác nhận)
        ──► TranscribeEngine.transcribe_block   (Qwen3-ASR)
        ──► ForcedAlignerService.align          (Qwen3-ForcedAligner)
        ──► group_words_to_subtitles + split_oversized_sentences
        ──► in bảng mốc tuyệt đối của từng phụ đề

GHI CHÚ THIẾT KẾ (2026-10-04): tuyến này **không** hiệu chỉnh mốc từ, **không** trừ chồng lấn, và
**không** hàn mảnh cuối — mốc aligner được dùng nguyên vẹn vì khối đã được cắt tại khoảng lặng thật.

Cách dùng:
    .venv\\Scripts\\python.exe -m backend.tools.diag_pipeline_b
    .venv\\Scripts\\python.exe -m backend.tools.diag_pipeline_b --only 60 90 --max-chunks 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.asr.forced_aligner import (  # noqa: E402
    AlignedWord,
    ForcedAlignerService,
    SubtitleSentence,
    resolve_aligner_language,
)
from backend.config import config  # noqa: E402
from backend.core.lookahead_chunker import LookaheadChunker  # noqa: E402
from backend.core.lookahead_timeline import ContinuousAudioTimeline  # noqa: E402
from backend.core.vad_silence import VADSilenceScanner  # noqa: E402


# ─────────────────────────────────────────────────────────────── audio

def load_audio_16k(path: Path, start: float = 0.0, end: Optional[float] = None) -> np.ndarray:
    """Giải mã audio về PCM float32 16 kHz mono (dùng PyAV, cùng thư viện với demuxer)."""
    import av

    container = av.open(str(path))
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    pieces: List[np.ndarray] = []
    try:
        container.seek(int(start * av.time_base), any_frame=False, backward=True)
    except Exception:
        pass
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
        if t < start - 0.5:
            continue
        if end is not None and t > end:
            break
        for out in resampler.resample(frame):
            pieces.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):
        pieces.append(out.to_ndarray().reshape(-1))
    container.close()
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32)


# ─────────────────────────────────────────────────── ASR + aligner

class PipelineBProbe:
    """Chạy tuyến OFFLINE_BATCH thật (ASR + ForcedAligner + cắt khối bằng VAD)."""

    def __init__(self, use_asr: bool = True, use_aligner: bool = True):
        self.la = config.lookahead
        self.timeline = ContinuousAudioTimeline(sample_rate=16000, max_pending_sec=90.0)
        self.scanner = VADSilenceScanner(
            engine_name="silero-vad",
            min_silence_ms=float(self.la.batch_vad_silence_ms),
            threshold=float(self.la.batch_vad_silence_threshold),
        )
        self.chunker = LookaheadChunker(
            timeline=self.timeline,
            sample_rate=16000,
            target_window_sec=float(self.la.batch_target_sec),
            min_window_sec=float(self.la.batch_min_sec),
            min_silence_ms=float(self.la.batch_min_silence_ms),
            strong_silence_ms=float(self.la.batch_strong_silence_ms),
            silence_rel_db=float(self.la.batch_silence_rel_db),
            silence_floor_rms=float(self.la.batch_silence_floor_rms),
            min_rms_ratio=float(self.la.batch_min_rms_ratio),
            min_rms_hold_ms=float(self.la.batch_min_rms_hold_ms),
            dip_search_sec=float(self.la.batch_dip_search_sec),
            silence_scanner=self.scanner,
            vad_silence_ms=float(self.la.batch_vad_silence_ms),
            max_audio_sec=float(self.la.batch_max_audio_sec),
        )
        self.sub_opts: Dict[str, Any] = {
            "max_words": int(self.la.batch_sub_max_words),
            "max_duration_sec": float(self.la.batch_sub_max_duration_sec),
            "max_chars": int(self.la.batch_sub_max_chars),
            "min_words": max(1, int(self.la.batch_sub_min_words)),
            "defer_words": max(0, int(self.la.batch_sub_defer_words)),
            "sentence_max_words": int(self.la.batch_sub_sentence_max_words),
            "sentence_max_duration_sec": float(self.la.batch_sub_sentence_max_duration_sec),
            "sentence_max_chars": int(self.la.batch_sub_sentence_max_chars),
        }
        self.asr = None
        self.aligner = None
        if use_asr:
            from backend.asr.engine import TranscribeEngine

            self.asr = TranscribeEngine(session_id="diag_pb")
        if use_aligner:
            self.aligner = ForcedAlignerService.get_instance()

    def prewarm(self) -> None:
        self.scanner.prewarm()
        if self.aligner is not None:
            self.aligner.prewarm()

    def transcribe(self, pcm: np.ndarray) -> str:
        if self.asr is None:
            return ""
        return self.asr.transcribe_block(pcm, language=None)

    def align(self, pcm: np.ndarray, text: str) -> Tuple[List[AlignedWord], str]:
        if self.aligner is None or not text.strip():
            return [], "English"
        lang = resolve_aligner_language("en") or "English"
        return self.aligner.align(pcm, text, lang), lang

    def group(self, words: List[AlignedWord], lang: str) -> List[SubtitleSentence]:
        if self.aligner is None:
            return []
        subs = self.aligner.group_words_to_subtitles(words, language=lang, **self.sub_opts)
        return ForcedAlignerService.split_oversized_sentences(
            subs,
            language=lang,
            max_chars=int(self.sub_opts["sentence_max_chars"]),
            max_words=int(self.sub_opts["sentence_max_words"]),
        )


# ─────────────────────────────────────────────────────────────── main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="wav_test/youtube/The_Big_Bang_Theory.mp3")
    ap.add_argument("--only", nargs=2, type=float, default=None,
                    help="Chỉ nạp dải audio [start end] (giây) — nhanh hơn nhiều.")
    ap.add_argument("--no-asr", action="store_true", help="Bỏ qua ASR (dùng --text theo thứ tự khối).")
    ap.add_argument("--text", nargs="*", default=None, help="Text ASR cho từng khối (với --no-asr).")
    ap.add_argument("--max-chunks", type=int, default=8)
    args = ap.parse_args()

    audio_path = REPO_ROOT / args.audio
    if not audio_path.exists():
        print(f"Không thấy file audio: {audio_path}")
        return 2

    start = args.only[0] if args.only else 0.0
    end = args.only[1] if args.only else None

    t0 = time.perf_counter()
    pcm = load_audio_16k(audio_path, start=start, end=end)
    print(f"[diag] Nạp {len(pcm) / 16000.0:.2f}s audio từ {audio_path.name} trong "
          f"{time.perf_counter() - t0:.2f}s (dải {start:.2f}s → {end if end else 'hết'})")
    if pcm.size == 0:
        print("[diag] Không giải mã được audio.")
        return 2

    probe = PipelineBProbe(use_asr=not args.no_asr, use_aligner=True)
    probe.timeline.append(start, pcm)
    t0 = time.perf_counter()
    probe.prewarm()
    print(f"[diag] prewarm (VAD + Aligner): {time.perf_counter() - t0:.2f}s")

    from_pts = start
    texts = list(args.text or [])

    for ci in range(args.max_chunks):
        chunk = probe.chunker.next_chunk(from_pts=from_pts)
        if chunk is None:
            print(f"[diag] Hết audio để cắt khối (from_pts={from_pts:.2f}s).")
            break

        print("\n" + "=" * 110)
        print(f"KHỐI #{ci}  [{chunk.pts_start:.2f}s → {chunk.pts_end:.2f}s]  "
              f"dur={chunk.duration:.2f}s  mode={chunk.fallback_mode}  "
              f"silence={chunk.silence_gap_ms:.0f}ms  strength='{chunk.boundary_strength}'  "
              f"target={probe.chunker.last_effective_target_sec:.1f}s  "
              f"search_max={probe.chunker.last_effective_search_max_sec:.1f}s  "
              f"VAD_scan={probe.scanner.last_scan_ms:.0f}ms  "
              f"next_read={chunk.next_read_pts:.2f}s")
        print("=" * 110)

        if args.no_asr and ci < len(texts):
            text = texts[ci]
        else:
            text = probe.transcribe(chunk.pcm)
        print(f"ASR: {text!r}")
        if not text.strip():
            from_pts = chunk.next_read_pts
            continue

        words, lang = probe.align(chunk.pcm, text)
        print(f"Aligner ({lang}): {len(words)} từ")
        if words:
            print("   từ (mốc TUYỆT ĐỐI trong khối):")
            for w in words:
                print(f"     {chunk.pts_start + w.start_time:8.3f} → {chunk.pts_start + w.end_time:8.3f}  "
                      f"{w.text!r}")

        # KHÔNG hiệu chỉnh mốc, KHÔNG trừ chồng lấn, KHÔNG hàn mảnh cuối: mốc aligner dùng nguyên vẹn.
        subs = probe.group(words, lang)
        print(f"→ {len(subs)} phụ đề:")
        for s in subs:
            abs_start = chunk.pts_start + s.start_time
            abs_end = chunk.pts_start + s.end_time
            print(f"   [{abs_start:8.3f}s → {abs_end:8.3f}s] ({abs_end - abs_start:5.2f}s) "
                  f"{len(s.words):2d} từ  {s.text!r}")

        from_pts = chunk.next_read_pts

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
