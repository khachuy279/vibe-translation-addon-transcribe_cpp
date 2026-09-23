"""Harness đo độ trễ streaming THẬT (tầng B) — P0.2/T0.2.

Khác với `test_07_e2e_comparison.py` (E2E 3 model, chậm), harness này:
- Chỉ chạy VAD + ASR thật (KHÔNG nạp translation/TTS) => nhanh, tập trung vào hot path.
- PACING theo thời gian thực (có `await`) nên coroutine preview THẬT SỰ chạy.
- In ra số đo p50/p95 cho `asr.preview_ms`, `asr.commit_ms`, `asr.preview_audio_sec`,
  `vad.chunk_ms`, cùng các counter drop/merge.

Cách dùng:
    python backend/tests/test_08_streaming_latency.py --model qwen3-asr-0.6b --speed 4
    python backend/tests/test_08_streaming_latency.py --wav wav_test/English_low_speech_quality_19s.wav
    TRANSCRIBE_PERF_DEBUG=1 python backend/tests/test_08_streaming_latency.py   # số native

Không được pytest thu thập (không có hàm `test_*`) để bộ test mặc định vẫn nhanh.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# F-40: VAD (model torch trên CPU) phải chạy TRÊN LUỒNG WORKER, không phải event loop —
# đúng như app thật (`backend/ws/handler.py` dùng `_VAD_EXECUTOR`). Nếu chạy ngay trong
# event loop, một lần `feed_chunk` bị chậm/treo sẽ CHẶN ĐỨNG cả loop: mọi đồng hồ
# watchdog của asyncio ngừng hoạt động và tiến trình treo im lặng (đã bắt gặp: log dừng
# ở dòng VAD, không hàng rào Python nào phát hiện — xem báo cáo §13.5).
_VAD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench_vad")

# Thứ tự nạp DLL quan trọng trên Windows (xem backend/tests/conftest.py)
from backend.utils.cuda import setup_cuda_dll_paths

setup_cuda_dll_paths()
try:
    # Bootstrap TRƯỚC để `TRANSCRIBE_LIBRARY` trỏ vào bundle `backend/bin/` (CUDA); nếu không,
    # native của wheel (Vulkan) đã vào sys.modules và backend/bin/ không áp được.
    from backend.asr import native as _asr_native

    _asr_native.bootstrap()
    import transcribe_cpp  # noqa: F401
except Exception:
    pass

from backend.config import config
from backend.core.metrics import metrics_collector

# Mock WebSocket (không cần mạng)
from backend.tests.conftest import MockWebSocket  # noqa: E402


def load_wav_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        import scipy.signal

        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def make_continuous(audio: np.ndarray, target_sec: float, frame: int = 400, thresh: float = 0.02) -> np.ndarray:
    """Ghép các đoạn "có tiếng" thành audio nói LIÊN TỤC (bỏ khoảng lặng).

    Cần thiết vì `wav_test/` không có file nói liên tục dài, nên bài toán O(N²) của
    vòng lặp preview không bao giờ được kích hoạt.
    """
    n = len(audio) // frame * frame
    frames = audio[:n].reshape(-1, frame)
    rms = np.sqrt((frames.astype(np.float32) ** 2).mean(axis=1))
    voiced = frames[rms > thresh]
    if len(voiced) == 0:
        voiced = frames
    need = int(target_sec * 16000 / frame) + 1
    reps = int(np.ceil(need / max(1, len(voiced))))
    out = np.tile(voiced, (reps, 1)).reshape(-1)[: int(target_sec * 16000)]
    return np.ascontiguousarray(out, dtype=np.float32)


async def run_paced(
    audio: np.ndarray,
    model_key: str,
    speed: float,
    poll_interval_ms: int,
    chunk_samples: int = 400,
    tail_silence_sec: float = 1.2,
    preload_first: bool = True,
):
    """Chạy VAD + ASR thật với pacing thời gian thực."""
    from backend.ws.connection import SafeWebSocketConnection
    from backend.ws.handler import _stream_asr_tokens
    from backend.ws.session import SessionState

    config.asr.active_model = model_key
    config.asr.poll_interval_ms = poll_interval_ms

    ws = MockWebSocket()
    session = SessionState(SafeWebSocketConnection(ws))

    from backend.asr.engine import TranscribeEngine
    from backend.asr.registry import ModelRegistry
    from backend.vad.processor import VADProcessor

    registry = ModelRegistry.get_instance()
    registry.set_active_model_key(model_key)
    engine = TranscribeEngine(model_key=model_key, session_id="bench")
    session.asr_engine = engine
    session.vad_processor = VADProcessor(
        sample_rate=config.vad.sample_rate,
        vad_engine=config.vad.vad_engine,
        threshold=config.vad.effective_threshold,
        silence_duration_ms=config.vad.effective_silence_ms,
        enabled=True,
        on_speech_chunk=engine.feed_audio,
        on_speech_start=engine.on_speech_start,
        on_speech_end=engine.on_speech_end,
    )

    asr_task = asyncio.create_task(_stream_asr_tokens(session))

    # Preload + warm-up TRƯỚC khi feed, để phản ánh app thật (backend prewarm lúc khởi
    # động nên model đã ấm). Nếu không, warm-up (~7-10s) sẽ chồng lên lúc feed và làm
    # mất hết preview.
    if preload_first:
        loop = asyncio.get_running_loop()
        t_pre = time.perf_counter()
        await loop.run_in_executor(None, engine._preload_model)
        print(f"   [preload+warmup] {time.perf_counter() - t_pre:.1f}s (không tính vào số đo)")

    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    payload = pcm16 + (b"\x00\x00" * int(tail_silence_sec * 16000))
    total_chunks = len(payload) // (chunk_samples * 2)

    step = (chunk_samples / 16000.0) / max(0.01, speed)
    next_deadline = time.perf_counter()
    t_start = time.perf_counter()
    loop = asyncio.get_running_loop()

    for idx in range(total_chunks):
        chunk = payload[idx * chunk_samples * 2 : (idx + 1) * chunk_samples * 2]
        # F-40: đẩy VAD sang luồng worker (giống app) để KHÔNG chặn event loop.
        await loop.run_in_executor(
            _VAD_EXECUTOR,
            session.vad_processor.feed_chunk,
            chunk,
            idx * (chunk_samples / 16000.0),
        )
        next_deadline += step
        delay = next_deadline - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)

    # Đợi pipeline xử lý hết. Dùng `has_pending_work()` chứ KHÔNG chỉ kiểm tra
    # `_pending_commits` — commit bị pop ra ngay khi bắt đầu inference, nên deque rỗng
    # chưa có nghĩa là đã xử lý xong (đã từng khiến harness cắt ngang inference).
    for _ in range(600):
        await asyncio.sleep(0.05)
        if not engine.has_pending_work():
            break

    wall = time.perf_counter() - t_start
    asr_task.cancel()
    await asyncio.gather(asr_task, return_exceptions=True)
    # F-39: huỷ task asyncio KHÔNG dừng được lời gọi native đang chạy trong executor.
    # Nếu không huỷ/chờ native xong, lượt `run_paced` sau sẽ nộp thêm việc trong lúc
    # native cũ vẫn giữ `_infer_lock` ⇒ hàng đợi executor phình vô hạn (đã đo được
    # +66-85 MB/s khi chạy harness WER đủ 8 file × 3 cấu hình × 3 lần lặp).
    try:
        await asyncio.wait_for(engine.cancel_inference(), timeout=2.0)
    except Exception:  # noqa: BLE001
        pass
    for _ in range(120):
        if not engine.has_pending_work():
            break
        await asyncio.sleep(0.05)
    await session.cleanup()
    return wall, ws.sent_messages


def _fmt(stats) -> str:
    if not stats or not stats.get("count"):
        return "      –"
    return (f"n={stats['count']:<4} p50={stats['p50_ms']:>7.1f} p95={stats['p95_ms']:>7.1f} "
            f"max={stats['max_ms']:>7.1f}")


def main() -> int:
    # RÀO CHẮN RAM CỨNG: lỗi phình bộ nhớ nằm trong native (báo cáo §12.6) và có lúc đã
    # đẩy tiến trình lên 47 GB. Trần mặc định 4000 MB, đổi bằng `MEM_GUARD_MB`; 0 = tắt.
    from backend.utils.mem_guard import start_guard

    start_guard()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-asr-0.6b", help="model key trong backend/models.yaml")
    ap.add_argument("--wav", default=None, help="file wav nguồn (mặc định lấy từ wav_test/)")
    ap.add_argument("--seconds", type=float, default=8.0, help="độ dài audio nói liên tục")
    ap.add_argument("--speed", type=float, default=1.0, help="hệ số tốc độ feed (1.0 = realtime)")
    ap.add_argument("--poll", type=int, default=None, help="poll_interval_ms (mặc định theo config)")
    ap.add_argument("--json", default=None, help="ghi kết quả ra file JSON")
    args = ap.parse_args()

    wav = Path(args.wav) if args.wav else (_ROOT / "wav_test" / "English_low_speech_quality_19s.wav")
    if not wav.exists():
        print(f"❌ Không thấy file: {wav}")
        return 2

    metrics_collector.reset()
    audio = load_wav_16k(wav)
    if args.seconds > 0:
        audio = make_continuous(audio, args.seconds)

    poll = args.poll or config.asr.poll_interval_ms
    print("=" * 78)
    print("ĐO ĐỘ TRỄ STREAMING THẬT (tầng B)")
    print("=" * 78)
    print(f"model        : {args.model}")
    print(f"audio        : {len(audio) / 16000:.2f}s nói liên tục (nguồn {wav.name})")
    print(f"speed        : {args.speed}x   poll_interval={poll}ms")
    print(f"preview_win  : {config.asr.preview_window_sec}s  max_duration={config.sentence.max_duration_sec}s")
    print(f"vad          : {config.vad.vad_engine}  "
          f"silence={config.vad.effective_silence_ms if config.vad.effective_silence_ms else 'mặc định engine'}ms")
    print(f"TRANSCRIBE_PERF_DEBUG={'1' if os.environ.get('TRANSCRIBE_PERF_DEBUG') else '0'}")
    print("-" * 78)

    t_load = time.perf_counter()
    wall, msgs = asyncio.run(run_paced(audio, args.model, args.speed, poll))
    total = time.perf_counter() - t_load

    previews = [m for m in msgs if m.get("type") == "utterance_update" and not m.get("is_final")]
    finals = [m for m in msgs if m.get("type") == "utterance_update" and m.get("is_final")]

    print(f"Thời gian chạy (gồm nạp model): {total:.1f}s | feed {wall:.1f}s")
    print(f"Preview gửi ra   : {len(previews)}")
    print(f"Câu chốt (final) : {len(finals)}")
    if finals:
        reasons = {}
        for m in finals:
            r = m.get("commit_reason", "?")
            reasons[r] = reasons.get(r, 0) + 1
        print(f"Lý do chốt câu   : {reasons}")
        for m in finals[:5]:
            print(f"   [{m.get('commit_reason')}] {m.get('text','')[:80]}")
    print("-" * 78)

    report = metrics_collector.generate_report()
    stages = report["stages"]
    print("SỐ ĐO HOT PATH")
    for key in ("asr.preview_ms", "asr.commit_ms", "asr.first_preview_ms", "asr.e2e_commit_ms",
                "asr.preview_audio_sec",
                "asr.commit_audio_sec", "asr.slice_ms", "asr.normalize_ms",
                "vad.chunk_ms", "ws.send_ms", "session.cleanup_ms"):
        print(f"  {key:26} {_fmt(stages.get(key))}")

    print("-" * 78)
    interesting = {k: v for k, v in report["counters"].items()
                   if any(t in k for t in ("drop", "merge", "carry", "trim", "trunc",
                                           "skip", "reason", "gap", "cancel"))}
    print(f"COUNTER: {json.dumps(interesting, ensure_ascii=False)}")
    print(f"GAUGE  : {json.dumps(report.get('gauges', {}), ensure_ascii=False)}")

    # K3: hệ số lặp audio (preview) — mục tiêu < 2x SAU KHI áp dụng cửa sổ
    prev = stages.get("asr.preview_audio_sec", {})
    shown = stages.get("asr.preview_ms", {})
    if prev.get("count"):
        total_preview_audio = prev["avg_ms"] * prev["count"]
        print(f"\nK3: tổng audio đã xử lý cho preview = {total_preview_audio:.1f}s "
              f"cho {len(audio)/16000:.1f}s audio => hệ số {total_preview_audio/(len(audio)/16000):.1f}x")
    if shown.get("count"):
        print(f"K1: p95 asr.preview_ms = {shown['p95_ms']:.0f} ms (mục tiêu < 250 ms)")
    first = stages.get("asr.first_preview_ms", {})
    if first.get("count"):
        print(f"K2: preview đầu tiên sau khi bắt đầu nói: p50 {first['avg_ms']:.0f} ms, "
              f"max {first['max_ms']:.0f} ms (mục tiêu < 500 ms)")
    e2e = stages.get("asr.e2e_commit_ms", {})
    if e2e.get("count"):
        print(f"K4: E2E ngừng nói -> phụ đề gốc chốt: p50 {e2e['avg_ms']:.0f} ms, "
              f"max {e2e['max_ms']:.0f} ms (mục tiêu p95 < 1200 ms)")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "model": args.model, "seconds": len(audio) / 16000, "speed": args.speed,
            "poll_interval_ms": poll, "wall_sec": wall,
            "previews": len(previews), "finals": len(finals),
            "report": report,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã ghi JSON: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
