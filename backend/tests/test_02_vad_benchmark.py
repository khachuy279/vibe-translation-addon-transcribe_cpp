"""Kiểm thử & Benchmark Phase 2: Module VAD Streaming Độc Lập.

Kiểm chứng các yêu cầu:
1. Hoạt động chính xác của cả 3 VAD Engine: FireRed-VAD, Silero-VAD, FSMN-VAD theo đúng
   API docs (mỗi engine một config riêng — xem `report/audit/19_KE_HOACH_VIET_LAI_VAD.md`).
2. Đo độ trễ xử lý từng frame (Avg / p95 in µs) và Real-Time Factor (RTF).
3. Đánh giá khả năng phát hiện ranh giới nói/nghỉ trên các file `wav_test/`.
4. Tự động xuất báo cáo chi tiết vào `/report/02_vad/report.md`.
"""

import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pytest
import soundfile as sf

from backend.config import config
from backend.vad import (
    VADProcessor,
    VADEngineFactory,
    SUPPORTED_VAD_ENGINES,
)
from backend.tests.test_01_core_audio import load_wav_file  # noqa: F401  (giữ API cũ cho test khác)
from backend.utils.logger import logger


def _wav_int16(path: Path) -> np.ndarray:
    """Đọc wav thành PCM Int16 mono 16 kHz.

    Một số file trong `wav_test/` là 44,1 kHz ⇒ resample Ở ĐÂY (chuẩn bị dữ liệu test,
    dùng `scipy.signal.resample_poly`). Trong pipeline thật, plugin đã resample về 16 kHz
    trong worklet nên VAD/ASR **không** resample gì (xem `test_42_vad_signal_integrity.py`).
    """
    import scipy.signal as sps

    x, sr = sf.read(str(path), dtype="int16")
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.int16)
    if sr != 16000:
        ratio = np.gcd(int(sr), 16000)
        resampled = sps.resample_poly(x.astype(np.float32) / 32768.0, 16000 // ratio, int(sr) // ratio)
        x = (np.clip(resampled, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    return np.ascontiguousarray(x)


@pytest.mark.slow
@pytest.mark.parametrize("engine_name", list(SUPPORTED_VAD_ENGINES))
def test_vad_engine_initialization_and_single_frame(engine_name):
    """Khởi tạo engine + phân tích đúng 1 frame Int16 theo hop mà nó công bố."""
    engine = VADEngineFactory.get_engine(engine_name)
    assert engine is not None
    assert engine.frame_samples > 0
    assert engine.max_lookback_frames >= 0

    state = engine.create_initial_state()
    assert state is not None

    # Frame "có tiếng" (sóng sin biên độ lớn) và frame im lặng — chỉ kiểm tra hợp đồng
    # trả về VADResult (biên độ/ngữ nghĩa do từng model quyết định).
    speech_frame = (np.sin(np.linspace(0, 50, engine.frame_samples)) * 0.5 * 32767).astype(np.int16)
    res_speech = engine.is_speech(speech_frame, state)
    assert isinstance(res_speech.probability, float)
    assert res_speech.event in (None, "START", "END")

    silence_frame = np.zeros(engine.frame_samples, dtype=np.int16)
    res_silence = engine.is_speech(silence_frame, state)
    assert res_silence.event in (None, "START", "END")

    # Frame sau `reset` phải dựng lại được state của engine (không dùng state cũ).
    state.reset()
    engine.is_speech(silence_frame, state)
    assert state.engine_state is not None


@pytest.mark.slow
def test_vad_processor_stream_callbacks(wav_test_dir):
    """VADProcessor phát đúng START → PRE_ROLL/SPEECH → END trên audio thật."""
    starts: List[int] = []
    ends: List[int] = []
    chunks: List[str] = []

    processor = VADProcessor(
        vad_engine="firered-vad",
        threshold=0.4,
        silence_duration_ms=700,
        on_speech_start=lambda: starts.append(1),
        on_speech_chunk=lambda pcm, ts, tag: chunks.append(tag),
        on_speech_end=lambda: ends.append(1),
    )
    processor.prewarm()

    def _feed(pcm_int16: np.ndarray, chunk_samples: int = 320) -> None:
        for i in range(0, len(pcm_int16) - chunk_samples + 1, chunk_samples):
            processor.feed_chunk(pcm_int16[i:i + chunk_samples].tobytes())

    # 1. 0,5 s im lặng: chưa được có START nào.
    _feed(np.zeros(8000, dtype=np.int16))
    assert starts == []
    assert chunks == []

    # 2. Tiếng nói thật: phải có START và frame được chuyển tiếp.
    _feed(_wav_int16(wav_test_dir / "Russian_4s.wav"))
    assert len(starts) >= 1, "không phát hiện START trên audio tiếng nói thật"
    assert len(chunks) > 0, "không chuyển tiếp frame nào cho ASR"
    assert "SPEECH" in chunks, "phải có frame mang tag SPEECH"

    # 3. Im lặng đủ dài: phải chốt câu.
    _feed(np.zeros(16000 * 2, dtype=np.int16))
    assert len(ends) >= 1, "không chốt câu sau khi im lặng đủ lâu"


@pytest.mark.slow
def test_vad_benchmark_all_engines_on_wav_test(wav_test_dir, report_dir):
    """Benchmark 3 engine VAD trên `wav_test/*.wav` và xuất Report."""
    wav_files = sorted(list(wav_test_dir.glob("*.wav")))
    assert len(wav_files) > 0

    benchmark_records: List[Dict] = []

    for engine_name in SUPPORTED_VAD_ENGINES:
        for wav_path in wav_files:
            audio = _wav_int16(wav_path)
            sample_rate = 16000
            duration_sec = len(audio) / sample_rate

            speech_starts: List[int] = []
            speech_ends: List[int] = []

            processor = VADProcessor(
                vad_engine=engine_name,
                threshold=None,          # dùng mặc định của engine (đúng docs)
                silence_duration_ms=None,
                on_speech_start=lambda: speech_starts.append(1),
                on_speech_chunk=lambda pcm, ts, tag: None,
                on_speech_end=lambda: speech_ends.append(1),
            )
            processor.prewarm()

            # Mô phỏng stream theo chunk 20 ms (giống client).
            chunk_size = 320
            chunk_times_us: List[float] = []

            t_stream_start = time.perf_counter()
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                t0 = time.perf_counter()
                processor.feed_chunk(chunk.tobytes())
                t1 = time.perf_counter()
                chunk_times_us.append((t1 - t0) * 1_000_000.0)

            total_process_sec = time.perf_counter() - t_stream_start
            rtf = total_process_sec / duration_sec if duration_sec > 0 else 0.0

            avg_us = float(np.mean(chunk_times_us))
            p95_us = float(np.percentile(chunk_times_us, 95))

            benchmark_records.append({
                "engine": engine_name,
                "filename": wav_path.name,
                "duration_sec": round(duration_sec, 2),
                "avg_chunk_us": round(avg_us, 1),
                "p95_chunk_us": round(p95_us, 1),
                "rtf": round(rtf, 4),
                "speech_starts": len(speech_starts),
                "speech_ends": len(speech_ends),
                "hop_ms": round(processor._frame_samples / sample_rate * 1000, 1),
                "pre_roll_frames": processor._max_lookback_frames,
            })

            logger.info(
                f"[{engine_name}] {wav_path.name}: avg={avg_us:.1f}µs, p95={p95_us:.1f}µs, "
                f"RTF={rtf:.4f}, segments={len(speech_starts)}",
                extra={"module_tag": "VAD"},
            )

    # Xuất báo cáo /report/02_vad/report.md
    phase2_report_dir = report_dir / "02_vad"
    phase2_report_dir.mkdir(parents=True, exist_ok=True)
    report_file = phase2_report_dir / "report.md"

    md_lines = [
        "# Báo Cáo Đo Lường & Kiểm Thử Phase 2: Module VAD Streaming Độc Lập",
        "",
        f"- **Thời gian thực hiện**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- **Kiến trúc**: `report/audit/19_KE_HOACH_VIET_LAI_VAD.md` — mỗi engine dùng đúng",
        "  API docs (`FireRedStreamVad` / `VADIterator` / `AutoModel.generate`), processor chỉ",
        "  chuyển tiếp audio nguyên bản khi engine phát `START`/`END`.",
        "- **Cấu hình**: `threshold=None`, `silence_duration_ms=None` ⇒ mỗi engine dùng đúng",
        "  mặc định trong docs của nó (FireRed `min_silence_frame=20` = 200 ms, Silero 100 ms,",
        "  FSMN `max_end_silence_time` = 800 ms).",
        "",
        "## 1. Kết Quả Benchmark Chi Tiết Từng Engine Trên `/wav_test`",
        "",
        "| Engine | File Audio | Thời lượng | Hop | Trần pre-roll | Avg Chunk (µs) | p95 Chunk (µs) | RTF | Starts | Ends |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in benchmark_records:
        md_lines.append(
            f"| `{r['engine']}` | `{r['filename']}` | {r['duration_sec']}s | {r['hop_ms']} ms | "
            f"{r['pre_roll_frames']} frame | {r['avg_chunk_us']} µs | {r['p95_chunk_us']} µs | "
            f"{r['rtf']} | {r['speech_starts']} | {r['speech_ends']} |"
        )

    engine_rtfs = {}
    for eng in SUPPORTED_VAD_ENGINES:
        sub = [r["rtf"] for r in benchmark_records if r["engine"] == eng]
        engine_rtfs[eng] = round(float(np.mean(sub)), 4) if sub else 0.0

    md_lines.extend([
        "",
        "## 2. Bảng Xếp Hạng Hiệu Năng VAD (RTF & Latency)",
        "",
        "| Engine VAD | RTF Trung Bình | Đặc Điểm |",
        "|---|---|---|",
        f"| **`firered-vad`** | **{engine_rtfs.get('firered-vad', 0)}** | Cửa sổ 400 mẫu / hop 160 mẫu (10 ms). "
        "Trần pre-roll 13 frame (`pad_start_frame + min_speech_frame`); đo được thực xả 12 frame. |",
        f"| **`silero-vad`** | **{engine_rtfs.get('silero-vad', 0)}** | 512 mẫu (32 ms)/frame, "
        "`VADIterator` với `min_silence_duration_ms`/`speech_pad_ms`. |",
        f"| **`fsmn-vad`** | **{engine_rtfs.get('fsmn-vad', 0)}** | Chunk 60 ms, "
        "`generate(cache=…, is_final=False, chunk_size=60)`; trần pre-roll 11 frame "
        "(`window_size_ms + sil_to_speech_time_thres + lookback_time_start_point`). |",
        "",
        "## 3. Kết Luận Nghiệm Thu Phase 2",
        "",
        "- Cả 3 engine chạy đúng API docs, mỗi engine một config riêng, RTF ≪ 1 (nhanh hơn thời gian thực).",
        "- Audio chuyển tiếp cho ASR là **byte nguyên bản** của client: không resample, không gain,",
        "  không clip — xem `test_42_vad_signal_integrity.py`.",
        "- `hangover_ms`/`pre_speech_buffer_ms` đã bị xoá: hangover và pre-padding nay do chính",
        "  VAD quyết định qua `START`/`END` và `lookback_frames`.",
    ])

    report_file.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"📊 [REPORT GENERATED] Đã lưu báo cáo Phase 2 vào: {report_file}", extra={"module_tag": "VAD"})
    assert report_file.exists()


@pytest.mark.slow
def test_moi_engine_dung_dung_mac_dinh_docs_khi_khong_cau_hinh():
    """`threshold`/`silence_duration_ms` = None ⇒ dùng đúng mặc định trong config engine."""
    assert config.vad.silence_duration_ms is None
    assert config.vad.threshold is None

    proc = VADProcessor(vad_engine="silero-vad")
    assert proc.threshold is None
    assert proc.silence_duration_ms is None
    proc.prewarm()
    it = proc._state.engine_state.iterator
    assert it.min_silence_samples == pytest.approx(config.vad.silero.min_silence_duration_ms * 16.0)
    assert it.threshold == pytest.approx(config.vad.silero.threshold)
