"""Unit tests for Phase 2: Lookahead Audio Demuxer & Timeline Aligner.

Bổ sung (v2 — sau sự cố "mất phụ đề gốc và dịch"):
  * `StreamDemuxer` — bộ ghép nối byte LIÊN TỤC cho mảnh MSE bị cắt giữa cluster.
    Đây là regression test cho nguyên nhân gốc: giải mã từng mảnh 32 KB độc lập chỉ
    thu được ~35 % lượng audio.
"""

import io
import time
import numpy as np
import pytest
import soundfile as sf
from backend.core.lookahead_demuxer import LookaheadDemuxer, DecodedAudioChunk
from backend.core.timeline_aligner import TimelineAligner
from backend.core.stream_demuxer import (
    StreamDemuxer,
    detect_container,
    looks_like_init_segment,
    scan_mp4_boundaries,
    scan_webm_boundaries,
)


def create_synthetic_wav_bytes(duration_sec: float = 3.0, sr: int = 48000, freq: float = 440.0) -> bytes:
    """Tạo byte WAV mẫu 48kHz stereo/mono để giả lập chunk từ trình duyệt."""
    num_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    sine = (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="WAV")
    return buf.getvalue()


def test_demuxer_decode_wav():
    """Kiểm tra giải mã chunk và resample sang 16kHz mono float32."""
    demuxer = LookaheadDemuxer(target_sample_rate=16000)
    raw_wav = create_synthetic_wav_bytes(duration_sec=2.5, sr=48000)

    chunk = demuxer.decode_chunk(raw_wav, timestamp_offset=10.0)
    assert chunk is not None
    assert chunk.sample_rate == 16000
    assert chunk.channels == 1
    assert abs(chunk.duration - 2.5) < 0.05
    assert abs(chunk.pts_start - 10.0) < 0.01
    assert abs(chunk.pts_end - 12.5) < 0.05
    assert len(chunk.pcm) == pytest.approx(16000 * 2.5, abs=100)
    assert chunk.pcm.dtype == np.float32
    assert -1.05 <= np.max(chunk.pcm) <= 1.05


def test_demuxer_speed_rtf():
    """Đảm bảo tốc độ giải mã cực nhanh (RTF < 0.01, nhanh hơn 100x realtime)."""
    demuxer = LookaheadDemuxer(target_sample_rate=16000)
    raw_wav = create_synthetic_wav_bytes(duration_sec=10.0, sr=48000)

    t0 = time.perf_counter()
    chunk = demuxer.decode_chunk(raw_wav, timestamp_offset=0.0)
    t1 = time.perf_counter()

    assert chunk is not None
    elapsed_sec = t1 - t0
    rtf = elapsed_sec / 10.0
    print(f"\n[DEMUX BENCHMARK] Decoded 10s in {elapsed_sec*1000:.2f}ms (RTF: {rtf:.5f}, {1/rtf:.1f}x RT)")
    assert rtf < 0.01, f"Demuxer RTF quá cao: {rtf}"


def test_timeline_aligner_stitching():
    """Kiểm tra khả năng ghép nối 2 chunk liên tiếp và trích xuất lát cắt."""
    aligner = TimelineAligner(sample_rate=16000)

    # Tạo 2 chunk: [0.0s -> 5.0s] và [5.0s -> 10.0s]
    pcm1 = np.ones(16000 * 5, dtype=np.float32) * 0.5
    pcm2 = np.ones(16000 * 5, dtype=np.float32) * 0.8

    chunk1 = DecodedAudioChunk(pts_start=0.0, pts_end=5.0, duration=5.0, pcm=pcm1)
    chunk2 = DecodedAudioChunk(pts_start=5.0, pts_end=10.0, duration=5.0, pcm=pcm2)

    aligner.add_chunk(chunk1)
    aligner.add_chunk(chunk2)

    buffered_min, buffered_max = aligner.get_buffered_range()
    assert buffered_min == 0.0
    assert buffered_max == 10.0

    # Trích xuất đoạn giữa [2.0s -> 8.0s]
    slice_out = aligner.extract_slice(start_pts=2.0, end_pts=8.0)
    assert slice_out is not None
    assert abs(slice_out.duration - 6.0) < 0.01
    assert len(slice_out.pcm) == 16000 * 6


def test_timeline_aligner_prune_and_reset():
    """Kiểm tra chức năng dọn dẹp cache quá khứ và tua video reset."""
    aligner = TimelineAligner(sample_rate=16000)

    for i in range(10):
        pcm = np.zeros(16000 * 2, dtype=np.float32)
        chunk = DecodedAudioChunk(pts_start=i * 2.0, pts_end=(i + 1) * 2.0, duration=2.0, pcm=pcm)
        aligner.add_chunk(chunk)

    # Tổng cộng có 20s [0.0 -> 20.0]
    assert aligner.get_buffered_range() == (0.0, 20.0)

    # Prune tại current_time = 15.0s, giữ lại 6s trước đó -> cutoff = 9.0s
    aligner.prune_past(current_time=15.0, keep_behind_sec=6.0)
    buffered_min, buffered_max = aligner.get_buffered_range()
    assert buffered_min >= 8.0
    assert buffered_max == 20.0

    # Reset khi seek
    aligner.reset(target_time=50.0)
    assert aligner.get_buffered_range() == (0.0, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# v2: StreamDemuxer — ghép nối byte liên tục
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def webm_opus_sample() -> bytes:
    """Tạo file WebM/Opus 8 s có 3 đoạn "tiếng nói" xen im lặng (bằng PyAV)."""
    av = pytest.importorskip("av")
    sr, duration = 48000, 8.0
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="webm")
    st = out.add_stream("libopus", rate=sr)
    st.layout = "mono"
    st.bit_rate = 128000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t, dtype=np.float32)
    for k, (a, b) in enumerate([(0.5, 3.0), (4.0, 7.0)]):
        m = (t >= a) & (t < b)
        sig[m] = 0.4 * np.sin(2 * np.pi * (300 + 100 * k) * t[m])
    frame_len = 960
    for i in range(0, len(sig) - frame_len, frame_len):
        frame = av.AudioFrame.from_ndarray(
            sig[i:i + frame_len].reshape(1, -1), format="fltp", layout="mono"
        )
        frame.sample_rate = sr
        for pkt in st.encode(frame):
            out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()
    return buf.getvalue()


def test_detect_container_and_init_segment(webm_opus_sample):
    """Nhận diện container và init segment của WebM."""
    assert detect_container(webm_opus_sample) == "webm"
    bounds = scan_webm_boundaries(webm_opus_sample)
    assert bounds, "Phải tìm được ít nhất 1 Cluster"
    assert bounds[0] > 0
    init = webm_opus_sample[:bounds[0]]
    assert looks_like_init_segment(init)
    assert not looks_like_init_segment(webm_opus_sample)
    # MP4: không có moof ⇒ không có ranh giới fragment
    assert scan_mp4_boundaries(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 32) == []


def test_stream_demuxer_reassembles_fragmented_webm(webm_opus_sample):
    """REGRESSION: mảnh 32 KB cắt giữa cluster phải được ghép lại và giải mã ĐỦ.

    Bản v1 giải mã từng mảnh độc lập nên chỉ thu được ~35 % audio (12 s -> 4,2 s);
    hậu quả là phần lớn câu không bao giờ tới được ASR và mốc phụ đề sai hoàn toàn.
    """
    bounds = scan_webm_boundaries(webm_opus_sample)
    init = webm_opus_sample[:bounds[0]]
    body = webm_opus_sample[bounds[0]:]

    demux = StreamDemuxer(target_sample_rate=16000)
    assert demux.feed(init, timestamp_offset=0.0, is_init=True, epoch=0) == []

    chunks = []
    for i in range(0, len(body), 32768):
        chunks.extend(demux.feed(body[i:i + 32768], timestamp_offset=0.0, epoch=0))

    total = sum(c.duration for c in chunks)
    assert total >= 7.8, f"Chỉ thu được {total:.2f}s / 8.00s audio"
    assert demux.average_rtf < 0.05

    # Không chồng lấn và phủ kín timeline
    prev_end = None
    for chunk in chunks:
        if prev_end is not None:
            assert chunk.pts_start >= prev_end - 0.05, "Các đoạn bị chồng lấn"
        prev_end = chunk.pts_start + len(chunk.pcm) / 16000.0

    # Ghép lại thành 1 mảng và kiểm chứng năng lượng đúng vị trí
    full = np.zeros(int(16000 * 8.5), dtype=np.float32)
    for chunk in chunks:
        i0 = int(chunk.pts_start * 16000)
        full[i0:i0 + len(chunk.pcm)] += chunk.pcm
    rms_speech = float(np.sqrt(np.mean(full[int(1.0 * 16000):int(2.5 * 16000)] ** 2)))
    rms_silence = float(np.sqrt(np.mean(full[int(3.2 * 16000):int(3.8 * 16000)] ** 2)))
    assert rms_speech > 0.05, "Mất tiếng ở đoạn nói"
    assert rms_silence < 0.01, "Đoạn im lặng bị lẫn tiếng"


def test_stream_demuxer_handles_full_file_without_init_flag(webm_opus_sample):
    """Nếu mảnh đầu là file WebM ĐẦY ĐỦ (không phải DASH) thì vẫn phải giải mã được."""
    demux = StreamDemuxer(target_sample_rate=16000)
    chunks = demux.feed(webm_opus_sample, timestamp_offset=0.0, epoch=0)
    assert chunks, "Không giải mã được file WebM tự chứa"
    assert sum(c.duration for c in chunks) > 7.5


def test_stream_demuxer_epoch_change_resets(webm_opus_sample):
    """Đổi epoch (SourceBuffer mới) phải xoá bộ đệm ghép nối để không trộn 2 luồng."""
    bounds = scan_webm_boundaries(webm_opus_sample)
    init = webm_opus_sample[:bounds[0]]
    body = webm_opus_sample[bounds[0]:]

    demux = StreamDemuxer(target_sample_rate=16000)
    demux.feed(init, timestamp_offset=0.0, is_init=True, epoch=0)
    demux.feed(body[:32768], timestamp_offset=0.0, epoch=0)
    assert demux.buffered_bytes > 0

    demux.feed(body[32768:65536], timestamp_offset=0.0, epoch=5)
    assert demux.epoch == 5
    # Đệm cũ đã bị xoá và chỉ còn đúng mảnh của epoch mới (chưa có init mới nên không
    # giải mã được -> không được ném lỗi).
    assert demux.buffered_bytes <= 32768


def test_stream_demuxer_rejects_garbage_without_crash():
    """Dữ liệu rác không được làm sập handler."""
    demux = StreamDemuxer(target_sample_rate=16000)
    assert demux.feed(b"", timestamp_offset=0.0) == []
    assert demux.feed(b"\x00" * 1024, timestamp_offset=0.0) == []

