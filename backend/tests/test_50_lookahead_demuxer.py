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


def test_stream_demuxer_pts_frontier_never_passes_emitted_audio(webm_opus_sample):
    """REGRESSION (2026-10-01): mốc tiến độ KHÔNG được vượt phần audio đã phát ra PCM.

    Sự cố thật: mỗi lượt giải mã chỉ thu được ~4,1 s trong khi ~20 s audio đã về (trang
    ngoài YouTube), phụ đề chỉ hiện 1/3-5 câu. Nguyên nhân là `_last_pts` được cập nhật
    theo `pts + frame_dur` của MỌI frame — kể cả frame mà resampler chưa nhả mẫu nào —
    nên nó chạy trước phần PCM thực có; lượt sau lọc `pts < _last_pts` liền bỏ vĩnh viễn
    đoạn chênh lệch đó.

    Bất biến kiểm ở đây: mốc đã phát (`last_pts`) luôn ≤ tổng audio ĐÃ PHÁT + mép frame,
    và tổng audio thu hồi được phải phủ gần hết file nguồn.
    """
    bounds = scan_webm_boundaries(webm_opus_sample)
    init = webm_opus_sample[:bounds[0]]
    body = webm_opus_sample[bounds[0]:]

    demux = StreamDemuxer(target_sample_rate=16000)
    demux.feed(init, timestamp_offset=0.0, is_init=True, epoch=0)

    chunks = []
    for i in range(0, len(body), 16384):   # mảnh nhỏ hơn 32 KB: nhiều lượt giải mã hơn
        chunks.extend(demux.feed(body[i:i + 16384], timestamp_offset=0.0, epoch=0))

    emitted = sum(c.duration for c in chunks)
    assert emitted >= 7.8, f"Chỉ thu được {emitted:.2f}s / 8.00s audio"

    # Mốc tiến độ không được vượt quá phần đã phát (cho phép đúng 1 frame Opus ở mép).
    assert demux.last_pts <= emitted + 0.05, (
        f"mốc tiến độ {demux.last_pts:.3f}s vượt phần đã phát {emitted:.3f}s "
        f"=> audio sẽ bị bỏ ở lượt sau"
    )
    # Các đoạn phải liền nhau, không chồng lấn, không hở.
    prev_end = None
    for chunk in chunks:
        if prev_end is not None:
            assert chunk.pts_start >= prev_end - 0.02, "Các đoạn bị chồng lấn"
            assert chunk.pts_start <= prev_end + 0.02, "Có khe hở giả giữa các đoạn"
        prev_end = chunk.pts_end


def test_stream_demuxer_uses_timestamp_offset_on_one_time_axis(webm_opus_sample):
    """`timestampOffset` của SourceBuffer phải được áp NHẤT QUÁN (PTS ra = pts + offset).

    Bản cũ seek theo `container_pts` nhưng lọc theo mốc đã cộng offset (hoặc ngược lại) —
    với trang có offset lớn, mọi frame mới bị coi là "cũ" và bị bỏ sạch.
    """
    bounds = scan_webm_boundaries(webm_opus_sample)
    init = webm_opus_sample[:bounds[0]]
    body = webm_opus_sample[bounds[0]:]
    offset = 3600.0  # kiểu `timestampOffset` để neo segment vào timeline video dài

    demux = StreamDemuxer(target_sample_rate=16000)
    demux.feed(init, timestamp_offset=offset, is_init=True, epoch=0)

    chunks = []
    for i in range(0, len(body), 32768):
        chunks.extend(demux.feed(body[i:i + 32768], timestamp_offset=offset, epoch=0))

    assert chunks, "Không giải mã được gì khi có timestampOffset"
    assert sum(c.duration for c in chunks) >= 7.8
    assert chunks[0].pts_start >= offset - 0.1, (
        f"PTS phải nằm trên trục video (>= {offset}s) nhưng nhận {chunks[0].pts_start:.2f}s"
    )
    assert chunks[-1].pts_end <= offset + 8.5


def test_adts_frame_parser_roundtrip():
    """Bộ tách frame ADTS phải đọc lại ĐÚNG các frame đã đóng gói (nền của đường dự phòng)."""
    from backend.core.stream_demuxer import _iter_adts_frames

    def hdr(frame_len: int, sr_index: int = 4, channels: int = 1) -> bytes:
        b = bytearray(7)
        b[0], b[1] = 0xFF, 0xF1
        b[2] = ((1 & 0x3) << 6) | ((sr_index & 0xF) << 2) | ((channels >> 2) & 0x1)
        b[3] = ((channels & 0x3) << 6) | ((frame_len >> 11) & 0x3)
        b[4] = (frame_len >> 3) & 0xFF
        b[5] = ((frame_len & 0x7) << 5) | 0x1F
        b[6] = 0xFC
        return bytes(b)

    frames = [hdr(100 + i + 7) + bytes([i % 251]) * (100 + i) for i in range(50)]
    assert _iter_adts_frames(b"".join(frames)) == frames


def test_raw_aac_fallback_refuses_false_positives():
    """Đường dự phòng AAC thô KHÔNG được đẩy PCM rác khi payload không phải luồng ADTS.

    Quét `0xFFF` trên AAC thô (fMP4) khớp giả ở rất nhiều vị trí; nếu không chặn thì PCM rác
    sẽ vào thẳng phụ đề. Độ phủ byte phải gần 100 % mới coi là luồng ADTS thật.
    """
    demux = StreamDemuxer(target_sample_rate=16000)
    demux._container = "mp4"
    # Byte "AAC thô" giả: có mẫu 0xFFF rải rác nhưng KHÔNG phải chuỗi frame ADTS hợp lệ
    # (header khai frame dài hơn dữ liệu còn lại ⇒ không có frame nào hợp lệ).
    demux._media = bytearray((b"\xff\xf1\x50\x80\x00\x1f\xfc" + b"\x11" * 9) * 2000)
    assert demux._decode_raw_aac() == []
    assert demux._raw_fallback_used == 0


def test_mp4_fragment_chain_report_on_real_fmp4():
    """Chuỗi fragment fMP4 THẬT: `track_id` và `tfdt` phải đọc đúng, và báo LIỀN MẠCH.

    Dùng file do PyAV ghi ra (không tự đóng gói box bằng tay) — đây là dạng dữ liệu đúng như
    trình duyệt append, nên test có giá trị hồi quy thực.
    """
    av = pytest.importorskip("av")
    import io

    buf = io.BytesIO()
    out = av.open(
        buf, mode="w", format="mp4",
        options={"movflags": "frag_keyframe+empty_moov+default_base_moof",
                 "frag_duration": "2000000"},
    )
    sr, duration = 44100, 20.0
    st = out.add_stream("aac", rate=sr)
    st.layout = "mono"
    st.bit_rate = 128000
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    for i in range(0, n - 1024, 1024):
        seg = (0.4 * np.sin(2 * np.pi * 300 * t[i:i + 1024])).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(seg.reshape(1, -1), format="fltp", layout="mono")
        frame.sample_rate = sr
        frame.pts = i
        for pkt in st.encode(frame):
            out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()
    data = buf.getvalue()

    pos = data.find(b"moof")
    assert pos > 4
    body = data[pos - 4:]

    from backend.core.stream_demuxer import scan_mp4_boundaries, scan_mp4_fragment_times

    assert len(scan_mp4_boundaries(body)) >= 5
    times = scan_mp4_fragment_times(body)
    assert times, "Không đọc được moof nào"
    assert {t for _d, t in times} == {1}, f"track_id sai: {times}"
    decode_times = [d for d, _t in times]
    assert decode_times == sorted(decode_times), "tfdt phải tăng dần"

    demux = StreamDemuxer(target_sample_rate=16000)
    demux._container = "mp4"
    demux._media = bytearray(body)
    report = demux._mp4_fragment_chain_report(sr)
    assert "LIỀN MẠCH" in report, report


