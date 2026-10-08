"""Unit tests cho `StreamDemuxer` — bộ ghép nối byte LIÊN TỤC của Pipeline B (Lookahead).

Regression test cho nguyên nhân gốc "mất phụ đề gốc và dịch": giải mã từng mảnh 32 KB độc lập
(Giải pháp v1 `LookaheadDemuxer` + `TimelineAligner` trong `backend/core/`) chỉ thu được ~35 %
lượng audio, vì mảnh bị cắt GIỮA cluster WebM / fragment fMP4 không tự parse được.
Hai module v1 đó đã bị XOÁ ngày 2026-10-06 (không còn ai dùng trong runtime).
"""

import io
import time
import numpy as np
import pytest
from backend.core.stream_demuxer import (
    StreamAudioChunk,
    StreamDemuxer,
    detect_container,
    looks_like_init_segment,
    scan_mp4_boundaries,
    scan_webm_boundaries,
)


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


def test_stream_demuxer_learns_axis_when_session_starts_mid_stream(webm_opus_sample):
    """HỒI QUY 2026-10-05 (xhamster.com — "tắt mở lại session cũng không được").

    Phiên mới neo ở 984s (`min_pts` = 983.55) trong khi PTS bên trong container chỉ 0..8s.
    Bản cũ: không có neo trục nào để học (`_browser_axis_bias is None`) nên frontier lọc sạch
    mọi frame ⇒ `chunks` luôn rỗng ⇒ KHÔNG BAO GIỜ học được trục ⇒ log backend lặp
    "Audio RAM: 0.0s | Đã dịch: 0.0s (giải mã 0.00x, PCM/byte 0.000)" suốt phiên.

    Nay trục được HỌC từ `media_end` do extension gửi kèm (byte này kết thúc ở `media_end`).
    """
    bounds = scan_webm_boundaries(webm_opus_sample)
    init = webm_opus_sample[:bounds[0]]
    body = webm_opus_sample[bounds[0]:]

    demux = StreamDemuxer(target_sample_rate=16000)
    demux.reset(epoch=1, min_pts=983.55)
    assert demux._browser_axis_bias is None, "phiên mới chưa có neo trục"

    assert demux.feed(init, is_init=True, epoch=1, media_start=984.0, media_end=992.0) == []

    chunks = []
    for i in range(0, len(body), 32768):
        chunks.extend(
            demux.feed(body[i:i + 32768], epoch=1, media_start=984.0, media_end=992.0)
        )

    total = sum(c.duration for c in chunks)
    assert total >= 7.8, f"phiên mới phải giải mã được audio, chỉ thu {total:.2f}s / 8.00s"
    assert chunks[0].pts_start >= 983.0, (
        f"PCM phải nằm trên TRỤC TRÌNH DUYỆT (~984s), nhận được {chunks[0].pts_start:.2f}s"
    )


def test_axis_shift_reversal_is_ignored():
    """HỒI QUY 2026-10-05 (xvideos.com — HLS fMP4 tách track): chặn ĐẢO TRỤC.

    Log thật: `LỆCH TRỤC +10,01s` rồi `LỆCH TRỤC −10,01s` sau 4 giây. Mỗi lần đảo, PCM của các
    fragment sau bị đặt LÙI một segment ⇒ rơi vào vùng đã có audio ⇒ bị frontier lọc ⇒ audio
    vùng thật MẤT HẲN ⇒ timeline thủng một khe 20s và con trỏ khối ASR kẹt ở mép khe
    ("chạy một chút lại dừng").
    """
    demux = StreamDemuxer(target_sample_rate=16000)

    def mk(start: float, end: float):
        return [StreamAudioChunk(
            pts_start=start, pts_end=end, duration=end - start,
            pcm=np.zeros(160, dtype=np.float32),
        )]

    # Lần 1: PCM kết thúc ở 10s, trình duyệt báo byte kết thúc ở 20s ⇒ neo +10s.
    first = demux._anchor_to_browser_axis(mk(0.0, 10.0), 10.0, 20.0)
    assert first[0].pts_start == pytest.approx(10.0)
    assert demux._browser_axis_bias == pytest.approx(10.0)

    # Lần 2 (4 giây sau): ngược dấu, cùng độ lớn ⇒ phải BỎ QUA.
    demux._axis_shift_signed_at = time.perf_counter() - 4.0
    second = demux._anchor_to_browser_axis(mk(20.0, 30.0), 10.0, 20.0)
    assert second[0].pts_start == pytest.approx(20.0), "KHÔNG được dịch ngược (sẽ tạo khe hở audio)"
    assert demux._browser_axis_bias == pytest.approx(10.0), "phải giữ nguyên trục đang dùng"

    # Nhưng một thay đổi trục THẬT (khác độ lớn) vẫn phải được áp dụng.
    third = demux._anchor_to_browser_axis(mk(20.0, 30.0), 60.0, 80.0)
    assert third[0].pts_start == pytest.approx(70.0), "lệch +50s là đổi trục thật ⇒ phải neo lại"


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


# ──────────────────────────────────────────────────────────────────────────────
# Neo trục theo `SourceBuffer.buffered` (browser media timeline)
# ──────────────────────────────────────────────────────────────────────────────

def test_axis_anchor_follows_browser_hint_when_container_is_offset(webm_opus_sample):
    """REGRESSION (2026-10-03, trang ngoài YouTube): PCM phải nằm trên TRỤC TRÌNH DUYỆT.

    Sự cố thật: extension gửi mảnh audio 0→56,6 s (theo `SourceBuffer.buffered`) nhưng
    backend giải mã ra PCM ở 250,0→306,8 s ⇒ `Đệm trước: 299,7 s`, `Đã dịch: 0,0 s`, video kẹt
    ở 7,1 s. Nguyên nhân: `container_pts + timestampOffset` KHÔNG trùng trục media của trình
    duyệt khi trang dùng `mode="sequence"`, `tfdt` tuyệt đối, hoặc đổi `timestampOffset` ngay
    sau `appendBuffer`.

    Ở đây container 0→8 s còn `media_start/media_end` (trục trình duyệt) = container + 250 s:
    PCM phải được NEO về 250→258 s, không được giữ ở 0→8 s.
    """
    bounds = scan_webm_boundaries(webm_opus_sample)
    init = webm_opus_sample[:bounds[0]]
    body = webm_opus_sample[bounds[0]:]
    shift = 250.0

    demux = StreamDemuxer(target_sample_rate=16000)
    demux.feed(init, timestamp_offset=0.0, is_init=True, epoch=0, media_start=shift, media_end=shift)

    chunks = []
    for i in range(0, len(body), 32768):
        start = i / float(len(body)) * 8.0 + shift      # trục trình duyệt = container + 250
        chunks.extend(demux.feed(
            body[i:i + 32768], timestamp_offset=0.0, epoch=0,
            media_start=start, media_end=start + 32768 / float(len(body)) * 8.0 + 0.2,
        ))

    assert chunks, "Không giải mã được gì"
    assert chunks[0].pts_start >= shift - 0.2, (
        f"PCM phải nằm trên trục trình duyệt (~{shift}s) nhưng ở {chunks[0].pts_start:.2f}s"
    )
    assert chunks[-1].pts_end <= shift + 8.5
    assert demux._browser_axis_bias is not None
    assert abs(demux._browser_axis_bias - shift) < 0.75, (
        f"Bias học được phải ≈ {shift}s, nhận {demux._browser_axis_bias}"
    )


def test_axis_anchor_keeps_container_axis_when_hints_match(webm_opus_sample):
    """Trang "chuẩn" (YouTube): hint trùng trục container ⇒ KHÔNG dịch PCM, chỉ khoá trục."""
    bounds = scan_webm_boundaries(webm_opus_sample)
    init = webm_opus_sample[:bounds[0]]
    body = webm_opus_sample[bounds[0]:]

    demux = StreamDemuxer(target_sample_rate=16000)
    demux.feed(init, timestamp_offset=0.0, is_init=True, epoch=0)

    chunks = []
    n = len(body)
    for i in range(0, n, 32768):
        end = (i + 32768) / float(n) * 8.0
        chunks.extend(demux.feed(
            body[i:i + 32768], timestamp_offset=0.0, epoch=0,
            media_start=i / float(n) * 8.0, media_end=end,
        ))

    assert chunks
    assert abs(chunks[0].pts_start) < 0.6, f"PCM bị dịch oan: {chunks[0].pts_start:.2f}s"
    assert demux._browser_axis_bias == 0.0


def test_axis_anchor_does_not_double_apply_timestamp_offset(webm_opus_sample):
    """Hint khớp `timestampOffset` ⇒ neo KHÔNG được cộng offset lần thứ hai."""
    bounds = scan_webm_boundaries(webm_opus_sample)
    init = webm_opus_sample[:bounds[0]]
    body = webm_opus_sample[bounds[0]:]
    offset = 3600.0

    demux = StreamDemuxer(target_sample_rate=16000)
    demux.feed(init, timestamp_offset=offset, is_init=True, epoch=0)

    chunks = []
    n = len(body)
    for i in range(0, n, 32768):
        end = (i + 32768) / float(n) * 8.0
        chunks.extend(demux.feed(
            body[i:i + 32768], timestamp_offset=offset, epoch=0,
            media_start=offset + i / float(n) * 8.0, media_end=offset + end,
        ))

    assert chunks
    assert chunks[0].pts_start >= offset - 0.6
    assert chunks[-1].pts_end <= offset + 8.5
    assert abs(demux._browser_axis_bias - offset) < 0.75


def test_axis_anchor_defers_small_shift_until_confirmed(webm_opus_sample):
    """Lệch nhỏ (≤ 5 s) chỉ sửa sau khi có quan sát thứ hai khớp — tránh mảnh hỏng mép cuối."""
    bounds = scan_webm_boundaries(webm_opus_sample)
    init = webm_opus_sample[:bounds[0]]
    body = webm_opus_sample[bounds[0]:]
    shift = 3.0

    demux = StreamDemuxer(target_sample_rate=16000)
    demux.feed(init, timestamp_offset=0.0, is_init=True, epoch=0)

    first = demux.feed(
        body[:32768], timestamp_offset=0.0, epoch=0,
        media_start=shift, media_end=shift + 32768 / float(len(body)) * 8.0,
    )
    assert first, "Lượt đầu vẫn phải trả PCM (chỉ chưa neo trục)"
    assert first[0].pts_start < shift - 1.0, "Lượt đầu KHÔNG được sửa trục"
    assert demux._axis_shift_obs is not None

    demux.feed(
        body[32768:65536], timestamp_offset=0.0, epoch=0,
        media_start=shift + 32768 / float(len(body)) * 8.0,
        media_end=shift + 65536 / float(len(body)) * 8.0,
    )
    assert demux._browser_axis_bias is not None, "Quan sát thứ hai khớp ⇒ phải neo trục"


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


def test_continuous_buffering_does_not_cause_axis_shift_or_audio_gap():
    """HỒI QUY 2026-10-08: LỆCH TRỤC THỜI GIAN ma khi browser nạp trước segment (+7.37s).

    Kịch bản thực tế:
    - Segment 1: PCM [2.04s -> 9.99s], media_end=10.0s. Trục khớp (+0.01s), khoá trục bias=0.0s.
    - Segment 2: media_start=9.99s, media_end=17.36s (trình duyệt buffer trước 7.37s).
    - Bản cũ: so media_end (17.36s) với PCM cuối (9.99s) thấy lệch +7.37s (> 5.0s) ⇒ vội vàng
      đổi bias lên +7.37s và kéo _last_pts lên 17.36s ⇒ tạo khe hở 7.37s và mất trọn 1 đoạn phụ đề!
    - Bản mới: nhận biết media_start=9.99s liên tục với mốc PCM kết thúc 9.99s ⇒ KHÔNG đổi trục.
    """
    demux = StreamDemuxer(target_sample_rate=16000)

    def mk(start: float, end: float):
        return [StreamAudioChunk(
            pts_start=start, pts_end=end, duration=end - start,
            pcm=np.zeros(int(round((end - start) * 16000)), dtype=np.float32),
        )]

    # Segment 1: [2.04 -> 9.99], media_end=10.0s => khoá trục
    seg1 = demux._anchor_to_browser_axis(mk(2.04, 9.99), media_start=1.54, media_end=10.0)
    assert seg1[0].pts_start == pytest.approx(2.04)
    assert demux._browser_axis_bias == 0.0

    # Segment 2 nạp vào: media_start=9.99s, media_end=17.36s
    # Lúc này nếu có chunk mới bắt đầu từ 9.99s (hoặc chunk cũ kết thúc ở 9.99s)
    seg2 = demux._anchor_to_browser_axis(mk(9.99, 17.36), media_start=9.99, media_end=17.36)
    assert seg2[0].pts_start == pytest.approx(9.99)
    assert seg2[0].pts_end == pytest.approx(17.36)
    assert demux._browser_axis_bias == 0.0, "Trục phải giữ nguyên bias=0.0s, không được nhảy +7.37s"

    # Trường hợp mảnh byte của segment 2 mới chỉ giải mã được 1 phần nhỏ (vd 9.99s -> 11.0s)
    # trong khi media_end đã là 17.36s (chênh 6.36s > 5.0s)
    seg2_partial = demux._anchor_to_browser_axis(mk(9.99, 11.0), media_start=9.99, media_end=17.36)
    assert seg2_partial[0].pts_start == pytest.approx(9.99)
    assert seg2_partial[0].pts_end == pytest.approx(11.0)
    assert demux._browser_axis_bias == 0.0, "Không được lệch trục khi mảnh mới bắt đầu đúng media_start"


