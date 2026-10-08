"""test_77 — Demuxer KHÔNG được làm mất audio đã giải mã hoặc chưa hề phát.

BỐI CẢNH (log thật 2026-10-08, Pipeline B kẹt vĩnh viễn ở playhead 15.7s)

Phiên mở GIỮA video: playhead 9.38s, neo `min_pts=8.88`, client replay 10 mảnh phủ
0→40s (≈16.7 MB) trong một lần. Backend nhận đủ 13 mảnh nhưng chỉ giải mã ra PCM
tới **16.00s** rồi đứng im mãi: `decoded_chunks` kẹt ở 2, `ready_until_pts=16.00`
suốt 40 giây, video tạm dừng và không bao giờ có mảnh mới (trình phát đã đệm sẵn
30s nên không `appendBuffer` nữa) ⇒ **deadlock**.

Hai khiếm khuyết trong `StreamDemuxer._decode_new` / `_decode_fragments_individually`
giải thích đúng triệu chứng đó:

1. **`_last_pts` nhảy qua audio CHƯA HỀ PHÁT.** Khi đường giải mã từng-fragment không
   ra PCM nào, hàm vẫn đẩy `_last_pts` tới `recovered_frontier` (tfdt của mảnh CUỐI
   trong bộ đệm) để "lượt sau tiến tiếp". Nếu bộ đệm đang giữ 0→40s còn frontier ở
   16s thì **một** lượt như vậy đẩy frontier tới ~38s ⇒ mọi frame của 16→38s vĩnh viễn
   bị coi là "cũ" (`pts < _last_pts`) và bị lọc. Đây là mất dữ liệu không thể phục hồi,
   và đúng bằng triệu chứng "kẹt ở một mốc duy nhất".

2. **Kết quả đường chuẩn bị VỨT.** Nhánh dự phòng `return per_frag` thay vì nối thêm,
   nên PCM mà đường giải mã nối-liền vừa phát ra bị bỏ đi.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from backend.core.stream_demuxer import StreamAudioChunk, StreamDemuxer

SR = 44100
DURATION = 40.0


@pytest.fixture(scope="module")
def fmp4_40s() -> tuple:
    """fMP4/AAC 40s, mảnh 4s (do PyAV ghi ra — cùng dạng trình duyệt append)."""
    av = pytest.importorskip("av")
    buf = io.BytesIO()
    out = av.open(
        buf,
        mode="w",
        format="mp4",
        options={
            "movflags": "frag_keyframe+empty_moov+default_base_moof",
            "frag_duration": "4000000",
        },
    )
    st = out.add_stream("aac", rate=SR)
    st.layout = "mono"
    st.bit_rate = 128000
    n = int(SR * DURATION)
    t = np.linspace(0, DURATION, n, endpoint=False)
    for i in range(0, n - 1024, 1024):
        seg = (0.4 * np.sin(2 * np.pi * 300 * t[i:i + 1024])).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(seg.reshape(1, -1), format="fltp", layout="mono")
        frame.sample_rate = SR
        frame.pts = i
        for pkt in st.encode(frame):
            out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()

    data = buf.getvalue()
    offs = []
    i = 0
    while True:
        j = data.find(b"moof", i)
        if j < 0:
            break
        offs.append(j - 4)
        i = j + 4
    init = data[: offs[0]]
    frags = [
        data[s: (offs[k + 1] if k + 1 < len(offs) else len(data))]
        for k, s in enumerate(offs)
    ]
    assert len(frags) >= 8, f"cần nhiều mảnh để test, chỉ có {len(frags)}"
    return init, frags


def _demux_with_buffer(init: bytes, frags, last_pts: float, upto: int | None = None):
    demux = StreamDemuxer(target_sample_rate=16000)
    demux._container = "mp4"
    demux._init = init
    demux._media = bytearray(b"".join(frags if upto is None else frags[:upto]))
    demux._last_pts = float(last_pts)
    demux._browser_axis_bias = 0.0
    return demux


# ── 1. Frontier không được nhảy qua audio chưa phát ───────────────────────────

def test_fallback_khong_ra_pcm_thi_khong_duoc_day_frontier(fmp4_40s, monkeypatch):
    """Đường từng-fragment ra PCM RỖNG ⇒ TUYỆT ĐỐI không được đẩy `_last_pts`.

    Trước khi sửa: `_last_pts` nhảy tới `recovered_frontier` (~38s) dù chưa phát ra
    một mẫu PCM nào ⇒ 16→38s bị lọc vĩnh viễn ⇒ pipeline kẹt ở 16.00s như log thật.
    """
    init, frags = fmp4_40s
    demux = _demux_with_buffer(init, frags, last_pts=16.0)
    frontier_before = demux._last_pts

    # Mô phỏng: bộ dò ranh giới trả về một cặp mảnh ở tận cuối bộ đệm nhưng KHÔNG
    # mở/giải mã được (mảnh hỏng, codec lạ…) ⇒ `out` rỗng.
    monkeypatch.setattr(demux, "_fragment_byte_ranges", lambda: [(0, 4, 36.0)])
    out = demux._decode_fragments_individually()

    assert out == [], "điều kiện dựng: đường từng-fragment phải ra RỖNG"
    assert demux._last_pts == pytest.approx(frontier_before), (
        f"`_last_pts` nhảy {frontier_before:.2f}s → {demux._last_pts:.2f}s dù KHÔNG phát "
        f"ra PCM nào ⇒ audio {frontier_before:.1f}–{demux._last_pts:.1f}s bị lọc vĩnh viễn"
    )


# ── 2. Kết quả đường chuẩn không được bị vứt ──────────────────────────────────

def test_ket_qua_fallback_khong_duoc_vut_pcm_cua_duong_chuan(fmp4_40s, monkeypatch):
    """Kết quả dự phòng phải NỐI THÊM vào đường chuẩn, không được thay thế nó."""
    init, frags = fmp4_40s

    # Đối chứng: đường dự phòng không ra gì ⇒ lượng PCM mà đường CHUẨN phát được.
    control = _demux_with_buffer(init, frags, last_pts=8.88)
    monkeypatch.setattr(control, "_decode_fragments_individually", lambda: [])
    base = control._decode_new()
    base_span = sum(c.duration for c in base)
    assert base_span > 1.0, "điều kiện dựng: đường chuẩn phải phát ra PCM"

    # Có đường dự phòng trả về một đoạn ở tận cuối bộ đệm. Ép nhánh dự phòng chạy
    # bằng cách báo `last_tfdt` xa hơn frontier (đúng tình huống bộ đệm 0→40s mà
    # frontier mới ở 16s như log thật).
    demux = _demux_with_buffer(init, frags, last_pts=8.88)
    monkeypatch.setattr(demux, "_last_audio_tfdt_sec", lambda: 200.0)
    sentinel = StreamAudioChunk(
        pts_start=39.0, pts_end=39.5, duration=0.5,
        pcm=np.zeros(8000, dtype=np.float32),
    )
    monkeypatch.setattr(demux, "_decode_fragments_individually", lambda: [sentinel])

    chunks = demux._decode_new()
    got_span = sum(c.duration for c in chunks)

    assert got_span >= base_span, (
        f"nhánh dự phòng làm MẤT PCM: đường chuẩn phát {base_span:.2f}s nhưng kết quả "
        f"trả về chỉ còn {got_span:.2f}s — `return per_frag` đã vứt `chunks`"
    )
    assert any(c.pts_start >= 38.0 for c in chunks), "PCM của đường dự phòng phải được giữ"


# ── 3. `tfdt` phải quy đổi theo TIMESCALE của track ───────────────────────────

def test_quy_doi_tfdt_theo_timescale_cua_track(fmp4_40s, monkeypatch):
    """`tfdt` chia cho TIMESCALE của track, KHÔNG chia cho sample rate.

    Log thật 2026-10-08 (xhamster, fMP4 muxed AV1+AAC): bộ đệm chỉ giữ ~100 s audio
    nhưng `tfdt` quy ra tới 180 s. Vì mốc bị thổi phồng, điều kiện "còn audio phía
    trước chưa lấy" (`_last_pts < last_tfdt - 2`) LUÔN đúng ⇒ đường giải mã từng-fragment
    chạy trên MỌI mảnh, mở container cho từng mảnh mà không bao giờ ra PCM, và sàn quét
    bị đẩy vượt xa mốc đã phát (cơ chế cứu hộ tự vô hiệu hoá).
    """
    init, frags = fmp4_40s
    av = pytest.importorskip("av")

    base = _demux_with_buffer(init, frags, last_pts=0.0)
    ts_base = base._audio_timescale()
    assert ts_base > 0, "phải đọc được timescale của track audio từ init segment"

    container = av.open(io.BytesIO(init))
    try:
        expect = int(round(1.0 / float(container.streams.audio[0].time_base)))
    finally:
        container.close()
    assert ts_base == expect, (
        f"timescale phải lấy từ `mdhd`/`stream.time_base` ({expect}), không phải "
        f"sample rate — đang trả {ts_base}"
    )

    # Nhân đôi timescale ⇒ CÙNG `tfdt` thô nhưng số giây phải GIẢM một nửa.
    ranges_base = base._fragment_byte_ranges()
    assert ranges_base, "điều kiện dựng: phải đọc được cặp moof+mdat"

    doubled = _demux_with_buffer(init, frags, last_pts=0.0)
    monkeypatch.setattr(doubled, "_audio_timescale", lambda: ts_base * 2)
    ranges_double = doubled._fragment_byte_ranges()
    assert ranges_double, "điều kiện dựng: phải đọc được cặp moof+mdat"

    assert ranges_double[-1][2] == pytest.approx(ranges_base[-1][2] / 2.0, rel=0.02), (
        f"`tfdt` không bám theo timescale: {ranges_base[-1][2]:.2f}s → "
        f"{ranges_double[-1][2]:.2f}s khi timescale gấp đôi"
    )
    assert base._last_audio_tfdt_sec() == pytest.approx(ranges_base[-1][2], rel=0.02), (
        "`_last_audio_tfdt_sec` phải dùng CÙNG cách quy đổi với `_fragment_byte_ranges`"
    )


def test_doi_init_thi_bo_cache_timescale(fmp4_40s):
    """Init segment mới ⇒ bỏ cache timescale/sample rate (nếu không, quy đổi sai đơn vị)."""
    init, _frags = fmp4_40s
    demux = StreamDemuxer(target_sample_rate=16000)
    demux._set_init(init)
    assert demux._audio_timescale() > 0
    assert demux._cached_timescale > 0

    demux._set_init(init + b"\x00\x00\x00\x08free")
    assert demux._cached_timescale == 0, "cache timescale phải bị xoá khi init đổi"
    assert demux._cached_sample_rate == 0, "cache sample rate phải bị xoá khi init đổi"
