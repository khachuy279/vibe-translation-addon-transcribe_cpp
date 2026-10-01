"""Trace: StreamDemuxer co giai ma DU audio khi manh MSE den tung cuc bo khong?

Mo phong:
  1. Tao file WebM/Opus 60s (nhieu Cluster, ~ cluster 1s => giong MSE that).
  2. Tach init segment (truoc Cluster dau) + body.
  3. Nap TUNG manh nho (32 KB) nhu browser appendBuffer that.
  4. In ra tung lan decode: so chunk, pts_start, pts_end, ty le phu.
"""
from __future__ import annotations

import io
import sys

import av
import numpy as np

sys.path.insert(0, ".")

from backend.core.stream_demuxer import StreamDemuxer, scan_webm_boundaries  # noqa: E402


def build_webm(duration=60.0, sr=48000, cluster_sec=1.0):
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="webm")
    st = out.add_stream("libopus", rate=sr)
    st.layout = "mono"
    st.bit_rate = 128000
    # Buoc muxer cat cluster theo kich thuoc/ thoi gian mac dinh; ta thu voi max_cluster
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
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


def main():
    data = build_webm(60.0)
    bounds = scan_webm_boundaries(data)
    print(f"file={len(data)}B clusters={len(bounds)} first_cluster@{bounds[0]}")
    if len(bounds) > 1:
        print("cluster sizes:", [bounds[i + 1] - bounds[i] for i in range(min(6, len(bounds) - 1))])

    init = data[: bounds[0]]
    body = data[bounds[0]:]

    demux = StreamDemuxer(target_sample_rate=16000)
    demux.feed(init, timestamp_offset=0.0, is_init=True, epoch=0)

    frag = 32768
    total = 0.0
    n = 0
    for i in range(0, len(body), frag):
        chunks = demux.feed(body[i:i + frag], timestamp_offset=0.0, epoch=0)
        if chunks:
            n += 1
            d = sum(c.duration for c in chunks)
            total += d
            print(
                f"  feed#{i // frag:3d} last_pts={demux.last_pts:7.2f} "
                f"chunks={len(chunks)} span=[{chunks[0].pts_start:7.2f},{chunks[-1].pts_end:7.2f}] "
                f"dur={d:5.2f}s"
            )
    print(f"TONG: {total:.2f}s / 60.00s ({100 * total / 60.0:.1f}%) qua {n} lan decode co ket qua")


if __name__ == "__main__":
    main()
