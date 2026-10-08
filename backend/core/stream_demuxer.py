"""StreamDemuxer — Bộ ghép nối & giải mã LIÊN TỤC luồng âm thanh MSE (Lookahead).

VẤN ĐỀ GỐC (đã đo thật, xem `scratch_demux_probe.py`):
    Trình duyệt (YouTube/Bilibili) KHÔNG gọi `SourceBuffer.appendBuffer()` một lần cho
    mỗi segment hoàn chỉnh. Nó append các mảnh ~32 KB nằm GIỮA cluster WebM / fragment
    fMP4. Giải mã từng mảnh độc lập bằng PyAV chỉ thu được ~35% lượng audio (12 s -> 4,2 s)
    vì mảnh cắt giữa cluster không tự parse được. Đó là nguyên nhân trực tiếp khiến mọi
    câu phụ đề Lookahead bị sai mốc thời gian và mất chữ.

GIẢI PHÁP:
    1. Giữ RIÊNG init segment (EBML header của WebM hoặc ftyp+moov của MP4) — trình duyệt
       chỉ append một lần duy nhất ở đầu mỗi SourceBuffer nên nếu không giữ lại thì không
       bao giờ giải mã được nữa.
    2. Ghép mọi mảnh media vào MỘT bộ đệm byte liên tục theo đúng thứ tự append.
    3. Mỗi lần có mảnh mới: mở lại container trên `init + media`, `seek()` tới ngay trước
       frame cuối đã phát, giải mã tiếp và CHỈ trả về các frame MỚI (PTS > mốc cuối).
       Nhờ vậy chi phí mỗi lần giải mã ~ phần audio mới, không phải toàn bộ bộ đệm.
    4. Cắt tỉa bộ đệm byte theo ranh giới cluster/fragment (không cắt giữa cluster).

Mốc thời gian trả về tuân theo đúng đặc tả MSE:
    presentation_time = container_pts + sourceBuffer.timestampOffset
"""

from __future__ import annotations

import bisect
import io
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import av
import numpy as np

from backend.utils.logger import get_logger

logger = get_logger("core.stream_demuxer")

#: Magic của EBML (WebM/Matroska).
EBML_MAGIC = b"\x1a\x45\xdf\xa3"
#: ID phần tử Segment / Cluster trong Matroska (đọc dưới dạng VINT 4 byte).
WEBM_SEGMENT_ID = 0x18538067
WEBM_CLUSTER_ID = 0x1F43B675

#: Trần bộ đệm byte media giữ trong RAM (WebM Opus 160 kbps ~ 20 KB/s ⇒ 8 MB ~ 400 s).
DEFAULT_MAX_MEDIA_BYTES = 8 * 1024 * 1024
#: Số ranh giới cluster/fragment giữ lại khi cắt tỉa.
DEFAULT_KEEP_BOUNDARIES = 60
#: Lùi lại bao nhiêu giây trước mốc cuối khi seek (bù preroll của codec).
_SEEK_BACKOFF_SEC = 0.5
#: Neo trục PCM theo `SourceBuffer.buffered` (xem `_anchor_to_browser_axis`):
#:   * ≤ MIN: coi là mép frame, KHÔNG sửa (chỉ khoá trục).
#:   * > STRONG: lệch trục thật, sửa NGAY (không chờ xác nhận).
#:   * ở giữa: chờ quan sát thứ hai khớp trong CONFIRM giây (tránh mảnh hỏng ở mép cuối).
_AXIS_ANCHOR_MIN_SEC = 0.5
_AXIS_ANCHOR_CONFIRM_SEC = 0.6
_AXIS_ANCHOR_STRONG_SEC = 5.0
#: Cửa sổ (giây) chống ĐẢO TRỤC: nếu lần neo trước vừa xảy ra trong khoảng này và lần này có độ
#: lớn tương đương nhưng NGƯỢC DẤU thì bỏ qua (xem `_anchor_to_browser_axis`). Đo thật
#: 2026-10-05 (xvideos.com): neo +10,01s rồi −10,01s sau 4s ⇒ timeline thủng 20s audio.
_AXIS_REVERSAL_GUARD_SEC = 10.0


@dataclass
class StreamAudioChunk:
    """Một đoạn PCM liên tục đã giải mã, kèm mốc thời gian tuyệt đối trên timeline video."""

    pts_start: float
    pts_end: float
    duration: float
    pcm: np.ndarray  # float32 mono @ target_sample_rate


# ──────────────────────────────────────────────────────────────────────────────
# EBML / ISO-BMFF: quét ranh giới cluster & fragment
# ──────────────────────────────────────────────────────────────────────────────

def _vint_length(first_byte: int) -> int:
    """Độ dài (byte) của một VINT theo bit đánh dấu đầu tiên. 0 = không hợp lệ."""
    for i in range(8):
        if first_byte & (0x80 >> i):
            return i + 1
    return 0


def _read_element_header(buf: bytes, pos: int) -> Optional[Tuple[int, int, int, int, bool]]:
    """Đọc header EBML tại `pos`.

    Returns:
        (element_id, id_len, data_start, size, size_unknown) hoặc None nếu dữ liệu cụt.
    """
    if pos < 0 or pos >= len(buf):
        return None
    id_len = _vint_length(buf[pos])
    if id_len == 0 or pos + id_len > len(buf):
        return None
    element_id = int.from_bytes(buf[pos:pos + id_len], "big")
    p = pos + id_len
    if p >= len(buf):
        return None
    size_len = _vint_length(buf[p])
    if size_len == 0 or p + size_len > len(buf):
        return None
    raw = int.from_bytes(buf[p:p + size_len], "big")
    mask = (1 << (7 * size_len)) - 1
    size = raw & mask
    unknown = size == mask
    return element_id, id_len, p + size_len, size, unknown


def scan_webm_boundaries(buf: bytes) -> List[int]:
    """Trả về offset (byte) của các Cluster cấp cao nhất trong `buf`.

    Chịu được dữ liệu cụt ở cuối (mảnh đang nhận dở) — chỉ trả các Cluster parse được.
    """
    out: List[int] = []
    pos = 0
    # 1. Tìm Segment ở cấp cao nhất.
    while pos < len(buf):
        head = _read_element_header(buf, pos)
        if head is None:
            break
        element_id, _id_len, data_start, size, unknown = head
        if element_id == WEBM_SEGMENT_ID:
            seg_end = len(buf) if unknown else min(len(buf), data_start + size)
            p = data_start
            while p < seg_end:
                head2 = _read_element_header(buf, p)
                if head2 is None:
                    break
                eid2, _il2, ds2, size2, unk2 = head2
                if eid2 == WEBM_CLUSTER_ID:
                    out.append(p)
                if unk2:
                    break
                p = ds2 + size2
            break
        if unknown:
            break
        pos = data_start + size
    return out


def scan_mp4_boundaries(buf: bytes, base: int = 0) -> List[int]:
    """Trả về offset (byte) của các box `moof` (đầu mỗi fragment fMP4).

    `base`: cộng thêm vào mọi offset trả về — cho phép quét một LÁT của bộ đệm lớn mà vẫn
    nhận được offset tuyệt đối (dùng để bắt đầu quét từ mảnh đang cần).
    """
    out: List[int] = []
    pos = 0
    n = len(buf)
    while pos + 8 <= n:
        size = int.from_bytes(buf[pos:pos + 4], "big")
        box_type = buf[pos + 4:pos + 8]
        if size == 1:
            if pos + 16 > n:
                break
            size = int.from_bytes(buf[pos + 8:pos + 16], "big")
            header = 16
        elif size == 0:
            size = n - pos
            header = 8
        else:
            header = 8
        if size < header:
            break
        if box_type == b"moof":
            out.append(pos + base)
        pos += size
    return out


def scan_mp4_track_at(buf: bytes, moof_off: int) -> Optional[Tuple[float, int]]:
    """(baseMediaDecodeTime, track_id) của RIÊNG `moof` tại `moof_off`. None nếu không đọc được."""
    size = int.from_bytes(buf[moof_off:moof_off + 4], "big")
    if size < 8:
        return None
    traf = _find_box(buf, moof_off + 8, size - 8, b"traf")
    if traf is None:
        return None
    tstart, tsize = traf
    decode_time = 0.0
    tfdt = _find_box(buf, tstart, tsize, b"tfdt")
    if tfdt is not None:
        dstart, dsize = tfdt
        if dsize >= 12:
            decode_time = float(int.from_bytes(buf[dstart + 4:dstart + 12], "big"))
    track_id = 0
    tfhd = _find_box(buf, tstart, tsize, b"tfhd")
    if tfhd is not None:
        hstart, hsize = tfhd
        if hsize >= 8:
            flags = int.from_bytes(buf[hstart + 1:hstart + 4], "big")
            p = hstart + 4
            if flags & 0x000001:   # base-data-offset-present
                p += 8
            if p + 4 <= hstart + hsize:
                track_id = int.from_bytes(buf[p:p + 4], "big")
    return decode_time, track_id


def scan_mp4_fragment_times(buf: bytes, limit: int = 100000) -> List[Tuple[float, int]]:
    """(baseMediaDecodeTime THEO TIMESCALE CỦA TRACK, track_id) của từng `moof` trong `buf`.

    Đây là cách DUY NHẤT đo được "dòng fragment fMP4 nhận được có liền mạch không": mỗi
    `moof` mang `tfdt.baseMediaDecodeTime` (mốc media) và `tfhd.track_ID`. Nếu hai mốc liên
    tiếp của CÙNG track cách nhau vài chục giây thì **mảnh của khoảng đó chưa bao giờ được
    append** — tức audio mất ở phía nguồn, khác hẳn với mất do logic giải mã.

    Giá trị trả về CHƯA đổi sang giây (timescale lấy từ track, xem
    `_mp4_fragment_chain_report`).
    """
    out: List[Tuple[float, int]] = []
    for off in scan_mp4_boundaries(buf)[:limit]:
        pos = off + 8
        traf = _find_box(buf, pos, int.from_bytes(buf[off:off + 4], "big") - 8, b"traf")
        if traf is None:
            continue
        tstart, tsize = traf
        tfdt = _find_box(buf, tstart, tsize, b"tfdt")
        tfhd = _find_box(buf, tstart, tsize, b"tfhd")
        decode_time = 0.0
        if tfdt is not None:
            # `_find_box` trả offset NGAY SAU header box ⇒ `dstart` là byte version/flags.
            dstart, dsize = tfdt
            if dsize >= 12:
                decode_time = float(int.from_bytes(buf[dstart + 4:dstart + 12], "big"))
        track_id = 0
        if tfhd is not None:
            # `tfhd`: [version(1)][flags(3)][track_ID(4)]... ⇒ track_ID ở `hstart + 4`.
            hstart, hsize = tfhd
            if hsize >= 8:
                flags = int.from_bytes(buf[hstart + 1:hstart + 4], "big")
                p = hstart + 4
                if flags & 0x000001:   # base-data-offset-present
                    p += 8
                if p + 4 <= hstart + hsize:
                    track_id = int.from_bytes(buf[p:p + 4], "big")
        out.append((decode_time, track_id))
    return out


def _iter_adts_frames(buf: bytes) -> List[bytes]:
    """Tách các frame ADTS (`0xFFF`) khỏi payload AAC thô. Bỏ qua byte rác giữa các frame."""
    out: List[bytes] = []
    i = 0
    n = len(buf)
    while i + 7 <= n:
        if buf[i] != 0xFF or (buf[i + 1] & 0xF0) != 0xF0:
            i += 1
            continue
        frame_len = ((buf[i + 3] & 0x03) << 11) | (buf[i + 4] << 3) | ((buf[i + 5] & 0xE0) >> 5)
        if frame_len < 7 or i + frame_len > n:
            i += 1
            continue
        out.append(buf[i:i + frame_len])
        i += frame_len
    return out


def _find_box(buf: bytes, start: int, size: int, want: bytes) -> Optional[Tuple[int, int]]:
    pos = start
    end = min(len(buf), start + size)
    while pos + 8 <= end:
        bsize = int.from_bytes(buf[pos:pos + 4], "big")
        btype = buf[pos + 4:pos + 8]
        header = 8
        if bsize == 1:
            if pos + 16 > end:
                return None
            bsize = int.from_bytes(buf[pos + 8:pos + 16], "big")
            header = 16
        elif bsize == 0:
            bsize = end - pos
        if bsize < header:
            return None
        if btype == want:
            return pos + header, bsize - header
        pos += bsize
    return None


def detect_container(data: bytes) -> str:
    """Nhận diện container: 'webm', 'mp4' hay 'unknown'."""
    if not data:
        return "unknown"
    if data.startswith(EBML_MAGIC) or data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if len(data) >= 8 and data[4:8] in (b"ftyp", b"styp", b"moof", b"moov"):
        return "mp4"
    if data[4:8] == b"ftyp":
        return "mp4"
    return "unknown"


def looks_like_init_segment(data: bytes) -> bool:
    """Đoán xem mảnh này có phải init segment (header) hay không.

    Init segment của WebM là EBML header (không chứa Cluster); của fMP4 là `ftyp`+`moov`
    (không chứa `moof`).
    """
    if not data:
        return False
    if data.startswith(EBML_MAGIC):
        # Có Cluster ⇒ đây là file đầy đủ, không thuần init.
        return not scan_webm_boundaries(data[: min(len(data), 1 << 20)])
    if len(data) >= 8 and data[4:8] in (b"ftyp", b"styp"):
        return b"moof" not in data[: min(len(data), 1 << 20)]
    return False


def _split_init_media(data: bytes, container: str) -> Tuple[bytes, bytes]:
    """Tách một mảnh vừa chứa header vừa chứa media (file WebM/MP4 đầy đủ)."""
    if container == "webm":
        bounds = scan_webm_boundaries(data)
        if bounds:
            return data[: bounds[0]], data[bounds[0]:]
        return data, b""
    if container == "mp4":
        bounds = scan_mp4_boundaries(data)
        if bounds:
            return data[: bounds[0]], data[bounds[0]:]
    return data, b""


# ──────────────────────────────────────────────────────────────────────────────
# StreamDemuxer
# ──────────────────────────────────────────────────────────────────────────────

class StreamDemuxer:
    """Ghép nối các mảnh MSE thành một luồng byte liên tục rồi giải mã tăng dần.

    Đối tượng này **có trạng thái** và **không thread-safe**; mọi lời gọi `feed()` phải
    được tuần tự hoá (handler Lookahead chạy nó trong một executor duy nhất theo thứ tự).
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
        keep_boundaries: int = DEFAULT_KEEP_BOUNDARIES,
    ):
        self.target_sample_rate = int(target_sample_rate)
        self.max_media_bytes = int(max_media_bytes)
        self.keep_boundaries = int(keep_boundaries)

        self._init: bytes = b""
        self._media: bytearray = bytearray()
        self._container: str = "unknown"
        self._epoch: int = 0
        self._min_pts: Optional[float] = None
        self._last_pts: float = float("-inf")
        #: Sàn QUÉT của đường giải mã từng-fragment — xem `_decode_fragments_individually`.
        #: Chỉ nghĩa "đã thử và không ra PCM", KHÔNG nghĩa "đã phát".
        self._indiv_scan_floor: float = float("-inf")
        self._latest_ts_offset: float = 0.0
        #: `timestampOffset` suy ra từ `SourceBuffer.buffered` (extension gửi kèm) — xem
        #: `_anchor_to_browser_axis`. Khi đã có, nó THAY THẾ `_latest_ts_offset` làm trục giải mã
        #: vì đây mới là trục thật của trình duyệt.
        self._browser_axis_bias: Optional[float] = None
        #: Quan sát lệch trục gần nhất (chờ xác nhận lần 2 trước khi sửa) — xem
        #: `_anchor_to_browser_axis`.
        self._axis_shift_obs: Optional[float] = None
        self._axis_shift_logged: bool = False
        #: Lần neo trục GẦN NHẤT (độ lớn có dấu + thời điểm) — dùng để chặn ĐẢO TRỤC.
        self._axis_shift_signed: float = 0.0
        self._axis_shift_signed_at: float = 0.0
        self._lock = threading.RLock()

        # Thống kê
        self.fragments: int = 0
        self.decoded_seconds: float = 0.0
        self.decode_time_sec: float = 0.0
        self._seek_fallbacks: int = 0
        #: Chẩn đoán mất audio (2026-10-01): `emitted_sec` = tổng audio ĐÃ PHÁT ra PCM;
        #: `frames_skipped` = frame bị lọc vì cũ hơn mốc đã phát; `gaps_in_media` = số khe hở
        #: thật giữa hai frame trong dữ liệu nhận được (dấu hiệu mảnh bị cắt/mất).
        self._emitted_sec: float = 0.0
        self._frames_skipped: int = 0
        self._gaps_in_media: int = 0
        #: `tfdt.baseMediaDecodeTime` của fragment ĐẦU TIÊN (mốc media của mẫu đầu bộ đệm) và
        #: số lần phải dùng đường dự phòng bóc AAC thô.
        self._first_audio_tfdt: Optional[float] = None
        self._raw_fallback_used: int = 0
        self._raw_fallback_rejected: int = 0
        self._frag_decode_used: int = 0
        #: Cache [(tfdt, track_id)] của fragment audio + sample rate (tránh quét lại mỗi lượt).
        self._frag_times_cache: Optional[List[Tuple[float, int]]] = None
        self._frag_times_key: int = -1
        self._cached_sample_rate: int = 0
        #: Timescale của track audio (đơn vị `tfdt`) — xem `_audio_timescale`.
        self._cached_timescale: int = 0
        self._offsets_cache: List[int] = []
        self._offsets_key: int = -1

    # ------------------------------------------------------------------ quản trị
    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def container(self) -> str:
        return self._container

    @property
    def buffered_bytes(self) -> int:
        return len(self._media)

    @property
    def last_pts(self) -> float:
        return self._last_pts

    def _axis_offset(self) -> float:
        """Bias đang dùng để đưa PTS container về trục thời gian của TRÌNH DUYỆT.

        Ưu tiên bias HỌC ĐƯỢC từ `SourceBuffer.buffered` (`_anchor_to_browser_axis`); chỉ khi
        chưa học được mới dùng `timestampOffset` do extension báo.
        """
        if self._browser_axis_bias is not None:
            return float(self._browser_axis_bias)
        return float(self._latest_ts_offset)

    def _set_init(self, init_part: bytes) -> None:
        """Gán init segment MỚI và bỏ cache suy ra từ nó.

        Cache `sample_rate`/`timescale` được đọc từ chính `_init`; init đổi (SourceBuffer
        mới, nguồn nạp lại) mà giữ cache thì mọi `tfdt` sau đó quy đổi sai đơn vị.
        """
        if init_part == self._init:
            return
        self._init = init_part
        self._cached_sample_rate = 0
        self._cached_timescale = 0

    def reset(self, epoch: Optional[int] = None, min_pts: Optional[float] = None) -> None:
        """Xoá sạch trạng thái (tua video / đổi SourceBuffer / init segment mới)."""
        with self._lock:
            self._media.clear()
            if min_pts is not None:
                self._min_pts = float(min_pts)
            self._last_pts = float(self._min_pts) if self._min_pts is not None else float("-inf")
            self._indiv_scan_floor = float("-inf")
            self._latest_ts_offset = 0.0
            if epoch is not None and int(epoch) != self._epoch:
                # SourceBuffer MỚI ⇒ luồng mới ⇒ mọi neo trục cũ không còn giá trị.
                self._browser_axis_bias = None
                self._axis_shift_obs = None
                self._axis_shift_logged = False
                self._axis_shift_signed = 0.0
                self._axis_shift_signed_at = 0.0
            if epoch is not None:
                self._epoch = int(epoch)
            else:
                self._epoch += 1
            # Init segment VẪN được giữ: SourceBuffer không đổi thì header không đổi.
            logger.info(
                f"StreamDemuxer reset (epoch={self._epoch}, min_pts={self._min_pts}, init={len(self._init)}B, "
                f"container={self._container})",
                extra={"module_tag": "WS"},
            )

    def set_min_pts(self, min_pts: Optional[float]) -> None:
        """Đặt mốc PTS tối thiểu (bỏ qua toàn bộ frame trước mốc này)."""
        with self._lock:
            if min_pts is not None:
                val = float(min_pts)
                self._min_pts = val
                if val > self._last_pts:
                    self._last_pts = val
            else:
                self._min_pts = None

    # ------------------------------------------------------------------ ingress
    def feed(
        self,
        raw_bytes: bytes,
        timestamp_offset: float = 0.0,
        mime_type: str = "auto",
        is_init: bool = False,
        epoch: Optional[int] = None,
        media_start: Optional[float] = None,
        media_end: Optional[float] = None,
    ) -> List[StreamAudioChunk]:
        """Nạp một mảnh MSE và trả về các đoạn PCM MỚI đã giải mã được.

        Args:
            raw_bytes: byte thô của `SourceBuffer.appendBuffer()`.
            timestamp_offset: `sourceBuffer.timestampOffset` tại thời điểm append.
            mime_type: gợi ý định dạng (chỉ dùng cho log).
            is_init: client khẳng định đây là init segment.
            epoch: số thứ tự SourceBuffer; đổi epoch ⇒ reset bộ đệm ghép nối.
            media_start/media_end: khoảng THẬT (giây, trục thời gian của trình duyệt) mà mảnh
                này chiếm trong `SourceBuffer.buffered`. Dùng để NEO lại PCM khi mốc container
                không khớp `video.currentTime` — xem `_anchor_to_browser_axis`.
        """
        if not raw_bytes:
            return []

        with self._lock:
            if epoch is not None and int(epoch) != self._epoch:
                self._media.clear()
                self._epoch = int(epoch)
                self._last_pts = float(self._min_pts) if self._min_pts is not None else float("-inf")
                self._indiv_scan_floor = float("-inf")
                self._browser_axis_bias = None
                self._axis_shift_obs = None
                self._axis_shift_logged = False
                self._axis_shift_signed = 0.0
                self._axis_shift_signed_at = 0.0
                logger.info(
                    f"StreamDemuxer: SourceBuffer epoch mới = {self._epoch} (min_pts={self._min_pts}) — xoá bộ đệm ghép nối",
                    extra={"module_tag": "WS"},
                )

            self.fragments += 1
            self._latest_ts_offset = float(timestamp_offset or 0.0)
            container = detect_container(raw_bytes)
            if self._container == "unknown" and container != "unknown":
                self._container = container
            # Ghi mốc media của fragment ĐẦU TIÊN (dùng làm neo cho đường dự phòng AAC thô).
            if self._first_audio_tfdt is None and container == "mp4":
                try:
                    times = scan_mp4_fragment_times(raw_bytes, limit=4)
                    if times:
                        self._first_audio_tfdt = float(times[0][0])
                except Exception:  # noqa: BLE001
                    pass

            # Định dạng TỰ CHỨA (WAV/MP3/…): không ghép vào bộ đệm byte — mỗi mảnh là một
            # file hoàn chỉnh, ghép lại sẽ hỏng header. Giải mã độc lập.
            if container == "unknown" and not self._media and not self._init:
                return self._anchor_to_browser_axis(
                    self._decode_standalone(raw_bytes, self._axis_offset()),
                    media_start,
                    media_end,
                )

            treat_as_init = bool(is_init) or looks_like_init_segment(raw_bytes)
            if treat_as_init and self._init != raw_bytes:
                init_part, media_part = _split_init_media(raw_bytes, container)
                if init_part:
                    self._set_init(init_part)
                if media_part:
                    self._media.extend(media_part)
            elif self._init and raw_bytes.startswith(EBML_MAGIC):
                # Init đến lần nữa (replay sau khi kết nối lại) — thay thế, không nhân đôi.
                init_part, media_part = _split_init_media(raw_bytes, container)
                if init_part:
                    self._set_init(init_part)
                if media_part:
                    self._media.extend(media_part)
            else:
                self._media.extend(raw_bytes)

            self._trim()

            if not self._init and not self._media:
                return []

            chunks = self._decode_new()
            if not chunks and media_end is not None and self._browser_axis_bias is None:
                # ── HỌC LẠI TRỤC TỪ "DẤU VẾT TRÌNH DUYỆT" (chỉ khi CHƯA có neo trục) ──
                # SỰ CỐ THẬT 2026-10-05 (xhamster.com): mở PHIÊN MỚI giữa video (hoặc trang tự
                # nạp lại nguồn) làm PTS trong container quay về 0 trong khi trục trình duyệt
                # đang ở ~984s. `_browser_axis_bias` CHƯA học được ở phiên mới ⇒ mọi frame bị
                # frontier (`min_pts`) lọc sạch ⇒ `chunks` luôn rỗng ⇒ KHÔNG BAO GIỜ học được
                # trục ⇒ "Audio RAM: 0.0s" suốt phiên ("tắt mở lại session cũng không được").
                #
                # Cách phá vòng luẩn quẩn: extension gửi kèm `media_start`/`media_end` = khoảng
                # THẬT trong `SourceBuffer.buffered` của mảnh này. Byte này KẾT THÚC ở `media_end`
                # ⇒ hiệu `media_end − pts_end(container)` chính là bias cần học.
                raw_end = self._probe_last_frame_pts_end()
                if raw_end is not None:
                    learned = float(media_end) - float(raw_end)
                    self._browser_axis_bias = learned
                    self._axis_shift_obs = None
                    self._last_pts = float(self._min_pts) if self._min_pts is not None else float("-inf")
                    chunks = self._decode_new()
                    if chunks:
                        logger.warning(
                            f"HỌC LẠI TRỤC THỜI GIAN từ dấu vết trình duyệt: byte kết thúc ở "
                            f"{float(media_end):.2f}s (trục trình duyệt) nhưng PTS container kết "
                            f"thúc ở {float(raw_end):.2f}s ⇒ bias {learned:+.2f}s. "
                            f"Đã giải mã được {len(chunks)} đoạn PCM.",
                            extra={"module_tag": "WS"},
                        )
                    else:
                        # Học xong vẫn không ra PCM ⇒ trả lại trạng thái cũ, không để lại một
                        # neo sai làm hỏng cả các mảnh sau.
                        self._browser_axis_bias = None
            return self._anchor_to_browser_axis(chunks, media_start, media_end)

    def _probe_last_frame_pts_end(self) -> Optional[float]:
        """PTS KẾT THÚC (giây, trục CONTAINER) của frame cuối trong bộ đệm hiện tại.

        Dùng để HỌC LẠI trục thời gian khi chưa có neo nào (xem khối "HỌC LẠI TRỤC" trong
        `feed`). Chỉ giải mã để ĐỌC MỐC, không giữ PCM, nên không ảnh hưởng frontier.
        Trả `None` nếu không mở được container hoặc không có frame nào.
        """
        payload = self._init + bytes(self._media)
        if not payload:
            return None
        last_end: Optional[float] = None
        try:
            container = av.open(io.BytesIO(payload))
        except Exception:  # noqa: BLE001
            return None
        try:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            tb = float(stream.time_base or 0)
            if tb <= 0:
                return None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                start = float(frame.pts) * tb
                rate = float(getattr(frame, "sample_rate", 0) or 0)
                dur = (float(frame.samples) / rate) if (rate > 0 and frame.samples) else 0.02
                last_end = start + dur
        except Exception:  # noqa: BLE001
            return last_end
        return last_end

    def _anchor_to_browser_axis(
        self,
        chunks: List[StreamAudioChunk],
        media_start: Optional[float],
        media_end: Optional[float],
    ) -> List[StreamAudioChunk]:
        """Neo PCM vào TRỤC THỜI GIAN CỦA TRÌNH DUYỆT (`SourceBuffer.buffered`).

        Vì sao cần: `container_pts + timestampOffset` chỉ trùng trục media của trình duyệt khi
        trang dùng `SourceBuffer.mode = "segments"` VÀ báo đúng `timestampOffset`. Rất nhiều
        trang khác:
          * chèn segment có `tfdt`/timecode TUYỆT ĐỐI (mốc chương trình) trong khi timeline
            trình duyệt bắt đầu từ 0 (chế độ `"sequence"`, hoặc `timestampOffset` âm),
          * hoặc đổi `timestampOffset` NGAY SAU `appendBuffer` (extension đọc được giá trị cũ).
        Khi đó PCM rơi lệch ĐÚNG BẰNG offset đó. Đo thật 2026-10-03 (trang ngoài YouTube):
        PCM giải mã ra ở 250,0→306,8 s trong khi playhead ở 7,1 s ⇒ `Đệm trước: 299,7 s`,
        `Đã dịch: 0,0 s`, video kẹt ở mốc cũ dù cache interceptor đã có đủ audio 0→56,6 s.

        Cách làm: extension gửi kèm `media_start`/`media_end` = khoảng THẬT trong `buffered`
        sau khi append. Hiệu `media_end − pts_end(PCM cuối)` là một HẰNG SỐ cho cả một
        SourceBuffer (một epoch) nên chỉ cần học một lần; nếu luồng đổi base giữa chừng thì
        hiệu số đó lại xuất hiện và được học lại. Lệch nhỏ (≤ 0,5 s) coi là mép frame.
        """
        if media_end is None or not chunks:
            return chunks
        try:
            observed = float(media_end) - float(chunks[-1].pts_end)
        except (TypeError, ValueError):
            return chunks
        if not math.isfinite(observed):
            return chunks

        if abs(observed) <= _AXIS_ANCHOR_MIN_SEC:
            if self._browser_axis_bias is None:
                # Trục container đã đúng ⇒ khoá lại để các fragment sau không đổi trục.
                self._browser_axis_bias = self._latest_ts_offset
                logger.info(
                    f"Trục PCM khớp trục media của trình duyệt (lệch {observed:+.2f}s) — khoá trục.",
                    extra={"module_tag": "WS"},
                )
            self._axis_shift_obs = None
            return chunks

        # ── KIỂM TRA TÍNH LIÊN TỤC THEO `media_start` ───────────────────────────
        # Khi trục đã khoá (self._browser_axis_bias is not None) và có `media_start`:
        # Nếu media_start khớp với điểm bắt đầu của chunk (hoặc khớp với điểm kết thúc
        # của chunk), tức là luồng audio hoàn toàn LIÊN TỤC với trục hiện tại.
        # Sai khác `observed = media_end - chunks[-1].pts_end` thực chất chỉ là độ dài
        # vùng đệm trước (lead time) mà trình duyệt vừa nạp thêm vào SourceBuffer.
        # TUYỆT ĐỐI KHÔNG được coi đó là lệch trục thời gian (sẽ tạo khe hở mất phụ đề).
        if self._browser_axis_bias is not None and media_start is not None:
            try:
                m_start = float(media_start)
                is_continuous = (
                    abs(m_start - float(chunks[0].pts_start)) <= _AXIS_ANCHOR_MIN_SEC
                    or abs(m_start - float(chunks[-1].pts_end)) <= _AXIS_ANCHOR_MIN_SEC
                )
                if is_continuous:
                    self._axis_shift_obs = None
                    return chunks
            except (TypeError, ValueError):
                pass

        previous = self._axis_shift_obs
        confirmed = previous is not None and abs(observed - previous) <= _AXIS_ANCHOR_CONFIRM_SEC
        # Lệch LỚN (> _AXIS_ANCHOR_STRONG_SEC) là lỗi trục thật, sửa ngay; lệch vừa phải chờ
        # xác nhận lần 2 để không sửa oan vì một mảnh hỏng ở mép cuối.
        if not confirmed and abs(observed) <= _AXIS_ANCHOR_STRONG_SEC:
            self._axis_shift_obs = observed
            return chunks

        # ── CHỐNG ĐẢO TRỤC (đo thật 2026-10-05, xvideos.com — HLS fMP4 tách track) ──────
        # Log thật: `LỆCH TRỤC +10,01s` lúc 22:00:46 rồi `LỆCH TRỤC −10,01s` lúc 22:00:50.
        # Mỗi lần đảo, PCM của các fragment sau được đặt LÙI 10s so với thực tế ⇒ rơi vào vùng
        # đã có audio ⇒ bị frontier (`_last_pts`) lọc ⇒ audio của vùng thật (20→41s) MẤT HẲN.
        # Timeline thủng một khe 20s; con trỏ khối ASR chạy tới mép khe (20.04s) rồi kẹt vì
        # `available = 0.0s` — đúng hiện tượng "chạy một chút lại dừng" của người dùng.
        #
        # Vì sao đảo được: `observed` so `media_end` (khoảng vừa thêm vào `SourceBuffer`) với
        # mốc PCM cuối của TOÀN BỘ frame còn trong bộ đệm ghép nối; hai đại lượng chỉ khớp nhau
        # khi tập frame giải mã đúng bằng mảnh vừa append. Với luồng HLS tách track (mỗi segment
        # 10s), sai khác đó xuất hiện đúng bằng một segment và ĐỔI DẤU giữa hai lần.
        # ⇒ Nếu lần neo trước vừa xảy ra và độ lớn y hệt nhưng NGƯỢC DẤU thì bỏ qua lần này.
        now = time.perf_counter()
        reversed_shift = (
            self._axis_shift_signed_at > 0.0
            and self._axis_shift_signed != 0.0
            and observed * self._axis_shift_signed < 0.0
            and abs(abs(observed) - abs(self._axis_shift_signed)) <= _AXIS_ANCHOR_CONFIRM_SEC
            and (now - self._axis_shift_signed_at) <= _AXIS_REVERSAL_GUARD_SEC
        )
        if reversed_shift:
            self._axis_shift_obs = None
            logger.warning(
                f"Bỏ qua ĐẢO TRỤC: lần neo trước {self._axis_shift_signed:+.2f}s cách đây "
                f"{now - self._axis_shift_signed_at:.1f}s, lần này ngược dấu {observed:+.2f}s "
                f"(cùng độ lớn) ⇒ giữ nguyên trục đang dùng để KHÔNG tạo khe hở audio.",
                extra={"module_tag": "WS"},
            )
            return chunks

        used = self._axis_offset()
        self._browser_axis_bias = used + observed
        self._axis_shift_signed = float(observed)
        self._axis_shift_signed_at = now
        for chunk in chunks:
            chunk.pts_start += observed
            chunk.pts_end += observed
        self._last_pts += observed
        self._axis_shift_obs = None
        logger.warning(
            f"LỆCH TRỤC THỜI GIAN: PCM giải mã lệch {observed:+.2f}s so với timeline trình duyệt "
            f"— đã neo lại theo `SourceBuffer.buffered` (trục mới = container + "
            f"{self._browser_axis_bias:+.2f}s).",
            extra={"module_tag": "WS"},
        )
        self._axis_shift_logged = True
        return chunks

    # ------------------------------------------------------------------ nội bộ
    def _trim(self) -> None:
        """Giữ bộ đệm byte trong trần RAM, cắt theo ĐÚNG ranh giới cluster/fragment.

        ⚠️ TUYỆT ĐỐI không cắt thô giữa cấu trúc: một lần cắt sai làm `scan_*_boundaries`
        không tìm thấy ranh giới nào nữa ⇒ CẢ HAI đường giải mã đều chết vĩnh viễn (đo thật
        2026-10-01: bộ đệm 8 MB, `n_moof=0`, `frames=0`, `emitted` đứng yên). Thà giữ bộ đệm
        lớn hơn trần một chút còn hơn làm hỏng luồng byte.
        """
        if len(self._media) <= self.max_media_bytes:
            return
        # Giữ 3/4 trần: cắt bớt một lần rồi thôi, không cắt lại ở mọi mảnh.
        target = int(self.max_media_bytes * 0.75)
        snapshot = bytes(self._media)
        bounds = (
            scan_mp4_boundaries(snapshot)
            if self._container == "mp4"
            else scan_webm_boundaries(snapshot)
        )
        # Chỉ cắt tại ranh giới thật và phải chừa lại ít nhất một ranh giới trong bộ đệm.
        usable = [b for b in bounds if b <= len(self._media) - target]
        if not usable:
            # Không có ranh giới an toàn ⇒ GIỮ NGUYÊN (chỉ cảnh báo), vì cắt thô sẽ phá luồng.
            logger.warning(
                f"Bộ đệm ghép nối {len(self._media)}B vượt trần nhưng không tìm được ranh "
                f"giới cluster/fragment an toàn — giữ nguyên thay vì cắt hỏng luồng byte.",
                extra={"module_tag": "WS"},
            )
            return
        cut = usable[-1]
        if cut <= 0:
            return
        del self._media[:cut]
        # Mọi offset/mốc đã cache theo độ dài bộ đệm đều không còn hợp lệ.
        self._offsets_cache = []
        self._offsets_key = -1
        self._frag_times_cache = None
        self._frag_times_key = -1

    def _decode_standalone(self, raw_bytes: bytes, ts_offset: float) -> List[StreamAudioChunk]:
        """Giải mã một mảnh TỰ CHỨA (WAV/MP3/…): PTS = container_pts + timestampOffset."""
        chunks: List[StreamAudioChunk] = []
        try:
            container = av.open(io.BytesIO(raw_bytes))
        except Exception:  # noqa: BLE001
            return []
        try:
            if not container.streams.audio:
                return []
            stream = container.streams.audio[0]
            time_base = float(stream.time_base or 0) or None
            resampler = av.AudioResampler(format="flt", layout="mono", rate=self.target_sample_rate)
            run_pts: Optional[float] = None
            run_frames: List[np.ndarray] = []

            def flush_run() -> None:
                if run_frames and run_pts is not None:
                    pcm = np.concatenate(run_frames) if len(run_frames) > 1 else run_frames[0]
                    if pcm.size:
                        end = run_pts + pcm.size / float(self.target_sample_rate)
                        chunks.append(StreamAudioChunk(
                            pts_start=run_pts + ts_offset,
                            pts_end=end + ts_offset,
                            duration=end - run_pts,
                            pcm=pcm,
                        ))

            for frame in container.decode(stream):
                pts = float(frame.pts) * time_base if (frame.pts is not None and time_base) else 0.0
                if run_pts is None:
                    run_pts = pts
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray()
                    arr = arr.reshape(-1) if arr.ndim > 1 else arr
                    if arr.size:
                        run_frames.append(arr.astype(np.float32, copy=True))
            try:
                for resampled in resampler.resample(None):
                    arr = resampled.to_ndarray()
                    arr = arr.reshape(-1) if arr.ndim > 1 else arr
                    if arr.size:
                        run_frames.append(arr.astype(np.float32, copy=True))
            except Exception:  # noqa: BLE001
                pass
            flush_run()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Không giải mã được mảnh tự chứa: {exc}", extra={"module_tag": "WS"})
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass

        self.decoded_seconds += sum(c.duration for c in chunks)
        return chunks

    def _decode_new(self) -> List[StreamAudioChunk]:
        """Giải mã `init + media` và chỉ trả các frame MỚI hơn phần đã phát ra PCM.

        Ba quy tắc chống-mất-audio (đúc kết từ log thật 2026-10-01, khi mỗi lượt giải mã
        chỉ thu được ~4,1 s trong khi ~20 s audio đã về ⇒ phụ đề chỉ hiện 1/3-5 câu):

        1. **Mọi phép so mốc đều trên CÙNG một trục thời gian** = `container_pts + bias`.
           `self._last_pts` là mốc đã PHÁT (đã cộng `bias`). Seek mà trừ `bias` còn filter
           thì không là tự tạo khoảng lệch đúng bằng `timestampOffset` của SourceBuffer —
           với trang có offset lớn, mọi frame mới đều bị coi là "cũ" và bị bỏ.
        2. **`_last_pts` chỉ tiến khi PCM THỰC SỰ được phát ra.** Trước đây nó tiến theo
           `run_end` của MỌI frame (kể cả frame mà resampler chưa nhả mẫu nào) ⇒ mốc tiến
           vượt phần audio đã có, và vì lượt sau lọc `pts < _last_pts`, phần bị vượt đó bị
           bỏ VĨNH VIỄN.
        3. **Cắt phần chồng lấn ở mép frame** để mốc chỉ tiến đúng số mẫu đã phát (không
           phát lặp 20 ms mỗi lượt).
        """
        t0 = time.perf_counter()
        payload = self._init + bytes(self._media)
        chunks: List[StreamAudioChunk] = []

        try:
            container = av.open(io.BytesIO(payload))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Không mở được container ({len(payload)}B): {exc}", extra={"module_tag": "WS"})
            return []

        try:
            if not container.streams.audio:
                return []
            stream = container.streams.audio[0]
            time_base = stream.time_base
            if time_base is None:
                return []
            tb = float(time_base)
            #: Mọi mốc dưới đây nằm trên trục "giây media" = `container_pts + timestampOffset`
            #: (đúng đặc tả MSE). `container_pts` MỘT MÌNH chỉ đúng khi `timestampOffset = 0`
            #: (YouTube); với SourceBuffer có offset, trộn hai trục sẽ tạo khoảng lệch đúng
            #: bằng offset ⇒ mọi frame mới bị coi là cũ và bị bỏ.
            bias = self._axis_offset()
            frontier = self._last_pts

            if frontier > float("-inf"):
                # `frontier` nằm trên trục media (đã cộng bias) còn `container.seek()` nhận
                # mốc trên trục CONTAINER ⇒ phải trừ bias, nếu không sẽ seek vượt cuối
                # container (trang neo segment bằng `timestampOffset` lớn) và mất gần hết audio.
                back = max(0.0, frontier - bias - _SEEK_BACKOFF_SEC)
                try:
                    container.seek(int(back / tb), backward=True, stream=stream)
                except Exception:  # noqa: BLE001
                    self._seek_fallbacks += 1

            resampler = av.AudioResampler(
                format="flt", layout="mono", rate=self.target_sample_rate
            )
            prev_pts: Optional[float] = None
            run_pts: Optional[float] = None
            run_end: Optional[float] = None
            run_frames: List[np.ndarray] = []
            #: Mốc ĐÃ PHÁT cuối cùng (trong lượt này hoặc lượt trước). Mọi phép cắt chồng lấn
            #: phải so với mốc này, KHÔNG so với `self._last_pts` — `_last_pts` được cập nhật
            #: ngay khi flush nên nếu dùng nó để cắt thì chunk vừa tạo sẽ bị cắt mất toàn bộ.
            final_pts = float(self._last_pts) if self._last_pts > float("-inf") else None
            stats = {"emitted": 0.0}

            def flush_run() -> None:
                """Chốt đoạn đã gom: cắt chồng lấn mép đầu rồi phát ra PCM."""
                nonlocal run_frames, final_pts
                if not run_frames or run_pts is None:
                    return
                pcm = np.concatenate(run_frames) if len(run_frames) > 1 else run_frames[0]
                start = run_pts
                if final_pts is not None and start < final_pts:
                    trim = int(round((final_pts - start) * self.target_sample_rate))
                    if trim >= pcm.size:
                        return
                    if trim > 0:
                        pcm = pcm[trim:]
                        start = final_pts
                if pcm.size <= 0:
                    return
                # `pts_end` suy từ ĐỘ DÀI PCM thực để chunk luôn tự nhất quán (timeline phía
                # sau đối chiếu số mẫu, không tin `frame.samples`).
                end = start + float(pcm.size) / float(self.target_sample_rate)
                chunks.append(
                    StreamAudioChunk(
                        pts_start=start,
                        pts_end=end,
                        duration=end - start,
                        pcm=np.ascontiguousarray(pcm),
                    )
                )
                self._last_pts = max(self._last_pts, end)
                self._emitted_sec += end - start
                stats["emitted"] += end - start
                final_pts = end

            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                # `+ bias` đưa frame về ĐÚNG trục thời gian mà `_last_pts` đang dùng.
                pts_media = float(frame.pts) * tb + bias
                # `_last_pts` là mốc KẾT THÚC (exclusive) của audio đã phát. Frame bắt đầu
                # ĐÚNG tại mốc đó là frame MỚI ⇒ phải dùng `<` chứ không phải `<=`, nếu
                # không sẽ bỏ sót đúng một nửa số frame (đã đo: WebM 32 KB/mảnh chỉ thu
                # được ~50% audio vì lý do này).
                if pts_media < self._last_pts - 1e-6:
                    self._frames_skipped += 1
                    continue

                # Độ dài frame suy từ CHÍNH frame đầu vào (PTS kế tiếp - PTS hiện tại),
                # KHÔNG suy từ số mẫu PCM ra của resampler: resampler có bộ đệm nội bộ nên
                # vài frame đầu trả về RỖNG.
                rate = float(getattr(frame, "sample_rate", 0) or 0)
                if rate > 0 and frame.samples:
                    frame_dur = float(frame.samples) / rate
                elif prev_pts is not None:
                    frame_dur = max(0.001, pts_media - prev_pts)
                else:
                    frame_dur = 0.02
                prev_pts = pts_media

                if run_end is not None and abs(pts_media - run_end) > 0.02:
                    # Khe hở thật trong dữ liệu ⇒ tách thành 2 đoạn rời.
                    flush_run()
                    run_frames = []
                    run_pts = None
                    run_end = None
                    self._gaps_in_media += 1
                if run_pts is None:
                    # Điểm bắt đầu của đoạn = mép frame, KHÔNG được sớm hơn mốc đã phát.
                    run_pts = pts_media if final_pts is None else max(pts_media, final_pts)

                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray()
                    arr = arr.reshape(-1) if arr.ndim > 1 else arr
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    if arr.size:
                        run_frames.append(arr.copy())

                run_end = pts_media + frame_dur

            # Xả nốt phần mẫu còn lại trong resampler (nếu bỏ bước này, mỗi lượt mất ~1 frame
            # ở mép cuối ⇒ dòng PCM bị "răng cưa" và phải bù silence liên tục).
            try:
                for resampled in resampler.resample(None):
                    arr = resampled.to_ndarray()
                    arr = arr.reshape(-1) if arr.ndim > 1 else arr
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    if arr.size:
                        run_frames.append(arr.copy())
            except Exception:  # noqa: BLE001
                pass
            flush_run()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Lỗi giải mã luồng liên tục: {exc}", extra={"module_tag": "WS"})
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass

        elapsed = time.perf_counter() - t0
        self.decode_time_sec += elapsed
        self.decoded_seconds += sum(c.duration for c in chunks)

        # ── DỰ PHÒNG fMP4: giải mã TỪNG fragment độc lập ───────────────────────────
        # Đường chuẩn nối mọi mảnh thành MỘT container; nếu có một `moof` hỏng/không khớp,
        # PyAV dừng ở đó và phần sau KHÔNG BAO GIỜ được đọc (triệu chứng thật: bộ đệm đặc
        # `duration/span=1.00`, `skipped_frames=0` mà mỗi lượt chỉ ra 1,5–8 s PCM). Giải mã
        # riêng từng cặp `moof`+`mdat` (ghép với init) thì một mảnh hỏng chỉ mất chính nó.
        last_tfdt = self._last_audio_tfdt_sec()
        if self._container == "mp4" and self._init and not (
            last_tfdt > float("-inf") and self._last_pts >= last_tfdt - 2.0
        ):
            per_frag = self._decode_fragments_individually()
            if per_frag:
                self._frag_decode_used += 1
                if self._frag_decode_used == 1:
                    logger.warning(
                        f"Đường giải mã nối-liền không ra PCM — đã chuyển sang giải mã TỪNG "
                        f"fragment ({sum(c.duration for c in per_frag):.1f}s).",
                        extra={"module_tag": "WS"},
                    )
                self.decoded_seconds += sum(c.duration for c in per_frag)
                self._emitted_sec += sum(c.duration for c in per_frag)
                self._last_pts = max(self._last_pts, per_frag[-1].pts_end)
                # ── NỐI THÊM, KHÔNG THAY THẾ ──────────────────────────────────────
                # `_decode_fragments_individually` lọc frame theo `_last_pts` ĐỌC SAU khi
                # đường chuẩn đã chạy, nên `per_frag` chỉ chứa audio NẰM SAU phần `chunks`
                # vừa phát — hai tập RỜI NHAU. Trả `per_frag` không thôi là vứt toàn bộ PCM
                # mà đường chuẩn vừa giải mã được (đo được: 31,11s → 0,50s).
                return chunks + per_frag

        # ── DỰ PHÒNG fMP4/AAC ─────────────────────────────────────────────────────
        # Nếu đường chuẩn không lấy ra được gì trong khi bộ đệm ĐÃ có byte media, thử bóc
        # thẳng frame AAC từ `mdat`. Chỉ chạy khi thực sự cần nên không tốn CPU lúc bình thường.
        if not chunks and len(self._media) > 64 * 1024 and self._container == "mp4":
            fallback = self._decode_raw_aac()
            if fallback:
                self._raw_fallback_used += 1
                self.decoded_seconds += sum(c.duration for c in fallback)
                self._last_pts = max(self._last_pts, fallback[-1].pts_end)
                self._emitted_sec += sum(c.duration for c in fallback)
                if self._raw_fallback_used == 1:
                    logger.warning(
                        f"Đường giải mã chuẩn không ra PCM — đã chuyển sang bóc frame AAC thô "
                        f"từ mdat ({sum(c.duration for c in fallback):.1f}s).",
                        extra={"module_tag": "WS"},
                    )
                return fallback
        return chunks

    def _mp4_fragment_chain_report(self, sample_rate_hz: Optional[int]) -> str:
        """Tính LIỀN MẠCH của dòng fragment fMP4 (chuyển `tfdt` sang giây bằng `audio_tb`).

        Đây là phép đo quyết định để phân biệt hai nguyên nhân hoàn toàn khác nhau:
          * Có KHE HỞ lớn giữa hai `moof` ⇒ mảnh của khoảng đó **chưa bao giờ được append**
            (mất ở phía nguồn/extension), không phải lỗi giải mã.
          * Không có khe hở ⇒ byte liền mạch mà decoder vẫn không lấy ra hết ⇒ lỗi ở tầng giải mã.
        """
        try:
            times = scan_mp4_fragment_times(bytes(self._media))
            if not times:
                return "n_moof=0"
            # CHỈ xét fragment của track AUDIO: nếu dòng có xen kẽ track khác (video), so mốc
            # giữa hai `moof` khác track là vô nghĩa và sẽ báo khe hở giả.
            tracks: Dict[int, int] = {}
            for _d, t in times:
                tracks[t] = tracks.get(t, 0) + 1
            audio_track = max(tracks, key=lambda k: tracks[k])
            scale = 1.0 / (float(sample_rate_hz or 0) or 1.0)
            secs = [d * scale for d, t in times if t == audio_track]
            if len(secs) < 2:
                return f"n_moof={len(times)}, tracks={tracks} (thiếu fragment audio liền kề)"
            gaps = [(secs[i] - secs[i - 1], secs[i - 1], secs[i]) for i in range(1, len(secs))]
            # `tfdt` là mốc BẮT ĐẦU của fragment ⇒ hiệu hai mốc liên tiếp chính là ĐỘ DÀI
            # fragment bình thường (2 s với `frag_duration` 2 s), KHÔNG phải khe hở. Chỉ coi là
            # khe hở khi hiệu đó lớn hơn hẳn độ dài điển hình (trung vị) của dòng.
            deltas = sorted(g[0] for g in gaps)
            typical = deltas[len(deltas) // 2] if deltas else 0.0
            threshold = max(1.5, typical * 1.8)
            big = [g for g in gaps if g[0] > threshold]
            out = (
                f"n_moof={len(times)}, tracks={tracks}, "
                f"tfdt_giây=[{secs[0]:.2f}, {secs[-1]:.2f}], "
                f"bước_điển_hình={typical:.2f}s"
            )
            if big:
                # Khe hở thật = phần vượt quá độ dài fragment điển hình.
                missing = sum(g[0] - typical for g in big)
                out += (
                    f", KHE_HỞ={len(big)} (thiếu ~{missing:.1f}s; ví dụ "
                    + ", ".join(f"{a:.1f}->{b:.1f}" for _g, a, b in big[:3])
                    + ")"
                )
            else:
                out += ", LIỀN MẠCH"
            return out
        except Exception as exc:  # noqa: BLE001
            return f"chuỗi fragment lỗi: {exc}"

    def _last_audio_tfdt_sec(self) -> float:
        """Mốc media (giây) của fragment audio CUỐI CÙNG trong bộ đệm.

        Dùng để biết đường giải mã nối-liền đã lấy hết audio đang có chưa: `_last_pts` tiến
        tới đây nghĩa là không còn gì để lấy, nên KHÔNG cần chạy đường dự phòng (tránh tốn CPU
        mở lại container cho từng mảnh mỗi lượt).
        """
        try:
            times = scan_mp4_fragment_times(bytes(self._media))
        except Exception:  # noqa: BLE001
            return float("-inf")
        if not times:
            return float("-inf")
        ts = self._audio_timescale()
        scale = 1.0 / float(ts) if ts > 0 else 1.0
        return max(d for d, _t in times) * scale + self._axis_offset()

    def _decode_fragments_individually(self) -> List[StreamAudioChunk]:
        """Giải mã RIÊNG từng cặp `moof`+`mdat` (ghép với init), bỏ qua fragment hỏng.

        Trả về các đoạn PCM > `_last_pts`, ĐÃ sắp theo PTS và cắt chồng lấn. Trả `[]` nếu
        cách này không tốt hơn đường nối-liền (tránh thay thế vô ích).
        """
        pairs = self._fragment_byte_ranges()
        if not pairs:
            return []
        # Nhắm thẳng vào vùng cần: bỏ qua các mảnh ĐÃ phát và chỉ lấy tối đa 200 mảnh kế tiếp.
        # Nếu lấy 200 mảnh ĐẦU bộ đệm thì hàm này giải mã lại mãi phần cũ và không bao giờ tới
        # phần đang cần (lỗi đã gặp: bộ đệm 6 MB / 3000+ mảnh, chỉ 46 mảnh đầu được xét).
        #: Mốc CHỌN mảnh để giải mã từng-fragment: `max(_last_pts, _indiv_scan_floor)`.
        #: PHẢI tách khỏi `_last_pts` (mốc PCM ĐÃ PHÁT). Dùng chung một biến thì việc
        #: "đã quét qua vùng không ra PCM" bị ghi nhầm thành "đã phát hết vùng đó", và
        #: audio chưa hề phát sẽ bị frontier lọc VĨNH VIỄN — đúng cơ chế làm Pipeline B
        #: kẹt ở một mốc duy nhất (xem `test_77_demuxer_no_audio_loss.py`).
        self_floor = self._indiv_scan_floor
        frontier = max(self._last_pts, self_floor)
        pending = []
        for pair in pairs:
            if pair[2] < frontier - 0.05:
                continue
            pending.append(pair)
            if len(pending) >= 200:
                break
        if not pending:
            return []
        pairs = pending

        # `_last_pts` âm vô cực có nghĩa: chưa có mốc đã phát. Khi đó mốc tiến độ được giữ ở
        # `recovered_frontier` (mốc của mảnh CUỐI vùng vừa xét) — nhưng CHỈ ghi vào sàn quét
        # `_indiv_scan_floor`, không bao giờ ghi vào `_last_pts` (xem cuối hàm).
        recovered_frontier = float("-inf")
        for _s, _e, dt in pairs:
            if dt > recovered_frontier:
                recovered_frontier = dt
        out: List[StreamAudioChunk] = []
        frontier = self._last_pts
        for start, end, decode_time in pairs:
            block = self._init + bytes(self._media[start:end])
            try:
                container = av.open(io.BytesIO(block))
            except Exception:  # noqa: BLE001
                continue
            try:
                if not container.streams.audio:
                    continue
                stream = container.streams.audio[0]
                tb = float(stream.time_base or 0) or 0.0
                if tb <= 0:
                    continue
                resampler = av.AudioResampler(
                    format="flt", layout="mono", rate=self.target_sample_rate
                )
                parts: List[np.ndarray] = []
                first_pts: Optional[float] = None
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    pts = float(frame.pts) * tb + self._axis_offset()
                    if pts < frontier - 1e-6:
                        continue
                    if first_pts is None:
                        first_pts = pts
                    for rs in resampler.resample(frame):
                        arr = rs.to_ndarray()
                        arr = arr.reshape(-1) if arr.ndim > 1 else arr
                        if arr.size:
                            parts.append(arr.astype(np.float32, copy=True))
                try:
                    for rs in resampler.resample(None):
                        arr = rs.to_ndarray()
                        arr = arr.reshape(-1) if arr.ndim > 1 else arr
                        if arr.size:
                            parts.append(arr.astype(np.float32, copy=True))
                except Exception:  # noqa: BLE001
                    pass
                if not parts or first_pts is None:
                    continue
                pcm = np.concatenate(parts) if len(parts) > 1 else parts[0]
                if pcm.size <= 0:
                    continue
                start_pts = max(first_pts, frontier)
                trim = int(round((start_pts - first_pts) * self.target_sample_rate))
                if trim > 0:
                    if trim >= pcm.size:
                        continue
                    pcm = pcm[trim:]
                end_pts = start_pts + pcm.size / float(self.target_sample_rate)
                out.append(StreamAudioChunk(
                    pts_start=start_pts, pts_end=end_pts, duration=end_pts - start_pts,
                    pcm=np.ascontiguousarray(pcm),
                ))
                frontier = end_pts
            except Exception:  # noqa: BLE001
                continue
            finally:
                try:
                    container.close()
                except Exception:  # noqa: BLE001
                    pass
        # Không ra PCM nhưng vùng này đã được xử lý: ghi nhận mốc để lượt sau TIẾN tiếp thay vì
        # giải mã lại đúng 200 mảnh đó mãi (nguyên nhân `emitted` đứng yên ở log thật).
        #
        # ⚠️ CHỈ nâng SÀN QUÉT, TUYỆT ĐỐI KHÔNG nâng `_last_pts`. `_last_pts` là mốc PCM ĐÃ
        # PHÁT; nâng nó ở đây nghĩa là "coi như đã phát" một vùng chưa hề phát ra mẫu nào ⇒
        # mọi frame của vùng đó vĩnh viễn bị lọc (`pts < _last_pts`) và không đường nào lấy
        # lại được. Log thật 2026-10-08: bộ đệm giữ 0→40s, `_last_pts`=16s, một lượt dự phòng
        # rỗng đẩy thẳng frontier tới 36,02s ⇒ pipeline đứng im ở 16,00s suốt phiên.
        if not out and recovered_frontier > float("-inf"):
            advanced = recovered_frontier + 0.02 > self._indiv_scan_floor
            self._indiv_scan_floor = max(self._indiv_scan_floor, recovered_frontier + 0.02)
            if advanced:
                # Nhánh này TỪNG im lặng tuyệt đối, và chính sự im lặng đó làm sự cố
                # 2026-10-08 khó chẩn đoán: bộ đệm đầy audio nhưng không ra PCM nào.
                logger.warning(
                    f"Giải mã từng-fragment không ra PCM cho vùng "
                    f"{self._last_pts:.2f}s→{recovered_frontier:.2f}s "
                    f"({len(pairs)} mảnh) — nâng SÀN QUÉT (không đụng mốc đã phát). "
                    f"Byte vẫn nằm trong bộ đệm nên đường giải mã nối-liền vẫn thử lại được.",
                    extra={"module_tag": "WS"},
                )
        return out

    def _fragment_byte_ranges(self) -> List[Tuple[int, int, float]]:
        """[(byte_start, byte_end, tfdt_giây)] cho từng cặp `moof`+`mdat` liền nhau.

        `tfdt` được quy về GIÂY để bên gọi lọc được theo mốc đã phát (nếu không, hàm giải mã
        từng mảnh sẽ luôn quét các mảnh ĐẦU bộ đệm và không bao giờ tới phần đang cần).
        """
        buf = bytes(self._media)
        ts = self._audio_timescale()
        scale = (1.0 / float(ts)) if ts > 0 else 1.0
        times = self._audio_fragment_times()
        audio_track = self._dominant_audio_track(times)
        first_needed = self._last_pts - 0.05

        # ── ĐƯỜNG NHANH ───────────────────────────────────────────────────────────
        # Đa số trường hợp chỉ có MỘT track trong dòng và cửa sổ cần nằm ở CUỐI bộ đệm. Quét
        # từ offset của mảnh gần `first_needed` trở đi thay vì duyệt toàn bộ (bộ đệm có thể
        # tới hàng nghìn `moof`; duyệt hết mỗi lượt là quá đắt).
        start_off = self._first_offset_at_or_after(first_needed, scale, audio_track, times)
        if start_off is not None:
            out_fast: List[Tuple[int, int, float]] = []
            for off in scan_mp4_boundaries(buf[start_off:], base=start_off):
                size = int.from_bytes(buf[off:off + 4], "big")
                if size < 8 or off + size + 8 > len(buf):
                    break
                info = scan_mp4_track_at(buf, off)
                if audio_track >= 0 and info is not None and info[1] != audio_track:
                    continue
                mdat_off = off + size
                if buf[mdat_off + 4:mdat_off + 8] != b"mdat":
                    continue
                mdat_size = int.from_bytes(buf[mdat_off:mdat_off + 4], "big")
                if mdat_size < 8:
                    continue
                decode_time = info[0] * scale + self._axis_offset() if info is not None else 0.0
                out_fast.append((off, mdat_off + mdat_size, decode_time))
            if out_fast:
                return out_fast

        # ── ĐƯỜNG ĐẦY ĐỦ (dự phòng) ───────────────────────────────────────────────
        by_offset: Dict[int, Tuple[float, int]] = {}
        for off in scan_mp4_boundaries(buf):
            info = scan_mp4_track_at(buf, off)
            if info is not None:
                by_offset[off] = info
        out: List[Tuple[int, int, float]] = []
        for off in scan_mp4_boundaries(buf):
            size = int.from_bytes(buf[off:off + 4], "big")
            if size < 8 or off + size + 8 > len(buf):
                continue
            info = by_offset.get(off)
            if audio_track >= 0 and info is not None and info[1] != audio_track:
                continue
            mdat_off = off + size
            if buf[mdat_off + 4:mdat_off + 8] != b"mdat":
                continue
            mdat_size = int.from_bytes(buf[mdat_off:mdat_off + 4], "big")
            if mdat_size < 8:
                continue
            decode_time = info[0] * scale + self._axis_offset() if info is not None else 0.0
            out.append((off, mdat_off + mdat_size, decode_time))
        return out

    def _dominant_audio_track(self, times: List[Tuple[float, int]]) -> int:
        """`track_id` chiếm đa số trong các `moof` (track audio), -1 nếu chưa biết."""
        if not times:
            return -1
        counts: Dict[int, int] = {}
        for _d, t in times:
            counts[t] = counts.get(t, 0) + 1
        return max(counts, key=lambda k: counts[k])

    def _first_offset_at_or_after(
        self, target_sec: float, scale: float, audio_track: int,
        times: List[Tuple[float, int]],
    ) -> Optional[int]:
        """Offset của `moof` audio ĐẦU TIÊN có mốc ≥ `target_sec` (None nếu không xác định được).

        Dùng `self._offsets_cache` (offset của mọi `moof`) + tìm nhị phân trên `times` (đã sắp
        theo thời gian) ⇒ không phải duyệt toàn bộ bộ đệm mỗi lượt.
        """
        offsets = self._offsets_cache
        if not offsets or self._offsets_key != len(self._media) or not times:
            buf = bytes(self._media)
            offsets = scan_mp4_boundaries(buf)
            self._offsets_cache = offsets
            self._offsets_key = len(self._media)
        # Chỉ ghép được offset với mốc thời gian khi HAI danh sách cùng độ dài (cùng một lần
        # quét `moof`). Lệch độ dài ⇒ không suy ra được offset, để bên gọi dùng đường đầy đủ.
        if not offsets or len(offsets) != len(times):
            return None
        target = target_sec if scale <= 0 else (target_sec - self._axis_offset()) / scale
        idx = bisect.bisect_left([d for d, _t in times], target)
        if idx <= 0:
            return offsets[0]
        if idx >= len(offsets):
            return None
        return offsets[idx]

    def _audio_sample_rate(self) -> int:
        """Sample rate của track audio, 0 nếu chưa đọc được.

        ⚠️ Đây là SAMPLE RATE, **không** phải timescale của `tfdt`. Muốn quy đổi `tfdt`
        sang giây phải dùng `_audio_timescale()` — xem chú thích ở đó.
        """
        if self._cached_sample_rate:
            return self._cached_sample_rate
        try:
            container = av.open(io.BytesIO(self._init))
            try:
                if container.streams.audio:
                    self._cached_sample_rate = int(
                        container.streams.audio[0].codec_context.sample_rate or 0
                    )
            finally:
                container.close()
        except Exception:  # noqa: BLE001
            pass
        return self._cached_sample_rate

    def _audio_timescale(self) -> int:
        """Timescale THẬT của track audio — đơn vị của `baseMediaDecodeTime` (`tfdt`).

        Vì sao KHÔNG dùng `codec_context.sample_rate`: hai đại lượng này chỉ tình cờ bằng
        nhau với AAC thường do FFmpeg đóng gói. Với HE-AAC/SBR (sample rate báo bằng một
        nửa timescale) và với nhiều bộ đóng gói tuỳ biến, chúng KHÁC nhau. Quy đổi `tfdt`
        bằng sample rate làm mốc thời gian bị thổi phồng.

        Đo thật 2026-10-08 (xhamster, fMP4 muxed AV1+AAC): bộ đệm chỉ giữ ~100 s audio
        nhưng `tfdt` quy ra tới **180 s**. Hệ quả: điều kiện "còn audio phía trước chưa lấy"
        (`_last_pts < last_tfdt - 2`) LUÔN đúng ⇒ đường giải mã từng-fragment chạy trên MỌI
        mảnh, mở container cho từng mảnh mà không bao giờ ra PCM, và sàn quét (`_indiv_scan_floor`)
        bị đẩy vượt xa mốc đã phát — tức cơ chế cứu hộ tự vô hiệu hoá.

        `stream.time_base` của PyAV (đọc từ `mdhd` của track) mới là nguồn sự thật.
        """
        if self._cached_timescale:
            return self._cached_timescale
        try:
            container = av.open(io.BytesIO(self._init))
            try:
                if container.streams.audio:
                    tb = float(container.streams.audio[0].time_base or 0.0)
                    if tb > 0:
                        self._cached_timescale = int(round(1.0 / tb))
            finally:
                container.close()
        except Exception:  # noqa: BLE001
            pass
        if not self._cached_timescale:
            # Dự phòng cuối: sample rate (đúng cho phần lớn AAC do FFmpeg đóng gói).
            self._cached_timescale = self._audio_sample_rate()
        return self._cached_timescale

    def _audio_fragment_times(self) -> List[Tuple[float, int]]:
        """[(tfdt, track_id)] của các fragment AUDIO (track chiếm đa số), có cache theo byte.

        Cache theo `len(self._media)`: dữ liệu chỉ được THÊM vào nên chỉ cần quét lại khi độ
        dài thay đổi — tránh quét lại toàn bộ mỗi lượt giải mã.
        """
        key = len(self._media)
        if self._frag_times_cache is not None and self._frag_times_key == key:
            return self._frag_times_cache
        times: List[Tuple[float, int]] = []
        try:
            raw = scan_mp4_fragment_times(bytes(self._media))
            if raw:
                counts: Dict[int, int] = {}
                for _d, t in raw:
                    counts[t] = counts.get(t, 0) + 1
                audio_track = max(counts, key=lambda k: counts[k])
                times = [t for t in raw if t[1] == audio_track]
        except Exception:  # noqa: BLE001
            times = []
        self._frag_times_cache = times
        self._frag_times_key = key
        return times

    def _decode_raw_aac(self) -> List[StreamAudioChunk]:
        """Dự phòng cho fMP4/AAC: giải mã TRỰC TIẾP các frame AAC trong `mdat`.

        Vì sao cần: khi đường chuẩn (`container.decode()`) không lấy ra hết audio dù byte đã
        liền mạch, ta vẫn còn nguyên payload trong `mdat`. Bóc `mdat` rồi bọc lại thành luồng
        ADTS cho decoder AAC thô là đường KHÔNG phụ thuộc việc PyAV có đi hết được chuỗi
        `moof` hay không.

        PTS neo vào `tfdt.baseMediaDecodeTime` của fragment audio ĐẦU TIÊN (mốc media của mẫu
        đầu trong bộ đệm) cộng `timestampOffset` — cùng trục thời gian với `_last_pts`.
        """
        raw = self._extract_mdat_payload()
        if not raw:
            return []
        frames = _iter_adts_frames(raw)
        if not frames:
            return []
        # ⚠️ CHỐNG BÁO ĐỘNG GIẢ: quét `0xFFF` trên AAC THÔ (không ADTS) sẽ khớp giả ở rất
        # nhiều vị trí, tạo ra "frame" rác. Luồng ADTS thật phủ gần hết payload; nếu độ phủ
        # thấp thì đây KHÔNG phải luồng ADTS ⇒ bỏ, tuyệt đối không đẩy PCM rác vào phụ đề.
        covered = sum(len(f) for f in frames)
        if covered < 0.9 * len(raw) or len(frames) < 4:
            self._raw_fallback_rejected += 1
            return []
        payload = b"".join(frames)
        try:
            codec = av.CodecContext.create("aac", "r")
            codec.sample_rate = 44100
            codec.layout = "mono"
            pcm_parts: List[np.ndarray] = []
            resampler = av.AudioResampler(format="flt", layout="mono", rate=self.target_sample_rate)
            for frame in codec.decode(av.Packet(payload)):
                for rs in resampler.resample(frame):
                    arr = rs.to_ndarray()
                    arr = arr.reshape(-1) if arr.ndim > 1 else arr
                    if arr.size:
                        pcm_parts.append(arr.astype(np.float32, copy=True))
            try:
                for rs in resampler.resample(None):
                    arr = rs.to_ndarray()
                    arr = arr.reshape(-1) if arr.ndim > 1 else arr
                    if arr.size:
                        pcm_parts.append(arr.astype(np.float32, copy=True))
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Giải mã AAC thô thất bại: {exc}", extra={"module_tag": "WS"})
            return []

        if not pcm_parts:
            return []
        pcm = np.concatenate(pcm_parts) if len(pcm_parts) > 1 else pcm_parts[0]
        if pcm.size <= 0:
            return []
        # Mốc neo: `tfdt` tính theo timescale của track (thường = sample_rate của AAC).
        anchor = float(self._axis_offset())
        if self._first_audio_tfdt is not None and codec.sample_rate:
            anchor += float(self._first_audio_tfdt) / float(codec.sample_rate)
        return [StreamAudioChunk(
            pts_start=anchor,
            pts_end=anchor + pcm.size / float(self.target_sample_rate),
            duration=pcm.size / float(self.target_sample_rate),
            pcm=np.ascontiguousarray(pcm),
        )]

    def _extract_mdat_payload(self) -> bytes:
        """Ghép payload của MỌI box `mdat` cấp cao nhất trong bộ đệm media."""
        buf = bytes(self._media)
        out: List[bytes] = []
        pos = 0
        n = len(buf)
        while pos + 8 <= n:
            size = int.from_bytes(buf[pos:pos + 4], "big")
            box_type = buf[pos + 4:pos + 8]
            header = 8
            if size == 1:
                if pos + 16 > n:
                    break
                size = int.from_bytes(buf[pos + 8:pos + 16], "big")
                header = 16
            elif size == 0:
                size = n - pos
            if size < header:
                break
            if box_type == b"mdat":
                out.append(buf[pos + header:pos + size])
            pos += size
        return b"".join(out)

    @property
    def average_rtf(self) -> float:
        if self.decoded_seconds <= 0:
            return 0.0
        return self.decode_time_sec / self.decoded_seconds

    def structure_report(self) -> str:
        """Cấu trúc container THẬT của bộ đệm hiện tại (chẩn đoán nút cổ chai giải mã).

        Trả lời ba câu hỏi không thể suy ra từ log thường:
          * Container/codec gì, có **mấy stream audio** (nhiều stream ⇒ chỉ giải mã stream 0).
          * Bao nhiêu frame audio trong bộ đệm byte hiện có và chúng trải tới mốc nào.
          * Bộ đệm byte lớn nhưng frame ít ⇒ phần lớn byte KHÔNG phải audio (mảnh của
            SourceBuffer khác lọt vào) — đúng loại lỗi làm decoder "dừng sớm" mỗi lượt.
        """
        payload = self._init + bytes(self._media)
        info = (
            f"container={self._container}, init={len(self._init)}B, media={len(self._media)}B"
        )
        if not payload:
            return info
        try:
            container = av.open(io.BytesIO(payload))
        except Exception as exc:  # noqa: BLE001
            return f"{info}, MỞ LỖI: {exc}"
        try:
            audio_streams = list(container.streams.audio)
            parts = [info, f"n_audio_streams={len(audio_streams)}"]
            if not audio_streams:
                parts.append("codec=KHÔNG CÓ STREAM AUDIO")
                return ", ".join(parts)
            stream = audio_streams[0]
            n_frames = 0
            first_pts: Optional[float] = None
            last_pts: Optional[float] = None
            sum_dur = 0.0
            tb = float(stream.time_base or 0) or 1.0
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                pts = float(frame.pts) * tb
                if first_pts is None:
                    first_pts = pts
                last_pts = pts
                n_frames += 1
                rate = float(getattr(frame, "sample_rate", 0) or 0)
                if rate > 0 and frame.samples:
                    sum_dur += float(frame.samples) / rate
            span = 0.0 if (first_pts is None or last_pts is None) else (last_pts - first_pts)
            parts.append(
                f"codec={stream.codec_context.name}, rate={stream.codec_context.sample_rate}, "
                f"time_base={tb:.6f}, frames={n_frames}, pts=[{first_pts}, {last_pts}], span={span:.1f}s"
            )
            # `duration/span > ~1,2` ⇒ dữ liệu nhận được có MẢNH TRÙNG (cùng đoạn media được
            # append nhiều lần). Đây là dấu hiệu định lượng phân biệt "trùng lặp" với "mất
            # audio" — hai nguyên nhân có cách xử lý ngược nhau.
            if span > 0.5:
                parts.append(f"duration/span={sum_dur / span:.2f}x (≈1.0 là bình thường)")
            if first_pts is not None and self._last_pts > float("-inf"):
                parts.append(f"chưa_giải_mã_tới={(last_pts or 0.0) + self._axis_offset():.2f}s")
            if self._container == "mp4":
                parts.append(self._mp4_fragment_chain_report(stream.codec_context.sample_rate))
            return ", ".join(parts)
        except Exception as exc:  # noqa: BLE001
            return f"{info}, QUÉT LỖI: {exc}"
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass
