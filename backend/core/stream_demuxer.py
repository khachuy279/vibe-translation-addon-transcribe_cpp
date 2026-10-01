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

import io
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

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


def scan_mp4_boundaries(buf: bytes) -> List[int]:
    """Trả về offset (byte) của các box `moof` (đầu mỗi fragment fMP4)."""
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
            out.append(pos)
        pos += size
    return out


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
        self._latest_ts_offset: float = 0.0
        self._lock = threading.RLock()

        # Thống kê
        self.fragments: int = 0
        self.decoded_seconds: float = 0.0
        self.decode_time_sec: float = 0.0
        self.init_bytes: int = 0
        self._seek_fallbacks: int = 0

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

    def reset(self, epoch: Optional[int] = None, min_pts: Optional[float] = None) -> None:
        """Xoá sạch trạng thái (tua video / đổi SourceBuffer / init segment mới)."""
        with self._lock:
            self._media.clear()
            if min_pts is not None:
                self._min_pts = float(min_pts)
            self._last_pts = float(self._min_pts) if self._min_pts is not None else float("-inf")
            self._latest_ts_offset = 0.0
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

    def drop_init(self) -> None:
        """Quên init segment (khi SourceBuffer tạo lại ⇒ header mới)."""
        with self._lock:
            self._init = b""
            self.init_bytes = 0

    # ------------------------------------------------------------------ ingress
    def feed(
        self,
        raw_bytes: bytes,
        timestamp_offset: float = 0.0,
        mime_type: str = "auto",
        is_init: bool = False,
        epoch: Optional[int] = None,
    ) -> List[StreamAudioChunk]:
        """Nạp một mảnh MSE và trả về các đoạn PCM MỚI đã giải mã được.

        Args:
            raw_bytes: byte thô của `SourceBuffer.appendBuffer()`.
            timestamp_offset: `sourceBuffer.timestampOffset` tại thời điểm append.
            mime_type: gợi ý định dạng (chỉ dùng cho log).
            is_init: client khẳng định đây là init segment.
            epoch: số thứ tự SourceBuffer; đổi epoch ⇒ reset bộ đệm ghép nối.
        """
        if not raw_bytes:
            return []

        with self._lock:
            if epoch is not None and int(epoch) != self._epoch:
                self._media.clear()
                self._epoch = int(epoch)
                self._last_pts = float(self._min_pts) if self._min_pts is not None else float("-inf")
                logger.info(
                    f"StreamDemuxer: SourceBuffer epoch mới = {self._epoch} (min_pts={self._min_pts}) — xoá bộ đệm ghép nối",
                    extra={"module_tag": "WS"},
                )

            self.fragments += 1
            self._latest_ts_offset = float(timestamp_offset or 0.0)
            container = detect_container(raw_bytes)
            if self._container == "unknown" and container != "unknown":
                self._container = container

            # Định dạng TỰ CHỨA (WAV/MP3/…): không ghép vào bộ đệm byte — mỗi mảnh là một
            # file hoàn chỉnh, ghép lại sẽ hỏng header. Giải mã độc lập.
            if container == "unknown" and not self._media and not self._init:
                return self._decode_standalone(raw_bytes, self._latest_ts_offset)

            treat_as_init = bool(is_init) or looks_like_init_segment(raw_bytes)
            if treat_as_init and self._init != raw_bytes:
                init_part, media_part = _split_init_media(raw_bytes, container)
                if init_part:
                    self._init = init_part
                    self.init_bytes = len(init_part)
                if media_part:
                    self._media.extend(media_part)
            elif self._init and raw_bytes.startswith(EBML_MAGIC):
                # Init đến lần nữa (replay sau khi kết nối lại) — thay thế, không nhân đôi.
                init_part, media_part = _split_init_media(raw_bytes, container)
                if init_part:
                    self._init = init_part
                    self.init_bytes = len(init_part)
                if media_part:
                    self._media.extend(media_part)
            else:
                self._media.extend(raw_bytes)

            self._trim()

            if not self._init and not self._media:
                return []

            return self._decode_new()

    # ------------------------------------------------------------------ nội bộ
    def _trim(self) -> None:
        """Giữ bộ đệm byte trong trần RAM, cắt theo ĐÚNG ranh giới cluster/fragment."""
        if len(self._media) <= self.max_media_bytes:
            return
        snapshot = bytes(self._media)
        bounds = (
            scan_mp4_boundaries(snapshot)
            if self._container == "mp4"
            else scan_webm_boundaries(snapshot)
        )
        if len(bounds) <= self.keep_boundaries:
            # Không đủ ranh giới để cắt an toàn (mảnh quá lớn / container lạ): cắt thô
            # phần đầu và buộc giải mã lại từ mốc mới.
            cut = len(self._media) - self.max_media_bytes // 2
            if cut > 0:
                del self._media[:cut]
                self._last_pts = float("-inf")
            return
        cut = bounds[len(bounds) - self.keep_boundaries]
        if cut > 0:
            del self._media[:cut]

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
        """Giải mã `init + media` và chỉ trả các frame mới hơn `self._last_pts`."""
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

            if self._last_pts > float("-inf"):
                back = max(0.0, self._last_pts - _SEEK_BACKOFF_SEC)
                try:
                    container.seek(int(back / float(time_base)), backward=True, stream=stream)
                except Exception:  # noqa: BLE001
                    self._seek_fallbacks += 1

            resampler = av.AudioResampler(
                format="flt", layout="mono", rate=self.target_sample_rate
            )
            bias = self._latest_ts_offset
            prev_pts: Optional[float] = None
            run_pts: Optional[float] = None
            run_end: Optional[float] = None
            run_frames: List[np.ndarray] = []

            def flush_run() -> None:
                if run_frames and run_pts is not None:
                    pcm = np.concatenate(run_frames) if len(run_frames) > 1 else run_frames[0]
                    if pcm.size > 0:
                        # `pts_end` suy ra từ ĐỘ DÀI PCM thực để chunk luôn tự nhất quán
                        # (timeline phía sau đối chiếu số mẫu, không tin `frame.samples`).
                        end = run_pts + float(pcm.size) / float(self.target_sample_rate)
                        chunks.append(
                            StreamAudioChunk(
                                pts_start=run_pts + bias,
                                pts_end=end + bias,
                                duration=end - run_pts,
                                pcm=pcm,
                            )
                        )

            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                pts = float(frame.pts) * float(time_base)
                # `_last_pts` là mốc KẾT THÚC (exclusive) của audio đã phát. Frame bắt đầu
                # ĐÚNG tại mốc đó là frame MỚI ⇒ phải dùng `<` chứ không phải `<=`, nếu
                # không sẽ bỏ sót đúng một nửa số frame (đã đo: WebM 32 KB/mảnh chỉ thu
                # được ~50% audio vì lý do này).
                if pts < self._last_pts - 1e-6:
                    continue

                # Độ dài frame suy từ CHÍNH frame đầu vào (PTS kế tiếp - PTS hiện tại),
                # KHÔNG suy từ số mẫu PCM ra của resampler: resampler có bộ đệm nội bộ nên
                # vài frame đầu trả về RỖNG. Nếu lấy PCM làm thước đo thì mốc tiến độ sẽ
                # đứng yên rồi nhảy, tạo "khe hở giả" và làm mất audio ở lượt sau.
                rate = float(getattr(frame, "sample_rate", 0) or 0)
                if rate > 0 and frame.samples:
                    frame_dur = float(frame.samples) / rate
                elif prev_pts is not None:
                    frame_dur = max(0.001, pts - prev_pts)
                else:
                    frame_dur = 0.02
                prev_pts = pts

                if run_end is not None and abs(pts - run_end) > 0.02:
                    # Khe hở thật trong dữ liệu ⇒ tách thành 2 đoạn rời.
                    flush_run()
                    run_frames = []
                    run_pts = None
                    run_end = None
                if run_pts is None:
                    run_pts = pts

                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray()
                    arr = arr.reshape(-1) if arr.ndim > 1 else arr
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    if arr.size:
                        run_frames.append(arr.copy())

                run_end = pts + frame_dur
                # Cập nhật tiến độ NGAY để nếu lỗi giữa chừng thì lần sau không phát lại.
                self._last_pts = max(self._last_pts, run_end)

            # Xả nốt phần mẫu còn lại trong resampler. Nếu bỏ bước này, mỗi lượt giải mã
            # mất ~1 frame (20 ms) ở mép cuối ⇒ dòng PCM bị "răng cưa" và phải bù silence
            # liên tục. Các mẫu này thuộc frame cuối (đã tính vào `run_end`) nên lượt sau
            # (seek lùi 0.5 s) sẽ không phát lại chúng (frame đó có pts < `_last_pts`).
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
        return chunks

    @property
    def average_rtf(self) -> float:
        if self.decoded_seconds <= 0:
            return 0.0
        return self.decode_time_sec / self.decoded_seconds
