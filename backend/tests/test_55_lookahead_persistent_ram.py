"""test_55_lookahead_persistent_ram.py — Kiểm thử Kho Audio Bền Vững trên RAM cho Pipeline B.

Mục tiêu kiểm thử:
1. Audio đã giải mã được LƯU BỀN VỮNG trên RAM trong suốt phiên (không bị xoá khi read).
2. Khi Seek Lùi (từ tương lai về quá khứ): Audio của quá khứ VẪN CÒN NGUYÊN trong RAM,
   ASR có thể đọc và giải mã ngay lập tức mà không phụ thuộc vào trình duyệt.
3. Khi Seek tới vùng đã buffer sẵn (vệt xám): Audio có sẵn ngay, không bị "đói audio".
4. Chấp nhận các mảnh nạp bất kỳ thứ tự nào (out-of-order) khi người dùng nhảy cóc.
5. Khi đóng phiên (`close()`), toàn bộ RAM audio được giải phóng sạch sẽ.

GHI CHÚ (2026-10-02): Pipeline B v2 ("streaming": `ingest_once()` + `_asr_loop()`) đã bị XOÁ,
nên bộ test này đọc audio qua ĐÚNG API còn sống của `ContinuousAudioTimeline` — thay vì đi qua
tầng nạp VAD đã bị xoá. Đây vẫn là các bất biến "Kho RAM bền vững" chứ không phải test tầng nạp.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import numpy as np
import pytest

from backend.core.lookahead_timeline import ContinuousAudioTimeline
from backend.tests.fakes import (
    FakeInferenceEngine,
    FakeTranslator,
    FakeVADEngine,
)
from backend.ws.lookahead_handler import LookaheadSessionState


class MockSafeConnection:
    """Ghi nhận mọi bản tin JSON gửi về client."""

    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        self.sent_messages.append(payload)
        return True

    def of_type(self, msg_type: str) -> List[Dict[str, Any]]:
        return [m for m in self.sent_messages if m.get("type") == msg_type]


def _make_session() -> LookaheadSessionState:
    """Phiên Pipeline B tối thiểu: engine GIẢ (không nạp model), VAD giả.

    `FakeVADEngine` ở đây chỉ để `init_components()` dựng được `vad_processor` (tuyến batch chỉ
    đọc `start_pad_ms` của nó, không chạy VAD để cắt câu).
    """
    conn = MockSafeConnection()
    session = LookaheadSessionState(
        ws=conn,  # type: ignore[arg-type]
        asr_engine=FakeInferenceEngine(text_fn=lambda n: "hello world from lookahead"),
        translation_engine=FakeTranslator(),
    )
    session.conn = conn  # type: ignore[attr-defined]
    session.init_components(vad_engine_override=FakeVADEngine())
    return session


def test_first_audio_pts_at_or_after_bridges_gap():
    """`first_audio_pts_at_or_after` là cơ sở cho bước "nhảy con trỏ ra khỏi KHE HỞ"."""
    tl = ContinuousAudioTimeline(sample_rate=16000)
    tl.append(330.0, np.zeros(16000 * 2, dtype=np.float32))   # audio [330, 332]

    # Trước đoạn audio ⇒ trả về ĐẦU đoạn (đây chính là mốc để nhảy con trỏ tới).
    assert tl.first_audio_pts_at_or_after(322.07) == pytest.approx(330.0)
    # Nằm TRONG đoạn ⇒ audio có ngay tại đó, trả về đúng mốc hỏi.
    assert tl.first_audio_pts_at_or_after(331.0) == pytest.approx(331.0)
    # Sau tất cả ⇒ không có gì.
    assert tl.first_audio_pts_at_or_after(400.0) is None

    # Và `buffered_end_from` đúng là trả None ở khe hở > MAX_GAP_FILL_SEC (5s) — tiền đề của lỗi.
    assert tl.buffered_end_from(322.07) is None


def test_nudge_cursor_out_of_gap_after_seek():
    """HỒI QUY 2026-10-05 (xvideos.com): sau khi TUA, con trỏ khối rơi vào KHE HỞ.

    Log thật: `Seek … mốc=322.07s`, sau đó `mốc-đã-xử-lý=322.07s` đứng im suốt 35 giây dù
    `Audio RAM: 245.7s` và `Đệm trước: 36.0s` — vì audio thật của vị trí mới bắt đầu muộn hơn
    (trình phát nạp lại theo ranh giới segment) nên `next_chunk()` trả `None` MÃI MÃI.
    """
    session = _make_session()
    session.timeline.append(330.0, np.zeros(16000 * 3, dtype=np.float32))
    session._batch_from_pts = 322.07
    session._feed_pts = 322.07

    # Trong thời gian chờ kiên nhẫn ban đầu (grace period), KHÔNG được nhảy vội
    assert session._nudge_cursor_out_of_gap(322.07) is False, "phải chờ kiên nhẫn trong grace period"
    # Khi ép buộc (hoặc đã hết thời gian chờ kiên nhẫn), mới nhảy ra khỏi khe hở
    moved = session._nudge_cursor_out_of_gap(322.07, force=True)

    assert moved is True, "phải nhảy con trỏ ra khỏi khe hở"
    assert session._batch_from_pts == pytest.approx(330.0)
    # `_feed_pts` ("đã qua ASR") KHÔNG được đẩy theo: vùng bị nhảy qua không có audio nên chưa
    # hề được xử lý — đẩy theo sẽ thổi phồng `fed_ahead` và làm client resume sớm.
    assert session._feed_pts == pytest.approx(322.07)

    # Khi CÓ audio tại con trỏ thì KHÔNG được nhảy (đang chờ nạp thêm là chuyện bình thường).
    assert session._nudge_cursor_out_of_gap(331.0, force=True) is False
    assert session._batch_from_pts == pytest.approx(330.0)


def test_nudge_cursor_out_of_sliver_then_hole():
    """HỒI QUY 2026-10-05 (xvideos.com, sau khi tua lần 2): con trỏ ở 330,03 s — có ĐÚNG 0,0 s
    audio tại đó rồi khe hở mới tới đoạn kế.

    Lúc này `buffered_end_from(con_trỏ)` trả về ~330,03 (KHÁC `None`) nên van "khe hở hoàn toàn"
    không kích hoạt, mà `next_chunk()` vẫn trả `None` (`buffered_end <= from_pts + 0.05`) ⇒
    `mốc-đã-xử-lý` đứng im và client pause/resume liên tục ("seek xong thì bị pause liên tục").
    """
    session = _make_session()
    # Mẩu vụn 0,1 s tại 330,0 rồi khe hở 9,9 s tới đoạn thật ở 340,0.
    session.timeline.append(330.0, np.zeros(int(16000 * 0.1), dtype=np.float32))
    session.timeline.append(340.0, np.zeros(16000 * 3, dtype=np.float32))
    session._batch_from_pts = 330.0

    # Trong grace period, không nhảy vội
    assert session._nudge_cursor_out_of_gap(330.0) is False, "phải chờ kiên nhẫn trong grace period"
    # Sau grace period hoặc khi force=True, mới nhảy
    assert session._nudge_cursor_out_of_gap(330.0, force=True) is True, "phải nhảy qua mẩu vụn + khe hở"
    assert session._batch_from_pts == pytest.approx(340.0)


def test_nudge_does_not_jump_at_live_edge():
    """Ở MÉP NẠP (không còn đoạn nào phía sau) thì phải CHỜ, không nhảy — nếu không sẽ bỏ audio."""
    session = _make_session()
    session.timeline.append(100.0, np.zeros(16000, dtype=np.float32))   # audio [100, 101]
    session._batch_from_pts = 100.5

    assert session._nudge_cursor_out_of_gap(100.5) is False
    assert session._batch_from_pts == pytest.approx(100.5)


def _drain_timeline(session: LookaheadSessionState, target_pts: float, block_sec: float = 0.5) -> int:
    """Đọc timeline như tầng xử lý khối vẫn làm; trả về số khối đã đọc."""
    fed = 0
    while True:
        if session.timeline.cursor_pts is None or session.timeline.cursor_pts >= target_pts:
            break
        item = session.timeline.read(block_sec)
        if item is None:
            break
        pts, pcm = item
        if pcm.size:
            session._feed_pts = pts + pcm.size / 16000.0
            fed += 1
    return fed


async def _drain_timeline_async(
    session: LookaheadSessionState,
    target_pts: float,
    block_sec: float = 0.5,
    seen: List[float] | None = None,
) -> int:
    """Bản bất đồng bộ của `_drain_timeline`, GHI LẠI mốc của từng khối đã đọc.

    Vòng lặp batch thật đọc từng khối qua `asyncio.to_thread(...)` nên giữa hai khối LUÔN có điểm
    nhường event loop — và seek có thể chen vào đúng lúc đó. Test mô phỏng đúng điểm nhường ấy bằng
    `await asyncio.sleep(0)` sau mỗi khối.
    """
    fed = 0
    while True:
        if session.timeline.cursor_pts is None or session.timeline.cursor_pts >= target_pts:
            break
        item = session.timeline.read(block_sec)
        if item is None:
            break
        pts, pcm = item
        if pcm.size:
            session._feed_pts = pts + pcm.size / 16000.0
            fed += 1
            if seen is not None:
                seen.append(pts)
        await asyncio.sleep(0)
    return fed


@pytest.mark.asyncio
async def test_pipeline_b_audio_persists_after_read():
    """Kiểm tra audio không bị xoá sau khi ASR đọc."""
    session = _make_session()
    # Nạp 10 giây audio từ 0.0s đến 10.0s
    pcm = np.full(16000 * 10, 0.4, dtype=np.float32)
    session.timeline.append(0.0, pcm)

    assert session.timeline.total_stored_seconds() == pytest.approx(10.0, abs=0.01)

    # Đọc hết 10 giây qua API đọc của timeline (đúng cách tầng xử lý khối đọc audio)
    fed = _drain_timeline(session, 10.0)
    assert fed > 0
    assert session.timeline.cursor_pts == pytest.approx(10.0, abs=0.01)

    # Sau khi đọc: audio VẪN NGUYÊN VẸN trong RAM!
    assert session.timeline.total_stored_seconds() == pytest.approx(10.0, abs=0.01)
    await session.close()


@pytest.mark.asyncio
async def test_pipeline_b_seek_backward_instantly_available():
    """Tua lùi từ phút 60 về phút 1: audio phút 1 vẫn còn trong RAM và giải mã được ngay."""
    session = _make_session()

    # 1. Phát đoạn đầu: phút 0 đến phút 2 (0..120s)
    pcm_early = np.full(16000 * 120, 0.2, dtype=np.float32)
    session.timeline.append(0.0, pcm_early)

    session.current_time = 0.0
    session.lead_time = 5.0
    _drain_timeline(session, session.current_time + session.lead_time)
    assert session.timeline.cursor_pts is not None
    assert session.timeline.cursor_pts > 0.0

    # 2. Người dùng nhảy cóc tới phút 60 (3600s)
    await session.handle_seek("seek_jump_60m", 3600.0)
    assert session.current_time == 3600.0
    assert session.timeline.cursor_pts == 3600.0

    # Nạp audio của phút 60 (3600s .. 3630s)
    pcm_late = np.full(16000 * 30, 0.8, dtype=np.float32)
    session.timeline.append(3600.0, pcm_late)
    assert session.timeline.total_stored_seconds() == pytest.approx(150.0, abs=0.05)

    # Đọc thử tại phút 60
    _drain_timeline(session, 3610.0)

    # 3. Người dùng TUA LÙI về giây thứ 30 (30.0s, thuộc đoạn đầu)
    await session.handle_seek("seek_back_30s", 30.0)
    assert session.current_time == 30.0
    assert session.timeline.cursor_pts == 30.0

    # KIỂM TRA QUAN TRỌNG:
    # Tại 30.0s, RAM đã có sẵn audio tới tận 120s!
    assert session.timeline.has_audio_at(30.0) is True
    assert session.timeline.buffered_end_from(30.0) == pytest.approx(120.0, abs=0.1)
    assert session._decoded_end_pts == pytest.approx(120.0, abs=0.1)

    # ASR đọc ngay lập tức được audio tại 30.0s
    item = session.timeline.read(5.0)
    assert item is not None
    pts, pcm_read = item
    assert pts == pytest.approx(30.0, abs=1e-5)
    assert len(pcm_read) == 16000 * 5
    assert pcm_read[0] == pytest.approx(0.2)

    await session.close()


@pytest.mark.asyncio
async def test_pipeline_b_seek_within_buffered_range_no_starvation():
    """Tua trong vùng video đã buffer sẵn: không bị đói audio và prebuffer_ready lập tức."""
    session = _make_session()

    # Trình duyệt đã nạp trước audio từ 0.0s đến 60.0s
    pcm = np.full(16000 * 60, 0.5, dtype=np.float32)
    session.timeline.append(0.0, pcm)

    # Đang phát ở 5.0s
    session.current_time = 5.0
    _drain_timeline(session, 20.0)

    # Người dùng bấm tua tới 40.0s (nằm trong vệt xám 0..60s đã nạp)
    await session.handle_seek("seek_to_40", 40.0)

    # Hệ thống lập tức nhận diện còn 20s audio đệm phía trước
    assert session.current_time == 40.0
    assert session._decoded_end_pts == pytest.approx(60.0, abs=0.05)
    buffered_ahead = session.timeline.pending_seconds()
    assert buffered_ahead == pytest.approx(20.0, abs=0.05)

    # Nạp tiếp mượt mà mà không cần trình duyệt gửi thêm byte nào
    fed = _drain_timeline(session, 60.0)
    assert fed > 0

    await session.close()


@pytest.mark.asyncio
async def test_pipeline_b_out_of_order_sparse_buffering():
    """Các mảnh đến rải rác out-of-order đều được xếp đúng vị trí trên trục thời gian."""
    session = _make_session()

    # Chunk 1: phút 60 (3600..3610s) đến trước
    session.timeline.append(3600.0, np.full(16000 * 10, 0.6, dtype=np.float32))
    # Chunk 2: phút 0 (0..10s) đến sau
    session.timeline.append(0.0, np.full(16000 * 10, 0.1, dtype=np.float32))
    # Chunk 3: phút 30 (1800..1810s) đến cuối cùng
    session.timeline.append(1800.0, np.full(16000 * 10, 0.3, dtype=np.float32))

    assert session.timeline.total_stored_seconds() == pytest.approx(30.0, abs=0.05)

    # Kiểm tra truy vấn tại từng mốc
    assert session.timeline.has_audio_at(0.0) is True
    assert session.timeline.has_audio_at(1800.0) is True
    assert session.timeline.has_audio_at(3600.0) is True

    # Vùng giữa (ví dụ phút 15: 900s) chưa có audio -> trả về False
    assert session.timeline.has_audio_at(900.0) is False

    await session.close()


@pytest.mark.asyncio
async def test_pipeline_b_close_frees_memory():
    """Khi đóng phiên, toàn bộ audio trên RAM được giải phóng (clear)."""
    session = _make_session()
    session.timeline.append(0.0, np.full(16000 * 60, 0.5, dtype=np.float32))
    assert session.timeline.total_stored_seconds() > 0

    await session.close()
    assert session.timeline.total_stored_seconds() == 0.0
    assert len(session.timeline._chunks) == 0


@pytest.mark.asyncio
async def test_pipeline_b_backward_seek_resets_ready_until_pts():
    """Tua lùi phải reset _ready_until_pts về target_time để không báo sai phụ đề đã sẵn sàng."""
    session = _make_session()
    # Nạp 100s audio vào RAM
    session.timeline.append(0.0, np.full(16000 * 100, 0.5, dtype=np.float32))

    # Phát tới 70s
    session.current_time = 70.0
    session.lead_time = 15.0
    session._session_start_pts = 70.0
    session.timeline.seek(70.0)
    session._feed_pts = 70.0
    session._ready_until_pts = 85.0

    # Tua lùi về 5.0s
    await session.handle_seek("seek_back_5s", 5.0)

    # 1. _ready_until_pts phải quay về 5.0 (không được là 85.0 hay 100.0)
    assert session._ready_until_pts == pytest.approx(5.0)
    assert session.current_time == pytest.approx(5.0)
    assert session._decoded_end_pts == pytest.approx(100.0)  # Audio RAM vẫn còn tới 100s

    # 2. ready_ahead phải là 0.0s (chưa có phụ đề mới cho đoạn 5s)
    ready_ahead = max(0.0, session._ready_until_pts - session.current_time)
    assert ready_ahead == pytest.approx(0.0)

    # 3. Đọc tiếp phải bắt đầu từ 5.0s trở đi (con trỏ đọc không bị kẹt ở 70s)
    assert session.timeline.cursor_pts == pytest.approx(5.0)
    fed = _drain_timeline(session, 5.0 + session.lead_time)
    assert fed > 0
    assert session.timeline.cursor_pts is not None and session.timeline.cursor_pts > 5.0
    assert session._feed_pts is not None and session._feed_pts > 5.0

    await session.close()


@pytest.mark.asyncio
async def test_pipeline_b_seek_midway_through_read():
    """Tua video trong lúc tầng xử lý khối ĐANG đọc timeline không được làm con trỏ đọc nhảy về tương lai.

    Bất biến: audio trong RAM là bền vững nên seek chỉ được DỜI con trỏ đọc; lượt đọc đang chạy dở
    không được "thắng" seek rồi để con trỏ ở mốc cũ (70s) — nếu không, phiên sẽ phiên âm lại đúng
    đoạn vừa bị người dùng rời đi và bỏ qua vùng họ vừa tua tới.
    """
    session = _make_session()
    session.timeline.append(0.0, np.full(16000 * 100, 0.5, dtype=np.float32))

    session.current_time = 70.0
    session.lead_time = 15.0
    session._session_start_pts = 70.0
    session.timeline.seek(70.0)
    session._feed_pts = 70.0
    session._ready_until_pts = 85.0

    # Tầng xử lý khối bắt đầu đọc (bất đồng bộ để nhường event loop, giống `asyncio.to_thread` của
    # vòng lặp batch thật).
    seen: List[float] = []
    drain = asyncio.create_task(_drain_timeline_async(session, 75.0, seen=seen))
    await asyncio.sleep(0)

    # Ngay lúc đó, người dùng tua lùi về 5.0s.
    await session.handle_seek("seek_back_concurrency", 5.0)
    await drain

    # 1. Lượt đọc đầu tiên phải thuộc mốc CŨ (trước khi tua) — chứng minh lượt đọc thật sự đang bay
    #    khi seek ập tới, tức đây là race thật chứ không phải hai thao tác tuần tự.
    assert seen, "lượt đọc không chạy"
    assert seen[0] == pytest.approx(70.0, abs=1e-3)

    # 2. NGAY SAU seek, mốc đọc phải nhảy về mốc tua rồi tiến dần từ đó — TUYỆT ĐỐI không đọc tiếp
    #    vùng tương lai (70s) mà người dùng vừa rời đi.
    assert session.current_time == pytest.approx(5.0)
    assert session._ready_until_pts == pytest.approx(5.0)
    assert seen[1] == pytest.approx(5.0, abs=1e-3), (
        f"lượt đọc ngay sau tua không bắt đầu ở mốc tua mà ở {seen[1]:.2f}s"
    )
    assert seen[1:] == sorted(seen[1:]), "mốc đọc bị nhảy lùi sau khi tua"

    # 3. Con trỏ đọc không được để lại dấu vết "đã đọc tới tương lai trước khi tua": mọi mốc trong
    #    khoảng (5s, 70s) chỉ được đọc SAU khi seek chạy, không có mốc nào nhảy cóc qua vùng tua.
    between = [p for p in seen[1:] if 5.0 < p < 70.0]
    assert not between or between[0] == pytest.approx(5.5, abs=1e-3)

    # 4. Audio trong RAM vẫn nguyên vẹn (đọc không huỷ dữ liệu).
    assert session.timeline.total_stored_seconds() == pytest.approx(100.0, abs=0.05)

    await session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Bất biến của chính Kho RAM (đọc KHÔNG huỷ dữ liệu) — nền tảng cho các test trên
# ──────────────────────────────────────────────────────────────────────────────

def test_timeline_read_is_non_destructive():
    """`read()` chỉ tiến con trỏ: seek lùi về vùng đã đọc phải đọc lại được y nguyên mẫu."""
    tl = ContinuousAudioTimeline(sample_rate=16000)
    tl.append(0.0, np.full(16000 * 4, 0.25, dtype=np.float32))

    first = tl.read(2.0)
    assert first is not None
    pts_first, pcm_first = first
    assert pts_first == pytest.approx(0.0)
    assert len(pcm_first) == 16000 * 2

    # Đã đọc hết 4s ⇒ RAM vẫn giữ đủ 4s.
    tl.read(2.0)
    assert tl.total_stored_seconds() == pytest.approx(4.0, abs=0.01)

    # Tua lùi: đọc lại đúng dữ liệu cũ, không bị mất.
    tl.seek(0.0)
    again = tl.read(2.0)
    assert again is not None
    pts_again, pcm_again = again
    assert pts_again == pytest.approx(0.0)
    assert np.array_equal(pcm_again, pcm_first)


def test_timeline_buffered_range_is_queryable_per_position() -> None:
    """`buffered_end_from` / `has_audio_at` phải trả lời theo TỪNG mốc (không phụ thuộc con trỏ)."""
    tl = ContinuousAudioTimeline(sample_rate=16000)
    tl.append(0.0, np.full(16000 * 10, 0.5, dtype=np.float32))
    tl.seek(6.0)

    assert tl.has_audio_at(2.0) is True
    assert tl.buffered_end_from(2.0) == pytest.approx(10.0, abs=0.05)
    assert tl.pending_seconds() == pytest.approx(4.0, abs=0.05)
    assert tl.has_audio_at(20.0) is False


def test_buffered_end_bridges_small_decode_holes() -> None:
    """Khe hở NHỎ do giải mã MSE tăng dần không được coi là "hết đệm".

    SỰ CỐ THẬT 2026-10-04 (phiên `7afc26e3`): `buffered_end_from` cắt vùng tại khe hở
    > `REPORT_EPS_SEC` (0.15 s). Bộ giải mã MSE tăng dần sinh ra các khe ~0.2 s, nên **chỉ một khe
    0.2 s là đủ để hàm báo "chỉ có 1 s audio phía trước" trong khi RAM giữ 30 s**. Bộ cắt khối đọc
    giá trị đó làm `available_sec` nên bị kẹp xuống ~9–18 s và chỉ cắt khối **8 s** dù log ghi
    `bộ đệm trước 43.3s`:

        [SEG_BATCH] Chọn khối: dài 8.1s (mục tiêu 6.3s, dải tìm tới 8.3s, bộ đệm trước 43.3s)

    `get_audio_range`/`read` đã lấp các khe này bằng silence (tới `MAX_GAP_FILL_SEC`), nên vùng
    "đọc được" phải rộng hơn vùng "liền mạch byte".
    """
    sr = 16000
    tl = ContinuousAudioTimeline(sample_rate=sr, max_pending_sec=90.0)
    # 30 mảnh 1 s, cách nhau 0.2 s (mô phỏng khe của bộ giải mã tăng dần).
    t = 0.0
    for _ in range(30):
        tl.append(t, np.full(sr, 0.3, dtype=np.float32))
        t += 1.2

    end = tl.buffered_end_from(0.0)
    assert end is not None
    # Trước khi vá: 1.0 s. Sau khi vá: phủ hết phần đã lưu (khe cuối được lấp bằng silence).
    assert end >= 29.0, f"chỉ báo {end:.1f}s trong khi RAM giữ {tl.total_stored_seconds():.1f}s"

    # Và phần được báo là "có audio" phải THỰC SỰ đọc ra được (liên tục theo trục thời gian).
    res = tl.get_audio_range(0.0, 25.0)
    assert res is not None
    pts, pcm = res
    assert pts == pytest.approx(0.0)
    assert len(pcm) / sr == pytest.approx(25.0, abs=1.0)

    # Khe hở LỚN (lỗ tua thật) vẫn phải cắt vùng — không được lấp bằng im lặng giả.
    tl2 = ContinuousAudioTimeline(sample_rate=sr, max_pending_sec=90.0)
    tl2.append(0.0, np.full(sr * 5, 0.3, dtype=np.float32))
    tl2.append(40.0, np.full(sr * 5, 0.3, dtype=np.float32))
    assert tl2.buffered_end_from(0.0) == pytest.approx(5.0, abs=0.05)
