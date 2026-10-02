"""test_55_lookahead_persistent_ram.py — Kiểm thử Kho Audio Bền Vững trên RAM cho Pipeline B.

Mục tiêu kiểm thử:
1. Audio đã giải mã được LƯU BỀN VỮNG trên RAM trong suốt phiên (không bị xoá khi read).
2. Khi Seek Lùi (từ tương lai về quá khứ): Audio của quá khứ VẪN CÒN NGUYÊN trong RAM,
   ASR có thể đọc và giải mã ngay lập tức mà không phụ thuộc vào trình duyệt.
3. Khi Seek tới vùng đã buffer sẵn (vệt xám): Audio có sẵn ngay, không bị "đói audio".
4. Chấp nhận các mảnh nạp bất kỳ thứ tự nào (out-of-order) khi người dùng nhảy cóc.
5. Khi đóng phiên (`close()`), toàn bộ RAM audio được giải phóng sạch sẽ.
"""

from __future__ import annotations

import asyncio
import numpy as np
import pytest

from backend.config import config
from backend.core.lookahead_timeline import ContinuousAudioTimeline
from backend.ws.lookahead_handler import LookaheadSessionState
from backend.tests.test_54_lookahead_pipeline_v2 import (
    FakeInferenceEngine,
    FakeTranslator,
    FakeVADEngine,
    MockSafeConnection,
    _make_session,
)


@pytest.mark.asyncio
async def test_pipeline_b_audio_persists_after_read():
    """Kiểm tra audio không bị xoá sau khi ASR đọc."""
    session = _make_session()
    # Nạp 10 giây audio từ 0.0s đến 10.0s
    pcm = np.full(16000 * 10, 0.4, dtype=np.float32)
    session.timeline.append(0.0, pcm)

    assert session.timeline.total_stored_seconds() == pytest.approx(10.0, abs=0.01)

    # Đọc hết 10 giây qua ingest_once
    session.current_time = 0.0
    session.lead_time = 10.0
    fed = await session.ingest_once()
    assert fed > 0

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
    await session.ingest_once()
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
    await session.ingest_once()

    # 3. Người dùng TUA LÙI về giây thứ 30 (30.0s, thuộc đoạn đầu)
    await session.handle_seek("seek_back_30s", 30.0)
    assert session.current_time == 30.0
    assert session.timeline.cursor_pts == 30.0

    # KIỂM TRA QUAN TRỌNG:
    # Tại 30.0s, RAM đã có sẵn audio tới tận 120s!
    assert session.timeline.has_audio_at(30.0) is True
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
    await session.ingest_once()

    # Người dùng bấm tua tới 40.0s (nằm trong vệt xám 0..60s đã nạp)
    await session.handle_seek("seek_to_40", 40.0)

    # Hệ thống lập tức nhận diện còn 20s audio đệm phía trước
    assert session.current_time == 40.0
    assert session._decoded_end_pts == pytest.approx(60.0, abs=0.05)
    buffered_ahead = session.timeline.pending_seconds()
    assert buffered_ahead == pytest.approx(20.0, abs=0.05)

    # Nạp tiếp mượt mà mà không cần trình duyệt gửi thêm byte nào
    fed = await session.ingest_once()
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

    # 3. Ingest phải nạp từ 5.0s trở đi
    fed = await session.ingest_once()
    assert fed > 0
    assert session.timeline.cursor_pts is not None and session.timeline.cursor_pts > 5.0
    assert session._feed_pts is not None and session._feed_pts > 5.0

    await session.close()


@pytest.mark.asyncio
async def test_pipeline_b_seek_concurrency_race_condition():
    """Kiểm tra tua video trong lúc ingest_once đang chạy trên thread không làm hỏng _feed_pts."""
    session = _make_session()
    session.timeline.append(0.0, np.full(16000 * 100, 0.5, dtype=np.float32))

    session.current_time = 70.0
    session.lead_time = 15.0
    session._session_start_pts = 70.0
    session.timeline.seek(70.0)
    session._feed_pts = 70.0

    # Giả lập VAD feed_block có độ trễ nhẹ trên worker thread
    import time
    def slow_feed_block(pcm, pts):
        time.sleep(0.04)

    session._feed_block = slow_feed_block

    # Bắt đầu ingest_once cho đoạn 70s
    task = asyncio.create_task(session.ingest_once())
    await asyncio.sleep(0.01)

    # Ngay lúc đó, người dùng tua lùi về 5.0s
    await session.handle_seek("seek_back_concurrency", 5.0)
    await task

    # Sau khi task cũ kết thúc: con trỏ và _feed_pts PHẢI ở mốc 5.0s, không được bị nhảy lên 70s!
    assert session.timeline.cursor_pts == pytest.approx(5.0)
    assert session._feed_pts == pytest.approx(5.0)
    assert session.current_time == pytest.approx(5.0)

    # Lượt ingest tiếp theo phải đọc được đoạn 5.0s bình thường
    fed = await session.ingest_once()
    assert fed > 0
    assert session.timeline.cursor_pts > 5.0
    assert session._feed_pts > 5.0

    await session.close()

