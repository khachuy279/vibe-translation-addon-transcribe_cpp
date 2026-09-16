"""Test tầng A — FIX-02/FIX-03: hàng đợi final KHÔNG được vứt câu khi đầy.

Bug gốc (đã xác nhận, xem `report/audit/17_XAC_NHAN_AUDIT_VA_KE_HOACH_SUA.md`):
`translation_queue` và `tts_queue` CHỈ chứa câu FINAL. Bản cũ dùng
`put_nowait` + `except QueueFull: drop` ⇒ khi đầy, câu đã chốt **vĩnh viễn không có bản
dịch / không có lồng tiếng**: phụ đề gốc vẫn hiện kèm dấu "..." treo.

Cách sửa: `_coalesce_enqueue()` GỘP câu cũ nhất thay vì vứt (cùng triết lý với
`_pending_commits` của ASR), giữ trần hàng đợi nên RAM vẫn bị chặn.
"""

import asyncio

import pytest

from backend.core.metrics import metrics_collector
from backend.ws.handler import _coalesce_enqueue, _join_texts

LABEL_KW = dict(
    label="Translation",
    merged_counter="queue.translation_merged",
    dropped_counter="queue.translation_dropped",
)


def _item(text: str, utt: str) -> dict:
    return {"text": text, "utterance_id": utt, "_queued_at": 0.0}


def _counter(name: str) -> int:
    return metrics_collector.get_counter(name)


def test_duoi_tran_thi_xep_hang_binh_thuong():
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    assert _coalesce_enqueue(q, _item("a", "u1"), **LABEL_KW) == "queued"
    assert q.qsize() == 1
    assert not q.empty()


def test_day_thi_GOP_chu_khong_VUT():
    """FIX-02: đầy ⇒ gộp vào câu cũ nhất, KHÔNG mất chữ nào, KHÔNG tăng counter drop."""
    dropped_before = _counter("queue.translation_dropped")
    merged_before = _counter("queue.translation_merged")

    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    _coalesce_enqueue(q, _item("câu một", "u1"), **LABEL_KW)
    _coalesce_enqueue(q, _item("câu hai", "u2"), **LABEL_KW)

    outcome = _coalesce_enqueue(q, _item("câu ba", "u3"), **LABEL_KW)
    assert outcome == "merged", "đầy thì phải GỘP, không được vứt"

    assert q.qsize() == 2, "trần hàng đợi vẫn phải được giữ (RAM bound)"
    first = q.get_nowait()
    second = q.get_nowait()
    # Thứ tự FIFO không đổi: câu cũ nhất vẫn ở đầu.
    assert first["text"] == "câu một" and first["utterance_id"] == "u1"
    # Câu mới được gộp vào câu MỚI NHẤT đang chờ (liền kề thời gian), không phải câu đầu.
    assert "câu hai" in second["text"] and "câu ba" in second["text"], second["text"]
    assert second["merged_utterance_ids"] == ["u2", "u3"]
    assert second["_merged_from"] == 2
    assert second["utterance_id"] == "u3", "câu mới là câu nhận bản dịch tổng hợp"

    assert _counter("queue.translation_dropped") == dropped_before, "KHÔNG được đếm là drop"
    assert _counter("queue.translation_merged") == merged_before + 1


def test_gop_nhieu_lan_lien_tiep_van_khong_mat_chu():
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    _coalesce_enqueue(q, _item("A", "u1"), **LABEL_KW)
    for i, text in enumerate(["B", "C", "D"], start=2):
        _coalesce_enqueue(q, _item(text, f"u{i}"), **LABEL_KW)

    merged = q.get_nowait()
    for piece in ["A", "B", "C", "D"]:
        assert piece in merged["text"], f"mất chữ {piece}: {merged['text']!r}"
    assert merged["merged_utterance_ids"] == ["u1", "u2", "u3", "u4"]
    assert merged["_merged_from"] == 4


def test_unfinished_tasks_khong_bi_ro():
    """Gộp KHÔNG được làm rò `_unfinished_tasks`, nếu không `drain_queues()`/`join()` treo."""
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    _coalesce_enqueue(q, _item("A", "u1"), **LABEL_KW)
    _coalesce_enqueue(q, _item("B", "u2"), **LABEL_KW)   # gộp

    # Mô phỏng worker: get() luôn đi kèm task_done().
    async def _drain():
        while not q.empty():
            q.get_nowait()
            q.task_done()
        await asyncio.wait_for(q.join(), timeout=1.0)

    asyncio.run(_drain())  # nếu rò unfinished thì join() treo và test fail


def test_tts_dung_counter_rieng():
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    _coalesce_enqueue(q, {"text": "một", "utterance_id": "u1"}, label="TTS",
                      merged_counter="queue.tts_merged", dropped_counter="queue.tts_dropped")
    before = _counter("queue.tts_merged")
    _coalesce_enqueue(q, {"text": "hai", "utterance_id": "u2"}, label="TTS",
                      merged_counter="queue.tts_merged", dropped_counter="queue.tts_dropped")
    assert _counter("queue.tts_merged") == before + 1


def test_loi_ra_cua_handler_dung_ham_gop():
    """Chốt ở mức mã nguồn: 2 chỗ enqueue final KHÔNG được quay lại `put_nowait` + drop."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "backend" / "ws" / "handler.py").read_text(encoding="utf-8")
    assert "queue.translation_merged" in src
    assert "queue.tts_merged" in src
    assert "_coalesce_enqueue(" in src
    # Không còn khối "QueueFull -> bỏ qua" cho 2 hàng đợi final.
    assert "Hàng đợi Translation đầy, bỏ qua" not in src
    assert "Hàng đợi TTS đầy, bỏ qua câu này" not in src


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("a", "b", "a b"),
        ("  a  ", "b", "a b"),
        ("", "b", "b"),
        ("a", "", "a"),
        ("", "", ""),
    ],
)
def test_join_texts(a, b, expected):
    assert _join_texts(a, b) == expected
