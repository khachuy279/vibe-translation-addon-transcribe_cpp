"""Test tầng A — F-41: log KHÔNG được in 2 lần.

Người dùng báo hai dòng log VAD giống hệt nhau (cùng mili-giây) khi khởi động. Nguyên nhân
khả dĩ nhất nằm ở `SafeStreamHandler.emit`: nó ghi `stream.write(msg)`, nếu lỗi thì **ghi
lại** toàn bộ ⇒ console Windows (có thể lỗi giữa chừng) làm dòng đó hiện 2 lần.

Test ở đây chốt:
1. `write()` lỗi giữa chừng ⇒ handler KHÔNG ghi lại (không sinh dòng thứ hai).
2. Hai record giống hệt nhau trong khoảng rất ngắn ⇒ chỉ in 1 lần.
3. Hai record giống nhau nhưng cách xa nhau (quá cửa sổ chống trùng) ⇒ vẫn in đủ 2 lần.
4. `LOG_DEDUP_MS=0` tắt chống trùng; các dòng khác nhau thì không bị chặn.
5. Ghi được tiếng Việt (không phụ thuộc encoding console).
"""

import logging
import os
import re
import time

import pytest

from backend.utils.logger import ColoredFormatter, SafeStreamHandler


class _FlakyStream:
    """Stream lỗi kiểu console Windows.

    - `fail_after=N`: ghi N ký tự rồi ném OSError (lỗi GIỮA CHỪNG).
    - `fail_after_full=True`: ghi HẾT rồi mới ném OSError (lỗi SAU KHI đã ghi xong) —
      đây chính là ca làm dòng log bị in 2 lần ở bản cũ (vì nhánh dự phòng ghi lại).
    - `fail_after=None`: ghi bình thường.
    """

    encoding = "utf-8"

    def __init__(self, fail_after=None, fail_after_full: bool = False):
        self.chunks = []
        self.fail_after = fail_after
        self.fail_after_full = fail_after_full
        self.flushed = 0

    def write(self, text: str) -> int:
        if self.fail_after_full:
            self.chunks.append(text)
            raise OSError("console write failed SAU KHI đã ghi xong")
        if self.fail_after is not None:
            part, rest = text[: self.fail_after], text[self.fail_after:]
            self.chunks.append(part)
            if rest:
                raise OSError("console write failed giữa chừng")
            return len(text)
        self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        self.flushed += 1


def _make_handler(stream) -> SafeStreamHandler:
    handler = SafeStreamHandler(stream=stream)
    handler.setFormatter(ColoredFormatter(use_color=False))
    return handler


def _record(msg: str = "Loaded FireRed Stream-VAD Model", name: str = "backend", level=logging.INFO):
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )


@pytest.fixture(autouse=True)
def _reset_dedup():
    SafeStreamHandler._last_key = None
    SafeStreamHandler._last_key_at = 0.0
    SafeStreamHandler.suppressed_records = 0
    saved = os.environ.get("LOG_DEDUP_MS")
    yield
    if saved is None:
        os.environ.pop("LOG_DEDUP_MS", None)
    else:
        os.environ["LOG_DEDUP_MS"] = saved


def test_partial_write_is_not_retried():
    """F-41: write() lỗi GIỮA CHỪNG ⇒ không ghi lại phần còn thiếu (không sinh dòng thứ hai).

    LƯU Ý: trước đây test này hardcode `"2026-09-15"`, nên nó chỉ xanh đúng MỘT ngày và
    đỏ từ hôm sau — dù handler hoàn toàn đúng. Nay lấy ngày từ cùng nguồn đồng hồ với
    `%(asctime)s` của logging (`time.strftime`), và vẫn chốt đúng định dạng ngày.
    """
    stream = _FlakyStream(fail_after=10)
    handler = _make_handler(stream)
    handler.emit(_record())

    expected_date = time.strftime("%Y-%m-%d")  # cùng nguồn với %(asctime)s
    assert stream.chunks == [expected_date], (
        f"handler đã ghi lại sau lỗi giữa chừng: {stream.chunks}"
    )
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", stream.chunks[0]), (
        f"phần đầu dòng log phải là ngày ISO, nhận được: {stream.chunks[0]!r}"
    )


def test_full_write_then_error_is_not_reprinted():
    """F-41 (ca đã gây lỗi thật): ghi xong rồi mới lỗi ⇒ TUYỆT ĐỐI không in lại."""
    stream = _FlakyStream(fail_after_full=True)
    handler = _make_handler(stream)
    handler.emit(_record())

    text = "".join(stream.chunks)
    assert text.count("Loaded FireRed") == 1, (
        f"dòng bị in 2 lần khi write() lỗi sau khi đã ghi xong: {text!r}"
    )


def test_write_error_before_any_output_is_silent():
    """Lỗi trước khi ghi được gì ⇒ không có nội dung nào lọt ra, và không ném ra ngoài."""
    stream = _FlakyStream(fail_after=0)
    handler = _make_handler(stream)
    handler.emit(_record())
    assert "".join(stream.chunks) == "", f"có nội dung lọt ra: {stream.chunks!r}"


def test_identical_records_within_window_are_printed_once():
    """Hai record giống hệt nhau trong cùng khoảng ngắn ⇒ chỉ in 1 lần.

    Ghim `LOG_DEDUP_MS` tường minh để không phụ thuộc tốc độ máy. Từ F-41b (2026-09-22)
    khoá chống trùng KHÔNG còn chứa mốc thời gian (xem test ngay dưới), nên bài này xanh
    ổn định kể cả khi hai lần `emit()` rơi vào hai mili-giây khác nhau.
    """
    os.environ["LOG_DEDUP_MS"] = "1000"
    stream = _FlakyStream()
    handler = _make_handler(stream)
    handler.emit(_record())
    handler.emit(_record())

    text = "".join(stream.chunks)
    assert text.count("Loaded FireRed") == 1, f"in trùng: {text!r}"
    assert SafeStreamHandler.suppressed_records == 1


def test_chong_trung_khong_phu_thuoc_moc_thoi_gian_trong_dong_log():
    """F-41b: cùng nội dung nhưng KHÁC mốc ms vẫn phải bị chặn.

    Bug gốc: khoá chống trùng là `self.format(record)` — chuỗi này chứa `%(asctime)s` tới
    mili-giây, nên hai dòng "trùng" luôn khác nhau ⇒ cơ chế chống trùng **không bao giờ
    chạy** trên thực tế (và test cũ chỉ xanh khi cả hai rơi vào cùng một ms).
    """
    os.environ["LOG_DEDUP_MS"] = "1000"
    stream = _FlakyStream()
    handler = _make_handler(stream)

    class _TickingFormatter(ColoredFormatter):
        """Mỗi lần format trả về một mốc thời gian KHÁC NHAU (mô phỏng emit cách >1 ms)."""

        ticks = 0

        def formatTime(self, record, datefmt=None):  # noqa: N802 — API của logging
            _TickingFormatter.ticks += 1
            return f"2026-09-22 21:00:00.{_TickingFormatter.ticks:03d}"

    handler.setFormatter(_TickingFormatter(use_color=False))
    handler.emit(_record())
    handler.emit(_record())

    text = "".join(stream.chunks)
    assert text.count("Loaded FireRed") == 1, f"in trùng dù khác mốc ms: {text!r}"
    assert SafeStreamHandler.suppressed_records == 1


def test_identical_records_far_apart_are_both_printed():
    """Khác thời điểm (ngoài cửa sổ) thì KHÔNG được chặn — tránh giấu log thật."""
    os.environ["LOG_DEDUP_MS"] = "10"
    stream = _FlakyStream()
    handler = _make_handler(stream)
    handler.emit(_record())
    time.sleep(0.05)
    handler.emit(_record())

    text = "".join(stream.chunks)
    assert text.count("Loaded FireRed") == 2


def test_dedup_can_be_disabled():
    os.environ["LOG_DEDUP_MS"] = "0"
    stream = _FlakyStream()
    handler = _make_handler(stream)
    handler.emit(_record())
    handler.emit(_record())
    assert "".join(stream.chunks).count("Loaded FireRed") == 2


def test_different_messages_are_not_suppressed():
    """Chống trùng chỉ áp cho nội dung Y HỆT — không được nuốt các dòng khác nhau."""
    os.environ["LOG_DEDUP_MS"] = "1000"
    stream = _FlakyStream()
    handler = _make_handler(stream)
    handler.emit(_record("Dòng một"))
    handler.emit(_record("Dòng hai"))
    handler.emit(_record("Dòng ba"))
    text = "".join(stream.chunks)
    assert "Dòng một" in text and "Dòng hai" in text and "Dòng ba" in text
    assert SafeStreamHandler.suppressed_records == 0


def test_different_logger_name_is_not_suppressed():
    """Cùng nội dung nhưng khác logger ⇒ là hai sự kiện khác nhau, phải in đủ."""
    os.environ["LOG_DEDUP_MS"] = "1000"
    stream = _FlakyStream()
    handler = _make_handler(stream)
    handler.emit(_record("Giống nhau", name="backend"))
    handler.emit(_record("Giống nhau", name="main"))
    assert "".join(stream.chunks).count("Giống nhau") == 2


def test_vietnamese_text_is_written():
    """Chữ tiếng Việt phải ghi được (không phụ thuộc encoding của console)."""
    stream = _FlakyStream()
    handler = _make_handler(stream)
    handler.emit(_record("VAD engine mặc định 'firered-vad' đã sẵn sàng!"))
    assert "đã sẵn sàng" in "".join(stream.chunks)
