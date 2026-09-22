"""Module logging tập trung cho Backend.

Hỗ trợ:
- Định dạng log chuyên nghiệp với màu sắc ANSI trên Windows và Linux.
- Định danh module rõ ràng, in ra console với emoji trực quan.
- Độ chính xác microsecond/millisecond cho phân tích độ trễ real-time.
- Ngăn chặn lỗi charmap/emoji trên Windows console.

============================= QUY ƯỚC LOGGING =============================
(Thống nhất toàn backend — có test tầng A canh: `test_20_logging_convention.py`)

1. TAG: MỌI lời gọi phải truyền `extra={"module_tag": TAG}` với TAG thuộc danh sách:
       CORE, VAD, ASR, ASR_PREVIEW, ASR_COMMIT, TRANSLATE, TTS, WS, MAIN, METRICS
   Không dựa vào tên logger để suy ra tag (trước đây 86 lời gọi bị in tag sai chỗ).

2. KHÔNG lặp tag trong thông điệp: formatter đã in `[TAG]`, nên viết
       logger.info("Đã nạp model X", extra={"module_tag": "ASR"})
   chứ KHÔNG viết `"[ASR] Đã nạp model X"`.
   Chỉ giữ tiền tố PHA khi cần phân biệt vòng đời: `[STARTUP]`, `[SHUTDOWN]`.

3. MỘT DÒNG, tiếng Việt có dấu, ngắn gọn, số liệu kèm đơn vị (`ms`, `s`, `MB`).
   Câu có ngữ cảnh câu nói thì mở đầu bằng `[utt=<8 ký tự>]`.

4. MỨC LOG:
   - DEBUG   : chi tiết đường nóng (mỗi frame/chunk/preview), chẩn đoán.
   - INFO    : vòng đời (nạp/giải phóng model, kết nối, kết quả mỗi câu).
   - WARNING : vẫn chạy được nhưng có suy giảm (bỏ câu, nghẽn hàng đợi, fallback).
   - ERROR   : hỏng chức năng, cần người dùng biết.

5. KHÔNG emoji trong log.
==========================================================================
"""

import logging
import os
import sys
import threading
import time
from typing import Optional

# Cấu hình UTF-8 cho stdout/stderr trên Windows để tránh crash emoji
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass

# Bảng mã màu ANSI
class LogColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


# Emoji cho từng cấp độ và module
LEVEL_ICONS = {
    logging.DEBUG: "🔍",
    logging.INFO: "ℹ️",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "🚨",
}

MODULE_COLORS = {
    "CORE": LogColors.CYAN,
    "MAIN": LogColors.BOLD + LogColors.WHITE,
    "VAD": LogColors.GREEN,
    "ASR": LogColors.BLUE,
    "ASR_PREVIEW": LogColors.DIM + LogColors.BLUE,
    "ASR_COMMIT": LogColors.BOLD + LogColors.BLUE,
    "TRANSLATE": LogColors.MAGENTA,
    "TTS": LogColors.YELLOW,
    "WS": LogColors.CYAN,
    "METRICS": LogColors.GREEN,
    # Tầng SEG: trace từng nhịp preview để tinh chỉnh mốc ngắt câu.
    "SEG": LogColors.BOLD + LogColors.MAGENTA,
}


class ColoredFormatter(logging.Formatter):
    """Formatter tùy biến thêm màu sắc và biểu tượng trực quan."""

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and (sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR")))

    def format(self, record: logging.LogRecord) -> str:
        created = record.created
        msec = int((created - int(created)) * 1000)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created)) + f".{msec:03d}"
        
        icon = LEVEL_ICONS.get(record.levelno, "•")
        levelname = record.levelname
        
        module_tag = getattr(record, "module_tag", record.name.replace("backend.", "").upper())
        
        if self.use_color:
            level_color = {
                logging.DEBUG: LogColors.DIM,
                logging.INFO: LogColors.GREEN,
                logging.WARNING: LogColors.YELLOW,
                logging.ERROR: LogColors.RED,
                logging.CRITICAL: LogColors.BG_RED + LogColors.WHITE,
            }.get(record.levelno, LogColors.RESET)
            
            tag_color = MODULE_COLORS.get(module_tag, LogColors.CYAN)
            
            header = (
                f"{LogColors.DIM}{time_str}{LogColors.RESET} "
                f"{level_color}[{levelname:<5}]{LogColors.RESET} "
                f"{tag_color}[{module_tag}]{LogColors.RESET} "
            )
        else:
            header = f"{time_str} [{levelname:<5}] [{module_tag}] "
            
        message = record.getMessage()
        
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                message = f"{message}\n{record.exc_text}"
                
        return f"{header}{message}"


class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler an toàn chống WinError 1 và UnicodeEncodeError trên Windows.

    P4.1: trước đây `flush()` được gọi sau MỖI record, và `_lock` là class-level nên
    serialize mọi thread ghi log. Trên hot path (~8-15 record/giây khi đang stream)
    đây là nguồn jitter không cần thiết. Nay chỉ flush khi:
      - level >= WARNING (cần thấy ngay), hoặc
      - đã quá `_FLUSH_INTERVAL_SEC` kể từ lần flush trước.

    F-41 (log bị in 2 lần): hai rào chắn.
      1. KHÔNG bao giờ ghi lại lần hai. Bản cũ ghi `msg`, nếu `write()` ném lỗi SAU KHI đã
         ghi được một phần (console Windows có thể làm vậy) thì nhánh dự phòng ghi lại
         toàn bộ ⇒ **dòng đó hiện 2 lần**. Nay chuỗi được mã hoá an toàn TRƯỚC, chỉ có
         MỘT lần ghi; nếu vẫn lỗi thì bỏ qua chứ không ghi lại.
      2. Chặn trùng lặp tức thời: cùng logger + cùng level + cùng nội dung trong
         `_DEDUP_WINDOW_SEC` ⇒ bỏ qua (đếm vào `suppressed`). Có thể chỉnh bằng biến môi
         trường `LOG_DEDUP_MS` (0 = tắt).
    """

    _lock = threading.RLock()
    _FLUSH_INTERVAL_SEC = 0.25
    _last_flush = 0.0
    _DEDUP_WINDOW_SEC = 0.05
    _last_key = None
    _last_key_at = 0.0
    suppressed_records = 0

    @classmethod
    def _dedup_window_sec(cls) -> float:
        raw = os.environ.get("LOG_DEDUP_MS")
        if raw is None:
            return cls._DEDUP_WINDOW_SEC
        try:
            return max(0.0, float(raw) / 1000.0)
        except ValueError:
            return cls._DEDUP_WINDOW_SEC

    def _is_immediate_duplicate(self, record: logging.LogRecord, msg: str) -> bool:
        """True nếu vừa in y hệt dòng này trong một khoảng rất ngắn (chống in 2 lần).

        F-41b (2026-09-22): khoá chống trùng PHẢI không phụ thuộc mốc thời gian. Bản trước
        dùng `msg` = `self.format(record)` — chuỗi này chứa `%(asctime)s` tới **mili-giây**,
        nên hai bản ghi "giống hệt nhau" luôn khác nhau ở phần ms và cơ chế chống trùng
        **thực tế không bao giờ chạy** (test `test_19` chỉ xanh khi cả hai rơi vào cùng một
        ms — đỏ ngay khi máy bận). Nay khoá = (logger, level, nội dung KHÔNG thời gian,
        module_tag). Cửa sổ thời gian `_dedup_window_sec()` vẫn giới hạn phạm vi chặn.
        """
        window = self._dedup_window_sec()
        if window <= 0:
            return False
        try:
            content = record.getMessage()
        except Exception:  # noqa: BLE001 — record hỏng thì đành so theo chuỗi đã format
            content = msg
        key = (record.name, record.levelno, content, getattr(record, "module_tag", ""))
        now = time.monotonic()
        if key == SafeStreamHandler._last_key and (now - SafeStreamHandler._last_key_at) <= window:
            SafeStreamHandler.suppressed_records += 1
            return True
        SafeStreamHandler._last_key = key
        SafeStreamHandler._last_key_at = now
        return False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if self._is_immediate_duplicate(record, msg):
                return

            stream = self.stream
            enc = getattr(stream, "encoding", "utf-8") or "utf-8"
            # Mã hoá an toàn TRƯỚC => lần ghi đầu tiên không thể ném UnicodeEncodeError.
            text = msg + self.terminator
            try:
                text.encode(enc)
            except (UnicodeEncodeError, LookupError):
                text = text.encode(enc, errors="backslashreplace").decode(enc, errors="replace")

            with self._lock:
                written = False
                try:
                    stream.write(text)
                    written = True
                except OSError:
                    # KHÔNG ghi lại: nếu write() đã ghi một phần rồi mới lỗi thì ghi lại
                    # chính là nguyên nhân dòng bị in 2 lần.
                    pass

                if written:
                    now = time.monotonic()
                    if (
                        record.levelno >= logging.WARNING
                        or (now - SafeStreamHandler._last_flush) >= self._FLUSH_INTERVAL_SEC
                    ):
                        try:
                            self.flush()
                            SafeStreamHandler._last_flush = now
                        except Exception:
                            pass
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        try:
            super().handleError(record)
        except Exception:
            # Triệt tiêu exception trong handleError để không sập luồng ứng dụng
            pass


def get_logger(name: str = "backend", level: int = logging.INFO) -> logging.Logger:
    """Khởi tạo hoặc lấy logger cấu hình chuẩn cho Backend."""
    logger_instance = logging.getLogger(name)
    logger_instance.propagate = False
    
    # Giữ tối đa 1 SafeStreamHandler duy nhất, xóa bỏ các handler thừa
    while len(logger_instance.handlers) > 1:
        logger_instance.removeHandler(logger_instance.handlers[-1])

    if not logger_instance.handlers:
        handler = SafeStreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter(use_color=True))
        logger_instance.addHandler(handler)
        logger_instance.setLevel(level)
        
    return logger_instance


# P4.1: logger "backend" trước đây ở mức DEBUG, khiến các `logger.debug` trong hot
# path (ví dụ fallback stream mỗi lần inference) luôn được ghi. Nay dùng INFO.
logger = get_logger("backend", level=logging.INFO)
