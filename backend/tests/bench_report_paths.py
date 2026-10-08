"""Đường dẫn ghi báo cáo benchmark của bộ test.

Tách riêng khỏi `conftest.py` vì hai lý do:

1. `backend/tests/` KHÔNG phải package (không có `__init__.py`), nên
   `from backend.tests import conftest` sẽ chạy conftest LẦN THỨ HAI — ngoài
   instance plugin mà pytest đã nạp. Module này import nhẹ, không kéo theo thiết
   lập DLL/model của conftest.
2. `test_06_tts_benchmark` / `test_07_e2e_comparison` không dùng fixture nào (chúng
   ghi báo cáo trong `main()`, chỉ chạy khi gọi tay) — nếu sau này cần, chúng import
   được hàm ở đây thay vì phải bắt chước cơ chế fixture.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

#: Thư mục gốc dự án (`backend/tests/bench_report_paths.py` → lên 2 cấp).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Cây báo cáo THẬT đã commit vào git.
REAL_REPORT_DIR = PROJECT_ROOT / "report"

#: Cờ môi trường cho phép ghi vào `REAL_REPORT_DIR`. Mặc định TẮT.
WRITE_BENCH_REPORT_ENV = "WRITE_BENCH_REPORT"

#: Các thư mục báo cáo BENCHMARK mà test được phép ghi ra. Đây cũng là phạm vi mà
#: rào chắn `_guard_real_report_tree` canh. KHÔNG gồm `report/audit/**`,
#: `report/12_qwen35_translation_ab/**`… — đó là tài liệu do NGƯỜI viết, sửa chúng
#: là việc bình thường và không được làm đỏ bộ test.
GUARDED_REPORT_SUBDIRS = (
    "01_core_audio",
    "02_vad",
    "03_asr",
    "04_commit_logic",
    "05_translation",
    "06_tts",
    "07_final_e2e_comparison",
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def write_real_bench_reports() -> bool:
    """True nếu người dùng CHỦ ĐỘNG yêu cầu ghi báo cáo vào `report/` thật."""
    return os.environ.get(WRITE_BENCH_REPORT_ENV, "").strip().lower() in _TRUTHY


def resolve_report_dir(tmp_base: Path) -> Path:
    """Thư mục gốc để test ghi báo cáo benchmark.

    MẶC ĐỊNH trả về thư mục TẠM (`tmp_base/report`), để `pytest` không bao giờ làm
    bẩn cây làm việc. Chỉ trả `REAL_REPORT_DIR` khi cờ môi trường được bật tường minh:

        $env:WRITE_BENCH_REPORT = "1"; pytest -m slow
    """
    if write_real_bench_reports():
        return REAL_REPORT_DIR
    return tmp_base / "report"


def report_tree_digest() -> dict:
    """Băm mọi file trong các thư mục báo cáo BENCHMARK (rỗng nếu chưa có gì).

    Khoá là đường dẫn TƯƠNG ĐỐI so với `REAL_REPORT_DIR` (dùng `/` trên mọi hệ điều
    hành, nhờ `Path.relative_to` + `as_posix`).
    """
    digest: dict = {}
    for sub in GUARDED_REPORT_SUBDIRS:
        base = REAL_REPORT_DIR / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                digest[path.relative_to(REAL_REPORT_DIR).as_posix()] = (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
    return digest
