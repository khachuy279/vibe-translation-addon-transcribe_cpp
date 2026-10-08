"""test_76 — Test KHÔNG được ghi vào cây `report/` đã commit khi chưa được yêu cầu.

BỐI CẢNH (phát hiện 2026-10-08, khi rà soát dead code / chồng chéo)

`conftest.report_dir` trước đây trả thẳng `report/` thật. Trong 6 test ghi báo cáo
benchmark, `test_04_commit_logic` là test DUY NHẤT không có marker `slow` — nên nó
chạy trong MỌI lần `pytest` mặc định và ghi đè `report/04_commit_logic/report.md`
đã commit (file này chứa timestamp nên luôn khác). Hệ quả:

- `git status` bẩn sau mỗi lần chạy test;
- báo cáo đo lường suýt bị commit nhầm — đã phải `git commit --amend` để loại ra.

Từ 2026-10-08 `report_dir` mặc định trỏ vào thư mục TẠM; muốn ghi báo cáo thật phải
bật `WRITE_BENCH_REPORT=1`. File này chốt lại hợp đồng đó.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backend.tests import bench_report_paths as brp

TESTS_DIR = Path(__file__).resolve().parent
TEST_04 = TESTS_DIR / "test_04_commit_logic.py"


# ── 1. Hàm phân giải đường dẫn ────────────────────────────────────────────────

def test_mac_dinh_tra_ve_thu_muc_tam(tmp_path, monkeypatch):
    """Không bật cờ ⇒ phải trỏ vào thư mục TẠM, không đụng `report/`."""
    monkeypatch.delenv(brp.WRITE_BENCH_REPORT_ENV, raising=False)

    got = brp.resolve_report_dir(tmp_path)

    assert got == tmp_path / "report"
    assert got != brp.REAL_REPORT_DIR, "trả về cây report THẬT khi chưa được yêu cầu"
    assert brp.REAL_REPORT_DIR not in got.parents


def test_bat_co_thi_tra_ve_cay_report_that(tmp_path, monkeypatch):
    """Bật cờ ⇒ trỏ vào `report/` thật (để chủ động cập nhật báo cáo)."""
    monkeypatch.setenv(brp.WRITE_BENCH_REPORT_ENV, "1")

    assert brp.resolve_report_dir(tmp_path) == brp.REAL_REPORT_DIR


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on", " on "])
def test_gia_tri_bat_hop_le(monkeypatch, raw):
    monkeypatch.setenv(brp.WRITE_BENCH_REPORT_ENV, raw)
    assert brp.write_real_bench_reports() is True


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "off", "  ", "maybe"])
def test_gia_tri_tat_hop_le(monkeypatch, raw):
    monkeypatch.setenv(brp.WRITE_BENCH_REPORT_ENV, raw)
    assert brp.write_real_bench_reports() is False


def test_thieu_bien_moi_truong_thi_tat(monkeypatch):
    monkeypatch.delenv(brp.WRITE_BENCH_REPORT_ENV, raising=False)
    assert brp.write_real_bench_reports() is False


# ── 2. Fixture `report_dir` phải thật sự nối vào hàm trên ─────────────────────

def test_fixture_report_dir_khong_tro_vao_cay_that(report_dir: Path):
    """Fixture thật (không phải hàm) không được trỏ vào cây `report/` đã commit."""
    if brp.write_real_bench_reports():  # pragma: no cover - chế độ ghi báo cáo thật
        pytest.skip(
            f"đang chạy với {brp.WRITE_BENCH_REPORT_ENV}=1 — chế độ ghi báo cáo thật"
        )

    assert report_dir != brp.REAL_REPORT_DIR
    assert brp.REAL_REPORT_DIR not in report_dir.parents
    # Không được nằm ở đâu đó bên trong repo.
    assert brp.PROJECT_ROOT not in report_dir.parents


# ── 3. Phạm vi của rào chắn ───────────────────────────────────────────────────

def test_guard_chi_canh_bao_cao_benchmark():
    """Rào chắn chỉ canh 7 thư mục báo cáo benchmark, KHÔNG canh tài liệu người viết.

    Bản đầu của rào chắn băm CẢ cây `report/` và đã báo động sai ngay lần chạy đầy
    đủ đầu tiên: chính người viết đang sửa `report/audit/26_…md` trong lúc suite
    chạy. Sửa tài liệu phân tích là việc bình thường, không được làm đỏ bộ test.
    """
    digest = brp.report_tree_digest()

    assert digest, "không thấy file báo cáo benchmark nào để canh — rào chắn vô dụng"

    for key in digest:
        top = key.split("/", 1)[0]
        assert top in brp.GUARDED_REPORT_SUBDIRS, (
            f"`{key}` nằm ngoài phạm vi được canh — rào chắn đang canh quá rộng"
        )

    # Chứng minh việc loại trừ là THẬT, không phải rỗng: `report/audit/` có file trên
    # đĩa nhưng không file nào lọt vào phạm vi canh.
    audit_dir = brp.REAL_REPORT_DIR / "audit"
    assert audit_dir.is_dir(), "thiếu `report/audit/` — test này sẽ không chứng minh được gì"
    on_disk = [
        p.relative_to(brp.REAL_REPORT_DIR).as_posix()
        for p in audit_dir.rglob("*")
        if p.is_file()
    ]
    assert on_disk, "`report/audit/` rỗng — test này sẽ không chứng minh được gì"
    assert not (set(on_disk) & set(digest)), "tài liệu người viết lọt vào phạm vi canh"


# ── 4. Chốt đường ghi của test_04 (nguyên nhân gốc) ───────────────────────────

def _func_of(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Không tìm thấy hàm {name} trong {path.name}")


def test_test_04_ghi_bao_cao_qua_fixture_report_dir():
    """Test ghi báo cáo của phase 4 PHẢI lấy đường dẫn từ fixture `report_dir`.

    Nếu ai đó đổi lại thành hằng số module trỏ vào `report/` thật thì guard này đổ —
    đó chính là cấu hình đã gây ra việc ghi đè báo cáo đã commit.
    """
    fn = _func_of(TEST_04, "test_commit_benchmark_and_generate_report")
    params = [a.arg for a in fn.args.args]

    assert "report_dir" in params, (
        "test_commit_benchmark_and_generate_report không nhận `report_dir` nữa — "
        "nó đang lấy đường dẫn báo cáo từ nguồn khác, nhiều khả năng là cây report thật"
    )


def test_test_04_khong_tu_dinh_duong_dan_report():
    """`test_04_commit_logic` không được tự dựng đường dẫn vào cây `report/`."""
    tree = ast.parse(TEST_04.read_text(encoding="utf-8"))

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value_src = ast.unparse(node.value)
        targets = ", ".join(ast.unparse(t) for t in node.targets)
        assert '"report"' not in value_src and "'report'" not in value_src, (
            f"hằng số module `{targets} = {value_src}` trỏ vào cây report thật; "
            "phải lấy qua fixture `report_dir`"
        )
