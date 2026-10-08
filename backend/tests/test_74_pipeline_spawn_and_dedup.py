"""test_74 — Hai lớp phiên PHẢI dùng chung `spawn_background_task`, và `_is_duplicate` không
được khai tham số thừa.

BỐI CẢNH (§5.7 của `report/audit/26_RA_SOAT_CODE_CHET_VA_CHONG_CHEO.md`)

`SessionState._spawn` và `LookaheadSessionState._spawn` từng **giống hệt nhau từng byte** (khác
đúng một dòng log). `LookaheadSessionState._is_duplicate` khai `pts_end` nhưng **không dùng**.

Phần CỐ Ý khác nhau của §5.7 (chính sách hàng đợi TTS, `_schedule_*_model_switch`, chuỗi 3 lớp
`send_json`) KHÔNG bị gộp — đã ghi lý do vào docstring từng chỗ; xem §5.7 của báo cáo.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path

import pytest

from backend.ws.lookahead_handler import LookaheadSessionState
from backend.ws.session import SessionState, spawn_background_task

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "backend" / "ws"
EXCLUDE_DIRS = {"bin", "models", "__pycache__", ".venv", ".git", "external", ".research", "tests"}


def _py_sources() -> list[Path]:
    return sorted(
        p
        for p in (ROOT / "backend").rglob("*.py")
        if not (EXCLUDE_DIRS & set(p.relative_to(ROOT / "backend").parts))
    )


# ──────────────────────────────────────────────── chốt mã nguồn


def test_create_task_chi_con_mot_cho() -> None:
    """Chốt: hai lớp phiên không được tự tạo task nền theo hai bản khác nhau."""
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in _py_sources()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"get_running_loop\(\)\s*\.\s*create_task", line)
    ]
    assert len(offenders) == 1, (
        "`create_task` phải chỉ xuất hiện MỘT lần (trong `spawn_background_task`), "
        f"nhưng thấy {len(offenders)} chỗ:\n  " + "\n  ".join(offenders)
    )
    assert "session.py" in offenders[0], (
        f"`spawn_background_task` phải nằm ở `ws/session.py`, thấy ở {offenders[0]}"
    )


@pytest.mark.parametrize("cls", [SessionState, LookaheadSessionState])
def test_hai_lop_deu_uy_quyen_spawn(cls: type) -> None:
    """Cả hai lớp phải gọi helper dùng chung, không tự cài lại."""
    src = inspect.getsource(cls._spawn)
    assert "spawn_background_task" in src, f"{cls.__name__}._spawn chưa uỷ quyền cho helper dùng chung"


def test_is_duplicate_khong_con_tham_so_thua() -> None:
    """`pts_end` từng được khai nhưng không dùng — chữ ký không được nói dối."""
    params = list(inspect.signature(LookaheadSessionState._is_duplicate).parameters)
    assert params == ["self", "pts_start", "text"], (
        f"`_is_duplicate` phải có chữ ký (self, pts_start, text), nhận được {params}"
    )


# ──────────────────────────────────────────────── hành vi helper


def test_spawn_khong_co_event_loop_thi_bo_qua_an_toan() -> None:
    """Test đồng bộ (không có event loop) ⇒ bỏ qua, KHÔNG ném."""
    sink: set = set()

    async def _never() -> None:  # pragma: no cover - không bao giờ chạy
        raise AssertionError("không được chạy khi thiếu event loop")

    coro = _never()
    spawn_background_task(coro, sink)  # không được ném
    # Helper thoát sớm nên KHÔNG tạo task ⇒ tự đóng coroutine để không có RuntimeWarning
    # "coroutine was never awaited".
    coro.close()
    assert sink == set(), "thiếu event loop thì không được thêm task vào sink"


def test_spawn_giu_tham_chieu_roi_nha_khi_xong() -> None:
    """Task phải được GIỮ trong sink lúc chạy (GC không thu hồi) và nhả ra khi xong."""
    sink: set = set()

    async def _main() -> None:
        async def _work() -> str:
            await asyncio.sleep(0)
            return "xong"

        spawn_background_task(_work(), sink)
        assert len(sink) == 1, "task đang chạy phải nằm trong sink (chống GC thu hồi giữa đường)"
        await asyncio.gather(*sink)
        # `add_done_callback` chạy ở vòng lặp kế tiếp.
        await asyncio.sleep(0)

    asyncio.run(_main())
    assert sink == set(), "task xong phải được nhả khỏi sink"
