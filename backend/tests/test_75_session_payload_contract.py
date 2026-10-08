"""test_75 — Mọi khoá `main.py` gửi cho phiên PHẢI được `SessionConfigPayload` hiểu.

BỐI CẢNH (bug tìm thấy 2026-10-08 khi rà §5.9)

`main.py` dựng `session_payload` bằng dict thủ công rồi gọi `sess.apply_config(...)` cho các
phiên đang chạy. `SessionConfigPayload` khai `extra="ignore"`, nên **bất kỳ khoá nào không phải
field name hay alias đều bị NUỐT IM LẶNG** — không lỗi, không log, không tác dụng.

Bug thật đã xảy ra: `main.py` gửi `stability_duration_ms` (mili-giây) trong khi payload chỉ có
`stability_duration_sec` (giây, alias `stabilityDurationSec`). Hệ quả: kéo slider "Stable for (ms)"
thì `config.sentence` TOÀN CỤC được cập nhật, nhưng **phiên đang chạy không nhận gì** qua đường
REST. Không test nào bắt được vì đường WS (`buildWsConfig`) gửi đúng `stabilityDurationSec` nên
tính năng vẫn chạy.

Bộ test này khoá bất biến tổng quát: **tập khoá `main.py` gửi ⊆ (field name ∪ alias)**.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.ws.session import SessionConfigPayload

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "backend" / "main.py"


def _accepted_keys() -> set[str]:
    """Tên field + mọi alias mà `SessionConfigPayload` chấp nhận."""
    keys: set[str] = set()
    for name, field in SessionConfigPayload.model_fields.items():
        keys.add(name)
        alias = getattr(field, "alias", None)
        if alias:
            keys.add(alias)
    return keys


def _session_payload_keys() -> set[str]:
    """Các khoá `main.py` gán vào `session_payload`."""
    src = MAIN.read_text(encoding="utf-8")
    return set(re.findall(r'session_payload\[\s*"([^"]+)"\s*\]\s*=', src))


def test_main_gan_duoc_it_nhat_mot_khoa() -> None:
    """Chốt chính phép quét còn hoạt động (nếu `main.py` đổi cách dựng payload, test phải đỏ)."""
    keys = _session_payload_keys()
    assert len(keys) >= 10, (
        f"chỉ quét được {len(keys)} khoá `session_payload` — cách dựng payload trong main.py đã đổi, "
        "hãy cập nhật phép quét của test này"
    )


def test_moi_khoa_session_payload_deu_duoc_hieu() -> None:
    """BẤT BIẾN CHÍNH: khoá lạ ⇒ bị `extra="ignore"` nuốt im lặng."""
    accepted = _accepted_keys()
    unknown = sorted(_session_payload_keys() - accepted)
    assert not unknown, (
        f"`main.py` gửi {len(unknown)} khoá mà `SessionConfigPayload` KHÔNG hiểu "
        f"⇒ bị nuốt im lặng:\n  " + "\n  ".join(unknown) + "\n"
        "Sửa tên khoá cho khớp field/alias của payload, hoặc thêm field tương ứng."
    )


def test_stability_duration_gui_bang_giay() -> None:
    """Chốt hồi quy trực tiếp cho bug đã sửa: payload phiên dùng GIÂY, không dùng mili-giây."""
    keys = _session_payload_keys()
    assert "stability_duration_sec" in keys, (
        "`main.py` phải gửi `stability_duration_sec` (giây) cho phiên"
    )
    assert "stability_duration_ms" not in keys, (
        "`main.py` còn gửi `stability_duration_ms` — payload phiên KHÔNG có khoá này "
        "nên nó bị nuốt im lặng"
    )


@pytest.mark.parametrize(
    ("payload", "expected_sec"),
    [
        # Đường WS: extension gửi camelCase + GIÂY.
        ({"stabilityDurationSec": 0.15}, 0.15),
        # Đường REST: main.py gửi snake_case + GIÂY.
        ({"stability_duration_sec": 0.15}, 0.15),
    ],
)
def test_hai_duong_gui_stability_deu_doc_duoc(payload: dict, expected_sec: float) -> None:
    """Cả WS lẫn REST phải cho cùng kết quả — trước đây REST cho `None`."""
    parsed = SessionConfigPayload.model_validate(payload)
    assert parsed.stability_duration_sec == expected_sec


def test_khoa_la_bi_nuot_that_su() -> None:
    """Ghi lại CƠ CHẾ của bug để không ai tưởng `extra="ignore"` là vô hại."""
    parsed = SessionConfigPayload.model_validate({"stability_duration_ms": 150.0})
    assert parsed.stability_duration_sec is None, (
        "nếu Pydantic đổi hành vi `extra`, test này phải được xem lại — "
        "lúc đó khoá lạ sẽ NÉM lỗi thay vì bị nuốt"
    )
