"""test_75 — Payload `main.py` gửi cho phiên PHẢI khớp hợp đồng `SessionConfigPayload`.

BỐI CẢNH (bug tìm thấy 2026-10-08 khi rà §5.9)

`SessionConfigPayload` khai `extra="ignore"`, nên **bất kỳ khoá nào không phải field name hay
alias đều bị NUỐT IM LẶNG** — không lỗi, không log, không tác dụng.

Bug thật đã xảy ra: `main.py` gửi `stability_duration_ms` (mili-giây) trong khi payload chỉ có
`stability_duration_sec` (giây, alias `stabilityDurationSec`). Hệ quả: kéo slider "Stable for (ms)"
thì `config.sentence` TOÀN CỤC được cập nhật, nhưng **phiên đang chạy không nhận gì** qua đường
REST. Không test nào bắt được vì đường WS gửi đúng `stabilityDurationSec`.

Từ 2026-10-08 `main.py` KHÔNG còn dựng dict thủ công: nó gọi `session_payload_from_rest(req)`,
và hàm đó validate qua `SessionConfigPayload`. Bộ test này kiểm chứng **hành vi của hàm đó**
(không quét source text nữa) + chốt rằng dict viết tay không quay trở lại.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.main import SwitchModelRequest, session_payload_from_rest
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


def _full_request() -> SwitchModelRequest:
    """Request có ĐỦ trường — để phép kiểm không bỏ sót khoá nào chỉ vì nó None."""
    return SwitchModelRequest(
        model_id="qwen3-asr-1.7b",
        asr_engine="qwen3-asr-1.7b",
        vad_engine="silero-vad",
        vad_threshold=0.42,
        silence_duration_ms=600,
        source_lang="ja",
        target_lang="vi",
        translation_model="index-translate-2b",
        min_words_to_commit=4,
        stability_duration_ms=150.0,
        stability_min_duration_sec=2.5,
        stability_min_words=4,
        split_on_stability=True,
        trace_stability=False,
        hold_short_sentence=True,
        tts_enabled=True,
        tts_voice="voice-a.wav",
        tts_speed=1.25,
        lookahead_enabled=True,
    )


# ──────────────────────────────────────────────── hợp đồng khoá


def test_moi_khoa_payload_deu_duoc_hieu() -> None:
    """BẤT BIẾN CHÍNH: khoá lạ ⇒ bị `extra="ignore"` nuốt im lặng."""
    payload = session_payload_from_rest(_full_request())
    assert payload, "payload rỗng — request đầy đủ phải sinh ra payload khác rỗng"
    unknown = sorted(set(payload) - _accepted_keys())
    assert not unknown, (
        f"payload gửi cho phiên chứa {len(unknown)} khoá mà `SessionConfigPayload` KHÔNG hiểu "
        f"⇒ bị nuốt im lặng:\n  " + "\n  ".join(unknown)
    )


def test_stability_duration_gui_bang_giay() -> None:
    """Chốt hồi quy trực tiếp cho bug đã sửa: payload phiên dùng GIÂY, không dùng mili-giây."""
    payload = session_payload_from_rest(_full_request())
    assert payload.get("stability_duration_sec") == 0.15, (
        f"150 ms phải thành 0.15 giây, nhận được {payload.get('stability_duration_sec')!r}"
    )
    assert "stability_duration_ms" not in payload


def test_khong_gui_truong_rest_only() -> None:
    """Đổi model (và cờ lookahead) do đường RIÊNG lo — không được lẫn vào `apply_config`."""
    payload = session_payload_from_rest(_full_request())
    for key in ("model_id", "asr_engine", "translation_model", "lookahead_enabled"):
        assert key not in payload, (
            f"`{key}` không được nằm trong payload cấu hình phiên (xem `_REST_ONLY_FIELDS`)"
        )


def test_bo_qua_truong_none() -> None:
    """Chỉ gửi trường người dùng THẬT SỰ đặt — không ghi đè bằng None."""
    payload = session_payload_from_rest(SwitchModelRequest(target_lang="en"))
    assert payload == {"target_lang": "en"}


def test_nhan_lai_duoc_bang_session_payload() -> None:
    """Đầu ra phải `model_validate` lại được và giữ nguyên giá trị (round-trip)."""
    payload = session_payload_from_rest(_full_request())
    parsed = SessionConfigPayload.model_validate(payload)
    assert parsed.vad_threshold == 0.42
    assert parsed.silence_duration_ms == 600
    assert parsed.stability_duration_sec == 0.15
    assert parsed.source_lang == "ja"
    assert parsed.tts_speed == 1.25


def test_khong_quay_lai_dict_viet_tay() -> None:
    """Chốt mã nguồn: `main.py` phải dùng helper, không gán `session_payload[...]` thủ công."""
    src = MAIN.read_text(encoding="utf-8")
    offenders = [
        f"main.py:{i}"
        for i, line in enumerate(src.splitlines(), 1)
        if re.search(r"session_payload\[\s*['\"]", line)
    ]
    assert not offenders, (
        "`main.py` quay lại dựng `session_payload` bằng dict viết tay:\n  " + "\n  ".join(offenders)
        + "\nDùng `session_payload_from_rest(req)` để mọi khoá đi qua validate của "
        "`SessionConfigPayload`."
    )
    assert "session_payload_from_rest(req)" in src, "main.py phải gọi helper"


# ──────────────────────────────────────────────── cơ chế gây bug


def test_khoa_la_bi_nuot_that_su() -> None:
    """Ghi lại CƠ CHẾ của bug để không ai tưởng `extra="ignore"` là vô hại."""
    parsed = SessionConfigPayload.model_validate({"stability_duration_ms": 150.0})
    assert parsed.stability_duration_sec is None, (
        "nếu Pydantic đổi hành vi `extra`, test này phải được xem lại — "
        "lúc đó khoá lạ sẽ NÉM lỗi thay vì bị nuốt"
    )


@pytest.mark.parametrize(
    ("payload", "expected_sec"),
    [
        # Đường WS: extension gửi camelCase + GIÂY.
        ({"stabilityDurationSec": 0.15}, 0.15),
        # Đường REST: helper gửi snake_case + GIÂY.
        ({"stability_duration_sec": 0.15}, 0.15),
    ],
)
def test_hai_duong_gui_stability_deu_doc_duoc(payload: dict, expected_sec: float) -> None:
    """Cả WS lẫn REST phải cho cùng kết quả — trước đây REST cho `None`."""
    parsed = SessionConfigPayload.model_validate(payload)
    assert parsed.stability_duration_sec == expected_sec
