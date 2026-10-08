"""test_73 — Chuẩn hoá config phải nằm ở MỘT chỗ, không viết lặp.

BỐI CẢNH (đã hợp nhất 2026-10-08)

§5.9 — cùng một quy ước config từng bị viết lại 2–3 lần ở `main.py`, `session.py`,
`lookahead_handler.py`:

  * alias `vad_threshold` ↔ `threshold`: 3 chỗ
  * `0 = "tắt"` ⇒ `None`: 3 chỗ
  * kẹp offset đồng bộ ±1500 ms: 2 chỗ
  * `min_words_to_commit` ≥ 0: 3 chỗ (và `lookahead_handler.py` còn **trùng y hệt trong
    cùng một file**: `_apply_config_sync` + `apply_init`)

§5.10 — `main.py` tự kiểm tra lại onnxruntime/aligner dù `env_check.runtime_status()` đã có;
`popup.js` ghi mỗi giá trị settings dưới 2–3 khoá alias; payload REST trộn hai quy ước.

Nay quy ước nằm ở `backend/ws/session.py` (cạnh `SessionConfigPayload`) và mọi nơi uỷ quyền.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.ws.session import (
    SYNC_OFFSET_LIMIT_MS,
    clamp_min_words,
    clamp_sync_offset_ms,
    off_means_none_ms,
    pick_vad_threshold,
)

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
EXCLUDE_DIRS = {"bin", "models", "__pycache__", ".venv", ".git", "external", ".research", "tests"}


def _py_sources() -> list[Path]:
    return sorted(
        p for p in BACKEND.rglob("*.py") if not (EXCLUDE_DIRS & set(p.relative_to(BACKEND).parts))
    )


# ──────────────────────────────────────────────── §5.9 hành vi các helper


class _Payload:
    """Payload giả tối thiểu cho `pick_vad_threshold` (chỉ cần 2 thuộc tính)."""

    def __init__(self, vad_threshold=None, threshold=None):
        self.vad_threshold = vad_threshold
        self.threshold = threshold


def test_pick_vad_threshold_uu_tien_ten_moi() -> None:
    assert pick_vad_threshold(_Payload(vad_threshold=0.7, threshold=0.3)) == 0.7


def test_pick_vad_threshold_roi_ve_alias_cu() -> None:
    assert pick_vad_threshold(_Payload(vad_threshold=None, threshold=0.3)) == 0.3


def test_pick_vad_threshold_khong_co_thi_tra_none() -> None:
    assert pick_vad_threshold(_Payload()) is None
    assert pick_vad_threshold(None) is None


@pytest.mark.parametrize("raw", [0, -1, -500])
def test_off_means_none_ms_tat(raw: int) -> None:
    """0 hoặc âm = "TẮT" ⇒ None (engine tự dùng mặc định của nó)."""
    assert off_means_none_ms(raw) is None


@pytest.mark.parametrize(("raw", "expected"), [(1, 1), (600, 600), (1500, 1500)])
def test_off_means_none_ms_giu_gia_tri_duong(raw: int, expected: int) -> None:
    assert off_means_none_ms(raw) == expected


def test_clamp_min_words_khong_am() -> None:
    assert clamp_min_words(-5) == 0
    assert clamp_min_words(0) == 0
    assert clamp_min_words(4) == 4


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, 0.0), (1500, 1500.0), (-1500, -1500.0), (99999, SYNC_OFFSET_LIMIT_MS),
     (-99999, -SYNC_OFFSET_LIMIT_MS), ("250", 250.0)],
)
def test_clamp_sync_offset_ms_kep_dung(raw, expected) -> None:
    assert clamp_sync_offset_ms(raw) == expected


@pytest.mark.parametrize("raw", [None, "abc", object()])
def test_clamp_sync_offset_ms_khong_nem(raw) -> None:
    """Giá trị popup hỏng KHÔNG được làm sập phiên — trả None thay vì ném."""
    assert clamp_sync_offset_ms(raw) is None


# ──────────────────────────────────────────────── §5.9 chốt mã nguồn


def test_khong_con_viet_lai_quy_uoc_config() -> None:
    """Mọi quy ước §5.9 phải chỉ còn trong `ws/session.py`."""
    patterns = {
        "kẹp ±1500": r"min\(\s*1500",
        '0 = "tắt" ⇒ None (biến tạm)': r"raw_ms if raw_ms > 0 else",
        "alias threshold đọc tay": r"vad_threshold if .* is not None else .*\.threshold",
    }
    for label, pattern in patterns.items():
        offenders = [
            f"{p.relative_to(ROOT)}:{i}"
            for p in _py_sources()
            if p.name != "session.py"
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(pattern, line)
        ]
        assert not offenders, (
            f"còn viết lặp quy ước `{label}`:\n  " + "\n  ".join(offenders) + "\n"
            "Dùng helper trong `backend/ws/session.py`."
        )


def test_lookahead_khong_con_trung_trong_cung_file() -> None:
    """`min_words_to_commit` từng được xử lý y hệt ở 2 chỗ trong `lookahead_handler.py`."""
    src = (BACKEND / "ws" / "lookahead_handler.py").read_text(encoding="utf-8")
    assert src.count("self._batch_sub_opts[\"min_words\"] =") == 1, (
        "`min_words` phải chỉ được gán ở MỘT chỗ (`_set_min_words_to_commit`)"
    )


def test_apply_init_chi_ap_ngon_ngu_mot_lan() -> None:
    """`apply_init` từng gán source/target lang HAI lần và `set_language` trước giá trị cuối."""
    src = (BACKEND / "ws" / "lookahead_handler.py").read_text(encoding="utf-8")
    body = _block_until_next(src, "def apply_init", markers=_PY_METHOD_END)
    assert body.count("self.source_lang =") == 1, "`source_lang` phải được gán ĐÚNG một lần"
    assert body.count("self.target_lang =") == 1, "`target_lang` phải được gán ĐÚNG một lần"
    assert body.count("set_language(") == 1, "`set_language` phải chỉ gọi một lần, sau giá trị cuối"
    assert body.count("_parse_config(") == 1, "payload phải chỉ được parse một lần"


def test_apply_init_parse_truoc_khi_dung_components() -> None:
    """`init_components` đọc `self._parsed_config`, nên `apply_init` phải chạy TRƯỚC nó."""
    src = (BACKEND / "ws" / "lookahead_handler.py").read_text(encoding="utf-8")
    assert src.index("session.apply_init(data)") < src.index("session.init_components("), (
        "`apply_init` phải được gọi trước `init_components` — nếu không VAD sẽ dựng bằng "
        "thông số mặc định thay vì thông số popup"
    )


# ──────────────────────────────────────────────── §5.10 chốt mã nguồn


#: Marker "khai báo kế tiếp" cho từng ngôn ngữ. Phải xét CẢ mốc LÙI cấp, vì có hàm là thứ cuối
#: của class/module (ví dụ `apply_init`).
_PY_METHOD_END = ("\n    def ", "\n    async def ", "\n    @", "\ndef ", "\nasync def ", "\nclass ", "\n@")
_PY_TOPLEVEL_END = ("\ndef ", "\nasync def ", "\nclass ", "\n@")
_JS_FUNCTION_END = ("\n  function ", "\n  async function ", "\n})();")


def _block_until_next(src: str, header: str, *, markers: tuple[str, ...]) -> str:
    """Cắt đoạn từ `header` tới marker khai báo kế tiếp GẦN NHẤT.

    Cần thiết vì cùng một chuỗi có thể xuất hiện hợp lệ ở hàm khác — payload của
    `handleEngineSwitch` cũng gửi `split_on_stability`, và `_vad_runtime_info` cũng
    `import onnxruntime`.
    """
    start = src.index(header)
    after = start + len(header)
    positions = [p for marker in markers if (p := src.find(marker, after)) != -1]
    return src[start:] if not positions else src[start:min(positions)]


def test_main_dung_env_check_thay_vi_tu_kiem_tra() -> None:
    """`main.py` không được tự kiểm tra onnxruntime/aligner chỉ để LOG trạng thái khởi động."""
    src = (BACKEND / "main.py").read_text(encoding="utf-8")
    body = _block_until_next(src, "def _log_runtime_status_at_startup", markers=_PY_TOPLEVEL_END)
    assert "env_check.runtime_status()" in body, (
        "`_log_runtime_status_at_startup` phải đọc từ `env_check.runtime_status()`"
    )
    assert "import onnxruntime" not in body, (
        "hàm log khởi động còn tự import onnxruntime ⇒ đang kiểm tra lặp với env_check"
    )
    assert "_fa_available" not in body, (
        "hàm log khởi động còn tự gọi aligner ⇒ đang kiểm tra lặp với env_check"
    )


def test_popup_khong_ghi_alias_snake_case_thua() -> None:
    """`getSettings()` chỉ nên có MỘT tên cho mỗi giá trị.

    Chỉ soi THÂN `getSettings`: payload của `handleEngineSwitch` hợp lệ khi gửi snake_case
    thẳng cho backend, nên không được tính là vi phạm.
    """
    src = (ROOT / "extension_src" / "popup" / "popup.js").read_text(encoding="utf-8")
    body = _block_until_next(src, "function getSettings()", markers=_JS_FUNCTION_END)
    for stale in (
        "silence_duration_ms:",
        "vad_threshold:",
        "min_words_to_commit:",
        "split_on_stability:",
        "trace_stability:",
        "stability_duration_ms:",
        "stability_min_duration_sec:",
        "stability_min_words:",
    ):
        assert stale not in body, (
            f"`getSettings()` còn ghi khoá alias snake_case `{stale}` — chỉ giữ tên camelCase, "
            "các phía đọc đã tự fallback (`buildWsConfig`, các chỗ đọc `bs_settings`)."
        )
    assert "stabilityDurationMs:" in body, (
        "`getSettings()` phải có `stabilityDurationMs` (bản ms) để payload REST khỏi nhân lại "
        "`stabilityDurationSec * 1000` (sai số dấu phẩy động)."
    )


def test_payload_rest_cua_popup_doc_nhat_quan_camel_case() -> None:
    """Payload REST gửi snake_case nhưng phải ĐỌC từ khoá camelCase — không trộn hai quy ước."""
    src = (ROOT / "extension_src" / "popup" / "popup.js").read_text(encoding="utf-8")
    for expected in (
        "silence_duration_ms: cfg.vadSilenceDurationMs,",
        "stability_duration_ms: cfg.stabilityDurationMs,",
        "stability_min_duration_sec: cfg.stabilityMinDurationSec,",
        "stability_min_words: cfg.stabilityMinWords,",
    ):
        assert expected in src, f"payload REST phải đọc bằng camelCase: thiếu `{expected}`"
