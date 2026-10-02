"""test_59 — Chốt cấu hình "Pipeline B chỉ có OFFLINE_BATCH" và hành vi popup/fallback.

Bối cảnh: Pipeline B v2 ("streaming giả lập": VAD + CommitManager + preview) đã bị XOÁ ngày
2026-10-02 theo yêu cầu người dùng. Bộ test này khoá lại các bất biến của quyết định đó để không
ai vô tình đưa v2 trở lại:

1. `LookaheadConfig` KHÔNG còn `processing_mode`/các knob chỉ dùng cho v2 (preview, inactivity,
   feed_block, min_prebuffer, decode_starve).
2. `LookaheadSessionState` chỉ còn vòng xử lý khối (`_offline_batch_loop`) — không còn `_ingest_loop`
   / `_asr_loop` / `_handle_asr_message` / `_pts_at` / `_on_speech_chunk`.
3. Extension: mặc định BẬT Lookahead, và giữ đường FALLBACK sang Pipeline A khi Lookahead không khả
   dụng hoặc khi người dùng tắt công tắc.
4. Popup: khi Lookahead bật thì nhóm tuỳ chỉnh chỉ dành cho Pipeline A bị VÔ HIỆU (có hàm + id +
   ghi chú trong HTML).
5. Hai bản extension (Firefox và Chrome/Edge) phải đồng bộ ở các file vốn giống nhau.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import LookaheadConfig
from backend.ws.lookahead_handler import LookaheadSessionState

ROOT = Path(__file__).resolve().parents[2]
EXT_DIRS = ("extension_firefox", "extension_chrome_edge")

#: Các khoá cấu hình chỉ phục vụ Pipeline B v2 (đã xoá).
REMOVED_LOOKAHEAD_KEYS = (
    "processing_mode",
    "preview_enabled",
    "preview_min_new_audio_sec",
    "inactivity_timeout_sec",
    "feed_block_sec",
    "decode_starve_warn_sec",
    "min_prebuffer_sec",
)

#: Các phương thức chỉ phục vụ Pipeline B v2 (đã xoá).
REMOVED_SESSION_METHODS = (
    "_ingest_loop",
    "ingest_once",
    "_feed_block",
    "_next_feed_pts",
    "_feed_target_pts",
    "_asr_loop",
    "_handle_asr_message",
    "_pts_at",
    "_record_anchor",
    "_prune_anchors",
    "_on_speech_chunk",
    "_request_replay",
    "_apply_lookahead_overrides",
)

#: id các nhóm tuỳ chỉnh CHỈ dành cho Pipeline A (popup phải vô hiệu khi Lookahead bật).
PIPELINE_A_ONLY_IDS = (
    "groupVadEngine",
    "rowVadSliders",
    "groupStableMinSec",
    "groupMinWords",
    "sectionSegmentation",
    "groupStableCut",
    "groupStableMs",
    "groupStableMinWords",
    "groupStableTrace",
)


def test_lookahead_config_has_no_streaming_v2_knobs():
    """`LookaheadConfig` chỉ còn cấu hình cho tuyến OFFLINE_BATCH."""
    fields = set(LookaheadConfig.model_fields)
    for key in REMOVED_LOOKAHEAD_KEYS:
        assert key not in fields, f"khoá '{key}' của Pipeline B v2 vẫn còn trong LookaheadConfig"
    # Các khoá của tuyến batch phải còn nguyên
    for key in ("enabled", "batch_target_sec", "batch_search_max_sec", "batch_sub_max_words"):
        assert key in fields


def test_lookahead_session_has_no_streaming_v2_methods():
    """Lớp phiên Lookahead không còn API của tuyến streaming v2."""
    for name in REMOVED_SESSION_METHODS:
        assert not hasattr(LookaheadSessionState, name), f"'{name}' của Pipeline B v2 vẫn còn"
    # Tuyến duy nhất còn lại
    assert hasattr(LookaheadSessionState, "_offline_batch_loop")


def test_start_tasks_only_starts_batch_loop():
    """`start_tasks()` chỉ được tạo task của tuyến batch (không còn ingest/asr streaming)."""
    src = (
        ROOT / "backend" / "ws" / "lookahead_handler.py"
    ).read_text(encoding="utf-8")
    assert "la_batch_" in src and "la_status_" in src
    assert "la_ingest_" not in src, "task 'la_ingest_' của Pipeline B v2 vẫn còn"
    assert "la_asr_" not in src, "task 'la_asr_' của Pipeline B v2 vẫn còn"
    assert "processing_mode" not in src, "tham chiếu 'processing_mode' vẫn còn trong handler"


@pytest.mark.parametrize("ext", EXT_DIRS)
def test_extension_defaults_to_pipeline_b_with_pipeline_a_fallback(ext: str):
    """Mặc định BẬT Lookahead, và giữ nguyên đường fallback sang Pipeline A."""
    html = (ROOT / ext / "popup" / "popup.html").read_text(encoding="utf-8")
    # Công tắc Lookahead mặc định BẬT (checked) ⇒ cài mới chạy Pipeline B.
    assert 'id="chkEnableLookahead" checked' in html

    content = (ROOT / ext / "content" / "content-script.js").read_text(encoding="utf-8")
    # Không bị tắt tường minh ⇒ coi như BẬT (giá trị chưa lưu = mặc định).
    assert "settings.lookaheadEnabled !== false" in content
    # Lookahead không khả dụng ⇒ trả `fallbackToA` để caller chạy Pipeline A.
    assert "fallbackToA: true" in content
    assert "startPipelineA(" in content


@pytest.mark.parametrize("ext", EXT_DIRS)
def test_popup_disables_pipeline_a_only_controls(ext: str):
    """Popup phải có đủ id nhóm tuỳ chỉnh Pipeline A + hàm vô hiệu/mở lại chúng."""
    html = (ROOT / ext / "popup" / "popup.html").read_text(encoding="utf-8")
    for dom_id in PIPELINE_A_ONLY_IDS:
        assert f'id="{dom_id}"' in html, f"{ext}: thiếu id '{dom_id}' trong popup.html"
    for note_id in ("pipelineAOnlyNote", "pipelineAOnlyNote2"):
        assert f'id="{note_id}"' in html, f"{ext}: thiếu ghi chú '{note_id}'"

    js = (ROOT / ext / "popup" / "popup.js").read_text(encoding="utf-8")
    assert "function updatePipelineAOnlyUi()" in js
    # Phải được gọi lại ở cả 3 nhịp: cập nhật nhãn, đổi trạng thái session, và bấm công tắc.
    assert js.count("updatePipelineAOnlyUi();") >= 3
    for dom_id in PIPELINE_A_ONLY_IDS:
        assert f'"{dom_id}"' in js, f"{ext}: popup.js không tham chiếu '{dom_id}'"
    # Tuỳ chỉnh bị vô hiệu thật (disabled), không chỉ làm mờ.
    assert ".disabled = !editable" in js


@pytest.mark.parametrize("ext", EXT_DIRS)
def test_popup_has_no_prebuffer_ahead_slider(ext: str):
    """Popup KHÔNG còn "Thời gian dịch trước" (lead time).

    Ở tuyến OFFLINE_BATCH, khoảng dịch trước do BỘ CẮT KHỐI quyết định (12–30 s), nên slider
    10–15 s chỉ còn là cổng throttle nội bộ ⇒ gây hiểu nhầm là người dùng điều khiển được độ xa
    dịch trước. Backend dùng mặc định `lookahead.lead_time_sec`.
    """
    html = (ROOT / ext / "popup" / "popup.html").read_text(encoding="utf-8")
    js = (ROOT / ext / "popup" / "popup.js").read_text(encoding="utf-8")
    content = (ROOT / ext / "content" / "content-script.js").read_text(encoding="utf-8")
    for needle in ("lookaheadLeadTimeGroup", "rangeLookaheadLeadTime", "valLookaheadLeadTime"):
        assert needle not in html, f"{ext}: popup.html vẫn còn '{needle}'"
        assert needle not in js, f"{ext}: popup.js vẫn còn '{needle}'"
    assert "cfg.lookaheadLeadTimeSec" not in content, f"{ext}: vẫn còn gửi lookaheadLeadTimeSec"
    # "Đồng bộ phụ đề / lồng tiếng" thì VẪN còn: nó dịch mốc phụ đề + TTS theo từng trang.
    assert "lookaheadSyncOffsetMs" in content and "rangeLookaheadSync" in js


def test_extension_mirrors_stay_in_sync():
    """Hai bản extension phải giống nhau ở các file không có khác biệt cố ý (popup + timeline)."""
    for rel in ("popup/popup.js", "lib/lookahead-timeline.js"):
        a = (ROOT / EXT_DIRS[0] / rel).read_text(encoding="utf-8")
        b = (ROOT / EXT_DIRS[1] / rel).read_text(encoding="utf-8")
        assert a == b, f"{rel} lệch nhau giữa {EXT_DIRS[0]} và {EXT_DIRS[1]}"


@pytest.mark.asyncio
async def test_lookahead_session_apply_config_runtime():
    """`apply_config()` từ popup khi đổi source_lang / target_lang / sync_offset không được văng lỗi."""
    from unittest.mock import MagicMock

    conn = MagicMock()
    conn.is_closed = False
    session = LookaheadSessionState(ws=conn, asr_engine=MagicMock(), translation_engine=MagicMock())

    # Giả lập payload đổi SourceLang từ Extension Popup
    payload = {
        "type": "set_config",
        "source_lang": "ja",
        "target_lang": "vi",
        "lookahead_sync_offset_ms": 150,
    }
    applied = await session.apply_config(payload)
    assert session.source_lang == "ja"
    assert session.target_lang == "vi"
    assert session.sync_offset_ms == 150.0
    assert applied.get("source_lang") == "ja"

