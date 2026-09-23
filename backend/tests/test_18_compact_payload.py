"""Test tầng A — F-30: payload gọn (protocol v3), không gửi field trùng lặp.

Bối cảnh: mỗi thông điệp trước đây gửi 2-3 tên cho CÙNG một giá trị
(`utterance_id`+`utteranceId`, `original`+`ui_text`+`text`, `stable_text`+`stableText`,
`duration_sec`+`durationSec`…). Với câu dài, payload phình 2-3×.

Quy tắc mới:
- backend `protocol_version = 3`;
- client khai báo `protocol_version >= 3` ⇒ nhận payload GỌN (chỉ snake_case);
- client cũ (v1/v2) ⇒ vẫn nhận payload đầy đủ (không phá vỡ tương thích).
"""

import asyncio
import json
from pathlib import Path

from backend.config import config
from backend.ws import serializers as sz

ROOT = Path(__file__).resolve().parent.parent.parent
EXT = ROOT / "extension_firefox"


# --------------------------------------------------------------- serializers gọn
def test_utterance_compact_has_no_duplicate_fields():
    """v3: chỉ còn MỘT tên cho mỗi giá trị (không `original`/`ui_text`/`utteranceId`)."""
    msg = sz.make_utterance_update_msg(
        utt_id="u1", text="xin chào", stable_text="xin", unstable_text="chào",
        is_final=True, compact=True,
    )
    assert msg["utterance_id"] == "u1" and msg["text"] == "xin chào"
    assert msg["stable_text"] == "xin" and msg["unstable_text"] == "chào"
    assert msg["is_final"] is True
    for dup in ("utteranceId", "original", "ui_text", "stableText", "unstableText", "isFinal"):
        assert dup not in msg, f"payload gọn còn field trùng lặp '{dup}'"
    assert len(msg) == 8


def test_utterance_fat_keeps_backward_compatible_aliases():
    """Client cũ vẫn nhận đủ alias => không phá vỡ tương thích."""
    msg = sz.make_utterance_update_msg(utt_id="u1", text="xin chào", is_final=True)
    for alias in ("utteranceId", "original", "ui_text", "isFinal"):
        assert alias in msg, f"payload đầy đủ thiếu alias '{alias}'"


def test_translation_compact_and_fat():
    compact = sz.make_translation_msg("u1", "xin chào", 42, "vi", partial=True, compact=True)
    assert compact["sentence_id"] == "u1" and compact["translated"] == "xin chào"
    assert compact["partial"] is True and compact["status"] == "ok"
    for dup in ("sentenceId", "utterance_id", "utteranceId", "text", "translateTimeMs", "targetLang"):
        assert dup not in compact, f"payload gọn còn field trùng lặp '{dup}'"

    fat = sz.make_translation_msg("u1", "xin chào", 42, "vi")
    for alias in ("sentenceId", "utterance_id", "utteranceId", "text", "targetLang"):
        assert alias in fat


def test_pong_compact_drops_camel_case():
    compact = sz.make_pong_msg(123, compact=True)
    assert compact["timestamp"] == 123 and "server_time" in compact
    assert "serverTime" not in compact
    assert "serverTime" in sz.make_pong_msg(123)


def test_tts_binary_header_compact(tmp_path=None):
    """Khung nhị phân TTS: header v3 không còn alias, magic/độ dài vẫn đúng."""
    frame = sz.make_tts_binary_frame("u1", "xin chào", b"RIFFxxxx", 1.5, 24000, compact=True)
    assert frame[:4] == b"BTTS"
    ver = frame[4]
    header_len = int.from_bytes(frame[5:7], "little")
    header = json.loads(frame[7:7 + header_len].decode("utf-8"))
    assert ver == 1
    assert header["utterance_id"] == "u1" and header["duration_sec"] == 1.5
    for dup in ("utteranceId", "sentence_id", "sentenceId", "durationSec", "sampleRate"):
        assert dup not in header, f"header gọn còn field trùng lặp '{dup}'"
    assert frame[7 + header_len:] == b"RIFFxxxx"


def test_compact_payload_is_materially_smaller():
    """Đúng mục tiêu F-30: payload gọn phải nhỏ hơn hẳn (đo bằng bytes JSON)."""
    text = "Đây là một câu dịch khá dài để đo kích thước payload thật sự." * 3
    fat = sz.make_utterance_update_msg("utt-12345678", text, "bản dịch", True, "ổn định", "đang chạy")
    compact = sz.make_utterance_update_msg(
        "utt-12345678", text, "bản dịch", True, "ổn định", "đang chạy", compact=True
    )
    fat_bytes = len(json.dumps(fat, ensure_ascii=False).encode("utf-8"))
    compact_bytes = len(json.dumps(compact, ensure_ascii=False).encode("utf-8"))
    assert compact_bytes < fat_bytes * 0.65, (
        f"payload gọn chưa nhỏ hơn đáng kể: {compact_bytes} vs {fat_bytes} bytes"
    )


# --------------------------------------------------------------- session/negotation
def test_session_only_compact_for_protocol_v3():
    from backend.tests.conftest import MockWebSocket
    from backend.ws.connection import SafeWebSocketConnection
    from backend.ws.session import SessionState

    session = SessionState(SafeWebSocketConnection(MockWebSocket()))
    assert session.protocol_version == 1
    assert session.supports_compact_payload is False, "client v1 phải nhận payload đầy đủ"

    session.protocol_version = 2
    assert session.supports_compact_payload is False

    session.protocol_version = 3
    assert session.supports_compact_payload is True


def test_backend_protocol_version_is_3():
    assert int(config.ws.protocol_version) >= 3


# --------------------------------------------------------------- wiring trong handler
def test_handler_sends_compact_for_v3_client(session_factory, restore_config, monkeypatch):
    """Chạy THẬT `_stream_asr_tokens` với client v3 và soi thông điệp đã gửi."""
    from backend.tests.fakes import FakeInferenceEngine
    from backend.ws.handler import _stream_asr_tokens

    session = session_factory(text_fn=lambda n: "alpha beta")
    session.protocol_version = 3

    canned = [
        {
            "type": "utterance_update",
            "utterance_id": "u1",
            "text": "xin chào",
            "is_final": False,
            "stable_text": "xin",
            "unstable_text": "chào",
            "language": "vi",
            "inference_ms": 1.0,
        }
    ]

    async def _fake_stream():
        for m in canned:
            yield m

    monkeypatch.setattr(session.asr_engine, "stream_tokens", _fake_stream, raising=False)
    asyncio.run(_stream_asr_tokens(session))

    sent = session.mock_ws.sent_messages
    assert sent, "không gửi thông điệp nào"
    msg = sent[0]
    assert msg["utterance_id"] == "u1" and msg["text"] == "xin chào"
    for dup in ("utteranceId", "original", "ui_text", "stableText", "isFinal"):
        assert dup not in msg, f"client v3 vẫn nhận field trùng lặp '{dup}'"


def test_handler_sends_fat_for_legacy_client(session_factory, restore_config, monkeypatch):
    """Client v2 (extension cũ) vẫn phải nhận payload đầy đủ như trước."""
    from backend.ws.handler import _stream_asr_tokens

    session = session_factory(text_fn=lambda n: "alpha beta")
    session.protocol_version = 2

    async def _fake_stream():
        yield {
            "type": "utterance_update",
            "utterance_id": "u1",
            "text": "xin chào",
            "is_final": False,
            "language": "vi",
            "inference_ms": 1.0,
        }

    monkeypatch.setattr(session.asr_engine, "stream_tokens", _fake_stream, raising=False)
    asyncio.run(_stream_asr_tokens(session))

    msg = session.mock_ws.sent_messages[0]
    assert "ui_text" in msg and "utteranceId" in msg, "client cũ phải giữ payload đầy đủ"


# --------------------------------------------------------------- extension (v3)
def test_extension_declares_protocol_v3():
    """Extension hiện hành khai báo giao thức v3 (payload gọn)."""
    src = (EXT / "content" / "content-script.js").read_text(encoding="utf-8")
    assert "const PROTOCOL_VERSION = 3;" in src, "extension phải khai báo giao thức v3"


def test_extension_reads_only_canonical_fields():
    """Không được fallback sang alias (nếu còn thì F-30 chưa sạch)."""
    files = [
        EXT / "lib" / "subtitle-renderer.js",
        EXT / "lib" / "tts-player.js",
        EXT / "content" / "content-script.js",
    ]
    forbidden = [
        "payload.ui_text",
        "payload.original",
        "payload.utteranceId",
        "payload.sentenceId",
        "payload.stableText",
        "payload.isFinal",
        "payload.durationSec",
        "payload.audio_base64",
        "header.utteranceId",
    ]
    for path in files:
        src = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in src, f"{path.name} còn đọc alias '{token}'"


def test_extension_manifest_is_production_one():
    """Sau khi thay thế, manifest phải là bản chính thức (không còn nhãn TEST/id test)."""
    import json as _json

    manifest = _json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "Bilingual Subtitle"
    assert manifest["browser_specific_settings"]["gecko"]["id"] == "bilingual-subtitle@local"
    assert int(manifest["version"].split(".")[0]) >= 0


# ------------------------------- P1.1: không hướng dẫn cài gói CUDA không tồn tại
def test_backend_log_never_suggests_bogus_package(caplog):
    """Log backend ASR KHÔNG được hướng dẫn `pip install transcribe-cpp-native-cu12`.

    Gói đó trên PyPI chỉ có `0.0.0` — name reservation, wheel 1380 byte không có native
    code. CUDA cho ASR nay có được là nhờ bundle **có sẵn trong `backend/bin/`** (xem
    `report/audit/KE_HOACH_FIX_LOI_Hy3.md` §4.1.2), không phải nhờ cài gói kia.
    """
    import logging

    import backend.main as main_mod

    src = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")
    assert "pip install transcribe-cpp-native-cu12" not in src, (
        "không được hướng dẫn cài gói CUDA không tồn tại (PyPI chỉ có 0.0.0)"
    )
    assert "pip uninstall transcribe-cpp-native" not in src

    with caplog.at_level(logging.INFO):
        main_mod._log_asr_backend_at_startup()
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "transcribe-cpp-native-cu12" not in joined
    assert "pip install" not in joined
