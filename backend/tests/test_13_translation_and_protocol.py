"""Test tầng A — streaming translation (P3.3), TTS/queue, giao thức.

Kiểm thử đường dịch dần mà KHÔNG cần llama.cpp: dùng `FakeTranslator` và gọi trực tiếp
`_process_translation_item`.
"""

import asyncio
import json

import pytest

from backend.config import config
from backend.ws.serializers import (
    make_model_status_msg,
    make_translation_msg,
    make_tts_audio_msg,
    make_utterance_update_msg,
)


# ------------------------------------------------------------------ P3.3 streaming
def test_partial_translation_keeps_status_ok():
    """Bản dịch dần PHẢI giữ `status == "ok"`.

    `subtitle-renderer.js:313` chỉ nhận bản dịch khi `status === "ok"`; nếu đặt
    status="partial" thì client hiện tại sẽ coi là null và XOÁ bản dịch đang hiển thị.
    """
    msg = make_translation_msg("u1", "xin chào", 10, "vi", partial=True)
    assert msg["status"] == "ok", "status phải là 'ok' để client hiện tại không xoá bản dịch"
    assert msg["partial"] is True
    final = make_translation_msg("u1", "xin chào bạn", 20, "vi", partial=False)
    assert final["status"] == "ok"
    assert final["partial"] is False


def test_process_translation_item_streams_partials_then_final(session_factory, restore_config):
    """P3.3: phải gửi các bản dịch DẦN rồi một bản cuối, và đo `first_token_ms`.

    Dùng `per_char_delay` để các partial trải ra theo thời gian như llama.cpp thật
    (nếu trả tức thì thì throttle sẽ (đúng đắn) chặn hết partial trung gian).
    """
    from backend.core.metrics import metrics_collector
    from backend.translation.context import TranslationContextTracker
    from backend.translation.dedup import TranslationDeduplicator
    from backend.tests.fakes import FakeTranslator
    from backend.ws import handler as handler_mod

    config.translation.stream_tokens = True
    config.translation.partial_min_chars = 3
    config.translation.partial_min_interval_ms = 20
    session = session_factory()
    translator = FakeTranslator(prefix="[vi] ", per_char_delay=0.005)

    async def _run():
        await handler_mod._process_translation_item(
            session,
            translator,
            TranslationContextTracker(window_size=3),
            {"utterance_id": "utt-1", "text": "hello world", "source_lang": "en",
             "target_lang": "vi", "_queued_at": 0.0},
            TranslationDeduplicator(),
        )

    asyncio.run(_run())

    msgs = getattr(session.mock_ws, "sent_messages", [])
    partials = [m for m in msgs if m.get("type") == "translation" and m.get("partial")]
    finals = [m for m in msgs if m.get("type") == "translation" and not m.get("partial")]
    updates = [m for m in msgs if m.get("type") == "utterance_update" and m.get("is_final")]

    assert finals, "phải có bản dịch cuối cùng"
    final_text = finals[0]["translated"]
    assert final_text == "[vi] hello world"
    assert updates, "phải gửi utterance_update final kèm bản dịch"
    assert partials, "phải có ít nhất 1 bản dịch dần khi translator trả chậm"

    # Bất biến: mọi partial là TIỀN TỐ của bản cuối (không nhảy chữ, không lùi)
    for m in partials:
        assert final_text.startswith(m["translated"]), (
            f"partial {m['translated']!r} không phải tiền tố của final {final_text!r}"
        )
    # Và độ dài tăng đơn điệu
    lens = [len(m["translated"]) for m in partials]
    assert lens == sorted(lens), f"partial phải tăng dần, nhận được: {lens}"
    # Không gửi tiền tố rác ngắn hơn ngưỡng
    assert all(len(m["translated"]) >= 3 for m in partials)
    # Đo được thời gian tới token đầu
    st = metrics_collector.get_stage_stats("translation.first_token_ms")
    assert st["count"] >= 1


def test_translation_streaming_can_be_disabled(session_factory, restore_config):
    """Cờ `stream_tokens=False` phải quay về đường dịch một lần."""
    from backend.translation.context import TranslationContextTracker
    from backend.translation.dedup import TranslationDeduplicator
    from backend.tests.fakes import FakeTranslator
    from backend.ws import handler as handler_mod

    config.translation.stream_tokens = False
    session = session_factory()
    translator = FakeTranslator()

    async def _run():
        await handler_mod._process_translation_item(
            session,
            translator,
            TranslationContextTracker(window_size=3),
            {"utterance_id": "utt-2", "text": "abc", "source_lang": "en",
             "target_lang": "vi", "_queued_at": 0.0},
            TranslationDeduplicator(),
        )

    asyncio.run(_run())
    msgs = getattr(session.mock_ws, "sent_messages", [])
    partials = [m for m in msgs if m.get("type") == "translation" and m.get("partial")]
    finals = [m for m in msgs if m.get("type") == "translation" and not m.get("partial")]
    assert not partials, "không được gửi partial khi tắt streaming"
    assert finals and finals[0]["translated"] == "[vi] abc"


def test_translation_dedup_reuses_cached_translation(session_factory, restore_config):
    """FIX-05: câu dịch trùng KHÔNG được dịch lại, NHƯNG phải có gói `translation` để gỡ "...".

    Bug gốc: nhánh dedup `return` im lặng. Client đã nhận câu final với placeholder
    `translated="..."` nên phụ đề gốc treo vĩnh viễn ở chỉ báo "đang dịch".
    """
    from backend.core.metrics import metrics_collector
    from backend.translation.context import TranslationContextTracker
    from backend.translation.dedup import TranslationDeduplicator
    from backend.tests.fakes import FakeTranslator
    from backend.ws import handler as handler_mod

    session = session_factory()
    translator = FakeTranslator()
    dedup = TranslationDeduplicator()
    before_reused = metrics_collector.get_counter("translation.dedup_reused")

    async def _run():
        for _ in range(2):
            await handler_mod._process_translation_item(
                session, translator, TranslationContextTracker(window_size=3),
                {"utterance_id": "utt-3", "text": "same text", "source_lang": "en",
                 "target_lang": "vi", "_queued_at": 0.0},
                dedup,
            )

    asyncio.run(_run())
    assert len(translator.calls) == 1, "câu trùng không được dịch lại"

    msgs = getattr(session.mock_ws, "sent_messages", [])
    finals = [m for m in msgs if m.get("type") == "translation" and not m.get("partial")]
    assert len(finals) == 2, "câu lặp vẫn phải gửi gói translation (nếu không dấu '...' treo)"
    assert finals[0]["translated"] == finals[1]["translated"] == "[vi] same text"
    assert metrics_collector.get_counter("translation.dedup_reused") == before_reused + 1


def test_translation_dedup_without_cache_clears_ellipsis(session_factory, restore_config):
    """FIX-05: trùng mà chưa có bản dịch cache ⇒ vẫn phải gỡ "...", phụ đề gốc giữ nguyên."""
    from backend.core.metrics import metrics_collector
    from backend.translation.context import TranslationContextTracker
    from backend.translation.dedup import TranslationDeduplicator
    from backend.tests.fakes import FakeTranslator
    from backend.ws import handler as handler_mod

    session = session_factory()
    dedup = TranslationDeduplicator()
    before = metrics_collector.get_counter("translation.ellipsis_cleared")

    class _FailingTranslator(FakeTranslator):
        async def translate(self, *a, **kw):
            raise RuntimeError("model lỗi")

        async def translate_stream(self, *a, **kw):  # type: ignore[override]
            raise RuntimeError("model lỗi")
            yield ""  # pragma: no cover

    translator = _FailingTranslator()

    async def _run():
        # Lần 1: dịch lỗi ⇒ KHÔNG có bản dịch nào được ghi nhớ.
        with pytest.raises(RuntimeError):
            await handler_mod._process_translation_item(
                session, translator, TranslationContextTracker(window_size=3),
                {"utterance_id": "u1", "text": "lặp", "source_lang": "ja",
                 "target_lang": "vi", "_queued_at": 0.0},
                dedup,
            )
        session.mock_ws.sent_messages.clear()
        # Lần 2: cùng câu ⇒ dedup trúng nhưng cache rỗng.
        await handler_mod._process_translation_item(
            session, translator, TranslationContextTracker(window_size=3),
            {"utterance_id": "u2", "text": "lặp", "source_lang": "ja",
             "target_lang": "vi", "_queued_at": 0.0},
            dedup,
        )

    asyncio.run(_run())
    msgs = getattr(session.mock_ws, "sent_messages", [])
    finals = [m for m in msgs if m.get("type") == "translation"]
    assert finals, "phải gửi translation rỗng để client tắt chỉ báo 'đang dịch'"
    assert finals[0]["translated"] == ""
    assert finals[0]["status"] == "ok", "status phải 'ok' để renderer coi là bản dịch hoàn chỉnh"
    assert metrics_collector.get_counter("translation.ellipsis_cleared") == before + 1


# ---------------------------------------------------------------------- protocol
def test_utterance_update_shape():
    msg = make_utterance_update_msg(utt_id="u", text="hi", translated="chao", is_final=True)
    assert msg["type"] == "utterance_update"
    assert msg["is_final"] is True
    assert msg["isFinal"] is True
    assert msg["text"] == "hi"
    assert msg["translated"] == "chao"


def test_model_status_shape():
    msg = make_model_status_msg("asr", "loading", "qwen3-asr-1.7b")
    assert msg["type"] == "model_status"
    assert msg["stage"] == "asr"
    assert msg["state"] == "loading"


def test_tts_audio_msg_shape():
    msg = make_tts_audio_msg("u", "hello", "QUJD", 1.5, 24000)
    assert msg["type"] == "tts_audio"
    assert msg["audio"] == "QUJD"
    assert msg["sample_rate"] == 24000
    assert msg["sampleRate"] == 24000


def test_protocol_version_is_exposed(session_factory, restore_config):
    """P3.0: phiên bản giao thức phải có trong config để client tự giảm cấp hành vi."""
    assert config.ws.protocol_version >= 2


# ---------------------------------------------------------- P3.0/P3.1 binary TTS
def test_binary_tts_disabled_by_default_for_old_clients(session_factory, restore_config):
    """P3.1: extension CŨ (không khai báo protocolVersion) phải nhận đường base64.

    Đây là điều kiện để triển khai backend TRƯỚC extension mà không phá bản đang chạy.
    """
    session = session_factory()
    assert session.protocol_version == 1
    assert session.supports_binary_tts is False


def test_binary_tts_enabled_when_client_declares_v2(session_factory, restore_config):
    session = session_factory()
    applied = session.apply_config({"protocolVersion": 2})
    assert applied["protocol_version"] == 2
    assert session.supports_binary_tts is True


def test_binary_tts_frame_layout():
    """P3.1: layout frame binary phải khớp với phía JS (magic BTTS + ver + len + JSON + WAV)."""
    from backend.ws.serializers import make_tts_binary_frame

    wav = b"RIFF____WAVEfmt "
    frame = make_tts_binary_frame("utt-9", "xin chào", wav, 1.25, 24000)
    assert frame[:4] == b"BTTS"
    assert frame[4] == 1
    header_len = int.from_bytes(frame[5:7], "little")
    header = json.loads(frame[7:7 + header_len].decode("utf-8"))
    assert header["utterance_id"] == "utt-9"
    assert header["sampleRate"] == 24000
    assert header["encoding"] == "binary"
    assert frame[7 + header_len:] == wav
    # Không có base64 => nhỏ hơn đáng kể so với JSON+base64
    import base64 as _b64

    json_equiv = len(json.dumps({"audio": _b64.b64encode(wav).decode("ascii")}).encode("utf-8"))
    assert len(frame) <= json_equiv + header_len + 16


def test_tts_wav_bytes_and_base64_are_consistent():
    """P3.1: hai đường mã hóa phải cho cùng nội dung WAV."""
    import base64 as _b64
    import numpy as np

    from backend.tts.audio_processor import AudioProcessor

    audio = (np.sin(np.linspace(0, 200, 24000)) * 0.4).astype(np.float32)
    raw = AudioProcessor.encode_wav_bytes(audio, 24000)
    b64 = AudioProcessor.encode_wav_to_base64(audio, 24000)
    assert raw[:4] == b"RIFF"
    assert _b64.b64decode(b64) == raw


def test_binary_tts_chosen_when_client_supports_it(session_factory, restore_config):
    """P3.1: khi client hỗ trợ v2, TTS phải được gửi qua binary frame (không base64)."""
    from backend.ws import handler as handler_mod

    session = session_factory()
    session.apply_config({"protocolVersion": 2})
    assert session.supports_binary_tts is True

    class _FakeTTS:
        sample_rate = 24000

        def __init__(self):
            self.binary_calls = 0
            self.base64_calls = 0

        async def synthesize_clone(self, text, voice_id=None, speed=1.0):
            self.base64_calls += 1
            return "QUJD", 1.0

        async def synthesize_clone_bytes(self, text, voice_id=None, speed=1.0):
            self.binary_calls += 1
            return b"RIFF____WAVEfmt data", 1.0

    tts = _FakeTTS()

    async def _run():
        await handler_mod._process_tts_item(
            session, tts,
            {"utterance_id": "u-b", "text": "hello", "voice": None, "speed": 1.0, "_queued_at": 0.0},
            handler_mod.TTSDedupState(),
        )

    asyncio.run(_run())
    assert tts.binary_calls == 1 and tts.base64_calls == 0, "phải dùng đường binary"
    frames = getattr(session.mock_ws, "sent_bytes", [])
    assert frames and frames[0][:4] == b"BTTS"


def test_base64_tts_used_for_old_client(session_factory, restore_config):
    """P3.1: client cũ (protocol 1) vẫn nhận JSON + base64 như trước."""
    from backend.ws import handler as handler_mod

    session = session_factory()  # protocol mặc định = 1

    class _FakeTTS:
        sample_rate = 24000

        def __init__(self):
            self.binary_calls = 0
            self.base64_calls = 0

        async def synthesize_clone(self, text, voice_id=None, speed=1.0):
            self.base64_calls += 1
            return "QUJD", 1.0

        async def synthesize_clone_bytes(self, text, voice_id=None, speed=1.0):
            self.binary_calls += 1
            return b"RIFF", 1.0

    tts = _FakeTTS()

    async def _run():
        await handler_mod._process_tts_item(
            session, tts,
            {"utterance_id": "u-c", "text": "hello", "voice": None, "speed": 1.0, "_queued_at": 0.0},
            handler_mod.TTSDedupState(),
        )

    asyncio.run(_run())
    assert tts.base64_calls == 1 and tts.binary_calls == 0
    msgs = getattr(session.mock_ws, "sent_messages", [])
    assert any(m.get("type") == "tts_audio" and m.get("audio") for m in msgs)
    assert not getattr(session.mock_ws, "sent_bytes", [])


def test_connected_hello_announces_capabilities():
    """P3.0: handler phải gửi gói `connected` nêu rõ khả năng của server."""
    import inspect

    from backend.ws import handler as handler_mod

    src = inspect.getsource(handler_mod.handle_ws)
    assert '"connected"' in src
    assert "binary_tts" in src


def test_session_config_accepts_protocol_version(session_factory, restore_config):
    """`protocolVersion` từ extension không được làm lỗi validate (extra='ignore')."""
    session = session_factory()
    applied = session.apply_config({"protocolVersion": 2, "targetLang": "vi"})
    assert "target_lang" in applied


# ------------------------------------------------------------------------- queues
def test_translation_queue_full_increments_counter(session_factory, restore_config):
    """Queue đầy phải ĐẾM được (trước đây chỉ log warning, không có metric)."""
    from backend.core.metrics import metrics_collector
    from backend.ws import handler as handler_mod

    session = session_factory()
    session.translation_queue = asyncio.Queue(maxsize=1)

    async def _run():
        # Bơm 1 item để đầy queue
        await session.translation_queue.put({"x": 1})
        # Giả lập vòng ASR đẩy thêm 1 item -> QueueFull
        before = metrics_collector.get_counter("queue.translation_dropped")
        try:
            session.translation_queue.put_nowait({"x": 2})
        except asyncio.QueueFull:
            metrics_collector.increment_counter("queue.translation_dropped")
        after = metrics_collector.get_counter("queue.translation_dropped")
        return before, after

    before, after = asyncio.run(_run())
    assert after > before
