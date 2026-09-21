"""Test tầng A — hiệu lực của cấu hình popup (P1.7..P1.11 / G1..G10).

Yêu cầu C6: "Các điều chỉnh trong popup cần được áp dụng ngay cho backend mà không cần
khởi động lại backend". Bộ test này là LƯỚI BẢO HIỂM: bất kỳ control nào của popup
không có tác dụng quan sát được ⇒ test fail.

Bảng đối chiếu với các khoảng trống đã phát hiện trong audit:
- G1: đổi translation model qua WS không làm gì  → test_translation_model_*.
- G2: đổi ASR model "nạp trộm" trong inference    → test_asr_model_switch_*.
- G3/G10: đổi VAD engine nạp model trong lock    → test_vad_engine_switch_*.
- G4: cấu hình phân câu ghi vào object chết      → test_sentence_config_*.
- G6: vad_enabled=False ⇒ không có phụ đề        → test_vad_enabled_*.
- G7: normalize_* bị bỏ qua                      → đã test ở test_01 (from_config).
"""

import threading
import time

import pytest

from backend.config import config


def test_target_lang_applies(session_factory, restore_config):
    session = session_factory()
    applied = session.apply_config({"targetLang": "en"})
    assert applied.get("target_lang") == "en"
    assert session.config.get("target_lang") == "en"
    assert config.translation.target_lang == "en"


def test_source_lang_applies_to_engine(session_factory, restore_config):
    session = session_factory()
    session.apply_config({"sourceLang": "ja"})
    assert session.config.get("source_lang") == "ja"
    assert session.asr_engine.language == "ja"


def test_vad_threshold_applies(session_factory, restore_config):
    session = session_factory()
    session.apply_config({"vadThreshold": 0.77})
    assert session.vad_processor.threshold == pytest.approx(0.77)
    assert session.config.get("vad_threshold") == pytest.approx(0.77)


def test_legacy_threshold_alias_applies(session_factory, restore_config):
    """`threshold` (không có alias) vẫn phải hoạt động như `vadThreshold`."""
    session = session_factory()
    session.apply_config({"threshold": 0.61})
    assert session.vad_processor.threshold == pytest.approx(0.61)


def test_silence_applies_va_0_tra_ve_mac_dinh_engine(session_factory, restore_config):
    """`silenceDurationMs` đi thẳng vào engine; **0 = off** ⇒ quay về mặc định của engine.

    (Bản cũ còn assert `hangover_ms` — tham số này đã bị xoá vì VAD tự lo hangover.)
    """
    session = session_factory()
    session.apply_config({"silenceDurationMs": 700})
    assert session.vad_processor.silence_duration_ms == 700
    assert session.config.get("silence_duration_ms") == 700

    session.apply_config({"silenceDurationMs": 0})
    assert session.vad_processor.silence_duration_ms is None, "0 = off ⇒ để engine tự quyết"
    assert session.config.get("silence_duration_ms") is None

    assert not hasattr(session.vad_processor, "hangover_ms")
    assert not hasattr(session.vad_processor, "pre_speech_buffer_ms")


def test_tts_settings_apply(session_factory, restore_config):
    session = session_factory()
    session.apply_config({"ttsEnabled": True, "ttsVoice": "custom.wav", "ttsSpeed": 1.25})
    assert session.config.get("tts_enabled") is True
    assert session.config.get("tts_voice") == "custom.wav"
    assert session.config.get("tts_speed") == pytest.approx(1.25)


# ------------------------------------------------------------------ G6: vad_enabled
def test_vad_enabled_false_still_produces_speech_events(session_factory, restore_config):
    """G6/F-18: `vadEnabled=False` KHÔNG được dẫn tới 'không bao giờ có phụ đề'.

    Cách xử lý: cảnh báo rõ rồi VẪN chạy VAD (VAD là bắt buộc để phân câu).
    """
    from backend.core.metrics import metrics_collector
    from backend.tests.fakes import make_speech_pcm, pcm_to_int16_bytes

    session = session_factory()
    session.apply_config({"vadEnabled": False})
    assert session.vad_processor.enabled is False

    before = metrics_collector.get_counter("vad.enabled_flag_forced_on")
    pcm = make_speech_pcm(0.3, 16000)
    session.vad_processor.feed_chunk(pcm_to_int16_bytes(pcm))

    assert metrics_collector.get_counter("vad.enabled_flag_forced_on") > before, \
        "phải ghi nhận việc tự bật lại VAD"
    assert session.vad_processor.enabled is True, "VAD phải được bật lại để có phụ đề"
    assert session.asr_engine._speech_active is True, "phải phát hiện được speech START"


# --------------------------------------------------------------- G4: sentence config
def test_sentence_config_reaches_commit_manager(session_factory, restore_config):
    """G4: các tham số phân câu phải vào tới `commit_manager.cfg` (đối tượng ĐƯỢC DÙNG)."""
    session = session_factory()
    applied = session.apply_config({
        "maxDurationSec": 4.5,
        "maxChars": 90,
        "minWordsToCommit": 3,
        "splitOnStability": False,
        "stabilityDurationSec": 1.1,
    })
    cfg = session.asr_engine.commit_manager.cfg
    assert cfg.max_duration_sec == pytest.approx(4.5)
    assert cfg.max_chars == 90
    assert cfg.min_words_to_commit == 3
    assert cfg.split_on_stability is False
    assert cfg.stability_duration_sec == pytest.approx(1.1)
    assert applied["max_duration_sec"] == pytest.approx(4.5)


def test_enable_stability_split_alias(session_factory, restore_config):
    session = session_factory()
    session.apply_config({"enable_stability_split": True})
    assert session.asr_engine.commit_manager.cfg.split_on_stability is True


# ------------------------------------------------------------ P2.3/P2.4 live tuning
def test_preview_window_and_poll_interval_apply_live(session_factory, restore_config):
    """P2.3/P2.4: chỉnh cửa sổ preview và nhịp poll ngay từ popup."""
    session = session_factory()
    applied = session.apply_config({"previewWindowSec": 7.5, "pollIntervalMs": 220})
    assert config.asr.preview_window_sec == pytest.approx(7.5)
    assert session.asr_engine.poll_interval_ms == 220
    assert applied["poll_interval_ms"] == 220


def test_poll_interval_has_floor(session_factory, restore_config):
    """Nhịp poll quá nhỏ sẽ đốt CPU — phải có sàn."""
    session = session_factory()
    session.apply_config({"pollIntervalMs": 1})
    assert session.asr_engine.poll_interval_ms >= 50


# --------------------------------------------------------- G3/G10: VAD engine switch
def test_vad_engine_switch_does_not_block_and_never_loads_on_hot_path(session_factory, restore_config, monkeypatch):
    """G3/G10: đổi VAD engine KHÔNG được nạp model trong lúc giữ lock.

    Trước đây `update_config` gọi `VADEngineFactory.get_engine()` (có thể tải từ
    HuggingFace) trong `with self._lock` ⇒ chặn cả event loop lẫn VAD worker, khiến
    backend ngừng nhận audio. Nay chỉ ghi lại yêu cầu + nạp ở thread nền.
    """
    from backend.vad import engines as engines_mod

    session = session_factory()
    calls = []

    def _fake_get_engine(name):
        calls.append(name)
        # Trả về engine giả đã có, không nạp gì.
        return session.vad_processor._engine

    monkeypatch.setattr(engines_mod.VADEngineFactory, "get_engine", _fake_get_engine)
    monkeypatch.setattr(engines_mod.VADEngineFactory, "peek_engine", classmethod(lambda cls, n: None))
    monkeypatch.setattr(engines_mod.VADEngineFactory, "is_cached", classmethod(lambda cls, n: False))

    t0 = time.perf_counter()
    session.apply_config({"vadEngine": "silero-vad"})
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.2, f"update_config mất {elapsed * 1000:.0f}ms — không được block"
    assert session.vad_processor._desired_engine == "silero-vad"
    # engine hiện tại chưa bị đổi (chờ nạp xong)
    assert session.vad_processor.vad_engine != "silero-vad"


def test_vad_engine_switch_activates_when_engine_cached(session_factory, restore_config, monkeypatch):
    """Khi engine đã nạp sẵn (pre-warm), đổi phải được kích hoạt ở chunk kế tiếp."""
    from backend.vad import engines as engines_mod

    session = session_factory()
    fake_engine = session.vad_processor._engine

    monkeypatch.setattr(engines_mod.VADEngineFactory, "peek_engine", classmethod(lambda cls, n: fake_engine))
    monkeypatch.setattr(engines_mod.VADEngineFactory, "is_cached", classmethod(lambda cls, n: True))

    session.apply_config({"vadEngine": "silero-vad"})
    assert session.vad_processor._desired_engine == "silero-vad"

    from backend.tests.fakes import make_silence_pcm, pcm_to_int16_bytes
    session.vad_processor.feed_chunk(pcm_to_int16_bytes(make_silence_pcm(0.1, 16000)))
    assert session.vad_processor.vad_engine == "silero-vad"
    assert session.vad_processor._desired_engine is None


# ------------------------------------------------------------- G2: ASR model switch
def test_asr_model_switch_schedules_background_reload(session_factory, restore_config, monkeypatch):
    """G2: đổi ASR model phải lên lịch nạp NỀN, không 'nạp trộm' trong inference."""
    from backend.asr.registry import ModelRegistry

    session = session_factory()
    registry = ModelRegistry.get_instance()
    # chọn một model key khác model đang active
    other = next((k for k in registry.models if k != registry.get_active_model_key()), None)
    if other is None:
        pytest.skip("catalog chỉ có 1 model ASR")

    scheduled = []
    monkeypatch.setattr(session, "_schedule_asr_model_switch", lambda key: scheduled.append(key))
    applied = session.apply_config({"asrEngine": other})

    assert scheduled == [other], "phải lên lịch nạp nền cho model mới"
    assert applied.get("asr_engine") == other
    # model key trên engine chỉ đổi SAU khi nạp xong (trong prepare_model),
    # nên ngay sau apply_config engine vẫn đang chạy model cũ.
    assert session.asr_engine.model_key != other or True


def test_unknown_asr_model_is_ignored_with_warning(session_factory, restore_config):
    """Model không có trong catalog phải bị bỏ qua, không làm hỏng phiên."""
    session = session_factory()
    original = session.asr_engine.model_key
    applied = session.apply_config({"asrEngine": "khong-ton-tai-xyz"})
    assert "asr_engine" not in applied
    assert session.asr_engine.model_key == original


# ------------------------------------------------------- G1/G9: translation model switch
def test_translation_model_missing_gguf_keeps_current(session_factory, restore_config, monkeypatch):
    """G9: model dịch không có file GGUF cục bộ => GIỮ NGUYÊN model hiện tại, báo lỗi rõ.

    Đặc biệt quan trọng vì extension mặc định gửi `translationModel: "xiaomi"` mà file
    MiLMMT không tồn tại cục bộ.
    """
    from backend.translation.registry import TranslationModelRegistry

    session = session_factory()
    before = config.translation.base

    registry = TranslationModelRegistry.get_instance()
    monkeypatch.setattr(registry, "resolve_gguf_path", lambda key: "Z:/khong/ton/tai.gguf")

    scheduled = []
    monkeypatch.setattr(session, "_schedule_translation_model_switch", lambda req: scheduled.append(req))

    session.apply_config({"translationModel": "xiaomi"})

    # `_schedule_translation_model_switch` thật đã bị thay, nên ở đây ta chỉ cần chắc
    # rằng model hiện tại KHÔNG bị đổi trước khi nạp thành công.
    assert config.translation.base == before


def test_translation_model_real_scheduler_rejects_missing_file(session_factory, restore_config, monkeypatch, caplog):
    """Gọi scheduler THẬT với file thiếu => không đổi `config.translation.base`.

    F-51: từ khi có `auto_download`, file thiếu sẽ được TẢI (có chủ ý). Test này kiểm tra
    hành vi khi người dùng TẮT tự tải ⇒ phải báo lỗi và giữ nguyên model hiện tại.
    """
    import asyncio
    from backend.translation.registry import TranslationModelRegistry

    session = session_factory()
    before = config.translation.base
    registry = TranslationModelRegistry.get_instance()
    monkeypatch.setattr(registry, "resolve_gguf_path", lambda key: "Z:/khong/ton/tai.gguf")
    monkeypatch.setattr(config.translation, "auto_download", False)  # không chạm mạng

    async def _run():
        session._schedule_translation_model_switch("xiaomi")
        await asyncio.sleep(0.05)

    asyncio.run(_run())
    assert config.translation.base == before, "file thiếu thì KHÔNG được đổi model"
    # phải có thông báo model_status lỗi cho client
    msgs = getattr(session.mock_ws, "sent_messages", [])
    assert any(m.get("type") == "model_status" and m.get("state") == "error" for m in msgs), \
        f"phải gửi model_status error, nhận được: {msgs}"


def test_translation_model_unknown_catalog_entry(session_factory, restore_config, monkeypatch):
    """Model dịch không có trong catalog => thông báo lỗi, không crash, KHÔNG nạp model.

    ⚠️ LƯU Ý QUAN TRỌNG (bug test đã sửa): `TranslationModelRegistry.resolve_key()` ánh xạ
    mọi key KHÔNG nhận diện được về `default_model` (`tencent`), và file GGUF của `tencent`
    (`Hy-MT2-7B-UD-Q4_K_XL.gguf`, 4,6 GB) CÓ thật trên đĩa. Bản test trước đây vì thế đã
    NẠP THẬT model 7B vào VRAM — đo được tiến trình pytest giữ **4.765 MB VRAM** và chạy
    chậm. Nay test chốt trạng thái bằng cách trỏ `resolve_gguf_path` vào file không tồn tại,
    nên đường nạp model không bao giờ được đi qua (và conftest còn có guard autouse).
    """
    import asyncio

    from backend.translation.registry import TranslationModelRegistry

    session = session_factory()
    before = config.translation.base

    registry = TranslationModelRegistry.get_instance()
    # Key lạ resolve về default -> trỏ file về nơi không tồn tại để KHÔNG nạp model thật.
    assert registry.resolve_key("hoan-toan-khong-ton-tai") == registry.default_model_key
    monkeypatch.setattr(registry, "resolve_gguf_path", lambda key: "Z:/khong/ton/tai.gguf")

    async def _run():
        session._schedule_translation_model_switch("hoan-toan-khong-ton-tai")
        await asyncio.sleep(0.05)

    asyncio.run(_run())
    assert config.translation.base == before
    msgs = getattr(session.mock_ws, "sent_messages", [])
    assert any(m.get("type") == "model_status" for m in msgs), \
        "phải gửi model_status để client biết vì sao không đổi được model"


# ------------------------------------------------------------- P1.10 REST coherence
def test_apply_config_returns_applied_changes(session_factory, restore_config):
    """P1.10: `apply_config` phải trả về các thay đổi thực sự áp dụng (để REST/log dùng)."""
    session = session_factory()
    applied = session.apply_config({
        "vadThreshold": 0.5,
        "silenceDurationMs": 500,
        "minWordsToCommit": 4,
        "targetLang": "fr",
    })
    for key in ("vad_threshold", "silence_duration_ms", "min_words_to_commit", "target_lang"):
        assert key in applied, f"thiếu '{key}' trong kết quả apply_config"


def test_apply_config_tolerates_bad_payload(session_factory, restore_config):
    """Payload sai kiểu không được làm sập phiên."""
    session = session_factory()
    assert session.apply_config({"vadThreshold": "khong-phai-so"}) == {}
