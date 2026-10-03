"""Test kiểm tra xác thực các model khi khởi động backend (test_61_startup_model_verification.py).

Đảm bảo:
1. Hàm _verify_all_models_ready() phát hiện chính xác khi có model bị thiếu.
2. Khi đầy đủ model (VAD, ASR, Translate, TTS), hàm trả về (True, []).
3. Lifespan ném RuntimeError và không thông báo pipeline sẵn sàng nếu thiếu model.
"""

from pathlib import Path
import pytest

import backend.main as main_mod
from backend.config import config, MODELS_DIR


def test_verify_all_models_ready_success():
    """Khi tất cả model đều có trên đĩa, verify phải trả về True và danh sách rỗng."""
    ok, missing = main_mod._verify_all_models_ready()
    assert ok is True, f"Kỳ vọng thành công nhưng bị thiếu: {missing}"
    assert missing == []


def test_verify_all_models_ready_catches_missing_tts(monkeypatch):
    """Khi thiếu file model TTS, verify phải bắt được và trả về False."""
    monkeypatch.setattr(config.tts, "model_base", "non_existent_base.gguf")
    ok, missing = main_mod._verify_all_models_ready()
    assert ok is False
    assert any("TTS Base" in m for m in missing)


def test_verify_all_models_ready_catches_missing_asr(monkeypatch):
    """Khi thiếu file model ASR, verify phải bắt được."""
    monkeypatch.setattr(config.asr, "active_model", "non_existent_asr_model")
    ok, missing = main_mod._verify_all_models_ready()
    assert ok is False
    assert any("ASR" in m for m in missing)


def test_verify_all_models_ready_catches_missing_translate(monkeypatch):
    """Khi thiếu file model dịch, verify phải bắt được."""
    from backend.translation.registry import TranslationModelRegistry

    monkeypatch.setattr(
        TranslationModelRegistry.get_instance(),
        "resolve_gguf_path",
        lambda *args, **kwargs: str(MODELS_DIR / "non_existent_trans_file.gguf"),
    )
    ok, missing = main_mod._verify_all_models_ready()
    assert ok is False
    assert any("Translate" in m for m in missing)


def test_verify_all_models_ready_catches_missing_vad(monkeypatch):
    """Khi thiếu file model VAD đang chọn, verify phải bắt được."""
    monkeypatch.setattr(config.vad, "vad_engine", "silero-vad")
    monkeypatch.setattr(main_mod, "MODELS_DIR", Path("non_existent_folder_xyz"))
    ok, missing = main_mod._verify_all_models_ready()
    assert ok is False
    assert any("VAD" in m for m in missing)


@pytest.mark.asyncio
async def test_lifespan_raises_when_model_missing(monkeypatch):
    """Lifespan phải ném RuntimeError khi thiếu model bắt buộc."""
    monkeypatch.setattr(main_mod, "_verify_all_models_ready", lambda: (False, ["TestMissingModel"]))
    # Bỏ qua các bước prewarm trong test này để chỉ kiểm tra logic verify của lifespan
    monkeypatch.setattr(main_mod, "_prewarm_asr", lambda: None)
    monkeypatch.setattr(main_mod, "_prewarm_translation", lambda: None)
    monkeypatch.setattr(main_mod, "_prewarm_vad_default", lambda: None)
    monkeypatch.setattr(main_mod, "_ensure_tts_models_downloaded", lambda: None)
    monkeypatch.setattr(main_mod, "_prewarm_forced_aligner", lambda: None)
    monkeypatch.setattr(main_mod, "_prewarm_vad_others", lambda: {})

    with pytest.raises(RuntimeError, match="KHỞI ĐỘNG THẤT BẠI"):
        async with main_mod.lifespan(None):
            pass
