"""QWEN-Q3 — chuỗi "không prewarm" (P1, độ trễ câu đầu).

Ba lỗ hổng:
1. `lifespan` KHÔNG có TTS ⇒ `_synthesize_audio` nạp lazy (`if not self._is_loaded:
   self.load_model()`) nên câu lồng tiếng ĐẦU TIÊN sau mỗi lần bật TTS phải trả giá vài GB
   trọng số + warmup (nhiều giây) — đúng câu người dùng vừa chờ.
2. `_prewarm_translation` chỉ gọi `load_model()`, KHÔNG chạy câu giả ⇒ lần dịch đầu vẫn phải
   compile kernel CUDA. `GGUFTranslator.prewarm()` tồn tại nhưng không có call-site nào
   ngoài test.
3. Đường init khi người dùng đã bật TTS trong storage không gửi `set_config` ⇒
   `handler.py` không prewarm.

Test ở đây kiểm dây nối (wiring) + hành vi của từng hàm prewarm với engine giả.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import backend.main as main_mod

ROOT = Path(__file__).resolve().parent.parent.parent
MAIN_SRC = ROOT / "backend" / "main.py"


def _unparse(func_name: str) -> str:
    tree = ast.parse(MAIN_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return ast.unparse(node)
    raise AssertionError(f"không tìm thấy {func_name} trong main.py")


# ─────────────────────────────────────────────── 1. translation dùng prewarm()

def test_prewarm_translation_goi_prewarm_chu_khong_chi_load_model():
    """`prewarm()` chạy thêm 1 câu giả để build kernel/graph CUDA cho lần dịch đầu."""
    body = _unparse("_prewarm_translation")
    assert ".prewarm()" in body, (
        "`_prewarm_translation` phải gọi `prewarm()` — chỉ `load_model()` thì câu dịch ĐẦU "
        "TIÊN vẫn phải compile kernel CUDA (QWEN-Q3)"
    )


def test_prewarm_translation_goi_engine_that_va_nuot_loi(monkeypatch):
    """Hợp đồng: gọi `prewarm()` trên engine dịch, lỗi chỉ log (không làm chết startup)."""
    calls = []
    fake = SimpleNamespace(prewarm=lambda: calls.append("prewarm"),
                           load_model=lambda: calls.append("load_model"))
    monkeypatch.setattr(main_mod, "get_translation_engine", lambda: fake)

    main_mod._prewarm_translation()
    assert calls == ["prewarm"], f"phải gọi đúng `prewarm()`, nhận được {calls}"

    boom = SimpleNamespace(prewarm=lambda: (_ for _ in ()).throw(RuntimeError("hết VRAM")))
    monkeypatch.setattr(main_mod, "get_translation_engine", lambda: boom)
    main_mod._prewarm_translation()   # không được raise


# ─────────────────────────────────────────────── 2. TTS có trong lifespan

def test_lifespan_co_prewarm_tts_co_dieu_kien():
    """`lifespan` phải dựng job prewarm TTS, nhưng CHỈ khi `config.tts.enabled`."""
    body = _unparse("lifespan")
    assert "_prewarm_tts" in body, "lifespan thiếu prewarm TTS (QWEN-Q3)"
    assert "config.tts" in body and "enabled" in body, (
        "prewarm TTS phải có điều kiện `config.tts.enabled` — nếu không sẽ chiếm vài GB VRAM "
        "cho người không dùng TTS"
    )
    assert "prewarm_jobs" in body, "job prewarm phải được gom vào danh sách rồi gather"


def test_prewarm_tts_goi_load_model_va_nuot_loi(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main_mod, "get_tts_engine",
        lambda: SimpleNamespace(load_model=lambda: calls.append("load")),
    )
    main_mod._prewarm_tts()
    assert calls == ["load"]

    monkeypatch.setattr(
        main_mod, "get_tts_engine",
        lambda: SimpleNamespace(load_model=lambda: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    main_mod._prewarm_tts()   # không được raise


# ─────────────────────────────────────────────── 3. prewarm TTS có thật sự nạp model

def test_tts_prewarm_that_su_nap_model(monkeypatch):
    """Chốt ngữ nghĩa: `OmniVoiceTTS.prewarm()` phải gọi `load_model` khi chưa nạp."""
    import asyncio

    from backend.tts.engine import OmniVoiceTTS

    engine = OmniVoiceTTS()
    loads = []
    monkeypatch.setattr(engine, "_is_loaded", False)
    monkeypatch.setattr(engine, "load_model", lambda: loads.append(1))

    assert asyncio.run(engine.prewarm()) is False, "`_is_loaded` vẫn False ⇒ trả False"
    assert loads == [1], "phải nạp model"

    # Đã nạp rồi thì không nạp lại.
    monkeypatch.setattr(engine, "_is_loaded", True)
    engine._is_loaded = True
    assert asyncio.run(engine.prewarm()) is True
    assert loads == [1], "không được nạp lại khi đã sẵn sàng"


# ─────────────────────────────────────────────── 4. Lưu / nạp trạng thái TTS runtime

def test_runtime_state_persistence_and_tts_prewarm(monkeypatch, tmp_path):
    """Kiểm tra lưu / đọc trạng thái runtime_state.json phục vụ prewarm TTS khi khởi động."""
    import sys
    from backend import config as _
    cfg_mod = sys.modules["backend.config"]
    from backend.config import load_runtime_state, save_runtime_state

    test_state_file = tmp_path / "runtime_state.json"
    monkeypatch.setattr(cfg_mod, "RUNTIME_STATE_FILE", test_state_file)

    # Ban đầu chưa có file
    assert load_runtime_state() == {}

    # Lưu trạng thái bật TTS
    save_runtime_state({"tts_enabled": True, "tts_speed": 1.25})
    loaded = load_runtime_state()
    assert loaded.get("tts_enabled") is True
    assert loaded.get("tts_speed") == 1.25

    # Cập nhật tắt TTS
    save_runtime_state({"tts_enabled": False})
    loaded_after = load_runtime_state()
    assert loaded_after.get("tts_enabled") is False
    assert loaded_after.get("tts_speed") == 1.25

