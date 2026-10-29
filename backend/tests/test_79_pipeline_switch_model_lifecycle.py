"""test_79 — Kiểm tra vòng đời mô hình khi chuyển đổi giữa Pipeline A và Pipeline B.

Yêu cầu:
- Khi chuyển qua Pipeline A: các model chỉ dùng cho Pipeline B (ForcedAligner, Nemotron-3 Diarization)
  phải được unload để giải phóng VRAM và tài nguyên.
- Khi kích hoạt lại Pipeline B: load/prewarm lại các model này.
- Chuyển Pipeline thông qua extension (gửi qua REST API /api/config, /api/lookahead/..., /api/pipeline/...).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from backend.config import config, load_runtime_state, save_runtime_state
from backend.main import app, _prewarm_pipeline_b_models, _unload_pipeline_b_models
from backend.asr.forced_aligner import ForcedAlignerService
from backend.diarization.service import DiarizationService


@pytest.mark.asyncio
async def test_forced_aligner_and_diarization_unload_prewarm_methods():
    """Kiểm tra ForcedAlignerService và DiarizationService có phương thức unload và prewarm."""
    fa = ForcedAlignerService.get_instance()
    assert hasattr(fa, "unload_model")
    assert hasattr(fa, "prewarm")
    assert hasattr(fa, "is_loaded")

    assert hasattr(DiarizationService, "unload_model")
    assert hasattr(DiarizationService, "prewarm")
    assert hasattr(DiarizationService, "is_ready")


def test_helper_prewarm_and_unload_pipeline_b_models(monkeypatch):
    """Kiểm tra các hàm điều phối _prewarm_pipeline_b_models và _unload_pipeline_b_models gọi đúng các service."""
    fa_called = {"prewarm": 0, "unload": 0}
    diar_called = {"prewarm": 0, "unload": 0}

    monkeypatch.setattr(ForcedAlignerService.get_instance(), "prewarm", lambda: fa_called.__setitem__("prewarm", fa_called["prewarm"] + 1))
    monkeypatch.setattr(ForcedAlignerService.get_instance(), "unload_model", lambda: fa_called.__setitem__("unload", fa_called["unload"] + 1))
    monkeypatch.setattr(DiarizationService, "prewarm", classmethod(lambda cls: diar_called.__setitem__("prewarm", diar_called["prewarm"] + 1)))
    monkeypatch.setattr(DiarizationService, "unload_model", classmethod(lambda cls: diar_called.__setitem__("unload", diar_called["unload"] + 1)))

    _prewarm_pipeline_b_models()
    assert fa_called["prewarm"] == 1
    assert diar_called["prewarm"] == 1

    _unload_pipeline_b_models()
    assert fa_called["unload"] == 1
    assert diar_called["unload"] == 1


@pytest.mark.asyncio
async def test_switch_pipeline_via_api_config(monkeypatch):
    """POST /api/config với lookahead_enabled=False (Pipeline A) => unload Pipeline B models.
    POST /api/config với lookahead_enabled=True (Pipeline B) => prewarm Pipeline B models."""
    unload_calls = []
    prewarm_calls = []

    monkeypatch.setattr("backend.main._unload_pipeline_b_models", lambda: unload_calls.append(True))
    monkeypatch.setattr("backend.main._prewarm_pipeline_b_models", lambda: prewarm_calls.append(True))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Chuyển sang Pipeline A (lookahead_enabled = False)
        resp_a = await client.post("/api/config", json={"lookahead_enabled": False})
        assert resp_a.status_code == 200
        data_a = resp_a.json()
        assert data_a["lookahead_enabled"] is False
        assert data_a["pipeline"] == "A"
        assert config.lookahead.enabled is False

        # Đợi background task chạy xong
        await asyncio.sleep(0.05)
        assert len(unload_calls) >= 1

        # 2. Kích hoạt lại Pipeline B (lookahead_enabled = True)
        resp_b = await client.post("/api/config", json={"lookahead_enabled": True})
        assert resp_b.status_code == 200
        data_b = resp_b.json()
        assert data_b["lookahead_enabled"] is True
        assert data_b["pipeline"] == "B"
        assert config.lookahead.enabled is True

        # Đợi background task chạy xong
        await asyncio.sleep(0.05)
        assert len(prewarm_calls) >= 1


@pytest.mark.asyncio
async def test_dedicated_lookahead_endpoints(monkeypatch):
    """Kiểm tra các endpoint /api/lookahead/unload và /api/lookahead/prewarm."""
    unload_called = []
    prewarm_called = []

    monkeypatch.setattr("backend.main._unload_pipeline_b_models", lambda: unload_called.append(True))
    monkeypatch.setattr("backend.main._prewarm_pipeline_b_models", lambda: prewarm_called.append(True))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Unload
        resp = await client.post("/api/lookahead/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["pipeline"] == "A"
        assert config.lookahead.enabled is False
        assert len(unload_called) == 1

        # Prewarm
        resp = await client.post("/api/lookahead/prewarm")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["pipeline"] == "B"
        assert config.lookahead.enabled is True
        assert len(prewarm_called) == 1


@pytest.mark.asyncio
async def test_pipeline_switch_endpoint(monkeypatch):
    """Kiểm tra endpoint /api/pipeline/switch."""
    unload_called = []
    prewarm_called = []

    monkeypatch.setattr("backend.main._unload_pipeline_b_models", lambda: unload_called.append(True))
    monkeypatch.setattr("backend.main._prewarm_pipeline_b_models", lambda: prewarm_called.append(True))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Switch to A
        resp = await client.post("/api/pipeline/switch", json={"pipeline": "A"})
        assert resp.status_code == 200
        assert resp.json()["pipeline"] == "A"
        assert config.lookahead.enabled is False
        assert len(unload_called) == 1

        # Switch to B
        resp = await client.post("/api/pipeline/switch", json={"pipeline": "B"})
        assert resp.status_code == 200
        assert resp.json()["pipeline"] == "B"
        assert config.lookahead.enabled is True
        assert len(prewarm_called) == 1

        # Invalid pipeline name
        resp_invalid = await client.post("/api/pipeline/switch", json={"pipeline": "INVALID"})
        assert resp_invalid.status_code == 400


def test_extension_popup_lookahead_wiring():
    """Kiểm tra file popup.js của extension gửi lookahead_enabled và đấu dây unload/prewarm."""
    root = Path(__file__).resolve().parents[2]
    popup_js = (root / "extension_firefox" / "popup" / "popup.js").read_text(encoding="utf-8")

    # Kiểm tra liveUpdateSettings gửi lookahead_enabled
    assert "lookahead_enabled: cfg.lookaheadEnabled" in popup_js

    # Kiểm tra có hàm unloadLookahead và prewarmLookahead
    assert "function unloadLookahead()" in popup_js
    assert "function prewarmLookahead()" in popup_js

    # Kiểm tra chkEnableLookahead.onchange gọi unloadLookahead / prewarmLookahead
    assert "unloadLookahead()" in popup_js
    assert "prewarmLookahead()" in popup_js

