"""Test tầng A — tự tải model ASR từ models.yaml + đổi model NGUYÊN TỬ.

Catalog `models.yaml` chỉ có **2 model**, cả hai đều là **Qwen3-ASR**: `qwen3-asr-1.7b` (mặc định)
và `qwen3-asr-0.6b`.

Kiểm tra:
1. Catalog models.yaml có cờ `is_downloaded` chính xác (ví dụ `qwen3-asr-1.7b` chưa tải).
2. `ModelRegistry.ensure_model_file`: tự tải từ HuggingFace qua `model_download.py`.
3. `ASRConfig.auto_download`: cho phép bật/tắt tải tự động.
4. `asr/hotswap.py`: quản lý trạng thái idle/downloading/loading/ready/error.
5. `TranscribeEngine.prepare_model`: tự tải nếu thiếu file và swap an toàn.
6. REST API `/api/config`: trả HTTP 202 khi model chưa tải (tải nền), 409 khi đang bận tải model khác.
7. WebSocket session: gửi message `model_status` với `state: "downloading"`.
8. **Catalog chỉ còn Qwen3-ASR** — không còn entry `whisper-*` / `cohere-*` (đã gỡ khỏi dự án).
"""

import asyncio
from pathlib import Path
import pytest
from fastapi import HTTPException

from backend.config import config


@pytest.fixture
def reset_asr_hotswap_state():
    """Trạng thái tải/nạp ASR là biến toàn cục — dọn trước và sau mỗi test."""
    from backend.asr import lifecycle as asr_hotswap

    asr_hotswap.reset_state()
    yield asr_hotswap
    asr_hotswap.reset_state()


@pytest.fixture
def tmp_asr_models(monkeypatch, tmp_path):
    """Cô lập hoàn toàn vị trí model ASR (không đụng file GGUF thật)."""
    from backend.asr import registry as asr_reg
    from backend.utils import model_download

    monkeypatch.setattr(asr_reg, "MODELS_DIR", tmp_path)

    real_ensure = model_download.ensure_model_file

    def _ensure(repo_id, filename, **kwargs):
        kwargs.setdefault("local_dir", tmp_path)
        return real_ensure(repo_id, filename, **kwargs)

    monkeypatch.setattr(model_download, "ensure_model_file", _ensure)
    return tmp_path


# ─────────────────────────────────────────────── catalog / cờ đã tải
def test_asr_list_models_marks_is_downloaded(tmp_asr_models):
    """Popup cần biết model nào đã có file để hiển thị icon thích hợp.

    ⚠️ KHÔNG hard-code tên file/quant: catalog `models.yaml` đổi quant (Q8_0 ⇄ Q4_K_M…) là
    chuyện bình thường; test phải hỏi chính registry.
    """
    from backend.asr.registry import ModelRegistry

    reg = ModelRegistry.get_instance()
    present_key, absent_key = "qwen3-asr-0.6b", "qwen3-asr-1.7b"

    # Tạo giả file cho model đầu tiên, đúng TÊN trong catalog; model còn lại để trống.
    (tmp_asr_models / reg.file_name(present_key)).write_bytes(b"gguf")
    assert not (tmp_asr_models / reg.file_name(absent_key)).exists()

    models = {m["id"]: m for m in reg.list_models()}
    assert models[present_key]["is_downloaded"] is True
    assert models[absent_key]["is_downloaded"] is False
    assert all("is_downloaded" in m for m in models.values())


def test_asr_registry_helpers():
    """Helper phải trả ĐÚNG giá trị trong catalog (repo + tên file).

    Hỏi CHÍNH registry thay vì hard-code tên file/quant: đổi quant trong `models.yaml` là chuyện
    bình thường và test không được vỡ vì thế. Chỉ ghim thứ KHÔNG được đổi: repo HuggingFace.
    """
    from backend.asr.registry import ModelRegistry

    reg = ModelRegistry.get_instance()
    for key, repo in (
        ("qwen3-asr-1.7b", "handy-computer/Qwen3-ASR-1.7B-gguf"),
        ("qwen3-asr-0.6b", "handy-computer/Qwen3-ASR-0.6B-gguf"),
    ):
        info = reg.get_model_info(key) or {}
        assert reg.repo_id(key) == info["hf_repo"] == repo, f"{key}: repo lệch catalog"
        assert reg.file_name(key) == info["file"]
        assert reg.file_name(key).startswith("Qwen3-ASR-"), reg.file_name(key)
        assert reg.file_name(key).endswith(".gguf")


def test_catalog_chi_con_qwen3_asr():
    """Catalog chỉ được chứa **Qwen3-ASR** (0.6B + 1.7B) — không còn whisper/cohere/…

    Đây là yêu cầu của dự án: ASR chỉ dùng Qwen3-ASR. Nếu ai đó thêm lại family khác thì test này
    báo ngay, vì mọi đường ASR (chọn model ở popup, `/api/config`, hot-swap, pre-flight) đều dựa
    trên `models.yaml`.
    """
    from backend.asr.registry import ModelRegistry

    reg = ModelRegistry.get_instance()
    keys = {m["id"] for m in reg.list_models()}
    assert keys == {"qwen3-asr-1.7b", "qwen3-asr-0.6b"}, f"catalog lạ: {keys}"

    for m in reg.list_models():
        info = reg.get_model_info(m["id"]) or {}
        assert info.get("family") == "qwen3_asr", f"{m['id']}: family phải là qwen3_asr"
        assert "Qwen3-ASR" in str(info.get("file", "")), f"{m['id']}: tên file phải là Qwen3-ASR-*"
        assert "Qwen3" in str(info.get("hf_repo", "")), f"{m['id']}: repo phải thuộc Qwen3"

    # Model mặc định phải nằm trong catalog.
    assert config.asr.active_model in keys, (
        f"active_model={config.asr.active_model!r} không có trong models.yaml"
    )


# ─────────────────────────────────────────────── ensure_model_file
def test_ensure_asr_model_file_existing_never_touches_network(monkeypatch, tmp_asr_models):
    """File đã có cục bộ ⇒ trả về ngay, không gọi mạng."""
    from backend.asr.registry import ModelRegistry
    from backend.utils import model_download

    target = tmp_asr_models / ModelRegistry.get_instance().file_name("qwen3-asr-1.7b")
    target.write_bytes(b"gguf-data")

    def _boom(**kwargs):
        raise AssertionError("Không được gọi mạng khi file đã tồn tại")

    monkeypatch.setattr(model_download, "hf_hub_download", _boom)
    reg = ModelRegistry.get_instance()
    out = reg.ensure_model_file("qwen3-asr-1.7b")
    assert out == str(target)


def test_ensure_asr_model_file_missing_and_download_disabled(monkeypatch, tmp_asr_models):
    """Tắt auto_download + thiếu file ⇒ ném ModelFileMissing có thông báo rõ."""
    from backend.asr.registry import ModelRegistry
    from backend.utils import model_download

    def _boom(**kwargs):
        raise AssertionError("allow_download=False thì không được gọi mạng")

    monkeypatch.setattr(model_download, "hf_hub_download", _boom)
    reg = ModelRegistry.get_instance()
    with pytest.raises(model_download.ModelFileMissing) as excinfo:
        reg.ensure_model_file("qwen3-asr-1.7b", allow_download=False)
    message = str(excinfo.value)
    assert reg.file_name("qwen3-asr-1.7b") in message
    assert "huggingface.co" in message


def test_ensure_asr_model_file_downloads_when_missing(monkeypatch, tmp_asr_models):
    """Tự tải file GGUF vào MODELS_DIR nếu chưa có."""
    from backend.asr.registry import ModelRegistry
    from backend.utils import model_download

    def _fake_list_repo_files(repo_id):
        return ["README.md", ModelRegistry.get_instance().file_name("qwen3-asr-1.7b")]

    def _fake_download(repo_id, filename, local_dir):
        path = tmp_asr_models / "downloaded_raw.gguf"
        path.write_bytes(b"x" * 1024)
        return str(path)

    monkeypatch.setattr(model_download, "list_repo_files", _fake_list_repo_files)
    monkeypatch.setattr(model_download, "hf_hub_download", _fake_download)
    monkeypatch.setattr(model_download, "_expected_size_bytes", lambda repo, name: 1024)

    reg = ModelRegistry.get_instance()
    out = reg.ensure_model_file("qwen3-asr-1.7b", allow_download=True)
    expected_path = tmp_asr_models / reg.file_name("qwen3-asr-1.7b")
    assert out == str(expected_path)
    assert expected_path.is_file()
    assert expected_path.stat().st_size == 1024


# ─────────────────────────────────────────────── hotswap
def test_asr_hotswap_reserve_and_busy(reset_asr_hotswap_state, monkeypatch, tmp_asr_models):
    """reserve() đặt trạng thái đồng bộ, chặn race condition."""
    hotswap = reset_asr_hotswap_state
    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)

    snapshot = hotswap.reserve("qwen3-asr-1.7b")
    assert snapshot["started"] is True
    assert snapshot["state"] == "downloading"
    assert hotswap.is_busy() is True
    assert hotswap.is_busy("qwen3-asr-1.7b") is True
    assert hotswap.is_busy("qwen3-asr-0.6b") is False

    again = hotswap.reserve("qwen3-asr-1.7b")
    assert again["started"] is False


def test_asr_hotswap_failure_keeps_config(reset_asr_hotswap_state, monkeypatch, tmp_asr_models):
    """Tải/nạp thất bại ⇒ giữ nguyên config.asr.active_model."""
    from backend.utils import model_download

    hotswap = reset_asr_hotswap_state
    before = config.asr.active_model
    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)

    def _boom(*args, **kwargs):
        raise model_download.ModelDownloadError("mạng hỏng")

    monkeypatch.setattr(model_download, "ensure_model_file", _boom)

    hotswap.reserve("qwen3-asr-1.7b")
    with pytest.raises(model_download.ModelDownloadError):
        asyncio.run(hotswap.run_reserved("qwen3-asr-1.7b"))

    assert config.asr.active_model == before
    st = hotswap.status()
    assert st["state"] == "error"
    assert "mạng hỏng" in st["error"]


# ─────────────────────────────────────────────── TranscribeEngine.prepare_model
def test_engine_prepare_model_auto_downloads(monkeypatch, tmp_asr_models):
    """prepare_model tự động tải file nếu chưa có và auto_download=True."""
    from backend.asr.engine import TranscribeEngine

    downloaded_keys = []

    def _fake_ensure(self, model_key=None, allow_download=True):
        downloaded_keys.append(model_key)
        p = tmp_asr_models / "fake.gguf"
        p.write_bytes(b"mock")
        return str(p)

    from backend.asr.registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "ensure_model_file", _fake_ensure)

    class _FakeSession:
        limits = None
        def close(self): pass

    class _FakeModel:
        backend = "mock"
        capabilities = None
        def session(self, n_threads=4):
            return _FakeSession()
        def close(self): pass

    import transcribe_cpp
    monkeypatch.setattr(transcribe_cpp, "Model", lambda path, backend: _FakeModel())

    engine = TranscribeEngine("qwen3-asr-1.7b")
    engine.prepare_model("qwen3-asr-1.7b")
    assert "qwen3-asr-1.7b" in downloaded_keys


# ─────────────────────────────────────────────── REST /api/config
def _post_config(**payload):
    import backend.main as main_mod

    req = main_mod.SwitchModelRequest(**payload)
    return asyncio.run(main_mod.update_backend_config(req))


def test_rest_rejects_unknown_asr_model(reset_asr_hotswap_state, tmp_asr_models):
    with pytest.raises(HTTPException) as excinfo:
        _post_config(model_id="khong-ton-tai")
    assert excinfo.value.status_code == 400
    assert "models.yaml" in str(excinfo.value.detail)


def test_rest_asr_missing_file_auto_download_off_returns_400(
    reset_asr_hotswap_state, monkeypatch, tmp_asr_models
):
    """Khi auto_download tắt mà thiếu file ⇒ trả 400 rõ ràng."""
    hotswap = reset_asr_hotswap_state
    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)
    monkeypatch.setattr(hotswap, "auto_download_enabled", lambda: False)
    before = config.asr.active_model

    with pytest.raises(HTTPException) as excinfo:
        _post_config(asr_engine="qwen3-asr-1.7b")
    assert excinfo.value.status_code == 400
    assert "Chưa có file GGUF cục bộ" in str(excinfo.value.detail)
    assert config.asr.active_model == before


def test_rest_asr_missing_file_auto_download_on_returns_202(
    reset_asr_hotswap_state, monkeypatch, tmp_asr_models
):
    """Khi auto_download bật và thiếu file ⇒ trả 202 Accepted, xếp lịch chạy nền."""
    import backend.main as main_mod
    from backend.asr import hotswap

    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)
    monkeypatch.setattr(hotswap, "auto_download_enabled", lambda: True)

    started = []

    async def _fake_bg(target_model, allow_download=True):
        started.append((target_model, allow_download))

    monkeypatch.setattr(main_mod, "_activate_asr_model_bg", _fake_bg)
    before = config.asr.active_model

    resp = _post_config(asr_engine="qwen3-asr-1.7b")
    assert resp.status_code == 202
    assert config.asr.active_model == before
    assert len(started) == 1
    assert started[0][0] == "qwen3-asr-1.7b"


def test_rest_asr_busy_returns_409(reset_asr_hotswap_state, monkeypatch, tmp_asr_models):
    """Đang tải model ASR này mà yêu cầu model ASR khác ⇒ trả 409 Conflict."""
    hotswap = reset_asr_hotswap_state
    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)
    monkeypatch.setattr(hotswap, "auto_download_enabled", lambda: True)

    hotswap.reserve("qwen3-asr-1.7b")

    with pytest.raises(HTTPException) as excinfo:
        _post_config(asr_engine="qwen3-asr-0.6b")
    assert excinfo.value.status_code == 409
    assert "Đang tải/nạp model ASR" in str(excinfo.value.detail)


# ─────────────────────────────────────────────── WebSocket
@pytest.mark.asyncio
async def test_ws_schedule_asr_switch_downloading_message(monkeypatch, tmp_asr_models):
    """Session WS gửi message model_status với state=downloading nếu model chưa tải."""
    from backend.ws.session import SessionState
    from backend.asr import hotswap
    from backend.tests.conftest import MockWebSocket
    from backend.ws.connection import SafeWebSocketConnection

    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)
    monkeypatch.setattr(hotswap, "auto_download_enabled", lambda: True)

    sent = []

    session = SessionState(SafeWebSocketConnection(MockWebSocket()))

    async def _fake_send_json(payload):
        sent.append(payload)

    session.send_json = _fake_send_json

    async def _fake_activate(model_key, allow_download=None):
        pass

    monkeypatch.setattr(hotswap, "activate_model", _fake_activate)

    session._schedule_asr_model_switch("qwen3-asr-1.7b")

    # Cho loop chạy tác vụ nền
    await asyncio.sleep(0.05)

    assert any(
        msg.get("type") == "model_status"
        and msg.get("stage") == "asr"
        and msg.get("state") == "downloading"
        for msg in sent
    )
