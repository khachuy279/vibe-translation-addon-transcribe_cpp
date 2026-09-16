"""Test tầng A — tự tải model dịch + đổi model NGUYÊN TỬ (F-50 / F-51).

Bối cảnh thật (người dùng báo):
    POST /api/config {"translation_model": "tencent-1.8b"}
    → FileNotFoundError: Không tìm thấy file GGUF Translation tại:
      backend\\models\\Hy-MT2-1.8B-UD-Q8_K_XL.gguf

Hai lỗi lộ ra cùng lúc:
1. **F-50 (nghiêm trọng):** `main.py` ghi `config.translation.base` TRƯỚC khi nạp, còn
   `GGUFTranslator.reconfigure()` gọi `unload_model()` rồi mới `load_model()`. Nạp lỗi ⇒
   model 7B đang chạy tốt đã bị giải phóng + config trỏ vào model hỏng ⇒ backend mất khả
   năng dịch cho tới khi restart.
2. **F-51:** catalog chỉ khai báo tên file trong `backend/models`, không có đường tải, nên
   chọn model chưa có file là chắc chắn lỗi — trong khi popup vẫn liệt kê model đó.

Bộ test này chốt: file thiếu ⇒ tải về (luồng nền, không chặn hot path) hoặc báo lỗi rõ;
nạp lỗi ⇒ model cũ còn nguyên; config chỉ đổi khi model mới đã sẵn sàng.
Không test nào chạm mạng hay nạp model thật.
"""

import asyncio

import pytest
from fastapi import HTTPException

from backend.config import TranslationConfig, config


@pytest.fixture
def reset_hotswap_state():
    """Trạng thái tải/nạp là biến toàn cục — dọn trước và sau mỗi test."""
    from backend.translation import hotswap

    hotswap.reset_state()
    yield hotswap
    hotswap.reset_state()


@pytest.fixture
def tmp_models(monkeypatch, tmp_path):
    """Cô lập hoàn toàn vị trí model dịch (không đụng file GGUF thật).

    VÌ SAO PHẢI VÁ CẢ HAI ĐẦU (bug của chính bộ test, đã gây 2 test đỏ vô nghĩa):
    `resolve_gguf_path()` đọc `registry.MODELS_DIR`, nhưng `ensure_model_file()` lại nhận
    `local_dir` mặc định đã bind sẵn từ `config.MODELS_DIR` **tại thời điểm import**.
    Trước đây fixture chỉ vá `trans_registry.MODELS_DIR`, nên:
      - `resolve_gguf_path()` ⇒ trỏ thư mục tạm (file *thiếu*);
      - `ensure_model_file()` ⇒ đọc `backend/models` THẬT (file *có sẵn*).
    Kết quả: test tưởng đang kiểm tra "file thiếu" nhưng thực tế **nạp model 1.8B thật lên
    GPU**, và `/api/config` trả 202 thay vì 409. Vì vậy fixture này vá thêm
    `model_download.ensure_model_file` (và bản đã import vào `translation.engine`) để mọi
    đường tải/nạp đều nhìn vào `tmp_path`.
    """
    from backend.translation import engine as trans_engine
    from backend.translation import registry as trans_registry
    from backend.utils import model_download

    monkeypatch.setattr(trans_registry, "MODELS_DIR", tmp_path)

    # `local_dir` là keyword-only ⇒ chỉ cần tiêm mặc định mới khi caller không truyền.
    real_ensure = model_download.ensure_model_file

    def _ensure(repo_id, filename, **kwargs):
        kwargs.setdefault("local_dir", tmp_path)
        return real_ensure(repo_id, filename, **kwargs)

    monkeypatch.setattr(model_download, "ensure_model_file", _ensure)
    monkeypatch.setattr(trans_engine, "ensure_model_file", _ensure)
    return tmp_path


# ─────────────────────────────────────────────── catalog / cờ đã tải
def test_list_models_marks_is_downloaded(tmp_models):
    """Popup cần biết model nào đã có file để không hiển thị như thể dùng được ngay."""
    from backend.translation.registry import TranslationModelRegistry

    (tmp_models / "Hy-MT2-7B-UD-Q4_K_XL.gguf").write_bytes(b"gguf")

    models = {m["id"]: m for m in TranslationModelRegistry.get_instance().list_models()}
    assert models["tencent"]["is_downloaded"] is True
    assert models["tencent-1.8b"]["is_downloaded"] is False
    assert all("is_downloaded" in m for m in models.values())


def test_is_known_rejects_typo_instead_of_falling_back(tmp_models):
    """`resolve_key` fallback về model mặc định — nên phải có `is_known` để bắt tên sai."""
    from backend.translation.registry import TranslationModelRegistry

    registry = TranslationModelRegistry.get_instance()
    assert registry.is_known("tencent-1.8b")
    assert registry.is_known("hy-mt2-1.8b")  # alias
    assert not registry.is_known("khong-ton-tai")
    assert registry.resolve_key("khong-ton-tai") == registry.default_model_key


# ─────────────────────────────────────────────── ensure_model_file
def test_ensure_model_file_existing_never_touches_network(monkeypatch, tmp_path):
    """File đã có (kể cả do người dùng copy tay) ⇒ trả về ngay, không gọi mạng."""
    from backend.utils import model_download

    target = tmp_path / "Hy-MT2-1.8B-UD-Q8_K_XL.gguf"
    target.write_bytes(b"gguf")

    def _boom(**kwargs):  # pragma: no cover - chỉ chạy nếu test sai
        raise AssertionError("Không được gọi mạng khi file đã tồn tại")

    monkeypatch.setattr(model_download, "hf_hub_download", _boom)
    assert model_download.ensure_model_file("unsloth/Hy-MT2-1.8B-GGUF", target.name, local_dir=tmp_path) == str(target)


def test_ensure_model_file_missing_and_download_disabled(monkeypatch, tmp_path):
    """Tắt auto_download + thiếu file ⇒ lỗi rõ ràng, có đường dẫn và link tải tay."""
    from backend.utils import model_download

    def _boom(**kwargs):  # pragma: no cover - chỉ chạy nếu test sai
        raise AssertionError("allow_download=False thì không được gọi mạng")

    monkeypatch.setattr(model_download, "hf_hub_download", _boom)
    with pytest.raises(model_download.ModelFileMissing) as excinfo:
        model_download.ensure_model_file(
            "unsloth/Hy-MT2-1.8B-GGUF", "Hy-MT2-1.8B-UD-Q8_K_XL.gguf",
            local_dir=tmp_path, allow_download=False,
        )
    message = str(excinfo.value)
    assert "Hy-MT2-1.8B-UD-Q8_K_XL.gguf" in message
    assert "huggingface.co" in message


def test_ensure_model_file_downloads_into_models_dir(monkeypatch, tmp_path):
    """Tải xong phải nằm đúng `MODELS_DIR/<tên file>` — nơi `resolve_gguf_path()` trỏ tới."""
    from backend.utils import model_download

    def _fake_list_repo_files(repo_id):
        return ["README.md", "sub/Hy-MT2-1.8B-UD-Q8_K_XL.gguf"]

    def _fake_download(repo_id, filename, local_dir):
        path = tmp_path / "sub" / "Hy-MT2-1.8B-UD-Q8_K_XL.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 2048)
        return str(path)

    monkeypatch.setattr(model_download, "list_repo_files", _fake_list_repo_files)
    monkeypatch.setattr(model_download, "hf_hub_download", _fake_download)
    monkeypatch.setattr(model_download, "_expected_size_bytes", lambda repo, name: 2048)

    out = model_download.ensure_model_file(
        "unsloth/Hy-MT2-1.8B-GGUF", "Hy-MT2-1.8B-UD-Q8_K_XL.gguf", local_dir=tmp_path,
    )
    assert out == str(tmp_path / "Hy-MT2-1.8B-UD-Q8_K_XL.gguf")
    state = model_download.get_state("unsloth/Hy-MT2-1.8B-GGUF", "Hy-MT2-1.8B-UD-Q8_K_XL.gguf")
    assert state["state"] == "ready"
    assert state["percent"] == 100.0


def test_ensure_model_file_download_failure_sets_error_state(monkeypatch, tmp_path):
    """Lỗi mạng ⇒ `ModelDownloadError` + trạng thái error (popup hiển thị được)."""
    from backend.utils import model_download

    monkeypatch.setattr(model_download, "list_repo_files", lambda repo: ["m.gguf"])
    monkeypatch.setattr(model_download, "_expected_size_bytes", lambda repo, name: None)

    def _boom(repo_id, filename, local_dir):
        raise OSError("mất mạng")

    monkeypatch.setattr(model_download, "hf_hub_download", _boom)
    with pytest.raises(model_download.ModelDownloadError):
        model_download.ensure_model_file("repo/x", "m.gguf", local_dir=tmp_path)
    state = model_download.get_state("repo/x", "m.gguf")
    assert state["state"] == "error"
    assert "mất mạng" in state["error"]


# ─────────────────────────────────────────────── hotswap (state machine)
def test_hotswap_reserve_sets_state_before_task_starts(reset_hotswap_state, monkeypatch, tmp_models):
    """REST trả 202 ngay ⇒ trạng thái phải được đặt ĐỒNG BỘ, không phụ thuộc task nền."""
    hotswap = reset_hotswap_state
    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)

    snapshot = hotswap.reserve("tencent-1.8b")
    assert snapshot["started"] is True
    assert snapshot["state"] == "downloading"
    assert hotswap.status()["model"] == "tencent-1.8b"

    again = hotswap.reserve("tencent-1.8b")
    assert again["started"] is False, "bấm hai lần không được sinh hai lượt tải"


def test_hotswap_failure_keeps_current_model_and_config(reset_hotswap_state, monkeypatch, tmp_models):
    """Nạp/tải lỗi ⇒ `config.translation.base` KHÔNG đổi và trạng thái là error."""
    from backend.utils import model_download

    hotswap = reset_hotswap_state
    before = config.translation.base
    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)

    def _boom(*args, **kwargs):
        raise model_download.ModelDownloadError("tải thất bại")

    monkeypatch.setattr(model_download, "ensure_model_file", _boom)

    hotswap.reserve("tencent-1.8b")
    with pytest.raises(model_download.ModelDownloadError):
        asyncio.run(hotswap.run_reserved("tencent-1.8b"))

    assert config.translation.base == before
    assert hotswap.status()["state"] == "error"


def test_hotswap_disabled_download_raises_before_touching_model(reset_hotswap_state, monkeypatch, tmp_models):
    """Tắt auto_download ⇒ lỗi `ModelFileMissing` ngay ở bước tải, chưa chạm model đang chạy."""
    from backend.utils import model_download
    from backend.translation import hotswap

    monkeypatch.setattr(hotswap, "auto_download_enabled", lambda: False)
    before = config.translation.base
    hotswap.reserve("tencent-1.8b", allow_download=False)
    with pytest.raises(model_download.ModelFileMissing):
        asyncio.run(hotswap.run_reserved("tencent-1.8b", allow_download=False))
    assert config.translation.base == before


# ─────────────────────────────────────────────── swap nguyên tử trong engine
class _FakeLlama:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


def test_reconfigure_failure_keeps_working_model(
    allow_real_translation_methods, monkeypatch, tmp_models
):
    """F-50: model mới nạp lỗi ⇒ model cũ VẪN còn (trước đây bị unload mất)."""
    from backend.translation.engine import GGUFTranslator

    old = _FakeLlama("old-7b")
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", old)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", "tencent")

    translator = GGUFTranslator.get_instance()
    monkeypatch.setattr(translator, "canonical_key", "tencent")

    with pytest.raises(FileNotFoundError):
        translator.reconfigure(TranslationConfig(base="tencent-1.8b", auto_download=False), allow_download=False)

    assert GGUFTranslator._shared_llm is old, "model cũ KHÔNG được giải phóng khi nạp model mới lỗi"
    assert GGUFTranslator._shared_model_key == "tencent"
    assert translator.canonical_key == "tencent"
    assert old.closed is False


def test_reconfigure_success_swaps_then_releases_old(
    allow_real_translation_methods, monkeypatch, tmp_models
):
    """Thành công: swap sang model mới rồi mới đóng model cũ (không có khoảng trống dịch)."""
    from backend.translation.engine import GGUFTranslator

    old = _FakeLlama("old-7b")
    new = _FakeLlama("new-1.8b")
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", old)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", "tencent")
    monkeypatch.setattr(
        GGUFTranslator, "_build_llm",
        lambda self, key, cfg, allow: (new, "prompt-moi"),
    )

    translator = GGUFTranslator.get_instance()
    monkeypatch.setattr(translator, "canonical_key", "tencent")

    translator.reconfigure(TranslationConfig(base="tencent-1.8b", auto_download=False), allow_download=False)

    assert GGUFTranslator._shared_llm is new
    assert GGUFTranslator._shared_model_key == "tencent-1.8b"
    assert translator.canonical_key == "tencent-1.8b"
    assert translator.prompt_strategy == "prompt-moi"
    assert old.closed is True, "model cũ phải được giải phóng SAU khi swap xong"


def test_hot_path_degrades_to_passthrough_when_model_unavailable(monkeypatch, tmp_models):
    """Model thiếu/hỏng ⇒ trả nguyên văn bản gốc, KHÔNG ném lỗi làm chết đường dịch."""
    from backend.translation.engine import GGUFTranslator

    translator = GGUFTranslator.get_instance()
    monkeypatch.setattr(translator, "canonical_key", "tencent-1.8b")

    def _boom(self, allow_download=False):
        raise FileNotFoundError("thiếu file")

    # PHẢI vá ở CLASS, không vá ở instance: `load_model` là method của class nên
    # `monkeypatch.setattr(translator, "load_model", ...)` sẽ tạo một ATTRIBUTE INSTANCE;
    # khi teardown, monkeypatch "khôi phục" bằng cách setattr lại giá trị cũ lên chính
    # instance đó ⇒ attribute instance TỒN TẠI VĨNH VIỄN và che method của class cho mọi
    # test sau dùng cùng singleton. Đây chính là thứ làm `test_25` đỏ khi chạy full suite
    # nhưng xanh khi chạy riêng.
    monkeypatch.setattr(GGUFTranslator, "load_model", _boom)
    out = translator._translate_sync("Xin chào", source_lang="en", target_lang="vi")
    assert out["translated_text"] == "Xin chào"
    assert out["elapsed_ms"] == 0.0


# ─────────────────────────────────────────────── REST /api/config
def _post_config(**payload):
    import backend.main as main_mod

    req = main_mod.SwitchModelRequest(**payload)
    return asyncio.run(main_mod.update_backend_config(req))


def test_rest_rejects_unknown_translation_model(reset_hotswap_state, tmp_models):
    with pytest.raises(HTTPException) as excinfo:
        _post_config(translation_model="khong-ton-tai")
    assert excinfo.value.status_code == 400
    assert "translation_models.yaml" in str(excinfo.value.detail)


def test_rest_missing_file_with_auto_download_off_returns_400(
    reset_hotswap_state, monkeypatch, tmp_models
):
    """Tắt auto_download ⇒ 400 rõ ràng, `config.translation.base` giữ nguyên."""
    hotswap = reset_hotswap_state
    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)
    monkeypatch.setattr(hotswap, "auto_download_enabled", lambda: False)
    before = config.translation.base

    with pytest.raises(HTTPException) as excinfo:
        _post_config(translation_model="tencent-1.8b")
    assert excinfo.value.status_code == 400
    assert config.translation.base == before


def test_rest_missing_file_with_auto_download_returns_202_and_keeps_model(
    reset_hotswap_state, monkeypatch, tmp_models
):
    """Chưa có file + bật auto_download ⇒ 202 ngay, tải ở nền, model cũ vẫn phục vụ."""
    import backend.main as main_mod
    from backend.translation import hotswap

    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)
    monkeypatch.setattr(hotswap, "auto_download_enabled", lambda: True)

    started = []

    async def _fake_bg(canonical_key, allow_download=True):
        started.append((canonical_key, allow_download))

    monkeypatch.setattr(main_mod, "_activate_translation_model_bg", _fake_bg)
    before = config.translation.base

    async def _run():
        resp = await main_mod.update_backend_config(
            main_mod.SwitchModelRequest(translation_model="tencent-1.8b")
        )
        await asyncio.sleep(0.01)
        return resp

    resp = asyncio.run(_run())
    assert resp.status_code == 202
    body = resp.body.decode("utf-8")
    assert "downloading" in body
    assert started == [("tencent-1.8b", True)]
    assert config.translation.base == before, "chưa nạp xong thì KHÔNG được đổi model"


def test_rest_busy_with_another_model_returns_409(reset_hotswap_state, monkeypatch, tmp_models):
    """Đang tải model khác ⇒ 409 thay vì hai lượt tải/nạp chồng nhau.

    LƯU Ý: KHÔNG monkeypatch `hotswap.is_busy`. `main.py` kiểm tra theo khoá
    (`is_busy() and not is_busy(canonical_key)`), nên một stub `lambda key=None: True`
    sẽ vô hiệu hoá vế thứ hai và biến test thành "mong đợi 409 nhưng nhận 202".
    Chỉ cần đặt trạng thái thật là đủ để tái hiện tình huống.
    """
    hotswap = reset_hotswap_state

    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)
    hotswap._set_state(state="downloading", model="xiaomi")

    assert hotswap.is_busy() is True
    assert hotswap.is_busy("tencent-1.8b") is False, "đang tải model KHÁC"

    with pytest.raises(HTTPException) as excinfo:
        _post_config(translation_model="tencent-1.8b")
    assert excinfo.value.status_code == 409


def test_rest_busy_with_same_model_does_not_409(reset_hotswap_state, monkeypatch, tmp_models):
    """Bấm lại ĐÚNG model đang tải ⇒ không 409 (tránh chặn nhầm thao tác lặp của user)."""
    hotswap = reset_hotswap_state

    monkeypatch.setattr(hotswap, "needs_download", lambda key: True)
    hotswap._set_state(state="downloading", model="tencent-1.8b")

    assert hotswap.is_busy() is True
    assert hotswap.is_busy("tencent-1.8b") is True, "cùng model ⇒ nhánh 409 phải bị bỏ qua"

    # `reserve()` thấy đã có lượt chạy cho đúng model này ⇒ không xếp thêm task nền (202).
    resp = _post_config(translation_model="tencent-1.8b")
    assert getattr(resp, "status_code", None) == 202


def test_config_response_exposes_download_status(tmp_models):
    """`/api/config` phải có `translation.download` + `auto_download` + cờ is_downloaded."""
    import backend.main as main_mod

    payload = main_mod._build_config_response(include_catalog=False)
    assert "download" in payload["translation"]
    assert payload["translation"]["download"]["state"] in ("idle", "downloading", "loading", "ready", "error")
    assert payload["translation"]["auto_download"] is True


# ─────────────────────────────────────────────── đường WebSocket (dùng chung hotswap)
def test_ws_switch_announces_download_then_ready(
    session_factory, reset_hotswap_state, monkeypatch, tmp_models
):
    """WS: thiếu file ⇒ báo `downloading` cho client rồi mới `ready` (không im lặng)."""
    from backend.translation import hotswap

    session = session_factory()
    called = {}

    async def _fake_run_reserved(canonical_key, allow_download=None):
        called["key"] = canonical_key
        called["allow"] = allow_download

    monkeypatch.setattr(hotswap, "run_reserved", _fake_run_reserved)

    async def _run():
        session._schedule_translation_model_switch("tencent-1.8b")
        await asyncio.sleep(0.05)

    asyncio.run(_run())

    states = [m.get("state") for m in session.mock_ws.sent_messages if m.get("type") == "model_status"]
    assert states[0] == "downloading", f"phải báo đang tải trước, nhận được: {states}"
    assert "ready" in states
    assert called["key"] == "tencent-1.8b"


def test_popup_handles_202_and_polls_progress():
    """Popup phải biết xử lý 202 (tải nền) và hiển thị trạng thái 'chưa tải'."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "extension_firefox" / "popup" / "popup.js").read_text(encoding="utf-8")
    assert "res.status === 202" in src
    assert "waitForTranslationActivation" in src
    assert "chưa tải" in src
