"""Test tự động tải model TTS GGUF từ Hugging Face về backend/models."""

import pytest
from pathlib import Path

from backend.config import config
from backend.tts import downloader
from backend.utils.model_download import ModelFileMissing, ModelDownloadError


@pytest.fixture
def tmp_tts_models(monkeypatch, tmp_path):
    """Cô lập thư mục models để không đụng vào backend/models thật."""
    monkeypatch.setattr(downloader, "MODELS_DIR", tmp_path)
    from backend.utils import model_download
    monkeypatch.setattr(model_download, "MODELS_DIR", tmp_path)
    return tmp_path


def test_is_model_available_false_when_missing(tmp_tts_models):
    """Khi chưa có file trong backend/models, is_model_available phải trả về False."""
    assert not downloader.is_model_available()


def test_is_model_available_true_when_present(tmp_tts_models):
    """Khi cả 2 file đã có trong backend/models, is_model_available trả về True."""
    base_path, tok_path = downloader.get_model_paths()
    base_path.write_bytes(b"dummy base gguf")
    tok_path.write_bytes(b"dummy tok gguf")

    assert downloader.is_model_available()


def test_ensure_tts_models_raises_when_missing_and_download_disabled(tmp_tts_models, monkeypatch):
    """Khi file chưa có và allow_download=False, phải ném lỗi ModelFileMissing."""
    monkeypatch.setattr(config.tts, "auto_download", False)

    with pytest.raises(ModelFileMissing):
        downloader.ensure_tts_models(allow_download=False)


def test_ensure_tts_models_returns_immediately_when_present(tmp_tts_models, monkeypatch):
    """Khi file đã có sẵn, trả về ngay lập tức mà không gọi mạng."""
    base_path, tok_path = downloader.get_model_paths()
    base_path.write_bytes(b"dummy base gguf")
    tok_path.write_bytes(b"dummy tok gguf")

    # Monkeypatch hf_hub_download to fail if called
    from backend.utils import model_download
    def _fail(*args, **kwargs):
        raise AssertionError("hf_hub_download should not be called when files exist!")
    monkeypatch.setattr(model_download, "hf_hub_download", _fail)

    b, t = downloader.ensure_tts_models(allow_download=True)
    assert b == base_path
    assert t == tok_path


def test_ensure_tts_models_downloads_into_models_dir(tmp_tts_models, monkeypatch):
    """Khi thiếu file và allow_download=True, tải về thẳng backend/models."""
    from backend.utils import model_download

    def _mock_download(repo_id, filename, local_dir, **kwargs):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mocked downloaded content")
        return str(dest)

    monkeypatch.setattr(model_download, "hf_hub_download", _mock_download)
    monkeypatch.setattr(model_download, "_expected_size_bytes", lambda *args: 1024)

    base_p, tok_p = downloader.ensure_tts_models(allow_download=True)

    assert base_p.exists()
    assert tok_p.exists()
    assert base_p.parent == tmp_tts_models
    assert tok_p.parent == tmp_tts_models
    assert downloader.is_model_available()


def test_get_download_status(tmp_tts_models):
    """Kiểm tra cấu trúc trả về của get_download_status."""
    status = downloader.get_download_status()
    assert "is_available" in status
    assert "base_model" in status
    assert "tokenizer_model" in status
    assert status["base_model"]["filename"] == config.tts.model_base
    assert status["tokenizer_model"]["filename"] == config.tts.model_tokenizer
