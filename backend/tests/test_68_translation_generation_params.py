"""test_68 — Thông số sinh (sampling) lấy từ `translation_models.yaml`, không hardcode.

Bối cảnh: đường Pipeline B (`translate_batch_sync`) từng hardcode greedy
(``temperature=0.0, top_p=1.0, top_k=1, repeat_penalty=1.05``) cho MỌI model, nên
`translate-gemma-4-sub` chạy sai thông số mà catalog của nó khai (0.1 / 0.95 / 40 / 1.05).

Bộ test này chốt:

* `resolve_generation_params` — thứ tự ưu tiên ``cfg`` → hook strategy → YAML → dự phòng;
* catalog THẬT của cả `index-translate-9b` và `translate-gemma-4-sub` chảy vào kwargs;
* hook `batch_generation_params()` là điểm ghi đè DUY NHẤT (không còn hardcode);
* strategy cũ/không có hook, hoặc hook ném lỗi, đều không làm chết đường dịch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.translation.engine import resolve_generation_params
from backend.translation.prompts import get_prompt_strategy
from backend.translation.prompts_gemma import GemmaSubPromptStrategy
from backend.translation.registry import TranslationModelRegistry

# ─────────────────────────────────────────── Tầng 1: hàm phân giải thông số


def test_yaml_la_nguon_chinh():
    info = {"temperature": 0.1, "top_p": 0.95, "top_k": 40, "repetition_penalty": 1.05}
    out = resolve_generation_params(None, info)
    assert out == {"temperature": 0.1, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.05}


def test_cfg_dat_tuong_minh_de_len_yaml():
    info = {"temperature": 0.1, "top_p": 0.95, "top_k": 40, "repetition_penalty": 1.05}
    cfg = SimpleNamespace(temperature=0.0, top_p=None, top_k=None, repetition_penalty=2.0)
    out = resolve_generation_params(cfg, info)
    assert out["temperature"] == 0.0          # cfg đè
    assert out["top_p"] == 0.95               # None ⇒ vẫn lấy YAML
    assert out["repeat_penalty"] == 2.0       # cfg đè


def test_hook_de_len_yaml_nhung_thua_cfg():
    info = {"temperature": 0.1, "top_p": 0.95, "top_k": 40, "repetition_penalty": 1.05}
    hook = {"temperature": 0.0, "top_k": 1}

    out = resolve_generation_params(None, info, hook)
    assert out["temperature"] == 0.0          # hook đè YAML
    assert out["top_k"] == 1
    assert out["top_p"] == 0.95               # không khai ⇒ vẫn YAML

    cfg = SimpleNamespace(temperature=0.3, top_p=None, top_k=None, repetition_penalty=None)
    out2 = resolve_generation_params(cfg, info, hook)
    assert out2["temperature"] == 0.3         # cfg thắng hook
    assert out2["top_k"] == 1


def test_du_phong_khi_ca_yaml_lan_cfg_deu_thieu():
    out = resolve_generation_params(None, {})
    assert out["temperature"] == 0.0
    assert out["top_p"] == 1.0
    assert out["top_k"] == 1
    assert out["repeat_penalty"] == 1.05
    assert set(out) == {"temperature", "top_p", "top_k", "repeat_penalty"}


def test_key_la_trong_hook_duoc_truyen_thang():
    out = resolve_generation_params(None, {}, {"temperature": 0.0, "max_tokens": 42})
    assert out["max_tokens"] == 42
    assert out["temperature"] == 0.0


# ─────────────────────────────────────────── Tầng 2: catalog THẬT của repo


def test_catalog_that_khai_du_thong_so_cho_moi_model():
    registry = TranslationModelRegistry.get_instance()
    for key in ("index-translate-2b", "index-translate-9b", "translate-gemma-4-sub"):
        info = registry.get_model(key) or {}
        for field in ("temperature", "top_p", "top_k", "repetition_penalty"):
            assert info.get(field) is not None, f"{key} thiếu '{field}' trong translation_models.yaml"

    gemma = registry.get_model("translate-gemma-4-sub")
    assert (gemma["temperature"], gemma["top_p"], gemma["top_k"]) == (0.0, 1.0, 1)
    assert gemma["repetition_penalty"] == 1.0


def test_pipeline_a_va_b_cung_lay_mot_nguon_yaml():
    """Cùng một model ⇒ Pipeline A và Pipeline B phải ra CÙNG thông số (không lệch hardcode)."""
    registry = TranslationModelRegistry.get_instance()
    for key in ("index-translate-9b", "translate-gemma-4-sub"):
        info = registry.get_model(key)
        single = resolve_generation_params(None, info)
        batch = resolve_generation_params(None, info, {})
        assert single == batch


# ─────────────────────────────────────────── Tầng 3: hook trên strategy

def test_hook_mac_dinh_tra_none():
    assert get_prompt_strategy("index").batch_generation_params() is None
    assert get_prompt_strategy("pipeline_b").batch_generation_params() is None


def test_gemma_hook_tra_none_de_dung_yaml():
    assert GemmaSubPromptStrategy().batch_generation_params() is None


def test_hook_hong_khong_lam_chet_duong_dich():
    from backend.translation.engine import _strategy_batch_params

    class _Boom:
        def batch_generation_params(self):
            raise RuntimeError("hook hỏng")

    class _SaiKieu:
        def batch_generation_params(self):
            return 123

    class _KhongCoHook:
        pass

    assert _strategy_batch_params(_Boom()) == {}
    assert _strategy_batch_params(_SaiKieu()) == {}
    assert _strategy_batch_params(_KhongCoHook()) == {}


# ─────────────────────────────────────────── Tầng 4: engine gọi THẬT

class _CapturingLlama:
    """LLM giả: ghi lại prompt + kwargs của từng lượt gọi."""

    def __init__(self, payload: str):
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {
            "choices": [{"text": self.payload, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10},
        }

    def close(self):
        pass


def _wire_engine(monkeypatch, llm, model_key: str, strategy):
    from backend.translation.engine import GGUFTranslator

    t = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", llm)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", model_key)
    t.canonical_key = model_key
    t.prompt_strategy = strategy
    t.cfg = SimpleNamespace(
        max_tokens=128, temperature=None, top_p=None, top_k=None, repetition_penalty=None,
        use_context=False, glossary_file="", glossary={}, glossary_derive_names=True,
    )
    return t


def test_batch_gemma_dung_thong_so_trong_yaml(allow_real_translation_methods, monkeypatch):
    """Gemma-Sub dùng đúng thông số greedy từ catalog YAML."""
    llm = _CapturingLlama('{"1": "Xin chào", "2": "Tạm biệt"}')
    t = _wire_engine(monkeypatch, llm, "translate-gemma-4-sub", GemmaSubPromptStrategy())

    out = t.translate_batch_sync(["Hello", "Bye"], "en", "vi")

    assert out == ["Xin chào", "Tạm biệt"]
    assert len(llm.calls) == 1
    _, kwargs = llm.calls[0]
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == 1
    assert kwargs["repeat_penalty"] == 1.0
    assert kwargs["max_tokens"] == 256  # min(1536, max(256, 2 * 80))


def test_batch_index_khong_bi_doi_hanh_vi(allow_real_translation_methods, monkeypatch):
    llm = _CapturingLlama('{"1": "Xin chào", "2": "Tạm biệt"}')
    t = _wire_engine(monkeypatch, llm, "index-translate-9b", get_prompt_strategy("index"))

    t.translate_batch_sync(["Hello", "Bye"], "en", "vi")

    _, kwargs = llm.calls[0]
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == 1
    assert kwargs["repeat_penalty"] == 1.0  # YAML khai 1.0 (bản cũ hardcode 1.05)


def test_single_gemma_cung_dung_thong_so_trong_yaml(allow_real_translation_methods, monkeypatch):
    llm = _CapturingLlama("Xin chào")
    t = _wire_engine(monkeypatch, llm, "translate-gemma-4-sub", GemmaSubPromptStrategy())

    out = t.translate_batch_sync(["Hello"], "en", "vi")  # 1 câu ⇒ đi Pipeline A

    assert out == ["Xin chào"]
    _, kwargs = llm.calls[0]
    assert (kwargs["temperature"], kwargs["top_p"], kwargs["top_k"]) == (0.0, 1.0, 1)
    assert kwargs["repeat_penalty"] == 1.0
    assert kwargs["max_tokens"] == 128  # Pipeline A vẫn dùng `cfg.max_tokens`


def test_hook_strategy_ghi_de_yaml_that_su(allow_real_translation_methods, monkeypatch):
    """Hook `batch_generation_params()` phải chảy vào kwargs THẬT gửi llama.cpp."""
    class _CustomSamplingBatch(GemmaSubPromptStrategy):
        def batch_generation_params(self):
            return {"temperature": 0.7, "top_p": 0.9, "top_k": 50}

    llm = _CapturingLlama('{"1": "Xin chào", "2": "Tạm biệt"}')
    t = _wire_engine(monkeypatch, llm, "translate-gemma-4-sub", _CustomSamplingBatch())

    t.translate_batch_sync(["Hello", "Bye"], "en", "vi")

    _, kwargs = llm.calls[0]
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9
    assert kwargs["top_k"] == 50
    assert kwargs["repeat_penalty"] == 1.0  # không khai ⇒ vẫn YAML


def test_cfg_tuong_minh_thang_ca_yaml(allow_real_translation_methods, monkeypatch):
    llm = _CapturingLlama('{"1": "Xin chào", "2": "Tạm biệt"}')
    t = _wire_engine(monkeypatch, llm, "translate-gemma-4-sub", GemmaSubPromptStrategy())
    t.cfg.temperature = 0.4

    t.translate_batch_sync(["Hello", "Bye"], "en", "vi")

    _, kwargs = llm.calls[0]
    assert kwargs["temperature"] == 0.4
    assert kwargs["top_k"] == 1  # các field không đặt vẫn lấy YAML
    assert kwargs["top_p"] == 1.0


@pytest.mark.parametrize("strategy_style", ["index", "gemma-sub"])
def test_khong_bao_gio_tra_ve_gia_tri_none(strategy_style, allow_real_translation_methods, monkeypatch):
    """`None` lọt vào kwargs của llama.cpp sẽ nổ ⇒ mọi thông số phải luôn có giá trị."""
    registry = TranslationModelRegistry.get_instance()
    info = registry.get_model("index-translate-9b")
    llm = _CapturingLlama('{"1": "a", "2": "b"}')
    t = _wire_engine(monkeypatch, llm, "index-translate-9b", get_prompt_strategy(strategy_style))

    t.translate_batch_sync(["a", "b"], "en", "vi")

    _, kwargs = llm.calls[0]
    for field in ("temperature", "top_p", "top_k", "repeat_penalty", "max_tokens"):
        assert kwargs[field] is not None
    assert resolve_generation_params(None, info)["temperature"] is not None
