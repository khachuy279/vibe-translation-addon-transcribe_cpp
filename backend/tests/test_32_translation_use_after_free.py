"""QWEN-Q1 (P0) — race `use-after-free` khi hot-swap model dịch.

**Lỗi gốc:** `_translate_sync` / `_translate_stream_sync` đọc con trỏ
`cls._shared_llm` **NGOÀI** `_infer_lock`, build prompt (hàng trăm µs → ms), rồi mới
xin lock. Trong cửa sổ đó `reconfigure()` / `load_model()` có thể:
  1. build model mới,
  2. swap `_shared_llm = mới` (dưới `_shared_lock`),
  3. `_release_llm(cũ)` → `close()` model cũ.
Thread đang xếp hàng chờ lock vẫn giữ con trỏ CŨ ⇒ gọi llama.cpp trên context đã
`close()` ⇒ **segfault cả tiến trình** (không phải exception Python).

**Vì sao hàng rào cũ không cứu được:** `_release_llm` chỉ `with cls._infer_lock: pass`
— nó chặn được thread CHƯA vào vùng suy luận, nhưng không chặn được thread đã ĐỌC con
trỏ rồi mới xếp hàng.

**Cách sửa:** đọc lại con trỏ **BÊN TRONG** `_infer_lock` (`_snapshot_infer_state`), và
lấy `_shared_lock` trong đó để ảnh chụp (llm, key, cfg, prompt_strategy) không bị nửa
cũ nửa mới. Vì `_release_llm` chỉ `close()` sau khi đã đi qua `_infer_lock`, con trỏ đọc
trong lock **chắc chắn còn sống**.

Các test dưới đây tái hiện race một cách **tất định** (không dựa vào may rủi của thread
scheduler) bằng cách tự tay giữ `_infer_lock` rồi `close()` model cũ — đúng thứ tự mà
`_release_llm` thực hiện.
"""

import ast
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from backend.translation.engine import GGUFTranslator

ROOT = Path(__file__).resolve().parent.parent.parent
ENGINE_SRC = ROOT / "backend" / "translation" / "engine.py"


# ─────────────── tiện ích AST: kiểm CODE, không kiểm docstring/comment ───────────────

def _ast_unparse(func_name: str) -> str:
    """Mã nguồn của một method trong `engine.py`, CHUẨN HOÁ (bỏ docstring/comment)."""
    tree = ast.parse(ENGINE_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                        and isinstance(stmt.value.value, str):
                    node.body.remove(stmt)      # bỏ docstring
                    break
            return ast.unparse(node)
    raise AssertionError(f"không tìm thấy hàm {func_name} trong engine.py")


class _UseAfterFree(RuntimeError):
    """Tương đương `segfault` trong thế giới Python: gọi model đã bị `close()`."""


class _FakeLlama:
    """Model GGUF giả: đếm số lần bị GỌI và NỔ nếu bị gọi sau `close()`.

    `invocations` được tăng TRƯỚC khi kiểm `closed` — nếu tăng sau, ta không phân biệt
    được "không hề gọi" với "gọi rồi nổ", và test sẽ xanh giả.
    """

    def __init__(self, name: str):
        self.name = name
        self.closed = False
        self.invocations = 0
        self.calls = 0
        self.prompts = []

    def close(self):
        self.closed = True

    def __call__(self, prompt, **kwargs):
        self.invocations += 1
        if self.closed:
            raise _UseAfterFree(
                f"use-after-free: model '{self.name}' đã bị close() nhưng vẫn bị gọi "
                f"(llama.cpp thật sẽ segfault ở đây)"
            )
        self.calls += 1
        self.prompts.append(prompt)
        if kwargs.get("stream"):
            return iter([{"choices": [{"text": f"[{self.name}]"}]}])
        return {
            "choices": [{"text": f"[{self.name}]"}],
            "usage": {"completion_tokens": 3},
        }


class _FakeStrategy:
    def build_prompt(self, text, source_lang, target_lang, context, use_context):
        return f"<{source_lang}->{target_lang}>{text}"

    def get_stop_tokens(self):
        return ["</s>"]


def _fake_cfg(**over):
    base = dict(
        max_tokens=16,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        use_context=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _prepare(monkeypatch, llm, key="m-old"):
    """Dựng GGUFTranslator cô lập: instance mới, không registry thật, không model thật."""
    monkeypatch.setattr(GGUFTranslator, "_instance", None)
    translator = GGUFTranslator.get_instance()
    monkeypatch.setattr(GGUFTranslator, "_shared_llm", llm)
    monkeypatch.setattr(GGUFTranslator, "_shared_model_key", key)
    monkeypatch.setattr(translator.registry, "get_model", lambda k: {})
    translator.canonical_key = key
    translator.cfg = _fake_cfg()
    translator.prompt_strategy = _FakeStrategy()
    translator._load_failure_logged = False
    return translator


def _swap_under_shared_lock(translator, new_llm, new_key):
    """Đúng phần swap mà `reconfigure()`/`load_model()` làm (dưới `_shared_lock`)."""
    with GGUFTranslator._shared_lock:
        GGUFTranslator._shared_llm = new_llm
        GGUFTranslator._shared_model_key = new_key
        translator.canonical_key = new_key


# ─────────────────────────────────────────────── _translate_sync

def test_translate_sync_khong_goi_model_da_close_khi_swap_giua_luc_cho_lock(
    allow_real_translation_methods, monkeypatch
):
    """Q1: thread đã đọc con trỏ CŨ ngoài lock rồi chờ lock KHÔNG được gọi model cũ."""
    old = _FakeLlama("old")
    new = _FakeLlama("new")
    translator = _prepare(monkeypatch, llm=old, key="m-old")

    outcome = {}

    def _run():
        try:
            outcome["out"] = translator._translate_sync("xin chào", "vi", "en")
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    # Giữ `_infer_lock` để mô phỏng một lượt suy luận khác đang chạy trên model CŨ.
    GGUFTranslator._infer_lock.acquire()
    t = threading.Thread(target=_run, daemon=True)
    try:
        t.start()
        time.sleep(0.2)  # để thread đi qua bước đọc con trỏ rồi block ở lock

        # Trong lúc nó đang chờ: đổi model + close() model CŨ (đúng thứ tự `_release_llm`).
        _swap_under_shared_lock(translator, new, "m-new")
        old.close()
    finally:
        GGUFTranslator._infer_lock.release()

    t.join(timeout=5.0)
    assert not t.is_alive(), "thread dịch không kết thúc"
    assert "error" not in outcome, (
        f"use-after-free: gọi model đã close() ⇒ {outcome.get('error')!r}. "
        "Phải đọc lại `_shared_llm` BÊN TRONG `_infer_lock`."
    )
    assert old.invocations == 0, "đã gọi model CŨ sau khi nó bị close() ⇒ segfault thật"
    assert new.calls == 1, "phải chạy trên model MỚI sau khi đọc lại con trỏ trong lock"


def test_translate_sync_tra_nguyen_van_khi_model_bi_unload_trong_cua_so_race(
    allow_real_translation_methods, monkeypatch
):
    """Nếu model bị unload trong cửa sổ race ⇒ trả nguyên văn, KHÔNG gọi con trỏ đã chết."""
    old = _FakeLlama("old")
    translator = _prepare(monkeypatch, llm=old, key="m-old")

    outcome = {}

    def _run():
        try:
            outcome["out"] = translator._translate_sync("xin chào", "vi", "en")
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    GGUFTranslator._infer_lock.acquire()
    t = threading.Thread(target=_run, daemon=True)
    try:
        t.start()
        time.sleep(0.2)
        with GGUFTranslator._shared_lock:
            GGUFTranslator._shared_llm = None
            GGUFTranslator._shared_model_key = None
        old.close()
    finally:
        GGUFTranslator._infer_lock.release()

    t.join(timeout=5.0)
    assert not t.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert outcome["out"]["translated_text"] == "xin chào", "phải trả nguyên văn bản gốc"
    assert old.invocations == 0, "không được gọi model đã bị unload/close"


# ─────────────────────────────────────────────── _translate_stream_sync

def test_translate_stream_sync_khong_goi_model_da_close_khi_swap_giua_luc_cho_lock(
    allow_real_translation_methods, monkeypatch
):
    """Q1 (đường streaming): generator cũng phải đọc lại con trỏ trong `_infer_lock`."""
    old = _FakeLlama("old")
    new = _FakeLlama("new")
    translator = _prepare(monkeypatch, llm=old, key="m-old")

    outcome = {}

    def _run():
        try:
            outcome["chunks"] = list(
                translator._translate_stream_sync("xin chào", "vi", "en", "")
            )
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    GGUFTranslator._infer_lock.acquire()
    t = threading.Thread(target=_run, daemon=True)
    try:
        t.start()
        time.sleep(0.2)
        _swap_under_shared_lock(translator, new, "m-new")
        old.close()
    finally:
        GGUFTranslator._infer_lock.release()

    t.join(timeout=5.0)
    assert not t.is_alive()
    assert "error" not in outcome, (
        f"use-after-free ở đường streaming ⇒ {outcome.get('error')!r}"
    )
    assert old.invocations == 0, "generator đã gọi model CŨ sau khi close()"
    assert new.calls == 1, "generator phải chạy trên model MỚI"
    assert outcome["chunks"] == ["[new]"]


# ─────────────────────────────────────────────── chốt ở mức mã nguồn

def test_hai_duong_suy_luan_doc_lai_con_tro_trong_infer_lock():
    """Chốt hồi quy: cả hai đường phải snapshot TRONG `_infer_lock`.

    Bản cũ dùng biến `llm` đọc ngoài lock để gọi model — đúng cái phải biến mất.
    """
    src = ENGINE_SRC.read_text(encoding="utf-8")

    for func in ("def _translate_sync(", "def _translate_stream_sync("):
        start = src.index(func)
        end = src.find("\n    async def ", start + 1)
        body = src[start:end if end != -1 else len(src)]

        lock_idx = body.index("with self.__class__._infer_lock:")
        # `body.index` từ lock_idx: phần docstring/comment phía trên cũng nhắc tên hàm này.
        snap_idx = body.index("_snapshot_infer_state()", lock_idx)
        assert snap_idx > lock_idx, f"{func}: phải snapshot SAU khi đã giữ `_infer_lock`"

        pre_lock = body[:lock_idx]
        assert "llm = self.__class__._shared_llm" not in pre_lock, (
            f"{func}: vẫn còn đọc `llm = cls._shared_llm` TRƯỚC lock — cửa sổ race Q1"
        )
        assert "llm = self.__class__._shared_llm" not in body, (
            f"{func}: biến `llm` phải lấy từ `_snapshot_infer_state()`, không đọc trực tiếp"
        )
        assert "llm, shared_key, cfg, strategy = self._snapshot_infer_state()" in body, (
            f"{func}: phải dùng ảnh chụp đầy đủ (llm, key, cfg, strategy)"
        )
        assert "llm(prompt" in body, f"{func}: không còn gọi model?"


def test_snapshot_infer_state_lay_shared_lock_de_anh_chup_nhat_quan():
    """Ảnh chụp phải gồm cả key/cfg/strategy, và phải lấy `_shared_lock`.

    Nếu chỉ đọc `_shared_llm` mà không lấy `_shared_lock`, ta có thể thấy con trỏ MỚI
    nhưng `self.prompt_strategy`/`self.cfg` CŨ (`reconfigure` ghi con trỏ trước) ⇒ prompt
    của model cũ gửi cho model mới.
    """
    src = ENGINE_SRC.read_text(encoding="utf-8")
    start = src.index("def _snapshot_infer_state(")
    end = src.find("\n    def ", start + 1)
    body = src[start:end]

    assert "with cls._shared_lock:" in body, "phải lấy `_shared_lock` để ảnh chụp nhất quán"
    assert "cls._shared_llm" in body
    assert "cls._shared_model_key" in body
    assert "self.cfg" in body
    assert "self.prompt_strategy" in body


def test_build_prompt_dung_anh_chup_key_cfg_strategy_nhat_quan():
    """`(key, cfg, strategy)` dùng để build prompt phải lấy trong CÙNG `_shared_lock`.

    Đọc rời `self.canonical_key` rồi `self.cfg` ngoài lock có thể ghép key model MỚI với
    cfg/prompt CŨ (`reconfigure` ghi 5 field trong cùng một `_shared_lock`) ⇒ prompt của
    model cũ gửi cho model mới.
    """
    body = _ast_unparse("_shared_config_snapshot")
    assert "with cls._shared_lock" in body
    assert "cls._shared_model_key" in body
    assert "self.cfg" in body
    assert "self.prompt_strategy" in body

    for func in ("_translate_sync", "_translate_stream_sync"):
        fsrc = _ast_unparse(func)
        assert "_shared_config_snapshot()" in fsrc, (
            f"{func}: phải lấy ảnh chụp (key, cfg, strategy) nhất quán trước khi build prompt"
        )


def test_khong_deadlock_khi_long_infer_lock_roi_shared_lock():
    """Thứ tự `_infer_lock` → `_shared_lock` phải chạy được (RLock, không nghịch đảo).

    `load_model`/`reconfigure` nhả `_shared_lock` TRƯỚC khi chờ `_infer_lock`
    (qua `_release_llm`), nên không có chu trình chờ.
    """
    done = threading.Event()

    def _nested():
        with GGUFTranslator._infer_lock:
            with GGUFTranslator._shared_lock:
                pass
        done.set()

    t = threading.Thread(target=_nested, daemon=True)
    t.start()
    assert done.wait(timeout=2.0), "lồng `_infer_lock` → `_shared_lock` bị treo"
    t.join(timeout=2.0)
    assert not t.is_alive()
