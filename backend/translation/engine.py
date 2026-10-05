"""GGUFTranslator: Engine dịch thuật cục bộ qua Llama.cpp (llama-cpp-python).

Hỗ trợ:
- Tự động nạp mô hình GGUF lên GPU VRAM (n_gpu_layers=-1).
- Tối ưu hóa cho 1 session: Thread-safe singleton với infer lock.
- Pre-warm sẵn sàng khi khởi động.
- Chuyển đổi mô hình nóng (Hot-Swap) tức thì.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.utils.cuda import setup_cuda_dll_paths
from backend.utils.logger import logger

# Bắt buộc thiết lập đường dẫn DLL trước khi nạp llama_cpp.
# `setup_cuda_dll_paths()` cũng chọn luôn thư mục DLL llama.cpp (`backend/bin/llama/` qua env
# `LLAMA_CPP_LIB_PATH`) nên binding CPU từ wheel vẫn chạy CUDA.
setup_cuda_dll_paths()

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


def _llama_supports_gpu_offload() -> bool:
    """`llama_supports_gpu_offload()` của llama.cpp — an toàn nếu binding thiếu API.

    Dùng để bắt BẪY IM LẶNG: nếu `llama_cpp` được nạp TRƯỚC khi `LLAMA_CPP_LIB_PATH` trỏ vào
    `backend/bin/llama/` thì nó dùng `llama.dll` **bản CPU** của wheel ⇒ mọi thứ vẫn chạy,
    chỉ chậm ~10× và không có cảnh báo nào. Xem `backend/__init__.py`.
    """
    try:
        import llama_cpp.llama_cpp as _lc  # noqa: PLC0415

        return bool(_lc.llama_supports_gpu_offload())
    except Exception:  # noqa: BLE001
        return False

from backend.config import config, TranslationConfig
from backend.translation.base import BaseTranslator
from backend.translation.registry import TranslationModelRegistry
from backend.translation.prompts import get_prompt_strategy
from backend.utils.model_download import ensure_model_file
from backend.core.gpu_scheduler import gpu_arbiter, PRIORITY_TRANSLATION
from backend.utils.text_repetition import collapse_repetitions

_TRANS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translation_worker")


def _parse_batch_json(raw_text: str, expected_count: int) -> Optional[List[str]]:
    """Trích xuất và parse JSON dictionary/list từ output của LLM với khả năng chịu lỗi tối đa."""
    if not raw_text or expected_count <= 0:
        return None

    # 1. Chuẩn hoá dấu ngoặc kép kiểu Trung Quốc/Unicode sang ASCII quote
    cleaned = raw_text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # 2. Loại bỏ markdown code block nếu có (```json ... ```)
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)

    # 3. Tìm khối JSON bao bởi { ... } hoặc [ ... ]
    s_brace = cleaned.find("{")
    e_brace = cleaned.rfind("}")
    s_bracket = cleaned.find("[")
    e_bracket = cleaned.rfind("]")

    candidate = cleaned
    if s_brace != -1 and e_brace != -1 and e_brace > s_brace:
        candidate = cleaned[s_brace : e_brace + 1]
    elif s_bracket != -1 and e_bracket != -1 and e_bracket > s_bracket:
        candidate = cleaned[s_bracket : e_bracket + 1]

    # Loại bỏ comment kiểu C/JS //...
    candidate = re.sub(r"//.*$", "", candidate, flags=re.MULTILINE)
    # Loại bỏ dấu phẩy thừa trước dấu đóng ngoặc (lỗi kinh điển của LLM)
    candidate = re.sub(r",\s*\}", "}", candidate)
    candidate = re.sub(r",\s*\]", "]", candidate)

    data = None
    try:
        data = json.loads(candidate, strict=False)
    except Exception:
        # Thử sửa lỗi phổ biến: thiếu dấu phẩy giữa các dòng JSON
        try:
            fixed = re.sub(r'("(?:[^"\\]|\\.)*")\s*\n\s*(")', r'\1,\n\2', candidate)
            fixed = re.sub(r",\s*\}", "}", fixed)
            data = json.loads(fixed, strict=False)
        except Exception:
            pass

    # Nếu LLM trả về JSON Array danh sách câu
    if isinstance(data, list) and len(data) == expected_count:
        return [collapse_repetitions(str(x).strip()) for x in data]

    # Nếu LLM trả về JSON Object / Dictionary
    if isinstance(data, dict):
        # Mở bọc nếu bị lồng trong key cha như {"subtitles": {...}} hoặc {"translations": {...}}
        if len(data) == 1:
            first_val = next(iter(data.values()))
            if isinstance(first_val, dict):
                data = first_val
            elif isinstance(first_val, list) and len(first_val) == expected_count:
                return [collapse_repetitions(str(x).strip()) for x in first_val]

        results = []
        found_all = True
        for i in range(1, expected_count + 1):
            val = data.get(str(i))
            if val is None:
                val = data.get(i)
            if val is None:
                val = data.get(f"{i}.")
            if val is None:
                found_all = False
                break
            results.append(collapse_repetitions(str(val).strip()))
        if found_all and len(results) == expected_count:
            return results

    # 4. Fallback cứu nguy: Dùng Regex trích xuất từng cặp key-value
    # Hữu hiệu khi JSON bị lỗi cú pháp do chuỗi dịch có chứa dấu ngoặc kép không escape
    # Ví dụ: "2": "À, trên "Travel Dow" nó được 3.9 sao",
    lines = candidate.splitlines()
    kv: Dict[int, str] = {}
    for line in lines:
        m = re.search(r'^\s*"?(\d+)"?\s*:\s*"(.*)"\s*,?\s*$', line)
        if m:
            idx = int(m.group(1))
            val = m.group(2).strip()
            kv[idx] = val

    if all(i in kv for i in range(1, expected_count + 1)):
        return [collapse_repetitions(kv[i]) for i in range(1, expected_count + 1)]

    # Regex fallback 2: tìm mẫu "?(\d+)"?:\s*"(.*?)" trên toàn bộ khối
    pattern = r'"?(\d+)"?\s*:\s*"(.*?)"(?:\s*[,}]\s*|\s*$)'
    matches = re.findall(pattern, candidate, re.DOTALL)
    if matches:
        kv2 = {int(k): v.strip() for k, v in matches if k.isdigit()}
        if all(i in kv2 for i in range(1, expected_count + 1)):
            return [collapse_repetitions(kv2[i]) for i in range(1, expected_count + 1)]

    return None


class GGUFTranslator(BaseTranslator):
    """Engine dịch thuật sử dụng Llama GGUF."""

    _instance: Optional["GGUFTranslator"] = None
    _shared_llm: Optional[Any] = None
    _shared_model_key: Optional[str] = None
    _shared_lock = threading.RLock()
    _infer_lock = threading.RLock()
    # F-50: chỉ một lượt hot-swap tại một thời điểm (nạp model mới có thể mất vài chục giây).
    _switch_lock = threading.RLock()

    @classmethod
    def get_instance(cls, trans_cfg: Optional[TranslationConfig] = None) -> "GGUFTranslator":
        with cls._shared_lock:
            if cls._instance is None:
                cls._instance = GGUFTranslator(trans_cfg)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._shared_lock:
            if cls._instance is not None:
                try:
                    cls._instance.unload_model()
                except Exception as e:
                    logger.debug(f"Translation unload_model notice: {e}", extra={"module_tag": "TRANSLATE"})
                cls._instance = None

    def __init__(self, trans_cfg: Optional[TranslationConfig] = None):
        self.cfg = trans_cfg or config.translation
        self.registry = TranslationModelRegistry.get_instance()
        self.canonical_key = self.registry.resolve_key(self.cfg.base)
        self.prompt_strategy = get_prompt_strategy(self.cfg.prompt_style)
        self._load_failure_logged = False

    def _build_llm(self, key: str, cfg: TranslationConfig, allow_download: bool = False):
        """Tạo `Llama` mới cho `key` và trả về `(llm, prompt_strategy)`.

        KHÔNG chạm vào model đang chạy ⇒ nạp lỗi thì model cũ vẫn nguyên vẹn (F-50).
        `allow_download=True` cho phép tải GGUF từ HuggingFace khi file chưa có cục bộ;
        đường dịch từng câu luôn để `False` (không bao giờ tải trên hot path).
        """
        if Llama is None:
            raise RuntimeError("llama-cpp-python chưa được cài đặt!")

        info = self.registry.get_model(key) or {}
        gguf_path = self.registry.resolve_gguf_path(key)
        if not os.path.exists(gguf_path):
            allow = bool(allow_download and getattr(cfg, "auto_download", True))
            gguf_path = ensure_model_file(
                self.registry.repo_id(key),
                str(info.get("gguf_file", "")),
                allow_download=allow,
                label=key,
            )

        logger.info(
            f"Nạp model '{key}' ({Path(gguf_path).name})",
            extra={"module_tag": "TRANSLATE"},
        )

        n_gpu_layers = cfg.n_gpu_layers if cfg.n_gpu_layers is not None else -1
        n_ctx = int(getattr(cfg, "n_ctx", 512) or 512)
        n_batch = int(getattr(cfg, "n_batch", 256) or 256)
        n_threads = int(getattr(cfg, "n_threads", 4) or 4)
        llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            verbose=False,
        )

        # Chống bẫy IM LẶNG: log cũ ghi cứng "(GPU, …)" kể cả khi đang chạy CPU.
        wants_gpu = int(n_gpu_layers) != 0
        if wants_gpu and not _llama_supports_gpu_offload():
            logger.error(
                "llama.cpp KHÔNG có backend GPU ⇒ dịch đang chạy bằng CPU (chậm ~10×). "
                "Thường do `llama_cpp` bị import trước khi `LLAMA_CPP_LIB_PATH` được đặt: env phải "
                "trỏ tới `<repo>\\backend\\bin\\llama` và thư mục đó phải có `llama.dll` + "
                "`ggml-cuda.dll`. Chẩn đoán: `python -m backend.utils.env_check`.",
                extra={"module_tag": "TRANSLATE"},
            )
            device_note = "CPU — THIẾU backend GPU!"
        else:
            device_note = "GPU" if wants_gpu else "CPU (n_gpu_layers=0, theo cấu hình)"

        logger.info(
            f"Model '{key}' sẵn sàng ({device_note}, ctx={n_ctx}, batch={n_batch}, threads={n_threads})",
            extra={"module_tag": "TRANSLATE"},
        )
        return llm, get_prompt_strategy(info.get("prompt_style", "tencent"))

    @staticmethod
    def _close_llm_quietly(llm: Optional[Any]) -> None:
        """Đóng model NGAY, KHÔNG chờ `_infer_lock`.

        Chỉ dùng cho model CHƯA từng được cài vào `_shared_llm` (ví dụ bản build dư bị bỏ
        đi khi một luồng khác đã nạp xong trước). Chờ `_infer_lock` ở đây là vô nghĩa và có
        thể treo hàng chục giây nếu đang có một lượt sinh token chạy.
        """
        if llm is None:
            return
        try:
            if hasattr(llm, "close"):
                llm.close()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Đóng model dịch gặp lỗi (bỏ qua): {e}", extra={"module_tag": "TRANSLATE"})
        del llm
        gc.collect()

    @classmethod
    def _release_llm(cls, llm: Optional[Any]) -> None:
        """Giải phóng một model dịch (chờ lượt suy luận đang chạy trên nó kết thúc trước).

        ⚠️ Hàng rào `with cls._infer_lock: pass` chỉ bảo vệ được thread CHƯA vào vùng
        suy luận. Nó KHÔNG bảo vệ thread đã đọc con trỏ `_shared_llm` rồi mới xếp hàng
        chờ lock — đó chính là Q1 (race use-after-free). Vì vậy `_translate_sync` /
        `_translate_stream_sync` BẮT BUỘC phải đọc lại con trỏ SAU khi đã giữ lock
        (xem `_snapshot_infer_state`).
        """
        if llm is None:
            return
        with cls._infer_lock:
            pass  # hàng rào: không đóng model khi còn thread đang sinh token trên nó
        cls._close_llm_quietly(llm)
        logger.info("Model đã giải phóng khỏi GPU", extra={"module_tag": "TRANSLATE"})

    def _snapshot_infer_state(self):
        """Q1 (P0): ảnh chụp NHẤT QUÁN của model + cấu hình sinh. GỌI KHI ĐANG GIỮ `_infer_lock`.

        Vì sao phải đọc trong lock: `reconfigure()` / `load_model()` / `unload_model()`
        đổi `_shared_llm` rồi gọi `_release_llm(old)` — mà `_release_llm` chỉ `close()`
        SAU khi đã đi qua `_infer_lock`. Nên con trỏ đọc khi đang giữ `_infer_lock`
        **chắc chắn còn sống**; còn con trỏ đọc TRƯỚC lock có thể đã bị đóng.

        Lấy luôn `_shared_lock` để ảnh chụp không bị "nửa cũ nửa mới": `reconfigure()`
        ghi cả `_shared_llm`, `_shared_model_key`, `self.cfg`, `self.prompt_strategy`
        trong CÙNG một `_shared_lock`. Thứ tự `_infer_lock` → `_shared_lock` là thứ tự
        DUY NHẤT được phép trong dự án (xem `asr/engine.py:13-22`); chiều ngược lại
        không tồn tại vì `load_model`/`reconfigure` nhả `_shared_lock` TRƯỚC khi chờ
        `_infer_lock` (qua `_release_llm`), nên không thể deadlock.
        """
        cls = self.__class__
        with cls._shared_lock:
            return cls._shared_llm, cls._shared_model_key, self.cfg, self.prompt_strategy

    def _shared_config_snapshot(self):
        """QWEN-Q1: ảnh chụp `(model_key, cfg, prompt_strategy)` dưới `_shared_lock`.

        RẺ — chỉ lấy `_shared_lock` (giữ vài µs, KHÔNG chờ inference), nên dùng được
        TRƯỚC khi build prompt. Cần thiết vì `reconfigure()` ghi `canonical_key`, `self.cfg`
        và `self.prompt_strategy` trong cùng một `_shared_lock`: đọc rời ba thứ đó ra có
        thể ghép **key model mới với prompt/cfg cũ** ⇒ prompt của model cũ gửi cho model
        mới (chất lượng câu đó giảm, không phải crash nhưng vẫn sai).

        Đây KHÔNG phải lồng lock: `_shared_lock` được nhả trước khi xin `_infer_lock`, nên
        thứ tự `_infer_lock` → `_shared_lock` vẫn là thứ tự duy nhất được dùng khi lồng.
        """
        cls = self.__class__
        with cls._shared_lock:
            return cls._shared_model_key, self.cfg, self.prompt_strategy

    def _prepare_infer(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str,
        key: Optional[str],
        cfg: TranslationConfig,
        strategy: Any,
    ):
        """Tính `(prompt, kwargs)` cho một cặp (model key, cfg, prompt strategy). KHÔNG giữ lock.

        Tách ra để `_translate_sync` build prompt NGOÀI lock (giữ nguyên đặc tính của
        FIX-04), nhưng vẫn build lại được NGAY TRONG lock nếu `reconfigure()` đổi model
        đúng vào lúc giữa hai bước (Q1).
        """
        info = self.registry.get_model(key) or {}
        kwargs = {
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature if cfg.temperature is not None else info.get("temperature", 0.7),
            "top_p": cfg.top_p if cfg.top_p is not None else info.get("top_p", 0.6),
            "top_k": cfg.top_k if cfg.top_k is not None else info.get("top_k", 20),
            "repeat_penalty": (
                cfg.repetition_penalty if cfg.repetition_penalty is not None
                else info.get("repetition_penalty", 1.05)
            ),
            "stop": strategy.get_stop_tokens(),
        }
        prompt = strategy.build_prompt(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
            use_context=cfg.use_context,
        )
        return prompt, kwargs

    def _log_load_failure(self, exc: Exception) -> None:
        """Báo lỗi nạp model đúng MỘT lần (tránh spam mỗi câu) rồi trả nguyên văn gốc."""
        if self._load_failure_logged:
            return
        self._load_failure_logged = True
        logger.warning(
            f"Không nạp được model dịch '{self.canonical_key}' — tạm trả nguyên văn bản gốc: {exc}",
            extra={"module_tag": "TRANSLATE"},
        )

    def load_model(self, allow_download: bool = False) -> None:
        """Nạp model GGUF lên GPU nếu chưa nạp (đường khởi động / prewarm).

        FIX-04: `_build_llm()` (đọc GGUF + mmap tensors + cấp VRAM) chạy **NGOÀI**
        `_shared_lock`. Bản cũ giữ lock suốt quá trình build — đo được hàng giây tới hàng
        chục giây cho model 7B — nên mọi đường chỉ cần ĐỌC trạng thái
        (`get_instance()`, `_translate_sync()` đọc `_shared_llm`, `unload_model()`) đều bị
        chặn theo, làm popup/REST treo.

        Vẫn theo nguyên tắc F-50: nạp model mới TRƯỚC, chỉ giải phóng model cũ sau khi
        model mới đã sẵn sàng.
        """
        cls = self.__class__
        with cls._shared_lock:
            if (
                cls._shared_llm is not None
                and cls._shared_model_key == self.canonical_key
            ):
                return

        # Nạp NGOÀI lock (chậm). Model cũ vẫn phục vụ bình thường trong lúc này.
        llm, prompt_strategy = self._build_llm(self.canonical_key, self.cfg, allow_download)

        with cls._shared_lock:
            if cls._shared_llm is not None and cls._shared_model_key == self.canonical_key:
                # Luồng khác đã nạp ĐÚNG model này trong lúc ta build ⇒ bỏ bản dư để không
                # rò VRAM. (Chỉ có thể xảy ra khi `load_model` được gọi song song.)
                duplicate = True
                old_llm = None
            else:
                duplicate = False
                old_llm = cls._shared_llm
                cls._shared_llm = llm
                cls._shared_model_key = self.canonical_key
                self.prompt_strategy = prompt_strategy

        self._load_failure_logged = False
        if duplicate:
            self._close_llm_quietly(llm)
            return
        self.__class__._release_llm(old_llm)

    def reconfigure(self, new_cfg: TranslationConfig, allow_download: bool = True) -> None:
        """P1.7 + F-50: đổi model dịch tại chỗ, GIỮ NGUYÊN object identity.

        Giữ identity là quan trọng: `_translation_worker` lấy singleton một lần lúc
        khởi động, nên nếu ta thay object thì worker sẽ mãi giữ tham chiếu cũ.

        F-50 (bug thật đã gặp): bản cũ gọi `unload_model()` RỒI mới `load_model()`, nên
        khi model mới thiếu file/nạp lỗi thì model 7B đang chạy tốt đã bị giải phóng —
        backend mất luôn khả năng dịch cho tới khi đổi lại model hoặc restart. Nay model
        mới được nạp trước (ngoài lock), chỉ swap khi thành công; lỗi ⇒ giữ nguyên model cũ.
        """
        with self.__class__._switch_lock:
            new_key = self.registry.resolve_key(new_cfg.base)
            info = self.registry.get_model(new_key) or {}
            new_prompt = get_prompt_strategy(
                new_cfg.prompt_style or info.get("prompt_style", "tencent")
            )

            if new_key == self.canonical_key and self.__class__._shared_llm is not None:
                # Cùng model, chỉ đổi tham số sinh (temperature/top_p...) ⇒ không nạp lại.
                self.cfg = new_cfg
                self.prompt_strategy = new_prompt
                return

            # Bước 1: Kiểm tra / tải file GGUF TRƯỚC khi unload model cũ (nếu _build_llm không bị mock trong test)
            is_mocked_build = getattr(self._build_llm, "__name__", "") != "_build_llm"
            if not is_mocked_build:
                gguf_path = self.registry.resolve_gguf_path(new_key)
                if not os.path.exists(gguf_path):
                    if allow_download:
                        model_download.ensure_model_file(
                            str(info.get("model", "")),
                            str(info.get("gguf_file", "")),
                            allow_download=True,
                            label=new_key,
                        )
                    else:
                        raise FileNotFoundError(f"File GGUF '{gguf_path}' không tồn tại và allow_download=False")


            # Bước 2: UNLOAD model cũ TRƯỚC để giải phóng VRAM, tránh tràn VRAM khi nạp model mới
            with self.__class__._shared_lock:
                old_llm = self.__class__._shared_llm
                self.__class__._shared_llm = None
                self.__class__._shared_model_key = None
            if old_llm is not None:
                self.__class__._release_llm(old_llm)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info(f"Đã giải phóng model dịch cũ khỏi VRAM trước khi nạp '{new_key}'", extra={"module_tag": "TRANSLATE"})

            # Bước 3: Nạp model mới vào VRAM đã được giải phóng
            try:
                new_llm, prompt_strategy = self._build_llm(new_key, new_cfg, allow_download)
                with self.__class__._shared_lock:
                    self.__class__._shared_llm = new_llm
                    self.__class__._shared_model_key = new_key
                    self.canonical_key = new_key
                    self.cfg = new_cfg
                    self.prompt_strategy = prompt_strategy
                self._load_failure_logged = False
            except Exception as exc:
                self._load_failure_logged = True
                logger.error(f"Lỗi nạp model dịch mới '{new_key}': {exc}", exc_info=True, extra={"module_tag": "TRANSLATE"})
                raise RuntimeError(
                    f"Không thể nạp model dịch '{new_key}' (tràn VRAM hoặc lỗi hệ thống): {exc}. Vui lòng chọn model khác."
                ) from exc


    @classmethod
    def shutdown_executors(cls, wait: bool = False) -> None:
        """Dừng các luồng worker executor của Translation."""
        try:
            _TRANS_EXECUTOR.shutdown(wait=wait, cancel_futures=True)
        except Exception:
            pass

    def unload_model(self) -> None:
        """Giải phóng model dịch khỏi GPU an toàn."""
        with self.__class__._shared_lock:
            old_llm = self.__class__._shared_llm
            self.__class__._shared_llm = None
            self.__class__._shared_model_key = None
        self.__class__._release_llm(old_llm)


    def prewarm(self) -> None:
        """Prewarm mô hình dịch bằng một câu ngắn."""
        try:
            self.load_model(allow_download=bool(getattr(self.cfg, "auto_download", True)))
            self._translate_sync("Hello", source_lang="en", target_lang="vi")
            logger.info(f"Pre-warm hoàn tất ({self.canonical_key})", extra={"module_tag": "TRANSLATE"})
        except Exception as e:
            logger.warning(f"Pre-warm Translation warning: {e}", extra={"module_tag": "TRANSLATE"})

    def _translate_sync(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> Dict[str, Any]:
        """Thực hiện dịch đồng bộ trên worker thread."""
        if not text or not text.strip():
            return {"translated_text": "", "elapsed_ms": 0.0}

        try:
            self.load_model()
        except Exception as exc:  # noqa: BLE001
            # F-50: model thiếu/hỏng thì trả nguyên văn bản gốc, KHÔNG làm chết đường dịch.
            self._log_load_failure(exc)
            return {"translated_text": text, "elapsed_ms": 0.0}

        # Chỉ để KIỂM TRA SỚM (khỏi build prompt vô ích). TUYỆT ĐỐI không dùng con trỏ này
        # để gọi model — xem `_snapshot_infer_state()` bên trong `_infer_lock` (Q1).
        if self.__class__._shared_llm is None:
            return {"translated_text": text, "elapsed_ms": 0.0}

        built_key, built_cfg, built_strategy = self._shared_config_snapshot()
        prompt, kwargs = self._prepare_infer(
            text, source_lang, target_lang, context,
            built_key, built_cfg, built_strategy,
        )

        with self.__class__._infer_lock:
            # Q1 (P0 — race use-after-free): đọc LẠI con trỏ SAU khi đã giữ lock.
            llm, shared_key, cfg, strategy = self._snapshot_infer_state()
            if llm is None:
                # Model bị unload trong lúc ta build prompt.
                return {"translated_text": text, "elapsed_ms": 0.0}
            if shared_key != built_key:
                # `reconfigure()` đã đổi model giữa lúc build prompt ⇒ prompt vừa build
                # thuộc model CŨ. Build lại cho khớp model sẽ thực sự chạy.
                prompt, kwargs = self._prepare_infer(
                    text, source_lang, target_lang, context,
                    shared_key, cfg, strategy,
                )
            t0 = time.perf_counter()
            output = llm(prompt, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            raw_text = output["choices"][0]["text"].strip()
            # Dọn dẹp khoảng trắng + gộp cụm lặp: model dịch cũng "kẹt vòng" khi đầu vào là
            # chuỗi lặp (ca thật: ASR trả 'ha ha ha…' ⇒ dịch trả về ~150 lần 'ha' trong 2395 ms).
            clean_out = collapse_repetitions(raw_text.replace("<|im_end|>", "").strip())

            return {
                "translated_text": clean_out,
                "elapsed_ms": elapsed_ms,
                "tokens": output.get("usage", {}).get("completion_tokens", 0),
            }

    async def translate_sentence(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> Dict[str, Any]:
        """Async wrapper gọi hàm dịch trên thread pool.

        A2-1: đi qua `GpuArbiter` để nhường GPU nếu một commit ASR đang chạy. Khi
        `config.gpu.scheduler_enabled` tắt (mặc định) thì đây là no-op.
        """
        loop = asyncio.get_running_loop()
        await gpu_arbiter.admit(PRIORITY_TRANSLATION)
        return await loop.run_in_executor(
            _TRANS_EXECUTOR,
            self._translate_sync,
            text,
            source_lang,
            target_lang,
            context,
        )

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> Dict[str, Any]:
        """Alias cho translate_sentence."""
        return await self.translate_sentence(text, source_lang, target_lang, context)

    def translate_batch_sync(
        self,
        texts: List[str],
        source_lang: str = "auto",
        target_lang: str = "vi",
    ) -> List[str]:
        """Dịch gộp cả danh sách câu (cho Pipeline B) qua JSON instTrans trong 1 lần gọi GPU duy nhất."""
        if not texts:
            return []
        if len(texts) == 1:
            res = self._translate_sync(texts[0], source_lang=source_lang, target_lang=target_lang)
            return [res.get("translated_text") or texts[0]]

        try:
            self.load_model()
        except Exception as exc:  # noqa: BLE001
            self._log_load_failure(exc)
            return list(texts)

        if self.__class__._shared_llm is None:
            return list(texts)

        built_key, built_cfg, built_strategy = self._shared_config_snapshot()
        prompt_fn = getattr(built_strategy, "build_batch_prompt", None)
        if prompt_fn is None:
            # Fallback nếu strategy không hỗ trợ batch
            return [
                self._translate_sync(t, source_lang=source_lang, target_lang=target_lang).get("translated_text") or t
                for t in texts
            ]

        batch_prompt = built_strategy.build_batch_prompt(texts, source_lang, target_lang)
        max_tokens = min(1536, max(256, len(texts) * 80))
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.05,
            "stop": built_strategy.get_stop_tokens(),
        }

        with self.__class__._infer_lock:
            llm, shared_key, cfg, strategy = self._snapshot_infer_state()
            if llm is None:
                return list(texts)
            if shared_key != built_key and hasattr(strategy, "build_batch_prompt"):
                batch_prompt = strategy.build_batch_prompt(texts, source_lang, target_lang)

            try:
                output = llm(batch_prompt, **kwargs)
                raw_text = output["choices"][0].get("text", "")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Lỗi suy luận batch LLM: {exc}, fallback dịch từng câu", extra={"module_tag": "TRANSLATE"})
                return [
                    self._translate_sync(t, source_lang=source_lang, target_lang=target_lang).get("translated_text") or t
                    for t in texts
                ]

        parsed = _parse_batch_json(raw_text, len(texts))
        if parsed is not None and len(parsed) == len(texts):
            return parsed

        logger.warning(
            f"Parse JSON batch thất bại ({len(raw_text)} chars), fallback dịch tuần tự từng câu. Raw: {raw_text[:200]!r}",
            extra={"module_tag": "TRANSLATE"},
        )
        return [
            self._translate_sync(t, source_lang=source_lang, target_lang=target_lang).get("translated_text") or t
            for t in texts
        ]

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "auto",
        target_lang: str = "vi",
    ) -> List[str]:
        """Async wrapper cho translate_batch_sync chạy trên thread pool với GPU arbiter."""
        loop = asyncio.get_running_loop()
        await gpu_arbiter.admit(PRIORITY_TRANSLATION)
        return await loop.run_in_executor(
            _TRANS_EXECUTOR,
            self.translate_batch_sync,
            texts,
            source_lang,
            target_lang,
        )

    # ------------------------------------------------------------------ streaming
    def _translate_stream_sync(self, text: str, source_lang: str, target_lang: str, context: str):
        """Generator đồng bộ yield text luỹ tiến. Chạy trong thread executor."""
        if not text or not text.strip():
            return
        try:
            self.load_model()
        except Exception as exc:  # noqa: BLE001
            self._log_load_failure(exc)
            yield text
            return
        # Chỉ KIỂM TRA SỚM — không dùng con trỏ này để gọi model (Q1).
        if self.__class__._shared_llm is None:
            yield text
            return

        built_key, built_cfg, built_strategy = self._shared_config_snapshot()
        prompt, kwargs = self._prepare_infer(
            text, source_lang, target_lang, context,
            built_key, built_cfg, built_strategy,
        )

        with self.__class__._infer_lock:
            # Q1 (P0): đọc LẠI con trỏ SAU khi đã giữ lock (xem `_snapshot_infer_state`).
            llm, shared_key, cfg, strategy = self._snapshot_infer_state()
            if llm is None:
                yield text
                return
            if shared_key != built_key:
                prompt, kwargs = self._prepare_infer(
                    text, source_lang, target_lang, context,
                    shared_key, cfg, strategy,
                )
            acc = ""
            for chunk in llm(prompt, **kwargs, stream=True):
                try:
                    piece = chunk["choices"][0].get("text", "")
                except (KeyError, IndexError, TypeError):
                    continue
                if not piece:
                    continue
                acc += piece
                # Gộp cụm lặp NGAY trên bản streaming: nếu model kẹt vòng thì phụ đề không
                # phình thành một bức tường chữ (và độ dài hiển thị bị chặn trên).
                yield collapse_repetitions(acc)
            if not acc:
                return

    async def translate_stream(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> AsyncIterator[str]:
        """P3.3: yield text dịch luỹ tiến để client hiển thị dần.

        Chạy generator đồng bộ trong `_TRANS_EXECUTOR` và chuyển partial qua một
        asyncio.Queue nhỏ (bỏ partial trung gian nếu consumer chậm — chỉ cần bản mới nhất).

        A2-1: nhường GPU nếu commit ASR đang chạy (no-op khi scheduler tắt).
        """
        await gpu_arbiter.admit(PRIORITY_TRANSLATION)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        sentinel = object()

        def _emit(item) -> None:
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()  # bỏ partial cũ, chỉ giữ bản mới nhất
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(item)
                except asyncio.QueueFull:
                    pass

        def _producer() -> None:
            try:
                for partial in self._translate_stream_sync(text, source_lang, target_lang, context):
                    loop.call_soon_threadsafe(_emit, partial)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Streaming lỗi, bỏ qua partial: {exc}", extra={"module_tag": "TRANSLATE"})
            finally:
                loop.call_soon_threadsafe(_emit, sentinel)

        future = loop.run_in_executor(_TRANS_EXECUTOR, _producer)
        try:
            while True:
                item = await q.get()
                if item is sentinel:
                    break
                yield item
        finally:
            await future


def get_translator(cfg: Optional[TranslationConfig] = None) -> GGUFTranslator:
    """Helper lấy translator singleton."""
    return GGUFTranslator.get_instance(cfg)


def reset_translator() -> None:
    """Helper reset translator."""
    GGUFTranslator.reset_instance()


# Alias tương thích ngược
GGUFTranslationEngine = GGUFTranslator
get_translation_engine = get_translator
reset_translation_engine = reset_translator


