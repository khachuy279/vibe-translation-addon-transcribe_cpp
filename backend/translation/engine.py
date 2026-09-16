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
import logging
import os
import threading
import time
from typing import Any, AsyncIterator, Dict, Optional

from backend.utils.cuda import setup_cuda_dll_paths
from backend.utils.logger import logger

# Bắt buộc thiết lập đường dẫn CUDA DLL trước khi nạp llama_cpp
setup_cuda_dll_paths()

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from backend.config import config, TranslationConfig
from backend.translation.base import BaseTranslator
from backend.translation.registry import TranslationModelRegistry
from backend.translation.prompts import get_prompt_strategy
from backend.utils.model_download import ensure_model_file

_TRANS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translation_worker")


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
            f"Đang nạp mô hình dịch GGUF '{key}' từ: {gguf_path}",
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

        logger.info(
            f"Nạp thành công mô hình dịch '{key}' trên GPU "
                f"(n_ctx={n_ctx}, n_batch={n_batch}, n_threads={n_threads})",
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
        """Giải phóng một model dịch (chờ lượt suy luận đang chạy trên nó kết thúc trước)."""
        if llm is None:
            return
        with cls._infer_lock:
            pass  # hàng rào: không đóng model khi còn thread đang sinh token trên nó
        cls._close_llm_quietly(llm)
        logger.info("Đã giải phóng mô hình dịch khỏi GPU VRAM.", extra={"module_tag": "TRANSLATE"})

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

            # Có thể raise (thiếu file & tắt auto_download, lỗi tải, hết VRAM...) — model cũ
            # vẫn đang phục vụ bình thường vì ta chưa chạm tới nó.
            new_llm, prompt_strategy = self._build_llm(new_key, new_cfg, allow_download)

            with self.__class__._shared_lock:
                old_llm = self.__class__._shared_llm
                self.__class__._shared_llm = new_llm
                self.__class__._shared_model_key = new_key
                self.canonical_key = new_key
                self.cfg = new_cfg
                self.prompt_strategy = prompt_strategy
            self._load_failure_logged = False
            self.__class__._release_llm(old_llm)

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
            self.load_model()
            self._translate_sync("Hello", source_lang="en", target_lang="vi")
            logger.info(f"Pre-warm hoàn tất cho Translation model '{self.canonical_key}'", extra={"module_tag": "TRANSLATE"})
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

        llm = self.__class__._shared_llm
        if llm is None:
            return {"translated_text": text, "elapsed_ms": 0.0}

        info = self.registry.get_model(self.canonical_key) or {}
        temperature = self.cfg.temperature if self.cfg.temperature is not None else info.get("temperature", 0.7)
        top_p = self.cfg.top_p if self.cfg.top_p is not None else info.get("top_p", 0.6)
        top_k = self.cfg.top_k if self.cfg.top_k is not None else info.get("top_k", 20)
        repetition_penalty = self.cfg.repetition_penalty if self.cfg.repetition_penalty is not None else info.get("repetition_penalty", 1.05)

        prompt = self.prompt_strategy.build_prompt(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
            use_context=self.cfg.use_context,
        )
        stop_tokens = self.prompt_strategy.get_stop_tokens()

        with self.__class__._infer_lock:
            t0 = time.perf_counter()
            output = llm(
                prompt,
                max_tokens=self.cfg.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stop=stop_tokens,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            raw_text = output["choices"][0]["text"].strip()
            # Dọn dẹp khoảng trắng
            clean_out = raw_text.replace("<|im_end|>", "").strip()

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
        """Async wrapper gọi hàm dịch trên thread pool."""
        loop = asyncio.get_running_loop()
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
        llm = self.__class__._shared_llm
        if llm is None:
            yield text
            return

        info = self.registry.get_model(self.canonical_key) or {}
        temperature = self.cfg.temperature if self.cfg.temperature is not None else info.get("temperature", 0.7)
        top_p = self.cfg.top_p if self.cfg.top_p is not None else info.get("top_p", 0.6)
        top_k = self.cfg.top_k if self.cfg.top_k is not None else info.get("top_k", 20)
        repetition_penalty = (
            self.cfg.repetition_penalty if self.cfg.repetition_penalty is not None
            else info.get("repetition_penalty", 1.05)
        )
        prompt = self.prompt_strategy.build_prompt(
            text=text, source_lang=source_lang, target_lang=target_lang,
            context=context, use_context=self.cfg.use_context,
        )
        stop_tokens = self.prompt_strategy.get_stop_tokens()

        with self.__class__._infer_lock:
            acc = ""
            for chunk in llm(
                prompt,
                max_tokens=self.cfg.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stop=stop_tokens,
                stream=True,
            ):
                try:
                    piece = chunk["choices"][0].get("text", "")
                except (KeyError, IndexError, TypeError):
                    continue
                if not piece:
                    continue
                acc += piece
                yield acc
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
        """
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


