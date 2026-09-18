"""Engine tổng hợp giọng nói Voice Cloning OmniVoice PyTorch Native cho Backend.

Đặc điểm tối ưu:
- Singleton Pattern với Thread-safe double-checked locking.
- Tự động nạp mô hình từ HuggingFace cache cục bộ hoặc tải tự động.
- Caching VoiceClonePrompt: Tiết kiệm ~70ms tiền xử lý âm thanh mẫu cho mỗi câu.
- Cơ chế tự động xử lý cuDNN mismatch (cuDNN auto-fallback) để đảm bảo không bị lỗi crash.
- Thực thi hoàn toàn trong `torch.inference_mode()` loại bỏ overhead tính gradient.
- Đo đạc chi tiết RTF, thời gian xử lý (ms) và thời lượng audio sinh ra (s).
"""

import asyncio
import contextlib
import gc
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import numpy as np
import torch

from backend.config import config, MODELS_DIR
from backend.tts.base import BaseTTSEngine
from backend.tts.audio_processor import AudioProcessor
from backend.tts.voice_manager import VoiceManager
from backend.core.gpu_scheduler import gpu_arbiter, PRIORITY_TTS
from backend.utils.logger import get_logger

logger = get_logger("tts.omnivoice")


class OmniVoiceTTS(BaseTTSEngine):
    """Singleton TTS Engine bao bọc PyTorch OmniVoice phục vụ tổng hợp giọng nói độ trễ thấp."""

    _instance: Optional["OmniVoiceTTS"] = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "OmniVoiceTTS":
        """Truy cập Singleton instance thread-safe."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = OmniVoiceTTS()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Giải phóng hoàn toàn Singleton instance khi shutdown."""
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance.unload_model()
                except Exception as e:
                    logger.debug(f"TTS unload_model notice: {e}", extra={"module_tag": "TTS"})
                cls._instance = None

    def __init__(self):
        self.sample_rate = 24000
        self.device = config.tts.device if hasattr(config, "tts") and config.tts.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._is_loaded = False
        self._cudnn_disabled = False
        self._voice_prompt_cache: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()  # LRU, max 8 entries
        self._voice_prompt_cache_max: int = 8
        # QWEN-Q15: `_get_voice_clone_prompt` là check-then-act trên OrderedDict KHÔNG lock.
        # Hai phiên (hai luồng `to_thread`) cùng dùng một giọng mới ⇒ cả hai tính
        # VoiceClonePrompt (giải mã audio + trích embedding: hàng trăm ms CPU/GPU) rồi ghi
        # cache đè nhau — lãng phí, và `move_to_end`/`popitem` chạy song song có thể ném
        # RuntimeError. Lock RIÊNG (không dùng `_infer_lock`) để không chặn inference TTS
        # của phiên khác lâu hơn mức cần.
        self._voice_prompt_lock = threading.Lock()
        # QWEN-Q15 (bổ sung): lock theo TỪNG giọng. Chỉ một lock chung thì hai phiên khác
        # giọng phải xếp hàng vô ích; chỉ double-checked locking thì CÙNG một giọng vẫn bị
        # tính prompt 2 lần (phần chậm nằm ngoài lock). Per-key lock cho cả hai tính chất.
        self._voice_prompt_key_locks: "Dict[Tuple[str, str], threading.Lock]" = {}
        self._voice_prompt_key_locks_guard = threading.Lock()
        self._init_lock = threading.RLock()
        self._infer_lock = threading.RLock()
        # A2-2 (Hy3): CUDA stream RIÊNG cho TTS thay vì đồng bộ toàn device.
        # `torch.cuda.synchronize()` cũ đồng bộ MỌI stream CUDA đang chờ => chặn cả
        # inference dịch (CUDA) đang xếp hàng, gây priority inversion và latency spike.
        # Dùng stream riêng + `stream.synchronize()` chỉ chờ công việc của TTS này.
        self._cuda_stream: Any = None

    def _resolve_model_path(self) -> str:
        """Xác định đường dẫn mô hình (ưu tiên cache cục bộ, fallback HF repo id)."""
        cfg_model = config.tts.model if hasattr(config, "tts") and config.tts.model else "splendor1811/omnivoice-vietnamese"

        # 1. Đường dẫn tệp/thư mục cục bộ rõ ràng nếu tồn tại
        if cfg_model and os.path.exists(cfg_model):
            return cfg_model

        # 2. Kiểm tra trong MODELS_DIR / huggingface
        snap_dirs = [
            MODELS_DIR / "huggingface" / "models--splendor1811--omnivoice-vietnamese" / "snapshots",
        ]
        for s_dir in snap_dirs:
            if s_dir.exists():
                for child in s_dir.iterdir():
                    if child.is_dir() and (child / "config.json").exists():
                        return str(child)

        return cfg_model or "splendor1811/omnivoice-vietnamese"

    def is_model_ready(self) -> bool:
        """Kiểm tra xem mô hình có sẵn sàng hay không."""
        p = self._resolve_model_path()
        return bool(p and (os.path.exists(p) or "splendor1811" in p))

    def _invoke_generate(self, gen_kwargs: dict) -> Any:
        """Gọi `model.generate` — trên CUDA stream riêng của TTS nếu đã có.

        A2-2 (Hy3): dùng `with torch.cuda.stream(...)` (API ổn định ở mọi bản PyTorch)
        thay vì dạng decorator `stream(fn)` để tránh phụ thuộc vào `StreamContext.__call__`.
        """
        if self._cuda_stream is not None:
            with torch.cuda.stream(self._cuda_stream):
                return self.model.generate(**gen_kwargs)
        return self.model.generate(**gen_kwargs)

    def _run_generate(self, gen_kwargs: dict) -> Any:
        """Thực thi model.generate trong inference_mode và xử lý cuDNN an toàn.

        A2-2 (Hy3): chạy trên CUDA stream RIÊNG của TTS (`self._cuda_stream`) thay vì
        stream mặc định, để sau này chỉ cần `stream.synchronize()` (chờ riêng TTS) thay
        vì `torch.cuda.synchronize()` (chờ TOÀN device, làm tắc nghẽn inference dịch CUDA
        khác đang chạy => priority inversion / latency spike).
        """
        with torch.inference_mode():
            if self._cudnn_disabled:
                with torch.backends.cudnn.flags(enabled=False):
                    return self._invoke_generate(gen_kwargs)

            try:
                return self._invoke_generate(gen_kwargs)
            except RuntimeError as e:
                if "CUDNN" in str(e):
                    logger.warning(
                        f"Phát hiện xung đột cuDNN ({e}). "
                            f"Tự động tắt cuDNN cho TTS để duy trì độ ổn định tối đa.", extra={"module_tag": "TTS"}
                    )
                    self._cudnn_disabled = True
                    with torch.backends.cudnn.flags(enabled=False):
                        return self._invoke_generate(gen_kwargs)
                raise

    def _lock_for_voice(self, cache_key: Tuple[str, str]) -> threading.Lock:
        """Lock riêng cho một `cache_key` giọng (QWEN-Q15). Số entry bị chặn trên."""
        with self._voice_prompt_key_locks_guard:
            lock = self._voice_prompt_key_locks.get(cache_key)
            if lock is None:
                # Giữ dict có trần: số giọng thật rất nhỏ, nhưng không để nó phình vô hạn.
                if len(self._voice_prompt_key_locks) >= 32:
                    self._voice_prompt_key_locks = {
                        k: v for k, v in self._voice_prompt_key_locks.items()
                        if k in self._voice_prompt_cache
                    } or {cache_key: threading.Lock()}
                lock = self._voice_prompt_key_locks.setdefault(cache_key, threading.Lock())
            return lock

    def _get_voice_clone_prompt(self, ref_audio_path: str, ref_text: str) -> Optional[Any]:
        """Tạo hoặc lấy từ cache VoiceClonePrompt tái sử dụng cho mẫu giọng tham chiếu.

        Tiết kiệm ~70ms cho mỗi câu nói tiếp theo do không phải lặp lại Disk I/O,
        giải mã âm thanh và trích xuất embedding.

        QWEN-Q15: check-then-act phải được bảo vệ. Dùng **lock theo từng giọng**
        (`_lock_for_voice`) chứ không phải một lock chung: hai phiên khác giọng vẫn tính
        song song, còn hai phiên CÙNG giọng mới thì chỉ tính prompt MỘT lần (mỗi lần tính
        tốn hàng trăm ms: giải mã audio + trích embedding).
        """
        if self.model is None or not hasattr(self.model, "create_voice_clone_prompt"):
            return None

        cache_key = (ref_audio_path, ref_text)
        with self._lock_for_voice(cache_key):
            with self._voice_prompt_lock:
                if cache_key in self._voice_prompt_cache:
                    # LRU: promote to most-recently-used position
                    self._voice_prompt_cache.move_to_end(cache_key)
                    return self._voice_prompt_cache[cache_key]

            try:
                prompt = self.model.create_voice_clone_prompt(ref_audio=ref_audio_path, ref_text=ref_text)
            except Exception as e:
                logger.debug(f"Không tạo được cache VoiceClonePrompt ({e}), dùng thẳng ref_audio.", extra={"module_tag": "TTS"})
                return None

            with self._voice_prompt_lock:
                self._voice_prompt_cache[cache_key] = prompt
                # LRU eviction: xoa entry cu nhat neu vuot maxsize
                if len(self._voice_prompt_cache) > self._voice_prompt_cache_max:
                    evicted_key, _ = self._voice_prompt_cache.popitem(last=False)
                    logger.debug(f"Đã xoá cache giọng cũ: {Path(evicted_key).name}", extra={"module_tag": "TTS"})
                logger.debug(f"Đã lưu cache VoiceClonePrompt cho: {Path(ref_audio_path).name} (cache={len(self._voice_prompt_cache)})", extra={"module_tag": "TTS"})
            return prompt

    def load_model(self) -> None:
        """Nạp và warm-up mô hình OmniVoice vào GPU memory."""
        with self._init_lock:
            if self._is_loaded and self.model is not None:
                return

            try:
                from omnivoice import OmniVoice
            except ImportError:
                logger.error("Chưa cài đặt thư viện 'omnivoice'.", extra={"module_tag": "TTS"})
                raise RuntimeError("Thư viện omnivoice chưa được cài đặt")

            model_path = self._resolve_model_path()
            logger.info(f"OmniVoice đang nạp trên thiết bị {self.device}...", extra={"module_tag": "TTS"})
            t0 = time.perf_counter()

            torch_dtype = torch.float16 if self.device != "cpu" and "cuda" in str(self.device) else torch.float32
            if torch.cuda.is_available() and "cuda" in str(self.device):
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass

            self.model = OmniVoice.from_pretrained(
                model_path,
                device_map=self.device if self.device != "cpu" else None,
                dtype=torch_dtype,
            )

            # Warmup với đoạn văn bản ngắn và tiền nạp cache VoiceClonePrompt
            try:
                ref_audio_path, ref_text = VoiceManager.resolve_voice(config.tts.default_voice)
                if os.path.exists(ref_audio_path):
                    cached_prompt = self._get_voice_clone_prompt(ref_audio_path, ref_text)
                    if cached_prompt is not None:
                        warmup_kwargs = {
                            "text": "Sẵn sàng.",
                            "voice_clone_prompt": cached_prompt,
                            "num_step": 4,
                        }
                    else:
                        warmup_kwargs = {
                            "text": "Sẵn sàng.",
                            "ref_audio": ref_audio_path,
                            "ref_text": ref_text,
                            "num_step": 4,
                        }
                    if torch.cuda.is_available() and "cuda" in str(self.device):
                        # A2-2 (Hy3): tạo CUDA stream RIÊNG cho TTS một lần duy nhất.
                        # Đồng bộ MỘT LẦN ở đây (đường nạp model, không phải hot path) để
                        # mọi công việc nạp trọng số trên stream mặc định đã hoàn tất trước
                        # khi TTS bắt đầu đọc trọng số trên stream mới. Nếu bỏ bước này, lần
                        # sinh đầu tiên có thể đọc trọng số chưa ghi xong (race cross-stream).
                        torch.cuda.synchronize()
                        self._cuda_stream = torch.cuda.Stream()
                    with self._infer_lock:
                        _ = self._run_generate(warmup_kwargs)
                        if self._cuda_stream is not None:
                            self._cuda_stream.synchronize()
            except Exception as e:
                logger.debug(f"Warmup notice: {e}", extra={"module_tag": "TTS"})

            # Thu hồi toàn bộ bộ nhớ trung gian tạm thời trong quá trình nạp trọng số & trích xuất prompt
            # Giảm Reserved VRAM từ ~6.0GB xuống ~2.0GB (tiết kiệm ~4.0GB VRAM)
            if torch.cuda.is_available() and "cuda" in str(self.device):
                torch.cuda.empty_cache()

            self._is_loaded = True
            elapsed = time.perf_counter() - t0
            logger.info(f"OmniVoice đã sẵn sàng ({elapsed:.2f}s)", extra={"module_tag": "TTS"})

    async def prewarm(self) -> bool:
        """Khởi động và nạp sẵn mô hình trong tiến trình nền."""
        if not self._is_loaded:
            await asyncio.to_thread(self.load_model)
        return self._is_loaded

    def unload_model(self) -> None:
        """Giải phóng hoàn toàn mô hình khỏi RAM và VRAM GPU.

        A2-3 (Hy3): khi `tts_enabled` chuyển sang False, caller (main.py / handler.py)
        NÊN gọi hàm này để trả VRAM vài GB (OmniVoice float16) thay vì giữ vô ích.
        """
        with self._init_lock:
            with self._infer_lock:
                self._is_loaded = False

                # Giải phóng LRU cache của transformers (anti-pattern @lru_cache trên class method
                # _get_conv1d_layers giữ chặt instance model và 527 weight tensors khiến rò rỉ ~800MB allocated / ~2GB reserved VRAM)
                try:
                    from transformers.models.higgs_audio_v2_tokenizer.modeling_higgs_audio_v2_tokenizer import (
                        HiggsAudioV2TokenizerPreTrainedModel,
                    )
                    if hasattr(HiggsAudioV2TokenizerPreTrainedModel._get_conv1d_layers, "cache_clear"):
                        HiggsAudioV2TokenizerPreTrainedModel._get_conv1d_layers.cache_clear()
                except Exception:
                    pass

                if self.model is not None:
                    try:
                        if hasattr(self.model, "audio_tokenizer"):
                            self.model.audio_tokenizer = None
                        del self.model
                    except Exception:
                        pass
                self.model = None
                self._voice_prompt_cache.clear()
                self._cuda_stream = None
                # B9-4 (Hy3): CHỈ giữ `gc.collect()` ở đường unload. Module PyTorch tạo
                # vòng tham chiếu (module ↔ parameter ↔ hook) nên nếu không ép GC thì
                # `empty_cache()` bên dưới không đòi lại được VRAM — tức là làm hỏng chính
                # mục tiêu của A2-3. Đây là đường RẤT hiếm (tắt TTS / đổi model) nên chi phí
                # GC không nằm trên hot path; `gc.collect()` ở warmup (hot path khởi động)
                # đã được bỏ.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    with contextlib.suppress(Exception):
                        torch.cuda.ipc_collect()
        logger.info("OmniVoice đã giải phóng (VRAM cleared)", extra={"module_tag": "TTS"})

    def _synthesize_audio(self, text: str, voice_id: Optional[str], speed: float) -> Tuple[Optional[np.ndarray], float]:
        """Sinh audio float32 (bỏ qua bước mã hóa). Trả (audio, duration_sec)."""
        if not text or not text.strip():
            return None, 0.0

        if not self._is_loaded:
            self.load_model()

        ref_audio_path, ref_text = VoiceManager.resolve_voice(voice_id or config.tts.default_voice)
        if not os.path.exists(ref_audio_path):
            logger.warning(f"Không tìm thấy file mẫu giọng: {ref_audio_path}", extra={"module_tag": "TTS"})
            return None, 0.0

        clean_text = text.strip()
        voice_name = Path(ref_audio_path).name

        # Dùng VoiceClonePrompt đã được cache nếu khả dụng
        cached_prompt = self._get_voice_clone_prompt(ref_audio_path, ref_text)
        num_steps = max(4, int(getattr(config.tts, "num_inference_steps", 8)))
        if cached_prompt is not None:
            gen_kwargs = {
                "text": clean_text,
                "voice_clone_prompt": cached_prompt,
                "num_step": num_steps,
            }
        else:
            gen_kwargs = {
                "text": clean_text,
                "ref_audio": ref_audio_path,
                "ref_text": ref_text,
                "num_step": num_steps,
            }

        t_start = time.perf_counter()
        with self._infer_lock:
            audio_output = self._run_generate(gen_kwargs)
            # A2-2 (Hy3): chỉ chờ stream RIÊNG của TTS, KHÔNG `torch.cuda.synchronize()`
            # (cái cũ đồng bộ TOÀN device => chặn inference dịch CUDA khác đang chạy =>
            # priority inversion / latency spike). Nếu chưa có stream riêng (CPU/trường
            # hợp lỗi) thì fall back về synchronize toàn device để vẫn đúng kết quả.
            if self._cuda_stream is not None:
                self._cuda_stream.synchronize()
            elif torch.cuda.is_available() and "cuda" in str(self.device):
                torch.cuda.synchronize()

        infer_time = time.perf_counter() - t_start

        # 1. Chuyển đổi sang mảng 1D float32 numpy
        audio_np = AudioProcessor.convert_to_numpy(audio_output)
        del audio_output

        # 2. Điều chỉnh tốc độ (Time-stretching) nếu cần
        effective_speed = float(speed if speed is not None else getattr(config.tts, "speed", 1.0) or 1.0)
        if abs(effective_speed - 1.0) >= 0.02 and len(audio_np) > 0:
            audio_np = AudioProcessor.apply_time_stretch(
                audio_np, speed=effective_speed, sample_rate=self.sample_rate
            )

        # 3. Chuẩn hóa đỉnh biên độ và âm lượng
        vol = float(getattr(config.tts, "volume", 1.0) or 1.0)
        audio_np = AudioProcessor.normalize_audio(audio_np, volume=vol, target_peak=0.95)

        duration_sec = len(audio_np) / self.sample_rate if self.sample_rate > 0 else 0.0
        elapsed_ms = int(infer_time * 1000)
        rtf = (infer_time / duration_sec) if duration_sec > 0 else 0.0
        logger.info(
            f"Synth {elapsed_ms}ms | audio {duration_sec:.2f}s | RTF {rtf:.2f} | voice={voice_name}: '{clean_text}'",
            extra={"module_tag": "TTS"},
        )
        return audio_np, duration_sec

    def synthesize_sync(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Tổng hợp giọng nói đồng bộ, trả về (audio_base64_wav, duration_sec)."""
        audio_np, duration_sec = self._synthesize_audio(text, voice_id, speed)
        if audio_np is None:
            return None, 0.0
        return AudioProcessor.encode_wav_to_base64(audio_np, self.sample_rate), duration_sec

    def synthesize_wav_bytes(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[bytes], float]:
        """P3.1: như `synthesize_sync` nhưng trả WAV thô (bytes).

        Dùng cho đường gửi binary frame: bỏ base64 (+33% kích thước) và bỏ vòng lặp
        per-byte `atob` trên main thread của client.
        """
        audio_np, duration_sec = self._synthesize_audio(text, voice_id, speed)
        if audio_np is None:
            return None, 0.0
        return AudioProcessor.encode_wav_bytes(audio_np, self.sample_rate), duration_sec

    async def synthesize_clone_bytes(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[bytes], float]:
        """Bản async của `synthesize_wav_bytes` (không block event loop).

        A2-1: nhường GPU nếu commit ASR đang chạy (no-op khi scheduler tắt).
        """
        await gpu_arbiter.admit(PRIORITY_TTS)
        return await asyncio.to_thread(self.synthesize_wav_bytes, text, voice_id, speed)

    async def synthesize_clone(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Tổng hợp giọng nói bất đồng bộ không làm block asyncio event loop.

        A2-1: nhường GPU nếu commit ASR đang chạy (no-op khi scheduler tắt).
        """
        await gpu_arbiter.admit(PRIORITY_TTS)
        return await asyncio.to_thread(self.synthesize_sync, text, voice_id, speed)
