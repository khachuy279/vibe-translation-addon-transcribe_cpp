"""Engine tổng hợp giọng nói Voice Cloning OmniVoice Native C++ (omnivoice.cpp GGUF).

Đặc điểm kiến trúc:
- Singleton Pattern với Thread-safe double-checked locking.
- GGUF Native Inference: Chạy 100% C++ GGML native qua Process Isolation Worker,
  hoàn toàn loại bỏ phụ thuộc PyTorch runtime, giải phóng ~4GB VRAM.
- Không xung đột DLL: Toàn bộ module native chạy trong process riêng, độc lập
  với `backend/bin/ggml.dll` của ASR.
- Caching Voice Reference: Trích xuất mã RVQ 1 lần duy nhất, tái sử dụng cho các câu sau.
- Tự động nạp/tải mô hình GGUF từ `Serveurperso/OmniVoice-GGUF` về duy nhất `backend/models/`.
- Bảo toàn trọn vẹn hợp đồng API: `BaseTTSEngine`, `synthesize_fitted_sync` (WSOLA lookahead),
  `synthesize_wav_bytes`, `prewarm()`, `unload_model()`.
"""

import asyncio
import gc
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import numpy as np
import soundfile as sf

from backend.config import config, MODELS_DIR
from backend.tts.base import BaseTTSEngine
from backend.tts.audio_processor import AudioProcessor
from backend.tts.voice_manager import VoiceManager
from backend.tts.worker import TTSWorkerClient
from backend.tts.downloader import (
    ensure_tts_models,
    get_model_paths,
)
from backend.core.gpu_scheduler import gpu_arbiter, PRIORITY_TTS
from backend.utils.logger import get_logger

logger = get_logger("tts.omnivoice")
_TAG = "TTS"


class OmniVoiceTTS(BaseTTSEngine):
    """Singleton TTS Engine bao bọc omnivoice.cpp phục vụ tổng hợp giọng nói độ trễ thấp."""

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
                    logger.debug(f"TTS unload_model notice: {e}", extra={"module_tag": _TAG})
                cls._instance = None

    def __init__(self):
        self.sample_rate = 24000
        self.model = None  # Giữ cho mock testing hoặc legacy fallback
        self._is_loaded = False
        self._voice_prompt_cache: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()  # LRU, max 8 entries
        self._voice_prompt_cache_max: int = 8
        self._voice_prompt_lock = threading.Lock()
        self._voice_prompt_key_locks: "Dict[Tuple[str, str], threading.Lock]" = {}
        self._voice_prompt_key_locks_guard = threading.Lock()
        self._init_lock = threading.RLock()
        self._infer_lock = threading.RLock()
        self._worker_client = TTSWorkerClient.get_instance()

    def _resolve_model_path(self) -> str:
        """Xác định đường dẫn mô hình base GGUF."""
        base_path, _ = get_model_paths()
        if base_path.is_file():
            return str(base_path)
        return str(base_path)

    def _lock_for_voice(self, cache_key: Tuple[str, str]) -> threading.Lock:
        """Lock riêng cho một cache_key giọng (QWEN-Q15). Số entry bị chặn trên."""
        with self._voice_prompt_key_locks_guard:
            lock = self._voice_prompt_key_locks.get(cache_key)
            if lock is None:
                if len(self._voice_prompt_key_locks) >= 32:
                    self._voice_prompt_key_locks = {
                        k: v for k, v in self._voice_prompt_key_locks.items()
                        if k in self._voice_prompt_cache
                    } or {cache_key: threading.Lock()}
                lock = self._voice_prompt_key_locks.setdefault(cache_key, threading.Lock())
            return lock

    def _get_voice_clone_prompt(self, ref_audio_path: str, ref_text: str) -> Optional[Any]:
        """Tạo hoặc lấy từ cache mã voice reference tái sử dụng cho mẫu giọng tham chiếu.

        Tiết kiệm ~70ms cho mỗi câu nói tiếp theo do không phải lặp lại Disk I/O,
        giải mã âm thanh và trích xuất embedding RVQ.
        """
        cache_key = (ref_audio_path, ref_text)
        with self._lock_for_voice(cache_key):
            with self._voice_prompt_lock:
                if cache_key in self._voice_prompt_cache:
                    self._voice_prompt_cache.move_to_end(cache_key)
                    return self._voice_prompt_cache[cache_key]

            # 1. Hỗ trợ mock testing (nếu self.model có create_voice_clone_prompt)
            if self.model is not None and hasattr(self.model, "create_voice_clone_prompt"):
                try:
                    prompt = self.model.create_voice_clone_prompt(ref_audio=ref_audio_path, ref_text=ref_text)
                    with self._voice_prompt_lock:
                        self._voice_prompt_cache[cache_key] = prompt
                        if len(self._voice_prompt_cache) > self._voice_prompt_cache_max:
                            self._voice_prompt_cache.popitem(last=False)
                    return prompt
                except Exception as e:
                    logger.debug(f"Mock create_voice_clone_prompt notice: {e}", extra={"module_tag": _TAG})

            # 2. Xử lý native C++ qua TTS Worker
            prompt = None
            if os.path.isfile(ref_audio_path):
                try:
                    data, sr = sf.read(ref_audio_path, dtype="float32")
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if sr != self.sample_rate and len(data) > 0:
                        # Đổi tần số mẫu sang 24kHz nếu cần
                        num_target_samples = int(len(data) * self.sample_rate / sr)
                        from scipy import signal
                        data = signal.resample(data, num_target_samples).astype(np.float32)

                    # Trích xuất mã RVQ vào worker cache
                    voice_key = str(Path(ref_audio_path).resolve())
                    self._worker_client.extract_voice(voice_key, data)
                    prompt = {"voice_key": voice_key, "ref_text": ref_text}
                except Exception as exc:
                    logger.warning(f"Lỗi nạp mẫu giọng '{ref_audio_path}': {exc}", extra={"module_tag": _TAG})
                    prompt = {"voice_key": str(Path(ref_audio_path).resolve()), "ref_text": ref_text}

            with self._voice_prompt_lock:
                self._voice_prompt_cache[cache_key] = prompt
                if len(self._voice_prompt_cache) > self._voice_prompt_cache_max:
                    self._voice_prompt_cache.popitem(last=False)
                logger.debug(
                    f"Đã lưu cache VoiceReference cho: {Path(ref_audio_path).name} (cache={len(self._voice_prompt_cache)})",
                    extra={"module_tag": _TAG},
                )
            return prompt

    def load_model(self) -> None:
        """Nạp và warm-up mô hình OmniVoice GGUF Native."""
        with self._init_lock:
            if self._is_loaded and self._worker_client.is_loaded:
                return

            logger.info("OmniVoice GGUF đang nạp qua Process Worker...", extra={"module_tag": _TAG})
            t0 = time.perf_counter()

            # Đảm bảo model đã có trong backend/models (tự tải nếu cấu hình auto_download=True)
            auto_dl = getattr(config.tts, "auto_download", True)
            base_path, tok_path = ensure_tts_models(allow_download=auto_dl)

            use_fa = True
            clamp_fp16 = False
            self._worker_client.load_model(str(base_path), str(tok_path), use_fa=use_fa, clamp_fp16=clamp_fp16)

            # Warmup với đoạn văn bản ngắn và tiền nạp cache VoiceReference
            try:
                ref_audio_path, ref_text = VoiceManager.resolve_voice(config.tts.default_voice)
                if os.path.exists(ref_audio_path):
                    cached_prompt = self._get_voice_clone_prompt(ref_audio_path, ref_text)
                    voice_key = (cached_prompt or {}).get("voice_key") if isinstance(cached_prompt, dict) else None
                    with self._infer_lock:
                        _ = self._worker_client.synthesize(
                            text="Sẵn sàng.",
                            voice_key=voice_key,
                            ref_text=ref_text,
                            num_steps=4,
                        )
            except Exception as e:
                logger.debug(f"Warmup notice: {e}", extra={"module_tag": _TAG})

            self._is_loaded = True
            elapsed = time.perf_counter() - t0
            logger.info(f"OmniVoice GGUF đã sẵn sàng ({elapsed:.2f}s)", extra={"module_tag": _TAG})

    async def prewarm(self) -> bool:
        """Khởi động và nạp sẵn mô hình trong tiến trình nền."""
        if not self._is_loaded:
            await asyncio.to_thread(self.load_model)
        return self._is_loaded

    def unload_model(self) -> None:
        """Giải phóng hoàn toàn mô hình khỏi RAM và VRAM GPU."""
        with self._init_lock:
            with self._infer_lock:
                self._is_loaded = False
                self._voice_prompt_cache.clear()

                if self.model is not None:
                    try:
                        del self.model
                    except Exception:
                        pass
                    self.model = None

                self._worker_client.unload_model()
                gc.collect()

        logger.info("OmniVoice GGUF đã giải phóng (VRAM cleared)", extra={"module_tag": _TAG})

    def _synthesize_audio(self, text: str, voice_id: Optional[str], speed: float) -> Tuple[Optional[np.ndarray], float]:
        """Sinh audio float32 (bỏ qua bước mã hóa). Trả (audio, duration_sec)."""
        if not text or not text.strip():
            return None, 0.0

        if not self._is_loaded:
            self.load_model()

        ref_audio_path, ref_text = VoiceManager.resolve_voice(voice_id or config.tts.default_voice)
        if not os.path.exists(ref_audio_path):
            logger.warning(f"Không tìm thấy file mẫu giọng: {ref_audio_path}", extra={"module_tag": _TAG})
            return None, 0.0

        clean_text = text.strip()
        voice_name = Path(ref_audio_path).name

        # Dùng VoiceClonePrompt / VoiceRef đã được cache
        cached_prompt = self._get_voice_clone_prompt(ref_audio_path, ref_text)
        voice_key = (cached_prompt or {}).get("voice_key") if isinstance(cached_prompt, dict) else str(Path(ref_audio_path).resolve())
        num_steps = max(4, int(getattr(config.tts, "num_inference_steps", 16)))

        t_start = time.perf_counter()
        with self._infer_lock:
            # Nếu mock model đang được gán (trong unit test)
            if self.model is not None and hasattr(self.model, "generate"):
                gen_kwargs = {
                    "text": clean_text,
                    "voice_clone_prompt": cached_prompt,
                    "num_step": num_steps,
                }
                audio_output = self.model.generate(**gen_kwargs)
                audio_np = AudioProcessor.convert_to_numpy(audio_output)
            else:
                audio_np = self._worker_client.synthesize(
                    text=clean_text,
                    voice_key=voice_key,
                    ref_text=ref_text,
                    num_steps=num_steps,
                )

        infer_time = time.perf_counter() - t_start

        # 1. Chuyển đổi sang mảng 1D float32 numpy
        audio_np = AudioProcessor.convert_to_numpy(audio_np)

        # 1b. Cắt tỉa khoảng lặng thừa (Silence Trimming) ở đầu và cuối câu
        audio_np = AudioProcessor.trim_silence(audio_np, sample_rate=self.sample_rate)

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
            extra={"module_tag": _TAG},
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
        """Trả WAV thô (bytes). Dùng cho binary frame qua WebSocket."""
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
        """Bản async của `synthesize_wav_bytes` (không block event loop)."""
        await gpu_arbiter.admit(PRIORITY_TTS)
        return await asyncio.to_thread(self.synthesize_wav_bytes, text, voice_id, speed)

    def synthesize_fitted_sync(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        max_duration_sec: float = 0.0,
        max_speed: float = 1.5,
    ) -> Tuple[Optional[bytes], float]:
        """Tổng hợp giọng nói VỪA một ngân sách thời gian (dùng cho lồng tiếng Lookahead Pipeline B)."""
        audio_np, duration_sec = self._synthesize_audio(text, voice_id, speed)
        if audio_np is None or len(audio_np) == 0:
            return None, 0.0

        budget = float(max_duration_sec or 0.0)
        if budget > 0.05 and duration_sec > budget:
            base_speed = max(0.01, float(speed or 1.0))
            allowed = max(1.0, float(max_speed) / base_speed)
            factor = min(duration_sec / budget, allowed)
            if factor > 1.02:
                before = duration_sec
                audio_np = AudioProcessor.apply_time_stretch(
                    audio_np, speed=factor, sample_rate=self.sample_rate
                )
                vol = float(getattr(config.tts, "volume", 1.0) or 1.0)
                audio_np = AudioProcessor.normalize_audio(audio_np, volume=vol, target_peak=0.95)
                duration_sec = len(audio_np) / self.sample_rate if self.sample_rate > 0 else 0.0
                logger.info(
                    f"Nén lồng tiếng {before:.2f}s -> {duration_sec:.2f}s "
                    f"(ngân sách {budget:.2f}s, hệ số {factor:.2f}x WSOLA)",
                    extra={"module_tag": _TAG},
                )
        return AudioProcessor.encode_wav_bytes(audio_np, self.sample_rate), duration_sec

    async def synthesize_fitted_bytes(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        max_duration_sec: float = 0.0,
        max_speed: float = 1.5,
    ) -> Tuple[Optional[bytes], float]:
        """Bản async của `synthesize_fitted_sync` (chạy trên thread, không block event loop)."""
        await gpu_arbiter.admit(PRIORITY_TTS)
        return await asyncio.to_thread(
            self.synthesize_fitted_sync, text, voice_id, speed, max_duration_sec, max_speed
        )

    async def synthesize_clone(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Tổng hợp giọng nói bất đồng bộ không làm block asyncio event loop."""
        await gpu_arbiter.admit(PRIORITY_TTS)
        return await asyncio.to_thread(self.synthesize_sync, text, voice_id, speed)
