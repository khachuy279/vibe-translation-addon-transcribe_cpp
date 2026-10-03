"""Native Python ctypes bindings cho omnivoice.cpp (C99 ABI).

Cung cấp:
- Các struct C: OVInitParams, OVTTSParams, OVAudio, OVVoiceRef.
- Class OmniVoiceBinding bao bọc DLL native `omnivoice.dll` trong `backend/bin/omnivoice/`.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

from backend.config import BACKEND_DIR
from backend.utils.logger import logger

_TAG = "TTS"

# Hằng số ABI từ omnivoice.h
OV_ABI_VERSION = 3

# ov_status enum
OV_STATUS_OK = 0
OV_STATUS_INVALID_PARAMS = -1
OV_STATUS_INSTRUCT_INVALID = -2
OV_STATUS_GENERATE_FAILED = -3
OV_STATUS_OOM = -4
OV_STATUS_CANCELLED = -5


class OVInitParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("model_path", ctypes.c_char_p),
        ("codec_path", ctypes.c_char_p),
        ("use_fa", ctypes.c_bool),
        ("clamp_fp16", ctypes.c_bool),
    ]


class OVAudio(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]


class OVVoiceRef(ctypes.Structure):
    _fields_ = [
        ("ref_codes", ctypes.POINTER(ctypes.c_int32)),
        ("ref_T", ctypes.c_int),
        ("num_codebooks", ctypes.c_int),
    ]


class OVTTSParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("text", ctypes.c_char_p),
        ("lang", ctypes.c_char_p),
        ("instruct", ctypes.c_char_p),
        ("T_override", ctypes.c_int),
        ("chunk_duration_sec", ctypes.c_float),
        ("chunk_threshold_sec", ctypes.c_float),
        ("denoise", ctypes.c_bool),
        ("preprocess_prompt", ctypes.c_bool),
        ("mg_num_step", ctypes.c_int),
        ("mg_guidance_scale", ctypes.c_float),
        ("mg_t_shift", ctypes.c_float),
        ("mg_layer_penalty_factor", ctypes.c_float),
        ("mg_position_temperature", ctypes.c_float),
        ("mg_class_temperature", ctypes.c_float),
        ("mg_seed", ctypes.c_uint64),
        ("ref_audio_tokens", ctypes.POINTER(ctypes.c_int32)),
        ("ref_T", ctypes.c_int),
        ("ref_audio_24k", ctypes.POINTER(ctypes.c_float)),
        ("ref_n_samples", ctypes.c_int),
        ("ref_text", ctypes.c_char_p),
        ("dump_dir", ctypes.c_char_p),
        ("cancel", ctypes.c_void_p),
        ("cancel_user_data", ctypes.c_void_p),
        ("on_chunk", ctypes.c_void_p),
        ("on_chunk_user_data", ctypes.c_void_p),
        ("postproc", ctypes.c_bool),
    ]


class OmniVoiceNative:
    """Wrapper bậc cao quản lý handle ov_context của omnivoice.cpp."""

    def __init__(self, dll_dir: Optional[Path] = None):
        self.dll_dir = Path(dll_dir or (BACKEND_DIR / "bin" / "omnivoice")).resolve()
        self._lib = self._load_library()
        self._ctx: Optional[ctypes.c_void_p] = None
        self._cached_voice_refs: Dict[str, OVVoiceRef] = {}

    def _load_library(self) -> ctypes.CDLL:
        dll_path = self.dll_dir / "omnivoice.dll"
        if not dll_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy omnivoice.dll tại: {dll_path}")

        if sys.platform == "win32":
            try:
                os.add_dll_directory(str(self.dll_dir))
            except Exception as e:
                logger.debug(f"add_dll_directory notice: {e}", extra={"module_tag": _TAG})

        lib = ctypes.CDLL(str(dll_path))

        # Khai báo kiểu dữ liệu các hàm C ABI
        lib.ov_version.argtypes = []
        lib.ov_version.restype = ctypes.c_char_p

        lib.ov_last_error.argtypes = []
        lib.ov_last_error.restype = ctypes.c_char_p

        lib.ov_init_default_params.argtypes = [ctypes.POINTER(OVInitParams)]
        lib.ov_init_default_params.restype = None

        lib.ov_init.argtypes = [ctypes.POINTER(OVInitParams)]
        lib.ov_init.restype = ctypes.c_void_p

        lib.ov_free.argtypes = [ctypes.c_void_p]
        lib.ov_free.restype = None

        lib.ov_tts_default_params.argtypes = [ctypes.POINTER(OVTTSParams)]
        lib.ov_tts_default_params.restype = None

        lib.ov_synthesize.argtypes = [ctypes.c_void_p, ctypes.POINTER(OVTTSParams), ctypes.POINTER(OVAudio)]
        lib.ov_synthesize.restype = ctypes.c_int

        lib.ov_audio_free.argtypes = [ctypes.POINTER(OVAudio)]
        lib.ov_audio_free.restype = None

        lib.ov_extract_voice_ref.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(OVVoiceRef),
        ]
        lib.ov_extract_voice_ref.restype = ctypes.c_int

        lib.ov_voice_ref_free.argtypes = [ctypes.POINTER(OVVoiceRef)]
        lib.ov_voice_ref_free.restype = None

        version_str = lib.ov_version()
        ver = version_str.decode("utf-8", errors="ignore") if version_str else "unknown"
        logger.info(f"Đã nạp omnivoice.dll native C ABI (phiên bản: {ver})", extra={"module_tag": _TAG})
        return lib

    @property
    def is_loaded(self) -> bool:
        return self._ctx is not None

    def get_last_error(self) -> str:
        err = self._lib.ov_last_error()
        return err.decode("utf-8", errors="ignore") if err else ""

    def load_model(
        self,
        model_path: str,
        codec_path: str,
        use_fa: bool = True,
        clamp_fp16: bool = False,
    ) -> None:
        """Khởi tạo mô hình qua ov_init."""
        if self._ctx is not None:
            self.unload_model()

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Không tìm thấy model base GGUF: {model_path}")
        if not os.path.isfile(codec_path):
            raise FileNotFoundError(f"Không tìm thấy model tokenizer GGUF: {codec_path}")

        params = OVInitParams()
        self._lib.ov_init_default_params(ctypes.byref(params))
        params.abi_version = OV_ABI_VERSION
        params.model_path = model_path.encode("utf-8")
        params.codec_path = codec_path.encode("utf-8")
        params.use_fa = use_fa
        params.clamp_fp16 = clamp_fp16

        ctx = self._lib.ov_init(ctypes.byref(params))
        if not ctx:
            err = self.get_last_error()
            raise RuntimeError(f"ov_init thất bại: {err or 'unknown error'}")

        self._ctx = ctx
        logger.info(f"OmniVoice native model đã nạp thành công", extra={"module_tag": _TAG})

    def extract_voice_ref(self, voice_key: str, audio_24k: np.ndarray) -> bool:
        """Trích xuất và lưu cache mã RVQ từ âm thanh mẫu float32 24kHz."""
        if not self._ctx:
            raise RuntimeError("Model chưa được nạp (ov_context is NULL)")

        # Nếu đã có trong cache thì giải phóng trước
        if voice_key in self._cached_voice_refs:
            old_ref = self._cached_voice_refs.pop(voice_key)
            self._lib.ov_voice_ref_free(ctypes.byref(old_ref))

        audio_mono = np.asarray(audio_24k, dtype=np.float32).ravel()
        n_samples = len(audio_mono)
        if n_samples == 0:
            return False

        audio_ptr = audio_mono.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ref_out = OVVoiceRef()

        status = self._lib.ov_extract_voice_ref(self._ctx, audio_ptr, n_samples, ctypes.byref(ref_out))
        if status != OV_STATUS_OK:
            err = self.get_last_error()
            logger.warning(f"Không thể trích xuất voice_ref cho {voice_key}: status={status}, err={err}", extra={"module_tag": _TAG})
            return False

        self._cached_voice_refs[voice_key] = ref_out
        logger.debug(f"Đã trích xuất & cache voice_ref '{voice_key}': T={ref_out.ref_T}, codebooks={ref_out.num_codebooks}", extra={"module_tag": _TAG})
        return True

    def synthesize(
        self,
        text: str,
        voice_key: Optional[str] = None,
        ref_text: Optional[str] = None,
        raw_ref_audio_24k: Optional[np.ndarray] = None,
        num_steps: int = 16,
        seed: int = 42,
        lang: str = "",
        instruct: str = "",
    ) -> np.ndarray:
        """Tổng hợp giọng nói thành mảng float32 24kHz mono."""
        if not self._ctx:
            raise RuntimeError("Model chưa được nạp (ov_context is NULL)")

        clean_text = (text or "").strip()
        if not clean_text:
            return np.zeros((0,), dtype=np.float32)

        params = OVTTSParams()
        self._lib.ov_tts_default_params(ctypes.byref(params))
        params.abi_version = OV_ABI_VERSION
        params.text = clean_text.encode("utf-8")
        params.lang = (lang or "").encode("utf-8")
        params.instruct = (instruct or "").encode("utf-8")
        params.mg_num_step = max(4, int(num_steps))
        params.mg_seed = int(seed)

        ref_text_bytes = (ref_text or "").encode("utf-8") if ref_text else None

        # 1. Ưu tiên dùng RVQ codes đã cache
        if voice_key and voice_key in self._cached_voice_refs:
            vref = self._cached_voice_refs[voice_key]
            params.ref_audio_tokens = vref.ref_codes
            params.ref_T = vref.ref_T
            params.ref_text = ref_text_bytes
        # 2. Hoặc dùng âm thanh raw mẫu nếu có
        elif raw_ref_audio_24k is not None and len(raw_ref_audio_24k) > 0:
            audio_mono = np.asarray(raw_ref_audio_24k, dtype=np.float32).ravel()
            params.ref_audio_24k = audio_mono.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            params.ref_n_samples = len(audio_mono)
            params.ref_text = ref_text_bytes

        out_audio = OVAudio()
        status = self._lib.ov_synthesize(self._ctx, ctypes.byref(params), ctypes.byref(out_audio))
        if status != OV_STATUS_OK:
            err = self.get_last_error()
            raise RuntimeError(f"ov_synthesize thất bại (status={status}): {err or 'unknown error'}")

        try:
            if out_audio.n_samples > 0 and out_audio.samples:
                arr = np.ctypeslib.as_array(out_audio.samples, shape=(out_audio.n_samples,)).copy()
                return arr
            return np.zeros((0,), dtype=np.float32)
        finally:
            self._lib.ov_audio_free(ctypes.byref(out_audio))

    def unload_model(self) -> None:
        """Giải phóng hoàn toàn mô hình và các mẫu giọng đã cache."""
        for key, ref in list(self._cached_voice_refs.items()):
            try:
                self._lib.ov_voice_ref_free(ctypes.byref(ref))
            except Exception:
                pass
        self._cached_voice_refs.clear()

        if self._ctx:
            try:
                self._lib.ov_free(self._ctx)
            except Exception as e:
                logger.debug(f"ov_free notice: {e}", extra={"module_tag": _TAG})
            self._ctx = None

        logger.info("OmniVoice native context đã được giải phóng", extra={"module_tag": _TAG})
