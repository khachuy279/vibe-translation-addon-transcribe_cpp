"""Engine FSMN-VAD (Alibaba FunASR Streaming Industrial VAD)."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch

from backend.config import config, MODELS_DIR
from backend.vad.base import BaseVADEngine, VADResult, VADStreamState
from backend.utils.logger import logger


class FsmnVADEngine(BaseVADEngine):
    """Engine FSMN-VAD của Alibaba DAMO Academy."""

    name = "fsmn-vad"
    default_threshold = config.vad.fsmn.speech_noise_thres or config.vad.threshold
    native_frame_samples = 960  # 60ms @ 16kHz

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self.model_dir = self._resolve_model_dir(model_dir)
        self._ensure_model_files()

        import logging as _logging
        _logging.getLogger("funasr").setLevel(_logging.ERROR)
        _logging.getLogger("modelscope").setLevel(_logging.ERROR)

        from funasr import AutoModel
        self.model = AutoModel(
            model=str(self.model_dir),
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )

        cfg = config.vad.fsmn
        self.model.model.vad_opts.output_frame_probs = True
        self.model.model.vad_opts.speech_noise_thres = float(cfg.speech_noise_thres or config.vad.threshold)
        self.model.model.vad_opts.max_end_silence_time = int(cfg.max_end_silence_time)
        self.model.model.vad_opts.speech_to_sil_time_thres = int(cfg.speech_to_sil_time_thres)
        self.model.model.vad_opts.sil_to_speech_time_thres = int(cfg.sil_to_speech_time_thres)

        logger.info("Model FSMN sẵn sàng", extra={"module_tag": "VAD"})

    def _resolve_model_dir(self, explicit_dir: Optional[Union[str, Path]]) -> Path:
        if explicit_dir and Path(explicit_dir).exists():
            return Path(explicit_dir)
        return MODELS_DIR / "fsmn_vad"

    def _ensure_model_files(self) -> None:
        """Đảm bảo các file model FSMN tồn tại."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        required = ["model.pt", "am.mvn", "config.yaml", "configuration.json"]
        if not all((self.model_dir / f).exists() for f in required):
            from huggingface_hub import snapshot_download
            logger.info("Đang tải model FSMN-VAD từ HuggingFace...", extra={"module_tag": "VAD"})
            snapshot_download("funasr/fsmn-vad", local_dir=str(self.model_dir))

    def create_initial_state(self, threshold: Optional[float] = None) -> VADStreamState:
        cfg = config.vad.fsmn
        active_thresh = threshold if threshold is not None else (cfg.speech_noise_thres or config.vad.threshold)
        state = VADStreamState()
        state.fsmn_cache = {}
        self.model.model.init_cache(
            state.fsmn_cache,
            speech_noise_thres=float(active_thresh),
            max_end_silence_time=int(cfg.max_end_silence_time),
            speech_to_sil_time_thres=int(cfg.speech_to_sil_time_thres),
            sil_to_speech_time_thres=int(cfg.sil_to_speech_time_thres),
        )
        state.fsmn_in_speech = False
        return state

    def is_speech(
        self,
        chunk_float32: Optional[np.ndarray],
        state: VADStreamState,
        threshold: float,
        chunk_raw: Optional[bytes] = None,
    ) -> VADResult:
        if not state.fsmn_cache:
            init_s = self.create_initial_state(threshold)
            state.fsmn_cache = init_s.fsmn_cache

        if threshold is not None and "stats" in state.fsmn_cache:
            stats = state.fsmn_cache["stats"]
            if getattr(stats, "speech_noise_thres", None) != threshold:
                stats.speech_noise_thres = float(threshold)

        if chunk_float32 is None:
            if chunk_raw is not None:
                chunk_float32 = np.frombuffer(chunk_raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                raise ValueError("Cần cung cấp ít nhất chunk_raw hoặc chunk_float32 cho FSMN VAD")

        if len(chunk_float32) != self.native_frame_samples:
            if len(chunk_float32) < self.native_frame_samples:
                chunk_float32 = np.pad(chunk_float32, (0, self.native_frame_samples - len(chunk_float32)))
            else:
                chunk_float32 = chunk_float32[: self.native_frame_samples]

        tensor_chunk = torch.from_numpy(chunk_float32).float()

        res = self.model.generate(
            input=[tensor_chunk],
            cache=state.fsmn_cache,
            is_final=False,
            chunk_size=60,
            dynamic_silence=False,
            disable_pbar=True,
            disable_log=True,
        )

        signals = res[0].get("value", []) if res else []
        event = None

        for sig in signals:
            if sig[0] >= 0 and sig[1] == -1:
                event = "START"
                state.fsmn_in_speech = True
            elif sig[0] == -1 and sig[1] >= 0:
                event = "END"
                state.fsmn_in_speech = False
            elif sig[0] >= 0 and sig[1] >= 0:
                event = "START"
                state.fsmn_in_speech = False

        stats = state.fsmn_cache.get("stats")
        sil_cnt = getattr(stats, "continous_silence_frame_count", 0) if stats else 0
        is_speech_frame = bool(state.fsmn_in_speech and sil_cnt == 0)

        prob = 0.9 if is_speech_frame else 0.1
        if stats is not None and getattr(stats, "frame_probs", None):
            last_fp = stats.frame_probs[-1]
            if hasattr(last_fp, "score"):
                prob = float(last_fp.score)

        return VADResult(is_speech=is_speech_frame, probability=prob, event=event)
