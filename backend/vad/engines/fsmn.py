"""Engine FSMN-VAD (Alibaba FunASR) — dùng đúng API streaming trong docs.

    from funasr import AutoModel

    model = AutoModel(model="fsmn-vad", disable_update=True)
    cache = {}
    model.model.init_cache(cache, speech_noise_thres=0.6, max_end_silence_time=800, …)
    res = model.generate(input=[chunk], cache=cache, is_final=False,
                         chunk_size=60, dynamic_silence=False, …)

Kết quả `res[0]["value"]` là danh sách tín hiệu theo đồng hồ phiên (ms):
    [start, -1]  → bắt đầu một đoạn nói (start_ms)
    [-1, end]    → kết thúc đoạn nói (end_ms)
    [start, end] → đoạn nói trọn vẹn trong 1 lần gọi

Ghi chú thiết kế:

* `AutoModel` (model + frontend, ~vài chục MB) là **dùng chung**; state của phiên nằm
  hoàn toàn trong `cache` dict ⇒ mỗi phiên một `cache`, đúng mô hình streaming của funasr.
* `init_cache(max_end_silence_time=…)` có ghi ngược vào `model.vad_opts` (hành vi của
  thư viện). Các tham số TĨNH còn lại (`window_size_ms`, `lookback_time_start_point`, …)
  được đặt một lần trong `__init__`, trước khi có phiên nào.
* `probability` chỉ có khi `output_frame_probs=True`; mọi truy cập `frame_probs` đều được
  guard (bản cũ đọc thẳng `frame_probs[-1]` nên nổ `IndexError` khi list rỗng).
"""

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from backend.config import config, MODELS_DIR
from backend.vad.base import (
    EVENT_END,
    EVENT_START,
    BaseVADEngine,
    VADResult,
    VADStreamState,
)
from backend.utils.logger import logger


@dataclass
class _Session:
    """State riêng của FSMN cho 1 phiên: `cache` của thư viện + máy trạng thái nhỏ."""

    cache: Dict[str, Any] = field(default_factory=dict)
    in_speech: bool = False
    frames: int = 0


class FsmnVADEngine(BaseVADEngine):
    """Engine FSMN-VAD của Alibaba DAMO Academy (đường streaming)."""

    name = "fsmn-vad"
    default_threshold = 0.6
    frame_samples = 960  # 60 ms @ 16 kHz (ghi đè theo `chunk_size_ms` trong __init__)

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self.model_dir = self.resolve_model_dir(model_dir)
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
        self.chunk_size_ms = max(1, int(cfg.chunk_size_ms))
        # Bước nhảy của engine = `chunk_size` truyền vào generate() (đơn vị ms → mẫu).
        self.frame_samples = int(16000 * self.chunk_size_ms / 1000)

        # Tham số TĨNH của VADXOptions (đặt 1 lần, trước mọi phiên).
        opts = self.model.model.vad_opts
        opts.output_frame_probs = bool(cfg.output_frame_probs)
        opts.speech_to_sil_time_thres = int(cfg.speech_to_sil_time_thres)
        opts.sil_to_speech_time_thres = int(cfg.sil_to_speech_time_thres)
        opts.window_size_ms = int(cfg.window_size_ms)
        opts.lookback_time_start_point = int(cfg.lookback_time_start_point)
        opts.lookahead_time_end_point = int(cfg.lookahead_time_end_point)

        # Pre-roll: VAD báo START muộn hơn mép đầu đoạn nói (lookback + xác nhận speech),
        # nên vòng đệm phải đủ lớn để xả lại trọn phần đó (+1 frame an toàn).
        self.max_lookback_frames = max(
            1,
            int(ceil((int(cfg.window_size_ms) + int(cfg.sil_to_speech_time_thres)
                      + int(cfg.lookback_time_start_point)) / float(self.chunk_size_ms))) + 1,
        )

        logger.info(
            f"FSMN-VAD sẵn sàng (chunk {self.chunk_size_ms} ms = {self.frame_samples} mẫu, "
            f"model_dir={self.model_dir})",
            extra={"module_tag": "VAD"},
        )

    # ------------------------------------------------------------------ model files
    @staticmethod
    def resolve_model_dir(explicit_dir: Optional[Union[str, Path]] = None) -> Path:
        if explicit_dir and Path(explicit_dir).exists():
            return Path(explicit_dir)
        return MODELS_DIR / "fsmn_vad"

    @classmethod
    def prepare_files(cls) -> None:
        """QWEN-Q2: tải model NGOÀI lock cấp lớp. Xem `BaseVADEngine.prepare_files`."""
        cls._ensure_files_in(cls.resolve_model_dir())

    @staticmethod
    def _ensure_files_in(model_dir: Path) -> None:
        """Đảm bảo các file model FSMN tồn tại."""
        model_dir.mkdir(parents=True, exist_ok=True)
        required = ["model.pt", "am.mvn", "config.yaml", "configuration.json"]
        if not all((model_dir / f).exists() for f in required):
            from huggingface_hub import snapshot_download

            logger.info("Đang tải model FSMN-VAD từ HuggingFace...", extra={"module_tag": "VAD"})
            snapshot_download("funasr/fsmn-vad", local_dir=str(model_dir))

    def _ensure_model_files(self) -> None:
        """Giữ lại cho tương thích: uỷ quyền cho `_ensure_files_in`."""
        self._ensure_files_in(self.model_dir)

    # ------------------------------------------------------------------ config
    def _resolve_effective(self, threshold: Optional[float], silence_ms: Optional[int]):
        """Giá trị đang có hiệu lực: knob phiên (nếu có) hay mặc định config engine."""
        cfg = config.vad.fsmn
        eff_threshold = float(threshold) if threshold is not None else float(cfg.speech_noise_thres)
        eff_silence_ms = int(cfg.max_end_silence_time if silence_ms is None else silence_ms)
        return eff_threshold, eff_silence_ms

    def create_initial_state(
        self,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADStreamState:
        cfg = config.vad.fsmn
        eff_threshold, eff_silence_ms = self._resolve_effective(threshold, silence_ms)

        session = _Session(cache={})
        self.model.model.init_cache(
            session.cache,
            speech_noise_thres=float(eff_threshold),
            max_end_silence_time=int(eff_silence_ms),
            speech_to_sil_time_thres=int(cfg.speech_to_sil_time_thres),
            sil_to_speech_time_thres=int(cfg.sil_to_speech_time_thres),
        )

        state = VADStreamState()
        state.engine_state = session
        return state

    # ------------------------------------------------------------------ streaming
    def is_speech(
        self,
        frame_int16: np.ndarray,
        state: VADStreamState,
        threshold: Optional[float] = None,
        silence_ms: Optional[int] = None,
    ) -> VADResult:
        session: Optional[_Session] = state.engine_state
        if session is None:
            session = self.create_initial_state(threshold, silence_ms).engine_state
            state.engine_state = session

        eff_threshold, eff_silence_ms = self._resolve_effective(threshold, silence_ms)
        self._sync_runtime_config(session, eff_threshold, eff_silence_ms)

        import torch

        # Bản sao float32 [-1, 1]: funasr nhận tensor nguyên trạng (không tự chia 32768).
        tensor = torch.from_numpy(frame_int16.astype(np.float32) / 32768.0)

        res = self.model.generate(
            input=[tensor],
            cache=session.cache,
            is_final=False,
            chunk_size=self.chunk_size_ms,
            dynamic_silence=bool(config.vad.fsmn.dynamic_silence),
            disable_pbar=True,
            disable_log=True,
        )

        session.frames += 1
        signals: List[Any] = res[0].get("value", []) if res else []

        result = VADResult(
            probability=self._last_frame_prob(session),
            is_speech=session.in_speech,
        )
        for sig in signals:
            if not sig or len(sig) < 2:
                continue
            start_ms, end_ms = int(sig[0]), int(sig[1])
            if start_ms >= 0 and end_ms == -1:
                self._on_start(session, start_ms, result)
            elif start_ms == -1 and end_ms >= 0:
                self._on_end(session, result)
            elif start_ms >= 0 and end_ms >= 0:
                # Đoạn trọn vẹn trong 1 lần gọi: đóng đoạn đang mở, hoặc mở đoạn mới.
                if session.in_speech:
                    self._on_end(session, result)
                else:
                    self._on_start(session, start_ms, result)
        return result

    def _on_start(self, session: _Session, start_ms: int, result: VADResult) -> None:
        session.in_speech = True
        now_ms = session.frames * self.chunk_size_ms
        lookback_ms = max(0, now_ms - int(start_ms))
        result.event = EVENT_START
        result.lookback_frames = min(
            self.max_lookback_frames,
            int(ceil(lookback_ms / float(self.chunk_size_ms))),
        )
        result.is_speech = True

    @staticmethod
    def _on_end(session: _Session, result: VADResult) -> None:
        session.in_speech = False
        result.event = EVENT_END
        result.lookback_frames = 0
        result.is_speech = False

    @staticmethod
    def _last_frame_prob(session: _Session) -> float:
        """Xác suất frame cuối — chỉ có khi `output_frame_probs=True`, luôn được guard."""
        stats = session.cache.get("stats") if isinstance(session.cache, dict) else None
        frame_probs = getattr(stats, "frame_probs", None) if stats is not None else None
        if not frame_probs:
            return 0.0
        score = getattr(frame_probs[-1], "score", None)
        try:
            return float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _sync_runtime_config(self, session: _Session, threshold: float, silence_ms: int) -> None:
        """Áp cấu hình runtime lên `stats` của phiên (đổi ngưỡng/silence giữa phiên)."""
        stats = session.cache.get("stats") if isinstance(session.cache, dict) else None
        if stats is None:
            return
        if float(getattr(stats, "speech_noise_thres", threshold)) != float(threshold):
            stats.speech_noise_thres = float(threshold)
        # `max_end_sil_frame_cnt_thresh` = thời gian im lặng cần thiết - thời gian chuyển
        # speech→silence (đúng cách `DynamicStreamingVAD` của funasr làm).
        desired = max(0, int(silence_ms) - int(config.vad.fsmn.speech_to_sil_time_thres))
        if int(getattr(stats, "max_end_sil_frame_cnt_thresh", desired)) != desired:
            stats.max_end_sil_frame_cnt_thresh = desired
