"""Hệ thống thu thập và phân tích Metrics hiệu năng thời gian thực.

Hỗ trợ:
- Đo lường độ trễ chi tiết từng chặng (VAD, ASR TTFT, ASR Commit, Translation, TTS, E2E Pipeline).
- Tính toán phân vị độ trễ (p50, p90, p95, p99).
- Giám sát số lượng sample drop, số lần lọc dedup trùng, số lần force commit.
- Xuất báo cáo hiệu năng JSON và định dạng bảng Markdown.
"""

from collections import OrderedDict, defaultdict, deque
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np

# Trần số lượng checkpoint giữ lại. Trước đây `_checkpoints` là dict không bound và
# handler ghi 2 key duy nhất theo session => tăng vô hạn (F-15 / P4.4).
_MAX_CHECKPOINTS = 200


class MetricsCollector:
    """Bộ thu thập số liệu hiệu năng tập trung cho Backend."""

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Singleton accessor."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = MetricsCollector()
            return cls._instance

    def __init__(self, max_history: int = 10000):
        self._latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, int] = defaultdict(int)
        # Bounded LRU-ish checkpoint store: giữ tối đa _MAX_CHECKPOINTS mục, đẩy mục cũ nhất ra.
        self._checkpoints: "OrderedDict[str, float]" = OrderedDict()
        # Gauge: giá trị tức thời ghi đè (độ sâu queue, VRAM, số session...).
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()

    def record_latency(self, stage: str, latency_ms: float) -> None:
        """Ghi nhận thời gian thực thi (ms) cho một công đoạn."""
        if latency_ms < 0:
            return
        with self._lock:
            self._latencies[stage].append(latency_ms)

    def record_gauge(self, stage: str, name: str, value: float) -> None:
        """Ghi giá trị tức thời (độ sâu queue, số mục đang chờ...). Ghi đè giá trị cũ."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._gauges[f"{stage}.{name}"] = v

    def get_gauge(self, stage: str, name: str, default: float = 0.0) -> float:
        """Đọc giá trị gauge gần nhất."""
        with self._lock:
            return self._gauges.get(f"{stage}.{name}", default)

    # Tương thích ngược: tên cũ dùng trong code cũ.
    record_value = record_gauge

    def increment_counter(self, name: str, count: int = 1) -> None:
        """Tăng bộ đếm sự kiện (ví dụ sample drop, dedup skip)."""
        with self._lock:
            self._counters[name] += count

    def get_counter(self, name: str) -> int:
        """Lấy giá trị hiện tại của một bộ đếm."""
        with self._lock:
            return self._counters[name]

    def get_stage_stats(self, stage: str) -> Dict[str, float]:
        """Tính toán thống kê chi tiết (count, min, max, avg, p50, p95, p99) cho một stage."""
        with self._lock:
            values = list(self._latencies.get(stage, []))

        if not values:
            return {
                "count": 0,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        arr = np.array(values, dtype=np.float64)
        return {
            "count": int(len(arr)),
            "avg_ms": round(float(np.mean(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p90_ms": round(float(np.percentile(arr, 90)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }

    def generate_report(self) -> Dict[str, Any]:
        """Tạo báo cáo tổng hợp toàn bộ số liệu hiệu năng."""
        uptime_sec = time.time() - self._start_time

        stages_summary = {}
        with self._lock:
            stage_keys = list(self._latencies.keys())
            counters_copy = dict(self._counters)
            gauges_copy = dict(self._gauges)
            n_checkpoints = len(self._checkpoints)

        for stage in stage_keys:
            stages_summary[stage] = self.get_stage_stats(stage)

        return {
            "uptime_sec": round(uptime_sec, 2),
            "counters": counters_copy,
            "gauges": gauges_copy,
            "stages": stages_summary,
            "checkpoints_retained": n_checkpoints,
            # TỰ MÔ TẢ: nếu không có hai mục dưới đây thì file report vô dụng cho A/B —
            # mở ra sẽ không biết số liệu được đo với backend/cờ nào.
            "runtime": self._runtime_snapshot(),
            "run_config": self._run_config_snapshot(),
        }

    @staticmethod
    def _runtime_snapshot() -> Dict[str, Any]:
        """Trạng thái runtime ảnh hưởng tới số đo (backend ASR, arbiter)."""
        out: Dict[str, Any] = {}
        try:
            from backend.core.gpu_scheduler import gpu_arbiter

            out["gpu_arbiter"] = gpu_arbiter.stats()
        except Exception:  # noqa: BLE001
            pass
        try:
            from backend.asr import native as _native

            out["asr_backends_available"] = sorted(_native._available_kinds())
        except Exception:  # noqa: BLE001
            pass
        try:
            from backend.asr.engine import TranscribeEngine

            model = TranscribeEngine._shared_model
            if model is not None:
                out["asr_backend_in_use"] = getattr(model, "backend", None)
                out["asr_model"] = TranscribeEngine._shared_model_key
        except Exception:  # noqa: BLE001
            pass
        return out

    @staticmethod
    def _run_config_snapshot() -> Dict[str, Any]:
        """Ảnh chụp các cờ cấu hình ảnh hưởng trực tiếp tới hiệu năng.

        Chỉ lấy một allowlist nhỏ, cố định — KHÔNG dump toàn bộ config (dễ lộ đường dẫn
        tuyệt đối và phình file).
        """
        try:
            from backend.config import config as _cfg
        except Exception:  # noqa: BLE001
            return {}
        picks = {
            "asr.backend": getattr(_cfg.asr, "backend", None),
            "asr.backend_fallback": getattr(_cfg.asr, "backend_fallback", None),
            "asr.threads": getattr(_cfg.asr, "threads", None),
            "asr.preview_window_sec": getattr(_cfg.asr, "preview_window_sec", None),
            "asr.poll_interval_ms": getattr(_cfg.asr, "poll_interval_ms", None),
            "asr.max_inflight_infer": getattr(_cfg.asr, "max_inflight_infer", None),
            "asr.preview_reuse_for_commit": getattr(_cfg.asr, "preview_reuse_for_commit", None),
            "asr.normalize_speech": getattr(_cfg.asr, "normalize_speech", None),
            "gpu.scheduler_enabled": getattr(_cfg.gpu, "scheduler_enabled", None),
            "gpu.commit_reserve_ms": getattr(_cfg.gpu, "commit_reserve_ms", None),
            "gpu.admit_max_wait_ms": getattr(_cfg.gpu, "admit_max_wait_ms", None),
            "tts.enabled": getattr(_cfg.tts, "enabled", None),
            "tts.speed": getattr(_cfg.tts, "speed", None),
            "translation.base": getattr(_cfg.translation, "base", None),
            "translation.n_ctx": getattr(_cfg.translation, "n_ctx", None),
            "sentence.max_duration_sec": getattr(_cfg.sentence, "max_duration_sec", None),
        }
        return picks

    def dump_json(self, filepath: str) -> None:
        """Lưu báo cáo hiệu năng ra file JSON."""
        data = self.generate_report()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def record_metric(self, stage: str, metric_name: str, value: float) -> None:
        """Helper ghi nhận metric."""
        self.record_latency(f"{stage}.{metric_name}", value)

    def record_checkpoint(self, name: str) -> None:
        """Ghi nhận checkpoint thời gian (bounded, tự đẩy mục cũ nhất ra)."""
        with self._lock:
            self._checkpoints[name] = time.time()
            self._checkpoints.move_to_end(name)
            while len(self._checkpoints) > _MAX_CHECKPOINTS:
                self._checkpoints.popitem(last=False)

    def get_checkpoint(self, name: str) -> Optional[float]:
        """Đọc timestamp của checkpoint gần nhất theo tên."""
        with self._lock:
            return self._checkpoints.get(name)

    def reset(self) -> None:
        """Xóa toàn bộ số liệu đo lường."""
        with self._lock:
            self._latencies.clear()
            self._counters.clear()
            self._checkpoints.clear()
            self._gauges.clear()
            self._start_time = time.time()

    def snapshot_pipeline(self) -> Dict[str, Any]:
        """Ảnh chụp gọn cho hot path: các stage/gauge quan trọng của pipeline.

        Dùng ở /api/metrics để không phải duyệt toàn bộ lịch sử percentile.
        """
        wanted_stages = [
            "asr.preview_ms",
            "asr.commit_ms",
            "asr.first_preview_ms",
            "asr.e2e_commit_ms",
            "asr.preview_audio_sec",
            "asr.commit_audio_sec",
            # FIX-10: đo mức lãng phí compute của preview (xem engine._finalize_recompute_metrics).
            "asr.audio_seconds_unique_preview",
            "asr.preview_recompute_ratio",
            "asr.idle_wait_ms",
            "vad.chunk_ms",
            "vad.frame_ms",
            "translation.queue_wait_ms",
            "translation.infer_ms",
            "tts.queue_wait_ms",
            "tts.synthesis_ms",
            "pipeline.e2e_asr_to_sub_ms",
            "pipeline.e2e_sub_to_tts_ms",
        ]
        with self._lock:
            counters_copy = dict(self._counters)
            gauges_copy = dict(self._gauges)
        return {
            "stages": {s: self.get_stage_stats(s) for s in wanted_stages},
            "counters": counters_copy,
            "gauges": gauges_copy,
        }


metrics = MetricsCollector.get_instance()
metrics_collector = metrics


# --------------------------------------------------------------------------- ghi report
def resolve_report_path(filename: Optional[str] = None) -> str:
    """Đường dẫn TUYỆT ĐỐI cho file report.

    `config.metrics.report_file` mặc định là tên trần `"metrics_report.json"`. Nếu để
    `dump_json()` tự phân giải theo CWD thì file rơi vào thư mục tuỳ theo cách khởi chạy
    backend (`uvicorn backend.main:app` từ đâu) ⇒ rất khó tìm. Ở đây tên tương đối được
    neo vào `PROJECT_ROOT` để luôn nằm cạnh `README.md`.
    """
    from pathlib import Path

    from backend.config import PROJECT_ROOT, config

    name = (
        filename
        or getattr(getattr(config, "metrics", None), "report_file", None)
        or "metrics_report.json"
    )
    p = Path(str(name))
    if not p.is_absolute():
        p = Path(PROJECT_ROOT) / p
    return str(p)


def dump_metrics_report(reason: str = "manual", filename: Optional[str] = None) -> Optional[str]:
    """Ghi metrics ra JSON và trả về đường dẫn (None nếu tắt hoặc ghi lỗi).

    **Vì sao hàm này tồn tại:** `MetricsCollector.dump_json()` đã có từ lâu nhưng
    **không ai gọi**, còn `config.metrics.dump_report_on_disconnect` / `report_file` /
    `enabled` thì **không ai đọc** — nên `metrics_report.json` chưa bao giờ được tạo ra
    dù config nói sẽ tạo. Hàm này đấu dây lại ba mảnh đó.

    Không bao giờ raise: ghi report là tiện ích, không được làm hỏng đường cleanup phiên.
    """
    try:
        from backend.config import config

        if not bool(getattr(config.metrics, "enabled", True)):
            return None
        path = resolve_report_path(filename)
        metrics.dump_json(path)
        try:
            from backend.utils.logger import logger as _log

            _log.info(
                f"Đã ghi metrics report ({reason}) vào: {path}",
                extra={"module_tag": "METRICS"},
            )
        except Exception:  # noqa: BLE001
            pass
        return path
    except Exception as exc:  # noqa: BLE001
        try:
            from backend.utils.logger import logger as _log

            _log.warning(
                f"Không ghi được metrics report ({reason}): {exc}",
                extra={"module_tag": "METRICS"},
            )
        except Exception:  # noqa: BLE001
            pass
        return None

