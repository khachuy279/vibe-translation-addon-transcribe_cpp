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
        }

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

