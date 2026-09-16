"""Test tầng A — FIX-10: đo mức LÃNG PHÍ compute của ASR preview.

ChatGPT ước lượng preview xử lý lại ~8× lượng audio. Trước khi bỏ công sức (và rủi ro hồi
quy WER) vào incremental/adaptive preview, phải ĐO được con số thật. Test này chốt rằng
phép đo tồn tại, đúng công thức và được reset theo từng câu.
"""

from backend.asr.engine import TranscribeEngine
from backend.core.metrics import metrics_collector


def _engine(restore_config) -> TranscribeEngine:
    return TranscribeEngine()


def test_bon_so_lieu_duoc_khoi_tao(restore_config):
    engine = _engine(restore_config)
    assert engine._preview_sec_processed == 0.0
    assert engine._preview_sec_unique_peak == 0.0


def test_publish_recompute_ratio_dung_cong_thuc(restore_config):
    """ratio = tổng giây audio đã xử lý / số giây audio THẬT của câu."""
    engine = _engine(restore_config)

    # Câu 6 s, preview 4 lượt × 6 s = 24 s audio đưa qua model ⇒ ratio = 4.
    engine._preview_sec_processed = 24.0
    engine._preview_sec_unique_peak = 6.0
    engine._publish_recompute_ratio()

    ratio = metrics_collector.get_gauge("asr", "preview_recompute_ratio")
    assert abs(ratio - 4.0) < 1e-6, ratio
    assert abs(metrics_collector.get_gauge("asr", "audio_seconds_unique_preview") - 6.0) < 1e-6


def test_khong_chia_cho_khong(restore_config):
    """Chưa có audio thật (unique = 0) ⇒ không được ghi ratio (tránh chia 0)."""
    engine = _engine(restore_config)
    metrics_collector.record_gauge("asr", "preview_recompute_ratio", -1.0)

    engine._preview_sec_processed = 5.0
    engine._preview_sec_unique_peak = 0.0
    engine._publish_recompute_ratio()

    assert metrics_collector.get_gauge("asr", "preview_recompute_ratio", -1.0) == -1.0


def test_finalize_ghi_metric_va_reset(restore_config):
    """Chốt câu ⇒ ghi metric (để có p50/p95) rồi reset bộ đếm cho câu sau."""
    engine = _engine(restore_config)
    engine._preview_sec_processed = 30.0
    engine._preview_sec_unique_peak = 6.0

    engine._finalize_recompute_metrics()

    stats = metrics_collector.get_stage_stats("asr.preview_recompute_ratio")
    assert stats["count"] >= 1
    assert abs(stats["max_ms"] - 5.0) < 1e-6, stats
    # Reset để câu sau không cộng dồn nhầm.
    assert engine._preview_sec_processed == 0.0
    assert engine._preview_sec_unique_peak == 0.0


def test_metric_co_trong_snapshot_pipeline():
    """`/api/metrics` phải trả được 2 chỉ số mới (nếu không benchmark không đọc được)."""
    snapshot = metrics_collector.snapshot_pipeline()
    assert "asr.preview_recompute_ratio" in snapshot["stages"]
    assert "asr.audio_seconds_unique_preview" in snapshot["stages"]


def test_speech_start_reset_bo_dem(restore_config):
    """Câu mới ⇒ bộ đếm phải về 0 (không mang số liệu của câu trước sang)."""
    engine = _engine(restore_config)
    engine._preview_sec_processed = 12.0
    engine._preview_sec_unique_peak = 3.0

    engine.on_speech_start()

    assert engine._preview_sec_processed == 0.0
    assert engine._preview_sec_unique_peak == 0.0
