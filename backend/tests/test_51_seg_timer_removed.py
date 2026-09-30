"""Test tầng A — CHỐT VIỆC GỠ BỎ tầng SEG + timer (2026-09-29).

Yêu cầu người dùng: *"xoá tính năng chốt câu theo DẤU CÂU của ASR + Engine timer... trong cả
extension và những thứ liên quan. Chỉ giữ lại stable_cut (text đứng im trong khoảng thời gian
cài đặt ⇒ COMMIT, không cần check có dấu câu hay không)."*

Bài này chốt lại bằng ASSERT để không ai vô tình đưa hai tầng đó trở lại (mã nguồn + model +
UI + khoá cấu hình). Tầng A: chỉ đọc file và import.
"""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


# ------------------------------------------------------------------ backend: mã nguồn
@pytest.mark.parametrize(
    "module",
    ["backend.segmentation", "backend.asr.timer", "backend.asr.aligner_timer"],
)
def test_module_da_bi_xoa(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_config_khong_con_khoi_segmentation(restore_config):
    from backend.config import config

    assert not hasattr(config, "segmentation"), "khối `segmentation` phải bị gỡ khỏi config"


def test_commit_reason_khong_con_seg_punct():
    from backend.core.pipeline_events import CommitReason

    assert not hasattr(CommitReason, "SEG_PUNCT")
    # 4 cửa chốt còn lại phải nguyên vẹn.
    for name in ("VAD_SILENCE", "MAX_DURATION", "STABLE_PREFIX", "TIMEOUT_FORCE"):
        assert hasattr(CommitReason, name), name


def test_engine_khong_con_api_seg_va_timer(restore_config):
    from backend.asr import engine as engine_mod
    from backend.config import config

    eng = engine_mod.TranscribeEngine(model_key=config.asr.active_model)
    for name in (
        "_seg", "_seg_decision", "_seg_cfg", "_seg_shadow", "_seg_commits",
        "update_seg_config", "prewarm_seg_timer", "_reset_seg_timer",
        "_evaluate_seg", "_seg_cut_sample", "_seg_base_cut_sample",
        "_log_seg_trace", "_trace_seg_shadow",
    ):
        assert not hasattr(eng, name), f"engine còn API của tầng SEG/timer: {name}"
    assert not hasattr(engine_mod, "_best_matching_sentence")
    # Chẩn đoán drift thì PHẢI còn (không thuộc tầng SEG).
    assert hasattr(eng, "_hypothesis_consensus") and hasattr(eng, "_warn_unseen_commit_text")


def test_engine_khong_con_tham_chieu_timer_trong_ma_nguon():
    src = (ROOT / "backend" / "asr" / "engine.py").read_text(encoding="utf-8")
    for token in ("from backend.asr.timer", "aligner_timer", "SEG_PUNCT", "prewarm_seg_timer",
                  "whisper_model_key", "_seg_cut_sample"):
        assert token not in src, f"engine.py còn tham chiếu '{token}'"


def test_khong_con_khoa_cau_hinh_seg_timer():
    from backend.config import config
    from backend.ws.session import SessionConfigPayload

    dumped = set(config.model_dump().keys())
    assert "segmentation" not in dumped
    for field in ("seg_enabled", "seg_use_whisper_timer", "seg_timer_engine", "seg_stable_cut"):
        assert field not in SessionConfigPayload.model_fields, field


def test_api_config_tra_khoi_stable_thay_cho_seg():
    from backend.main import _build_config_response

    resp = _build_config_response(include_catalog=False)
    assert "seg" not in resp, "khối `seg` phải bị gỡ khỏi /api/config"
    assert resp["stable"]["split_on_stability"] is True
    assert resp["stable"]["duration_ms"] >= 0
    assert "stability_duration_sec" in resp["streaming"]


# ------------------------------------------------------------------ extension
def _ext(*parts: str) -> str:
    return (ROOT / "extension_firefox" / Path(*parts)).read_text(encoding="utf-8")


def test_popup_bo_khoi_seg_va_co_dieu_khien_stable():
    html = _ext("popup", "popup.html")
    for dead in ("chkSegEnabled", "chkSegTimer", "chkSegTrace", "chkSegStable",
                 "rangeSegMaxChars", "rangeSegTailChars", "rangeSegTailScans", "rangeSegStableMs"):
        assert dead not in html, f"popup.html còn điều khiển SEG: {dead}"
    for live in ("chkStableCut", "rangeStableMs", "rangeStableMinSec", "rangeStableMinWords"):
        assert f'id="{live}"' in html, f"popup.html thiếu điều khiển stable_cut: {live}"


def test_popup_js_gui_cau_hinh_stable_qua_ws_va_rest():
    js = _ext("popup", "popup.js")
    assert "segEnabled" not in js and "segStableCut" not in js, "popup.js còn gửi cấu hình SEG"
    assert "stabilityDurationSec" in js and "splitOnStability" in js
    assert "stability_duration_ms" in js and "stability_min_duration_sec" in js


def test_content_script_chuyen_tiep_cau_hinh_stable():
    js = _ext("content", "content-script.js")
    assert "segUseWhisperTimer" not in js, "content-script còn chuyển tiếp cấu hình SEG"
    assert "stabilityDurationSec" in js and "splitOnStability" in js


def test_khong_con_tai_lieu_hay_probe_nao_cua_seg_timer():
    for rel in (
        "report/audit/21_SEG_VA_FIX_DEDUP_PHU_DE.md",
        "report/audit/26_KHA_THI_QWEN3_ASR_TRANSFORMERS.md",
        "report/audit/27_SEG_AB_ALIGNER_VS_WHISPER.md",
        "scratch/seg_ja_probe.py",
        "scratch/seg_timer_e2e.py",
    ):
        assert not (ROOT / rel).exists(), f"tài liệu/probe của SEG còn sót: {rel}"
