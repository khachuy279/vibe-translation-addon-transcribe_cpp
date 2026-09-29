"""Test tầng A — trace chẩn đoán `[SEG_TRACE]` / `[SEG_CUT]` cho cơ chế cắt theo độ ổn định.

Bối cảnh: khi gỡ tầng SEG (chốt câu theo dấu câu) + timer (2026-09-29), phần trace từng nhịp
preview cũng bị gỡ theo. Bài này chốt lại trace đó ở dạng mới — **chỉ để CHẨN ĐOÁN** cơ chế
`stable_cut` (không cắt câu, không đổi văn bản) — với hai yêu cầu:

1. **Mặc định TẮT** (`sentence.trace_stability = False`) vì mỗi nhịp một dòng ≈ 4 dòng/giây;
2. bật/tắt được từ popup ("Log each poll (SEG_TRACE)") và có hiệu lực NGAY.

Giữ nguyên token `SEG_TRACE` / `SEG_CUT` + các khoá `t=+`, `hold=`, `c=`, `cut=` như bản cũ để
log cũ/mới còn grep và so sánh được.

Tầng A: không nạp model.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _make_engine():
    from backend.asr.engine import TranscribeEngine
    from backend.config import config

    return TranscribeEngine(model_key=config.asr.active_model)


def _prep_engine(*, trace: bool = True, **sentence_overrides):
    """Engine với `stable_cut` sẵn sàng cắt nhanh (không phải chờ 2,5 s + 3 nhịp thật)."""
    from backend.config import config

    eng = _make_engine()
    cfg = eng.commit_manager.cfg
    cfg.max_duration_sec = 999.0
    cfg.split_on_stability = True
    cfg.stability_threshold_polls = 1
    cfg.stability_duration_sec = 0.0
    cfg.stability_min_duration_sec = 0.0
    cfg.stability_min_words = 0
    for key, value in sentence_overrides.items():
        setattr(cfg, key, value)
    eng._trace_stability = bool(trace)
    return eng


# ------------------------------------------------------------------ mặc định TẮT
def test_mac_dinh_trace_tat(restore_config):
    from backend.config import config

    assert config.sentence.trace_stability is False, "trace mỗi nhịp phải MẶC ĐỊNH TẮT"
    eng = _make_engine()
    assert eng._trace_stability is False


def test_khong_ghi_log_khi_trace_tat(restore_config, caplog):
    import logging as _logging

    eng = _prep_engine(trace=False)
    with caplog.at_level(_logging.INFO):
        for _ in range(3):
            reason = eng._evaluate_tier234("alpha beta gamma delta", duration_sec=3.0)
            eng._trace_stability_poll(reason, "alpha beta gamma delta", 3.0, 48_000, 0)
    assert "SEG_TRACE" not in caplog.text and "SEG_CUT" not in caplog.text, caplog.text


# ------------------------------------------------------------------ nội dung trace
def test_ghi_seg_trace_tung_nhip_kem_ly_do_con_cho(restore_config, caplog):
    """Nhịp chưa đủ điều kiện phải ghi `[SEG_TRACE]` + `hold=` + `n=<nhịp>/<cần>`."""
    import logging as _logging

    eng = _prep_engine(trace=True)
    # Còn chờ vì câu chưa đủ dài (`stability_min_duration_sec`).
    eng.commit_manager.cfg.stability_min_duration_sec = 5.0
    with caplog.at_level(_logging.INFO):
        reason = eng._evaluate_tier234("alpha beta gamma", duration_sec=1.0)
        eng._trace_stability_poll(reason, "alpha beta gamma", 1.0, 16_000, 0)

    assert reason is None
    line = caplog.text
    assert "[SEG_TRACE]" in line, line
    assert "hold=floor_duration 1.0/5.0s" in line, line
    assert "n=0/1" in line, line
    assert "nw=3" in line, line
    assert "'alpha beta gamma'" in line, line


def test_ghi_ly_do_dang_cho_du_do_on_dinh(restore_config, caplog):
    """Đủ sàn rồi nhưng text chưa đứng im đủ lâu ⇒ `hold=stability n=../.. t=../..ms`."""
    import logging as _logging

    eng = _prep_engine(trace=True)
    eng.commit_manager.cfg.stability_threshold_polls = 3
    eng.commit_manager.cfg.stability_duration_sec = 10.0
    with caplog.at_level(_logging.INFO):
        for _ in range(2):
            reason = eng._evaluate_tier234("alpha beta gamma", duration_sec=3.0)
        eng._trace_stability_poll(reason, "alpha beta gamma", 3.0, 48_000, 0)

    assert reason is None
    assert "hold=stability n=2/3" in caplog.text, caplog.text
    assert "t=" in caplog.text and "/10000ms" in caplog.text, caplog.text


def test_ghi_seg_cut_o_nhip_chot_cau(restore_config, caplog):
    """Nhịp CHỐT câu phải ghi `[SEG_CUT]` kèm lý do, độ dài câu và số nhịp."""
    import logging as _logging

    eng = _prep_engine(trace=True)
    with caplog.at_level(_logging.INFO):
        eng._evaluate_tier234("alpha beta gamma", duration_sec=3.0)   # nhịp 1: ghi nhận text
        reason = eng._evaluate_tier234("alpha beta gamma", duration_sec=3.0)  # nhịp 2: đủ ổn định
        eng._trace_stability_poll(reason, "alpha beta gamma", 3.0, 48_000, 16_000)

    assert reason is not None and reason.value == "STABLE_PREFIX"
    line = caplog.text
    assert "[SEG_CUT]" in line, line
    assert "cut=STABLE_PREFIX" in line and "dur=3.0s" in line, line
    assert "t=+2.0s" in line, line


def test_trace_khong_doi_quyet_dinh_cat(restore_config):
    """Bật trace KHÔNG được đổi hành vi: cùng input ⇒ cùng kết quả."""
    out = []
    for trace in (False, True):
        eng = _prep_engine(trace=trace)
        out.append([
            eng._evaluate_tier234("alpha beta gamma", duration_sec=3.0) for _ in range(3)
        ])
    assert out[0] == out[1]


# ------------------------------------------------------------------ cấu hình & UI
def test_api_config_tra_co_trace(restore_config):
    from backend.main import _build_config_response

    resp = _build_config_response(include_catalog=False)
    assert resp["stable"]["trace"] is False


def test_ws_payload_co_truong_trace():
    from backend.ws.session import SessionConfigPayload

    field = SessionConfigPayload.model_fields.get("trace_stability")
    assert field is not None and field.alias == "traceStability"


def test_bat_trace_qua_engine_co_hieu_luc_ngay(restore_config):
    from backend.config import config

    eng = _make_engine()
    eng.update_sentence_config(trace_stability=True)
    assert eng._trace_stability is True
    eng.update_sentence_config(trace_stability=False)
    assert eng._trace_stability is False
    assert config.sentence.trace_stability is False


def test_popup_co_cong_tac_trace_mac_dinh_tat():
    html = (ROOT / "extension_firefox" / "popup" / "popup.html").read_text(encoding="utf-8")
    assert 'id="chkStableTrace"' in html
    # Công tắc KHÔNG được có `checked` (mặc định TẮT).
    start = html.index('id="chkStableTrace"')
    tag = html[html.rindex("<input", 0, start):html.index(">", start)]
    assert "checked" not in tag, tag
    assert "SEG_TRACE" in html, "nhãn công tắc phải nói rõ token log"


def test_popup_js_va_content_script_truyen_trace():
    js = (ROOT / "extension_firefox" / "popup" / "popup.js").read_text(encoding="utf-8")
    cs = (ROOT / "extension_firefox" / "content" / "content-script.js").read_text(encoding="utf-8")
    assert "chkStableTrace" in js and "trace_stability" in js
    assert "traceStability" in cs, "content-script phải chuyển tiếp công tắc xuống backend"
