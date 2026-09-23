"""Test tầng A — TIMER cho SEG: lấy mốc cắt chính xác bằng model có timestamp.

Bối cảnh (đo thực tế, xem `report/audit/21_SEG_VA_FIX_DEDUP_PHU_DE.md` §8):
  * Qwen3-ASR KHÔNG có timestamp (`qwen3_asr/capabilities.cpp`: `TIMESTAMPS_NONE`).
  * whisper-large-v3-turbo trả timestamp mức SEGMENT; nemotron/parakeet trả mức TOKEN.
⇒ Timer chạy whisper trên mảnh audio đã đóng để lấy mốc ms, rồi **căn chỉnh văn bản**
Qwen3 ↔ Whisper để biết dấu `。` nằm ở mốc nào.

Tầng A KHÔNG nạp model thật: phần căn chỉnh là hàm thuần; phần `WhisperTimer` dùng
"session giả" bơm segment tổng hợp.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from backend.asr import timer as timer_mod
from backend.asr.timer import (
    TimedChar,
    WhisperTimer,
    build_char_timeline,
    locate_cut_ms,
    normalize_for_align,
)


def _seg(text: str, t0: float, t1: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, t0_ms=t0, t1_ms=t1)


# ------------------------------------------------------------------ căn chỉnh thuần
def test_chuan_hoa_bo_dau_cau_va_khoang_trang():
    assert normalize_for_align("技術決定論の、解釈は。") == "技術決定論の解釈は"
    assert normalize_for_align("Hello, world!") == "Helloworld"


def test_timeline_rai_deu_ky_tu_theo_segment():
    tl = build_char_timeline([_seg("あいう", 0.0, 300.0)])
    assert [tc.char for tc in tl] == ["あ", "い", "う"]
    assert tl[0].start_ms == 0.0 and tl[0].end_ms == 100.0
    assert tl[2].start_ms == 200.0 and tl[2].end_ms == 300.0


def test_timeline_bo_qua_segment_rong():
    tl = build_char_timeline([_seg("", 0, 100), _seg("あ", 100, 200)])
    assert len(tl) == 1 and tl[0].char == "あ"


def test_dinh_vi_ranh_gioi_khi_hai_van_ban_giong_nhau():
    """Whisper đọc đúng như Qwen3 ⇒ mốc cắt = mốc bắt đầu câu kế tiếp."""
    whisper = [_seg("今日はいい天気ですね。", 0, 1000), _seg("明日は雨です。", 1000, 2000)]
    timeline = build_char_timeline(whisper)
    main = "今日はいい天気ですね。明日は雨は雨です。"
    boundary = main.index("明日")
    res = locate_cut_ms(main, boundary, timeline)
    assert res is not None
    # 10 ký tự nội dung ở câu 1 ⇒ ký tự "明" bắt đầu ở 1000 ms (đầu segment 2).
    assert res.cut_ms == pytest.approx(1000.0, abs=1.0)
    assert res.confidence > 0.9


def test_dinh_vi_chiu_duoc_whisper_nhan_sai_gan_ranh_gioi():
    """Whisper nhận sai 1 ký tự ngay trước ranh giới ⇒ mốc cắt vẫn đúng (sai số 1 ký tự)."""
    whisper = [_seg("今日はいい天気ですね。", 0, 1000), _seg("明日は雨です。", 1000, 2000)]
    timeline = build_char_timeline(whisper)
    main = "今日はいい天気ですね。明日は雨は雨です。"     # Qwen3 (chuẩn)
    boundary = main.index("明日")
    res = locate_cut_ms(main, boundary, timeline)
    assert res is not None and res.cut_ms == pytest.approx(1000.0, abs=1.0)

    # Whisper nghe nhầm "天気" -> "転記" trong câu 1.
    whisper_bad = [_seg("今日はいい転記ですね。", 0, 1000), _seg("明日は雨です。", 1000, 2000)]
    res2 = locate_cut_ms(main, boundary, build_char_timeline(whisper_bad))
    assert res2 is not None
    assert res2.cut_ms == pytest.approx(1000.0, abs=150.0), res2


def test_dinh_vi_khi_whisper_thieu_han_mot_cau():
    """Whisper bỏ sót cả câu đầu ⇒ vẫn trả mốc nằm trong vùng hợp lệ, tin cậy thấp."""
    whisper = [_seg("明日は雨です。", 1000, 2000)]
    timeline = build_char_timeline(whisper)
    main = "今日はいい天気ですね。明日は雨は雨です。"
    res = locate_cut_ms(main, main.index("明日"), timeline)
    assert res is not None
    assert 900.0 <= res.cut_ms <= 2100.0
    assert res.confidence < 0.9


def test_dinh_vi_ranh_gioi_o_cuoi_van_ban():
    timeline = build_char_timeline([_seg("おわり。", 0, 800)])
    res = locate_cut_ms("おわり。", len("おわり。"), timeline)
    assert res is not None and res.cut_ms == pytest.approx(800.0, abs=1.0)


def test_dinh_vi_tra_None_khi_thieu_du_lieu():
    assert locate_cut_ms("", 0, []) is None
    assert locate_cut_ms("abc", 1, []) is None
    assert locate_cut_ms("!!!", 1, build_char_timeline([_seg("あ", 0, 100)])) is None


def test_nguong_tin_cay_loc_ket_qua_xau():
    whisper = [_seg("全然違う内容です。", 0, 1000)]
    timeline = build_char_timeline(whisper)
    main = "今日はいい天気ですね。明日は雨は雨です。"
    assert locate_cut_ms(main, main.index("明日"), timeline, min_confidence=0.9) is None


# ------------------------------------------------------------------ WhisperTimer
def test_whisper_timer_dung_session_rieng(monkeypatch):
    """Timer phải tạo session RIÊNG, không đụng `TranscribeEngine._shared_session`."""
    from backend.asr.engine import TranscribeEngine

    created = {}

    class _FakeModel:
        def __init__(self, path, backend=None):
            created["path"] = path
            self.closed = False

        def session(self, n_threads=4):
            return _FakeSession()

        def close(self):
            self.closed = True

    class _FakeSession:
        def run(self, pcm, language=None, family=None, timestamps=None):
            assert timestamps == "segment"
            return SimpleNamespace(segments=[
                _seg("今日はいい天気ですね。", 0, 1000),
                _seg("明日は雨です。", 1000, 2000),
            ])

        def close(self):
            pass

    import transcribe_cpp

    monkeypatch.setattr(transcribe_cpp, "Model", _FakeModel, raising=False)
    monkeypatch.setattr(transcribe_cpp, "WhisperOptions", lambda: None, raising=False)
    monkeypatch.setattr(WhisperTimer, "available", lambda self, registry=None: True)

    before = TranscribeEngine._shared_session
    timer = WhisperTimer(model_key="whisper-large-v3-turbo", language="ja")
    main = "今日はいい天気ですね。明日は雨は雨です。"
    res = timer.locate_boundary(np.zeros(16000, dtype=np.float32), main.index("明日"), main)
    assert res is not None and res.cut_ms == pytest.approx(1000.0, abs=1.0)
    assert res.method == "whisper_segment"
    assert TranscribeEngine._shared_session is before, "không được đụng session của engine chính"
    assert created["path"].endswith(".gguf")
    timer.close()


def test_whisper_timer_khong_co_model_thi_tra_None(monkeypatch):
    timer = WhisperTimer(model_key="khong-ton-tai")
    monkeypatch.setattr(WhisperTimer, "available", lambda self, registry=None: False)
    assert timer.locate_boundary(np.zeros(16000, dtype=np.float32), 0, "abc") is None


# ------------------------------------------------------------------ nối vào engine
def _make_engine():
    from backend.asr.engine import TranscribeEngine
    from backend.config import config

    return TranscribeEngine(model_key=config.asr.active_model)


def test_engine_bat_seg_va_tat_bac3_khi_thay_the(restore_config):
    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.replace_stable_prefix = True
    eng = _make_engine()
    assert eng._seg is not None
    # BẬC 3 phải bị bỏ qua ⇒ dù text đứng yên mãi cũng không trả STABLE_PREFIX.
    assert eng._evaluate_tier234("これは十分に長い文です。", 3.0) is None


def test_engine_seg_chot_theo_dau_cau(restore_config):
    from backend.core.pipeline_events import CommitReason
    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = False
    # Bài này kiểm tra ĐƯỜNG NỐI, không kiểm tra luật chờ tail ⇒ cho chốt ngay nhịp đầu.
    config.segmentation.tail_scans = 1
    config.segmentation.tail_stable_ms = 0.0
    eng = _make_engine()
    assert eng._seg is not None

    text = "今日はいい天気ですね。明日は雨"
    assert eng._evaluate_seg(text, 16_000) is CommitReason.SEG_PUNCT
    assert eng._seg_decision.text == "今日はいい天気ですね。"

    # Chưa có bằng chứng (dấu câu ở cuối preview) ⇒ chưa chốt.
    eng._seg.reset()
    assert eng._evaluate_seg("今日はいい天気ですね。", 16_000) is None


def test_engine_seg_tat_thi_khong_anh_huong(restore_config):
    from backend.config import config

    config.segmentation.enabled = False
    eng = _make_engine()
    assert eng._seg is None
    assert eng._evaluate_seg("今日はいい天気ですね。明日は雨", 16_000) is None


def test_engine_dung_timer_de_lay_moc_cat(restore_config, monkeypatch):
    """Khi bật timer: mốc cắt = mốc ms của whisper, KHÔNG phải cuối vùng nói."""
    from backend.config import config
    from backend.core.pipeline_events import CommitReason

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = True
    config.segmentation.timer_min_confidence = 0.3
    config.segmentation.tail_scans = 1
    config.segmentation.tail_stable_ms = 0.0
    eng = _make_engine()

    fake = SimpleNamespace(
        available=lambda registry=None: True,
        locate_boundary=lambda pcm, idx, text, min_confidence=0.0: timer_mod.TimerResult(
            cut_ms=1500.0, confidence=0.9, elapsed_ms=12.0
        ),
    )
    monkeypatch.setattr(timer_mod, "get_seg_timer", lambda: fake)

    text = "今日はいい天気ですね。明日は雨"
    assert eng._evaluate_seg(text, 32_000) is CommitReason.SEG_PUNCT
    cut = eng._seg_cut_sample(0, 32_000, text)
    assert cut == int(1.5 * 16000), "phải dùng mốc của timer"


def test_engine_timer_vo_ly_thi_lui_ve_moc_an_toan(restore_config, monkeypatch):
    """Timer trả mốc ngoài vùng nói ⇒ phải bỏ, dùng mốc dự phòng của SEG."""
    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = True
    config.segmentation.tail_scans = 1
    config.segmentation.tail_stable_ms = 0.0
    eng = _make_engine()

    fake = SimpleNamespace(
        available=lambda registry=None: True,
        locate_boundary=lambda pcm, idx, text, min_confidence=0.0: timer_mod.TimerResult(
            cut_ms=999_000.0, confidence=0.9, elapsed_ms=5.0
        ),
    )
    monkeypatch.setattr(timer_mod, "get_seg_timer", lambda: fake)
    text = "今日はいい天気ですね。明日は雨"
    eng._evaluate_seg(text, 32_000)
    fallback = eng._seg_decision.cut_sample
    assert eng._seg_cut_sample(0, 32_000, text) == int(fallback)


def test_engine_timer_thieu_model_thi_khong_vo(restore_config, monkeypatch):
    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = True
    config.segmentation.tail_scans = 1
    config.segmentation.tail_stable_ms = 0.0
    eng = _make_engine()
    fake = SimpleNamespace(available=lambda registry=None: False, locate_boundary=None)
    monkeypatch.setattr(timer_mod, "get_seg_timer", lambda: fake)
    text = "今日はいい天気ですね。明日は雨"
    eng._evaluate_seg(text, 32_000)
    assert eng._seg_cut_sample(0, 32_000, text) >= 0


# ------------------------------------------------------------------ cấu hình runtime
def test_update_seg_config_dung_lai_may_trang_thai(restore_config):
    """Đổi cấu hình SEG phải có hiệu lực NGAY (không được là 'cấu hình vô hiệu')."""
    from backend.config import config
    from backend.core.pipeline_events import CommitReason

    config.segmentation.enabled = True
    config.segmentation.max_chars = 60
    config.segmentation.tail_scans = 2
    # Tầng A KHÔNG được nạp model thật (whisper 845 MB): tắt timer cho bài này.
    config.segmentation.use_whisper_timer = False
    eng = _make_engine()

    eng.update_seg_config(max_chars=20, tail_min_chars=1, tail_scans=1, tail_stable_ms=0.0)
    assert eng._seg is not None
    assert config.segmentation.max_chars == 20
    # `max_chars=20` phải cắt cưỡng bức ngay khi câu dài hơn 20 ký tự nội dung.
    text = "あ" * 12 + "、" + "い" * 15
    assert eng._evaluate_seg(text, 32_000) is CommitReason.SEG_PUNCT
    assert eng._seg_decision.reason == "max_chars"

    # Tắt SEG ⇒ engine bỏ hẳn máy trạng thái.
    eng.update_seg_config(enabled=False)
    assert eng._seg is None
    assert eng._evaluate_seg("今日はいい天気ですね。明日は雨", 32_000) is None


def test_session_apply_config_seg(session_factory, restore_config):
    """Popup gửi `segEnabled`/`segMaxChars`/… qua WS ⇒ áp vào engine đang chạy."""
    from backend.config import config

    session = session_factory()
    session.apply_config({
        "segEnabled": False,
        "segMaxChars": 90,
        "segTailMinChars": 6,
        "segTailScans": 3,
        "segUseWhisperTimer": False,
    })
    assert config.segmentation.max_chars == 90
    assert config.segmentation.tail_min_chars == 6
    assert config.segmentation.tail_scans == 3
    assert config.segmentation.use_whisper_timer is False
    assert session.config["seg_enabled"] is False
    assert session.config["seg_max_chars"] == 90
    eng = session.asr_engine
    if eng is not None:
        assert eng._seg is None, "tắt SEG thì engine phải bỏ máy trạng thái"


def test_api_config_co_khoi_seg(restore_config):
    from backend import main as main_mod

    payload = main_mod._build_config_response(include_catalog=False)
    seg = payload.get("seg")
    assert isinstance(seg, dict)
    for key in ("enabled", "max_chars", "tail_min_chars", "tail_scans", "use_whisper_timer"):
        assert key in seg, f"thiếu khoá '{key}' trong /api/config"


def test_popup_va_content_script_truyen_duoc_cau_hinh_seg():
    """Chốt ở mức mã nguồn: công tắc A/B cho SEG phải đi hết chuỗi popup → WS."""
    from pathlib import Path

    ext = Path(__file__).resolve().parents[2] / "extension_firefox"
    html = (ext / "popup" / "popup.html").read_text(encoding="utf-8")
    js = (ext / "popup" / "popup.js").read_text(encoding="utf-8")
    content = (ext / "content" / "content-script.js").read_text(encoding="utf-8")

    assert 'id="chkSegEnabled"' in html and 'id="chkSegTimer"' in html
    assert "rangeSegMaxChars" in html and "rangeSegTailChars" in html
    assert "segEnabled" in js and "seg_use_whisper_timer" in js
    assert "segEnabled" in content and "out.segMaxChars" in content


def test_chon_cau_khop_voi_cau_seg_da_chot():
    """FIX-12b: văn bản chạy lại có NHIỀU câu ⇒ giữ câu KHỚP với câu SEG đã chốt.

    Log thật: SEG chốt `'Okay. Well, … Aquaman statue.'` nhưng mảnh chạy lại dài hơn
    (`… Aquaman. This isn't a gag gift, Stewart.`). Bản FIX-12 đầu tiên lấy CÂU ĐẦU
    (`'Okay.'`) ⇒ bị coi là quá ngắn ⇒ gộp lại ⇒ vùng không tiến ⇒ LẶP 6 lần tới
    MAX_DURATION (câu phình 15 s).
    """
    from backend.asr.engine import _best_matching_sentence

    parts = [
        "Okay.",
        "Well, I don't know how much you want to spend, but I do have this pretty cool Aquaman statue.",
        "Aquaman.",
        "This isn't a gag gift, Stewart.",
    ]
    target = ("Okay. Well, I don't know how much you want to spend, but I do have this "
              "pretty cool Aquaman statue.")
    assert _best_matching_sentence(parts, target) == parts[1]
    # Không có target ⇒ lấy câu dài nhất (không lấy mảnh vụn).
    assert _best_matching_sentence(parts, "") == parts[1]


def test_cat_vong_lap_manh_ngan_lap_lai(restore_config, monkeypatch):
    """FIX-12b: cùng một mảnh ngắn bị gộp HAI lần ⇒ cắt vòng lặp, đẩy vùng lên `end_s`."""
    import asyncio

    from backend.config import config
    from backend.core.pipeline_events import CommitReason

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = False
    eng = _make_engine()
    eng._speech_active = True

    async def _fake_infer(audio_slice, is_commit=False):
        return "Okay."

    monkeypatch.setattr(eng, "_infer_with_watchdog", _fake_infer)
    eng.audio_buffer.write(np.zeros(64_000, dtype=np.float32))

    def _req(end_sample: int) -> dict:
        return {
            "utterance_id": "u1",
            "start_sample": 0,
            "end_sample": end_sample,
            "reason": CommitReason.SEG_PUNCT.value,
            "generation": eng._stream_generation,
        }

    async def _run(req):
        out = []
        async for msg in eng._emit_commit(req):
            out.append(msg)
        return out

    # Lần 1: mảnh ngắn ⇒ gộp (vùng vẫn ở `start_s`).
    assert asyncio.run(_run(_req(32_000))) == []
    assert eng._speech_start_sample == 0
    # Lần 2: CÙNG mảnh ⇒ phải cắt vòng lặp và đẩy vùng lên `end_s`.
    assert asyncio.run(_run(_req(32_000))) == []
    assert eng._speech_start_sample == 32_000, "phải phá vòng lặp, không đọc lại mãi"


def test_engine_mac_dinh_tat_stable_cut(restore_config):
    """Engine phải đọc `segmentation.stable_cut` (mặc định TẮT) và truyền vào SEG."""
    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = False
    config.segmentation.stable_cut = False
    eng = _make_engine()
    assert eng._seg.completer.allow_stable_cut is False
    # Không có câu mới ⇒ KHÔNG cắt dù dấu câu đứng yên.
    for _ in range(4):
        assert eng._evaluate_seg("あ、そうじゃん。", 16_000, 0) is None

    eng.update_seg_config(stable_cut=True)
    assert eng._seg.completer.allow_stable_cut is True


def test_timer_lay_ngon_ngu_cua_phien(restore_config):
    """Timer PHẢI chạy theo ngôn ngữ phiên (trước đây hard-code 'ja' ⇒ phiên EN bị sai)."""
    from unittest.mock import patch

    from backend.asr.timer import WhisperTimer

    seen = {}

    class _FakeSession:
        def run(self, pcm, language=None, family=None, timestamps=None):
            seen["language"] = language
            return SimpleNamespace(segments=[_seg("hello", 0, 500)])

    timer = WhisperTimer(model_key="x", language="ja")
    with patch.object(timer, "_ensure_session", lambda: _FakeSession()):
        timer.run_segments(np.zeros(1600, dtype=np.float32), language="en")
        assert seen["language"] == "en"
        timer.run_segments(np.zeros(1600, dtype=np.float32), language="auto")
        assert seen["language"] == "auto"
        # Không truyền ⇒ dùng mặc định của timer.
        timer.run_segments(np.zeros(1600, dtype=np.float32))
        assert seen["language"] == "ja"


def test_hot_reload_chap_nhan_khoi_segmentation(restore_config):
    """`hot_reload` phải chấp nhận khối `segmentation` (đường cấu hình nóng)."""
    from backend.config import config

    config.hot_reload({"segmentation": {"max_chars": 77, "tail_scans": 3}})
    assert config.segmentation.max_chars == 77
    assert config.segmentation.tail_scans == 3


# ------------------------------------------------------------------ trace từng nhịp
def test_trace_tung_nhip_ghi_ly_do_cho(restore_config, caplog):
    """Mỗi nhịp ASR phải ghi [SEG_TRACE] kèm LÝ DO còn chờ (để tinh chỉnh mốc cắt)."""
    import logging as _logging

    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = False
    config.segmentation.debug_trace = True
    config.segmentation.tail_min_chars = 4
    config.segmentation.tail_scans = 2
    config.segmentation.tail_stable_ms = 0.0
    eng = _make_engine()

    with caplog.at_level(_logging.INFO, logger="backend.utils.logger"):
        eng._evaluate_seg("Leonard will be back in a couple days.", 16_000, 0)
        eng._evaluate_seg("Leonard will be back in a couple days. He", 16_000, 0)
        eng._evaluate_seg("Leonard will be back in a couple days. He is", 16_000, 0)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[SEG_TRACE]" in text, text
    assert "hold=" in text
    assert "tail" in text


def test_trace_ghi_ca_khi_chot(restore_config, caplog):
    import logging as _logging

    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = False
    config.segmentation.tail_min_chars = 1
    config.segmentation.tail_scans = 1
    config.segmentation.tail_stable_ms = 0.0
    eng = _make_engine()
    with caplog.at_level(_logging.INFO, logger="backend.utils.logger"):
        assert eng._evaluate_seg("Hello world. Super man", 16_000, 0) is not None
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[SEG_CUT]" in text and "cut=" in text


def test_shadow_log_khi_seg_tat(restore_config, caplog):
    """SEG TẮT vẫn phải log nó SẼ cắt ở đâu ⇒ so sánh A/B trong MỘT lần chạy video."""
    import logging as _logging

    from backend.config import config

    config.segmentation.enabled = False
    config.segmentation.debug_trace = True
    config.segmentation.shadow_when_disabled = True
    config.segmentation.tail_min_chars = 1
    config.segmentation.tail_scans = 1
    config.segmentation.tail_stable_ms = 0.0
    eng = _make_engine()
    assert eng._seg is None and eng._seg_shadow is not None
    with caplog.at_level(_logging.INFO, logger="backend.utils.logger"):
        assert eng._evaluate_seg("Hello world. Super man", 16_000, 0) is None  # KHÔNG cắt thật
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[SEG_SHADOW]" in text, text


def test_may_trang_thai_mo_ta_duoc_ly_do_cho():
    from backend.segmentation import SentenceCompleter

    c = SentenceCompleter(tail_min_chars=4, tail_scans=2, tail_stable_ms=0.0)
    c.observe("Hello world. Hi", now=1.0)
    st = c.trace_state()
    assert st["hold"], "phải có lý do chờ"
    assert "hold=tail_short" in st["hold"]
    assert st["boundary_index"] > 0 and st["tail"] == "Hi"


def test_manh_chi_con_dau_cau_bi_bo_han_khong_lap(restore_config, monkeypatch):
    """FIX-11: `'Aquaman.' -> '.'` không được lặp vô hạn (log thật lặp 4 lần + 4 lần timer).

    Mảnh cắt ra chỉ còn dấu câu ⇒ phải BỎ HẲN và đẩy mốc vùng đọc lên `end_s`.
    """
    import asyncio

    from backend.config import config
    from backend.core.pipeline_events import CommitReason

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = False
    eng = _make_engine()
    engine = eng
    engine._speech_active = True

    async def _fake_infer(audio_slice, is_commit=False):
        return "."

    monkeypatch.setattr(engine, "_infer_with_watchdog", _fake_infer)
    engine.audio_buffer.write(np.zeros(32_000, dtype=np.float32))

    req = {
        "utterance_id": "u1",
        "start_sample": 0,
        "end_sample": 32_000,
        "reason": CommitReason.SEG_PUNCT.value,
        "generation": engine._stream_generation,
    }
    engine._pending_commits.clear()

    async def _drain():
        out = []
        async for msg in engine._emit_commit(req):
            out.append(msg)
        return out

    msgs = asyncio.run(_drain())
    assert msgs == [], "mảnh chỉ có dấu câu ⇒ không được phát phụ đề"
    assert engine._speech_start_sample == 32_000, "phải đẩy mốc vùng đọc lên end_s"
