"""Test tầng A — CỬA SỔ GIỮ CÂU khi VAD báo hết tiếng quá sớm (`sentence.hold_short_sentence`).

Vấn đề (người dùng phát hiện 2026-09-29)
---------------------------------------
`VAD_SILENCE` là BẬC 1 nên nó LUÔN thắng `stability_min_duration_sec` (sàn chỉ gác BẬC 3 —
STABLE_PREFIX): câu mới nói ~1,2 s mà người nói ngừng ~1 s (`vad.silence_duration_ms`) là bị
chốt ngay ⇒ phụ đề CỤT, hai sàn chống-cắt-sớm vô hiệu.

Quy tắc mới (3 nhánh, đúng yêu cầu):
  1. VAD END khi vùng nói < `stability_min_duration_sec` ⇒ **GIỮ câu**, chưa chốt;
  2. hết cửa sổ giữ mà vẫn im lặng ⇒ **chốt** (reason = VAD_SILENCE như cũ);
  3. trong cửa sổ giữ mà VAD START lại ⇒ **GHÉP** audio mới vào CÙNG câu (giữ `utterance_id`,
     giữ mốc đầu câu) — không cắt cụt rồi mở câu mới.

Tầng A: không nạp model (không cần VAD thật — gọi thẳng callback của engine).
"""

import time

import numpy as np
import pytest

SR = 16000


@pytest.fixture
def engine(restore_config):
    from backend.asr.engine import TranscribeEngine
    from backend.config import config

    eng = TranscribeEngine(model_key=config.asr.active_model)
    config.sentence.stability_min_duration_sec = 2.5
    config.sentence.hold_short_sentence = True
    return eng


def _feed(eng, sec: float) -> None:
    """Nạp `sec` giây audio vào buffer (coi như đã có tiếng nói)."""
    eng.audio_buffer.write(np.zeros(int(sec * SR), dtype=np.float32))


def _pending(eng):
    return list(eng._pending_commits)


# ------------------------------------------------------------------ nhánh 1: GIỮ câu
def test_vad_end_som_thi_giu_cau_chua_chot(engine):
    """Câu 1,2 s < sàn 2,5 s ⇒ VAD END KHÔNG được chốt ngay."""
    _feed(engine, 1.2)
    engine._speech_active = True
    engine._speech_start_sample = 0

    engine.on_speech_end()

    assert _pending(engine) == [], "chưa đủ `stability_min_duration_sec` ⇒ KHÔNG được chốt"
    assert engine._hold_until > time.perf_counter(), "phải mở cửa sổ giữ câu"
    assert engine._hold_reason == "VAD_SILENCE"
    assert engine._speech_active is False


def test_vad_end_khi_cau_du_dai_thi_chot_ngay(engine):
    """Câu 3 s ≥ sàn 2,5 s ⇒ chốt ngay như cũ (không giữ)."""
    _feed(engine, 3.0)
    engine._speech_active = True
    engine._speech_start_sample = 0

    engine.on_speech_end()

    reqs = _pending(engine)
    assert len(reqs) == 1 and reqs[0]["reason"] == "VAD_SILENCE"
    assert reqs[0]["start_sample"] == 0 and reqs[0]["end_sample"] == int(3.0 * SR)
    assert engine._hold_until == 0.0


def test_tat_co_thi_quay_lai_hanh_vi_cu(engine):
    """`hold_short_sentence=False` ⇒ VAD END chốt ngay dù câu rất ngắn."""
    from backend.config import config

    config.sentence.hold_short_sentence = False
    _feed(engine, 0.5)
    engine._speech_active = True
    engine._speech_start_sample = 0

    engine.on_speech_end()

    assert len(_pending(engine)) == 1
    assert engine._hold_until == 0.0


def test_cua_so_giu_bang_phan_con_thieu(engine):
    """Cửa sổ giữ = sàn − độ dài audio đã có (1,2 s ⇒ giữ thêm ~1,3 s)."""
    _feed(engine, 1.2)
    deadline = engine._hold_deadline(0, int(1.2 * SR))
    remaining = deadline - time.perf_counter()
    assert 1.0 < remaining <= 1.35, remaining


# ------------------------------------------------------------------ nhánh 2: hết hạn ⇒ chốt
def test_het_cua_so_thi_chot_phan_da_co(engine):
    _feed(engine, 1.2)
    engine._speech_active = True
    engine._speech_start_sample = 0
    engine.on_speech_end()
    assert _pending(engine) == []

    # Mô phỏng đồng hồ đã qua mốc giữ (không phải chờ thật 1,3 s).
    engine._hold_until = time.perf_counter() - 0.01
    assert engine._flush_held_commit() is True

    reqs = _pending(engine)
    assert len(reqs) == 1, reqs
    assert reqs[0]["reason"] == "VAD_SILENCE"
    assert reqs[0]["start_sample"] == 0
    assert reqs[0]["end_sample"] == int(1.2 * SR)
    assert engine._hold_until == 0.0 and engine._hold_reason == ""


def test_flush_khi_khong_giu_thi_khong_lam_gi(engine):
    assert engine._flush_held_commit() is False
    assert _pending(engine) == []


def test_khong_chot_hai_lan_sau_khi_het_cua_so(engine):
    """Sau khi chốt do hết cửa sổ, VAD END lần nữa (không có START) không được chốt thêm."""
    _feed(engine, 1.0)
    engine._speech_active = True
    engine._speech_start_sample = 0
    engine.on_speech_end()
    engine._hold_until = time.perf_counter() - 0.01
    engine._flush_held_commit()
    before = len(_pending(engine))

    engine.on_speech_end()   # VAD gọi lại END khi đang không nói

    assert len(_pending(engine)) == before


# ------------------------------------------------------------------ nhánh 3: GHÉP
def test_vad_start_trong_cua_so_thi_ghep_cung_cau(engine):
    """Nói tiếp trong cửa sổ ⇒ GHÉP: cùng utterance_id, mốc đầu câu KHÔNG đổi."""
    _feed(engine, 1.2)
    engine._speech_active = True
    engine._speech_start_sample = 0
    engine.on_speech_end()
    utt_before = engine._active_utterance_id

    engine.on_speech_start()          # người nói tiếp
    _feed(engine, 1.5)                # audio mới được ghi tiếp vào buffer

    assert engine._speech_active is True
    assert engine._hold_until == 0.0, "đã ghép thì không còn cửa sổ giữ"
    assert engine._active_utterance_id == utt_before, "ghép thì phải GIỮ nguyên câu"
    assert engine._speech_start_sample == 0, "mốc đầu câu phải giữ để không mất chữ đầu"
    assert _pending(engine) == [], "ghép thì KHÔNG chốt câu"

    # Câu ghép đã dài 2,7 s ≥ sàn ⇒ lần END này chốt bình thường, trọn cả hai đoạn.
    engine.on_speech_end()
    reqs = _pending(engine)
    assert len(reqs) == 1 and reqs[0]["start_sample"] == 0
    assert reqs[0]["end_sample"] == int(2.7 * SR), reqs[0]["end_sample"]


def test_vad_start_sau_khi_het_cua_so_thi_mo_cau_moi(engine):
    """Hết cửa sổ rồi mới nói lại ⇒ chốt phần CŨ trước, rồi mở câu mới."""
    _feed(engine, 1.2)
    engine._speech_active = True
    engine._speech_start_sample = 0
    engine.on_speech_end()
    utt_old = engine._active_utterance_id
    engine._hold_until = time.perf_counter() - 0.01     # cửa sổ đã hết

    engine.on_speech_start()

    reqs = _pending(engine)
    assert len(reqs) == 1, "phải chốt phần cũ trước khi mở câu mới"
    assert reqs[0]["utterance_id"] == utt_old
    assert engine._active_utterance_id != utt_old
    assert engine._speech_active is True
    assert engine._hold_until == 0.0


def test_start_binh_thuong_khong_bi_anh_huong(engine):
    """Không có cửa sổ giữ ⇒ START vẫn reset câu mới như cũ."""
    engine._speech_active = False
    engine._hold_until = 0.0
    _feed(engine, 1.0)
    engine._speech_start_sample = 0
    utt = engine._active_utterance_id

    engine.on_speech_start()

    assert engine._active_utterance_id != utt
    assert engine._speech_start_sample == -1 and engine._awaiting_pre_roll is True


# ------------------------------------------------------------------ dọn state
def test_tua_video_thi_bo_cua_so_giu(engine):
    import asyncio

    _feed(engine, 1.0)
    engine._speech_active = True
    engine._speech_start_sample = 0
    engine.on_speech_end()
    assert engine._hold_until > 0

    asyncio.run(engine.reset_stream("seek"))

    assert engine._hold_until == 0.0 and engine._hold_reason == ""
    assert _pending(engine) == [], "không được chốt câu của đoạn CŨ sau khi tua"


def test_api_config_va_ws_co_co_hold(restore_config):
    from backend.main import _build_config_response
    from backend.ws.session import SessionConfigPayload

    resp = _build_config_response(include_catalog=False)
    assert resp["stable"]["hold_short_sentence"] is True
    field = SessionConfigPayload.model_fields.get("hold_short_sentence")
    assert field is not None and field.alias == "holdShortSentence"


def test_mac_dinh_bat(restore_config):
    from backend.config import config

    assert config.sentence.hold_short_sentence is True
