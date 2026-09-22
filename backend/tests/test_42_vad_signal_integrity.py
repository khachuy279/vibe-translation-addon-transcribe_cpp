"""Toàn vẹn tín hiệu: `plugin → VAD → ASR` không bị suy hao/đổi/méo (tầng A, không model).

Đây là bài test chốt yêu cầu gắt nhất của kế hoạch
(`report/audit/19_KE_HOACH_VIET_LAI_VAD.md` §3):

1. Mọi frame VAD chuyển tiếp là **byte nguyên bản** client gửi, LIỀN MẠCH, không lặp/mất.
2. Engine VAD chỉ nhận **view chỉ-đọc** của đúng buffer đó (không thể sửa audio của ASR).
3. Buffer ASR nhận đúng chuỗi byte đó (kiểm tra qua `TranscribeEngine.feed_audio`), và
   chuyển Int16 → Float32 `/32768` là hoàn nguyên được bit-exact (chia luỹ thừa 2).
4. `silence_duration_ms = 0/None` ⇒ engine dùng mặc định docs; `> 0` ⇒ nó là **điều kiện
   số 1** để chốt END (processor là bên quyết định, ghi đè `min_silence_frame`…).
"""

import numpy as np
import pytest

from backend.tests.fakes import (
    FakeVADEngine,
    make_silence_pcm,
    make_speech_pcm,
    pcm_to_int16_bytes,
)


def _feed_all(proc, payload: bytes, chunk_bytes: int = 640) -> None:
    for i in range(0, len(payload), chunk_bytes):
        proc.feed_chunk(payload[i:i + chunk_bytes])


def test_byte_chuyen_tiep_nguyen_ban_khi_chunk_lech_bien(session_factory):
    """Chunk client gửi lệch biên frame ⇒ vẫn phải chuyển tiếp byte NGUYÊN BẢN, liền mạch."""
    session = session_factory()
    proc = session.vad_processor

    forwarded: list[bytes] = []
    proc.on_speech_chunk = lambda b, ts, tag: forwarded.append(bytes(b))

    pcm = np.concatenate([
        make_silence_pcm(0.3),
        make_speech_pcm(1.0, amplitude=0.3),
        make_silence_pcm(1.0),
    ])
    payload = pcm_to_int16_bytes(pcm)

    # 640 byte = 320 mẫu: KHÔNG chia hết cho frame 400 mẫu của FakeVADEngine ⇒ frame phải
    # vắt qua nhiều chunk, đúng tình huống thật của Silero (512) và FSMN (960).
    _feed_all(proc, payload, chunk_bytes=640)

    assert forwarded, "phải có frame được chuyển tiếp"
    joined = b"".join(forwarded)
    pos = payload.find(joined)
    assert pos >= 0, "audio chuyển tiếp không phải slice liền mạch của PCM gốc"
    assert pos % 2 == 0

    # Không nhân bản: nối lại phải bằng CHÍNH slice đó, không có frame lặp ở giữa.
    assert payload[pos:pos + len(joined)] == joined
    # Và phải phủ phần tiếng nói (không bị cắt cụt).
    speech_start = int(0.3 * 16000) * 2
    assert pos <= speech_start
    assert pos + len(joined) >= int(1.3 * 16000) * 2


def test_engine_chi_nhan_view_chi_doc(session_factory):
    """Engine không được phép (và không thể) sửa audio trên đường tới ASR."""
    session = session_factory()
    proc = session.vad_processor

    forwarded: list[bytes] = []
    proc.on_speech_chunk = lambda b, ts, tag: forwarded.append(bytes(b))

    original = proc._engine.is_speech
    seen: list[np.ndarray] = []

    def _spy(frame, state, threshold=None, silence_ms=None):
        seen.append(frame)
        assert not frame.flags.writeable, (
            "frame đưa cho engine phải là view CHỈ-ĐỌC (byte đó còn đi thẳng tới ASR)"
        )
        return original(frame, state, threshold=threshold, silence_ms=silence_ms)

    proc._engine.is_speech = _spy  # type: ignore[method-assign]

    payload = pcm_to_int16_bytes(np.concatenate([
        make_speech_pcm(0.5, amplitude=0.3), make_silence_pcm(1.0),
    ]))
    _feed_all(proc, payload)

    assert seen, "engine phải nhận frame"
    assert forwarded, "phải có frame được chuyển tiếp"
    # Byte engine thấy == byte ASR nhận (không có bản sao trung gian bị biến đổi).
    joined = b"".join(forwarded)
    engine_bytes = b"".join(f.tobytes() for f in seen)
    assert joined in engine_bytes, "chuỗi byte chuyển tiếp phải nằm trong chuỗi byte engine thấy"


def test_float32_ingress_duoc_chuyen_mot_lan_duy_nhat(session_factory):
    """Đường `np.float32` (test/harness) được lượng tử hoá ĐÚNG MỘT LẦN ở cổng vào."""
    session = session_factory()
    proc = session.vad_processor

    forwarded: list[bytes] = []
    proc.on_speech_chunk = lambda b, ts, tag: forwarded.append(bytes(b))

    pcm = np.concatenate([make_speech_pcm(0.6, amplitude=0.3), make_silence_pcm(1.0)])
    expected = pcm_to_int16_bytes(pcm)  # cùng công thức 32767.0 mà processor dùng

    for i in range(0, len(pcm), 320):
        proc.feed_chunk(pcm[i:i + 320])

    joined = b"".join(forwarded)
    assert joined, "phải có frame được chuyển tiếp"
    assert expected.find(joined) >= 0, (
        "byte sau khi vào cổng float32 không khớp bản lượng tử hoá chuẩn ⇒ có chuyển đổi thêm"
    )


def test_asr_buffer_nhan_dung_byte_va_hoan_nguyen_bit_exact(session_factory):
    """`TranscribeEngine.feed_audio` nhận đúng chuỗi byte VAD phát ra (bit-exact)."""
    session = session_factory()
    engine = session.asr_engine
    proc = session.vad_processor

    received: list[bytes] = []
    original = engine.feed_audio

    def _recorder(frame_bytes: bytes, timestamp: float = 0.0, vad_state: str = "") -> None:
        # Ghi lại ĐÚNG byte mà VAD chuyển cho ASR rồi mới đưa vào đúng đường thật.
        received.append(bytes(frame_bytes))
        return original(frame_bytes, timestamp, vad_state)

    proc.on_speech_chunk = _recorder

    pcm = np.concatenate([
        make_silence_pcm(0.2),
        make_speech_pcm(0.8, amplitude=0.3),
        make_silence_pcm(1.0),
    ])
    payload = pcm_to_int16_bytes(pcm)
    _feed_all(proc, payload)

    assert received, "ASR phải nhận được audio"
    joined = b"".join(received)
    assert payload.find(joined) >= 0, "audio ASR nhận không phải slice liền mạch của PCM gốc"

    # Buffer ASR lưu Float32 = Int16/32768 → hoàn nguyên phải BIT-EXACT (chia luỹ thừa 2).
    written = engine.audio_buffer.get_slice(0, len(joined) // 2)
    restored = np.clip(np.round(np.asarray(written, dtype=np.float32) * 32768.0), -32768, 32767)
    assert restored.astype(np.int16).tobytes() == joined, (
        "Int16 -> Float32 (/32768) -> Int16 không hoàn nguyên được ⇒ audio bị sai lệch mẫu"
    )


def test_khong_doi_audio_khi_doi_engine_giua_stream(session_factory):
    """Đổi engine giữa stream vẫn không làm biến đổi byte đã chuyển tiếp."""
    from backend.tests.fakes import FakeVADEngine

    session = session_factory()
    proc = session.vad_processor
    forwarded: list[bytes] = []
    proc.on_speech_chunk = lambda b, ts, tag: forwarded.append(bytes(b))

    pcm = np.concatenate([make_speech_pcm(0.6, amplitude=0.3), make_silence_pcm(0.6)])
    payload = pcm_to_int16_bytes(pcm)
    _feed_all(proc, payload[: len(payload) // 2])

    proc._engine = FakeVADEngine(rms_threshold=0.05)   # engine "khác" sau khi đổi
    proc._attach_engine(proc._engine)
    proc._state = proc._new_state(proc._engine)
    _feed_all(proc, payload[len(payload) // 2:])

    if forwarded:
        joined = b"".join(forwarded)
        assert payload.find(joined) >= 0


@pytest.mark.parametrize("value", [0, -1, None])
def test_silence_0_hoac_am_nghia_la_mac_dinh_engine(value):
    """`silence_duration_ms` = 0/âm/None ⇒ `None` = để VAD dùng mặc định trong docs."""
    from backend.vad.processor import VADProcessor

    proc = VADProcessor(vad_engine="firered-vad", silence_duration_ms=value)
    assert proc.silence_duration_ms is None


def test_prewarm_giu_dung_maxlen_pre_roll(monkeypatch):
    """Hồi quy: `prewarm()` phải gắn engine TRƯỚC khi dựng state.

    Bản đầu dựng state trước nên `deque(maxlen=self._max_lookback_frames)` bị chốt
    `maxlen=1` ⇒ mất gần hết pre-roll (đầu câu bị cắt) trên ĐÚNG đường khởi động thật
    (`main._prewarm_vad_default`, `auto_load=False`).
    """
    from backend.vad.engines import VADEngineFactory
    from backend.vad.processor import VADProcessor
    from backend.tests.fakes import FakeVADEngine

    class _FakeVADEngineLonLookback(FakeVADEngine):
        max_lookback_frames = 3

    fake = _FakeVADEngineLonLookback()
    monkeypatch.setattr(VADEngineFactory, "get_engine", classmethod(lambda cls, n: fake))
    monkeypatch.setattr(VADEngineFactory, "peek_engine", classmethod(lambda cls, n: None))

    proc = VADProcessor(vad_engine="firered-vad", auto_load=False)
    assert proc._engine is None

    proc.prewarm()

    assert proc._engine is fake
    assert proc._max_lookback_frames == 3
    assert proc._state.pre_roll.maxlen == 3, (
        "pre-roll bị chốt sai maxlen ⇒ mất đầu câu ngay từ đường prewarm lúc khởi động"
    )
    proc.reset()
    assert proc._state.pre_roll.maxlen == 3, "state sau `reset()` cũng phải giữ đúng maxlen"


# ─────────────────────────── VAD Silence = 0 => docs; > 0 => điều kiện số 1 ──────────


class _EarlyEndVADEngine(FakeVADEngine):
    """Engine 'hỏng': tự chốt END sau đúng 1 frame im lặng, KHÔNG tôn trọng `silence_ms`.

    Dùng để chứng minh ở chế độ ghi đè, quyết định của processor thắng quyết định của engine.
    """

    def is_speech(self, frame_int16, state, threshold=None, silence_ms=None):
        return super().is_speech(frame_int16, state, threshold=threshold, silence_ms=25)


def _do_override(engine, *, silence_ms, speech_sec: float = 1.0, silence_sec: float = 2.0):
    """Chạy 1 phiên với engine cho trước; trả (mẫu cuối có bằng chứng, mẫu chốt END, lý do)."""
    from backend.vad.processor import VADProcessor
    from backend.tests.fakes import make_silence_pcm, make_speech_pcm, pcm_to_int16_bytes

    proc = VADProcessor(
        vad_engine="firered-vad",
        threshold=None,
        silence_duration_ms=silence_ms,
        engine_override=engine,
        auto_load=False,
    )
    recorded = {"last_evidence": 0, "end_sample": None, "reason": None, "samples": 0}

    original = engine.is_speech

    def _wrapped(frame, state, threshold=None, silence_ms=None):
        res = original(frame, state, threshold=threshold, silence_ms=silence_ms)
        recorded["samples"] += len(frame)
        if res.is_speech:
            recorded["last_evidence"] = recorded["samples"]
        return res

    engine.is_speech = _wrapped  # type: ignore[method-assign]

    original_close = proc._close_utterance

    def _close_spy(state, callbacks, *, reason, detail, prob):
        recorded["end_sample"] = state.total_samples_processed
        recorded["reason"] = reason
        return original_close(state, callbacks, reason=reason, detail=detail, prob=prob)

    proc._close_utterance = _close_spy  # type: ignore[method-assign]

    pcm = np.concatenate([
        make_silence_pcm(0.3),
        make_speech_pcm(speech_sec, amplitude=0.3),
        make_silence_pcm(silence_sec),
    ])
    payload = pcm_to_int16_bytes(pcm)
    _feed_all(proc, payload, chunk_bytes=640)
    return recorded


def test_silence_duong_la_dieu_kien_so_1_chot_end():
    """`silence_duration_ms > 0` ⇒ END đúng sau `silence_duration_ms` im lặng."""
    from backend.tests.fakes import FakeVADEngine

    hop = FakeVADEngine.frame_samples
    for silence_ms in (300, 700, 1400):
        rec = _do_override(FakeVADEngine(), silence_ms=silence_ms)
        assert rec["end_sample"] is not None, f"silence={silence_ms}: không chốt được câu"
        silent_ms = (rec["end_sample"] - rec["last_evidence"]) / 16000 * 1000
        assert abs(silent_ms - silence_ms) <= 2 * hop / 16000 * 1000, (
            f"silence={silence_ms}: END ở {silent_ms:.0f} ms im lặng (lệch quá 2 frame)"
        )
        # Với engine TỐT, hai điều kiện trùng frame: engine cũng END đúng lúc đó nên lý do
        # có thể là `engine_end`. Điều bắt buộc là THỜI ĐIỂM (kiểm ở trên).
        assert rec["reason"] in ("engine_end", "silence_override")


def test_silence_duong_ghi_de_end_som_cua_engine():
    """END sớm của engine bị BỎ QUA khi popup đặt VAD Silence > 0 (điều kiện số 1)."""
    hop = _EarlyEndVADEngine.frame_samples
    silence_ms = 700

    rec = _do_override(_EarlyEndVADEngine(), silence_ms=silence_ms)
    silent_ms = (rec["end_sample"] - rec["last_evidence"]) / 16000 * 1000
    assert rec["reason"] == "silence_override", (
        "engine tự END sớm nhưng processor phải chờ đủ silence_duration_ms "
        f"(lý do nhận được: {rec['reason']})"
    )
    assert silent_ms >= silence_ms, (
        f"câu bị chốt sau {silent_ms:.0f} ms im lặng — sớm hơn yêu cầu {silence_ms} ms"
    )
    assert silent_ms <= silence_ms + 2 * hop / 16000 * 1000


class _EndWithSpeechEvidenceEngine(FakeVADEngine):
    """Engine phát END trên frame VẪN CÒN bằng chứng nói — ca thật của FireRed.

    Log phim thật 2026-09-22 22:15: `[VAD] END [engine_end, im lặng 0ms >= 1500ms] (p=1.00)`
    xảy ra GIỮA CÂU "Don't try and trick me into buying something I don't want." ⇒ câu bị
    chẻ đôi. Nguyên nhân: processor cũ coi `END + is_speech=True` là "cắt cưỡng bức".
    """

    def is_speech(self, frame_int16, state, threshold=None, silence_ms=None):
        res = super().is_speech(frame_int16, state, threshold=threshold, silence_ms=silence_ms)
        session = state.engine_state
        # Ép phát END ở frame có tiếng nói thứ 3 (SAU frame START) và VẪN giữ is_speech=True.
        if res.is_speech and res.event is None and state.is_speech:
            session._speech_frames = getattr(session, "_speech_frames", 0) + 1
            if session._speech_frames >= 3:
                res.event = "END"
                res.probability = 1.0
                res.is_speech = True     # frame chuyển tiếp: VẪN còn bằng chứng nói
        return res


def test_end_kem_bang_chung_noi_KHONG_duoc_cat_giua_cau():
    """FIX-13: `END` + `is_speech=True` KHÔNG phải cắt cưỡng bức ⇒ phải chờ đủ silence.

    Đây chính là lỗi đã chẻ đôi câu trong log phim thật (`im lặng 0ms >= 1500ms`).
    """
    silence_ms = 700
    hop = _EndWithSpeechEvidenceEngine.frame_samples
    rec = _do_override(_EndWithSpeechEvidenceEngine(), silence_ms=silence_ms)
    assert rec["reason"] == "silence_override", (
        f"phải chốt bằng đồng hồ im lặng, nhận được: {rec['reason']}"
    )
    silent_ms = (rec["end_sample"] - rec["last_evidence"]) / 16000 * 1000
    assert silent_ms >= silence_ms, (
        f"câu bị chốt sau {silent_ms:.0f} ms im lặng — sớm hơn yêu cầu {silence_ms} ms"
    )
    assert silent_ms <= silence_ms + 2 * hop / 16000 * 1000


def test_end_kem_bang_chung_noi_o_che_do_docs_van_ton_trong_engine():
    """Chế độ docs (`silence_duration_ms=None`): END của engine vẫn là nguồn sự thật."""
    rec = _do_override(_EndWithSpeechEvidenceEngine(), silence_ms=None)
    assert rec["reason"] == "engine_end"


def test_forced_flag_moi_duoc_cat_cuong_buc():
    """Chỉ khi engine đánh dấu `forced=True` processor mới cắt ngay giữa lúc đang nói."""
    class _ForcedEngine(_EndWithSpeechEvidenceEngine):
        def is_speech(self, frame_int16, state, threshold=None, silence_ms=None):
            res = super().is_speech(frame_int16, state, threshold=threshold, silence_ms=silence_ms)
            if res.event == "END":
                res.forced = True
            return res

    rec = _do_override(_ForcedEngine(), silence_ms=700)
    assert rec["reason"] == "engine_end"
    silent_ms = (rec["end_sample"] - rec["last_evidence"]) / 16000 * 1000
    assert silent_ms < 700, "cắt cưỡng bức phải xảy ra NGAY, không chờ đủ im lặng"


def test_silence_bang_0_thi_engine_quyet_dinh():
    """`silence_duration_ms = None` ⇒ tôn trọng END của engine (mặc định docs của nó)."""
    rec = _do_override(_EarlyEndVADEngine(), silence_ms=None)
    assert rec["reason"] == "engine_end"
    silent_ms = (rec["end_sample"] - rec["last_evidence"]) / 16000 * 1000
    # Engine giả này chốt sau đúng 1 frame (25 ms) im lặng.
    assert silent_ms <= 2 * _EarlyEndVADEngine.frame_samples / 16000 * 1000


def test_gat_silence_ve_0_giua_cau_khong_treo_cau():
    """Edge case: đang ghi đè (700 ms) mà người dùng gạt VAD Silence về 0 giữa câu.

    Engine đã phát END sớm và bị hoãn (`engine_end_pending`); khi chế độ đổi về docs, frame
    kế tiếp phải tôn trọng quyết định đó — nếu không, câu sẽ treo vĩnh viễn.
    """
    from backend.vad.processor import VADProcessor
    from backend.tests.fakes import make_silence_pcm, make_speech_pcm, pcm_to_int16_bytes

    proc = VADProcessor(
        vad_engine="firered-vad",
        silence_duration_ms=700,
        engine_override=_EarlyEndVADEngine(),
        auto_load=False,
    )
    ends = []
    proc.on_speech_end = lambda: ends.append(proc._state.total_samples_processed)
    proc.on_speech_chunk = lambda *_: None

    speech = pcm_to_int16_bytes(make_speech_pcm(0.8, amplitude=0.3))
    silence = pcm_to_int16_bytes(make_silence_pcm(1.5))
    _feed_all(proc, speech)
    assert proc._state.is_speech is True

    # 0,2 s im lặng: engine đã muốn END (mặc định nội bộ 25 ms) nhưng bị hoãn vì < 700 ms.
    _feed_all(proc, silence[: int(0.2 * 16000) * 2])
    assert not ends, "chưa đủ 700 ms im lặng thì chưa được chốt"
    assert proc._state.is_speech is True
    assert proc._state.engine_end_pending is True

    # Người dùng gạt VAD Silence về 0 ⇒ chế độ docs; frame im lặng kế tiếp phải chốt câu.
    proc.update_config(silence_duration_ms=0)
    _feed_all(proc, silence[int(0.2 * 16000) * 2:])
    assert ends, "gạt về 0 mà câu vẫn treo ⇒ quyết định END của engine bị mất"
    assert proc._state.is_speech is False
