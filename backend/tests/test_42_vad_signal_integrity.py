"""Toàn vẹn tín hiệu: `plugin → VAD → ASR` không bị suy hao/đổi/méo (tầng A, không model).

Đây là bài test chốt yêu cầu gắt nhất của kế hoạch
(`report/audit/19_KE_HOACH_VIET_LAI_VAD.md` §3):

1. Mọi frame VAD chuyển tiếp là **byte nguyên bản** client gửi, LIỀN MẠCH, không lặp/mất.
2. Engine VAD chỉ nhận **view chỉ-đọc** của đúng buffer đó (không thể sửa audio của ASR).
3. Buffer ASR nhận đúng chuỗi byte đó (kiểm tra qua `TranscribeEngine.feed_audio`), và
   chuyển Int16 → Float32 `/32768` là hoàn nguyên được bit-exact (chia luỹ thừa 2).
"""

import numpy as np
import pytest

from backend.tests.fakes import make_silence_pcm, make_speech_pcm, pcm_to_int16_bytes


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
