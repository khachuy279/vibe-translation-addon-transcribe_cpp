"""Test tầng A — FIX-06 / FIX-07 / FIX-14: copy, RAM và config có tác dụng thật.

- FIX-06: `CircularAudioBuffer.clear()` KHÔNG zero-fill 3,84 MB mỗi lần reset.
- FIX-07: `AudioBufferConfig.capacity_sec` phải là source of truth cho ASR engine.
- FIX-14: `config.ws.max_payload_bytes` phải được dùng thật (không còn config chết).
"""

from pathlib import Path

import numpy as np
import pytest

from backend.config import config
from backend.core.audio_buffer import CircularAudioBuffer

ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────── FIX-06
def test_clear_khong_zero_fill_va_khong_tra_audio_cu():
    """Sau `clear()`, buffer phải coi như rỗng — nhưng KHÔNG ghi 3,84 MB số 0."""
    buf = CircularAudioBuffer(sample_rate=16000, capacity_sec=1.0)
    buf.write(np.ones(16000, dtype=np.float32) * 0.5)
    assert buf.total_written == 16000

    buf.clear()

    assert buf.total_written == 0
    assert buf.dropped_samples == 0
    # Mọi đường đọc phải trả RỖNG, tuyệt đối không lộ audio cũ.
    assert buf.get_slice(0, 100).size == 0
    assert buf.get_slice(-500, 100).size == 0
    assert buf.get_recent(0.5).size == 0

    # Ghi lại sau clear phải bắt đầu từ vị trí 0 và dữ liệu phải ĐÚNG.
    buf.write(np.ones(100, dtype=np.float32) * 0.25)
    got = buf.get_slice(0, 100)
    assert got.size == 100
    assert np.allclose(got, 0.25)


def test_clear_khong_ghi_de_toan_bo_mang(monkeypatch):
    """Chốt ở mức mã nguồn: `clear()` không được gọi `self._buffer.fill(...)`."""
    src = (ROOT / "backend" / "core" / "audio_buffer.py").read_text(encoding="utf-8")
    idx = src.index("def clear(self)")
    body = src[idx:idx + 1500]
    assert "_buffer.fill(" not in body, "clear() không được zero-fill toàn bộ buffer"
    assert "_total_written = 0" in body


def test_write_bytes_int16_giong_het_cach_chia_cu():
    """FIX-06b: chia tại chỗ phải cho KẾT QUẢ Y HỆT (bit-exact) cách cũ."""
    pcm = np.array([-32768, -1, 0, 1, 12345, 32767], dtype=np.int16)
    expected = pcm.astype(np.float32) / 32768.0

    buf = CircularAudioBuffer(sample_rate=16000, capacity_sec=1.0)
    assert buf.write_bytes(pcm.tobytes(), dtype="int16") == len(pcm)
    got = buf.get_slice(0, len(pcm))
    assert np.array_equal(got, expected), f"{got} != {expected}"


# ─────────────────────────────────────────────── FIX-07
def test_engine_audio_buffer_doc_capacity_tu_config(restore_config):
    """`AudioBufferConfig.capacity_sec` phải điều khiển buffer của ASR engine."""
    from backend.asr.engine import TranscribeEngine

    config.audio_buffer.capacity_sec = 12.5
    config.audio_buffer.sample_rate = 16000

    engine = TranscribeEngine()
    assert engine.audio_buffer.sample_rate == 16000
    assert engine.audio_buffer.capacity_samples == int(16000 * 12.5)


def test_engine_audio_buffer_khong_hardcode_60s():
    """Chốt ở mức mã nguồn: không được hard-code `capacity_sec=60.0`."""
    src = (ROOT / "backend" / "asr" / "engine.py").read_text(encoding="utf-8")
    assert "CircularAudioBuffer(sample_rate=16000, capacity_sec=60.0)" not in src
    assert "config.audio_buffer" in src


# ─────────────────────────────────────────────── FIX-14
def test_max_payload_bytes_duoc_dung_that():
    """`max_payload_bytes` phải được `_handle_binary_message` đọc (trước đây là config chết)."""
    src = (ROOT / "backend" / "ws" / "handler.py").read_text(encoding="utf-8")
    idx = src.index("async def _handle_binary_message")
    body = src[idx:idx + 1200]
    assert "max_payload_bytes" in body
    assert "payload_rejected" in body


def test_khung_qua_kho_thi_bi_bo(session_factory, restore_config, monkeypatch):
    """Khung nhị phân vượt trần ⇒ bỏ, KHÔNG đẩy vào VAD/ASR."""
    import asyncio

    from backend.ws import handler as handler_mod

    session = session_factory()
    config.ws.max_payload_bytes = 64

    pushed = []

    def _fake_process(sess, data):
        pushed.append(len(data))

    monkeypatch.setattr(handler_mod, "_process_binary_chunk", _fake_process)

    before = handler_mod.metrics_collector.get_counter("ws.payload_rejected")

    asyncio.run(handler_mod._handle_binary_message(session, b"x" * 128))
    assert pushed == [], "khung quá khổ không được đẩy vào pipeline"

    asyncio.run(handler_mod._handle_binary_message(session, b"x" * 32))
    assert pushed == [32], "khung hợp lệ vẫn phải đi qua"

    assert handler_mod.metrics_collector.get_counter("ws.payload_rejected") == before + 1
