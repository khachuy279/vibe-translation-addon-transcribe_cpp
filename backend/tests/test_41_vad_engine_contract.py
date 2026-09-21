"""Hợp đồng của tầng VAD sau khi viết lại (tầng B — cần model thật).

Chốt các bảo đảm của `report/audit/19_KE_HOACH_VIET_LAI_VAD.md`:

1. Mỗi engine công bố đúng hình học frame (`frame_samples`, `max_lookback_frames`).
2. Engine phát `START`/`END` và processor chốt câu bằng CHÍNH event đó.
3. Audio chuyển tiếp là **byte nguyên bản** của client, LIỀN MẠCH, không mất đầu câu,
   không lặp frame.
4. Knob chung (`threshold`, `silence_duration_ms`) được chiếu đúng xuống field native.

Dùng audio tiếng nói THẬT (`wav_test/Russian_4s.wav`) vì VAD học trên speech: sóng sin
thuần bị cả 3 model coi là không phải tiếng nói.

Chạy: `pytest -m slow backend/tests/test_41_vad_engine_contract.py -q`
"""

from pathlib import Path
import time

import numpy as np
import pytest
import soundfile as sf

from backend.config import config
from backend.vad.engines import VADEngineFactory
from backend.vad.processor import VADStreamProcessor

ENGINES = ["firered-vad", "silero-vad", "fsmn-vad"]

_HOP_SAMPLES = {"firered-vad": 160, "silero-vad": 512, "fsmn-vad": 960}

_SILENCE_HEAD_SEC = 0.5
_SILENCE_TAIL_SEC = 2.0
_CHUNK_SEC = 0.02               # client gửi từng 20 ms

_SPEECH_WAV = Path(__file__).resolve().parent.parent.parent / "wav_test" / "Russian_4s.wav"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def stream() -> dict:
    """PCM Int16 (head lặng + speech thật + tail lặng) + vùng "lõi tiếng nói".

    Ground truth ở đây KHÔNG phải nhãn ngữ nghĩa mà là **vùng năng lượng lõi**: các frame
    10 ms có RMS > 0.5 × đỉnh. Đó là proxy ổn định cho "chỗ chắc chắn có tiếng nói" mà mọi
    VAD đều phải giữ, dùng để kiểm tra không mất đầu/đuôi câu.
    """
    x, sr = sf.read(str(_SPEECH_WAV), dtype="int16")
    assert sr == 16000
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.int16)

    head = np.zeros(int(16000 * _SILENCE_HEAD_SEC), dtype=np.int16)
    tail = np.zeros(int(16000 * _SILENCE_TAIL_SEC), dtype=np.int16)
    pcm = np.concatenate([head, x, tail])

    win = 160  # 10 ms
    n_win = len(pcm) // win
    env = np.sqrt((pcm[: n_win * win].reshape(n_win, win).astype(np.float32) ** 2).mean(axis=1))
    peak = float(env.max())
    core = np.flatnonzero(env > 0.5 * peak)

    return {
        "pcm": pcm,
        "bytes": pcm.tobytes(),
        "core_start": int(core[0]) * win,
        "core_end": int(core[-1] + 1) * win,
        "total_energy": float((pcm.astype(np.float32) ** 2).sum()),
        "samples": len(pcm),
    }


class _Recorder:
    """Ghi lại mọi callback của VADProcessor để kiểm tra hợp đồng."""

    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.chunks = []      # (bytes, ts, tag)
        self.results = []     # VADResult của từng frame (theo thứ tự engine gọi)

    @property
    def forward_bytes(self) -> bytes:
        return b"".join(chunk for chunk, _, _ in self.chunks)

    @property
    def forward_samples(self) -> int:
        return len(self.forward_bytes) // 2

    @property
    def start_index(self) -> int:
        """Chỉ số frame (0-based) mà engine phát START."""
        for i, res in enumerate(self.results):
            if res.event == "START":
                return i
        return -1

    @property
    def requested_lookback(self) -> int:
        for res in self.results:
            if res.event == "START":
                return int(res.lookback_frames)
        return 0


def _run(engine_name: str, pcm_bytes: bytes, *, threshold=None, silence_ms=None):
    """Nạp PCM theo từng chunk 20 ms kèm capture_timestamp tăng dần (như client thật)."""
    rec = _Recorder()
    proc = VADStreamProcessor(
        vad_engine=engine_name,
        threshold=threshold,
        silence_duration_ms=silence_ms,
        on_speech_start=lambda: setattr(rec, "starts", rec.starts + 1),
        on_speech_chunk=lambda b, ts, tag: rec.chunks.append((b, ts, tag)),
        on_speech_end=lambda: setattr(rec, "ends", rec.ends + 1),
    )
    proc.prewarm()

    # Bọc `is_speech` để ghi lại VADResult của từng frame (kiểm tra engine "xin" bao nhiêu
    # frame pre-roll mà KHÔNG cần đọc ruột engine).
    original = proc._engine.is_speech

    def _wrapped(frame, state, threshold=None, silence_ms=None):
        res = original(frame, state, threshold=threshold, silence_ms=silence_ms)
        rec.results.append(res)
        return res

    proc._engine.is_speech = _wrapped  # type: ignore[method-assign]

    chunk_bytes = int(16000 * _CHUNK_SEC) * 2
    for i in range(0, len(pcm_bytes), chunk_bytes):
        capture_ts = (i / 2) / 16000.0
        proc.feed_chunk(pcm_bytes[i:i + chunk_bytes], capture_timestamp=capture_ts)
    return proc, rec


# ─────────────────────────────────────────────── 0. config == docs (tầng A, không model)


def test_config_3_engine_khop_docs():
    """Khóa yêu cầu "mỗi VAD có cấu hình GIỐNG docs" vào CI.

    So trực tiếp với nguồn chân lý của từng thư viện:
      * `fireredvad.FireRedStreamVadConfig` (dataclass) — riêng `speech_threshold` là ghi đè
        CÓ CHỦ ĐÍCH theo ví dụ CLI/README (0.4 thay vì 0.5 của dataclass);
      * `silero_vad.VADIterator.__init__` (signature), `sampling_rate` không phơi ra vì
        pipeline cố định 16 kHz;
      * `config.yaml` của checkpoint `funasr/fsmn-vad` (VADXOptions).

    Không cần nạp model ⇒ chạy ở tầng A (không có marker `slow`).
    """
    import dataclasses
    import inspect

    import yaml

    from backend.config import config as app_config

    # ── FireRed ──────────────────────────────────────────────────────────────
    from fireredvad import FireRedStreamVadConfig

    docs = {f.name: f.default for f in dataclasses.fields(FireRedStreamVadConfig)}
    ours = app_config.vad.firered.model_dump()
    assert set(ours) == set(docs), f"tập field khác docs: {set(ours) ^ set(docs)}"
    for name, default in docs.items():
        if name == "speech_threshold":
            assert ours[name] == 0.4, (
                "`speech_threshold` là ghi đè có chủ đích theo ví dụ CLI/README của "
                f"fireredvad (0.4); nếu đổi, phải cập nhật cả ghi chú trong config.py"
            )
            continue
        assert ours[name] == default, f"FireRed.{name}: backend={ours[name]!r} docs={default!r}"

    # ── Silero ───────────────────────────────────────────────────────────────
    from silero_vad import VADIterator

    params = {
        k: v.default
        for k, v in inspect.signature(VADIterator.__init__).parameters.items()
        if k != "self" and v.default is not inspect.Parameter.empty
    }
    ours = app_config.vad.silero.model_dump()
    assert set(ours) == set(params) - {"model", "sampling_rate"}, (
        f"tập field khác docs: {set(ours) ^ (set(params) - {'model', 'sampling_rate'})}"
    )
    for name, value in ours.items():
        assert value == params[name], f"Silero.{name}: backend={value!r} docs={params[name]!r}"
    assert app_config.vad.sample_rate == 16000, (
        "VADIterator chỉ nhận 8000/16000; pipeline phải giữ đúng 16 kHz"
    )

    # ── FSMN ─────────────────────────────────────────────────────────────────
    ckpt = yaml.safe_load(
        (PROJECT_ROOT / "backend" / "models" / "fsmn_vad" / "config.yaml").read_text(encoding="utf-8")
    )
    docs = ckpt["model_conf"]
    ours = app_config.vad.fsmn.model_dump()
    for name in ("max_end_silence_time", "speech_noise_thres", "speech_to_sil_time_thres",
                 "sil_to_speech_time_thres", "window_size_ms", "lookback_time_start_point",
                 "lookahead_time_end_point", "output_frame_probs"):
        assert ours[name] == docs[name], f"FSMN.{name}: backend={ours[name]!r} docs={docs[name]!r}"
    assert ours["chunk_size_ms"] == 60, (
        "`chunk_size_ms` = mặc định của wrapper streaming chính thức "
        "`funasr.models.fsmn_vad_streaming.dynamic_vad.DynamicStreamingVAD`"
    )

    # ── Mặc định KHÔNG ghi đè (để engine dùng đúng docs) ─────────────────────
    assert app_config.vad.threshold is None
    assert app_config.vad.silence_duration_ms is None


# ─────────────────────────────────────────────── 1. hình học frame


@pytest.mark.slow
@pytest.mark.parametrize("engine_name", ENGINES)
def test_hinh_hoc_frame_va_pre_roll_cua_tung_engine(engine_name):
    """`frame_samples` = bước nhảy docs; `max_lookback_frames` đủ giữ pre-padding."""
    engine = VADEngineFactory.get_engine(engine_name)

    expected_hop = _HOP_SAMPLES[engine_name]
    assert engine.frame_samples == expected_hop, (
        f"{engine_name}: hop phải là {expected_hop} mẫu (docs), đang là {engine.frame_samples}"
    )
    assert engine.max_lookback_frames >= 1, "phải giữ được ít nhất 1 frame pre-roll"

    proc = VADStreamProcessor(vad_engine=engine_name)
    proc.prewarm()
    assert proc._frame_samples == expected_hop
    assert proc._frame_size_bytes == expected_hop * 2


# ─────────────────────────────────────────────── 2 + 3. event & toàn vẹn tín hiệu


@pytest.mark.slow
@pytest.mark.parametrize("engine_name", ENGINES)
def test_engine_phat_start_end_va_chuyen_tiep_byte_nguyen_ban(engine_name, stream):
    """Đúng 1 START + 1 END; audio chuyển tiếp là slice LIỀN MẠCH, byte-nguyên-bản."""
    pcm_bytes = stream["bytes"]
    t0 = time.perf_counter()
    proc, rec = _run(engine_name, pcm_bytes)
    elapsed = time.perf_counter() - t0
    audio_sec = stream["samples"] / 16000

    assert rec.starts == 1, f"{engine_name}: phải có đúng 1 START, nhận {rec.starts}"
    assert rec.ends == 1, f"{engine_name}: phải có đúng 1 END, nhận {rec.ends}"
    assert rec.forward_samples > 0, "phải chuyển tiếp audio cho ASR"

    forwarded = rec.forward_bytes
    pos = pcm_bytes.find(forwarded)
    assert pos >= 0, (
        f"{engine_name}: audio chuyển tiếp KHÔNG phải slice liền mạch của PCM gốc "
        f"(đã bị đổi thứ tự/chèn thêm/xáo trộn)"
    )
    assert pos % 2 == 0, "slice phải căn biên mẫu 16-bit"

    start_sample = pos // 2
    end_sample = start_sample + rec.forward_samples
    hop = _HOP_SAMPLES[engine_name]
    max_lookback = int(proc._max_lookback_frames)
    requested = rec.requested_lookback

    # (a) KHÔNG mất đầu câu: vùng chuyển tiếp phải bắt đầu trước lõi tiếng nói.
    assert start_sample <= stream["core_start"], (
        f"{engine_name}: bắt đầu chuyển tiếp ở mẫu {start_sample} > lõi tiếng nói "
        f"{stream['core_start']} ⇒ đã mất đầu câu"
    )
    # (b) Pre-roll phải ĐÚNG BẰNG số frame mà engine yêu cầu (không thừa, không thiếu)
    #     và không vượt trần engine tự công bố.
    assert 0 <= requested <= max_lookback, (
        f"{engine_name}: engine xin {requested} frame pre-roll > trần công bố {max_lookback}"
    )
    expected_start = (rec.start_index - requested) * hop
    assert start_sample == expected_start, (
        f"{engine_name}: frame đầu được chuyển tiếp ở mẫu {start_sample} nhưng theo yêu cầu "
        f"của engine phải là {expected_start} (START ở frame {rec.start_index}, "
        f"lookback {requested} frame)"
    )
    assert len([1 for _, _, tag in rec.chunks if tag == "PRE_ROLL"]) == requested
    # (c) Không cắt đuôi: vùng chuyển tiếp phải phủ hết lõi tiếng nói.
    assert end_sample >= stream["core_end"], (
        f"{engine_name}: kết thúc chuyển tiếp ở mẫu {end_sample} < hết lõi "
        f"{stream['core_end']} ⇒ đã mất đuôi câu"
    )
    # (d) Không mất năng lượng: vùng chuyển tiếp phải chứa ~toàn bộ năng lượng tín hiệu.
    seg = stream["pcm"][start_sample:end_sample].astype(np.float32)
    coverage = float((seg ** 2).sum()) / max(1e-9, stream["total_energy"])
    assert coverage >= 0.99, (
        f"{engine_name}: chỉ phủ {coverage * 100:.2f}% năng lượng tín hiệu ⇒ có đoạn nói bị bỏ"
    )
    # (e) Không gửi nhiều hơn tổng độ dài stream (không nhân bản frame).
    assert rec.forward_samples <= stream["samples"]

    tags = [tag for _, _, tag in rec.chunks]
    assert "SPEECH" in tags
    if requested:
        assert "PRE_ROLL" in tags, f"{engine_name}: không có frame pre-roll nào được xả"

    print(
        f"[{engine_name}] forward {rec.forward_samples / 16000:.2f}s "
        f"(mẫu {start_sample}..{end_sample}), pre-roll {requested} frame "
        f"({requested * hop / 16:.0f} ms), năng lượng phủ {coverage * 100:.2f}%, "
        f"RTF={elapsed / audio_sec:.3f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("engine_name", ENGINES)
def test_khong_co_frame_nao_bi_gui_hai_lan(engine_name, stream):
    """Timestamp các frame đã chuyển tiếp phải tăng đều ĐÚNG 1 hop (không lặp, không nhảy)."""
    _proc, rec = _run(engine_name, stream["bytes"])
    assert rec.chunks, "phải có frame được chuyển tiếp"

    hop_sec = _HOP_SAMPLES[engine_name] / 16000.0
    ts_list = [ts for _, ts, _ in rec.chunks]
    assert ts_list[0] == pytest.approx(0.0, abs=1e-6) or ts_list[0] > 0.0
    for prev, cur in zip(ts_list, ts_list[1:]):
        step = (cur - prev) / hop_sec
        assert step == pytest.approx(1.0, abs=1e-6), (
            f"{engine_name}: bước timestamp giữa hai frame liên tiếp = {step:.3f} hop "
            f"({(cur - prev) * 1000:.1f} ms) ⇒ có frame bị lặp hoặc bị nhảy"
        )


# ─────────────────────────────────────────────── 4. chiếu config


@pytest.mark.slow
@pytest.mark.parametrize("engine_name", ENGINES)
def test_chieu_silence_va_threshold_xuong_field_native(engine_name):
    """`silence_duration_ms`/`threshold` phải tới đúng field native của engine."""
    engine = VADEngineFactory.get_engine(engine_name)
    state = engine.create_initial_state(threshold=0.77, silence_ms=700)

    if engine_name == "firered-vad":
        post = state.engine_state.vad.postprocessor
        assert post.speech_threshold == pytest.approx(0.77)
        assert post.min_silence_frame == 70, "700 ms / 10 ms mỗi frame = 70 frame"
    elif engine_name == "silero-vad":
        it = state.engine_state.iterator
        assert it.threshold == pytest.approx(0.77)
        assert it.min_silence_samples == pytest.approx(16000 * 0.7)
    else:
        stats = state.engine_state.cache["stats"]
        assert stats.speech_noise_thres == pytest.approx(0.77)
        assert stats.max_end_sil_frame_cnt_thresh == 700 - config.vad.fsmn.speech_to_sil_time_thres

    # `silence_ms=None` ⇒ quay về đúng mặc định trong config của engine (đúng docs).
    state_default = engine.create_initial_state(threshold=None, silence_ms=None)
    if engine_name == "firered-vad":
        post = state_default.engine_state.vad.postprocessor
        assert post.min_silence_frame == config.vad.firered.min_silence_frame
        assert post.speech_threshold == pytest.approx(config.vad.firered.speech_threshold)
    elif engine_name == "silero-vad":
        it = state_default.engine_state.iterator
        assert it.min_silence_samples == pytest.approx(
            config.vad.silero.min_silence_duration_ms * 16.0
        )
        assert it.threshold == pytest.approx(config.vad.silero.threshold)
    else:
        stats = state_default.engine_state.cache["stats"]
        assert stats.max_end_sil_frame_cnt_thresh == (
            config.vad.fsmn.max_end_silence_time - config.vad.fsmn.speech_to_sil_time_thres
        )
        assert stats.speech_noise_thres == pytest.approx(config.vad.fsmn.speech_noise_thres)


@pytest.mark.slow
@pytest.mark.parametrize("engine_name", ENGINES)
def test_doi_silence_giua_phien_co_hieu_luc_ngay(engine_name, stream):
    """`update_config(silence_duration_ms=…)` phải có tác dụng ngay ở frame kế tiếp."""
    head = stream["bytes"][: int(16000 * 1.0) * 2]  # 1 s đầu (lặng)
    _proc, _rec = _run(engine_name, head)

    proc = VADStreamProcessor(vad_engine=engine_name)
    proc.prewarm()
    proc.feed_chunk(head)
    proc.update_config(silence_duration_ms=1400, threshold=0.55)
    proc.feed_chunk(head[: 320 * 4])

    if engine_name == "firered-vad":
        post = proc._state.engine_state.vad.postprocessor
        assert post.min_silence_frame == 140
        assert post.speech_threshold == pytest.approx(0.55)
    elif engine_name == "silero-vad":
        it = proc._state.engine_state.iterator
        assert it.min_silence_samples == pytest.approx(16000 * 1.4)
        assert it.threshold == pytest.approx(0.55)
    else:
        stats = proc._state.engine_state.cache["stats"]
        assert stats.max_end_sil_frame_cnt_thresh == 1400 - config.vad.fsmn.speech_to_sil_time_thres
        assert stats.speech_noise_thres == pytest.approx(0.55)


@pytest.mark.slow
def test_silence_bang_0_nghia_la_dung_mac_dinh_engine():
    """Popup gửi `silenceDurationMs = 0` (off) ⇒ engine dùng mặc định docs của nó."""
    proc = VADStreamProcessor(vad_engine="silero-vad", silence_duration_ms=0)
    assert proc.silence_duration_ms is None
    proc.prewarm()
    it = proc._state.engine_state.iterator
    assert it.min_silence_samples == pytest.approx(config.vad.silero.min_silence_duration_ms * 16.0)


# ─────────────────────────────────────────────── 5. đổi engine giữa câu


@pytest.mark.slow
def test_doi_engine_giua_cau_phai_chot_cau_cu(stream):
    """Đổi engine trong lúc đang nói ⇒ `on_speech_end` được gọi TRƯỚC khi swap."""
    ends = []
    proc = VADStreamProcessor(
        vad_engine="firered-vad",
        on_speech_chunk=lambda *_: None,
        on_speech_end=lambda: ends.append(1),
    )
    proc.prewarm()

    # Nạp phần LÕI tiếng nói thật (bỏ lặng đệm) để chắc chắn đang trong câu.
    speech = stream["bytes"][int(stream["core_start"]) * 2: int(stream["core_end"]) * 2]
    for i in range(0, min(len(speech), int(16000 * 1.5) * 2), 320):
        proc.feed_chunk(speech[i:i + 320])
    assert proc._state.is_speech, "phải đang trong đoạn nói trước khi đổi engine"

    proc.apply_engine_change("silero-vad")
    assert ends, "phải chốt câu cũ trước khi đổi engine"
    assert proc.vad_engine == "silero-vad"
    assert not proc._state.is_speech, "state mới phải bắt đầu ở trạng thái im lặng"
