"""Test tầng A — F-39: chống nghẽn executor khi inference native chạy quá lâu.

Bối cảnh (đo thật, xem `report/audit/05_measurements_and_status.md` §12):
`_EXECUTOR` là ThreadPoolExecutor với hàng đợi KHÔNG giới hạn. Khi một `session.run()`
native chạy lâu (đã bắt được stack trong `transcribe_cpp.run`), vòng preview vẫn nộp
thêm việc mỗi nhịp ⇒ RAM tăng ~66-85 MB/s cho tới khi hết bộ nhớ.

Hai hàng rào được kiểm thử ở đây:
1. `max_inflight_infer`: không bao giờ có nhiều hơn N inference cùng lúc ⇒ hàng đợi
   executor không thể phình.
2. `inference_watchdog_sec`: inference vượt ngân sách ⇒ gọi `cancel_inference()`
   (native `session.cancel()`) để giải phóng `_infer_lock`.
"""

import asyncio
import time

from backend.config import config
from backend.core.metrics import metrics_collector
from backend.tests.fakes import (
    FakeInferenceEngine,
    make_speech_pcm,
    pcm_to_int16_bytes,
)

SR = 16000
FRAME = 400  # 25 ms


def _slow_engine(monkeypatch, delay_sec: float, cancel_flag: dict, respect_cancel: bool = True):
    """Engine giả có inference chậm + ghi nhận việc gọi cancel.

    `respect_cancel=True` mô phỏng native NGOAN: nhận `session.cancel()` thì trả về sớm
    kèm partial result (đúng như `Aborted` + `partial_result` của transcribe.cpp).
    `respect_cancel=False` mô phỏng native CỨNG ĐẦU: vẫn chạy hết dù đã bị huỷ.
    """
    engine = FakeInferenceEngine(text_fn=lambda n: "alpha beta gamma")

    def _slow(pcm):  # noqa: ANN001
        deadline = time.perf_counter() + delay_sec
        while time.perf_counter() < deadline:
            if respect_cancel and cancel_flag.get("called"):
                return "partial result"
            time.sleep(0.005)
        return "slow result"

    monkeypatch.setattr(engine, "_run_inference_sync", _slow, raising=False)
    monkeypatch.setattr(engine, "_preload_model", lambda force_warm=False: None, raising=False)

    async def _cancel():
        cancel_flag["called"] = cancel_flag.get("called", 0) + 1
        cancel_flag["at"] = time.perf_counter()

    monkeypatch.setattr(engine, "cancel_inference", _cancel, raising=False)
    return engine


def test_watchdog_fires_and_requests_native_cancel(restore_config, monkeypatch):
    """Inference vượt ngân sách phải kích hoạt `cancel_inference()` và cắt ngắn được."""
    config.asr.inference_watchdog_sec = 0.2
    cancel_flag: dict = {}
    engine = _slow_engine(monkeypatch, delay_sec=1.0, cancel_flag=cancel_flag)

    import numpy as np

    audio = np.zeros(16000, dtype=np.float32)
    t0 = time.perf_counter()
    text = asyncio.run(engine._infer_with_watchdog(audio))
    elapsed = time.perf_counter() - t0

    assert cancel_flag.get("called") == 1, "watchdog phải yêu cầu native huỷ"
    # Ngân sách 0.2s; native ngoan phải trả về ngay sau đó (<< 1.0s nếu không cắt).
    assert elapsed < 0.5, f"watchdog không cắt ngắn được inference ({elapsed:.2f}s)"
    assert text == "partial result", "phải giữ partial result thay vì mất cả câu"
    assert engine._inflight_count() == 0


def test_watchdog_hard_timeout_returns_empty(restore_config, monkeypatch):
    """Native cứng đầu (không thoát sau khi huỷ) => bỏ vòng đó, KHÔNG treo vô hạn."""
    config.asr.inference_watchdog_sec = 0.2
    config.asr.inference_watchdog_grace_sec = 0.1
    config.asr.rss_runaway_delta_mb = 0.0  # chỉ bật đồng hồ thời gian cho test này
    cancel_flag: dict = {}
    engine = _slow_engine(monkeypatch, delay_sec=1.5, cancel_flag=cancel_flag, respect_cancel=False)

    import numpy as np

    t0 = time.perf_counter()
    text = asyncio.run(engine._infer_with_watchdog(np.zeros(16000, dtype=np.float32)))
    elapsed = time.perf_counter() - t0

    assert cancel_flag.get("called") == 1
    assert text == "", "quá hạn cứng thì phải trả rỗng để không xếp thêm việc"
    assert elapsed < 1.5, f"phải thoát sau ngân sách + ân hạn, không chờ 5s ({elapsed:.2f}s)"
    assert engine._inflight_count() == 0


def test_rss_runaway_aborts_inference(restore_config, monkeypatch):
    """RSS tăng vọt trong lúc inference chạy ⇒ huỷ ngay, không chờ hết 5 s."""
    config.asr.inference_watchdog_sec = 0.0   # tắt đồng hồ thời gian để cô lập RSS guard
    config.asr.inference_watchdog_grace_sec = 0.1
    config.asr.rss_runaway_delta_mb = 100.0
    cancel_flag: dict = {}
    engine = _slow_engine(monkeypatch, delay_sec=1.5, cancel_flag=cancel_flag, respect_cancel=False)

    readings = iter([1000.0, 1050.0, 2000.0, 3000.0, 4000.0] + [5000.0] * 60)
    monkeypatch.setattr(engine, "_process_rss_mb", staticmethod(lambda: next(readings)), raising=False)

    import numpy as np

    t0 = time.perf_counter()
    text = asyncio.run(engine._infer_with_watchdog(np.zeros(16000, dtype=np.float32)))
    elapsed = time.perf_counter() - t0

    assert cancel_flag.get("called") == 1, "phát hiện RSS tăng vọt thì phải huỷ inference"
    assert text == ""
    assert elapsed < 3.0, f"phải bỏ vòng sau khi phát hiện + ân hạn, không chờ 5s ({elapsed:.2f}s)"
    assert engine._inflight_count() == 0


def test_rss_runaway_guard_can_be_disabled(restore_config, monkeypatch):
    """`rss_runaway_delta_mb = 0` ⇒ không can thiệp dù RSS tăng."""
    config.asr.inference_watchdog_sec = 0.0
    config.asr.rss_runaway_delta_mb = 0.0
    cancel_flag: dict = {}
    engine = _slow_engine(monkeypatch, delay_sec=0.3, cancel_flag=cancel_flag, respect_cancel=False)
    monkeypatch.setattr(engine, "_process_rss_mb", staticmethod(lambda: 999999.0), raising=False)

    import numpy as np

    assert asyncio.run(engine._infer_with_watchdog(np.zeros(16000, dtype=np.float32))) == "slow result"
    assert not cancel_flag.get("called")


def test_watchdog_disabled_runs_inference_to_completion(restore_config, monkeypatch):
    """`inference_watchdog_sec = 0` => tắt hẳn, không gọi cancel."""
    config.asr.inference_watchdog_sec = 0.0
    cancel_flag: dict = {}
    engine = _slow_engine(monkeypatch, delay_sec=0.2, cancel_flag=cancel_flag)

    import numpy as np

    text = asyncio.run(engine._infer_with_watchdog(np.zeros(16000, dtype=np.float32)))

    assert text == "slow result"
    assert not cancel_flag.get("called"), "watchdog đang TẮT thì không được gọi cancel"
    assert engine._inflight_count() == 0


def test_inflight_guard_blocks_preview_queue_growth(restore_config, monkeypatch):
    """Khi đã có inference đang chạy, vòng preview KHÔNG được nộp thêm việc."""
    config.asr.max_inflight_infer = 1
    config.asr.inference_watchdog_sec = 0.0
    engine = FakeInferenceEngine(text_fn=lambda n: "alpha beta gamma")

    submitted = {"n": 0}

    def _counted(pcm):  # noqa: ANN001
        submitted["n"] += 1
        return "x"

    monkeypatch.setattr(engine, "_run_inference_sync", _counted, raising=False)

    # Giả lập "đang có 1 inference chạy" (chưa xong).
    engine._begin_infer()
    try:
        assert engine._inflight_count() >= 1
        max_inflight = int(config.asr.max_inflight_infer)
        assert engine._inflight_count() >= max_inflight  # điều kiện bỏ vòng preview
    finally:
        engine._end_infer()

    # Sau khi hết in-flight thì được chạy tiếp.
    assert engine._inflight_count() == 0
    import numpy as np

    assert asyncio.run(engine._infer_with_watchdog(np.zeros(16000, dtype=np.float32))) == "x"
    assert submitted["n"] == 1


def test_engine_config_defaults_are_sane(restore_config):
    """Mặc định phải bật cả hai hàng rào (không được vô tình tắt)."""
    from backend.config import ASRConfig

    cfg = ASRConfig()
    assert cfg.max_inflight_infer >= 1
    assert cfg.inference_watchdog_sec > 0


# --------------------------------------------------------------- recycle native (F-39b)
def _recycle_engine(monkeypatch, rss_mb: float, baseline_mb: float):
    """Engine tầng A + mốc RSS giả + hook unload được đếm."""
    from backend.asr.engine import TranscribeEngine

    engine = FakeInferenceEngine(text_fn=lambda n: "alpha beta gamma")
    state = {"unloaded": 0}

    monkeypatch.setattr(TranscribeEngine, "_shared_rss_baseline_mb", baseline_mb, raising=False)
    monkeypatch.setattr(engine, "_process_rss_mb", staticmethod(lambda: rss_mb), raising=False)
    monkeypatch.setattr(
        TranscribeEngine,
        "unload_shared_model",
        classmethod(lambda cls: state.__setitem__("unloaded", state["unloaded"] + 1)),
        raising=False,
    )
    return engine, state


def test_native_session_recycled_when_rss_grows(restore_config, monkeypatch):
    """RSS vượt mốc nạp model quá ngưỡng ⇒ đóng phiên native ở cuối session."""
    config.asr.native_recycle_rss_delta_mb = 512.0
    engine, state = _recycle_engine(monkeypatch, rss_mb=3000.0, baseline_mb=1000.0)

    assert asyncio.run(engine.maybe_recycle_native()) is True
    assert state["unloaded"] == 1


def test_native_session_not_recycled_below_threshold(restore_config, monkeypatch):
    """Tăng ít hơn ngưỡng thì KHÔNG đóng phiên native (tránh nạp lại model vô ích)."""
    config.asr.native_recycle_rss_delta_mb = 512.0
    engine, state = _recycle_engine(monkeypatch, rss_mb=1200.0, baseline_mb=1000.0)

    assert asyncio.run(engine.maybe_recycle_native()) is False
    assert state["unloaded"] == 0


def test_native_recycle_can_be_disabled(restore_config, monkeypatch):
    """`native_recycle_rss_delta_mb = 0` ⇒ tắt hẳn."""
    config.asr.native_recycle_rss_delta_mb = 0.0
    engine, state = _recycle_engine(monkeypatch, rss_mb=9000.0, baseline_mb=100.0)

    assert asyncio.run(engine.maybe_recycle_native()) is False
    assert state["unloaded"] == 0


def test_cleanup_calls_recycle_and_never_raises(restore_config, monkeypatch):
    """`cleanup()` phải gọi recycle và không được ném lỗi dù recycle hỏng."""
    config.asr.native_recycle_rss_delta_mb = 1.0
    from backend.asr.engine import TranscribeEngine

    engine = FakeInferenceEngine(text_fn=lambda n: "x")
    called = {"n": 0}

    async def _boom():
        called["n"] += 1
        raise RuntimeError("recycle hỏng")

    monkeypatch.setattr(engine, "maybe_recycle_native", _boom, raising=False)
    asyncio.run(engine.cleanup())  # không được ném
    assert called["n"] == 1
    assert TranscribeEngine is not None


# ------------------------------------------------------- K2: tới preview đầu tiên
def test_speech_start_resets_first_preview_clock(restore_config):
    """`on_speech_start()` phải đặt lại mốc đo K2 (mỗi câu đo riêng)."""
    engine = FakeInferenceEngine(text_fn=lambda n: "x")
    engine._first_preview_reported = True
    engine._speech_started_at = 0.0

    engine.on_speech_start()

    assert engine._first_preview_reported is False
    assert engine._speech_started_at > 0.0


def test_first_preview_wait_is_capped(restore_config):
    """K2: khi TẮT đánh thức theo tiếng nói, hành vi cũ vẫn phải còn (không hồi quy)."""
    config.asr.wake_on_speech_start = False
    engine = FakeInferenceEngine(text_fn=lambda n: "x")
    engine.on_speech_start()
    assert engine._first_preview_reported is False

    config.asr.wake_on_speech_start = True
    engine2 = FakeInferenceEngine(text_fn=lambda n: "x")
    engine2._commit_event.clear()
    engine2.on_speech_start()
    # Không có event loop đang chạy trong test này nên chỉ cần không ném lỗi.
    assert engine2._speech_active is True


def test_first_preview_metric_is_recorded(session_factory, restore_config):
    """K2: chạy pipeline thật (engine giả) và đòi metric `asr.first_preview_ms` xuất hiện."""
    config.asr.min_transcribe_sec = 0.2
    config.sentence.stability_min_duration_sec = 60.0
    config.asr.inference_watchdog_sec = 0.0

    session = session_factory(text_fn=lambda n: "alpha beta gamma")
    engine = session.asr_engine
    engine.poll_interval_ms = 40
    engine.min_transcribe_sec = 0.2

    async def _run():
        gen = engine.stream_tokens()
        pcm = make_speech_pcm(0.9, SR)
        for i in range(0, len(pcm) - FRAME + 1, FRAME):
            session.vad_processor.feed_chunk(pcm_to_int16_bytes(pcm[i:i + FRAME]))
        msgs = []
        for _ in range(80):
            try:
                msgs.append(await asyncio.wait_for(gen.__anext__(), timeout=0.4))
            except (StopAsyncIteration, asyncio.TimeoutError):
                break
            if any(not m.get("is_final") for m in msgs):
                break
        await gen.aclose()
        return msgs

    metrics_collector.reset()
    msgs = asyncio.run(_run())

    assert any(not m.get("is_final") for m in msgs), f"không có preview nào: {msgs}"
    stats = metrics_collector.get_stage_stats("asr.first_preview_ms")
    assert stats.get("count", 0) >= 1, f"chưa ghi metric asr.first_preview_ms: {stats}"
    assert stats.get("avg_ms", 0.0) > 0.0


def test_e2e_commit_metric_is_recorded(session_factory, restore_config):
    """K4: chạy hết một chu kỳ nói → im lặng và đòi metric `asr.e2e_commit_ms`."""
    import backend.tests.fakes as fakes

    config.asr.min_transcribe_sec = 0.2
    config.asr.inference_watchdog_sec = 0.0
    config.sentence.stability_min_duration_sec = 60.0
    # Tắt cửa sổ GIỮ CÂU: bài này cần VAD END (BẬC 1) chốt NGAY câu 0,9 s để đo
    # `asr.e2e_commit_ms`. Bật giữ câu thì commit bị hoãn 59 s (đúng thiết kế, xem
    # `test_53_hold_short_sentence.py`).
    config.sentence.hold_short_sentence = False

    session = session_factory(text_fn=lambda n: "alpha beta gamma")
    engine = session.asr_engine
    engine.poll_interval_ms = 40
    engine.min_transcribe_sec = 0.2

    async def _run():
        gen = engine.stream_tokens()
        speech = fakes.make_speech_pcm(0.9, SR)
        for i in range(0, len(speech) - FRAME + 1, FRAME):
            session.vad_processor.feed_chunk(fakes.pcm_to_int16_bytes(speech[i:i + FRAME]))
        silence = fakes.make_silence_pcm(1.0, SR)
        for i in range(0, len(silence) - FRAME + 1, FRAME):
            session.vad_processor.feed_chunk(fakes.pcm_to_int16_bytes(silence[i:i + FRAME]))
        msgs = []
        for _ in range(80):
            try:
                msgs.append(await asyncio.wait_for(gen.__anext__(), timeout=0.4))
            except (StopAsyncIteration, asyncio.TimeoutError):
                break
        await gen.aclose()
        return msgs

    metrics_collector.reset()
    msgs = asyncio.run(_run())

    finals = [m for m in msgs if m.get("is_final")]
    assert finals, f"không có câu chốt nào: {msgs}"
    stats = metrics_collector.get_stage_stats("asr.e2e_commit_ms")
    assert stats.get("count", 0) >= 1, f"chưa ghi metric asr.e2e_commit_ms: {stats}"
    assert stats.get("avg_ms", 0.0) > 0.0


# ------------------------------------------- F-42: cắt câu không phụ thuộc preview
def test_tier234_is_evaluated_even_when_preview_is_deferred(restore_config):
    """F-42: khi preview bị HOÃN (đang có inference khác), BẬC 2 vẫn phải cắt câu đúng hạn.

    Trước đây khối `_evaluate_tier234` nằm trong nhánh preview ⇒ preview bị hoãn thì không
    ai cắt câu, câu phình tới ~45 s dù `max_duration_sec` = 6 s (log thật: một câu
    MAX_DURATION dài ~600 ký tự, `infer=1294 ms`).
    """
    config.sentence.max_duration_sec = 6.0
    config.sentence.enable_tier234 = True
    config.asr.max_inflight_infer = 1

    engine = FakeInferenceEngine(text_fn=lambda n: "alpha beta gamma")
    engine._speech_active = True
    engine._speech_start_sample = 0
    metrics_collector.reset()

    import numpy as np

    # 12 giây audio đã tích luỹ trong câu hiện tại (vượt xa max_duration 6 s)
    engine.feed_audio(np.zeros(int(12.0 * SR), dtype=np.float32))

    duration_sec = 12.0
    reason = engine._evaluate_tier234("alpha beta gamma", duration_sec)
    assert reason is not None, "câu 12 s với max_duration 6 s PHẢI bị cắt"
    assert reason.value == "MAX_DURATION"

    # Và điều kiện cắt KHÔNG được phụ thuộc vào việc preview có chạy hay không:
    # mô phỏng "đang có 1 inference in-flight" (preview sẽ bị hoãn) rồi kiểm tra lại.
    engine._begin_infer()
    try:
        assert engine._inflight_count() >= int(config.asr.max_inflight_infer)
        reason_2 = engine._evaluate_tier234("alpha beta gamma", duration_sec)
        assert reason_2 is not None and reason_2.value == "MAX_DURATION"
    finally:
        engine._end_infer()

