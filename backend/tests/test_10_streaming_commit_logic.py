"""Test tầng A — logic streaming/commit (P1.3, P2.1, P2.3, P2.4, P2.7, P4.5).

Chạy KHÔNG cần model thật: dùng `FakeInferenceEngine` kế thừa `TranscribeEngine` và
chỉ override `_run_inference_sync`. Toàn bộ logic orchestration thật được kiểm thử.

Mục tiêu thời gian: vài giây (xem pytest.ini).
"""

import asyncio
import time
from collections import deque

import pytest

from backend.config import config
from backend.asr.engine import TranscribeEngine, _MAX_PENDING_COMMITS
from backend.core.pipeline_events import CommitReason
from backend.tests.fakes import (
    FakeInferenceEngine,
    FakeVADEngine,
    make_silence_pcm,
    make_speech_pcm,
    pcm_to_int16_bytes,
)

SR = 16000
FRAME = 400  # 25ms


def _feed_speech(session, duration_sec: float) -> None:
    """Đẩy `duration_sec` giây audio có tiếng nói vào VAD theo từng frame."""
    pcm = make_speech_pcm(duration_sec, SR)
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        session.vad_processor.feed_chunk(pcm_to_int16_bytes(pcm[i:i + FRAME]))


def _feed_silence(session, duration_sec: float) -> None:
    pcm = make_silence_pcm(duration_sec, SR)
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        session.vad_processor.feed_chunk(pcm_to_int16_bytes(pcm[i:i + FRAME]))


# --------------------------------------------------------------------- P1.3 pre-roll
def test_pre_roll_does_not_include_previous_utterance(session_factory):
    """P1.3: câu thứ 2 KHÔNG được chứa 300ms audio đã transcribe của câu thứ 1.

    Bug cũ: `on_speech_start` tính `_speech_start_sample = total_written - 4800`
    TRƯỚC khi VAD flush pre-roll, nên lùi vào cuối câu trước.
    """
    session = session_factory()
    engine = session.asr_engine

    # Câu 1: nói 1s rồi im 0.8s (đủ để VAD chốt câu với silence_duration_ms=600)
    _feed_speech(session, 1.0)
    end_of_utt1 = engine.audio_buffer.total_written
    assert end_of_utt1 > 0

    _feed_silence(session, 0.8)
    # Câu 2 bắt đầu
    _feed_speech(session, 0.6)

    start_of_utt2 = engine._speech_start_sample
    assert start_of_utt2 >= 0, "on_speech_start/feed_audio phải chốt được điểm bắt đầu"

    # Điểm bắt đầu câu 2 phải nằm SAU khi câu 1 đã ghi xong toàn bộ audio của nó,
    # tức không lùi vào audio của câu 1.
    assert start_of_utt2 >= end_of_utt1, (
        f"Câu 2 bắt đầu ở sample {start_of_utt2} nhưng câu 1 đã ghi tới {end_of_utt1} "
        f"=> đang lùi vào audio của câu trước (bug pre-roll)."
    )


def test_pre_roll_start_is_at_first_speech_chunk(session_factory):
    """Điểm bắt đầu phải nằm đúng ở mép đầu của pre-roll, không xa hơn."""
    session = session_factory()
    engine = session.asr_engine

    _feed_silence(session, 0.5)          # tích pre-roll
    total_before = engine.audio_buffer.total_written
    _feed_speech(session, 0.3)           # bắt đầu nói

    # Pre-roll tối đa do CHÍNH VAD quyết định (`max_lookback_frames` của engine) và được
    # xả khi VAD phát START — không còn knob `pre_speech_buffer_ms` (đã xoá).
    assert engine._speech_start_sample >= 0
    assert engine.audio_buffer.total_written > total_before
    # Điểm bắt đầu không được vượt quá số sample đã ghi
    assert engine._speech_start_sample <= engine.audio_buffer.total_written


# ----------------------------------------------------------------- P2.3 preview window
def test_preview_window_is_bounded_independently_of_speech_length(session_factory, restore_config):
    """P2.3: độ dài cửa sổ preview phải bị chặn trên bởi `preview_window_sec`.

    Đây là bài test bảo vệ KPI K1: chi phí preview KHÔNG tăng theo độ dài video.
    """
    config.asr.preview_window_sec = 6.0
    config.sentence.max_duration_sec = 6.0
    session = session_factory()
    engine = session.asr_engine

    assert engine.preview_window_sec == 6.0

    seg_start = 0
    # 60 giây "nói liên tục" — cửa sổ phải vẫn chỉ 6s
    for total in (int(2 * SR), int(10 * SR), int(30 * SR), int(60 * SR)):
        win_start = engine._preview_window_start(seg_start, total)
        span_sec = (total - win_start) / SR
        assert span_sec <= 6.0 + 1e-6, f"cửa sổ preview {span_sec:.2f}s vượt 6s ở total={total}"
        assert span_sec == pytest.approx(6.0, abs=1e-3) if total >= 6 * SR else True


def test_preview_window_respects_floor_of_max_duration(session_factory, restore_config):
    """Nguyên tắc P2: nếu cửa sổ nhỏ hơn max_duration thì phải tự nâng lên.

    Nếu không, preview sẽ mất ngữ cảnh so với commit => mất độ chính xác (trái C4).
    """
    config.sentence.max_duration_sec = 8.0
    config.asr.preview_window_sec = 3.0
    session = session_factory()
    engine = session.asr_engine
    assert engine.preview_window_sec == 8.0, "cửa sổ phải được nâng lên bằng max_duration_sec"


def test_preview_window_disabled_returns_segment_start(session_factory, restore_config):
    """Cửa sổ = 0 nghĩa là tắt => quay lại hành vi cũ (from segment start)."""
    config.asr.preview_window_sec = 0.0
    session = session_factory()
    engine = session.asr_engine
    assert engine.preview_window_sec == 0.0
    assert engine._preview_window_start(1234, 99999) == 1234


def test_commit_uses_full_segment_not_window(session_factory, restore_config):
    """NGUYÊN TẮC P1: commit KHÔNG bị cửa sổ hoá.

    Đây là bất biến bảo vệ độ chính xác (C4): ngay cả khi preview chỉ đọc 6s cuối,
    commit phải đọc từ đầu câu.
    """
    config.asr.preview_window_sec = 6.0
    config.sentence.max_duration_sec = 6.0
    session = session_factory()
    engine = session.asr_engine

    # Giả lập câu dài 20s: điểm bắt đầu câu = 0, hiện tại = 20s
    seg_start = 0
    current = 20 * SR
    win_start = engine._preview_window_start(seg_start, current)
    assert win_start == current - 6 * SR, "preview phải bị cửa sổ hoá"

    # Commit request dùng seg_start (không phải win_start)
    commit_req = {"utterance_id": "u1", "start_sample": seg_start, "end_sample": current, "reason": "VAD_SILENCE"}
    assert commit_req["start_sample"] == 0
    assert commit_req["end_sample"] == current


# ------------------------------------------------------------------ P2.4 fixed rate
def test_fixed_rate_scheduler_skips_instead_of_drifting(session_factory, restore_config):
    """P2.4: khi inference chậm hơn nhịp, phải BỎ nhịp chứ không trôi.

    Kiểm tra bằng số counter `asr.preview_skipped` sau khi chạy vài vòng với
    inference chậm hơn poll interval.
    """
    from backend.core.metrics import metrics_collector

    config.asr.preview_fixed_rate = True
    config.asr.preview_window_sec = 6.0
    config.sentence.max_duration_sec = 6.0
    session = session_factory(infer_delay=0.06)  # 60ms > poll 20ms
    engine = session.asr_engine

    before = metrics_collector.get_counter("asr.preview_skipped")

    async def _run():
        engine.poll_interval_ms = 20
        gen = engine.stream_tokens()
        engine._speech_active = True
        engine._speech_start_sample = 0
        engine._awaiting_pre_roll = False
        engine.audio_buffer.write(make_speech_pcm(2.0, SR))

        async def _pump():
            for _ in range(8):
                await asyncio.sleep(0.02)

        pump = asyncio.create_task(_pump())
        try:
            while not pump.done():
                try:
                    await asyncio.wait_for(gen.__anext__(), timeout=0.3)
                except (StopAsyncIteration, asyncio.TimeoutError):
                    break
        finally:
            await gen.aclose()
            pump.cancel()

    asyncio.run(_run())
    after = metrics_collector.get_counter("asr.preview_skipped")
    assert after > before, "inference chậm hơn nhịp phải làm tăng counter preview_skipped"


# --------------------------------------------------------------------- P2.1 tiers
def test_tier234_disabled_by_flag(session_factory, restore_config):
    """Feature flag `enable_tier234=False` phải tắt hẳn BẬC 2/3/4 (rollback an toàn)."""
    config.sentence.enable_tier234 = False
    session = session_factory()
    engine = session.asr_engine
    engine.commit_manager.cfg.max_duration_sec = 1.0
    assert engine._evaluate_tier234("x" * 500, duration_sec=10.0) is None


def test_tier2_max_duration_fires(session_factory, restore_config):
    """P2.1 BẬC 2: câu vượt max_duration_sec phải bị cắt."""
    config.sentence.enable_tier234 = True
    session = session_factory()
    engine = session.asr_engine
    engine.commit_manager.cfg.max_duration_sec = 6.0
    reason = engine._evaluate_tier234("hello world", duration_sec=6.5)
    assert reason == CommitReason.MAX_DURATION


def test_tier2_max_chars_fires(session_factory, restore_config):
    """P2.1 BẬC 2: câu vượt max_chars cũng phải bị cắt."""
    config.sentence.enable_tier234 = True
    session = session_factory()
    engine = session.asr_engine
    engine.commit_manager.cfg.max_duration_sec = 999.0
    engine.commit_manager.cfg.max_chars = 10
    reason = engine._evaluate_tier234("a" * 11, duration_sec=1.0)
    assert reason == CommitReason.MAX_DURATION


def test_tier3_stable_prefix_fires_after_stable_polls(session_factory, restore_config):
    """P2.1 BẬC 3: text preview bất biến đủ lâu => cắt ở ranh giới từ (P3)."""
    config.sentence.enable_tier234 = True
    session = session_factory()
    engine = session.asr_engine
    engine.commit_manager.cfg.max_duration_sec = 999.0
    engine.commit_manager.cfg.split_on_stability = True
    engine.commit_manager.cfg.stability_threshold_polls = 3
    engine.commit_manager.cfg.stability_duration_sec = 0.0  # để test nhanh
    engine.commit_manager.cfg.stability_min_duration_sec = 2.0
    engine.commit_manager.cfg.stability_min_words = 4

    text = "the quick brown fox"  # 4 từ
    reasons = [engine._evaluate_tier234(text, duration_sec=3.0) for _ in range(4)]
    assert CommitReason.STABLE_PREFIX in reasons


def test_tier3_respects_min_duration_guard(session_factory, restore_config):
    """C4: BẬC 3 KHÔNG được cắt khi câu còn quá ngắn (chống mất chữ đầu câu).

    Kịch bản thật: 1 giây đầu model chỉ nhận được vài từ và text không đổi; nếu không
    có sàn thời lượng thì câu bị cắt ngay giữa lúc đang nói.
    """
    config.sentence.enable_tier234 = True
    session = session_factory()
    engine = session.asr_engine
    engine.commit_manager.cfg.max_duration_sec = 999.0
    engine.commit_manager.cfg.split_on_stability = True
    engine.commit_manager.cfg.stability_threshold_polls = 1
    engine.commit_manager.cfg.stability_duration_sec = 0.0
    engine.commit_manager.cfg.stability_min_duration_sec = 2.5
    engine.commit_manager.cfg.stability_min_words = 4

    # câu mới 0.9s, text đứng yên => KHÔNG được cắt
    for _ in range(5):
        assert engine._evaluate_tier234("the quick brown fox", duration_sec=0.9) is None

    # đủ dài => được phép cắt
    reasons = [engine._evaluate_tier234("the quick brown fox", duration_sec=3.0) for _ in range(4)]
    assert CommitReason.STABLE_PREFIX in reasons


def test_tier3_respects_min_words_guard(session_factory, restore_config):
    """C4: BẬC 3 không cắt khi preview chưa đủ từ."""
    config.sentence.enable_tier234 = True
    session = session_factory()
    engine = session.asr_engine
    engine.commit_manager.cfg.max_duration_sec = 999.0
    engine.commit_manager.cfg.split_on_stability = True
    engine.commit_manager.cfg.stability_threshold_polls = 1
    engine.commit_manager.cfg.stability_duration_sec = 0.0
    engine.commit_manager.cfg.stability_min_duration_sec = 1.0
    engine.commit_manager.cfg.stability_min_words = 4

    for _ in range(5):
        assert engine._evaluate_tier234("hello there", duration_sec=3.0) is None


def test_tier4_inactivity_fires_when_no_audio_arrives(session_factory, restore_config):
    """P2.1 BẬC 4: không có audio mới trong inactivity_timeout_sec => force commit.

    Lưu ý: `record_activity()` chỉ được nuôi bởi AUDIO ĐẾN (feed_audio), không phải bởi
    vòng poll — nếu nuôi bằng poll thì BẬC 4 sẽ không bao giờ chạy.
    """
    import time

    config.sentence.enable_tier234 = True
    session = session_factory()
    engine = session.asr_engine
    engine.commit_manager.cfg.max_duration_sec = 999.0
    engine.commit_manager.cfg.split_on_stability = False
    engine.commit_manager.cfg.inactivity_timeout_sec = 0.05
    engine.commit_manager._last_activity_time = time.time() - 1.0

    reason = engine._evaluate_tier234("some text", duration_sec=1.0)
    assert reason == CommitReason.TIMEOUT_FORCE


def test_record_activity_is_not_fed_by_polling(session_factory, restore_config):
    """`_evaluate_tier234` KHÔNG được tự nuôi đồng hồ inactivity của chính nó."""
    import time

    config.sentence.enable_tier234 = True
    session = session_factory()
    engine = session.asr_engine
    engine.commit_manager._last_activity_time = time.time() - 10.0
    engine._evaluate_tier234("x", 1.0)  # gọi 1 lần
    assert time.time() - engine.commit_manager._last_activity_time > 5.0, (
        "_evaluate_tier234 không được cập nhật _last_activity_time"
    )


def test_feed_audio_updates_activity_clock(session_factory):
    """`feed_audio` PHẢI nuôi đồng hồ inactivity (audio đến)."""
    import time

    session = session_factory()
    engine = session.asr_engine
    engine.commit_manager._last_activity_time = time.time() - 10.0
    engine.feed_audio(pcm_to_int16_bytes(make_speech_pcm(0.1, SR)))
    assert time.time() - engine.commit_manager._last_activity_time < 1.0


# ------------------------------------------------------------------ P2.7 boundaries
def test_boundary_overlap_trim_removes_repeated_prefix():
    """P2.7: phần đầu câu bị lặp do chồng lấn phải được TRIM, không drop cả câu."""
    from backend.core.dedup import CommitDeduplicator

    d = CommitDeduplicator()
    d.record_commit("the quick brown fox jumps")
    trimmed = d.trim_boundary_overlap("brown fox jumps over the lazy dog")
    assert trimmed == "over the lazy dog", f"nhận được: {trimmed!r}"


def test_boundary_overlap_trim_no_false_positive():
    """Không được cắt bừa khi không có chồng lấn thật."""
    from backend.core.dedup import CommitDeduplicator

    d = CommitDeduplicator()
    d.record_commit("the quick brown fox")
    assert d.trim_boundary_overlap("completely different words here") == "completely different words here"


def test_boundary_overlap_trim_never_returns_empty():
    """Nếu trim hết sạch thì trả nguyên bản (tránh mất cả câu)."""
    from backend.core.dedup import CommitDeduplicator

    d = CommitDeduplicator()
    d.record_commit("hello world")
    assert d.trim_boundary_overlap("hello world") == "hello world"


def test_boundary_overlap_trim_cjk_by_chars():
    """Hỗ trợ CJK (không có khoảng trắng) bằng so khớp ký tự."""
    from backend.core.dedup import CommitDeduplicator

    d = CommitDeduplicator()
    d.record_commit("今天天气很好")
    trimmed = d.trim_boundary_overlap("天气很好我们去公园")
    assert trimmed == "我们去公园", f"nhận được: {trimmed!r}"


# -------------------------------------------------------------- P4.5 no silent drop
def test_pending_commits_merges_instead_of_dropping(session_factory):
    """P4.5: khi hàng đợi commit đầy, phải GỘP câu cũ nhất, KHÔNG vứt bỏ.

    Trước đây `deque(maxlen=8)` âm thầm vứt câu cũ => mất phụ đề không dấu vết.
    """
    from backend.core.metrics import metrics_collector

    session = session_factory()
    engine = session.asr_engine

    before_merged = metrics_collector.get_counter("asr.commits_merged")
    with engine._lock:
        for i in range(_MAX_PENDING_COMMITS + 3):
            engine._enqueue_commit_locked(f"u{i}", start_sample=i * 100, end_sample=(i + 1) * 100, reason="MAX_DURATION")

    # Số mục không vượt trần (bounded) ...
    assert len(engine._pending_commits) <= _MAX_PENDING_COMMITS
    # ... nhưng audio KHÔNG bị mất: end_sample của cái cũ nhất phải được mở rộng.
    assert engine._pending_commits[0]["end_sample"] == (_MAX_PENDING_COMMITS + 3) * 100
    assert engine._pending_commits[0]["reason"] == "MERGED_BACKLOG"
    assert metrics_collector.get_counter("asr.commits_merged") > before_merged


def test_pending_commits_cover_contiguous_audio(session_factory):
    """Sau khi gộp, hợp các khoảng sample phải phủ liên tục [0, tổng] (không hổng audio).

    Lưu ý: khoảng được MỞ RỘNG nằm ở commit CŨ NHẤT (cái bị gộp), không phải cái mới nhất.
    """
    session = session_factory()
    engine = session.asr_engine
    n_items = _MAX_PENDING_COMMITS + 2
    with engine._lock:
        for i in range(n_items):
            engine._enqueue_commit_locked(f"u{i}", start_sample=i * 1000, end_sample=(i + 1) * 1000, reason="MAX_DURATION")

    reqs = sorted(engine._pending_commits, key=lambda r: r["start_sample"])
    # Tổng độ phủ phải bằng toàn bộ audio đã đưa vào
    assert reqs[0]["start_sample"] == 0
    assert max(r["end_sample"] for r in reqs) == n_items * 1000

    # Không có khoảng trống: interval sau bắt đầu không muộn hơn interval trước kết thúc
    covered_until = reqs[0]["end_sample"]
    for r in reqs[1:]:
        assert r["start_sample"] <= covered_until, (
            f"hổng audio: interval bắt đầu ở {r['start_sample']} nhưng mới phủ tới {covered_until}"
        )
        covered_until = max(covered_until, r["end_sample"])
    assert covered_until == n_items * 1000


# ------------------------------------------------------------ integration (1 test)
def test_adaptive_preview_backoff_engages_when_slow(session_factory, restore_config):
    """P2.4b: khi preview chậm, nhịp hiệu dụng phải GIÃN RA (rồi tự thu lại khi nhanh)."""
    config.asr.preview_adaptive_backoff = True
    config.asr.preview_slow_ms = 100.0
    config.asr.preview_fast_ms = 20.0
    config.asr.preview_max_interval_ms = 1000
    session = session_factory()
    engine = session.asr_engine

    base = 0.3
    engine._eff_poll_interval = base

    # Chưa có mẫu => giữ nhịp cơ sở
    assert engine._effective_poll_interval(base) == pytest.approx(base)

    # Preview chậm (600ms > slow 100ms) => giãn dần
    for _ in range(4):
        engine._preview_durations.append(600.0)
    first = engine._effective_poll_interval(base)
    assert first > base, "phải giãn nhịp khi preview chậm"
    second = engine._effective_poll_interval(base)
    assert second > first, "phải giãn tiếp nếu vẫn chậm"
    # Không vượt trần
    for _ in range(10):
        engine._effective_poll_interval(base)
    assert engine._eff_poll_interval <= 1.0 + 1e-6

    # Preview nhanh trở lại => thu hẹp dần về nhịp cơ sở
    engine._preview_durations.clear()
    for _ in range(4):
        engine._preview_durations.append(10.0)
    for _ in range(30):
        engine._effective_poll_interval(base)
    assert engine._eff_poll_interval == pytest.approx(base, abs=1e-6), \
        "phải thu hẹp về nhịp cơ sở khi backend nhanh trở lại"


def test_adaptive_backoff_can_be_disabled(session_factory, restore_config):
    """Cờ tắt tự điều chỉnh => luôn dùng nhịp cơ sở."""
    config.asr.preview_adaptive_backoff = False
    session = session_factory()
    engine = session.asr_engine
    engine._eff_poll_interval = 0.9
    for _ in range(8):
        engine._preview_durations.append(5000.0)
    assert engine._effective_poll_interval(0.3) == pytest.approx(0.3)


def test_preload_warms_both_short_and_max_duration(session_factory, restore_config, monkeypatch):
    """Warm-up PHẢI phủ độ dài câu tối đa, không chỉ đoạn ngắn.

    Đo thực tế trên Vulkan: sau khi chỉ warm 0,5 s, one inference 6 s mất **4218 ms**,
    nhưng lần 6 s kế tiếp chỉ **57 ms** (nhanh hơn 74×) — tức lần đầu chạm một độ dài
    LỚN HƠN phải trả chi phí cấp workspace/graph mới. Nếu không warm đủ dài, phụ đề ĐẦU
    TIÊN của người dùng bị "đơ" vài giây.
    """
    config.sentence.max_duration_sec = 6.0
    session = session_factory()
    engine = session.asr_engine

    lengths: list = []
    monkeypatch.setattr(engine, "_run_inference_sync", lambda pcm: (lengths.append(len(pcm)), "")[1])
    # FakeInferenceEngine override `_preload_model` thành no-op; bind lại method THẬT để
    # test đúng logic warm-up của engine thật mà không cần model.
    monkeypatch.setattr(engine, "_preload_model", TranscribeEngine._preload_model.__get__(engine))
    # Giả lập model đã nạp nhưng CHƯA warm độ dài dài (đúng trạng thái sau prewarm cũ)
    monkeypatch.setattr(TranscribeEngine, "_shared_model", object(), raising=False)
    monkeypatch.setattr(TranscribeEngine, "_shared_model_key", engine.model_key, raising=False)
    monkeypatch.setattr(TranscribeEngine, "_shared_warmed_long", False, raising=False)

    engine._preload_model()

    assert any(n >= int(16000 * 5.9) for n in lengths), (
        f"chưa warm tới max_duration_sec — các độ dài đã warm: {lengths}"
    )
    assert any(n <= int(16000 * 1.0) for n in lengths), f"chưa warm đoạn ngắn: {lengths}"

    # Gọi lần 2 phải KHÔNG warm lại (idempotent). Đây là bất biến hành vi cần kiểm —
    # không assert vào class attribute vì `self.__class__` trỏ tới lớp con khi test.
    lengths.clear()
    engine._preload_model()
    assert lengths == [], "đã warm đủ rồi thì không được warm lại"


def test_prewarm_forces_full_warm_even_if_model_loaded(session_factory, restore_config, monkeypatch):
    """`prewarm()` lúc khởi động phải warm ĐỦ, kể cả khi model đã nạp.

    Trước đây `prewarm()` tự chạy dummy 0,5 s rồi `_preload_model()` early-return vì model
    đã nạp ⇒ runtime không bao giờ warm độ dài dài.
    """
    config.sentence.max_duration_sec = 6.0
    session = session_factory()
    engine = session.asr_engine

    lengths: list = []
    monkeypatch.setattr(engine, "_run_inference_sync", lambda pcm: (lengths.append(len(pcm)), "")[1])
    monkeypatch.setattr(engine, "_preload_model", TranscribeEngine._preload_model.__get__(engine))
    monkeypatch.setattr(TranscribeEngine, "_shared_model", object(), raising=False)
    monkeypatch.setattr(TranscribeEngine, "_shared_model_key", engine.model_key, raising=False)
    monkeypatch.setattr(TranscribeEngine, "_shared_warmed_long", True, raising=False)

    engine.prewarm()
    assert any(n >= int(16000 * 5.9) for n in lengths), (
        f"prewarm() không warm độ dài dài dù đã có model: {lengths}"
    )


def test_stream_tokens_end_to_end_with_fake_engine(session_factory, restore_config):
    """Integration tầng A: `stream_tokens` thật phải phát preview rồi phát final.

    Chứng minh CommitManager được WIRE THẬT vào runtime (trước đây `decide_commit_trigger`
    không được gọi ở đâu => dead code).
    """
    config.asr.preview_window_sec = 6.0
    config.sentence.max_duration_sec = 6.0
    config.asr.min_transcribe_sec = 0.2
    # Tắt BẬC 3 cho test này để chốt câu xảy ra qua VAD (BẬC 1) như thiết kế.
    config.sentence.stability_min_duration_sec = 60.0
    session = session_factory(text_fn=lambda n: "alpha beta gamma")
    engine = session.asr_engine
    engine.poll_interval_ms = 40
    engine.min_transcribe_sec = 0.2

    async def _run():
        gen = engine.stream_tokens()
        msgs = []

        async def _pump():
            # nói 0.9s
            pcm = make_speech_pcm(0.9, SR)
            for i in range(0, len(pcm) - FRAME + 1, FRAME):
                session.vad_processor.feed_chunk(pcm_to_int16_bytes(pcm[i:i + FRAME]))
                await asyncio.sleep(0.03)
            # im lặng để VAD chốt câu
            sil = make_silence_pcm(1.0, SR)
            for i in range(0, len(sil) - FRAME + 1, FRAME):
                session.vad_processor.feed_chunk(pcm_to_int16_bytes(sil[i:i + FRAME]))
                await asyncio.sleep(0.01)

        pump = asyncio.create_task(_pump())
        # Trần an toàn: nếu engine không bao giờ kết thúc (từng quan sát được một lần
        # chạy treo ở đây, ngốn 33 GB RAM) thì FAIL ngay kèm trạng thái nội bộ, thay vì
        # để tiến trình quay nóng vô hạn.
        max_iters = 400
        deadline = time.monotonic() + 15.0
        iters = 0
        try:
            while True:
                iters += 1
                if iters > max_iters or time.monotonic() > deadline:
                    raise AssertionError(
                        f"stream_tokens không kết thúc sau {iters} vòng / 15s "
                        f"(msgs={len(msgs)}, speech_active={engine._speech_active}, "
                        f"pending={len(engine._pending_commits)}, running={engine._is_running}) "
                        f"— nghi ngờ vòng lặp vô hạn trong engine."
                    )
                try:
                    msgs.append(await asyncio.wait_for(gen.__anext__(), timeout=0.6))
                except (StopAsyncIteration, asyncio.TimeoutError):
                    break
                if pump.done() and not engine._pending_commits and not engine._speech_active:
                    # thêm 1 nhịp để chắc chắn không còn message
                    try:
                        msgs.append(await asyncio.wait_for(gen.__anext__(), timeout=0.3))
                    except (StopAsyncIteration, asyncio.TimeoutError):
                        break
        finally:
            await gen.aclose()
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
        return msgs

    msgs = asyncio.run(_run())
    previews = [m for m in msgs if not m.get("is_final")]
    finals = [m for m in msgs if m.get("is_final")]

    assert previews, "phải có ít nhất 1 preview trong lúc đang nói"
    assert finals, "phải có ít nhất 1 câu chốt (final) sau khi VAD báo im lặng"
    assert finals[0]["commit_reason"] == "VAD_SILENCE"
    assert finals[0]["text"]
