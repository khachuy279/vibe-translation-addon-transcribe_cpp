"""Mô phỏng ĐẦU-CUỐI Pipeline B trên audio thật, có mô phỏng bộ đệm lookahead như trình duyệt.

VÌ SAO CẦN (thay vì chỉ đọc log):
    Hành vi của `LookaheadChunker` phụ thuộc `buffered_end` (bộ đệm tiến của trình duyệt) và
    `LookaheadSessionState` phụ thuộc `current_time` (vị trí phát). Chỉ mô phỏng đúng hai đại
    lượng đó mới tái hiện được các sự cố THẬT:
      • `Trừ chồng lấn ranh giới` cắt mất từ đầu câu,
      • câu "bắt đầu khối" xuất hiện quá sớm,
      • câu backend gửi đi nhưng không hiện trên Extension.

Script chạy CHÍNH vòng lặp `_offline_batch_loop` (không viết lại logic), nạp đệm dần theo
thời gian phát, rồi kiểm tra từng phụ đề gửi đi bằng cách mô phỏng `SubtitleTimelineQueue`
của Extension (cùng công thức `_findSubtitleAt` + `addSubtitles`).

Cách dùng:
    .venv\\Scripts\\python.exe -m backend.tools.sim_pipeline_b --audio wav_test/youtube/The_Big_Bang_Theory.mp3 --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.asr.forced_aligner import AlignedWord, ForcedAlignerService, SubtitleSentence  # noqa: E402
from backend.config import config  # noqa: E402
from backend.core.lookahead_timeline import ContinuousAudioTimeline  # noqa: E402
from backend.tools.diag_pipeline_b import load_audio_16k  # noqa: E402
from backend.ws.lookahead_handler import LookaheadSessionState  # noqa: E402


class FakeASR:
    """Trả về văn bản ASR CỐ ĐỊNH theo từng khối (đo trước bằng ASR thật).

    Chỉ dùng khi muốn chạy nhanh, tất định, không nạp model ASR. Mốc vẫn do ForcedAligner
    THẬT tính trên audio thật nên vẫn tái hiện được sự cố mốc thời gian.
    """

    def __init__(self, texts: List[str]):
        self.texts = list(texts)
        self.calls = 0

    def transcribe_block(self, pcm: np.ndarray, language: Optional[str] = None) -> str:
        i = self.calls
        self.calls += 1
        return self.texts[i] if i < len(self.texts) else ""

    def set_language(self, _lang: str) -> None:
        pass

    def has_pending_work(self) -> bool:
        return False

    async def reset_stream(self, _reason: str = "seek") -> None:
        pass


class CaptureConn:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        self.messages.append(payload)
        return True

    async def send_bytes(self, _data: bytes) -> bool:
        return True

    @property
    def is_closed(self) -> bool:
        return False

    def of_type(self, t: str) -> List[Dict[str, Any]]:
        return [m for m in self.messages if m.get("type") == t]


# ───────────────────────────────────────────── mô phỏng client Extension

class ClientSim:
    """Mô phỏng `SubtitleTimelineQueue._findSubtitleAt` + `addSubtitles` của Extension."""

    def __init__(self, min_display_sec: float = 1.6, hold_after_end: float = 0.6):
        self.items: List[Dict[str, Any]] = []
        self.active: Optional[Dict[str, Any]] = None
        self.min_display_sec = min_display_sec
        self.hold_after_end = hold_after_end
        #: (mốc playhead, item) mỗi lần phụ đề ĐỔI
        self.shown: List[Tuple[float, Optional[Dict[str, Any]]]] = []
        self.dropped_invalid = 0

    def add(self, items: List[Dict[str, Any]]) -> None:
        for it in items:
            s = float(it["start_pts"])
            e = float(it["end_pts"])
            if not (e > s):
                self.dropped_invalid += 1
                continue
            if any(x["original_text"] == it["original_text"] and abs(x["start_pts"] - s) < 1.0
                   for x in self.items):
                continue
            self.items.append(dict(it))
        self.items.sort(key=lambda x: x["start_pts"])
        # Cùng công thức giãn thời lượng tối thiểu như `addSubtitles`.
        for i, it in enumerate(self.items):
            nxt = self.items[i + 1] if i + 1 < len(self.items) else None
            max_allowed = (max(it["start_pts"] + 0.5, nxt["start_pts"] - 0.05)
                           if nxt else it["start_pts"] + 5.0)
            if max_allowed > it["start_pts"]:
                it["end_pts"] = max(it["end_pts"],
                                    min(it["start_pts"] + self.min_display_sec, max_allowed))

    def tick(self, t: float) -> Optional[Dict[str, Any]]:
        matched = None
        for it in self.items:
            if it["start_pts"] <= t < it["end_pts"]:
                matched = it
                break
        if matched is None and self.active is not None:
            if self.active["start_pts"] <= t < self.active["end_pts"] + self.hold_after_end:
                matched = self.active
        if matched is not self.active:
            self.active = matched
            self.shown.append((t, matched))
        return matched


def simulate_playback(
    sent_items: List[Dict[str, Any]],
    start: float,
    end: float,
    step: float = 0.02,
) -> ClientSim:
    """Chạy playhead qua `[start, end]` với các item đã gửi (mô phỏng client đã nhận hết TRƯỚC)."""
    sim = ClientSim()
    sim.add(sent_items)
    t = start
    while t <= end:
        sim.tick(t)
        t += step
    return sim


# ─────────────────────────────────────────────────────── vòng lặp thật

async def run_sim(args: argparse.Namespace) -> int:
    audio_path = REPO_ROOT / args.audio
    pcm_full = load_audio_16k(audio_path)
    total_sec = len(pcm_full) / 16000.0
    limit = min(args.limit, total_sec)
    print(f"[sim] Audio {total_sec:.1f}s, mô phỏng tới {limit:.1f}s "
          f"(đệm tiến {args.buffer_ahead}s, bước {args.step}s)")

    conn = CaptureConn()
    texts = args.texts or []
    asr = FakeASR(texts) if texts else None
    if asr is None:
        from backend.asr.engine import TranscribeEngine

        asr = TranscribeEngine(session_id="sim_pb")

    session = LookaheadSessionState(ws=conn, asr_engine=asr, translation_engine=None)
    session.source_lang = "en"
    session.target_lang = "vi"
    session.current_time = 0.0
    session._session_start_pts = 0.0
    #: Nạp đệm DẦN như trình duyệt: audio chỉ xuất hiện tới `playhead + buffer_ahead`.
    #: Nạp sẵn một đoạn đầu để khối ASR đầu tiên có đủ ngữ cảnh (giống lúc video vừa mở).
    initial_feed = min(limit, float(args.buffer_ahead))
    session.timeline.append(0.0, pcm_full[: int(initial_feed * 16000)])
    _fed_upto = initial_feed
    session._batch_from_pts = 0.0
    session._decoded_end_pts = _fed_upto
    session.tts_enabled = False

    # Tắt TTS worker: sim chỉ quan tâm đường phụ đề.
    session.tts_queue = None

    asyncio.create_task(session._offline_batch_loop())

    # Vòng playhead: mỗi bước cập nhật current_time rồi đánh thức vòng lặp.
    t = 0.0
    step = max(0.05, float(args.step))
    sim = ClientSim()
    _consumed_subs = 0
    _dropped_behind = 0

    async def _drain_client() -> None:
        nonlocal _consumed_subs, _dropped_behind
        subs = conn.of_type("lookahead_subtitles")
        while _consumed_subs < len(subs):
            items = subs[_consumed_subs].get("items", [])
            _consumed_subs += 1
            if items:
                sim.add(items)

    async def _feed_playhead() -> None:
        """Playhead đi theo thời gian THẬT: mỗi `step` mô phỏng, chờ `step/rate` giây tường."""
        nonlocal _fed_upto
        t2 = 0.0
        while t2 <= limit:
            session.current_time = t2
            session.is_paused = False
            want_feed = min(limit, t2 + float(args.buffer_ahead))
            if want_feed > _fed_upto + 1e-3:
                a = int(_fed_upto * 16000)
                b = int(want_feed * 16000)
                session.timeline.append(_fed_upto, pcm_full[a:b])
                _fed_upto = want_feed
            session._decoded_end_pts = _fed_upto
            session._ingest_event.set()
            await _drain_client()
            sim.tick(t2)
            await asyncio.sleep(step / max(0.01, float(args.rate)))
            t2 += step

    await _feed_playhead()
    # Chờ backend xử lý nốt phần audio còn lại rồi thu thập phụ đề cuối
    for _ in range(int(args.flush_sec / 0.1)):
        await asyncio.sleep(0.1)
        await _drain_client()
        if session._batch_from_pts is not None and session._batch_from_pts >= limit - 1e-3:
            break
    await asyncio.sleep(0.5)
    await _drain_client()

    await session.close()

    # ── Báo cáo
    sent_all = list(sim.items)
    print(f"\n[sim] Tổng {len(sent_all)} phụ đề backend đã gửi "
          f"(item sai mốc bị client bỏ: {sim.dropped_invalid})")
    print("\n── PHỤ ĐỀ BACKEND GỬI ──")
    for it in sent_all:
        print(f"   [{it['start_pts']:8.3f} → {it['end_pts']:8.3f}]  {it['original_text']}")

    # Kiểm tra câu nào KHÔNG BAO GIỜ được vẽ trong suốt dải phát
    sim2 = simulate_playback(sent_all, 0.0, limit)
    shown_texts = {s["original_text"] for _, s in sim2.shown if s}
    print("\n── PHỤ ĐỀ KHÔNG BAO GIỜ HIỆN ──")
    missing = [it for it in sent_all if it["original_text"] not in shown_texts]
    for it in missing:
        print(f"   ✗ [{it['start_pts']:.3f} → {it['end_pts']:.3f}]  {it['original_text']}")
    if not missing:
        print("   (không có)")
    print(f"\n[sim] {len(missing)}/{len(sent_all)} câu không bao giờ hiện.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="wav_test/youtube/The_Big_Bang_Theory.mp3")
    ap.add_argument("--limit", type=float, default=140.0)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--buffer-ahead", type=float, default=45.0)
    ap.add_argument("--rate", type=float, default=0.35,
                    help="Tốc độ playhead so với thời gian tường (1.0 = thời gian thật).")
    ap.add_argument("--flush-sec", type=float, default=60.0)
    ap.add_argument("--texts", nargs="*", default=None,
                    help="Danh sách văn bản ASR theo khối (bỏ trống ⇒ dùng ASR thật).")
    return asyncio.run(run_sim(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
