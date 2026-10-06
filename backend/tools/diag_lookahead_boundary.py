"""Tái hiện/kiểm chứng lỗi ngắt câu của tuyến OFFLINE_BATCH (Pipeline B v3).

Chạy: python backend/tools/diag_lookahead_boundary.py

Mô hình hoá đúng những gì pipeline thật làm, dùng ĐÚNG cấu hình production
(`config.lookahead`) và ĐÚNG code thật (LookaheadChunker + group_words_to_subtitles):
  1. LookaheadChunker.next_chunk() -> khối audio [pts_start, pts_end) + next_read_pts
  2. ASR offline nhận TRỌN khối và trả text cho toàn bộ khối
  3. ForcedAlignerService.group_words_to_subtitles() -> cắt text thành phụ đề
  4. lookahead_handler gửi mỗi phụ đề với pts = chunk.pts_start + word_time

Hai kịch bản audio:
  A. Nhạc nền TO (không frame nào trũng sâu) -> không có khoảng lặng => nhánh cưỡng bức + overlap
  B. Nhạc nền NHỎ (nhiều điểm trũng ngắn)  -> nhánh điểm trũng / khoảng lặng yếu
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"D:\vibe-translation-addon-transcribe_cpp")

from backend.config import config
from backend.core.dedup import normalize_for_dedup
from backend.core.lookahead_chunker import LookaheadChunker
from backend.core.lookahead_timeline import ContinuousAudioTimeline
from backend.asr.forced_aligner import AlignedWord, ForcedAlignerService
from backend.ws.lookahead_handler import _ends_sentence

SR = 16000

SCRIPT = [
    "That man, and why?",
    "At the end of the day, he's just another Hollywood phony.",
    "Is it really worth getting upset about?",
    "Yeah, they say don't meet your heroes.",
    "Don't peek behind that curtain of fame and celebrity,",
    "because if you do, you'll see them as they really are,",
    "degenerate carnival folk.",
    "Come on, he's a retired kids' show host.",
    "Yeah, it's even worse.",
    "They're using the sweet candy of science to trick children into loving him, pervert.",
    "Have you ever thought about why Arthur didn't want you to read his paper?",
    "Yes, I have.",
]


def build_ground_truth(repeat: int = 3):
    words: list[AlignedWord] = []
    t = 0.5
    for _ in range(repeat):
        for sent in SCRIPT:
            for w in sent.split():
                words.append(AlignedWord(text=w, start_time=t, end_time=t + 0.30))
                t += 0.36  # 300ms nói + 60ms khe -> KHÔNG bao giờ đạt 250ms im lặng
    return words, t + 2.0


def build_audio(gt, duration_sec: float, bed_amp: float) -> np.ndarray:
    n = int(duration_sec * SR)
    rng = np.random.default_rng(7)
    t_axis = np.arange(n) / SR
    bed = bed_amp * np.sin(2 * np.pi * 110.0 * t_axis) + (bed_amp * 0.4) * rng.standard_normal(n)
    speech = np.zeros(n, dtype=np.float32)
    for w in gt:
        i0, i1 = int(w.start_time * SR), int(w.end_time * SR)
        env = np.hanning(max(1, i1 - i0))
        speech[i0:i1] += (0.25 * rng.standard_normal(i1 - i0) * env).astype(np.float32)
    return (speech + bed).astype(np.float32)


def make_chunker(pcm: np.ndarray) -> LookaheadChunker:
    """Dựng chunker bằng ĐÚNG cấu hình production (giống lookahead_handler)."""
    la = config.lookahead
    tl = ContinuousAudioTimeline(sample_rate=SR, max_storage_sec=3600.0)
    tl.append(0.0, pcm)
    return LookaheadChunker(
        timeline=tl,
        sample_rate=SR,
        min_window_sec=float(la.batch_min_sec),
        min_silence_ms=float(la.batch_min_silence_ms),
        strong_silence_ms=float(la.batch_strong_silence_ms),
        silence_rel_db=float(la.batch_silence_rel_db),
        silence_floor_rms=float(la.batch_silence_floor_rms),
        min_rms_ratio=float(la.batch_min_rms_ratio),
        min_rms_hold_ms=float(la.batch_min_rms_hold_ms),
        dip_search_sec=float(la.batch_dip_search_sec),
    )


def word_dup(prev_text: str, cur_text: str) -> int:
    pn = [w.strip(".,?!").lower() for w in prev_text.split()]
    cn = [w.strip(".,?!").lower() for w in cur_text.split()]
    for k in range(min(len(pn), len(cn)), 0, -1):
        if pn[-k:] == cn[:k]:
            return k
    return 0


def run(label: str, bed_amp: float, gt, total: float) -> dict:
    pcm = build_audio(gt, total, bed_amp)
    chunker = make_chunker(pcm)
    la = config.lookahead

    print("\n" + "=" * 104)
    print(f"{label}  (nhạc nền amp={bed_amp})")
    print("=" * 104)

    chunks = []
    from_pts = 0.0
    for _ in range(12):
        c = chunker.next_chunk(from_pts=from_pts, is_stream_end=False)
        if c is None:
            break
        chunks.append(c)
        from_pts = c.next_read_pts
        if from_pts >= total - 1.0:
            break

    print("-- Các khối do LookaheadChunker cắt ra (cấu hình production) --")
    for c in chunks:
        ov = c.pts_end - c.next_read_pts
        print(
            f"   [{c.pts_start:6.2f}s -> {c.pts_end:6.2f}s] dur={c.duration:5.2f}s "
            f"mode={c.fallback_mode:<15s} strength={c.boundary_strength or '-':<6s} "
            f"next={c.next_read_pts:6.2f}s"
            + (f"  <== CHỒNG LẤN {ov:.2f}s (sẽ trừ ở tầng từ)" if ov > 1e-6 else "")
        )

    # Mô phỏng ĐÚNG phía nhận: cắt khối -> ASR trọn khối -> align -> TRỪ CHỒNG LẤN -> GHÉP mảnh
    # cuối chưa kết câu -> gom câu (giống hệt `LookaheadSessionState._offline_batch_loop`).
    emitted: list[tuple[float, float, str]] = []
    prev_emitted_end: float | None = None
    tail_norm: list[str] = []
    carry: list[AlignedWord] = []
    carried_blocks = 0
    print("\n-- Phụ đề THẬT SỰ phát ra --")
    for ci, c in enumerate(chunks):
        words = [
            w for w in gt
            if w.start_time >= c.pts_start - 1e-6 and w.end_time <= c.pts_end + 1e-6
        ]
        if not words:
            continue
        rel = [
            AlignedWord(w.text, w.start_time - c.pts_start, w.end_time - c.pts_start)
            for w in words
        ]
        # (1) Trừ chồng lấn: theo mốc, rồi theo CHUỖI TỪ
        dropped_ts = dropped_seq = 0
        if prev_emitted_end is not None and c.pts_start < prev_emitted_end - 1e-3:
            kept = [w for w in rel if c.pts_start + w.start_time >= prev_emitted_end - 0.05]
            dropped_ts = len(rel) - len(kept)
            rel = kept
            if tail_norm and rel:
                window = sum(
                    1 for w in rel if c.pts_start + w.start_time < prev_emitted_end + 0.35
                )
                head = [normalize_for_dedup(w.text) for w in rel]
                for k in range(min(len(tail_norm), len(rel), window + 1), 0, -1):
                    if not any(head[:k]):
                        continue
                    hit = any(
                        len(tail_norm) - k - s >= 0
                        and tail_norm[len(tail_norm) - k - s : len(tail_norm) - s] == head[:k]
                        for s in range(0, 4)
                    )
                    if hit:
                        dropped_seq = k
                        rel = rel[k:]
                        break
        # (2) Ghép mảnh cuối chưa kết câu của khối trước
        merged_note = ""
        if carry:
            rel = [
                AlignedWord(w.text, w.start_time - c.pts_start, w.end_time - c.pts_start)
                for w in carry
            ] + rel
            merged_note = f"  (ghép {len(carry)} từ giữ lại)"
            carry = []

        subs = ForcedAlignerService.group_words_to_subtitles(
            rel,
            language="English",
            max_words=int(la.batch_sub_max_words),
            max_duration_sec=float(la.batch_sub_max_duration_sec),
            max_chars=int(la.batch_sub_max_chars),
            min_words=int(la.batch_sub_min_words),
            defer_words=int(la.batch_sub_defer_words),
            sentence_max_words=int(la.batch_sub_sentence_max_words),
            sentence_max_duration_sec=float(la.batch_sub_sentence_max_duration_sec),
        )
        # (3) Giữ lại mảnh cuối CHƯA kết câu khi khối bị cắt giữa câu
        if subs and not c.is_silence_boundary and not _ends_sentence(subs[-1].text) and subs[-1].words:
            carry = [
                AlignedWord(w.text, c.pts_start + w.start_time, c.pts_start + w.end_time)
                for w in subs[-1].words
            ]
            subs = subs[:-1]
            carried_blocks += 1

        print(
            f"\n  Khối {ci} [{c.pts_start:.2f}->{c.pts_end:.2f}] mode={c.fallback_mode}"
            + (f"  (trừ {dropped_ts} từ theo mốc + {dropped_seq} theo chuỗi)" if dropped_ts or dropped_seq else "")
            + merged_note
            + (f"  ⇒ GIỮ LẠI mảnh cuối {len(carry)} từ" if carry else "")
        )
        for s in subs:
            a, b = c.pts_start + s.start_time, c.pts_start + s.end_time
            emitted.append((a, b, s.text))
            print(f"    [{a:6.2f}-{b:6.2f}] {s.text}")
            for tok in s.text.split():
                nt = normalize_for_dedup(tok)
                if nt:
                    tail_norm.append(nt)
            tail_norm = tail_norm[-40:]
        if subs:
            prev_emitted_end = max(c.pts_start + s.end_time for s in subs)

    print("\n-- Chẩn đoán phụ đề PHÁT RA --")
    dups = 0
    for i in range(1, len(emitted)):
        k = word_dup(emitted[i - 1][2], emitted[i][2])
        if k:
            dups += 1
            print(f"  ✗ LẶP {k} từ: '{' '.join(emitted[i-1][2].split()[-k:])}'")
            print(f"      [{emitted[i-1][0]:.2f}-{emitted[i-1][1]:.2f}] {emitted[i-1][2]}")
            print(f"      [{emitted[i][0]:.2f}-{emitted[i][1]:.2f}] {emitted[i][2]}")
    print(f"  => {dups} cặp phụ đề liền nhau bị lặp từ ở ranh giới / {max(1, len(emitted)-1)} cặp")

    frags = [(a, b, t) for a, b, t in emitted if not _ends_sentence(t)]
    # Phân biệt hai loại: (a) cắt ở DẤU PHẨY = câu quá dài được chia để hiển thị (theo thiết kế);
    # (b) KHÔNG có dấu câu nào = mảnh cụt thật (lỗi).
    hard_frags = [
        (a, b, t) for a, b, t in frags
        if not t.strip().endswith((",", ";", "、", "；", "—", "-"))
    ]
    for a, b, t in hard_frags:
        print(f"  ✗ MẢNH CỤT THẬT (không có dấu câu nào): [{a:.2f}-{b:.2f}] {t}")
    print(
        f"  => {len(hard_frags)}/{len(emitted)} phụ đề là MẢNH CỤT THẬT; "
        f"{len(frags) - len(hard_frags)} phụ đề cắt ở dấu phẩy (câu quá dài, theo thiết kế)"
    )

    # Ranh giới GIẢI MÃ rơi vào giữa câu: không sao, miễn là tầng phụ đề đã GHÉP lại (mục trên).
    splits = 0
    for i, c in enumerate(chunks[:-1]):
        before = [w for w in gt if w.start_time < c.pts_end - 1e-6]
        after = [w for w in gt if w.start_time >= c.pts_end - 1e-6]
        if not before or not after:
            continue
        for sent in SCRIPT:
            toks = sent.split()
            if before[-1].text in toks and after[0].text in toks:
                ib, ia = toks.index(before[-1].text), toks.index(after[0].text)
                if ia == ib + 1:
                    splits += 1
                break
    print(
        f"  => {splits} ranh giới GIẢI MÃ rơi vào giữa câu (bình thường — ranh giới khối chỉ là "
        f"ranh giới giải mã, câu được ghép lại ở tầng phụ đề)"
    )
    return {"dups": dups, "frags": len(hard_frags), "clause_splits": len(frags) - len(hard_frags),
            "splits": splits, "chunks": len(chunks), "subs": len(emitted)}


def demo_sentence_split(la) -> None:
    """So sánh ngắt câu TRƯỚC/SAU khi gắn lại dấu câu từ văn bản ASR.

    Đây là sự cố 2026-10-02 (ảnh chụp màn hình thật): ASR offline trả câu hoàn hảo, nhưng
    `Qwen3-ForcedAligner` đã bỏ hết dấu câu khi tokenize nên tầng gom câu chỉ còn cắt theo trần
    `max_words=10` ⇒ phụ đề ra mảnh cụt, mất dấu câu, và bản dịch nhận được mảnh vụn.
    """
    from backend.core.dedup import normalize_for_dedup  # noqa: F401  (giữ import rõ ràng)

    samples = [
        "You took ballet. God, you never listen. I'm excited to meet Emily. Me too. "
        "I just hope he doesn't blow it.",
        "In front of me. I promise I'll be on my best behavior. You better be. "
        "No jokes about how close I am with my dog, or the truth about how close I am with my dog. "
        "You got it. No jokes about the year I took ballet.",
    ]
    print("\n" + "=" * 104)
    print("NGẮT CÂU: TRƯỚC vs SAU khi gắn lại dấu câu từ văn bản ASR")
    print("=" * 104)
    for text in samples:
        tokens = text.split()
        step = 12.0 / max(1, len(tokens))
        # Item ĐÚNG như aligner thật trả về: đã bị bỏ hết dấu câu (chỉ còn chữ/số + nháy đơn)
        raw_items = [
            AlignedWord(
                text="".join(ch for ch in tok if ch.isalnum() or ch == "'"),
                start_time=round(i * step, 3),
                end_time=round((i + 1) * step - 0.05, 3),
            )
            for i, tok in enumerate(tokens)
        ]
        fixed = ForcedAlignerService.merge_source_text(raw_items, text)
        kw = dict(
            max_words=int(la.batch_sub_max_words),
            max_duration_sec=float(la.batch_sub_max_duration_sec),
            max_chars=int(la.batch_sub_max_chars),
            min_words=int(la.batch_sub_min_words),
            defer_words=int(la.batch_sub_defer_words),
            sentence_max_words=int(la.batch_sub_sentence_max_words),
            sentence_max_duration_sec=float(la.batch_sub_sentence_max_duration_sec),
        )
        before = ForcedAlignerService.group_words_to_subtitles(raw_items, language="English", **kw)
        after = ForcedAlignerService.group_words_to_subtitles(fixed, language="English", **kw)
        print(f"\nASR (nguyên văn): {text}")
        print("  TRƯỚC (item mất dấu câu):")
        for s in before:
            print(f"     • {s.text}")
        print("  SAU (gắn lại dấu câu từ ASR):")
        for s in after:
            print(f"     ✓ {s.text}")


def main() -> None:
    gt, total = build_ground_truth(repeat=3)
    la = config.lookahead
    print(
        f"Ground truth: {len(gt)} từ, {total:.1f}s hội thoại liên tục (khe giữa từ = 60ms < "
        f"{la.batch_min_silence_ms:.0f}ms)"
    )
    print(
        f"Cấu hình: cỡ khối tối đa (trần audio)={la.batch_max_audio_sec}s "
        f"| ranh giới mạnh >= {la.batch_strong_silence_ms}ms "
        f"| ngưỡng lặng = p90 - {la.batch_silence_rel_db}dB"
    )
    a = run("KỊCH BẢN A — nhạc nền TO", 0.05, gt, total)
    b = run("KỊCH BẢN B — nhạc nền NHỎ", 0.020, gt, total)
    demo_sentence_split(la)
    print("\n" + "=" * 104)
    print(f"TỔNG KẾT ranh giới: A={a}  B={b}")
    print("=" * 104)


if __name__ == "__main__":
    main()
