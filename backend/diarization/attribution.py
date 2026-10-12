"""Gán người nói cho phụ đề: TÁCH CÂU TẠI RANH GIỚI LƯỢT NÓI (P0.2) + CỜ CHỒNG TIẾNG (P0.3).

VÌ SAO CẦN (đo thật 2026-10-24, `scratch/diar_stream_probe.py` + `scratch/diar_chunk_split.py`):
`assign_speakers_to_subtitles()` cũ chỉ trả **MỘT nhãn cho MỘT câu** theo overlap lớn nhất. Hệ quả:

* Câu vắt qua ranh giới đổi người nói (`A nói … rồi B nói …` trong cùng một câu ASR) bị dán nhãn
  của người nói NHIỀU HƠN ⇒ LLM dịch nhận sai vai cho nửa sau của câu.
* Khi hai người nói CHỒNG TIẾNG, `turns` do Nemotron-3 trả về **chồng nhau về thời gian** (đo được:
  `speaker_0 0.00–10.08s` bao quanh các mảnh `speaker_1 3.32–10.77s`). Bản cũ vứt hết thông tin đó
  và gán đại một nhãn, nên prompt dịch không hề biết câu này có ≥2 giọng.

HAI ĐIỀU MODULE NÀY **KHÔNG** LÀM (đừng kỳ vọng nhầm):

* **Không tách nguồn âm thanh.** Diarization chỉ nói "ai nói lúc nào"; khi hai người nói cùng lúc
  thì bản phiên âm của ASR vẫn là MỘT chuỗi trộn. Muốn tách chữ theo từng người trong đoạn chồng
  tiếng phải có ASR theo-người-nói (kiểu `SpeakerTaggedASR` của NeMo) hoặc tách nguồn — xem
  `report/` và kết luận ở hội thoại thiết kế. Ở đây chỉ **đánh dấu** để prompt dịch không bịa
  chủ ngữ.
* **Không ổn định nhãn xuyên khối.** Nhãn đánh số theo thứ tự xuất hiện **trong một khối** (đúng
  như `turns` của lần gọi đó). Việc nối danh tính qua các khối là P0.1, chưa làm ở đây.

Module này CỐ Ý không import `backend.asr.forced_aligner` (import đó kéo theo `transcribe.dll` qua
`backend/asr/__init__.py`): mọi thứ ở đây chỉ cần đối tượng "giống từ" có `.text`, `.start_time`,
`.end_time` — nhờ vậy test được thuần Python, không cần native.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Pipeline B luôn đưa PCM 16 kHz vào diarization.
DEFAULT_SAMPLE_RATE = 16000

#: Ngưỡng mặc định (khớp `DiarizationConfig` trong `backend/config.py`).
DEFAULT_SPLIT_MIN_SEC = 0.7
DEFAULT_SPLIT_MIN_WORDS = 2
DEFAULT_OVERLAP_MIN_SEC = 0.30


@dataclass(frozen=True)
class SpeakerTurn:
    """Một lượt nói đã chuẩn hoá về GIÂY (đầu vào từ `--turns-out` là mẫu)."""

    start_sec: float
    end_sec: float
    speaker_id: str


@dataclass
class SpeakerPiece:
    """Một mảnh phụ đề thuộc về MỘT người nói (kết quả của P0.2), kèm cờ chồng tiếng (P0.3)."""

    text: str
    start_time: float
    end_time: float
    words: List[Any] = field(default_factory=list)
    #: Nhãn thân thiện gửi sang LLM dịch ("speaker 1", "speaker 2"…); None = không có dữ liệu.
    speaker: Optional[str] = None
    #: Nhãn thô của model ("speaker_0"…) — giữ lại để log/chẩn đoán.
    raw_speaker: Optional[str] = None
    #: True khi trong khoảng của mảnh có ≥2 người nói CÙNG LÚC (P0.3).
    overlap: bool = False
    overlap_sec: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


# ────────────────────────────────────────────────────────────────── chuẩn hoá lượt nói


def normalize_turns(
    turns: Optional[Iterable[Any]],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> List[SpeakerTurn]:
    """Đổi `turns` của `audiocpp_cli --turns-out` (mẫu) sang giây, bỏ lượt rỗng, sắp theo thời gian.

    Chấp nhận cả dict (`{"start_sample", "end_sample", "speaker_id"}`) lẫn đối tượng đã có
    `start_sec`/`end_sec` (tiện cho test).
    """
    out: List[SpeakerTurn] = []
    sr = float(sample_rate or DEFAULT_SAMPLE_RATE)
    for t in turns or []:
        try:
            if isinstance(t, SpeakerTurn):
                start, end, spk = t.start_sec, t.end_sec, t.speaker_id
            elif isinstance(t, dict):
                if "start_sec" in t or "end_sec" in t:
                    start = float(t.get("start_sec") or 0.0)
                    end = float(t.get("end_sec") or 0.0)
                else:
                    start = float(t.get("start_sample") or 0) / sr
                    end = float(t.get("end_sample") or 0) / sr
                spk = str(t.get("speaker_id") or "")
            else:  # đối tượng kiểu dataclass khác
                start = float(getattr(t, "start_sec"))
                end = float(getattr(t, "end_sec"))
                spk = str(getattr(t, "speaker_id"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not spk or end <= start:
            continue
        out.append(SpeakerTurn(start_sec=start, end_sec=end, speaker_id=spk))
    out.sort(key=lambda x: (x.start_sec, x.end_sec))
    return out


# ────────────────────────────────────────────────────────────────── hình học thời gian


def _overlap_map(span: Tuple[float, float], turns: Sequence[SpeakerTurn]) -> Dict[str, float]:
    """Tổng thời lượng chồng lấn của từng speaker với `span` (giây)."""
    start, end = float(span[0]), float(span[1])
    acc: Dict[str, float] = {}
    if end <= start:
        return acc
    for t in turns:
        ov = min(end, t.end_sec) - max(start, t.start_sec)
        if ov > 0:
            acc[t.speaker_id] = acc.get(t.speaker_id, 0.0) + ov
    return acc


def active_speakers(
    span: Tuple[float, float],
    turns: Sequence[SpeakerTurn],
) -> List[Tuple[str, float]]:
    """Danh sách `(speaker_id, giây hoạt động)` trong `span`, giảm dần theo thời lượng.

    Điểm bằng nhau thì giữ thứ tự xuất hiện (ổn định) nhờ `sorted` trên dict đã chèn theo lượt.
    """
    acc = _overlap_map(span, turns)
    return sorted(acc.items(), key=lambda kv: -kv[1])


def overlap_seconds(span: Tuple[float, float], turns: Sequence[SpeakerTurn]) -> float:
    """Tổng thời gian mà **≥2 người nói cùng hoạt động** bên trong `span`.

    Quét biên (sweep line) trên các lượt đã cắt về `span`. Cùng một mốc thời gian thì xử lý
    kết thúc trước bắt đầu (nhờ sắp theo giá trị delta tăng dần: -1 trước +1) để không tính
    nhầm đoạn độ dài 0.
    """
    start, end = float(span[0]), float(span[1])
    if end <= start:
        return 0.0
    events: List[Tuple[float, int]] = []
    for t in turns:
        a, b = max(start, t.start_sec), min(end, t.end_sec)
        if b > a:
            events.append((a, 1))
            events.append((b, -1))
    if not events:
        return 0.0
    events.sort(key=lambda e: (e[0], e[1]))
    total = 0.0
    active = 0
    prev = events[0][0]
    for when, delta in events:
        if active >= 2 and when > prev:
            total += when - prev
        active += delta
        prev = when
    return total


def dominant_speaker(
    span: Tuple[float, float],
    turns: Sequence[SpeakerTurn],
) -> Optional[str]:
    """Speaker chiếm nhiều thời gian nhất trong `span`; không ai giao nhau thì lấy lượt gần nhất.

    Giữ đúng ngữ nghĩa của `assign_speakers_to_subtitles()` cũ (điểm rơi vào khoảng lặng nhỏ giữa
    hai lượt thì chọn lượt gần nhất) để hành vi khi KHÔNG tách câu không đổi.
    """
    best = active_speakers(span, turns)
    if best:
        return best[0][0]
    if not turns:
        return None
    start, end = float(span[0]), float(span[1])
    mid = (start + end) / 2.0
    nearest_id: Optional[str] = None
    nearest_d = float("inf")
    for t in turns:
        d = min(abs(t.start_sec - mid), abs(t.end_sec - mid))
        if d < nearest_d:
            nearest_d = d
            nearest_id = t.speaker_id
    return nearest_id


def arrival_labels(turns: Sequence[SpeakerTurn]) -> Dict[str, str]:
    """Ánh xạ `speaker_0 → "speaker 1"` theo THỨ TỰ XUẤT HIỆN của lượt nói.

    Dùng thứ tự đến (arrival order) — đúng ngữ nghĩa của Nemotron-3 (`arrival-order speaker
    identities`) và ổn định trong cả khối, thay vì đánh số theo câu như bản cũ.
    """
    labels: Dict[str, str] = {}
    for t in turns:
        if t.speaker_id not in labels:
            labels[t.speaker_id] = f"speaker {len(labels) + 1}"
    return labels


# ────────────────────────────────────────────────────────────────── tách câu (P0.2)


def _count_tokens(text: str) -> int:
    """Đếm token thô của một từ: có khoảng trắng ⇒ đếm từ; không (CJK) ⇒ đếm ký tự chữ/số."""
    t = (text or "").strip()
    if not t:
        return 0
    if " " in t:
        return len([p for p in t.split() if p])
    return len([c for c in t if c.isalnum()])


def _run_tokens(run: Tuple[int, int], words: Sequence[Any]) -> int:
    return sum(_count_tokens(getattr(w, "text", "")) for w in words[run[0] : run[1] + 1])


def _run_duration(run: Tuple[int, int], words: Sequence[Any]) -> float:
    ws = words[run[0] : run[1] + 1]
    if not ws:
        return 0.0
    return max(0.0, float(ws[-1].end_time) - float(ws[0].start_time))


def _group_runs(word_speakers: Sequence[Optional[str]]) -> List[Tuple[int, int]]:
    """Gom các từ liên tiếp cùng speaker thành các đoạn chỉ số `[i0, i1]` (bao gồm hai đầu)."""
    runs: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(word_speakers) + 1):
        if i == len(word_speakers) or word_speakers[i] != word_speakers[start]:
            runs.append((start, i - 1))
            start = i
    return runs


def _merge_short_runs(
    runs: List[Tuple[int, int]],
    words: Sequence[Any],
    min_sec: float,
    min_words: int,
) -> List[Tuple[int, int]]:
    """Gộp các đoạn QUÁ NGẮN vào đoạn lân cận cho tới khi mọi đoạn đạt ngưỡng.

    Lý do: model diarization có lúc "nhấp nháy" nhãn giữa một mạch nói của cùng một người (đo thật:
    mảnh 0,05–0,5 s). Cắt phụ đề tại đó chỉ tạo ra mảnh vụn vô nghĩa.

    Quy tắc gộp: đoạn ngắn gộp vào đoạn TRƯỚC; nếu nó là đoạn đầu tiên thì gộp vào đoạn SAU.
    Lặp lại tới khi ổn định (gộp có thể tạo đoạn mới vẫn còn ngắn).
    """
    runs = list(runs)
    if min_sec <= 0 and min_words <= 1:
        return runs

    def _ok(run: Tuple[int, int]) -> bool:
        if _run_duration(run, words) < min_sec:
            return False
        if _run_tokens(run, words) < max(1, min_words):
            return False
        return True

    guard = 0
    while len(runs) > 1 and guard <= len(runs) * 2 + 2:
        guard += 1
        for i, run in enumerate(runs):
            if _ok(run):
                continue
            target = 1 if i == 0 else i - 1
            lo, hi = sorted((i, target))
            runs[lo] = (runs[lo][0], runs[hi][1])
            del runs[hi]
            break
        else:
            break
    return runs


def _coalesce_same_speaker(
    runs: List[Tuple[int, int]],
    word_speakers: Sequence[Optional[str]],
) -> List[Tuple[int, int]]:
    """Gộp hai đoạn LIỀN KỀ mang CÙNG người nói (chúng chỉ tách nhau vì một mảnh ở giữa đã bị gộp).

    Không có bước này, ca "chớp nhãn ngắn" vẫn ra hai phụ đề cùng một người — đúng lỗi mà
    `test_nhap_nhay_nhan_ngan_thi_gop_lai` khoá lại.
    """
    out: List[Tuple[int, int]] = []
    for run in runs:
        if out and word_speakers[out[-1][0]] == word_speakers[run[0]]:
            out[-1] = (out[-1][0], run[1])
        else:
            out.append(run)
    return out


def attribute_sentences(
    sentences: Sequence[Any],
    turns: Optional[Iterable[Any]],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    split: bool = True,
    split_min_sec: float = DEFAULT_SPLIT_MIN_SEC,
    split_min_words: int = DEFAULT_SPLIT_MIN_WORDS,
    overlap_min_sec: float = DEFAULT_OVERLAP_MIN_SEC,
    language: str = "",
) -> List[SpeakerPiece]:
    """Gán nhãn người nói cho từng câu, TÁCH câu tại ranh giới lượt nói, và đánh cờ chồng tiếng.

    Đầu vào `sentences`: các đối tượng có `.text`, `.start_time`, `.end_time`, `.words`
    (`SubtitleSentence` của `backend/asr/forced_aligner.py`), trong đó `.words` là danh sách từ có
    `.text`, `.start_time`, `.end_time` — mốc thời gian **cùng hệ với `turns`** (đều tính từ đầu khối).

    Trả về danh sách `SpeakerPiece` ĐÃ TÁCH. Câu không tách được giữ NGUYÊN văn bản gốc (không
    dựng lại từ danh sách từ) để không làm mất dấu câu mà `merge_source_text` đã gắn.

    `split=False` ⇒ chỉ gán nhãn + cờ (giữ nguyên số câu) — dùng để tắt P0.2 mà vẫn giữ P0.3.
    """
    turns_n = normalize_turns(turns, sample_rate)
    labels = arrival_labels(turns_n)
    joiner = "" if _is_cjk_language(language) else " "

    out: List[SpeakerPiece] = []

    for sentence in sentences:
        text = str(getattr(sentence, "text", "") or "").strip()
        if not text:
            continue
        words = list(getattr(sentence, "words", None) or [])
        s_start = float(getattr(sentence, "start_time", 0.0) or 0.0)
        s_end = float(getattr(sentence, "end_time", s_start) or s_start)
        span = (s_start, s_end)
        span_overlap_sec = overlap_seconds(span, turns_n)

        def _whole_piece() -> SpeakerPiece:
            raw = dominant_speaker(span, turns_n)
            return SpeakerPiece(
                text=text,
                start_time=s_start,
                end_time=s_end,
                words=words,
                speaker=labels.get(raw) if raw else None,
                raw_speaker=raw,
                overlap=span_overlap_sec >= overlap_min_sec,
                overlap_sec=span_overlap_sec,
            )

        if not turns_n or not split or len(words) < 2:
            out.append(_whole_piece())
            continue

        word_speakers = [
            dominant_speaker((float(w.start_time), float(w.end_time)), turns_n) for w in words
        ]

        # ── KHÔNG CHẺ trong vùng CHỒNG TIẾNG ────────────────────────────────────────────────
        # Trong vùng ≥2 người nói cùng lúc, bản phiên âm là HỖN HỢP nên nhãn từng từ không đáng
        # tin; chẻ ở đó sẽ gán nửa câu trộn cho một người. Vì vậy từ nằm trong vùng chồng tiếng
        # KHÔNG được mở đoạn mới — nó nối tiếp đoạn đang có, và cả mảnh đó được đánh cờ ở dưới.
        prev_spk: Optional[str] = None
        for i, w in enumerate(words):
            w_span = (float(w.start_time), float(w.end_time))
            if overlap_seconds(w_span, turns_n) >= overlap_min_sec:
                if prev_spk is not None:
                    word_speakers[i] = prev_spk
            prev_spk = word_speakers[i]

        runs = _group_runs(word_speakers)
        runs = _merge_short_runs(runs, words, float(split_min_sec), int(split_min_words))
        runs = _coalesce_same_speaker(runs, word_speakers)

        if len(runs) <= 1:
            out.append(_whole_piece())
            continue

        produced = 0
        for i0, i1 in runs:
            run_words = words[i0 : i1 + 1]
            run_text = joiner.join(str(getattr(w, "text", "") or "").strip() for w in run_words).strip()
            if not run_text:
                continue
            r_start = max(s_start, float(run_words[0].start_time))
            r_end = min(s_end, float(run_words[-1].end_time))
            if r_end <= r_start:
                r_end = r_start + 0.01
            r_span = (r_start, r_end)
            r_ov = overlap_seconds(r_span, turns_n)
            raw = dominant_speaker(r_span, turns_n) or word_speakers[i0]
            out.append(
                SpeakerPiece(
                    text=run_text,
                    start_time=r_start,
                    end_time=r_end,
                    words=run_words,
                    speaker=labels.get(raw) if raw else None,
                    raw_speaker=raw,
                    overlap=r_ov >= overlap_min_sec,
                    overlap_sec=r_ov,
                )
            )
            produced += 1

        if produced == 0:  # phòng xa: mọi mảnh đều rỗng ⇒ giữ nguyên câu gốc
            out.append(_whole_piece())

    return out


# ────────────────────────────────────────────────────────────────── tiện ích


def realign_attribution(
    sentences: Sequence[Any],
    pieces: Sequence[SpeakerPiece],
) -> Tuple[List[Optional[str]], List[bool]]:
    """Gán lại nhãn/cờ cho danh sách câu MỚI theo thời gian, sau khi lớp ngắt câu khác chẻ tiếp.

    VÌ SAO CẦN: P0.2 chạy trên câu CÒN `words`, mà `split_oversized_sentences` (lớp ngắt theo trần
    ký tự/từ) chạy SAU đó và có thể chẻ một mảnh thành nhiều phụ đề. Nếu đấu chỉ số theo vị trí thì
    từ câu bị chẻ trở đi mọi nhãn đều lệch. Ở đây mỗi câu MỚI nhận nhãn của mảnh đã-gán chồng thời
    gian nhiều nhất; không chồng giây nào thì lấy mảnh gần trung điểm nhất (giống
    `dominant_speaker` xử lý khoảng lặng nhỏ).
    """
    tags: List[Optional[str]] = []
    overlaps: List[bool] = []
    if not pieces:
        return [None] * len(sentences), [False] * len(sentences)

    for s in sentences:
        s_start = float(getattr(s, "start_time", 0.0) or 0.0)
        s_end = float(getattr(s, "end_time", s_start) or s_start)
        mid = (s_start + s_end) / 2.0
        best: Optional[SpeakerPiece] = None
        best_ov = float("-inf")
        best_dist = float("inf")
        for p in pieces:
            ov = min(s_end, p.end_time) - max(s_start, p.start_time)
            dist = min(abs(p.start_time - mid), abs(p.end_time - mid))
            # Ưu tiên chồng lấn thời gian; hoà thì lấy mảnh gần trung điểm hơn.
            if ov > best_ov or (ov == best_ov and dist < best_dist):
                best, best_ov, best_dist = p, ov, dist
        tags.append(best.speaker if best else None)
        overlaps.append(bool(best.overlap) if best else False)
    return tags, overlaps


def _is_cjk_language(language: str) -> bool:
    return str(language or "").strip().lower() in ("japanese", "chinese", "korean", "cantonese")


def overlap_indices(flags: Sequence[bool]) -> List[int]:
    """Chỉ số 1-based của các mục bị đánh cờ chồng tiếng (dùng để dựng ghi chú prompt)."""
    return [i + 1 for i, f in enumerate(flags or []) if f]
