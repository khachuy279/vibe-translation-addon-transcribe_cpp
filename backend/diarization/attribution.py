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

#: Ký tự KHÔNG THỂ BẮT ĐẦU một từ tiếng Nhật: cỡ nhỏ (っ ゃ ゅ ょ ぁ ぃ ぅ ぇ ぉ ゎ) và âm dài `ー`.
#:
#: VÌ SAO CẦN (log thật 2026-10-12 22:34:40): ForcedAligner tách tiếng Nhật theo **KÝ TỰ**, nên khi
#: P0.2 cắt câu tại ranh giới lượt nói, điểm cắt rơi vào GIỮA TỪ:
#:     "…ああ、ち"  ⏐  "ょっと先ほどの"      ← chữ ちょっと bị chẻ đôi qua hai phụ đề
#: Phụ đề hiện ra hai mảnh vô nghĩa, và prompt dịch cũng nhận hai mảnh cụt đó.
_SMALL_KANA = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶーゝゞヽヾ")
#: Ký tự KHÔNG THỂ BẮT ĐẦU một mảnh phụ đề (dấu câu đóng/kết câu).
_BAD_PIECE_START = set("、。，．・！？!?…‥」』】〉》〕）)]}：:；;")

#: Dịch tối đa bao nhiêu token khi đi tìm ranh giới không chẻ đôi từ (nhỏ để không phá ý nghĩa lượt).
_MAX_BOUNDARY_SHIFT = 3

#: Sai số (giây) để coi một lượt trong lịch sử là "bị MỐC KHỐI cắt ngang".
#:
#: VÌ SAO CẦN (đo thật 2026-10-24, `scratch/p01_real_probe.py`): khối được cắt tại khoảng lặng,
#: nhưng người đang nói dở ngay trước mốc cắt chỉ kịp để lại **0,33 s** trong đuôi ngữ cảnh —
#: DƯỚI ngưỡng khớp 0,35 s ⇒ bị cấp số mới dù rõ ràng là cùng người. Luật liên tục bù đúng ca đó:
#: lượt mới PHỦ QUA mốc khối + lượt lịch sử KẾT THÚC sát mốc khối ⇒ cùng một người.
BOUNDARY_TOUCH_TOL_SEC = 0.5


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
    words: Optional[Sequence[Any]] = None,
) -> Optional[str]:
    """Speaker chiếm nhiều thời gian nhất; ưu tiên tính trên các từ thực tế (words) để tránh khoảng lặng.

    - Nếu có `words` và có speaker giao với các từ: lấy speaker có tổng thời lượng lớn nhất trên các từ.
    - Nếu không có `words` hoặc không ai giao với các từ: lấy theo `span` (không ai giao thì lấy lượt gần nhất).
    """
    if words:
        acc: Dict[str, float] = {}
        for w in words:
            w_st = float(getattr(w, "start_time", 0.0) or 0.0)
            w_en = float(getattr(w, "end_time", w_st) or w_st)
            if w_en > w_st:
                for spk, dur in _overlap_map((w_st, w_en), turns).items():
                    acc[spk] = acc.get(spk, 0.0) + dur
        if acc:
            return sorted(acc.items(), key=lambda kv: -kv[1])[0][0]

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


def _looks_cjk(text: str) -> bool:
    """True nếu văn bản có ký tự Nhật/Trung/Hàn (dùng khi caller KHÔNG truyền `language`).

    VÌ SAO CẦN (lỗi thật 2026-10-12): handler gọi `attribute_sentences()` mà quên `language` ⇒ joiner
    thành `" "` ⇒ tiếng Nhật (aligner tách theo ký tự) bị chèn dấu cách giữa MỌI ký tự:
    `"そ うです。"`, `"あ あ、 ち ょ っ と"`. Suy ra từ chính văn bản để không phụ thuộc caller.
    """
    for ch in text or "":
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF:
            return True
    return False


def _can_start_piece(token: str) -> bool:
    """Token có thể BẮT ĐẦU một mảnh phụ đề không (không phải cỡ nhỏ/âm dài/dấu câu đóng)."""
    t = (token or "").strip()
    if not t:
        return False
    return t[0] not in _SMALL_KANA and t[0] not in _BAD_PIECE_START


def _can_end_piece(token: str) -> bool:
    """Token có thể KẾT THÚC một mảnh phụ đề không (không kết bằng cỡ nhỏ/âm dài)."""
    t = (token or "").strip()
    if not t:
        return False
    return t[-1] not in _SMALL_KANA


def _dominant_label(
    run: Tuple[int, int],
    word_key: Sequence[Optional[str]],
    words: Sequence[Any],
) -> Tuple[Optional[str], int]:
    """Nhãn của một đoạn = nhãn của **PHẦN LỚN thời lượng** trong đoạn, kèm chỉ số từ đại diện.

    VÌ SAO KHÔNG dùng token ĐẦU (lỗi thật 2026-10-12): sau khi dịch ranh giới để không chẻ đôi từ
    (CJK tách theo ký tự), token đầu của đoạn mới có thể vẫn thuộc người nói TRƯỚC — khi đó hai đoạn
    liền nhau bị coi là cùng nhãn và bị gộp lại, làm mất hẳn việc tách câu. Ví dụ thật:
    `…ああ、ちょっと` bị cắt vào giữa `ちょっと`; ranh giới lùi về `ち` thì `ち,ょ` (người cũ) đứng
    trước `っと` (người mới) ⇒ nếu lấy token đầu, cả mảnh bị gán cho người CŨ.
    """
    weights: Dict[Optional[str], float] = {}
    first_idx: Dict[Optional[str], int] = {}
    for i in range(run[0], run[1] + 1):
        k = word_key[i]
        dur = float(getattr(words[i], "duration", 0.0) or 0.0)
        weights[k] = weights.get(k, 0.0) + max(dur, 0.01)
        first_idx.setdefault(k, i)
    if not weights:
        return None, run[0]
    best = max(weights, key=lambda k: (weights[k], -first_idx[k]))
    return best, first_idx[best]


def _coalesce_runs_by_label(
    runs: List[Tuple[int, int]],
    labels: Sequence[Optional[str]],
) -> Tuple[List[Tuple[int, int]], List[Optional[str]]]:
    """Gộp hai đoạn LIỀN KỀ cùng NHÃN (nhãn tính theo đa số thời lượng, xem `_dominant_label`)."""
    out: List[Tuple[int, int]] = []
    labs: List[Optional[str]] = []
    for run, lab in zip(runs, labels):
        if out and labs[-1] == lab:
            out[-1] = (out[-1][0], run[1])
        else:
            out.append(run)
            labs.append(lab)
    return out, labs


def _fix_runs_word_boundaries(
    runs: List[Tuple[int, int]],
    words: Sequence[Any],
    max_shift: int = _MAX_BOUNDARY_SHIFT,
) -> List[Tuple[int, int]]:
    """Dịch ranh giới giữa hai đoạn để KHÔNG chẻ đôi từ (chỉ có ý nghĩa với CJK).

    Điều kiện hợp lệ tại ranh giới `idx` (token `idx` mở đoạn mới):
      * token `idx - 1` kết thúc được một từ, VÀ
      * token `idx` bắt đầu được một từ.
    Tìm `idx` hợp lệ GẦN NHẤT trong phạm vi `±max_shift`; không có ⇒ gộp hai đoạn (thà một phụ đề
    hơi dài còn hơn hai mảnh cụt).
    """
    if len(runs) < 2:
        return runs
    out: List[Tuple[int, int]] = [runs[0]]
    for run in runs[1:]:
        prev = out[-1]
        idx = run[0]
        best: Optional[int] = None
        for shift in range(0, max_shift + 1):
            for cand in ((idx - shift,) if shift == 0 else (idx - shift, idx + shift)):
                if not (prev[0] + 1 <= cand <= run[1]):
                    continue
                if _can_end_piece(getattr(words[cand - 1], "text", "")) and _can_start_piece(
                    getattr(words[cand], "text", "")
                ):
                    best = cand
                    break
            if best is not None:
                break
        if best is None:
            out[-1] = (prev[0], run[1])  # gộp: không tìm được chỗ cắt sạch
            continue
        out[-1] = (prev[0], best - 1)
        out.append((best, run[1]))
    return [r for r in out if r[0] <= r[1]]


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
    """Gộp hai đoạn LIỀN KỀ mang CÙNG người nói (một mảnh ở giữa đã bị gộp nên chúng tách nhau).

    Dùng cho các lớp gộp theo **khoá từng từ** (khi chưa biết nhãn đa số của đoạn). Ở
    `attribute_sentences` lớp tương ứng là `_coalesce_runs_by_label` (nhãn theo đa số thời lượng).
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
    label_map: Optional[Dict[str, str]] = None,
) -> List[SpeakerPiece]:
    """Gán nhãn người nói cho từng câu, TÁCH câu tại ranh giới lượt nói, và đánh cờ chồng tiếng.

    Đầu vào `sentences`: các đối tượng có `.text`, `.start_time`, `.end_time`, `.words`
    (`SubtitleSentence` của `backend/asr/forced_aligner.py`), trong đó `.words` là danh sách từ có
    `.text`, `.start_time`, `.end_time` — mốc thời gian **cùng hệ với `turns`** (đều tính từ đầu khối).

    `label_map` (P0.1): ánh xạ `speaker_id` thô của model → **nhãn PHIÊN** ("speaker 1"…). Có nó thì
    danh tính người nói không đổi số theo từng khối. Không truyền ⇒ đánh số theo thứ tự xuất hiện
    trong chính khối đó (hành vi cũ).

    Trả về danh sách `SpeakerPiece` ĐÃ TÁCH. Câu không tách được giữ NGUYÊN văn bản gốc (không
    dựng lại từ danh sách từ) để không làm mất dấu câu mà `merge_source_text` đã gắn.

    `split=False` ⇒ chỉ gán nhãn + cờ (giữ nguyên số câu) — dùng để tắt P0.2 mà vẫn giữ P0.3.
    """
    turns_n = normalize_turns(turns, sample_rate)
    labels: Dict[str, str] = dict(label_map) if label_map else arrival_labels(turns_n)
    joiner = "" if _is_cjk_language(language) else " "
    #: LƯỚI AN TOÀN: caller quên truyền `language` (lỗi thật 2026-10-12) ⇒ suy từ chính văn bản,
    #: nếu không tiếng Nhật sẽ bị nối bằng dấu cách giữa từng ký tự.
    if joiner == " ":
        sample_text = " ".join(str(getattr(s, "text", "") or "") for s in sentences[:3])
        if _looks_cjk(sample_text):
            joiner = ""
            language = language or "auto-cjk"
    is_cjk = joiner == ""

    def _label_of(raw: Optional[str]) -> Optional[str]:
        return labels.get(raw) if raw else None

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
            raw = dominant_speaker(span, turns_n, words=words)
            return SpeakerPiece(
                text=text,
                start_time=s_start,
                end_time=s_end,
                words=words,
                speaker=_label_of(raw),
                raw_speaker=raw,
                overlap=span_overlap_sec >= overlap_min_sec,
                overlap_sec=span_overlap_sec,
            )

        if not turns_n or not split or len(words) < 2:
            out.append(_whole_piece())
            continue

        word_raw = [
            dominant_speaker((float(w.start_time), float(w.end_time)), turns_n) for w in words
        ]
        #: KHOÁ GOM ĐOẠN = **nhãn phiên**, không phải id thô: khi registry P0.1 ánh xạ hai id thô
        #: (model "chớp nhãn" hoặc đổi số giữa hai cửa sổ) về CÙNG một người, chúng phải nằm chung
        #: một đoạn — nếu gom theo id thô sẽ ra hai phụ đề liền kề cùng nhãn.
        word_key: List[Optional[str]] = [_label_of(r) for r in word_raw]

        # ── KHÔNG CHẺ trong vùng CHỒNG TIẾNG ────────────────────────────────────────────────
        # Trong vùng ≥2 người nói cùng lúc, bản phiên âm là HỖN HỢP nên nhãn từng từ không đáng
        # tin; chẻ ở đó sẽ gán nửa câu trộn cho một người. Vì vậy từ nằm trong vùng chồng tiếng
        # KHÔNG được mở đoạn mới — nó nối tiếp đoạn đang có, và cả mảnh đó được đánh cờ ở dưới.
        prev_key: Optional[str] = None
        for i, w in enumerate(words):
            if overlap_seconds((float(w.start_time), float(w.end_time)), turns_n) >= overlap_min_sec:
                if prev_key is not None:
                    word_key[i] = prev_key
            prev_key = word_key[i]

        runs = _group_runs(word_key)
        runs = _merge_short_runs(runs, words, float(split_min_sec), int(split_min_words))
        if is_cjk and len(runs) > 1:
            # Chỉ với CJK: aligner tách theo KÝ TỰ nên ranh giới có thể rơi vào GIỮA TỪ
            # ("…ああ、ち" ⏐ "ょっと…"). Dịch ranh giới cho sạch rồi gộp lại mảnh quá ngắn.
            runs = _fix_runs_word_boundaries(runs, words)
            runs = _merge_short_runs(runs, words, float(split_min_sec), int(split_min_words))

        def _labels_of(rs: List[Tuple[int, int]]) -> List[Optional[str]]:
            return [_dominant_label(r, word_key, words)[0] for r in rs]

        #: Nhãn của đoạn = bên chiếm ĐA SỐ thời lượng (không phải token đầu — xem `_dominant_label`).
        runs, run_labels = _coalesce_runs_by_label(runs, _labels_of(runs))
        # Gộp làm đoạn dài ra ⇒ bên đa số có thể đổi ⇒ tính lại nhãn rồi gộp lần cuối.
        runs, run_labels = _coalesce_runs_by_label(runs, _labels_of(runs))

        if len(runs) <= 1:
            out.append(_whole_piece())
            continue

        produced = 0
        for (i0, i1), lab in zip(runs, run_labels):
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
            _lab, dom_idx = _dominant_label((i0, i1), word_key, words)
            out.append(
                SpeakerPiece(
                    text=run_text,
                    start_time=r_start,
                    end_time=r_end,
                    words=run_words,
                    speaker=lab if lab is not None else _label_of(word_raw[i0]),
                    raw_speaker=word_raw[dom_idx],
                    overlap=r_ov >= overlap_min_sec,
                    overlap_sec=r_ov,
                )
            )
            produced += 1

        if produced == 0:  # phòng xa: mọi mảnh đều rỗng ⇒ giữ nguyên câu gốc
            out.append(_whole_piece())

    return out


# ────────────────────────────────────────────────────────────────── P0.1 — DANH TÍNH THEO PHIÊN


@dataclass(frozen=True)
class SessionTurn:
    """Một lượt nói đã được GÁN NHÃN PHIÊN, ở hệ thời gian TUYỆT ĐỐI của video (giây)."""

    start_pts: float
    end_pts: float
    label: str


class SpeakerRegistry:
    """Bảng danh tính người nói cho CẢ PHIÊN: `speaker_id` thô của model → nhãn bền ("speaker 1"…).

    VÌ SAO CẦN (đo thật 2026-10-24, `scratch/diar_chunk_split.py`): mỗi lượt gọi diarization là một
    session OFFLINE mới, model đánh số người nói theo **thứ tự xuất hiện trong chính cửa sổ đó**.
    Cùng một người bị đổi số khi sang khối kế:

        khối 1 (0–5 s)    → speaker_0 = A
        khối 2 (5–13,2 s) → speaker_0 = B, speaker_1 = A     ← A từ "speaker 1" thành "speaker 2"

    Prompt dịch lại giả định "speaker 1" là một người xuyên suốt video ⇒ đại từ/vai bị sai.

    CÁCH LÀM: mỗi lượt diarize một cửa sổ có **ngữ cảnh chồng lấn** (đuôi của khối trước). Các lượt
    đã gán nhãn được ghi vào lịch sử theo mốc TUYỆT ĐỐI; lượt mới trong vùng chồng lấn khớp với
    lịch sử ⇒ nhận lại đúng nhãn cũ. Id thô không khớp ai ⇒ người nói MỚI.

    GIỚI HẠN (đừng kỳ vọng nhầm): việc khớp dựa trên **chồng lấn thời gian**, nên danh tính bền khi
    người đó còn xuất hiện trong `history_sec` gần nhất. Ai vắng lâu hơn thế (hoặc sau khi tua tới
    vùng chưa từng gán nhãn) sẽ được cấp số mới. Model chỉ có 8 slot nên quá `max_speakers` thì nhãn
    **lâu nhất không thấy** bị tái sử dụng (không bao giờ sinh "speaker 9").
    """

    def __init__(
        self,
        *,
        max_speakers: int = 8,
        min_match_sec: float = 0.35,
        history_sec: float = 90.0,
        max_history_turns: int = 600,
    ) -> None:
        self._max_speakers = max(1, int(max_speakers))
        self._min_match_sec = max(0.0, float(min_match_sec))
        self._history_sec = max(5.0, float(history_sec))
        self._max_history_turns = max(16, int(max_history_turns))
        self._history: List[SessionTurn] = []
        self._last_seen: Dict[str, float] = {}
        self._next_num = 1
        self._max_pts_seen = 0.0

    # ── truy vấn ─────────────────────────────────────────────────────────────────────────
    @property
    def labels(self) -> List[str]:
        """Các nhãn phiên đã cấp (theo thứ tự cấp)."""
        return [f"speaker {i}" for i in range(1, self._next_num)]

    def history(self) -> List[SessionTurn]:
        return list(self._history)

    # ── cấp nhãn ─────────────────────────────────────────────────────────────────────────
    def _new_label(self) -> str:
        if self._next_num <= self._max_speakers:
            label = f"speaker {self._next_num}"
            self._next_num += 1
            return label
        # Quá trần (model chỉ hỗ trợ 8 slot) ⇒ tái dùng nhãn LÂU NHẤT không thấy.
        candidates = self._last_seen or {f"speaker {i}": 0.0 for i in range(1, self._max_speakers + 1)}
        return min(candidates, key=lambda k: candidates[k])

    def label_window_turns(
        self,
        turns: Sequence[Any],
        window_start_pts: float,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        boundary_pts: Optional[float] = None,
    ) -> Dict[str, str]:
        """Ánh xạ `speaker_id` thô trong cửa sổ → nhãn PHIÊN, bằng cách khớp vùng chồng lấn.

        `turns`: lượt trong cửa sổ (mẫu hoặc giây, gốc = đầu cửa sổ — dùng `normalize_turns`).
        `window_start_pts`: mốc TUYỆT ĐỐI (giây) của đầu cửa sổ.
        `boundary_pts`: mốc TUYỆT ĐỐI của ranh giới khối (đầu khối hiện tại). Khi có, áp thêm
            **luật liên tục**: lượt mới phủ qua mốc khối khớp với lượt lịch sử kết thúc sát mốc khối
            ngay cả khi phần chồng lấn rất nhỏ (`BOUNDARY_TOUCH_TOL_SEC`).
        """
        turns_n = normalize_turns(turns, sample_rate)
        order: List[str] = []
        for t in turns_n:
            if t.speaker_id not in order:
                order.append(t.speaker_id)
        if not order:
            return {}

        #: Khoảng thời gian (tuyệt đối) mà mỗi id thô chiếm, để biết nó có PHỦ QUA mốc khối không.
        spans: Dict[str, Tuple[float, float]] = {}
        for t in turns_n:
            a = window_start_pts + t.start_sec
            b = window_start_pts + t.end_sec
            prev = spans.get(t.speaker_id)
            spans[t.speaker_id] = (min(a, prev[0]), max(b, prev[1])) if prev else (a, b)

        scores: Dict[str, Dict[str, float]] = {}
        continuity: Dict[str, Dict[str, float]] = {}
        if self._history:
            for t in turns_n:
                a = window_start_pts + t.start_sec
                b = window_start_pts + t.end_sec
                for h in self._history:
                    ov = min(b, h.end_pts) - max(a, h.start_pts)
                    if ov <= 0:
                        continue
                    per_label = scores.setdefault(t.speaker_id, {})
                    per_label[h.label] = per_label.get(h.label, 0.0) + ov
                    # Lượt lịch sử bị MỐC KHỐI cắt ngang (kết thúc sát mốc) ⇒ bằng chứng liên tục.
                    if boundary_pts is not None and abs(h.end_pts - boundary_pts) <= BOUNDARY_TOUCH_TOL_SEC:
                        cont = continuity.setdefault(t.speaker_id, {})
                        cont[h.label] = cont.get(h.label, 0.0) + ov

        def _spans_boundary(raw: str) -> bool:
            if boundary_pts is None:
                return False
            start, end = spans[raw]
            return start < boundary_pts - 0.05 and end > boundary_pts + 0.05

        #: `(raw, label, điểm)` cho mọi id thô có bằng chứng — mạnh nhất xét trước.
        candidates: List[Tuple[str, str, float]] = []
        for raw in order:
            best_label, best_sec = None, 0.0
            for label, sec in (scores.get(raw) or {}).items():
                if sec > best_sec:
                    best_label, best_sec = label, sec
            if best_label is not None and best_sec >= self._min_match_sec:
                candidates.append((raw, best_label, best_sec))
                continue
            # Luật liên tục: người đang nói dở ngay trước mốc khối.
            if _spans_boundary(raw):
                for label, sec in (continuity.get(raw) or {}).items():
                    if sec > best_sec:
                        best_label, best_sec = label, sec
                if best_label is not None and best_sec > 0:
                    candidates.append((raw, best_label, best_sec))

        candidates.sort(key=lambda c: -c[2])
        assigned: Dict[str, str] = {}
        taken: Dict[str, str] = {}
        for raw, label, _sec in candidates:
            if raw in assigned or label in taken:
                continue
            assigned[raw] = label
            taken[label] = raw

        for raw in order:
            if raw not in assigned:
                label = self._new_label()
                assigned[raw] = label
                taken.setdefault(label, raw)
        return assigned

    # ── ghi lịch sử ──────────────────────────────────────────────────────────────────────
    def observe(
        self,
        turns: Sequence[Any],
        window_start_pts: float,
        labels: Dict[str, str],
        *,
        already_covered_until_pts: float = float("-inf"),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        """Ghi các lượt THUỘC PHẦN MỚI của cửa sổ vào lịch sử (kèm nhãn phiên).

        `already_covered_until_pts`: lượt kết thúc trước mốc này là phần đuôi ngữ cảnh đã ghi ở lượt
        trước ⇒ bỏ qua để lịch sử không phình bản sao.
        """
        for t in normalize_turns(turns, sample_rate):
            a = window_start_pts + t.start_sec
            b = window_start_pts + t.end_sec
            if b <= already_covered_until_pts + 0.25:
                continue
            label = labels.get(t.speaker_id)
            if not label:
                continue
            self._history.append(SessionTurn(start_pts=a, end_pts=b, label=label))
            self._last_seen[label] = max(b, self._last_seen.get(label, 0.0))
            self._max_pts_seen = max(self._max_pts_seen, b)
        self._prune()

    def _prune(self) -> None:
        cutoff = self._max_pts_seen - self._history_sec
        if cutoff > 0:
            self._history = [h for h in self._history if h.end_pts >= cutoff]
        if len(self._history) > self._max_history_turns:
            self._history = self._history[-self._max_history_turns :]


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
