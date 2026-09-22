"""SEG — tách câu HẬU-ASR (VAD > ASR > SEG).

Vì sao cần tầng này
--------------------
VAD cắt theo *im lặng*. Trong phim/talkshow tiếng Nhật, người nói có thể:
  * ngắt rất ngắn ở dấu `、` ⇒ VAD cắt sớm ⇒ câu vụn ⇒ dịch sai nghĩa;
  * nói liên tục > 10 s không có khoảng lặng ⇒ VAD cắt muộn ⇒ câu dài, khó đọc.
ASR (Qwen3-ASR) đã tự sinh dấu câu, nên ta có thêm một tín hiệu độc lập với VAD để
quyết định "câu này đã trọn nghĩa chưa".

Phạm vi module
--------------
`boundary.py` chỉ làm việc trên VĂN BẢN (thuần, test được bằng `test.tsv`):
  * `split_complete_sentences(text)` — tách một văn bản đã hoàn chỉnh.
  * `SentenceCompleter` — máy trạng thái cho văn bản ASR *tăng dần*: mỗi scan
    (preview) có thể VIẾT LẠI phần đuôi, nên chỉ chốt khi có bằng chứng:
      - `punct_tail`   : sau dấu kết câu đã có thêm chữ của câu MỚI ⇒ câu trước chắc chắn xong.
      - `punct_stable` : dấu kết câu đứng ở cuối preview và đứng yên đủ lâu/đủ số scan.
      - `max_chars`    : quá dài mà không có dấu ⇒ cắt cưỡng bức (bảo vệ khả năng đọc).
      - `flush`        : hết vùng nói (VAD END) ⇒ chốt phần còn lại.

Việc map "ranh giới văn bản" sang "ranh giới mẫu audio" nằm ở `segmenter.py`
(`StreamingSegmenter` + `choose_anchor`), KHÔNG ở đây.
"""

from dataclasses import dataclass
import re
import time
from typing import List, Optional, Tuple

#: Dấu kết câu (tiếng Nhật + Latin). `、`/`,` KHÔNG phải dấu kết câu.
TERMINAL_PUNCT = "。！？!?…"

#: Dấu ngắt mệnh đề — chỉ dùng khi phải cắt cưỡng bức vì câu quá dài.
SECONDARY_PUNCT = "、，,;；:："

#: Ký tự đóng đi kèm dấu kết câu: `「ああ。」` ⇒ câu kết thúc SAU `」`.
CLOSERS = "」』）)】》〉”\"'"

#: Ký tự không tính là "nội dung" khi đo độ dài câu.
_IGNORED = set(TERMINAL_PUNCT + SECONDARY_PUNCT + CLOSERS + " \t\n\u3000") | set("「『（(【《〈“")


def content_len(text: str) -> int:
    """Số ký tự NỘI DUNG (bỏ dấu câu, khoảng trắng, ngoặc)."""
    return sum(1 for ch in text if ch not in _IGNORED)


_RE_LATIN_WORD = re.compile(r"[A-Za-z0-9'’\-]+")


def count_tokens(text: str) -> int:
    """Đếm "từ" theo cùng quy ước với `min_words_to_commit` (CJK: ký tự, Latin: từ).

    Vì sao SEG cần: cắt một câu chỉ 1 từ (`Laughter.`, `Okay.`, `Zero.`) rồi để tầng trên
    lọc/gộp lại làm VÙNG AUDIO KHÔNG TIẾN ⇒ nhịp sau lại cắt đúng chỗ đó (log phim thật
    2026-09-22 lặp 4 lần + 4 lần gọi timer). Không cắt thì không lặp.
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    latin = len(_RE_LATIN_WORD.findall("".join(ch if ch.isascii() else " " for ch in text)))
    return cjk + latin


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (0x3040 <= cp <= 0x30FF) or (0x4E00 <= cp <= 0x9FFF) or (0xAC00 <= cp <= 0xD7AF)


def is_terminal_at(text: str, i: int) -> bool:
    """`text[i]` có phải dấu kết câu không (đã trừ số thập phân / viết tắt Latin)."""
    if i < 0 or i >= len(text):
        return False
    ch = text[i]
    if ch in TERMINAL_PUNCT:
        return True
    if ch != ".":
        return False
    prev = text[i - 1] if i > 0 else ""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if prev.isdigit() and nxt.isdigit():
        return False        # 3.5
    if nxt and not (nxt.isspace() or nxt in CLOSERS or nxt in TERMINAL_PUNCT):
        return False        # U.S. / e.g.
    return True


def sentence_end(text: str, i: int) -> int:
    """Vị trí NGAY SAU câu kết thúc ở dấu `text[i]` (nuốt luôn ngoặc đóng)."""
    j = i + 1
    while j < len(text) and text[j] in CLOSERS:
        j += 1
    return j


def first_boundary(text: str) -> Optional[Tuple[int, int]]:
    """Tìm dấu kết câu ĐẦU TIÊN: trả `(vị_trí_dấu, vị_trí_hết_câu)`."""
    for i, ch in enumerate(text):
        if ch in TERMINAL_PUNCT or ch == ".":
            if is_terminal_at(text, i):
                return i, sentence_end(text, i)
    return None


def last_secondary_before(text: str, limit: int) -> Optional[int]:
    """Vị trí dấu ngắt mệnh đề CUỐI CÙNG nằm trước `limit` (để cắt cưỡng bức)."""
    best = None
    for i in range(min(limit, len(text))):
        if text[i] in SECONDARY_PUNCT:
            best = i
    return best


def split_complete_sentences(
    text: str,
    *,
    min_chars: int = 2,
    max_chars: int = 80,
) -> List[str]:
    """Tách một văn bản ĐÃ HOÀN CHỈNH thành các câu.

    Dùng cho: (a) test đối chiếu `test.tsv`, (b) văn bản ASR chạy lại trên mảnh audio
    đã đóng (final text) — lúc đó không còn chuyện dấu câu đổi theo scan nữa.
    """
    out: List[str] = []
    buf = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf += ch
        i += 1

        if (ch in TERMINAL_PUNCT or ch == ".") and is_terminal_at(text, i - 1):
            # Nuốt luôn ngoặc đóng: `「ああ。」` ⇒ ranh giới nằm SAU `」`.
            while i < n and text[i] in CLOSERS:
                buf += text[i]
                i += 1
            if content_len(buf) >= min_chars:
                out.append(buf.strip())
                buf = ""
            continue

        if content_len(buf) >= max_chars:
            cut = last_secondary_before(buf, len(buf))
            if cut is not None and content_len(buf[:cut + 1]) >= min_chars:
                out.append(buf[:cut + 1].strip())
                buf = buf[cut + 1:]
            else:
                sp = buf.rfind(" ", max(0, len(buf) - 12))
                if sp > min_chars and not buf[sp:].strip().endswith(tuple(TERMINAL_PUNCT)):
                    out.append(buf[:sp].strip())
                    buf = buf[sp:]
                else:
                    out.append(buf.strip())
                    buf = ""

    rest = buf.strip()
    if rest:
        if content_len(rest) >= min_chars or not out:
            out.append(rest)
        else:
            out[-1] = out[-1] + rest
    return [s for s in out if s]


@dataclass
class SegmentDecision:
    """Một câu đã được chốt bởi tầng SEG."""

    text: str
    end_index: int          # vị trí kết thúc (exclusive) trong preview text lúc chốt
    reason: str             # punct_tail | punct_stable | max_chars | flush
    confidence: float = 1.0
    stale_text: bool = False  # phần văn bản TRƯỚC đó đã bị ASR viết lại (engine phải chạy lại ASR)


class SentenceCompleter:
    """Máy trạng thái chốt câu trên văn bản ASR TĂNG DẦN (có thể bị viết lại).

    Hợp đồng quan trọng: KHÔNG BAO GIỜ chốt một câu chỉ vì "thấy dấu kết câu". Dấu câu
    của ASR streaming thay đổi theo từng scan (ví dụ `、` thành `。`), nên phải có bằng
    chứng thứ hai (chữ của câu kế tiếp, hoặc dấu đứng yên đủ lâu).
    """

    def __init__(
        self,
        *,
        min_chars: int = 2,
        min_words: int = 2,
        max_chars: int = 60,
        tail_min_chars: int = 4,
        tail_scans: int = 2,
        tail_stable_ms: float = 280.0,
        stable_ms: float = 350.0,
        stable_scans: int = 2,
        allow_stable_cut: bool = True,
    ):
        self.min_chars = int(min_chars)
        #: Số "từ" tối thiểu của một câu để được phép cắt (0 = tắt). Dùng CHUNG giá trị với
        #: `min_words_to_commit` để SEG không cắt ra câu mà tầng trên chắc chắn lọc bỏ.
        self.min_words = int(min_words)
        self.max_chars = int(max_chars)
        #: Cho phép chốt khi dấu kết câu đứng yên đủ lâu mà CHƯA có câu mới (`punct_stable`).
        #: Với tiếng Nhật nên để False: Qwen3 thả `。` rất sớm (`終え。`, `済ませ。`, `とは。`)
        #: nên cắt theo độ ổn định sẽ chẻ câu; chờ câu mới hoặc im lặng VAD an toàn hơn.
        self.allow_stable_cut = bool(allow_stable_cut)
        self.tail_min_chars = int(tail_min_chars)
        self.tail_scans = int(tail_scans)
        self.tail_stable_ms = float(tail_stable_ms)
        self.stable_ms = float(stable_ms)
        self.stable_scans = int(stable_scans)

        self.reset()

    # ------------------------------------------------------------------ trạng thái
    def reset(self) -> None:
        self._cursor = 0            # con trỏ: đầu phần CHƯA chốt
        self._sentence_start = 0    # đầu CÂU đang chờ chốt
        self._emitted_text = ""     # tiền tố đã chốt (ĐÓNG BĂNG, không bao giờ lùi)
        self._last_span: Optional[Tuple[int, int]] = None
        self._pending_punct: Optional[int] = None   # tương đối so với `_cursor`
        self._pending_since = 0.0
        self._pending_scans = 0
        self._tail_punct: Optional[int] = None      # bộ đếm riêng cho tail (punct_tail)
        self._tail_since = 0.0
        self._tail_scans = 0
        self.stale_text = False     # ASR đã viết lại phần ĐÃ chốt ⇒ engine phải chạy lại
        self.resync_count = 0
        #: Chẩn đoán cho log trace từng nhịp (xem `trace_state()`).
        self.last_hold = ""
        self.last_boundary_index = -1
        self.last_tail = ""

    @property
    def committed_text(self) -> str:
        """Phần văn bản đã chốt (tiền tố của preview)."""
        return self._emitted_text

    # ------------------------------------------------------------------ nội bộ
    def _resync(self, text: str) -> None:
        """ASR viết lại văn bản ⇒ lùi con trỏ về ĐẦU CÂU đang chờ (không phát lại câu cũ).

        Đo bằng ASR thật (xem `scratch/seg_ja_probe.py`): cùng một đoạn audio, Qwen3-ASR
        có thể đổi cả nội dung giữa hai lần quét (`二つ` ↔ `2つ`, `貿易` ↔ `防衛`). Vì vậy:
          * phần ĐÃ chốt được ĐÓNG BĂNG (`_emitted_text` không bao giờ lùi) — không phát
            lại câu cũ (sẽ sinh phụ đề trùng);
          * câu ĐANG CHỜ được tính lại từ đầu câu theo văn bản MỚI;
          * nếu văn bản mới lệch ngay trong phần đã chốt thì đánh dấu `stale_text=True`:
            engine PHẢI chạy lại ASR trên mảnh audio đã cắt, không được tin văn bản tăng dần.
        """
        if text.startswith(self._emitted_text):
            return
        k = 0
        limit = min(len(text), len(self._emitted_text))
        while k < limit and text[k] == self._emitted_text[k]:
            k += 1
        if k < self._sentence_start:
            self.stale_text = True
        self._cursor = self._sentence_start
        self._pending_punct = None
        self._pending_scans = 0
        self.resync_count += 1

    def _forced_cut(self, tail: str) -> int:
        cut = last_secondary_before(tail, self.max_chars)
        if cut is not None and content_len(tail[:cut + 1]) >= self.min_chars:
            return cut + 1
        idx = min(self.max_chars, len(tail))
        # Không cắt giữa một "từ" Latin: lùi về khoảng trắng gần nhất (trong 12 ký tự).
        sp = tail.rfind(" ", max(0, idx - 12), idx)
        if sp > self.min_chars:
            return sp
        return idx

    # ------------------------------------------------------------------ API
    def observe(self, preview_text: str, now: Optional[float] = None) -> List[SegmentDecision]:
        """Nạp một bản preview mới, trả về các câu ĐỦ ĐIỀU KIỆN chốt (thường 0 hoặc 1).

        Luôn cập nhật `self.last_hold` (mô tả VÌ SAO chưa chốt) để tầng trên ghi trace
        từng nhịp ASR — cần cho việc tinh chỉnh mốc ngắt câu bằng mắt người.
        """
        now = time.monotonic() if now is None else now
        text = preview_text or ""
        self.last_hold = ""
        self.last_boundary_index = -1
        self.last_tail = ""
        self._resync(text)

        out: List[SegmentDecision] = []
        while True:
            tail = text[self._cursor:]
            if not tail.strip():
                self.last_hold = "không còn chữ chưa chốt"
                break

            cand = first_boundary(tail)
            if cand is None:
                self.last_hold = (
                    f"chưa có dấu kết câu (dài {content_len(text[self._sentence_start:])}"
                    f"/{self.max_chars} ký tự nội dung)"
                )
                if content_len(text[self._sentence_start:]) >= self.max_chars:
                    cut = self._forced_cut(tail)
                    if cut > 0:
                        out.append(self._commit(text, self._cursor + cut, "max_chars", 0.5))
                        continue
                break

            idx, end = cand
            piece_so_far = text[self._sentence_start:self._cursor + end]
            if content_len(piece_so_far) < self.min_chars or self._too_few_words(piece_so_far):
                # Câu quá ngắn (`え。`) hoặc quá ít từ (`Laughter.`) ⇒ gộp với câu kế tiếp
                # thay vì chốt vụn. Chốt vụn còn gây LẶP: tầng trên gộp lại ⇒ vùng audio
                # không tiến ⇒ nhịp sau cắt đúng chỗ đó (đã gặp 4 lần liên tiếp trong log).
                nxt = first_boundary(tail[end:])
                if nxt is None:
                    self._mark_pending(idx, now)
                    self._mark_tail(idx, now, False)
                    self.last_hold = (
                        f"câu mới {count_tokens(piece_so_far)}/{self.min_words} từ "
                        f"(hoặc < {self.min_chars} ký tự) — chờ gộp"
                    )
                    break
                idx, end = end + nxt[0], end + nxt[1]

            after = tail[end:]
            self.last_boundary_index = self._cursor + end
            self.last_tail = after.strip()
            self._mark_pending(idx, now)
            self._mark_tail(idx, now, content_len(after) > 0)
            # `punct_tail`: phải thấy ĐỦ chữ của câu mới VÀ tail phải xuất hiện ĐỦ NHỊP.
            # Lý do (đo từ log phim thật 2026-09-22): ASR có lúc thả dấu `.` SAI giữa câu
            # (`Don't try and trick.` / `Me into buying…`) rồi tự sửa ở nhịp sau. Chốt ngay
            # ở nhịp đầu tiên ⇒ câu bị chẻ đôi. Chờ thêm nhịp vừa cho ASR cơ hội sửa dấu,
            # vừa bảo đảm tail thật sự là câu mới (không phải một từ đang viết dở).
            if self._tail_ready(after) and self._is_tail_stable(now):
                out.append(self._commit(text, self._cursor + end, "punct_tail", 0.95))
                continue
            # Dấu kết câu nằm ở CUỐI preview (không có tail): chỉ chốt khi được phép.
            if (
                self.allow_stable_cut
                and content_len(after) == 0
                and self._is_stable(idx, now)
            ):
                out.append(self._commit(text, self._cursor + end, "punct_stable", 0.8))
                continue
            self.last_hold = self._hold_reason(after, now)
            break
        return out

    def _hold_reason(self, after: str, now: float) -> str:
        """Mô tả ngắn vì sao CHƯA chốt (dùng cho log trace từng nhịp)."""
        tail_n = content_len(after)
        elapsed_ms = (now - self._tail_since) * 1000.0 if self._tail_scans else 0.0
        if tail_n == 0:
            if not self.allow_stable_cut:
                return (
                    f"dấu ở cuối preview, CHƯA có câu mới (đang tắt cắt-theo-độ-ổn-định; "
                    f"chờ câu mới hoặc im lặng VAD)"
                )
            return (
                f"dấu ở cuối preview, chưa có tail (nhịp {self._pending_scans}/{self.stable_scans}, "
                f"{(now - self._pending_since) * 1000.0:.0f}/{self.stable_ms:.0f}ms)"
            )
        if tail_n < self.tail_min_chars:
            return (
                f"tail '{after.strip()}' = {tail_n}/{self.tail_min_chars} ký tự nội dung"
            )
        if not self._tail_ready(after):
            return f"tail '{after.strip()}' dính giữa từ (Latin chưa có khoảng trắng)"
        return (
            f"tail '{after.strip()}' đủ chữ nhưng mới {self._tail_scans}/{self.tail_scans} nhịp "
            f"({elapsed_ms:.0f}/{self.tail_stable_ms:.0f}ms)"
        )

    def trace_state(self) -> dict:
        """Trạng thái hiện tại của máy trạng thái (cho log trace/ghi metric)."""
        return {
            "cursor": self._cursor,
            "sentence_start": self._sentence_start,
            "committed_len": len(self._emitted_text),
            "boundary_index": self.last_boundary_index,
            "tail": self.last_tail,
            "tail_scans": self._tail_scans,
            "pending_scans": self._pending_scans,
            "hold": self.last_hold,
            "stale_text": self.stale_text,
            "resync_count": self.resync_count,
        }

    def flush(self, preview_text: Optional[str] = None) -> List[SegmentDecision]:
        """Chốt phần còn lại (gọi khi VAD đóng vùng nói hoặc hết stream)."""
        text = self._emitted_text if preview_text is None else (preview_text or "")
        self._resync(text)
        if not text[self._sentence_start:].strip():
            return []
        return [self._commit(text, len(text), "flush", 0.7)]

    # ------------------------------------------------------------------ helpers
    def _mark_pending(self, idx: int, now: float) -> None:
        if idx != self._pending_punct:
            self._pending_punct = idx
            self._pending_since = now
            self._pending_scans = 1
        else:
            self._pending_scans += 1

    def _too_few_words(self, piece: str) -> bool:
        """Câu đã đủ từ để gửi đi dịch chưa? (`min_words<=0` = không kiểm tra)"""
        if self.min_words <= 0:
            return False
        return count_tokens(piece) < self.min_words

    def is_cjk_char(self, ch: str) -> bool:
        cp = ord(ch)
        return (0x3040 <= cp <= 0x30FF) or (0x4E00 <= cp <= 0x9FFF) or (0xAC00 <= cp <= 0xD7AF)

    def _tail_ready(self, after: str) -> bool:
        """Tail đã đủ chữ để coi là "câu mới đã bắt đầu" chưa?

        Thêm một chốt cho chữ Latin: nếu tail mới chỉ là một mảnh chữ cái dính liền
        (chưa có khoảng trắng) thì ASR đang viết dở một từ ⇒ chưa đủ bằng chứng.
        """
        if content_len(after) < self.tail_min_chars:
            return False
        stripped = after.strip()
        if not stripped:
            return False
        if not any(self.is_cjk_char(ch) for ch in stripped):
            if " " not in stripped and stripped[0].isalpha():
                return False
        return True

    def _mark_tail(self, idx: int, now: float, has_tail: bool) -> None:
        """Đếm số NHỊP LIÊN TIẾP đã thấy tail của cùng một ranh giới.

        Chỉ đếm nhịp CÓ tail: nhịp mà dấu kết câu còn nằm ở cuối preview không được tính,
        nếu không "chờ 2 nhịp" sẽ thành "chờ 1 nhịp" (bug đã gặp khi viết test).
        """
        if not has_tail or idx != self._tail_punct:
            self._tail_punct = idx if has_tail else None
            self._tail_since = now
            self._tail_scans = 1 if has_tail else 0
            return
        self._tail_scans += 1

    def _is_tail_stable(self, now: float) -> bool:
        """Tail đã xuất hiện đủ số nhịp và đủ thời gian chưa?"""
        return (
            self._tail_scans >= self.tail_scans
            and (now - self._tail_since) * 1000.0 >= self.tail_stable_ms
        )

    def _is_stable(self, idx: int, now: float) -> bool:
        if idx != self._pending_punct:
            return False
        return (
            self._pending_scans >= self.stable_scans
            and (now - self._pending_since) * 1000.0 >= self.stable_ms
        )

    def _commit(self, text: str, end_index: int, reason: str,
                confidence: float) -> SegmentDecision:
        """Chốt câu `text[_sentence_start:end_index]`."""
        piece = text[self._sentence_start:end_index].strip()
        self._last_span = (self._sentence_start, end_index)
        self._sentence_start = end_index
        self._cursor = end_index
        self._emitted_text = text[:end_index]
        self._pending_punct = None
        self._pending_scans = 0
        return SegmentDecision(text=piece, end_index=end_index, reason=reason,
                               confidence=confidence, stale_text=self.stale_text)
