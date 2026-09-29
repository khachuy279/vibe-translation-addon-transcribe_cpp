"""Đếm độ ủng hộ của các giả thuyết ASR giữa các nhịp preview (chẩn đoán, không quyết định).

Vì sao cần (log thật 2026-09-29 21:29, phim tiếng Nhật)
------------------------------------------------------
ASR streaming KHÔNG ổn định theo tiền tố: cùng một đoạn audio, khi đuôi audio dài thêm thì
model có thể viết lại cả phần ĐẦU của câu::

    t=+0.4s  '焼いて。'
    t=+0.7s  '寄り行ってくる。'   ← câu đúng (người dùng đối chiếu video)
    t=+0.9s  '泳いてくるね。'     ← trôi, rồi lặp lại 6 nhịp
    [ASR_COMMIT] [VAD_SILENCE]: '泳いてくるね。'

Trước đây không tầng nào phát hiện ra hiện tượng này vì commit chỉ phát văn bản mới nhất. Bộ đếm
này ghi lại "nhịp nào đọc ra câu gì" để `TranscribeEngine._warn_unseen_commit_text()` cảnh báo
`[DRIFT]` + tăng counter `asr.commit_text_unseen_in_preview`.

⚠️ Đây KHÔNG phải bộ bỏ phiếu chọn câu đúng — và cố ý không dùng để ĐỔI văn bản phát đi. Khi model
trôi sang một câu SAI rồi lặp lại câu sai đó nhiều nhịp (ca trên: 6 nhịp so với 1 nhịp của câu
đúng) thì không bộ đếm nào cứu được: đó là hạn chế của model ASR. Nguồn đáng tin nhất vẫn là bản
CHẠY LẠI trên mảnh audio đã đóng (`preview_reuse_for_commit=False`).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

#: Ký tự KHÔNG tính là nội dung khi so hai giả thuyết (dấu câu + khoảng trắng + ngoặc).
_IGNORED = set("。！？!?…、，,;；:：「」『』（）()【】《》〈〉\"'”’ \t\n\u3000")

#: Dấu kết câu dùng để tách một nhịp preview thành các CÂU.
_TERMINAL = "。！？!?…"


def normalize_hypothesis(text: str) -> str:
    """Bỏ dấu câu/khoảng trắng để so hai giả thuyết theo KÝ TỰ."""
    return "".join(ch for ch in (text or "") if ch not in _IGNORED)


def _is_terminal_at(text: str, i: int) -> bool:
    """`text[i]` có phải dấu kết câu không (bỏ qua dấu `.` của số thập phân/viết tắt)."""
    ch = text[i]
    if ch in _TERMINAL:
        return True
    if ch != ".":
        return False
    prev = text[i - 1] if i > 0 else ""
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if prev.isdigit() and nxt.isdigit():
        return False        # 3.5
    return not nxt or nxt.isspace() or nxt in _TERMINAL


def _content_len(text: str) -> int:
    return sum(1 for ch in text if ch not in _IGNORED)


def _split_sentences(text: str, *, min_chars: int = 2) -> List[str]:
    """Tách văn bản preview (có thể đang viết dở) thành các câu để đếm.

    Giữ luôn phần ĐUÔI chưa có dấu kết câu (nó là một giả thuyết đang hình thành) — nhờ vậy
    "câu đứng im" giữa hai nhịp vẫn được tính.
    """
    out: List[str] = []
    buf = ""
    for i, ch in enumerate(text or ""):
        buf += ch
        if _is_terminal_at(text, i) and _content_len(buf) >= min_chars:
            out.append(buf.strip())
            buf = ""
    rest = buf.strip()
    if rest and (_content_len(rest) >= min_chars or not out):
        out.append(rest)
    return [s for s in out if s]


class HypothesisConsensus:
    """Bộ đếm "bao nhiêu nhịp preview đã đọc ra câu này" cho CÂU ĐANG NÓI.

    Vòng đời gắn với MỘT vùng nói: engine `reset()` khi vùng audio tiến lên (chốt câu xong,
    tua video, bỏ mảnh lỗi) — xem `TranscribeEngine._emit_commit`.
    """

    def __init__(self, *, max_entries: int = 64, min_similarity: float = 0.8):
        #: Trần số câu khác nhau được nhớ (chỉ để bộ nhớ không phình nếu ASR trôi liên tục).
        self.max_entries = max(4, int(max_entries))
        #: Ngưỡng giống nhau để câu chỉ khác dấu câu vẫn tính là cùng giả thuyết.
        self.min_similarity = float(min_similarity)
        self.reset()

    def reset(self) -> None:
        """Quên toàn bộ lịch sử (bắt đầu vùng nói mới)."""
        self._counts: Dict[str, int] = {}
        self._raw: Dict[str, str] = {}
        #: Số nhịp preview đã nạp (chỉ để ghi log/metric).
        self.scans: int = 0

    def observe(self, preview_text: str) -> None:
        """Nạp MỘT nhịp preview: tách thành câu rồi +1 cho mỗi câu."""
        text = (preview_text or "").strip()
        if not text:
            return
        self.scans += 1
        for piece in _split_sentences(text):
            key = normalize_hypothesis(piece)
            if not key:
                continue
            self._counts[key] = self._counts.get(key, 0) + 1
            self._raw[key] = piece.strip()
        self._evict()

    def support(self, text: str, *, min_length_ratio: float = 0.75) -> int:
        """Số nhịp preview đã từng đọc ra câu này (0 = chưa từng thấy).

        Khớp CHÍNH XÁC trước; nếu không có thì lấy ứng viên giống nhất đạt `min_similarity`
        **và** có độ dài tương đương — chốt an toàn để một mảnh NGẮN đang viết dở (`荷物持つんで。`)
        không thổi phồng độ ủng hộ của cả câu dài.
        """
        key = normalize_hypothesis(text)
        if not key:
            return 0
        exact = self._counts.get(key)
        if exact is not None:
            return exact

        best = 0
        for other, count in self._counts.items():
            if count <= best:
                continue
            shorter, longer = sorted((len(key), len(other)))
            if longer <= 0 or (shorter / float(longer)) < float(min_length_ratio):
                continue
            if _similarity(key, other) >= self.min_similarity:
                best = count
        return best

    def tally(self, top: int = 3) -> List[Tuple[str, int]]:
        """Các giả thuyết được ủng hộ nhiều nhất, kèm số nhịp (để ghi log/metric)."""
        items = sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(self._raw.get(key, key), count) for key, count in items[: max(1, int(top))]]

    def _evict(self) -> None:
        """Giữ bộ đếm nhỏ: quá `max_entries` thì bỏ câu ít được ủng hộ nhất (ổn định)."""
        while len(self._counts) > self.max_entries:
            key = min(self._counts, key=lambda k: (self._counts[k], k))
            self._counts.pop(key, None)
            self._raw.pop(key, None)


def _similarity(a: str, b: str) -> float:
    """Tỉ lệ giống nhau của hai chuỗi đã chuẩn hoá (dùng `difflib` cho chuỗi ngắn)."""
    if a == b:
        return 1.0
    import difflib  # noqa: PLC0415 — chỉ nạp khi thật cần (hot path dùng khớp chính xác)

    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
