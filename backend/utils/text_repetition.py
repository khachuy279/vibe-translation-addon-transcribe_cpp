"""Gộp cụm từ bị lặp do decoder "kẹt vòng" (ASR hoặc dịch).

**Vì sao cần:** gặp thật trong phiên video (log 2026-09-18 00:04). Một đoạn nhạc/tiếng cười
nền làm decoder Qwen3-ASR kẹt vòng và sinh **256 token** toàn `ha`:

    [ASR_COMMIT] [MAX_DURATION] (infer=1494.4ms): 'Ha ha ha ... (≈250 lần)'
    [TRANSLATE]  (en -> vi in 2395ms): 'Ha ha ha ... (≈150 lần)'

Hệ quả: 1,5 s GPU cho ASR + 2,4 s GPU cho dịch, phụ đề một dòng khổng lồ, và cảnh báo
"truncated" (chạm trần `n_ctx` của session). Binding transcribe.cpp **không** expose
`max_tokens`/repetition-penalty cho family `qwen3_asr` (chỉ có `n_threads`, `kv_type`,
`n_ctx`) nên không thể chặn ở tầng decoder — phải làm sạch ở tầng văn bản.

**Nguyên tắc:** chỉ gộp khi lặp ĐỦ NHIỀU để chắc chắn là lỗi decoder, và vẫn giữ lại vài lần
vì lặp ngắn là nội dung THẬT:
  * `ha ha ha` (tiếng cười thật) — giữ nguyên (dưới ngưỡng);
  * `no no no` / `very very good` — giữ nguyên;
  * `ha` ×250 — gộp còn `ha ha ha`;
  * `I don't know` ×6 — gộp còn `I don't know`.

So khớp KHÔNG phân biệt hoa/thường và dấu câu ở hai đầu token (`Ha!` ≡ `ha,` ≡ `ha`), nhưng
khi giữ lại thì giữ nguyên token gốc của lần xuất hiện đầu.

Văn bản không tách được thành ≥ `single_min_run` token (ví dụ tiếng Trung/Nhật không dấu
cách) được trả về NGUYÊN VẸN — hàm này không bao giờ cắt bừa.

**Hai hàm, hai mục đích khác nhau — đừng lẫn:**
  * `collapse_repetitions()` — LÀM SẠCH văn bản, vẫn GIỮ vài lần lặp (vì lặp ngắn là nội dung
    thật). Dùng trên đầu ra ASR/dịch.
  * `is_repetition_hallucination()` — QUYẾT ĐỊNH **BỎ CẢ CÂU**, nên phải chặt hơn hẳn: đoạn
    lặp còn phải chiếm ≥ `REPEAT_MIN_DOMINANCE` độ dài câu. Bản cũ chỉ cần "có lặp ở đâu đó"
    nên đã bỏ nhầm câu tiếng Nhật bình thường (log 2026-10-11).
"""

import re
import string

# Ngưỡng mặc định — chọn theo đúng ca thật đã gặp, không phải theo cảm tính.
SINGLE_MIN_RUN = 5      # lặp 1 token: từ 5 lần trở lên mới coi là kẹt vòng
SINGLE_MAX_KEEP = 3     # giữ tối đa 3 (tiếng cười "ha ha ha" vẫn là nội dung thật)
PHRASE_MAX_UNIT = 4     # cụm lặp dài tối đa 4 token
PHRASE_MIN_REPS = 3     # cụm phải lặp ≥ 3 lần
PHRASE_MAX_KEEP = 1     # cụm lặp verbatim thường là lỗi decoder ⇒ giữ 1 lần

_STRIP = string.punctuation + "…“”‘’«»–—"

#: Đoạn lặp phải chiếm ÍT NHẤT tỉ lệ này của câu mới bị coi là ảo giác kẹt vòng.
#:
#: **Vì sao cần (log thật 2026-10-11):** `min_reps=3` một mình bắt nhầm câu tiếng Nhật bình
#: thường. Câu ``かわきたならちゃんと独身でまともないいい男が選べる立場なんだから。`` có đoạn
#: ``いいい`` (3 ký tự ``い`` liền nhau = 3/33 ký tự) do ``まともな`` + ``いい男`` ghép lại,
#: hoàn toàn KHÔNG kẹt vòng — nhưng bộ lọc theo câu ở ``lookahead_handler`` đã **bỏ cả câu**,
#: làm mất một phụ đề thật. Tiếng Nhật/Trung không có dấu cách nên "có ký tự lặp" không đồng
#: nghĩa với "decoder kẹt vòng" như ở chữ Latinh.
#:
#: Ngưỡng 0.5 tách đúng hai nhóm: ``ああああ…`` (100 %) vẫn bị bỏ, còn lặp ngắn nằm trong câu
#: dài (``いいい``, ``No, no, no, I don't think so.``) được GIỮ.
REPEAT_MIN_DOMINANCE = 0.5


def _norm(token: str) -> str:
    """Chuẩn hoá để SO KHỚP: bỏ dấu câu hai đầu, bỏ hoa/thường."""
    return token.strip(_STRIP).lower()


def collapse_repetitions(
    text: str,
    *,
    single_min_run: int = SINGLE_MIN_RUN,
    single_max_keep: int = SINGLE_MAX_KEEP,
    phrase_max_unit: int = PHRASE_MAX_UNIT,
    phrase_min_reps: int = PHRASE_MIN_REPS,
    phrase_max_keep: int = PHRASE_MAX_KEEP,
) -> str:
    """Gộp các đoạn lặp liên tiếp. Trả về `text` nguyên vẹn nếu không có gì đáng gộp."""
    if not text:
        return text

    tokens = text.split()
    if len(tokens) < single_min_run:
        # Không thể có run 1-token đủ dài, và cụm lặp cũng cần ≥ 2*3 token.
        return text

    out = []
    i = 0
    n = len(tokens)
    while i < n:
        # ── 1) Một token lặp liên tiếp (ca thật: "ha ha ha ...")
        first_norm = _norm(tokens[i])
        j = i + 1
        while j < n and _norm(tokens[j]) == first_norm:
            j += 1
        if j - i >= single_min_run:
            out.extend(tokens[i:i + single_max_keep])
            i = j
            continue

        # ── 2) Một CỤM lặp liên tiếp ("I don't know I don't know I don't know")
        matched = False
        for unit_len in range(2, phrase_max_unit + 1):
            if i + unit_len * phrase_min_reps > n:
                break
            unit = [_norm(t) for t in tokens[i:i + unit_len]]
            if any(u == "" for u in unit):
                continue
            reps = 1
            k = i + unit_len
            while k + unit_len <= n and [_norm(t) for t in tokens[k:k + unit_len]] == unit:
                reps += 1
                k += unit_len
            if reps >= phrase_min_reps:
                out.extend(tokens[i:i + unit_len * phrase_max_keep])
                i = k
                matched = True
                break
        if matched:
            continue

        out.append(tokens[i])
        i += 1

    return " ".join(out)


def collapse_ratio(original: str, collapsed: str) -> float:
    """Tỉ lệ độ dài bị cắt (0.0 = không cắt gì, 0.9 = cắt 90 %) — để log/chẩn đoán."""
    if not original:
        return 0.0
    return max(0.0, 1.0 - (len(collapsed) / len(original)))


def is_repetition_hallucination(
    text: str,
    min_reps: int = 3,
    *,
    min_dominance: float = REPEAT_MIN_DOMINANCE,
) -> bool:
    """Kiểm tra xem văn bản có phải ảo giác kẹt vòng (từ/ký tự lặp liên tiếp >= min_reps lần).

    Quy tắc:
    1. Ký tự chữ cái/chữ Nhật-Trung đơn lặp liên tiếp (ví dụ 'あああ', 'aaaa' -> True).
    2. Cụm 2-4 ký tự lặp liên tiếp (ví dụ 'そうそうそう' -> True).
    3. Từ phân cách bằng khoảng trắng hoặc dấu ngắt lặp liên tiếp (ví dụ 'ha ha ha', 'no, no, no', 'はい、はい、はい' -> True).
    4. Không bắt nhầm dấu câu thông thường ('...') hay chữ số ('1000').

    **Điều kiện đủ (thêm 2026-10-11):** đoạn lặp phải chiếm >= `min_dominance` độ dài câu.
    Chỉ "có một đoạn lặp ở đâu đó" là CHƯA đủ — xem `REPEAT_MIN_DOMINANCE`.
    """
    if not text or not text.strip():
        return False
    s = text.strip()
    if len(s) < min_reps:
        return False

    span = _longest_repeat_span(s, min_reps)
    if span <= 0:
        return False
    return span / len(s) >= min_dominance


def _longest_repeat_span(s: str, min_reps: int) -> int:
    """Độ dài (số ký tự) của đoạn lặp liên tiếp DÀI NHẤT tìm được; 0 nếu không có.

    Gộp cả 3 quy tắc của `is_repetition_hallucination` vào một phép đo duy nhất, để hàm
    quyết định chỉ còn một ngưỡng "chiếm bao nhiêu phần câu".
    """
    best = 0

    # 1. Ký tự đơn lặp liên tiếp >= min_reps lần (bỏ qua số và dấu câu)
    for m in re.finditer(r"([^\s\W\d_])\1{" + str(min_reps - 1) + r",}", s):
        best = max(best, m.end() - m.start())

    # 2. Cụm 2-4 ký tự lặp liên tiếp >= min_reps lần (tiếng Nhật/Trung không dấu cách: ví dụ そうそうそう)
    for k in range(2, 5):
        if len(s) < k * min_reps:
            continue
        for m in re.finditer(r"([^\s\W\d_]{" + str(k) + r"})\1{" + str(min_reps - 1) + r",}", s):
            best = max(best, m.end() - m.start())

    # 3. Từ phân cách bằng khoảng trắng hoặc dấu ngắt lặp liên tiếp >= min_reps lần
    pattern_word = r"(?:\b|^)(\w+)(?:\s*[,、，\.!\?]?\s*\1){" + str(min_reps - 1) + r",}(?:\b|$)"
    for m in re.finditer(pattern_word, s, flags=re.IGNORECASE):
        best = max(best, m.end() - m.start())

    return best

