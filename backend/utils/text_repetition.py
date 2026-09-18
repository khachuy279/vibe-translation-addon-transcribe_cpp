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


_WS = re.compile(r"\s+")


def collapse_repetitions_preserve_spacing(text: str, **kwargs) -> str:
    """Như `collapse_repetitions` nhưng giữ nguyên khoảng trắng gốc nếu KHÔNG gộp gì.

    Dùng cho đường phụ đề: tránh việc hàm làm sạch vô tình chuẩn hoá khoảng trắng của câu
    bình thường (gây diff khó hiểu ở test đang chốt văn bản).
    """
    collapsed = collapse_repetitions(text, **kwargs)
    if collapsed == " ".join(text.split()):
        return text
    return collapsed
