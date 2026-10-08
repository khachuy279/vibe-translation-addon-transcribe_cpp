"""Ép ràng buộc đại từ tiếng Việt ở tầng VĂN BẢN — không phụ thuộc việc model có tuân thủ hay không.

**Vì sao cần (sự cố thật 2026-10-07).** Prompt đã ghi rõ ``never use the word mình anywhere``
(xem ``prompts._PRONOUN_GUIDANCE_VI``), nhưng Index-Translate-9B vẫn trả:

    "tôi còn nói là **mình** chắc chắn không thể quay lại làm nhân viên văn phòng nữa đâu."

Ràng buộc đặt ở prompt là **hy vọng**; ràng buộc ở tầng văn bản là **bảo đảm**. Thí nghiệm A/B
(``report/12_qwen35_translation_ab``) cho thấy Qwen3.5 tuân thủ ràng buộc này tốt hơn hẳn
Index-Translate, nhưng lại thua về tên riêng/kính ngữ — nên giữ Index-Translate và ép ràng buộc
bằng hậu kiểm là lựa chọn rẻ hơn nhiều so với đổi model.

**Nguyên tắc an toàn — chỉ thay khi ``mình`` là ĐẠI TỪ NGÔI THỨ NHẤT.**
Rất nhiều cụm tiếng Việt hợp lệ chứa ``mình`` mà **không** phải đại từ; thay bừa sẽ tạo câu
sai ngữ pháp:

===================  ==========================  ===========================
Cụm nguồn            Thay bừa thành              Đúng phải là
===================  ==========================  ===========================
``một mình``         ``một tôi`` ❌               giữ nguyên ("alone")
``chính mình``       ``chính tôi`` ⚠️            giữ nguyên ("oneself")
``tự mình``          ``tự tôi`` ⚠️               giữ nguyên ("by oneself")
``bản thân mình``    ``bản thân tôi`` ⚠️         giữ nguyên
``nhà mình``         ``nhà tôi`` ⚠️              giữ nguyên ("our place")
``mình ơi``          ``tôi ơi`` ❌               giữ nguyên (hô ngữ)
``chúng mình``       —                           ``chúng ta`` ✅
``mình nghĩ là…``    —                           ``tôi nghĩ là…`` ✅
===================  ==========================  ===========================

Nên cách làm là **danh sách từ đứng trước bị CHẶN** (``_BLOCKED_BEFORE``) chứ không phải thay
tất. Ngoài ra ``chúng/bọn/tụi + mình`` được xử lý riêng thành ``chúng ta`` vì đó là "we" —
đúng theo hướng dẫn đại từ của dự án.
"""

from __future__ import annotations

import re

#: Từ đứng NGAY TRƯỚC "mình" mà khiến nó KHÔNG còn là đại từ ngôi thứ nhất.
#: Giữ danh sách này hẹp — mỗi lần thêm một từ là mỗi lần bỏ sót một ca vi phạm thật.
_BLOCKED_BEFORE = frozenset({
    "một",
    "chính",
    "tự",
    "bản",
    "thân",
    "riêng",
    "nhà",
    "mỗi",
    "duy",
    "cả",
})

#: "we" dạng thân mật ⇒ phải thành "chúng ta" (không phải "tôi").
_WE_FORMS = re.compile(r"\b(chúng|bọn|tụi)\s+mình\b", re.IGNORECASE)

#: Một từ "mình" độc lập (ranh giới từ để không dính "mình mẩy", "tìmình"…).
_MINH = re.compile(r"\bmình\b", re.IGNORECASE)

#: "mình ơi" là hô ngữ (gọi người yêu/bạn đời), không phải đại từ ngôi thứ nhất.
_VOCATIVE_AFTER = re.compile(r"^\s*ơi\b", re.IGNORECASE)

_PREV_WORD = re.compile(r"([^\W\d_]+)\s*$", re.UNICODE)


def _match_case(replacement: str, sample: str) -> str:
    """Giữ hoa/thường của từ gốc cho từ thay thế."""
    return replacement.capitalize() if sample[:1].isupper() else replacement


def contains_replaceable_minh(text: str) -> bool:
    """True nếu văn bản có ``mình`` mang nghĩa đại từ (tức là ``enforce_no_minh`` sẽ sửa gì đó)."""
    return bool(text) and enforce_no_minh(text) != text


def count_occurrences(text: str) -> int:
    """Đếm số lần ``mình`` xuất hiện (mọi dạng) — dùng cho log/metrics."""
    return len(_MINH.findall(text or ""))


def enforce_no_minh(text: str) -> str:
    """Thay mọi ``mình`` mang nghĩa đại từ ngôi thứ nhất bằng ``tôi`` (``chúng mình`` ⇒ ``chúng ta``).

    Trả về **nguyên văn** nếu không có gì để thay (không tạo object mới một cách vô ích, và
    để tầng gọi so sánh ``trước != sau`` một cách rẻ).
    """
    if not text or "mình" not in text.lower():
        return text

    # Bước 1: "chúng mình" / "bọn mình" / "tụi mình" ⇒ "chúng ta".
    out = _WE_FORMS.sub(lambda m: _match_case("chúng ta", m.group(0)), text)

    # Bước 2: các "mình" còn lại, bỏ qua ca bị chặn và ca hô ngữ.
    pieces: list[str] = []
    cursor = 0
    changed = False
    for match in _MINH.finditer(out):
        before = out[: match.start()]
        prev = _PREV_WORD.search(before)
        if prev and prev.group(1).lower() in _BLOCKED_BEFORE:
            continue
        if _VOCATIVE_AFTER.match(out[match.end():]):
            continue
        pieces.append(out[cursor : match.start()])
        pieces.append(_match_case("tôi", match.group(0)))
        cursor = match.end()
        changed = True

    if not changed:
        return out
    pieces.append(out[cursor:])
    return "".join(pieces)
