"""Romaji hoá tên riêng tiếng Nhật — hàm XÁC ĐỊNH của văn bản NGUỒN.

**Vì sao KHÔNG lấy cách viết tên từ bản dịch.** Ý tưởng "tự học glossary từ cặp
``orig -> trans``" có một lỗ hổng không vá được: cặp đó do **chính model đang cần ràng buộc**
sinh ra. Lấy nó làm chuẩn là (a) khoá cứng lựa chọn của model — `美咲` bị bịa thành `Meisaka`
một lần sẽ thành lỗi hệ thống ở mọi khối sau, (b) không xác định được token đích nào ứng với
token nguồn nào, (c) sai số tương quan nên hỏi lại chính model đó không mang thêm thông tin.
Chi tiết + dữ liệu: ``report/13_pronoun_and_glossary`` §"Câu hỏi về tự học glossary".

**Cách đúng:** cách viết một tên Nhật là hàm xác định của **chuỗi nguồn**, không phải của bản
dịch. ``美咲`` đọc là *misaki* — điều đó đúng bất kể model dịch nó thành gì. Module này biến
điều đó thành hiện thực:

* **Ưu điểm lớn nhất: KHÔNG cần lưu trạng thái.** Vì là hàm thuần, cùng một tên luôn cho cùng
  một cách viết ở mọi khối, mọi phiên, mọi lần chạy — nhất quán mà không cần "học", không cần
  ghi file, không có gì để hỏng.
* **Không bao giờ ghi đè glossary thủ công.** `glossary.py` chỉ dùng kết quả ở đây cho những
  tên CHƯA có mục khai tay.

Hai chế độ, tự chọn theo thư viện có mặt:

* ``pykakasi`` (khai trong ``backend/requirements.txt``) — xử lý cả **kanji**: ``美咲 -> Misaki``,
  ``佐藤 -> Sato``. Dùng kiểu ``passport`` chứ không phải ``hepburn`` vì passport được thiết kế
  cho TÊN NGƯỜI và rút gọn nguyên âm dài đúng chỗ (``佐藤``: passport ``Sato`` / hepburn ``Satou``).
* Fallback **thuần Python** khi thiếu ``pykakasi`` — vẫn romaji hoá được **katakana/hiragana**
  (``ミサキ -> Misaki``), chỉ chịu thua kanji.

.. warning::

   Đây là **cách đọc**, không phải bản dịch. Nanori (đọc tên riêng) có thể mơ hồ — ``一`` có thể
   là *Ichi / Kazu / Hajime*; ``pykakasi`` chọn một cách đọc phổ biến. Một cách đọc **sai nhưng
   ổn định** vẫn tốt hơn hẳn một cách viết **bịa ra**, và người dùng luôn ghi đè được bằng
   ``backend/glossary.yaml``. Vài tên ngoại lai bị phiên âm theo kana thay vì theo chính tả gốc
   (``メアリー`` → ``Mearii`` chứ không phải ``Mary``) — cũng ghi đè bằng glossary.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

from backend.utils.logger import logger

#: Nạp pykakasi KIỂU LƯỜI. Đo thực: `import pykakasi` 98 ms + `pykakasi.kakasi()` 156 ms
#: ⇒ nếu làm ở cấp module thì mỗi lần `import backend.translation` (mỗi tiến trình backend,
#: mỗi tiến trình pytest) đều cõng thêm ~254 ms. Chỉ trả giá khi thật sự có tên cần romaji.
_KAKASI: Optional[Any] = None
_KAKASI_TRIED = False
_KAKASI_LOCK = threading.Lock()
_KAKASI_ERROR: Optional[str] = None


def _get_kakasi() -> Optional[Any]:
    """Khởi tạo `kakasi()` đúng MỘT lần, an toàn đa luồng. Thiếu gói ⇒ `None` (không ném)."""
    global _KAKASI, _KAKASI_TRIED, _KAKASI_ERROR
    if _KAKASI_TRIED:
        return _KAKASI
    with _KAKASI_LOCK:
        if _KAKASI_TRIED:
            return _KAKASI
        try:  # pragma: no cover - phụ thuộc bản cài đặt
            import pykakasi  # noqa: PLC0415

            _KAKASI = pykakasi.kakasi()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            _KAKASI = None
            _KAKASI_ERROR = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"pykakasi không nạp được ({_KAKASI_ERROR}) — romaji tên kanji sẽ bị tắt, "
                f"katakana/hiragana vẫn chạy bằng bảng kana thuần Python.",
                extra={"module_tag": "TRANSLATE"},
            )
        _KAKASI_TRIED = True
    return _KAKASI


def pykakasi_available() -> bool:
    """True nếu `pykakasi` dùng được (⇒ romaji hoá được cả kanji)."""
    return _get_kakasi() is not None


def import_error() -> Optional[str]:
    """Thông báo lỗi khi không nạp được `pykakasi` (None nếu nạp được hoặc chưa thử)."""
    return _KAKASI_ERROR

_KANJI_RE = re.compile(r"[\u4e00-\u9fff]")
_KANA_RE = re.compile(r"[\u3041-\u3096\u30a1-\u30f6\u30fc]")
_LATIN_OK_RE = re.compile(r"^[A-Za-z][A-Za-z'\- ]*$")

#: Âm ghép (digraph) — phải thử TRƯỚC âm đơn, nếu không `きゃ` sẽ ra `kiya`.
_DIGRAPHS = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "てぃ": "ti", "でぃ": "di", "でゅ": "dyu",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "ゔ": "vu",
}

#: Âm đơn.
_MONOGRAPHS = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "i", "ゑ": "e", "を": "o", "ん": "n",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa",
}

_VOWELS = frozenset("aiueo")


def _katakana_to_hiragana(text: str) -> str:
    """Katakana → hiragana bằng dịch mã điểm (khối katakana nằm ngay sau khối hiragana)."""
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _kana_to_romaji(text: str) -> str:
    """Romaji hoá chuỗi CHỈ gồm kana (thuần Python, không cần thư viện)."""
    src = _katakana_to_hiragana(text)
    out: list[str] = []
    i = 0
    while i < len(src):
        two = src[i : i + 2]
        if two in _DIGRAPHS:
            out.append(_DIGRAPHS[two])
            i += 2
            continue
        ch = src[i]
        i += 1
        if ch == "っ":  # sokuon: gấp đôi phụ âm đầu của âm kế tiếp
            nxt = ""
            two2 = src[i : i + 2]
            if two2 in _DIGRAPHS:
                nxt = _DIGRAPHS[two2]
            elif src[i : i + 1] in _MONOGRAPHS:
                nxt = _MONOGRAPHS[src[i]]
            if nxt and nxt[0] not in _VOWELS:
                out.append(nxt[0])
            continue
        if ch == "ー":  # trường âm: lặp nguyên âm cuối
            if out and out[-1] and out[-1][-1] in _VOWELS:
                out.append(out[-1][-1])
            continue
        out.append(_MONOGRAPHS.get(ch, ch))
    return "".join(out)


def _titlecase(text: str) -> str:
    """`misaki` -> `Misaki`; giữ dấu gạch nối (``mary-ann`` -> ``Mary-Ann``)."""
    return "-".join(part[:1].upper() + part[1:] for part in text.split("-") if part) or text


def romanize(name: str) -> Optional[str]:
    """Romaji hoá một tên riêng. Trả ``None`` nếu không áp dụng được (⇒ giữ nguyên ràng buộc
    "nhất quán" thay vì ấn định một cách viết sai).

    Từ chối khi: rỗng · không có ký tự CJK/kana · kết quả không phải chữ Latinh · quá ngắn.
    """
    clean = (name or "").strip()
    if not clean:
        return None
    if not (_KANJI_RE.search(clean) or _KANA_RE.search(clean)):
        return None  # đã là chữ Latinh (hoặc không phải tiếng Nhật) — không cần suy

    romaji = _convert(clean)
    if not romaji:
        return None
    romaji = romaji.strip()
    if len(romaji) < 2 or not _LATIN_OK_RE.match(romaji):
        return None
    return _titlecase(romaji)


def _convert(text: str) -> Optional[str]:
    """Ưu tiên `pykakasi`; nếu thiếu thì fallback kana (chịu thua kanji)."""
    kakasi = _get_kakasi()
    if kakasi is not None:
        try:
            parts = kakasi.convert(text)
            # `passport` là kiểu dành cho TÊN NGƯỜI; thiếu khoá thì lùi về hepburn.
            out = "".join((p.get("passport") or p.get("hepburn") or "") for p in parts)
            if out:
                return out
        except Exception:  # noqa: BLE001
            pass
    if _KANJI_RE.search(text):
        return None  # không có pykakasi thì kanji là bó tay
    return _kana_to_romaji(text)


__all__ = ["romanize", "pykakasi_available", "import_error"]
