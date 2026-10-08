"""Glossary tên riêng / thuật ngữ cho tầng dịch — nguồn sự thật cho cách viết CỐ ĐỊNH.

**Vì sao cần (sự cố thật 2026-10-07).** Khối phụ đề tiếng Nhật có ``美咲さん`` bị dịch sai tên:

* Index-Translate-9B (đang chạy) → ``Misaki-san`` ✅
* Qwen3.5-4B → ``Meisaka`` / ``Misa`` / ``Ms. Misaki`` ❌
* Qwen3.5-9B → ``Miaki-san`` / ``Misaki-chan`` ❌

Thí nghiệm A/B (``report/12_qwen35_translation_ab``) kết luận: **không đổi model**, mà bơm cho
model một bảng "cách viết cố định" — đúng cơ chế ``TEMPLATE_TERMINOLOGY`` mà upstream
Index-Translate đã định nghĩa sẵn cho Pipeline A, nhưng **chưa từng được nối vào Pipeline B**.

Module này có 2 nguồn, hợp nhất theo thứ tự ưu tiên (sau ghi đè trước):

1. ``backend/glossary.yaml`` — bảng tĩnh, đọc lại tự động khi ``mtime`` đổi (không cần restart).
2. ``config.translation.glossary`` — bảng runtime (dict), dùng khi cần ghi đè theo phiên.

Ngoài ra module **phát hiện tên riêng** trong chính khối nguồn và suy **romaji** cho những tên
**chưa** có trong bảng (``derived_renderings`` → ``romaji.romanize``): model không được bịa ra
hai cách viết khác nhau cho cùng một tên (đúng kiểu lỗi ``Meisaka`` vs ``Misa`` ở trên), và
những tên suy được thì bị **ấn định** luôn cách viết.

.. note::

   **Vì sao suy từ NGUỒN chứ không "học" từ bản dịch.** Cặp ``orig -> trans`` do chính model
   sinh ra, nên lấy nó làm chuẩn là khoá cứng sai số của model (``美咲`` bị bịa một lần sẽ
   thành lỗi hệ thống ở mọi khối sau). Cách viết ``美咲 -> Misaki`` là hàm xác định của **chuỗi
   nguồn**, không phải của bản dịch. Chi tiết + dữ liệu: ``report/13_pronoun_and_glossary``.
   Mục khai tay trong ``glossary.yaml`` **luôn đè** kết quả suy tự động.

.. note::

   Bảng glossary khớp theo **chuỗi con nguyên văn trong văn bản nguồn**, nên nó độc lập với
   ngôn ngữ nguồn: ``美咲`` cho tiếng Nhật, ``Mîsaki`` cho tiếng Latinh, v.v. đều dùng cùng
   một cơ chế.
"""

from __future__ import annotations

from pathlib import Path
import re
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from backend.translation.romaji import romanize
from backend.utils.logger import logger

#: Trần an toàn: hint dài làm loãng prompt và tốn token của ngân sách dịch.
MAX_TERMS_IN_HINT = 20
MAX_NAMES_IN_HINT = 8

#: Ký tự CJK để cắt tên đứng ngay trước kính ngữ.
_KANJI_CLASS = r"\u4e00-\u9fff"
_KATAKANA_CLASS = r"\u30a1-\u30f6\u30fc"

#: Kính ngữ Nhật — dấu hiệu mạnh nhất cho "đây là tên người".
_HONORIFICS = ("さん", "ちゃん", "くん", "君", "様", "氏", "先輩", "先生")

#: Chức danh ghép sau tên (``佐藤部長``). Không có chúng thì tên + chức danh bị bỏ sót hoàn toàn.
_TITLES = ("部長", "課長", "社長", "専務", "常務", "係長", "主任", "会長", "教授")

#: Ứng viên là CHỨC DANH/danh từ chung chứ không phải tên riêng — chặn trước khi romaji hoá,
#: nếu không ``社長さん`` sẽ bị ấn định thành ``社長 -> Shachou`` (sai hoàn toàn).
_NOT_A_NAME = frozenset({
    "社長", "部長", "課長", "専務", "常務", "係長", "主任", "会長", "教授", "先生", "先輩",
    "社員", "店長", "店員", "駅員", "医者", "看護師", "教師", "客", "お客", "皆", "全員",
})

#: Số đếm — ``五年先輩`` cho run ``五年`` ("5 năm"), ``三人``, ``五回``… KHÔNG phải tên.
#: Cố ý chỉ khớp **số + trợ từ đếm** (hoặc số trần), KHÔNG khớp mọi thứ chứa chữ số: ``一郎``
#: là tên thật (``郎`` không phải trợ từ đếm) nên vẫn được giữ.
_NUMERAL_CLASS = "0-9０-９一二三四五六七八九十百千万億何数幾"
_COUNTERS = (
    "年", "月", "日", "人", "回", "番", "個", "本", "枚", "台", "歳", "才", "階", "時", "分",
    "秒", "週", "円", "度", "名", "位", "匹", "冊", "杯", "件", "組", "割", "倍", "点", "文字",
    "人前", "ヶ月", "か月", "カ月", "つ",
)
_NUMERIC_RE = re.compile(r"^[" + _NUMERAL_CLASS + r"]+(?:" + "|".join(_COUNTERS) + r")?$")

#: Chạy katakana ngay trước kính ngữ (``ミサキちゃん``). KHÔNG dùng cho katakana trần — xem
#: cảnh báo trong `detect_names`.
_KATAKANA_BEFORE_RE = re.compile(r"([" + _KATAKANA_CLASS + r"]{1,10})$")
_KANJI_BEFORE_RE = re.compile(r"([" + _KANJI_CLASS + r"]{1,4})$")


class TranslationGlossary:
    """Singleton đọc bảng thuật ngữ + phát hiện tên riêng cho prompt dịch."""

    _instance: Optional["TranslationGlossary"] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        #: cache theo (đường dẫn, mtime, size) — sửa file là có hiệu lực ngay lần gọi sau.
        self._file_cache: Tuple[Optional[str], float, int, Dict[str, str]] = (None, 0.0, 0, {})
        self._cache_lock = threading.RLock()
        #: Tên đã romaji-hoá tự động và đã ghi log — chỉ log MỘT lần mỗi tên để không spam.
        self._logged_derived: set[str] = set()

    @classmethod
    def get_instance(cls) -> "TranslationGlossary":
        with cls._lock:
            if cls._instance is None:
                cls._instance = TranslationGlossary()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Xoá singleton (dùng cho test)."""
        with cls._lock:
            cls._instance = None

    # ───────────────────────────────────────────────────────────── nạp bảng
    def load_terms(self, cfg: Any = None) -> Dict[str, str]:
        """Hợp nhất bảng từ file và bảng runtime trong ``cfg.glossary`` (runtime thắng)."""
        terms: Dict[str, str] = {}
        terms.update(self._load_file_terms(self._resolve_path(cfg)))
        runtime = getattr(cfg, "glossary", None) if cfg is not None else None
        if isinstance(runtime, dict):
            for src, tgt in runtime.items():
                key, val = str(src or "").strip(), str(tgt or "").strip()
                if key and val:
                    terms[key] = val
        return terms

    @staticmethod
    def _resolve_path(cfg: Any) -> Optional[Path]:
        raw = getattr(cfg, "glossary_file", None) if cfg is not None else None
        if raw is None:
            # Không có cfg (test/gọi lẻ) ⇒ dùng mặc định của ứng dụng.
            try:
                from backend.config import config as _app_config  # noqa: PLC0415

                raw = _app_config.translation.glossary_file
            except Exception:  # noqa: BLE001
                return None
        raw = str(raw or "").strip()
        return Path(raw) if raw else None

    def _load_file_terms(self, path: Optional[Path]) -> Dict[str, str]:
        if path is None:
            return {}
        try:
            stat = path.stat()
        except OSError:
            return {}  # file không tồn tại là trạng thái BÌNH THƯỜNG (glossary tuỳ chọn)

        with self._cache_lock:
            cached_path, cached_mtime, cached_size, cached_terms = self._file_cache
            if (
                cached_path == str(path)
                and cached_mtime == stat.st_mtime
                and cached_size == stat.st_size
            ):
                return dict(cached_terms)

        terms = self._parse_terms_file(path)
        with self._cache_lock:
            self._file_cache = (str(path), stat.st_mtime, stat.st_size, dict(terms))
        if terms:
            logger.info(
                f"Glossary: nạp {len(terms)} mục từ '{path.name}'",
                extra={"module_tag": "TRANSLATE"},
            )
        return terms

    @staticmethod
    def _parse_terms_file(path: Path) -> Dict[str, str]:
        """Đọc ``{terms: {...}}`` hoặc mapping phẳng. File hỏng ⇒ bỏ qua, KHÔNG làm chết dịch."""
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Glossary '{path}' không đọc được (bỏ qua): {exc}", extra={"module_tag": "TRANSLATE"})
            return {}
        if isinstance(data, dict) and isinstance(data.get("terms"), dict):
            data = data["terms"]
        if not isinstance(data, dict):
            logger.warning(f"Glossary '{path}' phải là mapping hoặc có khoá 'terms' — bỏ qua.", extra={"module_tag": "TRANSLATE"})
            return {}
        out: Dict[str, str] = {}
        for src, tgt in data.items():
            key, val = str(src or "").strip(), str(tgt or "").strip()
            if key and val:
                out[key] = val
        return out

    # ───────────────────────────────────────────────────── lọc theo văn bản
    def matched_terms(self, texts: Sequence[str], cfg: Any = None) -> Dict[str, str]:
        """Chỉ giữ các mục glossary **thực sự xuất hiện** trong khối văn bản nguồn."""
        joined = "\n".join(t for t in texts if t)
        if not joined:
            return {}
        terms = self.load_terms(cfg)
        matched = {src: tgt for src, tgt in terms.items() if src in joined}
        if len(matched) > MAX_TERMS_IN_HINT:
            logger.debug(
                f"Glossary: {len(matched)} mục khớp, chỉ gửi {MAX_TERMS_IN_HINT} mục vào prompt.",
                extra={"module_tag": "TRANSLATE"},
            )
            matched = dict(list(matched.items())[:MAX_TERMS_IN_HINT])
        return matched

    # ─────────────────────────────────────────────────── phát hiện tên riêng
    @staticmethod
    def detect_names(texts: Sequence[str], limit: int = MAX_NAMES_IN_HINT) -> List[str]:
        """Đoán tên riêng trong văn bản nguồn (hiện hỗ trợ tiếng Nhật/Trung).

        **Một tín hiệu duy nhất: KÍNH NGỮ/CHỨC DANH đứng ngay sau.** Ứng viên là chuỗi
        kanji *hoặc* katakana liền trước kính ngữ:

        * ``美咲さん``, ``田中くん``, ``佐藤部長`` → ``美咲``, ``田中``, ``佐藤``
        * ``ミサキちゃん``, ``カミちゃん`` → ``ミサキ``, ``カミ``

        Cắt kanji đứng trước kính ngữ: lấy tối đa 3 kanji liền trước; nếu nhiều hơn (tức là
        dính cả từ phía trước, ví dụ ``今日美咲さん`` cho run ``今日美咲``) thì lấy **2 kanji
        cuối** — độ dài họ tên tiếng Nhật phổ biến nhất. Katakana **giữ nguyên cả run** vì tên
        phiên âm dài ngắn tuỳ tên (``ミサキ`` 3, ``アメリア`` 4) và ranh giới với từ trước rõ ràng.

        .. warning::

           **KHÔNG nhận katakana trần (không có kính ngữ).** Bản đầu tiên làm vậy và hỏng thật
           (log 2026-10-08): katakana trong tiếng Nhật **chủ yếu là từ ngoại lai**, nên
           ``チーム -> Chiimu``, ``ホテル -> Hoteru``, ``センス -> Sensu``, ``サッカー -> Sakkaa``,
           ``バーベキュー -> Baabekyuu``, ``グリル -> Guriru`` đều bị ấn định thành romaji và
           **thay mất bản dịch đúng** ("đội", "khách sạn", "gu", "bóng đá", "tiệc nướng", "vỉ nướng").
           Dương tính giả ở đây **phá hoại** (đè lên bản dịch đúng), còn âm tính chỉ quay về hành
           vi cũ — nên thà bỏ sót. Tên ngoại lai không có kính ngữ vẫn khai được trong
           ``glossary.yaml``.

        Chỉ nhận ứng viên **≥ 2 ký tự** và **không phải số/trợ từ đếm** (``五年先輩`` → ``五年``
        là "5 năm", không phải tên). Hạ trần xuống 1 ký tự sẽ bắt thêm vài họ một kanji hiếm
        (林, 森) nhưng đổi lại là nhiều nhiễu từ danh từ chung + kính ngữ (``お客さん`` → ``客``).

        KHÔNG đoán tên Latinh: không có POS tagger thì ``\\b[A-Z][a-z]+\\b`` bắt gần như mọi từ
        đầu câu, nhiễu hơn lợi. Tên Latinh vẫn khai báo được qua ``glossary.yaml``.
        """
        names: List[str] = []
        for text in texts:
            if not text:
                continue
            for honorific in _HONORIFICS + _TITLES:
                start = 0
                while True:
                    idx = text.find(honorific, start)
                    if idx < 0:
                        break
                    start = idx + len(honorific)
                    prefix = text[:idx]

                    run = _KANJI_BEFORE_RE.search(prefix)
                    if run:
                        candidate = run.group(1)
                        names.append(candidate if len(candidate) <= 3 else candidate[-2:])
                        continue

                    run = _KATAKANA_BEFORE_RE.search(prefix)
                    if run:
                        names.append(run.group(1))

        # Giữ thứ tự xuất hiện, bỏ trùng, loại chức danh/số-đếm/danh từ chung, cắt theo trần.
        unique: List[str] = []
        for name in names:
            if len(name) < 2 or name in unique:
                continue
            if name in _NOT_A_NAME or _NUMERIC_RE.match(name):
                continue
            unique.append(name)
        return unique[:limit]

    # ───────────────────────────────────────────── romaji hoá tên (P4.2b)
    def _log_derived_once(self, derived: Dict[str, str]) -> None:
        """Ghi log lần ĐẦU tiên một tên được suy romaji — để người dùng audit được.

        Log một lần cho mỗi tên (không phải mỗi khối) vì hàm này chạy trên mọi khối phụ đề.
        """
        fresh = [f"{src} -> {tgt}" for src, tgt in derived.items() if src not in self._logged_derived]
        if not fresh:
            return
        self._logged_derived.update(derived.keys())
        logger.info(
            "Glossary: suy romaji cho tên chưa khai báo — "
            + "; ".join(fresh)
            + " (ghi đè được trong glossary.yaml)",
            extra={"module_tag": "TRANSLATE"},
        )

    def derived_renderings(self, texts: Sequence[str], cfg: Any = None) -> Dict[str, str]:
        """Suy cách viết romaji cho các tên **chưa** khai trong glossary.

        Đây là điểm khác biệt cốt lõi so với "tự học từ bản dịch": cách viết được suy từ
        **chuỗi nguồn** (``romaji.romanize``) nên không phụ thuộc model, không cần lưu trạng
        thái, và cùng một tên luôn cho cùng kết quả ở mọi khối. Xem ``backend/translation/romaji.py``.

        Trả về ``{}`` khi bị tắt bằng ``cfg.glossary_derive_names = False``.
        """
        if cfg is not None and not getattr(cfg, "glossary_derive_names", True):
            return {}
        manual = self.matched_terms(texts, cfg)
        out: Dict[str, str] = {}
        for name in self.detect_names(texts):
            if any(name in src or src in name for src in manual):
                continue  # đã có mục khai tay ⇒ tay luôn thắng
            romaji = romanize(name)
            if romaji:
                out[name] = romaji
        return out

    # ───────────────────────────────────────────────────────── dựng prompt
    def build_hint(self, texts: Sequence[str], cfg: Any = None) -> str:
        """Khối ràng buộc thuật ngữ/tên riêng để chèn vào prompt dịch.

        Trả về ``""`` khi không có gì để ràng buộc — tầng gọi phải chịu được chuỗi rỗng
        (prompt giữ nguyên như trước, để không làm hồi quy khi chưa khai báo glossary).
        """
        lines: List[str] = []

        matched = self.matched_terms(texts, cfg)
        derived = self.derived_renderings(texts, cfg)
        self._log_derived_once(derived)

        fixed = {**derived, **matched}  # tay khai ĐÈ lên suy tự động
        if fixed:
            pairs = "; ".join(f"{src} -> {tgt}" for src, tgt in fixed.items())
            lines.append(
                "Fixed renderings — use these EXACTLY and identically in every sentence: " + pairs + "."
            )

        # Ràng buộc "nhất quán" chỉ có nghĩa khi khối có ≥ 2 câu (một câu thì không có gì
        # để mâu thuẫn) và khi tên đó chưa được ấn định (khai tay HOẶC suy romaji).
        # So khớp hai chiều để `美咲` (phát hiện được) không lặp lại khi glossary khai `美咲さん`.
        if len([t for t in texts if t and t.strip()]) >= 2:
            names = [
                n for n in self.detect_names(texts)
                if not any(n in src or src in n for src in fixed)
            ]
            if names:
                lines.append(
                    "Proper nouns in the source (" + ", ".join(names) + "): render each one "
                    "consistently and identically wherever it appears; never invent a second "
                    "spelling for the same name."
                )

        return "\n".join(lines)

    def terms_for_single(self, text: str, cfg: Any = None) -> str:
        """Chuỗi ``nguồn -> đích`` cho Pipeline A (đổ vào ``TEMPLATE_TERMINOLOGY.term``).

        Gồm cả tên suy romaji tự động (P4.2b) — Pipeline A dịch TỪNG CÂU nên càng cần ấn định
        cách viết tên, nếu không mỗi câu lại bịa một kiểu.
        """
        matched = self.matched_terms([text], cfg)
        derived = self.derived_renderings([text], cfg)
        self._log_derived_once(derived)
        fixed = {**derived, **matched}  # tay khai ĐÈ lên suy tự động
        return ", ".join(f"{src}->{tgt}" for src, tgt in fixed.items())


def get_glossary() -> TranslationGlossary:
    """Helper lấy singleton glossary."""
    return TranslationGlossary.get_instance()


def build_glossary_hint(texts: Iterable[str], cfg: Any = None) -> str:
    """Tiện ích cho engine: dựng hint từ danh sách câu."""
    return get_glossary().build_hint(list(texts), cfg)


__all__ = [
    "TranslationGlossary",
    "get_glossary",
    "build_glossary_hint",
    "MAX_TERMS_IN_HINT",
    "MAX_NAMES_IN_HINT",
]
