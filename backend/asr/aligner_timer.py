"""Timer SEG bằng **Qwen3-ForcedAligner-0.6B** (transformers native) — thay Whisper.

Vì sao
------
`backend/asr/timer.py::WhisperTimer` lấy mốc bằng cách chạy một model KHÁC (whisper) rồi
**căn chéo văn bản** Qwen3 ↔ Whisper bằng `difflib`, sau đó **nội suy tuyến tính ký tự**
trong từng segment (whisper chỉ trả mốc mức SEGMENT). Hai nguồn sai số:
  1. Whisper nhận sai/thiếu chữ so với Qwen3 ⇒ `difflib` neo lệch;
  2. Trong 1 segment, ký tự được rải ĐỀU theo độ dài văn bản, không theo thời gian thật.

`Qwen3-ForcedAligner-0.6B` căn **chính văn bản của Qwen3-ASR** và trả mốc mức TỪ/KÝ TỰ
(không phải nội suy) ⇒ bỏ cả hai nguồn sai số trên. Đo thật (RTX 5060 Ti, xem
`report/audit/26_KHA_THI_QWEN3_ASR_TRANSFORMERS.md`): 99 ms/lần align cho vùng 10 s,
so với 117 ms (p50) của WhisperTimer.

Ràng buộc quan trọng
--------------------
* Aligner chỉ hỗ trợ **11 ngôn ngữ** (zh, en, yue, fr, de, it, ja, ko, pt, ru, es) trong khi
  ASR hỗ trợ 30 (có `vi`, `th`, `id`…). Với ngôn ngữ ngoài danh sách, lớp này **trả None**
  để engine rơi về đường dự phòng (WhisperTimer / khoảng lặng VAD / chồng lấn) — KHÔNG được
  im lặng align sai ngôn ngữ.
* Phiên ở chế độ `auto`: suy ngôn ngữ theo **chữ viết** của văn bản (kana → Japanese,
  hangul → Korean, CJK → Chinese, Cyrillic → Russian). Chữ Latin **không phân biệt được**
  (en/fr/de… so với vi/id không được hỗ trợ) ⇒ trả None, trừ khi cấu hình
  `segmentation.aligner_language_fallback` chỉ định tên ngôn ngữ.
* Token hoá khác nhau theo ngôn ngữ (nguồn: `transformers/models/qwen3_asr/
  processing_qwen3_asr.py::split_words_for_alignment`): Japanese cần `nagisa`, Korean cần
  `soynlp`, còn lại tách từ theo khoảng trắng + tách từng ký tự CJK.

Module này **không** import `torch`/`transformers` ở cấp module: model chỉ được nạp khi
`ensure_loaded()` được gọi (từ luồng prewarm hoặc từ worker thread của ASR).
"""

from __future__ import annotations

import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from backend.asr.timer import SAMPLE_RATE, TimerResult
from backend.config import MODELS_DIR
from backend.utils.logger import logger

#: 11 ngôn ngữ aligner hỗ trợ. Nguồn: `transformers.models.qwen3_asr.processing_qwen3_asr`
#: (`FORCED_ALIGNER_LANGUAGES`, dòng 69) và model card `Qwen/Qwen3-ForcedAligner-0.6B`.
ALIGNER_LANGUAGE_BY_CODE: Dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}
ALIGNER_LANGUAGES = frozenset(ALIGNER_LANGUAGE_BY_CODE.values())
#: Tên đầy đủ (đã hạ chữ) → tên canonical.
_BY_NAME = {name.lower(): name for name in ALIGNER_LANGUAGES}

#: Dải ký tự dùng cho suy ngôn ngữ theo chữ viết khi phiên ở chế độ `auto`.
_KANA = ((0x3040, 0x30FF),)
_HANGUL = ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))
_CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))
_CYRILLIC = ((0x0400, 0x04FF),)


def _has_char_in(text: str, ranges: Sequence[tuple]) -> bool:
    for ch in text:
        code = ord(ch)
        for lo, hi in ranges:
            if lo <= code <= hi:
                return True
    return False


def keep_char_for_aligner(char: str) -> bool:
    """Đúng tập ký tự mà tokenizer của aligner GIỮ LẠI.

    Nguồn: `transformers.models.qwen3_asr.processing_qwen3_asr._is_kept_char` — giữ
    chữ cái (`L*`), chữ số (`N*`), ký tự CJK và dấu nháy đơn; BỎ mọi dấu câu khác.
    """
    if char == "'":
        return True
    category = unicodedata.category(char)
    if category[0] in ("L", "N"):
        return True
    return _has_char_in(char, _CJK)


def normalize_for_aligner(text: str) -> str:
    """Chuẩn hoá văn bản theo ĐÚNG tập ký tự của aligner (xem `keep_char_for_aligner`).

    ⚠️ Vì sao KHÔNG dùng `timer.normalize_for_align()`: hai tập luật KHÁC NHAU. `_IGNORED`
    của `segmentation.boundary` **không** chứa `.` `・` `-` `/` `%`… (aligner BỎ các ký tự
    này) nhưng LẠI bỏ `'` (aligner GIỮ). Lệch một ký tự ⇒ mốc cắt bị đẩy lùi đúng một đơn vị:
    đo thật trên FLEURS ja thấy lệch **+600…+700 ms** ở nhiều mẫu (xem
    `report/audit/27_SEG_AB_ALIGNER_VS_WHISPER.md` §"vì sao dùng chuẩn hoá riêng").
    """
    return "".join(ch for ch in (text or "") if keep_char_for_aligner(ch))


def language_for_aligner(
    session_language: Optional[str],
    text: str = "",
    fallback: str = "",
) -> Optional[str]:
    """Chuẩn hoá ngôn ngữ phiên → tên canonical trong 11 ngôn ngữ aligner, hoặc `None`.

    `None` nghĩa là "KHÔNG align" (engine sẽ rơi về đường dự phòng). Hàm này thuần, test
    được không cần model.

    Chấp nhận: `"ja"`, `"ja-JP"`, `"Japanese"`, `"japanese"`, `"JA"`; `auto`/`None`/rỗng ⇒
    suy theo chữ viết của `text`, cuối cùng mới dùng `fallback`.
    """
    raw = (session_language or "").strip()
    if raw:
        base = raw.replace("_", "-").split("-")[0].strip().lower()
        if base and base not in ("auto", "none"):
            if base in ALIGNER_LANGUAGE_BY_CODE:
                return ALIGNER_LANGUAGE_BY_CODE[base]
            full = raw.strip().lower()
            if full in _BY_NAME:
                return _BY_NAME[full]
            if base in _BY_NAME:
                return _BY_NAME[base]
            # Ngôn ngữ phiên CỤ THỂ nhưng aligner không hỗ trợ (vd "vi", "th") ⇒ không align.
            return None

    body = text or ""
    if _has_char_in(body, _KANA):
        return "Japanese"
    if _has_char_in(body, _HANGUL):
        return "Korean"
    if _has_char_in(body, _CJK):
        return "Chinese"
    if _has_char_in(body, _CYRILLIC):
        return "Russian"

    hint = (fallback or "").strip()
    if hint:
        base = hint.replace("_", "-").split("-")[0].strip().lower()
        if base in ALIGNER_LANGUAGE_BY_CODE:
            return ALIGNER_LANGUAGE_BY_CODE[base]
        if base in _BY_NAME:
            return _BY_NAME[base]
    return None


def locate_boundary_in_items(
    main_text: str,
    boundary_index: int,
    items: Sequence[Dict[str, Any]],
    *,
    audio_ms: float = 0.0,
) -> Optional[TimerResult]:
    """Tra ranh giới ký tự lên danh sách mốc từ/ký tự của aligner (hàm THUẦN).

    `boundary_index` = vị trí KÝ TỰ trong `main_text` ngay sau câu đã chốt (chỗ câu kế tiếp
    bắt đầu). Trả `TimerResult.cut_ms` = mốc BẮT ĐẦU của đơn vị aligner đầu tiên nằm sau
    ranh giới — cùng ngữ nghĩa với `timer.locate_cut_ms()`.

    So khớp dùng `normalize_for_aligner` cho CẢ HAI vế (văn bản và từng đơn vị của aligner)
    để hai vế cùng một tập ký tự — xem docstring `normalize_for_aligner` về lý do.

    Độ tin cậy = `coverage × (1.0 nếu ranh giới nằm GIỮA danh sách, 0.6 nếu ở mép)`, trong
    đó `coverage` = tỉ lệ ký tự nội dung của `main_text` mà aligner thật sự trả về đơn vị.
    """
    if not main_text or not items:
        return None

    norm_main = normalize_for_aligner(main_text)
    if not norm_main:
        return None

    target = len(normalize_for_aligner(main_text[: max(0, int(boundary_index))]))
    acc = 0
    matched = 0
    cut_ms: Optional[float] = None
    idx: Optional[int] = None
    for i, item in enumerate(items):
        token_len = len(normalize_for_aligner(str(item.get("text", "") or "")))
        if token_len <= 0:
            continue
        matched += token_len
        if cut_ms is None and acc >= target:
            cut_ms = float(item.get("start_time", 0.0) or 0.0) * 1000.0
            idx = i
        acc += token_len

    if cut_ms is None:
        # Ranh giới ở CUỐI văn bản ⇒ lấy mốc kết thúc của đơn vị cuối cùng.
        last = items[-1]
        cut_ms = float(last.get("end_time", 0.0) or 0.0) * 1000.0
        idx = len(items) - 1
        interior = False
    else:
        interior = bool(idx is not None and 0 < idx < len(items) - 1)

    coverage = min(1.0, matched / float(len(norm_main)))
    if audio_ms > 0.0 and (cut_ms < 0.0 or cut_ms > audio_ms + 50.0):
        return None
    return TimerResult(
        cut_ms=cut_ms,
        confidence=coverage * (1.0 if interior else 0.6),
        method="qwen3_aligner",
        segments=list(items),
        matched_chars=matched,
    )


class Qwen3AlignerTimer:
    """Timer SEG dùng `Qwen3-ForcedAligner-0.6B` qua transformers (API giống WhisperTimer).

    Vòng đời: nạp LAZY trong luồng nền (`TranscribeEngine.prewarm_seg_timer`) hoặc ở lần
    chốt câu đầu tiên. Model là singleton dùng chung (xem `timer.get_seg_timer`) nên
    KHÔNG được dựng nhiều instance song song (AGENTS.md §2.1).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        *,
        local_dir: str = "",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        language: str = "auto",
        language_fallback: str = "",
        auto_download: bool = True,
        whisper_fallback: bool = True,
        whisper_model_key: str = "whisper-large-v3-turbo",
    ):
        self.model_id = model_id
        #: `engine.prewarm_seg_timer()` log khoá này (giống `WhisperTimer.model_key`).
        self.model_key = model_id
        self.local_dir = (local_dir or "").strip()
        self.device = device or "cuda:0"
        self.dtype = dtype or "bfloat16"
        self.language = language or "auto"
        self.language_fallback = language_fallback or ""
        self.auto_download = bool(auto_download)
        #: Aligner chỉ phủ 11/30 ngôn ngữ của ASR (KHÔNG có tiếng Việt/Thái/Indonesia…).
        #: Bật cờ này ⇒ khi aligner không dùng được cho ngôn ngữ của phiên, timer **tự động
        #: chuyển sang `WhisperTimer`** (chất lượng mốc cắt y như hiện tại) thay vì để engine
        #: rơi thẳng về mốc chồng lấn. Model whisper chỉ được nạp khi thật sự cần.
        self.whisper_fallback = bool(whisper_fallback)
        self.whisper_model_key = whisper_model_key
        self._whisper: Any = None
        self._whisper_lock = threading.RLock()

        self.last_error: str = ""
        self.load_ms: float = 0.0
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.RLock()
        self._infer_lock = threading.RLock()
        self._load_error: Optional[str] = None
        self._available_cache: Optional[bool] = None
        self._available_checked_at: float = 0.0
        #: Kết quả "chưa có model" chỉ được cache trong bấy nhiêu giây (model có thể được
        #: tải xong SAU lần kiểm tra đầu; cache vĩnh viễn sẽ khiến timer không bao giờ dùng).
        self.available_negative_ttl_sec: float = 10.0

    # ------------------------------------------------------------------ vòng đời
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _local_dir_path(self) -> Path:
        """Thư mục model CỤC BỘ trong `backend/models/`.

        Rỗng ⇒ `backend/models/<org>__<name>` (cùng quy ước `backend/models/_hf/<org>__<name>`
        mà `backend/tools/build_finetune.py` dùng). **Model KHÔNG bao giờ được nạp thẳng từ
        cache HuggingFace** (`~/.cache/huggingface`): mọi thứ đi qua `backend/models/` giống
        ASR GGUF và whisper timer, để (a) chạy được hoàn toàn offline, (b) không phát sinh
        hàng loạt request HEAD lên HF mỗi lần nạp, (c) dễ sao lưu/di chuyển.
        """
        if self.local_dir:
            return Path(self.local_dir)
        return MODELS_DIR / self.model_id.replace("/", "__")

    def _local_dir_ready(self, path: Path) -> bool:
        """True nếu thư mục cục bộ có đủ file tối thiểu để nạp (không chạm mạng)."""
        required = ("config.json", "tokenizer_config.json", "processor_config.json")
        if not all((path / name).is_file() for name in required):
            return False
        weights = (path / "model.safetensors").is_file() or (
            path / "model.safetensors.index.json"
        ).is_file()
        return bool(weights)

    def ensure_local_model(self, *, progress: bool = True) -> Path:
        """Bảo đảm model có trong `backend/models/`; tải về đó nếu cần (một lần).

        Dùng `snapshot_download(local_dir=...)` — bản sao nằm HẲN trong repo (không symlink
        sang cache), nên các lần nạp sau chỉ đọc đĩa cục bộ.
        """
        path = self._local_dir_path()
        if self._local_dir_ready(path):
            return path
        if not self.auto_download:
            raise FileNotFoundError(
                f"Chưa có model aligner trong '{path}' và aligner_auto_download=False"
            )
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        logger.info(
            f"Đang tải aligner SEG '{self.model_id}' về {path} (lần đầu, ~1,7 GB)…",
            extra={"module_tag": "ASR"},
        )
        t0 = time.perf_counter()
        snapshot_download(
            repo_id=self.model_id,
            local_dir=str(path),
            allow_patterns=[
                "*.json", "*.jinja", "*.safetensors", "*.txt", "*.model",
            ],
        )
        elapsed = time.perf_counter() - t0
        if not self._local_dir_ready(path):
            raise FileNotFoundError(f"Tải xong nhưng thiếu file model trong '{path}'")
        logger.info(
            f"Đã tải aligner SEG về {path} ({elapsed:.1f}s, "
            f"{sum(f.stat().st_size for f in path.glob('*') if f.is_file()) / 2**20:.0f} MB)",
            extra={"module_tag": "ASR"},
        )
        return path

    def available(self, registry: Any = None) -> bool:
        """True nếu model dùng được (đã có trong `backend/models/` HOẶC cho phép tự tải).

        Kết quả DƯƠNG được cache vĩnh viễn (hàm này nằm trên hot path —
        `engine._seg_base_cut_sample` gọi trước mỗi lần định vị — và việc dò đĩa là I/O);
        kết quả ÂM chỉ cache trong `available_negative_ttl_sec` giây, vì model có thể được
        tải xong sau lần kiểm tra đầu tiên.
        """
        now = time.perf_counter()
        if self._available_cache is True:
            return True
        if (
            self._available_cache is False
            and (now - self._available_checked_at) < float(self.available_negative_ttl_sec)
        ):
            return False
        ok = self._local_dir_ready(self._local_dir_path()) or bool(self.auto_download)
        if not ok and self.whisper_fallback:
            # Aligner không dùng được NHƯNG còn whisper ⇒ timer vẫn có ích.
            ok = bool(self._fallback_timer().available())
        self._available_cache = bool(ok)
        self._available_checked_at = now
        return bool(ok)

    def _torch_dtype(self, torch_mod: Any) -> Any:
        key = (self.dtype or "").strip().lower()
        mapping = {
            "bfloat16": getattr(torch_mod, "bfloat16", None),
            "bf16": getattr(torch_mod, "bfloat16", None),
            "float16": torch_mod.float16,
            "fp16": torch_mod.float16,
            "float32": torch_mod.float32,
            "fp32": torch_mod.float32,
        }
        return mapping.get(key) or torch_mod.bfloat16

    def _pick_device(self, torch_mod: Any) -> str:
        want = (self.device or "").strip().lower()
        if want in ("", "auto"):
            return "cuda:0" if torch_mod.cuda.is_available() else "cpu"
        if want.startswith("cuda") and not torch_mod.cuda.is_available():
            logger.warning(
                "Aligner: yêu cầu CUDA nhưng không có GPU ⇒ chạy CPU (rất chậm).",
                extra={"module_tag": "ASR"},
            )
            return "cpu"
        return self.device

    def ensure_loaded(self) -> bool:
        """Nạp model + processor (lazy, có khoá). True nếu sẵn sàng."""
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            if self._load_error:
                return False
            t0 = time.perf_counter()
            try:
                import torch  # noqa: PLC0415
                from transformers import (  # noqa: PLC0415
                    AutoProcessor,
                    Qwen3ASRForTokenClassification,
                )

                # CHỈ nạp từ `backend/models/<org>__<name>` (tải về đó nếu chưa có) —
                # tuyệt đối không nạp thẳng từ cache HuggingFace (xem `_local_dir_path`).
                source = str(self.ensure_local_model())
                device = self._pick_device(torch)
                processor = AutoProcessor.from_pretrained(source, local_files_only=True)
                model = Qwen3ASRForTokenClassification.from_pretrained(
                    source,
                    dtype=self._torch_dtype(torch),
                    device_map=device,
                    local_files_only=True,
                ).eval()
                self._processor = processor
                self._model = model
                self.load_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    f"Aligner SEG sẵn sàng ({self.model_id}, device={device}, "
                    f"dtype={self.dtype}, nguồn={source}, {self.load_ms:.0f} ms)",
                    extra={"module_tag": "ASR"},
                )
                self._warmup(device)
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                self.last_error = self._load_error
                logger.warning(
                    f"Nạp aligner SEG thất bại: {self._load_error}",
                    extra={"module_tag": "ASR"},
                )
                return False
        return True

    def _fallback_timer(self) -> Any:
        """`WhisperTimer` dùng khi aligner không phủ ngôn ngữ phiên (tạo lười, nạp lười)."""
        if self._whisper is None:
            with self._whisper_lock:
                if self._whisper is None:
                    from backend.asr.timer import WhisperTimer  # noqa: PLC0415

                    self._whisper = WhisperTimer(
                        model_key=self.whisper_model_key, language=self.language
                    )
        return self._whisper

    def _explicit_language_supported(self) -> Optional[bool]:
        """`True`/`False` nếu ngôn ngữ phiên được đặt TƯỜNG MINH; `None` nếu là `auto`."""
        raw = (self.language or "").strip()
        base = raw.replace("_", "-").split("-")[0].strip().lower()
        if not raw or base in ("auto", "none"):
            return None
        return language_for_aligner(raw) is not None

    def prewarm(self) -> Any:
        """Nạp model cho lần chốt câu đầu tiên (engine gọi trong luồng nền).

        Nếu ngôn ngữ phiên được đặt tường minh và aligner KHÔNG phủ (vd tiếng Việt) thì nạp
        luôn whisper thay vì aligner: tiết kiệm 1,8 GB VRAM vô ích và giữ nguyên chất lượng
        mốc cắt hiện tại.
        """
        if self.whisper_fallback and self._explicit_language_supported() is False:
            return self._fallback_timer()._ensure_session()  # noqa: SLF001 — cùng package
        return self._ensure_session()

    def _ensure_session(self) -> Any:
        """Alias tương thích `WhisperTimer._ensure_session()` (engine prewarm dùng chung)."""
        return self._model if self.ensure_loaded() else None

    def _warmup(self, device: str) -> None:
        """Chạy vài lần align giả để trả chi phí "lần đầu" lúc PREWARM (luồng nền).

        Đo thật (`scratch/aligner_warmup_probe.py`):
          * lần align THẬT đầu tiên: ~1,57 s (có warmup) / ~2,04 s (không warmup), sau đó ~0,09 s;
          * thủ phạm chính là **từ điển của tokenizer theo ngôn ngữ**: lần gọi
            `split_words_for_alignment(..., "Japanese")` ĐẦU TIÊN tốn **~1,47 s** (nạp từ điển
            nagisa); tiếng Hàn tương tự với soynlp. Warmup bằng văn bản tiếng Anh KHÔNG hấp thụ
            được khoản này ⇒ phải warmup theo đúng ngôn ngữ sẽ dùng.
        """
        lengths = (1.0,) if str(device).startswith("cpu") else (1.0, 30.0)
        resolved = language_for_aligner(self.language, "", self.language_fallback)
        # Phiên `auto`: không biết trước ngôn ngữ ⇒ làm ấm cả 3 đường tokenizer
        # (en = mặc định, ja = nagisa, ko = soynlp) để câu chốt đầu tiên không bị đơ.
        langs = [resolved] if resolved else ["English", "Japanese", "Korean"]
        for lang in langs:
            dummy = {
                "Japanese": "これはテストです。天気がいいですね。",
                "Korean": "안녕하세요 좋은 아침입니다",
            }.get(lang, "ok")
            text = (dummy + " ") * 12 if lang != "Japanese" else dummy * 12
            total = 0.0
            for secs in lengths:
                try:
                    pcm = np.zeros(int(secs * SAMPLE_RATE), dtype=np.float32)
                    t0 = time.perf_counter()
                    self.align_items(pcm, text, lang)
                    total += (time.perf_counter() - t0) * 1000.0
                except Exception as exc:  # noqa: BLE001 — warmup là tối ưu, không được chặn nạp
                    logger.debug(
                        f"Warmup aligner bỏ qua ({lang}, {secs}s): {exc}",
                        extra={"module_tag": "ASR"},
                    )
            logger.info(
                f"Aligner SEG warmup {lang} xong ({total:.0f} ms cho {len(lengths)} độ dài, "
                f"device={device})",
                extra={"module_tag": "ASR"},
            )

    def close(self) -> None:
        with self._lock:
            model, self._model = self._model, None
            self._processor = None
            self._load_error = None
            self._available_cache = None
        if self._whisper is not None:
            try:
                self._whisper.close()
            except Exception:  # noqa: BLE001
                pass
            self._whisper = None
        if model is not None:
            try:
                del model
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ suy luận
    def align_items(self, pcm_audio: Any, text: str, language: str) -> List[Dict[str, Any]]:
        """Chạy aligner cho 1 mảnh audio + văn bản đầy đủ; trả danh sách mốc từ/ký tự.

        Trả `[]` khi lỗi (KHÔNG ném ra ngoài: đây là đường phụ trợ, không được làm chết
        luồng chốt câu).
        """
        if not self.ensure_loaded():
            return []
        pcm = np.ascontiguousarray(np.asarray(pcm_audio, dtype=np.float32).reshape(-1))
        if pcm.size <= 0 or not text.strip():
            return []
        try:
            import torch  # noqa: PLC0415

            with self._infer_lock:
                inputs, word_lists = self._processor.prepare_forced_aligner_inputs(
                    audio=pcm, transcript=text, language=language
                )
                inputs = inputs.to(self._model.device, dtype=self._model.dtype)
                with torch.inference_mode():
                    logits = self._model(**inputs).logits
                items = self._processor.decode_forced_alignment(
                    logits,
                    inputs["input_ids"],
                    word_lists,
                    timestamp_token_id=self._model.config.timestamp_token_id,
                )[0]
                del logits, inputs
            self.last_error = ""
            return [dict(it) for it in (items or [])]
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(f"Aligner SEG lỗi: {self.last_error}", extra={"module_tag": "ASR"})
            return []

    def locate_boundary(
        self,
        pcm_audio: Any,
        boundary_index: int,
        main_text: str,
        *,
        min_confidence: float = 0.0,
        language: Optional[str] = None,
    ) -> Optional[TimerResult]:
        """Định vị ranh giới câu của `main_text` trên `pcm_audio` (API giống WhisperTimer).

        Thứ tự: aligner (ngôn ngữ được phủ) → **whisper dự phòng** (ngôn ngữ ngoài 11 ngôn
        ngữ, hoặc aligner lỗi/không neo được) → `None` (engine tự dùng mốc chồng lấn).
        """
        t0 = time.perf_counter()
        if not main_text or not self.available():
            return None

        lang = language_for_aligner(
            language or self.language, main_text, self.language_fallback
        )
        if lang is None:
            # Ngôn ngữ phiên không nằm trong 11 ngôn ngữ aligner (vd tiếng Việt) ⇒ để engine
            # tự dự phòng, hoặc chuyển sang whisper nếu bật `whisper_fallback`.
            self.last_error = (
                f"aligner không hỗ trợ ngôn ngữ '{language or self.language}' "
                f"(hỗ trợ: {sorted(ALIGNER_LANGUAGES)})"
            )
            return self._fallback_result(
                pcm_audio, boundary_index, main_text,
                min_confidence=min_confidence, language=language,
            )

        pcm = np.asarray(pcm_audio, dtype=np.float32).reshape(-1)
        if pcm.size < int(0.1 * SAMPLE_RATE):
            return None
        audio_ms = pcm.size * 1000.0 / SAMPLE_RATE

        items = self.align_items(pcm, main_text, lang)
        res = (
            locate_boundary_in_items(main_text, boundary_index, items, audio_ms=audio_ms)
            if items
            else None
        )
        if res is not None:
            res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if res.confidence >= min_confidence:
                return res
        # Aligner không neo được (hoặc tin cậy dưới ngưỡng) ⇒ thử whisper trước khi bỏ cuộc.
        return self._fallback_result(
            pcm_audio, boundary_index, main_text,
            min_confidence=min_confidence, language=language,
        )

    def _fallback_result(
        self,
        pcm_audio: Any,
        boundary_index: int,
        main_text: str,
        *,
        min_confidence: float,
        language: Optional[str],
    ) -> Optional[TimerResult]:
        """Chạy `WhisperTimer` dự phòng (nếu bật); `None` ⇒ engine dùng mốc chồng lấn."""
        if not self.whisper_fallback:
            return None
        fallback = self._fallback_timer()
        if not fallback.available():
            return None
        return fallback.locate_boundary(
            pcm_audio,
            boundary_index,
            main_text,
            min_confidence=min_confidence,
            language=language,
        )
