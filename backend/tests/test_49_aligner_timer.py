"""Test tầng A — TIMER SEG bằng **Qwen3-ForcedAligner-0.6B** (`backend/asr/aligner_timer.py`).

Bối cảnh: xem `report/audit/26_KHA_THI_QWEN3_ASR_TRANSFORMERS.md` (khả thi + số đo) và
`report/audit/27_SEG_AB_ALIGNER_VS_WHISPER.md` (A/B mốc cắt).

Tầng A **KHÔNG nạp model thật** (aligner 1,8 GB VRAM): mọi test ở đây dùng hàm thuần hoặc
model giả. Test cần model thật phải đánh dấu `@pytest.mark.slow`.
"""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.asr import timer as timer_mod
from backend.asr.aligner_timer import (
    ALIGNER_LANGUAGES,
    Qwen3AlignerTimer,
    language_for_aligner,
    locate_boundary_in_items,
    normalize_for_aligner,
)
from backend.asr.timer import TimerResult, WhisperTimer, normalize_timer_engine

ROOT = Path(__file__).resolve().parent.parent.parent
ALIGNER_SRC = ROOT / "backend" / "asr" / "aligner_timer.py"

#: `ensure_loaded` THẬT, lấy NGAY LÚC IMPORT — tức TRƯỚC khi guard autouse
#: `_forbid_heavy_model_loads` của conftest thay nó bằng bản "nổ" (guard chặn nạp model thật
#: ở tầng A). Hai test dưới cần chạy ĐÚNG thân hàm thật nhưng đã monkeypatch toàn bộ
#: `transformers`/`huggingface_hub` nên không nạp model nào.
_REAL_ENSURE_LOADED = Qwen3AlignerTimer.ensure_loaded


# ─────────────────────────────────────────────────────── 0. an toàn khi import
def test_khong_import_torch_transformers_o_cap_module():
    """Module phải import được mà KHÔNG kéo torch/transformers (backend khởi động nhanh,
    và `import backend.asr` không được nạp thêm CUDA context)."""
    tree = ast.parse(ALIGNER_SRC.read_text(encoding="utf-8"))
    top_imports: list[str] = []
    for node in tree.body:  # chỉ cấp MODULE, bỏ qua import trong hàm
        if isinstance(node, ast.Import):
            top_imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.append(node.module)
    roots = {name.split(".")[0] for name in top_imports}
    assert "torch" not in roots, f"aligner_timer import torch ở cấp module: {top_imports}"
    assert "transformers" not in roots, f"aligner_timer import transformers ở cấp module: {top_imports}"


# ─────────────────────────────────────────────── 1. chuẩn hoá ngôn ngữ (hàm thuần)
def test_language_for_aligner_ma_va_ten():
    assert language_for_aligner("ja") == "Japanese"
    assert language_for_aligner("JA") == "Japanese"
    assert language_for_aligner("ja-JP") == "Japanese"
    assert language_for_aligner("Japanese") == "Japanese"
    assert language_for_aligner("japanese") == "Japanese"
    assert language_for_aligner("zh-CN") == "Chinese"
    assert language_for_aligner("yue") == "Cantonese"
    assert language_for_aligner("en") == "English"
    assert language_for_aligner("ru-RU") == "Russian"


def test_language_for_aligner_ngon_ngu_khong_ho_tro_thi_tra_None():
    """Aligner chỉ hỗ trợ 11 ngôn ngữ: `vi`/`th`/`id` PHẢI trả None để engine dự phòng
    (KHÔNG được im lặng align sai ngôn ngữ)."""
    for bad in ("vi", "th", "id", "ar", "hi", "fil", "tr", "vi-VN"):
        assert language_for_aligner(bad) is None, bad
    # `ko` thì CÓ hỗ trợ (đối chứng để test trên không vô nghĩa); BCP-47 nhiều subtag vẫn
    # lấy được mã cơ sở.
    assert language_for_aligner("ko") == "Korean"
    assert language_for_aligner("ko-KR") == "Korean"
    assert "Vietnamese" not in ALIGNER_LANGUAGES


def test_language_for_aligner_auto_suy_theo_chu_viet():
    assert language_for_aligner("auto", "今日はいい天気ですね。") == "Japanese"
    assert language_for_aligner(None, "カタカナのテスト") == "Japanese"
    assert language_for_aligner("auto", "안녕하세요 반갑습니다") == "Korean"
    assert language_for_aligner("auto", "甚至出现交易几乎停滞") == "Chinese"
    assert language_for_aligner("auto", "Да, конечно") == "Russian"


def test_language_for_aligner_auto_latin_thi_can_fallback():
    """Chữ Latin KHÔNG phân biệt được en/fr/de… với vi/id (không hỗ trợ) ⇒ mặc định None."""
    assert language_for_aligner("auto", "Okay, Charles. It looks like a problem.") is None
    assert language_for_aligner("", "Hello world") is None
    # Có chỉ định tường minh thì dùng.
    assert language_for_aligner("auto", "Hello world", "English") == "English"
    assert language_for_aligner("auto", "Hello world", "en") == "English"
    # Fallback cũng không được hỗ trợ ⇒ vẫn None.
    assert language_for_aligner("auto", "Xin chào các bạn", "vi") is None


# ─────────────────────────────────────── 2. tra ranh giới lên mốc của aligner (thuần)
_ITEMS = [
    {"text": "Okay", "start_time": 0.50, "end_time": 0.90},
    {"text": "Charles", "start_time": 0.90, "end_time": 1.40},
    {"text": "It", "start_time": 1.50, "end_time": 1.70},
    {"text": "looks", "start_time": 1.70, "end_time": 2.10},
]
_TEXT = "Okay, Charles. It looks"

# Mốc từ của aligner cho câu tiếng Nhật (nagisa: 今日/は/いい/天気/です/ね/明日/は/雨/です).
_JA_ITEMS = [
    {"text": "今日", "start_time": 0.20, "end_time": 0.50},
    {"text": "は", "start_time": 0.50, "end_time": 0.60},
    {"text": "いい", "start_time": 0.60, "end_time": 0.85},
    {"text": "天気", "start_time": 0.85, "end_time": 1.10},
    {"text": "です", "start_time": 1.10, "end_time": 1.35},
    {"text": "ね", "start_time": 1.35, "end_time": 1.45},
    {"text": "明日", "start_time": 1.45, "end_time": 1.70},
    {"text": "は", "start_time": 1.70, "end_time": 1.80},
    {"text": "雨", "start_time": 1.80, "end_time": 2.00},
    {"text": "です", "start_time": 2.00, "end_time": 2.25},
]
_JA_TEXT = "今日はいい天気ですね。明日は雨です"


def test_tra_ranh_gioi_vao_item_giua_danh_sach():
    """Tiếng Nhật: `。` nằm trong `_IGNORED` nên ranh giới khớp CHÍNH XÁC mốc của từ đầu
    câu kế tiếp (đây là ca dùng chính của dự án)."""
    res = locate_boundary_in_items(_JA_TEXT, _JA_TEXT.index("明日"), _JA_ITEMS, audio_ms=3000.0)
    assert res is not None
    assert res.cut_ms == pytest.approx(1450.0)          # mốc BẮT ĐẦU của "明日"
    assert res.method == "qwen3_aligner"
    assert res.matched_chars == len("今日はいい天気ですね明日は雨です")
    # ranh giới nằm GIỮA danh sách ⇒ hệ số 1.0; coverage = 16/16 ký tự nội dung.
    assert res.confidence == pytest.approx(1.0)


def test_chuan_hoa_rieng_cua_aligner_khong_bi_lech_vi_dau_cham_ascii():
    """Vì sao aligner dùng `normalize_for_aligner` chứ KHÔNG dùng `timer.normalize_for_align`:

    `_IGNORED` của `segmentation.boundary` **không chứa dấu `.` ASCII** (chỉ có `。！？…`)
    và LẠI bỏ `'` — trong khi tokenizer của aligner BỎ `.` và GIỮ `'`. Hai tập luật lệch nhau
    ⇒ đếm ký tự bằng `_IGNORED` sẽ đẩy mốc cắt lùi đúng 1 đơn vị. Đo thật trên FLEURS ja:
    lệch **+600…+700 ms** ở nhiều mẫu (xem `report/audit/27_...md`).
    """
    from backend.asr.timer import normalize_for_align

    # Hai phép chuẩn hoá KHÁC nhau trên cùng một chuỗi:
    assert normalize_for_align("Okay, Charles. ") == "OkayCharles."
    assert normalize_for_aligner("Okay, Charles. ") == "OkayCharles"
    assert normalize_for_align("don't stop") == "dontstop"
    assert normalize_for_aligner("don't stop") == "don'tstop"

    # Nhờ dùng đúng tập ký tự của aligner, ranh giới khớp CHÍNH XÁC từ đầu câu mới.
    res = locate_boundary_in_items(_TEXT, _TEXT.index("It"), _ITEMS, audio_ms=3000.0)
    assert res is not None
    assert res.cut_ms == pytest.approx(1500.0), "mốc phải là mốc BẮT ĐẦU của 'It'"
    assert res.matched_chars == len("OkayCharlesItlooks")
    assert res.confidence == pytest.approx(1.0)


def test_tra_ranh_gioi_o_cuoi_van_ban_lay_moc_ket_thuc():
    res = locate_boundary_in_items(_TEXT, len(_TEXT), _ITEMS, audio_ms=3000.0)
    assert res is not None
    assert res.cut_ms == pytest.approx(2100.0)          # end_time của item cuối
    assert res.confidence < 0.7                          # ở mép ⇒ tin cậy thấp hơn


def test_tra_ranh_gioi_bo_qua_du_lieu_thieu():
    assert locate_boundary_in_items("", 0, _JA_ITEMS) is None
    assert locate_boundary_in_items(_JA_TEXT, 0, []) is None
    assert locate_boundary_in_items("!!!", 1, _JA_ITEMS) is None
    # Item rỗng (aligner có thể trả token toàn dấu câu) không được làm lệch con trỏ.
    items = [{"text": "。", "start_time": 0.0, "end_time": 0.1}] + _JA_ITEMS
    res = locate_boundary_in_items(_JA_TEXT, _JA_TEXT.index("明日"), items, audio_ms=3000.0)
    assert res is not None and res.cut_ms == pytest.approx(1450.0)


def test_moc_ngoai_do_dai_audio_thi_tra_None():
    """Mốc vượt quá độ dài vùng audio ⇒ dữ liệu vô lý, phải bỏ."""
    assert locate_boundary_in_items(_TEXT, _TEXT.index("It"), _ITEMS, audio_ms=1000.0) is None


# ─────────────────────────────────────────────── 3. Qwen3AlignerTimer (model giả)
def _timer_with_fake_model(monkeypatch, items=None, *, language="ja", text=_JA_TEXT,
                           whisper_fallback=True):
    timer = Qwen3AlignerTimer(
        model_id="fake/aligner", language=language, auto_download=True,
        whisper_fallback=whisper_fallback,
    )
    monkeypatch.setattr(timer, "ensure_loaded", lambda: True)
    monkeypatch.setattr(timer, "available", lambda registry=None: True)
    seen: dict = {}

    def _align(pcm, text_in, lang):
        seen["pcm_len"] = int(np.asarray(pcm).size)
        seen["text"] = text_in
        seen["lang"] = lang
        return list(items if items is not None else _JA_ITEMS)

    monkeypatch.setattr(timer, "align_items", _align)
    return timer, seen


def test_aligner_locate_boundary_dung_model_gia(monkeypatch):
    timer, seen = _timer_with_fake_model(monkeypatch)
    pcm = np.zeros(3 * 16000, dtype=np.float32)
    res = timer.locate_boundary(pcm, _JA_TEXT.index("明日"), _JA_TEXT, language="ja")
    assert res is not None
    assert res.cut_ms == pytest.approx(1450.0)
    assert res.elapsed_ms >= 0.0
    assert seen["lang"] == "Japanese", "phải truyền TÊN ngôn ngữ canonical cho processor"
    assert seen["pcm_len"] == pcm.size
    assert seen["text"] == _JA_TEXT


def test_aligner_ngon_ngu_khong_ho_tro_thi_khong_goi_aligner(monkeypatch):
    """Phiên tiếng Việt ⇒ KHÔNG được chạy aligner (kể cả không nạp model).

    Bật `whisper_fallback=False` để cô lập hành vi của aligner (mặc định CÓ fallback ⇒ xem
    `test_aligner_tu_chuyen_sang_whisper_khi_ngon_ngu_khong_ho_tro`).
    """
    timer, seen = _timer_with_fake_model(monkeypatch, language="vi", whisper_fallback=False)
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("không được gọi aligner cho ngôn ngữ không hỗ trợ")

    monkeypatch.setattr(timer, "align_items", _boom)
    res = timer.locate_boundary(np.zeros(16000, dtype=np.float32), 5, _JA_TEXT, language="vi")
    assert res is None
    assert called["n"] == 0
    assert "không hỗ trợ" in timer.last_error
    assert seen == {}


def test_aligner_tu_chuyen_sang_whisper_khi_ngon_ngu_khong_ho_tro(monkeypatch):
    """Mặc định (`whisper_fallback=True`): ngôn ngữ ngoài 11 ngôn ngữ ⇒ dùng whisper,
    KHÔNG để mốc cắt rơi về chồng lấn (giữ nguyên chất lượng hiện tại cho phiên tiếng Việt)."""
    timer, seen = _timer_with_fake_model(monkeypatch, language="vi")
    monkeypatch.setattr(
        timer, "align_items",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("không được chạy aligner")),
    )
    fake_whisper = SimpleNamespace(
        available=lambda registry=None: True,
        close=lambda: None,
        locate_boundary=lambda pcm, idx, text, **kw: TimerResult(
            cut_ms=2222.0, confidence=0.8, method="whisper_segment"
        ),
    )
    monkeypatch.setattr(timer, "_fallback_timer", lambda: fake_whisper)

    res = timer.locate_boundary(np.zeros(16000, dtype=np.float32), 5, _JA_TEXT, language="vi")
    assert res is not None
    assert res.method == "whisper_segment" and res.cut_ms == 2222.0
    assert seen == {}


def test_aligner_fallback_khi_aligner_khong_neo_duoc(monkeypatch):
    """Aligner chạy được nhưng không neo được ⇒ vẫn thử whisper trước khi bỏ cuộc."""
    timer, _ = _timer_with_fake_model(monkeypatch, items=[])
    monkeypatch.setattr(
        timer, "_fallback_timer",
        lambda: SimpleNamespace(
            available=lambda registry=None: True,
            locate_boundary=lambda pcm, idx, text, **kw: TimerResult(
                cut_ms=3333.0, confidence=0.7, method="whisper_segment"
            ),
        ),
    )
    res = timer.locate_boundary(np.zeros(3 * 16000, dtype=np.float32), 20, _JA_TEXT, language="ja")
    assert res is not None and res.method == "whisper_segment" and res.cut_ms == 3333.0


def test_aligner_fallback_tat_thi_tra_None(monkeypatch):
    timer, _ = _timer_with_fake_model(monkeypatch, items=[], whisper_fallback=False)
    monkeypatch.setattr(
        timer, "_fallback_timer",
        lambda: (_ for _ in ()).throw(AssertionError("đã tắt fallback")),
    )
    assert timer.locate_boundary(
        np.zeros(3 * 16000, dtype=np.float32), 20, _JA_TEXT, language="ja"
    ) is None


def test_aligner_prewarm_nap_whisper_khi_ngon_ngu_khong_ho_tro(monkeypatch):
    """Phiên tiếng Việt: prewarm phải nạp whisper, KHÔNG nạp aligner (tiết kiệm 1,8 GB)."""
    timer = Qwen3AlignerTimer(model_id="fake/aligner", language="vi")
    monkeypatch.setattr(
        timer, "ensure_loaded",
        lambda: (_ for _ in ()).throw(AssertionError("không được nạp aligner cho tiếng Việt")),
    )
    loaded = {"whisper": 0}
    monkeypatch.setattr(
        timer, "_fallback_timer",
        lambda: SimpleNamespace(_ensure_session=lambda: loaded.__setitem__("whisper", 1)),
    )
    timer.prewarm()
    assert loaded["whisper"] == 1


def test_aligner_prewarm_nap_aligner_khi_ngon_ngu_duoc_phu(monkeypatch):
    timer = Qwen3AlignerTimer(model_id="fake/aligner", language="ja")
    loaded = {"aligner": 0}
    monkeypatch.setattr(timer, "ensure_loaded", lambda: loaded.__setitem__("aligner", 1) or True)
    monkeypatch.setattr(
        timer, "_fallback_timer",
        lambda: (_ for _ in ()).throw(AssertionError("tiếng Nhật phải dùng aligner")),
    )
    timer.prewarm()
    assert loaded["aligner"] == 1


def test_aligner_chua_co_model_thi_tra_None(monkeypatch):
    timer = Qwen3AlignerTimer(model_id="fake/aligner")
    monkeypatch.setattr(timer, "available", lambda registry=None: False)
    assert timer.locate_boundary(np.zeros(16000, dtype=np.float32), 0, _JA_TEXT) is None


def test_aligner_audio_qua_ngan_thi_tra_None(monkeypatch):
    timer, _ = _timer_with_fake_model(monkeypatch)
    assert timer.locate_boundary(np.zeros(800, dtype=np.float32), 0, _JA_TEXT) is None


def test_aligner_min_confidence_loc_ket_qua(monkeypatch):
    timer, _ = _timer_with_fake_model(monkeypatch, items=[
        {"text": "Okay", "start_time": 0.0, "end_time": 0.4},
        {"text": "Charles", "start_time": 0.4, "end_time": 0.9},
    ])
    pcm = np.zeros(16000, dtype=np.float32)
    # Aligner chỉ trả 2/4 từ ⇒ coverage thấp ⇒ bị ngưỡng tin cậy loại.
    assert timer.locate_boundary(pcm, _TEXT.index("It"), _TEXT, min_confidence=0.9) is None
    assert timer.locate_boundary(pcm, _TEXT.index("It"), _TEXT, min_confidence=0.1) is not None


def test_aligner_align_items_nuot_loi(monkeypatch):
    """Lỗi suy luận chỉ được log + trả [] (không làm chết luồng chốt câu)."""
    timer = Qwen3AlignerTimer(model_id="fake/aligner")
    monkeypatch.setattr(timer, "ensure_loaded", lambda: True)

    class _Boom:
        def prepare_forced_aligner_inputs(self, **kwargs):
            raise RuntimeError("hết VRAM")

    timer._processor = _Boom()
    timer._model = SimpleNamespace(device="cpu", dtype=None)
    assert timer.align_items(np.zeros(16000, dtype=np.float32), "hello", "English") == []
    assert "hết VRAM" in timer.last_error


def _make_local_model(tmp_path, *, weights: bool = True) -> None:
    """Dựng thư mục model cục bộ tối thiểu để `_local_dir_ready()` chấp nhận."""
    for name in ("config.json", "tokenizer_config.json", "processor_config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    if weights:
        (tmp_path / "model.safetensors").write_bytes(b"\x00")


def test_aligner_available_theo_thu_muc_cuc_bo(tmp_path):
    timer = Qwen3AlignerTimer(
        model_id="fake/aligner", local_dir=str(tmp_path), auto_download=False,
        whisper_fallback=False,
    )
    assert timer.available() is False, "thư mục rỗng ⇒ chưa sẵn sàng"
    _make_local_model(tmp_path, weights=False)
    timer.available_negative_ttl_sec = 0.0
    assert timer.available() is False, "thiếu file trọng số ⇒ chưa sẵn sàng"
    _make_local_model(tmp_path)
    timer.available_negative_ttl_sec = 0.0
    assert timer.available() is True


def test_aligner_mac_dinh_dung_backend_models():
    """Đường dẫn cục bộ mặc định phải nằm trong `backend/models/` (KHÔNG dùng cache HF)."""
    from backend.config import MODELS_DIR

    timer = Qwen3AlignerTimer(model_id="Qwen/Qwen3-ForcedAligner-0.6B-hf")
    path = timer._local_dir_path()
    assert path.parent == MODELS_DIR, path
    assert path.name == "Qwen__Qwen3-ForcedAligner-0.6B-hf", path
    # Cấu hình tường minh thì tôn trọng cấu hình.
    assert Qwen3AlignerTimer(model_id="x/y", local_dir="D:/tmp/z")._local_dir_path().name == "z"


def test_aligner_tu_tai_vao_backend_models_va_chi_nap_tu_do(tmp_path, monkeypatch):
    """Model phải được tải về thư mục cục bộ rồi nạp từ ĐÓ (không nạp từ cache HF)."""
    timer = Qwen3AlignerTimer(
        model_id="fake/aligner", local_dir=str(tmp_path), auto_download=True
    )
    seen: dict = {}

    def _fake_snapshot(repo_id, local_dir=None, allow_patterns=None):
        seen["repo_id"] = repo_id
        seen["local_dir"] = local_dir
        _make_local_model(Path(local_dir))
        return local_dir

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot, raising=False)

    class _FakeProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            seen["processor_source"] = source
            seen["processor_kwargs"] = kwargs
            return object()

    class _FakeModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            seen["model_source"] = source
            seen["model_kwargs"] = kwargs
            return SimpleNamespace(eval=lambda: SimpleNamespace())

    import torch

    # ⚠️ `transformers` là LAZY MODULE: `getattr(transformers, "AutoProcessor")` trả bản đã
    # monkeypatch, nhưng `from transformers import AutoProcessor` (đúng câu lệnh trong
    # `aligner_timer.ensure_loaded`) lại trả về lớp THẬT từ submodule. Phải vá cả 3 chỗ.
    import transformers
    import transformers.models.auto.processing_auto as _proc_mod
    import transformers.models.qwen3_asr.modeling_qwen3_asr as _model_mod

    for target in (transformers, _proc_mod):
        monkeypatch.setattr(target, "AutoProcessor", _FakeProcessor, raising=False)
    for target in (transformers, _model_mod):
        monkeypatch.setattr(
            target, "Qwen3ASRForTokenClassification", _FakeModel, raising=False
        )
    monkeypatch.setattr(timer, "_warmup", lambda device: None)
    monkeypatch.setattr(timer, "_pick_device", lambda torch_mod: "cpu")
    monkeypatch.setattr(Qwen3AlignerTimer, "ensure_loaded", _REAL_ENSURE_LOADED)

    assert timer.ensure_loaded() is True
    assert seen["repo_id"] == "fake/aligner"
    assert Path(seen["local_dir"]) == tmp_path
    assert seen["processor_source"] == str(tmp_path), "phải nạp từ thư mục cục bộ"
    assert seen["model_source"] == str(tmp_path), "phải nạp từ thư mục cục bộ"
    assert seen["model_kwargs"].get("local_files_only") is True, "không được chạm mạng"


def test_aligner_khong_tu_tai_khi_tat_auto_download(tmp_path, monkeypatch):
    timer = Qwen3AlignerTimer(
        model_id="fake/aligner", local_dir=str(tmp_path), auto_download=False,
        whisper_fallback=False,
    )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("không được tải")),
        raising=False,
    )
    monkeypatch.setattr(Qwen3AlignerTimer, "ensure_loaded", _REAL_ENSURE_LOADED)
    assert timer.ensure_loaded() is False
    assert "auto_download=False" in timer.last_error


def test_aligner_available_nho_whisper_du_phong(monkeypatch):
    """Không có model aligner cục bộ NHƯNG còn whisper ⇒ timer vẫn khả dụng (không được báo
    'không có model' rồi để mốc cắt rơi về chồng lấn)."""
    timer = Qwen3AlignerTimer(model_id="fake/aligner", auto_download=False)
    monkeypatch.setattr(timer, "_local_dir_ready", lambda path: False)
    monkeypatch.setattr(
        timer, "_fallback_timer",
        lambda: SimpleNamespace(available=lambda registry=None: True),
    )
    assert timer.available() is True

    off = Qwen3AlignerTimer(
        model_id="fake/aligner", auto_download=False, whisper_fallback=False
    )
    monkeypatch.setattr(off, "_local_dir_ready", lambda path: False)
    assert off.available() is False


def test_aligner_dtype_va_device(monkeypatch):
    import torch

    timer = Qwen3AlignerTimer(dtype="float16", device="cpu")
    assert timer._torch_dtype(torch) is torch.float16
    assert Qwen3AlignerTimer(dtype="bf16")._torch_dtype(torch) is torch.bfloat16
    assert Qwen3AlignerTimer(dtype="lạ")._torch_dtype(torch) is torch.bfloat16
    assert timer._pick_device(torch) == "cpu"


# ───────────────────────────────────────────── 4. dispatch theo `timer_engine`
def test_normalize_timer_engine():
    assert normalize_timer_engine("qwen3-aligner") == "qwen3-aligner"
    assert normalize_timer_engine("qwen3_aligner") == "qwen3-aligner"
    assert normalize_timer_engine("ALIGNER") == "qwen3-aligner"
    assert normalize_timer_engine("whisper") == "whisper"
    assert normalize_timer_engine("") == "whisper"
    assert normalize_timer_engine("rác") == "whisper"


def test_get_seg_timer_dispatch_theo_cau_hinh(restore_config):
    from backend.config import config

    config.segmentation.timer_engine = "qwen3-aligner"
    config.segmentation.aligner_model = "fake/aligner"
    timer_mod.reset_seg_timer()
    assert isinstance(timer_mod.get_seg_timer(), Qwen3AlignerTimer)
    # Cùng engine ⇒ cùng instance (singleton).
    assert timer_mod.get_seg_timer() is timer_mod.get_seg_timer()

    config.segmentation.timer_engine = "whisper"
    assert isinstance(timer_mod.get_seg_timer(), WhisperTimer)


def test_doi_engine_thi_dong_timer_cu(restore_config, monkeypatch):
    from backend.config import config

    closed = []
    monkeypatch.setattr(WhisperTimer, "close", lambda self: closed.append("whisper"))

    config.segmentation.timer_engine = "whisper"
    timer_mod.reset_seg_timer()
    old = timer_mod.get_seg_timer()
    assert isinstance(old, WhisperTimer)

    config.segmentation.timer_engine = "qwen3-aligner"
    new = timer_mod.get_seg_timer()
    assert isinstance(new, Qwen3AlignerTimer)
    assert closed == ["whisper"], "đổi engine phải ĐÓNG timer cũ (nhả model/VRAM)"
    timer_mod.reset_seg_timer()


def test_update_seg_config_doi_engine_reset_prewarm(restore_config, monkeypatch):
    """Đổi engine qua WS/REST phải có hiệu lực NGAY (không phải 'cấu hình vô hiệu')."""
    from backend.asr.engine import TranscribeEngine
    from backend.config import config

    monkeypatch.setattr(TranscribeEngine, "prewarm_seg_timer", lambda self: None)
    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = True
    config.segmentation.timer_engine = "whisper"      # trạng thái ĐẦU (không phụ thuộc default)
    engine = TranscribeEngine(model_key=config.asr.active_model)
    engine._seg_timer_prewarmed = True   # giả như đã nạp whisper

    engine.update_seg_config(timer_engine="qwen3-aligner")
    assert config.segmentation.timer_engine == "qwen3-aligner"
    assert engine._seg_timer_prewarmed is False, "phải cho phép nạp lại model của engine mới"


def test_update_seg_config_khong_nap_lai_khi_cau_hinh_khong_doi(restore_config, monkeypatch):
    """REGRESSION: popup đồng bộ cấu hình ở MỖI lần kết nối và luôn gửi `use_whisper_timer`.

    Trước đây `update_seg_config` reset timer chỉ vì khoá CÓ MẶT trong kwargs ⇒ mỗi lần đồng
    bộ đóng model aligner (1,8 GB) rồi **nạp lại từ đầu**: log phiên thật cho thấy 5 lần nạp
    × ~9–12 s cho 2 phiên. Giờ chỉ reset khi ENGINE ĐỔI.
    """
    from backend.asr.engine import TranscribeEngine
    from backend.asr import timer as timer_mod
    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.timer_engine = "qwen3-aligner"
    engine = TranscribeEngine(model_key=config.asr.active_model)
    monkeypatch.setattr(TranscribeEngine, "prewarm_seg_timer", lambda self: None)
    engine._seg_timer_prewarmed = True                # model đã nạp

    resets: list[int] = []
    monkeypatch.setattr(timer_mod, "reset_seg_timer", lambda: resets.append(1))

    # Đúng chuỗi thông điệp mà popup gửi khi đồng bộ (giá trị KHÔNG đổi).
    for _ in range(3):
        engine.update_seg_config(
            enabled=True, max_chars=80, tail_min_chars=3, tail_scans=2,
            use_whisper_timer=True, timer_engine="qwen3-aligner",
        )
    assert resets == [], "cấu hình không đổi ⇒ KHÔNG được đóng/nạp lại model timer"
    assert engine._seg_timer_prewarmed is True

    # Đổi engine thật ⇒ phải reset.
    engine.update_seg_config(timer_engine="whisper")
    assert resets == [1]


def test_api_config_co_timer_engine(restore_config):
    from backend import main as main_mod

    payload = main_mod._build_config_response(include_catalog=False)
    seg = payload.get("seg")
    assert isinstance(seg, dict)
    for key in ("timer_engine", "aligner_model", "aligner_device", "use_whisper_timer"):
        assert key in seg, f"thiếu khoá '{key}' trong /api/config"


def test_session_apply_config_doi_engine_timer(session_factory, restore_config, monkeypatch):
    from backend.asr.engine import TranscribeEngine
    from backend.config import config

    # Tầng A: không nạp model timer thật (xem conftest `_forbid_heavy_model_loads`).
    monkeypatch.setattr(TranscribeEngine, "prewarm_seg_timer", lambda self: None)

    session = session_factory()
    session.apply_config({"segTimerEngine": "qwen3-aligner"})
    assert config.segmentation.timer_engine == "qwen3-aligner"
    assert session.config["seg_timer_engine"] == "qwen3-aligner"

    # Giá trị rác ⇒ quay về mặc định (không được đặt engine không tồn tại).
    session.apply_config({"segTimerEngine": "rác"})
    assert config.segmentation.timer_engine == "whisper"


def test_get_seg_timer_lam_moi_ngon_ngu_phien(restore_config):
    """Timer là singleton dùng chung nhiều phiên ⇒ `language` phải được làm mới, nếu không
    phiên tiếng Việt sau phiên tiếng Nhật sẽ khiến `prewarm()` nạp aligner vô ích."""
    from backend.config import config

    config.segmentation.timer_engine = "qwen3-aligner"
    config.asr.language = "ja"
    timer_mod.reset_seg_timer()
    timer = timer_mod.get_seg_timer()
    assert timer.language == "ja"

    config.asr.language = "vi"
    assert timer_mod.get_seg_timer() is timer, "vẫn là singleton"
    assert timer.language == "vi", "phải cập nhật ngôn ngữ của phiên hiện tại"
    timer_mod.reset_seg_timer()


def test_engine_prewarm_dung_engine_dang_chon(restore_config, monkeypatch):
    """`prewarm_seg_timer()` phải nạp đúng engine đang cấu hình (đây là chỗ duy nhất
    nạp model timer trong luồng nền)."""
    from backend.asr.engine import TranscribeEngine
    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = True
    config.segmentation.timer_engine = "qwen3-aligner"
    engine = TranscribeEngine(model_key=config.asr.active_model)
    engine._seg_timer_prewarmed = False

    done = threading.Event()
    fake = SimpleNamespace(
        model_key="fake/aligner",
        last_error="",
        available=lambda registry=None: True,
        _ensure_session=lambda: (done.set(), object())[1],
    )
    monkeypatch.setattr(timer_mod, "get_seg_timer", lambda: fake)

    engine.prewarm_seg_timer()
    assert done.wait(5.0), "prewarm phải nạp model trong luồng nền"


def test_prewarm_khong_nap_khi_tat_timer(restore_config, monkeypatch):
    from backend.asr.engine import TranscribeEngine
    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = False
    engine = TranscribeEngine(model_key=config.asr.active_model)
    engine._seg_timer_prewarmed = False

    def _boom():
        raise AssertionError("tắt timer thì KHÔNG được nạp model timer")

    monkeypatch.setattr(timer_mod, "get_seg_timer", _boom)
    engine.prewarm_seg_timer()
    assert engine._seg_timer_prewarmed is False


def test_engine_dung_aligner_lam_moc_cat(restore_config, monkeypatch):
    """Đường nối engine → aligner: mốc cắt = mốc của aligner (không phải cuối vùng nói)."""
    from backend.asr.engine import TranscribeEngine
    from backend.config import config
    from backend.core.pipeline_events import CommitReason

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = True
    config.segmentation.timer_engine = "qwen3-aligner"
    config.segmentation.timer_min_confidence = 0.3
    config.segmentation.tail_scans = 1
    config.segmentation.tail_stable_ms = 0.0
    engine = TranscribeEngine(model_key=config.asr.active_model)

    class _FakeAligner(Qwen3AlignerTimer):
        def available(self, registry=None):
            return True

        def locate_boundary(self, pcm, boundary_index, main_text, **kwargs):
            self.seen_language = kwargs.get("language")
            return TimerResult(cut_ms=1234.0, confidence=0.9, method="qwen3_aligner")

    fake = _FakeAligner(model_id="fake/aligner")
    monkeypatch.setattr(timer_mod, "get_seg_timer", lambda: fake)

    text = "こんにちは。今日はいい天気ですね。"
    assert engine._evaluate_seg(text, 48_000) is CommitReason.SEG_PUNCT
    assert engine._seg_cut_sample(0, 48_000, text) == int(1.234 * 16000)
    assert fake.seen_language == engine.language, "phải truyền ngôn ngữ của phiên cho aligner"
