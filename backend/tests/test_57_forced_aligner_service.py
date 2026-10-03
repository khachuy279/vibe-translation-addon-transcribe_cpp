"""Unit tests cho ForcedAlignerService và Sentence Grouping logic."""

from __future__ import annotations

import numpy as np
import pytest

from backend.asr.forced_aligner import (
    AlignedWord,
    ForcedAlignerService,
    SubtitleSentence,
)


class TestForcedAlignerService:
    """Kiểm thử cấu trúc dữ liệu và logic gom câu của ForcedAlignerService."""

    def test_singleton_instance(self):
        s1 = ForcedAlignerService.get_instance()
        s2 = ForcedAlignerService.get_instance()
        assert s1 is s2

    def test_sentence_grouping_english_punctuation(self):
        """Gom câu tiếng Anh dựa vào dấu chấm kết thúc câu."""
        words = [
            AlignedWord(text="Hello", start_time=0.2, end_time=0.5),
            AlignedWord(text="everyone.", start_time=0.6, end_time=1.1),
            AlignedWord(text="Welcome", start_time=1.5, end_time=1.9),
            AlignedWord(text="to", start_time=2.0, end_time=2.1),
            AlignedWord(text="AI", start_time=2.2, end_time=2.5),
            AlignedWord(text="series!", start_time=2.6, end_time=3.0),
        ]

        subs = ForcedAlignerService.group_words_to_subtitles(
            words, language="English", max_duration_sec=4.5, max_words=10
        )

        assert len(subs) == 2
        # Câu 1
        assert subs[0].text == "Hello everyone."
        assert subs[0].start_time == 0.2
        assert subs[0].end_time == 1.1
        assert len(subs[0].words) == 2

        # Câu 2
        assert subs[1].text == "Welcome to AI series!"
        assert subs[1].start_time == 1.5
        assert subs[1].end_time == 3.0
        assert len(subs[1].words) == 4

    def test_sentence_grouping_japanese_punctuation(self):
        """Gom câu tiếng Nhật dựa vào dấu 。 và không chèn khoảng trắng."""
        words = [
            AlignedWord(text="私", start_time=0.48, end_time=0.72),
            AlignedWord(text="の", start_time=0.72, end_time=1.12),
            AlignedWord(text="人生", start_time=1.44, end_time=1.76),
            AlignedWord(text="だ。", start_time=1.84, end_time=2.50),
            AlignedWord(text="今日", start_time=3.00, end_time=3.40),
            AlignedWord(text="も", start_time=3.40, end_time=3.60),
            AlignedWord(text="頑張る。", start_time=3.70, end_time=4.20),
        ]

        subs = ForcedAlignerService.group_words_to_subtitles(
            words, language="Japanese", max_duration_sec=4.5
        )

        assert len(subs) == 2
        # Câu 1: không có khoảng trắng
        assert subs[0].text == "私の人生だ。"
        assert subs[0].start_time == 0.48
        assert subs[0].end_time == 2.50

        # Câu 2:
        assert subs[1].text == "今日も頑張る。"
        assert subs[1].start_time == 3.00
        assert subs[1].end_time == 4.20

    def test_sentence_grouping_max_duration_split(self):
        """Tách câu khi vượt quá max_duration mà không có dấu câu."""
        words = [
            AlignedWord(text=f"word{i}", start_time=float(i), end_time=float(i) + 0.8)
            for i in range(10)  # 10s không có dấu câu
        ]

        subs = ForcedAlignerService.group_words_to_subtitles(
            words, language="English", max_duration_sec=3.0, max_words=10
        )

        assert len(subs) > 1
        for s in subs:
            assert s.duration <= 4.0  # mỗi câu bị giới hạn thời lượng

    def test_cap_defers_to_punctuation_instead_of_splitting_phrase(self):
        """Chạm trần từ mà sắp có dấu kết câu ⇒ NỚI tới dấu câu, KHÔNG chẻ giữa cụm từ.

        Sự cố 2026-10-02: `max_words = 10` cắt cứng nên "…to trick children" và "into loving him,"
        thành hai phụ đề — đúng triệu chứng "cuối câu này là đầu của câu sau" NGAY TRONG một khối.
        """
        words = [
            AlignedWord(text=f"w{i}", start_time=i * 0.4, end_time=i * 0.4 + 0.35)
            for i in range(12)
        ]
        words[11] = AlignedWord(text="end.", start_time=11 * 0.4, end_time=11 * 0.4 + 0.35)

        subs = ForcedAlignerService.group_words_to_subtitles(
            words,
            language="English",
            max_words=10,
            max_duration_sec=30.0,
            max_chars=400,
            defer_words=3,
        )

        assert len(subs) == 1, subs
        assert subs[0].text.endswith("end.")
        assert len(subs[0].words) == 12, "phải nới qua trần 10 từ để đóng trọn câu"

    def test_lowercase_continuation_is_merged(self):
        """Mảnh kết thúc KHÔNG có dấu câu + mảnh sau bắt đầu chữ thường ⇒ gộp lại thành một câu."""
        tokens = "we went to the park and then home".split()
        words = [
            AlignedWord(text=t, start_time=i * 0.3, end_time=i * 0.3 + 0.25)
            for i, t in enumerate(tokens)
        ]

        subs = ForcedAlignerService.group_words_to_subtitles(
            words,
            language="English",
            max_words=6,
            max_duration_sec=30.0,
            max_chars=400,
            defer_words=3,
            min_words=2,
        )

        assert len(subs) == 1, [s.text for s in subs]
        assert subs[0].text == "we went to the park and then home"

    def test_sentence_boundary_still_splits_when_merged_would_be_too_big(self):
        """Bước gộp có TRẦN CỨNG: không được biến nhiều câu thành một phụ đề khổng lồ."""
        tokens = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
                  "nu xi omicron pi rho sigma tau").split()
        words = [
            AlignedWord(text=t, start_time=i * 0.5, end_time=i * 0.5 + 0.4)
            for i, t in enumerate(tokens)
        ]

        subs = ForcedAlignerService.group_words_to_subtitles(
            words,
            language="English",
            max_words=6,
            max_duration_sec=3.0,
            max_chars=40,
            defer_words=3,
            min_words=2,
        )

        assert len(subs) > 1
        # Trần cứng = max_words + defer_words = 9 từ
        for s in subs:
            assert len(s.words) <= 9, s.text

    def test_ensure_forced_aligner_model_local_path(self, tmp_path):
        """Kiểm tra ensure_forced_aligner_model luôn trả về đường dẫn cục bộ khi đã có file."""
        from backend.asr.forced_aligner import ensure_forced_aligner_model, ALIGNER_LOCAL_DIR

        # 1. Thư mục giả lập đầy đủ file
        fake_dir = tmp_path / "Qwen3-ForcedAligner-0.6B"
        fake_dir.mkdir()
        (fake_dir / "config.json").write_text("{}", encoding="utf-8")
        (fake_dir / "model.safetensors").write_text("fake_weights", encoding="utf-8")

        resolved = ensure_forced_aligner_model(local_dir=fake_dir, allow_download=False)
        assert resolved == fake_dir

        # 2. ForcedAlignerService mặc định trỏ vào backend/models/Qwen3-ForcedAligner-0.6B
        service = ForcedAlignerService()
        assert service.model_path == ALIGNER_LOCAL_DIR


class TestSourcePunctuationReattachment:
    """Sự cố 2026-10-02: aligner BỎ HẾT dấu câu khi tokenize ⇒ phụ đề phải gắn lại từ văn bản ASR.

    Bằng chứng upstream: `qwen_asr/inference/qwen3_forced_aligner.py::is_kept_char` chỉ giữ category
    Unicode L/N + dấu nháy đơn. Hệ quả thật (xem `report/09_lookahead_offline_batch/
    phase1_poc_results.md`): ASR trả `"私の人生の中心で最も魅力的だ。"` nhưng item cuối là `"魅力的だ"`,
    phụ đề gom ra mất hẳn `。`.
    """

    def test_reattach_english_punctuation(self):
        """Từ không dấu câu + văn bản ASR có dấu câu ⇒ từng từ nhận đúng dấu của nó."""
        items = [
            AlignedWord("You", 0.0, 0.3),
            AlignedWord("took", 0.3, 0.6),
            AlignedWord("ballet", 0.6, 1.0),
            AlignedWord("God", 1.2, 1.5),
            AlignedWord("you", 1.5, 1.7),
            AlignedWord("never", 1.7, 2.0),
            AlignedWord("listen", 2.0, 2.4),
        ]
        merged = ForcedAlignerService.merge_source_text(
            items, "You took ballet. God, you never listen."
        )
        assert [w.text for w in merged] == [
            "You", "took", "ballet.", "God,", "you", "never", "listen.",
        ]
        # Mốc thời gian KHÔNG đổi
        assert [w.start_time for w in merged] == [i.start_time for i in items]

    def test_reattach_japanese_punctuation(self):
        """CJK: ký tự cuối nhận dấu 。 — đúng ca đã đo trong POC Phase 1."""
        items = [
            AlignedWord("私", 0.48, 0.72),
            AlignedWord("の", 0.72, 1.12),
            AlignedWord("人生", 1.44, 1.76),
            AlignedWord("だ", 1.84, 2.50),
        ]
        merged = ForcedAlignerService.merge_source_text(items, "私の人生だ。")
        assert [w.text for w in merged] == ["私", "の", "人生", "だ。"]
        # Nối lại đúng nguyên văn văn bản ASR
        assert "".join(w.text for w in merged) == "私の人生だ。"

    def test_reattach_keeps_inner_punctuation(self):
        """Dấu nháy typographic BÊN TRONG từ cũng phải được giữ (upstream loại nó)."""
        items = [AlignedWord("Its", 0.0, 0.3), AlignedWord("fine", 0.3, 0.6)]
        merged = ForcedAlignerService.merge_source_text(items, "It’s fine.")
        assert merged[0].text == "It’s"
        assert merged[1].text == "fine."

    def test_grouping_after_reattach_yields_whole_sentences(self):
        """Đúng đề bài người dùng: đoạn ASR hoàn hảo ⇒ tách thành những CÂU hoàn hảo."""
        asr_text = (
            "I promise I'll be on my best behavior. You better be. "
            "No jokes about how close I am with my dog, or the truth about how close I am with my dog. "
            "You got it."
        )
        tokens = asr_text.split()
        items = [
            # Aligner: cùng thứ tự nhưng KHÔNG có dấu câu (giống hệt hành vi upstream)
            AlignedWord(
                "".join(ch for ch in tok if ch.isalnum() or ch == "'"),
                i * 0.35,
                i * 0.35 + 0.30,
            )
            for i, tok in enumerate(tokens)
        ]
        merged = ForcedAlignerService.merge_source_text(items, asr_text)
        subs = ForcedAlignerService.group_words_to_subtitles(
            merged,
            language="English",
            max_words=10,
            max_duration_sec=4.5,
            sentence_max_words=24,
            sentence_max_duration_sec=8.0,
        )
        assert [s.text for s in subs] == [
            "I promise I'll be on my best behavior.",
            "You better be.",
            "No jokes about how close I am with my dog, or the truth about how close I am with my dog.",
            "You got it.",
        ]

    # ── Hồi quy 2026-10-03: "Pipeline B không ngắt câu dài" trên trang tiếng Nhật ──────────

    def test_detect_aligner_language_from_text(self):
        """`source_lang="auto"` ⇒ phải ĐOÁN ngôn ngữ từ văn bản ASR (nếu không sẽ rơi về English)."""
        from backend.asr.forced_aligner import detect_aligner_language_from_text as detect
        from backend.asr.forced_aligner import resolve_aligner_language

        assert resolve_aligner_language("auto") is None
        assert detect("ほら、あれ温泉街じゃない？") == "Japanese"
        assert detect("こんにちは") == "Japanese"
        assert detect("カタカナです") == "Japanese"
        assert detect("今天天气很好。") == "Chinese"
        assert detect("안녕하세요 반갑습니다") == "Korean"
        assert detect("We're not getting divorced.") is None
        assert detect("") is None
        # Người dùng chọn tay thì luôn thắng phép đoán.
        assert resolve_aligner_language("ja") == "Japanese"

    def test_wrong_aligner_language_collapses_whole_sentence_into_one_item(self):
        """GỐC RỄ: tiếng Nhật + `language="English"` ⇒ upstream tách theo KHOẢNG TRẮNG.

        Tiếng Nhật không có khoảng trắng ⇒ cả câu thành MỘT token ⇒ aligner trả một item ⇒ tầng
        gom câu không thể ngắt ⇒ 3 câu hiện thành 1 phụ đề dài (đúng ảnh chụp của người dùng).
        """
        text = "ほら、あれ温泉街じゃない？うわ、なんかお土産屋とか書いてあるよ。ああ、なんか温泉の香りしてきた。"
        # Mô phỏng `tokenize_space_lang` (language="English"): cả chuỗi là 1 token.
        one_item = [AlignedWord(text=text, start_time=0.0, end_time=6.4)]
        merged = ForcedAlignerService.merge_source_text(one_item, text)
        subs = ForcedAlignerService.group_words_to_subtitles(
            merged, language="English", max_words=10, max_duration_sec=4.5
        )
        assert len(subs) == 1, "Bản cũ: cả khối thành một phụ đề (đây là lỗi)"
        assert subs[0].text == text

    def test_safety_net_splits_sentence_carrying_many_sentences(self):
        """Lưới AN TOÀN: một phụ đề chứa nhiều câu phải bị tách theo dấu kết câu."""
        text = "ほら、あれ温泉街じゃない？うわ、なんかお土産屋とか書いてあるよ。ああ、なんか温泉の香りしてきた。"
        subs = [SubtitleSentence(text=text, start_time=10.0, end_time=16.4)]

        fixed = ForcedAlignerService.split_oversized_sentences(
            subs, language="Japanese", max_chars=45
        )

        assert [s.text for s in fixed] == [
            "ほら、あれ温泉街じゃない？",
            "うわ、なんかお土産屋とか書いてあるよ。",
            "ああ、なんか温泉の香りしてきた。",
        ]
        # Thời gian: tăng dần, phủ đúng khoảng gốc, không chồng lấn.
        assert fixed[0].start_time == 10.0
        assert fixed[-1].end_time == pytest.approx(16.4, abs=1e-6)
        prev_end = None
        for s in fixed:
            assert s.end_time > s.start_time
            if prev_end is not None:
                assert s.start_time == pytest.approx(prev_end, abs=1e-6)
            prev_end = s.end_time

    def test_safety_net_hard_splits_when_no_punctuation(self):
        """ASR không sinh dấu câu ⇒ vẫn phải cắt theo trần ký tự (CJK), không để một dòng dài."""
        text = "あ" * 100  # 100 ký tự, không dấu câu
        subs = [SubtitleSentence(text=text, start_time=0.0, end_time=20.0)]
        fixed = ForcedAlignerService.split_oversized_sentences(
            subs, language="Japanese", max_chars=45
        )
        assert [len(s.text) for s in fixed] == [45, 45, 10]
        assert "".join(s.text for s in fixed) == text

    def test_safety_net_leaves_normal_subtitles_untouched(self):
        """Không được đụng vào phụ đề vốn đã đúng (mỗi phụ đề một câu)."""
        subs = [
            SubtitleSentence(text="You better be.", start_time=0.0, end_time=1.0),
            SubtitleSentence(text="You got it.", start_time=1.0, end_time=2.0),
        ]
        fixed = ForcedAlignerService.split_oversized_sentences(
            subs, language="English", max_words=10
        )
        assert fixed == subs

    def test_real_upstream_tokenizer_explains_single_long_subtitle(self):
        """GỐC RỄ, đo bằng TOKENIZER THẬT của upstream (không cần model).

        `language="English"` ⇒ `tokenize_space_lang` + `split_segment_with_chinese` gộp cả cụm
        kana thành token, nên dấu `？`/`。` rơi vào GIỮA token ⇒ tầng gom câu KHÔNG thấy dấu kết
        câu ở cuối token nào ⇒ không ngắt được; bước gộp mảnh cuối (rule 3) lại gộp về một ⇒ cả
        khối 3 câu thành MỘT phụ đề dài (đúng ảnh chụp của người dùng).
        `language="Japanese"` (nagisa) cho 28 từ đúng ⇒ 3 phụ đề.
        """
        pytest.importorskip("nagisa")
        proc_mod = pytest.importorskip(
            "backend.asr.qwen_asr.inference.qwen3_forced_aligner"
        )
        processor = proc_mod.Qwen3ForceAlignProcessor()
        text = (
            "ほら、あれ温泉街じゃない？"
            "うわ、なんかお土産屋とか書いてあるよ。"
            "ああ、なんか温泉の香りしてきた。"
        )

        def group_with(language: str):
            words, _ = processor.encode_timestamp(text, language)
            step = 6.4 / max(1, len(words))
            items = [
                AlignedWord(text=w, start_time=i * step, end_time=(i + 1) * step)
                for i, w in enumerate(words)
            ]
            merged = ForcedAlignerService.merge_source_text(items, text)
            return ForcedAlignerService.group_words_to_subtitles(
                merged, language=language, max_words=10, max_duration_sec=4.5,
                max_chars=45, min_words=2, defer_words=3,
                sentence_max_words=24, sentence_max_duration_sec=8.0,
            )

        wrong = group_with("English")
        assert len(wrong) == 1, "Bản CŨ: cả khối thành một phụ đề (đây là lỗi)"
        assert " " in wrong[0].text, "chọn sai ngôn ngữ ⇒ text còn bị nối bằng KHOẢNG TRẮNG"

        right = group_with("Japanese")
        assert [s.text for s in right] == [
            "ほら、あれ温泉街じゃない？",
            "うわ、なんかお土産屋とか書いてあるよ。",
            "ああ、なんか温泉の香りしてきた。",
        ], [s.text for s in right]

    def test_align_service_attaches_punctuation_end_to_end(self, monkeypatch):
        """`ForcedAlignerService.align()` phải tự gắn dấu câu (không cần model thật)."""
        from backend.asr.forced_aligner import ForcedAlignerService as SVC

        raw = [
            ("You", 0.0, 0.3),
            ("took", 0.3, 0.6),
            ("ballet", 0.6, 1.0),
            ("God", 1.2, 1.5),
        ]

        class _Item:
            def __init__(self, text, start, end):
                self.text, self.start_time, self.end_time = text, start, end

        class _Result:
            def __init__(self):
                self.items = [_Item(*r) for r in raw]

        class _StubAligner:
            def align(self, audio, text, language):
                return [_Result()]

        svc = SVC()
        monkeypatch.setattr(svc, "load_model", lambda: _StubAligner())
        monkeypatch.setattr(svc, "_is_warmed", True, raising=False)
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        words = svc.align(
            np.zeros(16000, dtype=np.float32),
            "You took ballet. God, you never listen.",
            "English",
        )
        assert [w.text for w in words] == ["You", "took", "ballet.", "God,"]
        assert any(w.text.endswith(".") for w in words), "phải có dấu kết câu để tầng gom câu dùng"
