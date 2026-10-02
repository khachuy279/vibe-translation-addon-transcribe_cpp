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
