"""Unit tests cho ForcedAlignerService và Sentence Grouping logic."""

from __future__ import annotations

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
