import pytest
from backend.asr.forced_aligner import AlignedWord, ForcedAlignerService, SubtitleSentence


def test_abbreviations_do_not_split_sentences():
    """HỒI QUY 2026-10-05: `Dr.` bị coi là KẾT CÂU ⇒ phụ đề vụn và bản dịch sai.

    Log thật (align=English):
      'Fine, Dr. Kuthrapali. … Right on time, Dr. Kuthrapali. I present Dr. Milstone from MIT. …'
      bị ngắt thành 15 phụ đề: "Fine, Dr." ⏐ "Kuthrapali." ⏐ … ⏐ "I present Dr." ⏐ "Milstone from MIT."
    """
    text = (
        "Fine, Dr. Kuthrapali. Thank you, sir. I'm sorry. Am I late? No, no, no. "
        "Right on time, Dr. Kuthrapali. I present Dr. Milstone from MIT. "
        "She'll be heading up our data analysis team. "
        "It's nice to meet you, Dr. Kuthrapali."
    )
    words = text.split()
    step = 24.0 / max(1, len(words))
    items = [
        AlignedWord(text=w, start_time=round(i * step, 3), end_time=round((i + 1) * step, 3))
        for i, w in enumerate(words)
    ]
    merged = ForcedAlignerService.merge_source_text(items, text)
    subs = ForcedAlignerService.group_words_to_subtitles(merged, language="English")
    texts = [s.text for s in subs]

    # 1. Không được có mảnh CỤT kết thúc bằng "Dr." (chính dạng lỗi trong log:
    #    'Fine, Dr.' ⏐ 'Kuthrapali.' ⏐ 'I present Dr.' ⏐ 'Milstone from MIT.').
    assert not any(t.strip().endswith("Dr.") for t in texts), f"còn mảnh cụt 'Dr.': {texts}"
    assert not any(t.strip() == "Kuthrapali." for t in texts), f"còn mảnh cụt tên riêng: {texts}"
    # 2. "Dr. <Tên>" phải nằm nguyên trong MỘT câu.
    assert any("Fine, Dr. Kuthrapali." in t for t in texts), texts
    assert any("I present Dr. Milstone from MIT." in t for t in texts), texts
    # 3. Nhưng dấu chấm THẬT vẫn phải kết câu (không gộp cả khối thành một câu khổng lồ).
    assert len(texts) >= 4, texts

    # 4. Lưới an toàn theo trần ký tự cũng phải tôn trọng từ viết tắt.
    res = ForcedAlignerService.split_oversized_sentences(
        [SubtitleSentence(text=text, start_time=0.0, end_time=24.0)],
        language="English", max_chars=80, max_words=12,
    )
    res_texts = [r.text for r in res]
    assert not any(t.strip() in ("Dr.", "Kuthrapali.", "Dr") for t in res_texts), res_texts
    assert any("Fine, Dr. Kuthrapali." in t for t in res_texts), res_texts


def test_english_period_segmentation_safety_net():
    """Kiểm tra không bị chẻ vụn các câu tiếng Anh có dấu chấm kết câu."""
    s1 = SubtitleSentence(
        text="I can actually feel the toxins being pulled out of my skin. Well, this is a moisturizing mask.",
        start_time=3.30,
        end_time=15.41,
    )
    res1 = ForcedAlignerService.split_oversized_sentences([s1], language="English", max_chars=160, max_words=24)
    assert [r.text for r in res1] == [
        "I can actually feel the toxins being pulled out of my skin.",
        "Well, this is a moisturizing mask.",
    ]

    s2 = SubtitleSentence(
        text="Or we could do something we'll all enjoy, like play a board game.",
        start_time=39.10,
        end_time=51.27,
    )
    res2 = ForcedAlignerService.split_oversized_sentences([s2], language="English", max_chars=160, max_words=24)
    assert len(res2) == 1
    assert res2[0].text == "Or we could do something we'll all enjoy, like play a board game."

    s3 = SubtitleSentence(
        text="Bird, Rebecca, I'd like to apologize for my insensitive comment earlier. Don't worry about it. It's fine. See, it was.",
        start_time=107.53,
        end_time=119.82,
    )
    res3 = ForcedAlignerService.split_oversized_sentences([s3], language="English", max_chars=160, max_words=24)
    assert [r.text for r in res3] == [
        "Bird, Rebecca, I'd like to apologize for my insensitive comment earlier.",
        "Don't worry about it.",
        "It's fine.",
        "See, it was.",
    ]


def test_hyphenated_compound_words_are_not_split():
    """HỒI QUY 2026-10-08: [SEG_BATCH] ngắt nhầm từ ghép nối bằng gạch nối.

    Ví dụ:
      'Before you go, consider this: not only do I have a deep-cycle marine battery power source...'
      bị chẻ thành "not only do I have a deep-" ⏐ "cycle marine battery power source"
      và "I also have all sixty-" ⏐ "one episodes..."
    """
    text = (
        "Before you go, consider this: not only do I have a deep-cycle marine battery power source, "
        "which is more than capable of running our entertainment system, "
        "I also have all sixty-one episodes of the BBC series Red Dwarf."
    )
    # Giả lập Aligner tokenize 'deep' và 'cycle' thành 2 token riêng biệt
    raw_tokens = [
        "Before", "you", "go,", "consider", "this:", "not", "only", "do", "I", "have", "a",
        "deep", "cycle", "marine", "battery", "power", "source,", "which", "is", "more", "than",
        "capable", "of", "running", "our", "entertainment", "system,", "I", "also", "have", "all",
        "sixty", "one", "episodes", "of", "the", "BBC", "series", "Red", "Dwarf."
    ]
    step = 20.0 / len(raw_tokens)
    items = [
        AlignedWord(text=w, start_time=round(i * step, 3), end_time=round((i + 1) * step, 3))
        for i, w in enumerate(raw_tokens)
    ]
    merged = ForcedAlignerService.merge_source_text(items, text)
    # Xác nhận merge_source_text đã gộp thành deep-cycle và sixty-one
    merged_texts = [m.text for m in merged]
    assert "deep-cycle" in merged_texts
    assert "sixty-one" in merged_texts
    assert not any(t == "deep-" for t in merged_texts)
    assert not any(t == "sixty-" for t in merged_texts)

    subs = ForcedAlignerService.group_words_to_subtitles(merged, language="English")
    sub_texts = [s.text for s in subs]
    assert not any(t.endswith("deep-") for t in sub_texts)
    assert not any(t.startswith("cycle") for t in sub_texts)
    assert not any(t.endswith("sixty-") for t in sub_texts)
    assert not any(t.startswith("one") for t in sub_texts)
    assert any("deep-cycle" in t for t in sub_texts)
    assert any("sixty-one" in t for t in sub_texts)

