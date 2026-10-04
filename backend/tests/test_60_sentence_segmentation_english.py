import pytest
from backend.asr.forced_aligner import ForcedAlignerService, SubtitleSentence


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
