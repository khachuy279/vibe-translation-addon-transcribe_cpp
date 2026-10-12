import pytest
from backend.asr.forced_aligner import AlignedWord, ForcedAlignerService, SubtitleSentence


# ─────────────── LỚP NGẮT CÂU BẰNG KHOẢNG LẶNG (VAD) — log thật 2026-10-11 14:51 ──────────

#: Câu THẬT trong log: ASR trả một câu dài KHÔNG dấu câu, gộp lời 2–3 người.
_PAUSE_CASE_TEXT = (
    "じゃステージランドよよし次こそ勝つぞあらあら楽しそうね"
    "これいただき物だけどよかったらどうぞうまそう"
)

#: Cách aligner tokenize câu trên (mốc từ giây, khe ngắt hơi chèn trước 'あらあら' và 'うまそう').
_PAUSE_CASE_WORDS = [
    "じゃ", "ステージ", "ランド", "よ", "よし", "次こそ", "勝つぞ",
    "あらあら", "楽しそう", "ね", "これ", "いただき物", "だけど", "よかったら", "どうぞ",
    "うまそう",
]
#: Chỉ số từ BẮT ĐẦU một lượt lời mới ⇒ chèn khe 0.6 s ngay trước nó.
_PAUSE_AFTER_INDEXES = (7, 15)
_PAUSE_GAP_SEC = 0.6
_WORD_SEC = 0.25


def _pause_case_sentence() -> SubtitleSentence:
    words = []
    t = 0.0
    for i, w in enumerate(_PAUSE_CASE_WORDS):
        start = t + (_PAUSE_GAP_SEC if i in _PAUSE_AFTER_INDEXES else 0.0)
        words.append(AlignedWord(text=w, start_time=start, end_time=start + _WORD_SEC))
        t = start + _WORD_SEC
    return SubtitleSentence(
        text=_PAUSE_CASE_TEXT, start_time=0.0, end_time=words[-1].end_time, words=words
    )


def test_ngat_cau_bang_khoang_lang_vad_tach_cau_dai_tieng_nhat():
    """Ca thật: một câu dài không dấu câu ⇒ VAD xác nhận 2 khe ⇒ tách thành 3 phụ đề."""
    sub = _pause_case_sentence()
    # Hai khoảng lặng VAD xác nhận, ĐÚNG hệ thời gian của `words` (tính từ đầu khối).
    silences = [(1.6, 2.4), (4.6, 5.4)]

    out = ForcedAlignerService.split_subtitles_by_pause(
        [sub], silences=silences, min_pause_ms=350, min_tokens=3, language="Japanese"
    )

    assert [s.text for s in out] == [
        "じゃステージランドよよし次こそ勝つぞ",
        "あらあら楽しそうねこれいただき物だけどよかったらどうぞ",
        "うまそう",
    ]
    # BẤT BIẾN: không được mất hay thêm ký tự nào so với câu gốc.
    assert "".join(s.text for s in out) == _PAUSE_CASE_TEXT
    # Mốc thời gian lấy CHÍNH XÁC từ mốc từ (không chia theo tỉ lệ độ dài văn bản).
    assert out[0].start_time == pytest.approx(0.0)
    assert out[0].end_time == pytest.approx(1.75)
    assert out[1].end_time == pytest.approx(4.35)
    assert out[2].end_time == pytest.approx(5.2)
    # Mỗi mảnh vẫn giữ `words` của mình (để lớp sau / diarization còn dùng).
    assert all(s.words for s in out)


def test_vad_khong_xac_nhan_thi_khong_ngat():
    """VAD chạy nhưng KHÔNG thấy khoảng lặng nào ⇒ giữ nguyên câu (VAD là bên xác nhận)."""
    sub = _pause_case_sentence()
    out = ForcedAlignerService.split_subtitles_by_pause(
        [sub], silences=[], min_pause_ms=350, min_tokens=3, language="Japanese"
    )
    assert len(out) == 1
    assert out[0].text == _PAUSE_CASE_TEXT


def test_vad_tat_thi_van_ngat_theo_khe_tu():
    """`silences=None` (VAD tắt/lỗi) ⇒ dùng khe từ của Forced Aligner; vẫn tốt hơn cắt cứng."""
    sub = _pause_case_sentence()
    out = ForcedAlignerService.split_subtitles_by_pause(
        [sub], silences=None, min_pause_ms=350, min_tokens=3, language="Japanese"
    )
    assert len(out) == 3
    assert "".join(s.text for s in out) == _PAUSE_CASE_TEXT


def test_khong_tao_manh_cut_duoi_nguong_token():
    """Khe nằm quá sớm/quá muộn (mảnh < `min_tokens`) ⇒ BỎ QUA điểm tách, không tạo mảnh cụt.

    Lưu ý: với tiếng Nhật, 1 "token" = 1 ký tự (xem `_count_language_tokens`), nên 'じゃ' = 2
    token còn 'うまそう' = 4 token.
    """
    sub = _pause_case_sentence()
    # Khe ngay sau từ ĐẦU TIÊN: mảnh đầu 'じゃ' = 2 token < 3 ⇒ phải bỏ qua.
    early = ForcedAlignerService.split_subtitles_by_pause(
        [sub], silences=[(0.2, 0.9)], min_pause_ms=350, min_tokens=3, language="Japanese"
    )
    assert len(early) == 1

    # Khe ngay trước từ CUỐI: mảnh cuối 'うまそう' = 4 token ⇒ với min_tokens=5 cũng phải bỏ qua.
    late = ForcedAlignerService.split_subtitles_by_pause(
        [sub], silences=[(4.3, 4.9)], min_pause_ms=350, min_tokens=5, language="Japanese"
    )
    assert len(late) == 1


def test_khong_ngat_khi_khe_ngan_hon_nguong():
    """Khe 0.6 s > 350 ms mới đủ; hạ `min_pause_ms` lên cao hơn khe ⇒ không tách."""
    sub = _pause_case_sentence()
    out = ForcedAlignerService.split_subtitles_by_pause(
        [sub], silences=None, min_pause_ms=900, min_tokens=3, language="Japanese"
    )
    assert len(out) == 1


def test_cau_khong_co_moc_tu_giu_nguyen():
    """Câu đã qua `split_oversized_sentences` (mất `words`) ⇒ lớp này không đụng vào."""
    sub = SubtitleSentence(text=_PAUSE_CASE_TEXT, start_time=0.0, end_time=10.0)
    assert len(ForcedAlignerService.split_subtitles_by_pause([sub], silences=None)) == 1


def test_tieng_anh_noi_manh_bang_dau_cach():
    """Tiếng Anh: mảnh nối bằng dấu cách (khác CJK nối liền)."""
    words = [
        AlignedWord("I", 0.0, 0.2), AlignedWord("won", 0.2, 0.5), AlignedWord("again", 0.5, 0.9),
        AlignedWord("you", 1.6, 1.8), AlignedWord("lost", 1.8, 2.1), AlignedWord("fair", 2.1, 2.5),
        AlignedWord("and", 2.5, 2.7), AlignedWord("square", 2.7, 3.2),
    ]
    sub = SubtitleSentence(
        text="I won again you lost fair and square", start_time=0.0, end_time=3.2, words=words
    )
    out = ForcedAlignerService.split_subtitles_by_pause(
        [sub], silences=None, min_pause_ms=350, min_tokens=2, language="English"
    )
    assert [s.text for s in out] == ["I won again", "you lost fair and square"]


# ─────────────────────────────────────────────── các ca hồi quy cũ


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


# ── Hồi quy 2026-10-12: Lớp 3 không chẻ vụn câu có dấu câu ở giữa hoặc câu ngắn ──────────


def test_pause_split_khong_che_vun_cau_co_dau_cau_o_giua_hoac_ngan():
    """HỒI QUY 2026-10-12: ASR trả về câu bình thường có dấu câu ở giữa hoặc câu ngắn
    ⇒ Lớp 3 (split_subtitles_by_pause) KHÔNG ĐƯỢC chẻ vụn.

    Ca thật trong log người dùng:
      'あ、本当申し訳ない。ま、私は妻があれば別に大丈夫ですよ。'
      trước Lớp 3: 2 câu đẹp
      Lớp 3 cũ: chẻ thành 5 mảnh vụn:
        "あ、本当" ⏐ "申し訳ない。" ⏐ "ま、私は" ⏐ "妻があれば" ⏐ "別に大丈夫ですよ。"
    """
    # Câu 1: "あ、本当申し訳ない。" (10 ký tự, có dấu '、' ở giữa)
    w1 = [
        AlignedWord("あ、", 0.0, 0.3),
        AlignedWord("本当", 0.3, 0.8),
        # Khe 0.5 s giữa '本当' và '申し訳ない。'
        AlignedWord("申し訳ない。", 1.3, 2.0),
    ]
    sub1 = SubtitleSentence(text="あ、本当申し訳ない。", start_time=0.0, end_time=2.0, words=w1)

    # Câu 2: "ま、私は妻があれば別に大丈夫ですよ。" (18 ký tự, có dấu '、' ở giữa)
    w2 = [
        AlignedWord("ま、", 2.1, 2.4),
        AlignedWord("私は", 2.4, 2.8),
        # Khe 0.45 s
        AlignedWord("妻があれば", 3.25, 3.8),
        # Khe 0.5 s
        AlignedWord("別に大丈夫ですよ。", 4.3, 5.2),
    ]
    sub2 = SubtitleSentence(
        text="ま、私は妻があれば別に大丈夫ですよ。", start_time=2.1, end_time=5.2, words=w2
    )

    stats: dict = {}
    out = ForcedAlignerService.split_subtitles_by_pause(
        [sub1, sub2],
        silences=None,
        min_pause_ms=350.0,
        language="Japanese",
        stats=stats,
    )

    # Giữ NGUYÊN 2 câu, KHÔNG chẻ thành 5 mảnh!
    assert len(out) == 2, f"Bị chẻ thành {len(out)} phụ đề: {[s.text for s in out]}"
    assert [s.text for s in out] == [
        "あ、本当申し訳ない。",
        "ま、私は妻があれば別に大丈夫ですよ。",
    ]
    assert stats["accepted"] == 0


def test_pause_split_khong_che_cau_ngan_khong_dau_cau():
    """Câu ngắn (< min_chars = 20) dù KHÔNG có dấu câu cũng KHÔNG được chẻ vụn."""
    words = [
        AlignedWord("妻があれば", 0.0, 0.5),
        # Khe 0.6 s
        AlignedWord("大丈夫ですよ", 1.1, 1.8),
    ]
    sub = SubtitleSentence(text="妻があれば大丈夫ですよ", start_time=0.0, end_time=1.8, words=words)
    out = ForcedAlignerService.split_subtitles_by_pause(
        [sub], silences=None, min_pause_ms=350.0, min_chars=20, language="Japanese"
    )
    assert len(out) == 1
    assert out[0].text == "妻があれば大丈夫ですよ"


def test_pause_split_khong_che_cau_tieng_anh_co_dau_phay_o_giua():
    """Tiếng Anh: câu có dấu phẩy ở giữa (Well, this is fine) ⇒ KHÔNG chẻ bằng pause."""
    words = [
        AlignedWord("Well,", 0.0, 0.4),
        # Khe 0.6 s
        AlignedWord("this", 1.0, 1.3),
        AlignedWord("is", 1.3, 1.5),
        AlignedWord("fine.", 1.5, 1.9),
    ]
    sub = SubtitleSentence(text="Well, this is fine.", start_time=0.0, end_time=1.9, words=words)
    out = ForcedAlignerService.split_subtitles_by_pause(
        [sub], silences=None, min_pause_ms=350.0, min_words=6, language="English"
    )
    assert len(out) == 1
    assert out[0].text == "Well, this is fine."


