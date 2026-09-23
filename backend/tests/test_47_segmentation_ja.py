"""Test tầng SEG (VAD > ASR > SEG) — tách câu tiếng Nhật theo dấu câu.

Tiêu chí nghiệm thu (theo yêu cầu): **tách câu đúng theo dấu câu trong `test.tsv`**
của Google FLEURS `ja_jp` (~60 giây đầu), tức bám dấu câu của chính bản tham chiếu
chứ không phải bám khoảng lặng VAD.

Bối cảnh quan trọng (đã khảo sát):
  * Qwen3-ASR trong `transcribe.cpp` KHÔNG có timestamp
    (`external/transcribe.cpp/src/arch/qwen3_asr/capabilities.cpp`: `TIMESTAMPS_NONE`)
    ⇒ không thể map dấu câu sang mốc thời gian ⇒ phải neo vào khoảng lặng VAD.
  * Dấu câu của ASR streaming ĐỔI theo từng scan ⇒ không được chốt chỉ vì "thấy dấu".
"""

from pathlib import Path

import pytest

from backend.segmentation import (
    SentenceCompleter,
    SilenceGap,
    StreamingSegmenter,
    choose_anchor,
    content_len,
    split_complete_sentences,
)
from backend.tests.fixtures import ja_fleurs


def _root(wav_test_dir: Path) -> Path:
    return wav_test_dir.parent


class _Clock:
    """Đồng hồ giả: mỗi nhịp quét cách nhau `step` giây (giống poll preview thật)."""

    def __init__(self, start: float = 1000.0, step: float = 0.4):
        self.t = start
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def _need_fleurs(wav_test_dir: Path) -> None:
    if not ja_fleurs.has_data(_root(wav_test_dir)):
        pytest.skip("Không có dữ liệu FLEURS ja_jp trong wav_test/")


# --------------------------------------------------------------- tiêu chí nghiệm thu
def test_tach_cau_dung_dau_cau_fleurs_60s(wav_test_dir):
    """~60 giây đầu FLEURS: SEG phải tách ĐÚNG từng câu theo dấu câu của `test.tsv`."""
    _need_fleurs(wav_test_dir)
    rows = ja_fleurs.pick_rows(_root(wav_test_dir), target_sec=60.0)
    assert rows, "không chọn được câu FLEURS nào"

    text = ja_fleurs.concat_text(rows)
    expected = ja_fleurs.reference_sentences(text)
    assert len(expected) >= 5, f"quá ít câu tham chiếu ({len(expected)}) — dữ liệu bất thường"

    # `max_chars` để rất cao: bài này kiểm tra việc tách theo DẤU CÂU của bản tham chiếu.
    # (Trần độ dài câu — bảo vệ khả năng đọc phụ đề — có test riêng
    # `test_cat_cuong_buc_khi_qua_dai`; bật nó ở đây sẽ cắt các câu FLEURS dài 40–70 ký tự.)
    completer = SentenceCompleter(max_chars=200)
    clk = _Clock()
    got = []
    for preview in ja_fleurs.stream_incremental(text, chunk=3):
        got.extend(d.text for d in completer.observe(preview, now=clk()))
    got.extend(d.text for d in completer.flush(text))

    assert got == expected, (
        "SEG tách câu lệch bản tham chiếu\n"
        f"  tham chiếu ({len(expected)}): {expected[:6]}...\n"
        f"  SEG       ({len(got)}): {got[:6]}..."
    )
    assert content_len("".join(got)) == content_len(text), "không được mất chữ nào"


def test_moc_audio_ranh_gioi_nam_trong_cua_so_scan(wav_test_dir):
    """Câu chốt phải kèm vùng audio hợp lệ và tăng đơn điệu theo thời gian."""
    _need_fleurs(wav_test_dir)
    rows = ja_fleurs.pick_rows(_root(wav_test_dir), target_sec=20.0)
    text = ja_fleurs.concat_text(rows)

    seg = StreamingSegmenter()
    out = []
    end_sample = 0
    for i, preview in enumerate(ja_fleurs.stream_incremental(text, chunk=3)):
        end_sample = (i + 1) * 3000
        out.extend(seg.observe(preview, end_sample, now=1000.0 + i * 0.3))
    out.extend(seg.flush(text, end_sample))

    assert out
    prev_cut = -1
    for item in out:
        lo, hi = item.window
        assert 0 <= lo <= hi, item
        assert item.cut_sample is not None and item.cut_sample >= 0
        assert item.cut_sample >= prev_cut, "mốc cắt phải tăng dần"
        prev_cut = item.cut_sample


def test_ghep_wav_1_phut_tao_duoc_file(wav_test_dir, tmp_path):
    """Fixture audio ~1 phút ghép từ FLEURS phải dựng được (nghe thử + đo bằng ASR)."""
    _need_fleurs(wav_test_dir)
    rows = ja_fleurs.pick_rows(_root(wav_test_dir), target_sec=60.0)
    out, pcm, spans = ja_fleurs.build_concat_wav(_root(wav_test_dir), rows,
                                                 out_path=tmp_path / "ja_1min.wav")
    assert out.exists() and out.stat().st_size > 0
    dur_sec = len(pcm) / ja_fleurs.SAMPLE_RATE
    assert 45.0 <= dur_sec <= 75.0, f"độ dài ghép bất thường: {dur_sec:.1f}s"
    assert len(spans) == len(rows)
    # Mỗi câu phải nằm sau câu trước (đã chèn khoảng lặng giữa các câu).
    assert all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))


# --------------------------------------------------------------- dấu câu đổi theo scan
def test_dau_cau_doi_theo_scan_khong_chot_som():
    """Ca thật của người dùng: `、` thành `。` rồi có câu mới ⇒ chốt ĐÚNG một lần.

    Chuỗi preview mô phỏng Qwen3-ASR (câu chữ lớn dần, dấu câu bị viết lại):
        "地殻が薄いため、" → "…ため、近い方には…あります。" → "…あります。溶岩が…"
    """
    c = SentenceCompleter()
    clk = _Clock(100.0)
    scans = [
        "地殻が薄いため、",
        "地殻が薄いため、近い方には",
        "地殻が薄いため、近い方には海が多くなることがあります。",
    ]
    out = []
    for text in scans:
        out.extend(c.observe(text, now=clk()))
    assert out == [], "chưa có câu kế tiếp ⇒ KHÔNG được chốt (dấu câu còn có thể đổi)"

    # Dấu `。` đứng ở cuối preview đủ lâu (≥ 2 scan và ≥ 350 ms) ⇒ được phép chốt.
    out.extend(c.observe(scans[-1], now=clk()))
    out.extend(c.observe(scans[-1], now=clk()))
    assert len(out) == 1
    assert out[0].text == "地殻が薄いため、近い方には海が多くなることがあります。"
    assert out[0].reason == "punct_stable"

    # Câu kế tiếp: dấu `。` ở CUỐI preview vẫn phải chờ (chưa có bằng chứng).
    full = "地殻が薄いため、近い方には海が多くなることがあります。溶岩が浮上しやすくなっていました。"
    assert c.observe(full, now=clk()) == []
    # Thấy chữ của câu MỚI nhưng phải đứng yên 2 nhịp mới chốt (chống cắt sớm).
    assert c.observe(full + "溶岩が流れ", now=clk()) == [], "nhịp đầu chưa đủ bằng chứng"
    out2 = c.observe(full + "溶岩が流れ", now=clk())
    assert len(out2) == 1
    assert out2[0].text == "溶岩が浮上しやすくなっていました。"
    assert out2[0].reason == "punct_tail"
    assert c.committed_text == full


def test_chong_cat_som_khi_asr_thay_dau_sai_giua_cau():
    """Log phim thật: `Don't try and trick.` rồi `Me into buying…` ⇒ KHÔNG được chẻ đôi.

    ASR thả dấu `.` sai rồi tự sửa ở nhịp sau. Luật mới: chờ tail đủ dài + đứng yên 2 nhịp.
    """
    c = SentenceCompleter()
    clk = _Clock(10.0)
    assert c.observe("Don't try and trick.", now=clk()) == []
    # Tail mới 2 ký tự ⇒ chưa đủ (`tail_min_chars=4`).
    assert c.observe("Don't try and trick. Me", now=clk()) == []
    # ASR tự sửa: bỏ dấu sai, câu liền lại ⇒ preview mới KHÔNG còn dấu ở giữa.
    healed = "Don't try and trick me into buying something I don't want."
    assert c.observe(healed, now=clk()) == []
    assert c.observe(healed + " Now let", now=clk()) == [], "nhịp đầu của tail"
    out = c.observe(healed + " Now let", now=clk())
    assert [d.text for d in out] == [healed], "phải chốt TRỌN câu, không chẻ ở 'trick.'"


def test_tail_latin_dinh_giua_tu_thi_chua_chot():
    """Tail kiểu `Super` (chưa có khoảng trắng) ⇒ ASR đang viết dở một từ ⇒ chờ."""
    c = SentenceCompleter()
    clk = _Clock()
    assert c.observe("Hello world. Super", now=clk()) == []
    assert c.observe("Hello world. Super", now=clk()) == []
    out = c.observe("Hello world. Superman is here", now=clk())
    assert [d.text for d in out] == ["Hello world."]


def test_chot_ngay_khi_co_cau_ke_tiep():
    """`punct_tail`: đủ chữ câu mới + đứng yên 2 nhịp ⇒ chốt."""
    c = SentenceCompleter()
    clk = _Clock()
    assert c.observe("今日はいい天気ですね。", now=clk()) == []
    assert c.observe("今日はいい天気ですね。明日は雨", now=clk()) == []
    out = c.observe("今日はいい天気ですね。明日は雨", now=clk())
    assert [d.text for d in out] == ["今日はいい天気ですね。"]
    assert out[0].reason == "punct_tail"


def test_dau_ngat_menh_de_khong_phai_dau_ket_cau():
    c = SentenceCompleter()
    out = c.observe("都市はモロッコのスルタンによって、貿易拠点を設立した。", now=1.0)
    assert out == [], "dấu `、` không được tách câu"


def test_cau_qua_ngan_thi_gop_voi_cau_sau():
    """`え。` (1 ký tự nội dung) không được tách riêng — gộp vào câu kế tiếp.

    (Đặt luật tail ở mức tối thiểu để chỉ kiểm tra hành vi GỘP câu ngắn.)
    """
    c = SentenceCompleter(min_chars=2, tail_min_chars=1, tail_scans=1, tail_stable_ms=0.0)
    assert c.observe("え。", now=1.0) == []
    out = c.observe("え。そうですか。はい。", now=1.1)
    assert [d.text for d in out] == ["え。そうですか。"]
    rest = c.flush("え。そうですか。はい。")
    assert [d.text for d in rest] == ["はい。"]


def test_khong_tach_o_so_thap_phan_va_viet_tat():
    assert split_complete_sentences("Pi is 3.14 exactly.") == ["Pi is 3.14 exactly."]
    assert split_complete_sentences("Hello world. Good bye.") == ["Hello world.", "Good bye."]
    c = SentenceCompleter()
    assert c.observe("バージョン 2.5 を使います。", now=1.0) == []


def test_cat_cuong_buc_khi_qua_dai():
    """Không dấu kết câu mà dài quá `max_chars` ⇒ cắt ở dấu ngắt mệnh đề gần nhất."""
    c = SentenceCompleter(max_chars=20)
    text = "あ" * 10 + "、" + "い" * 15
    out = c.observe(text, now=1.0)
    assert out and out[0].reason == "max_chars"
    assert out[0].text.endswith("、")


def test_khong_cat_cau_qua_it_tu_tranh_lap():
    """`Laughter.` (1 từ) KHÔNG được cắt.

    Log phim thật 2026-09-22: SEG cắt `Laughter.` → tầng commit gộp lại ("quá ngắn") ⇒ vùng
    audio KHÔNG tiến ⇒ nhịp sau cắt đúng chỗ đó — lặp 4 lần và gọi timer 4 lần. Không cắt
    thì không lặp.
    """
    c = SentenceCompleter(min_words=2)
    clk = _Clock()
    for _ in range(4):
        assert c.observe("Laughter.", now=clk()) == [], "câu 1 từ không được cắt"
    assert "hold=min_words" in c.trace_state()["hold"]

    # Khi câu sau tới, hai mảnh được GỘP thành một câu đủ từ.
    out = []
    for _ in range(3):
        out.extend(c.observe("Laughter. Okay, well, I don't know.", now=clk()))
    assert [d.text for d in out] == ["Laughter. Okay, well, I don't know."]


def test_min_words_bang_0_thi_cho_cat_cau_ngan():
    """`min_words=0` (người dùng tắt lọc ở popup) ⇒ SEG được phép cắt câu 1 từ."""
    c = SentenceCompleter(min_words=0)
    clk = _Clock()
    out = []
    for _ in range(3):
        out.extend(c.observe("Okay.", now=clk()))
    assert [d.text for d in out] == ["Okay."]


def test_flush_luon_chot_du_cau_ngan():
    """Hết vùng nói thì `flush` vẫn chốt (không mất chữ), kể cả câu 1 từ."""
    c = SentenceCompleter(min_words=2)
    assert c.observe("Laughter.", now=1.0) == []
    assert [d.text for d in c.flush("Laughter.")] == ["Laughter."]


def test_engine_lay_min_words_tu_config_sentence(restore_config):
    """Ngưỡng từ của SEG phải ĐI CHUNG với 'Min Words' ở popup (tránh hai tầng lệch nhau)."""
    from backend.asr.engine import TranscribeEngine
    from backend.config import config

    config.segmentation.enabled = True
    config.segmentation.use_whisper_timer = False
    config.sentence.min_words_to_commit = 5
    eng = TranscribeEngine(model_key=config.asr.active_model)
    assert eng._seg.completer.min_words == 5
    eng.update_sentence_config(min_words_to_commit=1)
    assert eng._seg.completer.min_words == 1


def test_mac_dinh_KHONG_cat_khi_chua_co_cau_moi():
    """Mặc định `allow_stable_cut=False`: dấu `。` ở cuối preview KHÔNG đủ để cắt.

    Log tiếng Nhật 2026-09-22 22:44: Qwen3 thả `。` sớm giữa câu (`営業回りを終え。`,
    `夕食を済ませ。`, `…とは。`) ⇒ nếu cắt theo độ ổn định thì câu bị chẻ. Nay phải chờ
    câu mới (hoặc im lặng VAD).
    """
    c = SentenceCompleter(allow_stable_cut=False)
    clk = _Clock()
    for _ in range(4):
        assert c.observe("あ、そうじゃん。", now=clk()) == [], "chưa có câu mới ⇒ chưa được cắt"
    assert "hold=stable_off" in c.trace_state()["hold"]

    # Khi câu mới xuất hiện (đủ nhịp) ⇒ cắt như thường.
    out = []
    for _ in range(2):
        out.extend(c.observe("あ、そうじゃん。次はこれ", now=clk()))
    assert [d.text for d in out] == ["あ、そうじゃん。"]


def test_bat_stable_cut_thi_cat_khong_can_cau_moi():
    """Bật `allow_stable_cut=True` (tuỳ chọn) ⇒ quay lại hành vi cắt-theo-độ-ổn-định."""
    c = SentenceCompleter(allow_stable_cut=True)
    clk = _Clock()
    out = []
    for _ in range(3):
        out.extend(c.observe("あ、そうじゃん。", now=clk()))
    assert [d.text for d in out] == ["あ、そうじゃん。"]


def test_flush_chot_phan_con_lai():
    """Hết vùng nói ⇒ `flush` chốt phần còn lại (đủ hay chưa đủ từ đều không mất chữ)."""
    c = SentenceCompleter()
    c.observe("まだ終わっていない", now=1.0)
    out = c.flush("まだ終わっていない")
    assert [d.text for d in out] == ["まだ終わっていない"]
    assert out[0].reason == "flush"


def test_asr_viet_lai_phan_da_chot_thi_khong_phat_lai():
    """ASR viết lại phần ĐÃ chốt (đo được với Qwen3 thật: `二つ` ↔ `2つ`).

    Hợp đồng: phần đã chốt ĐÓNG BĂNG — không phát lại câu cũ (sẽ sinh phụ đề trùng),
    nhưng phải báo `stale_text=True` để engine chạy lại ASR trên mảnh audio đã cắt.
    """
    c = SentenceCompleter()
    clk = _Clock(1.0)
    assert c.observe("最初の文です。次の文章", now=clk()) == []
    assert [d.text for d in c.observe("最初の文です。次の文章", now=clk())] == ["最初の文です。"]
    assert c.committed_text == "最初の文です。"
    out = c.observe("最初の文でした。次の文章", now=clk())
    assert c.resync_count == 1
    assert c.stale_text is True
    assert out == [], "không được phát lại câu đã chốt"
    assert c.committed_text == "最初の文です。", "tiền tố đã chốt phải đóng băng"

    # Viết lại trong phần ĐANG CHỜ (sau tiền tố đã chốt) thì KHÔNG cần resync:
    # câu đang chờ được tính lại từ đầu câu theo văn bản mới.
    c2 = SentenceCompleter()
    clk = _Clock()
    assert c2.observe("最初の文です。次の文章を読みます。あいうえ", now=clk()) == []
    assert [d.text for d in c2.observe("最初の文です。次の文章を読みます。あいうえ", now=clk())] \
        == ["最初の文です。"]
    assert c2.committed_text == "最初の文です。"
    assert c2.observe("最初の文です。次の文章が変わっても大丈夫。あいうえ", now=clk()) == []
    out2 = c2.observe("最初の文です。次の文章が変わっても大丈夫。あいうえ", now=clk())
    assert c2.resync_count == 0 and c2.stale_text is False
    assert [d.text for d in out2] == ["次の文章が変わっても大丈夫。"]


# --------------------------------------------------------------- neo audio (khoảng lặng)
def test_chon_neo_im_lang_dai_nhat_trong_cua_so():
    gaps = [
        SilenceGap(1000, 1500),      # 500 mẫu = 31 ms (quá ngắn)
        SilenceGap(60_000, 70_000),  # 10 000 mẫu = 625 ms (đủ dài)
        SilenceGap(90_000, 95_000),  # 5 000 mẫu = 312 ms
    ]
    cut, anchored, gap_ms = choose_anchor((50_000, 96_000), gaps, min_gap_ms=150)
    assert anchored is True
    assert 60_000 <= cut <= 70_000
    assert gap_ms > 600


def test_khong_co_im_lang_thi_lui_lai_de_chong_lan():
    """Không có khoảng lặng ⇒ mốc cắt phải LÙI để vùng mới bao gồm phần ranh giới.

    Đo bằng ASR thật: cắt đúng ở cuối vùng quét làm câu sau mất chữ đầu
    (`溶岩が浮上しやすく…` chỉ còn `しやすくなっていました。`).
    """
    cut, anchored, gap_ms = choose_anchor(
        (50_000, 60_000), [SilenceGap(10, 20)], min_gap_ms=150, fallback_overlap_ms=600
    )
    assert anchored is False
    assert gap_ms == 0.0
    # 600 ms = 9600 mẫu ⇒ cắt lùi về 50 400 (không vượt quá `lo`).
    assert cut == 60_000 - 9_600
    # Nếu cửa sổ hẹp hơn phần chồng lấn thì lấy `lo`.
    assert choose_anchor((59_500, 60_000), [], fallback_overlap_ms=600)[0] == 59_500
    # Tắt chồng lấn ( = 0 ) thì quay lại hành vi cũ.
    assert choose_anchor((50_000, 60_000), [], fallback_overlap_ms=0)[0] == 60_000


def test_cua_so_ranh_gioi_lay_tu_scan_cuoi_chua_co_cau_moi():
    """Vùng ranh giới phải bám scan CUỐI CÙNG chỉ chứa câu đã chốt."""
    seg = StreamingSegmenter()
    clk = _Clock(10.0)
    out = []
    for preview in ["最初の文です。", "最初の文です。次の文章", "最初の文です。次の文章"]:
        out.extend(seg.observe(preview, end_sample=32_000, now=clk()))
    assert len(out) == 1
    lo, hi = out[0].window
    assert lo <= 32_000 <= hi
    seg.note_silence(31_200, 32_400)
    out2 = seg.flush("最初の文です。次の文章", 48_000)
    assert out2  # flush vẫn hoạt động sau khi đã chốt


def test_turn_scorer_duoc_dung_khi_khong_co_dau_cau():
    """Hook cho model phát hiện kết thúc lượt (Namo…): điểm cao ⇒ chốt dù thiếu dấu câu."""
    calls = []

    def scorer(text: str):
        calls.append(text)
        return 0.9 if text.endswith("ます") else 0.1

    seg = StreamingSegmenter(turn_scorer=scorer, turn_threshold=0.5, turn_min_chars=4)
    out = seg.observe("今日はいい天気ですねます", 32_000, now=1.0)
    assert calls, "scorer phải được gọi khi không có dấu kết câu"
    assert out and out[0].reason == "turn_detector"

    seg2 = StreamingSegmenter(turn_scorer=lambda t: 0.1, turn_threshold=0.5, turn_min_chars=4)
    assert seg2.observe("今日はいい天気ですねます", 32_000, now=1.0) == []
