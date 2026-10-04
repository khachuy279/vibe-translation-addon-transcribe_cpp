# Phase 8 — Đơn giản hoá Pipeline B: cắt khối bằng VAD + tin tuyệt đối vào ASR (2026-10-04)

**Ngày**: 2026-10-04
**Audio kiểm chứng**: `wav_test/youtube/The_Big_Bang_Theory.mp3`

---

## 0. Yêu cầu thiết kế (từ người dùng)

1. **Đơn giản hoá `lookahead_handler.py`** — loại bỏ MỌI cách lọc chồng chéo phức tạp.
2. **Tích hợp VAD để cắt khối cho `[ASR]`**: đưa một đoạn audio (ví dụ 30 s) cho VAD, nó sẽ tìm
   thấy nhiều khoảng lặng > 1.5 s; **lấy khoảng lặng > 1.5 s ở CUỐI CÙNG** và gửi **toàn bộ audio
   từ đầu tới hết khoảng lặng đó** cho Lookahead Batch ASR.
3. **Tin `[ASR]` tuyệt đối**: ASR tạo ra câu như thế nào thì gửi ra phụ đề như vậy — không cắt xén,
   không bù đắp. Forced Aligner chỉ xác định thời điểm câu bắt đầu và kết thúc.
4. **Đơn giản hoá Extension**: nhận phụ đề gốc + dịch, hiển thị theo mốc Forced Aligner.

---

## 1. Vấn đề của kiến trúc cũ (vì sao phải xoá, không phải vá)

Bộ cắt khối cũ chọn điểm cắt bằng **năng lượng RMS tương đối**, rồi cho khối sau **LÙI LẠI**
`overlap_sec = 1.0 s` "cho chắc". Khoản lùi đó sinh ra vùng chồng lấn, và phía nhận phải bù bằng ba
tầng xử lý chồng lên nhau:

| Tầng | Việc nó làm | Lỗi nó gây ra (log thật) |
|---|---|---|
| `_trim_batch_leading_overlap` | Trừ từ đầu khối theo **mốc** đã phát + theo chuỗi từ | Cắt mất từ ĐẦU CÂU: `'Or'`, `'I'`, và **7 từ** của `'Let me sit on the stairs and think about what I did.'` → `'think about what I did.'` |
| `_batch_carry_words` + `_hold_back_unfinished_tail` | Giữ mảnh cuối chưa kết câu, ghép vào đầu khối sau | **Dán liền hai câu của hai khối khác nhau**: xem §2 |
| `calibrate_word_times` | Trải lại đoạn từ bị aligner bỏ mất/dồn thời gian | Sửa nhầm đối tượng — xem §3 |

Cả ba chỉ tồn tại để **bù cho việc cắt khối ở giữa câu**. Khi mép khối rơi vào giữa khoảng lặng
thật thì cả ba trở nên vô nghĩa — và chính chúng sinh lỗi mới.

---

## 2. Bằng chứng: tầng "hàn gắn" dán hai câu khác khối vào nhau

Log thật của phiên bản có hiệu chỉnh mốc (người dùng cung cấp):

```
[ASR] Lookahead Batch ASR [62.04s -> 74.27s]: 'And hold you tight. So happy together.
      I can't see me loving nobody but you for all my'
[ASR] Giữ lại mảnh cuối CHƯA kết câu (11 từ, ranh giới khối 74.27s): 'I can't see me loving nobody but you for all my'
[ASR] Lookahead Batch ASR [74.27s -> 86.37s]: 'This is Rebecca. Hi. She's younger.'
[ASR] Ghép 11 từ giữ lại từ khối trước vào khối [74.27s -> 86.37s].
[ASR] Ngắt câu khối [74.27s -> 86.37s] (4 phụ đề): "I can't see me" |
      "loving nobody but you for all my This is Rebecca." | "Hi." | "She's younger."
```

**Phụ đề hiện trên màn hình (ảnh người dùng gửi)**:
`"loving nobody but you for all my This is Rebecca."` — hai câu của hai cảnh khác nhau bị dán liền.

Tầng hàn gắn còn làm hỏng **cả câu ở giữa**: `"I can't see me"` trở thành một phụ đề riêng, còn
`"loving nobody but you for all my life."` thì mất chữ `"life."` (rơi vào khối sau).

---

## 3. Vì sao hiệu chỉnh mốc từ cũng bị xoá

`Qwen3-ForcedAligner` là model **NAR**: phát đúng một token mốc cho mỗi token văn bản, **không có cơ
chế từ chối**. Khi audio không khớp văn bản, nó dồn toàn bộ thời gian vào MỘT token:

```
khối [66.30 → 78.30]s   Aligner:  'I' 67.420 → 76.540 (9.12 s)
                                  "can't"…"life." mỗi từ 0.000 s
```

Phép hiệu chỉnh đã được viết và đo (trải lại đoạn hỏng theo tỉ lệ ký tự). Nhưng:

* **Đo thật bằng ASR trên chính đoạn audio đó** (`probe_segment_asr --start 3.0 --end 8.6` trả về
  `'I.'`) chứng minh aligner **trung thành với những gì ASR nghe** — không phải aligner sai.
* Hiệu chỉnh không sửa được gốc, chỉ che triệu chứng, và nó **thay đổi mốc của những từ đang đúng**.
* Người dùng yêu cầu rõ: **tin ASR tuyệt đối**, Forced Aligner chỉ xác định mốc đầu/cuối.

⇒ Toàn bộ `backend/core/speech_activity.py` + test của nó đã bị **XOÁ**. Mốc của Forced Aligner được
dùng **nguyên vẹn**.

---

## 4. Kiến trúc mới

### 4.1. `backend/core/vad_silence.py` — bộ dò khoảng lặng bằng VAD Silero

**Vì sao Silero (đo thật trên máy chuẩn, RTX 5060 Ti):**

| Engine | Cách chạy | 90 s audio | Hệ số |
|---|---|---|---|
| `firered-vad` | `is_speech()` từng frame 10 ms | 28 700 ms | **0.31× thời gian thực** |
| `silero-vad` | từng cửa sổ 512 mẫu | 1 001 ms | 90× |
| `silero-vad` | ~~theo lô 256 cửa sổ~~ | 127 ms | **KHÔNG dùng được** — xem cảnh báo dưới |

> ⚠️ **Bẫy đã đo và ghi lại**: gọi Silero theo **lô > 1 làm kết quả SAI HẲN**. Silero giữ state nội bộ
> chạy tiếp giữa các cửa sổ (`_state`, `_context`); khi batch > 1, state `[2, batch, 128]` được xử lý
> theo cách khác. Đoạn lặng dài nhất trên 60 s audio đầu, theo batch:
> `batch=1 → 5.50 s` (khớp streaming thật) ⏐ `batch=8 → 1.41 s` (sai) ⏐ `batch=64 → 0.90 s` (sai)
> ⏐ `batch=256 → 6.27 s` ⏐ `batch=1024 → 6.82 s`.
> Vì vậy `SILERO_BATCH_WINDOWS = 1` là **ràng buộc cứng**, không phải lựa chọn hiệu năng.

**Ngưỡng** `batch_vad_silence_threshold = 0.30` (thấp hơn 0.5 mặc định của thư viện, CÓ CHỦ Ý):
Silero coi tiếng **CƯỜI** và lời **HÁT** là "không có tiếng nói" (đo thật: p50 ≈ 0.06 ở đoạn cười
3.8–8.5 s; p50 ≈ 0.03 ở đoạn nhạc 65–83 s). Ở ngưỡng 0.5, cả đoạn nhạc 18 s bị coi là "khoảng lặng"
⇒ cắt khối vào đó sẽ chẻ đôi lời hát.

**Cache theo lô**: xác suất đã tính được giữ lại theo từng lô `(t0, hop, probs)`; lần quét sau nối
tiếp phần đã có nên chi phí trung bình chỉ còn phần audio MỚI.

### 4.2. `LookaheadChunker._find_vad_cut` — quy tắc cắt

```
vùng tìm kiếm = [from_pts + min_window_sec, from_pts + search_max_sec]
gap           = khoảng lặng CUỐI CÙNG (xa nhất) có độ dài >= 1500 ms mà VAD xác nhận
cut_pts       = GIỮA khoảng lặng đó
audio gửi ASR = TRỌN [from_pts, cut_pts]  (bao gồm cả khoảng lặng ở cuối)
next_read_pts = cut_pts                    (KHÔNG lùi — overlap_sec = 0.0)
```

* **"CUỐI CÙNG" chứ không phải "dài nhất"**: khối ASR càng dài càng tốt cho ngữ cảnh, nên đẩy mép
  khối xa nhất có thể mà vẫn dừng ở khoảng lặng chắc chắn.
* **Ngưỡng 1.5 s là NGƯỠNG CỨNG**: không nới xuống khoảng lặng ngắn hơn, vì cắt vào chỗ VAD chưa
  chắc chắn chính là loại lỗi cơ chế này sinh ra để tránh.
* **Dự phòng**: nếu không có khoảng lặng nào đạt ngưỡng trong vùng, dùng lại đường dò năng lượng
  (dải tìm kiếm đã bị chặn trên nên không cắt bừa giữa câu).

### 4.3. Đã XOÁ khỏi `lookahead_handler.py`

* `_trim_batch_leading_overlap`, `_trim_repeated_block_prefix`, `_batch_trim_overlap`;
* `_hold_back_unfinished_tail`, `_batch_carry_words`, bước "ghép 11 từ giữ lại";
* `calibrate_word_times` và module `backend/core/speech_activity.py`;
* cấu hình `batch_trim_boundary_overlap`, `batch_overlap_sec` (giữ lại = `0.0`).

Luồng còn lại của `_offline_batch_loop`: **cắt khối (VAD) → ASR → Forced Aligner → gom câu → dịch →
gửi**. Không cắt xén, không bù đắp, không hàn gắn.

### 4.4. Extension

`extension_firefox/lib/lookahead-timeline.js` (và bản `extension_chrome_edge` giữ nguyên bản):
client **không còn tự giãn/kẹp cửa sổ hiển thị**. Trước đây:

```js
maxAllowed = max(start + 0.5, next.start - 0.05)
end        = max(end, min(start + minDuration, maxAllowed))
```

Khi `next.start < start + 0.55` (hai khối chồng mốc), câu trước bị co còn 0.5 s và cửa sổ hai câu
chồng nhau ⇒ câu bắt đầu sau có thể bị câu cũ che (đúng lỗi "backend gửi mà không hiện").

Nay client chỉ:
1. Nhận `[start_pts, end_pts]` do backend gửi và vẽ câu nào có `start_pts <= currentTime < end_pts`.
2. **Sàn cứng duy nhất**: câu dài < `minDurationSec` (0.6 s) được **kéo dài** — chỉ kéo dài, không
   bao giờ cắt ngắn, không vượt mốc bắt đầu câu kế tiếp. Đây là chống nhấp nháy một khung hình.
3. Ưu tiên câu có `start_pts` muộn nhất khi có chồng lấn (lưới an toàn).

---

## 5. Kết quả đo lại (mô phỏng đầu-cuối với ASR thật)

```
[ASR] Lookahead Batch ASR [0.00s -> 15.42s]  ...  "Okay, how is that?" ⏐
      "I can actually feel the toxins being pulled out of my skin." ⏐ "Well, this is a moisturizing mask."
[ASR] Lookahead Batch ASR [68.36s -> 80.36s] ...  "I can't see loving nobody but you for all my life." ⏐
      "When you're with me, baby, the sky's the limit."
```

* **0/23 câu bị mất** (trước đây mất `'Or'`, `'I'`, `'This'`, và 7 từ của một câu).
* **Không còn câu nào bị dán liền** — hết hẳn dạng `"…for all my This is Rebecca."`.
* Câu ở ranh giới khối **trọn vẹn**: `"Or we could do something we'll all enjoy, like play a board
  game."` ⏐ `"I can't see loving nobody but you for all my life."`.
* Chi phí VAD: **128–215 ms cho mỗi khối 12–16 s** (Silero, batch = 1 cửa sổ).

---

## 6. Hạn chế còn lại (ghi rõ để không kỳ vọng sai)

1. **Mốc của Forced Aligner vẫn có thể lệch khi audio không khớp văn bản.** Ví dụ đo được: phụ đề
   `"I can actually feel the toxins being pulled out of my skin."` có mốc `3.04 → 12.00` (8.96 s) vì
   aligner kéo dài từ `'I'` suốt đoạn đó. Kiểm chứng độc lập bằng ASR trên chính đoạn audio ấy trả về
   `'I.'` ⇒ model thật sự "nghe" ra chữ ở đó; **đây là hành vi của model, không phải bug ở tầng mốc**.
   Theo yêu cầu thiết kế, tầng mốc không can thiệp.
2. **Khi bộ dò VAD không tìm thấy khoảng lặng ≥ 1.5 s** trong dải tìm kiếm, khối rơi về đường dò năng
   lượng (cắt ở điểm trũng). Điểm trũng có thể nằm giữa từ ⇒ từ ở mép khối có thể bị cắt đôi ở tầng
   âm học. Không sinh ra từ lặp (vì `overlap_sec = 0`), nhưng đó là đánh đổi đã biết.
3. **Cắt ở GIỮA khoảng lặng** (không phải mép) để chắc chắn không dính tiếng nói ở cả hai phía.

---

## 7. Công cụ chẩn đoán đi kèm

| Công cụ | Việc nó làm |
|---|---|
| `backend/tools/probe_vad_silence.py` | In khoảng lặng VAD xác nhận + chi phí quét + khoảng lặng CUỐI CÙNG trong các cửa sổ cắt khối điển hình. |
| `backend/tools/diag_pipeline_b.py` | Chạy đúng chuỗi cắt khối → ASR → Aligner → gom câu trên audio thật, in mốc từng từ. |
| `backend/tools/sim_pipeline_b.py` | Mô phỏng **đầu-cuối** (nạp đệm theo playhead + chạy thật `_offline_batch_loop`) rồi kiểm tra từng phụ đề bằng cách mô phỏng `SubtitleTimelineQueue`; in ra câu nào **không bao giờ hiện**. |
| `backend/tools/probe_levels.py` | Đo RMS theo khung để đối chiếu vùng có tiếng. |
| `backend/tools/probe_segment_asr.py` | Hỏi thẳng ASR một đoạn audio ngắn (dùng để phân xử "chỗ này có tiếng nói thật hay không"). |
| `backend/tests/js/demo_client_sync.js` | Nạp **chính** `extension_firefox/lib/lookahead-timeline.js` trong Node và in ra câu nào không bao giờ được vẽ. |

---

## 8. Kiểm thử

| Test | Nội dung |
|---|---|
| `test_58_lookahead_offline_batch.py::test_vad_silence_cut_replaces_boundary_patching` | Các thuộc tính/thuật toán "sửa chữa ranh giới" phải **không còn tồn tại**; chunker dùng bộ dò VAD; `overlap_sec == 0`; ngưỡng VAD ≥ 1000 ms. |
| `test_56_lookahead_chunker.py` | Hợp đồng cắt khối (đường dự phòng năng lượng + `next_read_pts`). |
| `extension_firefox/tests/lookahead-timeline.test.js` | Không câu nào bị câu sau nuốt mất; câu `start_pts` muộn hơn được ưu tiên. |

Kết quả: `pytest` (suite nhanh) **toàn bộ pass**; test JS Extension **14/14 pass**.
