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
vùng quét VAD  = [from_pts, from_pts + min(available_sec, max_audio_sec)]
vùng tìm kiếm  = [from_pts + min_window_sec, from_pts + search_max_sec]
gap            = khoảng lặng >= 1500 ms mà VAD xác nhận, GẦN mốc lý tưởng nhất
cut_pts        = GIỮA khoảng lặng đó
audio gửi ASR  = TRỌN [from_pts, cut_pts]  (bao gồm cả khoảng lặng ở cuối)
next_read_pts  = cut_pts                    (KHÔNG lùi — overlap_sec = 0.0)
```

* **"GẦN MỐC LÝ TƯỞNG NHẤT"** chứ không phải "khoảng lặng cuối cùng": bản đầu dùng quy tắc "cuối
  cùng" và **log thật phiên `bcbd0d7c` cho thấy nó hỏng** — khối chỉ **8–13 s** trong khi mục tiêu
  25–28 s, dù đệm trước tới 36–50 s:

  ```
  [ASR] Lookahead Batch ASR [28.81s -> 39.10s]     (10.3 s)
  [ASR] Lookahead Batch ASR [39.10s -> 47.82s]     ( 8.7 s)
  [ASR] Lookahead Batch ASR [47.82s -> 56.07s]     ( 8.3 s)
  ```

  Nguyên nhân: trong dải tìm kiếm có nhiều khoảng lặng ngắn (khe giữa hai câu, tiếng cười) và
  khoảng "cuối cùng" lại rơi SỚM HƠN mốc lý tưởng. Nay chọn khoảng lặng **bám sát mốc lý tưởng
  nhất** (hoà thì ưu tiên khoảng dài hơn). Vẫn đúng tinh thần yêu cầu: khoảng lặng > 1.5 s, và gửi
  TRỌN `[from_pts, cut_pts]` (bao gồm khoảng lặng) cho ASR.
* **Quét trọn phần audio đã có** (tới `from_pts + max_audio_sec`) rồi mới CHỌN điểm trong
  `[search_start, search_end]`: nếu chỉ quét tới `search_end` thì khoảng lặng vắt qua mép dải bị
  cắt cụt và bị loại oan.
* **Ngưỡng 1.5 s là NGƯỠNG CỨNG**: không nới xuống khoảng lặng ngắn hơn, vì cắt vào chỗ VAD chưa
  chắc chắn chính là loại lỗi cơ chế này sinh ra để tránh.
* **Dự phòng**: nếu không có khoảng lặng nào đạt ngưỡng trong vùng, dùng lại đường dò năng lượng
  (dải tìm kiếm đã bị chặn trên nên không cắt bừa giữa câu).

Sau khi sửa, đo lại trên audio thật (`diag_pipeline_b`): khối **28.9 / 28.6 / 17.1 / 22.0 / 13.4 /
27.0 / 27.5 / 22.4 s**. Các khối ngắn hơn mục tiêu đều có lý do rõ: khoảng lặng đủ dài gần nhất nằm
sớm (17–22 s), hoặc khoảng lặng kéo dài tới 17.9 s nên `ideal+2` rơi vào giữa nó.

**Chẩn đoán**: mỗi khối log thêm một dòng `[SEG_BATCH] Chọn khối: dài Xs (mục tiêu …, dải tìm tới
…, mốc cắt …, biên liên tục tới …, RAM …, đệm giải mã trước …, playhead …, mode …)`.

### 4.2d. SỰ CỐ 2026-10-04 (phiên `7afc26e3`): khối 8 s dù log ghi "bộ đệm trước 43.3 s"

Log người dùng:

```
[SEG_BATCH] Chọn khối: dài  8.1s (mục tiêu  6.3s, dải tìm tới  8.3s, bộ đệm trước 43.3s, mode=forced_overlap)
[SEG_BATCH] Chọn khối: dài  8.1s (mục tiêu  6.3s, dải tìm tới  8.3s, bộ đệm trước 44.7s, mode=forced_overlap)
```

`dải tìm tới 8.3s` = `available_sec`, tức `buffered_end_from(from_pts) - from_pts ≈ 8.3 s` — **mâu
thuẫn** với "bộ đệm trước 43.3 s" (con số đó lấy từ `_decoded_end_pts`, không phải từ frontier
liên tục của timeline).

**Nguyên nhân #1 (đã vá)**: `ContinuousAudioTimeline.buffered_end_from` cắt vùng tại khe hở
> `REPORT_EPS_SEC` = **0.15 s**. Bộ giải mã MSE tăng dần sinh ra các khe ~0.2 s, nên **chỉ một khe
0.2 s là đủ để hàm báo "chỉ có 1 s audio phía trước" trong khi RAM giữ 30 s**. Đo lại:

```
30 mảnh 1 s cách nhau 0.2 s  →  buffered_end_from(0) = 1.0 s   (RAM: 30.0 s)   ← TRƯỚC
                             →  buffered_end_from(0) = 35.8 s  (phủ hết phần đã lưu) ← SAU
```

Trong khi đó `get_audio_range`/`read` **đã lấp** các khe nhỏ bằng silence (tới `MAX_GAP_FILL_SEC`
= 5 s) ⇒ vùng "ĐỌC ĐƯỢC" rộng hơn hẳn vùng "liền mạch byte". Nay `buffered_end_from` dùng đúng
`MAX_GAP_FILL_SEC`, còn khe LỚN (lỗ tua thật) vẫn cắt vùng như cũ.

**Nguyên nhân #2 (đã vá — phiên `3f4a90ec`)**: sau khi frontier được vá, log mới cho thấy frontier
KHÔNG còn là nút cổ chai, và thủ phạm thật lộ ra:

```
[SEG_BATCH] Chọn khối: dài 11.9s (mục tiêu 12.2s, dải tìm tới 14.2s,
            biên liên tục tới 69.99s, RAM 67.6s, đệm giải mã trước 52.2s, playhead 18.3s, ...)
```

`biên liên tục tới 69.99s` và `RAM 67.6s` — bộ đệm thừa sức cho khối dài hơn, nhưng **dải tìm chỉ
tới 14.2 s**. Đó là do chính tôi: biến `effective_search_max` bị kẹp theo `adaptive_max_sec`
(30 s) rồi theo `budget`. VAD chỉ được nhìn thấy khoảng lặng đầu tiên (~12 s) nên cắt ở đó.

**Đã sửa**: cỡ khối nay là **"TỐI ĐA CÓ THỂ"**, chỉ bị chặn bởi ĐÚNG HAI thứ:

```python
budget             = min(available_sec, max_audio_sec)   # (1) đệm đọc được, (2) trần token DLL
effective_search_max = max(min_window_sec, budget)       # KHÔNG còn trần nhân tạo 30 s
effective_target     = min(max_audio_sec, max(target_window_sec, budget - 2))
```

Đo lại trên audio thật sau khi sửa: khối **31.5 / 39.6 / 38.9 / 42.3 / 44.0 / 44.2 / 41.5 s** — chạm
trần 45 s, đúng "gửi đoạn âm thanh tối đa có thể vào ASR".

Kiểm chứng ASR không bị nuốt chữ ở cỡ này: `truncated=False` ở 30 / 40 / 44 s, và nội dung khớp
đúng lời thoại (44 s → 96 từ, 497 ký tự, kết thúc trọn câu).

**Đánh đổi cần biết** (khối dài hơn không miễn phí):

| | Khối 8–15 s | Khối 31–44 s |
|---|---|---|
| Ngữ cảnh ASR | ngắn | dài (CER/WER tốt hơn) |
| Số ranh giới khối | nhiều | ít (ít rủi ro mép khối) |
| Thời gian dịch 1 khối | ~0.3–0.9 s | **~2.9–3.8 s**, có lần `Parse JSON batch thất bại` phải fallback dịch tuần tự |
| Độ trễ phụ đề cho đoạn mới | thấp | cao hơn (phải chờ ASR xong khối dài) |

Nếu thấy dịch quá chậm hoặc phụ đề tới muộn, hạ `batch_max_audio_sec` (ví dụ 30) hoặc
`batch_target_sec` — không cần sửa code.

Mô phỏng đầu-cuối `sim_pipeline_b --hole-sec 0.2` (khe PTS ~0.2 s như bộ giải mã thật) được giữ lại
làm công cụ tái hiện.

### 4.2b. CỠ KHỐI THÍCH ỨNG THEO BỘ ĐỆM (2026-10-04)

**Vấn đề đo được**: ảnh popup cho thấy YouTube giữ `Lookahead available +117.95s`, log cho thấy
`Đệm trước: 88.3s`, nhưng bộ cắt khối chỉ gửi khối **12–18 s** cho ASR ⇒ lãng phí phần lớn khoảng
đệm đã có sẵn.

**Cỡ khối nay suy từ `available_sec` (audio đã có phía trước `from_pts`) tại mỗi lần cắt:**

| Tình huống | target | dải tìm | khối thực tế |
|---|---|---|---|
| Bộ đệm dồi dào (>= 30 s, như YouTube) | 25 s | tới 30 s | **~25–29 s** |
| Bộ đệm vừa (ví dụ 20 s) | 18 s | 18 s | ~18 s |
| Bộ đệm eo hẹp (< `min_window_sec` = 8 s) | — | — | chờ thêm audio (`None`) |

Công thức:

```python
budget         = min(available_sec, max_audio_sec)          # trần token của DLL
search_max     = adaptive_max_sec nếu available_sec >= adaptive_threshold_sec
                 ngược lại search_max_sec
search_max     = clamp(search_max, min_window_sec, budget)
search_max     = max(search_max, min(budget, target_window_sec + 2))   # đủ chỗ chọn khoảng lặng
effective_target = min(max_audio_sec, max(target_window_sec, budget - 2))
effective_target = min(effective_target, search_max - 2)
```

Thay đổi giá trị mặc định: `batch_target_sec 15→25`, `batch_min_sec 12→8`, `batch_max_sec 18→45`,
`batch_search_max_sec 18→30`, `batch_adaptive_max_sec 18→30`.
Sàn `batch_min_sec` hạ xuống 8 s vì sàn cũ 12 s **chặn luôn những khoảng lặng dài nhất nằm trong
12 s đầu** vùng tìm kiếm — tức bỏ phí các điểm cắt sạch nhất ngay gần mép khối.

### 4.2c. TRẦN CỨNG `batch_max_audio_sec = 45 s` — RÀNG BUỘC NATIVE

`external/transcribe.cpp/src/arch/qwen3_asr/model.cpp`:

```cpp
constexpr int k_max_new = 256;                     // dòng 78
const int max_audio_tokens = hp.dec_max_position_embeddings - k_prompt_overhead - k_max_new;
...
const int32_t max_new = k_max_new;                 // dòng 874 — ngân sách sinh token
```

`struct transcribe_run_params` (C ABI) **không** có trường `max_new_tokens` ⇒ **chừng nào chưa sửa
C++ và build lại DLL**, một lần `run()` chỉ sinh được tối đa **256 token**; khối dài hơn sẽ bị **nuốt
chữ âm thầm ở cuối**.

Đo thật để chọn trần (`transcribe_block` trên audio liên tục):

| Thời lượng khối | Số từ | Ký tự | `was_truncated` |
|---|---|---|---|
| 30 s | 58 | 302 | False |
| 45 s | 97 | 502 | False |
| 60 s | 125 | 630 | False nhưng câu bị cụt giữa chừng |

Tốc độ nói đo được trên đoạn đối thoại liên tục: **~2.6 từ/giây (~13 ký tự/giây)**. Với ~1.3
token/từ ⇒ 256 token ≈ **75 s audio** ⇒ khớp với ước lượng 60–90 s mà người dùng đưa ra.

⇒ Chọn trần **45 s** (biên an toàn ~1.7×). Khối thực tế hiện tại 25–29 s nằm sâu trong vùng an toàn.
`enforce_session_limits` **không** cứu được trường hợp này: `_max_audio_samples` đọc từ
`transcribe_limits` là 5218 s (trần NGỮ CẢNH, không phải trần SINH TOKEN).

**Khi nào nâng trần**: sau khi thêm `max_new_tokens` vào `struct transcribe_run_params`, tăng
`k_max_new` trong `src/arch/qwen3_asr/model.cpp`, build lại DLL, rồi nâng `batch_max_audio_sec`
tương ứng (và `batch_target_sec` nếu muốn khối dài hơn).

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
