# Phase 7 — Phiên 2 mất âm thanh: mốc MEDIA THẬT của mảnh audio (Pipeline B)

**Ngày**: 2026-10-03
**Triệu chứng**: Mở phiên lookahead lần 1 → chạy tốt. Bấm Stop rồi Start lại (phiên 2) trên **cùng
video, không reload trang** → không lấy được âm thanh: backend `Đã dịch: 0.0s`, giải mã `0.00x`,
video kẹt ở mốc cũ.

---

## 1. Bằng chứng từ log

```
14:37:24.564 [WS] Lookahead init: neo vị trí phát hiện tại @38.46s          ← phiên 2
14:37:24.566 [WS] StreamDemuxer: SourceBuffer epoch mới = 10 (min_pts=37.96)
14:37:34.726 [ASR] Playhead: 38.5s | Audio RAM: 20.0s | Đệm trước: 71.5s | Đã dịch: 0.0s (giải mã 1.96x)
14:37:44.867 [ASR] Playhead: 38.5s | Audio RAM: 20.0s | Đệm trước: 71.5s | Đã dịch: 0.0s (giải mã 0.00x)
... (50 s sau vẫn y nguyên)
```

Đọc ra ba sự thật:

1. `buffered_end = 38.5 + 71.5 = 110.0s` ⇒ PCM **đã giải mã tới tận 110 s** (`_decoded_end_pts`), trong
   khi `Audio RAM = 20.0s` ⇒ timeline RAM **thủng một lỗ lớn ngay tại playhead** (38,5 s → ~90 s).
2. `Đã dịch 0.0s` / `đã qua ASR 0.0s` ⇒ bộ cắt khối không có audio nào để chạy; `ready_until_pts`
   đứng ở 38,5 s ⇒ `timelineQueue` khoá cứng playhead ⇒ video pause/resume liên tục
   (browser log: 8 dòng `✅ Phát video với phụ đề Lookahead (prebuffer_ready)`).
3. `giải mã 1.96x` ở cửa sổ đầu rồi `0.00x` về sau ⇒ **toàn bộ 20 s PCM đến trong một cú bung đầu
   tiên rồi cạn**: sau đó không có byte nào mới được gửi sang backend.

Browser log phiên 2 cũng chỉ có:
`📦 Replay 24 mảnh audio quanh vị trí phát (38.5s; bỏ 159 mảnh ngoài cửa sổ, 0 mảnh trùng)` — tức
interceptor **có** 183 mảnh trong cache nhưng chỉ gửi 24 mảnh… và 24 mảnh đó là audio ở **tương lai**.

## 2. Nguyên nhân gốc

`content/buffer_interceptor_poc.js` (v1.3) gắn mỗi mảnh bằng **vị trí phát tại lúc append**:

```js
const videoPts = video ? Number(video.currentTime) : undefined;   // KHÔNG phải mốc của dữ liệu
...
const selected = state.cachedAudioChunks.filter(pkt => pkt.payload.videoPts >= from && <= to);
```

Trình phát MSE **luôn tải trước**: một mảnh `appendBuffer` lúc đang phát 5 s thường chứa media ở
30–60 s. Vì vậy khi phiên 2 neo ở 38,46 s:

| Nhóm mảnh trong cache | `videoPts` (mốc gắn) | Media thật | Bản 1.3 chọn? |
|---|---|---|---|
| Append trong lúc phát 5–24 s (phiên 1) | 5–24 s | **~30–60 s (đúng chỗ cần)** | ✗ bị "ngoài cửa sổ" |
| Append trong lúc phát 30–38,5 s (lúc rảnh giữa 2 phiên, player fetch bù) | 30–38 s | ~90–110 s (tương lai) | ✓ được gửi |

⇒ Backend nhận **audio ở tương lai**, bỏ sót đúng đoạn chứa playhead. Đây cũng là lý do phải
`remote`/phân tích `videoPts` mới thấy vô lý: 159 mảnh "ngoài cửa sổ" chính là các mảnh chứa audio
cần thiết.

Lỗi phụ đi kèm (đã sửa luôn): **script bị tiêm lặp** (content script tạo `<script src>` + background
`scripting.executeScript({allFrames: true})`) — bản cũ `clearInterval` rồi chạy lại nên mỗi lần tiêm
thêm một listener `message` trong khi hook vẫn trỏ vào `state` của lần tiêm đầu ⇒ vòng lặp định kỳ
của state đó bị giết và mỗi bản tin `RESET_REPLAY_TOKEN` bị log N lần (log thật: **20 dòng cho 1 lần
Stop**).

## 3. Cách sửa

**Nguồn sự thật duy nhất cho mốc media là `SourceBuffer.buffered`.** Trong hook `appendBuffer`:
chụp `buffered` trước khi append, chờ `updateend`, chụp lại, lấy hiệu số → gắn
`payload.mediaStart/mediaEnd` cho mảnh (append bị trình duyệt tuần tự hoá nên mỗi mảnh có đúng một
`updateend`).

Sau đó:

* Chọn mảnh replay / prune cache theo **giao nhau của khoảng media** với cửa sổ
  `[playhead − 8 s, playhead + 45 s]`; chỉ lùi về `videoPts` khi không đọc được `buffered`.
* Cache giữ thêm mọi mảnh nằm trong vùng đệm hiện tại của trình phát (`video.buffered`) — chống
  trường hợp `getActiveVideo()` trả về thẻ `<video>` khác (đo thật: `videoPts` ở tận 83–218 s).
* Tải lại theo khoảng byte: **học `bytesPerSec` thật** từ (byte append ÷ giây media tăng thêm) thay
  cho hằng số 27 KB/s (AAC 128k thật ~16 KB/s ⇒ tải lệch về tương lai), và ưu tiên URL audio-only.
* Chỉ tải lại khi cache **không phủ playhead** (trước đây chỉ cần "có mảnh gần đó" theo `videoPts`).
* Log replay giờ in độ phủ thật: `phủ 30.1s→60.3s, CÓ audio tại playhead` (vàng = KHÔNG phủ).
* Interceptor trở thành **singleton** (`window.__VIBE_LOOKAHEAD_INTERCEPTOR__`) — tiêm lặp là no-op.
* Client (`lookahead-client.js`) lọc mảnh đệm khi flush bằng `payloadInReplayWindow()` (ưu tiên mốc
  media).

### Về đề xuất "xoá buffer của video để trang gửi lại"

Có tác dụng và đã được cài sẵn, nhưng để **thủ công** vì nó can thiệp vào chính bộ đệm trình phát
(khựng tiếng/hình, có trang bỏ qua khoảng bị thiếu):

```js
window.__VIBE_LOOKAHEAD_DEBUG__.nudgeRefetch(12)   // xoá [playhead, playhead+12s] khỏi SourceBuffer audio
```

Muốn tự động khi cache không phủ và không có URL để tự tải lại: đổi
`AUTO_NUDGE_ON_STARVATION = false` → `true` trong `buffer_interceptor_poc.js`. Nó chỉ là phương án
cuối; sửa mốc media mới là chữa gốc.

## 4. Kiểm thử

`extension_firefox/tests/buffer-interceptor.test.js` (+ bản sao bên `extension_chrome_edge`) — 9 test
dựng giả lập MSE/`<video>` trong `node:vm`. **7/9 test FAIL với interceptor bản cũ** (đã kiểm chứng
bằng cách chạy test với file lấy từ `git show HEAD:...`), trong đó test trọng tâm:

> `replay phiên mở GIỮA video chọn mảnh theo mốc media (không theo vị trí phát)`
> Mảnh A (append lúc phát 5 s, media 30–50 s) và mảnh B (append lúc phát 35 s, media 90–110 s);
> phiên mới neo 38,5 s ⇒ **chỉ được gửi A**.

Chạy (sandbox của DSH chặn `node --test <dir>` vì spawn con — chạy trực tiếp từng file):

```powershell
node extension_firefox/tests/buffer-interceptor.test.js        # 9/9
node extension_firefox/tests/lookahead-client.test.js          # 11/11
node extension_chrome_edge/tests/buffer-interceptor.test.js    # 9/9
```

## 5. Cách xác nhận trên trình duyệt

1. Reload add-on (bắt buộc — script tiêm được nạp lại chỉ khi trang reload: `Ctrl+Shift+R`).
2. Start → Stop → Start lại (không reload trang) ở cùng video, tại một mốc giữa video.
3. Console trang phải in: `📦 Replay N mảnh audio quanh vị trí phát (38.5s; phủ 30.1s→60.3s, CÓ audio
   tại playhead; ...)` — **màu tím**, không phải vàng.
4. Backend: `Đã dịch` tăng dần, `giải mã` > 0, `Playhead` chạy theo video.
5. `window.__VIBE_LOOKAHEAD_DEBUG__.printReport()` → `mediaTaggedChunks` phải ≈ `cachedChunksCount`.
   Nếu `mediaTaggedChunks = 0`: trang không dùng SourceBuffer audio riêng (muxed A/V) — Pipeline B
   không có byte audio, extension sẽ tự quay về Pipeline A.

## 6. Phạm vi thay đổi

| File | Nội dung |
|---|---|
| `extension_firefox/content/buffer_interceptor_poc.js` | mốc media thật, prune/replay theo mốc media, singleton, học bytes/s, `nudgeRefetch` |
| `extension_chrome_edge/content/buffer_interceptor_poc.js` | bản sao y hệt (2 extension dùng chung nội dung interceptor) |
| `extension_firefox/lib/lookahead-client.js` + bản chrome | `payloadInReplayWindow()` cho flush mảnh đệm |
| `extension_*/tests/buffer-interceptor.test.js` (mới) | 9 test hồi quy |

Không cần sửa backend: backend vốn đã xử lý đúng khi nhận đúng vùng audio.

---

# Phần 2 — Trang KHÁC (không phải YouTube): PCM lệch trục 250 s

**Log thật (2026-10-03 15:32)** — cache interceptor đã đúng (nhờ Phần 1), nhưng backend vẫn
`Đã dịch: 0.0s`:

```
[Lookahead] 📦 Replay 50 mảnh audio quanh vị trí phát (7.1s; phủ 0.0s→56.6s, CÓ audio tại playhead; ...)
[WS]  StreamDemuxer: SourceBuffer epoch mới = 1 (min_pts=6.64)
[WS]  Đường giải mã nối-liền không ra PCM — đã chuyển sang giải mã TỪNG fragment (5.0s).
[ASR] Playhead: 7.1s | Audio RAM: 96.4s | Đệm trước: 299.7s | Đã dịch: 0.0s (giải mã 9.49x … 0.00x)
```

## 7. Nguyên nhân: hai TRỤC THỜI GIAN khác nhau

`Đệm trước 299.7s = 306.8 − 7.1` ⇒ backend giải mã ra PCM **kết thúc ở 306,8 s**. Trong khi
interceptor báo vùng media thật (theo `SourceBuffer.buffered`) là **0,0 → 56,6 s**.

```
trục CONTAINER (tfdt/timecode) : 250,0 → 306,8 s      ← backend dùng cái này
trục TRÌNH DUYỆT (`buffered`)  :   0,0 →  56,6 s      ← playhead 7,1 s nằm ở đây
lệch = 250,2 s (hằng số, cả hai đầu)
```

`container_pts + timestampOffset` chỉ bằng trục media của trình duyệt khi trang dùng
`SourceBuffer.mode = "segments"` **và** báo đúng `timestampOffset`. Trang này rơi vào một trong
các trường hợp:

* `mode = "sequence"`: trình duyệt **bỏ qua** mốc container, xếp segment nối tiếp theo
  `timestampOffset` ⇒ `tfdt` tuyệt đối (250+) không còn nghĩa;
* `timestampOffset` âm/khác được đặt **ngay sau** `appendBuffer()` — trình duyệt áp giá trị CUỐI
  cho chính mảnh đang chờ, còn interceptor đọc `this.timestampOffset` lúc gọi ⇒ gửi 0;
* luồng có `tfdt` tuyệt đối (mốc chương trình) trong khi timeline video bắt đầu từ 0.

Kết quả: PCM rơi vào 250→306 s, playhead ở 7,1 s ⇒ timeline **thủng** từ 7,1 s tới 250 s ⇒ bộ
cắt khối không có gì để chạy (`Đã dịch 0.0s`), video kẹt ở mốc cũ. Đây là **cùng một triệu
chứng** với Phần 1 nhưng ở tầng backend (Phần 1 là chọn sai mảnh, Phần 2 là đặt sai mốc).

## 8. Cách sửa: NEO PCM vào trục của trình duyệt

1. **Extension** (`buffer_interceptor_poc.js`): đọc `this.timestampOffset` ở **`updateend`**
   (thời điểm dữ liệu thật sự được đặt vào buffer) thay vì lúc gọi `appendBuffer()`.
2. **Extension** (`lookahead-client.js`): gửi kèm `media_start` / `media_end` (khoảng THẬT
   trong `SourceBuffer.buffered`) trong header khung nhị phân.
3. **Backend** (`stream_demuxer.py::_anchor_to_browser_axis`): sau mỗi lượt giải mã, hiệu
   `media_end − pts_end(PCM cuối)` chính là độ lệch trục. Lệch là **hằng số cho cả một
   SourceBuffer** nên chỉ cần học một lần (`_browser_axis_bias`) rồi dùng nó thay
   `timestampOffset` cho mọi lượt sau (kể cả đường giải mã từng fragment và đường AAC thô).

Chốt an toàn:

| Độ lệch đo được | Hành vi |
|---|---|
| ≤ 0,5 s | coi là mép frame — **không sửa**, chỉ khoá trục |
| 0,5 – 5 s | chờ quan sát thứ hai khớp (tránh mảnh hỏng ở mép cuối) rồi mới sửa |
| > 5 s | lỗi trục thật — sửa **ngay** (đúng ca 250 s này) |

Thêm cơ chế tự hồi phục: nếu có byte mới mà **không ra PCM nào** trong khi đã khoá trục (luồng
đổi base/period), neo bị bỏ và giải mã lại một lần.

Log mới cần thấy ở backend khi gặp trang dạng này:

```
[WARNING] [WS] LỆCH TRỤC THỜI GIAN: PCM giải mã lệch -250.20s so với timeline trình duyệt
               — đã neo lại theo `SourceBuffer.buffered` (trục mới = container + -250.20s).
[INFO   ] [WS] Trục PCM khớp trục media của trình duyệt (lệch +0.02s) — khoá trục.   ← trang YouTube
```

## 9. Kiểm thử Phần 2

`backend/tests/test_50_lookahead_demuxer.py` (+4 test, tổng 18/18 PASS):

* `test_axis_anchor_follows_browser_hint_when_container_is_offset` — **hồi quy chính**: hint
  lệch +250 s ⇒ PCM phải nằm ở 250→258 s (bản cũ: 0→8 s ⇒ FAIL).
* `test_axis_anchor_keeps_container_axis_when_hints_match` — hint trùng trục ⇒ **không dịch**
  (không phá YouTube).
* `test_axis_anchor_does_not_double_apply_timestamp_offset` — hint đã gồm `timestampOffset`
  ⇒ không cộng hai lần.
* `test_axis_anchor_defers_small_shift_until_confirmed` — lệch 3 s: lượt 1 giữ nguyên, lượt 2
  mới neo.

`extension_*/tests/buffer-interceptor.test.js` (+2 test, tổng 11/11 PASS): `timestampOffset`
gửi đi là giá trị tại `updateend`; header khung nhị phân mang `media_start/media_end`.

Toàn bộ suite lookahead backend: 72/72 PASS. `pytest` toàn repo: 0 FAILED (43 ERROR còn lại là
do sandbox chặn `tmp_path`, không liên quan).

---

# Phần 3 — "Pipeline B không ngắt câu dài" (ảnh chụp phụ đề Nhật, 2026-10-03)

**Triệu chứng**: một phụ đề chứa TRỌN 3 câu tiếng Nhật (tràn 2 dòng) và bản dịch cũng thành một
khối dài; tầng lịch sử (Layer 1) hiện thêm một bản dịch khác của cùng nội dung.

## 10. Nguyên nhân: `source_lang="auto"` ⇒ aligner bị gọi bằng `language="English"`

`lookahead_handler` cũ:

```python
align_lang = resolve_aligner_language(self.source_lang) or "English"   # "auto" ⇒ None ⇒ "English"
```

`Qwen3-ForcedAligner.encode_timestamp()` **chỉ** tách từ riêng cho `japanese` (`nagisa`) và
`korean`; mọi ngôn ngữ khác dùng `tokenize_space_lang` → `split_segment_with_chinese`, tức cắt
theo Hán tự và **gộp cả cụm kana thành một token**. Với câu tiếng Nhật (không có khoảng trắng),
`？`/`。` rơi vào GIỮA token ⇒ tầng gom câu không thấy dấu kết câu ở cuối token nào ⇒ không ngắt
được; bước "gộp mảnh bị chẻ" (rule 3) lại gộp về một ⇒ **cả khối thành một phụ đề**.

Đo bằng chính tokenizer THẬT của upstream (không cần model) trên đúng câu trong ảnh:

| `language` | số token | token kết thúc bằng dấu kết câu | kết quả gom câu |
|---|---|---|---|
| `"English"` (bản cũ) | 16 (cụm kana gộp, `温泉街` bị chẻ ký tự) | **1** (`。` cuối cùng) | **1 phụ đề** (text còn bị nối bằng khoảng trắng) |
| `"Japanese"` (đã sửa) | 28 từ đúng (nagisa) | **3** (`？`, `。`, `。`) | **3 phụ đề** |

Đúng ảnh chụp: dòng tiếng Nhật có khoảng trắng lạ sau `？`/`。` — dấu vết của `" ".join(...)` khi
`is_cjk = False`.

## 11. Cách sửa

1. `forced_aligner.detect_aligner_language_from_text()` — đoán ngôn ngữ cho aligner TỪ VĂN BẢN ASR
   khi người dùng để `auto` (Hangul ⇒ Korean, Kana ⇒ Japanese, chỉ Hán tự ⇒ Chinese). Người dùng
   chọn tay vẫn luôn thắng.
2. `ForcedAlignerService.split_oversized_sentences()` — **lưới an toàn** ở tầng VĂN BẢN: phụ đề nào
   chứa ≥ 2 dấu kết câu thì bị tách theo dấu câu (kèm ngoặc/nháy đóng), thời gian chia theo TỈ LỆ
   ĐỘ DÀI VĂN BẢN; mảnh nào vẫn quá dài (ASR không sinh dấu câu) thì cắt cứng theo trần ký tự/từ.
   Áp cho MỌI đường (kể cả nhánh aligner lỗi ⇒ trước đây cả khối thành một câu).
3. Log `[SEG_BATCH]` giờ in kèm `align=<Ngôn ngữ>` để nhìn ra ngay aligner đang chạy ngôn ngữ nào.

## 12. Kiểm thử Phần 3

* `test_57_forced_aligner_service.py`: `detect_aligner_language_from_text` (ja/zh/ko/en);
  `test_real_upstream_tokenizer_explains_single_long_subtitle` — **hồi quy gốc rễ**, đo bằng
  tokenizer thật của upstream: `"English"` ⇒ 1 phụ đề + text có khoảng trắng, `"Japanese"` ⇒ 3;
  lưới an toàn (tách theo dấu câu, cắt cứng khi không có dấu câu, không đụng phụ đề đã đúng).
* `test_58_lookahead_offline_batch.py`:
  `test_batch_auto_source_language_splits_long_japanese_utterance` — chạy TRỌN vòng batch với
  `source_lang="auto"` + văn bản Nhật, assert aligner nhận `"Japanese"` và ra **đúng 3 phụ đề**
  (đã xác nhận test FAIL nếu bỏ phần đoán ngôn ngữ).

## 13. Liên quan tới Layer 1 (bản dịch lặp)

Ảnh chụp còn cho thấy Layer 1 hiện một bản dịch KHÁC của cùng nội dung: đó là phụ đề của khối
TRƯỚC (khối chồng lấn). Với aligner chạy sai ngôn ngữ, cả hai việc đều hỏng: mốc từ sai (⇒ lớp trừ
chồng lấn theo thời gian không loại được phần đã phát) và `sub.text.split()` trên tiếng Nhật chỉ ra
MỘT token (⇒ lớp trừ theo chuỗi từ cũng không khớp được). Sau khi chọn đúng ngôn ngữ, mốc từ đúng
nên phần chồng lấn được trừ như với tiếng Anh.


