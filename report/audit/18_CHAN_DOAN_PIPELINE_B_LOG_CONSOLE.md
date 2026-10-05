# 18 — Chẩn đoán Pipeline B bằng Console trình duyệt

Ngày: 2026-10-05
Phiên bản: cả hai bản extension đều là 0.6.7
Phạm vi: `extension_firefox/` là bản được port đầy đủ và là bản để kiểm thử; `extension_chrome_edge/`
giữ nguyên phần chẩn đoán đã viết (không còn là nơi phát triển chính).

---

## 1. Vấn đề

Pipeline B (Lookahead) đã ổn định với YouTube/Bilibili nhưng khi thử các trang khác thì:

| Trang | POPUP hiện | Thực tế |
|---|---|---|
| `xvideos.com` | `Lookahead available +Xs` | Bấm Start: video đứng hình, `Đã dịch: 0.0s` mãi |
| `iq.com` | `Lookahead available +5.95s (đang nạp)` | Vùng đệm ("vùng xám") của video đã ~60s |
| `xhamster.com`, `av01.media` | `Lookahead: không khả dụng (chạy Realtime)` | Không rõ vì sao |

Trước bản này, **không có một dòng log nào ở phía trình duyệt** giải thích nguyên nhân:
interceptor chỉ log khi bắt được SourceBuffer audio, content script chỉ log khi Start, popup chỉ
hiện hai câu trạng thái thô. Log backend (`[WS]`, `[ASR]`, `[CORE]`) lại không biết client đang làm gì.

---

## 2. Dùng trong 3 bước

Mở **DevTools Console của trang video** (F12), rồi:

```js
// 1) Kết luận ngay: vì sao Pipeline B chạy được / không chạy được trên trang này
__VIBE_LOOKAHEAD_DEBUG__.diagnose()      // (MAIN world — luôn có nếu interceptor đã tiêm)
__bsLookaheadDiagReport()                // (content script — kèm thống kê phiên B đang chạy)

// 2) Lọc toàn bộ chuỗi quyết định
//    Gõ vào ô Filter của Console:  [Diag]

// 3) Bật/tắt log chẩn đoán của content script
__bsLookaheadDiag(false)   // tắt
__bsLookaheadDiag(true)    // bật lại
```

Lọc theo kênh:

| Tiền tố | Ở đâu | Nội dung |
|---|---|---|
| `[Lookahead][Diag]` | MAIN world (interceptor) | `addSourceBuffer`, sniff container/codec, DRM, lệch thẻ video, đứng yên khi pause |
| `[BS][Diag][gate]` | content script | Cổng quyết định Pipeline B + báo cáo chẩn đoán đầy đủ |
| `[BS][Diag][pipeline]` | content script | Đã chọn Pipeline A hay B và vì sao |
| `[BS][Diag][session]` | content script | Vòng ⏸️ TẠM DỪNG / ▶️ PHÁT TIẾP khi nạp đệm |
| `[BS][Diag][status]` | content script | `lookahead_status` rút gọn (nhịp 3s) |
| `[BS][Diag][stall]` | content script | BẾ TẮC + van an toàn |
| `[Lookahead Client][Diag]` | lib client | Socket, khung audio đã gửi, bản tin nhận được |
| `[Popup][Diag]` | popup (Inspect popup) | Mã lý do + bảng chẩn đoán |

---

## 3. Mã lý do (một nguồn sự thật: `lib/lookahead-diagnostics.js`)

`LookaheadDiagnostics.classifyLookaheadAvailability(facts)` trả về **đúng một** mã:

| Mã | Nghĩa | Xử lý |
|---|---|---|
| `OK` | MSE audio-only, đã bắt được mảnh + mốc media | chạy Pipeline B |
| `NO_VIDEO` | không có thẻ `<video>` trong frame | bấm Play / kiểm tra iframe |
| `LIVE_STREAM` | `duration` = Infinity/NaN | phải dùng Pipeline A |
| `DRM` | trang gọi `requestMediaKeySystemAccess` (Widevine…) | audio ĐÃ MÃ HOÁ ⇒ bất khả thi |
| `NO_MSE_API` | frame không có `window.MediaSource` | môi trường lạ |
| `MSE_UNUSED` | có `MediaSource` nhưng **0** `addSourceBuffer` | trang phát trực tiếp/HLS native ⇒ xem §7 |
| `VIDEO_MISMATCH` | interceptor theo dõi thẻ `<video>` KHÁC content script | đã sửa: neo theo `currentSrc` (§6) |
| `MUXED_ONLY` | audio chỉ nằm trong SourceBuffer `video+audio` **và cờ nhận-MUXED đang TẮT** | bật `ACCEPT_MUXED_AUDIO` (mặc định BẬT từ 0.6.7) |
| `UNKNOWN_MIME_SB` | SourceBuffer tạo trước khi hook chạy | đã sửa: sniff container/codec (§6) |
| `VIDEO_ONLY_SB` | chỉ có SourceBuffer video | audio đi đường khác (EME/WebAudio) |
| `NO_APPEND_YET` | chưa có lần append nào rơi vào buffer AUDIO | bấm Play / tua nhẹ (xem `recentAppends`) |
| `SOURCE_CHANGED_RESET` | trang vừa đổi nguồn video ⇒ cache vừa bị xoá | chờ video chính ổn định rồi mới Start |
| `INIT_ONLY` | chỉ có header, chưa có mảnh media | tua để buộc nạp mảnh mới |
| `NO_AUDIO_BYTES` | có append audio nhưng cache rỗng | xem `appendsNoRawBytes`/`appendsPrunedOut`/`recentAppends` |
| `NO_MEDIA_TIMELINE` | có mảnh nhưng 0 mảnh có mốc media | lỗi `buffered`/append |
| `BUFFERED_LOW` | đệm trước < 1.5s | chờ nạp thêm |
| `STALL_NO_PROGRESS` / `HORIZON_DEADLOCK` | phiên B bế tắc (xem §5) | van an toàn tự phá (§6) |

Mỗi mã đều có `why` (vì sao) và `fix` (cách xử lý) bằng tiếng Việt, in ra trong báo cáo nhóm.

---

## 4. Lệch thẻ video — nguyên nhân đã xác định của `iq.com`

`content/buffer_interceptor_poc.js` chọn `<video>` theo tiêu chí **khác** với `findVideo()` của
content script (interceptor: thẻ đang phát → thẻ to nhất; content script: chấm điểm
playing/readyState/diện tích). Trên trang nhiều thẻ (quảng cáo, preview, player phụ) hai bên nói
về **hai video khác nhau**:

* POPUP hiện đệm của **thẻ interceptor** (5.95s) — không phải của video đang xem (~60s).
* Mảnh audio gửi lên backend thuộc video khác ⇒ `Đã dịch 0.0s` dù cache đầy.

Đã sửa: content script gửi `SET_PREFERRED_VIDEO` (`currentSrc`) khi kiểm tra cổng và trước mỗi
lần hỏi trạng thái; `getActiveVideo()` ưu tiên thẻ khớp `currentSrc`. Khi vẫn lệch, log
`[Lookahead][Diag] ⚠️ LỆCH THẺ VIDEO` nêu rõ hai `src` + `currentTime` và payload có
`diag.videoMismatch` để popup cảnh báo.

---

## 5. Đọc log của `xvideos.com` (bế tắc horizon)

Log backend người dùng gửi có dạng "đứng yên" rất đặc trưng:

```
Playhead: 15.7s | Audio RAM: 10.0s | Đệm trước: 34.3s | Đã dịch: 0.0s
   (đã qua ASR 0.0s, giải mã 0.98x → 0.00x, PCM/byte 0.000)     ← lặp y hệt mỗi 10s
```

Suy ra từ mã nguồn:

1. `Playhead` **không đổi** trong ~50s ⇒ video đang bị extension **tạm dừng** (Pipeline B pause để nạp đệm).
2. `PCM/byte 0.000` ⇒ byte về nhưng **không giải mã ra PCM mới** (hoặc PCM bị bỏ vì nằm quá xa playhead).
3. `đã qua ASR 0.0s` ⇒ backend **chưa từng chạy ASR**, nên `ready_until_pts` không tiến.
4. Ràng buộc cứng ở client (`SubtitleTimelineQueue._tick` → `isBehindHorizon`) thấy
   `playhead ≥ ready_until_pts` ⇒ tạm dừng tiếp.

Vòng lặp kín:

```
horizon chặn playhead
      ↓
video bị tạm dừng
      ↓
trình phát KHÔNG appendBuffer thêm mảnh mới   ← đây là mắt xích bị bỏ qua
      ↓
backend không có audio mới ⇒ mốc đã xử lý đứng yên
      ↓
quay lại "horizon chặn playhead"
```

Đó là lý do Pipeline B chạy tốt trên YouTube/Bilibili (đệm trước sâu 30–60s, có sẵn đủ audio để
backend cắt khối 12–30s và đẩy mốc lên) nhưng **chết cứng** trên trang chỉ có ~10s đệm.

Bản này in ra đúng vòng lặp đó:

```
[BS][Diag][session] ⏸️ TẠM DỪNG video để nạp đệm (lần 5, lý do=underrun, vừa chạy lại được 0.2s)
    | playhead=15.72s | mốc-đã-xử-lý=15.71s ready=0.00s fed=0.00s fragments=3 tts_sent=0
[BS][Diag][session] ⛔ Chạm mốc đã xử lý: playhead=15.72s ≥ mốc-đã-xử-lý=15.71s ⇒ tạm dừng chờ backend.
    Nếu dòng này lặp lại mãi cùng một mốc ⇒ BẾ TẮC (xem [BS][Diag][stall]).
[Lookahead][Diag] ⏸️ Trình phát đang TẠM DỪNG và đã 14s không có mảnh append nào mới...
[BS][Diag][stall] 🚑 PHÁ BẾ TẮC (lần 1): ... Cho video CHẠY LẠI và treo horizon 20s.
```

---

## 6. Sửa đổi kèm theo (để Pipeline B chạy được thật, không chỉ để biết lý do)

1. **Nhận SourceBuffer MUXED** (`ACCEPT_MUXED_AUDIO`, mặc định BẬT)
   Nhiều trang chỉ tạo MỘT SourceBuffer `video/mp4; codecs="avc1…,mp4a…"`. Bản cũ chỉ nhận mime có
   chữ `audio` ⇒ bỏ qua toàn bộ mảnh. Nay phân loại theo **codec** (`mp4a/opus/aac…`), và backend
   (PyAV `streams.audio[0]`) tự chọn track tiếng. Tắt bằng
   `window.__VIBE_LOOKAHEAD_ACCEPT_MUXED__ = false` trước khi trang tạo SourceBuffer.
   *Lưu ý:* byte gửi lên nhiều hơn (kèm video) — theo dõi `[Lookahead Client][Diag] 📊 Đã gửi … MB`.

2. **Phân loại lại SourceBuffer "mù mime"** (tạo trước khi hook chạy)
   Byte đầu tiên được sniff container (`fmp4/init`, `webm/matroska`, `mpeg-ts`, `adts-aac`…) và
   fourcc codec (`mp4a`/`opus` ⇒ audio, `avc1`/`hvc1` ⇒ video, cả hai ⇒ muxed). Buffer được xác
   định là VIDEO sẽ **không** bị cache ⇒ POPUP không còn báo "available" sai.

3. **Neo thẻ video** theo `currentSrc` (§4).

4. **Van an toàn chống bế tắc** (`lookaheadStallGuard`, mặc định BẬT)
   Khi video đang bị extension tạm dừng (từ lần thứ HAI trở đi) mà `fragments` **và**
   `ready_until_pts` của backend không tiến trong **25s** (cùng ngưỡng watchdog backend) ⇒ cho video
   chạy lại và **treo tạm ràng buộc horizon 20s** (`SubtitleTimelineQueue.suspendHorizon`) để trình
   phát nạp thêm audio; horizon được gỡ ngay khi backend đuổi kịp playhead. Điều kiện "lần thứ hai"
   để không phá nhầm lần nạp đệm đầu tiên hay một khối ASR dài. Tắt: `window.__bsStallGuard = false`
   (đặt trước khi bấm Start) hoặc `lookaheadStallGuard: false` trong `bs_settings`.

5. **Thống kê gửi/nhận** ở `LookaheadClient.getDiag()`: `framesSent`, `framesSentBytes`,
   `framesQueued`, `framesDropped` (mất audio thật), `framesBlocked`, `serverMessages`
   (backend có gửi `lookahead_status`/`lookahead_subtitles` không).

---

## 7. Nhóm "không dùng MSE" (`xhamster.com`, `av01.media`) — việc còn lại

Nếu chẩn đoán ra `MSE_UNUSED` (hoặc `VIDEO_ONLY_SB`), nghĩa là trang **không đưa audio qua
`SourceBuffer`** — không có `appendBuffer` nào để hook. Các hướng khả thi, theo thứ tự chi phí:

1. **Chế độ tải theo URL (Range)** — interceptor đã ghi lại `{url, start, end, audioOnly}` của mọi
   request có header `Range` (hook `fetch`/XHR) và đã có sẵn `refetchForPlayhead()` để tải lại
   đúng khoảng byte. Chỉ cần: lấy init segment bằng một request `bytes=0-…` cho URL media trực tiếp,
   rồi nuôi backend từng khoảng byte phía trước playhead. **Chưa làm** — cần thêm bước xác nhận
   codec của URL (fMP4 progressive vs HLS phân đoạn) và luật "vùng nào cần".
   Kiểm tra tiền đề bằng: `__VIBE_LOOKAHEAD_DEBUG__.diagnose()` → dòng
   `Range đã ghi để tải lại: N`; nếu `N > 0` là khả thi.
2. **HLS/DASH native** (`video.src` kết thúc `.m3u8`): phải đọc manifest và tải segment theo
   timeline — nhiều việc hơn, dễ vỡ khi manifest có DRM/khoá.
3. **DRM**: dừng lại — không có đường nào giải mã hợp lệ.

---

## 8. Trạng thái hai bản extension

`extension_firefox/` là **bản chính để kiểm thử** và đã có ĐẦY ĐỦ chẩn đoán:

| File | Trạng thái |
|---|---|
| `content/buffer_interceptor_poc.js` | giống `chrome_edge` từng byte |
| `content/content-script.js` | giống `chrome_edge`, chỉ khác 2 khối chú thích riêng của Firefox |
| `lib/lookahead-client.js` | giống `chrome_edge`, KHÁC đúng một điểm: `useBridge` mặc định = `chrome.runtime.connect` có tồn tại (Firefox dùng Background Bridge; Chromium phải dùng WebSocket trực tiếp) |
| `lib/lookahead-diagnostics.js` | giống `chrome_edge` |
| `lib/lookahead-timeline.js`, `popup/popup.js`, `popup/popup.html` | giống `chrome_edge` (ràng buộc `test_extension_mirrors_stay_in_sync`) || `manifest.json` | giữ đặc thù Firefox (`browser_specific_settings`, `background.scripts`, `match_origin_as_fallback`) + thêm `lib/lookahead-diagnostics.js` |
| `tests/` | đã thêm `lookahead-diagnostics.test.js`, cập nhật `buffer-interceptor.test.js` |

`extension_chrome_edge/` **được giữ nguyên có chủ ý** (theo yêu cầu người dùng: chỉ làm việc với
bản Firefox trước, Chrome cập nhật sau nếu bản Firefox chạy tốt). Vì vậy bản Chrome hiện vẫn còn
lỗi `MUXED_ONLY` và `isInitSegment` cũ; khi cần đồng bộ thì chép đè 3 file:
`content/buffer_interceptor_poc.js`, `lib/lookahead-diagnostics.js`,
`tests/buffer-interceptor.test.js`, `tests/lookahead-diagnostics.test.js`
(riêng `content/content-script.js` và `lib/lookahead-client.js` phải giữ khác biệt đặc thù —
xem bảng trên).

Nạp lại để dùng: `about:debugging` → **Load Temporary Add-on…** → chọn
`extension_firefox/manifest.json`, rồi kiểm tra phiên bản hiện là **0.6.7**.

---

## 9. Kiểm thử
```powershell
# Test của extension (chạy từng file — `node --test <thư mục>` cần spawn tiến trình con)
node extension_firefox/tests/lookahead-diagnostics.test.js
node extension_firefox/tests/buffer-interceptor.test.js
node extension_firefox/tests/lookahead-client.test.js
node extension_firefox/tests/lookahead-timeline.test.js
node extension_firefox/tests/tts-timeline.test.js
node extension_firefox/tests/backpressure-gate.test.js

# Harness JS của backend (đọc trực tiếp file trong extension_firefox)
node backend/tests/js/demo_client_sync.js

# Test backend có đọc file extension
.venv\Scripts\python.exe -m pytest backend/tests/test_59_pipeline_b_only_and_popup.py -q
```

Test mới khoá lại: phân loại MUXED/video-only, sniff codec cho buffer mù mime, payload trạng thái
có đủ dữ kiện chẩn đoán, tích hợp "dữ kiện interceptor → mã lý do", toàn bộ từ điển mã lý do,
đếm riêng append audio/video, mảnh bị prune ngay, và init segment fMP4 bắt đầu bằng `moov`.

---

## 10. Ba log thực tế (2026-10-05) và những gì đã sửa

### 10.1 `xhamster.com` và `av01.media` — báo `MUXED_ONLY` nhưng THỰC RA chạy được

```
Mime đã thấy: video/mp4;codecs=mp4a.40.2,av01.0.05M.08.0.111.01.01.01.0   (xhamster)
Mime đã thấy: video/mp4;codecs=mp4a.40.2,avc1.64001E                      (av01.media)
• Mảnh trong cache: 3 (có mốc media thật: 3)     ← interceptor ĐÃ bắt được audio
• Kết luận: MUXED_ONLY                            ← SAI: gây hiểu là "không khả dụng"
```

**Lỗi ở bộ chẩn đoán (đã sửa):** `classifyLookaheadAvailability` trả `MUXED_ONLY` ngay khi
`audioSbCount === 0 && muxedSbCount > 0`, **không xét** việc cờ nhận-MUXED đã BẬT và cache đã có
mảnh. Hệ quả: log/popup nói "không chạy được" trong khi cổng Pipeline B thực tế vẫn ĐẠT (vì cổng
chỉ xét `cachedChunksCount > 0 || hasInitSegment`).

Nay: `audioCapable = audioSb > 0 || (acceptMuxed && muxedSb > 0)`; chỉ trả `MUXED_ONLY` khi cờ
nhận-MUXED TẮT. Hai trang này giờ kết luận **OK — chế độ MUXED**, và `MUXED_ONLY` chỉ còn nghĩa
"tính năng nhận MUXED đang tắt".

### 10.2 `xvideos.com` — báo `NO_AUDIO_BYTES` với "14 append nhưng cache rỗng"

```
SourceBuffer: audio=1, video=1, muxed=0, không rõ mime=0
Mime đã thấy: audio/mp4;codecs=mp4a.40.2 | video/mp4;codecs=avc1.4d401e
Container sniff: fmp4/init        ← CÓ byte audio được sniff...
Init segment: chưa                ← ...nhưng KHÔNG được nhận là init segment
Mảnh trong cache: 0
Byte đã append: 14 lần
```

Hai lỗi riêng biệt lộ ra ở đây:

1. **`14 append` gộp cả buffer VIDEO.** `diag.appendCount` đếm mọi `appendBuffer`, nên con số đó
   không nói được audio có về hay không ⇒ kết luận `NO_AUDIO_BYTES` có thể sai hoàn toàn.
   *Đã sửa:* đếm tách `appendsAudioBearing` / `appendsSkippedVideo` / `appendsNoRawBytes` /
   `appendsPrunedOut` / `appendsInit` / `appendsMedia`, và với cache rỗng thì phân biệt
   `NO_APPEND_YET` (0 lần append audio) với `NO_AUDIO_BYTES` (có append audio mà vẫn rỗng).

2. **`isInitSegment` bỏ sót init segment fMP4 không bắt đầu bằng `ftyp`/`styp`.** Bản cũ chỉ nhận
   `ftyp`/`styp`; nhiều init bắt đầu bằng `moov`, `sidx`, hoặc `free/skip/wide` rồi mới tới `moov`.
   Khi đó header bị coi là mảnh media: `Init segment: chưa` (và backend không bao giờ có header để
   ghép nối) — đúng như log trên. *Đã sửa:* không chứa `moof` mà có `moov` (hoặc `sidx`, hoặc
   `free/skip/wide` + `moov`/`ftyp`) ⇒ nhận là init.

3. **Cache có thể rỗng vì trang vừa ĐỔI NGUỒN VIDEO.** `handleVideoOrUrlChange` (đúng theo thiết
   kế) xoá sạch cache khi `video.currentSrc` đổi, nhưng sự kiện này trước đây KHÔNG được ghi lại.
   *Đã sửa:* ghi `cacheResetCount` / `lastCacheResetReason` / `msSinceCacheReset`, và thêm mã
   `SOURCE_CHANGED_RESET` để báo cáo không còn nói nhầm "chưa bắt được byte".

Ngoài ra, với nguồn MUXED thì mỗi mảnh mang cả byte video nên trần cache 12 MB chỉ giữ được vài
giây ⇒ **trần riêng 32 MB** cho chế độ MUXED (`state.muxedCacheMaxBytes`).

### 10.3 Đọc lại nhanh

```js
__VIBE_LOOKAHEAD_DEBUG__.diagnose()
```

In thêm 2 thứ mới:

* dòng `Append: N lần — X trên buffer AUDIO (init=…, media=…), Y bỏ qua (video), Z không đọc được byte, W bị prune ngay`
* bảng `recentAppends` (8 lần append gần nhất) với cột `note` nói rõ từng mảnh đã đi đâu:
  `đã cache` / `bị prune ngay (playhead=…s)` / `buffer video — bỏ qua` / `KHÔNG đọc được byte`.

---

## 11. `xhamster.com` chạy được đoạn đầu rồi "nạp lại liên tục" (2026-10-05)

Hiện tượng: đoạn đầu có phụ đề, sau đó video khựng/nạp lại liên tục dù vùng đệm phía trước còn dài.

### 11.1 Dấu vết quyết định trong log

```
[BS][Diag][session] ⏹️ Kết thúc phiên Pipeline B: pause=39, resume=38, van-an-toàn=0,
                    lý do dừng={"start":1,"seek":38}, mốc-đã-xử-lý cuối=96.00s.
[BS][Diag][status]  playhead=175.83s | ready=0.00s fed=0.00s target=2.50s |
                    mốc-đã-xử-lý=96.00s | fragments=65 decoded=24 utterances=23
[BS][Diag][stall]   ⏳ Đang chờ backend: video tạm dừng 4s (lý do=seek), ... mốc-đã-xử-lý tiến cách đây 212s
[Lookahead Client][Diag] 📊 Đã gửi 65 khung / 61.90 MB
```

Backend cùng lúc:

```
[SEG_GATE] Chờ gom khối dài từ 94.18s: có 1.8s/30.0s audio, đã dịch trước playhead 29.4s
Trạng thái Lookahead | Playhead: 175.8s | Audio RAM: 93.2s | Đệm trước: 0.0s | Đã dịch: 0.0s
                        (đã qua ASR 0.0s, giải mã 0.00x, PCM/byte 0.000)   ← lặp y hệt 30 lần
```

Ba dữ kiện:

1. **`lý do dừng={"start":1,"seek":38}`** — 38 trong 39 lần tạm dừng có lý do `seek`. Người dùng
   không hề tua 38 lần: đó là các sự kiện `seeking` **do trình phát tự sinh** (hết đệm/ngay sau khi
   ta gọi `video.play()`, hoặc trang tự nạp lại nguồn).
2. **`mốc-đã-xử-lý` đứng im ở 96.00s** trong khi playhead chạy 96s → 201s, `decoded=24` không nhích
   dù `fragments` vẫn tăng (65 → 71).
3. **`Audio RAM: 93.2s` không đổi** và `PCM/byte 0.000` ⇒ mảnh vẫn về nhưng **không giải mã ra PCM**.

### 11.2 Vòng lặp (phần thuộc về extension) — ĐÃ SỬA

```
trình phát bắn `seeking` (hết đệm / tự nạp lại nguồn)
      ↓
SubtitleTimelineQueue coi là TUA THẬT
      ↓
notifySeek() ⇒ backend RESET pipeline + pauseForBuffering("seek") ⇒ video bị tạm dừng 5s
      ↓
resume ⇒ playhead ngoài vùng đệm ⇒ trình phát lại bắn `seeking` …
```

Sửa trong `lib/lookahead-timeline.js`:

* Chỉ coi là **tua thật** khi vị trí nhảy ≥ `minSeekJumpSec` (mặc định 3s). Sự kiện `seeking`
  nhảy nhỏ hơn ⇒ **bỏ qua hoàn toàn**: không xoá phụ đề, không sinh `seek_id` mới, không gọi
  `notifySeek()`, không tạm dừng video — chỉ ghi log `[BS][Diag][session] ↩️ Bỏ qua "seeking" do
  TRÌNH PHÁT TỰ SINH (nhảy 1,40s) …`.
* Trần tần suất: quá `seekChurnLimit` (4) lần tua thật trong `seekChurnWindowMs` (10s) ⇒ chặn bớt
  để một trang "nhảy loạn" không thể reset backend liên tục.
* `stats()` trả thêm `ignoredSeeks` / `acceptedSeeks` / `horizonSuspended`; tổng kết phiên in thêm
  `seek-bỏ-qua=N`.

### 11.3 Vòng lặp (phần thuộc về backend) — CÓ CẢNH BÁO + TỰ NGẮT VÒNG

Van `fireStallGuard` cũ đòi **cả** `fragments` **và** mốc đã xử lý đứng yên, nên không bắt được ca
"mảnh vẫn về mà không ra PCM" (`van-an-toàn=0` trong log). Nay có nhánh riêng:

```
[BS][Diag][stall] ⚠️ Backend ĐANG NHẬN mảnh (fragments=71, mảnh mới cách đây 5s) nhưng
   mốc-đã-xử-lý KHÔNG tiến 282s (decoded=24, ready=0.00s). ⇒ byte về mà KHÔNG ra PCM/phụ đề:
   nút cổ chai ở tầng GIẢI MÃ phía backend. Xem log backend các dòng: PCM/byte,
   LỆCH TRỤC THỜI GIAN, Đường giải mã nối-liền không ra PCM, StreamDemuxer reset.
[BS][Diag][stall] ⏯️ Đã TREO ràng buộc horizon 180s để DỪNG vòng pause/seek: giữ video phát
   bình thường (tạm thời không có phụ đề) thay vì nạp lại liên tục.
```

Tức là: khi backend không giải mã được thì việc tạm dừng video là **vô nghĩa** — client ngừng tạm
dừng, để video phát bình thường và chờ backend đuổi kịp (horizon tự gỡ khi `ready_until_pts` vượt
playhead).

### 11.4 Việc còn lại (phía backend, cần log kèm)
`PCM/byte 0.000` + `Audio RAM` đứng im là lỗi **tầng ghép nối/giải mã**, không phải thiếu dữ liệu.
Để chốt nguyên nhân cần lọc log backend quanh thời điểm RAM ngừng tăng:

```
LỆCH TRỤC THỜI GIAN
Đường giải mã nối-liền không ra PCM
Bộ đệm ghép nối ... vượt trần
StreamDemuxer reset
```

Giả thuyết ưu tiên: sau khi trang **tự nạp lại nguồn**, mốc PTS trong container quay về 0 trong khi
trục trình duyệt đang ở ~175s; frontier `_last_pts` của demuxer (được neo theo `min_pts` của lần
`reset` gần nhất) lọc sạch mọi frame mới ⇒ không bao giờ ra PCM. Nếu đúng, cách sửa nằm ở
`backend/core/stream_demuxer.py`: phát hiện "trục media khởi động lại" bằng `media_start` do
extension gửi kèm và neo lại frontier thay vì lọc bỏ.

---

## 12. Vòng test #2 trên xhamster.com (2026-10-05 tối) — đã tìm ra và sửa 3 lỗi

Kết quả lần chạy này: **đoạn đầu chạy tốt rất dài** (tới ~575s), nhưng:

```text
# TUA THẬT (556s → 967s)
[BS][Diag][status] playhead=967.53s | ready=0.00s ... | mốc-đã-xử-lý=575.01s   ← đứng im
[BS][Diag][session]⏹️ pause=39 ... lý do dừng={"start":1,"seek":38}
# BACKEND cùng lúc
Trạng thái Lookahead | Playhead: 970.1s | Audio RAM: 736.2s | Đệm trước: 29.9s | Đã dịch: 0.0s

# TẮT/MỞ LẠI PHIÊN (Start lại ở 984s) — cũng không được
StreamDemuxer: SourceBuffer epoch mới = 1 (min_pts=983.55) — xoá bộ đệm ghép nối
Trạng thái Lookahead | Playhead: 984.0s → 1023.9s | Audio RAM: 0.0s | Đã dịch: 0.0s   ← suốt phiên
```

### 12.1 LỖI 1 (extension, do bản sửa trước gây ra): TUA THẬT BỊ BỎ QUA

Bộ lọc seek ở §11.2 so vị trí seek với `_lastKnownTime` — nhưng biến đó được cập nhật **mỗi khung
hình**. Trình duyệt đổi `currentTime` rồi để `requestAnimationFrame` chạy TRƯỚC khi phát `seeking`,
nên cú tua thật bị đo thành "nhảy 0s" và bị BỎ QUA ⇒ **không có `seek_reset` nào tới backend** ⇒
con trỏ khối của backend kẹt ở 575s trong khi playhead đã ở 967s ⇒ `Đã dịch: 0.0s` mãi mãi
(backend vẫn có audio: `Đệm trước: 29.9s`).

**Sửa** (`lib/lookahead-timeline.js`): bỏ hẳn debounce; quyết định **đồng bộ** bằng HAI nguồn bằng
chứng, lấy giá trị lớn hơn:

1. `target − _lastTickTime` — vị trí lúc nghe sự kiện so với mốc tick gần nhất;
2. **"nhảy bất thường" mà `_tick` quan sát theo ĐỒNG HỒ THỰC**: `|Δ currentTime| > dtReal ×
   playbackRate + 0,75s` ⇒ chắc chắn có tua, kể cả khi rAF đã chạy trước sự kiện.

Tua dồn dập (kéo thanh trượt) **không còn bị chặn** — mọi cú tua thật đều tới backend, chỉ ghi
cảnh báo `⚡ Tua dồn dập`. Đã có test tái hiện đúng ca hỏng:
`TUA THẬT được nhận dù currentTime đã đổi TRƯỚC khi sự kiện seeking bắn`.

### 12.2 LỖI 2 (backend): phiên mới KHÔNG BAO GIỜ học được trục thời gian

Phiên mới neo ở 984s ⇒ `min_pts = 983.55`, `_last_pts = 983.55`, nhưng `_browser_axis_bias` là
`None` (chưa học). Frontier lọc mọi frame có `pts < 983.55` ⇒ `chunks` luôn rỗng ⇒
`_anchor_to_browser_axis` (`if ... not chunks: return`) **không bao giờ** học được trục ⇒
`Audio RAM: 0.0s` suốt phiên. Vòng luẩn quẩn không có lối ra.

**Sửa** (`backend/core/stream_demuxer.py`): khi `chunks` rỗng và có `media_end` từ extension, HỌC
LẠI trục từ **dấu vết trình duyệt**: byte này kết thúc ở `media_end` ⇒
`bias = media_end − pts_end(container)`; đặt bias, xoá frontier về `min_pts`, giải mã lại. Nếu vẫn
rỗng thì trả lại trạng thái cũ (không để lại neo sai). Log:
`[WS] HỌC LẠI TRỤC THỜI GIAN từ dấu vết trình duyệt: byte kết thúc ở 992.00s … bias +990.45s`.

Regression test: `test_stream_demuxer_learns_axis_when_session_starts_mid_stream` — đo được
**0,0s PCM khi tắt fix** và **7,96s (đúng trục ~984s) khi bật fix**.

### 12.3 LỖI 3 (backend): con trỏ khối kẹt PHÍA SAU playhead

Van cũ `_maybe_reanchor_batch_frontier` chỉ xử lý "kẹt QUÁ XA PHÍA TRƯỚC tầm nhìn". Ca thật ở đây
ngược lại: con trỏ ở 575s (một mốc KHÔNG còn audio — khe hở của timeline) trong khi playhead ở
967s ⇒ `_batch_gate` trả `"no_audio"`, `next_chunk()` trả `None`, vòng lặp quay mãi với cùng một
`from_pts`, `mốc-đã-xử-lý` không bao giờ tiến.

**Sửa** (`backend/ws/lookahead_handler.py`): thêm van đối xứng — nếu con trỏ chậm hơn playhead
≥ `_FRONTIER_BEHIND_RESYNC_SEC` (20s) **và** `timeline.buffered_end_from(con_trỏ) is None` (không
có audio tại con trỏ) kéo dài ≥ `_FRONTIER_RESYNC_AFTER_SEC` (2s) ⇒ NEO LẠI về vị trí phát + log
`Con trỏ khối batch KẸT PHÍA SAU vị trí phát …`.

### 12.4 Bài học

* Mọi phép so sánh "vị trí trước/sau" trên `video.currentTime` phải dùng **đồng hồ thực** làm
  chuẩn, không dùng mốc cập nhật mỗi khung hình — thứ tự `seeking` vs `requestAnimationFrame`
  KHÔNG được bảo đảm.
* Ở tầng giải mã, "không giải mã được gì" và "chưa học được trục" là **cùng một trạng thái** nếu
  frontier chặn trước khi trục được học ⇒ phải có đường học trục KHÔNG phụ thuộc PCM đã phát
  (dùng `media_start`/`media_end` từ extension).
* Van chống kẹt phải có cả hai chiều: kẹt **trước** và kẹt **sau**.

---

## 13. Vòng test #3 trên xvideos.com (2026-10-05) — HLS tách track, "chạy một chút lại dừng"

Trang này dùng `player.html5hls` (HLS) với **hai SourceBuffer tách track**
(`audio/mp4;codecs=mp4a.40.2` + `video/mp4;codecs=avc1.4d401e`). Hiện tượng: có phụ đề một đoạn
đầu rồi video khựng theo chu kỳ.

### 13.1 Dấu vết trong log

```text
[WS] Đường giải mã nối-liền không ra PCM — đã chuyển sang giải mã TỪNG fragment (10.0s).
[WS] Trục PCM khớp trục media của trình duyệt (lệch +0.00s) — khoá trục.
[ASR] Lookahead Batch ASR [7.22s -> 17.49s]: 'I'm just a little bit afraid.'        ← đoạn đầu OK
[WS] LỆCH TRỤC THỜI GIAN: PCM giải mã lệch +10.01s … (trục mới = container + +10.01s)  22:00:46
[WS] LỆCH TRỤC THỜI GIAN: PCM giải mã lệch -10.01s … (trục mới = container + +0.00s)   22:00:50  ← ĐẢO
[ASR] [SEG_GATE] Chờ gom khối dài từ 20.04s: có 0.0s/30.0s audio
[ASR] Playhead: 41.0s | Audio RAM: 51.1s | Đệm trước: 39.0s | Đã dịch: 0.0s   ← có audio, không dịch
```

Client cùng lúc: `mốc-đã-xử-lý=20.04s` đứng im, playhead 41–43s, `pause=9 resume=8`,
`van-an-toàn đã phá bế tắc 1 lần`.

### 13.2 Nguyên nhân: ĐẢO TRỤC ±10s làm thủng timeline

`_anchor_to_browser_axis` so `media_end` (khoảng vừa thêm vào `SourceBuffer`) với mốc PCM cuối của
**toàn bộ** frame còn trong bộ đệm ghép nối. Với HLS tách track (mỗi segment ~10s), hai đại lượng
này lệch nhau **đúng một segment** và có thể **đổi dấu** giữa hai lần ⇒ neo +10,01s rồi −10,01s.

Hệ quả: sau lần đảo thứ hai, mọi fragment được giải mã với trục +0 (lệch −10s so với thực tế) ⇒
PCM rơi vào vùng **đã có** audio ⇒ bị frontier (`_last_pts`) lọc bỏ ⇒ audio của vùng thật
(20→41s) **mất hẳn**. Timeline còn `[6.7 → 20.04]` + `[41 → 80]`: một khe hở 21s. Con trỏ khối ASR
chạy tới mép khe (20.04s) rồi kẹt: `available = 0.0s` nên bộ cắt khối không cắt được gì.

**Sửa 1 — chặn ĐẢO TRỤC** (`backend/core/stream_demuxer.py`): nếu lần neo trước vừa xảy ra trong
`_AXIS_REVERSAL_GUARD_SEC` (10s) và lần này có **cùng độ lớn nhưng ngược dấu** thì BỎ QUA (giữ
nguyên trục đang dùng), log `Bỏ qua ĐẢO TRỤC: …`. Thay đổi trục thật (khác độ lớn) vẫn được áp
dụng ngay. Regression test chứng minh: tắt van ⇒ `pts_start` bị kéo lùi 10s và bias về 0; bật van
⇒ giữ đúng trục.

**Sửa 2 — van con trỏ kẹt ở mép khe hở** (`backend/ws/lookahead_handler.py`): điều kiện "kẹt phía
sau" nay dùng **ngưỡng lượng audio còn lại** thay vì `buffered_end_from(...) is None`:

```python
available_at_cursor = frontier - from_pts        # (0.0 nếu không có audio)
if behind < 20s or available_at_cursor >= chunker.min_window_sec:  # đủ để cắt khối ⇒ bình thường
    ...
# ngược lại (chậm ≥20s mà chỉ còn < min_window audio) ⇒ NEO LẠI về vị trí phát
```

Đúng ca log trên: `buffered_end_from(20.04)` trả về ~20.04 (KHÁC `None`), `available = 0.0s`
⇒ trước đây van không kích hoạt, nay kích hoạt và con trỏ nhảy về playhead (41s) — nơi có đủ
39s audio ⇒ phụ đề chạy tiếp (vùng 20–41s đã bị người dùng xem qua thì coi như mất).

### 13.3 Giảm khựng ở phía client

Van an toàn cũ chỉ treo horizon 20s mỗi lần, nên sau khi hết 20s lại pause ⇒ nhịp
"chạy một chút lại dừng" (`pause=9 resume=8`). Nay: **từ lần phá bế tắc thứ HAI trở đi, treo
horizon 180s** (`extension_firefox/content/content-script.js`) để video phát liền mạch thay vì
khựng theo chu kỳ — vẫn log rõ là tạm thời không có phụ đề.

> Ghi chú: `extension_chrome_edge/content/content-script.js` đã được chép đè từ bản Firefox (hai
> file vốn chỉ khác nhau ở chú thích) để hai bản không lệch nhau.

---

## 14. Vòng test #4 trên xvideos.com — "seek thì không có phụ đề nữa"

Lần này khúc đầu **đã ra phụ đề** (tới ~196s, `utterances=70`), nhưng **sau khi tua thì không có
phụ đề nữa** dù backend có audio.

### 14.1 Dấu vết

```text
[WS] Seek seek_id=… mốc=322.07s (RAM=210.0s, đệm trước=0.0s)
[WS] StreamDemuxer reset (epoch=1, min_pts=321.57, init=628B, container=mp4)
[ASR] Playhead: 324.0s | Audio RAM: 235.7s | Đệm trước: 36.0s | Đã dịch: 0.0s (đã qua ASR 0.0s)
   … lặp y hệt trong 35 giây, RAM vẫn tăng (235.7 → 245.7s) …
[BS][Diag][status] playhead=324.99s | mốc-đã-xử-lý=322.07s (đứng im) | fragments=55 decoded=24
```

Backend CÓ audio quanh vị trí phát (`Đệm trước: 36.0s`) nhưng `mốc-đã-xử-lý` đứng đúng ở mốc tua.

### 14.2 Nguyên nhân: con trỏ khối rơi vào KHE HỞ

`handle_seek` neo con trỏ khối đúng vào mốc tua (322,07 s), nhưng audio thật của vị trí mới chỉ
bắt đầu **muộn hơn** (trình phát nạp lại theo ranh giới segment). Vì khe hở lớn hơn
`MAX_GAP_FILL_SEC` (5 s):

* `timeline.buffered_end_from(322.07)` → `None`;
* `LookaheadChunker.next_chunk()` → `return None` ngay dòng đầu;
* `_fast_bootstrap` vẫn `True` nên vòng lặp quay lại liên tục mà **không bao giờ commit** khối nào
  ⇒ `_ready_until_pts` đứng im ở 322,07 s.

Chú ý: `Đệm trước: 36.0s` KHÔNG đo từ con trỏ — `send_status` lấy
`max(buffered_end_pts(), self._decoded_end_pts)` nên nó vẫn báo 36 s dù vùng quanh con trỏ trống.
Đây là lý do log "trông như có audio" mà pipeline vẫn đứng.

### 14.3 Sửa: nhảy con trỏ ra khỏi khe hở

* `ContinuousAudioTimeline.first_audio_pts_at_or_after(pts)` — mốc bắt đầu của đoạn audio đầu tiên
  chứa/ở sau `pts`.
* `LookaheadSessionState._nudge_cursor_out_of_gap(from_pts)` — nếu KHÔNG có audio tại con trỏ thì
  nhảy con trỏ tới đầu đoạn audio kế tiếp, log
  `[SEG_BATCH] Con trỏ khối 322.07s nằm trong KHE HỞ … nhảy tới 330.00s (lệch 7.93s) …` và tăng
  counter `lookahead.batch_cursor_gap_nudged`. Audio trong khe hở không tồn tại nên không mất gì.
* Chỉ nhảy khi KHÔNG có audio tại con trỏ (đang chờ nạp thêm là chuyện bình thường, không nhảy).

### 14.4 Giảm "giật khúc đầu"

Khúc đầu trước đây phải chờ ~25 s (2 lần van an toàn) mới ra phụ đề đầu. Nay khi **chưa có phụ đề
nào** (`utterances=0`), van an toàn dùng ngưỡng riêng `LA_FIRST_STALL_GUARD_MS = 12s` thay vì 25 s.

### 14.5 Kiểm thử

* `test_55`: thêm `test_first_audio_pts_at_or_after_bridges_gap` (kèm khẳng định
  `buffered_end_from` trả `None` ở khe hở > 5 s — tiền đề của lỗi) và
  `test_nudge_cursor_out_of_gap_after_seek` (mô phỏng đúng log: con trỏ 322,07 s, audio 330 s).
* Toàn bộ tuyến lookahead: **81 test pass**; 12/12 file test extension pass.

---

## 15. Vòng test #5 trên xvideos.com — "seek xong thì bị pause liên tục"

Van ở §14 đã chạy (thấy trong log: `Con trỏ khối 313.44s nằm trong KHE HỞ … nhảy tới 320.04s`),
và phụ đề ra được 2 khối sau khi tua (`[320.04→324.04]`, `[324.04→330.03]`). Nhưng rồi:

```text
[ASR] [SEG_GATE] Chờ gom khối dài từ 330.03s: có 0.0s/30.0s audio     ← con trỏ 330.03s, 0.0s audio
[ASR] Playhead: 332.6s | Audio RAM: 134.3s | Đệm trước: 37.4s | Đã dịch: 0.0s   ← rồi đứng im
[BS][Diag][session] pause=10, resume=7, lý do dừng={"start":1,"underrun":3,"seek":6}
```

### 15.1 Nguyên nhân: "MẨU VỤN rồi khe hở" — van cũ không bắt được

Con trỏ ở 330,03 s có **đúng 0,0 s audio** tại chỗ (mép cuối một đoạn) rồi mới tới khe hở và đoạn
kế. Lúc này `buffered_end_from(330.03)` trả về ~330,03 (**KHÁC** `None`) nên van "khe hở hoàn toàn"
ở §14 KHÔNG kích hoạt, trong khi `next_chunk()` vẫn trả `None` vì
`buffered_end <= from_pts + 0.05`. Kết quả: `mốc-đã-xử-lý` đứng im ở 330,03 s và client
pause/resume liên tục — dù RAM đã có 134 s audio và `Đệm trước: 37 s`.

### 15.2 Sửa: tổng quát hoá van thành "không đủ audio để cắt khối"

`_nudge_cursor_out_of_gap()` nay xét **lượng audio đọc được tại con trỏ** thay vì chỉ xét
`None`:

```python
available = buffered_end_from(from_pts) - from_pts      # 0.0 nếu không có audio
if available >= _MIN_USEFUL_AUDIO_SEC:                  # 2.5 s — mức tối thiểu bộ cắt khối cần
    return False                                        # vẫn cắt được khối ⇒ không nhảy
nxt = first_audio_pts_at_or_after(from_pts + max(0.05, available) + 0.05)
if nxt is None:                                         # đang ở MÉP NẠP ⇒ chờ, KHÔNG nhảy
    return False
# … nhảy con trỏ tới `nxt`
```

Hai điều kiện then chốt:
* **Dò từ NGAY SAU phần audio hiện có** — nếu phía sau không còn đoạn nào thì đây là mép nạp
  (đang chờ dữ liệu mới), nhảy sẽ làm MẤT audio ⇒ không nhảy.
* **`_feed_pts` KHÔNG được đẩy theo.** `_feed_pts` nghĩa là "audio ĐÃ QUA ASR"; vùng bị nhảy qua
  không có audio nên chưa xử lý. Bản nháp đầu đẩy theo và log cho thấy hệ quả ngay:
  `prebuffer_ready=True (ready_ahead=0.00s, fed_ahead=29.85s …)` — client tưởng vùng đó đã quét
  xong nên resume sớm.

### 15.3 Giảm pause/resume ở phía client

Nhánh "backend nhận mảnh mà không ra PCM" (§11.3) trước đây đòi `stalledFrag < 20s`; log lần này
có `mảnh mới cách đây 21s` nên **trượt đúng 1 giây**. Nay điều kiện là
`stalledReady > 20s && stalledFrag <= stalledReady` (mốc đã xử lý kẹt lâu hơn hẳn mảnh mới) ⇒
nhánh này bắt được cả ca "mảnh cũng chậm" và treo horizon 180 s để hết pause theo chu kỳ.

### 15.4 Kiểm thử

* `test_55`: thêm `test_nudge_cursor_out_of_sliver_then_hole` (mẩu vụn 0,1 s tại 330,0 rồi khe hở
  9,9 s tới 340,0 ⇒ nhảy đúng tới 340,0) và `test_nudge_does_not_jump_at_live_edge` (ở mép nạp
  thì KHÔNG nhảy).
* Tuyến lookahead: **83 test pass**; 12/12 file test extension pass.

### 15.5 Điều còn đáng ngờ (cần log vòng sau)

Ở đầu phiên, van đã nhảy từ 9,69 s tới **40,03 s** (lệch 30,34 s) — nghĩa là audio của 9,7→40 s
KHÔNG có trong timeline dù extension đã gửi mảnh. Cần xem vòng log sau: nếu vẫn lặp lại thì phần
audio đó đang bị mất ở tầng giải mã/frontier (không phải ở tầng cắt khối), và cần lọc log backend
`Đường giải mã nối-liền không ra PCM`, `LỆCH TRỤC THỜI GIAN`, `_frames_skipped`.

---

## 16. www.av01.media — "khúc đầu có phụ đề, về sau không hiện nữa"

Trang này cũng là **MUXED** (`video/mp4;codecs=mp4a.40.2,avc1.64001E`, 1 SourceBuffer), và van
nhận-MUXED hoạt động tốt: cổng Pipeline B ĐẠT, `acceptMuxedAudio=true`.

### 16.1 Dấu vết: pipeline "khoẻ" nhưng không ra câu nào

```text
[ASR] Lookahead Batch ASR [402.82s -> 431.99s]: 7 phụ đề            ← câu cuối cùng (22:45:29)
[ASR] [SEG_GATE] Chờ gom khối dài từ 431.99s: có 18.5s/30.0s audio
[ASR] [SEG_GATE] Chờ gom khối dài từ 460.56s: có  7.9s/30.0s audio   ← con trỏ ĐÃ TIẾN...
[ASR] [SEG_GATE] Chờ gom khối dài từ 483.94s …  503.24s …  539.65s …  559.65s …  588.56s …  608.23s
```

Backend **không** có dòng `Lookahead Batch ASR` / `Ngắt câu khối` nào sau 431,99 s, nhưng con trỏ
khối vẫn tiến đều 431,99 → 608,23 s, `Audio RAM` vẫn tăng 177 → 348 s, `Trục PCM khớp trục media`
vẫn in đều ⇒ audio VẪN được giải mã và neo đúng trục. Client cùng lúc:

```text
[BS][Diag][status] playhead=391.33s | ready=40.66s fed=40.66s | mốc-đã-xử-lý=431.99s | utterances=14
[BS][Diag][status] playhead=555.00s | ready=53.23s fed=53.23s | mốc-đã-xử-lý=608.23s | utterances=14
```

`utterances` **đứng im ở 14** suốt 160 giây trong khi `ready_ahead` 30–66 s (trông rất khoẻ).

### 16.2 Nút cổ chai: nhánh "ASR trả về RỖNG" không có log nào

`_offline_batch_loop` có nhánh:

```python
if not clean_text:
    # Không nhận diện được tiếng nói (im lặng hoặc nhạc nền)
    self._commit_batch_progress(seq, chunk.next_read_pts, chunk.pts_end)   # con trỏ + marker VẪN tiến
    await self.send_status()
    continue
```

Khối vẫn được cắt, `_ready_until_pts` vẫn tiến (nên `ready_ahead` đẹp), mà **không ghi log gì** ⇒
nhìn từ ngoài y hệt "pipeline chạy tốt nhưng phụ đề biến mất". Cùng lúc, bộ cắt khối báo
`mode=vad_silence, silence=23168ms` / `18912ms` — dấu hiệu PCM phần lớn là im lặng.

### 16.3 Đã thêm chẩn đoán (chưa kết luận được thủ phạm)

`LookaheadSessionState._log_empty_block()` — mỗi khối rỗng (throttle 5 s) in:

```text
[SEG_BATCH] Khối [432.00s → 460.56s] ASR trả về RỖNG
   (PCM 28.6s | RMS=0.00000 | đỉnh=0.00000 | mẫu-0=100.0%; mode=vad_silence; silence=18912ms;
    im lặng-do-lấp=0.0s; playhead=389.4s) — KHÔNG có phụ đề cho đoạn này.
```

Ba chỉ số phân biệt ngay nguyên nhân:

| Dấu hiệu | Kết luận |
|---|---|
| `RMS`/`đỉnh` ≈ 0, `mẫu-0` ≈ 100 % | PCM là **im lặng thật** ⇒ lỗi tầng giải mã/ghép nối (demuxer), không phải ASR |
| `RMS` bình thường mà vẫn rỗng | **model/VAD** không nhận ra (không phải audio) |
| `im lặng-do-lấp` lớn | timeline đang **thiếu audio** và `get_audio_range/read` tự lấp silence |

Dòng trạng thái 10 s nay in thêm `im lặng-do-lấp Xs, câu N` để theo dõi cùng lúc.

> Giả thuyết ưu tiên (cần log mới để chốt): audio giải mã từ các mảnh MUXED **mới** là im lặng,
> trong khi audio **replay từ cache** ngay sau khi tua (374→432 s) lại ra chữ bình thường. Nếu
> đúng thì lỗi nằm ở đường giải mã mảnh MUXED (AV1+AAC) chứ không phải ở tầng cắt khối/ASR.

### 16.4 Kiểm thử

504 test backend pass; tuyến lookahead 83 test pass (không đổi hành vi, chỉ thêm log).

---

## 17. Lọc ảo giác lặp từ: NGẮT CÂU TRƯỚC, chỉ bỏ CÂU hỏng (yêu cầu người dùng 2026-10-05)

### 17.1 Hành vi cũ

```text
[ASR] Lookahead Batch ASR phát hiện ảo giác kẹt vòng (repetition loop), bỏ qua khối:
      'ああああああああ…'                                    ⇒ ĐÚNG
[ASR] Lookahead Batch ASR phát hiện ảo giác kẹt vòng (repetition loop), bỏ qua khối:
      '一時しないと収まんないよこれ。ちょっと入れるだけだから。ちょっと入れちょっとだから。
       ちょっとちょっと。あああああ…'                        ⇒ SAI: mất luôn 4 câu thật
```

`is_repetition_hallucination()` trả `True` khi có **bất kỳ** đoạn lặp ≥ `min_reps` ở đâu trong chuỗi.
Bản cũ gọi nó trên **toàn bộ `clean_text`** nên chỉ cần ĐUÔI khối kẹt vòng là vứt cả khối — kể cả
khi đầu khối có nhiều câu hoàn toàn bình thường.

### 17.2 Nguyên tắc mới (và lý do KHÔNG thêm hàm tách câu mới)

> "Ngắt câu khối trước, câu nào repetition loop thì bỏ câu đó thôi."

Việc **ngắt câu đã có sẵn** trong chính tuyến này, chạy sau khi align:

```
group_words_to_subtitles      (gom từ thành câu theo dấu câu của văn bản ASR)
  → split_subtitles_by_clause_comma   (tách vế câu tại '、'/',' )
  → split_oversized_sentences         (lưới an toàn theo trần ký tự/từ)
```

và bộ lọc ảo giác **theo TỪNG CÂU** cũng đã có sẵn ở vòng `candidate_subs` (dòng ~1450). Vấn đề
chỉ là **bộ lọc cả khối nằm TRƯỚC** nên chặn luôn đường đi tới đó.

Vì vậy cách sửa đúng là **XOÁ bộ lọc cả khối**, không thêm gì mới:

```diff
-            if is_repetition_hallucination(clean_text, min_reps=5):
-                logger.warning("… bỏ qua khối: …")
-                self._commit_batch_progress(...); await self.send_status(); continue
+            # (không lọc ở đây — xem khối chú thích; lọc theo TỪNG CÂU ở `candidate_subs`)
```

(Trước đó tôi có thêm một hàm tách câu thô chỉ để "chặn sớm khối rác thuần" — **đã gỡ bỏ**: nó
trùng chức năng với các bộ tách câu đã có và tạo thêm một định nghĩa "câu" thứ hai không cần thiết.
Khối rác thuần vẫn ra 0 phụ đề vì câu duy nhất của nó bị lọc; chi phí thêm chỉ là một lượt aligner.)

Bộ lọc theo câu nay log ở mức `WARNING` để nhìn thấy:

```
[SEG_BATCH] Bỏ 1 CÂU ảo giác lặp từ (các câu khác trong khối vẫn giữ): 'あああああ…'
```

Kiểm chứng trên chính chuỗi trong log:

| Đầu vào | `is_repetition_hallucination` cả khối (logic cũ) | Từng câu (logic mới) |
|---|---|---|
| `あああ…` (30 ký tự) | `True` ⇒ bỏ khối | 1 câu `True` ⇒ bỏ câu ⇒ 0 phụ đề ✔ |
| `…4 câu thật…ちょっとちょっと。あああああ…` | `True` ⇒ **bỏ nhầm cả khối** ✗ | `[False, False, False, True]` ⇒ giữ 3-4 câu thật ✔ |

### 17.3 Kiểm thử

`test_58`: thay test giả bằng **2 test chạy đúng vòng lặp batch thật** (`_offline_batch_loop`),
khoá đúng HÀNH VI người dùng thấy:

* `test_mixed_repetition_tail_keeps_the_real_sentences` — ASR trả chuỗi lẫn lộn ⇒ `lookahead_subtitles`
  phải có câu thật (`収まんない…`) và **không** có câu `ああああ…`. Test này FAIL với bản cũ (không có
  bản tin nào được gửi).
* `test_pure_repetition_block_emits_nothing` — khối rác thuần ⇒ không gửi phụ đề nào, nhưng con trỏ
  vẫn tiến (không kẹt).

Toàn bộ backend: **505 test pass**.

---

## 18. YouTube: nhịp pause/play 1–2 s dù "Lookahead available > 60s" (2026-10-05)

### 18.1 Hai đại lượng khác nhau bị nhầm là một

```text
POPUP:  Lookahead available +60s                        ← ĐỆM VIDEO
[BS][Diag][status] playhead=4.04s | ready=0.00s … | mốc-đã-xử-lý=4.04s   ← ĐỆM ĐÃ DỊCH = 0
[BS][Diag][session] ▶️ PHÁT TIẾP (lần 1, lý do=prebuffer_ready, đã dừng 2.2s)
[BS][Diag][status] playhead=6.69s | ready=0.32s …   ← vừa chạy đã hết đệm đã dịch
[BS][Diag][session] ▶️ PHÁT TIẾP (lần 2, lý do=prebuffer_ready, đã dừng 1.5s)
[BS][Diag][status] playhead=15.28s | ready=22.44s …  ← từ đây mới thật sự dư đệm
```

`Lookahead available +60s` là **đệm VIDEO** (`video.buffered` phía trước, interceptor đọc). Việc
tạm dừng để nạp đệm lại dựa trên **đệm đã dịch** (`ready_ahead` của backend) — lúc đầu chỉ
0,3–3 s. Vì vậy "available > 60s" hoàn toàn không mâu thuẫn với việc video bị tạm dừng.

### 18.2 Vì sao thành nhịp 1–2 s

Client cũ cho video chạy lại ngay khi:

* `prebuffer_ready` của backend — nhưng mục tiêu của backend có lúc chỉ **2,5 s** (bootstrap), hoặc
* `ready_ahead >= 1,5 s` (TTS tắt) — quá mỏng so với nhịp backend đẩy khối (20–30 s),

⇒ chạy được 1–2 s là playhead chạm `ready_until_pts` ⇒ `_tick()` tạm dừng lại
(`lý do dừng={"start":1,"underrun":2}` trong log) ⇒ lại chờ ⇒ lại chạy… Đệm video dồi dào không
giúp gì cho vòng này.

### 18.3 Sửa: chỉ chạy lại khi ĐỦ ĐỆM ĐÃ DỊCH

`extension_firefox/content/content-script.js`:

| Thay đổi | Nội dung |
|---|---|
| `LA_MIN_RESUME_AHEAD_SEC = 4.0` | Ngưỡng đệm đã dịch tối thiểu để cho video chạy lại |
| `currentReadyAhead()` | Đệm đã dịch hiện có, **đã trừ thời gian trôi** kể từ lúc backend báo (không dùng số cũ) |
| `hasEnoughHeadroomToResume()` | `ready ≥ 4s`, **hoặc** đoạn trước là khoảng lặng (`fed ≥ 4s && ready ≈ 0`) — không có gì để dịch thì chờ thêm là vô ích |
| Mọi đường resume đều qua cổng này | `onStatus`, `onPrebufferReady` (chỉ còn là lưới an toàn khi backend không gửi `ready_ahead`), `onSubtitles("subtitle_arrived")`, `onTts("tts_first_ready")`, và `onBufferingStateChange` (lưới an toàn 8 s của timeline) |
| Gia hạn thông minh hơn | Lúc bắt đầu/tua: nếu đệm đã dịch còn dưới ngưỡng **và backend vẫn đang nhận mảnh** (`fragments` mới trong 2 s) thì gia hạn thêm (tối đa 3 × 2 s) — chờ một lần cho đủ, thay vì chạy–dừng liên tục |
| Log rõ lý do chờ | `⏳ Chưa cho phát: đệm đã dịch 1.2s < 4.0s (prebuffer_ready=true, ready=1.2s, fed=1.2s) — chờ thêm để tránh nhịp pause/play ngắn.` |

**Không thể treo trình phát**: các `resumeTimer` theo đồng hồ thực (3,5–12 s, gia hạn tối đa 3 lần)
vẫn là chốt an toàn cuối cùng, nên dù backend không bao giờ đạt 4 s thì video vẫn phát.

### 18.4 POPUP nói rõ cả hai con số

`Lookahead available +60.00s · đã dịch +3.4s (tạm dừng nạp đệm)`

`lookaheadSession` (đệm đã dịch, số lần pause/resume) nay được content script gửi kèm trong
`GET_STATUS` để POPUP hiển thị — tránh hiểu nhầm "available > 60s mà vẫn pause".

### 18.5 Kiểm thử

12/12 file test extension pass; 91 test backend đọc file extension pass; `demo_client_sync.js` pass.
(Phần này là logic định thời trong content script — cần bạn xác nhận trên YouTube.)

---

## 19. Video phát HẾT nhưng phiên vẫn sống (ảnh người dùng 2026-10-05)

Hiện tượng: video chạy hết, trình phát hiện lưới "video kế tiếp" nhưng overlay của extension vẫn
nằm giữa màn hình (`Đang xử lý tiếp đoạn video…`), POPUP vẫn ở trạng thái đang dịch, backend vẫn
giữ RAM/GPU. Yêu cầu: **hết video thì extension tự tắt session**.

### 19.1 Sửa

| Chỗ | Thay đổi |
|---|---|
| `content/content-script.js` — `startCapture()` | `video.addEventListener("timeupdate", maybeStopNearEnd)` + `addEventListener("ended", …)`: **còn ≤ `LA_END_STOP_MARGIN_SEC` (3 s) là hết** thì `stopCapture()` + `overlayManager.destroy()`. Đặt ở `startCapture` (dùng chung `captureAbortController.signal` nên tự gỡ khi Stop) ⇒ áp dụng cho **cả Pipeline A và B**. |
| `content/content-script.js` — `cleanup()` | `if (v && v.paused) v.play()` → `if (v && v.paused && !v.ended) v.play()`. Gọi `play()` trên video đã `ended` sẽ khiến nó **CHẠY LẠI TỪ ĐẦU** — đúng lúc ta vừa tự đóng phiên. |
| `popup/popup.js` — `refreshLookaheadStatus()` | Thêm nhánh `!status.isCapturing && isCapturingNow` ⇒ `setUI(false)`: phiên đã tự kết thúc ở phía trang thì POPUP trả nút về "Bắt đầu" và mở khoá các nhóm cài đặt. |

Vì sao dùng `timeupdate` chứ không chỉ `ended`: nếu extension đang **tạm dừng video sát mép cuối**
(hoặc trình phát bị can thiệp nên không bắn `ended`) thì phiên sẽ "mồ côi" mãi. Điều kiện:

```js
const d = Number(video.duration), t = Number(video.currentTime);
if (!Number.isFinite(d) || d < LA_END_STOP_MIN_DURATION_SEC) return;  // livestream / clip quá ngắn (15s)
if (t >= d - LA_END_STOP_MARGIN_SEC) stopAtVideoEnd(`sắp hết (còn ${d - t}s / ${d}s)`);
```

Log khi đó: `⏹️ Video sắp hết (còn 2.6s / 112.7s) — tự dừng phiên dịch và gỡ overlay…`

`stopCapture()` → `cleanup()` → `lookaheadClient.disconnect()` gửi `{"type":"stop"}` ⇒ backend đóng
phiên và giải phóng RAM/GPU (log `[CORE] Đã giải phóng bộ nhớ RAM …`) — tức "tắt session" thật sự,
không chỉ ẩn overlay. 3 giây cuối không có phụ đề, đổi lại không bao giờ còn phiên treo.

### 19.2 Kiểm thử

`test_59`: `test_extension_auto_stops_session_when_video_ends` (chạy cho CẢ hai bản extension) khoá:
có ngưỡng `LA_END_STOP_MARGIN_SEC` + theo dõi `timeupdate` + `ended`; điều kiện `t >= d - margin`
gọi `stopCapture()` và `overlayManager.destroy()`; `cleanup()` không `play()` video đã `ended`;
POPUP có nhánh phát hiện phiên đã kết thúc.

Toàn bộ backend: **508 test pass**; 12/12 file test extension pass.

---

## 20. Đồng bộ sang `extension_chrome_edge` (2026-10-05)

Sau khi bản Firefox chạy ổn qua các vòng test thực tế, `extension_chrome_edge` đã được cập nhật
đầy đủ. Trạng thái sau khi đồng bộ — **chỉ còn 4 file khác nhau, tất cả đều CÓ CHỦ Ý**:

| File | Vì sao khác |
|---|---|
| `manifest.json` | Firefox cần `browser_specific_settings.gecko`, `background.scripts` (kèm `lib/backpressure-gate.js`) và `match_origin_as_fallback`; Chrome dùng `background.service_worker` |
| `background/service-worker.js` | Chromium MV3 chỉ cho một `service_worker` ⇒ phải `importScripts("../lib/backpressure-gate.js", …)`; Firefox nạp bằng `background.scripts` |
| `lib/ws-client.js` | `useBridge`: Firefox dùng Background Bridge (`chrome.runtime.connect`), Chrome/Edge mở WebSocket trực tiếp (service worker bị kill sau ~30s idle, không kế thừa SSL exception của trang) |
| `lib/lookahead-client.js` | Giống hệt bản Firefox **trừ đúng** khối `useBridge` nói trên (`: false` cho Chrome) |

Mọi file còn lại **giống hệt** giữa hai bản, gồm toàn bộ việc đã làm trong phiên này:
`content/buffer_interceptor_poc.js` (MUXED, sniff codec, `isInitSegment` fMP4, nhật ký append,
xoá cache do đổi nguồn), `content/content-script.js` (log cổng Pipeline B, van an toàn chống bế tắc,
ngưỡng đệm đã dịch 4s để chống nhịp pause/play, tự tắt phiên khi video còn ≤3s là hết),
`lib/lookahead-diagnostics.js`, `lib/lookahead-timeline.js` (chống bão seek), `popup/*` (hiện mã lý do
+ đệm đã dịch, tự trả UI về trạng thái rảnh khi phiên kết thúc) và `tests/*`.

Kiểm thử sau đồng bộ: **508 test backend pass**, **108 test backend đọc file extension pass**,
**12/12 file test extension pass** (cả hai bản), `demo_client_sync.js` pass.

---

## 21. "Ngắt câu khối" chẻ sai ở từ viết tắt (`Dr.`) — 2026-10-05

### 21.1 Hiện tượng

```text
[ASR] Lookahead Batch ASR [37.47s -> 61.36s]: 'Fine, Dr. Kuthrapali. … I present Dr. Milstone from MIT. …'
[ASR] [SEG_BATCH] Ngắt câu khối [37.47s -> 61.36s] (15 phụ đề, align=English):
      "Fine, Dr." ⏐ "Kuthrapali." ⏐ "Thank you, sir." ⏐ … ⏐ "Right on time, Dr." ⏐ "Kuthrapali."
      ⏐ "I present Dr." ⏐ "Milstone from MIT." ⏐ …(+5)
```

Dấu `.` của **từ viết tắt** bị coi là kết câu ⇒ phụ đề vụn và bản dịch sai (dịch "Dr." thành một câu).

### 21.2 Nguyên nhân

Hai tầng ngắt câu đều chỉ loại trừ **số thập phân** (`3.9`) và **tên miền/từ viết liền**
(`domain.com`), **không** biết từ viết tắt:

* `_split_text_by_sentence()` — lưới an toàn theo trần ký tự;
* `group_words_to_subtitles()._is_sent_end()` — tầng ngắt câu CHÍNH (dấu câu do
  `merge_source_text` gắn lại từ văn bản ASR).

### 21.3 Sửa (`backend/asr/forced_aligner.py`)

Thêm bộ nhận diện dùng chung rồi áp vào **cả hai** tầng:

```python
_ABBREVIATIONS_CI = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "mt", "vs", "etc",
                     "inc", "ltd", "corp", "fig", "vol", "jan"…"dec", "mon"…"sun", "phd", "md", …}

def _ends_with_abbreviation(text):
    word = _word_before_trailing_period(text)
    if word.lower() in _ABBREVIATIONS_CI:      # Dr. / Mr. / Prof. / St. / Ph.D. …
        return True
    return len(word) == 1 and word.isupper()   # chữ cái đầu: J. / A. / U.S. / A.B.
```

Hai điều kiện an toàn:
* **Chỉ bỏ qua khi PHÍA SAU còn nội dung** — viết tắt nằm ở cuối khối thì dấu chấm đó vẫn kết câu;
* Danh sách **cố ý bỏ các từ mơ hồ** hay gặp ở dạng từ thường (`no`, `am`, `pm`, `us`) để không
  gộp nhầm câu — đã kiểm: `MIT.`, `late.`, `sir.`, `No.`, `no.`, `ok.` vẫn kết câu bình thường.

Kết quả trên chính chuỗi trong log: **13 mảnh → 9 câu**, và
`'Fine, Dr. Kuthrapali.'`, `'I present Dr. Milstone from MIT.'`, `'It's nice to meet you, Dr. Kuthrapali.'`
đều nguyên vẹn.

### 21.4 Kiểm thử

`test_60`: thêm `test_abbreviations_do_not_split_sentences` với **nguyên văn câu trong log** (cả
tầng `group_words_to_subtitles` lẫn `split_oversized_sentences`). Đã kiểm chứng test này **FAIL khi
tắt bản sửa** (`còn mảnh cụt 'Dr.': ['Fine, Dr.', 'Kuthrapali.', …]`) và **PASS khi bật**.

Toàn bộ backend: **509 test pass**.
