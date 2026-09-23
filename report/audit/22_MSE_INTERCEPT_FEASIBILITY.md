# 22 — NGHIÊN CỨU KHẢ THI: CHẶN BẮT PHÂN ĐOẠN MEDIA (MSE) ĐỂ CHẠY PHỤ ĐỀ TRƯỚC KHI PHÁT

> **Loại tài liệu:** nghiên cứu khả thi (feasibility study) + thiết kế. **KHÔNG** chứa mã sản phẩm, **KHÔNG** sửa file nào khác ngoài chính file này.
> **Ngày:** 2026-02 · **Repo:** `vibe-translation-addon-transcribe_cpp`
> **Đề xuất được đánh giá:** tiêm script vào MAIN world của trang, monkey-patch `MediaSource.prototype.addSourceBuffer` + `SourceBuffer.prototype.appendBuffer`, sao chép `ArrayBuffer` mỗi lần append, demux, lấy track audio, giữ phần có timestamp **trước** `video.currentTime`, chạy ASR/dịch trước ⇒ mục tiêu "~0 ms độ trễ cảm nhận".

**Quy ước nhãn bằng chứng** (dùng xuyên suốt tài liệu):

| Nhãn | Nghĩa |
| :--- | :--- |
| `[VERIFIED-REPO]` | Đã đọc trực tiếp file trong repo này (kèm `file:line`). |
| `[VERIFIED-NUM]` | Số đo đã có trong `report/` **của chính dự án này** (kèm file nguồn). |
| `[VERIFIED-EXT]` | Khẳng định bên ngoài, có URL nguồn. |
| `[NOT VERIFIED]` | Chưa kiểm chứng được — **không** được dùng làm căn cứ quyết định. |

Giới hạn công cụ của phiên làm việc này: `web_fetch` **thất bại** (lỗi phân giải DNS), nên mọi trích dẫn bên ngoài chỉ dựa trên **kết quả tìm kiếm** (tiêu đề + URL), không đọc được toàn văn đặc tả. Vì vậy các chi tiết đặc tả MSE/EME dưới đây được đánh dấu `[VERIFIED-EXT]` ở mức **"tồn tại nguồn nói về điều này"**, còn các con số/ngữ nghĩa chính xác phải được xác nhận lại bằng cách đọc đặc tả hoặc bằng thực nghiệm M1.

---

## 0. KẾT LUẬN ĐIỀU HÀNH

1. **Cơ chế là đúng về nguyên lý cho một tập con nội dung, nhưng giả định trung tâm của đề xuất — "chạy ASR trước để có ~0 ms độ trễ" — chỉ đúng ở một chế độ duy nhất: VOD không DRM có buffer dẫn trước.**
2. **"~0 ms độ trễ cảm nhận" không đạt được như một tính chất chung.** Nó chỉ đạt được khi *thời gian dẫn trước của buffer* (`buffered.end − currentTime`) lớn hơn *toàn bộ chi phí pipeline*. Với VOD điển hình (dẫn trước 10–30 s) thì đạt; với live ở mép (dẫn trước ≈ 0) thì **không thể** — và đó chính là chế độ mà phụ đề trễ gây khó chịu nhất.
3. **Đề xuất không loại bỏ được độ trễ cốt lõi của ASR.** ASR chỉ chốt câu khi đã thấy đuôi im lặng / dấu câu; khoản này (`~350–700 ms`) nằm trong *bản thân* thuật toán, không nằm ở đường capture. MSE chỉ **dịch khoản đó ra trước theo thời gian phát**.
4. **Có một biến thể rẻ hơn nhiều so với demux thủ công:** nạp phân đoạn vào một `HTMLMediaElement` ẩn, đưa qua `MediaElementSource → AudioWorklet` để lấy PCM 16 kHz, rồi **tái dùng nguyên xi** đường WebSocket/`audio_frame` hiện có. Việc này bỏ hẳn `mp4box.js`/`mux.js`/demux WebM khỏi đường găng. Chi tiết ở §7.4.
5. **DRM là rào cứng.** Byte tại `appendBuffer` của nội dung CENC **vẫn là byte mã hoá**; hook ở page world không thể giải mã vì không có CDM. Netflix/Disney+/Prime Video ⇒ đề xuất **không hoạt động**. YouTube cần thực nghiệm M1 mới kết luận được.
6. **Đề xuất có giá trị thật, nhưng giá trị đó là "đổi ngân sách độ trễ", không phải "xoá độ trễ":** nó biến độ trễ từ *phụ thuộc độ trễ suy luận* thành *phụ thuộc độ dẫn trước của buffer*. Kèm theo là tăng mạnh độ phức tạp và số chế độ hỏng (seek, ABR, quảng cáo, live, DRM).

**Bảng phán quyết theo lớp nội dung** (chi tiết ở §4):

| Lớp nội dung | Chặn bắt được? | "~0 ms" khả thi? | Ghi chú |
| :--- | :--- | :--- | :--- |
| MP4 progressive (`src=` trực tiếp) | ❌ Không có MSE để hook | — | Đường capture hiện tại vẫn là phương án duy nhất |
| MSE không DRM (YouTube free, HLS/DASH thường) | ✅ Có | ⚠️ Tuỳ thời gian dẫn trước | Trường hợp mục tiêu của đề xuất |
| MSE + EME/CENC (Netflix, Disney+, Prime) | ⚠️ Thấy byte, **byte đã mã hoá** | ❌ Không | Rào pháp lý + kỹ thuật |
| File cục bộ (`file://` / blob) | ❌ Không có MSE | — | Có thể đọc trực tiếp file — dễ hơn nhiều, nhưng khác bài toán |
| Live (LL-HLS / DASH low-latency) | ✅ Có | ❌ Không (dẫn trước ≈ 0) | Xấu hơn cả đường capture hiện tại về mặt kiến trúc |

---

## 1. HIỆN TRẠNG ĐÃ KIỂM CHỨNG: ĐƯỜNG CAPTURE VÀ SỐ ĐO CỦA CHÍNH DỰ ÁN

Phần này là **cơ sở để so sánh độ trễ ở §5**. Mọi số ở đây lấy từ `report/` và từ code đã đọc.

### 1.1 API path thực tế đang dùng `[VERIFIED-REPO]`

| Câu hỏi | Trả lời | Bằng chứng |
| :--- | :--- | :--- |
| `tabCapture`? | **Không.** Không có `tabCapture` trong manifest (permission chỉ có `activeTab`, `storage`, `scripting`) | `extension_firefox/manifest.json:16-20` |
| `getUserMedia`? | **Không** ở đường chính | `lib/audio-capture.js` không có `getUserMedia` |
| `MediaRecorder`? | **Không** — PCM thô, không nén | `lib/audio-capture.js:327-341`, `lib/audio-processor.js` |
| WebAudio? | **Có, và là đường chính.** `createMediaElementSource(video)` → `GainNode` (nhánh nghe, phục vụ ducking) + `AudioWorkletNode("audio-capture-processor")` (nhánh ASR) | `lib/audio-capture.js:87-105`, `:180-245` |
| Dự phòng? | `captureStream()/mozCaptureStream()`; nếu worklet chết thì rơi về `ScriptProcessorNode(4096)` trên main thread | `lib/audio-capture.js:120-140`, `:248-269`, watchdog 2,5 s ở `:230-235` |

### 1.2 Định dạng PCM, nhịp chunk, đường truyền `[VERIFIED-REPO]`

| Thuộc tính | Giá trị | Bằng chứng |
| :--- | :--- | :--- |
| Định dạng | **PCM 16-bit mono, 16 000 Hz** | `lib/audio-capture.js:12`, `:327-341`; `lib/audio-processor.js` (`DEFAULT_TARGET_RATE = 16000`) |
| Resample | Trong worklet, từ rate của `AudioContext` (thường 48 kHz) → 16 kHz; có boxcar `ratio`-tap khi tỉ lệ nguyên ≥ 2 | `lib/audio-processor.js` (`_resample`, `_useBoxcar`) |
| `chunkSize` | **1024 mẫu = 64 ms** @16 kHz | `lib/audio-capture.js:14-15`; `lib/audio-processor.js` (`DEFAULT_CHUNK_SIZE = 1024`) |
| Nhịp | ~15,6 chunk/s, đều 64 ms (worklet gom đủ 1024 rồi `postMessage` kèm transferable) | `lib/audio-processor.js` (`port.postMessage(..., [this.buffer.buffer])`) |
| Timestamp | `captureTimestamp = firstBlockTime + emittedSamples/16000` (giây, theo timeline `AudioContext`) | `lib/audio-processor.js` |
| Đóng gói | `[uint32 LE = len(header JSON)][header JSON][PCM bytes]` | `lib/frame-builder.js:33-42`; khớp `backend/ws/protocol.py:29-48` |
| Đường tới backend | content script → `port.postMessage` (bridge) → service worker dựng lại packet → `wss://localhost:8765/ws` | `lib/ws-client.js:289-296`; `background/service-worker.js:239-254` |

**Điểm quan trọng cho đề xuất MSE:** `ws-client.js:291-295` gọi `port.postMessage` **không có transfer list** ⇒ `pcmBuffer` bị **structured-clone** (copy) sang service worker, rồi `buildBinaryAudioPacket` lại copy lần nữa (`frame-builder.js:33-40`). Với 2 KB/chunk thì không đáng kể; nhưng cùng khuôn mẫu này áp cho `ArrayBuffer` phân đoạn media (0,5–2 MB) sẽ là **~1–3 ms copy mỗi MB trên main thread của trang** `[NOT VERIFIED]` (suy luận từ chi phí structured clone, cần đo ở M1).

### 1.3 Số đo độ trễ đã có trong `report/` `[VERIFIED-NUM]`

**a) `metrics_report.json` (phiên thật, ~1785 s uptime, `qwen3-asr-0.6b`, backend CUDA, `preview_window_sec=15.0`, `poll_interval_ms=300`):**

| Chỉ số | p50 | avg | p95 | Ý nghĩa |
| :--- | ---: | ---: | ---: | :--- |
| `asr.infer_core_ms` | 74,7 ms | 83,5 ms | 146,5 ms | Lõi suy luận ASR mỗi lần chạy |
| `asr.preview_ms` | 75,5 ms | 84,4 ms | 148,8 ms | Một lượt preview |
| `asr.commit_ms` | 72,5 ms | 78,4 ms | 133,5 ms | Lượt suy luận chốt câu |
| `asr.e2e_commit_ms` | **217,6 ms** | 236,9 ms | **468,4 ms** | Từ lúc nhận audio chốt → có chữ |
| `asr.first_preview_ms` | **431,2 ms** | 439,0 ms | 629,4 ms | Từ lúc bắt đầu nói → preview đầu tiên |
| `translation.first_token_ms` | **31,4 ms** | 34,3 ms | 51,0 ms | Token dịch đầu |
| `translation.infer_ms` | **281 ms** | 316,4 ms | 597,3 ms | Dịch trọn câu |
| `pipeline.e2e_asr_to_sub_ms` | **282,5 ms** | **317,8 ms** | 598,7 ms | Chốt ASR → phụ đề đã dịch tới client |
| `vad.chunk_ms` | 17,1 ms | 17,5 ms | 23,3 ms | VAD mỗi frame |
| `ws.send_ms` | 0,14 ms | 0,14 ms | 0,22 ms | Không phải nút thắt |

**b) `report/audit/16_latency_after_f39.json`** (harness 4× speed — **cảnh báo**: chính `report/audit/BAO_CAO_AUDIT_HIEU_NANG_QWEN.md:393` ghi rõ con số 4× **không dùng để suy ra realtime**): `asr.preview_median_ms` 122,6 ms; `asr.commit_ms` p50 165,1 ms cho 6,22 s audio ⇒ RTF ≈ 0,027.

**c) `report/audit/05_measurements_and_status.md`** (tầng B, Vulkan, audio nói liên tục 20 s):

| Chỉ số | Giá trị | Nguồn |
| :--- | :--- | :--- |
| `asr.commit_ms` | p50 **106,9 ms**, p95 168,7 ms | `:117` |
| RTF commit (câu 4,5 s) | **≈ 0,024** | `:117` |
| `asr.preview_ms` | p50 87,2 ms, **p95 1450 ms**, max 2941 ms (đuôi cao, chưa giải quyết) | `:118`, §4.1 |
| `vad.chunk_ms` | p50 3,2 ms / frame 25 ms ≈ 13 % một nhân | `:120` |
| nút thắt e2e | **dịch ~430 ms**, không phải ASR ~100 ms | `report/audit/XAC_NHAN_GEMINI_VA_KE_HOACH_TOI_UU.md:347-349, 369` |

**d) Client, `report/audit/01_phu_luc_extension_firefox.md:43-52`:** ngân sách `~175–240 ms` (kịch bản ScriptProcessor cũ ở 48 kHz) — **con số này đã lỗi thời**: code hiện tại (`audio-capture.js:157-161`, `:180-245`) đã nối AudioWorklet thật. Vì vậy phần capture ngày nay chỉ còn **64 ms gom chunk + ~0–2 ms đóng gói/gửi**, và "capture latency < 70 ms" (`report/audit/05_measurements_and_status.md:1016`, K11 = 64 ms đo bằng harness Node) `[VERIFIED-NUM]`.

### 1.4 Ước lượng tổng độ trễ "từ lúc phát tiếng → phụ đề đã dịch" (dựng từ số đo, không phải một phép đo nguyên khối) `[NOT VERIFIED — suy luận]`

| Chặng | Giá trị dùng | Nguồn |
| :--- | ---: | :--- |
| Đóng gói chunk trong worklet | 0–64 ms | `lib/audio-processor.js` |
| Gửi qua bridge + WS (`ws.send_ms` + localhost) | ~1 ms | `metrics_report.json` |
| VAD phát hiện onset (FireRed, trung vị) | ~100 ms | `report/02_vad/ava_speech_report.md:53` |
| Chốt câu (im lặng cấu hình mặc định của engine) | ~350–600 ms | `README.md:562` (FireRed 200 ms × 10 ms = 200 ms khi ép cấu hình; engine tự quyết khi `silence=0`) |
| Lượt suy luận ASR lúc chốt | 73–107 ms | §1.3 |
| Dịch trọn câu + gửi | 282–430 ms | §1.3 |
| **Tổng ước lượng (p50)** | **~0,8–1,2 s** | — |

Con số này khớp với mục tiêu "< 1,0 s" mà `report/audit/BAO_CAO_AUDIT_HIEU_NANG_Gemini.md:9` ghi là mục tiêu hệ thống, và với kết luận e2e p95 ~1,14 s ở `XAC_NHAN_GEMINI_VA_KE_HOACH_TOI_UU.md:349`.

---

## 2. TÍNH ĐÚNG ĐẮN CỦA CƠ CHẾ (MECHANISM CORRECTNESS)

### 2.1 Quan sát được gì tại `appendBuffer`?

`SourceBuffer.appendBuffer(data)` nhận `BufferSource` (ArrayBuffer hoặc view). Trong một lời gọi đồng bộ của trang, ta quan sát được `[VERIFIED-EXT]` (nguồn: [MDN SourceBuffer](https://developer.mozilla.org/en-US/docs/Web/API/SourceBuffer)):

| Quan sát | Có/không | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| Con trỏ `this` = đối tượng `SourceBuffer` | ✅ | ⚠️ Không suy ra được `track type` (audio hay video) từ `this` — xem §2.4 |
| Byte (`data`) | ✅ | Bao gồm **cả** init segment (`ftyp`+`moov`) lẫn media segment (`moof`+`mdat`), và cả segment WebM |
| `byteLength`, view offset | ✅ | Phải chuẩn hoá `byteOffset`/`byteLength` khi là view |
| Nội dung có bị biến đổi sau đó? | ⚠️ | Trang có thể tái dùng `ArrayBuffer`. Nếu ta đọc **trễ** (async) thì phải `slice()` **ngay** trong hook. `[NOT VERIFIED]` về tần suất thực tế |
| Tính nguyên tử | ⚠️ | Với `appendBuffer`, theo đặc tả có thể ném `QuotaExceededError` **đồng bộ** nếu bộ đệm đầy; hook nên `try/finally` để không đổi ngữ nghĩa `[VERIFIED-EXT]` [MSE spec](https://www.w3.org/TR/2025/WD-media-source-2-20251031/) |
| `update` / `error` event | ✅ | Đăng ký được; **không** cho biết thời lượng media đã nạp |
| `buffered` (TimeRanges) | ✅ | Đây là nguồn duy nhất để biết **thời gian phát thật** ↔ byte đã nạp |

**Kết luận:** byte là quan sát được, nhưng **ánh xạ byte → thời gian trình bày không có sẵn** — phải tự tính (§2.2).

### 2.2 Ánh xạ media time → presentation time

Công thức cần dùng (chuẩn hoá theo đặc tả MSE, phần *coded frame processing*) `[VERIFIED-EXT]` ([MSE 2](https://www.w3.org/TR/2025/WD-media-source-2-20251031/)):

```
presentation_ts = media_ts + timestampOffset − earliest_presentation_timestamp
                  (khi appendWindowStart/End không cắt frame)
```

Trong đó:

| Thành phần | Cách lấy | Ghi chú |
| :--- | :--- | :--- |
| `media_ts` (fMP4) | `tfdt.baseMediaDecodeTime` của `moof` + offset trong `trun`; ở track chưa có `tfdt` thì theo tổng thời lượng sample đã append | `tfdt` là nguồn **chuẩn** cho fMP4 `[VERIFIED-EXT]` |
| `media_ts` (MP4 progressive) | Dựng sample table từ `moov` (`stts`/`stsc`/`stsz`/`stco`/`co64`) | Phải giữ `moov` từ init segment |
| `media_ts` (WebM) | `Cluster.Timecode × TimestampScale` + `SimpleBlock.relative_timecode` | Cộng thêm `CodecDelay`/`DiscardPadding` nếu có ⇒ **lệch cỡ ms** `[NOT VERIFIED]` |
| `timestampOffset` | Hook setter của `timestampOffset` và ghi lại giá trị; giá trị mặc định 0 | `[VERIFIED-EXT]` [MDN SourceBuffer](https://developer.mozilla.org/en-US/docs/Web/API/SourceBuffer) |
| `appendWindowStart/End` | Hook setter tương tự; mặc định `[0, +∞)` | Frame rơi ngoài cửa sổ bị **loại** ⇒ không được giả định mọi byte đã append đều phát |
| `earliest_presentation_timestamp` | Với fMP4 ≈ `tfdt` của segment đầu; với MP4 progressive = 0 | Nguồn sai số chính nếu bỏ qua |
| `video.currentTime` | Đọc trực tiếp | Nhưng `currentTime` là **thời gian phát hiện tại**, không phải timeline của `SourceBuffer` khi có `timestampOffset ≠ 0` — phải đối chiếu qua `buffered` |

**Hệ quả thiết kế quan trọng:** chỉ số `buffered` là **nguồn sự thật** cho "cái gì sắp được phát". Vì vậy M1 (§8) phải ghi **cả** `tfdt` suy ra từ byte **và** `buffered.start(i)/end(i)` để hiệu chỉnh công thức trên dữ liệu thật, thay vì tin vào công thức.

### 2.3 Audio+video xen kẽ trong một segment

Đây là chế độ **khó nhất** và cũng phổ biến nhất trên web (fMP4 progressive / byte-range).

| Tình huống | Việc phải làm | Rủi ro |
| :--- | :--- | :--- |
| Một segment chứa cả `moof`+`mdat` của 2 track | Duyệt `traf` trong `moof`, lọc `tfhd.track_ID` thuộc track audio (tra từ `moov.trak.mdia.hdlr = 'soun'`), rồi cắt **từng vùng byte** của track audio ra khỏi `mdat` | Nếu cắt thô cả `mdat` rồi đưa cho bộ giải mã audio ⇒ lỗi |
| Segment chỉ có video (thường gặp) | Bỏ qua, không được coi là "thất bại" | Đếm sai tỉ lệ thành công |
| `mdat` không có `moof` (progressive) | Dùng sample table của `moov` | Cần giữ `moov` từ init segment đầu tiên |
| Segment audio-only riêng | Trường hợp dễ nhất — chiếm đa số trên YouTube/Spotify-style DASH | — |

### 2.4 Nhiều `SourceBuffer` (track tách rời)

- Trên thực tế (YouTube DASH, hầu hết HLS/DASH của trình phát web) audio và video nằm ở **hai `SourceBuffer` khác nhau**. Khi đó hook **không biết** `this` là audio hay video `[VERIFIED-EXT]` — phải suy ra bằng cách:
  1. Hook `addSourceBuffer(mimeType)` ⇒ biết thứ tự và `mimeType` (`audio/mp4; codecs="mp4a.40.2"` vs `video/mp4; codecs="avc1..."`). Đây là **cách xác định chính**;
  2. Nếu `changeType()` được gọi, cập nhật lại;
  3. Đối chiếu `buffered` của hai buffer để phát hiện lệch.
- **Hệ quả:** trạng thái hook phải là **per-SourceBuffer**, tra cứu bằng `WeakMap<SourceBuffer, TrackInfo>` — không dùng biến toàn cục.

### 2.5 Các đường API khác phải xử lý

| API/đường | Điều xảy ra | Cách xử lý |
| :--- | :--- | :--- |
| `SourceBuffer.changeType(mime)` | Đổi codec giữa phiên (ví dụ sau ABR switch), chỉ hợp lệ khi buffer rỗng | Hook; reset `TrackInfo`, `timestampOffset` |
| `SourceBuffer.remove(start,end)` | Xoá dải thời gian (seek, dọn buffer) ⇒ byte đã ghi nhớ trở nên vô hiệu | Hook; **vô hiệu hoá kế hoạch phụ đề** trong dải bị xoá |
| `SourceBuffer.abort()` | Huỷ append đang dở | Hook; bỏ mọi trạng thái đang phân tích |
| `MediaSource.endOfStream()` | Hết stream | Dùng để chốt |
| **MSE trong Worker** (`MediaSource.handle` + `postMessage` handle sang worker) | `MediaSource`/`SourceBuffer` sống trong **worker**, hook prototype ở main world **không thấy gì** | ❌ Không hook được từ MAIN world. Phải bổ sung hook trong worker (khó, không ổn định) hoặc chấp nhận thất bại ⇒ fallback `[VERIFIED-EXT]` tồn tại trong đặc tả ([MSE 2](https://www.w3.org/TR/2023/WD-media-source-2-20231025/#4), [Chromium issue 40591101](https://issues.chromium.org/issues/40591101)) |
| `ManagedMediaSource` (Safari) | Lớp con của `MediaSource`, có thêm `startstreaming`/`endstreaming` | Hook cả `ManagedMediaSource.prototype` nếu tồn tại |
| `WebKitMediaSource` (cũ) | Tiền tố cũ | Hook nếu tồn tại |

---

## 3. TIÊM VÀO FIREFOX

### 3.1 Manifest hiện tại `[VERIFIED-REPO]`

| Thuộc tính | Giá trị | Bằng chứng |
| :--- | :--- | :--- |
| `manifest_version` | **3** | `manifest.json:2` |
| `permissions` | `activeTab`, `storage`, `scripting` | `manifest.json:16-20` |
| `background` | `"scripts": [...]` (event page, **không** persistent) | `manifest.json:31-37` |
| `content_scripts` | khai báo tĩnh, `all_frames: true`, `match_origin_as_fallback: true`, `run_at: "document_idle"` | `manifest.json:38-57` |
| `web_accessible_resources` | chỉ `lib/audio-processor.js` | `manifest.json:72-81` |
| CSP extension pages | `script-src 'self'; connect-src ...` | `manifest.json:28-30` |
| Đang dùng `scripting.executeScript`? | Có — nhưng **không có `world`**, nên chạy ở world mặc định (ISOLATED) của content script | `popup/popup.js:747-771` |

### 3.2 `world: "MAIN"` trong Firefox

| Cách | Hỗ trợ | Ghi chú |
| :--- | :--- | :--- |
| `browser.scripting.registerContentScripts([{ world: "MAIN", runAt: "document_start", ... }])` | **Firefox ≥ 128** `[VERIFIED-EXT]` — bug 1736575, xem [MDN ExecutionWorld](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/scripting/ExecutionWorld), [blog MV3 Firefox 128](https://blog.mozilla.org/addons/2024/07/10/manifest-v3-updates-landed-in-firefox-128/), [phabricator D211865/D211867](https://phabricator.services.mozilla.com/D211865), [bcd issue #24055](https://github.com/mdn/browser-compat-data/issues/24055) | Đây là **đường khuyến nghị** |
| `browser.contentScripts.register({ world: "MAIN" })` (API cũ, MV2-style) | Có `world` nhờ bug 1736575 `[VERIFIED-EXT]` (cùng nguồn trên) | Đường dự phòng cho Firefox cũ hơn 128 |
| phiên bản Firefox tối thiểu của repo | **Chưa khai báo** — `browser_specific_settings.gecko` chỉ có `id` + `data_collection_permissions` | `manifest.json:6-15` ⇒ hiện **không** ràng buộc, sẽ phải thêm `strict_min_version` |

**Hệ quả:** để dùng `world: "MAIN"` một cách đúng đắn, manifest **phải** thêm `browser_specific_settings.gecko.strict_min_version` (đề xuất `"128.0"`) và `scripting` phải được gọi ở `document_start` (nếu không, trang đã tạo `MediaSource` trước khi hook kịp gắn) `[VERIFIED-EXT]` về yêu cầu thời điểm.

### 3.3 Vấn đề Xray vision và cách tiêm thay thế `[VERIFIED-EXT]` / `[NOT VERIFIED]`

Firefox cô lập content script bằng **Xray vision**: content script nhìn thấy bản "sạch" của đối tượng trang. Monkey-patch `MediaSource.prototype.appendBuffer` từ content script (ISOLATED world) **không** ảnh hưởng tới trang. Các đường thay thế:

| Cách | Cơ chế | Đánh giá |
| :--- | :--- | :--- |
| `wrappedJSObject` | `window.wrappedJSObject.MediaSource` để chạm vào đối tượng thật của trang | ⚠️ `[NOT VERIFIED]` — tôi **không** xác minh được trong phiên này rằng vá prototype qua `wrappedJSObject` được trang nhìn thấy ổn định. Cần M1 xác nhận |
| `cloneInto` / `exportFunction` | Chuyển hàm từ world extension sang world trang | ✅ Đường **chắc chắn hoạt động** khi phải chạy code trong world trang từ content script, nhưng đối tượng trả về vẫn cần `wrappedJSObject` để chạm tới prototype thật |
| Chèn `<script src="moz-extension://...">` | Thẻ script trỏ tới file trong `web_accessible_resources` | ⚠️ **CSP của trang có thể chặn** `moz-extension:` trong `script-src` `[NOT VERIFIED]` về hành vi thực tế |
| Chèn `<script>` inline với nội dung hàm | Trang thường chặn inline script bằng CSP | ❌ Không đáng tin |
| **`scripting.executeScript` / `registerContentScripts` với `world: "MAIN"`** | Trình duyệt tự chạy code trong main world, **không** đi qua CSP của trang | ✅ **Đường nên dùng** (Firefox ≥ 128) |
| Patch từ popup qua `scripting.executeScript({world:"MAIN"})` | Hiện popup đã gọi `scripting.executeScript` (`popup.js:747-771`) | ✅ Dùng lại được, nhưng quá muộn (`document_start` là bắt buộc) |

**Lưu ý về MV2:** `manifest.json` hiện là MV3. Nếu chuyển về MV2 để dùng `contentScripts.register` thì phải bỏ `host_permissions` (MV2 gộp vào `permissions`) và bỏ `background.service_worker` — và việc này đi ngược lộ trình MV3 của Firefox `[NOT VERIFIED]` về mốc thời gian cụ thể.

### 3.4 Đưa `ArrayBuffer` từ page world về extension

| Đường | Chi phí | Nhận xét |
| :--- | :--- | :--- |
| `window.postMessage(msg, "*", [transferable])` | **Zero-copy** cho transfer list `[VERIFIED-EXT]` (HTML structured serialize with transfer) | ✅ **Khuyến nghị.** Bắt buộc dùng transfer list, nếu không là copy đầy đủ |
| `MessageChannel` port | Zero-copy nếu dùng transfer list; tách kênh riêng khỏi `postMessage` toàn cục | ✅ Tốt nhất về mặt nhiễu (tránh trang nghe lén sự kiện) |
| `CustomEvent(detail: ArrayBuffer)` | ⚠️ Firefox **hạn chế** việc truyền một số đối tượng qua `detail` giữa các world (cơ chế permission check của `CustomEvent`) | ❌ Không nên dùng cho dữ liệu nhị phân |
| `dispatchEvent` + `Object` đóng gói base64 | Tốn +33 % kích thước và CPU | ❌ |
| Chuyển tiếp extension: content script → SW qua `port.postMessage` | **Hiện tại KHÔNG có transfer list** (`ws-client.js:291-295`) ⇒ copy | ⚠️ Nếu gửi phân đoạn media nguyên khối (MB), đây là **copy trên main thread trang** — phải thêm transfer list hoặc chỉ gửi PCM đã demux (nhỏ hơn rất nhiều) |

**Chiến lược giảm chi phí then chốt:** **không** chuyển byte container về extension. Demux + giải mã thành PCM 16 kHz **ngay trong page world** (hoặc trong `AudioWorklet` của trang), rồi chỉ gửi PCM. 1 giây audio 16 kHz mono 16-bit = **32 KB**, tức 32 KB/s thay vì 0,5–2 MB cho mỗi phân đoạn.

---

## 4. THỰC TẾ DRM

### 4.1 Điều xảy ra với nội dung EME/CENC

| Mệnh đề | Trạng thái | Nguồn/lý do |
| :--- | :--- | :--- |
| Byte tại `appendBuffer` của nội dung CENC **vẫn là byte đã mã hoá**; `mdat` là `encv`/`enca` với `senc`/`saiz`/`saio` mô tả IV và subsample | `[VERIFIED-EXT]` ở mức "đặc tả EME mô tả mô hình này" ([EME 2](https://www.w3.org/TR/2026/WD-encrypted-media-2-20260508/)) | Lý do: MSE chuyển byte cho media pipeline, CDM giải mã **sau** đó |
| `encrypted` event (`MediaEncryptedEvent`) chỉ lộ `initData` + `initDataType` (`"cenc"`/`"keyids"`/`"webm"`), **không** lộ key, không lộ byte đã giải mã | `[VERIFIED-EXT]` ([EME 2 WebIDL](https://www.w3.org/TR/2026/WD-encrypted-media-2-20260508/)) | `initData` là hộp `pssh` — mô tả **hệ thống khoá**, không phải khoá |
| Hook page world **không thể** giải mã vì không có CDM; CDM chạy trong sandbox riêng, giao tiếp qua `MediaKeys`/`MediaKeySession`, khoá là `ArrayBuffer` chỉ có ý nghĩa với CDM | `[VERIFIED-EXT]` (cùng nguồn) | Ngay cả khi đọc được `session.load()`/`update()`, dữ liệu khoá là opaque |
| Mức "an ninh" của Widevine (L1 vs L3) quyết định khoá có nằm trên host hay không | `[NOT VERIFIED]` — tôi **không** xác minh được chi tiết mức L trong phiên này | Chỉ nêu như định hướng |

### 4.2 Trích xuất khoá L3 — ngoài phạm vi

- Về mặt kỹ thuật, trích xuất khoá Widevine L3 là chủ đề của tài liệu nghiên cứu (ví dụ luận án `theses.hal.science/tel-04446310` xuất hiện trong kết quả tìm kiếm) `[NOT VERIFIED]` — tôi **không** đọc toàn văn và **không** khẳng định tính khả thi.
- Về pháp lý: các biện pháp kiểm soát truy cập bị luật bảo vệ; ở Hoa Kỳ Điều 1201 DMCA có các miễn trừ được ban hành lại theo chu kỳ 3 năm — [Copyright Office 1201 (2024)](https://copyright.gov/1201/2024/index.html), [Federal Register 89 FR 85437](https://www.federalregister.gov/documents/2024/10/28/2024-24563/exemption-to-prohibition-on-circumvention-of-copyright-protection-systems-for-access-control) `[VERIFIED-EXT]` (sự tồn tại của quy trình và văn bản), `[NOT VERIFIED]` (nội dung miễn trừ cụ thể không đọc).
- Ngoài pháp luật, việc này vi phạm **điều khoản sử dụng** của mọi dịch vụ DRM lớn.
- **Kết luận:** đây là hạng mục **ngoài phạm vi**, không phải "khó nhưng làm được". Đề xuất phải **phát hiện DRM và từ chối** (fallback), chứ không cố vượt.

### 4.3 Bảng "chạy / không chạy" theo lớp nội dung

| Lớp | Có `appendBuffer`? | Byte đọc được? | Audio lấy được? | Kết luận |
| :--- | :--- | :--- | :--- | :--- |
| **MP4 progressive (`<video src="x.mp4">`)** | ❌ không có MSE | — | — | ❌ Đề xuất không áp dụng; capture hiện tại vẫn cần |
| **MSE, không DRM (fMP4/DASH audio-only riêng)** | ✅ | ✅ | ✅ (demux hoặc offscreen decode) | ✅ **Chạy** — trường hợp mục tiêu |
| **MSE, không DRM, fMP4 audio+video xen kẽ** | ✅ | ✅ | ✅ nhưng cần tách track theo `traf` | ⚠️ Chạy, chi phí demux + rủi ro cao hơn |
| **MSE không DRM, WebM** | ✅ | ✅ | ✅ nhưng cần demux EBML | ⚠️ Chạy, thêm một bộ demux nữa |
| **MSE + EME (CENC/Widevine/PlayReady/FairPlay)** | ✅ | ⚠️ byte **đã mã hoá** | ❌ | ❌ **Không chạy** — Netflix, Disney+, Prime Video, nhiều nội dung YouTube trả phí |
| **YouTube** | ✅ | `[NOT VERIFIED]` — cần M1 kiểm chứng có gặp byte CENC hay không | ? | ⚠️ **Phải đo mới kết luận** |
| **File cục bộ / blob** | ❌ thường không | — | — | ❌ Khác bài toán (đọc file trực tiếp dễ hơn nhiều) |
| **Live (LL-HLS/LL-DASH)** | ✅ | ✅ (nếu không DRM) | ✅ | ❌ Không đạt "trước khi phát" vì dẫn trước ≈ 0 (§5.3) |

---

## 5. PHÂN TÍCH ĐỘ TRỄ

### 5.1 Ngân sách so sánh

Ký hiệu: `T_pipe` = chi phí toàn pipeline từ lúc có byte audio đến lúc phụ đề hiển thị; `L` = thời gian dẫn trước = `buffered.end − currentTime`; `S` = tốc độ phát (`playbackRate`).

| Kiến trúc | Độ trễ cảm nhận | Điều kiện |
| :--- | :--- | :--- |
| Hiện tại (phát → capture → ASR → dịch) | ≈ `T_pipe` ≈ **0,8–1,2 s** (§1.4) | Không phụ thuộc buffer |
| Chặn bắt MSE | **≈ 0** nếu `L/S > T_pipe_value + T_demux`; nếu không thì vẫn là `(L/S)` khi phụ đề tới sau khi trình phát đã đi qua mốc | Phụ thuộc `L` |

Trong đó `T_pipe_value` là công suất xử lý **tính theo giây audio mỗi giây wall-clock**, không phải độ trễ một câu. Từ số đo §1.3:

| Thành phần | Giá trị (RTF theo nghĩa "giây xử lý / giây audio") |
| :--- | :--- |
| ASR `infer_core` (p50 74,7 ms cho cửa sổ trung bình 2,55 s) | ≈ **0,029** |
| ASR commit (p50 106,9 ms cho câu 4,5 s) | ≈ **0,024** (khớp RTF 0,0232 trong `README.md`) |
| VAD (p50 3,2 ms cho frame 25 ms) | ≈ **0,13** (13 % một nhân CPU) |
| Dịch | Không theo RTF — **~281 ms/câu** ở p50, **~430 ms/câu** ở baseline e2e của `XAC_NHAN_GEMINI_VA_KE_HOACH_TOI_UU.md:369` |
| Demux + giải mã (chưa đo) | `[NOT VERIFIED]` — đối tượng đo chính của M1 |

⇒ **Tổng công suất tiêu thụ ≈ 0,2–0,3 × realtime** (chưa kể demux) ⇒ còn **dư ~3–5×** so với phát 1×. Đây là tin tốt cho đề xuất.

### 5.2 Trần cứng không thể xoá: ASR cần thấy đuôi câu

| Khoản | Giá trị | Vì sao MSE không xoá được |
| :--- | ---: | :--- |
| VAD phát hiện onset (trung vị, FireRed) | ~100 ms | Là bản chất của thuật toán VAD |
| Chờ im lặng/dấu câu để chốt | ~350–600 ms | Là điều kiện **ngữ nghĩa** để biết câu đã hết |
| `refine_lead_ms = 350` (bù độ trễ nhận dạng) | 350 ms | `report/audit/21_SEG_VA_FIX_DEDUP_PHU_DE.md:514` |

MSE chỉ **dịch** khoản này ra trước theo đồng hồ phát. Nếu `L/S` nhỏ hơn khoản này, phụ đề vẫn tới sau khi trình phát đã đi qua mốc.

### 5.3 Kết luận theo từng chế độ

| Chế độ | `L` điển hình | `T_pipe` cần | Kết luận |
| :--- | :--- | :--- | :--- |
| **VOD, mạng tốt, DASH/HLS 6 s segment** | **10–30 s** (must be measured per player; `[NOT VERIFIED]` cho con số cụ thể của từng trang) | ~1,4 s + demux | ✅ **"~0 ms" khả thi**, dư 10–20× |
| **VOD, mạng yếu / buffer cạn** | có thể < 2 s | ~1,4 s | ⚠️ Biên rất mỏng, phải chấp nhận trễ |
| **VOD ở 2× (`playbackRate = 2`)** | giảm một nửa (5–15 s) | RTF yêu cầu tăng gấp đôi: cần < 0,5 | ✅ Vẫn khả thi (dư ~2×) nhưng phải giải mã nhanh hơn realtime; nếu dùng đường `HTMLMediaElement` ẩn thì phải đặt chính nó ở `playbackRate = 2–4` để theo kịp |
| **Live, độ trễ thấp (LL-HLS/LL-DASH)** | **≈ 0–3 s** | ~1,4 s | ❌ Không đạt "trước khi phát"; thực tế **không tốt hơn** capture hiện tại |
| **Live, mép phát** | ≈ 0 | — | ❌ Về mặt kiến trúc không thể |

**Vì sao vẫn nên làm dù không phải "0 ms" ở live:** ở live, chặn bắt MSE vẫn cho một lợi ích khác — **không phụ thuộc âm lượng/tắt tiếng của trình phát** (`content-script.js:128-145` cho thấy đường capture hiện tại **mất hoàn toàn** phụ đề khi `video.muted = true`) và **không bị jank main thread của trang làm hỏng audio**. Đây là lập luận độc lập với độ trễ và đáng đưa vào tiêu chí đánh giá M1.

---

## 6. RỦI RO HIỆU NĂNG VÀ ĐÚNG ĐẮN

| # | Rủi ro | Mức | Ghi chú / cách giảm |
| :-: | :--- | :--- | :--- |
| R1 | **Chi phí demux mỗi phân đoạn.** fMP4 audio+video xen kẽ phải parse `moof`/`traf`/`trun` và cắt theo byte range | 🟠 Cao | `[NOT VERIFIED]` về ms/segment. **Không** đặt demux trên main thread trang — dùng Worker. Biến thể offscreen-decode (§7.4) bỏ hẳn khoản này |
| R2 | **Tăng bộ nhớ.** Giữ init segment + bản sao byte + PCM đã giải mã + kế hoạch phụ đề | 🟠 Cao | Trần cứng theo phút: 16 kHz mono 16-bit = **1,92 MB/phút**. Nếu giữ cả container (0,5–2 MB / 6 s) thì ~**5–20 MB/phút** ⇒ phải giải phóng ngay sau khi lấy PCM |
| R3 | **Seek.** `remove()` + đổi `timestampOffset` + audio cũ trong VAD/ASR | 🔴 Rất cao | Repo **đã có** cơ chế tương ứng cho đường capture: `reset_stream` (`backend/ws/handler.py:565-605`) + `audioCapture.reset()` (`content-script.js:338-361`). Phải nối cùng sự kiện `seeking/seeked` để báo backend reset |
| R4 | **ABR đổi chất lượng ⇒ byte layout đổi** | 🟡 Trung bình | Init segment mới (`changeType` hoặc init mới) ⇒ phải reset `TrackInfo`, `tfdt` gốc; không dùng lại ánh xạ cũ |
| R5 | **Quảng cáo / SCTE-35 / chèn quảng cáo phía server** | 🟠 Cao | Timeline có **khoảng trống hoặc nhảy**; `[VERIFIED-EXT]` về khái niệm chèn quảng cáo dựa trên dấu (`DATERANGE`/`EXT-X-CUE-OUT`, `EventStream`) — [AWS MediaTailor](https://docs.aws.amazon.com/mediatailor/latest/ug/mediatailor-guide.pdf), [SCTE 301 2025](https://account.scte.org/documents/8396/SCTE_301_2025.pdf). Phải phát hiện `buffered` không liên tục ⇒ đối xử như seek |
| R6 | **Trình phát không dùng MSE** | 🟡 Trung bình | Progressive `src=` trực tiếp; **Native HLS trên Safari**; DRM có pipeline riêng. Hook sẽ **không bao giờ** bắn ⇒ cần timeout phát hiện và fallback |
| R7 | **Mã hoá ở tầng vận chuyển (HLS AES-128)** | 🟡 Trung bình | Byte tại `appendBuffer` là **plaintext** (trình phát đã giải mã trước khi append) ⇒ **không cản trở**. Nhưng nếu chọn hook tầng `fetch`/XHR thay vì `appendBuffer` để bắt sớm hơn thì gặp byte mã hoá và phải xử lý khoá — [hls.js issue #6803](https://github.com/video-dev/hls.js/issues/6803), [#6557](https://github.com/video-dev/hls.js/issues/6557). **Khuyến nghị: bám `appendBuffer`, không bám network layer** |
| R8 | **Đổi ngữ nghĩa trang** | 🟠 Cao | Nếu hook ném lỗi hoặc chặn `appendBuffer` ⇒ **video không phát được**. Bắt buộc: gọi bản gốc trước, bọc `try/catch`, mọi lỗi của ta **không** được nổi lên trang |
| R9 | **Phụ đề không khớp timeline** | 🔴 Rất cao | Phải kiểm tra chéo `buffered` ↔ `currentTime` ↔ `tfdt`; nếu không khớp quá ngưỡng ⇒ từ bỏ kế hoạch "chạy trước" và quay lại đường capture |
| R10 | **Trùng lặp với TTS đang phát** | 🟡 Trung bình | Nếu chạy trước nhiều giây, hàng đợi TTS phía client (`tts-player.js`) phải có lịch phát theo `currentTime`, nếu không tiếng lồng sẽ lệch. Hiện hàng đợi này **không so với `currentTime`** — `report/audit/00_BAO_CAO_AUDIT_HIEU_NANG.md:330` |

### 6.1 Fallback tự động về đường capture hiện tại

Điều kiện chuyển (đề xuất, tất cả đều đo được ở M1/M4):

| Điều kiện | Ngưỡng đề xuất | Hành động |
| :--- | :--- | :--- |
| Không hook được `MediaSource` (không có MSE) | sau `document_start` + 2 s | Fallback |
| Hook được nhưng **0 phân đoạn audio** trong N giây khi `video.currentTime` đang tăng | N = 5 s | Fallback |
| Byte trông như CENC (`senc`/`saiz`/`saio` trong `moov`, hoặc `pssh`) hoặc đã có `encrypted` event | ngay lập tức | Fallback + thông báo cho người dùng "nội dung có DRM" |
| Lệch `buffered` ↔ `currentTime` ↔ `tfdt` > ngưỡng | > 500 ms `[NOT VERIFIED]` | Fallback |
| `L/S < T_pipe` liên tục | `L < 3 s` trong 2 lần đo liên tiếp | Chuyển sang chế độ "chạy song song, ưu tiên capture" |
| Lỗi demux > X % phân đoạn | X = 20 % trong 30 s | Fallback |
| `MediaSource` sống trong Worker | phát hiện `MediaSource.handle` được đọc, hoặc `addSourceBuffer` không bao giờ bị gọi trong 5 s | Fallback |

Nguyên tắc: **fallback là mặc định trạng thái; chặn bắt MSE là "tính năng bật lên" chỉ khi tự chứng minh được.** Fallback phải **không** làm gián đoạn phụ đề đang chạy (giữ `wsClient` và `audioCapture` như hiện tại, chỉ đổi nguồn PCM).

---

## 7. THIẾT KẾ ĐỀ XUẤT (KHÔNG PHẢI MÃ SẢN PHẨM)

### 7.1 Kiến trúc tổng thể

```
[MAIN world, document_start]
  hook: MediaSource.prototype.addSourceBuffer / SourceBuffer.prototype.appendBuffer
        / changeType / remove / abort / timestampOffset setter
        MediaSource.isTypeSupported, ManagedMediaSource, WebKitMediaSource
   │  (byte, mimeType, thời điểm, chỉ số)
   ├─► Worker: demux (mp4box.js | mux.js | EBML) + trích PCM 16 kHz mono
   │      hoặc  ──► HEADLESS DECODE: blob → HTMLMediaElement.playbackRate=N
   │                     → MediaElementSource → AudioWorklet → PCM 16 kHz
   │
   └─► MessageChannel (transferable)  →  [ISOLATED world content script]
                                             │  tái dùng WSClient.sendBinary
                                             ▼
                                    wss://localhost:8765/ws  (khung audio_chunk hiện có)
                                             │
                                    VAD → ASR → SEG → dịch → TTS   (không đổi)
                                             │
                                             ▼
                        { mediaTime, leadTime }  →  lịch hiển thị ở overlay
```

**Nguyên tắc:** backend **không đổi** ở M1–M3. Chỉ phần "nguồn PCM" và "lịch hiển thị" thay đổi.

### 7.2 Ánh xạ thời gian và lập lịch

1. Mỗi lần append, gán cho dải byte một khoảng `[mediaStart, mediaEnd]` suy từ `tfdt`/`trun`.
2. `presentation = media + timestampOffset − earliest`.
3. Lưu `buffered` (TimeRanges) làm nguồn sự thật; kiểm tra `presentation ≈ buffered` cho các mốc đã nạp.
4. Với mỗi câu đã dịch gắn `[tStart, tEnd]` (theo timeline media), overlay **không** hiển thị theo "vừa nhận được" như hiện tại mà theo `video.currentTime ∈ [tStart, tEnd]`.
5. Ranh giới an toàn: nếu `currentTime` đã vượt `tEnd` ⇒ **bỏ** câu đó (tránh hiện phụ đề của đoạn đã qua).

### 7.3 Điểm nối với backend hiện có

- Khung gửi giữ nguyên `type: "audio_chunk"` + `[uint32 len][JSON][PCM]` (`backend/ws/protocol.py:35-41`).
- **`captureTimestamp`**: backend dùng nó làm mốc đồng hồ stream (`backend/vad/processor.py:363-368`, `stream_ts`), nên chỉ cần **đơn điệu tăng** và **khớp tốc độ 1:1 với số mẫu**. Vì vậy ở chế độ chạy trước, gửi `captureTimestamp` = mốc media (hoặc mốc wall-clock tăng đều) là **đủ** — không cần khớp đồng hồ `AudioContext`.
- Khi phát hiện seek: gửi `{"type":"reset_stream"}` như `content-script.js:358` đang làm.

### 7.4 Biến thể rẻ hơn: giải mã offscreen thay vì tự demux `[đề xuất ưu tiên]`

**Ý tưởng:** trình duyệt **đã có** bộ demux và bộ giải mã. Thay vì tự viết demux MP4/WebM, ta để trình duyệt làm:

1. Ghép `init segment + media segment` thành một `Blob` (đúng chuỗi byte đã append).
2. `URL.createObjectURL(blob)` → một `HTMLMediaElement` **ẩn, không hiển thị** (`audio` hoặc `video` tuỳ codec).
3. `createMediaElementSource` → `AudioWorkletNode("audio-capture-processor")` — **chính file worklet đã có trong repo** (`extension_firefox/lib/audio-processor.js`, đã là web-accessible resource).
4. Đặt `playbackRate = 1–4` để giải mã nhanh hơn realtime.
5. Ghi lại mốc `mediaStart` của phân đoạn, gửi PCM kèm `captureTimestamp = mediaStart + n·0.064`.

| Tiêu chí | Tự demux (mp4box/mux.js/EBML) | Offscreen decode |
| :--- | :--- | :--- |
| Độ phức tạp | Cao (3 định dạng container, biến thể CENC/fMP4) | **Thấp** |
| Tái dùng hạ tầng repo | Không | **Có** (`audio-processor.js`, `sendBinary`, VAD/ASR) |
| Hỗ trợ nhiều codec | Tự lo | Trình duyệt lo (`Opus`, `AAC`, `MP3`, `Vorbis`, `FLAC`) |
| Chi phí CPU | Demux + decode một lần | Decode (browser, có thể tăng tốc) + worklet |
| Ràng buộc | — | Segment phải **tự phát được** (cần init đúng); một số codec/segment có thể không phát rời được nếu thiếu `moov`/dependency |
| Rủi ro | Bug demux ⇒ im lặng | `play()` cần tương tác người dùng (autoplay policy) cho phần **nghe**; nhưng element ẩn + `MediaElementSource` vẫn có thể bị chặn nếu chưa có user gesture |

**Kết luận thiết kế:** làm **M2 theo biến thể này trước**; chỉ chuyển sang tự demux nếu offscreen decode thất bại ở tỉ lệ đáng kể trên các trang thực tế.

---

## 8. KẾ HOẠCH MỐC (MỖI MỐC KIỂM CHỨNG ĐỘC LẬP)

> Nguyên tắc chung: mỗi mốc phải có **một artifact đo được** (JSON log / số liệu) và **một điều kiện dừng** rõ ràng. Không mốc nào được phép phá đường capture hiện tại.

### M0 — Hạ tầng đo (nửa ngày, không có rủi ro)

- Thêm khuôn khổ ghi log `scratch/` (script rời, không nằm trong extension) để thu: `ms()`, `buffered ranges`, `currentTime`, `playbackRate`.
- **Kiểm chứng:** chạy được trên 1 trang, ghi ra JSON.

### M1 — Hook MAIN world, chỉ log (mục tiêu: chứng minh bắt được dữ liệu)

- Tiêm hook ở `document_start`, world `MAIN`, `all_frames: true` (`manifest.json:38-57` đã có `all_frames`).
- Ghi log: `mimeType` từ `addSourceBuffer`, `byteLength` mỗi append, `tfdt` parse được (fMP4), `timestampOffset`, `buffered.start/end` trong `update`, `currentTime`, `playbackRate`, dấu hiệu CENC.
- **Chạy trên 3 trang thật:** 1 trang DASH fMP4 (YouTube), 1 trang HLS fMP4 audio/video tách rời, 1 trang WebM/VoD.
- **Kiểm chứng (điều kiện ĐẠT):**
  1. ≥ 95 % lời gọi `appendBuffer` bắt được `mimeType` và kích thước;
  2. Với trang tách track, phân biệt được buffer audio vs video qua `mimeType`;
  3. **Ghi lại được `L = buffered.end − currentTime` phân bố p50/p5** — đây là **số quyết định** toàn bộ đề xuất;
  4. Ghi nhận có/không dấu hiệu CENC trên từng trang.
- **Điều kiện DỪNG:** nếu `L` p5 < 3 s ở đa số trang khảo sát ⇒ toàn bộ lợi ích độ trễ của đề xuất **biến mất**; dừng trước khi đầu tư M2.
- **Không** ảnh hưởng người dùng: chỉ log.

### M2 — Lấy PCM và đẩy vào backend hiện có

- Dùng **biến thể §7.4** (offscreen decode) trước.
- Gửi qua **đúng** giao thức `audio_chunk` hiện có; backend không đổi.
- **Kiểm chứng:**
  1. Có phụ đề hiển thị từ nguồn MSE (đối chiếu nội dung với transcript của chính video);
  2. `pipeline.e2e_asr_to_sub_ms` vẫn trong khoảng đã đo (§1.3) — tức không làm backend chậm đi;
  3. Tỉ lệ phân đoạn giải mã thành công ≥ 90 %;
  4. Không có `RangeError`/`QuotaExceededError` do hook làm thay đổi ngữ nghĩa `appendBuffer`.

### M3 — Ánh xạ media time → `currentTime` và lập lịch trước

- Gắn `[mediaStart, mediaEnd]` cho từng câu; overlay hiển thị theo `currentTime`, không theo "thời điểm nhận gói".
- Thêm hàng đợi chờ có trần (ví dụ 30 s audio) và **bỏ** câu đã qua.
- **Kiểm chứng:**
  1. Sai số mốc phụ đề so với mốc thật của video ≤ **150 ms trung vị** (lấy ngưỡng này từ tiêu chí tương tự ở `report/audit/21_SEG_VA_FIX_DEDUP_PHU_DE.md:601`);
  2. Không có phụ đề "của đoạn trước" sau khi seek (kiểm tra bằng seek liên tục theo kịch bản G ở `report/audit/17_XAC_NHAN_AUDIT_VA_KE_HOACH_SUA.md:300`);
  3. Bộ nhớ tăng trưởng có trần khi phát 30 phút (đo bằng `performance.memory` hoặc RSS của tiến trình trình duyệt).

### M4 — Fallback tự động (bắt buộc trước khi phát hành)

- Cài đặt bảng điều kiện §6.1, kèm **test chủ động**: chạy trên (a) trang progressive `src=`, (b) nội dung DRM, (c) chế độ live, (d) MSE trong Worker (nếu tìm được trang thật).
- **Kiểm chứng:** trong cả 4 trường hợp, phụ đề vẫn xuất hiện bằng đường capture hiện tại, và người dùng không phải bấm gì thêm.

### M5 — Ghi chú cổng sang Chrome (MV3)

- Chrome cũng có `world: "MAIN"` trong `scripting.registerContentScripts` `[VERIFIED-EXT]` (xem [Chrome Extensions what's-new](https://developer.chrome.com/docs/extensions/whats-new)) — nhưng repo hiện dùng `background.scripts` (kiểu Firefox) thay vì `service_worker` (`manifest.json:31-37`), nên bản Chrome cần một manifest riêng.
- Khác biệt cần kiểm: `createMediaElementSource` + autoplay policy, `ManagedMediaSource` chỉ có ở Safari, MSE-in-Workers có ở Chrome `[VERIFIED-EXT]` ([Chromium issue 40591101](https://issues.chromium.org/issues/40591101)) ⇒ phải tính đến việc hook **không** thấy gì.
- **Kiểm chứng:** M1 chạy lại được trên Chrome với cùng script đo.

---

## 9. TIỀN LỆ (PRIOR ART)

Các dự án thực sự hook tầng media/MSE để lấy byte. Tình trạng bảo trì `[NOT VERIFIED]` — tôi chỉ thấy trang dự án/tài liệu qua tìm kiếm, **chưa** kiểm tra commit gần nhất.

| Dự án | Cách tiếp cận | URL | Ghi chú |
| :--- | :--- | :--- | :--- |
| **jaysonlong/webvideo-downloader** | Có script **`MediaSourceExport`** riêng; extension tiêm vào trang để lấy dữ liệu qua MSE | [github.com/jaysonlong/webvideo-downloader](https://github.com/jaysonlong/webvideo-downloader), [DeepWiki: MediaSourceExport](https://deepwiki.com/jaysonlong/webvideo-downloader/2.3-mediasourceexport-script) | Tiền lệ **gần nhất** với đề xuất này. Hỗ trợ Bilibili/iQIYI/Tencent/MGTV/WeTV ⇒ cho thấy hook MSE qua extension **chạy được trong thực tế** |
| **xifangczy/cat-catch (猫抓)** | Bắt media ở tầng network (`m3u8`/`mpd`/`m4s`) + "cache capture" | [DeepWiki cat-catch](https://deepwiki.com/xifangczy/cat-catch), [cat-catch trên Edge Add-ons](https://microsoftedge.microsoft.com/addons/detail/oohmdefbjalncfplafanlagojlakmjci) | Đi theo hướng **network layer**, không phải `appendBuffer` ⇒ gặp vấn đề AES-128/DRM (§R7) |
| **"无限制下载器" (userscript, Tampermonkey)** | Userscript chặn bắt nguồn media để tải | [appinn.com/445751](https://www.appinn.com/445751-unlimited-downloader/) | Tiền lệ rằng cách này **hay vỡ** khi trang đổi implementation |
| **Video DownloadHelper (Firefox)** | Phát hiện + tải media, có hỗ trợ một phần luồng MSE | [addons.mozilla.org](https://addons.mozilla.org/en-CA/firefox/addon/video-downloadhelper-allinone/) | Ổn định nhất về mặt bảo trì; nhưng mục tiêu là tải, không phải transcribe |
| **hls.js** (không phải công cụ chặn bắt) | Tài liệu hoá việc giải mã AES-128 và luồng khoá | [issue #6803](https://github.com/video-dev/hls.js/issues/6803), [#6557](https://github.com/video-dev/hls.js/issues/6557) | Nguồn tham chiếu cho R7 |

**Bài học chung từ tiền lệ:**

1. Hook MSE **đã được chứng minh chạy được** trong extension trình duyệt (webvideo-downloader).
2. Các công cụ này **vỡ liên tục** khi trang đổi player ⇒ hàm ý: **fallback tự động (M4) không phải tuỳ chọn, mà là điều kiện tồn tại**.
3. Công cụ tải chỉ cần *byte*; đề xuất của chúng ta cần *PCM đúng thời gian* — mức khó cao hơn một bậc.

---

## 10. CÂU HỎI MỞ VÀ VIỆC PHẢI ĐO TRƯỚC

| # | Câu hỏi mở | Cách đo nhỏ nhất | Nếu câu trả lời là "xấu" thì sao |
| :-: | :--- | :--- | :--- |
| Q1 | `L = buffered.end − currentTime` phân bố thế nào trên trang thật? | M1: hook + log, 3 trang, 60 s | L p5 < 3 s ⇒ lợi ích độ trễ về 0 ⇒ **dừng** |
| Q2 | Byte tại `appendBuffer` của YouTube có phải CENC? | M1: đọc `moov` tìm `senc`/`saiz`/`saio`/`pssh`, bắt `encrypted` event | Có ⇒ YouTube ra khỏi phạm vi |
| Q3 | Chi phí demux + giải mã trên máy tham chiếu? | Đo ms cho 100 phân đoạn ở 3 định dạng | > 50 % ngân sách `L` ⇒ phải dùng §7.4 hoặc Worker |
| Q4 | Bao nhiêu phần trăm trang dùng **MSE trong Worker**? | M1: cờ "hook gắn nhưng 0 append trong 5 s" | ≥ 20 % ⇒ thu hẹp phạm vi đáng kể |
| Q5 | `captureTimestamp` gửi theo mốc media có làm hỏng VAD/ASR không? | M2: chạy lại `test_08_streaming_latency.py` và so p50/p95 | Hỏng ⇒ phải thêm trường `mediaTime` riêng, đổi `backend/ws/protocol.py` |
| Q6 | Hàng đợi TTS phía client có chịu được phụ đề tới sớm nhiều giây? | M3: đọc `tts-player.js`, thêm lịch phát theo `currentTime` | Không ⇒ TTS phải tắt ở chế độ chạy trước, hoặc phải sửa client |
| Q7 | Firefox thực tế có cho vá prototype qua `wrappedJSObject` ổn định không? | Test 5 dòng trong console trang | Không ⇒ **bắt buộc** đi đường `world: "MAIN"` + `strict_min_version` |

### 10.1 Thí nghiệm nhỏ nhất có thể **phủ định** đề xuất (falsification test)

> **Một file userscript/`executeScript` ~40 dòng, chạy ở `document_start`, world MAIN, trên đúng 3 trang trong 60 giây.**
>
> Nó chỉ làm 3 việc: (1) vá `addSourceBuffer` để ghi `mimeType`; (2) vá `appendBuffer` để ghi `byteLength` + có/không dấu hiệu CENC + `tfdt`; (3) mỗi 200 ms ghi một mẫu `{currentTime, playbackRate, buffered}`.
>
> **Phán quyết dựa trên đúng một biểu thức:**
>
> ```
> ĐẠT  khi  median(buffered.end − currentTime) / playbackRate  >  2 × (1,4 s + demux_ms)
> ```
>
> - Nếu **ĐẠT**: tiếp tục M2; lợi ích độ trễ là thật và đo được.
> - Nếu **KHÔNG ĐẠT** ở đa số trang, hoặc byte có dấu hiệu CENC ở các trang mục tiêu: **đề xuất bị phủ định** — chi phí kỹ thuật (demux 3 container + MSE-in-Worker + seek/ABR/quảng cáo + fallback) **không** đổi lấy được lợi ích nào so với đường capture hiện tại (vốn đã chỉ ~0,8–1,2 s và không phụ thuộc buffer).

---

## 11. KHUYẾN NGHỊ

1. **Chạy M1 trước khi viết bất kỳ dòng mã sản phẩm nào.** Đây là nghiên cứu khả thi; chi phí M1 thấp, và nó phủ định/khẳng định được toàn bộ đề xuất.
2. **Ưu tiên §7.4 (offscreen decode)** hơn tự viết demux, vì nó tái dùng `lib/audio-processor.js` và toàn bộ đường `audio_chunk` hiện có ⇒ bề mặt rủi ro nhỏ hơn nhiều.
3. **Đặt trần kỳ vọng đúng:** "chạy trước" chỉ có nghĩa với VOD không DRM. Với live và với nội dung DRM, đề xuất **không** cải thiện được so với hiện tại — và trong trường hợp DRM, nó còn phải **từ chối** rõ ràng.
4. **Không vượt DRM.** Cả về kỹ thuật (không có CDM) lẫn pháp lý/ToS ([Copyright Office 1201](https://copyright.gov/1201/2024/index.html)). Đây là ràng buộc thiết kế, không phải chướng ngại cần khắc phục.
5. **Giữ đường capture hiện tại là mặc định.** Nếu triển khai, nó là **đường bổ sung có điều kiện**, không phải đường thay thế.
6. **Cơ hội độ trễ rẻ hơn, không cần MSE** (đã có bằng chứng trong repo, nên làm song song hoặc trước): tấn công **đường dịch** — nút thắt e2e là dịch (~430 ms), không phải ASR (~100 ms) `[VERIFIED-NUM]` (`report/audit/XAC_NHAN_GEMINI_VA_KE_HOACH_TOI_UU.md:347-349, 369`); giảm p95 preview (đuôi 1,45 s, §4.1 của `05_measurements_and_status.md`) và đuôi `e2e_commit` p99 2,8 s (`BAO_CAO_AUDIT_HIEU_NANG_QWEN.md:76`).

---

## PHỤ LỤC A — BẢNG NHÃN BẰNG CHỨNG CHO CÁC KHẲNG ĐỊNH CHÍNH

| Khẳng định | Nhãn |
| :--- | :--- |
| Đường capture hiện tại = `createMediaElementSource` → `AudioWorklet` 1024 mẫu, PCM16 16 kHz mono | `[VERIFIED-REPO]` |
| `postMessage` PCM sang SW **không** có transfer list | `[VERIFIED-REPO]` |
| Manifest là MV3, `background.scripts`, `scripting` đã khai báo | `[VERIFIED-REPO]` |
| Popup dùng `scripting.executeScript` **không** có `world` | `[VERIFIED-REPO]` |
| Số đo độ trễ ASR/dịch/e2e | `[VERIFIED-NUM]` |
| `world: "MAIN"` trong Firefox có từ 128 | `[VERIFIED-EXT]` |
| Byte CENC tại `appendBuffer` vẫn mã hoá; `encrypted` chỉ lộ `initData` | `[VERIFIED-EXT]` ở mức "đặc tả mô tả mô hình này" |
| Công thức `presentation_ts = media_ts + timestampOffset − earliest` | `[VERIFIED-EXT]` (tồn tại trong đặc tả MSE) |
| MSE-in-Workers tồn tại và có hook riêng | `[VERIFIED-EXT]` |
| Điều 1201 DMCA có miễn trừ định kỳ | `[VERIFIED-EXT]` (văn bản tồn tại) |
| `L` (thời gian dẫn trước buffer) của từng trang | `[NOT VERIFIED]` — **chính là mục tiêu M1** |
| Chi phí ms của demux mỗi phân đoạn | `[NOT VERIFIED]` |
| `wrappedJSObject` vá prototype có hiệu lực | `[NOT VERIFIED]` |
| YouTube có gặp byte CENC hay không | `[NOT VERIFIED]` |
| Tình trạng bảo trì của các dự án tiền lệ | `[NOT VERIFIED]` |
| Chi phí structured clone ArrayBuffer theo MB | `[NOT VERIFIED]` (suy luận) |

## PHỤ LỤC B — NHỮNG ĐIỀU KHÔNG ĐƯỢC LÀM (RÀNG BUỘC)

1. **Không** sửa `extension_firefox/` hay `backend/` trong giai đoạn nghiên cứu này (tài liệu này là file duy nhất được tạo).
2. **Không** thay đổi ngữ nghĩa `appendBuffer` của trang theo bất kỳ cách nào có thể làm hỏng phát lại (R8).
3. **Không** cố vượt DRM (kỹ thuật, pháp lý, ToS).
4. **Không** biến chặn bắt MSE thành đường mặc định trước khi M4 (fallback) hoàn tất.
