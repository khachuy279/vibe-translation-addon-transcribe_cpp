# Thiết Kế Lại Pipeline B — Lookahead Video Buffering (v2)

- **Ngày**: 2026-10-01
- **Trạng thái**: ✅ Đã triển khai + kiểm thử (không regression)
- **Phạm vi**: `backend/core/stream_demuxer.py` (mới), `backend/core/lookahead_timeline.py` (mới),
  `backend/ws/lookahead_handler.py` (viết lại), `backend/asr/engine.py`, `backend/config.py`,
  `extension_firefox/` + `extension_chrome_edge/`.

---

## 1. Triệu chứng người dùng báo

> "Đã nhận được âm thanh vào server, nhưng đang gặp lỗi **mất phụ đề gốc và dịch**."

Log thật (`/ws/lookahead`) cho thấy backend VẪN chạy ASR/dịch và vẫn gửi `lookahead_subtitles`,
nhưng:

* Khối lượng audio giải mã được chỉ ~1–2 s cho mỗi 10 s video
  (`Giải mã audio chunk: +1.97s [0.0s -> 2.0s]`, `+1.51s [10.0s -> 11.5s]`, …).
* Cùng một đoạn được ASR **nhiều lần với nội dung khác nhau**
  (`28.0s-30.1s: '所以。'` rồi `28.0s-31.6s: '特斯拉。'`).
* Câu bị cắt giữa từ (`'Out two.'` → `'Out too.'` → `'That too. Why is that good?'`).

⇒ Mốc thời gian gửi về client không khớp `video.currentTime`, nên `SubtitleTimelineQueue`
không bao giờ khớp câu nào ⇒ màn hình trống.

---

## 2. Phân tích nguyên nhân gốc (có đo lường)

### 2.1. Giải mã từng mảnh MSE độc lập là bất khả thi  ⚠️ NGUYÊN NHÂN CHÍNH

Trình duyệt gọi `SourceBuffer.appendBuffer()` với các mảnh **~32 KB nằm GIỮA cluster WebM /
fragment fMP4** (YouTube/Bilibili đều vậy). Bản v1 gọi `LookaheadDemuxer.decode_chunk()`
cho từng mảnh độc lập ⇒ PyAV chỉ parse được phần đầu của mỗi mảnh.

Đo thực tế (`scratch_demux_probe.py`, WebM/Opus 12 s cắt thành mảnh 32 KB):

| Cách giải mã | Audio thu hồi |
| :--- | :--- |
| Từng mảnh 32 KB độc lập (bản v1) | **4,24 s / 12,00 s (35 %)** |
| Ghép nối liên tục rồi giải mã (v2) | **11,95 s / 12,00 s (99,6 %)** |

### 2.2. Cắt câu theo lưới 4 giây cố định

`process_lookahead_window()` quét lưới `step_sec = 4.0` và gọi ASR cho từng ô rời rạc:

* Không có VAD ⇒ câu bị cắt giữa từ, model trả chữ rác/hallucination.
* `is_pts_processed()` so khớp cả `start` lẫn `end` trong dung sai 0,4 s; khi buffer lớn dần,
  `slice_end = min(current_scan + 4, buffered_max)` **đổi giá trị** ⇒ cùng một ô bị xử lý lại
  và sinh **phụ đề trùng với nội dung khác nhau**.
* Mốc gửi về client là mốc lưới, không phải mốc người nói.

### 2.3. Init segment bị đẩy khỏi cache

`cachedAudioChunks` giới hạn 60 phần tử và bị `shift()` dần; init segment (EBML header) chỉ
được append **một lần** ở đầu mỗi SourceBuffer. Nếu người dùng bấm Start sau khi video đã chạy
vài phút thì init segment đã biến mất ⇒ không thể giải mã bất cứ mảnh nào.

### 2.4. Phía Extension: resume video sai thời điểm

`LookaheadClient` cũ gọi `_triggerPrebufferReady()` ngay khi nhận **câu phụ đề ĐẦU TIÊN** —
câu đó có thể nằm ở tận 30 s phía trước ⇒ video phát trong khi chưa có phụ đề nào cho vị trí
hiện tại.

---

## 3. Kiến trúc v2 — DÙNG LẠI ĐÚNG PIPELINE A

```text
  mảnh MSE (appendBuffer ~32 KB, cắt giữa cluster)
        │
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ StreamDemuxer  (backend/core/stream_demuxer.py)             │
  │  • giữ RIÊNG init segment (EBML / ftyp+moov)                │
  │  • ghép mọi mảnh media vào MỘT bộ đệm byte liên tục         │
  │  • mỗi lần có mảnh mới: av.open(init+media) → seek(mốc cuối │
  │    − 0,5 s) → chỉ trả frame MỚI (pts > mốc đã phát)         │
  │  • cắt tỉa bộ đệm theo ĐÚNG ranh giới Cluster/moof          │
  │  • PTS = container_pts + sourceBuffer.timestampOffset (MSE) │
  └─────────────────────────────────────────────────────────────┘
        │  StreamAudioChunk(pts_start, pts_end, pcm 16 kHz mono)
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ ContinuousAudioTimeline (backend/core/lookahead_timeline.py)│
  │  • ghép thành MỘT dòng PCM liên tục theo PTS tuyệt đối      │
  │  • khe hở ⇒ LẤP BẰNG SILENCE (bảo toàn mốc thời gian)       │
  │  • chồng lấn ⇒ cắt phần đã đọc (không vứt audio)            │
  │  • read(max_sec) theo NHU CẦU                               │
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
  VADProcessor.feed_chunk()        ← GIỐNG HỆT /ws realtime
        │  (chỉ frame ĐOẠN NÓI được chuyển tiếp, kèm PTS tuyệt đối)
        ▼
  TranscribeEngine.stream_tokens() ← GIỐNG HỆT /ws realtime
        │  (VAD → commit câu; TẮT preview vì Lookahead chỉ cần câu chốt)
        │  utterance_update(final) nay kèm {start_sample, end_sample}
        ▼
  Anchors: (sample_index → PTS tuyệt đối) ghi ngay lúc nạp
        │  ⇒ pts_start/pts_end của câu CHÍNH XÁC theo timeline video
        ▼
  Dịch GGUF  →  lookahead_subtitles {start_pts, end_pts, original, translated}
        │
        ▼
  SubtitleTimelineQueue (extension)  →  hiện ĐÚNG lúc video.currentTime đi qua
```

### 3.1. Ánh xạ sample → PTS (điểm mấu chốt)

`TranscribeEngine` chỉ nhận frame thuộc ĐOẠN NÓI (VAD lọc), nên chỉ số mẫu trong
`audio_buffer` **không** tuyến tính với thời gian video. Vì vậy handler ghi một **anchor**
`(sample_index, absolute_pts)` mỗi khi phát hiện điểm không liên tục:

```python
def _on_speech_chunk(self, pcm_bytes, ts, vad_state):
    sample_index = self.asr_engine.audio_buffer.total_written
    self._record_anchor(sample_index, ts)      # chỉ ghi khi PTS không nối tiếp
    self.asr_engine.feed_audio(pcm_bytes, timestamp=ts, vad_state=vad_state)
```

`ts` do `VADProcessor` tính (`stream_ts + offset/bytes_per_sec`) nên đã là PTS tuyệt đối.
`_pts_at(sample)` nội suy từ anchor gần nhất ⇒ mốc câu chính xác trong mọi trường hợp
(kể cả khi VAD bỏ qua những đoạn im lặng dài ở giữa).

### 3.2. Điều khiển theo nhu cầu (không đốt GPU)

Chỉ nạp audio tới `currentTime + lead_time × playbackRate + margin` (mặc định 6 s). Nhờ vậy
một video có 50 s buffer cũng không đẩy 50 s qua ASR ngay lập tức.

### 3.3. Tua video (seek)

`seek_reset` → `engine.reset_stream()` (tăng `_stream_generation` ⇒ commit cũ bị bỏ),
`vad.reset()`, `demuxer.reset(epoch giữ nguyên)`, `timeline.reset(target_time)`, xoá anchors.
Kết quả dịch đang bay có `seek_seq` cũ sẽ **không được gửi**.

---

## 4. Thay đổi phía Extension

| File | Thay đổi |
| :--- | :--- |
| `content/buffer_interceptor_poc.js` | Giữ **riêng** init segment (không bao giờ bị evict); thêm `epoch` cho mỗi SourceBuffer mới; cache 120 mảnh; replay init TRƯỚC các mảnh khi Extension yêu cầu. |
| `lib/lookahead-client.js` | Gửi `is_init`/`epoch` trong header nhị phân; **không** resume video khi nhận câu đầu tiên — chỉ resume khi backend báo `prebuffer_ready`; gửi lại yêu cầu replay sau mỗi lần tua; chuyển tiếp `lookahead_unavailable`. |
| `lib/lookahead-timeline.js` | Giữ câu thêm 0,6 s sau khi hết để không nhấp nháy; sửa việc gỡ listener (`bind` tạo hàm mới nên trước đây không gỡ được); dọn cache tất định. |
| `content/content-script.js` | Chọn Pipeline B **chỉ khi** interceptor thực sự bắt được byte audio (`cachedChunksCount > 0` hoặc `hasInitSegment`) — tránh chọn B cho site không dùng MSE; tự động quay về Pipeline A khi backend báo `lookahead_unavailable`; HUD tiến độ nạp đệm. |
| `popup/popup.html` | Thanh chọn thời gian dịch trước đổi thành **10–15 s** (bước 1 s, mặc định 15 s). |

---

## 5. Kiểm thử & Nghiệm thu

### 5.1. Test mới

| File | Nội dung |
| :--- | :--- |
| `backend/tests/test_50_lookahead_demuxer.py` | **Regression chính**: mảnh WebM/Opus 32 KB cắt giữa cluster phải thu hồi ≥ 7,8/8,0 s và không chồng lấn; nhận diện init segment; đổi epoch; dữ liệu rác không làm sập. |
| `backend/tests/test_54_lookahead_pipeline_v2.py` | `ContinuousAudioTimeline` (ghép nối/lấp silence/cắt chồng lấn/trần RAM); ánh xạ anchor qua điểm không liên tục; **E2E không cần model** (PCM → VAD → commit → ASR → dịch → phụ đề có PTS đúng); tua trong lúc dịch ⇒ huỷ kết quả; cửa sổ nạp theo `lead_time`. |
| `backend/tests/test_51_lookahead_pipeline.py` | Giao thức `/ws/lookahead`: init/ready, kẹp `lead_time` 10–15 s, `lookahead_unavailable`, mảnh nhị phân → phụ đề, bỏ mảnh mang seek cũ, khung hỏng không sập. |
| `backend/tests/test_52_lookahead_e2e_stress.py` | Tua liên tục 10 lần (0 % rò rỉ), co giãn cửa sổ theo `playbackRate`, 1 giờ audio với bộ nhớ O(1), bắt tay Dual-Engine. |
| `extension_firefox/tests/lookahead-timeline.test.js` | `SubtitleTimelineQueue`: hiện đúng theo `currentTime`, bỏ `seek_id` cũ, chống trùng, tua xoá sạch, evict ±30 s, `detach` gỡ listener. |

### 5.2. Kết quả

```text
pytest                       → toàn bộ suite PASS (0 failed)
node --test extension_firefox/tests/    → 7/7 PASS
node --test extension_chrome_edge/tests/ → 7/7 PASS
```

Bằng chứng E2E trong `test_54` (không dùng model thật, dùng `FakeInferenceEngine`/`FakeVADEngine`):

```text
PCM: [0,0–0,5) im lặng | [0,5–3,5) tiếng nói | [3,5–4,5) im lặng | [4,5–7,5) tiếng nói | [7,5–8,5) im lặng
→ lookahead_subtitles #1: start_pts = 0.475, end_pts = 4.475
→ lookahead_subtitles #2: (câu thứ hai, mốc tăng dần, không chồng lấn)
```

---

## 6. Vận hành & chẩn đoán

* Log mới: `Giải mã liên tục: +X.XXs [a -> b] (mảnh N B, tổng M đoạn, buffer K B)`
  — nếu `+X.XXs` nhỏ hơn nhiều so với `b - a` thì decoder có vấn đề.
* `lookahead_status` gửi về client mỗi 500 ms:
  `buffered_ahead`, `ready_ahead`, `fed_ahead`, `target_ahead`, `prebuffer_ready`,
  `fragments`, `decoded_chunks`, `utterances`.
* Trạng thái buffer hiển thị trong **POPUP** (`Lookahead available +Xs`), không còn HUD nổi
  trên trang. Trong console của trang vẫn có `window.__VIBE_LOOKAHEAD_DEBUG__.printReport()`.
* Nếu backend không dựng được pipeline (native ASR lỗi), nó gửi `lookahead_unavailable`
  và Extension **tự động** chuyển sang Pipeline A — người dùng không bao giờ mất phụ đề hoàn toàn.

---

## 7. Bổ sung sau lần thử thật đầu tiên (2026-10-01, v2.1)

### 7.1. Câu quá dài — BẬC 3 (STABLE_PREFIX) đã bị vô hiệu hoá

**Triệu chứng** (log thật): câu dài 10,4 s / 11,3 s / 15,5 s, hầu hết chốt bằng
`[VAD_SILENCE]` hoặc `[MAX_DURATION]`.

**Nguyên nhân**: bản v2.0 đặt `lookahead.preview_enabled = False` để tiết kiệm GPU. Nhưng
`preview` chính là **nguồn duy nhất** của BẬC 3 — cơ chế "cắt câu khi text đứng im"
(`CommitManager.evaluate_preview_stability`). Tắt preview ⇒ chỉ còn VAD-END và
MAX_DURATION (15 s) ⇒ câu rất dài.

**Khắc phục**:

* `lookahead.preview_enabled = True` — Pipeline B dùng lại ĐÚNG cơ chế cắt câu của A.
* Thêm `preview_requires_new_audio` (chỉ Pipeline B bật) + `preview_min_new_audio_sec = 0,5s`:
  * Inference preview chỉ chạy khi đã có ≥ 0,5 s audio MỚI ⇒ không đốt GPU khi audio
    không tiến (nguồn theo lô), nhịp thực tế ≤ 2 preview/s.
  * **BẬC 3 không được đánh giá trên text preview cũ**: nếu đánh giá, mỗi lần bộ nạp chờ
    dữ liệu sẽ thấy "text đứng im" và **cắt oan giữa câu** — đúng cái bẫy đã khiến bản
    đầu phải tắt preview.
* Pipeline A KHÔNG đổi hành vi (`preview_requires_new_audio=False`, `preview_min_new_audio_sec=0`).

**Kiểm chứng**: `test_lookahead_cuts_long_speech_by_text_stability` (câu 12 s phải bị cắt
trước `max_duration_sec`) và `test_lookahead_no_spurious_cut_when_audio_stalls` (audio
không tiến ⇒ tuyệt đối không cắt; chỉ chốt khi VAD đóng câu).

### 7.2. Bỏ HUD nổi, đưa trạng thái vào Popup

* `buffer_interceptor_poc.js` (v1.3): **xoá toàn bộ** widget `#vibe-lookahead-hud` (trước đây
  luôn hiện ở góc phải mọi trang web). Giữ nguyên phần hook + thống kê.
* `content-script.js`: `getLookaheadBufferStatus()` hỏi interceptor qua `postMessage`;
  `GET_STATUS` (và `window.__bsGetStatus()`) nay trả thêm `pipeline` + `lookahead`.
* `popup.html` / `popup.css` / `popup.js`: thêm dòng trạng thái
  **`Lookahead available +Xs`** (xanh khi ≥ 10 s, vàng khi đang nạp, xám khi không khả dụng),
  tự làm mới mỗi 1,2 s khi popup đang mở.

### 7.3. Stop không dừng phiên (server vẫn nhận audio)

**Triệu chứng**: bấm Stop nhưng backend vẫn nhận mảnh audio và vẫn chạy ASR/dịch.

**Nguyên nhân**: đường Stop chỉ dựa vào việc đóng WebSocket, và có hai lỗ:
1. Nếu Stop rơi vào lúc socket còn `CONNECTING` (trong ~3 s chờ backend trả
   `lookahead_ready`), `close()` không bảo đảm `onopen` không bắn ⇒ socket vẫn có thể mở và
   tiếp tục gửi mảnh.
2. `startCapture()` đang `await` (kiểm tra buffer / chờ Lookahead) không hề biết người dùng
   đã Stop: khi promise giải phóng, nó vẫn gán `isCapturing = true` và để pipeline "hồi sinh"
   trong trạng thái không ai quản lý.
3. Phía server, `handle_lookahead_ws` không có đường dừng tường minh và `session.close()`
   có thể chờ vô hạn ở `cancel_inference()` (đang giữ `_shared_lock` khi nạp model).

**Khắc phục (nhiều lớp, không phụ thuộc một mắt xích)**:

| Lớp | Thay đổi |
| :--- | :--- |
| `LookaheadClient` | Cờ `_destroyed`: **mọi** đường gửi (`_sendAudioBinaryFrame`, `sendJSON`, `_sendSyncNow`, message bridge, `_handleServerMessage`) thành no-op ngay khi Stop. |
| `LookaheadClient.disconnect()` | Gửi lệnh `{"type":"stop"}` **trước** khi đóng; nếu socket đang `CONNECTING` thì gắn `onopen = () => close()` + gỡ `onmessage/onerror` ⇒ socket không bao giờ mở rồi gửi dữ liệu. |
| `content-script` | `captureGeneration` tăng ở mỗi Start **và** mỗi `cleanup()`; `startCapture()` kiểm tra sau mỗi `await` ⇒ Stop giữa chừng thì huỷ phiên vừa dựng, không bật `isCapturing`. |
| `/ws/lookahead` | Xử lý `{"type":"stop"/"bye"/"close"}` ⇒ thoát vòng nhận ngay; kiểm tra `safe_conn.is_closed` mỗi vòng; `session.close()` có trần thời gian (`wait_for`) và `await safe_conn.close()` tường minh. |

**Kiểm chứng**: `test_lookahead_stop_command_ends_session` (backend) và 3 test trong
`extension_firefox/tests/lookahead-client.test.js`
(`disconnect ... CHẶN mọi frame sau đó`, `disconnect khi socket còn CONNECTING`,
`bỏ qua thông điệp server sau khi Stop`).

### 7.4. Pipeline B phải chạy ĐÚNG thông số trong Popup

**Trước đây**: `lookahead_init` chỉ gửi `source_lang` / `target_lang` / `lead_time`. Mọi
thiết lập khác trong popup (VAD engine, VAD threshold, VAD silence, model ASR, model dịch,
`minWordsToCommit`, cắt câu theo độ ổn định, `stabilityMinDurationSec`, `stabilityMinWords`,
`holdShortSentence`, cửa sổ preview, nhịp poll) đều bị **bỏ qua** — Pipeline B chạy bằng mặc
định của server.

**Bây giờ**:

* Extension gửi kèm `buildWsConfig(settings)` (đúng payload như Pipeline A) trong
  `lookahead_init`, và gửi `set_config` khi người dùng đổi thiết lập lúc đang chạy
  (`LookaheadClient.updateConfig`).
* `LookaheadSessionState.apply_config()` / `init_components()` áp: ngôn ngữ nguồn/đích,
  VAD engine + threshold + silence + enabled, `min_words_to_commit`, toàn bộ tham số phân câu,
  cửa sổ preview, nhịp poll; model ASR và model dịch đổi qua `hotswap` trong tác vụ nền
  (không chặn vòng nhận tin) kèm bản tin `model_status`.
* VAD được dựng **ngay** với thông số popup (không phải chờ cơ chế đổi engine ở chunk sau).
* Các override riêng của Lookahead (`inactivity_timeout_sec`, `preview_requires_new_audio`,
  `preview_min_new_audio_sec`, `poll_interval_ms`) được áp **sau cùng** để luôn thắng.
* Cấu hình phân câu là **bản sao** theo phiên ⇒ không đụng Pipeline A.
* Bộ lọc `min_words_to_commit` của Lookahead nay đọc cấu hình **của phiên** thay vì mặc định
  toàn cục.

**Chưa hỗ trợ**: TTS (lồng tiếng) trong Pipeline B — **đã bổ sung ở §7.5**.

**Kiểm chứng**: `test_lookahead_init_applies_popup_vad_and_sentence_settings`,
`test_lookahead_live_config_update_changes_vad_and_min_words`,
`test_lookahead_filters_short_sentences_by_popup_min_words`, và 2 test JS
(`gửi đủ cấu hình popup trong lookahead_init`, `updateConfig đẩy thiết lập mới sang backend`).

### 7.5. Lồng tiếng (TTS) cho Pipeline B — phát đúng lúc & KHÔNG trễ dây chuyền

**Bài toán kép**:
1. Ở Pipeline B, phụ đề có sẵn TRƯỚC `lead_time` giây ⇒ không thể "phát ngay khi nhận được"
   như Pipeline A; audio phải đợi tới lúc `video.currentTime` đi qua `start_pts`.
2. Nếu câu đọc **dài hơn cửa sổ phụ đề** mà cứ phát nối đuôi, mọi câu sau bị **đẩy lùi**
   ⇒ lồng tiếng trễ dần và không bao giờ bắt kịp.

**Ba tầng chống trễ** (theo thứ tự can thiệp):

| Tầng | Ở đâu | Việc làm |
| :--- | :--- | :--- |
| 1 | Backend `OmniVoiceTTS.synthesize_fitted_sync` | Sinh audio ở tốc độ người dùng, rồi **nén thời gian giữ nguyên cao độ** (phase vocoder) để câu đọc nằm gọn trong ngân sách `(end_pts - start_pts) + 0,6 s`, trần `tts_max_speed = 1,45×(tốc độ gốc)`. |
| 2 | Client `TtsTimelineScheduler._play` | Tính `playbackRate` còn thiếu cho ĐÚNG khoảng trống tới **câu kế tiếp** (`nextStart - startPts`), trần `maxRate = 1,35`, nhân thêm `video.playbackRate`. |
| 3 | Client `_tick` | Mỗi câu có `stopAtPts` **tuyệt đối**: quá mốc là CẮT, nhường chỗ cho câu sau. Mọi câu được kích hoạt bằng mốc thời gian video ⇒ **sai số không bao giờ tích luỹ**; xấu nhất chỉ là một câu bị cụt đuôi. |

Quy tắc bổ sung:

* Câu tới muộn quá `lateToleranceSec = 0,3 s` (video đã chạy qua đầu câu) ⇒ **BỎ**, không phát
  lệch ("thà mất một câu còn hơn lệch tiếng").
* Video **tạm dừng** ⇒ dừng lồng tiếng (không đọc trước); video phát 1,5x ⇒ lồng tiếng 1,5x.
* Tua video / Stop ⇒ `clear()` dừng ngay câu đang đọc và xoá hàng đợi.
* Audio TTS đi bằng **khung nhị phân** `[4B hdrlen][JSON][WAV]` (cùng bố cục khung audio
  client gửi lên) — không base64, không phát sinh +33 %.
* TTS chạy trong **worker riêng** (không chặn vòng ASR/dịch); câu nào đã trôi qua thì bỏ
  không tổng hợp. Đổi `ttsEnabled` giữa phiên sẽ tạo/huỷ worker (idempotent).
* Ducking tiếng gốc dùng lại nguyên cơ chế của Pipeline A (`TTSAudioPlayer.setTargetVideo` +
  volume guard). Ở Pipeline B không có `createMediaElementSource` nên ducking bằng
  `video.volume` **không** ảnh hưởng ASR.

**Cấu hình mới** (`backend/config.py → LookaheadConfig`): `tts_max_speed = 1.45`,
`tts_tail_allowance_sec = 0.6`, `tts_max_budget_sec = 12.0`.

**Chẩn đoán**: log `TTS Lookahead [a-b]: đọc X.XXs / cửa sổ Y.YYs (synth ...ms)` phía server và
`[BS TTS-LA] @a đọc X / cửa sổ Y -> rate Z (trễ ...ms)` phía client; khi Stop in tổng kết
`phát / bỏ muộn / cụt đuôi / trễ nhất / rate tối đa`.

**Kiểm chứng**: 5 test backend (`synthesize_fitted_sync` nén đúng ngân sách & tôn trọng trần
tốc độ, xếp hàng theo `ttsEnabled`, bố cục khung nhị phân, tạo worker khi bật giữa phiên) và
**13 test JS** trong `extension_firefox/tests/tts-timeline.test.js` — trong đó có ca trọng tâm
"câu đọc 5 s trong cửa sổ 3 s: bị nén + bị CẮT để câu sau vẫn đúng giờ".

### 7.6. Ba lỗi phát hiện sau lần thử thật thứ hai (2026-10-01)

#### (a) Câu phụ đề ĐẦU TIÊN không hiển thị, phải đợi vài câu

Hai nguyên nhân chồng nhau:

1. `resumePlayback()` (chạy khi backend báo `prebuffer_ready`) gọi `om.clear()` để xoá dòng
   "⏳ Đang nạp…", nhưng `SubtitleTimelineQueue.activeSubtitle` vẫn trỏ vào câu đang đúng ⇒
   `_tick` thấy "không đổi" nên **không bao giờ vẽ lại** cho tới khi sang câu kế tiếp.
2. Renderer tự cho câu **hết hạn theo đồng hồ thực**. Video bị tạm dừng để nạp đệm nên câu
   hiển thị trong lúc chờ có thể hết hạn trước khi video chạy.

Khắc phục:

* `SubtitleTimelineQueue.refreshNow()` — buộc vẽ lại câu đang khớp ngay (gọi trong
  `resumePlayback` sau `om.clear()`).
* **Keep-alive**: khi vẫn là cùng một câu, phát lại sự kiện phụ đề mỗi `keepAliveMs = 1200 ms`
  ⇒ renderer gia hạn câu, không bị mất giữa câu.
* Test: `refreshNow() vẽ lại câu đang khớp sau khi overlay bị clear()`,
  `giữ sống câu đang hiển thị bằng cách phát lại định kỳ`.

#### (b) Tua video KHÔNG tạm dừng (sai so với thiết kế)

`_onVideoSeeking` chỉ bật cờ nội bộ `_isPrebuffering` mà **không ai lắng nghe**
(`onBufferingStateChange` không được nối ở Pipeline B) ⇒ video chạy tiếp trong lúc backend
còn đang dịch lại từ vị trí mới.

Khắc phục:

* `_onVideoSeeking` luôn gọi `_startPrebuffering()`; content script nối
  `onSeekTriggered` → `pauseForBuffering("seek")`: tạm dừng video, hiện dòng
  "Đang dịch trước đoạn vừa tua…", đặt **chốt an toàn 8 s**.
* `resumePlayback()` không còn là one-shot: nó dùng cờ `lookaheadPausedVideo` nên hoạt động
  cho CẢ lần bắt đầu VÀ mỗi lần tua; khi resume thì `refreshNow()` để vẽ lại phụ đề ngay.
* `LookaheadClient` **bỏ qua `lookahead_status` mang `seek_id` cũ** — nếu không, trạng thái
  "sẵn sàng" của đoạn trước sẽ phát video trở lại trước khi backend kịp dịch vị trí mới.
* Test: `tua video LUÔN bật trạng thái nạp đệm`,
  `bỏ qua lookahead_status của seek CŨ`.

#### (c) TTS tổng hợp xong nhưng KHÔNG nghe thấy

Trong log thật, câu lồng tiếng đầu tiên có `start_pts = 155,89 s` trong khi video mới ở
~126 s ⇒ **chưa tới lúc phát** (đúng thiết kế, nhưng trước đây không có log nào cho biết
điều đó). Đồng thời có một rủi ro thật: `AudioContext` tạo trong callback mạng (không phải
cử chỉ người dùng) có thể ở trạng thái `suspended`; `resume()` bị từ chối ⇒ audio tổng hợp
xong mà **im lặng**.

Khắc phục:

* **Log chẩn đoán ở mọi mốc**: `[BS TTS-LA] Nhận lồng tiếng đầu tiên…`, `▶ phát @… (rate …)`,
  `✖ bỏ câu muộn …`, và cảnh báo khi AudioContext không chạy.
* **Mở khoá Web Audio theo cử chỉ người dùng**: `unlockAudio()` gắn vào `click`/`play`/`keydown`
  của trang (capture) + gọi ngay khi attach.
* **Đường dự phòng `<audio>`**: nếu `AudioContext` không ở trạng thái `running` (hoặc
  `decodeAudioData` lỗi), audio WAV được phát bằng `HTMLAudioElement` với Blob URL — vẫn giữ
  `playbackRate`, mốc cắt và thứ tự. `snapshot()` nay báo cả `audioContextState`.
* Test: `AudioContext bị treo ⇒ phát bằng thẻ <audio> (không im lặng)`,
  `unlockAudio() resume AudioContext khi người dùng tương tác`.

---

### 7.10. Lần thử thật thứ tư (2026-10-01 17:17) — trang NGOÀI YouTube chỉ hiện 1/3–5 câu

**Triệu chứng**: Pipeline B chạy tốt trên YouTube, nhưng ở một trang khác phụ đề "chập chờn":
khoảng 3–5 câu mới hiện được 1 câu, dù `lead_time` rất xa.

**Chẩn đoán** (số đo thật, `[ASR] CHẨN ĐOÁN NĂNG LỰC` mới thêm):

```text
audio vào 913.9 KB/s → PCM giải mã 3.80x  (burst: replay cache của interceptor)
audio vào  16.6 KB/s → PCM giải mã 0.40x  (trạng thái ổn định)
Giải mã ĐÓI AUDIO: 20.0s không sinh ra PCM mới trong khi vẫn nhận 342036 B audio
```

16,6 KB/s audio tương đương đúng 1× thời gian thực ⇒ **342 KB ≈ 20 s audio đã về nhưng chỉ
4,10 s được giải mã** (mất 80 %). Thời gian wall của mỗi lượt giải mã chỉ **0,1 s** ⇒ KHÔNG
phải nghẽn tốc độ, mà là **mốc tiến độ (`_last_pts`) chạy vượt phần audio thực sự đã phát ra
PCM**, khiến lượt sau lọc `pts < _last_pts` và bỏ vĩnh viễn đoạn chênh lệch. Đây cũng là lý do
`fed_ahead` luôn đứng ở 21 s (đúng trần nạp) trong khi `ready_ahead` tụt về 0.

**Nguyên nhân gốc trong `StreamDemuxer._decode_new()`** — hai lỗi trộn trục thời gian:

| # | Lỗi | Hệ quả |
| :--- | :--- | :--- |
| 1 | `_last_pts` được cập nhật theo `pts + frame_dur` của **MỌI** frame, kể cả frame mà resampler chưa nhả mẫu PCM nào | Mốc tiến trước phần PCM thực có; lượt sau bỏ đúng phần bị vượt |
| 2 | `container.seek()` dùng mốc trên trục **container**, còn việc lọc frame dùng mốc trên trục **media** (`pts + timestampOffset`) | Với SourceBuffer có `timestampOffset` ≠ 0 (trang ngoài YouTube hay dùng), seek vượt cuối container ⇒ mất gần hết audio |

Thêm một lỗi phụ: chunk vừa tạo bị cắt theo `self._last_pts` **sau khi** đã cập nhật ⇒ tự cắt
mất phần vừa giải mã (phát hiện ngay khi chạy test, đã sửa bằng cách cắt lúc `flush_run()`).

**Khắc phục** (`backend/core/stream_demuxer.py`):

* Mọi so sánh mốc nằm trên **một trục duy nhất** = `container_pts + timestampOffset`
  (`_last_pts` cũng là mốc đã-phát trên trục đó); `container.seek()` trừ `bias` trước khi đổi
  sang đơn vị `time_base`.
* `_last_pts` **chỉ tiến khi PCM thực sự được phát ra** (`flush_run()`), kèm cắt phần chồng lấn
  0,5 s ở mép đầu đoạn ⇒ không lặp 20 ms mỗi lượt mà cũng không bỏ sót frame nào.
* Bộ đếm chẩn đoán mới: `coverage_report()` → `emitted`, `last_pts`, `skipped_frames`,
  `gaps_in_media`, `seek_fallbacks`; in kèm mỗi dòng `CHẨN ĐOÁN NĂNG LỰC`.

**Kiểm chứng**: 2 test mới trong `backend/tests/test_50_lookahead_demuxer.py`:

* `test_stream_demuxer_pts_frontier_never_passes_emitted_audio` — mảnh 16 KB, khẳng định
  `last_pts <= tổng audio đã phát + 1 frame`, các đoạn liền nhau không hở/chồng.
* `test_stream_demuxer_uses_timestamp_offset_on_one_time_axis` — `timestamp_offset = 3600 s`
  vẫn phải thu hồi ≥ 7,8/8,0 s và PTS nằm đúng trên trục video.

Toàn bộ suite backend PASS sau khi sửa.

### 7.11. Lần thử thật thứ năm (2026-10-01 17:25) — cache mảnh lệch khỏi vị trí phát

**Triệu chứng**: phiên neo ở **đầu video** (`current_time = 1,35 s`) nhưng PCM giải mã ra lại
nằm ở **144 s trở đi**:

```text
Lookahead init: neo vị trí phát hiện tại @1.35s
Giải mã liên tục: +6.15s [144.10s -> 150.26s]
Giải mã liên tục: +8.22s [152.04s -> 160.26s]
CHẨN ĐOÁN NĂNG LỰC ... đệm ĐÃ DỊCH 0.0s (fed 21.0s) ... bỏ_xa=0
```

Lưu ý: `skipped_frames=0` và `gaps_in_media=0` ⇒ logic mốc của demuxer đã ĐÚNG (không còn bỏ
audio trong bộ đệm); vấn đề nằm ở chỗ dữ liệu nhận được không thuộc vùng đang phát.

**Nguyên nhân**: interceptor replay cache **120 mảnh gần nhất**. Ở trang tải trước rất sâu
(video element báo `buffered_end = 278 s`), 120 mảnh đó chỉ phủ **đoạn cuối** của vùng đã tải
(~144 s → ~399 s) — không có mảnh nào quanh vị trí phát. Hệ quả kép:

1. `ContinuousAudioTimeline` phải **lấp hàng trăm giây im lặng** để đi từ 1,35 s tới 144 s ⇒
   đốt CPU vô ích và `fed_ahead` luôn ở trần 21 s.
2. Mọi câu nhận dạng được mang mốc **143 s TRONG TƯƠNG LAI** ⇒ `SubtitleTimelineQueue` không
   bao giờ phát chúng ⇒ màn hình trống.

**Khắc phục (hai lớp)**:

* **Backend** (`lookahead.max_decode_lead_sec = 240 s`): PCM giải mã ra mà bắt đầu xa hơn ngưỡng
  này phía trước vị trí phát thì **KHÔNG nạp vào timeline** (đếm ở `chunks_dropped_far`, log
  WARNING). Pipeline chỉ chờ mảnh mới đúng vị trí phát — vẫn luôn có phụ đề.
* **Extension** (`buffer_interceptor_poc.js`, cả Firefox và Chrome/Edge): mỗi mảnh cache được
  gắn `videoPts` (mốc video lúc append); khi replay chỉ gửi mảnh trong cửa sổ
  `[playhead − 8 s, playhead + 45 s]` và log rõ số mảnh bị bỏ. Nhờ vậy cache **luôn** liên quan
  tới vị trí đang phát, không phụ thuộc việc trang tải trước bao xa.

**Kiểm chứng**: `test_decoded_pcm_far_ahead_of_playhead_is_dropped` (media neo ở 1000 s, vị trí
phát 1 s ⇒ bỏ hết, timeline rỗng; media quanh vị trí phát ⇒ nhận bình thường).

**Số đo đã sửa cho đúng**: bỏ tỷ lệ `PCM/byte` — nó giả định 160 kbps nên gây hiểu sai (log cũ
hiện `0.05x` trong khi bitrate thật chỉ ~17 KB/s). Thay bằng `media giải mã` (giây media thật),
`bỏ_xa` và `lượt có PCM`, tách hẳn khỏi `bytes_in` là byte thô.

### 7.12. Lần thử thật thứ sáu (2026-10-01 17:36) — mảnh bị replay TRÙNG nhiều lần

Sau §7.11 phụ đề đã hiện lại (`ready_ahead` 15–20 s, `prebuffer_ready=True`), nhưng còn hai
điều bất thường trong log:

```text
audio vào 1034.9 KB/s            ← 10 s đầu, gấp ~60x bitrate thật
CẤU TRÚC NGUỒN: container=mp4, n_audio_streams=1, codec=aac, rate=44100,
                frames=15851, pts=[291.22, 361.23], span=70.0s
```

15.851 frame AAC @ 44,1 kHz = **368 giây audio** nhưng chỉ trải trên **70 giây** timeline ⇒
**dữ liệu nhận được có ~5,3 bản sao của cùng đoạn media**.

**Nguyên nhân**: `LookaheadClient.requestInitialChunks()` được gọi từ **3 đường**
(`_handleConnected`, `lookahead_ready`, `seek_acknowledged`) và mỗi lần interceptor lại
`postMessage` **toàn bộ** cache; thêm nữa `_handleConnected` còn flush `_pendingChunks` (những
mảnh đã nằm sẵn trong cache). Kết quả: một cache ~10 MB được append 2–3 lần ⇒ backend phải giải
mã và lọc trùng gấp ~5 lần, làm đói cả pipeline (mỗi lượt chỉ ra 1,5–8 s PCM mới).

**Khắc phục (ba lớp)**:

* **Extension — dedup theo token**: `requestInitialChunks(currentTime, token)` gửi kèm
  `replayToken`; interceptor bỏ qua yêu cầu nếu token trùng lần replay trước. Token là
  `init_<seekId>` cho đầu phiên (gọi bao nhiêu lần cũng chỉ replay một lần) và `seek_<seekId>`
  cho mỗi lần tua (mỗi mốc mới replay đúng một lần).
* **Extension — không flush trùng**: `_pendingChunks` chỉ gửi những mảnh NẰM NGOÀI cửa sổ cache
  sắp replay; đồng thời `lookahead_ready` không còn gọi `requestInitialChunks()` nữa.
* **Backend — chốt an toàn**: `handle_audio_fragment` băm nội dung mảnh (`blake2b`, cửa sổ
  15 s) và BỎ mảnh trùng; đếm ở `fragments_deduped`/`mảnh_trùng` (cảnh báo mỗi 50 mảnh). Cửa
  sổ thời gian để mảnh nạp lại hợp lệ sau khi tua vẫn được chấp nhận.

**Chẩn đoán bổ sung**: `structure_report()` nay in `duration/span` (≈1,0 là bình thường;
>1,2 là có trùng lặp) — dấu hiệu định lượng phân biệt "trùng lặp" với "mất audio".

**Kiểm chứng**: `test_duplicate_audio_fragments_are_ignored` (gửi lại y hệt 5 lần ⇒
`fragments_deduped == 5`, bộ đệm ghép nối KHÔNG phình; mảnh khác vẫn được nhận).

### 7.13. Lần thử thật thứ bảy (2026-10-01 18:12) — trùng lặp đã hết, còn "decoder dừng sớm"

**Kết quả tốt**: `duration/span=1.00x` (hết trùng lặp), `mảnh_trùng=0`, phụ đề chạy với
`ready_ahead` 7–19 s và mốc khớp video.

**Còn lại**: mỗi lượt giải mã vẫn chỉ ra **1,5–8 s PCM** rồi ~20 s sau mới ra tiếp, dù
`CẤU TRÚC NGUỒN` cho thấy bộ đệm ĐẶC (`duration/span = 1,00`, `gaps_in_media=0`,
`skipped_frames=0`). Byte vào 67 KB/s gấp ~4× bitrate audio thật (16,5 KB/s).

**Đã làm trong vòng này (đo được, không phải suy đoán)**:

1. **Đo chuỗi fragment fMP4 thật** — `scan_mp4_fragment_times()` đọc `tfhd.track_ID` +
   `tfdt.baseMediaDecodeTime` của từng `moof`; `_mp4_fragment_chain_report()` quy về giây và
   tự chuẩn hoá theo bước fragment điển hình (vì `tfdt` là mốc BẮT ĐẦU nên hiệu hai mốc chính
   là độ dài fragment — nếu so với ngưỡng 0,6 s sẽ báo khe hở GIẢ). Đây là phép đo phân biệt
   dứt khoát "mảnh chưa bao giờ được append" với "decoder không lấy ra hết".
2. **Đường dự phòng bóc AAC thô** (`_decode_raw_aac`): khi đường chuẩn không ra PCM mà bộ đệm
   đã có byte media, bóc thẳng frame trong `mdat` → ADTS → decode. Có **chốt chống báo động
   giả**: quét `0xFFF` trên AAC thô khớp giả rất nhiều (đo được 71 "frame" rác trên một file
   thật), nên chỉ dùng khi độ phủ byte ≥ 90 % (`_raw_fallback_rejected` ghi số lần từ chối).
3. **Sửa hai lỗi trong chính bộ đọc box** (phát hiện nhờ chạy trên fMP4 thật):
   `tfdt`/`tfhd` nằm ở `data_start + 0` (không phải `+4`) vì `_find_box` đã trả offset NGAY SAU
   header box.

**Kiểm chứng**: `test_mp4_fragment_chain_report_on_real_fmp4` (fMP4 do PyAV ghi: `track_id`
đúng, `tfdt` tăng dần, báo `LIỀN MẠCH`), `test_adts_frame_parser_roundtrip`,
`test_raw_aac_fallback_refuses_false_positives`. Toàn bộ suite: **526 passed**.

### 7.14. Lần thử thật thứ tám (2026-10-01 18:22) — cache mảnh vẫn lệch, và cách chữa tận gốc

Log cho hai thông tin quyết định:

```text
Lookahead init: neo vị trí phát hiện tại @2.68s
Giải mã liên tục: +5.02s [74.44s -> 79.46s]          ← mảnh ĐẦU TIÊN đã ở 74,44 s
audio vào 4195 KB trong 4 giây (419 KB/s)            ← vẫn replay NGUYÊN cache
CẤU TRÚC NGUỒN: tracks={2: 44}, tfdt_giây=[74.44, 276.22], KHE_HỞ=9 (thiếu ~46s),
                duration/span=1.24x
```

Hệ quả đúng như người dùng báo: **74 giây đầu không có phụ đề** (audio của đoạn đó chưa bao
giờ tới backend), rồi từ ~74 s phụ đề mới chạy (3 commit: 74,50–77,53 / 77,58–79,59 s).

**Nguyên nhân gốc**: cache "120 mảnh gần nhất" của interceptor. Trang này tải trước tới ~279 s;
khi prefetch nhảy xa, 120 mảnh **chỉ toàn đoạn ở tương lai** ⇒ mất audio quanh vị trí phát
dù cache "đầy". Đây là lỗi thiết kế của trần đếm-mảnh, không phải lỗi parser.

**Khắc phục**:

1. **Extension — cache bám vị trí phát** (thay trần 120 mảnh): giữ
   `[playhead − 30 s, playhead + 180 s]` + trần 12 MB, prune sau mỗi lần append
   (`pruneCacheAroundPlayhead`). Cache KHÔNG THỂ trôi xa khỏi vị trí phát nữa.
2. **Backend — giải mã TỪNG fragment** (`_decode_fragments_individually`): ghép `init` với
   từng cặp `moof`+`mdat` rồi giải mã riêng, nên một mảnh hỏng chỉ mất chính nó thay vì chặn
   toàn bộ phần sau. Chỉ chạy khi đường nối-liền chưa lấy hết audio (`_last_audio_tfdt_sec`)
   và số mảnh ≤ 200 (chặn chi phí). Đo trên fMP4 thật: **12,0 s / 12,0 s qua 6 mảnh**.
3. **Backend — cảnh báo lệch đầu phiên**: nếu PCM đầu tiên tới muộn hơn vị trí phát > 20 s thì
   log WARNING nói rõ "đoạn đầu sẽ không có phụ đề" + nguyên nhân (cache chưa bám vị trí phát),
   để không phải suy đoán ở lần sau.

**Lưu ý vận hành**: cả ba lần thử gần đây đều cho thấy phần **extension chưa được nạp lại**
(`audio vào` tăng vọt 4 MB trong 4 s = replay nguyên cache; mảnh đầu vẫn lệch hàng chục giây).
Sửa ở extension chỉ có hiệu lực sau khi **Reload** trong `about:debugging` **và tải lại trang**.

**Kiểm chứng**: `_decode_fragments_individually` đo trực tiếp 12/12 s; toàn bộ suite backend
**526 passed**; JS extension **46/46** cho cả Firefox và Chrome/Edge.

### 7.15. Lần thử thật thứ chín (2026-10-01 18:31) — đường "giải mã từng fragment" ĐÃ CHẠY

Đây là lần đầu phụ đề hiện **liên tục** ngay từ giây thứ 2. Log xác nhận cơ chế mới hoạt động:

```text
Đường giải mã nối-liền không ra PCM — đã chuyển sang giải mã TỪNG fragment (8.3s)
Giải mã liên tục: [1.70→10.03] [10.05→15.07] [15.07→20.06] [20.09→25.05] [25.05→30.07] …
duration/span=1.00x          (hết trùng lặp)
giải_mã_từng_mảnh=63         (đường dự phòng chạy liên tục)
```

Trước đây mỗi lượt chỉ ra 1,5–8 s rồi hụt 20 s; nay các đoạn **nối liền nhau** và phụ đề ra
đều (`Stand please. Excuse me.`, `What happened?`, `I was diagnosed with anemia.`,
`すごく嬉しかったです。お座りください。`, …).

**Còn một lỗi trong chính đường dự phòng, đã sửa**: `_decode_fragments_individually` chỉ lấy
**200 mảnh ĐẦU** bộ đệm, nên sau khi phát hết phần đầu nó giải mã lại mãi phần cũ và không bao
giờ tới các mảnh xa hơn (bộ đệm 6 MB / 3000+ mảnh). Nay nó:

* **nhắm đúng vùng cần**: chỉ lấy các mảnh có `tfdt ≥ _last_pts − 0,05` (ngân sách 200 mảnh);
* **quét từ offset của mảnh cần** (`_first_offset_at_or_after` + tìm nhị phân trên mốc thời
  gian đã cache) thay vì duyệt toàn bộ `moof` mỗi lượt;
* **chỉ lấy mảnh của track AUDIO** (`scan_mp4_track_at` đọc `tfhd.track_ID` theo từng box), nên
  không trộn lẫn khi dòng có nhiều track.

**Kiểm chứng**: đo trực tiếp trên fMP4 do PyAV ghi — đặt `_last_pts = 14,14 s` (giữa bộ đệm)
thì đường dự phòng trả đúng **15,86 s từ 14,14 s → 30,00 s** (bắt đầu ≥ mốc đã phát). Toàn bộ
suite **526 passed**; JS **46/46**.

**Còn tồn tại (không phải lỗi pipeline)**: ở phiên này site chỉ có audio từ ~10 s trở đi trong
vùng đã tải, nên cảnh báo "Audio nhận được bắt đầu ở … lệch …" xuất hiện; pipeline vẫn chạy
đúng phần audio mà nó có. Cảnh báo này là công cụ để phân biệt với lỗi logic.

### 7.16. Lần thử thật thứ mười (2026-10-01 21:27) — hết mất câu, canh mốc và một lỗi trần RAM

Người dùng xác nhận **phụ đề đã hiện đủ**, chỉ còn **sớm hơn tiếng nói ~0,5–1 s**.

**Lỗi thật phát hiện trong log** (sẽ làm pipeline đứng sau ~4 phút):

```text
CẤU TRÚC NGUỒN: ... media=8014927B ... frames=0, pts=[None, None], n_moof=0
demuxer: emitted=239.1s, last_pts=-infs
```

Chuỗi nhân–quả:

1. Bộ đệm ghép nối vượt trần 8 MB. `_trim()` cũ, khi không đủ ranh giới cluster, **cắt thô
   giữa cấu trúc** ⇒ dòng byte mất đồng bộ.
2. `scan_mp4_boundaries` không còn tìm thấy `moof` nào (`n_moof=0`) ⇒ cả đường nối-liền lẫn
   đường từng-mảnh đều không có gì để giải mã.
3. `_last_pts` giữ `-inf` ⇒ mọi lượt sau lại bắt đầu lại từ mảnh đầu ⇒ `emitted` đứng yên
   vĩnh viễn. Cảnh báo "Audio nhận được bắt đầu ở …" in mỗi lượt (nhiễu log).

**Khắc phục**:

* `_trim()` **không bao giờ cắt thô**: chỉ cắt tại ranh giới cluster/`moof` thật và luôn chừa
  ít nhất một ranh giới; không tìm được ranh giới an toàn thì **giữ nguyên** + WARNING (thà
  tốn RAM hơn hỏng luồng byte). Cache offset/mốc bị xoá sau mỗi lần cắt.
* Nới trần quét `moof` (4096 → 100000) và chỉ ghép offset↔mốc khi **hai danh sách cùng độ dài**.
* `_decode_fragments_individually` khi không ra PCM vẫn **ghi nhận mốc vùng đã xử lý** để lượt
  sau tiến tiếp, không lặp lại 200 mảnh cũ.
* Cảnh báo lệch đầu phiên chỉ in **một lần** mỗi phiên.

**Kiểm chứng**: đo trực tiếp với trần bộ đệm nhỏ (120 KB) để buộc trim 4 lần trên file 24 s →
thu hồi **24,01/24 s**, ranh giới `moof` còn nguyên (`test` cục bộ + toàn bộ suite 526 pass).

**Canh mốc (sớm 0,5–1 s)**: nới dải thanh "Đồng bộ phụ đề / lồng tiếng" từ ±600 ms lên
**±1500 ms** (khớp trần phía backend), cả `popup.html` và clamp trong `content-script.js`.
Hệ thống đã tự bù 50 ms cho FireRed; phần lệch còn lại là đặc thù từng trang nên người dùng
tinh chỉnh bằng thanh này (tăng `+` = hiện muộn hơn). Trị số gợi ý ban đầu cho trang này:
**+500…+1000 ms**.

### 7.17. ĐƠN GIẢN HOÁ: seek = phiên mới (theo đề xuất người dùng)

Đề xuất: "mỗi lần tua thì xoá toàn bộ âm thanh/phụ đề/TTS và reset như phiên mới, tạm dừng
video, đợi có bản dịch ngay tại vị trí đó (+TTS) rồi mới phát". Đây là thiết kế ĐÚNG và làm
mất hẳn một lớp lỗi phức tạp. Đã áp dụng:

* **XOÁ cơ chế tự đoán "lệch timeline"** (`sync_state` so cursor với `currentTime`): nó tạo
  VÒNG LẶP reset mỗi 3 s vì sau mỗi lần reset cursor chưa kịp tiêu thụ audio nào (đo thật:
  13 lần reset liên tiếp, mỗi lần xoá ASR ⇒ phụ đề "lúc hiện lúc không"). Nay **mọi lần tua
  đi qua đúng một đường**: `handle_seek` (từ `seek_id` mới hoặc `seek_reset`) ⇒ reset toàn bộ
  audio + commit + neo + hàng đợi TTS. Bớt hẳn `_cursor_stalled_sec`, `_STALL_RESYNC_SEC` và
  hai trường theo dõi cursor.
* **Client vẫn là bên quyết định "đã đủ để phát lại"**: `_onVideoSeeking` xoá queue phụ đề +
  `ttsTimeline.clear()` + `pauseForBuffering("seek")`, và chỉ `resumePlayback()` khi
  `prebuffer_ready` (hoặc TTS đầu tiên tới).
* **Chờ sau khi tua thông minh hơn (không cắt cứng 3,5 s)**: hạn chờ 5 s (6 s nếu bật TTS) và
  **gia hạn thêm 2 s (tối đa 3 lần) khi `ready_ahead`/`fed_ahead` còn tăng** — tức đợi tới khi
  thật sự có bản dịch tại vị trí mới, nhưng vẫn có trần để không treo trình phát.

**Điều đơn giản hoá này KHÔNG giải quyết được** (và cần nói rõ): nguồn audio. Nếu tua vào vùng
trình phát ĐÃ tải sẵn, site không append lại ⇒ không có byte nào tới backend, nên không có gì
để dịch dù reset sạch cỡ nào. Đó là lý do vẫn giữ cơ chế **tải lại phân đoạn media** ở
`buffer_interceptor_poc.js` (ghi URL + `Range` qua hook `fetch`/`XHR`, tải lại khi cache không
có mảnh quanh vị trí phát) — reset lo phần TRẠNG THÁI, tải lại lo phần DỮ LIỆU.

**Kiểm chứng**: toàn bộ suite **526 passed**; JS **46/46** cả hai bản extension.

### 7.18. Tua vào vùng YouTube ĐÃ tải sẵn — sửa hai lỗi trong cơ chế tải lại

Log thật (2026-10-01 22:24) cho thấy cơ chế tải lại phân đoạn **bị chặn bởi chính bộ lọc trùng**:

```text
22:24:41.395  Yêu cầu client gửi lại mảnh quanh vị trí phát 4.3s     ← đã yêu cầu
22:24:41.748  Bỏ mảnh audio TRÙNG (tổng đã bỏ: 1) … 51 … 53          ← dedup chặn chính nó
22:24:51      Seek reset target_time=28.59s
22:24:58.685  Giải mã liên tục: [60.00s -> 60.53s]                   ← vẫn 30 s SAU vị trí tua
```

Hai lỗi thiết kế:

1. **Chọn sai khoảng byte**: mã cũ lấy `mediaRanges[last]` (range CUỐI đã tải) chứ không quy đổi
   từ VỊ TRÍ TUA ⇒ tải lại đúng chỗ vô ích.
2. **Mảnh tải lại bị dedup chặn**: byte của nó trùng mảnh đã nhận nên backend bỏ — cơ chế tải lại
   không bao giờ cứu được vùng đã buffer.

**Khắc phục**:

* `refetchForPlayhead(playhead, bytesPerSec)`: quy đổi vị trí tua → byte ước lượng
  (`playhead × bytesPerSec`, lùi 3 s cho chắc), chọn tối đa 8 range gần nhất, bỏ qua range đã
  tải lại, và chặn trần `refetchMaxBytes` (6 MB).
* Cờ `refetched` đi suốt chuỗi **interceptor → content script → header khung nhị phân →
  `handle_audio_fragment(refetched=True)`**, và backend **miễn dedup** cho mảnh này
  (`if (not refetched) and …`). Vì đây là dữ liệu do chính ta yêu cầu, trùng lặp là đúng ý.

**Còn phụ thuộc dữ liệu thật** (chưa kiểm chứng cục bộ được): ước lượng byte/giây. Nếu sai
nhiều, khoảng tải lại có thể lệch khỏi vị trí cần — log `⬇️ Tải lại … cho vùng quanh Xs` ở
Console là chỗ kiểm tra đầu tiên.

---

## 8. Giới hạn đã biết

### 7.9. Sự cố TRÀN VRAM: TTS synth 30 s làm mất cả phụ đề

**Hiện tượng** (log thật 08:35): GPU 15,6/16,0 GB, utilization 100 %, **không có phụ đề và
không có TTS**.

**Nguyên nhân gốc — TTS chiếm GPU và làm ASR đói:**

```text
08:36:05.670 [ASR_COMMIT] (infer=172.7ms)      ← bình thường
08:36:06.7   bắt đầu tổng hợp TTS đầu tiên
08:36:15.645 [ASR_COMMIT] (infer=3534.5ms)     ← chậm 20×
08:36:34.331 [ASR_COMMIT] (infer=2571.3ms)
08:36:37.263 [TTS] Synth 30547ms | audio 4.34s | RTF 7.04   ← 30 s cho 4,3 s audio!
08:36:37.274 [TTS] Synth 32599ms | audio 2.65s | RTF 12.30
```

Ba model (ASR + dịch + OmniVoice) cùng nằm trong 16 GB. Khi VRAM tới hạn, một lần diffusion
TTS đội từ ~0,8 s lên **30 s**; trong suốt 30 s đó ASR không chạy được ⇒ không có phụ đề mới
⇒ `ready_ahead` âm ⇒ màn hình trống, và hàng đợi TTS càng lúc càng dài.

**Khắc phục — TTS trở thành BEST-EFFORT, không bao giờ được làm hỏng đường phụ đề:**

| Chốt | Hành vi |
| :--- | :--- |
| `tts_min_free_vram_mb` (1500) | Trước mỗi câu: đo VRAM trống; nếu dưới ngưỡng thì `torch.cuda.empty_cache()` rồi đo lại; **vẫn thấp ⇒ BỎ câu lồng tiếng** + WARNING. Thà mất tiếng lồng còn hơn tràn VRAM làm mất cả phụ đề. |
| `tts_asr_yield_max_sec` (3 s) | Trước mỗi câu: chờ (có trần) cho ASR rảnh (`engine.has_pending_work()`) — phụ đề là đường chính. |
| `tts_max_queue` (6) | Hàng đợi vượt trần ⇒ **bỏ câu CŨ NHẤT** (đã trôi xa vị trí phát), không để phình rồi đốt GPU hàng phút. |
| `tts_max_chars` (220) | Cắt bớt văn bản ở ranh giới từ — câu quá dài làm diffusion tốn VRAM/thời gian vô ích. |
| `tts_slow_warn_ms` (4000) | Cảnh báo rõ khi một lần tổng hợp vượt 4 s (dấu hiệu GPU quá tải). |

**Chốt phía client — không bao giờ để màn hình trống:** đang phát mà `ready_ahead < 0,3 s`
trong 3 nhịp liên tiếp (và cách lần phát trước ≥ 8 s) thì **tạm dừng video** để pipeline đuổi
kịp, rồi tự phát lại khi đã có phụ đề (`pauseForBuffering("underrun")`).

**Khuyến nghị vận hành** (16 GB VRAM): chạy `nvidia-smi` để xem model nào chiếm bao nhiêu.
Nếu vẫn căng, dùng model dịch nhẹ hơn (`tencent-1.8b`) hoặc tắt TTS khi xem phim dài.

**Kiểm chứng**: `test_tts_queue_is_trimmed_and_stale_jobs_skipped`,
`test_clip_dubbing_text_cuts_at_word_boundary`, `test_free_vram_probe_never_raises`.


### 7.8. Canh mốc: phụ đề/TTS hiện SỚM hơn tiếng nói

**Triệu chứng**: phụ đề hiện trước tiếng nói trong video một chút, và lồng tiếng cũng đọc
trước người nói.

**Nguyên nhân**: mỗi VAD engine **lùi mép đầu đoạn nói** để lấy ngữ cảnh, và mốc đó đi thẳng
vào `start_pts`:

| Engine | Tham số lùi mép | Độ lệch điển hình |
| :--- | :--- | :--- |
| FireRed | `pad_start_frame` (5 frame × 10 ms) | ~50 ms |
| Silero | `speech_pad_ms` (30) | ~30 ms |
| **FSMN** | `lookback_time_start_point` (200) | **~200 ms** |

Với FSMN (đang dùng trong log thật), phụ đề/TTS xuất hiện sớm ~200 ms — đúng như quan sát.

**Khắc phục (2 lớp):**

1. **Bù tự động theo engine**: mỗi engine VAD nay công bố `start_pad_ms`
   (`backend/vad/engines/{firered,silero,fsmn}.py`), `VADProcessor.start_pad_ms` đọc lại, và
   `LookaheadSessionState._handle_asr_message` cộng bù vào `start_pts` trước khi gửi phụ đề.
   Vì TTS lấy mốc từ chính phụ đề nên **lồng tiếng cũng khớp theo**, không cần sửa gì thêm.
   Mốc bắt đầu được kẹp để không vượt quá `end_pts - 0,15 s` với câu cực ngắn.
2. **Tinh chỉnh thủ công**: popup có thêm thanh **“Đồng bộ phụ đề / lồng tiếng”** (−600…+600 ms,
   bước 50). Giá trị này (`lookaheadSyncOffsetMs`) đi kèm `lookahead_init`/`set_config`; dấu
   dương = hiện muộn hơn (dùng khi vẫn thấy sớm), dấu âm = hiện sớm hơn.

Log xác nhận khi khởi động phiên:
`Canh mốc phụ đề/TTS: VAD 'fsmn-vad' báo mép nói sớm 200ms — sẽ cộng bù (offset người dùng +0ms).`

**Kiểm chứng**: `test_asr_start_pts_is_shifted_by_vad_start_pad`,
`test_user_sync_offset_is_applied_and_clamped`, `test_vad_engines_declare_start_pad`.


### 7.7. Lần thử thật thứ ba (2026-10-01 08:12) — TTS vẫn không kêu

**Chẩn đoán từ log hai phía:**

* **Server tính ĐÚNG**: probe `scratch_probe_ready.py` mô phỏng đúng trạng thái lúc 08:12:04
  (`current=20`, `decoded_end=200`, `feed=41`, `ready_until=23.89`) cho
  `prebuffer_ready=True, ready_ahead=3.89s, fed_ahead=21s`. Nay đã thêm log
  `Lookahead prebuffer_ready=…` mỗi khi giá trị này ĐỔI.
* **Client không phản ứng**: console KHÔNG có bất kỳ dòng `[BS TTS-LA]` nào và video chỉ phát
  nhờ chốt an toàn 8 s ⇒ hai đường đều không chạy: `lookahead_status` không kích hoạt resume,
  và audio TTS không tới được scheduler.

**Khắc phục (loại bỏ phụ thuộc vào hành vi trình duyệt):**

1. **Audio TTS nay gửi kèm `audio_b64` trong bản tin JSON `lookahead_tts`** — đúng đường mà
   phụ đề đang đi (đã chứng minh chạy được), **không** còn phụ thuộc `ws.binaryType`. Khung
   nhị phân vẫn được gửi song song và client vẫn hiểu (chống trùng theo `start_pts`).
2. Hàm nhận khung nhị phân chấp nhận thêm **Blob** và mọi đối tượng có `byteLength` (Firefox/
   Chrome có thể trả Blob nếu `binaryType` không được áp như mong đợi — trước đây rơi vào
   nhánh im lặng).
3. **Chốt an toàn phía client**: khi đang tạm dừng, chỉ cần `ready_ahead >= 2.5 s` là phát
   video (`resumePlayback("status_ready")`) — không chờ đủ `lead_time` vì backend xử lý nhanh
   hơn thời gian thực ~20×.
4. **Log chẩn đoán đổi sang `console.log`** (trước đây `console.info` có thể bị bộ lọc của
   DevTools ẩn): log thiết lập hiệu dụng khi khởi động Pipeline B (`ttsEnabled`, `vadEngine`,
   `duckingLevel`…), trạng thái scheduler + `AudioContext`, lần nhận TTS đầu tiên, mỗi lần
   phát/bỏ câu, và lý do bỏ qua `lookahead_status` của seek cũ.

**Kiểm chứng**: test JS `nhận audio lồng tiếng qua JSON base64 (không phụ thuộc binaryType)`
+ toàn bộ 43 test JS và backend suite đều PASS.

* **TTS (lồng tiếng)** đã chạy được ở Pipeline B (§7.5) với cơ chế chống trễ 3 tầng. Điểm cần
  lưu ý: nếu câu dịch quá dài so với cửa sổ phụ đề, phần vượt quá trần nén (1,45× backend +
  1,35× client) sẽ bị **cắt đuôi** — đây là đánh đổi có chủ ý để không bao giờ lệch tiếng.
* ASR native là **singleton** (một stream in-flight cho mỗi model): không nên chạy Pipeline B
  song song với một phiên `/ws` realtime khác. Backend ghi WARNING khi phát hiện.


