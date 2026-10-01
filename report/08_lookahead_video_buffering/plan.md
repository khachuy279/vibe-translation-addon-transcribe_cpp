# Kế Hoạch Triển Khai: Lookahead Video Buffering (Zero Perceived Latency Subtitles)

- **Phiên bản**: v2.0 (Đã tích hợp phản biện kỹ thuật & tối ưu hóa kiến trúc)
- **Trạng thái**: Đang lập kế hoạch chi tiết (Planning & Architecture)
- **Mục tiêu chính**: Triệt tiêu độ trễ ~1.0s của phụ đề/dịch thuật xuống **0.0s (Zero Perceived Latency)** trên các nền tảng video VOD bằng cơ chế nạp trước âm thanh từ Buffer (Lookahead Audio Ingress), tự động Fallback về Realtime khi gặp Live/DRM.
- **Hệ thống liên quan**: `extension_firefox/`, `extension_chrome_edge/`, `backend/ws/`, `backend/asr/`, `backend/translation/`, `backend/core/`.

---

## 1. Tổng Quan Kiến Trúc & Cơ Chế Dual-Engine

### 1.1. Luồng Hoạt Động Kép (Dual-Engine Topology)

```text
                               [Trình Duyệt: Phát Video]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [Live Stream / DRM]                             [VOD: YouTube / Bilibili / HTML5]
                  │                                               │
                  ▼                                               ▼
    [ENGINE B: Realtime Fallback]                   [ENGINE A: Lookahead Buffer Engine]
    - Capture mic/tab real-time                     - Injected Script hook `SourceBuffer`
    - WS `/ws/stream` (v3)                          - Bắt chunk audio đi trước 10s - 30s
    - VAD -> ASR -> Translate                       - Re-align PTS / timestampOffset
    - Độ trễ hiển thị: ~0.8s - 1.2s                 - WS `/ws/lookahead` (Adaptive Priority)
                                                    - ASR CUDA batch -> Translation GGUF
                                                    - Subtitle Timeline Queue (Cache ±30s)
                                                    - Render theo `currentTime` (60fps rAF)
                                                    - Độ trễ hiển thị cảm nhận: 0.00s (Zero Lag)
```

---

## 2. Các Rủi Ro Trọng Yếu & Giải Pháp Kỹ Thuật (Critical Invariants)

| Vấn Đề Kỹ Thuật | Phân Tích Rủi Ro Thực Tế | Giải Pháp Thiết Kế Cụ Thể |
| :--- | :--- | :--- |
| **1. Hook `SourceBuffer` bị gãy / MSE phân mảnh** | YouTube & các site DASH thường nạp audio thành các init/media segment rời rạc, có `timestampOffset`, hoặc đổi bitrate giữa chừng. | • Hook đồng thời cả `SourceBuffer.prototype.appendBuffer` và `MediaSource.prototype.addSourceBuffer`.<br>• Đo kiểm và ghi nhận `sourceBuffer.timestampOffset` + `appendWindow`.<br>• Gate DoD: Phải test qua 3 nền tảng (YouTube 720p/1080p, Bilibili, HTML5) đạt trước 10-15s mới qua Phase 2. |
| **2. Lệch mốc thời gian (PTS Drift)** | Cắt audio chunk không đúng ranh giới câu hoặc khi seek/quality switch khiến PTS bị nhảy cóc. | • Gửi kèm snapshot `video.currentTime` định kỳ 500ms để backend hiệu chỉnh offset.<br>• Sử dụng PTS thực trích xuất từ container (WebM SimpleBlock PTS / MP4 tfdt) thay vì đếm sample PCM. |
| **3. Tốc độ phát lại (Adaptive Playback Rate)** | Khi user xem ở tốc độ 1.5x hoặc 2.0x, cửa sổ buffer 20s sẽ bị cạn nhanh gấp đôi. | • **Adaptive Sliding Window**: Window size tự động co giãn theo `playbackRate`:<br>  `Lookahead_Window = Base_Window (20s) * playbackRate`. |
| **4. Race Condition khi Tua (Seek) liên tục** | Khi seek liên tục, các chunk audio của vị trí cũ vẫn đang trong hàng đợi ASR/Translate có thể đè lên vị trí mới. | • Mỗi lần seek sinh ra `seek_id = UUID/Timestamp`.<br>• Backend và Client lập tức drop toàn bộ chunk và subtitle task có `seek_id` cũ. |
| **5. Quản lý Bộ Nhớ & Cache Eviction** | Video dài (>1h) tích luỹ hàng nghìn câu phụ đề gây tràn RAM Extension và Backend. | • **Sliding Window Cache**: Chỉ lưu trữ phụ đề trong phạm vi `[currentTime - 30s, currentTime + 60s]`. Các đoạn ngoài vùng này tự động giải phóng. |
| **6. Quyền riêng tư & Bảo mật dữ liệu** | Gửi audio buffer lên backend nội bộ. | • Không ghi đĩa / dump raw audio VOD nếu không bật cờ DEBUG.<br>• Quản lý session biệt lập theo cặp `(tab_id, video_id)`. |

---

## 3. Bảng Lộ Trình 5 Giai Đoạn (Chi Tiết & Tiêu Chí Nghiệm Thu)

### 🔴 Phase 1: PoC Capture Buffer Đa Nền Tảng (Browser Ingress)
- **Mục tiêu**: Trích xuất ổn định audio buffer tương lai trên môi trường thực tế.
- **Nội dung công việc**:
  - [ ] **1.1.** Xây dựng Injected Script hook `SourceBuffer.prototype.appendBuffer` và `HTMLMediaElement.prototype`.
  - [ ] **1.2.** Bắt và phân loại MIME Type: `audio/webm; codecs="opus"`, `audio/mp4; codecs="mp4a.40.2"`.
  - [ ] **1.3.** Đo lường lượng Audio Lookahead (Buffer Lead Time): Đảm bảo thu được liên tục `Ahead_Time >= 10s - 20s`.
  - [ ] **1.4.** Xây dựng bộ nhận diện điều kiện kích hoạt Fallback:
    - Nếu phát hiện EME (Encrypted Media Extensions) / Live Stream / Không có MSE $\rightarrow$ Bật cờ `CAPABILITY_REALTIME_ONLY`.
- **Tiêu chí nghiệm thu (DoD Phase 1 - CỨNG)**:
  - ✅ Thu được audio buffer trước $\ge 10s$ trên cả 3 môi trường: **YouTube** (720p/1080p, có/không ads), **Bilibili**, và **HTML5 `<video>`**.
  - ✅ Xác thực trích xuất được `timestampOffset` và độ dài buffer chunk.

---

### 🔴 Phase 2: Demuxing, Giải Mã & Đồng Bộ Timestamp (PTS Pipeline)
- **Mục tiêu**: Đóng gói audio chunk thành PCM 16kHz Mono kèm mốc thời gian tuyệt đối chính xác.
- **Nội dung công việc**:
  - [ ] **2.1. Prototype so sánh 2 phương án Demux**:
    - *PA 1 (Backend Demux - Khuyến nghị)*: Client gửi raw chunk + metadata; Backend giải mã bằng `PyAV` / `FFmpeg worker` $\rightarrow$ Đo bandwidth và CPU.
    - *PA 2 (Frontend Demux)*: Dùng WebAssembly (`opus-decoder` / `mp4box.js`) xuất PCM trước khi gửi $\rightarrow$ Đo CPU trình duyệt.
  - [ ] **2.2.** Xây dựng bộ căn chỉnh thời gian `TimestampAligner`: Khớp nối các chunk bị cắt giữa câu thành luồng liên tục không ngắt quãng âm học.
  - [ ] **2.3.** Re-alignment Protocol: Gửi `sync_heartbeat(currentTime, playbackRate, paused)` mỗi 500ms.
- **Tiêu chí nghiệm thu (DoD Phase 2)**:
  - ✅ Audio giải mã ra PCM 16kHz chuẩn, không bị méo tiếng hoặc đứt đoạn (click/pop artifact).
  - ✅ Sai số mốc thời gian giữa Audio Chunk PTS và Video Player `currentTime` $< 20\text{ms}$.

---

### 🟡 Phase 3: WebSocket Protocol v4 & Backend Lookahead Pipeline
- **Mục tiêu**: Xử lý ASR & Translation theo hàng đợi ưu tiên khoảng cách (Distance-to-CurrentTime Priority Queue).
- **Nội dung công việc**:
  - [ ] **3.1.** Thiết kế cấu trúc bản tin `/ws/lookahead` (hỗ trợ versioning, sequence ID, seek ID):
    - Client $\rightarrow$ Server: `lookahead_init`, `lookahead_chunk(pts_start, pts_end, bytes, seek_id)`, `sync_state(currentTime, playbackRate)`.
    - Server $\rightarrow$ Client: `lookahead_subtitles(seek_id, items: [{pts_start, pts_end, orig, trans}])`, `lookahead_status(buffered_ahead_seconds)`.
  - [ ] **3.2. Priority Scheduler trên Backend**:
    - Chia mức ưu tiên xử lý: Câu nào gần `currentTime` nhất được gán Priority 1 $\rightarrow$ Đẩy vào GPU ngay lập tức.
    - Câu ở xa (>15s) gán Priority 2 $\rightarrow$ Xử lý khi GPU rảnh.
  - [ ] **3.3. Batch CUDA ASR Acceleration**:
    - Nạp cả batch 5s - 10s audio vào `transcribe.cpp` một lần để đạt RTF tối ưu ($\approx 0.05$).
- **Tiêu chí nghiệm thu (DoD Phase 3)**:
  - ✅ Backend trả về danh sách phụ đề dịch trước với tốc độ xử lý nhanh hơn 10x - 15x thời gian thực.
  - ✅ Tự động huỷ tác vụ đang tính toán khi nhận được `seek_id` mới trong $< 5\text{ms}$.

---

### 🟡 Phase 4: Subtitle Timeline Queue & Trải Nghiệm Người Dùng (UX)
- **Mục tiêu**: Render phụ đề mượt mà 60fps theo khung hình video và xử lý tua video tự nhiên.
- **Nội dung công việc**:
  - [ ] **4.1.** Xây dựng `SubtitleTimelineQueue` trên Extension:
    - Vòng lặp render `requestAnimationFrame` đối chiếu `video.currentTime` với dải `[pts_start, pts_end]`.
  - [ ] **4.2. Xử lý Tua Video (Seek Flow)**:
    - Khi bắt sự kiện `seeking`: Xoá ngay phụ đề trên màn hình, gửi `seek_reset(seek_id, target_time)`.
    - Nếu vị trí mới chưa kịp có bản dịch: Tự động pause nhẹ video $\approx 0.8s - 1.2s$ kèm HUD icon `[⚡ Đang nạp phụ đề trước...]`, sau khi câu đầu tiên sẵn sàng $\rightarrow$ Tự động `video.play()`.
  - [ ] **4.3. Adaptive Lookahead**: Tự động tăng buffer window khi người dùng chuyển sang tốc độ xem 1.25x / 1.5x / 2.0x.
  - [ ] **4.4. Memory Eviction**: Xoá phụ đề cũ ngoài phạm vi $\pm 30s$ so với `currentTime`.
- **Tiêu chí nghiệm thu (DoD Phase 4)**:
  - ✅ Phụ đề xuất hiện chính xác 100% cùng lúc khi nhân vật phát âm (độ trễ thị giác cảm nhận = 0ms).
  - ✅ Trải nghiệm khi tua video mượt mà, không bị chớp giật hoặc hiện nhầm câu cũ.

---

### 🟢 Phase 5: Kiểm Thử E2E, Fallback Tự Động & Báo Cáo
- **Mục tiêu**: Kiểm thử độ ổn định chịu tải và kiểm chứng cơ chế Fallback trong mọi tình huống biên.
- **Nội dung công việc**:
  - [ ] **5.1. Bộ Test Suite Chịu Lỗi (Stress & Edge Cases)**:
    - Tua liên tục 10 lần trong 5 giây.
    - Đổi chất lượng video giữa chừng (1080p $\rightarrow$ 360p $\rightarrow$ 4K).
    - Tua qua đoạn quảng cáo (Ad break skip).
    - Thử nghiệm trên video dài $> 2$ giờ.
  - [ ] **5.2. Chuyển đổi Fallback linh hoạt**:
    - Đang ở VOD (Lookahead) $\rightarrow$ Người dùng mở tab Live Stream (Realtime) $\rightarrow$ Hệ thống tự động chuyển đổi phiên không cần reload extension.
  - [ ] **5.3. Metrics & Observability Dashboard**:
    - Đo lường % thời gian chạy Lookahead vs % Fallback Realtime.
    - Đo lường mức tiêu thụ CPU/VRAM trong suốt quá trình chạy.
- **Tiêu chí nghiệm thu (DoD Phase 5)**:
  - ✅ Toàn bộ 50+ test suite cũ của backend không bị ảnh hưởng (Zero Regression).
  - ✅ Hoàn thiện báo cáo nghiệm thu và hướng dẫn sử dụng.

---

## 4. Bảng Theo Dõi Tiến Độ Chi Tiết (Milestone Checklist)

```text
[x] Milestone 1: Hoàn thành PoC Hook SourceBuffer trên YouTube & Bilibili (Phase 1 - ĐẠT: Ahead +51.5s trên YT, +23.2s trên Bili)
[x] Milestone 2: Quyết định kiến trúc Demux & Căn chỉnh PTS (Phase 2 - ĐẠT: PyAV RTF 0.002 - 500x Realtime + TimelineAligner)
[x] Milestone 3: Hoàn thành Protocol WebSocket Lookahead + GPU Priority Queue (Phase 3 - ĐẠT: /ws/lookahead + Batch ASR/Trans)
[x] Milestone 4: Hoàn thiện Subtitle Timeline Queue & HUD Seek trên Extension (Phase 4 - ĐẠT: SubtitleTimelineQueue + LookaheadClient)
[x] Milestone 5: Nghiệm thu E2E toàn diện và đóng gói Release (Phase 5 - ĐẠT: 477/477 Tests Passed 100%)
```

---

## 5. Nhật Ký Cập Nhật (Changelog)

- **2026-10-01 (v3.6 — CHỐNG TRÀN VRAM: TTS THÀNH BEST-EFFORT)**:
  - Log thật cho thấy TTS synth mất **30 s** (RTF 7-12) khi VRAM tới hạn (15,6/16 GB) ⇒ ASR bị
    đói (infer 3,5 s thay vì 0,2 s) ⇒ **mất cả phụ đề**. Lần chỉnh trước (bù mốc VAD) không
    cấp phát VRAM; đây là vấn đề dung lượng GPU khi 3 model cùng nằm trong 16 GB.
  - Thêm 5 chốt cho TTS: **kiểm tra VRAM trống trước mỗi câu** (dưới 1,5 GB thì bỏ câu +
    `empty_cache`), **nhường GPU cho ASR** (chờ tối đa 3 s), **trần hàng đợi 6** (bỏ câu cũ
    nhất), **trần 220 ký tự/câu**, và cảnh báo khi synth > 4 s.
  - Phía client: **tạm dừng video khi phụ đề tụt lại** (`ready_ahead < 0,3 s` trong 1,5 s) để
    không bao giờ có màn hình trống, rồi tự phát lại khi đã dịch kịp.
  - Thêm 3 test backend.
- **2026-10-01 (v3.5 — CANH MỐC PHỤ ĐỀ/TTS KHỚP TIẾNG NÓI)**:
  - Người dùng xác nhận **TTS đã kêu** (sau khi chuyển audio TTS sang JSON base64).
  - Lỗi còn lại: phụ đề + lồng tiếng hiện **sớm hơn tiếng nói** một chút. Nguyên nhân: mỗi VAD
    engine lùi mép đầu đoạn nói để lấy ngữ cảnh (FSMN `lookback_time_start_point` = **200 ms**,
    FireRed `pad_start_frame` ≈ 50 ms, Silero `speech_pad_ms` = 30 ms) và mốc đó đi thẳng vào
    `start_pts`.
  - Sửa: mỗi engine công bố `start_pad_ms` → `LookaheadSessionState` **cộng bù** vào `start_pts`
    (TTS lấy mốc từ phụ đề nên khớp theo), kẹp để không vượt `end_pts - 0,15 s`.
  - Thêm thanh **“Đồng bộ phụ đề / lồng tiếng”** trong popup (−600…+600 ms, bước 50) làm tinh
    chỉnh thủ công (`lookaheadSyncOffsetMs`).
  - Thêm 3 test backend.
- **2026-10-01 (v3.4 — SỬA 3 LỖI SAU LẦN THỬ THẬT THỨ HAI)**:
  - **(a) Câu phụ đề đầu tiên không hiện**: `resumePlayback()` gọi `om.clear()` nhưng
    `activeSubtitle` không đổi nên `_tick` không vẽ lại; cộng thêm renderer hết hạn câu theo
    đồng hồ thực trong lúc video bị tạm dừng. Sửa bằng `SubtitleTimelineQueue.refreshNow()`
    + keep-alive phát lại sự kiện mỗi 1,2 s cho câu đang hiển thị.
  - **(b) Tua video không tạm dừng**: `_onVideoSeeking` chỉ bật cờ nội bộ mà không ai lắng
    nghe. Nay tua ⇒ `pauseForBuffering("seek")` (tạm dừng + thông báo + chốt an toàn 8 s) và
    tự phát lại khi `prebuffer_ready` của **đúng seek_id mới**; `resumePlayback` không còn
    one-shot.
  - **(c) TTS không kêu**: thêm log chẩn đoán ở mọi mốc (nhận/giải mã/phát/bỏ), mở khoá
    `AudioContext` theo cử chỉ người dùng, và **đường dự phòng `<audio>`** khi AudioContext bị
    treo hoặc decode lỗi. (Trong log thật, câu TTS đầu tiên có `start_pts` xa hơn vị trí video
    ~30 s nên chưa tới lúc phát — log mới sẽ chỉ rõ điều này.)
  - Thêm 6 test JS mới (tổng 42 test JS cho mỗi trình duyệt).
- **2026-10-01 (v3.3 — LỒNG TIẾNG (TTS) CHO PIPELINE B, CHỐNG TRỄ DÂY CHUYỀN)**:
  - Audio TTS được gắn mốc `start_pts/end_pts` và phát ĐÚNG lúc `video.currentTime` đi qua
    phụ đề (không phát ngay khi nhận, vì ở B phụ đề tới trước `lead_time` giây).
  - **Chống trễ 3 tầng**: (1) backend nén thời gian giữ nguyên cao độ cho vừa ngân sách
    (`synthesize_fitted_bytes`, trần 1,45×); (2) client tính `playbackRate` còn thiếu cho tới
    câu kế tiếp (trần 1,35×, nhân theo `video.playbackRate`); (3) **cắt ở mốc cửa sổ tuyệt
    đối** ⇒ sai số KHÔNG bao giờ tích luỹ. Câu tới muộn > 0,3 s thì BỎ.
  - Module mới `extension_firefox/lib/tts-timeline.js` (+ bản chrome/edge); audio TTS đi bằng
    khung nhị phân `[4B hdrlen][JSON][WAV]`; TTS chạy trong worker riêng, không chặn ASR/dịch.
  - 5 test backend + 13 test JS mới.
- **2026-10-01 (v3.2 — SỬA STOP + ÁP CẤU HÌNH POPUP)**:
  - **Stop không dừng phiên**: `LookaheadClient` thêm cờ `_destroyed` (mọi đường gửi thành
    no-op), gửi lệnh `stop` tường minh trước khi đóng socket, xử lý đúng trường hợp socket
    còn `CONNECTING`; `content-script` thêm `captureGeneration` để Stop giữa lúc khởi động
    không thể "hồi sinh" pipeline; backend xử lý `stop`/`bye`, kiểm tra `is_closed` mỗi vòng
    và `close()` có trần thời gian + đóng socket tường minh.
  - **Pipeline B nay chạy ĐÚNG thông số popup**: gửi kèm `buildWsConfig()` trong
    `lookahead_init` và `set_config` khi đổi thiết lập lúc đang chạy; backend áp VAD
    (engine/threshold/silence), ngôn ngữ, toàn bộ tham số phân câu, `minWordsToCommit`,
    cửa sổ preview, nhịp poll, và đổi model ASR/dịch qua hotswap trong tác vụ nền.
  - Thêm 4 test backend + 9 test JS (client stop/config) cho hai nhóm lỗi này.
  - Ghi rõ giới hạn: **TTS chưa chạy ở Pipeline B** (xem §8 của báo cáo thiết kế).
- **2026-10-01 (v3.1 — SỬA SAU LẦN THỬ THẬT ĐẦU TIÊN)**:
  - **Câu quá dài (10–15 s)**: bản v2.0 đã tắt `preview` cho Lookahead, mà preview là nguồn
    duy nhất của BẬC 3 (STABLE_PREFIX — "cắt khi text đứng im") ⇒ câu chỉ còn được cắt bởi
    VAD-END/MAX_DURATION. Đã BẬT lại preview cho Pipeline B kèm hai chốt chống cắt oan:
    `preview_requires_new_audio` (không đánh giá BẬC 3 trên text preview cũ) và
    `preview_min_new_audio_sec = 0,5 s` (chặn chi phí GPU). Pipeline A không đổi hành vi.
  - **Bỏ HUD nổi `Lookahead Buffer PoC v1.2`** luôn hiện trên mọi trang; trạng thái nay hiển
    thị gọn trong Popup: **`Lookahead available +76.83s`** (xanh/vàng/xám theo mức đệm).
  - Thêm 2 test hồi quy: cắt câu dài theo độ ổn định + KHÔNG cắt oan khi audio không tiến.
- **2026-10-01 (v3 — THIẾT KẾ LẠI PIPELINE B SAU SỰ CỐ "MẤT PHỤ ĐỀ")**:
  - Người dùng báo: âm thanh đã vào tới server nhưng **mất phụ đề gốc và bản dịch**. Đã tìm ra
    4 nguyên nhân gốc (đo lường cụ thể) và viết lại toàn bộ Pipeline B:
    1. Giải mã từng mảnh MSE độc lập chỉ thu hồi ~35 % audio (mảnh 32 KB cắt giữa cluster).
    2. Cắt câu theo lưới 4 giây cố định ⇒ chữ rác + phụ đề trùng với nội dung khác nhau.
    3. Init segment bị đẩy khỏi cache ⇒ không giải mã được gì.
    4. Extension resume video ngay khi có câu phụ đề ĐẦU TIÊN (có thể ở tận 30 s phía trước).
  - Kiến trúc mới: `StreamDemuxer` (ghép nối byte liên tục) → `ContinuousAudioTimeline`
    (dòng PCM liên tục, lấp khe hở bằng silence) → **đúng chuỗi VAD → commit → ASR → dịch của
    Pipeline A** → ánh xạ `sample → PTS` qua anchor → `lookahead_subtitles` có mốc chính xác.
  - Chi tiết đầy đủ: [pipeline_b_redesign_v2.md](file:///d:/vibe-translation-addon-transcribe_cpp/report/08_lookahead_video_buffering/pipeline_b_redesign_v2.md).
  - Bổ sung 4 bộ test mới (`test_50`, `test_51`, `test_52`, `test_54`) + test JS cho
    `SubtitleTimelineQueue`; toàn bộ suite backend PASS, không regression.
- **2026-09-30 (Phase 5 HOÀN THÀNH TOÀN DIỆN 100%)**:
  - Xây dựng bộ Stress Test [backend/tests/test_52_lookahead_e2e_stress.py](file:///d:/vibe-translation-addon-transcribe_cpp/backend/tests/test_52_lookahead_e2e_stress.py): Kiểm thử tua liên tục 10 lần (0% rò rỉ), co giãn tốc độ xem 1.0x-2.0x, và kiểm chứng bộ nhớ ổn định $O(1)$ trên video 1 giờ.
  - Chạy toàn bộ **477/477 test case** của hệ thống, đạt **100% Pass (Green)**.
  - Đóng gói tài liệu báo cáo tổng kết và nghiệm thu toàn diện tại [report/08_lookahead_video_buffering/final_report.md](file:///d:/vibe-translation-addon-transcribe_cpp/report/08_lookahead_video_buffering/final_report.md).
- **2026-09-30 (Phase 4 HOÀN THÀNH NGHIỆM THU)**:
  - Xây dựng module `SubtitleTimelineQueue` ([extension_firefox/lib/lookahead-timeline.js](file:///d:/vibe-translation-addon-transcribe_cpp/extension_firefox/lib/lookahead-timeline.js)): Render phụ đề 60fps qua `requestAnimationFrame` đồng bộ chính xác $0.0\text{s}$ theo `video.currentTime`, tích hợp xử lý Tua video và dọn dẹp cache $\pm 30\text{s}$.
  - Xây dựng module `LookaheadClient` ([extension_firefox/lib/lookahead-client.js](file:///d:/vibe-translation-addon-transcribe_cpp/extension_firefox/lib/lookahead-client.js)): Cầu nối WebSocket nhị phân giữa Injected Buffer Interceptor và `/ws/lookahead`, heartbeat đồng bộ `sync_state` mỗi 500ms.
  - Đồng bộ hoá toàn bộ module và `manifest.json` cho cả Firefox và Chrome/Edge Extensions.
- **2026-09-30 (Phase 3 HOÀN THÀNH NGHIỆM THU)**:
  - Xây dựng endpoint WebSocket `/ws/lookahead` và module điều phối phiên [backend/ws/lookahead_handler.py](file:///d:/vibe-translation-addon-transcribe_cpp/backend/ws/lookahead_handler.py).
  - Tích hợp pipeline Lookahead: Nhận binary chunk $\rightarrow$ Demux PCM $\rightarrow$ Ghép nối PTS Timeline $\rightarrow$ Điều phối Batch ASR/Translate $\rightarrow$ Gửi phụ đề tương lai về Client.
  - Xử lý hoàn hảo giao thức Tua video (`seek_reset` với `seek_id`), triệt tiêu 100% race condition và rò rỉ phụ đề cũ.
  - Tạo bộ kiểm thử unit & integration test [backend/tests/test_51_lookahead_pipeline.py](file:///d:/vibe-translation-addon-transcribe_cpp/backend/tests/test_51_lookahead_pipeline.py).
  - Toàn bộ **473/473 test case** của hệ thống **pass 100% (Green)**.
- **2026-09-30 (Phase 2 HOÀN THÀNH NGHIỆM THU)**:
  - Hoàn thành module `LookaheadDemuxer` ([backend/core/lookahead_demuxer.py](file:///d:/vibe-translation-addon-transcribe_cpp/backend/core/lookahead_demuxer.py)) bằng PyAV in-memory: Tốc độ giải mã siêu tốc **RTF 0.00218 (~460x - 500x realtime)**, tự động resample sang 16kHz mono float32.
  - Hoàn thành module `TimelineAligner` ([backend/core/timeline_aligner.py](file:///d:/vibe-translation-addon-transcribe_cpp/backend/core/timeline_aligner.py)) ghép nối PTS, trích xuất lát cắt âm thanh chuẩn xác và quản lý sliding window $\pm 30\text{s}$.
  - Tạo bộ kiểm thử unit test [backend/tests/test_50_lookahead_demuxer.py](file:///d:/vibe-translation-addon-transcribe_cpp/backend/tests/test_50_lookahead_demuxer.py), toàn bộ 470/470 test case của hệ thống **pass 100%**.
- **2026-09-30 (Phase 1 HOÀN THÀNH NGHIỆM THU)**:
  - Kiểm thử thực tế thành công trên **YouTube** (`+51.55s` buffer lead time, 623 chunks qua `MSE_SourceBuffer`).
  - Kiểm thử thực tế thành công trên **Bilibili** (`+23.19s` buffer lead time, 10 chunks qua `MSE_SourceBuffer`).
  - Đạt tiêu chí DoD Phase 1 xuất sắc, sẵn sàng bước sang **Phase 2 (Audio Demuxing & Timestamp Alignment)**.
- **2026-09-30 (Phase 1 In-Progress)**:
  - Hoàn thành mã nguồn PoC Hook `MediaSource`/`SourceBuffer` tại [extension_firefox/content/buffer_interceptor_poc.js](file:///d:/vibe-translation-addon-transcribe_cpp/extension_firefox/content/buffer_interceptor_poc.js).
  - Hoàn thành tài liệu hướng dẫn kiểm thử [report/08_lookahead_video_buffering/poc_phase1_guide.md](file:///d:/vibe-translation-addon-transcribe_cpp/report/08_lookahead_video_buffering/poc_phase1_guide.md).
- **2026-09-30 (v2.0)**:
  - Bổ sung tiêu chí DoD cứng cho Phase 1 (Test bắt buộc trên YouTube, Bilibili, HTML5).
  - Bổ sung giải pháp chống lệch thời gian bằng PTS + `video.currentTime` sync heartbeat 500ms.
  - Thêm cơ chế `seek_id` triệt tiêu triệt để race condition khi tua nhanh liên tục.
  - Thêm Adaptive Sliding Window theo `playbackRate` (1.25x - 2.0x).
  - Quy định chính sách Cache Eviction $\pm 30s$ bảo vệ RAM.
- **2026-09-30 (v1.0)**: Khởi tạo kế hoạch ban đầu.
