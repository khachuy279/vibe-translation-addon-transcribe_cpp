# Phase 6 — Gỡ Pipeline B v2 (streaming) & chốt OFFLINE_BATCH làm mặc định

- **Ngày**: 2026-10-02 (tiếp ngay sau Phase 5 / §7–§9 của `phase5_chan_doan_ngat_cau.md`)
- **Yêu cầu người dùng**:
  1. Xoá Pipeline B v2 (streaming) và những thứ liên quan (không còn dùng).
  2. Đặt Pipeline B **OFFLINE_BATCH** (Qwen3-ASR + Forced Aligner) làm **mặc định** ở server + extension.
  3. Pipeline A được bật khi **Lookahead không khả dụng** (fallback) hoặc người dùng **tắt Lookahead
     Video Buffering** trong popup.
  4. Popup: khi Lookahead BẬT ⇒ **vô hiệu** các tuỳ chỉnh chỉ dành cho Pipeline A (VAD Silence, VAD
     Threshold, Segmentation…) và **mở lại** khi Lookahead TẮT.
  5. README: giải thích ngắn gọn tính năng của Pipeline B OFFLINE_BATCH.

---

## 1. Đã XOÁ (Pipeline B v2 "streaming giả lập")

### `backend/ws/lookahead_handler.py`

| Xoá | Vì sao |
|---|---|
| `processing_mode` (thuộc tính + nhánh rẽ ở `start_tasks`) | Chỉ còn một tuyến duy nhất |
| `_ingest_loop()`, `ingest_once()`, `_feed_block()`, `_next_feed_pts()`, `_feed_target_pts()` | Vòng nạp audio 0,5 s qua VAD — đặc trưng của v2 |
| `_asr_loop()`, `_handle_asr_message()` | Tiêu thụ `stream_tokens()` + CommitManager của v2 |
| `_pts_at()`, `_record_anchor()`, `_prune_anchors()`, `_anchor_lock/_anchor_samples/_anchor_pts` | Ánh xạ sample→PTS chỉ cần khi audio bị VAD lọc thành từng đoạn |
| `_on_speech_chunk()` | Callback VAD đẩy PCM vào ASR |
| `_request_replay()` (+ hằng số cooldown) | Không nơi nào gọi (dead code) |
| `_apply_lookahead_overrides()` | Override preview/CommitManager của v2 |
| `fed_seconds`, `feed_wall_sec`, `_feed_wall_max`, `_diag_fed_seconds`, `_diag_feed_wall`, `_starved_*` | Số đo của tầng nạp VAD đã bị xoá |
| `_MAX_ANCHORS`, `_REPLAY_REQUEST_COOLDOWN_SEC` | Hằng số của các thành phần trên |

`_feed_pts` được **giữ** nhưng đổi nguồn: `_commit_batch_progress()` đặt nó bằng mốc kết thúc khối
vừa xử lý ⇒ `fed_ahead` trong `lookahead_status`/log giờ phản ánh đúng "audio đã qua ASR".

`self.vad_processor` cũng được **giữ** nhưng chỉ để đọc `start_pad_ms` (phần bù mép-nói-sớm để canh
mốc phụ đề/TTS — xem `_refresh_start_pad`). Tuyến batch **không** đẩy audio qua VAD nữa.

### `backend/config.py` (`LookaheadConfig`)

Xoá: `processing_mode`, `preview_enabled`, `preview_min_new_audio_sec`, `inactivity_timeout_sec`,
`feed_block_sec`, `decode_starve_warn_sec`, `min_prebuffer_sec`.

### `backend/main.py`

Xoá request field `lookahead_processing_mode` / `processing_mode`, phần khôi phục từ
runtime-state và nhánh API tương ứng. Prewarm ForcedAligner giờ chỉ phụ thuộc
`config.lookahead.enabled`.

### Khác

* Comment trong `backend/vad/engines/firered.py` trỏ tới `_handle_asr_message` (đã xoá) → cập nhật
  sang `_refresh_start_pad`.
* Log khởi động: `"Khởi động Lookahead Pipeline B ở chế độ: OFFLINE_BATCH (Qwen3-ASR + Forced Aligner)"`.

---

## 2. Mặc định & Fallback

| Tình huống | Tuyến chạy |
|---|---|
| Cài mới, Lookahead bật (mặc định) | **Pipeline B OFFLINE_BATCH** |
| Người dùng tắt *Lookahead Video Buffering* trong popup | **Pipeline A** (`/ws`) |
| Trang không cho chặn buffer MSE (`checkBufferAvailable` = false) | **Pipeline A** |
| `/ws/lookahead` không kết nối được / backend báo `lookahead_unavailable` (thiếu model) | **Pipeline A** |

* Server: `LookaheadConfig.enabled = True` và **không còn** tham số chế độ — chạy Pipeline B luôn là
  OFFLINE_BATCH.
* Extension: `chkEnableLookahead` mặc định `checked`; `settings.lookaheadEnabled !== false` ⇒ cài mới
  là bật. Fallback đã có sẵn ở `content-script.js` (`result.fallbackToA` → `startPipelineA`).

---

## 3. Popup: vô hiệu tuỳ chỉnh chỉ dành cho Pipeline A

Khi Lookahead BẬT, các điều khiển sau bị `disabled` + làm mờ (`opacity 0.45`) + gắn `title` giải
thích, kèm 2 ghi chú trong popup; tắt Lookahead ⇒ mở lại bình thường:

* **VAD Engine**, **VAD Silence**, **VAD Threshold**
* **Hold cut until (s)** (`stability_min_duration_sec`)
* **Min Words Filter** (`min_words_to_commit`)
* Cả mục **Segmentation (Pipeline A)**: *Cut when the text stable*, *Stable for (ms)*,
  *Stable min words*, *Log each poll (SEG_TRACE)*

Lý do: Pipeline B cắt câu bằng dấu câu của bản phiên âm + mốc từ của Forced Aligner, **không** dùng
VAD/CommitManager; để các tuỳ chỉnh đó ở trạng thái chỉnh được sẽ khiến người dùng tưởng nhầm chúng
có tác dụng.

Thực hiện ở **cả hai** bản extension (`extension_firefox/` và `extension_chrome_edge/`):
`popup.html` thêm id cho từng nhóm + 2 ghi chú (`pipelineAOnlyNote`, `pipelineAOnlyNote2`);
`popup.js` thêm `updatePipelineAOnlyUi()` (gọi từ `updateRangeLabels()`, `setUI()`, và ngay khi bấm
công tắc Lookahead).

---

## 4. Các sửa lỗi tua (Phase 5 §9) được đồng bộ sang bản Chrome/Edge

`extension_firefox/lib/lookahead-timeline.js` đã được sao chép nguyên trạng sang
`extension_chrome_edge/lib/lookahead-timeline.js` (hai file vốn giống nhau; bản Chrome/Edge còn thiếu
phần `setReadyHorizon`/`isBehindHorizon`), và `content-script.js` của bản Chrome/Edge được áp lại
đúng 4 thay đổi: tạm dừng `underrun`, timeout 9–12 s, nạp mốc đã xử lý vào timeline queue, và cập
nhật ghi chú "đang phát". Nhờ vậy **cả hai trình duyệt** đều có ràng buộc cứng "không phát video qua
vùng chưa xử lý".

---

## 5. README

`README.md` (gốc repo) đã cập nhật:
* Sơ đồ 2 pipeline: Pipeline B ghi rõ `Lookahead + OFFLINE_BATCH` và luồng
  `RAM timeline → cắt khối 12–30 s → Qwen3-ASR trọn khối → ForcedAligner → tách câu theo dấu câu`.
* Bảng so sánh thêm các dòng: **Đầu vào ASR**, **Cách tách câu**, **Xử lý khi tua**, và ghi chú
  **tự động dự phòng** sang Pipeline A.
* Mục *"Cơ chế hoạt động của Pipeline B — OFFLINE_BATCH"*: 6 bước có sơ đồ + 6 tính năng chính +
  model cần thiết.
* Danh sách *Tính năng*: thêm các gạch đầu dòng cho OFFLINE_BATCH, ràng buộc cứng khi tua, dịch theo
  ngữ cảnh, và popup theo ngữ cảnh; mục "Cắt câu 4 bậc" ghi rõ là của Pipeline A.

---

## 6. Kiểm thử

| Lệnh | Kết quả |
|---|---|
| `python -m pytest -q` (suite nhanh) | **544 test PASS, 0 fail** (exit 0) |
| `node --test extension_firefox/tests/` | **48 pass, 0 fail** |
| `node --test extension_chrome_edge/tests/` | **46 pass, 0 fail** |

Thay đổi ở tầng test:

* **Xoá** `backend/tests/test_54_lookahead_pipeline_v2.py` (~60 test của tuyến v2; file này còn là
  nơi 2 file test khác import fixture).
* `test_55_lookahead_persistent_ram.py` (9 test) và `test_52_lookahead_e2e_stress.py` (5 test): tự
  khai báo fixture (không import từ file đã xoá) và thay `session.ingest_once()` bằng **đọc
  `ContinuousAudioTimeline` trực tiếp** — giữ nguyên các bất biến đang kiểm (audio bền vững trong
  RAM, tua lùi có audio ngay, seek trong lúc đọc không làm con trỏ nhảy về tương lai, trần RAM).
* `test_58_lookahead_offline_batch.py` (11 test): thay test "định tuyến 2 chế độ" bằng test khẳng
  định **chỉ có tuyến batch** (`la_batch_*` + `la_status_*`, không còn `la_ingest_*`/`la_asr_*`).
* **Mới** `test_59_pipeline_b_only_and_popup.py` (8 test): khoá các bất biến của yêu cầu này —
  `LookaheadConfig` không còn knob của v2; `LookaheadSessionState` không còn method của v2;
  extension mặc định bật Lookahead + giữ `fallbackToA`; popup có đủ id nhóm tuỳ chỉnh Pipeline A và
  hàm `updatePipelineAOnlyUi()`; **hai bản extension đồng bộ** ở `popup.js` và
  `lookahead-timeline.js`.

---

## 7. Ghi chú kỹ thuật còn lại

* **Bỏ slider "Thời gian dịch trước" (Pre-buffer Ahead)** khỏi popup (2026-10-02, cùng ngày):
  ở tuyến OFFLINE_BATCH, độ xa "dịch trước" do **bộ cắt khối** quyết định (12–30 s) chứ không phải
  slider 10–15 s; giá trị đó chỉ còn là cổng throttle nội bộ (`max_ahead = currentTime + lead_time +
  margin`) nên gây hiểu nhầm. Đã xoá khỏi `popup.html` + `popup.js` (cả 2 bản extension) và ngừng gửi
  `lookaheadLeadTimeSec` trong `buildWsConfig`; client gửi `lead_time` cố định 15 s, backend giữ
  `lookahead.lead_time_sec` làm mặc định. Guard: `test_59::test_popup_has_no_prebuffer_ahead_slider`.
* **GIỮ slider "Đồng bộ phụ đề / lồng tiếng"** (`lookaheadSyncOffsetMs`): nó **vẫn hoạt động** —
  `apply_init`/`apply_config` đặt `sync_offset_ms`, và `_offline_batch_loop` cộng `shift = start_pad_ms
  + sync_offset_ms` vào **mọi** `pts_start/pts_end` của phụ đề lẫn TTS (đổi được ngay khi đang chạy).
  Đây là núm canh lệch theo từng trang (một số trang lệch tới ~1 s) nên cần giữ.
* `self.vad_processor` vẫn được dựng trong `init_components()` **chỉ để đọc `start_pad_ms`** (bù mép
  nói sớm ~50 ms cho FireRed). Nếu sau này muốn bỏ hẳn VAD khỏi tuyến batch thì phải thay nguồn
  `start_pad_ms` (ví dụ dùng trực tiếp hằng số của engine) — **không** bỏ đột ngột vì sẽ lệch mốc
  phụ đề/TTS toàn bộ.
* `ContinuousAudioTimeline.read()` / `cursor_pts` / `pending_seconds()` giờ chỉ còn được dùng bởi
  test (tuyến batch dùng `get_audio_range`/`buffered_end_from`); giữ lại vì là API nạp audio của
  timeline và đang được test phủ (đã ghi chú ở docstring của test_55).
* Bản Chrome/Edge trước đây thiếu các sửa lỗi tua của Phase 5 §9 → đã đồng bộ (xem §4).

