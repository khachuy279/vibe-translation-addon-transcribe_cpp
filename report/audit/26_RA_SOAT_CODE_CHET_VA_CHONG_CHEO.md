# 26 — Rà soát CODE CHẾT và CHỒNG CHÉO CHỨC NĂNG

**Ngày:** 2026-10-08 · **Phạm vi:** `backend/` (156 file `.py`, 37.948 dòng) + `extension_src/` (23 file `.js`, 12.149 dòng)
**Trạng thái:** RÀ SOÁT + **đã áp P0, P1, P2**. Xem §11 để biết chính xác đã xoá gì và còn lại gì.

**Đã sửa trong P0 (§6):**
- §6.1 — viết lại docstring `lookahead_handler.py:1-44` theo luồng OFFLINE_BATCH v3 thật. ✅
- §6.2 — viết lại docstring `config.py:603-630`: nêu rõ đường `torch` đã xoá và trường `backend` không còn tác dụng. ✅
- §6.3 — sửa comment `lookahead_handler.py:253-258` về `_batch_dedup` (chỉ ghi, không lọc). ✅
- `.gitignore` mục 12 — bỏ tuyên bố sai về xác thực SHA-256. ✅
- §6.4 — **đính chính 2 mục sai của chính báo cáo này** (xem §6.4). ✅

---

## 0. Phương pháp và độ tin cậy

| Công cụ | Vai trò | Hạn chế đo được |
|---|---|---|
| `vulture` | Dò thô | Nhiều **dương tính giả**: route FastAPI (`@app.get`), `__getattr__`/`__dir__`, hàm chỉ gọi từ `tools/`, và "unused variable" thực chất là **tham số hàm** (API surface) |
| `scratch/audit_dead_code.py` | Quét AST toàn repo (kể cả `tests/`, `tools/`) | Đếm cả comment/docstring ⇒ chỉ **bỏ sót**, không báo sai |
| `scratch/audit_config_fields.py` | Field `config.py` không được đọc | Không phát hiện field *được đọc nhưng không ai rẽ nhánh theo* |

**Bài học đo được:** bộ đếm đơn giản rất dễ sai. `SessionState.drain_queues` ban đầu bị xếp "live" vì
`ws/handler.py:242,368` khớp grep — nhưng đó là **comment**. Sau khi đọc code: nó chỉ có caller trong
`backend/tests/`. Tương tự, `_audit_out.txt` do công cụ phụ tạo ra đã làm nhiễu phép đếm.

**Quy ước:** `refs=1` nghĩa là tên chỉ xuất hiện đúng một lần — chính dòng định nghĩa nó.

---

## 1. CODE CHẾT — hàm cấp module

| # | Vị trí | Ký hiệu | Bằng chứng | Đề xuất |
|---|---|---|---|---|
| 1 | `asr/forced_aligner.py:178` | `_looks_like_sentence_end` | `refs=1`. Hàm nó gọi (`_ends_with_abbreviation`) **vẫn live** (`:184,210,669`) | **Xoá** — an toàn, không đụng logic viết tắt |
| 2 | `asr/native.py:321` | `reset_backend_cache` | `refs=1`. Docstring ghi "dùng khi đổi cấu hình lúc chạy / trong test" — **không test nào gọi** | Xoá, hoặc đấu dây nếu thật sự cần reset cache |
| 3 | `utils/text_repetition.py:115` | `collapse_repetitions_preserve_spacing` | `refs=1`. Mọi caller dùng `collapse_repetitions` trực tiếp | Xoá (hoặc đấu dây nếu ý định "giữ khoảng trắng" còn cần) |
| 4 | `utils/text_repetition.py:112` | `_WS` | Khai báo, không dùng; code dùng `" ".join(text.split())` | Xoá |
| 5 | `vad/engines/firered_onnx.py:313` | `_sha256_of` | `refs=1` — trùng chức năng `crispasr_native._sha256_file` | Xoá (xem §5.1) |
| 6 | `ws/serializers.py:17` | `strip_dup_fields` | `refs=1`. Tự mô tả "dùng để kiểm chứng payload gọn" nhưng không nơi nào dùng | Xoá |
| 7 | `utils/crispasr_native.py:92` | `BUNDLE_INNER_DIR` | `refs=1` | Xoá |
| 8 | `utils/logger.py:39` | `from typing import Optional` | 0 lần dùng trong file | Xoá import |
| 9 | `utils/logger.py:56,66,67,68` | `LogColors.BLACK/BG_GREEN/BG_YELLOW/BG_BLUE` | Chỉ `WHITE` + `BG_RED` được dùng (`:82,121`) | Xoá 4 hằng |
| 10 | `utils/logger.py:110` + `:72` | `icon` + `LEVEL_ICONS` | `icon` được gán, **không bao giờ đọc** (docstring quy tắc 5: "KHÔNG emoji trong log") | Xoá cả hai |
| 11 | `asr/engine.py:58` | `import native_bundle_dir` | Import không dùng (vulture 90%) | Xoá import |
| 12 | `ws/lookahead_handler.py:41`, `:57` | `import sys`, `AlignedWord` | Mỗi cái 1 lần xuất hiện | Xoá import |

## 2. CODE CHẾT — method / thuộc tính

| # | Vị trí | Ký hiệu | Bằng chứng | Đề xuất |
|---|---|---|---|---|
| 13 | `asr/engine.py:1185` | `transcribe_pcm` | `refs=1`. Docstring nói "dùng cho Lookahead Batch Ingress" nhưng Lookahead dùng `_run_asr_block` (`lookahead_handler.py:929`) | Xoá — API bị thay thế |
| 14 | `asr/registry.py:134` | `is_streaming_model` | `refs=1` | Xoá |
| 15 | `config.py:655` | `AppConfig.hot_reload` | `refs=1`, kể cả trong test | Xoá |
| 16 | `core/lookahead_timeline.py:106` | `total_stored_bytes` | `refs=1`. Anh em `total_stored_seconds` (`:101`) **live** | Xoá |
| 17 | `core/lookahead_timeline.py:93` | `session_base_pts` (property) | `refs=1` | Xoá |
| 18 | `core/stream_demuxer.py:459` | `drop_init` | `refs=1` | Xoá |
| 19 | `core/stream_demuxer.py:1454` | `coverage_report` | `refs=1`. Anh em `structure_report` (`:1468`) **live** | Xoá |
| 20 | `tts/base.py:12` + `tts/engine.py:89` | `is_model_ready` | Hợp đồng base + override, **0 caller**. Bị bộ dò bỏ sót vì tên định nghĩa 2 lần | Xoá cả hai |
| 21 | `vad/engines/__init__.py:122` | `reset_pool` (classmethod) | `refs=1` | Xoá |
| 22 | `ws/session.py:125` | `SessionConfig.to_dict` | `refs=1` | Xoá |
| 23 | `ws/session.py:166-176` | `SessionState.ws` property + setter | 0 truy cập `session.ws` toàn repo | Xoá |
| 24 | `ws/session.py:182` | `SessionState.send_text` | 0 caller (các hit `.send_text(` là `SafeWebSocketConnection.send_text`) | Xoá |
| 25 | `ws/session.py:544` | `drain_queues` | **Chỉ test gọi** (`test_07/21/24/25`). `handler.py:242,368` là **comment** — production dùng `_discard_queued` (`:371`) | Xoá hoặc đánh dấu rõ là test seam |
| 26 | `ws/lookahead_handler.py:118` | `_ends_sentence` | Chỉ `tools/diag_lookahead_boundary.py` gọi | Chuyển vào tool, hoặc giữ |
| 27 | `ws/lookahead_handler.py:76,312,314,316,317,323` | `_START_OFFSET_WARN_SEC`, `decode_windows`, `start_offset_warned`, `_seek_reset_at`, `_seek_diag_left`, `_last_decoded_at` | Mỗi cái ghi 1 lần, **không bao giờ đọc** — di sản chẩn đoán v2 | Xoá |
| 28 | `ws/lookahead_handler.py:2075` | biến cục bộ `d_media` | Tính ra nhưng **không có trong f-string log** (`:2082-2091`) | Xoá |
| 29 | `ws/lookahead_handler.py:242` | `_batch_dedup` | **Chỉ ghi**: `.clear()` (`:877,985`) + `.record_commit()` (`:1636`). **Không bao giờ** `.is_duplicate()`/`.trim_boundary_overlap()`. Bộ lọc thật là `_is_duplicate` (`:1945`) | Xoá, hoặc đấu dây |

## 3. Extension — code chết (đã xác minh)

| # | Vị trí | Ký hiệu | Bằng chứng |
|---|---|---|---|
| 30 | `content/buffer_interceptor_poc.js:1258-1292` | `refetchAudioRange` | **35 dòng**, 0 caller. Bản live là `refetchForPlayhead` (gọi ở `:689`) |
| 31 | `lib/ws-client.js:332-343` + `background/service-worker.js:273` | `_buildPacketFallback` + fallback nội tuyến | **Không thể chạy**: `frame-builder.js` luôn được nạp (content_scripts + `importScripts`) |
| 32 | `popup/popup.js:70` + 7 nhánh `if (lblActiveModel)` | `lblActiveModel` | `popup.html:53-58` đã **comment out** khối HTML ⇒ `getElementById` luôn `null` |
| 33 | `content/content-script.js:566-569` → `overlay-manager.js:430-432` → `subtitle-renderer.js:198-201` | `set_overlay_mode` → `setMode()` | Chuỗi chết: `setMode()` là no-op, `this.mode` chỉ ghi không đọc |
| 34 | `lib/audio-capture.js:39-41` | `supportsGainDucking` | 0 caller (kể cả test) |
| 35 | `lib/lookahead-timeline.js:486-501` | `SubtitleTimelineQueue.stats` | 0 caller |
| 36 | `lib/lookahead-diagnostics.js:361,368` | `verbose` + `setVerbose` | Ghi 2 lần, **không bao giờ đọc** |
| 37 | `lib/lookahead-diagnostics.js:407-432` | `once`, `resetOnce`, `count` | 0 caller (chỉ `throttled` được dùng) |
| 38 | `lib/ws-client.js:401,46,243` | `off()`, `supportsBinaryTts`, emit `"reconnecting"` | `off()` 0 caller; `supportsBinaryTts` ghi mà không đọc; không có listener `reconnecting` |
| 39 | `content/content-script.js:67,1835,1842` | `__bsLookaheadDiag`, `__bsLookaheadVerbose`, `__bsFindVideo`, `__bsLookaheadDiagReport` | Chỉ xuất hiện ở khai báo + chuỗi console. Đối chiếu: `__bsStartCapture/…` **được popup đọc** (`popup.js:1035-1045`) ⇒ live |
| 40 | `content/buffer_interceptor_poc.js:1434` | `__VIBE_LOOKAHEAD_INTERVAL__` | Handle `setInterval` lưu mà không đọc, **không `clearInterval`** |
| 41 | `lib/lookahead-diagnostics.js:495` | `SRC_KIND_TEXT` | Chỉ dùng nội bộ `:203`; không đọc như export |
| 42 | `popup/popup.css:365,394,400-402` | `.toggle-icon`, `.model-info-*` | Không HTML/JS nào tham chiếu — di sản của HUD đã comment |
| 43 | `popup/popup.html:79,84,226,254,312,326` | `lookaheadStatusDot`, `lookaheadSyncGroup`, `stableToggleRow`, `stableTraceToggleRow`, `duckingSliderRow`, `translationOnceToggleRow` | id có trong HTML nhưng **không JS nào đọc** (đối chiếu `showOriginalToggleRow`/`ttsToggleRow` **có** handler) ⇒ còn là lỗ hổng UX |

## 4. Config chết

| # | Vị trí | Field | Bằng chứng |
|---|---|---|---|
| 44 | `config.py:459` | `MetricsConfig.alert_threshold_ms` | **0 lần đọc** toàn repo |
| 45 | `config.py:586` | `LookaheadConfig.tts_tail_allowance_sec` | **0 lần đọc** toàn repo |
| 46 | `config.py:101` | `FireRedVADConfig.chunk_max_frame` | Chỉ `test_41` đọc (và còn **loại trừ nó** khỏi so sánh). Comment tự ghi "không dùng khi stream" |
| 47 | `config.py:606-624` | `ForcedAlignerConfig.backend` | **Được đọc nhưng không ai rẽ nhánh theo.** Xem §6.2 — nghiêm trọng vì tài liệu sai |

**Đối chiếu tích cực:** chỉ **2/193** field trong `config.py` là chết hoàn toàn. Config nhìn chung được đấu dây tốt.

## 5. Chồng chéo chức năng

### 5.1 SHA-256 cài 5 lần ở 4 module
`utils/crispasr_native.py:112` (`_sha256_file`) · `utils/crispasr_native.py:175` (nội tuyến) ·
`vad/silero_onnx.py:114` (`_sha256`) · `vad/engines/firered_onnx.py:313` (`_sha256_of`, **chết**) ·
`vad/engines/firered_onnx.py:360` (`_sha256_of_bytes`)
⇒ Nên có **một** helper dùng chung trong `utils/`.

### 5.2 Bốn bộ tải riêng, và một lỗ hổng toàn vẹn dữ liệu
`utils/model_download.py` là bản chuẩn (`hf_hub_download`, có `ModelFileMissing`/`ModelDownloadError`,
progress, phân tầng). Nhưng **4 module tự viết lại bằng `urllib`**:

| Module | Cách tải | Xác thực SHA-256 |
|---|---|---|
| `vad/engines/firered_onnx.py:323` | `urlopen` | ✅ có |
| `vad/silero_onnx.py:120` | `urlopen` | ✅ có |
| `utils/crispasr_native.py:151-175` | `urlopen` + zip | ✅ có |
| **`diarization/service.py:58,63,87`** | **`urlretrieve` ×3** | ❌ **KHÔNG có** — chỉ kiểm tra kích thước > 100 MB |

`.gitignore:111-116` tuyên bố audiocpp tải "**tương tự cơ chế của CrispASR**" (tức có xác thực SHA-256).
**Thực tế không có.** Đây là hậu quả trực tiếp của việc nhân bản logic tải.

### 5.3 Đọc RSS cài 2 lần (~35 dòng gần y hệt)
`utils/mem_guard.py:28-63` `rss_mb()` **so với** `asr/engine.py:147-180` `_process_rss_mb()`.
Khác biệt **duy nhất**: `K32GetProcessMemoryInfo` vs `psapi.GetProcessMemoryInfo` (cùng một hàm Win32).
⇒ `engine.py` nên gọi `mem_guard.rss_mb()`.

### 5.4 Định dạng khung audio cài 4 lần
`[4 byte LE header len][JSON][PCM]` được dựng lại ở: `lib/frame-builder.js:12-43` (chuẩn, live) ·
`lib/lookahead-client.js:629` · `lib/ws-client.js:332-343` (*không thể chạy*) · `background/service-worker.js:273` (*không thể chạy*).

### 5.5 Ba bộ dedup với BA quy tắc chuẩn hoá khác nhau
| Module | Chuẩn hoá | Ngữ nghĩa |
|---|---|---|
| `core/dedup.py:19` `normalize_for_dedup` | bỏ dấu câu (kể cả CJK) + lower + gộp space | 3 tầng: exact + substring ≥0.85 + Jaccard ≥0.80, cửa sổ thời gian |
| `translation/dedup.py:33` `_key` | **chỉ `strip().lower()`** | exact match, TTL 10 s |
| `tts/dedup.py:21` `_normalize` | lower + bỏ dấu câu **ở hai ĐẦU** + gộp space | exact + tiền tố/hậu tố ≥8 ký tự |

⇒ Cùng một câu có dấu câu cuối sẽ được lọc **khác nhau** ở đường dịch và đường TTS. Chưa xác nhận là bug,
nhưng là điểm không nhất quán cần biết.

### 5.6 `popup.js` trùng lặp ~200 dòng bên trong một file
`waitForAsrActivation` (`:725-770`) vs `waitForTranslationActivation` (`:774-819`) — **83 % giống** ·
`renderAsrEngineOptions` (`:345-372`) vs `renderTranslationModelOptions` (`:374-401`) — **65 % giống** ·
`monitorAsrDownload` vs `monitorTranslationDownload` · `handleEngineSwitch` (`:609-710`) vs `handleTranslationModelSwitch` (`:875-933`).

### 5.7 Pipeline A vs Pipeline B
- `_spawn`: `ws/session.py:186` vs `ws/lookahead_handler.py:569` — **giống hệt từng byte**, chỉ khác log.
- `send_json` **3 lớp** bọc nhau, hai lớp cuối chỉ khác một kiểm tra `_closed`.
- `_schedule_asr_model_switch`/`_schedule_translation_model_switch` **cài đặt 2 lần** (`session.py:241,309` vs `lookahead_handler.py:501,537`).
- **Chính sách hàng đợi TTS NGƯỢC NHAU**: `handler.py:192 _coalesce_enqueue` (khoá `ws.tts_queue_maxsize`) **GỘP** khi đầy;
  `lookahead_handler.py:1658 _enqueue_tts` + `:1675 _trim_tts_queue` (khoá `lookahead.tts_max_queue`) **BỎ CŨ NHẤT**.
  Cùng khái niệm, 2 config key, 2 hành vi trái ngược.
- `_is_duplicate` (`lookahead_handler.py:1945`) nhận tham số `pts_end` nhưng **không dùng**.

### 5.8 Payload `model_status` lặp 18 lần dạng dict literal
`session.py` 12 chỗ (`:251,272,280,290,297,303,326,348,356,366,373,379`) + `lookahead_handler.py` 6 chỗ (`:516,527,531,554,559,563`),
trong khi `serializers.make_model_status_msg` (`:173`) đã tồn tại nhưng **chỉ test dùng**.

### 5.9 Parse config lặp
`min_words_to_commit`: **5 chỗ** (`main.py:1013,1147`, `session.py:503`, `lookahead_handler.py:475,2125`) ·
VAD `"0 = tắt" → None`: **4 chỗ** · clamp `lookaheadSyncOffsetMs`: **2 chỗ** (`:588-594`, `:2131-2136`) ·
`main.py:1138-1170` lặp lại bảng alias của `SessionConfigPayload` (`session.py:37-77`) ·
Pipeline B parse payload **2 lần** (`apply_config` và `apply_init`).

### 5.10 Khác
- `main.py:112-175` `_log_runtime_status_at_startup()` cài lại `env_check.runtime_status()`.
- `content/content-script.js:107-170` vs `popup/popup.js:166-236`: **2 bản** chuẩn hoá settings→protocol.
- `buffer_interceptor_poc.js:196-201` `containsMagic` vs `hasMagic` nội tuyến `:486-491`.
- **Tách câu: 2 cài đặt ngữ nghĩa khác nhau** — `core/hypothesis.py:57 _split_sentences` (đơn giản, ~17 dòng,
  **không xử lý viết tắt/số thập phân**) vs `asr/forced_aligner.py:188 _split_text_by_sentence` (đầy đủ: `3.14`,
  `domain.com`, `Dr.`, ngoặc đóng). Phục vụ 2 mục đích khác nhau nên **chấp nhận được**, nhưng cần biết là
  bộ đếm preview dùng bản đơn giản.

## 6. Tài liệu/comment SAI SỰ THẬT — nhóm nguy hiểm nhất

Sai ở đây không làm test đỏ, nhưng làm **người đọc tin vào thứ không tồn tại**.

### 6.1 `lookahead_handler.py:1-32` mô tả kiến trúc đã bị xoá — ✅ ĐÃ SỬA (P0)
Docstring module vẫn vẽ luồng **Pipeline B v2**: `VADProcessor.feed_chunk` → `TranscribeEngine.stream_tokens`
→ ánh xạ "anchor". Trong file đó, `feed_chunk`/`stream_tokens` **chỉ xuất hiện ở đúng 2 dòng docstring này**
(`:24,25`), không có trong mã chạy.
**Mâu thuẫn ngay trong cùng file:** docstring class `:178-184` ghi rõ *"v2 'streaming giả lập' … đã bị XOÁ ngày 2026-10-02"*.
⇒ **Sửa docstring `:1-32`** (ưu tiên cao nhất về chi phí gây nhầm lẫn).

### 6.2 `config.py:613-621` hướng dẫn bật một đường đã bị xoá — ✅ ĐÃ SỬA (P0)
Docstring nói còn **2 đường**: `"crispasr"` và `"torch"`, và:
> *"Vẫn giữ làm ĐƯỜNG DỰ PHÒNG: nếu thiếu DLL/model GGUF thì ForcedAlignerService tự rơi về đây kèm cảnh báo"*
> *"Muốn quay lại đường cũ: đặt `forced_aligner.backend = "torch"` (không cần sửa mã)."*

Nhưng `asr/forced_aligner.py:37-42` ghi: *"KHÔNG còn torch/transformers/qwen_asr trong file này … Giai đoạn 3
(2026-10-06) đã XOÁ hẳn đường torch"*, và `test_64_crispasr_aligner.py:160` khẳng định các API torch đã biến mất.
Không mã nào rẽ nhánh theo `forced_aligner.backend` (chỉ `main.py:211` và `env_check.py:67` đọc để **báo cáo**).
**Đối chiếu:** `asr.backend` thật sự hoạt động — có `resolve_backend()` và fallback (`native.py`).
⇒ Đặt `"torch"` hiện **không có tác dụng gì** và không có fallback nào tồn tại.

### 6.3 `lookahead_handler.py:241` mô tả sai về `_batch_dedup` — ✅ ĐÃ SỬA (P0)
Comment: *"Bộ lọc trùng của Pipeline A — chỉ còn dùng cho trường hợp aligner lỗi hẳn (fallback)."*
Thực tế: field chỉ được `.clear()` và `.record_commit()`; **không nhánh fallback nào tồn tại**, và
`is_duplicate()` không bao giờ được gọi trên nó.

### 6.4 Khác
- `.gitignore:111-116` tuyên bố audiocpp xác thực SHA-256 "tương tự cơ chế của CrispASR" — xem §5.2.

**⚠️ ĐÍNH CHÍNH (2026-10-08, sau khi kiểm chứng lại).** Bản đầu của mục này có 2 khẳng định **SAI**,
đã rút lại:

| Khẳng định sai | Sự thật |
|---|---|
| "`tools/diag_lookahead_boundary.py:1` gọi tuyến batch là *Pipeline B v3*" — ngụ ý lỗi thời | **"Pipeline B v3" là TÊN ĐÚNG** của kiến trúc hiện hành. Dùng nhất quán ở `core/lookahead_chunker.py:1,5,10`, `core/lookahead_timeline.py:6`, `lookahead_handler.py` (`_offline_batch_loop`), `test_56:112`, `test_58:1`. Quy ước: **v2 = streaming giả lập đã xoá**, **v3 = OFFLINE_BATCH hiện hành**. Thứ thật sự sai là docstring đầu `lookahead_handler.py` dùng nhãn "v2". |
| "`ws/handler.py:383-385` nói config metrics là dead — nay đã stale" | Đọc trong ngữ cảnh, câu đó mở đầu bằng *"TRƯỚC ĐÂY ĐOẠN NÀY BỊ THIẾU"* ⇒ nó là **ghi chú lịch sử** giải thích vì sao khối lệnh bên dưới tồn tại, không phải khẳng định sai ở hiện tại. Không cần sửa. |

**Bài học:** chính bộ dò tự động cũng mắc lỗi mà nó đi tìm — 2/4 mục của §6.4 là dương tính giả.
Đây là lý do P0 chỉ sửa những chỗ đã đọc code xác nhận.

## 7. Giao thức WebSocket chết

| `type` | Chiều | Bằng chứng | Kết luận |
|---|---|---|---|
| `stream_reset` | B→E | `handler.py:605` gửi; `extension_src` có **0** tham chiếu | **CHẾT** (chỉ `test_21` assert) |
| `pong` | B→E | `handler.py:444`, `lookahead_handler.py:2305`; không listener nào đọc | **CHẾT** (lành tính — `ping` chỉ để giữ nhịp) |
| `lookahead_request_replay` | E | `lookahead-client.js:514-526` xử lý; backend **0** nơi phát (producer đã xoá) | **CHẾT** (di sản v2) |
| `partial_transcript` | E | `content-script.js:826` lắng nghe; backend **không bao giờ phát** | **CHẾT** — cả chuỗi `:322,330,340` → `overlay-manager:492-496` → `subtitle-renderer:730-740` |
| `sentence_complete` | E | Không producer; `content-script.js:332,342` | **CHẾT** — chuỗi nhánh |
| `set_overlay_mode` | E | Không producer; `content-script.js:566-569` | **CHẾT** |
| `bye`, `close`, `reset_stream` (trên `/ws/lookahead`) | E→B | `lookahead_handler.py:2282,2295`; không sender (extension gửi `stop`) | **CHẾT** — alias cũ |
| `ws_json_raw` | E | `service-worker.js:225` phát; không ai đọc | **CHẾT** |
| `SEND_BUFFER` | E | Không producer; `service-worker.js:280` xử lý (alias live là `SEND_RAW_BINARY`) | **CHẾT** |
| `binary_tts` | B→E | `handler.py:297`; chỉ `ws-client.js:88` ghi, không đọc | **CHẾT** field |
| `model_status` trên `/ws/lookahead` | B→E | `lookahead_handler.py:516…563`; `lookahead-client` **không có case** | **CHẾT** trên tuyến B (live trên `/ws`) |
| `backpressure` trên Pipeline B | B→E | `service-worker.js:99` → chỉ `ws-client.js:97` đọc | **CHẾT** với Pipeline B — `lookahead-client` bỏ im lặng |
| `ping` trên `/ws/lookahead` | E→B | `lookahead_handler.py:2304`; chỉ test gửi | **CHỈ TEST** |

**Live (không xoá):** `connected`, `utterance_update`, `translation`, `tts_audio`/`tts_binary`, `model_status` (trên `/ws`),
`set_config`+`action:configure`, `reset_stream`, `ping` (trên `/ws`), `audio_chunk`, `lookahead_init`, `sync_state`,
`seek_reset`, `stop`, `lookahead_ready`/`_unavailable`/`_subtitles`/`_tts`/`_status`, `seek_acknowledged`,
`INJECT_MAIN_WORLD_INTERCEPTOR`, `BROADCAST_SUBTITLE`, `SUBTITLE_RENDER`, `GET_STATUS`, `START_TRANSLATION`,
`STOP_TRANSLATION`, `update_settings`, `CONNECT`, `SEND_JSON`, `SEND_BINARY`, `SEND_RAW_BINARY`, `FLUSH_PENDING`, `DISCONNECT`.

## 8. Thứ tự hành động đề xuất

**P0 — Sửa tài liệu sai (rủi ro 0, giá trị cao)**
1. Viết lại `lookahead_handler.py:1-32` theo luồng OFFLINE_BATCH thật.
2. Sửa `config.py:606-624`: bỏ mô tả đường `torch`, hoặc khôi phục fallback thật.
3. Sửa comment `lookahead_handler.py:241` và `.gitignore:111-116`.

**P1 — Vá lỗ hổng toàn vẹn dữ liệu**
4. Thêm xác thực SHA-256 cho 3 lượt tải trong `diarization/service.py` (hoặc chuyển sang `utils/model_download`).

**P2 — Xoá code chết đã xác minh (37 mục ở §1–§3)**
5. Ưu tiên nhóm không phải sửa test: 12 hàm/thuộc tính §1, các mục §2 không có caller test.

**P3 — Hợp nhất chồng chéo**
6. Một helper SHA-256 dùng chung; `engine.py` dùng `mem_guard.rss_mb`; một bộ dựng khung audio.
7. `make_model_status_msg` thành bộ dựng duy nhất; gộp `_spawn`; thống nhất chính sách hàng đợi TTS.
8. Tách `popup.js` và gộp 2 cặp `waitFor*`/`render*`.

## 9. KHÔNG NÊN XOÁ (trông chết nhưng đang sống)

- **`utils/env_check.py`** — hợp đồng CLI (`README.md:93`, `requirements.txt:18`) + `test_45`.
- **`serializers`**: `make_utterance_update_msg`, `make_translation_msg`, `make_tts_audio_msg`, `make_tts_binary_frame`, `make_pong_msg`, `parse_audio_frame` — đều live.
- **`CommitDeduplicator` / `normalize_for_dedup`** — live qua `core/commit_manager.py:72`.
- **`commit_manager.is_text_filtered`** — **chủ ý giữ**: docstring `:95-98` đã ghi rõ "KHÔNG ĐƯỢC GỌI TRONG RUNTIME… Giữ lại để không phá test cũ".
- **7 script `backend/tools/*.py`** — đã kiểm tra import runtime: **tất cả nạp sạch**, không cái nào hỏng.
- **`SessionConfigPayload`** — schema validate duy nhất cho cả 2 pipeline + 4 file test.
- **`ForcedAlignerService` / `forced_aligner.py`** — LIVE (`lookahead_handler.py:1462,1467,1487,1495`); đường torch đã được dọn sạch trước đó.
- **`lib/browser-config.js`** — không có trong `extension_src` **là đúng thiết kế** (do `tools/build_extensions.py` sinh).
- **`lib/backpressure-gate.js`** — không nằm trong `content_scripts` nhưng là entry `background.scripts` của Firefox + `importScripts` ở Chrome.
- **`audio-processor.js process()`** — entry point AudioWorklet, do audio thread gọi.
- **`overlay-manager._onWindowResize`/`_onFullscreenChange`, `lookahead-timeline._onVideoSeeking`/`_onVideoSeeked`** — truyền dưới dạng callback đã bind, không có call site `.name(`.
- **`window.__bsStartCapture/StopCapture/GetStatus/UpdateSettings`** — popup đọc thật (`popup.js:1035-1045`).
- **`BackpressureGate.ACTIONS/SOFT_LIMIT_BYTES/HARD_LIMIT_BYTES`** — test đọc.
- **`popup.css .message-*`** — dựng động qua `"message-area message-" + tp`.

## 10. Chưa xác minh

- Không có client ngoài `extension_src/` (bản extension cũ đã phát hành, script của người dùng) đang gửi
  `bye`/`close`/`lookahead reset_stream` hoặc đọc `stream_reset`/`pong` hay không — chỉ tìm được trong repo.
- Các kết luận đều dựa trên **tham chiếu tĩnh**; chưa chạy end-to-end trong trình duyệt thật. Riêng
  `lblActiveModel` và các id HTML chết được kết luận từ đọc HTML tĩnh, chưa dump DOM sống.
- Chưa đo được lợi ích hiệu năng của việc xoá code chết (dự kiến ~0).

---

## 11. TRẠNG THÁI THỰC THI (P0 / P1 / P2)

### P1 — Xác thực SHA-256 cho Diarization ✅ XONG

| Việc | Chi tiết |
|---|---|
| Hash lấy từ NGUỒN CHÍNH THỨC | 2 zip: trường `digest` của GitHub API cho release `v0.9.1`. Model GGUF: LFS `oid` của HuggingFace (= SHA-256). **Không tự băm file cục bộ để suy ra hash.** |
| Đối chiếu chéo | SHA-256 của `nemotron-3-diarization-bf16.gguf` trên đĩa **khớp đúng** LFS oid (và khớp cả kích thước 198.720.928 byte) |
| Helper dùng chung | `model_download.sha256_file()` / `sha256_bytes()` / `verify_sha256()` — tránh tạo bản SHA-256 thứ 6 |
| `_fetch_verified()` | Tải → kiểm hash → **TỪ CHỐI** nếu lệch; xoá file tạm khi bị từ chối |
| Test hồi quy | `backend/tests/test_68_diarization_download_integrity.py` — 12 test, gồm kiểm chứng **hành vi** (hash sai ⇒ trả `False`), không chỉ đọc mã nguồn |

Cố ý **không** băm lại file đã có lúc khởi động: `ggml-cuda.dll` nặng 1,09 GB, băm mỗi lần khởi động
tốn vài giây vô ích. Xác thực đặt ở **ranh giới tải** (supply-chain), không phải mỗi lần đọc.

### P2 — Xoá code chết

**Backend ✅ XONG — bộ dò còn 0 mục.**

Xoá 8 hàm cấp module: `_looks_like_sentence_end`, `native_bundle_dir`, `native_bundle_source`,
`reset_backend_cache`, `build_glossary_hint`, `import_error`, `collapse_repetitions_preserve_spacing`,
`_sha256_of`, `strip_dup_fields`.
Xoá 15 method/thuộc tính: `transcribe_pcm`, `is_streaming_model`, `hot_reload`, `total_stored_bytes`,
`session_base_pts`, `drop_init`, `coverage_report`, `is_model_ready` (×2), `has_voice`, `snapshot`,
`restore`, `reset_pool`, `to_dict`, `SessionState.ws` (property+setter), `send_text`, `init_bytes`,
`off`, `supportsBinaryTts`.
Xoá import/hằng chết: `import sys`, `AlignedWord`, `from typing import Optional`, `Iterable` (×2),
`native_bundle_dir` (import), `LogColors.{BLACK,BG_GREEN,BG_YELLOW,BG_BLUE}`, `LEVEL_ICONS` + biến `icon`,
`BUNDLE_INNER_DIR`, và 6 thuộc tính chẩn đoán v2 của `lookahead_handler` + biến cục bộ `d_media`.

**Extension 🔶 LÀM MỘT PHẦN** — đã xoá:
`refetchAudioRange` (35 dòng, trùng `refetchForPlayhead`), `_buildPacketFallback` + fallback nội tuyến
trong service worker (không thể chạy), `SubtitleTimelineQueue.stats()` (19 dòng),
`LookaheadDiag.{once,resetOnce,count,setVerbose}` + `verbose`/`counters`/`_onceKeys`,
`WSClient.off()`, `supportsBinaryTts` + nhánh `binary_tts`, export `SRC_KIND_TEXT`,
và `window.__bsLookaheadVerbose` (API nói dối: cờ `verbose` không được đọc ở đâu).

**CÒN LẠI của P2 (chưa làm, cần quyết định riêng):**
- Chuỗi giao thức chết nhiều file: `partial_transcript`, `sentence_complete`, `set_overlay_mode`
  (chạm `content-script.js` → `overlay-manager.js` → `subtitle-renderer.js`), `ws_json_raw`, `SEND_BUFFER`.
- `popup/popup.js:70` `lblActiveModel` + 7 nhánh `if` — HTML đã comment out. Cần chọn: mở lại HTML **hoặc** xoá JS.
- HTML/CSS chết: 6 id trong `popup.html`, 5 class trong `popup.css`.
- Hợp nhất chồng chéo (§5): SHA-256 còn 3 bản ở VAD/crispasr, 3 bộ tải `urllib`, đọc RSS 2 bản,
  định dạng khung audio, 3 bộ dedup khác quy tắc, `popup.js` ~200 dòng lặp.

### Ba bài học đo được (đã trả giá để biết)

1. **Đếm bằng regex là SAI.** Bản đầu của bộ dò tính cả comment/docstring là "tham chiếu". Hậu quả:
   `native_bundle_source`, `build_glossary_hint`, `import_error`, `has_voice`, `restore` bị ẩn khỏi báo cáo.
   Tệ hơn, một comment do chính tác giả viết để tài liệu hoá việc xoá lại che mất thứ nó đang nói tới.
   ⇒ Bộ dò nay chỉ đếm **định danh trong AST**.
2. **Phải quét CẢ nơi tiêu thụ ngoài thư mục nguồn.** `AudioCapture.supportsGainDucking()` bị xoá nhầm vì
   nó **chỉ được gọi từ `backend/tests/js/audio_capture_ducking_test.js`**, nằm ngoài `extension_src/`.
   Test suite bắt được ngay (`test_31`), đã khôi phục. ⇒ Trước khi xoá symbol của extension, phải grep
   cả `backend/tests/`.
3. **Không thể tin một nguồn duy nhất.** Chính báo cáo này có 2/4 mục §6.4 sai; chính bộ dò của tôi có
   3 lỗi logic. Mọi kết luận "chết" đều phải đọc code xác nhận, không chỉ dựa vào công cụ.

