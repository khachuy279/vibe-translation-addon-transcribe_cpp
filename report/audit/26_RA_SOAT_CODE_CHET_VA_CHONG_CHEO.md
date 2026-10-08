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

**TRẠNG THÁI TỪNG MỤC** (cập nhật 2026-10-08):

| Mục | Nội dung | Trạng thái |
|---|---|---|
| 5.1 | SHA-256 cài 5 lần | ✅ xong (P2-bis) |
| 5.2 | 4 bộ tải riêng + lỗ hổng SHA của diarization | ✅ xong (P1 + P3-bis) |
| 5.3 | Đọc RSS 2 lần | ✅ xong (P2-bis) |
| 5.4 | Định dạng khung audio 4 lần | ✅ xong (P3-ter) |
| 5.5 | 3 bộ dedup, 3 quy tắc chuẩn hoá | ✅ xong (chuẩn hoá hợp nhất; chiến lược khớp giữ riêng có chủ đích) |
| 5.6 | `popup.js` trùng lặp | ✅ xong một phần (P3; 2 cặp `handle*Switch` cố ý không gộp) |
| 5.7 | Pipeline A vs B trùng lặp | ✅ xong (`_spawn` gộp, `pts_end` bỏ; 2 mục còn lại **cố ý** giữ riêng, đã ghi lý do) |
| 5.8 | `model_status` 18 dict literal | ✅ xong (P3-ter) |
| 5.9 | Parse config lặp | ✅ xong (4 quy ước gom về `ws/session.py`; 2 mục payload để lại có lý do) |
| 5.10 | Khác | ✅ xong 3/4 (tách câu giữ nguyên có chủ đích) |

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

### 5.5 Ba bộ dedup với BA quy tắc chuẩn hoá khác nhau — ✅ ĐÃ HỢP NHẤT

| Module | Chuẩn hoá TRƯỚC | Ngữ nghĩa khớp (giữ nguyên) |
|---|---|---|
| `core/dedup.py:19` `normalize_for_dedup` | bỏ dấu câu (kể cả CJK) + lower + gộp space | 3 tầng: exact + substring ≥0.85 + Jaccard ≥0.80, cửa sổ thời gian |
| `translation/dedup.py:33` `_key` | **chỉ `strip().lower()`** | exact match, TTL 10 s + cache bản dịch |
| `tts/dedup.py:21` `_normalize` | lower + bỏ dấu câu **ở hai ĐẦU** + gộp space | exact + tiền tố/hậu tố ≥8 ký tự |

**Hệ quả đo được trước khi sửa:** cùng câu `"Xin chào, các bạn!"` so với `"Xin chào các bạn"` cho **ba kết luận khác nhau** — `core` nói trùng, `tts` nói trùng, `translation` nói KHÔNG trùng. Dấu câu từ ASR vốn không đáng tin (aligner bỏ hết khi tokenize), nên bỏ dấu câu khi so trùng là quy tắc đúng cho cả ba tầng.

**ĐÃ SỬA:** `translation/dedup._key` và `tts/dedup._normalize` nay uỷ quyền cho `core.dedup.normalize_for_dedup`. Kiểm chứng: 11 mẫu (dấu câu cuối, dấu câu GIỮA, CJK, khoảng trắng thừa, gạch nối, rỗng) cho kết quả **đồng nhất 100 %** ở cả ba lối vào.

**CHIẾN LƯỢC KHỚP thì CỐ Ý KHÔNG GỘP** — ba bộ phục vụ ba mục đích khác nhau ở ba tầng pipeline (chống hallucination / tránh dịch lại / tránh phát lại) với vòng đời khác nhau (cửa sổ thời gian / TTL / số câu). Gộp chúng thành một lớp sẽ là abstraction giả và có thể đổi hành vi lọc.

**Test hồi quy:** `backend/tests/test_72_dedup_normalization.py` — 15 test, gồm:
- bất biến CHÍNH: ba lối vào chuẩn hoá cho cùng kết quả trên 11 mẫu;
- cùng câu khác dấu câu ⇒ trùng ở **cả ba** tầng (trước đây chỉ 2/3);
- câu khác thật sự vẫn được chấp nhận (chuẩn hoá mạnh hơn không được gây dương tính giả);
- chốt mã nguồn: `translation/dedup.py` và `tts/dedup.py` không được chứa `re.sub` (tự chuẩn hoá trở lại).
  Đã kiểm chứng guard **FAIL thật** khi ép `tts/dedup.py` quay về quy tắc cũ, và pass lại sau khi khôi phục.


### 5.6 `popup.js` trùng lặp ~200 dòng bên trong một file
`waitForAsrActivation` (`:725-770`) vs `waitForTranslationActivation` (`:774-819`) — **83 % giống** ·
`renderAsrEngineOptions` (`:345-372`) vs `renderTranslationModelOptions` (`:374-401`) — **65 % giống** ·
`monitorAsrDownload` vs `monitorTranslationDownload` · `handleEngineSwitch` (`:609-710`) vs `handleTranslationModelSwitch` (`:875-933`).

### 5.7 Pipeline A vs Pipeline B — ✅ ĐÃ SỬA XONG (2 mục sửa, 2 mục ghi rõ là cố ý)

| Mục | Kết quả |
|---|---|
| `_spawn` giống hệt từng byte | ✅ **ĐÃ SỬA** — tách `spawn_background_task(coro, sink)` ở `session.py`; cả hai lớp uỷ quyền. `get_running_loop().create_task` nay chỉ còn **1 chỗ** trong toàn backend |
| `_is_duplicate` nhận `pts_end` mà không dùng | ✅ **ĐÃ SỬA** — bỏ tham số; chữ ký nay là `(pts_start, text)` |
| `send_json` 3 lớp bọc nhau | ✅ **GIỮ NGUYÊN có chủ đích** + ghi rõ lý do vào docstring cả hai lớp. Mỗi lớp làm một việc: `connection` = tuần tự hoá + metric + chặn socket đóng; `SessionState` = API lớp phiên; `LookaheadSessionState` = thoát SỚM khi `_closed` (phiên Lookahead đóng TRƯỚC khi socket đóng). Gộp = sửa mọi call-site và mất ý nghĩa "phiên đã đóng" của Pipeline B |
| `_schedule_*_model_switch` cài đặt 2 lần | ✅ **GIỮ NGUYÊN có chủ đích** + ghi rõ khác biệt vào docstring cả 4 hàm. Bản Pipeline A GỬI `model_status(state="error")` ở mọi nhánh từ chối và phân biệt `downloading`/`loading`; bản Lookahead chỉ LOG rồi `return`, luôn gửi `loading`, ném lỗi TRONG task, và đồng bộ thêm `asr_engine.model_key/model_info`. Gộp = **đổi thông báo người dùng thấy ở một trong hai pipeline** |

**Chính sách hàng đợi TTS NGƯỢC NHAU — ✅ CỐ Ý, KHÔNG PHẢI LỖI.** Người dùng đã xác nhận lý do:

| | Pipeline A (`handler.py::_coalesce_enqueue`) | Pipeline B (`lookahead_handler.py::_trim_tts_queue`) |
|---|---|---|
| Khi hàng đợi đầy | **GỘP** vào câu mới nhất | **BỎ CÂU CŨ NHẤT** |
| Vì sao | A phải đợi chốt xong câu dịch mới phát được TTS ⇒ các câu DỒN LẠI. Không có mốc thời gian video để bám ⇒ không có khái niệm "không kịp". Vứt một câu = câu đó **vĩnh viễn** không có bản dịch/lồng tiếng (phụ đề gốc treo ở "…") ⇒ mất mát về ĐÚNG ĐẮN | B hiển thị bản dịch trong ĐÚNG khoảng nhân vật nói (`start_pts`→`end_pts`), nên TTS **phải phát kịp trong cửa sổ đó**. Không kịp thì phải HUỶ để nhường câu sau. Gộp hai câu càng sai: chuỗi gộp DÀI HƠN nên càng không thể phát kịp |

⇒ Đã ghi lý do này vào **docstring của CẢ HAI hàm**, kèm câu "ĐỪNG hợp nhất hai bên".

**⚠️ ĐÍNH CHÍNH.** Bản trước của mục này ghi "✅ ĐÃ XỬ LÝ (phần cơ học)" trong khi thực tế tôi
**mới chỉ thêm docstring**, chưa sửa mục nào. Người dùng đã phát hiện. Nay 2 mục đã sửa thật, 2 mục
còn lại được ghi rõ là giữ nguyên có chủ đích kèm bằng chứng khác biệt.

**Test hồi quy:** `backend/tests/test_74_pipeline_spawn_and_dedup.py` — 6 test, gồm chốt
`create_task` chỉ 1 chỗ (đã kiểm chứng guard **FAIL thật** khi chèn chỗ thứ hai), chữ ký
`_is_duplicate`, và hành vi của `spawn_background_task` (không có event loop ⇒ bỏ qua an toàn;
có loop ⇒ giữ task trong sink rồi nhả ra khi xong).

### 5.8 Payload `model_status` lặp 18 lần dạng dict literal
`session.py` 12 chỗ (`:251,272,280,290,297,303,326,348,356,366,373,379`) + `lookahead_handler.py` 6 chỗ (`:516,527,531,554,559,563`),
trong khi `serializers.make_model_status_msg` (`:173`) đã tồn tại nhưng **chỉ test dùng**.

### 5.9 Parse config lặp — ✅ ĐÃ SỬA

Trước: `min_words_to_commit` **5+ chỗ** · VAD `"0 = tắt" → None` **3 chỗ** · alias
`vad_threshold if not None else threshold` **3 chỗ** · clamp `lookaheadSyncOffsetMs` **2 chỗ** ·
`lookahead_handler.py:482-485` và `:2125-2127` **trùng nhau y hệt TRONG cùng một file**.

Nay mọi quy ước nằm ở **`backend/ws/session.py`** (cạnh `SessionConfigPayload`) và mọi nơi uỷ quyền:

| Helper | Thay cho |
|---|---|
| `pick_vad_threshold(parsed)` | 3 nhánh alias `vad_threshold`/`threshold` |
| `off_means_none_ms(raw)` | 3 chỗ `raw if raw > 0 else None` |
| `clamp_min_words(raw)` | 3 chỗ `max(0, int(...))` |
| `clamp_sync_offset_ms(raw)` | 2 khối `try/except` + `min(1500.0, …)` |
| `LookaheadSessionState._set_min_words_to_commit(parsed)` | 2 khối trùng trong cùng file |

Kiểm chứng: `min(1500`, `raw_ms if raw_ms > 0 else`, `vad_threshold if … else …threshold` đều còn
**0 lần** ngoài `session.py`; `_batch_sub_opts["min_words"]` chỉ còn **1 chỗ gán**.

**✅ ĐÃ LÀM NỐT (2026-10-08).**

1. **`main.py` không còn dựng `session_payload` thủ công.** Nay gọi
   `session_payload_from_rest(req)` — hàm này loại 4 trường REST-only, đổi
   `stability_duration_ms` → `stability_duration_sec`, rồi **validate qua
   `SessionConfigPayload`**. Kiểm chứng: `main.py` còn **0** dòng gán `session_payload[...]`.
   Khối 39 dòng viết tay → 1 dòng gọi hàm.

2. **`apply_init` không còn áp ngôn ngữ hai lần.** Trước đây hàm gán `source_lang`/`target_lang`
   từ raw `data`, rồi gán LẠI từ `parsed` (đọc CÙNG khoá), và đồng bộ ngôn ngữ xuống engine
   TRƯỚC khi có giá trị cuối cùng. Nay: parse một lần → chọn giá trị một lần → mới đồng bộ engine.
   `set_language` và `_parse_config` mỗi thứ đúng **1 lần** trong thân hàm.

> **⚠️ ĐÍNH CHÍNH về cách hiểu "parse payload 2 lần".** Câu cũ *"Pipeline B parse payload 2 lần
> (`apply_config` và `apply_init`)"* **gây hiểu nhầm**: `apply_init` phục vụ message
> `lookahead_init` (L2293) còn `apply_config` phục vụ `set_config` (L2352) — **hai loại message
> KHÁC NHAU**, không phải cùng một payload bị parse hai lần. Việc parse mỗi message một lần là
> ĐÚNG. Trùng lặp thật nằm ở **thân `apply_init`** (gán ngôn ngữ 2 lần) và đã sửa ở mục 2.

**Test hồi quy:** `test_75_session_payload_contract.py` (9 test — kiểm chứng **hành vi** của
`session_payload_from_rest`: mọi khoá đầu ra đều được payload hiểu, ms→giây đúng, không lẫn trường
REST-only, bỏ trường None, round-trip lại được; + chốt `main.py` không quay lại dict viết tay) và
`test_73` thêm 2 test cho `apply_init` (gán ngôn ngữ 1 lần; `apply_init` phải chạy TRƯỚC
`init_components` vì `init_components` đọc `self._parsed_config`).

> **⚠️ BUG TÌM THẤY KHI RÀ CHÍNH MỤC NÀY (đã vá).** Vì `SessionConfigPayload` khai
> `extra="ignore"`, mọi khoá `main.py` gửi mà payload KHÔNG hiểu đều bị **nuốt im lặng** — không
> lỗi, không log, không tác dụng. Và đã có đúng một khoá như vậy:
>
> | Đường | Gửi | `parsed.stability_duration_sec` |
> |---|---|---|
> | REST (`main.py:1151`) | `stability_duration_ms: 150.0` | **`None`** ← bị nuốt |
> | WS (`buildWsConfig`) | `stabilityDurationSec: 0.15` | `0.15` ✅ |
>
> Hệ quả: kéo slider "Stable for (ms)" thì `config.sentence` TOÀN CỤC vẫn được cập nhật, nhưng
> **phiên ĐANG CHẠY không nhận gì** qua đường REST. Không test nào bắt được vì đường WS gửi đúng
> `stabilityDurationSec` nên tính năng vẫn chạy — đây là loại lỗi chỉ lộ ra khi ĐỌC code.
>
> **Đã vá:** `main.py` nay gửi `stability_duration_sec = ms / 1000`.
> **Test hồi quy:** `backend/tests/test_75_session_payload_contract.py` — 6 test, trong đó có bất biến
> TỔNG QUÁT *"mọi khoá `main.py` gửi cho phiên phải là field name hoặc alias của
> `SessionConfigPayload`"*. Guard này bắt được **cả những khoá lệch trong tương lai**, không chỉ
> khoá này. Đã kiểm chứng nó **FAIL thật** khi quay lại code cũ.


### 5.10 Khác — ✅ ĐÃ SỬA (3/4)

- **`main.py::_log_runtime_status_at_startup()` cài lại `env_check.runtime_status()`** → ✅ hàm log nay
  đọc thẳng từ `env_check.runtime_status()`; chỉ giữ 2 kiểm tra mà `env_check` KHÔNG có (Diarization và
  cảnh báo `torch` lọt vào tiến trình). Lưu ý: `main.py::_vad_runtime_info()` vẫn `import onnxruntime`
  — đó là hàm KHÁC, hợp lệ, không phải trùng lặp.
- **`getSettings()` ghi mỗi giá trị dưới 2–3 khoá alias, và payload REST trộn hai quy ước** → ✅
  `getSettings()` nay chỉ ghi **một tên camelCase** cho mỗi giá trị; payload REST gửi snake_case nhưng
  **đọc từ camelCase** nhất quán. An toàn vì mọi phía đọc đều đã có nhánh fallback
  (`content-script.js::buildWsConfig`, các chỗ đọc `bs_settings`) — đã kiểm từng chỗ.
  Thêm `stabilityDurationMs` để payload REST khỏi nhân `stabilityDurationSec * 1000`
  (sai số dấu phẩy động: `0.05 * 1000 = 50.00000000000001`).
- **`containsMagic` vs `hasMagic` nội tuyến** (`buffer_interceptor_poc.js`) → ✅ `isInitSegment` nay
  gọi `containsMagic` thay vì tự viết lại vòng quét 4 byte.
- **Tách câu: 2 cài đặt ngữ nghĩa khác nhau** → **GIỮ NGUYÊN, có chủ đích**.
  `core/hypothesis.py::_split_sentences` (đơn giản, đếm nhịp preview) vs
  `asr/forced_aligner.py::_split_text_by_sentence` (đầy đủ: `3.14`, `domain.com`, `Dr.`, ngoặc đóng —
  dùng để ngắt phụ đề). Hai mục đích khác nhau, gộp lại sẽ làm bản đơn giản phức tạp lên mà không có lợi.

**Test hồi quy:** `backend/tests/test_73_config_coercion.py` — 24 test: hành vi của 4 helper + chốt
mã nguồn (quy ước §5.9 chỉ còn ở `session.py`; `min_words` chỉ gán 1 chỗ; hàm log khởi động không tự
import onnxruntime/gọi aligner; `getSettings()` không ghi alias snake_case; payload REST đọc camelCase).
Đã kiểm chứng guard **FAIL thật** khi chèn lại alias snake_case vào `getSettings()`, và pass lại sau
khi khôi phục.



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

### P3 — Giảm trùng lặp trong `popup.js` ✅ XONG (một phần, có chủ đích)

Ba cặp hàm gần trùng đã được hợp nhất, mỗi cặp nay còn **1 cài đặt dùng chung + 2 wrapper mỏng giữ
tên cũ** (tên cũ được `test_22` chốt bằng chuỗi, và vẫn đọc rõ nghĩa ở call-site):

| Cặp | Trước | Sau |
|---|---|---|
| `renderAsrEngineOptions` / `renderTranslationModelOptions` | 2 bản giống ~65 % | `renderModelOptions(selectEl, …)` dùng chung |
| `waitForAsrActivation` / `waitForTranslationActivation` | 2 bản giống ~83 % (vòng poll 46 dòng) | `waitForActivation(modelId, shortDesc, spec)` — khác biệt gói trong `spec.readState` |
| `monitorAsrDownload` / `monitorTranslationDownload` | 2 bản ~24 dòng gần y hệt | `monitorModelDownload(kind, …)` + bảng `ACTIVATION_TARGETS` (gom bất đối xứng ASR/dịch về một chỗ) |

**Số đo trung thực (không phải "~200 dòng" như ước lượng ban đầu):**

| Chỉ số | Trước | Sau | Chênh |
|---|---|---|---|
| Tổng dòng file | 1563 | 1541 | **−22** |
| Dòng CODE thuần (bỏ comment/dòng trắng) | 1334 | 1291 | **−43** |
| `note !== lastNote` (lõi vòng poll) | 2 bản | **1 bản** | −1 |
| `dl.model && dl.model !== modelId` | 2 bản | **1 bản** | −1 |
| `item.is_downloaded === false` | 2 bản | **1 bản** | −1 |
| `console.error("[Popup] Translation activation` | 1 bản | 0 (đã thành template) | — |

Con số "~200 dòng trùng lặp" trong §5.6 đếm **số dòng giống nhau giữa hai bản**, không phải số dòng
**xoá được**: gộp hai hàm 46 dòng giống 83 % chỉ tiết kiệm được phần *khác biệt*, và tôi còn **thêm**
docstring giải thích ⇒ tổng file gần như không đổi. Giá trị thật nằm ở chỗ khác: **hết nguy cơ sửa
một bên mà quên bên kia**.

**CỐ Ý KHÔNG GỘP `handleEngineSwitch` / `handleTranslationModelSwitch`.** Hai hàm chỉ giống ~33 %
(đo LCS 29/87 dòng): bản ASR dựng payload VAD/SEG đầy đủ, gửi `asr_engine`+`vad_engine`, rồi
**fetch lại** `/api/config` để lấy `result`, cập nhật `lastActiveAsr/Vad/Lang` và render lại dropdown
ngôn ngữ; bản dịch chỉ gửi `translation_model` và không fetch lại. Gộp chúng sẽ cần một "spec" 6–7
trường — tức là **abstraction giả**, khó đọc hơn hai hàm tường minh. Đây là cùng loại lỗi với
"chồng chéo chức năng", chỉ ngược chiều.

### P3-bis — Hợp nhất 3 bộ tải `urllib` ✅ XONG

Ba đường tải khác nhau về hình dạng, nay cùng đi qua hai helper dùng chung trong
`backend/utils/model_download.py`:

| | Trước | Sau |
|---|---|---|
| `utils/crispasr_native.py` | `urlopen` + hash gói + ghi zip tạm 157 MB + giải nén | `fetch_verified_bytes([BUNDLE_URL], …)` + giải nén **trong RAM** (`io.BytesIO`) — bỏ hẳn file zip tạm |
| `vad/silero_onnx.py` | `urlopen` trong vòng lặp nhiều URL + min-size + ghi `.onnx.part` | `fetch_verified_bytes(SILERO_ONNX_URLS, …, min_bytes=…)` + `write_bytes_atomic` |
| `vad/engines/firered_onnx.py` | `urlopen` + hash + ghi `.part` | `fetch_verified_bytes([url], …)` + `write_bytes_atomic` |

**Hai helper mới:**
- `fetch_verified_bytes(urls, expected_sha256, *, label, module_tag, timeout_sec, min_bytes)`
  — thử lần lượt các URL ứng viên, trả bản **đầu tiên khớp SHA-256**; trả `None` nếu hỏng hết
  (**không ném**, vì hai đường hiện có hai hợp đồng khác nhau: CrispASR ném, VAD trả `False`).
- `write_bytes_atomic(target, data)` — ghi qua `.part` rồi `replace()`, không để lại file dở dang.

**Kiểm chứng:** `urlopen` giờ chỉ xuất hiện **1 lần** trong toàn bộ backend (trong `model_download.py`).
Đã xoá kèm: `crispasr_native` bỏ import `urllib.*`, `silero_onnx` bỏ `urllib.*`,
`firered_onnx` bỏ `urllib.*` + hàm `_sha256_of_bytes` (nay mồ côi).

**Test hồi quy:** `backend/tests/test_70_shared_download_helper.py` — 6 test, gồm chốt mã nguồn
(`urlopen` phải đúng 1 chỗ, đã kiểm chứng guard FAIL khi chèn chỗ thứ hai) và 4 test HÀNH VI
(hash sai ⇒ từ chối; URL hỏng ⇒ thử URL dự phòng; `min_bytes` chặn file cụt; không để lại `.part`).

**Một va chạm test đáng ghi lại:** bản đầu tôi đặt `module_tag: str = "DL"` → `test_20` báo
"module_tag ngoài danh sách chuẩn". Sửa thành **bắt buộc** truyền `module_tag` thì `test_20` lại
báo "thiếu module_tag", vì bộ phân giải AST chỉ đọc được literal / hằng cấp module / **tham số có
default literal**. Kết luận: giữ tham số nhưng cho default hợp lệ (`"MAIN"`), và mọi caller trong
repo đều truyền tag riêng (`"ASR"` cho CrispASR, `"VAD"` cho 2 đường ONNX).

### P3-ter — Hợp nhất định dạng khung audio + gói `model_status` ✅ XONG

**1. Khung audio** `[4B header_len][JSON header][PCM]` từ **4 nơi → 1**:

| Nơi | Trước | Sau |
|---|---|---|
| `lib/frame-builder.js` | bản chuẩn | **vẫn là bản duy nhất** |
| `lib/lookahead-client.js` | tự dựng lại (4 dòng) | gọi `buildBinaryAudioPacket(hdrObj, rawBytes)` |
| `lib/ws-client.js` | fallback không thể chạy | đã xoá ở P2-ter |
| `background/service-worker.js` | fallback không thể chạy | đã xoá ở P2-ter |

Kiểm chứng: `setUint32(0,` (dấu hiệu ghi độ dài header) giờ chỉ còn ở `frame-builder.js:38`.

⚠️ **Một cái bẫy đã sập và được sửa:** `lookahead-client.js` nay phụ thuộc biến toàn cục
`buildBinaryAudioPacket`. Trình duyệt luôn có (manifest nạp `frame-builder.js` trước), nhưng
`buffer-interceptor.test.js` cũng `require("../lib/lookahead-client.js")` mà **không** nạp
`frame-builder.js` ⇒ test đó **TREO** (không phải fail — một test bị `◖` treo, che mất nguyên nhân)
thay vì báo lỗi rõ. Cách sửa: cho `lookahead-client.js` tự nạp phụ thuộc **khi chạy dưới Node**:

```js
if (typeof module !== "undefined" && module.exports && typeof buildBinaryAudioPacket !== "function") {
  require("./frame-builder.js");
}
```

Khối này không chạy trong trình duyệt (cùng pattern với `module.exports` đã có ở cuối file), và
giúp mọi test hiện/ sau này không phải tự nhớ thứ tự nạp của manifest.

**2. Gói `model_status`** từ **18 dict literal → 1 hàm**:
`serializers.make_model_status_msg(stage, state, model, message)` nay được dùng ở cả 12 chỗ trong
`session.py` và 6 chỗ trong `lookahead_handler.py` (trước đây hàm này **chỉ test dùng**). Cả hai file
đã import hàm; `session.py` bỏ hẳn 12 khối dict 4 dòng.

**Test hồi quy:** `backend/tests/test_71_shared_payload_builders.py` — 5 test, gồm:
- chốt `"type": "model_status"` không được xuất hiện ngoài `serializers.py` (đã kiểm chứng guard
  FAIL khi chèn literal giả vào `text_repetition.py`, và pass lại sau khi khôi phục);
- chốt `setUint32(0,` chỉ có trong `frame-builder.js`;
- kiểm chứng **hành vi thật** của `frame-builder.js` dưới Node (bố cục 4 byte LE + JSON + PCM khớp).

### CÒN LẠI — ✅ KHÔNG CÒN VIỆC NÀO TRONG §5 (đã kiểm chứng lại 2026-10-08)

> **Đính chính cùng ngày:** đúng sau khi mục này được viết, một lỗi **ngoài §5** lộ ra —
> test ghi đè báo cáo đã commit. Xem **§P2-quinquies**. Câu "không còn việc nào" chỉ đúng cho §5.

Toàn bộ §5 đã xử lý xong. Ba mục **cố ý KHÔNG gộp** (khác biệt là có chủ đích, đã ghi lý do vào
docstring từng chỗ):
- **Chính sách hàng đợi TTS** ngược nhau giữa Pipeline A (GỘP) và B (BỎ CŨ NHẤT) — xem §5.7.
- **`send_json` 3 lớp** và **`_schedule_*_model_switch` × 2** — xem §5.7.
- **`handleEngineSwitch` / `handleTranslationModelSwitch`** — chỉ giống ~33 %; gộp sẽ cần "spec"
  6–7 trường, tức abstraction giả (xem §P3).
- **Hai bộ tách câu** (`core/hypothesis` vs `asr/forced_aligner`) — phục vụ hai mục đích khác nhau
  (xem §5.10).

`main.py` vẫn còn một điểm nhỏ **không thuộc §5**: trường `lookahead_enabled` của
`SwitchModelRequest` không được dùng ở đâu trong đường phiên (chỉ có ở config toàn cục) — đã ghi
vào `_REST_ONLY_FIELDS` kèm lý do.

### P2-quater — Đấu dây 6 id mồ côi trong popup ✅ XONG

Phát hiện: **6/78 id** trong `popup.html` không được `popup.js` lẫn `popup.css` nhắc tới. Đây không
phải "markup chết" mà là **điều khiển ma** — hiện ra nhưng không làm gì:

| Id | Vấn đề thật | Cách xử lý |
|---|---|---|
| `stableToggleRow` | Bấm cả HÀNG không có gì xảy ra, trong khi `showOriginalToggleRow`/`ttsToggleRow` bấm được ⇒ cùng một UI, hai hành vi | Đấu dây qua `wireToggleRow` |
| `stableTraceToggleRow` | nt | nt |
| `translationOnceToggleRow` | nt | nt |
| `duckingSliderRow` | Slider "Original audio %" vẫn kéo được khi Auto-Ducking = Off ⇒ điều khiển vô tác dụng | Ẩn khi ducking Off (`updateDuckingUi`) |
| `lookaheadSyncGroup` | Slider đồng bộ vẫn chỉnh được khi chạy Pipeline A, nhưng `lookaheadSyncOffsetMs` **chỉ** được `ws/lookahead_handler.py` đọc | Thêm `pipelineBOnlyGroups`: làm mờ + ghi chú khi không phải Pipeline B |
| `lookaheadStatusDot` | **Không phải lỗi** — màu chấm do CSS quyết định qua `.lookahead-status.is-ready/is-low/is-off`. Thêm JS sẽ là LẶP logic | **Xoá id thừa** (giữ class) |

Đồng thời gộp 2 handler bấm-hàng trùng lặp thành một helper `wireToggleRow` (5 hàng dùng chung).

**Test hồi quy:** `backend/tests/test_69_popup_ids_wired.py` — bắt buộc mọi id phải được đấu dây (JS),
style (CSS), hoặc bị xoá. Đã kiểm chứng guard thật sự FAIL khi chèn một id mồ côi giả tạo.

### P2-bis — Hợp nhất chồng chéo ✅ XONG (phần cơ học)

| Việc | Kết quả |
|---|---|
| **SHA-256** | Từ **5 bản ở 4 module** → **1 bản duy nhất** ở `model_download.py`. `crispasr_native._sha256_file`, `silero_onnx._sha256`, `firered_onnx._sha256_of_bytes` nay uỷ quyền; `crispasr_native` bỏ luôn `hashlib` nội tuyến. Kiểm chứng: `hashlib.sha256` chỉ còn xuất hiện trong `model_download.py` |
| **Đọc RSS** | `ASREngine._process_rss_mb` từ bản sao ~35 dòng → uỷ quyền `mem_guard.rss_mb`. **Giữ method** vì `test_17_executor_backpressure.py` monkeypatch chính attribute này làm seam kiểm thử |

### P2-ter — Dọn giao thức chết ở extension ✅ XONG

Xoá: `partial_transcript`, `sentence_complete`, `set_overlay_mode` (chuỗi 3 file:
`content-script.js` → `overlay-manager.js` → `subtitle-renderer.js`, gồm cả `SubtitleRenderer.setMode`
và thuộc tính `this.mode` chỉ-ghi), `ws_json_raw` (thay bằng `console.warn` để lỗi parse vẫn thấy được),
alias `SEND_BUFFER`, emit `reconnecting` (**đã khôi phục — xem bài học 4**), và theo quyết định của người
dùng: khối HTML comment `#lblActiveModel` + `popup.js:70` + 3 nhánh `if` + 5 class CSS mồ côi
(`.model-info-*`, `.toggle-icon`).

**GIỮ LẠI có chủ đích:** `stream_reset` (backend gửi, extension không nhận) — `test_21_seek_reset.py:143`
khẳng định đây là hành vi CÓ CHỦ ĐÍCH. Client bỏ qua một thông báo thông tin không biến nó thành code chết.

### P2-quinquies — Test ghi đè báo cáo ĐÃ COMMIT ✅ XONG

**Phát hiện 2026-10-08, SAU khi mục "CÒN LẠI" ở trên được viết.** Đây không phải code chết mà là
một lỗi vệ sinh repo, lộ ra qua hậu quả: nó suýt làm bẩn một commit.

`conftest.report_dir` trả thẳng `report/` — cây **đã được git track**. Trong 6 test ghi báo cáo
benchmark, `test_04_commit_logic` là test **duy nhất không có marker `slow`**:

| Test | Marker | Chạy trong `pytest` mặc định? | Ghi vào |
|---|---|---|---|
| `test_01_core_audio` | `slow` | không | `report/01_core_audio/report.md` |
| `test_02_vad_benchmark` | `slow` | không | `report/02_vad/report.md` |
| `test_03_asr_benchmark` | `slow` | không | `report/03_asr/report.md` |
| **`test_04_commit_logic`** | **(không có)** | **CÓ** | **`report/04_commit_logic/report.md`** |
| `test_05_translation_benchmark` | `slow` | không | `report/05_translation/report.md` |
| `test_43_vad_ava_speech` | `slow` | không | `report/02_vad/ava_speech_report.md` |

Báo cáo chứa dòng `Thời gian thực hiện`, nên **mỗi lần chạy `pytest` mặc định là file đổi nội dung**.
Hậu quả thật đã xảy ra: `report/04_commit_logic/report.md` lọt vào commit và phải `git commit --amend`
để loại ra.

`test_06_tts_benchmark` / `test_07_e2e_comparison` **không** thuộc nhóm này: chúng ghi báo cáo trong
`main()`, không hàm `test_*` nào ghi — pytest không bao giờ thu thập chúng. Ghi thẳng vào `report/`
khi chạy tay `python -m backend.tests.test_06…` là **đúng**, giữ nguyên.

**Cách sửa:** `report_dir` mặc định trỏ vào thư mục TẠM; muốn cập nhật báo cáo đã commit phải bật cờ
tường minh `WRITE_BENCH_REPORT=1`. Logic tách ra `backend/tests/bench_report_paths.py` vì
`backend/tests/` không phải package (không có `__init__.py`) — import `conftest` từ test sẽ chạy nó
**lần thứ hai**, lặp lại toàn bộ thiết lập DLL/model.

**Hai lớp guard, cả hai đã được chứng minh là THẬT SỰ đổ khi tái tạo bug:**

1. `backend/tests/test_76_bench_report_isolation.py` — chốt hàm phân giải đường dẫn (cả hai nhánh),
   và chốt `test_04` phải lấy đường dẫn từ fixture chứ không tự dựng hằng số trỏ vào `report/`.
2. Fixture session autouse `_guard_real_report_tree` trong `conftest.py` — băm **toàn bộ** cây
   `report/` trước và sau phiên test, đổ nếu có gì thay đổi. Nó không quan tâm test ghi bằng cách
   nào hay dùng fixture gì, nên bắt được cả những đường ghi trong tương lai mà `report_dir` không
   kiểm soát.

**Kiểm chứng ngược (falsification):** tạm sửa `resolve_report_dir` trả về cây thật ⇒ guard đổ đúng
`AssertionError: … sửa=['04_commit_logic\\report.md']`; khôi phục ⇒ xanh lại. Nhánh opt-in cũng đã
kiểm: bật `WRITE_BENCH_REPORT=1` thì guard **bỏ qua** và báo cáo được ghi vào `report/` thật như chủ ý.

**Một false positive đã gặp và đã sửa (đáng ghi lại).** Bản đầu của guard băm **cả cây** `report/`.
Nó báo động sai ngay ở lần chạy đầy đủ đầu tiên — vì chính người viết đang sửa
`report/audit/26_…md` trong lúc suite chạy. Sửa tài liệu phân tích là việc bình thường, không được
làm đỏ bộ test. Guard nay chỉ canh **7 thư mục báo cáo benchmark** (`01_core_audio` … `07_final_e2e_comparison`),
đúng tập mà test có thể ghi ra; tài liệu do người viết nằm ngoài phạm vi. Đã kiểm chứng lại: sửa
`report/audit/**` giữa lúc suite chạy **không** còn gây đỏ, còn tái tạo bug cũ thì guard vẫn đổ.

### Năm bài học đo được (đã trả giá để biết)

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
4. **"Không có listener/consumer" KHÔNG đồng nghĩa với chết — với API event-emitter thì càng sai.**
   Tôi xoá `WSClient._emit("reconnecting", …)` với lý do "không ai nghe trong `extension_src`".
   `backend/tests/js/ws_reconnect_on_port_death_test.js` khẳng định sự kiện này là **hợp đồng** và đã
   đỏ ngay. Đã khôi phục kèm comment cảnh báo. Cùng lớp lỗi với `supportsGainDucking` (bài học 2):
   **API công khai tồn tại để bên ngoài dùng, kể cả khi bên ngoài đó là test.**
   → Quy tắc rút ra: trước khi xoá bất kỳ symbol nào của extension, bắt buộc grep **cả `backend/tests/`**.
5. **Test không được ghi vào artifact đã commit.** `test_04_commit_logic` ghi đè
   `report/04_commit_logic/report.md` trong mọi lần `pytest` mặc định, chỉ vì nó thiếu marker `slow`
   mà 5 test anh em đều có. Không test nào đỏ, không ai báo — nó chỉ lộ ra khi `git status` bẩn và
   một báo cáo đo lường suýt vào commit. ⇒ Báo cáo benchmark phải ghi vào thư mục tạm theo mặc định,
   và chỉ ghi vào cây thật khi có cờ tường minh (xem §P2-quinquies).

