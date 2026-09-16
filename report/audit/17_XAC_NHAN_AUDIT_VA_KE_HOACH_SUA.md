# XÁC NHẬN 3 BÁO CÁO AUDIT & KẾ HOẠCH SỬA

**Ngày kiểm chứng:** 2026-09-16
**Đối tượng kiểm chứng:** `BAO_CAO_AUDIT_HIEU_NANG_CHATGPT.md`, `BAO_CAO_AUDIT_HIEU_NANG_GROK.md`, `BAO_CAO_AUDIT_HIEU_NANG_QWEN.md`
**Nguyên tắc:** mỗi phát hiện chỉ được "xác nhận" khi đọc được đúng code hiện tại trong repo (file + dòng) hoặc chạy được test. Phát hiện không tái hiện được ⇒ ghi rõ là **bác bỏ / lỗi thời**, không đưa vào kế hoạch sửa.
**Ràng buộc bất biến:** mọi thay đổi phải **không giảm chất lượng** (không đổi model, precision, cửa sổ ASR, số bước TTS, chất lượng bản dịch).

---

## 0. Cách kiểm chứng

| Bước | Lệnh / cách làm | Kết quả |
|---|---|---|
| Đọc 3 báo cáo | đọc toàn văn | xong |
| Đối chiếu code | `read`/`grep` trực tiếp trên `backend/**`, `extension_firefox/**`, `external/transcribe.cpp/src/**` | xong |
| Chạy test suite | `python -m pytest backend/tests --tb=short` | **5 failed / 221 passed / 226 collected** |
| Kiểm tra native | đọc `external/transcribe.cpp/src/arch/qwen3_asr/model.cpp` | xong |
| Kiểm tra phụ thuộc | `python -c "find_spec('llama_cpp')"`, `find_spec('transcribe_cpp')` | cả hai **đã cài** |

> Lưu ý môi trường: `pytest` cần ghi `tmp_path` ra thư mục temp ngoài workspace; trong sandbox mặc định 19 test của `test_22` bị `PermissionError` ở bước setup (lỗi hạ tầng, không phải lỗi sản phẩm). Số liệu 5 failed ở trên là khi chạy với quyền đầy đủ.

---

## 1. Kết luận nhanh

| Báo cáo | Đúng & cần sửa | Đúng nhưng là design/upstream | Sai / lỗi thời / nguy hiểm |
|---|---|---|---|
| **ChatGPT** | 9 phát hiện | 5 | 3 (test-failure không tái hiện, "native không có trong ZIP") |
| **Grok** | 2 (RC-B, RC-E) | 5 (scale/dual-GPU/native) | 3 (AudioWorklet "dead", hardcode n_ctx, queue drop "cần giám sát") |
| **QWEN** | ~0 bug riêng (2 phát hiện trùng ChatGPT) | 1 (F-39 native RSS) | **6 (O(N²), 90 MB copy, executor vô hạn, base64 hot path, tách instance lock, reuse preview)** |

**Phát hiện nghiêm trọng nhất và là bug thật duy nhất ở mức P0:** backpressure HARD phía extension **không có đường tự phục hồi** ⇒ capture có thể treo vĩnh viễn. Đây là kết luận trùng khớp giữa ChatGPT (§5) và code hiện tại, và tôi đã xác minh được toàn bộ chuỗi nhân–quả.

QWEN là báo cáo **lỗi thời nhất**: nó mô tả trạng thái code trước các bản vá P2.3/P4.5/F-39 (số dòng không khớp, ví dụ `engine.py:353-376` hiện là code nạp model, không phải vòng preview). Nhiều khuyến nghị P0 của QWEN **đã được implement từ trước** hoặc **sẽ làm giảm chất lượng nếu làm theo**.

---

## 2. Bảng xác nhận từng phát hiện

### 2.1 ChatGPT

| # | Phát hiện | Kết luận | Bằng chứng |
|---|---|---|---|
| 1 | P0 — ASR preview re-infer toàn bộ cửa sổ (bounded 6 s) | **XÁC NHẬN** (hiệu năng, không phải correctness) | `engine.py:940-949` (`_preview_window_start`), `config.py:126` (`preview_window_sec=6.0`), `engine.py:1218-1224` |
| 2 | P0 — Backpressure HARD không có resume chủ động | **XÁC NHẬN — BUG THẬT P0** | `service-worker.js:61-71` vs `:81-84`; `FLUSH_PENDING` (`:169-177`) **không có producer**; `content-script.js:316-328`; `audio-capture.js:144` |
| 3 | P1 — Chuỗi copy audio browser→bridge→NumPy→ring→slice | **XÁC NHẬN** | `ws-client.js` `.slice()`, `service-worker.js:154-166` (packet mới), `audio_buffer.py:92-99`, `:130-141` |
| 4 | P1 — Singleton + global infer lock | **XÁC NHẬN nhưng là design** (README: 1 session) | `engine.py:66-80`, `translation/engine.py:42-47`, `tts/engine.py:67` |
| 5 | P1 — `_TRANS_EXECUTOR(max_workers=1)` + `_infer_lock` giữ suốt generation | **XÁC NHẬN** | `translation/engine.py:35`, `:349-368` |
| 6 | P1 — WS single send lock ⇒ head-of-line blocking | **XÁC NHẬN (rủi ro)** | `connection.py:36`, `:63`, `:81` |
| 7 | P1 — `MetricsCollector` global mutex trong hot path | **XÁC NHẬN** | `metrics.py:27,44,51,60,65,73…`; gọi theo từng item ở `handler.py:544,649` |
| 8 | P1 — `translation.load_model()` giữ `_shared_lock` suốt lúc build | **XÁC NHẬN** | `translation/engine.py:152-163` (build **trong** lock) vs `:193-203` reconfigure (build **ngoài** lock) |
| 9 | P1 — Drop final translation khi queue đầy | **XÁC NHẬN — mất chất lượng** | `handler.py:314-326`; queue chỉ chứa **final** (`if is_final`), `maxsize=4` (`session.py:193`) |
| 10 | P1 — TTS GPU→CPU→NumPy→stretch→normalize→WAV nhiều bản copy | **XÁC NHẬN** | `tts/engine.py:253-261`, `audio_processor.py:40-54`, `:86-108`, `:119-128` |
| 11 | P1 — Binary TTS vẫn thêm 1 full-frame copy; browser TTS 2 copy | **XÁC NHẬN** | `serializers.py:127+`; `content-script.js:299` + `tts-player.js:135` (`arrayBuffer.slice(0)`) |
| 12 | P2 — `AudioBuffer.clear()` zero-fill 3.84 MB mỗi reset | **XÁC NHẬN** | `audio_buffer.py:174-179` |
| 13 | P2 — `write_bytes(int16)` tạo float32 rồi ring copy lại | **XÁC NHẬN** | `audio_buffer.py:94-99` → `write()` `:56-78` |
| 14 | P2 — Phase vocoder CPU khi speed ≠ 1 | **XÁC NHẬN** (dormant khi speed=1) | `tts/engine.py:258-261`, `audio_processor.py:57-108` |
| 15 | P2 — `qsize()`+metrics mỗi item | **XÁC NHẬN** | `handler.py:74-80`, `:544`, `:649` |
| 16 | P2 — `max_payload_bytes` không có consumer | **XÁC NHẬN (dead config)** | chỉ xuất hiện ở `config.py:39` |
| 17 | P2 — `capacity_sec` không phải source of truth (hardcode 60.0) | **XÁC NHẬN** | `engine.py:230` vs `config.py:258` |
| 18 | P2 — `cancel_inference()` đọc `_shared_session` không lock | **XÁC NHẬN (rủi ro thấp)** | `engine.py:1403-1407` |
| 19 | P2 — `drain_queues()` chạy sau khi worker đã bị cancel | **XÁC NHẬN** | `handler.py:167-176` |
| 20 | P2 — Executor mặc định dùng chung cho TTS + prewarm | **XÁC NHẬN** | `tts/engine.py:209,312,321` (`asyncio.to_thread`) |
| 21 | P2 — Prewarm TTS lặp khi config đổi | **XÁC NHẬN (nhẹ)** | `main.py:569,654`, `handler.py:215`; `tts/engine.py:206-210` |
| 22 | P2 — ScriptProcessor fallback + watchdog 2.5 s | **XÁC NHẬN** | `audio-capture.js:168-177`, `:213+` |
| 23 | P2 — Chunk 1024 mẫu ≈ 64 ms | **XÁC NHẬN** | `audio-processor.js:18` |
| 24 | §2.2 — 6 test failure | **CHỈ 3/6 TÁI HIỆN** (xem §2.4) | #1 wav fixture **có đủ**, #2 cuda test **pass**, #5 nguyên nhân khác |
| 25 | §2.3 — "native source không có trong ZIP" | **SAI ở repo hiện tại** | `external/transcribe.cpp/src/**` đầy đủ (kể cả `arch/qwen3_asr/model.cpp`) |
| 26 | §40 — thiếu metric recompute ratio | **XÁC NHẬN** | chỉ có `asr.preview_audio_sec` (`engine.py:1233`), không có `unique`/`ratio` |

### 2.2 Grok

| # | Phát hiện | Kết luận | Bằng chứng |
|---|---|---|---|
| 1 | RC-A / C-01 — Singleton + global lock ⇒ 1 session | **XÁC NHẬN (design)** | như ChatGPT #4 |
| 2 | RC-B / C-02 — mỗi inference ASR = full offline run trên slice, không incremental | **XÁC NHẬN** | `engine.py:1218-1224`; `model.cpp:768-775` (KV wipe mỗi run) |
| 3 | RC-C / C-03 — dual GPU API (Vulkan ASR + CUDA MT/TTS) | **XÁC NHẬN (quan sát kiến trúc)** | `config.py:108` `backend="auto"`; `translation/engine.py:100-110` dùng CUDA |
| 4 | C-04 — `get_slice` luôn `np.empty`; int16↔float32 nhiều tầng | **XÁC NHẬN** | `audio_buffer.py:130`, `:92-99` |
| 5 | RC-D / C-07 — "AudioWorklet vẫn dead, capture vẫn ScriptProcessor-heavy" | **BÁC BỎ** | `audio-capture.js:122-187`: AudioWorklet là đường **chính**, ScriptProcessor chỉ là fallback khi lỗi/timeout |
| 6 | C-08 — policy drop queue cần giám sát | **XÁC NHẬN — và tệ hơn: drop final** | như ChatGPT #9 |
| 7 | C-10 — `n_ctx/n_batch/n_threads` translation còn hardcode | **BÁC BỎ** | đọc từ config: `translation/engine.py:100-108`, `config.py:225-227` |
| 8 | RC-E / C-06 — native không reuse state, KV wipe mỗi run | **XÁC NHẬN** (thuộc upstream vendored) | `external/transcribe.cpp/src/arch/qwen3_asr/model.cpp:768-775` |
| 9 | C-13 — payload JSON lặp field | **XÁC NHẬN (thấp)** | `serializers.py` full-prefix partial |
| 10 | §6 P0.2 — "expose 100% n_ctx/n_batch/n_threads" | **ĐÃ XONG** | như #7 |

### 2.3 QWEN

| # | Phát hiện | Kết luận | Bằng chứng |
|---|---|---|---|
| 1 | F-01 — ASR preview **O(N²)**, `engine.py:353-376`, không có window | **BÁC BỎ (lỗi thời)** | `engine.py:940-949` + `config.py:126`: preview **đã bị cửa sổ hoá 6 s**. Dòng 353-376 hiện là `transcribe_cpp.Model(...)`. Độ phức tạp thực: `O(số_preview × cửa_sổ)`, bị chặn trên |
| 2 | F-01b — "câu 30 s ⇒ ~90 MB copy" | **BÁC BỎ** | commit slice bị clamp còn `max_duration×1.5 = 9 s` (`engine.py:1011-1022`) |
| 3 | F-04 — `_EXECUTOR` hàng đợi **không giới hạn** ⇒ RAM 66–85 MB/s | **PHẦN LỚN ĐÃ MITIGATE** | guard `max_inflight_infer=1` (`engine.py:1208-1215`, `config.py:147`); có test riêng `test_17_executor_backpressure.py`. Hàng đợi executor về bản chất vẫn unbounded ⇒ rủi ro tồn dư **chỉ khi** nâng `max_inflight_infer` |
| 4 | P0.1 — "thêm sliding window cho preview" | **ĐÃ CÓ** (sẽ là no-op) | như #1 |
| 5 | P0.2 — "reuse preview cho commit, giảm 30–50 %, không ảnh hưởng chất lượng" | **NGUY HIỂM — KHÔNG LÀM** | `preview_reuse_for_commit=False` mặc định (`config.py:138`); có harness WER A/B (`test_09_wer_ab.py`, `report/audit/06_wer_ab.json`, `15_wer_ab_repeats.json`) vì reuse từng gây hồi quy WER |
| 6 | F-03 / P1.5 — "tách `_infer_lock` thành instance attribute để cho phép song song" | **NGUY HIỂM — KHÔNG LÀM** | module docstring `engine.py:20-22`: thư viện native chỉ cho **1 `run()` in-flight / session**. Tách lock ⇒ 2 `run()` đồng thời trên cùng `_shared_session` ⇒ use-after-free/crash |
| 7 | "TTS base64 trên hot path +33 %" | **SAI** | binary là mặc định cho protocol ≥ 2 (`handler.py:582-587`); base64 chỉ cho client cũ protocol 1 (`serializers.py:192`, `test_13:269-300`) |
| 8 | F-39 — +73 GB private RAM trong inference native | **XÁC NHẬN (đã có mitigation)** | `report/audit/05_measurements_and_status.md:746`; guard `engine.py:791-877` |
| 9 | "materialize timestamps" | **ĐÃ XONG** | `engine.py:658-660` (`timestamps="none"` mặc định) |
| 10 | P1.4 "bắt buộc compact payload" | **ĐÃ XONG** | `serializers.py`, `test_18_compact_payload.py` |
| 11 | P2.8 "loại bỏ abstraction thừa `return self.engine.run()`" | **KHÔNG TÌM THẤY** | các wrapper đều thêm validation/metrics |

### 2.4 Kiểm chứng 6 test failure mà ChatGPT liệt kê

Chạy thật: **5 failed / 221 passed / 226 collected**. Đối chiếu:

| ChatGPT liệt kê | Thực tế |
|---|---|
| 1. `test_bit_exact_audio_integrity` thiếu WAV fixture | **KHÔNG tái hiện** — `wav_test/` có đủ 9 `.wav`; test PASS |
| 2. `test_cuda_warning_helper_never_raises` fail | **KHÔNG tái hiện** — test PASS |
| 3. `test_partial_write_is_not_retried` phụ thuộc ngày | **ĐÚNG** — hardcode `"2026-09-15"` (`test_19:90`), runtime `2026-09-16` |
| 4. 3 logger call thiếu `module_tag` trong `hotswap.py` | **HIỂU SAI BẢN CHẤT** — `hotswap.py:151,180,187` **có** `extra={"module_tag": _TAG}`; lỗi là test AST chỉ nhận **literal**, không resolve hằng `_TAG` (`test_20:76-82`) |
| 5. `test_reconfigure_failure_keeps_working_model` thiếu `llama-cpp-python` | **FAIL NHƯNG NGUYÊN NHÂN KHÁC** — `llama_cpp` **đã cài**; file `Hy-MT2-1.8B-UD-Q8_K_XL.gguf` **có thật trên đĩa** nên model thật được nạp (log: "Nạp thành công … n_ctx=512"), `FileNotFoundError` không bao giờ được raise |
| 6. `test_rest_busy_with_another_model_returns_409` | **ĐÚNG (test sai, không phải code sai)** — monkeypatch `is_busy=lambda key=None: True` làm điều kiện key-aware ở `main.py:523` thành `False` ⇒ trả 202 |
| — | **+ 1 failure ChatGPT bỏ sót:** `test_hotswap_disabled_download_raises_before_touching_model` — thiếu fixture `allow_real_translation_methods` ⇒ guard autouse `conftest.py:170-204` bắn AssertionError thay vì `FileNotFoundError` |

---

## 3. DANH SÁCH BUG ĐÃ XÁC NHẬN

| ID | Mức | Bug | Vị trí | Ảnh hưởng |
|---|---|---|---|---|
| **FIX-01** | **P0** | Backpressure HARD không có đường tự phục hồi ⇒ capture treo vĩnh viễn | `service-worker.js:61-71`, `:169-177`; `content-script.js:316-328` | Mất tiếng hoàn toàn cho tới khi user Stop/Start lại |
| **FIX-02** | P1 | Drop bản dịch **final** khi `translation_queue` đầy | `handler.py:314-326`, `session.py:193` | Phụ đề gốc hiện nhưng **không bao giờ có bản dịch** |
| **FIX-03** | P1 | Drop TTS **final** khi `tts_queue` đầy | `handler.py:518-530` | Mất lồng tiếng cho câu đã chốt |
| **FIX-04** | P1 | `load_model()` build LLM bên trong `_shared_lock` | `translation/engine.py:152-163` | Chặn `get_instance`/`unload`/đọc `_shared_llm` vài chục giây |
| **FIX-05** | P1 | Bộ test có 5 failure che mất regression thật | `test_19:90`, `test_20:76-82`, `test_22:194,208-227,336-346` | Không phân biệt được lỗi mới |
| **FIX-06** | P2 | `AudioBuffer.clear()` zero-fill 3.84 MB mỗi reset | `audio_buffer.py:174-179` | CPU/cache traffic vô ích khi seek |
| **FIX-07** | P2 | `capacity_sec` không phải source of truth | `engine.py:230` vs `config.py:258` | Config không có tác dụng |
| **FIX-08** | P2 | WS single `_send_lock` cho cả JSON phụ đề và binary TTS | `connection.py:36,63,81` | Head-of-line blocking khi TTS frame lớn |
| **FIX-09** | P2 | Metric queue gauge ghi **mỗi item** + global mutex | `metrics.py:44+`, `handler.py:74-80,544,649` | Lock contention hot path |
| **FIX-10** | P2 | Preview ASR tính lại cửa sổ chồng lấn, không đo được mức lãng phí | `engine.py:1218-1224,1233` | GPU/CPU lãng phí; không có metric để quyết định |
| **FIX-11** | P3 | Browser TTS copy buffer 2 lần | `content-script.js:299`, `tts-player.js:135` | GC/heap |
| **FIX-12** | P3 | `cancel_inference()` đọc `_shared_session` không lock | `engine.py:1403` | Race nhỏ khi hot-swap |
| **FIX-13** | P3 | `drain_queues()` sau khi đã cancel worker | `handler.py:167-176` | Cleanup vô nghĩa + chờ timeout |
| **FIX-14** | P3 | `max_payload_bytes` là dead config | `config.py:39` | Nhiễu cấu hình |

---

## 4. KẾ HOẠCH SỬA

Thứ tự ưu tiên: **đúng đắn/ổn định trước → hồi quy test → copy/RAM → concurrency → ASR compute (đo rồi mới sửa)**.

### Giai đoạn 0 — Dọn test để nhìn thấy hồi quy (FIX-05)  ⏱ ~1 h

Điều kiện tiên quyết: không sửa gì khác trước khi `pytest` xanh.

| Việc | File | Cách sửa |
|---|---|---|
| 0.1 | `backend/tests/test_19_log_no_duplicate.py:90` | Bỏ hardcode `"2026-09-15"`. So sánh theo dạng bằng regex `^\d{4}-\d{2}-\d{2}$` (mục đích test là "không ghi lại sau lỗi giữa chừng", không phải kiểm tra ngày) |
| 0.2 | `backend/tests/test_20_logging_convention.py:76-82` | Cho `_module_tag()` resolve hằng chuỗi module-level: thu thập `ast.Assign` có `value` là `ast.Constant` cấp module trước, rồi tra tên. **Đây là sửa test, không phải hạ chuẩn quy ước** — tag vẫn được enforce |
| 0.3 | `backend/tests/test_22_..._download.py:194` | Thêm fixture `allow_real_translation_methods` (đã có sẵn ở `conftest.py:207-219`) để `reconfigure` thật chạy và raise `FileNotFoundError` đúng như test kỳ vọng |
| 0.4 | `backend/tests/test_22_..._download.py:208-227` | Test đang phụ thuộc "file GGUF không tồn tại trên máy". Phải **ép** điều kiện: `monkeypatch.setattr(registry, "resolve_gguf_path", lambda key: str(tmp_path/"missing.gguf"))` và `monkeypatch.setattr(engine, "ensure_model_file", boom)`. Bổ sung assert chống nạp model thật |
| 0.5 | `backend/tests/test_22_..._download.py:336-346` | Bỏ monkeypatch `is_busy=lambda key=None: True`; chỉ cần `hotswap._set_state(state="downloading", model="xiaomi")` là đủ để `is_busy()` True và `is_busy("tencent-1.8b")` False ⇒ 409 được raise đúng. Đồng thời thêm 1 test khẳng định nhánh ngược lại: cùng model ⇒ **không** 409 |

**Nghiệm thu G0:** `python -m pytest backend/tests -q` ⇒ `0 failed`.

---

### Giai đoạn 1 — P0: Backpressure tự phục hồi (FIX-01)  ⏱ ~2–3 h

**Nguyên nhân gốc:** `pausedByBackpressure` chỉ được xoá bên trong `sendOrQueue()`, mà `sendOrQueue()` chỉ được gọi từ `SEND_BINARY`. Khi `content-script.js:320` đặt `audioCapture.isCapturing = false`, `audio-capture.js:144` ngừng phát chunk ⇒ không còn `SEND_BINARY` ⇒ cờ pause **không bao giờ** được xoá. `FLUSH_PENDING` có implement nhưng **không có caller** (đã grep toàn extension: chỉ xuất hiện ở chính handler của nó).

**Thiết kế sửa (2 lớp, có hysteresis):**

1. **Tách state machine thành module thuần để test được** — tạo `extension_firefox/lib/backpressure-gate.js`:
   - Hàm thuần, **không** phụ thuộc `WebSocket`/`browser`: nhận `bufferedAmount`, `softLimit`, `hardLimit`; trả `"send" | "drop" | "pause" | "resume"`.
   - Hysteresis: vào PAUSE ở `>= HARD`, chỉ ra khỏi PAUSE khi `< SOFT`.
   - Đếm `pauseCount`, `resumeCount`, `droppedFrames`, `pausedMs`.
2. **`service-worker.js`**: khi chuyển sang PAUSE, khởi động `setInterval` 100 ms kiểm tra `ws.bufferedAmount < SOFT`:
   - nếu thoả ⇒ xoá cờ pause, `notifyBackpressure("ok")`, flush `pendingAudio` (chính là `FLUSH_PENDING` gọi nội bộ), dừng interval;
   - `clearInterval` trong `cleanup()` **và** trong nhánh `CONNECT` (tránh timer zombie khi reconnect).
3. **`content-script.js:316-328`**: thêm timer "liveness" 250 ms **chỉ khi** `bp.state === "paused"`, gửi `SEND_JSON`/`FLUSH_PENDING` qua bridge. Lý do: MV3 service worker có thể bị tạm dừng; nếu chỉ dựa vào `setInterval` trong SW thì nhánh resume có thể mất. Timer này dừng ngay khi nhận `"ok"`.
4. **Quan sát**: ghi `backpressure_pause_count`, `backpressure_resume_count`, `audio_frames_dropped`, `paused_ms` (đúng như ChatGPT §5.1 khuyến nghị).

**Test:** tạo `extension_firefox/tests/backpressure-gate.test.mjs`, chạy `node --test`.
- `buffered >= HARD` ⇒ `pause`; `buffered` giảm xuống giữa SOFT và HARD ⇒ **vẫn** `pause` (kiểm tra hysteresis, chống rung pause/resume);
- `buffered < SOFT` ⇒ `resume`;
- chuỗi `pause → (drain) → resume` phải phát đúng 1 lần mỗi trạng thái;
- `drop` khi `SOFT <= buffered < HARD` giữ frame **mới nhất**.

**Nghiệm thu G1:** test node xanh + kiểm thử thủ công theo kịch bản F (§7).

---

### Giai đoạn 2 — P1: Không mất bản dịch/TTS final (FIX-02, FIX-03)  ⏱ ~2 h

Triết lý sẵn có của repo là **gộp thay vì vứt** (`_pending_commits`, `_MAX_PENDING_COMMITS`) — áp dụng đúng như vậy:

| Việc | File | Cách sửa |
|---|---|---|
| 2.1 | `handler.py:314-326` | Khi `QueueFull`: **không drop**. Pop phần tử cũ nhất, **gộp** `text` vào phần tử đó (`{"text": old + " " + new, "utterance_id": merged}`, `_queued_at` giữ nguyên), rồi `put_nowait` lại. Thêm counter `queue.translation_merged`. Chỉ khi gộp thất bại mới đếm `queue.translation_dropped` |
| 2.2 | `handler.py:518-530` | Tương tự cho `tts_queue`: gộp text của 2 câu final liền kề |
| 2.3 | `ws/session.py:193-194` | Giữ `maxsize=4` (RAM bound vẫn giữ nguyên, không nới) |
| 2.4 | metric | Thêm `translation.final_merged`, `translation.final_dropped` (mục tiêu: `final_dropped == 0`) |

**Test mới** (`test_13_translation_and_protocol.py`): bơm 6 final liên tiếp vào queue `maxsize=4` với worker chậm ⇒ khẳng định (a) không item nào bị mất chữ, (b) counter `final_dropped == 0`, (c) `merged >= 1`.

---

### Giai đoạn 3 — P1: Lock hygiene (FIX-04, FIX-12)  ⏱ ~2 h

| Việc | File | Cách sửa |
|---|---|---|
| 3.1 | `translation/engine.py:146-165` | Áp dụng **đúng pattern đã có ở `reconfigure()`**: đọc/kiểm tra trong lock → nhả lock → `_build_llm()` **ngoài** lock → lấy lock, double-check (`_shared_model_key` có đổi không) → swap → nhả lock → `_release_llm(old)` |
| 3.2 | `engine.py:1403-1407` | `with cls._shared_lock: session = cls._shared_session` rồi mới `session.cancel()` trên tham chiếu cục bộ. **Không giữ lock trong lời gọi native** |
| 3.3 | `handler.py:167-176` | Đổi thứ tự cleanup: `stop intake → clear queue + task_done() → cancel workers → await gather → cleanup()`. Xoá `drain_queues(timeout=0.15)` khỏi `finally` (giữ hàm cho đường flush chủ động) |
| 3.4 | tài liệu | Ghi rõ thứ tự lock `_infer_lock → _shared_lock` (ASR) và `_switch_lock → _shared_lock` (Translation) vào docstring, kèm test khẳng định |

**Test:** mở rộng `test_12_locks_and_metrics.py`: (a) gọi `load_model()` đồng thời từ 2 thread ⇒ chỉ build 1 lần; (b) trong lúc `load_model()` đang build, `get_instance()` phải trả về **ngay** (< 100 ms); (c) sau cancel, không còn item treo trong queue.

---

### Giai đoạn 4 — P2: Copy / RAM / config (FIX-06, FIX-07, FIX-11)  ⏱ ~2 h

| Việc | File | Cách sửa | Vì sao không giảm chất lượng |
|---|---|---|---|
| 4.1 | `audio_buffer.py:174-179` | Bỏ `self._buffer.fill(0)`; chỉ reset `_total_written`, `_dropped_samples_count`. An toàn vì mọi `get_slice` đã bị chặn bởi `_total_written`/`earliest_available` (`:114-127`) | Audio đọc ra không đổi; chỉ bỏ ghi đè 3.84 MB |
| 4.2 | `engine.py:230` | `CircularAudioBuffer(sample_rate=config.audio_buffer.sample_rate, capacity_sec=config.audio_buffer.capacity_sec)` | Cửa sổ đọc vẫn do cùng tham số quyết định |
| 4.3 | `tts-player.js:135` | Bỏ `arrayBuffer.slice(0)` — `content-script.js:299` đã tạo slice mới, và `decodeAudioData` detach buffer đó nên không ai dùng lại | WAV byte-identical |
| 4.4 | `audio_buffer.py:94-99` | Bỏ bước kiểm tra dtype thừa trong `write()` khi nguồn đã chắc chắn float32 (thêm đường `_write_preconverted`) | Bit-exact: giá trị float32 như cũ |
| 4.5 | `config.py:39` | Xoá `max_payload_bytes` (dead) hoặc nối vào giới hạn frame WS | Không ảnh hưởng pipeline |

**Test:** mở rộng `test_01_core_audio.py`: sau `clear()`, `get_slice(0, N)` phải trả mảng rỗng/rỗng-dữ-liệu; test bit-exact hiện có phải tiếp tục PASS. Thêm `test_11_config_effectiveness.py`: đổi `config.audio_buffer.capacity_sec` ⇒ `engine.audio_buffer.capacity_samples` phải đổi theo.

---

### Giai đoạn 5 — P2: Giảm contention (FIX-08, FIX-09)  ⏱ ~4–6 h

| Việc | Cách sửa | Ghi chú |
|---|---|---|
| 5.1 | **Metric sampling**: `_record_queue_gauges()` chỉ ghi khi `qsize` **đổi**, hoặc vượt high-watermark, hoặc ≥100 ms kể từ lần ghi trước | Giảm mạnh số lần lấy mutex; vẫn đủ để phát hiện backlog |
| 5.2 | **WS outbound priority queue**: worker **không** tự `send_*`; đẩy `(priority, payload)` vào queue có trần, một sender task duy nhất ghi ra socket. Ưu tiên: `subtitle_final > subtitle_preview > translation_partial > tts_binary` | Xoá head-of-line blocking; **không** đổi nội dung gói tin. Đây là thay đổi lớn nhất trong kế hoạch — làm **sau** Giai đoạn 0–4 |
| 5.3 | **Executor riêng cho TTS** (`ThreadPoolExecutor(max_workers=1, "tts_worker")`) và executor riêng cho model lifecycle | Cô lập blocking work khỏi default pool; không tăng tổng số thread |

**Test:** `test_17_executor_backpressure.py` + test mới: bơm 1000 metric ⇒ số lần acquire lock giảm ≥ 10×; test outbound: TTS frame 2 MB đang gửi thì preview phải vượt lên trước.

---

### Giai đoạn 6 — ASR compute: ĐO trước, SỬA sau (FIX-10)  ⏱ ~1 ngày

**Bước 1 — đo (không sửa gì):** thêm 2 counter
```
asr.audio_seconds_processed_preview   # tổng đã có: asr.preview_audio_sec
asr.audio_seconds_unique_preview      # audio mới thực sự mỗi vòng
asr.preview_recompute_ratio = processed / unique
```
Chạy lại `test_08_streaming_latency.py` và ghi số vào `report/audit/`. ChatGPT ước lượng ratio ~8×; **phải đo thật** trước khi tối ưu.

**Bước 2 — nếu ratio đáng kể, chọn biện pháp không mất chất lượng (theo thứ tự an toàn):**
1. **Ngưỡng audio mới (content-aware cadence)** — trong `stream_tokens`, bỏ qua vòng preview nếu `current_total - last_preview_end_sample < 150 ms`, trừ khi VAD vừa đổi trạng thái hoặc vừa có commit. **Không** đụng vào commit: commit vẫn infer trên mảnh đầy đủ ⇒ chất lượng phụ đề cuối **không đổi**.
2. **Dual cadence** — preview dày ở 0–1 s, thưa dần sau đó; commit không đổi.

**Bước 3 — TUYỆT ĐỐI KHÔNG** bật `preview_reuse_for_commit=True` (mặc định đang `False`) cho tới khi `test_09_wer_ab.py` chạy trên **từng family model** và chứng minh WER/CER không hồi quy (đã có `06_wer_ab.json`, `15_wer_ab_repeats.json` làm nền).

---

### Giai đoạn 7 — Native (ngoài phạm vi repo, ghi nhận)  ⏱ không ước lượng

`external/transcribe.cpp/src/arch/qwen3_asr/model.cpp:768-775` xoá sạch KV cache mỗi `run()` ⇒ mọi preview/commit đều là full offline inference. Đây là **upstream vendored**: theo `external/transcribe.cpp/AGENTS.md` **không patch/format trực tiếp** cây này. Hành động đúng: mở issue/PR ở upstream, hoặc `git submodule`/fork có kiểm soát. Điều này **chặn trần** mọi tối ưu ASR phía Python — nêu rõ trong tài liệu để không kỳ vọng sai.

---

## 5. NHỮNG VIỆC KHÔNG ĐƯỢC LÀM (rủi ro chất lượng / an toàn)

| Không làm | Lý do |
|---|---|
| Bật `preview_reuse_for_commit=True` để "giảm 30–50 % compute" (QWEN P0.2) | Từng gây hồi quy WER; có harness A/B riêng. Chỉ bật khi có bằng chứng WER theo từng model |
| Tách `_infer_lock` thành instance attribute (QWEN P1.5) | Native chỉ cho 1 `run()` in-flight/session (`engine.py:20-22`); tách lock ⇒ UAF/crash |
| Gỡ `_infer_lock` để benchmark nhanh hơn | Cùng lý do trên |
| Giảm precision model / n_ctx / số bước TTS / cửa sổ ASR để tiết kiệm VRAM | Vi phạm ràng buộc "không giảm chất lượng" |
| Patch trực tiếp `external/transcribe.cpp/**` | Vi phạm quy ước vendored của chính upstream |
| Bỏ `AudioBuffer.clear()` zero-fill **mà không** kiểm tra `_total_written` | Chỉ an toàn vì `get_slice` đã chặn theo `_total_written`; phải giữ test bit-exact |

---

## 6. TIÊU CHÍ NGHIỆM THU

| # | Tiêu chí | Cách đo |
|---|---|---|
| 1 | `pytest` 0 failed | `python -m pytest backend/tests -q` |
| 2 | Backpressure phục hồi | Kịch bản F: chặn receiver ~10 s ⇒ thấy `paused` rồi **tự** `ok`; phụ đề hồi phục; `paused_ms` được ghi |
| 3 | Không mất bản dịch final | Scenario D: 20 câu final liên tiếp khi translation chậm ⇒ `translation.final_dropped == 0`, mọi câu đều có bản dịch |
| 4 | `load_model()` không chặn | Trong lúc nạp model 7B, `get_instance()` trả về < 100 ms |
| 5 | Chất lượng không đổi | WER/CER phụ đề **giống hệt** baseline (`test_09_wer_ab.py`, `15_wer_ab_repeats.json`); TTS WAV byte-identical khi `speed=1` |
| 6 | Copy/RAM giảm | `asr.preview_ms` p95, RSS sau 1000 vòng seek/reset, browser heap khi phát TTS |
| 7 | Metric ASR | `asr.preview_recompute_ratio` xuất hiện trong `/api/metrics` |

---

## 7. Ma trận benchmark đi kèm (theo ChatGPT §39, giữ nguyên)

| Kịch bản | Mục tiêu kiểm chứng |
|---|---|
| A — nói ngắn 1–2 từ | độ trễ preview đầu, commit giả |
| B — câu dài 5–10 s | số lần preview, `recompute_ratio`, p95 preview_ms |
| C — nói nhanh | backlog queue, độ chính xác commit |
| D — nhiều câu dài (translation-heavy) | `translation.final_dropped == 0`, TTFT |
| E — nhiều câu ngắn (TTS-heavy) | TTS queue depth, HOL blocking |
| F — nghẽn mạng | `bufferedAmount`, pause, **resume latency**, phục hồi phụ đề |
| G — seek liên tục mỗi 0.5–2 s | reset latency, nhiễm audio cũ, allocation spike |

---

## 8. Phụ lục — kết quả pytest thô

```
FAILED backend/tests/test_19_log_no_duplicate.py::test_partial_write_is_not_retried
FAILED backend/tests/test_20_logging_convention.py::test_every_logger_call_has_module_tag
FAILED backend/tests/test_22_translation_model_download.py::test_hotswap_disabled_download_raises_before_touching_model
FAILED backend/tests/test_22_translation_model_download.py::test_reconfigure_failure_keeps_working_model
FAILED backend/tests/test_22_translation_model_download.py::test_rest_busy_with_another_model_returns_409
```

Trích traceback (đã rút gọn):

- `test_19:90` → `assert ['2026-09-16'] == ['2026-09-15']`
- `test_20:91` → `3 lời gọi logger thiếu module_tag: ['backend\\translation\\hotswap.py:180', ':151', ':187']`
- `test_22:194` → `AssertionError: Test tầng A cố NẠP MODEL THẬT (translation)` (guard `conftest.py:193`)
- `test_22:221` → `Failed: DID NOT RAISE FileNotFoundError` (log: `Nạp thành công mô hình dịch 'tencent-1.8b' trên GPU`)
- `test_22:344` → `Failed: DID NOT RAISE HTTPException` (log: `main.py:546 … đã xếp lịch tải nền`)

---

## 9. Tóm tắt một câu

Ba báo cáo **không cùng độ tin cậy**: ChatGPT đúng phần lớn và đúng ở bug P0 quan trọng nhất (backpressure treo), Grok đúng về chẩn đoán kiến trúc nhưng sai 2 chi tiết code, **QWEN lỗi thời và có 2 khuyến nghị nguy hiểm (reuse preview, tách instance lock) phải loại bỏ**. Kế hoạch sửa gồm 8 giai đoạn, bắt đầu bằng dọn 5 test failure để nhìn thấy hồi quy, rồi P0 backpressure, P1 không mất final translation/TTS, lock hygiene, copy/RAM, contention — và **chỉ tối ưu ASR sau khi đo `recompute_ratio`**, tuyệt đối không đánh đổi chất lượng.

---
---

# PHẦN II — TRẠNG THÁI TRIỂN KHAI (2026-09-16)

Phần I ở trên là **báo cáo kiểm chứng**. Phần II ghi lại **những gì đã thực sự được sửa**, kèm cách kiểm chứng lại.

## II.1 Kết quả test sau khi sửa

| Bộ test | Trước | Sau |
|---|---|---|
| `pytest backend/tests` | **5 failed / 221 passed** | **0 failed / 268 passed** |
| `node --test extension_firefox/tests/` (mới) | — | **9 passed / 0 failed** |

> Ghi chú môi trường: `pytest` cần `tmp_path` ghi ra thư mục temp ngoài workspace. Nếu chạy trong sandbox hạn chế ghi, 19 test của `test_22` sẽ báo `PermissionError` ở bước setup — đó là lỗi hạ tầng, không phải lỗi sản phẩm.

## II.2 Bảng đối chiếu bug → bản sửa

| ID | Bug | File đã sửa | Cách sửa |
|---|---|---|---|
| **FIX-01** | P0 backpressure HARD treo capture | `extension_firefox/lib/backpressure-gate.js` (**mới**), `background/service-worker.js`, `content/content-script.js`, `lib/ws-client.js`, `manifest.json` | Tách máy trạng thái thuần có **hysteresis** (vào PAUSE ở `>= HARD`, chỉ nhả khi `< SOFT`); service worker tự kiểm tra lại bằng `setInterval` 100 ms; content script bắn probe `FLUSH_PENDING` 250 ms khi đang pause (chống MV3 tạm ngưng SW); timer được huỷ trong `cleanup()` và khi `CONNECT` lại; thêm `pauseCount/resumeCount/pausedMs` |
| **FIX-02** | Drop bản dịch **final** | `backend/ws/handler.py` (`_coalesce_enqueue`), `backend/ws/session.py`, `backend/config.py` | Trần hàng đợi 4 → **32** (cấu hình được); khi vẫn đầy thì **GỘP** vào câu mới nhất đang chờ thay vì vứt; ghi `merged_utterance_ids` và phát bản dịch cho **mọi** câu bị gộp (không còn phụ đề treo ở `"..."`); counter `queue.translation_merged` |
| **FIX-03** | Drop TTS **final** | như trên | Cùng cơ chế; counter `queue.tts_merged` |
| **FIX-04** | `load_model()` giữ `_shared_lock` khi build | `backend/translation/engine.py` | Build **ngoài** lock + double-check khi cài đặt; bản build dư bị đóng bằng `_close_llm_quietly()` (không chờ `_infer_lock`) |
| **FIX-05** | 5 test failure | `backend/tests/test_19,20,22` | Ngày hard-code → lấy từ cùng nguồn đồng hồ với `%(asctime)s`; test AST resolve được hằng chuỗi cấp module; fixture `tmp_models` vá **cả** `ensure_model_file()` (trước chỉ vá registry nên nạp model thật); bỏ monkeypatch `is_busy` sai; thêm test nhánh ngược lại (cùng model ⇒ không 409) |
| **FIX-06** | `clear()` zero-fill 3,84 MB | `backend/core/audio_buffer.py` | Bỏ `_buffer.fill(0)` (an toàn vì mọi đường đọc đều bị chặn bởi `_total_written`); `write_bytes`/`feed_audio` chia **tại chỗ** (1 mảng tạm thay vì 2) |
| **FIX-07** | `capacity_sec` bị hard-code | `backend/asr/engine.py` | Đọc `config.audio_buffer.{sample_rate,capacity_sec}` |
| **FIX-08** | WS single send lock (HOL) | — | **CHƯA LÀM** — cần outbound priority queue, là thay đổi lớn; xem II.3 |
| **FIX-09** | Metric ghi mỗi item + mutex toàn cục | `backend/ws/handler.py` (`_record_queue_gauges`) | Chỉ ghi khi độ sâu **đổi**, tạo high-watermark mới, hoặc quá 100 ms (nhịp tim khi queue kẹt > 0) |
| **FIX-10** | Không đo được lãng phí preview | `backend/asr/engine.py`, `backend/core/metrics.py` | Thêm `audio_seconds_processed_preview`, `audio_seconds_unique_preview`, `preview_recompute_ratio`; gauge sống mỗi vòng + metric chốt theo từng câu (có p50/p95) và reset khi bắt đầu câu mới |
| **FIX-11** | Browser TTS copy 2 lần | `extension_firefox/lib/tts-player.js` | Bỏ `arrayBuffer.slice(0)` (caller đã tạo buffer mới; `playedIds` chặn phát trùng nên detach an toàn) |
| **FIX-12** | `cancel_inference()` đọc session không lock | `backend/asr/engine.py` | Snapshot `_shared_session` trong `_shared_lock` rồi mới `cancel()` trên tham chiếu cục bộ |
| **FIX-13** | `drain_queues()` sau khi đã cancel worker | `backend/ws/handler.py` | Tách `_discard_queued()` (vứt + cân `_unfinished_tasks`), dùng chung cho tua video và dọn phiên; đổi thứ tự thành **huỷ worker → vứt hàng đợi**, bỏ `drain_queues(0.15)` vô nghĩa |
| **FIX-14** | `max_payload_bytes` là config chết | `backend/ws/handler.py` | `_handle_binary_message` chặn khung vượt trần, counter `ws.payload_rejected` |

## II.3 Việc CHƯA làm (có lý do)

| Việc | Lý do hoãn |
|---|---|
| **FIX-08** outbound priority queue (WS) | Thay đổi kiến trúc lớn nhất trong kế hoạch (mọi worker phải ngừng tự `send_*`). Nên làm thành một thay đổi riêng, có benchmark HOL blocking trước/sau. Hiện tượng chỉ thành vấn đề khi TTS frame lớn + receiver chậm. |
| **Giai đoạn 6 bước 2** (adaptive/dual cadence preview) | **Cố ý chờ số liệu.** Metric `asr.preview_recompute_ratio` giờ đã có; phải chạy benchmark (kịch bản B) để biết ratio thật rồi mới quyết định, tránh tối ưu mù. |
| Bật `preview_reuse_for_commit` | **Vẫn KHÔNG** — cần bằng chứng WER theo từng family (`test_09_wer_ab.py`). |
| Patch native KV cache (`external/transcribe.cpp`) | Thuộc upstream vendored; quy ước của chính upstream cấm patch trực tiếp. |

## II.4 Phát hiện MỚI trong lúc sửa (chưa xử lý, ghi để theo dõi)

1. **`TranslationDedupState.is_duplicate()` chỉ lưu văn bản NGUỒN, không lưu bản dịch.** Khi một câu bị coi là trùng lặp, `_process_translation_item()` `return` sớm ⇒ câu đó **không bao giờ nhận bản dịch** và phụ đề treo ở `"..."`. Đây là cùng loại lỗi "mất chất lượng" như FIX-02/FIX-03 nhưng ở đường khác. Hướng sửa: lưu kèm bản dịch vào history và phát lại bản dịch đã cache thay vì bỏ trắng. `backend/translation/dedup.py`, `backend/ws/handler.py:422-425`.
2. **`monkeypatch.setattr(<instance>, "<method>", ...)` để lại attribute instance vĩnh viễn.** Đã gặp thật: `test_22::test_hot_path_degrades_to_passthrough...` vá `load_model` lên **instance**; khi teardown, monkeypatch "khôi phục" bằng `setattr` lên chính instance ⇒ attribute đó **che method của class** cho mọi test sau dùng cùng singleton. Triệu chứng: `test_25` đỏ khi chạy full suite nhưng xanh khi chạy riêng. **Đã sửa** (vá ở class) và ghi chú ngay tại chỗ. Quy ước: với method của class, **luôn vá ở class**.
3. **`GGUFTranslator._instance` (singleton) tồn tại xuyên test.** Cần `reset_instance()` trong fixture thì các test translation mới thật sự độc lập. Hiện chưa gây lỗi nhưng là bẫy cho test sau.

## II.5 Cách kiểm chứng lại

```bash
# 1. Toàn bộ test backend (kỳ vọng: 268 passed, 0 failed)
python -m pytest backend/tests -q

# 2. Test máy trạng thái backpressure (kỳ vọng: 9 passed)
node extension_firefox/tests/backpressure-gate.test.js
#    hoặc bằng runner: node --test extension_firefox/tests/

# 3. Kiểm tra cú pháp JS + manifest
node --check extension_firefox/background/service-worker.js
node -e "JSON.parse(require('fs').readFileSync('extension_firefox/manifest.json','utf8'))"
```

**Kiểm thử thủ công cho FIX-01 (quan trọng nhất):** bật phụ đề, chặn/ngắt kết nối backend ~10 s rồi mở lại để `bufferedAmount` vượt HARD (hoặc ngắt tạm receiver). Kỳ vọng trong console:
- `[BS Background] Backpressure HARD: tạm dừng gửi audio.` → `[BS] Backend chậm: tạm dừng gửi audio… (pause #1, …)`
- khi mạng thông: `[BS Background] Backpressure đã rút hết: chạy lại capture.` → `[BS] Backend đã bắt kịp: tiếp tục gửi audio. (paused …ms, resume #1)`
- **Trước khi sửa:** chỉ có dòng đầu, phụ đề đứng vĩnh viễn cho tới khi bấm Stop/Start.

