# BÁO CÁO AUDIT HIỆU NĂNG CHUYÊN SÂU — Hệ thống Vibe Translation Addon (transcribe.cpp)

> **Người thực hiện:** Senior Software Architect + Performance Engineer + Code Reviewer (đánh giá độc lập)
> **Phạm vi:** Toàn bộ `backend/**` (Python, hot-path pipeline), `extension_firefox/**` (MV3 JS), và tham chiếu `external/transcribe.cpp` (C++/ggml) qua các phụ lục đã có.
> **Phương pháp:** Đọc tĩnh (static read) toàn bộ mã nguồn hot-path + cross-check với `report/audit/00_*.md` để phân biệt **đã sửa** vs **còn lại**.
> **Nguyên tắc:** Bắt đúng `file:line`. Không sửa code.

---

## 0. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Hệ thống đã trải qua **một đợt tái cấu trúc lớn** so với báo cáo audit gốc (`00_*.md`). Nhiều root-cause cũ đã được xử lý thực tế trong mã hiện tại:

| Root-cause cũ (trong `00_*.md`) | Trạng thái hiện tại | Bằng chứng |
| :--- | :--- | :--- |
| F-01 preview O(N²) | ✅ **Đã sửa** — preview bị cửa sổ hoá `preview_window_sec` | `asr/engine.py:1000-1009`, `:1263` |
| F-02 CommitManager dead code | ✅ **Đã sửa** — `_evaluate_tier234` được gọi mọi vòng | `asr/engine.py:1333-1359` |
| F-04 race `unload_shared_model` | ✅ **Đã sửa** — giữ `_infer_lock` trước `_shared_lock` | `asr/engine.py:113-114` |
| F-05 drop âm thầm `_pending_commits` | ✅ **Đã sửa** — gộp (merge) thay vứt | `asr/engine.py:600-636` |
| F-08 TTS base64 | ✅ **Đã sửa** — binary frame `BTTS` | `tts/engine.py:289-312`, `serializers.py:119-170` |
| F-12 logger flush mỗi record | ✅ **Đã sửa** — flush mỗi 0.25s / WARNING+ | `utils/logger.py:160-229` |
| F-13 VAD inference trong lock | ✅ **Đã sửa** — tách model ra ngoài lock | `vad/processor.py:248-332` |
| F-15 `_checkpoints` vô hạn | ✅ **Đã sửa** — bounded `maxlen=200` | `core/metrics.py:20,148-149` |
| F-30 payload phình | ✅ **Đã sửa** — protocol v3 gọn | `serializers.py:47-58`, `config.py:44` |
| F-16/F-07 queue vô hạn client/server | ✅ **Đã sửa** — bounded + backpressure gate | `tts-player.js:19-21`, `backpressure-gate.js` |

**Tuy nhiên, vẫn còn các điểm nghẽn thực sự cần xử lý** — chia thành 3 nhóm:

1. **Kiến trúc singleton / global lock (CRITICAL, không đổi):** toàn bộ ASR + dịch + TTS là singleton chia sẻ 1 session/model, mọi session xếp hàng. Đây là giới hạn cứng, không thể chạy >1 stream đồng thời chất lượng cao. Là thiết kế có chủ ý ("1 phiên / 1 video") nhưng là **bottleneck scale**.
2. **GPU contention giữa Vulkan (ASR) và CUDA (dịch/TTS) + `torch.cuda.synchronize()` toàn device (HIGH):** gây priority inversion và latency spike.
3. **Các lãng phí vi mô / cache / threading chưa tận dụng (MEDIUM):** warmup lặp file, int16→float32 copy kép, dedup double-pass, `use_context` build prompt mỗi câu, `torch.inference_mode` không bật Autocast, v.v.

Dưới đây là báo cáo chi tiết từng phát hiện, mức độ, và đề xuất (ROI từ cao → thấp).

---

## 1. KIẾN TRÚC & MÔ HÌNH THỰC THI (bản đồ thread/lock)

```
┌─ uvicorn event loop (1 thread) ──────────────────────────────────────────────┐
│  handle_ws() → _stream_asr_tokens() ─┐                                       │
│  _translation_worker()              │  coroutine / session                    │
│  _tts_worker()                      ┘                                         │
└───────────────┬───────────────────────┬──────────────────┬───────────────────┘
                │ run_in_executor        │ run_in_executor   │ asyncio.to_thread
                ▼                       ▼                  ▼
   ┌────────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
   │ _VAD_EXECUTOR      │  │ _EXECUTOR        │  │ default executor         │
   │ max_workers=1      │  │ max_workers=2    │  │ (TTS synthesize_clone_    │
   │ (handler.py:38)    │  │ (engine.py:56)   │  │  bytes → to_thread)      │
   │ VAD + frame slicing│  │ ASR inference    │  │ (tts/engine.py:312)      │
   └─────────┬──────────┘  └────────┬─────────┘  └──────────────────────────┘
             │                      │
             ▼                      ▼
   ┌──────────────────────┐  ┌──────────────────────────────────────────────┐
   │ _shared_session       │  │ GGUFTranslator._shared_llm (singleton)       │
   │ _shared_model (ASR)   │  │ OmniVoiceTTS.model (singleton)               │
   │ _infer_lock (RLock)   │  │ _infer_lock (translation), _infer_lock (TTS)│
   │ Vulkan GPU            │  │ CUDA GPU                                      │
   └──────────────────────┘  └──────────────────────────────────────────────┘
```

**Quan sát then chốt:** 1 ASR session + 1 dịch LLM + 1 TTS engine cho **toàn bộ** server. Mọi tài nguyên GPU là singleton có lock. Đây là mô hình "tối ưu cho đúng 1 session" như README mô tả — nhưng là **giới hạn cứng**, không phải lựa chọn mở rộng (xem §2).

---

## 2. BOTTLENECK THEO TÀI NGUYÊN

### A1. CPU BOTTLENECK

#### 🔴 A1-1 — Thread oversubscription trên máy <12 nhân (HIGH)
`backend/main.py:30-45` + `vad/engines/firered.py:33-35` + `vad/engines/silero.py:48-49` + `asr/engine.py:319 (n_threads=4)` + `translation/engine.py:103 (n_threads=4)`.

| Thành phần | Threads CPU |
| :--- | :--- |
| PyTorch (VAD + TTS) | 2 (`main.py:41`) |
| llama.cpp dịch | 4 (`n_threads=4`) |
| transcribe.cpp ASR | 4 (`asr.threads=4`) |
| GIL-bound Python workers | VAD×1 + ASR×2 + dịch×1 + TTS(to_thread) |
| OMP/MKL | 2 (PASSIVE — ✅ tốt) |

→ **~14 thread compute thường trực** + browser (video decode/render + audio worklet + extension JS). Trên CPU 8 nhân (máy tham chiếu là RTX 5060 Ti thường gắn với Ryzen 7/9, nhưng máy người dùng có thể 4–6 nhân) đây là **oversubscription thực sự**. Patch threadpool ggml đã được áp dụng (README) nhưng **VAD engine FireRed/Silero tự gọi `torch.set_num_threads(2)` chỉ khi hiện tại >2** — nếu process cha đã set 2 thì OK, nhưng nếu extension/browser chiếm nhân, ASR vẫn đòi 4.

**Đề xuất (không giảm chất lượng):** hạ `asr.threads` và `translation.n_threads` xuống 3 (hoặc auto = `max(2, cpu_count//3)`). Đo thực tế: ASR Vulkan không dùng CPU nhiều; dịch CUDA cũng chủ yếu GPU. Giữ CPU cho VAD + audio.

#### 🟡 A1-2 — `SpeechNormalizer` vẫn cấp phát 1 mảng đích mỗi lần (MEDIUM, đã tốt hơn cũ)
`backend/core/normalizer.py:135-137`. Đã tối ưu (RMS bằng `np.dot`, peak bằng `max/min`, 1 mảng `out`). Nhưng vẫn `np.empty_like(arr)` + 2 phép biến đổi mỗi lần normalize, chạy **mỗi lần inference** (preview + commit). Với câu 6s @16k = 96k mẫu × 4 byte = 384 KB × ~4 lần/giây ≈ **1.5 MB/s allocation churn**. Có thể dùng buffer tái sử dụng (pinned/reused) nhưng lợi ích nhỏ.

#### 🟡 A1-3 — `np.frombuffer(int16).astype(float32)/32768` copy kép (MEDIUM) — xem §B3-1
Đường VAD/ASR vẫn chuyển Int16 → Float32 2 lần (VAD feed_chunk `processor.py:303-304`, ASR feed_audio `engine.py:513-516`).

### A2. GPU / VRAM BOTTLENECK

#### 🔴 A2-1 — ASR (Vulkan) + Dịch/TTS (CUDA) tranh chấp 1 GPU, không có scheduler (HIGH)
`backend/main.py:99-105` xác nhận ASR chạy Vulkan (không có CUDA wheel), dịch + TTS chạy CUDA. Ba model cùng 1 card, **không có cơ chế ưu tiên**:

| Stage | Engine | Deadline |
| :--- | :--- | :--- |
| ASR preview/commit | transcribe.cpp (Vulkan) | cần xong trong poll (~300ms) |
| Translation | llama.cpp (CUDA), `_infer_lock` | 250–800ms/câu |
| TTS | OmniVoice (CUDA), `_infer_lock` | ~420ms/câu |

Vì translation/TTS chạy trong worker coroutine riêng, chúng có thể chiếm GPU **đúng lúc** ASR commit đang chạy → **priority inversion**: câu TTS 420ms đẩy ASR commit trễ thêm vài trăm ms (spike phụ đề nhìn thấy được). Đây là nguyên nhân chính của các spike ~2.5s đã ghi nhận (`asr/engine.py:1014-1018`).

**Đề xuất (không giảm chất lượng):**
- Thêm **GPU time-slicing / priority**: chạy ASR trên luồng riêng ưu tiên (đã có `_EXECUTOR`), nhưng CUDA `torch.cuda.synchronize()` không phân biệt stream (xem A2-2).
- Cân nhắc ép ASR chạy CUDA nếu build được `transcribe-cpp-native-cu12` (README thừa nhận chưa có wheel) — sẽ loại bỏ tranh chấp 2 backend.

#### 🔴 A2-2 — `torch.cuda.synchronize()` đồng bộ TOÀN DEVICE trong TTS (HIGH)
`backend/tts/engine.py:193, 249`:
```python
if torch.cuda.is_available() and "cuda" in str(self.device):
    torch.cuda.synchronize()   # ← drain toàn bộ GPU, chặn cả ASR/Vulkan? (Vulkan riêng)
```
`torch.cuda.synchronize()` đồng bộ **mọi stream CUDA đang chờ** — không chỉ stream TTS. Nếu dịch và TTS đan xen, mỗi lần TTS xong đều ép device CUDA nghỉ → làm trễ inference dịch tiếp theo. Trong khi ASR là Vulkan (không bị chặn trực tiếp bởi CUDA sync), nhưng **dịch bị block** → pipeline dịch bị gián đoạn.

**Đề xuất:** thay bằng `torch.cuda.Stream.synchronize()` trên stream riêng của TTS, hoặc bỏ hẳn (OmniVoice trả tensor đã hoàn tất trên stream của nó; `convert_to_numpy` chạy trên CPU sau đó). Đo để xác nhận không đổi chất lượng.

#### 🟠 A2-3 — TTS model không giải phóng khi tắt `tts_enabled` (MEDIUM)
`backend/tts/engine.py:323-342` `unload_model` tồn tại và đúng, nhưng `main.py:566-569` / `handler.py:380-388` khi tắt TTS chỉ đổi cờ `config.tts.enabled`, **model vẫn nằm trong VRAM** (vài GB float16). Đây là **cơ hội giảm VRAM không giảm chất lượng**: gọi `unload_model()` khi `tts_enabled → False`, nạp lại khi bật.

#### 🟡 A2-4 — `_voice_prompt_cache` giữ embedding GPU (LOW, đã bounded)
`backend/tts/engine.py:64-65, 130-140`: LRU max 8 — ✅ đã tốt. Với 1 voice thực tế không đáng lo.

### A3. RAM / MEMORY GROWTH

#### 🟠 A3-1 — Phình bộ nhớ native (transcribe.cpp / Vulkan) chưa xử lý tận gốc (HIGH, mang tính nền tảng)
`backend/asr/engine.py:170-204` (`maybe_recycle_native`) + `config.asr.native_recycle_rss_delta_mb=2048`. Đây là **workaround**: khi RSS vượt mốc nạp model + 2 GB, đóng phiên native để lần sau nạp lại (tốn ~10s). Nguyên nhân gốc nằm trong native (`external/transcribe.cpp`, xem `02_phu_luc_transcribe_cpp.md` F-31…F-35: threadpool CPU dùng-một-lần mỗi graph compute, scheduler/compute context dựng lại mỗi `run()`, H2D/D2H không cần thiết). **Chưa sửa tận gốc** — chỉ giảm triệu chứng. Đây là lý do README cảnh báo "RAM tăng liên tục khi chạy lâu".

#### 🟡 A3-2 — `_checkpoints` đã bounded nhưng `_latencies` deque 10000 (LOW)
`core/metrics.py:20,38`. Đã bounded ✅. Không phải leak.

#### 🟢 A3-3 — Buffer cố định (GOOD)
`CircularAudioBuffer` 60s = 3.84 MB ✅; VAD `raw_buffer` cap 3s ✅; `_pending_commits` bounded ✅; client TTS queue `maxQueueLength=3` + `maxLagMs=12000` ✅.

### A4. I/O BOTTLENECK

#### 🟠 A4-1 — Đổi model ASR/TTS đọc lại GGUF từ đĩa trong request (MEDIUM)
`backend/main.py:479` `await asyncio.to_thread(engine.prepare_model, ...)` đọc lại GGUF 1.3–3.3 GB + dummy inference, **trong request HTTP** → request treo vài giây, mọi session đang chạy mất ASR trong lúc đó. Đã có hot-swap an toàn (prepare ngoài lock) nhưng **vẫn block 1 session duy nhất** vì là singleton. Không thể song song vì giới hạn singleton (§2).

#### 🟡 A4-2 — Warm-up dịch lặp file 2 lần (LOW)
`translation/engine.py:255-262` `prewarm()` gọi `load_model()` (đọc GGUF) rồi `_translate_sync("Hello")` (dummy). Cùng file đọc 2 lần (1 lần mmap, 1 lần dummy run). Có thể gộp nhưng tác động nhỏ.

#### 🟢 A4-3 — TTS binary frame đã bỏ base64 (GOOD)
`serializers.py:119-170`, `tts/engine.py:289-312`. ✅ KHÔNG còn +33% + vòng lặp per-byte `atob` trên main thread. Đây là cải tiến lớn so với `00_*.md`.

---

## 3. COPY / ALLOCATION / CONVERSION KHÔNG CẦN THIẾT

### 🟠 B3-1 — Int16→Float32 conversion kép qua VAD→ASR (MEDIUM)
- VAD `feed_chunk` (`vad/processor.py:303-304`): với Silero/FSMN, `np.frombuffer(frame_bytes, int16).astype(float32)/32768` → copy + div.
- ASR `feed_audio` (`asr/engine.py:513-516`): lại `np.frombuffer(bytes, int16).astype(float32)/32768` → copy + div thứ 2.
- `CircularAudioBuffer` lưu Float32 (`audio_buffer.py:24`).

Tổng **≥ 4 copy** cho mỗi mẫu (client Int16 → bytes → VAD float32 → ASR float32 → ring float32). FireRed đã tránh (truyền Int16 thẳng), nhưng ASR vẫn chuyển Int16→Float32.

**Đề xuất:** giữ Int16 tới tận `CircularAudioBuffer` (buffer `int16`, ½ RAM) và chỉ convert Float32 **1 lần** ngay trước khi đưa vào model (có thể ngay trong `get_slice` hoặc `_run_inference_sync`). Không đổi chất lượng, giảm RAM + CPU.

### 🟡 B3-2 — `get_slice` copy toàn bộ đoạn mỗi preview/commit (LOW–MEDIUM)
`core/audio_buffer.py:131-145`: `np.empty` + copy từ ring. Với cửa sổ 6s = 96k mẫu × 4B = 384 KB × ~3 lần/giây. Có thể trả `memoryview`/zero-copy nếu model chấp nhận buffer không liên tục, nhưng transcribe.cpp cần mảng liên tục → giữ copy là hợp lý. Đã bounded bởi `preview_window_sec`.

### 🟡 B3-3 — `parse_audio_frame` slice PCM (LOW)
`ws/protocol.py:36`: `pcm = data[4+header_len:]` → bytes slice (view, không copy sâu nhờ Python bytes slicing là O(1) view). ✅ tốt. Nhưng sau đó `feed_chunk` → `raw_buf.extend(pcm_data)` copy vào bytearray. Khó tránh vì VAD cần buffer tăng dần.

### 🟡 B3-4 — TTS `convert_to_numpy` + `apply_time_stretch` allocate nhiều mảng (LOW)
`tts/audio_processor.py:27-54, 80-108`: `concatenate`, STFT 3 mảng phức, ISTFT. Với speed≠1 (mặc định 1.0 nên thường bỏ qua `:259` guard). Chỉ chạy khi `tts_speed≠1`.

---

## 4. LOCK / MUTEX / RACE / DEADLOCK

### ✅ ĐÃ SỬA — các race cũ
- `unload_shared_model` giờ giữ `_infer_lock` trước `_shared_lock` (`asr/engine.py:113-114`).
- VAD `is_speech()` chạy **ngoài** `self._lock` (`vad/processor.py:295-312`), chỉ lock cho đọc/ghi state. Spike cleanup 102ms cũ đã được xử lý.
- `cancel_inference` đọc session trong lock (`asr/engine.py:476-477`).

### 🟡 B4-1 — `_infer_lock` (RLock) ASR là global, serializes mọi session (MEDIUM, thiết kế)
Mọi session dùng chung `_shared_session` + `_infer_lock` (`asr/engine.py:79-80`). 1 inference chạy → các session khác chờ. Đây là **root cause của scale=1**. Không phải bug, nhưng là bottleneck kiến trúc. Xem §2.

### 🟡 B4-2 — `SafeWebSocketConnection._send_lock` serialize mọi writer (LOW–MEDIUM)
`ws/connection.py:36, 63, 81`: mọi `send_text`/`send_bytes` qua 1 `asyncio.Lock`. Với 3 worker (ASR preview, dịch, TTS) cùng gửi, chúng serialize trên lock này. Trên localhost băng thông cao, tác động nhỏ, nhưng là điểm tranh chấp tiềm tàng khi nhiều session. Có thể tách lock per-message-type nhưng lợi ích hạn chế.

### 🟢 B4-3 — `_coalesce_enqueue` drain queue O(n) khi đầy (GOOD tradeoff)
`ws/handler.py:196-247`: chỉ chạy khi queue đầy (hiếm nhờ maxsize=32). Merge thay vứt ✅. An toàn vì event loop đơn luồng.

---

## 5. THREAD / PROCESS CONTENTION & QUEUE

### 🟠 B5-1 — ASR executor `max_workers=2` nhưng `max_inflight_infer=1` (LOW–MEDIUM)
`asr/engine.py:56` (`_EXECUTOR = ThreadPoolExecutor(max_workers=2)`) vs `config.asr.max_inflight_infer=1` (`config.py:154`). Với `max_inflight=1`, chỉ 1 inference chạy → 1 worker đủ. Worker thứ 2 chỉ dùng khi `max_inflight_infer>1`. Hiện tại **worker thứ 2 là dư thừa** (nhưng vô hại). Để mở rộng throughput ASR cần nâng `max_inflight_infer` — nhưng native `transcribe.cpp` giới hạn 1 stream in-flight/session (`external/transcribe.cpp/include/transcribe.h`).

### 🟡 B5-2 — Translation executor `max_workers=1` (MEDIUM, thiết kế)
`translation/engine.py:36`: 1 worker → dịch tuần tự. Với 1 session là đủ (dịch nhanh ~67 tok/s, câu 3s dịch ~2-3s). Nhưng nếu câu dài/nhieu, queue dịch (maxsize=32) có thể dồn ứ. Đã có backpressure (merge) nên không mất câu. Có thể nâng `max_workers=2` cho dịch nếu GPU còn dư (CUDA này riêng biệt với ASR Vulkan).

### 🟢 B5-3 — Backpressure thực sự đã nối (GOOD)
`extension_firefox/background/service-worker.js` + `backpressure-gate.js` + `ws-client.js:333-341` (`flushPending` liveness probe). Đã sửa lỗi "capture treo vĩnh viễn" cũ (FIX-01). ✅

---

## 6. LATENCY SPIKE & THROUGHPUT

### 🔴 B6-1 — Spike preview do GPU contention + scheduler native (HIGH)
`asr/engine.py:1014-1038` (`_effective_poll_interval` adaptive backoff) đã **giảm thiểu** bằng cách giãn nhịp khi chậm. Nhưng nguyên nhân gốc: (a) CUDA sync toàn device (A2-2), (b) transcribe.cpp dựng lại scheduler/compute context mỗi `run()` (`02_phu_luc_transcribe_cpp.md` F-32). Adaptive backoff giúp "đều hơn" nhưng không loại bỏ spike.

### 🟠 B6-2 — `inference_watchdog` + RSS runaway guard (GOOD nhưng tốn CPU)
`asr/engine.py:809-895`: watchdog poll mỗi 0.25s khi chạy. Đã bounded bởi `inference_watchdog_sec=8`. Cơ chế an toàn cần thiết do native có thể treo. Không phải bottleneck thường nhật.

### 🟡 B6-3 — `use_context` build prompt mỗi câu dịch (MEDIUM)
`ws/handler.py:587` `ctx_tracker.get_context_str()` chạy mỗi câu (dù `use_context=False` mặc định → trả `""`). Nếu bật, build string từ deque mỗi lần. Hiện tại default off ✅. Khi bật, có thể cache formatted string và chỉ rebuild khi history đổi.

---

## 7. CÁC ĐOẠN XỬ LÝ TUẦN TỰ CÓ THỂ SONG SONG HÓA

| Đoạn | Hiện tại | Có thể song song? |
| :--- | :--- | :--- |
| ASR preview vs commit | Tuần tự trong 1 generator (`engine.py:1208-1403`) | Preview nhường commit (`:1243-1246`) ✅ đã ưu tiên |
| Dịch vs TTS | 2 worker riêng ✅ | Đã song song |
| VAD vs ASR | VAD executor riêng ✅ | Đã song song |
| **Dịch nhiều câu** | 1 translator worker | Có thể 2 worker nếu GPU dư (CUDA riêng ASR) |
| **ASR multi-stream** | 1 session (native limit) | ❌ không thể với 1 model; cần N model cho N session |

**Kết luận:** trong 1 session, pipeline đã song song hoá tối đa (VAD∥ASR∥dịch∥TTS trên các executor/worker khác nhau). Nút thắt là **GPU contention** (A2) chứ không phải thiếu song song.

---

## 8. CODE LẶP LẠI & ABSTRACTION

### 🟡 B8-1 — Double dedup pass (MEDIUM)
`ws/handler.py`: `(1)` TranslationDeduplicator (`translation/dedup.py`) exact + TTL 10s, `(2)` TTSDedupState (`tts/dedup.py`). Hai lớp dedup riêng biệt, mỗi câu chạy 2 lần kiểm tra. Logic gần giống. Có thể gộp thành 1 shared dedup service hoặc truyền kết quả dedup từ dịch → TTS (vì TTS chỉ nhận câu đã qua dịch). Giảm ~1 pass/câu.

### 🟡 B8-2 — `count_content_tokens` gọi 2 lần cho 1 câu (LOW)
`ws/handler.py:459` (filter) + `engine.py` internal. Có thể tính 1 lần truyền xuống. Tác động nhỏ.

### 🟢 B8-3 — Abstraction hợp lý (GOOD)
`BaseASREngine`/`BaseVADEngine`/`BaseTTSEngine`/`BaseTranslator` tách interface rõ. `from_config` pattern cho normalizer ✅. `hotswap` module riêng ✅. Không thấy abstraction dư thừa gây chậm.

---

## 9. CACHE & OBJECT LIFETIME / GC

### 🟢 B9-1 — Voice prompt cache (GOOD)
`tts/engine.py:114-140` LRU max 8, promote MRU. ✅ tiết kiệm ~70ms/câu.

### 🟡 B9-2 — Translation prompt strategy không cache (LOW)
`translation/engine.py:292-299` build prompt mỗi câu (template format). Có thể cache template string. Nhỏ.

### 🟡 B9-3 — `maybe_recycle_native` đóng/mở model định kỳ (MEDIUM, workaround)
`asr/engine.py:170-204`: khi RSS vượt +2GB, đóng native session → **lần sau nạp lại mất ~10s** (warmup lại). Đây là GC thủ công do native leak. Giải pháp thực tế duy nhất không patch upstream, nhưng gây **gián đoạn ~10s** nếu trigger. Nên để `native_recycle_rss_delta_mb=0` (tắt) nếu máy đủ RAM, hoặc patch `external/transcribe.cpp` tận gốc.

### 🟡 B9-4 — `gc.collect()` trong TTS load/unload (LOW)
`tts/engine.py:198, 335` gọi `gc.collect()` explicit. Có thể loại bỏ (Python tự GC) trừ khi đo thấy fragmentation. Nhỏ.

---

## 10. EXTENSION CLIENT-SIDE (MV3)

### ✅ ĐÃ SỬA so với `00_*.md`
- **F-06 AudioWorklet dead code** → giờ worklet là đường chính (`audio-capture.js:122-187`), ScriptProcessor chỉ fallback. ✅
- **F-26 resample per-sample + atob** → worklet resample phase-linear, transferable buffer (`audio-processor.js:62-85, 117-132`). ✅
- **F-27 findVideo quét DOM mỗi sự kiện** → đã cache `cachedVideo` + TTL miss 2s (`content-script.js:83-98, 472-537`). ✅
- **F-28 TTS phát 2 lần trong iframe** → `shouldPlay` logic (`content-script.js:120-121, 324-325`). ✅
- **F-29 force 16kHz AudioContext** → dùng rate gốc (`audio-capture.js:43-47`). ✅

### 🟡 B10-1 — `findVideo()` vẫn quét Shadow DOM đệ quy mỗi miss (MEDIUM)
`content-script.js:483-508`: khi `cachedVideo` mất kết nối (video bị thay thế DOM), quét toàn bộ `querySelectorAll("*")` đệ quy shadow root + iframe. Trên trang phức tạp (YouTube, Bilibili) đây là **O(DOM size)** mỗi lần miss (tối đa 1 lần/2s nhờ TTL). Đã có TTL nhưng vẫn có thể jank nếu DOM lớn. Có thể dùng `IntersectionObserver` hoặc chỉ quét `video` (không `*`) để giảm.

### 🟡 B10-2 — `decodeAudioData` detach buffer (LOW, đã tối ưu)
`tts-player.js:134-154`: dùng chung AudioContext, decodeAudioData off-thread. Đã bỏ `slice(0)` thừa (FIX-11). ✅

### 🟢 B10-3 — Backpressure bridge (GOOD)
service-worker + backpressure-gate hysteresis ✅. Capture tạm dừng khi `bufferedAmount` vượt HARD, resume khi < SOFT.

---

## 11. NHỮNG CHỖ CÓ THỂ GIẢM LATENCY (KHÔNG GIẢM CHẤT LƯỢNG)

| # | Điểm | Kỹ thuật | Ước lượng |
| :--- | :--- | :--- | :--- |
| L1 | `torch.cuda.synchronize()` toàn device (A2-2) | → `Stream.synchronize()` hoặc bỏ | giảm spike dịch/TTS 100–400ms |
| L2 | ASR preview windowed (đã làm) + adaptive backoff (đã làm) | giữ nguyên | ✅ đã tối ưu |
| L3 | GPU priority: chạy ASR trên luồng ưu tiên / CUDA stream riêng | tránh inversion | giảm spike commit |
| L4 | Int16 giữ tới tận model (B3-1) | giảm copy + RAM | ~1.5 MB/s alloc giảm |
| L5 | Bật `preview_reuse_for_commit` sau đo WER (config.py:145) | tiết 1 inference/câu | -~420ms/câu (ASR) |

## 12. NHỮNG CHỖ CÓ THỂ GIẢM RAM/VRAM (KHÔNG GIẢM CHẤT LƯỢNG)

| # | Điểm | Kỹ thuật |
| :--- | :--- | :--- |
| R1 | TTS model unload khi tắt `tts_enabled` (A2-3) | gọi `unload_model()` |
| R2 | Int16 ring buffer (B3-1) | ½ RAM audio |
| R3 | `native_recycle_rss_delta_mb=0` nếu RAM dư (B9-3) | tránh gián đoạn 10s |
| R4 | TTS `_voice_prompt_cache` đã bounded ✅ | giữ nguyên |

## 13. NHỮNG CHỖ CÓ THỂ TĂNG THROUGHPUT

| # | Điểm | Kỹ thuật | Rủi ro |
| :--- | :--- | :--- | :--- |
| T1 | Dịch `max_workers=2` (B5-2) | song song 2 câu dịch | VRAM/dịch có thể kẹt |
| T2 | ASR `max_inflight_infer=2` | cần native support (chưa có) | ❌ giới hạn thư viện |
| T3 | Ép ASR chạy CUDA (build cu12) | 1 backend GPU | cần build riêng |

---

## 14. KẾT LUẬN & KHUYẾN NGHỊ ƯU TIÊN

### 🔴 CRITICAL (kiến trúc, cần quyết định sản phẩm)
1. **Scale = 1 session** do singleton + global lock + native 1-stream limit. Nếu muốn N phiên: cần N model instance (VRAM ×N) + model pool + per-session lock. Đây là thiết kế có chủ ý ("cá nhân, 1 video") — giữ nguyên trừ khi có yêu cầu multi-stream.

### 🟠 HIGH (sửa trong vài giờ, ROI cao)
2. **A2-2:** bỏ / thu hẹp `torch.cuda.synchronize()` trong TTS → giảm spike dịch/TTS.
3. **A2-1:** thêm GPU scheduling / CUDA stream riêng cho TTS, ưu tiên ASR commit.
4. **A3-1:** theo dõi native leak; cân nhắc patch `external/transcribe.cpp` (F-31…F-35) tận gốc thay vì recycle.

### 🟡 MEDIUM (sửa khi rảnh, lợi ích tích luỹ)
5. **B3-1:** Int16→Float32 single conversion.
6. **A1-1:** hạ `asr.threads`/`translation.n_threads` trên máy ít nhân.
7. **A2-3:** unload TTS khi tắt.
8. **B8-1:** gộp dedup pass.

### 🟢 LOW (nice-to-have)
9. B10-1 findVideo chỉ quét `video`. B6-3 cache context string. B9-2 cache prompt template.

---

## 15. PHỤ LỤC — CHECKLIST FINDING (mới vs cũ)

| ID | Mô tả | Mức | Trạng thái hiện tại |
| :--- | :--- | :--- | :--- |
| F-01 | ASR preview O(N²) | CRIT | ✅ ĐÃ SỬA (windowed) |
| F-02 | CommitManager dead code | CRIT | ✅ ĐÃ SỬA (wired) |
| F-03 | No multi-session | CRIT | ⚠️ THIẾT KẾ (scale=1) |
| F-04 | Race unload model | CRIT | ✅ ĐÃ SỬA (lock order) |
| F-05 | Drop commit âm thầm | CRIT | ✅ ĐÃ SỬA (merge) |
| F-06 | Worklet dead code | CRIT | ✅ ĐÃ SỬA (worklet primary) |
| F-07 | No backpressure | CRIT | ✅ ĐÃ SỬA (gate) |
| F-08 | TTS base64 | HIGH | ✅ ĐÃ SỬA (binary) |
| F-12 | Logger flush/spam | HIGH | ✅ ĐÃ SỬA (interval) |
| F-13 | VAD trong lock | HIGH | ✅ ĐÃ SỬA (tách) |
| F-15 | `_checkpoints` vô hạn | HIGH | ✅ ĐÃ SỬA (bounded) |
| F-16 | TTS client queue vô hạn | HIGH | ✅ ĐÃ SỬA (maxQueueLength) |
| F-19 | Vulkan+Cuda tranh chấp | HIGH | 🔴 CÒN (A2-1) |
| F-30 | Payload phình | LOW | ✅ ĐÃ SỬA (v3) |
| F-31…35 | Native per-run overhead | CRIT | 🔴 CÒN (A3-1, cần patch upstream) |
| **A2-2** | `torch.cuda.synchronize()` toàn device | HIGH | 🔴 CÒN (mới phát hiện) |
| **B3-1** | Int16→Float32 double convert | MED | 🔴 CÒN (mới phát hiện) |
| **A2-3** | TTS không unload khi tắt | MED | 🔴 CÒN (mới phát hiện) |
| **B8-1** | Double dedup pass | MED | 🔴 CÒN (mới phát hiện) |
| **B10-1** | findVideo shadow DOM scan | MED | 🔴 CÒN (đã có TTL) |

---

**Tài liệu tham chiếu:** `README.md`, `report/audit/00_BAO_CAO_AUDIT_HIEU_NANG.md`, `report/audit/02_phu_luc_transcribe_cpp.md`, `report/audit/05_measurements_and_status.md`.
