# BÁO CÁO AUDIT HIỆU NĂNG & KIẾN TRÚC (TƯƠI)
## Vibe Translation Addon — transcribe.cpp (Real-time bilingual subtitle + neural MT + TTS)

| | |
| :--- | :--- |
| **Vai trò** | Senior Software Architect + Performance Engineer + Code Reviewer |
| **Phạm vi** | Toàn bộ tree trong zip đính kèm (backend Python + extension_firefox JS + tham chiếu report/audit cũ + README) |
| **Nguồn** | Code hiện tại trong archive + README + report/audit/* (đối chiếu) |
| **Nguyên tắc** | Đọc source-level; mỗi finding kèm file/đường dẫn; phân loại theo tài nguyên & mức độ |
| **Mục tiêu hệ thống** | E2E live subtitle thấp latency, 100% offline, 1 GPU, tối đa 1 session/video (theo README) |
| **Ngày** | 2026-09-16 |

---

## 0. TÓM TẮT ĐIỀU HÀNH

### 0.1 Bối cảnh

Pipeline:

```
Browser tab audio → Extension capture → WSS binary frames
  → VAD (FireRed/Silero/FSMN) → ASR (transcribe.cpp / Vulkan|CPU)
  → Commit (4 bậc) → Translation (llama.cpp GGUF CUDA) → TTS (OmniVoice CUDA)
  → Subtitle overlay + binary TTS audio
```

README và code hiện tại **đã sửa** nhiều CRITICAL của audit cũ (`report/audit/00_...`):

- Commit Manager 4 bậc **đã được wire** vào `stream_tokens` / `_evaluate_tier234`.
- Preview **có cửa sổ** (`preview_window_sec`, đảm bảo ≥ `max_duration_sec`) → không còn O(N²) không giới hạn.
- `_pending_commits` không còn `maxlen` drop âm thầm; có trần + gộp/merge + gauge.
- VAD `is_speech()` chạy **ngoài** RLock.
- Logger không còn flush mỗi record + không còn DEBUG mặc định trên hot path.
- Metrics `_checkpoints` có trần (OrderedDict).
- TTS gửi **binary frame** (`BTTS` + WAV thô), không còn base64 +33% trên đường chính.
- Client có backpressure `bufferedAmount`, TTS queue trần, seek/reset an toàn qua `_stream_generation`.
- Hot-swap model, streaming translation partial, metrics pipeline endpoint.

### 0.2 Root causes còn lại (sau các vá P0–P3)

| # | Vấn đề gốc còn lại | Bản chất | Hệ quả |
| :--- | :--- | :--- | :--- |
| **RC-A** | **Singleton + global lock toàn pipeline GPU** | 1× ASR session (`_shared_session` + `_infer_lock`), 1× Translation LLM, 1× TTS engine; executor VAD/Trans = 1 worker | Scale cứng = **1 session**. Multi-tab / multi-user xếp hàng, latency tăng tuyến tính |
| **RC-B** | **ASR offline-style re-run mỗi poll** (kể cả có cửa sổ) | Hầu hết model (Qwen3-ASR, SenseVoice, …) không giữ KV/encoder state giữa các `run()`; mỗi preview/commit = full mel+encoder+decode trên cửa sổ | GPU/CPU fixed cost mỗi poll; native ggml còn rebuild scheduler/threadpool (theo phụ lục native cũ) |
| **RC-C** | **Hai stack GPU trên cùng card** | ASR = **Vulkan**; Translation + TTS = **CUDA/PyTorch** | VRAM không thống nhất, context switch driver, khó pin/graph, peak VRAM ~9.5 GB đo được nhưng không tối ưu |
| **RC-D** | **AudioWorklet vẫn dead / capture vẫn ScriptProcessor-heavy** | Harness test tồn tại nhưng runtime extension vẫn phụ thuộc ScriptProcessor + resample JS (theo plan P3.4 chưa xong) | Main-thread jank, latency capture, glitch khi tab busy |
| **RC-E** | **Native transcribe.cpp chưa reuse state** | Mỗi `session.run()` dựng lại compute context / threadpool (trừ một số arch); không dùng `cancel()` an toàn; không CUDA graphs | Amplify RC-B; latency spike khi câu dài / cold |

### 0.3 Xếp hạng finding hiện tại (sau vá)

| ID | Phát hiện | Mức | Tác động |
| :--- | :--- | :--- | :--- |
| **C-01** | Singleton + `_infer_lock` / `_TRANS_EXECUTOR(1)` / TTS singleton | 🔴 CRITICAL (scale) | Không multi-session |
| **C-02** | Mỗi inference ASR = full offline run trên slice (dù đã window) | 🔴 CRITICAL (cost) | GPU bound, preview cost vẫn cao khi max_duration lớn |
| **C-03** | Dual GPU API (Vulkan ASR + CUDA MT/TTS) | 🟠 HIGH | VRAM/scheduling kém, khó scale GPU |
| **C-04** | `get_slice` luôn allocate `np.empty` mới; int16↔float32 copy nhiều tầng | 🟠 HIGH | Alloc churn, CPU, cache miss |
| **C-05** | Executor contention: VAD=1, Trans=1, ASR=2, default pool cho TTS+prewarm | 🟠 HIGH | Queue backlog khi speech dày |
| **C-06** | Native: không cancel an toàn; không state reuse; threadpool churn (nếu lib chưa vá) | 🟠 HIGH | Latency spike, CPU |
| **C-07** | Extension: AudioWorklet chưa production; ScriptProcessor + resample per-sample | 🟠 HIGH | Capture latency + jank |
| **C-08** | Queue translation/TTS có maxsize nhưng policy “drop oldest / keep latest” cần giám sát chặt dưới load | 🟡 MEDIUM | Mất câu dịch/TTS khi nghẽn |
| **C-09** | Logging vẫn có global `RLock` trên handler (đã giảm flush) | 🟡 MEDIUM | Jitter nhẹ khi nhiều thread log |
| **C-10** | Config còn một số field “chết” / hardcode (n_ctx, n_batch, n_threads translation) | 🟡 MEDIUM | Khó tune |
| **C-11** | Metrics gauge tốt; nhưng thiếu histogram depth dài hạn + alert khi pending_commits/inflight cao | 🟡 MEDIUM | Observability |
| **C-12** | Pre-warm dài (~47s dịch 7B) + warm long path | 🟢 LOW (UX) | First-token sau cold vẫn phụ thuộc warm |
| **C-13** | Payload JSON vẫn lặp field (snake/camel, text/original) ở một số msg | 🟢 LOW | Bandwidth nhẹ |
| **C-14** | Race unload model: đã sửa thứ tự lock; vẫn cần cẩn khi REST config trong lúc infer | 🟢 LOW (đã mitig) | Crash native nếu regress |

---

## 1. KIẾN TRÚC RUNTIME (hiện tại)

```
┌─ uvicorn / asyncio event loop (1 thread) ─────────────────────────────┐
│  handle_ws ──► receive binary/text                                    │
│  stream_tokens (ASR) ──► yield utterance_update / commit              │
│  _translation_worker ──► streaming partial + final                    │
│  _tts_worker ──► synthesize + send binary                             │
└───────────────────────────────────────────────────────────────────────┘
        │ run_in_executor / to_thread
        ▼
┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐
│ _VAD_EXECUTOR    │  │ _EXECUTOR (ASR)  │  │ _TRANS_EXECUTOR (1)        │
│ max_workers=1    │  │ max_workers=2    │  │ + GGUFTranslator._infer_lock│
│ VAD forward      │  │ + _infer_lock    │  │                            │
└──────────────────┘  │   class RLock    │  └────────────────────────────┘
                      │ _shared_session  │
                      │ (1 model only)   │  default executor → TTS / prewarm
                      └──────────────────┘
```

- **Một** `TranscribeEngine._shared_session` + `_infer_lock` cho toàn process.
- **Một** `GGUFTranslationEngine._shared_llm` + `_infer_lock`.
- **Một** TTS engine (OmniVoice) + lock.
- `transcribe.cpp` 0.x: **một** `run()` in-flight / model (header library). Đây là giới hạn cứng, không phải chỉ “chưa implement pool”.

**Kết luận scale:** đúng như README — thiết kế **1 phiên / 1 video**. Muốn N phiên phải N bản model (VRAM × N) + N lock domain.

---

## 2. PHÂN TÍCH THEO TÀI NGUYÊN

### 2.1 CPU bottleneck

| Vùng | Hiện tượng | Ghi chú |
| :--- | :--- | :--- |
| ASR host pre/post | `get_slice` copy, normalize, text clean, materialize result | Mỗi poll; đã có metric `slice_ms`, `normalize_ms` |
| VAD | 1 worker; forward Torch (CPU) mỗi chunk | Đã tách khỏi lock → cleanup không bị block 100ms nữa |
| Logging | Global RLock trên `SafeStreamHandler`; flush theo interval / WARNING | Đã tốt hơn DEBUG+flush mỗi dòng; vẫn serialize nhẹ |
| Extension capture | ScriptProcessor + resample JS | Main thread; P3.4 AudioWorklet chưa xong |
| Native ggml (nếu chưa vá) | Threadpool create/join mỗi graph; mel STFT threads | Amplify CPU khi poll dày |

**Khuyến nghị CPU:**
- Giữ `max_inflight_infer=1` (đã có) để không xếp hàng slice lớn.
- Adaptive poll (P2.4b) đã có — giữ và tune theo `preview_ms` p95.
- Khi native cho phép: reuse threadpool / pinned host buffer.

### 2.2 GPU / VRAM bottleneck

| Thành phần | Backend | Ghi chú |
| :--- | :--- | :--- |
| ASR | **Vulkan** (PyPI `transcribe-cpp-native`) | Không có CUDA wheel thật trên PyPI; cu12 là name reservation |
| Translation | **CUDA** (llama-cpp-python cu124) | n_ctx/n_batch/n_threads hardcode một phần |
| TTS | **CUDA** (PyTorch OmniVoice) | Voice clone; binary out |
| VAD | CPU Torch | Không chiếm VRAM đáng kể |

- Peak VRAM đo ~9.5 GB (4 model) trên RTX 5060 Ti 16 GB — còn headroom nhưng dual-stack Vulkan+CUDA kém tối ưu hơn single-stack.
- Không CUDA graphs / session cancel an toàn → khó abort khi seek/hotswap giữa chừng (code cố ý **không** gọi `cancel()` vì làm chết worker).

**Khuyến nghị GPU:**
- Ưu tiên build CUDA backend ASR từ source nếu toolchain có (giảm dual API).
- Hoặc chấp nhận Vulkan + giảm kích thước model ASR / cửa sổ preview khi VRAM tight.
- Translation: expose `n_ctx`, `n_batch`, `n_threads` đầy đủ từ config (một số còn hardcode).

### 2.3 RAM / memory growth

| Nguồn | Trạng thái |
| :--- | :--- |
| `CircularAudioBuffer` | Fixed capacity float32 — OK |
| `get_slice` | **Allocate mới mỗi lần** — churn |
| `_pending_commits` | Bounded (`_MAX_PENDING_COMMITS=6`) + merge — OK |
| Translation queue | `maxsize=8` + drop intermediate partial — OK |
| TTS client queue | Trần 3 + age drop — OK (extension) |
| Metrics checkpoints | OrderedDict có trần — OK |
| Native RSS | Có baseline + `maybe_recycle_native` (F-39) — tốt |
| Mem guard / stall watchdog | Có (`utils/mem_guard.py`, `stall_watchdog.py`) |

**Rủi ro còn lại:** khi ASR chậm + speech liên tục, nhiều `audio_slice` có thể sống trên queue executor nếu `max_inflight` bị tăng nhầm → RAM tăng. Default `max_inflight_infer=1` là đúng.

### 2.4 I/O & transport

- WSS binary audio in; binary TTS out (`BTTS`) — tốt.
- JSON text cho subtitle/translation — vẫn ổn; một số field trùng (thấp ưu tiên).
- Client backpressure `bufferedAmount` (SOFT 128KB / HARD 512KB) — đã có.
- Không thấy server-side `bufferedAmount` check (ít cần vì 1 session).

### 2.5 Lock / mutex / thread contention

| Lock | Phạm vi | Rủi ro |
| :--- | :--- | :--- |
| `TranscribeEngine._infer_lock` | Mọi `run()` ASR | Serialize toàn bộ ASR; đúng vì native yêu cầu |
| `TranscribeEngine._shared_lock` | Load/unload model | Thứ tự bắt buộc `_infer_lock` → `_shared_lock` (đã document, tránh UAF) |
| `GGUFTranslator._infer_lock` | Mọi generate | 1 translation at a time |
| VAD `RLock` | Chỉ state machine, **không** ôm `is_speech()` | Đã sửa |
| Logger `SafeStreamHandler._lock` | Class-level | Serialize log emit |
| `_ACTIVE_SESSIONS_LOCK` | Registry session | Nhẹ |

**Không thấy deadlock cycle** rõ trong thứ tự lock đã document. Race unload đã được mitig bằng giữ `_infer_lock` khi close native handle.

### 2.6 Queue / producer-consumer

- ASR → commit deque bounded.
- Commit → translation `asyncio.Queue(maxsize=…)` + QueueFull handling.
- Translation → TTS queue + QueueFull.
- Policy: ưu tiên commit hơn preview; drop partial translation cũ; TTS drop oldest/age.

**Cân bằng:** khi ASR commit burst (nhiều câu ngắn), translation single-worker có thể backlog → partial bị drop (chấp nhận được nếu final còn). Cần metric `queue.translation_depth` / `tts_depth` (đã có gauge) theo dõi production.

### 2.7 Copy / allocation / conversion không cần thiết

1. **int16 → float32** khi nhận frame (ASR `feed_audio` / buffer `write_bytes`).
2. **`get_slice` → `np.empty` + copy** mỗi preview/commit.
3. Normalizer (nếu bật) tạo mảng tạm.
4. Extension: base64 path legacy vẫn tồn tại cho client cũ (OK); binary path chính.

**Giảm latency/RAM không mất chất lượng:**
- Giữ buffer int16 ở phía circular (P2.9 trong plan — **chưa làm**), chỉ convert khi đưa vào model.
- Zero-copy / view khi slice không wrap (circular đã có logic wrap).
- Tránh materialize timestamps/words nếu model/API không cần (một phần đã có `timestamps` arg).

### 2.8 Latency path & parallelization

**Đã song song (per session):**
- VAD executor ⊥ ASR executor ⊥ translation executor ⊥ TTS to_thread.
- Preview defer khi inflight đầy.
- Streaming translation (first token ~26 ms đo được).

**Vẫn tuần tự bắt buộc:**
- ASR run (native single in-flight).
- Translation generate (1 lock).
- TTS synthesize (1 engine).

**Có thể song song thêm (chất lượng không đổi):**
- Pre-fetch / pipeline: trong lúc TTS câu N, translation câu N+1 (đã gần như vậy qua queue).
- Parallel VAD trên nhiều session **chỉ khi** có multi-model (hiện không).

### 2.9 Race / deadlock / unbounded

- Unload vs infer: **đã khóa đúng thứ tự**.
- Seek: `_stream_generation` drop stale result — **không** gọi `session.cancel()` (tránh kill worker) — đúng trade-off.
- Empty commit streak guard + `await sleep(0)` chống quay nóng event loop — tốt (F-47/F-48).
- Không còn unbounded `_pending_commits` / metrics checkpoints.

### 2.10 Logging / polling / cache

- Logger: INFO default, flush interval — OK.
- Poll ASR: fixed rate + adaptive stretch khi chậm — tốt.
- Voice prompt LRU / model registry: có.
- `findVideo` cache TTL phía extension — đã vá.

---

## 3. PIPELINE STAGE — NGHẼN DOWNSTREAM

| Stage | Nghẽn chính | Downstream impact |
| :--- | :--- | :--- |
| Capture (ext) | ScriptProcessor / main thread | Frame trễ → VAD/ASR lệch thời gian |
| VAD | Single worker | Nếu VAD chậm → ASR nhận ít / trễ speech boundary |
| ASR preview | GPU full run trên window | Chiếm GPU → commit phải chờ; translation đói |
| Commit | 4 bậc (OK) | Burst commit → translation queue |
| Translation | Single GGUF + lock | Backlog → drop partial; TTS thiếu input |
| TTS | Single OmniVoice | Queue client drop → lệch tiếng |
| WS send | Binary tốt | Client backpressure bảo vệ RAM tab |

**Ưu tiên cắt latency không mất chất lượng:**
1. Giữ cửa sổ preview nhỏ nhất vẫn ≥ max_duration (đã enforce).
2. Commit ưu tiên hơn preview (đã có).
3. Streaming translation partial (đã có).
4. Binary TTS (đã có).
5. Adaptive poll (đã có).

**Tăng throughput (1 session):**
- Giảm fixed cost native mỗi `run` (cần native patch).
- Model ASR nhỏ hơn / quant mạnh hơn cho preview, model lớn chỉ commit (architecture “dual model” — chưa có).
- Tăng `n_batch` translation cẩn thận (VRAM).

**Giảm RAM/VRAM không mất chất lượng:**
- int16 circular buffer.
- Unload model không dùng (hotswap đã nạp trước rồi swap).
- Quant Q4/Q5 cho translation khi 16 GB tight.
- Không load TTS nếu user tắt lồng tiếng.

---

## 4. ARCHITECTURE SCALE

- **Hiện tại:** tối ưu 1 session — đúng product goal README.
- **Muốn N session:**  
  - N × model ASR (Vulkan/CUDA) + N session handle.  
  - N × translation context hoặc queue + multi-LLM (VRAM).  
  - Session-scoped executor / lock domain.  
  - Connection limit + admission control.  
  - Đây là redesign lớn, không phải “bật flag”.

Abstraction hiện tại (Engine registry, adapters, CommitManager, metrics) **không** là nguyên nhân chậm; chúng giúp test/hotswap. Phần chậm nằm ở **native re-run + singleton GPU + dual API**.

---

## 5. NHỮNG GÌ ĐÃ TỐT (giữ)

- Commit 4 bậc wired + stability + max_duration + overlap boundary.
- Preview window + priority commit + max_inflight.
- Seek reset qua generation counter (an toàn native).
- Binary TTS + client backpressure + TTS queue bound.
- Streaming translation + first-token metric.
- VAD ngoài lock; logger không flush storm; metrics bounded + pipeline snapshot.
- Stall watchdog + mem/RSS recycle hooks.
- Hot-swap ASR/translation với load-before-swap.
- Test suite rộng (core, VAD, ASR, commit, translation, TTS, e2e, locks, executor backpressure, …).

---

## 6. KHUYẾN NGHỊ ƯU TIÊN (roadmap ngắn)

### P0 — An toàn / quan sát (nhanh)
1. Dashboard/alert trên `pending_commits`, `queue.*_depth`, `asr.preview_ms` p95, `inflight`.
2. Đảm bảo config translation `n_ctx/n_batch/n_threads` đọc 100% từ config (bỏ hardcode còn sót).
3. Document rõ: **1 session only**; API reject session thứ 2 hoặc queue với warning.

### P1 — Latency / cost ASR (impact lớn nhất)
1. Giữ / tune `preview_window_sec` ≈ `max_duration_sec` (6s class).
2. Nếu native cho phép: incremental / stateful decode cho model streaming (Nemotron/Voxtral family).
3. Đánh giá dual-path: model nhỏ cho preview, model lớn chỉ lúc commit (chất lượng commit không giảm).

### P2 — GPU stack
1. Build CUDA ASR từ `external/transcribe.cpp` nếu có toolchain → thống nhất CUDA.
2. Hoặc profile Vulkan vs CUDA trên cùng model để chọn.

### P3 — Capture & copy
1. Hoàn thành AudioWorklet production (P3.4).
2. Circular buffer int16 + convert muộn (P2.9).

### P4 — Native (nếu maintain fork)
1. Reuse ggml threadpool / compute context giữa `run()`.
2. Pinned memory; tránh D2H→H2D thừa; optional CUDA graphs.
3. `cancel()` an toàn hoặc cooperative abort flag.

---

## 7. KẾT LUẬN

Hệ thống **đã trưởng thành rõ** so với audit tĩnh ban đầu: các lỗi thiết kế gây O(N²) không giới hạn, drop commit âm thầm, VAD lock, base64 TTS, metrics leak, logging storm đã được xử lý có chủ đích (comment P1/P2/F-xx trong code).

**Điểm nghẽn còn lại mang tính kiến trúc:**

1. **1 session / singleton GPU** (by design + limit thư viện).
2. **Full offline re-inference mỗi poll** dù đã window (chi phí GPU cố định).
3. **Vulkan ASR + CUDA MT/TTS** trên cùng card.
4. **Capture path extension** chưa AudioWorklet.
5. **Native ggml** còn room tối ưu state reuse / threadpool.

Với mục tiêu **cá nhân, 1 video, E2E thấp latency**, stack hiện tại hợp lý và đã đo được commit p50 ~107 ms / preview p95 ~108 ms / first token dịch ~26 ms trên RTX 5060 Ti 16 GB. Muốn scale multi-stream hoặc cắt thêm 30–50% GPU ASR thì cần native stateful + (tùy chọn) CUDA unified + model dual-path — không còn là “micro-optimization Python”.

---

*Báo cáo này dựa trên source trong archive tại thời điểm review; nếu working tree đã diverge thêm so với zip, cần re-diff các file `backend/asr/engine.py`, `backend/ws/handler.py`, `backend/translation/engine.py`, extension `lib/audio-capture.js`.*
