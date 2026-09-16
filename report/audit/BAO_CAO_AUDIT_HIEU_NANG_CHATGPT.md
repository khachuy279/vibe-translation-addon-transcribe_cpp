# Deep Performance / Architecture / Concurrency Audit
## `vibe-translation-addon-transcribe_cpp`

**Vai trò review:** Senior Software Architect + Performance Engineer + Code Reviewer  
**Snapshot chính:** ZIP người dùng cung cấp ngày 2026-09-16  
**Repository tham chiếu:** `https://github.com/khachuy279/vibe-translation-addon-transcribe_cpp`  
**Phạm vi:** Backend Python, Firefox Extension JavaScript, test suite, config, metrics/logging, lifecycle model, queue/concurrency và các report/benchmark đi kèm.  
**Không bao gồm:** source nội bộ của `transcribe.cpp` native C/C++ vì source upstream/native không nằm trong ZIP snapshot. Vì vậy các vấn đề nằm bên trong native Vulkan/ggml chỉ có thể đánh giá ở mức integration boundary, benchmark evidence và lifecycle; không thể thực hiện line-by-line code review native.

---

## 1. Executive Summary

Đây là một pipeline realtime khá hoàn chỉnh:

```text
Browser Video Audio
   ↓
AudioWorklet / ScriptProcessor
   ↓
PCM16 16 kHz
   ↓
WebExtension bridge / WebSocket
   ↓
VAD
   ↓
ASR transcribe.cpp
   ↓
Preview / Commit Manager
   ↓
GGUF Translation
   ↓
TTS OmniVoice
   ↓
WebSocket
   ↓
Browser subtitle + audio playback
```

Kiến trúc hiện tại đã sửa được nhiều lỗi lớn của các revision trước: queue đã có giới hạn ở nhiều tầng, commit backlog được merge thay vì drop thẳng, VAD model inference được đưa ra ngoài state lock, metrics checkpoint được bound, binary TTS đã được bổ sung, config n_ctx/n_batch/n_threads của translation đã được expose, ASR preview đã có bounded window, watchdog/cancel và native recycle guard đã được thêm.

Tuy nhiên, sau khi đọc toàn bộ source trong ZIP và chạy test suite, vẫn có một số bottleneck mang tính kiến trúc chứ không chỉ micro-optimization.

### 1.1 Các bottleneck quan trọng nhất

| ID | Vấn đề | Mức độ | Tác động chính | Chất lượng có thể giữ nguyên? |
|---|---|---|---|---|
| P0 | ASR preview vẫn re-infer toàn bộ cửa sổ preview tối đa 6s | **Critical** | GPU/CPU ASR, latency, repeated work | Có, nếu chuyển sang incremental state/cache; preview reuse trực tiếp hiện chưa chứng minh được WER |
| P0 | Backpressure HARD ở Extension có trạng thái pause nhưng không có cơ chế resume chủ động | **Critical** | Có thể treo capture sau khi congestion | Có |
| P1 | Audio bị copy nhiều lần qua browser → bridge → packet → NumPy → ring buffer → ASR slice | **High** | CPU, allocation, memory bandwidth, latency jitter | Có |
| P1 | Global singleton + global inference locks biến process thành single-lane cho ASR/Translation/TTS | **High** cho scale, **Medium** cho single-session | Không scale nhiều session | Có nếu giữ design single-session hiện tại |
| P1 | Translation worker/executor chỉ có 1 lane; streaming generator giữ infer lock trong toàn bộ generation | **High** | Queue backlog + cancellation latency | Có |
| P1 | WebSocket outbound single lock gây head-of-line blocking giữa subtitle/translation/TTS | **High** | Output latency spike | Có |
| P1 | `MetricsCollector` dùng global mutex cho mọi latency/gauge/counter | **Medium/High** | Lock contention trong hot path | Có |
| P1 | Translation `load_model()` giữ `_shared_lock` trong toàn bộ model build | **Medium/High** | Model switch/prewarm block | Có |
| P1 | TTS GPU → CPU → NumPy + time-stretch + normalize + WAV encode tạo nhiều memory traffic | **Medium/High** | TTS latency/CPU | Có |
| P1 | Binary TTS vẫn tạo thêm full-frame copy; browser TTS path còn thêm 2 copy buffer | **Medium/High** | RAM bandwidth, GC | Có |
| P2 | `AudioBuffer.clear()` zero-fill toàn bộ 60s buffer mỗi reset | **Medium** | CPU/cache traffic | Có |
| P2 | `write_bytes(int16)` tạo `float32` array mới rồi ring-buffer lại copy một lần nữa | **Medium** | CPU + allocation | Có |
| P2 | `apply_time_stretch()` dùng Phase Vocoder SciPy trên CPU | **Medium** khi speed != 1 | CPU spike | Có, nếu giữ algorithm/đưa sang dedicated path |
| P2 | `qsize()` + metrics write trên mỗi queue item | **Low/Medium** | Lock overhead | Có |
| P2 | Một số config/compatibility path còn dead/legacy | **Low** | Complexity/maintenance | Có |

### 1.2 Điều quan trọng nhất về ASR

Preview loop đã được cải thiện từ kiểu O(N²) không bị giới hạn xuống kiểu **bounded-window repeated inference**, nhưng vẫn chưa phải incremental streaming thực sự.

Hiện tại mỗi preview tiếp tục thực hiện:

```text
AudioBuffer.get_slice(...)
   ↓
copy tối đa 6s float32
   ↓
normalization
   ↓
transcribe.cpp inference lại trên cửa sổ đó
```

Nghĩa là khi câu đang nói, cùng một lượng audio cũ tiếp tục bị encoder/model xử lý lặp lại.

Đây là bottleneck GPU/CPU lớn nhất còn lại trong chính architecture Python-side.

### 1.3 Điều quan trọng nhất phía browser

`service-worker.js` có:

```js
if (buffered >= SEND_BUFFER_HARD_LIMIT) {
    pausedByBackpressure = true;
    ...
    return;
}
```

nhưng cơ chế `FLUSH_PENDING` hiện không có caller tương ứng trong extension snapshot. Vì capture đã bị pause thì sẽ không còn frame mới đi vào `sendOrQueue()` để kích hoạt việc tự kiểm tra lại socket.

Kết quả có thể là:

```text
WS congestion
    ↓
HARD limit
    ↓
pausedByBackpressure = true
    ↓
audio capture bị dừng
    ↓
không còn SEND_BINARY mới
    ↓
không có ai tự flush/resume
    ↓
capture có thể ở trạng thái paused lâu dài
```

Đây là lỗi behavior/backpressure thực sự, không phải micro-optimization.

---

# 2. Evidence & Confidence

### 2.1 Những gì đã được kiểm chứng trực tiếp

- ZIP được giải nén và đọc toàn bộ source tree.
- Đã đọc `README.md` trước khi review code.
- Đã rà source backend + extension + test suite.
- Python backend `compileall`: **PASS**.
- JavaScript `node --check`: **PASS** cho các file JS trong snapshot.
- `pytest -q`: **268 passed / 6 failed**.

### 2.2 Sáu test failure hiện tại

1. `test_bit_exact_audio_integrity`: thiếu WAV fixture trong `/wav_test`.
2. `test_cuda_warning_helper_never_raises`: runtime hiện tại không expose provider nên assertion test fail.
3. `test_partial_write_is_not_retried`: test fixture kỳ vọng date hard-coded khác date runtime.
4. `test_every_logger_call_has_module_tag`: còn 3 logger calls thiếu `module_tag` trong `backend/translation/hotswap.py`.
5. `test_reconfigure_failure_keeps_working_model`: môi trường test không có `llama-cpp-python`, nên fail sớm bằng `RuntimeError` thay vì tới assertion lifecycle.
6. `test_rest_busy_with_another_model_returns_409`: test giả lập busy không khớp đường implementation hiện tại; request được xếp lịch background thay vì raise 409.

Các failure này không chứng minh pipeline production hỏng. Chúng chứng minh **test environment / test contract hiện chưa sạch**, và đặc biệt regression coverage quanh hotswap/model lifecycle chưa đáng tin tuyệt đối.

### 2.3 Những gì chưa thể verify line-by-line

Source native `transcribe.cpp` không có trong ZIP. Vì vậy:

- memory leak bên trong Vulkan/ggml allocator chưa thể xác nhận bằng source inspection;
- scheduler/context recreation trong native chỉ được đánh giá từ integration behavior/report;
- H2D/D2H, pinned memory, ggml threadpool và allocator internals không thể kết luận chính xác hơn mức integration boundary.

---

# 3. Architecture Map

## 3.1 Backend process model

Backend đang thiên mạnh về **single-process / single-model-instance / single-session**.

Các global/singleton đáng chú ý:

```text
ASR:
  _shared_model
  _shared_session
  _shared_lock
  _infer_lock
  _EXECUTOR(max_workers=2)

Translation:
  _shared_llm
  _shared_lock
  _infer_lock
  _TRANS_EXECUTOR(max_workers=1)

TTS:
  OmniVoice singleton
  _init_lock
  _infer_lock

VAD:
  global _VAD_EXECUTOR(max_workers=1)

Metrics:
  singleton + global mutex

WS connection:
  per-connection send mutex
```

Design này hoàn toàn hợp lý cho mục tiêu **1 session → 1 video → 1 audio stream**, nhưng nó tạo một ranh giới rất rõ:

> Process hiện tại được tối ưu cho *latency + safety của một pipeline*, không phải throughput của nhiều pipeline.

Đây không phải lỗi nếu product scope cố định một session. Nó chỉ trở thành lỗi nếu sau này chạy 2–N session trong cùng process.

---

# 4. P0 — ASR Preview Repeated Inference

## 4.1 Code path

`backend/asr/engine.py`

- `_preview_window_start()`
- `stream_tokens()`
- `audio_buffer.get_slice()`
- `_infer_with_watchdog()`
- `_run_inference_sync()`

Trong `stream_tokens()`, preview được thực hiện lại theo polling loop, nhưng input là cửa sổ audio hiện tại chứ không phải state incremental.

Ví dụ logic hiện tại:

```text
speech starts
   ↓
preview #1 → slice audio window → ASR
   ↓
preview #2 → slice larger/current window → ASR
   ↓
preview #3 → slice larger/current window → ASR
   ↓
...
```

Window đã bị bound ở 6 giây nên không còn O(N²) vô hạn. Nhưng về compute vẫn là:

```text
O(number_of_preview_calls × ASR_cost(window))
```

thay vì gần:

```text
O(total_new_audio)
```

nếu có true streaming cache.

## 4.2 Vì sao đây là bottleneck số 1

ASR là stage nặng nhất và là stage mà downstream không thể bù.

Nếu mỗi 300 ms có một preview inference, trong 6 giây câu nói có thể xuất hiện hàng chục lần inference trên phần audio chồng lấn rất lớn.

Đặc biệt:

```python
win_start = self._preview_window_start(...)
audio_slice = self.audio_buffer.get_slice(win_start, current_total)
await self._infer_with_watchdog(audio_slice)
```

đồng nghĩa mỗi vòng preview có:

1. ring-buffer copy;
2. optional normalization;
3. native ASR encode/decode lại;
4. text cleanup;
5. metrics.

## 4.3 Tối ưu đúng hướng

### Phương án A — true incremental ASR state

Nếu binding/native model hỗ trợ streaming state đúng nghĩa:

```text
new PCM chunk
   ↓
stream_state.feed(new_pcm)
   ↓
reuse encoder/cache
   ↓
new tokens only
```

Đây là hướng tốt nhất cho performance.

### Phương án B — dual cadence

Không cần preview với cùng chi phí:

```text
0–1s: frequent low-cost preview
1–6s: slower preview
commit: full-quality inference
```

Chất lượng commit không đổi.

### Phương án C — content-aware preview

Chỉ re-infer khi audio mới đủ lớn hoặc VAD confidence thay đổi đáng kể.

Ví dụ:

```text
new_audio < 100ms → skip
new_audio >= 150ms → preview
VAD boundary change → preview immediately
```

Điều này giữ latency tốt nhưng giảm redundant compute.

## 4.4 Không nên bật lại preview reuse mù quáng

README/report cho thấy preview reuse đã từng bị tắt vì WER regression. Vì vậy:

> Không nên dùng “reuse preview for commit” chỉ vì nó nhanh hơn.

Phải benchmark WER/CER theo từng family/model và chứng minh equivalence hoặc có threshold kiểm soát.

---

# 5. P0 — Browser Backpressure Can Get Stuck in PAUSED

File: `extension_firefox/background/service-worker.js`

Các threshold:

```js
SOFT = 128 KB
HARD = 512 KB
```

Logic HARD:

```js
if (buffered >= HARD) {
  pausedByBackpressure = true;
  notifyBackpressure("paused");
  return;
}
```

Logic resume nằm trong branch:

```js
if (pausedByBackpressure) {
  pausedByBackpressure = false;
  notifyBackpressure("ok");
}
```

nhưng branch này chỉ được tới khi `sendOrQueue()` được gọi lại.

Trong snapshot, `FLUSH_PENDING` có implement nhưng không thấy producer/caller tương ứng từ extension path.

## 5.1 Fix đề xuất

Khi chuyển sang HARD:

```text
start backpressure monitor
    ↓
setTimeout / timer 50–100 ms
    ↓
check bufferedAmount
    ↓
if < SOFT:
    clear paused state
    emit 
```

Ở đây nên có thêm hysteresis:

```text
HARD = 512 KB
resume < SOFT = 128 KB
```

để tránh pause/resume liên tục.

Ngoài ra nên ghi:

- `backpressure_pause_count`
- `backpressure_resume_count`
- `audio_frames_dropped`
- thời gian ở trạng thái paused

để xác định congestion thật hay chỉ spike ngắn.

---

# 6. P1 — Audio Copy / Allocation Chain

Đây là bottleneck memory-bandwidth khá rõ.

## 6.1 Browser side

`ws-client.js`:

```js
if (ArrayBuffer.isView(pcmData)) {
    rawBuffer = pcmData.buffer.slice(...)
}
```

Nếu caller đã đưa `ArrayBuffer`, path zero-copy được giữ. Nhưng khi caller đưa view thì `.slice()` tạo buffer mới.

Sau đó bridge:

```js
port.postMessage({
    action: "SEND_BINARY",
    header,
    pcmBuffer: rawBuffer
});
```

Extension messaging dùng structured clone semantics; snapshot không có transfer-list path ở đây. Vì vậy có nguy cơ có thêm copy khi đi qua runtime port.

Cuối cùng service worker:

```js
buildBinaryAudioPacket(header, msg.pcmBuffer, textEncoder)
```

tạo **một ArrayBuffer packet mới** rồi copy header + PCM vào.

Một frame 64ms ở 16kHz PCM16 chỉ khoảng 2048 bytes, nên mỗi copy riêng lẻ nhỏ. Nhưng ở ~15–16 frame/s, chi phí cumulative + GC vẫn đáng kể hơn nếu pipeline chạy liên tục hàng giờ.

## 6.2 Backend ingress

`AudioBuffer.write_bytes()`:

```python
raw_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
data = raw_int16.astype(np.float32) / 32768.0
return self.write(data)
```

`np.frombuffer()` là zero-copy view. Nhưng `astype(float32)` tạo allocation mới.

Sau đó `write()` lại copy `data` vào ring buffer:

```python
self._buffer[...] = audio_data
```

Do đó:

```text
PCM bytes
  ↓ view int16                 zero-copy
  ↓ astype(float32)            ALLOC
  ↓ ring-buffer assignment     COPY
```

Đây là hợp lý nếu toàn bộ backend pipeline cần float32, nhưng chưa tối ưu memory path.

## 6.3 Preview path

`get_slice()` luôn tạo `np.empty(requested_len)` và copy từ ring buffer.

Mỗi preview vì thế tạo một full contiguous NumPy array mới.

Chất lượng-preserving optimization:

- giữ ring buffer như storage chính;
- thêm API `get_contiguous_views()` trả 1 hoặc 2 ndarray views;
- chỉ materialize contiguous array khi binding native thực sự yêu cầu;
- nếu native binding nhận được 2 segment thì có thể feed từng segment nếu streaming API cho phép.

Nếu binding bắt buộc contiguous input thì vẫn còn lợi ích ở việc tránh copy giữa intermediate stages, nhưng không thể loại bỏ copy cuối.

---

# 7. P2 — `AudioBuffer.clear()` Zeroes 60 Seconds Every Reset

`backend/core/audio_buffer.py:174-179`:

```python
self._buffer.fill(0)
self._total_written = 0
self._dropped_samples_count = 0
```

Buffer mặc định 60 giây float32:

```text
960,000 samples × 4 bytes = 3.84 MB
```

Mỗi `reset_stream()` / cleanup lại ghi 3.84 MB zero vào memory.

Điều này **không giúp correctness** nếu mọi read đều được kiểm soát bởi `_total_written` và earliest available.

### Khuyến nghị

Chỉ cần:

```python
self._total_written = 0
self._dropped_samples_count = 0
```

và giữ một generation/valid-range marker.

Chỉ zero-fill khi có yêu cầu security/data-scrubbing thực sự.

Lợi ích:

- giảm memory bandwidth;
- giảm cache pollution;
- reset latency thấp hơn;
- đặc biệt hữu ích khi user seek liên tục.

---

# 8. P1 — Global Locks / Single-Lane Architecture

## 8.1 ASR

`backend/asr/engine.py` có:

```python
_shared_lock
_infer_lock
```

và `_infer_lock` bọc toàn bộ native inference.

Lock này hiện là **đúng về safety** nếu binding/native session chỉ cho phép một `run()` in-flight trên shared session.

Vì vậy không nên bỏ lock chỉ để benchmark nhanh hơn.

Nhưng consequence architecture là:

```text
Session A ASR
      ↓
   _infer_lock
      ↓
Session B ASR → WAIT
Session C ASR → WAIT
```

Đây là process-wide serialization.

Nếu product chỉ có một session thì không cần sửa.

Nếu sau này multi-session:

> shared model weights + per-session inference context/state

sẽ là kiến trúc cần hướng tới, thay vì shared mutable session.

## 8.2 Translation

`_TRANS_EXECUTOR = ThreadPoolExecutor(max_workers=1)`.

Đây là serialization process-wide.

Thêm vào đó `_infer_lock` bọc toàn bộ streaming generation:

```python
with self.__class__._infer_lock:
    for chunk in llm(..., stream=True):
        ...
        yield acc
```

Trong current architecture executor 1 worker đã serialization rồi; lock thứ hai tạo complexity nhưng không mang thêm throughput.

Nếu giữ một worker duy nhất, lock có thể được giữ như safety boundary nhưng nên được xem là redundant và documented rõ.

Nếu chuyển sang multi-session thì cần model-context isolation, không chỉ tăng `max_workers`.

## 8.3 VAD

`_VAD_EXECUTOR = ThreadPoolExecutor(max_workers=1)`.

Một session: tốt cho deterministic latency.  
Nhiều session: mọi session chia chung một VAD lane.

Do VAD frame chỉ ~60ms hoặc chunk-size tương tự, nếu model inference có spike thì toàn pipeline ingress có thể trễ theo.

---

# 9. P1 — Translation Queue Backlog and Dropped Final Translations

`handler.py` khi translation queue đầy:

```python
except asyncio.QueueFull:
    logger.warning(...)
    metrics_collector.increment_counter("queue.translation_dropped")
```

Điều này tạo ra một policy:

```text
ASR final
   ↓
queue full
   ↓
translation request dropped
```

Về performance đây giúp memory không tăng vô hạn.

Nhưng về product semantics, **final subtitle có thể không được dịch**.

Nếu mục tiêu là “giảm latency/RAM nhưng không giảm quality”, đây là vùng cần thay policy, không chỉ tối ưu.

## 9.1 Policy đề xuất

Không drop final translation. Có thể drop/replace partial translation.

Ví dụ:

```text
Priority 0: final translation
Priority 1: latest partial translation
Priority 2: stale partials
```

Queue chỉ cần giữ:

- 1 final đang xử lý;
- 1 final kế tiếp;
- latest partial.

Đây là semantic backpressure tốt hơn queue FIFO mù.

---

# 10. P1 — WebSocket Head-of-Line Blocking

`backend/ws/connection.py` sử dụng một `_send_lock` cho cả:

- JSON subtitle;
- translation partial;
- binary TTS.

Nếu đang:

```text
await websocket.send_bytes(large_tts_frame)
```

thì subtitle/translation phải chờ lock.

Với mạng/local backend thông thường không nghiêm trọng, nhưng trong điều kiện browser main-thread load, service-worker scheduling, extension bridge hoặc slow receiver, đây có thể biến thành latency spike.

## Khuyến nghị kiến trúc

Thay vì từng worker tự send:

```text
ASR ─────┐
TRANS ───┼→ priority outbound queue → sender task → websocket
TTS ─────┘
```

Sender ưu tiên:

```text
subtitle final > subtitle preview > translation partial > TTS
```

Không nhất thiết cần nhiều socket.

Chỉ cần **một outbound scheduler** có priority và bounded queue là đã giảm HOL blocking.

---

# 11. P1 — MetricsCollector Mutex in Hot Path

`backend/core/metrics.py` mọi lần:

```python
record_latency()
record_gauge()
increment_counter()
```

đều lấy `_lock`.

Trong realtime pipeline, metrics được ghi ở nhiều nơi:

```text
VAD chunk/frame
ASR preview
ASR commit
translation queue
translation infer
TTS queue
TTS synthesis
WS send
```

Một session chưa chắc thấy vấn đề lớn. Nhưng metrics collector hiện là process-wide serialization point.

## Khuyến nghị

Giữ API hiện tại nhưng backend implementation chuyển sang:

```text
thread-local counters/rings
       ↓
periodic merge
       ↓
snapshot
```

hoặc giảm tần suất metric cho hot path:

```text
sample 1/N frames
```

Đặc biệt không cần ghi gauge queue depth sau **mọi item** nếu chỉ cần detect backlog.

Ví dụ chỉ ghi khi:

```text
qsize changes
OR
qsize > high-watermark
OR
100ms elapsed
```

---

# 12. P1 — Translation Model Load Holds Shared Lock Too Long

`backend/translation/engine.py:146-165`:

```python
with self.__class__._shared_lock:
    ...
    llm, prompt_strategy = self._build_llm(...)
    ...
```

`_build_llm()` là heavy operation: đọc GGUF, mmap/load tensors, GPU layer setup, initialization.

Vì lock được giữ trong suốt operation, bất kỳ code nào cần `_shared_lock` sẽ phải chờ.

Revision `reconfigure()` lại làm đúng hướng:

```text
build new model outside shared_lock
       ↓
swap under short lock
       ↓
release old
```

Nên `load_model()` cũng nên theo pattern đó.

### Correct pattern

```python
with shared_lock:
    if already_loaded:
        return

new_llm = build_llm(...)       # outside lock

with shared_lock:
    if another thread installed model meanwhile:
        release new_llm
    else:
        swap(new_llm)
```

Đây vừa giảm contention vừa giữ safety.

---

# 13. P1 — TTS Conversion and CPU Post-Processing

`backend/tts/audio_processor.py`:

```python
arr = item.detach().cpu().numpy().astype(np.float32)
```

Đây có thể gồm:

1. GPU synchronization;
2. device-to-host copy;
3. NumPy view creation;
4. dtype conversion copy nếu không phải float32.

TTS cuối cùng vẫn cần CPU bytes để encode WAV, vì vậy D2H về bản chất là khó tránh.

Tối ưu hợp lý:

- đảm bảo model output đã là float32 contiguous nếu API hỗ trợ;
- tránh `.astype(np.float32)` nếu dtype đã đúng;
- tránh concatenate nhiều segment nếu model thường trả single tensor.

Hiện tại `np.concatenate(converted)` sẽ copy toàn bộ output một lần nữa nếu có nhiều items.

## Normalize audio

```python
max_peak = np.max(np.abs(audio))
normalized = (audio / max_peak) * target
np.clip(normalized, ...).astype(np.float32)
```

Có nhiều temporary arrays.

Có thể giảm xuống bằng in-place operations nếu lifetime an toàn.

Tuy nhiên đây là micro-optimization hơn là root bottleneck so với TTS model inference.

---

# 14. P2 — Phase Vocoder Can Cause CPU Latency Spike

`apply_time_stretch()` thực hiện:

```text
STFT
→ interpolation
→ angle
→ diff
→ phase unwrap-like arithmetic
→ exp
→ ISTFT
```

toàn bộ trên CPU.

Khi `speed != 1`, đây có thể là phần hậu xử lý đáng kể so với model generation đối với câu dài.

Nếu mặc định speed=1 thì path này dormant.

Khuyến nghị:

- profile thực tế theo utterance length;
- nếu speed adjustment được dùng thường xuyên, cân nhắc implementation GPU hoặc streaming DSP chuyên dụng;
- không chuyển sang browser playbackRate nếu yêu cầu giữ pitch/quality như hiện tại.

---

# 15. P2 — TTS Binary Path Still Has Copies

Backend:

```text
audio_np
 ↓ encode_wav_bytes → bytes
 ↓ make_tts_binary_frame → new frame
```

Đã bỏ base64, rất tốt, nhưng vẫn có ít nhất một full-frame copy để prepend protocol header.

Có thể tối ưu bằng buffer construction trực tiếp nếu serializer hỗ trợ `bytearray`/memoryview mà không cần intermediate `bytes`.

Đừng ưu tiên việc này trước khi đo TTS frame size và CPU profile.

## Browser TTS path

`content-script.js`:

```js
buf.slice(7 + jsonLen)
```

sau đó `tts-player.js` lại có path copy buffer trước khi decode.

Như vậy WAV payload có thể đi qua **2 full copies** bên client.

Khuyến nghị:

```text
single slice / typed-array view
        ↓
decodeAudioData / consumer
```

và đo lại browser heap allocation.

---

# 16. P2 — Capture Chunk Size / Latency Floor

AudioWorklet hiện dùng:

```text
chunkSize = 1024 samples @ 16 kHz
≈ 64 ms
```

VAD frame config có thể nhỏ hơn, nhưng capture ingress đã có một latency floor khoảng 64ms trước khi backend nhận được chunk đầy đủ.

Đây không phải bottleneck throughput, nhưng là latency floor.

Nếu cần giảm latency:

```text
1024 → 512 samples ≈ 32ms
```

Tuy nhiên packet/message frequency tăng gấp đôi.

Do đó nên benchmark:

```text
capture latency
CPU/message overhead
VAD queue cost
WS packet rate
```

và chỉ giảm chunk size nếu end-to-end latency thực sự hưởng lợi.

---

# 17. P2 — ScriptProcessor Fallback Is Much More Expensive

Fallback path tạo `Float32Array`, resample bằng linear interpolation trên JavaScript main thread và tạo `Int16Array` bằng loop JS.

Đây là CPU work khá nặng so với AudioWorklet.

Worklet path đã được dùng hiện tại, nên không xem đây là production hot path bình thường.

Nhưng watchdog hiện chờ tới **2.5s** trước khi fallback. Trong trường hợp Worklet lỗi/liveness issue, user có thể chịu startup/capture gap lớn.

Nếu browser compatibility cho phép, có thể giảm watchdog sau khi telemetry chứng minh false-positive thấp.

---

# 18. P2 — Event Loop / Native Cancellation Boundary

`_infer_with_watchdog()` dùng:

```python
loop.run_in_executor(...)
```

và `asyncio.shield(fut)`.

Đây là lựa chọn hợp lý vì cancellation của asyncio không tự giết native thread.

Tuy nhiên có một subtle risk:

```python
cancel_inference()
    ↓
shared_session.cancel()
```

`cancel_inference()` đọc `_shared_session` mà không lấy `_shared_lock` trước khi gọi cancel.

Trong current one-session architecture, hazard nhỏ hơn. Nhưng trong lúc hot-swap/unload có thể có race giữa:

```text
thread A: cancel_inference()
thread B: unload/swap shared session
```

Nên snapshot/cancel handle atomically:

```python
with _shared_lock:
    session = _shared_session
```

sau đó gọi cancel trên local reference.

Không giữ lock trong native call.

---

# 19. P1 — Model Hot-Swap VRAM Peak

Architecture hiện chủ động:

```text
new model load
   ↓
new model ready
   ↓
swap
   ↓
old model release
```

Đây là pattern đúng về availability.

Nhưng cost là peak memory:

```text
old model + new model + allocator overhead
```

đồng thời translation/TTS/ASR có thể vẫn resident.

Trên GPU 16GB, đây có thể trở thành OOM spike khi đổi sang model lớn.

Khuyến nghị:

1. preflight estimate VRAM;
2. expose `required_vram` trong model registry;
3. refuse/soft-fail switch nếu headroom không đủ;
4. hoặc có policy staged swap theo workload.

Không nên chỉ `empty_cache()` rồi hy vọng đủ VRAM; allocator cache không giải quyết tổng VRAM requirement của hai model đồng thời.

---

# 20. P2 — Executor Contention Around TTS / Default Async Executor

`tts.synthesize_clone_bytes()` sử dụng:

```python
await asyncio.to_thread(...)
```

tức là default asyncio threadpool, không phải dedicated executor.

Các thao tác background khác cũng dùng `asyncio.to_thread()`:

- model prewarm;
- model load;
- TTS prewarm;
- miscellaneous blocking helpers.

Khi startup/config change xảy ra đồng thời, có thể tạo contention trong default executor.

Khuyến nghị:

- dành một executor riêng cho TTS blocking work;
- một executor riêng cho model lifecycle/loading;
- không nhất thiết tăng tổng thread count; mục tiêu là cô lập workload.

---

# 21. P2 — Repeated TTS Prewarm on Config Update

`handler.py` và REST config path có thể gọi:

```python
get_tts_engine().prewarm()
```

mỗi khi TTS enable/config event xảy ra.

`load_model()` có guard `_is_loaded`, nên không load lại model, nhưng vẫn tạo background coroutine/task.

Với popup gửi nhiều config changes liên tiếp, có thể có nhiều prewarm calls nối nhau.

Khuyến nghị coalesce:

```text
TTS prewarm already pending/running?
   ↓ yes → reuse same Future/Task
```

Một `AsyncOnce` pattern sẽ tốt hơn việc tạo task nhiều lần.

---

# 22. P2 — `drain_queues()` Semantics at Cleanup

Trong `handle_ws()`:

```text
cancel workers
await gather(workers)
drain_queues()
```

nhưng queue workers đã bị cancel trước khi drain.

Nếu queue còn item, `queue.join()` không còn consumer để decrement unfinished count. Snapshot tránh deadlock bằng timeout ngắn, nhưng về semantics thì `drain_queues()` sau worker cancellation không có nhiều giá trị.

Pattern sạch hơn:

```text
stop intake
cancel workers
clear queued items + task_done()
await worker shutdown
```

hoặc drain trước khi cancel nếu muốn flush final outputs.

Vì mục tiêu realtime, cleanup nên ưu tiên discard nhanh; do đó clear explicit thường phù hợp hơn.

---

# 23. P2 — Metrics/Logging Can Distort Very Short Latency Measurements

Metrics và logger đã được cải thiện đáng kể, nhưng pipeline vẫn đo nhiều stage ở granularity cao.

Ví dụ một preview cycle có thể làm:

```text
perf_counter
metrics lock
queue gauge
logging formatting
perf_counter
metrics lock
```

Với một inference ~100ms, overhead không lớn.

Nhưng với VAD frame 20–60ms, fixed overhead có thể chiếm tỷ lệ đáng kể.

Khuyến nghị benchmark cả:

```text
metrics enabled
vs
metrics disabled/sampled
```

Nếu p95 khác đáng kể thì metrics đang tham gia hot path quá sâu.

---

# 24. Config / Dead-Code / Complexity Audit

Một số field trong config không còn thấy consumer rõ ràng trong production source snapshot, đáng chú ý:

- `max_payload_bytes`
- `preview_reuse_max_delta_sec`
- `normalize_*` một phần đã được consumer ở `SpeechNormalizer`, nên không được xếp dead chỉ vì simple grep;
- `request_timestamps` consumer trực tiếp trong ASR nhưng low-level test usage ít;
- `n_ctx/n_batch/n_threads` hiện consumer đúng trong translation engine;
- `AudioBufferConfig.capacity_sec`, `max_speech_segment_sec` cần tránh để configuration tồn tại nếu engine hard-code giá trị khác.

Đặc biệt cần chú ý:

```python
self.audio_buffer = CircularAudioBuffer(sample_rate=16000, capacity_sec=60.0)
```

trong ASR engine.

Nó hard-code `60.0` thay vì đọc `config.audio_buffer.capacity_sec`.

Điều này làm config `capacity_sec` không thực sự là source of truth.

### Khuyến nghị

Chọn một source of truth:

```python
CircularAudioBuffer(
    sample_rate=config.audio_buffer.sample_rate,
    capacity_sec=config.audio_buffer.capacity_sec,
)
```

và test config effectiveness.

---

# 25. Lock / Mutex Map

| Lock | Scope | Hiện trạng | Đánh giá |
|---|---|---|---|
| ASR `_infer_lock` | process-wide native inference | giữ toàn bộ infer | cần cho shared session hiện tại |
| ASR `_shared_lock` | model/session metadata | tương đối nhỏ | cần |
| Translation `_infer_lock` | process-wide generation | giữ stream full duration | safe nhưng serialization |
| Translation `_shared_lock` | shared LLM | load_model giữ quá lâu | cần refactor |
| Translation `_switch_lock` | hot-swap | đúng hướng | giữ |
| TTS `_init_lock` | lifecycle | load/warm | cần |
| TTS `_infer_lock` | generation | full generation | cần với shared model |
| Metrics `_lock` | mọi metric | hot path global | tối ưu nên làm |
| Audio buffer lock | read/write ring | mỗi get/write | phù hợp single-writer, nhưng copy cost lớn |
| WS `_send_lock` | all outbound frames | HOL blocking | nên thay bằng outbound scheduler |

Không thấy deadlock cycle rõ ràng trong code hiện tại ở ASR vì comment + implementation duy trì thứ tự:

```text
_infer_lock → _shared_lock
```

Nhưng Translation `load_model()` và `reconfigure()` có pattern lock khác nhau; nên chuẩn hóa lock ordering trong toàn module bằng document + assertion/test.

---

# 26. Queue / Producer-Consumer Map

```text
Audio browser
  ↓ WS
VAD
  ↓
ASR preview/commit
  ↓ translation_queue(max 4)
Translation worker(1)
  ↓ tts_queue(max 4)
TTS worker(1)
  ↓ WS sender lock
Browser
```

## Điểm mất cân bằng tự nhiên

### ASR → Translation

Nếu ASR commit rate > translation service rate:

```text
translation queue depth ↑
```

Sau 4 item => final translation drop.

### Translation → TTS

Nếu TTS generation > sentence arrival rate:

```text
tts queue depth ↑
```

Sau 4 item => TTS drop/degradation.

### Browser → Backend

Nếu network/backend chậm:

```text
WebSocket bufferedAmount ↑
```

Extension hard pause.

### Vấn đề architecture

Đây là một **pipeline with bounded queues nhưng chưa có end-to-end admission control**.

Mỗi stage tự quyết định drop/merge riêng.

Khuyến nghị thêm global pipeline state:

```text
HEALTHY
DEGRADED
SATURATED
RECOVERING
```

để các stage cùng hiểu saturation.

Ví dụ khi translation backlog cao:

```text
ASR preview cadence ↓
partial translation ↓
TTS partial ↓
final commit vẫn giữ
```

Đây là adaptive backpressure chất lượng tốt hơn drop từng tầng độc lập.

---

# 27. CPU Bottleneck Summary

## High

1. repeated ASR preview inference;
2. JS fallback resampling nếu Worklet fallback;
3. TTS Phase Vocoder khi speed != 1;
4. repeated NumPy conversion/copy;
5. metrics lock + frequent metrics updates trong hot path.

## Medium

6. VAD PCM conversion int16 → float32;
7. audio packet construction/copy;
8. serialization/JSON for frequent partials;
9. WAV encoding.

## Low

10. `AudioBuffer.clear()` 3.84MB zero fill;
11. small dedup/list allocations.

---

# 28. GPU Bottleneck Summary

## ASR

Main issue: repeated inference trên overlapping audio window.

Nếu native model execution là dominant stage, việc giảm số inference calls sẽ có tác động lớn hơn nhiều so với micro-optimizing Python.

## Translation

Một worker → một generation tại một thời điểm.

n_ctx/n_batch/n_threads đã được expose, tốt; cần profile để chọn tối ưu theo model thay vì dùng một set fixed cho mọi model.

## TTS

Một generation tại một thời điểm; model resident trên GPU.

Nếu TTS enabled nhưng output không liên tục, VRAM vẫn bị giữ.

Đây là trade-off latency vs VRAM, không phải leak.

---

# 29. RAM / VRAM Risk Summary

## RAM

Các source rõ ràng tạo allocations thường xuyên:

```text
incoming bytes → float32
preview slice → float32
normalization output
translation accumulated strings
TTS audio conversion
WAV bytes
binary frame bytes
browser clones
```

Không thấy Python-side unbounded queue ở pipeline chính hiện tại. Đây là điểm tốt.

## Native memory growth

README/report đi kèm có mô tả native/Vulkan memory growth lớn trong một số pathological inference. Vì native source không có trong ZIP, phát hiện này **chưa thể source-verify**.

Guard hiện tại là sensible containment:

```text
watchdog
RSS runaway detection
cancel
session recycle
```

Nhưng đây là mitigation chứ không phải root-cause fix.

---

# 30. Memory Leak vs Legitimate Growth

Cần phân biệt 4 loại:

### A. Intentional resident memory

- ASR model
- Translation model
- TTS model
- VAD models/cache

Không phải leak.

### B. allocator cache

PyTorch CUDA allocator / ggml allocator có thể giữ cached blocks.

Không nên đánh đồng reserved memory với leak.

### C. bounded temporary growth

Preview slices, WAV bytes, queue items.

Đây là normal transient allocation nhưng có thể gây GC/jitter.

### D. pathological native growth

Nếu RSS tăng liên tục trong cùng một inference mà output không tương xứng, đây mới là leak/native working-set abnormality.

Cần measurement:

```text
RSS
Private Bytes
Committed Bytes
CUDA allocated
CUDA reserved
Vulkan heap usage
native session lifetime
```

đồng thời sample theo từng inference.

---

# 31. Serialization / Deserialization Review

## Audio

Binary audio protocol là hướng đúng và tốt hơn base64.

Base64 path vẫn giữ để backward compatibility, không cần xóa ngay.

## JSON

Compact payload đã được implement.

Partial translation hiện vẫn truyền `acc` tích lũy thay vì delta.

Ví dụ:

```text
"xin"
"xin chào"
"xin chào bạn"
"xin chào bạn nhé"
```

Thay vì:

```text
"xin"
" chào"
" bạn"
" nhé"
```

Full-prefix streaming làm:

- bytes gửi tăng theo tổng output;
- JSON serialization lặp lại prefix;
- client parse lại full string.

Nếu protocol có thể chuyển sang delta-token event, đây là tối ưu network/CPU tốt mà không đổi chất lượng.

Cần xem client/UI có phụ thuộc full prefix hay không trước khi thay.

---

# 32. Repeated Work Hotspots

Danh sách những phép tính đang lặp:

### ASR

- infer lại overlapping audio;
- normalize lại cùng audio preview;
- `get_slice` copy lại vùng audio.

### Translation

- build accumulated prefix `acc += piece`;
- gửi full accumulated prefix;
- parse/serialize partial nhiều lần.

### TTS

- model output → CPU;
- normalize;
- encode WAV;
- binary frame assembly.

### Browser

- packet construction;
- clone through extension bridge;
- TTS payload slice/copy.

---

# 33. Sequential Sections That Could Be Parallelized

Không nên parallelize bừa bãi các model inference vì GPU/RAM và shared model state.

## Safe parallelism

### Startup

`main.lifespan()` đã prewarm ASR + Translation + VAD song song. Đây là đúng hướng.

### VAD engines khác

Background prewarm đã được tách khỏi startup blocking path. Tốt.

### Audio preprocessing

Nếu cần, một số CPU preprocessing có thể chạy concurrently với downstream inference, nhưng phải tránh queue tăng vô hạn.

### Outbound serialization

Có thể serialize payload trước, sau đó gửi qua dedicated sender.

## Không nên parallelize

- hai ASR inference cùng shared session;
- hai TTS generation cùng singleton;
- hai Translation generation cùng shared LLM nếu backend context không isolate.

---

# 34. Latency Budget

Một target thực tế nên được biểu diễn như:

```text
Capture
 + ingress
 + VAD
 + ASR wait
 + ASR infer
 + commit decision
 + translation queue wait
 + translation first token
 + TTS queue wait
 + TTS synthesis
 + WS send
 + browser decode/play
```

Trong hệ thống hiện tại, ASR preview repeated inference có thể làm queueing delay xuất hiện ngay cả khi individual model latency đẹp.

Do đó **p95/p99 queue_wait_ms** quan trọng không kém model infer_ms.

Metric nên theo dõi:

```text
ASR:
  preview_wait_ms
  preview_infer_ms
  commit_wait_ms
  commit_infer_ms

Translation:
  queue_wait_ms
  TTFT
  total_generation_ms

TTS:
  queue_wait_ms
  synthesis_ms
  encode_ms

WS:
  send_wait_ms
  send_bytes

Backpressure:
  paused_ms
  dropped_frames
```

---

# 35. Performance Optimizations That Preserve Quality

Ưu tiên các thay đổi sau vì **không cần đổi model output quality**:

## Tier A

1. Fix HARD backpressure auto-resume.
2. Incremental ASR state/caching nếu native API hỗ trợ.
3. Nếu chưa có incremental ASR: event-driven preview suppression khi chưa đủ new audio.
4. Bỏ zero-fill 60s trong `AudioBuffer.clear()`.
5. Tránh duplicate browser TTS buffer copies.
6. Move translation model build outside shared lock.
7. Dedicated TTS executor.
8. Priority outbound queue thay vì một send lock cho mọi message.

## Tier B

9. Reduce metrics lock frequency.
10. Avoid `astype(float32)` when already float32.
11. Make `AudioBufferConfig` thực sự là source of truth.
12. Coalesce repeated TTS prewarm.
13. Change translation partial protocol toward delta events.

## Tier C

14. In-place audio normalize.
15. Reduce binary frame assembly copies.
16. Tune capture chunk size.
17. Optimize logger formatting in very hot paths.

---

# 36. RAM/VRAM Optimizations Without Quality Loss

### RAM

- zero-copy views wherever native API permits;
- avoid duplicate WAV buffers;
- avoid accumulated translation prefix copies;
- bounded queues with semantic priority;
- release temporary arrays eagerly where Python lifetime matters.

### VRAM

- keep only active VAD models resident if all-engine prewarm is not required;
- unload TTS when feature disabled for long periods, but only if reload latency is acceptable;
- preflight model-switch VRAM;
- use model-specific GPU layer policy for translation;
- don't interpret allocator reserved memory as leak.

### Important

Không giảm VRAM bằng:

- reducing model precision;
- reducing context;
- reducing TTS steps;
- reducing ASR window;

nếu các thay đổi đó có thể làm giảm quality.

---

# 37. Multi-Session Scaling Assessment

Current design is intentionally not a multi-session architecture.

### Hard serialization points

```text
ASR shared session + infer lock
Translation shared LLM + executor 1
TTS singleton + infer lock
VAD executor 1
Metrics singleton lock
```

Nếu chạy 4 sessions trong cùng process:

```text
session 1 waits
session 2 waits
session 3 waits
session 4 waits
```

và GPU utilization không tự động tăng linearly.

## Architecture đề xuất nếu sau này cần multi-session

```text
                 ┌─ Session A state ─┐
Shared model ────┼─ Session B state ─┼─ scheduler
                 ├─ Session C state ─┤
                 └─ Session D state ─┘
```

Nhưng chỉ làm sau khi requirement thay đổi. Với scope 1 session hiện tại, tăng complexity để support N session có thể không đáng.

---

# 38. Test / Quality Engineering Gaps

Test suite khá mạnh về regression behavior, nhưng hiện có 6 failure.

### Cần sửa ngay

1. fixture path của `wav_test`;
2. CUDA/provider test phải tolerate môi trường không có native provider;
3. date-dependent logger test phải dùng injected/fixed clock;
4. logger module-tag test còn 3 failures;
5. translation tests phải skip hoặc fake `llama-cpp-python` đúng contract;
6. hotswap REST busy test phải thống nhất contract “409 ngay” hay “queue background”.

### Performance tests còn thiếu

Cần thêm automated test cho:

- no allocation growth after 1000 preview ticks;
- bounded translation/TTS queue;
- backpressure HARD → automatic resume;
- outbound queue priority;
- metrics disabled vs enabled overhead;
- repeated seek/reset throughput;
- model hot-swap peak RSS/VRAM;
- TTS binary copy count nếu có browser benchmark harness.

---

# 39. Recommended Benchmark Matrix

## Scenario A — Quiet speech

```text
1–2 words
short pauses
```

Measure:

- first preview latency;
- false commits;
- CPU idle.

## Scenario B — Long sentence

```text
5–10s continuous speech
```

Measure:

- number of preview calls;
- total ASR compute;
- duplicated audio seconds processed;
- p95 preview latency.

## Scenario C — Rapid speech

Measure queue backlog and commit accuracy.

## Scenario D — Translation-heavy

Long source sentences, high token output.

Measure:

- translation queue depth;
- TTFT;
- final-drop rate.

## Scenario E — TTS-heavy

Many short final sentences.

Measure:

- TTS queue depth;
- synthesis RTF;
- WS send HOL blocking.

## Scenario F — Network congestion

Artificially throttle receiver/network.

Measure:

- bufferedAmount;
- HARD pause;
- resume latency;
- dropped frame count;
- eventual caption recovery.

## Scenario G — Seek stress

Rapid seeking every 0.5–2 sec.

Measure:

- reset latency;
- queue cleanup;
- old-audio contamination;
- allocation spikes.

---

# 40. Instrumentation Recommended for Next Revision

Thêm 3 counters rất quan trọng:

```text
asr.audio_seconds_processed_preview
asr.audio_seconds_unique_preview
asr.preview_recompute_ratio
```

Trong đó:

```text
recompute_ratio = processed / unique_audio
```

Nếu ratio là 8×, nghĩa là pipeline đang xử lý lại trung bình 8 lần cùng một lượng audio.

Đây sẽ là metric tốt nhất để chứng minh incremental ASR có tác dụng.

Thêm:

```text
translation.final_dropped
translation.partial_dropped
translation.queue_wait_p95

tts.final_dropped
tts.queue_wait_p95

audio.browser_copy_estimate
ws.buffered_amount_p95
ws.backpressure_paused_ms
```

---

# 41. Concrete Refactoring Roadmap

## Phase R3.1 — Correctness / Backpressure

**Mục tiêu:** không đổi model/quality.

1. Fix HARD backpressure auto-resume.
2. Add backpressure hysteresis.
3. Separate final vs partial queue semantics.
4. Fix `drain_queues()` cleanup ordering.
5. Fix/correct six failing tests.

## Phase R3.2 — Memory / Copy Reduction

1. `AudioBuffer.clear()` no zero-fill.
2. `AudioBufferConfig` becomes source of truth.
3. Remove unnecessary int16→float32 copy when possible.
4. Browser TTS single-copy path.
5. Review extension structured-clone boundaries.
6. In-place TTS normalization where safe.

## Phase R3.3 — Concurrency / Scheduling

1. translation `_build_llm()` outside lock;
2. dedicated TTS executor;
3. outbound priority sender;
4. metric sampling/thread-local collector.

## Phase R3.4 — ASR Core Performance

1. measure recompute ratio;
2. prototype incremental native streaming state;
3. compare WER/CER against current commit path;
4. only enable reuse where quality regression is proven absent;
5. benchmark preview cadence/adaptive policy.

## Phase R3.5 — Native Investigation

This phase requires native `transcribe.cpp` source or an upstream commit/version exact match.

Trace:

```text
transcribe session creation
scheduler creation
compute context
Vulkan buffers
H2D/D2H
allocator
threadpool
cancel path
session close
```

Use:

- RSS / Private Bytes;
- GPU memory telemetry;
- Vulkan validation/profiling if available;
- CPU sampling profiler;
- allocation profiler.

---

# 42. Final Assessment

## Current architecture strengths

- clear pipeline separation;
- sensible singleton/resource strategy for one-session product scope;
- good use of bounded queues;
- commit backlog merge instead of silent drop;
- VAD inference outside state lock;
- ASR watchdog and native recycle guard;
- binary TTS protocol;
- compact payload support;
- prewarm path separated from request hot path;
- decent benchmark/regression discipline.

## Main remaining architectural debt

The remaining performance problem is no longer “Python is slow” in the generic sense.

It is primarily:

```text
RECOMPUTE
+ COPY
+ SERIALIZATION
+ SINGLE-LANE RESOURCE OWNERSHIP
```

with the largest individual issue being **repeated ASR inference over overlapping preview audio**.

The second most serious issue is **browser backpressure recovery correctness**.

After those, the next meaningful gains come from eliminating data copies, reducing process-wide locks, and introducing priority scheduling between realtime subtitle traffic and lower-priority audio/TTS traffic.

---

# 43. Priority Table

| Priority | Action | Why |
|---|---|---|
| **P0** | Auto-resume HARD backpressure | Prevents capture getting stuck |
| **P0** | Measure + attack repeated ASR preview recompute | Largest compute/latency waste |
| **P1** | Semantic queue policy: never drop final translation | Preserves correctness under saturation |
| **P1** | Outbound priority scheduler | Removes WS head-of-line blocking |
| **P1** | Translation load outside shared lock | Removes long lock hold |
| **P1** | Reduce audio copy chain | Lower CPU/RAM/latency jitter |
| **P1** | Metrics lock sampling/thread-local | Lower hot-path contention |
| **P1** | TTS dedicated executor | Isolate blocking CPU work |
| **P2** | Remove `clear()` zero-fill | Cheap memory-bandwidth win |
| **P2** | TTS/browser buffer copy reduction | Lower GC/memory traffic |
| **P2** | Translation delta partial protocol | Lower network/serialization cost |
| **P2** | Capture chunk-size tuning | Latency vs message overhead trade-off |
| **P2** | Clean dead/config inconsistency | Simpler architecture |
| **R3.5** | Native transcribe.cpp source audit | Needed for true GPU/native leak root cause |

---

# 44. Bottom Line

Nếu giữ đúng constraint hiện tại là **1 session / 1 video / 1 audio stream**, không cần biến hệ thống thành distributed/multi-session architecture.

Nên tập trung vào ba việc:

```text
1. giảm số lần ASR xử lý lại audio;
2. giảm số lần copy audio/WAV;
3. làm backpressure + outbound scheduling có priority và recovery đúng.
```

Ba nhóm này có khả năng cải thiện latency/CPU/RAM/throughput mà **không cần giảm model size, precision, ASR quality, translation quality hay TTS quality**.

Nếu requirement sau này chuyển sang nhiều session đồng thời, khi đó cần refactor resource ownership sang per-session state + shared immutable model weights + explicit GPU scheduler. Không nên chỉ tăng thread pool.

---

## Appendix A — Files Reviewed (representative hot-path list)

### Backend

- `backend/main.py`
- `backend/config.py`
- `backend/asr/engine.py`
- `backend/asr/adapters.py`
- `backend/asr/registry.py`
- `backend/asr/text_cleaner.py`
- `backend/core/audio_buffer.py`
- `backend/core/commit_manager.py`
- `backend/core/metrics.py`
- `backend/core/normalizer.py`
- `backend/translation/engine.py`
- `backend/translation/hotswap.py`
- `backend/translation/context.py`
- `backend/translation/dedup.py`
- `backend/tts/engine.py`
- `backend/tts/audio_processor.py`
- `backend/tts/dedup.py`
- `backend/tts/voice_manager.py`
- `backend/vad/processor.py`
- `backend/vad/engines/*`
- `backend/ws/handler.py`
- `backend/ws/session.py`
- `backend/ws/connection.py`
- `backend/ws/protocol.py`
- `backend/ws/serializers.py`
- `backend/utils/logger.py`
- `backend/utils/mem_guard.py`
- `backend/utils/stall_watchdog.py`
- `backend/utils/model_download.py`

### Firefox Extension

- `extension_firefox/background/service-worker.js`
- `extension_firefox/content/content-script.js`
- `extension_firefox/lib/audio-capture.js`
- `extension_firefox/lib/audio-processor.js`
- `extension_firefox/lib/frame-builder.js`
- `extension_firefox/lib/ws-client.js`
- `extension_firefox/lib/tts-player.js`
- `extension_firefox/lib/subtitle-renderer.js`
- `extension_firefox/lib/subtitle-policy.js`
- popup and overlay components

### Tests

- `backend/tests/test_01_core_audio.py` … `test_22_translation_model_download.py`
- JavaScript harnesses under `backend/tests/js/`

---

## Appendix B — Validation Commands Run

```text
python -m compileall -q backend
→ PASS

node --check <all extension JS files>
→ PASS

pytest -q
→ 268 passed / 6 failed
```

The six test failures are documented in §2.2 and should be resolved before treating the suite as a clean regression gate.

---

## Appendix C — Scope Caveat About GitHub vs ZIP

The live GitHub repository page and attached ZIP snapshot are not byte-for-byte identical in tree layout. The current GitHub page exposes a `backend_cpp/` directory, while the uploaded snapshot contains `backend/` and does not contain the native `external/transcribe.cpp` source.

Therefore:

- **line-level findings in this report refer to the attached ZIP snapshot**;
- GitHub was used only as repository/README/context cross-check;
- native transcribe.cpp internals are explicitly marked as unverified where relevant.

This distinction is important because otherwise a finding could accidentally be attributed to a different revision than the code actually reviewed.

---

**End of audit.**
