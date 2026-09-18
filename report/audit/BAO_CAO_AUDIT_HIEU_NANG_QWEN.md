# BÁO CÁO AUDIT HIỆU NĂNG — LẦN ĐỌC ĐỘC LẬP bởi QWEN
### (`/backend` + `/extension_firefox`, tính năng + tài nguyên + độ đúng đắn đồng bộ)

- **Người thực hiện**: Senior Software Architect / Performance Engineer / Code Reviewer (mô hình QWEN), đọc **độc lập** với các audit trước (F-01…F-51 trong `00_*.md`, A/B-series trong `Hy3.md`).
- **Phương pháp**: (1) đọc trực tiếp từng file nóng, lần theo từng đường gọi trên hot path; (2) đối chiếu số đo THẬT từ `metrics_sched_ON.json` / `metrics_sched_OFF.json` (A/B GpuArbiter, 1 phiên realtime) và `16_latency_after_f39.json` (harness 4× speed); (3) **không** re-report các lỗi đã sửa — mọi mục bên dưới là phát hiện MỚI hoặc diễn giải MỚI trên code hiện tại.
- **Ký hiệu**: 🔴 P0 = crash / mất dữ liệu / nghẽn hệ thống; 🟠 P1 = độ trễ lớn, chi phí tài nguyên đáng kể, leak; 🟡 P2 = trung bình / config nói dối; ⚪ P3 = vi mô / vệ sinh. Định danh: `QWEN-Q*` = backend, `QWEN-E*` = extension.

---

## 1. TL;DR — TOP 5 điều phải làm

| # | Phát hiện | Vì sao đứng đầu |
|---|----------|----------------|
| 1 | **🔴 QWEN-Q1 — Race `use-after-free` dịch thuật**: `_translate_sync` / `_translate_stream_sync` đọc con trỏ `cls._shared_llm` **không lock** rồi mới vào `_infer_lock`; `reconfigure()` có thể giải phóng model NGAY GIỮA HAI bước → llama.cpp bị gọi trên context đã đóng → **segfault cả tiến trình**. | Đây là lỗi đúng-đắn (correctness), không phải tối ưu. Hàng rào `_release_llm` (`engine.py:144`) bảo vệ KHÔNG được thread đang giữ pointer cũ. |
| 2 | **🔴 QWEN-Q2 — Event loop đóng băng khi nạp VAD engine**: `VADProcessor.__init__` chạy `_ensure_engine()` **ngay trên event loop**, đường xuống factory giữ **lock cấp lớp** qua `hf_hub_download` (vài phút nếu mạng chậm) — mọi coroutine (WS của mọi phiên, heartbeat, gửi phụ đề) đứng hình. | Một lần đổi engine = toàn backend treo; nghiêm trọng hơn vì nó nằm TRÊN LOOP, không phải thread nền. |
| 3 | **🟠 QWEN-E1/E3 — Chuóng main-thread tab trình duyệt**: MỖI sự kiện phụ đề chạy lại `attachToVideo` đầy đủ (disconnect/rebuild ResizeObserver + `getComputedStyle` + 2× `getBoundingClientRect` = **forced synchronous layout**) và service worker **broadcast mọi message tới MỌI iframe**. | p50 infer 93 ms nhưng tab có thể jank 50–200 ms/sự kiện trên DOM lớn — người dùng CẢM NHẬN trực tiếp, không metric backend nào đo được. |
| 4 | **🟠 QWEN-Q7 — Lãng phí GPU #1: preview recompute ratio p50 6,83×** (26,1 s audio qua model cho 3,8 s audio thật). Cửa sổ preview 6 s chạy lại TOÀN BỘ mỗi 300 ms; `preview_reuse_for_commit` mặc định TẮT nên commit còn tính lại lần nữa. | GPU là tài nguyên số 1 của hệ. Bật reuse + incremental-window giảm ~25–35% duty cycle ASR — chính là margin biến p99 2,8 s thành p99 < 1 s. |
| 5 | **🟠 QWEN-Q3 — Câu TTS ĐẦU TIÊN luôn chậm nhiều giây**: `lifespan` không prewarm TTS (`main.py:289-300`), model OmniVoice chỉ nạp khi có yêu cầu tổng hợp đầu tiên; `TranslationEngine.prewarm()` TỒN TẠI nhưng không ai gọi. | Triệu chứng lặp lại ở MỌI phiên: "câu 1 trễ vài giây, câu 2 nhanh" — và không metric p50 nào phản ánh được vì nó là model-load, không phải suy thoái ổn định. |

---

## 2. Kiến trúc thực tế — bản đồ thread / lock / queue (đã kiểm chứng từng dòng)

### 2.1 Luồng dữ liệu

```
Firefox tab                                     FastAPI backend (1 event loop asyncio)
─────────────                                   ─────────────────────────────────────────
worklet 16 kHz ──1024-sample frames (64 ms,     ws/handler.py:331-341 receive loop
~15.6 fps)────► SW (backpressure-gate) ──WS──►  _handle_binary_message (:459-473)
                                                    │ run_in_executor
                                                    ▼
                                                _VAD_EXECUTOR (1 worker, UNBOUNDED)   ← QWEN-Q6
                                                    │ feed_chunk (P4.6: model ngoài lock)
                                                    ▼
                                                VADProcessor._lock (RLock) ── 40 frame/s
                                                    │ callbacks on_speech_start/end
                                                    ▼
                                                ASRStreamEngine.stream_tokens (generator trên loop)
                                                    │ _infer_with_watchdog → _EXECUTOR (1 worker)
                                                    ▼
                                                transcribe.cpp NATIVE singleton
                                                (_infer_lock → _shared_lock, 1 stream in-flight)
                                                    │ commit → translation_queue (maxsize 32)
                                                    ▼
                                                _translation_worker → GGUFTranslator._TRANS_EXECUTOR (1 worker)
                                                    │ tts_queue (maxsize 32)
                                                    ▼
                                                _tts_worker → asyncio.to_thread → OmniVoice (PyTorch, _infer_lock)
                                                    │
                                    WS text JSON v3 + binary BTTS ◄── send từng message
```

### 2.2 Ma trận lock (tính cả GIL — mọi "thread nền" bên dưới đều ĐANG GIỮ GIL)

| Lock | Giữ bởi | Đường lấy | Xung đột đáng chú ý |
|---|---|---|---|
| ASR `_infer_lock`→`_shared_lock` | thread `_EXECUTOR` (giữ GIL suốt khi chạy Python trước khi vào native) | `engine.py:692-700` | Đã chốt thứ tự đúng (docs `:13-22`) ✅ |
| Translation `_infer_lock` | `_TRANS_EXECUTOR` | `:302`, `:388` | **pointer `_shared_llm` được đọc NGOÀI lock** → Q1 |
| Translation `_shared_lock` (RLock) | swap/reconfigure | `:212-237` | `_release_llm` barrier không cứu được race Q1 |
| VAD `self._lock` (RLock) | per-processor (per-session) | `processor.py:248-331` | `force_end` gọi callback **trong lock** → Q4 |
| VAD factory `_lock` (CẤP LỚP, global) | thread nào build engine | `firered.py:50-55`, `fsmn.py:53-57` | Giữ qua `hf_hub_download`; `is_cached/peek_engine` cũng cần → Q2 |
| `_voice_prompt_cache` (KHÔNG lock) | TTS | `tts/engine.py:147-161` | check-then-act → tính prompt đúp (Q15) |
| `metrics._lock` (global) | MỌI thread record | `core/metrics.py:44,201-207` | ~100+ ops/s × 5 đường; rẻ nhưng cộng dồn trên GIL (Q16) |
| `SafeStreamHandler._lock` (class-level!) | logger | `utils/logger.py` | ghi log = mutex toàn cục; đã throttle (⚠️ nhưng class-level → Q17) |
| GpuArbiter | asyncio.Event, P0–P3 | `core/gpu_scheduler.py` | Event-based đúng, **mặc định TẮT** (`config.py:338`) — xem §6 |

### 2.3 Số đo thật làm nền cho mọi nhận định bên dưới

Nguồn: `metrics_sched_ON.json` (1 phiên realtime, GpuArbiter BẬT):

| Metric | p50 | p95 | p99 | max |
|---|---|---|---|---|
| `asr.infer_core_ms` | 92.9 | 191.2 | — | **1760.5** |
| `asr.commit_ms` | 107.0 | — | 780.3 | 1470.5 |
| `asr.e2e_commit_ms` (dừng nói → phụ đề chốt) | 215 | — | **2793** | 3432 |
| `translation.infer_ms` | 433 | **1137** | — | 2061 |
| `translation.first_token_ms` (TTFT) | 33 | | | |
| `translation.queue_wait_ms` | — | — | 369 | 801 |
| e2e ASR→subtitle | 436 | 1138 | | |
| `asr.first_preview_ms` | 426 | | | |
| `vad.chunk_ms` | **10** | | | |
| `ws.send_ms` | **0.13** | | | |
| `preview_recompute_ratio` | **6.83** (26.1 s xử lý / 3.8 s unique) | | | |

Kết luận đọc từ bảng: VAD và WS-send **không** phải điểm nghẽn; mọi phần dư của `e2e_commit p99 2.8 s` nằm ở (a) hàng đợi GPU phía native (spike 1.5–1.8 s tái diễn cả khi arbiter OFF — max commit 1484 ms), (b) translation p95 > 1.1 s xếp sau preview ASR trên cùng GPU, (c) recompute ratio 6.8× ăn duty cycle.

---

## 3. PHÁT HIỆN BACKEND (QWEN-Q*)

### 🔴 QWEN-Q1 — Race `use-after-free` model dịch (CRASH CLASS) — P0
**File**: `backend/translation/engine.py:283` (+`:369`), `:302`, `:388`, `:212-237`, `:144`

```python
# :283  (trong _translate_sync, ĐANG CHẠY Ở _TRANS_EXECUTOR)
llm = cls._shared_llm          # ← ĐỌC KHÔNG LOCK
prompt = self.prompt_strategy.build_prompt(...)  # :293-300 — hàng trăm µs tới ms
...
with cls._infer_lock:          # :302 — MỘT LÚC SAU MỚI GIỮ LOCK
    for chunk in llm(prompt, ...):   # llm có thể đã bị ĐÓNG
```

Đường đối thủ: `reconfigure()` (`:212-237`) — được gọi từ `hotswap.activate_model` / REST `/api/config` (bg task trên loop) — giữ `_shared_lock`, gán `cls._shared_llm = mới`, rồi `_release_llm(old)` (`:237`). Hàng rào trong `_release_llm` là `with cls._infer_lock: pass` (`:144`) — hàng rào này **chỉ chặn được thread CHƯA vào** vùng inference; thread đã đọc pointer ở `:283` và đang xếp hàng chờ `_infer_lock` sẽ bước vào `llm(...)` với **llama-cpp context đã `close()`** → segfault hoặc UB trong native. `build_prompt` + `registry.get_model` + sắp xếp tham số nằm giữa hai bước đọc-pointer và lấy-lock chính là cửa sổ race (vài ms, nhưng swap model lúc đang có phiên chạy là kịch bản người dùng thật: đổi model từ popup khi video đang phát).

`_translate_stream_sync` **y hệt**: đọc `self.__class__._shared_llm` tại `:369`, vào `_infer_lock` tại `:388`.

**Fix** (chọn 1, rẻ nhất là (a)):
- (a) Đọc pointer BÊN TRONG lock: `with cls._infer_lock: llm = cls._shared_llm` rồi mới chạy — chuyển build_prompt vào trong hoặc build trước nhưng re-validate `cls._shared_llm is llm` sau khi có lock; nếu khác ⇒ dùng `cls._shared_llm` mới.
- (b) Refcount: `_release_llm` chờ `inflight == 0` (counter + Event) thay vì barrier rỗng.
- Tương tự đã làm ĐÚNG bên ASR (`_run_inference_sync` giữ `_infer_lock`→`_shared_lock` suốt, docs `asr/engine.py:13-22`) — dịch thuật là chỗ duy nhất còn hở.

---

### 🔴 QWEN-Q2 — Nạp VAD engine trên EVENT LOOP + global lock qua `hf_hub_download` — P0 (availability)
**File**: `backend/vad/processor.py:92` (`__init__` → `_ensure_engine()`), `:122`, `:194`; `backend/vad/engines/firered.py:50-55`, `fsmn.py:53-57`; `backend/ws/handler.py:288`

Chuỗi gọi: `handle_ws` (loop) → `session.init_components()` (loop, `handler.py:288`) → `VADProcessor.__init__` → `_ensure_engine()` NGAY TRÊN LOOP. `_ensure_engine` → factory `get_engine(name)` — factory giữ **`_lock` cấp lớp** suốt constructor, mà constructor của FireRed/FSMN gọi `hf_hub_download` khi chưa cache (`firered.py:50-55`) — download model VAD có thể mất **vài phút** tùy mạng. Hệ quả:
1. Toàn bộ event loop (mọi phiên WS, health, REST, heartbeat) đóng băng — asyncio không preempt được code đồng bộ.
2. Ngay cả khi chạy nền, `is_cached()` / `peek_engine()` (processor `:122`, `:194` — được path khác của loop gọi) cũng phải lấy cùng cái lock lớp đó → vẫn treo loop.
3. Người dùng đổi engine VAD CHƯA TẢI = sập tạm thời toàn backend, không log cảnh báo.

**Fix**: (1) `__init__` KHÔNG build engine — dùng `enabled=False` đến khi `_spawn_engine_load` xong; (2) `hf_hub_download` phải chạy TRƯỚC, NGOÀI lock (pattern `prepare_model` của ASR — `asr/engine.py:404-463` — làm đúng y như vậy: nạp ngoài lock, swap trong lock ngắn; áp nguyên mẫu đó cho VAD); (3) tách `is_cached/peek` sang lock riêng không chắn bởi download.

---

### 🟠 QWEN-Q3 — Chuỗi "không prewarm": TTS không có trong lifespan; `translation.prewarm()` không ai gọi — P1 (latency)
**File**: `backend/main.py:289-300`; `backend/tts/engine.py:238-242`, `:293-294`; `backend/translation/engine.py:256-263`

- Lifespan gather prewarm ASR + Translation + VAD-default — **thiếu TTS**. `synth_clone_bytes` gọi `load_model()` lazy (`tts/engine.py:293-294`): câu lồng tiếng ĐẦU TIÊN của mỗi lần bật TTS trả giá nạp vài GB trọng số + warmup (`:196-225`) — nhiều giây. Popup bật TTS giữa video ⇒ đúng câu người dùng vừa chờ bị trễ cục bộ; `handler.py:415-423` có prewarm theo-set-config nhưng CHỈ khi nhận `set_config` có `tts_enabled` (slider drag → debounce 150 ms phía client cũng chỉ gửi khi thôi kéo, OK, nhưng nếu user bật sẵn trong storage rồi Start thì không có message nào cả — path init không prewarm).
- `GGUFTranslator.prewarm()` (`translation/engine.py:256-263`, chạy 1 generate giả để GPU build kernel/graph) **tồn tại và KHÔNG có call-site nào** ngoài test. Lifespan chỉ gọi `load_model()` → lần dịch ĐẦU TIÊN vẫn phải compile kernel CUDA — đó chính là một phần giải thích `translation.infer_ms` max 2061 ms trong khi p50 433.

**Fix**: lifespan thêm `asyncio.to_thread(TTS.load_model)` (có điều kiện `config.tts.enabled`), và đổi `TranslationEngine.load_model` → `prewarm()` trong lifespan. Chi phí: khởi động chậm hơn ~3–5 s (đổi 1 câu chậm đầu lấy mọi câu nhanh — đúng trade-off).

---

### 🟠 QWEN-Q4 — `force_end` kích hoạt callback BÊN TRONG `self._lock` — P1 (deadlock class + head-of-line)
**File**: `backend/vad/processor.py:432-441` (vs thiết kế P4.6 `:295-331`)

`feed_chunk` đã được thiết kế đúng: bước 2 chạy model NGOÀI lock, bước 3 gom callback và **gọi ngoài lock** (`:314-330`, comment "Kích hoạt callbacks bên ngoài lock để tránh deadlock"). Nhưng `force_end()` lại gọi callback (`on_speech_end`) **khi đang giữ** `self._lock`. Chuỗi ngược dòng: `on_speech_end` → `ASRStreamEngine.on_speech_end` (lấy `asr._lock` + `call_soon_threadsafe`) — tức thứ tự lock **VAD→ASR** được tạo từ trong lock. Hôm nay chưa chết vì ASR không bao giờ gọi ngược vào VAD processor; nhưng chỉ cần MỘT callback tương lai (vd. `update_config` được gọi từ trong chain reset — `session.cleanup` đã chạy force_end + update path) là thành deadlock cổ điển RLock-per-session. Ngoài ra force_end chạy trên `_VAD_EXECUTOR`/cleanup — giữ lock suốt callback = chặn `feed_chunk` của CHÍNH phiên đó xếp hàng (head-of-line).

**Fix**: copy pattern feed_chunk — gom callbacks, nhả lock, rồi gọi. 5 dòng.

---

### 🟠 QWEN-Q5 — `_spawn_engine_load._worker` gọi `_ensure_engine()` không lock + `reset()` tái-dùng object state — P1
**File**: `backend/vad/processor.py:150-155`, `:443-449`

Hai đường nạp engine có thể chạy đồng thời: đường (a) có kiểm soát (spawn worker) và đường (b) inline `_ensure_engine` — worker KHÔNG giữ `self._lock` khi build ⇒ nếu một `feed_chunk` song song (đa phiên dùng chung… cùng engine singleton của factory thì OK nhờ lock lớp; NHƯNG lock lớp đó là Q2!) thì worst case vẫn là Q2; best case là hai worker nạp cùng tên engine, mỗi cái `hf_hub_download` — download ghi file đồng thời vào cùng cache dir.
Riêng `reset()` (`:443-449`): thay vì gán state object MỚI, nó **reset các field của object cũ in-place**. Kết hợp với cơ chế discard-an-toàn của P4.6 (`:317-320` so `self._state is not state`) ⇒ một batch đang chạy model khi `reset()` được gọi sẽ **không bị nhận diện là cũ** (identity không đổi) và ghi kết quả vào state vừa reset → trạng thái VAD nhiễu (đếm im lặng lẫn câu cũ/câu mới). Đây là lỗ hổng ĐÚNG của thiết kế identity-check — chính dòng `:317` chứng minh tác giả định danh là hợp đồng, nhưng `reset()` phá hợp đồng.

**Fix**: `reset()` phải tạo `VADStreamState()` MỚI gán vào `self._state` (bên trong lock) — một dòng; worker nạp engine giữ cờ `self._loading` để trùng-tên-thì-join.

---

### 🟠 QWEN-Q6 — `_VAD_EXECUTOR`: 1 worker, hàng đợi UNBOUNDED, nộp 1 task/khung hình — P1 (latency + backlog)
**File**: `backend/ws/handler.py:38`, `:459-473`

Mỗi binary frame (~15.6 lần/s) được `run_in_executor(_VAD_EXECUTOR, ...)` mà KHÔNG kiểm tra độ sâu hàng đợi (contrast: ASR có `max_inflight_infer` chặn xếp hàng — `asr/engine.py:1285-1296`, comment giải thích vì sao cần). VAD worker normally chạy `vad.chunk_ms` p50 10 ms ⇒ không sao. Nhưng CHỈ CẦN worker kẹt (Q2 engine load giữ lock lớp, GC pause, hf_hub call) thì backlog tích vô hạn; mỗi item giữ `data` (~2 KB) + reference session — RAM không đáng, ĐỘ TRỄ mới đáng: audio vẫn được gửi, phụ đề sẽ "đuổi" cả chục giây nội dung cũ (đúng loại triệu chứng người dùng báo "sub trôi"), và mọi `on_speech_end` bị trễ theo ⇒ commit trễ toàn phần.

**Fix**: đếm depth trước khi nộp (`_VAD_EXECUTOR._work_queue.qsize()` hoặc counter atomic); quá ngưỡng (vd. 64 ≈ 4 s audio) ⇒ BỎ FRAME (counter `vad.frames_dropped_backlog`) — bỏ frame vẫn tốt hơn đuổi phụ đề; và Q2 fix sẽ khiến worker khó kẹt hơn nhiều.

---

### 🟠 QWEN-Q7 — GPU recompute 6,83×: preview full-window mỗi nhịp + commit tính lại từ đầu — P1 (chi phí lớn nhất đo được)
**File**: `backend/asr/engine.py:1020-1029` (`_preview_window_start`), `:1109-1118` (`preview_reuse_for_commit`, mặc định **False** trong `config.py`), `:1280-1321`; `backend/config.py:154,166`

Đo trực tiếp bằng gauge FIX-10: p50 **6.83** — mỗi giây audio đi qua model ASR ~7 lần (câu 3.8 s thực tế ⇒ 26.1 s inference). Nguyên nhân cộng dồn:
1. Cửa sổ preview = 6 s (`preview_window_sec=6.0`), poll ~300 ms ⇒ mỗi giây nói có ~3.3 lần decode nguyên khối 6 s (tuyến tính theo độ dài window, không incremental).
2. `preview_reuse_for_commit=False` ⇒ khi VAD báo hết tiếng, commit decode LẠI toàn bộ span vừa preview xong — lần cuối cùng này trả đủ 100% chi phí cho kết quả gần như đã có trong `_last_preview_text`.
3. Không có prefix/KV reuse giữa các nhịp (hạn chế API transcribe.cpp 0.x — stream in-flight=1; pipeline `feed`/`finalize` hiện tại re-feed cả window, `:656+`).

Hệ quả dây chuyền: duty cycle ASR ~30–40% GPU nền (p50 93 ms × 3.3/s cho 1× realtime) + spike p99 2.8 s e2e_commit; và đây là lý do arbiter P0–P3 gần như vô dụng ở 1 phiên (§6) — kẻ chiếm GPU là preview, không phải dịch.

**Fix theo thứ tự rủi ro tăng dần**: (a) BẬT `preview_reuse_for_commit` sau 1 vòng A/B WER (code đã có sẵn, gate `:1109-1118`, chỉ cần test_09 chạy lại) — cắt ngay ~15–20% tổng giờ GPU; (b) giảm `preview_window_sec` 6→4 kết hợp `min_transcribe_sec` không đổi, hoặc scale window theo `speech_duration` (câu đầu ngắn window nhỏ); (c) incremental preview thật (chỉ decode tail + stable prefix) — cần API phía native, để roadmap dài.

---

### 🟠 QWEN-Q8 — `e2e_commit p99 2.8 s`: thành phần queue không được bảo vệ khi arbiter OFF — P1
**File**: `metrics_sched_ON/OFF.json`; `backend/config.py:338`; `backend/asr/engine.py:821-916`

A/B (§6 chi tiết) cho thấy arbiter mặc định OFF và khi bật thì tổng thời gian chặn CHỈ 41.8 ms/3 lần — vì với 1 phiên, kẻ gây xếp hàng không phải "dịch chen vào commit" mà là **commit xếp sau preview đã nộp trước nó** + spike native. Chuỗi có thật trong code: generator `stream_tokens` có nhường-commit (`:1258-1266` `preview_yielded_to_commit`) NHƯNG một preview đã nộp xuống `_EXECUTOR` trước đó thì commit phải chờ hết (FIFO của executor) — preview tối đa 6 s audio ≈ p95 191 ms nhưng max 1760 ms. Cộng translation p95 1137 ms (job GPU khác chen vào) là khớp p99 2793 ms.
**Fix**: không cần arbiter: thêm `ThreadPoolExecutor` THỨ HAI cho commit (2 slot: preview-vs-commit — commit không bao giờ xếp sau preview) — model native dùng chung qua `_infer_lock` vẫn đúng 1-stream; hoặc nâng `max_inflight_infer` context: khi `_pending_commits` không rỗng ⇒ preview đang chạy sẽ nhường (đã có) nhưng chưa nộp mới (đã có `:1290-1296`) — còn thiếu đúng chỗ "preview đã vào hàng đợi executor rồi". Với 1 worker, không thể ưu tiên trong hàng đợi → giải pháp 2-queue là đơn giản nhất.

---

### 🟡 QWEN-Q9 — `/health` gọi native ctypes MỖI LẦN POLL trên loop — P2
**File**: `backend/main.py:107-168` (`_asr_runtime_info`), `:388-407`

`_asr_runtime_info()` đọc trạng thái qua ctypes vào native lib (backend name, samples…) mỗi lần `/health` được hỏi; UI/popup monitor vài giây/lần. Bản thân call nhanh (µs) nhưng nó lấy `_shared_lock`/touch native state trong lúc inference thread đang giữ — có thể kéo dài health-check và (hiếm) chen vào giữa vòng lock nóng. **Fix**: cache 2 s (TTL) + đánh dấu stale. Health không cần real-time.

---

### 🟡 QWEN-Q10 — `config.metrics.enabled = False` là config NÓI DỐI — P2
**File**: `backend/config.py:317`; `backend/core/metrics.py:303` (chỗ DUY NHẤT đọc cờ này, và chỉ để chặn `dump_metrics_report`)

`record_metric / increment_counter / record_gauge` chạy VÔ ĐIỀU KIỆN. Nghĩa là mục "tắt metrics để tăng tốc" không có tác dụng; toàn bộ §2.2 (lock metrics, deque append, percentiles) luôn bật. May: chi phí thật nhỏ (xem Q16) — nhưng cờ chết thì phải sửa hoặc xóa. **Fix**: gate `self._enabled` ngay đầu `record_*` (1 if, không lock khi off).

---

### 🟡 QWEN-Q11 — `FireRed.min_silence_frame=60` (=1.5 s) áp đảo `silence_duration_ms=600` — P2 (chất lượng/trễ bị ẩn)
**File**: `backend/config.py:70-71`

Engine FireRed dùng tham số nội bộ `min_silence_frame` (60 frame × 25 ms = **1.5 s**) trong khi tầng cấu hình chung khai báo 600 ms; người dùng kéo "silence" trong popup chỉ có tác dụng với FSMN path. Hệ quả với engine mặc định: MỖI câu chốt trễ thêm ~900 ms so với cấu hình người dùng nghĩ — chiếm 1/3 khoản e2e p99. **Fix**: map `silence_duration_ms → min_silence_frame = ms/25` lúc build engine, hoặc ghi rõ trong UI.

---

### 🟡 QWEN-Q12 — `apply_config` của MỘT phiên ghi thẳng vào config TOÀN CỤC — P2 (scale)
**File**: `backend/ws/session.py:360` (`config.translation.target_lang = ...`), `:443-452` (`config.asr.preview_window_sec / poll_interval_ms`)

Session A đổi ngôn ngữ đích / kéo window preview ⇒ session B (nếu tồn tại) âm thầm đổi theo, kể cả khi B không gửi config nào. Đây là architecture blocker "cứng" thứ hai sau singleton native: không có per-session override layer cho ASR tuning. **Fix**: đọc qua `self.config` của session thay vì `config.asr` trong engine loop (engine đã nhận phần lớn qua constructor; chỉ window/poll là đọc global mỗi vòng).

---

### 🟡 QWEN-Q13 — Warm-up câu đầu dài tới `max_duration` — P2
**File**: `backend/asr/engine.py:967-1011`

Session mới chạy chuỗi warm-up inference tới tận ngưỡng 6 s trước khi phục vụ bình thường — spike có chủ đích, đã document. OK nhưng nên dịch warm-up vào prewarm (Q3) để người dùng không gặp nó ở đúng câu đầu sau khi đổi model (đường `prepare_model` đã warm 0.5 s — `:465-479` — tốt; đường lifespan thì chưa đủ).

---

### 🟡 QWEN-Q14 — `dump_metrics_report` ghi file ĐỒNG BỘ trên loop lúc cleanup — P2
**File**: `backend/ws/handler.py:355-387` (finally path), `core/metrics.py:290-303`

Cuối mỗi phiên: serialize toàn bộ metrics + `json.dump` file — trên event loop, giữa lúc đang có phiên khác chạy (đóng A → treo B vài ms–chục ms tùy kích thước). **Fix**: `asyncio.to_thread` cho bước dump.

---

### 🟡 QWEN-Q15 — `_voice_prompt_cache` check-then-act không lock (TTS) — P2
**File**: `backend/tts/engine.py:147-161`

Hai luồng `to_thread` tổng hợp cùng voice mới ⇒ cả hai tính `VoiceClonePrompt` (trích xuất embedding — CPU/GPU hàng trăm ms) rồi ghi cache đè nhau. Lãng phí, không hỏng kết quả. **Fix**: per-voice-key `threading.Lock` hoặc lock ngắn.

---

### ⚪ QWEN-Q16 — `metrics_collector._lock`: mutex toàn cục trên MỌI thao tác đo — P3
**File**: `backend/core/metrics.py:44,201-252`

Mọi `record_metric/increment_counter/record_gauge` đều `with self._lock` — một mutex duy nhất cho ~5 nguồn thread (loop, VAD worker, ASR executor, TRANS executor, TTS thread). Tần suất ước tính 100–200 ops/s; mỗi op dưới µs ⇒ KHÔNG phải điểm nghẽn đã đo (vad/ws p50 không nhiễm). Vẫn đáng sửa vì cộng dồn GIL-contention vô ích: **Fix**: counter dùng `itertools.count`/lock-free dict-per-thread-aggregate, hoặc chỉ cần bọc 4 record của `_run_inference_sync` vào 1 lock — tức giữ nguyên thiết kế, ghi nhận là chấp nhận được ở quy mô 1 phiên.

---

### ⚪ QWEN-Q17 — Logger: `os.environ.get` mỗi record + lock class-level — P3
**File**: `backend/utils/logger.py:169-176`

Mỗi record log đọc env 2 lần (đáng ra resolve 1 lần lúc init) và lấy mutex class-level. Flush đã throttle 0.25 s ✅. Đường hot-path (`feed_chunk`, vòng preview) không log mức INFO ✅ — chỉ cần chuyển env-read ra module-level.

---

### ⚪ QWEN-Q18 — `serializers.py` chế độ non-compact nhân đôi field — P3
**File**: `backend/ws/serializers.py`; `backend/config.py` (`stream_tokens=True` `:276`)

Client protocol < 3 nhận payload có cả field trùng (`text` + `stable_text`, ...) — dung lượng gấp ~2×, JSON parse nặng hơn. Bản thân không đáng kể (KB/s) nhưng cần nhớ khi so sánh benchmark cũ/mới. Ngoài ra `stream_tokens=True` MẶC ĐỊNH có nghĩa mọi câu đều chạy qua generator streaming (queue 8 + `call_soon_threadsafe`/token, `translation/engine.py:425-460`) — chi phí GPU của streaming ≈ 100% so với one-shot (không tiết kiệm gì về compute, chỉ đổi trải nghiệm TTFT 33 ms). Nên để popup tắt được hẳn khi user không cần "chữ hiện dần".

---

### ⚪ QWEN-Q19 — `_process_binary_chunk`: `bytes(raw_buf[a:b])` mỗi frame VAD — P3
**File**: `backend/vad/processor.py:286`, `:297-311`

Mỗi frame 400-sample được cắt thành `bytes` MỚI (copy 800 B) rồi đưa qua `is_speech(..., chunk_raw=...)`. Với 40 frame/s = 32 KB/s — không đáng RAM, nhưng mỗi lần cắt là một lần cấp phát + GIL. P4.6 đã tối ưu phần model; phần cắt có thể giữ memoryview khi engine nội bộ không giữ reference. Bỏ qua nếu đo được < 0.1 ms/frame (đo: `vad.chunk_ms` p50 10 ms cho CẢ batch ⇒ dư địa không nhiều).

---

### ⚪ QWEN-Q20 — Poll-then-sleep lẫn nhau giữa watchdog và executor — P3
**File**: `backend/asr/engine.py:863-912`

Vòng watchdog dùng `wait_for(shield(fut), timeout=slice_s)` với slice 0.25–1 s ⇒ 4 lần đánh thức/giây trong khi inference chạy. Đúng thiết kế (canh RSS) nhưng chi phí loop đều: khi `rss_runaway_delta_mb` chỉ cần cho harness, production có thể tắt (`=0` → branch `:853` thoát sớm). Ghi chú: config mặc định 1024 MB ⇒ luôn bật; cân nhắc tắt mặc định ở máy 16 GB để giảm nhiễu timer.

---

## 4. PHÁT HIỆN EXTENSION FIREFOX (QWEN-E*)

### 🟠 QWEN-E1 — Mọi sự kiện phụ đề gọi lại `attachToVideo` đầy đủ → forced layout — P1
**File**: `extension_firefox/content/content-script.js:192-210` (mỗi event → `ensureOverlay(video)`), `:155-165`; `overlay-manager.js:97-104`

`handleSubtitleEvent` → `ensureOverlay(video)` → `overlayManager.attachToVideo(video)` — chạy MỖI LẦN CÓ VIDEO, kể cả khi video không đổi. `attachToVideo` (`:97-104`) làm 3 việc nặng: `_attachHost()` (getComputedStyle + có thể appendChild = reflow), `_setupResizeObserver()` (**disconnect + new ResizeObserver + observe ×2** — rebuild observer mỗi event!), `_updateScale()` (`:260-266` → `_calculateFontSizes` → `getBoundingClientRect` ×1-2 = **layout sync**). Với tốc độ preview ~3–7 event/s, mỗi event trả ~1–2 lần forced layout; trang player lớn (YouTube DOM dày) = 5–30 ms mỗi lần jank, cộng dồn thành giật rõ khi scroll/di chuột.
**Fix**: `ensureOverlay` nên no-op khi `video === this.targetVideo && host.isConnected`; ResizeObserver tạo MỘT lần trong `init`, chỉ re-observe khi target đổi. Check nhanh `if (this.host.isConnected && this.targetVideo === video) return;` ở đầu `attachToVideo`.

### 🟠 QWEN-E2 — MutationObserver trên TOÀN BỘ `document.body` (subtree) chỉ để giữ host — P1
**File**: `overlay-manager.js:79-91`

`observe(document.body, {childList:true, subtree:true})` — trên trang video, DOM mutations xảy ra hàng chục–hàng trăm lần/giây (progress bar, ad overlay, suggestion list). Callback của observer chạy ĐỒNG BỘ theo microtask batch, chi phí nhỏ × tần suất lớn, và mỗi lần detached lại làm nguyên chuỗi attach+RO rebuild+scale (E1). **Fix**: thay bằng `customElements`/slot trick, hoặc observe RỘNG HƠN NHƯNG RẺ HƠN: đăng ký trên `document` chỉ khi `host` thật sự rời DOM — cách khả dụng nhất: giữ nguyên nhưng throttle callback bằng `requestAnimationFrame` + thêm guard `if (!host.isConnected)` (đã có) — điểm chính là CHUYỂN subtree observer sang observer trên CHÍNH PARENT của host (`host.parentElement`) + 1 observer mức body `childList:false`. Parent bị remove thì RO/custom `DOMNodeRemoved` trên document capture-phase vẫn bắt được; thực dụng nhất: observer trên `document.documentElement` với `childList:true, subtree:true` nhưng callback rỗng siêu nhanh (chỉ đọc 1 flag) + work dời sang rAF.

### 🟠 QWEN-E3 — Service worker broadcast mọi message tới MỌI frame — P1
**File**: `extension_firefox/background/service-worker.js:278-286`

Không truyền `{frameId}` khi `tabs.sendMessage` ⇒ mỗi subtitle event được giao tới TẤT CẢ content scripts của TẤT CẢ iframe (YouTube có hàng chục iframe: ITP, ads, embed). Mỗi frame tỉnh dậy, parse, rồi tự quyết định loại bỏ (content-script.js:213-215 "no video, not capturing → ignore"). CPU đốt miễn phí × số frame × tốc độ event. Ngược chiều cũng lỗi tương tự nhưng đã vá (P3.5b chỉ 1 frame phát TTS).
**Fix**: khi Start capture, ghi lại `frameId` của frame sở hữu video; broadcast → `sendMessage({frameId})` trực tiếp cho frame đó; bản tin điều khiển (config) mới gửi all.

### 🟠 QWEN-E4 — Backpressure ngưỡng quá rộng + HANG không nhả tới SOFT — P1
**File**: `lib/backpressure-gate.js:30-31`; `service-worker.js:255-270`

SOFT=128 KB ≈ **4 giây audio**, HARD=512 KB ≈ **16 giây**. Khi socket chậm (wifi yếu, backend spike 1.7 s), buffer vượt SOFT ⇒ DỪNG GỬI, nhưng logic chỉ cho phép gửi tiếp khi về dưới SOFT — trong khi đang tạm dừng thì bufferedAmount gần như không giảm nhanh (backend xử lý chậm đúng lúc nghẽn) ⇒ hệ “đuổi phụ đề” 4–16 s thay vì dropping mượt. DROP mode giữ mẫu MỚI NHẤT (đúng) nhưng người dùng không có tín hiệu nào.
**Fix**: SOFT≈16–32 KB (0.5–1 s), HARD≈128 KB, và thay “dừng hẳn” bằng “downgrade”: khi trên SOFT ⇒ gửi mỗi khung THỨ NHÌ (halve rate) — mất 50% độ mịn PCM nhưng không trôi timestamp; chỉ dừng hẳn ở HARD.

### 🟠 QWEN-E5 — Resampler 3:1 (48k→16k) KHÔNG lọc anti-alias ⇒ aliasing vào dải ASR — P1 (chất lượng)
**File**: `lib/audio-processor.js:62-85` (đọc `ratio`, `phase`)

Với `inputSampleRate/outputSampleRate = 48000/16000 = 3.0`, `phase` luôn nguyên (`src = i + 0` vì frac=0) ⇒ vòng nội suy tuyến tính SUY BIẾN thành naive decimation: giữ 1 mẫu, bỏ 2. Mọi năng lượng trên 8 kHz (cymbal, sibilance, noise hiss — rất nhiều trong nhạc nền video) GẤP NHẬP (fold) ngược vào dải 0–8 kHz làm méo phổ tiếng nói mà model ASR phải ăn. Đây là thất thoát CHẤT LƯỢNG nằm ngoài mọi A/B WER hiện tại (đo bằng file 16 kHz sẵn nên không thấy). (Ngoài ra `:67,:84` trả `subarray` của buffer dùng chung — người nhận phải copy ngay; caller hiện đúng nên chỉ là hợp đồng ngầm.)
**Fix**: ratio ≥ 2 ⇒ tiền-lọc trung bình 3 mẫu (boxcar 3-tap đủ cho 3:1, rẻ: 1 phép nhân/mẫu đầu vào, chạy trong worklet không đáng kể): `out[n] = (s[i-1]+s[i]+s[i+1])/3`. Giữ linear interp cho ratio lẻ. Kiểm chứng bằng WER A/B trên dữ liệu 48 kHz thật (`test_09`).

### 🟠 QWEN-E6 — Cụm leak vòng đời kết nối: port chết không reconnect + double-CONNECT + Blob URL không revoke — P1
**File**: `lib/ws-client.js:52` (đổi port khi chưa disconnect), `:117-124` (port chết sau settle → không emit/reconnect); `service-worker.js:181-231` (CONNECT mới khi đã có WS cũ → leak WS cũ); `lib/tts-player.js:300-301` + `:329-345` (`URL.createObjectURL` khi fallback Blob-URL nhưng `_trimQueue` loại bỏ item không revoke)

Ba đường leak nhỏ độc lập cùng họ “tài nguyên không thu hồi”: (1) đổi host/port lúc đang nối → socket cũ mồ côi; (2) Firefox kill idle SW-port sau 30 s-ish, client chỉ biết khi `onDisconnect` NHƯNG handler sau settle không bấm `_scheduleReconnect` ⇒ phiên “sống sót ảo” — user tưởng đang chạy, không có phụ đề mới, không reconnect; (3) mỗi lần Start khi WS cũ còn mở → WS cũ không close, 2 socket cùng ghi; (4) Blob WAV ~200–500 KB/câu lọt qua `_trimQueue` (queue max 3, drop khi lag >12 s) không revoke ⇒ RAM tab tăng theo số câu bị bỏ.
**Fix**: chuẩn hoá teardown: `disconnect()` trước mọi `connect()`; luôn `_scheduleReconnect()` trong mọi handler `onDisconnect`; `_trimQueue` revoke `item.audioUrl` trước khi shift; `port.onDisconnect` có idempotency flag.

### 🟡 QWEN-E7 — Popup kéo slider: `executeScript allFrames` + storage write + POST /api/config mỗi 150 ms — P2
**File**: `popup/popup.js:603-631` (handler slider → debounce 150 ms `:817`), `:432`

Mỗi lần đổi cỡ chữ/độ sáng: inject script vào MỌI frame (scripting API round-trip), ghi `browser.storage.local`, VÀ gọi REST — chuỗi 3 IPC cho mỗi lần “thả” hoặc mỗi 150 ms khi kéo liên tục. Kéo thanh trượt 2 giây = ~13 vòng. **Fix**: chỉ gửi giá trị CUỐI sau khi `change` (không phải `input`+debounce); chỉ target frame sở hữu overlay (cùng hạ tầng frameId của E3); gộp config qua WS message thay vì REST.

### 🟡 QWEN-E8 — Chuỗi tuần tự render TTS: `onended` mới xếp câu kế — P2
**File**: `tts-player.js:348-378`

`_playNextBuffer` chỉ được gọi trong `src.onended` ⇒ luôn có gap ≈ `ctx.resume()` + tạo node + scheduling giữa hai câu (đo chủ quan 50–150 ms, đủ nhận ra khi nghe liên tục). Backend sinh TTS RTF ~0.13 (nhanh gấp ~8× realtime) nên buffer luỹ kế cho phép crossfade 30 ms mà không lệch. **Fix**: giữ 1 `nextStartTime = max(ctx.currentTime, prev)` và `src.start(nextStartTime + 0.02)` schedule chồng (queue-based scheduling thay vì event-chain) — xoá luôn gap.

### ⚪ QWEN-E9 — `setInterval` ping 30 s + probe backpressure: ổn — ghi nhận
`ws-client.js:232-237` ping 30 s OK. Video-discovery cache có TTL miss (`content-script.js:138-153`) — làm ĐÚNG (không quét DOM mỗi event). Renderer không dùng innerHTML động cho text (đã xác nhận grep — không có innerHTML ở path phụ đề), ducking bằng `setTargetAtTime` (audio-capture.js:56-57) — đúng chuẩn, không zipper noise.

---

## 5. TÀI NGUYÊN: CPU / GPU / RAM / VRAM / I-O

### 5.1 GPU — bottleneck số 1 (đã lượng hoá)
- Duty cycle nền ASR ≈ `93 ms × 3.3 preview/s ≈ 30%` cho MỘT phiên realtime, phần lớn là **recompute vô ích** (Q7 ratio 6.83×). Translation p50 433 ms + TTS xen kẽ ⇒ các job P2/P3 chờ nhau và chờ P0 spike tới p99 2.8 s (Q8).
- Spike native (max 1484–1760 ms) TỒN TẠI CẢ KHI arbiter OFF ⇒ đến từ chính engine Vulkan/CUDA của transcribe.cpp (long-window compile/allocation), không phải contention Python — arbiter không chữa được (khớp `05_measurements §12.6`: phần phình/thăng trầm thuộc native).
- VRAM: các tầng đã làm đúng EMPTY-cache có chủ đích (TTS unload A2-3; `empty_cache` sau load `tts/engine.py:231-232`). Không phát hiện chỗ nào giữ VRAM oan ngoài việc TTS Loaded-không-dùng khi user tắt TTS từ đầu (đã vá A2-3 cho đường config).

### 5.2 CPU / GIL
- VAD: p50 10 ms/chunk = ~16% của 1 worker-thread, không phải điểm nghẽn (kết luận đúng như README).
- GIL thật sự là “bottleneck ẩn” khi 5+ threads cùng chạy Python (slice, normalize, metrics, log). Các đường chạy dài đã nhả GIL trong native (transcribe.cpp, llama.cpp binding) ⇒ mức chấp nhận được; việc cần làm là KHÔNG THÊM work Python vào loop (E-chains, Q9, Q14).
- `audio_buffer.get_slice` copy: đo p50 0.04 ms ⇒ **đừng tối ưu** (§7).

### 5.3 RAM
- Backend: ring 60 s ≈ 3.84 MB/phiên ✅; queues có trần 32 ✅; `_pending_commits` ≤ 6 gộp-không-vứt ✅; checkpoints LRU-200 ✅; deques bounded ✅. Rủi ro thật duy nhất là **native runaway (66–85 MB/s)** — đã có 2 hàng rào F-39 (`inference_watchdog_sec`, `rss_runaway_delta_mb`, `engine.py:835-912`) + recycle RSS (đã fix trước). KHÔNG phát hiện đường Python nào leak mới; extension thì có E6.
- `_VAD_EXECUTOR` unbounded (Q6) là hàng đợi Python không-trần DUY NHẤT còn sót — đã phân tích: RAM không đáng, latency đáng.

### 5.4 I-O
- Không có đọc/ghi file trên hot path sau khi model đã cache (model_download chỉ lúc setup ✅). `dump_metrics_report` cuối phiên (Q14) và log flush 0.25 s — cả hai ngoài đường tiếng. Disk không phải điểm nghẽn.

---

## 6. A/B GpuArbiter (dữ liệu trong repo) — PHÁN QUYẾT: giữ mặc định TẮT với 1 phiên

| | sched ON | sched OFF |
|---|---|---|
| infer_core p50 / max | 92.9 / **1760** | 87.7 / 1483 |
| commit p50 / max | 107.0 / 1470 | 99.9 / 1484 |
| queue_wait max | 801 | 440 |
| e2e_commit max | 3432 | 2643 |
| Arbiter activity | blocked=3, tổng chờ 41.8 ms | — |

Kết luận trung thực: ở 1 phiên, arbiter CHẶN 3 LẦN/41.8 ms — không đổi bức tranh; p50 ON thậm chí CAO HƠN nhẹ (đo nhiễu ±5%). Giá trị thật của nó chỉ xuất hiện khi TTS+translation+ASR cùng tranh GPU (đa phiên hoặc TTS dày) — lúc đó nó rẻ (Event-based, `admit_max_wait_ms=1200` chống starvation ✅). Khuyến nghị: giữ `scheduler_enabled=False` (đúng mặc định `config.py:338`), BẬT qua REST khi có ≥2 phiên; đừng bán nó như “fix latency” cho 1 phiên.

## 7. Những gì ĐÃ ĐÚNG — đừng động vào
1. `audio_buffer` ring 60 s + copy-on-slice (p50 0.04 ms — mọi đề xuất zero-copy ở đây là over-engineering).
2. P4.6 VAD: model ngoài lock + discard theo identity (`processor.py:295-331`) — thiết kế mẫu mực (trừ Q4/Q5/Q19 là ngoại lệ/vi mô).
3. F-44 generation-check cho commit/preview stale; F-47 chống busy-loop (`engine.py:1237-1250`); `_enqueue_commit_locked` gộp-không-vứt (P4.5).
4. `_coalesce_enqueue` (FIX-02/03) drain-merge-task_done đúng nghiệp vụ asyncio.Queue (đọc `handler.py:218-275` — từng chi tiết cân bằng `task_done` đã kiểm).
5. `prepare_model` ASR zero-downtime + pre-warm 0.5 s (`:404-479`) — chính là pattern để sao chép sang VAD (Q2).
6. ws send p50 0.13 ms + orjson + v3 compact; serializers không thừa khi client mới.
7. Cleanup thứ tự FIX-13; heartbeat 0.25 s; stall_watchdog faulthandler; worklet transferable zero-copy framing; ducking `setTargetAtTime`; video-scan TTL cache.
8. Normalizer zero-temp-alloc (P2.6); dedup 3 tầng bounded 30 s.

## 8. Quy mô đa phiên — kiến trúc có chặn không?
CÓ, 3 tầng, theo thứ tự cứng dần:
1. **Native transcribe.cpp 0.x: 1 stream/session, model singleton** (`_shared_session`) — đã biết + README thừa nhận; cần native hỗ trợ multi-context.
2. Session ghi config global (Q12) — hai phiên đánh nhau qua `config.asr.poll_interval_ms` v.v.
3. `_TRANS_EXECUTOR`/`_EXECUTOR` 1 worker ⇒ phiên 2 queue-wait sau phiên 1 (queue_wait max 801 ms đã đo ở 1 phiên; nhân theo số phiên).
Với hiện trạng, 2 phiên trên RTX 5060 Ti: duty cycle ASR ×2 ≈ 60% + translation + TTS ⇒ p99 sẽ vượt 5 s. Khuyến nghị ghi rõ vào README: **multi-session cần Q7 (giảm recompute) TRƯỚC**, vì nó là hệ số nhân.

## 9. Thứ tự sửa đề nghị (effort × impact)
| Tuần tự | Mục | Effort | Impact |
|---|---|---|---|
| 1 | Q1 race `_shared_llm` | ~15 dòng | Chặn crash khi đổi model dịch |
| 2 | Q4 force_end callback ngoài lock | ~10 dòng | Dẹp deadlock class |
| 3 | Q2 VAD engine nạp ngoài lock + không trên loop | ~40 dòng (copy `prepare_model`) | Dẹp treo-loop toàn cục |
| 4 | Q3 prewarm TTS + translation.prewarm() lifespan | ~5 dòng | Câu đầu tiên nhanh |
| 5 | Q7a bật `preview_reuse_for_commit` + A/B WER | config + chạy `test_09` | −15–20% GPU, −1 commit pass |
| 6 | E1/E2/E3 extension (guard attachToVideo, rAF observer, frameId) | ~60 dòng JS | Hết jank tab |
| 7 | E5 anti-alias decimation + WER A/B | ~15 dòng worklet | Chất lượng ASR 48 kHz |
| 8 | Q11 map `min_silence_frame`; Q6 VAD backlog guard; Q8 commit-priority-queue | vừa | p99 xuống |
| 9 | E4 backpressure; E6 teardown chuẩn hoá; E7/E8 | vừa | Ổn định phiên dài |
| 10 | Q10/Q16/Q17/Q19/Q20, Q9, Q14, Q15 | nhỏ | Vệ sinh |

## PHỤ LỤC A — Đối chiếu với audit cũ (không re-report)
ĐÃ SỬA (xác nhận còn nguyên trong code hiện tại): F-01/F-02/F-04/F-05/F-07/F-08/F-12/F-13/F-15/F-30, F-39 watchdog + RSS (dòng 821-916), F-44 generation, F-47 anti-busy-loop, F-48/F-49, F-50, A2-1 arbiter, A2-2 CUDA stream riêng TTS, A2-3 trả VRAM TTS, B5-1, B9-4, FIX-01/02/03/06/09/10/11/13/14, P2.x/P3.x/P4.x.
BÀI NÀY LÀ **MỚI**: Q1 (TOCTOU `_shared_llm`), Q2 (VAD-on-loop + factory-lock-qua-download), Q3 (prewarm thiếu), Q4 (force_end in-lock), Q5 (reset identity hole), Q6 (VAD backlog guard), Q7 (lượng hoá ratio 6.83 + reuse off), Q8 (commit xếp sau preview — arbiter không trị được), Q11 (min_silence_frame ghi đè), Q12/Q13/Q14/Q15/Q16/Q17/Q19/Q20; extension E1–E8.
KHÔNG đổi kết luận cũ: native runaway §12.6 vẫn là rủi ro lớn nhất ngoài tầm Python; WER CUDA≈Vulkan; bundle CUDA tự build ở `bin/`.

## PHỤ LỤC B — Bảng metric nguồn
(`metrics_sched_ON.json`, `metrics_sched_OFF.json`, `16_latency_after_f39.json` — trích ở §2.3 và §6; harness 4× speed trong `16_*.json` có infer_core p50 159.8/max 6809 ms — con số 4× KHÔNG dùng để suy realtime, chỉ dùng để so cấu hình.)

*— Hết báo cáo. Mọi dòng `file:line` đối chiếu code tại commit hiện hành của nhánh audit (workspace `D:\vibe-translation-addon-transcribe_cpp`).*