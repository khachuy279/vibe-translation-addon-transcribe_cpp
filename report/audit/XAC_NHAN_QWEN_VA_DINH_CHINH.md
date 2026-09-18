# XÁC NHẬN BÁO CÁO AUDIT QWEN (`BAO_CAO_AUDIT_HIEU_NANG_QWEN.md`) + ĐÍNH CHÍNH

**Ngày:** 2025-09-17
**Phạm vi:** 20 finding `QWEN-Q*` (backend) + 8 finding `QWEN-E*` (extension) trong bản QWEN cập nhật.
**Máy tham chiếu:** RTX 5060 Ti 16 GB, driver 596.36, Ryzen 5 5600X (12 luồng logic), Python 3.13, Windows 11.
**Cách kiểm:** đọc code tại đúng số dòng QWEN trích + **đo lại bằng thực nghiệm** ở 3 script mới
(`external/build-tmp/g01_ratio_1x.py`, `g02_min_silence_frame.py`, `g03_health_cost.py`).
Không có kết luận nào dưới đây chỉ dựa vào suy luận: mỗi mục TRUE/FALSE đều có dòng code hoặc số đo kèm theo.

---

## 0. TÓM TẮT ĐIỀU HÀNH

### 0.1. Kết quả thẩm định

| Nhóm | Số lượng | Ghi chú |
| :--- | ---: | :--- |
| ✅ **TRUE** (đúng, giữ nguyên) | **19** | gồm cả hai P0 (Q1, Q2) và các P1 lớn (Q3, Q4, Q5, Q7, Q8, E1, E3, E5, E6) |
| 🟡 **PHÓNG ĐẠI** (đúng hiện tượng, sai mức độ/cơ chế) | **4** | Q6, Q9, E2, E7 |
| ❌ **FALSE** (sai, phải bỏ hoặc sửa lại) | **1** | **Q11** — `silence_duration_ms` **CÓ** tác dụng với FireRed |
| ⚪ **P3, không kiểm dòng-by-dòng** | **4** | Q16, Q18, Q19, Q20 (QWEN tự xếp P3; đo được ≈ 0) |

**Chất lượng báo cáo QWEN: cao.** Khác hẳn báo cáo Gemini trước đó: mọi đường dẫn file/dòng đều
trỏ đúng chỗ, hai finding P0 (Q1, Q2) đều **thật và nghiêm trọng**, và Q7 thậm chí còn **đo thấp hơn
thực tế**. Tỉ lệ sai chỉ 1/24 — và cái sai đó (Q11) không ảnh hưởng kế hoạch.

### 0.2. Ba đính chính quan trọng nhất

1. **ĐÍNH CHÍNH CỦA CHÍNH TÔI (nặng nhất).** Báo cáo Gemini §2.1 mà tôi viết trước đây đã **bác bỏ
   sai** finding "preview recompute 9×" bằng số đo **1,00–1,67×**. Số đo đó chạy harness ở **8× tốc độ
   thật**, mà `preview_recompute_ratio` **tỉ lệ thuận với tốc độ pacing** ⇒ bị đo thấp giả tạo đúng
   ~8 lần. Đo lại ở **1× (realtime): 8,75 – 9,23×**. **Gemini đúng. QWEN đúng (6,83 là còn khiêm tốn). Tôi sai.**
   → Kéo theo: khuyến nghị "C.1 — KHÔNG làm incremental preview" trong kế hoạch cũ phải **đảo ngược**.
2. **QWEN-Q11 SAI.** QWEN khẳng định `min_silence_frame=60` (1,5 s) áp đảo `silence_duration_ms=600`
   khiến mỗi câu chốt trễ thêm ~900 ms và slider "silence" trong popup vô tác dụng với FireRed.
   **Thực nghiệm với chính thư viện `fireredvad` cho thấy ngược lại:** `is_speech` mà `processor.py:348`
   đọc là **quyết định ngưỡng TỪNG FRAME**, không phải trạng thái máy trạng thái. Nó về `False` sau
   **75 ms** (bằng đúng `smooth_window_size=5`), nên bộ đếm im lặng của processor chạy và **600 ms thắng**.
   Slider hoạt động bình thường.
3. **QWEN-Q6 PHÓNG ĐẠI CƠ CHẾ.** QWEN viết hàng đợi VAD "UNBOUNDED, nộp 1 task/khung hình ⇒ backlog
   tích vô hạn". Đúng là nộp 1 task/khung hình, **nhưng `handler.py:473` `await` chính future đó** ⇒
   mỗi phiên chỉ có **1 task VAD in-flight**; độ sâu hàng đợi bị chặn bởi **số phiên**, không phải bởi
   số khung hình. Fix QWEN đề xuất (bỏ khung khi `qsize > 64`) **không bao giờ kích hoạt**.

---

## 1. BẢNG THẨM ĐỊNH TOÀN BỘ

| # | Mức QWEN | Phán quyết | Bằng chứng |
| :--- | :--- | :--- | :--- |
| Q1 | 🔴 P0 | ✅ **TRUE** | `translation/engine.py:283` (và `:369`) đọc `cls._shared_llm` **ngoài** lock; lock chỉ lấy ở `:302`/`:388`; `_release_llm` (`:139-146`) chỉ `with cls._infer_lock: pass` ⇒ barrier rỗng không cứu thread đã đọc pointer |
| Q2 | 🔴 P0 | ✅ **TRUE** (kèm giảm nhẹ) | `vad/engines/__init__.py:28-40` giữ `cls._lock` **suốt constructor**; `firered.py:50-55` gọi `hf_hub_download` **bên trong** lock; `processor.py:92` gọi `_ensure_engine()` ngay trong `__init__`; `ws/session.py:203` dựng `VADProcessor` và `handler.py:288` chạy trên event loop. **Giảm nhẹ:** `main.py:292` prewarm engine mặc định qua `asyncio.to_thread` **trước khi phục vụ** ⇒ treo loop chỉ xảy ra khi engine **chưa cache** (đổi engine lần đầu / prewarm lỗi) |
| Q3 | 🟠 P1 | ✅ **TRUE** | `translation.prewarm()` chỉ có call-site trong `tests/` (grep toàn `backend/`), lifespan chỉ gọi `load_model()`; lifespan (`main.py:289-294`) **không có TTS** |
| Q4 | 🟠 P1 | ✅ **TRUE** | `processor.py:432-441`: `on_speech_end()` gọi **trong** `with self._lock`, trong khi `feed_chunk` cố ý gọi callback **ngoài** lock (`:325-330`) |
| Q5 | 🟠 P1 | ✅ **TRUE** | `processor.py:443-449` `reset()` sửa field **in-place**; hợp đồng identity ở `:317` `if self._state is not state` không phát hiện được ⇒ batch đang bay ghi vào state vừa reset |
| Q6 | 🟠 P1 | 🟡 **PHÓNG ĐẠI** | `handler.py:473` `await loop.run_in_executor(...)` ⇒ **await chính future** = backpressure tự nhiên, 1 in-flight/phiên. Không có "backlog vô hạn"; fix `qsize>64` là vô dụng |
| Q7 | 🟠 P1 | ✅ **TRUE** (đo thấp) | Đo lại 1×: **9,23 / 8,75×** (QWEN ghi 6,83). `config.py:178 preview_reuse_for_commit=False`; một `_EXECUTOR` duy nhất (`asr/engine.py:849`) |
| Q8 | 🟠 P1 | ✅ **TRUE** | preview `:1306` và commit `:1121` **cùng** `_EXECUTOR`, `max_workers = max_inflight_infer = 1` (`:65-66`) ⇒ FIFO: preview đã nộp chặn commit nộp sau |
| Q9 | 🟡 P2 | 🟡 **PHÓNG ĐẠI** | Đo 50 lần: `_asr_runtime_info()` = **1,009 ms/lần**; bản thân call native 0,0001–0,0006 ms. **Không** lấy `_shared_lock` (chỉ `transcribe_cpp` module API + đọc attribute) ⇒ không "chen vào vòng lock nóng". Nên hạ xuống P3 |
| Q10 | 🟡 P2 | ✅ **TRUE** | `metrics.py:201-203` `record_metric → record_latency` **không** kiểm `_enabled`; `:303` là chỗ **duy nhất** đọc cờ, và chỉ để chặn `dump_metrics_report` |
| Q11 | 🟡 P2 | ❌ **FALSE** | Thực nghiệm `g02`: `is_speech=False` tại **1075 ms**, `is_speech_end=True` tại **2550 ms** ⇒ chốt câu ở ~1675 ms, **không phải** 2550 ms. `min_silence_frame` chỉ điều khiển event END của engine — event này đến **sau** khi processor đã tự chốt câu ⇒ **dead config**, vô hại |
| Q12 | 🟡 P2 | ✅ **TRUE** | `ws/session.py:360` `config.translation.target_lang = …`, `:443` `config.asr.preview_window_sec`, `:447` `config.asr.poll_interval_ms` — đều ghi vào singleton toàn cục |
| Q13 | 🟡 P2 | ✅ **TRUE** | `asr/engine.py:992-1003`: `long_sec = max_duration_sec` (mặc định 6,0) rồi `_run_inference_sync(zeros(6 s))` |
| Q14 | 🟡 P2 | ✅ **TRUE** | `ws/handler.py:387` gọi `dump_metrics_report(...)` **đồng bộ** trên loop. **Bất nhất nội bộ:** `main.py:731` làm đúng (`await asyncio.to_thread(...)`) |
| Q15 | 🟡 P2 | ✅ **TRUE** | `tts/engine.py:147-161`: check-then-act trên `OrderedDict` **không lock**, `move_to_end`/`popitem` có thể chạy song song từ 2 phiên |
| Q16 | ⚪ P3 | ✅ TRUE (không đo sâu) | `metrics.py:207,215`, `record_latency` đều `with self._lock` — QWEN tự đánh giá "không phải điểm nghẽn đã đo", đồng ý |
| Q17 | ⚪ P3 | ✅ **TRUE** | `utils/logger.py:160` `_lock = threading.RLock()` **cấp lớp**; `:170` `os.environ.get("LOG_DEDUP_MS")` gọi **mỗi record** qua `_dedup_window_sec()` (`:180`) |
| Q18 | ⚪ P3 | ⚪ không kiểm | Ảnh hưởng KB/s, không cần |
| Q19 | ⚪ P3 | ⚪ không kiểm | QWEN tự ghi "bỏ qua nếu < 0,1 ms/frame"; `vad.chunk_ms` p50 10 ms/batch ⇒ dư địa ~0 |
| Q20 | ⚪ P3 | ⚪ không kiểm | Chỉ là nhiễu timer |
| E1 | 🟠 P1 | ✅ **TRUE** | `content-script.js:155-165` `ensureOverlay` → `attachToVideo` mỗi lần có video; `overlay-manager.js:97-104` **không guard identity** → `_attachHost()` + `_setupResizeObserver()` (`:271` disconnect, `:275` `new ResizeObserver`) + `_updateScale()` (`:219` `getBoundingClientRect`) |
| E2 | 🟠 P1 | 🟡 **PHÓNG ĐẠI** | `overlay-manager.js:79-85`: callback **đã có guard rẻ** `if (this.isActive && this.host && !this.host.isConnected)` ⇒ **không** forced layout trừ khi host thật sự rời DOM. Chi phí thật = 3 phép đọc thuộc tính/batch microtask (µs). Nên hạ P1 → **P3** |
| E3 | 🟠 P1 | ✅ **TRUE** | `service-worker.js:280` `api.tabs.sendMessage(sender.tab.id, {...})` — **không** truyền `{frameId}` ⇒ broadcast mọi iframe |
| E4 | 🟠 P1 | ✅ **TRUE** (số đúng) | `backpressure-gate.js:30-31`: SOFT `128*1024` B, HARD `512*1024` B; 16 kHz×2 B = 32 000 B/s ⇒ **4,1 s / 16,4 s** khớp đúng chú thích QWEN |
| E5 | 🟠 P1 | ✅ **TRUE** (tinh vi, đáng giá) | `audio-processor.js:62-85`: `ratio = 48 000/16 000 = 3,0` **nguyên**; `src` khởi tạo từ `phase` nguyên và cộng `ratio` nguyên ⇒ `frac = src - Math.floor(src) = 0` **luôn luôn** ⇒ `out[n] = s0` = **decimation trần, không lọc chống aliasing**. Mọi năng lượng > 8 kHz gập vào dải thoại |
| E6 | 🟠 P1 | ✅ **TRUE** | `ws-client.js:117-124`: listener `port.onDisconnect` nằm trong `if (!settled)`; sau khi phiên đã "settled" thì port chết **không** gọi `_scheduleReconnect` (mọi call-site `_scheduleReconnect` ở `:105,:164`, trong nhánh còn `!settled`) |
| E7 | 🟡 P2 | 🟡 **PHÓNG ĐẠI** | `popup.js:610` `target: { tabId, allFrames: true }` ✅ và `:817` `setTimeout(apply, 150)` ✅ — hai dữ kiện đúng; nhưng chỉ chạy khi user **thả/kéo slider**, không phải trên đường tiếng ⇒ P2 là hợp lý, không phải P1 |

---

## 2. ĐÍNH CHÍNH #1 — `preview_recompute_ratio` là **8,75–9,23×**: tôi đã sai, Gemini và QWEN đúng

### 2.1. Sai ở đâu

Harness `test_08_streaming_latency.run_paced(audio, model, speed=…)` đẩy audio vào **nhanh hơn thời
gian thật** `speed` lần. Nhưng số vòng preview lại do **thời gian thực** quyết định
(`poll_interval_ms=300`). Bộ đếm thì cộng theo **số vòng**:

```python
# backend/asr/engine.py:1317-1319
self._preview_sec_processed += len(audio_slice) / 16000.0
self._preview_sec_unique_peak = max(self._preview_sec_unique_peak, (current_total - seg_start) / 16000.0)
```

⇒ `ratio = Σ(audio mỗi vòng) / cửa sổ lớn nhất ≈ số vòng preview`, mà

```
số vòng ≈ độ_dài_câu / (poll_interval_ms/1000 × speed)
```

Chạy ở **8×** thì số vòng giảm **8 lần** ⇒ ratio đo được **1,00–1,67**, đúng bằng `3,8 / (0,3 × 8) = 1,58`.
Đây là **artefact thuần tuý của harness**, không phải hành vi hệ thống.

### 2.2. Đo lại ở 1× (realtime)

`external/build-tmp/g01_ratio_1x.py` (`SPEED=1.0`, `poll=300 ms`, `preview_window_sec=8.0`, backend CUDA):

| File | Audio | Đưa qua model | Audio thật | **ratio** | skip | gate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Japanese_5s.wav` | 5,1 s | **45,7 s** | 5,0 s | **9,23** | 0 | 0 |
| `Chinese_noise_28s.wav` | 28,3 s | **42,5 s** | 4,8 s | **8,75** | 0 | 0 |

Kết quả lưu tại `external/build-tmp/g01_recompute_1x.json`.

### 2.3. Kết luận đúng

| Nguồn | Ratio | Đánh giá |
| :--- | ---: | :--- |
| Gemini (bản gốc) | ~9× | ✅ **ĐÚNG** |
| QWEN (phiên thật, window 6 s) | 6,83× | ✅ **ĐÚNG** (window 8 s ⇒ ~9×, khớp tuyến tính) |
| **Tôi (8× pacing)** | 1,00–1,67× | ❌ **SAI** — artefact của harness |
| **Đo lại 1×** | **8,75–9,23×** | ✅ chuẩn |

**Hệ quả:** mỗi giây audio thật tiêu tốn **~9 giây** inference ASR. Với `preview p50 ≈ 93 ms` và
~3,3 vòng/s ⇒ **duty cycle ASR ≈ 30% GPU chỉ riêng cho preview**, và đây là chi phí GPU lớn nhất đo được của
hệ thống — lớn hơn dịch thuật. Khuyến nghị **C.1 "KHÔNG làm incremental/adaptive preview"** trong
`XAC_NHAN_GEMINI_VA_KE_HOACH_TOI_UU.md` (ngưỡng đặt là "ratio < 2,5 thì không làm") **phải đảo ngược**:
ratio thật vượt ngưỡng **3,5 lần**.

---

## 3. ĐÍNH CHÍNH #2 — QWEN-Q11 **SAI**: `silence_duration_ms=600` có tác dụng

### 3.1. Lập luận của QWEN

> Engine FireRed dùng `min_silence_frame` (60 × 25 ms = **1,5 s**) trong khi tầng cấu hình khai báo
> 600 ms; người dùng kéo "silence" trong popup chỉ có tác dụng với FSMN path ⇒ MỖI câu chốt trễ thêm
> ~900 ms.

### 3.2. Vì sao sai

QWEN giả định `frame_result.is_speech` (⇒ `VADResult.is_speech` ⇒ `processor.py:348 is_speech_frame`)
là **trạng thái máy trạng thái** (giữ `True` qua `POSSIBLE_SILENCE` tới `min_silence_frame`). Đọc
`stream_vad_postprocessor.py` thì không phải:

```python
# fireredvad/core/stream_vad_postprocessor.py:64,66-71
is_speech = self.apply_threshold(smoothed_prob)     # ← quyết định NGƯỠNG TỪNG FRAME
result = StreamVadFrameResult(is_speech=is_speech, …)   # :68
result = self.state_transition(is_speech, result)       # máy trạng thái chỉ ĐIỀN THÊM cờ
```

`min_silence_frame` **chỉ** điều khiển `result.is_speech_end` (`:154-156`) — và `processor.py` xử lý
`vad_event == "END"` ở `:382`, tức là **sau khi** nhánh `elif is_speech_frame:` ở `:391` đã có thể
chốt câu bằng bộ đếm riêng `silence_duration_ms` (`:399-420`).

### 3.3. Thực nghiệm (`external/build-tmp/g02_min_silence_frame.py`)

1 s "nói" (prob 0,9) rồi 2 s "im" (prob 0,0), chạy chính `StreamVadPostprocessor` với cấu hình thật
(`smooth=5`, `threshold=0.45`, `min_speech=8`, `min_silence=60`):

```
het tieng that su tai     : 1000 ms
frame dau `is_speech=False`: 1075 ms  (tre 75 ms)  [state may trang thai = POSSIBLE_SILENCE]
frame `is_speech_end=True` : 2550 ms  (tre 1550 ms)

=> bo dem im lang cua processor.py:399 bat dau chay tu 1075 ms, cong `silence_duration_ms=600`
   => chot cau tai ~1675 ms  (KHONG phai 2550 ms)

KET LUAN: QWEN-Q11 SAI.
```

75 ms trễ = đúng `smooth_window_size=5` frame × 25 ms. Trễ chốt câu thật ≈ **675 ms**, sát cấu hình
600 ms của người dùng — **không** phải 1500 ms.

### 3.4. Nhưng Q11 chỉ đúng ở một nửa: `min_silence_frame=60` là **config chết**

`min_silence_frame: int = 60  # 20 frames * 25 ms tối thiểu xác nhận kết thúc nói` (`config.py:71`)
— **comment nói 20 frame (500 ms) nhưng giá trị là 60 (1500 ms)**: comment lệch giá trị.
Và vì event END của engine luôn đến **sau** khi processor đã chốt câu, giá trị 60 **không có tác
dụng gì** với đường FireRed. Nên sửa: đặt `min_silence_frame` khớp `silence_duration_ms/25`
(hoặc ghi rõ nó là tham số chết của engine). Đây là **vệ sinh config**, không phải lỗi độ trễ.

---

## 4. ĐÍNH CHÍNH #3 — Q9 phóng đại: `/health` tốn **1,009 ms**, không chạm lock nóng

Đo 50 lần liên tiếp (`external/build-tmp/g03_health_cost.py`, sau khi bootstrap `bin/`):

```
50 lan goi: tong 50.45 ms | trung binh 1.009 ms/lan
  native_provider   : 0.0001 ms/lan
  library_path      : 0.0001 ms/lan
  backend_available : 0.0006 ms/lan
```

* `_asr_runtime_info()` (`main.py:107-168`) **không** lấy `_shared_lock` — nó gọi API module
  `transcribe_cpp` (`:128,130,134`) và đọc attribute `TranscribeEngine._shared_model` (`:159`).
  Claim "có thể chen vào giữa vòng lock nóng" là **không có cơ sở trong code**.
* 1 ms/lần × 1 lần vài giây = **~0,02–0,05% thời gian loop**. P2 → **P3**.

---

## 5. ĐÍNH CHÍNH #4 — Q6 phóng đại: `await` là backpressure tự nhiên

```python
# backend/ws/handler.py:472-473
loop = asyncio.get_running_loop()
await loop.run_in_executor(_VAD_EXECUTOR, _process_binary_chunk, session, data)
```

QWEN trích đúng dòng này nhưng bỏ qua chữ `await`: mỗi vòng đọc WS **chờ** future VAD xong mới đọc
khung kế. Vì vậy:

* in-flight **tối đa 1 task VAD / phiên** — không phải 15,6 task/s tích luỹ;
* độ sâu `_work_queue` bị chặn bởi **số phiên đồng thời**, không bởi thời gian;
* fix đề xuất `if _VAD_EXECUTOR._work_queue.qsize() > 64: drop` **không bao giờ chạy**.

Triệu chứng "sub trôi" mà QWEN gán cho Q6 thì **có thật, nhưng nguyên nhân là Q2** (worker kẹt trong
`hf_hub_download` giữ lock lớp) — lúc đó phiên bị chặn ở `await`, và audio tích ở **buffer transport**,
không phải ở hàng đợi executor. Sửa Q2 là đủ; đừng thêm cơ chế drop vô dụng.

---

## 6. ĐÍNH CHÍNH #5 — E2 phóng đại: guard rẻ đã có sẵn

```js
// extension_firefox/content/overlay-manager.js:79-85
this._domObserver = new MutationObserver(() => {
  if (this.isActive && this.host && !this.host.isConnected) {   // ← guard rẻ
    this._attachHost(); this._setupResizeObserver(); this._updateScale();
  }
});
this._domObserver.observe(document.body, { childList: true, subtree: true });
```

QWEN thừa nhận "(đã có)" guard này nhưng vẫn xếp E2 **P1** với lập luận "mỗi lần detached lại làm
nguyên chuỗi attach+RO rebuild". Đúng — **nhưng chỉ khi host thật sự rời DOM**, là biến cố hiếm.
Chi phí thường trực chỉ là **gọi callback rỗng 3 phép đọc** mỗi batch microtask ⇒ **µs**. Xếp **P3**.
Việc đáng làm vẫn là **E1** (guard identity trong `attachToVideo`), vì E1 chạy trên **mỗi sự kiện
phụ đề** chứ không phải mỗi DOM mutation.

---

## 7. FINDING ĐƯỢC XÁC NHẬN — chi tiết đáng chú ý

Ngoài bảng §1, ba finding xứng đáng được nhấn vì QWEN mô tả **chính xác đến mức hiếm gặp**:

### 7.1. Q1 — race `use-after-free` (P0, phải sửa trước mọi thứ khác)

`_release_llm` chỉ có `with cls._infer_lock: pass` — barrier này chặn được thread **chưa vào** vùng
inference, nhưng **không** chặn thread đã đọc pointer ở `:283` rồi đang xếp hàng chờ `_infer_lock`.
Cửa sổ race = toàn bộ `build_prompt` (`:293-300`, hàng trăm µs → ms). Kịch bản người dùng thật: đổi
model dịch từ popup khi video đang chạy ⇒ **segfault cả tiến trình**. Fix rẻ nhất đúng như QWEN đề
xuất (a): đọc lại `cls._shared_llm` **sau khi** đã giữ `_infer_lock`, và nếu khác pointer đã dùng để
build prompt thì dùng cái mới. Đối chiếu ASR đã làm đúng (`asr/engine.py:13-22`), nên đây là chỗ
**duy nhất còn hở** — nhất quán với kiến trúc hiện có.

### 7.2. Q5 — `reset()` phá hợp đồng identity (P1, dễ sửa 1 dòng)

```python
# processor.py:317 (hợp đồng)         # processor.py:443-449 (phá hợp đồng)
if self._state is not state:          def reset(self):
    metrics…("vad.batch_discarded…")      with self._lock:
                                              self._state.reset()   # ← in-place, identity KHÔNG đổi
```

Một batch đang chạy model khi `reset()` được gọi sẽ **không** bị nhận diện là cũ ⇒ ghi kết quả vào
state vừa reset ⇒ trộn câu cũ/mới. Fix: `self._state = VADStreamState()` (object **mới**) trong lock.

### 7.3. E5 — decimation 3:1 không lọc chống aliasing (P1 chất lượng, *không* A/B WER nào thấy)

`ratio` là **số nguyên** 3,0 nên `frac` luôn bằng 0 và vòng nội suy tuyến tính **suy biến thành
decimation trần**. Đây là phát hiện mà toàn bộ harness A/B WER hiện tại **không thể** phát hiện, vì
`test_09` feed file **16 kHz sẵn** — không đi qua resampler. Muốn kiểm chứng phải A/B trên dữ liệu
48 kHz thật. Fix 3 dòng (boxcar 3-tap trước khi decimate) rẻ và rõ ràng.

---

## 8. KẾ HOẠCH CẬP NHẬT (theo thứ tự ưu tiên đã hiệu chỉnh)

> **TIẾN ĐỘ:** mục **1 (Q1)** và **2 (Q2)** — hai lỗi P0 — **✅ ĐÃ SỬA XONG** (xem §8b).
> Người dùng chốt "Phase A = Q1 + Q2" và đồng ý `git rm --cached bin/*.dll` (§4.2).

| # | Việc | Nguồn | Trạng thái | Vì sao thứ tự này |
| ---: | :--- | :--- | :--- | :--- |
| 1 | **Q1** — re-validate `_shared_llm` trong `_infer_lock` (~15 dòng) | QWEN P0 | ✅ **XONG** | Chống **crash**, không phải tối ưu; rủi ro thấp, không cần đo |
| 2 | **Q2** — không build VAD engine trong `__init__`; `hf_hub_download` ngoài lock lớp (~90 dòng) | QWEN P0 | ✅ **XONG** | Dẹp treo-loop toàn cục |
| 3 | **Q7a** — bật `preview_reuse_for_commit` + A/B WER | Gemini + QWEN | ❌ **BÁC BỎ** (đo được: chất lượng **xấu hơn sàn nhiễu** +0,27/+0,46 pp; tiết kiệm −17,4 % audio-giây commit) | Giữ mặc định `False`; xem §8c |
| 4 | **Q7b / C.1** — adaptive window + incremental preview | Đo ratio 8,75–9,23× | ⏳ kế tiếp | Nay là ưu tiên cao: duty cycle preview ~30 % GPU |
| 5 | **Q5** — `reset()` tạo state mới | QWEN P1 | ✅ **XONG** | Rẻ, sửa lỗi đúng-đắn |
| 6 | **Q4** — `force_end` gọi callback ngoài lock | QWEN P1 | ✅ **XONG** | Rẻ, xoá lớp deadlock tiềm ẩn |
| 7 | **Q3** — prewarm TTS trong lifespan; translation dùng `prewarm()` | QWEN P1 | ✅ **XONG** | Cắt spike câu đầu |
| 8 | **E1 + E3** — guard `attachToVideo`; `sendMessage({frameId: 0})` | QWEN P1 | ✅ **XONG** | Hết forced-layout mỗi sự kiện + hết broadcast mọi iframe |
| 9 | **E5** — boxcar trước decimation | QWEN P1 | ✅ **XONG** | 9 kHz −4,6 dB, 12 kHz −9,5 dB; dải thoại giữ 99,9 % |
| 10 | **E6** — reconnect khi port chết sau khi nối | QWEN P1 | ✅ **XONG** | Fix "phiên sống sót ảo" |
| 11 | **Q8** — executor thứ hai cho commit | QWEN P1 | ⏳ chờ số | Chỉ làm nếu p99 vẫn cao sau Q7b |
| 12 | **Q10, Q14, Q15, Q11** | QWEN P2 | ✅ **XONG** | Vệ sinh + sửa cái bẫy mặc định của Q10 |
| 13 | **Q12** (per-session config), Q16, Q17, Q19, Q20 | QWEN P2/P3 | ⏳ để sau | Q12 là refactor kiến trúc; Q16/Q17/Q19/Q20 đo được ≈ 0 |
| — | ~~Q6 drop-frame~~ | **bỏ** | ❌ | Cơ chế sai (§5): `await` đã là backpressure |
| — | ~~Q9 cache TTL /health~~ | **hạ P3** | ❌ | Đo được 1,009 ms (§4) |
| — | **A1-1 + 2 lỗi harness + bootstrap-order** | **tự phát hiện** | ✅ **XONG** | Xem §8c mục "4 lỗi phát hiện thêm" |

---

## 8b. NHẬT KÝ THỰC HIỆN PHASE A (Q1 + Q2) — 2025-09-17

### Q1 — race `use-after-free` model dịch ✅

| Mục | Nội dung |
| :--- | :--- |
| **File sửa** | `backend/translation/engine.py` |
| **Cách sửa** | Thêm `_snapshot_infer_state()` (đọc `_shared_llm` + `_shared_model_key` + `cfg` + `prompt_strategy` **trong** `_infer_lock`), `_shared_config_snapshot()` (chụp `(key, cfg, strategy)` dưới `_shared_lock` **trước** khi build prompt — vì `reconfigure` ghi 5 field trong cùng một lock, đọc rời có thể ghép key mới với prompt cũ) và `_prepare_infer()`. Cả `_translate_sync` lẫn `_translate_stream_sync` nay: (1) chỉ kiểm tra sớm `_shared_llm is None`, (2) chụp `(key, cfg, strategy)` rồi build prompt ngoài lock như cũ, (3) **snapshot lại trong lock**, (4) nếu `_shared_model_key` đã đổi thì **build lại prompt** cho khớp model sẽ chạy |
| **Test** | `backend/tests/test_32_translation_use_after_free.py` — 7 test, tái hiện race **tất định** (tự giữ `_infer_lock` rồi `close()` model cũ) + chốt AST cho cả ba helper |
| **Vì sao an toàn** | `_release_llm` chỉ `close()` **sau khi đã đi qua** `_infer_lock` ⇒ con trỏ đọc trong lock chắc chắn còn sống. Thứ tự `_infer_lock` → `_shared_lock` là thứ tự duy nhất được phép trong dự án; `load_model`/`reconfigure` nhả `_shared_lock` trước khi chờ `_infer_lock` ⇒ không deadlock |
| **Test** | `backend/tests/test_32_translation_use_after_free.py` — 6 test, tái hiện race **tất định** (tự giữ `_infer_lock` rồi `close()` model cũ) |
| **Chứng minh** | `external/build-tmp/g04_q1_proof.py`: code CŨ → **gọi model đã close (1 lần) ⇒ nổ**; code MỚI → 0 lần, chạy trên model mới. Chạy lại toàn bộ test 32 với code cũ (git stash): **FAIL toàn bộ** |
| **Kiểm chứng tích hợp** | Toàn bộ suite `backend/tests`: **311 passed, 0 failed**. Backend khởi động thật (`python backend\main.py`) → `lifespan` prewarm ASR+Translation+VAD sạch, `GET /health` → `status=ok`, `asr.backend=CUDA0`, `native_bundle_dir=…\bin` |

### Q2 — nạp VAD engine trên event loop + lock cấp lớp qua `hf_hub_download` ✅

| Mục | Nội dung |
| :--- | :--- |
| **File sửa** | `backend/vad/base.py`, `backend/vad/engines/__init__.py`, `backend/vad/engines/firered.py`, `backend/vad/engines/fsmn.py`, `backend/vad/processor.py`, `backend/main.py` |
| **Cách sửa** | (1) `BaseVADEngine.prepare_files()` (classmethod) gom việc **tải file model**; `get_engine` gọi nó **NGOÀI** `_lock`, dưới `_prepare_lock` **riêng** (chỉ chặn các luồng đang tải, không chặn `is_cached`/`peek_engine`); `get_engine` dùng double-checked locking. (2) `VADProcessor.__init__` chỉ `peek_engine` (đọc biến, rẻ) — chưa có thì **hẹn nạp ở thread nền**, và **không dựng state** (state để VAD worker dựng ở chunk đầu, vì `create_initial_state` của Silero nạp JIT). (3) `_ensure_engine()` thành **non-blocking**, trả `False` khi engine chưa có; `feed_chunk` **bỏ chunk** + đếm `vad.chunks_dropped_engine_not_ready` thay vì tải model trên đường hot. (4) Thêm `VADProcessor.prewarm()` (đồng bộ) + tham số `auto_load=False`; `main._prewarm_vad_default` dùng nó |
| **Đo trên engine THẬT** (`external/build-tmp/g05_q2_verify.py`) | `VADProcessor(...)` trước prewarm: **0,7 ms** (code cũ khi phải nạp: **500 ms** — xem bằng chứng dưới). Sau prewarm: **0,0 ms**. `prewarm()`: 33 ms, nạp **đúng 1 lần**. Audio thật 14 s: 4 START / 4 END / 404 chunk, ratio 0,13× realtime ⇒ **phân câu vẫn nguyên vẹn** |
| **Test** | `backend/tests/test_33_vad_engine_off_loop.py` — 9 test (gồm chốt AST: `prepare_files` phải nằm trong `_prepare_lock`; `_ensure_engine` không được gọi `get_engine`; `__init__` không được gọi `_ensure_engine`) |
| **Chứng minh** | Chạy test 33 với code CŨ (git stash): **9/9 FAIL**, trong đó test quyết định báo đúng `VADProcessor.__init__ chặn 500ms — nó đang nạp engine trên event loop` |
| **Kiểm chứng tích hợp** | Backend khởi động thật: log `[VAD] Model FireRed Stream sẵn sàng` → `[VAD] [STARTUP] Engine 'firered-vad' sẵn sàng` xuất hiện **đúng MỘT lần** (trước khi thêm `auto_load=False` thì nạp 2 lần do thread nền và `prewarm()` chạy song song). Ba engine (firered/fsmn/silero) đều nạp xong ở nền |

### Ảnh hưởng tới hành vi (cần biết)

* **Q1:** khi `reconfigure()` đổi model đúng vào lúc một câu đang build prompt, câu đó được
  build lại prompt theo model mới ⇒ không còn khả năng gửi prompt của model cũ cho model mới.
  Chi phí thêm: 0 ở đường thường (không swap), ~0,1–1 ms ở đúng câu bị swap.
* **Q2:** trong cửa sổ "engine chưa nạp xong", các chunk audio đến sớm bị **BỎ** (không còn
  treo backend). Đường thường gặp không bao giờ rơi vào đây vì `lifespan` đã prewarm engine
  mặc định qua `asyncio.to_thread` **trước khi phục vụ request**. Chỉ xảy ra khi: đổi sang
  engine **chưa tải** (lần đầu), prewarm khởi động thất bại, hoặc `reset_pool()`. Có log
  WARNING một lần + counter `vad.chunks_dropped_engine_not_ready` để chẩn đoán.
* **Q2:** `VADProcessor.__init__` không còn dựng session state. Mọi đường dùng state đều đã
  kiểm `None` (`force_end`, `reset`) hoặc đi qua `_ensure_engine` (`feed_chunk`).

---

## 8c. NHẬT KÝ ĐỢT 2 — Q7a, Q5, Q4, Q3, Q10/Q14/Q15/Q11, E1/E3/E5/E6 — 2025-09-17

### Q7a — **BÁC BỎ**: bật `preview_reuse_for_commit` KHÔNG miễn phí

A/B ở **1× realtime** (đúng pacing thật), **backend CUDA trong `bin/`** (đúng bản phát hành),
8 file × 3 repeat × 3 cấu hình. Script tổng hợp: `external/build-tmp/g08_wer_summary.py`.

| | error vs GT | vs REF (mất mát do phân đoạn) |
| :--- | ---: | ---: |
| `base` (reuse OFF) | 41,25 % | 17,85 % |
| `base_repeat` (đối chứng, y hệt base) | 41,23 % | 17,95 % |
| **`reuse` (ON)** | **41,52 %** | **18,31 %** |
| **Sàn nhiễu** (base vs base_repeat) | **0,02 pp** | **0,10 pp** |
| **Δ của `reuse`** | **+0,27 pp** | **+0,46 pp** |
| Kết luận | **XẤU HƠN sàn nhiễu** | **XẤU HƠN sàn nhiễu** |

Chênh lệch tập trung ở **một file**: `English_low_speech_quality_19s` 11,83 % → **13,98 %**
(+2,15 pp); 6/8 file còn lại **giống hệt** hoặc lệch ≤0,2 pp. Đúng loại dữ liệu mà preview
không ổn định ⇒ lấy text preview làm bản chốt sẽ sai hơn decode lại.

**Đổi lại, tiết kiệm GPU là THẬT** (`external/build-tmp/g09_q7a_cost.py`, 1× realtime, CUDA,
2 file ≈31 s audio):

| | reuse OFF | reuse ON | Δ |
| :--- | ---: | ---: | ---: |
| Số lần decode ở bước commit | 10 | 7 | **−30,0 %** |
| Audio-giây phải decode ở commit | 30,55 s | 25,22 s | **−17,4 %** |
| `commit_ms` tổng | 776 ms | 606 ms | −22 % |
| Số lần lấy thẳng kết quả preview | 0 | 2 | — |

⇒ Khớp đúng con số QWEN dự đoán (−15–20 %), nhưng **kèm mất mát chất lượng đo được**. Vì vậy
**KHÔNG đổi mặc định** (`preview_reuse_for_commit` giữ `False`); ai ưu tiên tốc độ hơn độ
chính xác thì tự bật, nay đã có số để cân.

> Ghi chú phương pháp: gauge `audio_seconds_processed_preview` **reset mỗi câu** nên không
> cộng dồn được qua các câu — dòng "TỔNG audio qua model" trong `g09` là **artifact của
> script**, không dùng. Chỉ số đáng tin là phần **commit** (là mẫu latency, cộng dồn được).

### Các mục P1/P2/P3 đã sửa trong đợt 2

| Mục | Sửa gì | Bằng chứng |
| :--- | :--- | :--- |
| **Q5** | `reset()` gán `VADStreamState` **MỚI** (`_new_bare_state`, O(1)) thay vì sửa in-place ⇒ batch đang bay bị bỏ đúng như hợp đồng identity ở `processor.py:317`. State rỗng là đủ vì cả 3 engine tự khởi tạo cache ở frame kế tiếp, nên **không** nạp lại JIT của Silero trên event loop | `test_34` (8 test): 4 test **FAIL** với code cũ; có ca tất định "batch qua `reset()` phải bị bỏ" |
| **Q4** | `force_end()` gom callback rồi gọi **NGOÀI** `self._lock` (giống `feed_chunk` bước 3) ⇒ hết lớp deadlock VAD→ASR + head-of-line | `test_34`: test "trong lúc callback chạy, `_lock` phải đã được nhả" **FAIL** với code cũ |
| **Q3** | `lifespan` thêm prewarm **TTS** (có điều kiện `config.tts.enabled`); `_prewarm_translation` đổi `load_model()` → **`prewarm()`** (chạy 1 câu giả để build kernel CUDA) | `test_35` (5 test) |
| **Q10** | Gate `config.metrics.enabled` **thật** ở đầu mọi hàm ghi. ⚠️ **Phải đổi mặc định `enabled` False → True**: bản cũ để `False` mà việc ghi vẫn luôn chạy, nên `False` mới là hành vi thực tế — áp QWEN nguyên văn mà giữ `False` sẽ **âm thầm tắt toàn bộ metrics** (đã tái hiện: **22 test fail**) | `test_36`, `test_30`; toàn suite xanh lại |
| **Q14** | Cleanup phiên gọi `dump_metrics_report` qua `asyncio.to_thread` (không serialize + `json.dump` trên loop) | `test_36`, `test_30` (chốt "phải đẩy sang thread") |
| **Q15** | `_get_voice_clone_prompt` dùng **lock theo từng giọng** (`_lock_for_voice`): hai phiên khác giọng vẫn tính song song, hai phiên **cùng** giọng chỉ tính **một** lần | `test_36`: 4 luồng cùng giọng ⇒ `create_voice_clone_prompt` gọi **1 lần** (trước: 4) |
| **Q11** | Sửa comment `min_silence_frame` (ghi sai số khung) + ghi rõ đây là **config CHẾT** với FireRed và trỏ sang `silence_duration_ms` | `test_36` |
| **E1** | `attachToVideo` có guard identity + `_hostInPlace()` (dùng `parentElement.contains(video)` — không ép layout, vẫn tự chữa khi video đổi container); `_setupResizeObserver` idempotent theo đối tượng đang theo dõi | `overlay_attach_guard_test.js`: **20 lần** gọi liên tiếp ⇒ **0** ResizeObserver mới, **0** `getBoundingClientRect`, **0** `getComputedStyle` (bản cũ: 20 observer + 40 lần đo layout) |
| **E3** | `tabs.sendMessage(tabId, msg, { frameId: 0 })` — nhắm frame TRÊN CÙNG thay vì broadcast mọi iframe | `sw_broadcast_frame_target_test.js` (bản cũ FAIL: không có options) |
| **E5** | Boxcar `ratio` tap trước decimation khi tỉ lệ **nguyên** ≥ 2, có mang lịch sử mẫu qua biên block (tránh artifact 2,7 ms) | `resample_alias_test.js`: 9 kHz còn **58,8 %** (−4,6 dB), 12 kHz còn **33,3 %** (−9,5 dB); 440 Hz giữ **99,9 %**; số mẫu ra không đổi. Bản cũ: **100 %** cho cả 9 và 12 kHz (gập phổ nguyên biên độ) |
| **E6** | `port.onDisconnect` (nhánh đã nối) phát `disconnected` + `_scheduleReconnect()` | `ws_reconnect_on_port_death_test.js` (bản cũ FAIL: không hẹn nối lại) |

### 4 lỗi PHÁT HIỆN THÊM khi làm (không có trong báo cáo QWEN/Gemini)

1. **Harness WER đo SAI BACKEND.** `test_09_wer_ab.py` `import transcribe_cpp` trực tiếp
   **trước** khi `backend.asr` bootstrap ⇒ native của **wheel PyPI** (chỉ Vulkan) vào
   `sys.modules`, bundle `bin/` (CUDA) **không áp được** (log: *"đã được import TRƯỚC khi
   bootstrap"*). Cùng cái bẫy đó có ở **`conftest.py`** và **`test_08_streaming_latency.py`**
   ⇒ **cả suite test** cũng chạy trên wheel. Đã sửa cả 3 (bootstrap trước, vẫn giữ đúng yêu
   cầu thứ tự nạp DLL). Chốt hồi quy: `test_37`.
2. **CPU bị đốt ~11 core bởi chính harness** — và đây là hệ quả của (1). Đo trực tiếp:
   wheel/Vulkan giữ **~11 thread ở 80–90 % CPU LIÊN TỤC** kể cả khi `nvidia-smi` báo GPU 0 %;
   sau khi trỏ về `bin/`+CUDA còn **~0,5 core** (giảm ~24×). Không phải Node, không phải
   OpenBLAS, không phải Vulkan busy-wait — mà là đường Vulkan của wheel.
3. **A1-1 chỉ được áp ở `main.py`.** Guard chống oversubscription thread chỉ nằm trong
   `backend/main.py`, nên harness/benchmark/pytest chạy bằng mặc định OpenMP/OpenBLAS. Đã
   chuyển lên `backend/__init__.py` (chạy trước mọi submodule) + bổ sung
   **`OPENBLAS_NUM_THREADS`** (OpenBLAS đọc biến này TRƯỚC `OMP_NUM_THREADS`; numpy ở đây
   build trên `scipy-openblas`) + `OMP_WAIT_POLICY=PASSIVE`. Dùng `setdefault` nên người dùng
   vẫn ép được số luồng. Chốt: `test_37`.
4. **`MetricsConfig.enabled` mặc định `False`** trong khi việc ghi vốn luôn chạy (xem Q10 ở
   trên) — đây là lý do QWEN-Q10 nhìn ra "config nói dối", và cũng là cái bẫy nếu sửa nửa vời.

---

## 9. ARTEFACT & CÁCH TÁI LẬP

| File | Nội dung |
| :--- | :--- |
| `external/build-tmp/g01_ratio_1x.py` | Đo `preview_recompute_ratio` ở pacing thật 1× (sửa lỗi 8× cũ) |
| `external/build-tmp/g08_wer_summary.py` | Tổng hợp A/B WER: trung bình theo config + **sàn nhiễu** + Δ có kết luận |
| `external/build-tmp/g09_q7a_cost.py` | Đo phần **tiết kiệm GPU** của `preview_reuse_for_commit` bằng counter commit |
| `external/build-tmp/g06_thread_spin.py` | Đo số core CPU mà VAD + numpy dùng (1,1 core sau guard) |
| `external/build-tmp/g07_asr_cpu_spin.py` | Đo core theo cửa sổ: nạp / suy luận / **rảnh** (phát hiện spin-wait) |
| `external/build-tmp/q7a_wer_ab_1x.json` | Kết quả thô A/B 1× trên CUDA (8 file × 3 repeat × 3 config) |
| `backend/tests/js/overlay_attach_guard_test.js` | E1 — đếm ResizeObserver/rect thật (20 lần gọi ⇒ 0) |
| `backend/tests/js/sw_broadcast_frame_target_test.js` | E3 — broadcast phải có `frameId: 0` |
| `backend/tests/js/resample_alias_test.js` | E5 — đo gập phổ 9 kHz / 12 kHz, giữ dải thoại |
| `backend/tests/js/ws_reconnect_on_port_death_test.js` | E6 — port chết sau khi nối ⇒ hẹn nối lại |
| `external/build-tmp/g01_recompute_1x.json` | Kết quả: 9,23 / 8,75× |
| `external/build-tmp/g02_min_silence_frame.py` | Thực nghiệm `is_speech` vs `is_speech_end` với `fireredvad` thật |
| `external/build-tmp/g03_health_cost.py` | Đo chi phí `_asr_runtime_info()` cho `/health` |

Chạy lại:

```powershell
python external\build-tmp\g01_ratio_1x.py           # ~90 s
python external\build-tmp\g02_min_silence_frame.py  # ~1 s
python external\build-tmp\g03_health_cost.py        # ~2 s
```

---

## 10. KẾT LUẬN

Báo cáo QWEN **đáng tin và nên dùng làm cơ sở hành động**. Hai P0 đều thật, finding lớn nhất (Q7)
thậm chí bị QWEN **đo thấp hơn thực tế**, và các đường dẫn file/dòng đều chính xác. Chỉ cần:

* **bỏ Q11** (đã chứng minh sai bằng thực nghiệm với chính thư viện);
* **hạ Q6, Q9, E2** (cơ chế/mức độ sai — Q6 dựa trên đọc thiếu chữ `await`);
* **đảo ngược quyết định C.1** trong kế hoạch Gemini của tôi: incremental/adaptive preview **nay là
  ưu tiên cao**, vì ratio thật là **8,75–9,23×**, không phải 1,0–1,67× như tôi đã báo cáo sai.

Bài học phương pháp: **một harness đo nhanh hơn thời gian thật có thể bóp méo chính metric mà nó đo**
— `preview_recompute_ratio` tỉ lệ thuận với tốc độ pacing, nên phải luôn chạy ở **1×** khi con số
được dùng để ra quyết định.

---

## 11. KẾT LUẬN ĐỢT 2

**Đã sửa xong 11 mục** (Q1, Q2, Q3, Q4, Q5, Q10, Q11, Q14, Q15 + E1, E3, E5, E6), **bác bỏ 1 mục**
(Q7a) bằng số đo, và **tự phát hiện + sửa 4 lỗi** không có trong bất kỳ báo cáo audit nào — trong đó
lỗi nặng nhất là **harness WER (và cả suite test) đang đo trên native của wheel thay vì `bin/`**, kéo
theo **~11 core CPU bị đốt liên tục** (sau khi sửa còn **~0,5 core**).

Ba bài học đáng giữ:

1. **Một con số chỉ có nghĩa khi biết nó được đo ở đâu.** Cùng `test_09_wer_ab.py`: chạy wheel thì ra
   Vulkan + 11 core CPU; chạy `bin/` thì ra CUDA + 0,5 core. Không đọc kỹ dòng `backend=` trong log
   thì mọi kết luận về sau đều sai — và tôi đã suýt kết luận sai lần thứ hai trong cùng một ngày.
2. **Sửa theo mô tả mà không kiểm chứng hệ quả có thể gây hại nhiều hơn để nguyên.** QWEN-Q10 mô tả
   đúng ("config nói dối"), nhưng áp nguyên văn thì **tắt metrics toàn hệ thống** vì mặc định là
   `False`. Phát hiện được là nhờ **chạy toàn suite**: 22 test đổ.
3. **Đo cả hai vế của một trade-off.** `preview_reuse_for_commit` tiết kiệm thật (−17,4 % audio-giây
   commit, khớp dự đoán 15–20 % của QWEN) nhưng **mất chính xác trên audio chất lượng thấp**
   (+2,15 pp ở một file, +0,27 pp tổng, so với sàn nhiễu 0,02 pp). Nếu chỉ đo phần tiết kiệm thì đã
   bật mặc định sai.
