# XÁC NHẬN BÁO CÁO AUDIT GEMINI + KẾ HOẠCH TỐI ƯU

> **Nguồn thẩm định:** `report/audit/BAO_CAO_AUDIT_HIEU_NANG_Gemini.md` (2026-09-16)
> **Phương pháp:** đọc mã tại đúng `file:line` + **đo thật trên máy** (RTX 5060 Ti, Qwen3-ASR-0.6B,
> Hy-MT2-1.8B-Q8, `bin/` CUDA bundle). Không chấp nhận con số ước lượng khi có thể đo.
> **Nguyên tắc:** mỗi kết luận phải kèm bằng chứng tái lập được. Con số không tái lập được thì
> **không** đưa vào kế hoạch.

---

## 0. TÓM TẮT ĐIỀU HÀNH

> ⚠️ **ĐÍNH CHÍNH 2025-09-17:** mục G-01 trong tài liệu này **đã sai** và đã được sửa (§2.1).
> Số đo "1,0–1,67×" là **artefact của harness chạy ở 8× tốc độ thật**; đo lại ở **1× realtime cho
> 8,75–9,23×**. **Gemini đúng ở finding "preview 9×".** Xem
> `report/audit/XAC_NHAN_QWEN_VA_DINH_CHINH.md`.

Báo cáo Gemini đúng về **phần lớn cấu trúc** (đọc mã chính xác, `file:line` khớp), nhưng
**một số con số định lượng bị phóng đại**, và **2/15 finding sai hẳn**. Nếu áp dụng
nguyên văn lộ trình của họ, ta sẽ:

- ~~Đầu tư vào "incremental Mel cache"…~~ → **NGƯỢC LẠI: đây chính là việc nên làm** (ratio thật
  8,75–9,23×, xem §2.1b). Mục này là **kết luận sai của tôi**, không phải của Gemini.
- Viết lại kiến trúc đa phiên theo một mô hình mà **thư viện native tuyên bố không hỗ trợ**.
- Giải phóng "450 MB VRAM" từ KV cache — con số thật là **33,6 MB**.
- "Sửa" một cache vốn **đã được chặn ở 8 entry**.

### Bảng tổng kết thẩm định

| Kết luận | Số lượng | Gồm |
| :--- | :---: | :--- |
| ✅ Đúng, giữ nguyên mức độ | 8 | **G-01 (đã sửa — trước đây tôi xếp sai vào nhóm dưới)**, G-05, G-07, G-08, G-09, G-11, G-12, G-13 |
| ⚠️ Đúng về cấu trúc, **SAI về mức độ** | 2 | G-04 (20–40% → không tái lập), G-10 |
| 🔶 Đúng nhưng **fix đề xuất KHÔNG an toàn** | 1 | G-03 |
| 🔶 Đúng nhưng **không đáng làm** | 1 | G-15 |
| ❌ **SAI hẳn** | 2 | G-02, G-14 |
| ❓ Không kiểm chứng được / phóng đại | 1 | con số VRAM & "audio drift 5–15 s" |

**Con số then chốt của báo cáo Gemini (sau đính chính):**

| Gemini nói | Đo/kiểm thực tế | Kết luận |
| :--- | :--- | :--- |
| Preview lãng phí **9×** | **8,75 – 9,23×** (1× realtime) | ✅ **Gemini ĐÚNG** — số "1,00–1,67×" cũ của tôi là artefact 8× pacing |
| `n_ctx=512` tốn **~450 MB** VRAM | **33,6 MB** (32 layer × 4 KV-head × 128 dim × 2 × 2 byte) | ❌ phóng đại **~13×** |
| Vulkan+CUDA gây **+20–40%** latency | p50 **không đổi** (187–199 vs 195–202 ms) | ❌ không tái lập |

---

## 1. BẢNG XÁC NHẬN CHI TIẾT (15 finding)

| ID | Gemini claim | Kết luận | Bằng chứng |
| :--- | :--- | :---: | :--- |
| **G-01** | Preview recompute **9×** trên cửa sổ 6 s | ✅ **ĐÚNG** (⚠️ **tôi từng bác bỏ sai**) | Số đo cũ của tôi (**1,00 / 1,67 / 1,58**) chạy ở **8× pacing** nên bị bóp méo — ratio tỉ lệ thuận với tốc độ pacing (`asr/engine.py:1317`). **Đo lại 1× realtime: 8,75 – 9,23×**; QWEN đo phiên thật: 6,83× (window 6 s). Gemini đúng. Xem §2.1 + `XAC_NHAN_QWEN_VA_DINH_CHINH.md` §2 |
| **G-02** | Tạo **Session Pool** N session dùng chung weights để chạy đa phiên | ❌ **SAI** | `transcribe_cpp.Model.__doc__`: *"at most one run/stream may be IN FLIGHT across all sessions of a model at a time — sessions share the model's compute backend, so overlapping runs race... or **load one Model per worker** for true parallelism."* Xem §2.2 |
| **G-03** | `_infer_lock` giữ suốt generator streaming | 🔶 **Đúng, fix không an toàn** | `translation/engine.py:388-409` xác nhận. Nhưng **1 `llama_context` dùng chung**: nhả lock giữa các chunk cho phép inference khác ghi đè KV cache ⇒ hỏng dữ liệu. Xem §2.3 |
| **G-04** | Vulkan(ASR)+CUDA gây **+20–40%** latency do context switch | ⚠️ **Không tái lập** | Đo 2 thứ tự: ASR p50 Vulkan 199,4/186,9 vs CUDA 201,9/194,8 (**không chênh**). CUDA chỉ tốt hơn ở **đuôi** p95 (237→213, 249→203). Xem §2.4 |
| **G-05** | Tensor→CPU→numpy→concat = 4 bản sao | ✅ **Đúng** | `tts/audio_processor.py:42` `.detach().cpu().numpy().astype(np.float32)` — `astype` mặc định `copy=True` nên tạo bản sao **thừa** khi đã là float32; rồi `np.concatenate` (dòng 54) tạo bản nữa |
| **G-06** | Phase Vocoder SciPy tốn **45–120 ms** | ✅ Đúng, **nhưng mặc định KHÔNG chạy** | `audio_processor.py:70` thoát sớm nếu `abs(rate-1.0) < 0.02`; `TTSConfig.speed = 1.0` ⇒ **không nằm trên hot path mặc định** |
| **G-07** | Encode WAV rồi browser decode lại | ✅ **Đúng** | `audio_processor.py:137` `sf.write(... PCM_16)` → extension `decodeAudioData` |
| **G-08** | Một mutex cho mọi thao tác metrics | ✅ **Đúng** | `core/metrics.py:44` một `self._lock` cho `_latencies`/`_gauges`/`_counters` + mọi getter |
| **G-09** | `_send_lock` gây head-of-line blocking | ✅ **Đúng** | `ws/connection.py:36` một lock; `send_text` (63) và `send_bytes` (81) dùng chung |
| **G-10** | `del raw_buf[:offset]` dịch chuyển mảng | ✅ Đúng, **mức độ nhỏ** | `vad/processor.py:277,290`. Nhưng buffer bị chặn ở **3 s = 96 KB** (dòng 274) nên memmove rất nhỏ |
| **G-11** | `write_bytes` cấp phát mảng mỗi chunk | ✅ **Đúng** | `core/audio_buffer.py:96` `frombuffer(int16).astype(float32)` tạo mảng mới mỗi lần |
| **G-12** | `GpuArbiter` polling 15 ms | ✅ **Đúng** (tôi viết) | `core/gpu_scheduler.py` — cố ý dùng poll vì `commit_begin()` chạy **trong thread ASR**, không phải event loop |
| **G-13** | `findVideo()` duyệt 20 000 node | ✅ **Đúng** (tôi viết) | `content-script.js` — `MAX_SHADOW_SCAN_NODES = 20000`, đã có throttle 250 ms |
| **G-14** | Cache VoiceClonePrompt **không giới hạn** ⇒ rò VRAM | ❌ **SAI** | `tts/engine.py:66-67`: `OrderedDict`, `_voice_prompt_cache_max = 8`; dòng 157-158 **có eviction LRU** `popitem(last=False)`. Không có rò rỉ |
| **G-15** | `_coalesce_enqueue` tạo CPU burst O(K) | 🔶 Đúng, **không đáng sửa** | `ws/handler.py:227-267` rút/nạp lại hàng đợi. Nhưng `K ≤ 32` và chỉ chạy **khi đã quá tải**; dịch chuyển 32 tham chiếu dict là hàng nano-giây |

### Các claim khác không kiểm chứng được hoặc phóng đại

| Claim | Đánh giá |
| :--- | :--- |
| `n_ctx=512` tốn ~450 MB VRAM KV cache | ❌ **Sai ~13×.** Kiến trúc thật (đọc GGUF metadata): 32 layer, 16 head, **4 KV-head**, embed 2048 ⇒ head_dim 128. KV f16 = 2×32×512×4×128×2 = **33,6 MB** |
| "Audio drift 5–15 s sau 5 phút" | ❓ Chưa đo. Không có cơ chế bù tốc độ phát trong extension, nên **có thể đúng** — nhưng cần phiên thật để xác nhận |
| "Spike commit p99 lên 1,2–1,8 s" | ❓ Không tái lập. Đo được commit max **167–270 ms** (kể cả khi bị TTS/dịch tranh chấp) |
| "Allocation churn 15–25 MB/s" | ❓ Không đo được bằng chứng. Ước lượng từ danh sách copy — nghe hợp lý nhưng chưa có số |
| "GPU dành 85% năng lực để nhận dạng lại" | ❌ Suy ra từ con số 9× sai ⇒ **không đúng** |

---

## 2. ĐÍNH CHÍNH CHI TIẾT

### 2.1. G-01 — ⚠️ **ĐÍNH CHÍNH: số đo 1,0–1,67× CỦA TÔI LÀ SAI. Ratio thật là 8,75–9,23× — Gemini ĐÚNG.**

> **Sửa ngày 2025-09-17.** Mục dưới đây đã **bị bác bỏ bằng đo lại**. Giữ nguyên phần cũ để truy vết
> lỗi phương pháp, nhưng **kết luận đúng** nằm ở §2.1b và ở
> `report/audit/XAC_NHAN_QWEN_VA_DINH_CHINH.md` §2.

**Lỗi phương pháp:** harness `run_paced(..., speed=…)` đẩy audio **nhanh hơn thời gian thật** `speed`
lần, nhưng số vòng preview do **thời gian thực** quyết định (`poll_interval_ms=300`). Bộ đếm cộng theo
**số vòng** (`asr/engine.py:1317`), nên `ratio ≈ số vòng ≈ độ_dài_câu / (0,3 × speed)`. Chạy ở **8×**
⇒ ratio bị đo **thấp giả tạo đúng ~8 lần** (1,58 ≈ 3,8 / (0,3 × 8)). Con số 1,0–1,67× là **artefact
của harness**, không phải hành vi hệ thống — và nó đã dẫn tôi tới khuyến nghị **sai** ở mục C.1.

**Cách đo cũ (SAI):** chạy `test_08_streaming_latency.run_paced` ở 8× tốc độ →
`_stream_asr_tokens`, đọc gauge `asr.preview_recompute_ratio` mà code đã instrument sẵn (FIX-10).
Script: `external/build-tmp/g01_measure_ratio.py`.

#### 2.1b. Đo lại ở **1× (realtime)** — `external/build-tmp/g01_ratio_1x.py`

`SPEED=1.0`, `poll=300 ms`, `preview_window_sec=8.0`, backend CUDA:

| File | Audio | Đưa qua model | Audio thật | **ratio** |
| :--- | ---: | ---: | ---: | ---: |
| `Japanese_5s.wav` | 5,1 s | **45,7 s** | 5,0 s | **9,23** |
| `Chinese_noise_28s.wav` | 28,3 s | **42,5 s** | 4,8 s | **8,75** |

| Nguồn | Ratio | Đánh giá |
| :--- | ---: | :--- |
| Gemini (bản gốc) | ~9× | ✅ **ĐÚNG** |
| QWEN (phiên thật, window 6 s) | 6,83× | ✅ **ĐÚNG** |
| Tôi (8× pacing) | 1,00–1,67× | ❌ **SAI — artefact harness** |
| **Đo lại 1×** | **8,75–9,23×** | ✅ chuẩn |

⇒ Mỗi giây audio thật tốn **~9 giây** inference ASR. **Gemini đúng, QWEN đúng, tôi sai.**
Khuyến nghị **C.1 phải đảo ngược** (xem bảng cuối §3).

---

#### (nội dung gốc, đã bị bác bỏ — chỉ để truy vết)

**Cách đo:** chạy **chính pipeline streaming thật** (`test_08_streaming_latency.run_paced` →
`_stream_asr_tokens`), đọc gauge `asr.preview_recompute_ratio` mà code đã instrument sẵn (FIX-10).
Script: `external/build-tmp/g01_measure_ratio.py`.

| File | Dài | Audio đưa qua model | Audio thật | **ratio** |
| :--- | ---: | ---: | ---: | ---: |
| `English_low_speech_quality_19s` | 19,0 s | 0,9 s | 0,9 s | **1,00** |
| `English_multiple_kinds_of_noise_88s` | 88,2 s | 9,5 s | 5,7 s | **1,67** |
| `Chinese_noise_28s` | 28,3 s | 9,9 s | 6,3 s | **1,58** |

**~~Vì sao Gemini sai~~ (lập luận ĐÃ BỊ BÁC BỎ — các "cơ chế chặn" dưới đây CHỈ giới hạn số vòng
xuống ~9 vòng/câu, KHÔNG đưa ratio về 1,0; chúng không hề mâu thuẫn với con số 9×):**

> 1. `preview_adaptive_backoff` (P2.4b) — **tự giãn nhịp** khi inference chậm (`preview_slow_ms=350`).
> 2. `preview_yielded_to_commit` — bỏ vòng preview khi có commit đang chờ.
> 3. `defer_preview` — chỉ cho `max_inflight_infer=1` inference cùng lúc.
> 4. `preview_window_sec` chặn trần (hiện mặc định **8.0**, không phải 6.0), và
>    `_finalize_recompute_metrics` **reset bộ đếm mỗi câu**. → Điểm 4 chính là điều làm phép đo ở
>    **8× pacing** sai: bộ đếm reset mỗi câu + số vòng do thời gian thực quyết định.

**Ý nghĩa cho kế hoạch (ĐÃ SỬA):** với `preview_window_sec=8`, ratio thật **8,75–9,23×** ⇒ cận trên lợi
ích từ incremental/reuse preview **không phải ~40% mà là ~85–90%** compute ASR. Duty cycle ASR ≈ 30%
GPU **chỉ riêng cho preview** ở 1 phiên realtime. Đây **là ưu tiên CAO**, không phải P3: đảo ngược
hoàn toàn khuyến nghị cũ ở mục C.1 (§3).

### 2.2. G-02 — Kiến trúc "Session Pool chia sẻ weights" **không khả thi**

Gemini viết: *"`transcribe.cpp` chỉ cho phép 1 luồng suy luận trên một `transcribe_session`. Nhưng ta
hoàn toàn có thể tạo Session Pool (N sessions trỏ chung vào 1 Model weights)"*.

**Sai.** Binding tài liệu hoá rõ ràng giới hạn ở cấp **model**, không phải cấp session:

> *"Known 0.x limitation: at most one run/stream may be IN FLIGHT **across all sessions of a model** at a
> time — sessions share the model's compute backend, so overlapping runs race. Run sessions serially (a
> pool behind a lock is fine), or **load one Model per worker** for true parallelism."*

Nghĩa là: nhiều session trên **một** model vẫn phải chạy **tuần tự**. Muốn song song thật phải nạp
**một model cho mỗi worker** ⇒ **nhân bản weights** ⇒ VRAM × N. Sơ đồ của Gemini (weights dùng chung +
context độc lập) **không đạt được** nếu không sửa `transcribe.cpp`.

**Kết luận:** G-02 **không phải bug** mà là **giới hạn đã biết của thư viện**, và dự án đã ghi rõ trong
README ("1 phiên / 1 video"). Giữ nguyên. Nếu muốn đa phiên thật, việc đầu tiên là **đo VRAM × N** —
với ASR 1,95 GB + dịch 4,6 GB + TTS 2,3 GB thì 2 phiên đã vượt 16 GB.

### 2.3. G-03 — Quan sát đúng, nhưng **fix đề xuất sẽ gây hỏng dữ liệu**

Gemini đề xuất Task 2.1: *"Chỉ giữ `_infer_lock` khi lấy chunk token tiếp theo từ generator, giải phóng
lock giữa các chunk"*.

**Không được làm.** Toàn bộ engine dịch dùng **một** `Llama` object = **một** `llama_context`. Khi
`stream=True`, llama.cpp **giữ trạng thái KV cache trong suốt quá trình sinh token**. Nhả lock giữa các
chunk cho phép một luồng khác gọi `llm(...)` trên **cùng context** ⇒ ghi đè KV cache ⇒ đầu ra rác hoặc
crash. Lock bọc cả vòng lặp là **điều kiện đúng đắn**, không phải bug.

**Điều đúng của G-03:** hệ quả là engine dịch bị độc quyền ~200–800 ms mỗi câu. Cách sửa **an toàn duy
nhất** là tách **context riêng** (chính là G-02) — mà G-02 lại bị chặn bởi `transcribe.cpp` cho phần ASR.
Nên đây là **đánh đổi của thiết kế 1 phiên**, không phải việc sửa được bằng cách đổi lock.

### 2.4. G-04 — Không tái lập được "+20–40%"

**Cách đo:** model dịch thật (Hy-MT2-1.8B-Q8, CUDA) chạy **liên tục** trong khi commit ASR thực thi;
lặp lại với `Model(backend="vulkan")` và `Model(backend="cuda")` **trong cùng tiến trình** (loại trừ
mọi khác biệt khác). Chạy **2 thứ tự** để loại trừ hiệu ứng warmup.

| Chỉ số | ASR=Vulkan | ASR=CUDA | Chênh |
| :--- | ---: | ---: | ---: |
| asr_p50 (vulkan-first) | 199,4 ms | 201,9 ms | **+2,5** |
| asr_p50 (cuda-first) | 186,9 ms | 194,8 ms | **+7,9** |
| asr_p95 (vulkan-first) | 236,9 ms | **213,1 ms** | −23,8 |
| asr_p95 (cuda-first) | 249,1 ms | **203,0 ms** | −46,1 |

**Đọc kết quả:**
- **p50 hoàn toàn không chênh** (187–199 vs 195–202) ⇒ **không có "context switch penalty" 20–40%**.
- Cả hai thứ tự đều cho thấy CUDA **tốt hơn ở đuôi** p95/max ~20–46 ms.
- Lưu ý phương pháp: `trans_p50_baseline` lệch tới 60 ms giữa hai lượt ⇒ phép đo **baseline translation
  nhạy với thứ tự/warmup**; tôi **không** kết luận gì từ cột `trans_slowdown_pct`.

**Kết luận:** CUDA là một **thắng lợi nhẹ ở đuôi**, không phải "tranh chấp phần cứng phải sửa gấp".
Việc đổi default sang CUDA (bạn đã commit) là hợp lý vì lý do khác — **nhanh hơn 1,53× khi đo tách rời**
(§4.1.3 của `KE_HOACH_FIX_LOI_Hy3.md`) — chứ không phải vì tranh chấp.

### 2.5. G-14 — Cache **đã** bounded, không có rò VRAM

`tts/engine.py`:

```python
self._voice_prompt_cache: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()  # LRU, max 8 entries
self._voice_prompt_cache_max: int = 8
...
if len(self._voice_prompt_cache) > self._voice_prompt_cache_max:
    evicted_key, _ = self._voice_prompt_cache.popitem(last=False)   # evict LRU
```

Gemini khẳng định "Cache VoiceClonePrompt không giới hạn số lượng tensor trên VRAM" — **trái với mã**.
Đây là lỗi đọc mã. **Không sửa gì.**

---

## 3. KẾ HOẠCH TỐI ƯU (chỉ gồm việc đã có bằng chứng)

### Phase A — Làm ngay (rủi ro thấp, rẻ, đo được)

> **TRẠNG THÁI: đã thực hiện xong.** Kết quả thật ở §3.0 bên dưới — 2/5 làm, **3/5 bác bỏ sau khi đo**
> (A.2 và A.4 hoá ra không đáng hoặc không an toàn; A.1 lợi ích nhỏ hơn dự kiến).

| # | Việc | File | Kết quả |
| :--- | :--- | :--- | :--- |
| **A.1** | `convert_to_numpy`: `.astype(np.float32)` → `np.asarray(..., dtype=np.float32)` | `tts/audio_processor.py:40,42` | ✅ **Đã làm** — lợi ~8 µs/câu trên đường ndarray, ~0 trên đường torch |
| **A.2** | `_fast_dumps`: bỏ `.decode("utf-8")`, gửi JSON qua `send_bytes` | `ws/connection.py:23` | ❌ **BÁC BỎ** — phá giao thức, client mất hết phụ đề |
| **A.3** | `GpuArbiter`: `asyncio.Event` + `call_soon_threadsafe`, giữ poll làm trần | `core/gpu_scheduler.py` | ✅ **Đã làm** — bỏ tới 15 ms trễ thức giấc; có test xuyên-thread |
| **A.4** | Tái sử dụng buffer cho `write_bytes` + `SpeechNormalizer` | `core/audio_buffer.py`, `core/normalizer.py` | ❌ **BÁC BỎ** — `write_bytes` không có caller production; `normalize` tốn CPU ở số học chứ không ở alloc, và tái dùng buffer là **race** |
| **A.5** | Ghi rõ điều kiện kích hoạt đường chậm (`gpu.scheduler_enabled`, `tts.speed != 1.0`, `n_ctx`) | `backend/config.py` | ✅ **Đã làm** |

#### 3.0. Kết quả đo thực tế Phase A

**A.1 — lợi ích NHỎ hơn Gemini mô tả.** Benchmark 300 lần (`external/build-tmp/phaseA_measure.py`),
audio 3 s @ 24 kHz:

| Kiểu đầu vào | `astype` (cũ) | `asarray` (mới) | Tiết kiệm |
| :--- | ---: | ---: | ---: |
| ndarray float32 | 21 µs | 13 µs | **38 %** |
| ndarray float64 | 53 µs | 51 µs | 4 % |
| torch float32 *(đường TTS thật)* | 19 µs | 20 µs | ~0 (nhiễu) |
| torch float16 | 27 µs | 38 µs | nhiễu |

`AudioProcessor.convert_to_numpy` trên đường thật = **15 µs**. Với TTS ~420 ms/câu, mức tiết kiệm
vài µs là **~0,002 %** ⇒ **giữ thay đổi** (vô hại, giảm churn) nhưng **không coi là thắng lợi hiệu năng**.
Con số "15–25 MB/s allocation churn" của Gemini **không tái lập được**.

**A.2 — phải bác bỏ vì phá giao thức.** Client phân loại frame bằng **opcode**:

- `extension_firefox/lib/ws-client.js:359` — `if (typeof event.data === "string")` mới `JSON.parse`.
- `extension_firefox/background/service-worker.js:198` — text ⇒ `ws_json`; **binary ⇒ `ws_binary`**.
- `extension_firefox/content/content-script.js:330` — frame binary **bắt buộc** magic `"BTTS"`, nếu không
  thì `return` (bỏ luôn).

⇒ Gửi JSON qua `send_bytes` sẽ khiến client coi là audio TTS, trượt magic và **vứt toàn bộ phụ đề/bản
dịch**. Starlette chỉ nhận `str` cho frame text nên **không có cách nào bỏ `.decode()` mà giữ opcode text**.
Đổi được thì phải sửa giao thức extension — để tiết kiệm ~1–2 µs/payload. **Không đáng.**

**A.3 — thắng lợi thật, đã có test chứng minh.** Polling cũ ngủ đủ `poll_s` (15 ms) rồi mới kiểm tra lại.
Nay `commit_end()` (chạy trong **thread ASR**) gọi `loop.call_soon_threadsafe(ev.set)` ⇒ waiter thức ngay.
`poll_s` được giữ làm **trần** một lần chờ nên nếu tín hiệu bị lỡ thì hành vi tệ nhất **y hệt bản cũ**.

Test mới trong `backend/tests/test_29_gpu_arbiter.py`:
- `test_event_wakes_waiter_immediately_from_another_thread` — đặt `admit_poll_ms = 500` rồi gọi
  `commit_end()` từ **thread pool**: `admit()` phải trả về **< 350 ms** (nếu còn poll thuần sẽ là ~500 ms).
- `test_event_notify_is_safe_without_waiters` — không raise khi chưa có waiter.
- `test_waiter_on_new_loop_gets_fresh_event` — hai `asyncio.run()` liên tiếp (loop khác nhau) đều được đánh thức.

**A.4 — bác bỏ vì tiền đề sai.** Hai lý do độc lập:

1. **`CircularAudioBuffer.write_bytes` không có caller production nào.** `grep` toàn `backend/`:
   chỉ `tests/test_26_copy_and_config.py:57` gọi nó. Đường ASR thật dùng
   `feed_audio()` → `audio_buffer.write(float32)`. Tối ưu một hàm không ai gọi là vô nghĩa.
2. **`SpeechNormalizer.normalize` đã tối ưu từ P2.6** (RMS bằng `np.dot`, peak bằng `max(a.max(),-a.min())`,
   **đúng 1** `np.empty_like` + `multiply`/`clip` tại chỗ, im lặng thì trả về chính mảng vào = 0 alloc).
   Đo: **82 µs** cho 3 s @ 24 kHz — nhưng thời gian đó nằm ở **số học** (`multiply` + `clip` trên 72 000 mẫu),
   **không** ở cấp phát. Tái dùng buffer chỉ tiết kiệm phần alloc (~µs) mà lại gây **race**: `audio_to_infer`
   được truyền vào `session.run()` trong executor, và `_preload_model()` cũng gọi `_run_inference_sync`
   từ **thread khác** — buffer dùng chung có thể bị ghi đè giữa chừng.


### Phase B — Có lợi nhưng cần đo lại sau Phase A

| # | Việc | File | Điều kiện |
| :--- | :--- | :--- | :--- |
| **B.1** | Tách `MetricsCollector._lock` thành 3 lock theo domain (`_latencies_lock`, `_gauges_lock`, `_counters_lock`) | `core/metrics.py` | G-08. Chỉ đáng làm nếu đo được contention — hiện **chưa có bằng chứng** `record_metric` nằm trong top profile. Ưu tiên thấp |
| **B.2** | Tách lane gửi WS: text phụ đề ưu tiên, audio TTS chunk 8–16 KB | `ws/connection.py`, `ws/handler.py` | G-09. Trên localhost frame TTS 50–150 KB đi rất nhanh; **đo trên mạng thật** trước |
| **B.3** | VAD: thay `del raw_buf[:offset]` bằng con trỏ đọc vòng | `vad/processor.py:277,290` | G-10. Buffer bị chặn 3 s nên lợi ích nhỏ; nhưng **rẻ** vì không đụng ngữ nghĩa — ứng viên tốt nhất của Phase B |

### Phase C — Chỉ làm sau khi có số đo trên **phiên thật**

#### 3.9b. KẾT QUẢ A/B TRÊN PHIÊN THẬT — `gpu.scheduler_enabled` ON vs OFF

**Phép đo:** hai phiên ~4,4 phút (262 s vs 257 s) trên cùng máy, cùng `qwen3-asr-1.7b` + `tencent`,
**cùng `asr.backend = "cuda"`**. File: `metrics_sched_ON.json` / `metrics_sched_OFF.json`,
phân tích bằng `external/build-tmp/compare_sched_ab.py`.

**Kiểm tra tính hợp lệ trước (quan trọng nhất):** so `run_config` của hai file — **chỉ đúng MỘT
trường khác nhau**:

| Trường | ON | OFF |
| :--- | :--- | :--- |
| `gpu.scheduler_enabled` | **True** | **False** |
| *16 trường còn lại* | giống hệt | giống hệt |

⇒ Phép A/B **hợp lệ**: biến duy nhất được thay đổi là scheduler.

**Kết quả — KHÔNG có khác biệt có ý nghĩa:**

| Chỉ số | n ON | ON p50 | ON p95 | n OFF | OFF p50 | OFF p95 | Δp50 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `asr.commit_ms` | 57 | 107,0 | 203,6 | 61 | 99,9 | 220,9 | **+7,1** |
| `pipeline.e2e_asr_to_sub_ms` *(metric người dùng cảm nhận)* | 55 | 435,8 | 1138,0 | 58 | 445,0 | 1143,1 | **−9,2** |
| `translation.infer_ms` | 55 | 433,0 | 1136,5 | 58 | 429,0 | 1141,8 | +4,0 |
| `asr.preview_ms` | 663 | 91,5 | 191,5 | 678 | 86,9 | 192,9 | +4,6 |
| `translation.first_token_ms` | 55 | 33,3 | 59,3 | 59 | 34,8 | 67,2 | −1,5 |
| `gpu.admit_wait_ms` | **3** | 0,3 | 37,1 | 0 | — | — | — |

**Vì sao arbiter gần như không làm gì:** `blocked_total = 3` trên **57 commit** trong 4,4 phút; tổng
thời gian chờ chỉ **41,8 ms** cho cả phiên. Nguyên nhân là số học đơn giản:

- `commit_end()` **đóng cửa sổ ngay** khi commit xong (tôi cố ý làm vậy để không bắt dịch/TTS chờ vô
  ích) ⇒ cửa sổ hiệu dụng ≈ **thời lượng commit (~100–300 ms)**, KHÔNG phải `commit_reserve_ms=1500`.
- Dịch chạy một câu mỗi ~4,7 s (`infer_ms` 433 ms, 55 câu / 257 s).
- ⇒ Xác suất một lần *bắt đầu* dịch rơi vào cửa sổ commit ≈ 55 × (0,2/4,7) ≈ **2–3 lần** — khớp đúng
  `blocked_total = 3`.

**DIỄN GIẢI PHẢI NÓI RÕ:** con số ấn tượng ở §4.3b của `KE_HOACH_FIX_LOI_Hy3.md` (commit p50 **−49 %**)
đo trên **tải tổng hợp bão hoà** — dịch chạy *liên tục back-to-back*. Trong dùng thật, dịch có
**những khoảng nghỉ lớn** (RTF ~0,03, thừa ~30× công suất) nên ASR và dịch **hiếm khi chồng lấn**.
Scheduler chỉ có ích khi có tranh chấp, mà thực tế gần như không có.

**Quyết định A.6 — giữ nguyên lựa chọn của bạn, nhưng biết rõ nó không mua gì:**
- Lợi ích đo được trên phiên thật: **không có** (Δp50 cả hai chiều, đều ≤ 10 ms ≈ 1–2 %).
- Chi phí đo được: cũng **không đáng kể** (`translation.queue_wait_ms` p99 ON 368,9 vs OFF 180,7 ms —
  nhưng với n≈55 thì p99 gần bằng max, một mẫu xấu là đủ lệch).
- `p99` của `e2e_asr_to_sub_ms` ON 1696 vs OFF 1358 ms: **cũng n≈55 ⇒ không kết luận được**.
- ⇒ Bật hay tắt đều **không sai**. Nếu muốn ít rủi ro nhất cho độ trễ dịch: đặt `False`.
  **Khuyến nghị: đặt `False`** — arbiter chỉ chạm 3 lần/4,4 phút (tổng 41,8 ms), tức không mang
  lại lợi ích đo được, trong khi `translation.queue_wait_ms` p99 có dấu hiệu xấu hơn (368,9 vs
  180,7 ms). **Giữ nguyên mã** vì nó đã có test và hoạt động đúng — chỉ là chưa có tranh chấp
  thật để nó phát huy.

  > **Hệ quả quan trọng cho kế hoạch:** nút thắt e2e là **dịch (430 ms)**, không phải ASR (100 ms).
  > Mọi đề xuất tối ưu ASR/preview — kể cả G-01 của Gemini — nhắm vào **~19 %** độ trễ e2e. Muốn
  > cải thiện p95 e2e (đang 1,14 s, vượt mục tiêu "< 1,0 s" ~14 %) thì phải tấn công **đường dịch**:
  > dùng model nhỏ hơn (`tencent-1.8b` thay vì `tencent` 7B), giảm `max_tokens`, hoặc tăng
  > `n_batch`. Đây mới là hướng đúng — và **chưa** nằm trong kế hoạch Phase A/B/C hiện tại.

⚠️ **Giới hạn của phép đo này (phải nói rõ):** n≈55–61 mẫu mỗi cấu hình ⇒ **p99 vô nghĩa** (bằng
gần đúng giá trị max); chỉ **p50/p95** đáng tin. Ngoài ra nội dung hai phiên không giống hệt
(`speech_start` 23 vs 24, `subtitles_delivered` 55 vs 58) nên vẫn còn ~5 % khác biệt về tải.

#### 3.9d. BASELINE THẬT CỦA HỆ THỐNG (từ hai phiên này)

Số đo trực tiếp, `asr.backend = cuda`, `qwen3-asr-1.7b` + dịch `tencent` (Hy-MT2-7B Q4), TTS bật,
n≈55–61 câu mỗi phiên:

| Chặng | p50 | p95 | Ghi chú |
| :--- | ---: | ---: | :--- |
| VAD chunk | 10,0 ms | 12,3 ms | ổn định |
| ASR preview | ~87–92 ms | ~192 ms | README ghi p95 108 ms ⇒ **cần cập nhật** |
| ASR commit | ~100–107 ms | ~204–221 ms | **khớp** README (~107 ms) |
| ASR normalize | 0,1 ms | 0,5 ms | không đáng tối ưu |
| Dịch — token đầu | ~33–35 ms | ~59–67 ms | README ghi 26 ms, gần đúng |
| Dịch — cả câu | ~429–433 ms | ~1137 ms | chiếm **phần lớn** e2e |
| TTS queue | 0,0 ms | — | không nghẽn |
| **E2E: ASR commit → phụ đề tới client** | **~436–445 ms** | **~1138–1143 ms** | mục tiêu "< 1,0 s" ⇒ **p50 đạt, p95 vượt ~14 %** |
| WS send | 0,1 ms | 0,2 ms | localhost, không phải nút thắt |

**Đọc baseline này ra hành động:** nút thắt e2e **không phải** ASR (100 ms) mà là **dịch cả câu
(430 ms)**. Muốn giảm p95 e2e thì phải tấn công đường dịch, không phải preview/ASR. Điều này củng cố
kết luận ở §2.1: tối ưu incremental preview (G-01) là **sai mục tiêu** — nó chỉ chạm 100 ms ASR, trong
khi 430 ms dịch mới là phần lớn độ trễ.

#### 3.9e. Hai vấn đề chất lượng báo cáo phát hiện khi phân tích
1. **Ba chỉ số KHÔNG phải latency đang bị chấm như latency.** `_finalize_recompute_metrics()`
   (`asr/engine.py:958-963`) gọi `record_metric` cho `audio_seconds_processed_preview`,
   `audio_seconds_unique_preview` và `preview_recompute_ratio` ⇒ chúng xuất hiện trong `stages` với
   `p50_ms`… trong khi đơn vị thật là **giây** và **tỉ lệ**. Đó là lý do bảng trên có dòng
   `asr.audio_seconds_processed_preview p50 = 26,1` trông như 26 ms nhưng thực ra là 26 giây.
   Đề xuất: chỉ ghi bằng `record_gauge`. **Chưa sửa** vì `test_28_recompute_metrics.py` đang assert
   `"asr.preview_recompute_ratio" in snapshot["stages"]` — cần đổi cả test.
2. **README lệch số so với đo thật:** README ghi *"preview p95 108 ms"* nhưng phiên thật cho
   **192 ms** (cả ON lẫn OFF). Con số 108 ms có thể đo trước khi `preview_window_sec=6.0` (P2.3) làm
   preview dài hơn. Nên cập nhật README theo số mới.

| # | Việc | Điều kiện tiên quyết |
| :--- | :--- | :--- |
| **C.1** | Incremental / adaptive preview (mel cache) | ✅ **NÊN LÀM — ưu tiên cao** (đảo ngược khuyến nghị cũ). Ngưỡng đặt ra là "ratio < 2,5 thì KHÔNG làm"; đo lại ở **1× realtime cho ratio 8,75–9,23×**, vượt ngưỡng **~3,5 lần**. Bước rẻ nhất trước: **bật `preview_reuse_for_commit`** (`config.py:178`, code có sẵn, chạy `test_09` A/B WER). Số đo cũ "1,0–1,67 ⇒ chưa đủ lý do" là **SAI** (artefact 8× pacing) |
| **C.2** | Dynamic `n_ctx` cho dịch | Tiết kiệm thật **~8 MB** (384 vs 512 ctx). **Không đáng** — loại |
| **C.3** | GPU time-stretch (torch.stft trên CUDA) | Chỉ có lợi khi `speed != 1.0`. Đo thật thời gian `apply_time_stretch()` trước; nếu < 60 ms thì để nguyên |

### Phase D — KHÔNG LÀM (đã bác bỏ bằng bằng chứng)

| Việc Gemini đề xuất | Lý do bác bỏ |
| :--- | :--- |
| Task 1.1: "Ép `asr.backend='cuda'` mặc định, chấm dứt Vulkan+CUDA" | G-04 **không tái lập**: p50 không chênh. Vulkan là đường `transcribe.cpp` **hỗ trợ chính thức** nên phải giữ làm fallback. *(Bạn đã đặt default `cuda` — hợp lý vì tốc độ 1,53×, nhưng lý do **không phải** tranh chấp.)* |
| Task 2.1: "Nhả `_infer_lock` giữa các chunk" | **Nguy hiểm** — hỏng KV cache (§2.3) |
| Task 3.1: Incremental Mel cache / bật `preview_reuse_for_commit` | ✅ **NÊN LÀM — ưu tiên cao.** Lợi ích thật **8,75–9,23×** ở 1× realtime ⇒ tới **~90% compute ASR preview** (§2.1b). Khuyến nghị cũ "1,0–1,67× ⇒ không đáng" là **SAI** (artefact 8× pacing) |
| Task 3.2: Session Pool chia sẻ weights | **Không khả thi** — binding cấm (§2.2) |
| G-14: "bounded cache cho VoiceClonePrompt" | Đã bounded ở 8 entry từ trước (§2.5) |
| G-15: ring-buffer queue | `K ≤ 32`, chỉ khi quá tải ⇒ không đáng (§1) |
| "Dynamic LLM context giảm 450 MB" | Thật là 33,6 MB; giảm được ~8 MB ⇒ không đáng |
| Batching TTS 2 câu | Chưa có bằng chứng `tts_queue` sâu thường xuyên; đổi batch làm phức tạp và **tăng latency câu đầu** |

---

## 3.9. BUG PHÁT HIỆN KHI CHẠY THẬT — `metrics_report.json` chưa bao giờ được ghi

**Phát hiện từ:** người dùng chạy một phiên 5 phút với `gpu.scheduler_enabled = True` rồi báo
*không tìm thấy `metrics_report.json` ở đâu cả*.

**Nguyên nhân — ba mảnh của cùng một tính năng, không mảnh nào được đấu dây:**

| Mảnh | Vị trí | Trạng thái trước khi sửa |
| :--- | :--- | :--- |
| Config `dump_report_on_disconnect`, `report_file`, `enabled` | `config.py:318-320` | **Không dòng nào trong `backend/` đọc** — config chết hoàn toàn |
| Method `MetricsCollector.dump_json()` / `generate_report()` | `metrics.py:110,132` | `generate_report()` chỉ được gọi ở `/api/metrics`; **`dump_json()` không có caller nào** |
| Đường cleanup phiên | `ws/handler.py` khối `finally` | Chỉ unregister + cancel worker + giải phóng tài nguyên, **không ghi report** |

⇒ Tính năng "dump metrics khi ngắt kết nối" tồn tại ở dạng *thiết kế* (có config, có method, có
docstring) nhưng **chưa bao giờ chạy**. Đây là dạng lỗi im lặng đúng nghĩa: không có ngoại lệ, không
có log, chỉ là file không xuất hiện.

**Đã sửa:**

1. `backend/core/metrics.py` — thêm `resolve_report_path()` và `dump_metrics_report(reason)`.
   Tên tương đối (`"metrics_report.json"`) được **neo vào `PROJECT_ROOT`**, không theo CWD — trước
   đây `dump_json` phân giải theo CWD nên chỗ rơi phụ thuộc cách khởi chạy, rất khó tìm.
   Hàm **không bao giờ raise** (ghi report không được làm hỏng đường cleanup).
2. `backend/ws/handler.py` — gọi `dump_metrics_report()` trong `finally` của cleanup phiên khi
   `config.metrics.dump_report_on_disconnect` bật (nên vẫn ghi dù phiên đóng do lỗi).
3. `backend/main.py` — **lưới an toàn** ở shutdown (Ctrl+C / kill / phiên không đóng sạch), và
   endpoint `POST /api/metrics/dump` để chốt số liệu **giữa phiên** mà không cần ngắt kết nối.
4. Report giờ **tự mô tả**: thêm `runtime` (trạng thái `GpuArbiter`, backend ASR đang dùng) và
   `run_config` (allowlist cờ ảnh hưởng hiệu năng). Không có hai mục này thì file A/B vô dụng vì
   mở ra không biết số liệu được đo với cấu hình nào.
5. `.gitignore` — thêm `metrics_report.json` (artifact sinh ra, không phải mã nguồn).
6. Test mới `backend/tests/test_30_metrics_report.py` (8 test), trong đó có 2 test **chốt ở mức mã
   nguồn** rằng đường disconnect và đường shutdown **phải** gọi dump — để bug này không tái diễn.

**Bài học:** config không có người đọc là config chết, và một method không có caller sẽ **không bao
giờ** xuất hiện trong test nào (test chỉ gọi thứ nó biết). Đây là loại lỗi mà test hiện có **không
thể** phát hiện, vì mọi test đều xanh trong khi tính năng không chạy.

---

## 4. HAI VIỆC CẦN QUYẾT ĐỊNH NGAY

### 4.1. README và `config.py` về backend mặc định — ✅ **ĐÃ THỐNG NHẤT**

| Nguồn | Giá trị |
| :--- | :--- |
| `backend/config.py:139` (đã commit) | `backend: str = "cuda"` |
| `README.md` (đã commit) | *"`asr.backend` — mặc định `"auto"` — auto (Vulkan trước, rồi CUDA)"* |
| `backend/asr/native.py` `_PREFERENCE["auto"]` | `("vulkan", "cuda")` |

Ba nguồn nói ba kiểu. Cần chốt **một** hành vi rồi sửa hai chỗ còn lại. Đề xuất của tôi (khớp với
lựa chọn bạn đã commit):

- `config.asr.backend = "cuda"` — mặc định dùng CUDA (nhanh hơn 1,53×, đã đo).
- `_PREFERENCE["auto"] = ("cuda", "vulkan")` — `auto` = "tốt nhất có thể"; vì Vulkan là đường **chính
  thức** nên nó là **fallback**, không phải ưu tiên đầu.
- README ghi: *mặc định `cuda`, tự **fallback về Vulkan** (đường transcribe.cpp hỗ trợ chính thức) nếu
  bundle không có `ggml-cuda.dll`, có WARNING nêu rõ.*

### 4.2. `bin/*.dll` (≈180 MB) đã bị **commit vào git**

Commit `8160f76` thêm `bin/ggml-base.dll`, `bin/ggml-cpu.dll`, `bin/ggml-cuda.dll` (71 MB),
`bin/ggml-vulkan.dll` (52,6 MB) — dù `.gitignore` có `bin/` và `*.dll`. Mỗi lần build lại CUDA sẽ tạo
diff nhị phân 71 MB, làm repo phình nhanh và không review được. Cân nhắc `git rm --cached bin/*.dll` +
phân phối bundle qua release artifact/script build.

---

## 5. THỨ TỰ THỰC HIỆN ĐỀ XUẤT

```
XONG      : Phase A — A.1 ✅ · A.3 ✅ · A.5 ✅ · A.2 ❌ bác bỏ · A.4 ❌ bác bỏ
Tiếp      : đo M1/M2/M5 (§6.2 của KE_HOACH_FIX_LOI_Hy3.md) trên một phiên video thật
Phase B   : B.3 → B.1 → B.2               (chỉ sau khi có số)
Phase C   : chỉ C.1, và chỉ nếu ratio > 2,5
Bỏ        : toàn bộ Phase D
```

**Nguyên tắc chốt:** không đưa vào kế hoạch bất kỳ việc nào chỉ dựa trên con số ước lượng. Mọi mục
đều phải có **số đo trước và sau** mới được coi là hoàn thành — và phải chấp nhận kết quả **bác bỏ**
khi số đo không ủng hộ (như A.2 và A.4 ở Phase A).

---

**Tài liệu liên quan:** `BAO_CAO_AUDIT_HIEU_NANG_Gemini.md` (bản gốc),
`BAO_CAO_AUDIT_HIEU_NANG_QWEN.md` (audit QWEN),
`XAC_NHAN_QWEN_VA_DINH_CHINH.md` (**thẩm định QWEN + 5 đính chính, gồm đính chính §2.1 của tài liệu này**),
`KE_HOACH_FIX_LOI_Hy3.md` (§4.1.2–4.3b: gate build CUDA, hạ tầng `bin/`, GpuArbiter),
`05_measurements_and_status.md`.
**Script đo sinh ra tài liệu này:** `external/build-tmp/g01_measure_ratio.py` (§2.1 — **lỗi 8× pacing**),
`g01_ratio_1x.py` (**đo lại 1× — số đúng**), `g04_contention.py` (§2.4), `g_kv_verify.py` (§1),
`g02_min_silence_frame.py` + `g03_health_cost.py` (thẩm định Q11/Q9).
