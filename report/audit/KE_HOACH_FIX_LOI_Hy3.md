# KẾ HOẠCH FIX LỖI CHI TIẾT — theo `BAO_CAO_AUDIT_HIEU_NANG_Hy3.md`

> **Người thực hiện:** Senior Performance Engineer
> **Nguồn:** `report/audit/BAO_CAO_AUDIT_HIEU_NANG_Hy3.md`
> **Phạm vi:** `backend/**`, `extension_firefox/**`
> **Ràng buộc đã tuân thủ:**
> 1. Các lỗi **không liên quan A2-1** được sửa **ngay**.
> 2. **A2-1** (ASR Vulkan + Dịch/TTS CUDA tranh chấp 1 GPU) chỉ **lập kế hoạch**, thực thi **sau cùng**, có **gate** phụ thuộc việc build được ASR chạy CUDA. Các fix liên quan cũng lùi lại theo.
> 3. **KHÔNG commit git.** Mọi thay đổi nằm ở working tree.

---

## 0. TÓM TẮT ĐIỀU HÀNH

| Nhóm | Nội dung | Trạng thái |
| :--- | :--- | :--- |
| **Phase 1** | 7 fix không liên quan A2-1 | ✅ **ĐÃ SỬA XONG**, đã chạy test |
| **Phase 2** | 7 đề xuất trong audit bị **LOẠI** sau khi kiểm chứng bằng mã + test | ✅ Đã ghi bằng chứng |
| **Phase 3** | A2-1 — GPU scheduler | 📋 **CHỈ KẾ HOẠCH**, làm sau cùng |
| **Phase 4** | A3-1 — leak native (patch upstream) | 📋 Kế hoạch, ngoài phạm vi repo này |

**Kết quả chạy test sau Phase 1:** bộ mặc định (`pytest -m "not slow and not full"`) thu thập **268 test**:
**249 PASS**, **0 FAILED**, **19 ERROR**.
Toàn bộ 19 `ERROR` nằm trong `test_22_translation_model_download.py` và đều là lỗi **môi trường**, không
phải lỗi mã: fixture `tmp_path` của pytest cố ghi vào `C:\Users\khach\AppData\Local\Temp\dsh-VoOS9C\pytest-of-khach`
và bị sandbox từ chối (`PermissionError: [WinError 5]`). Lỗi xảy ra ở bước dựng fixture, **trước khi**
thân test chạy, nên không liên quan tới các file đã sửa. Cách xác nhận: chạy lại bộ này ở môi trường có
`TEMP`/`TMPDIR` ghi được.

---

## 1. BẢNG TỔNG HỢP: FINDING → HÀNH ĐỘNG

| ID | Mô tả (audit) | Mức | Quyết định | File |
| :--- | :--- | :--- | :--- | :--- |
| **A2-2** | `torch.cuda.synchronize()` đồng bộ TOÀN device trong TTS | HIGH | ✅ **SỬA** — CUDA stream riêng cho TTS | `backend/tts/engine.py` |
| **A2-3** | TTS không giải phóng VRAM khi tắt `tts_enabled` | MED | ✅ **SỬA** — unload ở cả REST và WS | `backend/main.py`, `backend/ws/handler.py` |
| **B10-1** | `findVideo()` quét Shadow DOM đệ quy mỗi miss | MED | ✅ **SỬA** — TreeWalker + trần node + throttle | `extension_firefox/content/content-script.js` |
| **B6-3** | `get_context_str()` build chuỗi mỗi câu dịch | MED | ✅ **SỬA** — cache, invalidate khi history đổi | `backend/translation/context.py` |
| **A1-1** | Thread oversubscription trên máy <12 nhân | HIGH | ✅ **SỬA** — thread tự động theo số nhân | `backend/config.py` |
| **B5-1** | ASR executor `max_workers=2` vs `max_inflight_infer=1` | LOW–MED | ✅ **SỬA** — executor suy ra từ config | `backend/asr/engine.py` |
| **B9-4** | `gc.collect()` trong TTS load/unload | LOW | ✅ **SỬA** — chỉ giữ ở đường unload | `backend/tts/engine.py` |
| **B3-1** | Int16→Float32 "conversion kép" | MED | ❌ **LOẠI** — tiền đề sai, phá bảo đảm bit-exact | (đã revert) |
| **B8-1** | "Double dedup pass" | MED | ❌ **LOẠI** — 2 dedup khác khoá, tiết kiệm ~0,5 µs | (đã revert) |
| **B9-2** | Không cache prompt template dịch | LOW | ❌ **LOẠI** — cỡ µs, thêm cache = thêm rủi ro | — |
| **A1-2** | `SpeechNormalizer` cấp phát mảng mỗi lần | MED | ❌ **LOẠI** — churn 1,5 MB/s là nhiễu so với 400 ms suy luận | — |
| **A4-2** | Warm-up dịch "đọc file 2 lần" | LOW | ❌ **LOẠI** — không có lần đọc thứ hai (xem §3.5) | — |
| **B4-2** | `_send_lock` serialize mọi writer | LOW–MED | ⏸ **HOÃN** — chỉ đáng làm khi có nhiều session | — |
| **B5-2** | Translation executor `max_workers=1` | MED | ⏸ **HOÃN** — cần đo VRAM trước, thuộc Phase 3 | — |
| **A2-1** | ASR (Vulkan) + Dịch/TTS (CUDA) tranh chấp GPU | HIGH | 📋 **KẾ HOẠCH** — Phase 3, làm sau cùng | xem §4 |
| **A3-1** | Phình RAM native (transcribe.cpp) | HIGH | 📋 **KẾ HOẠCH** — patch upstream, ngoài repo | xem §5 |
| **A4-1** | Đổi model ASR/TTS đọc lại GGUF trong request | MED | 📋 **KẾ HOẠCH** — hệ quả của singleton, xem §5.2 | — |
| **F-03 / B4-1** | Scale = 1 session | CRIT (thiết kế) | ⛔ **KHÔNG SỬA** — thiết kế có chủ ý ("1 phiên / 1 video") | — |

---

## 2. PHASE 1 — CÁC FIX ĐÃ THỰC HIỆN

### 2.1. A2-2 — Bỏ `torch.cuda.synchronize()` toàn device trong TTS

**Vấn đề (audit §2 / A2-2).** `backend/tts/engine.py` gọi `torch.cuda.synchronize()` sau mỗi lần
sinh audio và trong warmup. Hàm này chờ **mọi** stream CUDA trên device, không chỉ stream của
TTS. Vì dịch (llama.cpp/CUDA) và TTS dùng chung một device, mỗi câu TTS xong đều ép device
"nghỉ", làm trễ inference dịch kế tiếp → **priority inversion** và spike phụ đề.

**Cách sửa.**

1. Thêm `self._cuda_stream: Any = None` trong `__init__` (`backend/tts/engine.py:73`).
2. Tách hàm `_invoke_generate()`: chạy `model.generate` trong `with torch.cuda.stream(self._cuda_stream)`
   khi có stream riêng, ngược lại gọi trực tiếp như cũ.
   Dùng dạng `with` (API ổn định mọi bản PyTorch) **thay vì** dạng decorator `stream(fn)`, để không
   phụ thuộc `StreamContext.__call__`.
3. Tạo stream **một lần** trong `load_model()` (`backend/tts/engine.py:206-213`), ngay trước warmup.
   Trước khi tạo stream có **đúng một** `torch.cuda.synchronize()`: đây là đường nạp model (không
   phải hot path) và cần thiết để mọi công việc ghi trọng số trên stream mặc định hoàn tất trước
   khi TTS đọc trọng số trên stream mới — nếu bỏ, lần sinh đầu tiên có thể đọc trọng số chưa ghi
   xong (race cross-stream).
4. Thay `torch.cuda.synchronize()` bằng `self._cuda_stream.synchronize()` ở warmup và ở
   `_synthesize_audio()` (`backend/tts/engine.py:294-301`). Vẫn giữ fallback `torch.cuda.synchronize()`
   khi `_cuda_stream is None` (CPU hoặc trường hợp lỗi) để không đổi hành vi ở cấu hình đó.
5. `unload_model()` đặt `self._cuda_stream = None` để lần nạp sau tạo stream mới.

**Kiểm chứng.**
- `python -m py_compile backend/tts/engine.py` → OK.
- `pytest backend/tests/test_06_tts_benchmark.py` → PASS (test dùng fake engine, không cần GPU).
- Cần đo trên máy GPU (chưa thực hiện được ở đây): p95/p99 `commit_latency_ms` khi bật TTS, so
  sánh trước/sau; kỳ vọng giảm 100–400 ms ở các spike (đúng ước lượng L1 của audit).

**Rủi ro & giảm thiểu.**
- *Rủi ro:* nếu model OmniVoice tự tạo stream nội bộ, việc bọc thêm stream ngoài là vô hại (chỉ
  đổi stream hiện hành).
- *Rủi ro:* quên fallback ở đường CPU → đã xử lý bằng `if self._cuda_stream is not None`.
- *Rollback:* trả `_run_generate`/`_synthesize_audio` về `torch.cuda.synchronize()`; xoá `_cuda_stream`.

---

### 2.2. A2-3 — Giải phóng VRAM TTS khi tắt `tts_enabled`

**Vấn đề (audit §2 / A2-3).** Tắt TTS chỉ đổi cờ cấu hình; model OmniVoice (vài GB float16) vẫn
nằm trong VRAM. Đây là cơ hội **giảm VRAM không giảm chất lượng**.

**Cách sửa.** Hai đường vào, vì TTS có thể bị tắt từ 2 nơi:

| Đường | Vị trí | Xử lý |
| :--- | :--- | :--- |
| REST `POST /api/config` | `backend/main.py:566-577` | `await asyncio.to_thread(get_tts_engine().unload_model)` |
| WS `set_config` (popup) | `backend/ws/handler.py` `_handle_text_message` | `await _maybe_unload_tts_when_idle(session)` |

Hàm mới `_maybe_unload_tts_when_idle()` (`backend/ws/handler.py:66-90`) chỉ giải phóng khi **mọi**
phiên đang mở đều đã tắt TTS (`get_active_sessions()`), tránh làm phiên khác phải nạp lại model
giữa chừng. Bật lại TTS sẽ `prewarm()` nạp lại như cũ ⇒ không mất chức năng.

**Hai chi tiết quan trọng về mặt hiệu năng:**

1. `unload_model()` được gọi qua `asyncio.to_thread(...)`, **không** gọi trực tiếp trên event loop.
   Lý do: `unload_model()` có `gc.collect()` + `torch.cuda.empty_cache()`, có thể mất hàng trăm ms;
   chặn event loop ở đây sẽ tự tạo ra một latency spike — đúng thứ A2-3 muốn tránh.
2. `unload_model()` được gọi qua `asyncio.to_thread(...)` ở cả hai đường; không có đường nào chặn loop.

**Kiểm chứng.**
- `py_compile` OK; toàn bộ test mặc định PASS (bao gồm `test_11_config_effectiveness.py`,
  `test_16_server_wiring.py` — hai test chạm đường cấu hình TTS).
- Cần đo trên máy GPU: VRAM qua `nvidia-smi` trước/sau khi tắt TTS; kỳ vọng giảm đúng phần model
  (vài GB). Đo thời gian nạp lại khi bật lại để biết chi phí đánh đổi.

**Quyết định thiết kế đã cân nhắc và LOẠI:** **không** unload TTS khi session ngắt kết nối. Extension
ngắt/mở kết nối rất thường xuyên (mỗi lần chuyển video); unload ở đó sẽ khiến mỗi lần quay lại phải
nạp lại model (tốn vài giây) ⇒ hại latency nhiều hơn lợi VRAM. Nếu sau này cần, nên thêm **trễ ân hạn**
(ví dụ unload sau 5 phút không có session) chứ không unload ngay.

---

### 2.3. B10-1 — Chặn trên cho `findVideo()` (extension)

**Vấn đề (audit §10 / B10-1).** `content-script.js` quét `root.querySelectorAll("*")` đệ quy qua mọi
Shadow Root + iframe mỗi lần cache miss. Trên trang lớn (YouTube/Bilibili) đây là O(kích thước DOM)
trên **main thread** ⇒ jank. TTL 2 s chỉ áp dụng cho đường `getVideo()`; các chỗ gọi thẳng
`findVideo()` (popup, status, attach) vẫn quét lại toàn DOM mỗi lần.

**Cách sửa** (`extension_firefox/content/content-script.js`):

1. Thêm 2 hằng số + 1 biến trạng thái:
   - `MAX_SHADOW_SCAN_NODES = 20000` — trần số node duyệt trong **một** lần quét.
   - `VIDEO_SCAN_MIN_INTERVAL_MS = 250` — khoảng nghỉ tối thiểu giữa hai lần quét.
   - `lastVideoScanAt` — mốc thời gian lần quét gần nhất.
2. **Fast path:** luôn chạy `document.querySelectorAll("video")` trước (native, rất nhanh kể cả DOM lớn).
3. **Chỉ** đi vào Shadow DOM/iframe khi light DOM **chưa có video dùng được**
   (`!paused && !ended` hoặc `readyState > 0`) **và** đã qua `VIDEO_SCAN_MIN_INTERVAL_MS`.
   ⇒ Trang thông thường (video ở light DOM) **không bao giờ** duyệt cây.
4. Thay `querySelectorAll("*")` bằng `document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT)`:
   không cấp phát `NodeList` cho cả cây, và trừ dần `budget` để có chặn trên cứng.
5. `budget` dùng chung cho cả nhánh Shadow DOM và iframe ⇒ tổng chi phí một lần quét bị chặn.

**Vì sao không đổi hành vi đúng:** điều kiện `lightUsable` giữ nguyên khả năng tìm video trong Shadow
DOM khi light DOM chỉ có `<video>` "rác" (paused, `readyState = 0`). Logic xếp hạng video phía sau
**không đổi**.

**Kiểm chứng.**
- `node --check extension_firefox/content/content-script.js` → OK.
- Cần kiểm thủ công (chưa chạy được ở đây): mở YouTube/Bilibili, xác nhận (a) tìm đúng video,
  (b) không còn jank khi cache miss, (c) `window.__bsFindVideo()` vẫn trả đúng phần tử.
- Đo bằng Performance panel: thời gian 1 lần quét trước/sau trên trang DOM lớn.

---

### 2.4. B6-3 — Cache chuỗi ngữ cảnh dịch

**Vấn đề (audit §6 / B6-3).** `get_context_str()` ghép lại chuỗi từ deque mỗi lần gọi. Hàm này
được gọi cho **mỗi câu dịch** khi `translation.use_context = True` (`backend/ws/handler.py:587`).

**Cách sửa** (`backend/translation/context.py`): thêm `_context_cache` + `_context_dirty`.
`add()` và `clear()` bật cờ dirty; `get_context_str()` chỉ dựng lại chuỗi khi dirty.

**Lưu ý trung thực về mức lợi:** mặc định `use_context = False`, và `handler.py:587` chỉ gọi hàm khi
cờ bật ⇒ ở cấu hình mặc định đây là **thay đổi trung tính** (không gọi thì không có gì để cache). Đây
là tối ưu cho cấu hình có bật ngữ cảnh, không phải một bottleneck đang hoạt động.

**Kiểm chứng.** `test_05_translation_benchmark.py::test_...ContextManager...` (assert
`"Hello -> Xin chào" in ctx.get_context_str()`) PASS — chứng minh cache không làm mất dữ liệu.
Toàn bộ test mặc định PASS.

---

### 2.5. A1-1 — Chống oversubscription thread CPU trên máy ít nhân

**Vấn đề (audit §2 / A1-1).** Tổng thread compute thường trực ~14 (PyTorch 2 + llama.cpp 4 +
transcribe.cpp 4 + worker GIL). Trên máy 4–8 nhân đây là oversubscription thật, tranh nhân với
decode/render video của trình duyệt.

**Cách sửa** (`backend/config.py:14-25, 130, 244-250`):

```python
_CPU_COUNT = os.cpu_count() or 4
_AUTO_THREADS = 3 if _CPU_COUNT < 12 else 4
```

Áp dụng cho **cả hai** engine:
- `ASRConfig.threads: int = _AUTO_THREADS` (trước: `4`)
- `TranslationConfig.n_threads: int = _AUTO_THREADS` (trước: `4`)

**Vì sao 3 là mức an toàn:** ASR chạy Vulkan và dịch chạy CUDA — cả hai chủ yếu dùng GPU, thread CPU
chỉ để feed/parse. Máy ≥12 nhân (gồm máy tham chiếu RTX 5060 Ti + Ryzen 7/9) giữ nguyên `4` ⇒
**không đổi hành vi trên máy tham chiếu**. Vẫn có thể ghi đè qua `config`/REST như trước.

**Chưa làm (cố ý):** không đụng `torch.set_num_threads(2)` trong `backend/main.py:39-45`. Đó là
thiết lập cấp tiến trình, đã đúng và đã được ghi nhận là "PASSIVE — tốt" trong audit.

**Kiểm chứng.** Toàn bộ test mặc định PASS (bao gồm `test_11_config_effectiveness.py` — test này
kiểm tra config có tác dụng thật). Cần đo trên máy 4–6 nhân: CPU utilization + `commit_latency_ms`
trước/sau.

---

### 2.6. B5-1 — Executor ASR khớp với trần inference đồng thời

**Vấn đề (audit §5 / B5-1).** `_EXECUTOR = ThreadPoolExecutor(max_workers=2)` nhưng
`config.asr.max_inflight_infer = 1`. Semaphore chỉ cho 1 inference chạy ⇒ worker thứ hai **không bao
giờ có việc** (một thread nền thường trực dư thừa).

**Cách sửa** (`backend/asr/engine.py:56-63`):

```python
_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(getattr(config.asr, "max_inflight_infer", 1) or 1)),
    thread_name_prefix="asr_worker",
)
```

Nay executor **suy ra từ cấu hình**: ai nâng `max_inflight_infer` thì executor nới theo. Hành vi
tương đương ở cấu hình mặc định (1 worker thay vì 1 worker hữu dụng + 1 worker rảnh), và an toàn vì
semaphore vẫn là thứ chặn số inference đồng thời.

**Kiểm chứng.** Toàn bộ test mặc định PASS, gồm `test_17_executor_backpressure.py` (test trực tiếp
hành vi executor/backpressure của ASR).

---

### 2.7. B9-4 — Thu hẹp `gc.collect()` trong TTS

**Vấn đề (audit §9 / B9-4).** `gc.collect()` được gọi ở cả warmup (`load_model`) và `unload_model`.

**Cách sửa** (`backend/tts/engine.py`):
- **Bỏ** `gc.collect()` khỏi warmup — đây là đường nạp model, GC toàn phần ở đó là chi phí không cần
  thiết (Python tự GC, và tiến trình vừa mới cấp phát vài GB nên GC quét heap lớn).
- **Giữ** `gc.collect()` ở `unload_model()` — có chủ ý: module PyTorch tạo **vòng tham chiếu**
  (module ↔ parameter ↔ hook); nếu không ép GC thì `torch.cuda.empty_cache()` bên dưới **không đòi
  lại được VRAM**, tức là làm hỏng chính mục tiêu của A2-3. Đây là đường rất hiếm (tắt TTS / đổi
  model), không nằm trên hot path.

**Kiểm chứng.** `test_06_tts_benchmark.py` PASS; toàn bộ test mặc định PASS.

---

## 3. PHASE 2 — CÁC ĐỀ XUẤT **BỊ LOẠI** (kèm bằng chứng)

Phần này quan trọng ngang Phase 1: "sửa" một thứ không phải bottleneck sẽ **thêm** rủi ro mà không
được gì. Mỗi mục dưới đây đã được kiểm chứng bằng mã nguồn gốc (`git diff` so với HEAD) và/hoặc test.

### 3.1. B3-1 — Int16→Float32 "conversion kép" ⇒ **LOẠI, đã revert**

**Tiền đề của audit không đúng.** Audit nói đường VAD/ASR "chuyển Int16 → Float32 **2 lần**". Kiểm
tra mã gốc:

- `feed_audio()` (bản gốc): `np.frombuffer(int16).astype(np.float32); /= 32768.0` → **1 lần** convert.
- `CircularAudioBuffer.write(float32)`: chỉ **copy** float32 vào ring, không convert.
- `get_slice()`: chỉ copy float32 ra.

⇒ Chỉ có **một** conversion trên đường ghi, không phải hai.

**Đổi ring buffer sang Int16 gây 3 hệ quả xấu:**

1. **Phá bảo đảm đã được test chốt.** `backend/tests/test_01_core_audio.py::test_bit_exact_audio_integrity`
   ghi dữ liệu WAV float32 vào buffer rồi đọc ra và assert `max_diff == 0.0`. Lưu Int16 làm sai số
   1 LSB: **đo được `3.0517578125e-05`** — test FAIL (đã tái hiện thực tế khi tôi áp dụng bản Int16).
   Docstring lớp ghi rõ "Đảm bảo độ toàn vẹn 100% mẫu âm thanh (Bit-Exact Integrity)" — đây là
   **hợp đồng đã tuyên bố**, không phải chi tiết nội bộ.
2. **Đẩy việc convert sang đường ĐỌC.** `get_slice()` chạy mỗi preview/commit (cửa sổ 6 s = 96k mẫu,
   ~3 lần/giây) sẽ phải convert Int16→Float32, cộng thêm mảng tạm. Đây là **đổi chỗ** công việc, không
   giảm việc.
3. **Lợi ích RAM không đáng kể:** 3,84 MB → 1,92 MB, tức tiết kiệm **1,92 MB/session**, trong khi
   pipeline đang dùng ~9,5 GB VRAM và vài GB RAM.

**Trạng thái:** đã `git checkout -- backend/core/audio_buffer.py` và khôi phục `feed_audio()` về
nguyên trạng (kèm comment giải thích lý do loại). Test bit-exact PASS trở lại.

> **Nếu vẫn muốn theo B3-1 sau này:** phải làm **opt-in** (`config.audio_buffer.storage_dtype`), mặc
> định `float32`; đồng thời phải **thay đổi tường minh** hợp đồng bit-exact và cập nhật test_01 với
> sự đồng ý của người dùng. Không nên làm ngầm.

### 3.2. B8-1 — "Double dedup pass" ⇒ **LOẠI, đã revert**

**Hai lớp dedup KHÔNG cùng khoá:**

| Lớp | File | Khoá | TTL/history |
| :--- | :--- | :--- | :--- |
| `TranslationDeduplicator` | `translation/dedup.py` | **văn bản GỐC** (sau ASR) | TTL 10 s |
| `TTSDedupState` | `tts/dedup.py` | **văn bản ĐÃ DỊCH** | 15 mục gần nhất |

Truyền `already_deduped=True` từ dịch sang TTS sẽ **tắt hẳn** lớp dedup theo văn bản đã dịch — lớp
này bắt được trường hợp **hai câu gốc khác nhau cho ra cùng một câu dịch**, thứ mà dedup theo văn bản
gốc **không** bắt được. Hệ quả: có thể phát lại audio trùng.

**Chi phí thật của "pass" bị bỏ:** `report/04_commit_logic/report.md` (sinh bởi chính bộ test của
repo) đo **Deduplication: 490.172 thao tác/giây ≈ 0,5 µs/câu**. Đổi một bảo hiểm chống trùng lặp lấy
0,5 µs trên một câu mất 400 ms là một giao dịch tồi.

**Trạng thái:** đã revert cả 3 chỗ (`_process_translation_item` không còn thêm cờ, `_process_tts_item`
không còn đọc cờ và không còn `not already_deduped`).

### 3.3. B9-2 — Cache prompt template dịch ⇒ **LOẠI**

`build_prompt()` (`translation/prompts.py`) chỉ làm vài f-string + một tra dict
(`resolve_lang_name`). Chi phí cỡ **vài µs** cho một câu, trong khi chính câu đó tốn 250–800 ms
suy luận LLM. Thêm cache ở đây phải giải quyết invalidate khi `target_lang`/`prompt_style`/`use_context`
đổi — tức thêm state và thêm khả năng sai — để tiết kiệm < 0,001 % thời gian. **Không đáng.**

### 3.4. A1-2 — `SpeechNormalizer` cấp phát mảng mỗi lần ⇒ **LOẠI**

Audit tự ước lượng "churn" ~1,5 MB/s. Với câu 6 s @16 kHz, `np.empty_like` = 384 KB; numpy dùng lại
vùng nhớ đã cấp từ allocator nên chi phí thực là vài µs. Đổi sang buffer tái sử dụng đòi hỏi quản lý
vòng đời buffer **dùng chung giữa các lần inference** — tức phải chứng minh không có reader nào còn
giữ tham chiếu (preview/commit chạy trên executor khác nhau). Rủi ro race cao hơn nhiều so với vài µs
tiết kiệm được. **Không làm.**

### 3.5. A4-2 — "Warm-up dịch đọc file 2 lần" ⇒ **LOẠI**

Đọc `translation/engine.py`: `load_model()` chỉ gọi `_build_llm()` (**một** lần mmap GGUF), **không**
có dummy run bên trong. `prewarm()` gọi `load_model()` rồi `_translate_sync("Hello")` — lần thứ hai
là **dummy run để warm CUDA kernel**, không phải đọc lại file. Gộp lại sẽ **mất** warmup ⇒ câu dịch
đầu tiên chậm hơn. **Không sửa.**

### 3.6. B4-2 — Tách `_send_lock` theo loại message ⇒ **HOÃN**

Trên localhost, 3 writer (preview ASR / dịch / TTS) serialize trên một `asyncio.Lock` có chi phí
không đáng kể, và audit cũng ghi "lợi ích hạn chế". Tách lock theo loại message làm tăng khả năng
**xen kẽ frame** trên cùng một socket — rủi ro tính đúng đắn của thứ tự frame cao hơn lợi ích. Chỉ
đáng xem lại nếu chuyển sang nhiều session (Phase 3+).

### 3.7. B5-2 — Nâng translation executor lên 2 worker ⇒ **HOÃN sang Phase 3**

Đây là thay đổi **tăng throughput có đánh đổi VRAM** (2 phiên llama.cpp song song trên cùng một
card vốn đã tranh chấp với ASR/TTS). Làm nó **trước** khi có scheduler (A2-1) sẽ làm spike tệ hơn.
Điều kiện để làm: sau khi A2-1 xong và đo được VRAM còn dư. Xem §4.5.

---

## 4. PHASE 3 — KẾ HOẠCH A2-1 (LÀM SAU CÙNG)

### 4.0. Vì sao A2-1 phải làm sau cùng

A2-1 là **thay đổi kiến trúc điều phối GPU**, không phải một fix cục bộ. Nó:

- Chạm vào cả 3 engine (ASR, dịch, TTS) và vòng đời worker của session.
- **Phụ thuộc một gate bên ngoài repo:** liệu có build được ASR chạy **CUDA** hay không. Nếu build
  được, tranh chấp Vulkan-vs-CUDA **biến mất về bản chất** và thiết kế scheduler đơn giản hơn nhiều
  (một backend, một hàng đợi ưu tiên). Nếu không build được, phải thiết kế cho **hai backend GPU
  độc lập** chia sẻ một card — bài toán khó hơn và cần đo đạc nhiều hơn.
- Cần số liệu **trước/sau** trên máy GPU thật (p95/p99 commit latency, số spike > 1,5 s, GPU util,
  VRAM peak). Không thể nghiệm thu bằng test đơn vị.

⇒ Làm A2-1 trước các fix khác sẽ khiến ta tối ưu dựa trên giả định chưa kiểm chứng.

### 4.1. Gate quyết định (làm TRƯỚC mọi dòng code A2-1)

| Bước | Việc | Tiêu chí PASS | Nếu FAIL |
| :--- | :--- | :--- | :--- |
| G1 | Kiểm tra wheel `transcribe-cpp-native-cu12` trên PyPI có bản thật (không phải `0.0.0` rỗng) | Có wheel chứa `ggml-cuda.dll` | → G2 |
| G2 | Tự build `transcribe.cpp` với `GGML_CUDA=ON` (CMake + CUDA Toolkit khớp driver) | `transcribe_cpp.backend_available("cuda") == True` và một inference chạy được trên CUDA | → **Nhánh B** (§4.4) |
| G3 | Chạy 1 phiên đầy đủ với ASR CUDA + dịch/TTS CUDA | WER không tệ hơn bản Vulkan (cùng model, cùng audio) | → quay lại Vulkan |

#### 4.1.1. KẾT QUẢ CHẠY GATE — 2026-09-16 (dừng tại đây, chờ quyết định)

**G1 → ❌ FAIL (kết luận dứt điểm, có bằng chứng).**

| Bằng chứng | Nội dung |
| :--- | :--- |
| PyPI JSON `transcribe-cpp-native-cu12` | **Chỉ có duy nhất release `0.0.0`**; wheel `transcribe_cpp_native_cu12-0.0.0-py3-none-any.whl` **1380 byte** (`py3-none-any` ⇒ không chứa native code). Summary ghi thẳng: *"name reservation; real wheels arrive with 0.1.0"*. Không có version nào khác trong `releases`. |
| `contract.json` của provider đã cài | `transcribe_cpp_native/_native/contract.json` → `"backends": ["vulkan","cpu"]`, `"lane": "cpu-vulkan"`. Thư mục `_native` có `ggml-vulkan.dll` + 9 `ggml-cpu-*.dll`, **không có `ggml-cuda.dll`**. |
| Repo `external/transcribe.cpp` (bản 0.2.3 + 1 commit local) | Có sẵn `bindings/python-native-cu12/` và `.github/workflows/cuda-windows.yml`, nhưng workflow chỉ chạy **khi release tag / dispatch thủ công** (không chạy mỗi PR) vì "nvcc over ggml-cuda × 5 arches is the heaviest build in the project". |
| Ghi chú tự thú của chính upstream | Workflow ghi rõ: *"There is no Windows NVIDIA hardware in the fleet, so the RUNTIME evidence is carried by the Linux T4 smoke … **documented gap**"* ⇒ bản CUDA trên Windows **chưa từng được upstream chạy thử trên phần cứng NVIDIA thật**. |

⇒ **Đường "cài wheel cu12 có sẵn" đã đóng.** Muốn ASR chạy CUDA thì **buộc phải tự build** (G2).

**G2 → ⏸ LÚC ĐO CHƯA CHẠY ĐƯỢC: thiếu toolchain — ⚠️ ĐÃ CHẠY SAU ĐÓ, xem §4.1.2 (kết quả: PASS).**

*Phần đã sẵn sàng (kiểm tra thực tế trên máy này):*

| Hạng mục | Trạng thái |
| :--- | :--- |
| GPU | ✅ RTX 5060 Ti, **driver 596.36** (hỗ trợ tới CUDA 13.0), 16311 MiB |
| CUDA runtime cho torch | ✅ `torch 2.12.0+cu130`, `torch.cuda.is_available() == True` ⇒ đường CUDA của dịch/TTS đang chạy tốt |
| Trình biên dịch | ✅ Visual Studio **18** Community (khớp `build_x64/CMakeCache.txt`: generator `Visual Studio 18 2026`, platform x64) |
| CMake | ✅ 4.4.3 |
| Mã nguồn | ✅ `external/transcribe.cpp` đầy đủ, có `ggml/src/ggml-cuda/` và `bindings/python-native-cu12/` |
| ggml-cuda hỗ trợ CUDA 13 | ✅ `CMakeLists.txt` có nhánh `if (CUDAToolkit_VERSION VERSION_LESS "13")` ⇒ CUDA 12.9 **và** 13.x đều được |
| Ổ đĩa | ✅ C: 146,3 GB trống · D: 271 GB trống |
| CPU | ✅ 12 logical (⇒ `_AUTO_THREADS = 4`, đúng như thiết kế A1-1) |

*Phần còn thiếu — chính là chốt chặn:*

| Hạng mục | Trạng thái | Ghi chú |
| :--- | :--- | :--- |
| **`nvcc` (CUDA Toolkit)** | ❌ **KHÔNG CÓ** | Không có `CUDA_PATH`, không có khoá registry `NVIDIA Corporation\GPU Computing Toolkit\CUDA`, không có `C:\Program Files\NVIDIA GPU Computing Toolkit`. Đây là **nguyên nhân duy nhất** G2 chưa chạy được. |
| `ninja` | ❌ không có | (VS generator vẫn dùng được, chỉ chậm hơn) |
| `uv` | ❌ không có | Repo `external/transcribe.cpp/AGENTS.md` bắt buộc `uv run` cho mọi lệnh Python |
| Quyền admin | ❌ **không phải admin** | `IsInRole(Administrator) == False` ⇒ `winget install Nvidia.CUDA` (cài machine-wide) sẽ cần UAC |

*Đường đi **không cần admin** đã xác minh được:* PyPI có wheel **`nvidia-cuda-nvcc-cu12==12.9.86`** cho
`win_amd64` (**34,7 MB**, upload 2025-06-05) ⇒ có thể lấy `nvcc` qua `pip install` vào
site-packages của user, không cần elevation. Kèm `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`
(đúng như `bindings/python-native-cu12/pyproject.toml` khai báo) và `ninja` (cũng có wheel pip).

*Hai rủi ro kỹ thuật đã nhận diện trước, phải xử lý trong G2:*

1. **Thiếu `sm_120` cho RTX 5060 Ti (Blackwell).** `bindings/python-native-cu12/pyproject.toml` pin
   `-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;90` — **không có `120`**. Với danh sách này, kiến trúc
   `sm_120` sẽ phải JIT từ PTX của `sm_90` (chạy được nhờ forward-compat của PTX, nhưng tốn thời gian
   JIT lúc khởi động và **không phải đường đã được test**). Cần thêm `120` vào danh sách.
2. **Chi phí build.** Upstream mô tả đây là build nặng nhất dự án: **245 file `.cu` × 5 arches**, trên
   runner 16 vcpu mất tới ~180 phút (timeout của họ). Trên máy này (12 logical) nếu **chỉ build 1 arch
   (`120`)** thay vì 5 thì nhanh hơn nhiều lần, nhưng vẫn là một build dài (ước lượng hàng chục phút)
   và chưa từng được chạy trên cấu hình này.

**G3 → ⚠️ ĐÃ CHẠY SAU KHI G2 PASS — xem §4.1.3 (kết quả: PASS, CUDA nhanh hơn 1,53×).**
*Đánh giá ban đầu (khi G2 còn treo):* không có bước nào của G3 chạy được khi chưa có `ggml-cuda` module.

**Kết luận gate (chưa chọn nhánh):**

- G1 = FAIL ⇒ nhánh "dùng wheel có sẵn" không tồn tại.
- G2 = **khả thi nhưng cần cài toolchain** (đường pip, không cần admin) **+ sửa arch list** để thêm
  `sm_120`. Đây là hành động **có thay đổi hệ thống** (cài ~100+ MB package) và **tốn thời gian build
  dài**, nên dừng lại chờ quyết định thay vì tự ý chạy.
  → **Cập nhật §4.1.2:** người dùng đã chọn chạy G2 ⇒ **PASS**. Toolchain dựng được **hoàn toàn trong
  workspace, không cần admin**; build lại cần thêm `-DGGML_CUDA_GRAPHS=ON` (upstream để OFF).
- Nếu G2 FAIL sau khi đã thử ⇒ đi **Nhánh B** (§4.4): giữ ASR Vulkan, chuyển mục tiêu sang "giảm số lần
  hai backend chồng lấn", bắt đầu bằng đo đạc.
  → **Cập nhật §4.1.3:** G3 cũng **PASS** ⇒ **Nhánh A đã đủ điều kiện**, Nhánh B chỉ còn là dự phòng.

**Điểm quan trọng cho việc ra quyết định:** ngay cả khi G2+G3 PASS, lợi ích của A2-1 phần lớn **đã
được lấy trước** bằng A2-2 (bỏ `torch.cuda.synchronize()` toàn device) và A2-3 (trả VRAM) ở Phase 1.
Vì vậy nên **đo lại M1/M2/M3 trước** (§6.2) để biết còn bao nhiêu spike thật sự cần A2-1 xử lý — nếu
đã về mức chấp nhận được thì Nhánh B (hoặc không làm gì thêm) là kết luận đúng, không phải thất bại.

**Ghi chú từ mã hiện tại:** `backend/main.py:68-89` (`_warn_if_cuda_provider_missing`) đã ghi nhận
bằng chứng rằng bản CUDA **chưa phát hành** và wheel `-cu12` chỉ có version `0.0.0` rỗng. Vì vậy G1
gần như chắc chắn FAIL, và G2 (tự build) là đường duy nhất. Cần đánh giá chi phí build + phân phối
(người dùng cuối phải có wheel CUDA khớp driver ⇒ ma trận hỗ trợ phình ra).

#### 4.1.2. KẾT QUẢ G2 — ✅ **PASS** (build CUDA thành công, không cần admin)

Đã tự build `transcribe.cpp` với backend CUDA. **Toàn bộ toolchain nằm trong workspace**, không cài
gì lên hệ thống, không cần quyền admin.

**Phát hiện quan trọng khi dựng toolchain (2 cái bẫy, đã xử lý):**

| Bẫy | Chi tiết | Cách xử lý |
| :--- | :--- | :--- |
| Wheel `nvidia-cuda-nvcc-cu12` (CUDA 12) **KHÔNG có `nvcc.exe`** | Giải nén wheel 12.9.86: chỉ có `ptxas.exe` + `nvvm64_40_0.dll` + header CRT (35 entry). Đây là bộ phận **JIT**, không phải trình biên dịch đầy đủ. | Dùng package CUDA 13 **`nvidia-cuda-nvcc`** (13.4.59) — có `nvcc.exe`, `cudafe++.exe`, `fatbinary.exe`, `nvlink.exe`, `cicc.exe` (cicc nằm trong `nvidia-nvvm`) |
| Wheel pip chỉ có DLL, **không có `.lib`** | `nvidia-cublas` ship `cublas64_13.dll` + `cublasLt64_13.dll` nhưng **không** có `cublas.lib` ⇒ MSVC không link được | Tự sinh import library: `dumpbin /exports` → `.def` → `lib /def /machine:x64`. Đã sinh `cublas.lib` (756 export), `cublasLt.lib` (1019), `nvblas.lib` (60) |

**Cấu hình build** (khác upstream `bindings/python-native-cu12/pyproject.toml` ở 2 điểm):

```
cmake -S external/transcribe.cpp -B .build-cuda -G Ninja
  -DCMAKE_CUDA_COMPILER=.cuda-toolkit/nvidia/cu13/bin/nvcc.exe
  -DCUDAToolkit_ROOT=.cuda-toolkit/nvidia/cu13
  -DCMAKE_CUDA_ARCHITECTURES=120          # ← upstream pin 75;80;86;89;90, THIẾU sm_120
  -DGGML_CUDA_GRAPHS=ON                   # ← upstream để OFF; đây là chìa khoá hiệu năng (xem 4.1.3)
  -DGGML_CUDA_FA_ALL_QUANTS=ON
  -DTRANSCRIBE_CUDA=ON -DTRANSCRIBE_BUILD_SHARED=ON -DTRANSCRIBE_GGML_BACKEND_DL=ON
  -DTRANSCRIBE_VULKAN=OFF -DTRANSCRIBE_X86_CONSERVATIVE=ON
  -DTRANSCRIBE_BUILD_TESTS=OFF -DTRANSCRIBE_BUILD_EXAMPLES=OFF -DTRANSCRIBE_BUILD_TOOLS=OFF
```

**Bằng chứng G2 PASS** (nạp qua `TRANSCRIBE_LIBRARY=.build-cuda/bin/transcribe.dll`):

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16310 MiB):
  Device 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, VMM: yes, VRAM: 16310 MiB
load_backend: loaded CUDA backend from .build-cuda\bin\ggml-cuda.dll
qwen3_asr: using cuda backend: CUDA0

backend_available: cuda=True vulkan=False cpu=True
backends(): [CUDA0 (NVIDIA GeForce RTX 5060 Ti, 16310 MiB), CPU (AMD Ryzen 5 5600X)]
```

Artifact: `ggml-cuda.dll` **50,8 MB**, `transcribe.dll` 1,72 MB, `ggml-cpu.dll` 0,76 MB, `ggml-base.dll` 0,63 MB.
CMake tự nâng `120` → `120a` (kiến trúc-specific cho Blackwell) — đúng như mong đợi.

#### 4.1.3. KẾT QUẢ G3 — ✅ **PASS** (CUDA nhanh hơn VÀ chính xác tương đương)

**Phép đo:** cùng model (`Qwen3-ASR-0.6B-Q8_0.gguf`), cùng 6 file audio, cùng `n_threads=4`, mỗi file
**5 lần lặp lấy median** (bỏ lần đầu vì phải capture CUDA graph). Chấm điểm bằng chính bộ scorer của
dự án (`backend/tests/test_09_wer_ab.py`: CJK → CER, Latin → WER).

| Audio | CUDA (median) | Vulkan (median) | CUDA nhanh hơn | CUDA err | VK err | Chữ giống nhau |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| Japanese_5s | 100 ms | 156 ms | 1,56× | 0,0800 | 0,0800 | ✅ |
| Russian_4s | 113 ms | 200 ms | 1,77× | 0,4000 | 0,3000 | ❌ |
| Chinese_fast_speed_11s | 299 ms | 460 ms | 1,54× | 0,1385 | 0,1385 | ❌ |
| Cross_lingual_6s | 106 ms | 144 ms | 1,36× | 0,4286 | 0,4286 | ✅ |
| English_low_speech_quality_19s | 189 ms | 270 ms | 1,43× | 0,1290 | 0,1290 | ✅ |
| Chinese_noise_28s | 457 ms | 707 ms | 1,55× | 0,0055 | 0,0055 | ✅ |
| **Tổng** | **1264 ms** | **1937 ms** | **1,53× (−34,7 %)** | 0,1969 | 0,1803 | **4/6** |

**Chìa khoá hiệu năng: `GGML_CUDA_GRAPHS=ON`.** Bản build đầu tiên (theo đúng mặc định upstream,
`GRAPHS=OFF`) **chậm hơn Vulkan 32 %**. Bật CUDA graphs giảm thời gian ASR **~49 %** và đảo ngược kết
luận:

| Audio | CUDA `GRAPHS=OFF` | CUDA `GRAPHS=ON` | Cải thiện |
| :--- | ---: | ---: | ---: |
| Japanese_5s | 187 ms | 96 ms | −49 % |
| Russian_4s | 205 ms | 123 ms | −40 % |
| Chinese_fast_speed_11s | 627 ms | 298 ms | −52 % |
| Cross_lingual_6s | 190 ms | 115 ms | −39 % |
| English_low_speech_quality_19s | 347 ms | 189 ms | −46 % |
| Chinese_noise_28s | 940 ms | 457 ms | −51 % |
| **Tổng** | **2495 ms** | **1278 ms** | **−49 %** |

Đây là kết quả đáng chú ý vì upstream **không** bật cờ này ở bất kỳ lane nào (cả `pyproject.toml`
gốc lẫn `bindings/python-native-cu12/pyproject.toml` và `build_x64/CMakeCache.txt` đều để
`GGML_CUDA_GRAPHS:BOOL=OFF`). Với model nhỏ sinh nhiều kernel ngắn, overhead launch chiếm ưu thế —
đúng trường hợp CUDA graphs sinh ra để giải quyết.

**Độ chính xác — đọc kết quả cho đúng:**

- **4/6 file cho ra chữ giống hệt nhau từng byte** giữa CUDA và Vulkan.
- 2 file khác nhau, và **cả hai đều chỉ khác đúng 1 token**:
  - `Russian_4s`: CUDA "побег **от** своего валиера" vs Vulkan "побег **из** своего валиера" (giới từ).
    Đây là nguồn duy nhất của chênh lệch mean error (+0,0167): file chỉ có 10 từ tham chiếu ⇒ 1 từ = 0,10 WER.
    **Lưu ý cả hai bản đều khác reference ở các từ khác** ("Борсук" vs "Барсук", "запарке" vs "зоопарке",
    "валиера" vs "вольера") ⇒ bản thân reference không khớp hoàn hảo với audio.
  - `Chinese_fast_speed_11s`: CUDA "轮**叉**" vs Vulkan "轮**圈**"/"轴**叉**" — 1 ký tự, hai chỗ.
- **Kết luận trung thực:** trên 6 file, CUDA **không** chứng minh được là chính xác hơn hay kém hơn
  Vulkan. Chênh lệch nằm trong nhiễu của phép lấy mẫu nhỏ. Muốn kết luận chắc chắn phải chạy harness
  WER đầy đủ của dự án (`test_09_wer_ab.py`) trên toàn bộ `wav_test/` — việc này **chưa làm**.

  > ⚠️ **ĐÍNH CHÍNH:** việc đó **đã được làm sau** — xem **§4.1.4**. Kết quả đầy đủ xác nhận hai
  > backend tương đương, **và** chỉ ra rằng các khác biệt token-level nêu ngay trên đây phần lớn là
  > **nhiễu do bộ resampler** mà tôi dùng, không phải khác biệt backend. Đọc §4.1.4 để có kết luận
  > độ chính xác chính thức.

**VRAM (đo bằng `nvidia-smi` device-level, model 0.6B thường trú):**

| Backend | Baseline | Peak | **ASR chiếm** |
| :--- | ---: | ---: | ---: |
| Vulkan (wheel cài sẵn) | 1591 MB | 3388 MB | **+1797 MB** |
| CUDA (bản tự build) | 1571 MB | 3611 MB | **+2040 MB** |

⇒ CUDA tốn thêm **~243 MB VRAM** (+13,5 %). Trên card 16 GB vốn đạt đỉnh ~9,5 GB, mức này **chấp nhận
được** và không đe doạ ngân sách VRAM.

**Độ ổn định (phát hiện phụ nhưng quan trọng):** CUDA cho thời gian rất ổn định
(spread min↔max **1–3 %**), trong khi Vulkan có lần chênh tới **~2×** trên cùng một file
(`Chinese_fast_speed_11s`: 447 ms vs 874 ms giữa hai lần chạy). Một phần "spike" mà audit A2-1 mô tả
có thể đến từ chính độ biến động này của backend Vulkan.

**Trạng thái gate sau G1/G2/G3:** G1 ❌ FAIL → G2 ✅ PASS → G3 ✅ PASS ⇒ **điều kiện để chọn Nhánh A
(§4.2) đã được thoả**. Nhánh B (§4.4) chỉ còn là phương án dự phòng.

#### 4.1.4. G3 (đầy đủ) — Harness WER của dự án, toàn bộ 8 file, `repeats=3`

**Phép đo:** dùng chính `backend/tests/test_09_wer_ab.py` — harness chạy **pipeline VAD + ASR streaming
thật** có pacing (`--speed 6 --max-sec 0 --configs base --repeats 3`), chạy 2 lần: một lần với wheel cài
sẵn (Vulkan), một lần với bản tự build qua `TRANSCRIBE_LIBRARY` (CUDA). `MEM_GUARD_MB=0` để tắt trần RAM.

| Chỉ số | Vulkan | CUDA | Chênh | Sàn nhiễu (max−min) |
| :--- | ---: | ---: | ---: | :--- |
| REF error vs GT (offline, không phân đoạn) | 54,66 % | 55,39 % | +0,74 % | — |
| **STREAM error vs GT** | **17,69 %** | **16,87 %** | **−0,82 %** | 2,28 / 4,12 điểm % |
| STREAM vs REF (mất mát do phân đoạn) | 128,68 % | 127,47 % | −1,21 % | 2,28 / 4,12 điểm % |
| Tổng commits | 104 | 110 | +6 | — |
| Câu rỗng (empty hypothesis) | 0 | 0 | 0 | — |

Chi tiết từng lần lặp (`STREAM error vs GT`):

| Backend | rep 1 | rep 2 | rep 3 | khoảng (max−min) |
| :--- | ---: | ---: | ---: | ---: |
| Vulkan | 17,77 % | 17,89 % | 17,40 % | 0,49 điểm % |
| CUDA | 16,73 % | 16,30 % | 17,59 % | 1,29 điểm % |

**KẾT LUẬN ĐỘ CHÍNH XÁC: hai backend KHÔNG phân biệt được về mặt thống kê.**

`noise_band` của harness là **khoảng max−min** giữa các lần lặp của **cùng một cấu hình**, và chính
harness ghi rõ quy tắc quyết định: *"Chênh lệch A/B NHỎ HƠN sàn nhiễu này KHÔNG kết luận được."*

- Chênh lệch quan sát: **0,82 điểm %** (STREAM vs GT) và **1,21 điểm %** (STREAM vs REF).
- Sàn nhiễu: **2,28 điểm %** (Vulkan) và **4,12 điểm %** (CUDA).
- ⇒ Chênh lệch **nhỏ hơn sàn nhiễu** ở cả hai thước đo ⇒ **không được phép kết luận** CUDA tốt hơn
  hay kém hơn.
- Cách diễn giải trực quan nhất: **khoảng cách CUDA↔Vulkan (0,82 pp) còn NHỎ HƠN chính độ dao động
  của CUDA giữa các lần chạy (1,29 pp).**

**REF text giống hệt nhau trên 5/8 file.** Ba file khác nhau chỉ ở mức rất nhỏ:

| File | Vulkan err | CUDA err | Khác biệt |
| :--- | ---: | ---: | :--- |
| `00_ingress_stream.wav` | 70,52 % | 72,11 % | Vài từ tiếng Nhật (迫って/相次い, 始めて/初めて, …) |
| `Chinese_fast_speed_11s.wav` | 13,85 % | 14,62 % | 1 từ: "倒到" vs "倒出" |
| `English_multiple_kinds_of_noise_88s.wav` | 66,51 % | 66,03 % | 1 dấu câu (". It's" vs ", it's") |
| 5 file còn lại | = | = | **giống hệt từng byte** |

**Đính chính quan trọng về phép đo cũ ở §4.1.3.** Bảng A/B tách rời ở §4.1.3 dùng **bộ resampler của
tôi** (`scipy.signal.resample_poly` trên dữ liệu int16), trong khi harness của dự án dùng
`load_wav_16k()` → `scipy.signal.resample` (FFT) trên float32 (xem
`backend/tests/test_08_streaming_latency.py`). Hai thuật toán cho ra audio hơi khác nhau, và ở các
ranh giới quyết định của model, khác biệt đó **đủ để đổi token** ("轮叉" vs "轮圈").

⇒ Bằng chứng token-level ở §4.1.3 ("CUDA `от` vs Vulkan `из`", "轮叉 vs 轮圈") phần lớn là **nhiễu do
resampler**, không phải khác biệt backend. Số liệu ở §4.1.4 (dùng loader của chính dự án) **thay thế**
phần độ chính xác của §4.1.3.

⇒ **Số liệu TỐC ĐỘ ở §4.1.3 vẫn hợp lệ** vì hai bên dùng cùng một đoạn audio (cùng resampler của tôi)
và tốc độ không phụ thuộc vào việc resample — nên kết luận "CUDA nhanh hơn 1,53×" giữ nguyên.

**Cảnh báo khi đọc chỉ số "REF error vs GT" (54,66 %):** con số này rất cao vì REF chạy **offline toàn
file một lần** và bị **cắt theo trần context của model** (chính docstring harness ghi rõ điều này). Trên
các file dài (120 s, 88 s), REF chỉ phiên âm được phần đầu ⇒ error so với ground truth đầy đủ là vô
nghĩa như một thước đo chất lượng backend. **Chỉ số đáng tin là `STREAM error vs GT`** — và ở đó hai
backend bằng nhau.

**Tổng kết G3 (đầy đủ):**

| Tiêu chí | Kết quả |
| :--- | :--- |
| Chạy được trên CUDA | ✅ (compute capability 12.0, `ggml-cuda.dll`) |
| Chính xác so với Vulkan | ✅ **Tương đương** (chênh 0,82 điểm % < sàn nhiễu 2,28–4,12 điểm %) |
| Tốc độ so với Vulkan | ✅ **Nhanh hơn 1,53×** (đo tách rời, §4.1.3) |
| VRAM | ⚠️ +243 MB (+13,5 %) — chấp nhận được |
| Ổn định | ✅ CUDA spread 1–3 % vs Vulkan tới ~2× |

⇒ **G3 PASS.** Không có rào cản về độ chính xác để chuyển ASR sang CUDA.



### 4.2. Nhánh A — ASR chạy được CUDA (thiết kế ưu tiên)

Khi cả 3 stage cùng một backend, thay tranh chấp **hai backend** bằng tranh chấp **một backend**, và
điều đó **điều phối được** bằng hàng đợi ưu tiên.

**Thiết kế đề xuất:** module mới `backend/core/gpu_scheduler.py` cung cấp một `GpuArbiter` singleton.

```
Lớp ưu tiên (deadline ngắn nhất trước):
  P0  ASR commit      — deadline ~300 ms, KHÔNG được bỏ (mất chữ)
  P1  ASR preview     — best-effort, được phép bỏ
  P2  Translation     — 250–800 ms/câu, trễ được trong giới hạn queue
  P3  TTS             — ~420 ms/câu, trễ được
```

Cơ chế:

1. **Priority semaphore** (1 slot GPU compute): worker xin slot theo lớp ưu tiên; P0/P1 được
   **vượt hàng** P2/P3.
2. **Cửa sổ dành riêng (reservation window) cho ASR commit:** khi có commit đang chờ/chạy, arbiter
   bật cờ `commit_reserved`; các worker P2/P3 **không được nhận job mới**. Điểm cắt tự nhiên đã tồn
   tại: dịch và TTS đều xử lý **từng câu một** (`_process_translation_item`, `_process_tts_item`),
   nên chỉ cần chặn ở **đầu mỗi câu**, không cần preempt giữa chừng (không thể preempt an toàn giữa
   một lượt sinh token).
3. **Chống starvation:** `commit_reserved` có trần thời gian (ví dụ 1,5 s). Hết trần, P2/P3 được chạy
   lại để queue dịch không đói vô hạn. Ghi metric mỗi lần phải nới trần.
4. **CUDA stream priority** (bổ trợ, không thay thế): tạo stream dịch/TTS với `priority` thấp hơn
   stream ASR (`torch.cuda.Stream(priority=-1)` cho ASR). Chỉ là gợi ý cho driver, **không** đủ để
   đảm bảo deadline ⇒ vẫn cần (1)–(3).
5. **Feature flag:** `config.gpu.scheduler_enabled` (mặc định `False`). Cho phép A/B đo và tắt ngay
   nếu hồi quy.

**Tận dụng những gì đã có (không viết lại):**

| Đã có | Vai trò trong A2-1 |
| :--- | :--- |
| `config.asr.max_inflight_infer = 1` + semaphore | Chính là "slot GPU" của ASR — arbiter chỉ cần **phối hợp** với nó |
| `preview_adaptive_backoff` (`asr/engine.py`) | Đã giãn nhịp preview khi chậm — tương thích, giữ nguyên |
| `_coalesce_enqueue` + `maxsize=32` (handler) | Đã bảo đảm không mất câu khi trễ — arbiter chỉ đổi **thứ tự**, không đổi **tính đủ** |
| `_VAD_EXECUTOR` tách riêng | VAD không nằm trên GPU ⇒ không cần đưa vào arbiter |
| `_EXECUTOR` ASR | Đã đúng kích thước sau B5-1 |

**Ràng buộc kỹ thuật phải tôn trọng:** transcribe.cpp cho **1 stream in-flight / model**
(`external/transcribe.cpp/include/transcribe.h`). Arbiter **không** được cố chạy 2 inference ASR
song song — nó chỉ **xếp thứ tự** giữa ASR và dịch/TTS. Muốn ASR song song thật phải có N model
instance (VRAM ×N) — nằm ngoài phạm vi.

### 4.2b. Nhánh A — HẠ TẦNG ĐÃ TRIỂN KHAI (bước A.0)

Phần "nối ASR CUDA vào ứng dụng" đã xong. Đây là **điều kiện tiên quyết** để làm `GpuArbiter`
(§4.3 A.1–A.6), vì arbiter chỉ có nghĩa khi cả 3 stage thực sự chạy trên cùng một backend.

**Bố cục mới**

```
<project>/
├── bin/                       ← bundle native LÚC CHẠY (ứng dụng luôn nạp từ đây)
│   ├── transcribe.dll         libtranscribe (bản dựng cục bộ của ta)
│   ├── ggml.dll               ggml dispatcher
│   ├── ggml-base.dll          ggml core
│   ├── ggml-cpu.dll           backend CPU   (fallback cuối)
│   ├── ggml-cuda.dll          backend CUDA  (67,7 MB)
│   └── ggml-vulkan.dll        backend VULKAN (50,2 MB — nguồn của fallback)
└── external/
    ├── transcribe.cpp/        mã nguồn (không đổi)
    ├── cuda-toolkit/          CUDA 13.4.59 dựng từ wheel pip (nvcc, cublas, headers)
    ├── build-cuda/            cây build CMake
    └── build-tmp/             script build + log + JSON kết quả đo G2/G3
```

Toàn bộ `bin/`, `external/*` đã được thêm vào `.gitignore` (`*.dll` cũng đã bị chặn sẵn).

**Nguồn của `ggml-vulkan.dll` — và vì sao hợp lệ.** Build của ta đặt `TRANSCRIBE_VULKAN=OFF` vì
máy này **không có Vulkan SDK** (`glslc`), mà ggml-vulkan cần `glslc` để biên dịch shader lúc build
(`find_package(Vulkan COMPONENTS glslc REQUIRED)`). Để có fallback thật, `ggml-vulkan.dll` được lấy
từ wheel `transcribe-cpp-native` đã cài. **An toàn vì:** `ggml` là submodule và commit `e2f82cb`
("patch offline voxtral") **không chạm vào `ggml/`** (kiểm chứng bằng `git show --stat e2f82cb`:
chỉ sửa docs/scripts/src của voxtral) ⇒ ggml trong cây nguồn **giống hệt** ggml của wheel ⇒ cùng
ABI. Đã kiểm chứng thực tế: `bin/ggml-vulkan.dll` nạp được trên `ggml-base.dll` của ta và
`Model(backend="vulkan")` chạy ra `Vulkan0`.

**Ba việc đã làm**

1. **Dọn file build vào `external/`** — `.cuda-toolkit` → `external/cuda-toolkit`,
   `.build-cuda` → `external/build-cuda`, `.build-tmp` → `external/build-tmp`. `external/` vốn đã
   bị `.gitignore` chặn nên không còn rác ở thư mục gốc.
2. **Chép bản build vào `bin/` và chạy từ đó** — `backend/asr/native.py::bootstrap()` đặt
   `TRANSCRIBE_LIBRARY=<project>/bin/transcribe.dll` **trước khi** `import transcribe_cpp`, nên
   binding nạp đúng bundle này thay vì provider trong site-packages. Nếu `bin/` không tồn tại
   (bản cài cho người dùng cuối chỉ có wheel) thì **không đặt gì cả** và hành vi cũ giữ nguyên.
3. **Config chọn backend + fallback + log** — xem bảng dưới.

**Config mới** (`ASRConfig` trong `backend/config.py`):

| Trường | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `backend` | `"auto"` | `auto` (Vulkan trước) / `cuda` / `vulkan` |
| `backend_fallback` | `True` | Tự fallback khi backend yêu cầu không khả dụng (kèm log WARNING) |
| `use_local_native` | `True` | Ưu tiên bundle trong `bin/` hơn provider đã cài |
| `native_dir` | `""` | Thư mục bundle; trống = `<project_root>/bin` |

**Thứ tự ưu tiên & fallback** (`backend/asr/native.py::resolve_backend`):

| Yêu cầu | Thứ tự thử | Fallback về đâu |
| :--- | :--- | :--- |
| `auto` *(mặc định)* | **vulkan** → cuda | Vulkan |
| `cuda` | cuda → vulkan | **Vulkan** |
| `vulkan` | vulkan → cuda | CUDA — chỉ khi bundle không có Vulkan |

> **Vì sao `auto` ưu tiên Vulkan (quyết định của người dùng).** `transcribe.cpp` hỗ trợ Vulkan
> **chính thức** ⇒ wheel trên PyPI là đường **chắc chắn chạy trên mọi máy**. Bản CUDA do dự án
> **tự build** và **chưa được kiểm chứng trên mọi cấu hình**, nên nó là **tuỳ chọn phải chỉ định
> rõ** (`asr.backend = "cuda"`), không phải mặc định. Đây là đánh đổi có chủ ý: bỏ 1,53× tốc độ để
> lấy tính chắc chắn — và cả hai đều nhanh hơn thời gian thực rất nhiều (RTF 0,031 vs 0,020).
> Đổi lại chỉ cần một dòng nếu muốn tốc độ.

Log khi fallback là **WARNING** và nêu rõ lý do + những gì thực có:

```
[WARNING] [ASR] ASR backend: 'cuda' KHÔNG khả dụng ⇒ FALLBACK sang 'vulkan'.
                (yêu cầu='cuda', hiện có: cpu, vulkan; device: vulkan=Vulkan0, cpu=CPU)
```

Log lúc khởi động (một dòng, cho người vận hành):

```
[INFO ] [ASR] [STARTUP] ASR backend: yêu cầu='auto' → thực tế='cuda' |
              native: bundle bin/ (default) | có sẵn: cpu, cuda, vulkan |
              device: cuda=CUDA0, vulkan=Vulkan0, cpu=CPU
```

Log lúc nạp model nay nêu cả yêu cầu và nguồn native:

```
[INFO ] [ASR] Nạp thành công ASR Model 'qwen3-asr-0.6b'(Arch: qwen3_asr, Backend: CUDA0,
              yêu cầu: 'auto', Streaming: False, max_audio=5218.6s, native: bin/)
```

`/health` (`_asr_runtime_info`) nay trả thêm: `native_source`, `native_bundle_dir`,
`library_path`, `available_backends`, `devices`, `backend_requested`.

**Một lỗi thứ tự nạp đã phát hiện và sửa.** `backend/asr/engine.py` trước đây `import transcribe_cpp`
ở dòng 37 nhưng `setup_cuda_dll_paths()` tận dòng 54 — tức native được dlopen **trước khi** các
thư mục DLL CUDA (torch/lib) được đăng ký. Với backend Vulkan thì vô hại, nhưng với CUDA thì
`ggml-cuda.dll` có thể không tìm thấy `cudart64_13.dll`/`cublas64_13.dll`. Nay `backend/asr/__init__.py`
gọi `bootstrap()` ở **dòng đầu**, trước mọi import chạm `transcribe_cpp` (kể cả
`backend/asr/adapters.py` cũng import nó). `bootstrap()` còn **phát hiện và cảnh báo** trường hợp
`transcribe_cpp` đã nằm trong `sys.modules` — tức bundle không được áp dụng — thay vì log sai rằng
"đang dùng bin/".

**Kiểm chứng đã chạy**

| Hạng mục | Kết quả |
| :--- | :--- |
| `bin/` nạp được cả 3 backend | ✅ `Model(backend=)` cho cuda/vulkan/cpu đều OK |
| CUDA không cần toolkit trên PATH | ✅ chỉ dựa vào `torch/lib` (cudart64_13/cublas64_13/cublasLt64_13) |
| Fallback `cuda` → `vulkan` | ✅ bundle không có `ggml-cuda.dll`: `resolve_backend("cuda") == "vulkan"`, model nạp ra `Vulkan0`, có WARNING |
| `auto`/`cuda`/`vulkan`/giá trị sai | ✅ lần lượt cuda/cuda/vulkan; giá trị sai (`"cpu"`, `"bogus"`) → WARNING + coi như `auto` |
| Backend `cpu` đã bị loại | ✅ `_available_kinds()` chỉ trả cuda/vulkan. CPU *chạy được* nhưng ở **RTF 1,6** (8,1 s cho clip 5,08 s) — chậm hơn thời gian thực nên vô dụng cho phụ đề (đo bằng `external/build-tmp/probe_cpu_backend.py`). `bin/ggml-cpu.dll` vẫn được giữ vì ggml cần nó cho op không offload được |
| Đường ứng dụng thật (`backend.main`) | ✅ log khởi động đúng, model nạp `Backend: CUDA0 ... native: bin/` |
| CUDA graphs có hiệu lực | ✅ log native có `ggml_backend_cuda_graph_compute: CUDA graph warmup complete` |
| Bộ test đầy đủ | ✅ **268 PASS, 0 FAILED, 0 ERROR** |

**Còn lại của Nhánh A:** `GpuArbiter` — **ĐÃ TRIỂN KHAI XONG** (bước A.1–A.4 + đo A.5 trên tải tổng
hợp + quyết định A.6), xem **§4.3b**. Mặc định vẫn `config.gpu.scheduler_enabled = False`; việc duy
nhất còn lại là đo lại trên **phiên video thật** để quyết định có bật mặc định hay không.

### 4.3. Nhánh A — kế hoạch triển khai theo bước

| Bước | Việc | Sản phẩm | Nghiệm thu |
| :--- | :--- | :--- | :--- |
| A.1 | Thêm `GpuConfig` vào `config.py` (`scheduler_enabled=False`, `commit_reserve_max_ms=1500`, `preview_droppable=True`) | config + test đọc config | test config PASS |
| A.2 | Viết `backend/core/gpu_scheduler.py` (priority semaphore + reservation window + metric) | module + test đơn vị thuần asyncio (không cần GPU) | test: P0 vượt P3; trần chống starvation; tắt flag = hành vi cũ |
| A.3 | Nối arbiter vào 3 điểm nhận job: `_process_translation_item`, `_process_tts_item`, vòng preview/commit ASR | code nối sau flag | tắt flag ⇒ toàn bộ test hiện có PASS nguyên trạng |
| A.4 | Thêm metric: `gpu.wait_ms{stage}`, `gpu.commit_preempted_total`, `gpu.reserve_extended_total` | metric mới | metric xuất hiện trong `metrics_report.json` |
| A.5 | Đo A/B trên máy GPU: flag off vs on, cùng một video, cùng model | bảng số | p99 commit latency giảm; số spike > 1,5 s giảm; **WER không đổi** |
| A.6 | Chỉ bật mặc định nếu A.5 đạt; nếu không, giữ `False` và ghi lại số liệu phản bác | quyết định | ghi vào `05_measurements_and_status.md` |

### 4.3b. Nhánh A — KẾT QUẢ TRIỂN KHAI GpuArbiter (bước A.1–A.6)

#### A.1–A.2 — Config + module ✅

`GpuConfig` trong `backend/config.py` (mặc định **TẮT**):

| Trường | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `scheduler_enabled` | `False` | Bật/tắt arbiter |
| `commit_reserve_ms` | `1500` | Cửa sổ dành riêng GPU sau khi commit bắt đầu **và** trần chống starvation |
| `admit_max_wait_ms` | `1200` | Trần thời gian một job ưu tiên thấp chịu chờ |
| `admit_poll_ms` | `15` | Chu kỳ kiểm tra khi đang chờ |

`backend/core/gpu_scheduler.py` — `GpuArbiter`. **Điểm quan trọng về thiết kế:** nó **không** đặt
"GPU lock" toàn cục. Khóa toàn cục sẽ ép ASR + dịch + TTS chạy tuần tự và làm e2e latency **tệ hơn**
(mất phần chồng lấn vốn có). Arbiter chỉ làm **một** việc: **trì hoãn việc KHỞI ĐỘNG** job ưu tiên
thấp khi ASR đang chạy. Nó không thể preempt CUDA kernel đang chạy (driver không cho, và hủy giữa
lượt sinh token là không an toàn) ⇒ đây là bài toán **thời điểm**, không phải bài toán lock.

Vì commit ASR chạy trong `ThreadPoolExecutor` (không phải event loop), trạng thái được giữ bằng
`threading.Lock` + mốc thời gian, còn `admit()` là async và chỉ *poll* — không cần
`call_soon_threadsafe`, không phụ thuộc event loop cụ thể. Mọi lần chờ đều có trần ⇒ **không thể
treo pipeline**. `commit_end()` đóng cửa sổ **ngay** (không bắt chờ hết `commit_reserve_ms`).

#### A.3–A.4 — Nối vào 3 stage + metric ✅

| Stage | Điểm nối | Cơ chế |
| :--- | :--- | :--- |
| ASR commit | `asr/engine.py::_infer_with_watchdog(..., is_commit=True)` (gọi từ dòng `final_text = await ...`) | `commit_begin()` / `commit_end()` trong `try/finally` |
| ASR preview | cùng hàm, `is_commit=False` (mặc định) | đi thẳng, không bao giờ chờ |
| Dịch | `translation/engine.py::translate_sentence` + `translate_stream` | `await gpu_arbiter.admit(PRIORITY_TRANSLATION)` |
| TTS | `tts/engine.py::synthesize_clone_bytes` + `synthesize_clone` | `await gpu_arbiter.admit(PRIORITY_TTS)` |

Metric: `gpu.admit_blocked` (counter), `gpu.admit_wait_ms` (metric), `gpu.reserve_extended` (counter).
`/health` trả thêm `gpu_arbiter` = `{enabled, commit_active, blocked_total, waited_ms_total}`.

#### A.5 — Đo A/B (⚠️ **trên tải tổng hợp**, chưa phải phiên video thật)

Kịch bản: `external/build-tmp/a2_1_ab_measure.py` — chạy **model thật** (ASR `qwen3-asr-0.6b` trên
CUDA + dịch `Hy-MT2-1.8B-Q8` trên CUDA) tranh chấp cùng một RTX 5060 Ti. Dịch chạy **liên tục
back-to-back** (bão hoà GPU có chủ ý), commit ASR phát mỗi ~0,35 s. **20 commit mỗi cấu hình, chạy
hai thứ tự** để loại trừ hiệu ứng thứ tự/thời điểm:

| Chỉ số | OFF (off-first) | ON (off-first) | OFF (on-first) | ON (on-first) | Kết luận |
| :--- | ---: | ---: | ---: | ---: | :--- |
| commit p50 | 226,5 ms | **110,4 ms** | 222,2 ms | **116,8 ms** | **−49 %** |
| commit p95 | 246,4 ms | **144,3 ms** | 245,4 ms | **153,0 ms** | **−39 %** |
| commit max | 246,8 ms | **180,7 ms** | 268,6 ms | **153,0 ms** | **−35 %** |
| dịch xong | 60 | 45 | 58 | 45 | **−24 %** |

Hai thứ tự cho kết quả gần như trùng nhau ⇒ **hiệu ứng tái lập được, không phải artefact**.
Thống kê arbiter: `blocked_total=20` (mọi commit đều chặn được ít nhất 1 job dịch),
`waited_ms_total≈1500–1570 ms` ⇒ trung bình **~77 ms** chờ cho mỗi câu dịch bị hoãn.

**Đọc kết quả cho đúng:**
- Arbiter **giảm mạnh latency commit** ở mọi percentile (p50 −49 %, p95 −39 %, max −35 %). Đây là
  chỉ số người dùng cảm nhận trực tiếp (độ nhạy phụ đề).
- Cái giá là **−24 % thông lượng dịch** — nhưng con số này chỉ xuất hiện dưới **bão hoà tổng hợp**.
  Trong dùng thật, dịch chạy ở RTF ~0,03 (nhanh hơn thời gian thực ~30×) nên còn thừa công suất
  rất lớn; 45 hay 60 câu trong ~10 s đều vượt xa nhu cầu (thực tế ~1 câu/2–5 s).
- ⚠️ **Giới hạn của phép đo:** đây là **tải tổng hợp**, KHÔNG phải "cùng một video, cùng model" như
  A.5 yêu cầu. Chưa đo được p99 trên phiên thật, chưa đo số spike > 1,5 s, và **chưa xác nhận WER
  không đổi trong điều kiện có arbiter** (arbiter không đổi nội dung suy luận — nó chỉ đổi *thời
  điểm* — nên về lý thuyết WER không đổi, nhưng chưa chạy harness để chứng minh).

#### A.6 — Quyết định: **giữ mặc định TẮT**

Lý do:
1. A.5 mới đạt một **nửa**: cơ chế đã được chứng minh dưới tải tổng hợp, nhưng **chưa đo trên phiên
   video thật** — mà đó mới là điều kiện A.6 đặt ra.
2. Bật mặc định sẽ đổi đặc tính độ trễ cho **mọi** người dùng dựa trên một phép đo proxy.
3. Chi phí thông lượng (dù nhỏ trong dùng thật) chưa được kiểm chứng là vô hại trên tải thật.

**Cách bật (một dòng):** đặt `gpu.scheduler_enabled = true` trong `backend/config.py`, hoặc qua
`POST /api/config` (`{"gpu": {"scheduler_enabled": true}}` — `hot_reload` đã hỗ trợ nested dict).
Theo dõi qua `/health` → `gpu_arbiter`.

**Việc còn lại để chốt A.6 (cần một buổi chạy phiên thật):** chạy M1/M2 (§6.2) với flag off vs on
trên cùng một video, xác nhận p99 và số spike > 1,5 s giảm, rồi chạy `test_09_wer_ab.py` với flag on
để chốt WER không đổi. Nếu đạt ⇒ đổi mặc định sang `True`.


**Rollback:** flag `scheduler_enabled=False` là rollback tức thời, không cần revert code.

### 4.4. Nhánh B — KHÔNG build được ASR CUDA (kịch bản gần như chắc chắn)

Giữ ASR trên **Vulkan** và dịch/TTS trên **CUDA**. Đây là **hai** backend GPU độc lập dùng chung một
card vật lý, nên arbiter **không thể** điều phối bằng semaphore đơn giản. Kế hoạch thực dụng:

| Bước | Việc | Vì sao |
| :--- | :--- | :--- |
| B.1 | **Đo trước, sửa sau.** Dùng `gpu_usage_probe.py` + `nvidia-smi dmon` để lấy số thật: bao nhiêu % spike commit trùng thời điểm dịch/TTS chạy | Tránh tối ưu theo phỏng đoán; audit ghi spike ~2,5 s nhưng chưa tách nguyên nhân |
| B.2 | **Giãn dịch/TTS quanh commit (soft reservation).** Arbiter vẫn dùng được, nhưng thay vì chặn theo slot GPU thì nó chỉ **trì hoãn khởi động câu dịch/TTS kế tiếp** trong cửa sổ ngắn quanh commit ASR | Không cần điều khiển được Vulkan; chỉ cần **dịch chuyển thời điểm** công việc CUDA |
| B.3 | **Hạ ưu tiên CUDA stream** cho dịch/TTS (`torch.cuda.Stream(priority=...)`) | Bổ trợ B.2, rẻ |
| B.4 | **Xem lại `preview_window_sec` / nhịp preview**: giảm số lần ASR chiếm GPU trong lúc dịch chạy | Ít lần tranh chấp hơn ⇒ ít spike hơn, đổi lại preview thưa hơn (cần đo WER + cảm nhận) |
| B.5 | **A2-2 đã làm ở Phase 1 chính là fix lớn nhất của nhánh này** — bỏ sync toàn device đã gỡ phần lớn priority inversion do TTS gây ra | Đo lại sau Phase 1 để biết còn lại bao nhiêu |

**Kết luận nhánh B:** nếu không có ASR CUDA, mục tiêu thực tế **không phải** "scheduler hoàn hảo" mà
là "**giảm số lần hai backend chồng lấn**" — và đó là bài toán **đo đạc + dịch chuyển thời điểm**,
không phải bài toán lock. Đây là lý do A2-1 được xếp **sau cùng**: kết quả Phase 1 (đặc biệt A2-2) có
thể đã giải quyết phần lớn triệu chứng, khiến khối lượng công việc còn lại nhỏ hơn nhiều so với dự
đoán trong audit.

### 4.5. Các fix "liên quan A2-1" bị hoãn theo (sẽ làm sau, nếu build được ASR CUDA)

| ID | Việc | Điều kiện làm |
| :--- | :--- | :--- |
| **B5-2** | `translation` executor `max_workers=2` | Sau A2-1, và **chỉ khi** đo được VRAM còn dư ≥ ~2 GB |
| **T2** | ASR `max_inflight_infer = 2` | ❌ Không làm được: native giới hạn 1 stream/model |
| **L5** | Bật `preview_reuse_for_commit` | **Không phụ thuộc A2-1** — nhưng cần đo WER chênh ≤ 0,3 % trước (audit đã ghi rõ). Có thể làm độc lập khi có máy GPU |
| **T3** | Ép ASR chạy CUDA | Chính là gate G1/G2 của A2-1 |

---

## 5. PHASE 4 — CÁC VIỆC NGOÀI PHẠM VI SỬA NGAY

### 5.1. A3-1 — Phình RAM native (transcribe.cpp) — cần patch upstream

Hiện chỉ có **workaround**: `maybe_recycle_native()` (`asr/engine.py:170-204`) đóng phiên native khi
RSS vượt mốc nạp model + `native_recycle_rss_delta_mb` (2048 MB), lần sau nạp lại **tốn ~10 s**.

**Kế hoạch (khuyến nghị, không thực hiện trong repo này):**

1. **Ngắn hạn:** nếu máy đủ RAM, đặt `config.asr.native_recycle_rss_delta_mb = 0` để tránh gián đoạn
   10 s; theo dõi RSS để biết ngưỡng an toàn thật.
2. **Trung hạn:** patch `external/transcribe.cpp` theo các mục F-31…F-35 trong
   `02_phu_luc_transcribe_cpp.md` (threadpool CPU dùng-một-lần mỗi graph compute; scheduler/compute
   context dựng lại mỗi `run()`; H2D/D2H không cần thiết).
3. **Nghiệm thu:** chạy 1 phiên ≥ 60 phút, RSS phải **phẳng** (không tăng đơn điệu); không cần
   recycle.

Đây là việc **lớn nhất về ROI dài hạn** nhưng nằm ngoài repo Python ⇒ tách thành công việc riêng.

### 5.2. A4-1 — Đổi model đọc lại GGUF trong request HTTP

Là hệ quả trực tiếp của thiết kế singleton: chỉ có 1 model dùng chung nên không thể nạp song song.
Các hướng (đều là quyết định sản phẩm, không phải fix bug):

- Chấp nhận (hiện tại) + **cải thiện UX**: trả `202` ngay, đẩy tiến trình qua WS (một phần đã có ở
  `translation_hotswap`).
- Hoặc model pool (VRAM ×N) — chỉ khi cần nhiều phiên.

**Khuyến nghị:** giữ nguyên hành vi, chỉ cải thiện thông báo tiến trình.

### 5.3. F-03 / B4-1 — Scale = 1 session

**Không sửa.** Đây là thiết kế có chủ ý ("cá nhân, 1 video") và được ghi trong README. Muốn N phiên
phải có N model instance + model pool + lock theo phiên ⇒ VRAM ×N. Chỉ làm khi có yêu cầu sản phẩm
rõ ràng.

---

## 6. THỨ TỰ THỰC HIỆN & TIÊU CHÍ NGHIỆM THU

### 6.1. Thứ tự

```
✅ Phase 1  (xong)  A2-2 → A2-3 → B10-1 → B6-3 → A1-1 → B5-1 → B9-4
📋 Phase 3  (sau)   Đo lại sau Phase 1  →  Gate G1/G2  →  Nhánh A hoặc Nhánh B  →  A/B  →  quyết định
📋 Phase 4  (tách)  A3-1 (patch upstream)  →  A4-1 (UX)
```

**Vì sao đo lại sau Phase 1 là bước bắt buộc trước Phase 3:** A2-2 (bỏ sync toàn device) và A2-3
(trả VRAM) trực tiếp làm giảm tranh chấp GPU. Nếu sau Phase 1 số spike đã về mức chấp nhận được thì
A2-1 trở thành **không cần thiết** — và đó là kết quả tốt, không phải thất bại.

### 6.2. Bộ số phải đo (trên máy GPU thật, cùng video + cùng model)

| # | Chỉ số | Lấy từ | Kỳ vọng |
| :--- | :--- | :--- | :--- |
| M1 | `commit_latency_ms` p50 / p95 / p99 | `metrics_report.json` | p99 giảm sau A2-2 |
| M2 | Số spike > 1500 ms / 10 phút | metrics | giảm |
| M3 | VRAM peak (MB) | `nvidia-smi dmon` | giảm vài GB khi TTS tắt (A2-3) |
| M4 | RSS tiến trình theo thời gian | `gpu_usage_probe.py` | phẳng trong 30 phút (đối chiếu A3-1) |
| M5 | **WER** | `test_09_wer_ab.py` | **KHÔNG tăng** (điều kiện tiên quyết của mọi fix) |
| M6 | CPU utilization trên máy 4–6 nhân | Task Manager | giảm (A1-1) |
| M7 | Thời gian 1 lần quét DOM khi cache miss | Firefox Performance | giảm rõ (B10-1) |

**Nguyên tắc bất di bất dịch:** mọi fix ở Phase 1 được thiết kế để **không đổi kết quả nhận dạng/dịch**.
Nếu M5 xấu đi, revert fix tương ứng trước khi làm bất cứ việc gì khác.

### 6.3. Trạng thái kiểm chứng hiện tại (trung thực)

| Việc | Đã làm | Chưa làm (cần máy GPU) |
| :--- | :--- | :--- |
| Cú pháp Python | ✅ `py_compile` tất cả file đã sửa | — |
| Cú pháp JS | ✅ `node --check` | — |
| Bộ test mặc định | ✅ **249 PASS / 0 FAILED** / 19 ERROR do sandbox chặn `tmp_path` (không liên quan) | — |
| Test cần model/GPU (`-m slow`, `-m full`) | ❌ | Cần chạy |
| Đo M1–M7 | ❌ | Cần chạy |

---

## 7. RỦI RO & ROLLBACK

| Fix | Rủi ro chính | Giảm thiểu | Rollback |
| :--- | :--- | :--- | :--- |
| A2-2 | Race cross-stream khi đọc trọng số | 1 `torch.cuda.synchronize()` ở đường nạp model; fallback về sync toàn device khi không có stream | trả `_run_generate`/`_synthesize_audio` về sync cũ |
| A2-3 | Phiên khác phải nạp lại TTS | chỉ unload khi **mọi** phiên đều tắt TTS | bỏ 2 lời gọi `unload_model` |
| B10-1 | Không tìm thấy video nằm sâu trong Shadow DOM | fast path chỉ bỏ qua khi light DOM **đã có** video dùng được; logic xếp hạng không đổi | trả `findVideo` về bản cũ |
| B6-3 | Cache trả chuỗi cũ | `add`/`clear` đều bật dirty; test_05 kiểm tra nội dung | bỏ cache |
| A1-1 | Giảm thread làm chậm trên máy nhiều nhân | chỉ hạ khi `< 12` nhân; máy tham chiếu giữ `4` | trả `threads`/`n_threads` về `4` |
| B5-1 | Executor 1 worker | tương đương hành vi cũ vì semaphore = 1 | trả `max_workers=2` |
| B9-4 | VRAM không được trả khi unload | **giữ** `gc.collect()` ở unload | — |

**Rollback toàn bộ:** `git checkout -- backend/asr/engine.py backend/config.py backend/main.py backend/translation/context.py backend/tts/engine.py backend/ws/handler.py extension_firefox/content/content-script.js`
(không có commit nào được tạo, nên đây là rollback sạch).

---

## 8. VIỆC CẦN NGƯỜI DÙNG QUYẾT ĐỊNH

1. **A3-1 (leak native):** có đầu tư patch `external/transcribe.cpp` không? Đây là ROI dài hạn lớn
   nhất nhưng là công việc C++/ggml riêng, không phải fix Python.
2. **B3-1 (ring buffer Int16):** chấp nhận đánh đổi "mất bảo đảm bit-exact float32 để tiết kiệm
   1,92 MB" hay không? Khuyến nghị của tôi: **không**.
3. **A2-1 / G2:** có sẵn sàng tự build transcribe.cpp với CUDA (và gánh ma trận wheel CUDA theo driver)
   không? Nếu không, Phase 3 đi theo **Nhánh B** (§4.4) và nên bắt đầu bằng đo đạc.
4. **L5 (`preview_reuse_for_commit`):** cần một buổi đo WER trên máy GPU để quyết định; tiết kiệm
   ~420 ms/câu nhưng có nguy cơ mất từ cuối câu.
5. **A4-1:** chấp nhận việc đổi model chặn phiên đang chạy, hay muốn cải thiện UX thông báo tiến trình?

---

## 9. PHỤ LỤC — DANH SÁCH THAY ĐỔI THỰC TẾ TRONG WORKING TREE

| File | Nội dung | Finding |
| :--- | :--- | :--- |
| `backend/tts/engine.py` | `_cuda_stream` + `_invoke_generate()` + sync theo stream; `unload_model()` sạch hơn; bỏ `gc.collect()` ở warmup, giữ ở unload | A2-2, A2-3, B9-4 |
| `backend/main.py` | `await asyncio.to_thread(...unload_model)` khi tắt TTS qua REST | A2-3 |
| `backend/ws/handler.py` | `_maybe_unload_tts_when_idle()` + gọi khi popup tắt TTS | A2-3 |
| `backend/config.py` | `_AUTO_THREADS` theo `os.cpu_count()`; áp cho `asr.threads` và `translation.n_threads` | A1-1 |
| `backend/asr/engine.py` | `_EXECUTOR` suy ra từ `config.asr.max_inflight_infer`; comment giải thích việc loại B3-1 | B5-1 |
| `backend/translation/context.py` | Cache `get_context_str()` + cờ dirty | B6-3 |
| `extension_firefox/content/content-script.js` | `findVideo()`: fast path light DOM, TreeWalker, trần node, throttle quét | B10-1 |

**Không thay đổi (đã revert có chủ ý):** `backend/core/audio_buffer.py` (B3-1),
`feed_audio()` trong `backend/asr/engine.py` (B3-1), cờ `already_deduped` trong `backend/ws/handler.py` (B8-1).

---

**Tài liệu tham chiếu:** `report/audit/BAO_CAO_AUDIT_HIEU_NANG_Hy3.md`,
`report/audit/00_BAO_CAO_AUDIT_HIEU_NANG.md`, `report/audit/02_phu_luc_transcribe_cpp.md`,
`report/audit/05_measurements_and_status.md`, `report/04_commit_logic/report.md`.
