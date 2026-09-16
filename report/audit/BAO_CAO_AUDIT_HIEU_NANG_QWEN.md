# 📊 BÁO CÁO AUDIT HIỆU NĂNG & KIẾN TRÚC CHUYÊN SÂU
**Dự án:** `khachuy279/vibe-translation-addon-transcribe_cpp`  
**Vai trò:** Senior Software Architect + Performance Engineer + Code Reviewer  
**Ngày đánh giá:** 16/09/2026  
**Phạm vi:** Toàn bộ backend Python, binding C++ (`transcribe.cpp`), pipeline WebSocket và cơ chế quản lý tài nguyên.

---

## 📌 TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)
Hệ thống được thiết kế tốt cho trường hợp sử dụng cá nhân (100% local, privacy-first) với pipeline rõ ràng: **Audio → VAD → ASR → Translation → TTS → WebSocket**. Tuy nhiên, khi phân tích sâu vào mã nguồn và báo cáo đo đạc nội bộ (`report/audit/`), hệ thống tồn tại **các điểm nghẽn nghiêm trọng (Critical Bottlenecks)** ngăn cản việc mở rộng quy mô (scaling) và gây lãng phí tài nguyênCompute lớn. 

Vấn đề cốt lõi nhất là **cơ chế ASR Preview có độ phức tạp O(N²)** kết hợp với **kiến trúc Singleton toàn cục**, khiến hệ thống bị chặn cứng ở 1 session, tiêu tốn CPU/GPU không cần thiết và dễ gặp sự cố tràn bộ nhớ (OOM) khi tải cao.

---

## 🔍 PHÂN TÍCH CHI TIẾT THEO DANH MỤC YÊU CẦU

### 1. Điểm nghẽn hiệu năng (Performance Bottlenecks)
* **🔴 CPU Bottleneck (F-01 - CRITICAL):** Cơ chế ASR Preview trong `backend/asr/engine.py` (dòng 353-376) có độ phức tạp **O(N²)**. Mỗi lần poll (300ms), hệ thống gọi `_run_inference_sync` trên toàn bộ đoạn audio từ đầu câu (`start_s` đến `current_total`). Không có incremental state hay KV cache reuse. Ở câu dài 30s, chi phí compute tăng theo cấp số nhân.
* **🔴 GPU Bottleneck (F-01, F-19):** Tầng native C++ (`transcribe.cpp`) luôn tính lại Mel spectrogram từ `pcm[0]` và wipe sạch KV cache mỗi lần `run()` (không có API prefix/state reuse). Ngoài ra, có dấu hiệu runtime fallback về Vulkan/CPU (`ggml-vulkan.dll`) thay vì tận dụng CUDA, gây nghẽn GPU nghiêm trọng.
* **🟠 I/O Bottleneck (F-21, F-16):** Translation engine có nguy cơ tải model từ HuggingFace trên hot path nếu cấu hình không chặt. Ở phía client, việc gửi TTS dưới dạng base64 qua WebSocket làm tăng 33% payload (dù đã có hỗ trợ binary frame, cần đảm bảo client luôn dùng protocol v3).

### 2. Vấn đề về Bộ nhớ & Quản lý tài nguyên
* **🔴 Memory Growth bất thường (F-39):** Báo cáo audit ghi nhận mức tăng RSS lên tới **+73 GB private RAM** trong lúc inference native/Vulkan đang chạy. Đây là dấu hiệu rò rỉ bộ nhớ ở tầng C++ mà Python không kiểm soát được.
* **🔴 Unbounded Buffer / Queue Backlog (F-04):** `ThreadPoolExecutor` (`_EXECUTOR`) trong `backend/asr/engine.py` có **hàng đợi không giới hạn**. Nếu inference bị treo/chậm, các task `audio_slice` tiếp tục được push vào → RAM tăng 66-85 MB/s cho đến khi OOM.
* **🟠 Copy dữ liệu không cần thiết (F-01b):** Hàm `audio_buffer.get_slice` tạo `np.empty(requested_len)` và copy toàn bộ đoạn audio mỗi lần preview. Ở 30s, điều này gây ra ~90 MB copy bộ nhớ cho một câu duy nhất.
* **🟠 Allocation quá nhiều:** Việc materialize segments/words/tokens khi không cần thiết (khi `timestamps="auto"`) gây áp lực allocation liên tục lên heap.

### 3. Vấn đề về Đồng thời & Luồng (Concurrency)
* **🔴 Lock / Mutex Contention (F-03):** Các class `TranscribeEngine`, `GGUFTranslator`, `OmniVoiceTTS` đều sử dụng **class-level `_infer_lock` và `_shared_lock`**. Mọi inference trên toàn hệ thống bị **serialize toàn cục**. Không thể tận dụng đa lõi CPU cho các tác vụ độc lập.
* **🟠 Thread Contention:** VAD Processor (`backend/vad/processor.py`) và luồng ghi WebSocket có thể tranh chấp khi truy cập vào shared audio buffer, dù đã có `threading.Lock`, nhưng tần suất cao vẫn gây overhead.
* **🟠 Race Condition tiềm ẩn (F-50):** Cơ chế hot-swap model dịch từng có lỗi "unload trước khi load xong". Dù đã được patch bằng `_switch_lock` và load-before-swap, đây vẫn là vùng mã nhạy cảm cần giám sát chặt.

### 4. Vấn đề về Pipeline & Luồng dữ liệu
* **🔴 Xử lý lặp lại (F-09):** Preview và Commit trùng lặp 100% compute. Commit chạy lại `_run_inference_sync` trên đoạn audio mà preview cuối cùng vừa xử lý xong (chỉ dài thêm 300-600ms do hangover). **~50% compute ASR bị lãng phí** cho các câu ngắn.
* **🟠 Pipeline stage gây nghẽn downstream:** Chuỗi ASR → Translation → TTS là tuần tự chặt chẽ. Nếu ASR bị trễ (do O(N²)), toàn bộ pipeline phía sau sẽ bị backlog, gây ra **Latency Spike** tích lũy.
* **🟠 CPU ↔ GPU Transfer & Conversion:** Việc chuyển đổi liên tục `NumPy (float32) → C++ binding → PyTorch Tensor` mà không có zero-copy optimization gây overhead đáng kể trên bus hệ thống.

### 5. Vấn đề về Logging & Instrumentation
* **🟠 Logging trong hot path (F-12):** Mặc dù đã dùng `logger.debug`, nhưng nếu level log được đặt quá thấp, việc ghi log trong vòng lặp poll 300ms sẽ gây I/O disk contention và khóa GIL.
* **🔴 Polling không hiệu quả (F-01):** Polling mỗi 300ms để lấy preview, nhưng mỗi lần poll lại kích hoạt một inference O(N) đầy đủ. Đây là sự kết hợp chết người giữa polling và thuật toán kém hiệu quả.

### 6. Khả năng mở rộng (Scalability)
* **🔴 Kiến trúc chặn cứng ở 1 Session (F-03):** 
  - `TranscribeEngine._shared_session`: 1 native session cho tất cả → audio các session sẽ bị trộn lẫn state.
  - Muốn chạy N session song song, hệ thống bắt buộc phải nạp N bản model vào VRAM (không khả thi với GPU 16GB như RTX 5060 Ti 16GB của user nếu model lớn).
  - Đây là rào cản kiến trúc lớn nhất để biến dự án từ "cá nhân" thành "dịch vụ".

---

## 🚀 ĐỀ XUẤT TỐI ƯU HÓA (ACTIONABLE RECOMMENDATIONS)

### 🛑 P0: Sửa ngay (Critical - Đúng đắn / An toàn / Tiết kiệm lớn)
1. **Giới hạn cửa sổ Preview (Sliding Window):** 
   - **Vị trí:** `backend/asr/engine.py:353-376`
   - **Hành động:** Thay vì `get_slice(start_s, current_total)`, chỉ lấy `get_slice(max(start_s, current_total - 6s), current_total)`. Hoặc tốt hơn, chuyển sang cơ chế **incremental feed** trên 1 stream sống suốt câu nếu `transcribe.cpp` hỗ trợ.
   - **Lợi ích:** Giảm chi phí ASR từ **O(N²) → O(N)**, giảm độ trễ preview xuống mức chunk.
2. **Tái sử dụng kết quả Preview cho Commit:**
   - **Vị trí:** `backend/asr/engine.py:322-351`
   - **Hành động:** Nếu cửa sổ audio của commit không thay đổi đáng kể so với preview cuối cùng (chênh lệch < 0.5s), hãy tái sử dụng trực tiếp `preview_text` thay vì gọi lại `_run_inference_sync`.
   - **Lợi ích:** **Giảm ~30-50% compute ASR** thực tế, giảm latency end-to-end mà không ảnh hưởng chất lượng.
3. **Triển khai Backpressure cho Executor:**
   - **Vị trí:** `backend/asr/engine.py` (khởi tạo `_EXECUTOR`)
   - **Hành động:** Sử dụng `ThreadPoolExecutor(max_workers=2)` kết hợp với `asyncio.Queue(maxsize=3)` thay vì submit trực tiếp. Nếu queue đầy, **bỏ qua (drop)** các preview frame mới nhất (preview là best-effort, chỉ commit mới là bắt buộc).
   - **Lợi ích:** Ngăn chặn hoàn toàn kịch bản RAM tăng 66-85 MB/s (F-04).

### ⚠️ P1: Cải thiện cao (High Impact - Giảm tài nguyên & Tăng throughput)
4. **Tắt Timestamps khi không cần thiết:**
   - **Vị trí:** `backend/asr/engine.py:296`
   - **Hành động:** Đảm bảo `timestamps="none"` khi gọi `session.run()` cho preview.
   - **Lợi ích:** Tránh materialize segments/words/tokens, giảm mạnh allocation CPU và overhead binding.
5. **Tách Lock theo Instance thay vì Class-level:**
   - **Vị trí:** `backend/asr/engine.py`, `backend/translation/engine.py`
   - **Hành động:** Chuyển `_infer_lock` từ class attribute sang instance attribute. Nếu tương lai hỗ trợ multi-session, mỗi session sẽ có lock riêng, cho phép song song hóa inference trên các model instance khác nhau.
6. **Tối ưu hóa Serialisation (WebSocket):**
   - **Vị trí:** `backend/ws/serializers.py`, `extension_firefox/lib/tts-player.js`
   - **Hành động:** Bắt buộc sử dụng `compact=True` (payload v3) và `synthesize_clone_bytes` (binary frame) cho mọi kết nối hiện đại. Loại bỏ hoàn toàn base64 encoding ở hot path.

### 💡 P2: Tinh chỉnh kiến trúc (Medium Impact - Clean code & Scalability prep)
7. **Giám sát và Recycle Native Session (F-39):**
   - **Vị trí:** `backend/config.py` (`rss_runaway_delta_mb`, `native_recycle_rss_delta_mb`)
   - **Hành động:** Đảm bảo watchdog RSS hoạt động chính xác. Khi phát hiện tăng đột biến, chủ động gọi `session.cancel()` và tái tạo session thay vì để treo.
8. **Loại bỏ Abstraction không cần thiết:**
   - **Vị trí:** Các lớp wrapper giữa FastAPI và `transcribe.cpp`.
   - **Hành động:** Rà soát xem có lớp nào chỉ làm nhiệm vụ `return self.engine.run()` mà không thêm logic validation hay metrics không. Nếu có, hãy inline hoặc loại bỏ để giảm stack depth và GIL contention.
9. **Caching kết quả Translation:**
   - **Hành động:** Bổ sung LRU Cache (ví dụ: `functools.lru_cache` hoặc Redis local) cho các câu dịch trùng lặp phổ biến, bổ sung cho cơ chế `TranslationDeduplicator` hiện tại (chỉ lọc liên tiếp).

---

## 📈 KỲ VỌNG SAU KHI TỐI ƯU
| Chỉ số | Hiện tại (Ước tính) | Sau khi áp dụng P0 + P1 |
| :--- | :--- | :--- |
| **ASR Compute (câu 10s)** | O(N²) ~ 3-4 lần inference đầy đủ | **O(N)** ~ 1 lần inference + 1 partial |
| **Peak RAM Usage** | Tăng vô hạn (nguy cơ OOM) | **Giới hạn cứng** (nhờ backpressure queue) |
| **End-to-End Latency** | ~1.5s - 3.0s (tùy độ dài câu) | **< 800ms** (nhờ reuse preview & windowing) |
| **Khả năng mở rộng** | **1 session** (Singleton lock) | **N session** (sau khi tách instance lock) |

---

## 📎 PHỤ LỤC: CÁC FILE CẦN RÀ SOÁT ƯU TIÊN
1. `backend/asr/engine.py` (Dòng 320-390): Logic O(N²) và Lock toàn cục.
2. `backend/core/audio_buffer.py` (Dòng 130): Cơ chế copy `np.empty` trong `get_slice`.
3. `backend/config.py` (Dòng 39-55): Cấu hình `max_inflight_infer`, `inference_watchdog_sec`, `rss_runaway_delta_mb`.
4. `external/transcribe.cpp/src/arch/qwen3_asr/model.cpp` (Dòng 768-775): Xác nhận việc wipe KV cache ở tầng C++.

> **Lời khuyên của Architect:** Hệ thống hiện tại có nền tảng rất tốt (modular, có đo đạc metrics, có audit nội bộ). Tuy nhiên, để đạt được mục tiêu "real-time" thực sự và ổn định, **việc sửa lỗi O(N²) ở ASR Preview (P0.1) và triển khai Backpressure (P0.3) là bắt buộc phải làm ngay trước khi đưa vào sử dụng thực tế kéo dài.**