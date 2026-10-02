# Kế Hoạch Triển Khai: Lookahead Offline Batch Pipeline (v3)
## Kiến Trúc Hybrid: Qwen3-ASR + Qwen3-ForcedAligner-0.6B + Bi-directional Context Translation

- **Đơn vị / Dự án**: `vibe-translation-addon-transcribe_cpp`
- **Tập trung**: Nâng cấp **Pipeline B (Lookahead Persistent Audio RAM)** từ cơ chế "Giả lập Streaming" sang **"Offline Batch Windowing + Qwen3-ForcedAligner (Word Timestamps) + Bi-directional Context Translation"**.
- **Mục tiêu chính**:
  1. Tận dụng tối đa năng lực nhận diện của mô hình Audio-LLM chủ đạo **Qwen3-ASR** với cửa sổ ngữ cảnh 15s – 30s để **giảm thiểu cực đại WER và CER**.
  2. Khắc phục hoàn toàn việc Qwen3-ASR không có native timestamps bằng việc tích hợp **`Qwen3-ForcedAligner-0.6B`** (NAR aligner siêu tốc ~30-50ms) để gắn mốc thời gian từng từ/ký tự với sai số cực nhỏ (AAS 32–52ms).
  3. Gom câu phụ đề tự nhiên theo dấu câu (Punctuation-guided Sentence Grouping), đảm bảo phụ đề hiển thị vừa mắt (4–10 từ, 2–4s) và khớp 100% với nhịp nói của video.
  4. Nâng cấp chất lượng dịch thuật LLM GGUF (Hunyuan-MT2, MiLM, Qwen) bằng cơ chế **ngữ cảnh trượt 2 chiều (Bi-directional Sliding Context)** mà không phá vỡ cấu trúc hiển thị 1-1 của phụ đề và nhịp lồng tiếng (TTS).
  5. Duy trì các cam kết vàng: **0.0s Zero Perceived Latency**, không giật lag, hỗ trợ tua video (Seek) tức thì qua cơ chế **Fast-Bootstrap**.

---

## 1. Bối Cảnh & Động Lực (Current Bottlenecks vs Proposed Vision)

### 1.1. Hiện Trạng Pipeline B (v2)
Trong phiên bản v2 hiện tại:
- Extension chặn các phân đoạn MSE từ trình duyệt và gửi về WebSocket `/ws/lookahead`.
- Backend giải mã qua `StreamDemuxer` và lưu trữ âm thanh liên tục vào `ContinuousAudioTimeline` (hỗ trợ tới 3 giờ RAM).
- **Hạn chế kỹ thuật**: Mặc dù backend đã sở hữu trước 10s – 15s audio tương lai, vòng lặp `_ingest_loop` và `_asr_loop` trong [`backend/ws/lookahead_handler.py`](file:///d:/vibe-translation-addon-transcribe_cpp/backend/ws/lookahead_handler.py) vẫn chia nhỏ audio thành các block 0.5s đẩy vào `VADProcessor` rồi chuyển tiếp sang `TranscribeEngine.stream_tokens()`.
- **Hệ quả**:
  - Mô hình ASR chỉ nhìn thấy từng lát cắt audio cục bộ, phải dựa vào `CommitManager` (VAD_SILENCE, MAX_DURATION, STABLE_PREFIX) để chốt câu $\rightarrow$ Dễ bị ngắt câu non, mất liên kết âm học ngữ điệu, không phục hồi được dấu câu chuẩn.
  - LLM dịch thuật chỉ nhận từng câu vụn vặt độc lập $\rightarrow$ Xưng hô lộn xộn, dịch máy móc, mất mạch văn.

### 1.2. Nhận Định Kỹ Thuật Quan Trọng Về Qwen3-ASR & Timestamps
- **Qwen3-ASR** (0.6B / 1.7B) là mô hình Audio-LLM thế hệ mới. Trong `transcribe.cpp` cũng như upstream HuggingFace, Qwen3-ASR **không hỗ trợ native timestamp tokens** (như `<|0.00|>` của Whisper). Decoder của model tập trung 100% vào việc sinh văn bản tự nhiên kèm dấu câu chuẩn xác.
- Để trích xuất mốc thời gian cấp từ/ký tự, giải pháp chính thức từ Alibaba là mô hình **`Qwen3-ForcedAligner-0.6B`**:
  - Kiến trúc Non-Autoregressive (NAR) chuyên trách việc gióng hàng (Forced Alignment) giữa Audio và Văn bản đã nhận dạng.
  - Hỗ trợ tới 11 ngôn ngữ (Anh, Nhật, Trung, Pháp, Đức, Tây Ban Nha, Ý, Nga, Hàn, Quảng Đông, Bồ Đào Nha).
  - Tốc độ suy luận cực nhanh: chỉ mất **30ms – 50ms** trên GPU cho 15s audio.
  - Độ chính xác căn chỉnh trung bình (AAS - Average Alignment Shift) chỉ **32.4ms – 52.9ms** (vượt trội so với WhisperX và MFA).

### 1.3. Luồng Kiến Trúc Mới (v3 Offline Batch + Forced Aligner)

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRÌNH DUYỆT: MSE Audio Chunks (đi trước 10s - 15s)                                     │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ CONTINUOUS AUDIO TIMELINE (Lưu trữ RAM tuyến tính 16kHz PCM Mono)                      │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 1: VAD-ASSISTED SILENCE BOUNDARY CHUNKER                                         │
│ • Quét tìm khoảng lặng VAD (>250ms) trong dải [playhead + 12s, playhead + 18s]        │
│ • Cắt thành 1 khối âm thanh trọn vẹn (VD: 14.5s), không bao giờ cắt vào giữa từ.      │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 2A: OFFLINE ASR DECODING (Qwen3-ASR qua transcribe.cpp GGUF)                      │
│ • Chạy 1 lần duy nhất cho toàn bộ khối 14.5s (Full audio context)                     │
│ • Trả về văn bản trọn vẹn kèm dấu câu chuẩn xác: "Hello world. How are you doing? ..." │
│ • RTF ~ 0.04 - 0.06 (khoảng 600ms GPU)                                                 │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 2B: NON-AUTOREGRESSIVE FORCED ALIGNMENT (Qwen3-ForcedAligner-0.6B)               │
│ • Đầu vào: PCM 14.5s + Văn bản nhận dạng từ Bước 2A + Ngôn ngữ                         │
│ • Suy luận siêu tốc (~35ms GPU), trả về mốc thời gian từng từ: [(word, t0, t1), ...]  │
│ • Sai số mốc thời gian < 50ms                                                          │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 2C: PUNCTUATION-GUIDED SUBTITLE GROUPING                                         │
│ • Dựa vào dấu câu (., ?, !, ...) gom các từ thành từng câu phụ đề vừa vặn (4-10 từ)   │
│ • Timestamp từng câu: pts_start = block_pts + words[0].t0, pts_end = words[-1].t1      │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 3: BI-DIRECTIONAL CONTEXT TRANSLATION (LLM GGUF)                                  │
│ • Dịch tuần tự từng câu con, nhưng BƠM NGỮ CẢNH:                                      │
│   - Quá khứ: 2 câu gốc + 2 câu dịch trước đó                                          │
│   - Tương lai: 1 câu gốc tiếp theo                                                    │
│ • Đầu ra: Đúng 1 câu dịch tương ứng 1-1, xưng hô nhất quán, văn phong điện ảnh         │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ BƯỚC 4: DISPATCH PHỤ ĐỀ & LỒNG TIẾNG (TTS)                                            │
│ • Đẩy mẻ phụ đề lookahead_subtitles về Extension trước 10s (độ trễ hiển thị: 0.0s)    │
│ • Nạp OmniVoice TTS tổng hợp trước thong thả, nén WSOLA khớp cửa sổ phụ đề            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Giải Pháp Chi Tiết Cho 3 Rào Cản Kỹ Thuật

### 2.1. Rào Cản 1: Cắt Khối Âm Thanh (Audio Chunking & Boundary Protection)

**Vấn đề**: Nếu cắt cứng audio ở mốc thời gian cố định (ví dụ cứ 15.0s cắt 1 lần), hệ thống chắc chắn sẽ cắt đôi một từ ngữ hoặc một âm tiết đang nói dở $\rightarrow$ Gây lỗi ảo giác (hallucination) ở cuối khối 1 và mất âm đầu ở khối 2.

**Giải pháp**: **VAD Silence-Gap Boundary Scanner**
1. **Dải quét mục tiêu**:
   - Khi vị trí hiện tại đang xử lý là $T_{\text{start}}$, ta đặt mục tiêu nạp một khối có thời lượng lý tưởng $L \approx 15.0\text{s}$ (phù hợp với cửa sổ tối ưu của Qwen3-ASR).
   - Vùng tìm kiếm ranh giới an toàn: $[T_{\text{start}} + 12.0\text{s}, \ T_{\text{start}} + 18.0\text{s}]$.
2. **Cơ chế tìm điểm cắt**:
   - Chạy engine VAD nhanh (FireRed / Silero) trên luồng PCM trong vùng tìm kiếm để xác định các nhịp lặng (silence gap $\ge 250\text{ms}$).
   - Điểm cắt $T_{\text{cut}}$ được chọn là điểm giữa của khoảng lặng dài nhất trong vùng tìm kiếm.
   - *Fallback an toàn*: Nếu người nói liên tục không ngừng nghỉ trong suốt 18s (không tìm thấy khoảng lặng $>250\text{ms}$), chọn điểm có năng lượng RMS nhỏ nhất hoặc cưỡng bức ngắt ở $T_{\text{start}} + 20.0\text{s}$ kèm 1.0s overlap (chồng lấn) cho khối sau.

---

### 2.2. Rào Cản 2: Phân Câu & Gắn Mốc Thời Gian Chuẩn Xác (ASR Segmentation & Alignment)

**Vấn đề**:
- Qwen3-ASR không trả native timestamps.
- Nếu chỉ chia câu theo dấu câu rồi nội suy tuyến tính theo độ dài ký tự thì với các ngôn ngữ có nhịp nói không đều (tiếng Nhật, tiếng Đức, tiếng Anh nhấn trọng âm), sai số có thể lên tới 300ms – 600ms, khiến phụ đề bị lệch khẩu hình và TTS đọc sai nhịp.

**Giải pháp**: **Pipeline 3 Tầng: Qwen3-ASR (Text) $\rightarrow$ Qwen3-ForcedAligner (Word Timestamps) $\rightarrow$ Punctuation Grouping**

#### Tầng 1: ASR Decoding trọn khối qua `transcribe.cpp`
- Nạp khối 15s vào Qwen3-ASR:
  $$\text{transcript\_text} = \text{Qwen3\_ASR}(\text{pcm\_15s})$$
- Kết quả thu được: Văn bản đầy đủ ngữ cảnh, chính xác 100% ngữ âm, có dấu câu rõ ràng (`.`, `,`, `?`, `!`, `。`, `、`).

#### Tầng 2: Căn chỉnh từ/ký tự qua `Qwen3-ForcedAligner-0.6B`
- Gọi forced aligner cho toàn bộ khối 15s và `transcript_text`:
  $$\text{aligned\_items} = \text{aligner.align}(\text{audio}=\text{pcm\_15s}, \text{text}=\text{transcript\_text}, \text{language}=\text{lang})$$
- Mỗi phần tử trong `aligned_items`:
  ```python
  ForcedAlignItem(text="Welcome", start_time=0.32, end_time=0.74)
  ForcedAlignItem(text="back", start_time=0.76, end_time=0.98)
  ForcedAlignItem(text="everyone.", start_time=1.02, end_time=1.56)
  ```
- Thời gian chạy trên GPU: **~35ms** (gần như không ảnh hưởng tới latency tổng thể). Sai số căn chỉnh: **$\pm 30\text{ms}$**.

#### Tầng 3: Gom từ thành câu phụ đề chuẩn (Subtitle Segment Reconstructor)
- Thuật toán gom câu dựa trên 2 điều kiện:
  1. **Ngắt theo dấu câu**: Khi gặp từ kết thúc bằng dấu chấm (`.`, `。`), chấm hỏi (`?`, `？`), chấm than (`!`, `！`) hoặc dấu phẩy (`.`, `、`) nếu câu đã đủ dài ($> 6$ từ).
  2. **Trần ký tự và thời lượng**: Nếu một câu không có dấu câu nhưng thời lượng kéo dài $> 5.0\text{s}$ hoặc $> 60$ ký tự $\rightarrow$ Ngắt tại khoảng trống từ gần nhất có khoảng cách âm học lớn nhất giữa 2 từ liên tiếp (`t0_{i+1} - t1_i > 150\text{ms}`).
- Mốc thời gian tuyệt đối cho từng câu phụ đề:
  $$\text{pts\_start} = PTS_{\text{base}} + \text{words}[0].\text{start\_time} + \Delta_{\text{sync}}$$
  $$\text{pts\_end} = PTS_{\text{base}} + \text{words}[-1].\text{end\_time} + \Delta_{\text{sync}}$$

---

### 2.3. Rào Cản 3: Dịch Thuật Đa Ngữ Cảnh (Bi-directional Sliding Context Translation)

**Vấn đề**:
- Dịch cả đoạn 15s: Chất lượng dịch rất hay, nhưng cấu trúc ngữ pháp giữa ngôn ngữ nguồn (tiếng Nhật SOV, tiếng Đức đảo ngữ) và tiếng Việt (SVO) làm trật tự câu bị xáo trộn $\rightarrow$ Không thể gán bản dịch trở lại từng mốc thời gian phụ đề con.
- Dịch từng câu con độc lập: Phụ đề khớp thời gian nhưng mất ngữ cảnh, xưng hô lung tung, tối nghĩa.

**Giải pháp**: **Dịch từng câu con nhưng Bơm Ngữ Cảnh 2 Chiều (Bi-directional Context)**
Mỗi câu phụ đề $S_i$ sau khi gom từ Bước 2C được dịch độc lập để duy trì ánh xạ $1:1$ với $[\text{pts\_start}_i, \text{pts\_end}_i]$, nhưng câu lệnh Prompt gửi tới LLM GGUF sẽ được đóng gói đầy đủ ngữ cảnh xung quanh:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ PROMPT STRATEGY (LOOKAHEAD CONTEXT MODE)                               │
├────────────────────────────────────────────────────────────────────────┤
│ SYSTEM:                                                                │
│ Bạn là chuyên gia biên dịch phụ đề video. Nhiệm vụ của bạn là dịch     │
│ [CÂU MỤC TIÊU] sang tiếng Việt.                                        │
│ - Phải giữ xưng hô, đại từ và thuật ngữ nhất quán với [NGỮ CẢNH TRƯỚC].│
│ - Tham khảo [NGỮ CẢNH TIẾP THEO] để hiểu đúng nghĩa của câu lấp lửng.   │
│ - CHỈ XUẤT DUY NHẤT câu dịch của [CÂU MỤC TIÊU], không giải thích thêm.│
│                                                                        │
│ USER:                                                                  │
│ [NGỮ CẢNH TRƯỚC]:                                                      │
│ - Gốc: "The doctor entered the room." -> Dịch: "Bác sĩ bước vào phòng."│
│ - Gốc: "The patient looked nervous." -> Dịch: "Bệnh nhân trông lo lắng."│
│                                                                        │
│ [NGỮ CẢNH TIẾP THEO]:                                                  │
│ - "We must begin the surgery now."                                     │
│                                                                        │
│ [CÂU MỤC TIÊU]:                                                        │
│ "He picked up the scalpel calmly."                                     │
│                                                                        │
│ ASSISTANT:                                                             │
│ Ông bình tĩnh cầm dao mổ lên.                                          │
└────────────────────────────────────────────────────────────────────────┘
```

> **Các chốt chặn kiểm soát LLM**:
> 1. `max_tokens` được giới hạn dựa theo độ dài câu gốc ($\text{len}(\text{words}) \times 3 + 16$).
> 2. `stop_tokens`: `["\n", "<|im_end|>", "<end_of_turn>"]` để ngăn model "nói thêm".
> 3. Post-filter loại bỏ các tiền tố lặp như `"Bản dịch: "`, `"Dịch: "`.

---

## 3. Kiến Trúc Kỹ Thuật, Quản Lý VRAM & Bộ Nhớ

### 3.1. Phân Bổ VRAM Trên GPU (VRAM Budgeting)

Hệ thống chạy trên GPU NVIDIA với các thành phần cùng chia sẻ VRAM:

| Thành Phần | Công Nghệ / Model | Ước Tính VRAM | Ghi Chú |
|---|---|---|---|
| **ASR Engine** | Qwen3-ASR 1.7B (Q8_0 / Q4_K_M) qua `transcribe.cpp` | ~1.6 – 2.1 GB | Nạp cố định, Singleton (`_infer_lock`) |
| **Forced Aligner** | Qwen3-ForcedAligner-0.6B qua `PyTorch` (BF16 / FP16) | ~1.1 – 1.3 GB | NAR model, không cần KV-cache lớn |
| **Translation** | Hunyuan-MT2 1.8B / MiLM 2B qua `llama-cpp-python` | ~1.8 – 2.2 GB | GGUF CUDA, n_ctx=512 |
| **TTS Engine** | OmniVoice qua PyTorch CUDA | ~1.8 – 2.0 GB | Chạy theo nhu cầu, có yield GPU cho ASR |
| **Tổng Cộng Peak** | **Toàn bộ pipeline hoạt động** | **~6.5 – 7.6 GB** | **Vừa vặn trên card 8GB (RTX 3060/4060) và cực kỳ thoải mái trên 12GB/16GB** |

> **Quy tắc An Toàn Luồng (GPU Scheduling)**:
> 1. `Qwen3-ASR` và `Qwen3-ForcedAligner` chạy tuần tự trong cùng một lượt xử lý batch: ASR nhả text xong $\rightarrow$ Aligner chạy 35ms $\rightarrow$ Giải phóng lock GPU ngay cho LLM dịch.
> 2. LLM dịch tuần tự từng câu $\rightarrow$ Gửi phụ đề về Client.
> 3. TTS OmniVoice chỉ tổng hợp khi có slot GPU trống (`gpu_arbiter.admit()`), tự động kiểm tra `_free_vram_mb() > 800MB`.

---

### 3.2. Cơ Chế Xử Lý Khi Tua Video (Seek Protocol & Fast-Bootstrap)

Khi người dùng tua video tới mốc $T_{\text{target}}$:
- **Tua lùi vào vùng đã có trong RAM**:
  - Dữ liệu phụ đề và TTS của vùng này đã được tính toán từ trước $\rightarrow$ Gửi ngay `lookahead_subtitles` cho client, video tiếp tục phát tức thì (**độ trễ 0.0s**).
- **Tua tới vùng mới chưa giải mã (Cold Seek)**:
  - Nếu giải mã ngay 15s audio, người dùng có thể phải đợi 1.0s – 1.5s để model chạy xong $\rightarrow$ Gây cảm giác khựng video.
  - **Giải pháp Fast-Bootstrap (2 nhịp)**:
    - *Nhịp 1 (Bootstrap)*: Cắt một khối ngắn **3.0s – 4.0s** đầu tiên sau mốc tua. Chạy ASR + Aligner siêu tốc (< 150ms) để nhả ngay câu phụ đề đầu tiên $\rightarrow$ Video lập tức tiếp tục phát (`prebuffer_ready = True`).
    - *Nhịp 2 (Steady State)*: Các khối tiếp theo tự động chuyển về kích thước đầy đủ **12s – 18s** để tận dụng tối đa chất lượng offline.

---

## 4. Kế Hoạch Triển Khai 5 Giai Đoạn (Phases & DoD)

### 🔴 Phase 1: PoC Tích Hợp Qwen3-ASR + Qwen3-ForcedAligner-0.6B
- **Mục tiêu**: Xác thực luồng chạy độc lập: nạp 15s audio $\rightarrow$ ASR (transcribe.cpp) nhả text $\rightarrow$ ForcedAligner (PyTorch) nhả word timestamps $\rightarrow$ Punctuation grouping.
- **Nội dung công việc**:
  - [ ] **1.1.** Viết script kiểm thử độc lập [`backend/tools/test_qwen3_aligner_poc.py`](file:///d:/vibe-translation-addon-transcribe_cpp/backend/tools/test_qwen3_aligner_poc.py).
  - [ ] **1.2.** Đo thời gian suy luận (Latency) và VRAM của `Qwen3-ForcedAligner-0.6B` trên audio 15s.
  - [ ] **1.3.** Kiểm tra độ chính xác mốc thời gian trên 2 ngôn ngữ chính: Tiếng Anh và Tiếng Nhật (đảm bảo package `nagisa` hoạt động trơn tru).
  - [ ] **1.4.** Hiện thực hàm gom từ thành câu phụ đề `group_words_to_sentences(aligned_items, max_words=10, max_duration=4.5)`.
- **Tiêu chí nghiệm thu (DoD Phase 1)**:
  - ✅ `Qwen3-ForcedAligner-0.6B` chạy thành công trên GPU CUDA, thời gian gióng hàng cho 15s audio $< 60\text{ms}$.
  - ✅ Thuật toán gom câu tạo ra danh sách phụ đề có mốc `start_pts`, `end_pts` chuẩn xác, không bị đè timestamp.

---

### 🟢 Phase 2: Xây Dựng `LookaheadChunker` (VAD-assisted Silence Detection) [HOÀN THÀNH]
- **Mục tiêu**: Tự động tìm điểm cắt thông minh trên `ContinuousAudioTimeline` để không bao giờ cắt vào giữa từ.
- **Nội dung công việc**:
  - [x] **2.1.** Cài đặt module `backend/core/lookahead_chunker.py`.
  - [x] **2.2.** Tích hợp với `VADProcessor` để quét nhanh nhãn tiếng nói trên RAM timeline.
  - [x] **2.3.** Viết unit test cho các kịch bản:
    - Audio có nhiều khoảng lặng tự nhiên.
    - Audio nói liên tục không ngừng (xử lý fallback overlap).
    - Audio ở đầu và cuối video (`test_56_lookahead_chunker.py`).
- **Tiêu chí nghiệm thu (DoD Phase 2)**:
  - ✅ Điểm cắt luôn rơi vào vùng im lặng (RMS $< 0.01$ hoặc VAD probability $< 0.3$).
  - ✅ Thời gian quét ranh giới VAD cho 18s audio trong RAM $< 20\text{ms}$.

---

### 🟢 Phase 3: Nâng Cấp Engine Dịch Thuật Đa Ngữ Cảnh (Bi-directional Context) [HOÀN THÀNH]
- **Mục tiêu**: Bơm ngữ cảnh câu trước và câu sau vào prompt dịch của `GGUFTranslator`.
- **Nội dung công việc**:
  - [x] **3.1.** Cập nhật `PromptStrategy` và `lookahead_handler.py` hỗ trợ format Bi-directional context (2 câu trước + 1 câu kế tiếp).
  - [x] **3.2.** Cung cấp context cho `_translate()` và `translate_sentence()` trong engine dịch.
  - [x] **3.3.** Viết test xác thực truyền context và không vỡ định dạng subtitle (`test_58_lookahead_offline_batch.py`).
- **Tiêu chí nghiệm thu (DoD Phase 3)**:
  - ✅ 100% câu dịch trả về đúng cấu trúc 1 dòng, không bị lặp token hoặc vỡ định dạng.
  - ✅ Thời gian dịch mỗi câu trên GGUF CUDA không tăng quá 15% so với dịch không context.

---

### 🟢 Phase 4: Tích Hợp Vào `lookahead_handler.py` & Cơ Chế Fast-Bootstrap (Dual-Path) [HOÀN THÀNH]
- **Mục tiêu**: Kết nối hoàn chỉnh chu trình: `Timeline` $\rightarrow$ `Chunker` $\rightarrow$ `Qwen3 ASR` $\rightarrow$ `ForcedAligner` $\rightarrow$ `Context Translate` $\rightarrow$ `WS Subtitle Dispatch`.
- **Nội dung công việc**:
  - [x] **4.1.** Thêm cờ cấu hình `processing_mode: str = "streaming"` ("streaming" | "offline_batch") trong `LookaheadConfig` ([`backend/config.py`](file:///d:/vibe-translation-addon-transcribe_cpp/backend/config.py)).
  - [x] **4.2.** Viết luồng `_offline_batch_loop()` trong `lookahead_handler.py`.
  - [x] **4.3.** Cài đặt cơ chế Fast-Bootstrap khi nhận bản tin `seek_reset` / `handle_seek`.
  - [x] **4.4.** Nối kết quả đầu ra với hàng đợi `tts_queue` của OmniVoice.
  - [x] **4.5.** Bộ integration test `test_58_lookahead_offline_batch.py` pass 100%.
- **Tiêu chí nghiệm thu (DoD Phase 4)**:
  - ✅ Bảo toàn Dual-Path an toàn: mode mặc định `"streaming"` không bị ảnh hưởng (pass 546/546 pytest).
  - ✅ Khi bật `"offline_batch"`, luồng cắt khối, gióng hàng và dịch đa ngữ cảnh chạy mượt mà.
  - ✅ Khi tua video (Seek), cơ chế Fast-Bootstrap chuyển `_batch_from_pts` và kích hoạt cắt khối 3-4s tức thì.

---

### 🟢 Phase 5: Đo Kiểm Toàn Diện, Benchmark & Nghiệm Thu
- **Mục tiêu**: Chứng minh định lượng sự vượt trội của kiến trúc mới về CER, WER, độ chính xác timestamp và độ ổn định hệ thống.
- **Nội dung công việc**:
  - [ ] **5.1.** Chạy benchmark trên bộ test tiếng Anh và tiếng Nhật:
    - So sánh WER/CER giữa Pipeline B v2 (streaming giả lập) và Pipeline B v3 (offline batch + forced aligner).
    - So sánh độ lệch timestamp (Alignment Drift) so với ground-truth.
  - [ ] **5.2.** Kiểm tra tải bộ nhớ và VRAM:
    - Đảm bảo không xảy ra hiện tượng leak RAM hoặc tràn VRAM khi xem liên tục 1 giờ video.
  - [ ] **5.3.** Đảm bảo toàn bộ test suite hiện có của backend (`pytest`) đều pass 100%.
- **Tiêu chí nghiệm thu (DoD Phase 5)**:
  - ✅ WER/CER của chế độ Offline Batch giảm tối thiểu 15% – 30% so với chế độ streaming giả lập.
  - ✅ Độ lệch timestamp trung bình $\le 50\text{ms}$.
  - ✅ Không vi phạm bất kỳ Invariant nào trong [`AGENTS.md`](file:///d:/vibe-translation-addon-transcribe_cpp/AGENTS.md).

---

## 5. Ma Trận Đánh Giá Rủi Ro & Kế Hoạch Phòng Ngừa

| Rủi Ro Kỹ Thuật | Tác Động | Biện Pháp Phòng Ngừa & Fallback |
| :--- | :--- | :--- |
| **`Qwen3-ForcedAligner-0.6B` bị lỗi hoặc quá tải VRAM** | Không có mốc thời gian từng từ | Tự động fallback sang bộ phân câu theo dấu câu của ASR + nội suy theo năng lượng âm thanh VAD. |
| **Tua video liên tục gây nghẽn hàng đợi ASR/Aligner** | Giật lag video sau khi tua | Huỷ ngay tác vụ đang giải mã của `seek_id` cũ qua cancellation token / abort flag; áp dụng Fast-Bootstrap (3s) cho vị trí mới. |
| **LLM dịch nuốt câu hoặc gộp nhiều câu làm một** | Lệch số lượng câu với ASR | Thiết lập `max_tokens` chặt chẽ cho từng câu; nếu kết quả dịch rỗng hoặc bất thường, fallback về dịch câu đơn lẻ không context. |
| **Xung đột VRAM giữa ASR, Aligner, Translate và TTS** | OOM / Crash tiến trình | Tuân thủ nghiêm ngặt `gpu_arbiter`: ASR + Aligner chạy xong giải phóng lock GPU rồi mới nhường cho LLM Dịch và TTS; kiểm tra `_free_vram_mb()` trước khi tổng hợp. |

---

## 6. Kết Luận

Bằng việc kết hợp sức mạnh nhận dạng offline vượt trội của **Qwen3-ASR** với khả năng gióng hàng thời gian mili-giây của **Qwen3-ForcedAligner-0.6B**, kiến trúc Pipeline B v3 giải quyết trọn vẹn điểm nghẽn thiếu timestamp của Qwen3 mà không phải thỏa hiệp chất lượng hay dùng các biện pháp phỏng đoán thô sơ. Hệ thống sẽ đạt tới tiêu chuẩn vàng: **nhận diện chuẩn xác như offline, timestamp chính xác từng từ, dịch tự nhiên theo ngữ cảnh điện ảnh, và độ trễ hiển thị 0.0s tuyệt đối**.
