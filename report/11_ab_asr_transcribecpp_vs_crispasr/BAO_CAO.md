# A/B: Qwen3-ASR-1.7B — `transcribe.cpp` vs `CrispASR`

**Ngày:** 2026-02 (đo trên máy dev)
**Câu hỏi:** có nên thay `transcribe.cpp` bằng CrispASR cho tầng ASR, để đồng bộ runtime với forced-aligner?
**Trả lời:** **KHÔNG.** Giữ `transcribe.cpp`. CrispASR thua ở cả tốc độ, độ chính xác, và có lỗi làm hỏng ký tự tiếng Trung.

---

## 1. Điều kiện đo

| Yếu tố | Giá trị |
|---|---|
| Model | Qwen3-ASR-1.7B, lượng tử hoá Q8_0, cùng máy |
| transcribe.cpp | `backend/models/Qwen3-ASR-1.7B-Q8_0.gguf` (2083.8 MB) |
| CrispASR | `backend/models/CrispASR-qwen3-asr-1.7b-q8_0.gguf` (2390.6 MB) |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB, CUDA 13.2 |
| Audio | 16 kHz mono PCM; VAD đã tách sẵn, cùng file cho cả hai |
| Đo | mỗi file chạy 2 lần, lấy **min** (đã warm-up) |
| Script | `.research/asr_ab_transcribecpp.py`, `.research/asr_ab_crispasr.py`, `.research/asr_ab_compare.py` |

Lưu ý: CrispASR chỉ đọc được GGUF do `cstr/*` convert. GGUF `handy-computer/*` sẵn có **không đọc được** (thiếu `audio.blk.N.attn_q.bias`, `token_embd.weight`, `output.weight`) → phải tải thêm bộ thứ hai, tốn thêm ~2.4 GB dung lượng và buộc đổi nguồn model.

---

## 2. Tốc độ (warm, min của 2 lần)

| File | transcribe.cpp | RTF | CrispASR | RTF | CrispASR chậm hơn |
|---|---:|---:|---:|---:|---:|
| Japanese_5s | 164 ms | 0.0322 | 305 ms | 0.0600 | 1.86× |
| Chinese_fast_speed_11s | 595 ms | 0.0522 | 1202 ms | 0.1056 | 2.02× |
| Chinese_noise_28s | 882 ms | 0.0312 | 1859 ms | 0.0658 | 2.11× |
| English_low_speech_quality_19s | 348 ms | 0.0183 | 778 ms | 0.0409 | 2.24× |
| Russian_4s | 222 ms | 0.0467 | 412 ms | 0.0866 | 1.86× |
| **Tổng** | **2211 ms** | | **4556 ms** | | **2.06×** |

CrispASR chỉ thắng ở **thời gian nạp model**: 1444 ms vs 2160 ms (nhanh hơn 1.5×). Khoản này chỉ trả một lần khi khởi động, không bù được phần suy luận chậm gấp đôi.

---

## 3. Độ chính xác (CER cho CJK, WER cho phần còn lại — càng thấp càng tốt)

| File | transcribe.cpp | CrispASR |
|---|---:|---:|
| Japanese_5s | 0.00% | 0.00% |
| Chinese_fast_speed_11s | **8.46%** | 10.77% |
| Chinese_noise_28s | **0.00%** | 4.92% |
| English_low_speech_quality_19s | 19.35% | 19.35% |
| Russian_4s | 0.00% | 0.00% |
| **Trung bình** | **5.56%** | 7.01% |

Chênh lệch không chỉ ở điểm số: trên `Chinese_noise_28s` transcribe.cpp ra **văn bản tiếng Trung sạch và đầy đủ (0.00%)**, còn CrispASR vừa sai chữ vừa mất nội dung.

---

## 4. Lỗi chặn: `segment_text` của CrispASR ghi đè byte giữa ký tự CJK bằng 0x20

Đây là phát hiện quyết định. Qua C ABI `crispasr_session_result_segment_text`, CrispASR trả về **chuỗi UTF-8 không hợp lệ**:

| File | Byte hỏng | Bị ghi đè bằng `0x20` | UTF-8 |
|---|---:|---:|---|
| Chinese_noise_28s | 9 | 9 | **KHÔNG hợp lệ** |
| Chinese_fast_speed_11s | 2 | 2 | **KHÔNG hợp lệ** |
| Japanese_5s | 0 | 0 | hợp lệ |

Ví dụ thật (`Chinese_noise_28s.wav`):

```
segment_text : 母亲� 为发生交通事故      敲了� 下杯子      � 这小子
word_text    : 母亲因为发生交通事故      敲了几下杯子      你这小子
```

Cùng một lần suy luận, cùng một segment — chỉ khác hàm truy vấn. **Lặp lại y hệt qua 3 lần chạy.**

### Cơ chế (đã khoanh vùng tới từng byte)

So sánh byte-đối-byte giữa `segment_text` và `word_text`:

```
seg  ... \xe6\xaf\x8d\xe4\xba\xb2  \xe5\x9b         \x20  \xe4\xb8\xba ...
word ... \xe6\xaf\x8d\xe4\xba\xb2  \xe5\x9b\xa0           \xe4\xb8\xba ...
                                    └── 因 (U+56E0)
```

`因` = `E5 9B A0`. Ở `segment_text`, **byte giữa `A0` bị thay bằng `0x20` (dấu cách)**, byte đầu và byte cuối còn nguyên.

- Độ dài byte **không đổi** (606 = 606) → không thể phát hiện bằng so kích thước.
- 100% các byte lệch đều là `0x20` (9/9 và 2/2).
- Chỉ xảy ra khi BPE byte-level tách một ký tự đa byte qua ranh giới token; ký tự đó mất hẳn và UTF-8 vỡ.

`word_text` (và do đó đường `crispasr_align_words_abi` mà aligner đang dùng) **không dính lỗi** — khớp với việc `test_64` cho văn bản ZH giống torch 10/10.

### Rủi ro cho tiếng Nhật

`Japanese_5s.wav` sạch, nhưng **không** kết luận được tiếng Nhật miễn nhiễm: tiếng Nhật cũng là CJK 3 byte nên cùng lớp lỗi, chỉ là file mẫu này không rơi vào ca bị tách token. Lỗi phụ thuộc nội dung, không phụ thuộc ngôn ngữ.

### Tương quan upstream

Cùng một lớp lỗi CrispASR đã sửa 2 lần:
- **#475** — byte-level BPE tách ký tự làm `-ojf` ra UTF-8 hỏng; bản sửa nối lại ký tự cho mảng `words` nhưng **không** cho `segment text` qua C ABI.
- **#117 / #118 / #124** — mojibake byte-fallback của Cohere.

Ta đang ở **v0.8.41, mới hơn cả hai bản sửa**, mà đường C ABI `segment_text` vẫn còn lỗi → biểu hiện chưa được sửa. Có repro tối thiểu và deterministic, nên **đáng báo upstream**.

---

## 5. Lỗi phụ

- **Crash thay vì trả lỗi:** nạp GGUF không tương thích (`handy-computer/*`) làm `crispasr_session_open` **access violation** — sập tiến trình thay vì trả `NULL`. Với app phát trực tiếp, một model sai định dạng là sập cả tiến trình ASR.
- **Cảnh báo CUDA:** `ggml_cuda_init: CUDA minor version mismatch — compiled 13.0, runtime 13.2`.

---

## 6. Kết luận

Giữ **`transcribe.cpp`** cho ASR.

| Tiêu chí | Ai thắng |
|---|---|
| Tốc độ suy luận | **transcribe.cpp** (2.06×) |
| Độ chính xác | **transcribe.cpp** (5.56% vs 7.01%) |
| Toàn vẹn văn bản CJK | **transcribe.cpp** (0.00% CER, không hỏng ký tự) |
| Tương thích GGUF sẵn có | **transcribe.cpp** |
| Chịu lỗi khi model sai | **transcribe.cpp** |
| Thời gian nạp model | CrispASR (1444 vs 2160 ms) |

Lợi ích duy nhất của việc hợp nhất runtime là bớt một bộ DLL — không đáng đổi lấy tốc độ chậm gấp đôi và văn bản tiếng Trung hỏng. Aligner đã chạy CrispASR và **không** dính lỗi này (đi qua `word_text`), nên giữ nguyên: hai runtime, mỗi cái ở việc nó làm tốt hơn.

**Điều kiện để xem lại:** nếu upstream sửa `segment_text` qua C ABI và ta đo lại thấy ngang tốc độ. Ngoài ra `cstr/qwen3-ja-anime` (Qwen3-ASR-1.7B fine-tune cho anime/galgame, Apache-2.0, ~1.3 GB) là lợi thế thật sự của hệ CrispASR cho repo Nhật-focused này — đáng đánh giá riêng, độc lập với quyết định runtime.
