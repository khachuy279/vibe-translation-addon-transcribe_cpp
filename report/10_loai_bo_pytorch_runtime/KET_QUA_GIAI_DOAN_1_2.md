# Kết quả thực thi Giai đoạn 1 & 2 — Loại bỏ PyTorch runtime

- **Ngày**: 2026-10-06
- **Máy đo**: RTX 5060 Ti 16 GB, Windows 11, Python 3.13.14, onnxruntime 1.30.0, driver hỗ trợ CUDA 13
- **Phạm vi**: VAD (Silero + FireRed) → onnxruntime; ForcedAligner → Qwen3-ForcedAligner-0.6B GGUF (CrispASR)
- **Trạng thái**: **Giai đoạn 1 XONG + kiểm chứng; Giai đoạn 2 XONG + đã chốt mặc định** (đường torch trở thành dự phòng)

---

## 1. Giai đoạn 1 — VAD sang ONNX (XONG)

### 1.1. Thay đổi

| File | Việc |
|---|---|
| `backend/vad/silero_onnx.py` **(mới)** | Model ONNX + **port chính xác** `silero_vad.VADIterator` (giữ nguyên cả hằng số trễ `threshold - 0.15` mà upstream hardcode). Có `snapshot()`/`restore()` state. |
| `backend/vad/engines/firered_onnx.py` **(mới)** | Vendor `KaldifeatFbank` + `CMVN` + `StreamVadPostprocessor` từ `fireredvad` (Apache-2.0, ghi rõ nguồn); chạy `fireredvad_stream_vad_with_cache.onnx`. |
| `backend/vad/engines/silero.py` | Viết lại sang ONNX. |
| `backend/vad/engines/firered.py` | Viết lại sang ONNX. |
| `backend/core/vad_silence.py` | `VADSilenceScanner` sang ONNX (bỏ `torch.no_grad()` + tensor). |
| `backend/vad/engines/fsmn.py` | Thành engine **tuỳ chọn** (`available()` + lỗi có hướng dẫn); không tải model khi thiếu `funasr`. |
| `backend/vad/base.py`, `backend/vad/engines/__init__.py` | Thêm `BaseVADEngine.available()`; `prewarm_engines` báo `skipped` thay vì lỗi mỗi lần khởi động. |
| `backend/main.py` | Pre-flight + kiểm tra model đổi sang file ONNX. |
| `backend/requirements.txt` | Bỏ `fireredvad`, `silero-vad`; bỏ gói `omnivoice`; `funasr` thành tuỳ chọn; giữ `torch`/`torchaudio` **có ghi chú** vì Giai đoạn 2 chưa bật mặc định. |
| `README.md` | Bảng VAD engine + mục model + mục ForcedAligner 2 runtime. |

### 1.2. Kiểm chứng tương đương (đo thật)

`backend/tests/test_62_vad_onnx_parity.py` — so **từng frame** với đường torch cũ:

| Engine | Số frame | `max\|Δp\|` | Frame lệch quyết định @0.30/0.40/0.50 |
|---|---|---|---|
| Silero (3 file) | 3 797 | ≤ 2e-6 | **0 / 0 / 0** |
| FireRed (4 file) | 12 621 | ≤ 1e-6 | **0 / 0 / 0** |

Ngoài ra `test_vad_onnx_khong_nap_torch` chạy trong **tiến trình con sạch** và khẳng định
`torch`/`transformers` không hề xuất hiện trong `sys.modules` sau khi nạp + chạy cả hai engine.

### 1.3. Toàn vẹn âm thanh — yêu cầu trọng tâm (ĐẠT)

`backend/tests/test_63_vad_audio_integrity.py` — 14 test, 2 engine × 3 cấu hình ngưỡng/silence:

1. **Engine không sửa buffer đầu vào** — frame chứa "mẫu bẫy" (các giá trị KHÔNG sống sót qua vòng
   `int16 → float32 → int16`: `1, 3, 7, 32767, -32768, -32767`…) phải nguyên từng bit sau
   `is_speech()`. Buffer đọc từ `bytes` là read-only nên mọi ghi tại chỗ sẽ ném lỗi.
2. **Mọi frame chuyển tiếp khớp ĐÚNG mốc thời gian của nó** — engine báo `frame_ts` là thời điểm
   của mẫu đầu tiên; test khẳng định `bytes` chuyển tiếp == PCM gốc tại **đúng** mốc đó. Đây là
   bằng chứng loại trừ đồng thời **lệch pha**, đổi thứ tự, chèn/mất mẫu, gain/clip, và **lượng tử
   hoá lại** (resample sẽ làm mốc lệch ngay).
3. **Không lệch pha** — dịch khối chuyển tiếp ±1 mẫu phải cho nội dung KHÁC.
4. **Chuỗi liền mạch** — từng chunk (kể cả pre-roll xả lại khi START) phải nối tiếp nhau **không
   hụt, không chồng**; SHA-256 của toàn bộ phần chuyển tiếp bằng SHA-256 của đúng đoạn PCM gốc.

> **Kết luận**: chất lượng tín hiệu tới ASR **không thể suy giảm**, vì đó là cùng một chuỗi byte.

### 1.4. Lợi ích đo được

- ONNX Silero nhanh hơn JIT: 88,2 s audio: **0,38 s** (ONNX) vs 1,01 s (JIT); 5,1 s: **0,06 s** vs 1,55 s.
- FireRed ONNX ~12× thời gian thực (bản torch cũ đo 0,31× theo tài liệu trong `vad_silence.py`).
- Bỏ được 2 gói `import torch` ở **cấp module** ⇒ tiến trình backend không còn bị kéo PyTorch vì VAD.

---

## 2. Giai đoạn 2 — ForcedAligner sang GGUF CrispASR

### 2.1. Thay đổi

| File | Việc |
|---|---|
| `backend/utils/crispasr_native.py` **(mới)** | Binding ctypes ABI `crispasr_align_words_abi` + **vòng lặp tiến trình con**. Đặt ở `utils/` **có chủ đích** (xem 2.3). |
| `backend/asr/crispasr_aligner.py` **(mới)** | Client singleton: spawn worker, IPC `multiprocessing.connection`, prewarm/free/shutdown, `_sanitized_env()`. |
| `backend/asr/forced_aligner.py` | Thêm nhánh `_use_crispasr()` / `_align_crispasr()`; **bỏ `import torch` ở cấp module** + làm LAZY `transformers`/`qwen_asr`/`torch.cuda.*` ⇒ đường mặc định không kéo PyTorch nữa. |
| `backend/config.py` | Thêm `ForcedAlignerConfig`; **`backend = "crispasr"` là mặc định** (đường torch thành dự phòng). |
| `backend/main.py` | Pre-flight kiểm tra ĐÚNG runtime đang chọn (GGUF khi `crispasr`, safetensors khi `torch`); không chặn khởi động khi thiếu GGUF vì có dự phòng. |
| `backend/bin/crispasr/` **(mới, 167 MB)** | `crispasr.dll` + `ggml.dll` + `ggml-base.dll` + `ggml-cpu.dll` + `ggml-cuda.dll` (CUDA 13). |
| `backend/tests/test_64_crispasr_aligner.py` **(mới)** | 7 test tầng A + 5 test tầng B (gồm 2 test chạy trong **tiến trình con sạch** để chứng minh không nạp torch). |
| `backend/tests/test_57_forced_aligner_service.py` | Ghim `backend="torch"` cho test stub đường torch (giữ nguyên ý nghĩa test). |

### 2.1b. Bằng chứng "đã THAY THẾ đường torch" (không chỉ thêm đường mới)

Test chạy trong **tiến trình Python sạch**:

```
import backend.asr.forced_aligner   →  torch trong sys.modules? False
sau khi gọi align() thật            →  torch? False | transformers? False
torch aligner đã nạp chưa?             False
```

Tức là với mặc định mới, cả **import** lẫn **suy luận** của aligner đều **không dùng PyTorch**.
Trước thay đổi này, chỉ cần `import` file là `torch` (~2,7 GB đĩa, 1–2 s khởi động) đã bị nạp.

### 2.2. Chọn build nào — đo thật

| Build CrispASR | DLL phải mang | Khối 28,26 s (warm) | RTF |
|---|---|---|---|
| CPU (`libcrispasr-windows-x86_64`) | 56 MB | 3 956 ms | 0,140 |
| Vulkan (`+vulkan` wheel) | 87 MB | ~8 200 ms | **1,62 — không dùng được** |
| CUDA 12 (wheel `+cuda`) | 988 MB (kèm cuBLAS 12) | 194 ms | 0,0069 |
| **CUDA 13 `-cuda13-non-cuda` + toolkit trong repo** | **167 MB** | **195 ms** | **0,0069** |

⇒ Chọn bản cuối: **tốc độ y hệt bản CUDA 12 nhưng nhỏ hơn 6 lần**, vì dùng đúng
`cudart64_13.dll`/`cublas64_13.dll`/`cublasLt64_13.dll` mà repo **đã có** trong
`external/cuda-toolkit` (cùng bộ `setup_cuda_dll_paths()` đang đăng ký cho ASR/llama.cpp).

Qua client + worker (có IPC): khối 28,26 s = **307 ms (RTF 0,0109)**, tiếng Nhật 5,08 s = 94 ms,
tiếng Anh 19 s = 188 ms. Nạp model lần đầu 127–839 ms (torch: 1,6 s nạp + 0,5 s prewarm).

### 2.3. ⚠️ Phát hiện quan trọng: xung đột `ggml*.dll` là THẬT và đã tái hiện được

Dự đoán trong báo cáo khảo sát đã **xảy ra thật** khi triển khai:

```
OSError: [WinError 127] The specified procedure could not be found
```

**Nguyên nhân**: `backend/asr/__init__.py:14-16` gọi `bootstrap()` **ngay khi import**, mà
`bootstrap()` nạp `transcribe.dll` ⇒ kéo `backend/bin/ggml*.dll` vào tiến trình. Sau đó
`crispasr.dll` phân giải `ggml-base.dll` theo **tên module** và bind vào bản của transcribe.cpp ⇒
lệch ABI.

**Bằng chứng bisect** (`.research/bisect_dll.py`): nạp `crispasr.dll` trong tiến trình **sạch**
thành công với **mọi** tổ hợp thư mục (kể cả có/không CUDA runtime); chỉ hỏng khi tiến trình đã
nạp bộ ggml của transcribe.cpp.

**Cách xử lý** (đã áp dụng):
1. Toàn bộ phần native/worker đặt ở `backend/utils/crispasr_native.py` — **không** import
   `backend.asr` ⇒ tiến trình con không bao giờ nạp ggml của transcribe.cpp.
2. `_sanitized_env()` loại mọi thư mục chứa `ggml*.dll` khác khỏi `PATH` của tiến trình con.
3. Worker **kiểm chứng bind** bằng `GetModuleFileNameW("ggml-base.dll")` và **ném lỗi ngay** nếu
   DLL được nạp từ thư mục khác.
4. Giữ handle `os.add_dll_directory` (bỏ qua giá trị trả về ⇒ object bị GC ⇒ thư mục biến mất
   khỏi search path).

### 2.4. Độ chính xác — A/B ở tầng PHỤ ĐỀ

Chỉ so mốc từng đơn vị là chưa đủ. Đo trên `group_words_to_subtitles` + `split_oversized_sentences`:

| Audio | Đơn vị torch / GGUF | **Số phụ đề** | Nội dung phụ đề |
|---|---|---|---|
| `Japanese_5s.wav` | 14 từ (nagisa) / 25 ký tự | **1 / 1** | **giống hệt**, mốc đầu giống (0,48), cuối lệch 80 ms |
| `Chinese_noise_28s.wav` | 183 / 183 | **10 / 10** | **giống hệt**, chỉ 1 biên lệch 80 ms |
| `English_low_speech_quality_19s.wav` | 31 / 31 | **7 / 7** | **giống hệt** |

Kiểm chứng cơ chế: mốc theo KÝ TỰ của GGUF **phân rã khớp hoàn toàn** mốc theo TỪ của torch cho
tiếng Nhật (`抜群 0.48–0.96` = `抜 0.48–0.64` + `群 0.64–0.96`).

**GGUF chính xác HƠN ở một chỗ**: đơn vị đầu khi có im lặng dài — torch kéo `Okay` từ `0.00 s`
(trùm lên 1,28 s im lặng), GGUF đặt `1.28 s` đúng chỗ tiếng nói bắt đầu.

### 2.5. Rủi ro còn lại

1. **Câu tiếng Nhật/Trung RẤT DÀI không có dấu kết câu**: trần `sentence_max_words = 24` đếm đơn
   vị, mà đơn vị CJK của GGUF là **ký tự** ⇒ có thể chẻ sớm hơn đường torch. Chưa đo được trên dữ
   liệu sạch (file `00_ingress_stream.wav` có transcript lẫn nhãn người nói nên không dùng làm
   mốc được). **Đề xuất**: nếu bật mặc định, thêm test hồi quy trên video tiếng Nhật thật.
2. **Im lặng đầu khối**: CrispASR ghi rõ nhạy với im lặng đầu; khối của repo cắt ở giữa khoảng
   lặng nên có tối đa ~0,75 s. Đo được là GGUF **tốt hơn**, nhưng nên theo dõi.
3. **Nhịp phát hành CrispASR rất nhanh** (0.8.x mỗi 1–3 ngày) ⇒ phải **ghim `v0.8.41`** và ghi lại
   hash DLL.
4. **Án lệ AVX-512** (issue #394/#374): bản Windows CUDA v0.8.29 từng chết `0xC000001D` trên CPU
   không có AVX-512. Đã sửa trên `main` sau v0.8.29; đã kiểm tra bản `v0.8.41` này **chạy được**
   trên máy đo — nhưng **nên test thêm trên CPU không AVX-512** trước khi phát hành.

---

## 3. Trạng thái & việc còn lại

### Đã xong
- [x] VAD Silero + FireRed chạy ONNX, **bit-tương-đương** với torch (0 frame lệch quyết định).
- [x] **Toàn vẹn âm thanh qua VAD được chứng minh** (byte-nguyên-bản, không lệch pha, không
      lượng tử hoá lại, liền mạch) — 14 test.
- [x] Aligner GGUF chạy trong tiến trình con cách ly, ngang tốc độ torch trên GPU.
- [x] **Chốt `backend = "crispasr"` làm mặc định**; đường torch thành dự phòng tự động.
- [x] **Import + suy luận của aligner KHÔNG nạp `torch`/`transformers`** (test tiến trình con sạch).
- [x] A/B tầng phụ đề: cấu trúc phụ đề giữ nguyên (Nhật/Trung/Anh).
- [x] 12 test mới cho aligner; 30 test VAD tầng B; toàn bộ test cũ xanh trừ 11 test `test_40`
      **đã hỏng từ trước** (catalog `models.yaml` không còn `whisper-large-v3-turbo`).

### Còn lại (Giai đoạn 3 — dọn torch)
- [ ] Xoá `backend/asr/qwen_asr/**` và `backend/models/Qwen3-ForcedAligner-0.6B/` (1,84 GB).
- [ ] Bỏ khối `import torch` ở `backend/main.py:40-46`; đổi `/health` sang khối `vad_runtime`.
- [ ] Viết lại `backend/utils/env_check.py` (đang hoàn toàn xoay quanh torch) + `test_45`.
- [ ] Bỏ `_configure_alloc_conf` + nhánh quét `torch/lib` trong `backend/utils/cuda.py`.
- [ ] Bỏ `torch`, `torchaudio`, `transformers`, `nagisa`, `soynlp` khỏi `requirements.txt` và dòng
      `--extra-index-url https://download.pytorch.org/whl/cu130`.
- [ ] Xoá `backend/utils/env_check.py::check_silero_onnxruntime` (không còn gói `silero-vad`).
- [ ] Quyết định có `git add backend/bin/crispasr/` (167 MB) như các bundle native khác hay không.

### Ghi chú vận hành
- Đường torch **vẫn còn nguyên** để làm dự phòng; khi nào Giai đoạn 3 xong mới xoá được hẳn.
- Muốn quay lại hành vi cũ ngay: đặt `forced_aligner.backend = "torch"`, không cần sửa mã.

---

## Phụ lục — Script đo tái lập (`.research/`, đã gitignore)

| Script | Việc |
|---|---|
| `vad_onnx_parity.py` | Silero JIT vs ONNX |
| `firered_onnx_parity.py` | FireRed torch vs ONNX |
| `bench_crispasr.py` | Benchmark 1 build CrispASR trong tiến trình sạch |
| `bench_cuda13.py` | Benchmark bản CUDA 13 `-non-cuda` + toolkit trong repo |
| `ab_align.py` | A/B mốc từng đơn vị (torch vs GGUF) |
| `ab_subtitles.py` | A/B **tầng phụ đề** (quyết định) |
| `bisect_dll.py` | Chứng minh nguyên nhân `WinError 127` |
| `poc_crispasr_align.py` | PoC ABI đầu tiên |
