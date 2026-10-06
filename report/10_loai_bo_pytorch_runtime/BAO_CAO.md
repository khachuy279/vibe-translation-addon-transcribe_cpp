# Báo cáo: Loại bỏ hoàn toàn phụ thuộc PyTorch runtime

- **Ngày khảo sát**: 2026-10-06
- **Phạm vi**: `backend/` (VAD, ForcedAligner, ASR, TTS, dịch) + `requirements.txt` + `external/FireRedVAD`
- **Phương pháp**: đọc mã nguồn thật + đo thực nghiệm trên venv hiện có (RTX 5060 Ti, Python 3.13.14)
- **Kết quả đo tái lập được**: `.research/vad_onnx_parity.py`, `.research/firered_onnx_parity.py`

---

## 0. Kết luận nhanh (TL;DR)

| Câu hỏi | Trả lời |
|---|---|
| Chỉ còn VAD + ForcedAligner phụ thuộc PyTorch? | **Đúng về mặt "bắt buộc"**, nhưng thiếu 3 điểm: (1) `main.py` vẫn `import torch` ở cấp module lúc khởi động; (2) `requirements.txt` vẫn kéo `torch`/`torchaudio` qua gói `omnivoice` **mà backend không hề `import omnivoice`**; (3) `transformers` cũng chỉ tồn tại vì aligner. |
| VAD nên chuyển qua ONNX? | **Nên, và đã kiểm chứng là an toàn tuyệt đối.** Silero VAD ONNX cho **cùng xác suất từng frame** với bản TorchScript (max\|Δp\| ≤ 2e-6, **0 frame lệch quyết định** trên 3.797 cửa sổ). FireRed VAD (engine **mặc định**) đã có **ONNX chính thức ngay trong repo** tại `external/FireRedVAD/pretrained_models/onnx_models/fireredvad_stream_vad_with_cache.onnx`, và cũng **bit-exact** với đường PyTorch (max\|Δp\| ≤ 1e-6, **0 frame lệch** trên 12.621 frame). |
| Qwen3-ForcedAligner GGUF của CrispASR có tương thích? | **Tương thích về chức năng — là bản port trung thành** (cùng prompt, cùng head 5000 lớp, cùng thuật toán sửa mốc LIS). **Nhưng có 1 khác biệt ngữ nghĩa quan trọng**: CrispASR tách đơn vị align theo **từng KÝ TỰ** cho CJK/Hangul, còn backend hiện tại dùng **từ của nagisa/soynlp** ⇒ tầng gom câu theo `max_words` sẽ cho phụ đề Nhật/Trung/Hàn **ngắn hơn hẳn** nếu không chỉnh lại. **Và 1 rủi ro hạ tầng nghiêm trọng**: CrispASR mang bộ `ggml*.dll` **thứ ba** vào tiến trình, trùng tên module với `transcribe.cpp` và `llama.cpp` — đúng loại xung đột mà `backend/utils/cuda.py` đã ghi nhận gây `0xc0000139`. |

---

## 1. Kiểm kê thực tế phụ thuộc PyTorch

### 1.1. Phụ thuộc CỨNG (không thể chạy nếu thiếu torch)

| # | Vị trí | Bản chất |
|---|---|---|
| 1 | `backend/asr/forced_aligner.py:25` | `import torch` cấp module; `:419-420` chọn device/dtype; `:426-430` `Qwen3ForcedAligner.from_pretrained` (transformers + torch); `:455,505` `torch.cuda.synchronize()`; `:1044` `empty_cache()`. Đây là **phụ thuộc torch lớn nhất**. |
| 2 | `backend/vad/engines/silero.py:202-205` | `torch.from_numpy` + `torch.no_grad()` cho model JIT. |
| 3 | `backend/vad/engines/fsmn.py:186-199` | `torch.from_numpy` truyền vào `funasr.AutoModel.generate`. |
| 4 | `backend/vad/engines/firered.py:144,164` | `from fireredvad import …` — gói này `import torch` **ngay ở cấp module** (`fireredvad/core/detect_model.py:5-7`, `audio_feat.py:10`, `stream_vad.py:8`). Kiểm chứng: chặn `sys.modules['torch']` ⇒ `import fireredvad` ném `ModuleNotFoundError`. **Đây là engine VAD MẶC ĐỊNH** (`backend/config.py:145` `vad_engine: str = "firered-vad"`). |
| 5 | `backend/core/vad_silence.py:190,203,317-355` | `VADSilenceScanner` — bộ dò khoảng lặng cắt khối cho Pipeline B — dùng thẳng model JIT Silero + `torch.no_grad()`. |
| 6 | `backend/asr/qwen_asr/**` (cây vendored, ~10 file) | Bản sao `qwen-asr` của QwenLM: `import torch`, `torch.nn`, `transformers`. Chỉ được dùng qua `forced_aligner.py:58-69`. |

### 1.2. Phụ thuộc MỀM / tuỳ chọn (đã guard, không chặn chạy)

| Vị trí | Ghi chú |
|---|---|
| `backend/main.py:40-46` | `try: import torch; torch.set_num_threads(2) except ImportError: pass` — **đã là tuỳ chọn**, nhưng vẫn khiến torch bị nạp vào tiến trình ngay lúc khởi động (tốn ~1-2 s + vài trăm MB RAM) khi nó còn được cài. |
| `backend/main.py:115-165` | `_log_torch_status_at_startup()` + khối `torch` trong `/health`. Chỉ để **phát hiện pip âm thầm hạ torch xuống bản CPU-only** — khi bỏ torch thì cả cơ chế này biến mất (đó là điều tốt). |
| `backend/asr/engine.py:496-498` | `torch.cuda.empty_cache()` trong `try` — lazy, chỉ chạy nếu torch đã được nạp. |
| `backend/translation/engine.py:460-462` | Tương tự. |
| `backend/ws/lookahead_handler.py:134-147` | `_empty_torch_cache()` đọc `sys.modules.get("torch")` — **không bao giờ tự import torch**. Đã đúng thiết kế. |
| `backend/utils/env_check.py:38-90,227` | `torch_status()` cho `env_check`/`/health`. |
| `backend/utils/cuda.py:57-80,120-151` | `_configure_alloc_conf()` (biến `PYTORCH_ALLOC_CONF`) + quét `site-packages/torch/lib` để tìm `cudart64_*.dll`. |
| `backend/tools/test_qwen3_aligner_poc.py`, `backend/tests/test_07_e2e_comparison.py:35` | Script đo/test cũ. |

### 1.3. Hai phát hiện làm thay đổi kết luận "chỉ còn VAD + ForcedAligner"

**(a) TTS KHÔNG cần torch.** `grep "import omnivoice"` trên toàn `backend/` = **0 kết quả**. TTS chạy qua `backend/bin/omnivoice/omnivoice.dll` (native), còn `backend/tts/audio_processor.py:43-44` còn cố tình duck-type để **không phải import torch**. Nhưng `requirements.txt:95-99` vẫn cài:

```
torch>=2.4
torchaudio>=2.4
omnivoice>=0.2.1
```

và metadata của `omnivoice 0.2.1` khai báo cứng `torch>=2.4`, `torchaudio>=2.4`, `transformers>=5.3.0`, `accelerate`, `gradio`, `librosa`, `webdataset`… ⇒ **một gói Python không được dùng đang là lý do chính khiến torch nằm trong môi trường.** Bỏ `omnivoice` khỏi requirements là thay đổi 1 dòng, không rủi ro.

**(b) `silero-vad` và `funasr` cũng kéo torch.** Metadata: `silero-vad 6.2.2` → `torch>=1.12.0`, `torchaudio<2.10,>=0.12.0`; `funasr 1.4.16` → `transformers`, `torch_complex`, `tensorboardX`. `onnxruntime>=1.17` đã có sẵn trong requirements (comment ở `requirements.txt:117-121` giải thích nó bị `silero_vad.utils_vad` import ở cấp module).

**Kết luận mục 1**: đúng, chỉ VAD + ForcedAligner cần torch **về mặt logic**; nhưng để gỡ được `pip uninstall torch` thì phải xử lý cả `omnivoice` (TTS, 1 dòng), `main.py:41` (optional), và toàn bộ chuỗi gói VAD.

---

## 2. VAD → ONNX: đã kiểm chứng bằng đo thực tế

### 2.1. Silero VAD — ONNX bit-tương-đương JIT

Model đã có sẵn trong gói: `.venv/Lib/site-packages/silero_vad/data/silero_vad.onnx` (2,33 MB) — cùng chỗ với `silero_vad.jit` (2,27 MB) mà repo đang copy ra `backend/models/`.

API streaming (đúng như `OnnxWrapper` của gói):

```
input  : float32 [batch, 576]      # 576 = 64 mẫu context + 512 mẫu cửa sổ mới
state  : float32 [2, batch, 128]   # h/c LSTM — caller giữ
sr     : int64   scalar            # 16000
output : float32 [batch, 1]        # xác suất tiếng nói của cửa sổ
stateN : float32 [2, batch, 128]   # state cho lần gọi kế tiếp
context = 64 mẫu cuối của input 576
```

**Kết quả đo** (`.research/vad_onnx_parity.py`, chạy tuần tự từng cửa sổ 512 mẫu):

| File | Số cửa sổ | max\|Δp\| | Frame lệch quyết định @0.30 / 0.40 / 0.50 | JIT | ONNX |
|---|---|---|---|---|---|
| `wav_test/Japanese_5s.wav` | 158 | 0,000001 | 0 / 0 / 0 | 1,55 s | **0,06 s** |
| `wav_test/English_multiple_kinds_of_noise_88s.wav` | 2.756 | 0,000002 | 0 / 0 / 0 | 1,01 s | **0,38 s** |
| `wav_test/Chinese_noise_28s.wav` | 883 | 0,000001 | 0 / 0 / 0 | 0,39 s | **0,15 s** |

⇒ **Không có sai khác hành vi**, chỉ nhanh hơn (2,6× – 25×) và bỏ được torch.

### 2.2. FireRed VAD (engine mặc định) — ONNX chính thức đã nằm trong repo

`external/FireRedVAD/pretrained_models/onnx_models/` (do upstream FireRedTeam phát hành, kèm `fireredvad/bin/export_onnx.py` để tự sinh lại):

| File | Vai trò |
|---|---|
| `fireredvad_stream_vad_with_cache.onnx` (2,31 MB) | **Streaming, cache nằm trong model** — `feat [1,T,80]` + `caches_in [8,1,128,19]` → `probs [1,T,1]` + `caches_out` |
| `fireredvad_stream_vad.onnx` | Streaming, cache trả ra ngoài thành 8 tensor `cache_out_0..7` |
| `fireredvad_vad.onnx` / `fireredvad_aed.onnx` | Bản không streaming (dùng cho quét lô) |
| `cmvn.ark` (1,3 KB) | Chuẩn hoá CMVN, đọc bằng `kaldiio` |

**Kết quả đo** (`.research/firered_onnx_parity.py`, fbank 80 chiều 25 ms/10 ms qua `kaldi-native-fbank` + CMVN, so với `fireredvad.core.detect_model.DetectModel` chạy torch):

| File | Số frame | max\|Δp\| | Frame lệch @0.30 / 0.40 / 0.50 |
|---|---|---|---|
| `Japanese_5s.wav` | 506 | 0,000000 | 0 / 0 / 0 |
| `English_multiple_kinds_of_noise_88s.wav` | 8.817 | 0,000001 | 0 / 0 / 0 |
| `Chinese_noise_28s.wav` | 2.824 | 0,000000 | 0 / 0 / 0 |
| `Russian_4s.wav` | 474 | 0,000000 | 0 / 0 / 0 |

⇒ **Bit-exact.** Phần cần thay chỉ là 3 mảnh nhỏ:
- `KaldifeatFbank` + `CMVN` (`fireredvad/core/audio_feat.py:41-106`) — torch chỉ xuất hiện ở **1 dòng** `torch.from_numpy(fbank).float()` (`:36`), bỏ được. `kaldi-native-fbank 1.22.3` + `kaldiio 2.18.1` đã cài sẵn, **không kéo torch**.
- `StreamVadPostprocessor` (`fireredvad/core/stream_vad_postprocessor.py`, 163 dòng) — **thuần Python, không có torch**. Vendor nguyên văn được.
- Lời gọi model: đổi `DetectModel.forward(feat, caches)` → `sess.run({'feat','caches_in'})`.

### 2.3. FSMN-VAD (engine tuỳ chọn thứ ba)

`funasr` là gói nặng (modelscope, omegaconf, hydra, jieba, umap-learn…) và **chỉ có một mình nó kéo cả chuỗi đó vào**; không có sẵn ONNX cục bộ trong repo. Khuyến nghị: **bỏ hẳn engine này** (README ghi rõ `silero-vad` và `fsmn-vad` đều là "tuỳ chọn", mặc định là `firered-vad`) hoặc chuyển sang export ONNX của funasr nếu thực sự cần. Đây là engine duy nhất chưa có đường ONNX đã kiểm chứng.

### 2.4. Lợi ích kèm theo: sửa được một giới hạn thật của `core/vad_silence.py`

`SILERO_PRIME_WINDOWS = 320` (`backend/core/vad_silence.py:72-95`) tồn tại **chỉ vì** state nội bộ của model JIT là hộp đen: muốn xác suất không phụ thuộc chỗ cắt vùng quét, phải chạy "mồi" ~10 s audio trước đó, và tài liệu trong file tự nhận **"lần quét ĐẦU TIÊN của một scanner vẫn nguội"**.

Với ONNX, `stateN` là tensor **do caller sở hữu** ⇒ có thể `snapshot`/`restore` state chính xác:

- Bỏ được 320 cửa sổ mồi (~0,11 s/lần quét).
- **Xoá hẳn** giới hạn "lần quét đầu nguội" — kết quả quét trở nên tất định và tái lập được (tốt cho test).
- Kiểm soát được `batch > 1` đúng ngữ nghĩa (repo đã đo batch≠1 cho kết quả khác và phải khoá `SILERO_BATCH_WINDOWS = 1`).

Đây là cải thiện **chất lượng**, không chỉ là thay thế runtime.

---

## 3. ForcedAligner → GGUF CrispASR: phân tích tương thích

### 3.1. CrispASR là gì (số liệu kiểm chứng được)

| Thuộc tính | Giá trị |
|---|---|
| Repo | [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR) — "C++ ggml runtime hub for multilingual ASR and TTS models" |
| Giấy phép | **MIT** (giống repo này) |
| Quy mô / độ phổ biến | 736 ★, 116 fork, 3.738 file, tạo 2026-03-29, commit cuối 2026-10-06 |
| Phát hành | `v0.8.41` (2026-10-02); **release mỗi 1-3 ngày** |
| Tuyên bố cốt lõi | *"No Python. No PyTorch. No separate per-model binary. No pip install. Just one C++ binary and a GGUF file."* |
| Binary Windows dựng sẵn | `crispasr-windows-x86_64-cuda13.zip` (539 MB — CUDA 13, **khớp đúng cu130** của repo), `…-cuda.zip` (746 MB), `…-vulkan.zip` (36,9 MB), `…-cpu.zip` (8,5 MB) |
| Wheel Python | `crispasr-0.8.41+cuda-py3-none-win_amd64.whl` (737 MB), `…+vulkan-…whl` (30 MB) |
| Thư viện rời | `libcrispasr-windows-x86_64-cuda13.tar.gz`, `…-vulkan.tar.gz` |

### 3.2. Model aligner

- Registry nội bộ (`src/crispasr_model_registry.cpp:1336-1347`): `qwen3-forced-aligner` → `https://huggingface.co/cstr/qwen3-forced-aligner-0.6b-GGUF/resolve/main/qwen3-forced-aligner-0.6b-q4_k.gguf`, **~500 MB**, "Q4_K là quant khuyến nghị; Q5_0 / Q8_0 / F16 cũng có trên repo".
- Bản GGUF **có mang head timestamp**: script convert chính chủ (`models/convert-qwen3-asr-to-gguf.py:91`) map `thinker.lm_head.weight` → `output.weight`. Đúng với file safetensors cục bộ: `thinker.lm_head.weight` = **[5000, 1024] BF16** (không phải vocab 152064) — tức `config.json:23 "classify_num": 5000` là head riêng, khớp `modeling_qwen3_asr.py:1114-1115`:
  ```python
  if "forced_aligner" in config.model_type:
      self.lm_head = nn.Linear(config.text_config.hidden_size, config.classify_num, bias=False)
  ```
- Runtime tự nhận diện qua chiều `output.weight`: `qwen3_asr.cpp:481,2352` (`qwen3_asr_lm_head_dim`) và `crispasr_aligner.cpp:323-330` cảnh báo nếu dim ≠ ~5000.

### 3.3. Độ trung thành của bản port (đọc mã nguồn CrispASR)

| Khía cạnh | Upstream (torch, đang dùng) | CrispASR (GGML) | Khớp? |
|---|---|---|---|
| Prompt | `<\|audio_start\|><\|audio_pad\|><\|audio_end\|>` + các đơn vị nối bằng `<timestamp><timestamp>` (`qwen3_forced_aligner.py:249-250`) | `qwen3_asr.cpp:2403-2424`: *"unit`<timestamp><timestamp>`unit…"* | ✅ |
| Sinh mốc | `logits.argmax(-1)` tại vị trí `input_id == timestamp_token_id(151705)`, ×`timestamp_segment_time(80 ms)` (`:458-464`) | `qwen3_asr.cpp:2480-2514` "argmax over the lm_head_dim classes at each `<timestamp>`" | ✅ |
| Sửa mốc đơn điệu | `Qwen3ForceAlignProcessor.fix_timestamp` (LIS + nội suy, `:147-234`) | `src/core/qwen3_forced_aligner.h` — ghi rõ *"Exact port of Qwen3ForceAlignProcessor.fix_timestamp() at upstream QwenLM/Qwen3-ASR commit 7c6daf77a…"* | ✅ |
| Ghép cặp (start,end) theo đơn vị | `parse_timestamp` (`:254-267`) | `align_words_impl` 1:1 theo `label_words` | ✅ |
| Gắn lại dấu câu | `ForcedAlignerService.merge_source_text` (tầng Python của repo) | **Có sẵn bên trong**: `tokenise_words` + `tokenise_display_words` + `restore_text` (`crispasr_aligner.cpp:127-216, 741-743`) | ⚠️ tương đương, xem 3.4b |
| Audio dài | không giới hạn tường minh | tự cắt khối >120 s (`crispasr_aligner.cpp:261,337-376`) — khối của repo ≤30 s nên không đụng | ✅ |
| Mốc tuyệt đối | cộng offset ở tầng gọi | `t_offset_cs` là **tham số của API**, trả về centiseconds tuyệt đối | ✅ |
| GPU | torch CUDA | `qwen3_asr_context_default_params()` → `use_gpu = true`, tự chọn CUDA/Vulkan/Metal, fallback CPU | ✅ |

### 3.4. KHÁC BIỆT thực sự ảnh hưởng tới backend này

**(a) Đơn vị align — khác biệt lớn nhất.**

- Hiện tại: `encode_timestamp()` gọi `nagisa.tagging(text).words` cho tiếng Nhật và `soynlp` cho tiếng Hàn ⇒ đơn vị là **TỪ** (vd. `人生`, `魅力的だ`). Tiếng Anh tách theo khoảng trắng.
- CrispASR (`crispasr_aligner.cpp:77-85,127-159`): mỗi ký tự CJK (Hiragana/Katakana/Han/Hangul) là **một đơn vị riêng**; đây là **quyết định có chủ đích**, được ghi lại trong issue #32 (tiếng Nhật ban đầu hỏng vì tách theo khoảng trắng ⇒ cả câu thành 1 "từ" ⇒ 1 mốc duy nhất).

Hệ quả trực tiếp lên code hiện tại:

- `ForcedAlignerService.group_words_to_subtitles` / `split_oversized_sentences` đang cắt theo `max_words` (mặc định 24) và `max_duration`. Với tiếng Nhật, 24 **từ nagisa** ≈ 40-60 ký tự, còn 24 **ký tự** chỉ ≈ 1 câu ngắn ⇒ **phụ đề tiếng Nhật/Trung sẽ bị chẻ nhỏ hơn đáng kể** nếu giữ nguyên ngưỡng.
- `backend/tests/test_57_forced_aligner_service.py`, `test_58_lookahead_offline_batch.py`, `test_60_sentence_segmentation_english.py` đang mã hoá kỳ vọng theo **đơn vị từ** (bằng chứng: các test dùng `RealisticForcedAligner` mô phỏng "model bỏ dấu câu khi tokenize"). Sẽ phải viết lại phần test CJK.
- Việc "gom ký tự thành từ" cho phụ đề CJK phải do tầng Python của mình làm (CrispASR chỉ ghép khi in SRT — `crispasr_aligner.cpp` ghi chú #465 về khoảng trắng, và #465 là bug tiếng Hàn mất hết khoảng trắng khi xuất `-sp/-ml`).

**(b) `language=` biến mất khỏi API.** `crispasr_align_words_abi(aligner_model, transcript, samples, n_samples, t_offset_cs, n_threads)` không có tham số ngôn ngữ; CrispASR quyết định cách tách theo **chữ viết** (script), không theo tên ngôn ngữ. Hiện tại repo truyền `language="Japanese"` để chọn nhánh nagisa. Nếu chuyển sang CrispASR:
- `resolve_aligner_language()` và `detect_aligner_language_from_text()` (`forced_aligner.py:97-139`) trở thành **code chết**.
- Mất khả năng phân biệt tiếng Trung vs tiếng Nhật khi cả hai dùng chữ Hán (CrispASR chỉ thấy "có Han" — giống hệt heuristic hiện tại ở `:133-138`, nên rủi ro thực tế thấp).
- `nagisa`/`soynlp` (2 dependency trong `requirements.txt:111-112`) không còn cần.

**(c) Dấu câu gắn lại — cần kiểm tra chứ đừng giả định.** CrispASR đã tự gắn dấu câu vào đơn vị hiển thị (display word). Hàm `merge_source_text` của repo (`:529-560`) khớp theo chuỗi ký tự "được giữ" nên **nhiều khả năng là idempotent**, nhưng phải kiểm chứng bằng test thật trước khi quyết định bỏ hay giữ.

**(d) Nhạy với im lặng đầu khối (đã được upstream ghi nhận).** `docs/cli.md:743-745,779-780`: *"`qwen3-forced-aligner` … more sensitive to leading silence; `--vad` is strongly recommended with it"*; issue #8 (đã đóng) mổ xẻ đúng hiện tượng mốc đầu bị lệch khi khối có im lặng đầu. Với repo này, khối ASR được cắt ở **giữa** khoảng lặng nên luôn có tối đa ~min_silence/2 (≈0,75 s) im lặng đầu khối ⇒ **phải A/B đo lại** độ chính xác mốc so với đường torch hiện tại.

**(e) Ngữ nghĩa tham số im lặng.** Hiện tại repo chủ động hạ ngưỡng Silero xuống 0.30 cho bộ quét cắt khối (`config.lookahead.batch_vad_silence_threshold`, xem `core/vad_silence.py:144-151`). API ONNX giữ đúng tham số `threshold` này nên bảo toàn được hành vi — nhưng nếu định dùng VAD của CrispASR thì CrispASR có `crispasr_vad_options.threshold_explicit` và vài cơ chế auto-tune riêng (`crispasr_vad.h:66-74`) ⇒ dễ lệch hành vi. **Nên giữ VAD ở Python/ONNX, chỉ lấy aligner từ CrispASR.**

### 3.5. VAD của CrispASR (tham khảo, không khuyến nghị dùng thay ONNX)

CrispASR cũng có 4 VAD GGML: Silero (mặc định, ~885 KB), **FireRedVAD (`-vm firered`, 2,4 MB, F1=97,57%)**, MarbleNet (439 KB, 6 ngôn ngữ), Whisper-VAD-EncDec (thử nghiệm). C ABI: `crispasr_vad_slices` (dispatcher, `crispasr_vad.cpp:40-63` cache context theo path) và `whisper_vad_n_probs`/`whisper_vad_probs` cho xác suất từng frame.

Nhược điểm so với ONNX trực tiếp trong repo này: (1) là **bản viết lại GGML**, chưa có bằng chứng bit-parity với bản torch như ONNX đã kiểm chứng; (2) API thiên về **quét lô/khoảng lặng**, không phải VAD streaming per-frame có event START/END như `VADProcessor` (Pipeline A) cần; (3) kéo thêm ràng buộc DLL.

---

## 4. Rủi ro hạ tầng khi tích hợp CrispASR (QUAN TRỌNG NHẤT)

### 4.1. Xung đột `ggml*.dll` — bộ ggml thứ ba trong cùng tiến trình

`backend/utils/cuda.py:18-22` đã ghi rõ:

> ⚠️ PHẢI là thư mục RIÊNG, KHÔNG dùng chung với bundle ASR (`backend/bin/`): cả hai đều chứa `ggml.dll` / `ggml-base.dll` / `ggml-cpu.dll` / `ggml-cuda.dll` nhưng là **hai bản ggml khác nhau**. Windows phân giải DLL phụ thuộc theo TÊN MODULE, nên để chung một thư mục thì `transcribe.dll` sẽ bind nhầm `ggml-base.dll` của llama.cpp và cả tiến trình chết với `0xc0000139`.

Hiện repo đã có **hai** bộ trong tiến trình chính (`backend/bin/`, `backend/bin/llama/`) và **một** bộ trong tiến trình con (`backend/bin/omnivoice/` → `backend/tts/worker.py`). CrispASR (`BUILD_SHARED_LIBS=ON` + `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS`, `CMakeLists.txt:6-7,102`) sẽ là **bộ thứ ba**, và ggml của CrispASR là **fork riêng** (có op không có trong upstream — xem `tools/upstream-prs/`, vd. `col2im_1d`).

**Khuyến nghị**: **không** `ctypes.CDLL` trực tiếp vào tiến trình backend. Chạy CrispASR trong **tiến trình worker riêng**, đúng khuôn mẫu `backend/tts/worker.py` (đã có sẵn IPC, vòng đời, giải phóng VRAM). Trong worker đó dùng Python binding `crispasr` (ctypes) để **giữ model đã nạp** giữa các lần gọi — `crispasr_aligner.cpp:27-71` có **static cache context aligner** ("avoid loading/freeing the GGUF model on every call — alignment models are 300 MB–1 GB and the load/free dominates wall time for short segments"), nên chi phí nạp chỉ trả một lần.

Phương án thay thế (kém hơn): gọi `crispasr.exe --align-only -am qwen3-forced-aligner-0.6b-q4_k.gguf --text-file - --align-format json` mỗi khối ⇒ nạp lại ~500 MB GGUF mỗi 25-30 s ⇒ không chấp nhận được cho pipeline 0-latency.

### 4.2. Án lệ AVX-512 trên artifact Windows

Issue #394/#374: `crispasr-windows-x86_64-cuda.zip` v0.8.29 **chết ngay lập tức với `0xC000001D`** trên CPU người dùng vì build mất pin ISA và `GGML_NATIVE=ON` kéo `/arch:AVX512` từ CPU của runner CI. Đã sửa trên `main` (`398aabc9`) và phát hành sau v0.8.29 — nhưng repo này **đã từng dính đúng lỗi đó với wheel llama.cpp** (`requirements.txt:39-50`). ⇒ Bắt buộc kiểm tra baseline ISA của release được ghim (`docs/windows-illegal-instruction-dumps.md`) và test trên CPU không có AVX-512.

### 4.3. Khác

| Rủi ro | Mức | Ghi chú |
|---|---|---|
| Nhịp phát hành rất nhanh (0.8.x mỗi 1-3 ngày) | Trung bình | Phải **ghim chính xác** tag `v0.8.41` và ghi lại hash; không dùng "latest". |
| Kích thước | Thấp | Windows CUDA13 zip 539 MB (tự chứa, không cần CUDA Toolkit) hoặc wheel 737 MB. Bù lại: bỏ `Qwen3-ForcedAligner-0.6B/` (1,84 GB safetensors) → **tiết kiệm ~1,3 GB đĩa** và **~1,1 GB VRAM** (1.756 MB `bfloat16` → ~500-700 MB Q4_K). |
| Giấy phép | Thấp | CrispASR MIT; model Qwen3-ForcedAligner Apache-2.0. |
| Có endpoint HTTP cho align không? | — | `crispasr-server` có `/v1/audio/transcriptions` (kèm tham số aligner/VAD) nhưng **không có endpoint align-only**. Chế độ `--align-only` chỉ có ở CLI (`docs/cli.md:822-865`), nên đường "worker + HTTP" không dùng được cho align ⇒ dùng worker + binding. |
| Port GGUF khác của cùng model | — | Có `predict-woo/qwen3-asr.cpp` (GGML) và `soniqo/speech-swift` (CoreML/Swift — không dùng được trên Windows). Không có port nào khác cho Windows. |

---

## 5. Lộ trình đề xuất

### Giai đoạn 1 — VAD sang ONNX (rủi ro thấp, đã kiểm chứng, làm được ngay)

1. Viết engine ONNX cho Silero: `backend/vad/engines/silero.py` (giữ nguyên `frame_samples=512`, `start_pad_ms`, `_ProbProbe` → thay bằng state tường minh) + `backend/core/vad_silence.py`.
   - Copy `silero_vad.onnx` (2,33 MB) từ package ra `backend/models/` giống cách `silero_vad.jit` đang được copy (`silero.py:112-130`).
   - Bỏ `SILERO_PRIME_WINDOWS` bằng snapshot/restore state (mục 2.4).
2. Viết engine ONNX cho FireRed: vendor `KaldifeatFbank` + `CMVN` (bỏ 1 dòng `torch.from_numpy`) và `StreamVadPostprocessor`; dùng `fireredvad_stream_vad_with_cache.onnx` (đã có trong `external/FireRedVAD`) + `cmvn.ark`.
   - **Điều kiện tiên quyết**: `external/` nằm trong `.gitignore` ⇒ phải **copy** model ONNX (2,31 MB) + `cmvn.ark` (1,3 KB) vào `backend/models/` và thêm bước tự-tải/kiểm tra như các model khác.
3. Quyết định engine FSMN: bỏ, hoặc export ONNX (chưa kiểm chứng).
4. Bỏ `fireredvad`, `silero-vad`, `funasr`, `kaldiio`, `kaldi-native-fbank`? — giữ `kaldi-native-fbank` + `kaldiio` cho frontend FireRed (không kéo torch).
5. Viết lại `backend/tests/test_41_vad_engine_contract.py` theo API ONNX.

### Giai đoạn 2 — ForcedAligner sang CrispASR GGUF

1. **PoC trước khi cam kết**: worker tối giản nạp `qwen3-forced-aligner-0.6b-q4_k.gguf` + `libcrispasr-windows-x86_64-cuda13`, chạy `align_words()` trên **cùng các khối 25-30 s** đã dùng cho đường torch, và so:
   - mốc từng đơn vị (`max|Δt0|`, `max|Δt1|`) — kỳ vọng lệch ở mức vài chục ms do khác đơn vị align;
   - **biên phụ đề cuối cùng** sau `group_words_to_subtitles` trên bộ test tiếng Nhật/Anh sẵn có (`Japanese_5s.wav`, `Chinese_noise_28s.wav`, `English_multiple_kinds_of_noise_88s.wav`);
   - hành vi với im lặng đầu khối (mục 3.4d).
2. Quyết định chiến lược đơn vị CJK: (a) gom ký tự → từ bằng từ điển/`nagisa` ở tầng Python (giữ `nagisa`, giữ nguyên hành vi phụ đề), hay (b) chấp nhận đơn vị ký tự và **chỉnh `max_words`/`max_chars` cho CJK** rồi cập nhật test. (a) giữ được parity nhưng cần thêm tầng; (b) đơn giản hơn nhưng đổi hành vi người dùng.
3. Chuyển `ForcedAlignerService` thành client của worker; **giữ nguyên** `AlignedWord`, `SubtitleSentence`, `group_words_to_subtitles`, `split_oversized_sentences`, `merge_source_text` (nếu A/B xác nhận vẫn cần).
4. Xoá `backend/asr/qwen_asr/**` (~10 file) và bỏ `transformers`, `nagisa`?, `soynlp`? khỏi requirements.
5. Cập nhật pre-flight trong `main.py:494-531` (đang kiểm tra thư mục `Qwen3-ForcedAligner-0.6B/`) → kiểm tra file `.gguf`.

### Giai đoạn 3 — Dọn sạch torch

| File | Việc |
|---|---|
| `backend/main.py:40-46` | Xoá khối `import torch; set_num_threads` (hoặc thay bằng giới hạn luồng native). |
| `backend/main.py:115-165,725-727` | Xoá `_log_torch_status_at_startup` + khối `torch` trong `/health`; thay bằng khối `native_runtime`. |
| `backend/asr/engine.py:496-498`, `backend/translation/engine.py:460-462` | Xoá `empty_cache` (không còn torch trong tiến trình). |
| `backend/ws/lookahead_handler.py:134-147` | Đã an toàn; có thể xoá `_empty_torch_cache` cho gọn. |
| `backend/utils/env_check.py` | Bỏ `torch_status`/các dòng `torch.cuda`, đổi kỳ vọng `env_check` (README dòng 172 đang ghi *"Kỳ vọng: `torch CUDA: OK`"*). |
| `backend/utils/cuda.py:57-80,128-131` | Bỏ `_configure_alloc_conf` và nhánh quét `torch/lib`. |
| `backend/requirements.txt` | Bỏ `torch>=2.4`, `torchaudio>=2.4`, `omnivoice>=0.2.1`, `fireredvad`, `silero-vad`, `funasr`, `transformers`; bỏ `--extra-index-url https://download.pytorch.org/whl/cu130`; thêm `crispasr` (ghim `0.8.41`). Giữ `onnxruntime>=1.17`. |
| `backend/constraints.txt`, `README.md` (bảng môi trường, mục Khắc phục sự cố, mục Quản lý Model), `run.md` | Cập nhật. |
| `backend/tools/test_qwen3_aligner_poc.py`, `backend/tests/test_07_e2e_comparison.py` | Cập nhật/xoá. |

**Lợi ích ngoài việc bỏ torch** (số đo thật trên venv hiện tại):

| Thành phần có thể gỡ | Dung lượng trong `.venv/Lib/site-packages` |
|---|---|
| `torch` | 2.666 MB |
| `nvidia/` (CUDA runtime kéo theo torch) | 925 MB |
| `transformers` | 98 MB |
| `gradio` (kéo theo bởi `omnivoice`) | 75 MB |
| `modelscope` + `funasr` | 47 MB + 8 MB |
| `torchaudio` | 12 MB |
| **Tổng** | **≈ 3,8 GB** |

Cộng thêm **1,84 GB** (`Qwen3-ForcedAligner-0.6B/model.safetensors`) được thay bằng GGUF ~500 MB, và xoá luôn cả lớp sự cố *"`pip install -U <gói>` âm thầm hạ torch xuống bản CPU-only"* (`main.py:115-123`, `env_check.py:85-90`).

### Thứ tự thực thi khuyến nghị

```
GĐ1 (VAD ONNX)  → độc lập, đã kiểm chứng bit-parity, gỡ được fireredvad/silero-vad/funasr
      ↓
GĐ2 PoC aligner → CHỈ sau khi có số đo A/B; đây là phần chưa chắc chắn
      ↓
GĐ3 (dọn torch) → chỉ khi GĐ1 + GĐ2 xong, vì torch phải còn để làm đường so sánh A/B
```

> ⚠️ **Đừng `pip uninstall torch` trước khi GĐ2 xong**: cần torch làm "oracle" để so mốc thời gian.

---

## 6. Kiểm chứng cần làm trước khi chốt (acceptance)

1. **Silero ONNX**: chạy lại `.research/vad_onnx_parity.py`, kỳ vọng `max|Δp| ≤ 1e-5` và `0` frame lệch @0.30/0.40/0.50 (đã đạt).
2. **FireRed ONNX**: chạy lại `.research/firered_onnx_parity.py`, cùng kỳ vọng (đã đạt).
3. **VAD end-to-end**: `VADProcessor` trên cùng audio phải cho **cùng chuỗi event START/END** (kể cả `lookback_frames` và `start_pad_ms`) như engine torch — dùng `backend/tests/test_41_vad_engine_contract.py` làm khung.
4. **Aligner A/B**: max|Δt| theo đơn vị + **số phụ đề và biên phụ đề** trên ≥3 video thật (Nhật, Anh, Trung), so đường torch vs GGUF.
5. **Không hồi quy**: `pytest` (tầng A) phải xanh sau khi bỏ torch — hiện `pytest` chạy được **không cần model**; sau khi bỏ torch phải chạy được **không cần cả torch** (đây chính là bài test cốt lõi: môi trường sạch không có torch).
6. **Kiểm tra ISA**: chạy binary CrispASR được ghim trên CPU **không có AVX-512** (xem #394/#374).

---

## 7. Rủi ro & câu hỏi còn mở

| # | Nội dung | Trạng thái |
|---|---|---|
| 1 | Đơn vị align CJK theo ký tự có làm hỏng trải nghiệm phụ đề tiếng Nhật hiện tại không? | **Chưa đo** — cần A/B (mục 6.4). Đây là câu hỏi quyết định. |
| 2 | `merge_source_text` có còn cần khi CrispASR đã tự gắn dấu câu? | Chưa kiểm chứng — đọc code cho thấy nhiều khả năng idempotent nhưng phải test. |
| 3 | Tầng gom ký tự → từ cho CJK: làm ở Python (giữ `nagisa`) hay đổi ngữ nghĩa `max_words`? | Cần quyết định thiết kế. |
| 4 | Chạy CrispASR ở worker process: tốn thêm ~500-700 MB VRAM (đã bù lại vì bỏ 1.756 MB torch) và thêm 1 tiến trình; chấp nhận được không? | Đề xuất: có — đúng khuôn mẫu TTS worker sẵn có. |
| 5 | FSMN-VAD: bỏ hẳn hay tìm ONNX? | Đề xuất: bỏ (tuỳ chọn, không mặc định). |
| 6 | Nhánh CUDA13 (539 MB) hay Vulkan (36,9 MB)? | CUDA13 khớp cu130 hiện tại; Vulkan nhẹ hơn nhiều và repo đã có tiền lệ dùng Vulkan làm fallback. Cần đo RTF aligner trên cả hai. |

---

## Phụ lục A — Script đo tái lập

- `.research/vad_onnx_parity.py` — Silero JIT vs ONNX (giữ 64 mẫu context, state `[2,1,128]`).
- `.research/firered_onnx_parity.py` — FireRed torch `DetectModel` vs `fireredvad_stream_vad_with_cache.onnx` (fbank `kaldi-native-fbank` + CMVN `cmvn.ark`).
- `.research/CrispASR/` — bản clone `--depth 1` của `CrispStrobe/CrispASR@main` (2026-10-06) để đối chiếu mã nguồn; **đã thêm `.research/` vào `.gitignore`**.

## Phụ lục B — Nguồn tham chiếu bên ngoài

- CrispASR: <https://github.com/CrispStrobe/CrispASR> · aligner API `src/crispasr_aligner.{h,cpp}` · header C ABI `include/crispasr_session.h:333-340` · binding Python `python/crispasr/_binding.py` (`align_words`, `vad_segments`)
- Model GGUF: <https://huggingface.co/cstr/qwen3-forced-aligner-0.6b-GGUF> (Q4_K ~500 MB)
- Bản mirror cộng đồng cùng model: `OpenVoiceOS/qwen3-forced-aligner-0.6b-q8-0`, `…-q4-k-m`
- Issue đã đọc: #32 (aligner tiếng Nhật), #444 + #447 (mốc chồng lấn / gắn lại dấu câu), #8 (nhạy im lặng đầu), #465 (khoảng trắng tiếng Hàn), #394 + #374 (AVX-512 artifact Windows), #190 (registry)
- Port khác: `predict-woo/qwen3-asr.cpp` (GGML), `soniqo/speech-swift` (CoreML — không dùng được trên Windows)
