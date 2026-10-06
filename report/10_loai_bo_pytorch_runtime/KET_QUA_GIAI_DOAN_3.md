# Kết quả Giai đoạn 3 — Dọn sạch PyTorch

- **Ngày**: 2026-10-06
- **Máy đo**: RTX 5060 Ti 16 GB, Windows 11, Python 3.13.14, onnxruntime 1.30.0
- **Mục tiêu**: (a) FSMN-VAD → ONNX hoặc xoá; (b) dọn sạch mọi thứ liên quan tới torch;
  (c) xử lý giới hạn 100 MB/file của GitHub cho DLL CrispASR.
- **Trạng thái**: **XONG** — `pytest -m slow` 65/65 đạt, tier A 0 lỗi mới, JS 93/93 đạt.

---

## 1. FSMN-VAD — XOÁ (kèm lý do có bằng chứng)

**Có chuyển được sang ONNX không?** Về mặt kỹ thuật **CÓ**: FunASR hỗ trợ export ONNX
(`funasr/models/fsmn_vad_streaming/encoder.py` có `FsmnStack`/`FSMNExport`/`BasicBlock_export`,
`export_meta.py` có `export_rebuild_model`/`export_forward`/`export_dummy_inputs`).

**Nhưng không đáng**, vì để CHẠY được model ONNX đó còn phải vendor thêm:
- frontend fbank 80 chiều + LFR + CMVN (`funasr/frontends/wav_frontend.py`),
- quản lý cache tensor (`model.py` ~**52 KB**: `VADXOptions`, `init_cache`, `infer`),
- máy trạng thái streaming (`dynamic_vad.py` ~9 KB: `DynamicStreamingVAD.feed`).

Đổi lại chỉ được một engine **KHÔNG mặc định**, đã bị FireRed (mặc định) và Silero lấn át về cả
chất lượng lẫn tài liệu. Theo đúng phương án dự phòng bạn cho phép, **đã xoá hẳn**:

| Hạng mục | Việc đã làm |
|---|---|
| Engine | Xoá `backend/vad/engines/fsmn.py` |
| Registry | `SUPPORTED_VAD_ENGINES = ("firered-vad", "silero-vad")` (cả `vad/base.py` lẫn `vad/engines/__init__.py`) |
| Config | Xoá `FsmnVADConfig` + field `fsmn` khỏi `VADConfig`; `/api/config` không còn trả khối `fsmn` |
| Startup | Bỏ `FsmnVADEngine.prepare_files()` khỏi pre-warm; bỏ nhánh pre-flight `fsmn-vad` |
| Popup | Bỏ `<option value="fsmn-vad">` ở **cả hai** extension; fallback `\|\| "fsmn-vad"` → `\|\| "firered-vad"` |
| Test | `test_41` bỏ mọi nhánh FSMN; test JS đổi sang `silero-vad` |

---

## 2. Dọn sạch PyTorch

### 2.1. Xoá hẳn đường torch của aligner

| Việc | Chi tiết |
|---|---|
| Xoá cây vendored | `backend/asr/qwen_asr/**` (25 file, 0,71 MB) — chính nó kéo `transformers` + `qwen_asr` |
| Xoá API torch | `load_model()`, `_use_crispasr()`, `_align_crispasr()`, `ensure_forced_aligner_model()`, `ALIGNER_LOCAL_DIR`, `ALIGNER_DEFAULT_REPO_ID`, `_torch()`, `_load_qwen_asr_class()`, `_cuda_*` |
| Xoá tool PoC | `backend/tools/test_qwen3_aligner_poc.py` |
| Thay bằng | `ForcedAlignerService.is_ready()` → `(ok, lý do)`; `align()` ném lỗi **có hướng dẫn** khi runtime thiếu |
| `backend/asr/__init__.py` | Bỏ khối chèn `sys.path` cho `qwen_asr`; thêm cảnh báo về xung đột DLL |

### 2.2. Bỏ mọi `import torch` / `torch.cuda.*` trong runtime

| File | Việc |
|---|---|
| `backend/main.py` | Bỏ khối `import torch; torch.set_num_threads(2)` lúc khởi động; `/health`: khối `torch` → khối **`vad_runtime`**; `_log_torch_status_at_startup` → `_log_runtime_status_at_startup` (log onnxruntime + trạng thái aligner) |
| `backend/asr/engine.py` | Bỏ `torch.cuda.empty_cache()` khi hot-swap model ASR (VRAM đó thuộc `transcribe.dll`, `old_model.close()` đã trả) |
| `backend/translation/engine.py` | Bỏ `torch.cuda.empty_cache()` khi hot-swap model dịch |
| `backend/ws/lookahead_handler.py` | Xoá `_empty_torch_cache()` + call-site (không còn cache torch nào để dọn) |
| `backend/utils/cuda.py` | `_configure_alloc_conf()` thành **no-op có tài liệu**; bỏ nhánh quét `site-packages/torch/lib` |
| `backend/utils/env_check.py` | **Viết lại toàn bộ** (xem 2.3) |

### 2.3. `env_check.py` viết lại

File cũ hoàn toàn xoay quanh `torch_status()` + `EXPECTED_CUDA_TRIO` + chống pip hạ torch CPU-only.
Nay kiểm tra **runtime thật**:

```
== Môi trường runtime của dự án (không còn PyTorch) ==
python        : 3.13.14
onnxruntime   : 1.30.0  (AzureExecutionProvider, CPUExecutionProvider)
VAD engine    : firered-vad  (model ONNX đủ: True)
aligner       : crispasr  (sẵn sàng: True)
llama_cpp     : cài rồi
transcribe-cpp: cài rồi
torch         : CÒN CÀI (không subsystem nào cần)
llama_cpp nạp : OK
ASR binding   : OK
GỢI Ý: gỡ PyTorch để nhẹ môi trường — `pip uninstall torch torchaudio transformers`.
```

Hàm mới: `runtime_status()`, `check_llama_cpp()`, `check_asr_binding()`, `check_environment()`,
`format_report()`, `main()`. Dòng `torch:` chỉ để **báo cáo** (kèm gợi ý gỡ) — không còn là "vấn đề".

### 2.4. Manifest

- `requirements.txt`: gỡ `torch`, `torchaudio`, `transformers`, `nagisa`, `soynlp`, `omnivoice`,
  `fireredvad`, `silero-vad`, `funasr` và dòng `--extra-index-url https://download.pytorch.org/whl/cu130`.
  Còn lại: fastapi/uvicorn/pydantic/numpy/scipy/soundfile/PyYAML/psutil/cryptography/orjson,
  transcribe-cpp(+native), llama-cpp-python, **onnxruntime + kaldi-native-fbank + kaldiio**, av,
  huggingface_hub.
- `constraints.txt`: gỡ cả 3 dòng ghim `torch==2.14.0+cu130` / `torchaudio` / `torchvision`, kèm ghi
  chú vì sao file này từng tồn tại và vì sao nay không cần.

### 2.5. Test

| Việc | Chi tiết |
|---|---|
| Xoá | `test_45_torch_env_guard.py` (15 test về torch) |
| Thêm | `test_45_runtime_env_guard.py` (16 test cho runtime mới) |
| `test_62` (VAD) | Bỏ tham chiếu torch trực tiếp → **kỳ vọng VÀNG** `tests/fixtures/vad_onnx_golden.json` (frame count + chuỗi sự kiện + SHA-256 vector xác suất), sinh bằng `.research/gen_vad_golden.py` |
| `test_64` (aligner) | A/B với torch → **kỳ vọng vàng** (`_GOLDEN_SUBTITLES`, chụp từ đường torch trước khi xoá); thêm test chốt "không còn API đường torch" |
| `test_57` | Test stub `load_model()` → stub **client CrispASR**; test dùng tokenizer `qwen_asr` → test đơn vị **theo ký tự** |
| `test_07`, `test_41` | Bỏ `import torch`, bỏ mọi nhánh FSMN |

---

## 3. Giới hạn 100 MB của GitHub — đã xử lý

**Vấn đề có thật**: `backend/bin/crispasr/ggml-cuda.dll` = **149,63 MB > 100 MB** ⇒ GitHub từ chối
push ở chế độ thường. Các DLL cũ trong repo đều dưới ngưỡng (lớn nhất 67,73 MB) nên chưa từng gặp.

**Cách xử lý đã chọn** — tự tải khi thiếu, KHÔNG commit file lớn:

`backend/utils/crispasr_native.py::ensure_lib_files()`:
1. Nếu đủ 5 DLL → dùng ngay (đường nhanh, chỉ kiểm tồn tại — không băm 167 MB mỗi lần).
2. Nếu thiếu → tải `crispasr-windows-x86_64-cuda13-non-cuda.zip` (**157 MB**, ghim tag `v0.8.41`)
   từ GitHub Releases, **xác thực SHA-256 của gói**, giải nén đúng 5 DLL, rồi **xác thực SHA-256
   từng DLL**.
3. Lỗi mạng → thông báo có URL tải thủ công + hướng dẫn đặt `CRISPASR_LIB_DIR`.

**Đo thật**: tải + giải nén + xác thực = **9,0 s**, ra đủ `crispasr.dll`, `ggml.dll`, `ggml-base.dll`,
`ggml-cpu.dll`, `ggml-cuda.dll`.

**Nếu bạn vẫn muốn commit thẳng vào git** thì phải dùng Git LFS (đã cài sẵn `git-lfs 3.7.1`):

```powershell
git lfs install
git lfs track "backend/bin/crispasr/ggml-cuda.dll"
git add .gitattributes backend/bin/crispasr/
```

> Lưu ý: LFS free tier của GitHub là 1 GB lưu trữ + 1 GB băng thông/tháng; mỗi lần clone tốn
> 150 MB. Cách "tự tải khi thiếu" ở trên không tốn quota và giữ clone nhẹ — và vẫn hoạt động kể
> cả khi bạn commit 4 DLL nhỏ (17,9 MB) mà bỏ DLL lớn.

---

## 4. Kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Tier A `pytest` | **0 lỗi mới** — chỉ còn 11 test `test_40_asr_model_download.py` **đã hỏng từ trước** (`models.yaml` không còn `whisper-large-v3-turbo`; file đó tôi không sửa) |
| Tier B `pytest -m slow` | **65/65 đạt, 0 lỗi** |
| Test JS extension | **93/93 đạt** |
| `python -m backend.utils.env_check` | exit 0, in đúng runtime mới |

### Bằng chứng mạnh nhất: chạy thật với `torch` bị CHẶN Ở CẤP IMPORT

`test_45_runtime_env_guard.py::test_chay_that_khi_torch_bi_chan_import` cài hook `sys.meta_path`
chặn `torch`, `torchaudio`, `transformers`, `funasr`, `nagisa`, `soynlp`, `omnivoice`, `fireredvad`,
`silero_vad` — rồi chạy **đúng các đường thật** của backend trong tiến trình con:

```
VAD firered-vad: 508 frame, 2 su kien
VAD silero-vad: 158 frame, 4 su kien
VADSilenceScanner: OK
Aligner: 25 don vi, 1 phu de
OK-TORCH-FREE
```

⇒ Không còn phụ thuộc ẩn nào vào PyTorch (kể cả `import` lười nằm trong `except`).

---

## 5. Còn lại / khuyến nghị

1. **Dọn dung lượng đĩa trên máy bạn** (không tự xoá vì là dữ liệu tải về của bạn):
   - `backend/models/Qwen3-ForcedAligner-0.6B/` → **1,84 GB** (snapshot safetensors, nay vô dụng)
   - `pip uninstall torch torchaudio transformers` → **~3,8 GB** trong `.venv`
   - `backend/models/fsmn_vad/` → 6,5 MB (model FSMN, engine đã xoá)
2. `backend/tests/test_40_asr_model_download.py` **vẫn hỏng sẵn** (không liên quan Giai đoạn 3):
   nó chờ `whisper-large-v3-turbo` trong `backend/models.yaml` nhưng catalog chỉ còn qwen3-asr.
3. `backend/tools/build_finetune.py` có nhắc `torch`/`transformers` trong docstring — đây là công
   cụ fine-tune tách rời, không nằm trong đường chạy runtime; nếu không dùng nữa thì xoá.
4. Rủi ro đã ghi ở báo cáo Giai đoạn 2 vẫn giữ nguyên: câu **CJK rất dài không có dấu kết câu** có
   thể chẻ phụ đề sớm hơn đường torch (trần `sentence_max_words` đếm ký tự thay vì từ).
