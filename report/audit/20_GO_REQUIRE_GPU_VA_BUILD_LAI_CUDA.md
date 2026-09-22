# 20 — GỠ `require_gpu` + BUILD LẠI BUNDLE ASR CUDA

> Thực hiện 2026-09-22. Hai yêu cầu: (1) xoá tính năng `require_gpu` khỏi `ASRConfig`,
> (2) build lại bản build **CUDA chuẩn** cho `transcribe.cpp`.

---

## 1. Gỡ `require_gpu`

Trước đây (2026-09-20, `report/audit/18_KE_HOACH_TEST_TIENG_NHAT.md` §8) hệ thống có hai tầng
"bắt buộc GPU":

| Tầng | Cơ chế cũ | Trạng thái |
|---|---|---|
| Python | `ASRConfig.require_gpu` + `GpuRequiredError`: `resolve_backend()` **NÉM LỖI** khi không có backend GPU khả dụng | **ĐÃ XOÁ** |
| Native | env `GGML_SCHED_REQUIRE_GPU=1` do `bootstrap()` đặt, kèm patch ggml `ggml_backend_sched_require_gpu_check()` từ chối graph có op rơi về CPU | **ĐÃ XOÁ** (env + patch + code native) |

**Đã sửa:**

| File | Thay đổi |
|---|---|
| `backend/config.py` | Xoá field `ASRConfig.require_gpu` và toàn bộ khối chú thích về nó; ghi rõ hành vi mới ngay tại `backend_fallback` |
| `backend/asr/native.py` | Xoá `_REQUIRE_GPU_ENV`, `_REQUIRE_GPU_OVERRIDE_ENV`, class `GpuRequiredError`, `require_gpu_enabled()`, `apply_require_gpu_env()`; bỏ lời gọi trong `bootstrap()`; `resolve_backend()` không còn nhánh ném lỗi cứng |
| `external/transcribe.cpp/ggml/src/ggml-backend.cpp` | `git checkout` để **revert** patch đang áp (uncommitted) |
| `external/transcribe.cpp/patches/ggml/0002-require-gpu-no-cpu-fallback.patch` | **Đã xoá** (patch `0001-fix-threadpool-oversubscription` vẫn giữ nguyên) |
| `report/audit/18_KE_HOACH_TEST_TIENG_NHAT.md` | Ghi chú §8.2 rằng tính năng đã bị gỡ (giữ nguyên phần còn lại làm ghi chép lịch sử) |

**Hành vi hiện tại** (đúng như tài liệu `transcribe.cpp`): chọn backend theo `_PREFERENCE`
(`auto`/`cuda` → cuda rồi vulkan; `vulkan` → vulkan rồi cuda), tôn trọng
`config.asr.backend_fallback`, ghi **WARNING** rõ ràng khi không backend GPU nào khả dụng,
và **không** ném lỗi cứng. CPU vẫn không nằm trong danh sách lựa chọn.

Kiểm chứng: `python -c "from backend.asr import native; ..."` →
`resolve('auto')=cuda`, `resolve('cuda')=cuda`, `resolve('vulkan')=vulkan`, `resolve('cpu')` (không
hợp lệ) → coi như `auto` → `cuda`; `'require_gpu' in config.asr.model_dump()` = **False**;
`GGML_SCHED_REQUIRE_GPU` **không** còn được đặt vào môi trường.

---

## 2. Build lại bundle CUDA

Dùng đúng script của repo: `cmd /c external\build-tmp\build_cuda.bat` (build **sạch**):
`rmdir external\build-cuda` → CMake/Ninja configure → build 342 mục tiêu → copy vào `bin\`.

| Thành phần | Giá trị |
|---|---|
| Toolchain | MSVC 18 Community (`vcvars64.bat`) · CUDA 13.4.59 (`external/cuda-toolkit/nvidia/cu13`) · Ninja · CMake |
| Kiến trúc | `CMAKE_CUDA_ARCHITECTURES=120` (RTX 5060 Ti, sm_120) |
| Cờ | `TRANSCRIBE_CUDA=ON`, `TRANSCRIBE_VULKAN=OFF`, `GGML_CUDA=ON`, `GGML_CUDA_GRAPHS=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `TRANSCRIBE_BUILD_SHARED=ON`, `TRANSCRIBE_GGML_BACKEND_DL=ON`, `TRANSCRIBE_X86_CONSERVATIVE=ON`, tests/examples/tools = OFF |
| Kết quả | `BUILD_OK` — 342/342 mục tiêu, không lỗi |

`bin/` sau build (đã kiểm tra **không còn** chuỗi `GGML_SCHED_REQUIRE_GPU` trong cả 5 DLL):

| DLL | Kích thước | Ghi chú |
|---|---|---|
| `transcribe.dll` | 1.853.440 | mới (13:58) |
| `ggml.dll` | 68.096 | mới |
| `ggml-base.dll` | 665.600 | mới |
| `ggml-cpu.dll` | 797.696 | mới |
| **`ggml-cuda.dll`** | 71.015.424 | mới, sm_120 |
| `ggml-vulkan.dll` | 52.653.056 | **giữ nguyên** (lấy từ wheel `transcribe-cpp-native`, như script mô tả) |

---

## 3. Phát hiện kèm theo: môi trường đã mất CUDA runtime của torch

Trong lúc build, `bin/ggml-cuda.dll` (bản cũ) **không nạp được** dù file còn nguyên. Nguyên nhân:

* `torch` của môi trường đã bị đổi sang **bản CPU-only**: `torch 2.12.0+cu130` → **`2.9.1`**
  (thư mục `torch/` ghi lúc **07:52 cùng ngày**), `torchaudio` 2.9.1 theo cùng;
  `torchvision` 0.27.0+cu130 còn sót lại.
* Hệ quả: `torch/lib` **hết** `cudart64_13.dll` / `cublas64_13.dll` / `cublasLt64_13.dll`
  ⇒ mọi DLL CUDA (ASR `ggml-cuda.dll`, và cả llama.cpp/TTS nếu dùng CUDA torch) không resolve
  được phụ thuộc ⇒ backend CUDA "biến mất", chỉ còn Vulkan/CPU.

**Cách xử lý (đã làm):** `backend/utils/cuda.py` nay đăng ký thêm **CUDA 13 toolkit trong repo**
(`external/cuda-toolkit/nvidia/cu13/bin/x86_64`, hoặc `.cuda-toolkit/...`) — đúng bộ runtime mà
`build_cuda.bat` đã dùng để link. Nhờ vậy bundle ASR CUDA **tự đủ, không phụ thuộc bản torch nào**.

| Trước khi sửa | Sau khi sửa |
|---|---|
| `available: {cuda: False, vulkan: True, cpu: True}` | `available: {cuda: True, vulkan: True, cpu: True}` · `device: cuda CUDA0 16310 MB` |

> ⚠️ **Cần bạn quyết định**: nếu muốn TTS (OmniVoice `device=cuda:0`) và model dịch chạy CUDA
> như trước, phải cài lại torch CUDA:
> `pip install torch==2.12.0+cu130 torchaudio==2.12.0+cu130 --index-url https://download.pytorch.org/whl/cu130`
> (cần mạng, ~2,5 GB). Bản torch CPU hiện tại vẫn chạy được mọi thứ nhưng chậm hơn nhiều.

---

## 4. Kiểm chứng

| Hạng mục | Lệnh | Kết quả |
|---|---|---|
| Bundle nạp đủ 3 backend + nạp model thật | `python external/build-tmp/verify_bin.py` | `Model(backend='cuda') -> OK, actual=CUDA0, arch=qwen3_asr`; Vulkan0 OK; CPU OK |
| Không còn patch require-gpu trong binary | quét chuỗi trong 5 DLL | sạch ở cả 5 |
| **ASR thật end-to-end trên bundle mới** | `python scratch/verify_cuda_bundle_asr.py` | bundle `bin/` · backend chọn `cuda` · model nạp `backend=CUDA0` · **196 ms cho 4,76 s audio (RTF 0,041)** · transcript tiếng Nga đúng |
| ASR benchmark (tầng B) | `pytest -m slow backend/tests/test_03_asr_benchmark.py` | **2 passed**; `report/03_asr/report.md` được sinh lại trên model Q4_K_M + bundle mới: RTF 0,0096 (file 332 s), 0,0235 (file nhiễu 28 s)… |
| Tầng A | `pytest backend/tests -q` | **391 passed · 0 failed** |

Log đầy đủ: `scratch/f8_tierA.log`, `scratch/f8_asr.log`, `scratch/build_cuda_final.log`.

---

## 6. Phòng ngừa tái diễn: pip lại hạ torch CUDA (2026-09-22)

**Chuỗi sự kiện**: buổi sáng chủ dự án nâng `silero-vad` lên **6.2.2**; sau đó `torch` thành
`2.9.1+cpu` (thư mục torch ghi 07:52) và CUDA biến mất.

**Nguyên nhân thật** (kiểm bằng `pip show` + `pip check`):

* metadata của `silero-vad 6.2.2`: `Requires: packaging, torch, torchaudio` — **không giới hạn
  phiên bản**, nên nó KHÔNG ép hạ torch;
* interpreter Python này **dùng chung** với các gói có ràng buộc torch xung đột:
  `whisperx 3.8.6` → `torch~=2.8.0` · `compressed-tensors 0.18.0` → `torch>=2.10.0` ·
  `torchvision 0.27.0+cu130` → `torch==2.12.0`;
* giao của ba ràng buộc **rỗng** ⇒ pip chọn bản torch mới nhất thoả `whisperx` **từ PyPI, nơi
  torch là bản CPU-only**, kéo theo `torchaudio 2.9.1+cpu` và để `torchvision` cu130 mồ côi.

**Đã thêm 4 lớp chặn:**

| Lớp | Artifact | Tác dụng |
|---|---|---|
| Ghim phiên bản | `backend/constraints.txt` | `pip install … -c backend/constraints.txt` ⇒ pip **báo lỗi `ResolutionImpossible`** thay vì hạ torch |
| Công cụ kiểm tra | `backend/utils/env_check.py` (`python -m backend.utils.env_check`) | Phát hiện torch CPU-only, lệch torchaudio, torchvision CUDA đi kèm torch CPU, module cài-mà-hỏng; exit 1 khi có vấn đề |
| Cảnh báo lúc khởi động | `main._log_torch_status_at_startup()` trong `lifespan` | Biến sự cố im lặng thành WARNING đọc được ngay |
| Quan sát từ xa | `GET /health → "torch": {version, cuda_build, cuda_available, device_count, problems}` | Xác nhận môi trường không cần mở log |
| Test khoá hành vi | `backend/tests/test_45_torch_env_guard.py` (9 test, tầng A) | Không báo động giả khi bộ ba khớp; bắt đúng từng kiểu lệch; khoá cả `constraints.txt` và `/health` |

**Hướng dẫn cho người dùng** (đã ghi vào README, mục *“Nâng cấp gói phụ thuộc mà không phá torch CUDA”*):
(1) interpreter riêng cho dự án (`py -3.13 -m venv .venv`) — khuyến nghị nhất; (2) cài gói "lá"
bằng `pip install -U <gói> --no-deps`; (3) luôn kèm `-c backend/constraints.txt`; (4) chạy
`python -m backend.utils.env_check` sau mỗi lần cài; kèm lệnh cài lại bộ CUDA nếu đã lỡ bị hạ.

**Đính chính phiên bản ghim (14:0x cùng ngày).** Lệnh khôi phục đầu tiên bị pip từ chối:
`torchaudio==2.12.0+cu130` **không tồn tại**. Truy vấn index thật:

```
pip index versions torch      --index-url https://download.pytorch.org/whl/cu130 → tới 2.14.0+cu130
pip index versions torchaudio --index-url https://download.pytorch.org/whl/cu130 → tới 2.11.0+cu130
pip index versions torchvision--index-url https://download.pytorch.org/whl/cu130 → tới 0.29.0+cu130
```

⇒ **torchaudio đi sau torch một bậc**; bộ hợp lệ quanh torch 2.12 là
`torch 2.12.0+cu130 · torchaudio 2.11.0+cu130 · torchvision 0.27.0+cu130` (đúng bộ đã có lúc đầu
phiên). Đã sửa `backend/constraints.txt`; dry-run xác nhận:
`Would install torch-2.12.0+cu130 torchaudio-2.11.0+cu130` (torchvision đã đủ điều kiện).

**Bài học kèm theo**: so *số phiên bản* giữa torch và torchaudio là SAI. `env_check.py` nay so
theo **tag CUDA** (`+cu130`) và thêm test chống hồi quy
`test_khac_so_phien_ban_nhung_cung_tag_cuda_thi_khong_bao_loi` — nếu không, chính bộ đúng sẽ bị
báo lỗi oan.

Trạng thái hiện tại của máy: torch **vẫn là 2.9.1+cpu** (tôi không tự ý cài lại vì cần tải ~2,5 GB);
`env_check` đang báo đúng 1 vấn đề và `bin/ggml-cuda.dll` vẫn chạy CUDA nhờ runtime CUDA 13 trong
repo (xem §3).

---

## 5. Việc phát sinh: `models.yaml` đổi quant ⇒ 5 test của `test_40` đỏ

`backend/models.yaml` (và `backend/translation_models.yaml`) đã được đổi sang **Q4_K_M**
(mtime 10:59 cùng ngày) — không phải do lượt làm việc này. Vì vậy `test_40_asr_model_download.py`
(vốn hard-code `whisper-large-v3-turbo-Q8_0.gguf`) báo 5 lỗi.

Đã sửa theo hướng **không hard-code quant nữa**: các test lấy tên file từ chính catalog
(`ModelRegistry.file_name(key)`) nên lần sau đổi quant sẽ không làm đỏ test oan.
Đồng thời phát hiện `bin/` có thêm file lạ tên `phai` ở thư mục gốc repo (không phải do lượt này).

---

## 7. Venv mới: `import llama_cpp` nổ vì lệch MAJOR CUDA (14:22 cùng ngày)

**Triệu chứng** (người dùng chạy `python backend\main.py` trong `.venv` mới):

```
RuntimeError: Failed to load shared library
  '...\.venv\Lib\site-packages\llama_cpp\lib\llama.dll':
  Could not find module ... (or one of its dependencies)
```

**Chẩn đoán** (`dumpbin /dependents` với MSVC 14.51):

| DLL | Phụ thuộc CUDA |
|---|---|
| `llama.dll` | `ggml.dll`, `ggml-base.dll`, MSVC runtime (không có CUDA trực tiếp) |
| `ggml.dll` | `ggml-cpu.dll`, **`ggml-cuda.dll`**, `ggml-base.dll` |
| **`ggml-cuda.dll`** | **`cudart64_12.dll`**, **`cublas64_12.dll`**, `nvcuda.dll` |

Venv có `torch 2.9.1+cu130` ⇒ `torch/lib` chỉ có `cudart64_13.dll`/`cublas64_13.dll`
(**CUDA 13**), còn wheel `llama_cpp_python 0.3.35` được cài từ
`abetlen.github.io/llama-cpp-python/whl/**cu124**` (do `backend/requirements.txt` trỏ vào đó)
nên cần runtime **CUDA 12**. Ở interpreter toàn cục trước đây nó chạy được là nhờ có sẵn
`nvidia-cuda-runtime-cu12` + `nvidia-cublas-cu12` (wheels CUDA 12) — venv sạch không có.

**Đã sửa:**

| Việc | Chi tiết |
|---|---|
| `backend/requirements.txt` | Đổi index llama.cpp `whl/cu124` → **`whl/cu130`** (khớp torch cu130), kèm chú thích luật "llama.cpp phải cùng major CUDA với torch" + cách xử lý khi buộc dùng cu124 (`nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`) |
| `backend/utils/env_check.py` | Thêm `check_llama_cpp()`: thử `import llama_cpp`, nếu nổ thì báo **kèm lệnh sửa** (đã kiểm trong chính venv của người dùng: in đúng lỗi + gợi ý cu130) |
| `backend/utils/cuda.py` | Sửa deprecation: torch ≥ 2.12 dùng `PYTORCH_ALLOC_CONF` (biến cũ gây `Warning: PYTORCH_CUDA_ALLOC_CONF is deprecated`), torch cũ hơn vẫn dùng tên cũ — chọn theo `importlib.metadata` để không phải import torch |
| `README.md` | Bước 3 dùng index cu130 + cảnh báo lệch major; Bước 2 thêm nhắc đổi cả llama.cpp khi hạ torch; mục Khắc phục sự cố nêu **hai** nguyên nhân của lỗi `llama.dll` |
| `backend/tests/test_45_torch_env_guard.py` | +4 test: llama.cpp lỗi kèm gợi ý, chưa cài thì bỏ qua, `check_environment()` gộp cả hai nguồn, và tên biến allocator theo phiên bản torch (tổng **15 test**) |

**Việc người dùng cần chạy** (221 MB, một lần):

```powershell
.\.venv\Scripts\Activate.ps1
pip install --force-reinstall --no-deps "llama-cpp-python==0.3.35" `
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
python -m backend.utils.env_check     # kỳ vọng: llama_cpp OK
```

> ⚠️ **Kết luận ở trên đã bị THAY THẾ bởi §8** — wheel cu130 0.3.35 không dùng được trên CPU này.

---

## 8. Wheel llama.cpp mới build kèm AVX-512 ⇒ `0xC000001D` (14:2x–14:4x cùng ngày)

**Chuỗi sự kiện**: người dùng chạy lệnh ở §7 nhưng pip tải hỏng (`0 bytes … wheel is invalid`).
Tôi tải lại bằng `curl -L --retry 10 --continue-at -` (thành công, 210,9 MB; `zipfile.testzip()`
OK), cài vào `.venv`, `import llama_cpp` chạy và `GPU offload: True`, **nhưng nạp model thì nổ**:

```
llama_cpp._internals → llama_cpp.llama_init_from_model(...)
OSError: [WinError -1073741795] Windows Error 0xc000001d
```

**Cô lập** (`scratch/isolate_llama_crash.py`, gọi `setup_cuda_dll_paths()` trước, cùng script
cho cả hai interpreter):

| Interpreter | llama.cpp | n_gpu_layers=0 (CPU) | n_gpu_layers=-1 (GPU) |
|---|---|---|---|
| `.venv` | **0.3.35 + cu130** | **FAIL 0xC000001D** | **FAIL 0xC000001D** |
| global | 0.3.22 + cu124 | OK | OK |

⇒ Không phải CUDA, không phải DLL của dự án: nhánh **CPU** cũng chết, và chỉ khác nhau ở wheel.

**Nguyên nhân xác định bằng disassembly** (`dumpbin /disasm …\ggml-cpu.dll | findstr zmm`):

| Wheel | Số lệnh dùng `zmm` (AVX-512) |
|---|---|
| 0.3.35 cu130 | **7.177** |
| 0.3.22 cu124 | **0** |

CPU máy này: **AMD Ryzen 5 5600X (Zen 3) — KHÔNG có AVX-512** ⇒ gặp `zmm` là
`STATUS_ILLEGAL_INSTRUCTION`. Đây là lỗi của wheel dựng sẵn (build bằng `-march=native`/AVX-512
trên máy CI), không phải của dự án.

**Đã sửa (venv đang chạy tốt)**:
1. `pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12` (runtime CUDA 12 cho wheel cu124).
2. Tải `llama_cpp_python-0.3.22-py3-none-win_amd64.whl` (cu124, 433 MB) bằng curl → cài vào `.venv`.
3. Kiểm chứng: `n_gpu_layers=0` **OK**, `n_gpu_layers=-1` **OK**;
   `scratch/verify_llama_cuda_venv.py` → nạp model 2,2 s, dịch 1 câu **696 ms**:
   *"Hello, how are you today?"* → *"Xin chào, hôm nay bạn khỏe không?"*;
   `.venv\Scripts\python.exe -m backend.utils.env_check` → **OK, exit 0**;
   `import backend.main` (chuỗi từng lỗi) → **OK**.

**Sửa trong repo để không lặp lại**:

| File | Thay đổi |
|---|---|
| `backend/requirements.txt` | Index llama.cpp về **cu124**, ghim **`llama-cpp-python==0.3.22`**, kèm khối giải thích AVX-512 + cách kiểm wheel bằng `dumpbin … findstr zmm` + đường build từ source (`-DGGML_NATIVE=OFF`) nếu muốn cu130 |
| `backend/constraints.txt` | Ghim thêm `llama-cpp-python==0.3.22` (lý do AVX-512, tách khỏi nhóm CUDA) |
| `backend/utils/env_check.py` | `check_llama_cpp()` nay gọi `setup_cuda_dll_paths()` **trước** khi import (trước đó báo lỗi sai trong venv dù backend chạy được) + mô tả cả hai lớp lỗi (thiếu runtime CUDA, và wheel AVX-512 ở bước nạp model) |
| `README.md` | Bước 3 dùng bộ đã kiểm (0.3.22-cu124 + runtime cu12) + cảnh báo AVX-512 + lệnh `dumpbin`; mục Khắc phục sự cố thêm dòng `0xc000001d` |
| `scratch/llama_cu124/*.whl` | Giữ wheel 0.3.22 (đã kiểm) để cài lại offline; wheel cu130 hỏng đã xoá |

**Bài học**: (1) không tin wheel dựng sẵn chỉ theo số phiên bản — kiểm tập lệnh CPU bằng
`dumpbin … findstr zmm`; (2) `pip` tải từ GitHub Releases có thể ra file 0 byte — dùng
`curl --retry --continue-at` rồi kiểm `zipfile.testzip()`.

---

## 9. Venv khởi động backend: hai gói còn thiếu (14:40–14:45 cùng ngày)

Sau §7–§8, `python backend\main.py` trong `.venv` chạy được nhưng log còn hai lỗi:

```
[ASR] ASR backend: không backend nào trong ('cuda','vulkan') khả dụng … devices=n/a
[ASR] Nạp trước/pre-warm model thất bại: transcribe_cpp package chưa được cài đặt!
[VAD] Engine 'silero-vad' nạp nền thất bại: ModuleNotFoundError: No module named 'onnxruntime'
```

**Nguyên nhân** (đều là thiếu gói khai báo, không phải lỗi mã):

| Thiếu | Hệ quả | Vì sao |
|---|---|---|
| `transcribe-cpp` (binding) | Không có backend ASR nào (`devices=n/a`) dù `bin/ggml-cuda.dll` còn nguyên | `backend/requirements.txt` chỉ khai `transcribe-cpp-native` (provider) — binding phải cài riêng |
| `onnxruntime` | Engine Silero không nạp được | `silero_vad/utils_vad.py` (silero-vad 6.2.2) `import onnxruntime` **ở cấp module**, dù ta chỉ dùng đường JIT; metadata của gói KHÔNG khai báo |

**Đã sửa và kiểm chứng**:

```powershell
.\.venv\Scripts\python.exe -m pip install "transcribe-cpp>=0.2.3" onnxruntime
```

| Kiểm chứng | Kết quả |
|---|---|
| Log khởi động | **không còn dòng ERROR / "thất bại"**; `Engine 'fsmn-vad' nạp nền xong` + **`Engine 'silero-vad' nạp nền xong`** |
| `GET /health` | `asr_runtime.backend = "CUDA0"`, `available_backends = ["cuda","vulkan"]`, `devices = "cuda=CUDA0, vulkan=Vulkan0, cpu=CPU"`, `provider = "local-bin"`; `torch.problems = []` |
| Dịch (prewarm) | `Model 'tencent' sẵn sàng (GPU, ctx=512, batch=256, threads=4)` |

**Sửa trong repo**: `backend/requirements.txt` thêm `transcribe-cpp>=0.2.3` + `onnxruntime>=1.17`
(kèm ghi chú lý do); `backend/utils/env_check.py` thêm `check_asr_binding()` và
`check_silero_onnxruntime()` (báo đúng hai lỗi này kèm lệnh sửa); README Bước 4/5 + 3 dòng
Khắc phục sự cố; `test_45` thêm 3 test (tổng **18**).

Ngoài ra, kiểm tra lại thì **interpreter global đã được cài lại đúng bộ CUDA**:
`torch 2.12.0+cu130 · torchaudio 2.11.0+cu130 · torchvision 0.27.0+cu130`, `cuda available=True`. Ghi chú vận hành:
- Instance backend cũ (mở lúc 14:40) giữ cổng 8765 làm lần chạy kiểm tra đầu tiên báo
  `[Errno 10048] only one usage of each socket address` ⇒ đã tắt và chạy lại thành công; đã thêm
  dòng khắc phục cho lỗi này vào README.
- `llama_context: n_ctx_seq (512) < n_ctx_train (262144)` trong log là **bình thường** (dự án cố ý
  đặt `n_ctx=512`, xem `TranslationConfig.n_ctx`).
- `[WARNING] trust_remote_code: False` từ funasr cũng là thông tin bình thường.
