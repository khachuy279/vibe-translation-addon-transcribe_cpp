---
name: vibe-dev
description: >-
  Workflows, operational runbooks, and debugging procedures for the Vibe Translation Addon project.
  Use this skill whenever the user or agent needs to verify the CUDA/Vulkan environment, run the backend
  server, execute pytest test suites (unit, GPU, VAD, regression), manage models (ASR, translation, VAD),
  or build and debug the browser extension.
---

# Vibe Development & Operations Skill Runbook

Runbook hướng dẫn thao tác vận hành, kiểm thử, chẩn đoán lỗi và phát triển cho hệ thống **Vibe Translation Addon**.

---

## 1. Kiểm Tra Môi Trường & Chẩn Đoán Hệ Thống

Trước khi chạy backend hoặc sau khi cài đặt/nâng cấp bất kỳ thư viện nào, luôn chạy bước chẩn đoán này.

### 1.1. Kiểm tra PyTorch CUDA & GPU Offload
```powershell
# Kích hoạt venv (nếu chưa kích hoạt)
.venv\Scripts\activate

# Kiểm tra tính toàn vẹn của PyTorch CUDA (exit 1 nếu bị hạ xuống CPU)
python -m backend.utils.env_check

# Kiểm tra PyTorch nhận diện GPU
python -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### 1.2. Kiểm tra ASR Native Backends (CUDA vs Vulkan)
```powershell
python -c "import sys; sys.path.insert(0, '.'); from backend.asr import native as N; print('Có sẵn:', sorted(N._available_kinds()))"
```
* **Kỳ vọng**: Trả về `['cuda', 'vulkan']` (bundle trong `backend/bin/` đã sẵn sàng).
* **Nếu chỉ trả về `['vulkan']`**: Backend sẽ tự fallback về Vulkan (chính chủ từ wheel). Nếu cần CUDA, kiểm tra lại sự hiện diện của `backend/bin/ggml-cuda.dll` và `transcribe.dll`.

### 1.3. Kiểm tra llama.cpp GPU Offload (Translation)
```powershell
python -c "import sys; sys.path.insert(0, '.'); from backend.utils.cuda import setup_cuda_dll_paths; setup_cuda_dll_paths(); from llama_cpp import llama_cpp as L; print('llama.cpp GPU offload hỗ trợ:', L.llama_supports_gpu_offload())"
```
* **Kỳ vọng**: In ra `True`. DLL nạp từ `backend/bin/llama/`.

---

## 2. Vận Hành Backend Server

### 2.1. Khởi chạy Server thông thường
```powershell
# Chạy trực tiếp qua Python
python backend/main.py
```
* Server mặc định lắng nghe tại: `http://127.0.0.1:8765` (hoặc HTTPS nếu có cert).
* WebSocket endpoint: `ws://127.0.0.1:8765/ws/stream`

### 2.2. Kiểm tra Health Check & Config
```powershell
# Kiểm tra trạng thái health và backend runtime thực tế
curl http://127.0.0.1:8765/health

# Xem cấu hình hiện tại
curl http://127.0.0.1:8765/api/config
```

---

## 3. Quy Trình Chạy Pytest & Benchmark

Test suite được chia thành nhiều tầng thông qua pytest markers trong `pytest.ini`.

### 3.1. Chạy Unit Tests Nhanh (Regression Suite)
Chạy toàn bộ các bài test mock, kiểm tra luồng core, audio buffer, commit logic, VAD contract mà không nạp model nặng:
```powershell
pytest
```
*(Mặc định `pytest.ini` áp dụng `-m "not slow and not full" -q`)*

### 3.2. Chạy Theo Từng Nhóm Chức Năng Cốt Lõi
- **Kiểm tra VAD Contract & Signal Integrity**:
  ```powershell
  pytest backend/tests/test_41_vad_engine_contract.py backend/tests/test_42_vad_signal_integrity.py -v
  ```
- **Kiểm tra Tua / Seek Video Reset**:
  ```powershell
  pytest backend/tests/test_21_seek_reset.py -v
  ```
- **Kiểm tra Streaming Commit Logic**:
  ```powershell
  pytest backend/tests/test_10_streaming_commit_logic.py -v
  ```
- **Kiểm tra Không Block Loop & Backpressure**:
  ```powershell
  pytest backend/tests/test_17_executor_backpressure.py backend/tests/test_23_backpressure_recovery.py -v
  ```
- **Kiểm tra Chuẩn Logging & Tránh Duplicate Log**:
  ```powershell
  pytest backend/tests/test_19_log_no_duplicate.py backend/tests/test_20_logging_convention.py -v
  ```

### 3.3. Chạy Tests Tầng B & C (Cần GPU hoặc Nạp Model Thật)
- **Tầng B (Cần GPU / Nạp model)**:
  ```powershell
  pytest -m "slow"
  ```
- **Tầng C (E2E full 3 model ASR + Dịch + TTS)**:
  ```powershell
  pytest -m "full"
  ```

---

## 4. Quản Lý Mô Hình (Models Management)

Tất cả model lưu trữ trong `backend/models/`.

### 4.1. Model ASR (`backend/models.yaml`)
- ASR sử dụng định dạng `.gguf` tương thích với `transcribe.cpp`.
- Model ASR **không tự động tải**. Người dùng cần tải file `.gguf` và đặt vào `backend/models/`.
- File mặc định: `Qwen3-ASR-1.7B-Q8_0.gguf`.

### 4.2. Model Dịch Thuật (`backend/translation_models.yaml`)
- Dịch sử dụng GGUF qua `llama-cpp-python`.
- Model dịch có tính năng **tự động tải ở luồng nền** (auto-download) khi được chọn trong popup UI:
  - Mặc định: `tencent` (`Hy-MT2-7B-UD-Q4_K_XL.gguf`, ~4.6 GB).
  - Bản nhẹ: `tencent-1.8b` (`Hy-MT2-1.8B-UD-Q8_K_XL.gguf`, ~2.0 GB).
- Có thể kiểm tra tiến độ tải qua `GET /api/config`.

### 4.3. Model VAD
- **FireRed VAD (mặc định)**: Tự động tải từ Hugging Face vào `backend/models/firered_stream/` trong lần đầu kích hoạt.
- **Silero VAD**: Đi kèm gói `silero_vad`.
- **FSMN VAD**: Tải vào `backend/models/fsmn_vad/`.

---

## 5. Phát Triển & Kiểm Tra Browser Extension

Extension nằm tại thư mục `extension_firefox/` (chuẩn Manifest V3 tương thích Firefox và Chrome/Edge).

### 5.1. Debug trên Firefox
1. Mở Firefox, truy cập: `about:debugging#/runtime/this-firefox`
2. Bấm **"Load Temporary Add-on..."**
3. Chọn file `manifest.json` trong thư mục `extension_firefox/`.

### 5.2. Debug trên Chrome / Edge
1. Truy cập `chrome://extensions/` hoặc `edge://extensions/`
2. Bật chế độ **Developer mode**
3. Bấm **"Load unpacked"** và chọn thư mục `extension_firefox/`.
