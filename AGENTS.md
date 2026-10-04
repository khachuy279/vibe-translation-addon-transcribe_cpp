# AGENTS.md — Quy Chuẩn & Hướng Dẫn Phát Triển Cho AI Agent

Tài liệu này cung cấp toàn bộ quy tắc cốt lõi, kiến trúc hệ thống, các ràng buộc bất biến (invariants), và quy trình kiểm thử dành cho Agent hoạt động trên repository **`vibe-translation-addon-transcribe_cpp`**.

---

## 1. Tổng Quan Dự Án

- **Mục tiêu**: Hệ thống phụ đề song ngữ thời gian thực + dịch neural + lồng tiếng (TTS) chạy **100% ngoại tuyến (local inference)** cho trình duyệt (Firefox / Chrome).
- **Luồng dữ liệu**:
  ```text
  Audio Tab (Extension MV3)
    └──> WebSocket v3 (/ws/stream)
          └──> AudioBuffer & VAD (FireRed / Silero / FSMN)
                └──> ASR (transcribe.cpp — CUDA / Vulkan fallback)
                      └──> Phân đoạn câu (CommitManager: VAD_SILENCE > MAX_DURATION > STABLE_PREFIX > TIMEOUT_FORCE)
                            └──> Dịch thuật GGUF (llama-cpp-python CUDA — Hy-MT2 / MiLMMT / GemmaX)
                                  └──> Phụ đề song ngữ + OmniVoice TTS (Auto-ducking âm lượng video)
  ```
- **Nền tảng chính**:
  - Python 3.10 – 3.13 (chuẩn đo lường: 3.13 x64 trên Windows).
  - FastAPI + WebSocket (`backend/main.py`, `backend/ws/`).
  - Native C/C++: `transcribe.cpp` và `llama.cpp` tích hợp qua ctypes/wheels.
  - Extension: WebExtension Manifest V3 (`extension_firefox/`).

---

## 2. Các Ràng Buộc Bất Biến (Crucial Invariants & Non-Negotiables)

Khi đọc, sửa đổi hoặc thêm code, Agent **BẮT BUỘC** tuân thủ các quy tắc sau:

### 2.1. Giới Hạn Phiên Đơn (Singleton Session Invariant)
- **Thiết kế cho 1 phiên / 1 video**: `transcribe.cpp` (0.x) chỉ hỗ trợ **một stream in-flight trên mỗi model** (`external/transcribe.cpp/include/transcribe.h`).
- ASR, dịch thuật và TTS đều là singleton dùng chung. Không bao giờ cố mở nhiều luồng stream đồng thời vào cùng một native model instance mà không nạp model riêng.

### 2.2. Phân Tách Thư Mục DLL Tuyệt Đối (DLL Separation Invariant)
- `backend/bin/` (ASR bundle: `transcribe.dll`, `ggml-*.dll`, `ggml-cuda.dll`).
- `backend/bin/llama/` (llama.cpp CUDA: `llama.dll`, `ggml-cuda.dll`, `mtmd.dll`, ...).
- **CẤM gộp hai thư mục này**: Dù cùng có tên `ggml.dll`, `ggml-cuda.dll` nhưng đây là hai phiên bản ggml **khác nhau hoàn toàn**. Windows phân giải DLL theo tên module trong cùng thư mục, nếu để chung sẽ xung đột entry point gây crash tiến trình với mã `0xc0000139` (`STATUS_ENTRYPOINT_NOT_FOUND`).
- `backend/utils/cuda.py` và `backend/__init__.py` đã cấu hình `LLAMA_CPP_LIB_PATH` và `setup_cuda_dll_paths()` — không được sửa thứ tự import gây mất biến môi trường này.

### 2.3. Bảo Vệ Môi Trường PyTorch CUDA
- PyPI chỉ phân phối PyTorch bản CPU-only. Bản CUDA phải cài từ index của PyTorch (`--extra-index-url https://download.pytorch.org/whl/cu130` hoặc `cu124`).
- **CẤM** chạy `pip install` làm âm thầm hạ PyTorch xuống CPU. Luôn dùng kèm `-c backend/constraints.txt` khi cài đặt hoặc nâng cấp gói.
- Luôn kiểm tra môi trường bằng lệnh: `python -m backend.utils.env_check`.

### 2.4. Không Chặn Async Event Loop (Async Non-Blocking Rule)
- Toàn bộ tác vụ tính toán nặng (ASR decoding, GGUF token generation, VAD model inference, TTS audio synthesis) **tuyệt đối không được chạy đồng bộ** trong asyncio event loop của FastAPI.
- Phải dùng worker thread / `run_in_executor` / GPU scheduler chuyên dụng (`backend/core/gpu_scheduler.py`).
- Hệ thống có watchdog luồng thật (`[STALL WATCHDOG]`) sẽ dump stack của toàn bộ thread nếu event loop bị stall quá ngưỡng.

### 2.5. Toàn Vẹn Tín Hiệu Audio Của VAD (VAD Signal Integrity)
- VAD là một **tap thụ động**: audio truyền từ extension tới VAD và chuyển tiếp sang ASR là byte nguyên bản (16kHz PCM 16-bit mono).
- Không tự ý resample, gain, clip hoặc lượng tử hóa lại luồng byte này trong quá trình chuyển tiếp (đã được khoá bằng test `test_42_vad_signal_integrity.py`).

### 2.6. Xử Lý Tua / Seek Video (Seek Reset Protocol)
- Khi người dùng tua video (`seeking`/`seeked`), extension gửi bản tin WS `reset_stream`.
- Backend phải lập tức xoá buffer audio, reset trạng thái bộ đệm câu và huỷ các frame dịch/ASR cũ đang dở dang để phụ đề không bị ghép lẫn trước và sau khi tua (`test_21_seek_reset.py`).

---

## 3. Cấu Trúc Mã Nguồn

```text
vibe-translation-addon-transcribe_cpp/
├── backend/
│   ├── asr/                 # Engine ASR transcribe.cpp (native wrapper, fallback CUDA/Vulkan)
│   ├── bin/                 # Bundle DLL native ASR (transcribe.dll, ggml-cuda.dll)
│   │   └── llama/           # Bundle DLL native llama.cpp CUDA (tách riêng!)
│   ├── core/                # Audio buffer, commit manager, dedup, GPU scheduler, metrics,
│   │                        # vad_silence.py (dò khoảng lặng bằng VAD để cắt khối lookahead),
│   │                        # lookahead_chunker.py (cắt khối), lookahead_timeline.py
│   ├── segmentation/        # Thuật toán cắt câu (VAD_SILENCE, STABLE_PREFIX, MAX_DURATION)
│   ├── translation/         # Engine dịch GGUF qua llama-cpp-python
│   ├── tts/                 # Engine lồng tiếng OmniVoice
│   ├── vad/                 # Triển khai 3 engine VAD (FireRed, Silero, FSMN)
│   ├── ws/                  # WebSocket connection handler (protocol v3)
│   ├── utils/               # Quản lý CUDA DLL paths, env_check, logger
│   ├── tests/               # 50+ test files (unit, benchmark, regression, integration)
│   ├── config.py            # Cấu hình Pydantic tập trung cho toàn backend
│   ├── models.yaml          # Catalog model ASR
│   ├── translation_models.yaml # Catalog model dịch GGUF
│   └── main.py              # Điểm khởi chạy FastAPI & WebSocket server
├── extension_firefox/       # WebExtension Manifest V3 (popup, content, background)
├── external/                # Submodules hoặc thư viện ngoài (transcribe.cpp)
├── pytest.ini               # Cấu hình markers và addopts của pytest
└── README.md                # Tài liệu tổng thể dự án
```

---

## 4. Chuẩn Quy Ước Mã Nguồn & Logging

1. **Chuẩn Quy Ước Logging (Logging Conventions)**:
   - Sử dụng định dạng prefix rõ ràng theo subsystem: `[STARTUP]`, `[ASR]`, `[TRANSLATE]`, `[VAD]`, `[TTS]`, `[STREAM]`, `[GPU]`, `[STALL WATCHDOG]`.
   - Tránh duplicate log khi frame stream liên tục. Log trạng thái theo chu kỳ hoặc khi có thay đổi trạng thái (state transition). Đã có test kiểm tra `test_19_log_no_duplicate.py` và `test_20_logging_convention.py`.
2. **Quản Lý Bộ Nhớ & Thread Limits**:
   - Luôn tôn trọng `apply_thread_limits()` từ `backend/__init__.py` để tránh bão thread từ OpenMP / BLAS làm nghẽn CPU.
3. **Cấu Hình**:
   - Mọi tham số mới phải được khai báo trong `backend/config.py` với kiểu dữ liệu rõ ràng (dataclass / Pydantic model) và giá trị mặc định an toàn.

---

## 5. Quy Trình Kiểm Thử (Testing Guide)

Toàn bộ test suite nằm trong `backend/tests/`. Cấu hình lọc test mặc định trong `pytest.ini`: `-m "not slow and not full"`.

### Các Lệnh Kiểm Thử Thường Dùng:
- **Kiểm tra môi trường & DLL**:
  ```powershell
  python -m backend.utils.env_check
  python -c "import sys;sys.path.insert(0,'.');from backend.asr import native as N;print('ASR backends:', sorted(N._available_kinds()))"
  ```
- **Chạy toàn bộ unit test nhanh (CI / regression mặc định)**:
  ```powershell
  pytest
  ```
- **Chạy một file test cụ thể**:
  ```powershell
  pytest backend/tests/test_21_seek_reset.py -v
  pytest backend/tests/test_41_vad_engine_contract.py -v
  ```
- **Chạy test cần GPU (không nạp model nặng)**:
  ```powershell
  pytest -m "gpu and not slow and not full"
  ```
- **Chạy test tầng B (nạp model thật)**:
  ```powershell
  pytest -m "slow"
  ```

---

## 6. Hướng Dẫn Khi Thay Đổi Mã Nguồn (Agent Checklist)

Mỗi khi agent thực hiện thay đổi mã nguồn trong dự án:
- [ ] Đảm bảo không sửa đổi đường dẫn DLL hoặc gộp chung `backend/bin/` và `backend/bin/llama/`.
- [ ] Chạy `pytest` để xác nhận không gây regression cho các luồng core (audio buffer, commit manager, vad contract, seek reset).
- [ ] Nếu thay đổi liên quan đến cấu hình, kiểm tra tính tương thích ngược trong `backend/config.py`.
- [ ] Không chèn blocking code vào luồng async WebSocket hay FastAPI routes.
