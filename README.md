# Vibe Translation Addon

[![Python](https://img.shields.io/badge/python-3.10–3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![Extension](https://img.shields.io/badge/Extension-Manifest%20V3-FF7139)](https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions)
[![Runtime](https://img.shields.io/badge/runtime-native%20·%20no%20PyTorch-4B8BBE)](#runtime-toàn-native)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](#giấy-phép)

Phụ đề song ngữ thời gian thực + dịch neural + lồng tiếng, chạy **100% offline** trên trình duyệt.

```text
Audio tab ──> VAD ──> ASR ──> Cắt câu ──> Dịch GGUF ──> Phụ đề + TTS
```

> **Thiết kế cho 1 phiên / 1 video.** ASR, dịch và TTS là singleton; không hỗ trợ nhiều tab song song trên cùng model instance.

---

## Runtime: toàn native

Mọi subsystem chạy bằng C++/GGML hoặc ONNX Runtime. **Không có PyTorch, torchaudio hay transformers.**

| Subsystem | Runtime | Model |
|---|---|---|
| **VAD** | ONNX Runtime (CPU) | FireRed *(mặc định)* / Silero — đều ONNX |
| **ASR** | `transcribe.dll` (transcribe.cpp) | Qwen3-ASR 1.7B / 0.6B — GGUF Q8_0 |
| **Forced Aligner** | `crispasr.dll` (CrispASR) — *tiến trình con* | Qwen3-ForcedAligner-0.6B — GGUF Q4_K |
| **Dịch** | `llama.dll` (llama.cpp) | Index-Translate 2B / 9B — GGUF |
| **TTS** | `omnivoice.dll` (omnivoice.cpp) — *process worker* | OmniVoice GGUF + tokenizer |

`pip install` chỉ còn binding Python mỏng; DLL native nằm sẵn trong `backend/bin/`. Kiểm chứng môi trường sạch bằng `python -m backend.utils.env_check`.

> **Vì sao aligner phải chạy trong tiến trình con:** ba bộ DLL (`backend/bin/`, `bin/llama/`, `bin/crispasr/`) mang ba bản `ggml*.dll` **khác nhau**. Windows phân giải DLL theo **tên module**, nên nạp bộ thứ ba vào cùng tiến trình sẽ làm `transcribe.dll` bind nhầm và chết với `0xc0000139` / `WinError 127`. Chi tiết: `backend/utils/crispasr_native.py`.

---

## Kiến trúc 2 pipeline

```text
PIPELINE A — Real-time Streaming (livestream / fallback)                  độ trễ ~0.8–1.5 s
  Audio Tab ──> /ws ──> VAD + ASR realtime ──> Cắt câu 4 bậc ──> Dịch ──> TTS

PIPELINE B — Lookahead + OFFLINE_BATCH (video VoD / YouTube / phim)      độ trễ 0.0 s
  MSE Interceptor ──> RAM timeline (tới 3 h) ──> Cắt khối 12–30 s tại khoảng lặng thật
    ──> ASR trọn khối ──> ForcedAligner (mốc từng từ) ──> tách CÂU theo dấu câu
    ──> Dịch gộp khối (ngữ cảnh 2 chiều) ──> Phụ đề 0.0 s + TTS khớp cửa sổ
```

| | Pipeline A | Pipeline B |
|---|---|---|
| **Kịch bản** | Livestream, họp online, video không nạp đệm | Video VoD, YouTube, phim, khoá học |
| **Thu âm** | Web Audio Capture từ tab | Chặn phân đoạn MediaSource (MSE) vào RAM |
| **Đầu vào ASR** | Cửa sổ nhỏ theo VAD | **Trọn khối 12–30 s** ⇒ ngữ cảnh đầy đủ, WER/CER thấp hơn |
| **Tách câu** | VAD Silence › Max Duration › Stable Prefix › Timeout Force | Dấu câu của ASR + mốc từng từ của Forced Aligner |
| **Độ trễ** | ~0.8–1.5 s (chờ âm thanh thật) | **0.0 s** — phụ đề hiện ngay lúc nói |
| **Tua (seek)** | Xoá buffer, mất ngữ cảnh | Lấy ngay audio tại mốc tua từ RAM; đổi thế hệ seek nguyên tử |
| **Bật/tắt** | Tắt *Lookahead Video Buffering*, hoặc tự động khi Lookahead không khả dụng | Bật *Lookahead Video Buffering* (**mặc định**) |

**Fallback tự động:** extension luôn thử Pipeline B trước. Nếu trang chặn buffer MSE, không kết nối được `/ws/lookahead`, hoặc backend báo `lookahead_unavailable`, extension tự chuyển sang Pipeline A và ghi log lý do.

### Pipeline B hoạt động thế nào

1. **Cắt khối bằng VAD** — tìm khoảng lặng thật (≥ 1.5 s), đặt điểm cắt ở **giữa** khoảng lặng. Audio gửi vào ASR **nguyên vẹn 100%**, overlap = 0.0 s.
2. **Cổng kiên nhẫn (Patient Gate)** — gom đủ ~30 s mới gửi ASR khi phần đã dịch trước playhead còn dồi dào; kích hoạt khẩn cấp khi playhead sắp đuổi kịp.
3. **ASR trọn khối** — tối đa ngữ cảnh âm học và dấu câu.
4. **Forced Aligner** — gắn mốc từng từ (~30–50 ms) rồi **gắn lại dấu câu** từ văn bản ASR (model aligner bỏ dấu câu khi tokenize).
5. **Tách câu** — câu trọn vẹn (≤ 24 từ / 14 s) là một phụ đề; câu quá dài cắt ở dấu câu hoặc khe âm học.
6. **Dịch gộp khối** — một lượt GPU duy nhất qua JSON, giữ nguyên mốc thời gian.
7. **Phụ đề 0.0 s + TTS** — nén WSOLA vừa cửa sổ, auto-ducking âm lượng video gốc.

**Khi tua video:** client gửi `seek_reset` kèm mốc mới, backend tăng thế hệ seek và reset con trỏ (nguyên tử). Khối đầu tiên sau khi tua là Fast-Bootstrap 2.5–4 s ⇒ phụ đề hiện gần như tức thì. Backend chỉ báo `ready_until_pts` cho **đúng thế hệ seek hiện tại**, và client **tạm dừng video** khi playhead chạm mốc đó ⇒ không bao giờ phát qua vùng chưa xử lý.

**Model cần cho Pipeline B:** `Qwen3-ASR` (chọn ở popup) và `Qwen3-ForcedAligner-0.6B` (tự tải). Thiếu aligner ⇒ backend báo `lookahead_unavailable`, extension chuyển sang Pipeline A — **không chặn khởi động**.

**Hiệu năng đo trên RTX 5060 Ti:** aligner xử lý khối 28.26 s trong **195–307 ms** (RTF 0.0069–0.0109), nạp model 127–839 ms. Tầng phụ đề cho cùng số câu và cùng nội dung so với đường torch trước đây (kỳ vọng vàng ghim trong `backend/tests/test_64_crispasr_aligner.py`), và chính xác hơn ở đơn vị đầu khi có im lặng dài.

---

## Tính năng

- **Phụ đề song ngữ** — bản dịch hiện song song, chốt theo **câu trọn vẹn** (bật/tắt được phụ đề gốc).
- **Tua an toàn tuyệt đối** — audio nằm trong RAM, tua tới/lui đều có ngay; tự tạm dừng video khi chạm mốc chưa xử lý.
- **Dịch theo ngữ cảnh 2 chiều** — 2 câu trước + 1 câu sau cho mỗi câu, vẫn giữ ánh xạ 1:1 với mốc thời gian.
- **Lồng tiếng (TTS)** — OmniVoice GGUF native, voice-cloning, nén WSOLA, auto-ducking; giải phóng VRAM khi tắt.
- **Đổi model nóng** — thay ASR / VAD / model dịch / giọng TTS ngay trong popup, không cần restart.
- **Popup theo ngữ cảnh** — bật Lookahead thì các tuỳ chỉnh chỉ dành cho Pipeline A (VAD, Segmentation…) tự động bị vô hiệu hoá.
- **Hoàn toàn offline** — không telemetry, không API bên ngoài sau khi tải model.

---

## Yêu cầu hệ thống

| | |
|---|---|
| **OS** | Windows 10/11 64-bit |
| **Python** | 3.10 – 3.13 |
| **GPU** | NVIDIA RTX ≥ 8 GB VRAM (khuyến nghị 12–16 GB) |
| **Driver NVIDIA** | ≥ 580 (cu130) hoặc ≥ 550 (cu124) |
| **Trình duyệt** | Firefox (khuyến nghị) hoặc Chrome/Edge |
| **RAM** | ≥ 8 GB trống |
| **Đĩa** | ~15 GB (model + môi trường Python) |

---

## Cài đặt

```powershell
git clone https://github.com/khachuy279/vibe-translation-addon-transcribe_cpp.git
cd vibe-translation-addon-transcribe_cpp

python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt -c backend\constraints.txt
```

> Cần **Visual C++ Redistributable 2015–2022 (x64)** — tải tại [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) nếu chưa có.

Kiểm tra môi trường:

```powershell
python -m backend.utils.env_check
```

Kỳ vọng:

```text
onnxruntime   : 1.30.0  (...)
VAD engine    : firered-vad  (model ONNX đủ: True)
aligner       : crispasr  (sẵn sàng: True)
llama_cpp     : cài rồi
transcribe-cpp: cài rồi
torch         : không có (đúng)
llama_cpp nạp : OK
ASR binding   : OK
```

---

## Quản lý Model

Toàn bộ model cho **VAD, ASR, Forced Aligner, Dịch và TTS** tập trung tại `backend/models/` (đã `.gitignore`). Lần chạy đầu, backend **tự tải** phần còn thiếu:

| # | Subsystem | File | Nguồn | Dung lượng |
|---|---|---|---|---|
| 1 | ASR | `Qwen3-ASR-1.7B-Q8_0.gguf` | `handy-computer/Qwen3-ASR-1.7B-gguf` | ~2.1 GB |
| 2 | Forced Aligner | `Qwen3-ForcedAligner-0.6B-q4_k.gguf` | `cstr/qwen3-forced-aligner-0.6b-GGUF` | ~500 MB |
| 3 | Dịch | `Index-Translate-9B.Q4_K_M.gguf` | `IndexTeam/Index-Translate-9B-GGUF` | ~5.5 GB |
| 4 | VAD | `firered_stream/Stream-VAD/{cmvn.ark, fireredvad_stream_vad_with_cache.onnx}` + `silero_vad.onnx` | FireRedTeam / snakers4 | ~5 MB |
| 5 | TTS | `omnivoice-base-Q8_0.gguf`, `omnivoice-tokenizer-F32.gguf` | `Serveurperso/OmniVoice-GGUF` | ~1.3 GB |

> **Pre-flight verification:** server chỉ thông báo `[STARTUP] Pipeline sẵn sàng phục vụ` khi **tất cả model bắt buộc (VAD, ASR, Dịch, TTS)** đã hiện diện với dung lượng hợp lệ. Nếu thiếu, tiến trình báo lỗi cụ thể và dừng ngay. Riêng Forced Aligner là **cảnh báo**, không chặn khởi động.

### Catalog mở rộng

**ASR** (`backend/models.yaml`) — mặc định `qwen3-asr-1.7b`:

| Key | File GGUF | Nguồn |
|---|---|---|
| `qwen3-asr-1.7b` | `Qwen3-ASR-1.7B-Q8_0.gguf` | `handy-computer/Qwen3-ASR-1.7B-gguf` |
| `qwen3-asr-0.6b` | `Qwen3-ASR-0.6B-Q8_0.gguf` | `handy-computer/Qwen3-ASR-0.6B-gguf` |

**Dịch** (`backend/translation_models.yaml`) — mặc định `index-translate-9b`:

| Key | File GGUF | Dung lượng |
|---|---|---|
| `index-translate-2b` | `Index-Translate-2B.Q8_0.gguf` | ~2.0 GB |
| `index-translate-9b` | `Index-Translate-9B.Q4_K_M.gguf` | ~5.5 GB |

**VAD engine** — cả hai chạy ONNX Runtime, không cần PyTorch:

| Engine | Model | Runtime |
|---|---|---|
| `firered-vad` *(mặc định)* | `firered_stream/Stream-VAD/` | ONNX (CPU) |
| `silero-vad` | `silero_vad.onnx` | ONNX (CPU) |

> **Kiểm chứng ONNX:** cả hai engine dùng model ONNX **chính thức** của upstream và cho **cùng xác suất từng frame** như bản TorchScript cũ — đo trên 3.797 cửa sổ Silero + 12.621 frame FireRed: `max|Δp| ≤ 2e-6`, **0 frame lệch quyết định** (`backend/tests/test_62_vad_onnx_parity.py`). VAD là **tap thụ động**: audio gửi tới ASR vẫn **đúng từng byte** client gửi, không lệch pha, không đứt khoảng (`backend/tests/test_63_vad_audio_integrity.py`).

---

## Khởi chạy

```powershell
python backend\main.py
```

Lần đầu, hệ thống tự tải model còn thiếu và pre-warm toàn pipeline. Khi thấy `Uvicorn running on https://0.0.0.0:8765`, chấp nhận chứng chỉ tự ký (chỉ một lần):

- **Cách 1:** mở `https://localhost:8765/api/metrics/pipeline` → **Advanced** → **Proceed to localhost**.
- **Cách 2 (khuyến nghị cho Chrome/Edge):** đăng ký chứng chỉ vào Trusted Root — Chrome, Edge và toàn hệ thống tự tin tưởng:
  ```cmd
  certutil -user -addstore Root backend\cert.pem
  ```

---

## Cài extension

**Firefox:** `about:debugging` → **Load Temporary Add-on…** → chọn `extension_firefox/manifest.json`.

**Chrome / Edge:** `chrome://extensions` (hoặc `edge://extensions`) → bật *Developer mode* → **Load unpacked** → chọn thư mục `extension_chrome_edge/`.

Sau khi cài: mở trang video (YouTube, Netflix, Coursera…) → bấm icon extension → chọn ngôn ngữ/model → **Bắt đầu dịch**.

---

## ASR Backend (CUDA / Vulkan)

Bundle native (`backend/bin/`) đã có sẵn trong repo — không cần build lại.

| Backend | Ghi chú |
|---|---|
| **CUDA** *(mặc định)* | Nhanh hơn Vulkan ~1.53×, độ chính xác tương đương |
| **Vulkan** *(fallback tự động)* | Hỗ trợ chính thức qua wheel PyPI, chạy được trên mọi máy |

Nếu thiếu bundle CUDA, backend tự chuyển sang Vulkan và ghi log WARNING.

---

## Kiểm thử

```powershell
pytest              # Tầng A — không cần model, ~30 s (590 test)
pytest -m slow      # Tầng B — cần model thật
pytest -m full      # Tầng C — E2E ASR + Dịch + TTS
```

---

## Khắc phục sự cố

| Triệu chứng | Giải pháp |
|---|---|
| Extension báo *Server Offline* | Mở `https://localhost:8765` và chấp nhận chứng chỉ tự ký |
| Dịch rất chậm (> 3 s/câu) | `python -m backend.utils.env_check` — kiểm tra `llama GPU offload: True` |
| `0xc000001d` khi nạp model dịch | `LLAMA_CPP_LIB_PATH` chưa trỏ đúng về `backend/bin/llama/` (DLL bind nhầm bản AVX-512) |
| `0xc0000139` / `WinError 127` | Trộn lẫn hai bộ `ggml*.dll` trong cùng tiến trình — xem mục **Runtime** ở đầu file |
| ASR fallback về Vulkan | Thiếu `backend/bin/ggml-cuda.dll` — hành vi đúng, Vulkan vẫn chạy bình thường |
| Phụ đề trộn sau khi tua video | Reload extension tại `about:debugging` |
| Cổng 8765 bị chiếm | `Get-NetTCPConnection -LocalPort 8765 -State Listen` để tìm tiến trình cũ |
| RAM tăng liên tục | Lỗi phình bộ nhớ trong native layer — báo lại nếu RSS vượt ~2 GB |

> **DLL CrispASR:** `backend/bin/crispasr/ggml-cuda.dll` nặng **149.6 MB**, vượt giới hạn 100 MB/file của GitHub. Nếu thiếu, `backend/utils/crispasr_native.py` tự tải gói `crispasr-windows-x86_64-cuda13-non-cuda.zip` (ghim `v0.8.41`) và xác thực SHA-256 của cả gói lẫn từng DLL. Muốn commit trực tiếp file này thì phải dùng Git LFS: `git lfs track "backend/bin/crispasr/ggml-cuda.dll"`.

---

## Cấu trúc thư mục

```text
vibe-translation-addon-transcribe_cpp/
├── backend/
│   ├── main.py                    # FastAPI app, REST API, WebSocket endpoint
│   ├── config.py                  # Cấu hình tập trung (Pydantic v2)
│   ├── requirements.txt           # Phụ thuộc runtime
│   ├── constraints.txt            # Ghim đúng bộ phiên bản đã đo
│   ├── models.yaml                # Catalog model ASR
│   ├── translation_models.yaml    # Catalog model dịch
│   ├── asr/                       # ASR engine (transcribe.cpp) + CrispASR aligner
│   ├── vad/                       # VADProcessor + engine FireRed / Silero (ONNX)
│   ├── core/                      # Ring buffer, CommitManager, VAD silence, metrics
│   ├── translation/               # llama.cpp GGUF, hotswap, registry, prompts
│   ├── tts/                       # OmniVoice TTS (omnivoice.cpp binding & process worker)
│   ├── ws/                        # WebSocket handler, session, serializers
│   ├── utils/                     # env_check, CUDA DLL paths, crispasr_native, watchdog
│   ├── bin/                       # DLL native ASR (transcribe.dll, ggml*.dll)
│   │   ├── llama/                 # DLL llama.cpp (tách riêng — ggml khác bản)
│   │   ├── omnivoice/             # DLL omnivoice.cpp (tách riêng)
│   │   └── crispasr/              # DLL CrispASR cho Forced Aligner (tách riêng)
│   ├── models/                    # Model tập trung (gitignore)
│   └── tests/                     # Test suite (pytest)
├── extension_firefox/             # WebExtension Manifest V3
├── extension_chrome_edge/         # WebExtension Manifest V3
├── external/                      # Nguồn & toolchain build native (gitignore)
├── report/                        # Báo cáo audit & đo đạc hiệu năng
├── pytest.ini
└── README.md
```

---

## Giấy phép

MIT License. Mọi đóng góp (Pull Request / Issue) đều được hoan nghênh.
