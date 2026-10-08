# Vibe Translation Addon

[![Python](https://img.shields.io/badge/python-3.10–3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com/)
[![Extension](https://img.shields.io/badge/Extension-Manifest%20V3-FF7139)](https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions)
[![Runtime](https://img.shields.io/badge/runtime-native%20·%20no%20PyTorch-4B8BBE)](#runtime-toàn-native)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](#giấy-phép)

Phụ đề song ngữ thời gian thực + dịch ngữ cảnh neural (nhận diện người nói) + lồng tiếng, chạy **100% offline** trên trình duyệt.

```text
Audio tab ──> VAD ──> ASR ──> Aligner + Diarization (ai nói lúc nào) ──> Dịch GGUF (phân vai) ──> Phụ đề + TTS
```

> **Thiết kế cho 1 phiên / 1 video.** ASR, dịch và TTS là singleton; không hỗ trợ nhiều tab song song trên cùng model instance.

---

## Runtime: Toàn Native (Không PyTorch)

Mọi subsystem chạy bằng C++/GGML hoặc ONNX Runtime. **Không dùng PyTorch, torchaudio hay transformers.**

| Subsystem | Runtime | Model |
|---|---|---|
| **VAD** | ONNX Runtime (CPU) | FireRed *(mặc định)* / Silero — ONNX |
| **ASR** | `transcribe.dll` (transcribe.cpp) | Qwen3-ASR 1.7B / 0.6B — GGUF Q8_0 |
| **Forced Aligner** | `crispasr.dll` (CrispASR) — *tiến trình con* | Qwen3-ForcedAligner-0.6B — GGUF Q4_K |
| **Diarization** | `audiocpp_cli.exe` (audio.cpp CUDA) — *tiến trình con* | Nemotron-3-Diarization — GGUF BF16 |
| **Dịch** | `llama.dll` (llama.cpp CUDA) | Index-Translate 2B / 9B — GGUF |
| **TTS** | `omnivoice.dll` (omnivoice.cpp) — *process worker* | OmniVoice GGUF + tokenizer |

`pip install` chỉ cài binding Python mỏng. Các DLL/binary cồng kềnh (> 100 MB) như CrispASR hay audio.cpp được **tự động tải** khi chạy lần đầu, không lưu vào Git.

> **Cách ly DLL native:** các bộ DLL (`backend/bin/`, `bin/llama/`, `bin/crispasr/`, `bin/audiocpp/`) mang các bản `ggml*.dll` khác nhau. Việc chạy các module trong tiến trình riêng ngăn chặn triệt để lỗi xung đột nạp DLL (`0xc0000139` / `WinError 127`).

---

## Kiến trúc 2 Pipeline

```text
PIPELINE A — Real-time Streaming (livestream / fallback)                  độ trễ ~0.8–1.5 s
  Audio Tab ──> /ws ──> VAD + ASR realtime ──> Cắt câu 4 bậc ──> Dịch ──> TTS

PIPELINE B — Lookahead + OFFLINE_BATCH (video VoD / YouTube / phim)      độ trễ 0.0 s
  MSE Interceptor ──> RAM timeline (tới 3 h) ──> Cắt khối 12–30 s tại khoảng lặng
    ──> ASR trọn khối ──> ForcedAligner (mốc từng từ) ──> Tách câu
    ──> Nemotron-3 Diarization (gán nhãn speaker 1, speaker 2...)
    ──> Dịch gộp khối JSON (ngữ cảnh vai nói 2 chiều) ──> Phụ đề 0.0 s + TTS
```

| | Pipeline A | Pipeline B |
|---|---|---|
| **Kịch bản** | Livestream, họp online, video không nạp đệm | Video VoD, YouTube, phim, khoá học |
| **Thu âm** | Web Audio Capture từ tab | Chặn phân đoạn MediaSource (MSE) vào RAM |
| **Đầu vào ASR** | Cửa sổ nhỏ theo VAD | **Trọn khối 12–30 s** ⇒ ngữ cảnh đầy đủ, WER/CER thấp hơn |
| **Tách câu** | VAD Silence › Max Duration › Stable Prefix › Timeout Force | Dấu câu của ASR + mốc từng từ của Forced Aligner |
| **Phân vai nói** | Không | **Nemotron-3-Diarization** gán nhãn từng câu nói |
| **Độ trễ** | ~0.8–1.5 s (chờ âm thanh thật) | **0.0 s** — phụ đề hiện ngay lúc nói |
| **Tua (seek)** | Xoá buffer, mất ngữ cảnh | Lấy ngay audio tại mốc tua từ RAM; đổi thế hệ seek nguyên tử |
| **Bật/tắt** | Tắt *Lookahead Video Buffering*, hoặc fallback tự động | Bật *Lookahead Video Buffering* (**mặc định**) |

---

## Tính năng Nổi bật

- **Phụ đề song ngữ 0.0s** — bản dịch hiện song song theo câu trọn vẹn, căn đúng mốc thời gian video.
- **Nhận diện người nói (Speaker Diarization)** — tích hợp SOTA **Nemotron-3-Diarization** (NVIDIA), tự động phân tách tới 8 người nói (`speaker 1`, `speaker 2`...) giúp LLM dịch xưng hô chuẩn xác và giải quyết triệt để lỗi mất chủ ngữ trong hội thoại.
- **Tua an toàn tuyệt đối** — audio nằm trong RAM, tua tới/lui đều có ngay; tự dừng video khi chạm mốc chưa xử lý.
- **Dịch theo ngữ cảnh 2 chiều** — kế thừa câu trước và câu sau, dịch dạng JSON cấu trúc cao cấp.
- **Lồng tiếng (TTS)** — OmniVoice GGUF native, voice-cloning, nén WSOLA, auto-ducking.
- **Đổi model nóng** — thay ASR / VAD / model dịch / giọng TTS ngay trong popup, không cần khởi động lại.
- **Hoàn toàn offline** — không gửi dữ liệu ra ngoài, bảo mật 100%.

---

## Cài đặt & Khởi chạy

### 1. Cài đặt môi trường

```powershell
git clone https://github.com/khachuy279/vibe-translation-addon-transcribe_cpp.git
cd vibe-translation-addon-transcribe_cpp

python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt -c backend\constraints.txt
```

> Cần **Visual C++ Redistributable 2015–2022 (x64)** — tải tại [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe).

Kiểm tra môi trường:
```powershell
python -m backend.utils.env_check
```

### 2. Khởi chạy Backend

```powershell
python backend\main.py
```

*Lần chạy đầu tiên, hệ thống sẽ tự động tải các model và runtime còn thiếu (VAD, ASR, ForcedAligner, Diarization, LLM).*

Khi thấy `Uvicorn running on https://0.0.0.0:8765`, chấp nhận chứng chỉ tự ký (chỉ làm một lần):
```cmd
certutil -user -addstore Root backend\cert.pem
```

### 3. Cài Extension trình duyệt

- **Firefox:** Mở `about:debugging` → **Load Temporary Add-on…** → chọn file `extension_firefox/manifest.json`.
- **Chrome / Edge:** Mở `chrome://extensions` → bật **Developer mode** → **Load unpacked** → chọn thư mục `extension_chrome_edge/`.

> Hai thư mục trên là **output sinh ra tự động** từ `extension_src/`, không phải nơi sửa code.
> `git clone` đã có sẵn bản dựng nên load trực tiếp được ngay. Nếu bạn **sửa code extension**,
> sửa trong `extension_src/` rồi chạy `python tools/build_extensions.py`.

Mở video (YouTube, Netflix...) → bấm icon extension → chọn ngôn ngữ → **Bắt đầu dịch**.

---

## Phát triển Extension (một nguồn cho hai trình duyệt)

`extension_firefox/` và `extension_chrome_edge/` trước đây là hai bản copy tay song song, và
đã **lệch nhau thật** (566 KB trùng lặp; `lib/ws-client.js` + `lib/lookahead-client.js` khác nhau
ở đúng dòng quyết định cách kết nối backend, trong khi test chỉ phủ bản Firefox). Từ nay:

```text
extension_src/                 # NGUỒN DUY NHẤT — sửa code ở đây
  ├── manifest.chrome.json     # manifest theo trình duyệt
  ├── manifest.firefox.json
  └── lib|content|popup|background|tests|icons/…
        │
        │  python tools/build_extensions.py
        ▼
extension_firefox/             # OUTPUT SINH RA (không sửa tay)
extension_chrome_edge/         # OUTPUT SINH RA (không sửa tay)
```

```powershell
python tools\build_extensions.py           # sinh / cập nhật 2 bản
python tools\build_extensions.py --check   # chỉ kiểm tra lệch, không ghi (dùng trong CI)
```

Hai bản chỉ được phép khác nhau ở **đúng 2 file**, cả hai đều do script sinh:

| File | Vì sao buộc phải khác |
|---|---|
| `manifest.json` | Chromium MV3 chỉ cho phép một `background.service_worker`; Firefox MV3 dùng `background.scripts` + gecko id + `match_origin_as_fallback` |
| `lib/browser-config.js` | Cờ `useBridge`: Chromium kill service worker sau ~30s idle ⇒ phải mở WebSocket thẳng; Firefox giữ background script sống ⇒ dùng bridge để vượt CSP/CORS |

Mọi file khác — kể cả `background/service-worker.js` — **giống hệt từng byte**.
Bất biến này được khoá bằng test: `backend/tests/test_67_extension_single_source.py` và
`test_59::test_extension_mirrors_stay_in_sync`.

---

## Quản lý Model

Toàn bộ model tập trung tại `backend/models/` (đã `.gitignore`). Hệ thống tự tải từ HuggingFace khi thiếu:

| Subsystem | Model | Nguồn / Repo | Dung lượng |
|---|---|---|---|
| **ASR** | `Qwen3-ASR-1.7B-Q8_0.gguf` | `handy-computer/Qwen3-ASR-1.7B-gguf` | ~2.1 GB |
| **Forced Aligner** | `Qwen3-ForcedAligner-0.6B-q4_k.gguf` | `cstr/qwen3-forced-aligner-0.6b-GGUF` | ~500 MB |
| **Diarization** | `nemotron-3-diarization-bf16.gguf` | `audio-cpp/Nemotron-3-Diarization-GGUF` | ~190 MB |
| **Dịch** | `Index-Translate-9B.Q4_K_M.gguf` | `IndexTeam/Index-Translate-9B-GGUF` | ~5.5 GB |
| **VAD** | FireRed-VAD + Silero-VAD | FireRedTeam / snakers4 (ONNX) | ~5 MB |
| **TTS** | `omnivoice-base-Q8_0.gguf` | `Serveurperso/OmniVoice-GGUF` | ~1.3 GB |

---

## Cấu trúc thư mục

```text
vibe-translation-addon-transcribe_cpp/
├── backend/
│   ├── main.py                    # FastAPI server & WebSocket endpoints
│   ├── config.py                  # Cấu hình tập trung (Pydantic v2)
│   ├── requirements.txt           # Thư viện phụ thuộc
│   ├── asr/                       # ASR (transcribe.cpp) & CrispASR aligner
│   ├── diarization/               # Speaker Diarization (Nemotron-3 native CUDA)
│   ├── vad/                       # FireRed & Silero VAD (ONNX Runtime)
│   ├── core/                      # Timeline RAM, Chunker, Ring Buffer
│   ├── translation/               # llama.cpp GGUF, prompts lồng speaker, romaji
│   ├── tts/                       # OmniVoice native TTS worker
│   ├── ws/                        # WebSocket Session handlers (Lookahead / Realtime)
│   ├── utils/                     # env_check, logger màu, CUDA DLL helper
│   ├── bin/                       # Runtime DLL native (transcribe, llama, audiocpp)
│   └── models/                    # Lưu trữ weights model (gitignore)
├── extension_src/                 # NGUỒN DUY NHẤT của 2 bản extension (sửa code ở đây)
├── tools/build_extensions.py      # Sinh extension_firefox/ + extension_chrome_edge/
├── extension_firefox/             # Add-on cho Firefox (Manifest V3)      — OUTPUT SINH RA
├── extension_chrome_edge/         # Extension cho Chrome/Edge (Manifest V3) — OUTPUT SINH RA
├── pytest.ini
└── README.md
```

---

## Giấy phép

Phát hành theo [MIT License](LICENSE).
