# Vibe Translation Addon

[![Python](https://img.shields.io/badge/python-3.10–3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688)](https://fastapi.tiangolo.com/)
[![Extension](https://img.shields.io/badge/Extension-Manifest%20V3-FF7139)](https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](#giấy-phép)

Phụ đề song ngữ thời gian thực + dịch neural + lồng tiếng, chạy **100% offline** trên trình duyệt.

```
Audio tab → VAD → ASR (transcribe.cpp) → cắt câu → Dịch GGUF → Phụ đề + TTS
```

> **Thiết kế cho 1 phiên / 1 video.** ASR, dịch và TTS là singleton; không hỗ trợ nhiều tab song song trên cùng model instance.

---

## Tính năng

- **Phụ đề song ngữ**: preview dần, chốt khi hết câu, bản dịch hiện song song.
- **Cắt câu 4 bậc**: `VAD_SILENCE` › `MAX_DURATION` › `STABLE_PREFIX` › `TIMEOUT_FORCE`.
- **Đổi model nóng**: thay ASR / VAD / model dịch / giọng TTS ngay trong popup, không cần restart.
- **Lồng tiếng (TTS)**: OmniVoice voice-cloning, auto-ducking âm lượng video gốc.
- **Seek an toàn**: extension phát hiện tua video, backend xoá trạng thái cũ ngay lập tức.
- **Hoàn toàn offline**: không telemetry, không API bên ngoài sau khi tải model.

---

## Yêu cầu hệ thống

| | |
|---|---|
| **OS** | Windows 10/11 64-bit |
| **Python** | 3.10 – 3.13 |
| **GPU** | NVIDIA RTX ≥ 6 GB VRAM (khuyến nghị 12–16 GB) |
| **Driver NVIDIA** | ≥ 580 (cu130) hoặc ≥ 550 (cu124) |
| **Trình duyệt** | Firefox (khuyến nghị) hoặc Chrome/Edge |
| **RAM** | ≥ 8 GB trống |
| **Đĩa** | ~15 GB (model + môi trường Python) |

---

## Cài đặt

### 1. Clone và tạo môi trường

```powershell
git clone https://github.com/khachuy279/vibe-translation-addon-transcribe_cpp.git
cd vibe-translation-addon-transcribe_cpp

python -m venv .venv
.venv\Scripts\activate
```

### 2. Cài thư viện

```powershell
pip install -r backend\requirements.txt -c backend\constraints.txt
```

`requirements.txt` đã khai báo sẵn `--extra-index-url` của PyTorch CUDA. Tùy chọn `-c` ghim phiên bản torch để tránh pip tự hạ xuống bản CPU-only.

> **Lưu ý**: Cần có **Visual C++ Redistributable 2015–2022 (x64)**. Tải tại [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) nếu chưa có.

### 3. Kiểm tra môi trường

```powershell
python -m backend.utils.env_check
```

Kỳ vọng: `torch CUDA: OK`, `llama GPU offload: True`, ASR backends gồm `cuda` hoặc `vulkan`.

---

## Chuẩn bị model

Tất cả model đặt trong `backend/models/` (bị `.gitignore`, không kèm trong repo).

**ASR** — phải copy thủ công vào `backend/models/`:

| Key | File | Nguồn (HuggingFace) |
|---|---|---|
| `qwen3-asr-1.7b` *(mặc định)* | `Qwen3-ASR-1.7B-Q8_0.gguf` | `handy-computer/Qwen3-ASR-1.7B-gguf` |
| `qwen3-asr-0.6b` | `Qwen3-ASR-0.6B-Q8_0.gguf` | `handy-computer/Qwen3-ASR-0.6B-gguf` |
| `sensevoice-small` | `SenseVoiceSmall-F32.gguf` | `handy-computer/SenseVoiceSmall-gguf` |
| `cohere-transcribe` | `cohere-transcribe-03-2026-Q8_0.gguf` | `handy-computer/cohere-transcribe-03-2026-gguf` |
| `voxtral-mini-4b-realtime` | `Voxtral-Mini-4B-Realtime-2602-Q5_K_M.gguf` | `handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf` |

**Dịch** — tự tải khi chọn trong popup (hoặc copy thủ công vào `backend/models/`):

| Key | File | Ghi chú |
|---|---|---|
| `tencent` *(mặc định)* | `Hy-MT2-7B-UD-Q4_K_XL.gguf` | ~4,6 GB, chất lượng cao |
| `tencent-1.8b` | `Hy-MT2-1.8B-UD-Q8_K_XL.gguf` | ~2,0 GB, siêu nhanh |
| `xiaomi` | `MiLMMT-46-4B-v1.0.Q4_K_M.gguf` | ~2,5 GB |
| `gemmax` | `GemmaX2-28-9B-v0.2.i1-Q4_K_M.gguf` | ~5,8 GB, hỗ trợ 28 ngôn ngữ |

**VAD** — tự tải lần đầu khi chọn engine (cần mạng một lần duy nhất):

| Engine | Nguồn |
|---|---|
| `firered-vad` *(mặc định)* | HuggingFace `FireRedTeam/FireRedVAD` |
| `silero-vad` | Bundle trong gói `silero_vad` |
| `fsmn-vad` | HuggingFace `funasr/fsmn-vad` |

---

## Khởi chạy backend

```powershell
python backend\main.py
```

Lần đầu mất ~47 s để pre-warm ASR + dịch + VAD. Khi thấy `Uvicorn running on https://0.0.0.0:8765`, mở `https://localhost:8765` trong trình duyệt và chấp nhận chứng chỉ tự ký (chỉ cần làm một lần).

---

## Cài extension

**Firefox**: `about:debugging` → **Load Temporary Add-on…** → chọn `extension_firefox/manifest.json`.

**Chrome / Edge**: `chrome://extensions` → bật *Developer mode* → **Load unpacked** → chọn thư mục `extension_chrome_edge/`.

Sau khi cài: mở trang video → bấm icon extension → chọn ngôn ngữ/model → **Bắt đầu dịch**.

---

## ASR Backend

Bundle CUDA (`backend/bin/`) đã có sẵn trong repo — không cần build lại.

| Backend | Ghi chú |
|---|---|
| **CUDA** *(mặc định)* | Nhanh hơn Vulkan ~1,53×, cần `backend/bin/ggml-cuda.dll` |
| **Vulkan** *(fallback tự động)* | Hỗ trợ chính thức qua wheel PyPI, chạy được trên mọi máy |

Nếu thiếu bundle CUDA, backend tự chuyển sang Vulkan và ghi log WARNING.

---

## Kiểm thử

```powershell
# Tầng A — không cần model, ~15 s
pytest

# Tầng B — cần model thật
pytest -m slow
```

---

## Khắc phục sự cố

| Triệu chứng | Giải pháp |
|---|---|
| Extension báo *Server Offline* | Mở `https://localhost:8765` và chấp nhận chứng chỉ tự ký |
| Dịch rất chậm (>3 s/câu) | Chạy `python -m backend.utils.env_check` — kiểm tra `llama GPU offload: True` |
| `0xc000001d` khi nạp model dịch | `LLAMA_CPP_LIB_PATH` chưa trỏ đúng về `backend/bin/llama/` (DLL bị load nhầm bản AVX-512) |
| ASR fallback về Vulkan | Thiếu `backend/bin/ggml-cuda.dll` — hành vi đúng, Vulkan vẫn chạy bình thường |
| Phụ đề trộn sau khi tua video | Reload extension tại `about:debugging` |
| Cổng 8765 bị chiếm | `Get-NetTCPConnection -LocalPort 8765 -State Listen` để tìm tiến trình cũ |
| RAM tăng liên tục | Lỗi phình bộ nhớ trong native layer — báo lại nếu RSS vượt ~2 GB |

---

## Cấu trúc thư mục

```
vibe-translation-addon-transcribe_cpp/
├── backend/
│   ├── main.py                    # FastAPI app, REST API, WebSocket endpoint
│   ├── config.py                  # Cấu hình tập trung (Pydantic v2)
│   ├── requirements.txt           # Phụ thuộc runtime
│   ├── models.yaml                # Catalog model ASR
│   ├── translation_models.yaml    # Catalog model dịch
│   ├── asr/                       # ASR engine (transcribe.cpp wrapper)
│   ├── core/                      # Ring buffer, CommitManager, dedup, metrics
│   ├── vad/                       # VADProcessor + engine FireRed / Silero / FSMN
│   ├── translation/               # GGUF translator, hotswap, registry, prompts
│   ├── tts/                       # OmniVoice TTS
│   ├── ws/                        # WebSocket handler, session, serializers
│   ├── utils/                     # CUDA DLL paths, SSL, env_check, watchdog
│   ├── bin/                       # DLL native ASR (có trong git, không cần build)
│   │   └── llama/                 # DLL llama.cpp CUDA (tách riêng khỏi bin/)
│   ├── models/                    # Model cục bộ — gitignore
│   └── tests/                     # Test suite (pytest)
├── extension_firefox/             # WebExtension Manifest V3
├── external/                      # Submodules: transcribe.cpp, omnivoice.cpp
├── report/                        # Báo cáo audit hiệu năng
├── pytest.ini
└── README.md
```

---

## Giấy phép

MIT License. Mọi đóng góp (Pull Request / Issue) đều được hoan nghênh.

