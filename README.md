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

## Kiến trúc 2 Pipeline (Pipeline A vs Pipeline B)

Hệ thống hỗ trợ 2 chế độ xử lý linh hoạt tùy theo định dạng video và nhu cầu của người dùng:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PIPELINE A: Real-time Audio Streaming (Live / Livestream / Họp trực tuyến)             │
│   Audio Tab (Capture) ──> /ws ──> VAD & ASR Realtime ──> Cắt câu ──> Dịch ──> Live TTS │
│   (Độ trễ thấp ~1.0s, thích ứng backpressure cho các luồng phát trực tiếp)             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PIPELINE B: Lookahead Persistent Audio RAM (Video VoD / YouTube / Phim — Độ trễ 0.0s)  │
│   MSE Interceptor ──> Ghi RAM timeline (tới 3h) ──> Dịch trước 10-15s ──> Timeline Sync│
│   (Độ trễ 0.0s, lưu toàn bộ audio video vào RAM, tua tới/lui bất kỳ luôn có phụ đề/TTS)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### So sánh chi tiết

| Đặc điểm | Pipeline A (Real-time Streaming) | Pipeline B (Lookahead Persistent Audio RAM) |
|---|---|---|
| **Kịch bản phù hợp** | Livestream, họp online, video không nạp đệm | Video VoD, YouTube, phim dài tập, khoá học |
| **Cơ chế thu âm** | Web Audio Capture từ tab thời gian thực | Chặn phân đoạn MediaSource (MSE) ghi vào RAM |
| **Quản lý âm thanh** | Ring Buffer xoay vòng ngắn hạn | **Persistent RAM Timeline** (lưu trọn vẹn tới 3 giờ video) |
| **Độ trễ phụ đề** | ~0.8s – 1.5s (chờ phát âm thanh thật) | **0.0s (Zero Latency)** — phụ đề hiện ngay lúc nói |
| **Độ trễ lồng tiếng** | TTS đọc đuổi theo sau khi dịch xong | **0.0s** — giọng đọc khớp chuẩn xác mốc bắt đầu câu |
| **Khớp thời lượng TTS**| Trực tiếp theo luồng âm thanh | Tự động nén thời gian (WSOLA) vừa vặn cửa sổ phụ đề |
| **Xử lý khi tua (Seek)**| Xoá buffer, mất ngữ cảnh đã phát | **Lấy ngay âm thanh tại mốc tua từ RAM**, không mất audio |
| **Cách chọn chế độ** | Tắt công tắc *Lookahead Video Buffering* | Bật công tắc *Lookahead Video Buffering* (mặc định) |

---

### Cơ chế hoạt động của Pipeline B (Persistent Audio RAM)

Pipeline B được thiết kế chuyên sâu cho trải nghiệm xem video mượt mà, khắc phục triệt để vấn đề mất phụ đề khi tua video:

```text
Trình duyệt Video (YouTube / MSE)
  │ (appendBuffer / fetch)
  ▼
[Buffer Interceptor]
  │  Mảnh audio gốc kèm mốc thời gian (videoPts)
  ▼  (kết nối trực tiếp WebSocket /ws/lookahead)
[Persistent Audio RAM] ──> Lưu trọn vẹn luồng âm thanh vào RAM theo timeline tuyệt đối (0s -> 3h)
  │
  ├──> Khi Video đang phát bình thường:
  │      Trích xuất audio phía trước [playhead -> playhead + 15s]
  │      ──> VAD (FireRed) ──> ASR (transcribe.cpp) ──> Dịch (GGUF) ──> Lồng tiếng (OmniVoice)
  │      ──> Phụ đề & TTS lên lịch sẵn, xuất hiện đúng 0.0s khi người nói cất tiếng!
  │
  └──> Khi người dùng TUA VIDEO (Seek trước / Seek sau):
         1. Extension gửi tín hiệu `seek_reset` kèm mốc thời gian mới (`target_time`).
         2. Backend huỷ các câu dịch dở dang, lập tức chọn âm thanh tại vị trí tua từ RAM.
         3. Video tạm dừng trong tích tắc để nạp đệm (`prebuffer`).
         4. Ngay khi phụ đề (hoặc TTS) đầu tiên tại mốc tua sẵn sàng, video tự động tiếp tục phát!
         5. Toàn bộ audio đã lưu trong RAM được bảo toàn suốt phiên (chỉ giải phóng khi Stop/tắt tab).
```

1. **Khởi tạo & Ghi liên tục vào RAM**:
   - Khi bắt đầu phiên, backend cấp phát bộ đệm timeline hỗ trợ tới 3 giờ audio (~350 MB RAM tại chuẩn 16kHz PCM 16-bit mono).
   - Mọi phân đoạn audio trình duyệt nạp trước (kể cả khoảng lặng không có tiếng nói) đều được ghi nhận và sắp xếp đúng vị trí thời gian tuyệt đối.
2. **Dịch trước thông minh (Lookahead 10–15s)**:
   - Hệ thống quét trước 10–15 giây so với vị trí phát hiện tại.
   - VAD phát hiện vùng có tiếng nói, ASR giải mã và model GGUF dịch thuật sẵn sàng từ trước.
3. **Tua video tự do không lo mất âm thanh**:
   - Dù nhảy cóc từ phút 02:00 lên 60:00, rồi tua ngược về 30:00, hệ thống luôn có sẵn âm thanh trong RAM để trích xuất và tạo phụ đề tức thì.
   - Tự động bù đắp và đồng bộ lại nếu video tua vào vùng mạng chưa tải kịp.
4. **Đồng bộ hiển thị & Lồng tiếng (TTS Alignment)**:
   - Phụ đề tự động giữ trên màn hình cho tới khi câu lồng tiếng TTS đọc xong, tránh tình trạng phụ đề biến mất trước khi đọc dứt câu.
   - Tự động thích ứng với tốc độ phát của video (1.0x, 1.25x, 1.5x, 2.0x).

---

## Tính năng

- **Phụ đề song ngữ**: preview dần, chốt khi hết câu, bản dịch hiện song song (hỗ trợ bật/tắt phụ đề gốc).
- **Lookahead Persistent Audio RAM (0.0s Lag)**: lưu trọn vẹn audio video vào RAM, dịch trước 10–15s, loại bỏ hoàn toàn độ trễ nhận thức.
- **Cắt câu 4 bậc**: `VAD_SILENCE` › `MAX_DURATION` › `STABLE_PREFIX` › `TIMEOUT_FORCE`.
- **Đổi model nóng**: thay ASR / VAD / model dịch / giọng TTS ngay trong popup, không cần restart.
- **Lồng tiếng (TTS)**: OmniVoice voice-cloning, nén âm thanh thông minh (WSOLA), auto-ducking âm lượng video gốc.
- **Seek an toàn tuyệt đối**: tua tới/lui bất kỳ vị trí nào đều lập tức trích xuất âm thanh từ RAM để nạp đệm và phát lại mượt mà.
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

Lần đầu mất ~47 s để pre-warm ASR + dịch + VAD. Khi thấy `Uvicorn running on https://0.0.0.0:8765`, chấp nhận chứng chỉ SSL tự ký (chỉ cần làm một lần duy nhất):

- **Cách 1 (Nhanh qua trình duyệt)**: Mở `https://localhost:8765/api/metrics/pipeline` trên Firefox / Chrome / Edge → Bấm **Nâng cao (Advanced)** → Bấm **Tiếp tục truy cập (Proceed to localhost)**.
- **Cách 2 (Khuyến nghị cho Chrome/Edge trên Windows)**: Đăng ký chứng chỉ vào Trusted Root của Windows để Chrome, Edge và toàn hệ thống tự động tin tưởng:
  ```cmd
  certutil -user -addstore Root backend\cert.pem
  ```
  *(Bấm **Yes** khi hộp thoại cảnh báo tin cậy chứng chỉ của Windows xuất hiện)*.

---

## Cài extension

**Firefox**: `about:debugging` → **Load Temporary Add-on…** → chọn `extension_firefox/manifest.json`.

**Chrome / Edge**: `chrome://extensions` (hoặc `edge://extensions`) → bật *Developer mode* → **Load unpacked** → chọn thư mục `extension_chrome_edge/`.

Sau khi cài: mở trang video (YouTube, Netflix, Coursera…) → bấm icon extension → chọn ngôn ngữ/model → **Bắt đầu dịch**.

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

