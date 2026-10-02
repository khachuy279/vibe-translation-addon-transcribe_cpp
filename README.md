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
│ PIPELINE A: Real-time Audio Streaming (Live / Fallback)                                 │
│   Audio Tab (Capture) ──> /ws ──> VAD & ASR Realtime ──> Cắt câu 4 bậc ──> Dịch ──> TTS  │
│   (Độ trễ ~1.0s; dùng cho livestream/họp online và làm FALLBACK của Pipeline B)         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PIPELINE B: Lookahead + OFFLINE_BATCH (Video VoD / YouTube / Phim — Độ trễ 0.0s)        │
│   MSE Interceptor ──> RAM timeline (tới 3h) ──> Cắt khối 12-30s tại khoảng lặng THẬT    │
│   ──> Qwen3-ASR (trọn khối) ──> ForcedAligner (mốc từng từ) ──> tách CÂU theo dấu câu    │
│   ──> Dịch (ngữ cảnh 2 chiều) ──> Phụ đề 0.0s + TTS khớp cửa sổ                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### So sánh chi tiết

| Đặc điểm | Pipeline A (Real-time Streaming) | Pipeline B (Lookahead OFFLINE_BATCH) |
|---|---|---|
| **Kịch bản phù hợp** | Livestream, họp online, video không nạp đệm | Video VoD, YouTube, phim dài tập, khoá học |
| **Cơ chế thu âm** | Web Audio Capture từ tab thời gian thực | Chặn phân đoạn MediaSource (MSE) ghi vào RAM |
| **Quản lý âm thanh** | Ring Buffer xoay vòng ngắn hạn | **Persistent RAM Timeline** (lưu trọn vẹn tới 3 giờ video) |
| **Đầu vào ASR** | Cửa sổ nhỏ theo VAD (cắt câu theo CommitManager 4 bậc) | **Trọn khối 12–30 s** ⇒ ASR có ngữ cảnh đầy đủ (WER/CER thấp hơn) |
| **Cách tách câu** | VAD Silence › Max Duration › Stable Prefix › Timeout Force | **Dấu câu của bản phiên âm + mốc từng từ của Forced Aligner** |
| **Độ trễ phụ đề** | ~0.8s – 1.5s (chờ phát âm thanh thật) | **0.0s (Zero Latency)** — phụ đề hiện ngay lúc nói |
| **Độ trễ lồng tiếng** | TTS đọc đuổi theo sau khi dịch xong | **0.0s** — giọng đọc khớp chuẩn xác mốc bắt đầu câu |
| **Khớp thời lượng TTS**| Trực tiếp theo luồng âm thanh | Tự động nén thời gian (WSOLA) vừa vặn cửa sổ phụ đề |
| **Xử lý khi tua (Seek)**| Xoá buffer, mất ngữ cảnh đã phát | **Lấy ngay âm thanh tại mốc tua từ RAM**; đổi thế hệ seek nguyên tử, không phát qua vùng chưa xử lý |
| **Cách chọn chế độ** | Tắt công tắc *Lookahead Video Buffering*, hoặc tự động khi Lookahead **không khả dụng** | Bật công tắc *Lookahead Video Buffering* (**mặc định**) |

> **Tự động dự phòng (fallback):** extension luôn thử Pipeline B trước. Nếu trang không cho chặn
> buffer MSE, không kết nối được `/ws/lookahead`, hoặc backend báo `lookahead_unavailable`
> (thiếu model Qwen3-ASR / Forced Aligner), extension **tự chuyển sang Pipeline A** và ghi log lý do.

---

### Cơ chế hoạt động của Pipeline B — OFFLINE_BATCH (Qwen3-ASR + Forced Aligner)

Pipeline B là tuyến **duy nhất** cho video VoD: nó lưu trọn audio vào RAM, dịch trước 10–15 s và
nhờ vậy khắc phục triệt để vấn đề mất phụ đề khi tua video. Toàn bộ việc nhận dạng chạy ở chế độ
**offline theo khối** (không phải streaming giả lập):

```text
Trình duyệt Video (YouTube / MSE)
  │ (appendBuffer / fetch)
  ▼
[Buffer Interceptor] ── mảnh audio gốc kèm mốc thời gian (videoPts) ──> WebSocket /ws/lookahead
  ▼
[Persistent Audio RAM] ── luồng PCM 16 kHz liên tục theo timeline tuyệt đối (0s → 3h)
  │
  ├── 1. CẮT KHỐI (12–30 s): quét khoảng lặng THẬT trong dải [playhead+12s, playhead+30s]
  │        • ngưỡng lặng TƯƠNG ĐỐI (p90 − 22 dB) ⇒ đúng cho cả audio to/nhỏ
  │        • ranh giới MẠNH ≥ 450 ms cắt ngay; nếu người nói không ngừng thì cắt tại
  │          điểm trũng năng lượng gần độ dài lý tưởng + 1.0 s chồng lấn
  ├── 2. Qwen3-ASR (transcribe.cpp) nhận TRỌN khối ⇒ ngữ cảnh đầy đủ + dấu câu chuẩn (WER/CER thấp)
  ├── 3. Qwen3-ForcedAligner-0.6B gắn mốc TỪNG TỪ (~30–50 ms) và GẮN LẠI dấu câu từ văn bản ASR
  │        (model aligner bỏ hết dấu câu khi tokenize — nếu không gắn lại, phụ đề sẽ cụt và mất dấu)
  ├── 4. TÁCH CÂU: câu trọn vẹn (≤ 24 từ / 8 s) là MỘT phụ đề; câu quá dài cắt ở dấu phẩy/khe âm học;
  │        mảnh cụt ở cuối khối được GIỮ LẠI ghép vào khối sau ⇒ không bao giờ chẻ câu ở ranh giới
  ├── 5. DỊCH từng câu kèm ngữ cảnh 2 chiều (2 câu trước + 1 câu sau) ⇒ giữ ánh xạ 1:1 với mốc thời gian
  └── 6. PHỤ ĐỀ 0.0 s + TTS (nén WSOLA vừa cửa sổ, auto-ducking); tua tới/lui bất kỳ luôn có sẵn audio

Khi người dùng TUA VIDEO (Seek trước / Seek sau):
  1. Extension gửi `seek_reset` kèm mốc mới; backend tăng "thế hệ seek" và reset con trỏ (nguyên tử).
  2. Khối đầu tiên sau khi tua là Fast-Bootstrap 2.5–4 s ⇒ phụ đề hiện gần như tức thì.
  3. RÀNG BUỘC CỨNG: backend chỉ báo `ready_until_pts` cho ĐÚNG thế hệ seek hiện tại; client TẠM DỪNG
     video khi playhead chạm mốc đó ⇒ không bao giờ phát qua vùng chưa được ASR + dịch.
```

1. **Khối dài ⇒ nhận dạng tốt hơn**: ASR chỉ nhận khối 12–30 s liên tục thay vì từng mảnh 4 s, nên
   model có trọn ngữ cảnh âm học và trả về dấu câu chuẩn — đây là lý do WER/CER giảm so với cắt theo VAD.
2. **Mốc thời gian cấp TỪ**: Forced Aligner (NAR, ~35 ms cho 15 s audio, AAS 32–52 ms) gắn mốc từng
   từ, nhờ đó phụ đề khớp khẩu hình và TTS đọc đúng nhịp.
3. **Câu trọn vẹn**: phụ đề được cắt theo dấu câu của bản phiên âm, không cắt giữa cụm từ; phần chồng
   lấn ở ranh giới khối được trừ theo mốc từ **và** theo chuỗi từ nên không lặp/lost chữ.
4. **Dịch theo ngữ cảnh, giữ đúng mốc**: mỗi câu được dịch riêng nhưng prompt có 2 câu trước + 1 câu
   sau ⇒ xưng hô và văn phong nhất quán mà vẫn gán được bản dịch về đúng mốc thời gian (TTS/canvas).
5. **Tua an toàn tuyệt đối**: audio đã nằm trong RAM nên tua lùi/tới đều có ngay; con trỏ xử lý được
   neo lại theo vị trí phát nếu state lệch, và có watchdog chống "kẹt con trỏ" (video chạy mà không
   xử lý gì).
6. **Popup tự khóa các tuỳ chỉnh của Pipeline A**: khi Lookahead đang bật, nhóm *VAD Engine, VAD
   Silence, VAD Threshold, Hold cut, Min Words Filter* và cả mục *Segmentation* bị làm mờ + vô hiệu
   (kèm ghi chú), vì Pipeline B không dùng VAD/CommitManager để cắt câu.

**Model cần cho Pipeline B**: `Qwen3-ASR` (GGUF trong `backend/models/`, chọn ở popup) và
`Qwen3-ForcedAligner-0.6B` (tự động tải về `backend/models/Qwen3-ForcedAligner-0.6B/` trong lần chạy
đầu). Thiếu model ⇒ backend báo `lookahead_unavailable` và extension tự chuyển sang Pipeline A.

---

## Tính năng

- **Phụ đề song ngữ**: bản dịch hiện song song, chốt theo **câu trọn vẹn** (hỗ trợ bật/tắt phụ đề gốc).
- **Pipeline B — Lookahead OFFLINE_BATCH (0.0s Lag)**: Qwen3-ASR nhận trọn khối 12–30 s (ngữ cảnh đầy
  đủ ⇒ WER/CER thấp), Qwen3-ForcedAligner gắn mốc từng từ, tách câu theo dấu câu — mặc định cho video VoD.
- **Lookahead Persistent Audio RAM**: lưu trọn vẹn audio video vào RAM (tới 3 h), dịch trước 10–15 s.
- **Tua an toàn tuyệt đối**: tua tới/lui bất kỳ vị trí nào đều lấy ngay audio từ RAM; đổi thế hệ seek
  nguyên tử và **tạm dừng video khi chạm mốc chưa xử lý** ⇒ không bao giờ phát mà không có phụ đề.
- **Dịch theo ngữ cảnh 2 chiều**: 2 câu trước + 1 câu sau cho mỗi câu, vẫn giữ ánh xạ 1:1 với mốc thời gian.
- **Cắt câu 4 bậc (Pipeline A)**: `VAD_SILENCE` › `MAX_DURATION` › `STABLE_PREFIX` › `TIMEOUT_FORCE` —
  dùng khi tắt Lookahead hoặc khi Lookahead không khả dụng.
- **Đổi model nóng**: thay ASR / VAD / model dịch / giọng TTS ngay trong popup, không cần restart.
- **Lồng tiếng (TTS)**: OmniVoice voice-cloning, nén âm thanh thông minh (WSOLA), auto-ducking âm lượng video gốc.
- **Popup theo ngữ cảnh**: bật Lookahead thì các tuỳ chỉnh chỉ dành cho Pipeline A (VAD, Segmentation…)
  tự động bị vô hiệu hoá; tắt Lookahead thì mở lại.
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

