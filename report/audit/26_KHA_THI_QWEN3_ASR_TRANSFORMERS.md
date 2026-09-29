# 26 — Đánh giá khả thi: thay ASR bằng Qwen3-ASR (transformers) + Qwen3-ForcedAligner

> **Ngày:** 2026-09-28  ·  **Máy đo:** RTX 5060 Ti 16 GB (idle ~1,6 GB), Windows 11
> **Môi trường đo:** `.venv` của dự án — Python 3.13.14, torch 2.9.1+cu130, transformers 5.17.0
> **Audio đo:** `wav_test/English_low_speech_quality_19s.wav` (resample → 16 kHz mono float32)
> **Script:** `scratch/qwen3_hf_probe.py` (HF), `scratch/asr_lat_ab.py` (native), `scratch/qwen3_native_probe.py`
> **Log:** `scratch/qwen3_hf_probe.log`, `scratch/asr_lat_ab_17b.log`, `scratch/qwen3_native_probe.log`

---

## 0. Kết luận ngắn

| Hạng mục kế hoạch | Kết luận | Lý do cốt lõi |
|---|---|---|
| **Qwen3-ForcedAligner-0.6B thay Whisper cho timestamp (SEG)** | ✅ **Nên làm** | Đo thật: 99 ms/lần align (Whisper timer hiện tại 117 ms p50). Căn trên **chính văn bản ASR** ⇒ bỏ được `difflib` căn chéo 2 model. API drop-in với `WhisperTimer.locate_boundary()`. |
| **Viết lại `backend/asr` chỉ dùng Qwen3-ASR HF (`from_pretrained`)** | ⚠️ **Khả thi kỹ thuật nhưng là THOÁI BỘ trên hot path** | Cùng model, cùng audio: native GGUF **171 ms** (6 s) / **298 ms** (15 s) vs HF bf16 **1 095 ms** / **2 002 ms** ⇒ **chậm ~6,5×**; VRAM 2,1 GB → 3,9 GB. |
| **Nạp model "đúng chuẩn" `Qwen3ASRModel.from_pretrained()`** | ❌ **Không cần môi trường conda thứ hai** | transformers 5.17 **đã có sẵn** `qwen3_asr` + `Qwen3ASRForTokenClassification` (forced alignment). Chỉ cần checkpoint `-hf`. |

**Điểm quyết định:** kế hoạch tách được thành 2 phần độc lập. Phần **aligner** rất đáng làm; phần **thay engine ASR** không mang lại lợi ích chất lượng (cùng trọng số) mà lấy đi 6,5× tốc độ.

---

## 1. Phát hiện quan trọng: KHÔNG cần conda env thứ hai

### 1.1. Xung đột phiên bản là thật (đã kiểm chứng)

`qwen-asr==0.0.6` ghim cứng `transformers==4.57.6`, `accelerate==1.12.0`; trong khi TTS
OmniVoice của dự án yêu cầu `transformers>=5.3.0` (`omnivoice-0.2.1.dist-info/METADATA`) và
`backend/tts/engine.py:292` import `transformers.models.higgs_audio_v2_tokenizer`.
Hai ràng buộc này **không thể đồng thời thoả** trong một môi trường.

Kiểm chứng thêm (không chỉ là metadata): import `qwen_asr` trong `.venv` (transformers 5.17.0) **nổ ngay**:

```
File ".../qwen_asr/core/transformers_backend/modeling_qwen3_asr.py", line 986, in Qwen3ASRThinkerTextModel
    @check_model_inputs()
TypeError: check_model_inputs() missing 1 required positional argument: 'func'
```

Sau khi shim `check_model_inputs`, lỗi kế tiếp là `AutoConfig.register("qwen3_asr", …)` →
`ValueError: 'qwen3_asr' is already used by a Transformers config` — tức **transformers 5.17 đã
hỗ trợ native `qwen3_asr`**. ⇒ Hướng đúng là **dùng native, bỏ hẳn gói `qwen-asr`**.

### 1.2. Native transformers 5.17 chạy được cả ASR và Aligner (đã đo)

`.venv/Lib/site-packages/transformers/models/qwen3_asr/` có:
`Qwen3ASRForConditionalGeneration`, **`Qwen3ASRForTokenClassification`** (“token classification
head for timestamp prediction (forced alignment)”), `Qwen3ASRProcessor` với
`apply_transcription_request()`, `prepare_forced_aligner_inputs()`, `split_words_for_alignment()`,
`decode_forced_alignment()`, `FORCED_ALIGNER_LANGUAGES`, `_fix_timestamps()` (LIS, port từ
`qwen_asr`), `timestamp_segment_time=80 ms`.

**Checkpoint phải dùng bản `-hf`** (đã tải về cache, tổng ~5,5 GB):

| Repo | Kích thước | Ghi chú |
|---|---|---|
| `Qwen/Qwen3-ASR-1.7B-hf` | 3,81 GiB | 1 file `model.safetensors`, `chat_template.jinja` |
| `Qwen/Qwen3-ForcedAligner-0.6B-hf` | 1,72 GiB | như trên |
| `Qwen/Qwen3-ASR-0.6B-hf` | 1,47 GiB | (chưa tải) |

Checkpoint gốc **không `-hf`** (bản `qwen-asr` đã tải sẵn) **cũng nạp được native** nhưng phải:
(a) remap 4 quy tắc key (`thinker.audio_tower.proj1/2` → `model.multi_modal_projector.linear_1/2`,
`thinker.model.*` → `model.language_model.*`, `thinker.audio_tower.*`, `thinker.lm_head.*`) —
đo được **missing=0 / unexpected=0**; (b) vá `config.audio_config.output_dim` 3584 → 2048 (giá trị
trong `thinker_config.audio_config`); (c) **`language=None` bị lỗi chat template**
(`continue_final_message is set but the final message does not appear…`) — bản `-hf` không bị.
⇒ **Khuyến nghị: dùng `-hf`.**

---

## 2. Số đo (cùng audio, cùng cửa sổ, cùng máy)

### 2.1. Độ trễ ASR 1.7B — native GGUF vs HF bf16

| Cửa sổ | Native `Qwen3-ASR-1.7B-Q8_0.gguf` (CUDA, có CUDA-graph) | HF `Qwen3-ASR-1.7B-hf` bf16 | Tỉ lệ |
|---|---|---|---|
| 3,0 s | **130,6 ms** | 897 ms | 6,9× |
| 6,0 s | **171,1 ms** | 1 095 ms | 6,4× |
| 10,0 s | **220,7 ms** | 1 426 ms | 6,5× |
| 15,0 s | **298,0 ms** | 2 002 ms | 6,7× |

Văn bản nhận được **giống nhau** (cùng trọng số, Q8_0 vs bf16 không đổi nội dung ở clip này).

### 2.2. VRAM (phân bổ thật của torch)

| Thành phần | VRAM |
|---|---|
| ASR 1.7B-hf bf16 | **3 887 MiB** (~3,9 GB) |
| Aligner 0.6B-hf bf16 | **1 760 MiB** (~1,8 GB) |
| ASR + Aligner cùng nằm trong VRAM | **5 649 MiB** (peak 5 729) |
| *(tham chiếu)* ASR native 1.7B Q8_0 | ~2,1 GB (model 2,18 GB file + workspace) |
| *(tham chiếu)* toàn pipeline hiện tại (ASR + dịch 7B + TTS) | 9 473 MiB (K7: trần 14 GB) |

Thời gian nạp: ASR 3,2 s · Aligner 1,3 s (native: 1,5–5 s).

### 2.3. Aligner — chất lượng & độ trễ

* 10 s audio / 21 từ: **98,9 ms** (p50, 3 lần lặp) — so với `seg.timer_ms` p50 **117 ms** của
  `WhisperTimer` trong phiên thật ⇒ **rẻ hơn một chút và chính xác hơn về bản chất**.
* Mốc trả về là **mức từ/ký tự** (CJK tách từng ký tự), đơn vị **giây**, đã sửa đơn điệu (LIS).
* Bài toán SEG thật (vùng 10 s, văn bản đầy đủ, cắt sau câu đầu):
  `'Okay, Charles.'` → mốc cắt **1,920 s** — khớp đúng `end_time` của từ `Charles` (1,92 s).
* Khác biệt bản chất so với hiện tại: `WhisperTimer` căn **văn bản Qwen3 ↔ segment của Whisper**
  bằng `difflib` rồi **nội suy tuyến tính ký tự trong segment**; aligner căn **chính văn bản Qwen3**
  ⇒ bỏ hẳn sai số do 2 model nhận khác nhau.

---

## 3. Đánh giá chi tiết từng hạng mục

### 3.1. ✅ Aligner thay Whisper timer (SEG) — NÊN LÀM

**Vì sao khả thi:**
* `timer.py::WhisperTimer` là **nơi DUY NHẤT** timestamps đi vào hệ thống; toàn bộ
  `backend/ws/`, `commit_manager`, serializer **không** dùng timestamp (chỉ dùng sample index).
  Aligner là **drop-in** đúng 2 method: `available()` và
  `locate_boundary(pcm_audio, boundary_index, main_text, *, min_confidence, language) -> TimerResult`.
* Suy giảm đã có sẵn: timer miss → `choose_anchor` (khe im lặng VAD dài nhất) → chồng lấn 600 ms.
* Kết luận này **trùng** với `report/audit/21_SEG_VA_FIX_DEDUP_PHU_DE.md` §20 (“KHÔNG port vào
  C++; dùng như sidecar Python”) — nay có số đo xác nhận.

**Chi phí:** +1,8 GB VRAM (thay vì ~0,85 GB của whisper-turbo Q4) ⇒ +~0,9 GB; +torch CUDA context
nếu TTS đang tắt. Không đụng độ trễ ASR.

**Rủi ro & cách xử lý:**
1. **Chỉ 11 ngôn ngữ** (`FORCED_ALIGNER_LANGUAGES`: zh, en, yue, fr, de, it, ja, ko, pt, ru, es).
   ASR hỗ trợ 30 (có **vi**, th, id…). ⇒ Bắt buộc **fallback** về Whisper timer/chồng lấn khi
   ngôn ngữ phiên không nằm trong danh sách (processor **raise `ValueError`** nếu truyền vào).
2. **ASR và Aligner dùng 2 `chat_template` khác nhau** ⇒ phải giữ **2 processor riêng**. Dùng nhầm
   processor của ASR ⇒ input không có token `<timestamp>` ⇒ `decode_forced_alignment` nổ
   `max() iterable argument is empty`. (Đã vấp thật khi đo — ghi lại để người sau không mất thời gian.)
3. Aligner là **singleton** (AGENTS.md §2.1) + chạy trong worker thread của ASR
   (`_seg_base_cut_sample` đã ở executor) ⇒ **không** được gọi trong event loop.
4. Phải vá test: `backend/tests/test_48_seg_timer.py` (monkeypatch `transcribe_cpp.Model` /
   `WhisperOptions`, assert session riêng) và mở rộng `test_40_asr_model_download.py`
   (registry hiện chỉ biết tải **1 file** `hf_hub_download`; aligner cần **snapshot thư mục**).
5. Thêm nhánh cấu hình kiểu `VADConfig.vad_engine` / `TTSConfig.engine`:
   `SegmentationConfig.timer_engine = "whisper" | "qwen3-aligner"`.
6. `timestamp_segment_time = 80 ms` ⇒ mốc bị lượng tử hoá 80 ms (nhỏ hơn ngưỡng `min_gap_ms=150`),
   chấp nhận được.

### 3.2. ⚠️ Thay engine ASR bằng HF transformers — khả thi nhưng không nên làm mặc định

**Khả thi (đã chứng minh chạy được):** nạp `-hf` trong `.venv`, `language=None` (auto) OK,
`prompt=` (hotwords/context) OK, VRAM 3,9 GB, không xung đột dependency.

**Nhưng đánh đổi:**
* **6,5× chậm hơn** ở mọi độ dài cửa sổ. Vòng poll hiện tại: `poll_interval_ms=200`,
  `preview_window_sec=15` ⇒ một preview 15 s mất **2,0 s** thay vì 0,3 s. Nhịp preview thực tế sẽ
  bị đẩy lên ~2 s, GPU gần như bận liên tục, và vì `GpuArbiter` **không thể preempt** kernel đang
  chạy (chỉ trì hoãn việc *khởi động* job dịch/TTS), dịch + TTS sẽ bị đói GPU ⇒ phụ đề tăng
  ~1,5–2 s độ trễ.
* **+1,8 GB VRAM** cho ASR (2,1 → 3,9 GB). Nếu làm cả aligner: +2,7 GB so với hiện tại.
* **Mất đường Vulkan** — `resolve_backend()` hiện có `cuda|vulkan` (không có CPU), tức máy không
  NVIDIA vẫn chạy được ASR; engine torch chỉ có CUDA/CPU.
* **Mất catalog GGUF đa họ model** (`nemotron-3.5-streaming`, `voxtral-mini-4b-realtime`,
  `sensevoice`, `cohere-transcribe`, `whisper`, kotoba…) — tất cả đang chạy qua transcribe.cpp.
* **Không có streaming thật** ở backend transformers: `init_streaming_state` /
  `streaming_transcribe` / `finish_streaming_transcribe` **chỉ có ở backend vLLM** (và không hỗ trợ
  timestamps). vLLM lại đòi `gpu_memory_utilization` 0,65–0,9 ⇒ **12–14 GB** trên card 16 GB ⇒
  không thể sống chung với dịch 7B + TTS. Ngoài ra “streaming” của vLLM vẫn **re-feed toàn bộ audio
  đã nhận và chạy lại `generate()` mỗi chunk** (O(n²), không tái dùng KV) ⇒ không giải quyết bài
  toán chi phí.
* Vòng poll hiện tại **vốn đã** re-decode toàn cửa sổ mỗi lần (qwen3_asr trong transcribe.cpp
  `max_timestamp_kind=NONE`, không có streaming hook, KV bị wipe mỗi `run()`) — nhưng làm việc đó
  bằng ggml CUDA + CUDA graph thì rẻ hơn ~6,5×.

**Lợi ích thật (nếu vẫn muốn):**
* `context`/hotword qua system prompt (native **không** có: `apply_family_invariants()` chỉ bật
  cancellation; không có `prompt`/`initial_text`).
* Dùng trực tiếp finetune HF (`-hf`, `neosophie/Qwen3-ASR-1.7B-JA`, `qwen3-asr-enhanced-v0.1`)
  không cần convert GGUF.
* Bỏ được dây native cho ASR (DLL, PTX JIT, `TRANSCRIBE_LIBRARY`).

**Nếu vẫn triển khai, điều kiện bắt buộc:**
* Chọn **model 0.6B-hf** cho preview (1,47 GB, độ trễ ~1/2,5 so với 1.7B) và có thể giữ 1.7B cho commit.
* Giảm `preview_window_sec` (6–8 s) + tăng `poll_interval_ms` (800–1 000 ms) và **đo lại KPI**
  (`test_08_streaming_latency.py`, `asr.preview_median_ms`, `asr.commit_ms`).
* Giữ native làm đường mặc định/fallback: thêm `ASRConfig.engine = "transcribe_cpp" | "transformers"`
  (hiện **không có** công tắc engine; engine bị hard-wire ở `backend/ws/session.py:208`).
* Bọc `generate()` trong executor (AGENTS.md §2.4) và giữ nguyên seam
  `_run_inference_sync(pcm) -> str` để `backend/tests/fakes.py` + ~12 test tầng A không vỡ.

---

## 4. Bản đồ tích hợp (tóm tắt)

* **Hợp đồng thật** không phải `BaseASREngine` (đã cũ) mà là duck-typing quanh
  `TranscribeEngine`: `feed_audio(audio, ts, vad_state)`, `stream_tokens()`, `set_language()`,
  `reset_stream()`, `cleanup()`, `cancel_inference()`, `update_seg_config/update_sentence_config()`,
  `prepare_model()`, `prewarm()`, `maybe_recycle_native()`, và **`_run_inference_sync(pcm)`**.
* `backend/asr/__init__.py` **dlopen `transcribe.dll` ngay khi import** (`native.bootstrap()`) — engine
  torch nằm trong `backend/asr/` sẽ thừa hưởng phụ thuộc native này (cần làm bootstrap có điều kiện).
* **Không có trần VRAM** ở đâu trong repo: `gpu_scheduler.py` chỉ điều phối *thời điểm*,
  `mem_guard.py` là kill-switch **RAM**; `models.yaml::vram_estimate_mb` chỉ để tham khảo.
  ⇒ Thêm engine torch = thêm một CUDA context **không được hạch toán**.
* Ảnh hưởng test: nhóm A (native bắt buộc) gồm `conftest.py`, `test_03`, `test_07`, `test_08`,
  `test_09`, `test_37`, `test_40`, `test_48`, `test_45`, `test_18`, `test_16`; nhóm B (mock theo
  cấu trúc `TranscribeEngine`) ~12 file.

---

## 5. Đề xuất thứ tự công việc

| # | Việc | Rủi ro | Lợi ích |
|---|---|---|---|
| 1 | Thêm `Qwen3AlignerTimer` (native transformers + `-hf`) thay `WhisperTimer`, **có fallback theo ngôn ngữ**, cờ `segmentation.timer_engine` | Thấp | Mốc cắt chính xác hơn, 99 ms/lần, bỏ căn chéo 2 model |
| 2 | Đo A/B mốc cắt (aligner vs whisper) trên bộ JA30/AVA sẵn có (`scratch/ja_align_test.py` đã có khung) | Thấp | Số liệu quyết định có giữ không |
| 3 | *(tuỳ chọn)* Thêm `ASRConfig.engine` + `TransformersASREngine` **sau feature flag**, chỉ dùng cho chế độ offline/chất lượng cao, mặc định vẫn native | Trung bình | Có hotwords + finetune HF |
| 4 | **Không** gỡ transcribe.cpp khỏi đường mặc định | — | Giữ 6,5× tốc độ, Vulkan, catalog đa model |

---

## 6. Phụ lục — cách tái lập phép đo

```powershell
# 1) tải checkpoint native (một lần, ~5,5 GB)
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-ASR-1.7B-hf'); d('Qwen/Qwen3-ForcedAligner-0.6B-hf')"

# 2) HF bf16: nạp model, độ trễ 3/6/10/15 s, aligner + bài toán SEG
.venv\Scripts\python.exe scratch\qwen3_hf_probe.py

# 3) Native GGUF (cùng audio, cùng cửa sổ) để so sánh
.venv\Scripts\python.exe scratch\asr_lat_ab.py qwen3-asr-1.7b

# 4) (tham khảo) nạp checkpoint KHÔNG -hf bằng native: remap key + vá output_dim
.venv\Scripts\python.exe scratch\qwen3_native_probe.py
```
