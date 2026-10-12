# 20 — Pipeline C: MOSS-Transcribe-Diarize (một model thay ba model?)

**Ngày đo:** 2026-10-24 · **Máy:** RTX 5060 Ti 16 GB · **Backend:** CUDA0 · **Runtime:** `transcribe.cpp`

## 0. Câu hỏi

Model được thử là **MOSS-Transcribe-Diarize 0.9B** (OpenMOSS) — bản port GGUF cho `transcribe.cpp`:
[handy-computer/moss-transcribe-diarize-gguf](https://huggingface.co/handy-computer/moss-transcribe-diarize-gguf).
Nó hứa hẹn **thay cả Pipeline B** bằng MỘT model: phiên âm + nhãn người nói + mốc thời gian theo đoạn
(hiện tại Pipeline B cần **ba** model: Qwen3-ASR + Qwen3-ForcedAligner + Nemotron-3-Diarization).

## 1. Đã làm

> ⚠️ **Trạng thái: THÍ NGHIỆM ĐÃ DỪNG (2026-10-24).** Kết luận ở §5: CER quá lớn ⇒ **không thay
> Pipeline B**. Toàn bộ mã đã gỡ khỏi repo (package `backend/moss/`, entry trong `models.yaml`,
> model GGUF 941 MB, test, và các script đo trong `scratch/`). **Chỉ giữ lại báo cáo này** làm hồ sơ
> số đo. Vì vậy mọi đường dẫn script dưới đây chỉ là mô tả phương pháp, không còn file trên đĩa.

| Việc | Cách làm (để tái lập nếu cần) |
|---|---|
| Thêm vào catalog ASR | entry `moss-transcribe-diarize-q8_0` trong `backend/models.yaml` (`family: moss`) — **đã gỡ** |
| Tải GGUF Q8_0 | `hf_hub_download("handy-computer/moss-transcribe-diarize-gguf", "MOSS-Transcribe-Diarize-Q8_0.gguf")` → 986.900.160 byte; SHA-256 khớp **chính xác** LFS oid `2f9039f1…3ce015` |
| Chạy model | `transcribe_cpp.Model(gguf, backend="cuda")` → `model.session().run(pcm, language=None, timestamps="auto", diarize="on")` |
| Harness CER (cửa sổ cứng) | vòng lặp cửa sổ 10/20/30/40 s, audio RAW, nối text, Levenshtein trên chuỗi đã chuẩn hoá |
| Harness CER (cách Pipeline B) | `LookaheadChunker` + `VADSilenceScanner(silero, min_silence_ms=1500, threshold=0.30)`, `max_audio_sec` = 30 / 45 / 400, `overlap_sec=0` |
| Probe diarization | `ffmpeg -i <mp4> -vn -ac 1 -ar 16000` → chạy MOSS `diarize="on"`, in `segments`/`speaker_segments`; đối chiếu `DiarizationService.diarize_chunk` (Nemotron-3) trên cùng audio |

## 2. Hợp đồng với native (hai cái bẫy thật, đã xử lý trong code)

1. **`language`**: bản port chỉ khai `capabilities.languages = ('en','zh')`. Truyền `"auto"`, `"ja"`,
   `"ja-JP"`, `"japanese"` ⇒ `UnsupportedRequest: unsupported language (status 10)`. **Qwen3-ASR cũng
   vậy** với `"auto"` thô; production không dính vì `normalize_language_for_family("auto", …)` trả
   `None` (`backend/asr/adapters.py:49`).
   ⇒ Bài học cho lần tích hợp sau: **không thể dùng chung engine ASR** với Pipeline A/B — phải ép
   ngôn ngữ về `None` (model tự nhận từ audio), nếu không mọi phiên tiếng Nhật sẽ chết.
2. **Thứ tự import**: `backend.asr` phải bootstrap TRƯỚC khi `transcribe_cpp` được dlopen, nếu không
   bundle native trong `backend/bin/` bị bỏ qua và backend rơi về Vulkan (đã gặp thật ở lần chạy đầu:
   log `Backend 'cuda' không khả dụng -> Fallback 'vulkan'`; sau khi sửa thứ tự import thì
   `moss: using cuda backend: CUDA0`).
3. **`diarize="on"` đắt hơn text-only**: cùng file 97,6 s — text-only RTF **0,020** (6,8 s), bật
   diarize + timestamps RTF **0,044** (4,3 s cho 97,6 s nhưng sinh nhiều token hơn trên mỗi giây
   audio… đo ở hai khối lượng khác nhau nên chỉ so trong cùng một cấu hình).

## 3. CER — `wav_test/00_ingress_stream.wav` (332 s, tiếng Nhật, 2 người nói)

Tham chiếu: `00_ingress_stream.txt` (bỏ `[mm:ss]`/`Speaker N:`, bỏ mọi khoảng trắng + dấu câu, NFKC)
⇒ **821 ký tự**. Audio RAW cho cả hai model (không chuẩn hoá RMS, không lọc text).

### 3.1. Cách Pipeline B ĐANG CHẠY (VAD cắt tại khoảng lặng) — **số liệu chính**

Dùng **chính bộ cắt khối của repo** (`LookaheadChunker` + `VADSilenceScanner` silero, khoảng lặng
≥ 1500 ms, `overlap_sec = 0`), đúng như `LookaheadSessionState.init_components()` dựng (xem cách
tái lập ở §1 và §7).

| Trần khối | Số khối | Cắt bằng VAD | MOSS Q8_0 | Qwen3-ASR 1.7B | RTF MOSS / Qwen3 |
|---|---|---|---|---|---|
| **30 s** (yêu cầu đo) | 15 | 12/15 | 14,74 % | **7,43 %** | 0,022 / 0,015 |
| 45 s (mặc định repo) | 11 | 9/11 | 12,91 % | **8,77 %** | 0,021 / 0,015 |
| ~trọn file (1 khối 320 s) | 3 | 1/3 | **11,57 %** | 21,56 % | 0,038 / 0,017 |

Chi tiết lần cắt ở trần 30 s (khối TB 22,1 s, dài nhất 28,83 s): khối 1 và 2 KHÔNG có khoảng lặng
VAD ≥ 1,5 s nên rơi về dự phòng năng lượng (`silence_gap` 340 ms / `min_rms`), 12 khối còn lại cắt
đúng giữa khoảng lặng VAD (1,76 s–11,87 s); khối cuối 1,34 s là `end_of_stream`.

**Đây là kết luận ngược với bản đo cửa sổ cứng bên dưới: Qwen3 thắng rõ (7,43 % vs 14,74 %).**

Vì sao hai model có "cửa sổ tối ưu" NGƯỢC nhau:

* **Qwen3-ASR bị trần token native** (`k_max_new = 256`, xem `config.lookahead.batch_max_audio_sec`):
  khối càng dài càng bị cụt — ở khối 320 s nó chỉ ra 763/821 ký tự (mất ~7 % nội dung) và CER vọt
  lên 21,56 %. Khối ≤ 30 s là đúng vùng thiết kế của nó.
* **MOSS là LLM 0,9B, context tới ~2,9 h**: càng dài càng lợi (14,74 % → 12,91 % → 11,57 %), nhưng
  ngay cả khi được lợi nhất vẫn **kém Qwen3 ở vùng cắt của Pipeline B**.

### 3.2. Đối chứng: cửa sổ CỨNG (cách đo cũ — nay biết là gây nhiễu)

| Cửa sổ | MOSS Q8_0 | Qwen3-ASR 1.7B Q8_0 | MOSS RTF | Qwen3 RTF |
|---|---|---|---|---|
| 10 s | 14,98 % | **10,23 %** | 0,026 | 0,017 |
| 20 s | 10,84 % | **8,89 %** | 0,020 | 0,016 |
| 30 s | **12,79 %** | 16,08 % | 0,020 | 0,014 |
| 40 s | 9,62 % | **8,65 %** | 0,021 | 0,015 |
| **trung bình** | 12,06 % | **10,96 %** | 0,022 | 0,016 |

Bảng này **đảo chiều theo cửa sổ** (30 s: MOSS thắng 3,3 điểm) vì mép cửa sổ cứng cắt giữa từ và
không liên quan tới trần token thật của model ⇒ **không dùng để kết luận**. Giữ lại làm bằng chứng
cho thấy cách chia cửa sổ chi phối kết quả mạnh thế nào (chênh tới ±5 điểm CER trên cùng một model).

### 3.3. Kiểu lỗi KHÁC NHAU (quan trọng hơn con số)

```
Tham chiếu : 得意先への営業回りを終え、夕食を済まし…
MOSS       : 徳井崎への営業回りを終え…          ← 得意先 → 徳井崎 (sai từ)
Qwen3      : 得意先への営業回りを終え…          ✓

Tham chiếu : 私はふすまがあれば別に大丈夫ですよ。
MOSS       : まあ、私は、風間があれば、別に大丈夫ですよ。   ← ふすま → 風間 (sai từ, đổi nghĩa)
Qwen3      : ま、私は襖があれば別に大丈夫ですよ。            ✓

Tham chiếu : 羽田さんのホテル選びのセンス…
MOSS       : 華田さんのホテル選びのセンス…      ← 羽田 → 華田 (sai TÊN RIÊNG)
Qwen3      : 羽田さんのホテル選びのセンス…      ✓
```

MOSS sai theo kiểu **từ vô nghĩa / sai tên riêng** (edit distance 121 so với 61 của Qwen3 ở trần
30 s), còn Qwen3 sai chủ yếu ở dấu câu và filler (`あ、`, `うん、`). Vì CER đếm KÝ TỰ, khoảng cách
7,3 điểm **đánh giá THẤP hơn** mức thiệt hại thật của MOSS với phụ đề (tên riêng sai làm lệch cả
câu dịch).

**Lưu ý về tham chiếu:** transcript của người có chỗ lược filler (`あ、` trước `トラベルダウ`) nên cả
hai model đều bị trừ điểm oan ở vài chỗ; điều này ảnh hưởng như nhau nên không đổi thứ hạng.

* MOSS chạy **10.458 s audio trong một lượt** (`limits.effective_max_audio_ms`), nạp model 0,8 s.
* Cả hai đều ~10 % CER trên liệu này — **xa** con số quảng cáo (MOSS: 5,13 % WER FLEURS-en,
  9,23 % CER FLEURS-zh). Liệu ở đây là hội thoại 2 người, nhiều khoảng lặng, transcript tham chiếu
  của người.

## 4. Diarization — `wav_test/diarization/nemotron3_tts_8_open_voices.mp4` (97,6 s, 8 giọng TTS)

Đây là video demo của NVIDIA: 8 giọng nói xen kẽ **và có đoạn nói chồng tiếng** (tới 4–8 người cùng lúc).

| | MOSS-Transcribe-Diarize | Nemotron-3-Diarization |
|---|---|---|
| Người nói tìm được | **9** (1 người THỪA: `speaker 9` xuất hiện 2 lần trong đoạn chồng tiếng) | **8** (đúng) |
| Đơn vị | 41 đoạn **tuần tự**, mỗi đoạn có text | 31 lượt nói (không có text) |
| Mốc | theo ĐOẠN | theo lượt |
| **Chồng lấn** | **không biểu diễn được** (`has_overlap=False`) | **6,6 s chồng lấn thật** (7 % thời lượng) |
| Thời gian chạy | 4,3 s cho cả 97,6 s (RTF 0,044), MỘT lượt | ~0,6–1,2 s/khối 5–20 s |

MOSS gán đúng tên từng giọng theo kịch bản (khớp danh sách 8 model TTS trong video):

```
[  6.80 →   9.16] speaker 1  I'm Chris, the NemoTron voice from Nvidia.
[ 11.50 →  13.63] speaker 2  I'm Jane, a pocket voice from Q-Tie.
[ 15.93 →  18.33] speaker 3  I'm Serena, the Queen 3 voice.
[ 18.39 →  20.23] speaker 4  Hey, I'm Adam from Kokoro.
[ 23.73 →  26.66] speaker 5  Hello, hello. Chatterbox here, from Resemble AI.
[ 27.93 →  31.03] speaker 6  Hi, I'm John, a dear voice from Nari Labs.
[ 34.40 →  36.43] speaker 7  I'm the Maya 1 voice, from Maya Research.
[ 36.50 →  38.66] speaker 8  Hi all, I'm Piper, from Raspy.
```

**Nhưng chính đoạn chồng tiếng thì hỏng** — nơi 8 giọng nói cùng lúc, MOSS trả về chuỗi đoán:

```
[ 76.93 →  78.33] speaker 1  Here we go, all eight.
[ 78.40 →  78.93] speaker 7  Levergo.            ← rác
[ 79.00 →  79.93] speaker 9  All at once.
[ 79.99 →  81.33] speaker 8  Speaker eight, two. ← rác
```

## 5. Kết luận: có nên thay Pipeline B?

**Chưa — theo số đo trên chính cách cắt khối của Pipeline B, chất lượng sẽ GIẢM.** Bốn rào cản định lượng:

0. **CER gấp đôi ở đúng cấu hình Pipeline B.** VAD cắt tại khoảng lặng, trần 30 s: Qwen3 **7,43 %**
   vs MOSS **14,74 %**. Và kiểu lỗi của MOSS nặng hơn con số: nó sai **từ/tên riêng**
   (得意先→徳井崎, ふすま→風間, 羽田→華田) — tức bản dịch sẽ sai nghĩa, chứ không chỉ sai dấu câu.
   MOSS chỉ hơn khi khối bị kéo dài quá trần token của Qwen3 (>45 s), tức vùng mà Pipeline B **cố ý
   không dùng** vì `k_max_new = 256` của Qwen3-ASR.
1. **Không biểu diễn được nói chồng tiếng.** Cùng file demo, Nemotron-3 chỉ ra **6,6 s chồng lấn**;
   MOSS gộp tất cả vào đoạn tuần tự và còn sinh người nói thứ 9. Với phim/hội thoại có nói chồng,
   đây là mất mát thông tin thật — đúng thứ mà P0.3 của Pipeline B đang tận dụng để prompt dịch
   không bịa chủ ngữ.
2. **Mốc theo ĐOẠN, không theo TỪ.** Toàn bộ tầng ngắt câu/trừ chồng lấn/hiệu chỉnh ranh giới của
   Pipeline B dựa trên mốc từng từ (ForcedAligner) ⇒ không dùng lại nguyên trạng được; phải viết
   tầng mới cho Pipeline C.
3. **Nhãn người nói không bền giữa các khối.** Model đánh số lại theo từng lượt gọi ⇒ cần bản
   `SpeakerRegistry` riêng cho Pipeline C (giống P0.1 của Pipeline B).

**Vậy Pipeline C có chỗ đứng ở đâu?** Ở nơi **không cần 30 s/khối** và **không có nói chồng tiếng**:

* Nội dung một người nói / xen kẽ (podcast, phỏng vấn, khoá học) — nơi 1 model thay 3 là lợi thật:
  ít VRAM, ít mắt xích, text + người nói + mốc **trong một lượt**, RTF 0,02–0,04.
* File dài: MOSS xử lý **tới 2,9 h trong một lượt** (`limits.effective_max_audio_ms` = 10.458 s) và
  CER của nó **giảm** khi khối dài ra (14,74 % @30 s → 11,57 % @320 s) — ngược hẳn Qwen3.
* Nếu chọn Pipeline C, **phải đổi chiến lược cắt khối** (khối dài, không phải ≤30 s) thì mới tận
  dụng được thế mạnh của nó; dùng nguyên chunker của Pipeline B là dùng sai model.

## 6. Việc tiếp theo (nếu muốn đi tiếp)

1. ~~CER cắt khối tại khoảng lặng~~ — **ĐÃ LÀM** (§3.1).
2. Đo CER trên liệu **một người nói** (FLEURS-ja hoặc `wav_test` khác) — nơi MOSS được kỳ vọng thắng.
3. Nếu muốn dùng Pipeline C cho file dài: đo lại với **khối dài** (60–300 s) ở cả hai model để tìm
   điểm giao thoa thật (số ở §3.1 cho thấy giao thoa nằm giữa 45 s và 320 s).
4. Viết bản `SpeakerRegistry` riêng cho Pipeline C + tầng ghép mốc đoạn → phụ đề.
5. WS handler `/ws/moss` nếu muốn dùng thật trong extension.

## 7. Tái lập (không còn script — mô tả đủ để dựng lại)

Các script đo đã bị xoá cùng đợt dọn dẹp; dựng lại theo mô tả sau (đều là hàm có sẵn trong repo):

```python
# 0) Tải model
from huggingface_hub import hf_hub_download
p = hf_hub_download("handy-computer/moss-transcribe-diarize-gguf",
                    "MOSS-Transcribe-Diarize-Q8_0.gguf", local_dir="backend/models")

# 1) Cắt khối ĐÚNG như Pipeline B (xem backend/ws/lookahead_handler.py::init_components)
from backend.core.lookahead_timeline import ContinuousAudioTimeline
from backend.core.vad_silence import VADSilenceScanner
from backend.core.lookahead_chunker import LookaheadChunker
tl = ContinuousAudioTimeline(sample_rate=16000); tl.append(0.0, pcm_16k)
sc = VADSilenceScanner(engine_name="silero-vad", sample_rate=16000,
                       min_silence_ms=1500.0, threshold=0.30)
ck = LookaheadChunker(timeline=tl, sample_rate=16000, min_window_sec=8.0,
                      silence_scanner=sc, vad_silence_ms=1500.0, max_audio_sec=30.0)
chunks, from_pts = [], 0.0
while (ch := ck.next_chunk(from_pts=from_pts, is_stream_end=True)) is not None:
    chunks.append(ch)
    if ch.next_read_pts <= from_pts + 1e-3: break
    from_pts = ch.next_read_pts

# 2) Chạy model (nhớ `import backend.asr` TRƯỚC `import transcribe_cpp` để dùng bundle CUDA)
import transcribe_cpp
m = transcribe_cpp.Model(str(p), backend="cuda"); s = m.session(n_threads=4)
texts = [s.run(c.pcm, language=None, timestamps="none").text for c in chunks]   # MOSS: PHẢI language=None
# Qwen3-ASR: Model("backend/models/Qwen3-ASR-1.7B-Q8_0.gguf") với language=None (auto)

# 3) CER: bỏ [..]/<|..|>, NFKC, bỏ mọi khoảng trắng + dấu câu, rồi Levenshtein / len(thambao)
```

Tham chiếu để so: `wav_test/00_ingress_stream.txt` (bỏ `[mm:ss]` và `Speaker N:`) ⇒ 821 ký tự.

