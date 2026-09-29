# 27 — A/B mốc cắt SEG: `WhisperTimer` (đang dùng) vs `Qwen3AlignerTimer` (mới)

> **Ngày đo:** 2026-09-28 · **Máy:** RTX 5060 Ti 16 GB, Windows 11, `.venv` (Python 3.13.14, torch 2.9.1+cu130, transformers 5.17.0)
> **Dữ liệu:** Google FLEURS `ja_jp` (319 câu không trùng trong `wav_test/google_fleurs/ja_jp`)
> **Script:** `scratch/seg_ab_aligner.py` · **Tổng hợp:** `scratch/seg_ab_report.py` · **Soi mẫu sai:** `scratch/seg_ab_debug.py`, `scratch/seg_ab_onset_check.py`
> **JSON thô:** `scratch/seg_ab_aligner_ja_asr.json`, `scratch/seg_ab_aligner_ja_ref.json`
> **Tiền đề:** `report/audit/26_KHA_THI_QWEN3_ASR_TRANSFORMERS.md` (khả thi + số đo nền)

---

## 0. Kết luận

**`Qwen3AlignerTimer` thắng rõ rệt và nên dùng** (đã cài sẵn, mặc định vẫn là whisper — xem §6):

| Chế độ đo | Engine | mean \|Δ\| | p50 | p90 | max | ≤100 ms | cost/lần |
|---|---|---|---|---|---|---|---|
| **Văn bản ASR thật** (63 mẫu) | whisper-segment | 265 ms | 32 | 204 | **12 124** | 44/63 (70 %) | 202 ms |
| | **qwen3-aligner** | **22 ms** | **20** | **36** | **68** | **63/63 (100 %)** | **94 ms** |
| **Văn bản tham chiếu** (80 mẫu) | whisper-segment | 674 ms | 42 | 333 | **19 604** | 46/80 (58 %) | 198 ms |
| | **qwen3-aligner** | **38 ms** | **20** | **36** | **756** | **77/80 (96 %)** | **92 ms** |

* Đối đầu từng mẫu: aligner **tốt hơn 34**, xấu hơn 16, tương đương 13 (chế độ ASR thật).
* Sai số > 500 ms: whisper **1–3 mẫu** (tệ nhất **+19,6 s**), aligner **0–1 mẫu** (tệ nhất **0,76 s**).
* Chi phí: aligner **rẻ hơn 2,15×** (94 ms vs 202 ms cho vùng 9–34 s).

---

## 1. Cách đo (mốc thật độc lập)

1. Lấy từng CẶP câu FLEURS ja (đã lọc câu trùng văn bản), **cắt đúng phần tiếng nói bằng
   Silero VAD** rồi **ghép liền không chèn gì** ⇒ mối ghép **chính là biên câu thật**.
   Silero là nguồn độc lập với cả whisper lẫn aligner.
2. `main_text` = văn bản **ASR thật** của Qwen3-ASR-1.7B (đường native) trên clip đã ghép.
   Chạy thêm biến thể `--no-asr` (dùng văn bản tham chiếu FLEURS) để **cô lập chất lượng timer**
   khỏi sai số ASR.
3. `boundary_index` = vị trí hết câu 1 ánh xạ từ tham chiếu sang `main_text` bằng `difflib`.
   **Cả hai engine nhận ĐÚNG cùng `pcm`, cùng `main_text`, cùng `boundary_index`.**
4. Sai số = `cut_ms − true_ms`. Mẫu bị loại nếu `main_text` không thật sự chứa ranh giới
   (ASR bỏ mất câu 1 hoặc câu 2) — xem `skipped_detail` trong JSON.

**Kiểm chứng mốc thật** (`scratch/seg_ab_onset_check.py`): với các mẫu aligner lệch muộn,
onset theo NĂNG LƯỢNG (nguồn thứ ba) nằm trong **0–20 ms** kể từ mối ghép ⇒ mốc thật đúng,
lỗi là của phương pháp (không phải của thước đo).

---

## 2. Vì sao aligner thắng

| | `WhisperTimer` | `Qwen3AlignerTimer` |
|---|---|---|
| Model | `whisper-large-v3-turbo` (model KHÁC) | `Qwen3-ForcedAligner-0.6B` |
| Văn bản đem căn | văn bản của **Whisper** rồi `difflib` căn chéo sang văn bản Qwen3 | **chính văn bản Qwen3** |
| Độ mịn mốc | mức SEGMENT (whisper) → **nội suy tuyến tính ký tự** trong segment | mức **TỪ/KÝ TỰ** (mỗi từ 2 token `<timestamp>`, bước 80 ms) |
| Nguồn sai số | 2 model nhận khác nhau (difflib neo lệch) + nội suy không theo thời gian thật | chỉ còn sai số của bản thân aligner |
| Hỏng điển hình | neo vào câu SAI ⇒ lệch **hàng giây tới gần 20 s** | lệch muộn ≤ 0,76 s (1/80 mẫu) |

Ba mẫu whisper sai nặng nhất (chế độ tham chiếu) đều **lệch > 12 s** trong khi aligner lệch
≤ 36 ms — đây là hệ quả trực tiếp của việc `difflib` phải ghép hai văn bản do HAI model khác
nhau nhận: chỉ cần whisper nhận sai/thiếu một câu là mối neo rơi sang câu kế tiếp.

---

## 3. Lỗi đã tìm ra và sửa trong lúc A/B (quan trọng)

Bản aligner ĐẦU TIÊN dùng `timer.normalize_for_align()` (bộ lọc `_IGNORED` của
`segmentation.boundary`) để đếm ký tự. Sai, vì **hai tập luật khác nhau**:

* `_IGNORED` **không** chứa `.` `・` `-` `/` `%`… (tokenizer của aligner **BỎ** các ký tự này);
* `_IGNORED` **bỏ** dấu `'` nhưng tokenizer của aligner **GIỮ**.

Lệch 1 ký tự ⇒ mốc cắt bị đẩy lùi đúng 1 đơn vị. Đo được: **+600…+700 ms** ở 8/80 mẫu
(aligner "lệch muộn" hàng loạt), và mẫu #19 lệch hẳn 2 đơn vị (`脳` → `と`).

**Sửa:** `backend/asr/aligner_timer.normalize_for_aligner()` — dùng ĐÚNG tập ký tự của
tokenizer aligner (`unicodedata` category `L*`/`N*` + CJK + `'`) cho **cả hai vế**.

| Chỉ số (80 mẫu, văn bản tham chiếu) | Trước khi sửa | Sau khi sửa |
|---|---|---|
| aligner mean \|Δ\| | 135 ms | **38 ms** |
| aligner p90 | 468 ms | **36 ms** |
| aligner max | 2 772 ms | **756 ms** |
| aligner ≤100 ms | 66/80 (83 %) | **77/80 (96 %)** |

⇒ Đây cũng là bài học cho engine whisper: dùng `_IGNORED` để đếm ký tự là **sai lệch hệ
thống** (xem §7, việc còn lại).

---

## 4. Chi phí "lần đầu" và warmup theo ngôn ngữ

Đo được (`scratch/aligner_warmup_probe.py`):

* Lần align THẬT đầu tiên: **~1,57 s** (có warmup) / **~2,04 s** (không warmup); các lần sau **~0,09 s**.
* Thủ phạm: **tokenizer theo ngôn ngữ** — lần gọi `split_words_for_alignment(..., "Japanese")`
  ĐẦU TIÊN tốn **~1,47 s** (nạp từ điển `nagisa`); tiếng Hàn tương tự với `soynlp`.
  Warmup bằng văn bản tiếng Anh **không** hấp thụ được khoản này.

⇒ `Qwen3AlignerTimer._warmup()` làm ấm **theo đúng ngôn ngữ phiên** (phiên `auto` ⇒ làm ấm
cả `en`/`ja`/`ko`), chạy trong **luồng nền** của `prewarm_seg_timer()`: tốn ~2,1 s một lần lúc
mở phiên, đổi lại lần chốt câu đầu tiên chỉ còn **92 ms** (thay vì 1 570 ms).

⚠️ **Phụ thuộc mới:** `nagisa==0.3.0` + `soynlp==0.0.493` (đã thêm vào `backend/requirements.txt`).
Thiếu `nagisa` ⇒ timer tiếng Nhật **hỏng hoàn toàn** (`ImportError` trong
`split_words_for_alignment`). Cả hai là wheel sẵn cho cp313 win_amd64, cài kèm
`-c backend/constraints.txt` **không** đụng torch (đã kiểm bằng `python -m backend.utils.env_check`).

---

## 5. Phủ ngôn ngữ + fallback (thiết kế chống giảm chất lượng âm thầm)

Aligner chỉ phủ **11/30** ngôn ngữ của ASR: `zh en yue fr de it ja ko pt ru es` —
**KHÔNG có tiếng Việt / Thái / Indonesia**. Vì vậy:

| Tình huống | Hành vi |
|---|---|
| Ngôn ngữ phiên ∈ 11 ngôn ngữ | aligner (chính văn bản ASR) |
| Phiên `auto` + văn bản có kana/hangul/CJK/Cyrillic | aligner, suy ngôn ngữ theo **chữ viết** |
| Phiên `auto` + chữ Latin (không phân biệt được en ↔ vi) | **whisper** (hoặc đặt `segmentation.aligner_language_fallback` để ép, vd `"English"`) |
| Ngôn ngữ phiên tường minh ngoài danh sách (vd `vi`) | **tự chuyển sang `WhisperTimer`** (`aligner_whisper_fallback=True`, mặc định) — chất lượng mốc cắt y như hiện tại, KHÔNG rơi về chồng lấn |
| Aligner lỗi/không neo được/dưới ngưỡng tin cậy | thử whisper trước, hết đường mới trả `None` ⇒ engine dùng mốc chồng lấn |
| `prewarm` khi ngôn ngữ phiên không được phủ | nạp **whisper**, KHÔNG nạp aligner (tiết kiệm 1,8 GB VRAM vô ích) |

---

## 6. Cách bật / tắt (đã nối sẵn, mặc định vẫn là whisper)

```python
# backend/config.py — SegmentationConfig
timer_engine = "whisper"          # "whisper" | "qwen3-aligner"
use_whisper_timer = True          # công tắc CHUNG bật/tắt tầng timer (tên giữ để tương thích popup)
aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
aligner_device = "cuda:0"
aligner_dtype = "bfloat16"
aligner_local_dir = ""            # rỗng = cache HF (đã có sẵn 1,72 GiB trên máy này)
aligner_auto_download = True
aligner_language_fallback = ""    # ép ngôn ngữ cho phiên auto dùng chữ Latin (vd "English")
aligner_whisper_fallback = True   # ngôn ngữ ngoài 11 ⇒ tự dùng whisper
```

Đổi **ngay lúc đang chạy** (không cần khởi động lại) — dùng để A/B trên một video thật:

```powershell
# REST
curl -X POST http://127.0.0.1:8765/api/config -H "Content-Type: application/json" `
     -d '{"seg_timer_engine":"qwen3-aligner"}'
# hoặc qua WebSocket: {"type":"set_config","segTimerEngine":"qwen3-aligner"}
# hoặc xem cấu hình hiện tại: GET /api/config  →  .seg.timer_engine
```

Đổi engine sẽ **đóng timer cũ** (nhả model) rồi nạp timer mới ở luồng nền; log có
`[ASR] SEG cấu hình lại: … timer_engine=…` và `[ASR] SEG timer sẵn sàng (qwen3-aligner: …)`.

**Vì sao chưa đặt aligner làm mặc định:** máy không NVIDIA (đường Vulkan của ASR) sẽ chạy
aligner trên CPU — align 30 s audio trên CPU là hàng chục giây, không dùng được trong hot path.
Trên máy CUDA, đổi mặc định sau khi bạn xác nhận bằng video thật.

---

## 7. Việc còn lại (đề xuất, chưa làm)

1. **Sửa `_IGNORED` cho whisper** (`segmentation/boundary.py`): thêm `.` và các dấu câu ASCII
   khác, bỏ `'`. Sửa sẽ đổi hành vi của CẢ hai engine (`locate_cut_ms` + SEG) ⇒ phải làm riêng,
   chạy lại bộ `test_47/test_48` và đo lại A/B. Đây là **cải thiện thật** cho engine whisper.
2. **Nút chọn engine trên popup**: hiện chỉ có công tắc `⏱️ Whisper-cut mark`. Thêm dropdown
   `whisper | qwen3-aligner` (popup.html + popup.js + content-script) để người dùng A/B bằng UI.
3. **Trần độ dài**: aligner nhận tối đa ~180 s/lần (tài liệu Qwen) và engine đã chặn
   `timer_max_audio_sec=30` ⇒ an toàn; nếu nâng trần phải kiểm lại.
4. **Đo `asr.commit_ms` trên phiên thật**: bộ đếm `seg.timer_ms` / `seg.timer_used` /
   `seg.timer_miss` đã có sẵn, chỉ cần so trước/sau khi đổi engine.
5. **VRAM**: aligner 1,76 GB (thay vì ~0,85 GB của whisper-turbo Q4) ⇒ tổng pipeline
   (ASR 1,7B + dịch 7B + TTS) dự kiến ~10,4 GB, vẫn dưới trần K7 = 14 GB.

---

## 8. Kiểm tra E2E trong engine thật

`scratch/seg_aligner_e2e.py` chạy đúng đường production: `TranscribeEngine` →
`_evaluate_seg()` (chốt câu theo dấu câu) → `_seg_cut_sample()` → `get_seg_timer()` (đọc
`config.segmentation.timer_engine="qwen3-aligner"`) → aligner với **model thật**:

```
ASR 534 ms | 90 ký tự
prewarm xong 137 ms | timer=Qwen3AlignerTimer
aligner loaded=True | last_error=''
SEG reason=SEG_PUNCT | boundary_index=59 (tham chiếu 59, khớp 0.96)
mốc cắt của engine = 7040 ms | MỐC THẬT = 7004 ms | SAI SỐ +36 ms
KẾT QUẢ: PASS (≤200 ms)
```

---

## 8. Hai lỗi phát hiện từ phiên chạy thật (2026-09-28) và cách sửa

Người dùng bật `timer_engine="qwen3-aligner"` và chạy server thật; log cho thấy aligner bị
**nạp lại liên tục** (5 lần trong log, và `metrics_report.json` của phiên 236 s ghi
`seg.timer_prewarmed = 7`, `seg.timer_load_ms` p50 **9,5 s** / max **15,2 s**). Nguyên nhân và
cách sửa:

### 8.1. Reset timer mỗi lần popup đồng bộ cấu hình

`TranscribeEngine.update_seg_config()` reset timer khi **khoá `use_whisper_timer` CÓ MẶT** trong
kwargs — nhưng popup đồng bộ cấu hình ở MỖI lần kết nối và luôn gửi kèm khoá đó, nên mỗi lần
đồng bộ lại đóng model aligner (1,8 GB) rồi nạp lại từ đầu (mỗi lần ~9 s + ~1,5 s warmup).
Log thật: 3 lần "SEG cấu hình lại" liên tiếp khi một phiên kết nối ⇒ 3 lần nạp.

**Sửa:** chỉ reset khi **giá trị engine THỰC SỰ ĐỔI** (`normalize_timer_engine(trước) !=
sau`); bật/tắt `use_whisper_timer` không cần reset (đã có cờ `_seg_timer_prewarmed` chặn nạp
trùng). Test chống hồi quy:
`test_update_seg_config_khong_nap_lai_khi_cau_hinh_khong_doi`.

### 8.2. Nạp model từ cache HuggingFace thay vì `backend/models/`

Bản đầu dò cache HF bằng `snapshot_download(..., local_files_only=True)` rồi **fallback về
repo id** khi không thấy. Vì lần tải ban đầu dùng `allow_patterns` (bỏ `README.md`,
`.gitattributes`) nên `snapshot_download(local_files_only=True)` **không** thấy "đủ file" ⇒
luôn fallback về repo id ⇒ mỗi lần nạp là một chuỗi request HEAD/GET lên HF (thấy rõ trong log:
hàng chục request `processor_config.json`/`chat_template.json` mỗi lần), và **không chạy được
offline**.

**Sửa (theo yêu cầu):** model **chỉ** được nạp từ `backend/models/`:

* đường dẫn mặc định `backend/models/Qwen__Qwen3-ForcedAligner-0.6B-hf` (cùng quy ước
  `backend/models/_hf/<org>__<name>` của `build_finetune.py`), ghi đè bằng `aligner_local_dir`;
* chưa có ⇒ `snapshot_download(repo_id, local_dir=...)` tải MỘT LẦN về đó (log tiến trình +
  kích thước), sau đó `from_pretrained(<thư mục cục bộ>, local_files_only=True)`;
* không bao giờ truyền repo id cho `from_pretrained` ⇒ không có request mạng khi nạp;
* `available()` chỉ dựa trên thư mục cục bộ (`config.json` + `tokenizer_config.json` +
  `processor_config.json` + file trọng số), hoặc `auto_download`.

Đã tải sẵn: **1762 MB** trong `backend/models/Qwen__Qwen3-ForcedAligner-0.6B-hf/`
(`backend/models/` đã nằm trong `.gitignore`).

### 8.3. Kiểm chứng sau khi sửa (offline hoàn toàn, `HF_HUB_OFFLINE=1`)

```
ASR 541 ms | 90 ký tự
prewarm xong 1 ms | timer=Qwen3AlignerTimer
[ASR] Aligner SEG sẵn sàng (Qwen/Qwen3-ForcedAligner-0.6B-hf, device=cuda:0, dtype=bfloat16,
      nguồn=D:\...\backend\models\Qwen__Qwen3-ForcedAligner-0.6B-hf, 5049 ms)
nguồn model: D:\...\backend\models\Qwen__Qwen3-ForcedAligner-0.6B-hf
SEG reason=SEG_PUNCT | boundary_index=59 (tham chiếu 59, khớp 0.96)
mốc cắt của engine = 7040 ms | MỐC THẬT = 7004 ms | SAI SỐ +36 ms
3 lần đồng bộ cấu hình (như popup): cùng instance=True | timer.loaded=True | 0 ms
KẾT QUẢ: PASS (≤200 ms)
```

Nạp từ đĩa cục bộ còn **nhanh hơn ~2×** so với đường repo id (5,0 s vs 8,9–12,4 s) vì bỏ hết
request mạng.

---

## 9. Tái lập

```powershell
# 1) A/B (mốc thật Silero; --no-asr để cô lập chất lượng timer)
.venv\Scripts\python.exe scratch\seg_ab_aligner.py --n 80 --out scratch\seg_ab_aligner_ja_asr.json
.venv\Scripts\python.exe scratch\seg_ab_aligner.py --n 80 --no-asr --out scratch\seg_ab_aligner_ja_ref.json
.venv\Scripts\python.exe scratch\seg_ab_report.py            # bảng + đối đầu từng mẫu
.venv\Scripts\python.exe scratch\seg_ab_debug.py 19 71       # soi mẫu lệch
.venv\Scripts\python.exe scratch\seg_ab_onset_check.py 19 71 # kiểm chứng mốc thật
.venv\Scripts\python.exe scratch\aligner_warmup_probe.py     # chi phí "lần đầu" + warmup
.venv\Scripts\python.exe scratch\seg_aligner_e2e.py 19        # E2E engine → SEG → aligner (model thật)

# 2) Test tầng A (không nạp model thật)
.venv\Scripts\python.exe -m pytest backend/tests/test_49_aligner_timer.py backend/tests/test_48_seg_timer.py -q
.venv\Scripts\python.exe -m pytest -q
```
