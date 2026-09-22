# Vì sao điểm AVA-Speech khác bảng của nhà cung cấp (FireRedVAD / Silero)

> Trả lời câu hỏi: *"FireRedVAD công bố F1 97,57 · Silero 95,95, sao đo trên AVA-Speech lại
> 0,918 / 0,652 / 0,777 — có phải cấu hình Silero/FSMN sai hoặc bài test có vấn đề?"*
>
> Kết luận: **cấu hình và tích hợp ĐÚNG** (4 phép kiểm chứng độc lập bên dưới). Khác biệt nằm ở
> **domain + thước đo**: bảng nhà cung cấp đo trên **speech ĐỌC sạch (FLEURS-VAD-102, non-streaming)**,
> còn AVA-Speech là **audio phim** (nhạc, hiệu ứng, nói chồng tiếng).

---

## 1. Bảng nhà cung cấp được đo trên dữ liệu nào?

`external/FireRedVAD/README.md` (bản trong repo):

> "Non-streaming VAD achieves **97.57% F1 on FLEURS-VAD-102** … FLEURS-VAD-102: We randomly
> selected ~100 audio files per language from **FLEURS test set**, resulting in 9,443 audio files
> with manually annotated binary VAD labels (speech=1, silence=0)."

⇒ Bảng đó là **non-streaming**, trên **clip speech đọc ~10 s**, đa ngữ. Silero 95,95 trong bảng
ấy cũng do nhóm FireRed đo **trên cùng bộ FLEURS-VAD-102**, không phải trên audio phim.

Dữ liệu huấn luyện/đánh giá của Silero (`external/silero-vad/datasets/README.md`, bản trong repo)
là **150.547 giờ speech ĐỌC**: Bible.is · globalrecordings.net · VoxLingua107 · Common Voice · MLS
(đều là đọc sách/đọc câu, gán nhãn ở độ phân giải 512 mẫu ~30 ms). **Không có nhạc/hiệu ứng phim.**

AVA-Speech (`wav_test/vad/ava_speech_labels_v1.csv`) thì ngược lại: audio phim, 4 lớp nhãn
`CLEAN_SPEECH` · `SPEECH_WITH_NOISE` · `SPEECH_WITH_MUSIC` · `NO_SPEECH`, phủ 900–1800 s.

---

## 2. Kết quả đo trên AVA-Speech (10 phút, mặc định docs)

| Engine | F1 | Recall | Precision | Δ Segment (pred/GT) | Onset trung vị | RTF |
|---|---|---|---|---|---|---|
| `firered-vad` | **0,918** | 0,929 | 0,907 | 123 / 46 | 100 ms | 0,332 |
| `silero-vad` | 0,652 | 0,496 | 0,952 | 113 / 46 | 95 ms | 0,022 |
| `fsmn-vad` | 0,777 | 0,694 | 0,882 | 59 / 46 | 815 ms | 0,053 |

## 3. Điểm mấu chốt: recall theo LỚP NHÃN

| Engine | R `CLEAN_SPEECH` | R `SPEECH_WITH_NOISE` | R `SPEECH_WITH_MUSIC` | Báo động giả trên `NO_SPEECH` |
|---|---|---|---|---|
| `firered-vad` | 0,940 | **0,947** | 0,000 | 0,260 |
| `silero-vad` | **0,939** | **0,432** | 0,000 | **0,068** |
| `fsmn-vad` | **1,000** | 0,657 | 0,000 | 0,254 |

⇒ Trên **speech sạch**, Silero 0,939 ≈ FireRed 0,940. Toàn bộ khoảng cách đến từ
**`SPEECH_WITH_NOISE`** (thoại phim có nhạc/nhiễu nền): FireRed 0,947 so với Silero 0,432.
Đây là **độ bền nhiễu của model**, không phải lỗi cấu hình. Silero cũng là engine **thận trọng
nhất** (precision 0,952 · báo động giả 0,068) — đúng kiểu "thà bỏ sót còn hơn nhận nhầm", nên
recall thấp là hệ quả nhất quán chứ không phải bất thường.

> Lưu ý về cột "báo động giả": phần đuôi hangover hợp lệ (FireRed 200 ms · Silero 100 ms ·
> FSMN 800 ms) nằm trong vùng `NO_SPEECH` nên được tính vào đây; chỉ số sạch hơn để so sánh là
> **precision** (0,907 / 0,952 / 0,882).

(`SPEECH_WITH_MUSIC` chỉ có **1** đoạn trong 10 phút nên recall 0,000 không có ý nghĩa thống kê.)

---

## 4. Bốn phép kiểm chứng để loại trừ lỗi cấu hình / lỗi bài test

| # | Giả thuyết | Cách kiểm | Kết quả |
|---|---|---|---|
| 1 | Engine đang nạp **model Silero cũ (v4/v5)** chứ không phải v6 | So SHA-256 file local với file bundled của gói `silero-vad`; thử cửa sổ 512/1024/1536 | **Byte-identical** (`e1122837…`, 2.272.526 B) với `silero_vad.data/silero_vad.jit`; model chỉ nhận **512** mẫu (1024/1536 lỗi) ⇒ đúng v6 + đúng cửa sổ |
| 2 | Đường **streaming** của ta bóp méo kết quả | So với **API offline chính chủ** `get_speech_timestamps()` trên cùng audio (cửa sổ 3 phút) | streaming F1 **0,257** · offline `thr=0.5` **0,243** · `thr=0.35` 0,254 ⇒ **trùng nhau**; quét ngưỡng 0,25→0,5 không cải thiện ⇒ không phải lỗi tích hợp, cũng không phải lỗi ngưỡng |
| 3 | Config lệch docs | `test_41::test_config_3_engine_khop_docs` + `scratch/vad_configs_vs_docs.py` | 8/8 field FireRed, 3/3 knob Silero, 8/8 field FSMN khớp nguồn chân lý (`FireRedStreamVadConfig`, `VADIterator.__init__`, `config.yaml` của checkpoint) |
| 4 | Bài test/harness sai | **Benchmark in-domain** `test_44_vad_clean_speech_benchmark.py` — 5 clip speech ĐỌC sạch 16 kHz trong repo, ghép với khoảng lặng 3 s, chấm kiểu FLEURS-like | **cả 3 engine phát hiện 100 % clip**; F1 so tham chiếu năng lượng: FireRed **0,897** · Silero **0,893** · FSMN **0,976** ⇒ Silero ≈ FireRed trên đúng loại dữ liệu của bảng nhà cung cấp |

Số liệu phụ trợ:
* AUC-ROC của Silero trên 10 phút AVA = **0,633** (xác suất TB vùng nói 0,443 vs vùng lặng 0,046)
  ⇒ thứ hạng frame của model thật sự yếu trên domain này, không phải chỉ lệch ngưỡng.
* Chấm ở độ phân giải 1 s thay vì 10 ms gần như không đổi (F1 0,638 vs 0,608 cho Silero) ⇒ không
  phải vấn đề độ phân giải thước đo.
* Kênh audio: file nguồn 48 kHz stereo có **L ≈ R** (tương quan +0,998, RMS bằng nhau) ⇒ chọn
  kênh 0 như plugin là đúng. (Một dòng trong script chẩn đoán ban đầu quên chia 32768 trước khi
  `clip` nên cho kết quả mono-mean ~0 — lỗi của script nháp, không phải của pipeline; đã sửa nhận định.)

---

## 5. Vì sao vẫn nên ưu tiên FireRed cho dự án này

1. **Domain của dự án là video/phim** (plugin bắt audio từ tab video) — đúng thứ AVA-Speech đo.
   Trên `SPEECH_WITH_NOISE`, FireRed 0,947 so với Silero 0,432 và FSMN 0,657.
2. FireRed nhạy với cả thoại nền nhạc/hiệu ứng (đúng thiết kế: "speech/singing/music detection
   in 100+ languages"), lại vẫn nhanh hơn thời gian thực ~3× (RTF 0,33 trên CPU).
3. Silero vẫn là lựa chọn tốt khi cần **nhẹ** (RTF 0,022, nạp 0,1 s) hoặc cho speech sạch; popup
   đổi engine nóng, cấu hình riêng từng engine (`report/audit/19_KE_HOACH_VIET_LAI_VAD.md`).
4. Muốn so trực tiếp với bảng nhà cung cấp thì phải chạy **đúng bộ FLEURS-VAD-102** (9.443 clip,
   non-streaming) — không có sẵn offline trong repo; `test_44` là bản xấp xỉ in-domain tại chỗ.

---

## 6. Cách tái lập

```powershell
# 1) Điểm trên AVA-Speech + recall theo lớp nhãn + báo cáo
python -m pytest -m slow backend/tests/test_43_vad_ava_speech.py -q -s

# 2) Bằng chứng "không phải lỗi tích hợp": streaming vs API offline chính chủ
python scratch/vad_streaming_vs_offline_ava.py

# 3) Chẩn đoán Silero: AUC, quét ngưỡng, theo cửa sổ 60 s, kênh audio
python scratch/vad_silero_diagnose.py

# 4) Phân rã recall theo lớp nhãn
python scratch/vad_ava_class_breakdown.py

# 5) Benchmark in-domain (speech đọc sạch) — phải 100 % phát hiện cho cả 3 engine
python -m pytest -m slow backend/tests/test_44_vad_clean_speech_benchmark.py -q -s

# 6) Đối chiếu config với docs
python scratch/vad_configs_vs_docs.py
```
