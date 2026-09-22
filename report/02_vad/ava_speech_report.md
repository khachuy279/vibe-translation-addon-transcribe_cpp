# VAD trên AVA-Speech — 10 phút có nhãn người

- **Video**: `5BDj0ow5hnA` · cửa sổ **900–1500 s** (nhãn phủ liên tục)
- **Audio test**: `wav_test\vad\derived\5BDj0ow5hnA_900s_600s_16k_mono.wav` (600.0 s, 16 kHz mono Int16, kênh 0 như plugin)
- **Nhãn**: 98 dòng CSV, so khớp ở độ phân giải 10 ms (`SPEECH_*` = nói, `NO_SPEECH` = lặng)
- **Sinh lúc**: 2026-09-22 08:50:25
- **Pipeline**: `plugin → VAD → ASR`; VAD chỉ chuyển tiếp **byte nguyên bản** (kiểm tra `mismatches = 0` bên dưới).

## 1. Kết quả từng engine (mặc định docs: `threshold=None`, `silence=None`)

| Engine | Hop | Trần pre-roll | P | R | F1 | Accuracy | Segment (pred/GT) | Onset trung vị | RTF | Byte sai lệch |
|---|---|---|---|---|---|---|---|---|---|---|
| `firered-vad` | 160 mẫu | 13 frame | 0.907 | 0.929 | **0.918** | 0.878 | 123 / 46 | 100.0 ms | 0.317 | 0 |
| `silero-vad` | 512 mẫu | 2 frame | 0.952 | 0.496 | **0.652** | 0.613 | 113 / 46 | 95.0 ms | 0.023 | 0 |
| `fsmn-vad` | 960 mẫu | 11 frame | 0.882 | 0.694 | **0.777** | 0.708 | 59 / 46 | 815.0 ms | 0.064 | 0 |

## 2. Recall theo LỚP NHÃN — chỗ phân biệt "domain khó" với "tích hợp sai"

`CLEAN_SPEECH` = speech đọc sạch · `SPEECH_WITH_NOISE` / `SPEECH_WITH_MUSIC` = phim có
nhạc/nhiễu nền. Nếu cả 3 engine đều tốt ở `CLEAN_SPEECH` mà chênh nhau ở hai lớp sau thì
khác biệt là ĐỘ BỀN VỚI NHIỄU của model, không phải lỗi cấu hình/tích hợp.

| Engine | R clean | R noise | R music | Báo động giả trên NO_SPEECH |
|---|---|---|---|---|
| `firered-vad` | 0.940 | 0.947 | 0.000 | 0.260 |
| `silero-vad` | 0.940 | 0.432 | 0.000 | 0.068 |
| `fsmn-vad` | 1.000 | 0.657 | 0.000 | 0.254 |

> ⚠️ **Không so trực tiếp với bảng của nhà cung cấp**: bảng FireRedVAD (F1 97,57 · Silero 95,95)
> được đo **non-streaming trên FLEURS-VAD-102** — speech ĐỌC sạch, clip ~10 s, nhãn nhị phân
> (`external/FireRedVAD/README.md`). AVA-Speech là audio PHIM (nhạc, hiệu ứng, nói chồng tiếng).
> Xem `report/02_vad/ava_vs_vendor_benchmarks.md` để có bằng chứng đầy đủ.

## 3. Ngưỡng hồi quy đã chốt (thấp hơn số đo để tránh dao động nhỏ)

| Engine | F1 ≥ | Recall ≥ | Precision ≥ | Accuracy ≥ | Onset ≤ | RTF ≤ | Số đo lần này |
|---|---|---|---|---|---|---|---|
| `firered-vad` | 0.85 | 0.85 | 0.8 | 0.8 | 200 ms | 0.6 | F1 0,918 · R 0,929 · P 0,907 · RTF 0,333 |
| `silero-vad` | 0.55 | 0.4 | 0.85 | 0.55 | 200 ms | 0.1 | F1 0,652 · R 0,496 · P 0,952 · RTF 0,022 |
| `fsmn-vad` | 0.7 | 0.6 | 0.8 | 0.65 | 1000 ms | 0.15 | F1 0,777 · R 0,694 · P 0,882 · RTF 0,063 |

## 4. Quét `speech_threshold` của FireRed (2 phút đầu)

| speech_threshold | P | R | F1 | Accuracy |
|---|---|---|---|---|
| 0.3 | 0.859 | 0.942 | **0.899** | 0.846 |
| 0.4 | 0.869 | 0.933 | **0.900** | 0.849 |
| 0.5 | 0.876 | 0.912 | **0.894** | 0.843 |
| 0.6 | 0.880 | 0.872 | **0.876** | 0.821 |

## 5. Ghi chú diễn giải

- **FireRed** (mặc định của dự án) có F1 cao nhất và độ trễ onset trung vị ~100 ms;
  số segment nhiều hơn GT vì AVA-Speech gộp các khoảng lặng ngắn, còn VAD cắt theo
  `min_silence_frame = 20` frame (200 ms) đúng mặc định docs.
- **Silero** rất thận trọng trên audio có nhạc/nhiễu nền của AVA (precision cao, recall
  thấp): muốn nhạy hơn thì hạ `threshold` hoặc tăng `min_silence_duration_ms` trong popup.
- **FSMN** ở giữa; độ trễ onset trung vị lớn hơn do `lookback_time_start_point` +
  `window_size_ms` và bước nhảy 60 ms.
- **Toàn vẹn tín hiệu**: mọi frame chuyển tiếp khớp **byte-exact** PCM gốc (0 sai lệch)
  ⇒ VAD không resample/gain/clip trên đường tới ASR.
