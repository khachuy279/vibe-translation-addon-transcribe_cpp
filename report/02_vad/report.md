# Báo Cáo Đo Lường & Kiểm Thử Phase 2: Module VAD Streaming Độc Lập

- **Thời gian thực hiện**: 2026-09-21 22:55:08
- **Kiến trúc**: `report/audit/19_KE_HOACH_VIET_LAI_VAD.md` — mỗi engine dùng đúng
  API docs (`FireRedStreamVad` / `VADIterator` / `AutoModel.generate`), processor chỉ
  chuyển tiếp audio nguyên bản khi engine phát `START`/`END`.
- **Cấu hình**: `threshold=None`, `silence_duration_ms=None` ⇒ mỗi engine dùng đúng
  mặc định trong docs của nó (FireRed `min_silence_frame=20` = 200 ms, Silero 100 ms,
  FSMN `max_end_silence_time` = 800 ms).

## 1. Kết Quả Benchmark Chi Tiết Từng Engine Trên `/wav_test`

| Engine | File Audio | Thời lượng | Hop | Trần pre-roll | Avg Chunk (µs) | p95 Chunk (µs) | RTF | Starts | Ends |
|---|---|---|---|---|---|---|---|---|---|
| `firered-vad` | `00_ingress_stream.wav` | 332.03s | 10.0 ms | 13 frame | 6032.3 µs | 6779.9 µs | 0.3017 | 153 | 153 |
| `firered-vad` | `Chinese_fast_speed_11s.wav` | 11.38s | 10.0 ms | 13 frame | 5981.9 µs | 6673.5 µs | 0.2997 | 1 | 0 |
| `firered-vad` | `Chinese_noise_28s.wav` | 28.26s | 10.0 ms | 13 frame | 6077.4 µs | 6765.0 µs | 0.304 | 2 | 1 |
| `firered-vad` | `Cross_lingual_English_French_Italian_Spanish_6s.wav` | 6.23s | 10.0 ms | 13 frame | 5996.8 µs | 6695.8 µs | 0.3004 | 5 | 4 |
| `firered-vad` | `English_low_speech_quality_19s.wav` | 19.02s | 10.0 ms | 13 frame | 6007.6 µs | 6592.5 µs | 0.3005 | 7 | 7 |
| `firered-vad` | `English_multiple_kinds_of_noise_88s.wav` | 88.19s | 10.0 ms | 13 frame | 6011.4 µs | 6610.0 µs | 0.3007 | 13 | 12 |
| `firered-vad` | `Japanese_5s.wav` | 5.08s | 10.0 ms | 13 frame | 5993.9 µs | 6521.9 µs | 0.2998 | 1 | 1 |
| `firered-vad` | `Russian_4s.wav` | 4.76s | 10.0 ms | 13 frame | 5973.0 µs | 6488.6 µs | 0.2987 | 1 | 1 |
| `fsmn-vad` | `00_ingress_stream.wav` | 332.03s | 60.0 ms | 11 frame | 1065.5 µs | 3451.4 µs | 0.0533 | 77 | 77 |
| `fsmn-vad` | `Chinese_fast_speed_11s.wav` | 11.38s | 60.0 ms | 11 frame | 1046.9 µs | 3382.4 µs | 0.0525 | 1 | 0 |
| `fsmn-vad` | `Chinese_noise_28s.wav` | 28.26s | 60.0 ms | 11 frame | 1047.6 µs | 3381.9 µs | 0.0524 | 1 | 0 |
| `fsmn-vad` | `Cross_lingual_English_French_Italian_Spanish_6s.wav` | 6.23s | 60.0 ms | 11 frame | 1038.3 µs | 3405.8 µs | 0.052 | 1 | 0 |
| `fsmn-vad` | `English_low_speech_quality_19s.wav` | 19.02s | 60.0 ms | 11 frame | 1057.8 µs | 3407.4 µs | 0.0529 | 7 | 6 |
| `fsmn-vad` | `English_multiple_kinds_of_noise_88s.wav` | 88.19s | 60.0 ms | 11 frame | 1050.6 µs | 3408.5 µs | 0.0526 | 14 | 13 |
| `fsmn-vad` | `Japanese_5s.wav` | 5.08s | 60.0 ms | 11 frame | 1051.1 µs | 3434.3 µs | 0.0526 | 1 | 0 |
| `fsmn-vad` | `Russian_4s.wav` | 4.76s | 60.0 ms | 11 frame | 1081.1 µs | 3614.0 µs | 0.0541 | 1 | 0 |
| `silero-vad` | `00_ingress_stream.wav` | 332.03s | 32.0 ms | 2 frame | 470.3 µs | 841.1 µs | 0.0236 | 118 | 118 |
| `silero-vad` | `Chinese_fast_speed_11s.wav` | 11.38s | 32.0 ms | 2 frame | 564.1 µs | 878.6 µs | 0.0284 | 1 | 0 |
| `silero-vad` | `Chinese_noise_28s.wav` | 28.26s | 32.0 ms | 2 frame | 505.1 µs | 846.7 µs | 0.0253 | 1 | 0 |
| `silero-vad` | `Cross_lingual_English_French_Italian_Spanish_6s.wav` | 6.23s | 32.0 ms | 2 frame | 647.2 µs | 917.5 µs | 0.0325 | 5 | 4 |
| `silero-vad` | `English_low_speech_quality_19s.wav` | 19.02s | 32.0 ms | 2 frame | 529.0 µs | 870.6 µs | 0.0265 | 7 | 7 |
| `silero-vad` | `English_multiple_kinds_of_noise_88s.wav` | 88.19s | 32.0 ms | 2 frame | 488.2 µs | 874.8 µs | 0.0245 | 18 | 17 |
| `silero-vad` | `Japanese_5s.wav` | 5.08s | 32.0 ms | 2 frame | 705.6 µs | 990.7 µs | 0.0354 | 2 | 2 |
| `silero-vad` | `Russian_4s.wav` | 4.76s | 32.0 ms | 2 frame | 703.0 µs | 929.7 µs | 0.0352 | 1 | 1 |

## 2. Bảng Xếp Hạng Hiệu Năng VAD (RTF & Latency)

| Engine VAD | RTF Trung Bình | Đặc Điểm |
|---|---|---|
| **`firered-vad`** | **0.3007** | Cửa sổ 400 mẫu / hop 160 mẫu (10 ms). Trần pre-roll 13 frame (`pad_start_frame + min_speech_frame`); đo được thực xả 12 frame. |
| **`silero-vad`** | **0.0289** | 512 mẫu (32 ms)/frame, `VADIterator` với `min_silence_duration_ms`/`speech_pad_ms`. |
| **`fsmn-vad`** | **0.0528** | Chunk 60 ms, `generate(cache=…, is_final=False, chunk_size=60)`; trần pre-roll 11 frame (`window_size_ms + sil_to_speech_time_thres + lookback_time_start_point`). |

## 3. Kết Luận Nghiệm Thu Phase 2

- Cả 3 engine chạy đúng API docs, mỗi engine một config riêng, RTF ≪ 1 (nhanh hơn thời gian thực).
- Audio chuyển tiếp cho ASR là **byte nguyên bản** của client: không resample, không gain,
  không clip — xem `test_42_vad_signal_integrity.py`.
- `hangover_ms`/`pre_speech_buffer_ms` đã bị xoá: hangover và pre-padding nay do chính
  VAD quyết định qua `START`/`END` và `lookback_frames`.