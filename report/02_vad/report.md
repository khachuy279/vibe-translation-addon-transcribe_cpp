# Báo Cáo Đo Lường & Kiểm Thử Phase 2: Module VAD Streaming Độc Lập

- **Thời gian thực hiện**: 2026-09-21 23:39:22
- **Kiến trúc**: `report/audit/19_KE_HOACH_VIET_LAI_VAD.md` — mỗi engine dùng đúng
  API docs (`FireRedStreamVad` / `VADIterator` / `AutoModel.generate`), processor chỉ
  chuyển tiếp audio nguyên bản khi engine phát `START`/`END`.
- **Cấu hình**: `threshold=None`, `silence_duration_ms=None` ⇒ mỗi engine dùng đúng
  mặc định trong docs của nó (FireRed `min_silence_frame=20` = 200 ms, Silero 100 ms,
  FSMN `max_end_silence_time` = 800 ms).

## 1. Kết Quả Benchmark Chi Tiết Từng Engine Trên `/wav_test`

| Engine | File Audio | Thời lượng | Hop | Trần pre-roll | Avg Chunk (µs) | p95 Chunk (µs) | RTF | Starts | Ends |
|---|---|---|---|---|---|---|---|---|---|
| `firered-vad` | `00_ingress_stream.wav` | 332.03s | 10.0 ms | 13 frame | 6104.7 µs | 6823.7 µs | 0.3053 | 153 | 153 |
| `firered-vad` | `Chinese_fast_speed_11s.wav` | 11.38s | 10.0 ms | 13 frame | 6009.9 µs | 6628.4 µs | 0.3011 | 1 | 0 |
| `firered-vad` | `Chinese_noise_28s.wav` | 28.26s | 10.0 ms | 13 frame | 6089.7 µs | 6841.0 µs | 0.3046 | 2 | 1 |
| `firered-vad` | `Cross_lingual_English_French_Italian_Spanish_6s.wav` | 6.23s | 10.0 ms | 13 frame | 6107.6 µs | 6832.9 µs | 0.3059 | 5 | 4 |
| `firered-vad` | `English_low_speech_quality_19s.wav` | 19.02s | 10.0 ms | 13 frame | 6118.3 µs | 6830.0 µs | 0.306 | 7 | 7 |
| `firered-vad` | `English_multiple_kinds_of_noise_88s.wav` | 88.19s | 10.0 ms | 13 frame | 6052.3 µs | 6847.6 µs | 0.3027 | 13 | 12 |
| `firered-vad` | `Japanese_5s.wav` | 5.08s | 10.0 ms | 13 frame | 5898.9 µs | 6680.8 µs | 0.295 | 1 | 1 |
| `firered-vad` | `Russian_4s.wav` | 4.76s | 10.0 ms | 13 frame | 5787.1 µs | 6225.3 µs | 0.2894 | 1 | 1 |
| `fsmn-vad` | `00_ingress_stream.wav` | 332.03s | 60.0 ms | 11 frame | 1082.0 µs | 3422.8 µs | 0.0541 | 77 | 77 |
| `fsmn-vad` | `Chinese_fast_speed_11s.wav` | 11.38s | 60.0 ms | 11 frame | 1047.7 µs | 3299.6 µs | 0.0525 | 1 | 0 |
| `fsmn-vad` | `Chinese_noise_28s.wav` | 28.26s | 60.0 ms | 11 frame | 1041.8 µs | 3278.5 µs | 0.0521 | 1 | 0 |
| `fsmn-vad` | `Cross_lingual_English_French_Italian_Spanish_6s.wav` | 6.23s | 60.0 ms | 11 frame | 1038.5 µs | 3269.6 µs | 0.052 | 1 | 0 |
| `fsmn-vad` | `English_low_speech_quality_19s.wav` | 19.02s | 60.0 ms | 11 frame | 1051.2 µs | 3290.7 µs | 0.0526 | 7 | 6 |
| `fsmn-vad` | `English_multiple_kinds_of_noise_88s.wav` | 88.19s | 60.0 ms | 11 frame | 1035.0 µs | 3243.3 µs | 0.0518 | 14 | 13 |
| `fsmn-vad` | `Japanese_5s.wav` | 5.08s | 60.0 ms | 11 frame | 989.9 µs | 3157.6 µs | 0.0495 | 1 | 0 |
| `fsmn-vad` | `Russian_4s.wav` | 4.76s | 60.0 ms | 11 frame | 1034.7 µs | 3412.0 µs | 0.0518 | 1 | 0 |
| `silero-vad` | `00_ingress_stream.wav` | 332.03s | 32.0 ms | 2 frame | 468.3 µs | 815.1 µs | 0.0235 | 118 | 118 |
| `silero-vad` | `Chinese_fast_speed_11s.wav` | 11.38s | 32.0 ms | 2 frame | 558.4 µs | 841.1 µs | 0.0281 | 1 | 0 |
| `silero-vad` | `Chinese_noise_28s.wav` | 28.26s | 32.0 ms | 2 frame | 493.3 µs | 793.2 µs | 0.0247 | 1 | 0 |
| `silero-vad` | `Cross_lingual_English_French_Italian_Spanish_6s.wav` | 6.23s | 32.0 ms | 2 frame | 615.4 µs | 821.5 µs | 0.0309 | 5 | 4 |
| `silero-vad` | `English_low_speech_quality_19s.wav` | 19.02s | 32.0 ms | 2 frame | 505.4 µs | 793.3 µs | 0.0253 | 7 | 7 |
| `silero-vad` | `English_multiple_kinds_of_noise_88s.wav` | 88.19s | 32.0 ms | 2 frame | 474.5 µs | 798.5 µs | 0.0238 | 18 | 17 |
| `silero-vad` | `Japanese_5s.wav` | 5.08s | 32.0 ms | 2 frame | 660.7 µs | 812.0 µs | 0.0331 | 2 | 2 |
| `silero-vad` | `Russian_4s.wav` | 4.76s | 32.0 ms | 2 frame | 746.2 µs | 886.6 µs | 0.0374 | 1 | 1 |

## 2. Bảng Xếp Hạng Hiệu Năng VAD (RTF & Latency)

| Engine VAD | RTF Trung Bình | Đặc Điểm |
|---|---|---|
| **`firered-vad`** | **0.3013** | Cửa sổ 400 mẫu / hop 160 mẫu (10 ms). Trần pre-roll 13 frame (`pad_start_frame + min_speech_frame`); đo được thực xả 12 frame. |
| **`silero-vad`** | **0.0284** | 512 mẫu (32 ms)/frame, `VADIterator` với `min_silence_duration_ms`/`speech_pad_ms`. |
| **`fsmn-vad`** | **0.052** | Chunk 60 ms, `generate(cache=…, is_final=False, chunk_size=60)`; trần pre-roll 11 frame (`window_size_ms + sil_to_speech_time_thres + lookback_time_start_point`). |

## 3. Kết Luận Nghiệm Thu Phase 2

- Cả 3 engine chạy đúng API docs, mỗi engine một config riêng, RTF ≪ 1 (nhanh hơn thời gian thực).
- Audio chuyển tiếp cho ASR là **byte nguyên bản** của client: không resample, không gain,
  không clip — xem `test_42_vad_signal_integrity.py`.
- `hangover_ms`/`pre_speech_buffer_ms` đã bị xoá: hangover và pre-padding nay do chính
  VAD quyết định qua `START`/`END` và `lookback_frames`.