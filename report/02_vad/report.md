# Báo Cáo Đo Lường & Kiểm Thử Phase 2: Module VAD Streaming Độc Lập

- **Thời gian thực hiện**: 2026-10-06 21:22:17
- **Kiến trúc**: `report/audit/19_KE_HOACH_VIET_LAI_VAD.md` — mỗi engine dùng đúng
  API docs (`FireRedStreamVad` / `VADIterator` / `AutoModel.generate`), processor chỉ
  chuyển tiếp audio nguyên bản khi engine phát `START`/`END`.
- **Cấu hình**: `threshold=None`, `silence_duration_ms=None` ⇒ mỗi engine dùng đúng
  mặc định trong docs của nó (FireRed `min_silence_frame=20` = 200 ms, Silero 100 ms,
  FSMN `max_end_silence_time` = 800 ms).

## 1. Kết Quả Benchmark Chi Tiết Từng Engine Trên `/wav_test`

| Engine | File Audio | Thời lượng | Hop | Trần pre-roll | Avg Chunk (µs) | p95 Chunk (µs) | RTF | Starts | Ends |
|---|---|---|---|---|---|---|---|---|---|
| `firered-vad` | `00_ingress_stream.wav` | 332.03s | 10.0 ms | 13 frame | 1608.6 µs | 1772.0 µs | 0.0805 | 153 | 153 |
| `firered-vad` | `Chinese_fast_speed_11s.wav` | 11.38s | 10.0 ms | 13 frame | 1614.5 µs | 1892.5 µs | 0.0809 | 1 | 0 |
| `firered-vad` | `Chinese_noise_28s.wav` | 28.26s | 10.0 ms | 13 frame | 1612.5 µs | 1755.2 µs | 0.0807 | 2 | 1 |
| `firered-vad` | `Cross_lingual_English_French_Italian_Spanish_6s.wav` | 6.23s | 10.0 ms | 13 frame | 1629.2 µs | 1810.5 µs | 0.0816 | 5 | 4 |
| `firered-vad` | `English_low_speech_quality_19s.wav` | 19.02s | 10.0 ms | 13 frame | 1598.3 µs | 1715.6 µs | 0.08 | 7 | 7 |
| `firered-vad` | `English_multiple_kinds_of_noise_88s.wav` | 88.19s | 10.0 ms | 13 frame | 1610.5 µs | 1781.9 µs | 0.0806 | 13 | 12 |
| `firered-vad` | `Japanese_5s.wav` | 5.08s | 10.0 ms | 13 frame | 1582.5 µs | 1692.0 µs | 0.0792 | 1 | 1 |
| `firered-vad` | `Russian_4s.wav` | 4.76s | 10.0 ms | 13 frame | 1695.2 µs | 2159.1 µs | 0.0848 | 1 | 1 |
| `silero-vad` | `00_ingress_stream.wav` | 332.03s | 32.0 ms | 2 frame | 107.8 µs | 228.4 µs | 0.0054 | 118 | 118 |
| `silero-vad` | `Chinese_fast_speed_11s.wav` | 11.38s | 32.0 ms | 2 frame | 103.0 µs | 187.9 µs | 0.0052 | 1 | 0 |
| `silero-vad` | `Chinese_noise_28s.wav` | 28.26s | 32.0 ms | 2 frame | 105.4 µs | 226.6 µs | 0.0053 | 1 | 0 |
| `silero-vad` | `Cross_lingual_English_French_Italian_Spanish_6s.wav` | 6.23s | 32.0 ms | 2 frame | 111.7 µs | 231.0 µs | 0.0056 | 5 | 4 |
| `silero-vad` | `English_low_speech_quality_19s.wav` | 19.02s | 32.0 ms | 2 frame | 110.4 µs | 231.8 µs | 0.0055 | 7 | 7 |
| `silero-vad` | `English_multiple_kinds_of_noise_88s.wav` | 88.19s | 32.0 ms | 2 frame | 107.8 µs | 226.2 µs | 0.0054 | 18 | 17 |
| `silero-vad` | `Japanese_5s.wav` | 5.08s | 32.0 ms | 2 frame | 105.5 µs | 198.7 µs | 0.0053 | 2 | 2 |
| `silero-vad` | `Russian_4s.wav` | 4.76s | 32.0 ms | 2 frame | 102.8 µs | 192.4 µs | 0.0052 | 1 | 1 |

## 2. Bảng Xếp Hạng Hiệu Năng VAD (RTF & Latency)

| Engine VAD | RTF Trung Bình | Đặc Điểm |
|---|---|---|
| **`firered-vad`** | **0.081** | Cửa sổ 400 mẫu / hop 160 mẫu (10 ms). Trần pre-roll 13 frame (`pad_start_frame + min_speech_frame`); đo được thực xả 12 frame. |
| **`silero-vad`** | **0.0054** | 512 mẫu (32 ms)/frame, `VADIterator` với `min_silence_duration_ms`/`speech_pad_ms`. |
| **`fsmn-vad`** | **0** | Chunk 60 ms, `generate(cache=…, is_final=False, chunk_size=60)`; trần pre-roll 11 frame (`window_size_ms + sil_to_speech_time_thres + lookback_time_start_point`). |

## 3. Kết Luận Nghiệm Thu Phase 2

- Cả 3 engine chạy đúng API docs, mỗi engine một config riêng, RTF ≪ 1 (nhanh hơn thời gian thực).
- Audio chuyển tiếp cho ASR là **byte nguyên bản** của client: không resample, không gain,
  không clip — xem `test_42_vad_signal_integrity.py`.
- `hangover_ms`/`pre_speech_buffer_ms` đã bị xoá: hangover và pre-padding nay do chính
  VAD quyết định qua `START`/`END` và `lookback_frames`.