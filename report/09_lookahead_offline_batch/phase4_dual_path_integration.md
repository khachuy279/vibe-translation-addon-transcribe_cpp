# Báo Cáo Tích Hợp Phase 4: Dual-Path Integration & Bi-directional Context (Pipeline B v3)

**Ngày thực hiện**: 02/10/2026  
**Trạng thái**: ✅ **Hoàn thành 100% & Vượt qua 546/546 Tests (0 failed, 0 regression)**

---

## 1. Tổng Quan Kiến Trúc Tích Hợp (Dual-Path Architecture)

Tuân thủ nguyên tắc an toàn tuyệt đối và khuyến nghị của người dùng: **Không thay thế trực tiếp pipeline v2 hiện tại mà triển khai theo mô hình Feature Flag + Dual Path**.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        LookaheadSessionState                           │
│                                                                        │
│               processing_mode = config.lookahead.processing_mode       │
│                                                                        │
│        ┌─────────────────────────────┴─────────────────────────────┐   │
│        ▼                                                           ▼   │
│  ["streaming"] (Mặc định)                                   ["offline_batch"] (Mới)
│  - Pipeline B v2                                            - Pipeline B v3
│  - `_ingest_loop` (0.5s chunks)                             - `_offline_batch_loop`
│  - VAD streaming processor                                  - `LookaheadChunker` (12s-18s)
│  - `_asr_loop` (stream_tokens)                              - `transcribe_block` (ASR offline)
│  - CommitManager tiers (1/2/3/4)                            - `ForcedAlignerService` (word align)
│                                                             - Gom câu `SubtitleSentence`
│                                                             - Dịch Bi-directional Context
│        │                                                           │   │
│        └─────────────────────────────┬─────────────────────────────┘   │
│                                      ▼                                 │
│             Chung: ContinuousAudioTimeline + get_audio_range           │
│             Chung: SafeWebSocketConnection (`lookahead_subtitles`)     │
│             Chung: Hàng đợi TTS OmniVoice (`tts_queue`)                │
│             Chung: Giao thức Seek Reset + Fast-Bootstrap (3-4s)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Các Thay Đổi Cốt Lõi Đã Thực Hiện

### 2.1. Cấu hình Feature Flag (`backend/config.py`)
Bổ sung các trường vào `LookaheadConfig`:
- `processing_mode: str = "streaming"` ("streaming" | "offline_batch").
- `batch_target_sec: float = 15.0`.
- `batch_min_sec: float = 12.0`.
- `batch_max_sec: float = 18.0`.
- `batch_min_silence_ms: float = 250.0`.
- `batch_overlap_sec: float = 1.0`.

### 2.2. ASR Engine Offline Block Decoding (`backend/asr/engine.py`)
- Cài đặt `transcribe_block(pcm_audio, language)`: Gọi trọn khối đồng bộ qua `_run_inference_sync` dưới `_infer_lock`, tự động quản lý chuyển đổi ngôn ngữ an toàn và khôi phục ngôn ngữ cũ sau khi giải mã.

### 2.3. Lookup Audio Không Di Chuyển Cursor (`backend/core/lookahead_timeline.py`)
- Cài đặt `get_audio_range(start_pts, duration_sec)`:
  - Cho phép trích xuất audio liên tục từ RAM timeline mà **không đụng tới `_cursor`** của streaming v2.
  - Tự động ghép nối qua nhiều chunk MSE và lấp khoảng trống nhỏ ($\le 350\text{ms}$) bằng silence.

### 2.4. Vòng lặp `_offline_batch_loop` (`backend/ws/lookahead_handler.py`)
- Triển khai toàn bộ quy trình:
  1. `LookaheadChunker.next_chunk(from_pts, fast_bootstrap)` cắt khối thông minh tại VAD silence midpoint.
  2. `transcribe_block()` nhận diện trọn vẹn câu với ngữ cảnh tối đa.
  3. `ForcedAlignerService.align()` gióng hàng từ siêu tốc (~30-100ms trên GPU).
  4. `group_words_to_subtitles()` gom từ thành câu vừa mắt người xem.
  5. **Phase 3 Bi-directional Context**:
     - Ngữ cảnh trước: 2 câu gần nhất trong `_recent_utterances` (câu gốc + câu dịch).
     - Ngữ cảnh sau: Câu phụ đề tiếp theo ngay trong cùng batch (câu gốc).
  6. Gửi bản tin WS `lookahead_subtitles` với `start_pts` và `end_pts` chuẩn xác.
  7. Xếp bản dịch vào hàng đợi TTS OmniVoice.
- **Xử lý Seek Reset**:
  - Khi tua video (`handle_seek`):
    - Đặt lại `_batch_from_pts = seek_time`.
    - Bật cờ `_fast_bootstrap = True` (cắt khối nhanh 3-4s để phụ đề hiện tức thì).
    - Hủy các câu đang bay của `seek_id` cũ.

---

## 3. Kết Quả Kiểm Thử (Verification & Testing)

### 3.1. Integration Test Mới (`test_58_lookahead_offline_batch.py`)
- `test_lookahead_processing_mode_flag_routing`: ✅ PASS.
  - Xác nhận khi `processing_mode == "streaming"`, chỉ có `la_ingest` và `la_asr` chạy.
  - Khi `processing_mode == "offline_batch"`, chỉ có `la_batch` chạy.
- `test_offline_batch_loop_end_to_end`: ✅ PASS.
  - Xác nhận audio 16s đi qua toàn bộ Chunker $\rightarrow$ ASR $\rightarrow$ Aligner $\rightarrow$ Bi-directional Translation $\rightarrow$ Subtitles.
  - Xác thực translator nhận đúng `Previous context` và `Next sentence`.
- `test_offline_batch_seek_reset`: ✅ PASS.
  - Xác nhận khi tua video, `_fast_bootstrap` bật lên và `_batch_from_pts` chuyển đúng vị trí tua.

### 3.2. Test Toàn Bộ Các Thành Phần Lookahead
Chạy 7 file test liên quan đến Pipeline B:
- `test_50_lookahead_demuxer.py` (14 tests)
- `test_51_lookahead_pipeline.py` (7 tests)
- `test_54_lookahead_pipeline_v2.py` (32 tests)
- `test_55_lookahead_persistent_ram.py` (7 tests)
- `test_56_lookahead_chunker.py` (6 tests)
- `test_57_forced_aligner_service.py` (4 tests)
- `test_58_lookahead_offline_batch.py` (3 tests)
👉 **Tổng cộng: 73/73 tests PASS trong 13.86s.**

### 3.3. Test Hồi Quy Toàn Bộ Kho Mã Nguồn (`pytest`)
- Chạy toàn bộ test suite mặc định của dự án:
  👉 **546 passed, 45 deselected, 0 failed trong 33.75s.**
  👉 Không phá vỡ bất kỳ luồng streaming v2, CommitManager, hay invariant nào của repo.

---

## 4. Kết Luận & Hướng Dẫn Sử Dụng

1. **Trạng thái an toàn**: Mặc định hệ thống vẫn đang chạy `processing_mode = "streaming"` (Pipeline B v2 ổn định).
2. **Kích hoạt thử nghiệm Pipeline B v3**:
   - Trong `backend/config.py`:
     ```python
     lookahead.processing_mode = "offline_batch"
     ```
   - Hoặc gửi qua WebSocket init/popup payload:
     ```json
     {
       "type": "lookahead_init",
       "lookaheadProcessingMode": "offline_batch"
     }
     ```
3. Toàn bộ nền tảng đã sẵn sàng để kiểm thử trực tiếp trên browser extension với video thật.
