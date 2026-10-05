# Phase 9 — Cổng kiên nhẫn gom khối dài ~30s cho Pipeline B (2026-10-05)

## 1. Vấn đề phát hiện từ Log thật phiên `29d8eb1a`
Trình duyệt (YouTube) nạp audio từng đoạn ~10s qua SourceBuffer.
Tuy nhiên, tốc độ ASR GPU rất nhanh (~5.47x thời gian thực, xử lý 8s chỉ tốn ~170–250ms).
Ngay khi xử lý xong khối đầu tiên, con trỏ `_batch_from_pts` đuổi kịp mép nạp của timeline (`available ≈ 8–9s`).
Vì `available >= batch_min_sec (8.0s)`, vòng lặp `_offline_batch_loop` ngay lập tức cắt khối tiếp theo:
- Khối chỉ dài **8.0s - 8.6s**.
- Bị rơi vào nhánh `forced_overlap` vì trong 8s không tìm thấy khoảng lặng VAD >= 1.5s đạt chuẩn.
- Dẫn đến chuỗi khối bị băm vụn liên tục 8.0s, mặc dù `ready_ahead` (phần phụ đề đã dịch trước vị trí phát) lên tới 35s - 65s!

## 2. Giải pháp: Cổng kiên nhẫn `_batch_gate`
Vòng lặp batch theo dõi song song:
1. `available`: lượng audio đã có phía trước con trỏ khối `from_pts`.
2. `ready_ahead`: khoảng cách giữa con trỏ `from_pts` và `current_time` (phần đã dịch phía trước playhead).
3. `proc_time_ema`: thời gian xử lý thực tế 1 khối (ASR + Aligner + Dịch), đo trượt thực tế ~1.5s.

### Quy tắc quyết định:
1. **Khối dài đã sẵn sàng**: Nếu `available >= batch_ready_sec` (30.0s) -> Chạy ngay (`reason="full"`).
2. **Khẩn cấp (nguy cơ gián đoạn)**: Nếu `ready_ahead <= urgent_lead` (ngưỡng an toàn = `batch_urgent_lead_sec (5.0s)` + `2 * proc_time_ema * playback_rate`) -> Chạy ngay dù chưa đủ 30s (`reason="urgent"`). Khi khẩn cấp, nếu `available < 8s` (từ 2.5s trở lên) thì chunker vẫn cho phép lấy (`urgent_tail`) để không gián đoạn phát.
3. **Bootstrap sau tua / Hết stream**: Luôn cho phép chạy.
4. **Còn lại**: Tiếp tục **CHỜ** (`reason="wait"`). Trình duyệt tiếp tục nạp audio vào RAM cho tới khi tích đủ ~30s rồi mới gửi ASR.

## 3. Cập nhật cấu hình
- `batch_wait_for_full_block: bool = True`: Bật cổng kiên nhẫn.
- `batch_ready_sec: float = 30.0`: Ngưỡng gom khối dài.
- `batch_target_sec: float = 28.0`: Nâng mục tiêu cắt VAD bám sát 28-30s.
- `batch_urgent_lead_sec: float = 5.0`: Biên an toàn khẩn cấp.

## 4. Kết quả kiểm thử
Toàn bộ test suite liên quan (`test_58_lookahead_offline_batch`, `test_56_lookahead_chunker`, `test_52_lookahead_e2e_stress`, `test_55_lookahead_persistent_ram`, `test_59_pipeline_b_only_and_popup`) pass 100% (50/50 test cases).
