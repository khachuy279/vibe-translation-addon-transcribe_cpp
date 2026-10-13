# Kế Hoạch & TODO: Đại Tu Toàn Diện Kiến Trúc Ngắt Câu (SEG_BATCH - Pipeline B)

**Ngày lập**: 2026-10-13  
**Trạng thái**: Đang triển khai (In Progress)  
**Tác giả**: Antigravity & User  
**Mục tiêu**: Xóa bỏ hoàn toàn tình trạng chẻ vụn câu, tạo câu phụ đề tự nhiên, chuẩn ngữ nghĩa, ổn định cho cả hiển thị phụ đề và dịch thuật.

---

## 1. Bối cảnh & Vấn đề tồn đọng

### 1.1. Hiện tượng lỗi thực tế
Trong Pipeline B (Lookahead Batch / `SEG_BATCH`), hệ thống liên tục gặp tình trạng chẻ câu thành các mảnh vụn kỳ lạ:
- **Lỗi 1 (Cắt giữa cụm ngữ nghĩa)**:
  - *Câu gốc*: `"With our friends moving forward in their relationships, I've decided..."`
  - *Hiện tượng*: Bị chẻ thành `⏐ "With our" ⏐ "friends moving forward in their relationships," ⏐`
- **Lỗi 2 (Cắt sai trước mệnh đề phụ)**:
  - *Câu gốc*: `"Hope you're not thinking about some sort of LSD thought control, because there's only one mind-expanding drug..."`
  - *Hiện tượng*: Bị chẻ thành `⏐ "Hope you're not thinking about some sort of LSD thought" ⏐ "control, because there's only one mind-expanding drug..." ⏐`

### 1.2. Nguyên nhân gốc rễ (Root Cause)
1. **Kiến trúc chắp vá, nhiều tầng đối nghịch**:
   - `group_words_to_subtitles` (Bước 2E): Bị khống chế bởi trần từ cứng (`max_words=10`), ép chẻ câu dài bất chấp dấu câu.
   - `merge_continuation`: Cố gắng dán các mảnh chẻ lại với nhau bằng heuristic viết thường / độ dài.
   - `split_subtitles_by_clause_comma` (Bước 2E-bis): Sau khi gộp lại thấy dấu phẩy `,;` lại chẻ ra lần nữa.
   - `_attribute_speakers` (Bước 2F): Diarization phát hiện boundary sai lệch vài trăm ms thì lập tức chẻ đôi câu phụ đề đã hoàn chỉnh.
2. **Cơ chế xử lý quá nhiều lần chuyển đổi**:
   - VAD chạy dry-run quét khoảng lặng rồi lại quét VAD stream, trong khi mốc từ Forced Aligner đã có sẵn độ chính xác cao.
   - Các bản vá trước đây (như thêm danh sách `_SUBCONJ` tiếng Anh) chỉ là giải pháp tạm thời, không triệt để cho đa ngôn ngữ.

---

## 2. Kiến trúc đề xuất mới (Đơn giản, Ổn định, Ngữ nghĩa)

```text
audio_chunk + full_text (từ ASR)
        │
        ▼
   ForcedAligner.align() (1 lần duy nhất trên full chunk)
        │
        ▼
   AlignedWord[]   (text đã merge punctuation, start_time, end_time, confidence)
        │
        ├─ Case 1: Có dấu kết thúc câu (.?!。！？…)
        │     1. Quét dấu kết câu trên AlignedWord[] (bỏ qua từ viết tắt Dr./Mr. & số thập phân)
        │        → Tạo các Sentence Block [lo, hi]
        │     2. Đánh giá độ dài từng Block:
        │        - Nếu ngắn/vừa (<= max_words && <= max_duration): GIỮ NGUYÊN 1 CÂU TRỌN VẸN.
        │        - Nếu quá dài (> max_words hoặc > max_duration):
        │             Tìm dấu phẩy (,;、；) có >= 4-6 từ trước & >= 2 từ sau để ngắt vế.
        │             Nếu không có dấu phẩy: Cắt tại khe âm học lớn nhất trong Block.
        │
        └─ Case 2: Không có (hoặc rất ít) dấu kết thúc + câu dài liên tục
              1. Quét các khe giữa các từ: gap = words[i+1].start - words[i].end
              2. Tách tại các gap >= threshold (khuyến nghị 350–400ms)
              3. Guard: Mỗi đoạn tách ra phải đảm bảo >= min_words (tránh mảnh cụt)
              4. Fallback: Nếu đoạn nào vẫn quá dài, cắt theo dấu phẩy hoặc khe tương đối lớn nhất
        │
        ▼
   Subtitle candidates (List[SubtitleSentence] sạch sẽ, trọn vẹn ngữ nghĩa)
        │
        ▼
   Nemotron-3-Diarization (Chạy trên audio chunk)
        │
        ▼
   Gán Speaker theo Majority-Vote (KHÔNG chẻ đôi câu phụ đề đã hoàn chỉnh)
        │
        ▼
   Translation Engine
```

### Ưu điểm vượt trội:
- **Align chỉ 1 lần**: Tiết kiệm tối đa GPU/CPU, thời gian tính toán ổn định.
- **Xử lý trực tiếp trên mảng `AlignedWord[]`**: Cắt theo index $[lo, hi]$ $\rightarrow$ Lấy luôn `words[lo].start_time`, `words[hi].end_time` và `text`. Không cần so khớp chuỗi ngược lại.
- **Tôn trọng ngữ nghĩa câu**: Ưu tiên giữ trọn câu thay vì ép cắt theo trần cứng. Chỉ câu thật sự dài mới xét cắt vế.
- **Loại bỏ xung đột**: Không còn chu trình "cắt mù quáng $\rightarrow$ gộp lại $\rightarrow$ lại cắt $\rightarrow$ diarization lại chẻ".

---

## 3. Các quy tắc kỹ thuật trọng yếu

1. **Whitelist từ viết tắt & số**:
   - Sử dụng `_ABBREVIATIONS_CI` (`Dr.`, `Mr.`, `Mrs.`, `Prof.`, `vs.`, `etc.`...) và regex kiểm tra số thập phân (`3.14`).
   - Dấu `.` sau các từ này KHÔNG được coi là dấu kết thúc câu.
2. **Quy tắc dấu phẩy `,;、；`**:
   - **Chỉ kích hoạt khi câu QUÁ DÀI** ($> \text{max\_words}$ hoặc $> \text{max\_duration}$).
   - Điều kiện ngắt: Phía trước $\ge 4$ từ, phía sau $\ge 2$ từ.
   - Câu ngắn/vừa có dấu phẩy $\rightarrow$ Giữ nguyên toàn bộ câu.
3. **Quy tắc khoảng lặng (Case 2)**:
   - Dùng khoảng cách giữa 2 từ liên tiếp: `words[i+1].start_time - words[i].end_time >= 0.35s - 0.40s`.
   - Mỗi phân đoạn tối thiểu $\ge 3$ từ.
4. **Quy tắc Diarization Majority-Vote**:
   - Tính tổng thời lượng trùng khớp của từng speaker trong cửa sổ $[start, end]$ của câu phụ đề.
   - Gán speaker có thời lượng lớn nhất.
   - Tuyệt đối không chẻ phụ đề trừ khi điểm chuyển speaker trùng với khoảng lặng lớn $\ge 350\text{ms}$.

---

## 4. Bảng TODO Theo Dõi Tiến Độ

## 5. Bảng TODO & Tiến Độ Triển Khai Thực Tế

### 🚀 Đợt 1: Core Engine & Test TDD (ĐÃ HOÀN THÀNH ✅)
- [x] **TODO 4.1**: Viết unit test mô phỏng 2 lỗi thực tế của user trong `backend/tests/test_60_sentence_segmentation_english.py`:
  - Test 1: `"With our friends moving forward in their relationships, I've decided that we should make progress in ours as well."`
  - Test 2: `"Hope you're not thinking about some sort of LSD thought control, because there's only one mind-expanding drug that this man enjoys,"`
- [x] **TODO 1.1**: Viết hàm `_is_sentence_end_word(words, i, is_cjk)` và `_is_clause_end_word(w_text)` trong `backend/asr/forced_aligner.py`.
- [x] **TODO 1.2**: Viết hàm lõi `segment_aligned_words(words, language, ...)` trong `backend/asr/forced_aligner.py` bao gồm:
  - Tách Case 1 (dấu kết câu $\rightarrow$ ưu tiên trọn câu dưới `sentence_max_*` $\rightarrow$ tách dấu phẩy khi câu dài $\rightarrow$ khe âm học).
  - Tách Case 2 (câu dài không dấu $\rightarrow$ tách theo word gap $\ge 350-400\text{ms}$ $\rightarrow$ fallback trần).
- [x] **TODO 1.3**: Chạy test kiểm chứng TODO 4.1 pass 100%.

### 🚀 Đợt 2: Tích hợp Handler & Bảo vệ Diarization (ĐÃ HOÀN THÀNH ✅)
- [x] **TODO 2.1**: Thay thế Bước 2E cũ (`group_words_to_subtitles`) trong `backend/ws/lookahead_handler.py` bằng `segment_aligned_words`.
- [x] **TODO 2.2**: Gỡ bỏ Bước 2E-bis (`split_subtitles_by_clause_comma`) vì logic đã được tích hợp gọn gàng trong core.
- [x] **TODO 2.3**: Dọn dẹp/gỡ bỏ Bước 2E-ter (loại bỏ VAD pause scan rườm rà).
- [x] **TODO 3.1 & 3.2**: Bảo vệ ranh giới Diarization (`_attribute_speakers`): Gán speaker theo Majority-Vote, cấm chẻ đôi các cụm từ ngắn.

### 🚀 Đợt 3: Hoàn thiện, Dọn dẹp & Nghiệm Thu (ĐÃ HOÀN THÀNH ✅)
- [x] **TODO 1.6**: Revert bản vá cục bộ `_SUBCONJ` trong `split_subtitles_by_clause_comma`.
- [x] **TODO 1.7**: Xóa bỏ triệt để mọi giới hạn 10 từ, trần cứng `max_words`, fallback chẻ đôi câu:
  - Chỉ áp dụng đúng 2 trường hợp:
    1. Case 1 (dấu kết câu `.?!。！？…`): tách câu hoàn chỉnh, nếu có dấu phẩy `,;、；` ($\ge 4$ từ trước, $\ge 2$ từ sau) thì tách vế, không có dấu phẩy thì GIỮ NGUYÊN VẸN CÂU dù dài bao nhiêu từ.
    2. Case 2 (không có dấu kết câu): tách theo word gap $\ge 350-400\text{ms}$, sau đó tách theo dấu phẩy nếu có. Không có gap/phẩy thì giữ nguyên vẹn, không cắt theo trần 10 từ.
- [x] **TODO 4.2**: Chạy kiểm tra toàn bộ suite test liên quan (`test_57`, `test_58`, `test_60`, `test_77`): **87/87 PASSED (100%)**.
- [ ] **TODO 4.3**: Khởi động lại backend server (`Ctrl + C` và chạy lại `python backend\main.py`) để nạp mã nguồn mới.

---

## 6. Tiêu chí nghiệm thu (Acceptance Criteria)
1. **Độ trọn vẹn ngữ nghĩa**: Không còn hiện tượng xuất hiện phụ đề 1-2 từ rời rạc như `"With our"`, `"LSD thought"` hoặc câu dài bị chẻ vụn tại giới hạn 10 từ (`"What can be more special..."`).
2. **Hiệu năng**: Quá trình ngắt câu từ `AlignedWord[]` chạy tức thì ($< 2\text{ms}$ cho khối 30s), ForcedAligner chỉ gọi duy nhất 1 lần.
3. **Độ tương thích**: Hoạt động đồng thời chính xác trên cả tiếng Anh/Latinh và tiếng Nhật/Trung (CJK).
4. **Toàn bộ Test Suites Passed**: Bộ kiểm thử hồi quy đạt **87/87 PASSED (100%)**.

