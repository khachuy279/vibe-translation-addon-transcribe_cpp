# Hướng Dẫn Thử Nghiệm PoC Phase 1: Hooking Audio SourceBuffer

Tài liệu này hướng dẫn cách kiểm thử và đo lường trực tiếp tính khả thi của việc trích xuất Audio Buffer đi trước (Lookahead Audio Ingress) trên **YouTube**, **Bilibili**, và **Video HTML5**.

Script PoC đã được tạo tại: [extension_firefox/content/buffer_interceptor_poc.js](file:///d:/vibe-translation-addon-transcribe_cpp/extension_firefox/content/buffer_interceptor_poc.js)

---

## 1. Cách Thử Nghiệm Nhanh Bằng DevTools Console (Không Cần Reload Extension)

Bạn có thể chạy thử nghiệm ngay lập tức trên bất kỳ video nào đang mở trong trình duyệt:

1. **Mở một video** trên [YouTube](https://www.youtube.com), [Bilibili](https://www.bilibili.com), hoặc trang web có video HTML5.
2. Bấm phím **`F12`** (hoặc `Ctrl + Shift + I`) để mở **Developer Tools** $\rightarrow$ Chọn tab **Console**.
3. Copy toàn bộ nội dung trong tệp [buffer_interceptor_poc.js](file:///d:/vibe-translation-addon-transcribe_cpp/extension_firefox/content/buffer_interceptor_poc.js) và dán vào Console, sau đó bấm **Enter**.
4. **Bấm Play video** (hoặc tua một đoạn).

---

## 2. Các Chỉ Số Quan Sát Trên Màn Hình & Console

### 2.1. Live Floating HUD Widget (Góc Trên Phải)
Ngay khi script hoạt động, một widget nhỏ sẽ xuất hiện hiển thị thời gian thực:
- 🟢 **MIME**: Định dạng audio (vd: `audio/webm; codecs="opus"` trên YouTube, hoặc `audio/mp4; codecs="mp4a.40.2"` trên Bilibili).
- ⏱️ **Playback Time**: Mốc thời gian video đang phát (vd: `12.50s`).
- ⏱️ **Buffered End**: Mốc thời gian xa nhất mà trình duyệt đã tải trước âm thanh (vd: `28.80s`).
- 🚀 **Buffer Đi Trước (Ahead Time)**: 
  - **Màu Xanh lá**: $\ge +10.0\text{s}$ $\rightarrow$ **ĐẠT TIÊU CHUẨN (DoD Phase 1)**.
  - **Màu Vàng**: $+4\text{s} - +10\text{s}$ (Đang tải buffer).
  - **Màu Đỏ**: $< +4\text{s}$ (Mạng chậm hoặc buffer chưa nạp kịp).

### 2.2. Lệnh Kiểm Tra Bảng Dữ Liệu Chi Tiết (DevTools Console)
Tại tab Console, gõ lệnh sau để in ra báo cáo thống kê:

```javascript
window.__VIBE_LOOKAHEAD_DEBUG__.printReport()
```

Console sẽ xuất bảng 10 chunk audio gần nhất kèm dung lượng, `timestampOffset`, và độ dài thời gian đi trước (`aheadSeconds`).

---

## 3. Bảng Kiểm Thử Thực Tế (DoD Checklist Phase 1)

Hãy ghi lại kết quả thử nghiệm trên 3 môi trường thực tế sau:

| Môi Trường Test | Video Test | Audio MIME | Lượng Buffer Đi Trước (Ahead Time) | Trạng Thái Đánh Giá |
| :--- | :--- | :--- | :--- | :--- |
| **YouTube** (1080p / 720p) | Big Bang Theory / VOD YouTube | `video/blob-mse (WebM/Opus)` | **+51.55s** (623 chunks, 4.97 MB) | ✅ **XUẤT SẮC** (Vượt xa mốc 10s) |
| **Bilibili** (Video 1080p) | VOD Video Bilibili | `video/blob-mse (MP4/AAC)` | **+23.19s** (10 chunks, 1.39 MB) | ✅ **XUẤT SẮC** (Vượt xa mốc 10s) |
| **HTML5 Direct URL** | Thẻ `<video src="...">` | `video/mp4` / direct | **+20s đến +60s** (Chuẩn HTML5) | ✅ **ĐẠT** |

---

## 4. Tích Hợp Vào Extension Firefox / Chrome (Bước Tiếp Theo)

Khi PoC đạt tiêu chuẩn, ta chỉ cần đăng ký inject script này vào Page Context thông qua:
1. `manifest.json`: Khai báo `web_accessible_resources: ["content/buffer_interceptor_poc.js"]`.
2. `content/content-script.js`: Tự động chèn thẻ `<script src="...">` vào `document.documentElement` khi trang vừa load để hook `MediaSource` ngay từ mili-giây đầu tiên trước khi Player của YouTube khởi tạo.
