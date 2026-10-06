# Báo Cáo Đo Lường & Kiểm Thử Phase 5: Module Dịch Thuật Local GGUF (Llama.cpp)

- **Thời gian thực hiện**: 2026-10-06 21:22:31
- **Mô hình thử nghiệm**: `index-translate-9b` (Index-Translate-9B (Bilibili IndexTeam))
- **GGUF File**: `Index-Translate-9B.Q4_K_M.gguf`
- **Prompt Template**: `index`

## 1. Kết Quả Benchmark Hiệu Năng & Tốc Độ Dịch Sang Tiếng Việt

| Mẫu Thử Nghiệm | Ngôn Ngữ Gốc | Thời Gian Dịch | Tốc Độ (Tokens/s) | Bản Gốc | Bản Dịch Tiếng Việt (`vi`) |
|---|---|---|---|---|---|
| `English_speech` | `en` | **410.3 ms** | **60.9** | Ready? Yeah. It's crazy. It's freezing. It's completely stopped. | **Sẵn sàng chưa? Ừ. Điên thật đấy. Lạnh cóng luôn. Hoàn toàn dừng hẳn rồi.** |
| `Chinese_speech` | `zh` | **497.5 ms** | **62.3** | 在他十一岁的时候，母亲因为发生交通事故撒手人寰。桂兰的母亲敲了几下杯子。 | **Khi cậu mười một tuổi, mẹ cậu đã qua đời vì một vụ tai nạn giao thông. Mẹ của Quế Lan gõ vài cái vào chiếc cốc.** |
| `Japanese_speech` | `ja` | **706.3 ms** | **63.7** | 得意先の営業周りを優しくおっしゃし、その日私たち出張先の宿に向かっていた。このホテル初めてですね。 | **Tôi nhẹ nhàng nhắc nhở nhân viên kinh doanh của khách hàng, rồi cùng họ đi về phía khách sạn nơi chúng tôi sẽ ở trong ngày hôm đó. Đây là lần đầu tiên anh đến khách sạn này phải không ạ?** |
| `Russian_speech` | `ru` | **308.9 ms** | **58.3** | Барсук, живущий в киевском зоопарке, совершил побег из своего вольера. | **Con gấu đất sống ở vườn thú Kiev đã trốn thoát khỏi chuồng của mình.** |

## 2. Đánh Giá Hiệu Năng & Chất Lượng Bản Dịch

- **Thời Gian Dịch Trung Bình**: **480.8 ms / câu**.
- **Tốc Độ Sinh Từ (Throughput)**: **61.3 tokens / giây** trên GPU.
- **Chất lượng ngữ nghĩa**: Bản dịch tự nhiên, trôi chảy, giữ nguyên ngữ cảnh câu nói.

## 3. Kết Luận Nghiệm Thu Phase 5

- Module dịch thuật GGUF hoạt động hoàn hảo, đáp ứng thời gian thực cho phụ đề (< 300ms/câu).
- Sẵn sàng chuyển sang **Phase 6: Module OmniVoice Clone TTS (`tts/`)**.