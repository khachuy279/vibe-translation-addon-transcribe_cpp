# 14 — Index-Translate-2B vs 9B: có thật sự "không khác nhau mấy" không?

- **Ngày**: 2026-10-07
- **Câu hỏi**: "mình thấy 2 cái không khác nhau là mấy?"
- **Dữ liệu**: 21 câu phụ đề Nhật chia 4 nhóm độ khó + khối 5 câu thật từ log
  ⇒ **26 câu / model**, chạy 2 lần (bật và tắt glossary), greedy (`T=0, top_k=1`), cùng ngân sách
  production (`max_tokens = min(1536, max(256, n*80))`).
- **Model**: `Index-Translate-2B.Q8_0` (2.07 GiB) và `Index-Translate-9B.Q4_K_M` (5.38 GiB).

> ⚠️ n = 26 câu. Đủ để thấy XU HƯỚNG và chỉ ra loại lỗi, **không** phải một benchmark.

---

## 1. Kết luận ngắn

**Cảm nhận của bạn đúng một nửa.** Hai model gần như không khác nhau ở câu ngắn/đơn giản —
nhưng khác rõ ở câu khó, và 9B chậm hơn **2,1–2,3×**.

| Nhóm câu | Số câu | Giống hệt | Tương đồng ký tự | 2B | 9B | 9B/2B |
|---|---|---|---|---|---|---|
| `simple` (chào hỏi, câu ngắn) | 5 | 2 | **0.92** | 632 ms | 1221 ms | 1.9× |
| `long` (câu dài, nhiều mệnh đề) | 5 | 0 | **0.89** | 747 ms | 1628 ms | 2.2× |
| `block` (5 câu thật từ log) | 5 | 1 | 0.83 | 722 ms | 1572 ms | 2.2× |
| `names` (tên riêng + kính ngữ) | 3 | 0 | 0.75 | 374 ms | 880 ms | 2.4× |
| `tricky` (nghĩa bóng, thành ngữ) | 8 | 0 | **0.64–0.69** | 1075 ms | 2254 ms | 2.1× |
| **TỔNG** | **26** | **3 (12%)** | **0.79–0.81** | 3550 ms | 7556 ms | **2.13×** |

- Chỉ **3/26 câu (12%)** giống hệt nhau — nhưng **tương đồng ký tự 0.79** nghĩa là phần lớn
  chỉ là **diễn đạt lại**, cùng nghĩa. Đó chính xác là lý do bạn thấy "không khác mấy".
- Tốc độ sinh: **2B ≈ 130–137 tok/s**, **9B ≈ 57–61 tok/s**.
- JSON batch: **5/5 nhóm hợp lệ** ở cả hai model, cả hai chế độ glossary.
- Glossary (`Misaki` / `Tanaka` / `Sato`): **cả hai đều tuân thủ**.

---

## 2. Khác biệt THẬT (không phải cảm giác)

### 9B thắng — câu khó, nghĩa bóng, ngữ pháp điều kiện

| Câu nguồn | 2B | 9B | |
|---|---|---|---|
| 美咲さんとはうまくいってるみたいですね。 | "cô Misaki **đang đi rất tốt với anh ấy**" | "mọi chuyện với Misaki **đang diễn ra tốt đẹp**" | 9B ✅ |
| 自分の気持ちに正直になりたい。 | "**nói thật** với cảm xúc của mình" | "**thành thật** với cảm xúc của chính mình" | 9B ✅ |
| そんなの知るかよ。 | "Biết cái gì mà biết chứ!" | "Ai mà quan tâm chứ!" | 9B ✅ |
| この設定を変更すると、… | "**Sau khi** thay đổi cài đặt này…" | "**Nếu** thay đổi cài đặt này…" | 9B ✅ |
| 彼女が来るかどうかは分からない。 | "Tôi vẫn chưa biết cô ấy có đến hay không." | "Cô ấy có đến hay không thì tôi **cũng** không biết." | 2B nhỉnh hơn |

`うまくいってる` = quan hệ tốt đẹp, **không** phải "đang đi". `正直` = thành thật, **không** phải
"nói thật". `〜すると` ở đây là điều kiện, **không** phải trình tự.

### 2B thắng — vài chỗ đáng chú ý

| Câu nguồn | 2B | 9B | |
|---|---|---|---|
| 佐藤部長に報告しておきます。 | "báo cáo cho **Trưởng phòng** Sato" | "báo cáo lại với **giám đốc** Sato" | 2B ✅ |
| 田中さんはまだ来ていません。 | "**Anh Tanaka** vẫn chưa đến." | "**Tanaka-san** vẫn chưa đến." | 2B ✅ |
| 会社を辞めたのは正しい判断だったと思う。 | "việc **rời khỏi** công ty" | "việc **nghỉ việc ở** công ty" (lặp từ) | 2B ✅ |

`部長` = trưởng phòng/trưởng bộ phận, **không** phải giám đốc (`部長` ≠ `社長`/`取締役`).
Giữ nguyên `Tanaka-san` trong phụ đề tiếng Việt là kém tự nhiên.

### Cả hai đều sai — ca `mình` đúng như log của bạn

| | Câu 3 của khối |
|---|---|
| 2B | "**Cô ấy nói là tôi** chắc chắn không thể trở lại làm nhân viên công ty nữa đâu." — lẫn chủ thể (người nói là "cô ấy", chủ thể là "tôi") |
| 9B | "tôi còn nói là **mình** chắc chắn không thể quay lại làm nhân viên văn phòng nữa đâu." — vi phạm ràng buộc đại từ |

⇒ Ca thật trong log **chỉ 9B** tái hiện được lỗi `mình`; 2B tránh được `mình` nhưng trả giá bằng
một câu **sai chủ thể**. Không model nào giải quyết được vấn đề chủ thể bị lược bỏ.

---

## 3. Khuyến nghị

| Bối cảnh | Chọn | Lý do |
|---|---|---|
| **Pipeline B (Lookahead)** | **9B** | Audio đã nằm sẵn trong buffer nên độ trễ không phải ràng buộc; 2,2× chậm hơn nhưng đúng hơn ở đúng những câu khó. 9B vẫn giữ `Misaki-san` tự nhiên hơn. |
| **Pipeline A (realtime từng câu)** | **2B** cân nhắc | Ưu tiên độ trễ tuyệt đối; ~130 tok/s vs ~60 tok/s. Đổi lại phải chấp nhận lỗi nghĩa bóng. |
| Máy thiếu VRAM | 2B | 2,07 GiB vs 5,38 GiB |

**Đừng kết luận "2B = 9B" từ câu ngắn.** Ở nhóm `simple` chúng giống nhau 0,92 — nhưng ở
nhóm `tricky` chỉ 0,64 và đó mới là những câu quyết định chất lượng phụ đề.

> Lưu ý về quantization: 2B ở đây là **Q8_0** (bản chất lượng cao nhất của 2B). Nếu dùng
> `Index-Translate-2B.Q4_K_M` (1,31 GiB, cũng có sẵn trong repo) thì khoảng cách với 9B sẽ
> **rộng hơn** kết quả trên.

---

## 4. Tái lập

```powershell
.\.venv\Scripts\python.exe debug\ab_2b_9b.py            # chạy đầy đủ, ghi debug/ab_2b_9b.json
.\.venv\Scripts\python.exe debug\ab_2b_9b_summary.py    # bảng tổng hợp theo nhóm
```
