# 15 — Ràng buộc "không tự gán chủ thể": có nên thêm vào Pipeline B?

- **Ngày**: 2026-10-08
- **Câu hỏi**: *"Nếu ràng buộc thêm không có chủ thể thì không tự gán chủ thể vào bản dịch? Nó có
  thể dịch không mượt nhưng ít nhất sẽ không sai ngữ cảnh."*
- **Trả lời ngắn**: **Ý tưởng đúng hướng, nhưng cách làm đó KHÔNG hiệu quả — đã đo.** Ràng buộc
  trong prompt batch **không sửa được ca mục tiêu** (6/6 lần vẫn sai), lại còn **xoá mất "tôi"
  đúng** ở 2 chỗ khác. Có một biến thể **hiệu quả thật**, nhưng nó hẹp hơn: **chỉ áp cho câu có
  dấu hiệu thuật lại, và chỉ khi gọi lại riêng câu đó.**

Model: `Index-Translate-9B.Q4_K_M`, greedy. Script: `debug/subject_ab.py`,
`debug/subject_focus.py`, `debug/subject_retry.py`.

---

## 1. Vì sao ý tưởng này hợp lý về mặt ngôn ngữ

Tiếng Nhật lược chủ thể liên tục, và **tiếng Việt cũng cho phép lược chủ thể**. Nên
"không thêm chủ thể" không tạo ra câu sai ngữ pháp:

```
会社辞めて一年くらい経つから、   →  "Nghỉ việc được khoảng một năm rồi nên…"     ✅ tự nhiên
相変わらず何も調べてないんだ。  →  "Vẫn như mọi khi, chẳng tìm hiểu gì cả."      ✅ tự nhiên
```

Và đúng là **chủ thể sai tệ hơn chủ thể lược**. Đó là tiền đề hợp lý.

## 2. Nhưng có ba cái bẫy

**(a) Nhiều chỗ thêm "tôi" là ĐÚNG, không phải bịa.** `〜と思う`, `自分`, `私` ngầm định người nói.
Trong dữ liệu thật: `会社を辞めたのは正しい判断だったと思う。` → "**Tôi** nghĩ…" là đúng;
`自分の気持ちに正直になりたい。` → "**Tôi** muốn thành thật với cảm xúc của chính mình" là đúng.
Cấm tuyệt đối sẽ sửa cái đúng thành cái lúng túng.

**(b) Lược chủ thể ≠ không sai ngữ cảnh.** Trong phụ đề nối tiếp, một câu không chủ thể đứng ngay
sau câu nói về 花田 sẽ bị người xem **gán cho 花田**. Rủi ro **chuyển chỗ** chứ không mất. "Không
nói gì" không phải "an toàn".

**(c) Không hậu kiểm được.** Khác `mình` (thay được bằng regex), "câu này có thêm chủ thể không"
cần **alignment nguồn↔đích** — không làm chắc được. Nên đây chỉ có thể là ràng buộc prompt, tức
là **không có bảo đảm** — đúng bài học P4.1.

---

## 3. Đo lần 1 — 10 khối log thật, 46 câu (`subject_ab.py`)

Ba biến thể: **A** = prompt hiện tại · **B** = + "đừng tự gán chủ thể" · **C** = B + luật thuật lại
(`〜と言っている`, `〜って言ってた`, `〜らしい`, `〜みたい`, `〜そうだ`).

| Biến thể | Câu thêm ngôi thứ nhất (proxy) | JSON hợp lệ | Tổng ms |
|---|---|---|---|
| A baseline | **7/46** (0,15) | 9/10 | 13 224 |
| B no-subject | 6/46 (0,13) | 10/10 | 13 360 |
| C no-subject + reported | **5/46** (0,11) | 10/10 | 13 593 |

Cải thiện **rất nhỏ** (7→5) và **hai trong số "cải thiện" đó là xoá "tôi" ĐÚNG**:

| Câu | Baseline | Sau ràng buộc |
|---|---|---|
| `五年先輩の花田さんとは。` | "…người đi trước **tôi** năm năm…" ✅ | bỏ "tôi" ❌ (先輩 là tương quan với người nói) |
| `会社を辞めたのは正しい判断だったと思う。` | "**Tôi** nghĩ…" ✅ | vẫn "Tôi" (không đổi) |

⚠️ Chỉ số proxy có dương tính giả của chính nó: `と思う`/`自分` không chứa 私/僕/俺 literal nên bị
đếm là "thêm chủ thể" dù đúng.

---

## 4. Đo lần 2 — đúng ca mục tiêu, CÓ ngữ cảnh (`subject_focus.py`)

`subject_ab.py` chạy **không context**, không đại diện production. Đo lại khối gốc với ngữ cảnh
THẬT lấy từ log (`うん、まあね。` / `彼はそういう人だった。` — khối ngay trước đó):

Câu mục tiêu: `もう絶対会社員にも戻れないって言ってるよ。` (thuật lại — chủ thể **không** phải người nói)

| Ngữ cảnh | A baseline | C (+ ràng buộc) |
|---|---|---|
| không context | ❌ "tôi còn nói là tôi chắc chắn…" | ❌ "tôi còn nói là tôi chắc chắn…" |
| 2 lượt (production) | ❌ "tôi còn nói là chắc chắn…" | ❌ "tôi còn nói là chắc chắn…" |
| 4 lượt | ❌ "tôi còn nói là tôi chắc chắn…" | ❌ "tôi còn nói là tôi chắc chắn…" |

**6/6 thất bại.** Cả ràng buộc lẫn ngữ cảnh đều **không** sửa được. Ở chế độ batch, model tối ưu
độ trôi chảy của cả khối nên bỏ qua ràng buộc phủ định kiểu này.

---

## 5. Đo lần 3 — gọi lại MỘT CÂU khi có dấu hiệu thuật lại (`subject_retry.py`)

Giả thuyết: gọi riêng câu đó (Pipeline A, 1 câu) thì model tuân thủ tốt hơn.

| Ca | 1 câu — baseline | 1 câu — **+ràng buộc thuật lại** |
|---|---|---|
| ⓐ `…戻れないって言ってるよ。` (thuật lại) | ❌ "**Tôi** đã nói rồi, tôi tuyệt đối…" | ✅ "**Anh ấy** nói rằng mình sẽ không bao giờ…" |
| ⓑ `あいつが俺のイビキ…って。` | ✅ "tiếng ngáy **của tôi**" (đúng — 俺) | ✅ giữ nguyên |
| ⓒ `温泉がいいらしいよ。` | ✅ | ✅ |
| ⓓ `…正しい判断だったと思う。` (đối chứng) | ✅ "**Tôi** nghĩ…" (đúng) | ⚠️ đổi thành "việc **anh ấy** nghỉ việc" |
| ⓔ `自分の気持ちに正直になりたい。` (đối chứng) | ✅ "**Tôi** muốn thành thật…" (đúng) | ❌ "**Anh ấy** nói rằng muốn thành thật…" |
| ⓕ `彼女が来るかどうかは分からない。` | ✅ "Không biết cô ấy có đến hay không." | ✅ |

**Kết luận then chốt:**
* Ràng buộc **CÓ tác dụng** khi câu thật sự có lời thuật lại (ⓐ được sửa đúng).
* Nhưng nó **áp dụng quá rộng** và **phá câu đối chứng** (ⓔ: `自分` bị biến thành "Anh ấy").
* ⇒ Chỉ dùng được khi **cổng bằng bộ phát hiện**: chỉ gọi lại câu khớp
  `(って言|と言|だって|らしい|みたい|そうだ|んだって|とのこと)`. Câu ⓔ không khớp ⇒ không áp ⇒ giữ bản đúng.

---

## 6. Khuyến nghị

**KHÔNG thêm ràng buộc "không tự gán chủ thể" vào prompt batch** — dữ liệu không ủng hộ (không sửa
được ca mục tiêu, lại xoá "tôi" đúng ở 2 chỗ, tốn thêm 280–480 ký tự prompt).

**Nên làm thay vào đó — thiết kế "phát hiện + gọi lại 1 câu":**

1. Sau khi có bản dịch batch, quét **câu nguồn** khớp regex thuật lại.
2. Với câu khớp, kiểm tra **bản dịch có đại từ ngôi thứ nhất** trong mệnh đề ma trận.
   (Hẹp: `^\s*(tôi|mình|tớ)\b` hoặc `\b(tôi|mình)\b.{0,15}\b(nói|bảo|kể)\b`.)
3. Chỉ khi cả hai đúng ⇒ **gọi lại riêng câu đó** kèm chỉ dẫn thuật lại. Thay kết quả vào khối.
4. Không khớp ⇒ **không làm gì** (giữ nguyên hành vi hiện tại).

Chi phí: ~200–400 ms cho mỗi câu bị cổng bắt (hiếm — trong 46 câu log chỉ 1–2 câu). Pipeline B có
lead 8–30 s nên hoàn toàn chịu được. Rủi ro: thấp, vì mặc định là "không làm gì".

**Không nên** làm bước 3 mà bỏ bước 2 (đã chứng minh là phá câu đúng), và **không nên** cấm chủ
thể nói chung (bẫy (a)).

---

## 7. Tái lập

```powershell
.\.venv\Scripts\python.exe debug\subject_ab.py    --model index9b
.\.venv\Scripts\python.exe debug\subject_focus.py --model index9b
.\.venv\Scripts\python.exe debug\subject_retry.py --model index9b
```
