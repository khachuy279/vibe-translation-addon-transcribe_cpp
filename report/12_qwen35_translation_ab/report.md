# 12 — Thử nghiệm A/B: Qwen3.5-4B / Qwen3.5-9B (UD-Q4_K_XL) cho Pipeline B

- **Ngày**: 2026-10-07
- **Mục đích**: trả lời 3 câu hỏi **trước khi sửa bất kỳ dòng code production nào**:
  1. Qwen3.5 có trả về đúng JSON batch (định dạng Pipeline B) không?
  2. Bản dịch có khớp mong đợi / hiểu ngữ cảnh không?
  3. Bật thinking: được hay mất?
- **Kết luận ngắn**: **KHÔNG nên thay.** Qwen3.5-4B đủ điều kiện *kỹ thuật* (JSON hợp lệ, nhanh) nhưng
  **thua rõ rệt** Index-Translate-9B đang chạy về tên riêng, kính ngữ và độ tự nhiên. **Thinking phải loại bỏ**
  cho Pipeline B (chậm 24–35×, JSON hỏng 100% ở tham số khuyến nghị).

---

## 1. Phần cứng / phần mềm / model

| Hạng mục | Giá trị |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 16 311 MiB (đang dùng ~1.5 GiB khi rảnh) |
| Backend | `llama-cpp-python` 0.3.35 + `backend/bin/llama/llama.dll` |
| Kiến trúc trong DLL | `qwen35`, `qwen35moe`, `qwen3`, `qwen3vl`… → **load được Qwen3.5** |
| `Qwen3.5-4B-UD-Q4_K_XL.gguf` | 2.71 GiB — nạp 1.7 s |
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | 5.56 GiB — nạp 2.9 s |
| `Index-Translate-9B.Q4_K_M.gguf` (đối chứng) | 5.38 GiB — nạp 2.5 s |

**Xác nhận #1 — chạy được, không cần `mmproj`.** GGUF Qwen3.5 của Unsloth là model đa phương thức
(vision tách riêng ra `mmproj-*.gguf`), nhưng **suy luận text-only không cần mmproj**: cả 2 file nạp thẳng
qua `llama_cpp.Llama(model_path=..., n_gpu_layers=-1)`. Không cần build lại llama.cpp, không cần sửa
`backend/bin/llama/`.

**Xác nhận #2 — bật/tắt thinking chỉ là tiền tố lượt `assistant`.** Chat template nhúng trong GGUF:

```
{%- if enable_thinking is defined and enable_thinking is true %}{{- '<think>\n' }}
{%- else %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}
```

Pipeline hiện tại gọi **completion thô** (`llm(prompt)`, không dùng `chat_format`), và
`backend/translation/prompts.py:116` đã dùng đúng chuỗi `"<think>\n\n</think>\n\n"` — tức **không cần
đổi gì để chạy non-thinking**; muốn bật thinking chỉ cần đổi tiền tố đó.

⚠️ **Bẫy đã gặp trong lần chạy đầu**: prompt generic *thiếu* tiền tố no-think ⇒ Qwen3.5 **tự bật
reasoning** dù không được yêu cầu (greedy: 11.3 s, đụng trần 1024 token, **không ra JSON**). Tiền tố
no-think là **bắt buộc**, không phải tuỳ chọn.

---

## 2. Dữ liệu thử

Đúng khối 5 phụ đề trong log thật (JA → VI):

```
1. 美咲さんとはうまくいってるみたいですね。
2. まあおかげさまで会社辞めて一年くらい経つから、
3. もう絶対会社員にも戻れないって言ってるよ。
4. 風呂入って一杯やる か。
5. はい。
```

Bản dịch **mong đợi** của người dùng:

```
1. Có vẻ như mọi chuyện giữa anh và cô Misaki vẫn ổn nhỉ.
2. Ừ, cũng nhờ vậy mà... tính ra đã gần một năm kể từ khi cô ấy nghỉ việc rồi,
3. cô ấy bảo là giờ tuyệt đối không muốn quay lại làm nhân viên công ty nữa.
4. Đi tắm rồi làm một ly nhé?
5. Vâng.
```

Prompt: chấm bằng **chính** `_parse_batch_json` của production. Biến thể `index` = **nguyên văn**
`PipelineBPromptStrategy` (thay thế trực tiếp). Biến thể `generic` = prompt mới viết tại chỗ
(chưa đưa vào production). Ngân sách `index/nothink/prod-budget` = y hệt production
(`max_tokens = min(1536, max(256, 5*80)) = 400`).

---

## 3. Kết quả

### 3.1. JSON (Pipeline B)

| Model | Cấu hình | ms | token ra | finish | JSON qua `_parse_batch_json` |
|---|---|---|---|---|---|
| Index-9B | index / nothink / prod-budget | 1596 | 92 | stop | ✅ |
| **4B** | index / nothink / prod-budget | 1176 | 97 | stop | ✅ |
| 4B | index / nothink / ctx-prev | 1068 | 98 | stop | ✅ |
| 4B | generic / nothink / greedy | 1084 | 98 | stop | ✅ |
| 4B | generic / nothink / rec | 1044 | 98 | stop | ✅ |
| 4B | generic / nothink / rec / ctx-prev | 1005 | 93 | stop | ✅ |
| **4B** | generic / **think** / greedy | **37 832** | 3072 | **length** | ⚠️ "OK" giả (xem 3.3) |
| **4B** | generic / **think** / rec (T=1.0 p=.95 k=20) | **37 589** | 3072 | **length** | ❌ **FAIL** |
| **9B** | index / nothink / prod-budget | 1680 | 93 | stop | ✅ |
| 9B | index / nothink / ctx-prev | 1538 | 92 | stop | ✅ |
| 9B | generic / nothink / greedy | 1483 | 88 | stop | ✅ |
| 9B | generic / nothink / rec | 1468 | 88 | stop | ✅ |
| 9B | generic / nothink / rec / ctx-prev | 1493 | 89 | stop | ✅ |
| **9B** | generic / **think** / greedy | **55 481** | 3072 | **length** | ⚠️ "OK" giả |
| **9B** | generic / **think** / rec | **54 898** | 3072 | **length** | ❌ **FAIL** |

**→ Trả lời câu 1: Non-thinking ĐẠT** (100% JSON hợp lệ, 1.0–1.7 s, ~90–100 token, kể cả ở ngân sách
400 token của production). **Thinking HỎNG**: 37–55 s, luôn đụng trần token, `</think>` không bao giờ
được sinh ra nên không có cách tách JSON an toàn.

### 3.2. Chất lượng bản dịch (không ngữ cảnh)

| # | Mong đợi | Index-9B (production) | Qwen3.5-4B (index) | Qwen3.5-9B (index) |
|---|---|---|---|---|
| 1 | mọi chuyện giữa anh và cô **Misaki** vẫn ổn nhỉ | chuyện với **Misaki-san**… tốt đẹp nhỉ | chuyện với cô **Meisaka**… tốt đẹp | mọi thứ… tốt đẹp với **Misaki** |
| 2 | Ừ, cũng nhờ vậy mà… gần một năm kể từ khi **cô ấy** nghỉ việc | À, cũng nhờ vậy mà **tôi đã** nghỉ việc… | Thật ra, nhờ **có may mắn** nên **tôi đã** nghỉ việc | Chà, may mà đã một năm kể từ khi **tôi** nghỉ việc |
| 3 | **cô ấy bảo** là… không muốn quay lại… **nữa** | **tôi** còn nói là… không thể quay lại… nữa đâu | **Tôi** nói là… không thể trở lại… nữa | **Tôi đã nói** là… không thể quay lại… nữa |
| 4 | Đi tắm rồi làm một ly nhé? | Hay là vào tắm rồi uống một ly nhé? | **Bạn** muốn đi tắm và uống một chút không? | Có muốn vào tắm và uống một ly không? |
| 5 | Vâng. | Vâng. | **Có.** | Vâng. |

Nhận xét:

- **Tên riêng**: Index-9B giữ đúng `Misaki-san`. 4B bịa thành `Meisaka`, `Misa`, `Ms. Misaki`,
  `Misa-chan`; 9B thành `Miaki-san`, `Misaki-chan`. Với phụ đề phim đây là lỗi nặng.
- **Ngôi/chủ thể**: cả 3 model đều chọn sai chủ thể ở câu 2–3 (chọn "tôi" thay vì người thứ ba).
  Đây là **vấn đề ngữ cảnh**, không phải vấn đề cỡ model.
- **Câu 4**: 4B thêm "Bạn…" (sai ngôi, câu này là lời rủ rê chung).
- **Câu 5**: 4B trả "Có." thay vì "Vâng.".
- **Điểm cộng duy nhất của 4B**: **không vi phạm ràng buộc `"mình"`** — trong khi Index-9B ở log thật
  đã vi phạm (`"tôi còn nói là mình chắc chắn…"`). Đây là lý do chính đáng để siết prompt, không phải
  để đổi model.

**→ Trả lời câu 2: KHÔNG đạt.** 4B không khớp mong đợi (sai tên riêng + sai ngôi + sai sắc thái câu 5).
9B khá hơn nhưng vẫn chưa bằng Index-Translate-9B.

### 3.3. Vì sao "JSON(raw)=OK" ở chế độ thinking là **dương tính giả**

Ở `think/greedy`, `finish_reason=length` (bị cắt), không có `</think>` trong output. `_parse_batch_json`
tìm cặp `{…}` đầu tiên nên **bắt được bản JSON nháp nằm giữa phần reasoning** — không phải kết quả cuối.
Đúng loại lỗi im lặng đã dự đoán: reasoning + JSON batch là combo dễ vỡ nhất.

Trích output thinking của 4B (greedy, bị cắt ở 3072 token):

```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Role: Professional subtitle translator.
    ...
2.  **Analyze the Input Subtitles:**
   ...
    Better subtitle style: "Như thế nào? Vào tắm rồi đi chơi một chút?" -> ...
```

### 3.4. Ngữ cảnh (context injection)

Hai chế độ:

- `ctx-ref` — **cố ý rò rỉ đáp án** (đưa chính 2 bản dịch MONG ĐỢI vào context).
  Kết quả: **cả 3 model đều nhả ra gần đúng mong đợi**, kể cả Index-9B:
  `"Có vẻ như mọi chuyện giữa anh và cô Misaki vẫn ổn nhỉ."` / `"Ừ, cũng nhờ vậy mà... tính ra đã gần
  một năm kể từ khi cô ấy nghỉ việc rồi,"`.
  → Chứng minh **cơ chế `_CONTEXT_HEADER` + `use_context` của Pipeline B hoạt động tốt**; khoảng cách
  chất lượng nằm ở **chất lượng ngữ cảnh**, không ở model.
- `ctx-prev` — **ngữ cảnh sạch** (2 lượt thoại trước khối, không chứa đáp án):
  `田中さん、最近どうですか。 -> Anh Tanaka, dạo này thế nào?` / `彼、元気にしてるの？ -> Anh ấy có ổn không?`

  Kết quả **tệ hơn** ở nhiều trường hợp:
  - Qwen3.5-9B (generic/rec): câu 1 → `"Misaki-chan"`, câu 3 → **`"Nó bảo chắc chắn không thể…"`**
    (`言ってる` bị dịch thành "Nó bảo" — sai hẳn).
  - Qwen3.5-4B (index): câu 1 → `"Misa-san và cô ấy đang hợp tác tốt"` (sai nghĩa), câu 2 →
    `"nhờ có sự giúp đỡ này, tôi đã nghỉ việc"` (bịa).

  → Ngữ cảnh **không liên quan/chung chung làm nhiễu** model nhỏ. `lookahead_handler.py:1479-1491`
  hiện đưa 2 lượt gần nhất dạng `orig -> trans`; nên thử giảm còn 1 lượt hoặc chỉ đưa khi cùng chủ đề.

**→ Trả lời câu 3: thinking không giúp ích gì cho việc hiểu ngữ cảnh** — nó chỉ làm chậm và hỏng JSON.
Thứ cải thiện được chủ thể/đại từ là **ngữ cảnh đúng**, và điều đó **đã có sẵn** trong kiến trúc.

---

## 4. Kết luận & khuyến nghị

### Không nên thay model

| Tiêu chí | Index-Translate-9B | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---|---|
| JSON batch hợp lệ | ✅ | ✅ | ✅ |
| Độ trễ (5 câu) | ~1.6 s | ~1.1 s | ~1.5 s |
| Tên riêng / kính ngữ | ✅ tốt nhất | ❌ bịa tên | ⚠️ hay sai |
| Tự nhiên, đúng sắc thái | ✅ tốt nhất | ⚠️ | ⚠️ |
| Tuân thủ "không dùng *mình*" | ❌ vi phạm | ✅ | ✅ |
| Thinking khả dụng cho batch | — | ❌ | ❌ |

4B nhanh hơn ~0.5 s/khối nhưng **đánh đổi bằng tên riêng và ngôi** — không đáng, nhất là khi
GPU còn thừa VRAM (16 GiB, model chỉ dùng 5.4 GiB).

### Việc nên làm thay vào đó (theo thứ tự chi phí/lợi ích)

1. **Siết ràng buộc "không dùng mình"** cho Index-Translate. Đây là khiếm khuyết thật trong log.
   Ràng buộc đã có ở `prompts.py:45-48` nhưng đặt *trước* `{source_text}`; thử nhắc lại **sau** khối
   JSON (recency) hoặc thêm hậu kiểm `"mình" -> "tôi"` như một bước dọn văn bản.
2. **Glossary tên riêng** (`美咲 -> Misaki`) — Pipeline A đã có `TEMPLATE_TERMINOLOGY` + tham số `term`,
   nhưng `build_batch_prompt` của Pipeline B **không truyền `term`**. Đây là lỗ hổng cụ thể, sửa rẻ.
3. **Chọn lọc ngữ cảnh**: 2 lượt gần nhất có thể gây nhiễu (mục 3.4). Thử A/B 0/1/2 lượt.
4. Nếu vẫn muốn thử Qwen3.5: chỉ **9B non-thinking**, và phải chứng minh thắng Index-9B trên
   bộ mẫu lớn hơn — dữ liệu hiện tại chưa ủng hộ.

> **Cập nhật 2026-10-07:** mục 1 và 2 ở trên **đã được triển khai** — xem
> `report/13_pronoun_and_glossary`. Kết quả nghiệm thu: glossary sửa được lỗi bịa tên
> (4/6 → 0/6 lần sai trên Qwen3.5-4B), còn ràng buộc `mình` **chỉ** được bảo đảm bởi hậu kiểm
> tầng văn bản (`pronoun_guard`), không phải bởi việc siết câu chữ trong prompt.

### Nếu vẫn muốn theo Qwen3.5, đây là các chỗ **buộc phải** sửa

- Tiền tố no-think là **bắt buộc** (đã chứng minh ở mục 1).
- Không dùng thinking cho batch JSON.
- Prompt riêng cho họ Qwen3.5 (`prompt_style` mới) — template `index` là prompt fine-tune của
  Index-Translate, không tối ưu cho model generic.
- Không có `_strip_think()` trong `_translate_sync` (`engine.py:553-556`) hay `translate_batch_sync`.
- `test_05_translation_benchmark.py:34` assert `len(models) == 2` → thêm model vào YAML là vỡ test.

---

## 5. Tái lập thí nghiệm

```powershell
# Tải model (chỉ tải, không sửa code)
.\.venv\Scripts\python.exe debug\dl_qwen35.py

# Chạy A/B
.\.venv\Scripts\python.exe debug\qwen35_ab.py --model 4b
.\.venv\Scripts\python.exe debug\qwen35_ab.py --model 9b
.\.venv\Scripts\python.exe debug\qwen35_ab.py --model index9b --configs index/nothink

# Xem raw output từng cấu hình
.\.venv\Scripts\python.exe debug\qwen35_show.py debug\qwen35_ab_4b.json
```

Kết quả thô (JSON, gồm `raw_output` đầy đủ): `debug/qwen35_ab_{4b,9b,index9b}.json`.

> **Không có dòng code production nào bị sửa.** Toàn bộ thí nghiệm nằm trong `debug/`.
> Model tải về `backend/models/` (~8.3 GiB) — xoá được nếu không dùng nữa.
