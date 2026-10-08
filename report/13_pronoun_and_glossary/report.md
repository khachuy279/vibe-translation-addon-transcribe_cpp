# 13 — P4.1 (ép ràng buộc đại từ) & P4.2 (glossary tên riêng) cho tầng dịch

- **Ngày**: 2026-10-07
- **Tiền đề**: `report/12_qwen35_translation_ab` — kết luận *không đổi model*; chất lượng dịch phải
  cải thiện bằng (1) siết ràng buộc đại từ và (2) bảng tên riêng, chứ không phải bằng Qwen3.5.
- **Trạng thái**: đã triển khai, 626 test tầng A xanh, có nghiệm thu trên model thật.

---

## 1. Vấn đề gốc

| # | Hiện tượng (log thật) | Nguyên nhân |
|---|---|---|
| P4.1 | Prompt ghi rõ `never use the word mình anywhere` nhưng Index-Translate-9B vẫn trả `"tôi còn nói là **mình** chắc chắn…"` | Ràng buộc chỉ nằm ở prompt = **hy vọng**, model không có gì bắt buộc phải tuân |
| P4.2 | `美咲さん` → `Meisaka` / `Misa` / `Miaki-san` tuỳ model | Không có nguồn sự thật nào cho cách viết tên riêng; `TEMPLATE_TERMINOLOGY` của upstream **chưa từng được nối vào Pipeline B** |

---

## 2. Đã thay đổi gì

| File | Thay đổi |
|---|---|
| `backend/translation/pronoun_guard.py` | **Mới** — hậu kiểm đại từ ở tầng văn bản |
| `backend/translation/glossary.py` | **Mới** — nạp bảng thuật ngữ + phát hiện tên riêng + dựng hint |
| `backend/translation/romaji.py` | **Mới** — romaji hoá tên riêng **từ chuỗi nguồn** (P4.2b) |
| `backend/glossary.yaml` | **Mới** — bảng tên riêng/thuật ngữ, sửa là có hiệu lực ngay |
| `backend/translation/prompts.py` | Siết `_PRONOUN_GUIDANCE_VI`; thêm khối ràng buộc cứng **sau payload** cho Pipeline B; nhận `terms` |
| `backend/translation/engine.py` | Nối glossary vào cả 3 đường (`_translate_sync`, batch, streaming) + hậu kiểm đại từ |
| `backend/config.py` | `enforce_pronoun_policy`, `glossary_file`, `glossary`, `glossary_derive_names` |
| `backend/translation/lifecycle.py` | Chép các trường mới khi hot-swap model (nếu không sẽ âm thầm về mặc định) |
| `backend/requirements.txt` · `constraints.txt` | Thêm `pykakasi` (thuần Python, thuộc tính TUỲ CHỌN) |
| `backend/tests/test_65_pronoun_and_glossary.py` | **Mới** — 36 test |
| `backend/tests/test_66_name_romaji.py` | **Mới** — 47 test |

### 2.1. P4.1 — hai lớp, một bảo đảm

Lớp 1 (prompt): `_PRONOUN_GUIDANCE_VI` nêu đích danh `chúng mình` / `bọn mình` / `tụi mình` và ánh
xạ đại từ nguồn tiếng Nhật (私/僕/俺/自分 → *tôi*).
Lớp 2 (văn bản): `enforce_no_minh()` — **đây mới là bảo đảm**. Áp trên cả 3 đường dịch.

Điểm khó của lớp 2 là **không được thay bừa**: rất nhiều cụm hợp lệ chứa `mình` mà không phải đại
từ. Nên dùng **danh sách từ đứng trước bị CHẶN** thay vì thay tất:

| Nguồn | Kết quả | |
|---|---|---|
| `một mình` · `chính mình` · `tự mình` · `bản thân mình` · `nhà mình` · `mỗi mình` | **giữ nguyên** | thay bừa ⇒ "một tôi", "chính tôi" ❌ |
| `mình ơi` | **giữ nguyên** | hô ngữ, không phải đại từ |
| `chúng mình` · `bọn mình` · `tụi mình` | → `chúng ta` | đúng hướng dẫn "we = chúng ta" |
| `… là mình chắc chắn…` | → `… là tôi chắc chắn…` | ca thật trong log |

### 2.2. P4.2 — glossary có 2 tầng

1. **Ấn định** (`Fixed renderings`) — mục khớp trong `glossary.yaml` hoặc `config.translation.glossary`
   được đưa vào prompt dưới dạng `美咲 -> Misaki`, yêu cầu dùng **chính xác**.
   Khớp theo **chuỗi con** nên `美咲` cũng khớp trong `美咲さん` (model chỉ cần ghép kính ngữ).
2. **Nhất quán** (`Proper nouns in the source`) — với tên **chưa** khai báo, `detect_names()` tự
   phát hiện (katakana ≥ 2 ký tự; kanji + kính ngữ さん/ちゃん/くん/様…) và yêu cầu model **không
   được bịa ra hai cách viết** cho cùng một tên — đúng kiểu lỗi `Meisaka` vs `Misa`.

Chỉ phát hiện tên khi khối có **≥ 2 câu** (một câu thì không có gì để mâu thuẫn).

### 2.3. P4.2b — suy romaji từ NGUỒN (thay cho "tự học từ bản dịch")

**Vì sao KHÔNG "tự học glossary từ cặp `orig -> trans`".** Ý tưởng đó có một lỗ hổng không vá được:

1. **Vòng lặp tự tham chiếu.** Cặp `orig -> trans` do **chính model đang cần ràng buộc** sinh ra.
   Lấy nó làm chuẩn là khoá cứng lựa chọn của model: `美咲` bị bịa thành `Meisaka` **một lần**
   ⇒ lỗi hệ thống ở **mọi khối sau**. Tệ hơn hẳn là không học.
2. **Không xác định được ánh xạ.** Từ một cặp câu, không biết token đích nào ứng với token nguồn
   nào. Ví dụ thật: 9B dịch `佐藤部長` → "giám đốc Sato" — học theo cặp sẽ khoá luôn
   `部長 = giám đốc`, một lỗi *nghĩa*, không chỉ lỗi tên.
3. **Sai số tương quan.** Hỏi lại chính model đó "đúng không?" không mang thêm thông tin. Bỏ
   phiếu 2B vs 9B cũng yếu — cùng họ, cùng fine-tune, và bất đồng của chúng **không tương quan
   với đúng/sai** (report 14: 2B đúng ở `部長`, 9B đúng ở `うまくいってる`).

**Cách đúng: cách viết một tên Nhật là hàm XÁC ĐỊNH của chuỗi nguồn.** `美咲` đọc là *misaki* —
điều đó đúng bất kể model dịch nó thành gì. `backend/translation/romaji.py` biến điều đó thành
hiện thực:

* `pykakasi` kiểu **`passport`** (thiết kế cho TÊN NGƯỜI: `佐藤 -> Sato`, không phải `Satou` như
  `hepburn`). Xử lý cả kanji lẫn kana.
* Fallback **thuần Python** khi thiếu `pykakasi`: bảng kana đầy đủ (âm ghép, sokuon `っ`,
  trường âm `ー`) vẫn romaji hoá được `ミサキ -> Misaki`, chỉ chịu thua kanji.
* Phát hiện tên nay nhận thêm **chức danh** (`佐藤部長`, `田中課長`) và có **stoplist** để không
  biến chức danh thành tên (`社長さん` KHÔNG được ấn định thành `社長 -> Shacho`).

**Ưu điểm lớn nhất: KHÔNG cần lưu trạng thái.** Vì là hàm thuần, cùng một tên luôn ra cùng một
cách viết ở mọi khối, mọi phiên, mọi lần chạy — nhất quán mà không cần "học", không ghi file,
không có gì để hỏng. Mục khai tay trong `glossary.yaml` **luôn đè** kết quả suy tự động.

---

## 3. Nghiệm thu trên model thật

Model: `Index-Translate-9B.Q4_K_M` và `Qwen3.5-4B-UD-Q4_K_XL`, greedy (`T=0, top_k=1`),
ngân sách production (`max_tokens = min(1536, max(256, n*80))`).
Prompt "old" lấy nguyên văn từ `git show HEAD:backend/translation/prompts.py`.

### 3.1. P4.1 — phát hiện trung thực: **prompt không phải thứ sửa được lỗi này**

| Ca thử (Index-9B) | `mình` thô — prompt CŨ | `mình` thô — prompt MỚI | `mình` **sau guard** |
|---|---|---|---|
| Khối 5 câu, không ngữ cảnh | 0 | **1** | **0** |
| Khối 5 câu, có `ctx-prev` | 0 | 0 / **1** (bản +glossary) | **0** |
| Stress 8 câu ngôi thứ nhất | 1 | 1 | **0** |

⇒ Siết câu chữ trong prompt **không** làm giảm tỉ lệ vi phạm (thậm chí nhích lên 1 ca).
⇒ Toàn bộ giá trị nằm ở **hậu kiểm tầng văn bản**: `sau guard = 0` ở **mọi** ca thử.

Câu thật trong log đã tái lập được: model trả `"tôi còn nói là mình chắc chắn không thể quay lại
làm nhân viên văn phòng nữa đâu."` → guard trả `"tôi còn nói là tôi chắc chắn…"`.

### 3.2. P4.2 — glossary sửa được lỗi bịa tên (đo trên Qwen3.5-4B)

4B là model **hay bịa tên** (đã đo ở report 12), nên là phép thử tốt nhất:

| Prompt | Câu 1 dịch ra | Tên đúng? |
|---|---|---|
| CŨ (không glossary) | `… với cô **Meisaka** đang diễn ra tốt đẹp.` | ❌ |
| MỚI (không glossary) | `… với cô **Misa** đang diễn ra tốt đẹp.` | ❌ |
| MỚI (không glossary, ctx) | `… **Misa-san** và cô ấy đang hợp tác tốt.` | ❌ |
| MỚI (không glossary, ctx) | `… **Ms. Misaki** và cô ấy đang làm việc khá tốt.` | ❌ |
| **MỚI + glossary** | `… **Misaki** và anh ấy đang đi được.` | ✅ |
| **MỚI + glossary, ctx** | `… **Misaki** và anh ấy đang đi được.` | ✅ |

**4/6 lần bịa tên khi không có glossary → 0/6 khi có glossary.** Cơ chế hoạt động đúng như thiết kế.

⚠️ Glossary chỉ sửa **tên**, không sửa **model**: câu 1 của 4B vẫn vô nghĩa
(`"Misaki và anh ấy đang đi được."`). Điều này củng cố kết luận ở report 12 — **giữ Index-Translate-9B**.

### 3.3. P4.2b — suy romaji từ nguồn (đo trên Qwen3.5-4B, `glossary.yaml` **RỖNG**)

Kịch bản khắc nghiệt nhất: không khai tay mục nào, để model hay bịa tên nhất tự đối mặt.

| Nhóm tên riêng | Suy tên **TẮT** | Suy tên **BẬT** |
|---|---|---|
| 田中さんはまだ来ていません。 | "**Thầy Trung** chưa đến." ❌ | "**Anh Tanaka** vẫn chưa đến." ✅ |
| 美咲ちゃんに伝えておいてくれる？ | "Bạn có thể nhắn cho cô **Mai** sao?" ❌ | "Cứ nhắn lại cho bạn **Misaki** nhé?" ✅ |
| 佐藤部長に報告しておきます。 | "báo cáo với ông **Sato**" ✅ | "báo cáo với anh **Sato**" ✅ |
| 桜井さんも一緒に行くって。 | "Cô **Sakurai** cũng đi cùng." ✅ | "Bạn **Sakurai** cũng đi cùng." ✅ |
| **Tỉ lệ tên đúng** | **2/4** | **4/4** |
| Khối 5 câu (câu 1) | "…với cô **Misa**…" ❌ | "Có vẻ như **Misaki** và anh ấy…" ✅ |

Đáng chú ý: khi **tắt** suy tự động, model không chỉ bịa cách viết mà còn **Hán-Việt hoá tên
Nhật** (`田中 -> "Trung"`, `美咲 -> "Mai"`) — một kiểu lỗi mới, và cũng bị suy romaji chặn đứng.

Prompt gửi cho model khi bật (glossary rỗng hoàn toàn):

```
Fixed renderings — use these EXACTLY and identically in every sentence:
田中 -> Tanaka; 美咲 -> Misaki; 佐藤 -> Sato; 桜井 -> Sakurai.
```

### 3.4. HỒI QUY THẬT (log 2026-10-08): katakana từ ngoại lai ≠ tên riêng

Bản P4.2b đầu tiên coi **mọi chạy katakana** là tên riêng và ấn định romaji cho nó. Sai nghiêm
trọng, vì katakana trong tiếng Nhật **chủ yếu là từ ngoại lai**:

| Nguồn | Bản lỗi ấn định | Bản dịch ĐÚNG bị thay mất |
|---|---|---|
| `チーム` | `Chiimu` | đội / nhóm |
| `ホテル` | `Hoteru` | khách sạn |
| `センス` | `Sensu` | gu / khiếu |
| `サッカー` | `Sakkaa` | bóng đá |
| `バーベキュー` | `Baabekyuu` | tiệc nướng |
| `グリル` | `Guriru` | vỉ nướng |
| `カスタム` | `Kasutamu` | tuỳ chỉnh |
| `イビキ` | `Ibiki` | tiếng ngáy |
| `五年先輩` | `五年 -> Gonen` | "người đi trước 5 năm" |

Đây là loại lỗi **phá hoại**: một dương tính giả *đè lên* bản dịch đúng, còn âm tính chỉ quay về
hành vi cũ. Nên nguyên tắc là **thà bỏ sót**.

**Sửa:** bỏ hẳn tín hiệu "katakana trần". `detect_names` giờ chỉ có **một** tín hiệu — kính
ngữ/chức danh đứng ngay sau — và ứng viên có thể là kanji *hoặc* katakana:

* `美咲さん`, `田中くん`, `佐藤部長` → `美咲`, `田中`, `佐藤` ✅
* `ミサキちゃん`, `カミちゃん` → `ミサキ`, `カミ` ✅
* `チーム`, `ホテル`, `イビキ`… (katakana trần) → **không nhận** ✅
* `五年先輩` → `五年` bị chặn bằng bộ lọc **số + trợ từ đếm** ✅ (nhưng `一郎` vẫn giữ, vì `郎`
  không phải trợ từ đếm)

Kiểm chứng trên **nguyên văn 9 khối phụ đề** trong log người dùng (`debug/glossary_fp_check.py`):

```
00:29 カスタム                      []                 (không tên)
00:30 ホテル/トラベルダウ/羽田          ['羽田']           羽田 -> Haneda
00:30 花田/カミ/五年                 ['花田', 'カミ']     花田 -> Hanada; カミ -> Kami
00:31 チーム                       []                 (không tên)
00:31 川北/羽田                     ['羽田']           羽田 -> Haneda
00:32 サッカー/川北                  []                 (không tên)
00:33 イビキ                       []                 (không tên)
00:33 美咲 (đối chứng ĐÚNG)         ['美咲']           Misaki ✅
00:33 バーベキュー/グリル              []                 (không tên)
KẾT LUẬN: KHÔNG còn dương tính giả
```

**Đánh đổi đã chấp nhận:** tên ngoại lai **không có kính ngữ** (ví dụ `ジョン` đứng trần) sẽ
không được suy tự động. Model vẫn dịch được như trước, và tên đó vẫn khai được trong
`glossary.yaml`.

### 3.5. Chi phí

| | Prompt CŨ | Prompt MỚI | MỚI + glossary |
|---|---|---|---|
| Ký tự | 592 | 1140 | 1228 |
| Token | 168 | 336 | 361 |
| Độ trễ (Index-9B) | 1682 ms | 1627 ms | 1624 ms |

Khối ràng buộc tốn ~170 token prompt nhưng **không đo được độ trễ tăng thêm** ở cỡ prompt này.
Suy romaji gần như miễn phí: `pykakasi` chạy trên vài chuỗi ngắn, không đo được vào độ trễ.
JSON batch hợp lệ **100%** ở mọi cấu hình (12/12 lần chạy).

---

## 4. Cách dùng

**Bạn không cần khai tên riêng nữa** — `glossary.yaml` mặc định để trống, tên được suy tự động:

```
美咲 -> Misaki     田中 -> Tanaka     佐藤 -> Sato     桜井 -> Sakurai
ミサキ -> Misaki   長谷川 -> Hasegawa  林 -> Hayashi
```

Chỉ khai tay khi muốn **khác đi** (mục khai tay luôn đè kết quả suy tự động):

```yaml
# backend/glossary.yaml — sửa xong KHÔNG cần restart (đọc lại theo mtime)
terms:
  メアリー: Mary        # kana cho "mearii", nhưng bạn muốn "Mary"
  田中: Tanaka-san      # giữ kính ngữ Nhật thay vì "Anh Tanaka"
  株式会社: công ty cổ phần
```

- Tắt suy romaji: `config.translation.glossary_derive_names = false`.
- Tắt ép đại từ: `config.translation.enforce_pronoun_policy = false`.
- Tắt hẳn glossary file: `config.translation.glossary_file = ""`.
- Ghi đè theo phiên (ưu tiên cao hơn file): `config.translation.glossary = {"美咲": "Mỹ Sako"}`.
- File hỏng/thiếu, thiếu `pykakasi` **không** làm chết đường dịch — chỉ bỏ qua/mất phần đó.

---

## 5. Kiểm thử

```
backend/tests/test_65_pronoun_and_glossary.py   36 passed
backend/tests/test_66_name_romaji.py            47 passed
toàn bộ tầng A (not slow and not full)         674 passed
```

Bao phủ: 17 ca `enforce_no_minh` (cả ca phải sửa và ca phải giữ nguyên) · nạp file + ghi đè runtime
+ reload theo mtime + file hỏng · `detect_names` (kể cả hồi quy cắt `今日美咲さん` → `美咲`) ·
prompt có/không glossary · engine batch + single + streaming · tắt cờ · **strategy cũ không có
tham số `terms`** (hồi quy thật từ `test_32`) · romaji kanji/kana · `passport` ≠ `hepburn` ·
**fallback kana thuần Python khi thiếu `pykakasi`** (monkeypatch `_KAKASI = None`) · thứ tự ưu
tiên khai-tay-đè-suy-tự-động · stoplist chức danh.

Tái lập nghiệm thu:

```powershell
.\.venv\Scripts\python.exe debug\p4_verify.py --model index9b
.\.venv\Scripts\python.exe debug\romaji_verify.py --model 4b    # P4.2b, glossary RỖNG
```

---

## 6. Việc chưa làm (đề xuất tiếp)

1. **Chưa có UI nhập glossary.** Hiện chỉ sửa file YAML hoặc set `config.translation.glossary`.
   Đường `/api/config` (`main.py:873`) chưa trả về glossary ⇒ popup chưa hiển thị/sửa được.
2. ~~**Chưa tự học glossary từ ngữ cảnh.**~~ ⇒ **ĐÃ THAY BẰNG P4.2b (§2.3)**: học từ cặp
   `orig -> trans` là khoá cứng sai số của chính model; nay cách viết được **suy từ chuỗi
   nguồn** (`romaji.py`), xác định và không cần trạng thái. Cặp `orig -> trans` ở
   `lookahead_handler.py:1479-1491` **không** dùng làm nguồn sự thật cho glossary.
3. **Cách đọc nanori còn mơ hồ.** `pykakasi` chọn một cách đọc phổ biến (`一` → `Ichi`); tên
   ngoại lai bị phiên âm theo kana (`メアリー` → `Mearii`). Cả hai đều ghi đè được trong
   `glossary.yaml`, nhưng một bảng nanori riêng sẽ tốt hơn.
4. **Tên ngoại lai không kính ngữ không được suy.** Đánh đổi chủ ý để tránh dương tính giả
   katakana (§3.4). Có thể cải thiện bằng một từ điển từ ngoại lai (loanword lexicon) để phân
   biệt `ホテル` với `マイケル` — nhưng cần thêm dữ liệu.
5. **Chọn lọc ngữ cảnh.** Report 12 §3.4 cho thấy `ctx-prev` có thể làm **tệ hơn**; nên A/B lại
   0/1/2 lượt.
