# XÁC NHẬN BENCHMARK OMNIVOICE — GGUF vs PyTorch Native

> **Nguồn thẩm định:** `scratch/benchmark_results/COMPARISON_REPORT.md` + `scratch/test_omnivoice_gguf.py`
> **Phương pháp:** đọc mã script, đọc `--help` của binary thật, **đo lại trên cùng máy**,
> và đưa hai bên về **cùng một thước đo**.
> **Ngày:** 2026-09-17

---

## 0. KẾT LUẬN NGẮN

**Đồng ý một phần.** Kết luận cuối cùng (*giữ PyTorch, chưa chuyển sang GGUF*) là **hợp lý** — nhưng
**lý do chính mà báo cáo đưa ra thì không đứng vững**, và **con số headline 4,3× là sai phương pháp**.

| Nội dung | Đánh giá |
| :--- | :---: |
| Kết luận "chưa nên chuyển sang GGUF" | ✅ **Đồng ý** (vì lý do khác — xem §3) |
| "PyTorch nhanh hơn **4,3×**" | ❌ **SAI PHƯƠNG PHÁP** — 69 % chênh lệch chỉ là **số bước**, không phải runtime (§1) |
| Chất lượng tiếng Việt: PyTorch > GGUF | ⚠️ **Đúng nhưng lý do bị đặt sai chỗ** — đó là khác biệt **MODEL**, không phải **RUNTIME** (§2) |
| GGUF tiết kiệm ~1,2–1,6 GB VRAM | ✅ **Đúng về độ lớn** (đo được 1,47 GB) — nhưng **số tuyệt đối trong báo cáo sai** và **trộn hai thước đo** (§2) |
| "File nhẹ hơn 4,9×" | ✅ Đúng (388,6 + 240,8 = 629 MB vs 3,25 GB) |
| Vulkan crash `GET_ROWS` | ⚠️ **Không kiểm chứng lại** (không đủ thời gian dựng lại) — không phản bác |
| `.rvq` 3.201 bytes | ✅ Đúng — `speaker_01_0039.rvq` = 3,1 KB trên đĩa |
| CUDA chạy được sau khi trỏ `cublas64_13.dll` | ✅ Đúng — chạy lại thành công, `rc=0` |
| Báo cáo **tự mâu thuẫn** về độ trễ GGUF | ❌ Bảng nói **1877 ms**, phần kết luận nói **"~600–800 ms"** (§4) |

---

## 1. SAI SÓT QUYẾT ĐỊNH — hai bên chạy **KHÁC SỐ BƯỚC**

`omnivoice-tts.exe --help` (chạy trực tiếp trên binary của dự án):

```
--steps <int>           MaskGIT decode steps (default: 32, fewer is faster)
```

`scratch/test_omnivoice_gguf.py` **không truyền `--steps`** (grep toàn file: không có chuỗi nào)
⇒ GGUF chạy **32 bước**.

PyTorch thì dùng cấu hình của dự án:

```python
# backend/config.py:300
num_inference_steps: int = 8
# backend/tts/engine.py:306
num_steps = max(4, int(getattr(config.tts, "num_inference_steps", 8)))
```

⇒ PyTorch chạy **8 bước**.

**Vậy phép so là 8 bước vs 32 bước — chênh 4× về số vòng forward.** Không phải hai runtime ở cùng
điều kiện.

### Đo lại để chứng minh

`external/build-tmp/verify_omnivoice_steps.py` — chạy GGUF ở **cả hai** mức bước:

| Mẫu | Audio | **8 bước** | **32 bước** | Tỉ lệ | RTF@8 | RTF@32 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_en_vi` | 3,37 s | **486,9 ms** | 1495,6 ms | 3,1× | 0,144 | 0,444 |
| `medium_zh_vi` | 6,15 s | **605,2 ms** | 1889,8 ms | 3,1× | 0,098 | 0,307 |
| `long_ja_vi` | 9,06 s | **690,4 ms** | 2347,6 ms | 3,4× | 0,076 | 0,259 |
| `conversational` | 3,77 s | **517,7 ms** | 1609,3 ms | 3,1× | 0,137 | 0,427 |
| **TRUNG BÌNH** | | **575,0 ms** | **1835,6 ms** | **3,2×** | | |

**Phép đo của tôi tái lập đúng số của báo cáo ở 32 bước** (1835,6 vs 1877,2 ms — lệch 2 %), nên
thiết lập đo là khớp. Từ đó:

| | Thời gian | Kết luận |
| :--- | ---: | :--- |
| PyTorch (8 bước) | 429,1 ms | — |
| GGUF (32 bước) — *báo cáo dùng số này* | 1877,2 ms | "PyTorch nhanh **4,3×**" |
| **GGUF (8 bước) — cùng điều kiện** | **575,0 ms** | "PyTorch nhanh **1,34×**" |

⇒ **69 % chênh lệch đến từ số bước, không phải từ runtime.** Khoảng cách runtime thật chỉ còn
**~1,3×** — một đánh đổi rất khác so với "chậm gấp 4,3 lần".

### Cách sửa

```python
cmd = [ ..., "--steps", "8", ... ]        # ghim bằng đúng num_inference_steps của PyTorch
```

Nhưng **so ở cùng số bước vẫn chưa đủ** (xem §2): MaskGIT ở 8 bước và 32 bước cho **chất lượng
khác nhau**. Phép so đúng phải là **so ở cùng chất lượng**: tìm số bước của GGUF cho ra chất lượng
tương đương PyTorch@8, rồi mới so tốc độ. Việc đó **chưa được làm**.

---

## 2. HAI VẤN ĐỀ PHƯƠNG PHÁP KHÁC

### 2.1. So **hai MODEL khác nhau**, không phải hai **runtime** trên cùng model

| | Model | Nguồn |
| :--- | :--- | :--- |
| PyTorch | `splendor1811/omnivoice-vietnamese` | Fine-tune **1.000 h tiếng Việt**, FP16 |
| GGUF | `Serveurperso/OmniVoice-GGUF` | Convert từ **base** `k2-fsa/OmniVoice`, Q4_K_M |

Nên **mọi khác biệt** (tốc độ, VRAM, chất lượng giọng) đều bị **trộn** giữa ba biến: *runtime*,
*lượng tử hoá*, và *fine-tune*.

Đặc biệt phản trực giác: model Q4_K_M **nhỏ hơn 5×** lẽ ra phải **nhanh hơn** mỗi lượt forward
(ít băng thông bộ nhớ hơn), chứ không chậm hơn. Việc nó chậm hơn ở 32 bước là do **số bước**, còn
phần chênh 1,34× còn lại mới là do runtime (omnivoice.cpp chưa dùng CUDA Graph cho vòng MaskGIT).

Báo cáo **có** nhận ra điều này ở mục "Giải Pháp Nâng Cao" (*"convert trực tiếp `splendor1811` sang
GGUF"*) — nhưng bảng headline vẫn trình bày như thể runtime là thứ chậm hơn 4,3×.

### 2.2. Trộn hai thước đo VRAM

| Nguồn | PyTorch | GGUF | Vấn đề |
| :--- | ---: | ---: | :--- |
| `COMPARISON_REPORT.md` | 2001 MB (`torch.cuda.memory_allocated()`) | 395–550 MB | **Hai thước đo khác nhau.** `memory_allocated()` **không** tính CUDA context, cublas/cudnn workspace, phần reserved |
| **Đo lại, cùng thước (device-level delta)** | **2430 MB** | **958 MB** | `external/build-tmp/verify_pytorch_vram.py` + `verify_omnivoice_vram.py` |

Số tuyệt đối trong báo cáo **sai cả hai phía**: PyTorch bị đếm thiếu ~430 MB, GGUF bị đếm thiếu
~500 MB. May là **hiệu số** vẫn xấp xỉ đúng: đo được **2430 − 958 = 1.472 MB (~1,47 GB)** so với
con số "1,2–1,6 GB" của báo cáo.

### 2.3. Trường `vram_delta_mb` trong script **luôn bằng 0** (bug)

`test_omnivoice_gguf.py`:

```python
vram_before, _ = get_gpu_vram_mb()        # dòng 132
proc = subprocess.run(cmd, ...)           # dòng 147-153  ← tiến trình con chạy rồi THOÁT
vram_during, _ = get_gpu_vram_mb()        # dòng 155      ← đọc SAU khi con đã thoát
```

⇒ `vram_delta_mb = max(0, vram_during - vram_before)` **không đo được gì**. Nghĩa là con số VRAM
trong báo cáo **không đến từ script này** — nguồn không rõ. Đã sửa bằng cách poll `nvidia-smi`
**trong lúc chạy** (`verify_omnivoice_vram.py`).

---

## 3. VÌ SAO TÔI VẪN ĐỒNG Ý "CHƯA CHUYỂN"

Kết luận đúng, nhưng lý do đúng phải là:

1. **Chất lượng là yếu tố quyết định, và đó là câu hỏi về MODEL chứ không phải runtime.** Nếu
   `splendor1811` (fine-tune 1.000 h tiếng Việt) thật sự đọc tiếng Việt tốt hơn base model, thì
   giữ nó là đúng — **bất kể** runtime nhanh hay chậm. Tôi **không thể** xác nhận phần nghe bằng
   máy; đây là đánh giá chủ quan của người nghe và nên được kiểm bằng cách nghe 2 file trong
   `scratch/benchmark_results/`.
2. **1,34× là mức chênh nhỏ**, và nằm trong cùng bậc với rủi ro tích hợp. Đổi runtime để lấy 1,47 GB
   VRAM trên card 16 GB (đang dùng ~9,5 GB đỉnh) là **không cần thiết** — nhưng đây là kết luận
   **kinh tế**, không phải "vì GGUF chậm 4,3×".
3. **Chưa có phép đo ở cùng chất lượng.** Cả 8-vs-8 lẫn 8-vs-32 đều không phải so sánh công bằng.
   Kết luận "chưa chuyển" đúng, nhưng nó **chưa được chứng minh** bằng số — nó đang dựa trên một
   phép so sai.

⇒ **Giữ PyTorch là hợp lý**, và tôi cũng khuyến nghị **giữ**. Nhưng ghi lại đúng lý do, để sau này
nếu cần bản nhẹ cho máy yếu thì ta biết mình **chưa** loại GGUF vì tốc độ — ta loại vì **chưa có
phép so chất lượng ở cùng số bước**.

---

## 4. BÁO CÁO TỰ MÂU THUẪN

`COMPARISON_REPORT.md` có **hai con số loại trừ nhau** cho cùng đại lượng:

- **Bảng headline (dòng 15-18):** GGUF **1527 / 1926 / 2378 / 1678 ms** (trung bình 1877 ms)
- **Bảng kết luận (dòng 70/362):** *"Độ trễ MaskGIT qua CLI/Subprocess cao hơn chút do overhead
  (**~600–800 ms** vs ~400 ms)"*

Hai số này không thể cùng đúng. Đáng chú ý: **số ở bảng kết luận (600–800 ms) khớp với phép đo
8 bước của tôi (575 ms)**, còn số ở bảng headline khớp với 32 bước (1836 ms). Tức là **phần văn
kết luận của báo cáo đã đúng, còn bảng headline thì sai** — dấu hiệu bảng được sinh từ một lần chạy
khác (hoặc cấu hình khác) so với phần phân tích.

---

## 5. VIỆC NÊN LÀM TIẾP (nếu muốn kết luận chắc)

| # | Việc | Vì sao |
| :--- | :--- | :--- |
| 1 | Thêm `--steps` vào script, chạy **cả 8 và 32** cho GGUF | Để phép so không còn trộn số bước |
| 2 | Chạy PyTorch ở `num_inference_steps = 32` để có ô đối chứng **8-vs-8** và **32-vs-32** | Tách bạch "khác runtime" khỏi "khác số bước" |
| 3 | **Convert `splendor1811` sang GGUF Q8_0** rồi đo | Đây mới là phép so **cùng model** — báo cáo đã đề xuất nhưng chưa làm |
| 4 | Sửa `vram_delta_mb`: poll `nvidia-smi` trong lúc chạy | Trường hiện tại luôn ~0 |
| 5 | Đo VRAM hai bên bằng **cùng** cách (device-level), ghi rõ thước đo | Tránh so `memory_allocated()` với `nvidia-smi` |
| 6 | Nghe và chấm chất lượng 2 file `*_conversational.wav` (điểm mù của tôi) | Chất lượng là yếu tố quyết định thật |

**Script kiểm chứng sinh ra tài liệu này:**
`external/build-tmp/verify_omnivoice_steps.py` (§1), `verify_omnivoice_vram.py` và
`verify_pytorch_vram.py` (§2.2).
