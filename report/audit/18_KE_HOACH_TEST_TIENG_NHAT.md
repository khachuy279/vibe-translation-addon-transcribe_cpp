# 18 — Đánh giá khả thi kế hoạch test tối ưu tiếng Nhật (FLEURS ja_jp) & thiết kế vòng test điều chỉnh

**Ngày:** 2026-09-20 (cập nhật lần 2 — bổ sung kết quả JA30 ở §3.6)
**Phạm vi:** đường ASR tiếng Nhật của backend (`models.yaml` → 7 model), pipeline `Feed audio → VAD → ASR`, đối chiếu với số liệu công bố trong `external/transcribe.cpp/docs/models/*.md` và `external/transcribe.cpp/catalog/*.json`.
**Trạng thái:** đã đo thực (không phải ước lượng). Mọi con số dưới đây đều tái lập được bằng script trong `scratch/`.

---

## 0. Kết luận nhanh (TL;DR)

| Câu hỏi của bạn | Trả lời |
| --- | --- |
| Kế hoạch có khả thi không? | **Có, và rẻ hơn bạn nghĩ** — nhưng phần "vòng 2" đang đặt mục tiêu sai chỗ. |
| Vòng 1 với 10 audio có chọn được model không? | **Không đủ tin cậy.** Cỡ mẫu 10 cho CI95 ±4,0 điểm %, trong khi 3 model dẫn đầu chỉ cách nhau 0,5 điểm. Và chỉ tốn ~3 phút/model để chạy **cả 650 clip** ⇒ không có lý do gì dùng 10. |
| Vòng 2 "tiệm cận số công bố" có khả thi không? | **Đã đạt rồi — ngay hôm nay, chưa chỉnh gì cả.** Đo offline toàn bộ 650 clip: **CER 5,36%** so với công bố **5,29%** (CI95 4,81–5,80). Và **thông số VAD/ASR không thể cải thiện con số này**, vì đường offline không dùng VAD. |
| Vậy còn việc gì để tối ưu? | **Khoảng cách giữa pipeline (VAD→ASR) và trần offline** — đo được **+9,29 điểm %** trên JA30 (sàn nhiễu 0,57). Đây mới là mục tiêu đúng của vòng 2. |
| Nhóm knob nào thực sự hiệu quả? | **`max_duration_sec` + `boundary_overlap_ms`** — giảm **2,36 điểm %** (−15,7%), gap +9,29 → **+6,93**, parity 8,7 → **14/30** clip, và **4,1× sàn nhiễu**. `enable_tier234` **không** thêm gì. Đã kiểm chứng ở §3.6. |
| Có hồi quy trên audio thật không? | **Không.** Tập kiểm chứng chéo: **không file nào xấu đi**, còn cải thiện −13…−38 điểm trên audio nhiễu và nói nhanh. |
| Rủi ro lớn nhất | **Sàn nhiễu**: base lặp 3 lần đã chênh **0,57 điểm** (đo mới trên JA30) và **3,34 điểm** (đã ghi trong `report/audit/14_wer_noise_floor.json`). Không đo sàn nhiễu thì mọi kết luận A/B đều vô giá trị. |
| Đánh đổi mới phát hiện | **Độ trễ commit**: cấu hình mặc định **vốn đã vượt** ngân sách `poll_interval_ms=300ms` (p95 = 306–319 ms, 8/90 lượt chạy file); `long_seg` đẩy lên **331–359 ms**. Cần chọn `max_duration_sec` trung gian (12s) — xem §3.6.D. |

---

## 1. Những gì đã kiểm chứng bằng đo thực

### 1.1. Dữ liệu — dùng được ngay, khớp 100%

| Hạng mục | Kết quả đo |
| --- | --- |
| File wav trong `wav_test/google_fleurs/ja_jp/test` | **650** (khớp đúng 650 dòng của `test.tsv`, không thiếu/thừa file nào) |
| Độ dài clip | min 6,36s / trung vị 12,54s / trung bình 13,09s / max 28,20s |
| Không clip nào > 30s | ⇒ luôn nằm trong context window của model, chạy 1 lượt offline được cho **mọi** clip |
| Số ký tự tham chiếu | trung bình **50,3 ký tự/clip** |

**Cột tham chiếu đúng là cột index 3** (bản "normalized", dấu câu thay bằng khoảng trắng) — không phải cột 2 (bản raw có `、。`). Đây là cột mà `external/transcribe.cpp/scripts/wer/ingest.py` nạp vào trường `transcription`.

*Bằng chứng số học (công thức tường minh để tái lập):*

```
catalog qwen3-asr-1.7b / ja:
  errors      = sub + del + ins = 1162 + 321 + 245 = 1728
  err_pct     = 5.29%  ->  0.0529
  => ref_chars = 1728 / 0.0529 = 32 665 ky tu      (suy ra TU CATALOG)
  dia phuong   = 32 648 ky tu                       (dem tren cot index 3, cung bo cham)
  lech         = 17 ky tu = 0.05%
```

Nếu chọn sai cột (cột 2, bản raw có `、。`), tổng ký tự sẽ lệch hàng nghìn. Phép kiểm tra này là **chốt an toàn** cho toàn bộ kế hoạch ⇒ đưa vào harness dưới dạng `assert abs(ref_chars - 32665) / 32665 < 0.005`.

### 1.2. Số liệu công bố — không cần đọc file `.md`, catalog có đủ provenance

`external/transcribe.cpp/catalog/<model>.json` chứa **toàn bộ** thông tin cần để so sánh, gồm cả số lỗi tuyệt đối:

| Model trong `models.yaml` | CER ja (catalog) | CI95 | S / D / I | n_utts | batch |
| --- | ---: | --- | --- | ---: | ---: |
| `whisper-large-v3-turbo` | **4,82%** | 4,40–5,30 | 1109 / 295 / 171 | 650 | 8 |
| `cohere-transcribe` | **5,13%** | 4,48–5,84 | 871 / 670 / 133 | 650 | 8 |
| `qwen3-asr-1.7b` | **5,29%** | 4,81–5,80 | 1162 / 321 / 245 | 650 | 8 |
| `voxtral-mini-4b-realtime` | **5,44%** | 4,94–5,95 | 1227 / 365 / 184 | 650 | 8 |
| `sensevoice-small` | 7,63% | 7,10–8,22 | 1665 / 558 / 268 | 650 | 8 |
| `qwen3-asr-0.6b` | 8,61% | 7,98–9,28 | 1929 / 522 / 359 | 650 | 8 |
| `nemotron-3.5-streaming` | 13,52% | 12,78–14,27 | 3193 / 816 / 404 | 650 | 8 |

Điểm quan trọng: **`voxtral-realtime.md` không có bảng FLEURS** (bạn sẽ không tra được số của nó trong `docs/models`), nhưng catalog thì có. Nên **tra catalog, không tra `.md`**.

**Đọc thêm cột S / D / I — ba model đầu có "tính cách lỗi" rất khác nhau:**

| Model | S | D | I | Nhận xét |
| --- | ---: | ---: | ---: | --- |
| `qwen3-asr-1.7b` | 1162 | 321 | 245 | **Cân bằng**, hơi nghiêng về thay-thế (substitution). |
| `whisper-large-v3-turbo` | 1109 | 295 | 171 | Cân bằng, **ít insertion nhất** trong nhóm dẫn đầu. |
| `cohere-transcribe` | 871 | **670** | 133 | **Lệch hẳn về XOÁ (deletion)** — 670 xoá so với 871 thay-thế, tức gần một nửa lỗi là *mất chữ*. |
| `voxtral-mini-4b-realtime` | 1227 | 365 | 184 | Nhiều thay-thế nhất trong nhóm. |

Vì sao quan trọng: **hai loại lỗi này không tương đương với tầng dịch phía sau.** Xoá ký tự làm **mất nội dung** (bản dịch thiếu ý), còn thay-thế thường vẫn giữ được cấu trúc câu. Với cùng một CER ~5%, `cohere-transcribe` sẽ cho bản dịch *thiếu thông tin* nhiều hơn `whisper-large-v3-turbo`. Ngoài ra `cohere` cũng ít insertion nhất — nghĩa là nó không bịa thêm, một đặc tính đáng giá khi phải chạy trên audio nhiễu/nhạc nền mà FLEURS không kiểm tra được.

⇒ **Đừng chọn model chỉ bằng một con số CER.** Ghi cả S/D/I vào bảng so sánh, và dùng tập kiểm chứng chéo (§3.5) để phân xử khi CER xấp xỉ nhau.

### 1.3. Tài nguyên & tốc độ — kế hoạch rẻ hơn nhiều so với dự kiến

| Số đo | Giá trị (RTX 5060 Ti 16GB, backend CUDA qua `bin/`) |
| --- | --- |
| Nạp + prewarm `qwen3-asr-1.7b` Q8_0 | **2,0 giây** |
| RTF suy luận offline (1 lượt/clip) | **0,0212** (≈ 47× thời gian thực) |
| Chạy cả 650 clip (8.509s audio) | **~180 giây** ⇒ ~3 phút/model, **~25 phút cho cả 7 model** |
| GGUF cục bộ | **đủ cả 7 model** trong `backend/models/` ⇒ không cần tải (đặt `asr.auto_download=False` để chắc chắn không chạm mạng) |
| VAD FireRed | có sẵn `backend/models/firered_stream/Stream-VAD/` |
| VRAM đỉnh | model lớn nhất là voxtral Q5_K_M 3,13GB — thừa sức với 16GB |

### 1.4. Bộ chấm — đã cài đúng bộ của tác giả

`transcribe.cpp/scripts/wer/score.py` chấm bằng `whisper_normalizer.BasicTextNormalizer` + `jiwer`, và với CER thì **bỏ toàn bộ khoảng trắng** ở cả hai phía. Đã cài `whisper-normalizer` vào môi trường và xác nhận:

- Bộ chấm của repo (`backend/tests/test_09_wer_ab.py::score_pair`) cho **kết quả trùng khít** bộ của `score.py` trên 10 clip thử (4,44/4,44 · 14,81/14,81 · 7,41/7,41 · 0,00/0,00 …). ⇒ có thể tái dùng hạ tầng sẵn có, nhưng nên chuyển sang bộ của `score.py` để "khớp định nghĩa" khi so với catalog.
- **Cột `ref_chars` của hai bộ chấm lệch 0,05%** (32.648 vs 32.665) — không đáng kể.

### 1.5. KẾT QUẢ QUYẾT ĐỊNH: tái lập được số công bố, ngay bây giờ

Chạy **offline 1 lượt cho cả 650 clip**, batch 1, CUDA, đúng đường suy luận của backend (`TranscribeEngine._run_inference_sync`):

| Cấu hình | CER corpus | S / D / I | ref_chars | macro CER | RTF |
| --- | ---: | --- | ---: | ---: | ---: |
| **Pipeline hiện tại** (`normalize_speech=True`) | **5,36%** | 1188 / 316 / 247 | 32.648 | 5,35% | 0,0212 |
| `normalize_speech=False` | **5,34%** | 1183 / 313 / 249 | 32.648 | 5,37% | 0,0211 |
| *Catalog `qwen3-asr-1.7b`* | *5,29%* | *1162 / 321 / 245* | *32.665* | — | — |

**Đọc bảng này:**

1. **Lệch catalog chỉ 0,07 điểm %** (5,36 vs 5,29) và lệch hoàn toàn nằm trong CI95 của catalog (4,81–5,80). Phân rã lỗi gần như trùng khít (1188/316/247 vs 1162/321/245). ⇒ **Môi trường, lượng tử hoá Q8_0, đường suy luận CUDA, bộ chấm, và lựa chọn cột tham chiếu của ta đều ĐÚNG.** Không có "khoảng cách môi trường" nào để đi tìm.
2. **`normalize_speech` là no-op về độ chính xác** với FLEURS (Δ 0,02 điểm, dưới sàn nhiễu; 1761 vs 1745 lỗi). Audio FLEURS vốn đã chuẩn hoá âm lượng sẵn. ⇒ **Bỏ giả thuyết này khỏi vòng 2**, đừng tốn ngân sách.
3. Nghịch lý cần nhớ: 10 clip đầu tiên tôi lấy mẫu rải đều cho **CER macro 11,03%** (corpus 9,43%), trong khi sự thật của cả 650 là **5,35%**. Một mẫu 10 clip sai lệch **hơn 5 điểm %**. Đây là minh chứng cụ thể cho vấn đề ở §2.1.

---

## 2. Đánh giá kế hoạch của bạn — chỗ nào đúng, chỗ nào sẽ sai

### 2.1. Vòng 1: "10 audio → chọn model tốt nhất" — **sẽ sai**

Tôi bootstrap trên chính tập 650 đã đo để tính CI95 theo cỡ mẫu:

| n clip | ref_chars | CI95 của CER corpus | bề rộng |
| ---: | ---: | --- | ---: |
| **10** | ~500 | **[2,07% – 10,11%]** | **±4,02 điểm** |
| 20 | ~1.000 | 2,86% – 8,43% | ±2,79 |
| 30 | ~1.500 | 3,26% – 7,72% | ±2,23 |
| 50 | ~2.500 | 3,71% – 7,21% | ±1,75 |
| **100** | ~5.000 | 4,14% – 6,69% | **±1,28** |
| 200 | ~10.000 | 4,51% – 6,28% | ±0,89 |
| **650** | 32.648 | **4,90% – 5,90%** | **±0,50** |

Với n=10, ba model dẫn đầu (4,82 / 5,13 / 5,29) **nằm trọn trong một CI**. Bạn sẽ chọn "model tốt nhất" dựa trên nhiễu.

Và mô phỏng **so sánh cặp trên cùng clip** (thiết kế đúng cho A/B — mạnh hơn nhiều so với so hai CI rời) cho công suất phát hiện mức giảm lỗi tương đối:

| n | giảm 5% | giảm 10% | giảm 20% | giảm 30% |
| ---: | ---: | ---: | ---: | ---: |
| **10** | 6% | 22% | 61% | **77%** |
| 20 | 29% | 72% | 95% | 99% |
| **30** | 53% | **91%** | 100% | 100% |
| 50 | 81% | 99% | 100% | 100% |
| **100** | **100%** | 100% | 100% | 100% |

⇒ n=10 **chỉ** phát hiện được thay đổi từ ~30% tương đối trở lên. n=30 đủ cho hiệu ứng ≥20%. n=100 đủ cho ≥5%.

**Khuyến nghị:** vì RTF = 0,0212, hãy **chạy cả 650 clip cho vòng 1 ở đường offline** (~3 phút/model). Không subsample. Vừa khớp 100% protocol công bố, vừa có CI ±0,5 điểm.

### 2.2. Vòng 1: "CER tốt nhất" là tiêu chí chưa đủ

Ba model dẫn đầu đều là `offline_llm`; `nemotron`/`voxtral` là `streaming` (đi qua `session.stream`) nên **đường suy luận khác nhau** — nếu không nêu rõ, bảng so sánh sẽ không công bằng. Ngoài ra cần:

- **CER ở độ dài đoạn sản xuất (1–8s sau VAD)**, không phải cả clip 13s. Thứ hạng có thể đảo — đặc biệt `whisper` auto-detect ngôn ngữ trên đoạn ngắn rất dễ sai. ⇒ Tầng 1b **phải phân tầng kết quả theo bucket độ dài** (6–10s / 10–15s / >15s) ngay trong báo cáo, không gộp thành một số.
- **RTF phải là RTF của PIPELINE, không phải 0,0212.** Pipeline còn chạy preview mỗi `poll_interval_ms=300ms` trên cửa sổ 8s, cộng VAD mỗi 25ms, cộng chuẩn hoá. RTF sản phẩm thực tế ước tính **~0,5–0,6** (preview đã chiếm ~0,17s/300ms ≈ 57% GPU). Con số 0,0212 chỉ là "trần tốc độ model".
- **Độ trễ commit, không chỉ CER.** Với phụ đề realtime, người dùng cảm nhận `p95(commit_ms)` và **số câu chốt/phút** rõ hơn là chênh 0,5 điểm CER. Hai chỉ số bắt buộc ghi kèm: `commits/clip` (càng gần số câu thật càng tốt — 1,0 nghĩa là không cắt vụn) và `p95 commit_ms` (phải **< `poll_interval_ms` = 300ms**, nếu không pipeline sẽ bỏ nhịp preview).
  *Lưu ý kỹ thuật:* hai field này **không có trong gói WebSocket** — xem §2.5.
- **VRAM** (2,1GB cho qwen 1.7B vs 3,2GB cho voxtral vs 0,9GB cho whisper).
- **Hành vi xấu**: `empty_hyp`, lặp vòng, hallucination trên đoạn im lặng/nhạc — FLEURS sạch nên **không** đo được cái này, phải thêm tập kiểm chứng chéo (§3.5).
- **Hồ sơ lỗi S/D/I**, không chỉ tổng CER — xem ghi chú ở §1.2 (cohere xoá nhiều, whisper thay-thế nhiều; hai loại này ảnh hưởng khác nhau tới tầng dịch).

Tôi đã đo luôn một con số cho điểm này: chạy pipeline ở `speed=1.0`, `wall/audio = 1,11` (đã gồm 1,2s đuôi im lặng nhân tạo) ⇒ pipeline **theo kịp thời gian thực** với đầy đủ preview, không bỏ nhịp.

### 2.3. Vòng 2: mục tiêu đặt sai chỗ — **đây là góp ý quan trọng nhất**

Kế hoạch viết: *"thay đổi các thông số VAD ASR ... để giảm CER cho model được chọn. Mục tiêu là tiệm cận số liệu được báo cáo."*

Vấn đề: **con số được báo cáo (5,29%) là kết quả chạy OFFLINE 1 LƯỢT CHO CẢ CLIP, KHÔNG CÓ VAD.** Backend **không bao giờ** chạy như vậy — nó `VAD → cắt câu → commit từng đoạn`. Hệ quả:

1. **Thông số VAD không thể chạm tới con số 5,29%** — đường offline không đi qua VAD. Chỉnh `silence_duration_ms`/`threshold` sẽ **không đổi một ký tự nào** của CER offline.
2. **Con số đó ta đã đạt rồi** (5,36% đo ở §1.5). Vòng 2 như đang viết là **không có gì để tối ưu**.
3. Ngược lại, **CER của pipeline luôn cao hơn trần**, và đó mới là chất lượng sản phẩm thật.

**Đo được khoảng cách đó** (10 clip, `speed=1.0`, cùng model đã chọn). ⚠️ **Cách trình bày bên dưới đã được sửa theo góp ý review:** mẫu 10 clip này là mẫu "xấu" (macro offline 11,03% trong khi cả 650 là 5,35%), nên **cột mức tuyệt đối không được dùng để ra quyết định**. Chỉ đọc **Δ gap** và **số clip đạt parity** — hai đại lượng này bất biến với việc mẫu khó hay dễ:

| Cấu hình | Δ gap (pipeline − offline) | **Số clip đạt parity** | commits/clip | *mức tuyệt đối (tham khảo)* |
| --- | ---: | ---: | ---: | --- |
| `base` (mặc định: silence=600, max_dur=8, overlap=250, tier234 BẬT) | **+9,12** | 0/10 | 1,3 | *20,15%* |
| `repeat` (y hệt base — đo sàn nhiễu) | +8,69 | 0/10 | 1,4 | *19,71%* |
| `lang_ja` (`asr.language="ja"`) | +9,12 | 0/10 | 1,4 | *20,15%* |
| `long_seg` (`max_duration_sec=20`, `boundary_overlap_ms=0`) | **+8,16** | **2/10** | 1,2 | *19,18%* |
| `vad_only` (`enable_tier234=False`) | **+8,16** | **2/10** | 1,1 | *19,18%* |

Ba phát hiện từ bảng này:

- **Sàn nhiễu của pipeline = 0,44 điểm** ở n=10 (base 20,15 vs repeat 19,71). Hiệu ứng `long_seg` (0,97 điểm) chỉ **vừa** vượt sàn ⇒ ở n=10 **chưa được kết luận**. → **Đã giải quyết ở §3.6**: chạy lại trên JA30 với `base` lặp ×3 cho **sàn 0,57 điểm** và hiệu ứng **2,36 điểm = 4,1× sàn** ⇒ **kết luận vững**.
- `long_seg` đưa **2/10 clip về đúng bằng offline** (16,3s: 5,6%→1,9%; 14,7s: 9,0%→3,0%) — và `base` đạt **0/10**. Chỉ số parity là bằng chứng mạnh nhất ở đây vì nó không phụ thuộc độ khó của mẫu. ⇒ `max_duration_sec=8` + `boundary_overlap_ms=250` là **nguồn mất mát chính** với câu dài kiểu FLEURS, không phải VAD threshold.
- `lang_ja` **không đổi gì** (Δ gap y hệt base). Với clip sạch như FLEURS, auto-detect đã đúng. Knob này vẫn đáng thử trên **đoạn ngắn/nhiễu**, nhưng đừng kỳ vọng trên tập này.

> **Định nghĩa lại mục tiêu vòng 2:** *"thu hẹp `CER_pipeline − CER_offline` (mất mát do phân đoạn) từ +8…+9 điểm xuống mức nhỏ nhất, đồng thời giữ `wall/audio < 1` và p95 commit trong ngân sách."* Muốn so với công bố thì đó là **kiểm chứng tầng 1 của vòng 1** (đã đạt), không phải việc của vòng 2.

### 2.4. `speed > 1` làm đổi chính sách cắt câu — cái bẫy có sẵn trong repo

`backend/tests/test_09_wer_ab.py` mặc định `--speed 6`. Nhưng `SentenceConfig.stability_duration_sec`, `inactivity_timeout_sec`, `stability_threshold_polls`, `ASRConfig.poll_interval_ms` đều đọc **đồng hồ thực (`time.perf_counter`)**, không phải đồng hồ mẫu audio. Chạy `speed=6` cho ra **chính sách cắt câu khác** so với sản phẩm.

⇒ Nếu mục tiêu là tối ưu cho sản phẩm, vòng 2 **bắt buộc chạy `speed=1.0`**.

Chi phí: 1 cấu hình × 100 clip ở speed=1 ≈ **22 phút**. 10 cấu hình = 3,7 giờ. Vì vậy cần chiến lược sàng lọc 2 tầng (§3.3).

### 2.5. Ba lỗi kỹ thuật nhỏ nhưng sẽ làm báo cáo sai

1. **`commit_reason` và `inference_ms` KHÔNG có trong gói tin WebSocket.** `backend/ws/serializers.py:make_utterance_update_msg()` chỉ gửi `type/utterance_id/text/stable_text/unstable_text/translated/is_final/filtered`. Harness nào đọc `m.get("commit_reason")` từ `ws.sent_messages` sẽ nhận `?` và `0.0` — đúng như đã xảy ra với tôi lần đầu. ⇒ Lấy từ `metrics_collector` (`asr.commit_ms`, `asr.commit_reason.*`) hoặc hook thẳng `engine.stream_tokens()`.
2. **Gói `filtered=True`** (câu ngắn hơn `min_words_to_commit`) vẫn có `is_final=True` và vẫn mang text. Phải quyết định: chấm trên **chuỗi phụ đề người dùng thấy** (bỏ `filtered`) hay **toàn bộ text model sinh ra** (giữ). Tôi đề xuất báo cáo **cả hai**, lấy bản người-dùng-thấy làm chính.
3. **Tiến trình có thể thoát với exit code 1** sau khi đã ghi đủ JSON (native teardown khi interpreter tắt, quan sát ở lần chạy đầu). Bọc `os._exit(0)` ở cuối harness hoặc kiểm tra `$LASTEXITCODE` trước khi kết luận "harness lỗi".

### 2.6. Hiểu đúng về knob `asr.language`

`docs/models/qwen3-asr-1.7b.md` §"Public API caveat — language hints" nói **mọi hint đều trả `TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`**. **Tài liệu này đã cũ so với code**: `external/transcribe.cpp/src/arch/qwen3_asr/model.cpp:451-498` có bảng map BCP-47 → `"language <Name>"` + token `<asr_text>`, và tôi đã **chạy thử thành công** `session.run(audio, language="ja", ...)`.

⇒ `config.asr.language="ja"` là knob **hợp lệ** (không phải nguyên nhân lỗi), nhưng theo §2.3 nó **không cải thiện** gì trên FLEURS sạch. Giữ lại cho tập nhiễu/đoạn ngắn.

---

## 3. Thiết kế vòng test đề xuất (thay thế)

### 3.1. Giai đoạn 0 — Hạ tầng (nửa ngày)

1. **Adapter FLEURS** → manifest JSONL `{id, audio, ref_text, duration, gender}` đọc từ `test.tsv` cột index 3. Kèm **assert**: tổng `ref_chars` của cả 650 phải = 32.665 ± 0,5% (khớp catalog).
2. **Bộ chấm chuẩn**: `whisper_normalizer.BasicTextNormalizer` + `jiwer`, CER bỏ khoảng trắng — copy logic `scripts/wer/score.py`. Đã cài sẵn `whisper-normalizer`.
3. **Tập cố định** để A/B:
   - `JA100`: 100 clip chọn **deterministic** (băm tên file, không random theo giờ), **chia tầng theo độ dài** (6–10s / 10–15s / 15s+) để giảm phương sai.
   - `JA30`: 30 clip con của `JA100` cho sàng lọc nhanh.
4. **Ghi log**: tắt spam CUDA graph (chuyển stderr ra file, lọc), checkpoint JSONL để **resume**.
5. **Hàng rào RAM**: dùng `backend/utils/mem_guard.py` (đã có tiền lệ phình tới 33GB).

Đã có sẵn 3 script làm được việc này, xem §5.

### 3.2. Vòng 1 — Chọn model (3 tầng, ~3 giờ)

| Tầng | Dữ liệu | Đường chạy | Chi phí | Mục đích |
| --- | --- | --- | --- | --- |
| **1a** | **650** | offline 1 lượt, batch 1 | ~3’ × 7 = **25’** | Xếp hạng **trần CER** + RTF model. Đối chiếu catalog ⇒ chứng minh tái lập (đã làm cho qwen1.7b: 5,36 vs 5,29). |
| **1b** | **100** (`JA100`) | **pipeline, speed=1.0** | ~22’ × 7 = **2,5h** | Xếp hạng **CER sản phẩm thật** + biên realtime + commits/clip. |
| **1c** | **20** | pipeline, speed=1.0, **lặp ×3** | ~15’ × 7 | **Sàn nhiễu riêng cho từng model** — điều kiện bắt buộc để tin 1b. |

> ✅ **Tầng 1c đã chạy xong cho `qwen3-asr-1.7b`** (thực hiện trên JA30 thay vì 20 clip — xem §3.6.A): sàn nhiễu = **0,57 điểm %**. Đây là mốc so sánh dùng cho mọi A/B của vòng 2. Các model khác **chưa** có sàn riêng — nếu model được chọn ở vòng 1 không phải qwen3-1.7b, phải đo lại sàn cho nó trước khi tối ưu.

Nếu thiếu thời gian: bỏ 1b cho 4 model yếu, chỉ chạy top-3 của 1a.

**Quy tắc chọn model — chấm theo 5 trục, không phải argmin CER:**

```
BẮT BUỘC ĐẠT (cổng, loại trực tiếp nếu vi phạm):
    p95(commit_ms) < poll_interval_ms = 300 ms      # nếu không, pipeline bỏ nhịp preview
    wall/audio (speed=1.0) < 1.0                    # còn biên cho VAD + dịch + TTS
    empty_hyp = 0 trên JA100, không lặp vòng

XẾP HẠNG (trong số model đã qua cổng):
    1. CER_pipeline trên JA100 — có PHÂN TẦNG theo bucket độ dài, không gộp
    2. Δ gap = CER_pipeline − CER_offline            # đo riêng phần mất do phân đoạn
    3. commits/clip  (mong đợi ≈ số câu thật; 1,0 tốt hơn 1,5 vì ít cắt vụn)
    4. hồ sơ lỗi S/D/I                               # xoá chữ hại tầng dịch hơn thay-thế
    5. VRAM rồi dung lượng GGUF (tie-break)
```

**Vì sao xếp `p95(commit_ms)` và `commits/clip` ngang hàng với CER:** với phụ đề realtime, người dùng cảm nhận **độ trễ chốt câu** rõ hơn chênh lệch 0,5 điểm CER. Một model CER tốt hơn 0,4 điểm nhưng `p95 commit` vượt 300ms sẽ khiến pipeline bỏ nhịp preview — trải nghiệm tệ hơn hẳn. Chênh lệch CER trong nhóm dẫn đầu (4,82 / 5,13 / 5,29) **nhỏ hơn** ngưỡng phân giải ở mọi cỡ mẫu thực tế, nên các trục 3–5 sẽ là thứ thực sự phân xử.

Báo cáo kèm **khoảng cách so với catalog** cho cả 7 model (tầng 1a) để biết model nào có "vấn đề môi trường" riêng.

### 3.3. Vòng 2 — Thu hẹp khoảng cách phân đoạn (2 tầng)

**Mục tiêu viết lại:** giảm `CER_pipeline − CER_offline` trên `JA100`, không phải giảm CER tuyệt đối về 5,29%.

- **Sàng lọc:** `JA30`, speed=1.0, **1 lần chạy/cấu hình** (~7’/cấu hình) → giữ top-3.
- **Xác nhận:** top-3 trên `JA100`, speed=1.0, **lặp ≥2 lần** (~45’/cấu hình) → chỉ nhận cấu hình thắng **lớn hơn sàn nhiễu đo được**.
- **Chốt:** cấu hình thắng trên **cả 650** ở speed=1.0 (~2,5 giờ) để có con số phát hành.

**Ngân sách hợp lý:** ~10–14 cấu hình sàng lọc + 3 xác nhận + 1 chốt ≈ **6–8 giờ GPU**. Chạy 1 job tuần tự, không song song (tranh chấp GPU + native dùng `_shared_session` cấp lớp).

**Thứ tự chạy — TUYỆT ĐỐI tuân thủ (theo góp ý review):**

```
BƯỚC 1 (BẮT BUỘC LÀM TRƯỚC): đo SÀN NHIỄU
    base ×3 trên JA30, speed=1.0                        ~21 phút
    -> nếu chưa có con số này, MỌI kết luận A/B sau đây đều vô giá trị
       (tiền lệ: 14_wer_noise_floor.json cho sàn 3,34 điểm)

BƯỚC 2: NHÓM knob độ dài mảnh, như MỘT khối duy nhất
    long_seg  = {max_duration_sec 8→20, boundary_overlap_ms 250→0}   ~7 phút
    combo     = long_seg + {enable_tier234 False}                     ~7 phút
    -> DỪNG. Nếu cả hai không vượt sàn nhiễu ở bước 1 thì KHÔNG đi tiếp;
       ghi nhận "phân đoạn không phải nút thắt" rồi chuyển sang nhóm VAD.

BƯỚC 3 (chỉ khi bước 2 thắng RÕ): tách từng knob trong nhóm
    max_duration_sec đơn lẻ / boundary_overlap_ms đơn lẻ / tier234 đơn lẻ

BƯỚC 4 (chỉ khi bước 3 xong): nhóm VAD
    vad_engine (firered → silero → fsmn), rồi threshold, rồi silence_duration_ms,
    rồi pre_speech_buffer_ms, hangover_ms
    LƯU Ý: đổi `vad_engine` phải dựng lại VADProcessor; `threshold` có hiệu lực
    tức thì. Mỗi lần đổi engine là một "môi trường" mới ⇒ phải lặp lại bước 1.

KHÔNG BAO GIỜ: chạy song song, hoặc grid search toàn bộ tổ hợp.
```

**Danh sách knob, xếp theo kỳ vọng giảm khoảng cách** (kèm cơ sở trong code):

| # | Knob | Thay đổi đề xuất | Vì sao |
| --- | --- | --- | --- |
| 1 | `sentence.max_duration_sec` | 8 → **20 / 30** | `backend/asr/engine.py:1112` chặn mảnh commit ở `max_duration_sec × 1.5 = 12s`; clip FLEURS tới 28s ⇒ **câu dài bị cắt cụt phần đầu**. Đã thấy 2 clip về parity khi đổi. |
| 2 | `sentence.boundary_overlap_ms` | 250 → **0** | `engine.py:1183` cộng 250ms chồng lấn ở mỗi ranh giới không-VAD; chồng lấn sinh **insertion** (CER phạt trực tiếp). |
| 3 | `asr.preview_window_sec` | giữ ≥ `max_duration_sec` | Chỉ ảnh hưởng preview, **không** ảnh hưởng commit — đừng kỳ vọng nó sửa CER, nhưng phải đồng bộ kẻo preview lệch commit. |
| 4 | `vad.silence_duration_ms` | 600 → **300 / 450** | Cắt câu sớm hơn ⇒ đoạn ngắn hơn, ít bị clamp `max_duration`. Rủi ro: cắt giữa từ. |
| 5 | `vad.pre_speech_buffer_ms` | 300 → **500** | `engine.py` cắt theo mẫu; đệm trước thiếu ⇒ **mất phụ âm đầu** (deletion ở đầu mỗi câu). |
| 6 | `sentence.enable_tier234` | True → **False** | Chỉ cắt theo VAD ⇒ bớt cắt giữa câu. Đã thấy cùng mức cải thiện như `long_seg`. |
| 7 | `vad.hangover_ms` | 400 → **200** | Giảm phần đuôi im lặng lẫn vào đoạn. |
| 8 | `vad.vad_engine` | `firered-vad` (mặc định) → **`silero-vad`**, rồi `fsmn-vad` | Cùng audio, so số điểm cắt & CER. Cả 3 engine đều có model cục bộ. |
| 9 | `vad.threshold` | 0,45 → **0,30 / 0,60** | Đánh đổi bỏ sót lời nói vs cắt nhầm. |
| 10 | `asr.min_transcribe_sec` | 0,35 → **0,20** | Bắt các câu rất ngắn mà hiện tại bị trả về chuỗi rỗng (`engine.py:668`). |
| 11 | `asr.language` | `auto` → **`ja`** | Hợp lệ (§2.6) và no-op trên FLEURS sạch; **giữ cho tập nhiễu/đoạn ngắn**. |
| 12 | `sentence.stability_min_duration_sec` / `min_words_to_commit` | tăng nhẹ | Giảm cắt sớm bởi BẬC 3; đánh đổi độ trễ. |
| ~~—~~ | ~~`asr.normalize_speech`~~ | ~~ON/OFF~~ | **Đã đo: no-op về CER (Δ 0,02 điểm). Bỏ.** |
| ~~—~~ | ~~`vad.firered.min_silence_frame`~~ | ~~—~~ | **Config CHẾT** với đường FireRed — chính `config.py:71-79` đã ghi rõ. Đừng phí thời gian. |
| ~~—~~ | ~~`asr.request_timestamps=True`~~ | ~~—~~ | Model không hỗ trợ timestamps; chỉ tổn thời gian materialize. |
| ~~—~~ | ~~`asr.backend` cuda↔vulkan~~ | ~~—~~ | Đã đo trước đây: chênh độ chính xác < sàn nhiễu, CUDA nhanh hơn 1,53×. |

> **Đừng chạy grid search đầy đủ.** Các knob nhóm 1–3 và 6 tác động lên **cùng một cơ chế** (độ dài mảnh commit). Hãy chạy chúng như **một nhóm** (`long_seg`) trước, rồi tách riêng từng cái chỉ khi cần biết cái nào đóng góp bao nhiêu.

### 3.4. Tiêu chí "đạt" — định nghĩa trước khi đo

| Mức | Điều kiện | Kết quả CUỐI (650 clip, §3.7) |
| --- | --- | --- |
| **Tái lập trần** | `CER_offline(650)` nằm trong CI95 catalog | ✅ **ĐẠT** — 5,35% ∈ [4,81; 5,80] của catalog (5,29%) |
| **Chấp nhận pipeline** | `CER_pipeline(650) ≤ 1,5 × CER_offline(650)` = ≤ 8,03% | ❌ **CHƯA ĐẠT** — `vad_sil450` 10,97% = **2,05×** (đã cải thiện từ 2,24× của `base`) |
| **Tối ưu thành công** | giảm ≥ 20% tương đối **và** gap ≤ 3 điểm % | 🟡 **MỘT PHẦN** — giảm **8,3%** (mục tiêu 20%), gap **+5,62** (mục tiêu ≤3); **vượt sàn nhiễu và có ý nghĩa thống kê** |
| **Không hồi quy (độ chính xác)** | không file nào xấu đi trên tập kiểm chứng chéo | ✅ **ĐẠT** — 0/7 file xấu đi; 2 file cải thiện mạnh (−26,3 và −23,1 điểm) |
| **Không hồi quy (độ trễ)** | `p95 commit_ms < poll_interval_ms=300ms` | ❌ **CHƯA ĐẠT** — file vượt ngưỡng tăng **19 → 89/650**; p95 của p95 `292 → 360 ms` |

**Tóm tắt trung thực:** vòng 2 đã **lấy hết phần dễ**. Cải thiện có thật, có ý nghĩa thống kê,
không hồi quy độ chính xác — nhưng **không đạt** hai mục tiêu định lượng đã đặt trước
(20% và gap ≤3). Hai mục tiêu đó **không thể đạt bằng quét tham số**: gap còn lại nằm ở
**ranh giới commit bị chẻ từ** (xem §3.9), cần thay đổi logic tầng văn bản.

**Đọc bảng này cho đúng:** kế hoạch **không** thất bại, nhưng "đạt mục tiêu" còn xa. Cần thêm hai việc: (a) tìm `max_duration_sec` trung gian để kéo p95 về dưới 300 ms, (b) tấn công bucket 10–15s — nơi gap vẫn còn **+10 điểm** sau khi đã sửa `max_duration`. Gap còn lại nhiều khả năng nằm ở **ranh giới cắt giữa câu** (từ bị chẻ đôi qua hai mảnh), đúng nhóm knob #4–#5 trong §3.3.

⚠️ **Cổng "KHÔNG hồi quy" là điều kiện CHẤP NHẬN, không phải "theo dõi thêm".** FLEURS là audio đọc sạch, một giọng, không nhạc nền, không trộn ngôn ngữ. Tối ưu `long_seg`/`silence_duration_ms` trên FLEURS **có thể** làm hỏng hành vi trên video thật (ví dụ tăng `max_duration_sec` khiến một câu nói dài bị model bịa thêm, hoặc giảm `silence_duration_ms` khiến cắt giữa từ khi có nhạc nền). Nếu không đưa tập nhiễu vào tiêu chí chấp nhận cuối, ta sẽ **tối ưu quá khớp cho FLEURS rồi regress trên sản phẩm**.

### 3.5. Tập kiểm chứng chéo — bắt buộc, dùng chính `wav_test/` sẵn có

Không cần tải dữ liệu mới. Dùng các file đã có trong `wav_test/` (đều có `.txt` ground truth):

| File | Ngôn ngữ | Nó kiểm tra điều gì mà FLEURS không có |
| --- | --- | --- |
| `Japanese_5s.wav` | ja | Câu tiếng Nhật **rất ngắn** — nơi auto-detect ngôn ngữ và `min_transcribe_sec` dễ sai |
| `Chinese_noise_28s.wav` | zh | **Nhiễu nền** — nguy cơ hallucination/lặp vòng |
| `English_multiple_kinds_of_noise_88s.wav` | en | **Audio dài 88s + nhiều loại nhiễu** — kiểm tra phân đoạn dài, clamp `max_duration`, RAM |
| `Cross_lingual_English_French_Italian_Spanish_6s.wav` | trộn | **Chuyển ngôn ngữ giữa câu** — kiểm tra `language="ja"` có gây hại không |
| `English_low_speech_quality_19s.wav` | en | Chất lượng thu kém |
| `00_ingress_stream.wav` | en | Đường ingress thật (có nhãn `Speaker N:` / timestamp) |
| `Chinese_fast_speed_11s.wav` | zh | **Nói nhanh** — VAD dễ cắt giữa từ |
| `Russian_4s.wav` | ru | Ngôn ngữ không nằm trong nhóm tối ưu |

Cách chấm: dùng đúng logic `score.py` — CJK (ja/zh/yue/ko) → CER `BasicTextNormalizer` bỏ khoảng trắng; còn lại → WER `EnglishTextNormalizer` (đã có `whisper_normalizer`). Đây cũng là lý do **`asr.language="ja"` phải bị loại bỏ ngay** nếu nó làm `Cross_lingual_*` hoặc `Russian_4s` xấu đi — trên FLEURS thuần Nhật, knob này trông vô hại.

Chi phí: 2 cấu hình × 8 file (~2,7 phút audio) ở speed=1.0 ≈ **6 phút**. Rẻ đến mức không có lý do bỏ qua.

### 3.6. KẾT QUẢ JA30 — đã chạy xong, thay thế mọi kết luận từ mẫu 10 clip

Harness `scratch/ja30_pipeline.py`, model `qwen3-asr-1.7b`, speed=1.0, **JA30 phân tầng** (6–10s: 6 clip, 10–15s: 16, 15s+: 8), tổng audio 394s, **trần offline của mẫu = 5,75%** (so với 5,35% toàn 650 — mẫu phân tầng đại diện hơn hẳn mẫu 10 clip trước đó vốn cho 11,03%).

> ⚠️ **Cách đọc bảng:** chỉ so **Δ gap** và **parity** giữa các cấu hình với nhau. Cột mức tuyệt đối (pipe %) chỉ có nghĩa trong nội bộ JA30 vì mẫu khó hơn trung bình.

#### A. Sàn nhiễu — `base` lặp 3 lần

| Lần | CER pipeline | Δ gap | parity | commits/clip | previews/clip | p95 commit (max) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 14,80% | +9,05 | 9/30 | 2,00 | 27 | 306 ms |
| r2 | 15,37% | +9,62 | 7/30 | 2,00 | 27 | 319 ms |
| r3 | 14,97% | +9,22 | 10/30 | 1,93 | 27 | 310 ms |
| **Trung bình** | **15,05%** | **+9,29** | **8,7/30** | 1,98 | 27 | 319 ms |

⇒ **SÀN NHIỄU = 0,57 điểm %** (cả CER lẫn gap), parity trải **7…10 clip**. Đây là con số phải vượt qua. Ghi chú: `previews/clip` **giống hệt** ở cả 3 lần (27) ⇒ phần bất định nằm ở **điểm cắt câu của VAD và đầu ra model**, không phải ở nhịp preview.

#### B. Hiệu ứng NHÓM knob độ dài mảnh

| Cấu hình | CER pipeline | Δ gap | parity | commits/clip | p95 commit (max) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` (trung bình 3 lần) | 15,05% | +9,29 | 8,7/30 | 1,98 | 319 ms |
| **`long_seg`** (max_dur 8→20, overlap 250→0) | **12,69%** | **+6,93** | **14/30** | 1,60 | 359 ms |
| `combo` (`long_seg` + `tier234=False`) | 12,78% | +7,02 | 14/30 | 1,43 | 353 ms |

**Kết luận đã vượt sàn nhiễu:**

- `long_seg` giảm **2,36 điểm %** (−15,7% tương đối) và giảm gap **2,36 điểm** = **4,1× sàn nhiễu**. Số clip đạt parity tăng từ 8,7 lên **14/30**. ⇒ **Đây là kết luận vững**, khác hẳn trạng thái "chưa kết luận" ở mẫu 10 clip.
- `combo` **không khác** `long_seg` về CER (12,78 vs 12,69 = Δ0,09 ≪ sàn 0,57). Tác dụng duy nhất của `enable_tier234=False` là **giảm số câu cắt vụn** (1,60 → 1,43 commit/clip) **mà không tăng lỗi**. ⇒ Giữ hay bỏ đều được; đề xuất **bỏ** khỏi cấu hình phát hành để ít thay đổi hành vi hơn, trừ khi vấn đề "câu quá ngắn" xuất hiện trong sản phẩm thật.
- **`tier234` vốn KHÔNG phải nguyên nhân chính.** Ở mẫu 10 clip tôi từng thấy `long_seg` và `vad_only` cho **cùng** con số và nghi ngờ tier234; nay với n=30 rõ ràng: cái tạo ra khác biệt là **`max_duration_sec` + `boundary_overlap_ms`**, không phải tier234.

#### C. Phân tầng độ dài — **nút thắt nằm ở bucket 10–15s, không phải 15s+**

| Bucket | n | Trần offline | `base` (gap) | `long_seg` (gap) | `combo` (gap) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6–10s | 6 | 6,27% | 6,13% (**−0,1**) | 6,13% (**−0,1**) | 6,13% (**−0,1**) |
| **10–15s** | 16 | 5,38% | 17,95% (**+12,6**) | 15,35% (**+10,0**) | 15,60% (+10,2) |
| 15s+ | 8 | 6,12% | 15,93% (+9,8) | 12,28% (**+6,2**) | 12,10% (+6,0) |

Hai điều quan trọng:

1. **Clip ngắn (6–10s) đạt parity hoàn hảo** ở cả ba cấu hình — pipeline **không** mất gì khi câu không bị cắt. Đây là bằng chứng mạnh rằng mọi mất mát còn lại đều đến từ **phân đoạn**, không phải từ VAD/chuẩn hoá/preview.
2. **Bucket 10–15s mất mát nặng nhất (+12,6 điểm)**, nặng hơn cả 15s+. Lý do: `max_duration_sec=8` cắt một câu 12s thành **hai mảnh ~6s cắt giữa câu**, trong khi clip 15s+ cũng bị cắt nhưng rơi vào chỗ có khoảng nghỉ tự nhiên nhiều hơn. Đây là bucket **lớn nhất** của tập (348/650 clip), nên đây chính là chỗ đáng đầu tư nhất — và đây là thông tin mà mẫu 10 clip **không thể** cho ra.

#### D. Đánh đổi ĐỘ TRỄ — phát hiện mới, cần quyết định

| Cấu hình | p95 của per-file p95 | max tuyệt đối | số file có p95 > 300 ms |
| --- | ---: | ---: | ---: |
| `base` | 306 ms | 319 ms | 8/90 |
| `long_seg` | **331 ms** | **359 ms** | 8/30 |
| `combo` | **345 ms** | 353 ms | 10/30 |

**Cấu hình mặc định VỐN ĐÃ VƯỢT ngân sách `poll_interval_ms = 300 ms`** (p95 = 306–319 ms), và `long_seg` đẩy thêm **~25–40 ms**. Đây là đánh đổi thật: trả thêm độ trễ để lấy 2,36 điểm CER.

Ba lựa chọn, xếp theo đề xuất:
1. **Chọn `max_duration_sec` trung gian (12 s)** — đủ để bao trọn phần lớn câu 10–15s mà mảnh commit không dài tới 20s. Đây là việc của BƯỚC 3 và là lý do không nên nhảy thẳng 8 → 20.
2. **Nâng `poll_interval_ms` 300 → 400 ms** — preview thưa hơn ~33% nhưng commit vẫn kịp; đổi lại preview bớt mượt.
3. Giữ 20s và chấp nhận p95 ~350 ms (commit **không** bị bỏ khi vượt poll — chỉ preview bị bỏ nhịp, vì commit luôn được ưu tiên trước preview trong `stream_tokens`).

#### E. Tập kiểm chứng chéo — lo ngại "tối ưu quá khớp FLEURS" **đã bị bác bỏ bằng số liệu**

| File | Lang | `base` | `combo` | Δ | commits b/c |
| --- | --- | ---: | ---: | ---: | --- |
| `Japanese_5s.wav` | ja | 0,00% | 0,00% | 0 | 1/1 |
| `Russian_4s.wav` | ru | 10,00% | 10,00% | 0 | 1/1 |
| `English_low_speech_quality_19s.wav` | en | 30,00% | **16,67%** | **−13,3** | 4/3 |
| `Chinese_noise_28s.wav` | zh | 47,37% | **21,05%** | **−26,3** | 4/1 |
| `Chinese_fast_speed_11s.wav` | zh | 69,23% | **30,77%** | **−38,5** | 2/1 |
| `Cross_lingual_English_French_Italian_Spanish_6s.wav` | trộn | 35,71% | 35,71% | 0 | 1/1 |
| `00_ingress_stream.wav` | en | 99,67% | 99,56% | −0,1 | 73/65 |

**Số file bị `combo` làm xấu đi: KHÔNG CÓ.** Nó **cải thiện rất mạnh** trên audio nhiễu và nói nhanh (−13…−38 điểm), và trung tính trên phần còn lại. Lo ngại "tối ưu trên FLEURS sạch rồi regress trên dữ liệu thật" **không xảy ra với nhóm knob này** — ngược lại, chính `max_duration_sec=8` mới là thứ đang phá audio thật (4 commits cho `Chinese_noise_28s`, 2 commits cho câu nói nhanh 11s → cắt giữa từ).

*Ghi chú:* `00_ingress_stream.wav` cho ~99,6% ở cả hai cấu hình (73/65 commit) — file này gần như không dùng để chấm điểm được (ground truth là log dài, không khớp 1-1 với audio). Chỉ dùng nó làm **kiểm tra ổn định**: không crash, không rỗng, `wall/audio = 1.00`.

#### F. Việc còn lại của vòng 2 (đúng thứ tự đã định ở §3.3)

| Bước | Việc | Trạng thái |
| --- | --- | --- |
| 1 | Đo sàn nhiễu | ✅ **xong** — 0,27 điểm (base ×3, JA30, bản native sau patch) |
| 2 | Nhóm knob độ dài mảnh | ✅ **xong** — thắng **4,8–5,8× sàn nhiễu**, parity 8,7→13/30, không hồi quy |
| 3 | Tách knob + tìm `max_duration_sec` tối ưu độ trễ | ✅ **xong** — 12s/15s/20s **tương đương nhau**; chọn **15s** |
| 4 | Nhóm VAD | ✅ **xong** — `silence_duration_ms 600→450` thắng **3,0× sàn**; `pre_speech_buffer_ms 500` **thảm họa** |
| 5 | Xác nhận trên JA100 + chốt trên 650 | ✅ **xong** — chạy **cả 650 clip**, kiểm định cặp có ý nghĩa (§3.7) |
| 6 | Cổng kiểm chứng chéo cho cấu hình cuối | ✅ **xong** — **không file nào xấu đi** (§3.7) |

### 3.7. KẾT QUẢ CHỐT — 650 clip, kiểm định cặp (bước 5 + 6)

Chạy cả **650 clip** (không subsample), `speed=1.0`, `n_threads=4`, `require_gpu=True`.
Trần offline của cả 650 = **5,35%** (catalog 5,29%). Kiểm định bằng **bootstrap theo cụm
4000 lần trên hiệu số cặp** (cùng clip ⇒ mạnh hơn nhiều so với so hai CI rời).

| Cấu hình | CER | Δ vs base | **CI95 của Δ** | gap | parity | commits | p50 commit | p95(p95) | max | file >300ms |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` (mặc định hiện tại) | 11,969% | — | — | +6,62 | 254/650 | 1,77 | 179 ms | 292 ms | 365 ms | 19/650 |
| `mid15` (độ dài mảnh) | 11,207% | **−0,762** | [−1,028 ; −0,488] ✅ | +5,85 | 295/650 | 1,51 | 208 ms | 357 ms | 559 ms | 92/650 |
| **`vad_sil450`** (mid15 + silence=450) | **10,972%** | **−0,997** | **[−1,299 ; −0,700] ✅** | **+5,62** | 286/650 | 1,60 | 208 ms | 360 ms | 562 ms | 89/650 |

**Cả hai cấu hình đều có ý nghĩa thống kê** (CI95 của hiệu số không chứa 0). Kết luận:

- **`vad_sil450` là cấu hình tốt nhất**: giảm **0,997 điểm %** (−8,3% tương đối), 172 clip tốt
  hơn / 78 xấu hơn / 400 bằng.
- **Cải thiện tăng theo độ dài clip** — đúng như cơ chế dự đoán:

  | Bucket | n | base | mid15 | **vad_sil450** |
  | --- | ---: | ---: | ---: | ---: |
  | 6–10s | 131 | 9,49% | 9,57% (+0,08) | **9,05% (−0,45)** |
  | 10–15s | 348 | 11,80% | 11,04% (−0,76) | **10,81% (−0,99)** |
  | 15s+ | 171 | 14,20% | 12,80% (−1,41) | **12,77% (−1,44)** |

  `vad_sil450` còn cải thiện cả bucket ngắn (mid15 thì không) ⇒ rủi ro hồi quy thấp hơn.

- **Cổng kiểm chứng chéo: KHÔNG file nào xấu đi.**

  | File | base | mid15 | vad_sil450 |
  | --- | ---: | ---: | ---: |
  | `Japanese_5s` | 0,00% | 0,00% | 0,00% |
  | `Russian_4s` | 10,00% | 10,00% | 10,00% |
  | `English_low_speech_quality_19s` | 30,00% | 16,67% | **23,33%** |
  | `Chinese_noise_28s` | 47,37% | 21,05% | **21,05%** |
  | `Chinese_fast_speed_11s` | 53,85% | 30,77% | **30,77%** |
  | `Cross_lingual_…_6s` | 21,43% | 21,43% | 21,43% |
  | `00_ingress_stream` | 99,67% | 99,67% | 99,34% *(không chấm được)* |

  ⚠️ "Hồi quy `Cross_lingual`" từng thấy ở lượt JA30 **không tái hiện** ở n=650 (21,43% ở cả
  ba) ⇒ đó là **nhiễu chạy lại trên file đơn lẻ**, không phải tác động thật.

- **Chi phí độ trễ là thật:** p50 commit tăng 179→208 ms, p95 của p95 tăng 292→360 ms, số file
  vượt ngân sách `poll_interval_ms=300ms` tăng **19→89 trên 650**. Nguyên nhân: câu dài hơn
  được transcribe trong một commit. **Không sửa được bằng `max_duration_sec`** (12/15/20 đều
  cho p95 ~330 ms ở JA30) — muốn hạ p95 phải hạ `poll_interval_ms` hoặc chấp nhận.

### 3.8. CẤU HÌNH PHÁT HÀNH ĐỀ XUẤT

```yaml
sentence:
  max_duration_sec: 15.0       # 8.0  -> 15.0   (12/15/20 tương đương; 15 là điểm cân bằng)
  boundary_overlap_ms: 0       # 250  -> 0
  enable_tier234: true         # GIỮ NGUYÊN — tắt không thêm lợi ích nào đã đo được
asr:
  preview_window_sec: 15.0     # phải >= max_duration_sec
  require_gpu: true            # BẮT BUỘC GPU (xem §8)
vad:
  silence_duration_ms: 450     # 600 -> 450
  pre_speech_buffer_ms: 300    # GIỮ NGUYÊN — 500 làm CER 13,2% -> 30,5% (!)
  hangover_ms: 400             # GIỮ NGUYÊN — 200 không cải thiện (trong sàn nhiễu)
```

**Hiệu quả kỳ vọng trên 650 clip:** `11,97% → 10,97%` (**−1,00 điểm %**, −8,3% tương đối),
parity `254 → 286/650`, **không hồi quy trên bất kỳ file kiểm chứng chéo nào**.
Đánh đổi: p95 commit `292 → 360 ms`.

### 3.9. Vì sao CHƯA đạt mục tiêu 20% — và nên làm gì tiếp

Sau cả nhóm độ dài mảnh lẫn nhóm VAD, `gap` vẫn còn **+5,62 điểm** và CER pipeline vẫn gấp
**2,05×** trần offline (mục tiêu ≤1,5×). Nguyên nhân còn lại **không nằm ở tham số nữa**:

1. **Câu bị chẻ đôi qua hai mảnh commit.** Bucket 15s+ vẫn mất **+7,77 điểm** so với offline.
   Cơ chế: một từ ở ranh giới bị cắt thành hai nửa nằm ở hai commit khác nhau ⇒ vừa deletion
   vừa insertion. `boundary_overlap_ms=0` đã bỏ chồng lấn (tốt) nhưng **không có cơ chế nối
   văn bản giữa hai commit**, nên phần bị chẻ vẫn mất.
2. **Chấm CER ở mức câu nhưng pipeline chốt theo đoạn VAD.** Không có chuẩn hoá/khôi phục
   văn bản xuyên commit (ví dụ gộp từ bị chẻ, hoặc so khớp mờ ở ranh giới).
3. **Ranh giới do VAD quyết định, không do ngữ nghĩa.** `silence_duration_ms=450` giúp vì cắt
   ở khoảng nghỉ ngắn hơn, nhưng vẫn là tín hiệu âm học thuần.

⇒ Bước tiếp theo **không phải** quét thêm tham số, mà là **xử lý ranh giới ở tầng văn bản**:
gộp/khôi phục token bị chẻ giữa hai commit liên tiếp (ví dụ dùng `commit_manager`/`dedup` sẵn
có, hoặc so khớp mờ hậu kỳ). Đây là thay đổi logic, cần một vòng đo riêng.

**Điểm mấu chốt về kỳ vọng:** trần offline 5,35% **đã đạt** và không thể vượt qua bằng tham
số. Mọi cải thiện còn lại chỉ có thể đến từ việc **giảm mất mát do phân đoạn**, và phần dễ
(nhóm độ dài mảnh + VAD) **đã lấy hết**.

---

## 4. Rủi ro & giảm thiểu

| Rủi ro | Bằng chứng | Giảm thiểu |
| --- | --- | --- |
| **Kết luận A/B do nhiễu** | `14_wer_noise_floor.json`: cùng cấu hình, `error_rate` 17,14% vs 20,48% ⇒ **sàn 3,34 điểm** | Luôn chạy cấu hình đối chứng lặp; chỉ nhận khi Δ > sàn. Dùng **so sánh cặp trên cùng clip**. |
| **Mẫu nhỏ gây sai lệch hệ thống** | Mẫu 10 clip cho macro 11,03% trong khi cả 650 là 5,35% | Dùng ≥100 clip có phân tầng, hoặc cả 650. |
| **FLEURS không đại diện sản phẩm** | Đọc sạch, 1 giọng, không nhạc nền, không trộn ngôn ngữ ⇒ tối ưu quá khớp rồi regress trên sản phẩm | **Đưa tập kiểm chứng chéo (§3.5) vào CỔNG CHẤP NHẬN**, không chỉ "theo dõi thêm": không tăng lỗi, `empty_hyp=0`, không lặp vòng, `p95 commit_ms < 300ms`. |
| **Mẫu A/B bị chọn theo cảm tính** | `sample = rows[::step]` (bước cố định) vẫn rơi vào mẫu khó: macro 11,03% vs 5,35% toàn tập | Chọn mẫu bằng **băm tên file + phân tầng độ dài**, và **công bố macro CER offline của mẫu** cạnh mọi kết quả để người đọc biết mẫu khó hay dễ. |
| **Phình RAM native** | Đã có tiền lệ 33GB; `run_paced` phải `cancel_inference()` giữa các lần chạy | `mem_guard`, checkpoint JSONL, resume, 1 job tuần tự. |
| **Đổi model giữa các lần chạy** | `_ensure_model_loaded` tự nạp lại khi `model_key` khác; session cũ bị `close()` | **Gom theo model**: chạy hết mọi cấu hình của model A rồi mới sang B. |
| **Nhiễm chéo cấu hình** | `test_09.apply_config()` chỉ set 4 tham số và dùng chung `config` toàn cục | Viết `apply_config()` **reset toàn bộ** về mặc định trước khi set. |
| **Tranh chấp GPU** | Backend/GUI có thể đang chạy | Chạy tuần tự, kiểm tra `nvidia-smi` trước; đo RTF khi GPU rảnh. |

---

## 5. Phụ lục — script đã viết và cách chạy

| File | Việc nó làm |
| --- | --- |
| `scratch/fleurs_stats.py` | Kiểm tra 650/650 wav khớp TSV + phân bố độ dài/ký tự |
| `scratch/tsv_probe.py` | Xác minh cấu trúc 7 cột của `test.tsv` |
| `scratch/pilot_fleurs_ja.py` | Pilot nhanh n clip, đo nạp model/RTF/CER, thử `language` hint |
| `scratch/baseline_fleurs_ja.py` | **Baseline offline toàn 650 clip**, A/B `normalize_speech`, ghi JSONL resume |
| `scratch/pipeline_fleurs_ja.py` | **So nhiều cấu hình qua pipeline thật** (`run_paced`, speed=1.0): offline vs pipeline vs gap |
| `scratch/ja30_pipeline.py` | **Harness chính cho vòng 2**: JA30 phân tầng + `base`×3 (sàn nhiễu) + nhóm knob độ dài + **tập kiểm chứng chéo**. Tiêu thụ thẳng `engine.stream_tokens()` nên lấy được `commit_reason`/`inference_ms` (2 field bị `ws/serializers.py` bỏ). |
| `scratch/power_analysis.py` | CI95 theo cỡ mẫu + công suất so sánh cặp |
| `scratch/analyze_ja30.py` | Phân tích `ja30_*.json` (bản cũ, hardcode) |
| `scratch/analyze_run.py` | **Phân tích tổng quát** mọi file kết quả: sàn nhiễu, so cấu hình, phân tầng, p95, regression |
| `scratch/final_analysis.py` | **Kiểm định cặp trên 650 clip** (bootstrap theo cụm) + phân tầng + cổng kiểm chứng chéo |
| `scratch/paired_test_650.py` | Bản kiểm định cặp cho A/B base vs mid15 |
| `scratch/gpu_or_cpu_probe.py` | **Chẩn đoán GPU/CPU**: wall vs `process_time` cho từng `backend` × `n_threads` ⇒ `cores_busy` + tỉ lệ cuda/cpu |
| `scratch/cohere_segment_probe.py` | Quét độ dài đoạn cắt × `language` để tìm nguyên nhân cohere ra tiếng Anh |
| `scratch/lang_ab.py` | A/B `asr.language` offline có phân tầng (dùng khi cần kiểm tra hint) |
| `scratch/show_pilot.py`, `show_pipeline.py` | Đọc JSON kết quả, in chẩn đoán từng clip |

Lệnh tái lập:

```powershell
$env:PYTHONPATH="D:\vibe-translation-addon-transcribe_cpp"
python scratch\fleurs_stats.py
python scratch\baseline_fleurs_ja.py qwen3-asr-1.7b          # ~6 phút, 650 clip x 2 cấu hình
python scratch\ja30_pipeline.py qwen3-asr-1.7b --smoke       # ~2 phút, kiểm tra harness
python scratch\ja30_pipeline.py qwen3-asr-1.7b               # ~40 phút: sàn nhiễu + nhóm knob + regression
python scratch\analyze_ja30.py                               # bảng §3.6 (bản cũ)
python scratch\analyze_run.py scratch\ja650_final.json       # bảng tổng quát
python scratch\final_analysis.py scratch\ja650_final.json    # kiểm định cặp trên 650
python scratch\gpu_or_cpu_probe.py cohere-transcribe --n 3 --reps 2 --backends cuda,cpu --threads 4
python scratch\gpu_or_cpu_probe.py cohere-transcribe --n 3 --reps 2 --backends cuda --threads 1,2,4
python scratch\cohere_segment_probe.py cohere-transcribe
python scratch\pipeline_fleurs_ja.py qwen3-asr-1.7b --limit 10 --speed 1.0
python scratch\power_analysis.py
```

Kết quả thô: `scratch/ja30_qwen3-asr-1.7b.json` (JA30 + kiểm chứng chéo), `scratch/baseline_qwen3-asr-1.7b.jsonl` (650 clip), `scratch/pipeline_qwen3-asr-1.7b.json` (10 clip), log đã lọc `scratch/ja30_full.clean.log`, `scratch/baseline_qwen17.clean.log`, `scratch/pipeline_qwen17.clean.log`.

> **Lưu ý:** các script này nằm trong `scratch/` để khảo sát. Khi chốt thiết kế, nên chuyển thành `backend/tests/test_41_fleurs_ja_bench.py` (không có hàm `test_*` để `pytest` mặc định vẫn nhanh) theo đúng quy ước của `test_08`/`test_09`.

---

## 6. Tóm lại

### 6.1. Ba việc cần đổi trong kế hoạch ban đầu của bạn

1. **Vòng 1: đổi 10 audio → 650 audio ở đường offline** (chỉ ~3 phút/model). 10 clip không phân biệt nổi 4,82% với 5,29%.
2. **Vòng 2: đổi mục tiêu** từ *"tiệm cận 5,29%"* (đã đạt, và VAD/ASR không chạm tới được) sang *"thu hẹp khoảng cách pipeline − offline"*, với `speed=1.0` bắt buộc và sàn nhiễu đo được.
3. **Thêm một tầng "kiểm chứng tái lập"** cho từng model: so `CER_offline` cục bộ với catalog. Với qwen3-1.7b nó **khớp tới 0,07 điểm**, nên đây là tầng rẻ và đáng tin — dùng nó làm mốc neo cho mọi kết luận sau.

### 6.2. Kết quả CUỐI của vòng 2 (650 clip, kiểm định cặp) — §3.7

| Nội dung | Kết quả |
| --- | --- |
| Sàn nhiễu pipeline (`base` ×3, JA30) | **0,27 điểm %** |
| Trần offline 650 clip | **5,35%** (catalog 5,29% — nằm trong CI95) |
| Nhóm knob độ dài mảnh (`mid15`) | −0,762 điểm, CI95 [−1,028; −0,488] ✅, parity 254→295/650 |
| Nhóm VAD (`vad_sil450`) | **−0,997 điểm**, CI95 [−1,299; −0,700] ✅, gap +6,62→**+5,62** |
| Cải thiện theo độ dài | 6–10s −0,45 · 10–15s −0,99 · 15s+ −1,44 (đơn điệu, đúng cơ chế) |
| Cổng kiểm chứng chéo | **0/7 file xấu đi**; `Chinese_noise` −26,3 và `Chinese_fast_speed` −23,1 điểm |
| Độ trễ commit | p50 179→208 ms; p95(p95) 292→**360 ms**; file >300ms 19→**89/650** |
| `enable_tier234=False` | Trung tính về CER (Δ0,09 < sàn) ⇒ **giữ True** |
| `pre_speech_buffer_ms=500` | **THẢM HỌA** — CER 13,2%→30,5%, p95 1451 ms ⇒ **cấm tăng** |
| `silence_duration_ms` 300 | Trung tính (0,7× sàn) ⇒ chỉ 450 có tác dụng |

### 6.3. Cấu hình phát hành đề xuất (tóm tắt)

```
sentence.max_duration_sec   : 8.0  -> 15.0
sentence.boundary_overlap_ms: 250  -> 0
asr.preview_window_sec      : 8.0  -> 15.0
vad.silence_duration_ms     : 600  -> 450
asr.require_gpu             : true          (bắt buộc GPU, xem §8)
GIỮ NGUYÊN: enable_tier234=True, pre_speech_buffer_ms=300, hangover_ms=400, language=auto
```

**Kết quả kỳ vọng:** CER pipeline `11,97% → 10,97%` (−1,00 điểm, −8,3% tương đối),
parity `254 → 286/650`, không hồi quy trên tập kiểm chứng chéo. Đánh đổi: p95 commit +68 ms.

### 6.4. Điều KHÔNG đạt được và lý do (không nên thử lại bằng tham số)

Hai mục tiêu định lượng đặt ra ban đầu **không đạt**: giảm ≥20% (thực tế 8,3%) và gap ≤3
(thực tế +5,62). Nguyên nhân đã xác định ở §3.9: phần mất mát còn lại là **từ bị chẻ đôi qua
ranh giới hai commit**, không phải chọn tham số sai. Cải thiện tiếp **phải** đến từ tầng văn
bản (nối/khôi phục token ở ranh giới), không phải từ việc quét thêm `max_duration_sec`,
`silence_duration_ms` hay `threshold`.

---

## 7. Chuyển sang `cohere-transcribe` — hai phát hiện nghiêm trọng

Bỏ vòng 1 và chỉ định trước model là hợp lý (catalog đã xếp hạng sẵn). Nhưng khi chạy vòng 2
cho `cohere-transcribe-03-2026-Q8_0`, hai vấn đề xuất hiện mà **không** có với `qwen3-asr-1.7b`.
Cả hai đều đã được đo, không phải suy đoán.

### 7.1. 🔴 `language="auto"` phá hủy hoàn toàn kết quả của cohere

`src/arch/cohere/model.cpp:844`:

```cpp
const char * lang = (params && params->language) ? params->language : "en";
```

Cohere **không có auto-detect** (`catalog/cohere-transcribe-03-2026.json`:
`"lang_detect": {"supported": false}`), và khi caller không truyền hint thì prompt chứa
`<|en|>`. Backend mặc định `asr.language = "auto"` ⇒ `normalize_language_for_family()` trả
`None` ⇒ binding nhận `language=NULL` ⇒ **prompt tiếng Anh cho audio tiếng Nhật**.

Đo trên cùng audio, chỉ khác `language` (pipeline thật, `speed=1.0`):

| `language` | trần offline | **CER pipeline** | Δ gap | p95 commit | đầu ra điển hình |
| --- | ---: | ---: | ---: | ---: | --- |
| **`auto`** (mặc định sản phẩm) | 4,69% | **106,25%** | +101,6 | 812 ms | `'The book is a book called The book of the book'` |
| **`ja`** | **1,56%** | **8,94%** | **+7,38** | 584 ms | `'洞窟はメッカの北にあたる山頂にあり、外海から完全に隔離されています。'` |

**CER giảm 12×.** Với `auto`, cohere sinh **tiếng Anh**, **lặp vòng**
(`'He was a man of the world.'` ×5) và có lúc **lỗi decode**
(`OutputTruncated: transcribe_run: output truncated: decode`). Với `ja`, đầu ra tiếng Nhật
đúng và gap +7,38 nằm cùng thang với qwen (6,93–9,29).

Quét độ dài đoạn (`scratch/cohere_segment_probe.py`) xác nhận cả hai chiều:

| độ dài | `language="auto"` | `language="ja"` |
| --- | --- | --- |
| cả clip (7s) | ❌ `OutputTruncated` | ✅ câu đúng |
| 6s | `'Gunn Tonga Yangi Nianga, Tonga Tonga, and Kanga Tonga.'` | ✅ câu đúng |
| 4s | `'Guangdong and Nizami Dhaka are the most important...'` | `'群島や湖ではかなわない'` |
| 2s | `'I'm sorry.'` / `'Thank you.'` | `'よし!'` / `'何?'` |
| 1s | `'Thank you.'` | `'三井不動産は?'` |

Hai kết luận:
1. **`auto` không dùng được với cohere.** Số công bố 5,13% trong catalog chắc chắn đo bằng
   `--language ja`: `scripts/wer/run.py:330` **suy ngôn ngữ từ manifest** khi không truyền
   `--language`, nên benchmark của tác giả luôn có hint.
2. **Cohere hallucinate nặng trên đoạn ngắn** (≤2s → "Thank you.", "I'm sorry."). Đây là rủi ro
   thật vì pipeline sinh ra đoạn ngắn (VAD + clamp + mảnh đuôi). Trần offline 1,56% của cohere
   tốt hơn qwen, nhưng **độ bền trên đoạn ngắn thì kém hơn hẳn** — phải kiểm tra riêng.

### 7.2. 🔴 Lỗi trong backend: `config.asr.language` không có tác dụng khi tạo engine trực tiếp

`backend/asr/engine.py:217,228`:

```python
language: str = "auto",          # 217 — chuỗi TRUTHY
...
self.language = language or config.asr.language   # 228 — "auto" luôn thắng
```

Vì `"auto"` là truthy, `config.asr.language` **không bao giờ được đọc** ở constructor. Nó chỉ
hoạt động nhờ đường vòng: `ws/session.py:79` đưa `config.asr.language` vào
`session.config["source_lang"]`, rồi `session.py:449` gọi `asr_engine.set_language(...)`.

Hệ quả:
- **Sản phẩm:** vẫn đúng *nếu* đi qua `SessionState` — nhưng đổi `config.asr.language` một mình
  sẽ **không** có tác dụng ở mọi nơi khác (harness, `prewarm`, `hotswap`, test).
- **Chính báo cáo này:** A/B `lang_ja` ở §3.6 **là kết quả SAI** — harness tạo engine trực tiếp
  nên nó âm thầm chạy lại `auto`. Với qwen, kết luận **không đổi** (`auto` = auto-detect vốn
  đúng cho tiếng Nhật, nên `auto` ≈ `ja`); nhưng với cohere thì sai hoàn toàn.

**Đề xuất sửa (1 dòng):** đổi chữ ký thành `language: Optional[str] = None` để `or` hoạt động
đúng như chủ đích — và để harness không phải nhớ truyền tay.

### 7.3. 🟡 Cohere chỉ chạy một phần trên GPU (giải thích "%CPU > %GPU")

Nghi vấn "log báo CUDA nhưng thực ra chạy CPU" đã được kiểm chứng bằng đo trực tiếp
(`scratch/gpu_or_cpu_probe.py`, cùng 3 clip, `n_threads=4`):

| model | backend | wall | CPU-s | **cores busy** | RTF | ms/clip |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cohere | **cuda** | 2,89 s | 9,22 | **3,19** | 0,102 | 962 |
| cohere | cpu | 71,93 s | 285,78 | 3,97 | 2,551 | 23 977 |
| qwen 1.7b | **cuda** | 1,13 s | 1,19 | **1,05** | 0,020 | 189 |

**cuda nhanh hơn cpu 24,9×** ⇒ GPU **có** làm phần nặng. Nhưng quét `n_threads` cho thấy hai
model hành xử khác hẳn nhau:

| model (cuda) | thr=1 | thr=2 | thr=4 | kết luận |
| --- | --- | --- | --- | --- |
| **cohere** wall | 13,61 s | 8,01 s | **5,86 s** | thêm thread **nhanh hơn 2,3×** ⇒ **có tính toán THẬT trên CPU** |
| cohere cores | 1,00 | 1,78 | 3,39 | tăng theo `n_threads` |
| **qwen** wall | 1,23 s | 1,18 s | 1,13 s | gần như không đổi ⇒ thuần GPU |
| qwen cores | 1,00 | 1,06 | 1,05 | hằng số |

Đây **phủ định** giả thuyết "chỉ là threadpool dùng-một-lần + busy-wait" mà
`02_phu_luc_transcribe_cpp.md` §1b–1c mô tả: overhead thuần thì thêm thread sẽ **tốn CPU mà
không nhanh hơn**. Ở cohere, thêm thread **giảm wall 2,3×** ⇒ một phần đáng kể graph của cohere
thực thi trên backend CPU ngay trong lượt `backend=cuda`. Việc này khớp với thiết kế:
`transcribe-load-common.cpp` dựng `scheduler_list = [backend chính, CPU dự phòng]`, nên op nào
CUDA không hỗ trợ sẽ âm thầm rơi về CPU (chính repo ghi `bin/ggml-cpu.dll` "vẫn được giữ vì
ggml cần nó cho op không offload được").

*Chưa xác định được cụ thể op nào rơi về CPU* — cần instrument sâu hơn (liệt kê node theo
backend trong `ggml_backend_sched`). Điều **đã** chứng minh: hiện tượng là thật với cohere,
và **không** xảy ra với qwen.

### 7.4. Chi phí thật của việc chọn cohere

| Chỉ số | `qwen3-asr-1.7b` | `cohere-transcribe` | Chênh |
| --- | ---: | ---: | --- |
| RTF offline (CUDA) | **0,020** | 0,104 | cohere **chậm 5×** |
| cores busy (CUDA) | **1,05** | 3,39 | cohere **ăn 3,2 nhân CPU** |
| p95 commit (pipeline) | 306–319 ms | **584–924 ms** | vượt ngân sách `poll_interval_ms=300ms` ~2–3× |
| Trần offline (FLEURS ja) | 5,36% (catalog 5,29) | cần đo lại 650 | — |
| Bền trên đoạn ≤2s | tốt | **hallucinate** ("Thank you.") | — |

Hệ quả cho sản phẩm: `poll_interval_ms = 300 ms` mà preview một cửa sổ 8s ở RTF 0,104 tốn
**~830 ms** ⇒ preview **không thể theo nhịp 300 ms**, sẽ bị giãn/backoff liên tục. Cộng thêm
3,4 nhân CPU tranh chấp với VAD (torch), dịch (4 thread), TTS và browser decode.

### 7.5. Việc cần làm trước khi tiếp tục vòng 2 với cohere

1. **Sửa `engine.py:217` (`language: Optional[str] = None`)** — nếu không, mọi A/B về ngôn ngữ
   đều đo sai. ✅ *đã bù bằng cách truyền `language=` tường minh trong harness.*
2. **Đặt `source_lang = "ja"`** cho cohere (và xác nhận catalog cũng dùng `--language ja`).
3. **Đo lại trần offline cohere trên cả 650 clip** với `language="ja"` — chưa có, và mọi Δ gap
   đều cần nó. (~10 phút vì cohere chậm hơn qwen ~5×.)
4. **Đo sàn nhiễu riêng cho cohere** (`base` ×3 trên JA30) — sàn của qwen (0,57) không dùng lại được.
5. **Kiểm tra độ bền trên đoạn ngắn**: đo CER theo bucket thời lượng **đoạn commit** (không chỉ
   theo độ dài clip). Nếu cohere sụp ở đoạn ≤2s, phải tăng `silence_duration_ms`/`min_transcribe_sec`
   hoặc loại cohere.
6. **Cân nhắc lại lựa chọn model**: cohere thắng về trần (5,13% so với 5,29%) nhưng thua về
   RTF (5×), CPU (3,2×), độ trễ commit (2–3×) và độ bền đoạn ngắn. Với sản phẩm phụ đề realtime,
   `qwen3-asr-1.7b` vẫn là lựa chọn an toàn hơn cho tới khi (5) được đo.

---

## 8. BẮT BUỘC GPU — đã triển khai, build lại native và kiểm chứng

Theo yêu cầu: **xoá fallback CPU ngầm, bắt buộc chạy GPU, không có GPU thì báo lỗi.**
Đã làm xong ở **hai tầng**, có build lại native, và **đã kiểm chứng bằng đo thực**.

### 8.1. Vì sao không thể "xoá" thẳng fallback

Ba ràng buộc đã kiểm chứng bằng thực nghiệm, buộc phải làm ở tầng **chính sách** chứ không
phải bằng cách bỏ backend CPU:

1. `ggml/src/ggml-backend.cpp:1789` — `ggml_backend_sched_new` **assert backend CUỐI CÙNG phải là CPU**:
   ```cpp
   GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);
   ```
2. Dòng 1311 ghi rõ hệ quả nếu vi phạm: `// all nodes should be assigned by now, this can happen if there is no CPU fallback`.
3. **Đã thử nghiệm**: copy `bin/` sang bundle mới và **xoá `ggml-cpu.dll`** ⇒ **cả `qwen3-asr-1.7b` (100% GPU) cũng không nạp được**:
   ```
   NẠP LỖI: BackendError: loading model '...Qwen3-ASR-1.7B-Q8_0.gguf': backend error (status 8)
   ```
   (script: `scratch/no_cpu_fallback_test.py`)

⇒ Backend CPU là **yêu cầu cấu trúc** của ggml scheduler, không phải tuỳ chọn. Cách đúng là
**từ chối chạy** graph có node rơi về CPU, không phải bỏ backend.

### 8.2. Tầng 1 — Python: không còn fallback im lặng

| File | Thay đổi |
| --- | --- |
| `backend/config.py` | Thêm `ASRConfig.require_gpu: bool = True` kèm chú thích đầy đủ số đo |
| `backend/asr/native.py` | Thêm `GpuRequiredError`; `resolve_backend()` **NÉM LỖI** khi không có GPU khả dụng hoặc khi backend được chỉ định không có (trước đây chỉ WARNING rồi vẫn chạy tiếp); thêm `require_gpu_enabled()` / `apply_require_gpu_env()`; `bootstrap()` đặt `GGML_SCHED_REQUIRE_GPU` |

Escape hatch: `config.asr.require_gpu = False` **hoặc** env `TRANSCRIBE_REQUIRE_GPU=0`.
`config.asr.backend_fallback` từ nay **chỉ** điều khiển việc đổi giữa các backend GPU
(cuda ↔ vulkan), **không bao giờ** cho rơi xuống CPU.

Kiểm chứng (`scratch/verify_require_gpu.py`) — **9/9 PASS**:

| Ca | Kỳ vọng | Kết quả |
| --- | --- | --- |
| Máy có GPU | `resolve_backend("cuda") == "cuda"` | ✅ |
| `bootstrap()` | env `GGML_SCHED_REQUIRE_GPU=1` | ✅ |
| Giả lập không có GPU | ném `GpuRequiredError` | ✅ |
| `TRANSCRIBE_REQUIRE_GPU=0` | không ném, chạy tiếp | ✅ |
| `config.asr.require_gpu=False` | không ném, chạy tiếp | ✅ |
| Chỉ định `cuda` nhưng chỉ có `vulkan` | ném lỗi (không tự đổi) | ✅ |
| `auto` / `vulkan` bình thường | không hồi quy | ✅ |

### 8.3. Tầng 2 — Native: từ chối graph có op trên CPU

Thêm vào `ggml/src/ggml-backend.cpp`, hàm `ggml_backend_sched_require_gpu_check()`, gọi ngay
sau `ggml_backend_sched_split_graph()` trong `ggml_backend_sched_alloc_graph()` — **một điểm
chèn duy nhất phủ mọi family**, vì mọi arch đều gọi `alloc_graph`.

- Bật/tắt bằng env `GGML_SCHED_REQUIRE_GPU` (unset/`0` ⇒ hành vi upstream **không đổi**).
- Bỏ qua khi `sched->n_backends < 2` (kế hoạch CPU thuần — caller đã chủ động chọn CPU).
- Khi vi phạm: ghi **ERROR** liệt kê **tên op + số lượng**, rồi trả `false` ⇒ `alloc_graph`
  thất bại ⇒ lỗi nổi lên tận Python.

Patch lưu theo đúng quy ước repo: **`external/transcribe.cpp/patches/ggml/0002-require-gpu-no-cpu-fallback.patch`**
(cùng chỗ với `0001-fix-threadpool-oversubscription.patch`; `scripts/sync-ggml.sh` áp theo thứ tự tên).

### 8.4. Build lại native

Cây build **đã có sẵn** (`external/build-cuda`, `GGML_CUDA=ON`, `GGML_CUDA_GRAPHS=ON`, sm_120),
nhưng đường dẫn trong cache đã cũ vì thư mục được di chuyển:

| Vấn đề | Xử lý |
| --- | --- |
| Cache tạo ở `.build-cuda`, thư mục nay là `external/build-cuda` ⇒ CMake từ chối | Tạo **junction** `D:\...\.build-cuda` → `external/build-cuda` |
| Toolkit ở `.cuda-toolkit`, nay là `external/cuda-toolkit` | Junction `D:\...\.cuda-toolkit` → `external/cuda-toolkit` |
| Không có `cl`/`ninja` trên PATH | `vcvars64.bat` + `ninja.exe` trong `external/cuda-toolkit/ninja-1.13.2.data/scripts` |

Script: **`scratch/build_cuda.bat`**. Kết quả: `342/342`, sinh `ggml-base/ggml-cpu/ggml-cuda/ggml/transcribe.dll`.
(2 junction đã được thêm vào `.gitignore`.)

### 8.5. Kiểm chứng trên `bin/` THẬT

`bin/` đã được sao lưu vào `scratch/bin_backup_pre_require_gpu/` trước khi thay.

```
$ ggml_backend_sched_require_gpu_check: GGML_SCHED_REQUIRE_GPU=1 but 96/4729 node(s) were
  assigned to the CPU backend. Refusing to run a partially-CPU graph.
  Ops on CPU: FLASH_ATTN_EXT x48 SIGMOID x48.
  Set GGML_SCHED_REQUIRE_GPU=0 (or config.asr.require_gpu=False) to allow a partially-CPU graph,
  or fix the model port so these ops stay on the GPU.
```

| Model | Trước | Sau (require_gpu=1) |
| --- | --- | --- |
| `qwen3-asr-1.7b` | 100% GPU, RTF 0,020 | ✅ **không đổi** — `cores_busy=1,00`, RTF 0,0224, đầu ra y hệt |
| `cohere-transcribe` | lặng lẽ 2,5% node trên CPU, `cores_busy=3,39`, RTF 0,104 | ❌ **BỊ CHẶN**, nêu đúng `FLASH_ATTN_EXT x48 SIGMOID x48` |
| `cohere` khi `require_gpu=0` | — | ✅ chạy lại bình thường (escape hatch hoạt động) |

### 8.6. Kết quả test suite

**262 passed, 1 failed.** Lỗi duy nhất là
`test_20_logging_convention.py::test_every_logger_call_has_module_tag` (4 lời gọi logger thiếu
`module_tag` trong `backend/utils/model_download.py`) — **lỗi CÓ SẴN từ trước**, đã xác nhận
bằng `git status` rằng file đó **không** bị sửa trong thay đổi này.

### 8.7. Hệ quả cần biết

⚠️ **`cohere-transcribe` KHÔNG chạy được ở chế độ bắt buộc GPU.** Đây không phải lỗi của thay
đổi này — nó phơi ra một sự thật trước đây bị che: bản port cohere **không thể** đưa 48×
`FLASH_ATTN_EXT` + 48× `SIGMOID` lên CUDA (một cặp mỗi encoder block). Muốn dùng cohere, chọn
một trong ba:

1. Đặt `config.asr.require_gpu = False` (chấp nhận ~3,4 nhân CPU và RTF 0,104).
2. Sửa port cohere để encoder attention + sigmoid chạy trên CUDA (việc của upstream, không
   phải của backend).
3. **Dùng `qwen3-asr-1.7b`** — đã 100% GPU, không cần thay đổi gì.

### 8.8. Việc còn lại (đề xuất)

- **Nhãn lỗi gây nhầm**: `alloc_graph` trả `false` nên `transcribe_run` báo `OutOfMemory`
  (status 7), trong khi nguyên nhân thật là require-GPU. Dòng ERROR của native là nguồn
  chính xác, nhưng nên thêm một `ggml_status`/cờ riêng để tầng Python hiện đúng thông báo.
- Chạy `scripts/ci/clang-format.sh --check` trước khi upstream hoá patch (repo có gate CI).
- Cân nhắc đưa `require_gpu` vào `.env`/UI cấu hình để người dùng cuối thấy được.
