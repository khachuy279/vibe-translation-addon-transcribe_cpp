# 18 — Đánh giá khả thi kế hoạch test tối ưu tiếng Nhật (FLEURS ja_jp) & thiết kế vòng test điều chỉnh

**Ngày:** 2026-09-20
**Phạm vi:** đường ASR tiếng Nhật của backend (`models.yaml` → 7 model), pipeline `Feed audio → VAD → ASR`, đối chiếu với số liệu công bố trong `external/transcribe.cpp/docs/models/*.md` và `external/transcribe.cpp/catalog/*.json`.
**Trạng thái:** đã đo thực (không phải ước lượng). Mọi con số dưới đây đều tái lập được bằng script trong `scratch/`.

---

## 0. Kết luận nhanh (TL;DR)

| Câu hỏi của bạn | Trả lời |
| --- | --- |
| Kế hoạch có khả thi không? | **Có, và rẻ hơn bạn nghĩ** — nhưng phần "vòng 2" đang đặt mục tiêu sai chỗ. |
| Vòng 1 với 10 audio có chọn được model không? | **Không đủ tin cậy.** Cỡ mẫu 10 cho CI95 ±4,0 điểm %, trong khi 3 model dẫn đầu chỉ cách nhau 0,5 điểm. Và chỉ tốn ~3 phút/model để chạy **cả 650 clip** ⇒ không có lý do gì dùng 10. |
| Vòng 2 "tiệm cận số công bố" có khả thi không? | **Đã đạt rồi — ngay hôm nay, chưa chỉnh gì cả.** Đo offline toàn bộ 650 clip: **CER 5,36%** so với công bố **5,29%** (CI95 4,81–5,80). Và **thông số VAD/ASR không thể cải thiện con số này**, vì đường offline không dùng VAD. |
| Vậy còn việc gì để tối ưu? | **Khoảng cách giữa pipeline (VAD→ASR) và trần offline** — đo được là **+8…+9 điểm %** trên mẫu thử. Đây mới là mục tiêu đúng của vòng 2. |
| Rủi ro lớn nhất | **Sàn nhiễu**: cùng một cấu hình chạy lại đã cho chênh 0,44 điểm (đo mới) và **3,34 điểm** (đã ghi trong `report/audit/14_wer_noise_floor.json`). Không đo sàn nhiễu thì mọi kết luận A/B đều vô giá trị. |

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

*Bằng chứng số học:* catalog ghi `err_pct=5.29%` với `sub+del+ins = 1162+321+245 = 1728` ⇒ suy ra tổng tham chiếu = 1728 / 0,0529 = **32.665 ký tự**. Bản địa của ta trên cột 3 = **32.648 ký tự**. Lệch **0,05%**. Nếu chọn sai cột, con số này sẽ lệch hàng nghìn ký tự.

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

- **CER ở độ dài đoạn sản xuất (1–8s sau VAD)**, không phải cả clip 13s. Thứ hạng có thể đảo — đặc biệt `whisper` auto-detect ngôn ngữ trên đoạn ngắn rất dễ sai.
- **RTF phải là RTF của PIPELINE, không phải 0,0212.** Pipeline còn chạy preview mỗi `poll_interval_ms=300ms` trên cửa sổ 8s, cộng VAD mỗi 25ms, cộng chuẩn hoá. RTF sản phẩm thực tế ước tính **~0,5–0,6** (preview đã chiếm ~0,17s/300ms ≈ 57% GPU). Con số 0,0212 chỉ là "trần tốc độ model".
- **VRAM** (2,1GB cho qwen 1.7B vs 3,2GB cho voxtral vs 0,9GB cho whisper).
- **Hành vi xấu**: `empty_hyp`, lặp vòng, hallucination trên đoạn im lặng/nhạc — FLEURS sạch nên **không** đo được cái này, phải thêm tập kiểm chứng chéo.

Tôi đã đo luôn một con số cho điểm này: chạy pipeline ở `speed=1.0`, `wall/audio = 1,11` (đã gồm 1,2s đuôi im lặng nhân tạo) ⇒ pipeline **theo kịp thời gian thực** với đầy đủ preview, không bỏ nhịp.

### 2.3. Vòng 2: mục tiêu đặt sai chỗ — **đây là góp ý quan trọng nhất**

Kế hoạch viết: *"thay đổi các thông số VAD ASR ... để giảm CER cho model được chọn. Mục tiêu là tiệm cận số liệu được báo cáo."*

Vấn đề: **con số được báo cáo (5,29%) là kết quả chạy OFFLINE 1 LƯỢT CHO CẢ CLIP, KHÔNG CÓ VAD.** Backend **không bao giờ** chạy như vậy — nó `VAD → cắt câu → commit từng đoạn`. Hệ quả:

1. **Thông số VAD không thể chạm tới con số 5,29%** — đường offline không đi qua VAD. Chỉnh `silence_duration_ms`/`threshold` sẽ **không đổi một ký tự nào** của CER offline.
2. **Con số đó ta đã đạt rồi** (5,36% đo ở §1.5). Vòng 2 như đang viết là **không có gì để tối ưu**.
3. Ngược lại, **CER của pipeline luôn cao hơn trần**, và đó mới là chất lượng sản phẩm thật.

**Đo được khoảng cách đó** (10 clip, `speed=1.0`, cùng model đã chọn):

| Cấu hình | CER offline (trần) | CER pipeline | **Khoảng cách** | commits/clip |
| --- | ---: | ---: | ---: | ---: |
| `base` (mặc định: silence=600, max_dur=8, overlap=250, tier234 BẬT) | 11,03% | **20,15%** | **+9,12** | 1,3 |
| `repeat` (y hệt base — đo sàn nhiễu) | 11,03% | 19,71% | +8,69 | 1,4 |
| `lang_ja` (`asr.language="ja"`) | 11,03% | 20,15% | +9,12 | 1,4 |
| `long_seg` (`max_duration_sec=20`, `boundary_overlap_ms=0`) | 11,03% | **19,18%** | **+8,16** | 1,2 |
| `vad_only` (`enable_tier234=False`) | 11,03% | **19,18%** | **+8,16** | 1,1 |

*(10 clip này là mẫu "xấu" — macro offline 11,03% trong khi cả 650 là 5,35%. Chỉ đọc phần **khoảng cách**, đừng đọc mức tuyệt đối.)*

Ba phát hiện từ bảng này:

- **Sàn nhiễu của pipeline = 0,44 điểm** (base 20,15 vs repeat 19,71). Hiệu ứng `long_seg` (+0,97 điểm) chỉ **vừa** vượt sàn ⇒ chưa kết luận được ở n=10, nhưng **hướng đúng**.
- `long_seg` đưa **2 clip về đúng bằng offline** (16,3s: 5,6%→1,9%; 14,7s: 9,0%→3,0%). ⇒ `max_duration_sec=8` + `boundary_overlap_ms=250` đúng là **nguồn mất mát chính** với câu dài kiểu FLEURS. Đây là knob hạng 1, không phải VAD threshold.
- `lang_ja` **không đổi gì** (20,15 vs 20,15). Với clip sạch như FLEURS, auto-detect đã đúng. Knob này vẫn đáng thử trên **đoạn ngắn/nhiễu**, nhưng đừng kỳ vọng trên tập này.

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

Nếu thiếu thời gian: bỏ 1b cho 4 model yếu, chỉ chạy top-3 của 1a.

**Quy tắc chọn model:**
```
chọn = argmin( CER_pipeline )   ràng buộc:
        p95(commit_ms) < poll_interval_ms (300ms)
        wall/audio (speed=1) < 1.0
        empty_hyp = 0, không lặp vòng
  tie-break: VRAM nhỏ hơn, rồi tới dung lượng GGUF
```
Báo cáo kèm **khoảng cách so với catalog** cho cả 7 model (tầng 1a) để biết model nào có "vấn đề môi trường" riêng.

### 3.3. Vòng 2 — Thu hẹp khoảng cách phân đoạn (2 tầng)

**Mục tiêu viết lại:** giảm `CER_pipeline − CER_offline` trên `JA100`, không phải giảm CER tuyệt đối về 5,29%.

- **Sàng lọc:** `JA30`, speed=1.0, **1 lần chạy/cấu hình** (~7’/cấu hình) → giữ top-3.
- **Xác nhận:** top-3 trên `JA100`, speed=1.0, **lặp ≥2 lần** (~45’/cấu hình) → chỉ nhận cấu hình thắng **lớn hơn sàn nhiễu đo được**.
- **Chốt:** cấu hình thắng trên **cả 650** ở speed=1.0 (~2,5 giờ) để có con số phát hành.

**Ngân sách hợp lý:** ~10–14 cấu hình sàng lọc + 3 xác nhận + 1 chốt ≈ **6–8 giờ GPU**. Chạy 1 job tuần tự, không song song (tranh chấp GPU + native dùng `_shared_session` cấp lớp).

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

| Mức | Điều kiện | Trạng thái hiện tại |
| --- | --- | --- |
| **Tái lập trần** | `CER_offline(650)` nằm trong CI95 catalog | ✅ **đạt** cho qwen3-1.7b (5,36% ∈ [4,81; 5,80]) |
| **Chấp nhận pipeline** | `CER_pipeline(650) ≤ 1,5 × CER_offline(650)` (tức ≤ ~8,0%) | ❓ chưa đo trên 650; mẫu 10 clip cho tỉ lệ 1,8× |
| **Tối ưu thành công** | `CER_pipeline` giảm ≥ 20% tương đối so với `base`, **và** khoảng cách giảm còn ≤ 3 điểm % | ❓ `long_seg` cho −5% (dưới ngưỡng, chưa kết luận ở n=10) |

---

## 4. Rủi ro & giảm thiểu

| Rủi ro | Bằng chứng | Giảm thiểu |
| --- | --- | --- |
| **Kết luận A/B do nhiễu** | `14_wer_noise_floor.json`: cùng cấu hình, `error_rate` 17,14% vs 20,48% ⇒ **sàn 3,34 điểm** | Luôn chạy cấu hình đối chứng lặp; chỉ nhận khi Δ > sàn. Dùng **so sánh cặp trên cùng clip**. |
| **Mẫu nhỏ gây sai lệch hệ thống** | Mẫu 10 clip cho macro 11,03% trong khi cả 650 là 5,35% | Dùng ≥100 clip có phân tầng, hoặc cả 650. |
| **FLEURS không đại diện sản phẩm** | Đọc sạch, 1 giọng, không nhạc nền, không trộn ngôn ngữ | Thêm tập kiểm chứng chéo nhỏ từ `wav_test/` sẵn có (`Japanese_5s`, `Chinese_noise_28s`, `English_multiple_kinds_of_noise_88s`, `00_ingress_stream`) để bắt hồi quy hallucination/lặp. |
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
| `scratch/power_analysis.py` | CI95 theo cỡ mẫu + công suất so sánh cặp |
| `scratch/show_pilot.py`, `show_pipeline.py` | Đọc JSON kết quả, in chẩn đoán từng clip |

Lệnh tái lập:

```powershell
$env:PYTHONPATH="D:\vibe-translation-addon-transcribe_cpp"
python scratch\fleurs_stats.py
python scratch\baseline_fleurs_ja.py qwen3-asr-1.7b          # ~6 phút, 650 clip x 2 cấu hình
python scratch\pipeline_fleurs_ja.py qwen3-asr-1.7b --limit 30 --speed 1.0
python scratch\power_analysis.py
```

Kết quả thô: `scratch/baseline_qwen3-asr-1.7b.jsonl`, `scratch/pipeline_qwen3-asr-1.7b.json`, log đã lọc `scratch/baseline_qwen17.clean.log`, `scratch/pipeline_qwen17.clean.log`.

> **Lưu ý:** các script này nằm trong `scratch/` để khảo sát. Khi chốt thiết kế, nên chuyển thành `backend/tests/test_41_fleurs_ja_bench.py` (không có hàm `test_*` để `pytest` mặc định vẫn nhanh) theo đúng quy ước của `test_08`/`test_09`.

---

## 6. Tóm lại — ba việc cần đổi trong kế hoạch của bạn

1. **Vòng 1: đổi 10 audio → 650 audio ở đường offline** (chỉ ~3 phút/model). 10 clip không phân biệt nổi 4,82% với 5,29%.
2. **Vòng 2: đổi mục tiêu** từ *"tiệm cận 5,29%"* (đã đạt, và VAD/ASR không chạm tới được) sang *"thu hẹp khoảng cách pipeline − offline"*, với `speed=1.0` bắt buộc và sàn nhiễu đo được.
3. **Thêm một tầng "kiểm chứng tái lập"** cho từng model: so `CER_offline` cục bộ với catalog. Với qwen3-1.7b nó **khớp tới 0,07 điểm**, nên đây là tầng rẻ và đáng tin — dùng nó làm mốc neo cho mọi kết luận sau.
