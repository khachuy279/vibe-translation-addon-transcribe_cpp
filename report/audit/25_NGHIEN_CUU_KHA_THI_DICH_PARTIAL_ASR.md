# 25 — NGHIÊN CỨU KHẢ THI: DỊCH PARTIAL ASR VÀ XUẤT BẢN DỊCH DẦN (hướng thay thế mirror-MSE)

> **Loại tài liệu:** nghiên cứu khả thi (feasibility study) + thiết kế + kế hoạch đo.
> **KHÔNG** sửa `backend/` hay `extension_firefox/`; chỉ đọc và ghi chính file này.
> **Ngày:** 2026-09 · **Tiền đề:** `22_MSE_INTERCEPT_FEASIBILITY.md`, `23_M1_KET_QUA_KHAO_SAT_MSE.md`,
> `24_M2_MIRROR_MSE_PCM.md` (M2 **đã chứng minh** nhưng **chưa tích hợp**).
> **Đề xuất được đánh giá (hướng B):** *partial text đang được ASR sinh ra → đưa thẳng vào model dịch
> → xuất bản dịch dần (streaming) ra phụ đề*, **không** chặn bắt MSE, **không** chạy trước playhead.
>
> **Nhãn bằng chứng:**
>
> | Nhãn | Nghĩa |
> | :--- | :--- |
> | `[VERIFIED-REPO]` | Đọc trực tiếp trong repo này, kèm `file:line`. |
> | `[ĐO]` | Số đo đã có trong repo (kèm nguồn). |
> | `[SUY LUẬN]` | Suy ra **từ** số đo — chưa đo trực tiếp, không dùng làm căn cứ duy nhất để quyết định. |
> | `[NOT VERIFIED]` | Chưa kiểm chứng — **phải đo** trước khi kết luận. |
> | `[KIẾN THỨC CHUNG]` | Kiến thức bên ngoài; phiên này `web_search` **lỗi HTTP 402 (hết số dư)** nên **không** đọc được nguồn — không dùng để chốt quyết định. |

---

## 0. KẾT LUẬN ĐIỀU HÀNH

1. ✅ **KHẢ THI — và rẻ hơn M2 rất nhiều về rủi ro tích hợp.** Toàn bộ đường ống cần thiết **đã có
   sẵn trong repo**: ASR đã phát partial (`is_final=False`) mỗi ~300 ms, engine dịch **đã** có
   `translate_stream()` (token dần), protocol **đã** có cờ `partial`, và client **đã** có
   `subtitle-policy.js` để quyết định hiện hay bỏ bản dịch dở. Việc phải làm **không** phải là xây
   cơ chế mới, mà là **nối ba đoạn dây đã có** + thêm một làn "draft" có kiểm soát tải.
   `[VERIFIED-REPO]`
2. ✅ **Lợi ích độ trễ là thật và đo được bằng số đã có.** Phụ đề dịch hiện nay chỉ xuất hiện **sau
   khi câu/mệnh đề đã chốt**: mệnh đề trung vị dài `3,21 s` (`asr.commit_audio_sec` p50) + chốt
   `199 ms` + dịch `405 ms` ⇒ **≈ 3,6 s** sau khi bắt đầu nói. Nếu dịch partial, chữ dịch đầu tiên
   xuất hiện ở **≈ 0,62 s** (`asr.first_preview_ms` p50 `493 ms` + `translation.first_token_ms`
   p50 `28 ms`; thực tế **≈ 0,7–1,0 s** sau khi trừ ngưỡng chờ preview đủ dài — xem §3.2).
   **Giảm ≈ 2,6–3,0 s cho mệnh đề thường, tới ≈ 15 s cho câu dài không có dấu câu.**
   `[ĐO]` + `[SUY LUẬN]`
3. ⚠️ **Nhưng đây KHÔNG phải "phiên bản rẻ hơn của M2" — nó giải một bài toán khác.** M2 nhắm
   *"~0 ms độ trễ cảm nhận"* (chữ có trước/đúng lúc tiếng phát). Hướng B có **sàn cứng ≈ 0,6–1,0 s**
   (capture 64 ms + VAD onset ~100 ms + preview đầu tiên 493 ms), vì nó **không thể** biết nội dung
   trước khi người ta nói. Với mục tiêu "< 1,0 s" đã ghi trong tài liệu 22 §1.4 thì hướng B **đạt**;
   với mục tiêu "chữ trước tiếng" thì **không**. `[ĐO]` + `[SUY LUẬN]`
4. ⚠️ **Hệ quả quan trọng: hướng B không phục vụ được lồng tiếng (TTS/dubbing).** TTS cần văn bản
   **ổn định** và phải phát **đúng mốc thời gian**; partial chưa ổn định nên tổng hợp từ partial là
   tổng hợp lại liên tục. Nếu mục tiêu cuối cùng **bao gồm** lồng tiếng thì hướng B là **bổ sung**,
   không phải thay thế M2. `[SUY LUẬN]`
5. 🔴 **Rủi ro số 1 — và nó nằm ở CHẤT LƯỢNG DỊCH, không ở kỹ thuật:** partial của ASR **không đơn
   điệu**. Chính repo này đã ghi lại bằng đo thật: *"cùng một đoạn audio, Qwen3-ASR có thể đổi cả nội
   dung giữa hai lần quét (`二つ` ↔ `2つ`, `貿易` ↔ `防衛`)"* (`backend/segmentation/boundary.py:242-252`).
   Nghĩa là bản dịch partial có thể **bị vô hiệu** ở nhịp sau. Model dịch đang dùng
   (Hunyuan-MT2 7B) là model **mức câu**; dịch **câu cụt** — đặc biệt tiếng Nhật/Trung có **động từ
   ở cuối** — có thể ra bản dịch **sai nghĩa**, không chỉ thiếu chữ. **Đây là thứ phải đo trước.** `[VERIFIED-REPO]` + `[NOT VERIFIED]`
6. 🔴 **Rủi ro số 2 — tải GPU và thứ tự ưu tiên.** Inference dịch chạy **tuần tự, một worker, một
   `_infer_lock`, và KHÔNG huỷ được** (`backend/translation/engine.py:39,49,483-532`), khác ASR có
   `cancel_inference()` (`backend/asr/engine.py:1896`). Nếu draft cứ chạy đều, **bản dịch FINAL sẽ
   xếp hàng sau draft** ⇒ đúng cái phụ đề quan trọng nhất lại chậm đi. Ước lượng tải: ASR preview đã
   chiếm **≈ 35 %** chu kỳ (105 ms/300 ms) + commit 3 % + dịch final 12 % ≈ **50 %**; draft ở nhịp
   1 Hz thêm **≈ 21 %**, ở nhịp 0,5 Hz thêm **≈ 41 %**. ⇒ **Nhịp draft phải ≥ 800–1000 ms và phải
   nhường final**, không được chạy đều tay. `[VERIFIED-REPO]` + `[SUY LUẬN]`
7. 🔴 **Rủi ro số 3 — nháy chữ (flicker).** Chữ dịch sẽ **đổi tại chỗ** khi câu hoàn tất. Đây là
   đánh đổi **nhận thức** (người đọc phải đọc lại), không phải lỗi kỹ thuật. Cách chặn chuẩn mực
   của ngành là **dịch trên "tiền tố ổn định"** (stable prefix / local agreement), và repo **đã có
   sẵn** khái niệm này: `SentenceCompleter.committed_text` + hai trường protocol `stable_text` /
   `unstable_text` (hiện đang **giả**: cả hai lần lượt bằng `preview_text` và `""`,
   `backend/asr/engine.py:1734-1735`). `[VERIFIED-REPO]` + `[KIẾN THỨC CHUNG]`
8. **Khuyến nghị:** **GO có điều kiện** — nhưng **đo trước, viết sau**, đúng bài học đã trả giá ở M2
   (fMP4 sống sót **5 phiên** chỉ vì *chưa từng có kiểm thử offline*; tài liệu 24 §5.11b). Bước đầu
   **không** sửa `backend/`/`extension_firefox/`: viết **harness offline** trên `wav_test/` (đã có
   FLEURS `ja_jp`, EN/ZH) chạy **ASR thật + dịch thật**, đo 8 chỉ số ở §7.2. Chỉ khi đạt ngưỡng
   GO/NO-GO mới động vào sản phẩm.
9. **M2 không bị phủ định.** `scratch/m2_mirror/` + tài liệu 22/23/24 giữ nguyên như **tài sản đã
   chứng minh** (3 host, 2 codec, 4 mặt xanh). Hướng B **không** xoá nhu cầu M2 nếu sau này cần
   lồng tiếng hoặc "chữ trước tiếng"; nó chỉ **hạ ưu tiên** M2.2 (tích hợp vào extension).
10. **Ước lượng công sức nếu GO:** 4 mốc — B1 harness offline (§7.2), B2 làn draft ở backend +
    protocol (§6.2–6.3), B3 client (renderer/policy/CSS, §6.4), B4 phiên thật 3 host + A/B chính
    sách (§7.4). Ba trong bốn mốc **có test/harness sẵn** để mở rộng
    (`test_08_streaming_latency.py`, `test_13_translation_and_protocol.py`,
    `backend/tests/js/subtitle_renderer_harness.js`, `test_14_vad_lock_and_client.py`). `[VERIFIED-REPO]`

---

## 1. ĐỀ XUẤT ĐƯỢC ĐÁNH GIÁ, VÀ NÓ KHÁC M2 Ở ĐÂU

### 1.1 Phát biểu chính xác

```
[ASR]  preview mỗi ~300 ms (đã có, is_final=false)
          │  text đang lớn dần của câu CHƯA chốt
          ▼
   [chọn "tiền tố ổn định"]  ← PHẦN PHẢI XÂY (LCP + biên từ + ngưỡng)
          │
          ▼
[MT]   dịch partial (Hunyuan-MT2)  ← ĐÃ CÓ translate()/translate_stream()
          │  yield token dần
          ▼
[WS]   translation{partial:true, draft:true}  ← ĐÃ CÓ partial, THÊM draft
          │
          ▼
[UI]   hiện chữ dịch DẦN ở TẦNG 3 (live)  ← ĐÃ CÓ 3 tầng, PHẢI thêm nhánh draft
          │
          └─ khi câu chốt: bản dịch FINAL thay thế tại chỗ (đường cũ, không đổi)
```

**Điểm mấu chốt:** đây là **đường phụ (chiếu thêm)**, không thay đường chính. Bản dịch final vẫn
chạy y như hiện nay và vẫn là bản dịch "thật". Draft chỉ là thứ **lấp khoảng trống** trước nó.

### 1.2 Hướng B so với M2 — cùng đích, khác bài toán

| Tiêu chí | **M2 (mirror-MSE + decode offline)** | **Hướng B (dịch partial)** |
| :--- | :--- | :--- |
| Kiến trúc | Hook `appendBuffer` ở MAIN world, shadow `MediaSource`, `decodeAudioData` theo lô | Không đụng trang; chỉ thêm làn dịch + hiển thị |
| Nguồn dữ liệu | **Byte media** của trang (trước khi phát) | **Text** ASR của chính audio đang phát |
| Trần độ trễ | **~\>0 ms** (chữ/từ có trước playhead) | **≈ 0,6–1,0 s** (sàn cứng: chưa nói thì chưa có text) |
| Độ phủ | Chỉ MSE không DRM (tài liệu 22 §0 bảng phán quyết) | **Mọi nguồn** đang capture được (kể cả DRM, live, `src=` trực tiếp) |
| Rủi ro với trang web | **Cao**: hook prototype, CSP theo trang (Q9/Q16), seek/ABR/quảng cáo | **Không có** — không chạm vào trang |
| Rủi ro chất lượng | Thấp (vẫn dịch **câu trọn vẹn**) | **Cao** (dịch **câu cụt**, text có thể bị viết lại) |
| Lồng tiếng (TTS) | ✅ Dùng được (chạy trước, đủ ngân sách) | ❌ Không dùng được cho TTS |
| Chi phí compute | Giải mã PCM lần hai (144–157× realtime, tài liệu 24 §1.11) | Dịch lại nhiều lần trên cùng một câu (O(N) lượt) |
| Trạng thái | ✅ Đã chứng minh (3 host, 2 codec), **chưa tích hợp** | ❌ Chưa có dòng code nào; **cơ chế đã có sẵn 70 %** |

### 1.3 Một sự thật cần nói rõ: hệ thống **đã** dịch dần ở mức MỆNH ĐỀ

Đây là chỗ dễ đánh giá sai hướng B. Hiện nay **không phải** "chờ hết câu dài mới thấy chữ dịch":
tầng SEG (`backend/segmentation/`) đã cắt câu theo **dấu câu** và mỗi mệnh đề cắt ra được **chốt +
dịch ngay**. Số đo phiên mẫu: `asr.commit_reason.VAD_SILENCE 30`, `seg.reason.punct_tail 14`,
`seg.timer_used 13`, `asr.commit_audio_sec` p50 **3,21 s**. `[ĐO]`

⇒ **Hướng B không rút ngắn "từ 10 s xuống 0,7 s" trong đa số trường hợp** — nó rút ngắn **phần còn
lại giữa hai dấu câu**, tức **≈ 2,6–3,0 s** cho mệnh đề trung vị, và **tới ~15 s** cho câu dài
không có dấu câu (chạm trần `sentence.max_duration_sec = 15.0`). Đây vẫn là mức cải thiện lớn — và
là con số đúng để đưa vào quyết định. `[ĐO]` + `[SUY LUẬN]`

---

## 2. HIỆN TRẠNG ĐÃ KIỂM CHỨNG TRONG REPO

### 2.1 Bảy mảnh đã có sẵn (không phải xây mới)

| # | Mảnh | Bằng chứng |
| :-: | :--- | :--- |
| 1 | ASR phát **partial** mỗi nhịp preview, `is_final=false`, kèm `utterance_id` | `backend/asr/engine.py:1729-1738`; nhịp `asr.effective_poll_ms = 300` `[ĐO]` |
| 2 | **Partial và final dùng CHUNG một `utterance_id`** ⇒ có thể thay chữ tại chỗ | `engine.py:1731` (preview) và `engine.py:1767-1774` (commit cùng id, rồi mới xoay id mới) `[VERIFIED-REPO]` |
| 3 | Engine dịch có **`translate_stream()`** yield token dần + bỏ partial cũ khi consumer chậm | `backend/translation/engine.py:436-532`; bật mặc định `translation.stream_tokens=True` `backend/config.py:401` `[VERIFIED-REPO]` |
| 4 | Handler **đã** phát bản dịch dần với cờ `partial=True` + throttle kép (ký tự mới + thời gian) | `backend/ws/handler.py:786-825`; `translation.partial_min_chars=3`, `partial_min_interval_ms=120` `config.py:404-405` |
| 5 | Protocol **đã** có `partial`, `stable_text`, `unstable_text`, và payload gọn v3 | `backend/ws/serializers.py:36-111`; `test_18_compact_payload.py` |
| 6 | Client **đã** phân biệt bản dịch dở và có **công tắc "Tắt chạy chữ"** | `extension_firefox/lib/subtitle-policy.js:13-36`; `popup.js:139-140,852-854`; policy áp ở `subtitle-renderer.js:322-333` |
| 7 | Client **đã** có 3 tầng phụ đề (live/chờ/cũ) + handler test DOM giả lập | `subtitle-renderer.js:613-636` (Tầng 3 live), `:639-656` (Tầng 2 chờ dịch), harness `backend/tests/js/subtitle_renderer_harness.js` |

### 2.2 Sáu lỗ hổng phải bịt (đây là "công việc thật" của hướng B)

| # | Lỗ hổng | Vì sao là vấn đề | Bằng chứng |
| :-: | :--- | :--- | :--- |
| L1 | **Chỉ câu FINAL vào `translation_queue`** | Partial đi thẳng ra client, **không** qua dịch ⇒ chưa có gì để hiện | `backend/ws/handler.py:542` (`if is_final and text and session.translation_queue`) `[VERIFIED-REPO]` |
| L2 | **Renderer không có nhánh "draft"** | Bản dịch của một id đang là draft **không** khớp `pendingFocus` (`:341`) và **không** khớp `completedSentences` (`:350`) ⇒ rơi vào nhánh **tạo mới** ở `:365-394`: câu **chưa nói xong** bị đẩy lên **Tầng 1** với `isFinal=true` và đồng hồ `expireAt` bắt đầu chạy — sai vòng đời | `subtitle-renderer.js:341-412` `[VERIFIED-REPO]` |
| L3 | **`onUtteranceUpdate` ghi đè `currentDraft` mỗi nhịp** | Mỗi preview tạo object mới `{id, originalText, isFinal:false}` ⇒ **mất** `translatedText` vừa có nếu không sửa | `subtitle-renderer.js:296-299` `[VERIFIED-REPO]` |
| L4 | **`stable_text`/`unstable_text` đang là dữ liệu giả** | ASR gán `stable_text = preview_text`, `unstable_text = ""` ⇒ **không có** khái niệm "tiền tố ổn định" để dịch; client cũng chưa dùng 2 trường này | `engine.py:1734-1735`; không có kết quả `stableText` trong `extension_firefox/` (grep) `[VERIFIED-REPO]` |
| L5 | **Không huỷ được inference dịch đang chạy** | Draft đang suy luận ⇒ final **phải chờ hết** `translation.infer_ms` (p50 405 ms, **p95 1330 ms**) trước khi bắt đầu. ASR có `cancel_inference()` (`asr/engine.py:1896`), đường dịch **không có** gì tương đương | `translation/engine.py:39` (`max_workers=1`), `:49` (`_infer_lock`), `:323` (chỉ `shutdown(cancel_futures=True)` khi unload) `[VERIFIED-REPO]` |
| L6 | **Dedup + context chỉ đúng cho câu final** | `ctx_tracker.add()` và `dedup.remember_translation()` chạy sau khi dịch xong một câu (`handler.py:842-844`). Nếu draft đi qua cùng đường, **cửa sổ ngữ cảnh 3 câu** bị bơm bằng văn bản cụt và bộ nhớ đệm dedup bị nhiễm | `handler.py:842-844`; `translation/context.py`, `translation/dedup.py` `[VERIFIED-REPO]` |

### 2.3 Một phát hiện phụ quan trọng: ASR partial **không đơn điệu**

Đây là dữ liệu **quyết định** cho §5 R1–R2, và nó nằm ngay trong repo:

> `_resync()` — *"Đo bằng ASR thật: cùng một đoạn audio, Qwen3-ASR có thể đổi cả nội dung giữa hai
> lần quét (`二つ` ↔ `2つ`, `貿易` ↔ `防衛`)."* — `backend/segmentation/boundary.py:242-252`

Hệ quả kỹ thuật đã được code xử lý cho **việc cắt câu** (`_emitted_text` **đóng băng** tiền tố đã
chốt, `stale_text=True` khi phần đã chốt bị viết lại — `boundary.py:219-264`), nhưng **chưa** có gì
xử lý cho **việc dịch**. `[VERIFIED-REPO]`

⚠️ **Và ta đang thiếu chỉ số để biết tần suất việc này.** `seg.stale_text` chỉ đếm khi phần **đã
chốt** bị viết lại (phiên mẫu: **0 lần**), còn số lần viết lại **trong lòng câu đang chờ**
(`resync_count`, `boundary.py:263`) **không** được xuất ra metrics. ⇒ **B1 phải đo chỉ số này**
(§7.2, chỉ số I5). `[VERIFIED-REPO]` + `[NOT VERIFIED]`

---

## 3. NGÂN SÁCH ĐỘ TRỄ — SỐ ĐO ĐÃ CÓ

### 3.1 Từng chặng `[ĐO]`

Nguồn: `metrics_report.json` (phiên thật 284 s, `qwen3-asr-0.6b`, CUDA, `preview_window_sec=15`,
`poll_interval_ms=300`, 39 phụ đề đã giao) + `README.md` + `report/audit/22_...md:98,107`.

| Chặng | p50 | p95 | Nguồn |
| :--- | ---: | ---: | :--- |
| Capture (worklet gom 1024 mẫu) | 64 ms | — | README:638 (K11) |
| VAD phát hiện onset | ~100 ms | — | tài liệu 22 §1.4 |
| **ASR preview đầu tiên** (từ lúc VAD báo bắt đầu nói) | **493 ms** | 676 ms | `asr.first_preview_ms` (min 323 / max 715) |
| Mỗi lượt preview (suy luận cửa sổ trượt) | 105 ms | 202 ms | `asr.preview_ms` (n=544) |
| Nhịp preview hiệu dụng | 300 ms | — | `asr.effective_poll_ms` |
| Dứt tiếng → chốt câu ASR | 199 ms | 470 ms | `asr.e2e_commit_ms` (n=30) |
| **Token dịch đầu tiên** | **28 ms** | 56 ms | `translation.first_token_ms` (n=39) |
| Dịch trọn câu | 405 ms | 1330 ms | `translation.infer_ms` (n=39, max 1485) |
| Chốt ASR → phụ đề dịch tới client | 406 ms | 1332 ms | `pipeline.e2e_asr_to_sub_ms` |
| Tốc độ dịch Hunyuan-MT2 7B | — | — | **67,4 token/s** (README:648) |
| Chiều dài mệnh đề được chốt | 3,21 s | 11,25 s | `asr.commit_audio_sec` (n=44, max 13,64) |

### 3.2 So sánh hai đường (mệnh đề dài `L` giây)

Đường hiện tại (chỉ final): chữ dịch hiện ở `L + 0,2 s (im lặng VAD) + 0,2 s (chốt) + 0,03 s (token
đầu)`.
Đường B (draft): chữ dịch hiện ở `0,1 s (VAD onset) + 0,49 s (preview đầu) + 0,03 s (token đầu)`
`≈ 0,62 s`.

| `L` (mệnh đề) | Đường hiện tại (p50) | Đường B (p50) | **Lợi** | Ghi chú |
| ---: | ---: | ---: | ---: | :--- |
| 3,21 s (p50 thực đo) | **≈ 3,6 s** | ≈ 0,62 s | **≈ 2,9 s** | Trường hợp phổ biến nhất |
| 5 s | ≈ 5,4 s | ≈ 0,62 s | ≈ 4,8 s | |
| 10 s | ≈ 10,4 s | ≈ 0,62 s | ≈ 9,8 s | Câu nói liên tục không dấu câu |
| 15 s (trần cấu hình) | ≈ 15,4 s | ≈ 0,62 s | ≈ 14,8 s | |

> **Lưu ý riêng cho mệnh đề bị cắt bởi DẤU CÂU (SEG_PUNCT, 14 lần trong phiên mẫu):** lúc đó người
> nói **chưa dừng**, nên không có 0,2 s chờ im lặng ⇒ đường hiện tại chỉ là `L + 0,23 s` và **lợi ích
> của draft nhỏ hơn** (≈ 0,4 s cho `L = 3,21 s`). Vì vậy giá trị thật của hướng B tỉ lệ với **độ dài
> mệnh đề**: càng dài càng lợi — và đó là cơ sở cho tuỳ chọn "chỉ draft câu ≥ N giây" ở B5/B6. `[SUY LUẬN]`

⚠️ Ba điều chỉnh làm con số đường B **thực tế xấu hơn 0,62 s**, và **phải** đưa vào kế hoạch đo:

1. **Draft phải chờ preview đủ dài** để dịch có nghĩa (`min_transcribe_sec = 0.35 s`, cộng ngưỡng
   ký tự tối thiểu) ⇒ thực tế **≈ 0,7–1,0 s**.
2. **Draft phải nhường** nếu đang có commit/final chạy (`_infer_lock`, §5 R4) ⇒ đôi khi trễ thêm
   vài trăm ms.
3. **Bản dịch đầu tiên của một câu cụt thường ngắn và sẽ bị thay** ⇒ "chữ dịch đầu tiên" chưa phải
   "chữ dịch đúng". Đây chính là lý do có §5.

---

## 4. CHI PHÍ COMPUTE — ƯỚC LƯỢNG TỪ SỐ ĐO

### 4.1 Mức tải hiện tại của một GPU `[SUY LUẬN]`

| Đường | Chi phí/lượt | Tần suất | Duty |
| :--- | ---: | ---: | ---: |
| ASR preview | 105 ms | mỗi 300 ms | **≈ 35 %** |
| ASR commit | 103 ms | mỗi ~3,2 s audio | ≈ 3 % |
| Dịch FINAL | 405 ms | mỗi ~3,2 s | ≈ 12 % |
| | | | **≈ 50 %** |

### 4.2 Thêm làn draft

Giả định: câu dài `L = 10 s`, mỗi draft dịch **tiền tố trung bình nửa câu** ≈ 12 token VI
(`0,18 s` sinh token ở 67,4 token/s + `0,028 s` TTFT `≈ 0,21 s/draft`). `[SUY LUẬN]`

| Nhịp draft | Số draft / 10 s | GPU thêm | Tổng duty |
| ---: | ---: | ---: | ---: |
| 500 ms | 20 | ≈ 41 % | **≈ 91 %** ← quá sát trần, dễ bỏ nhịp preview |
| **1000 ms** | 10 | **≈ 21 %** | **≈ 71 %** ← vùng khả thi |
| 1500 ms | 7 | ≈ 15 % | ≈ 65 % |

⇒ **Thiết kế mặc định nên là ~1000 ms + "chỉ khi có chữ mới đủ dài" + "nhường final"**, không phải
"dịch mỗi nhịp preview". Đây cũng là cách ASR đã tự xử lý cho preview của chính nó
(`preview_adaptive_backoff`, `asr.preview_yielded_to_commit` — `config.py:244-250`,
`engine.py:1653-1661`): **bắt chước đúng khuôn mẫu đã có, không phát minh cơ chế mới.** `[VERIFIED-REPO]`

### 4.3 VRAM

Đỉnh hiện tại **9,5 GB / 16 GB** (README:634, K7 — ASR + dịch 7B + TTS). Nếu dùng **model 1.8B cho
draft** (`tencent-1.8b`, `backend/translation_models.yaml`): +~1,1–2,0 GB model (Q4/Q8) + ~33 MB KV
(`config.py:390-394`) ⇒ **≈ 11,5 GB**, vẫn dưới trần 14 GB. `[SUY LUẬN]`

> **Ý tưởng thiết kế đáng thử:** draft bằng **model nhỏ** (1.8B, nhanh) và final bằng **model lớn**
> (7B, chất lượng) — đúng triết lý đã có của dự án: *"model dịch lớn được chọn có chủ đích để bảo
> đảm chất lượng dịch; không đánh đổi chất lượng lấy độ trễ"* (`23_M1...md:302`). Draft **không**
> đánh đổi chất lượng của final; nó chỉ là lớp tạm. `[SUY LUẬN]`

---

## 5. RỦI RO CHẤT LƯỢNG VÀ CÁCH CHẶN

| # | Rủi ro | Bằng chứng / mức độ | Cách chặn đề xuất |
| :-: | :--- | :--- | :--- |
| **R1** | **ASR viết lại text đã hiện** ⇒ bản dịch draft bị vô hiệu, hai bản dịch mâu thuẫn nhau | `boundary.py:242-252` ghi **đo thật** (`二つ`↔`2つ`, `貿易`↔`防衛`) `[VERIFIED-REPO]`. Tần suất **chưa đo** `[NOT VERIFIED]` | Chỉ dịch **tiền tố ổn định** (LCP qua N nhịp liên tiếp, cắt ở biên từ); **đóng băng** draft đã phát cho phần đã ổn định |
| **R2** | **Model dịch mức câu gặp câu cụt** ⇒ có thể **sai nghĩa**, không chỉ thiếu chữ (JA/ZH có động từ ở cuối) | Hunyuan-MT2 7B, prompt "Translate the following …" (`translation/prompts.py:47-71`); chưa có số nào về chất lượng trên câu cụt `[NOT VERIFIED]` | Đo trên FLEURS `ja_jp` + EN/ZH (§7.2 I6): cắt tiền tố ở 25/50/75 % và chấm bằng người/chỉ số; nếu sai nghĩa > 25 % ⇒ **chỉ draft cho mệnh đề ≥ N giây** hoặc **bỏ hướng** |
| **R3** | **Nháy chữ** khi draft → final | Hệ quả tất yếu của R1/R2 `[SUY LUẬN]` | Trình bày draft **khác kiểu** (mờ/nghiêng) và **giữ draft đến khi final tới** (không để nhấp nháy "・・・"); có công tắc riêng trong popup |
| **R4** | **Draft ăn mất ưu tiên của final** (không huỷ được inference) | `translation/engine.py:39,49`; không có API huỷ (L5) `[VERIFIED-REPO]` | **Không xếp draft vào `translation_queue`.** Dùng làn riêng `maxsize=1` (latest-wins) + **cổng**: nếu `translation_queue` không rỗng ⇒ **bỏ draft nhịp đó**; đếm `translation.draft_yielded_to_final` |
| **R5** | **Nhiễm dedup / cửa sổ ngữ cảnh** bằng văn bản cụt | `handler.py:842-844` `[VERIFIED-REPO]` | Draft **tuyệt đối không** đi qua `ctx_tracker.add()` / `dedup.remember_translation()`; chỉ final mới ghi |
| **R6** | **Tranh chấp GPU với ASR preview** làm cả hai xấu đi | Duty ≈ 50 % hiện tại, preview p95 202 ms, đôi khi spike ~2,5 s (README:689) `[ĐO]` | Nhịp draft thích ứng theo `asr.preview_ms` gần đây (giống `_effective_poll_interval`, `engine.py:1374-1400`); ngưỡng NO-GO ở §7.3 (G3/G4) |
| **R7** | **Vòng đời phụ đề**: draft bị đẩy lên Tầng 1 sớm, đồng hồ `expireAt` chạy khi câu chưa nói xong | `subtitle-renderer.js:341-412` `[VERIFIED-REPO]` | Nhánh draft **chỉ** cập nhật `currentDraft.translatedText` và render ở **Tầng 3**; không đụng `pendingFocus`/`completedSentences` |
| **R8** | **Tua video / đổi tab** khi draft đang bay | Đã có `stream_reset` + `_discard_queued` cho final (`handler.py:565-605,150-178`) | Làn draft phải được **xoá cùng** `stream_reset`; draft cũ đến muộn bị bỏ theo `utterance_id` không còn tồn tại |

---

## 6. THIẾT KẾ ĐỀ XUẤT (nếu GO)

### 6.1 Nguyên tắc

1. **Không chạm đường final.** Draft là làn phụ, tắt được bằng một cờ cấu hình, xoá được mà không
   ảnh hưởng gì tới final.
2. **Không chạm ASR** ở vòng đầu: tính tiền tố ổn định **ở phía consumer** (handler), không sửa
   `asr/engine.py` ⇒ WER/CER của ASR không thể hồi quy vì việc này.
3. **Mọi nhánh thoát phải tự đếm** — nguyên tắc đã trả giá ở M2 (tài liệu 24 §5.10, "0 lỗi có thể
   chỉ là *không có việc gì được làm*").
4. **Đo trước, viết sau** (tài liệu 24 §5.11b: lỗi 4 byte sống sót 5 phiên vì fMP4 chưa từng có
   kiểm thử offline).

### 6.2 Backend

| # | Việc | Chi tiết |
| :-: | :--- | :--- |
| B-a | Làn draft riêng | `session.draft_queue = asyncio.Queue(maxsize=1)`; `put_nowait` + thay thế mục cũ ⇒ **latest-wins**, trần RAM hằng số. **Tuyệt đối không dùng chung `translation_queue`**: hàng đợi đó `maxsize=32` và khi ĐẦY thì **GỘP văn bản** vào câu mới nhất (`session.py:213-215`; `handler.py:192-275`) ⇒ draft lọt vào đó sẽ **trộn chữ cụt vào câu final** |
| B-b | Bộ theo dõi tiền tố ổn định | Trong `_stream_asr_tokens`: giữ `last_text[utt_id]`; `stable = LCP(text, last_text)` cắt về biên từ; chỉ phát khi `stable` dài hơn lần trước ≥ `draft_min_new_chars` **và** đã qua `draft_min_interval_ms` |
| B-c | Cổng nhường final | Trước khi lấy draft: `if not session.translation_queue.empty(): skip + counter` |
| B-d | Điền hai trường đang giả | `make_utterance_update_msg(stable_text=stable, unstable_text=text[len(stable):])` — sửa luôn L4 và cho client tô mờ phần đuôi |
| B-e | Config mới (mặc định TẮT ở vòng đầu) | `translation.draft_enabled=False`, `draft_min_interval_ms=1000`, `draft_min_new_chars=8`, `draft_min_words=3`, `draft_model=""` (trống = dùng model final), `draft_sentence_min_sec=0.0` |
| B-f | Metrics mới | `translation.draft_sent`, `draft_infer_ms`, `draft_first_char_ms`, `draft_yielded_to_final`, `draft_superseded`, `draft_invalidated`, `draft_skipped_short`; và **xuất** `seg.resync_count` (L: hiện không có) |

### 6.3 Protocol — **thêm `draft`, KHÔNG tái dùng `partial`**

`partial` hiện có **nghĩa đã ổn định và đã có test**: *"mọi partial là TIỀN TỐ của bản cuối (không
nhảy chữ, không lùi)"* — `backend/tests/test_13_translation_and_protocol.py:77-86`. Draft **vi phạm**
bất biến đó một cách hợp pháp (câu cụt bị viết lại). ⇒ Gói draft:

```json
{"type":"translation","sentence_id":"ab12cd34","status":"ok",
 "translated":"...","partial":true,"draft":true,"target_lang":"vi"}
```

* `partial: true` **cố ý** giữ ⇒ **tương thích ngược tự nhiên**: client cũ có "Tắt chạy chữ" (mặc
  định BẬT) sẽ **bỏ qua** draft và chỉ hiện final — đúng hành vi cũ, không cần cập nhật extension.
* Client cũ đã tắt "Tắt chạy chữ" sẽ hiện draft như chữ đang chạy — chấp nhận được, ghi vào ghi chú
  phát hành.
* Bất biến mới cần test: **final luôn tới sau draft cùng id và luôn thay thế draft.**

### 6.4 Client

| # | Việc | Chi tiết |
| :-: | :--- | :--- |
| C-a | `subtitle-policy.js` | Thêm `isDraftTranslation(payload)`; thêm tuỳ chọn `showDraftTranslations` (mặc định **BẬT** ở mốc B3, đo A/B ở B4). Giữ nguyên `shouldApplyTranslation` cho đường cũ |
| C-b | `subtitle-renderer.js::onTranslation` | **Nhánh đầu tiên**: nếu `payload.draft && this.currentDraft?.id === id` ⇒ gán `currentDraft.translatedText` + `_renderLiveLayer()` + `return` (bịt L2/R7) |
| C-c | `subtitle-renderer.js::onUtteranceUpdate` | Giữ `translatedText` khi tạo lại `currentDraft` cùng id (bịt L3) |
| C-d | `subtitle-renderer.js::_renderLiveLayer` | Thêm `div.bs-translated.bs-draft` (mờ/nghiêng) dưới chữ gốc; tô mờ `unstableText` nếu có |
| C-e | Bàn giao draft → final **không nhấp nháy** | Khi `is_final=true` tới: **giữ** draft đang hiện cho tới khi (a) bản dịch final tới, hoặc (b) hết `draft_hold_ms` (đề xuất 600 ms) ⇒ không để lộ "・・・" sau khi đã có chữ |
| C-f | Popup + CSS | Công tắc "Dịch dần khi đang nói" (cạnh "Tắt chạy chữ"), đúng khuôn `chkTranslationOnce` (`popup.js:139-140,852-854,1057`) |
| C-g | Test | Mở rộng `backend/tests/js/subtitle_renderer_harness.js` (draft→final tại chỗ, draft của id khác, `stream_reset` khi đang có draft) + `test_14_vad_lock_and_client.py` (policy) + test protocol mới cho `draft` |

---

## 7. KẾ HOẠCH ĐO — "ĐO TRƯỚC, VIẾT SAU"

### 7.1 Nguyên tắc và thứ tự

| Mốc | Việc | Chạm sản phẩm? | Vì sao ở bước này |
| :--- | :--- | :--- | :--- |
| **B0** | Chốt ngưỡng GO/NO-GO (§7.3) **trước** khi viết dòng code nào | — | Tránh "đo xong rồi mới biện luận" |
| **B1** | **Harness offline** trên `wav_test/` (§7.2) | ❌ Không | Đúng bài học M2: đường không có test offline thì lỗi sống sót qua mọi phiên đo |
| **B2** | Làn draft + protocol (§6.2–6.3), **mặc định TẮT** | ✅ backend | Chỉ sau khi B1 đạt |
| **B3** | Client (§6.4), vẫn TẮT mặc định | ✅ extension | |
| **B4** | A/B chính sách + **phiên thật 3 host** đã có số nền (`play.vlstream.net`, `www.xvideos.com`, `www.youtube.com`) | ✅ | So với số nền M1/M2 ⇒ biết ngay có hồi quy hay không |
| **B5** | (tuỳ kết quả) thử biến thể: model 1.8B cho draft; chỉ draft câu ≥ N giây; chỉ draft ở ranh giới mệnh đề | ✅ | Ba cách giảm tải/giảm nháy |

### 7.2 B1 — harness offline, chỉ số phải đo

Dựng trên khuôn `backend/tests/test_08_streaming_latency.py` (đã chạy **ASR thật, có pacing thời
gian thực**, không nạp translation ⇒ nay nạp thêm translation thật), dữ liệu:
`wav_test/google_fleurs/ja_jp/` (có `test.tsv` tham chiếu), `wav_test/Chinese_fast_speed_11s.wav`,
`wav_test/English_low_speech_quality_19s.wav`, `wav_test/English_multiple_kinds_of_noise_88s.wav`.

| # | Chỉ số | Vì sao quyết định |
| :-: | :--- | :--- |
| I1 | `draft.first_char_ms` (p50/p95) tính từ lúc VAD báo bắt đầu nói | Đây là **lợi ích thật** của hướng B |
| I2 | `draft.infer_ms` (p50/p95) và số token mỗi draft | Kiểm chứng §4.2 — nếu 1 draft đắt hơn 300 ms thì nhịp 1 Hz không đủ |
| I3 | Số draft / câu + tổng token draft / câu | Chi phí thật, thay cho suy luận |
| I4 | **`translation.infer_ms` của câu FINAL, A/B draft ON/OFF** | Chứng minh draft **không** làm chậm final (R4) |
| I5 | **Tỉ lệ nhịp ASR có `resync` (text bị viết lại trong lòng câu)**, và tỉ lệ nhịp có tiền tố ổn định **ngắn hơn** nhịp trước | Hiện `[NOT VERIFIED]` (§2.3); quyết định R1 |
| I6 | **Chất lượng dịch câu cụt**: cắt tiền tố 25 / 50 / 75 % và chấm (người + chỉ số) so với bản dịch câu trọn vẹn | Quyết định R2 — **rủi ro số 1** |
| I7 | **Độ "sống sót"**: tỉ lệ ký tự của draft cuối cùng còn trong final (tương đồng chuỗi) | Đo nháy chữ (R3) bằng số, không bằng cảm nhận |
| I8 | `asr.preview_ms` p95 và `asr.preview_deferred_inflight` (A/B) | Tranh chấp GPU (R6) |

### 7.3 Ngưỡng GO / NO-GO (chốt ở B0)

| # | Chỉ số | **GO** | **NO-GO** |
| :-: | :--- | :--- | :--- |
| G1 | `draft.first_char_ms` p50 | ≤ **1000 ms** | > 1500 ms |
| G2 | `draft.infer_ms` p95 | ≤ **400 ms** | > 800 ms |
| G3 | `translation.infer_ms` **final** p95 (A/B) | chênh ≤ **10 %** | > 25 % |
| G4 | `asr.preview_ms` p95 (A/B) | chênh ≤ **10 %** (sàn nhiễu) | > 25 % |
| G5 | Sống sót draft → final (I7) | ≥ **70 %** ký tự | < 50 % |
| G6 | Câu cụt **sai nghĩa rõ** (I6, người chấm ≥ 30 câu JA→VI) | ≤ **10 %** | > **25 %** |
| G7 | VRAM đỉnh | ≤ 14 GB (K7) | > 14 GB |
| G8 | WER/CER ASR (đường ASR **không** đổi) | trong sàn nhiễu | xấu hơn sàn |

> **G6 là ngưỡng quan trọng nhất.** Nếu dịch câu cụt cho ra bản dịch **sai nghĩa** ở > 25 % số câu,
> thì mọi lợi ích độ trễ đều vô nghĩa — *phụ đề nhanh mà sai thì tệ hơn phụ đề chậm mà đúng*.
> Khi đó phương án lùi là **chỉ draft câu ≥ N giây** (nơi lợi ích lớn nhất: B4 dùng chính số ở §3.2).

### 7.4 B4 — phiên thật (kế thừa bộ đo của M1/M2)

3 host đã có số nền: `play.vlstream.net` (fMP4/AAC), `www.xvideos.com` (fMP4/AAC),
`www.youtube.com` (WebM/Opus) — tài liệu 23 §1, 24 §1.11. Chạy extension với `draft_enabled` ON/OFF
trên **cùng video**, so:
`pipeline.e2e_asr_to_sub_ms` (final), `translation.first_token_ms`, `draft.*` mới,
`asr.preview_ms` p95, và **số phụ đề final mỗi phút** (đảm bảo không mất câu vì draft chiếm chỗ).

---

## 8. SO SÁNH QUYẾT ĐỊNH VÀ KHUYẾN NGHỊ

### 8.1 Nếu mục tiêu là **phụ đề dễ đọc, độ trễ thấp** ⇒ hướng B

* Đạt mục tiêu "< 1,0 s" (tài liệu 22 §1.4) mà **không** cần M2.
* Rủi ro tích hợp gần bằng 0: không hook trang, không CSP, không seek/ABR/quảng cáo, không DRM.
* Chấp nhận: chữ dịch **sẽ đổi** khi câu hoàn tất; sàn ~0,7–1,0 s; không dùng cho TTS.

### 8.2 Nếu mục tiêu là **lồng tiếng / chữ trước tiếng** ⇒ vẫn cần M2

* TTS cần text **ổn định** + phải phát **đúng mốc**; partial không dùng được.
* M2 đã có bằng chứng đầy đủ (3 host, 2 codec, `lead p50` 20,99 / 26,65 / 73,18 s; `min`
  14,79 / 19,68 / 7,29 s; lỗi 0; bỏ 0 — tài liệu 24 §1.11). Việc còn lại là **M2.2 tích hợp**
  (tài liệu 24 §0.13, §7). Hướng B **không** thay thế được phần này.

> **Khuyến nghị:** **park M2 (không xoá, không làm tiếp M2.2 lúc này) và đi hướng B qua B1 trước.**
> Lý do: B1 **không** chạm sản phẩm, chạy được trong vài giờ, và trả lời đúng câu hỏi đắt nhất
> (G6 — chất lượng dịch câu cụt). Nếu G6 trượt, hướng B chết **trước khi** ai đó viết một dòng code
> sản phẩm nào — đúng cách M2 đã **không** làm (và đã trả giá bằng 5 vòng hồi quy).
>
> **Nhưng có một điều kiện tiên quyết về sản phẩm cần người quyết định:** *"chữ dịch đổi tại chỗ khi
> câu nói xong" có chấp nhận được không?* Nếu câu trả lời là **không** (người dùng muốn chữ đứng
> yên), thì hướng B **không** khả thi về mặt sản phẩm dù mọi chỉ số kỹ thuật đều xanh — và khi đó
> M2 là con đường duy nhất còn lại.

### 8.3 Có thể làm **cả hai** (khuyến nghị dài hạn)

Draft (hướng B) cho **phụ đề**, M2 cho **lồng tiếng**. Hai cơ chế độc lập, không chồng lấn:
hướng B chỉ cần text ASR; M2 chỉ cần PCM. Tuy nhiên **không** nên làm song song — mỗi hướng đều
đụng vào cùng một GPU (xem §4) và cùng một ngân sách độ trễ.

---

## 9. GIỚI HẠN CỦA KẾT LUẬN NÀY

1. **Chưa đo** bất cứ thứ gì về hướng B: **không** có số nào về chất lượng dịch câu cụt, tần suất
   ASR viết lại text, hay tải GPU thật khi draft chạy. Toàn bộ §4 là `[SUY LUẬN]` từ số đo của các
   đường **khác**. `[NOT VERIFIED]`
2. **Mẫu số đo nhỏ và cũ:** `metrics_report.json` là **một phiên 284 s** (39 phụ đề, 30 commit) —
   đủ để dựng ngân sách, **không** đủ để kết luận phân bố. Các phiên trong tài liệu 22 §1.3 (1785 s)
   cho số **thấp hơn** (`infer_ms` p50 281 ms, `e2e_asr_to_sub` p50 282 ms) ⇒ độ trễ dịch dao động
   theo phiên; kế hoạch B1 phải tự đo lại, **không** dùng số ở §3 làm hằng số.
3. **Chưa kiểm chứng được nguồn ngoài:** `web_search` lỗi HTTP 402 trong phiên này. Các kết luận
   "cách chặn nháy chữ bằng tiền tố ổn định / local agreement là chuẩn mực của ngành" (§0.7, R1)
   dựa trên **[KIẾN THỨC CHUNG]** (kỹ thuật wait-k/prefix-to-prefix trong dịch đồng thời, và chính
   sách local-agreement dùng cho partial hypothesis), **không** đọc được nguồn để trích dẫn. Không
   dùng nhãn `[VERIFIED-EXT]` cho phần này.
4. **Chưa xét** các ngôn ngữ ngoài EN/ZH/JA (đặc biệt ngôn ngữ đảo trật tự mạnh hoặc không có dấu
   câu như tiếng Thái) — nơi "tiền tố ổn định" khó xác định hơn và `SentenceCompleter` cũng đã phải
   có nhánh riêng (`boundary.py:44-71`, `_is_cjk_char`).
5. **Chưa xét** tương tác với TTS đang bật: khi `tts_enabled`, draft **không** được đẩy vào
   `tts_queue` (§6.2 B-b không đụng TTS) — nhưng chưa đo ảnh hưởng của việc renderer đổi chữ liên
   tục lên `tts-player.js`.
6. **Chưa xét** chi phí CPU phía client khi cập nhật DOM mỗi ~1 s (thấp, nhưng chưa đo trên trang
   nặng).

---

## PHỤ LỤC A — BẢN ĐỒ BẰNG CHỨNG (`file:line`)

| Khẳng định | Nơi |
| :--- | :--- |
| ASR phát partial `is_final=false` mỗi nhịp | `backend/asr/engine.py:1729-1738` |
| Partial và final **chung** `utterance_id` | `backend/asr/engine.py:1731`, `:1767-1774` |
| `stable_text`/`unstable_text` đang là dữ liệu **giả** | `backend/asr/engine.py:1734-1735` |
| Chỉ **final** vào hàng đợi dịch | `backend/ws/handler.py:542-556` |
| Hàng đợi dịch `maxsize=32` + **gộp văn bản khi đầy** (lý do phải có làn riêng cho draft) | `backend/ws/session.py:213-215`; `backend/ws/handler.py:192-275` |
| Bản dịch dần + throttle kép + cờ `partial` | `backend/ws/handler.py:786-825`; `serializers.py:77-111` |
| `translate_stream` (yield token dần, bỏ partial cũ) | `backend/translation/engine.py:436-532` |
| Dịch: 1 worker, 1 `_infer_lock`, **không huỷ được** | `backend/translation/engine.py:39,49,483-532`; `backend/ws/handler.py:859-878` |
| ASR **có** huỷ được inference | `backend/asr/engine.py:1896` |
| Dedup/context chỉ ghi ở đường final | `backend/ws/handler.py:842-844` |
| ASR viết lại text giữa hai lần quét (đo thật) | `backend/segmentation/boundary.py:242-252` |
| Tiền tố đã chốt được đóng băng (`committed_text`) | `backend/segmentation/boundary.py:219-239`, `:466-477` |
| Client bỏ qua bản partial theo mặc định | `extension_firefox/lib/subtitle-policy.js:13-36`; `subtitle-renderer.js:322-333` |
| Renderer **không** có nhánh draft (đẩy lên Tầng 1) | `extension_firefox/lib/subtitle-renderer.js:341-412` |
| `currentDraft` bị ghi đè mỗi nhịp | `extension_firefox/lib/subtitle-renderer.js:296-299` |
| 3 tầng phụ đề | `extension_firefox/lib/subtitle-renderer.js:596-681` |
| Payload dịch được chuyển nguyên vẹn tới renderer | `extension_firefox/content/overlay-manager.js:448-453`; `content-script.js:391` |
| Bất biến "partial là tiền tố của final" (test đang giữ) | `backend/tests/test_13_translation_and_protocol.py:77-86` |
| Mẫu số ASR tự backoff khi chậm (khuôn để bắt chước) | `backend/config.py:244-250`; `asr/engine.py:1374-1400,1653-1661` |
| Model dịch: Hunyuan-MT2 7B (chính) / 1.8B (nhẹ) | `backend/translation_models.yaml` |
| Số đo độ trễ phiên thật | `metrics_report.json` (`stages.*`) |
| Mục tiêu K2/K3/K7 + tốc độ 67,4 token/s | `README.md:629,630,634,648` |

## PHỤ LỤC B — CHECKLIST NẾU GO (tóm tắt để giao việc)

- [ ] **B0** Chốt bảng ngưỡng §7.3 (đặc biệt G6) — *trước* khi viết code.
- [ ] **B1** Harness offline: ASR thật + dịch thật trên FLEURS `ja_jp` + EN/ZH, xuất I1–I8.
- [ ] **B2** Backend: `draft_queue` (maxsize=1), LCP + biên từ, cổng nhường final, config `draft_*`
      (mặc định TẮT), metrics `draft_*`, **xuất `seg.resync_count`**.
- [ ] **B3** Protocol: `draft:true` + `partial:true`; test bất biến "final luôn thay draft".
- [ ] **B4** Client: nhánh draft trong `onTranslation`, giữ `translatedText` khi cập nhật
      `currentDraft`, `bs-draft` CSS, bàn giao không nhấp nháy, công tắc popup, test JS.
- [ ] **B5** Phiên thật 3 host (A/B draft ON/OFF), so với số nền M1/M2.
- [ ] **B6** Nếu G6 trượt ⇒ thử "chỉ draft câu ≥ N giây" và/hoặc model 1.8B cho draft, **rồi mới**
      quyết định bỏ hẳn.
