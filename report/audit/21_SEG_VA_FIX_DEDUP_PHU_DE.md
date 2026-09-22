# 21 — Sửa "câu dịch trùng lặp" + Thiết kế tầng SEG (VAD > ASR > SEG)

Ngày: 2026-02 · Phạm vi: `backend/ws/handler.py`, `backend/translation/dedup.py`,
`backend/segmentation/*` (mới), test tầng A.

---

## PHẦN I — Bug "Bỏ qua câu dịch trùng lặp" treo phụ đề

### Triệu chứng (log người dùng)
```
[ASR_COMMIT] [utt=948dbed2] [VAD_SILENCE] (infer=65.1ms): 'うん。'
[WS] [utt=948dbed2] Bỏ qua câu dịch trùng lặp: 'うん。'
```
Phụ đề hiện câu gốc + dấu "・・・" (đang dịch) và **treo vĩnh viễn** cho tới khi có câu
final kế tiếp.

### Chuỗi nhân quả (đã xác minh trong mã)
1. `handler._stream_asr_tokens()` gửi `utterance_update{is_final: true, translated: "..."}`
   cho **mọi** câu final (`handler.py`, dòng ~526).
2. `subtitle-renderer.js` đưa câu đó vào `pendingFocus` và vẽ chỉ báo `.bs-translating`
   ("・・・") — câu chỉ rời Layer 2 khi có gói `translation` (kể cả bản dịch rỗng).
3. `_process_translation_item()` phát hiện trùng ⇒ `return` **im lặng**, không gửi gì
   ⇒ chỉ báo "・・・" không bao giờ được gỡ.

### Cách sửa
* `backend/translation/dedup.py` — `TranslationDedupState` nay nhớ **bản dịch** theo cùng
  khoá chuẩn hoá (`remember_translation`/`cached_translation`), TTL như cũ (10 s).
* `backend/ws/handler.py`
  * tách `_deliver_translation()` (gửi `translation` + `utterance_update` + phủ câu bị gộp +
    metrics + TTS) dùng chung cho đường dịch mới **và** đường tái dùng;
  * `_clear_pending_translation()` — gửi `translation` với `translated=""`, `status="ok"`
    để client đưa câu xuống danh sách đã xong và **tắt "・・・"**, phụ đề gốc giữ nguyên;
  * câu trùng **có** cache ⇒ gửi lại bản dịch cũ (`translation.dedup_reused`), **không**
    gọi GGUF lần hai;
  * câu trùng **không** có cache (câu đầu lỗi) ⇒ `translation.ellipsis_cleared`;
  * `translated="..."` chỉ đặt khi **chắc chắn có** hàng đợi dịch (`will_translate`) — trước
    đây nếu `session.translation_queue` rỗng thì dấu "..." treo vĩnh viễn.

### Test
`backend/tests/test_13_translation_and_protocol.py`
* `test_translation_dedup_reuses_cached_translation` — không dịch lại **nhưng** phải có gói
  `translation` (nếu không "..." treo).
* `test_translation_dedup_without_cache_clears_ellipsis` — cache rỗng ⇒ gửi bản dịch rỗng
  `status="ok"` + counter `translation.ellipsis_cleared`.
* Test cũ `test_translation_dedup_skips_repeat` (khẳng định "không gửi message nào") đã bị
  thay — chính hành vi đó là bug.

---

## PHẦN II — Tầng SEG (tách câu hậu-ASR) cho tiếng Nhật

### 1. Ràng buộc kỹ thuật quan trọng nhất (đã khảo sát, có bằng chứng)

**Qwen3-ASR trong `transcribe.cpp` KHÔNG có timestamp.**
* `external/transcribe.cpp/src/arch/qwen3_asr/capabilities.cpp`: `max_timestamp_kind =
  TRANSCRIBE_TIMESTAMPS_NONE`; family không đọc `params->timestamps`.
* Gọi thử với model thật: `timestamps="segment"|"word"|"token"` ⇒ lỗi `status 12`
  (`UnsupportedRequest`); `"auto"` trả về `kind="none"`, 1 segment phủ toàn clip.
* `session.stream()` bị `NotImplementedByModel` — không có streaming gốc.
* Family `qwen3_forced_aligner` (word-level) **chưa được port** trong checkout này.
* `config.asr.request_timestamps = True` là **no-op** với Qwen3 (`report/audit/18`: đã gạch bỏ).

⇒ **Không thể** map "dấu câu" sang mốc thời gian. Muốn cắt audio đúng ranh giới câu phải
dùng **neo khoảng lặng VAD** và/hoặc **chồng lấn + cắt lại**.

Đo thêm bằng ASR thật (probe, xem §3): cùng một đoạn audio, Qwen3-ASR **đổi cả nội dung**
giữa hai lần quét (`二つ` ↔ `2つ`, `貿易` ↔ `防衛`, `たどって` ↔ `たたんで`). Nên **không**
được tin văn bản ASR tăng dần để chốt câu — chỉ dùng nó làm *tín hiệu quyết định*.

### 2. Kiến trúc đề xuất

```
VAD ──► ASR (quét dần trên vùng đang nói) ──► SEG ──► (chốt vùng + mốc cắt)
                       ▲                                    │
                       └──── mở vùng mới tại mốc cắt ◄───────┘
```

* `backend/segmentation/boundary.py` — **quyết định văn bản**:
  * `split_complete_sentences()` cho văn bản đã hoàn chỉnh;
  * `SentenceCompleter.observe()` cho preview tăng dần, chỉ chốt khi có **bằng chứng**:
    | reason | điều kiện | ý nghĩa |
    |---|---|---|
    | `punct_tail` | sau dấu kết câu đã có chữ của câu MỚI | mạnh nhất (0.95) |
    | `punct_stable` | dấu kết câu ở cuối preview, đứng yên ≥ 2 scan và ≥ 350 ms | 0.8 |
    | `max_chars` | quá dài mà không có dấu ⇒ cắt ở `、` gần nhất | bảo vệ khả năng đọc |
    | `flush` | VAD đóng vùng nói | chốt phần còn lại |
  * `、`/`,` **không** phải dấu kết câu; câu < `min_chars` được **gộp** với câu sau
    (tránh "え。" đứng riêng); số thập phân (`3.14`) và viết tắt Latin không bị cắt;
    ngoặc đóng `」` được nuốt vào câu.
  * Phần đã chốt **đóng băng** (không phát lại câu cũ ⇒ không sinh phụ đề trùng); nếu ASR
    viết lại phần đã chốt thì bật `stale_text=True` để engine biết phải chạy lại ASR.
* `backend/segmentation/segmenter.py` — **ghép với audio**:
  * `StreamingSegmenter` giữ lịch sử scan `(text, end_sample)` để khoanh vùng mẫu chứa
    ranh giới (`window`), rồi `choose_anchor()` chọn mốc cắt:
    1. **neo khoảng lặng**: khoảng lặng dài nhất ≥ `min_gap_ms` trong cửa sổ ⇒ cắt giữa
       khoảng lặng đó (`anchored=True`);
    2. **chồng lấn dự phòng**: không có khoảng lặng ⇒ lùi `fallback_overlap_ms` (600 ms)
       để vùng mới bao gồm phần ranh giới, tầng trim cắt phần lặp;
  * `turn_scorer` — hook cho model phát hiện kết thúc lượt (Namo Turn Detector…): điểm ≥
    ngưỡng ⇒ chốt dù thiếu dấu câu.

### 3. Bằng chứng đo trên ASR thật (`scratch/seg_ja_probe.py`)

Dữ liệu: `wav_test/google_fleurs/ja_jp`, ghép ~40–60 s (có hàm dựng trong
`backend/tests/fixtures/ja_fleurs.py`, chèn 500 ms lặng giữa các câu).

| Kịch bản | Kết quả |
|---|---|
| Cắt ở **cuối vùng quét**, không chồng lấn | câu sau **MẤT CHỮ ĐẦU**: `溶岩が浮上しやすく…` → `しやすくなっていました。` |
| Cắt có **chồng lấn 600 ms** | câu sau **đủ đầu**: `…が浮上しやすくなっていました。` |
| Ranh giới câu SEG chốt | 5/5 khớp vị trí dấu câu của `test.tsv` (câu 3, 5 khớp nguyên văn) |
| Khác biệt chữ còn lại | lỗi nhận dạng của ASR (`2つ`↔`二つ`, `たどって`↔`たたんで`, `溶岩`↔`膀胱癌`), **không** phải lỗi tách câu |

Chi phí: ASR ~65 ms/lần decode (CUDA), bước quét 1.5 s.

### 4. Tiêu chí nghiệm thu người dùng đặt ra — ĐẠT

`backend/tests/test_47_segmentation_ja.py::test_tach_cau_dung_dau_cau_fleurs_60s`
> "Chỉ cần tách câu đúng dấu câu trong `test.tsv` đã có thể xem như là đạt yêu cầu."

⇒ Nạp ~60 s transcript FLEURS **tăng dần từng 3 ký tự** (mô phỏng preview) rồi `flush`;
danh sách câu SEG chốt **bằng đúng** bản tham chiếu tách theo dấu câu `test.tsv`, và
không mất ký tự nội dung nào.

16 test trong `test_47`: dấu câu đổi theo scan, `punct_tail`/`punct_stable`, gộp câu ngắn,
số thập phân/viết tắt, cắt cưỡng bức, flush, ASR viết lại (không phát lại), chọn neo khoảng
lặng, chồng lấn dự phòng, cửa sổ ranh giới theo lịch sử scan, hook turn detector.

### 5. Đánh giá 3 ý tưởng của người dùng

| Ý tưởng | Đánh giá |
|---|---|
| **(1) Cắt theo dấu câu cuối câu của ASR** | **Đúng hướng và là lõi của SEG** — nhưng phải xử lý 3 điều: dấu câu đổi theo scan (⇒ cần `punct_tail`/`punct_stable`), câu vụn khi gặp `、`, và **không có timestamp** nên cần neo audio. Đã cài đặt + test. |
| **(2) Namo Turn Detector** | Hữu ích như **tín hiệu bổ sung** khi ASR không thả dấu câu (câu dài liền mạch). Đã dựng sẵn hook `turn_scorer` (model chưa tải). Lưu ý: model huấn luyện trên hội thoại người–người, phim là kịch bản ⇒ cần đo lại trước khi tin; và nó **không** giải quyết mốc cắt audio. |
| **(3) Ý tưởng khác** | Xem §6. |

### 6. Các phương án khác (khuyến nghị theo thứ tự)

1. **Neo khoảng lặng + chồng lấn** (đã cài): rẻ, không cần model thêm, giữ nguyên hợp đồng
   VAD/ASR đã kiểm chứng. Hạn chế: ranh giới nằm giữa câu nói liền mạch chỉ chính xác ± bước quét.
2. **Tinh chỉnh ranh giới bằng decode lại cửa sổ con** (không cần aligner): khi SEG quyết
   định đã hết câu, quét lại các cửa sổ `[t_prev, t_prev+250ms], +500ms…` cho tới khi văn bản
   xuất hiện chữ đầu của câu mới ⇒ ranh giới chính xác ~±250 ms với ~0.4 s GPU. Đây là cách
   "ăn gian timestamp" hợp lý nhất với Qwen3.
3. **Dịch nguyên vùng dài rồi tách hiển thị**: giữ vùng ASR dài (chất lượng dịch tốt hơn) và
   chỉ **chia phụ đề tiếng Việt** thành 2 dòng/2 câu theo dấu câu bản dịch. Rẻ, giải quyết
   trực tiếp "cắt muộn ⇒ dịch tốt nhưng phụ đề dài"; có thể chạy **song song** với SEG.
4. **Aligner ngoài** (`Qwen3-ForcedAligner` port vào `transcribe.cpp`, hoặc
   WhisperX/`ctc-forced-aligner` trên mảnh đã đóng): chính xác nhất, nhưng nặng và cần model mới.
5. **Model khôi phục dấu câu tiếng Nhật** trên văn bản ASR: chỉ giúp dấu câu ổn định hơn,
   **không** cho mốc audio.

### 8. Khảo sát timestamp toàn catalog transcribe.cpp (bổ sung sau câu hỏi của người dùng)

Câu hỏi: *"có model nào khác có timestamp, hay transcribe.cpp không hỗ trợ timestamp cho bất kỳ
model nào?"* — Công cụ: `scratch/asr_timestamp_probe.py` (đọc `src/arch/*/capabilities.cpp`
+ **chạy live** trên model có sẵn cục bộ, in cả `capabilities` thật và 4 mức
`timestamps=none|segment|word|token`).

**transcribe.cpp CÓ hỗ trợ timestamp — cho 6/19 family (41/72 model trong catalog):**

| Family | Mức tối đa | Model tiêu biểu | Tiếng Nhật |
|---|---|---|---|
| `whisper` | **SEGMENT** | whisper-large-v3-turbo (99 ngôn ngữ) | ✅ có |
| `granite` | **WORD** | granite-speech-4.1-2b, granite-4.0-1b-speech | ✅ có (catalog) |
| `parakeet` | **TOKEN** | nemotron-3.5-asr-streaming-0.6b (ja-JP), parakeet-* (en) | ✅ ja-JP |
| `medasr` | TOKEN | medasr (y khoa, en) | ❌ |
| `gigaam` | TOKEN | gigaam-v3-* (ru) | ❌ |
| `moss` | SEGMENT | moss-transcribe-diarize (en, zh) | ❌ |

Các family **NONE**: `qwen3_asr` (**model mặc định của app**), `sensevoice`, `cohere`,
`voxtral`, `voxtral_realtime`, `canary`, `canary_qwen`, `moonshine*`, `granite_nar`,
`funasr_nano`, `sortformer`.

**Kiểm chứng LIVE trên audio tiếng Nhật (FLEURS ghép 23.1 s, 2 câu tham chiếu):**

1. **whisper-large-v3-turbo (đã có sẵn 845 MB)** — `timestamps="segment"` ⇒
   `kind='segment'`, **3 segment có mốc ms thật**, cắt đúng ranh giới câu:
   ```
   [     0 ->  6940 ms] 技術決定論のほとんどの解釈は、2つの一般論を共有しています。
   [  6940 -> 13340 ms] 1つは技術の発展自体が文化的、政治的な影響を大きく超えた道をたたっていること、
   [ 13340 -> 22620 ms] もう1つは技術が社会的に条件付けられたものではなく、むしろ内在する社会に影響を与えることです。
   timestamps="word"/"token" -> lỗi status 12 (vượt mức SEGMENT)   |   decode 225 ms
   ```
   ⇒ **Chính xác thứ SEG cần**: văn bản + mốc thời gian để map dấu `。` sang mẫu audio.
2. **nemotron-3.5-asr-streaming-0.6b (đã có sẵn)** — `capabilities: token,
   streaming=True`; `timestamps="token"` với `language="ja-JP"` ⇒ **117 token, mốc từng ký tự**:
   ```
   token đầu: [(' 技', 1680, 1760), ('術', 2000, 2080), ('決', 2320, 2400), ('定', 2480, 2560), ...]
   ```
   (Chất lượng tiếng Nhật thấp hơn Qwen3/Whisper: `決定路`↔`決定論`, `成治的`↔`政治的`.)
   Lưu ý: phải truyền `language="ja-JP"`, `"ja"` bị lỗi *unsupported language (status 10)*,
   và family `parakeet` **bắt buộc** `FamilyExtension` (`NemotronOptions`).
3. **qwen3-asr** — xác nhận lại lần nữa: `segment/word/token` ⇒ status 12.

**Hệ quả cho thiết kế SEG** (thay đổi khuyến nghị trước đó):
* Đường "neo khoảng lặng + chồng lấn" vẫn cần cho **preview realtime** (Qwen3 không có mốc).
* Nhưng khi **đóng câu**, có thể chạy thêm một lượt model CÓ timestamp trên chính mảnh audio
  đó để lấy mốc cắt **chính xác ms** thay vì ±bước quét:
  * **Whisper-large-v3-turbo**: 1 lượt ~225 ms cho 23 s audio (RTF ≈ 0.01), trả segment
    ~theo câu ⇒ map trực tiếp ranh giới câu; đã có sẵn trong `backend/models/`.
  * **Nemotron token**: mốc từng ký tự (chính xác nhất) nhưng phải **khớp văn bản** với
    Qwen3 (căn chỉnh mờ theo ký tự) và chất lượng JA thấp hơn.
* Vì vậy kiến trúc tốt nhất hiện có: `VAD → ASR chính (Qwen3, text tốt) → SEG (quyết định)
  → ASR phụ có timestamp (Whisper/Nemotron) trên mảnh đã đóng để lấy MỐC CẮT → cắt & mở
  vùng mới`. Đây là "aligner" nhẹ, không cần port `qwen3_forced_aligner`.

### 9. ĐÃ NỐI VÀO PIPELINE (hướng B: Qwen3 nhận dạng + Whisper làm timer)

**Đã cài đặt:**

* `backend/asr/timer.py` — `WhisperTimer` (session RIÊNG, không đụng
  `TranscribeEngine._shared_session`) + phần căn chỉnh thuần: `build_char_timeline()`
  (rải ký tự của từng segment whisper lên `[t0,t1]`) và `locate_cut_ms()` (ghép văn bản
  Qwen3 ↔ Whisper bằng `difflib`, tra mốc ms của ký tự NGAY SAU ranh giới câu, trả
  `confidence` theo tỉ lệ ký tự khớp).
* `backend/config.py` — `SegmentationConfig` (`enabled`, `replace_stable_prefix`,
  `max_chars`, `min_chars`, `tail_min_chars`, `stable_ms/scans`, `fallback_overlap_ms`,
  `use_whisper_timer`, `whisper_model_key`, `timer_min_confidence`, `timer_max_audio_sec`).
* `backend/core/pipeline_events.py` — thêm `CommitReason.SEG_PUNCT`.
* `backend/asr/engine.py` — `_evaluate_seg()` (chạy trước BẬC 3; thay BẬC 3 khi
  `replace_stable_prefix`) và `_seg_cut_sample()` (ưu tiên mốc timer, kiểm tra mốc phải
  nằm trong vùng nói, có counter/metric `seg.*`).
* **Bug phát hiện kèm theo**: `backend/asr/adapters.py` gọi
  `transcribe_cpp.WhisperOptions()` — lớp này **không tồn tại** (tên thật là
  `WhisperRunOptions`); `AttributeError` bị `except` nuốt nên **mọi tuỳ chọn whisper âm
  thầm bị bỏ**. Đã sửa.

**Đo end-to-end thật** (`scratch/seg_timer_e2e.py`, Qwen3 + Whisper, FLEURS ghép 34.4 s):

| t quét | SEG chốt | mốc timer | mốc dự phòng | segment Whisper |
|---|---|---|---|---|
| 7.5 s | `技術決定論のほとんどの解釈は、二つの一般論を共有しています。` | **7.25 s** (0.40, 722 ms) | 6.90 s | 7.00 s |
| 15.0 s | `…たたんでいること。` | **13.23 s** (0.96, 155 ms) | 14.40 s | 13.00 s |
| 24.0 s | `…ことです。` | **20.77 s** (0.50, 145 ms) | 23.50 s | 20.00 s |
| 30.0 s | `地殻が薄いため、…あります。` | **28.52 s** (0.40, 123 ms) | 29.40 s | — |

⇒ Mốc cắt bám đúng ranh giới câu thật (±0.25 s) thay vì ±0.6–1.5 s của chồng lấn cố định;
chi phí ~120–720 ms/lần, trung bình ~150 ms.

**Test:** `backend/tests/test_48_seg_timer.py` — 17 test (căn chỉnh thuần; `WhisperTimer`
với session giả + khẳng định KHÔNG đụng session engine; nối engine: bật/tắt SEG, thay
BẬC 3, dùng mốc timer, timer vô lý ⇒ lùi dự phòng, thiếu model không vỡ). Hai test BẬC 3
trong `test_10` nay tắt SEG tường minh vì chúng kiểm tra riêng cơ chế cũ.

### 10. Vòng tinh chỉnh theo log PHIM THẬT (2026-09-22 21:31)

Người dùng chạy phim tiếng Anh (The Big Bang Theory) và gửi log. Ba việc đã làm:

**(1) Chống CẮT SỚM — `punct_tail` phải chờ tail đủ dài + đủ nhịp.**
Log cho thấy 2 ca thật:
```
[utt=64eda859] [SEG_PUNCT] 'Don't try and trick.'                     ← ASR thả dấu `.` SAI giữa câu
[utt=15e9d4fa] [SEG_PUNCT] 'Me into buying something I don't want.'   ← câu bị chẻ đôi
[utt=003652e1] Mảnh cắt quá ngắn (1 < 2 từ): 'Okay.'                  ← chốt khi tail mới 1 ký tự
```
Cách sửa: `SentenceCompleter` có **bộ đếm riêng cho tail** (`_tail_scans`), chỉ tính nhịp
CÓ tail; chốt `punct_tail` khi: `content_len(tail) ≥ tail_min_chars` (mặc định **4**) VÀ
tail đã xuất hiện ≥ `tail_scans` (**2**) nhịp VÀ đứng yên ≥ `tail_stable_ms` (**280 ms**).
Thêm chốt cho chữ Latin: tail dính liền chưa có khoảng trắng (`" Super"`) = đang viết dở
một từ ⇒ chưa chốt. Nhờ vậy ASR có thêm 1–2 nhịp để **tự sửa dấu câu sai** trước khi cắt.

**(2) Prewarm timer.** `TranscribeEngine.prewarm_seg_timer()` nạp whisper trong **luồng nền**
khi mở phiên (đo thật: **0.67 s**, metric `seg.timer_load_ms`, counter `seg.timer_prewarmed`)
⇒ câu đầu tiên không còn bị trễ ~1 s đúng lúc người dùng đang chờ phụ đề. Nếu file model
chưa có thì ghi counter `seg.timer_unavailable` và dùng đường chồng lấn.

**(3) Cấu hình runtime + công tắc A/B.** Đã nối hết chuỗi:
`popup.html` (công tắc `chkSegEnabled`, `chkSegTimer` + 3 thanh trượt) → `popup.js`
(`getSettings`/`liveUpdateSettings`/khôi phục từ storage) → `content-script.js` (`set_config`)
→ `backend/ws/session.py` (`SessionConfigPayload` + `apply_config`) →
`TranscribeEngine.update_seg_config()` (**dựng lại** máy trạng thái SEG nên đổi có hiệu lực
ngay) và `POST /api/config` (`SwitchModelRequest`) cho mọi phiên đang chạy; `GET /api/config`
trả thêm khối `seg` đầy đủ.

**(4) Bug phát hiện khi chạy tier A: chống trùng log là NO-OP.**
`backend/utils/logger.py` dùng khoá `(logger, level, self.format(record))` — chuỗi này chứa
`%(asctime)s` tới **mili-giây**, nên hai dòng "trùng" luôn khác nhau ⇒ cơ chế F-41 chống in
2 lần **thực tế không bao giờ chạy** (test `test_19` chỉ xanh khi cả hai rơi vào cùng 1 ms —
đỏ ngay khi máy bận). Đã đổi khoá sang `(logger, level, record.getMessage(), module_tag)`
(không thời gian) + thêm test `test_chong_trung_khong_phu_thuoc_moc_thoi_gian_trong_dong_log`.

**Test:** tier A toàn bộ xanh (thêm 6 test cấu hình/prewarm trong `test_48`, 1 test log trong
`test_19`). Cấu hình mới trong popup: `SEG theo dấu câu` (bật/tắt), `Mốc cắt bằng Whisper`
(bật/tắt), `Max chars/câu`, `Tail (ký tự)`, `Nhịp chờ`.

### 11. Log TRACE từng nhịp ASR + SEG "bóng" (theo yêu cầu tinh chỉnh mốc cắt)

Người dùng cần tự chọn mốc ngắt câu bằng cách so sánh 2 lần chạy video (có/không SEG).
Đã thêm:

* `SegmentationConfig.debug_trace` (mặc định **BẬT**) — ghi log mỗi nhịp preview:
  * `[SEG_TRACE] … CHỜ: <lý do>` — vì sao CHƯA chốt, kèm số liệu thật:
    `dấu ở cuối preview, chưa có tail (nhịp 1/2, 0/350ms)`,
    `tail 'Su' dính giữa từ (Latin chưa có khoảng trắng)`,
    `tail '一つは技術' đủ chữ nhưng mới 1/2 nhịp (0/280ms)`,
    `chưa có dấu kết câu (dài 51/100 ký tự nội dung)`.
  * `[SEG_CUT] … CHỐT (<reason>): '<câu>' | ranh giới@N tail='…' | phần chưa chốt: '…'`
  * `[ASR] [utt=…] SEG timer: cắt tại X.XXs trong vùng (tin cậy …, …ms)` — mốc thật của Whisper.
* `SegmentationConfig.shadow_when_disabled` (mặc định BẬT) — khi SEG **TẮT**, engine vẫn chạy
  một máy trạng thái SEG "bóng" và ghi `[SEG_SHADOW] … nếu BẬT thì đã cắt (<reason>) tại đây:
  '<câu>'`. Nhờ vậy **một lần chạy video vẫn so sánh được A/B**, không cần chạy hai lần.
* Công tắc `🔍 Log từng nhịp SEG` trong popup (`seg_debug_trace`) để tắt khi không cần.
* `SentenceCompleter.trace_state()` trả chẩn đoán (`hold`, `tail`, `boundary_index`, số nhịp…).
* `SEG` được thêm vào `KNOWN_TAGS`/`MODULE_COLORS` (đúng quy ước log của repo) và `SEG_*`
  vào danh sách token `[...]` hợp lệ.

**FIX-11 (bug lộ ra từ log phim thật):** mảnh cắt ra chỉ còn dấu câu vẫn bị "gộp vào câu kế
tiếp" ⇒ vùng audio không tiến ⇒ nhịp sau đọc lại đúng mảnh đó, lại gọi timer (~105 ms) rồi
lại gộp — log cho thấy `'Aquaman.' -> '.'` lặp **4 lần**. Nay mảnh không còn ký tự nội dung
thì **bỏ hẳn** và đẩy mốc vùng đọc lên `end_s` (`asr.commit_content_free_dropped`).

**Cách đọc log khi chạy lại video:** mỗi câu có chuỗi `[SEG_TRACE] … [SEG_CUT] … SEG timer`.
Nếu câu bị chẻ đôi: tìm dòng `CHỐT (punct_tail)` ngay trước đó và xem `tail='…'` — tail ngắn
hoặc dấu câu là do ASR thả sai ⇒ tăng `Tail (ký tự)` / `Nhịp chờ` ở popup. Nếu câu quá dài
mới chốt: xem `CHỜ: chưa có dấu kết câu (dài N/100 ký tự)` ⇒ giảm `Max chars/câu`.

**Test:** thêm 6 test (`test_trace_tung_nhip_ghi_ly_do_cho`, `test_trace_ghi_ca_khi_chot`,
`test_shadow_log_khi_seg_tat`, `test_may_trang_thai_mo_ta_duoc_ly_do_cho`,
`test_manh_chi_con_dau_cau_bi_bo_han_khong_lap`, + cập nhật quy ước log). Tier A: **455/500
collected — toàn bộ PASS**.

### 12. Vòng tinh chỉnh thứ hai theo log TRACE (2026-09-22 22:07)

Log trace cho thấy 3 lỗi thật (đã sửa):

**(a) SEG cắt câu 1 TỪ rồi bị gộp lại ⇒ LẶP.** `'Laughter.'`, `'Zero.'`, `'Oh.'` được chốt
bằng `punct_stable`, tầng commit gộp lại ("quá ngắn") ⇒ **vùng audio không tiến** ⇒ nhịp sau
cắt đúng chỗ đó (log lặp 4 lần, gọi timer 4 lần ≈ 400 ms GPU vô ích).
→ `SentenceCompleter` nay có ngưỡng `min_words` **dùng chung `SentenceConfig.min_words_to_commit`**
(popup "Min Words"): câu chưa đủ từ thì **gộp tiếp** thay vì cắt. Đặt = 0 ở popup thì SEG lại
cho cắt câu ngắn (lựa chọn của người dùng) — nhưng khi đó WS cũng không lọc nữa nên không lặp.
`update_sentence_config(min_words_to_commit=…)` dựng lại SEG ngay.

**(b) Mảnh cắt xong bị dính ĐUÔI CÂU SAU.** `'Aquaman. This.'`, `'…squirt gun. It's'`,
`'…pretty cool lock.'` (mốc cắt của timer hơi muộn) ⇒ phụ đề gộp hai câu.
→ **FIX-12**: sau khi chạy lại ASR trên mảnh đã cắt, nếu văn bản chứa **nhiều câu** thì chỉ giữ
**câu đầu tiên** (`split_complete_sentences`), phần đuôi thuộc vùng kế tiếp (vùng mới bắt đầu
tại mốc cắt nên sẽ được đọc lại) — counter `asr.commit_tail_trimmed`.

**(c) Trace quá dài dòng khi `punct_stable`**: giữ nguyên (bật/tắt bằng `seg_debug_trace`),
nhưng thêm lý do "câu mới N/M từ — chờ gộp" để phân biệt với "chưa có dấu kết câu".

**Đọc log của bạn để chỉnh tiếp:**

| Trong log | Nghĩa | Chỉnh |
|---|---|---|
| `CHỜ: câu mới 1/2 từ … chờ gộp` | SEG đang chờ cho đủ từ (chống cắt vụn/lặp) | giảm "Min Words" nếu muốn câu ngắn hiện ngay |
| `CHỜ: tail '…' dính giữa từ` | ASR đang viết dở một từ | bình thường |
| `CHỜ: dấu ở cuối preview … (nhịp 2/2, 228/350ms)` | chờ đủ 350 ms cho dấu ổn định | giảm `stable_ms` nếu muốn nhanh hơn (code) |
| `CHỐT (punct_tail)` với `tail=''` | mốc cắt lấy từ timer | xem dòng `SEG timer: cắt tại … tin cậy …` |
| `SEG timer: … tin cậy 0.40` | Whisper khớp văn bản kém (ASR nghe sai nhiều) | tăng `timer_min_confidence` để bỏ qua (dùng chồng lấn) |

**Test:** thêm `test_khong_cat_cau_qua_it_tu_tranh_lap`, `test_min_words_bang_0_thi_cho_cat_cau_ngan`,
`test_flush_luon_chot_du_cau_ngan`, `test_engine_lay_min_words_tu_config_sentence`.
Tier A: **459/504 collected — toàn bộ PASS**.

### 13. Vì sao VAD đóng câu GIỮA CÂU — FIX-13 (log 2026-09-22 22:15)

Hiện tượng người dùng chỉ ra:
```
[SEG_TRACE] +3.0s | CHỜ: dấu ở cuối preview, chưa có tail … | 'Don't try and trick me into.'
[VAD] END [engine_end, im lặng 0ms >= 1500ms] (p=1.00)
[VAD] START (p=1.00)
[ASR_COMMIT] [VAD_SILENCE]: "Don't try and trick me into."
```
→ Câu đúng phải là *"Don't try and trick me into buying something I don't want."* nhưng bị chẻ
đôi. **Nguyên nhân nằm ở tầng VAD, không phải SEG.**

`backend/vad/processor.py` (chế độ ghi đè `silence_duration_ms > 0`) có nhánh:
```python
forced_split = res.event == "END" and bool(res.is_speech)   # ← SAI
if forced_split or silent_ms >= silence_duration_ms: close(...)
```
`is_speech=True` trên một frame `END` **không** có nghĩa là "engine cắt cưỡng bức vì trần
cứng": đó thường chỉ là **frame chuyển tiếp** (FireRed vừa đủ bộ đếm im lặng nội bộ của nó
lại vừa có xác suất cao trở lại — log ghi `p=1.00`). Hệ quả: điều kiện số 1 của người dùng
(`silence_duration_ms = 1500 ms`) bị **ghi đè**, câu bị đóng với `im lặng 0ms`, ASR chốt
ngay, vùng audio reset ⇒ câu bị chẻ đôi.

**Sửa:**
1. `VADResult.forced` (mặc định False) — engine chỉ đánh dấu khi đó là **trần cứng của chính
   nó** (ví dụ `max_speech_frame`).
2. Processor: `forced_split = (event == "END") and res.forced` — bỏ hoàn toàn suy diễn từ
   `is_speech`. Chế độ ghi đè nay CHỈ đóng theo đồng hồ im lặng (đúng đặc tả "điều kiện số 1").
3. Log rõ khi END của engine bị hoãn (một lần cho mỗi đoạn nói):
   `END của engine bị HOÃN: mới im lặng 0ms < 1500ms (p=1.00, is_speech=True) — giữ câu mở
   để không cắt giữa câu.` ⇒ lần chạy sau sẽ thấy ngay FireRed có hay phát END sớm không.
4. `engine_end_logged` reset ở mỗi START.

**Test mới** (`test_42_vad_signal_integrity.py`):
`test_end_kem_bang_chung_noi_KHONG_duoc_cat_giua_cau` (tái hiện đúng chữ ký lỗi:
END + `is_speech=True` + `p=1.0`), `test_end_kem_bang_chung_noi_o_che_do_docs_van_ton_trong_engine`,
`test_forced_flag_moi_duoc_cat_cuong_buc`. Tier A: **462/507 collected — toàn bộ PASS**.

### 14. FIX-12b — sửa vòng lặp do chính FIX-12 (log 2026-09-22 22:21)

Log cho thấy hậu quả của FIX-12 bản 1:
```
[SEG_CUT] CHỐT (punct_stable): 'Okay. Well, I don't know how much you want to spend, but I do have this pretty cool Aquaman statue.'
[ASR_COMMIT] SEG: bỏ mảnh đuôi … -> 'Okay.'
[ASR_COMMIT] Mảnh cắt quá ngắn (1 < 2 từ) — gộp vào câu kế tiếp: 'Okay.'
… lặp 6 lần … rồi [MAX_DURATION]: câu phình 15 s
```
**Nguyên nhân:** văn bản chạy lại trên mảnh đã cắt chứa NHIỀU câu
(`'Okay. Well, … Aquaman statue. Aquaman. This isn't a gag gift, Stewart.'`); FIX-12 bản 1
luôn giữ **câu ĐẦU** — mà câu đầu ở đây là `'Okay.'` (1 từ) ⇒ bị coi là quá ngắn ⇒ gộp lại ⇒
**vùng audio không tiến** ⇒ SEG cắt lại đúng ranh giới đó ⇒ lặp tới `MAX_DURATION`.
(Nguyên nhân sâu: ASR liên tục VIẾT LẠI cùng một đuôi — `lock` → `aquarium` → `Aquaman's` →
`Aquaman statue` — nên SEG giữ câu cho tới khi text ổn định; đây là hành vi ĐÚNG của SEG.)

**Sửa:**
1. **FIX-12 bản 2**: khi văn bản chạy lại có nhiều câu, chọn câu **KHỚP NHẤT với câu SEG đã
   chốt** (`_best_matching_sentence` dùng `difflib` trên văn bản đã bỏ dấu câu) thay vì câu đầu.
2. **FIX-12b (chốt an toàn)**: nếu CÙNG một mảnh ngắn bị gộp hai lần liên tiếp thì **cắt vòng
   lặp** — bỏ mảnh và đẩy mốc vùng đọc lên `end_s` (log WARNING + counter
   `asr.commit_carry_loop_broken`). Nhờ vậy mọi vòng lặp tương lai tự dừng sau 1 nhịp thay vì
   phình tới `MAX_DURATION`.

**Cũng trong log này:** `[VAD] END của engine bị HOÃN: mới im lặng 0ms < 1500ms (p=0.64,
is_speech=True)` ⇒ **FIX-13 hoạt động**: END sớm của FireRed không còn cắt giữa câu.

**Test:** `test_chon_cau_khop_voi_cau_seg_da_chot`, `test_cat_vong_lap_manh_ngan_lap_lai`.
Tier A: **464/509 collected — toàn bộ PASS**.

### 15. Tiếng Nhật: `punct_stable` cắt khi CHƯA có câu mới — FIX-14

Người dùng chuyển sang test tiếng Nhật và nhận xét đúng: *"hình như chưa xuất hiện đuôi mà đã
[SEG_CUT]?"*. Log xác nhận: những lần đó là **`punct_stable`** — dấu kết câu đứng yên ≥ 2 nhịp /
350 ms là cắt, **không cần** tail.

Với tiếng Nhật, Qwen3 thả `。` **rất sớm giữa câu** (thể tiếp diễn `連用形`):
`得意先への営業回りを終え。`, `夕食を済ませ。`, `同じチームになり。`, `五年先輩の羽田さんとは。`,
`その日。` ⇒ cắt theo độ ổn định là **chẻ câu**.

**Sửa (FIX-14):**
* `SentenceCompleter.allow_stable_cut` (mặc định True ở mức class) + `SegmentationConfig.stable_cut`
  (**mặc định `False`** trong pipeline) ⇒ SEG chỉ chốt khi có **câu mới thật** (`punct_tail`),
  hoặc khi **VAD đủ im lặng** (`silence_duration_ms` — điều kiện số 1), hoặc `max_chars`/`flush`.
* Log trace đổi lý do chờ thành `CHƯA có câu mới (đang tắt cắt-theo-độ-ổn-định; chờ câu mới hoặc
  im lặng VAD)` để không gây hiểu nhầm.
* Công tắc popup `⚡ Cắt khi dấu câu ổn định` (mặc định **OFF**) để bật lại khi cần giảm độ trễ.

**Cũng xác nhận trong log tiếng Nhật:**
* `[VAD] END của engine bị HOÃN: mới im lặng 0ms < 1500ms (p=0.99, is_speech=True)` ⇒ FIX-13 chạy đúng.
* `Mảnh 'だろ。' bị gộp LẶP LẠI — cắt vòng lặp, đẩy vùng đọc lên 32.43s` ⇒ chốt an toàn FIX-12b
  hoạt động (thay vì phình câu tới `MAX_DURATION`).
* `[VAD] END [silence_override, im lặng 1500ms >= 1500ms]` ⇒ chế độ ghi đè đúng đặc tả.

**Gợi ý thông số cho tiếng Nhật:** `Min Words` 2–3 (3 sẽ lọc `はい。`/`は。` — hợp lý),
`Max chars/câu` 60–80 (100 là khá dài cho phụ đề JP), `Tail (ký tự)` 4–6.

**Test:** `test_mac_dinh_KHONG_cat_khi_chua_co_cau_moi`, `test_bat_stable_cut_thi_cat_khong_can_cau_moi`,
`test_engine_mac_dinh_tat_stable_cut`. Tier A: **467/512 collected — toàn bộ PASS**.

### 16. `stable_ms` ra popup + bug ngôn ngữ của timer (2026-09-22 23:00)

Người dùng xác nhận chạy ổn với `stable_cut = False` và yêu cầu chỉnh được tham số **350 ms**.

* Expose `segmentation.stable_ms` (100–1500 ms, bước 50) ra popup + `POST/GET /api/config` +
  `SessionConfigPayload` (`seg_stable_ms`/`segStableMs`) → `update_seg_config(stable_ms=…)`
  dựng lại SEG ngay. Giá trị cũng được lưu/khôi phục trong `bs_settings`.
* **BUG sửa kèm**: `WhisperTimer` hard-code `language="ja"` trong `get_seg_timer()` ⇒ **phiên
  tiếng Anh vẫn đưa "ja" cho Whisper** (mốc cắt kém chính xác). Nay `run_segments()`/
  `locate_boundary()` nhận `language`, engine truyền `self.language` của phiên; mặc định lấy
  `config.asr.language` (`auto` ⇒ Whisper tự dò). Xem log cũ: các dòng `SEG timer: … tin cậy 0.40`
  ở phim tiếng Anh chính là hệ quả của lỗi này.

**Test:** `test_timer_lay_ngon_ngu_cua_phien`. Tier A: **468/513 collected — toàn bộ PASS**.

### 17. #2 — Tinh chỉnh mốc cắt: ĐÃ THỬ RỒI **GỠ BỎ** (2026-09-22 23:20)

Đã cài đặt và đo thật (`backend/segmentation/refine.py` + `_seg_cut_sample` async): decode lại
3 mốc thô quanh mốc gốc, chia đôi 2 vòng, bù độ trễ nhận dạng ⇒ mốc cắt neo vào bằng chứng văn bản:
```
[SEG tinh chỉnh mốc cắt: 9.90s -> 8.35s (-1550 ms, 5 probe, 1267ms) — chìa khoá '一つは']
```
**Chi phí đo được ~1.25 s GPU cho MỖI câu được cắt** — quá đắt so với lợi ích, kể cả khi đã có
đường bỏ qua lúc timer chắc (`refine_skip_timer_confidence`).

⇒ **Theo yêu cầu người dùng: đã GỠ BỎ HOÀN TOÀN** (module `refine.py`, các field
`SegmentationConfig.refine_*`, nhánh trong `_seg_cut_sample`, 6 test). `_seg_cut_sample` trở lại
**sync**, chỉ trả mốc gốc từ `_seg_base_cut_sample()` (Whisper timer hoặc chồng lấn).
Tier A sau khi gỡ: **468/513 collected — toàn bộ PASS**.

### 18. Nemotron thay Whisper làm SEG timer? (đo ngày 2026-09-22)

| | whisper-large-v3-turbo | nemotron-3.5-asr-streaming-0.6b |
|---|---|---|
| Mức timestamp | **SEGMENT** (6-7 s) | **TOKEN** (từng ký tự, ~100-200 ms) |
| Độ chi tiết | thô, phải nội suy | rất mịn |
| Chất lượng chữ Nhật | tốt | **kém hơn** (`決定路`↔`決定論`, `成治的`↔`政治的`, `ただって`↔`たどって`) |
| Tốc độ (đo CUDA, clip 23 s) | **~0.22 s** (RTF 0.01) | **~1.43 s** (RTF 0.06) — chậm ~6× |
| Tin cậy căn chỉnh đo được | 0.40-0.96 | 0.37 |

**Kết luận: KHÔNG nên đổi.** Timestamp mịn hơn là lợi thế thật, nhưng kiến trúc hiện tại **căn
chỉnh văn bản Qwen3 ↔ văn bản timer**; Nemotron nghe tiếng Nhật kém hơn ⇒ căn chỉnh yếu hơn, mà
lại chậm ~6× mỗi lần cắt — đúng thứ vừa phải gỡ vì đắt. Muốn chính xác hơn thì hướng đúng là
**forced aligner trên chính văn bản Qwen3** (không căn chỉnh chéo model), hoặc **bám mép segment**
của Whisper thay vì nội suy đều ký tự (miễn phí, nhưng có thể cắt lố).

### 17b. Chi tiết kỹ thuật của #2 (đã gỡ — giữ lại để tham khảo)

`backend/segmentation/refine.py` + `TranscribeEngine._seg_cut_sample()` (nay là `async`):
sau khi có mốc cắt GỐC (`_seg_base_cut_sample` — Whisper timer hoặc `end − chồng lấn`), SEG
biết **câu đã trọn** và **vài ký tự đầu câu kế tiếp** (`tail_key`), nên:

1. **Bước thô** — decode lại 3 mốc `base−search / base / base+search` (mặc định ±600 ms), mỗi
   mốc decode từ **đầu vùng nói** (đủ ngữ cảnh — đo thật: cửa sổ ngắn 2.5 s làm ASR mất ngữ
   cảnh, không nhận ra `tail_key`).
2. **Bước mịn** — chia đôi giữa "mốc sạch cuối" và "mốc đã thấy câu mới" (`refine_iterations`,
   mặc định 2 ⇒ tổng ~5 lượt decode).
3. **Bù độ trễ nhận dạng** (`refine_lead_ms = 350`): ASR cần thêm audio trước khi nhận ra chữ
   đầu câu mới ⇒ mốc phát hiện bị muộn; cắt `detect − lead` nhưng **không sớm hơn mốc sạch
   cuối cùng** (bằng chứng cứng).

**Đo thật** (`scratch/seg_trace_demo.py`, FLEURS, GPU RTX 5060 Ti):
```
[SEG_CUT] CHỐT (punct_tail): '技術決定論のほとんどの解釈は、二つの一般論を共有しています。'
[SEG tinh chỉnh mốc cắt: 9.90s -> 8.35s (-1550 ms, 5 probe, 1267ms) — chìa khoá '一つは']
```
⇒ Mốc cắt được **neo vào bằng chứng văn bản** thay vì phỏng đoán `−600 ms`; chi phí đo được
**~1.25 s/câu** (5 lượt decode cả vùng) — cao hơn ước tính 0.4 s ban đầu, nên:

* **bỏ qua tinh chỉnh khi timer đã chắc** (`refine_skip_timer_confidence = 0.7`) ⇒ ca thường
  gặp không tốn thêm gì;
* nếu không thấy `tail_key` trong dải dò ⇒ **giữ mốc gốc** (không đổi mốc khi thiếu bằng chứng);
* tinh chỉnh lỗi/ngoài vùng ⇒ luôn quay về mốc gốc (không bao giờ làm mất câu).

Tham số: `refine_boundary` (bật/tắt), `refine_search_ms`, `refine_iterations`,
`refine_lead_ms`, `refine_skip_timer_confidence`, `refine_window_ms` (0 = cả vùng).
Metric: `seg.refine_used`, `seg.refine_miss`, `seg.refine_ms`, `seg.refine_skipped_timer_confident`.

**Test:** `test_tail_key_lay_chu_dau_cau_ke_tiep`, `test_plan_probes_quanh_moc_goc`,
`test_pick_cut_cat_ngay_truoc_khi_cau_moi_xuat_hien`, `test_pick_cut_giu_moc_goc_khi_khong_thay_chia_khoa`,
`test_engine_tinh_chinh_moc_cat`, `test_engine_tinh_chinh_tat_thi_giu_moc_goc`.
Tier A: **474/519 collected — toàn bộ PASS**.


ĐÃ XONG (trước đây nằm trong danh sách này): nối SEG vào engine ✅ · thay BẬC 3 ✅ · mốc cắt từ
ASR phụ có timestamp (Whisper timer) ✅ · lớp căn chỉnh văn bản Qwen3 ↔ Whisper ✅ · tham số SEG
ra popup + `/api/config` + phiên WS ✅ · prewarm timer ✅ · trace từng nhịp + SEG "bóng" ✅.

CÒN LẠI:
1. **Namo Turn Detector** — mới chỉ có hook `turn_scorer`; chưa nạp model. Khảo sát cho thấy
   checkpoint JA bản `model_quant.onnx` **hỏng** (trả hằng số) và fp32 dao động mạnh theo partial
   ⇒ nếu làm phải debounce hoặc dùng `oboroge0/hayamimi-punct-ja-4class` (MIT, 37 MB int8).
2. **Tinh chỉnh ranh giới bằng decode lại cửa sổ con** — chưa có; sẽ giảm sai số mốc cắt từ
   ±bước quét xuống ±250 ms khi đoạn nói KHÔNG có khoảng lặng.
3. **Chế độ "dịch nguyên vùng dài rồi tách hiển thị"** — chưa có; hữu ích khi câu JP dài 100 ký tự.
4. **Test end-to-end tự động trên audio** — `test_47` hiện kiểm tra ở mức VĂN BẢN (đối chiếu
   `test.tsv`); chưa có test tự động chạy ASR thật + timer thật (chỉ có probe tay
   `scratch/seg_timer_e2e.py`).
5. **Dashboard metric SEG** — đã có counter/metric (`seg.reason.*`, `seg.timer_used`,
   `seg.timer_miss`, …) nhưng chưa hiện trên `/api/metrics` summary hay UI.
6. **`initial_prompt` cho Whisper timer** — truyền ngữ cảnh câu trước để mốc cắt ổn định hơn
   (tuỳ chọn, chưa làm).
7. **Nghe thử/đối chiếu thủ công trên phim thật** — người dùng đang làm; file ghép FLEURS:
   `wav_test/google_fleurs/ja_jp/ja_concat_1min.wav` (`python scratch/seg_trace_demo.py`).

**→ Kết thúc hạng mục SEG tại đây.** Còn lại (KHÔNG làm tiếp theo yêu cầu): Namo/hayamimi turn
detector, chế độ tách hiển thị cho câu dài, test end-to-end tự động trên audio, dashboard metric SEG.
