# Chẩn đoán lỗi NGẮT CÂU của tuyến OFFLINE_BATCH (Pipeline B v3)

- **Ngày**: 2026-10-02
- **Phạm vi**: `backend/core/lookahead_chunker.py`, `backend/ws/lookahead_handler.py::_offline_batch_loop`,
  `backend/asr/forced_aligner.py::group_words_to_subtitles`
- **Triệu chứng báo cáo**: "cuối câu này là đầu của câu sau" — ranh giới câu bị đặt sai chỗ,
  có đoạn bị lặp lại ở đầu câu kế tiếp.
- **Bằng chứng tái hiện**: `python backend/tools/diag_lookahead_boundary.py` (exit 0).
- **Trạng thái**: ✅ **ĐÃ SỬA P0 + P1** ngày 2026-10-02 — xem §7. ✅ **ĐÃ SỬA tiếp vòng 2**
  (phụ đề hiển thị khác bản ASR offline: mất dấu câu + cụt giữa câu) — xem §8. ✅ **ĐÃ SỬA vòng 3**
  (lỗi tua video: mất vùng audio vừa tua tới / kẹt con trỏ / phát qua vùng chưa xử lý) — xem §9.
  ➡️ **Sau đó**: Pipeline B v2 (streaming) đã bị **XOÁ** và OFFLINE_BATCH thành mặc định — xem
  [`phase6_don_dep_v2_va_mac_dinh.md`](phase6_don_dep_v2_va_mac_dinh.md).

---

## 0. TL;DR — 3 nguyên nhân gốc

| # | Nguyên nhân | Vị trí | Hệ quả quan sát được |
|---|---|---|---|
| **1** | **Không hề dùng VAD**: "silence boundary" thực chất là ngưỡng RMS tuyệt đối. `vad_engine` được nhận vào nhưng **không bao giờ được đọc**. | `lookahead_chunker.py:63,73` (nhận & lưu, không dùng), `:75-123` (quét RMS) | Với audio phim có nhạc nền/room tone, **không frame nào < 0.015** ⇒ nhánh `silence_gap` gần như không bao giờ chạy. |
| **2** | **Fallback `min_rms` quá dễ dãi**: chấp nhận mọi khung 50 ms có RMS < `silence_threshold × 2` = **0.03 tuyệt đối, không so với mức nói cục bộ**. | `lookahead_chunker.py:161-174` | Cắt vào **điểm trũng năng lượng bất kỳ** trong dải 6 s (khe giữa 2 từ, tiếng bật hơi, khe đóng của phụ âm) ⇒ **cắt giữa câu**; ASR tự thêm dấu `.` ở mép khối nên phụ đề trông "trọn câu" nhưng câu thật vẫn tiếp ở khối sau. |
| **3** | **Cắt cưỡng bức ở trần 18 s tạo 1.0 s chồng lấn nhưng KHÔNG cắt phần text trùng.** `next_read_pts = pts_end − 1.0` ⇒ 1 s audio cuối bị ASR phiên âm **hai lần**, và không có bước trừ lặp nào ở tầng text. | `lookahead_chunker.py:176-177`, `:274-278`; `lookahead_handler.py:1248`; đối chiếu `_is_duplicate` `:1644-1650` | **Đúng nguyên văn triệu chứng**: đuôi câu trước xuất hiện nguyên si ở đầu câu sau. |

Bổ sung (cùng họ lỗi, nên sửa luôn):

| # | Vấn đề | Vị trí |
|---|---|---|
| 4 | Bộ gom câu cắt theo **trần cứng 10 từ** ⇒ chẻ giữa cụm từ dù trong cùng một khối. | `forced_aligner.py:349-353,362` |
| 5 | Ngưỡng `max_words=10 / max_duration=4.5 / max_chars=45` **hard-code**, không đọc từ `config`; popup chỉnh phân câu (`max_chars`, `max_duration_sec`, `min_words_to_commit`, `stability_*`) **bị bỏ qua hoàn toàn** ở batch. | `forced_aligner.py:318-320`, gọi tại `lookahead_handler.py:1160` |
| 6 | `is_stream_end=False` vĩnh viễn ⇒ đoạn đuôi < 12 s **không bao giờ** được phiên âm. | `lookahead_handler.py:1100-1105` |
| 7 | Khối Fast-Bootstrap (2.5–4.0 s sau khi tua) cắt cứng tại `min(available, 4.0)` → **chẻ đôi một từ**, và không có overlap bù. | `lookahead_chunker.py:202-221` |

---

## 1. Bằng chứng số học từ log thật (không cần chạy lại)

```
[40.93s -> 58.91s]   dur = 17.98s   ← khối kế bắt đầu 58.91s ⇒ KHÔNG overlap ⇒ nhánh min_rms/silence
[58.91s -> 76.91s]   dur = 18.00s   ← ĐÚNG BẰNG TRẦN max_window_sec ⇒ nhánh forced_overlap
[75.91s -> 91.59s]   bắt đầu 75.91 = 76.91 − 1.00  ⇒ chính là `pts_end − overlap_sec`
```

Hai kết luận rút ra trực tiếp từ con số:

1. **`75.91` chỉ có thể sinh ra từ nhánh `forced_overlap`.** Nhánh `silence_gap`/`min_rms` đặt
   `next_read_pts = pts_end` (không lùi). Vậy 1.0 s audio vùng `[75.91, 76.91]` đã được đưa vào
   **cả hai** khối ⇒ ASR phiên âm hai lần.
2. Và đúng tại ranh giới đó, text lặp y nguyên:

```
Khối  5: '... I'm still working on it. From what I saw the other day, I could understand.'
Khối  6: 'Other day, I could understand why he and some people might find you. What? ...'
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ trùng với đuôi khối 5
```

Các cặp ranh giới khác trong log là **câu bị chẻ giữa dòng nói** (nguyên nhân 2):

```
Khối 2 kết: '... They're using the sweet candy of science to trick.'
Khối 3 mở : 'Children into loving him, pervert. ...'      ← "trick children into loving him" bị chẻ
Khối 1 kết: "... Don't peek behind that curtain of."
Khối 2 mở : 'Fame and celebrity, because ...'             ← "curtain of fame" bị chẻ
```

Bằng chứng "silence detector vô hiệu": trong log **không có** khối nào kết thúc ở cao trào của một
khoảng lặng dài thật; các khối kết ở 12.15 / 12.53 / 14.12 / 17.98 / 18.00 s — tức là điểm trũng
năng lượng nhỏ nhất trong dải quét, không phải ranh giới lặng.

---

## 2. Bằng chứng tái hiện (chạy được, exit 0)

`.tmp_audit/repro_boundary.py` mô phỏng đúng pipeline: chunker thật → giả lập ASR phiên âm **trọn
khối** → `ForcedAlignerService.group_words_to_subtitles` thật → ghép mốc PTS như
`_offline_batch_loop`. Audio: hội thoại liên tục 105 s, khe giữa từ 60 ms (< 250 ms).

### Kịch bản A — nhạc nền to (min RMS 0.039 > 0.03)

```
   [  0.00s ->  18.00s] mode=forced_overlap  next_read= 17.00s   <== CHỒNG LẤN 1.00s
   [ 17.00s ->  35.00s] mode=forced_overlap  next_read= 34.00s   <== CHỒNG LẤN 1.00s
   [ 34.00s ->  52.00s] mode=forced_overlap  next_read= 51.00s   <== CHỒNG LẤN 1.00s
   ...
  ✗ LẶP 2 từ: câu trước kết thúc bằng 'they really' và câu sau MỞ ĐẦU bằng đúng chỗ đó
      [15.62-17.72] you'll see them as they really
      [17.06-19.16] they really are, degenerate carnival folk.
  ✗ LẶP 2 từ: 'Yes, I have.'  ->  'I have.'
  => 4 cặp phụ đề liền nhau bị lặp từ ở ranh giới / 51 cặp
```

Khối bị cắt đúng tại `18.00 / 35.00 / 52.00 / 69.00 / 86.00 / 103.00` — **lưới cứng 18 s**, không
liên quan gì tới lời nói. Đây là bản sao chính xác của hiện tượng trong log.

### Kịch bản B — nhạc nền nhỏ (min RMS 0.0157)

```
   [  0.00s ->  16.32s] mode=min_rms   next_read= 16.32s
   [ 16.32s ->  29.65s] mode=min_rms   next_read= 29.65s
   [ 29.65s ->  45.12s] mode=min_rms   ...
  ✗ Ranh giới khối 0->1 tại 16.32s chẻ đôi câu: '...you do, you'll see' | 'them as they...'
  ✗ Ranh giới khối 1->2 tại 29.65s chẻ đôi câu: '...Have you ever' | 'thought about why...'
  ✗ Ranh giới khối 3->4 tại 62.40s chẻ đôi câu: '...children into loving him,' | 'pervert....'
  ✗ Ranh giới khối 4->5 tại 74.62s chẻ đôi câu: '...Is' | 'it really worth...'
  => 4 ranh giới khối cắt vào GIỮA câu
```

Cắt ở 16.32 / 29.65 / 62.40 / 74.62 s — đúng kiểu "ngẫu nhiên" như 14.66 / 28.78 / 40.93 trong log.
Chú ý: **0/119 frame** được coi là im lặng, nhưng `min_rms` vẫn nhận (0.0157 < 0.03) ⇒ cắt vào
khe 60 ms giữa hai từ, tức là **giữa câu**.

### Lỗi trần 10 từ trong nội bộ một khối (nguyên nhân 4)

```
    [ 23.54- 27.08] They're using the sweet candy of science to trick children
    [ 27.14- 28.52] into loving him, pervert.
```
`They're using the sweet candy of science to trick children` = **đúng 10 từ** ⇒ `is_over_length`
(`forced_aligner.py:349-353`) cắt ngay, chẻ đôi cụm "trick children into loving him". Không cần tới
ranh giới khối.

---

## 3. Luồng ngắt câu hiện tại của OFFLINE_BATCH

```text
timeline (PCM liên tục)
  └─ LookaheadChunker.next_chunk(from_pts)                    [chunker:179]
       ├─ available < 12s  → None (chờ)                       [chunker:224-227]
       ├─ fast_bootstrap   → cắt cứng 2.5–4.0s, KHÔNG tìm biên [chunker:202-221]   ← lỗi 7
       └─ quét dải [T+12, T+18], 3 nhánh:
            (a) silence_gap : RMS(frame 20ms) < 0.015 liên tục ≥250ms  [chunker:99-123] ← lỗi 1
            (b) min_rms     : min RMS(50ms) < 0.03  → cắt tại đó        [chunker:161-174] ← lỗi 2
            (c) forced      : cắt tại mép dải T+18, next = end − 1.0s   [chunker:176-177] ← lỗi 3
  └─ transcribe_block(TRỌN khối)                              [handler:1116]
  └─ aligner.align(TRỌN khối, text)                           [handler:1145]
  └─ group_words_to_subtitles(words, language)  ← chỉ truyền language [handler:1160] ← lỗi 4,5
  └─ gửi từng câu, chống trùng bằng _is_duplicate (so khớp CHÍNH XÁC text) [handler:1216,1644] ← lỗi 3
  └─ _batch_from_pts = chunk.next_read_pts                    [handler:1248]
```

Điểm mấu chốt: **ranh giới khối được coi là ranh giới câu**. Mọi câu đều được gom lại *bên trong*
một khối, nên khi khối bị cắt giữa câu thì phụ đề bị chẻ đôi ở đó, và mép khối luôn "trông như" hết
câu vì ASR tự thêm dấu `.` cho audio bị cụt.

---

## 4. Đề xuất sửa (theo thứ tự ưu tiên)

### P0-1. Trừ phần chồng lấn ở tầng TỪ bằng mốc thời gian aligner (sửa đúng triệu chứng)

Ở `_offline_batch_loop`, trước khi gom câu:

```python
# ranh giới đã phát của khối trước
prev_end = self._batch_prev_end_pts          # = pts_end của khối trước
if aligned_words and chunk.pts_start < prev_end - 1e-3:
    aligned_words = [
        w for w in aligned_words
        if chunk.pts_start + w.start_time >= prev_end - 0.05
    ]
```

- Chính xác về mặt thời gian (không phụ thuộc so khớp chuỗi), hoạt động cả với CJK.
- Fallback khi aligner lỗi (`aligned_words == []`): dùng ngay
  `CommitDeduplicator.trim_boundary_overlap()` — **đã có sẵn** ở `backend/core/dedup.py:102-165`
  và **đang được Pipeline A dùng** (`backend/asr/engine.py:1439-1447`) nhưng tuyến batch **không hề
  gọi**. Cần `record_commit()` sau mỗi câu đã phát.
- Lưu ý: sau khi trim, câu đầu của khối mới có thể bắt đầu giữa câu ("... why he and some people
  might find you."). Đó là mức tốt nhất có thể nếu không stitch (xem P2-1), nhưng **hết lặp**.

### P0-2. Thôi coi RMS là "im lặng"

1. **Dùng VAD thật**: `LookaheadChunker` đã có tham số `vad_engine` — truyền engine VAD của phiên
   (`lookahead_handler.py:171-179`) và quét nhãn speech/non-speech trong dải tìm kiếm. Đây đúng là
   thiết kế đã ghi trong `report/09_lookahead_offline_batch/plan.md` §2.1 & DoD Phase 2.
2. Nếu vẫn giữ RMS để tránh phụ thuộc: **đổi sang ngưỡng TƯƠNG ĐỐI theo mức chương trình**, ví dụ
   `thr_db = p90_dB(window) − 25 dB` (và sàn tuyệt đối ~0.004), thay vì hằng số 0.015/0.03.
3. **Siết nhánh `min_rms`**: chỉ nhận khi `min_rms < 0.35 × median_rms(window)` **và** điểm trũng
   kéo dài ≥ 120 ms. Không đạt ⇒ rơi xuống P1-1 (tìm tiếp) chứ không cắt bừa.
4. **Hai mức ranh giới**: `≥ 450 ms` lặng = ranh giới mạnh (cắt ngay, ưu tiên tuyệt đối);
   `250–450 ms` = ranh giới yếu (chỉ nhận nếu gần mốc lý tưởng). 250 ms là khoảng nghỉ giữa từ
   thông thường trong hội thoại phim, không phải ranh giới câu.

### P0-3. Cắt cưỡng bức cũng phải ở điểm trũng, không ở mép dải

`lookahead_chunker.py:176-177` đang cắt đúng `search_end_pts` (mép 18 s) — tức là chắc chắn giữa
từ nếu người nói liên tục. Kế hoạch gốc (§2.1) đã ghi rõ phải "chọn điểm có năng lượng RMS nhỏ nhất
**hoặc** cưỡng bức". Đổi thành: cắt tại frame 20 ms có năng lượng nhỏ nhất trong dải, **vẫn giữ**
1.0 s overlap (để không mất từ) nhưng overlap nay đã được P0-1 trừ ở tầng từ.

### P1-1. Nới dải tìm kiếm (đừng cắt ở 18 s chỉ vì hết dải)

Log cho thấy `Đệm trước: 70.8s` — pipeline đang có sẵn 40–70 s audio phía trước, trong khi
`lead_time = 15 s`. Vì vậy có thể tìm ranh giới thật xa hơn:

```python
# LookaheadConfig
batch_max_search_sec: float = 26.0   # trần mềm: ưu tiên ranh giới thật trong [T+12, T+26]
batch_hard_max_sec: float = 32.0     # trần cứng: buộc cắt + overlap
```

Ưu tiên: ranh giới mạnh trong `[T+12, T+26]` → ranh giới yếu → `min_rms`+overlap → cắt cứng.
Đánh đổi cần đo: khối đầu tiên dài hơn ⇒ `buffered_ahead` phải lớn hơn mới `prebuffer_ready`;
nên áp trần mềm **sau** khối đầu (khối cold-start giữ 12–18 s) hoặc chỉ khi `buffered_ahead` đã đủ.

### P1-2. Gom câu: không cắt giữa cụm từ

Trong `group_words_to_subtitles`:
- Trần từ/độ dài chỉ được cắt khi **không** có dấu câu trong ±2 từ quanh đó; nếu có, kéo dài tới dấu câu.
- Khi buộc phải cắt do trần: chọn **khe âm học lớn nhất** trong 3 từ cuối trước trần (`gap ≥ 0.18 s`),
  giống tinh thần §2.2 của plan, thay vì cắt đúng tại index trần.
- Thêm luật chống chẻ cụm: **một phụ đề không bao giờ được kết thúc ở vị trí mà phụ đề kế tiếp bắt
  đầu bằng chữ thường** (không có dấu câu kết) — gộp lại, trừ khi vượt trần cứng.
- Đưa `max_words / max_duration_sec / max_chars` vào `LookaheadConfig` và truyền từ call site
  (`lookahead_handler.py:1160` hiện chỉ truyền `language=`), để popup điều khiển được như Pipeline A.

### P1-3. Đuôi stream

`is_stream_end` không bao giờ được bật (`lookahead_handler.py:1100-1105`). Cần bật khi
`_decoded_end_pts` đã tiến sát cuối và không còn mảnh mới (hoặc khi client báo `paused/ended`),
để flush đoạn < 12 s cuối; đồng thời flush trước khi `close()`.

### P2-1. Stitch câu vắt qua ranh giới khối

Muốn câu không bị chẻ đôi ở ranh giới khối thì phải quyết định **trước khi gửi**: giữ lại
("hold back") phụ đề cuối cùng của khối nếu nó kết thúc đúng tại mép khối, rồi khi có khối kế tiếp
thì ghép lại và gom câu một lần. **Không thể gửi bản sửa sau**, vì client chỉ append và bỏ qua item
trùng: `extension_firefox/lib/lookahead-timeline.js:59-64` (`if (exists) continue;`) — không có
upsert. Đánh đổi: phụ đề cuối khối bị trễ thêm một nhịp xử lý (ASR ~0.4 s, không phải realtime), nên
chấp nhận được, nhưng cần giữ `ready_ahead ≥ target` để không chặn `prebuffer_ready`.

### P2-2. Fast-Bootstrap

`lookahead_chunker.py:202-221` cắt cứng 4.0 s ⇒ chẻ đôi từ ngay sau khi tua. Sửa: trong cửa sổ
bootstrap, chọn frame trũng năng lượng (hoặc ranh giới VAD trong 0.5 s cuối) và **cho overlap**;
P0-1 sẽ trừ phần lặp.

---

## 5. Test cần sửa / thêm

- ⚠️ `backend/tests/test_56_lookahead_chunker.py:96-99` đang **khoá cứng hành vi lỗi**
  (`next_read_pts == 17.0` cho khối cắt tại 18.0). Phải cập nhật cùng lúc với P0-3/P1-1.
- Thêm regression: audio hội thoại liên tục + nhạc nền ⇒
  (a) `forced_overlap` không được là nhánh mặc định;
  (b) **không** cặp phụ đề liền nhau nào có ≥ 2 từ trùng ở ranh giới;
  (c) không ranh giới khối nào rơi vào giữa câu khi có khoảng lặng ≥ 450 ms trong dải tìm kiếm.
- Thêm test cho `trim_boundary_overlap` trong luồng batch (đã có test cho nó ở Pipeline A:
  `test_10_streaming_commit_logic.py:347-379`).
- Thêm test cho đuôi stream (P1-3) và bootstrap (P2-2).

---

## 6. Kết luận

Tuyến OFFLINE_BATCH **không có tầng "ngắt câu" theo nghĩa tín hiệu**: nó chỉ có một bộ dò khoảng
lặng bằng RMS tuyệt đối với ngưỡng fallback quá cao, nên trong audio phim thực tế nó cắt ở điểm
trũng năng lượng bất kỳ hoặc ở mép trần 18 s. Cộng thêm việc cắt cưỡng bức tạo 1.0 s chồng lấn mà
**không** trừ phần text trùng (dù `CommitDeduplicator.trim_boundary_overlap` đã tồn tại và đang chạy
ở Pipeline A), kết quả đúng như báo cáo: đuôi câu này trở thành đầu câu sau.

Hai việc mang lại nhiều giá trị nhất, chi phí thấp nhất:
1. **P0-1** (trừ chồng lấn theo mốc từ) — xoá hẳn triệu chứng "lặp ở đầu câu sau".
2. **P0-2/P0-3** (ngưỡng lặng tương đối + cắt cưỡng bức tại điểm trũng) — giảm mạnh số lần chẻ câu.

---

## 7. ĐÃ SỬA (P0 + P1, 2026-10-02)

### 7.1. Nguyên tắc kiến trúc đã giữ đúng

Sửa lần này **không** kéo tuyến batch về cách ngắt câu của Pipeline A. Phân tầng rõ ràng:

| Tầng | Ai quyết định | Có bị đổi không |
|---|---|---|
| Cửa sổ audio đưa vào ASR | `LookaheadChunker` (ranh giới **GIẢI MÃ**) | Đổi để khối **DÀI HƠN** và ranh giới **SẠCH HƠN** |
| Chia câu/phụ đề hiển thị | `group_words_to_subtitles` trên bản phiên âm ĐÃ CÓ TRỌN NGỮ CẢNH | Đổi quy tắc chia, **không** đổi ngữ cảnh ASR |
| Chống lặp ở mép khối | `_offline_batch_loop` trừ theo mốc từ | Mới (trước đây không có) |

Nói cách khác: ASR vẫn nhận một khối liên tục 15–30 s (mục tiêu CER/WER của v3), còn mép khối trở
thành **vô hình** ở đầu ra nhờ overlap + trừ theo mốc từ.

### 7.2. Thay đổi cụ thể

1. **`LookaheadChunker`** (`backend/core/lookahead_chunker.py`)
   - Ngưỡng lặng **TƯƠNG ĐỐI**: `p90(RMS) − batch_silence_rel_db` (sàn `batch_silence_floor_rms`),
     đo trên TOÀN khối chứ không chỉ dải quét ⇒ hoạt động với cả audio to/nhỏ.
   - **Hai mức ranh giới**: `>= batch_strong_silence_ms` (450 ms) cắt ngay; `>= batch_min_silence_ms`
     (250 ms) chỉ cắt khi gần mốc lý tưởng.
   - **Nới dải tìm kiếm** tới `batch_search_max_sec` (30 s) ⇒ gặp khoảng lặng THẬT thay vì cắt cứng 18 s.
   - Fallback `min_rms` chỉ nhận khi điểm trũng **thấp hơn `batch_min_rms_ratio` × trung vị RMS** và
     **kéo dài ≥ `batch_min_rms_hold_ms`** (120 ms) — khe 40–80 ms giữa hai từ không còn là "ranh giới".
   - Cắt cưỡng bức tại **điểm trũng quanh độ dài lý tưởng** (`batch_dip_search_sec`), không ở mép dải.
   - Bỏ tham số chết `vad_engine`.
2. **`group_words_to_subtitles`** (`backend/asr/forced_aligner.py`)
   - Chạm trần mà sắp có dấu kết câu (trong `batch_sub_defer_words` từ) ⇒ **nới tới dấu câu**.
   - Không có dấu câu ⇒ cắt tại dấu phẩy cuối, rồi tới **khe âm học lớn nhất** (≥ 120 ms), cuối cùng mới tới trần.
   - **Gộp** mảnh kết thúc không có dấu câu + mảnh sau bắt đầu chữ thường (hoặc quá ngắn), có trần cứng.
3. **`_offline_batch_loop`** (`backend/ws/lookahead_handler.py`)
   - **Trừ chồng lấn ranh giới theo MỐC TỪ** của aligner trước khi gom câu; fallback
     `CommitDeduplicator.trim_boundary_overlap` khi aligner lỗi (`batch_trim_boundary_overlap`).
   - Chỉ cập nhật ranh giới khi khối THỰC SỰ phát câu ⇒ không trừ oan nội dung chưa phát.
   - `is_stream_end` giờ được bật khi trình phát dừng ở mép cuối buffer ⇒ flush được đoạn đuôi < 12 s.
   - Truyền trần gom câu từ config; `min_words_to_commit` của popup được dùng để **gộp** mảnh cụt
     (không lọc bỏ câu ngắn ⇒ không mất chữ).
4. **Config** (`backend/config.py`, `LookaheadConfig`): 13 khoá mới — xem §7.3.

### 7.3. Khoá cấu hình mới (đều có mặc định an toàn)

```
batch_search_max_sec = 30.0        # trần dải TÌM ranh giới thật (ASR 15-30s)
batch_strong_silence_ms = 450.0    # ranh giới MẠNH
batch_silence_rel_db = 22.0        # ngưỡng lặng = p90 - 22 dB
batch_silence_floor_rms = 0.004
batch_min_rms_ratio = 0.35         # điểm trũng phải thấp hơn 35% trung vị RMS
batch_min_rms_hold_ms = 120.0      # ... và kéo dài >= 120 ms
batch_dip_search_sec = 3.0         # cắt cưỡng bức quanh độ dài lý tưởng ±3s
batch_trim_boundary_overlap = True # TẮT = quay lại hành vi cũ (bị lặp từ)
batch_sub_max_words = 10           # trần gom câu phụ đề (trước đây hard-code)
batch_sub_max_duration_sec = 4.5
batch_sub_max_chars = 45
batch_sub_min_words = 2            # mảnh ngắn hơn được GỘP, không bị bỏ
batch_sub_defer_words = 3          # số từ được nới để đạt dấu câu
```

### 7.4. Kết quả đo lại (`python backend/tools/diag_lookahead_boundary.py`)

Audio kiểm chứng: hội thoại 105 s, khe giữa từ 60 ms (< 250 ms) — tức là **ca xấu nhất: người nói
không hề ngừng**. Trước/sau:

| Chỉ số | Trước | Sau |
|---|---|---|
| Cặp phụ đề liền nhau **lặp từ ở ranh giới** (kịch bản A) | 4 / 51 | **0 / 41** |
| Cặp phụ đề liền nhau **lặp từ ở ranh giới** (kịch bản B) | 0 / 51 | **0 / 42** |
| Mốc cắt khối kịch bản A | cứng đúng `18.00 / 35.00 / 52.00 / 69.00 / 86.00 / 103.00` (lưới cứng) | 12.1–16.9 s, tại điểm trũng gần độ dài lý tưởng |
| Ranh giới khối trùng khít mốc cũ | `76.91 → 75.91` (= end − 1.0) | có overlap nhưng **đã trừ ở tầng từ** |

Chất lượng gom câu cũng tốt lên rõ (cùng một khối):

```
TRƯỚC: [1.94-4.04] At the end of the day,
       [4.10-5.84] he's just another Hollywood phony.
SAU  : [1.94-5.84] At the end of the day, he's just another Hollywood phony.

TRƯỚC: [23.54-27.08] They're using the sweet candy of science to trick children
       [27.14-28.52] into loving him, pervert.
SAU  : ... (nới tới dấu phẩy, không chẻ giữa cụm "trick children into loving him")
```

### 7.5. Điều CÒN LẠI (chưa sửa — cần quyết định riêng)

Trong ca xấu nhất ở trên, **ranh giới KHỐI vẫn rơi vào giữa câu** (7/7 ở kịch bản A, 6/6 ở B) vì
người nói không hề có khoảng lặng ≥ 250 ms trong suốt 105 s — không thuật toán chọn điểm cắt nào
tránh được, và tầng gom câu không thể ghép xuyên khối. Hai hướng xử lý (đều thuộc P2-1, chưa làm):

* **Hold-back + stitch**: giữ lại phụ đề cuối của mỗi khối nếu nó kết thúc đúng mép khối mà không có
  dấu câu, rồi ghép với khối sau và gom lại **trước khi gửi**. Khả thi vì pipeline chạy trước 15 s.
  Lưu ý: client chỉ append và bỏ qua item trùng
  (`extension_firefox/lib/lookahead-timeline.js:59-64`) nên **không thể** gửi bản sửa sau.
* Nếu muốn sửa sau khi đã gửi thì phải thêm cơ chế **upsert theo `start_pts`** ở client.

Trong hội thoại phim thật, khoảng lặng ≥ 450 ms rất phổ biến nên chunker sẽ bám vào đó; ca xấu nhất ở
trên chỉ để chứng minh phần chống lặp hoạt động độc lập với chất lượng điểm cắt.

### 7.6. Test

* `test_56_lookahead_chunker.py`: cập nhật `test_continuous_speech_fallback_overlap` (bản cũ **khoá
  cứng hành vi lỗi** `next_read_pts == 17.0`), thêm 4 test: ranh giới thật ngoài trần cũ, ngưỡng lặng
  tương đối với audio nhỏ, khe 60 ms không phải ranh giới, (kèm) tiến trình nhiều khối.
* `test_57_forced_aligner_service.py`: thêm 3 test cho nới-theo-dấu-câu, gộp mảnh chữ thường, trần cứng.
* `test_58_lookahead_offline_batch.py`: thêm 2 test tích hợp — **không cặp phụ đề nào chồng mốc** tại
  ranh giới khối (đã kiểm chứng: tắt `batch_trim_boundary_overlap` thì test FAIL, bật thì PASS), và
  flush đoạn đuôi khi trình phát dừng.
* `test_54_lookahead_pipeline_v2.py`: **sửa 5 test đã hỏng SẴN từ trước** (không liên quan thay đổi
  này): bộ test đường streaming không chọn `processing_mode` nên chạy nhầm tuyến `offline_batch`
  (mặc định đã đổi từ commit e797815) ⇒ nay chọn tường minh `"streaming"`.
* Toàn bộ suite nhanh: `pytest` — **566 test PASS, 0 fail**; `node --test extension_firefox/tests/`
  — **48 pass, 0 fail**.

---

## 9. VÒNG 3 (2026-10-02): LỖI TUA VIDEO (seek)

Người dùng báo 3 triệu chứng sau khi ngắt câu đã ổn:

| # | Triệu chứng |
|---|---|
| 1 | Tua tới **vùng xám** (RAM đã có audio, chưa phiên âm): OK |
| 2 | Tua tới **vùng video chưa tải**: phụ đề **lúc hiện lúc không** |
| 3 | **Tua ngược về đầu**: **không hiện phụ đề gì** |

Yêu cầu kèm theo: *"phải có ràng buộc cứng là khi play hoặc seek thì ASR đã tiếp nhận được âm thanh
lúc đó + đã xử lý (có thể có phụ đề hoặc không) — tránh trường hợp vẫn phát video mà không có gì
hoạt động"*.

### 9.1. Bằng chứng từ log (không cần suy đoán)

**Race ghi đè mốc tua** (triệu chứng 2):

```
14:22:51.936  [WS] Seek seek_id=…_7thc mốc=34.99s          ← reset pipeline về 34.99s
14:22:52.030  [WS] prebuffer_ready=True (ready_ahead=7.42s, currentTime=34.99s)
                    ⇒ ready_until = 42.41s = CUỐI KHỐI CŨ (impossible nếu mốc tua còn nguyên)
14:22:52.303  [ASR] Lookahead Batch ASR [42.41s -> 54.83s]  ← chạy tiếp từ mốc CŨ
```
⇒ vùng `[34.99s, 42.41s]` **không bao giờ** được phiên âm ⇒ "lúc hiện phụ đề lúc không".

**Deadlock con trỏ** (triệu chứng 3):

```
14:23:26.449  [WS] Seek seek_id=…_8jek mốc=16.89s (RAM=159.1s, đệm trước=63.1s)
14:23:36.932  [ASR] … Playhead: 27.0s … Đã dịch: 232.9s      ← ready_until ≈ 260s (marker VỊ TRÍ CŨ)
14:23:47.213  … Đã dịch: 230.0s        ← 30 giây, KHÔNG có khối nào được xử lý
14:23:57.431  … Đã dịch: 222.8s
```
`_batch_from_pts` ≈ 259 > `current_time + lead + margin` (27+21=48) ⇒ vòng lặp batch `continue`
**mãi mãi**: không ASR, không phụ đề. Và `ready_until=260` khiến `prebuffer_ready=True` ⇒ video vẫn
phát qua vùng chưa xử lý.

### 9.2. Nguyên nhân gốc

| # | Lỗi | Vị trí |
|---|---|---|
| **B1** | **Race**: `_offline_batch_loop` có `await self.send_json(...)` ngay TRƯỚC khi ghi tiến độ, mà các dòng ghi `_batch_from_pts`/`_ready_until_pts` **không có guard theo `seek_seq`**. Trong lúc chờ await, cùng event loop xử lý `seek_reset` ⇒ `handle_seek()` reset mốc tua, rồi iteration CŨ quay lại **ghi đè** bằng mốc khối cũ. | `lookahead_handler.py` cuối `_offline_batch_loop` |
| **B2** | **Không có ràng buộc "luôn xử lý audio quanh vị trí phát"**: khi con trỏ vượt tầm nhìn lookahead, vòng lặp chỉ `continue` — nếu state lệch (tua ngược) thì **kẹt vĩnh viễn**. | `_offline_batch_loop` (nhánh `from_pts > max_ahead_pts`) |
| **B3** | **Marker "đã dịch" không gắn thế hệ seek**: `ready_ahead = _ready_until_pts − current_time` dùng giá trị của vị trí CŨ ⇒ `prebuffer_ready=True` cho vùng chưa xử lý. Thêm nữa, `_commit_batch_progress` dùng `max()` nên marker cũ (quá lớn) **sống sót** và bị "hợp thức hoá" bởi thế hệ mới. | `send_status`, `_commit_batch_progress` |
| **C1** | **Client cố tình KHÔNG bao giờ tạm dừng khi đang phát** (`// ĐANG PHÁT: Tuyệt đối KHÔNG tạm dừng video giữa chừng`), và `ready_until_pts` chưa được dùng làm ràng buộc ⇒ video chạy qua vùng chưa có gì. | `content-script.js`, `lookahead-timeline.js` |

### 9.3. Đã sửa

**Backend — ràng buộc cứng "ASR luôn xử lý audio tại vị trí phát":**

1. **`_commit_batch_progress(seq, next_read_pts, chunk_end)`**: mọi lần ghi tiến độ đều kiểm tra
   `seq == self._seek_seq`; iteration của thế hệ CŨ bị **bỏ qua hoàn toàn**. Marker cũ thuộc thế hệ
   khác bị **THAY** (không `max()`) nên không thể sống sót. Toàn bộ sách trạng thái ranh giới
   (`_batch_prev_emitted_end_audio`, `_batch_emitted_tail_norm`, carry) cũng được guard theo thế hệ.
2. **`_maybe_reanchor_batch_frontier()`**: nếu con trỏ khối kẹt **quá tầm nhìn + một khối dài nhất**
   trong **≥ 2 s bền vững** ⇒ neo lại về `current_time` (kèm xoá trạng thái ranh giới, bật
   Fast-Bootstrap) và log WARNING. Ngưỡng gồm cả một khối dài nhất nên cơ chế throttle bình thường
   (con trỏ nhô hơn tầm nhìn ~15-20 s) **không bao giờ** kích hoạt.
3. **Marker gắn thế hệ**: `_ready_until_seq`; `send_status` chỉ báo `ready_until_pts`/`ready_ahead`
   khi marker thuộc `_seek_seq` hiện tại, ngược lại trả về `current_time` ⇒ `ready_ahead = 0` ⇒
   `prebuffer_ready = False` ⇒ client **không** được phát. Log `prebuffer_ready` kèm `seek_seq` để
   chẩn đoán.

**Client — ràng buộc cứng "playhead không được vượt vùng đã xử lý":**

4. `SubtitleTimelineQueue.setReadyHorizon(pts, seekId)` + `isBehindHorizon(curTime)`: `_tick()` (60 fps)
   kiểm tra mỗi khung hình; chạm mốc ⇒ `_startPrebuffering(underrunPrebufferMs)` ⇒ tạm dừng video và
   hiện "Đang xử lý tiếp đoạn video...". Marker của thế hệ seek CŨ bị bỏ; `seeking`/`clear()` reset
   marker về 0 (chưa biết).
5. `content-script.js`: `onBufferingStateChange(true)` ⇒ `pauseForBuffering("underrun")` (timeout
   9-12 s, có gia hạn khi backend còn tiến triển); `onStatus` nạp `ready_until_pts` vào queue.

### 9.4. Hành vi sau khi sửa (đúng ràng buộc người dùng yêu cầu)

* **Tua tới vùng xám (RAM có audio)**: `handle_seek` reset mốc → Fast-Bootstrap 2.5-4 s → khối được
  phiên âm **ngay** → mốc đã xử lý nhảy lên → video phát. Race không còn ghi đè ⇒ **không mất vùng
  `[target, frontier cũ]`** nữa.
* **Tua tới vùng chưa tải**: `ready_ahead = 0` ⇒ video **đứng chờ** (có thông báo) cho tới khi khối
  đầu tiên xong; sau đó mỗi khi playhead chạm mốc đã xử lý, video **tự tạm dừng** thay vì chạy qua
  vùng trống.
* **Tua ngược về đầu**: nếu vì bất kỳ lý do gì con trỏ vẫn ở xa vị trí phát, watchdog **neo lại trong
  ≤ 2 s** và pipeline xử lý lại từ vị trí phát ⇒ có phụ đề.
* Chốt an toàn cuối vẫn giữ: nếu backend chết hẳn, `pauseForBuffering` hết hạn (9-12 s + gia hạn) và
  video phát tiếp (thà mất phụ đề còn hơn treo trình phát) — nhưng có log rõ ràng.

### 9.5. Test thêm ở vòng 3

* `test_seek_reset_is_not_overwritten_by_inflight_block` (B1) — mô phỏng đúng race: iteration cũ
  `_commit_batch_progress(stale_seq, 42.41, 42.41)` **phải** trả `False` và không đổi mốc tua.
* `test_status_does_not_claim_coverage_of_old_position_after_seek` (B3) — marker 260 s của thế hệ cũ
  không được lọt ra status; `prebuffer_ready` phải `False`; sau khi thế hệ mới xử lý thì marker mới có giá trị.
* `test_batch_frontier_reanchors_when_stuck_far_ahead` (B2) — kẹt 5 s ⇒ neo lại về 27 s; nhô hơn tầm
  nhìn trong phạm vi một khối ⇒ **không** neo.
* JS: `tạm dừng khi playhead chạm mốc backend đã xử lý`, `mốc đã xử lý của thế hệ seek CŨ không áp cho
  vị trí mới` (`node --test extension_firefox/tests/`).

---

## 8. VÒNG 2 (2026-10-02, cùng ngày): PHỤ ĐỀ HIỂN THỊ KHÁC BẢN ASR OFFLINE

### 8.1. Triệu chứng người dùng báo (kèm ảnh chụp màn hình)

```
ASR offline:  "You took ballet. God, you never listen. I'm excited to meet Emily. Me too. …"
Phụ đề hiện:  "You took ballet God you never listen I'm"        ← cụt giữa câu + MẤT HẾT dấu câu
Bản dịch:     "Anh lấy đi múa ballet rồi, trời ơi, anh chẳng bao giờ lắng nghe tôi cả."
```

### 8.2. Nguyên nhân gốc: **Forced Aligner bỏ hết dấu câu khi tokenize**

`qwen3-asr/Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py::is_kept_char()` chỉ giữ ký tự
Unicode category `L` (chữ) / `N` (số) + dấu nháy đơn; `clean_token()` xoá mọi thứ còn lại. Bằng chứng
đã có sẵn trong repo — `report/09_lookahead_offline_batch/phase1_poc_results.md` §1.1:

```
ASR:      "私の人生の中心で最も魅力的だ。"      ← có 。
Aligner:  私 / の / 人生 / … / "魅力的だ"        ← KHÔNG còn 。
Gom câu:  "私の人生の中心で最も魅力的だ"         ← mất dấu câu
```

Hệ quả dây chuyền trên tuyến OFFLINE_BATCH:

1. `AlignedWord.text` **không bao giờ** chứa `.`/`,`/`?`/`。` ⇒ trong `group_words_to_subtitles`,
   `is_sent_end` và `is_clause_end` **luôn False** ⇒ không bao giờ ngắt theo dấu câu;
2. chỉ còn cắt theo trần `max_words = 10` / `max_duration = 4.5 s` ⇒ **chẻ giữa câu**;
3. text phụ đề ghép từ `item.text` ⇒ **mất dấu câu**;
4. **bản dịch nhận mảnh vụn** thay vì câu trọn vẹn (chất lượng dịch giảm theo, dù prompt có ngữ cảnh).

### 8.3. Đã sửa

| # | Sửa gì | Ở đâu |
|---|---|---|
| **A** | `merge_source_text()`: gắn DẤU CÂU + chính tả gốc của văn bản ASR trở lại từng từ đã căn chỉnh (khớp theo chuỗi ký tự "được giữ", hoạt động cho cả Latin lẫn CJK/Nhật/Hàn). `align()` gọi tự động (`attach_source_punctuation=True`). | `backend/asr/forced_aligner.py` |
| **B** | Trần **2 tầng** cho gom câu: có dấu kết câu trong phạm vi `sentence_max_words` (24) / `sentence_max_duration_sec` (8 s) ⇒ **cả câu là MỘT phụ đề**; chỉ câu quá dài mới cắt bên trong (dấu phẩy → khe âm học → trần). | `group_words_to_subtitles` |
| **C** | Trừ chồng lấn ranh giới thêm **lớp theo CHUỖI TỪ** (bù sai số mốc 100–300 ms giữa hai lần align). Đo thật: lớp theo mốc bỏ 3 từ mà `"In front of"` vẫn lọt ra phụ đề. | `_trim_batch_leading_overlap` |
| **D** | **Giữ lại mảnh cuối CHƯA kết câu** của khối bị cắt giữa câu, ghép vào đầu khối sau ⇒ không bao giờ phát phụ đề cụt ở ranh giới khối (đúng `"…You better"` trong ảnh). Không giữ khi khối kết ở khoảng lặng thật hoặc khi video đã hết. | `_hold_back_unfinished_tail` + bước 2D |
| **E** | Log `[SEG_BATCH]` in ra **text THẬT SỰ gửi đi** (trước đây chỉ log text ASR nên không thể biết phụ đề khác ở đâu). | `_offline_batch_loop` |

Cấu hình mới: `batch_sub_sentence_max_words` (24), `batch_sub_sentence_max_duration_sec` (8.0).

### 8.4. Kết quả (`python backend/tools/diag_lookahead_boundary.py`)

Đúng hai ví dụ của người dùng — trước/sau:

```
ASR: "You took ballet. God, you never listen. I'm excited to meet Emily. Me too. I just hope he doesn't blow it."
  TRƯỚC: • You took ballet God you never listen I'm      ← khớp y ảnh chụp màn hình
         • excited to meet Emily Me too I just
         • hope he doesn't blow it
  SAU:   ✓ You took ballet.      ✓ God, you never listen.   ✓ I'm excited to meet Emily.
         ✓ Me too.               ✓ I just hope he doesn't blow it.

ASR: "In front of me. I promise I'll be on my best behavior. You better be. No jokes about how close I am
      with my dog, or the truth about how close I am with my dog. You got it. No jokes about the year I took ballet."
  SAU:   ✓ In front of me.
         ✓ I promise I'll be on my best behavior.
         ✓ You better be.
         ✓ No jokes about how close I am with my dog, or the truth about how close I am with my dog.
         ✓ You got it.
         ✓ No jokes about the year I took ballet.
```

Trên ca xấu nhất (105 s hội thoại **không có khe lặng nào**, 8 khối đều bị cắt cưỡng bức):

| Chỉ số | Trước vòng 2 | Sau vòng 2 |
|---|---|---|
| Cặp phụ đề liền nhau **lặp từ** | 0 / 30 | **0 / 30** |
| Phụ đề là **mảnh cụt thật** (không có dấu câu nào) | ~7 / 31 | **0 / 31** |
| Phụ đề cắt ở dấu phẩy (câu quá dài, theo thiết kế) | — | 1 / 31 |

### 8.5. Vì sao KHÔNG chọn "dịch nguyên đoạn rồi tách bản dịch"

Đề xuất của người dùng có hai nửa; nửa đầu **đã làm**, nửa sau **không nên làm**:

* ✅ **"Từ đoạn ASR hoàn hảo ⇒ tách thành những câu hoàn hảo"** — chính là fix A+B: văn bản ASR (có
  dấu câu) trở thành nguồn sự thật cho việc tách câu, aligner chỉ cấp mốc thời gian.
* ❌ **"Dịch nguyên đoạn rồi tách bản dịch"** — phá ánh xạ 1:1 giữa phụ đề và mốc thời gian: trật tự
  từ của tiếng Việt khác tiếng Anh nên không thể gán ngược câu dịch về mốc của câu gốc; TTS cũng
  không còn cửa sổ để khớp. Plan §2.3 đã cân nhắc và loại phương án này vì đúng lý do đó.
  Chất lượng ngữ cảnh vẫn được giữ bằng **dịch từng câu + bơm ngữ cảnh 2 chiều** (2 câu trước + 1 câu
  sau), và nay mỗi câu đưa vào dịch đã là **câu trọn vẹn** (trước đây là mảnh cụt) — đây mới là mức
  cải thiện chất lượng dịch lớn nhất.

### 8.6. Test thêm ở vòng 2

* `test_57`: gắn lại dấu câu (Anh, Nhật, dấu nháy trong từ), `align()` gắn dấu câu end-to-end (stub model,
  không cần GPU), và **đúng danh sách câu mong đợi của người dùng**.
* `test_58`: phụ đề phát ra phải là **câu trọn vẹn** (mọi item kết bằng dấu câu) với aligner giả bắt
  chước đúng hành vi bỏ dấu câu của upstream; giữ-lại-mảnh-cuối (4 nhánh: cắt giữa câu / khoảng lặng
  thật / hết stream / đã trọn câu); trừ chồng lấn theo chuỗi từ khi mốc align lệch.

