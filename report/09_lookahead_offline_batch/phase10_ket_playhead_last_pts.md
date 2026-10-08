# Phase 10 — Pipeline B kẹt vĩnh viễn ở playhead: `_last_pts` bị đẩy qua audio CHƯA PHÁT

- **Ngày**: 2026-10-08
- **Triệu chứng**: phiên Lookahead mở GIỮA video, video phát được ~6 s rồi **đứng im mãi**.
- **Trạng thái**: ✅ đã tìm ra nguyên nhân gốc và sửa. Có test hồi quy.

---

## 1. Log thật

```
[WS] Lookahead init: neo vị trí phát hiện tại @9.38s
[WS] StreamDemuxer: SourceBuffer epoch mới = 2 (min_pts=8.88) — xoá bộ đệm ghép nối
[Lookahead] 📦 Replay 10 mảnh audio quanh vị trí phát (9.4s; phủ 0.0s→40.0s, CÓ audio tại playhead;
             bỏ 0 mảnh ngoài cửa sổ, 0 mảnh trùng).
[Lookahead Client] 📊 Đã gửi 13 khung / 16.69 MB
[WS] Lookahead prebuffer_ready=True (ready_ahead=2.63s, fed_ahead=2.63s, buffered_ahead=6.62s, currentTime=9.38s)
[ASR] Lookahead Batch ASR [12.01s -> 16.00s] (197ms): '说的是。'
[WS] Lookahead prebuffer_ready=False (ready_ahead=0.82s, ..., currentTime=15.18s)
[ASR] Trạng thái Lookahead | Playhead: 15.7s | Audio RAM: 4.1s | Đệm trước: 0.3s | Đã dịch: 0.3s
      (giải mã 0.39x ⇒ 0.00x, PCM/byte 0.000, câu 1)      ← lặp lại y hệt suốt 40 giây
[BS][Diag][status] playhead=15.67s | ready=0.33s | mốc-đã-xử-lý=16.00s | fragments=13 decoded=2
```

Hai con số quyết định: **`mốc-đã-xử-lý=16.00s` đứng im** và **`decoded=2` không bao giờ tăng**.
`giải mã 0.00x` + `PCM/byte 0.000` ⇒ backend không nhận thêm byte nào — **và cũng không thể**,
vì trình phát đã đệm sẵn 30 s phía trước nên khi bị tạm dừng nó **không `appendBuffer` nữa**.

Đây là **deadlock**: video chờ backend, backend chờ `appendBuffer`, mà `appendBuffer` chỉ xảy ra
khi video phát. Đường duy nhất phá được là audio **đã có sẵn trong bộ đệm/lệnh replay** phải được
giải mã cho hết. Nó đã được gửi (16,69 MB, phủ 0→40 s) nhưng chỉ ra được PCM tới **16,00 s**.

---

## 2. Điều KHÔNG phải nguyên nhân (đã loại trừ bằng đo)

Dựng lại kịch bản bằng fMP4 thật (PyAV ghi ra, 40 s, mảnh 4 s) rồi nạp vào `StreamDemuxer` với
`reset(epoch=2, min_pts=8.88)` — **mọi biến thể đều cho đủ ~31 s PCM**:

| Kịch bản | Kết quả |
|---|---|
| fMP4 audio-only, mảnh tới theo thứ tự | 31,02 s ✅ |
| fMP4 **muxed** (video+audio), như log thật | 30,42 s ✅ |
| Ép `_trim()` chạy (hạ trần byte) | 31,02 s ✅ |
| **1 mảnh hỏng giữa chừng** | 29,47 s ✅ (tự phục hồi qua đường từng-fragment) |

⇒ Ghép nối byte, giải mã muxed, `_trim`, đường dự phòng từng-fragment, và `_anchor_to_browser_axis`
**đều không làm mất audio** trong các kịch bản đó. Việc loại trừ này quan trọng: nó chỉ ra rằng
audio không hề bị mất trong lúc giải mã, mà bị **vứt bỏ có chủ đích** bởi logic tiến mốc.

---

## 3. Nguyên nhân gốc — hai khiếm khuyết trong `core/stream_demuxer.py`

### 3.1 `_last_pts` bị đẩy qua vùng audio CHƯA HỀ PHÁT (nghiêm trọng)

`_last_pts` là **mốc PCM ĐÃ PHÁT**. Mọi frame có `pts < _last_pts` bị lọc vĩnh viễn
(`_decode_new`, nhánh `if pts_media < self._last_pts - 1e-6: continue`).

Cuối `_decode_fragments_individually` có:

```python
# (TRƯỚC KHI SỬA)
if not out and recovered_frontier > float("-inf"):
    self._last_pts = max(self._last_pts, recovered_frontier + 0.02)
```

`recovered_frontier` = `tfdt` của mảnh **CUỐI** trong tập vừa xét. Ý định ban đầu là "đừng giải mã
lại đúng 200 mảnh đó mãi" — nhưng nó **ghi nhầm "đã quét" thành "đã phát"**.

Với log thật: bộ đệm giữ 0→40 s, `_last_pts` = 16,07 s. Đường dự phòng được gọi (vì
`_last_pts < last_tfdt − 2`), ra **rỗng**, và đẩy `_last_pts` thẳng tới **36,02 s**.

Hệ quả: toàn bộ audio **16,07 → 36,02 s bị lọc vĩnh viễn**, dù byte vẫn còn nguyên trong bộ đệm.
`chunks_decoded` kẹt ở 2, `ready_until_pts` kẹt ở 16,00 s ⇒ **đúng triệu chứng log**.

Đo được (test hồi quy, trước khi sửa):

```
AssertionError: `_last_pts` nhảy 16.00s → 36.02s dù KHÔNG phát ra PCM nào
                ⇒ audio 16.0–36.0s bị lọc vĩnh viễn
```

### 3.2 Kết quả đường giải mã CHUẨN bị vứt

```python
# (TRƯỚC KHI SỬA)
per_frag = self._decode_fragments_individually()
if per_frag:
    ...
    return per_frag        # ← thay thế, không nối thêm ⇒ vứt `chunks`
```

`per_frag` chỉ chứa frame **nằm sau** `_last_pts` (đọc sau khi đường chuẩn đã chạy), nên hai tập
**rời nhau** — đúng ra phải trả về hợp của chúng.

Đo được (trước khi sửa):

```
AssertionError: nhánh dự phòng làm MẤT PCM: đường chuẩn phát 31.11s
                nhưng kết quả trả về chỉ còn 0.50s
```

---

## 4. Cách sửa

| # | Sửa | Ở đâu |
|---|---|---|
| 1 | **Tách sàn QUÉT khỏi mốc ĐÃ PHÁT**: thêm `_indiv_scan_floor` (nghĩa "đã thử, không ra PCM"). Mốc **chọn mảnh** = `max(_last_pts, _indiv_scan_floor)`; mốc **lọc frame** vẫn là `_last_pts`. Khi đường dự phòng ra rỗng, **chỉ nâng sàn quét** — không bao giờ đụng `_last_pts`. | `_decode_fragments_individually` |
| 2 | **Nối thêm thay vì thay thế**: `return chunks + per_frag`. | `_decode_new` |

Sàn quét được reset ở `reset()` và khi đổi epoch (nguồn mới ⇒ byte mới).

---

## 5. Kiểm chứng

**Test hồi quy mới** `backend/tests/test_77_demuxer_no_audio_loss.py` — cả hai test **đỏ trước khi
sửa** với đúng thông điệp ở §3, **xanh sau khi sửa**.

- `test_fallback_khong_ra_pcm_thi_khong_duoc_day_frontier`
- `test_ket_qua_fallback_khong_duoc_vut_pcm_cua_duong_chuan`

**Bộ test hiện có**: `test_50_lookahead_demuxer.py` (17 test) vẫn xanh — không hồi quy.

---

## 6. Bài học

1. **"Đã quét" ≠ "đã phát".** Hai khái niệm này bị trộn vào một biến `_last_pts` — và vì frontier
   là **cơ chế lọc vĩnh viễn**, ghi nhầm một lần là mất audio không thể phục hồi.
2. **Nhánh dự phòng phải NỐI THÊM, không được THAY THẾ.** Một `return` thay vì `return a + b`
   vứt đi 31 s PCM mà không một dòng log nào.
3. **Loại trừ trước khi kết luận.** Bốn kịch bản tái hiện đều cho thấy tầng giải mã *không* mất
   audio; chính việc đó chỉ ra thủ phạm nằm ở logic tiến mốc, không phải ở giải mã.
4. **Deadlock cần được nhìn như deadlock.** `giải mã 0.00x` không có nghĩa "backend chậm" mà có
   nghĩa "backend sẽ không bao giờ nhận thêm gì nữa" — vì nguồn cấp dữ liệu phụ thuộc chính
   thứ đang bị chặn.

---

## 7. Hồi quy phát hiện thêm khi chạy lại (cùng ngày) — timescale của `tfdt`

Sau khi sửa §4, phiên chạy **thành công** (video phát, phụ đề ra, `decoded` tăng đều
4 → 11 → 14 → 16 → 19 → 21 → 24). Nhưng log mới lộ ra vấn đề thứ ba, qua chính dòng cảnh báo
vừa thêm ở §4 — nó bắn **ở MỌI mảnh**:

```
[WARNING] Giải mã từng-fragment không ra PCM cho vùng 12.01s→15.00s (1 mảnh) …
[WARNING] Giải mã từng-fragment không ra PCM cho vùng 16.00s→22.50s (1 mảnh) …
[WARNING] Giải mã từng-fragment không ra PCM cho vùng 20.01s→30.00s (2 mảnh) …
[WARNING] Giải mã từng-fragment không ra PCM cho vùng 100.01s→180.00s (2 mảnh) …
```

### Bằng chứng định lượng

- Trần bộ đệm ghép nối là **8 MB**, mảnh ≈ 1,42 MB ⇒ bộ đệm chỉ giữ được **~5–6 mảnh ≈ 22 s**.
- `_last_pts` = 100,01 s (đúng, vì lấy từ PTS frame thật).
- Nhưng `recovered_frontier` = **180,00 s** ⇒ bộ đệm được cho là kéo dài tới 180 s, **cách mốc
  đã phát 80 s** — trong khi thực tế chỉ ~22 s.

Kết luận: **`tfdt` bị quy ra giây bằng một timescale NHỎ HƠN timescale thật.**

### Nguyên nhân

`_fragment_byte_ranges` và `_last_audio_tfdt_sec` đều dùng:

```python
scale = 1.0 / self._audio_sample_rate()      # (TRƯỚC KHI SỬA)
```

tức **giả định timescale của track == sample rate**. Hai đại lượng đó chỉ tình cờ bằng nhau với
AAC thường do FFmpeg đóng gói. Với **HE-AAC/SBR** (sample rate lõi báo bằng một nửa timescale —
ví dụ lõi 24 kHz, timescale 48 kHz) và với nhiều bộ đóng gói tuỳ biến, chúng **khác nhau**.

### Hậu quả (ba tầng)

1. `tfdt` thổi phồng ⇒ điều kiện "còn audio phía trước chưa lấy" (`_last_pts < last_tfdt − 2`)
   **luôn đúng** ⇒ đường giải mã từng-fragment chạy trên **mọi** mảnh, mở container cho từng
   mảnh mà không bao giờ ra PCM — **tốn CPU vô ích trên mỗi mảnh**.
2. **Sàn quét bị đẩy vượt xa mốc đã phát** (tới 180 s trong khi mới phát 100 s) ⇒ cơ chế cứu hộ
   **tự vô hiệu hoá**: nếu đường giải mã nối-liền hỏng ở vùng 100–180 s thì nhánh dự phòng đã
   "quét qua" vùng đó rồi và sẽ không bao giờ xét lại.
3. Cảnh báo nhiễu ở mọi mảnh, che mất cảnh báo thật.

### Cách sửa

Thêm `_audio_timescale()` đọc **timescale thật** của track từ container
(`stream.time_base` của PyAV, tức box `mdhd`) thay vì suy từ sample rate; chỉ dự phòng về
sample rate khi không đọc được. Cache timescale/sample rate được **xoá khi init segment đổi**
(`_set_init`) — nếu không, đổi nguồn giữa phiên sẽ quy đổi sai đơn vị.

**Test hồi quy:**

- `test_quy_doi_tfdt_theo_timescale_cua_track` — timescale phải lấy từ container, và nhân đôi
  timescale phải làm số giây giảm một nửa.
- `test_doi_init_thi_bo_cache_timescale` — init mới phải xoá cache.

### Mức độ chắc chắn

Hệ số lệch được **suy từ log** (`recovered_frontier` 180 s so với bộ đệm thật ~22 s quanh
`_last_pts` 100 s), **không đo trực tiếp** bằng cách đọc `mdhd` của luồng xhamster. Nếu chạy lại
mà cảnh báo §7 vẫn bắn ở mọi mảnh, thì giả thuyết timescale chưa đúng và cần đọc `tfdt`/`mdhd`
thô của luồng đó để chốt.
