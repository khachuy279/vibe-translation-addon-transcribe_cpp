# 23 — KẾT QUẢ M1: KHẢO SÁT CHẶN BẮT MSE TRÊN 6 TRANG THẬT

> **Loại tài liệu:** kết quả đo (M1 của `report/audit/22_MSE_INTERCEPT_FEASIBILITY.md` §8).
> **Ngày:** 2026-09 · **Dữ liệu thô:** `report/audit/m1_logs/*.json` (**15 phiên / 6 host** + 3 khung phụ, xuất bởi `scratch/m1_mse/m1_mse_probe.user.js`).
> **Công cụ:** `scratch/m1_mse/` — probe chỉ ghi log, không sửa `extension_firefox/` hay `backend/`.
> **Nhãn:** `[VERIFIED-M1]` = đọc trực tiếp từ JSON đo được; `[VERIFIED-REPO]`/`[VERIFIED-EXT]` như tài liệu 22; `[NOT VERIFIED]` = chưa kiểm chứng.
>
> **Lịch sử bản:** v1 (6 phiên) → v2 (thêm 3 phiên có `%<ng`) → **v3 (thêm 5 phiên probe 1.3: đủ 6 host, có phiên 2×, tiến tới 15 phiên)**.
> v2 **sửa một kết luận sai của v1** (§4.1); v3 **sửa một nhận định sai của v2** (§4.1b) và bổ sung §4.3/§4.6.

---

## 0. KẾT LUẬN ĐIỀU HÀNH

1. **M1 ĐẠT.** Trên **9 phiên / 6 host**: hook MAIN world bắt được `appendBuffer`, phân biệt được buffer audio/video, **không phiên nào có DRM**, không phiên nào giấu MSE trong Worker, và **`L` (dẫn trước buffer) luôn lớn hơn ngưỡng 3,5 s** ⇒ lợi ích độ trễ của đề xuất là **thật và đo được**. Theo §10.1, đề xuất **không bị phủ định**.
2. **Trả lời được 3 câu hỏi mở:** Q2 (YouTube free **không** CENC), Q4 (0/6 host dùng MSE-in-Worker), Q7 (userscript `@grant none` **chạy được ở MAIN world** — tự chứng minh bằng việc bắt được lời gọi thật của trang).
3. **Phát hiện làm đổi thiết kế M2:** **3/6 host không append nguyên một phân đoạn.** Trang thì tách `moof` và `mdat` thành hai lời gọi, YouTube thì cắt phân đoạn thành hàng chục mảnh nhỏ (trung vị ~9,6 KB) nằm giữa box. Hệ quả: **demux theo từng lời gọi `appendBuffer` là bất khả thi**; biến thể §7.4 của tài liệu 22 (offscreen decode) cũng phải sửa. Đề xuất thay thế ở §5: **mirror-MSE** — không parse byte, chỉ chép lại chuỗi byte y nguyên sang một `MediaSource` ẩn.
4. **`belowRequiredPct` (chỉ số quyết định) rất thấp:** 3 phiên chạy probe 1.1 cho **0,0% · 1,5% · 0,0%** số mẫu dưới ngưỡng, và histogram cho thấy toàn bộ số mẫu thấp nằm ở **đoạn khởi động đổ buffer**, không phải steady-state. Riêng YouTube: 14/926 mẫu (1,5%), tất cả trong các khoảng `<1s`, `1–2s`, `2–3,5s` — trong khi 509/926 mẫu nằm ở khoảng `120–300s`.
5. **Rủi ro R3/R4/R5 của tài liệu 22 nay có bằng chứng thật:** đã thấy `SourceBuffer.remove()` **12 lần** trong một phiên (bilibili), đã thấy init segment lặp lại **8 lần/track** (phimsrv, ABR), đã thấy `timestampOffset = −1` và `−1,147` (phimsrv, hai video khác nhau) và timeline có khoảng trống (phimsrv).
6. **Một lỗi của chính probe đã được tìm ra và sửa** (v1.2): `page.url` bị chụp một lần lúc `document_start`, nên với SPA (YouTube, bilibili) URL ghi lại là URL lúc tải trang, **không phải** URL lúc đo. Xem §4.1 — bản v1 đã từ đó rút ra một kết luận sai.

---

## 1. BẢNG KẾT QUẢ

Sinh bằng `python scratch/m1_mse/m1_analyze.py` (10 phiên; `—` = phiên chạy bằng probe cũ, chưa có trường đó):

| host | giờ | MSE | a/v | rate | appends | MB | DRM | n(L) | %<ng | L p5 | L p50 | L p95 | phán quyết |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| s.chichvl.blog | 09:12 | có | 1/1 | — | 466 | 163,3 | không | 188 | — | 228,08 | **302,65** | 308,02 | PASS |
| s.chichvl.blog | **10:16** | có | 1/1 | 1 | 266 | 115,3 | không | 301 | **0,0%** | 299,88 | **302,69** | 307,92 | PASS |
| api.phimsrv.com | 09:15 | có | 1/1 | — | 222 | 151,9 | không | 66 | — | 11,61 | **75,24** | 139,55 | PASS |
| api.phimsrv.com | 09:37 | có | 1/1 | — | 72 | 29,9 | không | 458 | **0,0%** | 137,21 | **139,00** | 141,54 | PASS |
| play.vlstream.net | 09:11 | có | 1/1 | — | 22 | 31,6 | không | 129 | — | 25,35 | **29,33** | 34,33 | PASS |
| play.vlstream.net | **10:14** | có | 1/1 | 1 | 34 | 25,0 | không | 500 | **0,0%** | 25,29 | **29,63** | 34,15 | PASS |
| www.xvideos.com | 09:15 | có | 1/1 | — | 30 | 15,4 | không | 98 | — | 30,28 | **35,79** | 39,30 | PASS |
| www.xvideos.com | **10:12** | có | 1/1 | 1 | 48 | 40,1 | không | 490 | **0,0%** | 30,29 | **34,89** | 39,28 | PASS |
| www.bilibili.tv | 09:11 | có | 1/1 | — | 44 | 2,5 | không | 107 | — | 19,71 | **24,91** | 28,75 | PASS |
| www.bilibili.tv | 09:33 | có | 1/1 | — | 50 | 3,2 | không | 501 | **0,0%** | 20,09 | **25,29** | 29,01 | PASS |
| www.youtube.com | 09:09 | có | 3/3 | — | 2605 | 26,0 | không | 327 | — | 3,90 | **42,58** | 78,01 | PASS |
| www.youtube.com | 09:31 | có | 3/3 | — | 2278 | 21,0 | không | 926 | **1,5%** | 11,66 | **120,77** | 128,35 | PASS |
| www.youtube.com | 09:52 | có | 3/3 | — | 2752 | 18,3 | không | 502 | **2,8%** | 6,40 | **87,56** | 125,51 | PASS |
| www.youtube.com | **10:07** | có | 1/1 | 1 | 2213 | 18,5 | không | 593 | **0,0%** | 38,28 | **102,14** | 127,32 | PASS |
| **www.youtube.com** | **10:09** | có | 1/1 | **2** | 101 | 19,3 | không | 451 | **0,0%** | 59,28 | **61,77** | 64,27 | PASS |
| | | | | | **11 203** | **~664** | | **5 637** | **28 mẫu** | | | | **15/15 PASS** |

Tổng: **11 203 lời gọi `appendBuffer`** (đếm theo `signals`; con số thật cao hơn một chút — xem §4.7),
**15/15 phiên PASS**, **`errors = 0` trên cả 18 khung**.
`L` = `buffered.end − currentTime`, đã chia `playbackRate`. `%<ng` = tỉ lệ mẫu dưới ngưỡng 3,5 s.
Ngoài 15 phiên còn **3 khung phụ `N/A`** (iframe `accounts.google.com` không có media) — không tính là thất bại, xem §2 hàng 12.

**Đọc bảng:** `L` p50 cách ngưỡng 7–86×. Chỉ YouTube có mẫu dưới ngưỡng (1,5%) — và histogram chứng minh đó là transient khởi động (§4.2).

**Histogram YouTube 09:31** `[VERIFIED-M1]` — hình dạng "ramp" đúng như dự đoán cho giai đoạn đổ buffer:

| Khoảng L | `<1s` | `1–2s` | `2–3,5s` | `3,5–5s` | `5–10s` | `10–30s` | `30–60s` | `60–120s` | `120–300s` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| số mẫu | 4 | 4 | 6 | 6 | 20 | 20 | 84 | 273 | **509** |

---

## 2. NHỮNG GÌ ĐÃ ĐƯỢC CHỨNG MINH `[VERIFIED-M1]`

| # | Khẳng định | Bằng chứng |
| :-: | :--- | :--- |
| 1 | Hook MAIN world bắt được lời gọi thật của trang | 3 389 append + byte fMP4/WebM hợp lệ (`head` khớp `ftyp`/`moof`/`mdat`) ở cả 6 host; probe ở ISOLATED world sẽ **không** thấy gì |
| 2 | Phân biệt audio vs video bằng `mimeType` (§2.4) | Mọi trang đều có cặp `audio/mp4;codecs=mp4a.40.2` ↔ `video/mp4;codecs=avc1…/av01…`; bilibili còn **đảo thứ tự** (sb1=video, sb2=audio) — phân loại theo `mimeType` vẫn đúng |
| 3 | Đọc được cấu trúc fMP4 | `hdlr=soun/vide`, `tfhd.track_ID=2/1`, `stsd→mp4a/avc1/av01/hev1`, `mdhd.timescale=44100/90000/30000/16000` |
| 4 | `tfdt` → giây chính xác | Δ giữa hai `tfdt` liên tiếp khớp độ dài phân đoạn: 4,99 s / 5,00 s (chichvl), 6,25 s (bilibili), 6,64 s (YouTube), 10,01 s (vlstream, xvideos) |
| 5 | **Không có DRM** trên cả 6 trang | `drm.cenc=false`, `markers=[]`, `encryptedEvents=0`; quét cả box walk (`senc/saiz/saio/pssh/tenc/encv/enca`) lẫn quét thô 256 KB đầu |
| 6 | **Không có MSE-in-Worker** | getter `MediaSource.handle` không bao giờ bị đọc (`mseInWorkerSuspect=false`) ở cả 6 |
| 7 | Chi phí parse của probe không đáng kể | `parseMs` = 0–3 ms/lần append, kể cả chunk 2,9 MB ⇒ Q3 không còn là ẩn số lớn: **quét cấu trúc box không phải nút thắt** |
| 8 | `timestampOffset` được dùng thật | phimsrv: `timestampOffset = −1,1469333…` trên **cả hai** SourceBuffer ⇒ hook setter (§2.5) là bắt buộc, và byte→thời gian **phải** đối chiếu `buffered` |
| 9 | `remove()` / ABR được dùng thật | bilibili: `remove×4`; phimsrv: **7 init segment**/track (re-init do đổi chất lượng), `addSourceBuffer×2` |
| 10 | Chỉ số `buffered` là nguồn sự thật | Ở phimsrv, công thức `media_ts + timestampOffset` cho ~0,33 s trong khi `buffered.start = 0` ⇒ chỉ `buffered` mới đáng tin (§2.2) |
| 11 | **Tổng hợp iframe chạy được** | Phiên 09:52 có `frames[]` chứa báo cáo của một iframe **khác origin** `accounts.google.com` ⇒ đường `postMessage` từ iframe lên top hoạt động. Cần thiết vì nhiều player nhúng nằm trong iframe |
| 12 | **Đã sửa một lỗi nhiễu của công cụ** | Khung không có `<video>`/`<audio>` nào (iframe tiện ích/đăng nhập) trước đây bị gắn `HARD-FAIL` "0 append" ⇒ nay là `N/A`. Sửa ở cả probe (v1.3, `signals.mediaElementsSeen`) và `m1_analyze.py` (chuẩn hoá lại cả dữ liệu cũ) |
| 13 | **2× ĐẠT** | Cùng một video, 1× → 2×: ngân sách wall-clock 102,14 s → **61,77 s**, `%<ng` = 0,0% ⇒ biên 17,6× (§4.3) |
| 14 | **Timeline đứt quãng thật** | chichvl 10:16: `buffered = [0–20] ∪ [415–795]` với `currentTime = 478` ⇒ khoảng trống 395 s (§4.4) |

**Trả lời câu hỏi mở của tài liệu 22:**

| # | Câu hỏi | Trả lời từ M1 |
| :-: | :--- | :--- |
| Q1 | `L` phân bố thế nào? | p50 **24,9–302,7 s**; số mẫu dưới ngưỡng **0–14 mẫu/phiên**, chỉ ở giai đoạn khởi động và chỉ ở 2/15 phiên ⇒ **ĐẠT** |
| Q1b | Ở **2×** thì sao? | Đo trên cùng một video: ngân sách wall-clock 102,14 s (1×) → **61,77 s** (2×), 0% mẫu dưới ngưỡng ⇒ **ĐẠT**, biên 17,6× (§4.3) |
| Q1c | Chế độ "preload cả file" có làm PASS giả không? | **Có, một phần** — chichvl/phimsrv nạp gần hết file nên `L` = thời lượng còn lại. Vẫn là dẫn trước thật, nhưng **không** đại diện cho streaming thông thường; các phiên VOD/HLS khác (vlstream 29,6 s · xvideos 34,9 s · bilibili 25,3 s) mới là con số tiêu biểu |
| Q2 | Byte YouTube có CENC? | **Không** — không `senc/saiz/saio/pssh`, không `encrypted`, ở cả 2 phiên YouTube |
| Q3 | Chi phí demux? | Quét cấu trúc: **0–3 ms/lần**. Chi phí *giải mã* PCM vẫn `[NOT VERIFIED]` |
| Q4 | Bao nhiêu % dùng MSE-in-Worker? | **0/6** (trên tập mẫu này) |
| Q7 | Firefox có cho vá prototype? | **Có** — userscript `@grant none` chạy ở MAIN world và hook có hiệu lực thật |

---

## 3. PHÁT HIỆN QUAN TRỌNG NHẤT: CHUỖI BYTE KHÔNG ĐƯỢC CHIA THEO PHÂN ĐOẠN

Đây là dữ liệu **phủ định một giả định ngầm** của §2.2/§2.3 (tài liệu 22): rằng mỗi lời gọi `appendBuffer` = một phân đoạn `moof+mdat` trọn vẹn.

| Trang | Dạng lời gọi | Bằng chứng (`appendEventsTail`) |
| :--- | :--- | :--- |
| api.phimsrv.com (×2) | ✅ trọn phân đoạn | 09:15: 30/30 append bắt đầu bằng box `moof`, `size` khớp `len`; 09:37: 66 media + 8 init, `unparsed=0` |
| play.vlstream.net | ✅ trọn phân đoạn | 10 media `moof` + 1 init `ftyp` |
| www.bilibili.tv (×2) | ✅ trọn phân đoạn | 09:11: 22 media `moof` + 1 init; 09:33: 30–31 media + 1 init, `unparsed=0` |
| s.chichvl.blog | ⚠️ **tách `moof` / `mdat`** | 15 append `moof` + **15 append `mdat`-only** (`head=00013842 6d646174`, `len=79938`) |
| www.xvideos.com | ⚠️ **tách `moof` / `mdat`** | 6 `moof` + **6 `mdat`-only** |
| www.youtube.com (×2) | ❌ **cắt nhỏ giữa box** | audio: `head=1f43b675` (EBML *Cluster*, **chỉ 1/15 mảnh**) lẫn `81c6aac4`, `b17c4fcb` (giữa element); video: `47007eac`, `d6547c69`… — `size` giả định vượt xa `len` thật ⇒ **mảnh vụn giữa box**. 2 198 append / ~38 phân đoạn (09:09) và 1 739 / ~67 (09:31) ⇒ **~26–57 mảnh mỗi phân đoạn**, trung vị **9,6 KB** |

**Hệ quả kỹ thuật:**

1. Bộ demux "một phân đoạn cho mỗi `appendBuffer`" **sẽ hỏng** ở 3/6 trang. Muốn tự demux thì phải dựng **bộ phân tích luồng có trạng thái** (đệm byte dở, ghép lại theo `size`, xử lý `moof` ở append này và `mdat` ở append sau) — tức là tăng độ phức tạp của R1 lên đáng kể, đúng vào chỗ tài liệu 22 lo ngại nhất.
2. Đường "offscreen decode" ở §7.4 (ghép `init + media` thành Blob rồi cho `HTMLMediaElement` ẩn phát) cũng phải sửa: phải **tự ghép lại** phân đoạn từ nhiều mảnh trước khi tạo Blob. Làm được, nhưng lại quay về bài toán ghép luồng.
3. Tin tốt: YouTube audio là **WebM/Opus tách riêng** (`audio/webm; codecs="opus"`), còn video là **AV1 fMP4** ⇒ chỉ cần quan tâm **một** luồng audio, không phải tách `traf` khỏi segment audio+video xen kẽ (§2.3 trường hợp khó nhất **không** xuất hiện trong mẫu này).

---

## 4. NHỮNG GÌ **CHƯA** CHỨNG MINH ĐƯỢC (giới hạn của tập dữ liệu)

### 4.1 ⚠️ SỬA KẾT LUẬN SAI CỦA BẢN v1 — probe ghi URL lúc `document_start`, không phải lúc đo

**Bản v1 của tài liệu này kết luận YouTube "đo sai đối tượng: đo trên trang chủ, chỉ có player preview". Kết luận đó SAI.**

Nguyên nhân: probe chụp `location.href` **một lần** lúc `document_start` và không cập nhật. YouTube là **SPA** — nó đổi URL sang `/watch?v=…` mà không tải lại trang, nên trường `page.url` giữ nguyên URL lúc mở tab (`/?themeRefresh=1`, `/?app=desktop&hl=vi`). Dấu hiệu mâu thuẫn nằm ngay trong file JSON:

| Trường | Giá trị (phiên 09:31) | Đọc lúc nào |
| :--- | :--- | :--- |
| `page.url` | `https://www.youtube.com/?app=desktop&hl=vi` | **lúc `document_start`** ← nguồn của kết luận sai |
| `page.title` | `'The Big Bang Theory' but Everyone Is on Their Lunch Break - YouTube` | **lúc bấm Lưu JSON** |
| `sources[sb6].buffered` | `[{start: 0, end: 352.352}]` — **352 s media** | liên tục |
| `signals.appends` | 2 278 | liên tục |

352 giây media được nạp **không thể** là player preview (~20 s). Vậy cả hai phiên YouTube đều là **phiên phát video thật**; chỉ nhãn URL là cũ.

**Sửa trong probe v1.2 `[VERIFIED-REPO]`:** `page.url` đọc lại lúc dựng báo cáo, thêm `page.urlInitial` (URL lúc tải) và `page.navigations` + `page.navUrls` (đếm số lần URL đổi, cập nhật trong vòng lấy mẫu 250 ms). `m1_analyze.py` in thêm hậu tố `⇄n` cạnh host khi phát hiện SPA. Các JSON v1.0/v1.1 vẫn thiếu trường này — **không** dùng `page.url` của chúng để suy ra "đã đo trang nào".

**Bài học phương pháp luận:** một trường metadata chụp sai thời điểm đã tạo ra một kết luận sai về *chính tập dữ liệu*. Các trường phái sinh từ `location`/`document.title` phải được đọc **tại thời điểm xuất báo cáo**, không phải lúc cài hook. Ghi vào §4.1 này để không lặp lại ở M2/M3.

**Xác nhận trực tiếp bằng phiên 09:52 (probe v1.2)** `[VERIFIED-M1]` — giả thuyết SPA được chứng minh bằng dữ liệu, không còn là suy luận:

| Trường | Giá trị |
| :--- | :--- |
| `page.urlInitial` | `https://www.youtube.com/?app=desktop&hl=vi` |
| `page.url` (lúc lưu) | `https://www.youtube.com/watch?v=-nuvG8atc7E` |
| `page.navigations` | **1** |
| `page.navUrls` | `["https://www.youtube.com/watch?v=-nuvG8atc7E"]` |
| `sources[sb6].buffered` | `[{start: 0, end: 237.446}]` — video 1 giờ, đã nạp 237 s |

⇒ Cùng một tab: URL lúc tải là trang chủ, URL lúc đo là trang watch. **Probe v1.2 hoạt động đúng.**

### 4.1b ⚠️ SỬA NHẬN ĐỊNH TRƯỚC: số mẫu dưới ngưỡng **không** phải hằng số

Bản trước của tài liệu này ghi: *"transient khởi động là hằng số, không phải hiện tượng ngẫu nhiên"*, dựa trên việc **2 phiên YouTube đều cho đúng 14 mẫu** dưới ngưỡng với histogram y hệt (`4 + 4 + 6`).

**5 phiên mới bác bỏ nhận định đó** `[VERIFIED-M1]`: cả 5 đều cho **0 mẫu dưới ngưỡng**, và histogram của chúng **không có bucket nào thấp hơn**:

| Phiên | n(L) | mẫu < 3,5 s | Bucket thấp nhất có mẫu | ⇒ min `L` của cả phiên |
| :--- | ---: | ---: | :--- | ---: |
| youtube 10:07 | 593 | **0** | `10–30s` (2 mẫu) | ≥ 10 s |
| youtube 10:09 (2×) | 451 | **0** | `30–60s` (70 mẫu) | ≥ 30 s |
| chichvl 10:16 | 301 | **0** | `60–120s` (1 mẫu) | ≥ 60 s |
| vlstream 10:14 | 500 | **0** | `10–30s` (268 mẫu) | ≥ 10 s |
| xvideos 10:12 | 490 | **0** | `10–30s` (10 mẫu) | ≥ 10 s |

Vì histogram bao **toàn bộ** mẫu của phiên, đây là bằng chứng cho cả phiên, không phải suy từ đuôi.

**Giải thích:** mẫu thấp chỉ xuất hiện khi buffer đang được **đổ từ trạng thái rỗng/khởi động**. Nếu phiên đo bắt đầu từ **vị trí resume** (hoặc việc lấy mẫu bắt đầu sau khi buffer đã dẫn trước), thì **không bao giờ** có mẫu dưới ngưỡng. Cả 5 phiên mới đều có `currentTime` ở vị trí resume ngay từ mẫu đầu tiên giữ được: 139 s · 366 s · 478 s · 110 s · 109 s.

**Kết luận đúng cho M1:** trong **15/15 phiên**, `L` chưa bao giờ ở dưới ngưỡng một cách dai dẳng. Toàn bộ số mẫu dưới ngưỡng từng quan sát được là **≤14 mẫu (~3,5 s) ngay lúc khởi động**, và chỉ xuất hiện ở **2/15** phiên. Đây vẫn là tin tốt — nhưng là kết luận **có điều kiện**, không phải hằng số.

**Còn lại thật sự phải đo lại ở YouTube:** chỉ là *phiên watch có chủ đích* (mở `/watch?v=…` trực tiếp, không tua, ≥120 s) để có số sạch theo kịch bản — không phải vì dữ liệu hiện tại vô nghĩa.

### 4.2 `p5` một mình là chỉ số dễ gây hiểu sai — nay đã có số thay thế

`p5` trộn giai đoạn khởi động đổ buffer vào cùng phân bố với steady-state, nên nó **thổi phồng mức xấu** (YouTube v1: `p5 = 3,90 s` nghe như sát nguy hiểm, trong khi steady-state là 120 s).

→ Probe **v1.1** thêm `lead.belowRequiredPct`, `lead.p1`, `lead.hist`. **3 phiên mới đã có số này** `[VERIFIED-M1]`:

| Phiên | n(L) | % dưới ngưỡng | p1 | p50 | Histogram |
| :--- | ---: | ---: | ---: | ---: | :--- |
| api.phimsrv.com 09:37 | 458 | **0,0%** | 137,07 | 139,00 | toàn bộ ở `120–300s` |
| www.bilibili.tv 09:33 | 501 | **0,0%** | 19,55 | 25,29 | toàn bộ ở `10–30s` |
| www.youtube.com 09:31 | 926 | **1,5%** | 2,40 | 120,77 | ramp khởi động: 14 mẫu `<3,5s`, 509 mẫu `120–300s` |

⇒ Ở cả 3 phiên, **toàn bộ** số mẫu dưới ngưỡng nằm ở đuôi thấp của một ramp khởi động, không có dấu hiệu "đói buffer" lặp lại ở steady-state. Đây là bằng chứng **mạnh hơn** p5 và là con số nên trích dẫn.

### 4.3 ✅ ĐÃ ĐO Ở 2× — ĐẠT, và đây là phép A/B sạch nhất của cả M1

Phiên **10:09** chạy đúng **cùng video** với phiên **10:07** (`youtube.com/watch?v=-nuvG8atc7E`), chỉ khác tốc độ phát ⇒ so sánh trực tiếp được `[VERIFIED-M1]`:

| | 10:07 (1×) | 10:09 (2×) | Nhận xét |
| :--- | ---: | ---: | :--- |
| `lead.rates` | `[{1, 593}]` | `[{2, 451}]` | 2× áp dụng cho **toàn bộ** 451 mẫu, không trộn |
| `rawP50` (`L`, giây **media**) | 102,14 | 123,54 | ở 2× player còn nạp **xa hơn** tính theo media |
| `p50` (`L/rate`, giây **wall-clock**) | **102,14** | **61,77** | **×0,60** — sát dự đoán ×0,5 |
| p5 / p95 (wall-clock) | 38,28 / 127,32 | 59,28 / 64,27 | ở 2× phân bố **hẹp hơn** và p5 **cao hơn** |
| `%<ng` | **0,0%** | **0,0%** | không mẫu nào dưới 3,5 s |
| Biên so với ngưỡng | 29× | **17,6×** | còn rất dư |
| Buffer đạt được (media) | video 259 s / audio 280 s | video 500 s / audio 520 s | player nạp ~4,6× realtime ⇒ ABR/theo kịp |

**Kết luận:** ở 2×, ngân sách wall-clock để chạy trước giảm còn **~62 s** — vẫn lớn hơn ngưỡng 3,5 s **17 lần**. Điều kiện §5.3 của tài liệu 22 ("2× vẫn khả thi nhưng phải giải mã nhanh hơn realtime") được **xác nhận bằng số**.

**Lưu ý phạm vi:** phép đo này chứng minh **ngân sách dẫn trước** ở 2×. Nó **không** chứng minh backend theo kịp 2× — xem ghi chú ở §6.

### 4.4 Seek có thật, và timeline có khoảng trống lớn

Phiên chichvl 10:16 có `buffered = [0–20 s] ∪ [415–795 s]` với `currentTime = 478 s` `[VERIFIED-M1]` ⇒ timeline **đứt quãng 395 s** giữa hai dải (resume từ vị trí đã xem, rồi player nạp tiếp). Đây là dạng dữ liệu mà R3/R5 của tài liệu 22 lo: mọi ánh xạ byte→thời gian phải dựa trên `buffered`, và "kế hoạch chạy trước" phải bị **vô hiệu hoá** khi gặp khoảng trống.

### 4.5 Chưa quan sát được hành vi "chạy trước" thật

M1 chỉ chứng minh **byte đến sớm**. Nó **không** chứng minh ta lấy được PCM từ byte đó (§7.4 chưa làm), cũng chưa chứng minh `captureTimestamp` theo mốc media không phá VAD/ASR (Q5).

### 4.6 Nghẽn thật của pipeline vẫn là dịch, không phải ASR

Không đổi so với tài liệu 22: `translation ~430 ms` > `ASR ~100 ms`. M1 **không** ảnh hưởng kết luận này; ngân sách 3,5 s trong công thức đã bao gồm nó.

### 4.7 ⚠️ Một bất nhất của công cụ: `signals.appends` và `sources[].appends` lệch nhau

Trong **7/15** file JSON, tổng `sources[].appends` **lớn hơn** `signals.appends` — ví dụ phiên YouTube 10:09: `signals = 101` nhưng tổng theo SourceBuffer `= 2 457` (lệch **2 356**) `[VERIFIED-M1]`.

**Nguyên nhân (đã tìm ra VÀ đã được người đo xác nhận):** nút **Reset** zero `signals.*` nhưng **không** zero bộ đếm của từng SourceBuffer ⇒ hai con số trong cùng một file kể hai câu chuyện khác nhau. Người đo **xác nhận đã bấm Reset một cách có chủ đích** để bắt đầu một cửa sổ đo mới (loại bỏ nhiễu từ player preview ở trang chủ YouTube trước khi đo video thật). Đây là **cách dùng đúng**, không phải thao tác sai — lỗi nằm ở probe khi không zero bộ đếm per-source theo. Các mẫu `L` **không** bị ảnh hưởng (leadPool cũng được zero cùng lúc, nên cửa sổ đo vẫn nhất quán).

**Đã sửa ở probe v1.4:** `resetState()` zero **cả** bộ đếm per-SourceBuffer, ghi `probe.resets`, panel hiện `⟲reset×n`, và `m1_analyze.py` in cảnh báo khi thấy lệch mà không có `resets` (để dữ liệu cũ vẫn bị soi ra). Cố ý **không** xoá dấu hiệu DRM khi Reset — một lần thấy CENC/EME là bằng chứng cứng.

**Hệ quả lên báo cáo này:** cột `appends` trong bảng là số **sau lần Reset cuối**, không phải tổng cả phiên. Các chỉ số dùng để phán quyết (`L`, `%<ng`, `hist`) **không** bị ảnh hưởng. Tổng "11 203" ở §1 vì vậy là **cận dưới**.

---

## 5. ĐỀ XUẤT SỬA THIẾT KẾ M2: **MIRROR-MSE** (thay §7.4)

> ⚠️ **CẬP NHẬT (xem tài liệu 24):** thiết kế dưới đây đã được **sửa một điểm cốt lõi**.
> Element ẩn phát **1×** nên PCM của nó **không thể vượt trước playhead** của trang — trần chỉ là
> khoảng lợi lúc khởi động (1–3 s), không phải 25–300 s như `L` của byte. Đường trích PCM đúng là
> **`decodeAudioData` trên `OfflineAudioContext` 16 kHz** (giải mã nhanh hơn thời gian thực).
> Chi tiết ở `24_M2_MIRROR_MSE_PCM.md` §2. Phần dưới giữ nguyên để làm mạch lập luận.

Từ phát hiện §3, đề xuất **không parse byte** nữa:

```
[MAIN world] hook appendBuffer của ĐÚNG SourceBuffer audio (biết qua mimeType từ addSourceBuffer)
   │
   ├─(1) gọi bản gốc appendBuffer  → trang phát bình thường, không đổi ngữ nghĩa
   │
   └─(2) chép y nguyên byte sang một MediaSource ẨN:
            shadow = new MediaSource()
            shadowSB = shadow.addSourceBuffer(<cùng mimeType>)
            shadowSB.appendBuffer(cùng BufferSource)
            shadowEl.src = URL.createObjectURL(shadow)     // <audio> không gắn DOM
            shadowEl.playbackRate = 1…4                     // giải mã nhanh hơn realtime
                  │
                  └─ createMediaElementSource(shadowEl) → AudioWorkletNode("audio-capture-processor")
                        → PCM16 16 kHz → TÁI DÙNG NGUYÊN XI đường audio_chunk hiện có
```

**Vì sao tốt hơn §7.4 và tốt hơn tự demux:**

| Tiêu chí | Tự demux (mp4box/mux.js/EBML) | §7.4 ghép Blob + element ẩn | **Mirror-MSE** |
| :--- | :--- | :--- | :--- |
| Xử lý append bị cắt vụn (§3) | ❌ phải tự ghép luồng | ⚠️ phải tự ghép luồng | ✅ **không cần biết byte là gì** |
| WebM/Opus (YouTube) | ❌ cần thêm bộ demux EBML | ⚠️ được, nếu ghép đúng | ✅ trình duyệt lo |
| AV1/HEVC/opus/aac | tự lo | trình duyệt lo | ✅ trình duyệt lo |
| Mã hóa ở tầng vận chuyển (AES-128 HLS) | gặp byte mã hóa nếu bám network | — | ✅ byte tại `appendBuffer` đã là plaintext (§R7) |
| DRM/CENC | ❌ không giải mã được | ❌ không giải mã được | ❌ **vẫn không** — mirror byte mã hóa thì shadow cũng không phát được ⇒ tự nhiên **fail-safe**, đúng yêu cầu "từ chối DRM" |
| Chi phí | demux + decode | decode | decode (×2, chỉ luồng audio) |
| Rủi ro chính | bug demux ⇒ im lặng | ghép Blob sai ⇒ không phát | autoplay/bịt tiếng của element ẩn — **đã đo, xem §5.1** |

### 5.1 Q8 — KẾT QUẢ: **ĐẠT** (mirror-MSE khả thi) `[VERIFIED-M1]`

Chạy `scratch/m1_mse/mirror_test.html` (Firefox, máy người dùng). Tín hiệu đo là sóng sin 440 Hz biên độ **0,3** ⇒ RMS lý thuyết = `0,3/√2` = **0,2121**:

| # | Kịch bản | RMS đo được | Kỳ vọng | Kết luận |
| :-: | :--- | ---: | :--- | :--- |
| A | element ẩn, **không muted**, `volume=1`, 1×, `src→analyser→gain(0)→destination` | **0,2128** | > 0 | ✅ **ĐẠT** — trùng khớp `0,3/√2` ⇒ mirror giải mã **đúng biên độ**, không suy hao |
| B | `muted = true` | 0,0000 | ≈ 0 | ✅ đúng dự đoán — khớp `content-script.js:128-145` |
| C | `volume = 0` (không muted) | 0,0000 | ≈ 0 | ✅ đúng dự đoán — khớp `audio-capture.js:92-94` |
| D | 1× → **2×**, không muted | **0,2131** | > 0 | ✅ ĐẠT — `currentTime` 0,68 → 3,53 ⇒ element ẩn **giải mã nhanh hơn realtime được** |
| E | ngắt `gain → destination` | 0,2121 | ≈ 0 | ⚠️ **khác dự đoán** — xem dưới |

Thêm: mirror **cùng một `ArrayBuffer`** sang hai `SourceBuffer` chạy tốt, **không** báo lỗi ⇒ `appendBuffer` **không detach** ⇒ không cần `slice()`, không tốn thêm bản copy byte nào.

**Ba quy tắc thiết kế rút ra (bắt buộc cho M2):**

1. Element ẩn phải **KHÔNG `muted`** và giữ `volume = 1`. Bịt tiếng bằng `GainNode.gain = 0` **phía sau** `createMediaElementSource` (KB B và C đều cho 0 ⇒ mọi cách "tắt tiếng ở element" đều giết luôn tín hiệu).
2. Đường `MediaElementSource → … → destination` cứ **giữ nguyên** dù KB E cho thấy `AnalyserNode` vẫn có mẫu khi ngắt. KB E chỉ chứng minh điều đó cho `AnalyserNode`; `audio-capture.js:222` ghi nhận `AudioWorkletNode` **cần** đường tới destination ở một số engine, và giữ nó thì không tốn gì (gain = 0 ⇒ vẫn im lặng).
3. `shadowEl.playbackRate` dùng được tới ít nhất **2×** (KB D) ⇒ có thể đuổi kịp buffer dẫn trước.

**Điều Q8 KHÔNG chứng minh (giới hạn):**

- KB E dùng `AnalyserNode`, **không** dùng `AudioWorkletNode` — không suy ra được hành vi của worklet trong `lib/audio-processor.js`.
- KB D chứng minh *có* PCM ở 2×, **không** chứng minh decode **theo kịp** 2× liên tục trong nhiều phút (browser thường giải mã nhanh hơn realtime, nhưng chưa đo ở tải thật có player chính chạy song song).
- Thí nghiệm chạy trên **WebM/Opus audio-only**; chưa thử với fMP4/AAC hay HEVC.
- Chưa đo **chi phí CPU** của việc giải mã audio lần thứ hai, và chưa đo autoplay policy khi `play()` được gọi **không** kèm cú click (thí nghiệm này có cú click mở đầu).

---

## 6. VIỆC TIẾP THEO (đề xuất thứ tự)

| # | Việc | Vì sao | Chi phí |
| :-: | :--- | :--- | :--- |
| 1 | ~~Chạy lại 3 host còn thiếu chỉ số~~ | ✅ **ĐÃ XONG** (phiên 10:12 / 10:14 / 10:16) | — |
| 2 | ~~Chạy 2×~~ | ✅ **ĐÃ XONG — ĐẠT** (§4.3) | — |
| 3 | ~~**Q8**: mirror-MSE~~ | ✅ **ĐÃ XONG — ĐẠT** (§5.1) | — |
| 4 | **M2 theo mirror-MSE** — không cần mp4box.js/mux.js | ✅ **ĐÃ ĐO ĐƯỢC: `lead p50 = 82,76 s`, `belowPct = 0 %`** (so với 0,19 s của element) — xem `24_M2_MIRROR_MSE_PCM.md` | **đang làm (còn 1 khiếm khuyết)** |
| 5 | ~~Tối ưu đường dịch (~430 ms)~~ | ❌ **NGƯỜI QUYẾT ĐỊNH BỎ** — model dịch lớn được chọn có chủ đích để bảo đảm chất lượng dịch; không đánh đổi chất lượng lấy độ trễ. Ghi ở đây để các vòng audit sau **không đề xuất lại** | — |

### 6.1 KẾT QUẢ ĐO M2 — ba câu hỏi đã có đáp án `[ĐO]`

Chi tiết ở `24_M2_MIRROR_MSE_PCM.md`. Hai đáp án **bác bỏ một giả định của chính §5**:

| # | Câu hỏi | Đáp án |
| :-: | :--- | :--- |
| **Q9** | Worklet nạp từ `blob:` có bị CSP của trang chặn không? | ⚠️ **CÓ, trên YouTube**: `addModule(blob) → "The operation was aborted."` ⇒ probe tự chuyển sang **ScriptProcessor** và vẫn cho **PCM sạch** (liên tục 0,9999; RMS 4009; 0% im lặng). Đường dự phòng là **bắt buộc**, không phải lý thuyết |
| **Q13** | PCM từ element ẩn có đi trước playhead không? | ❌ **KHÔNG — và về bản chất là không thể.** `lead p50 = 0,20 s` và `0,19 s` ở hai phiên; `stall 501/2006` chunk (25%); element lấy được **128,38 s PCM** so với **205,18 s** của đường offline trong cùng phiên. Element phát 1×, trang cũng 1× ⇒ `lead(t) ≤ lead(0)` |
| **Q14** | **Vậy đường nào ĐẠT?** | ✅ **`decodeAudioData` trên `OfflineAudioContext` 16 kHz.** Đo được **`lead p50 = 82,76 s`**, `belowPct = 0 %` (không mẫu nào dưới 3,5 s — biên an toàn **23,6×**), tốc độ giải mã **136,79×** thời gian thực |
| **Q15** | fMP4/AAC có ghép byte thành file giải mã được không? | ✅ **CÓ** — đo trên `www.xvideos.com` (`audio/mp4;codecs=mp4a.40.2`): byte fMP4 dùng **đúng cách ghép** như WebM (init + phân đoạn), **không cần demux**, và đường offline cho **lead DƯƠNG +19,85 s**. Lưu ý: xvideos append **cả phân đoạn một lần** (~58 KB/append) chứ không cắt vụn như YouTube (2,5 KB) ⇒ mỗi lô chỉ có 1–2 biên `moof` |
| **Q16** | CSP chặn worklet là chuyện toàn cục hay từng trang? | **Theo TỪNG TRANG.** YouTube chặn `blob:`, xvideos **không** chặn. ⇒ sản phẩm **phải** có cả hai đường, không thể chọn một |

> ⚠️ **NĂM vòng hồi quy đáng ghi lại (chi tiết ở tài liệu 24 §5.3, §5.6, §5.8, §5.10, §5.11):** bản sửa cho 72 lỗi
> `EncodingError` **cắt mất init segment** ⇒ lỗi đổi sang `unknown content type`, lead **âm**
> (−11,22 s). Sửa lại (ghép init) thì **hết lỗi giải mã** (`lỗi 0`, `bỏ 0`, `liên tục 0,9989`)
> nhưng **neo mốc media sai mảnh** ⇒ **kẹt 423 nhịp**, lead **âm** (−9,69 s). Bản v5 sửa cả hai ⇒
> **lead DƯƠNG trên cả hai host** (`+43,50 s` YouTube WebM/Opus, `+24,59 s` xvideos fMP4/AAC),
> **`belowPct = 0 %`** — tức **mọi mẫu** đều trên ngưỡng 3,5 s (kể cả lúc xấu nhất: `min = 4,58 s`
> và `9,48 s`). Khiếm khuyết còn lại **đổi bản chất** thành **độ phủ** (`liên tục` 0,30–0,40):
> PCM dẫn trước nhưng có **khoảng trống** ⇒ còn phải sửa trước khi tích hợp.
>
> 🔴 **Vòng thứ tư (v6) là vòng tệ nhất: 0 PCM trên CẢ HAI host.** Không tìm thấy init segment nên
> dòng dự phòng cũ `if (!init) init = S.byteLog[0]` đã **lấy một mảnh MEDIA làm init** ⇒ file thiếu
> header ⇒ `unknown content type` **193 lần** trên YouTube (bằng chứng nằm trong log: `initBoxes =
> "O)?` — tên box vô nghĩa); trên xvideos **0 lần thử giải mã** với nguyên nhân **chưa xác định**.
> **Bài học đắt nhất của cả M2:** bộ kiểm thử **129/129 vẫn xanh** trong khi hành vi thật xấu đi, vì
> nó chỉ kiểm trường hợp *có* init, chưa bao giờ kiểm trường hợp ***thiếu* init**. Bản v7 sửa gốc
> (nhận diện init theo **nội dung**, giữ mảnh đầu của **mọi** SourceBuffer trước mọi bộ lọc) và thêm
> **bộ đếm cho mọi nhánh thoát** của `offlineTick` (`stats.tickWhy`) để lần sau log tự trả lời thay
> vì suy đoán — dry-run **160/160** với phần I phủ đúng ca "thiếu init". **Chưa qua phiên đo.**
>
> 🎯 **Vòng thứ năm (v7) — và đây là vòng đáng học nhất.** Bản v7 sửa init thành công thật (đo được
> `initKind` = `webm`/`mp4` **đúng trên cả ba host**, kể cả host **chưa từng đo**
> `play.vlstream.net`, `offlineNoInit = 0`), nhưng **vẫn 0 PCM**. Bộ đếm `tickWhy` thêm ở v7 **chỉ
> thẳng nguyên nhân ngay phiên kế tiếp**: `waitBoundary` 16–25 nhịp trên cả ba host với
> `waitBoundaryGaveUp = 0` ⇒ van "chờ biên an toàn" **không bao giờ tới ngưỡng**. Gốc là
> `off.cursor = lastKnownAbs + 1` — **chốt con trỏ trước khi biết lô có giải mã được hay không**;
> nhánh "chờ" `return` mà không khôi phục con trỏ ⇒ **lô bị vứt** ⇒ nhịp sau chỉ còn một mảnh ⇒ van
> **đói byte vĩnh viễn** ⇒ **không lần nào gọi `decodeAudioData`**. Đó chính là trạng thái
> `0 blob, 0 lỗi` mà tôi từng ghi là "chưa xác định": **không phải "không có lỗi", mà là "không có
> việc gì được làm"**. Bản **v8** xoá đúng dòng đó; **phần J** của dry-run nay khẳng định byte phải
> **tích luỹ** qua các nhịp chờ, và đã được **kiểm chứng ngược** (cài lại dòng lỗi cũ ⇒ phần J thất
> bại đúng như trên trang thật, `calls=0`).
>
> ⭐ **Và khi viết bài kiểm thử fMP4 ĐẦU TIÊN của M2 thì lộ ra lỗi gốc thứ hai — lỗi giải thích câu
> hỏi để ngỏ suốt năm phiên: vì sao fMP4 chưa lần nào cắt biên được.** `moof`/`styp` là **trường
> TYPE** của box ISO-BMFF (`[size 4 byte][type 4 byte][payload]`), nên box bắt đầu ở `i − 4`. Bản cũ
> trả `i` ⇒ **sai 4 byte ở cả hai đầu file**: file bắt đầu bằng 4 chữ `m o o f` ngay tại vị trí đáng
> lẽ là `size` ⇒ bộ phân tích đọc `size` = **0x6D6F6F66 ≈ 1,8 GB** ⇒ hỏng toàn bộ cây box ⇒
> **`invalid content`**; đuôi file cụt 4 byte vào trong `moof` cuối. **Kiểm chứng ngược cho đúng con
> số**: cài lại `push(i)` ⇒ phần L của dry-run cho `size@216 = 1836019558` (= 0x6D6F6F66). Kèm theo,
> mỗi mốc cắt nay phải qua **kiểm tra cấu trúc** (`size` ≥ 8, ≤ 64 MB, box trọn trong đệm) ⇒ loại
> luôn rủi ro **dương tính giả** M3 đã ghi ở Phụ lục B của tài liệu 24. Dry-run **160/160** với ba
> phần mới (J: tích luỹ byte; K: fMP4 thật với `moof`/`mdat` tách rời; L: cắt ĐẦU file fMP4).
> Bài học phương pháp: suốt năm phiên, **fMP4 chỉ được đo trên trang thật mà không hề có kiểm thử
> offline** — và đó là lý do một lỗi 4 byte sống sót qua năm phiên.
>
> ✅ **KẾT QUẢ CUỐI CÙNG (v8, ba phiên đo thật):** cả ba host **ĐẠT**, bốn mặt **xanh đồng thời** —
> `play.vlstream.net` fMP4 `liên tục 1,00 / lead p50 20,99 s / min 14,79 s`, `www.xvideos.com` fMP4
> `1,00 / 26,65 / 19,68`, `www.youtube.com` WebM `1,00 / 73,18 / 7,29`, **lỗi = 0 và `bỏ` = 0 trên
> cả ba**. `cắt biên` = **11** và **2** trên hai host fMP4 ⇒ **nhánh `moof` cuối cùng đã được chứng
> minh trên trang thật**. Chi tiết: tài liệu 24 §1.11.
>
> ⇒ **Hệ quả ngược lại cho M1:** thiết kế mirror-MSE mà M1 đề xuất ở §5 **đã được chứng minh bằng
> số đo** ở M2, trên hai định dạng và ba host. Xem tài liệu 24.

> **Hệ quả cho §5:** thiết kế mirror-MSE như mô tả ở §5 (element ẩn → worklet → PCM) **không đủ**
> để giữ lợi thế dẫn trước 25–300 s mà §1 đo được — **lợi thế đó nằm ở BYTE, không phải ở PCM**.
> Nhưng phần khó nhất của §5 **vẫn đúng và vẫn dùng được**: byte lấy được nguyên vẹn, không cần
> demux, không cần mp4box.js/mux.js. Chỉ cần đổi **cách trích PCM** từ "element ẩn" sang
> "`decodeAudioData` theo lô" (tài liệu 24 §2.3) thì byte-lead **25–300 s trở thành PCM-lead 83 s**.
>
> Kết luận "mirror được, không cần demux" của Q8 **được giữ nguyên** — thậm chí mạnh hơn: đường
> `offline` còn **miễn nhiễm CSP** vì không nạp module nào.


> **Ghi chú về phép đo 2×:** §4.3 chứng minh **ngân sách dẫn trước** ở 2×, **không** chứng minh backend theo kịp.
> Ở 2×, một câu 5 s audio tới mỗi **2,5 s** wall-clock ⇒ ASR + dịch phải xong trong < 2,5 s. Với `translation ≈ 430 ms/câu`
> thì còn dư, nhưng đây là câu hỏi của **M2**.
>
> **Cài probe v1.4 trước khi đo M2:** v1.4 sửa bất nhất `signals.appends` ↔ `sources[].appends` sau khi bấm Reset (§4.7),
> ghi `probe.resets`, và hiện `rate đã đo` trên panel.

**Điều kiện dừng vẫn giữ nguyên:** nếu `belowRequiredPct` cao (dẫn trước không đủ thường xuyên) ở đa số trang, lợi ích "chạy trước" biến mất và nên dừng trước M2. Với dữ liệu hiện có, **chưa** thấy dấu hiệu đó.

---

## 7. TÁC ĐỘNG LÊN BẢNG RỦI RO CỦA TÀI LIỆU 22

| # | Trước | Sau M1 |
| :-: | :--- | :--- |
| R1 | 🟠 Cao — chi phí demux mỗi phân đoạn `[NOT VERIFIED]` | ⬆️ **Cao hơn nếu tự demux** (byte không chia theo phân đoạn, §3); ⬇️ **giảm mạnh nếu dùng mirror-MSE** (không parse). Chi phí quét box đo được: 0–3 ms |
| R2 | 🟠 Cao — bộ nhớ | Không đổi. Lưu ý mới: YouTube 2 198 append/277 s ⇒ nếu tự đệm byte để ghép phân đoạn, trần bộ nhớ tăng |
| R3 | 🔴 Rất cao — seek | ✅ **Có bằng chứng**: bilibili `remove×4` (09:11) và **`remove×12`** (09:33); phimsrv timeline có khoảng trống `[0–128] [181–326] [664–808]` |
| R4 | 🟡 Trung bình — ABR đổi init | ✅ **Có bằng chứng**: phimsrv **7** init/track (09:15) và **8** init/track (09:37); bilibili `hvc1` (HEVC) vs `av01`/`avc1` ở host khác ⇒ phải reset `TrackInfo` khi đổi codec |
| R5 | 🟠 Cao — quảng cáo/chèn quảng cáo | ⚠️ Chưa xác nhận trực tiếp; khoảng trống timeline của phimsrv là dấu hiệu tương thích |
| R6 | 🟡 Trung bình — player không dùng MSE | Không gặp trong 6/6 ⇒ thấp hơn dự kiến (tập mẫu nhỏ) |
| R7 | 🟡 Trung bình — AES-128 tầng vận chuyển | Không cản trở: byte tại `appendBuffer` là plaintext (không thấy container mã hóa) |
| R8 | 🟠 Cao — đổi ngữ nghĩa trang | Chưa thấy hỏng: 3 389 lời gọi, `errors=0`. **Mirror-MSE thêm một đường gọi `appendBuffer` thứ hai ⇒ phải giữ nguyên tắc "gọi bản gốc trước, lỗi của ta không nổi lên trang"** |
| R10 | 🟡 Trung bình — TTS lệch lịch | Không đổi; càng quan trọng vì dẫn trước 25–300 s |

---

## PHỤ LỤC A — TÁI LẬP

```powershell
# 1) Kiểm tra logic probe không cần trình duyệt (kỳ vọng 22/22)
node scratch/m1_mse/m1_probe_dryrun.mjs

# 2) Tổng hợp + bảng phán quyết (thêm cột %<ng sau khi chạy lại bằng probe 1.1)
python scratch/m1_mse/m1_analyze.py --md report/audit/m1_tong_hop.md

# 3) Soi chi tiết từng SourceBuffer / từng append
python scratch/m1_mse/m1_inspect.py
```

## PHỤ LỤC B — GIỚI HẠN CỦA KẾT LUẬN NÀY

- Mẫu **9 phiên / 6 host**, do người dùng chọn (các trang hay dùng) — không phải mẫu ngẫu nhiên, không suy ra được tỉ lệ cho "web nói chung".
- Không có phiên nào dùng **MSE-in-Worker**, **DRM**, **progressive `src=`**, hay **live** ⇒ các nhánh fallback của M4 **chưa** được thử trên trang thật.
- Probe **không** lưu nội dung media (chỉ kích thước, thời gian, 12 byte đầu) — dữ liệu trong `m1_logs/` không chứa âm thanh/hình ảnh.
- M1 chỉ đo **byte đến sớm**; phần "lấy PCM từ byte" (M2) đã có kết quả đầu: byte-lead **đã biến
  thành PCM-lead 82,76 s** trên `www.youtube.com` (tài liệu 24), nhưng **chưa** kiểm trên host thứ
  hai và **chưa** đạt độ liên tục (0,82).
- `page.url`/`page.urlInitial` của các phiên probe **1.0/1.1** là URL **lúc tải trang**, không phải URL lúc đo (§4.1). Chỉ dùng `page.title` + `sources[].buffered` + số append để nhận diện nội dung đã đo.
- `%<ng` chỉ có ở 3/9 phiên; 6 phiên cũ không thể tính lại vì probe 1.0 không xuất mẫu thô `L`.
