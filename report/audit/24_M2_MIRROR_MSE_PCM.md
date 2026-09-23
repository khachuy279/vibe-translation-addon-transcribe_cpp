# 24 — M2: MIRROR-MSE + GIẢI MÃ NGOẠI TUYẾN, VÀ CÂU HỎI "PCM CÓ CHẠY TRƯỚC ĐƯỢC KHÔNG?"

> **Ngày:** 2026-09 · **Trạng thái:** ✅ **ĐẠT — TRÊN BA HOST, HAI CODEC, BỐN MẶT XANH ĐỒNG THỜI.**
>
> **Kết quả cuối cùng (v8, ba phiên đo thật §1.11):**
>
> | Host | codec | `liên tục` | `lead p50` | **`min`** | lỗi | `cắt biên` | `bỏ` |
> | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
> | `play.vlstream.net` | fMP4/AAC | **1,00** | 20,99 s | **14,79 s** | **0** | 11 | **0** |
> | `www.xvideos.com` | fMP4/AAC | **1,00** | 26,65 s | **19,68 s** | **0** | 2 | **0** |
> | `www.youtube.com` | WebM/Opus | **1,00** | 73,18 s | **7,29 s** | **0** | 16 | **0** |
>
> Ngưỡng cần: `lead` > **3,5 s** (cả `p50` **và** `min`), `liên tục` ≥ 0,97, lỗi < 10 %, `bỏ = 0`.
> **Không host nào trượt một mặt nào** — lần đầu tiên trong toàn bộ dự án. `min` đạt biên
> **2,1×–5,6×** ngưỡng (trước đây chỉ 1,3×). Giải mã **144–157×** thời gian thực.
>
> **Câu trả lời cho câu hỏi quyết định của M2: CÓ.** PCM 16 kHz lấy từ mirror-MSE **đi trước
> playhead của trang 7–73 giây** (trung vị), giữ được **liên tục 100 %**, **không bỏ mảnh nào**, và
> **không phụ thuộc codec** (WebM/Opus và fMP4/AAC đều đạt) — trong khi đường element bị chặn ở trần
> **~1×** (đo được 0,09–0,25 s, §2).
>
> **Chặng đường tới đây gồm BỐN hồi quy tự gây ra** — và cả bốn đều được sửa bằng cùng một phương
> pháp: **làm cho log tự trả lời** thay vì suy đoán thêm một lần nữa (§5.3, §5.6, §5.10, §5.11).
> Hai lỗi gốc cuối cùng, sửa ở **v8**:
> 1. **Con trỏ bị chốt trước khi byte được tiêu thụ** ⇒ nhánh "chờ" vứt mất lô ⇒ van đói byte ⇒
>    **không lần nào gọi `decodeAudioData`** (§5.11a).
> 2. **`moof` bị coi là đầu box** trong khi nó là **trường type** ⇒ mốc cắt sai 4 byte ⇒ bộ phân
>    tích đọc chữ `moof` thành `size = 0x6D6F6F66 ≈ 1,8 GB` ⇒ **`invalid content`** (§5.11b). Lỗi 4
>    byte này **sống sót qua năm phiên đo** vì fMP4 **chưa từng có bài kiểm thử offline nào**.
> **vẫn 0 PCM** — bộ đếm mới `tickWhy` chỉ thẳng nguyên nhân: **van "chờ biên an toàn" bị ĐÓI BYTE**
> (§1.10) vì con trỏ bị chốt **trước** khi byte được tiêu thụ ⇒ lô bị vứt ⇒ van không bao giờ tích
> luỹ đủ. Bản **v8** sửa đúng dòng đó — và khi viết **bài kiểm thử fMP4 ĐẦU TIÊN** của M2 thì lộ ra
> **lỗi gốc thứ hai, lớn hơn**: `moof` bị coi là **đầu box** trong khi nó là **trường type**, nên mốc
> cắt **sai 4 byte** ⇒ bộ phân tích đọc chữ `moof` thành `size` = **0x6D6F6F66 ≈ 1,8 GB** ⇒ hỏng cây
> box ⇒ **`invalid content`**. ⭐ **Đây là lời giải cho câu hỏi để ngỏ suốt NĂM phiên: vì sao fMP4
> chưa lần nào cắt biên được** (§5.11b). Cả hai lỗi đã sửa và **kiểm chứng ngược** (§5.11c).
> **Những gì KHÔNG còn là vấn đề:** độ phủ (nay **1,00** trên cả ba host, trước chỉ 0,40),
> `min` mỏng (nay **7,29–19,68 s**, biên 2,1×–5,6×; trước 4,58 s = 1,3×), và fMP4 không cắt được
> biên (nay **11** và **2** lần cắt, `giữ init` đủ cả).
> **Những gì VẪN là giới hạn:** chưa đo **2×**; chưa đo host thứ tư; **YouTube vẫn là host mỏng
> nhất** (`min = 7,29 s` dù `p50` cao nhất) vì phiên dài nhất; và toàn bộ kết luận vẫn ở tầng
> **thư viện đo**, **chưa** tích hợp vào extension (M2.2).
> **Bốn** hồi quy ghi nguyên văn ở §5.3, §5.6, §5.10, §5.11.
> **Tiền đề:** `22_MSE_INTERCEPT_FEASIBILITY.md` (khả thi tổng thể) và
> `23_M1_KET_QUA_KHAO_SAT_MSE.md` (M1: byte audio có sẵn, dẫn trước 25–300 s; mirror-MSE khả thi
> theo Q8 §5.1).
> **Công cụ:** `scratch/m2_mirror/` — **chưa** đụng vào `extension_firefox/` hay `backend/`.
> **Nhãn:** `[ĐO]` = số từ phiên thật (16 phiên: 8 YouTube + 6 xvideos + 2 play.vlstream.net;
> **PASS 3**); `[VERIFIED-DRYRUN]` = dry-run 160/160; `[GIẢ ĐỊNH]` = chưa kiểm chứng.

---

## 0. KẾT LUẬN ĐIỀU HÀNH

1. ✅ **CÂU HỎI QUYẾT ĐỊNH CỦA M2 ĐÃ ĐƯỢC TRẢ LỜI — ĐẠT TRÊN BA HOST, HAI CODEC, BỐN MẶT XANH
   ĐỒNG THỜI (§1.11).**
   * `play.vlstream.net` (fMP4/AAC): **`liên tục 1,00`**, **`lead p50 = +20,99 s`**,
     **`min = +14,79 s`**, **lỗi 0**, **`cắt biên 11`**, **`bỏ 0`**.
   * `www.xvideos.com` (fMP4/AAC): **`liên tục 1,00`**, **`+26,65 s`**, **`min = +19,68 s`**,
     **lỗi 0**, **`cắt biên 2`**, **`bỏ 0`**.
   * `www.youtube.com` (WebM/Opus): **`liên tục 1,00`**, **`+73,18 s`**, **`min = +7,29 s`**,
     **lỗi 0**, **`cắt biên 16`**, **`bỏ 0`**.
   * **Không host nào trượt một mặt nào.** `belowPct = 0 %` ở cả ba: **không một mẫu nào** dưới
     ngưỡng 3,5 s. Tốc độ giải mã **144–157×** thời gian thực. `[ĐO]`
2. **Đường element bị loại bằng số đo, không bằng lập luận.** Cùng các phiên: `lead p50 = 0,20 s`
   (YouTube) và `0,09 s` (xvideos), `belowPct = 100 %`. Element phát 1× nên `lead(t) ≤ lead(0)` —
   về bản chất **không thể** đạt 3,5 s. `[ĐO]`
3. **⇒ Thiết kế mirror-MSE sống.** Byte audio của trang (M1: dẫn trước 25–300 s) **đã biến
   thành PCM dẫn trước 24–43 s** trên hai host, không cần demux, không cần mp4box.js/mux.js,
   không cần WASM. Đây là kết luận trung tâm của cả M2.
4. **Khiếm khuyết `liên tục 0,82` (72 lỗi cắt giữa phân đoạn) ĐÃ SỬA XONG** — phiên `14:38:30`
   cho **`lỗi giải mã = 0`, `bỏ = 0`, `liên tục = 0,9989`**. Bản sửa ghép-init chạy đúng. `[ĐO]`
5. 🔴 **Nhưng tôi đã gây NĂM hồi quy liên tiếp trong quá trình sửa, cả năm ghi lại nguyên văn:**
   * **v3** cắt mất init segment ⇒ `unknown content type`, **262/263** lỗi, lead **âm** (−11,22 s),
     bỏ mất **425** mảnh. Nguyên nhân: `firstCluster` nằm **sau** init (§5.3).
   * **v4** neo mốc media vào mảnh cuối **lô** trong khi file chỉ chứa tới mảnh cuối **cửa sổ** ⇒
     mục kế tiếp bị coi là "tua lùi" ⇒ **kẹt 423 nhịp**, PCM đứng ở 30 s khi trang đã 82 s ⇒
     lead **âm** (−9,69 s) dù `lỗi = 0` và `liên tục = 0,9989` (§1.6, §5.6).
   * **v6** (lấy mảnh media làm init) và **v7** (chốt con trỏ trước khi tiêu thụ byte) — xem #7/#8.
   * **v8b** — **`moof` bị coi là đầu box** (§5.11b): lỗi 4 byte làm mốc cắt fMP4 sai ⇒
     **`invalid content`**. Đây là lỗi **sống sót lâu nhất** (từ bản đầu tới v8, qua **năm phiên
     đo**) vì fMP4 **chưa từng có bài kiểm thử offline**.
   ⇒ **Bài học:** chỉ số xanh ở một mặt (lỗi / liên tục) **không** bảo đảm mặt còn lại (lead) xanh.
   Và mạnh hơn: **"0 lỗi" có thể chỉ là "không có việc gì được làm".** Và mạnh hơn nữa: **một đường
   không có kiểm thử offline thì lỗi trong đó sẽ sống sót qua mọi phiên đo thật** — vì trên trang
   thật ta chỉ thấy *triệu chứng* (`invalid content`), không thấy *byte*.
6. **Khiếm khuyết còn lại nay đã ĐỔI BẢN CHẤT — từ "lead âm / kẹt" sang "độ phủ".** Không còn kẹt
   (423 → **1** nhịp), lead **dương trên cả hai host**, nhưng `liên tục` chỉ 0,30–0,40 vì PCM có
   **khoảng trống**: lô **10 s ≈ 1 phân đoạn** nên chỉ có 1 biên, không cắt đuôi an toàn ⇒ giải mã
   file cụt ⇒ lỗi ⇒ **bỏ** (34 mảnh ở YouTube v5). Đã sửa ở §5.8: lô **30 s** + chỉ cần **một**
   biên + **chờ thay vì bỏ** + một **lỗi thật** ở `frontCut` (do phần H của dry-run bắt được).
   `[ĐO]` cho nguyên nhân, `[VERIFIED-DRYRUN]` (160/160) cho bản sửa.
7. 🔴 **v6 rồi v7 — hai hồi quy liên tiếp, và đến v7 thì cả BA host đều 0 PCM.** Nguyên nhân v6
   (bịa init) **đã sửa và đã kiểm chứng** trên ba host: `initKind` = `webm`/`mp4`/`mp4` đúng codec,
   `offlineNoInit = 0`, `initBoxes` hợp lệ. `[ĐO]`
8. 🎯 **Nguyên nhân v7 tìm ra nhờ chính bộ đếm thêm ở §5.10 — và đây là bài học phương pháp luận:**
   `tickWhy` cho `waitBoundary` 16–25 nhịp trên **cả ba host** với `waitBoundaryGaveUp = 0` ⇒ van
   "chờ biên an toàn" **chưa bao giờ tới ngưỡng**. Gốc: `off.cursor = lastKnownAbs + 1` **chốt con
   trỏ trước khi biết lô có giải mã được hay không**; nhánh "chờ" `return` mà không khôi phục con
   trỏ ⇒ **lô bị vứt** ⇒ nhịp sau chỉ còn một mảnh ⇒ van đói byte vĩnh viễn ⇒ **không lần nào gọi
   `decodeAudioData`** trên xvideos và `play.vlstream.net`. Đây **chính là** trạng thái "0 blob,
   0 lỗi" mà ở §1.9 tôi đã ghi là *chưa xác định*. `[ĐO]`
   ⇒ **Giá trị của việc thêm bộ đếm đã được trả lại ngay trong phiên kế tiếp.** Nếu v7 chỉ sửa
   init mà không thêm `tickWhy`, tôi đã tiếp tục đoán mò.
9. **Q9: CSP chặn worklet là chuyện THEO TỪNG TRANG** — YouTube chặn `blob:`, xvideos **không**.
   Sản phẩm **phải** có cả hai đường. Đường `offline` **miễn nhiễm CSP** vì không nạp module. `[ĐO]`
10. 🎯 **`moof` BỊ COI LÀ ĐẦU BOX — lỗi 4 byte sống sót qua NĂM phiên, và là lời giải cho fMP4.**
    `moof`/`styp` là **trường type** của box ISO-BMFF nên box bắt đầu ở `i − 4`; bản cũ trả `i` ⇒
    file bắt đầu bằng 4 chữ `m o o f` tại chỗ đáng lẽ là `size` ⇒ bộ phân tích đọc
    `size = 0x6D6F6F66 ≈ 1,8 GB` ⇒ hỏng cây box ⇒ **`invalid content`** (đúng lỗi chiếm ưu thế trên
    cả hai host fMP4), và đuôi file cụt 4 byte vào trong `moof` cuối. **Kiểm chứng ngược cho đúng
    con số đó**: cài lại `push(i)` ⇒ dry-run phần L cho `size@216 = 1836019558` (0x6D6F6F66).
    Nay mốc cắt phải qua **kiểm tra cấu trúc** (`size` ≥ 8, ≤ 64 MB, box trọn trong đệm) ⇒ loại luôn
    rủi ro **dương tính giả** M3. `[ĐO]` + `[VERIFIED-DRYRUN]`
11. 🎯 **CẢ BA DỰ ĐOÁN CỦA §0.11 ĐÃ ĐƯỢC XÁC NHẬN TRONG PHIÊN v8 (§1.11):** `waitBoundary` lớn
    **kèm `waitBoundaryGaveUp` = 5–12** ⇒ van chờ **hoạt động thật**; **`cắt biên` = 11 và 2** trên
    hai host fMP4 ⇒ **fMP4 đã cắt biên được, lần đầu tiên sau năm phiên**; và `chữ ký: moof = 3`
    chứng minh bộ quét **nhìn thấy** `moof` thật. Kèm theo: `giữ init = cắt biên` ở cả ba host ⇒
    không tái diễn hồi quy `unknown content type`.
12. ✅ **Hai mặt từng là nỗi lo lớn nhất nay đã xanh hẳn:** **độ phủ** `liên tục = 1,00` trên cả ba
    host (trước tốt nhất 0,40) và **biên an toàn của `min`** = **2,1×–5,6×** ngưỡng (trước 1,3×).
    YouTube vẫn là host **mỏng nhất** (`min = 7,29 s`) vì là phiên dài nhất. `[ĐO]`
13. **Việc tiếp theo: chuyển sang M2.2 — tích hợp vào `extension_firefox/`.** Điều kiện tiên quyết
    nay **đã đủ** (ba host xanh, cả hai codec). Xem §7 để biết thứ tự và các điểm phải giữ. Trước khi
    tích hợp, **đo thêm hai điều**: (a) host thứ tư để kiểm tính tổng quát, (b) chế độ **2×** — và đo
    **sai số mốc thời gian** của PCM offline (ảnh hưởng việc khớp phụ đề/TTS, R10 tài liệu 22).

---

## 1. CÁC PHIÊN ĐO THẬT

### 1.0 Tổng hợp

| Phiên | probe | host / codec | element lead p50 | **offline lead p50** | liên tục | lỗi / bỏ | kết luận |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `13:37:55` | v1 | youtube / WebM-Opus | 0,20 s | **không có PCM** | — | 0 / 0 | offline kẹt vì các lỗi ở §4 |
| `14:10:42` | v2 | youtube / WebM-Opus | 0,19 s | **82,76 s** ✅ | 0,82 ⚠️ | 72 / 89 | 🎯 **ĐẠT** — còn lỗi cắt lô |
| `14:28:18` | v3 (cắt biên SAI) | youtube / WebM-Opus | 0,18 s | **−11,22 s** ❌ | 1,00 | 262 / 425 | 🔴 **HỒI QUY** — cắt mất init (§5.3) |
| `14:30:19` | v3 (cắt biên SAI) | **xvideos / fMP4-AAC** | 0,20 s | **không có PCM** ❌ | — | 174 / 22 | cùng hồi quy; **lần đầu đo fMP4** |
| `14:38:30` | v4 | youtube / WebM-Opus | (0 PCM) | **−9,69 s** ❌ | **0,9989** ✅ | **0 / 0** ✅ | 🟢 **hết lỗi giải mã** nhưng **KẸT** chờ mép media (§1.6) |
| `14:40:04` | v4 | **xvideos / fMP4-AAC** | 0,19 s | **+19,85 s** ✅ | 0,33 ⚠️ | 156 / 17 | lead dương, nhưng còn cắt giữa phân đoạn (§1.7) |
| `15:06:23` | v5 | youtube / WebM-Opus | 0,20 s | **+43,50 s** ✅ | 0,30 ⚠️ | 24 / 34 | 🎯 **`min = 4,58 s`, `belowPct = 0 %`** — mọi mẫu đều trên ngưỡng (§1.8) |
| `15:04:34` | v5 | **xvideos / fMP4-AAC** | 0,09 s | **+24,59 s** ✅ | 0,40 ⚠️ | 124 / 10 | 🎯 **`min = 9,48 s`, `belowPct = 0 %`** trên **fMP4** (§1.8) |
| `15:28:54` | **v6** | youtube / WebM-Opus | 1,24 s | **không có PCM** ❌ | — | **193 / 68** | 🔴 **HỒI QUY:** mảnh media bị dùng làm init ⇒ `unknown content type` (§1.9) |
| `15:32:08` | **v6** | **xvideos / fMP4-AAC** | 0,20 s | **không có PCM** ❌ | — | 0 / 0 | 🔴 `0 lần thử giải mã` — **nay đã giải thích** ở §1.10 (van chờ đói byte) |
| `15:54:32` | **v7** | youtube / WebM-Opus | 0 s | 49,86 s PCM ⚠️ | **0,2267** ⚠️ | 39 / 38 | `init: webm` ✅ nhưng `waitBoundary 25` ⇒ van chờ **đói byte** (§1.10) |
| `15:51:42` | **v7** | xvideos / fMP4-AAC | 157,76 s | **không có PCM** ❌ | — | **0 / 0** | `init: mp4 [ftyp,moov]` ✅, `waitBoundary 16` ⇒ **van đói byte** (§1.10) |
| `15:57:02` | **v7** | **play.vlstream.net / fMP4-AAC** ← **host thứ BA** | 131,26 s | **không có PCM** ❌ | — | **0 / 0** | 🆕 host mới: init đúng, **cùng một nguyên nhân** (§1.10) |
| `16:18:45` | **v8** | **play.vlstream.net / fMP4-AAC** | 0,25 s | **+20,99 s** ✅ | **1,00** ✅ | **0 / 0** ✅ | ✅ **ĐẠT** — `min 14,79 s`, `cắt biên 11`, `giữ init 11` (§1.11) |
| `16:21:17` | **v8** | **xvideos / fMP4-AAC** | 0,18 s | **+26,65 s** ✅ | **1,00** ✅ | **0 / 0** ✅ | ✅ **ĐẠT** — `min 19,68 s`, `cắt biên 2` (§1.11) |
| `16:37:36` | **v8** | **youtube / WebM-Opus** | 0,17 s | **+73,18 s** ✅ | **1,00** ✅ | **0 / 0** ✅ | ✅ **ĐẠT** — `min 7,29 s`, `cắt biên 16`; fMP4 **đã cắt biên được** (§1.11) |

### 1.1 YouTube `14:10:42` — phiên quyết định `[ĐO]`

```
host: www.youtube.com   appends 552 (audio) / 3,54 MB   video: 1723 append av01
shadow: audio/webm; codecs="opus"   capture: scriptprocessor   ← CSP chặn blob worklet
playing: true  paused false  muted false  vol 1   ctxRate 48000
element : 2006 chunk = 128,38 s  liên tục 0,9999  RMS 4009  stall 501  | lead n=512 p50=0,19 max=0,23
offline : 3206 chunk = 205,18 s  liên tục 0,8208  RMS 4161  | lead n=534 p50=82,76 p95=126,41 max=128,68
offline : 22 blob giải mã / 72 lỗi, tăng tốc 136,79× (1500 ms cho 205 s audio)
evict 26 (−145 s)   shadow buffer 104 s   byteLogPruned 462   preOpenQueued 1   t1Unknown 1
offlineNoMediaSeen 42/552   offlineGuards 0   byteLogTruncated false
```

Đọc phiên này:

| Quan sát | Ý nghĩa |
| :--- | :--- |
| **`offline lead p50 = 82,76 s`, `belowPct = 0`** | 🎯 **Byte-lead 25–300 s của M1 đã thành PCM-lead 83 s.** Ngân sách 3,5 s được thoả **23,6×** |
| `tăng tốc 136,79×` | Giải mã nhanh hơn thời gian thực ~137 lần ⇒ đủ sức chạy trước |
| `element lead 0,19 s` + `stall 501/2006` | Trần 1× được xác nhận **lần thứ hai**; element không thể là đường chính |
| offline 205 s > element 128 s | Offline lấy được **nhiều PCM hơn** trong cùng 134,8 s phiên |
| `72 lỗi invalid content`, `liên tục 0,82` | ⚠️ Khiếm khuyết còn lại — lô cắt giữa phân đoạn (§5) |
| `evict 26 (−145 s)`, `byteLogPruned 462` | Cửa sổ trượt + dọn nhật ký byte chạy đúng, `byteLogTruncated = false` |
| `offlineGuards 0` | `decodeAudioData` **không** treo ⇒ giả thuyết "busy kẹt" ở phiên 1 **không** tái hiện |
| `offlineNoMediaSeen 42/552` (7,6%) | Phần lớn append **có** mở rộng `buffered.end` ⇒ luật "bỏ mảnh trùng mép" cũ sẽ phá stream (đã sửa, §4 #3) |

**Q10 (autoplay):** `play() chặn: 1` rồi `playUnlockedByGesture: 1` — element ẩn bị chặn lúc đầu
và **mở khoá được bằng cú click của người dùng**. `userGesturesSeen: 4`. ⇒ Q10 ĐẠT nếu có click.

**Q12 (bộ nhớ):** 26 lần evict, bỏ 145 s khỏi shadow, nhật ký byte dọn 462 mục, `byteLogTruncated false`.

### 1.2 YouTube `13:37:55` — phiên trước khi sửa (đối chứng) `[ĐO]`

```
element : 386 chunk = 24,7 s   liên tục 0,9992   stall 96   | lead n=98 p50=0,2 max=0,24
offline : 0 chunk = 0 s        còn chờ 594 mục
lỗi: addModule(blob) bị chặn
```

Phiên này đã lộ ra các lỗi ở §4 — trong đó nghi phạm chính là **kẹt vĩnh viễn** ở khâu chọn lô:
code cũ `break` ngay khi gặp mục đầu chưa có mép media (`t1 === null`), nên **một** mục như vậy
làm đường offline đứng im mãi mãi mà **không ghi lỗi gì**. Đã sửa (§4 #1, #2).

### 1.3 Self-test trong trình duyệt — `localhost:8099` `[ĐO]`

Lần chạy đầu `mirroredAppends: 1` nhưng `sb.error: 1`, `buffered: []`, 0 PCM. Nguyên nhân: trang
append **init segment trước khi shadow `sourceopen`** ⇒ code cũ bỏ mất init ⇒ shadow chỉ nhận
mảnh media trần. **Đây là lỗi sẽ làm mirror hỏng ở mọi trang**, và self-test đã bắt được nó.
Đã sửa (§4 #1).

### 1.4 YouTube `14:28:18` — phiên HỒI QUY của bản sửa cắt-biên `[ĐO]`

```
aligns 262   blob giải mã 1 / lỗi 262   bỏ mất 425 mảnh   droppedBytes 1 432 594
offline: 19,97 s  liên tục 1,0  lead p50 −11,22 s  max 19,98   tăng tốc 123×
element: 108,67 s  stall 424  lead p50 0,18
lỗi: EncodingError: ... contains an UNKNOWN CONTENT TYPE
```

🔴 **Bản sửa đầu tiên làm mọi thứ TỆ ĐI.** Lỗi đổi từ `invalid content` sang
**`unknown content type`** — dấu hiệu của một file **không có init segment**: không có EBML
header / `ftyp`+`moov` thì bộ giải mã không biết đó là định dạng gì. Lead **âm** (−11,22 s) vì
chỉ 1 lô ở đầu stream giải mã được, phần còn lại bị bỏ. Xem §5.2 để biết nguyên nhân.

### 1.5 xvideos `14:30:19` — **lần đầu đo host fMP4/AAC** `[ĐO]`

```
codec: audio/mp4;codecs=mp4a.40.2  (fMP4/AAC)     capture: audioworklet-blob  ← KHÔNG bị CSP chặn
page: 24 append audio / 1,41 MB  ⇒ ~58 KB mỗi append (CẢ PHÂN ĐOẠN một lần)
      (so với YouTube: 2,5 KB mỗi append, mỗi phân đoạn bị cắt 26–57 lần)
element: 1119 chunk = 71,62 s  liên tục sạch  stall 0  ← host này element KHÔNG đói
         lead p50 0,20 s  belowPct 100
shadow: playing, buffered 48,48 s   evict 12   noMedia 11/24   t1Unknown 2   preOpen 1
offline: 0 PCM   aligns 102   blob lỗi 174   bỏ 22 mảnh   droppedBytes 153 540
lỗi: unknown content type (3 mảnh/127 618 byte); invalid content (5 mảnh/256 889 byte)
```

Ba điều học được từ host này — **quan trọng cho M2.2**:

1. **fMP4/AAC cũng bị đúng hồi quy đó** (`unknown content type`) ⇒ cách ghép byte **không** phụ
   thuộc codec; cùng một lỗi, cùng một cách sửa.
2. **xvideos append CẢ PHÂN ĐOẠN một lần** (~58 KB), không cắt vụn như YouTube. Nghĩa là **cắt
   theo biên phân đoạn gần như không cần thiết ở host này** — mỗi lô vốn đã trùng biên. Đây là
   tin tốt: nếu cắt-biên gây khó chịu thì có thể chỉ bật khi cần (`append` nhỏ).
3. **`capture: audioworklet-blob`** — trên xvideos CSP **không** chặn worklet, khác YouTube. Vậy
   Q9 là **theo từng trang**, không phải toàn cục ⇒ sản phẩm **phải** có cả hai đường.

⚠️ Lưu ý: đây là phiên **rất ngắn** (73,8 s, 24 append). Không kết luận được gì về hành vi dài hơi
của host này. Nhưng nó đủ để trả lời "byte fMP4 có ghép được thành file giải mã được không" —
**có, một khi init được giữ**.

---

### 1.6 YouTube `14:38:30` (v4) — **hết lỗi giải mã**, nhưng lộ một kiểu hỏng MỚI `[ĐO]`

```
aligns 2   giữ init 2   noCut 0        ← cắt biên CÓ chạy và CÓ giữ init ✓
blob giải mã 3 / LỖI 0                 ← trước đó là 72 lỗi  ✅
bỏ mất 0 mảnh                          ← trước đó là 89 mảnh  ✅
offline: 468 chunk = 29,95 s  liên tục 0,9989   ← trước đó 0,82  ✅
page ct = 82,34 s   shadow buffered = 117,74 s
why = "chờ mép media (t1) — 1 mục, 423 nhịp"    ← ⚠️ ĐỨNG YÊN
pending 410   t1Unknown 4   preOpen 6
lead off: p50 −9,69 s   min −52,27   max 29,99   belowPct 65,4
element: 0 chunk (!)
```

**Nửa đầu là thắng thật:** ba chỉ số quyết định của §5 đều xanh — `lỗi giải mã = 0`,
`bỏ = 0`, `liên tục = 0,9989`. Bản sửa ghép-init hoạt động đúng (`giữ init = 2`).

**Nửa sau là một kiểu hỏng do chính tôi tạo ra.** Đường offline **đứng yên** ở mốc 30 s media
trong khi trang đã 82 s ⇒ lead **âm**. `offlineState.why` nói thẳng: *"chờ mép media (t1) — 1 mục,
423 nhịp"* — tức đã kẹt **423 nhịp liên tiếp**. Hai lỗi cộng lại (§5.6):

1. **Neo sai mốc media.** Khi mảnh cuối nằm trọn trong cửa sổ **không có** `t1`, code cũ vẫn neo
   bằng `t1` của mảnh **ở cuối lô** — muộn hơn hẳn so với vùng byte thật trong file. `lastEndMedia`
   thành quá lớn ⇒ mục kế tiếp bị coi là **tua lùi** ⇒ vòng lặp `break` ngay ⇒ lô chỉ còn 1 mục
   không có `t1` ⇒ nhánh "chờ mép media" ⇒ lặp y nguyên mãi mãi.
2. **Con trỏ dùng `indexOf`.** `S.byteLog.indexOf(batch[lastIncluded])` trả **−1** khi mục đó đã bị
   dọn khỏi nhật ký, và khi đó con trỏ **nhảy về 0** trong im lặng (quét lại từ đầu).

⚠️ **Quan sát chưa lý giải:** phiên này element cho **0 chunk PCM** (`lead.element.samples = 0`)
dù `capture: scriptprocessor` và `playing: true`. Các phiên khác cùng host đều có PCM element.
Đây là **điểm bất thường cần theo dõi**, chưa phải kết luận — có thể ScriptProcessor không được
gắn trong lần chạy đó.

### 1.7 xvideos `14:40:04` (v4) — lead **dương** trên fMP4, nhưng còn cắt giữa phân đoạn `[ĐO]`

```
codec audio/mp4;codecs=mp4a.40.2   capture audioworklet-blob (CSP không chặn)
aligns 0   withInit 0              ← cắt biên KHÔNG BAO GIỜ chạy ở host này
blob giải mã 3 / lỗi 156   bỏ 17 mảnh
offline: 468 chunk = 29,95 s  liên tục 0,3327   firstMedia 0   lastMedia 89,96
lead off: p50 +19,85 s   p95 37,39   belowPct 17,9     ← DƯƠNG ✅
element: 1216 chunk = 77,82 s  liên tục 1,0  lead p50 0,19 s  belowPct 100
lỗi: invalid content (1 mảnh/8 055 byte), (5 mảnh/258 024), (1 mảnh/121 712), …
```

Điểm đáng chú ý: **lead dương (+19,85 s) ngay cả khi 156 lô lỗi** — vì PCM giải mã được rải khắp
stream (`lastMedia 89,96` trong khi chỉ có 29,95 s PCM ⇒ `liên tục 0,33`). Nghĩa là ngay ở trạng thái
hỏng, đường offline **vẫn chạy trước**; sửa được các lô lỗi sẽ nâng `liên tục` mà không mất lead.

**Vì sao cắt biên không chạy:** xvideos append **cả phân đoạn một lần**, nên vùng media của một lô
(≈10 s) chỉ chứa **1–2 chữ ký** `moof`. Điều kiện cũ đòi chữ ký cuối nằm ở **nửa sau** file ⇒ với
1 chữ ký (nằm ngay sau init) thì **bị từ chối** ⇒ giải mã nguyên file ⇒ phân đoạn cuối **cụt** ⇒
`invalid content` ⇒ thử 12 lần ⇒ **bỏ**. Đã sửa: **cần ≥ 2 biên mới cắt**; nếu mới có 1 biên thì
**CHỜ thêm byte** (giữ nguyên con trỏ) thay vì thử-rồi-bỏ.

### 1.8 Hai phiên v5 — **câu hỏi quyết định nay đã được trả lời trên CẢ HAI host** `[ĐO]`

| | YouTube `15:06:23` | xvideos `15:04:34` |
| :--- | ---: | ---: |
| codec | `audio/webm; codecs="opus"` | `audio/mp4;codecs=mp4a.40.2` |
| **lead p50** | **+43,50 s** ✅ | **+24,59 s** ✅ |
| **lead p5 / min** | **+9,34 s / +4,58 s** ✅ | **+11,11 s / +9,48 s** ✅ |
| **`belowPct` (dưới 3,5 s)** | **0 %** ✅ | **0 %** ✅ |
| lead max | 113,44 s | 39,38 s |
| liên tục | 0,30 ⚠️ | 0,40 ⚠️ |
| PCM | 59,90 s | 39,94 s |
| lỗi / bỏ | 24 / 34 ⚠️ | 124 / 10 ⚠️ |
| cắt biên / giữ init | 4 / 4 ✅ | **0 / 0** ❌ |
| kẹt | **1 nhịp** ✅ (không còn 423) | 0 ✅ |
| tăng tốc | 124,02× | 156,61× |
| element lead p50 | 0,20 s | 0,09 s |

**Đây là kết quả mạnh nhất từ trước tới nay, và nó khác về CHẤT so với phiên `14:10:42`:**

* `14:10:42` có `p50 = 82,76 s` nhưng **không ai biết p5/min** — và `liên tục` chỉ 0,82.
* Phiên v5 có `p50` thấp hơn nhưng **`min` vẫn trên ngưỡng**: `+4,58 s` (YouTube) và `+9,48 s`
  (xvideos) — tức **KHÔNG một mẫu nào** dưới 3,5 s, kể cả lúc xấu nhất. Biên an toàn nhỏ nhất
  là **1,3×** (4,58 / 3,5) — mỏng, nhưng **dương trên cả hai host và cả hai codec**.
* **Hai lỗi hồi quy đã hết:** không còn kẹt (1 nhịp so với 423), và lead **không còn âm**.
* Đường element vẫn bị trần 1×: `0,20 s` / `0,09 s`, `belowPct = 100 %`.

**Khiếm khuyết còn lại đã đổi bản chất:** không còn là "lead âm" hay "kẹt", mà là **độ phủ**
(`liên tục` 0,30–0,40): PCM bị **khoảng trống** do các lô lỗi bị bỏ. Nguyên nhân đã truy ra và
sửa ở §5.8 — **lô quá ngắn (10 s ≈ 1 phân đoạn) nên chỉ có 1 biên, không cắt được đuôi**; và một
**lỗi thật mới** ở `frontCut`. Ở xvideos, `cắt biên = 0` nghĩa là **chưa lần nào** cắt được theo
biên ⇒ probe v6 nay **đếm và in ra loại chữ ký** (`cluster` / `moof` / `styp`) cùng **tên các box
ở đầu init segment** để phiên sau trả lời dứt điểm vì sao (không đoán).

### 1.9 Hai phiên v6 — **HỒI QUY: 0 PCM trên cả hai host** `[ĐO]`

| | YouTube `15:28:54` | xvideos `15:32:08` |
| :--- | ---: | ---: |
| PCM offline | **0** ❌ | **0** ❌ |
| blob giải mã / lỗi | **0 / 193** | **0 / 0** |
| bỏ mất | 68 mảnh | 0 |
| `cắt biên` / giữ init | 1 / 1 | **0 / 0** |
| chữ ký | cluster 3 / **moof 0** | cluster 0 / **moof 1** |
| `initBoxes` | **`"O)?`** ⚠️ | **`ftyp,moov`** ✅ |
| `why` | `unknown content type` (3/12) | `không có byte mới` |
| element (đối chứng) | 1,24 s | 0,20 s |

**Nguyên nhân trên YouTube — tìm ra và ĐÃ SỬA (chắc chắn):**
`initBoxes = "O)?`" là bằng chứng nằm ngay trong log: **tên box vô nghĩa**, tức "init segment" mà
probe dùng **không phải** init segment. Cơ chế: phiên này **không mảnh nào** được đánh dấu init
(`preOpen = 0` — hook lỡ mất mảnh init), nên dòng dự phòng cũ

```js
if (!init && S.byteLog.length && batch.indexOf(S.byteLog[0]) < 0) init = S.byteLog[0];
```

đã lấy **mảnh media cũ nhất** làm init. Hệ quả kép:
1. `mediaStart` = **cỡ mảnh media** (36 KB) ⇒ mọi chữ ký cluster nằm **trước** `mediaStart` bị lọc
   sạch ⇒ gần như không cắt biên được (`aligns = 1` trên ~193 nhịp).
2. File gửi đi giải mã **thiếu init segment thật** ⇒ `unknown content type` **193 lần** ⇒ 0 PCM.

**Nguyên nhân trên xvideos — nay ĐÃ XÁC ĐỊNH (xem §1.10 và §5.11).** Log cho
`0 blob, 0 lỗi, 0 bỏ, why = 'không có byte mới'` trong khi 44 mục **không** mục nào được đánh dấu
đã giải mã hay đã bỏ. Lúc viết mục này tôi ghi là **chưa biết** và **không suy diễn** — đúng như vậy.
Phiên v7 với bộ đếm `tickWhy` đã trả lời: đó là **van "chờ biên an toàn" bị đói byte**, vì con trỏ bị
chốt trước khi byte được tiêu thụ (§5.11a). Ghi lại đây để thấy **bộ đếm đã trả lại giá trị ngay ở
phiên kế tiếp** — nếu v7 chỉ sửa init mà không thêm `tickWhy`, tôi đã tiếp tục đoán mò.

**Phát hiện phụ đáng giá:** `initBoxes = ftyp,moov` xác nhận init segment của xvideos **đúng là
fMP4**. Con số `moof ≤ 1` thì **không dùng được làm bằng chứng** — phiên này gần như không dựng được
lô nào, nên bộ quét chỉ thấy một mảnh mỗi lần (§5.11b giải thích phần còn lại).

### 1.10 Ba phiên v7 (ba host) — bộ đếm chỉ thẳng nguyên nhân: **van "chờ" bị ĐÓI BYTE** `[ĐO]`

| | YouTube `15:54:32` | xvideos `15:51:42` | **play.vlstream.net `15:57:02`** ← host thứ BA |
| :--- | ---: | ---: | ---: |
| mime | WebM/Opus | fMP4/AAC | **fMP4/AAC** |
| `initKind` | **`webm`** ✅ | **`mp4`** ✅ | **`mp4`** ✅ |
| `initBoxes` | `webm` | `ftyp,moov` | **`ftyp,moov`** |
| `offlineNoInit` | **0** ✅ | **0** ✅ | **0** ✅ |
| PCM offline | 49,86 s | **0** ❌ | **0** ❌ |
| gọi giải mã / lỗi | 5 / **39** | **0 / 0** | **0 / 0** |
| liên tục | 0,2267 | — | — |
| `aligns` / giữ init | 2 / 2 | **0 / 0** | **0 / 0** |
| `tickWhy` | `allConsumed 742`, **`waitBoundary 25`**, `empty 11`, `noMediaEdge 1` | `allConsumed 762`, **`waitBoundary 16`**, `empty 10` | `allConsumed 642`, **`waitBoundary 16`**, `empty 8` |
| element | 0 s | 157,76 s | 131,26 s |

**Sửa nhận diện init của v7 chạy đúng trên cả ba host:** `initKind` khớp codec, `initBoxes` hợp lệ,
`offlineNoInit = 0` — hồi quy `15:28:54` đã hết. Và **host thứ ba** (`play.vlstream.net`, fMP4/AAC)
cũng vào được đường offline với init đúng.

**Nguyên nhân chung của cả ba — tìm ra nhờ chính bộ đếm vừa thêm ở §5.10:**
`waitBoundary` xuất hiện **16–25 nhịp ở CẢ BA host**, kèm `waitBoundaryGaveUp` **= 0** (van 40 nhịp
**chưa bao giờ** tới ngưỡng). Nghĩa là: **nhịp nào cũng có byte mới, nhịp nào cũng "chờ", mà lô
không bao giờ lớn lên.** Cơ chế nằm ở một dòng đã có từ v5:

```js
off.cursor = lastKnownAbs + 1;   // ← chốt con trỏ TRƯỚC khi biết lô có giải mã được hay không
```

Nhánh "chưa cắt được biên an toàn" `return` mà **không khôi phục con trỏ** ⇒ lô vừa dựng bị **vứt
bỏ** ⇒ nhịp sau chỉ còn **một mảnh vừa tới** ⇒ lại chưa đủ để cắt an toàn ⇒ lại chờ. Với xvideos
và vlstream (append **lớn và thưa**: 33 và 17 append trong ~130–160 s), `waitBursts` chỉ tăng 1 mỗi
lần trang append nên đạt **16** rồi phiên kết thúc — **không lần nào gọi `decodeAudioData`**.

**Đây chính là bí ẩn "0 blob, 0 lỗi" của phiên `15:32:08` mà tôi đã ghi là "chưa xác định".** Nó
không phải "không có lỗi": nó là **không có việc gì được làm**. Cách sửa ở §5.11a — và khi viết bài
kiểm thử fMP4 đầu tiên thì lộ thêm **lỗi gốc thứ hai**: mốc cắt fMP4 sai 4 byte (§5.11b).
`[ĐO]` cho hiện tượng, `[ĐO]` cho nguyên nhân (bộ đếm chỉ ra), `[VERIFIED-DRYRUN]` cho bản sửa.

**Điểm đáng chú ý về YouTube:** host này append **dày** (477 lần), nên thỉnh thoảng lô vẫn đủ lớn để
cắt được (`aligns = 2`) — nhưng 39/44 lần giải mã vẫn hỏng, vì phần lớn lô đã bị vứt theo đúng cơ
chế trên rồi dựng lại từ một mảnh ⇒ file cụt giữa cluster ⇒ `invalid content`.

> **Ghi chú về `allConsumed`:** đây là trạng thái **BÌNH THƯỜNG** — nghĩa là "không có byte mới"
  (con trỏ đã ở cuối nhật ký). Con số lớn (642–762) **không** phải lỗi: nhịp chạy ~5 lần/giây còn
  trang chỉ append 17–33 lần trong cả phiên. Tín hiệu **bất thường** là `waitBoundary` **lớn hơn 0
  mà `waitBoundaryGaveUp` bằng 0** — van chờ tự reset mỗi nhịp, tức nó đang bị đói byte.

### 1.11 Ba phiên v8 (ba host) — 🎯 **CẢ BA ĐẠT, bốn mặt XANH ĐỒNG THỜI, lần đầu tiên** `[ĐO]`

| Host | codec | append | PCM | **liên tục** | `lead p50` | **`min`** | `max` | tốc | blob | **lỗi** | **cắt biên** | giữ init | **bỏ** | kẹt | phán quyết |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `play.vlstream.net` | fMP4/AAC | 21 | 153,98 s | **1,00** | **20,99** | **14,79** | 26,63 | 150,82× | 18 | **0** | **11** | 11 | **0** | 0 | ✅ **ĐẠT** |
| `www.xvideos.com` | fMP4/AAC | 34 | 149,82 s | **1,00** | **26,65** | **19,68** | 31,71 | 157,05× | 14 | **0** | **2** | 2 | **0** | 0 | ✅ **ĐẠT** |
| `www.youtube.com` | WebM/Opus | 472 | 229,63 s | **1,00** | **73,18** | **7,29** | 112,62 | 144,33× | 21 | **0** | **16** | 16 | **0** | 0 | ✅ **ĐẠT** |

Ngưỡng: `lead` > 3,5 s (cả `p50` **và** `min`), `liên tục` ≥ 0,97, lỗi < 10 %, `bỏ = 0`. **Ba host,
hai codec, không host nào trượt một mặt nào.** Đây là lần đầu tiên trong toàn bộ dự án đạt được
điều đó — trước v8, **chưa từng có phiên nào xanh đồng thời cả bốn mặt** (Phụ lục B cũ).

**Bốn điều đáng chú ý:**

1. **`cắt biên = 11` và `2` trên hai host fMP4** — lần **đầu tiên trong năm phiên** nhánh `moof`
   thật sự chạy và cho kết quả đúng, và `chữ ký: moof = 3` chứng minh bộ quét **nhìn thấy** `moof`
   thật (trước đây `moof ≤ 1` vì lô bị vứt nên chỉ quét được một mảnh mỗi lần). Đây là xác nhận
   trực tiếp cho chẩn đoán §5.11b: lỗi **sai 4 byte** ở mốc cắt chính là thứ chặn fMP4.
2. **`giữ init = cắt biên` ở cả ba host** (11/11, 2/2, 16/16) — không lần nào cắt mất init. Nghĩa là
   hỏng hóc `unknown content type` của hồi quy `14:28:18` **không tái diễn**.
3. **`waitBoundary` lớn (463–610) nhưng `waitBoundaryGaveUp` = 5–12** ⇒ van chờ **đang hoạt động
   đúng thiết kế**: phần lớn nhịp nó chờ một biên an toàn, và khi không lấy được thì **thoát ra** và
   giải mã trọn file. Đối chiếu với v7 (`GaveUp = 0`) là thấy ngay khác biệt: **cùng một con số
   `waitBoundary`, hai ý nghĩa hoàn toàn trái ngược.** Đây là lý do bài học "0 lỗi chưa chắc là
   không có việc gì" (Phụ lục B) phải đọc kèm **cặp** `waitBoundary` / `waitBoundaryGaveUp`.
4. **Biên an toàn đã rộng hơn hẳn:** `min = 7,29 / 14,79 / 19,68 s` so với ngưỡng 3,5 s ⇒ biên
   **2,1× / 4,2× / 5,6×**. Phiên v5 trước đây chỉ đạt `min = 4,58 s` (biên **1,3×**) và đó từng là
   nỗi lo lớn nhất của M2. YouTube vẫn là host **mỏng nhất** (2,1×) dù `p50` cao nhất — vì nó là
   phiên **dài nhất** (229,63 s PCM) nên có nhiều cơ hội chạm đáy hơn.

**Lỗi duy nhất trong log YouTube — và nó KHÔNG phải lỗi của đường offline:**

```
addModule(blob) bị chặn: The operation was aborted.
```

Đây là **Q9/Q16 đã ghi trong tài liệu 23**: CSP của YouTube chặn `audioWorklet.addModule()` với URL
`blob:`. Probe **tự chuyển sang `ScriptProcessor`** (log ghi `capture = scriptprocessor`) và **đường
`offline` không hề bị ảnh hưởng** — nó không nạp module nào cả, chỉ gọi
`OfflineAudioContext.decodeAudioData()`. Bằng chứng định lượng: đúng phiên YouTube có lỗi này lại là
phiên cho **`liên tục = 1,00` và 0 lỗi giải mã**, PCM 229,63 s. Hai host fMP4 không có lỗi này vì
`blob:` worklet **không bị chặn** ở đó (log ghi `capture = audioworklet-blob`).

⇒ **Hệ quả thiết kế cho M2.2, không đổi:** sản phẩm phải có **cả hai** đường truyền PCM (worklet +
ScriptProcessor), vì CSP khác nhau theo trang. Nhưng **đường `offline` là đường chính** và nó
**miễn nhiễm** với khác biệt CSP — đây là một lợi ích thiết kế của nó mà trước đây chưa được ghi
nhận rõ.

**Các con số còn lại của phiên YouTube (v8):** `initKind webm`, `initBoxes webm`,
`offlineNoInit 0`, `initHintWrong 0`, `cluster 3 / moof 0 / styp 0`, `sigMoofBad 0`,
`dropped = 1 836 172 B` (phần byte bị cắt bỏ để lấy biên — 16 lần cắt), `tickWhy` =
`{waitBoundary 463, allConsumed 191, waitBoundaryGaveUp 5, empty 1}`, `cursor 1/1` (nhật ký đã được
prune sau khi giải mã — trạng thái kết thúc bình thường).
### 2.1 Lập luận

Element ẩn phát với `playbackRate = 1`. Trang cũng phát `1×`. Hai đồng hồ tiến bằng nhau, nên:

```
lead(t) = vị trí element(t) − playhead trang(t) ≤ lead(0) + (1 − rate_trang)·t
```

Nếu `rate_trang = 1` thì `lead(t) ≤ lead(0)` — **khoảng lợi không bao giờ tăng**. `lead(0)` là
khoảng lợi lúc khởi động (thời gian trang prebuffer trước khi phát). **Đo được: 0,20 s và 0,19 s.**

Muốn tăng lead thì phải cho element phát nhanh hơn `1×`, nhưng:

* `preservesPitch = false` ⇒ PCM bị **nén thời gian + đổi cao độ** ⇒ ASR hỏng;
* `preservesPitch = true` ⇒ time-stretch (pha-vocoder) làm méo pha ⇒ chất lượng ASR giảm.

Kết luận: **đường element không thể là đường chính** cho bài toán chạy trước.

### 2.2 Vì sao điều này không mâu thuẫn với M1 — và số đo xác nhận

M1 đo `L = buffered.end − currentTime` — **byte đã nằm trong buffer**, không phải PCM. Buffer đầy
byte từ trước, nhưng PCM chỉ được sinh ra khi có bộ giải mã *chạy*. Bộ giải mã chạy theo đồng hồ
thời gian thực (element) thì không bao giờ bắt kịp khoảng cách byte đó.

Số đo phiên `14:10:42` xác nhận, trên **cùng một phiên**:

| | element (1×) | offline (`decodeAudioData`) |
| :--- | ---: | ---: |
| PCM lấy được | 128,38 s | **205,18 s** |
| lead p50 | 0,19 s | **82,76 s** |
| chunk bị đói (`stall`) | 501/2006 (25%) | — |

Element **đói 25% số chunk** vì luôn ở mép buffer, và lấy được **ít hơn 77 s PCM** so với offline
trong cùng khoảng thời gian. Phải có bộ giải mã **không bị ràng buộc thời gian thực** ⇒ `decodeAudioData`.

### 2.3 Đường offline (đường đã ĐẠT)

| Bước | Việc |
| :--- | :--- |
| 1 | Hook `SourceBuffer.prototype.appendBuffer`; gọi bản gốc **trước**; chép byte của buffer **audio** vào nhật ký (`Uint8Array` **bản sao**) |
| 2 | Đưa mảnh vào **hàng đợi**; xả khi shadow `sourceopen` (giữ đúng thứ tự, không mất init) |
| 3 | Trên `updateend` của shadow, đọc `buffered.end(last)` ⇒ **mép media** của mảnh vừa append |
| 4 | Mỗi nhịp (1 s), gom lô: init segment + các mảnh liên tiếp, cắt lô khi gặp dấu hiệu **tua thật** (mép tụt lùi hoặc nhảy xa), chốt lô ở mục **đã biết mép** và đủ ~10 s media |
| 5 | **Cắt file theo BIÊN PHÂN ĐOẠN** (chữ ký cluster / `moof`) — nếu không, file bắt đầu/kết thúc giữa một cluster và bị `EncodingError` (§5) |
| 6 | `Blob([cửa sổ])` → `arrayBuffer()` → `OfflineAudioContext(1,1,16000).decodeAudioData()` |
| 7 | Neo mép **cuối** PCM vào mép media của **mảnh cuối nằm trọn trong cửa sổ**: `startMedia = endMedia − buf.duration`; trộn mono; cắt chunk 1024 mẫu; **khử trùng theo `emittedTo`** |
| 8 | Đẩy vào `accPcm('offline', …)` — **cùng định dạng** `audio_chunk` mà `lib/audio-processor.js` phát ra |

Vì `decodeAudioData` giải mã theo lô chứ không theo đồng hồ, nó **nhanh hơn thời gian thực**:
đo được **136,79×** (1500 ms cho 205 s audio). Đây chính là thứ biến byte-lead thành PCM-lead.

### 2.4 Vấn đề mới mà đường offline tạo ra (phải xử ở M2.2)

Giải mã nhanh là dao hai lưỡi: giải mã hết 300 s buffer ngay thì **đẩy 300 s audio vào ASR trong
vài giây** ⇒ ngập backend (rủi ro R4 của tài liệu 22). Probe để `offlineTargetLeadSec = 0`
(**không chặn**) vì M2.1 cần đo **trần** của lead; khi làm sản phẩm phải đặt ~**30 s**.

Đây là **van tiết lưu tự nhiên**: lead chỉ cần lớn hơn 3,5 s, không cần bằng cả buffer. Số đo cho
thấy ta có **83 s** để tiêu — dư sức đặt van ở 30 s.

---

## 3. HAI PHƯƠNG PHÁP, ĐO SONG SONG

| | `element` (đối chứng) | `offline` (đường thiết kế) |
| :--- | :--- | :--- |
| Đường tiếng | element ẩn → `createMediaElementSource` → worklet hoặc ScriptProcessor | `decodeAudioData` theo lô |
| Tốc độ | 1× (đo được: **chậm hơn 1×**) | nhanh hơn thời gian thực |
| Lead | đo được **0,2 s** ⇒ không đạt | kỳ vọng ≈ lead byte của trang |
| CSP | **bị chặn trên YouTube** ⇒ phải dùng ScriptProcessor | **miễn nhiễm CSP** (API có sẵn, không nạp module) |
| Vai trò | **đối chứng** + phương án dự phòng | **đường thiết kế** |

Bộ đếm **tách riêng** (`S.cap.element` / `S.cap.offline`): gộp chung thì phân bố lead của hai
đường trộn vào nhau và câu hỏi trung tâm của M2.1 không còn trả lời được.

> **Ghi chú thiết kế quan trọng:** đường `offline` **không cần worklet**, nên nó không bị CSP
> chặn. Nếu đường offline chạy được thì vấn đề Q9 chỉ còn ảnh hưởng tới phương án dự phòng.

---

## 4. HAI MƯƠI LỖI THẬT ĐÃ SỬA (do dry-run và phiên thật bắt được) `[VERIFIED-DRYRUN]`

| # | Lỗi | Hậu quả nếu ra tới trình duyệt | Sửa |
| :-: | :--- | :--- | :--- |
| 1 | Mảnh tới **trước** `sourceopen` bị bỏ (`return` khi `!sh.sb`) | **Mất init segment** ⇒ shadow chỉ nhận mảnh media trần ⇒ `sb.error`, buffer rỗng, không PCM. **Đây là lỗi self-test đã bắt** | Mọi mảnh đi qua hàng đợi, xả theo thứ tự khi `sourceopen` |
| 2 | Hook `addSourceBuffer` **toàn cục** ⇒ shadow tự dựng lại chính nó | mirror **không chạy ở bất kỳ trang nào** | `WeakSet` các MediaSource của shadow |
| 3 | **Bỏ mảnh "không thêm media"** (`t1` trùng mép) | Trên YouTube mỗi phân đoạn bị cắt thành **26–57 lần append** dùng chung `buffered.end` (M1 §3) ⇒ bỏ là **làm đứt chuỗi byte** ⇒ file hỏng | Không bỏ byte nào; chống trùng bằng **khử trùng theo `emittedTo`** |
| 4 | `updateend` bị hiểu nhầm: dùng cờ boolean cho `remove()` | `remove()` có thể **không** phát `updateend` ⇒ cờ kẹt ⇒ `updateend` của append kế tiếp bị coi là remove ⇒ mục đó **mất mép media vĩnh viễn** | Đếm `pendingAppends` (bộ đếm không kẹt được) |
| 5 | Vòng chọn lô `break` khi gặp mục `t1 === null` | **Một** mục thiếu mép media làm đường offline **kẹt vĩnh viễn, im lặng** — đúng triệu chứng phiên `13:37:55` ("0 blob, 0 lỗi, còn chờ 594") | Không `break`: gom mọi byte, chốt lô ở mục **cuối cùng đã biết mép**, phần đuôi để nhịp sau |
| 6 | `off.busy` kẹt vì promise giải mã không bao giờ kết thúc | Offline chết vĩnh viễn, im lặng | Lưới an toàn 10 s: mở khoá + **trả con trỏ về đầu lô** (không mất tiếng) + ghi lý do |
| 7 | `logBytes` cộng lại cả mảng mỗi lần append | O(n²) với 2213–2752 append/phiên ⇒ giật trang | Bộ đếm cộng dồn `byteLogBytes` |
| 8 | Nhật ký byte **không bao giờ dọn** | Phiên dài chạm trần 32 MB ⇒ `logBytes` ngừng ghi ⇒ **offline chết giữa phiên** | `pruneByteLog()` giữ init segment + mục chưa giải mã |
| 9 | **Cắt mất INIT SEGMENT khi cắt theo biên phân đoạn** (lỗi *do tôi gây ra* ở bản v3) | Lỗi đổi sang `unknown content type`; **262/263** lần giải mã lỗi, lead **âm**, bỏ mất **425** mảnh — **tệ hơn cả khi chưa sửa** | Ghép init trở lại phía trước file + fail-safe "cắt-biên lỗi ⇒ lần sau giải mã nguyên file" (§5.4) |
| 10 | **Neo mốc media vào mảnh CUỐI LÔ trong khi file chỉ chứa tới mảnh cuối CỬA SỔ** (lỗi *do tôi gây ra* ở bản v4) | `lastEndMedia` quá lớn ⇒ mục kế tiếp bị coi là **tua lùi** ⇒ `break` ⇒ lô chỉ còn 1 mục không `t1` ⇒ **kẹt 423 nhịp**, PCM đứng ở 30 s khi trang đã 82 s ⇒ **lead ÂM** | Mốc neo **bắt buộc** thuộc mảnh cuối của file **và** có `t1`; nếu không có thì bỏ cắt |
| 11 | **`S.byteLog.indexOf(...)` để tính con trỏ** (lỗi *do tôi gây ra* ở bản v3) | Mục đã bị dọn khỏi nhật ký ⇒ `indexOf` trả **−1** ⇒ con trỏ **nhảy về 0** trong im lặng (quét lại từ đầu) | Ghi chỉ số **tuyệt đối** (`batchAbs`) lúc gom lô, không tra lại |
| 12 | **Thử giải mã khi mới có 1 biên phân đoạn** | xvideos: phân đoạn cuối **cụt** ⇒ `invalid content` ⇒ thử 12 lần ⇒ **bỏ** (156 lỗi, 17 mảnh mất, `liên tục 0,33`) | Cần **≥ 2 biên** mới cắt; mới có 1 biên thì **CHỜ** thêm byte |
| 13 | **Kẹt "chờ mép media" không có lối thoát** | Đường offline đứng yên **vĩnh viễn** mà chỉ ghi `why` — lead tụt âm dần | Sau 30 nhịp: bỏ qua mục chặn, **báo lỗi rõ** và đếm `offlineStuckSkips` (mất tiếng **có ý thức**, không im lặng) |
| 14 | **`frontCut = 0` khi vùng media bắt đầu ĐÚNG tại biên phân đoạn** (lỗi *do tôi gây ra*) | Khi init được ghép thêm ở đầu `concat`, vùng `[0, mediaStart)` **chính là init** ⇒ `frontCut = 0` nhét init vào giữa file **lần thứ hai**: `init + init + media` ⇒ `invalid content`. **Bắt được nhờ phần H của dry-run**, không phải nhờ phiên thật | `frontCut = mediaStart` (không phải 0) khi không cần cắt đầu |
| 15 | **Lô quá ngắn để cắt được biên** | `offlineMaxBatchSec = 10` mà phân đoạn của **cả hai host** đều ~5–10 s ⇒ lô chỉ chứa **1 phân đoạn ⇒ 1 biên** ⇒ không cắt đuôi an toàn ⇒ rơi vào đường "giải mã nguyên file" ⇒ file cụt ⇒ lỗi ⇒ **bỏ** (24 lỗi + 34 mảnh mất ở `15:06:23`) | `offlineMaxBatchSec = 30` ⇒ lô chứa ≥2–3 phân đoạn ⇒ luôn cắt được |
| 16 | **Điều kiện cắt đuôi quá chặt** | Đòi chữ ký cuối nằm ở **nửa sau file** ⇒ khi biên nằm ở nửa đầu (đuôi dài) thì **từ chối cắt** ⇒ lại giải mã file cụt | Bỏ điều kiện nửa-file; chỉ cần **một** biên và `back − frontCut ≥ 2048` |
| 17 | **BỊA init segment từ một mảnh MEDIA** (lỗi *do tôi gây ra*, hồi quy v6) | Dòng `if (!init) init = S.byteLog[0]`: không tìm thấy init thì lấy mảnh cũ nhất làm init. Mảnh cũ nhất **là mảnh media** ⇒ `mediaStart` = cỡ mảnh media ⇒ lọc sạch mọi chữ ký, VÀ file gửi đi **thiếu header** ⇒ `unknown content type` **193 lần**, **0 PCM**. Bằng chứng: `initBoxes = "O)?` — tên box vô nghĩa | Xoá dòng dự phòng; nhận diện init theo **NỘI DUNG** (EBML magic / box `ftyp`); không có init thì **báo** `offlineNoInit` |
| 18 | **Tin thứ tự append để nhận diện init** (lỗi *do tôi gây ra*, hồi quy v6) | Cờ init lấy từ `rec.appends === 1` ⇒ hỏng ngay khi bộ lọc `activeAudioId` chạy trước lúc biết đâu là SourceBuffer audio ⇒ mảnh init **thất lạc vĩnh viễn** | Giữ **mảnh đầu của MỌI SourceBuffer** trước mọi bộ lọc (`firstAppend`), và nhận diện init theo nội dung |
| 19 | **Nhánh thoát sớm không tự đếm** (thiếu sót *của tôi*) | Phiên xvideos `15:32:08` cho `0 blob, 0 lỗi, 0 bỏ` mà con trỏ lại chạy tới hết ⇒ **không thể biết** vì sao chỉ bằng log. Tôi đã phải suy đoán | Mọi nhánh thoát của `offlineTick` đi qua `setWhy()/bumpWhy()` ⇒ `stats.tickWhy` liệt kê **từng nhánh và số nhịp** |
| 20 | **CHỐT CON TRỎ TRƯỚC KHI GIẢI MÃ** — gốc của hồi quy v6/v7 (lỗi *do tôi gây ra*) | `off.cursor = lastKnownAbs + 1` chốt con trỏ **trước** khi biết lô có giải mã được hay không. Nhánh "chờ biên an toàn" `return` mà **không khôi phục con trỏ** ⇒ **lô bị vứt** ⇒ nhịp sau chỉ còn 1 mảnh vừa tới ⇒ van chờ **không bao giờ tích luỹ đủ 40 nhịp** ⇒ **không lần nào gọi `decodeAudioData`**. Đo được: xvideos + vlstream **0 PCM, 0 lần gọi, 0 lỗi** — trạng thái "0 lỗi" mà thực chất là **không có việc gì được làm** | Con trỏ **chỉ chốt khi byte đã THỰC SỰ được tiêu thụ**: giải mã xong (`decoded`) hoặc bỏ có chủ ý (`skipped`). Mọi nhánh "chờ" để nguyên con trỏ; dựng lại lô là **idempotent** vì mục `decoded` bị vòng lặp bỏ qua |
| 21 | 🔴 **`moof` BỊ COI LÀ ĐẦU BOX — SAI 4 BYTE** (lỗi *của tôi*, có từ bản đầu; **đây là lời giải cho fMP4**) | `moof`/`styp` là **trường TYPE** của box ISO-BMFF, nên box bắt đầu ở `i − 4` (chỗ ghi `size`), **không** phải ở `i`. Bản cũ `out.push(i)` ⇒ **sai 4 byte ở CẢ HAI đầu**: file bắt đầu bằng 4 chữ `m o o f` ngay tại vị trí đáng lẽ là `size` ⇒ bộ phân tích đọc size = **0x6D6F6F66 ≈ 1,8 GB** ⇒ **hỏng toàn bộ cây box ⇒ `invalid content`**; và đuôi file cụt 4 byte vào trong `moof` cuối. Bằng chứng **kiểm chứng ngược**: cài lại `push(i)` ⇒ phần L của dry-run cho `size@216 = 1836019558` (đúng 0x6D6F6F66) và `@220 = [66,73,80,87]` (byte payload ngẫu nhiên thay vì `moof`) | Trả về **đầu box** (`i − 4`) cho `moof`/`styp`; WebM giữ nguyên `i` (EBML ID **là** byte đầu phần tử). Kèm **kiểm tra cấu trúc**: đọc `size` 4 byte trước đó, đòi `≥ 8`, `≤ 64 MB`, và box nằm **trọn** trong đệm — loại luôn dương tính giả (đếm `sigMoofBad`/`sigStypBad`) |

Ngoài ra dry-run còn kiểm: `timestampOffset` đồng bộ sang shadow (player HLS đặt offset khác 0 ⇒
lệch cả timeline), `catch-up` khi trang **resume ở phút thứ 8** (M1 §4.4 đo `ct = 478 s`), và
eviction **không được xoá dữ liệu mà shadow chưa giải mã tới**.

**Vì sao "0 blob giải mã, 0 lỗi, còn chờ N" là trạng thái tệ nhất:** nó không phân biệt được
mười sáu nguyên nhân khác nhau. Nay mọi nhánh thoát sớm ghi `offlineState.why`, và panel in thẳng ra:

```
offline: 0 blob giải mã, 0 lỗi, tăng tốc —, còn chờ 594 mục
offline ⛔ chưa có mép media (t1) cho mục nào   ← dòng mới: nói thẳng vì sao
```

---

## 5. SỬA KHIẾM KHUYẾT CÒN LẠI — VÀ MỘT HỒI QUY TỰ GÂY RA `[VERIFIED-DRYRUN]`

### 5.1 Bằng chứng

Phiên `14:10:42`: **72/94 lần giải mã lỗi**, tất cả cùng một thông báo:

```
EncodingError: The buffer passed to decodeAudioData contains invalid content
               which cannot be decoded successfully
```

⇒ 89 mảnh bị bỏ (`skipped`), `liên tục = 0,82` (mất ~45 s trên 250 s media đã bọc).
Kích thước các lô bị bỏ: 6 mảnh/74 KB, 17 mảnh/157 KB, 15 mảnh/92 KB, 32 mảnh/137 KB,
18 mảnh/116 KB… và **1 mảnh/46 KB**.

### 5.2 Nguyên nhân

Lô được chốt theo **thời gian media** (`endMedia − firstKnown ≥ 10 s`) tại mục cuối **đã biết
mép media**. Nhưng M1 §3 đã đo: **append của trang KHÔNG trùng biên phân đoạn** — YouTube cắt mỗi
phân đoạn thành 26–57 lần append, phần lớn cắt **giữa cluster**. Nên điểm chốt lô rơi vào **giữa
một cluster** ⇒ file = `init + cluster trọn + cluster CỤT` ⇒ bộ giải mã từ chối cả file.

Và vì lô **đã chốt** ở mục đó nên khi lỗi, lô **không lớn thêm được**: cơ chế "thử lại khi lô lớn
dần" không có byte mới để lớn ⇒ hết 12 lần thử ⇒ **bỏ cả lô**. Một điểm chốt tồi làm mất luôn
mọi thứ sau nó.

### 5.3 Cách sửa — và một HỒI QUY tự gây ra (bài học đắt nhất của M2)

Ý tưởng: quét **chữ ký phân đoạn** trong byte (byte đã có sẵn trong nhật ký, **không cần demux**)
rồi chỉ giải mã vùng kết thúc **đúng** biên phân đoạn:

| Định dạng | Chữ ký | Ý nghĩa |
| :--- | :--- | :--- |
| WebM/Matroska | `1F 43 B6 75` | EBML Cluster ID |
| fMP4 / ISO-BMFF | `6D 6F 6F 66` | box `moof` |
| (dự phòng) | `73 74 79 70` | box `styp` |

**Bản đầu tiên (SAI):** cắt file thành `concat[firstCluster .. lastCluster)`. Nghe hợp lý, nhưng
`firstCluster` là cluster **ĐẦU TIÊN** — mà init segment nằm **TRƯỚC** nó. Nên bản sửa này **cắt mất
init segment**. Kết quả đo được (phiên `14:28:18`):

| | trước khi cắt biên (`14:10:42`) | cắt biên SAI (`14:28:18`) |
| :--- | ---: | ---: |
| lỗi giải mã | 72 / 94 | **262 / 263** |
| thông báo lỗi | `invalid content` | **`unknown content type`** ← mất định dạng |
| PCM lấy được | 205,18 s | **19,97 s** |
| lead p50 | +82,76 s | **−11,22 s** |
| mảnh bị bỏ | 89 | **425** |

Đây là bài học: **`invalid content` và `unknown content type` là hai lỗi KHÁC NHAU và chỉ đúng hai
nguyên nhân khác nhau.** Lỗi thứ hai gần như luôn có nghĩa "file không có phần mô tả định dạng /
init segment". Nếu không đọc kỹ thông báo lỗi, rất dễ "sửa" theo hướng sai.

### 5.4 Cách sửa ĐÚNG (v4)

Không phải "đừng cắt đầu", mà là **GHÉP INIT TRỞ LẠI phía trước**: file =
`[init segment] + [vùng media từ biên phân đoạn tới biên phân đoạn)`. Vì khi lô nối tiếp giữa một
cluster thì phần media dở dang **phải** cắt, mà init thì **luôn** phải có — hai việc độc lập nhau.

| # | Quy tắc | Vì sao |
| :-: | :--- | :--- |
| 1 | **Init luôn được ghép ở đầu file** | Cắt init ⇒ `unknown content type` (hồi quy ở §5.3) |
| 2 | **Chỉ cắt đầu khi mảnh đầu của lô KHÔNG bắt đầu tại biên phân đoạn** | Nếu lô vốn đã trùng biên (xvideos: 58 KB/append) thì không cần cắt gì |
| 3 | Phần đầu bị cắt không quá **⅓** file | Chữ ký 4 byte có thể **dương tính giả** trong payload Opus/AAC — cắt nhầm nhiều là **mất tiếng thật** |
| 4 | Cắt đuôi tại chữ ký **cuối**; giữ được **≥ ½** file | Cluster cuối bị cụt là nguyên nhân gốc của 72 lỗi |
| 5 | **Phần cắt đuôi KHÔNG mất** — mảnh chứa nó không được đánh dấu đã giải mã | Lô sau lấy lại, và được cắt đầu đúng tại cluster đó |
| 6 | **Con trỏ chỉ nhảy tới sau mảnh cuối nằm trọn trong cửa sổ** | Không nhảy hết lô (nếu không là mất tiếng) |
| 7 | **Fail-safe: lô nào cắt-biên mà lỗi thì lần sau giải mã NGUYÊN file** | Bảo đảm cắt-biên **không bao giờ** làm mọi thứ tệ đi so với trước |

Bộ đếm mới để kiểm chứng sống: `stats.segmentAligns` (số lô cắt đúng biên),
`stats.segmentAlignKeptInit` (số lô giữ được init), `stats.segmentAlignWithInit` (vừa cắt biên vừa
giữ init — **trường hợp đúng**), `stats.segmentAlignRetryNoCut` (số lần phải giải mã lại nguyên file
vì cắt biên bị lỗi), `stats.segmentAlignDroppedBytes` (byte bị hoãn).

Dry-run (**160/160**) nay có **bốn** phần kiểm việc này:
* **Phần E** — luồng WebM cắt vụn 12 mảnh/4 cluster: khẳng định file gửi đi giải mã **bắt đầu bằng
  init `1A 45 DF A3`** và **ngay sau init là chữ ký cluster**.
* **Phần F** — lô **nối tiếp giữa một cluster** (mảnh đầu không có chữ ký): khẳng định file là
  `init + vùng đã cắt` với **kích thước đúng của phép ghép** (5 900 byte, không phải cả 9 200 byte),
  và phần đầu dở dang **không** nằm trong file.
* **Phần G** — **kẹt "chờ mép media"** (mục không có `t1` ngay trước một mục bị coi là "tua lùi"):
  khẳng định **con trỏ vẫn tiến** sau 40 nhịp, có ghi lỗi rõ, và có đếm `offlineStuckSkips`.

### 5.5 (điều kiện ĐẠT — xem bản cập nhật ở §5.9)

### 5.6 Sửa tiếp (v5) — hai lỗi do bản v4 gây ra

Phiên `14:38:30` chứng minh bản v4 **sửa đúng** chuyện giải mã (`lỗi 0`, `bỏ 0`, `liên tục 0,9989`)
nhưng lộ ra **hai lỗi mới, đều do tôi**:

| # | Lỗi | Cơ chế | Sửa |
| :-: | :--- | :--- | :--- |
| 10 | **Neo mốc media sai mảnh** | Mảnh cuối nằm trọn trong cửa sổ **không có** `t1` ⇒ code cũ vẫn neo bằng `t1` của mảnh cuối **lô** (muộn hơn hẳn vùng byte trong file) ⇒ `lastEndMedia` quá lớn ⇒ mục kế tiếp bị coi là **tua lùi** ⇒ `break` ⇒ lô chỉ còn 1 mục không `t1` ⇒ nhánh "chờ mép media" ⇒ **lặp y nguyên 423 nhịp** | Mốc neo **bắt buộc** là mảnh cuối **của file** *và* có `t1`; nếu không tìm được thì **bỏ cắt**, giải mã nguyên file |
| 11 | **`indexOf` để tính con trỏ** | `S.byteLog.indexOf(batch[lastIncluded])` trả **−1** khi mục đã bị dọn ⇒ con trỏ **nhảy về 0** trong im lặng | Ghi chỉ số **tuyệt đối** (`batchAbs`) lúc gom lô |
| 12 | **Thử giải mã khi mới có 1 biên** | xvideos: phân đoạn cuối **cụt** ⇒ `invalid content` ⇒ thử 12 lần ⇒ **bỏ** (156 lỗi, 17 mảnh mất, `liên tục 0,33`) | Cần **≥ 2 biên** mới cắt; mới có 1 biên thì **CHỜ** thêm byte (giữ nguyên con trỏ) |
| 13 | **Kẹt không có lối thoát** | Đường offline đứng yên **vĩnh viễn**, chỉ ghi `why` ⇒ lead tụt âm dần | Sau 30 nhịp: bỏ mục chặn, **báo lỗi rõ**, đếm `offlineStuckSkips` — mất tiếng **có ý thức**, không im lặng |

> **Nguyên tắc rút ra:** mọi phép biến đổi byte phải có **đường lùi** (fail-safe), và mọi trạng thái
> "chờ" phải có **van thoát**. Hai hồi quy vừa rồi đều thuộc đúng hai loại này.

Dry-run nay **160/160**, thêm **phần G** (ca kẹt) và **phần H** (biên ở nửa đầu file): mục không có `t1` nằm ngay trước một
mục bị coi là "tua lùi" ⇒ khẳng định **con trỏ vẫn tiến** sau 40 nhịp, có ghi lỗi, và có đếm
`offlineStuckSkips`.

### 5.8 Sửa tiếp (v6) — đổi bản chất khiếm khuyết: từ "lead âm / kẹt" sang "độ phủ"

Hai phiên v5 cho thấy **hai hồi quy trước đã hết** (kẹt: 423 → **1** nhịp; lead: âm → **+43,50 s**
và **+24,59 s**, `belowPct = 0 %` trên cả hai host). Khiếm khuyết còn lại **đổi bản chất** thành
**độ phủ** (`liên tục` 0,30–0,40): PCM có **khoảng trống** vì các lô lỗi bị bỏ. Ba nguyên nhân, đều
đã sửa:

| # | Nguyên nhân | Bằng chứng đo được | Sửa (v6) |
| :-: | :--- | :--- | :--- |
| 15 | **Lô 10 s ≈ 1 phân đoạn** ⇒ chỉ **1 biên** ⇒ không cắt đuôi an toàn ⇒ giải mã file cụt | 24 lỗi + **34 mảnh bị bỏ** ở `15:06:23`; lô lỗi là "17 mảnh / 97 829 byte" | `offlineMaxBatchSec = 30` ⇒ lô chứa ≥2–3 phân đoạn |
| 16 | **Đòi chữ ký cuối ở nửa sau file** ⇒ từ chối cắt khi đuôi dài | xvideos `cắt biên = 0` suốt 3 phiên dù mime là `audio/mp4` | Chỉ cần **một** biên; cắt đuôi tại **đầu phân đoạn cuối** |
| 14 | **`frontCut = 0`** khi vùng media bắt đầu đúng tại biên ⇒ file = `init + init + media` | **Bắt được nhờ phần H của dry-run**, không phải phiên thật | `frontCut = mediaStart` |

Thêm luật: **chờ thay vì bỏ.** Khi có biên nhưng chưa cắt được an toàn, probe **CHỜ thêm byte**
(giữ nguyên con trỏ) thay vì giải mã file cụt rồi bỏ — vì "thử-rồi-bỏ" **chính là** cách 34 mảnh bị
mất. Chỉ sau 40 nhịp chờ mới thử giải mã trọn file (van thoát cho trường hợp trang đã ngừng append).

**Phân đoạn cuối cùng KHÔNG bị mất:** nó chỉ bị **hoãn** sang lô sau, và lô sau cắt đầu đúng tại
biên đó. Chỉ phân đoạn cuối của **cả phiên** là không được giải mã (mất ~5–10 s ở đuôi).

**Điều tra thêm cho xvideos** (`cắt biên = 0` là chưa giải thích được): probe v6 **đếm chữ ký theo
loại** (`cluster` / `moof` / `styp`) và **liệt kê tên các box ở đầu init segment** (`ftyp`, `moov`,
…) — **chỉ tên box, không lưu nội dung media**. Phiên sau sẽ trả lời dứt điểm: fMP4 của xvideos có
`moof` hay không, thay vì đoán.

> **Nguyên tắc (đã thành lệ của dự án):** *mọi phép biến đổi byte phải có đường lùi; mọi trạng thái
> "chờ" phải có van thoát; và một lỗi không được sửa bằng cách thử-rồi-bỏ.*

### 5.9 Điều kiện ĐẠT — nay **ĐÃ ĐẠT TOÀN BỘ, trên BA host** (§1.11)

| # | Điều kiện | Nguồn | v8 |
| :-: | :--- | :--- | :--- |
| 1 | `pcm.offline.chunks > 0`, `rmsAvg > 0`, **`continuityRatio ≥ 0,97`** | PCM hợp lệ và liên tục | ✅ **1,00** cả ba host |
| 2 | **`stats.blobDecodeErrors` / (`blobsDecoded` + lỗi) < 10 %** | cắt theo biên phân đoạn hiệu quả | ✅ **0 lỗi** cả ba |
| 3 | **`lead.offline.p50 > 3,5 s`** | **câu hỏi quyết định** | ✅ **+20,99 / +26,65 / +73,18 s** |
| 4 | **`lead.offline.p5`/`min` phải > 3,5 s** (không chỉ p50) | p50 tốt **không** bảo đảm lúc xấu nhất tốt | ✅ **min = 14,79 / 19,68 / 7,29 s**, `belowPct = 0 %` |
| 5 | **`stats.segmentAlignWithInit > 0`** mỗi khi có cắt biên | ⚠️ nếu `segmentAligns > 0` mà `segmentAlignWithInit = 0` thì **init đang bị cắt** (§5.3) | ✅ **11/11, 2/2, 16/16** |
| 6 | **`offlineState.stuckBursts = 0`** | ⚠️ kẹt "chờ mép media" = offline **đứng yên** ⇒ lead âm dần (phiên `14:38:30`) | ✅ 0 cả ba |
| 7 | `offlineState.why` **trống** và `lastErr` **null** | không còn gì cản đường offline | ✅ |
| 8 | `mirrorErrors = 0`, không `QuotaExceededError` | cửa sổ trượt + eviction đủ | ✅ 0 |
| 9 | `stats.offlineGuards = 0` **và `stats.offlineStuckSkips = 0`** | decode không treo, và không phải bỏ mục để gỡ kẹt | ✅ 0 |
| 10 | **`stats.offlineNoInit = 0`** và `initKind` ∈ {`webm`, `mp4`} | ⚠️ **v7** — không có init segment thì file ghép ra là rác (hồi quy `15:28:54`) | ✅ `webm` / `mp4` / `mp4`, `noInit 0` |
| 11 | **`stats.tickWhy` chỉ chứa `busy`** (và `element` nếu chạy chế độ element) | ⚠️ **v7** — nhưng xem **đính chính ở #12** | xem #12 |
| 12 | 🎯 **`tickWhy.waitBoundary` chỉ được coi là BÌNH THƯỜNG khi `waitBoundaryGaveUp > 0`** | ⚠️ **đính chính ở v8** — `waitBoundary` lớn là **trạng thái hoạt động bình thường** (van đang chờ biên an toàn); nó chỉ là **lỗi** khi `GaveUp = 0` (van đói byte, §5.11a). **Cặp số này phải đọc cùng nhau** | ✅ `waitBoundary` 463–610 **kèm** `GaveUp` 5–12 |
| 13 | 🎯 **`sigMoofBad` = 0** trên dữ liệu thật | ⚠️ **v8** — nếu > 0 nghĩa là payload có chứa chuỗi `moof` tình cờ **và** bộ kiểm tra cấu trúc đã chặn được; nếu con số lớn thì nên xem lại ngưỡng quét | ✅ 0 cả ba |

> ⚠️ **Điều kiện #11 đã được ĐÍNH CHÍNH ở #12 sau khi có phiên v8.** Bản v7 viết #11 quá chặt: nó
> coi *mọi* giá trị khác `busy` là dấu hiệu đường offline không chạy. Phiên v8 cho thấy
> `waitBoundary` lớn chính là **cách thiết kế hoạt động** (chờ biên an toàn trước khi cắt). Nếu giữ
> #11 nguyên xi thì **cả ba phiên ĐẠT đều bị chấm là KHÔNG ĐẠT** — một tiêu chí sai làm hỏng đúng
> thứ nó định bảo vệ. Ghi lại đây vì đây là dạng lỗi dễ mắc nhất khi viết tiêu chí: **một ngưỡng
> đúng cho trạng thái hỏng có thể sai cho trạng thái lành.**
>
> **Hệ quả cho bộ phân tích:** `m2_analyze.py` nay chỉ coi `waitBoundary` là **lỗi** khi
> `waitBoundaryGaveUp = 0`; phân bố `tickWhy` được in ra như **thông tin (`ℹ️`)** và **không** làm
> phiên bị chấm trượt. Đồng thời phán quyết `PASS` nay đòi **cả `p50` lẫn `min`** > 3,5 s — trước
> đó mã chỉ kiểm `p50`, tức **tiêu chí #4 của báo cáo chưa từng được máy kiểm**.

Với **2×**: lead là khoảng cách trong timeline media, **không** chia 2; ngân sách wall-clock mới là
`lead / playbackRate` (ở 2×, 82,76 s ⇒ ~41 s wall-clock). Bộ phân tích in lead thô.

### 5.10 Sửa tiếp (v7) — không đoán nữa: nhận diện init theo NỘI DUNG, và mọi nhánh tự đếm

Hai phiên v6 cho **0 PCM trên cả hai host**. Một nguyên nhân tìm ra chắc chắn (§1.9), một nguyên
nhân **chưa xác định**. Cách sửa vì thế có hai phần: **sửa cái đã biết**, và **làm cho cái chưa
biết trở nên đọc được từ log** — chứ không thêm một tầng suy đoán nữa.

**a) Nhận diện init segment theo NỘI DUNG, không theo thứ tự append**

| | Cũ (v6) | Mới (v7) |
| :--- | :--- | :--- |
| Nguồn cờ init | `rec.appends === 1` (thứ tự append) | **nội dung**: EBML magic `1A 45 DF A3` ⇒ `webm`; box `ftyp` ở offset 4 ⇒ `mp4` |
| Khi không tìm thấy init | `init = S.byteLog[0]` — **lấy mảnh media làm init** ⇒ 193 lỗi | **xoá dòng đó**; đếm `offlineNoInit` và nói thẳng là không có |
| Mảnh init có thể thất lạc? | **Có** — bộ lọc `activeAudioId` có thể chạy trước | **Không** — mảnh đầu của **mọi** SourceBuffer được giữ riêng (`firstAppend`) **trước mọi bộ lọc** |
| Đối chiếu | — | gợi ý sai được đếm riêng: `initHintWrong` |

Nội dung là **quan toà**, thứ tự chỉ là gợi ý. Lý do: init của WebM **luôn** mở đầu bằng EBML
header, init của fMP4 **luôn** mở đầu bằng `ftyp` — đó là ràng buộc của định dạng, không phải
quan sát từ một trang. Còn thứ tự append thì phụ thuộc vào việc hook của ta có kịp hay không,
tức phụ thuộc đúng vào thứ ta không kiểm soát được.

**b) Mọi nhánh thoát sớm của `offlineTick` tự đếm mình**

`stats.tickWhy = {mã nhánh: số nhịp}`, với các mã `element`, `busy`, `empty`, `allConsumed`,
`emptySeekBack`, `noMediaEdge`, `waitBoundary`, `waitBoundaryGaveUp`. Panel in thẳng dòng
`vì sao offline dừng: {...}`.

Đây là bài học từ chính phiên `15:32:08`: `0 blob, 0 lỗi, 0 bỏ` **trông như** mọi thứ ổn, nhưng
thực ra đường offline **chưa từng thử làm gì**. Một bộ đếm "0 lỗi" không cho biết là "không có
lỗi" hay "không có việc gì được làm". Từ v7, log trả lời câu đó mà không cần tôi đọc mã.

**c) Không sửa được bằng cách thử-rồi-bỏ (luật §5.8 vẫn giữ)**

Không nhánh nào của v7 bỏ byte để "đi tiếp". Chỗ duy nhất còn bỏ là van thoát gỡ kẹt (`stuckBursts
≥ 30`) và nó **đếm riêng** (`offlineStuckSkips`), nên không thể lẫn với đường chạy bình thường.

**Kiểm thử:** `node scratch/m2_mirror/m2_dryrun.mjs` → **160/160**, thêm **phần I**: nhật ký
**không có** init segment ⇒ phải (1) đếm `initHintWrong`, (2) **không** mục nào bị bịa thành init,
(3) đếm `offlineNoInit`, (4) vẫn cắt được theo biên, (5) file gửi đi **không** mở đầu bằng init bịa.
Đây là bài kiểm thử **lẽ ra phải có từ trước** — bộ **129/129** cũ vẫn xanh trong khi hành vi thật
xấu đi, vì nó chỉ kiểm thứ tự append đúng, chưa bao giờ kiểm *thiếu* init.

```
python scratch/m2_mirror/m2_analyze.py          # bảng so sánh
python scratch/m2_mirror/m2_analyze.py --md     # dán vào báo cáo
```

Bộ phân tích nay in thẳng **lý do** khi offline không ra PCM, và cảnh báo khi chỉ có element:

```
⛔ www.xvideos.com [...] : offline KHÔNG ra PCM — lỗi giải mã (6/12): EncodingError: ... unknown content type
ℹ️  mảnh tới trước sourceopen (init segment): www.xvideos.com×1, www.youtube.com×1
✗ www.youtube.com [...] : ⏳ kẹt chờ mép media 423 nhịp: chờ mép media (t1) — 1 mục, 423 nhịp
✗ www.xvideos.com [...] : PCM đứt (liên tục 0.3327); ⚠️ BỎ MẤT 17 mảnh (mất tiếng thật)
```

---


### 5.11 Sửa tiếp (v8) — HAI lỗi gốc: con trỏ chốt sớm, và `moof` không phải đầu box

Ba phiên v7 (§1.10) cho ba kết quả khác nhau nhưng **một nguyên nhân duy nhất**, và bộ đếm thêm ở
§5.10 chính là thứ chỉ ra nó: `waitBoundary` 16–25 nhịp trên **cả ba host** trong khi
`waitBoundaryGaveUp = 0`. Khi sửa xong lỗi đó và viết bài kiểm thử fMP4 đầu tiên, **một lỗi thứ hai
lộ ra** — lỗi giải thích vì sao fMP4 chưa lần nào cắt biên được.

#### a) Con trỏ chỉ được chốt khi byte ĐÃ THỰC SỰ được tiêu thụ

| | Trước (≤ v7) | Sau (v8) |
| :--- | :--- | :--- |
| Chốt con trỏ | `off.cursor = lastKnownAbs + 1` — **trước** khi thử giải mã | **Không** chốt ở đây. Chỉ chốt khi (a) giải mã xong: `batchAbs[lastIncluded] + 1`; (b) bỏ có chủ ý: `batchAbs[0] + 1` + `skipped = true` |
| Nhánh "chờ biên an toàn" | Vứt lô (con trỏ đã đi qua) ⇒ van **đói byte** | Con trỏ đứng yên ⇒ lô **lớn lên** mỗi nhịp ⇒ van 40 nhịp hoạt động đúng thiết kế |
| Hệ quả đo được | xvideos + vlstream: **0 lần gọi giải mã**, 0 PCM; YouTube: `aligns 2`, 39/44 lần giải mã hỏng | Chưa đo — **cần phiên mới** |

#### b) `moof`/`styp` là TRƯỜNG TYPE, box bắt đầu ở `i − 4`

Box ISO-BMFF có dạng `[size 4 byte][type 4 byte][payload]`. Nên `moof` nằm ở `box_start + 4`, và mốc
cắt **đúng** là `box_start = i − 4`. Bản cũ trả về `i`, tức **sai 4 byte ở cả hai đầu file**:

* **Đầu:** file bắt đầu bằng 4 chữ `m o o f` ngay tại vị trí đáng lẽ là `size` ⇒ bộ phân tích đọc
  `size` = **0x6D6F6F66 ≈ 1,8 GB** rồi lấy 4 byte kế làm `type` ⇒ **hỏng toàn bộ cây box**.
* **Đuôi:** `winEnd = i` ⇒ file kết thúc **4 byte vào trong** header của `moof` cuối ⇒ đuôi cụt.

**Đây là lời giải cho câu hỏi để ngỏ suốt năm phiên:** vì sao hai host fMP4 có `initBoxes = ftyp,moov`
đúng, mime đúng `audio/mp4`, mà `cắt biên` hoặc `= 0` hoặc (khi chạy được, phiên `14:30:19`) lại cho
**174 lỗi giải mã**. Cắt theo mốc sai 4 byte thì file **luôn** hỏng — và hỏng theo kiểu
`invalid content`, đúng lỗi đã chiếm ưu thế trên cả hai host fMP4.

Kèm theo, mỗi mốc cắt nay phải qua **kiểm tra cấu trúc**: đọc `size` 4 byte ngay trước chữ ký, đòi
`≥ 8`, `≤ 64 MB`, và box nằm **trọn** trong đệm. Đây là cách loại **dương tính giả** — rủi ro M3 đã
ghi ở Phụ lục B: chữ ký 4 byte có thể xuất hiện tình cờ trong payload Opus/AAC, và cắt nhầm ở đó làm
**mất tiếng thật mà KHÔNG sinh lỗi giải mã**, nên fail-safe không bắt được. Số ứng viên bị từ chối
được đếm riêng (`sigMoofBad`, `sigStypBad`).

#### c) Kiểm thử — hai phần mới, cả hai đều **kiểm chứng ngược**

`m2_dryrun.mjs` nay **160/160**:

| Phần | Nội dung | Kiểm chứng ngược (cài lại lỗi cũ) |
| :--- | :--- | :--- |
| **J** | nhịp "chờ" phải **tích luỹ byte**; `cursor` đứng yên khi chờ | ✗ `calls=0` — không giải mã lần nào |
| **K** | fMP4 thật (box ISO-BMFF), append **tách** `moof`/`mdat`; cắt đúng đầu box; `moof` giả trong payload bị từ chối | ✗ `len=5820` (kỳ vọng 5816) — đuôi cụt 4 byte |
| **L** | fMP4 khi lô bắt đầu bằng `mdat` mồ côi ⇒ cắt **đầu** file | ✗ `size@216 = 1836019558` (**= 0x6D6F6F66**) và `@220 = [66,73,80,87]` |

Phần L là bài kiểm thử **đắt giá nhất của cả M2**: nó tái hiện **đúng con số** của cơ chế hỏng hóc
(đọc chữ `moof` thành `size` ≈ 1,8 GB) — không phải mô tả định tính, mà là **giá trị byte cụ thể**.
Đây cũng là bài kiểm thử fMP4 **đầu tiên** của M2; suốt năm phiên fMP4 chỉ được đo trên trang thật
mà **không hề có kiểm thử offline**, và đó là lý do một lỗi 4 byte sống sót qua năm phiên.

## 6. RỦI RO CÒN LẠI

| # | Rủi ro | Vì sao chưa loại được | Cách xử nếu xảy ra |
| :-: | :--- | :--- | :--- |
| M1 | `decodeAudioData` **từ chối** cách ghép init + media segment với **fMP4/AAC** | ✅ **ĐÃ LOẠI** — phiên xvideos `15:04:34` cho **`lead p50 = +24,59 s`**, `min = 9,48 s`, 4 blob giải mã thành công trên `mp4a.40.2` `[ĐO]` | — (không cần WebCodecs cho fMP4) |
| M2 | Lô cắt **giữa phân đoạn** ⇒ cluster cụt | ✅ **ĐÃ LOẠI** — v8: **0 lỗi** trên cả ba host; `cắt biên` 11/2/16 và `giữ init` đủ cả. Gốc thật là **mốc fMP4 sai 4 byte** (§5.11b) | — |
| M3 | Dương tính giả của chữ ký 4 byte ⇒ cắt nhầm, **mất tiếng** | ✅ **ĐÃ GIẢM MẠNH** — v8 đọc `size` của box (`≥ 8`, `≤ 64 MB`, trọn trong đệm) cho `moof`/`styp`; chữ ký giả bị **từ chối** và đếm ở `sigMoofBad`. **Nhưng WebM vẫn quét thô**: EBML Cluster ID **là** đầu phần tử nên không có `size` để kiểm ⇒ dương tính giả trong Opus **vẫn có thể** | Theo dõi `segmentAlignDroppedBytes` bất thường; nếu cần thì đọc cả **EBML element size** (VINT) cho WebM |
| M4 | Lô bị **bỏ** sau nhiều lần lỗi ⇒ **mất tiếng** | ✅ **ĐÃ LOẠI** — v8: **`bỏ = 0`** trên cả ba host (trước đó 34 / 10 / 38 / 68 mảnh) | — (giữ luật "chờ thay vì bỏ") |
| M5 | Chi phí CPU/bộ nhớ khi giữ byte để giải mã lại | 124–157× là con số tốt, nhưng chưa đo RAM thật | trần 32 MB + prune theo cửa sổ |
| M6 | CSP chặn worklet (đã xảy ra trên YouTube) | ✅ **Đã rõ**: theo **từng trang** (YouTube chặn, xvideos không). ScriptProcessor chạy tốt; **chưa đo CPU/jank** | Sản phẩm **phải** có cả hai đường; đường `offline` miễn nhiễm CSP |
| M7 | Nén 2× (trang phát 2×) ⇒ element phát nhanh hơn, nhưng **offline không đổi** | Chưa đo ở 2× | lead tính theo timeline media nên **không** bị chia 2; chỉ ngân sách wall-clock giảm |
| M8 | Cắt theo biên có thể **làm chậm tiến độ** (hoãn đuôi mỗi lô) | Chưa đo throughput ở phiên dài | Đo `blobsDecoded`/phút; nếu chậm thì tăng cỡ lô |
| M9 | **Hồi quy do sửa lỗi** — một bản sửa có thể làm mọi thứ tệ hơn mà vẫn "hợp lý" trên giấy | 🔴 **ĐÃ XẢY RA NĂM LẦN** (v3 lead +82,76 → **−11,22**; v4 `lỗi 0` mà lead **−9,69**; v6 **0 PCM**; v7 **0 PCM cả ba host**; **v8b** lỗi `moof` sai 4 byte) | Bài học: **fail-safe** cho mọi phép biến đổi byte, **van thoát** cho mọi trạng thái chờ, **luôn** so số với phiên trước — và **kiểm thử offline cho MỌI định dạng** trước khi tiêu phiên đo của người dùng |
| M10 | **Kẹt "chờ mép media"** làm offline đứng yên | ✅ **Đã hết** (v5: **1** nhịp, không còn 423) | Đã sửa: neo đúng mảnh + van thoát 30 nhịp + `offlineStuckSkips` |
| M11 | **Độ phủ PCM** — khoảng trống = **mất tiếng thật** | ✅ **ĐÃ LOẠI** — v8: **`liên tục = 1,00`** trên cả ba host (trước tốt nhất 0,40) | — |
| M12 | **`cắt biên = 0` ở xvideos** — vì sao fMP4 không tìm thấy biên | ✅ **ĐÃ TRẢ LỜI VÀ ĐÃ SỬA** — hai nguyên nhân cộng dồn: van đói byte (§5.11a) **và** mốc `moof` sai 4 byte (§5.11b). v8: `cắt biên` = **11** (vlstream) và **2** (xvideos) `[ĐO]` | — |
| M13 | **`min` lead mỏng** | ✅ **ĐÃ RỘNG HƠN HẲN** — v8: `min` = **14,79 / 19,68 / 7,29 s** ⇒ biên **2,1×–5,6×** (trước 1,3×). ⚠️ **YouTube vẫn mỏng nhất** (2,1×) vì phiên dài nhất | Theo dõi `min` trên host thứ tư và ở **2×** |

---

## 7. VIỆC TIẾP THEO

> ✅ **Điều kiện tiên quyết của M2.2 nay ĐÃ ĐỦ:** ba host, hai codec, bốn mặt xanh đồng thời (§1.11).
> Thứ tự dưới đây **cố ý** đặt các phép đo chưa làm lên **trước** việc tích hợp — vì cả ba đều có
> thể làm đổi **tham số thiết kế**, và sửa tham số **sau** khi đã tích hợp thì đắt hơn nhiều.

| # | Việc | Đạt được gì | Chi phí |
| :-: | :--- | :--- | :--- |
| 1 | **Đo host thứ TƯ** — một trang khác, lý tưởng là nơi có livestream hoặc segment dài | Kiểm **tính tổng quát**: tới giờ mới có 3 host, 2 định dạng. Chưa biết host lạ có phá giả định nào không (ví dụ `styp` đứng đầu mỗi segment, hoặc audio trong MP4 **không** phân mảnh ⇒ **không có `moof`**) | ~15 phút |
| 2 | **Đo ở chế độ 2×** | Rủi ro M7. Lead tính theo **timeline media** nên **không** chia 2, nhưng ngân sách **wall-clock** thì có: 73 s lead ở 2× ⇒ ~37 s. Cần xác nhận offline vẫn theo kịp vì nó **không** phụ thuộc tốc độ phát | ~15 phút |
| 3 | **Đo sai số mốc thời gian** của PCM offline (giữa `endMedia` và biên phân đoạn thật) | R10 của tài liệu 22: khớp phụ đề/TTS. Sai số dưới 1 s không ảnh hưởng **tính khả thi**, nhưng **có** ảnh hưởng **chất lượng khớp** | ~30 phút |
| 4 | **Đo phiên DÀI** (≥ 10 phút) | Cửa sổ trượt + prune + trần 32 MB **chưa** được thử ở quy mô đó. Mọi phiên tới nay đều ngắn hơn một bộ phim | ~15 phút |
| 5 | **Đặt `offlineTargetLeadSec ≈ 30 s`** rồi đo lại | Van tiết lưu cho backend; hiện để 0 (không chặn) để đo **trần**. Chỉ đặt **sau** khi biết `min` thật của host thứ tư | ~15 phút |
| 6 | **Tích hợp đường `offline` vào `extension_firefox/` (M2.2)** | Đây mới là M2 thật: MAIN world mirror → PCM → `window.postMessage` + transferable → content script → dùng lại đường WS `audio_chunk` sẵn có | ~1 ngày |
| 7 | Giữ **cả hai** đường trong sản phẩm: `offline` (chính) + element/ScriptProcessor (dự phòng) | Q9 đã xác nhận CSP **theo từng trang** (YouTube chặn blob-worklet, xvideos không). Đường `offline` **miễn nhiễm CSP** — lợi ích thiết kế chưa từng được ghi nhận | trong #6 |
| 8 | **Thêm kiểm thử WebM có cấu trúc** vào dry-run | Phần K/L phủ fMP4; **WebM mới chỉ được phủ ở mức quét chữ ký** (§5.11c). Đây là khoảng trống kiểm thử còn lại — và chính loại khoảng trống này đã để lỗi 4 byte sống sót qua năm phiên | ~45 phút |

> **Không** đề xuất tối ưu đường dịch: người quyết định đã chọn model dịch lớn để bảo đảm chất
> lượng, và **không** đánh đổi chất lượng lấy độ trễ (xem tài liệu 23 §6 việc #5).

---

## PHỤ LỤC A — TÁI LẬP

```powershell
node scratch/m2_mirror/m2_dryrun.mjs                 # kỳ vọng 160/160
python -m http.server 8099                           # rồi mở mirror_selftest.html
python scratch/m2_mirror/m2_analyze.py               # sau khi có log trong report/audit/m2_logs/
```

## PHỤ LỤC B — GIỚI HẠN CỦA BÁO CÁO NÀY

* ✅ **Kết luận trung tâm nay có số trên BA host và HAI codec, bốn mặt xanh ĐỒNG THỜI** (§1.11):
  `play.vlstream.net` fMP4/AAC `+20,99 s` (`min 14,79`), `www.xvideos.com` fMP4/AAC `+26,65 s`
  (`min 19,68`), `www.youtube.com` WebM/Opus `+73,18 s` (`min 7,29`); `liên tục = 1,00`,
  **lỗi = 0**, `bỏ = 0` trên cả ba. **Chưa đo ở 2×, chưa đo host thứ tư.**
* ✅ **fMP4 nay ĐÃ cắt biên được** (`cắt biên` = 11 và 2, `giữ init` đủ) — câu hỏi để ngỏ suốt năm
  phiên đã có lời giải (§5.11b). **Nhưng `moof ≤ 1` của các phiên cũ vẫn KHÔNG dùng được làm bằng
  chứng** về việc stream có `moof` hay không: trước v8 gần như không lô nào được dựng tới nơi tới
  chốn. Kết luận "fMP4 có `moof`" chỉ dựa trên **phiên v8** (`chữ ký: moof = 3`).
* 🔴 **Bản sửa mới nhất (v8) CHƯA qua phiên đo thật.** Mọi thứ ở §5.11 là `[VERIFIED-DRYRUN]`
  (160/160) — **nhưng lần này có thêm một bảo chứng mà các bản trước không có:** bài kiểm thử phần J
  đã được **kiểm chứng ngược** (cài lại dòng lỗi cũ ⇒ phần J **thất bại** đúng như trên trang thật,
  `calls=0`). Bản **v6** và **v7** đều từng là `[VERIFIED-DRYRUN]` **và đã sai trên thực tế**.
* 🔴 **BỐN hồi quy liên tiếp do tôi gây ra** (v3 cắt mất init; v4 kẹt 423 nhịp; **v6 bịa init ⇒ 0
  PCM**; **v7 chốt con trỏ sớm ⇒ van đói byte ⇒ 0 PCM trên cả ba host**). Đọc kèm §5.3, §5.6,
  §5.10, §5.11. **Đây là phần quan trọng nhất của báo cáo này**: nó cho thấy đường ghép-byte có
  nhiều trạng thái biên hơn mô hình hoá bằng suy đoán, và **cách duy nhất tiến được là làm cho log
  tự trả lời** — điều đã đúng ngay ở phiên v7 (bộ đếm `tickWhy` chỉ thẳng nguyên nhân mà 4 phiên
  trước đó không thấy).
* **Một phát hiện tích cực, không phụ thuộc các hồi quy trên:** nhận diện init **theo nội dung** đã
  chạy đúng trên **ba host, hai định dạng** (`webm`, `mp4`), kể cả host **chưa từng đo**
  (`play.vlstream.net`) — `offlineNoInit = 0` ở cả ba. Đây là phần **đã được kiểm chứng thật**, không
  phải dry-run.
* ✅ **`liên tục = 1,00` trên cả ba host ở v8** — trước đó tốt nhất chỉ 0,40. **Đã có ba phiên xanh đồng thời cả bốn mặt**
  (`lead dương kể cả min` + `liên tục ≥ 0,97` + `lỗi = 0` + `bỏ = 0`). Phiên v5 có **lead tốt nhất**
  nhưng `liên tục` 0,30–0,40; phiên `14:38:30` có lỗi/liên tục tốt nhất nhưng lead âm vì kẹt. Trước
  v8, đường offline đang **không ra PCM** trên 2/3 host ⇒ mọi kết luận về lead đều đến từ **v5**,
  **không** phải từ trạng thái hiện tại của mã. **Đây là ô trống lớn nhất của M2.**
* ✅ **Biên an toàn nay rộng:** `min` = 14,79 / 19,68 / 7,29 s ⇒ **2,1×–5,6×** ngưỡng. ⚠️ YouTube vẫn mỏng nhất (2,1×) và là phiên dài nhất ⇒ **host dài hơn có thể còn mỏng hơn**.
* ✅ **xvideos `cắt biên = 0` qua cả năm phiên — ĐÃ GIẢI QUYẾT.** Nguyên nhân nay
  đã tìm ra và đã sửa** (§5.11b: mốc cắt fMP4 sai 4 byte, cộng với van đói byte ở §5.11a) — nhưng
  **chưa có phiên đo nào xác nhận**. Con số `moof ≤ 1` của các phiên cũ **không dùng được làm bằng
  chứng** về việc stream có `moof` hay không: trước v8 gần như không lô nào được dựng tới nơi tới
  chốn, nên bộ quét chỉ thấy một mảnh mỗi lần.
* ⚠️ **Khoảng trống kiểm thử còn lại nay nằm ở WebM, không phải fMP4** (đảo chiều so với trước):
  fMP4 đã có **cả** kiểm thử offline có cấu trúc (phần K/L) **và** phiên thật xanh; WebM có phiên
  thật xanh **nhưng** chỉ được phủ ở mức **quét chữ ký thô** — EBML Cluster ID **là** byte đầu phần
  tử nên không có trường `size` để kiểm cấu trúc như ISO-BMFF. Đây là việc #7 ở §7, và đây chính là
  loại khoảng trống **đã để lỗi 4 byte sống sót qua năm phiên**.
* ✅ **Đã có ba phiên xanh ĐỒNG THỜI cả bốn mặt** (`lead` dương kể cả `min`, `liên tục ≥ 0,97`,
  `lỗi = 0`, `bỏ = 0`) — §1.11. **Ô trống lớn nhất của M2 nay đã được lấp.**
* ⚠️ **Hành vi DÀI HƠI chưa đo.** Phiên v8 dài nhất mới **230 s PCM**. Cửa sổ trượt + prune + trần
  32 MB **chưa** được thử ở quy mô hàng chục phút. Đây là rủi ro thật cho sản phẩm, không phải chi
  tiết nhỏ: mọi phiên đo tới nay đều ngắn hơn một bộ phim.
* ⚠️ **Quan sát chưa lý giải:** phiên `14:38:30` element cho **0 chunk PCM** dù `playing: true` và
  `capture: scriptprocessor` (các phiên khác cùng host đều có). **Không phiên v8 nào tái hiện**
  (element cho 0,17–0,25 s ở cả ba). Vẫn để ngỏ, **không** suy diễn.
* `[VERIFIED-DRYRUN]` chỉ nghĩa là logic đúng trong môi trường giả — **không** nói gì về hành vi
  của codec thật, CSP thật, hay autoplay thật.
* Chưa đo chi phí **CPU/RAM** trong trình duyệt thật, và chưa đo jank của ScriptProcessor (đường
  element chỉ còn là dự phòng, nhưng nó chạy trên main thread).
* Chưa đo **độ chính xác mốc thời gian** của PCM offline (sai số giữa `endMedia` và biên phân đoạn
  thật). Với ngân sách 24–43 s thì sai số dưới một giây không ảnh hưởng tính khả thi, nhưng **có**
  ảnh hưởng tới việc khớp phụ đề/TTS (R10 của tài liệu 22) — cần đo riêng ở M2.2.
* **Rủi ro dương tính giả của chữ ký 4 byte chưa định lượng.** Nếu chữ ký `1F 43 B6 75` / `moof`
  xuất hiện tình cờ trong payload, cắt đầu có thể **mất tiếng thật** mà **không** gây lỗi giải mã
  (file vẫn hợp lệ) ⇒ fail-safe không bắt được. Đã giới hạn cắt đầu ≤ ⅓ file, nhưng chưa có số.
  **Đây là lý do nên chuyển sang quét theo cấu trúc box** (đọc `size` + `type`) — vừa loại rủi ro
  này, vừa có thể là cách sửa cho xvideos. **Nhưng chỉ nên làm SAU khi van chờ chạy đúng** (§5.11),
  nếu không sẽ lại sửa mù trên một hệ chưa hoạt động.


---
