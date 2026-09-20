# BÁO CÁO AUDIT HIỆU NĂNG — GLM (độc lập, đo lại từ đầu)

- **Phạm vi**: `D:\vibe-translation-addon-transcribe_cpp` — backend FastAPI + extension Firefox, mục đích cá nhân/offline/1 máy.
- **Ngày**: 2026-09-18. Máy tham chiếu: RTX 5060 Ti 16 GB, driver 596.36, Ryzen 5 5600X (6C/12T), Python 3.13.14, Windows 11, shell `pwsh`.
- **Nguyên tắc thực thi**: không sửa code, không commit/push; mọi claim đều kèm `file:line` hoặc lệnh tái lập; mỗi finding gắn nhãn **ĐO** (đo được trong session này) / **ĐỌC** (suy từ code, chưa có repro runtime) / **GIẢ THUYẾT** (chưa kiểm chứng). GPU không chạy quá 2 phút mỗi lệnh.
- **Artifact môi trường auditor (KHÔNG phải lỗi dự án)**: sandbox chặn `C:\...\Temp\pytest-of-*` nên phải redirect `$env:TMP` về `scratch\tmpglm` mới chạy được pytest. Trên máy owner bình thường không cần bước này.
- **Quy tắc "comment không phải bằng chứng"** được áp dụng xuyên suốt: mọi ý dưới đây chỉ dựa trên code thực thi hoặc số đo.

---

## 0. Tôi đã kiểm bằng cách nào (lệnh + số thô)

### 0.1 Test suite backend
```
$env:TMP='D:\vibe-translation-addon-transcribe_cpp\scratch\tmpglm'
python -m pytest backend/tests -q --no-header        # ~2 phút
```
→ **363 passed, 363/363** (không fail, không error sau khi TMP redirect). 22 ERROR `WinError 5` trước đó là do sandbox, không phải code.

### 0.2 Test JS của extension (Node, không cần browser)
```
Get-ChildItem backend\tests\js\*.js | ForEach-Object { node $_.FullName; $LASTEXITCODE }
```
→ **10/10 file thoát mã 0** (worklet harness, resample alias, subtitle policy/renderer, overlay attach/scale, tts ducking, sw broadcast target, ws reconnect).

### 0.3 Ratio preview-recompute tại SPEED=1.0 (claim 1)
`python external\build-tmp\g01_ratio_1x.py` (backend CUDA, pacing 1.0×):
- `Japanese_5s.wav`: **ratio 9.22**
- `Chinese_noise_28s.wav`: **ratio 8.74**
- Quan sát thêm trong log: commit `MAX_DURATION` và `Trim chồng lấn` đều kích hoạt thật (không chỉ trong test).

### 0.4 CPU của ASR theo 2 chế độ backend + có/không guard thread (claim 3 + finding GLM-2)
`python external\build-tmp\g07_asr_cpu_spin.py cuda`:
- prewarm 1871 ms ≈ **0.98 core**; 4× `_run_inference_sync` ≈165 ms ≈ **1.04 core**; 3 cửa sổ rảnh: **0.00 core**.

Đo bổ sung với guard **bị bỏ** (chạy thẳng `python -c`, xóa env trước import — lệnh nguyên văn trong file report này, máy owner tái lập được):
- **CUDA (`bin/transcribe.dll`), không guard**: idle 3 s = **0.56 core** (rò CPU khi không làm gì).
- **Wheel PyPI (`transcribe_cpp_native`, Vulkan), không guard**: 4× inference = 0.89 core, idle 3 s = **0.00 core**; thư viện nạp: `site-packages\transcribe_cpp_native\_native\transcribe.dll`.
- **FireRed VAD, không guard**, bắn 600 frame liên tục: 1.77 s wall, **1.13 core** (CPU>GPU do vòng đồng bộ).

### 0.5 Vị trí guard thread env (claim 4)
```
python -c "import backend, os; print(os.environ['OMP_NUM_THREADS'], os.environ['OMP_WAIT_POLICY'])"
python -c "import transcribe_cpp, os; print(os.environ.get('OMP_NUM_THREADS'))"   # import ngược thứ tự
```
- `import backend` trước: cả 5 biến `*_NUM_THREADS=2` + `OMP_WAIT_POLICY=PASSIVE` + `KMP_BLOCKTIME=0` đều đặt (backend/__init__.py:15–41).
- `import transcribe_cpp` trước: env **không** đặt VÀ native nạp từ wheel (Vulkan) — bypass tồn tại về mặt ngôn ngữ, nhưng **không đường vào nào trong repo** chạm phải (main.py, conftest.py, toàn bộ harness trong `external/build-tmp/` đều bootstrap trước — grep `^import transcribe_cpp` chỉ thấy 1/13 file có bootstrap trước là `g03_health_cost.py`, các file còn lại là script đo độc lập ngoài pipeline).

### 0.6 WER A/B Q7a — tính lại từ dữ liệu có sẵn (claim 2, finding GLM-1)
Dùng `external/build-tmp/q7a_wer_ab_1x.json` (model qwen3-asr-0.6b, speed 1.0, 3 repeats, 8 file → n=24/config), tính paired per-file (script trong session):
- `error_rate`: mean delta reuse−base = **+0.266 pp**, sd 0.766, n=8, **95%CI [−0.37, +0.91]** (bao 0).
- `stream_vs_reference_error_rate`: mean delta = **+0.461 pp**, sd 0.812, **95%CI [−0.22, +1.14]** (bao 0).
- Sai số run-to-run thô của từng config là rất lớn giữa các file (sd_base 32.3 pp cho error_rate — do độ khó file khác nhau), nên "floor 0.02–0.10 pp" tính theo kiểu unpaired-difference **không phải** thước đo nhiễu có ý nghĩa thống kê (chi tiết ở GLM-1).
- Chi phí `/health`: `python external\build-tmp\g03_health_cost.py` → **0.995 ms/call** (bác thêm lần nữa QWEN-Q9 "health đắt").

### 0.7 Bộ lọc lặp từ (claim 5) — gọi trực tiếp
```
python -c "from backend.utils.text_repetition import collapse_repetitions; ..."
```
| Đầu vào | Đầu ra | Đánh giá |
|---|---|---|
| `Cô ấy nói cô ấy nói cô ấy nói chuyện` | `Cô ấy nói chuyện` | **cắt nhấn mạnh hợp lệ** |
| `anh xin lỗi anh xin lỗi anh xin lỗi` | `anh xin lỗi` | **cắt nhấn mạnh hợp lệ** |
| `ha`×11 | `ha ha ha` | đúng design (KEEP=3) |
| `we will we will rock you` | giữ nguyên | 2 lần lặp < PHRASE_MIN_REPS=3 |
| `blah blah blah` | giữ nguyên | như trên |

Bộ lọc chạy ở CẢ hai đường: ASR (`backend/utils/text_cleaner.py:28`) và dịch (`backend/translation/engine.py:394,479`).

### 0.8 Config "min_silence_frame" (dặm chân kết luận cũ)
`python external\build-tmp\g02_min_silence_frame.py`: FireRed commit đi theo `silence_duration_ms` (~1675 ms tới commit) trong khi engine kết thúc speech ở ~2550 ms — `min_silence_frame=60` (backend/config.py:79) **chết cho đường FireRed**, đúng như ghi chú trong config. (QWEN-Q11 "min_silence_frame chi phối" bị bác bỏ lần nữa.)

### 0.9 Trạng thái git của `bin/` (giả định trong brief)
```
git check-ignore -v bin/transcribe.dll   → exit 1 (KHÔNG bị ignore)
git ls-files bin/                        → 6 file, có bin/transcribe.dll, bin/ggml-cuda.dll
```
→ **`bin/` đang được track đầy đủ**; `.gitignore` chỉ có dòng `# bin/` đã bị chú thích. Giả định "clone mới rớt về Vulkan" **SAI** ở trạng thái repo hiện tại.

### 0.10 Đọc code chéo các vùng rủi ro cao
Toàn bộ các đường dưới đây được đọc từng dòng trong session (không chạy test GPU thêm):
- `backend/asr/engine.py` — lock order `_infer_lock`→`_shared_lock` (119–142, 692–700), `prepare_model` swap nguyên tử (404–487), tier234 (768–804), `cancel_inference` (1490–1515), `feed_audio`/`on_speech_start`/`on_speech_end` (503–601), pre-roll P1.3 (540–547), `_enqueue_commit_locked` gộp-không-vứt (612–622).
- `backend/ws/handler.py` — `_coalesce_enqueue` đầy đủ (192–271), vòng `handle_ws` + `finally` (331–397), `_handle_text_message` (399–452), `_process_binary_chunk` qua executor (455–482), `_reset_session_stream` chống coalesced seek (563–603), 2 worker queue (757–776, 867–889).
- `backend/ws/session.py` — `SessionConfigPayload` (37–71), `apply_config` (333–452), 2 hàm `_schedule_*_model_switch` (230–330).
- `backend/vad/processor.py` — `_ensure_engine` non-blocking (145–169), `prewarm` (171–185), `_spawn_engine_load` (217–250).
- `backend/translation/engine.py` — Q1 `_translate_sync` snapshot-under-lock (345–400), `translate_stream` sentinel (504–532), `hotswap.py` `run_reserved` (136–185).
- `backend/main.py` — lifespan prewarm + watchdog + cleanup (312–391).
- Extension: `popup.js` `getSettings` (77–124), `tts-player.js` binary/Blob paths (280–477).

---

## 1. Finding mới

| ID | Mức | Vị trí | Bằng chứng | Cách tái lập | Độ tin |
|----|-----|--------|-----------|--------------|--------|
| **GLM-1** | **P1 (phương pháp đo)** | `external/build-tmp/g08_wer_summary.py:69-71` (floor = \|mean(base)−mean(base_repeat)\|) | Floor Q7a là **một hiệu single-run** (khác biệt 2 mean liền kề), không phải std/CI; khi tính paired per-file từ chính `q7a_wer_ab_1x.json`: delta +0.27 pp, **95%CI [−0.37,+0.91]** và vs REF +0.46 pp **[−0.22,+1.14]** — cả hai **bao 0**. Sai số unpaired của pipeline WER là 9.3 pp (error_rate) / 5.0 pp (stream_vs_ref) — lớn gấp hàng chục lần "floor" đang công bố. | Tính lại: script python đọc `q7a_wer_ab_1x.json`, mean per-file delta + t(0.975, df=7)=2.365 (nguyên văn trong log session) | **ĐO** |
| **GLM-2** | **P2** | `backend/__init__.py:15-41` (guard) — nguy cơ tại mọi entry bỏ qua bootstrap | Bỏ guard: **CUDA idle rò 0.56 core** vô hạn (OMP_WAIT_POLICY ACTIVE mặc định); FireRed VAD tốn 1.13 core lúc bắn frame. Có guard: 0.00 core. Bypass thật sự tồn tại (`import transcribe_cpp` trực tiếp → wheel Vulkan + không guard) nhưng không đường vào nào trong repo chạm phải. | Xóa 5 env + `OMP_WAIT_POLICY` rồi `python -c` đo `psutil` như §0.4 | **ĐO** |
| **GLM-3** | **P2** | `backend/tests/test_04_commit_logic.py:172-175` (tương tự test_01/02/03/05/06/07) | Chạy pytest **ghi đè file report tracked** trong git: `report/04_commit_logic/report.md` chuyển thành `M` (timestamp + throughput đổi số). Làm bẩn working tree sau mỗi lần chạy test, dễ commit nhầm số liệu đo khác máy. (Tôi đã `git checkout --` phục hồi về HEAD.) | `git status` sạch → `python -m pytest backend/tests/test_04_commit_logic.py -q` → `git status` hiện `M report/04_commit_logic/report.md` | **ĐO** |
| **GLM-4** | **P2** | `backend/vad/processor.py:323-337` | Chunk audio bị **vứt im lặng** khi engine VAD chưa sẵn sàng: chỉ tăng counter `vad.chunks_dropped_engine_not_ready` + 1 log warning, **không có tín hiệu nào tới người dùng qua WS**. Cửa sổ mất audio thật: từ lúc client connect tới khi `_spawn_engine_load` xong (hàng giây nếu prewarm lỗi/chậm, hoặc khi đổi engine giữa phiên). Prewarm mặc định (main.py:319–322) che đa số trường hợp. | ĐỌC code path: `_ensure_engine` trả False → `feed_chunk` drop. Chưa dựng repro WS end-to-end | **ĐỌC** |
| **GLM-5** | **P3** | `extension_firefox/content/content-script.js:34`, `popup/popup.js:116` ↔ `backend/ws/session.py:37-71` | `ttsInstruct` luôn được gửi mỗi `set_config` nhưng `SessionConfigPayload` không có field nào và `extra="ignore"` → **config chết trên đường WS**. Hiện vô hại (popup gửi cứng `""`), nhưng nếu mai thêm ô nhập instruct sẽ "UI có, backend lơ". | Gửi `{"type":"set_config","ttsInstruct":"nói giọng Bắc"}` vào WS → log `apply_config` không có field này | **ĐỌC** |
| **GLM-6** | **P3** | `extension_firefox/popup/popup.html` (`rangeVadSilence value=450`) ↔ `backend/config.py` (`silence_duration_ms=600`) | Default popup ≠ default backend: lần đầu mở popup trước khi set_config gửi lên, backend chạy 450 ms thay vì 600 ms. Cosmetic, không mất dữ liệu. | Mở popup, không đụng slider → `set_config` log `silence=450ms` | **ĐỌC** |
| **GLM-7** | **P3** | `backend/vad/processor.py:364-370` | `raw_buffer` trần 3 s: tràn thì **drop phần cũ im lặng** (không counter/log riêng như GLM-4). Chỉ xảy ra khi VAD worker chậm hơn client ≥3 s audio — hiếm trên 1 máy, nhưng khi xảy ra là mất chữ không dấu vết. | ĐỌC. Repro tiềm năng: bắn audio nhanh hơn tốc độ xử lý VAD 3 s | **ĐỌC** |

**Những gì tôi chủ động đi tìm và KHÔNG tìm thấy** (nói thẳng để tránh "audit âm thầm"):
- Không tìm thấy deadlock/race mới trong bộ lock `_infer_lock`→`_shared_lock`: mọi điểm đọc lại đều đúng thứ tự (asr/engine.py:119–142, 692–700; translation/engine.py:157–190, 374–388).
- Không thấy rò rỉ phiên: `handle_ws` `finally` unregister → hủy + `gather` worker → `_discard_queued` (đã cân `task_done()`) → `session.cleanup()` (handler.py:355–380); worker queue dùng `finally: task_done()` đúng cả nhánh `continue` (handler.py:771, 883–884).
- Không thấy sync file I/O chặn event loop còn sót lại trên hot path: metrics dump lúc ngắt phiên đã qua `asyncio.to_thread` (handler.py:391–393).
- `_coalesce_enqueue` O(n) drain-refill (handler.py:218–271) chỉ chạy khi queue đầy, n≤32 — không đáng kể, nhận xét "an toàn event-loop đơn luồng" trong code là chính xác.

---

## 2. Xác nhận / bác bỏ claim của chủ dự án

| # | Claim | Kết luận | Bằng chứng |
|---|-------|----------|-----------|
| 1 | Ratio 8.75–9.23× tại 1× realtime | **XÁC NHẬN — ĐO** | g01: 9.22 (Japanese_5s), 8.74 (Chinese_noise_28s), backend CUDA (§0.3) |
| 2 | Q7a: reuse không làm mất chất lượng, delta ≫ noise floor | **XÁC NHẬN MỘT PHẦN** — số liệu đúng (+0.27 pp vs GT, +0.46 pp vs REF), kết luận hướng vẫn đứng, **nhưng "floor 0.02/0.10 pp" không đạt chuẩn thống kê** (GLM-1): paired 95%CI bao 0. Đúng hơn phải nói: "không phát hiện degradation với n=8×3; sai số đo là ±9.3 pp unpaired / CI ±0.6 pp paired" | §0.6 |
| 3 | CPU 11 → 0.5 core | **XÁC NHẬN — ĐO**, thậm chí tốt hơn: idle thực = **0.00 core** có guard (g07 cuda; 3 cửa sổ rảnh). "0.5 core" của owner có thể là prewarm/inference thoáng qua (0.98–1.04 core) | §0.4 |
| 4 | Thread guard nằm ở import `backend`, chạy trước mọi import native | **XÁC NHẬN — ĐO + ĐỌC**; bypass tồn tại về nguyên tắc nhưng không đường vào nào trong repo; hệ quả bypass đo được: wheel Vulkan + mất guard | §0.5 |
| 5 | Bộ lọc lặp an toàn, không cắt nội dung hợp lệ | **BÁC BỞ một phần — ĐO**: các câu lặp để **nhấn mạnh** bị cắt (`Cô ấy nói cô ấy nói cô ấy nói chuyện` → `Cô ấy nói chuyện`; `anh xin lỗi`×3 → ×1). Về "không mất cả câu" thì đúng (chỉ truncate phần lặp). Là trade-off chấp nhận được cho phụ đề, nhưng **đang cắt thật trong hội thoại thật** khi người nói nhắc cụm ≥3 lần | §0.7 |
| 6 | Các fix Q1/Q2/Q4/Q5/E1/E3/E5/E6 có test thật, không phải test "ô" | **XÁC NHẬN — ĐỌC + ĐO**: 363/363 pass; test_32 (Q1) và test_33 (Q2) là behavioral + AST chứ không khớp chuỗi. Q4/Q5/E1/E3/E5/E6: tôi xác nhận ở mức "suite pass + đọc đúng code path tương ứng", **không tái lập từng fix riêng lẻ** — xem mục 3 | §0.1, đọc test_32/33 |

**Bác bỏ thêm một giả định trong brief (không phải claim số)**: "bin/ không nằm trong git → clone mới rớt về Vulkan" — **SAI** hiện tại: `git ls-files bin/` trả 6 file gồm `bin/transcribe.dll` + `bin/ggml-cuda.dll` (§0.9).

---

## 3. Điều tôi KHÔNG kiểm chứng được (nói rõ, không giấu)

1. **E2E trong trình duyệt thật** (Firefox + extension + video thật): mọi con số trong report này là harness Python/Node. Độ trễ popup→phụ đề, hành vi backpressure service-worker (SOFT 128 KB / HARD 512 KB) và seek-reset trong điều kiện mạng thật **không đo được** trong khuôn khổ này.
2. **WER A/B mới**: tôi chỉ tính lại từ `q7a_wer_ab_1x.json` có sẵn (n=8 file × 3 repeats — mẫu nhỏ). Không chạy lại A/B để tạo dữ liệu mới (tránh GPU-heavy).
3. **GLM-4/GLM-7 (mất audio VAD)**: code path rõ ràng nhưng tôi chưa dựng repro WS end-to-end, vì vậy nhãn ĐỌC thay vì ĐO.
4. **Độ bền dài hạn** (VRAM fragmentation, rò rỉ sau nhiều giờ swap model liên tục): không chạy quá 2 phút GPU mỗi lệnh theo quy tắc — không kiểm chứng được.
5. **Q4/Q5/E1/E3/E5/E6 từng fix riêng lẻ**: chỉ xác nhận qua suite 363 test + đọc code, không có repro độc lập cho từng mục như đã làm cho Q1/Q2.

---

## 4. Đề xuất (lợi ích / chi phí / rủi ro / chỉ tiêu nghiệm thu)

| # | Đề xuất | Lợi ích | Chi phí | Rủi ro | Chỉ tiêu nghiệm thu |
|---|---------|---------|---------|--------|---------------------|
| 1 | **Tính lại Q7a theo paired CI** (thay "floor = \|2 mean liền kề\|"): giữ nguyên dữ liệu, chỉ thêm mean per-file delta + CI t(0.975, df=7) vào `g08_wer_summary.py` | Bằng chứng đạt chuẩn thống kê; claim "an toàn" không còn chỗ bị cào | ~30 phút, không cần chạy GPU mới (dùng JSON cũ, hoặc chạy lại nếu muốn n lớn hơn) | Thấp — chỉ đổi cách trình bày số | Báo cáo WER hiển thị CI; không còn con số "floor 0.02 pp" dạng single-difference |
| 2 | **GLM-4: báo user khi đang drop chunk VAD** — gửi 1 tin `{"type":"model_status","stage":"vad","state":"loading"}` (dùng lại kênh sẵn có) và/hoặc expose `vad.chunks_dropped_engine_not_ready` qua `/health` | Người dùng biết vì sao phụ đề trễ lúc mở tab/chuyển engine thay vì "mất chữ bí ẩn" | Nhỏ (~20 dòng) | Thấp — tin một chiều, không chặn hot path | Mở tab ngay sau start server (prewarm chưa xong) → client nhận status; counter tăng tương ứng số frame drop |
| 3 | **GLM-7: thêm counter `vad.raw_buffer_overflow_drops`** cho đường tràn 3 s | Mất-chữ-không-dấu-vết trở thành có số đo | ~5 dòng | Không | Metric xuất hiện khi đẩy audio nhanh hơn 3 s; 0 khi bình thường |
| 4 | **GLM-3: test không ghi vào report/ tracked** — chỉ ghi report benchmark khi env `MEASURE=1` (hoặc ghi vào `.pytest-tmp/` bị ignore) | `pytest` sạch → `git status` sạch; không commit nhầm số đo khác máy | ~10 dòng ở 6 file test | Không | Chạy full suite → `git status` chỉ hiện file thay đổi thật |
| 5 | **GLM-5: hoặc xóa `ttsInstruct` khỏi payload, hoặc thêm field backend** | Hết config chết; tránh bẫy cho chính owner sau này | 5 phút | Không | Gửi `ttsInstruct` → hoặc bị phản ánh trong config, hoặc không còn trong payload |
| 6 | **GLM-6: đồng bộ default 450↔600** popup vs backend | Tránh "default nào đúng?" | 1 dòng (sửa `value` của `rangeVadSilence` hoặc đổi backend) | Không | Slider lần đầu mở = giá trị backend thực chạy |
| 7 | **Giữ nguyên**: guard thread + thứ tự bootstrap `import backend` trước (đã đo tốt: idle 0.00 core); `_coalesce_enqueue` gộp-không-vứt; heartbeat + stall_watchdog; lock order hiện tại | — | — | **Cảnh báo**: mọi entry-point mới (script, notebook, service khác) phải `import backend` TRƯỚC khi chạm native, nếu không rò 0.56 core idle trên CUDA (GLM-2) | — |

---

*Hết báo cáo. Mọi lệnh tái lập ghi trong §0 đều chạy được trực tiếp bằng `pwsh` trên máy tham chiếu, trừ nơi đã ghi rõ là artifact sandbox.*
