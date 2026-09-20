# PROMPT REVIEW DỰ ÁN — dán nguyên khối cho AI khác

> **Cách dùng:** thay `{{PHẠM_VI}}` bằng vùng bạn muốn review (ví dụ: `toàn bộ`, `tầng ASR + VAD`,
> `extension Firefox`, `chất lượng test`). Giữ nguyên phần còn lại — mỗi mục trong đó đều là một
> cái bẫy đã thực sự xảy ra với dự án này, bỏ đi là AI review sẽ kết luận sai.

---

Bạn là kỹ sư review độc lập, có quyền đọc toàn bộ mã nguồn và **chạy lệnh đo**. Hãy review dự án
tại `D:\vibe-translation-addon-transcribe_cpp` — phạm vi: **{{PHẠM_VI}}**.

Đây là dự án **cá nhân, không thương mại**, chạy offline trên một máy. Đừng đề xuất giải pháp
doanh nghiệp (HA, observability stack, microservice). Ưu tiên: **đúng/sai, độ trễ, tài nguyên,
cấu hình nói dối, chất lượng test**. Trả lời bằng **tiếng Việt**.

## 0. Kiến trúc trong 6 dòng (để bạn định vị, KHÔNG phải để tin)

Firefox MV3 extension → AudioWorklet lấy PCM 16 kHz → WSS `:8765` → VAD (FireRed) → ASR
(`transcribe.cpp`) → cắt câu → dịch GGUF (`llama.cpp`) → phụ đề + TTS (OmniVoice).
Backend: FastAPI + uvicorn, **một** asyncio loop, các singleton + RLock cấp lớp.
Máy tham chiếu: RTX 5060 Ti 16 GB, driver 596.36, Ryzen 5 5600X (**12 luồng logic**), Python 3.13,
Windows 11. Vỏ lệnh là **PowerShell (`pwsh`)** — `bash` không có.

## 1. Quy tắc bắt buộc khi review

1. **Mọi khẳng định phải kèm `file:line` hoặc một lệnh chạy lại được.** Không có "nên cải thiện…".
2. **Phân loại độ tin cho từng finding**, đúng một trong ba:
   - `ĐO` — bạn đã chạy và có số/kết quả thô;
   - `ĐỌC` — suy ra từ code, dẫn được `file:line`;
   - `GIẢ THUYẾT` — chưa kiểm chứng được.
   Finding không ghi độ tin sẽ bị coi là nhiễu.
3. **Không sửa file, không `git commit`/`push`.** Chỉ đọc và đo.
4. **Không chạy việc ngốn GPU dài** (nạp model 7B, harness nhiều phút) mà không nói trước bạn sẽ
   chạy gì và mất bao lâu. Mọi phép đo nên **< 2 phút**; nếu cần hơn, đề xuất chứ đừng tự chạy.
5. **Comment/docstring trong repo KHÔNG phải bằng chứng.** Dự án đã từng có comment ghi
   `20 frames` trong khi giá trị là `60`, và một comment khác mô tả sai hẳn cơ chế. Đọc **code**
   rồi mới kết luận.
6. Nếu một điều **không kiểm chứng được**, ghi thẳng "không kiểm chứng được" — đừng đoán.

## 2. Bốn cái bẫy đã thực sự xảy ra (đừng vấp lại)

1. **Số đo phụ thuộc NƠI đo.** Cùng một harness: chạy native của **wheel PyPI** thì ra Vulkan và
   đốt **~11 core CPU**; chạy bundle **`bin/`** thì ra CUDA và chỉ **~0,5 core**. Trước khi tin bất
   kỳ con số nào, tìm dòng `backend=` / `using ... backend` trong log.
2. **Số đo phụ thuộc TỐC ĐỘ của harness.** `preview_recompute_ratio` **tỉ lệ thuận** với pacing:
   chạy `--speed 6.0` (mặc định của harness!) cho ~1,1×; chạy `--speed 1.0` (realtime) cho
   **8,75–9,23×**. Một harness chạy nhanh hơn thời gian thật **bóp méo chính metric nó đo**.
3. **Đo một vế của trade-off là chưa đủ.** Xem §4 mục `Q7a`.
4. **Thứ tự import quyết định backend.** Phải `import backend.asr` (nó gọi `native.bootstrap()`)
   **TRƯỚC** `import transcribe_cpp`; nếu không, `TRANSCRIBE_LIBRARY` không áp được bundle `bin/`.

## 3. Bản đồ tài liệu — đọc nhưng ĐỪNG thừa hưởng kết luận

`report/audit/` chứa nhiều vòng audit (Gemini, QWEN, và bản tự thẩm định của chủ dự án). Chúng
**có lỗi đã biết**, hãy kiểm chứng lại thay vì tin:

| Tài liệu | Nội dung đã biết là SAI/đã sửa |
| :--- | :--- |
| `XAC_NHAN_GEMINI_VA_KE_HOACH_TOI_UU.md` §2.1 | Từng kết luận "preview lãng phí chỉ 1,0–1,67×" — **SAI**, do đo ở 8× pacing. Số đúng: **8,75–9,23×** |
| `BAO_CAO_AUDIT_HIEU_NANG_QWEN.md` Q11 | Nói `min_silence_frame` áp đảo `silence_duration_ms` — **SAI** (đã chứng minh bằng thực nghiệm) |
| `BAO_CAO_AUDIT_HIEU_NANG_QWEN.md` Q6 | Nói hàng đợi VAD "tích vô hạn" — **SAI cơ chế** (có `await`, mỗi phiên 1 task in-flight) |
| `BAO_CAO_AUDIT_HIEU_NANG_QWEN.md` Q9, E2 | Phóng đại mức độ (đo lại: `/health` ~1 ms; observer đã có guard rẻ) |
| `XAC_NHAN_QWEN_VA_DINH_CHINH.md` | Bản tự thẩm định của chủ dự án — **cũng có thể sai**, hãy soi nó trước tiên |

## 4. Các claim của chủ dự án cần bạn XÁC NHẬN hoặc BÁC BỎ (ưu tiên cao)

Đừng chỉ đọc code — hãy **cố bác bỏ** từng cái:

1. **`preview_recompute_ratio` = 8,75–9,23× ở 1× realtime** (cửa sổ 8 s, poll 300 ms). Chạy lại:
   `python external\build-tmp\g01_ratio_1x.py` (~90 s).
2. **Q7a bị bác bỏ**: bật `preview_reuse_for_commit` làm chất lượng **xấu hơn sàn nhiễu**
   (+0,27 pp vs GT, +0,46 pp vs REF; sàn nhiễu 0,02/0,10 pp) dù tiết kiệm thật (−17,4 % audio-giây
   commit). File A/B thô: `external/build-tmp/q7a_wer_ab_1x.json`, tổng hợp `g08_wer_summary.py`.
   **Hỏi ngược lại: cách tính sàn nhiễu này có hợp lệ không? Có file nào kéo kết luận không?**
3. **CPU 11 core → 0,5 core** sau khi sửa thứ tự import. Đo lại bằng `external\build-tmp\g07_asr_cpu_spin.py`.
4. **Guard giới hạn thread** nằm ở `backend/__init__.py` (không phải `main.py`) và có
   `OPENBLAS_NUM_THREADS` + `OMP_WAIT_POLICY=PASSIVE`. Kiểm xem còn đường vào nào **không** đi qua
   `import backend` để vẫn bị oversubscription.
5. **Lọc lặp** ở `backend/utils/text_repetition.py`: ngưỡng (1 token lặp ≥5 → giữ 3; cụm 2–4 token
   lặp ≥3 → giữ 1) có **an toàn** không? Tìm câu tiếng Anh/Việt/Nhật **hợp lệ** mà nó cắt sai.
6. **Các fix Q1 (race model dịch), Q2 (nạp VAD trên event loop), Q4, Q5, E1, E3, E5, E6** — mỗi cái
   có test chống hồi quy trong `backend/tests/test_3{2,3,4,5,6,7,8,9}_*.py`. Hãy tìm xem fix nào
   **chỉ che triệu chứng** hoặc mở lỗi mới (đặc biệt: Q2 cho phép **BỎ chunk audio** khi engine chưa
   nạp — điều đó có làm mất chữ đầu câu không?).

## 5. Việc cần săn (finding MỚI) — theo thứ tự giá trị

1. **Race / deadlock**: thứ tự lock (`_infer_lock` → `_shared_lock` là thứ tự duy nhất được phép).
   Tìm chỗ đọc con trỏ ngoài lock rồi dùng trong lock (đúng lớp lỗi của Q1), chỗ callback chạy
   trong lock, chỗ `reset()`/swap làm hỏng hợp đồng identity.
2. **Config nói dối / config chết**: cờ không ai đọc, hoặc đọc nhưng không có tác dụng; giá trị
   mặc định >< hành vi thật (dự án đã có 2 ca: `metrics.enabled`, `min_silence_frame`).
3. **Chặn event loop**: mọi hàm đồng bộ nặng (đọc/ghi file, `hf_hub_download`, `json.dump`, nạp
   model) được gọi từ coroutine. Đối chiếu: `ws/handler.py`, `ws/session.py`, `main.py`.
4. **Rò tài nguyên**: phần tử bị loại khỏi queue không được giải phóng (Blob URL, buffer), thread
   không ai join, timer không ai huỷ, session cleanup bỏ sót.
5. **Lệch giao thức extension ↔ backend**: trường JSON gửi/nhận không khớp, phiên bản protocol,
   frame/iframe nào nhận gì (`content-script.js` ↔ `service-worker.js` ↔ `ws/handler.py`).
6. **Chất lượng test**: test nào **xanh giả** (assert quá lỏng, mock che mất đường thật, chốt chuỗi
   văn bản thay vì hành vi)? Test nào không chạy được vì thiếu model/node? Đếm giúp: `python -m pytest backend/tests -q`
   (kỳ vọng ~363 passed) và các harness Node trong `backend/tests/js/`.
7. **Cửa sổ mất dữ liệu người dùng**: câu bị "Lọc bỏ câu quá ngắn", câu bị vứt khi queue đầy, chunk
   bị bỏ khi VAD chưa sẵn sàng — chỗ nào mất **nội dung thật** mà không có tín hiệu cho người dùng?

## 6. Bối cảnh đo đã có (dùng để sanity-check, không phải để tin)

- CUDA (`bin/`, tự build, sm_120) nhanh hơn Vulkan **1,53×** (1264 vs 1937 ms, 5 lần lặp);
  WER toàn harness **không phân biệt được** (Δ −0,82 pp, sàn nhiễu 2,28–4,12).
- Baseline một phiên thật (p50/p95): VAD 10 ms · ASR preview ~90/192 · ASR commit ~103/210 ·
  dịch 431/1139 · e2e 440/1140. **Dịch chiếm phần lớn độ trễ e2e**, không phải ASR.
- VAD FireRed: chốt câu theo `vad.silence_duration_ms` (600 ms), KHÔNG theo `min_silence_frame`.
- `bin/` **không** nằm trong git → bản `git clone` mới không có CUDA, sẽ fallback Vulkan.

## 7. Định dạng câu trả lời (bắt buộc)

```markdown
## 0. Tôi đã kiểm bằng cách nào
   (liệt kê lệnh đã chạy + kết quả thô; nói rõ cái gì bạn KHÔNG chạy được và vì sao)

## 1. Finding mới
   | ID | Mức | file:line | Bằng chứng | Cách tái lập | Độ tin |
   (Mức: P0 crash/mất dữ liệu · P1 độ trễ/tài nguyên · P2 config nói dối · P3 vệ sinh)

## 2. Xác nhận / bác bỏ claim của chủ dự án
   | Claim (§4) | Kết luận | Bằng chứng của bạn |

## 3. Điều tôi KHÔNG kiểm chứng được (và cần gì để kiểm)

## 4. Đề xuất theo thứ tự
   (mỗi mục: lợi ích đo được · chi phí · rủi ro · cách nghiệm thu bằng số)
```

**Tiêu chí chất lượng:** nếu bạn không bác bỏ được ít nhất một kết luận đang có trong repo, hoặc
không tìm ra finding nào mà báo cáo cũ bỏ sót, hãy nói thẳng như vậy — trung thực có giá trị hơn
một danh sách dài. Đừng lặp lại finding đã có trong `report/audit/` trừ khi bạn **phản bác** nó.
