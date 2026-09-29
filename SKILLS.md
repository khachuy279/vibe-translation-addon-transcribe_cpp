# Danh Mục & Hướng Dẫn Sử Dụng Skills (Antigravity Skills)

Tài liệu này tổng hợp các **Skills** (năng lực / runbooks chuyên biệt) được tích hợp trong dự án **Vibe Translation Addon**, được tổ chức theo tiêu chuẩn của Antigravity Agentic Customization System.

---

## 1. Cơ Chế Khám Phá & Kích Hoạt Skill Của Antigravity

Trong Antigravity IDE / Agent:
- Thư mục gốc chứa skills của dự án nằm tại: **`.agents/skills/`**.
- Mỗi skill nằm trong một thư mục con riêng kèm theo file **`SKILL.md`** có chứa YAML frontmatter (`name` và `description`).
- **Cơ chế Progressive Disclosure**: Agent chỉ nạp tên và mô tả của skill vào context. Khi có tác vụ tương ứng (ví dụ: test ASR, kiểm tra CUDA, vận hành server), Agent sẽ tự động nạp toàn bộ nội dung của skill để thực thi chính xác từng bước.

---

## 2. Danh Sách Skills Hiện Có Trong Dự Án

### 🛠️ `vibe-dev`
- **Đường dẫn**: [`.agents/skills/vibe-dev/SKILL.md`](./.agents/skills/vibe-dev/SKILL.md)
- **Mô tả**: Hướng dẫn quy trình vận hành, chẩn đoán lỗi, quản lý mô hình và kiểm thử tự động cho hệ sinh thái Vibe Translation.
- **Các kịch bản sử dụng chính**:
  1. **Chẩn đoán môi trường**: Kiểm tra PyTorch CUDA, GPU offload của `llama.cpp` và bundle DLL ASR (`transcribe.dll`, `ggml-cuda.dll`).
  2. **Vận hành backend**: Khởi chạy FastAPI server, kiểm tra healthcheck và WebSocket stream.
  3. **Chạy kiểm thử Pytest**:
     - Fast unit tests (không nạp model nặng): `pytest`
     - Kiểm tra VAD contract & tính toàn vẹn tín hiệu: `pytest backend/tests/test_41_vad_engine_contract.py`
     - Kiểm tra reset buffer khi tua video: `pytest backend/tests/test_21_seek_reset.py`
     - Kiểm tra benchmarks GPU và tầng B: `pytest -m slow`
  4. **Quản lý mô hình**: Tải và kiểm tra các mô hình ASR (Qwen3-ASR), Dịch thuật (Hy-MT2 GGUF), VAD (FireRed, Silero, FSMN).
  5. **Nạp & debug extension**: Hướng dẫn tải unpacked trên Firefox và Chrome/Edge.

---

## 3. Cách Thêm Skill Mới Cho Dự Án

Để tạo thêm một Skill mới cho Agent:
1. Tạo thư mục mới trong `.agents/skills/<ten-skill>/`.
2. Tạo file `SKILL.md` bên trong với cú pháp:
   ```markdown
   ---
   name: ten-skill
   description: >-
     Mô tả rõ ràng skill này làm gì và khi nào Agent nên kích hoạt nó.
   ---

   # Tên Skill

   ## Hướng dẫn từng bước...
   ```
3. (Tùy chọn) Thêm các thư mục bổ trợ như `scripts/`, `references/` nếu có kịch bản chạy lệnh phức tạp.
