"""Script tải file audio mẫu theo tên file (audio_filename) từ Hugging Face Dataset."""

import os
import sys
import json
import io
import soundfile as sf
from datasets import load_dataset

# Đảm bảo console Windows in được tiếng Việt & emoji
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Dataset ID trên Hugging Face
DATASET_NAME = "dolly-vn/dolly-audio-1000h-vietnamese"

# 🎯 Nhập tên file bạn muốn tải từ Dataset Viewer tại đây (hoặc truyền qua dòng lệnh)
# Ví dụ: "520aa164-6a34-4611-8c6d-c401a3a43b02.wav" hoặc "607e02ea-dc2c-411e-a313-100c805191c6.wav"
DEFAULT_TARGET_FILENAME = "3d8cbf2e-352e-45bc-811b-31ec2dced7fb.wav"

# Tên file muốn lưu trong thư mục voices (nếu để None sẽ tự đặt theo tên hoặc voice_id)
# Ví dụ: "nu_calm_woman.wav" hoặc giữ nguyên DEFAULT_TARGET_FILENAME
CUSTOM_SAVE_NAME = None

target_file = sys.argv[1].strip() if len(sys.argv) > 1 else DEFAULT_TARGET_FILENAME
# Chuẩn hóa tên file tìm kiếm (loại bỏ khoảng trắng)
target_file_clean = os.path.basename(target_file)

current_dir = os.path.dirname(os.path.abspath(__file__))
voices_json_path = os.path.join(current_dir, "voices.json")

print(f"🔄 Đang kết nối tới dataset '{DATASET_NAME}'...")
print(f"🔍 Đang tìm kiếm file: '{target_file_clean}'...")

try:
    # streaming=True tải từng sample qua mạng cực nhanh
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    
    found_sample = None
    count = 0
    
    for sample in dataset:
        count += 1
        fn = sample.get("audio_filename") or ""
        # So sánh chính xác tên file hoặc id
        if fn == target_file_clean or fn.startswith(target_file_clean.replace(".wav", "")):
            found_sample = sample
            break
        
        # In tiến độ tìm kiếm mỗi 500 mẫu
        if count % 1000 == 0:
            print(f"⏳ Đã quét qua {count} mẫu...")

    if not found_sample:
        print(f"\n❌ Không tìm thấy file '{target_file_clean}' trong dataset.")
        sys.exit(1)

    print(f"\n✨ Đã tìm thấy file sau khi quét {count} mẫu!")
    
    voice_id = found_sample.get("voice_id", "Voice")
    transcript_text = found_sample.get("text") or ""
    
    # Đặt tên file lưu ra
    if CUSTOM_SAVE_NAME:
        save_filename = CUSTOM_SAVE_NAME
    else:
        # Tạo tên ngắn gọn theo voice_id (hoặc giữ tên UUID)
        clean_voice_name = voice_id.lower().replace(" ", "_")
        save_filename = f"{clean_voice_name}_{target_file_clean[:8]}.wav"

    output_path = os.path.join(current_dir, save_filename)

    # Trích xuất dữ liệu âm thanh
    audio_info = found_sample.get("audio", {})
    audio_array = None
    sample_rate = 24000

    if isinstance(audio_info, dict):
        audio_array = audio_info.get("array")
        sample_rate = audio_info.get("sampling_rate", 24000)
        if audio_array is None and "bytes" in audio_info and audio_info["bytes"]:
            audio_array, sample_rate = sf.read(io.BytesIO(audio_info["bytes"]))
    elif isinstance(audio_info, (bytes, bytearray)):
        audio_array, sample_rate = sf.read(io.BytesIO(audio_info))

    if audio_array is not None:
        # Ghi ra file WAV 16-bit PCM
        sf.write(output_path, audio_array, sample_rate, format="WAV", subtype="PCM_16")
        duration = len(audio_array) / sample_rate
        
        print("=" * 60)
        print(f"✅ ĐÃ TẢI THÀNH CÔNG:")
        print(f"📁 Đường dẫn lưu: {output_path}")
        print(f"🎭 Voice ID trong dataset: {voice_id}")
        print(f"🎵 Sample Rate: {sample_rate}Hz | Thời lượng: {duration:.2f}s")
        print(f"📝 Câu thoại mẫu (Transcript): \"{transcript_text}\"")
        print("=" * 60)

        # Cập nhật hoặc in cấu hình cho voices.json
        voice_entry = {
            "id": save_filename,
            "name": f"👩 Giọng {voice_id} ({save_filename})",
            "audio": f"backend/voices/{save_filename}",
            "text": transcript_text
        }

        # Tự động cập nhật vào voices.json nếu file tồn tại
        if os.path.exists(voices_json_path):
            try:
                with open(voices_json_path, "r", encoding="utf-8") as f:
                    voices_data = json.load(f)
                if isinstance(voices_data, list):
                    # Kiểm tra xem đã có id này chưa
                    existing_idx = next((i for i, v in enumerate(voices_data) if v.get("id") == save_filename), None)
                    if existing_idx is not None:
                        voices_data[existing_idx] = voice_entry
                    else:
                        voices_data.append(voice_entry)
                    with open(voices_json_path, "w", encoding="utf-8") as f:
                        json.dump(voices_data, f, ensure_ascii=False, indent=2)
                    print(f"🎉 Đã TỰ ĐỘNG THÊM giọng này vào file voices.json!")
            except Exception as json_err:
                print(f"⚠️ Không tự động ghi vào voices.json được ({json_err}). Bạn có thể thêm thủ công:")

        print("\n👉 Cấu hình JSON:")
        print(json.dumps(voice_entry, ensure_ascii=False, indent=2))

    else:
        print("❌ Không đọc được audio_array từ sample.")

except Exception as e:
    print(f"❌ Lỗi: {e}")
