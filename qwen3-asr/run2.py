import time
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32,
    max_new_tokens=256,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        # attn_implementation="flash_attention_2",
    ),
)

audio_list = [
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
]
language_list = ["Chinese", "English"]

# ===== Đo thời gian =====
torch.cuda.synchronize()          # đồng bộ GPU trước khi bắt đầu đo
start = time.perf_counter()

results = model.transcribe(
    audio=audio_list,
    language=language_list,
    return_time_stamps=True,
)

torch.cuda.synchronize()          # đồng bộ GPU sau khi xong
end = time.perf_counter()
total_proc_time = end - start     # tổng thời gian xử lý cả batch (giây)

print(f"\nTổng thời gian xử lý batch: {total_proc_time:.4f} s\n")

# ===== In kết quả + RTF từng câu =====
for i, r in enumerate(results):
    # Lấy độ dài audio từ timestamp cuối cùng (chính xác nhất khi dùng forced aligner)
    if r.time_stamps and len(r.time_stamps) > 0:
        audio_duration = r.time_stamps[-1].end_time
    else:
        audio_duration = None

    print(f"=== Sample {i} ===")
    print(f"Language : {r.language}")
    print(f"Text     : {r.text}")
    
    if audio_duration is not None:
        # Vì đang chạy batch, RTF của từng câu chỉ mang tính tham khảo
        # (thời gian thực tế của từng câu bị chia sẻ trong batch)
        rtf_approx = total_proc_time / audio_duration / len(results)
        print(f"Duration : {audio_duration:.3f} s")
        print(f"RTF (xấp xỉ) : {rtf_approx:.4f}")
    print()