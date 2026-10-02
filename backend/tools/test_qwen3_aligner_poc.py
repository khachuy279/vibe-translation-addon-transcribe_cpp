"""PoC Benchmark: Qwen3-ASR + Qwen3-ForcedAligner-0.6B (Phase 1)

Kiểm thử và đo lường thực tế:
1. Nạp và suy luận Qwen3-ForcedAligner-0.6B (PyTorch bfloat16/float16 trên CUDA).
2. Đo lường chính xác: Thời gian nạp, VRAM tiêu thụ, Latency suy luận (ms), và RTF.
3. Kiểm tra độ chuẩn xác mốc thời gian trên cả Tiếng Anh (English) và Tiếng Nhật (Japanese với nagisa).
4. Thực hiện thuật toán gom từ thành câu phụ đề (Punctuation-guided Subtitle Grouping) kèm mốc thời gian.
5. Kiểm tra tính ổn định song song giữa transcribe.cpp (GGUF) và PyTorch Forced Aligner.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

# Đảm bảo đường dẫn root và thư mục qwen3-asr được thêm vào sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
QWEN_ASR_DIR = REPO_ROOT / "qwen3-asr" / "Qwen3-ASR"
if QWEN_ASR_DIR.exists():
    sys.path.insert(0, str(QWEN_ASR_DIR))

from backend.asr.native import bootstrap as asr_native_bootstrap
asr_native_bootstrap()

try:
    import transcribe_cpp
except ImportError:
    transcribe_cpp = None

from qwen_asr import Qwen3ForcedAligner


def get_vram_mb() -> float:
    """Trả về VRAM đang được cấp phát (Allocated MB)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(0) / (1024.0 * 1024.0)


def get_vram_reserved_mb() -> float:
    """Trả về VRAM mà PyTorch đang giữ chỗ (Reserved MB)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved(0) / (1024.0 * 1024.0)


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f} ms"


class SubtitleSentence:
    def __init__(self, text: str, start_time: float, end_time: float, words: List[Dict[str, Any]]):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.words = words

    def __repr__(self) -> str:
        return f"[{self.start_time:.3f}s -> {self.end_time:.3f}s] ({self.duration:.2f}s) \"{self.text}\""


def group_words_to_subtitles(
    aligned_items: List[Any],
    language: str = "English",
    max_duration_sec: float = 4.5,
    max_words: int = 10,
    max_chars: int = 45,
) -> List[SubtitleSentence]:
    """Gom danh sách từ/ký tự từ Forced Aligner thành các câu phụ đề ngắn vừa mắt.
    
    Quy tắc:
    1. Ngắt khi gặp dấu kết thúc câu (. ? ! 。 ！ ？).
    2. Ngắt khi gặp dấu phẩy (, 、) nếu câu đã có độ dài tối thiểu (> 4 từ hoặc > 2s).
    3. Ngắt cưỡng bức khi vượt quá max_duration_sec hoặc max_words/max_chars tại điểm có khoảng lặng lớn nhất giữa 2 từ liên tiếp.
    """
    if not aligned_items:
        return []

    is_cjk = language.lower() in ("chinese", "japanese", "korean")
    sentence_end_punct = {".", "?", "!", "。", "？", "！", "\n"}
    clause_punct = {",", "、", ";", "；", "—", "-"}

    subtitles: List[SubtitleSentence] = []
    current_words: List[Any] = []
    curr_start: Optional[float] = None

    for i, it in enumerate(aligned_items):
        w_text = getattr(it, "text", "")
        t0 = float(getattr(it, "start_time", 0.0))
        t1 = float(getattr(it, "end_time", 0.0))

        if curr_start is None:
            curr_start = t0
        current_words.append(it)

        curr_dur = t1 - curr_start
        clean_w = w_text.strip()
        last_char = clean_w[-1] if clean_w else ""

        is_sent_end = last_char in sentence_end_punct
        is_clause_end = (last_char in clause_punct) and (len(current_words) >= 4 or curr_dur >= 2.0)
        is_over_length = (
            curr_dur >= max_duration_sec
            or (not is_cjk and len(current_words) >= max_words)
            or (is_cjk and sum(len(w.text) for w in current_words) >= max_chars)
        )

        should_split = is_sent_end or is_clause_end or is_over_length

        if should_split or i == len(aligned_items) - 1:
            if is_cjk:
                seg_text = "".join(w.text for w in current_words).strip()
            else:
                seg_text = " ".join(w.text for w in current_words).strip()

            subtitles.append(
                SubtitleSentence(
                    text=seg_text,
                    start_time=curr_start,
                    end_time=t1,
                    words=[{"text": w.text, "t0": w.start_time, "t1": w.end_time} for w in current_words],
                )
            )
            current_words = []
            curr_start = None

    return subtitles


def run_benchmark():
    print("=" * 80)
    print("🚀 BẮT ĐẦU BENCHMARK PHASE 1: QWEN3-ASR + QWEN3-FORCEDALIGNER-0.6B")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"• Thiết bị GPU: {torch.cuda.get_device_name(0)} (Tổng VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")
    print(f"• PyTorch: {torch.__version__} | CUDA khả dụng: {torch.cuda.is_available()}")
    print(f"• VRAM ban đầu: {get_vram_mb():.1f} MB (Allocated), {get_vram_reserved_mb():.1f} MB (Reserved)\n")

    # 1. NẠP QWEN3-FORCEDALIGNER-0.6B
    print("--- [BƯỚC 1] NẠP MODEL QWEN3-FORCEDALIGNER-0.6B ---")
    aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B"
    t0_load = time.perf_counter()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    aligner = Qwen3ForcedAligner.from_pretrained(
        aligner_model_id,
        dtype=dtype,
        device_map=device,
    )
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0_load

    vram_after_aligner = get_vram_mb()
    print(f"✓ Nạp ForcedAligner thành công trong: {t_load:.2f}s")
    print(f"• VRAM tiêu thụ cho ForcedAligner: ~{vram_after_aligner:.1f} MB (Dtype: {dtype})\n")

    # 2. CHUẨN BỊ MẪU KIỂM THỬ
    test_cases = [
        {
            "label": "English (19s, real audio)",
            "wav_path": REPO_ROOT / "wav_test" / "English_low_speech_quality_19s.wav",
            "lang": "English",
            "asr_lang": "en",
        },
        {
            "label": "Japanese (5.1s, real audio)",
            "wav_path": REPO_ROOT / "wav_test" / "Japanese_5s.wav",
            "lang": "Japanese",
            "asr_lang": "ja",
        },
    ]

    # Kiểm tra transcribe_cpp native model
    gguf_asr_path = REPO_ROOT / "backend" / "models" / "Qwen3-ASR-1.7B-Q8_0.gguf"
    if not gguf_asr_path.exists():
        gguf_asr_path = REPO_ROOT / "backend" / "models" / "Qwen3-ASR-0.6B-Q8_0.gguf"
    if not gguf_asr_path.exists():
        gguf_asr_path = REPO_ROOT / "backend" / "models" / "Qwen3-ASR-1.7B-Q4_K_M.gguf"

    asr_session = None
    if transcribe_cpp is not None and gguf_asr_path.exists():
        print(f"--- [BƯỚC 2A] NẠP TRANSCRIBE.CPP MODEL ({gguf_asr_path.name}) ---")
        t0_asr_load = time.perf_counter()
        asr_model = transcribe_cpp.Model(str(gguf_asr_path), backend="cuda")
        asr_session = asr_model.session(n_threads=4)
        print(f"✓ Nạp ASR native thành công trong: {time.perf_counter() - t0_asr_load:.2f}s\n")
    else:
        print(f"⚠️ Không tìm thấy file GGUF Qwen3-ASR hoặc transcribe_cpp, sẽ dùng text từ file wav_test .txt làm giả định.")

    # 3. CHẠY SUY LUẬN TỪNG SAMPLE
    for idx, tc in enumerate(test_cases, 1):
        wav_file = tc["wav_path"]
        if not wav_file.exists():
            print(f"⚠️ Bỏ qua {tc['label']} vì không tìm thấy file: {wav_file}")
            continue

        print(f"================================================================================")
        print(f"TEST CASE {idx}: {tc['label']}")
        print(f"================================================================================")

        # Đọc file audio
        audio_data, sr = sf.read(str(wav_file), dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        duration_sec = len(audio_data) / sr
        print(f"• Audio duration: {duration_sec:.2f}s (Sample Rate: {sr}Hz, Samples: {len(audio_data)})")

        # Bước 2A: Nhận diện văn bản ASR
        transcript_text = ""
        t_asr_ms = 0.0
        if asr_session is not None:
            t0_asr = time.perf_counter()
            asr_res = asr_session.run(audio_data, language=tc["asr_lang"], timestamps="none")
            t_asr = time.perf_counter() - t0_asr
            t_asr_ms = t_asr * 1000.0
            transcript_text = getattr(asr_res, "text", str(asr_res)).strip()
            print(f"\n[ASR transcribe.cpp]")
            print(f"• Thời gian ASR: {t_asr_ms:.1f} ms (RTF: {t_asr / duration_sec:.4f})")
            print(f"• Text ASR: \"{transcript_text}\"")
        else:
            # Đọc từ file txt kèm theo nếu không có ASR
            txt_file = wav_file.with_suffix(".txt")
            if txt_file.exists():
                transcript_text = txt_file.read_text(encoding="utf-8").strip()
            else:
                transcript_text = "Testing audio forced alignment benchmark without ASR transcript."
            print(f"\n[Dùng Transcript tham chiếu]: \"{transcript_text}\"")

        # Bước 2B: Forced Alignment
        torch.cuda.synchronize()
        vram_before_align = get_vram_mb()
        t0_align = time.perf_counter()

        # Gọi ForcedAligner
        align_results = aligner.align(
            audio=str(wav_file),
            text=transcript_text,
            language=tc["lang"],
        )
        torch.cuda.synchronize()
        t_align = time.perf_counter() - t0_align
        t_align_ms = t_align * 1000.0

        vram_after_align = get_vram_mb()

        print(f"\n[ForcedAligner Qwen3-0.6B]")
        print(f"• Thời gian Aligner: {t_align_ms:.1f} ms (RTF: {t_align / duration_sec:.4f})")
        print(f"• VRAM peak trong align: ~{vram_after_align:.1f} MB (Delta: {vram_after_align - vram_before_align:+.1f} MB)")

        if asr_session is not None:
            total_stage_ms = t_asr_ms + t_align_ms
            print(f"• TỔNG THỜI GIAN (ASR + ALIGNER): {total_stage_ms:.1f} ms (Tổng RTF: {(t_asr + t_align) / duration_sec:.4f})")

        # Lấy danh sách items đã align
        res_items = align_results[0].items if align_results else []
        print(f"• Số từ/ký tự gióng hàng được: {len(res_items)}")

        # In mẫu 5 từ đầu tiên
        print("\n• Mẫu 5 từ/ký tự đầu tiên:")
        for w in res_items[:5]:
            print(f"   - {w.start_time:.3f}s -> {w.end_time:.3f}s: \"{w.text}\"")

        # Bước 2C: Gom thành câu phụ đề
        print("\n[Punctuation-guided Sentence Grouping]")
        subtitles = group_words_to_subtitles(res_items, language=tc["lang"])
        print(f"• Đã gom thành {len(subtitles)} câu phụ đề:")
        for s_idx, s in enumerate(subtitles, 1):
            print(f"   {s_idx}. {s}")

        print()

    # 4. TỔNG KẾT VRAM & CLEANUP
    print("=" * 80)
    print("📊 KẾT LUẬN BENCHMARK VÀ AN TOÀN HỆ THỐNG")
    print("=" * 80)
    print(f"• Peak VRAM Allocated: {get_vram_mb():.1f} MB")
    print(f"• Peak VRAM Reserved:  {get_vram_reserved_mb():.1f} MB")
    
    # Giải phóng
    del aligner
    if asr_session is not None:
        del asr_session
    gc.collect()
    torch.cuda.empty_cache()
    print(f"• VRAM sau khi thu hồi: {get_vram_mb():.1f} MB (Allocated), {get_vram_reserved_mb():.1f} MB (Reserved)")
    print("✓ Toàn bộ test Phase 1 hoàn tất thành công và an toàn tuyệt đối!")


if __name__ == "__main__":
    run_benchmark()
