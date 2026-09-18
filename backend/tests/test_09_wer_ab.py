"""Harness WER A/B (tầng B) — đo độ chính xác THẬT để gỡ các quyết định bị chặn.

Mục đích (theo `03_KE_HOACH_TRIEN_KHAI.md`):
- **P1.6**: `silence_duration_ms` 600 ms → 450 ms có làm WER xấu đi không?
- **P2.5**: `reuse_preview_for_commit` có làm mất từ cuối câu không? (mặc định đang TẮT)
- **Kiểm chứng C4**: cắt câu theo VAD + CommitManager tốn bao nhiêu độ chính xác so với
  transcribe **toàn file một lần** (không phân đoạn)?

Cách làm cho mỗi file:
1. **REF (offline)**: `_run_inference_sync(toàn bộ audio)` → trần chất lượng của model khi
   không bị phân đoạn. (Lưu ý: vẫn bị cắt theo trần context của model.)
2. **STREAM**: chạy pipeline VAD + ASR thật có pacing, ghép các câu `final` lại.
3. Chấm điểm cả hai với ground truth `wav_test/*.txt`, và chấm thêm STREAM vs REF để
   **tách riêng phần mất mát do phân đoạn** khỏi sai số của model.

Chuẩn hoá & thang đo (không phụ thuộc `jiwer`/`uv`):
- Bỏ span metadata `[...]` / `<...>` và nhãn `Speaker N:`; lowercase; bỏ dấu câu; gộp khoảng trắng.
- Có CJK (zh/ja/ko) → **CER** theo ký tự; còn lại → **WER** theo từ.

Không có hàm `test_*` để bộ test mặc định (`pytest`) vẫn nhanh (< 15 s).

Cách dùng:
    python backend/tests/test_09_wer_ab.py --model qwen3-asr-0.6b --speed 6 --max-sec 25
    python backend/tests/test_09_wer_ab.py --configs base,reuse --json report/audit/06_wer_ab.json
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.utils.cuda import setup_cuda_dll_paths

setup_cuda_dll_paths()

# ⚠️ THỨ TỰ IMPORT LÀ QUAN TRỌNG — bản cũ SAI ở đây.
#
# Bản cũ `import transcribe_cpp` TRỰC TIẾP tại chỗ này, tức là nạp native của **wheel PyPI**
# vào `sys.modules` TRƯỚC khi `backend.asr` kịp chạy `bootstrap()`. Hệ quả: bootstrap không
# áp được bundle trong `bin/`, và cả harness đo trên **Vulkan của wheel** trong khi bản
# chạy thật dùng **CUDA trong `bin/`** (log cảnh báo:
# "transcribe_cpp đã được import TRƯỚC khi bootstrap ... bundle trong bin/ KHÔNG được áp dụng").
# Điều đó làm mọi số `infer_ms` của A/B không đại diện cho cấu hình phát hành.
#
# Nay: để `backend.asr` (→ `native.bootstrap()`) chạy TRƯỚC, rồi mới chạm `transcribe_cpp`.
from backend.asr import native as _asr_native  # noqa: E402  (bootstrap bin/ ngay tại đây)

_asr_native.bootstrap()
try:
    import transcribe_cpp  # noqa: F401
except Exception:
    pass

from backend.config import config  # noqa: E402
from backend.tests.test_08_streaming_latency import load_wav_16k, run_paced  # noqa: E402

WAV_DIR = _ROOT / "wav_test"

_METADATA_SPAN = re.compile(r"[\[<][^\]>]*[\]>]")
_SPEAKER_LABEL = re.compile(r"\bspeaker\s*\d+\s*:", flags=re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u1100-\u11ff]")


def normalize_text(text: Optional[str]) -> str:
    """Chuẩn hoá để chấm điểm: bỏ metadata/nhãn, lowercase, bỏ dấu câu, gộp khoảng trắng."""
    if not text:
        return ""
    t = _METADATA_SPAN.sub(" ", text)
    t = _SPEAKER_LABEL.sub(" ", t)
    t = t.lower()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def is_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))


def tokens_for(text: str, cjk: bool) -> List[str]:
    """Đơn vị chấm điểm: ký tự (bỏ khoảng trắng) cho CJK, từ cho chữ Latin."""
    norm = normalize_text(text)
    if cjk:
        return [ch for ch in norm if not ch.isspace()]
    return norm.split()


def levenshtein_counts(ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
    """Trả (substitutions, deletions, insertions) bằng DP tối giản bộ nhớ."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0
    prev_s = [0] * (m + 1)
    prev_d = [0] * (m + 1)
    prev_i = [0] * (m + 1)
    for j in range(1, m + 1):
        prev_i[j] = j
    for i in range(1, n + 1):
        cur_s = [0] * (m + 1)
        cur_d = [0] * (m + 1)
        cur_i = [0] * (m + 1)
        cur_d[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur_s[j], cur_d[j], cur_i[j] = prev_s[j - 1], prev_d[j - 1], prev_i[j - 1]
                continue
            # Chọn phép biến đổi có tổng lỗi nhỏ nhất
            sub_total = prev_s[j - 1] + prev_d[j - 1] + prev_i[j - 1] + 1
            del_total = prev_s[j] + prev_d[j] + prev_i[j] + 1
            ins_total = cur_s[j - 1] + cur_d[j - 1] + cur_i[j - 1] + 1
            best = min(sub_total, del_total, ins_total)
            if best == sub_total:
                cur_s[j], cur_d[j], cur_i[j] = prev_s[j - 1] + 1, prev_d[j - 1], prev_i[j - 1]
            elif best == del_total:
                cur_s[j], cur_d[j], cur_i[j] = prev_s[j], prev_d[j] + 1, prev_i[j]
            else:
                cur_s[j], cur_d[j], cur_i[j] = cur_s[j - 1], cur_d[j - 1], cur_i[j - 1] + 1
        prev_s, prev_d, prev_i = cur_s, cur_d, cur_i
    return prev_s[m], prev_d[m], prev_i[m]


def score_pair(ref_text: str, hyp_text: str) -> Dict[str, float]:
    """Chấm một cặp (ref, hyp). Tự chọn WER (từ) hay CER (ký tự) theo nội dung."""
    cjk = is_cjk(ref_text)
    ref = tokens_for(ref_text, cjk)
    hyp = tokens_for(hyp_text, cjk)
    sub, dele, ins = levenshtein_counts(ref, hyp)
    denom = max(1, len(ref))
    return {
        "metric": "cer" if cjk else "wer",
        "ref_units": len(ref),
        "hyp_units": len(hyp),
        "substitutions": sub,
        "deletions": dele,
        "insertions": ins,
        "error_rate": (sub + dele + ins) / denom,
        "empty_hypothesis": len(hyp) == 0,
    }


def ground_truth_for(wav_path: Path) -> str:
    txt = wav_path.with_suffix(".txt")
    if not txt.exists():
        return ""
    return txt.read_text(encoding="utf-8")


# ------------------------------------------------------------------ cấu hình A/B
CONFIGS: Dict[str, Dict[str, object]] = {
    "base": {
        "label": "Mặc định hiện tại (silence=600, reuse=False)",
        "silence_duration_ms": 600,
        "reuse_preview_for_commit": False,
    },
    # ĐỐI CHỨNG QUAN TRỌNG: y hệt `base`. Chạy `--configs base,base_repeat` để đo SÀN NHIỄU
    # (model nondeterminism + điểm cắt phụ thuộc pacing). Nếu base-vs-base đã lớn thì không
    # được quy phần chênh lệch base-vs-REF cho việc phân đoạn.
    "base_repeat": {
        "label": "Đối chứng sàn nhiễu (y hệt base)",
        "silence_duration_ms": 600,
        "reuse_preview_for_commit": False,
    },
    "reuse": {
        "label": "P2.5: reuse_preview_for_commit=True",
        "silence_duration_ms": 600,
        "reuse_preview_for_commit": True,
    },
    "silence450": {
        "label": "P1.6: silence_duration_ms=450",
        "silence_duration_ms": 450,
        "reuse_preview_for_commit": False,
    },
    "no_tier234": {
        "label": "Đối chứng: tắt BẬC 2/3/4 (chỉ cắt theo VAD)",
        "silence_duration_ms": 600,
        "reuse_preview_for_commit": False,
        "enable_tier234": False,
    },
    "window_off": {
        "label": "Đối chứng: tắt cửa sổ preview (O(N²) gốc)",
        "silence_duration_ms": 600,
        "reuse_preview_for_commit": False,
        "preview_window_sec": 0.0,
    },
}


def apply_config(spec: Dict[str, object]) -> None:
    config.vad.silence_duration_ms = int(spec.get("silence_duration_ms", 600))
    config.asr.preview_reuse_for_commit = bool(spec.get("reuse_preview_for_commit", False))
    config.sentence.enable_tier234 = bool(spec.get("enable_tier234", True))
    config.asr.preview_window_sec = float(spec.get("preview_window_sec", 6.0))


def main() -> int:
    # RÀO CHẮN RAM CỨNG (xem backend/utils/mem_guard.py): lỗi phình bộ nhớ native đã có lúc
    # đẩy tiến trình lên hàng chục GB. Trần mặc định 4000 MB, đổi bằng `MEM_GUARD_MB`; 0 = tắt.
    from backend.utils.mem_guard import start_guard

    start_guard()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-asr-0.6b")
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--max-sec", type=float, default=25.0, help="cắt bớt audio mỗi file")
    ap.add_argument("--configs", default="base,reuse,silence450")
    ap.add_argument("--json", default=None)
    ap.add_argument("--files", default=None, help="danh sách tên file, cách nhau dấu phẩy")
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="số lần lặp mỗi (file, cấu hình) để đo SÀN NHIỄU; >=3 mới đủ để kết luận A/B",
    )
    args = ap.parse_args()

    # Chẩn đoán treo: bật `WER_HANG_DUMP=1` để dump stack MỌI thread ra stderr mỗi
    # `WER_HANG_EVERY` giây. Đã hai lần bắt gặp tiến trình chạy tới ~33 GB RAM mà không
    # kết thúc; dump này chỉ ra chính xác nó đang kẹt ở frame Python nào.
    if os.environ.get("WER_HANG_DUMP") == "1":
        import faulthandler

        faulthandler.enable()
        faulthandler.dump_traceback_later(
            float(os.environ.get("WER_HANG_EVERY", "30")), repeat=True
        )
        print("[diag] đã bật faulthandler dump_traceback_later", flush=True)

    wanted = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
    if not wanted:
        print("Không có config hợp lệ. Chọn trong:", ", ".join(CONFIGS))
        return 2

    if args.files:
        names = [n.strip() for n in args.files.split(",") if n.strip()]
        wavs = [WAV_DIR / n for n in names if (WAV_DIR / n).exists()]
    else:
        wavs = sorted(p for p in WAV_DIR.glob("*.wav") if p.with_suffix(".txt").exists())

    print("=" * 100)
    print("WER/CER A/B — đo độ chính xác thật của pipeline streaming")
    print("=" * 100)
    print(f"model   : {args.model}")
    print(f"files   : {len(wavs)}  | speed={args.speed}x | max_sec={args.max_sec}")
    print(f"configs : {', '.join(wanted)}")
    print("-" * 100)
    print("LƯU Ý CÁCH ĐỌC:")
    print("  * 'vs REF' = streaming so với chính model đó chạy OFFLINE trên CÙNG đoạn audio.")
    print("    Đây là thước đo MẤT MÁT DO PHÂN ĐOẠN — không phụ thuộc việc cắt audio.")
    print("  * 'error vs GT' chỉ hợp lệ khi audio được dùng TRỌN VẸN (coverage >= 98%).")
    print("    File bị cắt bớt (coverage thấp) có error vô nghĩa do thiếu phần đuôi.")
    print("-" * 100)

    # --- Nạp model 1 lần cho cả REF và STREAM ---
    from backend.asr.engine import TranscribeEngine
    from backend.asr.registry import ModelRegistry

    registry = ModelRegistry.get_instance()
    registry.set_active_model_key(args.model)
    config.asr.active_model = args.model
    warm = TranscribeEngine(model_key=args.model, session_id="wer")
    t0 = time.perf_counter()
    warm._preload_model()
    print(f"[setup] nạp + pre-warm model: {time.perf_counter() - t0:.1f}s")

    results: List[Dict[str, object]] = []

    for wav in wavs:
        gt_raw = ground_truth_for(wav)
        audio_full = load_wav_16k(wav)
        full_duration = len(audio_full) / 16000.0
        audio = audio_full[: int(args.max_sec * 16000)] if args.max_sec > 0 else audio_full
        duration = len(audio) / 16000.0
        coverage = duration / full_duration if full_duration > 0 else 1.0

        # --- REF: transcribe toàn file một lần (trần chất lượng, không phân đoạn) ---
        # QUAN TRỌNG: gọi qua `_infer_with_watchdog` chứ KHÔNG gọi thẳng
        # `_run_inference_sync`. Gọi thẳng thì không có đồng hồ cắt ngắn nào: khi native
        # `transcribe_run` bị treo (đã bắt được bằng dump stack, kèm phình RAM tới hàng
        # chục GB — xem report §12.6) harness sẽ treo vĩnh viễn và không ghi được JSON.
        t_ref = time.perf_counter()
        ref_text = ""
        for attempt in (1, 2):
            ref_text = asyncio.run(warm._infer_with_watchdog(audio))
            if ref_text:
                break
            print(f"   [REF] lần {attempt} không có kết quả (native treo?) — thử lại")
        ref_ms = (time.perf_counter() - t_ref) * 1000.0
        ref_available = bool(ref_text)
        if not ref_available:
            print(f"   [REF] BỎ {wav.name}: không lấy được bản tham chiếu -> loại khỏi thống kê 'vs REF'")
        ref_score = score_pair(gt_raw, ref_text)

        row: Dict[str, object] = {
            "file": wav.name,
            "duration_sec": round(duration, 2),
            "full_duration_sec": round(full_duration, 2),
            "coverage": round(coverage, 4),
            "ground_truth_chars": len(gt_raw),
            "reference": {
                "text": ref_text,
                "available": ref_available,
                "infer_ms": round(ref_ms, 1),
                **ref_score,
            },
            "configs": {},
        }

        cov_tag = "" if coverage >= 0.98 else f" [coverage {coverage * 100:.0f}%]"
        line = (f"{wav.name[:40]:42} {duration:5.1f}s{cov_tag:16}  REF {ref_score['metric']}="
                f"{ref_score['error_rate'] * 100:6.2f}%")

        for cfg_name in wanted:
            spec = CONFIGS[cfg_name]
            runs: List[Dict[str, object]] = []

            # `--repeats N`: chạy N lần cùng một cấu hình để đo SÀN NHIỄU. Điểm cắt câu
            # phụ thuộc pacing/thời điểm VAD nên hai lần chạy y hệt nhau vẫn cho WER khác
            # nhau; chênh lệch A/B chỉ đáng tin nếu LỚN HƠN sàn nhiễu này.
            for _rep in range(max(1, args.repeats)):
                apply_config(spec)

                wall, msgs = asyncio.run(
                    run_paced(audio, args.model, args.speed, config.asr.poll_interval_ms)
                )
                finals = [m for m in msgs if m.get("type") == "utterance_update" and m.get("is_final")]
                previews = [m for m in msgs if m.get("type") == "utterance_update" and not m.get("is_final")]
                stream_text = " ".join((m.get("text") or "").strip() for m in finals).strip()

                gt_score = score_pair(gt_raw, stream_text)
                # STREAM vs REF: cô lập phần mất mát do phân đoạn (cùng model, cùng audio)
                vs_ref = score_pair(ref_text, stream_text)
                durations = [m.get("inference_ms", 0.0) for m in finals if m.get("inference_ms")]

                reasons: Dict[str, int] = {}
                for m in finals:
                    r = m.get("commit_reason", "?")
                    reasons[r] = reasons.get(r, 0) + 1

                runs.append({
                    "stream_text": stream_text,
                    "commits": len(finals),
                    "previews": len(previews),
                    "error_rate": gt_score["error_rate"],
                    "metric": gt_score["metric"],
                    "empty_hypothesis": gt_score["empty_hypothesis"],
                    "stream_vs_reference_error_rate": vs_ref["error_rate"],
                    "commit_reasons": reasons,
                    "wall_sec": round(wall, 1),
                    "commit_infer_ms": durations,
                })

            first = runs[0]
            row["configs"][cfg_name] = {
                "label": spec["label"],
                "runs": runs,
                # Giữ các trường phẳng = lần chạy ĐẦU để tương thích với báo cáo cũ.
                **first,
                "repeats": len(runs),
                "error_rate_mean": statistics.fmean(float(r["error_rate"]) for r in runs),
                "vs_reference_error_rate_mean": statistics.fmean(
                    float(r["stream_vs_reference_error_rate"]) for r in runs
                ),
                "commits_mean": statistics.fmean(int(r["commits"]) for r in runs),
            }

            def _fmt(values: List[float], prefix: str) -> str:
                if len(values) == 1:
                    return f"{prefix}{values[0] * 100:6.2f}%"
                return (
                    f"{prefix}{statistics.fmean(values) * 100:6.2f}%"
                    f"[{min(values) * 100:.2f}..{max(values) * 100:.2f}]"
                )

            errs = [float(r["error_rate"]) for r in runs]
            refs = [float(r["stream_vs_reference_error_rate"]) for r in runs]
            cmts = "/".join(str(int(r["commits"])) for r in runs)
            line += (
                f"  |  {cfg_name:11} {_fmt(errs, str(first['metric']) + '=')} "
                f"(vs REF {_fmt(refs, '')}, {cmts} câu)"
            )

        print(line)
        results.append(row)

    # --- Tổng hợp ---
    # Tách hai thang đo:
    #  (1) 'vs REF'  : mất mát do PHÂN ĐOẠN  -> tính trên MỌI file (không phụ thuộc coverage)
    #  (2) 'error'   : chất lượng tuyệt đối  -> CHỈ tính trên file phủ trọn (coverage >= 98%)
    print("=" * 100)
    full_files = [r for r in results if float(r["coverage"]) >= 0.98]
    capped_files = [r for r in results if float(r["coverage"]) < 0.98]
    print(f"File phủ trọn (dùng cho chất lượng tuyệt đối): {len(full_files)}/{len(results)}"
          + (f"  | bị cắt: {', '.join(str(r['file']) for r in capped_files)}" if capped_files else ""))

    summary: Dict[str, Dict[str, float]] = {}
    ref_files = [r for r in results if r["reference"].get("available", True)]  # type: ignore[union-attr]
    no_ref_files = [r for r in results if not r["reference"].get("available", True)]  # type: ignore[union-attr]
    if no_ref_files:
        print(f"⚠️  {len(no_ref_files)} file KHÔNG có REF (native treo) nên bị loại khỏi 'vs REF': "
              + ", ".join(str(r["file"]) for r in no_ref_files))
    units_all = sum(float(r["reference"]["ref_units"]) for r in ref_files) or 1.0  # type: ignore[index]
    units_full = sum(float(r["reference"]["ref_units"]) for r in full_files) or 1.0  # type: ignore[index]

    for cfg_name in wanted:
        per_repeat: List[Dict[str, float]] = []
        for rep in range(max(1, args.repeats)):
            acc_all = 0.0
            acc_full = 0.0
            empty = 0
            commits = 0
            for r in results:
                c = r["configs"][cfg_name]["runs"][rep]  # type: ignore[index]
                u = float(r["reference"]["ref_units"])  # type: ignore[index]
                if r in ref_files:
                    acc_all += float(c["stream_vs_reference_error_rate"]) * u
                commits += int(c["commits"])
                if r in full_files:
                    acc_full += float(c["error_rate"]) * u
                if c["empty_hypothesis"]:
                    empty += 1
            per_repeat.append({
                "vs_reference_error_rate": acc_all / (units_all or 1.0),
                "error_rate_full_coverage_only": acc_full / (units_full or 1.0),
                "empty_hypothesis_files": empty,
                "total_commits": commits,
            })

        vs_vals = [p["vs_reference_error_rate"] for p in per_repeat]
        err_vals = [p["error_rate_full_coverage_only"] for p in per_repeat]
        summary[cfg_name] = {
            # Giữ khoá cũ = TRUNG BÌNH qua các lần lặp (tương thích báo cáo cũ).
            "vs_reference_error_rate": statistics.fmean(vs_vals),
            "error_rate_full_coverage_only": statistics.fmean(err_vals),
            "empty_hypothesis_files": max(int(p["empty_hypothesis_files"]) for p in per_repeat),
            "total_commits": int(statistics.fmean(p["total_commits"] for p in per_repeat)),
            "repeats": len(per_repeat),
            "vs_reference_error_rate_min": min(vs_vals),
            "vs_reference_error_rate_max": max(vs_vals),
            "error_rate_full_coverage_min": min(err_vals),
            "error_rate_full_coverage_max": max(err_vals),
            "per_repeat": per_repeat,
        }

    ref_err_full = (
        sum(float(r["reference"]["error_rate"]) * float(r["reference"]["ref_units"]) for r in full_files)  # type: ignore[index]
        / (units_full or 1.0)
    )
    print(f"\n{'REF (offline, không phân đoạn)':38} error(GT, chỉ file phủ trọn)={ref_err_full * 100:6.2f}%")
    print(f"{'':38} {'vs REF':>10} {'error vs GT':>13} {'empty':>6} {'commits':>8}")
    for cfg_name, s in summary.items():
        span = ""
        if int(s["repeats"]) > 1:
            span = (f"   [{s['vs_reference_error_rate_min'] * 100:.2f}.."
                    f"{s['vs_reference_error_rate_max'] * 100:.2f}]")
        print(f"{cfg_name:38} {s['vs_reference_error_rate'] * 100:9.2f}% "
              f"{s['error_rate_full_coverage_only'] * 100:12.2f}% "
              f"{int(s['empty_hypothesis_files']):6d} {int(s['total_commits']):8d}{span}")
    print("-" * 100)

    # SÀN NHIỄU: chênh lệch giữa các lần lặp của CÙNG một cấu hình. Chênh lệch A/B chỉ
    # đáng tin nếu lớn hơn con số này; nếu không thì phải tăng --repeats rồi mới kết luận.
    noise_band = 0.0
    for s in summary.values():
        if int(s["repeats"]) > 1:
            noise_band = max(
                noise_band,
                float(s["vs_reference_error_rate_max"]) - float(s["vs_reference_error_rate_min"]),
                float(s["error_rate_full_coverage_max"]) - float(s["error_rate_full_coverage_min"]),
            )
    if noise_band > 0:
        print(f"SÀN NHIỄU đo được (cùng cấu hình, khác lần chạy): {noise_band * 100:.2f} điểm %")
        print("  => Chênh lệch A/B NHỎ HƠN sàn nhiễu này KHÔNG kết luận được.")
        print("  => Muốn kết luận: tăng --repeats (>=3) rồi so trung bình/khoảng.")

    print("KẾT LUẬN CẦN ĐỌC:")
    print("  * P2.5 (reuse_preview_for_commit): nếu 'vs REF' của 'reuse' CAO HƠN 'base'")
    print("    MỘT CÁCH RÕ RÀNG (lớn hơn sàn nhiễu) => tái sử dụng preview làm mất chữ => GIỮ TẮT.")
    print("  * P1.6 (silence_duration_ms=450): chỉ hạ ngưỡng nếu 'error vs GT' và 'vs REF'")
    print("    không xấu hơn 'base' NGOÀI sàn nhiễu (nếu không, giữ 600).")

    if args.json:
        out = {
            "model": args.model,
            "speed": args.speed,
            "max_sec": args.max_sec,
            "repeats": max(1, args.repeats),
            "noise_band": noise_band,
            "configs": {k: CONFIGS[k] for k in wanted},
            "reference_error_rate_full_coverage_only": ref_err_full,
            "files_fully_covered": [r["file"] for r in full_files],
            "files_capped": [r["file"] for r in capped_files],
            "summary": summary,
            "files": results,
        }
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã ghi JSON: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
