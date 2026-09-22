"""Kiểm tra VAD trên **10 phút AVA-Speech** (`5BDj0ow5hnA`, cửa sổ 900–1500 s).

Mục tiêu (theo yêu cầu dự án):
1. Chứng minh cả 3 engine chạy đúng trên audio thật dài, có **nhãn người** đối chiếu.
2. Chứng minh toàn vẹn tín hiệu trên 10 phút: mỗi frame VAD chuyển cho ASR phải BẰNG ĐÚNG
   slice PCM gốc tại đúng mốc thời gian của nó (0 sai lệch byte).
3. Chốt mức chất lượng tối thiểu (P/R/F1/accuracy + độ trễ onset + RTF) để phát hiện hồi quy.

Dữ liệu: `wav_test/vad/5BDj0ow5hnA.wav` (48 kHz stereo) + `ava_speech_labels_v1.csv`.
Đoạn 10 phút được cắt & resample 16 kHz **một lần** rồi cache (xem
`backend/tests/fixtures/ava_speech.py`); bước resample này chỉ phục vụ test — trong pipeline
thật plugin đã gửi thẳng PCM 16 kHz nên VAD/ASR không resample gì.

Chạy: `pytest -m slow backend/tests/test_43_vad_ava_speech.py -q -s`  (~4–5 phút cho 3 engine).

Ngưỡng dưới đây được chốt từ lần đo đầu (xem `report/02_vad/ava_speech_report.md`) và đặt
THẤP HƠN số đo một khoảng an toàn — mục đích là bắt hồi quy, không phải chặn dao động nhỏ.
"""

import json
from pathlib import Path
import time

import numpy as np
import pytest

from backend.tests.fixtures import ava_speech as av

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: Mức tối thiểu mỗi engine phải đạt trên 10 phút AVA-Speech (ĐO ĐƯỢC → ngưỡng).
FLOORS = {
    "firered-vad": {
        "f1": 0.85, "recall": 0.85, "precision": 0.80, "accuracy": 0.80,
        "rtf_max": 0.60, "onset_median_ms_max": 200, "measured": "F1 0,918 · R 0,929 · P 0,907 · RTF 0,333",
    },
    "silero-vad": {
        "f1": 0.55, "recall": 0.40, "precision": 0.85, "accuracy": 0.55,
        "rtf_max": 0.10, "onset_median_ms_max": 200, "measured": "F1 0,652 · R 0,496 · P 0,952 · RTF 0,022",
    },
    "fsmn-vad": {
        "f1": 0.70, "recall": 0.60, "precision": 0.80, "accuracy": 0.65,
        "rtf_max": 0.15, "onset_median_ms_max": 1000, "measured": "F1 0,777 · R 0,694 · P 0,882 · RTF 0,063",
    },
}

#: Kết quả từng engine cho báo cáo cuối (điền trong lúc chạy test).
_RESULTS: dict = {}


def _recall_by_class(pred, labels, n_frames) -> dict:
    """Recall theo từng lớp nhãn + tỉ lệ báo động giả trên NO_SPEECH.

    VÌ SAO CẦN: AVA-Speech tách `CLEAN_SPEECH` / `SPEECH_WITH_NOISE` / `SPEECH_WITH_MUSIC`.
    VAD nào chỉ tệ ở hai lớp sau thì đó là ĐỘ KHÓ CỦA DOMAIN (phim có nhạc/nhiễu), không phải
    lỗi cấu hình — xem `report/02_vad/ava_vs_vendor_benchmarks.md`.
    """
    out = {}
    for cls in ("CLEAN_SPEECH", "SPEECH_WITH_NOISE", "SPEECH_WITH_MUSIC"):
        mask = av.labels_to_timeline([(s, e, lab) for s, e, lab in labels if lab == cls], n_frames)
        total = int(mask.sum())
        out[cls] = round(float((pred & mask).sum()) / total, 4) if total else None
    silent = np.zeros(n_frames, dtype=bool)
    for start, end, lab in labels:
        if lab != "NO_SPEECH":
            continue
        # KHÔNG dùng `labels_to_timeline` ở đây: helper đó coi NO_SPEECH là "không phải speech"
        # nên bỏ qua ⇒ mặt nạ rỗng và cột báo động giả luôn vô nghĩa (lỗi đã gặp).
        a = max(0, int(round(start * 1000.0 / av.FRAME_MS)))
        b = min(n_frames, int(round(end * 1000.0 / av.FRAME_MS)))
        if b > a:
            silent[a:b] = True
    total_sil = int(silent.sum())
    out["false_alarm_on_no_speech"] = (
        round(float((pred & silent).sum()) / total_sil, 4) if total_sil else None
    )
    return out


@pytest.fixture(scope="module")
def ava_segment() -> av.Segment:
    try:
        return av.ensure_segment(PROJECT_ROOT)
    except FileNotFoundError as exc:  # pragma: no cover - phụ thuộc dữ liệu cục bộ
        pytest.skip(f"thiếu dữ liệu AVA-Speech: {exc}")


@pytest.mark.slow
@pytest.mark.parametrize("engine_name", ["firered-vad", "silero-vad", "fsmn-vad"])
def test_vad_tren_10_phut_ava_speech(engine_name, ava_segment, report_dir):
    """Chất lượng + toàn vẹn byte + sự kiện của 1 engine trên 10 phút audio có nhãn."""
    floors = FLOORS[engine_name]

    result = av.run_engine(engine_name, ava_segment.pcm)
    n_frames = len(ava_segment.pcm) * 1000 // (av.TARGET_SR * av.FRAME_MS)
    gt = av.labels_to_timeline(ava_segment.labels, n_frames)
    metrics = av.evaluate(result["pred"], gt)
    integrity = result["integrity"]

    # ── 1. Toàn vẹn tín hiệu (byte-exact trên 10 phút audio thật) ──────────────
    assert integrity["frames"] > 0, "engine không chuyển tiếp frame nào cho ASR"
    assert integrity["mismatches"] == 0, (
        f"{engine_name}: {integrity['mismatches']}/{integrity['frames']} frame KHÔNG khớp byte "
        f"gốc tại mốc thời gian của nó ⇒ audio bị biến đổi trên đường tới ASR"
    )

    # ── 2. Sự kiện START/END ──────────────────────────────────────────────────
    assert result["starts"] >= 1, "không phát hiện đoạn nói nào trong 10 phút"
    assert result["ends"] >= result["starts"] - 1, (
        f"{engine_name}: {result['starts']} START nhưng chỉ {result['ends']} END ⇒ câu bị treo"
    )

    # ── 3. Chất lượng so với nhãn người (AVA-Speech) ──────────────────────────
    assert metrics["recall"] >= floors["recall"], f"{engine_name}: recall {metrics}"
    assert metrics["precision"] >= floors["precision"], f"{engine_name}: precision {metrics}"
    assert metrics["f1"] >= floors["f1"], f"{engine_name}: f1 {metrics}"
    assert metrics["accuracy"] >= floors["accuracy"], f"{engine_name}: accuracy {metrics}"

    # ── 4. Độ trễ biên & chi phí ──────────────────────────────────────────────
    assert metrics["onset_median_ms"] is not None
    assert metrics["onset_median_ms"] <= floors["onset_median_ms_max"], (
        f"{engine_name}: độ trễ onset trung vị {metrics['onset_median_ms']} ms quá cao"
    )
    assert result["rtf"] <= floors["rtf_max"], (
        f"{engine_name}: RTF {result['rtf']:.3f} > {floors['rtf_max']} (không còn nhanh hơn thời gian thực đủ)"
    )

    _RESULTS[engine_name] = {
        "engine": engine_name,
        "hop_samples": result["hop_samples"],
        "pre_roll_frames": result["pre_roll_frames"],
        "starts": result["starts"],
        "ends": result["ends"],
        "elapsed_sec": round(result["elapsed_sec"], 1),
        "rtf": round(result["rtf"], 4),
        "integrity": integrity,
        "metrics": metrics,
        # Recall RIÊNG theo từng lớp nhãn: chỗ phân biệt "domain khó" với "tích hợp sai".
        "recall_by_class": _recall_by_class(result["pred"], ava_segment.labels, n_frames),
        "floors": {k: v for k, v in floors.items() if k != "measured"},
        "measured_note": floors["measured"],
    }

    print(
        f"\n[{engine_name}] hop={result['hop_samples']} mẫu · pre-roll tối đa "
        f"{result['pre_roll_frames']} frame · RTF={result['rtf']:.3f} "
        f"({result['elapsed_sec']:.0f}s cho 10 phút audio)\n"
        f"   byte-exact: {integrity['frames']} frame, {integrity['mismatches']} sai lệch\n"
        f"   P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
        f"acc={metrics['accuracy']:.3f} · segment {metrics['pred_segments']} (GT {metrics['gt_segments']}) · "
        f"onset trung vị {metrics['onset_median_ms']} ms\n"
        f"   recall theo lớp: {_RESULTS[engine_name]['recall_by_class']}"
    )


@pytest.mark.slow
def test_quet_threshold_firered_tren_cua_so_2_phut(ava_segment):
    """Bảng tham chiếu: ngưỡng `speech_threshold` ảnh hưởng thế nào (cửa sổ 2 phút đầu)."""
    sub = ava_segment.pcm[: 120 * av.TARGET_SR]
    n_frames = len(sub) * 1000 // (av.TARGET_SR * av.FRAME_MS)
    gt = av.labels_to_timeline(
        [(s, e, lab) for s, e, lab in ava_segment.labels if s < 120.0], n_frames
    )

    sweep = {}
    for threshold in (0.3, 0.4, 0.5, 0.6):
        r = av.run_engine("firered-vad", sub, threshold=threshold)
        sweep[str(threshold)] = av.evaluate(r["pred"], gt)

    _RESULTS["firered_threshold_sweep"] = sweep
    for thr, m in sweep.items():
        print(f"   speech_threshold={thr}: F1={m['f1']:.3f} R={m['recall']:.3f} P={m['precision']:.3f}")

    # Ngưỡng mặc định của dự án (0.4) phải nằm trong nhóm tốt nhất, không được tệ bất thường.
    best = max(sweep.values(), key=lambda m: m["f1"])
    assert sweep["0.4"]["f1"] >= best["f1"] - 0.05, (
        f"ngưỡng mặc định 0.4 (F1={sweep['0.4']['f1']:.3f}) kém xa mức tốt nhất "
        f"(F1={best['f1']:.3f}) trên cửa sổ 2 phút"
    )


@pytest.mark.slow
def test_xuat_bao_cao_ava_speech(ava_segment, report_dir):
    """Sinh `report/02_vad/ava_speech_report.md` + `.json` từ kết quả 3 engine."""
    assert set(_RESULTS) >= {"firered-vad", "silero-vad", "fsmn-vad"}, (
        "phải chạy test chất lượng của cả 3 engine trước khi xuất báo cáo"
    )

    out_dir = report_dir / "02_vad"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ava_speech.json"
    md_path = out_dir / "ava_speech_report.md"

    payload = {
        "dataset": "AVA-Speech",
        "video_id": av.VIDEO_ID,
        "segment": {
            "start_sec": av.SEGMENT_START_SEC,
            "duration_sec": round(ava_segment.duration_sec, 1),
            "wav": str(ava_segment.wav_path.relative_to(PROJECT_ROOT)),
            "labels": len(ava_segment.labels),
            "frame_ms": av.FRAME_MS,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engines": {k: v for k, v in _RESULTS.items() if k != "firered_threshold_sweep"},
        "firered_threshold_sweep": _RESULTS.get("firered_threshold_sweep", {}),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# VAD trên AVA-Speech — 10 phút có nhãn người",
        "",
        f"- **Video**: `{av.VIDEO_ID}` · cửa sổ **{int(av.SEGMENT_START_SEC)}–"
        f"{int(av.SEGMENT_START_SEC + av.SEGMENT_DUR_SEC)} s** (nhãn phủ liên tục)",
        f"- **Audio test**: `{payload['segment']['wav']}` "
        f"({payload['segment']['duration_sec']} s, 16 kHz mono Int16, kênh 0 như plugin)",
        f"- **Nhãn**: {payload['segment']['labels']} dòng CSV, so khớp ở độ phân giải "
        f"{av.FRAME_MS} ms (`SPEECH_*` = nói, `NO_SPEECH` = lặng)",
        f"- **Sinh lúc**: {payload['generated_at']}",
        "- **Pipeline**: `plugin → VAD → ASR`; VAD chỉ chuyển tiếp **byte nguyên bản** "
        "(kiểm tra `mismatches = 0` bên dưới).",
        "",
        "## 1. Kết quả từng engine (mặc định docs: `threshold=None`, `silence=None`)",
        "",
        "| Engine | Hop | Trần pre-roll | P | R | F1 | Accuracy | Segment (pred/GT) | Onset trung vị | RTF | Byte sai lệch |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for eng in ("firered-vad", "silero-vad", "fsmn-vad"):
        d = _RESULTS[eng]
        m = d["metrics"]
        lines.append(
            f"| `{eng}` | {d['hop_samples']} mẫu | {d['pre_roll_frames']} frame | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | **{m['f1']:.3f}** | {m['accuracy']:.3f} | "
            f"{m['pred_segments']} / {m['gt_segments']} | {m['onset_median_ms']} ms | "
            f"{d['rtf']:.3f} | {d['integrity']['mismatches']} |"
        )

    lines += [
        "",
        "## 2. Recall theo LỚP NHÃN — chỗ phân biệt \"domain khó\" với \"tích hợp sai\"",
        "",
        "`CLEAN_SPEECH` = speech đọc sạch · `SPEECH_WITH_NOISE` / `SPEECH_WITH_MUSIC` = phim có",
        "nhạc/nhiễu nền. Nếu cả 3 engine đều tốt ở `CLEAN_SPEECH` mà chênh nhau ở hai lớp sau thì",
        "khác biệt là ĐỘ BỀN VỚI NHIỄU của model, không phải lỗi cấu hình/tích hợp.",
        "",
        "| Engine | R clean | R noise | R music | Báo động giả trên NO_SPEECH |",
        "|---|---|---|---|---|",
    ]
    for eng in ("firered-vad", "silero-vad", "fsmn-vad"):
        rc = _RESULTS[eng]["recall_by_class"]
        fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(
            f"| `{eng}` | {fmt(rc['CLEAN_SPEECH'])} | {fmt(rc['SPEECH_WITH_NOISE'])} | "
            f"{fmt(rc['SPEECH_WITH_MUSIC'])} | {fmt(rc['false_alarm_on_no_speech'])} |"
        )
    lines += [
        "",
        "> ⚠️ **Không so trực tiếp với bảng của nhà cung cấp**: bảng FireRedVAD (F1 97,57 · Silero 95,95)",
        "> được đo **non-streaming trên FLEURS-VAD-102** — speech ĐỌC sạch, clip ~10 s, nhãn nhị phân",
        "> (`external/FireRedVAD/README.md`). AVA-Speech là audio PHIM (nhạc, hiệu ứng, nói chồng tiếng).",
        "> Xem `report/02_vad/ava_vs_vendor_benchmarks.md` để có bằng chứng đầy đủ.",
        "",
        "## 3. Ngưỡng hồi quy đã chốt (thấp hơn số đo để tránh dao động nhỏ)",
        "",
        "| Engine | F1 ≥ | Recall ≥ | Precision ≥ | Accuracy ≥ | Onset ≤ | RTF ≤ | Số đo lần này |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for eng in ("firered-vad", "silero-vad", "fsmn-vad"):
        d = _RESULTS[eng]
        f = d["floors"]
        lines.append(
            f"| `{eng}` | {f['f1']} | {f['recall']} | {f['precision']} | {f['accuracy']} | "
            f"{f['onset_median_ms_max']} ms | {f['rtf_max']} | {d['measured_note']} |"
        )

    sweep = _RESULTS.get("firered_threshold_sweep", {})
    if sweep:
        lines += [
            "",
            "## 4. Quét `speech_threshold` của FireRed (2 phút đầu)",
            "",
            "| speech_threshold | P | R | F1 | Accuracy |",
            "|---|---|---|---|---|",
        ]
        for thr, m in sweep.items():
            lines.append(
                f"| {thr} | {m['precision']:.3f} | {m['recall']:.3f} | **{m['f1']:.3f}** | {m['accuracy']:.3f} |"
            )

    lines += [
        "",
        "## 5. Ghi chú diễn giải",
        "",
        "- **FireRed** (mặc định của dự án) có F1 cao nhất và độ trễ onset trung vị ~100 ms;",
        "  số segment nhiều hơn GT vì AVA-Speech gộp các khoảng lặng ngắn, còn VAD cắt theo",
        "  `min_silence_frame = 20` frame (200 ms) đúng mặc định docs.",
        "- **Silero** rất thận trọng trên audio có nhạc/nhiễu nền của AVA (precision cao, recall",
        "  thấp): muốn nhạy hơn thì hạ `threshold` hoặc tăng `min_silence_duration_ms` trong popup.",
        "- **FSMN** ở giữa; độ trễ onset trung vị lớn hơn do `lookback_time_start_point` +",
        "  `window_size_ms` và bước nhảy 60 ms.",
        "- **Toàn vẹn tín hiệu**: mọi frame chuyển tiếp khớp **byte-exact** PCM gốc (0 sai lệch)",
        "  ⇒ VAD không resample/gain/clip trên đường tới ASR.",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert md_path.exists() and json_path.exists()

    print(f"\n📊 Đã ghi báo cáo: {md_path} và {json_path}")
