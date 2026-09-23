#!/usr/bin/env python3
"""Chuyển một finetune HuggingFace thành GGUF cho transcribe.cpp rồi đăng ký vào catalog.

Quy trình (đúng theo kiến trúc của transcribe.cpp: Python CHỈ sinh GGUF ở dtype
tham chiếu F32/F16/BF16, còn block-quantization là việc của binary C++
`transcribe-quantize`):

  1. Tải (hoặc dùng lại) snapshot HF vào ``backend/models/_hf/<org>__<name>/``.
  2. Gọi converter của family: ``external/transcribe.cpp/scripts/convert-<family>.py``.
  3. Đọc ``general.file_type`` của GGUF vừa sinh để biết tier tham chiếu thực tế.
  4. Nếu ``--quant`` khác tier đó thì quantize bằng ``transcribe-quantize``.
  5. Đặt file kết quả vào ``backend/models/<model-key>-<QUANT>.gguf``.
  6. Cập nhật ``backend/models.yaml`` (giữ nguyên định dạng + comment của các entry
     khác) rồi nạp lại ``ModelRegistry`` để xác nhận YAML vẫn hợp lệ.
  7. Nạp GGUF qua binding đúng như engine ASR làm và in transcript để biết bản build
     có chạy thật hay không.

Ví dụ:
    .venv\\Scripts\\python.exe -m backend.tools.build_finetune ^
        --hf-repo kotoba-tech/kotoba-whisper-v2.2 --family whisper ^
        --quant Q8_0 --model-key kotoba-whisper-v2.2 --vram-estimate-mb 1500

Idempotent: chạy lại cùng tham số không tải lại, không build lại, không sinh entry
trùng và không sửa `models.yaml`. Entry đã có sẽ được cập nhật tại chỗ; các field
tuỳ chọn không truyền trên CLI được giữ nguyên từ entry cũ (xem `keep()` trong `main`).

Phụ thuộc (interpreter chạy script, trong repo là ``.venv``):
``gguf``, ``torch``, ``transformers``, ``safetensors``, ``huggingface_hub``
(cho converter) và ``soundfile``, ``numpy``, ``yaml`` (cho bước kiểm tra).
RIÊNG ``transcribe-quantize`` là binary C++ phải build trước::

    cmake -S external/transcribe.cpp -B <build-dir> -DTRANSCRIBE_BUILD_TOOLS=ON
    cmake --build <build-dir> --target transcribe-quantize

Script tự dò binary trong ``bin/``, ``external/build-cuda/bin/``,
``.build-cuda/bin/`` và ``external/transcribe.cpp/build/bin/``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
MODELS_DIR = BACKEND_DIR / "models"
MODELS_YAML = BACKEND_DIR / "models.yaml"
CONVERTER_DIR = PROJECT_ROOT / "external" / "transcribe.cpp" / "scripts"
HF_SNAPSHOT_DIR = MODELS_DIR / "_hf"

#: Preset mà `transcribe-quantize` chấp nhận (tools/transcribe-quantize/policy.cpp).
QUANT_PRESETS = ("F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
DEFAULT_QUANT = "Q8_0"

#: Tier tham chiếu mà các converter Python có thể sinh ra.
REFERENCE_TIERS = ("F32", "F16", "BF16")

#: Các vị trí có thể có binary quantizer (đường dẫn junction trùng nhau sẽ bị lọc).
QUANTIZE_BIN_CANDIDATES = (
    PROJECT_ROOT / "bin" / "transcribe-quantize.exe",
    PROJECT_ROOT / "external" / "build-cuda" / "bin" / "transcribe-quantize.exe",
    PROJECT_ROOT / ".build-cuda" / "bin" / "transcribe-quantize.exe",
    PROJECT_ROOT / "external" / "transcribe.cpp" / "build" / "bin" / "transcribe-quantize.exe",
)

#: Hình học encoder của Whisper -> slug variant mà converter biết (VARIANT_DISPLAY_NAMES).
_WHISPER_GEOMETRY: Dict[Tuple[int, int, int], str] = {
    (384, 4, 80): "whisper-tiny",
    (512, 6, 80): "whisper-base",
    (768, 12, 80): "whisper-small",
    (1024, 24, 80): "whisper-medium",
    (1280, 32, 80): "whisper-large-v2",
    (1280, 32, 128): "whisper-large-v3",
}

_PLAIN_SCALAR_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./+-]*$")
_MODELS_LINE_RE = re.compile(r"^models:\s*(#.*)?$")
_ENTRY_LINE_RE = re.compile(r"^  (\S[^:]*):\s*(#.*)?$")


def log(message: str) -> None:
    """In log có tiền tố để dễ lọc trong output dài."""
    print(f"[build_finetune] {message}", flush=True)


class BuildError(RuntimeError):
    """Lỗi cấu hình/môi trường — thông báo phải đủ rõ để người dùng tự sửa."""


# --------------------------------------------------------------------------- helpers


def normalize_family(name: str) -> str:
    """`Cohere-Transcribe` -> `cohere_transcribe` (khoá tra converter)."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def discover_converters(scripts_dir: Path = CONVERTER_DIR) -> Dict[str, Path]:
    """Quét `scripts/convert-*.py` -> {family: path}. Không hardcode danh sách."""
    found: Dict[str, Path] = {}
    for path in sorted(scripts_dir.glob("convert-*.py")):
        found[normalize_family(path.stem[len("convert-"):])] = path
    return found


def find_quantize_bin(explicit: Optional[Path] = None) -> Optional[Path]:
    """Tìm binary `transcribe-quantize`; None nếu chưa build."""
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit if explicit.is_absolute() else PROJECT_ROOT / explicit)
    candidates.extend(QUANTIZE_BIN_CANDIDATES)
    seen: set[str] = set()
    for cand in candidates:
        try:
            key = str(cand.resolve()).lower()  # .resolve() đi qua junction
        except OSError:
            key = str(cand).lower()
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return cand
    return None


def read_reference_tier(gguf_path: Path) -> str:
    """Đọc `general.file_type` của GGUF -> 'F32' | 'F16' | 'BF16'."""
    from gguf import GGUFReader, LlamaFileType

    reader = GGUFReader(str(gguf_path))
    field = reader.fields.get("general.file_type")
    if field is None or not field.parts:
        raise BuildError(f"GGUF thiếu KV 'general.file_type': {gguf_path}")
    value = int(field.parts[field.data[0]][0])
    try:
        name = LlamaFileType(value).name
    except ValueError as exc:  # pragma: no cover - GGUF hỏng
        raise BuildError(f"general.file_type={value} không nhận dạng được") from exc
    tier = name.removeprefix("MOSTLY_").removeprefix("ALL_")
    if tier not in REFERENCE_TIERS:
        raise BuildError(f"converter sinh dtype {name!r}, không phải tier tham chiếu")
    return tier


def infer_variant(family: str, model_dir: Path) -> Optional[str]:
    """Suy ra `stt.variant` từ config của model (hiện chỉ Whisper cần).

    Finetune Whisper bắt buộc phải khai một slug có trong VARIANT_DISPLAY_NAMES của
    `convert-whisper.py`; slug này chỉ mang tính mô tả (loader không đổi hành vi theo
    nó) nhưng nó phải tồn tại nên ta suy ra từ hình học encoder + số mel bins.
    """
    if family != "whisper":
        return None
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    try:
        key = (int(cfg["d_model"]), int(cfg["encoder_layers"]), int(cfg["num_mel_bins"]))
        dec_layers = int(cfg.get("decoder_layers", 0))
    except (KeyError, TypeError, ValueError):
        return None
    slug = _WHISPER_GEOMETRY.get(key)
    if slug is None:
        return None
    if slug in ("whisper-large-v2", "whisper-large-v3") and 0 < dec_layers <= 8:
        slug = "whisper-large-v3-turbo"  # decoder đã distill (large-v3-turbo)
    gen_path = model_dir / "generation_config.json"
    if gen_path.is_file() and not json.loads(gen_path.read_text(encoding="utf-8")).get("lang_to_id"):
        slug += ".en"  # bản English-only (vocab không có token ngôn ngữ)
    return slug


def read_wav_16k_mono(path: Path) -> Any:
    """Đọc wav -> float32 mono 16 kHz (đúng định dạng `session.run` yêu cầu)."""
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(sample_rate) != 16000:
        raise BuildError(f"{path}: cần wav 16 kHz, file này {sample_rate} Hz")
    return np.ascontiguousarray(data.mean(axis=1, dtype=np.float32))


def default_validation_wav() -> Optional[Path]:
    """WAV ngắn có sẵn trong repo để smoke-test (FLEURS ja_jp nếu có)."""
    for pattern in ("wav_test/google_fleurs/ja_jp/test/*.wav", "wav_test/**/*.wav", "wav_test/*.wav"):
        found = sorted(PROJECT_ROOT.glob(pattern))
        if found:
            return found[0]
    return None


# --------------------------------------------------------------------------- converter


@dataclass(frozen=True)
class ConverterCli:
    """Hình dạng CLI của converter — các family không thống nhất tham số output."""

    path: Path
    out_style: str  # "positional" | "outdir" | "out"
    has_repo_id: bool
    has_revision: bool
    has_variant: bool


def probe_converter(path: Path) -> ConverterCli:
    """Suy ra CLI của converter từ chính source của nó (không chạy --help cho nhanh)."""
    text = path.read_text(encoding="utf-8")
    if "--outdir" in text:
        out_style = "outdir"
    elif '"--out"' in text or "'--out'" in text:
        out_style = "out"
    else:
        out_style = "positional"
    return ConverterCli(
        path=path,
        out_style=out_style,
        has_repo_id='"--repo-id"' in text,
        has_revision='"--revision"' in text,
        has_variant='"--variant"' in text,
    )


def converter_command(
    cli: ConverterCli,
    model_dir: Path,
    repo_id: str,
    revision: Optional[str],
    variant: Optional[str],
    out_target: Path,
) -> List[str]:
    """Dựng argv cho converter (chạy qua `sys.executable`, không dùng shell)."""
    cmd = [str(cli.path), str(model_dir)]
    if cli.out_style == "positional":
        cmd.append(str(out_target))
    elif cli.out_style == "out":
        cmd += ["--out", str(out_target)]
    if cli.has_repo_id:
        cmd += ["--repo-id", repo_id]
    if cli.has_revision and revision:
        cmd += ["--revision", revision]
    if cli.has_variant:
        if not variant:
            raise BuildError(
                f"{cli.path.name} cần --variant nhưng không suy ra được từ config; "
                f"hãy truyền --variant thủ công"
            )
        cmd += ["--variant", variant]
    return cmd


# --------------------------------------------------------------------------- models.yaml


def yaml_scalar(value: Any) -> str:
    """Ghi scalar theo kiểu models.yaml (plain nếu an toàn, còn lại single-quote)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if _PLAIN_SCALAR_RE.match(text):
        return text
    return "'" + text.replace("'", "''") + "'"


def entry_block(model_key: str, fields: Dict[str, Any]) -> List[str]:
    """Sinh block YAML của một entry, đúng thứ tự field và thụt lề 2/4 space."""
    lines = [f"  {model_key}:\n"]
    for name, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            lines.append(f"    {name}:\n")
            lines.extend(f"    - {yaml_scalar(item)}\n" for item in value)
        else:
            lines.append(f"    {name}: {yaml_scalar(value)}\n")
    return lines


def models_section(lines: Sequence[str], yaml_path: Path) -> Tuple[int, int, Dict[str, int]]:
    """Định vị `models:` -> (index dòng models, index kết thúc, {key: index bắt đầu})."""
    models_idx = next(
        (i for i, line in enumerate(lines) if _MODELS_LINE_RE.match(line.rstrip("\r\n"))),
        None,
    )
    if models_idx is None:
        raise BuildError(f"{yaml_path}: không tìm thấy khoá 'models:' ở cấp cao nhất")
    block_end = models_idx + 1
    while block_end < len(lines) and (lines[block_end].strip() == "" or lines[block_end][:1] in " \t"):
        block_end += 1
    starts: Dict[str, int] = {}
    for i in range(models_idx + 1, block_end):
        match = _ENTRY_LINE_RE.match(lines[i].rstrip("\r\n"))
        if match:
            starts[match.group(1)] = i
    return models_idx, block_end, starts


def existing_entry_fields(yaml_path: Path, model_key: str) -> Dict[str, Any]:
    """Đọc entry đã có (nếu có) để giữ lại các field tuỳ chọn khi ghi đè."""
    import yaml

    with yaml_path.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines(keepends=True)
    _, block_end, starts = models_section(lines, yaml_path)
    if model_key not in starts:
        return {}
    start = starts[model_key]
    end = next((i for i in sorted(starts.values()) if i > start), block_end)
    block_text = "".join(line[2:] for line in lines[start:end])
    loaded = yaml.safe_load(block_text) or {}
    return loaded.get(model_key) or {}


def upsert_models_yaml(yaml_path: Path, model_key: str, block: Sequence[str]) -> bool:
    """Thêm/cập nhật entry trong `models:`, giữ nguyên phần còn lại của file.

    Trả về True nếu file thực sự thay đổi (idempotent: chạy lại không đổi gì).
    """
    with yaml_path.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines(keepends=True)

    _, block_end, starts = models_section(lines, yaml_path)
    new_block = list(block)
    if model_key in starts:
        start = starts[model_key]
        end = next((i for i in sorted(starts.values()) if i > start), block_end)
        if lines[start:end] == new_block:
            return False
        lines[start:end] = new_block
    else:
        if block_end > 0 and not lines[block_end - 1].endswith("\n"):
            lines[block_end - 1] += "\n"
        lines[block_end:block_end] = new_block

    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    with yaml_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("".join(lines))
    return True


def verify_registry(model_key: str) -> Dict[str, Any]:
    """Nạp lại models.yaml qua ModelRegistry và trả info của entry vừa ghi."""
    from backend.asr.registry import ModelRegistry

    registry = ModelRegistry(yaml_path=str(MODELS_YAML))
    info = registry.get_model_info(model_key)
    if not info:
        raise BuildError(f"ModelRegistry không thấy entry '{model_key}' sau khi ghi YAML")
    path = registry.resolve_model_path(model_key)
    if not Path(path).is_file():
        raise BuildError(f"ModelRegistry trỏ tới file không tồn tại: {path}")
    return info


# --------------------------------------------------------------------------- validation


def validate_gguf(gguf_path: Path, family: str, wav: Path, language: str) -> Dict[str, Any]:
    """Nạp GGUF qua binding + chạy 1 câu audio, giống hệt đường đi của engine."""
    import backend.asr  # noqa: F401  (bootstrap bin/ TRƯỚC khi import transcribe_cpp)
    import transcribe_cpp
    from backend.asr.adapters import build_family_options
    from backend.asr.native import resolve_backend

    info = {"family": family}
    backend_name = resolve_backend(None)
    pcm = read_wav_16k_mono(wav)

    model = transcribe_cpp.Model(str(gguf_path), backend=backend_name)
    try:
        session = model.session(n_threads=0)
        try:
            result = session.run(
                pcm,
                language=language,
                family=build_family_options(family, info, model=model, slot="run"),
                timestamps="segment",
            )
        finally:
            session.close()
    finally:
        model.close()

    return {
        "backend": backend_name,
        "text": result.text,
        "language": result.language,
        "segments": [(s.t0_ms, s.t1_ms, s.text) for s in result.segments],
    }


# --------------------------------------------------------------------------- main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m backend.tools.build_finetune",
        description="Chuyển finetune HuggingFace sang GGUF cho transcribe.cpp và đăng ký vào models.yaml.",
    )
    parser.add_argument("--hf-repo", required=True, help="repo HuggingFace của finetune, vd kotoba-tech/kotoba-whisper-v2.2")
    parser.add_argument("--family", required=True, help="family transcribe.cpp (whisper, parakeet, qwen3_asr, ...)")
    parser.add_argument("--model-key", required=True, help="khoá trong models.yaml")
    parser.add_argument("--quant", default=DEFAULT_QUANT, help=f"preset quantize (mặc định {DEFAULT_QUANT}): {', '.join(QUANT_PRESETS)}")
    parser.add_argument("--name", default=None, help="tên hiển thị (mặc định '<model-key> (<QUANT>)')")
    parser.add_argument("--local-dir", type=Path, default=None, help="thư mục snapshot HF có sẵn (bỏ qua bước tải)")
    parser.add_argument("--revision", default=None, help="revision HF để ghim (khuyến nghị)")
    parser.add_argument("--variant", default=None, help="stt.variant truyền cho converter (mặc định: suy ra từ config)")
    parser.add_argument("--architecture-type", default=None, help="trường architecture_type (mặc định: offline_llm)")
    parser.add_argument("--sample-rate", type=int, default=None, help="trường sample_rate (mặc định: 16000)")
    parser.add_argument("--languages", default=None, help="'auto' hoặc danh sách cách nhau bởi dấu phẩy, vd 'ja,en' (mặc định: auto)")
    parser.add_argument("--vram-estimate-mb", type=int, default=None, help="trường vram_estimate_mb (bỏ qua nếu không truyền)")
    parser.add_argument("--context-window-sec", type=int, default=None, help="trường context_window_sec (bỏ qua nếu không truyền)")
    parser.add_argument("--description", default=None, help="trường description (mặc định: tự sinh)")
    parser.add_argument("--python", type=Path, default=None, help="interpreter chạy converter (mặc định: interpreter hiện tại)")
    parser.add_argument("--quantize-bin", type=Path, default=None, help="đường dẫn transcribe-quantize (mặc định: tự dò)")
    parser.add_argument("--force", action="store_true", help="build lại kể cả khi file đích đã có")
    parser.add_argument("--skip-validate", action="store_true", help="bỏ bước nạp binding + chạy thử audio")
    parser.add_argument("--wav", type=Path, default=None, help="wav 16 kHz dùng để validate (mặc định: FLEURS ja_jp trong wav_test/)")
    parser.add_argument("--language", default="ja", help="ngôn ngữ cho bước validate (mặc định ja)")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in kế hoạch, không ghi gì lên đĩa")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)

    family = normalize_family(args.family)
    quant = args.quant.strip().upper()
    # Khoá catalog là chữ thường (ModelRegistry so khớp ở dạng lowercase).
    model_key = args.model_key.strip().lower()
    if not model_key:
        raise BuildError("--model-key rỗng")
    if quant not in QUANT_PRESETS:
        raise BuildError(f"--quant {args.quant!r} không hợp lệ; chọn một trong: {', '.join(QUANT_PRESETS)}")

    converters = discover_converters()
    converter = converters.get(family)
    if converter is None:
        raise BuildError(
            f"family {args.family!r} không có converter trong {CONVERTER_DIR}. "
            f"Hiện có: {', '.join(sorted(converters))}"
        )
    cli = probe_converter(converter)

    slug = args.hf_repo.strip().strip("/").rsplit("/", 1)[-1]
    snapshot_dir = args.local_dir if args.local_dir else HF_SNAPSHOT_DIR / args.hf_repo.replace("/", "__")
    snapshot_dir = snapshot_dir.resolve() if snapshot_dir.is_absolute() else (PROJECT_ROOT / snapshot_dir)
    build_dir = snapshot_dir / "_build"
    final_path = MODELS_DIR / f"{model_key}-{quant}.gguf"

    snapshot_ready = (snapshot_dir / "config.json").is_file() and any(snapshot_dir.glob("*.safetensors"))
    variant = args.variant or (infer_variant(family, snapshot_dir) if snapshot_ready else None)
    quantize_bin = find_quantize_bin(args.quantize_bin)
    need_build = args.force or not final_path.is_file()

    if args.dry_run:
        log("KẾ HOẠCH (dry-run, không ghi gì lên đĩa)")
        log(f"  repo        : {args.hf_repo}" + (f"@{args.revision}" if args.revision else ""))
        log(f"  family      : {family} -> {converter.relative_to(PROJECT_ROOT)} (CLI: {cli.out_style})")
        log(f"  snapshot    : {snapshot_dir.relative_to(PROJECT_ROOT)} "
            f"({'có sẵn' if snapshot_ready else 'sẽ tải' if not args.local_dir else 'THIẾU'})")
        log(f"  variant     : {variant or '(sẽ suy ra từ config sau khi có snapshot)'}")
        log(f"  build dir   : {build_dir.relative_to(PROJECT_ROOT)}")
        log(f"  quantize    : {quantize_bin if quantize_bin else 'KHÔNG TÌM THẤY transcribe-quantize'}"
            f" (preset {quant}; tự bỏ qua nếu GGUF đã ở đúng tier)")
        log(f"  output      : {final_path.relative_to(PROJECT_ROOT)}"
            f"{' (đã có, sẽ bỏ qua build)' if final_path.is_file() and not args.force else ''}")
        entry_present = re.search(rf"^  {re.escape(model_key)}:", MODELS_YAML.read_text(encoding="utf-8"), re.M)
        log(f"  models.yaml : {'cập nhật' if entry_present else 'thêm'} entry '{model_key}'")
        if not args.skip_validate:
            wav = args.wav or default_validation_wav()
            log(f"  validate    : {wav if wav else 'KHÔNG có wav'} (language={args.language})")
        return 0

    # ---- 1+2+3+4+5: build
    if not need_build:
        log(f"Đã có {final_path.name} — bỏ qua build (dùng --force để build lại)")
    else:
        if args.local_dir and not snapshot_ready:
            raise BuildError(f"--local-dir {snapshot_dir} không có config.json + *.safetensors")
        if not snapshot_ready:
            from huggingface_hub import snapshot_download

            log(f"Tải {args.hf_repo} -> {snapshot_dir.relative_to(PROJECT_ROOT)}")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=args.hf_repo,
                revision=args.revision,
                local_dir=str(snapshot_dir),
                ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot"],
            )
        else:
            log(f"Dùng lại snapshot có sẵn: {snapshot_dir.relative_to(PROJECT_ROOT)}")

        if variant is None:
            variant = infer_variant(family, snapshot_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        produced = build_dir / f"{slug}-REF.gguf"
        cmd = converter_command(cli, snapshot_dir, args.hf_repo, args.revision, variant, produced if cli.out_style != "outdir" else build_dir)
        log("Converter: " + " ".join(str(part) for part in cmd))
        python = args.python or Path(sys.executable)
        exit_code = subprocess.call([str(python), *cmd])
        if exit_code != 0:
            raise BuildError(f"converter thoát với mã {exit_code}")

        if cli.out_style == "outdir":
            written = sorted(build_dir.glob("*.gguf"), key=lambda p: p.stat().st_mtime)
            if not written:
                raise BuildError(f"converter không sinh file .gguf nào trong {build_dir}")
            produced = written[-1]

        tier = read_reference_tier(produced)
        log(f"Converter sinh {produced.name} ở dtype tham chiếu {tier}")

        if quant == tier:
            log(f"--quant {quant} trùng tier tham chiếu — không cần quantize")
            source = produced
        else:
            if quantize_bin is None:
                raise BuildError(
                    "Không tìm thấy binary transcribe-quantize. Build trước:\n"
                    "  cmake -S external/transcribe.cpp -B .build-cuda -DTRANSCRIBE_BUILD_TOOLS=ON\n"
                    "  cmake --build .build-cuda --target transcribe-quantize"
                )
            source = build_dir / f"{slug}-{quant}.gguf"
            log(f"Quantize: {produced.name} -> {source.name} (preset {quant})")
            exit_code = subprocess.call([str(quantize_bin), str(produced), str(source), "--quant", quant])
            if exit_code != 0:
                raise BuildError(f"transcribe-quantize thoát với mã {exit_code}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            final_path.unlink()
        source.replace(final_path)
        log(f"File đích: {final_path.relative_to(PROJECT_ROOT)} ({final_path.stat().st_size / 1024**2:.1f} MB)")

    # ---- 6: models.yaml + ModelRegistry
    # Field luôn ghi lại theo CLI: family / hf_repo / default_quant / file.
    # Field tuỳ chọn: CLI > entry cũ (nếu đã tồn tại) > mặc định — nhờ vậy chạy lại
    # với ít cờ hơn KHÔNG âm thầm xoá bớt thông tin của entry đang có.
    previous = existing_entry_fields(MODELS_YAML, model_key)

    def keep(name: str, cli_value: Any, fallback: Any) -> Any:
        return cli_value if cli_value is not None else (previous.get(name, fallback))

    languages = keep("languages", args.languages, "auto")
    if isinstance(languages, str):
        languages = "auto" if languages.strip().lower() == "auto" else [
            item.strip() for item in languages.split(",") if item.strip()
        ]
    fields: Dict[str, Any] = {
        "name": keep("name", args.name, f"{model_key} ({quant})"),
        "family": family,
        "architecture_type": keep("architecture_type", args.architecture_type, "offline_llm"),
        "hf_repo": args.hf_repo,
        "default_quant": quant,
        "file": final_path.name,
        "sample_rate": keep("sample_rate", args.sample_rate, 16000),
        "languages": languages,
        "vram_estimate_mb": keep("vram_estimate_mb", args.vram_estimate_mb, None),
        "context_window_sec": keep("context_window_sec", args.context_window_sec, None),
        "description": keep(
            "description",
            args.description,
            f"{model_key}: finetune {args.hf_repo} chuyển sang GGUF ({quant}) bằng "
            f"backend/tools/build_finetune.py",
        ),
    }
    block = entry_block(model_key, fields)
    changed = upsert_models_yaml(MODELS_YAML, model_key, block)
    log(f"models.yaml: {'đã cập nhật' if changed else 'không đổi'} entry '{model_key}'")

    info = verify_registry(model_key)
    log(f"ModelRegistry OK: {model_key} -> {info.get('file')} "
        f"(family={info.get('family')}, quant={info.get('default_quant')})")

    # ---- 7: nạp qua binding như engine
    if not args.skip_validate:
        wav = (args.wav or default_validation_wav())
        if wav is None:
            raise BuildError("Không có wav để validate; truyền --wav hoặc --skip-validate")
        if not wav.is_file():
            raise BuildError(f"--wav {wav} không tồn tại")
        log(f"Validate: {final_path.name} + {wav.name} (language={args.language})")
        result = validate_gguf(final_path, family, wav, args.language)
        log(f"  backend  : {result['backend']}")
        log(f"  language : {result['language']}")
        log(f"  segments : {len(result['segments'])}")
        for t0, t1, text in result["segments"][:5]:
            log(f"    [{t0:>6} -> {t1:>6} ms] {text}")
        log(f"  TEXT     : {result['text']}")

    log(f"Xong. Tổng thời gian: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        print(f"[build_finetune] LỖI: {error}", file=sys.stderr)
        raise SystemExit(2)
