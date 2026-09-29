# Qwen3-ASR (`qwen_asr`) — Evidence-Based Technical Report

**Repo analyzed:** `D:\vibe-translation-addon-transcribe_cpp\qwen3-asr\Qwen3-ASR`
**Package:** `qwen-asr` version `0.0.6` (`qwen3-asr/Qwen3-ASR/pyproject.toml:7`)
**Method:** direct `read` of source files. Claims are cited as `path:line`. Anything not found is marked **not present in repo**.

---

## 1. Public API

### 1.1 Package exports

`qwen3-asr/Qwen3-ASR/qwen_asr/__init__.py:20-25`

```python
from .inference.qwen3_asr import Qwen3ASRModel
from .inference.qwen3_forced_aligner import Qwen3ForcedAligner

from .inference.utils import parse_asr_output

__all__ = ["__version__"]
```

**Two concrete observations:**
1. `__all__` lists only `"__version__"`, yet `Qwen3ASRModel`, `Qwen3ForcedAligner` and `parse_asr_output` are the actually importable names (used by every example, e.g. `from qwen_asr import Qwen3ASRModel`).
2. `__version__` is **never defined anywhere** in the package (grep for `__version__` across `qwen_asr/` matches only that single `__all__` line). So `__all__` is stale/incorrect; `from qwen_asr import *` yields nothing useful.

### 1.2 `Qwen3ASRModel.from_pretrained(...)` — exact signature

`qwen3-asr/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:175-224`

```python
@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    forced_aligner: Optional[str] = None,
    forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
    max_inference_batch_size: int = 32,
    max_new_tokens: Optional[int] = 512,
    **kwargs,
) -> "Qwen3ASRModel":
```

- `**kwargs` are forwarded verbatim to `AutoModel.from_pretrained(...)` (line 206), i.e. `dtype`, `device_map`, `attn_implementation`, etc.
- This builds the **transformers** backend. Body (lines 206-224): `AutoModel.from_pretrained`, then `AutoProcessor.from_pretrained(..., fix_mistral_regex=True)`, then optionally `Qwen3ForcedAligner.from_pretrained(forced_aligner, **(forced_aligner_kwargs or {}))`, then `cls(backend="transformers", ...)`.

### 1.3 The second constructor: `Qwen3ASRModel.LLM(...)` (vLLM)

`qwen3-asr/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:226-288`

```python
@classmethod
def LLM(
    cls,
    model: str,
    forced_aligner: Optional[str] = None,
    forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
    max_inference_batch_size: int = -1,
    max_new_tokens: Optional[int] = 4096,
    **kwargs,
) -> "Qwen3ASRModel":
```

- `**kwargs` go to `vllm.LLM(...)` (line 269).
- Raises `ImportError("vLLM is not available. Install with: pip install qwen-asr[vllm]")` if vLLM is missing (lines 264-267).
- Sampling is fixed greedy: `SamplingParams(**({"temperature": 0.0, "max_tokens": max_new_tokens}))` (line 272).
- **Important:** `LLM()` stores `max_new_tokens=None` on the instance (line 287) and reads the real cap from `sampling_params`.

### 1.4 `transcribe(...)` — exact signature

`qwen3-asr/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:299-306`

```python
@torch.no_grad()
def transcribe(
    self,
    audio: Union[AudioLike, List[AudioLike]],
    context: Union[str, List[str]] = "",
    language: Optional[Union[str, List[Optional[str]]]] = None,
    return_time_stamps: bool = False,
) -> List[ASRTranscription]:
```

- Only **four** parameters — no `**kwargs`, no callback, no `stream=`, no `chunk_size=`, no `prompt=`, no `hotwords=`.
- `AudioLike` (`qwen_asr/inference/utils.py:27-30`):

```python
AudioLike = Union[
    str,                      # wav path / URL / base64
    Tuple[np.ndarray, int],   # (waveform, sr)
]
```

- Raises `ValueError` if `return_time_stamps=True` without a `forced_aligner` (line 336), on batch-size mismatch of `context`/`language` (lines 345, 355), and on unsupported language (via `validate_language`, line 363).

### 1.5 Returned objects

`qwen3-asr/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:57-74`

```python
@dataclass
class ASRTranscription:
    """
    ...
        language (str):
            Merged language string for the sample, e.g. "Chinese" or "Chinese,English".
            Empty string if unknown or silent audio.
        text (str):
            Transcribed text.
        time_stamps (Optional[Any]):
            Forced aligner output (ForcedAlignResult).
            Present only when return_time_stamps=True.
    """
    language: str
    text: str
    time_stamps: Optional[Any] = None
```

So: **`List[ASRTranscription]`, one element per input audio** (not per chunk). `text` is the plain string; `language` is a comma-joined merged label; `time_stamps` is `None` unless requested.

`time_stamps` is a `ForcedAlignResult` (`qwen_asr/inference/qwen3_forced_aligner.py:288-306`) whose `.items` is `List[ForcedAlignItem]` with `.text`, `.start_time`, `.end_time` **in seconds**.

Timestamp merging across long-audio chunks is done by `_offset_align_result` (line 539) + `_merge_align_results` (line 563): offsets are added and items concatenated.

**Two annotation bugs worth knowing:**
- `ForcedAlignItem` declares `start_time: int` / `end_time: int` (`qwen3_forced_aligner.py:283-285`) but `_to_structured_items` actually passes `float(it.get("start_time", 0))` (lines 386-389). The runtime values are floats in seconds.
- `_offset_align_result` (line 539-561) reconstructs items with `type(it)(text=..., start_time=..., end_time=...)` — this only works because it assumes the upstream item type accepts those kwargs; it explicitly documents the frozen-dataclass assumption in its docstring (line 545).

### 1.6 Helper: `parse_asr_output`

`qwen3-asr/Qwen3-ASR/qwen_asr/inference/utils.py:403-470`, exported at package level.

```python
def parse_asr_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
```
Returns `(language, text)`. Documented cases (lines 410-414): `"language Chinese<asr_text>...."`, newline variant, no-tag (whole string is text, language `""`), and `"language None<asr_text>"` → `("", "")` for empty audio.

---

## 2. Streaming / incremental-audio inference

### 2.1 Verdict: YES, true incremental-audio streaming exists — but vLLM-only

There **is** real chunk-at-a-time audio streaming with an updated hypothesis after each audio chunk. It is implemented as three methods on `Qwen3ASRModel`, `qwen3-asr/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:584-830`.

**State object** (lines 77-128):

```python
@dataclass
class ASRStreamingState:
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int

    chunk_id: int
    buffer: np.ndarray
    audio_accum: np.ndarray

    prompt_raw: str
    context: str
    force_language: Optional[str]

    language: str
    text: str
    _raw_decoded: str
```

**API #1 — initialize** (line 584):

```python
def init_streaming_state(
    self,
    context: str = "",
    language: Optional[str] = None,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    chunk_size_sec: float = 2.0,
) -> ASRStreamingState:
```
Raises `ValueError` if `self.backend != "vllm"` (line 625-626) or `chunk_size_sec <= 0` (line 627-628).

**API #2 — feed audio** (line 657):

```python
def streaming_transcribe(self, pcm16k: np.ndarray, state: ASRStreamingState) -> ASRStreamingState:
```
Docstring (lines 661-668): *"accepts an arbitrary-length 16k PCM float numpy array (mono). It buffers incoming samples, and whenever enough samples are accumulated to form one full chunk (chunk_size_sec), it runs one incremental decode step and updates: `state.language`, `state.text`."* Accepts `float32/float64/int16`; `int16` is scaled by `/32768.0` (lines 710-713).

**API #3 — flush tail** (line 767):

```python
def finish_streaming_transcribe(self, state: ASRStreamingState) -> ASRStreamingState:
```
Sends the remaining sub-chunk tail without padding and updates `state` one final time (lines 796-829).

### 2.2 What the "streaming" actually is (precisely)

From the loop body, `qwen3-asr/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:719-765`:

```python
while state.buffer.shape[0] >= state.chunk_size_samples:
    chunk = state.buffer[: state.chunk_size_samples]
    state.buffer = state.buffer[state.chunk_size_samples :]

    # Accumulate audio (re-feed from start, no padding)
    if state.audio_accum.shape[0] == 0:
        state.audio_accum = chunk
    else:
        state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)
```

…then (line 748-761):

```python
    prompt = state.prompt_raw + prefix

    # vLLM input: single item
    inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}

    outputs = self.model.generate([inp], sampling_params=self.sampling_params, use_tqdm=False)
    gen_text = outputs[0].outputs[0].text

    # Accumulate raw decoded (then parse to lang/text)
    state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text

    lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
    state.language = lang
    state.text = txt

    state.chunk_id += 1
```

Key semantics, stated exactly:

| Question | Answer (evidence) |
|---|---|
| Does it accept incrementally arriving audio (e.g. 1 s at a time)? | **Yes.** `streaming_transcribe` accepts arbitrary-length PCM and buffers internally; the example feeds 500/1000/2000/4000 ms steps (`examples/example_qwen3_asr_vllm_streaming.py:100`). |
| Does it emit partial hypotheses? | **Yes, but per-audio-chunk, not per-token.** Each time a full `chunk_size_sec` accumulates, one decode happens and the emitted `state.text` is the **entire cumulative transcript so far**, not a delta. |
| Is it token streaming of a completed utterance? | **No.** There is no `stream=True`, no `TextIteratorStreamer`, no token callback, no generator anywhere in `qwen_asr/` (grep for `stream` returns only the streaming-state API and this demo). Output arrives only as whole re-decoded strings. |
| Is it a true sliding-window / online encoder? | **No.** Line 724-728 re-feeds **all** audio from stream start on every chunk, and the unit of work is a fresh `generate()` on the full accumulation. There is no KV-cache reuse and no left-context truncation. Cost grows ~O(n²) in audio length. |
| Latency granularity | `chunk_size_sec` audio duration (default 2.0 s in the method; 1.0 s in the CLI demo), **plus** the full generation time for a re-decode of all audio so far. |

### 2.3 Prefix-prompt / rollback strategy

`qwen3_asr/inference/qwen3_asr.py:730-747`:

```python
# Build prefix with rollback strategy
prefix = ""
if state.chunk_id < state.unfixed_chunk_num:
    prefix = ""
else:
    cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
    k = int(state.unfixed_token_num)
    while True:
        end_idx = max(0, len(cur_ids) - k)
        prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
        if '\ufffd' not in prefix:
            break
        else:
            if end_idx == 0:
                prefix = ""
                break
            k += 1
```

- First `unfixed_chunk_num` chunks: no prefix (fresh decode).
- After that: the previous transcript is fed back as an assistant-prefix prompt, with the last `unfixed_token_num` tokens rolled back to reduce boundary jitter. The `\ufffd` check pushes the rollback further until the decoded prefix is not a broken UTF-8 fragment.
- `finish_streaming_transcribe` uses the same idea but a simpler rollback (line 809-816), with `end_idx = max(1, len(cur_ids) - unfixed_token_num)`.

### 2.4 Documented constraints

Docstrings are explicit (`qwen3_asr/inference/qwen3_asr.py:595-598`, `678-681`, `775-778`):
- `"Streaming ASR is supported ONLY for vLLM backend."`
- `"Streaming ASR does NOT support timestamps (forced aligner is not used)."`
- `"Batch inference is NOT supported."` / `"Single stream only (no batching)."`

README section `#### Streaming Inference` (`README.md:289-291`) repeats this: *"Currently, streaming inference is only available with the vLLM backend. Note that streaming inference does not support batch inference or returning timestamps."*

### 2.5 The two streaming artifacts

**`examples/example_qwen3_asr_vllm_streaming.py`** — file-based simulation, not live mic. Downloads one wav, resamples to 16k, then loops `for step_ms in [500, 1000, 2000, 4000]` (line 100) slicing the waveform into `step` samples and calling `asr.streaming_transcribe(seg, state)` (line 81). Model init (lines 90-94):

```python
asr = Qwen3ASRModel.LLM(
    model=ASR_MODEL_PATH,
    gpu_memory_utilization=0.8,
    max_new_tokens=32, # set a small value for streaming
)
```
Note `init_streaming_state(unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0)` at lines 69-73 — so the 500 ms and 1000 ms steps are **not** decoded every 500/1000 ms; they only fill the 2 s chunk buffer.

**`qwen_asr/cli/demo_streaming.py`** — a real live-mic Flask demo (registered as console script `qwen-asr-demo-streaming`, `pyproject.toml:47`).
- Browser: `getUserMedia` mono capture, ScriptProcessor(4096), linear resample to 16 kHz, `CHUNK_MS = 500` (line 222), pushes raw Float32 via `POST /api/chunk?session_id=...` with `Content-Type: application/octet-stream` (lines 275-283).
- Server: `np.frombuffer(raw, dtype=np.float32)` then `asr.streaming_transcribe(wav, s.state)` (lines 444-446).
- Defaults (lines 479-481): `--unfixed-chunk-num 4`, `--unfixed-token-num 5`, `--chunk-size-sec 1.0`; model is loaded with `max_new_tokens=32` (line 500).
- Session TTL 10 min (`SESSION_TTL_SEC = 10 * 60`, line 53); sessions are GC'd and force-finished via `finish_streaming_transcribe` (lines 56-64).
- **Bug:** the browser declares `const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000));` inside `btnStart.onclick` (line 330) but never uses it; `pump()` recomputes its own (line 360). Harmless, but the declared variable is dead.

### 2.6 Offline long-audio "chunking" — a separate mechanism (do not confuse)

`transcribe()` also chunks, but this is **offline** and splits at low-energy boundaries, not at fixed increments:

`qwen3-asr/Qwen3-ASR/qwen_asr/inference/qwen3_asr.py:366-377`

```python
max_chunk_sec = MAX_FORCE_ALIGN_INPUT_SECONDS if return_time_stamps else MAX_ASR_INPUT_SECONDS

# chunk audios and record mapping
chunks: List[AudioChunk] = []
for i, wav in enumerate(wavs):
    parts = split_audio_into_chunks(
        wav=wav,
        sr=SAMPLE_RATE,
        max_chunk_sec=max_chunk_sec,
    )
```

`split_audio_into_chunks` (`qwen_asr/inference/utils.py:246-332`) guarantees byte-exact reconstruction (docstring lines 256-259: *"Concatenating all returned chunks reproduces the original audio exactly (total number of samples is identical, no overlaps, no gaps)"*), searches a ±5 s window for a minimum-energy cut (`search_expand_sec=5.0`, `min_window_ms=100.0`), and then zero-pads sub-0.5 s chunks.

---

## 3. ForcedAligner

### 3.1 Class names

All in `qwen3-asr/Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py`:

| Name | Line | Role |
|---|---|---|
| `Qwen3ForceAlignProcessor` | 37 | Text tokenization + timestamp encoding/decoding/repair |
| `ForcedAlignItem` | 270 | Frozen dataclass: `text`, `start_time`, `end_time` |
| `ForcedAlignResult` | 288 | Frozen dataclass wrapping `items: List[ForcedAlignItem]`; iterable, `len()`, `[]`-indexable |
| `Qwen3ForcedAligner` | 309 | The public wrapper — **yes, this is the exact class name** |

### 3.2 Construction

`qwen3_forced_aligner.py:340-380`

```python
@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    **kwargs,
) -> "Qwen3ForcedAligner":
```
Docstring says `**kwargs` are forwarded to `AutoModel.from_pretrained(...)`, *"Typical examples: device_map="cuda:0", dtype=torch.bfloat16"* (line 361). It registers the HF auto classes, loads the model, raises `TypeError` if `AutoModel` returns anything other than `Qwen3ASRForConditionalGeneration` (lines 372-375), then builds `Qwen3ForceAlignProcessor()`.

Invocation from the README (`README.md:301-312`) and the example (`examples/example_qwen3_forced_aligner.py:199-204`):

```python
aligner = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
)
```
Or via the ASR wrapper: `Qwen3ASRModel.from_pretrained(..., forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B", forced_aligner_kwargs=dict(...))`.

### 3.3 `align(...)` — exact signature

`qwen3_forced_aligner.py:394-400`

```python
@torch.inference_mode()
def align(
    self,
    audio: Union[AudioLike, List[AudioLike]],
    text: Union[str, List[str]],
    language: Union[str, List[str]],
) -> List[ForcedAlignResult]:
```

### 3.4 Input requirements — answering precisely

- **Full audio required?** Yes. `audio` is normalized to mono 16k float32 in [-1,1] via `normalize_audios(audio)` (line 422) and fed to the audio encoder.
- **Full transcript text required?** Yes, as a **string per sample**. The text is tokenized to words and turned into a `<timestamp><timestamp>`-interleaved aligner prompt (`encode_timestamp`, lines 236-252):
  ```python
  input_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
  input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text
  ```
  So the aligner emits a timestamp pair per token; **the transcript must already exist** — it does not transcribe.
- **`language` is mandatory** — no default value; it selects the tokenizer branch (`encode_timestamp`, lines 237-247) and is used for nothing else.
- **List of sentence segments?** **No such API — not present in repo.**
  The list form is a **parallel batch**, not a segmentation feature:
  ```python
  texts = ensure_list(text)
  languages = ensure_list(language)
  audios = normalize_audios(audio)

  if len(languages) == 1 and len(audios) > 1:
      languages = languages * len(audios)

  if not (len(audios) == len(texts) == len(languages)):
      raise ValueError(
          f"Batch size mismatch: audio={len(audios)}, text={len(texts)}, language={len(languages)}"
      )
  ```
  (`qwen3_forced_aligner.py:420-430`.) It is a strict zip of `audio[i] ↔ text[i] ↔ language[i]`; only `language` broadcasts from length 1. Therefore **one audio + N sentence strings raises `ValueError`.** To align sentence-by-sentence you must call `align()` once per sentence yourself, slicing the audio for each — there is no helper for that in the repo.

### 3.5 Output format and granularity

Output is `List[ForcedAlignResult]`, one per sample; each `items` element has `.text`, `.start_time`, `.end_time` **in seconds**, rounded to 3 decimals.

`qwen3_forced_aligner.py:450-460`

```python
for input_id, output_id, word_list in zip(inputs["input_ids"], output_ids, word_lists):
    masked_output_id = output_id[input_id == self.timestamp_token_id]
    timestamp_ms = (masked_output_id * self.timestamp_segment_time).to("cpu").numpy()
    timestamp_output = self.aligner_processor.parse_timestamp(word_list, timestamp_ms)
    for it in timestamp_output:
        it['start_time'] = round(it['start_time'] / 1000.0, 3)
        it['end_time'] = round(it['end_time'] / 1000.0, 3)
    results.append(self._to_structured_items(timestamp_output))
```

- `timestamp_token_id` and `timestamp_segment_time` are read from the **checkpoint's** `config.json` (`self.model.config.timestamp_token_id` / `.timestamp_segment_time`, lines 337-338) — **neither is defined or defaulted in this repo**.
- `masked_output_id` is the argmax token id at timestamp positions; × `timestamp_segment_time` (ms) → ms → `/1000` → seconds.
- Monotonicity repair: `fix_timestamp` (lines 147-234) runs a longest-increasing-subsequence pass over the raw ms values and repairs anomalies (nearest-neighbour fill for ≤2 consecutive anomalies, linear interpolation for longer runs).

**Granularity: word-level and character-level ONLY. Sentence-level timestamps are not produced — not present in repo.**
The granularity is decided by `Qwen3ForceAlignProcessor.encode_timestamp` (lines 236-252):

| Language | Tokenizer | Granularity |
|---|---|---|
| `japanese` | `nagisa.tagging(text).words` (line 102) | word |
| `korean` | `soynlp.tokenizer.LTokenizer(scores=self.ko_score)` + `assets/korean_dict_jieba.dict` (lines 41-48, 110-117, 242-245) | word |
| everything else | `tokenize_space_lang` (line 139): whitespace split, then `split_segment_with_chinese` splits **every CJK char** into its own token (lines 119-137) | word for space-delimited scripts, **per-character for CJK** |

The README confirms this framing (`README.md:295`): *"`Qwen3-ForcedAligner-0.6B` can align text–speech pairs and return word or character level timestamps."*

### 3.6 `get_supported_languages` on the aligner

`qwen3_forced_aligner.py:462-483` — thin wrapper over `self.model.get_support_languages()` (which returns `self.config.support_languages`, `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py:1321-1322`). Returns `sorted({str(x).lower() for x in langs})` or `None` if the model exposes nothing.

### 3.7 Timestamps through `transcribe()`

When `return_time_stamps=True`, alignment is **not** done inside the aligner for the whole utterance. `transcribe` chunks at `MAX_FORCE_ALIGN_INPUT_SECONDS` first (line 366), encodes each chunk via `forced_aligner.align(...)` in batches of `max_inference_batch_size` (lines 409-418), applies per-chunk offsets (line 424), and concatenates items across chunks (`_merge_align_results`, line 563).

---

## 4. `SegmentationConfig`

**`SegmentationConfig` does NOT exist in the Qwen3-ASR repository.** Grep for `SegmentationConfig|segmentation` across `qwen3-asr/Qwen3-ASR/` returns **zero matches**. There is no punctuation-splitting or sentence-segmentation module anywhere in `qwen_asr/`.

It is defined in the **parent host project** (this workspace), not in the cloned repo:

**`backend/config.py:331`** — `class SegmentationConfig(BaseModel)`, wired into the top-level config at **`backend/config.py:491`**:

```python
segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
```

Its docstring (`backend/config.py:332-338`) states its purpose — sentence committing based on **ASR punctuation**, used as a second signal on top of VAD silence (because VAD cuts on silence, which produces fragments in Japanese or >10 s run-ons).

Fields (from `backend/config.py:339-369`):

| Field | Default | Purpose (per inline comments) |
|---|---|---|
| `enabled` | `True` | Master switch |
| `replace_stable_prefix` | `True` | SEG replaces the level-3 `STABLE_PREFIX` cut to avoid two competing cut mechanisms |
| `max_chars` | `100` | Hard char ceiling per sentence before forced cut |
| `min_chars` | `2` | Sentences shorter than this merge into the next |
| `tail_min_chars` | `4` | Minimum content chars of the NEW sentence before committing |
| `tail_scans` | `2` | …across how many scan ticks |
| `tail_stable_ms` | `280.0` | …and stable for how long |
| `stable_ms` | `350.0` | Sentence-ending punctuation must hold still this long |
| `stable_scans` | `2` | …across how many scans |
| `stable_cut` | `False` | Allow committing on stable punctuation without seeing a new sentence; **default off** (ASR emits `.` early mid-sentence in Japanese/English) |
| `fallback_overlap_ms` | `600.0` | Overlap fallback when no silence anchor is found |
| `use_whisper_timer` | `True` | Use a timestamped model (whisper) for exact cut points — comment: *"Qwen3 không có timestamp"* |
| `whisper_model_key` | `"whisper-large-v3-turbo"` | Timer model |
| `timer_min_confidence` | `0.35` | Timer confidence floor |
| `timer_max_audio_sec` | `30.0` | Don't run timer for longer fragments |
| `debug_trace` | `True` | Log every ASR preview tick |
| `shadow_when_disabled` | `True` | Run a shadow SEG state machine when SEG is off |

Also referenced in `report/audit/21_SEG_VA_FIX_DEDUP_PHU_DE.md` (lines 218, 293, 301, 438, 483).

**Bottom line:** `SegmentationConfig` is host-project Python-side sentence cutting, **not** a Qwen3-ASR feature. Qwen3-ASR itself provides no sentence segmentation; only `split_audio_into_chunks` (acoustic, energy-minimum) and the aligner's word/char tokenization.

---

## 5. Context / biasing

### 5.1 The API: `context`

`transcribe(audio, context="", language=None, return_time_stamps=False)` — the second parameter.

How it is used, `qwen3_asr/inference/qwen3_asr.py:448-465`:

```python
def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
    ]

def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
    """
    Build the string prompt for one request.

    If force_language is provided, "language X<asr_text>" is appended after the generation prompt
    to request text-only output.
    """
    msgs = self._build_messages(context=context, audio_payload="")
    base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if force_language:
        base = base + f"language {force_language}{'<asr_text>'}"
    return base
```

So `context` becomes the **system message** of the chat template. It is free-form text. There is **no dedicated hotword API, no phrase-boost list, no logit bias, no `previous_transcript` parameter** — not present in repo. The only supported way to bias recognition is the free-text system prompt, which is what the official examples do with hotword-like content:

`examples/example_qwen3_asr_transformers.py:91-96` (identical in `example_qwen3_asr_vllm.py:95-100`):

```python
results = asr.transcribe(
    audio=[URL_ZH, zh_b64, (en_wav, en_sr)],
    context=["", "交易 停滞", ""],
    language=[None, "Chinese", "English"],
    return_time_stamps=False,
)
```
(`"交易 停滞"` = "transaction stagnation" — i.e. domain hotwords passed as context.)

Batch semantics (lines 341-345): a scalar `context` broadcasts to batch size; a list must match `len(audio)` exactly, else `ValueError`.

### 5.2 Context in streaming

`init_streaming_state(context: str = "", ...)` (line 586) stores `prompt_raw = self._build_text_prompt(context=context, force_language=force_language)` (line 639) and reuses it for every chunk, so streaming context is fixed for the lifetime of the stream.

### 5.3 Context for the aligner

**Not present in repo.** `Qwen3ForcedAligner.align(audio, text, language)` has no `context` parameter.

### 5.4 Forced language (a separate, weaker form of biasing)

Setting `language` appends `language {Language}<asr_text>` after the generation prompt (line 463-464) to force text-only output, and `parse_asr_output` then trusts the user value (utils.py:434-436). `language` must be in `SUPPORTED_LANGUAGES` (validated at line 363). The vLLM in-engine `get_generation_prompt` does the same thing for the OpenAI transcription endpoint (`qwen_asr/core/vllm_backend/qwen3_asr.py:986-990`).

---

## 6. Concrete numbers: sizes, dtypes, memory, batch, latency, install

### 6.1 Model sizes / family

- `Qwen3-ASR-1.7B`, `Qwen3-ASR-0.6B` (ASR), `Qwen3-ForcedAligner-0.6B` (NAR aligner) — `README.md:24`, `README.md:82-83`.
- HF-native variants referenced in News (`README.md:20-23`): `Qwen/Qwen3-ASR-1.7B-hf`, `Qwen/Qwen3-ASR-0.6B-hf`, `Qwen/Qwen3-ForcedAligner-0.6B-hf` (transformers-native + `torch.compile` support, dated 2026.6.26).
- Default checkpoint used throughout examples/CLI: `Qwen/Qwen3-ASR-1.7B`; aligner: `Qwen/Qwen3-ForcedAligner-0.6B`.
- Parameter-count claims only in the names; **no per-model parameter table beyond the names, no VRAM table.**

### 6.2 Dtypes

- Evaluation protocol (`README.md:617`): *"we ran inference for all models with `dtype=torch.bfloat16` and set `max_new_tokens=1024` using vLLM. Greedy search was used for all decoding."*
- Every example and every demo default uses `dtype=torch.bfloat16` with `device_map="cuda:0"` — e.g. `qwen_asr/cli/demo.py:229-234` (transformers) and `:243-247` (aligner).
- FlashAttention 2 requires fp16/bf16 (`README.md:146`: *"FlashAttention 2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`"*).
- **No fp8/int8/int4/quantization guidance — not present in repo.**

### 6.3 VRAM / GPU memory

**No VRAM requirements table exists in the repo.** The only memory-shaped numbers are:
- `gpu_memory_utilization=0.7` (README vLLM example, `README.md:223`), `0.8` (`demo_streaming.py:477` default and `example_qwen3_asr_vllm.py:134`), `0.9` (README streaming demo, `README.md:444`), `0.65` (documented as an override example, `README.md:382`).
- Docker: `--shm-size=4gb` (`README.md:595`).
- flash-attn build: *"If your machine has less than 96GB of RAM and lots of CPU cores, run: `MAX_JOBS=4 pip install -U flash-attn --no-build-isolation`"* (`README.md:140-144`). This is **host RAM**, not VRAM.

### 6.4 Batch size recommendations

`max_inference_batch_size` semantics: *"-1 means no chunking / unlimited. Small values can avoid OOM"* (docstrings at `qwen3_asr.py:195-196`, `:248-249`).

| Context | Value | Source |
|---|---|---|
| `from_pretrained` default | `32` | `qwen3_asr.py:181` |
| `LLM` default | `-1` (unlimited) | `qwen3_asr.py:232` |
| README transformers example | `32` | `README.md:163` |
| README vLLM example | `128` | `README.md:224` |
| Gradio demo defaults (both backends) | `4` | `qwen_asr/cli/demo.py:232, 238` |
| README demo `--backend-kwargs` examples | `8` | `README.md:356, 366` |
| Streaming | batching not supported (`batch_size` unused) | `qwen3_asr.py:598`, `:681` |

`max_new_tokens` guidance: `512` default in `from_pretrained`; `4096` default in `LLM`; README transformers example `256` with the note *"Set a larger value for long audio input"* (`README.md:164`); README vLLM example `4096`; evaluation used `1024`; streaming examples force `32` with the comment *"set a small value for streaming"* (`example_qwen3_asr_vllm_streaming.py:93`); demo defaults `512` (transformers) / `4096` (vLLM).

### 6.5 Throughput / latency / RTF

- **The only throughput number in the repo** (`README.md:63`): *"While the 0.6B version achieves accuracy-efficient trade-off, it reaches **2000 times throughput at a concurrency of 128**."* No units (×RTF implied), no hardware stated, no counterpart number for 1.7B.
- **RTF tables: not present in repo. Latency benchmarks: not present in repo. Tokens/s: not present in repo.**
- What *is* present are accuracy tables (all in `README.md`), summarized:
  - ASR WER on public datasets — LibriSpeech clean/other, GigaSpeech, CV-en, Fleurs-en, MLS-en, Tedlium, VoxPopuli, WenetSpeech, AISHELL-2, SpeechIO, Fleurs-zh, CV-zh, KeSpeech, Fleurs-yue, CV-yue, CV-zh-tw, WenetSpeech-Yue, WenetSpeech-Chuan (`README.md:619-828`). Best examples: 1.7B LibriSpeech clean **1.63** / other **3.38**; 0.6B LibriSpeech clean 2.11 / other 4.55.
  - Internal ASR WER: Dialog-Accented English, Elders&Kids, ExtremeNoise, TongueTwister, Dialog-Mandarin, Dialog-Cantonese, Dialog-Chinese Dialects (`README.md:830-930`).
  - Multilingual WER: MLS, CommonVoice, MLC-SLM, Fleurs/Fleurs†/Fleurs††, News-Multilingual (`README.md:932-1013`).
  - Language-ID accuracy (`README.md:1015-1062`): avg 96.8 % (0.6B), 97.9 % (1.7B), 94.1 % (Whisper-large-v3).
  - Singing/song WER: M4Singer, MIR-1k-vocal, Opencpop, Popcs, EntireSongs-en/zh (`README.md:1064-1143`).
  - **Streaming vs offline WER** (`README.md:1145-1193`) — 1.7B: offline avg 2.69 → streaming avg 3.33; 0.6B: offline avg 3.48 → streaming avg 4.40.
  - Forced-alignment AAS in ms (`README.md:1195-1424`): Qwen3-ForcedAligner-0.6B avg **42.9 ms** (MFA-Labeled Raw), **52.9 ms** (MFA-Labeled Concat-300s), **32.4 ms** (Human-Labeled), vs Monotonic-Aligner / NFA / WhisperX which reach 1742.4 / 2708.4 ms in the worst cells.

### 6.6 Install requirements

**Python** — `requires-python = ">=3.9"` (`pyproject.toml:10`), classifiers list 3.9 through 3.13 (`pyproject.toml:11-18`). README recommends a fresh **Python 3.12** env (`README.md:105-110`).

**Hard dependencies** (`pyproject.toml:22-34`) — all exact pins except the unpinned ones:

```toml
dependencies = [
  "transformers==4.57.6",
  "nagisa==0.2.11",
  "soynlp==0.0.493",
  "accelerate==1.12.0",
  "qwen-omni-utils",
  "librosa",
  "soundfile",
  "sox",
  "gradio",
  "flask",
  "pytz",
]
```

**vLLM extra** (`pyproject.toml:36-39`): `vllm = ["vllm==0.14.0"]`.

**torch is NOT a declared dependency** — it arrives transitively via `transformers`/`accelerate`. There is **no torch pin and no CUDA index URL anywhere in `pyproject.toml`**. (Directly relevant to this workspace's AGENTS.md §2.3 PyTorch-CUDA invariant: installing `qwen-asr` will not itself pull the CUDA wheel.)

**FlashAttention** (`README.md:134-146`): optional but recommended, `pip install -U flash-attn --no-build-isolation`, use `MAX_JOBS=4` if <96 GB RAM, requires Ampere+ hardware and fp16/bf16.

**vLLM nightly install recipe** (`README.md:456-465`) uses CUDA **12.9** wheels:
```bash
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]"
```
Note this README recipe (`cu129`) conflicts with the pinned `vllm==0.14.0` extra and with the Docker image (`cu128`).

**Docker** (`docker/Dockerfile-qwen3-asr-cu128:3-4, 43-57`): base `nvidia/cuda:12.8.0-devel-ubuntu22.04`, `MAX_JOBS=32`, `NVCC_THREADS=2`, installs `qwen-asr[vllm]`, builds flash-attn from `git+https://github.com/Dao-AILab/flash-attention.git` when `BUNDLE_FLASH_ATTENTION=true`, `EXPOSE 80`. Published image: `qwenllm/qwen3-asr` (README `README.md:584`).

**Memory note:** the large pinned dependency set (`gradio`, `flask`, `sox`, `nagisa`, `soynlp`) is installed even for headless ASR-only use — relevant if integrating into a constrained environment.

**Two packaging observations:**
- `pyproject.toml:54-55` declares package data under the wrong key — `qwen_tts = ["py.typed", "**/*.dict"]` (copy-paste from the TTS package). The Korean aligner dictionary `qwen_asr/inference/assets/korean_dict_jieba.dict` is nonetheless shipped via `MANIFEST.in:4` (`recursive-include qwen_asr *.dict`).
- Console scripts (`pyproject.toml:45-48`): `qwen-asr-demo`, `qwen-asr-demo-streaming`, `qwen-asr-serve` (the last is a thin `vllm serve` wrapper injecting `"serve"` into `sys.argv`, `qwen_asr/cli/serve.py:40-42`).

**Fine-tuning** (`finetuning/README.md`): needs `qwen-asr` + `datasets`, FlashAttention recommended, JSONL with `{"audio": path, "text": "language English<asr_text>..."}`; example batch size 32 with grad accum 4, lr 2e-5, 1 epoch; single-GPU and `torchrun --nproc_per_node=2` paths.

---

## 7. License

### 7.1 Code: Apache-2.0 — confirmed

- `qwen3-asr/Qwen3-ASR/LICENSE` is the full Apache License 2.0 text (line 1: `Apache License / Version 2.0, January 2004`).
- `pyproject.toml:19`: `license = { text = "Apache-2.0" }`.
- Every source file carries `# SPDX-License-Identifier: Apache-2.0` plus the Apache header (e.g. `qwen_asr/__init__.py:3`, `qwen_asr/inference/qwen3_asr.py:3`, `qwen_asr/inference/qwen3_forced_aligner.py:3`, `finetuning/qwen3_asr_sft.py:3`).
- Copyright line: `# Copyright 2026 The Alibaba Qwen team.`

### 7.2 Model weights: **not present in repo**

The README (the mirror of the HF model card) contains **zero occurrences of "license"** (grep for `[Ll]icense|GB|VRAM|latency|throughput|RTF|...` over `README.md` matches only lines 63, 134, 140 — no license text). There is no license section, no `LICENSE` field, and no per-model license statement. The model card mirror does not state the weights' license.

**Conclusion:** code license = Apache-2.0 (verifiable in-repo). The **model weights' license is not stated anywhere in this repo** — it must be read off the individual Hugging Face / ModelScope model cards (`README.md:10` links the HF collection).

---

## 8. Minimum / maximum audio duration

### 8.1 Constants

`qwen3-asr/Qwen3-ASR/qwen_asr/inference/utils.py:33-36`

```python
SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200
MAX_FORCE_ALIGN_INPUT_SECONDS = 180
MIN_ASR_INPUT_SECONDS = 0.5
```

| Constant | Value | Meaning |
|---|---|---|
| `SAMPLE_RATE` | 16000 | Everything is resampled to mono 16 kHz (`utils.py:198-199`) |
| `MAX_ASR_INPUT_SECONDS` | **1200 s (20 min)** | Chunking threshold for `transcribe()` without timestamps |
| `MAX_FORCE_ALIGN_INPUT_SECONDS` | **180 s (3 min)** | Chunking threshold for `transcribe(return_time_stamps=True)` |
| `MIN_ASR_INPUT_SECONDS` | **0.5 s** | Only used to zero-pad short *chunks* |

### 8.2 Exact behaviour for short inputs — important nuance

The 0.5 s minimum is **not** a validation or rejection rule; it is a padding rule applied **only to chunks produced by the splitter**, and the splitter short-circuits entirely when the audio already fits:

`utils.py:274-277`
```python
total_len = int(wav.shape[0])
total_sec = total_len / float(sr)
if total_sec <= max_chunk_sec:
    return [(wav, 0.0)]
```

`utils.py:322-330`
```python
# Pad too-short chunks to at least MIN_ASR_INPUT_SECONDS (zero-padding at tail)
min_len = int(MIN_ASR_INPUT_SECONDS * sr)
padded: List[Tuple[np.ndarray, float]] = []
for c, off in chunks:
    if c.shape[0] < min_len:
        pad = min_len - int(c.shape[0])
        c = np.pad(c, (0, pad), mode="constant", constant_values=0.0).astype(np.float32)
    padded.append((c, off))
chunks = padded
```

Therefore:
- Any audio **≤ 1200 s** (or ≤ 180 s with timestamps) is passed **unchunked and unpadded** — including a 0.1 s clip. There is **no minimum-duration check and no error** for very short input.
- Only when the total exceeds the max does the 0.5 s padding kick in, and only for the tail fragments produced by cutting.
- Zero-length audio is not explicitly guarded in `transcribe`; the empty/silent path is handled downstream by `parse_asr_output` returning `("", "")` for `language None<asr_text>` (`utils.py:449-455`).

### 8.3 Long-input behaviour

- Audio > `MAX_ASR_INPUT_SECONDS` is split at minimum-energy boundaries and each chunk is transcribed independently, then texts are concatenated and languages merged (`qwen3_asr.py:426-446`). The docstring states chunk concatenation is sample-exact (`utils.py:256-259`).
- With `return_time_stamps=True`, both the ASR chunking **and** the alignment are limited to 180 s per chunk (`qwen3_asr.py:366`).
- Direct `Qwen3ForcedAligner.align()` calls are **not** chunked or length-checked at all in this repo — the caller must respect the limit.
- README claims for the aligner (`README.md:65`): *"supports timestamp prediction for arbitrary units within **up to 5 minutes** of speech in 11 languages."*

> **Documented limit conflict:** README says **5 minutes**, the code uses `MAX_FORCE_ALIGN_INPUT_SECONDS = 180` (**3 minutes**) as the chunking threshold in `transcribe()`. The README is more permissive than the code path. Treat 180 s/chunk as the operative number when using `transcribe(return_time_stamps=True)`; the README's 5-minute claim is not backed by any constant in this repo.
>
> **Latent bug:** `qwen3_asr.py:563-582` `_merge_align_results` references `results[0]` outside the loop that filters `None` (line 582 `return type(results[0])(items=all_items)`) — safe today only because callers pass non-empty lists, but it is an unguarded index.
>
> Also note `_offset_align_result` (line 558-560) rounds to 3 decimals *after* adding the offset, whereas seconds are already rounded to 3 decimals in the aligner — double rounding can accumulate up to 0.5 ms drift per chunk boundary.

### 8.4 vLLM STT path

For the vLLM OpenAI-compatible transcription/translation path, `qwen_asr/core/vllm_backend/qwen3_asr.py:949-958`:

```python
@classmethod
def get_speech_to_text_config(cls, model_config, task_type) -> SpeechToTextConfig:
    processor = cached_processor_from_config(model_config)
    feature_extractor: WhisperFeatureExtractor = processor.feature_extractor
    return SpeechToTextConfig(
        max_audio_clip_s=feature_extractor.chunk_length,
        sample_rate=feature_extractor.sampling_rate,
    )
```

`max_audio_clip_s` is taken from the checkpoint's `WhisperFeatureExtractor.chunk_length` (a Whisper feature-extractor field; the value lives in the model's `preprocessor_config.json`, **not in this repo**). `get_supported_mm_limits` returns `{"audio": None}` — unlimited audios (line 553-554). This vLLM path is a *different* chunking mechanism from `qwen_asr/inference/utils.py`'s splitter.

### 8.5 No other documented limits

**There are no documented short-input thresholds, no "audio must be ≥ X seconds" statement, and no silence/SNR requirements in the README — not present in repo.**

---

## 9. Supported languages and language-detection output format

### 9.1 The authoritative list (code, 30 languages)

`qwen3-asr/Qwen3-ASR/qwen_asr/inference/utils.py:37-68`

```python
SUPPORTED_LANGUAGES: List[str] = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian"
]
```

30 entries. Exposed at runtime via `Qwen3ASRModel.get_supported_languages()` (`qwen3_asr.py:290-297`), which is what the Gradio demo populates its language dropdown from (`qwen_asr/cli/demo.py:351-353`).

Validation: `validate_language` raises `ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")` (`utils.py:105-106`). Normalization is first-letter-uppercase + rest-lowercase: `'cHINese' -> 'Chinese'` (`utils.py:73-92`).

### 9.2 README claim vs. code — a real inconsistency

`README.md:61`: *"Qwen3-ASR-1.7B and Qwen3-ASR-0.6B support language identification and speech recognition for **30 languages and 22 Chinese dialects**"*, and `README.md:16` says *"**52 languages and dialects**"*.

The dialects enumerated in `README.md:82` are: Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan, Zhejiang, Cantonese (Hong Kong accent), Cantonese (Guangdong accent), Wu language, Minnan language (22).

**None of these dialect names appear in `SUPPORTED_LANGUAGES`.** Because `transcribe(..., language=<str>)` calls `validate_language`, passing e.g. `"Sichuan"` raises `ValueError`. Dialects are handled implicitly by automatic detection only — they are not addressable as forced languages in this package.

### 9.3 ForcedAligner languages (11)

`README.md:83`: Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish — *"NAR"* inference mode.

The aligner code has no hard-coded language list; the runtime list comes from the checkpoint via `Qwen3ForcedAligner.get_supported_languages()` → `self.model.config.support_languages` (`qwen3_forced_aligner.py:462-483`, `modeling_qwen3_asr.py:1321-1322`), returning lowercased sorted names or `None`. Note `align()` itself does **not** validate the language against that list — it only lowercases it to pick a tokenizer branch (`encode_timestamp`, line 237); any language other than `japanese`/`korean` falls through to `tokenize_space_lang`.

### 9.4 Language-detection output format

**Raw model output** (what the LLM emits) is a two-part string separated by a literal `<asr_text>` tag:

```
language {Language}<asr_text>{text}
```
Constants (`utils.py:69-70`): `_ASR_TEXT_TAG = "<asr_text>"`, `_LANG_PREFIX = "language "`.

**Parsed form** — `parse_asr_output(raw, user_language=None) -> Tuple[str, str]` (`utils.py:403-470`):

| Case | Result |
|---|---|
| `"language Chinese<asr_text>你好"` | `("Chinese", "你好")` |
| `"language Chinese\n...\n<asr_text>..."` | lang parsed from the `language `-prefixed line (line 459-468) |
| No `<asr_text>` tag at all | `("", raw.strip())` — *"no tag => pure text"* (line 443-445) |
| `"language None<asr_text>"` with empty text | `("", "")` — empty audio heuristic (line 450-455) |
| `"language None<asr_text>something"` | `("", "something")` — language unknown, text kept (line 454-455) |
| `user_language` provided (non-empty) | `(user_language, raw_stripped)` — output treated as pure text (line 434-436) |

Repetition repair runs first: `detect_and_fix_repetitions(s)` with `threshold=20` (`utils.py:432`, implementation lines 335-400) collapses >20× character repeats and ≥20× pattern repeats — a hallucination guard for long/silent audio.

**Aggregated form in `ASRTranscription.language`** — `merge_languages` (`utils.py:473-497`) joins per-chunk languages in order, dropping empties and **consecutive** duplicates:

```
["Chinese", "English", "English"] -> "Chinese,English"
```

So a single audio that switches language yields a comma-separated multi-label string. Unknown/silent audio yields `""` (documented at `qwen3_asr.py:64-65`).

---

## 10. Summary of gaps — things explicitly NOT present in this repo

| Asked about | Status |
|---|---|
| Token-level streaming of a completed utterance (`stream=True`, token iterator) | **Not present.** Only per-chunk re-decoding of accumulated audio. |
| Streaming with the transformers backend | **Not supported.** Raises `ValueError` (`qwen3_asr.py:625-626`). |
| Streaming + timestamps | **Not supported.** Documented at `qwen3_asr.py:597`. |
| Streaming + batching | **Not supported.** Documented at `qwen3_asr.py:598`. |
| `SegmentationConfig` / punctuation-based sentence splitting inside Qwen3-ASR | **Not present in repo.** Lives in the host project at `backend/config.py:331`. |
| Sentence-level timestamps from the aligner | **Not present.** Word/char granularity only. |
| Aligner accepting one audio + a list of sentence segments | **Not present.** Lists are a parallel batch (strict zip), not segmentation. |
| Dedicated hotword / phrase-boost / logit-bias API | **Not present.** Only free-text `context` as a system message. |
| `previous_transcript` parameter | **Not present** on the public API (internal streaming prefix only). |
| VRAM requirements, RTF, latency, tokens/s tables | **Not present.** Only one throughput claim (2000× at concurrency 128 for 0.6B) and `gpu_memory_utilization` fractions. |
| torch version pin / CUDA index in packaging | **Not present.** torch is transitive only. |
| Model-weights license statement | **Not present** in this repo (code is Apache-2.0). |
| Quantization (fp8/int8/int4) guidance | **Not present.** |
| Documented minimum-duration / short-input rejection rule | **Not present.** `MIN_ASR_INPUT_SECONDS = 0.5` is a chunk-padding value only. |
