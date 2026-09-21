# 19 — KẾ HOẠCH VIẾT LẠI TẦNG VAD (firered-vad / fsmn-vad / silero-vad)

> Trạng thái: **ĐÃ TRIỂN KHAI** (xem §10 — nhật ký triển khai & khác biệt so với kế hoạch).
> Bằng chứng đo đạc trong kế hoạch này sinh từ `scratch/vad_rewrite_smoke.py`,
> `scratch/ava_align_check.py`, `scratch/vad_events_probe.py`, `scratch/vad_energy_probe.py`,
> `scratch/vad_configs_vs_docs.py` và `scratch/ava_vad_calibrate.py` (đều trong `scratch/`,
> đã bị gitignore).

---

## 0. Nguyên tắc thiết kế (mọi quyết định dưới đây suy ra từ 4 điều này)

1. **VAD là "tap" thụ động**: không đổi, không resample, không gain, không clip, không
   lượng tử hoá lại một mẫu nào trên đường `plugin → VAD → ASR`. VAD chỉ quyết định
   *đoạn nào* được chuyển tiếp.
2. **Engine sở hữu state machine**: mọi quyết định nói/im do chính VAD (theo docs) đưa ra
   qua event `START`/`END`. Processor **không** tự đoán bằng `is_speech` từng frame và
   **không** có đồng hồ im lặng riêng.
3. **Config 1–1 với docs** của từng engine, mỗi engine một config riêng; hai knob chung của
   popup (`threshold`, `silence_duration_ms`) được *chiếu* xuống đúng field native của
   engine đang chọn.
4. **Đơn giản nhất**: xoá `hangover_ms`, `pre_speech_buffer_ms`, `frame_hop_ms`, bộ đếm
   silence trong processor và các field config chết.

---

## 1. Hiện trạng — bằng chứng cho việc phải viết lại

| # | Vấn đề | Bằng chứng |
|---|---|---|
| 1 | Processor **bỏ qua state machine của VAD**: `res.is_speech` là quyết định ngưỡng từng frame nên START bắn ngay ở frame vượt ngưỡng đầu tiên, không chờ `min_speech_frame` và không dùng `pad_start_frame` | `backend/vad/processor.py:448` (`not state.is_speech and is_speech_frame`) |
| 2 | **Mất đầu câu**: `pre_speech_buffer_ms=0` ⇒ `deque(maxlen=max(1,0))=1` ⇒ chỉ 1 frame pre-roll, trong khi VAD cần `min_speech_frame`=8 frame (80 ms) + `pad_start_frame`=5 frame (50 ms) mới báo START ⇒ tới ~130 ms phụ âm đầu bị bỏ khỏi ASR | `processor.py:119-120`, `config.py:152`, `engines/firered.py:79-95` |
| 3 | `hangover_ms` là **config chết** (toàn bộ khối grace đã comment-out) | `processor.py:493-504` |
| 4 | FireRed tự dựng `AudioFeat`/`DetectModel`/`StreamVadPostprocessor` thay vì dùng `FireRedStreamVad.from_pretrained` như docs; cửa sổ trượt bị **zero-pad** ở frame đầu (`prev=zeros(400)`) ⇒ frame 1 lệch 240 mẫu so với upstream framewise | `engines/firered.py:29-45,127-142` |
| 5 | Silero: `neg_threshold_offset` là config chết (VADIterator 6.2.1 hardcode `threshold-0.15`) | `config.py:116` + source `silero_vad` |
| 6 | FSMN đọc `stats.frame_probs[-1]` khi list rỗng ⇒ `IndexError` | `engines/fsmn.py:147-150` (đã tái hiện trong smoke test) |
| 7 | `frame_hop_ms` là tham số lệch docs (docs: `FRAME_SHIFT_SAMPLE=160` = 10 ms cố định) | `config.py:98`, `fireredvad/core/constants.py` |

### 1.1 Smoke test API "đúng docs" đã chạy trên máy này

`python scratch/vad_rewrite_smoke.py` (file `wav_test/Russian_4s.wav`, 4,76 s):

```
FireRed: framewise (400 mau / buoc 160)
frames=474 starts=[(59, 47)] ends=[(450, 450)] rtf=0.278
detect_chunk(1840 mau) -> 10 frame; frame_idx: [1..10]

Silero: load_silero_vad + VADIterator (512 mau)
load=93ms frames=148 events=[(17, {'start': 7712}), (140, {'end': 69600})] rtf=0.031
reset_states() ok; triggered = False

FSMN: AutoModel + generate(cache, is_final=False, chunk_size=60ms)
load=0.19s  vad_opts: speech_noise_thres=0.6 max_end_silence_time=800
frames=79 signals=[(12, [220, -1]), ('final', [-1, 4730])] rtf=0.086
```

Kết luận rút ra (dùng trực tiếp cho thiết kế):
- FireRed báo START ở frame 59 nhưng `speech_start_frame=47` ⇒ **phải lùi 12 frame = 120 ms**
  để không mất đầu câu ⇒ pre-roll **suy ra từ config của VAD**, không phải knob người dùng.
- Silero: `{'start': 7712}` ⇒ `lookback = current_sample - start = 992 mẫu ≈ 2 frame 512`.
- FSMN: tín hiệu `[start_ms, -1]` / `[-1, end_ms]`; **END chỉ chắc chắn có khi `is_final=True`**
  ⇒ phiên phải gọi flush ở cuối (`force_end`), hoặc chờ khoảng lặng thật ≥
  `max_end_silence_time` (800 ms).
- RTF: firered 0,278 (hop 10 ms) — silero 0,031 — fsmn 0,086 ⇒ cả 3 đều nhanh hơn thời gian
  thực ≥ 3×.

---

## 2. Thiết kế mới

### 2.1 `backend/config.py` — mỗi engine một khối, khớp 1–1 docs

```python
class FireRedVADConfig(BaseModel):        # == fireredvad.FireRedStreamVadConfig
    use_gpu: bool = False
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    pad_start_frame: int = 5
    min_speech_frame: int = 8
    max_speech_frame: int = 2000          # 20 s
    min_silence_frame: int = 20
    chunk_max_frame: int = 30000

class SileroVADConfig(BaseModel):         # == silero_vad.VADIterator
    threshold: float = 0.5
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30

class FsmnVADConfig(BaseModel):           # == funasr VADXOptions (đường streaming)
    chunk_size_ms: int = 60               # số ms mỗi lần generate()
    speech_noise_thres: float = 0.6
    max_end_silence_time: int = 800
    speech_to_sil_time_thres: int = 150
    sil_to_speech_time_thres: int = 150
    window_size_ms: int = 200
    dynamic_silence: bool = False         # ngưỡng cố định = hành vi docs gốc
    output_frame_probs: bool = False      # bật nếu muốn probability thật

class VADConfig(BaseModel):
    enabled: bool = True
    vad_engine: str = "firered-vad"       # firered-vad | silero-vad | fsmn-vad
    sample_rate: int = 16000
    threshold: Optional[float] = None          # None ⇒ dùng default của engine (đúng docs)
    silence_duration_ms: Optional[int] = 450   # None ⇒ dùng default của engine (đúng docs)
    firered: FireRedVADConfig = ...
    silero:  SileroVADConfig  = ...
    fsmn:    FsmnVADConfig    = ...
```

- **Xoá**: `VADConfig.hangover_ms`, `VADConfig.pre_speech_buffer_ms`,
  `FireRedVADConfig.frame_hop_ms`, `FireRedVADConfig.threshold`,
  `SileroVADConfig.neg_threshold_offset`, `FsmnVADConfig.threshold` (property).
- Thêm `VADConfig.effective_threshold` / `effective_silence_ms` để `main.py` (prewarm,
  `/api/config`) trả về giá trị **đang có hiệu lực** của engine đang chọn — popup hiển thị đúng.
- Sửa comment `AudioBufferConfig.chunk_size_samples` (400) — không còn là "frame VAD".

### 2.2 Ánh xạ knob chung → field native (một nguồn sự thật, đúng đơn vị)

| Knob chung | firered-vad | silero-vad | fsmn-vad |
|---|---|---|---|
| `threshold` | `speech_threshold` | `threshold` | `speech_noise_thres` |
| `silence_duration_ms` | `min_silence_frame = round(ms/10)` | `min_silence_duration_ms` | `max_end_silence_time` |

`None` ⇒ dùng field native (đúng docs). Phép chiếu nằm **trong từng engine** (engine biết
hop và đơn vị của mình), không nằm ở processor.

### 2.3 `backend/vad/base.py` — hợp đồng tối giản

```python
@dataclass(slots=True)
class VADResult:
    probability: float = 0.0        # chỉ để log/metric
    event: Optional[str] = None     # "START" | "END" | None — NGUỒN SỰ THẬT DUY NHẤT
    lookback_frames: int = 0        # khi event=="START": số frame ĐÃ tiêu thụ thuộc đoạn nói
    is_speech: bool = False         # thông tin

@dataclass
class VADStreamState:
    engine_state: Any = None        # state riêng của engine (opaque với processor)
    raw_buffer: bytearray = field(default_factory=bytearray)
    pre_roll: Deque[Tuple[bytes, float]] = field(default_factory=deque)
    is_speech: bool = False
    total_samples_processed: int = 0
    def reset(self) -> None: ...

class BaseVADEngine(ABC):
    name: str
    frame_samples: int              # BƯỚC NHẢY (hop) mà processor phải cắt
    max_lookback_frames: int        # trần pre-roll, suy ra từ config của chính engine
    default_threshold: float

    @classmethod
    def prepare_files(cls) -> None: ...
    @abstractmethod
    def create_initial_state(self, threshold=None, silence_ms=None) -> VADStreamState: ...
    @abstractmethod
    def is_speech(self, frame_int16: np.ndarray, state, threshold, silence_ms) -> VADResult: ...
```

Engine tự đồng bộ `threshold`/`silence_ms` vào state mỗi frame (so sánh 1 lần, như code hiện
tại đang làm với `speech_threshold`) ⇒ **không cần** thêm API `apply_runtime_config`.

### 2.4 Ba engine

#### FireRed (`engines/firered.py`) — dùng đúng docs

```python
cfg = FireRedStreamVadConfig(
    use_gpu=..., smooth_window_size=..., speech_threshold=effective_threshold,
    pad_start_frame=..., min_speech_frame=..., max_speech_frame=...,
    min_silence_frame=round(silence_ms / 10), chunk_max_frame=...)
vad = FireRedStreamVad.from_pretrained(str(model_dir), cfg)   # model 2,3 MB ⇒ nạp/session
```

- `frame_samples = 160` (hop 10 ms); engine giữ **cửa sổ trượt 400 mẫu THẬT** và **không
  zero-pad**: chỉ phát frame đầu khi đã gom đủ 400 mẫu ⇒ giống hệt `vad_framewise` của
  upstream (trễ khởi động 15 ms, không lệch pha).
- START ⇒ `lookback_frames = frame_idx - speech_start_frame` (đo được = 12).
- `max_lookback_frames = pad_start_frame + min_speech_frame` (mặc định 13).
- Reset = `vad.reset()`.

#### Silero (`engines/silero.py`) — dùng đúng docs

```python
model = load_silero_vad()          # 93 ms ⇒ nạp/session (model CÓ state, không dùng chung được)
it = VADIterator(model, threshold=..., sampling_rate=16000,
                 min_silence_duration_ms=..., speech_pad_ms=...)
```

- `frame_samples = 512` (32 ms); input `int16 / 32768.0` float32 — đúng thang đo docs.
- START ⇒ `lookback_frames = ceil(((frame_idx*512) - start_sample)/512)` (đo được 2);
  END ⇒ event.
- `max_lookback_frames = ceil((speech_pad_ms + 32) / 32)` (mặc định 2).
- Bỏ hoàn toàn `SileroModelProbe` (hack lấy probability) — probability lấy từ model trực tiếp.
- Reset = `it.reset_states()`.

#### FSMN (`engines/fsmn.py`) — dùng đúng docs streaming

```python
self.model = AutoModel(model=str(model_dir), disable_update=True,
                       disable_pbar=True, disable_log=True)       # DÙNG CHUNG
cache = {}
self.model.model.init_cache(cache, speech_noise_thres=..., max_end_silence_time=..., ...)
res = self.model.generate(input=[tensor_float32], cache=cache, is_final=False,
                          chunk_size=cfg.chunk_size_ms,
                          dynamic_silence=cfg.dynamic_silence,
                          disable_pbar=True, disable_log=True)
```

- `frame_samples = 960` (60 ms); input float32 `int16/32768.0` (funasr nhận tensor nguyên
  trạng, không tự chia — đã kiểm chứng trong `funasr/utils/load_utils.py`).
- `max_lookback_frames = ceil((window_size_ms + sil_to_speech_time_thres)/chunk_size_ms)` (≈ 6).
- Đọc `probability` **có guard** (hết `IndexError`); cập nhật `silence_ms` lúc chạy qua
  `stats.max_end_sil_frame_cnt_thresh` — đúng cách `DynamicStreamingVAD` của fork làm.
- Không mutate `vad_opts` trong `__init__`; reset = `init_cache({})` mới.

### 2.5 `backend/vad/processor.py` — ngắn hơn, không đoán

Giữ nguyên các bảo đảm tốt đang có (model chạy **ngoài** lock QWEN-Q2/QWEN-Q4, đổi engine
không chặn event loop P1.9, `reset()` đổi identity QWEN-Q5, callback phát ngoài lock), nhưng:

1. Cắt frame theo `engine.frame_samples` — **hop, không chồng lấn** ⇒ ASR không bao giờ nhận
   mẫu trùng (cửa sổ chồng lấn 400/160 nằm **bên trong** VAD).
2. `_apply_frame_result` chỉ còn 4 nhánh:
   - `event == "START"` ⇒ `state.is_speech = True`, xả `min(lookback_frames, len(pre_roll))`
     frame **nguyên bản** ra `on_speech_chunk(tag=PRE_ROLL)` theo thứ tự thời gian, rồi phát
     frame hiện tại (`tag=SPEECH`);
   - đang `is_speech` ⇒ phát frame hiện tại;
   - `event == "END"` ⇒ `is_speech = False` → `on_speech_end()`;
   - còn lại ⇒ đẩy frame vào `pre_roll` (`maxlen = engine.max_lookback_frames`).
   **Xoá** `silence_samples`, khối hangover comment, `_max_pre_frames` theo ms cấu hình.
3. `VADProcessor.__init__` bỏ `hangover_ms`, `pre_speech_buffer_ms`; giữ tên
   `silence_duration_ms` nhưng truyền xuống engine qua mỗi `is_speech`.
4. `apply_engine_change`: nếu đang `is_speech` ⇒ gọi `on_speech_end()` **trước** khi swap
   (không để câu treo khi người dùng đổi engine giữa câu).
5. `update_config()` bỏ `hangover_ms`.

---

## 3. Bảo đảm toàn vẹn tín hiệu (yêu cầu gắt của đề bài)

Hợp đồng **phải được chứng minh bằng test**, không chỉ bằng lời:

- `b"".join(frame đã gửi on_speech_chunk)` == **đúng byte-slice gốc** của PCM plugin gửi cho
  đoạn `[start, end]` — bit-exact, không mẫu nào đổi/thêm/mất. Frame `PRE_ROLL` là frame đã
  buffer nguyên bản, **không sinh lại** từ float.
- Không engine nào gọi resample/gain/clip/normalize trên đường tới callback; mọi phép
  `int16 → float32` chỉ tạo **bản sao** để đưa vào model.
- Không frame nào được chuyển tiếp hai lần (kiểm bằng tổng số sample nhận được và bằng so
  byte trực tiếp).
- Tích hợp: `TranscribeEngine.feed_audio` (đích thật của callback) nhận đúng chuỗi byte đó.

**Ghi chú trung thực**: `config.asr.normalize_speech = True` (`SpeechNormalizer`, áp gain
trước khi đưa vào model native) là tầng **của ASR** — `backend/asr/engine.py:687` — không
thuộc VAD. Kế hoạch này **không** đổi nó, nhưng sẽ ghi rõ trong báo cáo và đề xuất một phép
đo A/B (`normalize_speech=True/False`) như hạng mục tuỳ chọn ở bước 9.

---

## 4. Danh sách file thay đổi

| File | Việc |
|---|---|
| `backend/vad/base.py` | Viết lại: `VADResult` (event + lookback), `VADStreamState` (engine_state), `BaseVADEngine` (`frame_samples`, `max_lookback_frames`) |
| `backend/vad/engines/firered.py` | Viết lại theo `FireRedStreamVad` + `FireRedStreamVadConfig` (bỏ tự dựng AudioFeat/DetectModel/Postprocessor, bỏ zero-pad) |
| `backend/vad/engines/silero.py` | Viết lại theo `load_silero_vad` + `VADIterator` (bỏ `SileroModelProbe`) |
| `backend/vad/engines/fsmn.py` | Viết lại theo `generate(cache, is_final=False, chunk_size)`; bỏ mutate global trong `__init__`; guard `frame_probs` |
| `backend/vad/engines/__init__.py` | Giữ factory/pool/prewarm (QWEN-Q2, P1.9) — chỉ đổi chữ ký `create_initial_state` |
| `backend/vad/processor.py` | Rút gọn theo §2.5 |
| `backend/vad/__init__.py` | Export như cũ |
| `backend/config.py` | Config mới (§2.1) + sửa comment (test_36 đọc comment này) |
| `backend/ws/session.py` | Bỏ `hangover_ms` khỏi `SessionConfigPayload`/`SessionConfig`/`apply_config`/`init_components`; truyền `silence_duration_ms` + `threshold` xuống processor |
| `backend/ws/handler.py` | Bỏ `hangover=...` trong log (dòng 417) |
| `backend/main.py` | Bỏ `hangover_ms` khỏi `/api/config`; trả thêm khối `vad: {engine, threshold, silence_ms, firered, silero, fsmn}`; prewarm dùng `effective_threshold` |
| `backend/tests/fakes.py` | `FakeVADEngine` → engine **có event** (state machine nhỏ suy từ `silence_ms`), thêm `frame_samples`, `max_lookback_frames` |
| `backend/tests/test_02_vad_benchmark.py` | Cập nhật chữ ký; viết lại phần report |
| `test_08`, `test_10`, `test_14`, `test_33`, `test_34` | Cập nhật chữ ký `VADProcessor(...)`, bỏ `hangover_ms`/`pre_speech_buffer_ms`, cập nhật kỳ vọng event |
| `backend/tests/test_11_config_effectiveness.py` | Thay assert `hangover_ms` bằng assert chiếu `silence_duration_ms`/`threshold` → field native |
| `backend/tests/test_36_metrics_and_tts_hygiene.py` | Viết lại test `min_silence_frame` (kết luận QWEN-Q11 "config CHẾT" nay **đảo chiều**) |
| `report/audit/19_KE_HOACH_VIET_LAI_VAD.md` | Chính tài liệu này |
| `report/02_vad/report.md`, `report/02_vad/ava_speech_report.md` + `.json` | Báo cáo benchmark 3 engine + AVA-Speech |
| `.gitignore` | Thêm `wav_test/vad/derived/` (file dẫn xuất ~19 MB) |

---

## 5. Kế hoạch test

### 5.1 Tầng A — nhanh, không nạp model

`backend/tests/test_41_vad_engine_contract.py`

1. `frame_samples` đúng (160 / 512 / 960); `max_lookback_frames` đúng theo config.
2. Contract event: PCM tổng hợp [lặng 1 s → sin 1 s → lặng 2 s] ⇒ **đúng 1 START rồi 1 END**,
   START sau ≥ `min_speech`, END sau ≈ `silence_ms` (dung sai 1 frame).
3. Chiếu config: `silence_duration_ms=700` ⇒ firered `min_silence_frame==70`,
   silero `min_silence_duration_ms==700`, fsmn `stats.max_end_sil_frame_cnt_thresh==550`;
   `threshold` ⇒ đúng field native của từng engine.
4. Đổi engine giữa câu ⇒ `on_speech_end()` được gọi trước khi swap, không rò state.
5. `reset()` đổi identity state (giữ QWEN-Q5) + pre-roll bị xoá.
6. Không nạp model trên event loop (giữ nguyên tinh thần `test_33`).
7. Thời gian `create_initial_state` mỗi engine (đo; nếu > 300 ms thì chuyển sang model dùng
   chung + state/session — FireRed có constructor công khai
   `FireRedStreamVad(audio_feat, vad_model, postprocessor, config)`).

`backend/tests/test_42_vad_signal_integrity.py`

8. **Bit-exact**: ghép mọi `on_speech_chunk` ⇒ so byte với PCM gốc, assert `==`, không mẫu
   nào lặp/mất — chạy cho cả 3 engine **và** `FakeVADEngine`.
9. Tích hợp `TranscribeEngine.feed_audio`: byte nhận được khớp byte VAD phát ra.

### 5.2 AVA-Speech 10 phút (`backend/tests/test_43_vad_ava_speech.py`, `@pytest.mark.slow`)

Fixture `backend/tests/fixtures/ava_speech_segment.py` (idempotent, chạy 1 lần rồi cache):

- Đọc `wav_test/vad/ava_speech_labels_v1.csv`, lọc `5BDj0ow5hnA`, cửa sổ **900–1500 s**
  (nhãn phủ **liên tục** 900 → 1800 s, đã kiểm chứng không có lỗ hổng).
- Cắt bằng `soundfile` (`seek` + `read`), lấy **kênh 0** (giống
  `extension_firefox/lib/audio-processor.js:139`), resample `scipy.signal.resample_poly(x, 1, 3)`
  48 kHz → 16 kHz, ép `int16`.
- Ghi `wav_test/vad/derived/5BDj0ow5hnA_900_1500_16k_mono.wav` (19,2 MB).

**Alignment đã kiểm chứng trước** bằng `scratch/ava_align_check.py` (cửa sổ 900–960 s):

```
frames=6000 speechGT=0.452 speechPred=0.579
P=0.646 R=0.828 F1=0.726 acc=0.717
offset=+0s   F1=0.726   <-- cao nhất
offset=+150s F1=0.700 | +300s 0.665 | +450s 0.682 | +600s 0.702
offset=-150s..-600s: không có nhãn (F1=0)
```

⇒ timeline khớp ở offset 0, **không cần bù trừ**.

Phép đo cho từng engine: stream 10 phút qua `VADStreamProcessor`, dựng timeline 10 ms từ
**chính các frame đã chuyển tiếp** (đúng thứ ASR thấy), so với nhãn
(`CLEAN_SPEECH`/`SPEECH_WITH_NOISE`/`SPEECH_WITH_MUSIC` = nói; `NO_SPEECH` = lặng):

- P / R / F1 / accuracy frame-level trên GT thô **và** GT làm mượt theo độ phân giải VAD (lấp
  khoảng lặng < `min_silence`, bỏ đoạn nói < `min_speech`) — vì VAD **không thể** tách khoảng
  lặng ngắn hơn config của nó, so với GT thô là bất công;
- độ trễ biên (median/p90 |Δt| tại các onset), số segment, số segment giả;
- RTF (dự kiến: firered ~0,28 – silero ~0,03 – fsmn ~0,09 ⇒ 10 phút audio mất ~2,8 phút /
  ~0,3 phút / ~0,9 phút);
- quét `speech_threshold` của firered ∈ {0,3; 0,4; 0,5; 0,6} để có bảng tham chiếu;
- **ngưỡng cứng chốt sau lần chạy đầu** (đề xuất khởi điểm: recall ≥ 0,85; F1 ≥ 0,70;
  median |Δt| ≤ 100 ms; RTF ≤ 0,5);
- xuất `report/02_vad/ava_speech_report.md` + `report/02_vad/ava_speech.json`.

---

## 6. Thứ tự thực hiện (mỗi bước có điểm kiểm chứng)

1. `vad/base.py` + `config.py` mới; cập nhật `tests/fakes.FakeVADEngine` (event-emitting).
2. `engines/firered.py` → `python scratch/vad_rewrite_smoke.py firered` + test tầng A.
3. `engines/silero.py` → như trên (silero).
4. `engines/fsmn.py` → như trên (fsmn); xác nhận hết `IndexError`, END đến sau
   ~`max_end_silence_time`, `force_end()` flush được đuôi.
5. `processor.py` viết lại (event-driven + pre-roll suy ra) + `engines/__init__.py`.
6. `ws/session.py`, `ws/handler.py`, `main.py` (bỏ hangover, thêm khối `vad` vào `/api/config`).
7. Cập nhật test cũ (`02/08/10/11/14/33/34/36`) ⇒ `pytest -q` tầng A phải xanh.
8. Thêm `test_41`, `test_42` ⇒ xanh.
9. Fixture AVA + `test_43` (`-m slow`) ⇒ chạy 3 engine, chốt ngưỡng, sinh báo cáo; tuỳ chọn
   đo A/B `normalize_speech`.
10. Cập nhật README/comment liên quan + đánh dấu hoàn tất tài liệu này.

---

## 7. Rủi ro & quyết định cần chốt

| Rủi ro | Xử lý trong kế hoạch |
|---|---|
| `silence_duration_ms` từng là nguồn duy nhất ⇒ sẽ ghi đè mặc định docs của engine | **Quyết định cuối (theo yêu cầu)**: mặc định `None` = **không ghi đè** ⇒ mỗi engine dùng đúng mặc định docs (FireRed `min_silence_frame=20` = 200 ms · Silero 100 ms · FSMN 800 ms). Popup đổi slider thành **0 = off** |
| ASR nhận thêm đuôi lặng bằng `min_silence` của engine (200–800 ms) | **Không** phải suy hao tín hiệu (không đổi mẫu nào) — đây đúng là "hangover tích hợp sẵn trong VAD". Đo được trong `test_43`: byte khớp 100 % |
| Engine không bắn END ⇒ câu treo | Lưới an toàn sẵn có ở ASR (`SentenceConfig.enable_tier234`: MAX_DURATION 15 s, TIMEOUT_FORCE) + `test_41` chốt hành vi END của cả 3 engine |
| funasr ghi `vad_opts.max_end_silence_time` toàn cục trong `init_cache` | Giới hạn đã biết (thực tế 1 phiên); cập nhật lúc chạy dùng `stats.max_end_sil_frame_cnt_thresh` (per-session). Ghi chú trong code |
| Nạp model mỗi session (silero 93 ms, firered 2,3 MB) | Đo trong `test_41`: Silero ~0,16 s · FireRed ~1,6 s (lần đầu, gồm import) ⇒ giữ nguyên thiết kế per-session |
| `test_36` assert `min_silence_frame == 60` + comment "config CHẾT" | ⚠️ **Phải cập nhật**: yêu cầu "config giống docs" (20 frame) mâu thuẫn trực tiếp với assert 60. Test được viết lại để chốt giá trị **20 = mặc định docs** và ghi rõ đây là knob chốt câu thật (QWEN-Q11 chỉ còn đúng cho kiến trúc cũ) |
| `test_11` assert `hangover_ms` | Đã viết lại thành assert chiếu silence/threshold xuống field native + `0 = off` |

---

## 8. Tiêu chí nghiệm thu (Definition of Done)

- [x] 3 engine chạy đúng API docs; đổi engine nóng trong popup không chặn event loop, không mất audio
      (đổi engine giữa câu còn **chốt câu cũ** trước khi swap).
- [x] Mỗi engine có config riêng đúng docs (đối chiếu tự động: `scratch/vad_configs_vs_docs.py`);
      `hangover_ms` / `pre_speech_buffer_ms` / `frame_hop_ms` đã bị xoá khỏi config, session payload,
      processor và test.
- [x] Test bit-exact chứng minh `plugin → VAD → ASR` không mất / không đổi / không lặp một mẫu nào
      (`test_42` tầng A + `test_41`/`test_43` trên model thật, kể cả 10 phút AVA-Speech: 0 sai lệch byte).
- [x] `pytest -q` (tầng A) **387 passed · 0 failed** (đã sửa `test_20` và `test_36`, xem §10.5).
- [x] `test_43` chạy 10 phút AVA-Speech cho cả 3 engine, có báo cáo P/R/F1 + độ trễ biên + RTF
      trong `report/02_vad/`.
- [x] Không còn config chết trong `backend/vad/**` và `VADConfig`.

---

## 9. Không nằm trong phạm vi (ghi rõ để tránh trôi)

- Không sửa resampler/ducking của worklet — chỉ **đọc** để đối chiếu định dạng PCM. (Ngoại lệ duy
  nhất ở frontend: slider VAD Silence đổi sang **0 = off** theo yêu cầu.)
- Không sửa boundary overlap / `SentenceConfig`; giữ nguyên `normalize_speech=True` của ASR.
- Không đổi giao thức WebSocket hay tên message; chỉ **bỏ** field `hangoverMs` (không client nào gửi)
  và cho `silenceDurationMs = 0` mang nghĩa "off".
- Không thêm VAD thứ tư, không thêm model mới.

---

## 10. Nhật ký triển khai (khác biệt so với kế hoạch)

### 10.1 Config đã đối chiếu 1-1 với docs

`python scratch/vad_configs_vs_docs.py` (đọc thẳng dataclass/signature của thư viện và
`backend/models/fsmn_vad/config.yaml`):

| Engine | Nguồn đối chiếu | Kết quả |
|---|---|---|
| `firered-vad` | `fireredvad.FireRedStreamVadConfig` | 8/8 field khớp (`min_silence_frame=20` = **200 ms**). Riêng `speech_threshold=0.4`: dataclass upstream mặc định 0.5, nhưng ví dụ CLI/README và yêu cầu dự án dùng **0.4** — đã ghi chú ngay tại field |
| `silero-vad` | `silero_vad.VADIterator.__init__` | 3/3 knob khớp (0.5 · 100 ms · 30 ms); `sampling_rate` không phơi ra vì VADIterator chỉ nhận 8000/16000 |
| `fsmn-vad` | `config.yaml` của checkpoint `funasr/fsmn-vad` (VADXOptions) | 8/8 khớp (800 · 0.6 · 150 · 150 · 200 · 200 · 100 · False); `chunk_size_ms=60` = mặc định của wrapper streaming chính thức `DynamicStreamingVAD` |

`VADConfig.threshold` và `VADConfig.silence_duration_ms` mặc định **None** = *không ghi đè*; popup
hiển thị **VAD Silence = 0 (mặc định VAD)**.

### 10.2 Quyết định theo yêu cầu người dùng

1. `silence_duration_ms = None` mặc định; slider popup min = 0 và `0` được map thành `None` ở cả
   `content-script.js` (`||` cũ biến 0 → 300 ms) lẫn `popup.js`, `ws/session.py`, `main.py`.
2. `min_silence_frame = 20` (đúng docs) ⇒ **buộc phải cập nhật `test_36`** vì test đó assert 60.
3. Giữ nguyên `normalize_speech=True` của ASR (không đo A/B trong lượt này).

### 10.3 Lỗi thật phát hiện thêm khi viết lại

| Lỗi | Cách sửa |
|---|---|
| **Timestamp frame sai** khi frame vắt qua nhiều chunk client (hop 32 ms của Silero / 60 ms của FSMN với chunk 20 ms): `frame_ts = capture_timestamp + offset/bytes_per_sec` tính offset trong chunk hiện tại ⇒ timestamp nhảy sai (đo được 40 ms thay vì 32 ms) | Thêm `VADStreamState.stream_ts` = mốc thời gian của BYTE ĐẦU TIÊN trong `raw_buffer`; mọi phép cắt/overflow đều tiến đồng hồ tương ứng ⇒ timestamp đúng từng frame (chốt bằng `test_41::test_khong_co_frame_nao_bi_gui_hai_lan`) |
| FSMN đọc `stats.frame_probs[-1]` khi list rỗng ⇒ `IndexError` | `_last_frame_prob()` có guard (trả 0.0) |
| Đổi engine giữa câu để câu treo ở phía ASR | `apply_engine_change()` gọi `on_speech_end()` trước khi swap |
| Mất đầu câu do pre-roll cố định 1 frame | Pre-roll nay do VAD yêu cầu qua `VADResult.lookback_frames`, trần = `max_lookback_frames` suy từ config của chính engine (FireRed 13 · Silero 2 · FSMN 11) |
| `prewarm()` dựng state TRƯỚC khi `_attach_engine()` ⇒ `deque(maxlen=self._max_lookback_frames)` bị chốt **maxlen = 1** (mất gần hết pre-roll) trên đúng đường khởi động thật `_prewarm_vad_default` | `_new_state()` lấy `maxlen` TRỰC TIẾP từ engine; `prewarm()` gắn engine trước khi dựng state. Chống hồi quy bằng `test_42::test_prewarm_giu_dung_maxlen_pre_roll` |

### 10.4 Số đo chốt (máy tham chiếu RTX 5060 Ti, Python 3.13)

`test_41` (audio thật `Russian_4s.wav`, 5,5 s gồm lặng đệm) — pre-roll đo được **đúng bằng** số
frame engine yêu cầu, năng lượng tín hiệu được phủ ≥ 99,97 %:

| Engine | Hop | Pre-roll xả | Năng lượng phủ | RTF |
|---|---|---|---|---|
| firered | 160 mẫu (10 ms) | 12 frame (120 ms) | 99,98 % | 0,29 |
| silero | 512 mẫu (32 ms) | 2 frame (64 ms) | 99,97 % | 0,04 |
| fsmn | 960 mẫu (60 ms) | 8 frame (480 ms) | 100,00 % | 0,07 |

`test_43` (10 phút AVA-Speech, 900–1500 s, mặc định docs): xem
`report/02_vad/ava_speech_report.md`:

| Engine | F1 | Recall | Precision | Accuracy | Segment (pred/GT) | Onset trung vị | RTF | Byte sai lệch |
|---|---|---|---|---|---|---|---|---|
| `firered-vad` | **0,918** | 0,929 | 0,907 | 0,878 | 123 / 46 | 100 ms | 0,332 | **0** / 44.958 frame |
| `silero-vad` | 0,652 | 0,496 | 0,952 | 0,613 | 113 / 46 | 95 ms | 0,022 | **0** / 7.120 frame |
| `fsmn-vad` | 0,777 | 0,694 | 0,882 | 0,708 | 59 / 46 | 815 ms | 0,053 | **0** / 5.763 frame |

Quét `speech_threshold` của FireRed trên 2 phút đầu: F1 = 0,899 (0,3) · **0,900 (0,4 — mặc định)** ·
0,894 (0,5) · 0,876 (0,6) ⇒ ngưỡng mặc định 0,4 nằm trong nhóm tốt nhất.

Kiểm chứng cuối cùng: `pytest -q` (tầng A) = **387 passed · 0 failed** (33 deselected là slow/full);
`pytest -m slow test_41 + test_02` = 23 passed; `pytest -m slow test_43` = 5 passed.

### 10.5 Hai test phải sửa (theo yêu cầu)

**`test_36` — test chốt giá trị + comment của `FireRedVADConfig`.** Yêu cầu "config giống docs"
đặt `min_silence_frame = 20`, trong khi test cũ assert `== 60` (mệnh đề QWEN-Q11 còn lại từ kiến
trúc cũ) ⇒ mâu thuẫn trực tiếp, buộc phải viết lại. Test mới chốt:
`value == 20`; comment (1400 ký tự trước lần xuất hiện đầu tiên) phải có `20 frame`, `200 ms`,
`10 ms`, `chốt câu`, `silence_duration_ms`, và **không còn** chữ `CHẾT`.

**`test_20` — quy ước logging.** Đây là test đỏ **có sẵn từ trước** (đã xác nhận đỏ y hệt tại
HEAD bằng `git worktree`): nó báo 4 lời gọi trong `utils/model_download.py` "thiếu module_tag"
trong khi runtime hoàn toàn đúng — tag ở đó là ĐỘNG theo giai đoạn tải
(`tag = stage.upper()` với `stage ∈ {"TRANSLATE","ASR"}`, và `extra={"module_tag": stage.upper()}`).
Checker AST chỉ nhận literal/hằng cấp module nên dương tính giả.

Cách sửa (không hạ chuẩn): thêm **suy diễn chuỗi tĩnh** nhỏ —
`_resolve_str_values()` hiểu literal, hằng cấp module, biến đã gán, `x.upper()/lower()/strip()`,
nối chuỗi, `a if c else b`, f-string tĩnh; `_build_scope()` gieo tham số có default literal rồi
lan truyền trong thân hàm; `_iter_logger_calls()` duyệt theo thứ tự tài liệu và không đi vào
hàm lồng nhau. `_module_tag()` trả **tập** giá trị có thể có, và:
- `test_every_logger_call_has_module_tag`: vẫn bắt mọi trường hợp không suy ra được (thiếu `extra`);
- `test_module_tags_are_canonical`: **mọi** giá trị suy ra phải thuộc `KNOWN_TAGS`;
- thêm `test_checker_van_bat_duoc_tag_thieu_va_tag_sai`: chạy checker trên đoạn mã tổng hợp và
  khẳng định nó vẫn bắt đúng `None` (thiếu) và `WS.HANDLER`/`asr` (ngoài danh sách chuẩn).

Nhờ vậy quy ước logging giữ nguyên độ chặt mà `pytest` tầng A xanh hoàn toàn.

### 10.6 Điều chỉnh `⏱️ VAD Silence` (theo yêu cầu lượt sau)

**Ngữ nghĩa chốt:**

| Giá trị popup | Hành vi |
|---|---|
| `0` (off) | `VADConfig.silence_duration_ms = None` ⇒ **engine quyết định** theo đúng mặc định docs (FireRed `min_silence_frame=20` = 200 ms · Silero 100 ms · FSMN 800 ms). Processor KHÔNG can thiệp: lý do chốt câu là `engine_end`. |
| `> 0` | `silence_duration_ms` là **ĐIỀU KIỆN SỐ 1**: processor tự chốt `END` sau ĐÚNG ngần ấy ms im lặng; `END` sớm của engine bị bỏ qua; engine phát muộn/không phát thì processor vẫn chốt đúng hạn (ghi đè `min_silence_frame` / `min_silence_duration_ms` / `max_end_silence_time`). `END` "cưỡng bức" của engine (vd trần `max_speech_frame`) vẫn được tôn trọng ngay vì frame đó còn bằng chứng tiếng nói. |

**Thay đổi mã:**

* `VADResult.is_speech` được định nghĩa lại thành **bằng chứng tiếng nói của riêng frame**
  (không phải state máy trạng thái): FireRed giữ nguyên (`frame_result.is_speech`), Silero chuyển
  sang `prob ≥ threshold` (thay vì `iterator.triggered` — vốn giữ `True` suốt `min_silence_samples`),
  FSMN dùng xác suất frame và **bật `output_frame_probs` một chiều** khi phiên cần override
  (config vẫn để `False` theo docs), fake engine dùng quyết định RMS từng frame.
* `VADStreamState.last_speech_sample`: mốc mẫu của frame cuối có bằng chứng → nguồn cho đồng hồ im lặng.
* `processor._apply_frame_result()` tách 2 nhánh (docs / ghi đè) + `_close_utterance()` để log rõ
  lý do chốt (`engine_end` / `silence_override`).

**Đo trên engine thật** (`test_41`, audio tiếng nói + đuôi lặng, mốc = frame cuối có bằng chứng):

| Engine | VAD Silence = 0 (docs) | VAD Silence = 700 |
|---|---|---|
| firered-vad | 200 ms (đúng 200 ms docs) | **700 ms** |
| silero-vad | 192 ms (docs 100 ms + pad/hysteresis) | **704 ms** |
| fsmn-vad | 720 ms (docs 800 ms) | **720 ms** |

Test chốt hành vi: `test_42` (tier A, không model) — `silence=0` để engine quyết định, `>0` chốt
đúng hạn kể cả khi engine "hỏng" tự END sau 1 frame; `test_41` — cùng phép đo trên cả 3 model thật.

**Edge case đã xử lý**: đang ghi đè (`>0`) mà người dùng gạt slider về `0` giữa câu. Engine có
thể đã phát `END` sớm và bị hoãn; nếu ta quên quyết định đó thì câu treo vĩnh viễn. Nay
`VADStreamState.engine_end_pending` ghi nhớ "engine đã muốn đóng", và ở chế độ docs frame kế
tiếp tôn trọng ngay (`test_42::test_gat_silence_ve_0_giua_cau_khong_treo_cau`).
