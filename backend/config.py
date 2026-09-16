"""Module quản lý cấu hình tập trung cho Backend.

Hỗ trợ:
- Định nghĩa kiểu dữ liệu chặt chẽ qua Pydantic v2.
- Nạp mặc định không phụ thuộc .env.
- Cơ chế Hot-Reload động (cập nhật cấu hình runtime không cần khởi động lại server).
- Đầy đủ chú thích tiếng Việt cho từng trường cấu hình.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

try:
    import os as _os

    _CPU_COUNT = _os.cpu_count() or 4
except Exception:  # noqa: BLE001
    _CPU_COUNT = 4

# A1-1 (Hy3): mặc định số thread compute sao cho TỔNG số thread CPU-bound (ASR 4 + dịch 4
# + VAD/TTS 2 + browser video decode/render) không vượt quá số nhân vật lý quá xa trên máy
# ít nhân. Giữ mỗi engine ở mức an toàn: ASR (Vulkan chủ yếu GPU) và dịch (CUDA chủ yếu GPU)
# chỉ cần ít thread CPU để feed/parse. Trên máy >=12 nhân giữ nguyên 4; <12 nhân hạ xuống 3.
_AUTO_THREADS = 3 if _CPU_COUNT < 12 else 4

# Đường dẫn thư mục gốc
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
MODELS_YAML_PATH = BACKEND_DIR / "models.yaml"
TRANSLATION_MODELS_YAML_PATH = BACKEND_DIR / "translation_models.yaml"
VOICES_DIR = BACKEND_DIR / "voices"

# Danh sách ngôn ngữ ASR được hỗ trợ
SUPPORTED_LANGUAGES = [
    {"code": "auto", "name": "Auto-detect"},
    {"code": "en", "name": "English (en)"},
    {"code": "ja", "name": "Japanese (ja)"},
    {"code": "zh", "name": "Chinese (zh)"},
    {"code": "ko", "name": "Korean (ko)"},
    {"code": "ru", "name": "Russian (ru)"},
]


class WSConfig(BaseModel):
    """Cấu hình WebSocket và Server WSS."""
    host: str = "0.0.0.0"
    port: int = 8765
    ping_interval: float = 20.0
    ping_timeout: float = 30.0
    max_payload_bytes: int = 10 * 1024 * 1024  # 10MB
    # P3.0: phiên bản giao thức để backend có thể triển khai trước extension.
    # v3 (F-30): payload GỌN — chỉ snake_case, không gửi field trùng lặp
    # (`utterance_id`+`utteranceId`, `original`+`ui_text`+`text`, …).
    # Client cũ khai báo v1/v2 vẫn nhận payload đầy đủ nên không bị ảnh hưởng.
    protocol_version: int = 3
    # FIX-02/FIX-03: trần hàng đợi dịch/TTS. Trước đây là 4 và khi đầy thì **VỨT** câu
    # cuối cùng (queue chỉ chứa câu final) ⇒ phụ đề gốc hiện mà bản dịch/lồng tiếng mất hẳn.
    # Mỗi item chỉ là một câu chữ (vài trăm byte) nên trần lớn hơn gần như không tốn RAM,
    # trong khi overflow trở nên cực hiếm. Khi vẫn overflow thì `_coalesce_enqueue()` GỘP
    # thay vì vứt (xem backend/ws/handler.py).
    translation_queue_maxsize: int = 32
    tts_queue_maxsize: int = 32

class FireRedVADConfig(BaseModel):
    """Cấu hình chuyên biệt cho FireRed-VAD (Xiaohongshu DFSMN)."""
    threshold: Optional[float] = None
    smooth_window_size: int = 5
    min_speech_frame: int = 8       # 8 frames * 25 ms tối thiểu xác nhận bắt đầu nói
    min_silence_frame: int = 60     # 20 frames * 25 ms tối thiểu xác nhận kết thúc nói
    pad_start_frame: int = 5        # 5 frames * 25 ms pre-padding


class SileroVADConfig(BaseModel):
    """Cấu hình chuyên biệt cho Silero VAD."""
    threshold: Optional[float] = None
    neg_threshold_offset: float = 0.15  # Negative threshold = threshold - offset
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30


class FsmnVADConfig(BaseModel):
    """Cấu hình chuyên biệt cho FSMN-VAD (Alibaba FunASR)."""
    speech_noise_thres: Optional[float] = None
    max_end_silence_time: int = 1500
    speech_to_sil_time_thres: int = 200
    sil_to_speech_time_thres: int = 100

    @property
    def threshold(self) -> Optional[float]:
        return self.speech_noise_thres

    @threshold.setter
    def threshold(self, val: Optional[float]) -> None:
        self.speech_noise_thres = val


class VADConfig(BaseModel):
    """Cấu hình tổng hợp cho Voice Activity Detection."""
    enabled: bool = True
    vad_engine: str = "firered-vad"  # firered-vad, silero-vad, fsmn-vad
    threshold: float = 0.45
    silence_duration_ms: int = 600   # Thời gian im lặng (ms) để kích hoạt ngắt câu
    hangover_ms: int = 400           # Giữ trạng thái nói thêm hangover_ms phòng ngắt quãng
    pre_speech_buffer_ms: int = 300  # Đệm âm thanh trước khi bắt đầu nói để tránh mất phụ âm đầu
    sample_rate: int = 16000

    firered: FireRedVADConfig = Field(default_factory=FireRedVADConfig)
    silero: SileroVADConfig = Field(default_factory=SileroVADConfig)
    fsmn: FsmnVADConfig = Field(default_factory=FsmnVADConfig)

    @model_validator(mode="after")
    def _sync_thresholds(self) -> "VADConfig":
        """Đồng bộ ngưỡng mặc định cho các sub-engine nếu chưa cấu hình riêng."""
        if self.firered.threshold is None:
            self.firered.threshold = self.threshold
        if self.silero.threshold is None:
            self.silero.threshold = self.threshold
        if self.fsmn.speech_noise_thres is None:
            self.fsmn.speech_noise_thres = self.threshold
        return self


class ASRConfig(BaseModel):
    """Cấu hình nhận dạng giọng nói ASR qua transcribe.cpp."""
    active_model: str = "qwen3-asr-1.7b"
    # Backend cho transcribe.cpp. Giá trị hợp lệ:
    #   "auto"   — CUDA nếu có, không thì Vulkan (mặc định).
    #   "cuda"   — ưu tiên CUDA; KHÔNG có thì fallback về Vulkan (xem `backend_fallback`).
    #   "vulkan" — chỉ Vulkan. Chỉ định rõ thì KHÔNG tự chuyển sang CUDA.
    # KHÔNG có "cpu": backend CPU vẫn nằm trong thư viện native (`bin/ggml-cpu.dll`) và phải
    # giữ vì ggml cần nó cho các op không offload được lên GPU, nhưng model ASR (audio-LLM)
    # chạy CPU ở RTF 1,6 (chậm hơn thời gian thực) — đo thật, xem backend/asr/native.py —
    # nên không phải lựa chọn hợp lệ cho phụ đề realtime.
    # Đo thật trên RTX 5060 Ti (report/audit/KE_HOACH_FIX_LOI_Hy3.md §4.1.3): CUDA nhanh hơn
    # Vulkan ~1,53x, độ chính xác tương đương (chênh 0,82 điểm % < sàn nhiễu 2,28-4,12),
    # tốn thêm ~243 MB VRAM.
    backend: str = "cuda"
    # Khi backend yêu cầu không khả dụng thì tự fallback theo thứ tự ưu tiên và ghi log
    # WARNING nêu rõ lý do. Đặt False để lỗi nổi lên thay vì âm thầm đổi backend.
    backend_fallback: bool = True
    # --- Bundle native cục bộ (`bin/`) ------------------------------------------
    # Ưu tiên bundle trong `bin/` (bản dựng cục bộ, có thể gồm ggml-cuda.dll) hơn provider
    # `transcribe-cpp-native` đã cài trong site-packages. Xem backend/asr/native.py.
    use_local_native: bool = True
    # Thư mục chứa bundle. Để trống = "<project_root>/bin".
    native_dir: str = ""
    language: str = "auto"
    threads: int = _AUTO_THREADS  # A1-1 (Hy3): tự động theo số nhân CPU
    # P2.8: hạ từ 0.6 -> 0.35 để preview đầu tiên xuất hiện sớm hơn (C5).
    min_transcribe_sec: float = 0.35
    # P2.8: hạ từ 350 -> 300 để nhịp cập nhật phụ đề mượt hơn (C5).
    poll_interval_ms: int = 300
    # K2: đánh thức generator ngay khi VAD báo bắt đầu nói. Không bật thì nhánh idle có
    # thể đang ngủ hết `poll_interval_ms` (300 ms) và preview đầu tiên bị trễ thêm ~300 ms.
    # Đo thật (report §17): K2 giảm từ ~1,29 s xuống ~0,62 s. Đặt False để tắt.
    wake_on_speech_start: bool = True
    models_yaml: str = str(MODELS_YAML_PATH)

    # --- P2.3: cửa sổ preview -------------------------------------------------
    # Chỉ PREVIEW bị cửa sổ hoá; commit luôn dùng toàn bộ ngữ cảnh câu (nguyên tắc P1).
    # Giữ giá trị >= SentenceConfig.max_duration_sec để cửa sổ luôn bao trùm trọn câu
    # hiện tại => preview thấy cùng ngữ cảnh như commit => KHÔNG mất độ chính xác.
    # Đặt 0 để tắt (quay lại hành vi cũ: transcribe từ đầu câu mỗi lần).
    preview_window_sec: float = 6.0
    # P2.4: lịch preview theo nhịp cố định (bỏ nhịp nếu inference vượt hạn).
    preview_fixed_rate: bool = True
    # P2.4b: TỰ ĐIỀU CHỈNH NHỊP. Khi backend ASR có đuôi độ trễ (spike), giữ nhịp cố định
    # sẽ liên tục bỏ nhịp và tranh chấp GPU. Thay vào đó giãn nhịp ra (preview ít hơn
    # nhưng đáng tin hơn) rồi tự thu hẹp lại khi nhanh trở lại.
    preview_adaptive_backoff: bool = True
    preview_slow_ms: float = 350.0        # trung vị preview >= mức này => giãn nhịp
    preview_fast_ms: float = 120.0        # trung vị <= mức này => thu hẹp nhịp
    preview_max_interval_ms: int = 1000   # trần nhịp preview
    # P2.5: tái sử dụng kết quả preview cho commit. MẶC ĐỊNH TẮT vì có thể mất từ
    # cuối câu (trái ưu tiên C4). Chỉ bật sau khi đo WER đạt chênh <= 0.3%.
    preview_reuse_for_commit: bool = False
    preview_reuse_max_delta_sec: float = 0.4

    # --- F-39: chống nghẽn executor ------------------------------------------
    # `_EXECUTOR` (ThreadPoolExecutor) có HÀNG ĐỢI KHÔNG GIỚI HẠN. Nếu một inference
    # native bị chậm/treo, mỗi vòng preview vẫn nộp thêm việc ⇒ hàng đợi giữ hàng nghìn
    # `audio_slice` ⇒ RAM tăng 66-85 MB/s cho tới khi hết bộ nhớ (đã đo được, xem
    # report/audit/05_measurements_and_status.md §12). Giới hạn số inference cùng lúc;
    # vòng preview vượt trần thì BỎ (preview là best-effort, commit không bị bỏ).
    max_inflight_infer: int = 1
    # Đồng hồ cắt ngắn: inference vượt ngần này giây thì yêu cầu native `session.cancel()`
    # để trả về sớm (kèm partial) thay vì giữ `_infer_lock` hàng phút. 0 = TẮT.
    inference_watchdog_sec: float = 8.0
    # Sau khi đã yêu cầu native huỷ, còn chờ thêm ngần này giây để native thoát êm (kèm
    # partial result) trước khi bỏ vòng đó. 0 = bỏ ngay sau khi yêu cầu huỷ.
    inference_watchdog_grace_sec: float = 2.0
    # F-39 (native): nếu RSS tăng quá ngần này MB NGAY TRONG lúc một inference đang chạy
    # thì đó là dấu hiệu phình bộ nhớ trong native/Vulkan (đã đo được +73 GB private) ⇒
    # huỷ inference và bỏ vòng đó. 0 = TẮT.
    rss_runaway_delta_mb: float = 1024.0
    # F-39 (native): nếu RSS tiến trình vượt mốc "lúc nạp model" quá ngần này MB thì đóng
    # phiên native ở cuối vòng đời session để lần sau nạp lại sạch. Bộ nhớ phình nằm trong
    # native/Vulkan (xem báo cáo §12.6) nên đây là cách duy nhất đòi lại mà không patch
    # upstream. 0 = TẮT. Lưu ý: lần nạp lại tốn ~10 s, nên chỉ nên để TẮT nếu không gặp.
    native_recycle_rss_delta_mb: float = 2048.0

    # P2.2: chỉ xin `full_text` từ binding, không materialize segments/words/tokens.
    request_timestamps: bool = False
    # P1.5: đọc session limits và chặn trên audio trước khi feed.
    enforce_session_limits: bool = True

    # Cấu hình Chuẩn hóa Âm Lượng (Speech Normalization)
    normalize_speech: bool = True
    normalize_target_rms: float = 0.10   # Mục tiêu RMS (~ -20 dBFS)
    normalize_target_peak: float = 0.95  # Trần biên độ cực đại tránh méo tiếng
    normalize_max_gain: float = 3.0      # Hệ số khuếch đại tối đa (không kéo nhiễu nền)
    normalize_min_gain: float = 0.3333   # Hệ số nén tối đa (1.0 / max_gain)
    normalize_knee_start: float = 0.025
    normalize_knee_end: float = 0.050
    normalize_log_stats: bool = True


class SentenceConfig(BaseModel):
    """Cấu hình ngắt câu và phân đoạn ngữ nghĩa."""
    max_chars: int = 150
    # P2.3: giữ BẰNG `ASRConfig.preview_window_sec` để cửa sổ preview luôn bao trùm
    # trọn câu hiện tại => preview không mất ngữ cảnh so với commit (nguyên tắc P2).
    max_duration_sec: float = 6.0          # Giới hạn tối đa độ dài 1 câu nói liên tục
    min_words_to_commit: int = 2           # Số từ tối thiểu để gửi sang dịch/TTS (lọc tiếng ậm ừ)
    split_on_stability: bool = True        # Tự động ngắt câu khi preview text ổn định
    stability_duration_sec: float = 0.6    # Thời gian (giây) preview text bất biến (P3: cắt ở ranh giới từ)
    stability_threshold_polls: int = 3     # Số chu kỳ poll tối thiểu xác nhận ổn định
    # C4: sàn thời lượng trước khi cho phép BẬC 3 cắt câu. Nếu không có sàn này, một
    # câu mới bắt đầu mà model trả text ngắn không đổi (ví dụ chỉ nhận được 1-2 từ
    # trong 1 giây đầu) sẽ bị cắt ngay => mất chữ. Chỉ cắt khi câu đã đủ dài.
    stability_min_duration_sec: float = 2.5
    # C4: số từ tối thiểu của preview trước khi cho phép BẬC 3 cắt câu.
    stability_min_words: int = 4
    inactivity_timeout_sec: float = 1.2    # Timeout ép chốt câu nếu không có frame mới
    # P2.1: feature flag cho 3 bậc commit không phải VAD (MAX_DURATION/STABLE_PREFIX/TIMEOUT_FORCE).
    # Đặt False để quay lại hành vi cũ (chỉ chốt theo VAD silence).
    enable_tier234: bool = True
    # P2.7: câu kế tiếp lùi lại bao nhiêu ms để không mất từ ở ranh giới cắt.
    boundary_overlap_ms: int = 250
    # P2.1: nếu mảnh cắt ra quá ngắn (< min_words_to_commit) thì gộp vào câu kế tiếp
    # thay vì để tầng trên lọc bỏ (tránh mất chữ).
    carry_over_short_fragment: bool = True


class TranslationConfig(BaseModel):
    """Cấu hình dịch thuật cục bộ GGUF qua Llama.cpp."""
    base: str = "tencent"  # tencent, tencent-1.8b, xiaomi, gemmax
    enabled: bool = True
    model: Optional[str] = None
    gguf_file: Optional[str] = None
    target_lang: str = "vi"
    source_lang: str = "auto"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_tokens: int = 128
    use_context: bool = False
    context_window: int = 3
    prompt_style: Optional[str] = None
    n_gpu_layers: int = -1
    # P4.3: trước đây các tham số này bị hardcode trong translation/engine.py.
    # A5 (audit Gemini, đã kiểm chứng): KV cache ở đây RẤT nhỏ nên đừng tối ưu.
    # Kiến trúc Hy-MT2-1.8B: 32 layer × 4 KV-head × head_dim 128 ⇒ KV cache f16 =
    # 2×32×n_ctx×4×128×2 byte. n_ctx=512 → **33,6 MB** (không phải "~450 MB" như báo cáo
    # Gemini ước lượng — sai ~13×). Hạ xuống 384 chỉ tiết kiệm ~8 MB ⇒ không đáng đổi.
    n_ctx: int = 512
    n_batch: int = 256
    # A1-1 (Hy3): cùng lý do như `ASRConfig.threads` — llama.cpp chỉ cần vài thread CPU để
    # feed/parse token (phần nặng nằm trên GPU CUDA), nên trên máy ít nhân hạ xuống 3 để
    # nhường nhân cho VAD + audio worklet + decode video của trình duyệt.
    n_threads: int = _AUTO_THREADS
    # P3.3: đẩy token dịch dần về client để bản dịch hiện sớm (~120ms thay vì 250-800ms).
    stream_tokens: bool = True
    # P3.3: throttle gửi partial — tối thiểu bao nhiêu ký tự mới đáng gửi, và tối thiểu
    # bao nhiêu ms giữa hai lần gửi. Tránh nháy tiền tố rác và tránh ngập WebSocket.
    partial_min_chars: int = 3
    partial_min_interval_ms: int = 120
    models_yaml: str = str(TRANSLATION_MODELS_YAML_PATH)
    # Nếu file GGUF của model được chọn chưa có trong `backend/models` thì tự tải từ
    # HuggingFace (repo ghi trong translation_models.yaml). Tải chạy ở luồng nền, KHÔNG
    # bao giờ chạy trên hot path dịch từng câu; tắt bằng cách đặt false.
    auto_download: bool = True


class TTSConfig(BaseModel):
    """Cấu hình tổng hợp giọng nói Voice Cloning OmniVoice."""
    enabled: bool = True
    engine: str = "omnivoice"  # PyTorch native OmniVoice
    model: str = "splendor1811/omnivoice-vietnamese"
    device: str = "cuda:0"
    # A5 (audit Gemini, đã kiểm chứng): `speed != 1.0` BẬT một đường CHẬM —
    # `AudioProcessor.apply_time_stretch()` dùng Phase Vocoder `scipy.signal.stft/istft`
    # thuần CPU (~45-120 ms cho câu 3-5 s, đo theo báo cáo Gemini nhưng CHƯA đo lại ở đây).
    # Ở mặc định 1.0 hàm thoát ngay ở dòng đầu (`abs(rate-1.0) < 0.02`) nên **không** nằm
    # trên hot path. Nếu đổi speed, hãy đo lại `tts.infer_ms` trước khi kết luận có hồi quy.
    speed: float = 1.0
    num_inference_steps: int = 8
    default_voice: str = "speaker_01_0039.wav"
    voices_dir: str = str(VOICES_DIR)
    volume: float = 0.8
    sample_rate: int = 24000


class AudioBufferConfig(BaseModel):
    """Cấu hình Circular Ring Buffer cho Audio Ingress."""
    sample_rate: int = 16000
    capacity_sec: float = 60.0             # Dung lượng cố định 60 giây audio (~960,000 samples Float32)
    chunk_size_samples: int = 400          # Kích thước frame chuẩn cho VAD (25ms @ 16kHz)
    max_speech_segment_sec: float = 30.0   # Độ dài tối đa 1 đoạn phát âm


class MetricsConfig(BaseModel):
    """Cấu hình đo lường hiệu năng thời gian thực."""
    enabled: bool = False
    alert_threshold_ms: float = 300.0
    dump_report_on_disconnect: bool = False
    report_file: str = "metrics_report.json"


class GpuConfig(BaseModel):
    """A2-1: điều phối tranh chấp GPU giữa ASR / dịch / TTS trên MỘT card.

    Vấn đề: cả ba stage đều dùng chung một GPU, không có scheduler. Khi một câu ASR
    `commit` tới đúng lúc dịch/TTS đang chạy, commit phải xếp sau và phụ đề bị spike.

    `GpuArbiter` (backend/core/gpu_scheduler.py) chỉ làm MỘT việc: **trì hoãn việc KHỞI
    ĐỘNG** job ưu tiên thấp khi ASR đang chạy, để nhường GPU cho commit. Nó KHÔNG thể
    preempt một CUDA kernel đang chạy (driver không cho) — nên đây là bài toán **thời
    điểm**, không phải bài toán lock.

    **Mặc định TẮT.** Bật lên rồi mới đo A/B (xem
    `report/audit/KE_HOACH_FIX_LOI_Hy3.md` §4.3 bước A.5) — A2-2 + ASR CUDA có thể đã
    giải quyết phần lớn triệu chứng, và bật scheduler khi không cần thiết chỉ thêm trễ.
    """
    scheduler_enabled: bool = False
    #: Cửa sổ "dành riêng GPU" sau khi một commit ASR bắt đầu (ms). Trong cửa sổ này job
    #: ưu tiên thấp không được KHỞI ĐỘNG. Đây cũng là trần chống starvation: quá hạn thì
    #: dịch/TTS được chạy lại dù commit còn đang chạy.
    commit_reserve_ms: int = 1500
    #: Trần thời gian một job ưu tiên thấp chịu chờ trước khi tự cho phép chạy (ms).
    #: Đảm bảo không bao giờ treo pipeline.
    admit_max_wait_ms: int = 1200
    #: Chu kỳ kiểm tra khi đang chờ (ms). Chỉ dùng trong lúc cửa sổ dành riêng đang mở.
    admit_poll_ms: int = 15


class AppConfig(BaseModel):
    """Cấu hình gốc toàn hệ thống Backend."""
    ws: WSConfig = Field(default_factory=WSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    sentence: SentenceConfig = Field(default_factory=SentenceConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio_buffer: AudioBufferConfig = Field(default_factory=AudioBufferConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)

    def hot_reload(self, updates: Dict[str, Any]) -> None:
        """Cập nhật cấu hình runtime nhanh chóng không cần khởi động lại server."""
        for key, value in updates.items():
            if hasattr(self, key) and isinstance(value, dict):
                sub_cfg = getattr(self, key)
                for sub_k, sub_v in value.items():
                    if hasattr(sub_cfg, sub_k):
                        setattr(sub_cfg, sub_k, sub_v)
            elif hasattr(self, key):
                setattr(self, key, value)


def load_config() -> AppConfig:
    """Khởi tạo cấu hình mặc định."""
    return AppConfig()


config = load_config()
