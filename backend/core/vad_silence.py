"""Dò KHOẢNG LẶNG CHẮC CHẮN (bằng VAD Silero) để cắt khối cho Lookahead Batch ASR.

VÌ SAO CẦN (thay cho bộ dò RMS + `overlap_sec` + trừ chồng lấn)
==============================================================
Bộ cắt khối cũ chọn điểm cắt bằng **năng lượng RMS tương đối** rồi cho khối sau LÙI LẠI
`overlap_sec` (1 s) "cho chắc". Chính khoản lùi đó sinh ra vùng chồng lấn, rồi phía nhận phải TRỪ
vùng chồng lấn ấy — phép trừ dựa vào mốc Forced Aligner ở mép khối, vốn không đủ chính xác, nên nó
ăn mất từ đầu câu:

    [44.22→56.45]   'Or we could do something…'                     ← mất 'Or'
    [67.06→79.06]   "I can't see loving nobody but you…"            ← mất 'I'
    [127.27→141.58] 'Let me sit on the stairs and think about what I did.'
                    → 'think about what I did.'                     ← mất 7 từ

Cách làm ở đây: **cắt khối CHỈ tại khoảng lặng mà VAD xác nhận** (`>= min_silence_ms`, mặc định
1500 ms). Mép khối rơi vào giữa khoảng lặng thật thì:
  * không từ nào bị chẻ đôi ở ranh giới ⇒ **KHÔNG cần** `overlap_sec`,
  * **KHÔNG cần** trừ chồng lấn, **KHÔNG cần** mang mảnh cuối sang khối sau,
  * câu ASR trả về là câu TRỌN VẸN ⇒ không cần hàn gắn/hiệu chỉnh gì thêm.

VÌ SAO CHỌN SILERO (đo thật trên máy chuẩn, RTX 5060 Ti)
=======================================================
| Engine | Cách chạy | Chi phí cho 90 s audio | Hệ số |
|---|---|---|---|
| `firered-vad` | `is_speech()` từng frame 10 ms | 28 700 ms | **0.31× thời gian thực** |
| `silero-vad` | từng cửa sổ 512 mẫu | 1 001 ms | 90× thời gian thực |
| `silero-vad` | **gọi theo LÔ** (256 cửa sổ/lần) | **127 ms** | **~700× thời gian thực** |

FireRed bị chi phối bởi chi phí gọi model cho TỪNG frame 10 ms ⇒ không thể quét toàn vùng tìm kiếm
(18–30 s) cho mỗi khối. Silero nhận lô cửa sổ và trả về xác suất cho cả lô ⇒ quét được **toàn bộ**
vùng tìm kiếm với chi phí không đáng kể. Nhờ vậy module này **không cần** tầng tiền quét năng lượng
làm "mẹo tăng tốc" nữa: VAD là nguồn quyết định duy nhất, đúng như yêu cầu thiết kế.

Engine dự phòng: nếu không nạp được Silero (thiếu file/gói), scanner suy giảm sang dò năng lượng
tương đối và **nói rõ trong log** — không bao giờ chặn pipeline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from backend.config import MODELS_DIR, config
from backend.utils.logger import get_logger

logger = get_logger("core.vad_silence")

#: Cửa sổ bắt buộc của Silero @16 kHz (32 ms).
SILERO_WINDOW_SAMPLES = 512
#: Số cửa sổ mỗi lần gọi model.
#:
#: ⚠️ PHẢI LÀ 1. Silero giữ state NỘI BỘ chạy tiếp giữa các cửa sổ (`_state`, `_context`); khi gọi
#: theo lô > 1, state `[2, batch, 128]` được model xử lý theo cách KHÁC và kết quả đổi hẳn. Đo thật
#: trên 60 s audio đầu (ngưỡng 0.5), đoạn lặng dài nhất theo batch:
#:     batch=1  → 5.50 s   (khớp cách chạy streaming thật của engine)
#:     batch=8  → 1.41 s   ← sai
#:     batch=64 → 0.90 s   ← sai
#:     batch=256→ 6.27 s   ← khác batch=1
#:     batch=1024→6.82 s
#: Vì vậy chỉ dùng batch=1: đúng ngữ nghĩa streaming, và vẫn đủ nhanh (đo được ~90× thời gian thực).
SILERO_BATCH_WINDOWS = 1
#: Bước nhảy của tầng dò năng lượng dự phòng (ms).
ENERGY_FRAME_MS = 20.0


@dataclass
class SilenceGap:
    """Một khoảng lặng (giây, trục thời gian TUYỆT ĐỐI)."""

    start_pts: float
    end_pts: float
    #: `True` khi VAD xác nhận (không có frame tiếng nói nào trong khoảng).
    vad_confirmed: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end_pts - self.start_pts)

    @property
    def center(self) -> float:
        return (self.start_pts + self.end_pts) / 2.0

    def __repr__(self) -> str:  # pragma: no cover - chỉ để log
        tag = "VAD" if self.vad_confirmed else "energy"
        return f"SilenceGap([{self.start_pts:.2f} → {self.end_pts:.2f}] {self.duration:.2f}s, {tag})"


class VADSilenceScanner:
    """Tìm khoảng lặng để cắt khối bằng VAD Silero (chạy theo LÔ, không giữ state phiên).

    Scanner giữ một model JIT **riêng** cho việc quét; `reset_states()` gọi trước mỗi lần quét nên
    không ảnh hưởng state VAD của phiên Pipeline A (đúng ràng buộc "engine sở hữu state machine"):
    state của phiên nằm trong `VADStreamState` do `VADProcessor` giữ, còn đây là một model độc lập.
    """

    def __init__(
        self,
        engine_name: str = "silero-vad",
        sample_rate: int = 16000,
        min_silence_ms: float = 1500.0,
        threshold: Optional[float] = None,
        silence_rel_db: float = 20.0,
        silence_floor_rms: float = 0.004,
    ):
        self.engine_name = (engine_name or "silero-vad").lower().strip()
        self.sample_rate = int(sample_rate)
        self.min_silence_ms = float(min_silence_ms)
        #: Ngưỡng xác suất Silero coi là "có tiếng nói". MẶC ĐỊNH THẤP HƠN 0.5 của thư viện: ngưỡng
        #: 0.5 phân loại tiếng CƯỜI và lời HÁT là "không có tiếng nói" (đo thật: p50 ≈ 0.06 ở đoạn
        #: 3.8–8.5 s và 0.03 ở đoạn nhạc 65–83 s), nên khoảng lặng 0.5 dài 18 s giữa bài hát. Đó
        #: KHÔNG phải khoảng lặng thật để cắt khối. Hạ ngưỡng làm VAD bám sát "có tiếng người" hơn.
        self.threshold = float(
            threshold if threshold is not None
            else getattr(config.lookahead, "batch_vad_silence_threshold", 0.30)
        )
        self.silence_rel_db = float(silence_rel_db)
        self.silence_floor_rms = float(silence_floor_rms)
        self._lock = threading.RLock()
        self._model = None
        self._use_gpu = bool(getattr(config.vad.silero, "use_gpu", False))
        #: Lý do VAD không dùng được (rỗng = dùng được).
        self.vad_unavailable_reason: str = ""
        #: Thống kê chẩn đoán.
        self.last_scan_ms: float = 0.0
        self.last_scanned_sec: float = 0.0
        #: Cache xác suất theo LÔ: mỗi phần tử là `(t0, hop, probs)`. Lần quét sau nối tiếp phần đã
        #: có nên chi phí trung bình chỉ còn phần audio MỚI (đo thật: quét 30 s ≈ 330 ms).
        self._prob_cache: List[tuple] = []

    # ------------------------------------------------------------------ model
    def _load_model(self):
        if self._model is not None:
            return self._model
        from silero_vad.utils_vad import init_jit_model

        path = Path(MODELS_DIR) / "silero_vad.jit"
        if not path.exists():
            try:
                from backend.vad.engines import SileroVADEngine

                SileroVADEngine.prepare_files()
            except Exception:  # noqa: BLE001
                pass
        if not path.exists():
            raise FileNotFoundError(f"Không thấy model Silero VAD tại {path}")
        model = init_jit_model(str(path))
        if self._use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    model = model.cuda()
            except Exception:  # noqa: BLE001
                pass
        self._model = model
        return model

    def prewarm(self) -> None:
        """Nạp model + chạy thử một lô nhỏ (BLOCKING — gọi từ thread nền)."""
        try:
            model = self._load_model()
            import torch

            dummy = torch.zeros(SILERO_BATCH_WINDOWS, SILERO_WINDOW_SAMPLES)
            if self._use_gpu:
                dummy = dummy.cuda()
            with torch.no_grad():
                model.reset_states()
                model(dummy, self.sample_rate)
            self.vad_unavailable_reason = ""
            logger.info(
                f"Bộ dò khoảng lặng: Silero VAD sẵn sàng (ngưỡng={self.threshold}, "
                f"lô={SILERO_BATCH_WINDOWS} cửa sổ, min_silence={self.min_silence_ms:.0f}ms).",
                extra={"module_tag": "VAD"},
            )
        except Exception as exc:  # noqa: BLE001
            self.vad_unavailable_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"Bộ dò khoảng lặng KHÔNG nạp được Silero VAD ({exc}) — suy giảm sang dò năng lượng.",
                extra={"module_tag": "VAD"},
            )

    # ------------------------------------------------------------------ quét
    def speech_mask(self, pcm: np.ndarray, base_pts: float = 0.0) -> np.ndarray:
        """Xác suất tiếng nói theo từng cửa sổ 512 mẫu (Silero, chạy tuần tự đúng streaming).

        Trả mảng rỗng khi không có model ⇒ caller dùng đường dự phòng năng lượng.

        `base_pts` là mốc tuyệt đối của mẫu đầu tiên trong `pcm`; dùng cho cache theo lô nên lần
        quét sau (vùng dịch về phía trước) chỉ phải tính phần audio MỚI.
        """
        hop_sec = SILERO_WINDOW_SAMPLES / float(self.sample_rate)
        arr = np.asarray(pcm, dtype=np.float32)
        n_win = arr.size // SILERO_WINDOW_SAMPLES
        if n_win <= 0:
            return np.zeros(0, dtype=np.float32)

        # ── Nối tiếp cache: bỏ các lô nằm TRƯỚC `base_pts`, giữ lô phủ `base_pts`.
        with self._lock:
            usable: List[tuple] = []
            for t0, hop, probs in self._prob_cache:
                t1 = t0 + probs.size * hop
                if t1 <= float(base_pts) + 1e-6:
                    continue
                usable.append((t0, hop, probs))
            self._prob_cache = usable
            cached_end = 0.0
            if usable:
                t0, hop, probs = usable[-1]
                cached_end = t0 + probs.size * hop
            aligned = max(float(base_pts), cached_end)
            aligned = float(int(round(aligned / hop_sec)) * hop_sec)

        start_idx = int(round((aligned - float(base_pts)) / hop_sec))
        if start_idx < 0:
            start_idx = 0
        new_win = n_win - start_idx
        if new_win > 0:
            probs_new = self._run_windows(arr[start_idx * SILERO_WINDOW_SAMPLES : n_win * SILERO_WINDOW_SAMPLES])
            if probs_new.size:
                with self._lock:
                    self._prob_cache.append((aligned, hop_sec, probs_new))

        # ── Ghép cache thành mảng xác suất cho đúng `[base_pts, base_pts + n_win*hop]`.
        out = np.zeros(n_win, dtype=np.float32)
        filled = np.zeros(n_win, dtype=bool)
        with self._lock:
            for t0, hop, probs in self._prob_cache:
                i0 = int(round((t0 - float(base_pts)) / hop_sec))
                for k in range(probs.size):
                    i = i0 + k
                    if 0 <= i < n_win and not filled[i]:
                        out[i] = probs[k]
                        filled[i] = True
        return out

    def _run_windows(self, samples: np.ndarray) -> np.ndarray:
        """Chạy model Silero trên một dải mẫu (bội số của 512) theo đúng thứ tự thời gian."""
        try:
            import torch

            model = self._load_model()
        except Exception as exc:  # noqa: BLE001
            if not self.vad_unavailable_reason:
                self.vad_unavailable_reason = f"{type(exc).__name__}: {exc}"
            return np.zeros(0, dtype=np.float32)

        n_win = samples.size // SILERO_WINDOW_SAMPLES
        if n_win <= 0:
            return np.zeros(0, dtype=np.float32)
        windows = torch.from_numpy(np.ascontiguousarray(samples[: n_win * SILERO_WINDOW_SAMPLES])).reshape(
            n_win, SILERO_WINDOW_SAMPLES
        )
        if self._use_gpu:
            windows = windows.cuda()

        probs: List[np.ndarray] = []
        t0 = time.perf_counter()
        with self._lock:
            with torch.no_grad():
                model.reset_states()
                for i in range(0, n_win, SILERO_BATCH_WINDOWS):
                    chunk = windows[i : i + SILERO_BATCH_WINDOWS]
                    out = model(chunk, self.sample_rate)
                    probs.append(out.detach().reshape(-1).cpu().numpy().astype(np.float32))
        self.last_scan_ms = (time.perf_counter() - t0) * 1000.0
        self.last_scanned_sec += n_win * SILERO_WINDOW_SAMPLES / float(self.sample_rate)
        return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)

    def _energy_speech_mask(self, pcm: np.ndarray) -> np.ndarray:
        """Đường DỰ PHÒNG khi không có VAD: năng lượng tương đối (chỉ để không chặn pipeline)."""
        arr = np.asarray(pcm, dtype=np.float32)
        frame = int(round(ENERGY_FRAME_MS * self.sample_rate / 1000.0))
        n = arr.size // frame
        if n <= 0:
            return np.zeros(0, dtype=bool)
        view = arr[: n * frame].reshape(n, frame)
        rms = np.sqrt(np.mean(view * view, axis=1))
        level = float(np.percentile(rms, 90)) if rms.size else 0.0
        thr = max(
            float(self.silence_floor_rms),
            level * (10.0 ** (-self.silence_rel_db / 20.0)),
        ) if level > 0 else float(self.silence_floor_rms)
        return rms >= thr

    def find_silence_gaps(
        self,
        pcm: np.ndarray,
        base_pts: float,
        min_silence_ms: Optional[float] = None,
    ) -> List[SilenceGap]:
        """Mọi khoảng lặng đạt ngưỡng (đã xác nhận bằng VAD khi VAD dùng được)."""
        if pcm is None or len(pcm) == 0:
            return []
        need_ms = float(min_silence_ms if min_silence_ms is not None else self.min_silence_ms)

        probs = self.speech_mask(pcm, base_pts=float(base_pts))
        if probs.size > 0:
            mask = probs >= self.threshold
            confirmed = True
            frame_sec = SILERO_WINDOW_SAMPLES / float(self.sample_rate)
        else:
            mask = self._energy_speech_mask(pcm)
            confirmed = False
            frame_sec = ENERGY_FRAME_MS / 1000.0

        if mask.size == 0:
            return []

        gaps: List[SilenceGap] = []
        start: Optional[int] = None
        for i, is_speech in enumerate(mask):
            if not is_speech:
                if start is None:
                    start = i
            elif start is not None:
                self._append(gaps, start, i, base_pts, frame_sec, need_ms, confirmed)
                start = None
        if start is not None:
            self._append(gaps, start, len(mask), base_pts, frame_sec, need_ms, confirmed)
        return gaps

    @staticmethod
    def _append(
        gaps: List[SilenceGap],
        start_idx: int,
        end_idx: int,
        base_pts: float,
        frame_sec: float,
        need_ms: float,
        confirmed: bool,
    ) -> None:
        if (end_idx - start_idx) * frame_sec * 1000.0 < need_ms:
            return
        gaps.append(
            SilenceGap(
                start_pts=float(base_pts) + start_idx * frame_sec,
                end_pts=float(base_pts) + end_idx * frame_sec,
                vad_confirmed=confirmed,
            )
        )

    def find_last_silence(
        self,
        pcm: np.ndarray,
        base_pts: float,
        region_start_pts: float,
        region_end_pts: float,
        min_silence_ms: Optional[float] = None,
    ) -> Optional[SilenceGap]:
        """Khoảng lặng CUỐI CÙNG (xa nhất về phía trước) đạt ngưỡng, giao với vùng tìm kiếm.

        VÌ SAO LẤY KHOẢNG CUỐI CÙNG (chứ không phải dài nhất): khối ASR càng dài càng tốt cho ngữ
        cảnh, nên ta muốn đẩy mép khối xa nhất có thể mà vẫn dừng ở một khoảng lặng CHẮC CHẮN. Trong
        ví dụ 30 s audio có nhiều khoảng lặng > 1.5 s, ta cắt ở khoảng CUỐI CÙNG và gửi TRỌN đoạn
        audio từ đầu khối tới mép đó (bao gồm cả khoảng lặng) cho ASR.

        Trả `None` khi không có khoảng lặng nào đạt ngưỡng trong vùng — khi đó caller phải quyết
        định (chờ thêm audio, hoặc dùng đường dự phòng).
        """
        gaps = self.find_silence_gaps(pcm, base_pts, min_silence_ms=min_silence_ms)
        inside = [
            g for g in gaps
            if g.end_pts > float(region_start_pts) and g.start_pts < float(region_end_pts)
        ]
        if not inside:
            return None
        # `find_silence_gaps` trả theo thứ tự thời gian ⇒ phần tử cuối là khoảng lặng XA NHẤT.
        return inside[-1]


__all__ = ["SilenceGap", "VADSilenceScanner"]