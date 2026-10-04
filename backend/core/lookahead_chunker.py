"""LookaheadChunker: Bộ cắt khối âm thanh cho tuyến OFFLINE_BATCH (Pipeline B v3).

VAI TRÒ TRONG KIẾN TRÚC (đọc trước khi sửa)
==========================================
Điểm khác biệt cốt lõi giữa Pipeline A và Pipeline B v3 nằm ở **đầu vào của ASR**, không phải ở
việc "có cắt câu hay không":

* Pipeline A: VAD + CommitManager quyết định **cửa sổ audio đưa vào model**. Cắt ở đâu thì model
  chỉ nghe được tới đó ⇒ mất ngữ cảnh âm học ⇒ WER/CER tăng.
* Pipeline B v3: ASR **luôn nhận một khối dài liên tục** (mục tiêu ~15 s, trải tới 30 s) để có
  ngữ cảnh đầy đủ ⇒ CER/WER thấp. Ranh giới khối ở đây chỉ là **ranh giới GIẢI MÃ**, KHÔNG phải
  ranh giới câu. Câu/phụ đề được cắt *bên trong* bản phiên âm đầy đủ đó bằng dấu câu + mốc từ của
  Qwen3-ForcedAligner (xem `ForcedAlignerService.group_words_to_subtitles`).

Vì vậy mọi thay đổi ở file này phải phục vụ: **khối càng dài & ranh giới càng "sạch" càng tốt**,
và mép khối phải VÔ HÌNH ở đầu ra. Mép khối vô hình nhờ hai cơ chế phối hợp:
  1. `overlap_sec` — khối sau lấy lùi lại một đoạn để không mất từ ở ranh giới.
  2. Phía nhận (`LookaheadSessionState._offline_batch_loop`) **BẮT BUỘC** trừ phần chồng lấn ở
     tầng từ trước khi gom câu. Nếu tầng đó không trừ, từ ở ranh giới sẽ được phát hai lần —
     đúng sự cố "cuối câu này là đầu của câu sau" (2026-10-02).
     ⇒ Nếu bạn bỏ bước trừ ở phía nhận thì ĐỪNG bật overlap ở đây.

Nhiệm vụ của bộ cắt khối:
1. Quét khoảng lặng trong dải [T + min_window, T + search_max] trên RAM timeline.
2. Ưu tiên ranh giới THẬT: khoảng lặng MẠNH (>= strong_silence_ms) ở đâu trong dải cũng tốt hơn
   việc cưỡng bức cắt — vì nó cho phép khối dài hơn mà mép khối vẫn sạch.
3. Fallback khi nói liên tục không ngừng nghỉ:
   - Điểm trũng năng lượng KÉO DÀI (>= min_rms_hold_ms) và thấp hơn `min_rms_ratio` × trung vị
     RMS ⇒ ranh giới mềm (kèm overlap).
   - Cưỡng bức: cắt tại điểm trũng nhất **quanh độ dài lý tưởng** (`dip_search_sec`) kèm overlap.
4. Xử lý điều kiện biên:
   - Fast-Bootstrap sau khi tua video (Seek): cửa sổ ngắn 3.0s - 4.0s.
   - Hết luồng (End of stream / Stream ended): trích xuất trọn vẹn phần còn lại.

SIẾT LẠI NGÀY 2026-10-02 (sau chẩn đoán "cuối câu này là đầu của câu sau")
-------------------------------------------------------------------------
Bản cũ dò "im lặng" bằng **ngưỡng RMS TUYỆT ĐỐI** (`silence_rms_threshold = 0.015`) và chấp nhận
fallback `min_rms` chỉ với điều kiện `min_rms < 0.015 * 2 = 0.03`. Với audio phim/talkshow thật
(luôn có nhạc nền / room tone), **không frame nào** xuống dưới 0.015 ⇒ nhánh khoảng lặng không bao
giờ chạy; còn 0.03 thì gần như frame nào cũng đạt ⇒ khối bị cắt tại điểm trũng năng lượng BẤT KỲ
(khe giữa hai từ, tiếng bật hơi, khe đóng của phụ âm) — tức là **cắt giữa câu**. Kèm theo, nhánh
cưỡng bức cắt đúng ở MÉP dải (T + 18 s) nên chắc chắn chẻ đôi từ đang nói, và dải quét chỉ 6 s nên
gần như không bao giờ gặp khoảng lặng thật.

Bản này:
  * Ngưỡng lặng **TƯƠNG ĐỐI** theo mức chương trình: `p90(RMS) − silence_rel_db`, sàn
    `silence_floor_rms` (xem `silence_threshold`) — không còn phụ thuộc âm lượng tuyệt đối.
  * **Hai mức ranh giới**: `>= strong_silence_ms` (450 ms) là MẠNH (cắt ngay), `>= min_silence_ms`
    (250 ms) là YẾU (chỉ cắt khi gần mốc lý tưởng).
  * **Nới dải tìm kiếm** tới `search_max_sec` (tuyến batch: 30 s — đúng cửa sổ tối ưu 15–30 s của
    Qwen3-ASR) ⇒ khối dài hơn, ít ranh giới hơn. Không còn cắt cứng ở 18 s.
  * Cắt cưỡng bức tại **điểm trũng quanh độ dài lý tưởng**, không phải ở mép dải.

GHI CHÚ: tham số `vad_engine` của bản cũ đã được BỎ vì nó được nhận vào nhưng **không bao giờ
được đọc** (dead code). Bộ dò hiện dùng năng lượng tương đối. Cố ý KHÔNG dùng VAD/CommitManager để
chốt câu ở đây: làm vậy là kéo tuyến batch về đúng cách ngắt câu của Pipeline A và mất hết lợi thế
ngữ cảnh dài của Qwen3-ASR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from backend.core.lookahead_timeline import ContinuousAudioTimeline
from backend.utils.logger import get_logger

logger = get_logger("core.lookahead_chunker")

#: Độ dài khung (ms) dùng để đo mức năng lượng / phát hiện điểm trũng.
_LEVEL_FRAME_MS = 20.0


@dataclass
class ChunkSlice:
    pts_start: float
    pts_end: float
    pcm: np.ndarray
    is_silence_boundary: bool
    fallback_mode: str  # "silence_gap", "min_rms", "forced_overlap", "end_of_stream", "bootstrap"
    next_read_pts: float
    silence_gap_ms: float = 0.0
    #: Độ mạnh ranh giới: "strong" | "weak" | "" (không có ranh giới thật) — dùng để chẩn đoán.
    boundary_strength: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.pts_end - self.pts_start)

    def __repr__(self) -> str:
        return (
            f"ChunkSlice([{self.pts_start:.3f}s -> {self.pts_end:.3f}s] "
            f"dur={self.duration:.2f}s, mode='{self.fallback_mode}', "
            f"silence={self.silence_gap_ms:.0f}ms, strength='{self.boundary_strength}', "
            f"next={self.next_read_pts:.3f}s)"
        )


class LookaheadChunker:
    """Bộ cắt khối âm thanh dựa vào khoảng lặng tương đối và năng lượng âm học."""

    def __init__(
        self,
        timeline: ContinuousAudioTimeline,
        sample_rate: int = 16000,
        target_window_sec: float = 15.0,
        min_window_sec: float = 12.0,
        max_window_sec: float = 18.0,
        min_silence_ms: float = 250.0,
        silence_rms_threshold: float = 0.015,
        overlap_sec: float = 1.0,
        # ── Siết lại 2026-10-02 ────────────────────────────────────────────────
        search_max_sec: Optional[float] = None,
        strong_silence_ms: float = 450.0,
        silence_rel_db: float = 22.0,
        silence_floor_rms: float = 0.004,
        min_rms_ratio: float = 0.35,
        min_rms_hold_ms: float = 120.0,
        dip_search_sec: float = 3.0,
        adaptive_max_sec: Optional[float] = None,
        adaptive_threshold_sec: float = 30.0,
    ):
        self.timeline = timeline
        self.sample_rate = int(sample_rate)
        #: Độ dài LÝ TƯỞNG của một khối (mốc để chấm điểm ranh giới).
        self.target_window_sec = float(target_window_sec)
        #: Không bao giờ cắt khối ngắn hơn mức này.
        self.min_window_sec = float(min_window_sec)
        #: Trần cũ (18 s) — giữ lại để tương thích tham số, không còn là trần cắt.
        self.max_window_sec = float(max_window_sec)
        self.min_silence_ms = float(min_silence_ms)
        #: Ngưỡng RMS tuyệt đối CŨ — chỉ còn dùng khi không đo được mức chương trình.
        self.silence_rms_threshold = float(silence_rms_threshold)
        self.overlap_sec = float(overlap_sec)
        #: Trần MỀM khi tìm ranh giới. Mặc định = trần cũ để caller cũ giữ nguyên hành vi; tuyến
        #: batch truyền `batch_search_max_sec` (30 s) để có cơ hội gặp khoảng lặng THẬT.
        search_max = float(search_max_sec) if search_max_sec else float(max_window_sec)
        self.search_max_sec = max(float(min_window_sec), search_max)
        #: Tự động mở rộng khi browser buffer có sẵn (>= adaptive_threshold_sec)
        self.adaptive_max_sec = max(self.search_max_sec, float(adaptive_max_sec)) if adaptive_max_sec else None
        self.adaptive_threshold_sec = float(adaptive_threshold_sec)
        #: Khoảng lặng >= mức này là ranh giới MẠNH (cắt ngay, không cần gần mốc lý tưởng).
        self.strong_silence_ms = max(float(min_silence_ms), float(strong_silence_ms))
        #: Ngưỡng lặng TƯƠNG ĐỐI: p90(mức chương trình) − bao nhiêu dB.
        self.silence_rel_db = float(silence_rel_db)
        self.silence_floor_rms = float(silence_floor_rms)
        #: Điểm trũng chỉ được coi là ranh giới mềm khi < `min_rms_ratio` × trung vị RMS...
        self.min_rms_ratio = float(min_rms_ratio)
        #: ... và kéo dài tối thiểu ngần này ms.
        self.min_rms_hold_ms = float(min_rms_hold_ms)
        #: Khi buộc phải cắt: chỉ tìm điểm trũng trong [ideal − dip_search_sec, ideal + dip_search_sec]
        #: để khối không bị ngắn đi (mất ngữ cảnh) mà vẫn cắt vào chỗ ít năng lượng.
        self.dip_search_sec = max(0.0, float(dip_search_sec))

    # ────────────────────────────────────────────────────────── đo năng lượng
    def _frame_rms(self, pcm: np.ndarray, frame_ms: float = _LEVEL_FRAME_MS) -> np.ndarray:
        """RMS từng khung `frame_ms` của mảng PCM (float32)."""
        if pcm is None or len(pcm) == 0:
            return np.zeros(0, dtype=np.float32)
        frame_samples = int(round(frame_ms * self.sample_rate / 1000.0))
        if frame_samples <= 0 or len(pcm) < frame_samples:
            return np.zeros(0, dtype=np.float32)
        num_frames = len(pcm) // frame_samples
        arr = np.asarray(pcm[: num_frames * frame_samples], dtype=np.float32).reshape(
            num_frames, frame_samples
        )
        return np.sqrt(np.mean(arr * arr, axis=1))

    def silence_threshold(self, pcm: np.ndarray) -> float:
        """Ngưỡng lặng TƯƠNG ĐỐI theo mức chương trình: `p90(RMS) − silence_rel_db` dB.

        Vì sao: ngưỡng tuyệt đối (0.015) chỉ đúng với một mức thu âm duy nhất. Audio phim có nhạc
        nền/room tone nằm trên 0.015 gần như suốt thời lượng ⇒ không bao giờ thấy "im lặng"; còn
        audio thu nhỏ lại bị coi là im lặng toàn bộ. Lấy mức chương trình (p90) làm mốc thì ngưỡng
        tự bám theo cả hai trường hợp.
        """
        rms = self._frame_rms(pcm)
        if rms.size == 0:
            return float(self.silence_rms_threshold)
        level = float(np.percentile(rms, 90))
        if level <= 0.0:
            return float(self.silence_rms_threshold)
        thr = level * (10.0 ** (-self.silence_rel_db / 20.0))
        return max(float(self.silence_floor_rms), float(thr))

    def scan_silence_gaps(
        self,
        pcm: np.ndarray,
        base_pts: float,
        frame_ms: float = 20.0,
        threshold: Optional[float] = None,
    ) -> List[Tuple[float, float, float]]:
        """Quét tìm các khoảng lặng liên tục trong mảng PCM.

        Args:
            pcm: PCM của vùng cần quét.
            base_pts: PTS tuyệt đối của mẫu đầu tiên trong `pcm`.
            frame_ms: độ dài khung phân tích.
            threshold: ngưỡng RMS coi là im lặng (mặc định: tự đo bằng `silence_threshold`).

        Returns:
            Danh sách các tuple `(gap_start_pts, gap_end_pts, gap_duration_ms)`.
        """
        if pcm is None or len(pcm) == 0:
            return []

        frame_samples = int(round(frame_ms * self.sample_rate / 1000.0))
        if frame_samples <= 0 or len(pcm) < frame_samples:
            return []

        thr = float(threshold) if threshold is not None else self.silence_threshold(pcm)
        num_frames = len(pcm) // frame_samples
        gaps: List[Tuple[float, float, float]] = []

        curr_gap_start: Optional[float] = None
        min_silence_sec = self.min_silence_ms / 1000.0

        for f_idx in range(num_frames):
            frame = pcm[f_idx * frame_samples : (f_idx + 1) * frame_samples]
            f_start_pts = base_pts + (f_idx * frame_samples) / self.sample_rate

            # Tính năng lượng RMS
            rms = float(np.sqrt(np.mean(np.asarray(frame, dtype=np.float32) ** 2)))
            is_silent = rms < thr

            if is_silent:
                if curr_gap_start is None:
                    curr_gap_start = f_start_pts
            else:
                if curr_gap_start is not None:
                    gap_dur = f_start_pts - curr_gap_start
                    if gap_dur >= min_silence_sec:
                        gaps.append((curr_gap_start, f_start_pts, gap_dur * 1000.0))
                    curr_gap_start = None

        if curr_gap_start is not None:
            gap_dur = (base_pts + (num_frames * frame_samples) / self.sample_rate) - curr_gap_start
            if gap_dur >= min_silence_sec:
                gaps.append((curr_gap_start, curr_gap_start + gap_dur, gap_dur * 1000.0))

        return gaps

    @staticmethod
    def _runs(mask: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
        """Các đoạn `True` liên tiếp có độ dài >= `min_len`, dạng `(start_index, length)`."""
        runs: List[Tuple[int, int]] = []
        start = 0
        length = 0
        for i, flag in enumerate(mask):
            if flag:
                if length == 0:
                    start = i
                length += 1
            else:
                if length >= min_len:
                    runs.append((start, length))
                length = 0
        if length >= min_len:
            runs.append((start, length))
        return runs

    def find_optimal_cut_point(
        self,
        pcm_search: np.ndarray,
        search_start_pts: float,
        ideal_cut_pts: float,
        threshold: Optional[float] = None,
    ) -> Tuple[float, bool, str, float]:
        """Tìm điểm cắt tối ưu trong vùng tìm kiếm.

        Thứ tự ưu tiên:
          1. Khoảng lặng MẠNH (>= `strong_silence_ms`) gần mốc lý tưởng nhất — khối được phép dài
             hơn mốc lý tưởng để đổi lấy một ranh giới SẠCH (giữ trọn ngữ cảnh cho ASR).
          2. Khoảng lặng YẾU (>= `min_silence_ms`) — chấm điểm theo độ dài và khoảng cách.
          3. Điểm trũng năng lượng KÉO DÀI (>= `min_rms_hold_ms`) thấp hơn `min_rms_ratio` × trung
             vị RMS, gần mốc lý tưởng nhất ⇒ ranh giới mềm (kèm overlap).
          4. Cưỡng bức: điểm trũng nhất QUANH mốc lý tưởng (`dip_search_sec`) — KHÔNG cắt ở mép dải
             như bản cũ (mép dải chắc chắn rơi vào giữa từ đang nói).

        Returns:
            Tuple `(cut_pts, is_silence_boundary, fallback_mode, silence_gap_ms)`.
        """
        search_dur = len(pcm_search) / self.sample_rate if pcm_search is not None else 0.0
        search_end_pts = search_start_pts + search_dur
        if search_dur <= 0.0:
            return float(search_end_pts), False, "forced_overlap", 0.0

        frame_sec = _LEVEL_FRAME_MS / 1000.0
        thr = float(threshold) if threshold is not None else self.silence_threshold(pcm_search)

        # 1 + 2. Khoảng lặng theo mức tương đối
        gaps = self.scan_silence_gaps(pcm_search, base_pts=search_start_pts, threshold=thr)
        if gaps:
            strong = [g for g in gaps if g[2] >= self.strong_silence_ms]
            if strong:
                # Khoảng lặng thật là bằng chứng người nói đã ngừng ⇒ không cần "thưởng" độ dài,
                # chỉ cần gần mốc lý tưởng nhất.
                g_start, g_end, g_ms = min(
                    strong, key=lambda g: abs((g[0] + g[1]) / 2.0 - ideal_cut_pts)
                )
                return float((g_start + g_end) / 2.0), True, "silence_gap", float(g_ms)

            best_gap = None
            best_score = float("-inf")
            for g_start, g_end, g_ms in gaps:
                g_mid = (g_start + g_end) / 2.0
                dist = abs(g_mid - ideal_cut_pts)
                # Score cao khi: khoảng lặng dài và gần mốc lý tưởng
                score = (g_ms / 1000.0) * 2.0 - dist
                if score > best_score:
                    best_score = score
                    best_gap = (g_start, g_end, g_ms)

            if best_gap is not None:
                g_start, g_end, g_ms = best_gap
                return float((g_start + g_end) / 2.0), True, "silence_gap", float(g_ms)

        # 3 + 4. Không có khoảng lặng thật ⇒ dùng năng lượng âm học
        rms = self._frame_rms(pcm_search)
        if rms.size:
            median_rms = float(np.median(rms))

            # 3. Điểm trũng KÉO DÀI (ranh giới mềm)
            if median_rms > 0.0:
                need_frames = max(1, int(round(self.min_rms_hold_ms / _LEVEL_FRAME_MS)))
                runs = self._runs(rms < self.min_rms_ratio * median_rms, need_frames)
                if runs:
                    def _run_mid(run: Tuple[int, int]) -> float:
                        return search_start_pts + (run[0] + run[1] / 2.0) * frame_sec

                    run = min(runs, key=lambda r: abs(_run_mid(r) - ideal_cut_pts))
                    local = rms[run[0] : run[0] + run[1]]
                    idx = run[0] + int(np.argmin(local))
                    return float(search_start_pts + (idx + 0.5) * frame_sec), False, "min_rms", 0.0

            # 4. Cưỡng bức: trũng nhất QUANH độ dài lý tưởng (giữ khối không ngắn đi)
            lo = int(max(0.0, round((ideal_cut_pts - self.dip_search_sec - search_start_pts) / frame_sec)))
            hi = int(
                min(float(len(rms)), round((ideal_cut_pts + self.dip_search_sec - search_start_pts) / frame_sec))
            )
            if hi - lo < 2:
                lo, hi = 0, len(rms)
            idx = lo + int(np.argmin(rms[lo:hi]))
            cut_pts = search_start_pts + (idx + 0.5) * frame_sec
            cut_pts = min(max(cut_pts, search_start_pts + 0.10), search_end_pts - 0.10)
            return float(cut_pts), False, "forced_overlap", 0.0

        # 5. Không đo được năng lượng (dải quá ngắn): cắt ở cuối vùng tìm kiếm
        return float(search_end_pts), False, "forced_overlap", 0.0

    def next_chunk(
        self,
        from_pts: float,
        fast_bootstrap: bool = False,
        is_stream_end: bool = False,
    ) -> Optional[ChunkSlice]:
        """Trích xuất khối audio tiếp theo tính từ `from_pts`.

        Args:
            from_pts: Mốc thời gian tuyệt đối (PTS giây) bắt đầu lấy audio.
            fast_bootstrap: Bật khi vừa Tua video (Seek) - lấy nhanh 3.0s - 4.0s để xuất câu đầu tiên tức thì.
            is_stream_end: Bật khi video đã kết thúc hoặc không còn nhận thêm audio.

        Returns:
            ChunkSlice nếu có đủ audio để xử lý, hoặc None nếu cần chờ buffer nạp thêm.
        """
        buffered_end = self.timeline.buffered_end_from(from_pts)
        if buffered_end is None or buffered_end <= from_pts + 0.05:
            return None

        available_sec = buffered_end - from_pts

        # 1. KỊCH BẢN FAST-BOOTSTRAP (Sau khi tua)
        if fast_bootstrap:
            bootstrap_min = 2.5
            bootstrap_max = 4.0
            if available_sec < bootstrap_min and not is_stream_end:
                return None
            target_dur = min(available_sec, bootstrap_max)
            audio_res = self.timeline.get_audio_range(from_pts, target_dur)
            if audio_res is None:
                return None
            pts_start, pcm = audio_res
            cut_pts = pts_start + len(pcm) / self.sample_rate
            return ChunkSlice(
                pts_start=pts_start,
                pts_end=cut_pts,
                pcm=pcm,
                is_silence_boundary=False,
                fallback_mode="bootstrap",
                next_read_pts=cut_pts,
                silence_gap_ms=0.0,
            )

        # 2. KIỂM TRA ĐỦ AUDIO ĐỂ CẮT CỬA SỔ CHUẨN
        if available_sec < self.min_window_sec:
            if not is_stream_end:
                # Chưa đủ audio, đợi buffer nạp thêm để đảm bảo ngữ cảnh offline
                return None
            # Hết stream: lấy nốt phần còn lại
            audio_res = self.timeline.get_audio_range(from_pts, available_sec)
            if audio_res is None:
                return None
            pts_start, pcm = audio_res
            cut_pts = pts_start + len(pcm) / self.sample_rate
            return ChunkSlice(
                pts_start=pts_start,
                pts_end=cut_pts,
                pcm=pcm,
                is_silence_boundary=False,
                fallback_mode="end_of_stream",
                next_read_pts=cut_pts,
                silence_gap_ms=0.0,
            )

        # 3. QUÉT TÌM ĐIỂM CẮT TỐI ƯU
        # Tự động mở rộng dải tìm ranh giới khi browser buffer có sẵn dồi dào
        effective_search_max = self.search_max_sec
        effective_target = self.target_window_sec
        if self.adaptive_max_sec and available_sec >= self.adaptive_threshold_sec:
            effective_search_max = min(float(self.adaptive_max_sec), available_sec)
            effective_target = min(effective_search_max - 2.0, max(self.target_window_sec, effective_search_max * 0.65))

        search_start_pts = from_pts + self.min_window_sec
        search_end_pts = min(buffered_end, from_pts + effective_search_max)
        ideal_cut_pts = min(from_pts + effective_target, search_end_pts)

        # MỘT lần đọc duy nhất vùng [from_pts, search_end_pts]: vừa để đo MỨC CHƯƠNG TRÌNH (ngưỡng
        # lặng tương đối), vừa là nguồn cho vùng quét (cắt lát, không copy thêm). Đo mức trên toàn
        # khối chứ không phải chỉ dải quét: dải quét có thể gần như im lặng hoàn toàn, khi đó p90
        # của riêng nó sụp về ~0 và ta không bao giờ nhận ra khoảng lặng.
        region_res = self.timeline.get_audio_range(from_pts, max(0.1, search_end_pts - from_pts))
        if region_res is None:
            cut_pts = float(ideal_cut_pts)
            is_silence = False
            fallback_mode = "forced_overlap"
            silence_ms = 0.0
        else:
            region_pcm = region_res[1]
            threshold = self.silence_threshold(region_pcm)
            offset = int(round((search_start_pts - from_pts) * self.sample_rate))
            search_pcm = region_pcm[offset:] if offset > 0 else region_pcm
            cut_pts, is_silence, fallback_mode, silence_ms = self.find_optimal_cut_point(
                search_pcm,
                search_start_pts=search_start_pts,
                ideal_cut_pts=ideal_cut_pts,
                threshold=threshold,
            )

        boundary_strength = (
            "" if not is_silence
            else ("strong" if float(silence_ms) >= self.strong_silence_ms else "weak")
        )
        # Không bao giờ cắt trước `min_window` và không bao giờ vượt trần dải tìm kiếm.
        cut_pts = min(max(float(cut_pts), search_start_pts), search_end_pts)

        total_take_sec = max(0.1, cut_pts - from_pts)
        final_res = self.timeline.get_audio_range(from_pts, total_take_sec)
        if final_res is None:
            return None

        actual_start, chunk_pcm = final_res
        actual_end = actual_start + len(chunk_pcm) / self.sample_rate

        # Cắt KHÔNG dựa trên khoảng lặng (min_rms/forced) ⇒ khối sau lấy lùi `overlap_sec` để không
        # mất từ ở ranh giới. PHÍA NHẬN BẮT BUỘC trừ phần chồng lấn ở tầng từ (xem docstring đầu file).
        if fallback_mode in ("forced_overlap", "min_rms"):
            next_start = max(actual_start + 1.0, actual_end - self.overlap_sec)
        else:
            next_start = actual_end

        return ChunkSlice(
            pts_start=actual_start,
            pts_end=actual_end,
            pcm=chunk_pcm,
            is_silence_boundary=is_silence,
            fallback_mode=fallback_mode,
            next_read_pts=float(next_start),
            silence_gap_ms=float(silence_ms),
            boundary_strength=boundary_strength,
        )
