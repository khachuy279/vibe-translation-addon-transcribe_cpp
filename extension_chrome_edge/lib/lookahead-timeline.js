/**
 * lookahead-timeline.js — Subtitle Timeline Queue & Zero-Latency Synchronization (v2)
 *
 * Kiến trúc:
 * 1. Lưu trữ và sắp xếp các câu phụ đề đã dịch trước theo mốc [start_pts, end_pts].
 * 2. Vòng lặp render 60fps qua requestAnimationFrame đối chiếu chính xác theo video.currentTime.
 * 3. Hiển thị phụ đề TRÙNG KHỚP 100% thời điểm nhân vật phát âm (Zero Perceived Latency).
 * 4. Xử lý Tua Video (Seek Flow): Xoá subtitle cũ, sinh seek_id, yêu cầu backend nạp lại.
 * 5. Tự động giải phóng bộ nhớ (Sliding Window Cache Eviction ±30s).
 *
 * v2: giữ câu cuối thêm một khoảng ngắn (`holdAfterEndSec`) để phụ đề KHÔNG nhấp nháy
 * giữa hai câu liền nhau, và sửa việc gỡ listener (bind tạo hàm mới nên không gỡ được).
 */

class SubtitleTimelineQueue {
  constructor(options = {}) {
    this.items = []; // [{ id, start_pts, end_pts, original_text, translated_text, seek_id }]
    this.activeSeekId = "init_0";
    this.activeSubtitle = null;
    this.videoElement = null;
    this.rafId = null;
    this.onSubtitleChange = options.onSubtitleChange || null;
    this.onSeekTriggered = options.onSeekTriggered || null;
    this.onBufferingStateChange = options.onBufferingStateChange || null;

    // Cài đặt dải cache (Sliding Window)
    this.keepBehindSec = options.keepBehindSec || 30.0;
    this.keepAheadSec = options.keepAheadSec || 120.0;
    //: Giữ câu vừa hết thêm bao lâu (giây) để tránh nhấp nháy giữa 2 câu.
    this.holdAfterEndSec = options.holdAfterEndSec !== undefined ? options.holdAfterEndSec : 0.6;
    //: SÀN cứng cho thời lượng hiển thị (giây) — chỉ để chống nhấp nháy khi aligner trả câu dài
    //: ~0 ms. KHÔNG dùng để "giãn" cửa sổ theo ý client: mốc là do backend quyết định.
    this.minDurationSec = options.minDurationSec !== undefined ? options.minDurationSec : 0.6;

    this._isPrebuffering = false;
    this._prebufferTimer = null;
    this._boundSeeking = null;
    this._boundSeeked = null;
    this._lastEvictAt = 0;
    //: Lần cuối cùng phát sự kiện phụ đề — dùng để "giữ sống" câu đang hiển thị.
    this._lastEmitAt = 0;
    //: Renderer tự hết hạn câu theo ĐỒNG HỒ THỰC. Ở Pipeline B video có thể bị TẠM DỪNG lâu
    //: (nạp đệm) ⇒ nếu không phát lại định kỳ thì câu đang đúng sẽ biến mất và không bao giờ
    //: được vẽ lại (vì `activeSubtitle` không đổi). Đây là nguyên nhân "câu đầu không hiện".
    this.keepAliveMs = options.keepAliveMs !== undefined ? options.keepAliveMs : 1200;
    //: MỐC BACKEND ĐÃ XỬ LÝ XONG (ASR + dịch) cho thế hệ seek hiện tại — RÀNG BUỘC CỨNG: playhead
    //: không được vượt qua mốc này, nếu không video sẽ phát qua vùng chưa hề được xử lý (không có
    //: phụ đề, không có gì hoạt động). `0` = chưa biết (không ràng buộc).
    this.readyUntilPts = 0;
    //: `seek_id` mà `readyUntilPts` thuộc về. Marker của thế hệ cũ không được áp cho vị trí mới.
    this.readyUntilSeekId = null;
    //: Sai số cho phép trước khi coi là "đã chạm mốc đã xử lý" (giây).
    this.underrunMarginSec = options.underrunMarginSec !== undefined ? options.underrunMarginSec : 0.35;
    //: Thời gian chờ tối đa khi tạm dừng vì hết vùng đã xử lý (ms) — dài hơn lúc khởi động vì
    //: backend cần ASR xong một khối (12-30 s audio) mới đẩy được mốc lên.
    this.underrunPrebufferMs = options.underrunPrebufferMs !== undefined ? options.underrunPrebufferMs : 8000;
    //: Callback báo "playhead đã chạm mốc đã xử lý" (chẩn đoán + cho phía content script biết lý do).
    this.onUnderrun = options.onUnderrun || null;
  }

  /**
   * Cập nhật MỐC ĐÃ XỬ LÝ từ `lookahead_status`.
   *
   * Chỉ nhận marker thuộc ĐÚNG thế hệ seek: sau khi tua, marker của vị trí cũ (có thể rất lớn) sẽ
   * khiến client tưởng đã sẵn sàng và phát video qua vùng chưa xử lý.
   */
  setReadyHorizon(readyUntilPts, seekId) {
    const pts = Number(readyUntilPts);
    if (!Number.isFinite(pts) || pts <= 0) return;
    if (seekId && this.activeSeekId && seekId !== this.activeSeekId) return;
    // Mốc không được LÙI trong cùng một thế hệ (coverage chỉ tăng); nhưng nếu là thế hệ mới thì nhận.
    if (this.readyUntilSeekId === seekId && pts < this.readyUntilPts) return;
    this.readyUntilPts = pts;
    this.readyUntilSeekId = seekId || this.activeSeekId || null;
  }

  /** Playhead đã chạm/vượt mốc đã xử lý chưa (⇒ cần tạm dừng chờ backend). */
  isBehindHorizon(curTime) {
    const cur = Number(curTime);
    if (!Number.isFinite(cur)) return false;
    if (!(this.readyUntilPts > 0)) return false;
    return cur > this.readyUntilPts - this.underrunMarginSec;
  }

  /**
   * Nạp danh sách các câu phụ đề đã dịch sẵn từ Backend vào Queue.
   */
  addSubtitles(newItems, seekId) {
    if (!newItems || !newItems.length) return;

    // Bỏ qua các item mang seekId cũ nếu đã tua sang mốc mới
    if (seekId && seekId !== this.activeSeekId) return;

    for (const item of newItems) {
      const startPts = Number(item.start_pts);
      const endPts = Number(item.end_pts);
      if (!Number.isFinite(startPts) || !Number.isFinite(endPts) || endPts <= startPts) continue;

      // Tránh trùng lặp: chỉ bỏ qua khi CÙNG nội dung và mốc thời gian sát nhau
      const exists = this.items.some(
        (it) => it.original_text === item.original_text && Math.abs(it.start_pts - startPts) < 1.0
      );
      if (exists) continue;

      this.items.push({
        id: `sub_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
        start_pts: startPts,
        end_pts: endPts,
        original_text: item.original_text || "",
        translated_text: item.translated_text || "",
        seek_id: seekId || this.activeSeekId,
      });
    }

    // Sắp xếp lại timeline theo start_pts tăng dần
    this.items.sort((a, b) => a.start_pts - b.start_pts);

    // ── ĐƠN GIẢN HOÁ 2026-10-04: HIỂN THỊ ĐÚNG MỐC BACKEND GỬI ──────────────────────────────
    // Trước đây client tự "giãn" cửa sổ hiển thị (`minDurationSec`), tự kẹp theo câu kế tiếp, và
    // tự ưu tiên câu này hay câu kia khi hai cửa sổ chồng nhau. Chính các phép chỉnh đó sinh ra lỗi
    // "backend gửi mà không hiện":
    //     maxAllowed = max(start + 0.5, next.start - 0.05)
    //     end = max(end, min(start + minDuration, maxAllowed))
    // Khi `next.start < start + 0.55` (hai khối chồng mốc), câu trước bị co còn 0.5 s và cửa sổ hai
    // câu chồng nhau ⇒ câu bắt đầu sau có thể bị câu cũ che.
    //
    // Nay backend chịu trách nhiệm về mốc: mỗi phụ đề có `[start_pts, end_pts]` riêng và các phụ đề
    // KHÔNG chồng nhau (bộ cắt khối cắt tại khoảng lặng đã xác nhận). Client chỉ cần vẽ câu nào có
    // `start_pts <= currentTime < end_pts` — không giãn, không kẹp, không ưu tiên.
    //
    // NGOẠI LỆ DUY NHẤT: chặn SÀN cưng cứng cho câu quá ngắn (aligner có thể trả câu dài 0 ms khi
    // audio không khớp văn bản) để phụ đề không nhấp nháy trong một khung hình. Chỉ được phép KÉO
    // DÀI, không bao giờ cắt ngắn, và không bao giờ vượt mốc bắt đầu của câu kế tiếp.
    const floorSec = Number(this.minDurationSec) > 0 ? Number(this.minDurationSec) : 0.0;
    for (let i = 0; i < this.items.length && floorSec > 0; i++) {
      const it = this.items[i];
      if (it.end_pts - it.start_pts >= floorSec) continue;
      const nextIt = this.items[i + 1];
      const limit = nextIt ? nextIt.start_pts : it.start_pts + floorSec;
      it.end_pts = Math.max(it.end_pts, Math.min(it.start_pts + floorSec, limit));
    }

    // Nếu đang chờ nạp đệm sau khi Tua -> phát tiếp khi đã có câu khớp vị trí hiện tại
    if (this._isPrebuffering && this.videoElement) {
      const curTime = this.videoElement.currentTime;
      const hasMatching = this.items.some((it) => curTime >= it.start_pts - 0.5 && curTime <= it.end_pts + 0.5);
      if (hasMatching) this._endPrebuffering();
    }
  }

  /**
   * Đảm bảo phụ đề hiển thị ít nhất bằng mốc kết thúc của TTS (`minEndPts`).
   * Không bao giờ kéo dài vượt quá mốc câu tiếp theo bắt đầu.
   */
  extendSubtitleEnd(startPts, minEndPts) {
    if (!Number.isFinite(startPts) || !Number.isFinite(minEndPts)) return;
    for (let i = 0; i < this.items.length; i++) {
      const it = this.items[i];
      if (Math.abs(it.start_pts - startPts) < 0.25) {
        const nextItem = this.items[i + 1];
        const maxAllowed = nextItem ? Math.max(it.end_pts, nextItem.start_pts - 0.02) : minEndPts;
        it.end_pts = Math.max(it.end_pts, Math.min(minEndPts, maxAllowed));
        break;
      }
    }
  }

  /**
   * Gắn vào thẻ <video> của trang để bắt đầu vòng lặp đồng bộ 60fps.
   */
  attachVideo(video) {
    if (!video) return;
    this.videoElement = video;
    this._boundSeeking = this._onVideoSeeking.bind(this);
    this._boundSeeked = this._onVideoSeeked.bind(this);
    video.addEventListener("seeking", this._boundSeeking);
    video.addEventListener("seeked", this._boundSeeked);
    this._startRenderLoop();
  }

  detach() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (this.videoElement) {
      if (this._boundSeeking) this.videoElement.removeEventListener("seeking", this._boundSeeking);
      if (this._boundSeeked) this.videoElement.removeEventListener("seeked", this._boundSeeked);
      this.videoElement = null;
    }
    this._boundSeeking = null;
    this._boundSeeked = null;
    this.clear();
  }

  _onVideoSeeking() {
    const targetTime = this.videoElement ? this.videoElement.currentTime : 0;
    const newSeekId = `seek_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    this.activeSeekId = newSeekId;

    // Lập tức xoá phụ đề trên màn hình và bỏ mọi câu của đoạn cũ.
    this.activeSubtitle = null;
    this._lastEmitAt = 0;
    this.items = [];
    // Mốc đã xử lý của vị trí CŨ không còn giá trị cho vị trí mới (nếu không, video sẽ phát qua
    // vùng chưa xử lý vì tưởng đã có phụ đề).
    this.readyUntilPts = 0;
    this.readyUntilSeekId = null;
    if (this.onSubtitleChange) this.onSubtitleChange(null);

    if (this.onSeekTriggered) this.onSeekTriggered(newSeekId, targetTime);

    // Vị trí mới chưa có phụ đề ⇒ báo cho content script TẠM DỪNG video chờ nạp lại.
    // (Trước đây chỉ bật cờ nội bộ mà không ai lắng nghe ⇒ video chạy tiếp không phụ đề.)
    this._startPrebuffering();
  }

  _onVideoSeeked() {
    this._tick();
  }

  _startPrebuffering(timeoutMs) {
    if (this._isPrebuffering) return;
    this._isPrebuffering = true;
    if (this.onBufferingStateChange) this.onBufferingStateChange(true);

    // Timeout an toàn: dù chưa có bản dịch cũng phải cho video chạy tiếp (không bao giờ treo cứng).
    const waitMs = Number.isFinite(timeoutMs) ? timeoutMs : 3000;
    if (this._prebufferTimer) clearTimeout(this._prebufferTimer);
    this._prebufferTimer = setTimeout(() => this._endPrebuffering(), waitMs);
  }

  _endPrebuffering() {
    if (!this._isPrebuffering) return;
    this._isPrebuffering = false;
    if (this._prebufferTimer) {
      clearTimeout(this._prebufferTimer);
      this._prebufferTimer = null;
    }
    if (this.onBufferingStateChange) this.onBufferingStateChange(false);
  }

  _startRenderLoop() {
    const loop = () => {
      this._tick();
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  /**
   * Câu phụ đề đang hiển thị tại `curTime`: câu có `start_pts <= curTime < end_pts`.
   *
   * Giữ nguyên quy tắc "câu bắt đầu MUỘN NHẤT thắng" khi có chồng lấn — chỉ là lưới an toàn, vì
   * backend đã bảo đảm các cửa sổ không chồng nhau.
   */
  _findSubtitleAt(curTime) {
    let best = null;
    for (const item of this.items) {
      if (curTime >= item.start_pts && curTime < item.end_pts) {
        if (!best || item.start_pts > best.start_pts) best = item;
      }
    }
    return best;
  }

  _tick() {
    if (!this.videoElement) return;
    const curTime = this.videoElement.currentTime;

    // ── RÀNG BUỘC CỨNG: KHÔNG phát video qua vùng CHƯA được xử lý ────────────────────────────
    // `readyUntilPts` là mốc cuối cùng backend đã ASR + dịch xong cho thế hệ seek hiện tại. Khi
    // playhead chạm mốc đó, tạm dừng và chờ backend đẩy mốc lên. Nếu không có ràng buộc này, video
    // vẫn chạy qua vùng backend chưa xử lý ⇒ "phát video mà không có gì hoạt động" (sự cố
    // 2026-10-02: tua tới vùng chưa tải thì phụ đề lúc hiện lúc không, tua ngược về đầu thì không
    // hiện gì).
    if (!this._isPrebuffering && this.isBehindHorizon(curTime)) {
      if (this.onUnderrun) {
        try { this.onUnderrun(curTime, this.readyUntilPts); } catch (e) {}
      }
      this._startPrebuffering(this.underrunPrebufferMs);
      return;
    }

    let matchedSub = this._findSubtitleAt(curTime);

    // Giữ câu vừa hết thêm `holdAfterEndSec` để không nhấp nháy giữa hai câu liền nhau.
    if (!matchedSub && this.activeSubtitle
        && curTime < this.activeSubtitle.end_pts + this.holdAfterEndSec
        && curTime >= this.activeSubtitle.start_pts) {
      matchedSub = this.activeSubtitle;
    }

    // Phát lại định kỳ cho CÙNG một câu: vừa giữ câu khỏi bị renderer cho hết hạn, vừa bảo
    // đảm câu được vẽ lại sau khi overlay bị `clear()` (lúc resume sau khi nạp đệm / tua).
    const now = Date.now();
    const sameSub = matchedSub && matchedSub === this.activeSubtitle;
    const stale = sameSub && (now - this._lastEmitAt) >= this.keepAliveMs;

    if (matchedSub !== this.activeSubtitle || stale) {
      this.activeSubtitle = matchedSub;
      this._lastEmitAt = now;
      if (this.onSubtitleChange) this.onSubtitleChange(matchedSub);
    }

    // Dọn dẹp định kỳ (không dùng xác suất ngẫu nhiên để hành vi tất định).
    if (now - this._lastEvictAt > 5000) {
      this._lastEvictAt = now;
      this._evict(curTime);
    }
  }

  /**
   * Buộc vẽ lại câu đang khớp NGAY BÂY GIỜ.
   *
   * Dùng sau khi overlay bị xoá (resume sau nạp đệm, hoặc vừa tua xong): nếu không, vì
   * `activeSubtitle` vẫn trỏ vào câu cũ nên `_tick` thấy "không đổi" và KHÔNG vẽ lại — phụ đề
   * trống cho tới khi sang câu kế tiếp.
   */
  refreshNow() {
    if (!this.videoElement) return;
    this.activeSubtitle = undefined;
    this._lastEmitAt = 0;
    const curTime = this.videoElement.currentTime;
    let matchedSub = this._findSubtitleAt(curTime);
    if (!matchedSub) {
      // Khi vừa resume sau khi nạp đệm / tua: cho phép dung sai sớm 0.25s để câu nói sát mép
      // playhead hiển thị NGAY LẬP TỨC trên màn hình mà không bị trễ khung hình đầu. Vẫn ưu tiên
      // câu có mốc bắt đầu MUỘN NHẤT khi nhiều câu nằm trong dung sai.
      for (const item of this.items) {
        if (curTime >= item.start_pts - 0.25 && curTime < item.end_pts) {
          if (!matchedSub || item.start_pts > matchedSub.start_pts) matchedSub = item;
        }
      }
    }
    if (matchedSub) {
      this.activeSubtitle = matchedSub;
      this._lastEmitAt = Date.now();
      if (this.onSubtitleChange) this.onSubtitleChange(matchedSub);
      return;
    }
    this._tick();
  }

  _evict(curTime) {
    const cutoff = Math.max(0, curTime - this.keepBehindSec);
    const before = this.items.length;
    this.items = this.items.filter((it) => it.end_pts >= cutoff);
    if (this.items.length !== before) {
      this.items.sort((a, b) => a.start_pts - b.start_pts);
    }
    return before - this.items.length;
  }

  /**
   * Chẩn đoán: số câu đang giữ và khoảng phủ phía trước.
   */
  stats() {
    if (!this.videoElement || !this.items.length) {
      return { count: this.items.length, ahead: 0 };
    }
    const cur = this.videoElement.currentTime;
    const ahead = this.items.filter((it) => it.end_pts > cur).reduce((m, it) => Math.max(m, it.end_pts - cur), 0);
    return { count: this.items.length, ahead };
  }

  clear() {
    this.items = [];
    this.activeSubtitle = null;
    this.readyUntilPts = 0;
    this.readyUntilSeekId = null;
    this._endPrebuffering();
    if (this.onSubtitleChange) this.onSubtitleChange(null);
  }
}

// Xuất module cho Extension
if (typeof module !== "undefined" && module.exports) {
  module.exports = { SubtitleTimelineQueue };
}
