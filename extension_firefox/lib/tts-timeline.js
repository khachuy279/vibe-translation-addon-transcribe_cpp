/**
 * tts-timeline.js — Bộ hẹn giờ phát LỒNG TIẾNG theo timeline video (Pipeline B / Lookahead).
 *
 * VẤN ĐỀ CỐT LÕI
 * --------------
 * Ở Pipeline B, phụ đề đã có sẵn từ trước `lead_time` giây, nên audio TTS phải được phát
 * ĐÚNG lúc `video.currentTime` đi qua `start_pts` — chứ không phải "phát ngay khi nhận được".
 *
 * Nếu chỉ phát nối đuôi nhau (kiểu hàng đợi của Pipeline A), một câu đọc DÀI HƠN cửa sổ phụ
 * đề sẽ **đẩy lùi toàn bộ các câu sau** ⇒ lồng tiếng trễ dần và không bao giờ bắt kịp.
 *
 * BA TẦNG CHỐNG TRỄ (theo thứ tự can thiệp)
 * -----------------------------------------
 *  1. Backend đã NÉN thời gian (giữ nguyên cao độ) để câu đọc vừa ngân sách cửa sổ.
 *  2. Ở đây tính `playbackRate` còn thiếu cho ĐÚNG khoảng trống tới câu kế tiếp
 *     (`nextStart - startPts`), trần `maxRate`.
 *  3. Mỗi câu có một mốc `stopAtPts` TUYỆT ĐỐI: quá mốc là CẮT, nhường chỗ cho câu sau.
 *     Vì mọi câu đều được kích hoạt bằng mốc thời gian tuyệt đối của video, sai số KHÔNG
 *     bao giờ tích luỹ — xấu nhất chỉ là một câu bị cụt đuôi.
 *
 * Câu đến muộn quá `lateToleranceSec` (video đã chạy qua đầu câu) sẽ bị BỎ thay vì phát
 * lệch — giữ đúng nguyên tắc "thà mất một câu còn hơn lệch tiếng".
 */

(function (global) {
  class TtsTimelineScheduler {
    constructor(options = {}) {
      this.video = null;
      //: Hàm trả về AudioContext dùng chung (tái sử dụng của TTSAudioPlayer để không mở
      //: nhiều AudioContext và để chung trạng thái "đã được người dùng mở khoá").
      this.getAudioContext = options.getAudioContext || (() => null);

      this.enabled = options.enabled !== undefined ? !!options.enabled : true;
      //: Trần nén phía client (dự phòng cho phần backend chưa nén hết).
      this.maxRate = options.maxRate || 1.35;
      //: Cho phép client ép tăng tốc AudioBufferSourceNode (chú ý: Web Audio API sẽ làm tăng cao độ pitch-shift)
      this.allowPitchShift = options.allowPitchShift !== undefined ? !!options.allowPitchShift : true;
      //: Bắt đầu sớm hơn mốc phụ đề một chút để bù độ trễ lập lịch.
      this.startLeadSec = options.startLeadSec !== undefined ? options.startLeadSec : 0.06;
      //: Trễ tối đa còn chấp nhận phát (giây); muộn hơn thì BỎ câu (mặc định nới lên 2.5s để không bỏ oan câu).
      this.lateToleranceSec = options.lateToleranceSec !== undefined ? options.lateToleranceSec : 2.50;
      //: Trần "được phép đọc thêm" khi không biết câu kế tiếp (giây).
      this.tailAllowanceSec = options.tailAllowanceSec !== undefined ? options.tailAllowanceSec : 1.0;
      //: Khoảng đệm ân hạn (giây) cho phép câu trước đọc nốt 1-2 từ cuối thay vì bị câu sau cắt ngang.
      this.preemptGraceSec = options.preemptGraceSec !== undefined ? options.preemptGraceSec : 0.50;
      this.tickMs = options.tickMs || 50;

      this.items = [];          // [{ id, startPts, endPts, audioBuffer, blobUrl }] — sắp theo startPts
      this._current = null;     // { src, item, stopAtPts, naturalSec }
      this._timer = null;
      this._destroyed = false;
      this._blobUrls = new Set();
      this._unlockHandler = null;
      this._diagnosed = { received: false, played: false, blocked: false };

      this.stats = {
        received: 0,
        played: 0,
        skippedLate: 0,
        cutShort: 0,
        droppedStale: 0,
        decodeErrors: 0,
        blockedByAudioContext: 0,
        maxLateMs: 0,
        maxRate: 1,
      };
    }

    attach(video) {
      this.video = video || null;
      if (this.video) this._startTimer();
      this._installUnlockListeners();
      console.log(
        `[BS TTS-LA] Scheduler ${this.enabled ? "BẬT" : "TẮT (ttsEnabled=false)"} — ` +
        `AudioContext=${(() => { const c = this.getAudioContext(); return c ? c.state : "none"; })()}`
      );
    }

    /**
     * "Mở khoá" AudioContext theo cử chỉ người dùng.
     *
     * VÌ SAO CẦN: AudioContext tạo trong callback mạng (không phải cử chỉ người dùng) có thể
     * khởi động ở trạng thái `suspended`, và `resume()` ngoài cử chỉ sẽ bị từ chối ⇒ audio
     * TTS tổng hợp xong nhưng KHÔNG kêu. Ta bám vào `click`/`play` của trang để resume.
     */
    _installUnlockListeners() {
      if (this._unlockHandler || typeof document === "undefined") return;
      this._unlockHandler = () => this.unlockAudio();
      try {
        document.addEventListener("click", this._unlockHandler, true);
        document.addEventListener("play", this._unlockHandler, true);
        document.addEventListener("keydown", this._unlockHandler, true);
      } catch (e) {}
      this.unlockAudio();
    }

    unlockAudio() {
      const ctx = this.getAudioContext();
      if (!ctx) return false;
      if (ctx.state === "suspended") {
        try {
          const p = ctx.resume();
          if (p && p.catch) p.catch(() => {});
        } catch (e) {}
      }
      return ctx.state === "running";
    }

    setEnabled(on) {
      this.enabled = !!on;
      if (!this.enabled) this.clear();
    }

    /**
     * Nạp một câu lồng tiếng đã tổng hợp: `header` có `start_pts`/`end_pts`, `arrayBuffer`
     * là WAV thô do backend gửi (khung nhị phân `[4B hdrlen][JSON][WAV]`).
     */
    enqueue(header, arrayBuffer) {
      if (this._destroyed || !this.enabled || !header || !arrayBuffer || !arrayBuffer.byteLength) return false;
      const startPts = Number(header.start_pts);
      const endPts = Number(header.end_pts);
      if (!isFinite(startPts) || !isFinite(endPts) || endPts <= startPts) return false;

      const ctx = this.getAudioContext();
      const id = `${header.seek_id || "s"}_${startPts.toFixed(2)}`;

      // Giữ thêm một Blob URL của WAV GỐC làm đường dự phòng khi AudioContext không chạy
      // được (xem `_playViaElement`). Blob copy dữ liệu nên `decodeAudioData` detach
      // ArrayBuffer gốc cũng không ảnh hưởng.
      let blobUrl = null;
      try {
        if (typeof Blob !== "undefined" && typeof URL !== "undefined" && URL.createObjectURL) {
          blobUrl = URL.createObjectURL(new Blob([arrayBuffer], { type: "audio/wav" }));
          this._blobUrls.add(blobUrl);
        }
      } catch (e) {}

      const push = (audioBuffer) => {
        if (this._destroyed) return;
        if (this.items.some((it) => Math.abs(it.startPts - startPts) < 0.05)) return;
        this.items.push({ id, startPts, endPts, audioBuffer, blobUrl });
        this.items.sort((a, b) => a.startPts - b.startPts);
        this.stats.received++;
        this._trim();
        if (!this._diagnosed.received) {
          this._diagnosed.received = true;
          console.log(
            `[BS TTS-LA] Nhận lồng tiếng đầu tiên: @${startPts.toFixed(2)}s ` +
            `(WAV ${(arrayBuffer.byteLength / 1024).toFixed(0)} KB, decode ${
              audioBuffer ? audioBuffer.duration.toFixed(2) + "s" : "LỖI"})`
          );
        }
      };

      if (!ctx) {
        // Không có Web Audio ⇒ dùng thẳng thẻ <audio>.
        push(null);
        return true;
      }
      try {
        // `decodeAudioData` chạy off-thread và DETACH buffer truyền vào.
        ctx.decodeAudioData(arrayBuffer, push, (err) => {
          this.stats.decodeErrors++;
          console.warn("[BS TTS-LA] decodeAudioData lỗi (sẽ phát bằng thẻ audio):", err);
          push(null);
        });
      } catch (e) {
        this.stats.decodeErrors++;
        push(null);
      }
      return true;
    }

    _trim() {
      const now = this.video ? this.video.currentTime : 0;
      const before = this.items.length;
      this.items = this.items.filter((it) => it.endPts > now - 0.5 && it.startPts < now + 90);
      this.stats.droppedStale += before - this.items.length;
    }

    // ── Vòng lặp hẹn giờ ────────────────────────────────────────────────────
    _startTimer() {
      if (this._timer !== null) return;
      this._timer = setInterval(() => this._tick(), this.tickMs);
      // Trong môi trường test (Node) timer không được giữ tiến trình sống.
      if (this._timer && typeof this._timer.unref === "function") this._timer.unref();
    }

    _stopTimer() {
      if (this._timer !== null) {
        clearInterval(this._timer);
        this._timer = null;
      }
    }

    _tick() {
      if (this._destroyed || !this.enabled) return;
      const video = this.video;
      if (!video) return;

      // Video đang tạm dừng ⇒ lồng tiếng cũng phải dừng (không đọc trước video).
      if (video.paused) {
        if (this._current) this._stopCurrent("video_paused");
        return;
      }

      const now = video.currentTime;

      // 1) Mốc cắt TUYỆT ĐỐI: quá hạn thì cắt, KHÔNG để nó đẩy lùi câu sau.
      if (this._current) {
        const cur = this._current;
        const playedSec = Math.max(0, ((now - cur.startedAtPts) || 0) * cur.rate);
        const remainingSec = cur.naturalSec - playedSec;
        // Nếu câu hiện tại đang đọc gần xong (chỉ còn <= preemptGraceSec), cho phép đọc thêm
        // tối đa preemptGraceSec trước khi cắt cứng bằng window_end để không cụt từ cuối.
        const effectiveStopPts = (remainingSec > 0.02 && remainingSec <= this.preemptGraceSec)
          ? cur.stopAtPts + this.preemptGraceSec
          : cur.stopAtPts;
        if (now >= effectiveStopPts) {
          this._stopCurrent("window_end");
        }
      }

      // 2) Bỏ các câu đã trôi qua.
      if (this.items.length) {
        const before = this.items.length;
        this.items = this.items.filter((it) => it.endPts > now - 0.3);
        this.stats.droppedStale += before - this.items.length;
      }

      // 3) Phát các câu đã tới hạn.
      while (this.items.length && this.items[0].startPts <= now + this.startLeadSec) {
        // Nếu câu trước đang đọc dở và CHỈ CÒN ĐUÔI NGẮN (1-2 từ cuối, <= preemptGraceSec):
        // Nhường cho câu trước đọc trọn vẹn, không cắt đứt câu!
        if (this._current) {
          const cur = this._current;
          const playedSec = Math.max(0, ((now - cur.startedAtPts) || 0) * cur.rate);
          const remainingSec = cur.naturalSec - playedSec;
          const overdueSec = Math.max(0, now - this.items[0].startPts);
          if (remainingSec > 0.02 && remainingSec <= this.preemptGraceSec && overdueSec < this.preemptGraceSec) {
            break; // Hoãn câu sau một vài tick để câu trước nói nốt trọn vẹn
          }
        }

        const item = this.items.shift();
        const lateMs = Math.max(0, (now - item.startPts) * 1000);
        // Chỉ bỏ câu khi video đã chạy qua hẳn phần kết thúc của câu (now >= item.endPts)
        // HOẶC độ trễ vượt quá trần cho phép (lateToleranceSec).
        if (now >= item.endPts || lateMs > this.lateToleranceSec * 1000) {
          this.stats.skippedLate++;
          console.warn(
            `[BS TTS-LA] Bỏ câu lồng tiếng @${item.startPts.toFixed(2)}s (trễ ${lateMs.toFixed(0)}ms, hết hạn @${item.endPts.toFixed(2)}s)`
          );
          continue;
        }
        if (this._current) this._stopCurrent("preempted");
        this._play(item, this.items[0] || null, now, lateMs);
      }
    }

    _play(item, nextItem, now, lateMs) {
      // Cửa sổ khả dụng = từ điểm bắt đầu hiệu dụng tới lúc câu KẾ TIẾP bắt đầu
      // (hoặc tới hết phụ đề + khoảng đuôi).
      const limit = nextItem
        ? Math.min(nextItem.startPts, item.endPts + this.tailAllowanceSec)
        : item.endPts + this.tailAllowanceSec;
      const effectiveStart = Math.max(item.startPts, now);
      const windowSec = Math.max(0.30, limit - effectiveStart);

      const naturalSec = item.audioBuffer
        ? item.audioBuffer.duration
        : Math.max(0.1, Number(item.declaredDuration) || windowSec);
      const videoRate = Math.max(0.25, Math.min(4.0, Number(this.video.playbackRate) || 1.0));
      let rate = videoRate;
      if (this.allowPitchShift && naturalSec * videoRate > windowSec) {
        // Cần nén thêm để vừa cửa sổ (chỉ kích hoạt khi allowPitchShift=true vì AudioBufferSourceNode không giữ cao độ).
        rate = Math.min(this.maxRate * videoRate, (naturalSec * videoRate) / windowSec);
      }
      const stopAtPts = effectiveStart + windowSec;

      const ctx = this.getAudioContext();
      const ctxRunning = !!ctx && ctx.state === "running";

      if (!ctxRunning) {
        // AudioContext bị treo (autoplay policy) ⇒ thử resume rồi phát bằng thẻ <audio>.
        // Nếu không có đường dự phòng này, TTS tổng hợp xong mà KHÔNG kêu.
        this.stats.blockedByAudioContext++;
        if (!this._diagnosed.blocked) {
          this._diagnosed.blocked = true;
          console.warn(
            `[BS TTS-LA] AudioContext ở trạng thái "${ctx ? ctx.state : "không có"}" — ` +
            `chuyển sang phát bằng thẻ <audio> (bấm vào trang video một lần để mở khoá Web Audio).`
          );
        }
        this.unlockAudio();
        this._playViaElement(item, rate, windowSec, stopAtPts, lateMs, naturalSec);
        return;
      }

      try {
        const src = ctx.createBufferSource();
        src.buffer = item.audioBuffer;
        src.playbackRate.value = rate;

        let gainNode = null;
        if (ctx && typeof ctx.createGain === "function") {
          gainNode = ctx.createGain();
          gainNode.gain.setValueAtTime(1.0, ctx.currentTime);
          src.connect(gainNode);
          gainNode.connect(ctx.destination);
        } else {
          src.connect(ctx.destination);
        }

        const entry = {
          src,
          gainNode,
          item,
          naturalSec,
          rate,
          stopAtPts,
          startedAtPts: now,
        };
        this._current = entry;

        src.onended = () => {
          if (this._current && this._current.src === src) {
            this._current = null;
            if (this.items.length && !this._destroyed && this.video && !this.video.paused) {
              this._tick();
            }
          }
          try {
            src.disconnect();
            if (gainNode) gainNode.disconnect();
          } catch (e) {}
        };

        // Bù độ trễ lập lịch: bắt đầu ngay (ctx.currentTime) — sai số còn lại chỉ là vài ms.
        src.start(ctx.currentTime + 0.005);

        this._afterPlay(item, rate, windowSec, lateMs, naturalSec);
      } catch (e) {
        console.warn("[BS TTS-LA] Web Audio lỗi, thử thẻ <audio>:", e);
        this._current = null;
        this._playViaElement(item, rate, windowSec, stopAtPts, lateMs, naturalSec);
      }
    }

    /**
     * Đường DỰ PHÒNG: phát bằng HTMLAudioElement (Blob URL).
     * Dùng khi AudioContext không chạy được (autoplay policy) hoặc decode lỗi.
     */
    _playViaElement(item, rate, windowSec, stopAtPts, lateMs, naturalSec) {
      if (!item.blobUrl || typeof Audio === "undefined") {
        this.stats.droppedStale++;
        return;
      }
      try {
        const audio = new Audio(item.blobUrl);
        audio.preservesPitch = true;
        audio.playbackRate = Math.max(0.25, Math.min(4.0, rate));
        const entry = { src: audio, item, naturalSec, rate, stopAtPts, startedAtPts: this.video.currentTime, isElement: true };
        this._current = entry;
        audio.onended = () => {
          if (this._current && this._current.src === audio) {
            this._current = null;
            if (this.items.length && !this._destroyed && this.video && !this.video.paused) {
              this._tick();
            }
          }
        };
        const p = audio.play();
        if (p && p.catch) {
          p.catch((err) => {
            console.warn("[BS TTS-LA] Thẻ <audio> cũng bị chặn:", err && err.name);
            if (this._current && this._current.src === audio) this._current = null;
          });
        }
        this._afterPlay(item, rate, windowSec, lateMs, naturalSec);
      } catch (e) {
        console.warn("[BS TTS-LA] Không phát được lồng tiếng:", e);
        this._current = null;
      }
    }

    _afterPlay(item, rate, windowSec, lateMs, naturalSec) {
      this.stats.played++;
      this.stats.maxLateMs = Math.max(this.stats.maxLateMs, lateMs);
      this.stats.maxRate = Math.max(this.stats.maxRate, rate);
      if (!this._diagnosed.played || rate > 1.02 || naturalSec > windowSec) {
        this._diagnosed.played = true;
        console.log(
          `[BS TTS-LA] ▶ phát @${item.startPts.toFixed(2)}s — đọc ${naturalSec.toFixed(2)}s / ` +
          `cửa sổ ${windowSec.toFixed(2)}s -> rate ${rate.toFixed(2)}x (trễ ${lateMs.toFixed(0)}ms)`
        );
      }
    }

    _stopCurrent(reason) {
      const cur = this._current;
      this._current = null;
      if (!cur) return;
      // Cắt khi thời lượng thật còn dài hơn thời gian đã phát ⇒ đánh dấu "cụt đuôi".
      const playedSec = Math.max(0, ((this.video ? this.video.currentTime : 0) - cur.startedAtPts) * cur.rate);
      if (playedSec + 0.05 < cur.naturalSec) {
        this.stats.cutShort++;
      }
      try {
        cur.src.onended = null;
        if (cur.isElement) {
          try { cur.src.volume = 0; } catch (e) {}
          cur.src.pause();
          cur.src.src = "";
        } else if (cur.gainNode) {
          // Micro fade-out 20ms: dập tắt hoàn toàn tiếng "bụp" (DC transient click/pop) khi ngắt
          const ctx = this.getAudioContext();
          const fadeSec = 0.020;
          if (ctx && ctx.state === "running") {
            const now = ctx.currentTime;
            cur.gainNode.gain.setValueAtTime(cur.gainNode.gain.value, now);
            cur.gainNode.gain.linearRampToValueAtTime(0.0001, now + fadeSec);
            try { cur.src.stop(now + fadeSec); } catch (e) {}
            setTimeout(() => {
              try {
                cur.src.disconnect();
                cur.gainNode.disconnect();
              } catch (e) {}
            }, (fadeSec + 0.04) * 1000);
          } else {
            try { cur.src.stop(); cur.src.disconnect(); cur.gainNode.disconnect(); } catch (e) {}
          }
        } else {
          try {
            cur.src.stop();
            cur.src.disconnect();
          } catch (e) {}
        }
      } catch (e) {}
      if (reason && reason !== "window_end") {
        console.log(`[BS TTS-LA] Dừng lồng tiếng đang phát (${reason}).`);
      }
    }

    /** Dọn sạch (khi tua video / Stop). */
    clear() {
      this.items = [];
      this._stopCurrent("clear");
    }

    snapshot() {
      return {
        queued: this.items.length,
        playing: !!this._current,
        audioContextState: (() => {
          const c = this.getAudioContext();
          return c ? c.state : "none";
        })(),
        ...this.stats,
      };
    }

    destroy() {
      this._destroyed = true;
      this.clear();
      this._stopTimer();
      if (this._unlockHandler && typeof document !== "undefined") {
        try {
          document.removeEventListener("click", this._unlockHandler, true);
          document.removeEventListener("play", this._unlockHandler, true);
          document.removeEventListener("keydown", this._unlockHandler, true);
        } catch (e) {}
      }
      this._unlockHandler = null;
      for (const url of this._blobUrls) {
        try { URL.revokeObjectURL(url); } catch (e) {}
      }
      this._blobUrls.clear();
      this.video = null;
    }
  }

  global.TtsTimelineScheduler = TtsTimelineScheduler;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { TtsTimelineScheduler };
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
