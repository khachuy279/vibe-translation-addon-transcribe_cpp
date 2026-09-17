// TTS Audio Player (Queue + Continuous Auto-Ducking + Blob URL lifecycle)
(function (global) {
  class TTSAudioPlayer {
    constructor() {
      this.queue = [];
      this.isPlaying = false;
      this.currentAudio = null;
      this.currentBlobUrl = null;
      this.ttsEnabled = false;
      this.autoDucking = true;
      this.duckingLevel = 0.25;
      this.targetVideo = null;
      this.originalVideoVolume = 1.0;
      // Ghi nhớ cả trạng thái MUTED. Bản cũ quy `volume === 0` thành 1.0, nên với video
      // đang tắt tiếng thì việc bật/tắt ducking sẽ làm tiếng gốc BẬT LẠI ở 100%.
      this.originalVideoMuted = false;
      this.isDucked = false;
      // Guard chống site ghi đè `video.volume` (xem `_attachVolumeGuard`).
      this._volumeGuard = null;
      this._writingVolume = false;
      // Nguồn ducking ưu tiên: AudioCapture (duck bằng GainNode trên nhánh NGHE, không đụng
      // `video.volume` nên ASR không bị ảnh hưởng). Xem `setDuckSink`.
      this.duckSink = null;
      this._duckedViaSink = false;
      this.playedIds = new Set();
      // P3.5a: TRẦN HÀNG ĐỢI. Trước đây `this.queue` không có giới hạn: backend sinh
      // TTS nhanh hơn realtime (~13x) nên khi consumer tụt lại thì queue phình vô hạn
      // => RAM tăng VÀ tiếng lồng trôi xa dần khỏi video mà không có cơ chế bắt kịp.
      this.maxQueueLength = 3;
      // Bỏ câu cũ hơn ngần này (ms) — thà mất lồng tiếng còn hơn lệch tiếng vĩnh viễn.
      this.maxLagMs = 12000;
      this.droppedCount = 0;
      // Dùng CHUNG một AudioContext cho mọi câu để decodeAudioData chạy off-thread và
      // audio lên lịch được (thay vì tạo HTMLAudioElement mới mỗi câu).
      this._audioCtx = null;
      this._activeSources = new Set();
    }

    _getAudioContext() {
      if (this._audioCtx && this._audioCtx.state !== "closed") return this._audioCtx;
      const Ctx = (typeof window !== "undefined") && (window.AudioContext || window.webkitAudioContext);
      if (!Ctx) return null;
      try {
        this._audioCtx = new Ctx();
      } catch (e) {
        this._audioCtx = null;
      }
      return this._audioCtx;
    }

    /**
     * Chọn nguồn ducking. Truyền `AudioCapture` (có `setDuckLevel`) để duck bằng GainNode.
     *
     * VÌ SAO QUAN TRỌNG: khi capture dùng `createMediaElementSource`, nhánh ASR lấy tín hiệu
     * từ CÙNG node nguồn đã bị nhân bởi `video.volume`. Duck bằng `video.volume` vì thế làm
     * ASR nhỏ đi theo — và ở 0% thì ASR nhận im lặng ⇒ **mất phụ đề, mất TTS**. Duck bằng
     * GainNode chỉ tác động nhánh nghe nên kéo slider về 0% vẫn giữ nguyên phụ đề.
     */
    setDuckSink(sink) {
      const next = sink && typeof sink.setDuckLevel === "function" ? sink : null;
      if (next === this.duckSink) {
        this._applyContinuousDucking();
        return;
      }

      // 1) Nhả ducking trên sink CŨ **trước khi** thay tham chiếu. Nếu xoá sink trước rồi mới
      //    nhả, gain của AudioCapture sẽ KẸT ở mức duck ⇒ tiếng gốc nhỏ vĩnh viễn.
      if (this._duckedViaSink && this.duckSink) {
        try { this.duckSink.setDuckLevel(1.0); } catch (e) {}
      }
      // 2) Nếu đang duck bằng `video.volume` thì trả lại volume gốc.
      if (this.isDucked && !this._duckedViaSink) {
        this._restoreVideoVolume();
      }
      this._duckedViaSink = false;
      this.isDucked = false;
      this.duckSink = next;
      // 3) Áp lại theo nguồn mới (nếu ducking vẫn đang bật).
      this._applyContinuousDucking();
    }

    /** Duck qua GainNode của AudioCapture. Trả `true` nếu sink đã xử lý. */
    _duckViaSink(level) {
      const sink = this.duckSink;
      if (!sink) return false;
      try {
        return sink.setDuckLevel(level) !== false;
      } catch (e) {
        return false;
      }
    }

    /**
     * Chuẩn hoá mức ducking về [0,1].
     *
     * Vì sao cần: `clamp(originalVolume * duckingLevel)`. Nếu ai đó truyền **5** (phần trăm)
     * thay vì **0.05**, ta được `clamp(1.0 * 5) = 1.0` — tức **âm lượng ĐẦY**, và triệu chứng
     * đúng là "kéo xuống 5% mà tiếng gốc vẫn lớn". Clamp ở đây bảo đảm một giá trị sai đơn vị
     * KHÔNG bao giờ âm thầm biến thành "mở to hết cỡ", và log ra để lỗi lộ diện.
     */
    _normalizeDuckingLevel(raw) {
      const n = typeof raw === "number" ? raw : parseFloat(raw);
      if (!isFinite(n)) return 0.25;
      if (n < 0 || n > 1) {
        console.warn(`[BS TTS] duckingLevel ngoài [0,1]: ${raw} — clamp về ${n < 0 ? 0 : 1}. ` +
          `Giao thức popup gửi 0..1 (5% = 0.05).`);
        return Math.max(0, Math.min(1, n));
      }
      return n;
    }

    /** Mức âm lượng mong muốn của tiếng gốc khi đang duck. */
    _desiredDuckedVolume() {
      const lvl = this._normalizeDuckingLevel(this.duckingLevel);
      return Math.max(0, Math.min(1, this.originalVideoVolume * lvl));
    }

    /** Ghi `video.volume` và tự bỏ qua chính sự kiện `volumechange` do mình gây ra. */
    _writeVideoVolume(value) {
      const v = Math.max(0, Math.min(1, value));
      this._writingVolume = true;
      try {
        this.targetVideo.volume = v;
      } finally {
        // `volumechange` của chính ta được xử lý ĐỒNG BỘ trong lúc gán ở hầu hết engine,
        // nhưng nhả cờ ở `finally` vẫn đúng nếu nó bắn bất đồng bộ.
        this._writingVolume = false;
      }
    }

    /**
     * Gắn guard `volumechange` để ducking "dính".
     *
     * VÌ SAO CẦN: extension chỉ ghi `video.volume` **một lần** mỗi lần đổi setting. Nếu trang
     * (hoặc người dùng kéo thanh âm lượng của chính trang đó) ghi `volume` sau đó, ducking bị
     * huỷ **âm thầm** — mà `isDucked` vẫn là `true`, nên ngay cả đường restore cũng sai. Đây
     * là nguyên nhân thực tế của "đặt 5% nhưng tiếng gốc vẫn lớn".
     *
     * Chống đệ quy: bỏ qua sự kiện do `_writeVideoVolume` của chính ta phát ra, và chỉ ghi lại
     * khi giá trị lệch khỏi mức mong muốn (epsilon 0.01).
     */
    _attachVolumeGuard() {
      const v = this.targetVideo;
      if (!v || this._volumeGuard) return;
      const handler = () => {
        if (!this.isDucked || this._writingVolume) return;
        const want = this._desiredDuckedVolume();
        if (Math.abs(v.volume - want) > 0.01) {
          this._writeVideoVolume(want);
        }
      };
      this._volumeGuard = handler;
      try {
        v.addEventListener("volumechange", handler);
      } catch (e) {
        this._volumeGuard = null;
      }
    }

    _detachVolumeGuard() {
      if (!this._volumeGuard) return;
      try {
        this.targetVideo?.removeEventListener("volumechange", this._volumeGuard);
      } catch (e) {}
      this._volumeGuard = null;
    }

    setTargetVideo(video, autoDucking = true, duckingLevel = 0.25, ttsEnabled = false) {
      if (this.targetVideo && this.targetVideo !== video) {
        this._detachVolumeGuard();
        if (this.isDucked) this._restoreVideoVolume();
      }
      this.targetVideo = video;
      this.autoDucking = autoDucking;
      this.ttsEnabled = ttsEnabled;
      if (duckingLevel !== undefined) this.duckingLevel = this._normalizeDuckingLevel(duckingLevel);
      if (video && video.volume !== undefined && !this.isDucked) {
        // Lấy ĐÚNG giá trị thật, kể cả 0 (người dùng tự tắt tiếng). Bản cũ thay 0 bằng 1.0
        // nên sau khi restore, video đang muted bị bật lên 100%.
        this.originalVideoVolume = video.volume;
        this.originalVideoMuted = !!video.muted;
      }
      this._applyContinuousDucking();
    }

    applySettings(autoDucking, duckingLevel, ttsEnabled) {
      if (autoDucking !== undefined) this.autoDucking = autoDucking;
      if (duckingLevel !== undefined) this.duckingLevel = this._normalizeDuckingLevel(duckingLevel);
      if (ttsEnabled !== undefined) this.ttsEnabled = ttsEnabled;
      this._applyContinuousDucking();
    }

    /** Áp lại mức ducking hiện tại (dùng khi trang vừa reset volume, ví dụ lúc `play`). */
    reapplyDucking() {
      if (!this.isDucked) {
        this._applyContinuousDucking();
        return;
      }
      if (this._duckedViaSink) {
        // GainNode không bị trang ghi đè, nhưng capture có thể vừa dựng lại gain ⇒ áp lại.
        this._duckViaSink(this._normalizeDuckingLevel(this.duckingLevel));
        return;
      }
      const want = this._desiredDuckedVolume();
      if (Math.abs(this.targetVideo.volume - want) > 0.01) {
        this._writeVideoVolume(want);
      }
    }

    _applyContinuousDucking() {
      if (!this.targetVideo) return;
      const shouldDuck = !!this.ttsEnabled && !!this.autoDucking;
      if (shouldDuck) {
        const level = this._normalizeDuckingLevel(this.duckingLevel);
        // ƯU TIÊN 1: duck bằng GainNode của AudioCapture — KHÔNG đụng `video.volume`, nên
        // nhánh ASR (lấy từ cùng `sourceNode`) vẫn full-scale ⇒ phụ đề/TTS không bao giờ mất.
        if (this._duckViaSink(level)) {
          this._duckedViaSink = true;
          this.isDucked = true;
          this._detachVolumeGuard(); // không tranh chấp `video.volume` ⇒ không cần guard
          return;
        }
        // DỰ PHÒNG: đường capture không có GainNode (captureStream / MediaStream) ⇒ buộc phải
        // duck bằng `video.volume`. Chấp nhận hạn chế: ASR cũng nhỏ đi theo.
        this._duckedViaSink = false;
        try {
          if (!this.isDucked) {
            this.originalVideoVolume = this.targetVideo.volume;
            this.originalVideoMuted = !!this.targetVideo.muted;
          }
          this._writeVideoVolume(Math.max(0, Math.min(1, this.originalVideoVolume * level)));
          this.isDucked = true;
          this._attachVolumeGuard();
        } catch (e) {
          console.warn("[BS TTS] Constant ducking error:", e);
        }
      } else if (this.isDucked) {
        this._restoreVideoVolume();
      }
    }

    _restoreVideoVolume() {
      this._detachVolumeGuard();
      // Nhả ducking trên GainNode (nếu phiên này duck bằng gain).
      if (this._duckedViaSink) {
        this._duckedViaSink = false;
        this._duckViaSink(1.0);
        this.isDucked = false;
        return;
      }
      if (!this.targetVideo || !this.isDucked) return;
      try {
        this._writeVideoVolume(this.originalVideoVolume);
        this.targetVideo.muted = this.originalVideoMuted;
        this.isDucked = false;
      } catch (e) {
        console.warn("[BS TTS] Restore volume error:", e);
      }
    }

    enqueue(item) {
      if (!item || !item.audioBase64) return;
      if (item.id) {
        if (this.playedIds.has(item.id)) return;
        this.playedIds.add(item.id);
        if (this.playedIds.size > 300) {
          const oldest = this.playedIds.values().next().value;
          this.playedIds.delete(oldest);
        }
      }
      item.enqueuedAt = performance.now();
      this.queue.push(item);
      this._trimQueue();
      if (!this.isPlaying) {
        this._playNext();
      }
    }

    /**
     * P3.1: nạp audio TTS dạng BINARY (ArrayBuffer) — không base64, không atob, không
     * vòng lặp per-byte trên main thread. Decode bằng decodeAudioData (chạy off-thread
     * trong browser) rồi phát qua Web Audio graph.
     */
    enqueueBinary(header, arrayBuffer) {
      if (!arrayBuffer || !arrayBuffer.byteLength) return;
      // v3 (F-30): header khung nhị phân chỉ có `utterance_id` (không còn alias).
      const id = header && header.utterance_id;
      if (id) {
        if (this.playedIds.has(id)) return;
        this.playedIds.add(id);
        if (this.playedIds.size > 300) {
          const oldest = this.playedIds.values().next().value;
          this.playedIds.delete(oldest);
        }
      }

      const ctx = this._getAudioContext();
      if (!ctx) {
        // Không có Web Audio: quay về đường Blob URL (vẫn không dùng base64).
        const url = URL.createObjectURL(new Blob([arrayBuffer], { type: "audio/wav" }));
        this.enqueue({ id: id, audioUrl: url, durationSec: (header && header.duration_sec) || 0 });
        return;
      }

      ctx.decodeAudioData(
        // FIX-11: KHÔNG `slice(0)` thêm lần nữa. Caller (`content-script.js`) đã tạo một
        // ArrayBuffer MỚI bằng `buf.slice(7 + jsonLen)` chỉ để đưa vào đây, nên bản sao
        // thứ hai là thừa (thêm một full copy WAV cho mỗi câu). `decodeAudioData` sẽ
        // DETACH buffer truyền vào — chấp nhận được vì buffer này không ai dùng lại
        // (`playedIds` chặn phát trùng theo `utterance_id`).
        arrayBuffer,
        (audioBuffer) => {
          this.queue.push({
            id: id,
            audioBuffer: audioBuffer,
            durationSec: audioBuffer.duration,
            enqueuedAt: performance.now(),
          });
          this._trimQueue();
          if (!this.isPlaying) this._playNextBuffer();
        },
        (err) => {
          console.warn("[BS TTS] decodeAudioData lỗi:", err);
        }
      );
    }

    // P3.5a: giới hạn hàng đợi + bỏ câu quá cũ.
    _trimQueue() {
      const now = performance.now();
      const before = this.queue.length;
      this.queue = this.queue.filter(
        (it) => now - (it.enqueuedAt || now) < this.maxLagMs
      );
      while (this.queue.length > this.maxQueueLength) {
        this.queue.shift();
      }
      const dropped = before - this.queue.length;
      if (dropped > 0) {
        this.droppedCount += dropped;
        console.warn(
          `[BS TTS] Bỏ ${dropped} câu lồng tiếng (tổng ${this.droppedCount}) để không lệch tiếng.`
        );
      }
    }

    // Phát item dạng AudioBuffer (đường binary).
    _playNextBuffer() {
      if (this.queue.length === 0) {
        this.isPlaying = false;
        return;
      }
      const item = this.queue.shift();
      const ctx = this._getAudioContext();
      if (!ctx || !item.audioBuffer) {
        this._playNext();
        return;
      }
      if (ctx.state === "suspended") {
        ctx.resume().catch(() => {});
      }
      try {
        const src = ctx.createBufferSource();
        src.buffer = item.audioBuffer;
        src.connect(ctx.destination);
        this.isPlaying = true;
        this._activeSources.add(src);
        src.onended = () => {
          this._activeSources.delete(src);
          try { src.disconnect(); } catch (e) {}
          this._playNextBuffer();
        };
        src.start();
      } catch (e) {
        console.warn("[BS TTS] Phát audio lỗi:", e);
        this.isPlaying = false;
        this._playNextBuffer();
      }
    }

    _base64ToBlobUrl(base64Str) {
      const binary = atob(base64Str);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "audio/wav" });
      return URL.createObjectURL(blob);
    }

    _playNext() {
      if (this.queue.length === 0) {
        this.isPlaying = false;
        this.currentAudio = null;
        this.currentBlobUrl = null;
        return;
      }

      this.isPlaying = true;
      const item = this.queue.shift();

      try {
        const audioUrl = item.audioUrl ? item.audioUrl : this._base64ToBlobUrl(item.audioBase64);
        const audio = new Audio(audioUrl);
        this.currentAudio = audio;
        this.currentBlobUrl = audioUrl;

        const onDone = () => {
          if (this.currentAudio === audio) {
            try {
              audio.onended = null;
              audio.onerror = null;
              audio.pause();
              audio.src = "";
            } catch (e) {}
            if (audioUrl) {
              try { URL.revokeObjectURL(audioUrl); } catch (e) {}
            }
            this.currentAudio = null;
            this.currentBlobUrl = null;
            this._playNext();
          }
        };

        audio.onended = onDone;
        audio.onerror = (e) => {
          console.warn("[BS TTS] Playback error on item:", item.id, e);
          onDone();
        };

        audio.play().catch((err) => {
          console.warn("[BS TTS] Play error (autoplay/seek):", err);
          onDone();
        });
      } catch (err) {
        console.error("[BS TTS] Failed to play audio chunk:", err);
        this.currentAudio = null;
        this.currentBlobUrl = null;
        this._playNext();
      }
    }

    clear() {
      this.queue = [];
      // Dừng mọi nguồn Web Audio đang phát (đường binary).
      for (const src of Array.from(this._activeSources)) {
        try {
          src.onended = null;
          src.stop();
          src.disconnect();
        } catch (e) {}
      }
      this._activeSources.clear();
      if (this.currentAudio) {
        try {
          this.currentAudio.onended = null;
          this.currentAudio.onerror = null;
          this.currentAudio.pause();
          this.currentAudio.src = "";
        } catch (e) {}
        this.currentAudio = null;
      }
      if (this.currentBlobUrl) {
        try { URL.revokeObjectURL(this.currentBlobUrl); } catch (e) {}
        this.currentBlobUrl = null;
      }
      this.isPlaying = false;
    }

    destroy() {
      this.clear();
      this.playedIds.clear();
      this._restoreVideoVolume();
      this.targetVideo = null;
    }
  }

  global.TTSAudioPlayer = TTSAudioPlayer;
})(typeof globalThis !== "undefined" ? globalThis : this);
