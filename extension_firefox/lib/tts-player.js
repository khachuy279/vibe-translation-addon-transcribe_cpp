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
      this.isDucked = false;
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

    setTargetVideo(video, autoDucking = true, duckingLevel = 0.25, ttsEnabled = false) {
      if (this.targetVideo && this.targetVideo !== video && this.isDucked) {
        this._restoreVideoVolume();
      }
      this.targetVideo = video;
      this.autoDucking = autoDucking;
      this.ttsEnabled = ttsEnabled;
      if (duckingLevel !== undefined) this.duckingLevel = duckingLevel;
      if (video && video.volume !== undefined && !this.isDucked) {
        this.originalVideoVolume = video.volume > 0 ? video.volume : 1.0;
      }
      this._applyContinuousDucking();
    }

    applySettings(autoDucking, duckingLevel, ttsEnabled) {
      if (autoDucking !== undefined) this.autoDucking = autoDucking;
      if (duckingLevel !== undefined) this.duckingLevel = duckingLevel;
      if (ttsEnabled !== undefined) this.ttsEnabled = ttsEnabled;
      this._applyContinuousDucking();
    }

    _applyContinuousDucking() {
      if (!this.targetVideo) return;
      const shouldDuck = !!this.ttsEnabled && !!this.autoDucking;
      if (shouldDuck) {
        try {
          if (!this.isDucked) {
            this.originalVideoVolume = this.targetVideo.volume > 0 ? this.targetVideo.volume : 1.0;
          }
          this.targetVideo.volume = Math.max(0, Math.min(1.0, this.originalVideoVolume * this.duckingLevel));
          this.isDucked = true;
        } catch (e) {
          console.warn("[BS TTS] Constant ducking error:", e);
        }
      } else if (this.isDucked) {
        this._restoreVideoVolume();
      }
    }

    _restoreVideoVolume() {
      if (!this.targetVideo || !this.isDucked) return;
      try {
        this.targetVideo.volume = Math.max(0, Math.min(1.0, this.originalVideoVolume));
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
