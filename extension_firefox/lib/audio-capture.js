// Audio capture module for Firefox & Chrome Content Scripts
// Processes HTMLVideoElement or MediaStream into 16kHz mono Int16 PCM chunks.

class AudioCapture {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.videoElement = null;
    this.sourceNode = null;
    this.processorNode = null;
    this.isCapturing = false;
    this.sampleRate = 16000;
    this.chunkIndex = 0;
    // 1024 samples @ 16kHz = 64ms chunk (divides 4096 buffer exactly by 4, 0 lost samples)
    this.chunkSize = 1024;
    this.onChunk = null; // callback(pcmData: ArrayBuffer, captureTimestamp: number, chunkIndex: number)
    this.onError = null;
    this.residualSamples = new Float32Array(0);
    this.resamplePhase = 0.0;
    // P3.4: đường capture bằng AudioWorklet (mặc định) và các trạng thái phụ trợ.
    this.workletNode = null;
    this._workletPathActive = false;
    this._workletWatchdog = null;
    this._workletChunksSeen = 0;
    // GainNode điều khiển ÂM LƯỢNG NGHE của tiếng gốc (auto-ducking). Xem `setDuckLevel`.
    this.duckGain = null;
    this.duckLevel = 1.0;
  }

  /**
   * Có thể duck bằng GainNode không (đường MediaElementSource).
   *
   * VÌ SAO QUAN TRỌNG: nhánh capture cho ASR lấy từ `sourceNode`, mà `sourceNode` **đã bị
   * nhân bởi `video.volume`**. Nếu duck bằng cách hạ `video.volume` thì ASR cũng nhận audio
   * nhỏ đi — và ở 0% thì nhận **im lặng kỹ thuật số** ⇒ VAD không thấy tiếng nói ⇒ **mất
   * phụ đề, mất dịch, mất TTS**. Duck bằng GainNode chỉ tác động lên nhánh NGHE, còn nhánh
   * capture luôn full-scale ⇒ kéo slider về 0% vẫn giữ nguyên phụ đề.
   */
  supportsGainDucking() {
    return !!this.duckGain;
  }

  /**
   * Đặt mức âm lượng NGHE của tiếng gốc (0..1). Không đụng `video.volume`.
   * Trả `true` nếu đã xử lý bằng gain; `false` để caller rơi về đường `video.volume`.
   */
  setDuckLevel(level) {
    if (!this.duckGain) return false;
    const v = Math.max(0, Math.min(1, Number(level)));
    if (!isFinite(v)) return false;
    this.duckLevel = v;
    try {
      const ctx = this.audioContext;
      const now = ctx ? ctx.currentTime : 0;
      // Ramp ngắn để không nghe tiếng "cụp" khi kéo slider.
      if (this.duckGain.gain.setTargetAtTime) {
        this.duckGain.gain.setTargetAtTime(v, now, 0.015);
      } else {
        this.duckGain.gain.value = v;
      }
      return true;
    } catch (e) {
      return false;
    }
  }


  /**
   * Start capturing from a MediaStream or HTMLVideoElement.
   * @param {MediaStream|HTMLMediaElement} target - Audio source
   */
  async start(target) {
    if (this.isCapturing) return;
    this.chunkIndex = 0;

    try {
      if (target instanceof HTMLMediaElement || (target && typeof target.play === "function")) {
        this.videoElement = target;

        // Try MediaElementSource first (direct HTMLMediaElement audio pipeline)
        if (!target.__bsAudioCtx || target.__bsAudioCtx.state === "closed") {
          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          // F-29: KHÔNG ép sampleRate về 16 kHz nữa. Trước đây context bị ép 16 kHz và
          // media element được route vào rồi ra chính context đó, khiến audio người dùng
          // nghe bị band-limit xuống 8 kHz. Nay dùng rate gốc của thiết bị; việc hạ về
          // 16 kHz do worklet (hoặc resampler JS ở đường dự phòng) đảm nhiệm.
          target.__bsAudioCtx = new AudioCtx({ latencyHint: "interactive" });
          try {
            target.__bsSourceNode = target.__bsAudioCtx.createMediaElementSource(target);
            // TÁCH NHÁNH NGHE KHỎI NHÁNH CAPTURE.
            //
            // `createMediaElementSource` cho ra tín hiệu ĐÃ bị nhân bởi `video.volume`.
            // Bản cũ nối thẳng `sourceNode -> destination` rồi duck bằng cách hạ
            // `video.volume`, nên ASR (lấy từ cùng `sourceNode`) cũng bị hạ theo — và ở 0%
            // thì ASR nhận im lặng ⇒ mất phụ đề/TTS.
            //
            // Nay: đưa một GainNode vào CHỈ nhánh nghe. Ducking điều khiển gain này, còn
            // `video.volume` giữ nguyên giá trị người dùng đặt trên trang.
            //
            //   sourceNode ─┬─> duckGain ──> destination      (người dùng nghe)
            //               └─> workletNode ──> ASR            (luôn full-scale)
            target.__bsDuckGain = target.__bsAudioCtx.createGain();
            target.__bsDuckGain.gain.value = 1.0;
            target.__bsSourceNode.connect(target.__bsDuckGain);
            target.__bsDuckGain.connect(target.__bsAudioCtx.destination);
          } catch (elemErr) {
            console.warn("[AudioCapture] createMediaElementSource failed, trying stream capture:", elemErr);
          }
        }

        if (target.__bsSourceNode) {
          this.audioContext = target.__bsAudioCtx;
          this.sourceNode = target.__bsSourceNode;
          this.duckGain = target.__bsDuckGain || null;
          console.log(
            "[AudioCapture] Source: MediaElementSource (direct video audio, routed to speakers)" +
            (this.duckGain ? " + duckGain (ducking không ảnh hưởng ASR)" : "")
          );
        } else {
          // Fallback: captureStream / mozCaptureStream
          let stream = null;
          try {
            if (typeof target.captureStream === "function") {
              stream = target.captureStream();
            } else if (typeof target.mozCaptureStream === "function") {
              stream = target.mozCaptureStream();
            }
          } catch (e) {}

          if (stream && stream.getAudioTracks().length > 0) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioCtx({ latencyHint: "interactive" });
            this.mediaStream = stream;
            this.sourceNode = this.audioContext.createMediaStreamSource(stream);
            this.sourceNode.connect(this.audioContext.destination);
            console.log("[AudioCapture] Source: captureStream (routed to speakers)");
          } else {
            throw new Error("Không thể trích xuất âm thanh từ thẻ video");
          }
        }
      } else if (target && typeof target.getAudioTracks === "function") {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioCtx({ latencyHint: "interactive" });
        this.mediaStream = target;
        this.sourceNode = this.audioContext.createMediaStreamSource(target);
        console.log("[AudioCapture] Source: direct MediaStream");
      } else {
        throw new Error("Invalid audio target: expected HTMLMediaElement or MediaStream");
      }

      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      // P3.4: ưu tiên AudioWorklet (chạy trên audio thread riêng, không bị jank main
      // thread). Chỉ rơi về ScriptProcessor khi môi trường không hỗ trợ.
      const workletOk = await this._setupAudioWorklet();
      if (!workletOk) {
        console.warn("[AudioCapture] AudioWorklet không khả dụng — dùng ScriptProcessor (main thread).");
        this._setupScriptProcessor();
      }

      this.isCapturing = true;
      console.log(
        `[AudioCapture] Started (Rate: ${this.audioContext.sampleRate}Hz -> 16000Hz, ` +
        `path: ${workletOk ? "AudioWorklet" : "ScriptProcessor"}, chunk: ${this.chunkSize} samples)`
      );
    } catch (e) {
      console.error("[AudioCapture] Start error:", e);
      this.stop();
      if (this.onError) this.onError(e);
      throw e;
    }
  }

  /**
   * P3.4: dựng đường capture bằng AudioWorklet.
   * Trả về true nếu thành công; false để caller rơi về ScriptProcessor.
   */
  async _setupAudioWorklet() {
    try {
      if (!this.audioContext || !this.audioContext.audioWorklet) return false;
      const runtime = (typeof browser !== "undefined" && browser.runtime)
        || (typeof chrome !== "undefined" && chrome.runtime);
      if (!runtime || typeof runtime.getURL !== "function") return false;
      const moduleUrl = runtime.getURL("lib/audio-processor.js");

      await this.audioContext.audioWorklet.addModule(moduleUrl);

      this.workletNode = new AudioWorkletNode(this.audioContext, "audio-capture-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: {
          targetSampleRate: this.sampleRate, // 16000
          chunkSize: this.chunkSize,         // 1024
        },
      });

      this.workletNode.port.onmessage = (ev) => {
        const msg = ev.data;
        if (!msg || msg.type !== "audio_chunk" || !this.isCapturing || !this.onChunk) return;
        this._workletChunksSeen++;
        const idx = this.chunkIndex++;
        const ts = typeof msg.captureTimestamp === "number" ? msg.captureTimestamp : 0;
        if (idx === 0 || idx % 200 === 0) {
          console.log(
            `[AudioCapture] Emitted audio chunk #${idx} via AudioWorklet ` +
            `(${msg.buffer.byteLength} bytes, rate: ${this.audioContext.sampleRate}Hz->16kHz)`
          );
        }
        this.onChunk(msg.buffer, ts, idx);
      };

      this.workletNode.onprocessorerror = (err) => {
        console.error("[AudioCapture] AudioWorklet processor error:", err);
        // Rơi về ScriptProcessor để không mất tiếng hoàn toàn.
        this._fallbackToScriptProcessor();
      };

      this.sourceNode.connect(this.workletNode);
      // Một số engine chỉ pump node khi nó được nối tới destination.
      // Worklet không ghi ra output nên đây là im lặng, không tạo tiếng đôi.
      this.workletNode.connect(this.audioContext.destination);

      // Watchdog: nếu sau 2.5 s chưa có chunk nào thì coi như worklet không chạy
      // (một số build Firefox cũ) và rơi về ScriptProcessor.
      this._workletChunksSeen = 0;
      const self = this;
      this._workletWatchdog = setTimeout(() => {
        if (self.isCapturing && self.workletNode && self._workletChunksSeen === 0) {
          console.warn("[AudioCapture] Worklet không phát chunk nào sau 2.5s — chuyển sang ScriptProcessor.");
          self._fallbackToScriptProcessor();
        }
      }, 2500);

      this._workletPathActive = true;
      return true;
    } catch (e) {
      console.warn("[AudioCapture] Không dùng được AudioWorklet:", e && e.message ? e.message : e);
      this.workletNode = null;
      this._workletPathActive = false;
      return false;
    }
  }

  /** Chuyển từ worklet sang ScriptProcessor mà không làm gián đoạn capture. */
  _fallbackToScriptProcessor() {
    if (!this._workletPathActive) return;
    this._workletPathActive = false;
    if (this._workletWatchdog) {
      try { clearTimeout(this._workletWatchdog); } catch (e) {}
      this._workletWatchdog = null;
    }
    if (this.workletNode) {
      try {
        this.workletNode.port.onmessage = null;
        this.workletNode.onprocessorerror = null;
        this.workletNode.disconnect();
      } catch (e) {}
      this.workletNode = null;
    }
    try {
      this._setupScriptProcessor();
      console.log("[AudioCapture] Đã chuyển sang ScriptProcessor (dự phòng).");
    } catch (e) {
      console.error("[AudioCapture] Không dựng được ScriptProcessor dự phòng:", e);
    }
  }

  _setupScriptProcessor() {
    const bufferSize = 4096;
    this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    const chunkSize = this.chunkSize || 1024;
    const actualSampleRate = this.audioContext.sampleRate;

    this.processorNode.onaudioprocess = (event) => {
      if (!this.isCapturing || !this.onChunk) return;

      const inputBuffer = event.inputBuffer.getChannelData(0);
      let float16k;

      if (actualSampleRate === 16000) {
        float16k = inputBuffer;
      } else {
        // Continuous phase linear interpolation downsampling to 16kHz
        const ratio = actualSampleRate / 16000.0;
        let srcIdx = this.resamplePhase || 0.0;
        const maxOut = Math.floor((inputBuffer.length - srcIdx) / ratio) + 2;
        const outArray = new Float32Array(Math.max(0, maxOut));
        let outCount = 0;

        while (srcIdx < inputBuffer.length) {
          const i = Math.floor(srcIdx);
          const frac = srcIdx - i;
          const s0 = inputBuffer[i];
          const s1 = (i + 1 < inputBuffer.length) ? inputBuffer[i + 1] : s0;
          outArray[outCount++] = s0 + frac * (s1 - s0);
          srcIdx += ratio;
        }

        this.resamplePhase = srcIdx - inputBuffer.length;
        float16k = (outCount === outArray.length) ? outArray : outArray.subarray(0, outCount);
      }

      // Concatenate residual carryover samples from previous cycle to guarantee ZERO lost audio
      let combined;
      if (this.residualSamples && this.residualSamples.length > 0) {
        combined = new Float32Array(this.residualSamples.length + float16k.length);
        combined.set(this.residualSamples, 0);
        combined.set(float16k, this.residualSamples.length);
      } else {
        combined = float16k;
      }

      const baseTimestamp = this.audioContext ? this.audioContext.currentTime : 0;
      let offset = 0;

      // P3.5c: SỬA TRÔI TIMESTAMP.
      // `combined` bắt đầu bằng các mẫu residual được capture ở callback TRƯỚC, tức là
      // SỚM HƠN `baseTimestamp`. Bản cũ đóng dấu chunk đầu bằng `baseTimestamp + 0`, gây
      // trễ hệ thống 21–64 ms (residual xoay vòng 341/682/1023 mẫu @16kHz). Backend dùng
      // giá trị này để đóng dấu frame VAD (backend/vad/processor.py) nên đây là sai số
      // thật của timing phụ đề.
      const residualSec = (this.residualSamples ? this.residualSamples.length : 0) / 16000.0;

      while (combined.length - offset >= chunkSize) {
        const slice = combined.subarray(offset, offset + chunkSize);
        const pcmData = new Int16Array(chunkSize);
        for (let i = 0; i < chunkSize; i++) {
          pcmData[i] = Math.max(-32768, Math.min(32767, Math.round(slice[i] * 32767)));
        }

        // Monotonic timestamp offset per chunk (đã trừ phần residual còn nợ)
        const chunkTime = baseTimestamp - residualSec + (offset / 16000.0);
        const idx = this.chunkIndex++;
        if (idx === 0 || idx % 200 === 0) {
          console.log(`[AudioCapture] Emitted audio chunk #${idx} (${pcmData.byteLength} bytes, rate: ${actualSampleRate}Hz->16kHz)`);
        }
        this.onChunk(pcmData.buffer, chunkTime, idx);
        offset += chunkSize;
      }

      // Preserve unchunked residual samples for next cycle (0% sample loss)
      if (offset < combined.length) {
        this.residualSamples = combined.slice(offset);
      } else {
        this.residualSamples = new Float32Array(0);
      }
    };

    this.sourceNode.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);
  }

  resumeAudioContext() {
    if (this.audioContext && this.audioContext.state === "suspended") {
      return this.audioContext.resume().catch(() => {});
    }
    return Promise.resolve();
  }

  /**
   * F-44: Xoá audio đang đệm khi video bị TUA.
   *
   * Không có bước này thì phần audio thu trước khi tua (còn nằm trong worklet/ScriptProcessor
   * + residual + pha resample) sẽ được gửi sang backend ngay sau khi tua ⇒ backend trộn
   * audio cũ/mới vào cùng một câu và phụ đề hiện nội dung lặp của đoạn trước.
   */
  reset() {
    // 1. Bỏ đệm trong worklet (mẫu chưa đủ 1 chunk, pha resample, mốc thời gian)
    if (this.workletNode && this.workletNode.port) {
      try {
        this.workletNode.port.postMessage({ type: "reset" });
      } catch (e) {}
    }
    // 2. Bỏ đệm phía main thread (dùng cho đường ScriptProcessor)
    this.residualSamples = new Float32Array(0);
    this.resamplePhase = 0.0;
    this.chunkIndex = 0;
    // 3. Mốc thời gian mới: tính lại từ AudioContext hiện tại (tránh timestamp nhảy âm)
    this.baseTimestamp = this.audioContext ? this.audioContext.currentTime : 0;
    this._workletChunksSeen = 0;
    console.log("[AudioCapture] Đã reset bộ đệm audio (tua video).");
  }

  stop() {
    this.isCapturing = false;
    this.residualSamples = new Float32Array(0);
    this.resamplePhase = 0.0;

    // P3.4: dọn worklet + watchdog
    if (this._workletWatchdog) {
      try { clearTimeout(this._workletWatchdog); } catch (e) {}
      this._workletWatchdog = null;
    }
    if (this.workletNode) {
      try {
        if (this.workletNode.port) this.workletNode.port.onmessage = null;
        this.workletNode.onprocessorerror = null;
        this.workletNode.disconnect();
      } catch (e) {}
      this.workletNode = null;
    }
    this._workletPathActive = false;
    this._workletChunksSeen = 0;

    if (this.processorNode) {
      try {
        if (this.processorNode.port) {
          this.processorNode.port.onmessage = null;
        }
        this.processorNode.onaudioprocess = null;
        this.processorNode.disconnect();
      } catch (e) {}
      this.processorNode = null;
    }

    if (this.sourceNode) {
      try { this.sourceNode.disconnect(); } catch (e) {}
      // If sourceNode was cached on video, re-route it so video audio still plays.
      // QUAN TRỌNG: nối lại QUA `__bsDuckGain` (không nối thẳng vào destination), nếu không
      // lần capture sau sẽ đi vòng qua gain và ducking mất tác dụng. Đồng thời trả gain về 1.0
      // để tiếng gốc hết bị nhỏ khi đã dừng capture.
      if (this.videoElement && this.videoElement.__bsSourceNode && this.audioContext) {
        try {
          const dg = this.videoElement.__bsDuckGain;
          if (dg) {
            try { dg.gain.value = 1.0; } catch (e) {}
            this.videoElement.__bsSourceNode.connect(dg);
          } else {
            this.videoElement.__bsSourceNode.connect(this.audioContext.destination);
          }
        } catch (e) {}
      }
      this.sourceNode = null;
    }

    // Trả ducking về mặc định và quên gain của phiên này (gain vẫn thuộc videoElement).
    this.duckLevel = 1.0;
    this.duckGain = null;

    // Clean up AudioContext if it belongs to a disconnected video element (preventing memory leak)
    if (this.videoElement && !this.videoElement.isConnected && this.videoElement.__bsAudioCtx) {
      try { this.videoElement.__bsAudioCtx.close(); } catch (e) {}
      delete this.videoElement.__bsAudioCtx;
      delete this.videoElement.__bsSourceNode;
      delete this.videoElement.__bsDuckGain;
    }

    // Do NOT close cached target.__bsAudioCtx if video is still active in DOM
    if (this.audioContext && (!this.videoElement || this.audioContext !== this.videoElement.__bsAudioCtx)) {
      try { this.audioContext.close(); } catch (e) {}
    }
    this.audioContext = null;

    if (this.mediaStream) {
      try {
        this.mediaStream.getTracks().forEach((t) => t.stop());
      } catch (e) {}
      this.mediaStream = null;
    }

    this.videoElement = null;
    this.chunkIndex = 0;
    console.log("[AudioCapture] Stopped and cleaned up");
  }
}

if (typeof self !== "undefined") self.AudioCapture = AudioCapture;
if (typeof window !== "undefined") window.AudioCapture = AudioCapture;
