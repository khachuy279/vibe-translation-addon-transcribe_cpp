/**
 * buffer_interceptor_poc.js — Multi-Layer Lookahead Buffer Interceptor (v1.3)
 *
 * Mục đích:
 * 1. Hook `MediaSource.addSourceBuffer` + `SourceBuffer.appendBuffer` để lấy mảnh audio
 *    tương lai (MSE) — nguồn duy nhất có thể dùng cho Pipeline B.
 * 2. Hook `window.fetch` để đo lượng audio tải qua mạng (chẩn đoán).
 * 3. Tính chính xác thời gian đi trước (Ahead Lead Time = Buffered End − Current Time).
 * 4. Trả trạng thái qua `postMessage` cho content script; POPUP hiển thị trạng thái này.
 *    (v1.2 từng vẽ một HUD nổi luôn hiện trên mọi trang — đã bỏ để không che video.)
 */

(function () {
  'use strict';

  // Cho phép chạy lại script khi paste mới vào Console
  if (window.__VIBE_LOOKAHEAD_INTERVAL__) {
    clearInterval(window.__VIBE_LOOKAHEAD_INTERVAL__);
  }

  console.log('%c[Lookahead] 🚀 Multi-Layer Interceptor đang hoạt động.', 'color: #00ffcc; font-weight: bold; font-size: 13px;');

  const state = {
    audioSourceBuffers: new Set(),
    chunkCount: 0,
    totalBytes: 0,
    lastMimeType: 'detecting...',
    lastTimestampOffset: 0,
    lastAheadSeconds: 0,
    lastVideoTime: 0,
    lastBufferedEnd: 0,
    networkAudioChunks: 0,
    networkBytes: 0,
    startTime: Date.now(),
    recentChunks: [],
    cachedAudioChunks: [],
    // ── GIỮ RIÊNG INIT SEGMENT (bắt buộc) ─────────────────────────────────────
    // Trình duyệt chỉ append init segment (EBML header WebM / ftyp+moov của MP4) MỘT LẦN
    // ở đầu mỗi SourceBuffer. Nếu để nó trôi khỏi cache thì backend KHÔNG BAO GIỜ giải mã
    // được các mảnh về sau (đó chính là lỗi "mất phụ đề" của bản v1).
    initSegmentPacket: null,
    // Mỗi SourceBuffer audio mới = một "epoch" mới. Backend dùng epoch để biết phải xoá
    // bộ đệm ghép nối cũ (header/bitrate đã đổi).
    bufferEpoch: 0
  };

  /**
   * Nhận diện init segment (header) của WebM (EBML) hoặc fMP4 (ftyp/styp không kèm moof).
   */
  function isInitSegment(bytes) {
    if (!bytes || bytes.byteLength < 8) return false;
    const u8 = new Uint8Array(bytes);
    const scanLen = Math.min(u8.length, 1 << 20);
    const hasMagic = (a, b, c, d) => {
      for (let i = 0; i + 4 <= scanLen; i++) {
        if (u8[i] === a && u8[i + 1] === b && u8[i + 2] === c && u8[i + 3] === d) return true;
      }
      return false;
    };
    // WebM / Matroska: EBML magic, và KHÔNG chứa Cluster (0x1F43B675) ⇒ thuần header.
    if (u8[0] === 0x1a && u8[1] === 0x45 && u8[2] === 0xdf && u8[3] === 0xa3) {
      return !hasMagic(0x1f, 0x43, 0xb6, 0x75);
    }
    // fMP4: box đầu là ftyp/styp và KHÔNG chứa moof.
    const boxType = String.fromCharCode(u8[4], u8[5], u8[6], u8[7]);
    if (boxType === 'ftyp' || boxType === 'styp') {
      return !hasMagic(0x6d, 0x6f, 0x6f, 0x66); // 'moof'
    }
    return false;
  }

  /**
   * Ảnh chụp trạng thái buffer để POPUP hiển thị (thay cho HUD nổi của bản v1.2).
   */
  function buildStatusPayload() {
    return {
      hasAudioBuffer: (state.cachedAudioChunks.length > 0 || !!state.initSegmentPacket
        || state.audioSourceBuffers.size > 0 || state.lastAheadSeconds > 1.0),
      chunkCount: state.chunkCount + state.networkAudioChunks,
      aheadSeconds: Number(state.lastAheadSeconds.toFixed(2)),
      cachedChunksCount: state.cachedAudioChunks.length,
      hasInitSegment: !!state.initSegmentPacket,
      epoch: state.bufferEpoch,
      mimeType: state.lastMimeType,
      bufferedEnd: Number(state.lastBufferedEnd.toFixed(2)),
      currentTime: Number(state.lastVideoTime.toFixed(2)),
      megabytes: Number(((state.totalBytes + state.networkBytes) / (1024 * 1024)).toFixed(2)),
    };
  }

  // Lắng nghe yêu cầu từ Extension (content script chạy ở isolated world).
  window.addEventListener('message', (event) => {
    if (event.source !== window || !event.data) return;

    if (event.data.source === 'VIBE_LOOKAHEAD_CLIENT' && event.data.type === 'REQUEST_INITIAL_CHUNKS') {
      // LUÔN gửi init segment TRƯỚC (kể cả khi cache media đã trôi mất).
      if (state.initSegmentPacket) {
        window.postMessage(state.initSegmentPacket, '*');
      }
      if (state.cachedAudioChunks && state.cachedAudioChunks.length > 0) {
        console.log(`%c[Lookahead] 📦 Replay ${state.cachedAudioChunks.length} mảnh audio đã đệm cho Extension...`, 'color: #a78bfa;');
        for (const pkt of state.cachedAudioChunks) {
          window.postMessage(pkt, '*');
        }
      }
    } else if (event.data.source === 'VIBE_LOOKAHEAD_CLIENT' && event.data.type === 'CHECK_BUFFER_STATUS') {
      window.postMessage({
        source: 'VIBE_LOOKAHEAD_POC',
        type: 'BUFFER_STATUS_RESPONSE',
        payload: buildStatusPayload()
      }, '*');
    }
  });

  // Tìm video element đang hiển thị hoặc phát
  function getActiveVideo() {
    const videos = Array.from(document.querySelectorAll('video'));
    if (!videos.length) return null;
    const playing = videos.find(v => !v.paused && v.currentTime > 0);
    if (playing) return playing;
    return videos.reduce((prev, curr) => {
      const prevArea = prev.clientWidth * prev.clientHeight;
      const currArea = curr.clientWidth * curr.clientHeight;
      return currArea > prevArea ? curr : prev;
    }, videos[0]);
  }

  // --- 1. TẦNG 1: HOOK MediaSource & SourceBuffer (MSE Level) ---
  if (window.MediaSource && MediaSource.prototype.addSourceBuffer) {
    if (!MediaSource.prototype.__vibe_orig_addSourceBuffer) {
      MediaSource.prototype.__vibe_orig_addSourceBuffer = MediaSource.prototype.addSourceBuffer;
      MediaSource.prototype.addSourceBuffer = function (mimeType) {
        const sourceBuffer = this.__vibe_orig_addSourceBuffer(mimeType);
        const isAudio = typeof mimeType === 'string' && mimeType.toLowerCase().includes('audio');

        sourceBuffer.__vibeMimeType = mimeType;
        sourceBuffer.__vibeIsAudio = isAudio;

        if (isAudio) {
          // SourceBuffer MỚI ⇒ epoch mới: header/bitrate có thể đã đổi nên mọi mảnh đã
          // cache của epoch cũ phải bị bỏ (backend cũng reset theo epoch).
          state.bufferEpoch += 1;
          state.cachedAudioChunks = [];
          state.initSegmentPacket = null;
          state.audioSourceBuffers.clear();
          state.audioSourceBuffers.add(sourceBuffer);
          state.lastMimeType = mimeType;
          console.log(`%c[Lookahead] 🎵 Audio SourceBuffer (epoch ${state.bufferEpoch}): ${mimeType}`, 'color: #38bdf8; font-weight: bold;');
        }
        return sourceBuffer;
      };
    }
  }

  if (window.SourceBuffer && SourceBuffer.prototype.appendBuffer) {
    if (!SourceBuffer.prototype.__vibe_orig_appendBuffer) {
      SourceBuffer.prototype.__vibe_orig_appendBuffer = SourceBuffer.prototype.appendBuffer;
      SourceBuffer.prototype.appendBuffer = function (data) {
        const isAudio = this.__vibeIsAudio || (this.__vibeMimeType && this.__vibeMimeType.includes('audio'));

        if (isAudio || !this.__vibeMimeType) {
          const byteLen = data ? (data.byteLength || data.length || 0) : 0;
          state.chunkCount++;
          state.totalBytes += byteLen;
          if (this.__vibeMimeType) state.lastMimeType = this.__vibeMimeType;

          recordChunkMeta('MSE_SourceBuffer', byteLen, this.timestampOffset || 0);

          // Trích xuất binary bytes để gửi sang Content Script
          let rawBytes = null;
          if (data) {
            try {
              if (data instanceof ArrayBuffer) {
                rawBytes = data.slice(0);
              } else if (data.buffer instanceof ArrayBuffer) {
                rawBytes = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
              }
            } catch (e) {}
          }

          if (rawBytes) {
            const isInit = isInitSegment(rawBytes);
            const chunkPacket = {
              source: 'VIBE_LOOKAHEAD_POC',
              type: 'AUDIO_CHUNK_INTERCEPTED',
              payload: {
                timestampOffset: this.timestampOffset || 0,
                mime: this.__vibeMimeType || state.lastMimeType,
                isInit: isInit,
                epoch: state.bufferEpoch
              },
              rawBytes: rawBytes
            };

            if (isInit) {
              // Init segment: giữ RIÊNG, KHÔNG bao giờ bị đẩy khỏi cache.
              state.initSegmentPacket = chunkPacket;
            } else {
              // Lưu đệm để replay khi Extension kết nối sau (120 mảnh ~ 4 phút audio).
              state.cachedAudioChunks.push(chunkPacket);
              if (state.cachedAudioChunks.length > 120) {
                state.cachedAudioChunks.shift();
              }
            }

            window.postMessage(chunkPacket, '*');
          }
        }
        return this.__vibe_orig_appendBuffer(data);
      };
    }
  }

  // --- 2. TẦNG 2: HOOK Network Requests (Fetch - Audio Chunks, chẩn đoán) ---
  function isAudioUrl(url) {
    if (!url || typeof url !== 'string') return false;
    const isYtAudio = url.includes('videoplayback') && (
      url.includes('mime=audio') ||
      url.includes('itag=140') || // M4A / AAC 128k
      url.includes('itag=251') || // Opus 160k
      url.includes('itag=250') || // Opus 70k
      url.includes('itag=249')    // Opus 50k
    );
    const isBiliAudio = url.includes('.m4s') || url.includes('audio') || url.includes('range=');
    return isYtAudio || isBiliAudio;
  }

  if (window.fetch && !window.__vibe_orig_fetch) {
    window.__vibe_orig_fetch = window.fetch;
    window.fetch = async function (...args) {
      const url = typeof args[0] === 'string' ? args[0] : (args[0]?.url || '');
      const response = await window.__vibe_orig_fetch.apply(this, args);

      if (isAudioUrl(url)) {
        try {
          const clone = response.clone();
          clone.arrayBuffer().then(buf => {
            state.networkAudioChunks++;
            state.networkBytes += buf.byteLength;
            if (url.includes('opus')) state.lastMimeType = 'audio/webm; codecs="opus" (Network)';
            else if (url.includes('mime=audio')) state.lastMimeType = 'audio/mp4 (Network)';
            else if (state.lastMimeType === 'detecting...') state.lastMimeType = 'audio/stream (Network)';
          }).catch(() => {});
        } catch (e) {}
      }
      return response;
    };
  }

  // Ghi nhận metadata của chunk audio thu thập được (KHÔNG vẽ DOM).
  function recordChunkMeta(source, byteLen, timestampOffset) {
    const video = getActiveVideo();
    const currentTime = video ? video.currentTime : 0;
    state.lastVideoTime = currentTime;

    let bufferedEnd = currentTime;
    if (video && video.buffered && video.buffered.length > 0) {
      for (let i = 0; i < video.buffered.length; i++) {
        if (currentTime >= video.buffered.start(i) && currentTime <= video.buffered.end(i)) {
          bufferedEnd = video.buffered.end(i);
          break;
        }
        bufferedEnd = video.buffered.end(video.buffered.length - 1);
      }
    }

    state.lastBufferedEnd = bufferedEnd;
    state.lastAheadSeconds = Math.max(0, bufferedEnd - currentTime);

    state.recentChunks.push({
      id: state.chunkCount + state.networkAudioChunks,
      source: source,
      bytes: byteLen,
      currentTime: Number(currentTime.toFixed(2)),
      bufferedEnd: Number(bufferedEnd.toFixed(2)),
      aheadSeconds: Number(state.lastAheadSeconds.toFixed(2)),
      mime: state.lastMimeType.split(';')[0]
    });
    if (state.recentChunks.length > 50) state.recentChunks.shift();
  }

  // --- 3. TẦNG 3: VÒNG LẶP ĐỌC TRỰC TIẾP TỪ VIDEO ELEMENT ---
  window.__VIBE_LOOKAHEAD_INTERVAL__ = setInterval(() => {
    const video = getActiveVideo();
    if (!video) return;
    state.lastVideoTime = video.currentTime;
    if (video.buffered && video.buffered.length > 0) {
      let matchedEnd = 0;
      for (let i = 0; i < video.buffered.length; i++) {
        if (video.currentTime >= video.buffered.start(i) - 0.5 && video.currentTime <= video.buffered.end(i) + 0.5) {
          matchedEnd = video.buffered.end(i);
          break;
        }
      }
      if (!matchedEnd) matchedEnd = video.buffered.end(video.buffered.length - 1);

      state.lastBufferedEnd = matchedEnd;
      state.lastAheadSeconds = Math.max(0, matchedEnd - video.currentTime);

      if (state.lastMimeType === 'detecting...') {
        const src = video.src || '';
        state.lastMimeType = src.startsWith('blob:') ? 'video/blob-mse (YouTube)' : 'video/html5-direct';
      }
    }
  }, 400);

  // Debug API (chỉ trong console của trang)
  window.__VIBE_LOOKAHEAD_DEBUG__ = {
    getState: () => state,
    getStatus: () => buildStatusPayload(),
    printReport: () => {
      const s = buildStatusPayload();
      console.group('%c📊 [Lookahead Buffer Report]', 'color: #38bdf8; font-weight: bold; font-size: 14px;');
      console.log(`Audio Format: ${s.mimeType}`);
      console.log(`Playback Time: ${s.currentTime.toFixed(2)}s`);
      console.log(`Buffered End: ${s.bufferedEnd.toFixed(2)}s`);
      console.log(`Buffer Ahead (Lead Time): +${s.aheadSeconds.toFixed(2)}s`);
      console.log(`Total Chunks: ${s.chunkCount} (${s.megabytes.toFixed(2)} MB)`);
      console.log(`Init segment: ${s.hasInitSegment ? 'có' : 'chưa'} | epoch ${s.epoch}`);
      console.table(state.recentChunks.slice(-10));
      console.groupEnd();
    }
  };

  console.log('%c[Lookahead] ✅ Interceptor sẵn sàng (trạng thái hiển thị trong Popup của Extension).', 'color: #00ffcc;');
})();
