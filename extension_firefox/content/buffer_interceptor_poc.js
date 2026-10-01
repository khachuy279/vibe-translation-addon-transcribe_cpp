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
    bufferEpoch: 0,
    //: Cửa sổ replay (giây): chỉ gửi lại mảnh có mốc video nằm trong [videoTime − 8s,
    //: videoTime + 45s]. Sự cố thật 2026-10-01: phiên neo ở đầu video (1,35 s) nhưng cache
    //: 120 mảnh chỉ chứa media ở tận cuối vùng đã tải (PTS ~144 s trở đi) ⇒ backend phải
    //: lấp ~143 s im lặng mới tới được audio, và phụ đề sinh ra mang mốc ở tương lai nên
    //: không bao giờ hiện. Lọc theo mốc video là cách duy nhất bảo đảm cache LUÔN liên quan
    //: tới vị trí đang phát.
    replayBehindSec: 8,
    replayAheadSec: 45,
    //: Token của lần replay gần nhất. Client gọi `REQUEST_INITIAL_CHUNKS` từ nhiều đường
    //: (kết nối / `lookahead_ready` / `seek_acknowledged`) nên nếu không dedup thì cùng một
    //: cache bị append 2-3 lần — đo thật 2026-10-01: 15.851 frame AAC (368 s audio) chỉ trải
    //: trên 70 s timeline, tức decoder phải giải mã rồi lọc trùng gấp ~5 lần.
    lastReplayToken: null,
    //: Cửa sổ cache (giây) tính theo vị trí phát: giữ [playhead − 30s, playhead + 180s] và bỏ
    //: dần mảnh cũ hơn. Trần cũ "120 mảnh gần nhất" SAI với trang tải trước rất sâu: khi
    //: prefetch nhảy xa, 120 mảnh chỉ toàn đoạn ở tương lai ⇒ mất audio quanh vị trí phát
    //: (đo thật: phiên neo 2,68 s nhưng mảnh đầu tiên đã ở 74,44 s ⇒ 74 giây đầu KHÔNG có
    //: phụ đề, trong khi vẫn "có" hàng trăm mảnh trong cache).
    cacheBehindSec: 30,
    cacheAheadSec: 180,
    cacheMaxBytes: 12 * 1024 * 1024,
    //: Khoảng byte đã tải của từng phân đoạn media: [{url, start, end}] — dùng để TẢI LẠI
    //: vùng đã buffer sẵn (YouTube không append lại nên không có mảnh nào để gửi).
    mediaRanges: [],
    refetchedRanges: new Set(),
    refetchMaxBytes: 6 * 1024 * 1024,
    //: Ước lượng byte/giây của luồng audio (từ lần tải lại thành công) — dùng để quy đổi
    //: "cần audio quanh giây X" thành "khoảng byte nào". Mặc định ~27 KB/s (Opus 128–160k).
    bytesPerSecHint: 0
  };

  /**
   * Giữ cache bám theo VỊ TRÍ PHÁT: bỏ mảnh cũ hơn `cacheBehindSec` và mảnh xa hơn
   * `cacheAheadSec`, đồng thời chặn trần byte. Nhờ vậy cache luôn chứa audio quanh playhead
   * kể cả khi trang tải trước hàng trăm giây (đo thật 2026-10-01).
   */
  function pruneCacheAroundPlayhead(playhead) {
    if (!Number.isFinite(playhead) || playhead <= 0) return;
    const from = playhead - state.cacheBehindSec;
    const to = playhead + state.cacheAheadSec;
    state.cachedAudioChunks = state.cachedAudioChunks.filter((pkt) => {
      const pts = pkt.payload ? pkt.payload.videoPts : undefined;
      if (typeof pts !== 'number' || !Number.isFinite(pts)) return true;
      return pts >= from && pts <= to;
    });
    let bytes = 0;
    for (const pkt of state.cachedAudioChunks) {
      bytes += pkt.rawBytes ? pkt.rawBytes.byteLength : 0;
    }
    while (bytes > state.cacheMaxBytes && state.cachedAudioChunks.length > 1) {
      const dropped = state.cachedAudioChunks.shift();
      bytes -= dropped.rawBytes ? dropped.rawBytes.byteLength : 0;
    }
  }

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
      const token = event.data.replayToken || null;
      if (token && token === state.lastReplayToken) {
        // Cùng một mốc đã replay rồi ⇒ KHÔNG gửi lại (chống append trùng).
        return;
      }
      state.lastReplayToken = token;
      // LUÔN gửi init segment TRƯỚC (kể cả khi cache media đã trôi mất).
      if (state.initSegmentPacket) {
        window.postMessage(state.initSegmentPacket, '*');
      }
      const curTime = Number(event.data.currentTime);
      const video = getActiveVideo();
      const playhead = Number.isFinite(curTime) && curTime > 0
        ? curTime
        : (video ? video.currentTime : 0);
      // Chỉ replay mảnh LIÊN QUAN tới vị trí phát hiện tại. Mảnh chưa biết mốc (append
      // trước khi đọc được video) được gửi như cũ để không mất dữ liệu.
      const from = playhead - state.replayBehindSec;
      const to = playhead + state.replayAheadSec;
      const selected = state.cachedAudioChunks.filter((pkt) => {
        const pts = pkt.payload ? pkt.payload.videoPts : undefined;
        if (typeof pts !== 'number' || !Number.isFinite(pts)) return true;
        return pts >= from && pts <= to;
      });
      const skipped = state.cachedAudioChunks.length - selected.length;
      // Dedup nội bộ: cache có thể chứa nhiều mảnh giống hệt nhau (trang append lại cùng
      // đoạn, hoặc init segment bị coi là media).
      const seen = new Set();
      const unique = [];
      for (const pkt of selected) {
        const rb = pkt.rawBytes;
        const sig = rb
          ? `${rb.byteLength}|${new Uint8Array(rb, 0, Math.min(64, rb.byteLength)).join(',')}`
          : String(Math.random());
        if (seen.has(sig)) continue;
        seen.add(sig);
        unique.push(pkt);
      }
      const dupInCache = selected.length - unique.length;
      if (unique.length > 0) {
        console.log(
          `%c[Lookahead] 📦 Replay ${unique.length} mảnh audio quanh vị trí phát ` +
          `(${playhead.toFixed(1)}s; bỏ ${skipped} mảnh ngoài cửa sổ, ${dupInCache} mảnh trùng).`,
          'color: #a78bfa;');
        for (const pkt of unique) {
          window.postMessage(pkt, '*');
        }
      } else if (state.cachedAudioChunks.length > 0) {
        console.warn(
          `[Lookahead] Cache ${state.cachedAudioChunks.length} mảnh không có mảnh nào quanh ` +
          `vị trí phát ${playhead.toFixed(1)}s — chờ mảnh mới (tránh dịch trước phần quá xa).`);
      }
      // ── VÙNG ĐÃ BUFFER SẴN ──────────────────────────────────────────────────
      // Nếu token là yêu cầu replay từ backend (hoặc cache không có gì quanh vị trí phát) thì
      // TẢI LẠI phân đoạn media cho vùng đó — YouTube không append lại nên đây là cách duy
      // nhất để có audio cho đoạn đã tải trước (đo thật: tua tới 80,5 s mà không có phụ đề).
      const cacheHasNearby = state.cachedAudioChunks.some((pkt) => {
        const pts = pkt.payload ? pkt.payload.videoPts : undefined;
        return typeof pts === 'number' && pts >= playhead - state.replayBehindSec
          && pts <= playhead + state.replayAheadSec;
      });
      if (!cacheHasNearby && state.mediaRanges.length > 0) {
        void refetchForPlayhead(playhead, state.bytesPerSecHint);
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
            const video = getActiveVideo();
            //: Mốc video tại thời điểm append — quyết định mảnh này có còn liên quan tới vị
            //: trí phát hay không khi replay (xem `state.replayAheadSec`).
            const videoPts = video ? Number(video.currentTime) : undefined;
            const chunkPacket = {
              source: 'VIBE_LOOKAHEAD_POC',
              type: 'AUDIO_CHUNK_INTERCEPTED',
              payload: {
                timestampOffset: this.timestampOffset || 0,
                mime: this.__vibeMimeType || state.lastMimeType,
                isInit: isInit,
                epoch: state.bufferEpoch,
                videoPts: videoPts
              },
              rawBytes: rawBytes
            };

            if (isInit) {
              // Init segment: giữ RIÊNG, KHÔNG bao giờ bị đẩy khỏi cache.
              state.initSegmentPacket = chunkPacket;
            } else {
              // Cache bám theo vị trí phát (không phải "120 mảnh gần nhất").
              state.cachedAudioChunks.push(chunkPacket);
              pruneCacheAroundPlayhead(typeof videoPts === 'number' ? videoPts : 0);
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

  // ── GHI LẠI KHOẢNG BYTE CỦA TỪNG PHÂN ĐOẠN MEDIA (để TẢI LẠI khi cần) ──────
  // Vì sao cần: khi người dùng tua vào vùng YouTube ĐÃ tải sẵn, trình phát KHÔNG gọi
  // `appendBuffer` lần nữa ⇒ interceptor không có mảnh nào để gửi, backend không có audio
  // để dịch ⇒ KHÔNG có phụ đề (đo thật 2026-10-01: tua tới 80,5 s nhưng audio nhận được ở
  // 110 s). Giữ lại URL + `Range` của từng phân đoạn cho phép TẢI LẠI đúng khoảng cần.
  function parseRange(header) {
    if (!header) return null;
    const m = /bytes=(\d+)-(\d*)/.exec(String(header));
    if (!m) return null;
    const start = parseInt(m[1], 10);
    const end = m[2] ? parseInt(m[2], 10) : start + 1024 * 1024 - 1;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
    return [start, end];
  }

  function recordMediaRange(url, rangeHeader) {
    const range = parseRange(rangeHeader);
    if (!url || !range || !isAudioUrl(url)) return;
    state.mediaRanges.push({ url: url, start: range[0], end: range[1] });
    if (state.mediaRanges.length > 200) {
      state.mediaRanges.splice(0, state.mediaRanges.length - 200);
    }
  }

  /**
   * Tải lại khoảng byte cho VÙNG CẦN (quanh vị trí phát), dùng chính sách "đã tải rồi" của
   * trình phát. Trả về số byte gửi được.
   *
   * ⚠️ Phải MIỄN dedup cho mảnh tải lại: trước đây backend bỏ nó vì byte giống mảnh đã nhận
   * (log: "Bỏ mảnh audio TRÙNG … 51 … 53") ⇒ cơ chế tải lại không bao giờ cứu được vùng đã buffer.
   */
  async function refetchForPlayhead(playhead, bytesPerSec) {
    if (!state.mediaRanges.length || !Number.isFinite(playhead) || playhead <= 0) return 0;
    const bps = Number.isFinite(bytesPerSec) && bytesPerSec > 2000 ? bytesPerSec : 27000;
    // Byte ước lượng cho vị trí phát, và lùi lại một đoạn để chắc chắn phủ đầu câu.
    const center = Math.max(0, Math.round(playhead * bps) - Math.round(3 * bps));
    // Chọn các range quanh byte đó, ưu tiên range CHƯA tải lại.
    const sorted = state.mediaRanges.slice().sort(
      (a, b) => Math.abs(a.start - center) - Math.abs(b.start - center)
    );
    const picked = [];
    for (const r of sorted) {
      if (state.refetchedRanges.has(`${r.start}-${r.end}`)) continue;
      picked.push(r);
      if (picked.length >= 8) break;
    }
    if (!picked.length) return 0;
    const start = Math.min(...picked.map((r) => r.start));
    const end = Math.max(...picked.map((r) => r.end));
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return 0;
    if (end - start > state.refetchMaxBytes) return 0;
    const url = picked[0].url;
    try {
      const resp = await window.__vibe_orig_fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
      if (!resp || !resp.ok) return 0;
      const buf = await resp.arrayBuffer();
      for (const r of picked) state.refetchedRanges.add(`${r.start}-${r.end}`);
      window.postMessage({
        source: 'VIBE_LOOKAHEAD_POC',
        type: 'AUDIO_CHUNK_INTERCEPTED',
        payload: {
          timestampOffset: 0,
          mime: state.lastMimeType,
          isInit: false,
          epoch: state.bufferEpoch,
          refetched: true          // ⇒ content script gửi kèm cờ này để backend MIỄN dedup
        },
        rawBytes: buf
      }, '*');
      console.log(
        `%c[Lookahead] ⬇️ Tải lại ${buf.byteLength}B (bytes=${start}-${end}) cho vùng quanh ` +
        `${playhead.toFixed(1)}s.`, 'color: #fbbf24;'
      );
      return buf.byteLength;
    } catch (e) {
      console.warn('[Lookahead] Tải lại phân đoạn thất bại:', e);
      return 0;
    }
  }

  /**
   * Tải lại khoảng byte đã ghi cho những phân đoạn CHƯA có trong cache (vùng đã buffer sẵn),
   * rồi gửi sang content script như một mảnh bình thường.
   */
  async function refetchAudioRange(fromByte, toByte, maxBytes) {
    if (!state.mediaRanges.length) return 0;
    const selected = state.mediaRanges.filter(
      (r) => r.end > fromByte && r.start < toByte && !state.refetchedRanges.has(`${r.start}-${r.end}`)
    );
    if (!selected.length) return 0;
    // Gộp thành MỘT request cho cả khoảng (YouTube chấp nhận range lớn).
    const start = Math.min(...selected.map((r) => r.start));
    const end = Math.min(toByte, Math.max(...selected.map((r) => r.end)));
    const url = selected[0].url;
    if (end - start > maxBytes) return 0;
    try {
      const resp = await window.__vibe_orig_fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
      if (!resp || !resp.ok) return 0;
      const buf = await resp.arrayBuffer();
      for (const r of selected) state.refetchedRanges.add(`${r.start}-${r.end}`);
      window.postMessage({
        source: 'VIBE_LOOKAHEAD_POC',
        type: 'AUDIO_CHUNK_INTERCEPTED',
        payload: {
          timestampOffset: 0,
          mime: state.lastMimeType,
          isInit: false,
          epoch: state.bufferEpoch,
          refetched: true
        },
        rawBytes: buf
      }, '*');
      console.log(`%c[Lookahead] ⬇️ Tải lại ${buf.byteLength}B (bytes=${start}-${end}) cho vùng đã buffer sẵn.`, 'color: #fbbf24;');
      return buf.byteLength;
    } catch (e) {
      console.warn('[Lookahead] Tải lại phân đoạn thất bại:', e);
      return 0;
    }
  }

  if (window.fetch && !window.__vibe_orig_fetch) {
    window.__vibe_orig_fetch = window.fetch;
    window.fetch = async function (...args) {
      const url = typeof args[0] === 'string' ? args[0] : (args[0]?.url || '');
      // Ghi lại Range của request để có thể tải lại sau (xem `refetchAudioRange`).
      try {
        const init = args[1] || {};
        const hdrs = init.headers || (args[0] && args[0].headers) || {};
        const rangeVal = typeof hdrs.get === 'function' ? hdrs.get('Range') : (hdrs.Range || hdrs.range);
        recordMediaRange(url, rangeVal);
      } catch (e) {}
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

  // XHR: nhiều trình phát (kể cả YouTube ở một số chế độ) tải phân đoạn bằng XHR.
  if (window.XMLHttpRequest && !window.__vibe_orig_xhr_open) {
    window.__vibe_orig_xhr_open = XMLHttpRequest.prototype.open;
    window.__vibe_orig_xhr_setHeader = XMLHttpRequest.prototype.setRequestHeader;
    XMLHttpRequest.prototype.open = function (method, url, ...rest) {
      try { this.__vibeUrl = url; } catch (e) {}
      return window.__vibe_orig_xhr_open.call(this, method, url, ...rest);
    };
    XMLHttpRequest.prototype.setRequestHeader = function (name, value) {
      try {
        if (String(name).toLowerCase() === 'range') recordMediaRange(this.__vibeUrl, value);
      } catch (e) {}
      return window.__vibe_orig_xhr_setHeader.call(this, name, value);
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
