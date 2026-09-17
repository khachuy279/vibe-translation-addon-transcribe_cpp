// Content Script - Full Pipeline: audio capture + WebSocket + overlay
(function () {
  if (window.__bsContentScriptLoaded) return;
  window.__bsContentScriptLoaded = true;
  const api = typeof browser !== "undefined" ? browser : chrome;

  // ── Unified WebSocket Configuration Builder ─────────────────────────────
  // G9/G10: KHÔNG gửi giá trị mặc định cứng cho `translationModel` / `vadEngine` nữa.
  // Trước đây client luôn gửi "xiaomi" và "fsmn-vad":
  //   - "xiaomi" (MiLMMT) không có file GGUF cục bộ -> khi backend xử lý thật sẽ lỗi.
  //   - "fsmn-vad" khác engine mặc định của backend (fired-vad) -> mỗi lần content
  //     script khởi động lại ghi đè VAD engine và kích hoạt đường nạp model trên hot path.
  // Nay chỉ gửi khi người dùng THỰC SỰ chọn; bỏ trống thì backend giữ mặc định của nó.
  // F-30: extension này khai báo giao thức v3 ⇒ backend gửi payload GỌN (một tên cho mỗi
  // giá trị, không còn `original`/`ui_text`/`utteranceId`/`stableText`…).
  const PROTOCOL_VERSION = 3;

  function buildWsConfig(cfg) {
    const silence = cfg.silenceDurationMs || cfg.vadSilenceDurationMs || cfg.silence_duration_ms || 300;
    const out = {
      type: "set_config",
      action: "configure",
      // P3.0: khai báo phiên bản giao thức để backend biết có được dùng binary frame.
      protocolVersion: PROTOCOL_VERSION,
      targetLang: cfg.targetLang || "vi",
      sourceLang: cfg.sourceLanguage || cfg.sourceLang || "auto",
      vadThreshold: cfg.vadThreshold !== undefined ? cfg.vadThreshold : (cfg.vad_threshold !== undefined ? cfg.vad_threshold : (cfg.threshold !== undefined ? cfg.threshold : 0.5)),
      threshold: cfg.vadThreshold !== undefined ? cfg.vadThreshold : (cfg.vad_threshold !== undefined ? cfg.vad_threshold : (cfg.threshold !== undefined ? cfg.threshold : 0.5)),
      silenceDurationMs: silence,
      minWordsToCommit: cfg.minWordsToCommit !== undefined && !isNaN(parseInt(cfg.minWordsToCommit, 10)) ? Math.max(0, parseInt(cfg.minWordsToCommit, 10)) : 2,
      ttsEnabled: !!cfg.ttsEnabled,
      ttsVoice: cfg.ttsVoice || "speaker_01_0039.wav",
      ttsSpeed: parseFloat(cfg.ttsSpeed || 1.0),
      ttsInstruct: cfg.ttsInstruct || "",
      ttsRefAudio: cfg.ttsRefAudio || "",
      ttsRefText: cfg.ttsRefText || "",
    };
    if (cfg.translationModel) out.translationModel = cfg.translationModel;
    if (cfg.vadEngine) out.vadEngine = cfg.vadEngine;
    return out;
  }

  const ttsPlayer = new TTSAudioPlayer();

  let wsClient = null, audioCapture = null, overlayManager = null;
  let captureAbortController = null;
  let isCapturing = false;
  let settings = { targetLang: "vi", subPosY: 10, subWidth: 80 };
  let cachedVideo = null;
  // FIX-01: handle của "liveness probe" backpressure. Khi socket nghẽn tới mức HARD,
  // content script tắt capture nên không còn frame nào để service worker tự kiểm tra lại;
  // timer này hỏi định kỳ cho tới khi nó báo "ok".
  let backpressureProbeTimer = null;
  const BACKPRESSURE_PROBE_MS = 250;
  // P3.5e: cache cả KẾT QUẢ ÂM của findVideo(). Trước đây nếu không tìm thấy video thì
  // `cachedVideo = null` và mọi sự kiện sau lại quét toàn DOM (`querySelectorAll("*")`)
  // — trên trang nhiều frame / DOM lớn đây là mục CPU chiếm ưu thế.
  let videoScanMissUntil = 0;
  const VIDEO_SCAN_MISS_TTL_MS = 2000;
  // B10-1 (Hy3): hai chốt chặn cho lần quét Shadow DOM đắt tiền.
  //  - MAX_SHADOW_SCAN_NODES: trần số phần tử duyệt trong MỘT lần quét. `querySelectorAll("*")`
  //    trên trang lớn (YouTube/Bilibili) cấp phát NodeList hàng trăm nghìn node và chặn
  //    main thread; TreeWalker + trần node giữ chi phí có chặn trên.
  //  - VIDEO_SCAN_MIN_INTERVAL_MS: khoảng nghỉ tối thiểu giữa hai lần quét. Trước đây TTL 2 s
  //    chỉ áp dụng cho đường `getVideo()`; các chỗ gọi thẳng `findVideo()` (popup, status,
  //    attach) vẫn quét lại toàn DOM mỗi lần.
  const MAX_SHADOW_SCAN_NODES = 20000;
  const VIDEO_SCAN_MIN_INTERVAL_MS = 250;
  let lastVideoScanAt = 0;

  /**
   * FIX-01: bắt đầu hỏi service worker xem socket đã rút hết hàng đợi chưa.
   * Chỉ chạy khi capture ĐANG bị tạm dừng (không còn frame nào để nó tự biết).
   */
  function startBackpressureProbe() {
    if (backpressureProbeTimer !== null) return;
    backpressureProbeTimer = setInterval(() => {
      if (!wsClient) {
        stopBackpressureProbe();
        return;
      }
      // Service worker sẽ trả "ok" (qua sự kiện backpressure) khi bufferedAmount < SOFT.
      wsClient.flushPending();
    }, BACKPRESSURE_PROBE_MS);
  }

  function stopBackpressureProbe() {
    if (backpressureProbeTimer === null) return;
    try { clearInterval(backpressureProbeTimer); } catch (e) {}
    backpressureProbeTimer = null;
  }

  function getVideo(forceRefresh = false) {
    if (!forceRefresh && cachedVideo && cachedVideo.isConnected && !cachedVideo.ended) {
      return cachedVideo;
    }
    const now = Date.now();
    if (!forceRefresh && now < videoScanMissUntil) {
      return null; // vừa quét không thấy -> không quét lại ngay
    }
    cachedVideo = findVideo();
    if (!cachedVideo) {
      videoScanMissUntil = now + VIDEO_SCAN_MISS_TTL_MS;
    } else {
      videoScanMissUntil = 0;
    }
    return cachedVideo;
  }

  function ensureOverlay(targetVideo = null) {
    const video = targetVideo || getVideo();
    if (!overlayManager) {
      overlayManager = new OverlayManager();
      overlayManager.init(video);
      overlayManager.applySettings(settings);
    } else if (video) {
      overlayManager.attachToVideo(video);
    }
    return overlayManager;
  }

  function handleSubtitleEvent(eventType, payload) {
    if (eventType === "tts_audio") {
      // P3.5b/F-28: CHỈ MỘT frame được phát.
      // Trước đây điều kiện là `window === window.top || isCapturing`: khi player nằm
      // trong iframe thì iframe (isCapturing) VÀ top frame (broadcast echo) cùng phát
      // => tiếng lồng bị phát 2 lần lệch nhau (playedIds là per-frame nên không chặn được).
      // Quy tắc đúng: frame SỞ HỮU video phát; nếu không frame nào có video thì frame
      // đang capture phát.
      const ownerVideo = cachedVideo && cachedVideo.isConnected;
      const shouldPlay = ownerVideo ? (window !== window.top || isCapturing) : isCapturing;
      if (shouldPlay) {
        // v3 (F-30): payload gọn — chỉ `audio`, `utterance_id`, `duration_sec`.
        const audioB64 = payload?.audio || (typeof payload === "string" ? payload : null);
        if (audioB64) {
          ttsPlayer.enqueue({
            id: payload?.utterance_id,
            audioBase64: audioB64,
            durationSec: payload?.duration_sec,
            text: payload?.text
          });
        }
      }
      return;
    }

    const video = getVideo();
    // 1. If this frame HAS the video element, render overlay directly inside this frame
    if (video) {
      const om = ensureOverlay(video);
      if (eventType === "partial_transcript") om.onPartialTranscript(payload);
      else if (eventType === "utterance_update") om.onUtteranceUpdate(payload);
      else if (eventType === "sentence_complete") om.onSentenceComplete(payload);
      else if (eventType === "translation") om.onTranslation(payload);
      return;
    }

    // 2. If this frame is actively capturing audio (even if video ref is temporary null)
    if (isCapturing) {
      const om = ensureOverlay(null);
      if (eventType === "partial_transcript") om.onPartialTranscript(payload);
      else if (eventType === "utterance_update") om.onUtteranceUpdate(payload);
      else if (eventType === "sentence_complete") om.onSentenceComplete(payload);
      else if (eventType === "translation") om.onTranslation(payload);
      return;
    }

    // If this frame has no video and is not capturing (e.g. top window hosting an iframe player),
    // do not render a fallback overlay on body to avoid duplicate or misplaced subtitles.
  }

  api.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    switch (msg.type || msg.action) {
      case "START_TRANSLATION": case "start_capture":
        if (msg.settings) Object.assign(settings, msg.settings);
        const v = findVideo();
        if (v) {
          ensureOverlay(v);
        }
        startCapture(msg).then(r => sendResponse?.(r));
        return true;
      case "STOP_TRANSLATION": case "stop_capture":
        stopCapture();
        if (overlayManager) {
          overlayManager.destroy();
          overlayManager = null;
        }
        sendResponse?.({ success: true });
        break;
      case "SUBTITLE_RENDER":
        handleSubtitleEvent(msg.eventType, msg.payload);
        break;
      case "GET_STATUS": case "content_status":
        sendResponse?.({ isCapturing, hasVideo: !!findVideo(), overlayActive: !!overlayManager });
        break;
      case "set_overlay_mode":
        if (msg.payload?.mode) settings.overlayStyle = msg.payload.mode;
        if (overlayManager) overlayManager.setMode(settings.overlayStyle);
        break;
      case "update_settings":
        if (msg.settings) {
          Object.assign(settings, msg.settings);
          if (overlayManager) overlayManager.applySettings(settings);
          if (ttsPlayer) {
            if (!ttsPlayer.targetVideo || !ttsPlayer.targetVideo.isConnected) {
              const v = getVideo();
              if (v) ttsPlayer.setTargetVideo(v, settings.ttsDucking !== false, settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25, !!settings.ttsEnabled);
            }
            ttsPlayer.applySettings(
              settings.ttsDucking !== false,
              settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25,
              !!settings.ttsEnabled
            );
          }
        }
        if (wsClient && wsClient.isConnected && msg.settings) {
          wsClient.sendJSON(buildWsConfig(settings));
          console.log("[BS] Settings updated live:", msg.settings);
        }
        sendResponse?.({ success: true, settings });
        break;
    }
  });

  async function startCapture(msg) {
    // F-45: nếu cờ `isCapturing` còn kẹt từ phiên trước nhưng WS đã đóng thì dọn trước,
    // để bấm Start không bị "Already capturing" (trước đây phải tải lại trang).
    if (isCapturing && (!wsClient || !wsClient.isConnected)) {
      console.warn("[BS] Trạng thái capture cũ đã kẹt — dọn dẹp rồi Start lại.");
      try {
        await cleanup();
      } catch (e) {}
    }
    if (isCapturing) return { success: false, error: "Already capturing" };
    try {
      if (msg.settings) Object.assign(settings, msg.settings);
      let video = findVideo();
      if (!video) {
        // Retry for up to ~1s if video element is lazily loaded upon play
        for (let i = 0; i < 4; i++) {
          await new Promise(r => setTimeout(r, 250));
          video = findVideo();
          if (video) break;
        }
      }
      if (!video) return { success: false, error: "No video found" };

      // Initialize AbortController for clean listener lifecycle management
      captureAbortController = new AbortController();
      const { signal } = captureAbortController;

      ttsPlayer.setTargetVideo(
        video,
        settings.ttsDucking !== false,
        settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25,
        !!settings.ttsEnabled
      );
      ttsPlayer.clear();

      // F-44: TUA VIDEO => phải reset pipeline, nếu không audio trước/sau khi tua bị trộn
      // vào cùng một câu (đã gặp phụ đề lặp nội dung cũ, mảnh commit dài cả phút).
      const handleSeek = (event) => {
        const reason = event && event.type === "seeking" ? "seeking" : "seek";
        try {
          ttsPlayer.clear();
        } catch (e) {}
        // 1. Xoá phụ đề đang hiển thị (đã thuộc đoạn cũ)
        if (overlayManager) {
          try {
            overlayManager.clear();
          } catch (e) {}
        }
        // 2. Bỏ audio còn đệm trong worklet/ScriptProcessor
        if (audioCapture && typeof audioCapture.reset === "function") {
          try {
            audioCapture.reset();
          } catch (e) {}
        }
        // 3. Báo backend xoá audio/commit/hàng đợi cũ
        if (wsClient) {
          try {
            wsClient.sendJSON({ type: "reset_stream", reason });
          } catch (e) {}
        }
      };
      // `seeking` bắt ngay khi người dùng kéo thanh thời gian (audio cũ dừng sớm nhất);
      // `seeked` là chốt cuối sau khi trình duyệt nhảy xong.
      video.addEventListener("seeking", handleSeek, { signal });
      video.addEventListener("seeked", handleSeek, { signal });

      // Connect to WebSocket Backend
      wsClient = new WSClient("wss://localhost:8765/ws");
      wsClient.on("connected", () => {
        wsClient.sendJSON(buildWsConfig(settings));
      });

      function emitSubtitleEvent(eventType, payload) {
        // 1. Render locally if eligible
        handleSubtitleEvent(eventType, payload);

        // 2. Broadcast to other frames (Top frame) only if inside a child iframe
        if (window !== window.top) {
          try {
            api.runtime.sendMessage({
              action: "BROADCAST_SUBTITLE",
              eventType,
              payload
            }).catch(() => {});
          } catch (e) {}
        }
      }

      wsClient.on("partial_transcript", p => emitSubtitleEvent("partial_transcript", p));
      wsClient.on("utterance_update", p => emitSubtitleEvent("utterance_update", p));
      wsClient.on("translation", p => emitSubtitleEvent("translation", p));
      wsClient.on("tts_audio", p => emitSubtitleEvent("tts_audio", p));
      wsClient.on("error", p => console.error("[BS] Backend:", p));

      // P1.7/P1.8: trạng thái nạp model — hiển thị rõ thay vì "im lặng vài giây".
      wsClient.on("model_status", p => {
        const msg = `[BS] Model ${p?.stage || "?"}: ${p?.model || ""} -> ${p?.state || "?"}`;
        if (p?.state === "error") console.warn(msg, p?.message || "");
        else console.log(msg);
      });

      // P3.1: audio TTS dạng binary (không base64) — decode off-thread.
      wsClient.on("tts_binary", (data) => {
        try {
          const buf = data instanceof ArrayBuffer ? data : (data && data.buffer) || null;
          if (!buf || buf.byteLength < 9) return;
          // Header nhỏ: magic "BTTS" + uint8 version + uint16 jsonLen + JSON header
          const view = new DataView(buf);
          const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
          if (magic !== "BTTS") return;
          const jsonLen = view.getUint16(5, true);
          const headerJson = new TextDecoder().decode(new Uint8Array(buf, 7, jsonLen));
          const header = JSON.parse(headerJson);
          const ownerVideo = cachedVideo && cachedVideo.isConnected;
          const shouldPlay = ownerVideo ? (window !== window.top || isCapturing) : isCapturing;
          if (shouldPlay) ttsPlayer.enqueueBinary(header, buf.slice(7 + jsonLen));
        } catch (e) {
          console.warn("[BS TTS] Không xử lý được binary TTS:", e);
        }
      });

      // F-45: WebSocket ĐÓNG (backend restart / phiên bị đóng) ⇒ dừng capture NGAY để
      // trạng thái không kẹt ở "đang capture". Trước đây cờ `isCapturing` vẫn true nên
      // popup báo Disconnected mà bấm Start lại bị "Already capturing", buộc phải tải lại trang.
      wsClient.on("disconnected", () => {
        if (!isCapturing) return;
        console.warn("[BS] Mất kết nối backend — dừng capture để có thể Start lại.");
        try {
          stopCapture();
        } catch (e) {}
      });

      wsClient.on("backpressure", (bp) => {
        if (bp.state === "paused") {
          if (audioCapture && audioCapture.isCapturing) {
            console.warn(
              "[BS] Backend chậm: tạm dừng gửi audio để tránh trôi độ trễ.",
              `(pause #${bp.pauseCount || 0}, đã mất ${bp.droppedFrames || 0} frame)`
            );
            audioCapture.isCapturing = false;
          }
          // FIX-01: capture đã dừng nên KHÔNG còn SEND_BINARY nào để service worker biết
          // lúc nào hàng đợi rút hết. Bắn probe định kỳ để nó nhả trạng thái tạm dừng.
          startBackpressureProbe();
        } else if (bp.state === "ok") {
          stopBackpressureProbe();
          if (audioCapture && !audioCapture.isCapturing) {
            audioCapture.isCapturing = true;
            console.log(
              "[BS] Backend đã bắt kịp: tiếp tục gửi audio.",
              `(paused ${Math.round((bp.pausedMs || 0))}ms, resume #${bp.resumeCount || 0})`
            );
          }
        }
      });

      await wsClient.connect();

      isCapturing = true;

      // Start Audio Capture module
      audioCapture = new AudioCapture();
      audioCapture.onChunk = (pcmBuffer, timestamp, chunkIdx) => {
        if (wsClient && wsClient.isConnected) {
          wsClient.sendBinary(pcmBuffer, timestamp, chunkIdx);
        }
      };
      await audioCapture.start(video);

      // Initialize overlay for Top window or if already in Fullscreen
      if (window === window.top || document.fullscreenElement) {
        ensureOverlay(video);
      }

      // Ensure AudioContext and Fullscreen overlay transitions
      const handleStateKeepAlive = () => {
        if (audioCapture) {
          audioCapture.resumeAudioContext();
        }
        if (document.fullscreenElement && !overlayManager) {
          ensureOverlay(video);
        }
      };

      document.addEventListener("fullscreenchange", handleStateKeepAlive, { signal });
      document.addEventListener("webkitfullscreenchange", handleStateKeepAlive, { signal });
      document.addEventListener("mozfullscreenchange", handleStateKeepAlive, { signal });
      if (video) {
        video.addEventListener("play", handleStateKeepAlive, { signal });
        video.addEventListener("playing", handleStateKeepAlive, { signal });
      }

      console.log("[BS] Capture started");
      return { success: true };
    } catch (e) {
      console.error("[BS] Start error:", e);
      await cleanup();
      const errText = e?.message || (typeof e === "string" ? e : "Không thể kết nối tới Backend WebSocket (wss://localhost:8765/ws)");
      return { success: false, error: errText };
    }
  }

  function stopCapture() { cleanup(); return { success: true }; }

  async function cleanup() {
    isCapturing = false;
    stopBackpressureProbe();   // FIX-01: không để timer probe sống sót qua Stop
    if (captureAbortController) {
      try { captureAbortController.abort(); } catch (e) {}
      captureAbortController = null;
    }
    ttsPlayer.destroy();
    if (audioCapture) {
      try { audioCapture.stop(); } catch (e) {}
      audioCapture = null;
    }
    if (wsClient) {
      try { wsClient.disconnect(); } catch (e) {}
      wsClient = null;
    }
    if (overlayManager) {
      try { overlayManager.destroy(); } catch (e) {}
      overlayManager = null;
    }
  }

  // Global helpers for cross-frame scripting
  window.__bsStartCapture = (msg) => {
    if (msg?.settings) Object.assign(settings, msg.settings);
    const v = findVideo();
    if (v) ensureOverlay(v);
    return startCapture(msg || {});
  };
  window.__bsStopCapture = () => {
    stopCapture();
    if (overlayManager) {
      overlayManager.destroy();
      overlayManager = null;
    }
    return { success: true };
  };
  window.__bsGetStatus = () => ({ isCapturing, hasVideo: !!findVideo(), overlayActive: !!overlayManager });
  window.__bsUpdateSettings = (s) => {
    if (s) {
      Object.assign(settings, s);
      if (overlayManager) overlayManager.applySettings(settings);
      if (ttsPlayer) {
        if (!ttsPlayer.targetVideo || !ttsPlayer.targetVideo.isConnected) {
          const v = getVideo();
          if (v) ttsPlayer.setTargetVideo(v, settings.ttsDucking !== false, settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25, !!settings.ttsEnabled);
        }
        ttsPlayer.applySettings(
          settings.ttsDucking !== false,
          settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25,
          !!settings.ttsEnabled
        );
      }
    }
    if (wsClient && wsClient.isConnected && s) {
      wsClient.sendJSON(buildWsConfig(settings));
      console.log("[BS] Settings updated live:", s);
    }
    return { success: true, settings };
  };

  function findVideo() {
    if (cachedVideo && cachedVideo.isConnected && !cachedVideo.ended) return cachedVideo;

    const allVideos = [];

    // 1. Light DOM: `document.querySelectorAll("video")` do engine native thực hiện nên rất
    //    nhanh kể cả trên DOM lớn.
    try {
      document.querySelectorAll("video").forEach(v => allVideos.push(v));
    } catch (e) {}

    // B10-1 (Hy3): chỉ đi tiếp vào Shadow DOM / iframe khi light DOM CHƯA có video dùng được
    // VÀ đã qua khoảng nghỉ tối thiểu. Nhờ vậy đường nóng (video ở light DOM — gần như mọi
    // trang) không bao giờ phải duyệt cây.
    const lightUsable =
      allVideos.some(v => !v.paused && !v.ended) || allVideos.some(v => v.readyState > 0);
    const now = Date.now();
    if (!lightUsable && now - lastVideoScanAt >= VIDEO_SCAN_MIN_INTERVAL_MS) {
      lastVideoScanAt = now;
      let budget = MAX_SHADOW_SCAN_NODES;

      // 2. Scan Shadow DOMs (TreeWalker: KHÔNG cấp phát NodeList cho cả cây).
      function collectVideosAndShadowRoots(root) {
        if (!root || budget <= 0) return;
        try {
          if (!root.querySelectorAll) return;
          root.querySelectorAll("video").forEach(v => allVideos.push(v));
          const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
          let el = walker.nextNode();
          while (el && budget > 0) {
            budget--;
            if (el.shadowRoot) {
              collectVideosAndShadowRoots(el.shadowRoot);
            }
            el = walker.nextNode();
          }
        } catch (e) {}
      }
      collectVideosAndShadowRoots(document.body || document.documentElement);

      // 3. Scan accessible same-origin/child iframes
      try {
        document.querySelectorAll("iframe").forEach(iframe => {
          if (budget <= 0) return;
          try {
            const doc = iframe.contentDocument || iframe.contentWindow?.document;
            if (doc) {
              collectVideosAndShadowRoots(doc.body || doc.documentElement);
            }
          } catch (e) {}
        });
      } catch (e) {}
    }

    if (!allVideos.length) return null;

    // 3. Rank videos: prioritize actively playing, readyState > 0, valid src, or largest dimensions
    let bestVideo = allVideos[0];
    let maxScore = -1;

    for (const v of allVideos) {
      let score = 0;
      const isPlaying = !v.paused && !v.ended;
      if (isPlaying && v.readyState > 2) score += 1000000;
      else if (isPlaying) score += 500000;
      else if (v.readyState > 0) score += 100000;

      if (v.currentTime > 0) score += 50000;
      if (v.src || v.currentSrc || v.srcObject) score += 20000;

      const rect = v.getBoundingClientRect ? v.getBoundingClientRect() : { width: 0, height: 0 };
      const displayArea = (rect.width * rect.height) || 0;
      const videoArea = (v.videoWidth * v.videoHeight) || 0;
      score += Math.max(displayArea, videoArea);

      if (score > maxScore) {
        maxScore = score;
        bestVideo = v;
      }
    }
    return bestVideo;
  }

  window.__bsFindVideo = findVideo;

  console.log("[BS Content] Ready (Frame: " + (window === window.top ? "Top" : "Iframe") + ")");
})();
