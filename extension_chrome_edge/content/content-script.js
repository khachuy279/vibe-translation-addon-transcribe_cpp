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

  // Tự động inject Buffer Interceptor vào Page Context (Main World)
  function injectBufferInterceptor() {
    try {
      api.runtime.sendMessage({ action: "INJECT_MAIN_WORLD_INTERCEPTOR" }).catch(() => {});
      if (document.getElementById("vibe-lookahead-injected-script")) return;
      const s = document.createElement("script");
      s.id = "vibe-lookahead-injected-script";
      s.src = api.runtime.getURL("content/buffer_interceptor_poc.js");
      s.onload = function () { this.remove(); };
      (document.head || document.documentElement).appendChild(s);
    } catch (e) {}
  }
  injectBufferInterceptor();

  let lookaheadClient = null;
  let timelineQueue = null;
  let ttsTimeline = null;
  //: Pipeline B tự tạm dừng video để nạp đệm trước ⇒ phải nhớ để TRẢ LẠI trạng thái phát
  //: khi Stop hoặc khi fallback sang Pipeline A (nếu không, video kẹt ở trạng thái pause).
  let lookaheadPausedVideo = false;
  let lookaheadPauseForBuffering = null;
  let activePipeline = "A";
  //: Thế hệ phiên capture — tăng mỗi lần Start VÀ mỗi lần Stop/cleanup. Dùng để chặn
  //: pipeline "hồi sinh" sau khi người dùng đã bấm Stop (xem `startCapture`).
  let captureGeneration = 0;

  function buildWsConfig(cfg) {
    // 0 = OFF: để backend dùng mặc định trong docs của VAD engine đang chọn
    // (FireRed 600 ms · Silero 100 ms · FSMN 800 ms). KHÔNG dùng `||` vì 0 là giá trị hợp lệ
    // và có nghĩa ("off") — `|| 300` cũ sẽ âm thầm biến OFF thành 300 ms.
    const rawSilence =
      cfg.silenceDurationMs !== undefined ? cfg.silenceDurationMs :
      (cfg.vadSilenceDurationMs !== undefined ? cfg.vadSilenceDurationMs :
        (cfg.silence_duration_ms !== undefined ? cfg.silence_duration_ms : 0));
    const parsedSilence = parseInt(rawSilence, 10);
    const silence = isNaN(parsedSilence) ? 0 : Math.max(0, parsedSilence);
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
      // Cắt câu theo ĐỘ ỔN ĐỊNH (stable_cut): text đứng im đủ lâu ⇒ chốt câu.
      splitOnStability: cfg.splitOnStability === undefined ? true : !!cfg.splitOnStability,
      // Chẩn đoán mỗi nhịp ([SEG_TRACE]) — mặc định TẮT.
      traceStability: !!cfg.traceStability,
      ttsEnabled: !!cfg.ttsEnabled,
      ttsVoice: cfg.ttsVoice || "speaker_01_0039.wav",
      ttsSpeed: parseFloat(cfg.ttsSpeed || 1.0),
      ttsInstruct: cfg.ttsInstruct || "",
      ttsRefAudio: cfg.ttsRefAudio || "",
      ttsRefText: cfg.ttsRefText || "",
    };
    if (cfg.translationModel) out.translationModel = cfg.translationModel;
    if (cfg.vadEngine) out.vadEngine = cfg.vadEngine;
    // Pipeline B: offset canh đồng bộ phụ đề/TTS (ms, người dùng chỉnh trong popup).
    // Trần ±1500 ms khớp với backend (`LookaheadSessionState.apply_config`); ±600 ms cũ quá
    // hẹp cho trường hợp phụ đề sớm ~1 s (đo thật 2026-10-01).
    const syncMs = parseInt(cfg.lookaheadSyncOffsetMs, 10);
    if (!isNaN(syncMs)) out.lookaheadSyncOffsetMs = Math.max(-1500, Math.min(1500, syncMs));
    if (cfg.lookaheadLeadTimeSec !== undefined) {
      const lead = parseInt(cfg.lookaheadLeadTimeSec, 10);
      if (!isNaN(lead)) out.lookaheadLeadTimeSec = lead;
    }
    // stable_cut: nhận cả camelCase (getSettings) lẫn snake_case (settings cũ đã lưu).
    const stableMs = cfg.stability_duration_ms !== undefined
      ? cfg.stability_duration_ms
      : (cfg.stabilityDurationSec !== undefined ? parseFloat(cfg.stabilityDurationSec) * 1000 : undefined);
    if (stableMs !== undefined && !isNaN(parseFloat(stableMs))) {
      out.stabilityDurationSec = Math.max(0.0, parseFloat(stableMs) / 1000);
    }
    const stableMinSec = cfg.stability_min_duration_sec !== undefined
      ? cfg.stability_min_duration_sec
      : cfg.stabilityMinDurationSec;
    if (stableMinSec !== undefined && !isNaN(parseFloat(stableMinSec))) {
      out.stabilityMinDurationSec = Math.max(0, parseFloat(stableMinSec));
    }
    const stableMinWords = cfg.stability_min_words !== undefined ? cfg.stability_min_words : cfg.stabilityMinWords;
    if (stableMinWords !== undefined && !isNaN(parseInt(stableMinWords, 10))) {
      out.stabilityMinWords = Math.max(0, parseInt(stableMinWords, 10));
    }
    // Chẩn đoán `[SEG_TRACE]`: nhận cả camelCase lẫn snake_case (cấu hình cũ đã lưu).
    const trace = cfg.trace_stability !== undefined ? cfg.trace_stability : cfg.traceStability;
    if (trace !== undefined) out.traceStability = !!trace;
    return out;
  }

  const ttsPlayer = new TTSAudioPlayer();

  let wsClient = null, audioCapture = null, overlayManager = null;
  let captureAbortController = null;
  let isCapturing = false;
  let settings = { targetLang: "vi", subPosY: 10, subWidth: 80, showOriginalSubtitles: true };
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

  // ─────────────────────────────────────────────────────────────────────────────
  // CẢNH BÁO "VIDEO ĐANG TẮT TIẾNG ⇒ MẤT PHỤ ĐỀ"
  //
  // Extension lấy audio bằng `createMediaElementSource`. Theo spec Web Audio (và đúng như
  // Firefox đã chuẩn hoá — bug 2010427 + WPT "MediaElementAudioSourceNode output reflects
  // HTMLMediaElement effective volume change"), đầu ra của node đó **đã bị nhân bởi
  // `video.volume` và `muted`**.
  //
  // ĐÃ ĐO THẬT (external/build-tmp/volume_vs_asr.py + volume_vs_asr_1p7b.py + vad_vs_amplitude.py)
  // để biết mức nào mới thực sự nguy hiểm:
  //
  //   scale  0.50  0.30  0.10  0.05  0.01       0.00
  //   ASR    không đổi suốt tới -60 dBFS      -> 1.0000  (sai 100%)
  //   VAD    vẫn phát hiện (không hề về 0)    -> 0 đoạn   (mất hẳn)
  //
  // Nghĩa là: **giảm nhỏ volume KHÔNG làm hỏng nhận dạng** — nhân một tín hiệu với hằng số
  // không đổi SNR nội tại của nó, và cả VAD lẫn ASR đều bất biến biên độ trong dải rất rộng.
  // Chỉ có ĐÚNG GIÁ TRỊ 0 (mute, hoặc volume = 0) mới phá huỷ thông tin: `x * 0 = 0` là mất
  // vĩnh viễn, không cách nào khôi phục.
  //
  // Vì vậy cảnh báo dưới đây chỉ kêu khi tín hiệu bị nhân bởi 0 — không kêu khi người dùng
  // chỉ hạ nhỏ. Slider 🔉 Original audio trong popup không gây ra vấn đề này vì nó điều khiển
  // GainNode trên nhánh nghe, không đụng `video.volume`.
  // ─────────────────────────────────────────────────────────────────────────────
  let captureSilenceWarned = false;

  function checkCaptureAudibility(video) {
    if (!isCapturing || !video) return;
    // Chỉ 0 mới chết. Dùng ngưỡng rất nhỏ để không báo động nhầm khi người dùng hạ nhỏ.
    const silent = !!video.muted || video.volume <= 1e-4;
    if (silent && !captureSilenceWarned) {
      captureSilenceWarned = true;
      console.warn(
        "[BS] ⚠️ Trình phát đang TẮT HẲN TIẾNG (muted hoặc volume = 0). Extension lấy audio qua " +
        "createMediaElementSource nên tín hiệu gửi ASR bị nhân với 0 ⇒ im lặng kỹ thuật số ⇒ " +
        "SẼ KHÔNG CÓ PHỤ ĐỀ/DỊCH/TTS. Hãy bật tiếng lại cho trình phát. " +
        "(Hạ nhỏ volume thì KHÔNG sao: đã đo thấy VAD và ASR bất biến biên độ tới -60 dBFS. " +
        "Muốn hạ tiếng gốc mà không ảnh hưởng ASR thì dùng slider '🔉 Original audio' trong popup.)"
      );
    } else if (!silent && captureSilenceWarned) {
      captureSilenceWarned = false;
      console.log("[BS] Trình phát đã có tiếng trở lại — ASR nhận được audio.");
    }
  }

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

    // Khi Lookahead đang hiển thị phụ đề đã nạp sẵn, chặn partial transcript đè nhấp nháy
    if (eventType === "partial_transcript" && timelineQueue && timelineQueue.activeSubtitle) {
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

  // Hỏi injected script (buffer_interceptor_poc.js) trạng thái đệm audio tương lai.
  // Đây là nguồn dữ liệu cho dòng "Lookahead available +Xs" trong POPUP.
  function getLookaheadBufferStatus(timeoutMs = 400) {
    return new Promise((resolve) => {
      let done = false;
      const finish = (value) => {
        if (done) return;
        done = true;
        window.removeEventListener("message", handler);
        resolve(value);
      };
      const handler = (event) => {
        if (event.source !== window || !event.data) return;
        if (event.data.source === "VIBE_LOOKAHEAD_POC" && event.data.type === "BUFFER_STATUS_RESPONSE") {
          finish(event.data.payload || null);
        }
      };
      window.addEventListener("message", handler);
      window.postMessage({ source: "VIBE_LOOKAHEAD_CLIENT", type: "CHECK_BUFFER_STATUS" }, "*");
      setTimeout(() => finish(null), timeoutMs);
    });
  }

  // Kiểm tra xem trình phát video có nạp sẵn buffer âm thanh (MSE / VOD) không
  async function checkBufferAvailable(video) {
    if (!video) return false;
    // Nếu là livestream không xác định độ dài -> không thể dùng Lookahead
    if (!isFinite(video.duration) || video.duration <= 0) {
      return false;
    }

    // 1. Kiểm tra trực tiếp TimeRanges trong video.buffered
    let hasVideoBuffered = false;
    if (video.buffered && video.buffered.length > 0) {
      const cur = video.currentTime;
      let ahead = 0;
      for (let i = 0; i < video.buffered.length; i++) {
        if (cur >= video.buffered.start(i) - 0.5 && cur <= video.buffered.end(i) + 0.5) {
          ahead = video.buffered.end(i) - cur;
          break;
        }
      }
      if (!ahead && video.buffered.length > 0) {
        ahead = video.buffered.end(video.buffered.length - 1) - cur;
      }
      if (ahead >= 1.5) {
        hasVideoBuffered = true;
      }
    }

    // 2. Hỏi injected script (buffer_interceptor_poc.js) qua postMessage
    const p = await getLookaheadBufferStatus();
    if (!p) return false; // Interceptor không phản hồi -> không lấy được audio raw

    // ĐIỀU KIỆN BẮT BUỘC: interceptor phải THỰC SỰ bắt được byte audio (mảnh MSE hoặc
    // init segment). Nếu site phát bằng blob URL / XHR (không MSE) thì `video.buffered`
    // vẫn lớn nhưng ta KHÔNG có byte nào để giải mã ⇒ Pipeline B sẽ chạy mà không bao giờ
    // có phụ đề. Trường hợp đó phải quay về Pipeline A.
    return Boolean(hasVideoBuffered && (p.cachedChunksCount > 0 || p.hasInitSegment));
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
        // Bất đồng bộ: cần hỏi injected script về buffer Lookahead trước khi trả lời.
        getLookaheadBufferStatus().then((lookahead) => {
          sendResponse?.({
            isCapturing,
            hasVideo: !!findVideo(),
            overlayActive: !!overlayManager,
            sampleRate: audioCapture?.audioContext?.sampleRate || null,
            pipeline: activePipeline,
            lookahead,
          });
        }).catch(() => {
          sendResponse?.({ isCapturing, hasVideo: !!findVideo(), overlayActive: !!overlayManager, pipeline: activePipeline, lookahead: null });
        });
        return true;
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
        // Pipeline B: đẩy cấu hình mới sang /ws/lookahead (VAD, phân câu, model, TTS…).
        if (lookaheadClient && msg.settings) {
          try { lookaheadClient.updateConfig(buildWsConfig(settings)); } catch (e) {}
          console.log("[BS] Lookahead settings updated live:", msg.settings);
        }
        if (ttsTimeline && msg.settings) {
          ttsTimeline.setEnabled(!!settings.ttsEnabled);
        }
        sendResponse?.({ success: true, settings });
        break;
    }
  });

  // ── PHÁT HIỆN ĐỔI VIDEO TRÊN TRANG SPA (YouTube) ────────────────────────────
  // YouTube không tải lại trang khi người dùng bấm video khác; video cũ bị thay bằng video
  // mới nhưng phiên Lookahead/WS vẫn sống ⇒ backend tiếp tục nhận mảnh của video CŨ (sai
  // timeline) và phiên không bao giờ tự tắt (đo thật 2026-10-01: bấm video khác mà không
  // Stop thì lỗi session cũ còn treo).
  let _navWatchTimer = null;
  let _watchedUrl = "";
  let _watchedVideo = null;

  function stopNavigationWatch() {
    if (_navWatchTimer) {
      clearInterval(_navWatchTimer);
      _navWatchTimer = null;
    }
    _watchedUrl = "";
    _watchedVideo = null;
  }

  function startNavigationWatch(video) {
    stopNavigationWatch();
    _watchedUrl = String(location.href || "");
    _watchedVideo = video || null;
    _navWatchTimer = setInterval(() => {
      if (!isCapturing) {
        stopNavigationWatch();
        return;
      }
      const url = String(location.href || "");
      const current = findVideo();
      const urlChanged = url !== _watchedUrl;
      const videoReplaced = !!(_watchedVideo && current && current !== _watchedVideo);
      if (!urlChanged && !videoReplaced) {
        if (current) _watchedVideo = current;
        return;
      }
      // Với YouTube, chỉ đổi `t=`/`list=` trên CÙNG video thì KHÔNG phải video mới.
      if (urlChanged && !videoReplaced && sameVideoId(_watchedUrl, url)) {
        _watchedUrl = url;
        if (current) _watchedVideo = current;
        return;
      }
      console.warn(
        `[BS] 🔄 Phát hiện ĐỔI VIDEO (${urlChanged ? "URL đổi" : "thẻ <video> bị thay"}) — ` +
        `tự động dừng phiên cũ để tránh gửi audio sai timeline.`
      );
      stopNavigationWatch();
      try { cleanup(); } catch (e) {}
    }, 1000);
  }

  /** Hai URL YouTube có cùng video id? (đổi `t=`/`list=` không phải video mới). */
  function sameVideoId(a, b) {
    const id = (u) => {
      const m = /[?&]v=([^&#]+)/.exec(u);
      return m ? m[1] : u;
    };
    return id(a) === id(b);
  }

  async function startCapture(msg) {
    // Thế hệ phiên: mỗi lần Start tăng lên, mỗi lần Stop/cleanup cũng tăng. Nhờ vậy một
    // tiến trình Start đang `await` (chờ kiểm tra buffer / chờ backend Lookahead) KHÔNG thể
    // "hồi sinh" pipeline sau khi người dùng đã bấm Stop — trước đây nó vẫn gán
    // `isCapturing = true` và có thể để lại socket/đồ thị audio sống sót.
    const myGen = ++captureGeneration;

    // F-45: nếu cờ `isCapturing` còn kẹt từ phiên trước nhưng WS đã đóng thì dọn trước,
    // để bấm Start không bị "Already capturing" (trước đây phải tải lại trang).
    if (isCapturing && (!wsClient || !wsClient.isConnected) && (!lookaheadClient || !lookaheadClient.isConnected)) {
      console.warn("[BS] Trạng thái capture cũ đã kẹt — dọn dẹp rồi Start lại.");
      try {
        await cleanup();
      } catch (e) {}
      if (myGen !== captureGeneration) return { success: false, error: "Đã dừng" };
    }
    if (isCapturing) {
      return {
        success: false,
        error: "Already capturing",
        isCapturing: true,
        sampleRate: audioCapture?.audioContext?.sampleRate || null
      };
    }
    try {
      if (msg.settings) Object.assign(settings, msg.settings);
      let video = findVideo();
      if (!video) {
        // Retry for up to ~1s if video element is lazily loaded upon play
        for (let i = 0; i < 4; i++) {
          await new Promise(r => setTimeout(r, 250));
          if (myGen !== captureGeneration) return { success: false, error: "Đã dừng" };
          video = findVideo();
          if (video) break;
        }
      }
      if (!video) return { success: false, error: "No video found" };

      // Initialize AbortController for clean listener lifecycle management
      captureAbortController = new AbortController();
      const { signal } = captureAbortController;

      // Xác định chạy Pipeline A hay Pipeline B
      const lookaheadRequested = settings.lookaheadEnabled !== false;
      let canBuffer = false;
      if (lookaheadRequested) {
        canBuffer = await checkBufferAvailable(video);
      }
      if (myGen !== captureGeneration) {
        console.warn("[BS] Đã bấm Stop trong lúc khởi động — huỷ phiên vừa dựng.");
        await cleanup();
        return { success: false, error: "Đã dừng" };
      }

      let result = null;
      if (lookaheadRequested && canBuffer) {
        activePipeline = "B";
        result = await startPipelineB(video, settings, signal);
        if (result && result.fallbackToA) {
          console.warn("[BS] Pipeline B không khả dụng (" + (result.reason || "unknown") + ") -> chuyển sang Pipeline A (Realtime Streaming)");
          activePipeline = "A";
          result = await startPipelineA(video, settings, signal);
        }
      } else {
        activePipeline = "A";
        if (lookaheadRequested && !canBuffer) {
          console.warn("[BS] Không phát hiện buffer video nạp trước -> Tự động chuyển sang Pipeline A (Realtime Streaming)");
        }
        result = await startPipelineA(video, settings, signal);
      }

      if (myGen !== captureGeneration) {
        // Stop đã được bấm trong lúc pipeline đang dựng ⇒ dọn sạch, KHÔNG bật cờ capture.
        console.warn("[BS] Stop trong lúc khởi động — dọn pipeline vừa tạo.");
        await cleanup();
        return { success: false, error: "Đã dừng" };
      }

      isCapturing = true;
      // Theo dõi đổi video (SPA) để tự tắt phiên cũ thay vì gửi audio sai timeline.
      startNavigationWatch(video);
      console.log(`[BS] Capture started via Pipeline ${activePipeline}`);
      return result;
    } catch (e) {
      console.error("[BS] Start error:", e);
      await cleanup();
      const errText = e?.message || (typeof e === "string" ? e : "Không thể khởi động dịch video");
      return { success: false, error: errText };
    }
  }

  // ── PIPELINE A: Realtime Audio Streaming (ASR + VAD Realtime) ──────────────
  async function startPipelineA(video, settings, signal) {
    console.log("%c[BS] 🎙️ Khởi động Pipeline A: Realtime Streaming qua /ws", "color: #38bdf8; font-weight: bold;");

    ttsPlayer.setTargetVideo(
      video,
      settings.ttsDucking !== false,
      settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25,
      !!settings.ttsEnabled
    );
    ttsPlayer.clear();

    const handleSeek = (event) => {
      const reason = event && event.type === "seeking" ? "seeking" : "seek";
      try { ttsPlayer.clear(); } catch (e) {}
      if (overlayManager) {
        try { overlayManager.clear(); } catch (e) {}
      }
      if (audioCapture && typeof audioCapture.reset === "function") {
        try { audioCapture.reset(); } catch (e) {}
      }
      if (wsClient) {
        try { wsClient.sendJSON({ type: "reset_stream", reason }); } catch (e) {}
      }
    };
    video.addEventListener("seeking", handleSeek, { signal });
    video.addEventListener("seeked", handleSeek, { signal });

    wsClient = new WSClient("wss://localhost:8765/ws");
    wsClient.on("connected", () => {
      wsClient.sendJSON(buildWsConfig(settings));
    });

    wsClient.on("partial_transcript", p => emitSubtitleEvent("partial_transcript", p));
    wsClient.on("utterance_update", p => emitSubtitleEvent("utterance_update", p));
    wsClient.on("translation", p => emitSubtitleEvent("translation", p));
    wsClient.on("tts_audio", p => emitSubtitleEvent("tts_audio", p));
    wsClient.on("error", p => console.error("[BS] Backend:", p));

    wsClient.on("model_status", p => {
      const msg = `[BS] Model ${p?.stage || "?"}: ${p?.model || ""} -> ${p?.state || "?"}`;
      if (p?.state === "error") console.warn(msg, p?.message || "");
      else console.log(msg);
    });

    wsClient.on("tts_binary", (data) => {
      try {
        const buf = data instanceof ArrayBuffer ? data : (data && data.buffer) || null;
        if (!buf || buf.byteLength < 9) return;
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

    wsClient.on("disconnected", () => {
      if (!isCapturing) return;
      console.warn("[BS] Mất kết nối backend — dừng capture để có thể Start lại.");
      try { stopCapture(); } catch (e) {}
    });

    wsClient.on("backpressure", (bp) => {
      if (bp.state === "paused") {
        if (audioCapture && audioCapture.isCapturing) {
          audioCapture.isCapturing = false;
        }
        startBackpressureProbe();
      } else if (bp.state === "ok") {
        stopBackpressureProbe();
        if (audioCapture && !audioCapture.isCapturing) {
          audioCapture.isCapturing = true;
        }
      }
    });

    await wsClient.connect();

    audioCapture = new AudioCapture();
    audioCapture.onChunk = (pcmBuffer, timestamp, chunkIdx) => {
      if (wsClient && wsClient.isConnected) {
        wsClient.sendBinary(pcmBuffer, timestamp, chunkIdx);
      }
    };
    await audioCapture.start(video);
    ttsPlayer.setDuckSink(audioCapture);

    if (window === window.top || document.fullscreenElement) {
      ensureOverlay(video);
    }

    const handleStateKeepAlive = () => {
      if (audioCapture) audioCapture.resumeAudioContext();
      if (document.fullscreenElement && !overlayManager) ensureOverlay(video);
      try { ttsPlayer.reapplyDucking(); } catch (e) {}
      checkCaptureAudibility(video);
    };

    document.addEventListener("fullscreenchange", handleStateKeepAlive, { signal });
    document.addEventListener("webkitfullscreenchange", handleStateKeepAlive, { signal });
    document.addEventListener("mozfullscreenchange", handleStateKeepAlive, { signal });
    video.addEventListener("play", handleStateKeepAlive, { signal });
    video.addEventListener("playing", handleStateKeepAlive, { signal });
    video.addEventListener("volumechange", handleStateKeepAlive, { signal });
    checkCaptureAudibility(video);

    const actualRate = audioCapture?.audioContext?.sampleRate || null;
    return { success: true, sampleRate: actualRate, pipeline: "A" };
  }

  // ── PIPELINE B: Lookahead Video Buffering (0.0s Zero Perceived Latency) ────
  async function startPipelineB(video, settings, signal) {
    const leadTimeSec = parseInt(settings.lookaheadLeadTimeSec || 15, 10);
    console.log(`%c[BS] 🚀 Khởi động Pipeline B: Lookahead Video Buffering (dịch trước ${leadTimeSec}s, 0.0s Lag)`, "color: #00ffcc; font-weight: bold;");
    // Log THIẾT LẬP hiệu dụng: nếu TTS không kêu thì đây là chỗ đầu tiên cần xem.
    console.log("[BS] Lookahead settings:", {
      lookaheadEnabled: settings.lookaheadEnabled,
      leadTimeSec,
      ttsEnabled: !!settings.ttsEnabled,
      ttsDucking: settings.ttsDucking !== false,
      duckingLevel: settings.duckingLevel,
      vadEngine: settings.vadEngine,
      sourceLanguage: settings.sourceLanguage,
      targetLang: settings.targetLang,
    });

    const om = ensureOverlay(video);

    // 1. Kiểm tra xem video đã có sẵn lượng buffer phía trước chưa
    const curTime = video.currentTime;
    let aheadSec = 0;
    if (video.buffered && video.buffered.length > 0) {
      for (let i = 0; i < video.buffered.length; i++) {
        if (curTime >= video.buffered.start(i) - 0.5 && curTime <= video.buffered.end(i) + 0.5) {
          aheadSec = video.buffered.end(i) - curTime;
          break;
        }
      }
      if (!aheadSec && video.buffered.length > 0) {
        aheadSec = video.buffered.end(video.buffered.length - 1) - curTime;
      }
    }

    let pausedByLookahead = false;
    let currentBufferingWhy = "start";
    const showBufferingStatus = (msg) => {
      if (om && typeof om.showBuffering === "function") {
        om.showBuffering(msg || "Đang nạp đệm và dịch trước...");
      }
    };
    const hideBufferingStatus = () => {
      if (om && typeof om.hideBuffering === "function") {
        om.hideBuffering();
      }
    };

    // Tạm dừng video để nạp phụ đề trước (chỉ lúc bắt đầu hoặc khi tua video).
    let resumeTimer = null;
    let lastResumeAt = 0;
    let firstTtsReceived = false;
    //: Tiến triển nạp đệm gần nhất (để gia hạn chờ sau khi tua — xem `scheduleResume`).
    let bufferProgress = { ready: 0, fed: 0, at: 0, extensions: 0 };
    const pauseForBuffering = (why) => {
      firstTtsReceived = false;
      if (!video.paused) {
        try { video.pause(); } catch (e) {}
      }
      pausedByLookahead = true;
      lookaheadPausedVideo = true;
      currentBufferingWhy = why || "start";
      showBufferingStatus(
        why === "seek"
          ? "Đang dịch trước đoạn vừa tua..."
          : (why === "tts_enable" ? "Đang chuẩn bị lồng tiếng TTS..." : "Đang nạp đệm và dịch trước...")
      );
      // Chốt an toàn (đồng hồ ĐỒNG HỒ THỰC). Với TUA, mục tiêu là "dịch xong ngay tại vị trí
      // mới rồi mới phát" nên KHÔNG cắt cứng ở 3,5 s: nếu backend còn tiến triển thì gia hạn
      // thêm, chỉ bỏ cuộc khi hết tiến triển (xem nhánh gia hạn bên dưới).
      const timeoutMs = why === "seek"
        ? (settings.ttsEnabled ? 6000 : 5000)
        : (settings.ttsEnabled ? 5000 : 3500);
      bufferProgress = { ready: 0, fed: 0, at: Date.now(), extensions: 0 };
      if (resumeTimer) clearTimeout(resumeTimer);
      const scheduleResume = (ms) => {
        resumeTimer = setTimeout(() => {
          if (!lookaheadPausedVideo) return;
          const grew = bufferProgress.ready > 0
            && Date.now() - bufferProgress.at < 1200
            && bufferProgress.extensions < 3;
          if (why === "seek" && grew) {
            bufferProgress.extensions += 1;
            console.log(
              `[BS] Chờ thêm bản dịch tại vị trí vừa tua (ready ${bufferProgress.ready.toFixed(1)}s, ` +
              `gia hạn ${bufferProgress.extensions}/3).`
            );
            scheduleResume(2000);
            return;
          }
          console.log("[BS] Đã nạp đệm ban đầu xong -> tiếp tục phát video.");
          resumePlayback("timeout");
        }, ms);
      };
      scheduleResume(timeoutMs);
    };
    lookaheadPauseForBuffering = pauseForBuffering;

    // Nếu video đã có buffer phía trước thì tạm dừng để xử lý dịch trước.
    if (aheadSec >= 1.0) pauseForBuffering("start");

    // 2. Khởi tạo SubtitleTimelineQueue
    timelineQueue = new SubtitleTimelineQueue({
      onSubtitleChange: (sub) => {
        if (sub) {
          emitSubtitleEvent("translation", {
            utterance_id: sub.id,
            original_text: sub.original_text,
            translated_text: sub.translated_text,
            translated: sub.translated_text,
            status: "ok",
            is_final: true,
          });
        } else {
          if (overlayManager) {
            try { overlayManager.clear(); } catch (e) {}
          }
        }
      },
      onSeekTriggered: (seekId, targetTime) => {
        // Lồng tiếng cũ thuộc đoạn trước khi tua ⇒ bỏ hết.
        if (ttsTimeline) {
          try { ttsTimeline.clear(); } catch (e) {}
        }
        if (lookaheadClient) {
          lookaheadClient.notifySeek(seekId, targetTime);
        }
        // Vị trí mới chưa có phụ đề: TẠM DỪNG video chờ backend dịch trước, rồi tự phát lại
        // khi `lookahead_status.prebuffer_ready` báo sẵn sàng.
        pauseForBuffering("seek");
      },
    });
    timelineQueue.attachVideo(video);

    // 2b. Lồng tiếng theo timeline: audio TTS được phát ĐÚNG lúc video.currentTime đi qua
    // `start_pts` (xem `lib/tts-timeline.js`), có nén/cắt để không trễ dây chuyền.
    ttsPlayer.setTargetVideo(
      video,
      settings.ttsDucking !== false,
      settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25,
      !!settings.ttsEnabled
    );
    ttsPlayer.clear();
    ttsTimeline = new TtsTimelineScheduler({
      enabled: !!settings.ttsEnabled,
      getAudioContext: () => ttsPlayer._getAudioContext(),
      allowPitchShift: false, // Chống méo tiếng: AudioBufferSourceNode không giữ cao độ, backend đã nén bằng WSOLA
      tailAllowanceSec: 2.0,  // Cho phép câu đọc kết thúc trọn vẹn vào khoảng lặng kế tiếp, không bị cụt từ cuối
    });
    ttsTimeline.attach(video);

    // 3. Khởi tạo LookaheadClient kết nối tới /ws/lookahead
    // `resumePlayback` KHÔNG còn là one-shot: nó được gọi lại sau mỗi lần tua.
    var resumePlayback = (reason) => {
      if (!lookaheadPausedVideo) return;
      lookaheadPausedVideo = false;
      lastResumeAt = Date.now();
      underrunStreak = 0;
      if (resumeTimer) {
        clearTimeout(resumeTimer);
        resumeTimer = null;
      }
      console.log(`%c[BS] ✅ Phát video với phụ đề Lookahead (${reason || "ready"})`, "color: #34c759; font-weight: bold;");
      hideBufferingStatus();
      try { om.clear(); } catch (e) {}
      try { video.play(); } catch (e) {}
      // `om.clear()` vừa xoá cả câu đang đúng ⇒ buộc timeline vẽ lại ngay, nếu không phải
      // đợi sang câu kế tiếp mới thấy phụ đề (lỗi "câu đầu không hiển thị").
      if (timelineQueue) {
        try { timelineQueue.refreshNow(); } catch (e) {}
      }
    };

    let settled = false;
    const availability = await new Promise((resolve) => {
      const finish = (value) => { if (!settled) { settled = true; resolve(value); } };

      lookaheadClient = new LookaheadClient({
        serverUrl: "wss://localhost:8765/ws/lookahead",
        timelineQueue: timelineQueue,
        leadTime: leadTimeSec,
        sourceLang: settings.sourceLanguage || "auto",
        targetLang: settings.targetLang || "vi",
        // Cấu hình ĐẦY ĐỦ từ popup (VAD engine/threshold/silence, model ASR, phân câu,
        // model dịch…) — Pipeline B phải chạy đúng thông số người dùng đã chọn.
        config: buildWsConfig(settings),
        onPrebufferReady: (st) => {
          // Nếu bật TTS nhưng chưa nhận câu TTS nào và đang có câu nói: chờ onTts kích hoạt
          if (settings.ttsEnabled && !firstTtsReceived && Number(st?.ready_ahead || 0) > 0) {
            return;
          }
          resumePlayback(st && st.reason ? st.reason : "ready");
        },
        onTts: (header, wavBuffer) => {
          if (!settings.ttsEnabled) return;
          if (timelineQueue && header && header.start_pts && header.duration_sec) {
            // Bước 1 & 2: Đồng bộ phụ đề giữ trên màn hình cho tới khi TTS đọc xong
            timelineQueue.extendSubtitleEnd(
              Number(header.start_pts),
              Number(header.start_pts) + Number(header.duration_sec) + 0.10
            );
          }
          if (ttsTimeline) ttsTimeline.enqueue(header, wavBuffer);

          // Nếu đang tạm dừng nạp đệm và bật TTS: nhận được câu TTS đầu tiên ⇒ phát video ngay!
          if (settings.ttsEnabled && lookaheadPausedVideo && !firstTtsReceived) {
            firstTtsReceived = true;
            resumePlayback("tts_first_ready");
          }
        },
        onReady: () => finish(null),
        onUnavailable: (reason) => {
          console.warn("[BS] LookaheadClient báo không khả dụng:", reason);
          finish(reason || "unavailable");
        },
        onError: (err) => {
          console.error("[BS] LookaheadClient báo lỗi kết nối:", err);
          finish(typeof err === "string" ? err : (err?.message || "connection_error"));
        },
        onStatus: (st) => {
          const ready = Number(st.ready_ahead || 0);
          const fed = Number(st.fed_ahead || 0);
          const isTts = Boolean(settings.ttsEnabled);
          // Ghi nhận tiến triển để quyết định có gia hạn chờ sau khi tua hay không.
          if (lookaheadPausedVideo && (ready > bufferProgress.ready + 0.05 || fed > bufferProgress.fed + 0.05)) {
            bufferProgress = { ...bufferProgress, ready, fed, at: Date.now() };
          }

          if (lookaheadPausedVideo) {
            const target = Number(st.target_ahead || 0);
            if (ready > 0 && target > 0) {
              const prefix = currentBufferingWhy === "seek" ? "Đang dịch trước đoạn vừa tua" : "Đang nạp đệm và dịch trước";
              const extra = isTts && !firstTtsReceived ? " (chờ lồng tiếng)..." : "...";
              showBufferingStatus(`${prefix} (${ready.toFixed(1)}s / ${target.toFixed(1)}s)${extra}`);
            }

            // Sẵn sàng phát lại khi:
            // 1. Backend báo st.prebuffer_ready (đã tính toán cả TTS nếu tts_enabled)
            // 2. HOẶC nếu TTS BẬT:
            //    - Đã nhận câu TTS đầu tiên (firstTtsReceived) VÀ ready >= 1.0s
            //    - HOẶC đoạn trước là khoảng lặng (fed >= 4.0s và ready === 0: không có tiếng nói để đọc TTS)
            // 3. HOẶC nếu TTS TẮT:
            //    - ready >= 1.5s
            //    - HOẶC fed >= 4.0s && ready === 0 (quét qua khoảng lặng)
            let readyToResume = false;
            if (st.prebuffer_ready) {
              if (!isTts || firstTtsReceived || ready === 0) {
                readyToResume = true;
              }
            } else if (isTts) {
              if (firstTtsReceived && ready >= 1.0) {
                readyToResume = true;
              } else if (fed >= 4.0 && ready === 0) {
                readyToResume = true;
              }
            } else {
              if (ready >= 1.5) {
                readyToResume = true;
              } else if (fed >= 4.0 && ready === 0) {
                readyToResume = true;
              }
            }

            if (readyToResume) {
              resumePlayback(st.prebuffer_ready ? "prebuffer_ready" : "status_ready");
            }
          }
          // ĐANG PHÁT: Tuyệt đối KHÔNG tạm dừng video giữa chừng ("tạm dừng để đuổi kịp").
          // Người dùng cần trải nghiệm phát mượt mà, phụ đề & TTS cập nhật mốc thời gian khi có.
        },
      });
      lookaheadClient.connect(video);

      // Nếu sau 3.5s chưa kết nối được socket -> coi như Pipeline B thất bại, tự động chuyển về Pipeline A!
      setTimeout(() => {
        if (!settled) {
          if (!lookaheadClient || !lookaheadClient.isConnected) {
            console.warn("[BS] Quá thời gian chờ kết nối /ws/lookahead (3.5s) -> tự động chuyển về Pipeline A.");
            finish("connection_timeout");
          } else {
            // Đã kết nối socket nhưng backend chưa phản hồi lookahead_ready (backend cũ/chậm): vẫn tiếp tục
            finish(null);
          }
        }
      }, 3500);
    });

    if (availability) {
      // Backend báo KHÔNG chạy được Pipeline B -> gỡ sạch để content script quay về Pipeline A.
      console.warn(`%c[BS] ⚠️ Pipeline B không khả dụng (${availability}) -> Tự động chuyển sang Pipeline A (Realtime Streaming)`, "color: #f59e0b; font-weight: bold;");
      try { lookaheadClient.disconnect(); } catch (e) {}
      lookaheadClient = null;
      try { timelineQueue.detach(); } catch (e) {}
      timelineQueue = null;
      if (ttsTimeline) {
        try { ttsTimeline.destroy(); } catch (e) {}
        ttsTimeline = null;
      }
      hideBufferingStatus();
      try { om.clear(); } catch (e) {}
      // Trả lại trạng thái phát cho video (ta đã pause nó để nạp đệm).
      lookaheadPausedVideo = false;
      if (resumeTimer) { clearTimeout(resumeTimer); resumeTimer = null; }
      if (pausedByLookahead) {
        try { video.play(); } catch (e) {}
      }
      return { success: false, fallbackToA: true, reason: availability, pipeline: "B" };
    }

    const handleStateKeepAlive = () => {
      if (document.fullscreenElement && !overlayManager) ensureOverlay(video);
    };
    document.addEventListener("fullscreenchange", handleStateKeepAlive, { signal });
    document.addEventListener("webkitfullscreenchange", handleStateKeepAlive, { signal });
    document.addEventListener("mozfullscreenchange", handleStateKeepAlive, { signal });

    return { success: true, sampleRate: null, pipeline: "B" };
  }

  function stopCapture() { cleanup(); return { success: true }; }

  async function cleanup() {
    isCapturing = false;
    stopNavigationWatch();
    // Vô hiệu hoá mọi tiến trình Start đang `await` (xem `startCapture`).
    captureGeneration++;
    stopBackpressureProbe();   // FIX-01: không để timer probe sống sót qua Stop
    captureSilenceWarned = false;   // phiên sau phải được cảnh báo lại từ đầu
    // Trả lại trạng thái phát nếu chính Pipeline B đã tạm dừng video để nạp đệm.
    lookaheadPauseForBuffering = null;
    if (lookaheadPausedVideo) {
      lookaheadPausedVideo = false;
      try {
        const v = getVideo();
        if (v && v.paused) v.play();
      } catch (e) {}
    }
    if (overlayManager && typeof overlayManager.hideBuffering === "function") {
      overlayManager.hideBuffering();
    }
    if (captureAbortController) {
      try { captureAbortController.abort(); } catch (e) {}
      captureAbortController = null;
    }
    if (lookaheadClient) {
      try { lookaheadClient.disconnect(); } catch (e) {}
      lookaheadClient = null;
    }
    if (timelineQueue) {
      try { timelineQueue.detach(); } catch (e) {}
      timelineQueue = null;
    }
    if (ttsTimeline) {
      try {
        const snap = ttsTimeline.snapshot();
        if (snap.received > 0) {
          console.log(
            `[BS TTS-LA] Tổng kết: phát ${snap.played}, bỏ muộn ${snap.skippedLate}, ` +
            `cụt đuôi ${snap.cutShort}, trễ nhất ${snap.maxLateMs.toFixed(0)}ms, rate tối đa ${snap.maxRate.toFixed(2)}x`
          );
        }
        ttsTimeline.destroy();
      } catch (e) {}
      ttsTimeline = null;
    }
    ttsPlayer.destroy();
    // Ngắt tham chiếu tới đồ thị audio sắp bị dỡ, để ducking không gọi vào sink đã chết.
    try { ttsPlayer.setDuckSink(null); } catch (e) {}
    if (audioCapture) {
      try { audioCapture.stop(); } catch (e) {}
      audioCapture = null;
    }
    if (wsClient) {
      try { wsClient.disconnect(); } catch (e) {}
      wsClient = null;
    }
    if (overlayManager) {
      try { overlayManager.clear(); } catch (e) {}
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
  window.__bsGetStatus = async () => ({
    isCapturing,
    hasVideo: !!findVideo(),
    overlayActive: !!overlayManager,
    sampleRate: audioCapture?.audioContext?.sampleRate || null,
    pipeline: activePipeline,
    lookahead: await getLookaheadBufferStatus()
  });
  window.__bsUpdateSettings = (s) => {
    if (s) {
      const prevTtsEnabled = !!settings.ttsEnabled;
      Object.assign(settings, s);
      const newTtsEnabled = !!settings.ttsEnabled;

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

      // Xử lý bật/tắt TTS động trong Pipeline B (Lookahead)
      if (activePipeline === "B") {
        if (prevTtsEnabled && !newTtsEnabled) {
          // Người dùng vừa TẮT TTS: chặn phát mọi câu TTS đang dở hoặc đã lên lịch
          if (ttsTimeline) {
            ttsTimeline.setEnabled(false);
          }
          if (ttsPlayer) {
            ttsPlayer.clear();
          }
          console.log("[BS] Đã tắt TTS trong Pipeline B — dừng mọi audio lồng tiếng.");
        } else if (!prevTtsEnabled && newTtsEnabled) {
          // Người dùng vừa BẬT LẠI TTS: kích hoạt lại timeline và tạm dừng video để nạp đệm TTS
          if (ttsTimeline) {
            ttsTimeline.setEnabled(true);
          }
          if (typeof lookaheadPauseForBuffering === "function") {
            lookaheadPauseForBuffering("tts_enable");
          }
          console.log("[BS] Đã bật lại TTS trong Pipeline B — tạm dừng để tổng hợp câu TTS đầu.");
        }
        if (lookaheadClient && lookaheadClient.isConnected) {
          lookaheadClient.updateConfig(buildWsConfig(settings));
        }
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
