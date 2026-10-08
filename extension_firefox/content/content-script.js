// Content Script - Full Pipeline: audio capture + WebSocket + overlay
(function () {
  if (window.__bsContentScriptLoaded) return;
  window.__bsContentScriptLoaded = true;
  const api = typeof browser !== "undefined" ? browser : chrome;

  // ── Unified WebSocket Configuration Builder ─────────────────────────────
  // G9/G10: KHÔNG gửi giá trị mặc định cứng cho `translationModel` / `vadEngine` nữa.
  // Trước đây client luôn gửi "xiaomi" và một tên VAD engine cứng:
  //   - "xiaomi" (MiLMMT) không có file GGUF cục bộ -> khi backend xử lý thật sẽ lỗi.
  //   - tên VAD engine cứng khác mặc định của backend -> mỗi lần content
  //     script khởi động lại ghi đè VAD engine và kích hoạt đường nạp model trên hot path.
  // Nay chỉ gửi khi người dùng THỰC SỰ chọn; bỏ trống thì backend giữ mặc định của nó.
  // (Backend chỉ còn 2 engine, đều chạy onnxruntime: `firered-vad` mặc định và `silero-vad`.
  //  Engine `fsmn-vad` đã bị xoá ở Giai đoạn 3 cùng PyTorch.)
  // F-30: extension này khai báo giao thức v3 ⇒ backend gửi payload GỌN (một tên cho mỗi
  // giá trị, không còn `original`/`ui_text`/`utteranceId`/`stableText`…).
  const PROTOCOL_VERSION = 3;

  // ── TỰ ĐÓNG PHIÊN KHI VIDEO SẮP/HẾT ────────────────────────────────────────
  //: Còn ≤ ngần này giây là coi như hết video ⇒ đóng phiên (không dịch 3 giây cuối, đổi lại
  //: không bao giờ để phiên "mồ côi" + overlay treo trên màn hình).
  const LA_END_STOP_MARGIN_SEC = 3.0;
  //: Clip NGẮN hơn ngần này giây thì bỏ qua luật trên (tránh tắt phiên ngay khi vừa bấm Bắt đầu).
  const LA_END_STOP_MIN_DURATION_SEC = 15.0;

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

  // ── CHẨN ĐOÁN PIPELINE B (v0.6.7) ──────────────────────────────────────────
  // Người dùng báo (2026-10-05): có trang POPUP báo "Lookahead available" mà Pipeline B vẫn
  // không chạy (xvideos.com), có trang báo "+5.95s (đang nạp)" trong khi vùng đệm thật đã ~60s
  // (iq.com), có trang báo "không khả dụng (chạy Realtime)" (xhamster.com, av01.media) — và
  // KHÔNG có cách nào biết vì sao. Khối này in ra Console của trang:
  //   • [BS][Diag][gate]     — cổng quyết định Pipeline B: từng điều kiện, giá trị, kết luận.
  //   • [BS][Diag][session]  — vòng pause/resume khi nạp đệm, kèm số liệu backend.
  //   • [BS][Diag][status]   — `lookahead_status` rút gọn (ready/fed/ready_until/fragments).
  //   • [BS][Diag][stall]    — BẾ TẮC: backend không tiến triển trong lúc video bị tạm dừng.
  // Toàn bộ mã lý do nằm trong `lib/lookahead-diagnostics.js` (một nguồn sự thật duy nhất).
  const LA_DIAG = (typeof window !== "undefined" && window.LookaheadDiagnostics) || null;
  const laDiag = LA_DIAG
    ? new LA_DIAG.LookaheadDiag("BS")
    : {
        log() {}, ok() {}, warn() {}, error() {}, group() {},
        throttled() { return false; },
        setEnabled() { return this; },
      };
  if (!LA_DIAG) {
    console.warn("[BS][Diag] Không nạp được lib/lookahead-diagnostics.js — log chẩn đoán sẽ ở mức tối thiểu.");
  }
  //: Bật/tắt log chẩn đoán từ Console: `window.__bsLookaheadDiag(false)`.
  window.__bsLookaheadDiag = (on) => { laDiag.setEnabled(on !== false); return "ok"; };

  /** Thông tin thẻ <video> mà content script đang dùng (để đối chiếu với interceptor). */
  function laVideoInfo(video) {
    if (!video) return { hasVideo: false, durationFinite: false, bufferedAheadVideo: 0 };
    let ahead = 0;
    try {
      if (video.buffered && video.buffered.length > 0) {
        const cur = video.currentTime;
        for (let i = 0; i < video.buffered.length; i++) {
          if (cur >= video.buffered.start(i) - 0.5 && cur <= video.buffered.end(i) + 0.5) {
            ahead = video.buffered.end(i) - cur;
            break;
          }
        }
        if (!ahead) ahead = video.buffered.end(video.buffered.length - 1) - cur;
      }
    } catch (e) {}
    return {
      hasVideo: true,
      durationFinite: isFinite(video.duration) && video.duration > 0,
      duration: isFinite(video.duration) ? Number(video.duration.toFixed(2)) : String(video.duration),
      bufferedAheadVideo: Number(Math.max(0, ahead).toFixed(2)),
      currentTime: Number((Number(video.currentTime) || 0).toFixed(2)),
      paused: !!video.paused,
      srcSample: String(video.currentSrc || video.src || "").split("?")[0].slice(0, 140),
    };
  }
  //: Pipeline B tự tạm dừng video để nạp đệm trước ⇒ phải nhớ để TRẢ LẠI trạng thái phát
  //: khi Stop hoặc khi fallback sang Pipeline A (nếu không, video kẹt ở trạng thái pause).
  let lookaheadPausedVideo = false;
  let lookaheadPauseForBuffering = null;
  //: Bộ đếm/vòng theo dõi chẩn đoán của phiên Pipeline B đang chạy (xem `startPipelineB`).
  let lookaheadSessionDiag = null;
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
    // ⚠️ KHÔNG còn gửi `lookaheadLeadTimeSec`: ở tuyến OFFLINE_BATCH khoảng dịch trước do bộ cắt
    // khối quyết định (12–30 s), nên popup đã bỏ slider này. Backend dùng mặc định
    // `lookahead.lead_time_sec` (15 s) làm cổng throttle nội bộ.
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
  //
  // `video`: gửi kèm `currentSrc`/`currentTime` để interceptor ĐỐI CHIẾU thẻ video nó đang theo
  // dõi với thẻ mà content script/popup đang xem (phát hiện LỆCH THẺ VIDEO — nguyên nhân đã đo
  // trên iq.com: đệm thật ~60s mà interceptor báo +5.95s).
  function getLookaheadBufferStatus(timeoutMs = 400, video = null) {
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
      window.postMessage({
        source: "VIBE_LOOKAHEAD_CLIENT",
        type: "CHECK_BUFFER_STATUS",
        currentSrc: video ? String(video.currentSrc || video.src || "") : "",
        currentTime: video ? Number((Number(video.currentTime) || 0).toFixed(2)) : null,
      }, "*");
      setTimeout(() => finish(null), timeoutMs);
    });
  }

  /** Neo thẻ video ưu tiên cho interceptor (chống lệch thẻ trên trang nhiều <video>). */
  function announcePreferredVideo(video) {
    if (!video) return;
    try {
      const currentSrc = String(video.currentSrc || video.src || "");
      if (!currentSrc) return;
      window.postMessage({
        source: "VIBE_LOOKAHEAD_CLIENT",
        type: "SET_PREFERRED_VIDEO",
        currentSrc,
        currentTime: Number((Number(video.currentTime) || 0).toFixed(2)),
      }, "*");
    } catch (e) {}
  }

  /** Gộp dữ kiện interceptor + số đo của content script thành một bộ `facts` thống nhất. */
  function buildLookaheadFacts(payload, video) {
    const info = laVideoInfo(video);
    const d = (payload && payload.diag) || {};
    return Object.assign({}, d, {
      hasVideo: info.hasVideo,
      durationFinite: info.durationFinite,
      duration: info.duration,
      bufferedAheadVideo: info.bufferedAheadVideo,
      cachedChunksCount: payload ? payload.cachedChunksCount : 0,
      hasInitSegment: payload ? payload.hasInitSegment : false,
      mediaTaggedChunks: payload ? payload.mediaTaggedChunks : 0,
      interceptorResponded: !!payload,
    });
  }

  /** In BÁO CÁO chẩn đoán đầy đủ (dùng cho popup/console/`__bsLookaheadDiagReport()`). */
  function printLookaheadDiagnosis(facts, verdict) {
    if (!LA_DIAG) return;
    const v = verdict || LA_DIAG.classifyLookaheadAvailability(facts);
    laDiag.group(`🔬 VÌ SAO PIPELINE B KHÔNG CHẠY? → ${v.code}`,
      LA_DIAG.buildDiagnosticRows(facts, v),
      v.code === "OK" ? "ok" : "warn");
    return v;
  }

  // Kiểm tra xem trình phát video có nạp sẵn buffer âm thanh (MSE / VOD) không
  async function checkBufferAvailable(video) {
    if (!video) {
      laDiag.warn("gate", "❌ Không có thẻ <video> ⇒ không thể chạy Pipeline B (Pipeline A cũng sẽ không có nguồn audio).");
      return false;
    }
    //: Neo thẻ video trước khi hỏi trạng thái — nếu trang có nhiều <video>, interceptor phải nói
    //: về CÙNG một thẻ với content script (xem `announcePreferredVideo`).
    announcePreferredVideo(video);

    // Nếu là livestream không xác định độ dài -> không thể dùng Lookahead
    if (!isFinite(video.duration) || video.duration <= 0) {
      laDiag.warn("gate", `❌ Không chạy Pipeline B: livestream hoặc video chưa có độ dài `
        + `(duration=${video.duration}).`);
      printLookaheadDiagnosis(buildLookaheadFacts(null, video));
      return false;
    }

    // 1. Kiểm tra trực tiếp TimeRanges trong video.buffered
    const info = laVideoInfo(video);
    const ahead = info.bufferedAheadVideo;
    const hasVideoBuffered = ahead >= 1.5;

    // 2. Hỏi injected script (buffer_interceptor_poc.js) qua postMessage
    const p = await getLookaheadBufferStatus(400, video);
    const facts = buildLookaheadFacts(p, video);
    const verdict = LA_DIAG
      ? LA_DIAG.classifyLookaheadAvailability(facts)
      : { code: p ? "OK" : "UNKNOWN", detail: "" };

    // ĐIỀU KIỆN BẮT BUỘC: interceptor phải THỰC SỰ bắt được byte audio (mảnh MSE hoặc
    // init segment). Nếu site phát bằng blob URL / XHR (không MSE) thì `video.buffered`
    // vẫn lớn nhưng ta KHÔNG có byte nào để giải mã ⇒ Pipeline B sẽ chạy mà không bao giờ
    // có phụ thu; trường hợp đó phải quay về Pipeline A.
    const hasBytes = !!(p && (p.cachedChunksCount > 0 || p.hasInitSegment));
    const pass = Boolean(hasVideoBuffered && hasBytes);

    // ── LOG CỔNG QUYẾT ĐỊNH ──────────────────────────────────────────────────
    // Đây là dòng log ĐẦU TIÊN cần đọc khi một trang "không chạy được Pipeline B".
    laDiag.log("gate",
      `Cổng Pipeline B: ${pass ? "ĐẠT" : "KHÔNG ĐẠT"} | `
      + `video=${info.hasVideo} duration=${info.duration} đệm-trước=${ahead.toFixed(2)}s `
      + `interceptor=${p ? "có phản hồi" : "KHÔNG phản hồi"} `
      + `mảnh=${p ? p.cachedChunksCount : "-"} init=${p ? (p.hasInitSegment ? "có" : "chưa") : "-"} `
      + `| kết luận=${verdict.code} (${verdict.detail})`,
      p ? p.diag : null);

    if (!pass) {
      if (!hasVideoBuffered) {
        laDiag.warn("gate", `❌ Không chạy Pipeline B: vùng đệm trước playhead chỉ ${ahead.toFixed(2)}s `
          + `(< 1.5s) ⇒ không có "tương lai" để dịch trước.`);
      } else if (!p) {
        laDiag.warn("gate", "❌ Không chạy Pipeline B: interceptor KHÔNG phản hồi `BUFFER_STATUS_RESPONSE` "
          + "⇒ không có byte audio thô. Kiểm tra log `[Lookahead] 🚀 Multi-Layer Interceptor` ở đầu Console.");
      } else if (!hasBytes) {
        laDiag.warn("gate", `❌ Không chạy Pipeline B: interceptor chưa bắt được byte audio nào `
          + `(mảnh=${p.cachedChunksCount}, init=${p.hasInitSegment}) ⇒ sẽ quay về Pipeline A.`);
      }
      printLookaheadDiagnosis(facts, verdict);
    } else {
      laDiag.ok("gate", `✅ Đủ điều kiện chạy Pipeline B (audio-only/MSE đã bắt được).`);
      if (verdict.code !== "OK") {
        // Đạt cổng nhưng dữ kiện vẫn có điểm bất thường (ví dụ: buffer không rõ mime, hoặc lệch
        // thẻ video) — in báo cáo để không bỏ sót.
        printLookaheadDiagnosis(facts, verdict);
      }
    }
    return pass;
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
        // Trả kèm `lookaheadVerdict` (mã lý do + chi tiết) để POPUP hiển thị và in log.
        {
          const statusVideo = findVideo();
          getLookaheadBufferStatus(400, statusVideo).then((lookahead) => {
            const facts = buildLookaheadFacts(lookahead, statusVideo);
            const verdict = LA_DIAG ? LA_DIAG.classifyLookaheadAvailability(facts) : null;
            // ĐỆM VIDEO (`la.aheadSeconds`) và ĐỆM ĐÃ DỊCH (`ready_ahead`) là HAI đại lượng khác
            // nhau: POPUP báo "Lookahead available +60s" (video) nhưng pipeline vẫn có thể tạm dừng
            // vì phần ĐÃ DỊCH chỉ vài giây. Gửi kèm số liệu phiên để POPUP nói rõ cả hai.
            const laSt = (lookaheadSessionDiag && lookaheadSessionDiag.lastStatus) || null;
            sendResponse?.({
              isCapturing,
              hasVideo: !!statusVideo,
              overlayActive: !!overlayManager,
              sampleRate: audioCapture?.audioContext?.sampleRate || null,
              pipeline: activePipeline,
              lookahead,
              lookaheadFacts: facts,
              lookaheadVerdict: verdict,
              lookaheadSession: laSt ? {
                readyAhead: Number(laSt.ready_ahead || 0),
                readyUntilPts: Number(laSt.ready_until_pts || 0),
                fedAhead: Number(laSt.fed_ahead || 0),
                paused: !!lookaheadPausedVideo,
                pauses: lookaheadSessionDiag.pauses,
                resumes: lookaheadSessionDiag.resumes,
                minResumeAheadSec: 4.0,
              } : null,
            });
          }).catch(() => {
            sendResponse?.({ isCapturing, hasVideo: !!findVideo(), overlayActive: !!overlayManager, pipeline: activePipeline, lookahead: null, lookaheadVerdict: null });
          });
          return true;
        }
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
  // Theo dõi 2 dấu hiệu, cả hai đều rẻ:
  //   1. URL đổi (bao gồm `?v=<id>` của YouTube) — SPA dùng history API nên `popstate`
  //      không đủ, phải so chuỗi URL theo nhịp.
  //   2. Phần tử <video> bị THAY THẾ (identity khác) — có site đổi nguồn mà không đổi URL.
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

      // ── VIDEO SẮP HẾT / ĐÃ HẾT ⇒ TỰ ĐÓNG PHIÊN ────────────────────────────────
      // Sự cố thật 2026-10-05 (ảnh người dùng): video chạy hết nhưng phiên vẫn sống — overlay
      // "Đang xử lý tiếp đoạn video…" nằm lại giữa màn hình, backend vẫn giữ RAM/GPU, POPUP vẫn
      // ở trạng thái "đang chạy". Phiên chỉ nên sống khi còn nội dung để dịch.
      //
      // Chỉ dựa vào `ended` là KHÔNG đủ: nếu extension đang tạm dừng video sát mép cuối (hoặc
      // trình phát không bắn `ended` vì bị can thiệp) thì phiên "mồ côi" mãi. Vì vậy dùng luôn
      // `timeupdate`: còn ≤ `LA_END_STOP_MARGIN_SEC` giây là đóng phiên.
      const stopAtVideoEnd = (reason) => {
        if (!isCapturing) return;
        laDiag.log("session",
          `⏹️ Video ${reason} — tự dừng phiên dịch và gỡ overlay (không để phiên 'mồ côi').`);
        try { stopCapture(); } catch (e) {}
        if (overlayManager) {
          try { overlayManager.destroy(); } catch (e) {}
          overlayManager = null;
        }
      };
      const maybeStopNearEnd = () => {
        if (!isCapturing) return;
        const d = Number(video.duration);
        const t = Number(video.currentTime);
        // Bỏ qua livestream / video chưa biết độ dài, và clip quá ngắn (tránh tắt ngay khi vừa mở).
        if (!Number.isFinite(d) || d < LA_END_STOP_MIN_DURATION_SEC) return;
        if (!Number.isFinite(t) || t <= 0) return;
        if (t >= d - LA_END_STOP_MARGIN_SEC) {
          stopAtVideoEnd(`sắp hết (còn ${Math.max(0, d - t).toFixed(1)}s / ${d.toFixed(1)}s)`);
        }
      };
      video.addEventListener("timeupdate", maybeStopNearEnd, { signal });
      video.addEventListener("ended", () => stopAtVideoEnd("đã phát hết"), { signal });

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
        laDiag.log("pipeline", "▶️ Chạy Pipeline B (Lookahead) — cổng kiểm tra đã ĐẠT.");
        result = await startPipelineB(video, settings, signal);
        if (result && result.fallbackToA) {
          console.warn("[BS] Pipeline B không khả dụng (" + (result.reason || "unknown") + ") -> chuyển sang Pipeline A (Realtime Streaming)");
          activePipeline = "A";
          result = await startPipelineA(video, settings, signal);
        }
      } else {
        activePipeline = "A";
        if (lookaheadRequested && !canBuffer) {
          laDiag.warn("pipeline", "▶️ Chạy Pipeline A (Realtime) — cổng Pipeline B KHÔNG ĐẠT "
            + "(xem báo cáo chẩn đoán ngay trên dòng này).");
        } else if (!lookaheadRequested) {
          laDiag.log("pipeline", "▶️ Chạy Pipeline A (Realtime) — người dùng đã tắt Lookahead.");
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
    //: Khoảng "dịch trước" do BỘ CẮT KHỐI quyết định (12–30 s, xem `LookaheadChunker`); giá trị
    //: dưới đây chỉ là cổng throttle nội bộ phía backend và không còn người dùng chỉnh trong popup.
    const leadTimeSec = 15;
    console.log(`%c[BS] 🚀 Khởi động Pipeline B: Lookahead OFFLINE_BATCH (0.0s Lag)`, "color: #00ffcc; font-weight: bold;");
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

    // ── CHẨN ĐOÁN PHIÊN PIPELINE B (v0.6.7) ────────────────────────────────────
    // Vì sao cần: đo thật trên xvideos.com (2026-10-05) — backend log cùng MỘT mốc Playhead
    // 15.7s lặp lại mỗi 10s, "Đã dịch: 0.0s", 'giải mã 0.00x, PCM/byte 0.000' trong ~50s, rồi
    // người dùng bấm Stop. Log backend KHÔNG nói được client đang làm gì; log trình duyệt cũng
    // không có gì. Vòng lặp thật là:
    //   horizon chặn playhead → pause → (video dừng nên trình phát KHÔNG append mảnh mới)
    //   → backend không có audio mới → mốc đã xử lý đứng yên → horizon lại chặn… BẾ TẮC.
    // Khối dưới đây in ra đúng vòng lặp đó (đếm pause/resume, số liệu từ `lookahead_status`)
    // và có VAN AN TOÀN tự phá bế tắc sau `LA_STALL_GUARD_MS`.
    const LA_STALL_GUARD_MS = 25000;      // không tiến triển bao lâu thì coi là bế tắc
    //: Ngưỡng RIÊNG cho lúc CHƯA có phụ đề nào (`utterances=0`): chờ lâu như trên sẽ khiến khúc
    //: đầu video khựng ~25s (đo thật 2026-10-05, xvideos.com: phải 2 lần van an toàn mới qua).
    const LA_FIRST_STALL_GUARD_MS = 12000;
    //: ĐỆM ĐÃ DỊCH tối thiểu để cho video CHẠY LẠI (giây).
    //:
    //: Vì sao cần (đo thật 2026-10-05, YouTube): POPUP báo `Lookahead available +60s` (đệm VIDEO)
    //: nhưng `ready_ahead` (đệm ĐÃ DỊCH) lúc đầu chỉ 0,3–3 s. Client cũ chạy lại ngay khi
    //: `prebuffer_ready`/`ready ≥ 1.5s` ⇒ chạy được 1–2 s là chạm mốc đã xử lý ⇒ tạm dừng lại
    //: (`lý do dừng={"start":1,"underrun":2}`, mỗi lần dừng 1–2 s). Đệm video dồi dào KHÔNG có
    //: nghĩa là đã dịch đủ — hai đại lượng khác nhau.
    const LA_MIN_RESUME_AHEAD_SEC = 4.0;
    const LA_HORIZON_SUSPEND_MS = 20000;  // treo ràng buộc horizon bao lâu khi phá bế tắc
    const stallGuardEnabled = (typeof window.__bsStallGuard === "boolean")
      ? window.__bsStallGuard
      : settings.lookaheadStallGuard !== false;
    const laSession = {
      startedAt: Date.now(),
      pauses: 0,
      resumes: 0,
      pauseReasons: {},
      pauseStartedAt: 0,
      lastPauseReason: null,
      lastStatus: null,
      lastStatusAt: 0,
      lastFragments: -1,
      lastDecoded: -1,
      lastReadyUntil: -1,
      lastFragmentsAt: Date.now(),
      lastReadyAt: Date.now(),
      lastStatusLogAt: 0,
      lastStatusLogCount: 0,
      lastGuardAt: 0,
      guardFired: 0,
      ignoredSeeks: 0,
      horizonSuspendedForStall: false,
      watchTimer: null,
    };
    laDiag.log("session", `Bắt đầu phiên Pipeline B: playhead=${curTime.toFixed(2)}s, `
      + `đệm trước=+${aheadSec.toFixed(2)}s, lead_time=${leadTimeSec}s, `
      + `TTS=${settings.ttsEnabled ? "BẬT" : "tắt"}, van-an-toàn=${stallGuardEnabled ? "BẬT" : "tắt"}.`);
    // Dừng vòng theo dõi của phiên trước (nếu Stop trước đó chưa dọn hết).
    if (lookaheadSessionDiag && lookaheadSessionDiag.watchTimer) {
      try { clearInterval(lookaheadSessionDiag.watchTimer); } catch (e) {}
      lookaheadSessionDiag.watchTimer = null;
    }
    lookaheadSessionDiag = laSession;

    /** Đệm đã dịch hiện có (giây), đã trừ thời gian trôi từ lúc backend báo. */
    const currentReadyAhead = () => {
      const st = laSession.lastStatus || {};
      const ready = Number(st.ready_ahead || 0);
      if (!(ready > 0)) return 0;
      const ageSec = Math.max(0, (Date.now() - Number(laSession.lastStatusAt || Date.now())) / 1000);
      const rate = Math.max(0.25, Math.abs(Number(video.playbackRate) || 1));
      return Math.max(0, ready - ageSec * rate);
    };

    /**
     * Đã đủ đệm ĐÃ DỊCH để cho video chạy mà không phải dừng lại ngay chưa?
     *
     * Hai điều kiện đạt:
     *   • `ready ≥ LA_MIN_RESUME_AHEAD_SEC`, hoặc
     *   • đoạn trước là KHOẢNG IM LẶNG (`fed ≥ 4s` mà `ready = 0`) — không có gì để dịch nên chờ
     *     thêm là vô ích.
     *
     * Không bao giờ chặn vĩnh viễn: các `resumeTimer` (chốt an toàn theo đồng hồ thực) vẫn chạy
     * song song và sẽ cho video phát kể cả khi backend không bao giờ đạt ngưỡng.
     */
    const hasEnoughHeadroomToResume = () => {
      const st = laSession.lastStatus || {};
      const fed = Number(st.fed_ahead || 0);
      if (st.prebuffer_ready && fed >= 3.5 && Number(st.ready_ahead || 0) <= 0.05) return true;  // khoảng lặng do backend xác nhận
      return currentReadyAhead() >= LA_MIN_RESUME_AHEAD_SEC;
    };

    /** In một dòng trạng thái rút gọn từ `lookahead_status` của backend. */
    const logLookaheadStatus = (st, tag) => {
      const s = st || {};
      laDiag.log("status",
        `${tag ? tag + " " : ""}playhead=${Number(s.currentTime || video.currentTime).toFixed(2)}s | `
        + `ready=${Number(s.ready_ahead || 0).toFixed(2)}s fed=${Number(s.fed_ahead || 0).toFixed(2)}s `
        + `target=${Number(s.target_ahead || 0).toFixed(2)}s | mốc-đã-xử-lý=${Number(s.ready_until_pts || 0).toFixed(2)}s `
        + `prebuffer=${s.prebuffer_ready} tts_sent=${s.tts_sent ?? "-"} | `
        + `fragments=${s.fragments ?? "-"} decoded=${s.decoded_chunks ?? "-"} utterances=${s.utterances ?? "-"}`);
    };

    /**
     * VAN AN TOÀN CHỐNG BẾ TẮC HORIZON.
     *
     * Chỉ kích hoạt khi: video ĐANG bị extension tạm dừng VÀ backend không nhận thêm mảnh nào
     * trong `LA_STALL_GUARD_MS`. Lúc đó cho video chạy lại + TREO ràng buộc horizon một lúc để
     * trình phát nạp thêm dữ liệu — thay vì đứng hình vô hạn.
     */
    const fireStallGuard = async () => {
      try {
      const st = laSession.lastStatus || {};
      laSession.guardFired += 1;
      laDiag.warn("stall",
        `🚑 PHÁ BẾ TẮC (lần ${laSession.guardFired}): playhead ghim ở ${video.currentTime.toFixed(2)}s, `
        + `${Math.round((Date.now() - laSession.lastFragmentsAt) / 1000)}s không có mảnh mới, `
        + `${Math.round((Date.now() - laSession.lastReadyAt) / 1000)}s mốc-đã-xử-lý không tiến. `
        + `fragments=${st.fragments ?? "-"} decoded=${st.decoded_chunks ?? "-"} `
        + `ready_until=${Number(st.ready_until_pts || 0).toFixed(2)}s. `
        + `NGUYÊN NHÂN: horizon chặn playhead ⇒ video dừng ⇒ trình phát không append mảnh mới ⇒ `
        + `backend không có audio mới. Cho video CHẠY LẠI và treo horizon ${LA_HORIZON_SUSPEND_MS / 1000}s.`);
      try { timelineQueue.suspendHorizon(LA_HORIZON_SUSPEND_MS); } catch (e) {}
      // Lần thứ 2 trở đi thì chờ lâu hơn hẳn: nếu backend vẫn không tiến triển sau khi đã được
      // cho thêm dữ liệu, việc tạm dừng video chỉ tạo nhịp "chạy một chút lại dừng" (đo thật
      // 2026-10-05, xvideos.com: pause=9/resume=8, van-an-toàn phá bế tắc 1 lần rồi lặp lại).
      if (laSession.guardFired >= 2) {
        try { timelineQueue.suspendHorizon(180000); } catch (e) {}
        laDiag.warn("stall",
          `⏯️ Đã phá bế tắc ${laSession.guardFired} lần ⇒ TREO ràng buộc horizon 180s để video `
          + `phát liền mạch (tạm thời không phụ đề) thay vì khựng theo chu kỳ.`);
      }
      laSession.lastFragmentsAt = Date.now();
      laSession.lastReadyAt = Date.now();
      resumePlayback("stall_guard");
      // In thêm số liệu interceptor (cache còn gì) để biết nguồn có thiếu audio thật không.
      try {
        const p = await getLookaheadBufferStatus(300, video);
        if (p) {
          laDiag.log("stall", `Interceptor: cache=${p.cachedChunksCount} mảnh `
            + `(có mốc media: ${p.mediaTaggedChunks}), init=${p.hasInitSegment ? "có" : "chưa"}, `
            + `đệm video trước=+${Number(p.aheadSeconds || 0).toFixed(2)}s, `
            + `append=${p.diag ? p.diag.appendCount : "-"} lần, `
            + `ms-từ-append-cuối=${p.diag && p.diag.msSinceLastAppend !== null ? Math.round(p.diag.msSinceLastAppend) : "-"}`);
        } else {
          laDiag.warn("stall", "Interceptor KHÔNG phản hồi khi hỏi trạng thái trong lúc bế tắc.");
        }
      } catch (e) {}
      } catch (e) {
        // Chẩn đoán/van an toàn KHÔNG được phép làm hỏng phiên dịch.
        console.warn("[BS][Diag] Lỗi trong van an toàn chống bế tắc:", e);
      }
    };

    // Vòng theo dõi bế tắc: chạy trong suốt phiên, tự tắt khi phiên kết thúc.
    laSession.watchTimer = setInterval(() => {
      try {
      if (!lookaheadPausedVideo) return;
      const now = Date.now();
      const stalledFrag = now - laSession.lastFragmentsAt;
      const stalledReady = now - laSession.lastReadyAt;
      // Điều kiện "bế tắc" phải chặt để KHÔNG phá nhầm lúc backend đang chạy một khối ASR dài:
      //   (1) không có mảnh mới VÀ mốc đã xử lý không tiến trong `LA_STALL_GUARD_MS`, và
      //   (2) đây đã là lần tạm dừng thứ HAI trở đi (lần đầu chỉ là nạp đệm bình thường).
      const looksDeadlocked = laSession.pauses >= 2;
      // Chưa ra được phụ đề nào ⇒ dùng ngưỡng ngắn hơn để thoát vòng pause/resume của khúc đầu.
      const noUtteranceYet = Number((laSession.lastStatus || {}).utterances || 0) === 0;
      const guardMs = noUtteranceYet ? LA_FIRST_STALL_GUARD_MS : LA_STALL_GUARD_MS;
      // ── BACKEND "NHẬN MÀ KHÔNG RA" (đo thật 2026-10-05, xhamster.com) ──────────
      // Mảnh vẫn về đều (`fragments` tăng) nhưng mốc-đã-xử-lý đứng im và `decoded_chunks` không
      // nhích: byte về mà KHÔNG giải mã ra PCM. Van `fireStallGuard` ở dưới không bắt được ca này
      // (nó đòi cả `fragments` đứng yên), nên vòng pause/seek cứ thế chạy mãi: log backend cho
      // thấy `giải mã 0.00x, PCM/byte 0.000` trong khi playhead chạy từ 96s tới 201s.
      // Việc tạm dừng video lúc này là VÔ NGHĨA (backend không thiếu audio, nó không giải mã
      // được) ⇒ log rõ và TREO ràng buộc horizon để dừng vòng nạp lại.
      const backendReceivingNoPcm = laSession.lastFragments > 0
        && stalledReady > 20000 && stalledFrag <= stalledReady;
      if (backendReceivingNoPcm) {
        const st = laSession.lastStatus || {};
        laDiag.throttled("backend-no-pcm", 10000, "warn", "stall",
          `⚠️ Backend ĐANG NHẬN mảnh (fragments=${st.fragments ?? "-"}, mảnh mới cách đây `
          + `${Math.round(stalledFrag / 1000)}s) nhưng mốc-đã-xử-lý KHÔNG tiến ${Math.round(stalledReady / 1000)}s `
          + `(decoded=${st.decoded_chunks ?? "-"}, ready=${Number(st.ready_ahead || 0).toFixed(2)}s). `
          + `⇒ byte về mà KHÔNG ra PCM/phụ đề: nút cổ chai ở tầng GIẢI MÃ phía backend. `
          + `Xem log backend các dòng: PCM/byte, LỆCH TRỤC THỜI GIAN, `
          + `Đường giải mã nối-liền không ra PCM, StreamDemuxer reset.`);
        if (stalledReady > 45000 && !laSession.horizonSuspendedForStall) {          laSession.horizonSuspendedForStall = true;
          try { timelineQueue.suspendHorizon(180000); } catch (e) {}
          laDiag.warn("stall",
            "⏯️ Đã TREO ràng buộc horizon 180s để DỪNG vòng pause/seek: giữ video phát bình thường "
            + "(tạm thời không có phụ đề) thay vì nạp lại liên tục. Horizon tự gỡ khi backend đuổi kịp.");
        }
        return;
      }
      if (looksDeadlocked && stalledFrag > guardMs && stalledReady > guardMs) {
        if (stallGuardEnabled) {
          if (now - (laSession.lastGuardAt || 0) > guardMs) {
            laSession.lastGuardAt = now;
            void fireStallGuard();
          }
        } else {
          laDiag.throttled("stall-off", 10000, "warn", "stall",
            `⛔ BẾ TẮC nhưng van an toàn đang TẮT: ${Math.round(stalledFrag / 1000)}s không có mảnh mới, `
            + `${Math.round(stalledReady / 1000)}s mốc-đã-xử-lý không tiến, video vẫn bị tạm dừng.`);
        }
      } else {
        laDiag.throttled("stall-info", 10000, "log", "stall",
          `⏳ Đang chờ backend: video tạm dừng ${Math.round((now - laSession.pauseStartedAt) / 1000)}s `
          + `(lý do=${laSession.lastPauseReason}), mảnh mới cách đây ${Math.round(stalledFrag / 1000)}s, `
          + `mốc-đã-xử-lý tiến cách đây ${Math.round(stalledReady / 1000)}s.`);
      }
      } catch (e) { /* chẩn đoán không được làm hỏng phiên */ }
    }, 2000);

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
      // ── LOG VÒNG LẶP PAUSE/RESUME (chẩn đoán bế tắc) ─────────────────────────
      // Trên xvideos.com vòng lặp này chạy mãi mà KHÔNG ai thấy: video bị tạm dừng ⇒ trình phát
      // không append mảnh mới ⇒ backend không tiến triển ⇒ horizon lại chặn ⇒ tạm dừng tiếp.
      {
        const now = Date.now();
        const wasPausedMs = laSession.pauseStartedAt ? now - laSession.pauseStartedAt : 0;
        laSession.pauses += 1;
        laSession.pauseReasons[why || "start"] = (laSession.pauseReasons[why || "start"] || 0) + 1;
        laSession.lastPauseReason = why || "start";
        laSession.pauseStartedAt = now;
        const st = laSession.lastStatus || {};
        laDiag.warn("session",
          `⏸️ TẠM DỪNG video để nạp đệm (lần ${laSession.pauses}, lý do=${why || "start"}` +
          `${wasPausedMs > 0 ? `, vừa chạy lại được ${(wasPausedMs / 1000).toFixed(1)}s` : ""}) | ` +
          `playhead=${video.currentTime.toFixed(2)}s | ` +
          `mốc-đã-xử-lý=${Number(st.ready_until_pts || 0).toFixed(2)}s ` +
          `ready=${Number(st.ready_ahead || 0).toFixed(2)}s fed=${Number(st.fed_ahead || 0).toFixed(2)}s ` +
          `fragments=${st.fragments ?? "-"} tts_sent=${st.tts_sent ?? "-"} | ` +
          `tổng pause=${laSession.pauses} resume=${laSession.resumes}`);
      }
      showBufferingStatus(
        why === "seek"
          ? "Đang dịch trước đoạn vừa tua..."
          : (why === "tts_enable"
              ? "Đang chuẩn bị lồng tiếng TTS..."
              : (why === "underrun"
                  ? "Đang xử lý tiếp đoạn video..."
                  : "Đang nạp đệm và dịch trước..."))
      );
      // Chốt an toàn (đồng hồ ĐỒNG HỒ THỰC). Với TUA, mục tiêu là "dịch xong ngay tại vị trí
      // mới rồi mới phát" nên KHÔNG cắt cứng ở 3,5 s: nếu backend còn tiến triển thì gia hạn
      // Chốt an toàn (đồng hồ ĐỒNG HỒ THỰC): kiên nhẫn chờ backend gom khối, ASR và TTS.
      // Với Pipeline B, cần đảm bảo mọi thứ đã sẵn sàng trước khi phát video.
      const timeoutMs = why === "seek"
        ? (settings.ttsEnabled ? 8000 : 6000)
        : (why === "underrun"
            ? (settings.ttsEnabled ? 12000 : 9000)
            : (settings.ttsEnabled ? 8000 : 6000));
      bufferProgress = { ready: 0, fed: 0, at: Date.now(), extensions: 0 };
      if (resumeTimer) clearTimeout(resumeTimer);
      const scheduleResume = (ms) => {
        resumeTimer = setTimeout(() => {
          if (!lookaheadPausedVideo) return;
          // Còn tiến triển (ready_ahead/fed_ahead tăng) ⇒ gia hạn thay vì phát khi chưa có
          // phụ đề. Tối đa 5 lần để không treo trình phát.
          const progressing = Date.now() - bufferProgress.at < 2500;
          const grew = (bufferProgress.ready > 0 || bufferProgress.fed > 0) && progressing;
          // Gia hạn THÊM khi đệm đã dịch còn dưới ngưỡng mà backend VẪN đang nhận/giải mã audio:
          // phát lúc này là chạy 1-2s rồi phải dừng lại (nhịp pause/play ngắn — đo thật trên
          // YouTube 2026-10-05). Chỉ gia hạn khi backend còn sống, tối đa 5 lần (≈10s).
          const backendAlive = Date.now() - Number(laSession.lastFragmentsAt || 0) < 3000;
          const belowTarget = !hasEnoughHeadroomToResume();
          if ((why === "seek" || why === "start")
              && bufferProgress.extensions < 5
              && (grew || (belowTarget && backendAlive))) {
            bufferProgress.extensions += 1;
            console.log(
              `[BS] Chờ thêm bản dịch tại vị trí hiện tại (ready ${bufferProgress.ready.toFixed(1)}s, ` +
              `cần ≥ ${LA_MIN_RESUME_AHEAD_SEC.toFixed(1)}s; gia hạn ${bufferProgress.extensions}/5).`
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

    // Với Pipeline B (Lookahead), LUÔN tạm dừng video khi bắt đầu để nạp đệm và dịch trước
    // (xác định mọi thứ đã sẵn sàng trước khi phát video, tránh mất phụ đề khúc đầu).
    pauseForBuffering("start");

    // 2. Khởi tạo SubtitleTimelineQueue
    timelineQueue = new SubtitleTimelineQueue({
      onSubtitleChange: (sub) => {
        if (sub) {
          emitSubtitleEvent("translation", {
            utterance_id: sub.id,
            original_text: sub.original_text,
            translated_text: sub.translated_text,
            translated: sub.translated_text,
            text: sub.translated_text || sub.original_text,
            status: "ok",
            is_final: true,
          });
        } else {
          if (overlayManager) {
            try { overlayManager.clear(); } catch (e) {}
          }
        }
      },
      onBufferingStateChange: (isBuffering) => {
        if (isBuffering) {
          // RÀNG BUỘC CỨNG: playhead đã chạm mốc backend xử lý xong (hoặc vừa tua) ⇒ TẠM DỪNG
          // video. Nếu không, video sẽ phát qua vùng chưa có phụ đề/bản dịch — "vẫn phát video mà
          // không có gì hoạt động" (sự cố 2026-10-02).
          pauseForBuffering(currentBufferingWhy === "seek" ? "seek" : "underrun");
          return;
        }
        if (lookaheadPausedVideo) {
          if (!settings.ttsEnabled || firstTtsReceived) {
            // Lưới an toàn 8s của `SubtitleTimelineQueue` chỉ là "hết thời gian chờ", KHÔNG có
            // nghĩa là đã đủ đệm để phát: resume ở đây mà đệm đã dịch còn mỏng thì video chạy được
            // 1-2s rồi lại bị `_tick` tạm dừng ngay (nhịp pause/play ngắn).
            if (hasEnoughHeadroomToResume()) {
              resumePlayback("prebuffer_ready");
            } else {
              laDiag.throttled("resume-blocked-timeline", 4000, "log", "session",
                `⏳ Hết thời gian nạp đệm nhưng đệm đã dịch mới ${currentReadyAhead().toFixed(1)}s `
                + `(< ${LA_MIN_RESUME_AHEAD_SEC.toFixed(1)}s) — chờ thêm để khỏi phải dừng lại ngay.`);
            }
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
        laDiag.log("session", `⏩ Người dùng TUA tới ${Number(targetTime).toFixed(2)}s (seek_id=${seekId}) `
          + `⇒ xoá phụ đề cũ, tạm dừng chờ backend dịch trước vị trí mới.`);
        // Vị trí mới chưa có phụ đề: TẠM DỪNG video chờ backend dịch trước, rồi tự phát lại
        // khi `lookahead_status.prebuffer_ready` báo sẵn sàng.
        pauseForBuffering("seek");
      },
      onUnderrun: (curTime, readyUntilPts) => {
        // RÀNG BUỘC CỨNG vừa kích hoạt: playhead đã chạm mốc backend xử lý xong. Nếu mốc này
        // KHÔNG tiến (backend không có thêm audio), đây chính là vế đầu của BẾ TẮC.
        laDiag.throttled("underrun", 4000, "warn", "session",
          `⛔ Chạm mốc đã xử lý: playhead=${Number(curTime).toFixed(2)}s ≥ `
          + `mốc-đã-xử-lý=${Number(readyUntilPts).toFixed(2)}s ⇒ tạm dừng chờ backend. `
          + `Nếu dòng này lặp lại mãi cùng một mốc ⇒ BẾ TẮC (xem [BS][Diag][stall]).`);
      },
      onSeekIgnored: (info) => {
        // ── `seeking` KHÔNG phải do người dùng tua (nền tảng của "video nạp lại liên tục") ──
        // Đo thật 2026-10-05 (xhamster.com): `lý do dừng={"start":1,"seek":38}` — 38 lần reset
        // backend + tạm dừng video chỉ vì trình phát tự bắn `seeking` khi hết đệm.
        // `churnOnly` = vẫn là TUA THẬT nhưng dồn dập: vẫn nhận (để vị trí cuối không bị mất),
        // chỉ cảnh báo.
        laSession.ignoredSeeks = (laSession.ignoredSeeks || 0) + (info.churnOnly ? 0 : 1);
        laDiag.throttled("seek-ignored", 2000, "warn", "session",
          info.churnOnly
            ? `⚡ Tua dồn dập (${info.accepted} lần/${(timelineQueue.seekChurnWindowMs / 1000).toFixed(0)}s) — `
              + `vẫn NHẬN vị trí cuối ${info.targetTime.toFixed(2)}s để không mất tua thật.`
            : `↩️ Bỏ qua "seeking" do TRÌNH PHÁT TỰ SINH (nhảy ${info.jump.toFixed(2)}s) — `
              + `KHÔNG reset backend, KHÔNG tạm dừng video. Tổng: bỏ qua ${info.ignored}, `
              + `chấp nhận ${info.accepted}. (Chỉ coi là tua thật khi nhảy ≥ `
              + `${timelineQueue ? timelineQueue.minSeekJumpSec : 3}s.)`);
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
      // ── LOG: vì sao được phát tiếp (đối chiếu với dòng ⏸️ TẠM DỪNG) ──────────
      laSession.resumes += 1;
      const pausedMs = laSession.pauseStartedAt ? Date.now() - laSession.pauseStartedAt : 0;
      laSession.pauseStartedAt = 0;
      laDiag.log("session",
        `▶️ PHÁT TIẾP video (lần ${laSession.resumes}, lý do=${reason || "ready"}` +
        `${pausedMs > 0 ? `, đã dừng ${(pausedMs / 1000).toFixed(1)}s` : ""}) | ` +
        `playhead=${video.currentTime.toFixed(2)}s | tổng pause=${laSession.pauses} resume=${laSession.resumes}` +
        `${laSession.guardFired ? ` | van-an-toàn đã phá bế tắc ${laSession.guardFired} lần` : ""}`);
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
        onSubtitles: (items) => {
          if (lookaheadPausedVideo && items && items.length) {
            const curTime = video ? video.currentTime : 0;
            const hasMatching = items.some((it) => curTime >= Number(it.start_pts) - 0.5 && curTime <= Number(it.end_pts) + 0.5);
            if (hasMatching) {
              if (!settings.ttsEnabled || firstTtsReceived) {
                // CHỈ chạy lại khi còn ĐỦ ĐỆM ĐÃ DỊCH. Câu vừa gửi khớp đúng vị trí phát nghĩa là
                // ta đang ở SÁT mốc đã xử lý — resume lúc này là chạy được 1-2s rồi phải dừng lại.
                if (hasEnoughHeadroomToResume()) resumePlayback("subtitle_arrived");
              }
            }
          }
        },
        onPrebufferReady: (st) => {
          // Lưới an toàn khi backend báo prebuffer_ready trực tiếp
          const isTts = Boolean(settings.ttsEnabled);
          if (isTts && !firstTtsReceived) return;
          if (hasEnoughHeadroomToResume()) {
            resumePlayback("prebuffer_ready");
          }
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

          // Câu TTS đầu tiên đã sẵn sàng — nhưng vẫn phải có đủ đệm đã dịch phía trước, nếu không
          // video chạy được vài giây lại phải dừng (đúng nhịp pause/play khó chịu cần tránh).
          if (settings.ttsEnabled && lookaheadPausedVideo && !firstTtsReceived) {
            firstTtsReceived = true;
            if (hasEnoughHeadroomToResume()) resumePlayback("tts_first_ready");
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
          // ── THEO DÕI TIẾN TRIỂN (chẩn đoán) ──────────────────────────────────
          // `fragments` = số mảnh audio backend đã nhận; `ready_until_pts` = mốc backend đã
          // ASR+dịch xong. Hai con số này đứng yên trong lúc video bị tạm dừng = BẾ TẮC (xem
          // `fireStallGuard`). Chúng cũng là thứ DUY NHẤT cho biết backend còn sống hay không.
          {
            const now = Date.now();
            const frag = Number(st.fragments);
            const dec = Number(st.decoded_chunks);
            const ru = Number(st.ready_until_pts);
            if (Number.isFinite(frag) && frag !== laSession.lastFragments) {
              laSession.lastFragments = frag;
              laSession.lastFragmentsAt = now;
            }
            if (Number.isFinite(dec)) laSession.lastDecoded = dec;
            if (Number.isFinite(ru) && ru !== laSession.lastReadyUntil) {
              if (ru > laSession.lastReadyUntil) laSession.lastReadyAt = now;
              laSession.lastReadyUntil = ru;
            }
            laSession.lastStatus = st;
            laSession.lastStatusAt = now;
            // Backend đã đuổi kịp playhead ⇒ gỡ treo horizon (nếu van an toàn đã treo trước đó).
            if (timelineQueue && Number.isFinite(ru) && ru > (Number(video.currentTime) || 0) + 0.5) {
              try { timelineQueue.resumeHorizon(); } catch (e) {}
            }
            // In trạng thái: lần đầu luôn in, sau đó mỗi 3s khi ĐANG chờ nạp đệm (lúc cần chẩn
            // đoán nhất) và mỗi 10s khi đang phát bình thường — `lookahead_status` được backend
            // gửi vài lần mỗi giây nên không thể in hết.
            const statusLogEveryMs = lookaheadPausedVideo ? 3000 : 10000;
            if (now - (laSession.lastStatusLogAt || 0) >= statusLogEveryMs) {
              laSession.lastStatusLogAt = now;
              logLookaheadStatus(st, laSession.lastStatusLogCount++ === 0 ? "(đầu tiên)" : "");
            }
          }
          // Nạp MỐC ĐÃ XỬ LÝ cho timeline queue: đây là ràng buộc cứng "playhead không được vượt
          // qua vùng backend chưa xử lý" (kiểm tra mỗi khung hình trong `_tick`).
          if (timelineQueue) {
            try { timelineQueue.setReadyHorizon(st.ready_until_pts, st.seek_id); } catch (e) {}
          }
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

            // Sẵn sàng CHẠY LẠI khi CẢ HAI điều kiện đều đạt:
            // 1. Backend đã xác nhận sẵn sàng (st.prebuffer_ready === true): đã dịch đủ lead time
            //    hoặc đã xác nhận khoảng lặng an toàn. Tuyệt đối không phát khi backend báo false.
            // 2. Client có đủ đệm đã dịch phía trước (hasEnoughHeadroomToResume()):
            //    - TTS tắt: ready >= LA_MIN_RESUME_AHEAD_SEC hoặc khoảng lặng (do backend xác nhận).
            //    - TTS bật: đã nhận câu TTS đầu tiên.
            const backendReady = Boolean(st.prebuffer_ready);
            let readyToResume = false;
            if (backendReady && hasEnoughHeadroomToResume() && (!isTts || firstTtsReceived)) {
              readyToResume = true;
            }

            if (readyToResume) {
              resumePlayback("prebuffer_ready");
            } else if (lookaheadPausedVideo && now - (laSession.lastResumeBlockedLogAt || 0) > 3000) {
              // Chẩn đoán: nói rõ VÌ SAO còn phải chờ dù backend đã báo `prebuffer_ready`.
              laSession.lastResumeBlockedLogAt = now;
              laDiag.log("session",
                `⏳ Chưa cho phát: chờ backend & đệm sẵn sàng (prebuffer_ready=${st.prebuffer_ready}, `
                + `ready=${ready.toFixed(1)}s/${LA_MIN_RESUME_AHEAD_SEC.toFixed(1)}s, fed=${fed.toFixed(1)}s`
                + `${isTts && !firstTtsReceived ? ", chờ TTS đầu" : ""}) — kiên nhẫn chờ để không mất phụ đề.`);
            }
          }
          // ĐANG PHÁT: việc TẠM DỪNG giữa chừng do RÀNG BUỘC CỨNG ở `timelineQueue._tick()`
          // quyết định (chỉ khi playhead chạm mốc backend đã xử lý — xem `setReadyHorizon`), chứ
          // không dựa vào `ready_ahead` của status: `ready_ahead` có thể còn giá trị của vị trí cũ.
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
      // Giải thích MÃ LÝ DO bằng tiếng Việt + cách xử lý (từ điển ở lib/lookahead-diagnostics.js).
      if (LA_DIAG) {
        const code = /timeout/i.test(availability) ? "CONNECTION_TIMEOUT"
          : (/unavailable/i.test(availability) ? "BACKEND_UNAVAILABLE" : null);
        const ex = LA_DIAG.explainLookaheadReason(code || availability);
        laDiag.group(`⚠️ Pipeline B thất bại: ${availability}`, [
          ["Vì sao", ex.why],
          ["Cách xử lý", ex.fix],
          ["Thống kê phiên", `pause=${laSession.pauses}, resume=${laSession.resumes}, `
            + `van-an-toàn=${laSession.guardFired}, lý do dừng=${JSON.stringify(laSession.pauseReasons)}`],
          ["Trạng thái backend cuối", JSON.stringify(laSession.lastStatus || null)],
        ], "warn");
      }
      if (laSession.watchTimer) { try { clearInterval(laSession.watchTimer); } catch (e) {} laSession.watchTimer = null; }
      lookaheadSessionDiag = null;
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
    // Dừng vòng theo dõi chẩn đoán của phiên (nếu để sống, nó sẽ báo bế tắc cho phiên đã Stop).
    if (lookaheadSessionDiag) {
      if (lookaheadSessionDiag.watchTimer) {
        try { clearInterval(lookaheadSessionDiag.watchTimer); } catch (e) {}
        lookaheadSessionDiag.watchTimer = null;
      }
      laDiag.log("session", `⏹️ Kết thúc phiên Pipeline B: pause=${lookaheadSessionDiag.pauses}, `
        + `resume=${lookaheadSessionDiag.resumes}, van-an-toàn=${lookaheadSessionDiag.guardFired}, `
        + `seek-bỏ-qua=${lookaheadSessionDiag.ignoredSeeks || 0}, `
        + `lý do dừng=${JSON.stringify(lookaheadSessionDiag.pauseReasons)}, `
        + `mốc-đã-xử-lý cuối=${Number((lookaheadSessionDiag.lastStatus || {}).ready_until_pts || 0).toFixed(2)}s.`);
      lookaheadSessionDiag = null;
    }
    if (lookaheadPausedVideo) {
      lookaheadPausedVideo = false;
      try {
        const v = getVideo();
        // KHÔNG `play()` khi video đã phát hết: gọi `play()` trên video `ended` sẽ khiến nó CHẠY
        // LẠI TỪ ĐẦU — đúng lúc ta vừa tự đóng phiên vì hết video (sự cố 2026-10-05).
        if (v && v.paused && !v.ended) v.play();
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
    if (typeof window !== "undefined" && typeof window.postMessage === "function") {
      try {
        window.postMessage({
          source: "VIBE_LOOKAHEAD_CLIENT",
          type: "RESET_REPLAY_TOKEN",
        }, "*");
      } catch (e) {}
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

  /**
   * CHẨN ĐOÁN THỦ CÔNG — gõ trong Console của trang:
   *   __bsLookaheadDiagReport()
   * In: quyết định cổng Pipeline B, dữ kiện thô của interceptor, và các bước xử lý.
   */
  window.__bsLookaheadDiagReport = async () => {
    const v = findVideo();
    announcePreferredVideo(v);
    const p = await getLookaheadBufferStatus(600, v);
    const facts = buildLookaheadFacts(p, v);
    const verdict = LA_DIAG ? LA_DIAG.classifyLookaheadAvailability(facts) : { code: "UNKNOWN", detail: "thiếu module" };
    printLookaheadDiagnosis(facts, verdict);
    if (lookaheadSessionDiag) {
      laDiag.group("Thống kê phiên Pipeline B đang chạy", [
        ["pause / resume", `${lookaheadSessionDiag.pauses} / ${lookaheadSessionDiag.resumes}`],
        ["lý do dừng", JSON.stringify(lookaheadSessionDiag.pauseReasons)],
        ["van an toàn đã phá bế tắc", String(lookaheadSessionDiag.guardFired)],
        ["trạng thái backend cuối", JSON.stringify(lookaheadSessionDiag.lastStatus || null)],
      ]);
    }
    if (lookaheadClient && typeof lookaheadClient.getDiag === "function") {
      laDiag.group("Thống kê gửi/nhận tới /ws/lookahead", Object.entries(lookaheadClient.getDiag())
        .map(([k, val]) => [k, typeof val === "object" ? JSON.stringify(val) : String(val)]));
    }
    console.log("[BS][Diag] Bước tiếp theo: nếu mã là MUXED_ONLY/UNKNOWN_MIME_SB hãy xem log "
      + "`[Lookahead][Diag] addSourceBuffer` và `sniff` ở đầu Console của trang.");
    return { verdict, facts, payload: p };
  };

  // ── WATCHDOG CHẨN ĐOÁN (khi CHƯA capture) ──────────────────────────────────
  // Chạy mỗi 5s khi trang có <video> nhưng chưa bắt đầu dịch: tự kết luận và in MỘT LẦN cho mỗi
  // mã lý do mới. Nhờ vậy người dùng chỉ cần mở Console là biết vì sao nút Start sẽ dùng
  // Pipeline A/B, không phải bấm Start rồi mới biết.
  let laWatchdogTimer = null;
  let laWatchdogLastCode = null;
  function startLookaheadWatchdog() {
    if (laWatchdogTimer) return;
    laWatchdogTimer = setInterval(async () => {
      try {
        if (isCapturing) return;
        const v = findVideo();
        if (!v) return;
        if (!(v.readyState > 0) && v.paused && !v.currentSrc) return; // trang chưa nạp video
        announcePreferredVideo(v);
        const p = await getLookaheadBufferStatus(300, v);
        const facts = buildLookaheadFacts(p, v);
        const verdict = LA_DIAG ? LA_DIAG.classifyLookaheadAvailability(facts) : null;
        if (!verdict) return;
        if (verdict.code === laWatchdogLastCode) return;
        laWatchdogLastCode = verdict.code;
        if (verdict.code === "OK") {
          laDiag.ok("watchdog", `Trang này ĐỦ điều kiện chạy Pipeline B: ${verdict.detail}`);
          return;
        }
        if (verdict.code === "NO_VIDEO" || verdict.code === "NO_APPEND_YET") {
          // Trạng thái bình thường khi video chưa được bấm Play — chỉ ghi ở mức thông tin.
          laDiag.log("watchdog", `Chưa sẵn sàng (${verdict.code}): ${verdict.detail}`);
          return;
        }
        laDiag.warn("watchdog", `Trang này sẽ KHÔNG chạy được Pipeline B: ${verdict.code} — ${verdict.detail}`);
        printLookaheadDiagnosis(facts, verdict);
      } catch (e) { /* im lặng: chẩn đoán không được làm hỏng trang */ }
    }, 5000);
  }
  startLookaheadWatchdog();

  console.log("[BS Content] Ready (Frame: " + (window === window.top ? "Top" : "Iframe") + ")");
  console.log("%c[BS][Diag] Chẩn đoán Lookahead: gõ `__bsLookaheadDiagReport()` trong Console của "
    + "trang. Lọc Console theo `[Diag]` để xem toàn bộ chuỗi quyết định.",
    "color: #94a3b8;");
})();
