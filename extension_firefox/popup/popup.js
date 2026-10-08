// Popup - Sends START/STOP to content script and manages persisted preferences & dynamic engine switching
const api = typeof browser !== "undefined" ? browser : chrome;

(function () {
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnTest = document.getElementById("btnTest");
  const statusBadge = document.getElementById("statusBadge");
  const messageArea = document.getElementById("messageArea");

  const selAsrEngine = document.getElementById("selAsrEngine");
  const selVadEngine = document.getElementById("selVadEngine");
  const rangeVadSilence = document.getElementById("rangeVadSilence");
  const valVadSilence = document.getElementById("valVadSilence");
  const rangeVadThreshold = document.getElementById("rangeVadThreshold");
  const valVadThreshold = document.getElementById("valVadThreshold");
  const rangeMinWords = document.getElementById("rangeMinWords");
  // Cắt câu theo ĐỘ ỔN ĐỊNH (stable_cut): text đứng im đủ lâu thì chốt câu.
  const chkStableCut = document.getElementById("chkStableCut");
  const rangeStableMs = document.getElementById("rangeStableMs");
  const valStableMs = document.getElementById("valStableMs");
  const rangeStableMinSec = document.getElementById("rangeStableMinSec");
  const valStableMinSec = document.getElementById("valStableMinSec");
  const rangeStableMinWords = document.getElementById("rangeStableMinWords");
  const valStableMinWords = document.getElementById("valStableMinWords");

  // Lookahead Video Buffering
  const chkEnableLookahead = document.getElementById("chkEnableLookahead");
  const lookaheadStatusEl = document.getElementById("lookaheadStatus");
  const lookaheadStatusText = document.getElementById("lookaheadStatusText");
  const rangeLookaheadSync = document.getElementById("rangeLookaheadSync");
  const valLookaheadSync = document.getElementById("valLookaheadSync");
  //: ── Tuỳ chỉnh CHỈ dành cho PIPELINE A ────────────────────────────────────────
  //: Pipeline B (Lookahead OFFLINE_BATCH) KHÔNG dùng VAD để cắt câu và KHÔNG dùng CommitManager:
  //: nó cắt câu bằng dấu câu của bản phiên âm + mốc từ của Qwen3-ForcedAligner. Vì vậy khi
  //: Lookahead đang bật, các tuỳ chỉnh dưới đây bị VÔ HIỆU (disabled + làm mờ + ghi chú) để người
  //: dùng không tưởng nhầm là chúng đang có tác dụng. Tắt Lookahead ⇒ mở lại bình thường.
  const pipelineAOnlyControls = [
    document.getElementById("selVadEngine"),
    document.getElementById("rangeVadSilence"),
    document.getElementById("rangeVadThreshold"),
    document.getElementById("rangeStableMinSec"),
    document.getElementById("rangeMinWords"),
    document.getElementById("chkStableCut"),
    document.getElementById("rangeStableMs"),
    document.getElementById("rangeStableMinWords"),
    document.getElementById("chkStableTrace"),
  ].filter(Boolean);
  const pipelineAOnlyGroups = [
    "groupVadEngine",
    "rowVadSliders",
    "groupStableMinSec",
    "groupMinWords",
    "sectionSegmentation",
    "groupStableCut",
    "groupStableMs",
    "groupStableMinWords",
    "groupStableTrace",
  ].map((id) => document.getElementById(id)).filter(Boolean);
  const PIPELINE_A_ONLY_HINT =
    "Chỉ áp dụng cho Pipeline A (Realtime Streaming). Tắt Lookahead Video Buffering để chỉnh.";
  //: Ngược lại với `pipelineAOnlyGroups`: các tuỳ chỉnh CHỈ dành cho Pipeline B (Lookahead).
  //: "Synchronize subtitles / dubbing" gửi `lookaheadSyncOffsetMs`, mà backend chỉ đọc nó trong
  //: `ws/lookahead_handler.py` (Pipeline A không dùng) ⇒ chạy Pipeline A thì slider vô tác dụng.
  const pipelineBOnlyGroups = ["lookaheadSyncGroup"]
    .map((id) => document.getElementById(id))
    .filter(Boolean);
  const PIPELINE_B_ONLY_HINT =
    "Chỉ áp dụng cho Pipeline B (Lookahead Video Buffering). Bật Lookahead để chỉnh.";
  //: Timer làm mới trạng thái Lookahead khi popup đang mở.
  let lookaheadStatusTimer = null;
  //: Trạng thái Pipeline và tính khả dụng Lookahead
  let currentActivePipeline = null; // "A" | "B" | null
  let isLookaheadUnavailable = false; // true khi video không hỗ trợ MSE / không khả dụng
  // Chẩn đoán: log mỗi nhịp preview ([SEG_TRACE]) — mặc định TẮT.
  const chkStableTrace = document.getElementById("chkStableTrace");
  //: Hàng bọc công tắc — bấm cả hàng cũng bật/tắt (xem `wireToggleRow`).
  const stableToggleRow = document.getElementById("stableToggleRow");
  const stableTraceToggleRow = document.getElementById("stableTraceToggleRow");
  const translationOnceToggleRow = document.getElementById("translationOnceToggleRow");
  const valMinWords = document.getElementById("valMinWords");
  const selSourceLang = document.getElementById("selSourceLang");
  const selTargetLang = document.getElementById("selTargetLang");
  const selTranslationModel = document.getElementById("selTranslationModel");
  const rangeSubPosY = document.getElementById("rangeSubPosY");
  const valSubPosY = document.getElementById("valSubPosY");
  const rangeSubWidth = document.getElementById("rangeSubWidth");
  const valSubWidth = document.getElementById("valSubWidth");
  const rangeOrigSize = document.getElementById("rangeOrigSize");
  const rangeTransSize = document.getElementById("rangeTransSize");
  const valOrigSize = document.getElementById("valOrigSize");
  const valTransSize = document.getElementById("valTransSize");
  const rangeFontWeight = document.getElementById("rangeFontWeight");
  const valFontWeight = document.getElementById("valFontWeight");
  const selFontFamily = document.getElementById("selFontFamily");
  const rangeMaxLines = document.getElementById("rangeMaxLines");
  const valMaxLines = document.getElementById("valMaxLines");
  // "Tắt chạy chữ" cho bản dịch (mặc định BẬT: hiện bản dịch 1 lần).
  const chkTranslationOnce = document.getElementById("chkTranslationOnce");
  const chkShowOriginal = document.getElementById("chkShowOriginal");
  const showOriginalToggleRow = document.getElementById("showOriginalToggleRow");

  const chkEnableTts = document.getElementById("chkEnableTts");
  const selTtsVoice = document.getElementById("selTtsVoice");
  const selTtsSpeed = document.getElementById("selTtsSpeed");
  const valTtsSpeed = document.getElementById("valTtsSpeed");
  const selTtsDucking = document.getElementById("selTtsDucking");
  const rangeDuckingLevel = document.getElementById("rangeDuckingLevel");
  const valDuckingLevel = document.getElementById("valDuckingLevel");
  const duckingSliderRow = document.getElementById("duckingSliderRow");

  let availableVoices = [];
  let savedPreferredVoiceId = null;
  let savedPreferredLang = null;

  let activeTab = null;
  let isCapturingNow = false;
  let currentAudioSampleRate = null;
  let isSwitchingEngine = false;
  let isSwitchingTranslationModel = false;
  let isMonitoringAsr = false;
  let isMonitoringTranslation = false;
  let currentModelError = null;
  let lastActiveAsr = "sensevoice";
  let lastActiveVad = "auto";
  let lastActiveLang = "auto";

  function updateStartButtonState() {
    if (isCapturingNow) {
      btnStart.disabled = true;
      btnStop.disabled = false;
      return;
    }
    btnStop.disabled = true;
    if (isSwitchingEngine || isSwitchingTranslationModel) {
      btnStart.disabled = true;
      return;
    }
    if (currentModelError) {
      btnStart.disabled = true;
      return;
    }
    btnStart.disabled = false;
  }


  function formatSampleRate(rate) {
    const numRate = Number(rate);
    if (!numRate || isNaN(numRate) || numRate <= 0) return "";
    const khz = +(numRate / 1000).toFixed(1);
    return `${khz} kHz`;
  }

  function getCapturingLabel(rate = currentAudioSampleRate) {
    const rateStr = formatSampleRate(rate);
    return rateStr ? `Capturing ${rateStr}` : "Capturing";
  }

  function renderVoiceOptions(voicesList, currentVoiceId) {
    if (!selTtsVoice || !voicesList || voicesList.length === 0) return;
    const prev = currentVoiceId || savedPreferredVoiceId || selTtsVoice.value;
    selTtsVoice.textContent = "";

    voicesList.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.id;
      opt.textContent = item.name || item.id;
      selTtsVoice.appendChild(opt);
    });

    if (prev && voicesList.some((item) => item.id === prev)) {
      selTtsVoice.value = prev;
    } else {
      selTtsVoice.value = voicesList[0].id;
    }
  }

  function getSettings() {
    const isTts = chkEnableTts ? chkEnableTts.checked : false;
    const selectedVoiceId = selTtsVoice ? selTtsVoice.value : "audio.wav";
    const voiceObj = availableVoices.find(v => v.id === selectedVoiceId) || availableVoices[0] || null;
    const refAudioPath = voiceObj ? voiceObj.audio : ("backend/voices/" + selectedVoiceId);
    const refAudioText = voiceObj ? (voiceObj.text || "") : "";

    const duckingPercent = parseInt(rangeDuckingLevel ? rangeDuckingLevel.value : 25, 10);
    const speedVal = selTtsSpeed ? (selTtsSpeed.value || "1.0") : "1.0";
    const rawSilence = rangeVadSilence ? parseInt(rangeVadSilence.value, 10) : 0;
    // 0 = OFF: backend hiểu là "để chính VAD dùng mặc định trong docs của nó".
    const vadSilenceMs = isNaN(rawSilence) ? 0 : Math.max(0, rawSilence);
    const rawThresh = rangeVadThreshold ? parseFloat(rangeVadThreshold.value) : 0.5;
    const vadThresholdVal = isNaN(rawThresh) ? 0.5 : rawThresh;
    const rawMinWords = rangeMinWords ? parseInt(rangeMinWords.value, 10) : 2;
    const minWords = isNaN(rawMinWords) ? 2 : Math.max(0, rawMinWords);
    const rawStableMs = rangeStableMs ? parseFloat(rangeStableMs.value) : 0;
    const stableMs = isNaN(rawStableMs) ? 0 : Math.max(0, rawStableMs);
    const cfg = {
      asrEngine: selAsrEngine ? selAsrEngine.value : undefined,
      vadEngine: selVadEngine ? selVadEngine.value : undefined,
      vadSilenceDurationMs: vadSilenceMs,
      silenceDurationMs: vadSilenceMs,
      silence_duration_ms: vadSilenceMs,
      vadThreshold: vadThresholdVal,
      vad_threshold: vadThresholdVal,
      threshold: vadThresholdVal,
      minWordsToCommit: minWords,
      min_words_to_commit: minWords,
      // Cắt câu theo ĐỘ ỔN ĐỊNH (stable_cut): text đứng im đủ lâu ⇒ chốt câu.
      splitOnStability: chkStableCut ? chkStableCut.checked : true,
      stabilityDurationSec: stableMs / 1000,
      stabilityMinDurationSec: rangeStableMinSec ? parseFloat(rangeStableMinSec.value) : 2.5,
      stabilityMinWords: rangeStableMinWords ? parseInt(rangeStableMinWords.value, 10) : 4,
      stability_duration_ms: stableMs,
      stability_min_duration_sec: rangeStableMinSec ? parseFloat(rangeStableMinSec.value) : 2.5,
      stability_min_words: rangeStableMinWords ? parseInt(rangeStableMinWords.value, 10) : 4,
      split_on_stability: chkStableCut ? chkStableCut.checked : true,
      traceStability: chkStableTrace ? chkStableTrace.checked : false,
      trace_stability: chkStableTrace ? chkStableTrace.checked : false,
      sourceLanguage: selSourceLang ? selSourceLang.value : "auto",
      targetLang: selTargetLang ? selTargetLang.value : "vi",
      translationModel: selTranslationModel ? selTranslationModel.value : "xiaomi",
      subPosY: parseInt(rangeSubPosY ? rangeSubPosY.value : 10, 10) || 10,
      subWidth: parseInt(rangeSubWidth ? rangeSubWidth.value : 80, 10) || 80,
      origFontSize: parseFloat(rangeOrigSize ? rangeOrigSize.value : 2.5) || 2.5,
      transFontSize: parseFloat(rangeTransSize ? rangeTransSize.value : 4.2) || 4.2,
      fontWeight: parseInt(rangeFontWeight ? rangeFontWeight.value : 600, 10) || 600,
      fontFamily: selFontFamily ? selFontFamily.value : "default",
      maxLines: parseInt(rangeMaxLines ? rangeMaxLines.value : 3, 10) || 3,
      // Lookahead Video Buffering (Pipeline B vs Pipeline A)
      lookaheadEnabled: chkEnableLookahead ? chkEnableLookahead.checked : true,
      //: ⚠️ ĐÃ BỎ "Thời gian dịch trước" (lead time) khỏi popup: ở tuyến OFFLINE_BATCH khoảng dịch
      //: trước do BỘ CẮT KHỐI quyết định (12–30 s, xem `LookaheadChunker`), nên slider 10–15 s chỉ
      //: còn tác dụng throttle nội bộ ⇒ gây hiểu nhầm. Backend dùng mặc định `lookahead.lead_time_sec`.
      lookaheadSyncOffsetMs: rangeLookaheadSync ? parseInt(rangeLookaheadSync.value, 10) : 0,
      // "Tắt chạy chữ": chỉ hiện bản dịch MỘT LẦN khi có bản dịch hoàn chỉnh.
      showTranslationOnce: chkTranslationOnce ? chkTranslationOnce.checked : true,
      // On/Off phụ đề gốc (mặc định BẬT; tắt thì chỉ hiện bản dịch, phù hợp pipeline B).
      showOriginalSubtitles: chkShowOriginal ? chkShowOriginal.checked : true,
      ttsEnabled: isTts,
      ttsVoice: selectedVoiceId,
      ttsInstruct: "",
      ttsRefAudio: refAudioPath,
      ttsRefText: refAudioText,
      ttsSpeed: speedVal,
      ttsDucking: selTtsDucking ? (selTtsDucking.value === "true") : true,
      duckingLevel: isNaN(duckingPercent) ? 0.25 : duckingPercent / 100,
    };
    return cfg;
  }

  function normalizeFontPercent(val, isTrans = false) {
    if (val === undefined || val === null) return isTrans ? 4.2 : 2.5;
    const num = parseFloat(val);
    if (isNaN(num)) return isTrans ? 4.2 : 2.5;
    // If value is >= 8, it was previously saved in pixels (e.g. 10 - 36 px)
    if (num >= 8) {
      const converted = (num / 480) * 100;
      const clamped = isTrans ? Math.min(8.0, Math.max(2.0, converted)) : Math.min(5.0, Math.max(1.0, converted));
      return Math.round(clamped * 10) / 10;
    }
    return isTrans ? Math.min(8.0, Math.max(2.0, num)) : Math.min(5.0, Math.max(1.0, num));
  }

  function updateRangeLabels() {
    if (valVadSilence && rangeVadSilence) {
      const ms = parseInt(rangeVadSilence.value, 10);
      const off = !ms || ms <= 0;
      valVadSilence.textContent = off ? "0" : String(ms);
      const unit = document.getElementById("valVadSilenceUnit");
      if (unit) unit.textContent = off ? " (OFF)" : "ms";
    }
    if (valVadThreshold && rangeVadThreshold) valVadThreshold.textContent = parseFloat(rangeVadThreshold.value).toFixed(2);
    if (valMinWords && rangeMinWords) {
      const mw = parseInt(rangeMinWords.value, 10);
      valMinWords.textContent = isNaN(mw) ? "2" : mw;
    }
    if (valSubPosY && rangeSubPosY) valSubPosY.textContent = rangeSubPosY.value;
    if (valSubWidth && rangeSubWidth) valSubWidth.textContent = rangeSubWidth.value;
    if (valOrigSize && rangeOrigSize) {
      const v = parseFloat(rangeOrigSize.value);
      valOrigSize.textContent = isNaN(v) ? "2.5" : v.toFixed(1);
    }
    if (valTransSize && rangeTransSize) {
      const v = parseFloat(rangeTransSize.value);
      valTransSize.textContent = isNaN(v) ? "4.2" : v.toFixed(1);
    }
    if (valFontWeight && rangeFontWeight) valFontWeight.textContent = rangeFontWeight.value;
    if (valMaxLines && rangeMaxLines) valMaxLines.textContent = rangeMaxLines.value;
    if (valTtsSpeed && selTtsSpeed) valTtsSpeed.textContent = selTtsSpeed.value || "1.0";
    if (valDuckingLevel && rangeDuckingLevel) valDuckingLevel.textContent = rangeDuckingLevel.value;
    // stable_cut
    if (valStableMs && rangeStableMs) valStableMs.textContent = rangeStableMs.value;
    if (valStableMinSec && rangeStableMinSec) valStableMinSec.textContent = rangeStableMinSec.value;
    if (valStableMinWords && rangeStableMinWords) valStableMinWords.textContent = rangeStableMinWords.value;
    // Lookahead Video Buffering
    if (valLookaheadSync && rangeLookaheadSync) valLookaheadSync.textContent = rangeLookaheadSync.value;
    // Vô hiệu/mở lại nhóm tuỳ chỉnh CHỈ dành cho Pipeline A theo trạng thái Lookahead.
    updatePipelineAOnlyUi();
    updateDuckingUi();
  }

  /**
   * Hàng "Original audio %" CHỈ có nghĩa khi Auto-Ducking BẬT.
   *
   * VÌ SAO CẦN ẨN: `rangeDuckingLevel` điều khiển mức âm lượng tiếng gốc khi lồng tiếng. Nếu
   * người dùng đặt Auto-Ducking = Off thì slider vẫn kéo được nhưng KHÔNG có tác dụng nào —
   * đúng loại "điều khiển ma" mà ghi chú ở `pipelineAOnlyGroups` muốn tránh.
   */
  function updateDuckingUi() {
    if (!duckingSliderRow) return;
    const duckingOn = !selTtsDucking || selTtsDucking.value === "true";
    duckingSliderRow.style.display = duckingOn ? "" : "none";
  }

  /**
   * Bấm vào CẢ HÀNG để bật/tắt công tắc của hàng đó.
   *
   * Bấm trực tiếp vào `.switch` thì nhường cho chính công tắc xử lý (nếu không sẽ lật 2 lần).
   * Dùng chung cho mọi `.toggle-row` để 5 hàng có cùng một hành vi.
   */
  function wireToggleRow(row, checkbox) {
    if (!row || !checkbox) return;
    row.addEventListener("click", (e) => {
      if (e.target.closest(".switch")) return;
      checkbox.checked = !checkbox.checked;
      checkbox.dispatchEvent(new Event("change"));
    });
  }

  /**
   * Xác định xem hệ thống đang chạy hoặc dự kiến chạy Pipeline B (Lookahead) hay không.
   * - Khi đang trong phiên (isCapturingNow): dựa vào pipeline thực tế đang chạy (`currentActivePipeline === "B"`).
   * - Khi chưa chạy: dựa vào công tắc Lookahead và tính khả dụng của video hiện tại.
   *   Nếu Lookahead tắt HOẶC video trên trang báo không khả dụng => sẽ chạy Pipeline A => không khoá VAD/SEG.
   */
  function isPipelineBLocked() {
    if (isCapturingNow) {
      return currentActivePipeline === "B";
    }
    const lookaheadOn = !chkEnableLookahead || chkEnableLookahead.checked !== false;
    if (!lookaheadOn) return false;
    if (isLookaheadUnavailable) return false;
    return true;
  }

  /**
   * Vô hiệu hoá nhóm tuỳ chỉnh CHỈ dành cho PIPELINE A khi đang chạy hoặc dự kiến chạy Pipeline B.
   * Nếu là Pipeline A: không khoá chỉnh VAD, SEG (cho phép live adjust sliders trong lúc streaming).
   */
  function updatePipelineAOnlyUi() {
    const isPipelineB = isPipelineBLocked();
    const editable = !isPipelineB;
    pipelineAOnlyControls.forEach((el) => {
      if (el === selVadEngine) {
        el.disabled = !editable || isCapturingNow || isSwitchingEngine;
      } else {
        el.disabled = !editable;
      }
      el.title = isPipelineB ? PIPELINE_A_ONLY_HINT : "";
    });
    pipelineAOnlyGroups.forEach((el) => {
      el.style.opacity = editable ? "1" : "0.45";
      el.title = isPipelineB ? PIPELINE_A_ONLY_HINT : "";
    });
    pipelineBOnlyGroups.forEach((el) => {
      el.style.opacity = isPipelineB ? "1" : "0.45";
      el.title = isPipelineB ? "" : PIPELINE_B_ONLY_HINT;
    });
    ["pipelineAOnlyNote", "pipelineAOnlyNote2"].forEach((id) => {
      const note = document.getElementById(id);
      if (note) note.style.display = isPipelineB ? "block" : "none";
    });
  }

  async function fetchActiveTab() {
    if (activeTab) return activeTab;
    try {
      let [tab] = await api.tabs.query({ active: true, lastFocusedWindow: true });
      if (!tab) [tab] = await api.tabs.query({ active: true, currentWindow: true });
      if (!tab) {
        const tabs = await api.tabs.query({ active: true });
        if (tabs && tabs.length > 0) tab = tabs[0];
      }
      activeTab = tab || null;
    } catch (e) {
      console.warn("[Popup] Tab query error:", e);
    }
    return activeTab;
  }

  function renderAsrEngineOptions(availableModels, currentActiveId) {
    if (!selAsrEngine || !availableModels || availableModels.length === 0) return;
    const previousSelection = currentActiveId || selAsrEngine.value;
    selAsrEngine.textContent = "";

    availableModels.forEach((item) => {
      const opt = document.createElement("option");
      const id = typeof item === "string" ? item : item.id;
      let name = typeof item === "string" ? item : (item.name || item.id);
      if (typeof item === "object" && item.is_downloaded === false) {
        name += " ⤓ chưa tải";
      } else if (typeof item === "object" && item.is_downloaded) {
        name += " ⚡";
      }
      opt.value = id;
      opt.textContent = name;
      if (typeof item === "object" && item.description) {
        opt.title = item.description;
      }
      selAsrEngine.appendChild(opt);
    });

    if (availableModels.some((item) => (typeof item === "string" ? item : item.id) === previousSelection)) {
      selAsrEngine.value = previousSelection;
    } else {
      selAsrEngine.value = typeof availableModels[0] === "string" ? availableModels[0] : availableModels[0].id;
    }
  }

  function renderTranslationModelOptions(availableModels, currentActiveId) {
    if (!selTranslationModel || !availableModels || !Array.isArray(availableModels) || availableModels.length === 0) return;
    const previousSelection = currentActiveId || selTranslationModel.value;
    selTranslationModel.textContent = "";

    availableModels.forEach((item) => {
      const opt = document.createElement("option");
      const id = typeof item === "string" ? item : item.id;
      let name = typeof item === "string" ? item : (item.name || item.id);
      if (typeof item === "object" && item.is_downloaded === false) {
        name += " ⤓ chưa tải";
      } else if (typeof item === "object" && item.is_downloaded) {
        name += " ⚡";
      }
      opt.value = id;
      opt.textContent = name;
      if (typeof item === "object" && (item.desc || item.description)) {
        opt.title = item.desc || item.description;
      }
      selTranslationModel.appendChild(opt);
    });

    if (availableModels.some((item) => (typeof item === "string" ? item : item.id) === previousSelection)) {
      selTranslationModel.value = previousSelection;
    } else {
      selTranslationModel.value = typeof availableModels[0] === "string" ? availableModels[0] : availableModels[0].id;
    }
  }

  function renderLanguageOptions(supportedLanguages, currentLangCode) {
    if (!selSourceLang || !supportedLanguages || supportedLanguages.length === 0) return;
    const targetSelection = currentLangCode || savedPreferredLang || selSourceLang.value || "auto";
    selSourceLang.textContent = "";

    supportedLanguages.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.code;
      opt.textContent = item.name;
      selSourceLang.appendChild(opt);
    });

    if (supportedLanguages.some((item) => item.code === targetSelection)) {
      selSourceLang.value = targetSelection;
    } else {
      selSourceLang.value = supportedLanguages[0].code || "auto";
    }
  }

  let cachedBackendScheme = (typeof sessionStorage !== "undefined" && sessionStorage.getItem("bs_preferred_scheme")) || "https";

  async function fetchBackend(path, options = {}) {
    const timeoutMs = options.timeoutMs || options.timeout || (
      options.method === "POST" ? 60000 : 8000
    );
    const { timeout, timeoutMs: _, ...fetchOpts } = options;
    const schemes = cachedBackendScheme === "http" ? ["http", "https"] : ["https", "http"];
    let lastRes = null;

    for (const scheme of schemes) {
      try {
        const url = `${scheme}://localhost:8765${path}`;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        const res = await fetch(url, { ...fetchOpts, signal: controller.signal });
        clearTimeout(timer);
        if (res) {
          if (res.ok && cachedBackendScheme !== scheme) {
            cachedBackendScheme = scheme;
            if (typeof sessionStorage !== "undefined") {
              sessionStorage.setItem("bs_preferred_scheme", scheme);
            }
          }
          return res;
        }
      } catch (e) {
        // Only retry fallback scheme on network-level failure
      }
    }
    return lastRes;
  }

  async function fetchBackendEngineConfig() {
    let data = null;
    try {
      const res = await fetchBackend("/api/config");
      if (res && res.ok) {
        data = await res.json();
      }
    } catch (e) {
      console.warn("[Popup] Could not fetch backend config:", e);
    }

    if (!data || data.status !== "ok") {
      statusBadge.textContent = "Server Offline";
      statusBadge.className = "badge badge-disconnected";
      showMsg("⚠️ Không kết nối được Backend! Nếu đã chạy 'python main.py', bấm vào đây để mở https://localhost:8765 và chọn 'Nâng cao -> Tiếp tục' để chấp nhận chứng chỉ SSL.", "error");
      if (messageArea) {
        messageArea.style.cursor = "pointer";
        messageArea.onclick = () => {
          api.tabs.create({ url: "https://localhost:8765" });
        };
      }
      btnStart.disabled = true;
      return false;
    } else {
      if (messageArea) {
        messageArea.style.cursor = "default";
        messageArea.onclick = null;
      }
    }

    // Sync ASR & VAD engines from backend
    const activeAsr = data.active_model || data.asr_engine || data.engine || "sensevoice";
    const activeVad = data.vad_engine || "firered-vad";
    lastActiveAsr = activeAsr;
    lastActiveVad = activeVad;

    // Dynamically populate available ASR engines from backend catalog
    if (data.available_models || data.available_asr_engines) {
      renderAsrEngineOptions(data.available_models || data.available_asr_engines, activeAsr);
    } else if (selAsrEngine) {
      selAsrEngine.value = activeAsr;
    }

    if (selVadEngine) selVadEngine.value = activeVad;
    if (rangeVadSilence && data.silence_duration_ms !== undefined && !rangeVadSilence.dataset.userEdited) {
      // Backend trả 0 khi "để VAD tự quyết định" ⇒ hiển thị đúng 0 (OFF).
      rangeVadSilence.value = data.silence_duration_ms;
      updateRangeLabels();
    }
    if (rangeVadThreshold && data.vad_threshold !== undefined && !rangeVadThreshold.dataset.userEdited) {
      rangeVadThreshold.value = data.vad_threshold;
      updateRangeLabels();
    }
    if (rangeMinWords && data.min_words_to_commit !== undefined && !rangeMinWords.dataset.userEdited) {
      rangeMinWords.value = data.min_words_to_commit;
      updateRangeLabels();
    }
    // stable_cut: backend là nguồn sự thật (trừ khi người dùng vừa kéo tay).
    if (data.stable) {
      if (chkStableCut && data.stable.split_on_stability !== undefined) {
        chkStableCut.checked = !!data.stable.split_on_stability;
      }
      if (rangeStableMs && data.stable.duration_ms !== undefined && !rangeStableMs.dataset.userEdited) {
        rangeStableMs.value = Math.round(data.stable.duration_ms);
      }
      if (rangeStableMinSec && data.stable.min_duration_sec !== undefined && !rangeStableMinSec.dataset.userEdited) {
        rangeStableMinSec.value = data.stable.min_duration_sec;
      }
      if (rangeStableMinWords && data.stable.min_words !== undefined && !rangeStableMinWords.dataset.userEdited) {
        rangeStableMinWords.value = data.stable.min_words;
      }
      if (chkStableTrace && data.stable.trace !== undefined) {
        chkStableTrace.checked = !!data.stable.trace;
      }
      updateRangeLabels();
    }

    // Populate source languages
    if (data.supported_languages) {
      renderLanguageOptions(data.supported_languages, savedPreferredLang || selSourceLang.value);
    }

    // Populate voice clone profiles
    if (data.tts && data.tts.voices && data.tts.voices.length > 0) {
      availableVoices = data.tts.voices;
      renderVoiceOptions(availableVoices, selTtsVoice ? selTtsVoice.value : null);
    }

    // Sync active translation model from backend & populate dynamic options
    if (data.translation) {
      const activeTrans = data.translation.base || data.translation.translation_model;
      if (data.translation.available_models && data.translation.available_models.length > 0) {
        renderTranslationModelOptions(data.translation.available_models, activeTrans);
      } else if (activeTrans && selTranslationModel) {
        selTranslationModel.value = activeTrans;
      }
    }

    // Backend có thể đang tải/nạp model Dịch ở nền ⇒ khoá ngay Start và theo dõi
    const dl = data.translation ? data.translation.download : null;
    const isDlActive = dl && dl.model && (dl.state === "downloading" || dl.state === "loading");
    if (isDlActive) {
      const verb = dl.state === "downloading" ? "tải" : "nạp";
      const note = dl.state === "downloading" ? ` (${describeDownloadProgress(dl)})` : "";
      statusBadge.textContent = `${verb === "tải" ? "Tải" : "Nạp"} dịch ${note}`.trim();
      statusBadge.className = "badge badge-reconnecting";
      showMsg(`⏳ Backend đang ${verb} model dịch '${dl.model}'${note}. Nút Bắt đầu bị khoá cho đến khi model sẵn sàng.`, "info");
      monitorTranslationDownload(dl.model, dl.model);
    } else if (dl && dl.state === "error" && dl.error) {
      currentModelError = dl.error;
      statusBadge.textContent = "LỖI DỊCH";
      statusBadge.className = "badge badge-disconnected";
      showMsg(`❌ Model dịch '${dl.model || "?"}' lỗi: ${dl.error}. Vui lòng chọn model khác!`, "error");
    } else if (dl && dl.state === "ready" && currentModelError) {
      currentModelError = null;
    }

    // Backend có thể đang tải/nạp model ASR ở nền ⇒ khoá ngay Start và theo dõi
    const asrDl = data.asr_download || (data.asr && data.asr.download);
    const isAsrDlActive = asrDl && asrDl.model && (asrDl.state === "downloading" || asrDl.state === "loading");
    if (isAsrDlActive) {
      const verb = asrDl.state === "downloading" ? "tải" : "nạp";
      const note = asrDl.state === "downloading" ? ` (${describeDownloadProgress(asrDl)})` : "";
      statusBadge.textContent = `${verb === "tải" ? "Tải" : "Nạp"} ASR ${note}`.trim();
      statusBadge.className = "badge badge-reconnecting";
      showMsg(`⏳ Backend đang ${verb} model ASR '${asrDl.model}'${note}. Nút Bắt đầu bị khoá cho đến khi model sẵn sàng.`, "info");
      monitorAsrDownload(asrDl.model, asrDl.model);
    } else if (asrDl && asrDl.state === "error" && asrDl.error) {
      currentModelError = asrDl.error;
      statusBadge.textContent = "LỖI ASR";
      statusBadge.className = "badge badge-disconnected";
      showMsg(`❌ Model ASR '${asrDl.model || "?"}' lỗi: ${asrDl.error}. Vui lòng chọn model khác!`, "error");
    } else if (asrDl && asrDl.state === "ready" && currentModelError) {
      currentModelError = null;
    }

    if (!isCapturingNow) {
      if (!isDlActive && !isAsrDlActive && !currentModelError && !isSwitchingEngine && !isSwitchingTranslationModel) {
        statusBadge.textContent = `${activeAsr.toUpperCase()}`;
        statusBadge.className = "badge badge-ready";
        showMsg(`✅ Server Online (ASR: ${activeAsr.toUpperCase()} | VAD: ${data.resolved_vad || activeVad})`, "success");
      }
      updateStartButtonState();
    }

    return true;
  }

  // ── Hot-swap ASR / VAD Engine on Backend ───────────────────
  async function handleEngineSwitch(force = false) {
    if (isCapturingNow) {
      showMsg("⚠️ Không thể đổi model khi session đang chạy! Hãy dừng session trước.", "warning");
      return;
    }
    if (isSwitchingEngine) return;

    const newAsr = selAsrEngine ? selAsrEngine.value : lastActiveAsr;
    const newVad = selVadEngine ? selVadEngine.value : lastActiveVad;
    const newLang = selSourceLang ? selSourceLang.value : "auto";

    if (!force && newAsr === lastActiveAsr && newVad === lastActiveVad && (newAsr !== "whisper" || newLang === lastActiveLang)) return;

    isSwitchingEngine = true;
    currentModelError = null;
    updateStartButtonState(); // Vô hiệu hoá ngay nút Start
    if (selAsrEngine) selAsrEngine.disabled = true;
    if (selVadEngine) selVadEngine.disabled = true;

    const labelDesc = newAsr === "whisper" ? `Whisper (${newLang})` : newAsr.toUpperCase();
    statusBadge.textContent = `Nạp ${labelDesc}...`;
    statusBadge.className = "badge badge-reconnecting";
    showMsg(`⏳ Đang kiểm tra, tải & giải phóng VRAM để nạp model ASR ${labelDesc}... Vui lòng đợi.`, "info");

    try {
      const payload = {
        asr_engine: newAsr,
        vad_engine: newVad,
        silence_duration_ms: (() => {
          const raw = parseInt(rangeVadSilence ? rangeVadSilence.value : 0, 10);
          return isNaN(raw) ? 0 : Math.max(0, raw);   // 0 = "để VAD tự quyết định"
        })(),
        vad_threshold: !isNaN(parseFloat(rangeVadThreshold?.value)) ? parseFloat(rangeVadThreshold.value) : 0.5,
        min_words_to_commit: !isNaN(parseInt(rangeMinWords?.value, 10)) ? Math.max(0, parseInt(rangeMinWords.value, 10)) : 2,
        split_on_stability: chkStableCut ? chkStableCut.checked : true,
        stability_duration_ms: !isNaN(parseFloat(rangeStableMs?.value)) ? Math.max(0, parseFloat(rangeStableMs.value)) : 0,
        stability_min_duration_sec: !isNaN(parseFloat(rangeStableMinSec?.value)) ? Math.max(0, parseFloat(rangeStableMinSec.value)) : 2.5,
        stability_min_words: !isNaN(parseInt(rangeStableMinWords?.value, 10)) ? Math.max(0, parseInt(rangeStableMinWords.value, 10)) : 4,
        trace_stability: chkStableTrace ? chkStableTrace.checked : false,
        source_lang: newLang,
        tts_enabled: chkEnableTts ? chkEnableTts.checked : false,
        tts_voice: selTtsVoice ? selTtsVoice.value : undefined,
        tts_speed: selTtsSpeed ? parseFloat(selTtsSpeed.value || 1.0) : 1.0,
      };

      const res = await fetchBackend("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        timeout: 180000,
      });

      if (res && res.status === 202) {
        // Backend chưa có file GGUF ⇒ trả 202 và tải trong nền
        const body = await res.json().catch(() => ({}));
        showMsg(`⏳ ${body.detail || `Đang tải model ASR ${labelDesc} về máy (chạy nền). Nút Bắt đầu đang được khoá.`}`, "info");
        await waitForAsrActivation(newAsr, labelDesc);
      } else if (!res || !res.ok) {
        const errorDetail = res ? ((await res.json().catch(() => null))?.detail || await res.text()) : "Network error";
        throw new Error(errorDetail);
      }

      const result = res.status === 202 ? (await (await fetchBackend("/api/config", { timeout: 15000 })).json().catch(() => ({}))) : (await res.json());
      lastActiveAsr = result.engine || newAsr;
      lastActiveVad = result.vad_engine || newVad;
      lastActiveLang = result.source_lang || newLang;

      // Update supported languages dropdown
      if (result.supported_languages) {
        renderLanguageOptions(result.supported_languages, savedPreferredLang || result.source_lang || selSourceLang.value);
      }

      // Save settings
      const cfg = getSettings();
      api.storage.local.set({ bs_settings: cfg });

      currentModelError = null;
      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`✅ Đã nạp thành công ASR: ${lastActiveAsr.toUpperCase()} (${lastActiveLang}) | VAD: ${result.resolved_vad || lastActiveVad}`, "success");
    } catch (err) {
      console.error("[Popup] Engine switch error:", err);
      currentModelError = err.message || "Không thể chuyển engine";
      statusBadge.textContent = "LỖI ASR";
      statusBadge.className = "badge badge-disconnected";
      showMsg(`❌ Lỗi nạp model ASR: ${currentModelError}. Vui lòng chọn model khác!`, "error");
    } finally {
      isSwitchingEngine = false;
      if (selAsrEngine) selAsrEngine.disabled = isCapturingNow;
      if (selVadEngine) selVadEngine.disabled = isCapturingNow;
      updateStartButtonState();
    }
  }

  // ── Hot-swap Translation Model on Backend ─────────────────
  const MODEL_DOWNLOAD_POLL_MS = 4000;
  const MODEL_DOWNLOAD_TIMEOUT_MS = 30 * 60 * 1000;


  function describeDownloadProgress(dl) {
    if (dl.percent != null) return `${Math.round(dl.percent)}%`;
    if (dl.downloaded_mb != null) return `${dl.downloaded_mb} MB`;
    return "đang tải";
  }

  // Chờ backend tải (nếu thiếu file) + nạp model ASR. Backend trả HTTP 202 và làm việc
  // trong nền, nên popup hỏi tiến độ qua /api/config cho tới khi ready/error.
  async function waitForAsrActivation(modelId, shortDesc) {
    const deadline = Date.now() + MODEL_DOWNLOAD_TIMEOUT_MS;
    let lastNote = "";

    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, MODEL_DOWNLOAD_POLL_MS));

      let data = null;
      try {
        const res = await fetchBackend("/api/config", { timeout: 15000 });
        if (!res || !res.ok) continue;
        data = await res.json();
      } catch (e) {
        continue;
      }

      const asr = data.asr || {};
      const dl = data.asr_download || asr.download || {};
      if (dl.model && dl.model !== modelId) continue; // lượt tải của model khác

      if (dl.state === "downloading") {
        const note = describeDownloadProgress(dl);
        if (note !== lastNote) {
          lastNote = note;
          if (statusBadge) statusBadge.textContent = `Tải ${shortDesc} ${note}`;
          showMsg(`⏳ Đang tải model ASR ${shortDesc} về máy: ${note}. Nút Bắt đầu đang được khoá.`, "info");
        }
        continue;
      }
      if (dl.state === "loading") {
        if (statusBadge) statusBadge.textContent = `Nạp ${shortDesc}...`;
        if (lastNote !== "loading") {
          lastNote = "loading";
          showMsg(`⏳ Đã tải xong ${shortDesc}, đang nạp vào GPU... Nút Bắt đầu đang được khoá.`, "info");
        }
        continue;
      }
      if (dl.state === "error") {
        throw new Error(dl.error || `Tải/nạp model ASR ${shortDesc} thất bại`);
      }
      if (dl.state === "ready" || data.engine === modelId || asr.active_model === modelId) {
        return true;
      }
    }
    throw new Error("Hết thời gian chờ tải model ASR (30 phút). Kiểm tra mạng rồi thử lại.");
  }

  // Chờ backend tải (nếu thiếu file) + nạp model dịch. Backend trả HTTP 202 và làm việc
  // trong nền, nên popup hỏi tiến độ qua /api/config cho tới khi ready/error.
  async function waitForTranslationActivation(modelId, shortDesc) {
    const deadline = Date.now() + MODEL_DOWNLOAD_TIMEOUT_MS;
    let lastNote = "";

    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, MODEL_DOWNLOAD_POLL_MS));

      let data = null;
      try {
        const res = await fetchBackend("/api/config", { timeout: 15000 });
        if (!res || !res.ok) continue;
        data = await res.json();
      } catch (e) {
        continue;
      }

      const tr = data.translation || {};
      const dl = tr.download || {};
      if (dl.model && dl.model !== modelId) continue; // lượt tải của model khác

      if (dl.state === "downloading") {
        const note = describeDownloadProgress(dl);
        if (note !== lastNote) {
          lastNote = note;
          if (statusBadge) statusBadge.textContent = `Tải ${shortDesc} ${note}`;
          showMsg(`⏳ Đang tải model dịch ${shortDesc} về máy: ${note}. Nút Bắt đầu đang được khoá.`, "info");
        }
        continue;
      }
      if (dl.state === "loading") {
        if (statusBadge) statusBadge.textContent = `Nạp ${shortDesc}...`;
        if (lastNote !== "loading") {
          lastNote = "loading";
          showMsg(`⏳ Đã tải xong ${shortDesc}, đang nạp vào GPU... Nút Bắt đầu đang được khoá.`, "info");
        }
        continue;
      }
      if (dl.state === "error") {
        throw new Error(dl.error || `Tải/nạp model dịch ${shortDesc} thất bại`);
      }
      if (dl.state === "ready" || tr.base === modelId) {
        return true;
      }
    }
    throw new Error("Hết thời gian chờ tải model dịch (30 phút). Kiểm tra mạng rồi thử lại.");
  }

  async function monitorTranslationDownload(modelId, shortDesc) {
    if (isMonitoringTranslation || isSwitchingTranslationModel) return;
    isMonitoringTranslation = true;
    isSwitchingTranslationModel = true;
    if (selTranslationModel) selTranslationModel.disabled = true;
    updateStartButtonState();
    try {
      await waitForTranslationActivation(modelId, shortDesc);
      currentModelError = null;
      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`✅ Đã nạp thành công model dịch: ${shortDesc}`, "success");
    } catch (err) {
      console.error("[Popup] Translation activation error:", err);
      currentModelError = err.message || "Không thể nạp model dịch";
      statusBadge.textContent = "LỖI DỊCH";
      statusBadge.className = "badge badge-disconnected";
      showMsg(`❌ Lỗi nạp model dịch: ${currentModelError}. Vui lòng chọn model khác!`, "error");
    } finally {
      isMonitoringTranslation = false;
      isSwitchingTranslationModel = false;
      if (selTranslationModel) selTranslationModel.disabled = isCapturingNow;
      updateStartButtonState();
    }
  }

  async function monitorAsrDownload(modelId, shortDesc) {
    if (isMonitoringAsr || isSwitchingEngine) return;
    isMonitoringAsr = true;
    isSwitchingEngine = true;
    if (selAsrEngine) selAsrEngine.disabled = true;
    if (selVadEngine) selVadEngine.disabled = true;
    updateStartButtonState();
    try {
      await waitForAsrActivation(modelId, shortDesc);
      currentModelError = null;
      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`✅ Đã nạp thành công model ASR: ${shortDesc}`, "success");
    } catch (err) {
      console.error("[Popup] ASR activation error:", err);
      currentModelError = err.message || "Không thể nạp model ASR";
      statusBadge.textContent = "LỖI ASR";
      statusBadge.className = "badge badge-disconnected";
      showMsg(`❌ Lỗi nạp model ASR: ${currentModelError}. Vui lòng chọn model khác!`, "error");
    } finally {
      isMonitoringAsr = false;
      isSwitchingEngine = false;
      if (selAsrEngine) selAsrEngine.disabled = isCapturingNow;
      if (selVadEngine) selVadEngine.disabled = isCapturingNow;
      updateStartButtonState();
    }
  }

  async function handleTranslationModelSwitch() {
    if (isCapturingNow) {
      showMsg("⚠️ Không thể đổi model khi session đang chạy! Hãy dừng session trước.", "warning");
      return;
    }
    if (isSwitchingTranslationModel) return;

    const newModel = selTranslationModel ? selTranslationModel.value : "index-translate-2b";
    const selectedOption = selTranslationModel && selTranslationModel.selectedOptions ? selTranslationModel.selectedOptions[0] : null;
    const modelDesc = selectedOption ? selectedOption.textContent.replace("⚡", "").trim() : newModel;
    const shortDesc = modelDesc.split("(")[0].trim() || newModel;

    isSwitchingTranslationModel = true;
    currentModelError = null;
    updateStartButtonState(); // Vô hiệu hoá ngay nút Start
    if (selTranslationModel) selTranslationModel.disabled = true;

    statusBadge.textContent = `Nạp ${shortDesc}...`;
    statusBadge.className = "badge badge-reconnecting";
    showMsg(`⏳ Đang kiểm tra, tải & giải phóng VRAM để nạp model dịch ${modelDesc}... Vui lòng đợi.`, "info");

    try {
      const res = await fetchBackend("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ translation_model: newModel }),
        timeout: 180000,
      });

      if (res && res.status === 202) {
        // Backend chưa có file GGUF ⇒ trả 202 và tải trong nền
        const body = await res.json().catch(() => ({}));
        showMsg(`⏳ ${body.detail || `Đang tải model dịch ${modelDesc} về máy (chạy nền). Nút Bắt đầu đang được khoá.`}`, "info");
        await waitForTranslationActivation(newModel, shortDesc);
      } else if (!res || !res.ok) {
        const errorDetail = res ? ((await res.json().catch(() => null))?.detail || await res.text()) : "Network error";
        throw new Error(errorDetail);
      }

      const cfg = getSettings();
      cfg.translationModel = newModel;
      await api.storage.local.set({ bs_settings: cfg });

      currentModelError = null;
      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`✅ Đã nạp thành công model dịch: ${modelDesc}`, "success");
    } catch (err) {
      console.error("[Popup] Translation model switch error:", err);
      currentModelError = err.message || "Không thể chuyển model dịch";
      statusBadge.textContent = "LỖI DỊCH";
      statusBadge.className = "badge badge-disconnected";
      showMsg(`❌ Lỗi nạp model dịch: ${currentModelError}. Vui lòng chọn model khác!`, "error");
    } finally {
      isSwitchingTranslationModel = false;
      if (selTranslationModel) selTranslationModel.disabled = isCapturingNow;
      updateStartButtonState();
    }
  }

  // ── Trạng thái buffer Lookahead (thay cho HUD nổi trên trang) ──────────────
  // Nguồn dữ liệu: buffer_interceptor_poc.js trong page → content script → popup.
  //
  // v0.6.7: kèm MÃ LÝ DO (`status.lookaheadVerdict.code`) — trước đây POPUP chỉ nói
  // "không khả dụng (chạy Realtime)" mà không nói VÌ SAO, nên không thể sửa. Mã lý do và lời
  // giải thích lấy từ `lib/lookahead-diagnostics.js` (dùng chung với content script).
  function renderLookaheadStatus(status) {
    if (!lookaheadStatusEl || !lookaheadStatusText) return;
    lookaheadStatusEl.classList.remove("is-ready", "is-low", "is-off");

    const diag = (typeof window !== "undefined" && window.LookaheadDiagnostics) || null;
    const verdict = status && status.lookaheadVerdict ? status.lookaheadVerdict : null;
    const reasonText = (code) => {
      if (!diag || !code || code === "OK") return "";
      const ex = diag.explainLookaheadReason(code);
      return ex && ex.vi ? ` — ${ex.vi}` : ` — ${code}`;
    };

    if (!status || !status.hasVideo) {
      lookaheadStatusEl.classList.add("is-off");
      lookaheadStatusText.textContent = "Lookahead: chưa tìm thấy video";
      isLookaheadUnavailable = false;
      updatePipelineAOnlyUi();
      return;
    }
    const la = status.lookahead;
    if (!la || (!la.cachedChunksCount && !la.hasInitSegment)) {
      lookaheadStatusEl.classList.add("is-off");
      const code = verdict ? verdict.code : "UNKNOWN";
      lookaheadStatusText.textContent = `Lookahead: không khả dụng (chạy Realtime)${reasonText(code)}`;
      isLookaheadUnavailable = true;
      updatePipelineAOnlyUi();
      // In chẩn đoán đầy đủ ra Console của POPUP (DevTools: chuột phải popup → Inspect).
      console.warn(`[Popup][Diag] Pipeline B KHÔNG khả dụng: ${code} — ${verdict ? verdict.detail : ""}`);
      if (diag && status.lookaheadFacts) {
        console.group("%c[Popup][Diag] Vì sao không chạy được Pipeline B?", "color:#f59e0b;font-weight:bold");
        for (const row of diag.buildDiagnosticRows(status.lookaheadFacts, verdict)) {
          console.log(`  • ${row[0]}: ${row[1]}`);
        }
        console.groupEnd();
      }
      return;
    }
    const ahead = Number(la.aheadSeconds || 0);
    const enough = ahead >= 10;
    lookaheadStatusEl.classList.add(enough ? "is-ready" : "is-low");
    // §2026-10-05: `ahead` là ĐỆM VIDEO (interceptor đọc từ `video.buffered`), còn việc tạm dừng
    // để nạp đệm lại dựa trên ĐỆM ĐÃ DỊCH (`lookaheadSession.readyAhead`). Hai số này khác nhau
    // hoàn toàn: đệm video 60s mà đã dịch 0,3s thì video vẫn bị tạm dừng — hiển thị cả hai để
    // không gây hiểu nhầm "available > 60s mà vẫn pause".
    const sess = status.lookaheadSession || null;
    let suffix = enough ? "" : " (đang nạp)";
    if (sess && status.isCapturing && status.pipeline === "B") {
      const readyAhead = Number(sess.readyAhead || 0);
      suffix = ` · đã dịch +${readyAhead.toFixed(1)}s`
        + (sess.paused ? ` (tạm dừng nạp đệm)` : "");
    }
    lookaheadStatusText.textContent = `Lookahead available +${ahead.toFixed(2)}s${suffix}`;
    isLookaheadUnavailable = false;
    updatePipelineAOnlyUi();
    // Chẩn đoán định kỳ khi có điểm bất thường (ví dụ LỆCH THẺ VIDEO, buffer mù mime).
    if (verdict && verdict.code !== "OK") {
      console.info(`[Popup][Diag] Lookahead khả dụng nhưng có điểm bất thường: ${verdict.code} — ${verdict.detail}`);
      if (verdict.code === "VIDEO_MISMATCH") {
        console.warn("[Popup][Diag] ⚠️ Interceptor đang theo dõi một thẻ <video> KHÁC với popup. "
          + "Đệm báo ở đây KHÔNG phải của video đang xem — phụ đề sẽ sai/không hiện.", verdict.detail);
      }
    }
  }

  async function refreshLookaheadStatus() {
    try {
      const statuses = await broadcastToFrames("GET_STATUS");
      if (!statuses || !statuses.length) return;
      const status = statuses.find(s => s && s.hasVideo) || statuses[0];
      if (status) {
        if (status.isCapturing && (!isCapturingNow || (status.pipeline && currentActivePipeline !== status.pipeline))) {
          setUI(true, status.sampleRate, status.pipeline);
        } else if (!status.isCapturing && isCapturingNow) {
          // PHIÊN ĐÃ TỰ KẾT THÚC ở phía trang (video phát HẾT, đổi video, backend ngắt…) nhưng
          // POPUP còn tưởng đang chạy ⇒ nút vẫn ở trạng thái "Dừng" và các nhóm cài đặt bị khoá.
          // Sự cố thật 2026-10-05: video hết mà POPUP vẫn hiện "đang dịch".
          console.log("[Popup] Phiên dịch đã kết thúc ở phía trang (video hết / đổi video) — trả UI về trạng thái sẵn sàng.");
          setUI(false);
        }
        renderLookaheadStatus(status);
      }
    } catch (e) { /* popup đang mở ở tab không có content script */ }
  }

  // ── Multi-Frame Broadcast Helper ──────────────────
  async function broadcastToFrames(action, payload = {}) {    const tab = await fetchActiveTab();
    if (!tab) return null;

    if (api.scripting && api.scripting.executeScript) {
      try {
        const results = await api.scripting.executeScript({
          target: { tabId: tab.id, allFrames: true },
          func: async (act, p) => {
            try {
              if (act === "START_TRANSLATION" && typeof window.__bsStartCapture === "function") {
                return await window.__bsStartCapture(p);
              }
              if (act === "STOP_TRANSLATION" && typeof window.__bsStopCapture === "function") {
                return await window.__bsStopCapture();
              }
              if (act === "GET_STATUS" && typeof window.__bsGetStatus === "function") {
                return await window.__bsGetStatus();
              }
              if (act === "update_settings" && typeof window.__bsUpdateSettings === "function") {
                return await window.__bsUpdateSettings(p.settings || p);
              }
            } catch (e) {
              return { success: false, error: e.message };
            }
            return null;
          },
          args: [action, payload]
        });
        if (results && results.length > 0) {
          return results.filter(r => r != null && r.result !== undefined).map(r => r.result).filter(Boolean);
        }
      } catch (e) {
        console.warn("[Popup] Scripting broadcast fallback to tabs.sendMessage:", e);
      }
    }

    return new Promise((resolve) => {
      api.tabs.sendMessage(tab.id, { action, ...payload }, (r) => {
        if (api.runtime.lastError) {
          resolve(null);
        } else {
          resolve(r ? [r] : []);
        }
      });
    });
  }

  // ── Init (async, non-blocking) ─────────────────────
  (async () => {
    try {
      const stored = await api.storage.local.get("bs_settings");
      const s = stored.bs_settings || {};
      if (s.asrEngine && selAsrEngine) selAsrEngine.value = s.asrEngine;
      if (s.vadEngine && selVadEngine) selVadEngine.value = s.vadEngine;
      if ((s.vadSilenceDurationMs !== undefined || s.silenceDurationMs !== undefined) && rangeVadSilence) {
        const saved = s.vadSilenceDurationMs !== undefined ? s.vadSilenceDurationMs : s.silenceDurationMs;
        // `||` sẽ biến 0 (OFF) thành giá trị khác ⇒ phải dùng kiểm tra tường minh.
        const ms = parseInt(saved, 10);
        rangeVadSilence.value = isNaN(ms) ? 0 : Math.max(0, ms);
        rangeVadSilence.dataset.userEdited = "true";
      }
      if ((s.vadThreshold !== undefined || s.vad_threshold !== undefined || s.threshold !== undefined) && rangeVadThreshold) {
        rangeVadThreshold.value = s.vadThreshold !== undefined ? s.vadThreshold : (s.vad_threshold !== undefined ? s.vad_threshold : s.threshold);
        rangeVadThreshold.dataset.userEdited = "true";
      }
      if (s.minWordsToCommit !== undefined && rangeMinWords) {
        const mw = parseInt(s.minWordsToCommit, 10);
        rangeMinWords.value = isNaN(mw) ? 2 : Math.max(0, mw);
        rangeMinWords.dataset.userEdited = "true";
      }
      // stable_cut (đã lưu ở lần dùng trước)
      if (s.splitOnStability !== undefined && chkStableCut) chkStableCut.checked = !!s.splitOnStability;
      if (s.split_on_stability !== undefined && chkStableCut) chkStableCut.checked = !!s.split_on_stability;
      const savedStableMs = s.stability_duration_ms !== undefined
        ? s.stability_duration_ms
        : (s.stabilityDurationSec !== undefined ? parseFloat(s.stabilityDurationSec) * 1000 : undefined);
      if (savedStableMs !== undefined && rangeStableMs) {
        const maxMs = parseFloat(rangeStableMs.max) || 500;
        if (savedStableMs > maxMs) {
          rangeStableMs.value = rangeStableMs.defaultValue || "0";
        } else {
          rangeStableMs.value = Math.round(savedStableMs);
        }
        rangeStableMs.dataset.userEdited = "true";
      }
      const savedMinSec = s.stability_min_duration_sec !== undefined
        ? s.stability_min_duration_sec
        : s.stabilityMinDurationSec;
      if (savedMinSec !== undefined && rangeStableMinSec) {
        rangeStableMinSec.value = savedMinSec;
        rangeStableMinSec.dataset.userEdited = "true";
      }
      const savedMinWords = s.stability_min_words !== undefined ? s.stability_min_words : s.stabilityMinWords;
      if (savedMinWords !== undefined && rangeStableMinWords) {
        rangeStableMinWords.value = savedMinWords;
        rangeStableMinWords.dataset.userEdited = "true";
      }
      const savedTrace = s.trace_stability !== undefined ? s.trace_stability : s.traceStability;
      if (savedTrace !== undefined && chkStableTrace) chkStableTrace.checked = !!savedTrace;
      if (s.sourceLanguage || s.sourceLang) {
        savedPreferredLang = s.sourceLanguage || s.sourceLang;
        if (selSourceLang) selSourceLang.value = savedPreferredLang;
      }
      if (s.targetLang && selTargetLang) selTargetLang.value = s.targetLang;
      if (s.translationModel && selTranslationModel) selTranslationModel.value = s.translationModel;
      if (s.subPosY !== undefined && rangeSubPosY) rangeSubPosY.value = s.subPosY;
      if (s.subWidth !== undefined && rangeSubWidth) rangeSubWidth.value = s.subWidth;
      if (s.origFontSize !== undefined && rangeOrigSize) {
        rangeOrigSize.value = normalizeFontPercent(s.origFontSize, false);
      }
      if (s.transFontSize !== undefined && rangeTransSize) {
        rangeTransSize.value = normalizeFontPercent(s.transFontSize, true);
      }
      if (s.fontWeight && rangeFontWeight) rangeFontWeight.value = s.fontWeight;
      if (s.fontFamily) selFontFamily.value = s.fontFamily;
      if (s.maxLines && rangeMaxLines) rangeMaxLines.value = s.maxLines;
      // Mặc định BẬT (tắt chạy chữ) nếu người dùng chưa từng đặt.
      if (chkTranslationOnce) {
        chkTranslationOnce.checked = s.showTranslationOnce === undefined ? true : !!s.showTranslationOnce;
      }
      if (chkShowOriginal) {
        chkShowOriginal.checked = s.showOriginalSubtitles === undefined ? true : !!s.showOriginalSubtitles;
      }

      // Restore TTS Settings
      if (s.ttsEnabled !== undefined && chkEnableTts) {
        chkEnableTts.checked = !!s.ttsEnabled;
      }
      if (s.ttsVoice) {
        savedPreferredVoiceId = s.ttsVoice;
        if (selTtsVoice) selTtsVoice.value = s.ttsVoice;
      }
      if (s.ttsSpeed !== undefined && selTtsSpeed) {
        let spd = String(s.ttsSpeed);
        if (spd === "1") spd = "1.0";
        selTtsSpeed.value = spd;
        if (!selTtsSpeed.value) selTtsSpeed.value = "1.0";
      }
      if (s.ttsDucking !== undefined && selTtsDucking) selTtsDucking.value = s.ttsDucking ? "true" : "false";
      // Restore Lookahead Settings
      if (s.lookaheadEnabled !== undefined && chkEnableLookahead) {
        chkEnableLookahead.checked = !!s.lookaheadEnabled;
      }
      if (s.lookaheadSyncOffsetMs !== undefined && rangeLookaheadSync) {
        rangeLookaheadSync.value = s.lookaheadSyncOffsetMs;
      }

      updateRangeLabels();
    } catch (e) { }

    await fetchBackendEngineConfig();

    if (chkEnableTts && chkEnableTts.checked) {
      prewarmTTS();
    }

    const tab = await fetchActiveTab();
    if (!tab) return;

    const statuses = await broadcastToFrames("GET_STATUS");
    const capturingFrame = statuses?.find(s => s?.isCapturing);
    if (capturingFrame) {
      const rate = capturingFrame.sampleRate || statuses.find(s => s?.sampleRate)?.sampleRate;
      const pipe = capturingFrame.pipeline || statuses.find(s => s?.pipeline)?.pipeline;
      setUI(true, rate, pipe);
    } else {
      api.tabs.sendMessage(tab.id, { action: "GET_STATUS", type: "GET_STATUS" }, (r) => {
        if (!api.runtime.lastError && r?.isCapturing) setUI(true, r?.sampleRate, r?.pipeline);
      });
    }

    // Trạng thái buffer Lookahead: hiển thị ngay và làm mới định kỳ khi popup còn mở.
    if (statuses && statuses.length) {
      renderLookaheadStatus(statuses.find(s => s && s.hasVideo) || statuses[0]);
    } else {
      refreshLookaheadStatus();
    }
    if (lookaheadStatusTimer === null) {
      lookaheadStatusTimer = setInterval(refreshLookaheadStatus, 1200);
    }
  })();

  btnStart.addEventListener("click", async () => {
    if (isSwitchingEngine || isSwitchingTranslationModel || currentModelError) {
      showMsg("⚠️ Không thể bắt đầu session khi model đang tải hoặc chưa sẵn sàng!", "warning");
      return;
    }

    const tab = await fetchActiveTab();
    if (!tab) { showMsg("No active tab found", "error"); return; }
    const cfg = getSettings();
    api.storage.local.set({ bs_settings: cfg });

    btnStart.disabled = true;
    showMsg("🔍 Đang tìm kiếm video player...", "info");

    const payload = { action: "START_TRANSLATION", sourceLanguage: selSourceLang.value, settings: cfg };
    const results = await broadcastToFrames("START_TRANSLATION", payload);

    if (!results || results.length === 0) {
      api.tabs.sendMessage(tab.id, payload, (r) => {
        if (api.runtime.lastError) {
          showMsg("⚠️ Hãy F5 lại trang và bấm Play video trước!", "error");
          updateStartButtonState();
          return;
        }
        if (r?.success || r?.isCapturing) {
          const pipe = r?.pipeline || (chkEnableLookahead?.checked && !isLookaheadUnavailable ? "B" : "A");
          setUI(true, r?.sampleRate, pipe);
          showMsg(`✅ Đã tìm thấy video và bắt đầu dịch qua Pipeline ${pipe}!`, "success");
        } else {
          showMsg("❌ " + (r?.error || "Không tìm thấy video nào (Hãy bấm Play video trước)"), "error");
          updateStartButtonState();
        }
      });
      return;
    }

    const successResult = results.find(r => r && (r.success || r.isCapturing));
    if (successResult) {
      const rate = successResult.sampleRate || results.find(r => r?.sampleRate)?.sampleRate;
      const pipe = successResult.pipeline || results.find(r => r?.pipeline)?.pipeline || (chkEnableLookahead?.checked && !isLookaheadUnavailable ? "B" : "A");
      setUI(true, rate, pipe);
      showMsg(`✅ Đã tìm thấy video và bắt đầu dịch qua Pipeline ${pipe}!`, "success");
    } else {
      const specificError = results.find(r => r?.error && r.error !== "No video found");
      const errorMsg = specificError?.error || results.map(r => r?.error).filter(Boolean)[0] || "Không tìm thấy video nào (Hãy bấm Play video trước)";
      showMsg("❌ " + errorMsg, "error");
      updateStartButtonState();
    }
  });

  btnStop.addEventListener("click", async () => {
    const tab = await fetchActiveTab();
    if (!tab) return;
    await broadcastToFrames("STOP_TRANSLATION");
    setUI(false);
    showMsg("⏹️ Đã dừng dịch", "info");
  });

  btnTest.addEventListener("click", () => {
    btnTest.disabled = true; btnTest.textContent = "Testing...";
    const ws = new WebSocket("wss://localhost:8765/ws");
    ws.onopen = () => { ws.close(); showMsg("✅ Backend reachable!", "success"); btnTest.disabled = false; btnTest.textContent = "🔌 Test"; };
    ws.onerror = () => { showMsg("❌ Cannot reach backend", "error"); btnTest.disabled = false; btnTest.textContent = "🔌 Test"; };
    setTimeout(() => { if (ws.readyState === 0) { ws.close(); showMsg("❌ Timeout", "error"); btnTest.disabled = false; btnTest.textContent = "🔌 Test"; } }, 5000);
  });

  // ── Live update settings with 150ms debounce for sliders ──────
  let _settingDebounceTimer = null;

  async function liveUpdateSettings(immediate = false) {
    updateRangeLabels();
    clearTimeout(_settingDebounceTimer);

    const apply = async () => {
      const cfg = getSettings();
      api.storage.local.set({ bs_settings: cfg });
      const tab = await fetchActiveTab();
      if (tab && isCapturingNow) {
        await broadcastToFrames("update_settings", { settings: cfg });
        showMsg("⚡ Settings applied live", "info");
      }
      fetchBackend("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          min_words_to_commit: cfg.minWordsToCommit,
          silence_duration_ms: cfg.silenceDurationMs,
          vad_threshold: cfg.vadThreshold,
          // stable_cut: đẩy sang REST để áp cho MỌI phiên đang chạy (giống VAD/threshold).
          split_on_stability: cfg.splitOnStability,
          stability_duration_ms: cfg.stability_duration_ms,
          stability_min_duration_sec: cfg.stability_min_duration_sec,
          stability_min_words: cfg.stability_min_words,
          trace_stability: cfg.traceStability,
        }),
      }).catch(() => { });
    };

    if (immediate) {
      await apply();
    } else {
      _settingDebounceTimer = setTimeout(apply, 150);
    }
  }

  function onSettingChange(immediate = false) {
    liveUpdateSettings(immediate);
  }

  if (selAsrEngine) selAsrEngine.onchange = handleEngineSwitch;
  if (selVadEngine) selVadEngine.onchange = handleEngineSwitch;
  if (rangeVadSilence) {
    rangeVadSilence.oninput = () => {
      rangeVadSilence.dataset.userEdited = "true";
      onSettingChange();
    };
  }
  if (rangeVadThreshold) {
    rangeVadThreshold.oninput = () => {
      rangeVadThreshold.dataset.userEdited = "true";
      onSettingChange();
    };
  }
  if (rangeMinWords) {
    rangeMinWords.oninput = () => {
      rangeMinWords.dataset.userEdited = "true";
      onSettingChange();
    };
  }
  // stable_cut: đổi là gửi ngay (cần thấy hiệu quả tức thì khi tinh chỉnh thời gian).
  if (chkStableCut) chkStableCut.onchange = () => onSettingChange(true);
  // Công tắc chẩn đoán: bật/tắt phải có hiệu lực NGAY (không debounce).
  if (chkStableTrace) chkStableTrace.onchange = () => onSettingChange(true);
  // Lookahead Video Buffering
  if (chkEnableLookahead) chkEnableLookahead.onchange = () => {
    if (isCapturingNow) {
      chkEnableLookahead.checked = !chkEnableLookahead.checked;
      return;
    }
    // Bật Lookahead ⇒ vô hiệu nhóm tuỳ chỉnh Pipeline A (và ngược lại) NGAY khi bấm.
    updatePipelineAOnlyUi();
    updateRangeLabels();
    onSettingChange(true);
  };
  if (rangeLookaheadSync) rangeLookaheadSync.oninput = () => onSettingChange();
  [rangeStableMs, rangeStableMinSec, rangeStableMinWords].forEach((el) => {
    if (el) {
      el.oninput = () => {
        el.dataset.userEdited = "true";
        onSettingChange();
      };
    }
  });

  if (selSourceLang) {
    selSourceLang.onchange = () => {
      savedPreferredLang = selSourceLang.value;
      if (selAsrEngine && selAsrEngine.value === "whisper") {
        handleEngineSwitch(true);
      } else {
        onSettingChange();
      }
    };
  }
  if (selTargetLang) selTargetLang.onchange = onSettingChange;
  if (selTranslationModel) selTranslationModel.onchange = handleTranslationModelSwitch;
  if (rangeSubPosY) rangeSubPosY.oninput = onSettingChange;
  if (rangeSubWidth) rangeSubWidth.oninput = onSettingChange;
  if (rangeOrigSize) rangeOrigSize.oninput = onSettingChange;
  if (rangeTransSize) rangeTransSize.oninput = onSettingChange;
  if (rangeFontWeight) rangeFontWeight.oninput = onSettingChange;
  if (selFontFamily) selFontFamily.onchange = onSettingChange;
  if (rangeMaxLines) rangeMaxLines.oninput = onSettingChange;
  // Tắt chạy chữ: áp dụng NGAY (không debounce) để thấy hiệu quả tức thì.
  if (chkTranslationOnce) chkTranslationOnce.onchange = () => onSettingChange(true);
  if (chkShowOriginal) chkShowOriginal.onchange = () => onSettingChange(true);

  //: Bấm vào cả hàng để bật/tắt — 5 hàng công tắc dùng chung một hành vi.
  wireToggleRow(showOriginalToggleRow, chkShowOriginal);
  wireToggleRow(stableToggleRow, chkStableCut);
  wireToggleRow(stableTraceToggleRow, chkStableTrace);
  wireToggleRow(translationOnceToggleRow, chkTranslationOnce);

  function syncTtsConfig(enabled) {
    const payload = {
      tts_enabled: enabled,
      tts_voice: selTtsVoice ? selTtsVoice.value : undefined,
      tts_speed: selTtsSpeed ? parseFloat(selTtsSpeed.value || 1.0) : 1.0,
    };
    fetchBackend("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).catch(() => { });
  }

  function prewarmTTS() {
    syncTtsConfig(true);
    fetchBackend("/api/tts/prewarm", { method: "POST" }).catch(() => { });
  }

  function unloadTTS() {
    syncTtsConfig(false);
    fetchBackend("/api/tts/unload", { method: "POST" }).catch(() => { });
  }

  if (chkEnableTts) {
    chkEnableTts.addEventListener("change", () => {
      onSettingChange(true);
      if (chkEnableTts.checked) {
        prewarmTTS();
      } else {
        unloadTTS();
      }
    });
  }

  wireToggleRow(document.getElementById("ttsToggleRow"), chkEnableTts);

  if (selTtsVoice) {
    selTtsVoice.onchange = () => {
      savedPreferredVoiceId = selTtsVoice.value;
      onSettingChange();
    };
  }
  if (selTtsSpeed) selTtsSpeed.onchange = onSettingChange;
  if (selTtsDucking) selTtsDucking.onchange = onSettingChange;
  if (selTtsDucking) selTtsDucking.addEventListener("change", updateDuckingUi);
  if (rangeDuckingLevel) rangeDuckingLevel.oninput = onSettingChange;

  function setUI(active, audioRate = null, pipeline = null) {
    isCapturingNow = active;
    if (active) {
      if (audioRate) {
        currentAudioSampleRate = audioRate;
      }
      if (pipeline) {
        currentActivePipeline = pipeline;
      } else if (!currentActivePipeline) {
        const lookaheadOn = !chkEnableLookahead || chkEnableLookahead.checked !== false;
        currentActivePipeline = (lookaheadOn && !isLookaheadUnavailable) ? "B" : "A";
      }
    } else {
      currentAudioSampleRate = null;
      currentActivePipeline = null;
    }

    // Ẩn/Hiện nhóm lựa chọn model theo trạng thái phiên
    const rowModelEngine = document.getElementById("rowModelEngine");
    const groupTranslationModel = document.getElementById("groupTranslationModel");
    const activeModelSummary = document.getElementById("activeModelSummary");
    const activeModelsText = document.getElementById("activeModelsText");
    const activePipelineText = document.getElementById("activePipelineText");
    const activePipelineBadge = document.getElementById("activePipelineBadge");

    if (rowModelEngine) {
      rowModelEngine.style.display = active ? "none" : "";
    }
    if (groupTranslationModel) {
      groupTranslationModel.style.display = active ? "none" : "";
    }
    if (activeModelSummary) {
      activeModelSummary.style.display = active ? "block" : "none";
      if (active) {
        const pipe = currentActivePipeline || "A";
        if (activePipelineText) {
          activePipelineText.textContent = pipe === "B"
            ? "Pipeline B"
            : "Pipeline A";
        }
        if (activePipelineBadge) {
          if (pipe === "B") {
            activePipelineBadge.textContent = "Lookahead";
            activePipelineBadge.style.background = "rgba(16, 185, 129, 0.15)";
            activePipelineBadge.style.color = "#047857";
          } else {
            activePipelineBadge.textContent = "Realtime";
            activePipelineBadge.style.background = "rgba(245, 158, 11, 0.15)";
            activePipelineBadge.style.color = "#b45309";
          }
        }
        if (activeModelsText) {
          const asrText = selAsrEngine && selAsrEngine.selectedOptions && selAsrEngine.selectedOptions[0]
            ? selAsrEngine.selectedOptions[0].textContent.replace("⚡", "").trim()
            : (lastActiveAsr || "--");
          const transText = selTranslationModel && selTranslationModel.selectedOptions && selTranslationModel.selectedOptions[0]
            ? selTranslationModel.selectedOptions[0].textContent.replace("⚡", "").replace("🌟", "").replace("🎯", "").replace("🎌", "").trim()
            : "Translation";
          activeModelsText.textContent = `${asrText.split("(")[0].trim()} ⏐ ${transText.split("(")[0].trim()}`;
        }
      }
    }

    if (selTranslationModel) {
      selTranslationModel.disabled = active || isSwitchingTranslationModel;
    }
    if (selAsrEngine) {
      selAsrEngine.disabled = active || isSwitchingEngine;
    }
    if (selVadEngine) {
      selVadEngine.disabled = active || isSwitchingEngine;
    }
    if (chkEnableLookahead) {
      chkEnableLookahead.disabled = active;
      const row = document.getElementById("lookaheadToggleRow");
      if (row) {
        row.style.opacity = active ? "0.6" : "1";
        row.title = active ? "Không thể chuyển đổi Pipeline khi đang chạy session" : "";
      }
    }

    updateStartButtonState();

    // Bật/tắt session thay đổi `isCapturingNow` và `currentActivePipeline` ⇒ tính lại nhóm tuỳ chỉnh Pipeline A.
    updatePipelineAOnlyUi();
    statusBadge.textContent = active ? getCapturingLabel(currentAudioSampleRate) : `${lastActiveAsr.toUpperCase()}`;
    statusBadge.className = active ? "badge badge-connected" : "badge badge-ready";
    statusBadge.title = active ? (currentActivePipeline === "B" ? "Đang chạy Pipeline B (Lookahead OFFLINE_BATCH)" : `Đang thu âm thanh: ${currentAudioSampleRate || 16000} Hz (Pipeline A)`) : "";
  }

  function showMsg(t, tp) {
    messageArea.textContent = t;
    messageArea.className = "message-area message-" + tp;
  }
})();
