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
  const valMinWords = document.getElementById("valMinWords");
  const lblActiveModel = document.getElementById("lblActiveModel");
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

  const chkEnableTts = document.getElementById("chkEnableTts");
  const selTtsVoice = document.getElementById("selTtsVoice");
  const selTtsSpeed = document.getElementById("selTtsSpeed");
  const valTtsSpeed = document.getElementById("valTtsSpeed");
  const selTtsDucking = document.getElementById("selTtsDucking");
  const rangeDuckingLevel = document.getElementById("rangeDuckingLevel");
  const valDuckingLevel = document.getElementById("valDuckingLevel");

  let availableVoices = [];
  let savedPreferredVoiceId = null;
  let savedPreferredLang = null;

  let activeTab = null;
  let isCapturingNow = false;
  let isSwitchingEngine = false;
  let lastActiveAsr = "sensevoice";
  let lastActiveVad = "auto";
  let lastActiveLang = "auto";

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
      // "Tắt chạy chữ": chỉ hiện bản dịch MỘT LẦN khi có bản dịch hoàn chỉnh.
      showTranslationOnce: chkTranslationOnce ? chkTranslationOnce.checked : true,
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
      if (unit) unit.textContent = off ? " (mặc định VAD)" : "ms";
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
    const activeVad = data.vad_engine || "fsmn-vad";
    lastActiveAsr = activeAsr;
    lastActiveVad = activeVad;

    if (lblActiveModel) {
      lblActiveModel.textContent = data.loaded_model || data.model_size || activeAsr;
      lblActiveModel.title = `Mô hình ASR đang nạp: ${data.loaded_model || data.model_size || activeAsr}`;
    }

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

    if (!isCapturingNow) {
      statusBadge.textContent = `${activeAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`✅ Server Online (ASR: ${activeAsr.toUpperCase()} | VAD: ${data.resolved_vad || activeVad})`, "success");
      btnStart.disabled = false;
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

      // Backend có thể đang tải/nạp model dịch ở nền (F-51) ⇒ báo rõ để không tưởng là treo.
      const dl = data.translation.download;
      if (dl && dl.model && (dl.state === "downloading" || dl.state === "loading")) {
        const verb = dl.state === "downloading" ? "tải" : "nạp";
        const note = dl.state === "downloading" ? ` (${describeDownloadProgress(dl)})` : "";
        showMsg(`⏳ Backend đang ${verb} model dịch '${dl.model}'${note} ở chế độ nền. Model hiện tại vẫn dịch bình thường.`, "info");
      } else if (dl && dl.state === "error" && dl.error) {
        showMsg(`❌ Model dịch '${dl.model || "?"}' lỗi: ${dl.error}`, "error");
      }
    }

    // Backend có thể đang tải/nạp model ASR ở nền ⇒ báo rõ để người dùng biết
    const asrDl = data.asr_download || (data.asr && data.asr.download);
    if (asrDl && asrDl.model && (asrDl.state === "downloading" || asrDl.state === "loading")) {
      const verb = asrDl.state === "downloading" ? "tải" : "nạp";
      const note = asrDl.state === "downloading" ? ` (${describeDownloadProgress(asrDl)})` : "";
      showMsg(`⏳ Backend đang ${verb} model ASR '${asrDl.model}'${note} ở chế độ nền. Model hiện tại vẫn nhận diện bình thường.`, "info");
    } else if (asrDl && asrDl.state === "error" && asrDl.error) {
      showMsg(`❌ Model ASR '${asrDl.model || "?"}' lỗi: ${asrDl.error}`, "error");
    }

    return true;
  }

  // ── Hot-swap ASR / VAD Engine on Backend ───────────────────
  async function handleEngineSwitch(force = false) {
    if (isSwitchingEngine) return;

    const newAsr = selAsrEngine ? selAsrEngine.value : lastActiveAsr;
    const newVad = selVadEngine ? selVadEngine.value : lastActiveVad;
    const newLang = selSourceLang ? selSourceLang.value : "auto";

    if (!force && newAsr === lastActiveAsr && newVad === lastActiveVad && (newAsr !== "whisper" || newLang === lastActiveLang)) return;

    isSwitchingEngine = true;
    if (selAsrEngine) selAsrEngine.disabled = true;
    if (selVadEngine) selVadEngine.disabled = true;
    btnStart.disabled = true;

    const labelDesc = newAsr === "whisper" ? `Whisper (${newLang})` : newAsr.toUpperCase();
    statusBadge.textContent = `Nạp ${labelDesc}...`;
    statusBadge.className = "badge badge-reconnecting";
    if (lblActiveModel) {
      lblActiveModel.textContent = `Đang nạp ${labelDesc}...`;
      lblActiveModel.title = `Đang nạp mô hình ${labelDesc}...`;
    }
    showMsg(`⏳ Đang chuyển đổi sang ${labelDesc} & nạp model Finetunes... Vui lòng đợi.`, "info");

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
        // Backend chưa có file GGUF ⇒ trả 202 và tải trong nền (model cũ vẫn chạy).
        const body = await res.json().catch(() => ({}));
        showMsg(`⏳ ${body.detail || `Đang tải model ASR ${labelDesc} về máy (chạy nền).`}`, "info");
        await waitForAsrActivation(newAsr, labelDesc);
      } else if (!res || !res.ok) {
        const errorDetail = res ? await res.text() : "Network error";
        throw new Error(errorDetail);
      }

      const result = res.status === 202 ? (await (await fetchBackend("/api/config", { timeout: 15000 })).json().catch(() => ({}))) : (await res.json());
      lastActiveAsr = result.engine || newAsr;
      lastActiveVad = result.vad_engine || newVad;
      lastActiveLang = result.source_lang || newLang;

      if (lblActiveModel) {
        lblActiveModel.textContent = result.loaded_model || result.model_size || lastActiveAsr;
        lblActiveModel.title = `Mô hình ASR đang nạp: ${result.loaded_model || result.model_size || lastActiveAsr}`;
      }

      // Update supported languages dropdown
      if (result.supported_languages) {
        renderLanguageOptions(result.supported_languages, savedPreferredLang || result.source_lang || selSourceLang.value);
      }

      // Save settings
      const cfg = getSettings();
      api.storage.local.set({ bs_settings: cfg });

      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = isCapturingNow ? "badge badge-connected" : "badge badge-ready";
      showMsg(`✅ Đã chuyển sang ASR: ${lastActiveAsr.toUpperCase()} (${lastActiveLang}) | VAD: ${result.resolved_vad || lastActiveVad}`, "success");

      // Notify content script of active changes
      const tab = await fetchActiveTab();
      if (tab && isCapturingNow) {
        await broadcastToFrames("update_settings", { settings: cfg });
      }
    } catch (err) {
      console.error("[Popup] Engine switch error:", err);
      // Rollback dropdown selections
      if (selAsrEngine) selAsrEngine.value = lastActiveAsr;
      if (selVadEngine) selVadEngine.value = lastActiveVad;
      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`❌ Lỗi nạp model: ${err.message || "Không thể chuyển engine"}`, "error");
    } finally {
      isSwitchingEngine = false;
      if (selAsrEngine) selAsrEngine.disabled = isCapturingNow;
      if (selVadEngine) selVadEngine.disabled = isCapturingNow;
      btnStart.disabled = isCapturingNow || isSwitchingTranslationModel;
    }
  }

  // ── Hot-swap Translation Model on Backend ─────────────────
  let isSwitchingTranslationModel = false;

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
          showMsg(`⏳ Đang tải model ASR ${shortDesc} về máy: ${note}. Model hiện tại vẫn nhận diện bình thường.`, "info");
        }
        continue;
      }
      if (dl.state === "loading") {
        if (statusBadge) statusBadge.textContent = `Nạp ${shortDesc}...`;
        if (lastNote !== "loading") {
          lastNote = "loading";
          showMsg(`⏳ Đã tải xong ${shortDesc}, đang nạp vào GPU...`, "info");
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
          showMsg(`⏳ Đang tải model dịch ${shortDesc} về máy: ${note}. Model hiện tại vẫn dịch bình thường.`, "info");
        }
        continue;
      }
      if (dl.state === "loading") {
        if (statusBadge) statusBadge.textContent = `Nạp ${shortDesc}...`;
        if (lastNote !== "loading") {
          lastNote = "loading";
          showMsg(`⏳ Đã tải xong ${shortDesc}, đang nạp vào GPU...`, "info");
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

  async function handleTranslationModelSwitch() {
    if (isCapturingNow) {
      showMsg("⚠️ Không thể đổi model dịch khi đang dịch! Hãy bấm 'Dừng dịch' trước.", "warning");
      return;
    }
    if (isSwitchingTranslationModel) return;

    const newModel = selTranslationModel ? selTranslationModel.value : "tencent";
    const selectedOption = selTranslationModel && selTranslationModel.selectedOptions ? selTranslationModel.selectedOptions[0] : null;
    const modelDesc = selectedOption ? selectedOption.textContent.replace("⚡", "").trim() : newModel;
    const shortDesc = modelDesc.split("(")[0].trim() || newModel;

    isSwitchingTranslationModel = true;
    if (selTranslationModel) selTranslationModel.disabled = true;
    btnStart.disabled = true;

    statusBadge.textContent = `Nạp ${shortDesc}...`;
    statusBadge.className = "badge badge-reconnecting";
    showMsg(`⏳ Đang chuyển đổi sang model dịch ${modelDesc}... (Nếu chưa có, model sẽ tự tải về)`, "info");

    try {
      const res = await fetchBackend("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ translation_model: newModel }),
        timeout: 180000,
      });

      if (res && res.status === 202) {
        // Backend chưa có file GGUF ⇒ trả 202 và tải trong nền (model cũ vẫn chạy).
        const body = await res.json().catch(() => ({}));
        showMsg(`⏳ ${body.detail || `Đang tải model dịch ${modelDesc} về máy (chạy nền).`}`, "info");
        await waitForTranslationActivation(newModel, shortDesc);
      } else if (!res || !res.ok) {
        const errorDetail = res ? await res.text() : "Network error";
        throw new Error(errorDetail);
      }

      const cfg = getSettings();
      cfg.translationModel = newModel;
      await api.storage.local.set({ bs_settings: cfg });

      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`✅ Đã chuyển sang model dịch: ${modelDesc}`, "success");
    } catch (err) {
      console.error("[Popup] Translation model switch error:", err);
      showMsg(`❌ Lỗi nạp model dịch: ${err.message || "Không thể chuyển model"}`, "error");
      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
    } finally {
      isSwitchingTranslationModel = false;
      if (selTranslationModel) selTranslationModel.disabled = isCapturingNow;
      btnStart.disabled = isCapturingNow || isSwitchingEngine;
    }
  }

  // ── Multi-Frame Broadcast Helper ──────────────────
  async function broadcastToFrames(action, payload = {}) {
    const tab = await fetchActiveTab();
    if (!tab) return null;

    if (api.scripting && api.scripting.executeScript) {
      try {
        const results = await api.scripting.executeScript({
          target: { tabId: tab.id, allFrames: true },
          func: (act, p) => {
            try {
              if (act === "START_TRANSLATION" && typeof window.__bsStartCapture === "function") {
                return window.__bsStartCapture(p);
              }
              if (act === "STOP_TRANSLATION" && typeof window.__bsStopCapture === "function") {
                return window.__bsStopCapture();
              }
              if (act === "GET_STATUS" && typeof window.__bsGetStatus === "function") {
                return window.__bsGetStatus();
              }
              if (act === "update_settings" && typeof window.__bsUpdateSettings === "function") {
                return window.__bsUpdateSettings(p.settings || p);
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
      if (s.duckingLevel !== undefined && rangeDuckingLevel) {
        rangeDuckingLevel.value = Math.round(s.duckingLevel * 100);
      }

      updateRangeLabels();
    } catch (e) { }

    await fetchBackendEngineConfig();

    const tab = await fetchActiveTab();
    if (!tab) return;

    const statuses = await broadcastToFrames("GET_STATUS");
    if (statuses && statuses.some(s => s?.isCapturing)) {
      setUI(true);
    } else {
      api.tabs.sendMessage(tab.id, { action: "GET_STATUS", type: "GET_STATUS" }, (r) => {
        if (!api.runtime.lastError && r?.isCapturing) setUI(true);
      });
    }
  })();

  btnStart.addEventListener("click", async () => {
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
          btnStart.disabled = false;
          return;
        }
        if (r?.success || r?.isCapturing) {
          setUI(true);
          showMsg("✅ Đã tìm thấy video và bắt đầu dịch!", "success");
        } else {
          showMsg("❌ " + (r?.error || "Không tìm thấy video nào (Hãy bấm Play video trước)"), "error");
          btnStart.disabled = false;
        }
      });
      return;
    }

    const successResult = results.find(r => r && (r.success || r.isCapturing));
    if (successResult) {
      setUI(true);
      showMsg("✅ Đã tìm thấy video và bắt đầu dịch!", "success");
    } else {
      const specificError = results.find(r => r?.error && r.error !== "No video found");
      const errorMsg = specificError?.error || results.map(r => r?.error).filter(Boolean)[0] || "Không tìm thấy video nào (Hãy bấm Play video trước)";
      showMsg("❌ " + errorMsg, "error");
      btnStart.disabled = false;
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

  if (chkEnableTts) {
    chkEnableTts.addEventListener("change", () => {
      onSettingChange();
      syncTtsConfig(chkEnableTts.checked);
      if (chkEnableTts.checked) {
        prewarmTTS();
      }
    });
  }

  const ttsToggleRow = document.getElementById("ttsToggleRow");
  if (ttsToggleRow && chkEnableTts) {
    ttsToggleRow.addEventListener("click", (e) => {
      if (e.target.closest(".switch")) {
        return;
      }
      chkEnableTts.checked = !chkEnableTts.checked;
      chkEnableTts.dispatchEvent(new Event("change"));
    });
  }

  if (selTtsVoice) {
    selTtsVoice.onchange = () => {
      savedPreferredVoiceId = selTtsVoice.value;
      onSettingChange();
    };
  }
  if (selTtsSpeed) selTtsSpeed.onchange = onSettingChange;
  if (selTtsDucking) selTtsDucking.onchange = onSettingChange;
  if (rangeDuckingLevel) rangeDuckingLevel.oninput = onSettingChange;

  function setUI(active) {
    isCapturingNow = active;
    btnStart.disabled = active || isSwitchingEngine || isSwitchingTranslationModel;
    btnStop.disabled = !active;
    if (selTranslationModel) {
      selTranslationModel.disabled = active || isSwitchingTranslationModel;
    }
    if (selAsrEngine) {
      selAsrEngine.disabled = active || isSwitchingEngine;
    }
    if (selVadEngine) {
      selVadEngine.disabled = active || isSwitchingEngine;
    }
    statusBadge.textContent = active ? "Capturing" : `${lastActiveAsr.toUpperCase()}`;
    statusBadge.className = active ? "badge badge-connected" : "badge badge-ready";
  }

  function showMsg(t, tp) {
    messageArea.textContent = t;
    messageArea.className = "message-area message-" + tp;
  }
})();
