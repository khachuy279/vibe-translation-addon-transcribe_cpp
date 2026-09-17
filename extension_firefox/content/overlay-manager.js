// Overlay Manager
// Injects subtitle overlay inside the video container using Shadow DOM for CSS isolation.
// Automatically adjusts according to video percentage coordinates (bottom offset %, width %)
// and maintains seamless positioning across resize and fullscreen changes.

class OverlayManager {
  constructor() {
    this.renderer = null;
    this.host = null;
    this.shadow = null;
    this.container = null;
    this.targetVideo = null;
    this.isActive = false;

    this.subPosY = 10; // Default: 10% from bottom
    this.subWidth = 80; // Default: 80% width of video container
    this.origFontPercent = 2.5; // Default: 2.5% of video height
    this.transFontPercent = 4.2; // Default: 4.2% of video height

    this._onFullscreenChange = this._onFullscreenChange.bind(this);
    this._onWindowResize = this._onWindowResize.bind(this);
    this._domObserver = null;
    this._resizeObserver = null;
  }

  // ── Lifecycle ────────────────────────────────────────────

  init(video = null) {
    if (this.isActive) {
      if (video && video !== this.targetVideo) {
        this.attachToVideo(video);
      }
      return;
    }

    this.targetVideo = video;

    // Create shadow DOM host
    this.host = document.createElement("div");
    this.host.id = "bs-overlay-host";
    this.host.style.cssText = "position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 2147483647; overflow: hidden;";
    this.shadow = this.host.attachShadow({ mode: "open" });

    // Container inside shadow DOM
    this.container = document.createElement("div");
    this.container.className = "bs-overlay";

    // Subtitle content container
    const contentArea = document.createElement("div");
    contentArea.className = "bs-content-area";
    this.container.appendChild(contentArea);

    this.shadow.appendChild(this.container);

    // Inject styles into shadow DOM
    const style = document.createElement("style");
    style.textContent = this._getStyles();
    this.shadow.appendChild(style);

    // Attach host to video container or body
    this._attachHost();

    // Create renderer inside content area
    this.renderer = new SubtitleRenderer(contentArea);

    // Initial scale calculation & resize observer
    this._setupResizeObserver();
    this._updateScale();

    // Listen for fullscreen changes
    document.addEventListener("fullscreenchange", this._onFullscreenChange);
    document.addEventListener("webkitfullscreenchange", this._onFullscreenChange);
    document.addEventListener("mozfullscreenchange", this._onFullscreenChange);
    if (typeof window !== "undefined") {
      window.addEventListener("resize", this._onWindowResize);
    }

    // Use MutationObserver instead of polling interval to re-attach if host is detached by page DOM updates
    this._domObserver = new MutationObserver(() => {
      if (this.isActive && this.host && !this.host.isConnected) {
        this._attachHost();
        this._setupResizeObserver();
        this._updateScale();
      }
    });
    try {
      this._domObserver.observe(document.body || document.documentElement, {
        childList: true,
        subtree: true,
      });
    } catch (e) {}

    this.isActive = true;
    console.log("[OverlayManager] Initialized inside video container");
  }

  attachToVideo(video) {
    this.targetVideo = video;
    if (this.isActive && this.host) {
      this._attachHost();
      this._setupResizeObserver();
      this._updateScale();
    }
  }

  _findPlayerContainer(video) {
    if (video) {
      // 1. Check known video player roots/wrappers (JWPlayer, YouTube, VideoJS, Plyr, etc.)
      const knownPlayer = video.closest(
        ".jw-wrapper, .jwplayer, #box, .html5-video-player, .video-js, .plyr, .dplayer, .bpx-player-video-wrap, .bilibili-player-video, .vjs-tech, [class*='player-wrapper'], [class*='player_container'], [class*='video-wrapper'], [class*='player-container'], [class*='video-player']"
      );
      if (knownPlayer) return knownPlayer;

      // 2. Direct parent or grandparent if parent is just a media wrapper (.jw-media, etc.)
      if (video.parentElement) {
        const parent = video.parentElement;
        if (parent.parentElement && parent.parentElement !== document.body && parent.parentElement !== document.documentElement) {
          const grandParent = parent.parentElement;
          if (
            grandParent.querySelector(".jw-controls, .controls, [class*='control'], [class*='overlay'], [class*='player']") ||
            grandParent.classList.contains("jwplayer") ||
            grandParent.classList.contains("jw-wrapper") ||
            grandParent.id === "box"
          ) {
            return grandParent;
          }
        }
        return parent;
      }
    }

    // 3. If no direct <video> in this frame (top frame hosting iframe), check for iframe player wrappers
    const iframeWrapper = document.querySelector(
      ".desktop.video-player, .video-player, #video, #player, .player-container, [class*='video-player'], [id*='video-player']"
    );
    if (iframeWrapper) return iframeWrapper;

    const iframe = document.querySelector("iframe[src*='embed'], iframe[src*='stream'], iframe[src*='player'], iframe[allowfullscreen]");
    if (iframe && iframe.parentElement && iframe.parentElement !== document.body) {
      return iframe.parentElement;
    }

    return null;
  }

  _attachHost() {
    if (!this.host) return;

    // 1. If document is in fullscreen mode
    const fsEl = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement;
    if (fsEl) {
      this.host.style.position = "absolute";
      this.host.style.top = "0";
      this.host.style.left = "0";
      this.host.style.width = "100%";
      this.host.style.height = "100%";
      this.host.style.zIndex = "2147483647";
      this.host.style.pointerEvents = "none";

      if (fsEl.tagName && fsEl.tagName.toLowerCase() === "video" && fsEl.parentElement) {
        const container = this._findPlayerContainer(fsEl) || fsEl.parentElement;
        if (this.host.parentNode !== container) {
          const pos = window.getComputedStyle(container).position;
          if (pos === "static") container.style.position = "relative";
          container.appendChild(this.host);
        }
      } else {
        if (this.host.parentNode !== fsEl) {
          fsEl.appendChild(this.host);
        }
      }
      return;
    }

    // 2. Normal mode: attach to target video's player container
    if (!this.targetVideo || !this.targetVideo.isConnected) {
      this.targetVideo = document.querySelector("video");
    }

    const container = this._findPlayerContainer(this.targetVideo);
    if (container) {
      const pos = window.getComputedStyle(container).position;
      if (pos === "static") {
        container.style.position = "relative";
      }

      this.host.style.position = "absolute";
      this.host.style.top = "0";
      this.host.style.left = "0";
      this.host.style.width = "100%";
      this.host.style.height = "100%";
      this.host.style.zIndex = "2147483647";
      this.host.style.pointerEvents = "none";

      if (this.host.parentNode !== container) {
        container.appendChild(this.host);
      }
      return;
    }

    // 3. Fallback: attach to body
    if (this.host.parentNode !== document.body) {
      this.host.style.position = "fixed";
      this.host.style.top = "0";
      this.host.style.left = "0";
      this.host.style.width = "100%";
      this.host.style.height = "100%";
      this.host.style.zIndex = "2147483647";
      this.host.style.pointerEvents = "none";
      document.body.appendChild(this.host);
    }
  }

  _getVideoDimensions() {
    let width = 0;
    let height = 0;

    if (this.targetVideo && this.targetVideo.isConnected) {
      const rect = this.targetVideo.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        width = rect.width;
        height = rect.height;
      }
    }

    if ((!width || !height) && this.host && this.host.isConnected) {
      const rect = this.host.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        width = rect.width;
        height = rect.height;
      }
    }

    if ((!width || !height) && typeof window !== "undefined") {
      if (document.fullscreenElement || window.innerHeight > 0) {
        width = window.innerWidth;
        height = window.innerHeight;
      }
    }

    return { width: width || 854, height: height || 480 };
  }

  _calculateFontSizes() {
    const { width, height } = this._getVideoDimensions();
    // Use height as the primary baseline for subtitle scaling.
    // For narrow/portrait aspect ratios (where width < height * 0.8, e.g. Shorts/TikTok),
    // clamp the effective height so subtitles don't overflow horizontally.
    const effectiveHeight = Math.min(height, width * 0.8);

    const origPct = this.origFontPercent || 2.5;
    const transPct = this.transFontPercent || 4.2;

    const origPx = Math.max(9, Math.min(60, (effectiveHeight * origPct) / 100));
    const transPx = Math.max(12, Math.min(90, (effectiveHeight * transPct) / 100));

    return { origPx, transPx, effectiveHeight, width, height };
  }

  _updateScale() {
    if (!this.container) return;
    const { origPx, transPx, effectiveHeight } = this._calculateFontSizes();
    this.container.style.setProperty("--bs-orig-size", `${origPx.toFixed(1)}px`);
    this.container.style.setProperty("--bs-trans-size", `${transPx.toFixed(1)}px`);
    this.container.style.setProperty("--bs-video-height", `${effectiveHeight.toFixed(0)}px`);
  }

  _setupResizeObserver() {
    if (typeof ResizeObserver === "undefined") return;
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }

    this._resizeObserver = new ResizeObserver(() => {
      this._updateScale();
    });

    if (this.targetVideo && this.targetVideo.isConnected) {
      this._resizeObserver.observe(this.targetVideo);
    }
    if (this.host && this.host.isConnected) {
      this._resizeObserver.observe(this.host);
    }
  }

  _onWindowResize() {
    this._updateScale();
  }

  _onFullscreenChange() {
    setTimeout(() => {
      this._attachHost();
      this._setupResizeObserver();
      this._updateScale();
    }, 100);
  }

  destroy() {
    if (!this.isActive) return;

    document.removeEventListener("fullscreenchange", this._onFullscreenChange);
    document.removeEventListener("webkitfullscreenchange", this._onFullscreenChange);
    document.removeEventListener("mozfullscreenchange", this._onFullscreenChange);
    if (typeof window !== "undefined") {
      window.removeEventListener("resize", this._onWindowResize);
    }

    if (this._resizeObserver) {
      try { this._resizeObserver.disconnect(); } catch (e) {}
      this._resizeObserver = null;
    }

    if (this._domObserver) {
      try { this._domObserver.disconnect(); } catch (e) {}
      this._domObserver = null;
    }

    if (this.host && this.host.parentNode) {
      this.host.parentNode.removeChild(this.host);
    }

    this.renderer = null;
    this.host = null;
    this.shadow = null;
    this.container = null;
    this.targetVideo = null;
    this.isActive = false;
    console.log("[OverlayManager] Destroyed");
  }

  setMode(mode) {
    // Kept for backward compatibility
  }

  applySettings(settings) {
    if (!settings) return;

    if (this.renderer) {
      this.renderer.applySettings(settings);
    }

    if (this.container) {
      if (settings.subPosY !== undefined && settings.subPosY !== null) {
        this.subPosY = settings.subPosY;
        this.container.style.setProperty("--bs-sub-bottom", `${settings.subPosY}%`);
      }
      if (settings.subWidth !== undefined && settings.subWidth !== null) {
        this.subWidth = settings.subWidth;
        this.container.style.setProperty("--bs-sub-width", `${settings.subWidth}%`);
      }
      if (settings.origFontSize !== undefined && settings.origFontSize !== null) {
        let val = parseFloat(settings.origFontSize);
        if (!isNaN(val)) {
          // Backward compatibility: if old px value (>= 8), convert to percent
          if (val >= 8) val = (val / 480) * 100;
          this.origFontPercent = Math.min(6.0, Math.max(0.8, val));
        }
      }
      if (settings.transFontSize !== undefined && settings.transFontSize !== null) {
        let val = parseFloat(settings.transFontSize);
        if (!isNaN(val)) {
          // Backward compatibility: if old px value (>= 8), convert to percent
          if (val >= 8) val = (val / 480) * 100;
          this.transFontPercent = Math.min(9.0, Math.max(1.5, val));
        }
      }
      if (settings.fontWeight) {
        this.container.style.setProperty("--bs-font-weight", settings.fontWeight);
      }
      if (settings.fontFamily && settings.fontFamily !== "default") {
        this.container.style.setProperty(
          "--bs-font-family",
          `"${settings.fontFamily}", "Noto Sans", "Noto Sans JP", "Noto Sans CJK JP", "Noto Sans SC", "Noto Sans CJK SC", sans-serif`
        );
      } else {
        this.container.style.removeProperty("--bs-font-family");
      }

      this._updateScale();
    }
  }

  // ── Transcript Events ────────────────────────────────────

  onUtteranceUpdate(payload) {
    if (!this.renderer) return;
    this.renderer.onUtteranceUpdate(payload);
  }

  onPartialTranscript(payload) {
    if (!this.renderer) return;
    const tokens = payload.tokens || [];
    this.renderer.onPartialTranscript(tokens);
  }

  onSentenceComplete(payload) {
    if (!this.renderer) return;
    this.renderer.onSentenceComplete(payload);
  }

  onTranslation(payload) {
    if (!this.renderer) return;
    // Chuyển NGUYÊN payload để renderer còn biết đây là mảnh dịch dở (`partial`) hay bản
    // hoàn chỉnh — cần cho tính năng "tắt chạy chữ" (chỉ hiện bản dịch 1 lần).
    this.renderer.onTranslation(payload);
  }

  clear() {
    if (this.renderer) {
      this.renderer.clear();
    }
  }

  // ── Styles ────────────────────────────────────────────────

  _getStyles() {
    return `
      .bs-overlay {
        --bs-orig-size: 14px;
        --bs-trans-size: 22px;
        --bs-font-weight: 600;
        --bs-font-family: "Noto Sans", "Noto Sans JP", "Noto Sans CJK JP", "Noto Sans SC", "Noto Sans CJK SC", "Inter", "Segoe UI", Arial, sans-serif;
        --bs-sub-bottom: 10%;
        --bs-sub-width: 80%;

        position: absolute;
        left: 50%;
        bottom: var(--bs-sub-bottom);
        transform: translateX(-50%);
        width: var(--bs-sub-width);
        max-width: var(--bs-sub-width);
        max-height: 45%;
        pointer-events: none !important;
        user-select: none !important;
        font-family: var(--bs-font-family);
        color: #ffffff;
        overflow-y: hidden;
        text-align: center;
        box-sizing: border-box;
        padding: 4px 12px;
        transition: bottom 0.15s ease, width 0.15s ease, max-width 0.15s ease;
      }
      .bs-content-area {
        pointer-events: none !important;
        user-select: none !important;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        position: relative;
      }
      
      /* ── 3-Layer Subtitle Architecture ───────────────────── */
      
      /* Layer 1: Lịch sử cũ (cuộn lên trên, font nhỏ hơn ~12%, mờ nhẹ) */
      .bs-history-layer {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        gap: 3px;
        opacity: 0.6;
        transition: opacity 0.3s ease;
        margin-bottom: 3px;
        overflow: hidden;
        pointer-events: none !important;
        user-select: none !important;
      }
      .bs-history-layer .bs-original {
        font-size: calc(var(--bs-orig-size) * 0.88);
        color: rgba(255, 255, 255, 0.8);
      }
      .bs-history-layer .bs-translated {
        font-size: calc(var(--bs-trans-size) * 0.88);
        color: #cbd5e1;
      }

      /* Layer 2: Tiêu điểm trung tâm CỐ ĐỊNH (Anchor Focus) */
      .bs-focus-layer {
        margin: 2px 0;
        min-height: calc(var(--bs-trans-size) + var(--bs-orig-size) + 4px);
        display: flex;
        flex-direction: column;
        justify-content: center;
        opacity: 1;
        flex-shrink: 0;
        pointer-events: none !important;
        user-select: none !important;
      }
      .bs-focus-layer .bs-original {
        font-size: var(--bs-orig-size);
        font-weight: calc(var(--bs-font-weight) - 100);
        color: #ffffff;
      }
      .bs-focus-layer .bs-translated {
        font-size: var(--bs-trans-size);
        font-weight: var(--bs-font-weight);
        color: #ffd866;
        text-shadow: 0 0 6px #000, 0 0 6px #000, 0 2px 4px #000;
      }

      /* Layer 3: Liveview đang nhận diện & chờ dịch (Dưới cùng, cố định chiều cao) */
      .bs-live-layer {
        margin-top: 2px;
        min-height: calc(var(--bs-orig-size) + 6px);
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0.85;
        flex-shrink: 0;
        transition: opacity 0.2s ease;
        pointer-events: none !important;
        user-select: none !important;
      }
      .bs-live-layer:empty {
        visibility: hidden;
      }
      .bs-live-layer .bs-sentence {
        display: flex;
        align-items: center;
        gap: 4px;
        justify-content: center;
        margin: 0;
        padding: 0;
      }
      .bs-live-layer .bs-original {
        font-size: calc(var(--bs-orig-size) * 0.88);
        color: #94a3b8;
        font-style: italic;
        display: inline;
      }
      .bs-live-layer .bs-translating {
        font-size: calc(var(--bs-orig-size) * 0.95);
        color: #fbbf24;
        font-style: normal;
        display: inline-block;
        margin-left: 4px;
        animation: bs-pulse 1.2s ease-in-out infinite;
      }

      .bs-history-layer .bs-history-item {
        animation: bs-history-in 0.25s ease-out;
      }
      @keyframes bs-history-in {
        from {
          opacity: 0;
          transform: translateY(6px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      .bs-sentence {
        margin-bottom: 2px;
        padding: 1px 0;
        max-height: 50vh;
        transition: transform 0.25s ease, opacity 0.3s ease, max-height 0.35s ease, margin 0.35s ease, padding 0.35s ease;
        pointer-events: none !important;
        user-select: none !important;
      }
      .bs-sentence.bs-fading-out {
        opacity: 0 !important;
        transition: opacity 0.5s ease-out !important;
      }
      .bs-sentence.bs-evicting {
        opacity: 0 !important;
        transform: translateY(-8px) scale(0.96) !important;
        max-height: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        overflow: hidden !important;
        animation: none !important;
        transition: opacity 0.35s ease-out, transform 0.35s ease-out, max-height 0.35s ease-in-out, margin 0.35s ease, padding 0.35s ease !important;
      }
      .bs-original { 
        font-size: var(--bs-orig-size); 
        font-weight: calc(var(--bs-font-weight) - 100); 
        line-height: 1.35; 
        color: #ffffff; 
        letter-spacing: 0.3px; 
        word-break: break-word; 
        overflow-wrap: break-word; 
        text-shadow: 0 0 4px #000, 0 0 4px #000, 0 1px 2px #000;
        pointer-events: none !important;
        user-select: none !important;
      }
      .bs-translated { 
        font-size: var(--bs-trans-size); 
        font-weight: var(--bs-font-weight); 
        line-height: 1.35; 
        color: #ffd866; 
        margin-top: 2px; 
        word-break: break-word; 
        overflow-wrap: break-word; 
        text-shadow: 0 0 5px #000, 0 0 5px #000, 0 2px 3px #000;
        pointer-events: none !important;
        user-select: none !important;
      }
      .bs-translating { 
        font-size: calc(var(--bs-orig-size) * 0.95); 
        color: #fbbf24; 
        font-style: italic; 
        margin-top: 1px; 
        animation: bs-pulse 1.2s ease-in-out infinite; 
        pointer-events: none !important;
        user-select: none !important;
      }
      @keyframes bs-pulse { 0%,100%{opacity:0.3} 50%{opacity:1} }
      .bs-overlay::-webkit-scrollbar { width: 3px; }
      .bs-overlay::-webkit-scrollbar-track { background: transparent; }
      .bs-overlay::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 2px; }
    `;
  }
}

if (typeof window !== "undefined") {
  window.OverlayManager = OverlayManager;
}
