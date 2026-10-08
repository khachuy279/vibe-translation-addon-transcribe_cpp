/**
 * lookahead-client.js — Lookahead WebSocket Client & Audio Ingress Bridge (v2)
 *
 * Kết nối Extension với endpoint Backend `/ws/lookahead`:
 * 1. Nhận mảnh âm thanh từ Injected Script (Buffer Interceptor), giữ ĐÚNG thứ tự append.
 * 2. Đóng gói Binary Frame (4-byte header length + JSON + Raw Bytes) gửi sang Backend,
 *    kèm `is_init` / `epoch` để backend biết khi nào phải dựng lại bộ ghép nối.
 * 3. Gửi heartbeat `sync_state` định kỳ 500ms (currentTime / playbackRate / paused).
 * 4. Nạp phụ đề đã dịch trước vào `SubtitleTimelineQueue` để hiện đúng lúc video phát tới.
 *
 * v2 sửa các lỗi khiến bản v1 "mất phụ đề gốc và bản dịch":
 *  - KHÔNG resume video chỉ vì nhận được câu phụ đề ĐẦU TIÊN (câu đó có thể ở tận đâu).
 *    Chỉ resume khi backend báo `prebuffer_ready` (đã phủ đủ `lead_time` phía trước).
 *  - Gửi lại yêu cầu replay mảnh audio sau MỖI lần tua video, để vùng mới có dữ liệu ngay.
 *  - Chuyển tiếp `lookahead_unavailable` để content script tự động quay về Pipeline A.
 */

// Trình duyệt: manifest nạp `lib/frame-builder.js` TRƯỚC file này trong cùng
// `content_scripts.js[]`, nên `buildBinaryAudioPacket` đã là biến toàn cục.
//
// Node (test): mô phỏng ĐÚNG thứ tự đó. Nếu thiếu, `_sendAudioBinaryFrame` ném ReferenceError
// và test sẽ TREO thay vì báo lỗi rõ ràng (đã xảy ra thật với `buffer-interceptor.test.js`
// khi chỗ dựng khung được hợp nhất về `frame-builder.js`). Khối này không chạy trong trình duyệt.
if (typeof module !== "undefined" && module.exports && typeof buildBinaryAudioPacket !== "function") {
  require("./frame-builder.js");
}

class LookaheadClient {
  constructor(options = {}) {
    this.serverUrl = options.serverUrl || "wss://localhost:8765/ws/lookahead";
    this.timelineQueue = options.timelineQueue || null;
    this.leadTime = options.leadTime || 15;
    this.sourceLang = options.sourceLang || "auto";
    this.targetLang = options.targetLang || "vi";
    // Cấu hình phiên đầy đủ từ popup (VAD/ASR/phân câu/dịch) — gửi kèm `lookahead_init`.
    this.config = options.config || null;

    // Callbacks
    this.onPrebufferReady = options.onPrebufferReady || null;
    this.onStatus = options.onStatus || null;
    this.onReady = options.onReady || null;
    this.onUnavailable = options.onUnavailable || null;
    this.onError = options.onError || null;
    this.onClose = options.onClose || null;
    //: Nhận audio lồng tiếng đã tổng hợp: (header, arrayBufferWav) => void
    this.onTts = options.onTts || null;
    //: Nhận danh sách phụ đề đã dịch: (items, seekId) => void
    this.onSubtitles = options.onSubtitles || null;

    this.videoElement = null;
    this.ws = null;
    this.port = null;
    // Cho phép dùng Background Bridge để vượt qua CSP / CORS của các trang web ngoài.
    // Mặc định per-browser nằm ở MỘT chỗ: `lib/browser-config.js` (sinh bởi
    // `tools/build_extensions.py`), KHÔNG hard-code ở đây nữa:
    //   • FIREFOX      — bridge là đường CHÍNH (background script không bị kill sau ~30s idle).
    //   • CHROME/EDGE  — service worker bị kill sau ~30s idle ⇒ WebSocket mở thẳng.
    // `options.useBridge` vẫn được ưu tiên cao nhất (test truyền tường minh).
    // Thiếu config (vd Node test) ⇒ giữ nguyên cách dò cũ `chrome.runtime.connect`.
    const bsCfg = globalThis.BS_BROWSER_CONFIG;
    this.useBridge = (typeof options.useBridge === "boolean")
      ? options.useBridge
      : (bsCfg
        ? bsCfg.useBridge === true
        : (typeof chrome !== "undefined" && !!chrome.runtime?.connect));

    this.isConnected = false;
    this.isServerReady = false;
    this.activeSeekId = "init_0";
    this.clientSessionId = `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;

    //: Đã Stop/ngắt phiên ⇒ MỌI đường gửi/nhận phải là no-op (chống rò audio lên server
    //: sau khi người dùng bấm Stop, kể cả khi socket đang ở trạng thái CONNECTING).
    this._destroyed = false;

    this._pendingChunks = [];
    this._syncTimer = null;
    this._messageListener = null;
    this._readyTimer = null;

    // ── THỐNG KÊ CHẨN ĐOÁN ────────────────────────────────────────────────────
    // Người dùng báo (2026-10-05): POPUP báo "Lookahead available" mà backend log "Đã dịch 0.0s,
    // PCM/byte 0.000" — cần biết extension có THỰC SỰ gửi mảnh nào sang `/ws/lookahead` không.
    // Bốn con số dưới đây trả lời chính xác điều đó:
    //   framesSent   — số khung nhị phân đã gửi THÀNH CÔNG sang backend
    //   framesQueued — số mảnh phải xếp hàng chờ socket mở
    //   framesDropped— số mảnh bị bỏ vì hàng đợi đầy (>160) ⇒ MẤT AUDIO
    //   framesBlocked— số mảnh KHÔNG gửi được vì socket không mở/destroyed
    this.diag = {
      framesSent: 0,
      framesSentBytes: 0,
      framesQueued: 0,
      framesDropped: 0,
      framesBlocked: 0,
      jsonSent: 0,
      serverMessages: Object.create(null),
      connectedAt: null,
      lastFrameAt: 0,
      lastFrameLogAt: 0,
    };
  }

  /** Ghi một dòng chẩn đoán có tiền tố riêng để lọc nhanh trong Console. */
  _diagLog(level, message, data) {
    const prefix = "%c[Lookahead Client][Diag]";
    try {
      const style = level === "warn" ? "color:#f59e0b;" : "color:#38bdf8;";
      // KHÔNG truyền thêm tham số khi không có dữ liệu: Console in ra `<empty string>` rất khó đọc.
      if (data === undefined || data === null) {
        if (level === "warn") console.warn(`${prefix} ${message}`, style);
        else console.log(`${prefix} ${message}`, style);
        return;
      }
      if (level === "warn") console.warn(`${prefix} ${message}`, style, data);
      else console.log(`${prefix} ${message}`, style, data);
    } catch (e) {}
  }

  /** Ảnh chụp thống kê gửi/nhận (dùng cho POPUP và `__bsLookaheadDiagReport`). */
  getDiag() {
    return {
      ...this.diag,
      serverMessages: { ...this.diag.serverMessages },
      isConnected: this.isConnected,
      isServerReady: this.isServerReady,
      pendingChunks: this._pendingChunks.length,
      activeSeekId: this.activeSeekId,
      clientSessionId: this.clientSessionId,
    };
  }

  /**
   * Bắt đầu phiên kết nối Lookahead.
   */
  connect(video) {
    this._destroyed = false;
    this.videoElement = video;
    this._pendingChunks = [];
    this.activeSeekId = "init_0";
    this.clientSessionId = `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
    // Thống kê của phiên trước không còn ý nghĩa — đếm lại từ đầu (xem `this.diag`).
    this.diag.framesSent = 0;
    this.diag.framesSentBytes = 0;
    this.diag.framesQueued = 0;
    this.diag.framesDropped = 0;
    this.diag.framesBlocked = 0;
    this.diag.jsonSent = 0;
    this.diag.serverMessages = Object.create(null);
    this.diag.connectedAt = null;
    this._diagLog("log", `Mở phiên Lookahead mới (session=${this.clientSessionId}) tới ${this.serverUrl} `
      + `| video.currentTime=${video ? Number(video.currentTime || 0).toFixed(2) : "?"}s`);
    if (this.useBridge) {
      this._initBridge();
    } else {
      this._initWebSocket();
    }
    this._initMessageBridge();
    this._startSyncLoop();

    // Nếu backend không phản hồi `lookahead_ready` trong 4s, coi như sẵn sàng để KHÔNG
    // bao giờ treo trình phát (backend cũ / phiên bản lệch).
    if (this._readyTimer) clearTimeout(this._readyTimer);
    this._readyTimer = setTimeout(() => {
      if (!this.isServerReady && !this._destroyed && this.isConnected) {
        this.isServerReady = true;
        if (this.onReady) this.onReady();
      }
    }, 4000);
  }

  /**
   * Ngắt phiên: dừng mọi đường gửi/nhận rồi đóng socket.
   */
  disconnect() {
    this._destroyed = true;      // chặn NGAY mọi frame audio/sync tiếp theo
    this._stopSyncLoop();
    this._pendingChunks = [];
    if (this._readyTimer) {
      clearTimeout(this._readyTimer);
      this._readyTimer = null;
    }
    if (this._messageListener && typeof window !== "undefined" && typeof window.removeEventListener === "function") {
      window.removeEventListener("message", this._messageListener);
      this._messageListener = null;
    }

    if (typeof window !== "undefined" && typeof window.postMessage === "function") {
      try {
        window.postMessage({
          source: "VIBE_LOOKAHEAD_CLIENT",
          type: "RESET_REPLAY_TOKEN",
        }, "*");
      } catch (e) {}
    }

    // Dọn dẹp port Background Bridge nếu có
    const port = this.port;
    this.port = null;
    if (port) {
      try {
        port.postMessage({ action: "SEND_JSON", data: { type: "stop", reason: "client_stop" } });
      } catch (e) {}
      try {
        port.postMessage({ action: "DISCONNECT" });
      } catch (e) {}
      try {
        port.disconnect();
      } catch (e) {}
    }

    // Dọn dẹp WebSocket trực tiếp nếu có
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      // Báo backend teardown tường minh TRƯỚC khi đóng (không phụ thuộc close-frame).
      try {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "stop", reason: "client_stop" }));
        }
      } catch (e) {}
      try {
        if (ws.readyState === WebSocket.CONNECTING) {
          // `close()` khi đang CONNECTING: đảm bảo socket KHÔNG BAO GIỜ mở rồi gửi dữ liệu.
          ws.onopen = () => { try { ws.close(); } catch (e) {} };
          ws.onmessage = null;
          ws.onerror = null;
        } else {
          ws.onmessage = null;
          ws.onclose = null;
          ws.onerror = null;
        }
        ws.close();
      } catch (e) {}
    }
    this.isConnected = false;
    this.isServerReady = false;
  }

  /**
   * Cập nhật cấu hình phiên khi người dùng đổi thiết lập trong popup (đang chạy).
   * Chỉ áp những gì backend Lookahead hỗ trợ: VAD, ngôn ngữ, phân câu, model.
   */
  updateConfig(config) {
    if (!config || this._destroyed) return;
    this.config = config;
    if (config.sourceLang) this.sourceLang = config.sourceLang;
    if (config.targetLang) this.targetLang = config.targetLang;
    this.sendJSON({ ...config, type: "set_config", action: "configure" });
  }

  /**
   * Yêu cầu injected script replay các mảnh đã đệm, CHỈ MỘT LẦN cho mỗi mốc (token).
   *
   * Sự cố thật 2026-10-01: hàm này bị gọi từ 3 nơi (`_handleConnected`, `lookahead_ready`,
   * `seek_acknowledged`) nên cùng một cache bị append 2-3 lần; riêng lần đầu còn cộng thêm
   * `_pendingChunks` ⇒ backend nhận ~5 bản sao của cùng đoạn audio. Đo được: 15.851 frame
   * AAC (368 s audio) chỉ trải trên 70 s timeline ⇒ decoder phải giải mã và lọc trùng gấp 5,
   * làm đói cả pipeline.
   *
   * `token`: "init_<seekId>" cho lần bắt đầu phiên (gửi bao nhiêu lần cũng chỉ replay 1 lần),
   * và `seek_<id>` cho mỗi lần tua (mỗi mốc mới được replay đúng một lần).
   */
  requestInitialChunks(currentTime, token, force = false) {
    if (this._destroyed) return;
    try {
      const curTime = currentTime !== undefined
        ? currentTime
        : (this.videoElement ? Number(this.videoElement.currentTime.toFixed(2)) : 0);
      if (typeof window !== "undefined" && typeof window.postMessage === "function") {
        window.postMessage({
          source: "VIBE_LOOKAHEAD_CLIENT",
          type: "REQUEST_INITIAL_CHUNKS",
          currentTime: curTime,
          replayToken: token || `init_${this.clientSessionId}_${this.activeSeekId}`,
          force: Boolean(force),
        }, "*");
      }
    } catch (e) {}
  }

  _initBridge() {
    const api = typeof browser !== "undefined" ? browser : chrome;
    try {
      console.log(`%c[Lookahead Client] 🌉 Khởi tạo kết nối qua Background Bridge (${this.serverUrl})...`, "color: #38bdf8;");
      this.port = api.runtime.connect({ name: "bs-lookahead-bridge" });

      this.port.onMessage.addListener((msg) => {
        if (this._destroyed || !msg) return;

        if (msg.type === "connected") {
          this._handleConnected();
        } else if (msg.type === "ws_json") {
          this._handleServerMessage(msg.data);
        } else if (msg.type === "ws_binary") {
          this._handleBinaryFrame(msg.data);
        } else if (msg.type === "error") {
          const errMsg = msg.error || "WebSocket connection error";
          console.error(`[Lookahead Client] ❌ Lỗi kết nối WebSocket từ Background Bridge:`, errMsg, msg.details || "");
          if (this.onError) this.onError(errMsg);
          if (!this.isConnected && this.onUnavailable) {
            this.onUnavailable(`bridge_error: ${errMsg}`);
          }
        } else if (msg.type === "disconnected") {
          this.isConnected = false;
          console.warn(`[Lookahead Client] 🔴 Mất kết nối WebSocket (Bridge, code: ${msg.code}, reason: ${msg.reason || "none"})`);
          if (this._readyTimer) { clearTimeout(this._readyTimer); this._readyTimer = null; }
          if (this.onClose) this.onClose(msg);
          if (this.onUnavailable && !this.isServerReady) {
            this.onUnavailable(`bridge_disconnected: code ${msg.code}`);
          }
        }
      });

      this.port.onDisconnect.addListener(() => {
        this.isConnected = false;
        this.port = null;
        if (this._destroyed) return;
        console.warn("[Lookahead Client] ⚠️ Cổng Background Bridge bị đóng (port disconnected)");
        if (this._readyTimer) { clearTimeout(this._readyTimer); this._readyTimer = null; }
        if (this.onError) this.onError("Background bridge port closed");
        if (this.onUnavailable && !this.isServerReady) {
          this.onUnavailable("bridge_port_closed");
        }
      });

      this.port.postMessage({ action: "CONNECT", url: this.serverUrl });
    } catch (e) {
      console.error("[Lookahead Client] ❌ Không thể tạo kết nối Background Bridge:", e);
      if (this.onError) this.onError(e);
      if (this.onUnavailable) this.onUnavailable(`bridge_init_failed: ${e.message}`);
    }
  }

  _handleConnected() {
    if (this._destroyed) {
      this.disconnect();
      return;
    }
    this.isConnected = true;
    this.diag.connectedAt = Date.now();
    console.log("%c[Lookahead Client] 🟢 Đã kết nối thành công tới /ws/lookahead", "color: #38bdf8; font-weight: bold;");
    this._diagLog("log", `Socket ĐÃ MỞ. Mảnh đang xếp hàng chờ gửi: ${this._pendingChunks.length}. `
      + `Sẽ gửi lookahead_init rồi xin replay cache từ interceptor.`);

    const curTime = this.videoElement ? Number(this.videoElement.currentTime.toFixed(2)) : 0;

    this.sendJSON({
      ...(this.config || {}),
      type: "lookahead_init",
      source_lang: this.sourceLang,
      target_lang: this.targetLang,
      lead_time: this.leadTime,
      current_time: curTime,
      currentTime: curTime,
      seek_id: this.activeSeekId,
    });

    // Flush các mảnh đã nhận TRƯỚC khi socket mở — CHỈ những mảnh KHÔNG nằm trong cửa sổ
    // cache sắp được replay (nếu không sẽ append trùng đúng đoạn đó 2 lần).
    if (this._pendingChunks.length > 0) {
      const cur = this.videoElement ? Number(this.videoElement.currentTime) : 0;
      const kept = [];
      for (const item of this._pendingChunks) {
        const inCacheWindow = payloadInReplayWindow(item.payload, cur - 8, cur + 45);
        if (!inCacheWindow) kept.push(item);
      }
      const skipped = this._pendingChunks.length - kept.length;
      this._pendingChunks = [];
      if (kept.length > 0) {
        console.log(`%c[Lookahead Client] 📤 Nạp ${kept.length} mảnh audio đệm sang backend ` +
          `(bỏ ${skipped} mảnh đã nằm trong cửa sổ cache).`, "color: #a78bfa;");
        for (const item of kept) {
          this._sendAudioBinaryFrame(item.rawBytes, item.payload);
        }
      } else if (skipped > 0) {
        console.log(`[Lookahead Client] Bỏ ${skipped} mảnh đệm trùng với cửa sổ cache.`);
      }
    }

    // Yêu cầu injected script replay init segment + các mảnh đã đệm (1 lần cho mốc này).
    this.requestInitialChunks(curTime, `init_${this.clientSessionId}_${this.activeSeekId}`, true);
  }

  _initWebSocket() {
    try {
      this.ws = new WebSocket(this.serverUrl);
      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => {
        this._handleConnected();
      };

      this.ws.onmessage = (event) => {
        if (this._destroyed) return;
        const data = event.data;
        if (typeof data === "string") {
          const msg = jsonParseSafe(data);
          if (msg) this._handleServerMessage(msg);
          return;
        }
        // Khung NHỊ PHÂN: [4B header_len][JSON header][WAV audio] — dùng cho lồng tiếng.
        if (data instanceof ArrayBuffer) {
          this._handleBinaryFrame(data);
          return;
        }
        // Firefox/Chrome có thể trả Blob nếu `binaryType` không được áp như mong đợi.
        if (typeof Blob !== "undefined" && data instanceof Blob) {
          data.arrayBuffer().then((buf) => this._handleBinaryFrame(buf)).catch(() => {});
          return;
        }
        if (data && typeof data.byteLength === "number") {
          try {
            this._handleBinaryFrame(
              data.buffer ? data.buffer.slice(data.byteOffset || 0, (data.byteOffset || 0) + data.byteLength) : data
            );
          } catch (e) {}
        }
      };

      this.ws.onclose = (event) => {
        this.isConnected = false;
        if (this._destroyed) return;
        console.log(`%c[Lookahead Client] 🔴 Disconnected (Direct WS, code: ${event?.code})`, "color: #f87171;");
        if (this.onClose) this.onClose(event);
        if (this.onUnavailable && !this.isServerReady) {
          this.onUnavailable(`ws_closed: code ${event?.code}`);
        }
      };

      this.ws.onerror = (err) => {
        if (this._destroyed) return;
        console.error("[Lookahead Client] ❌ Direct WS Error:", err);
        if (this.onError) this.onError(err);
        if (this.onUnavailable && !this.isServerReady) {
          this.onUnavailable("ws_error");
        }
      };
    } catch (e) {
      console.error("[Lookahead Client] ❌ Direct WS Connection failed:", e);
      if (this.onError) this.onError(e);
      if (this.onUnavailable && !this.isServerReady) {
        this.onUnavailable(`ws_init_failed: ${e.message}`);
      }
    }
  }

  _handleServerMessage(msg) {
    if (!msg) return;
    // ── ĐẾM & LOG LOẠI BẢN TIN TỪ BACKEND (chẩn đoán) ─────────────────────────
    // Câu hỏi thường gặp khi Pipeline B "chạy mà không có phụ đề": backend có gửi
    // `lookahead_status`/`lookahead_subtitles` không? Đây là chỗ trả lời.
    {
      const t = msg.type || "(không có type)";
      const first = !this.diag.serverMessages[t];
      this.diag.serverMessages[t] = (this.diag.serverMessages[t] || 0) + 1;
      if (first) {
        this._diagLog("log", `📥 Bản tin ĐẦU TIÊN từ backend: type=${t}`,
          t === "lookahead_status" ? {
            ready_ahead: msg.ready_ahead, fed_ahead: msg.fed_ahead,
            ready_until_pts: msg.ready_until_pts, prebuffer_ready: msg.prebuffer_ready,
            fragments: msg.fragments, decoded_chunks: msg.decoded_chunks,
          } : undefined);
      }
    }

    switch (msg.type) {
      case "lookahead_ready": {
        this.isServerReady = true;
        if (this._readyTimer) { clearTimeout(this._readyTimer); this._readyTimer = null; }
        if (msg.lead_time) this.leadTime = msg.lead_time;
        // Neo lại mốc thời gian. KHÔNG xin replay ở đây: `_handleConnected` đã xin (cùng
        // token) và interceptor dedup theo token ⇒ tránh append cache lần thứ hai.
        this._sendSyncNow();
        if (this.onReady) this.onReady();
        break;
      }
      case "lookahead_unavailable": {
        console.warn("[Lookahead Client] ⚠️ Backend không chạy được Pipeline B:", msg.reason || "");
        if (this.onUnavailable) this.onUnavailable(msg.reason || "lookahead_unavailable");
        break;
      }
      case "lookahead_subtitles": {
        const items = msg.items || [];
        if (this.timelineQueue && items.length) {
          this.timelineQueue.addSubtitles(items, msg.seek_id);
        }
        if (this.onSubtitles && items.length) {
          this.onSubtitles(items, msg.seek_id);
        }
        break;
      }
      case "lookahead_status": {
        // Bỏ qua trạng thái của đoạn CŨ sau khi vừa tua: nếu không, `prebuffer_ready` cũ sẽ
        // làm video phát lại ngay trước khi backend kịp dịch trước vị trí mới.
        if (msg.seek_id && msg.seek_id !== this.activeSeekId) {
          if (!this._warnedStaleStatus) {
            this._warnedStaleStatus = true;
            console.log(
              `[Lookahead Client] Bỏ qua lookahead_status của seek cũ (${msg.seek_id} != ${this.activeSeekId})`
            );
          }
          return;
        }
        if (this.onStatus) this.onStatus(msg);
        if (msg.prebuffer_ready && this.onPrebufferReady) {
          this.onPrebufferReady(msg);
        }
        break;
      }
      case "lookahead_tts": {
        // Audio lồng tiếng gửi kèm base64 (đường JSON — đường đã được chứng minh chạy được
        // ở mọi trình duyệt, không phụ thuộc `ws.binaryType`).
        if (!msg.audio_b64) return;
        const buf = base64ToArrayBuffer(msg.audio_b64);
        if (buf && this.onTts) this.onTts(msg, buf);
        break;
      }
      case "seek_acknowledged": {
        // Sau khi backend xác nhận tua: xin lại dữ liệu vùng mới. Token theo seek_id mới nên
        // interceptor THỰC SỰ replay lại (khác với các lần gọi lặp của cùng một mốc).
        this.requestInitialChunks(undefined, `seek_${this.activeSeekId}`);
        break;
      }
      case "lookahead_request_replay": {
        // Backend báo audio nhận được lệch xa vị trí phát (thường sau khi tua): yêu cầu
        // interceptor gửi lại mảnh quanh vị trí phát. Token DUY NHẤT để không bị dedup chặn.
        console.log(
          `[Lookahead Client] 🔁 Backend xin gửi lại mảnh quanh ${msg.currentTime ?? "?"}s ` +
          `(lý do: ${msg.reason || "?"})`
        );
        this.requestInitialChunks(
          msg.currentTime !== undefined ? Number(msg.currentTime) : undefined,
          `replay_${Date.now()}_${this.activeSeekId}`
        );
        break;
      }
      default:
        break;
    }
  }

  /**
   * Phân tích khung nhị phân từ backend: `[4B header_len][JSON header][payload]`.
   * Hiện dùng cho audio lồng tiếng (`type: "lookahead_tts"`).
   */
  _handleBinaryFrame(buffer) {
    if (this._destroyed || buffer.byteLength < 5) return;
    try {
      const view = new DataView(buffer);
      const hdrLen = view.getUint32(0, true);
      if (!hdrLen || hdrLen > 8192 || 4 + hdrLen > buffer.byteLength) return;
      const header = jsonParseSafe(new TextDecoder().decode(new Uint8Array(buffer, 4, hdrLen)));
      if (!header) return;
      if (header.type === "lookahead_tts") {
        if (this.onTts) this.onTts(header, buffer.slice(4 + hdrLen));
      }
    } catch (e) {
      console.warn("[Lookahead Client] Khung nhị phân không đọc được:", e);
    }
  }

  /**
   * Lắng nghe audio chunk từ Injected Script qua window.postMessage.
   */
  _initMessageBridge() {
    this._messageListener = (event) => {
      if (this._destroyed) return;
      if (event.source !== window || !event.data) return;
      const { source, type, payload, rawBytes } = event.data;

      if (source === "VIBE_LOOKAHEAD_POC" && type === "AUDIO_CHUNK_INTERCEPTED") {
        if (!rawBytes) return;
        if (this.isConnected && (this.port || this.ws?.readyState === WebSocket.OPEN)) {
          this._sendAudioBinaryFrame(rawBytes, payload);
        } else {
          this._pendingChunks.push({ rawBytes, payload });
          this.diag.framesQueued += 1;
          // Hàng đợi đầy ⇒ BỎ mảnh cũ nhất. Đây là MẤT AUDIO thật (không phải lỗi giải mã).
          if (this._pendingChunks.length > 160) {
            this._pendingChunks.shift();
            this.diag.framesDropped += 1;
            if (this.diag.framesDropped === 1 || this.diag.framesDropped % 20 === 0) {
              this._diagLog("warn",
                `⚠️ Hàng đợi mảnh audio đầy (>160) — đã BỎ ${this.diag.framesDropped} mảnh cũ. `
                + `Nguyên nhân: socket tới /ws/lookahead chưa mở hoặc đang nghẽn.`);
            }
          }
        }
      }
    };
    if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
      window.addEventListener("message", this._messageListener);
    }
  }

  /**
   * Đóng gói và gửi Binary Frame sang Backend.
   */
  _sendAudioBinaryFrame(rawBytes, meta) {
    if (this._destroyed) {
      this.diag.framesBlocked += 1;
      return;
    }
    if (!this.port && (!this.ws || this.ws.readyState !== WebSocket.OPEN)) {
      // Socket chưa mở: mảnh này KHÔNG được gửi. Nếu con số này tăng liên tục mà backend không
      // nhận được gì ⇒ cổng kết nối có vấn đề (xem `framesQueued`/`framesDropped`).
      this.diag.framesBlocked += 1;
      return;
    }

    const hdrObj = {
      timestamp_offset: meta?.timestampOffset || 0.0,
      mime_type: meta?.mime || "audio/webm",
      seek_id: this.activeSeekId,
      is_init: !!meta?.isInit,
    };
    if (typeof meta?.epoch === "number") hdrObj.epoch = meta.epoch;
    //: Khoảng media THẬT trong `SourceBuffer.buffered` mà mảnh này chiếm (trục thời gian của
    //: TRÌNH DUYỆT). Backend dùng nó để NEO PCM đã giải mã về đúng trục của trình duyệt: với
    //: trang dùng `SourceBuffer.mode = "sequence"`, `tfdt` tuyệt đối, hoặc đổi `timestampOffset`
    //: ngay sau `appendBuffer`, mốc container KHÔNG khớp `video.currentTime` (đo thật
    //: 2026-10-03: PCM lệch +250s ⇒ Đã dịch 0,0s dù cache có audio).
    const mediaStart = Number(meta?.mediaStart);
    const mediaEnd = Number(meta?.mediaEnd);
    if (Number.isFinite(mediaStart) && Number.isFinite(mediaEnd) && mediaEnd > mediaStart) {
      hdrObj.media_start = mediaStart;
      hdrObj.media_end = mediaEnd;
    }
    //: Mảnh TẢI LẠI cho vùng đã buffer sẵn: byte có thể trùng mảnh cũ nên backend phải MIỄN
    //: dedup, nếu không nó bị bỏ và vùng đã tải trước không bao giờ có phụ đề.
    if (meta?.refetched) hdrObj.refetched = true;

    // Dùng CHUNG bộ dựng khung với `frame-builder.js` — manifest nạp file đó TRƯỚC file này
    // trong cùng `content_scripts.js[]`. Trước đây chỗ này tự dựng lại `[4B len][JSON][PCM]`,
    // tức là bản thứ 4 của cùng một định dạng ⇒ sửa một nơi là lệch ba nơi còn lại.
    // Trả về ArrayBuffer (không phải Uint8Array) — dùng trực tiếp làm payload.
    const packet = buildBinaryAudioPacket(hdrObj, rawBytes);

    if (this.port) {
      try {
        this.port.postMessage({ action: "SEND_RAW_BINARY", buffer: packet });
        this._noteFrameSent(rawBytes, meta);
      } catch (e) {
        this.diag.framesBlocked += 1;
        console.error("[Lookahead Client] Gửi audio packet qua Bridge thất bại:", e);
      }
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(packet);
        this._noteFrameSent(rawBytes, meta);
      } catch (e) {
        this.diag.framesBlocked += 1;
        console.error("[Lookahead Client] Gửi audio packet qua Direct WS thất bại:", e);
      }
    }
  }

  /** Ghi nhận một khung audio đã gửi thành công + in tóm tắt định kỳ (chẩn đoán). */
  _noteFrameSent(rawBytes, meta) {
    const d = this.diag;
    d.framesSent += 1;
    d.framesSentBytes += rawBytes ? rawBytes.byteLength : 0;
    d.lastFrameAt = Date.now();
    const isInit = !!(meta && meta.isInit);
    if (d.framesSent === 1 || isInit) {
      this._diagLog("log",
        `📤 Khung ${isInit ? "INIT" : "media"} đầu tiên đã gửi (${rawBytes ? rawBytes.byteLength : 0}B, `
        + `mime=${(meta && meta.mime) || "?"}, epoch=${(meta && meta.epoch) ?? "-"}, `
        + `media=${meta && meta.mediaStart !== undefined ? `${meta.mediaStart}→${meta.mediaEnd}s` : "chưa rõ"}, `
        + `seek_id=${(meta && meta.seek_id) || this.activeSeekId})`);
      return;
    }
    // Tóm tắt mỗi 5s: đủ để thấy audio có chảy sang backend hay không mà không ngập Console.
    const now = Date.now();
    if (now - d.lastFrameLogAt >= 5000) {
      d.lastFrameLogAt = now;
      this._diagLog("log",
        `📊 Đã gửi ${d.framesSent} khung / ${(d.framesSentBytes / (1024 * 1024)).toFixed(2)} MB `
        + `| bị chặn (socket chưa mở)=${d.framesBlocked}, xếp hàng=${d.framesQueued}, `
        + `bỏ do hàng đợi đầy=${d.framesDropped}`
        + `${d.framesDropped > 0 ? " ⚠️ MẤT AUDIO" : ""}`);
    }
  }

  /**
   * Thông báo tới Backend khi người dùng tua video.
   */
  notifySeek(seekId, targetTime) {
    this.activeSeekId = seekId;
    this.sendJSON({
      type: "seek_reset",
      seek_id: seekId,
      target_time: Number(targetTime.toFixed(2)),
    });
    // Việc xin lại mảnh audio được thực hiện khi nhận `seek_acknowledged` (đảm bảo backend
    // đã reset xong trước khi bơm dữ liệu mới).
  }

  /**
   * Đồng bộ trạng thái currentTime và playbackRate mỗi 500ms.
   */
  _startSyncLoop() {
    this._syncTimer = setInterval(() => this._sendSyncNow(), 500);
  }

  _sendSyncNow() {
    if (this._destroyed) return;
    if (!this.isConnected || !this.videoElement) return;
    this.sendJSON({
      type: "sync_state",
      currentTime: Number(this.videoElement.currentTime.toFixed(2)),
      playbackRate: Number(this.videoElement.playbackRate.toFixed(2)),
      paused: Boolean(this.videoElement.paused),
      seek_id: this.activeSeekId,
    });
  }

  _stopSyncLoop() {
    if (this._syncTimer) {
      clearInterval(this._syncTimer);
      this._syncTimer = null;
    }
  }

  sendJSON(obj) {
    if (this._destroyed) return;
    if (obj && obj.type) this.diag.jsonSent += 1;
    if (this.port) {
      try {
        this.port.postMessage({ action: "SEND_JSON", data: obj });
      } catch (e) {
        console.error("[Lookahead Client] Gửi JSON qua Bridge thất bại:", e);
      }
      return;
    }
    if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(obj));
      } catch (e) {
        console.error("[Lookahead Client] Gửi JSON qua Direct WS thất bại:", e);
      }
    }
  }
}

function jsonParseSafe(str) {
  try {
    return JSON.parse(str);
  } catch (e) {
    return null;
  }
}

/**
 * Mảnh audio có nằm trong cửa sổ replay [from, to] không?
 *
 * Ưu tiên MỐC MEDIA THẬT (`mediaStart`/`mediaEnd` — interceptor lấy từ `SourceBuffer.buffered`).
 * `videoPts` chỉ là vị trí phát TẠI LÚC append, không nói lên nội dung mảnh: trình phát tải
 * trước nên mảnh append lúc đang phát 5 s thường chứa media ở 30–60 s (xem
 * `content/buffer_interceptor_poc.js`). Chưa biết mốc ⇒ trả `false` (KHÔNG coi là "sẽ được
 * replay") để mảnh vẫn được gửi — thà gửi thừa còn hơn mất audio.
 */
function payloadInReplayWindow(payload, from, to) {
  if (!payload) return false;
  const start = Number(payload.mediaStart);
  const end = Number(payload.mediaEnd);
  if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
    return end > from && start < to;
  }
  const pts = Number(payload.videoPts);
  if (Number.isFinite(pts)) return pts >= from && pts <= to;
  return false;
}

/** Giải mã base64 (audio WAV của TTS) thành ArrayBuffer. */
function base64ToArrayBuffer(b64) {
  try {
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  } catch (e) {
    console.warn("[Lookahead Client] base64 audio lỗi:", e);
    return null;
  }
}

// Xuất module cho Extension
if (typeof module !== "undefined" && module.exports) {
  module.exports = { LookaheadClient, payloadInReplayWindow };
}
