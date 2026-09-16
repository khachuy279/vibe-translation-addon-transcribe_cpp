// WebSocket client for Extension (Content Script & Popup)
// Uses Background Port Bridge to bypass iframe sandbox / cross-origin CSP restrictions.

class WSClient {
  constructor(url = "wss://localhost:8765/ws") {
    this.url = url;
    this.port = null;
    this.ws = null;
    this.useBridge = typeof chrome !== "undefined" && !!chrome.runtime?.connect;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectDelay = 30000;
    this.baseReconnectDelay = 1000;
    this.pingInterval = null;
    this.listeners = new Map();
    this._textEncoder = new TextEncoder();
    // P3.5d: handle của timer reconnect. Trước đây handle không được lưu nên KHÔNG THỂ
    // huỷ: sau khi người dùng bấm Stop, timer vẫn bắn (tới 30s sau) và mở WebSocket +
    // port "zombie" không có chủ.
    this._reconnectTimer = null;
    // P3.2: trạng thái backpressure do service worker báo về.
    this.backpressureState = "ok";
    this.droppedAudioFrames = 0;
    // FIX-01: số liệu chẩn đoán nghẽn (do service worker đếm).
    this.backpressurePauseCount = 0;
    this.backpressureResumeCount = 0;
    this.backpressurePausedMs = 0;
    // P3.1: backend có gửi audio TTS dạng binary frame không (do backend quyết định
    // theo protocol version mà client khai báo).
    this.supportsBinaryTts = false;
  }

  // ── Connection ──────────────────────────────────────────

  async connect() {
    if (this.isConnected) return;

    if (this.useBridge) {
      return this._connectViaBridge();
    } else {
      return this._connectDirect();
    }
  }

  _connectViaBridge() {
    const api = typeof browser !== "undefined" ? browser : chrome;

    return new Promise((resolve, reject) => {
      let settled = false;

      try {
        this.port = api.runtime.connect({ name: "bs-ws-bridge" });

        this.port.onMessage.addListener((msg) => {
          if (!msg) return;

          if (msg.type === "connected") {
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this._startPing();
            this._emit("connected");
            if (!settled) {
              settled = true;
              resolve();
            }
          } else if (msg.type === "ws_json") {
            const data = msg.data;
            const payload = data.payload !== undefined ? data.payload : data;
            if (data.type === "connected" || data.type === "hello") {
              // Backend xác nhận phiên bản giao thức -> biết có được dùng binary TTS không.
              if (typeof data.binary_tts === "boolean") {
                this.supportsBinaryTts = data.binary_tts;
              }
            }
            this._emit(data.type, payload);
            this._emit("message", data);
          } else if (msg.type === "ws_binary") {
            // P3.1: audio TTS dạng binary frame (không base64, không JSON).
            this._emit("tts_binary", msg.data);
            this._emit("message", { type: "tts_binary", data: msg.data });
          } else if (msg.type === "backpressure") {
            // P3.2: service worker báo socket đang tắc.
            this.backpressureState = msg.state;
            this.droppedAudioFrames = msg.droppedFrames || 0;
            this.backpressurePauseCount = msg.pauseCount || 0;
            this.backpressureResumeCount = msg.resumeCount || 0;
            this.backpressurePausedMs = msg.pausedMs || 0;
            this._emit("backpressure", {
              state: msg.state,
              bufferedAmount: msg.bufferedAmount,
              droppedFrames: this.droppedAudioFrames,
              // FIX-01: số liệu để biết nghẽn thật hay chỉ spike ngắn.
              pauseCount: this.backpressurePauseCount,
              resumeCount: this.backpressureResumeCount,
              pausedMs: this.backpressurePausedMs,
            });
          } else if (msg.type === "disconnected") {
            this.isConnected = false;
            this._stopPing();
            this._emit("disconnected", { code: msg.code, reason: msg.reason });
            if (!settled) {
              settled = true;
              reject(new Error(`WebSocket connection closed (code: ${msg.code})`));
            } else {
              this._scheduleReconnect();
            }
          } else if (msg.type === "error") {
            const errMsg = msg.error || "WebSocket connection error";
            this._emit("error", errMsg);
            if (!settled) {
              settled = true;
              reject(new Error(errMsg));
            }
          }
        });

        this.port.onDisconnect.addListener(() => {
          this.isConnected = false;
          this._stopPing();
          if (!settled) {
            settled = true;
            reject(new Error("Background bridge disconnected"));
          }
        });

        this.port.postMessage({ action: "CONNECT", url: this.url });
      } catch (e) {
        if (!settled) {
          settled = true;
          reject(e);
        }
      }
    });
  }

  _connectDirect() {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = "arraybuffer";
        let settled = false;

        this.ws.onopen = () => {
          settled = true;
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this._startPing();
          this._emit("connected");
          resolve();
        };

        this.ws.onmessage = (event) => {
          this._handleMessage(event);
        };

        this.ws.onclose = (event) => {
          this.isConnected = false;
          this._stopPing();
          this._emit("disconnected", { code: event.code, reason: event.reason });
          if (!settled) {
            settled = true;
            reject(new Error(`WebSocket connection closed (code ${event.code})`));
          } else {
            this._scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          this._emit("error", error);
          if (!settled) {
            settled = true;
            reject(new Error("Lỗi kết nối tới " + this.url));
          }
        };
      } catch (e) {
        reject(e);
      }
    });
  }

  disconnect() {
    this._stopPing();
    this._cancelReconnect();   // P3.5d: huỷ timer reconnect còn treo
    if (this.port) {
      try {
        this.port.postMessage({ action: "DISCONNECT" });
        this.port.disconnect();
      } catch (e) {}
      this.port = null;
    }
    if (this.ws) {
      try {
        this.ws.onclose = null;
        this.ws.close();
      } catch (e) {}
      this.ws = null;
    }
    this.isConnected = false;
    this.backpressureState = "ok";
  }

  // P3.5d: huỷ timer reconnect. Không có hàm này thì sau khi Stop, timer vẫn bắn và mở
  // WebSocket + port "zombie" không có chủ (rò kết nối mỗi lần stop-sau-khi-drop).
  _cancelReconnect() {
    if (this._reconnectTimer !== null) {
      try {
        clearTimeout(this._reconnectTimer);
      } catch (e) {}
      this._reconnectTimer = null;
    }
  }

  _scheduleReconnect() {
    if (this.reconnectAttempts > 10) return;
    const delay = Math.min(
      this.baseReconnectDelay * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay
    );
    this.reconnectAttempts++;
    this._emit("reconnecting", { attempt: this.reconnectAttempts, delayMs: delay });

    this._cancelReconnect();
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      // Kiểm tra lại tại thời điểm bắn: có thể người dùng đã Stop hoặc đã kết nối lại.
      if (!this.isConnected && (this.port || this.ws || this.reconnectAttempts > 0)) {
        this.connect().catch(() => {});
      }
    }, delay);
  }

  _startPing() {
    this._stopPing();
    this.pingInterval = setInterval(() => {
      this.sendJSON({ type: "ping", timestamp: performance.now() });
    }, 30000);
  }

  _stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  sendBinary(pcmData, captureTimestamp, chunkIndex, isPreSpeech = false) {
    if (!this.isConnected || !pcmData) return;

    let byteLength = 0;
    let rawBuffer = null;

    if (pcmData instanceof ArrayBuffer) {
      byteLength = pcmData.byteLength;
      rawBuffer = pcmData;
    } else if (ArrayBuffer.isView(pcmData)) {
      byteLength = pcmData.byteLength;
      rawBuffer = pcmData.buffer.slice(pcmData.byteOffset, pcmData.byteOffset + pcmData.byteLength);
    } else if (pcmData && typeof pcmData === "object") {
      const buf = pcmData.buffer || pcmData;
      const offset = pcmData.byteOffset || 0;
      byteLength = pcmData.byteLength || (buf.byteLength ? buf.byteLength - offset : 0);
      rawBuffer = buf.slice ? buf.slice(offset, offset + byteLength) : buf;
    }

    if (!rawBuffer || byteLength === 0) return;

    const samples = Math.floor(byteLength / 2);
    const chunkDurationMs = Math.round((samples / 16000) * 1000) || 20;

    const header = {
      type: "audio_chunk",
      captureTimestamp,
      chunkIndex,
      sampleRate: 16000,
      channels: 1,
      bitDepth: 16,
      chunkDurationMs,
      samples,
      isPreSpeech,
    };

    if (this.port) {
      // Send via port bridge
      this.port.postMessage({
        action: "SEND_BINARY",
        header,
        pcmBuffer: rawBuffer
      });
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      if (!this._textEncoder) {
        this._textEncoder = new TextEncoder();
      }

      const buffer = (typeof buildBinaryAudioPacket === "function")
        ? buildBinaryAudioPacket(header, rawBuffer, this._textEncoder)
        : this._buildPacketFallback(header, rawBuffer);

      try {
        this.ws.send(buffer);
      } catch (e) {
        console.error("[WSClient] Send error:", e);
      }
    }
  }

  _buildPacketFallback(header, rawBuffer) {
    const enc = this._textEncoder || new TextEncoder();
    const headerBytes = enc.encode(JSON.stringify(header));
    const pcmBytes = new Uint8Array(rawBuffer);
    const totalSize = 4 + headerBytes.length + pcmBytes.byteLength;
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    view.setUint32(0, headerBytes.length, true);
    new Uint8Array(buffer, 4, headerBytes.length).set(headerBytes);
    new Uint8Array(buffer, 4 + headerBytes.length).set(pcmBytes);
    return buffer;
  }

  /**
   * FIX-01: "liveness probe" cho đường backpressure của bridge.
   *
   * Khi capture đang bị tạm dừng vì socket nghẽn, KHÔNG còn `SEND_BINARY` nào được gửi
   * nữa, nên service worker không có cơ hội tự kiểm tra lại. Content script gọi hàm này
   * định kỳ để hỏi "đã rút hết hàng đợi chưa?" — service worker sẽ nhả trạng thái tạm dừng
   * và báo "ok" khi `bufferedAmount` đã xuống dưới ngưỡng SOFT.
   *
   * Chỉ có ý nghĩa ở chế độ bridge (đường trực tiếp không giám sát `bufferedAmount`).
   * Trả về true nếu thực sự đã gửi được probe.
   */
  flushPending() {
    if (!this.port) return false;
    try {
      this.port.postMessage({ action: "FLUSH_PENDING" });
      return true;
    } catch (e) {
      return false;
    }
  }

  sendJSON(data) {
    if (!this.isConnected) return;
    if (this.port) {
      this.port.postMessage({ action: "SEND_JSON", data });
      return;
    }
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(data));
      } catch (e) {
        console.error("[WSClient] JSON send error:", e);
      }
    }
  }

  _handleMessage(event) {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        const payload = msg.payload !== undefined ? msg.payload : msg;
        this._emit(msg.type, payload);
        this._emit("message", msg);
      } catch (e) {
        console.error("[WSClient] Parse error:", e);
      }
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }

  off(event, callback) {
    const cbs = this.listeners.get(event);
    if (cbs) cbs.delete(callback);
  }

  _emit(event, data) {
    const cbs = this.listeners.get(event);
    if (cbs) {
      cbs.forEach((cb) => {
        try { cb(data); } catch (e) { console.error("[WSClient] Listener error:", e); }
      });
    }
  }
}

if (typeof self !== "undefined") self.WSClient = WSClient;
if (typeof window !== "undefined") window.WSClient = WSClient;
