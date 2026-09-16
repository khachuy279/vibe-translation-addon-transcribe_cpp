// Background Service Worker
// Provides a WebSocket Bridge so content scripts inside cross-origin iframes
// can reliably connect to the backend (wss://localhost:8765/ws) without being
// blocked by iframe CSP, sandbox, or Private Network Access restrictions.

const api = typeof browser !== "undefined" ? browser : chrome;
const textEncoder = new TextEncoder();

// P3.2: NGƯỠNG BACKPRESSURE. Trước đây `bufferedAmount` không được đọc ở đâu cả
// (0 lần trong toàn bộ extension), nên khi backend nghẽn thì client vẫn bơm
// 32 KB/s audio vào buffer gửi của WebSocket mà không có giới hạn => RAM tăng và
// độ trễ KHÔNG BAO GIỜ hồi phục (audio được gửi là audio của hàng chục giây trước).
//   - vượt SOFT: bỏ frame audio cũ đang xếp (chỉ giữ audio mới nhất) + báo content script
//   - vượt HARD: ngừng gửi audio, báo content script tạm dừng capture
//
// FIX-01 (P0, bug thật đã xác nhận): trước đây cờ `pausedByBackpressure` CHỈ được xoá
// bên trong `sendOrQueue()`, mà hàm đó chỉ chạy khi có `SEND_BINARY`. Khi content script
// nhận "paused" thì nó đặt `audioCapture.isCapturing = false` ⇒ worklet ngừng phát chunk
// ⇒ KHÔNG còn `SEND_BINARY` ⇒ cờ pause không bao giờ được xoá ⇒ **capture treo vĩnh viễn**
// (phải Stop/Start lại mới có tiếng). `FLUSH_PENDING` có implement nhưng KHÔNG có producer.
// Cách sửa: dùng máy trạng thái có hysteresis ở `lib/backpressure-gate.js` và để service
// worker tự kiểm tra lại theo timer (không phụ thuộc vào frame mới).
const SEND_BUFFER_SOFT_LIMIT = 128 * 1024;   // 128 KB ~ 4 giây audio 16kHz PCM16
const SEND_BUFFER_HARD_LIMIT = 512 * 1024;   // 512 KB ~ 16 giây
// Nhịp kiểm tra lại khi ĐANG tạm dừng. Chỉ đọc `bufferedAmount` (rẻ) nên 100 ms là an toàn;
// đủ nhanh để capture chạy lại mà không tích thêm độ trễ.
const BACKPRESSURE_RESUME_CHECK_MS = 100;

api.runtime.onConnect.addListener((port) => {
  if (port.name !== "bs-ws-bridge") return;

  let ws = null;
  let isClosed = false;
  let droppedFrames = 0;
  // Chỉ giữ frame audio MỚI NHẤT khi socket nghẽn: audio cũ đã vô dụng cho phụ đề realtime.
  let pendingAudio = null;
  // FIX-01: timer tự kiểm tra lại khi đang tạm dừng.
  let resumeTimer = null;

  // FIX-01: máy trạng thái thuần (test ở extension_firefox/tests/).
  // Nếu vì lý do nào đó module chưa được nạp, thoái hoá về "luôn gửi" (không bao giờ
  // tạm dừng capture) — mất kiểm soát tải còn hơn treo vĩnh viễn không có tiếng.
  const BP = (typeof BackpressureGate !== "undefined" && BackpressureGate) ? BackpressureGate : null;
  const gate = BP
    ? BP.createGate({ softLimit: SEND_BUFFER_SOFT_LIMIT, hardLimit: SEND_BUFFER_HARD_LIMIT })
    : {
        evaluate: () => "send",
        isPaused: () => false,
        reset: () => {},
        snapshot: () => ({ paused: false, pauseCount: 0, resumeCount: 0, droppedFrames: 0, pausedMs: 0 }),
        actions: { SEND: "send", DROP: "drop", PAUSE: "pause", HOLD: "hold", RESUME: "resume" },
      };
  if (!BP) {
    console.error("[BS Background] Thiếu lib/backpressure-gate.js — tắt kiểm soát backpressure.");
  }

  function stopResumeTimer() {
    if (resumeTimer === null) return;
    try { clearInterval(resumeTimer); } catch (e) {}
    resumeTimer = null;
  }

  function cleanup() {
    isClosed = true;
    pendingAudio = null;
    stopResumeTimer();
    if (ws) {
      try {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        ws.close();
      } catch (e) {}
      ws = null;
    }
  }

  function notifyBackpressure(state) {
    const snap = gate.snapshot();
    droppedFrames = snap.droppedFrames;
    try {
      port.postMessage({
        type: "backpressure",
        state: state,                 // "ok" | "dropping" | "paused"
        bufferedAmount: ws ? ws.bufferedAmount : 0,
        droppedFrames: droppedFrames,
        // FIX-01: số liệu để biết nghẽn thật hay chỉ là spike ngắn.
        pauseCount: snap.pauseCount,
        resumeCount: snap.resumeCount,
        pausedMs: snap.pausedMs
      });
    } catch (e) {}
  }

  /**
   * FIX-01: kiểm tra xem đã rút đủ hàng đợi để chạy lại capture chưa.
   * Được gọi bởi timer nội bộ VÀ bởi `FLUSH_PENDING` từ content script.
   * Trả true nếu vừa thoát trạng thái tạm dừng.
   */
  function attemptResume() {
    if (!gate.isPaused()) {
      stopResumeTimer();
      return false;
    }
    if (isClosed || !ws || ws.readyState !== WebSocket.OPEN) {
      stopResumeTimer();
      return false;
    }
    const action = gate.evaluate(ws.bufferedAmount || 0);
    if (action !== gate.actions.RESUME) {
      return false;   // vẫn trên ngưỡng SOFT (hysteresis) -> tiếp tục chờ
    }
    stopResumeTimer();
    const buffered = pendingAudio;
    pendingAudio = null;
    console.log("[BS Background] Backpressure đã rút hết: chạy lại capture.");
    notifyBackpressure("ok");
    if (buffered) {
      try { ws.send(buffered); } catch (e) {}
    }
    return true;
  }

  function startResumeTimer() {
    if (resumeTimer !== null) return;
    resumeTimer = setInterval(attemptResume, BACKPRESSURE_RESUME_CHECK_MS);
  }

  function sendOrQueue(buffer) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      pendingAudio = null;
      return;
    }
    const A = gate.actions;
    const action = gate.evaluate(ws.bufferedAmount || 0);

    if (action === A.PAUSE) {
      // Quá tải nặng: bỏ frame, tạm dừng capture cho tới khi rút hết hàng đợi.
      pendingAudio = null;
      console.warn("[BS Background] Backpressure HARD: tạm dừng gửi audio.");
      notifyBackpressure("paused");
      startResumeTimer();
      return;
    }

    if (action === A.HOLD) {
      // Đã tạm dừng và chưa rút đủ: bỏ frame này, timer sẽ lo việc chạy lại.
      pendingAudio = null;
      return;
    }

    if (action === A.DROP) {
      // Đang tắc: chỉ giữ frame mới nhất, bỏ frame đang chờ.
      pendingAudio = buffer;
      notifyBackpressure("dropping");
      return;
    }

    if (action === A.RESUME) {
      // Hiếm khi tới đây (timer thường bắt trước), nhưng vẫn phải nhả trạng thái.
      stopResumeTimer();
      notifyBackpressure("ok");
    }

    pendingAudio = null;
    try {
      ws.send(buffer);
    } catch (e) {
      console.error("[BS Background] Send binary error:", e);
    }
  }

  port.onDisconnect.addListener(() => {
    cleanup();
  });

  port.onMessage.addListener((msg) => {
    if (!msg) return;

    if (msg.action === "CONNECT") {
      const url = msg.url || "wss://localhost:8765/ws";
      try {
        // FIX-01: kết nối mới => xoá trạng thái tạm dừng còn sót của kết nối cũ và
        // huỷ timer cũ, tránh timer "zombie" bắn vào socket đã đóng.
        stopResumeTimer();
        gate.reset();
        pendingAudio = null;
        ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
          if (!isClosed) port.postMessage({ type: "connected" });
        };

        ws.onmessage = (event) => {
          if (isClosed) return;
          if (typeof event.data === "string") {
            try {
              const parsed = JSON.parse(event.data);
              port.postMessage({ type: "ws_json", data: parsed });
            } catch (e) {
              port.postMessage({ type: "ws_json_raw", data: event.data });
            }
          } else {
            // P3.1: audio TTS có thể tới dưới dạng binary frame (không base64).
            port.postMessage({ type: "ws_binary", data: event.data });
          }
        };

        ws.onerror = (err) => {
          if (!isClosed) {
            console.error("[BS Background] WS error:", err);
            port.postMessage({
              type: "error",
              error: "Lỗi kết nối tới " + url + ". Vui lòng kiểm tra backend đang chạy."
            });
          }
        };

        ws.onclose = (event) => {
          if (!isClosed) {
            port.postMessage({ type: "disconnected", code: event.code, reason: event.reason });
          }
        };
      } catch (e) {
        if (!isClosed) {
          port.postMessage({ type: "error", error: e.message });
        }
      }
    } else if (msg.action === "SEND_JSON") {
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify(msg.data));
        } catch (e) {
          console.error("[BS Background] Send JSON error:", e);
        }
      }
    } else if (msg.action === "SEND_BINARY") {
      const buffer = (typeof buildBinaryAudioPacket === "function")
        ? buildBinaryAudioPacket(msg.header, msg.pcmBuffer, textEncoder)
        : (() => {
            const headerBytes = textEncoder.encode(JSON.stringify(msg.header));
            const pcmBytes = new Uint8Array(msg.pcmBuffer);
            const totalSize = 4 + headerBytes.length + pcmBytes.byteLength;
            const buf = new ArrayBuffer(totalSize);
            const view = new DataView(buf);
            view.setUint32(0, headerBytes.length, true);
            new Uint8Array(buf, 4, headerBytes.length).set(headerBytes);
            new Uint8Array(buf, 4 + headerBytes.length).set(pcmBytes);
            return buf;
          })();

      sendOrQueue(buffer);
    } else if (msg.action === "FLUSH_PENDING") {
      // FIX-01: đây là "liveness probe" do content script bắn định kỳ KHI ĐANG tạm dừng
      // (phòng trường hợp timer trong service worker bị MV3 tạm ngưng). Trước đây nhánh
      // này không có ai gọi nên trạng thái pause không bao giờ được nhả.
      if (gate.isPaused()) {
        attemptResume();
        return;
      }
      // Đường cũ: socket đã rút hết hàng đợi và còn frame đang chờ.
      if (pendingAudio && ws && ws.readyState === WebSocket.OPEN
          && (ws.bufferedAmount || 0) < SEND_BUFFER_SOFT_LIMIT) {
        const buf = pendingAudio;
        pendingAudio = null;
        try { ws.send(buf); } catch (e) {}
        notifyBackpressure("ok");
      }
    } else if (msg.action === "DISCONNECT") {
      cleanup();
    }
  });
});

// Broadcast subtitle events from capturing iframe to Top frame / all frames
api.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.action === "BROADCAST_SUBTITLE" && sender.tab?.id) {
    api.tabs.sendMessage(sender.tab.id, {
      action: "SUBTITLE_RENDER",
      eventType: msg.eventType,
      payload: msg.payload
    }).catch(() => {});
  }
});

console.log("[BS Background] Service worker & WebSocket bridge ready");
