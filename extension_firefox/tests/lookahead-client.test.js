// Test cho LookaheadClient — trọng tâm là ĐƯỜNG DỪNG (người dùng bấm Stop).
//   node --test extension_firefox/tests/
//   node extension_firefox/tests/lookahead-client.test.js
//
// Hồi quy: "Stop session nhưng server vẫn tiếp tục nhận âm thanh và chạy pipeline".
// Sau `disconnect()` thì TUYỆT ĐỐI không được gửi thêm frame audio/sync nào, socket phải
// được đóng (kể cả khi còn đang CONNECTING), và backend phải nhận lệnh `stop` tường minh.

const test = require("node:test");
const assert = require("node:assert/strict");

const { LookaheadClient } = require("../lib/lookahead-client.js");

// ── Giả lập môi trường trình duyệt ───────────────────────────────────────────
class FakeWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
    this.sent = [];
    this.closed = false;
    FakeWebSocket.instances.push(this);
  }
  send(data) {
    if (this.readyState !== FakeWebSocket.OPEN) throw new Error("socket chưa mở");
    this.sent.push(data);
  }
  close() {
    this.closed = true;
    this.readyState = FakeWebSocket.CLOSED;
  }
  // Tiện ích test
  open() {
    this.readyState = FakeWebSocket.OPEN;
    if (this.onopen) this.onopen();
  }
  emit(obj) {
    if (this.onmessage) this.onmessage({ data: JSON.stringify(obj) });
  }
  jsonSent() {
    return this.sent
      .filter((d) => typeof d === "string")
      .map((d) => JSON.parse(d));
  }
}
FakeWebSocket.CONNECTING = 0;
FakeWebSocket.OPEN = 1;
FakeWebSocket.CLOSING = 2;
FakeWebSocket.CLOSED = 3;
FakeWebSocket.instances = [];

function installBrowserStubs() {
  FakeWebSocket.instances = [];
  globalThis.WebSocket = FakeWebSocket;

  const listeners = [];
  globalThis.window = {
    postMessage(msg) {
      // Injected script trả lời ngay (mô phỏng interceptor).
      if (msg && msg.type === "REQUEST_INITIAL_CHUNKS") {
        dispatch({ source: "VIBE_LOOKAHEAD_POC", type: "AUDIO_CHUNK_INTERCEPTED",
                   payload: { isInit: false, epoch: 0 }, rawBytes: new ArrayBuffer(8) });
      }
    },
    addEventListener(type, fn) { if (type === "message") listeners.push(fn); },
    removeEventListener(type, fn) {
      if (type !== "message") return;
      const i = listeners.indexOf(fn);
      if (i >= 0) listeners.splice(i, 1);
    },
  };

  function dispatch(data) {
    listeners.slice().forEach((fn) => fn({ source: globalThis.window, data }));
  }
  return { dispatch, listenerCount: () => listeners.length };
}

function makeClient(env, overrides = {}) {
  const subs = [];
  const client = new LookaheadClient({
    serverUrl: "wss://localhost:8765/ws/lookahead",
    timelineQueue: { addSubtitles: (items, seekId) => subs.push({ items, seekId }) },
    leadTime: 12,
    sourceLang: "en",
    targetLang: "vi",
    config: { type: "set_config", vadEngine: "silero-vad", vadThreshold: 0.7, minWordsToCommit: 3 },
    ...overrides,
  });
  client.connect({ currentTime: 10, playbackRate: 1, paused: false });
  return { client, subs, ws: FakeWebSocket.instances[FakeWebSocket.instances.length - 1] };
}

test("gửi đủ cấu hình popup trong lookahead_init", () => {
  const env = installBrowserStubs();
  const { client, ws } = makeClient(env);
  ws.open();

  const init = ws.jsonSent().find((m) => m.type === "lookahead_init");
  assert.ok(init, "Thiếu lookahead_init");
  assert.equal(init.lead_time, 12);
  assert.equal(init.source_lang, "en");
  assert.equal(init.target_lang, "vi");
  // Cấu hình popup phải đi kèm (VAD/ASR/phân câu…).
  assert.equal(init.vadEngine, "silero-vad");
  assert.equal(init.vadThreshold, 0.7);
  assert.equal(init.minWordsToCommit, 3);
  client.disconnect();
});

test("updateConfig đẩy thiết lập mới sang backend", () => {
  const env = installBrowserStubs();
  const { client, ws } = makeClient(env);
  ws.open();

  client.updateConfig({ type: "set_config", vadEngine: "fsmn-vad", minWordsToCommit: 6 });
  const msg = ws.jsonSent().filter((m) => m.type === "set_config").pop();
  assert.ok(msg, "Không gửi set_config");
  assert.equal(msg.vadEngine, "fsmn-vad");
  assert.equal(msg.minWordsToCommit, 6);
  client.disconnect();
});

test("disconnect: gửi lệnh stop, đóng socket và CHẶN mọi frame sau đó", () => {
  const env = installBrowserStubs();
  const { client, ws } = makeClient(env);
  ws.open();

  env.dispatch({ source: "VIBE_LOOKAHEAD_POC", type: "AUDIO_CHUNK_INTERCEPTED",
                 payload: { isInit: false, epoch: 0 }, rawBytes: new ArrayBuffer(16) });
  const before = ws.sent.length;
  assert.ok(before > 0, "Phải gửi được frame audio trước khi Stop");

  client.disconnect();

  assert.ok(ws.closed, "disconnect() phải đóng WebSocket");
  assert.ok(ws.jsonSent().some((m) => m.type === "stop"), "Phải gửi lệnh stop tường minh");
  assert.equal(env.listenerCount(), 0, "Phải gỡ listener postMessage");

  // Sau Stop: mọi frame audio mới phải bị bỏ hoàn toàn.
  env.dispatch({ source: "VIBE_LOOKAHEAD_POC", type: "AUDIO_CHUNK_INTERCEPTED",
                 payload: { isInit: false, epoch: 0 }, rawBytes: new ArrayBuffer(32) });
  client.sendJSON({ type: "sync_state", currentTime: 99 });
  client.requestInitialChunks();
  assert.equal(ws.sent.length, before + 1, "Chỉ được có đúng lệnh stop, không thêm frame nào");
});

test("disconnect khi socket còn CONNECTING: không bao giờ mở rồi gửi dữ liệu", () => {
  const env = installBrowserStubs();
  const { client, ws } = makeClient(env);
  // Chưa open() — người dùng bấm Stop ngay trong lúc đang kết nối.
  client.disconnect();
  assert.ok(ws.closed, "Phải đóng socket đang CONNECTING");

  ws.readyState = FakeWebSocket.OPEN;
  if (ws.onopen) ws.onopen();
  assert.equal(ws.sent.length, 0, "Không được gửi gì sau khi đã Stop");
});

test("bỏ qua thông điệp server sau khi Stop", () => {
  const env = installBrowserStubs();
  const { client, subs, ws } = makeClient(env);
  ws.open();
  client.disconnect();

  ws.emit({ type: "lookahead_subtitles", seek_id: "init_0",
            items: [{ start_pts: 1, end_pts: 2, original_text: "x", translated_text: "y" }] });
  assert.equal(subs.length, 0, "Không được nạp phụ đề sau khi Stop");
});

test("chỉ resume khi backend báo prebuffer_ready", () => {
  const env = installBrowserStubs();
  let ready = 0;
  const { client, ws } = makeClient(env, { onPrebufferReady: () => { ready++; } });
  ws.open();

  // Nhận phụ đề KHÔNG được coi là "sẵn sàng phát" (câu có thể ở tận đâu phía trước).
  ws.emit({ type: "lookahead_subtitles", seek_id: "init_0",
            items: [{ start_pts: 30, end_pts: 32, original_text: "a", translated_text: "b" }] });
  assert.equal(ready, 0);

  ws.emit({ type: "lookahead_status", prebuffer_ready: false });
  assert.equal(ready, 0);

  ws.emit({ type: "lookahead_status", prebuffer_ready: true });
  assert.equal(ready, 1);
  client.disconnect();
});

test("nhận audio lồng tiếng qua JSON base64 (không phụ thuộc binaryType)", () => {
  const env = installBrowserStubs();
  const got = [];
  const { client, ws } = makeClient(env, {
    onTts: (header, buf) => got.push({ header, bytes: buf.byteLength }),
  });
  ws.open();

  const wav = Buffer.from([0x52, 0x49, 0x46, 0x46, 1, 2, 3, 4]);
  ws.emit({
    type: "lookahead_tts",
    seek_id: "init_0",
    start_pts: 12.5,
    end_pts: 15.0,
    duration_sec: 2.5,
    audio_b64: wav.toString("base64"),
  });

  assert.equal(got.length, 1, "Phải chuyển audio TTS cho scheduler");
  assert.equal(got[0].header.start_pts, 12.5);
  assert.equal(got[0].bytes, 8, "Giải mã base64 phải khôi phục đủ byte WAV");

  // Không có audio ⇒ bỏ qua, không gọi callback.
  ws.emit({ type: "lookahead_tts", start_pts: 20, end_pts: 22 });
  assert.equal(got.length, 1);
  client.disconnect();
});

test("lookahead_unavailable kích hoạt fallback", () => {
  const env = installBrowserStubs();
  let reason = null;
  const { client, ws } = makeClient(env, { onUnavailable: (r) => { reason = r; } });
  ws.open();
  ws.emit({ type: "lookahead_unavailable", reason: "ASR không khả dụng" });
  assert.equal(reason, "ASR không khả dụng");
  client.disconnect();
});

test("bỏ qua lookahead_status của seek CŨ (không phát lại quá sớm sau khi tua)", () => {  const env = installBrowserStubs();
  let ready = 0;
  const { client, ws } = makeClient(env, { onPrebufferReady: () => { ready++; } });
  ws.open();

  // Trạng thái sẵn sàng của đoạn ban đầu.
  ws.emit({ type: "lookahead_status", seek_id: "init_0", prebuffer_ready: true });
  assert.equal(ready, 1);

  // Người dùng tua: client sinh seek_id mới.
  client.notifySeek("seek_new", 500);
  assert.equal(client.activeSeekId, "seek_new");

  // Backend còn gửi trạng thái CŨ (chưa xử lý xong seek) ⇒ KHÔNG được resume.
  ws.emit({ type: "lookahead_status", seek_id: "init_0", prebuffer_ready: true });
  assert.equal(ready, 1, "Không được resume bằng trạng thái của seek cũ");

  // Trạng thái của seek mới thì mới resume.
  ws.emit({ type: "lookahead_status", seek_id: "seek_new", prebuffer_ready: true });
  assert.equal(ready, 2);
  client.disconnect();
});

test("kết nối qua Background Bridge thành công và truyền nhận dữ liệu", () => {
  const env = installBrowserStubs();
  const sent = [];
  const listeners = [];
  let disconnected = false;

  const fakePort = {
    name: "bs-lookahead-bridge",
    postMessage(msg) { sent.push(msg); },
    onMessage: {
      addListener(fn) { listeners.push(fn); },
    },
    onDisconnect: {
      addListener(fn) { this._disconnectFn = fn; },
    },
    disconnect() { disconnected = true; },
  };

  globalThis.chrome = {
    runtime: {
      connect({ name }) {
        assert.equal(name, "bs-lookahead-bridge");
        return fakePort;
      },
    },
  };

  let prebufferReady = false;
  const client = new LookaheadClient({
    serverUrl: "wss://localhost:8765/ws/lookahead",
    useBridge: true,
    leadTime: 15,
    onPrebufferReady: () => { prebufferReady = true; },
  });

  client.connect({ currentTime: 20, playbackRate: 1, paused: false });

  // Port phải gửi action CONNECT
  const connectMsg = sent.find((m) => m.action === "CONNECT");
  assert.ok(connectMsg, "Thiếu action CONNECT");
  assert.equal(connectMsg.url, "wss://localhost:8765/ws/lookahead");

  // Giả lập server connected từ bridge
  listeners.forEach((fn) => fn({ type: "connected" }));
  assert.equal(client.isConnected, true);

  // Phải gửi lookahead_init qua bridge SEND_JSON
  const initMsg = sent.find((m) => m.action === "SEND_JSON" && m.data?.type === "lookahead_init");
  assert.ok(initMsg, "Thiếu lookahead_init gửi qua bridge");
  assert.equal(initMsg.data.lead_time, 15);

  // Nhận lookahead_status qua bridge
  listeners.forEach((fn) => fn({
    type: "ws_json",
    data: { type: "lookahead_status", seek_id: "init_0", prebuffer_ready: true },
  }));
  assert.equal(prebufferReady, true);

  // Gửi audio chunk
  client._sendAudioBinaryFrame(new ArrayBuffer(32), { mime: "audio/webm", isInit: false });
  const binMsg = sent.find((m) => m.action === "SEND_RAW_BINARY");
  assert.ok(binMsg, "Thiếu SEND_RAW_BINARY qua bridge");
  assert.ok(binMsg.buffer instanceof ArrayBuffer);

  // Ngắt kết nối
  client.disconnect();
  assert.equal(disconnected, true);
  const stopMsg = sent.find((m) => m.action === "SEND_JSON" && m.data?.type === "stop");
  assert.ok(stopMsg, "Thiếu stop qua bridge");

  delete globalThis.chrome;
});

test("khi Background Bridge báo lỗi kết nối thì kích hoạt onError và onUnavailable", () => {
  const env = installBrowserStubs();
  const sent = [];
  const listeners = [];

  const fakePort = {
    name: "bs-lookahead-bridge",
    postMessage(msg) { sent.push(msg); },
    onMessage: {
      addListener(fn) { listeners.push(fn); },
    },
    onDisconnect: {
      addListener(fn) { this._disconnectFn = fn; },
    },
    disconnect() {},
  };

  globalThis.chrome = {
    runtime: {
      connect() { return fakePort; },
    },
  };

  let caughtError = null;
  let unavailableReason = null;

  const client = new LookaheadClient({
    serverUrl: "wss://localhost:8765/ws/lookahead",
    useBridge: true,
    onError: (err) => { caughtError = err; },
    onUnavailable: (r) => { unavailableReason = r; },
  });

  client.connect({ currentTime: 0, playbackRate: 1, paused: false });

  // Giả lập bridge báo lỗi
  listeners.forEach((fn) => fn({ type: "error", error: "Không thể kết nối máy chủ", details: "ECONNREFUSED" }));

  assert.ok(caughtError, "Không gọi onError khi bridge gặp lỗi");
  assert.ok(unavailableReason && unavailableReason.includes("Không thể kết nối máy chủ"), "Không gọi onUnavailable khi bridge lỗi lúc khởi động");

  client.disconnect();
  delete globalThis.chrome;
});

