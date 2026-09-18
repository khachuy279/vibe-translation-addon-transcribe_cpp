// QWEN-E6 (P1) — port chết SAU khi đã nối phải được hẹn nối lại.
//
// Firefox kill port của service worker khi SW idle (~30 s). Bản cũ xử lý
// `port.onDisconnect` chỉ trong nhánh `if (!settled)`: khi port chết SAU khi đã nối xong,
// KHÔNG ai gọi `_scheduleReconnect()` ⇒ phiên "sống sót ảo" — người dùng tưởng đang chạy
// nhưng không còn phụ đề mới và không tự nối lại.
//
// Dùng: node backend/tests/js/ws_reconnect_on_port_death_test.js  (exit 0 = PASS)

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function findClient() {
  let dir = __dirname;
  for (let i = 0; i < 6; i++) {
    const c = path.join(dir, "extension_firefox", "lib", "ws-client.js");
    if (fs.existsSync(c)) return c;
    dir = path.dirname(dir);
  }
  throw new Error("không tìm thấy ws-client.js");
}
const CLIENT = process.argv[2] || findClient();

const failures = [];
function check(cond, msg, extra) {
  if (cond) console.log("  PASS " + msg);
  else {
    console.log("  FAIL " + msg + (extra !== undefined ? `  (${extra})` : ""));
    failures.push(msg);
  }
}

// ---------------------------------------------------------------- port giả
let createdPorts = 0;
function makePort() {
  createdPorts++;
  const port = {
    name: "bs-ws-bridge",
    posted: [],
    disconnected: false,
    _msg: [],
    _disc: [],
    onMessage: { addListener(fn) { port._msg.push(fn); } },
    onDisconnect: { addListener(fn) { port._disc.push(fn); } },
    postMessage(m) { port.posted.push(m); },
    disconnect() { port.disconnected = true; },
    // tiện ích cho test
    emitMessage(m) { port._msg.forEach((f) => f(m)); },
    emitDisconnect() { port._disc.forEach((f) => f()); },
  };
  return port;
}

let lastPort = null;
const context = {
  console,
  chrome: {
    runtime: {
      connect() { lastPort = makePort(); return lastPort; },
    },
  },
  TextEncoder,
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h),
  setInterval: (fn, ms) => setInterval(fn, ms),
  clearInterval: (h) => clearInterval(h),
  performance: { now: () => Date.now() },
  WebSocket: class { constructor() { this.readyState = 1; } close() {} send() {} },
};
vm.createContext(context);
// `class WSClient` ở cấp script tạo binding LEXICAL, không phải thuộc tính của context
// (khác `var`/function) ⇒ phải tự gán ra `globalThis` để test lấy được.
vm.runInContext(
  fs.readFileSync(CLIENT, "utf8") + "\n;globalThis.__WSClient = WSClient;",
  context,
  { filename: CLIENT }
);
const WSClient = context.__WSClient;
check(typeof WSClient === "function", "WSClient được định nghĩa");

(async () => {
  const c = new WSClient("wss://localhost:8765/ws");
  check(c.useBridge === true, "dùng đường bridge (chrome.runtime.connect)");

  const events = [];
  c.on("connected", () => events.push("connected"));
  c.on("reconnecting", (p) => events.push("reconnecting"));
  c.on("disconnected", (p) => events.push("disconnected"));

  const connecting = c.connect();
  check(createdPorts === 1, "connect() mở đúng 1 port");
  check(lastPort.posted.some((m) => m.action === "CONNECT"), "gửi CONNECT qua port");

  lastPort.emitMessage({ type: "connected" });
  await connecting;
  check(c.isConnected === true, "nhận 'connected' ⇒ isConnected = true");
  check(events.includes("connected"), "phát sự kiện 'connected'");

  // ── điểm mấu chốt: port chết SAU khi đã nối
  lastPort.emitDisconnect();

  check(c.isConnected === false, "port chết ⇒ isConnected = false");
  check(events.includes("disconnected"), "phải phát 'disconnected' (bản cũ im lặng)");
  check(c._reconnectTimer !== null, "E6: PHẢI hẹn nối lại (bản cũ không hẹn ⇒ phiên sống sót ảo)");
  check(events.includes("reconnecting"), "phải phát 'reconnecting'");

  // ── không được hẹn nối lại sau khi người dùng chủ động Stop
  c.disconnect();
  check(c._reconnectTimer === null, "disconnect() phải huỷ timer (P3.5d giữ nguyên)");

  console.log("");
  if (failures.length === 0) {
    console.log("KẾT QUẢ: PASS (port chết sau khi nối ⇒ tự nối lại)");
    process.exit(0);
  }
  console.log(`KẾT QUẢ: FAIL (${failures.length} mục)`);
  failures.forEach((f) => console.log("  - " + f));
  process.exit(1);
})();
