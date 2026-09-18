// QWEN-E3 (P1) — broadcast phụ đề không được gửi tới MỌI iframe.
//
// `content-script.js` chỉ gửi `BROADCAST_SUBTITLE` khi `window !== window.top`, tức frame
// đang capture là iframe con và đích cần vẽ là frame TRÊN CÙNG. Bản cũ gọi
// `tabs.sendMessage(tabId, msg)` KHÔNG có `frameId` ⇒ mỗi sự kiện phụ đề được giao tới MỌI
// content script của MỌI iframe trong tab (YouTube có hàng chục: ads, embed, ITP); mỗi
// frame thức dậy, parse rồi tự loại bỏ. CPU đốt miễn phí × số frame × tốc độ event.
//
// Dùng: node backend/tests/js/sw_broadcast_frame_target_test.js  (exit 0 = PASS)

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function findSW() {
  let dir = __dirname;
  for (let i = 0; i < 6; i++) {
    const c = path.join(dir, "extension_firefox", "background", "service-worker.js");
    if (fs.existsSync(c)) return c;
    dir = path.dirname(dir);
  }
  throw new Error("không tìm thấy service-worker.js");
}
const SW = process.argv[2] || findSW();

const failures = [];
function check(cond, msg, extra) {
  if (cond) console.log("  PASS " + msg);
  else {
    console.log("  FAIL " + msg + (extra !== undefined ? `  (${extra})` : ""));
    failures.push(msg);
  }
}

const onMessageListeners = [];
const sent = [];

const api = {
  runtime: {
    onConnect: { addListener() {} },
    onMessage: { addListener(fn) { onMessageListeners.push(fn); } },
    connect() { return { onMessage: { addListener() {} }, onDisconnect: { addListener() {} }, postMessage() {} }; },
    getURL: (p) => p,
    lastError: null,
  },
  tabs: {
    sendMessage(tabId, msg, opts) { sent.push({ tabId, msg, opts }); return Promise.resolve(); },
    query() { return Promise.resolve([]); },
  },
  scripting: { executeScript() { return Promise.resolve([]); } },
  storage: { local: { get() { return Promise.resolve({}); }, set() { return Promise.resolve(); } } },
  action: { setBadgeText() {}, setBadgeBackgroundColor() {} },
};

const context = {
  console,
  chrome: api,
  browser: undefined,
  TextEncoder,
  WebSocket: class { constructor() {} close() {} send() {} },
  setInterval: () => 0,
  clearInterval: () => {},
  setTimeout: () => 0,
  clearTimeout: () => {},
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(SW, "utf8"), context, { filename: SW });

console.log(`Service worker: ${path.relative(process.cwd(), SW)}`);
check(onMessageListeners.length >= 1, "service worker đăng ký runtime.onMessage");

// Gọi listener đúng như runtime gọi: (msg, sender, sendResponse)
const sender = { tab: { id: 42 }, frameId: 7 };
for (const fn of onMessageListeners) {
  fn({ action: "BROADCAST_SUBTITLE", eventType: "utterance_update", payload: { text: "xin chào" } },
    sender, () => {});
}

check(sent.length === 1, `gửi đúng 1 message (nhận ${sent.length})`);
if (sent.length) {
  const { tabId, msg, opts } = sent[0];
  check(tabId === 42, "gửi tới đúng tab của sender");
  check(msg.action === "SUBTITLE_RENDER", "action đúng");
  check(msg.eventType === "utterance_update", "eventType được chuyển tiếp");
  check(msg.payload && msg.payload.text === "xin chào", "payload được chuyển tiếp");
  check(!!opts && typeof opts === "object", "PHẢI truyền options cho tabs.sendMessage", String(opts));
  check(!!opts && opts.frameId === 0, "E3: phải nhắm frame TRÊN CÙNG (frameId=0), không broadcast mọi iframe",
    opts && String(opts.frameId));
}

console.log("");
if (failures.length === 0) {
  console.log("KẾT QUẢ: PASS (broadcast chỉ tới top frame)");
  process.exit(0);
}
console.log(`KẾT QUẢ: FAIL (${failures.length} mục)`);
failures.forEach((f) => console.log("  - " + f));
process.exit(1);
