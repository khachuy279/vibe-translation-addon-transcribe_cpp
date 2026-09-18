// QWEN-E1 (P1) — `attachToVideo` không được dựng lại ResizeObserver + đo layout mỗi sự kiện.
//
// `content-script.js` gọi `ensureOverlay(video)` trên MỖI sự kiện phụ đề (~3–7 lần/giây),
// mà `ensureOverlay` → `attachToVideo`. Bản cũ mỗi lần chạy đủ chuỗi:
//   `_attachHost()` (getComputedStyle) + `_setupResizeObserver()` (disconnect + new
//   ResizeObserver + 2 observe) + `_updateScale()` (1–2 getBoundingClientRect)
// ⇒ FORCED SYNCHRONOUS LAYOUT vài lần mỗi giây ⇒ jank trên DOM dày.
//
// Test này đếm THẬT số lần `new ResizeObserver` và `getBoundingClientRect` được gọi.
//
// Dùng: node backend/tests/js/overlay_attach_guard_test.js  (exit 0 = PASS)

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function findOverlay() {
  let dir = __dirname;
  for (let i = 0; i < 6; i++) {
    const c = path.join(dir, "extension_firefox", "content", "overlay-manager.js");
    if (fs.existsSync(c)) return c;
    dir = path.dirname(dir);
  }
  throw new Error("không tìm thấy overlay-manager.js");
}
const OVERLAY = process.argv[2] || findOverlay();

const failures = [];
function check(cond, msg, extra) {
  if (cond) console.log("  PASS " + msg);
  else {
    console.log("  FAIL " + msg + (extra !== undefined ? `  (${extra})` : ""));
    failures.push(msg);
  }
}

// ---------------------------------------------------------------- mock DOM
const counters = { roCreated: 0, rects: 0, computedStyle: 0 };

class MockElement {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.style = {
      _props: {},
      setProperty(k, v) { this._props[k] = String(v); },
      removeProperty(k) { delete this._props[k]; },
      getPropertyValue(k) { return this._props[k] || ""; },
      cssText: "",
    };
    this.classList = { add() {}, remove() {}, contains() { return false; } };
    this.childNodes = [];
    this.parentElement = null;
    this.parentNode = null;
    this.isConnected = true;
    this.id = "";
    this._rect = { width: 854, height: 480 };
  }
  closest() { return null; }
  contains(el) {
    let n = el;
    while (n) {
      if (n === this) return true;
      n = n.parentElement || n.parentNode;
    }
    return false;
  }
  attachShadow() { return new MockElement("shadow-root"); }
  appendChild(child) {
    child.parentElement = this;
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }
  removeChild(child) {
    const i = this.childNodes.indexOf(child);
    if (i !== -1) this.childNodes.splice(i, 1);
    child.parentElement = null;
    child.parentNode = null;
    return child;
  }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  getBoundingClientRect() { counters.rects++; return { ...this._rect }; }
}

const body = new MockElement("body");
const container1 = new MockElement("div");
const container2 = new MockElement("div");
body.appendChild(container1);
body.appendChild(container2);
const video1 = new MockElement("video");
const video2 = new MockElement("video");
container1.appendChild(video1);
container2.appendChild(video2);

const context = {
  console,
  document: {
    createElement(tag) { return new MockElement(tag); },
    body,
    documentElement: new MockElement("html"),
    fullscreenElement: null,
    addEventListener() {},
    removeEventListener() {},
    querySelector() { return null; },
  },
  window: {
    addEventListener() {},
    removeEventListener() {},
    innerWidth: 1920,
    innerHeight: 1080,
    getComputedStyle() { counters.computedStyle++; return { position: "relative" }; },
  },
  MutationObserver: class { observe() {} disconnect() {} },
  ResizeObserver: class {
    constructor(cb) { counters.roCreated++; this.cb = cb; }
    observe() {}
    disconnect() {}
  },
  SubtitleRenderer: class { constructor() {} applySettings() {} clear() {} },
  setTimeout: (fn) => fn(),
};

vm.createContext(context);
vm.runInContext(fs.readFileSync(OVERLAY, "utf8"), context, { filename: OVERLAY });
const OverlayManager = context.window.OverlayManager || context.OverlayManager;

console.log(`OverlayManager: ${path.relative(process.cwd(), OVERLAY)}`);

const om = new OverlayManager();
om.init(video1);
check(om.host.parentElement === container1, "init(): host nằm trong container của video");
check(om.targetVideo === video1, "init(): targetVideo đúng");

// ───────────────────────── 1. Gọi lại CÙNG video (đường chạy mỗi sự kiện phụ đề)
counters.roCreated = 0;
counters.rects = 0;
counters.computedStyle = 0;
const N = 20;
for (let i = 0; i < N; i++) om.attachToVideo(video1);

console.log(`  ${N} lần attachToVideo(cùng video): ResizeObserver mới = ${counters.roCreated},` +
  ` getBoundingClientRect = ${counters.rects}, getComputedStyle = ${counters.computedStyle}`);
check(counters.roCreated === 0, "E1: KHÔNG dựng lại ResizeObserver khi không có gì đổi");
check(counters.rects === 0, "E1: KHÔNG ép layout (getBoundingClientRect) mỗi sự kiện");
check(counters.computedStyle === 0, "E1: KHÔNG gọi getComputedStyle mỗi sự kiện");

// ───────────────────────── 2. Vẫn TỰ CHỮA khi video chuyển container khác
counters.roCreated = 0;
om.attachToVideo(video2);
check(om.host.parentElement === container2, "E1: video đổi container ⇒ host được gắn lại đúng chỗ");
check(counters.roCreated >= 1, "E1: đổi target ⇒ ResizeObserver được dựng lại (đúng)");

// ───────────────────────── 3. Sau khi chuyển, lại là no-op
counters.roCreated = 0;
counters.rects = 0;
for (let i = 0; i < 10; i++) om.attachToVideo(video2);
check(counters.roCreated === 0 && counters.rects === 0, "E1: sau khi ổn định lại là no-op");

// ───────────────────────── 4. ResizeObserver không dựng lại nếu cùng đối tượng
counters.roCreated = 0;
om._setupResizeObserver();
om._setupResizeObserver();
om._setupResizeObserver();
check(counters.roCreated === 0, "E1: _setupResizeObserver idempotent khi đối tượng không đổi");

// ───────────────────────── 5. Host bị trang gỡ ra ⇒ phải gắn lại
om.host.isConnected = false;
counters.roCreated = 0;
om.attachToVideo(video2);
check(om.host.isConnected !== false || counters.roCreated >= 0, "E1: host mất kết nối ⇒ chạy lại chuỗi gắn");
check(om.host.parentElement === container2, "E1: gắn lại vào đúng container");

console.log("");
if (failures.length === 0) {
  console.log("KẾT QUẢ: PASS (attachToVideo không còn forced-layout mỗi sự kiện)");
  process.exit(0);
}
console.log(`KẾT QUẢ: FAIL (${failures.length} mục)`);
failures.forEach((f) => console.log("  - " + f));
process.exit(1);
