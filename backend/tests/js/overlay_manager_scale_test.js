// Test suite for OverlayManager auto-scale & percentage font size
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

// Mock basic browser DOM environment
class MockElement {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.style = {
      _props: {},
      setProperty(key, val) { this._props[key] = String(val); },
      removeProperty(key) { delete this._props[key]; },
      getPropertyValue(key) { return this._props[key] || ""; },
      cssText: "",
    };
    this.classList = {
      add() {},
      remove() {},
      contains() { return false; },
    };
    this.childNodes = [];
    this.parentNode = null;
    this.isConnected = true;
    this._rect = { width: 854, height: 480 };
  }
  closest() { return null; }
  attachShadow() {
    return new MockElement("shadow-root");
  }
  appendChild(child) {
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }
  removeChild(child) {
    const idx = this.childNodes.indexOf(child);
    if (idx !== -1) this.childNodes.splice(idx, 1);
    child.parentNode = null;
    return child;
  }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  getBoundingClientRect() {
    return { ...this._rect };
  }
  setRect(w, h) {
    this._rect = { width: w, height: h };
  }
}

const overlayManagerCode = fs.readFileSync(
  path.join(__dirname, "../../../extension_firefox/content/overlay-manager.js"),
  "utf8"
);

const context = {
  console,
  document: {
    createElement(tag) { return new MockElement(tag); },
    body: new MockElement("body"),
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
    getComputedStyle() { return { position: "relative" }; },
  },
  MutationObserver: class {
    observe() {}
    disconnect() {}
  },
  ResizeObserver: class {
    constructor(cb) { this.cb = cb; }
    observe() {}
    disconnect() {}
  },
  SubtitleRenderer: class {
    constructor() {}
    applySettings() {}
    clear() {}
  },
  setTimeout: (fn) => fn(),
};

vm.createContext(context);
vm.runInContext(overlayManagerCode, context);
const OverlayManager = context.window.OverlayManager || context.OverlayManager;

console.log("=== Testing OverlayManager Auto-Scaling ===");

const om = new OverlayManager();
const mockVideo = new MockElement("video");
mockVideo.setRect(854, 480);
om.init(mockVideo);

// 1. Check default settings
console.log("\n[Test 1] Default font percentage settings");
assert.strictEqual(om.origFontPercent, 2.5, "Default origFontPercent should be 2.5%");
assert.strictEqual(om.transFontPercent, 4.2, "Default transFontPercent should be 4.2%");
console.log("  PASS Default percentages are 2.5% and 4.2%");

// 2. Normal web mode (854 x 480)
console.log("\n[Test 2] Normal Web Mode (854x480)");
mockVideo.setRect(854, 480);
om._updateScale();
let orig = parseFloat(om.container.style.getPropertyValue("--bs-orig-size"));
let trans = parseFloat(om.container.style.getPropertyValue("--bs-trans-size"));
console.log(`  Rendered at 480p: Origin = ${orig}px, Translate = ${trans}px`);
assert(Math.abs(orig - 12.0) < 0.2, `Expected ~12px for orig, got ${orig}`);
assert(Math.abs(trans - 20.2) < 0.2, `Expected ~20.2px for trans, got ${trans}`);
console.log("  PASS 480p dimensions scaled accurately");

// 3. 720p Theater mode (1280 x 720)
console.log("\n[Test 3] 720p Theater Mode (1280x720)");
mockVideo.setRect(1280, 720);
om._updateScale();
orig = parseFloat(om.container.style.getPropertyValue("--bs-orig-size"));
trans = parseFloat(om.container.style.getPropertyValue("--bs-trans-size"));
console.log(`  Rendered at 720p: Origin = ${orig}px, Translate = ${trans}px`);
assert(Math.abs(orig - 18.0) < 0.2, `Expected ~18px for orig, got ${orig}`);
assert(Math.abs(trans - 30.2) < 0.2, `Expected ~30.2px for trans, got ${trans}`);
console.log("  PASS 720p dimensions scaled accurately");

// 4. 1080p Fullscreen (1920 x 1080)
console.log("\n[Test 4] 1080p Fullscreen (1920x1080)");
mockVideo.setRect(1920, 1080);
om._updateScale();
orig = parseFloat(om.container.style.getPropertyValue("--bs-orig-size"));
trans = parseFloat(om.container.style.getPropertyValue("--bs-trans-size"));
console.log(`  Rendered at 1080p: Origin = ${orig}px, Translate = ${trans}px`);
assert(Math.abs(orig - 27.0) < 0.2, `Expected ~27px for orig, got ${orig}`);
assert(Math.abs(trans - 45.4) < 0.2, `Expected ~45.4px for trans, got ${trans}`);
console.log("  PASS 1080p Fullscreen auto-scaled to readable 27px / 45.4px!");

// 5. 4K Fullscreen (3840 x 2160)
console.log("\n[Test 5] 4K Fullscreen (3840x2160)");
mockVideo.setRect(3840, 2160);
om._updateScale();
orig = parseFloat(om.container.style.getPropertyValue("--bs-orig-size"));
trans = parseFloat(om.container.style.getPropertyValue("--bs-trans-size"));
console.log(`  Rendered at 4K: Origin = ${orig}px, Translate = ${trans}px`);
assert(orig >= 50 && orig <= 60, `Origin clamped/scaled properly (${orig})`);
assert(trans >= 85 && trans <= 90, `Translate clamped/scaled properly (${trans})`);
console.log("  PASS 4K Fullscreen scaled with upper safety bounds");

// 6. Portrait Video / Shorts (450 x 800)
console.log("\n[Test 6] Portrait Shorts (450x800)");
mockVideo.setRect(450, 800);
om._updateScale();
orig = parseFloat(om.container.style.getPropertyValue("--bs-orig-size"));
trans = parseFloat(om.container.style.getPropertyValue("--bs-trans-size"));
console.log(`  Rendered on portrait: Origin = ${orig}px, Translate = ${trans}px`);
// In portrait (width 450, height 800), effectiveHeight is clamped to width * 0.8 = 360
assert(Math.abs(orig - 9.0) < 0.2, `Expected ~9px on portrait, got ${orig}`);
assert(Math.abs(trans - 15.1) < 0.2, `Expected ~15.1px on portrait, got ${trans}`);
console.log("  PASS Portrait video properly clamped to avoid overflowing width");

// 7. Backward compatibility: receiving old pixel values in applySettings
console.log("\n[Test 7] Backward Compatibility with old pixel values");
mockVideo.setRect(854, 480);
om.applySettings({ origFontSize: 12, transFontSize: 21 }); // Old px values
console.log(`  Migrated percentages: Origin = ${om.origFontPercent.toFixed(1)}%, Translate = ${om.transFontPercent.toFixed(1)}%`);
assert(Math.abs(om.origFontPercent - 2.5) < 0.2, "12px on 480 should migrate to ~2.5%");
assert(Math.abs(om.transFontPercent - 4.4) < 0.2, "21px on 480 should migrate to ~4.4%");
console.log("  PASS Old pixel values successfully migrated to percentages");

// 8. Applying custom percentages in applySettings
console.log("\n[Test 8] Applying custom percentages");
om.applySettings({ origFontSize: 3.0, transFontSize: 5.5 });
assert.strictEqual(om.origFontPercent, 3.0);
assert.strictEqual(om.transFontPercent, 5.5);
orig = parseFloat(om.container.style.getPropertyValue("--bs-orig-size"));
trans = parseFloat(om.container.style.getPropertyValue("--bs-trans-size"));
assert(Math.abs(orig - 14.4) < 0.2, `Expected ~14.4px on 480p, got ${orig}`);
assert(Math.abs(trans - 26.4) < 0.2, `Expected ~26.4px on 480p, got ${trans}`);
console.log("  PASS Custom percentage settings applied and rendered correctly");

om.destroy();
console.log("\nALL TESTS PASSED SUCCESSFULLY!");
