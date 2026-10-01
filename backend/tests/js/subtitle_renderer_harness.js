// Kiểm thử CHỨC NĂNG cho SubtitleRenderer (3 tầng) bằng DOM giả lập trong Node.
//
// Vì sao cần: có bug thật — câu đang ở TẦNG 2 (đang chờ bản dịch) khi bị đẩy lên TẦNG 1
// (do có câu mới) thì bản dịch đến sau KHÔNG bao giờ được vẽ ra (tầng 1 chỉ cập nhật
// `textContent` của `.bs-translated` NẾU node đó đã tồn tại; câu bị đẩy lên lúc chưa có
// bản dịch nên node đó chưa được tạo).
//
// Dùng: node backend/tests/js/subtitle_renderer_harness.js  (exit 0 = PASS)

const fs = require("fs");
const path = require("path");
const vm = require("vm");

// ─────────────────────────── DOM giả lập (tối thiểu cho renderer) ───────────────

function makeClassList(el) {
  return {
    add: (...names) => {
      const set = new Set(String(el._class || "").split(/\s+/).filter(Boolean));
      names.forEach((n) => set.add(n));
      el._class = Array.from(set).join(" ");
    },
    remove: (...names) => {
      const set = new Set(String(el._class || "").split(/\s+/).filter(Boolean));
      names.forEach((n) => set.delete(n));
      el._class = Array.from(set).join(" ");
    },
    toggle: (name, force) => {
      const set = new Set(String(el._class || "").split(/\s+/).filter(Boolean));
      const shouldHave = force !== undefined ? !!force : !set.has(name);
      if (shouldHave) set.add(name);
      else set.delete(name);
      el._class = Array.from(set).join(" ");
      return shouldHave;
    },
    contains: (name) => String(el._class || "").split(/\s+/).filter(Boolean).includes(name),
  };
}

function matchesSelector(el, selector) {
  const nots = [...selector.matchAll(/:not\(\.([\w-]+)\)/g)].map((m) => m[1]);
  let s = selector.replace(/:not\([^)]*\)/g, "");
  const attrs = [...s.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)];
  s = s.replace(/\[[^\]]*\]/g, "");
  const classes = s.split(".").filter(Boolean);
  for (const c of classes) {
    if (!el.classList.contains(c)) return false;
  }
  for (const a of attrs) {
    const value = el.getAttribute(a[1]);
    if (value === null) return false;
    if (a[2] !== undefined && value !== a[2]) return false;
  }
  for (const n of nots) {
    if (el.classList.contains(n)) return false;
  }
  return true;
}

function walk(root, out) {
  for (const child of root.children) {
    out.push(child);
    walk(child, out);
  }
  return out;
}

class FakeElement {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this._class = "";
    this._text = "";
    this._attrs = {};
    this.style = {};
  }
  get className() {
    return this._class;
  }
  set className(value) {
    this._class = value || "";
  }
  get classList() {
    if (!this._classList) this._classList = makeClassList(this);
    return this._classList;
  }
  get textContent() {
    if (this.children.length > 0) return this.children.map((c) => c.textContent).join("");
    return this._text;
  }
  set textContent(value) {
    this.children.forEach((c) => {
      c.parentNode = null;
    });
    this.children = [];
    this._text = value === undefined || value === null ? "" : String(value);
  }
  get firstElementChild() {
    return this.children.length > 0 ? this.children[0] : null;
  }
  appendChild(child) {
    if (!child) return child;
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx >= 0) {
      this.children.splice(idx, 1);
      child.parentNode = null;
    }
    return child;
  }
  replaceChildren(...nodes) {
    this.children.forEach((c) => {
      c.parentNode = null;
    });
    this.children = [];
    this._text = "";
    nodes.filter(Boolean).forEach((n) => this.appendChild(n));
  }
  remove() {
    if (this.parentNode) this.parentNode.removeChild(this);
  }
  setAttribute(key, value) {
    this._attrs[key] = String(value);
  }
  getAttribute(key) {
    return Object.prototype.hasOwnProperty.call(this._attrs, key) ? this._attrs[key] : null;
  }
  querySelector(selector) {
    const found = walk(this, []).find((el) => matchesSelector(el, selector));
    return found || null;
  }
  querySelectorAll(selector) {
    return walk(this, []).filter((el) => matchesSelector(el, selector));
  }
}

function makeDocument() {
  return {
    createElement: (tag) => new FakeElement(tag),
    body: new FakeElement("body"),
  };
}

// ─────────────────────────── Nạp renderer trong scope giả ───────────────────────

function findFile(rel) {
  let dir = __dirname;
  for (let i = 0; i < 6; i++) {
    const candidate = path.join(dir, rel);
    if (fs.existsSync(candidate)) return candidate;
    dir = path.dirname(dir);
  }
  throw new Error("không tìm thấy " + rel);
}

const RENDERER_PATH = process.argv[2] || findFile(path.join("extension_firefox", "lib", "subtitle-renderer.js"));
const POLICY_PATH = findFile(path.join("extension_firefox", "lib", "subtitle-policy.js"));

const failures = [];
function check(cond, msg, extra) {
  if (cond) {
    console.log("  PASS " + msg);
  } else {
    console.log("  FAIL " + msg + (extra !== undefined ? `  (${extra})` : ""));
    failures.push(msg);
  }
}

function newRenderer() {
  const policy = require(POLICY_PATH);
  const container = new FakeElement("div");
  const sandbox = {
    document: makeDocument(),
    window: {},
    console,
    BSSubtitlePolicy: policy,
    // Không chạy timer thật: test phải tất định và không giữ tiến trình sống.
    setTimeout: () => ({ unref() {} }),
    clearTimeout: () => {},
  };
  vm.createContext(sandbox);
  const src = fs.readFileSync(RENDERER_PATH, "utf8");
  vm.runInContext(src, sandbox, { filename: RENDERER_PATH });
  const renderer = new sandbox.window.SubtitleRenderer(container);
  return { renderer, container };
}

// Chỉ lấy phần ĐANG THẬT SỰ nhìn thấy (bỏ câu đang mờ dần / đang trượt đi).
function visibleText(el) {
  if (!el) return "";
  if (el.classList.contains("bs-evicting") || el.classList.contains("bs-fading-out")) return "";
  if (el.children.length === 0) return el._text || "";
  return el.children
    .map(visibleText)
    .filter((t) => t && t.trim())
    .join(" ");
}

function layers(renderer) {
  return {
    l1: visibleText(renderer.historyLayer).trim(),
    l2: visibleText(renderer.focusLayer).trim(),
    l3: visibleText(renderer.liveLayer).trim(),
  };
}

function hasTranslation(renderer, id, text) {
  const el = renderer.completedSentences.find((s) => s.id === id);
  const dom = renderer.historyLayer.querySelector(`[data-sentence-id="${id}"]`) ||
    renderer.focusLayer.querySelector(`[data-sentence-id="${id}"]`);
  return {
    state: !!(el && el.translatedText === text),
    dom: !!(dom && dom.textContent.includes(text)),
  };
}

const finalUpdate = (id, text) => ({ type: "utterance_update", utterance_id: id, text, is_final: true, filtered: false });
const preview = (id, text) => ({ type: "utterance_update", utterance_id: id, text, is_final: false, filtered: false });
const finalTranslation = (id, text) => ({ type: "translation", sentence_id: id, translated: text, status: "ok", partial: false });

console.log(`Renderer: ${path.relative(process.cwd(), RENDERER_PATH)}`);

// ─────────────────────────── 1. Luồng cơ bản ───────────────────────────────
console.log("\n[1] Luồng cơ bản: preview → chốt câu → có bản dịch");
{
  const { renderer } = newRenderer();
  renderer.onUtteranceUpdate(preview("A", "hello"));
  let snap = layers(renderer);
  check(snap.l3.includes("hello"), "câu đang nhận dạng hiện ở TẦNG 3", snap.l3);
  check(snap.l2 === "", "TẦNG 2 còn trống khi chưa chốt", snap.l2);

  renderer.onUtteranceUpdate(finalUpdate("A", "hello world"));
  snap = layers(renderer);
  check(snap.l2.includes("hello world"), "câu đã chốt hiện ở TẦNG 2", snap.l2);
  check(snap.l2.includes("・・・"), "TẦNG 2 hiện chỉ báo đang chờ dịch", snap.l2);
  check(snap.l3 === "", "TẦNG 3 được xoá sau khi chốt", snap.l3);

  renderer.onTranslation(finalTranslation("A", "xin chào"));
  snap = layers(renderer);
  check(snap.l2.includes("xin chào"), "bản dịch điền vào TẦNG 2", snap.l2);
  check(!snap.l2.includes("・・・"), "chỉ báo chờ dịch biến mất khi có bản dịch", snap.l2);
}

// ─────────────────────────── 2. Tầng 2 → Tầng 1 khi ĐÃ có bản dịch ─────────
console.log("\n[2] Câu ở TẦNG 2 có bản dịch rồi mới bị câu mới đẩy lên TẦNG 1");
{
  const { renderer } = newRenderer();
  renderer.onUtteranceUpdate(finalUpdate("A", "hello world"));
  renderer.onTranslation(finalTranslation("A", "xin chào"));
  renderer.onUtteranceUpdate(finalUpdate("B", "second sentence"));
  const snap = layers(renderer);
  check(snap.l1.includes("hello world") && snap.l1.includes("xin chào"), "TẦNG 1 giữ cả câu gốc + bản dịch", snap.l1);
  check(snap.l2.includes("second sentence"), "TẦNG 2 chuyển sang câu mới", snap.l2);
}

// ─────────────────────────── 3. BUG: bị đẩy lên khi CHƯA có bản dịch ────────
console.log("\n[3] BUG: câu ở TẦNG 2 CHƯA có bản dịch đã bị đẩy lên TẦNG 1, bản dịch tới sau");
{
  const { renderer } = newRenderer();
  renderer.onUtteranceUpdate(finalUpdate("A", "hello world"));
  renderer.onUtteranceUpdate(finalUpdate("B", "second sentence")); // A bị đẩy lên TẦNG 1

  let snap = layers(renderer);
  check(snap.l1.includes("hello world"), "A vẫn hiển thị (câu gốc) ở TẦNG 1 sau khi bị đẩy lên", snap.l1);

  renderer.onTranslation(finalTranslation("A", "xin chào")); // bản dịch đến MUỘN
  snap = layers(renderer);
  const found = hasTranslation(renderer, "A", "xin chào");
  check(found.state, "state của A có bản dịch", JSON.stringify(found));
  check(
    snap.l1.includes("xin chào"),
    "TẦNG 1 phải HIỆN bản dịch đến muộn (bug: node .bs-translated chưa được tạo)",
    snap.l1
  );
}

// ─────────────── 4. Bản dịch muộn khi câu đã bị đẩy lên và đã có câu khác ────
console.log("\n[4] Bản dịch tới muộn hơn nữa: A → B → (dịch A) → C → (dịch B)");
{
  const { renderer } = newRenderer();
  renderer.onUtteranceUpdate(finalUpdate("A", "câu một"));
  renderer.onUtteranceUpdate(finalUpdate("B", "câu hai"));
  renderer.onTranslation(finalTranslation("A", "sentence one"));
  renderer.onUtteranceUpdate(finalUpdate("C", "câu ba"));
  renderer.onTranslation(finalTranslation("B", "sentence two"));
  const snap = layers(renderer);
  check(snap.l1.includes("sentence two"), "bản dịch muộn của B vẫn hiện ở TẦNG 1", snap.l1);
  check(snap.l2.includes("câu ba"), "TẦNG 2 là câu mới nhất", snap.l2);
}

// ─────────────── 5. Chỉ hiện bản dịch 1 lần: bỏ qua partial ──────────────────
console.log("\n[5] Tắt 'chạy chữ': mảnh dịch dở bị bỏ qua, bản hoàn chỉnh mới áp");
{
  const { renderer } = newRenderer();
  renderer.onUtteranceUpdate(finalUpdate("A", "hello world"));
  renderer.onTranslation({ type: "translation", sentence_id: "A", translated: "xin", status: "ok", partial: true });
  let snap = layers(renderer);
  check(!snap.l2.includes("xin"), "mảnh dịch dở KHÔNG được hiện", snap.l2);
  renderer.onTranslation(finalTranslation("A", "xin chào"));
  snap = layers(renderer);
  check(snap.l2.includes("xin chào"), "bản dịch hoàn chỉnh được hiện", snap.l2);
}

// ─────────────── 6. filtered=True phải xoá câu khỏi mọi tầng ────────────────
console.log("\n[6] Backend lọc câu ngắn (filtered=true) ⇒ xoá khỏi mọi tầng");
{
  const { renderer } = newRenderer();
  renderer.onUtteranceUpdate(finalUpdate("A", "ừm"));
  renderer.onUtteranceUpdate({ type: "utterance_update", utterance_id: "A", text: "ừm", is_final: true, filtered: true });
  const snap = layers(renderer);
  check(snap.l1 === "" && snap.l2 === "" && snap.l3 === "", "không còn dấu vết của câu bị lọc", JSON.stringify(snap));
}

// ── 7. Bản dịch đến muộn khi câu đã ra khỏi cửa sổ (maxLines) ───────────────
console.log("\n[7] Bản dịch tới muộn khi câu đã ra khỏi cửa sổ hiển thị");
{
  const { renderer } = newRenderer();
  renderer.onUtteranceUpdate(finalUpdate("A", "câu một"));
  renderer.onUtteranceUpdate(finalUpdate("B", "câu hai"));
  renderer.onUtteranceUpdate(finalUpdate("C", "câu ba"));   // chỉ còn B (lịch sử) + C (tiêu điểm)
  renderer.onTranslation(finalTranslation("A", "sentence one"));

  let snap = layers(renderer);
  check(!snap.l1.includes("sentence one"), "câu ngoài cửa sổ KHÔNG chiếm chỗ của câu đang hiện", snap.l1);

  // Bản dịch muộn KHÔNG được mất: nới Max lines là thấy lại.
  renderer.setMaxLines(3);
  snap = layers(renderer);
  check(snap.l1.includes("sentence one"), "nới Max lines ⇒ bản dịch muộn hiện lại (không mất dữ liệu)", snap.l1);
  check(snap.l2.includes("câu ba"), "TẦNG 2 vẫn là câu mới nhất", snap.l2);
}

// ── 8. Không được có 2 phần tử cùng id còn nhìn thấy được ───────────────────
console.log("\n[8] Không sinh phần tử trùng id trong lúc cập nhật bản dịch muộn");
{
  const { renderer } = newRenderer();
  renderer.onUtteranceUpdate(finalUpdate("A", "câu một"));
  renderer.onUtteranceUpdate(finalUpdate("B", "câu hai"));
  renderer.onTranslation(finalTranslation("A", "sentence one"));
  renderer.onTranslation(finalTranslation("A", "sentence one")); // gửi lặp
  const all = [...renderer.historyLayer.querySelectorAll("[data-sentence-id]"),
               ...renderer.focusLayer.querySelectorAll("[data-sentence-id]")];
  const visible = all.filter((el) => !el.classList.contains("bs-evicting") && !el.classList.contains("bs-fading-out"));
  const ids = visible.map((el) => el.getAttribute("data-sentence-id"));
  check(new Set(ids).size === ids.length, "mỗi câu chỉ có 1 phần tử đang hiển thị", JSON.stringify(ids));
}

// ── 9. On/Off phụ đề gốc (showOriginalSubtitles) ─────────────────────────
console.log("\n[9] Bật / Tắt hiển thị phụ đề gốc (showOriginalSubtitles)");
{
  const { renderer, container } = newRenderer();
  renderer.onUtteranceUpdate(finalUpdate("A", "original text"));
  renderer.onTranslation(finalTranslation("A", "bản dịch tiếng Việt"));

  // Mặc định: hiện cả hai
  check(renderer.showOriginal === true, "mặc định showOriginal là true");
  let origEl = renderer.focusLayer.querySelector(".bs-original");
  check(origEl && origEl.style.display !== "none", "mặc định .bs-original không bị ẩn");

  // Tắt phụ đề gốc
  renderer.applySettings({ showOriginalSubtitles: false });
  check(renderer.showOriginal === false, "showOriginal đổi sang false");
  check(container.classList.contains("bs-hide-original"), "container có class bs-hide-original");
  origEl = renderer.focusLayer.querySelector(".bs-original");
  check(origEl && origEl.style.display === "none", ".bs-original bị ẩn display: none khi tắt");

  // Bật lại phụ đề gốc
  renderer.applySettings({ showOriginalSubtitles: true });
  check(renderer.showOriginal === true, "showOriginal bật lại true");
  check(!container.classList.contains("bs-hide-original"), "container gỡ class bs-hide-original");
  origEl = renderer.focusLayer.querySelector(".bs-original");
  check(origEl && origEl.style.display !== "none", ".bs-original hiện lại khi bật");
}

console.log("");
if (failures.length > 0) {
  console.log(`KẾT QUẢ: FAIL (${failures.length} mục)`);
  process.exit(1);
}
console.log("KẾT QUẢ: PASS (3 tầng hiển thị đúng, kể cả bản dịch đến muộn)");
