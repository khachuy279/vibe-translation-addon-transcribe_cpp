// Test cho SubtitleTimelineQueue (Pipeline B — Lookahead Video Buffering).
//   node --test extension_firefox/tests/
//   node extension_firefox/tests/lookahead-timeline.test.js
//
// Hồi quy cho lỗi "mất phụ đề gốc và bản dịch": phụ đề đã dịch trước PHẢI hiện đúng
// khoảng [start_pts, end_pts] theo `video.currentTime`, không nhấp nháy giữa hai câu, và
// phải bị xoá sạch khi người dùng tua video.

const test = require("node:test");
const assert = require("node:assert/strict");

// `attachVideo` khởi động vòng render bằng requestAnimationFrame — trong Node ta thay
// bằng stub rỗng để test điều khiển `_tick()` thủ công và tiến trình thoát được.
globalThis.requestAnimationFrame = () => 1;
globalThis.cancelAnimationFrame = () => {};

const { SubtitleTimelineQueue } = require("../lib/lookahead-timeline.js");

function makeVideo(currentTime = 0, paused = false) {
  return {
    currentTime,
    paused,
    listeners: {},
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    removeEventListener(type, fn) {
      this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn);
    },
    fire(type) { (this.listeners[type] || []).forEach((fn) => fn()); },
  };
}

function makeQueue(video) {
  const seen = [];
  const queue = new SubtitleTimelineQueue({
    onSubtitleChange: (sub) => seen.push(sub ? sub.original_text : null),
    onSeekTriggered: () => {},
  });
  if (video) queue.attachVideo(video);
  return { queue, seen };
}

test("hiện đúng câu theo video.currentTime", () => {
  const video = makeVideo(0);
  const { queue, seen } = makeQueue(video);

  queue.addSubtitles([
    { start_pts: 0.0, end_pts: 2.0, original_text: "A", translated_text: "a" },
    { start_pts: 5.0, end_pts: 7.5, original_text: "B", translated_text: "b" },
  ], "init_0");

  video.currentTime = 0.5;
  queue._tick();
  assert.equal(queue.activeSubtitle.original_text, "A");

  video.currentTime = 2.3; // ngay sau khi câu A hết -> vẫn giữ trong holdAfterEndSec
  queue._tick();
  assert.equal(queue.activeSubtitle.original_text, "A");

  video.currentTime = 6.0;
  queue._tick();
  assert.equal(queue.activeSubtitle.original_text, "B");

  video.currentTime = 20.0; // quá xa -> không còn câu nào
  queue._tick();
  assert.equal(queue.activeSubtitle, null);
  assert.deepEqual(seen, ["A", "B", null]);
});

test("bỏ qua phụ đề mang seek_id cũ", () => {
  const video = makeVideo(0);
  const { queue } = makeQueue(video);

  queue.addSubtitles([{ start_pts: 0, end_pts: 1, original_text: "old", translated_text: "cũ" }], "seek_cu");
  assert.equal(queue.items.length, 0);

  queue.addSubtitles([{ start_pts: 0, end_pts: 1, original_text: "new", translated_text: "mới" }], "init_0");
  assert.equal(queue.items.length, 1);
});

test("sắp xếp timeline và chống trùng lặp", () => {
  const video = makeVideo(0);
  const { queue } = makeQueue(video);

  queue.addSubtitles([
    { start_pts: 9, end_pts: 11, original_text: "C", translated_text: "c" },
    { start_pts: 1, end_pts: 3, original_text: "A", translated_text: "a" },
  ], "init_0");
  queue.addSubtitles([
    { start_pts: 1.05, end_pts: 3.05, original_text: "A", translated_text: "a" },
  ], "init_0");

  assert.deepEqual(queue.items.map((i) => i.original_text), ["A", "C"]);
});

test("bỏ qua item có mốc thời gian không hợp lệ", () => {
  const video = makeVideo(0);
  const { queue } = makeQueue(video);
  queue.addSubtitles([
    { start_pts: 5, end_pts: 5, original_text: "zero", translated_text: "z" },
    { start_pts: NaN, end_pts: 10, original_text: "nan", translated_text: "n" },
  ], "init_0");
  assert.equal(queue.items.length, 0);
});

test("tua video xoá sạch phụ đề và sinh seek_id mới", () => {
  const video = makeVideo(0);
  const { queue } = makeQueue(video);
  let seekArgs = null;
  queue.onSeekTriggered = (id, t) => { seekArgs = [id, t]; };

  queue.addSubtitles([{ start_pts: 0, end_pts: 2, original_text: "A", translated_text: "a" }], "init_0");
  video.currentTime = 300.0;
  video.fire("seeking");

  assert.equal(queue.items.length, 0);
  assert.equal(queue.activeSubtitle, null);
  assert.notEqual(queue.activeSeekId, "init_0");
  assert.ok(seekArgs && seekArgs[0] === queue.activeSeekId);
  assert.equal(seekArgs[1], 300.0);
});

test("dọn dẹp câu quá cũ khi video chạy dài", () => {
  const video = makeVideo(0);
  const { queue } = makeQueue(video);
  for (let i = 0; i < 50; i++) {
    queue.addSubtitles([{ start_pts: i * 2, end_pts: i * 2 + 1.5, original_text: `S${i}`, translated_text: `s${i}` }], "init_0");
  }
  video.currentTime = 90.0;
  const removed = queue._evict(90.0);
  assert.ok(removed > 0, "Phải có câu bị dọn");
  assert.ok(queue.items.every((it) => it.end_pts >= 90.0 - 30.0));
});

test("detach gỡ được listener (không rò rỉ)", () => {
  const video = makeVideo(0);
  const { queue } = makeQueue(video);
  assert.equal(video.listeners.seeking.length, 1);
  assert.equal(video.listeners.seeked.length, 1);
  queue.detach();
  assert.equal(video.listeners.seeking.length, 0);
  assert.equal(video.listeners.seeked.length, 0);
});

// ── Hồi quy: "câu đầu tiên không được hiển thị" ─────────────────────────────
test("giữ sống câu đang hiển thị bằng cách phát lại định kỳ", () => {
  // Renderer tự cho câu hết hạn theo ĐỒNG HỒ THỰC; khi video bị tạm dừng để nạp đệm, câu
  // đang đúng sẽ biến mất nếu không được phát lại.
  const video = makeVideo(0);
  const { queue, seen } = makeQueue(video);
  queue.keepAliveMs = 20;
  queue.addSubtitles([{ start_pts: 0, end_pts: 30, original_text: "A", translated_text: "a" }], "init_0");

  video.currentTime = 1.0;
  queue._tick();
  assert.deepEqual(seen, ["A"]);

  // Cùng một câu, chưa tới hạn keep-alive ⇒ không phát lại (tránh spam DOM).
  queue._tick();
  assert.deepEqual(seen, ["A"]);
});

test("refreshNow() vẽ lại câu đang khớp sau khi overlay bị clear()", () => {
  const video = makeVideo(3.0);
  const { queue, seen } = makeQueue(video);
  queue.addSubtitles([{ start_pts: 2, end_pts: 8, original_text: "A", translated_text: "a" }], "init_0");
  video.currentTime = 3.0;
  queue._tick();
  assert.deepEqual(seen, ["A"]);

  // `om.clear()` lúc resume ⇒ overlay trống nhưng `activeSubtitle` vẫn là A nên _tick
  // KHÔNG vẽ lại. `refreshNow()` phải buộc vẽ lại ngay.
  seen.length = 0;
  queue._tick();
  assert.deepEqual(seen, [], "Không có thay đổi thì không phát lại");

  queue.refreshNow();
  assert.deepEqual(seen, ["A"], "refreshNow phải vẽ lại câu đang khớp");
});

test("tua video LUÔN bật trạng thái nạp đệm (video phải được tạm dừng)", () => {
  const video = makeVideo(10, false);
  let buffering = null;
  const queue = new SubtitleTimelineQueue({
    onSubtitleChange: () => {},
    onSeekTriggered: () => {},
    onBufferingStateChange: (on) => { buffering = on; },
  });
  queue.attachVideo(video);
  video.currentTime = 500;
  video.fire("seeking");
  assert.equal(buffering, true, "Tua video phải phát tín hiệu tạm dừng để nạp lại phụ đề");
  queue.detach();
});
