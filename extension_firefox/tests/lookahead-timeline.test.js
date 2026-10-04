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

test("KHÔNG câu nào bị câu sau nuốt mất khi hai khối chồng mốc (sự cố 2026-10-04)", () => {
  // Log thật: backend gửi "When you're with me, baby." nhưng nó KHÔNG BAO GIỜ hiện, vì khối
  // trước gửi câu kết thúc muộn hơn mốc bắt đầu câu đầu của khối sau ⇒ công thức giãn cũ kẹp
  // câu trước còn 0.5 s và thứ tự duyệt mảng quyết định câu nào được vẽ.
  const video = makeVideo(70.0);
  const { queue, seen } = makeQueue(video);

  queue.addSubtitles([
    { start_pts: 70.2, end_pts: 79.06, original_text: "I can't see loving nobody but you for all my life.", translated_text: "a" },
  ], "init_0");
  // Khối sau bắt đầu TRƯỚC mốc kết thúc câu cuối của khối trước.
  queue.addSubtitles([
    { start_pts: 78.06, end_pts: 79.4, original_text: "When you're with me, baby.", translated_text: "b" },
    { start_pts: 79.5, end_pts: 81.0, original_text: "This is Rebecca.", translated_text: "c" },
  ], "init_0");

  // Câu trước phải được giữ đủ thời lượng tối thiểu, KHÔNG bị co xuống 0.5 s.
  const first = queue.items.find((i) => i.original_text.startsWith("I can't see"));
  assert.ok(
    first.end_pts - first.start_pts >= queue.minDurationSec - 1e-6,
    `câu trước bị co còn ${(first.end_pts - first.start_pts).toFixed(2)}s`
  );

  // Chạy hết dải: cả ba câu đều phải xuất hiện trên màn hình.
  for (let t = 70.0; t <= 81.2; t += 0.1) {
    video.currentTime = t;
    queue._tick();
  }
  const shown = new Set(seen.filter(Boolean));
  for (const text of [
    "I can't see loving nobody but you for all my life.",
    "When you're with me, baby.",
    "This is Rebecca.",
  ]) {
    assert.ok(shown.has(text), `câu không bao giờ hiện: ${text}`);
  }
});

test("câu có mốc bắt đầu muộn hơn được ưu tiên khi hai cửa sổ chồng nhau", () => {
  const video = makeVideo(10.0);
  const { queue } = makeQueue(video);
  queue.addSubtitles([
    { start_pts: 8.0, end_pts: 12.0, original_text: "câu cũ", translated_text: "a" },
    { start_pts: 10.5, end_pts: 13.0, original_text: "câu mới", translated_text: "b" },
  ], "init_0");

  video.currentTime = 10.6; // nằm trong CẢ HAI cửa sổ
  queue._tick();
  assert.equal(queue.activeSubtitle.original_text, "câu mới");
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

// ── RÀNG BUỘC CỨNG: không phát video qua vùng CHƯA được xử lý ────────────────
// Sự cố 2026-10-02: tua tới vùng video chưa tải ⇒ "lúc hiện phụ đề lúc không"; tua ngược về đầu
// ⇒ không hiện phụ đề gì. Nguyên nhân gồm việc client KHÔNG hề tạm dừng khi playhead vượt qua
// vùng backend đã xử lý (comment cũ: "Tuyệt đối KHÔNG tạm dừng video giữa chừng").
test("tạm dừng khi playhead chạm mốc backend đã xử lý", () => {
  const video = makeVideo(0);
  const buffering = [];
  const queue = new SubtitleTimelineQueue({
    onBufferingStateChange: (b) => buffering.push(b),
    onSeekTriggered: () => {},
  });
  queue.attachVideo(video);

  queue.addSubtitles([{ start_pts: 0, end_pts: 30, original_text: "A", translated_text: "a" }], "init_0");
  queue.setReadyHorizon(10.0, "init_0");

  // Còn trong vùng đã xử lý ⇒ KHÔNG tạm dừng (phát mượt)
  assert.equal(queue.isBehindHorizon(7.0), false);
  video.currentTime = 7.0;
  queue._tick();
  assert.deepEqual(buffering, []);

  // Chạm mốc đã xử lý ⇒ TẠM DỪNG chờ backend
  assert.equal(queue.isBehindHorizon(10.2), true);
  video.currentTime = 10.2;
  queue._tick();
  assert.deepEqual(buffering, [true], "phải báo bắt đầu nạp đệm khi hết vùng đã xử lý");
});

test("mốc đã xử lý của thế hệ seek CŨ không áp cho vị trí mới", () => {
  const video = makeVideo(0);
  const queue = new SubtitleTimelineQueue({ onSeekTriggered: () => {} });
  queue.attachVideo(video);

  // Backend báo đã xử lý tới 260s cho thế hệ hiện tại
  queue.setReadyHorizon(260.0, "init_0");
  assert.equal(queue.readyUntilPts, 260.0);
  assert.equal(queue.isBehindHorizon(27.0), false);

  // Người dùng tua ngược về 16.9s ⇒ marker cũ phải bị bỏ NGAY (nếu không, video phát qua vùng
  // chưa xử lý vì tưởng đã có phụ đề tới 260s)
  video.currentTime = 16.9;
  video.fire("seeking");
  assert.equal(queue.readyUntilPts, 0);
  assert.equal(queue.isBehindHorizon(16.9), false, "chưa biết mốc ⇒ không ràng buộc");

  // Marker của thế hệ seek CŨ gửi tới muộn cũng không được nhận
  queue.setReadyHorizon(260.0, "init_0");
  assert.equal(queue.readyUntilPts, 0, "marker thế hệ cũ bị bỏ");

  // Marker của thế hệ MỚI thì nhận
  queue.setReadyHorizon(20.5, queue.activeSeekId);
  assert.equal(queue.readyUntilPts, 20.5);
  assert.equal(queue.isBehindHorizon(20.4), true);
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
