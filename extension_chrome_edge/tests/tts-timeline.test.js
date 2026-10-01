// Test cho TtsTimelineScheduler — trọng tâm là CHỐNG TRỄ DÂY CHUYỀN của lồng tiếng.
//   node --test extension_firefox/tests/
//
// Bài toán: ở Pipeline B, mỗi câu lồng tiếng phải phát ĐÚNG lúc `video.currentTime` đi qua
// `start_pts`. Nếu câu đọc dài hơn cửa sổ phụ đề mà cứ phát nối đuôi thì các câu sau bị đẩy
// lùi mãi. Bộ test này khoá lại 3 quy tắc:
//   1. Phát theo MỐC TUYỆT ĐỐI của video (không phải "phát khi nhận được").
//   2. Câu dài bị NÉN (playbackRate) và bị CẮT ở mốc cửa sổ.
//   3. Câu đến muộn quá ngưỡng thì BỎ, không phát lệch.

const test = require("node:test");
const assert = require("node:assert/strict");

const { TtsTimelineScheduler } = require("../lib/tts-timeline.js");

// ── Giả lập Web Audio + video ────────────────────────────────────────────────
class FakeSource {
  constructor() {
    this.playbackRate = { value: 1 };
    this.started = false;
    this.stopped = false;
    this.startAt = null;
    this.onended = null;
  }
  connect() {}
  disconnect() {}
  start(when) { this.started = true; this.startAt = when; }
  stop() {
    if (this.stopped) return;
    this.stopped = true;
    if (this.onended) this.onended();
  }
}

class FakeAudioContext {
  constructor() {
    this.state = "running";
    this.currentTime = 0;
    this.destination = {};
    this.sources = [];
    this._nextDuration = 1.0;
    this.decoded = 0;
  }
  createBufferSource() {
    const src = new FakeSource();
    this.sources.push(src);
    return src;
  }
  resume() { this.state = "running"; return Promise.resolve(); }
  decodeAudioData(_buf, ok) {
    this.decoded++;
    const duration = this._nextDuration;
    ok({ duration });
  }
  /** Buffer kế tiếp sẽ được decodeAudioData trả về. */
  nextBuffer(durationSec) { this._nextDuration = durationSec; }
}

function makeVideo(currentTime = 0) {
  return { currentTime, paused: false, playbackRate: 1.0 };
}

function makeScheduler(video, ctx, overrides = {}) {
  const s = new TtsTimelineScheduler({
    enabled: true,
    getAudioContext: () => ctx,
    tickMs: 100000, // không để timer tự chạy; test gọi _tick() thủ công
    ...overrides,
  });
  s.attach(video);
  return s;
}

function enqueueItem(s, ctx, { start, end, duration, seekId = "init_0" }) {
  ctx.nextBuffer(duration);
  return s.enqueue(
    { type: "lookahead_tts", seek_id: seekId, start_pts: start, end_pts: end, duration_sec: duration },
    new ArrayBuffer(16)
  );
}

// ── 1. Phát theo mốc tuyệt đối ───────────────────────────────────────────────
test("chỉ phát khi video.currentTime tới start_pts", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 10, end: 12, duration: 2 });

  video.currentTime = 5;
  s._tick();
  assert.equal(ctx.sources.length, 0, "Chưa tới mốc thì không được phát");

  video.currentTime = 9.9;
  s._tick();
  assert.equal(ctx.sources.length, 0);

  video.currentTime = 10.0;   // trong startLead (0,06 s)
  s._tick();
  assert.equal(ctx.sources.length, 1, "Tới mốc thì phải phát");
  assert.ok(ctx.sources[0].started);

  s.destroy();
});

test("video đang tạm dừng thì KHÔNG đọc trước", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 0, end: 2, duration: 1 });

  video.paused = true;
  s._tick();
  assert.equal(ctx.sources.length, 0);

  video.paused = false;
  s._tick();
  assert.equal(ctx.sources.length, 1);
  s.destroy();
});

// ── 2. Chống trễ dây chuyền ──────────────────────────────────────────────────
test("câu đọc DÀI hơn cửa sổ: bị nén và bị CẮT để câu sau vẫn đúng giờ", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);

  // Câu A: phụ đề 0->2s nhưng đọc tới 5s. Câu B bắt đầu ở 3s.
  enqueueItem(s, ctx, { start: 0, end: 2, duration: 5 });
  enqueueItem(s, ctx, { start: 3, end: 5, duration: 1.5 });

  s._tick(); // t = 0 -> phát A
  assert.equal(ctx.sources.length, 1);
  const a = ctx.sources[0];
  assert.ok(a.playbackRate.value > 1.0, "Câu dài phải được nén (rate > 1)");
  assert.ok(a.playbackRate.value <= 1.35 + 1e-6, "Không vượt trần nén của client");

  // t = 1.0: A vẫn đang đọc (chưa tới mốc cắt 3s).
  video.currentTime = 1.0;
  s._tick();
  assert.equal(a.stopped, false);

  // t = 3.0: B phải bắt đầu; A bị CẮT (không được đẩy lùi B).
  video.currentTime = 3.0;
  s._tick();
  assert.equal(a.stopped, true, "Câu trước phải bị cắt khi câu sau tới hạn");
  assert.equal(ctx.sources.length, 2, "Câu B phải được phát đúng mốc 3s");
  const b = ctx.sources[1];
  assert.ok(b.started);
  assert.equal(b.playbackRate.value, 1.0, "Câu B vừa cửa sổ nên đọc tốc độ thường");
  assert.equal(s.stats.cutShort, 1);

  s.destroy();
});

test("câu vừa cửa sổ thì KHÔNG nén", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 0, end: 3, duration: 2.5 });
  s._tick();
  assert.equal(ctx.sources[0].playbackRate.value, 1.0);
  s.destroy();
});

test("cửa sổ tính tới câu KẾ TIẾP, không phải hết phụ đề", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  // Phụ đề 0->6s nhưng câu kế tiếp bắt đầu ở 2s ⇒ chỉ có 2s để đọc.
  enqueueItem(s, ctx, { start: 0, end: 6, duration: 4 });
  enqueueItem(s, ctx, { start: 2, end: 4, duration: 1 });
  s._tick();
  const a = ctx.sources[0];
  assert.ok(a.playbackRate.value > 1.0, "Phải nén vì chỉ có 2s trước câu kế tiếp");
  s.destroy();
});

test("câu bị cắt ở mốc cửa sổ ngay cả khi không có câu kế tiếp", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 0, end: 2, duration: 9 }); // 9s, cửa sổ ~3s
  s._tick();
  const a = ctx.sources[0];
  video.currentTime = 3.05; // quá stopAtPts = 0 + 3
  s._tick();
  assert.equal(a.stopped, true);
  s.destroy();
});

// ── 3. Câu đến muộn ─────────────────────────────────────────────────────────
test("câu đến MUỘN quá ngưỡng thì bỏ, không phát lệch", () => {
  const video = makeVideo(5);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 1, end: 4, duration: 2 }); // đã trôi qua
  s._tick();
  assert.equal(ctx.sources.length, 0, "Không được phát câu đã trôi qua");
  s.destroy();
});

test("câu đến hơi muộn (trong ngưỡng) vẫn phát", () => {
  const video = makeVideo(0.15);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 0, end: 2, duration: 1 });
  s._tick();
  assert.equal(ctx.sources.length, 1);
  s.destroy();
});

// ── 4. Đồng bộ theo tốc độ phát của video ───────────────────────────────────
test("video phát 1.5x thì lồng tiếng cũng nhanh 1.5x", () => {
  const video = makeVideo(0);
  video.playbackRate = 1.5;
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 0, end: 3, duration: 2 });
  s._tick();
  assert.ok(Math.abs(ctx.sources[0].playbackRate.value - 1.5) < 1e-6);
  s.destroy();
});

// ── 5. Vòng đời ─────────────────────────────────────────────────────────────
test("clear() dừng câu đang phát và xoá hàng đợi", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 0, end: 5, duration: 3 });
  enqueueItem(s, ctx, { start: 6, end: 8, duration: 1 });
  s._tick();
  const a = ctx.sources[0];
  s.clear();
  assert.equal(a.stopped, true);
  assert.equal(s.items.length, 0);
  assert.equal(s.snapshot().playing, false);
  s.destroy();
});

test("setEnabled(false) dừng câu đang phát, xoá hàng đợi và không nhận thêm enqueue", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 0, end: 3, duration: 1 });
  s._tick();
  assert.equal(ctx.sources.length, 1, "Đang phát 1 câu");
  assert.equal(ctx.sources[0].stopped, false);

  s.setEnabled(false);
  assert.equal(ctx.sources[0].stopped, true, "Phải dừng câu đang phát ngay lập tức");
  assert.equal(s.items.length, 0, "Hàng đợi phải bị xoá");

  // Khi đang tắt, cố nạp thêm câu mới ⇒ bị từ chối
  assert.equal(enqueueItem(s, ctx, { start: 5, end: 8, duration: 1 }), false);
  assert.equal(s.items.length, 0);

  // Khi bật lại, tiếp tục nhận câu mới
  s.setEnabled(true);
  assert.equal(enqueueItem(s, ctx, { start: 10, end: 13, duration: 1 }), true);
  assert.equal(s.items.length, 1);
  s.destroy();
});

test("chống nạp trùng theo mốc thời gian", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 0, end: 3, duration: 1 });
  enqueueItem(s, ctx, { start: 0, end: 3, duration: 1 });
  assert.equal(s.items.length, 1, "Chỉ được giữ một bản cho mỗi mốc thời gian");
  s._tick();
  assert.equal(ctx.sources.length, 1);
  s.destroy();
});

test("bỏ qua khung thiếu mốc thời gian", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx);
  assert.equal(s.enqueue({ type: "lookahead_tts" }, new ArrayBuffer(8)), false);
  assert.equal(s.enqueue({ type: "lookahead_tts", start_pts: "x", end_pts: 2 }, new ArrayBuffer(8)), false);
  assert.equal(s.items.length, 0);
  s.destroy();
});

// ── 6. Dự phòng khi AudioContext bị chặn (autoplay policy) ───────────────────
test("AudioContext bị treo ⇒ phát bằng thẻ <audio> (không im lặng)", () => {
  const video = makeVideo(0);

  // AudioContext ở trạng thái suspended và resume() KHÔNG thành công.
  const ctx = new FakeAudioContext();
  ctx.state = "suspended";
  ctx.resume = () => Promise.resolve();

  const played = [];
  globalThis.Blob = class Blob { constructor(parts) { this.parts = parts; } };
  globalThis.URL = { createObjectURL: () => "blob:tts-1", revokeObjectURL: () => {} };
  globalThis.Audio = class FakeAudio {
    constructor(url) { this.url = url; this.playbackRate = 1; this.played = false; this.paused = false; played.push(this); }
    play() { this.played = true; return Promise.resolve(); }
    pause() { this.paused = true; }
  };

  const s = makeScheduler(video, ctx);
  enqueueItem(s, ctx, { start: 0, end: 3, duration: 2 });
  s._tick();

  assert.equal(ctx.sources.length, 0, "Không dùng Web Audio khi context bị treo");
  assert.equal(played.length, 1, "Phải phát bằng thẻ <audio>");
  assert.equal(played[0].url, "blob:tts-1");
  assert.equal(played[0].played, true);
  assert.equal(s.stats.blockedByAudioContext, 1);
  assert.equal(s.stats.played, 1);

  // Câu sau tới hạn ⇒ câu đang phát bị dừng (không trễ dây chuyền).
  enqueueItem(s, ctx, { start: 3, end: 5, duration: 1 });
  video.currentTime = 3.0;
  s._tick();
  assert.equal(played[0].paused, true, "Câu trước phải bị dừng");
  assert.equal(played.length, 2);
  s.destroy();
  delete globalThis.Audio;
});

test("unlockAudio() resume AudioContext khi người dùng tương tác", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  ctx.state = "suspended";
  let resumed = 0;
  ctx.resume = () => { resumed++; ctx.state = "running"; return Promise.resolve(); };

  const s = makeScheduler(video, ctx);
  assert.equal(s.unlockAudio(), true);
  assert.equal(resumed, 1);
  assert.equal(ctx.state, "running");
  s.destroy();
});

test("câu trước chỉ còn đuôi ngắn (1-2 từ) thì câu sau nhường cho đọc nốt, không cắt cụt", () => {
  const video = makeVideo(0);
  const ctx = new FakeAudioContext();
  const s = makeScheduler(video, ctx, { allowPitchShift: false, preemptGraceSec: 0.5 });

  // Câu A: phát lúc 0s, dài 2.2s. Câu B bắt đầu lúc 2.0s.
  // Khi video tới 2.0s, Câu A đã đọc được 2.0s, chỉ còn 0.2s (< preemptGraceSec 0.5s).
  enqueueItem(s, ctx, { start: 0, end: 2, duration: 2.2 });
  enqueueItem(s, ctx, { start: 2.0, end: 4, duration: 1.0 });

  s._tick(); // t = 0 -> phát A
  assert.equal(ctx.sources.length, 1);
  const a = ctx.sources[0];

  // t = 2.0s: Câu B tới hạn nhưng Câu A chỉ còn 0.2s ⇒ Câu B NHƯỜNG, Câu A KHÔNG bị cắt!
  video.currentTime = 2.0;
  s._tick();
  assert.equal(a.stopped, false, "Câu A không được bị cắt cụt đuôi");
  assert.equal(ctx.sources.length, 1, "Câu B tạm hoãn chờ Câu A");

  // t = 2.2s: Câu A đọc xong (onended kích hoạt)
  video.currentTime = 2.2;
  a.stop(); // giả lập kết thúc audio

  // Câu B lập tức phát tiếp mà không bị gián đoạn
  assert.equal(ctx.sources.length, 2, "Câu B phát ngay sau khi Câu A kết thúc");
  const b = ctx.sources[1];
  assert.ok(b.started);

  s.destroy();
});

