// Test cho BackpressureGate (FIX-01).
//   node --test extension_firefox/tests/          # runner (spawn tiến trình con)
//   node extension_firefox/tests/backpressure-gate.test.js   # chạy trực tiếp (không spawn)
//
// Đây là test hồi quy cho bug P0 "capture treo vĩnh viễn": gate PHẢI có đường thoát
// khỏi trạng thái pause mà không cần một frame audio mới nào.

const test = require("node:test");
const assert = require("node:assert/strict");

const { createGate, ACTIONS, SOFT_LIMIT_BYTES, HARD_LIMIT_BYTES } = require("../lib/backpressure-gate.js");

const SOFT = SOFT_LIMIT_BYTES;
const HARD = HARD_LIMIT_BYTES;

/** Đồng hồ giả để kiểm tra `pausedMs` tất định. */
function fakeClock(start = 1000) {
  let value = start;
  return {
    now: () => value,
    advance: (ms) => { value += ms; return value; },
  };
}

test("dưới SOFT thì gửi bình thường", () => {
  const gate = createGate({});
  assert.equal(gate.evaluate(0), ACTIONS.SEND);
  assert.equal(gate.evaluate(SOFT - 1), ACTIONS.SEND);
  assert.equal(gate.isPaused(), false);
  assert.equal(gate.snapshot().droppedFrames, 0);
});

test("từ SOFT tới dưới HARD thì bỏ frame đang chờ (giữ frame mới nhất)", () => {
  const gate = createGate({});
  assert.equal(gate.evaluate(SOFT), ACTIONS.DROP);
  assert.equal(gate.evaluate(HARD - 1), ACTIONS.DROP);
  assert.equal(gate.isPaused(), false, "chưa tới HARD thì không được tạm dừng capture");
  assert.equal(gate.snapshot().droppedFrames, 2);
});

test("đạt HARD thì tạm dừng và đếm đúng một lần", () => {
  const gate = createGate({});
  assert.equal(gate.evaluate(HARD), ACTIONS.PAUSE);
  assert.equal(gate.isPaused(), true);
  assert.equal(gate.snapshot().pauseCount, 1);
  // Vẫn quá tải: không được đếm pause lần nữa.
  assert.equal(gate.evaluate(HARD + SOFT), ACTIONS.HOLD);
  assert.equal(gate.snapshot().pauseCount, 1);
});

test("HYSTERESIS: xuống giữa SOFT và HARD thì VẪN pause, không rung pause/resume", () => {
  const gate = createGate({});
  gate.evaluate(HARD);
  assert.equal(gate.isPaused(), true);
  // Hàng đợi đã rút bớt nhưng chưa dưới SOFT -> phải giữ nguyên trạng thái pause.
  assert.equal(gate.evaluate(SOFT), ACTIONS.HOLD);
  assert.equal(gate.evaluate(SOFT + 1), ACTIONS.HOLD);
  assert.equal(gate.isPaused(), true);
  assert.equal(gate.snapshot().resumeCount, 0);
});

test("P0 hồi quy: gate TỰ thoát pause khi bufferedAmount < SOFT (không cần frame mới)", () => {
  const gate = createGate({});
  assert.equal(gate.evaluate(HARD), ACTIONS.PAUSE);
  // Đây chính là tình huống trước kia bị kẹt: không còn SEND_BINARY nào tới nữa.
  // Timer của service worker gọi evaluate() với bufferedAmount đã rút hết.
  assert.equal(gate.evaluate(0), ACTIONS.RESUME);
  assert.equal(gate.isPaused(), false, "PHẢI thoát pause để capture chạy lại");
  assert.equal(gate.snapshot().resumeCount, 1);
  // Sau khi resume thì frame kế tiếp đi đường bình thường.
  assert.equal(gate.evaluate(0), ACTIONS.SEND);
});

test("chu kỳ pause -> resume -> pause đếm đúng, không cộng dồn sai", () => {
  const gate = createGate({});
  gate.evaluate(HARD);
  gate.evaluate(0);
  gate.evaluate(HARD);
  gate.evaluate(0);
  const snap = gate.snapshot();
  assert.equal(snap.pauseCount, 2);
  assert.equal(snap.resumeCount, 2);
  assert.equal(snap.paused, false);
});

test("pausedMs tích luỹ theo đồng hồ và không âm", () => {
  const clock = fakeClock();
  const gate = createGate({ now: clock.now });
  gate.evaluate(HARD);          // bắt đầu pause tại t=1000
  clock.advance(1500);          // t=2500
  const mid = gate.snapshot();
  assert.equal(mid.pausedMs, 1500);
  clock.advance(500);           // t=3000
  assert.equal(gate.snapshot().pausedMs, 2000, "snapshot lặp lại phải cộng dồn đúng");
  gate.evaluate(0);             // resume tại t=3000
  assert.equal(gate.snapshot().pausedMs, 2000);
  assert.equal(gate.isPaused(), false);
});

test("reset() xoá trạng thái pause nhưng giữ số đếm (đổi kết nối)", () => {
  const gate = createGate({});
  gate.evaluate(HARD);
  gate.reset();
  assert.equal(gate.isPaused(), false);
  assert.equal(gate.snapshot().pauseCount, 1, "số đếm là số liệu tích luỹ, không reset");
  assert.equal(gate.evaluate(0), ACTIONS.SEND);
});

test("ngưỡng tuỳ chỉnh được và hard luôn > soft", () => {
  const gate = createGate({ softLimit: 100, hardLimit: 400 });
  assert.equal(gate.softLimit, 100);
  assert.equal(gate.hardLimit, 400);
  assert.equal(gate.evaluate(100), ACTIONS.DROP);
  assert.equal(gate.evaluate(400), ACTIONS.PAUSE);
  assert.equal(gate.evaluate(99), ACTIONS.RESUME);

  // hard <= soft là cấu hình sai -> tự nâng để hysteresis luôn có hiệu lực.
  const broken = createGate({ softLimit: 1000, hardLimit: 10 });
  assert.ok(broken.hardLimit > broken.softLimit);
});
