// Backpressure gate — máy trạng thái THUẦN cho đường gửi audio của extension.
//
// VÌ SAO CẦN (FIX-01, lỗi P0 đã xác nhận):
// Trước đây `service-worker.js` đặt `pausedByBackpressure = true` khi `bufferedAmount`
// vượt HARD, nhưng cờ đó CHỈ được xoá bên trong `sendOrQueue()`. Mà `sendOrQueue()` chỉ
// được gọi từ `SEND_BINARY`. Khi content script nhận thông báo "paused", nó đặt
// `audioCapture.isCapturing = false` ⇒ worklet ngừng phát chunk ⇒ KHÔNG còn `SEND_BINARY`
// ⇒ cờ pause không bao giờ được xoá ⇒ **capture treo vĩnh viễn** cho tới khi user Stop/Start.
// (`FLUSH_PENDING` có implement nhưng không có producer nào — đã grep toàn bộ extension.)
//
// Cách sửa: tách logic ngưỡng thành module thuần, có HYSTERESIS
// (vào PAUSE ở >= HARD, chỉ ra khỏi PAUSE khi < SOFT) để không rung pause/resume,
// rồi để service worker tự kiểm tra lại theo timer thay vì phụ thuộc vào frame mới.
//
// Module này KHÔNG phụ thuộc WebSocket/browser nên test được bằng `node --test`:
//   node --test extension_firefox/tests/

(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.BackpressureGate = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // Ngưỡng mặc định (byte). SOFT ~ 4 s audio 16 kHz PCM16, HARD ~ 16 s.
  const SOFT_LIMIT_BYTES = 128 * 1024;
  const HARD_LIMIT_BYTES = 512 * 1024;

  const ACTIONS = {
    SEND: "send",     // gửi ngay
    DROP: "drop",     // giữ frame MỚI NHẤT, bỏ frame đang chờ
    PAUSE: "pause",   // vừa chuyển sang tạm dừng capture
    HOLD: "hold",     // vẫn đang tạm dừng, chưa rút đủ hàng đợi
    RESUME: "resume", // vừa thoát tạm dừng -> cho capture chạy lại
  };

  /**
   * Tạo một gate độc lập (mỗi kết nối WS bridge một gate).
   *
   * @param {object} [options]
   * @param {number} [options.softLimit] ngưỡng SOFT (byte)
   * @param {number} [options.hardLimit] ngưỡng HARD (byte), phải > softLimit
   * @param {function(): number} [options.now] đồng hồ tiêm được (test)
   */
  function createGate(options) {
    const opts = options || {};
    const soft = Number(opts.softLimit) > 0 ? Number(opts.softLimit) : SOFT_LIMIT_BYTES;
    const rawHard = Number(opts.hardLimit);
    const hard = rawHard > soft ? rawHard : Math.max(soft * 4, HARD_LIMIT_BYTES);
    const now = typeof opts.now === "function" ? opts.now : function () { return Date.now(); };

    let paused = false;
    let pausedSince = 0;
    let pausedMs = 0;
    let pauseCount = 0;
    let resumeCount = 0;
    let droppedFrames = 0;

    /** Cộng dồn thời gian đang pause và dời mốc, để `snapshot()` gọi nhiều lần vẫn đúng. */
    function foldPausedTime() {
      if (!paused || pausedSince <= 0) return;
      const at = now();
      if (at > pausedSince) pausedMs += at - pausedSince;
      pausedSince = at;
    }

    /**
     * Đánh giá một frame theo `bufferedAmount` hiện tại.
     * Trả về một trong `ACTIONS`.
     */
    function evaluate(bufferedAmount) {
      const buffered = Number(bufferedAmount) || 0;

      if (paused) {
        if (buffered < soft) {
          // HYSTERESIS: chỉ nhả khi đã rút xuống DƯỚI SOFT.
          paused = false;
          resumeCount += 1;
          foldPausedTime();
          pausedSince = 0;
          return ACTIONS.RESUME;
        }
        return ACTIONS.HOLD;
      }

      if (buffered >= hard) {
        paused = true;
        pauseCount += 1;
        pausedSince = now();
        droppedFrames += 1;
        return ACTIONS.PAUSE;
      }

      if (buffered >= soft) {
        droppedFrames += 1;
        return ACTIONS.DROP;
      }

      return ACTIONS.SEND;
    }

    function isPaused() {
      return paused;
    }

    /** Đưa về trạng thái ban đầu (dùng khi mở kết nối mới). Số đếm tích luỹ được giữ. */
    function reset() {
      foldPausedTime();
      paused = false;
      pausedSince = 0;
    }

    /** Ảnh chụp để ghi log/metric. Không làm thay đổi hành vi gửi. */
    function snapshot() {
      foldPausedTime();
      return {
        paused: paused,
        pauseCount: pauseCount,
        resumeCount: resumeCount,
        droppedFrames: droppedFrames,
        pausedMs: Math.round(pausedMs),
        softLimit: soft,
        hardLimit: hard,
      };
    }

    return {
      evaluate: evaluate,
      isPaused: isPaused,
      reset: reset,
      snapshot: snapshot,
      softLimit: soft,
      hardLimit: hard,
      actions: ACTIONS,
    };
  }

  return {
    createGate: createGate,
    ACTIONS: ACTIONS,
    SOFT_LIMIT_BYTES: SOFT_LIMIT_BYTES,
    HARD_LIMIT_BYTES: HARD_LIMIT_BYTES,
  };
});
