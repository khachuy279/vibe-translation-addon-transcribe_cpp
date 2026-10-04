/**
 * demo_client_sync.js — Mô phỏng `SubtitleTimelineQueue` của Extension với DỮ LIỆU THẬT trong log.
 *
 * CHẨN ĐOÁN: vì sao câu "When you're with me, baby." (backend ĐÃ gửi) không hiện lên phụ đề?
 *
 * Cách chạy (không cần Extension, không cần backend):
 *   node backend/tests/js/demo_client_sync.js
 *
 * Cách làm: nạp ĐÚNG file `extension_firefox/lib/lookahead-timeline.js`, dựng một <video> giả,
 * đẩy các item phụ đề đúng như log backend (đã dựng lại từ `[SEG_BATCH]` + `[TRANSLATE]`), rồi
 * chạy vòng tick mô phỏng playhead. Cuối cùng in ra: item nào KHÔNG BAO GIỜ được vẽ.
 *
 * Các mốc tuyệt đối của item lấy theo quan hệ THẬT đã đo được:
 *   `mốc_hiển_thị = pts_start_khối + thời_điểm_tương_đối_của_aligner + 0.05s (VAD start pad)`
 * Không thể dựng lại chính xác từng số từ log (log không in mốc từng câu), nên ở đây dùng
 * mốc TỐT NHẤT có thể suy ra và ghi rõ giả định. Mục tiêu là kiểm tra LOGIC của client, không
 * phải đo lại mốc.
 */

const path = require("path");

// Node không có `requestAnimationFrame`/`setTimeout` vòng render của trình duyệt: cấp bản giả
// để nạp ĐÚNG file timeline của Extension mà không cần trình duyệt.
global.requestAnimationFrame = () => 0;
global.cancelAnimationFrame = () => {};

const { SubtitleTimelineQueue } = require(
  path.resolve(__dirname, "../../../extension_firefox/lib/lookahead-timeline.js")
);

/** <video> giả tối thiểu mà `SubtitleTimelineQueue` cần. */
function fakeVideo() {
  const listeners = {};
  return {
    currentTime: 0,
    paused: false,
    playbackRate: 1.0,
    addEventListener: (t, fn) => { (listeners[t] = listeners[t] || []).push(fn); },
    removeEventListener: () => {},
    _emit: (t) => { (listeners[t] || []).forEach((fn) => fn()); },
  };
}

/**
 * Dựng lại danh sách item gửi về client từ log thật.
 * Mốc hiển thị trong log chỉ có mốc KHỐI; mốc từng câu do aligner quyết định. Ở đây dùng mốc
 * TỐT NHẤT suy ra được (khớp thứ tự thời gian thật của video), với cờ `assumed: true` cho những
 * câu phải ước lượng.
 */
const ITEMS = [
  // khối [2.84 → 15.40]
  { start: 4.30, end: 12.00, text: "I can actually feel the toxins being pulled out of my skin." },
  { start: 12.08, end: 15.36, text: "Well, this is a moisturizing mask." },
  // khối [15.40 → 32.97]
  { start: 16.10, end: 19.20, text: "Ah well, then I can actually feel the moisture going into my skin." },
  { start: 19.60, end: 21.40, text: "Hey, I hope you don't mind." },
  { start: 21.60, end: 24.20, text: "I used a little of your eye cream last night." },
  { start: 24.40, end: 27.40, text: "I thought someone looked brighter and tighter." },
  // khối [31.97 → 44.22]
  { start: 33.20, end: 35.60, text: "Still like to know who Jerry is." },
  { start: 36.00, end: 38.00, text: "Don't worry about it." },
  { start: 38.20, end: 41.60, text: "Hey, after this, how about we all go out and do something together?" },
  { start: 41.80, end: 43.00, text: "That would be great." },
  { start: 43.10, end: 44.10, text: "Thank you." },
  // khối [55.45 → 68.06]
  { start: 56.80, end: 58.90, text: "I think about you day and night." },
  { start: 59.10, end: 62.20, text: "It's only right to think about the good and old times." },
  { start: 62.40, end: 64.20, text: "So happy together." },
  // khối [67.06 → 79.06]   ← CÂU BỊ MẤT
  { start: 70.20, end: 76.40, text: "I can't see loving nobody but you for all my life." },
  { start: 76.60, end: 79.06, text: "When you're with me, baby." },
  // khối [78.06 → 90.25]
  { start: 79.20, end: 81.00, text: "This is Rebecca." },
  { start: 81.10, end: 82.00, text: "Hi." },
];

function run() {
  const video = fakeVideo();
  const shown = [];
  let lastShown = null;

  const queue = new SubtitleTimelineQueue({
    onSubtitleChange: (sub) => {
      const t = video.currentTime;
      if (sub) {
        if (!lastShown || lastShown.id !== sub.id) {
          shown.push({ at: Number(t.toFixed(2)), from: sub.start_pts, to: sub.end_pts, text: sub.original_text });
          lastShown = sub;
        }
      } else {
        lastShown = null;
      }
    },
    onBufferingStateChange: () => {},
  });
  queue.attachVideo(video);

  // Đẩy phụ đề theo nhịp backend: khối được xử lý TRƯỚC khi playhead tới (lookahead).
  const delivery = ITEMS.map((it, i) => ({ it, at: 0.5 + i * 0.35 }));
  let di = 0;

  const stepMs = 40; // 25 fps
  const endSec = 84.0;
  for (let t = 2.8; t <= endSec; t += stepMs / 1000) {
    video.currentTime = t;
    while (di < delivery.length && delivery[di].at <= t) {
      queue.addSubtitles([{
        start_pts: delivery[di].it.start,
        end_pts: delivery[di].it.end,
        original_text: delivery[di].it.text,
        translated_text: "[vi] " + delivery[di].it.text,
      }], "init_0");
      di++;
    }
    queue._tick();
  }

  const shownIds = new Set(shown.map((s) => s.text));
  console.log("── Các câu ĐÃ HIỆN (theo thứ tự thời gian) ──");
  for (const s of shown) {
    console.log(`   @${String(s.at).padStart(6)}s  cửa sổ [${s.from.toFixed(2)} → ${s.to.toFixed(2)}]  ${s.text}`);
  }
  console.log("\n── Các câu backend ĐÃ GỬI nhưng KHÔNG BAO GIỜ hiện ──");
  let missing = 0;
  for (const it of ITEMS) {
    if (!shownIds.has(it.text)) {
      missing++;
      console.log(`   ✗ [${it.start.toFixed(2)} → ${it.end.toFixed(2)}]  ${it.text}`);
    }
  }
  if (!missing) console.log("   (không có)");
  queue.detach();
}

run();
