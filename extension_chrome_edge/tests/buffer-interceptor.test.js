// Test cho Buffer Interceptor (script tiêm vào MAIN world) — trọng tâm là MỐC MEDIA THẬT.
//
//   node --test extension_firefox/tests/
//   node extension_firefox/tests/buffer-interceptor.test.js
//
// Hồi quy 2026-10-03 ("phiên 2 của cùng video không lấy được âm thanh"):
//   Interceptor cũ gắn mỗi mảnh audio bằng `video.currentTime` TẠI LÚC append. Trình phát luôn
//   tải TRƯỚC, nên mốc đó KHÔNG phải mốc của dữ liệu trong mảnh. Khi mở phiên mới giữa video,
//   bộ lọc replay theo `videoPts` chọn đúng những mảnh đang TẢI TRƯỚC (audio ở tương lai) và
//   bỏ đúng những mảnh chứa audio NGAY TẠI playhead ⇒ backend giải mã ra PCM kết thúc ở 110s
//   trong khi playhead ở 38,5s ("Đã dịch: 0.0s"), video kẹt ở mốc cũ.
//
//   Nay mốc media được lấy từ `SourceBuffer.buffered` (hiệu số trước/sau khi append) — nguồn
//   sự thật duy nhất. Các test dưới đây đều FAIL với bản cũ.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const { LookaheadClient, payloadInReplayWindow } = require("../lib/lookahead-client.js");

const CODE = fs.readFileSync(
  path.join(__dirname, "..", "content", "buffer_interceptor_poc.js"),
  "utf8"
);

// ── Giả lập trang web (MSE + <video>) ────────────────────────────────────────
function makeTimeRanges(ranges) {
  return {
    length: ranges.length,
    start: (i) => ranges[i][0],
    end: (i) => ranges[i][1],
  };
}

function makeVideo({ currentTime = 0, buffered = [], paused = false } = {}) {
  return {
    paused,
    currentTime,
    clientWidth: 1280,
    clientHeight: 720,
    buffered: makeTimeRanges(buffered),
    play() {},
    src: "blob:https://www.youtube.com/xyz",
    currentSrc: "blob:https://www.youtube.com/xyz",
  };
}

function createPage() {
  const messages = [];
  const listeners = [];
  const logs = [];
  const videos = [];
  let intervalCount = 0;

  const window = {
    postMessage(msg) { messages.push(msg); },
    addEventListener(type, fn) { if (type === "message") listeners.push(fn); },
    removeEventListener(type, fn) {
      const i = listeners.indexOf(fn);
      if (i >= 0) listeners.splice(i, 1);
    },
  };

  const document = {
    querySelectorAll: (sel) => (sel === "video" ? videos.slice() : []),
    getElementById: () => null,
    addEventListener() {},
    fullscreenElement: null,
  };

  class FakeSourceBuffer {
    constructor(mimeType) {
      this.mimeType = mimeType;
      this.timestampOffset = 0;
      this.updating = false;
      this.nextMediaRange = null;
      this.removed = [];
      this.lastAppended = null;
      this._ranges = [];
      this._listeners = Object.create(null);
    }
    get buffered() { return makeTimeRanges(this._ranges); }
    addEventListener(type, fn) {
      (this._listeners[type] = this._listeners[type] || []).push(fn);
    }
    removeEventListener(type, fn) {
      const arr = this._listeners[type];
      if (!arr) return;
      const i = arr.indexOf(fn);
      if (i >= 0) arr.splice(i, 1);
    }
    _emit(type) {
      for (const fn of (this._listeners[type] || []).slice()) fn({ type });
    }
    // "Bản gốc" mà interceptor bọc quanh: thêm khoảng media rồi phát `updateend`.
    appendBuffer(data) {
      if (this.nextMediaRange) {
        this._ranges.push([this.nextMediaRange[0], this.nextMediaRange[1]]);
        this._ranges.sort((a, b) => a[0] - b[0]);
        this.nextMediaRange = null;
      }
      // Mô phỏng trình phát chỉnh `timestampOffset` NGAY SAU `appendBuffer()` (trình duyệt áp
      // giá trị cuối cho chính mảnh đang chờ).
      if (this.nextTimestampOffset !== undefined && this.nextTimestampOffset !== null) {
        this.timestampOffset = this.nextTimestampOffset;
        this.nextTimestampOffset = null;
      }
      this.lastAppended = data;
      this._emit("updateend");
    }
    remove(start, end) {
      this.removed.push([start, end]);
      const out = [];
      for (const [s, e] of this._ranges) {
        if (e <= start || s >= end) { out.push([s, e]); continue; }
        if (s < start) out.push([s, start]);
        if (e > end) out.push([end, e]);
      }
      this._ranges = out;
      this._emit("updateend");
    }
  }

  class FakeMediaSource {
    addSourceBuffer(mimeType) { return new FakeSourceBuffer(mimeType); }
  }

  window.MediaSource = FakeMediaSource;
  window.SourceBuffer = FakeSourceBuffer;

  const sandbox = {
    console: {
      log: (...a) => logs.push(a.join(" ")),
      warn: (...a) => logs.push(a.join(" ")),
      error: (...a) => logs.push(a.join(" ")),
      group() {}, groupEnd() {}, table() {},
    },
    window,
    document,
    location: { href: "https://www.youtube.com/watch?v=abc" },
    MediaSource: FakeMediaSource,
    SourceBuffer: FakeSourceBuffer,
    setInterval: () => { intervalCount += 1; return intervalCount; },
    clearInterval: () => {},
    setTimeout: () => 0,
    clearTimeout: () => {},
    URL,
    TextEncoder,
  };
  vm.createContext(sandbox);
  // Tạo buffer TRONG realm của sandbox: `data instanceof ArrayBuffer` trong interceptor sẽ
  // trượt nếu ta truyền ArrayBuffer của realm Node.
  const mkBytes = vm.runInContext("(n) => new Uint8Array(n)", sandbox);

  function run() {
    vm.runInContext(CODE, sandbox, { filename: "buffer_interceptor_poc.js" });
  }

  function dispatch(data) {
    for (const fn of listeners.slice()) fn({ source: window, data });
  }

  return {
    window,
    videos,
    messages,
    logs,
    run,
    listenerCount: () => listeners.length,
    intervalCount: () => intervalCount,
    setVideo(video) { videos.length = 0; videos.push(video); return video; },
    newSourceBuffer(mime = 'audio/webm; codecs="opus"') {
      return new FakeMediaSource().addSourceBuffer(mime);
    },
    bytes(n) { return mkBytes(n).buffer; },
    bytesOf(list, size = list.length) {
      const u8 = mkBytes(size);
      u8.set(list);
      return u8.buffer;
    },
    /** Append một mảnh: `mediaRange` = khoảng media mà trình duyệt sẽ đưa vào `buffered`. */
    append(sb, byteLen, mediaRange, offsetAtUpdateEnd) {
      sb.nextMediaRange = mediaRange || null;
      sb.nextTimestampOffset = offsetAtUpdateEnd;
      sb.appendBuffer(this.bytes(byteLen));
    },
    requestChunks(currentTime, token, force = true) {
      dispatch({
        source: "VIBE_LOOKAHEAD_CLIENT",
        type: "REQUEST_INITIAL_CHUNKS",
        currentTime,
        replayToken: token,
        force,
      });
    },
    resetReplayToken() {
      dispatch({ source: "VIBE_LOOKAHEAD_CLIENT", type: "RESET_REPLAY_TOKEN" });
    },
    audioMessages() {
      return messages.filter((m) => m && m.type === "AUDIO_CHUNK_INTERCEPTED");
    },
    clearMessages() { messages.length = 0; },
    debug: () => window.__VIBE_LOOKAHEAD_DEBUG__,
    state: () => window.__VIBE_LOOKAHEAD_DEBUG__.getState(),
  };
}

// ── 1. Gắn mốc media thật ───────────────────────────────────────────────────
test("mảnh audio được gắn MỐC MEDIA THẬT từ SourceBuffer.buffered, không phải vị trí phát", () => {
  const page = createPage();
  page.setVideo(makeVideo({ currentTime: 5, buffered: [[0, 20]] }));
  page.run();

  const sb = page.newSourceBuffer();
  page.append(sb, 100, [60, 65]);

  const audios = page.audioMessages();
  assert.equal(audios.length, 1, "phải gửi đúng một mảnh audio");
  assert.equal(audios[0].payload.mediaStart, 60, "mốc media bắt đầu phải lấy từ `buffered`");
  assert.equal(audios[0].payload.mediaEnd, 65, "mốc media kết thúc phải lấy từ `buffered`");
  assert.equal(audios[0].payload.videoPts, 5, "`videoPts` vẫn giữ để tương thích ngược");
});

// ── 2. HỒI QUY CHÍNH: replay của phiên mở giữa video ────────────────────────
test("replay phiên mở GIỮA video chọn mảnh theo mốc media (không theo vị trí phát)", () => {
  const page = createPage();
  const video = page.setVideo(makeVideo({ currentTime: 5, buffered: [[0, 60]] }));
  page.run();

  const sb = page.newSourceBuffer();
  // Mảnh A: append lúc đang phát 5s nhưng CHỨA media 30–50s — đúng vùng phiên mới cần.
  page.append(sb, 64, [30, 50]);
  // Mảnh B: append lúc đang phát 35s nhưng chứa media 90–110s (trình phát tải trước).
  video.currentTime = 35;
  page.append(sb, 64, [90, 110]);

  page.clearMessages();
  video.currentTime = 38.5;
  page.requestChunks(38.5, "init_session2");

  const audios = page.audioMessages();
  assert.equal(audios.length, 1, "chỉ được gửi mảnh THỰC SỰ chứa audio tại playhead");
  assert.deepEqual(
    [audios[0].payload.mediaStart, audios[0].payload.mediaEnd],
    [30, 50],
    "mảnh được chọn phải là mảnh phủ 38,5s, KHÔNG phải mảnh tải trước 90–110s"
  );
});

test("replay vẫn phủ được playhead khi mảnh được append từ TRƯỚC đó rất lâu", () => {
  const page = createPage();
  const video = page.setVideo(makeVideo({ currentTime: 0.5, buffered: [[0, 120]] }));
  page.run();

  const sb = page.newSourceBuffer();
  page.append(sb, 64, [0, 40]);   // mảnh dài phủ tận 40s, append lúc video còn ở 0,5s

  page.clearMessages();
  video.currentTime = 38.5;       // phiên mới mở ở 38,5s — trong lòng mảnh đã append
  page.requestChunks(38.5, "init_late");

  const audios = page.audioMessages();
  assert.equal(audios.length, 1);
  assert.equal(audios[0].payload.mediaEnd, 40);
});

// ── 3. Init segment ─────────────────────────────────────────────────────────
test("init segment luôn được gửi TRƯỚC các mảnh media", () => {
  const page = createPage();
  page.setVideo(makeVideo({ currentTime: 10, buffered: [[0, 40]] }));
  page.run();

  const sb = page.newSourceBuffer();
  // EBML header WebM, KHÔNG chứa Cluster (0x1F43B675) ⇒ được nhận diện là init segment.
  sb.nextMediaRange = null;
  sb.appendBuffer(page.bytesOf([0x1a, 0x45, 0xdf, 0xa3], 16));
  page.append(sb, 64, [10, 20]);

  page.clearMessages();
  page.requestChunks(10, "init_t");

  const msgs = page.audioMessages();
  assert.ok(msgs.length >= 2, "phải gửi init segment + mảnh media");
  assert.equal(msgs[0].payload.isInit, true, "init segment phải đi trước");
  assert.equal(msgs[1].payload.isInit, false);
});

// ── 4. Cache: giữ mảnh phủ vùng đệm của trình phát ──────────────────────────
test("cache giữ mảnh phủ vùng đệm của trình phát dù tâm cửa sổ (videoPts) sai", () => {
  const page = createPage();
  // `getActiveVideo()` có thể trả về thẻ <video> khác ⇒ videoPts ở tận 500s.
  const video = page.setVideo(makeVideo({ currentTime: 500, buffered: [[30, 110]] }));
  page.run();

  const sb = page.newSourceBuffer();
  page.append(sb, 64, [30, 50]);
  assert.equal(page.state().cachedAudioChunks.length, 1, "mảnh phủ vùng đệm phải được giữ");

  // Mảnh nằm ngoài cả cửa sổ theo videoPts lẫn vùng đệm của trình phát ⇒ bị loại.
  page.append(sb, 64, [1000, 1020]);
  assert.equal(page.state().cachedAudioChunks.length, 1, "mảnh quá xa phải bị prune");
  assert.equal(video.currentTime, 500);
});

// ── 5. Chống tiêm lặp ───────────────────────────────────────────────────────
test("tiêm lặp cùng file không nhân bản listener / không giết vòng lặp định kỳ", () => {
  const page = createPage();
  page.setVideo(makeVideo({ currentTime: 3, buffered: [[0, 30]] }));
  page.run();
  const afterFirst = page.listenerCount();
  const timersAfterFirst = page.intervalCount();

  page.resetReplayToken();
  const resetsAfterFirst = page.logs.filter((l) => l.includes("Reset replay token")).length;
  assert.equal(resetsAfterFirst, 1, "một listener ⇒ một dòng log RESET");

  page.run(); // content script + background có thể tiêm lại cùng file vào cùng frame

  assert.equal(page.listenerCount(), afterFirst, "không được thêm listener `message`");
  assert.equal(page.intervalCount(), timersAfterFirst, "không được tạo vòng lặp định kỳ mới");

  page.resetReplayToken();
  assert.equal(
    page.logs.filter((l) => l.includes("Reset replay token")).length,
    resetsAfterFirst + 1,
    "lệnh RESET không được nhân bản theo số lần tiêm (bản cũ: N listener ⇒ N dòng)"
  );
});

// ── 6. Học byte/giây cho việc tải lại theo khoảng byte ──────────────────────
test("học byte/giây thật của luồng audio từ các mảnh đã append", () => {
  const page = createPage();
  page.setVideo(makeVideo({ currentTime: 0, buffered: [[0, 40]] }));
  page.run();

  const sb = page.newSourceBuffer();
  page.append(sb, 60000, [0, 6]);
  page.append(sb, 60000, [6, 12]);

  assert.equal(page.debug().bytesPerSec(), 10000, "120000 byte / 12 giây media = 10000 B/s");
});

// ── 7. Phương án cuối: ép trình phát tải lại ────────────────────────────────
test("nudgeRefetch xoá khoảng phía trước playhead khỏi SourceBuffer để trình phát fetch lại", () => {
  const page = createPage();
  page.setVideo(makeVideo({ currentTime: 20, buffered: [[0, 60]] }));
  page.run();

  const sb = page.newSourceBuffer();
  page.append(sb, 64, [0, 60]);

  const removed = page.debug().nudgeRefetch(12);
  assert.equal(removed, 12);
  assert.deepEqual(sb.removed, [[20, 32]], "phải xoá đúng [playhead, playhead+12s]");
});

// ── 8. `timestampOffset` phải đọc ở `updateend`, không phải lúc gọi appendBuffer ──
test("timestampOffset gửi kèm là giá trị tại `updateend` (trình phát chỉnh sau khi append)", () => {
  const page = createPage();
  page.setVideo(makeVideo({ currentTime: 7.1, buffered: [[0, 60]] }));
  page.run();

  const sb = page.newSourceBuffer();
  sb.timestampOffset = 0;
  // Trình phát gọi appendBuffer() rồi mới neo segment vào timeline (offset -250) — trình duyệt
  // áp giá trị CUỐI cho chính mảnh đang chờ. Đọc sớm ⇒ backend nhận 0 và PCM lệch 250s.
  page.append(sb, 64, [0, 5.4], -250);

  const audios = page.audioMessages();
  assert.equal(audios.length, 1);
  assert.equal(audios[0].payload.timestampOffset, -250);
});

// ── 9. Header khung nhị phân mang mốc media sang backend ────────────────────
test("khung nhị phân gửi backend mang media_start/media_end (trục trình duyệt)", () => {
  const sent = [];
  const listeners = [];
  globalThis.chrome = {
    runtime: {
      connect: () => ({
        postMessage(msg) { sent.push(msg); },
        onMessage: { addListener(fn) { listeners.push(fn); } },
        onDisconnect: { addListener() {} },
        disconnect() {},
      }),
    },
  };
  try {
    const client = new LookaheadClient({
      serverUrl: "wss://localhost:8765/ws/lookahead",
      useBridge: true,
    });
    client.connect({ currentTime: 7.1, playbackRate: 1, paused: false });
    listeners.forEach((fn) => fn({ type: "connected" }));

    client._sendAudioBinaryFrame(new ArrayBuffer(16), {
      mime: "audio/mp4", timestampOffset: -250, mediaStart: 0, mediaEnd: 5.4,
    });

    const bin = sent.find((m) => m.action === "SEND_RAW_BINARY");
    assert.ok(bin, "Thiếu khung nhị phân");
    const view = new DataView(bin.buffer);
    const hdrLen = view.getUint32(0, true);
    const hdr = JSON.parse(
      new TextDecoder().decode(new Uint8Array(bin.buffer, 4, hdrLen))
    );
    assert.equal(hdr.media_start, 0);
    assert.equal(hdr.media_end, 5.4);
    assert.equal(hdr.timestamp_offset, -250);
    assert.equal(hdr.is_init, false);
    client.disconnect();
  } finally {
    delete globalThis.chrome;
  }
});

// ── 10. Hàm lọc cửa sổ replay phía client ───────────────────────────────────
test("payloadInReplayWindow ưu tiên mốc media và không coi mảnh mù mốc là 'sẽ được replay'", () => {
  // Có mốc media: xét giao nhau.
  assert.equal(payloadInReplayWindow({ mediaStart: 30, mediaEnd: 50 }, 30.5, 83.5), true);
  assert.equal(payloadInReplayWindow({ mediaStart: 90, mediaEnd: 110 }, 30.5, 83.5), false);
  // Mốc media thắng cả khi `videoPts` nói khác.
  assert.equal(
    payloadInReplayWindow({ mediaStart: 30, mediaEnd: 50, videoPts: 999 }, 30.5, 83.5),
    true
  );
  // Không có mốc media ⇒ lùi về videoPts.
  assert.equal(payloadInReplayWindow({ videoPts: 40 }, 30.5, 83.5), true);
  assert.equal(payloadInReplayWindow({ videoPts: 5 }, 30.5, 83.5), false);
  // Mù mốc hoàn toàn ⇒ KHÔNG coi là "sẽ được replay" (mảnh vẫn được gửi).
  assert.equal(payloadInReplayWindow({}, 30.5, 83.5), false);
});
