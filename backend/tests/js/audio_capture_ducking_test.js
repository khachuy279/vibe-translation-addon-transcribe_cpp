// Regression test cho ĐỒ THỊ AUDIO của audio-capture (extension_firefox/lib/audio-capture.js).
//
// BOI CANH: nguoi dung bao "0% => khong co tieng goc, khong co phu de, khong co TTS".
//
// Nguyen nhan goc: `createMediaElementSource` cho ra tin hieu DA bi nhan boi `video.volume`.
// Ban cu noi THANG `sourceNode -> destination` roi duck bang cach ha `video.volume`, nen:
//   - nhanh CAPTURE (ASR) lay tu cung `sourceNode` => bi ha theo;
//   - o 0% => ASR nhan IM LANG KY THUAT SO => VAD khong thay tieng noi
//     => khong ASR => khong phu de => khong dich => khong TTS.
//
// Ban sua tach hai nhanh:
//   sourceNode ─┬─> duckGain ──> destination     (nguoi dung nghe, slider dieu khien)
//               └─> workletNode ──> ASR          (luon full-scale)
//
// Cach chay: node backend/tests/js/audio_capture_ducking_test.js

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const SRC = path.join(__dirname, "..", "..", "..", "extension_firefox", "lib", "audio-capture.js");

// ───────────────────────────────────────────────────── mock Web Audio API
const graph = [];

function makeNode(name) {
  return {
    _name: name,
    connect(dst) {
      graph.push(`${name}->${dst && dst._name ? dst._name : "?"}`);
      return dst;
    },
    disconnect() {
      graph.push(`${name}->DISCONNECT`);
    },
  };
}

function makeMockCtx() {
  const ctx = {
    state: "running",
    sampleRate: 48000,
    currentTime: 0,
    destination: makeNode("destination"),
    audioWorklet: { addModule: async () => {} },
    createMediaElementSource() {
      return makeNode("sourceNode");
    },
    createGain() {
      const n = makeNode("duckGain");
      n.gain = {
        value: 1.0,
        setTargetAtTime(v) {
          this.value = v;
        },
      };
      return n;
    },
    createScriptProcessor() {
      return makeNode("scriptProcessor");
    },
    resume: async () => {},
    close: async () => {},
  };
  return ctx;
}

// Cac global ma audio-capture.js can.
globalThis.window = {
  AudioContext: function () {
    return globalThis.__nextCtx || makeMockCtx();
  },
};
globalThis.HTMLMediaElement = function () {};
globalThis.AudioWorkletNode = function () {
  this.port = { onmessage: null, postMessage() {} };
  this._name = "workletNode";
  this.connect = (d) => {
    graph.push(`workletNode->${d && d._name ? d._name : "?"}`);
    return d;
  };
  this.disconnect = () => {
    graph.push("workletNode->DISCONNECT");
  };
};
// `runtime.getURL` de nap module worklet.
globalThis.browser = { runtime: { getURL: (p) => `moz-extension://test/${p}` } };
globalThis.self = globalThis;

eval(fs.readFileSync(SRC, "utf8"));
const AudioCapture = globalThis.AudioCapture;
assert.ok(AudioCapture, "khong nap duoc AudioCapture");

function mockVideoEl(volume = 1.0) {
  return {
    volume,
    muted: false,
    isConnected: true,
    play() {},
    _name: "video",
  };
}

let n = 0;
const ok = (m) => {
  n++;
  console.log("  PASS  " + m);
};
const has = (edge) => graph.includes(edge);
const show = () => console.log("        graph: " + graph.join(", "));

(async () => {
  console.log("=".repeat(90));
  console.log("DO THI AUDIO: capture(ASR) PHAI tach khoi nhanh nghe(ducking)");
  console.log("=".repeat(90));

  const ctx = makeMockCtx();
  globalThis.__nextCtx = ctx;
  const video = mockVideoEl(1.0);
  const cap = new AudioCapture();

  await cap.start(video);
  show();

  // 1. Nhanh NGHE phai di QUA duckGain - khong duoc noi thang vao destination.
  assert.ok(has("sourceNode->duckGain"), "sourceNode phai noi vao duckGain");
  assert.ok(has("duckGain->destination"), "duckGain phai noi vao destination");
  assert.ok(!has("sourceNode->destination"), "KHONG duoc noi thang sourceNode -> destination (loi cu)");
  ok("nhanh nghe di qua duckGain, khong noi thang destination");

  // 2. Nhanh CAPTURE (ASR) lay TRUC TIEP tu sourceNode => truoc gain => luon full-scale.
  assert.ok(has("sourceNode->workletNode"), "ASR phai lay truc tiep tu sourceNode (truoc gain)");
  assert.ok(!has("duckGain->workletNode"), "KHONG duoc lay ASR sau gain");
  ok("nhanh ASR lay truoc gain => ducking khong lam nho audio cho ASR");

  // 3. Ducking dieu khien gain, KHONG dung video.volume.
  assert.strictEqual(cap.supportsGainDucking(), true, "phai bao ho tro gain ducking");
  assert.strictEqual(cap.setDuckLevel(0.05), true);
  assert.strictEqual(ctx.createGain().gain.value, 1.0, "gain moi mac dinh 1.0");
  assert.strictEqual(cap.duckLevel, 0.05, "duckLevel phai la 0.05");
  assert.strictEqual(video.volume, 1.0, "video.volume KHONG duoc dung");
  ok("setDuckLevel(0.05): chi doi gain, video.volume giu 1.0");

  // 4. 0% = tat nhanh nghe nhung van con audio cho ASR.
  cap.setDuckLevel(0);
  assert.strictEqual(cap.duckLevel, 0, "0% => gain 0");
  assert.strictEqual(video.volume, 1.0, "0% KHONG duoc dung video.volume");
  ok("0%: chi tat nhanh nghe, ASR van nhan audio");

  // 5. stop() phai tra gain ve 1.0 va noi lai QUA gain (khong noi thang destination).
  graph.length = 0;
  cap.stop();
  show();
  assert.strictEqual(video.volume, 1.0, "stop khong duoc doi video.volume");
  assert.ok(
    has("sourceNode->duckGain"),
    "stop phai noi lai QUA duckGain (neu noi thang destination thi lan capture sau mat ducking)"
  );
  assert.ok(!has("sourceNode->destination"), "stop KHONG duoc noi thang vao destination");
  ok("stop: noi lai qua duckGain (khong phai thang destination)");

  // 6. Gain phai duoc tra ve 1.0 khi stop.
  assert.strictEqual(
    video.__bsDuckGain.gain.value,
    1.0,
    "stop phai tra gain ve 1.0 de tieng goc khong bi nho vinh vien"
  );
  ok("stop: tra gain ve 1.0");

  // 7. Sau stop, capture khong con bao ho tro gain ducking.
  assert.strictEqual(cap.supportsGainDucking(), false, "sau stop khong con duck bang gain");
  ok("sau stop: duckGain da duoc nha");

  console.log("\n" + "=".repeat(90));
  console.log(`KET QUA: PASS (${n} nhom kiem tra)`);
  console.log("=".repeat(90));
})().catch((e) => {
  console.error("\nFAIL:", e && e.message);
  process.exit(1);
});
