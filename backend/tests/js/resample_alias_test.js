// QWEN-E5 — kiểm thử CHỨC NĂNG chống aliasing của resampler (chạy bằng Node).
//
// Vấn đề: `lib/audio-processor.js` resample 48 kHz -> 16 kHz với `ratio = 3.0` NGUYÊN.
// `src` luôn là số nguyên (`phase` nguyên + `ratio` nguyên) ⇒ `frac` LUÔN = 0 ⇒ vòng nội
// suy tuyến tính suy biến thành "giữ 1 mẫu, bỏ 2 mẫu" — decimation TRẦN, không lọc. Mọi
// năng lượng trên Nyquist đích (8 kHz) GẤP NGƯỢC vào dải thoại 0–8 kHz: âm 9 kHz thành
// 7 kHz, âm 12 kHz thành 4 kHz... làm méo phổ mà ASR phải ăn.
//
// Cách kiểm: đưa sine CAO TẦN (trên 8 kHz) vào worklet. Ở đầu ra 16 kHz, tín hiệu đó chỉ
// có thể tồn tại dưới dạng ẢNH GẬP. Đo RMS đầu ra:
//   • decimation trần  : RMS ≈ biên độ vào (không suy giảm gì)
//   • có boxcar 3 tap  : RMS ≈ |H(f)| × biên độ vào (H(9k) ≈ 0,59; H(12k) ≈ 0,33)
// Đồng thời phải chứng minh KHÔNG làm hỏng dải thoại: sine 440 Hz giữ nguyên biên độ.
//
// Dùng: node backend/tests/js/resample_alias_test.js  (exit 0 = PASS)

const fs = require("fs");
const path = require("path");
const vm = require("vm");

function findWorklet() {
  let dir = __dirname;
  for (let i = 0; i < 6; i++) {
    const candidate = path.join(dir, "extension_firefox", "lib", "audio-processor.js");
    if (fs.existsSync(candidate)) return candidate;
    dir = path.dirname(dir);
  }
  throw new Error("không tìm thấy extension_firefox/lib/audio-processor.js");
}

const WORKLET = process.argv[2] || findWorklet();
const QUANTUM = 128;
const CHUNK = 1024;

const failures = [];
function check(cond, msg, extra) {
  if (cond) {
    console.log("  PASS " + msg);
  } else {
    console.log("  FAIL " + msg + (extra !== undefined ? `  (${extra})` : ""));
    failures.push(msg);
  }
}

// ---------------------------------------------------------------- scope giả
function makeScope(sampleRate) {
  const scope = { sampleRate, currentTime: 0, registered: {}, messages: [] };
  class FakePort {
    postMessage(payload) {
      scope.messages.push({ payload });
    }
  }
  class AudioWorkletProcessor {
    constructor() {
      this.port = new FakePort();
    }
  }
  scope.AudioWorkletProcessor = AudioWorkletProcessor;
  scope.registerProcessor = (name, cls) => {
    scope.registered[name] = cls;
  };
  const src = fs.readFileSync(WORKLET, "utf8");
  vm.runInContext(src, vm.createContext(scope), { filename: WORKLET });
  return scope;
}

const TARGET_RATE = 16000;

/** Chạy sine `freq` qua worklet thật, trả mảng mẫu float ở đầu ra. */
function runTone(sampleRate, freq, amplitude, seconds, targetRate = TARGET_RATE) {
  const scope = makeScope(sampleRate);
  const Proc = scope.registered["audio-capture-processor"];
  const proc = new Proc({ processorOptions: { targetSampleRate: targetRate, chunkSize: CHUNK } });

  const total = Math.floor(sampleRate * seconds);
  let n = 0;
  let phase = 0;
  while (n < total) {
    const block = new Float32Array(QUANTUM);
    for (let i = 0; i < QUANTUM && n < total; i++, n++, phase++) {
      block[i] = amplitude * Math.sin((2 * Math.PI * freq * phase) / sampleRate);
    }
    scope.currentTime += QUANTUM / sampleRate;
    proc.process([[block]], [[new Float32Array(QUANTUM)]], {});
  }
  // ép xả hết chunk còn lại
  proc.process([[new Float32Array(QUANTUM)]], [[new Float32Array(QUANTUM)]], {});

  const out = [];
  for (const m of scope.messages) {
    if (m.payload.type !== "audio_chunk") continue;
    const i16 = new Int16Array(m.payload.buffer);
    for (let i = 0; i < i16.length; i++) out.push(i16[i] / 32767);
  }
  return out;
}

function decimateRaw(sampleRate, freq, amplitude, seconds, ratio) {
  const total = Math.floor(sampleRate * seconds);
  const out = [];
  for (let i = 0; i < total; i += ratio) {
    out.push(amplitude * Math.sin((2 * Math.PI * freq * i) / sampleRate));
  }
  return out;
}

/** RMS của phần giữa (bỏ 10% đầu để tránh quá độ). */
function rmsMid(arr) {
  const a = Math.floor(arr.length * 0.1);
  const b = Math.floor(arr.length * 0.95);
  if (b <= a) return 0;
  let acc = 0;
  for (let i = a; i < b; i++) acc += arr[i] * arr[i];
  return Math.sqrt(acc / (b - a));
}

console.log(`Worklet: ${path.relative(process.cwd(), WORKLET)}`);
console.log("");

// ───────────────────────────────── 1. Sine CAO TẦN: phải bị suy giảm (không gập phổ full)
const SEC = 1.0;
const AMP = 0.8;

for (const [freq, label, maxRatio] of [
  [9000, "9 kHz (gập thành 7 kHz)", 0.75],
  [12000, "12 kHz (gập thành 4 kHz)", 0.5],
]) {
  const raw = rmsMid(decimateRaw(48000, freq, AMP, SEC, 3));
  const got = rmsMid(runTone(48000, freq, AMP, SEC));
  const rel = got / raw;
  console.log(`  ${label}: RMS decimation trần = ${raw.toFixed(4)} | có boxcar = ${got.toFixed(4)}` +
    `  => còn ${(rel * 100).toFixed(1)}% (${(20 * Math.log10(rel)).toFixed(1)} dB)`);
  check(
    rel < maxRatio,
    `E5: ${label} bị suy giảm đủ (còn ${(rel * 100).toFixed(1)}% < ${(maxRatio * 100).toFixed(0)}%)`,
    `tỉ lệ ${rel.toFixed(3)}`
  );
}

// ───────────────────────────────── 2. Dải thoại: KHÔNG được làm hỏng
const speechRaw = rmsMid(decimateRaw(48000, 440, AMP, SEC, 3));
const speechGot = rmsMid(runTone(48000, 440, AMP, SEC));
const speechRel = speechGot / speechRaw;
console.log("");
console.log(`  440 Hz (dải thoại): RMS = ${speechGot.toFixed(4)} (tham chiếu ${speechRaw.toFixed(4)})` +
  ` => giữ ${(speechRel * 100).toFixed(1)}%`);
check(speechRel > 0.97 && speechRel < 1.03, "E5: sine 440 Hz giữ nguyên biên độ (±3%)");

// ───────────────────────────────── 3. Tỉ lệ KHÔNG nguyên: vẫn dùng nội suy tuyến tính
const oddScope = makeScope(44100);
const OddProc = oddScope.registered["audio-capture-processor"];
const oddProc = new OddProc({ processorOptions: { targetSampleRate: TARGET_RATE, chunkSize: CHUNK } });
check(oddProc._useBoxcar === false, "E5: 44100/16000 (không nguyên) KHÔNG dùng boxcar");
check(oddProc.ratio !== Math.floor(oddProc.ratio), "E5: ratio 44100/16000 đúng là không nguyên");

// ───────────────────────────────── 4. Đường 48 kHz BẬT boxcar, đường 16 kHz tắt
const evenScope = makeScope(48000);
const EvenProc = evenScope.registered["audio-capture-processor"];
const evenProc = new EvenProc({ processorOptions: { targetSampleRate: TARGET_RATE, chunkSize: CHUNK } });
check(evenProc._useBoxcar === true, "E5: 48000/16000 = 3 (nguyên) BẬT boxcar");

const unityScope = makeScope(16000);
const UnityProc = unityScope.registered["audio-capture-processor"];
const unityProc = new UnityProc({ processorOptions: { targetSampleRate: TARGET_RATE, chunkSize: CHUNK } });
check(unityProc.ratio === 1 && unityProc._useBoxcar === false, "E5: ratio = 1 đi thẳng, không lọc");

// ───────────────────────────────── 5. Số mẫu ra không đổi so với trước (không trôi timestamp)
const outLen = runTone(48000, 440, AMP, SEC).length;
const expect = Math.floor(48000 * SEC * TARGET_RATE / 48000);
check(
  Math.abs(outLen - expect) <= CHUNK,
  `E5: số mẫu ra vẫn khớp tỉ lệ (${outLen} vs ~${expect}) — boxcar không làm trôi timestamp`
);

console.log("");
if (failures.length === 0) {
  console.log("KẾT QUẢ: PASS (resampler chống aliasing đúng, dải thoại không bị hại)");
  process.exit(0);
}
console.log(`KẾT QUẢ: FAIL (${failures.length} mục)`);
failures.forEach((f) => console.log("  - " + f));
process.exit(1);
