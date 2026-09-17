// Regression test cho auto-ducking cua extension (extension_firefox/lib/tts-player.js).
//
// BOI CANH: nguoi dung bao "keo 🔉 Original audio xuong 5% nhung tieng goc van con lon".
// Khi kiem chung bang cach CHAY THAT lop TTSAudioPlayer voi mock video giong trinh duyet,
// phat hien 3 loi that:
//
//   C. Extension chi ghi `video.volume` MOT LAN moi lan doi setting. Neu trang (hoac chinh
//      nguoi dung keo thanh am luong cua trang) ghi `volume` sau do, ducking bi huy AM THAM
//      — ma `isDucked` van la `true` nen ca duong restore cung sai. => "dat 5% van lon".
//   D. `duckingLevel` ngoai [0,1] (vi du ai do truyen 5 thay vi 0.05) bi clamp thanh 1.0
//      = AM LUONG DAY, tuc loi sai don vi bien thanh "mo to het co".
//   E. `volume > 0 ? volume : 1.0` quy video dang TAT TIENG thanh 1.0, nen khi restore,
//      video muted bi BAT LEN 100%.
//
// Cach chay:  node backend/tests/js/tts_ducking_test.js
// (test_31_extension_ducking.py goi file nay va bo qua neu may khong co Node)

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const SRC = path.join(__dirname, "..", "..", "..", "extension_firefox", "lib", "tts-player.js");
const src = fs.readFileSync(SRC, "utf8");
// File la IIFE gan vao globalThis.TTSAudioPlayer.
eval(src);
const TTSAudioPlayer = globalThis.TTSAudioPlayer;
assert.ok(TTSAudioPlayer, "khong nap duoc TTSAudioPlayer tu " + SRC);

/**
 * Mock video GIONG TRINH DUYET THAT:
 *  - co addEventListener / removeEventListener
 *  - ghi `volume` hoac `muted` PHAI phat sinh `volumechange` (ke ca khi chinh extension ghi,
 *    vi trinh duyet lam dung nhu vay — do la ly do phai co co `_writingVolume`).
 */
function mockVideo(volume = 1.0, muted = false) {
  const listeners = new Map();
  let _v = volume;
  let _m = muted;
  const v = {
    isConnected: true,
    ended: false,
    addEventListener(t, fn) {
      if (!listeners.has(t)) listeners.set(t, new Set());
      listeners.get(t).add(fn);
    },
    removeEventListener(t, fn) {
      const s = listeners.get(t);
      if (s) s.delete(fn);
    },
    _fire(t) {
      for (const fn of [...(listeners.get(t) || [])]) fn();
    },
    _count(t) {
      return (listeners.get(t) || new Set()).size;
    },
  };
  Object.defineProperty(v, "volume", {
    get: () => _v,
    set: (x) => {
      _v = x;
      v._fire("volumechange");
    },
  });
  Object.defineProperty(v, "muted", {
    get: () => _m,
    set: (x) => {
      _m = x;
      v._fire("volumechange");
    },
  });
  return v;
}

let n = 0;
function section(title) {
  console.log("\n" + "=".repeat(90) + "\n" + title + "\n" + "=".repeat(90));
}
function ok(msg) {
  n++;
  console.log("  PASS  " + msg);
}

// ─────────────────────────────────────────── A. duong binh thuong
section("A. Duong binh thuong: dat 5%");
{
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setTargetVideo(v, true, 0.05, true);
  assert.strictEqual(v.volume, 0.05, "duck 5% phai cho volume=0.05");
  p.applySettings(true, 0.05, true);
  assert.strictEqual(v.volume, 0.05, "applySettings lai khong duoc doi volume");
  ok("dat 5% tu luc chua duck -> 0.05");
}

// ─────────────────────────────────────────── B. keo slider khi dang chay
section("B. Keo slider 25% -> 5% khi dang chay");
{
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setTargetVideo(v, true, 0.25, true);
  assert.strictEqual(v.volume, 0.25);
  p.applySettings(true, 0.05, true);
  assert.strictEqual(v.volume, 0.05, "keo 25%->5% phai ap ngay");
  ok("25% -> 5%");
}

// ─────────────────────────────────────── C. TRANG GHI DE volume (loi chinh)
section("C. [LOI CU] Trang ghi de volume khi extension da duck 5%");
{
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setTargetVideo(v, true, 0.05, true);
  assert.strictEqual(v.volume, 0.05);

  v.volume = 1.0; // trang ghi de -> phat volumechange
  assert.strictEqual(v.volume, 0.05, "guard phai tu keo ve 0.05 ngay sau khi trang ghi 1.0");

  v.volume = 0.8; // trang ghi de lan nua
  assert.strictEqual(v.volume, 0.05, "guard phai keo ve 0.05 voi moi gia tri trang ghi");

  assert.strictEqual(v._count("volumechange"), 1, "guard phai duoc gan dung 1 lan");
  ok("trang ghi 1.0 roi 0.8 -> van 0.05 (ducking 'dinh')");
}

// ─────────────────────────────────────────── C2. tat ducking -> go guard
section("C2. Tat ducking -> go guard va tra volume goc");
{
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setTargetVideo(v, true, 0.05, true);
  v.volume = 1.0;
  assert.strictEqual(v.volume, 0.05);

  p.applySettings(true, 0.05, false); // tat TTS
  assert.strictEqual(v.volume, 1.0, "tat ducking phai tra ve volume goc");
  assert.strictEqual(v._count("volumechange"), 0, "phai GO listener, khong duoc ro ri");

  v.volume = 0.3; // sau khi tat, trang tu do ghi
  assert.strictEqual(v.volume, 0.3, "khong duoc can thiep nua khi da tat ducking");
  ok("tat ducking -> go guard, khong ro ri listener");
}

// ─────────────────────────────────────────── D. sai don vi
section("D. [LOI CU] duckingLevel sai don vi (5 thay vi 0.05)");
{
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setTargetVideo(v, true, 5, true);
  assert.strictEqual(p.duckingLevel, 1.0, "phai chuan hoa 5 -> 1.0 (co canh bao)");
  // clamp(1.0 * 1.0) = 1.0 — dung theo nghia "100%", KHONG con la ket qua cua 1.0*5
  assert.strictEqual(v.volume, 1.0);

  p.applySettings(true, -3, true);
  assert.strictEqual(p.duckingLevel, 0, "phai chuan hoa -3 -> 0 (tat tieng goc)");
  assert.strictEqual(v.volume, 0, "duckingLevel=0 phai tat han tieng goc");
  ok("5 -> 1.0 va -3 -> 0 (khong con am tham thanh 'mo het co')");
}

// ─────────────────────────────────────────── E. video muted (loi cu)
section("E. [LOI CU] Video dang MUTED (volume=0)");
{
  const p = new TTSAudioPlayer();
  const v = mockVideo(0.0, true);
  p.setTargetVideo(v, true, 0.05, true);
  assert.strictEqual(v.volume, 0.0, "muted thi volume giu 0");
  assert.strictEqual(v.muted, true, "khong duoc tu bo muted");

  p.applySettings(true, 0.05, false); // restore
  assert.strictEqual(v.volume, 0.0, "restore KHONG duoc bat len 1.0 (loi cu)");
  assert.strictEqual(v.muted, true, "restore phai giu muted");
  ok("video muted: restore khong bat tieng 100%");
}

section("E2. Video muted nhung volume=1.0 (kieu bam nut mute cua trang)");
{
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0, true);
  p.setTargetVideo(v, true, 0.05, true);
  assert.strictEqual(v.volume, 0.05);
  p.applySettings(true, 0.05, false);
  assert.strictEqual(v.volume, 1.0, "tra dung volume goc 1.0");
  assert.strictEqual(v.muted, true, "van phai muted");
  ok("muted + volume=1.0: restore dung trang thai goc");
}

// ─────────────────────────────────────────── F. reapplyDucking
section("F. reapplyDucking() — duong an toan tu su kien play/playing");
{
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setTargetVideo(v, true, 0.05, true);
  p._detachVolumeGuard(); // gia lap guard bi mat (vd trang thay element)
  v.volume = 1.0;
  p.reapplyDucking();
  assert.strictEqual(v.volume, 0.05, "reapplyDucking phai keo ve muc duck");
  ok("reapplyDucking keo ve 5%");
}

// ─────────────────────────────────────────── G. doi video
section("G. Doi sang video khac -> go guard khoi video CU");
{
  const p = new TTSAudioPlayer();
  const v1 = mockVideo(1.0);
  const v2 = mockVideo(1.0);
  p.setTargetVideo(v1, true, 0.05, true);
  assert.strictEqual(v1.volume, 0.05);
  p.setTargetVideo(v2, true, 0.05, true);
  assert.strictEqual(v1.volume, 1.0, "video cu phai duoc restore");
  assert.strictEqual(v1._count("volumechange"), 0, "phai go guard khoi video cu");
  assert.strictEqual(v2.volume, 0.05, "video moi phai duoc duck");
  assert.strictEqual(v2._count("volumechange"), 1, "guard phai gan tren video moi");
  ok("doi video: go guard cu, gan guard moi");
}

// ─────────────────────────────────────────── H. destroy
section("H. destroy() -> tra volume va go guard");
{
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setTargetVideo(v, true, 0.05, true);
  p.destroy();
  assert.strictEqual(v.volume, 1.0, "destroy phai tra volume goc");
  assert.strictEqual(v._count("volumechange"), 0, "destroy phai go guard");
  ok("destroy don sach");
}

// ────────────────────────── I. DUCK BẰNG GAIN (kiến trúc đúng: không đụng video.volume)
section("I. [KIEN TRUC] Duck bang GainNode -> KHONG duoc dung video.volume");
{
  // Sink gia lap AudioCapture: ghi lai moi muc duoc yeu cau.
  const calls = [];
  const sink = {
    setDuckLevel(level) {
      calls.push(level);
      return true;
    },
  };

  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setDuckSink(sink);
  p.setTargetVideo(v, true, 0.05, true);

  assert.strictEqual(v.volume, 1.0, "duck bang gain thi video.volume PHAI giu nguyen 1.0");
  assert.strictEqual(v._count("volumechange"), 0, "khong duoc gan guard khi duck bang gain");
  assert.deepStrictEqual(calls, [0.05], "phai goi setDuckLevel(0.05)");
  ok("duck bang gain: video.volume khong bi dung (ASR an toan)");

  // Keo slider ve 0% -> KHONG duoc dong toi video.volume, chi gain ve 0
  p.applySettings(true, 0, true);
  assert.strictEqual(v.volume, 1.0, "0% KHONG duoc dung video.volume (ASR van con audio)");
  assert.strictEqual(calls[calls.length - 1], 0, "gain phai ve 0");
  ok("0%: chi tat nhanh NGHE, video.volume van 1.0 -> ASR van chay");

  // 100%
  p.applySettings(true, 1.0, true);
  assert.strictEqual(v.volume, 1.0);
  assert.strictEqual(calls[calls.length - 1], 1.0, "100% -> gain 1.0 (tieng goc day du)");
  ok("100%: gain = 1.0");

  // Tat TTS -> gain ve 1.0 (tra lai tieng goc)
  p.applySettings(true, 0, false);
  assert.strictEqual(calls[calls.length - 1], 1.0, "tat ducking phai tra gain ve 1.0");
  ok("tat TTS: gain tra ve 1.0");
}

section("I2. Mat gain giua chung -> tu roi ve duong video.volume");
{
  const sink = { setDuckLevel: () => true };
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setDuckSink(sink);
  p.setTargetVideo(v, true, 0.05, true);
  assert.strictEqual(v.volume, 1.0);

  // Capture dung lai (vd video bi thay) -> sink bao khong xu ly duoc nua
  sink.setDuckLevel = () => false;
  p._restoreVideoVolume();
  p.setTargetVideo(v, true, 0.05, true);
  assert.strictEqual(v.volume, 0.05, "khong co gain thi phai duck bang video.volume");
  ok("sink chet -> tu chuyen sang duck bang video.volume");
}

section("I3. Bo duckSink giua chung -> tra gain ve 1.0");
{
  const calls = [];
  const sink = { setDuckLevel: (l) => { calls.push(l); return true; } };
  const p = new TTSAudioPlayer();
  const v = mockVideo(1.0);
  p.setDuckSink(sink);
  p.setTargetVideo(v, true, 0.05, true);
  assert.strictEqual(v.volume, 1.0);

  p.setDuckSink(null); // vd cleanup
  assert.strictEqual(calls[calls.length - 1], 1.0, "phai tra gain ve 1.0 khi bo sink");
  ok("bo sink -> tra gain ve 1.0");
}

console.log("\n" + "=".repeat(90));
console.log(`KET QUA: PASS (${n} nhom kiem tra)`);
console.log("=".repeat(90));
