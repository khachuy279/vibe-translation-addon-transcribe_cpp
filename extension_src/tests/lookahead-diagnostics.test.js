// Test cho `lib/lookahead-diagnostics.js` — bộ phân loại "vì sao Pipeline B không chạy".
//
//   node extension_firefox/tests/lookahead-diagnostics.test.js
//
// Bối cảnh (đo thật 2026-10-05): POPUP chỉ có hai câu trạng thái thô nên không thể biết nguyên
// nhân trên các trang ngoài YouTube/Bilibili. Bộ phân loại này là thứ biến dữ kiện thô (nguồn
// phát, mime SourceBuffer, DRM, lệch thẻ video…) thành MỘT mã lý do đọc được.

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  LA_REASON,
  explainLookaheadReason,
  classifyLookaheadAvailability,
  buildDiagnosticRows,
  describeSrcKind,
} = require("../lib/lookahead-diagnostics.js");

//: Dữ kiện "lành mạnh": MSE audio-only, đã bắt được mảnh có mốc media.
function okFacts(overrides = {}) {
  return Object.assign({
    hasVideo: true,
    durationFinite: true,
    duration: 600,
    hasMseApi: true,
    mseUsed: true,
    acceptMuxedAudio: true,
    audioSbCount: 1,
    videoSbCount: 0,
    muxedSbCount: 0,
    unknownSbCount: 0,
    mimeTypes: ['audio/mp4; codecs="mp4a.40.2"'],
    container: "fmp4/init",
    hasInitSegment: true,
    cachedChunksCount: 12,
    mediaTaggedChunks: 12,
    appendCount: 12,
    appendsAudioBearing: 12,
    appendsSkippedVideo: 0,
    appendsNoRawBytes: 0,
    appendErrorCount: 0,
    keySystems: [],
    srcKind: "blob:",
    bufferedAheadVideo: 40,
    mediaRangeCount: 4,
  }, overrides);
}

test("video lành mạnh (MSE audio-only) ⇒ OK", () => {
  const v = classifyLookaheadAvailability(okFacts());
  assert.equal(v.code, "OK");
});

test("không có thẻ video ⇒ NO_VIDEO", () => {
  assert.equal(classifyLookaheadAvailability({ hasVideo: false }).code, "NO_VIDEO");
});

test("livestream (duration vô hạn) ⇒ LIVE_STREAM, không phải lỗi MSE", () => {
  const v = classifyLookaheadAvailability(okFacts({ durationFinite: false, duration: "Infinity" }));
  assert.equal(v.code, "LIVE_STREAM");
});

test("DRM/EME được ưu tiên hơn mọi dữ kiện MSE khác", () => {
  const v = classifyLookaheadAvailability(okFacts({ keySystems: ["com.widevine.alpha"] }));
  assert.equal(v.code, "DRM");
  assert.match(v.detail, /widevine/);
});

test("trang phát trực tiếp (không gọi addSourceBuffer) ⇒ MSE_UNUSED kèm srcKind", () => {
  const v = classifyLookaheadAvailability(okFacts({
    mseUsed: false, audioSbCount: 0, muxedSbCount: 0, cachedChunksCount: 0,
    hasInitSegment: false, srcKind: "http-media",
  }));
  assert.equal(v.code, "MSE_UNUSED");
  assert.match(v.detail, /http-media/);
});

// HỒI QUY 2026-10-05 (xhamster.com, av01.media): cờ nhận-MUXED đã BẬT nên SourceBuffer
// video+audio PHẢI được coi là nguồn audio hợp lệ — bản trước trả `MUXED_ONLY` (nghe như
// "không khả dụng") trong khi interceptor đã bắt được mảnh và cổng Pipeline B vẫn ĐẠT.
test("SourceBuffer MUXED + đã bắt được mảnh ⇒ OK (KHÔNG phải MUXED_ONLY)", () => {
  const v = classifyLookaheadAvailability(okFacts({
    audioSbCount: 0, muxedSbCount: 1,
    mimeTypes: ['video/mp4;codecs=mp4a.40.2,av01.0.05M.08.0.111.01.01.01.0'],
    cachedChunksCount: 3, mediaTaggedChunks: 3, appendCount: 4, appendsAudioBearing: 3,
  }));
  assert.equal(v.code, "OK");
  assert.match(v.detail, /MUXED/);
});

test("SourceBuffer MUXED nhưng CHƯA có mảnh audio ⇒ NO_APPEND_YET (không phải MUXED_ONLY)", () => {
  const v = classifyLookaheadAvailability(okFacts({
    audioSbCount: 0, muxedSbCount: 1,
    mimeTypes: ['video/mp4;codecs=mp4a.40.2,avc1.64001E'],
    cachedChunksCount: 0, mediaTaggedChunks: 0, hasInitSegment: false,
    appendCount: 2, appendsAudioBearing: 0, appendsSkippedVideo: 2,
  }));
  assert.equal(v.code, "NO_APPEND_YET");
  assert.match(v.detail, /0 trong 2 lần append|0 trong 2/);
});

test("chỉ ra MUXED_ONLY khi cờ nhận-MUXED bị TẮT", () => {
  const v = classifyLookaheadAvailability(okFacts({
    acceptMuxedAudio: false, audioSbCount: 0, muxedSbCount: 1,
    mimeTypes: ['video/mp4; codecs="avc1.64001f,mp4a.40.2"'],
  }));
  assert.equal(v.code, "MUXED_ONLY");
  assert.match(v.detail, /acceptMuxedAudio=false/);
});

test("chỉ có SourceBuffer VIDEO ⇒ VIDEO_ONLY_SB", () => {
  const v = classifyLookaheadAvailability(okFacts({
    audioSbCount: 0, muxedSbCount: 0, videoSbCount: 1, cachedChunksCount: 0,
    hasInitSegment: false, mimeTypes: ["video/mp4"],
  }));
  assert.equal(v.code, "VIDEO_ONLY_SB");
});

test("SourceBuffer không rõ mime (tạo trước hook) ⇒ UNKNOWN_MIME_SB", () => {
  const v = classifyLookaheadAvailability(okFacts({
    audioSbCount: 0, muxedSbCount: 0, videoSbCount: 0, unknownSbCount: 1,
    cachedChunksCount: 0, hasInitSegment: false, container: "fmp4/media",
  }));
  assert.equal(v.code, "UNKNOWN_MIME_SB");
});

test("LỆCH THẺ VIDEO được báo riêng (nguyên nhân iq.com: đệm 60s mà popup báo +5.95s)", () => {
  const v = classifyLookaheadAvailability(okFacts({
    videoMismatch: { activeSrc: "blob:https://iq.com/aaa", activeTime: 12.5,
      requestedSrc: "blob:https://iq.com/bbb", requestedTime: 61.0 },
  }));
  assert.equal(v.code, "VIDEO_MISMATCH");
  assert.match(v.detail, /iq\.com/);
});

test("đã dùng MSE nhưng chưa append gì ⇒ NO_APPEND_YET", () => {
  const v = classifyLookaheadAvailability(okFacts({
    cachedChunksCount: 0, hasInitSegment: false, appendCount: 0, mediaTaggedChunks: 0,
    appendsAudioBearing: 0,
  }));
  assert.equal(v.code, "NO_APPEND_YET");
});

// HỒI QUY 2026-10-05 (xvideos.com): log thật báo "14 append nhưng cache rỗng" và kết luận
// `NO_AUDIO_BYTES` — nhưng 14 lần đó là append của BUFFER VIDEO. Phải đếm riêng append audio.
test("append chỉ nằm trên buffer VIDEO ⇒ NO_APPEND_YET, không phải NO_AUDIO_BYTES", () => {
  const v = classifyLookaheadAvailability(okFacts({
    audioSbCount: 1, videoSbCount: 1, muxedSbCount: 0,
    mimeTypes: ["audio/mp4;codecs=mp4a.40.2", "video/mp4;codecs=avc1.4d401e"],
    cachedChunksCount: 0, mediaTaggedChunks: 0, hasInitSegment: false,
    appendCount: 14, appendsAudioBearing: 0, appendsSkippedVideo: 14, msSinceCacheReset: null,
  }));
  assert.equal(v.code, "NO_APPEND_YET");
  assert.match(v.detail, /14/);
});

test("cache vừa bị xoá vì trang đổi nguồn video ⇒ SOURCE_CHANGED_RESET", () => {
  const v = classifyLookaheadAvailability(okFacts({
    cachedChunksCount: 0, mediaTaggedChunks: 0, hasInitSegment: false,
    appendCount: 14, appendsAudioBearing: 14,
    msSinceCacheReset: 3000, lastCacheResetReason: "video src change", cacheResetCount: 2,
  }));
  assert.equal(v.code, "SOURCE_CHANGED_RESET");
  assert.match(v.detail, /video src change/);
});

test("có append audio nhưng cache rỗng và KHÔNG do đổi nguồn ⇒ NO_AUDIO_BYTES kèm số liệu", () => {
  const v = classifyLookaheadAvailability(okFacts({
    cachedChunksCount: 0, mediaTaggedChunks: 0, hasInitSegment: false,
    appendCount: 9, appendsAudioBearing: 9, appendsNoRawBytes: 9,
    msSinceCacheReset: null, appendErrorCount: 0,
  }));
  assert.equal(v.code, "NO_AUDIO_BYTES");
  assert.match(v.detail, /không đọc được byte: 9/);
});

test("chỉ có init segment ⇒ INIT_ONLY (trình phát chưa append mảnh media)", () => {
  const v = classifyLookaheadAvailability(okFacts({
    cachedChunksCount: 0, hasInitSegment: true, appendCount: 1, mediaTaggedChunks: 0,
    appendsAudioBearing: 1,
  }));
  assert.equal(v.code, "INIT_ONLY");
});

test("mảnh trong cache nhưng không mảnh nào có mốc media ⇒ NO_MEDIA_TIMELINE", () => {
  const v = classifyLookaheadAvailability(okFacts({ mediaTaggedChunks: 0 }));
  assert.equal(v.code, "NO_MEDIA_TIMELINE");
});

test("đệm video quá ngắn cũng bị coi là chưa khả dụng", () => {
  const v = classifyLookaheadAvailability(okFacts({ bufferedAheadVideo: 0.4 }));
  assert.equal(v.code, "BUFFERED_LOW");
});

test("mọi mã lý do đều có giải thích + cách xử lý bằng tiếng Việt", () => {
  for (const [code, info] of Object.entries(LA_REASON)) {
    assert.ok(info.vi && info.vi.length > 3, `mã ${code} thiếu mô tả ngắn`);
    assert.ok(info.why && info.why.length > 10, `mã ${code} thiếu giải thích`);
    assert.ok(info.fix && info.fix.length > 3, `mã ${code} thiếu cách xử lý`);
  }
  // Mã lạ vẫn phải trả về câu an toàn (không ném lỗi, không trả undefined).
  assert.equal(explainLookaheadReason("KHONG_CO").code, "KHONG_CO");
  assert.equal(explainLookaheadReason(null).code, "UNKNOWN");
});

test("báo cáo chẩn đoán có đủ dòng khoá để đọc bằng mắt", () => {
  const facts = okFacts();
  const rows = buildDiagnosticRows(facts, null);
  const labels = rows.map((r) => r[0]);
  for (const need of ["Kết luận", "Nguồn phát (srcKind)", "SourceBuffer", "Container sniff",
    "Mảnh trong cache", "DRM (EME)", "Vì sao", "Cách xử lý"]) {
    assert.ok(labels.includes(need), `thiếu dòng "${need}"`);
  }
  assert.match(describeSrcKind("http-hls"), /HLS/);
});
