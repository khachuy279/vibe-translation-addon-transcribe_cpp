/**
 * lookahead-diagnostics.js — Từ điển MÃ LÝ DO + bộ phân loại "vì sao Pipeline B không chạy".
 *
 * VẤN ĐỀ GỐC (đo thật 2026-10-05):
 *   POPUP chỉ hiện hai trạng thái thô: "Lookahead available +Xs" và "Lookahead: không khả dụng
 *   (chạy Realtime)". Hai câu đó KHÔNG nói được vì sao. Trên các trang ngoài YouTube/Bilibili
 *   (xvideos.com, iq.com, xhamster.com, av01.media…) người dùng chỉ biết "không chạy được" mà
 *   không biết nút cổ chai nằm ở: trang không dùng MSE, audio nằm trong SourceBuffer MUXED,
 *   interceptor theo dõi NHẦM thẻ <video>, DRM, hay backend không tiến triển.
 *
 * Module này là NGUỒN SỰ THẬT DUY NHẤT cho:
 *   1. Danh sách mã lý do (`LA_REASON`) + lời giải thích tiếng Việt + gợi ý xử lý.
 *   2. `classifyLookaheadAvailability(facts)`: biến các DỮ KIỆN THÔ (đếm SourceBuffer, mime,
 *      loại nguồn phát, DRM, mảnh đã bắt…) thành MỘT mã lý do duy nhất + chi tiết.
 *   3. `LookaheadDiag`: logger có phân loại, chống trùng lặp, giữ thống kê để in báo cáo.
 *
 * Cách dùng (content script / popup):
 *   const d = new LookaheadDiag("BS");
 *   d.log("gate", `Cổng Pipeline B: ${code}`, { ... });
 *   d.warn("gate", "…");
 *   d.group("BÁO CÁO CHẨN ĐOÁN", [["dòng", "giá trị"], ...]);
 */

(function (globalScope) {
  "use strict";

  // ── MÃ LÝ DO ────────────────────────────────────────────────────────────────
  // `vi`   : câu ngắn để hiển thị trên POPUP.
  // `why`  : giải thích đầy đủ để đọc trong Console.
  // `fix`  : việc cần làm / giới hạn kỹ thuật đã biết.
  const LA_REASON = {
    OK: {
      vi: "MSE audio-only đã bắt được",
      why: "Trang dùng MediaSource với một SourceBuffer chỉ chứa audio; interceptor đã bắt được "
        + "init segment + mảnh media và gắn được mốc thời gian thật từ SourceBuffer.buffered.",
      fix: "Pipeline B có đủ điều kiện chạy.",
    },
    NO_VIDEO: {
      vi: "chưa tìm thấy thẻ <video>",
      why: "Không có thẻ <video> nào trong DOM (kể cả Shadow DOM / iframe cùng nguồn) tại thời "
        + "điểm kiểm tra — thường là video chưa được tạo (phải bấm Play) hoặc nằm trong iframe "
        + "khác nguồn (cross-origin) mà content script không đọc được.",
      fix: "Bấm Play cho video rồi mở lại popup; nếu video nằm trong iframe cross-origin thì "
        + "extension phải chạy trong chính frame đó (đã bật all_frames).",
    },
    LIVE_STREAM: {
      vi: "livestream (không có độ dài xác định)",
      why: "`video.duration` là Infinity/NaN ⇒ không có 'tương lai' để tải trước.",
      fix: "Pipeline B chỉ dành cho VOD. Livestream phải dùng Pipeline A (Realtime).",
    },
    DRM: {
      vi: "video có DRM (EME)",
      why: "Trang gọi `navigator.requestMediaKeySystemAccess` (Widevine/PlayReady/FairPlay) ⇒ "
        + "mảnh audio trong SourceBuffer là dữ liệu ĐÃ MÃ HOÁ, bộ giải mã phía backend không thể "
        + "đọc ra PCM.",
      fix: "Không thể chạy Pipeline B (và cả Pipeline A cũng chỉ lấy được tiếng đã giải mã nếu "
        + "trình phát cho phép). Cần nguồn không DRM.",
    },
    NO_MSE_API: {
      vi: "trình duyệt không có MediaSource",
      why: "`window.MediaSource` không tồn tại trong frame này.",
      fix: "Không thể hook MSE ⇒ không lấy được byte audio. Kiểm tra môi trường trình duyệt/frame.",
    },
    MSE_UNUSED: {
      vi: "trang không dùng MSE",
      why: "`MediaSource` có tồn tại nhưng KHÔNG có lời gọi `addSourceBuffer` nào: trang phát "
        + "video bằng nguồn trực tiếp (`<video src=\"…mp4\">`), HLS native (.m3u8), DASH native "
        + "hoặc `srcObject` (MSE/WebRTC tạo ở nơi khác). Interceptor không có mảnh nào để bắt.",
      fix: "Cần chế độ mới: tải khoảng byte bằng URL (đã ghi lại Range qua hook fetch/XHR) thay "
        + "vì chờ appendBuffer. Xem `diag.mediaRanges` — nếu > 0 là có thể làm được.",
    },
    VIDEO_MISMATCH: {
      vi: "interceptor đang theo dõi NHẦM thẻ video",
      why: "Content script chọn một thẻ <video> (thẻ lớn nhất/đang phát) nhưng interceptor chọn "
        + "thẻ khác. Trang nhiều video (quảng cáo, preview, player phụ) ⇒ mảnh audio gửi lên "
        + "backend thuộc video KHÁC ⇒ 'Đã dịch 0.0s' dù cache đầy. Đây là nguyên nhân đã đo được "
        + "trên iq.com: vùng đệm của video chính ~60s nhưng `aheadSeconds` báo chỉ ~5.95s.",
      fix: "Interceptor nay nhận `preferredSrc` từ content script và ưu tiên đúng thẻ đó; nếu vẫn "
        + "lệch, cần kiểm tra `diag.videoMismatch` để biết hai thẻ đang trỏ vào đâu.",
    },
    MUXED_ONLY: {
      vi: "audio chỉ nằm trong SourceBuffer MUXED (và tính năng nhận MUXED đang TẮT)",
      why: "Trang chỉ tạo MỘT SourceBuffer chứa cả hình lẫn tiếng (mime kiểu "
        + "`video/mp4; codecs=\"mp4a.40.2,av01…\"`, hoặc `avc1…,mp4a…`). Nếu cờ nhận-MUXED BẬT "
        + "(mặc định từ 0.6.7) thì mảnh của buffer này VẪN được dùng cho Pipeline B và mã này "
        + "KHÔNG xuất hiện. Mã này chỉ còn nghĩa: cờ đó đang TẮT.",
      fix: "Bật lại nhận-MUXED: `ACCEPT_MUXED_AUDIO` trong `content/buffer_interceptor_poc.js` "
        + "phải là true (hoặc đặt `window.__VIBE_LOOKAHEAD_ACCEPT_MUXED__ = true` trước khi trang "
        + "tạo SourceBuffer). Kiểm tra payload: `diag.acceptMuxedAudio` phải là true.",
    },
    SOURCE_CHANGED_RESET: {
      vi: "trang vừa ĐỔI NGUỒN VIDEO ⇒ cache audio vừa bị xoá",
      why: "Interceptor xoá toàn bộ cache (mảnh + init segment) mỗi khi `location.href` hoặc "
        + "`video.currentSrc` đổi — đúng theo thiết kế, để không gửi audio của video CŨ lên backend. "
        + "Nhưng nhiều trang (preview/quảng cáo → video chính, hoặc player tự đổi nguồn) đổi nguồn "
        + "liên tục, nên ngay sau khi bấm Start cache vẫn rỗng dù đã có hàng chục lần append.",
      fix: "Chờ video CHÍNH ổn định (không còn đổi `currentSrc`) rồi mới bấm Start. Xem "
        + "`msSinceCacheReset`/`lastCacheResetReason` trong báo cáo để biết vừa bị xoá vì gì.",
    },
    UNKNOWN_MIME_SB: {
      vi: "SourceBuffer tạo TRƯỚC khi hook kịp chạy",
      why: "Có `appendBuffer` trên một SourceBuffer mà interceptor không biết mime (thường do "
        + "SourceBuffer được tạo trước khi script được tiêm, hoặc do player gọi qua `ManagedMedia"
        + "Source`). Byte nhận được có thể là VIDEO, không phải audio.",
      fix: "Interceptor nay vẫn ghi nhận và SNIFF container/codec của byte đầu tiên để phân loại. "
        + "Xem log `[Lookahead][Diag] sniff` để biết container thật.",
    },
    VIDEO_ONLY_SB: {
      vi: "chỉ có SourceBuffer VIDEO",
      why: "Trang tạo SourceBuffer chỉ chứa hình (mime `video/mp4` / `video/webm`). Audio đi bằng "
        + "đường khác: WebAudio tự phát, EME, hoặc player tách track nhưng không qua MSE trong "
        + "frame này.",
      fix: "Cần xem `diag.srcKind` và các request Range đã ghi (`mediaRanges`) để tìm URL audio.",
    },
    NO_APPEND_YET: {
      vi: "trình phát chưa append AUDIO nào",
      why: "Đã thấy SourceBuffer chứa audio nhưng CHƯA có lần `appendBuffer` nào rơi vào buffer "
        + "đó (mọi lần append đều thuộc buffer VIDEO). Trình phát đang dừng, chưa nạp audio, hoặc "
        + "vùng đệm đã đầy sẵn từ trước khi extension chạy.",
      fix: "Bấm Play / tua nhẹ để trình phát phải nạp mảnh audio mới. Xem nhật ký "
        + "`diag.recentAppends` để biết từng lần append đi vào buffer nào.",
    },
    NO_AUDIO_BYTES: {
      vi: "có append AUDIO nhưng cache rỗng",
      why: "Đã có lần append rơi vào buffer audio nhưng cache vẫn rỗng. Ba nguyên nhân, đều được "
        + "đếm sẵn trong payload: (a) `appendsNoRawBytes` — không đọc được byte từ tham số "
        + "`appendBuffer`; (b) `appendsPrunedOut` — mảnh bị `pruneCacheAroundPlayhead` loại NGAY "
        + "vì nằm ngoài cửa sổ quanh playhead; (c) `appendErrorCount` — SourceBuffer báo lỗi.",
      fix: "Chạy `__VIBE_LOOKAHEAD_DEBUG__.diagnose()` và xem bảng `recentAppends` (cột `note`) "
        + "để biết mảnh nào bị loại và vì sao.",
    },
    INIT_ONLY: {
      vi: "chỉ bắt được init segment, chưa có mảnh media",
      why: "Đã có header (ftyp/moov hoặc EBML) nhưng chưa có mảnh media nào ⇒ chưa có audio để "
        + "giải mã. Vùng đệm của trình phát có thể đã nạp từ trước khi extension chạy.",
      fix: "Tua tới một vị trí mới để trình phát buộc phải append mảnh mới.",
    },
    NO_MEDIA_TIMELINE: {
      vi: "mảnh không gắn được mốc thời gian thật",
      why: "Đã có mảnh trong cache nhưng không mảnh nào có `mediaStart/mediaEnd` (đọc từ "
        + "`SourceBuffer.buffered`). Khi đó việc chọn mảnh để replay là 'gửi bừa'.",
      fix: "Thường do `buffered` chưa cập nhật lúc `updateend`, hoặc append lỗi. Xem `errorCount`.",
    },
    BUFFERED_LOW: {
      vi: "vùng đệm trước quá ngắn",
      why: "`video.buffered` phía trước playhead < 1.5s ⇒ không có 'tương lai' đủ dài để dịch trước.",
      fix: "Đợi trình phát nạp thêm, hoặc hạ chất lượng để buffer nhanh hơn.",
    },
    // ── Lý do ở tầng PHIÊN (client gửi/nhận với backend) ──────────────────────
    STALL_NO_PROGRESS: {
      vi: "backend không tiến triển",
      why: "Trong lúc video bị tạm dừng để nạp đệm, `fragments`/`decoded_chunks`/`ready_until_pts` "
        + "từ backend KHÔNG tăng. Nguyên nhân thường gặp: (a) trình phát đang bị extension tạm "
        + "dừng nên KHÔNG append thêm mảnh mới ⇒ backend không có audio mới; (b) byte gửi lên "
        + "không giải mã được (PCM/byte thấp); (c) backend đang chạy ASR khối dài.",
      fix: "Xem dòng `[BS][Diag] STALL` + `lookahead_status` để biết mốc nào đứng yên. Bản này có "
        + "van an toàn `stall_guard` tự phá bế tắc sau vài giây.",
    },
    HORIZON_DEADLOCK: {
      vi: "bế tắc ràng buộc mốc đã xử lý",
      why: "Vòng lặp: playhead chạm `ready_until_pts` ⇒ tạm dừng chờ backend ⇒ backend cần thêm "
        + "audio ⇒ audio chỉ có khi video CHẠY ⇒ video bị tạm dừng. Playhead đứng im vĩnh viễn "
        + "(log backend: cùng một mốc Playhead lặp lại mỗi 10s, 'Đã dịch 0.0s').",
      fix: "Van `stall_guard` sẽ tự cho video chạy lại và treo tạm ràng buộc horizon để trình phát "
        + "nạp thêm dữ liệu.",
    },
    CONNECTION_TIMEOUT: {
      vi: "không kết nối được /ws/lookahead",
      why: "Sau 3.5s chưa mở được WebSocket tới backend Lookahead ⇒ tự chuyển Pipeline A.",
      fix: "Kiểm tra backend đang chạy và chứng chỉ `wss://localhost:8765` đã được chấp nhận.",
    },
    BACKEND_UNAVAILABLE: {
      vi: "backend từ chối chạy Pipeline B",
      why: "Backend gửi `lookahead_unavailable` (model ASR/aligner chưa sẵn sàng, thiếu VRAM…).",
      fix: "Đọc dòng `[Lookahead Client] ⚠️ Backend không chạy được Pipeline B` ngay trước đó.",
    },
    UNKNOWN: {
      vi: "không xác định",
      why: "Chưa đủ dữ kiện để kết luận (interceptor chưa phản hồi hoặc trang chưa nạp video).",
      fix: "Chạy `__VIBE_LOOKAHEAD_DEBUG__.printReport()` trong Console của trang.",
    },
  };

  /** Lời giải thích cho một mã lý do (an toàn với mã lạ). */
  function explainLookaheadReason(code) {
    if (!code) return { code: "UNKNOWN", ...LA_REASON.UNKNOWN };
    return LA_REASON[code] || { code, vi: String(code), why: String(code), fix: "" };
  }

  /** `srcKind` của interceptor ⇒ câu mô tả ngắn. */
  const SRC_KIND_TEXT = {
    "blob:": "MSE (blob: URL) — có thể hook được",
    "http-media": "URL file media trực tiếp (.mp4/.webm/.mkv) — KHÔNG qua MSE",
    "http-hls": "HLS (.m3u8) — KHÔNG qua MSE (phát native)",
    "http-dash": "DASH (.mpd) — thường qua MSE, kiểm tra log addSourceBuffer",
    "srcObject": "MediaStream/MediaSource gán qua srcObject",
    "file": "file:// cục bộ",
    "none": "chưa có nguồn (video chưa nạp)",
    "unknown": "không xác định",
  };

  function describeSrcKind(srcKind) {
    return SRC_KIND_TEXT[srcKind] || SRC_KIND_TEXT.unknown;
  }

  /**
   * Biến DỮ KIỆN THÔ (từ interceptor + content script) thành mã lý do duy nhất.
   *
   * `facts` là hợp của:
   *   - payload `BUFFER_STATUS_RESPONSE` của interceptor (các trường audioSbCount, mimeTypes…)
   *   - thông tin content script tự đo (hasVideo, durationFinite, bufferedAheadVideo…)
   *
   * Trả về: { code, detail, facts } — `detail` là một câu tiếng Việt nêu ĐÚNG chỉ số đã chặn.
   */
  function classifyLookaheadAvailability(facts) {
    const f = facts || {};
    const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
    const base = { facts: f };

    const hasVideo = !!f.hasVideo;
    const cached = num(f.cachedChunksCount);
    const init = !!f.hasInitSegment;
    const appends = num(f.appendCount);
    //: Số lần append TRÊN BUFFER CÓ AUDIO. `appendCount` gộp cả buffer video nên không dùng được
    //: để kết luận "chưa có byte audio" (xem chú thích trong nhánh `NO_APPEND_YET`).
    const audioAppends = (f.appendsAudioBearing === undefined || f.appendsAudioBearing === null)
      ? appends : num(f.appendsAudioBearing);
    //: Cờ nhận SourceBuffer MUXED (video+audio). Bật từ 0.6.7 ⇒ MUXED vẫn dùng được cho Pipeline B.
    const acceptMuxed = f.acceptMuxedAudio !== false;
    const audioSb = num(f.audioSbCount);
    const muxedSb = num(f.muxedSbCount);
    const audioCapable = audioSb > 0 || (acceptMuxed && muxedSb > 0);
    const sourceMode = audioSb === 0 && muxedSb > 0 ? "MUXED" : (audioSb > 0 ? "audio-only" : "none");

    if (!hasVideo) return { ...base, code: "NO_VIDEO", detail: "Không có thẻ <video> nào." };
    if (f.durationFinite === false) {
      return { ...base, code: "LIVE_STREAM", detail: "duration = " + (f.duration ?? "Infinity") };
    }
    if (Array.isArray(f.keySystems) && f.keySystems.length) {
      return { ...base, code: "DRM", detail: "EME: " + f.keySystems.join(", ") };
    }
    if (f.hasMseApi === false) {
      return { ...base, code: "NO_MSE_API", detail: "window.MediaSource không tồn tại" };
    }
    if (f.mseUsed === false) {
      return {
        ...base,
        code: "MSE_UNUSED",
        detail: `0 lời gọi addSourceBuffer; srcKind=${f.srcKind || "unknown"} `
          + `(${describeSrcKind(f.srcKind)}); Range đã ghi: ${num(f.mediaRangeCount)}`,
      };
    }
    if (f.videoMismatch) {
      return {
        ...base,
        code: "VIDEO_MISMATCH",
        detail: `interceptor@${f.videoMismatch.activeSrc || "?"} (t=${f.videoMismatch.activeTime}s) `
          + `vs content-script@${f.videoMismatch.requestedSrc || "?"} (t=${f.videoMismatch.requestedTime}s)`,
      };
    }
    if (num(f.audioSbCount) === 0 && !audioCapable) {
      if (num(f.muxedSbCount) > 0) {
        // Chỉ tới đây khi cờ nhận-MUXED ĐANG TẮT (xem `audioCapable` phía trên).
        return {
          ...base,
          code: "MUXED_ONLY",
          detail: `${num(f.muxedSbCount)} SourceBuffer MUXED, 0 audio-only, acceptMuxedAudio=false: `
            + `${(f.mimeTypes || []).join(" | ")}`,
        };
      }
      if (num(f.unknownSbCount) > 0) {
        return {
          ...base,
          code: "UNKNOWN_MIME_SB",
          detail: `${num(f.unknownSbCount)} SourceBuffer không rõ mime; container sniff: ${f.container || "?"}`,
        };
      }
      if (num(f.videoSbCount) > 0) {
        return {
          ...base,
          code: "VIDEO_ONLY_SB",
          detail: `${num(f.videoSbCount)} SourceBuffer VIDEO: ${(f.mimeTypes || []).join(" | ")}`,
        };
      }
      return { ...base, code: "MSE_UNUSED", detail: "mseUsed=true nhưng chưa ghi nhận SourceBuffer nào" };
    }
    if (cached === 0 && !init) {
      // Chỉ đếm append TRÊN BUFFER CÓ AUDIO: `appendCount` gộp cả buffer video nên bản trước báo
      // "14 append nhưng cache rỗng" trong khi 14 lần đó đều là VIDEO (đo thật 2026-10-05, xvideos).
      if (audioAppends === 0) {
        return {
          ...base,
          code: "NO_APPEND_YET",
          detail: `${audioSb} SB audio / ${num(f.muxedSbCount)} muxed, 0 trong ${appends} lần append `
            + `là audio (${num(f.appendsSkippedVideo)} lần thuộc buffer video) — trình phát chưa nạp audio`,
        };
      }
      // Lưu ý: `Number(null)` = 0 ⇒ phải loại null/undefined TRƯỚC khi so sánh, nếu không mọi
      // báo cáo không có thông tin reset đều bị hiểu nhầm là "vừa bị xoá 0s trước".
      const sinceResetRaw = f.msSinceCacheReset;
      const sinceReset = (sinceResetRaw === null || sinceResetRaw === undefined || sinceResetRaw === "")
        ? NaN : Number(sinceResetRaw);
      if (Number.isFinite(sinceReset) && sinceReset >= 0 && sinceReset < 20000) {
        return {
          ...base,
          code: "SOURCE_CHANGED_RESET",
          detail: `cache bị XOÁ ${(sinceReset / 1000).toFixed(1)}s trước `
            + `(lý do: ${f.lastCacheResetReason || "?"}), sau ${audioAppends} lần append audio`,
        };
      }
      return {
        ...base,
        code: "NO_AUDIO_BYTES",
        detail: `${audioAppends} lần append audio nhưng cache rỗng `
          + `(init đã thấy: ${num(f.appendsInit)}, lỗi append: ${num(f.appendErrorCount)}, `
          + `không đọc được byte: ${num(f.appendsNoRawBytes)})`,
      };
    }
    if (cached === 0 && init) {
      return {
        ...base,
        code: "INIT_ONLY",
        detail: `có init segment (${audioAppends} append audio) nhưng chưa có mảnh media`,
      };
    }
    if (cached > 0 && num(f.mediaTaggedChunks) === 0) {
      return {
        ...base,
        code: "NO_MEDIA_TIMELINE",
        detail: `${cached} mảnh trong cache, 0 mảnh có mốc media (lỗi append: ${num(f.appendErrorCount)})`,
      };
    }
    if (f.bufferedAheadVideo !== undefined && num(f.bufferedAheadVideo) < 1.5) {
      return {
        ...base,
        code: "BUFFERED_LOW",
        detail: `video.buffered trước playhead chỉ ${num(f.bufferedAheadVideo).toFixed(2)}s`,
      };
    }
    return {
      ...base,
      code: "OK",
      detail: `${audioSb} SourceBuffer audio + ${num(f.muxedSbCount)} muxed (chế độ ${sourceMode}, `
        + `acceptMuxedAudio=${acceptMuxed}), ${cached} mảnh, `
        + `${num(f.mediaTaggedChunks)} mảnh có mốc, container=${f.container || "?"}`,
    };
  }

  // ── LOGGER ──────────────────────────────────────────────────────────────────
  /**
   * Logger chẩn đoán: mọi dòng đều có tiền tố `[<tag>][Diag][<kênh>]` để lọc nhanh trong
   * DevTools Console (gõ `[Diag]` vào ô Filter).
   *
   * Chống trùng: `once(key, fn)` chỉ in một lần cho mỗi `key`; `throttled(key, ms, fn)` in
   * tối đa một lần trong mỗi `ms` cho mỗi `key`.
   */
  class LookaheadDiag {
    constructor(tag) {
      this.tag = tag || "LA";
      this.enabled = true;
      this._throttleAt = Object.create(null);
    }

    setEnabled(on) { this.enabled = !!on; return this; }

    _prefix(channel) {
      return `%c[${this.tag}][Diag][${channel}]`;
    }

    _style(level) {
      if (level === "warn") return "color:#f59e0b;font-weight:bold";
      if (level === "error") return "color:#ef4444;font-weight:bold";
      if (level === "ok") return "color:#22c55e;font-weight:bold";
      return "color:#38bdf8";
    }

    /** Ghi log có phân loại. `data` là object tuỳ chọn in kèm. */
    log(channel, message, data) {
      if (!this.enabled) return;
      if (data === undefined) console.log(`${this._prefix(channel)} ${message}`, this._style("info"));
      else console.log(`${this._prefix(channel)} ${message}`, this._style("info"), data);
    }

    ok(channel, message, data) {
      if (!this.enabled) return;
      if (data === undefined) console.log(`${this._prefix(channel)} ${message}`, this._style("ok"));
      else console.log(`${this._prefix(channel)} ${message}`, this._style("ok"), data);
    }

    warn(channel, message, data) {
      if (!this.enabled) return;
      if (data === undefined) console.warn(`${this._prefix(channel)} ${message}`);
      else console.warn(`${this._prefix(channel)} ${message}`, data);
    }

    error(channel, message, data) {
      if (!this.enabled) return;
      if (data === undefined) console.error(`${this._prefix(channel)} ${message}`);
      else console.error(`${this._prefix(channel)} ${message}`, data);
    }

    /** In tối đa 1 lần / `ms` cho mỗi `key`. Trả về true nếu lần này THỰC SỰ in. */
    throttled(key, ms, level, channel, message, data) {
      const now = Date.now();
      const last = this._throttleAt[key] || 0;
      if (now - last < ms) return false;
      this._throttleAt[key] = now;
      if (level === "warn") this.warn(channel, message, data);
      else if (level === "error") this.error(channel, message, data);
      else this.log(channel, message, data);
      return true;
    }

    /**
     * In một BÁO CÁO nhiều dòng (mảng [nhãn, giá trị]) — dùng cho `printReport()`.
     * Chỉ dùng `console.group/groupEnd/log` để tương thích mọi môi trường.
     */
    group(title, rows, level = "info") {
      if (!this.enabled) return;
      try { console.group(`%c[${this.tag}][Diag] ${title}`, this._style(level)); }
      catch (e) { console.log(`[${this.tag}][Diag] ${title}`); }
      for (const row of rows || []) {
        if (!row) continue;
        console.log(`  • ${row[0]}: ${row[1]}`);
      }
      try { console.groupEnd(); } catch (e) {}
    }
  }

  /**
   * Bảng "dữ kiện thô" → các dòng báo cáo đọc được. Dùng chung cho content script và popup.
   */
  function buildDiagnosticRows(facts, verdict) {
    const f = facts || {};
    const v = verdict || classifyLookaheadAvailability(f);
    const rows = [];
    rows.push(["Kết luận", `${v.code} — ${explainLookaheadReason(v.code).vi}`]);
    if (v.detail) rows.push(["Chi tiết", v.detail]);
    rows.push(["Thẻ video", f.hasVideo ? "có" : "KHÔNG"]);
    rows.push(["Nguồn phát (srcKind)", `${f.srcKind || "?"} — ${describeSrcKind(f.srcKind)}`]);
    rows.push(["MediaSource API", f.hasMseApi === false ? "KHÔNG có" : "có"]);
    rows.push(["Đã dùng MSE", f.mseUsed ? "có" : "KHÔNG (0 addSourceBuffer)"]);
    rows.push(["SourceBuffer", `audio=${f.audioSbCount || 0}, video=${f.videoSbCount || 0}, `
      + `muxed=${f.muxedSbCount || 0}, không rõ mime=${f.unknownSbCount || 0}`]);
    rows.push(["Nhận MUXED", f.acceptMuxedAudio === false ? "TẮT (audio trong buffer muxed bị bỏ)"
      : "BẬT (dùng được cả SourceBuffer video+audio)"]);
    if (f.mimeTypes && f.mimeTypes.length) rows.push(["Mime đã thấy", f.mimeTypes.join(" | ")]);
    rows.push(["Container sniff", f.container || "(chưa có byte)"]);
    rows.push(["Init segment", f.hasInitSegment ? "có" : "chưa"]);
    rows.push(["Mảnh trong cache", `${f.cachedChunksCount || 0} `
      + `(có mốc media thật: ${f.mediaTaggedChunks || 0})`]);
    rows.push(["Lần append", `${f.appendCount || 0} tổng — trong đó `
      + `${f.appendsAudioBearing !== undefined ? f.appendsAudioBearing : "?"} trên buffer AUDIO, `
      + `${f.appendsSkippedVideo || 0} bỏ qua (buffer video), `
      + `${f.appendsNoRawBytes || 0} không đọc được byte, lỗi=${f.appendErrorCount || 0}`]);
    if (f.msSinceCacheReset !== undefined && f.msSinceCacheReset !== null) {
      rows.push(["Cache bị xoá gần nhất", `${Math.round(Number(f.msSinceCacheReset) / 1000)}s trước `
        + `(lý do: ${f.lastCacheResetReason || "?"}, tổng ${f.cacheResetCount || 0} lần)`]);
    }
    rows.push(["DRM (EME)", (f.keySystems && f.keySystems.length) ? f.keySystems.join(", ") : "không"]);
    rows.push(["Range đã ghi để tải lại", String(f.mediaRangeCount || 0)]);
    if (f.videoMismatch) {
      rows.push(["LỆCH THẺ VIDEO", `interceptor=${f.videoMismatch.activeSrc} `
        + `(t=${f.videoMismatch.activeTime}s) vs content=${f.videoMismatch.requestedSrc} `
        + `(t=${f.videoMismatch.requestedTime}s)`]);
    }
    const ex = explainLookaheadReason(v.code);
    if (ex.why) rows.push(["Vì sao", ex.why]);
    if (ex.fix) rows.push(["Cách xử lý", ex.fix]);
    return rows;
  }

  const api = {
    LA_REASON,
    explainLookaheadReason,
    describeSrcKind,
    classifyLookaheadAvailability,
    buildDiagnosticRows,
    LookaheadDiag,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (globalScope) globalScope.LookaheadDiagnostics = api;
})(typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : null));
