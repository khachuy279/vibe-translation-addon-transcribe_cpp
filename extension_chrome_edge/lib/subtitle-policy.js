// Quyết định HIỂN THỊ phụ đề (logic thuần, không phụ thuộc DOM) — dễ kiểm thử bằng Node.
//
// Bối cảnh: backend có thể gửi bản dịch theo từng mảnh (`partial: true`) để chữ hiện dần.
// Người dùng muốn TẮT "chạy chữ": chỉ hiện bản dịch MỘT LẦN khi đã có bản dịch hoàn chỉnh
// (`partial` không bật). Module này gom quyết định đó vào một chỗ để test được.
//
// Dùng: BSSubtitlePolicy.shouldApplyTranslation(payload, { showOnce: true })

(function (root) {
  "use strict";

  // Mặc định: TẮT chạy chữ cho bản dịch (hiện 1 lần khi có bản dịch).
  const DEFAULT_SHOW_TRANSLATION_ONCE = true;

  function isPartialTranslation(payload) {
    if (!payload || typeof payload !== "object") return false;
    if (payload.partial === true) return true;
    if (payload.is_partial === true) return true;
    if (payload.isPartial === true) return true;
    // Một số bản cũ dùng `status: "partial"`.
    const status = String(payload.status || "").toLowerCase();
    return status === "partial" || status === "streaming";
  }

  /**
   * Có nên áp bản dịch này vào phụ đề hay không?
   * @param {object} payload  thông điệp translation từ backend
   * @param {{showOnce?: boolean}} [opts]  showOnce=true (mặc định) => bỏ qua bản partial
   */
  function shouldApplyTranslation(payload, opts) {
    const showOnce = !opts || opts.showOnce === undefined
      ? DEFAULT_SHOW_TRANSLATION_ONCE
      : !!opts.showOnce;
    if (!showOnce) return true;              // cho phép chạy chữ (hành vi cũ)
    return !isPartialTranslation(payload);   // chỉ hiện khi đã có bản dịch
  }

  const api = {
    DEFAULT_SHOW_TRANSLATION_ONCE,
    isPartialTranslation,
    shouldApplyTranslation,
  };

  root.BSSubtitlePolicy = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api; // cho harness Node
  }
})(typeof self !== "undefined" ? self : this);
