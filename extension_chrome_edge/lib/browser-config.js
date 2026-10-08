// ⚠️  FILE SINH TỰ ĐỘNG — KHÔNG SỬA TAY.
//
// Nguồn duy nhất : extension_src/
// Script sinh ra : tools/build_extensions.py
// Sinh cho       : chromium
//
// Mọi thay đổi trực tiếp ở file này sẽ bị GHI ĐÈ ở lần build kế tiếp.
// Muốn đổi hành vi, sửa bảng BROWSERS trong tools/build_extensions.py.
//
// `useBridge`: dùng background bridge (`chrome.runtime.connect`) để vượt CSP/CORS
// của trang ngoài hay không. Xem giải thích đầy đủ ở đầu tools/build_extensions.py.
globalThis.BS_BROWSER_CONFIG = Object.freeze({
  browser: "chromium",
  useBridge: false,
});
