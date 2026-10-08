#!/usr/bin/env python3
"""Sinh 2 bản extension (Firefox và Chrome/Edge) từ MỘT nguồn duy nhất.

VÌ SAO CẦN SCRIPT NÀY
---------------------
Trước đây `extension_firefox/` và `extension_chrome_edge/` là hai bản copy tay song song.
Hậu quả đo được (2026-10):

  * 29/33 file giống hệt nhau từng byte — 566.714 byte trùng lặp vô ích.
  * 4 file đã LỆCH THẬT, trong đó 2 file (`lib/ws-client.js`, `lib/lookahead-client.js`)
    lệch ở đúng 1 dòng quyết định cách kết nối backend.
  * Backend test chỉ trỏ `extension_firefox` (17 tham chiếu) so với `extension_chrome_edge`
    (1 tham chiếu) ⇒ bản Chrome/Edge KHÔNG được test và âm thầm lệch.
  * Mọi bản vá phải áp 2 lần (xem `report/09_lookahead_offline_batch/phase6_*.md`).

Từ nay `extension_src/` là NGUỒN DUY NHẤT. Hai thư mục `extension_*` là OUTPUT SINH RA.
Sửa code ⇒ sửa trong `extension_src/` ⇒ chạy script này.

    python tools/build_extensions.py

KHÁC BIỆT DUY NHẤT GIỮA HAI TRÌNH DUYỆT (và lý do)
--------------------------------------------------
1. `manifest.json` — buộc phải khác, không thể gộp:
     • Chromium MV3 : `background.service_worker` (chỉ MỘT file duy nhất).
     • Firefox  MV3 : `background.scripts` (mảng nhiều file) + `browser_specific_settings.gecko`
                      + `match_origin_as_fallback`.
2. `lib/browser-config.js` — MỘT cờ `useBridge`, xem `BROWSERS` bên dưới.

`background/service-worker.js` KHÔNG cần biến thể riêng: nó tự dò xem thư viện phụ thuộc
đã có sẵn chưa rồi mới `importScripts()` (xem guard ở đầu file đó).

Cách ly 2 thư mục output: script chỉ ghi/xoá trong `extension_chrome_edge/` và
`extension_firefox/`, không bao giờ đụng tới `extension_src/`.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "extension_src"

#: Tên file manifest trong nguồn (không bao giờ được copy nguyên tên sang output).
MANIFEST_TEMPLATES = {"manifest.chrome.json", "manifest.firefox.json"}

#: File do script sinh ra, KHÔNG tồn tại trong `extension_src/`.
GENERATED_FILE = Path("lib") / "browser-config.js"

#: Cấu hình riêng cho từng trình duyệt.
#:
#: `use_bridge` là điểm khác biệt hành vi DUY NHẤT còn lại giữa hai bản:
#:   • firefox  = True  — background script KHÔNG bị kill sau ~30s idle ⇒ bridge qua
#:                        `chrome.runtime.connect` là đường chính, giúp vượt CSP/CORS
#:                        của các trang web bên ngoài.
#:   • chromium = False — MV3 service worker bị kill sau ~30s idle ⇒ bridge chết giữa
#:                        phiên (triệu chứng: luồng audio đứt) ⇒ mở WebSocket thẳng từ
#:                        content script.
#: ĐỔI GIÁ TRỊ Ở ĐÂY, không sửa trong `lib/ws-client.js` / `lib/lookahead-client.js`.
BROWSERS: dict[str, dict[str, object]] = {
    "extension_firefox": {
        "manifest": "manifest.firefox.json",
        "browser": "firefox",
        "use_bridge": True,
    },
    "extension_chrome_edge": {
        "manifest": "manifest.chrome.json",
        "browser": "chromium",
        "use_bridge": False,
    },
}

BROWSER_CONFIG_TEMPLATE = """\
// ⚠️  FILE SINH TỰ ĐỘNG — KHÔNG SỬA TAY.
//
// Nguồn duy nhất : extension_src/
// Script sinh ra : tools/build_extensions.py
// Sinh cho       : {browser}
//
// Mọi thay đổi trực tiếp ở file này sẽ bị GHI ĐÈ ở lần build kế tiếp.
// Muốn đổi hành vi, sửa bảng BROWSERS trong tools/build_extensions.py.
//
// `useBridge`: dùng background bridge (`chrome.runtime.connect`) để vượt CSP/CORS
// của trang ngoài hay không. Xem giải thích đầy đủ ở đầu tools/build_extensions.py.
globalThis.BS_BROWSER_CONFIG = Object.freeze({{
  browser: "{browser}",
  useBridge: {use_bridge},
}});
"""


def _iter_source_files() -> list[Path]:
    """Mọi file trong `extension_src/`, trừ các manifest template."""
    files = [
        p.relative_to(SRC)
        for p in SRC.rglob("*")
        if p.is_file() and p.name not in MANIFEST_TEMPLATES
    ]
    return sorted(files)


def _validate_source() -> None:
    """Kiểm tra nguồn trước khi sinh, để lỗi cấu hình lộ ra ngay."""
    if not SRC.is_dir():
        raise SystemExit(f"Không tìm thấy nguồn: {SRC}")

    for name in MANIFEST_TEMPLATES:
        if not (SRC / name).is_file():
            raise SystemExit(f"Thiếu manifest template: extension_src/{name}")

    # `browser-config.js` phải được nạp TRƯỚC `ws-client.js` / `lookahead-client.js`,
    # nếu không cờ `useBridge` sẽ là `undefined` và bridge âm thầm tắt.
    for browser, cfg in BROWSERS.items():
        manifest = json.loads((SRC / str(cfg["manifest"])).read_text(encoding="utf-8"))
        for block in manifest.get("content_scripts", []):
            js = block.get("js", [])
            if "lib/ws-client.js" not in js and "lib/lookahead-client.js" not in js:
                continue
            if not js or js[0] != "lib/browser-config.js":
                raise SystemExit(
                    f"{cfg['manifest']}: 'lib/browser-config.js' phải đứng ĐẦU content_scripts.js "
                    f"(vì ws-client.js/lookahead-client.js đọc cờ từ đó). Hiện tại: {js[:2]}"
                )


def build(browser_dir: str, cfg: dict[str, object], *, check_only: bool = False) -> list[str]:
    """Sinh một thư mục extension. Trả về danh sách thay đổi (rỗng nếu đã đồng bộ)."""
    out = ROOT / browser_dir
    if not out.is_dir():
        if check_only:
            return [f"{browser_dir}/ (thiếu hoàn toàn)"]
        out.mkdir(parents=True)

    changes: list[str] = []
    expected: set[Path] = set()

    # 1. Các file dùng chung: copy NHỊ PHÂN để giống hệt từng byte.
    for rel in _iter_source_files():
        expected.add(rel)
        src_file, dst_file = SRC / rel, out / rel
        if src_file.read_bytes() == (dst_file.read_bytes() if dst_file.is_file() else None):
            continue
        changes.append(f"{browser_dir}/{rel.as_posix()}")
        if not check_only:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_file, dst_file)

    # 2. manifest.json — sinh từ template của trình duyệt tương ứng.
    manifest_bytes = (SRC / str(cfg["manifest"])).read_bytes()
    expected.add(Path("manifest.json"))
    dst_manifest = out / "manifest.json"
    if manifest_bytes != (dst_manifest.read_bytes() if dst_manifest.is_file() else None):
        changes.append(f"{browser_dir}/manifest.json")
        if not check_only:
            dst_manifest.write_bytes(manifest_bytes)

    # 3. lib/browser-config.js — cờ per-browser.
    config_text = BROWSER_CONFIG_TEMPLATE.format(
        browser=cfg["browser"],
        use_bridge="true" if cfg["use_bridge"] else "false",
    )
    expected.add(GENERATED_FILE)
    dst_config = out / GENERATED_FILE
    config_bytes = config_text.encode("utf-8")
    if config_bytes != (dst_config.read_bytes() if dst_config.is_file() else None):
        changes.append(f"{browser_dir}/{GENERATED_FILE.as_posix()}")
        if not check_only:
            dst_config.parent.mkdir(parents=True, exist_ok=True)
            # Ghi NHỊ PHÂN: `write_text` sẽ đổi LF -> CRLF trên Windows và làm
            # phép so sánh ở lần chạy sau luôn báo lệch.
            dst_config.write_bytes(config_bytes)

    # 4. Dọn file lạ: đã xoá khỏi nguồn thì cũng phải biến mất khỏi output.
    for stray in sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file()):
        if stray not in expected:
            changes.append(f"{browser_dir}/{stray.as_posix()}  [XOÁ — không còn trong nguồn]")
            if not check_only:
                (out / stray).unlink()

    return changes


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    _validate_source()

    all_changes: list[str] = []
    for browser_dir, cfg in BROWSERS.items():
        all_changes += build(browser_dir, cfg, check_only=check_only)

    if check_only:
        if all_changes:
            print("LỆCH giữa extension_src/ và 2 bản sinh ra:")
            for line in all_changes:
                print(f"  • {line}")
            print("\nChạy: python tools/build_extensions.py")
            return 1
        print("OK — 2 bản extension đồng bộ với extension_src/.")
        return 0

    if all_changes:
        print(f"Đã cập nhật {len(all_changes)} file:")
        for line in all_changes:
            print(f"  • {line}")
    else:
        print("Không có thay đổi — 2 bản extension đã đồng bộ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
