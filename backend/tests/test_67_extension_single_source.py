"""test_67 — `extension_src/` là NGUỒN DUY NHẤT; 2 bản extension là output sinh ra.

BỐI CẢNH (đo được 2026-10):

Trước đây `extension_firefox/` và `extension_chrome_edge/` là hai bản copy tay song song:

  * 29/33 file giống hệt nhau từng byte — 566.714 byte trùng lặp vô ích.
  * 4 file đã LỆCH THẬT, trong đó `lib/ws-client.js` và `lib/lookahead-client.js` lệch ở
    đúng dòng quyết định cách kết nối backend (`useBridge`).
  * Backend test chỉ trỏ `extension_firefox` (17 tham chiếu) so với `extension_chrome_edge`
    (1 tham chiếu) ⇒ bản Chrome/Edge KHÔNG được test.
  * Mọi bản vá phải áp 2 lần (xem `report/09_lookahead_offline_batch/phase6_*.md`).

Nay `extension_src/` là nguồn duy nhất, `tools/build_extensions.py` sinh ra 2 bản.

Bộ test này khoá lại 3 bất biến:

1. Hai thư mục output ĐỒNG BỘ với `extension_src/` (chạy build ở chế độ `--check`).
2. Cờ per-browser trong `lib/browser-config.js` đúng giá trị, và file đó được nạp ĐẦU TIÊN
   trong `content_scripts.js` (nếu không, cờ là `undefined` ⇒ bridge âm thầm tắt).
3. `extension_src/` không chứa `manifest.json` trần (sẽ gây nhầm lẫn giữa nguồn và output).

Lưu ý: `test_59::test_extension_mirrors_stay_in_sync` kiểm hai OUTPUT giống nhau; file này
kiểm OUTPUT giống NGUỒN. Hai bài khác nhau — sửa cả hai output cho giống nhau vẫn có thể
lệch nguồn, và ngược lại.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "extension_src"
BUILD_SCRIPT = ROOT / "tools" / "build_extensions.py"


def _load_build_module() -> ModuleType:
    """Nạp `tools/build_extensions.py` (không phải package nên phải nạp theo đường dẫn)."""
    assert BUILD_SCRIPT.is_file(), f"thiếu script build: {BUILD_SCRIPT}"
    spec = importlib.util.spec_from_file_location("_build_extensions", BUILD_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_mod() -> ModuleType:
    return _load_build_module()


def test_nguon_extension_ton_tai(build_mod: ModuleType) -> None:
    """`extension_src/` phải tồn tại và có đủ 2 manifest template."""
    assert SRC.is_dir(), "thiếu extension_src/ — nguồn duy nhất của 2 bản extension"
    for name in sorted(build_mod.MANIFEST_TEMPLATES):
        assert (SRC / name).is_file(), f"thiếu extension_src/{name}"


def test_khong_co_manifest_tran_trong_nguon() -> None:
    """`extension_src/manifest.json` là SAI: manifest phải là template theo trình duyệt."""
    stray = SRC / "manifest.json"
    assert not stray.is_file(), (
        "extension_src/manifest.json không được tồn tại — dùng manifest.chrome.json / "
        "manifest.firefox.json (xem tools/build_extensions.py)"
    )


@pytest.mark.parametrize("browser_dir", ["extension_firefox", "extension_chrome_edge"])
def test_output_dong_bo_voi_nguon(build_mod: ModuleType, browser_dir: str) -> None:
    """Chạy build ở chế độ `--check`: không được có file nào lệch."""
    changes = build_mod.build(
        browser_dir, build_mod.BROWSERS[browser_dir], check_only=True
    )
    assert not changes, (
        f"{browser_dir}/ lệch khỏi extension_src/ ở {len(changes)} file:\n  "
        + "\n  ".join(changes)
        + "\n\nSửa trong extension_src/ rồi chạy: python tools/build_extensions.py"
    )


@pytest.mark.parametrize(
    ("browser_dir", "expected_browser", "expected_use_bridge"),
    [
        # Firefox: background script KHÔNG bị kill sau ~30s idle ⇒ bridge là đường chính
        # (cần để vượt CSP/CORS của trang ngoài).
        ("extension_firefox", "firefox", True),
        # Chromium: MV3 service worker bị kill sau ~30s idle ⇒ bridge chết giữa phiên
        # ⇒ mở WebSocket thẳng từ content script.
        ("extension_chrome_edge", "chromium", False),
    ],
)
def test_co_per_browser_dung_gia_tri(
    browser_dir: str, expected_browser: str, expected_use_bridge: bool
) -> None:
    """`lib/browser-config.js` phải khai đúng trình duyệt và đúng cờ `useBridge`."""
    config_path = ROOT / browser_dir / "lib" / "browser-config.js"
    assert config_path.is_file(), f"thiếu {config_path} — chưa chạy build?"

    src = config_path.read_text(encoding="utf-8")
    assert f'browser: "{expected_browser}"' in src, (
        f"{browser_dir}: browser phải là '{expected_browser}'"
    )
    expected_flag = "true" if expected_use_bridge else "false"
    assert f"useBridge: {expected_flag}" in src, (
        f"{browser_dir}: useBridge phải là {expected_flag} "
        f"(Chromium kill service worker sau ~30s idle)"
    )


@pytest.mark.parametrize("browser_dir", ["extension_firefox", "extension_chrome_edge"])
def test_browser_config_duoc_nap_truoc_ws_client(browser_dir: str) -> None:
    """`lib/browser-config.js` phải đứng ĐẦU `content_scripts.js`.

    `lib/ws-client.js` và `lib/lookahead-client.js` đọc cờ từ `globalThis.BS_BROWSER_CONFIG`
    lúc khởi tạo. Nếu file cấu hình nạp sau, cờ là `undefined` và bridge âm thầm tắt.
    """
    manifest = json.loads(
        (ROOT / browser_dir / "manifest.json").read_text(encoding="utf-8")
    )
    blocks = [
        block
        for block in manifest.get("content_scripts", [])
        if any(
            f in block.get("js", [])
            for f in ("lib/ws-client.js", "lib/lookahead-client.js")
        )
    ]
    assert blocks, f"{browser_dir}: không tìm thấy content_scripts nạp ws/lookahead client"
    for block in blocks:
        js = block["js"]
        assert js[0] == "lib/browser-config.js", (
            f"{browser_dir}: 'lib/browser-config.js' phải đứng đầu, thực tế là {js[:3]}"
        )


def test_service_worker_chung_cho_ca_hai_ban() -> None:
    """`background/service-worker.js` phải GIỐNG HỆT nhau (không còn biến thể riêng).

    Khác biệt Chromium (`importScripts` chỉ cho một `service_worker`) được xử lý bằng guard
    tự dò trong chính file đó, không bằng hai bản copy.
    """
    a = (ROOT / "extension_firefox" / "background" / "service-worker.js").read_bytes()
    b = (ROOT / "extension_chrome_edge" / "background" / "service-worker.js").read_bytes()
    assert a == b, (
        "service-worker.js đã lệch giữa 2 bản. Khác biệt nạp thư viện phải nằm trong guard "
        "`typeof BackpressureGate === \"undefined\"` ở đầu file, không phải hai bản copy."
    )
