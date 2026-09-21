"""Test tầng A — P3.x/P4.6: worklet, VAD ngoài lock, và các sửa client-side.

Bao gồm:
- P4.6/F-13: model VAD chạy NGOÀI `RLock` ⇒ `force_end()` không còn bị block.
- P4.6: batch inference đang bay bị BỎ sau force_end (không sinh START mới).
- P3.4: kiểm tra tĩnh worklet được tham chiếu đúng và có fallback (không cần browser).
- P3.5c/P3.5d: kiểm tra tĩnh các sửa trong JS (timestamp drift, huỷ timer reconnect).
"""

import re
import threading
import time
from pathlib import Path

import pytest

from backend.tests.fakes import FakeVADEngine, make_silence_pcm, pcm_to_int16_bytes

EXT_DIR = Path(__file__).resolve().parent.parent.parent / "extension_firefox"


class _SlowVADEngine(FakeVADEngine):
    """VAD giả ngủ `delay` giây mỗi frame — mô phỏng forward PyTorch."""

    def __init__(self, delay: float = 0.25, rms_threshold: float = 0.02):
        super().__init__(rms_threshold=rms_threshold)
        self.delay = delay
        self.calls = 0

    def is_speech(self, frame_int16, state, threshold=None, silence_ms=None):
        self.calls += 1
        time.sleep(self.delay)
        return super().is_speech(frame_int16, state, threshold=threshold, silence_ms=silence_ms)


# ------------------------------------------------------------------ P4.6 / F-13
def test_vad_model_runs_outside_lock(session_factory):
    """P4.6: `force_end()` KHÔNG được block bởi inference VAD đang chạy.

    Trước đây `engine.is_speech()` chạy trong `with self._lock`, nên `force_end()` gọi
    từ event loop (khi cleanup phiên) phải chờ hết forward ⇒ spike cleanup 102 ms.
    """
    session = session_factory()
    proc = session.vad_processor

    slow = _SlowVADEngine(delay=0.25)
    proc._engine = slow
    proc._attach_engine(slow)
    proc._state = proc._new_state(slow)
    proc._stopped = False

    # 800 bytes = 2 frame ⇒ inference mất ~0.5 s
    t = threading.Thread(target=lambda: proc.feed_chunk(bytes(800)), daemon=True)
    t.start()
    time.sleep(0.1)  # để feed_chunk vào tới bước chạy model (ngoài lock)

    t0 = time.perf_counter()
    proc.force_end()
    elapsed = time.perf_counter() - t0

    t.join(timeout=5.0)
    assert slow.calls >= 1, "engine phải đã được gọi"
    assert elapsed < 0.15, (
        f"force_end() bị block {elapsed * 1000:.0f} ms — model VAD vẫn chạy TRONG lock (P4.6 chưa đạt)"
    )


def test_vad_batch_discarded_after_force_end(session_factory):
    """P4.6: kết quả của batch đang bay phải bị bỏ sau force_end (không START mới)."""
    from backend.core.metrics import metrics_collector

    session = session_factory()
    proc = session.vad_processor
    slow = _SlowVADEngine(delay=0.2)
    proc._engine = slow
    proc._attach_engine(slow)
    proc._state = proc._new_state(slow)
    proc._stopped = False

    speech = (b"\x10\x27" * 400)  # biên độ ~0.3 ⇒ vượt ngưỡng RMS
    before = metrics_collector.get_counter("vad.batch_discarded_engine_switch")

    t = threading.Thread(target=lambda: proc.feed_chunk(speech), daemon=True)
    t.start()
    time.sleep(0.05)
    proc.force_end()          # chốt/đóng trong lúc batch đang bay
    t.join(timeout=5.0)

    assert metrics_collector.get_counter("vad.batch_discarded_engine_switch") > before
    assert proc._state.is_speech is False, "không được sinh speech START mới sau force_end"


def test_vad_reset_allows_audio_again(session_factory):
    """`reset()` phải mở lại đường nhận audio (xoá cờ `_stopped`)."""
    session = session_factory()
    proc = session.vad_processor
    proc.force_end()
    assert proc._stopped is True
    proc.reset()
    assert proc._stopped is False


def test_vad_still_detects_speech_normally(session_factory):
    """Sau refactor, phát hiện speech/im lặng vẫn phải hoạt động đúng."""
    session = session_factory()
    proc = session.vad_processor
    speech = (b"\x10\x27" * 400) * 4   # 4 frame ~ biên độ lớn
    for _ in range(4):
        proc.feed_chunk(speech)
    assert proc._state.is_speech is True
    assert proc._state.total_samples_processed > 0


# --------------------------------------------------------------- P3.4 (worklet)
def test_worklet_file_registers_processor_and_resamples():
    """P3.4: worklet phải đăng ký đúng tên processor và có resample + transferable."""
    src = (EXT_DIR / "lib" / "audio-processor.js").read_text(encoding="utf-8")
    assert 'registerProcessor("audio-capture-processor"' in src
    assert "_resample" in src, "worklet phải tự resample về 16 kHz"
    assert "targetSampleRate" in src
    assert "[this.buffer.buffer]" in src, "phải transfer ArrayBuffer (zero-copy), không clone"
    assert "captureTimestamp" in src, "mỗi chunk phải mang timestamp"


def test_audio_worklet_runs_in_simulated_scope():
    """P3.4: CHẠY THẬT `process()` của worklet trong AudioWorkletGlobalScope giả lập.

    Kiểm tra chức năng (không chỉ chuỗi trong file): resample 48k->16k giữ đúng tần số,
    không đứt gãy pha ở biên block, chunk đúng kích thước + timestamp không trôi,
    buffer được transfer, và ping/pong cho watchdog của main thread.

    Cần Node; nếu máy không có Node thì bỏ qua (không làm hỏng bộ test).
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("không có node để chạy kiểm thử chức năng worklet")

    script = Path(__file__).resolve().parent / "js" / "worklet_harness.js"
    assert script.exists(), f"thiếu {script}"
    proc = subprocess.run(
        [node, str(script)], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    assert proc.returncode == 0, f"worklet harness FAIL:\n{proc.stdout}\n{proc.stderr}"
    assert "KẾT QUẢ: PASS" in proc.stdout


def test_translation_shown_once_policy_runs():
    """Bản dịch chỉ hiện MỘT LẦN: chạy kiểm thử logic thuần của `subtitle-policy.js`."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("không có node để chạy kiểm thử policy phụ đề")

    script = Path(__file__).resolve().parent / "js" / "subtitle_policy_test.js"
    assert script.exists(), f"thiếu {script}"
    proc = subprocess.run(
        [node, str(script)], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60,
    )
    assert proc.returncode == 0, f"subtitle policy FAIL:\n{proc.stdout}\n{proc.stderr}"
    assert "KẾT QUẢ: PASS" in proc.stdout


def test_subtitle_renderer_three_layers_run_in_node():
    """F-52: chạy THẬT SubtitleRenderer trong DOM giả lập (Node) để chốt hành vi 3 tầng.

    Bug đã gặp: câu đang ở TẦNG 2 (chờ bản dịch) bị câu mới đẩy lên TẦNG 1, bản dịch tới
    sau KHÔNG được vẽ ra (tầng 1 chỉ cập nhật `.bs-translated` nếu node đã tồn tại).
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("không có node để chạy kiểm thử renderer phụ đề")

    script = Path(__file__).resolve().parent / "js" / "subtitle_renderer_harness.js"
    assert script.exists(), f"thiếu {script}"
    proc = subprocess.run(
        [node, str(script)], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60,
    )
    assert proc.returncode == 0, f"subtitle renderer FAIL:\n{proc.stdout}\n{proc.stderr}"
    assert "KẾT QUẢ: PASS" in proc.stdout


def test_renderer_skips_partial_translation_and_popup_has_toggle():
    """Nối dây: renderer phải hỏi policy + overlay phải chuyển nguyên payload có `partial`."""
    renderer = (EXT_DIR / "lib" / "subtitle-renderer.js").read_text(encoding="utf-8")
    assert "BSSubtitlePolicy" in renderer, "renderer phải dùng policy chung"
    assert "shouldApplyTranslation" in renderer, "renderer phải hỏi policy trước khi hiện"
    assert "showTranslationOnce" in renderer, "phải có cờ bật/tắt chạy chữ"
    assert "showTranslationOnce" in renderer and "applySettings" in renderer

    overlay = (EXT_DIR / "content" / "overlay-manager.js").read_text(encoding="utf-8")
    assert "this.renderer.onTranslation(payload)" in overlay, (
        "overlay phải chuyển NGUYÊN payload để renderer biết đây là mảnh dịch dở"
    )

    popup_html = (EXT_DIR / "popup" / "popup.html").read_text(encoding="utf-8")
    assert 'id="chkTranslationOnce"' in popup_html, "popup phải có công tắc tắt chạy chữ"
    popup_js = (EXT_DIR / "popup" / "popup.js").read_text(encoding="utf-8")
    assert "showTranslationOnce" in popup_js, "popup phải lưu cài đặt này"

    import json

    manifest = json.loads((EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
    scripts = []
    for entry in manifest.get("content_scripts", []):
        scripts.extend(entry.get("js", []))
    assert "lib/subtitle-policy.js" in scripts, "manifest phải nạp policy TRƯỚC renderer"
    assert scripts.index("lib/subtitle-policy.js") < scripts.index("lib/subtitle-renderer.js")


def test_capture_prefers_worklet_with_fallback():
    """P3.4: audio-capture phải ưu tiên worklet, có watchdog và fallback ScriptProcessor."""
    src = (EXT_DIR / "lib" / "audio-capture.js").read_text(encoding="utf-8")
    assert "audioWorklet.addModule" in src
    assert "new AudioWorkletNode" in src
    assert "_fallbackToScriptProcessor" in src
    assert "_workletWatchdog" in src, "cần watchdog nếu worklet không phát chunk"
    assert "audio-capture-processor" in src
    assert "createScriptProcessor" in src, "phải giữ đường dự phòng"
    # P3.4/F-29: không còn ép AudioContext về 16 kHz
    assert "sampleRate: this.sampleRate" not in src, (
        "không được ép sampleRate của AudioContext (band-limit audio người dùng xuống 8 kHz)"
    )


def test_manifest_exposes_worklet():
    """P3.4: worklet phải nằm trong web_accessible_resources để addModule nạp được."""
    import json

    manifest = json.loads((EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
    war = manifest.get("web_accessible_resources", [])
    flat = []
    for item in war:
        if isinstance(item, str):
            flat.append(item)
        elif isinstance(item, dict):
            flat.extend(item.get("resources", []))
    assert any("audio-processor.js" in r for r in flat), f"worklet chưa được expose: {war}"


# ------------------------------------------------- P3.5c / P3.5d / P3.2 (JS tĩnh)
def test_capture_fixes_timestamp_drift():
    """P3.5c: phải trừ phần residual khỏi timestamp của chunk (hết trôi 21–64 ms)."""
    src = (EXT_DIR / "lib" / "audio-capture.js").read_text(encoding="utf-8")
    assert "residualSec" in src
    assert "baseTimestamp - residualSec" in src


def test_ws_client_cancels_reconnect_timer():
    """P3.5d: phải lưu và huỷ timer reconnect (hết port/WS zombie sau Stop)."""
    src = (EXT_DIR / "lib" / "ws-client.js").read_text(encoding="utf-8")
    assert "_reconnectTimer" in src
    assert "_cancelReconnect" in src
    assert re.search(r"disconnect\(\)\s*\{[^}]*_cancelReconnect", src, flags=re.DOTALL), (
        "disconnect() phải gọi _cancelReconnect()"
    )


def test_service_worker_has_backpressure_limits():
    """P3.2: service worker phải đọc bufferedAmount và có ngưỡng."""
    src = (EXT_DIR / "background" / "service-worker.js").read_text(encoding="utf-8")
    assert "bufferedAmount" in src
    assert "SEND_BUFFER_SOFT_LIMIT" in src
    assert "SEND_BUFFER_HARD_LIMIT" in src
    assert "notifyBackpressure" in src


def test_tts_player_queue_is_bounded():
    """P3.5a: hàng đợi TTS phải có trần (trước đây không giới hạn ⇒ RAM + lệch tiếng)."""
    src = (EXT_DIR / "lib" / "tts-player.js").read_text(encoding="utf-8")
    assert "maxQueueLength" in src
    assert "_trimQueue" in src
    assert "maxLagMs" in src
    assert "decodeAudioData" in src, "phải dùng decodeAudioData cho đường binary (P3.1)"


def test_content_script_single_frame_tts_and_protocol_version():
    """P3.5b + P3.0: chỉ một frame phát TTS, và client khai báo protocolVersion."""
    raw = (EXT_DIR / "content" / "content-script.js").read_text(encoding="utf-8")
    # Bỏ dòng comment để không match vào phần giải thích bug cũ.
    src = "\n".join(
        line for line in raw.splitlines() if not line.lstrip().startswith("//")
    )
    assert "PROTOCOL_VERSION" in src
    assert "protocolVersion" in src
    assert "ownerVideo" in src, "phải xác định frame sở hữu video để chỉ phát 1 lần"
    # G9/G10: không còn default cứng cho model dịch / VAD engine
    assert 'cfg.translationModel || "xiaomi"' not in src, (
        "không được default translationModel='xiaomi' (không có file GGUF cục bộ)"
    )
    assert 'cfg.vadEngine || "fsmn-vad"' not in src, (
        "không được default vadEngine='fsmn-vad' (khác default backend, ghi đè mỗi lần khởi động)"
    )
    assert "out.translationModel = cfg.translationModel" in src
    assert "out.vadEngine = cfg.vadEngine" in src


def test_content_script_caches_negative_video_scan():
    """P3.5e: kết quả ÂM của findVideo phải được cache (hết quét DOM mỗi sự kiện)."""
    src = (EXT_DIR / "content" / "content-script.js").read_text(encoding="utf-8")
    assert "videoScanMissUntil" in src
    assert "VIDEO_SCAN_MISS_TTL_MS" in src
