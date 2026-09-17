"""Test tầng A — auto-ducking của extension (🔉 Original audio).

**Vì sao có file này.** Người dùng báo *"kéo 🔉 Original audio xuống 5% nhưng tiếng gốc vẫn
còn lớn lắm"*. Kiểm chứng bằng cách **chạy thật** lớp `TTSAudioPlayer` với một mock video
giống trình duyệt (có `addEventListener`, và ghi `volume`/`muted` **phát sinh `volumechange`**)
thì lộ ra 3 lỗi thật — cả ba đều **im lặng**, không exception, không log:

1. Extension chỉ ghi `video.volume` **một lần** mỗi lần đổi setting. Nếu trang (hoặc chính
   người dùng kéo thanh âm lượng của trang) ghi `volume` sau đó, ducking bị huỷ **âm thầm** —
   mà `isDucked` vẫn `true` nên cả đường restore cũng sai.
2. `duckingLevel` ngoài `[0,1]` (ví dụ truyền `5` thay vì `0.05`) bị clamp thành `1.0`
   = **âm lượng đầy** ⇒ một lỗi sai đơn vị biến thành "mở to hết cỡ".
3. `volume > 0 ? volume : 1.0` quy video đang **tắt tiếng** thành `1.0`, nên khi restore,
   video muted bị **bật lên 100%**.

Test chức năng nằm ở `backend/tests/js/tts_ducking_test.js` (Node). File này gọi nó và
**bỏ qua nếu máy không có Node** (không làm hỏng bộ test), theo đúng pattern của `test_14`.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
EXT_LIB = ROOT / "extension_firefox" / "lib"
PLAYER = EXT_LIB / "tts-player.js"
CAPTURE = EXT_LIB / "audio-capture.js"
HARNESS = Path(__file__).resolve().parent / "js" / "tts_ducking_test.js"
GRAPH_HARNESS = Path(__file__).resolve().parent / "js" / "audio_capture_ducking_test.js"


def _run_node(script: Path) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("không có node để chạy kiểm thử chức năng ducking")
    assert script.exists(), f"thiếu {script}"
    proc = subprocess.run(
        [node, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"{script.name} FAIL:\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "KET QUA: PASS" in proc.stdout
    return proc.stdout


def test_ducking_behaviour_suite_runs_in_node():
    """Chạy thật `TTSAudioPlayer` với mock video giống trình duyệt (16 nhóm kiểm tra)."""
    _run_node(HARNESS)


def test_audio_graph_keeps_asr_branch_before_duck_gain():
    """ĐỒ THỊ AUDIO: nhánh ASR phải lấy TRƯỚC gain ducking (7 nhóm kiểm tra).

    Đây là bản sửa cho triệu chứng *"0% ⇒ không có tiếng gốc, không có phụ đề, không có TTS"*:
    `createMediaElementSource` cho tín hiệu đã bị nhân bởi `video.volume`, nên nếu duck bằng
    `video.volume` thì ASR cũng nhận audio nhỏ đi — và ở 0% là im lặng kỹ thuật số.
    """
    _run_node(GRAPH_HARNESS)


def test_capture_graph_is_split_in_source():
    """Chốt ở mức mã nguồn: phải có `createGain` trên nhánh nghe và KHÔNG nối thẳng destination."""
    src = CAPTURE.read_text(encoding="utf-8")
    assert "createGain()" in src, "phải có GainNode cho nhánh nghe"
    assert "duckGain" in src, "phải đặt tên và giữ tham chiếu gain"
    assert "_bsSourceNode.connect(target.__bsAudioCtx.destination)" not in src, (
        "KHÔNG được nối thẳng sourceNode -> destination (đó là lỗi cũ làm ASR bị duck theo)"
    )
    # stop() phải nối lại qua gain, không nối thẳng
    idx = src.index("re-route it so video audio still plays")
    assert "connect(dg)" in src[idx: idx + 900], "stop() phải nối lại QUA duckGain"


def test_ducking_guard_is_wired_in_source():
    """Chốt ở mức mã nguồn: phải có guard `volumechange` và clamp `duckingLevel`.

    Đây là hai mắt xích sửa lỗi — nếu ai xoá, test này đổ để lỗi không quay lại.
    """
    src = PLAYER.read_text(encoding="utf-8")
    assert "volumechange" in src, "phải có guard volumechange chống trang ghi đè volume"
    assert "_attachVolumeGuard" in src and "_detachVolumeGuard" in src, "guard phải được gỡ được"
    assert "_normalizeDuckingLevel" in src, "phải clamp duckingLevel trước khi nhân"
    assert "originalVideoMuted" in src, "phải ghi nhớ trạng thái muted để restore đúng"


def test_ducking_guard_is_released_on_teardown():
    """Chốt: guard phải được gỡ ở cả đường tắt ducking, đổi video, và destroy.

    Nếu không gỡ, listener rò rỉ và mỗi lần đổi video lại chồng thêm một handler.
    """
    src = PLAYER.read_text(encoding="utf-8")
    # Neo vào ĐỊNH NGHĨA method, không phải call site (`_restoreVideoVolume()` cũng được
    # GỌI ở `setTargetVideo`, nên `index()` trần sẽ trúng nhầm chỗ đó).
    idx = src.index("_restoreVideoVolume() {")
    body = src[idx: idx + 700]
    assert "_detachVolumeGuard()" in body, "_restoreVideoVolume phải gỡ guard"
    # và phải trả lại cả trạng thái muted, không chỉ volume
    assert "originalVideoMuted" in body, "_restoreVideoVolume phải trả lại cả muted"


def test_content_script_reapplies_ducking_on_play():
    """Nhiều trang reset `video.volume` lúc bắt đầu phát ⇒ phải áp lại ở mốc play/playing."""
    src = (ROOT / "extension_firefox" / "content" / "content-script.js").read_text(encoding="utf-8")
    assert "reapplyDucking" in src, "content-script phải gọi reapplyDucking() (lưới an toàn thứ hai)"


def test_popup_sends_ducking_level_as_fraction():
    """Hợp đồng đơn vị: popup gửi 0..1 (5% ⇒ 0.05), KHÔNG gửi phần trăm thô."""
    src = (ROOT / "extension_firefox" / "popup" / "popup.js").read_text(encoding="utf-8")
    assert "duckingPercent / 100" in src, "popup phải chia 100 trước khi gửi"
