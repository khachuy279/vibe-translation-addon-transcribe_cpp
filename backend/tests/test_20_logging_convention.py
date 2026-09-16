"""Test tầng A — F-43: quy ước logging được giữ đồng bộ.

Quy ước nằm ở đầu `backend/utils/logger.py`. Test này **phân tích AST** mã nguồn backend
(chính xác hơn quét chuỗi, vì chỉ xét phần CHỮ của thông điệp, bỏ qua biểu thức trong
f-string) và chốt 5 điều:

1. Mọi `logger.<level>(...)` đều có `extra={"module_tag": TAG}`.
2. TAG thuộc danh sách chuẩn (không có tag tự phát như `CONFIG`, `WS.HANDLER`).
3. Phần chữ của thông điệp KHÔNG lặp lại tag module trong ngoặc vuông.
4. Trong thông điệp chỉ có 3 loại token ngoặc vuông: `[utt=…]`, `[STARTUP]`, `[SHUTDOWN]`.
5. Không có emoji trong thông điệp.

Nhờ vậy việc "dọn logging" về sau không trôi lại như cũ.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"
KNOWN_TAGS = {
    "CORE", "VAD", "ASR", "ASR_PREVIEW", "ASR_COMMIT",
    "TRANSLATE", "TTS", "WS", "MAIN", "METRICS",
}
ALLOWED_BRACKETS = ("utt=", "STARTUP", "SHUTDOWN")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]")
BRACKET_RE = re.compile(r"\[([^\[\]]{1,40})\]")
LEVELS = {"debug", "info", "warning", "error", "critical"}


def _backend_files():
    return sorted(p for p in BACKEND.rglob("*.py") if "tests" not in p.parts)


def _module_string_constants(tree: ast.Module) -> dict:
    """Tên -> giá trị của các hằng chuỗi cấp module (vd `_TAG = "TRANSLATE"`).

    VÌ SAO CẦN: `hotswap.py` đặt `_TAG = "TRANSLATE"` rồi dùng
    `extra={"module_tag": _TAG}`. Bản trước của test chỉ nhận `ast.Constant` nên coi
    hằng này là "thiếu module_tag" và báo 3 lời gọi sai — trong khi runtime hoàn toàn
    đúng. Resolve hằng chuỗi ở đây giữ nguyên độ chặt của quy ước (tag vẫn phải có và
    vẫn phải nằm trong danh sách chuẩn) mà không bắt lỗi sai.
    """
    consts: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                consts[target.id] = node.value.value
    return consts


def _logger_calls():
    """Sinh ra (file, dòng, node_call, hằng_chuỗi_cấp_module) cho mọi `logger.<level>(...)`."""
    for path in _backend_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        consts = _module_string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr in LEVELS
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "logger"
            ):
                yield path, node.lineno, node, consts


MARKER = "\x01"  # đại diện cho biểu thức trong f-string khi ghép phần chữ


def _literal_text(call: ast.Call) -> str:
    """Ghép phần CHỮ của thông điệp; token ĐỘNG (ví dụ `[{reason}]`) bị loại bỏ.

    Vì sao cần marker: nếu ghép bằng dấu cách thì `"[utt=" + "]"` sẽ tạo ra cặp ngoặc giả.
    Sau khi ghép, mọi cụm ngoặc chứa marker được xoá vì đó là token động hợp lệ.
    """
    parts = []
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            parts.append(arg.value)
        elif isinstance(arg, ast.JoinedStr):
            for value in arg.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    parts.append(MARKER)
    text = MARKER.join(parts)
    return re.sub(r"\[[^\[\]]*" + MARKER + r"[^\[\]]*\]", "", text)


def _module_tag(call: ast.Call, consts=None):
    """Giá trị `module_tag`. Chấp nhận literal HOẶC hằng chuỗi cấp module."""
    consts = consts or {}
    for kw in call.keywords:
        if kw.arg == "extra" and isinstance(kw.value, ast.Dict):
            for k, v in zip(kw.value.keys, kw.value.values):
                if isinstance(k, ast.Constant) and k.value == "module_tag":
                    if isinstance(v, ast.Constant):
                        return v.value
                    if isinstance(v, ast.Name):
                        return consts.get(v.id)
    return None


def test_every_logger_call_has_module_tag():
    missing = [
        f"{p.relative_to(ROOT)}:{line}"
        for p, line, call, consts in _logger_calls()
        if _module_tag(call, consts) is None
    ]
    assert not missing, f"{len(missing)} lời gọi logger thiếu module_tag: {missing[:10]}"


def test_module_tags_are_canonical():
    bad = {
        _module_tag(call, consts)
        for _p, _line, call, consts in _logger_calls()
        if _module_tag(call, consts) is not None and _module_tag(call, consts) not in KNOWN_TAGS
    }
    assert not bad, f"module_tag ngoài danh sách chuẩn: {sorted(bad)}"


def test_message_does_not_repeat_its_own_tag():
    offenders = []
    for path, line, call, consts in _logger_calls():
        tag = _module_tag(call, consts)
        if tag and f"[{tag}]" in _literal_text(call):
            offenders.append(f"{path.relative_to(ROOT)}:{line} (tag {tag})")
    assert not offenders, f"thông điệp lặp lại tag: {offenders[:10]}"


def test_only_allowed_bracket_tokens_in_messages():
    offenders = []
    for path, line, call, _consts in _logger_calls():
        for token in BRACKET_RE.findall(_literal_text(call)):
            if not any(token.strip().startswith(a) for a in ALLOWED_BRACKETS):
                offenders.append(f"{path.relative_to(ROOT)}:{line} [{token.strip()}]")
    assert not offenders, (
        "chỉ được dùng [utt=…], [STARTUP], [SHUTDOWN] trong thông điệp log; "
        f"còn: {offenders[:10]}"
    )


def test_no_emoji_in_log_messages():
    offenders = [
        f"{p.relative_to(ROOT)}:{line}"
        for p, line, call, _consts in _logger_calls()
        if EMOJI_RE.search(_literal_text(call))
    ]
    assert not offenders, f"còn emoji trong log: {offenders[:10]}"


def test_logging_convention_is_documented():
    src = (BACKEND / "utils" / "logger.py").read_text(encoding="utf-8")
    assert "QUY ƯỚC LOGGING" in src, "quy ước phải được ghi ngay trong utils/logger.py"
    for tag in sorted(KNOWN_TAGS):
        assert tag in src, f"danh sách tag trong tài liệu thiếu {tag}"


def test_module_colors_cover_all_standard_tags():
    """Mọi tag chuẩn phải có màu riêng (thiếu thì log hiển thị lệch nhau)."""
    from backend.utils.logger import MODULE_COLORS

    missing = KNOWN_TAGS - set(MODULE_COLORS)
    assert not missing, f"MODULE_COLORS thiếu: {sorted(missing)}"
