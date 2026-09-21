"""Test tầng A — F-43: quy ước logging được giữ đồng bộ.

Quy ước nằm ở đầu `backend/utils/logger.py`. Test này **phân tích AST** mã nguồn backend
(chính xác hơn quét chuỗi, vì chỉ xét phần CHỮ của thông điệp, bỏ qua biểu thức trong
f-string) và chốt 5 điều:

1. Mọi `logger.<level>(...)` đều có `extra={"module_tag": TAG}`.
2. TAG thuộc danh sách chuẩn (không có tag tự phát như `CONFIG`, `WS.HANDLER`).
3. Phần chữ của thông điệp KHÔNG lặp lại tag module trong ngoặc vuông.
4. Trong thông điệp chỉ có 3 loại token ngoặc vuông: `[utt=…]`, `[STARTUP]`, `[SHUTDOWN]`.
5. Không có emoji trong thông điệp.

TAG được **suy diễn tĩnh** (literal, hằng cấp module, biến gán, `x.upper()/lower()/strip()`,
nối chuỗi, `a if c else b`, f-string tĩnh và tham số có default literal) — nhờ vậy tag ĐỘNG
hợp lệ như `tag = stage.upper()` trong `utils/model_download.py` không còn bị báo sai, mà
quy ước vẫn được kiểm tra thật (xem `test_checker_van_bat_duoc_tag_thieu_va_tag_sai`).

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


# ── Suy diễn chuỗi TĨNH (constant propagation nhỏ) ───────────────────────────
# VÌ SAO CẦN: `utils/model_download.py` dùng tag ĐỘNG theo giai đoạn tải:
#     def _monitor_progress(..., stage: str = "TRANSLATE", ...):
#         tag = stage.upper()
#         logger.info(..., extra={"module_tag": tag})
# Runtime tag luôn là "TRANSLATE" hoặc "ASR" (caller truyền literal), nhưng bản checker
# chỉ soi literal/hằng cấp module nên báo 4 lời gọi "thiếu module_tag" — SAI. Thay vì hạ
# chuẩn (chấp nhận mọi biến), ta suy ra tập giá trị chuỗi có thể có: literal, biến đã gán,
# `x.upper()/lower()/strip()`, nối chuỗi, và `a if c else b`. Nhờ vậy quy ước vẫn được kiểm
# tra THẬT (tag phải suy ra được và phải thuộc danh sách chuẩn) mà không còn dương tính giả.

_MAX_RESOLVE_DEPTH = 8


def _resolve_str_values(expr: ast.AST, scope: dict, depth: int = 0):
    """Tập giá trị chuỗi có thể có của `expr`, hoặc `None` nếu không suy ra được."""
    if depth > _MAX_RESOLVE_DEPTH:
        return None

    if isinstance(expr, ast.Constant):
        return {expr.value} if isinstance(expr.value, str) else None

    if isinstance(expr, ast.Name):
        values = scope.get(expr.id)
        if not values:
            return None
        # Phòng thủ: scope lẽ ra luôn là TẬP giá trị; nếu ai đó nhét thẳng chuỗi vào thì
        # `set("VAD")` sẽ tách thành từng ký tự — đó là lỗi đã gặp khi viết checker này.
        if isinstance(values, str):
            return {values}
        return set(values)

    if isinstance(expr, ast.IfExp):
        left = _resolve_str_values(expr.body, scope, depth + 1)
        right = _resolve_str_values(expr.orelse, scope, depth + 1)
        if left is None or right is None:
            return None
        return left | right

    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left = _resolve_str_values(expr.left, scope, depth + 1)
        right = _resolve_str_values(expr.right, scope, depth + 1)
        if left is None or right is None:
            return None
        return {a + b for a in left for b in right}

    if isinstance(expr, ast.JoinedStr):
        # f-string: chỉ nhận khi MỌI phần đều tĩnh.
        parts = []
        for value in expr.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append({value.value})
            else:
                resolved = _resolve_str_values(value, scope, depth + 1)
                if resolved is None:
                    return None
                parts.append(resolved)
        combined = {""}
        for part in parts:
            combined = {a + b for a in combined for b in part}
            if len(combined) > 64:
                return None
        return combined

    if isinstance(expr, ast.Call):
        func = expr.func
        if isinstance(func, ast.Attribute) and not expr.args and not expr.keywords:
            base = _resolve_str_values(func.value, scope, depth + 1)
            if base is None:
                return None
            if func.attr == "upper":
                return {v.upper() for v in base}
            if func.attr == "lower":
                return {v.lower() for v in base}
            if func.attr == "strip":
                return {v.strip() for v in base}
            return None
    return None


def _build_scope(func: ast.AST, module_scope: dict) -> dict:
    """Scope trong 1 hàm: tham số có default literal + biến gán bằng biểu thức suy được."""
    scope: dict = dict(module_scope)
    args = getattr(func, "args", None)
    if args is not None:
        defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        positional = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        # Ghép default với tham số CUỐI (đúng ngữ nghĩa Python cho positional).
        offset = len(positional) - len(defaults)
        for idx, default in enumerate(defaults):
            param = positional[offset + idx]
            resolved = _resolve_str_values(default, scope)
            if resolved:
                scope[param.arg] = resolved

    for stmt in ast.walk(func):
        target = None
        value = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            target, value = stmt.target.id, stmt.value
        if target is None:
            continue
        resolved = _resolve_str_values(value, scope)
        if resolved:
            scope[target] = resolved
    return scope


def _iter_logger_calls(node: ast.AST):
    """Duyệt cây theo ĐÚNG thứ tự tài liệu, KHÔNG đi vào hàm/lớp lồng nhau.

    (Mỗi hàm có scope riêng nên được xử lý ở lượt riêng; duyệt theo thứ tự giúp thông báo
    lỗi và test tổng hợp dễ đọc.)
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield child
        yield from _iter_logger_calls(child)


def _is_logger_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    return (
        isinstance(fn, ast.Attribute)
        and fn.attr in LEVELS
        and isinstance(fn.value, ast.Name)
        and fn.value.id == "logger"
    )


def _logger_calls():
    """Sinh (file, dòng, node_call, scope_chuỗi) cho mọi `logger.<level>(...)`."""
    for path in _backend_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Scope chuỗi: mọi tên đều ánh xạ tới TẬP giá trị có thể có.
        module_scope = {name: {value} for name, value in _module_string_constants(tree).items()}

        for node in _iter_logger_calls(tree):
            if _is_logger_call(node):
                yield path, node.lineno, node, module_scope

        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            scope = _build_scope(func, module_scope)
            for node in _iter_logger_calls(func):
                if _is_logger_call(node):
                    yield path, node.lineno, node, scope


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


def _module_tag(call: ast.Call, scope=None):
    """Tập giá trị có thể có của `module_tag` (None = thiếu/không suy ra được).

    Chấp nhận literal, hằng chuỗi cấp module, VÀ biểu thức chuỗi suy diễn được
    (`stage.upper()`, `_TAG`, `"A" if x else "B"`, f-string tĩnh…).
    """
    for kw in call.keywords:
        if kw.arg == "extra" and isinstance(kw.value, ast.Dict):
            for k, v in zip(kw.value.keys, kw.value.values):
                if isinstance(k, ast.Constant) and k.value == "module_tag":
                    values = _resolve_str_values(v, scope or {})
                    if values:
                        return frozenset(values)
    return None


def test_every_logger_call_has_module_tag():
    missing = [
        f"{p.relative_to(ROOT)}:{line}"
        for p, line, call, scope in _logger_calls()
        if not _module_tag(call, scope)
    ]
    assert not missing, f"{len(missing)} lời gọi logger thiếu module_tag: {missing[:10]}"


def test_module_tags_are_canonical():
    bad = set()
    for _p, _line, call, scope in _logger_calls():
        values = _module_tag(call, scope)
        if values:
            bad |= {v for v in values if v not in KNOWN_TAGS}
    assert not bad, f"module_tag ngoài danh sách chuẩn: {sorted(bad)}"


def test_message_does_not_repeat_its_own_tag():
    offenders = []
    for path, line, call, scope in _logger_calls():
        values = _module_tag(call, scope) or set()
        text = _literal_text(call)
        for tag in values:
            if f"[{tag}]" in text:
                offenders.append(f"{path.relative_to(ROOT)}:{line} (tag {tag})")
                break
    assert not offenders, f"thông điệp lặp lại tag: {offenders[:10]}"


def test_only_allowed_bracket_tokens_in_messages():
    offenders = []
    for path, line, call, _scope in _logger_calls():
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
        for p, line, call, _scope in _logger_calls()
        if EMOJI_RE.search(_literal_text(call))
    ]
    assert not offenders, f"còn emoji trong log: {offenders[:10]}"


def _snippet_calls(src: str):
    """Chạy đúng đường phân tích của test trên một đoạn mã tổng hợp."""
    tree = ast.parse(src)
    module_scope = {name: {value} for name, value in _module_string_constants(tree).items()}
    calls = []
    for node in _iter_logger_calls(tree):
        if _is_logger_call(node):
            calls.append(_module_tag(node, module_scope))
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        scope = _build_scope(func, module_scope)
        for node in _iter_logger_calls(func):
            if _is_logger_call(node):
                calls.append(_module_tag(node, scope))
    return calls


def test_checker_van_bat_duoc_tag_thieu_va_tag_sai():
    """Chống "hạ chuẩn": suy diễn chuỗi KHÔNG được biến lỗi thật thành im lặng.

    Đây là bài test bảo vệ chính bản sửa của `test_20`: nới checker để hiểu tag động
    (`stage.upper()`) mà vẫn phải bắt được (a) thiếu `extra`, (b) tag không suy ra được,
    (c) tag không nằm trong danh sách chuẩn.
    """
    src = '''
import logging
logger = logging.getLogger("x")
_TAG = "VAD"
_BAD = "WS.HANDLER"

def f(stage: str = "ASR", unknown=None):
    tag = stage.upper()
    logger.info("a", extra={"module_tag": tag})              # OK: {"ASR"}
    logger.info("b", extra={"module_tag": _TAG})             # OK: {"VAD"}
    logger.info("c", extra={"module_tag": stage.lower()})    # OK: {"asr"} -> sẽ bị bắt ở test canonical
    logger.info("d", extra={"module_tag": unknown()})        # THIẾU: không suy ra được
    logger.info("e")                                         # THIẾU: không có extra
    logger.info("f", extra={"module_tag": _BAD})             # SAI: ngoài danh sách chuẩn
    logger.info("g", extra={"module_tag": "VAD" if stage == "ASR" else "TRANSLATE"})  # OK: 2 giá trị
'''
    calls = _snippet_calls(src)
    assert calls == [
        frozenset({"ASR"}),
        frozenset({"VAD"}),
        frozenset({"asr"}),
        None,
        None,
        frozenset({"WS.HANDLER"}),
        frozenset({"VAD", "TRANSLATE"}),
    ], calls

    # (b) tag không suy ra được / thiếu extra ⇒ bị coi là thiếu
    assert calls[3] is None and calls[4] is None
    # (c) tag ngoài danh sách chuẩn vẫn bị test canonical bắt
    assert frozenset({"WS.HANDLER"}) & (set(calls[5]) - KNOWN_TAGS)
    assert frozenset({"asr"}) - KNOWN_TAGS


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
