"""Module tuần tự hóa (Serialization) các thông điệp JSON gửi qua WebSocket.

F-30 — payload phình 2-3× vì gửi LẶP field: `utterance_id` + `utteranceId`,
`original` + `ui_text` + `text`, `stable_text` + `stableText`…

Cách xử lý: **protocol v3 = payload gọn** (chỉ snake_case, một tên cho mỗi giá trị).
- Client khai báo `protocol_version >= 3` (extension mới) ⇒ nhận payload gọn.
- Client cũ (v1/v2) ⇒ vẫn nhận payload đầy đủ như trước, KHÔNG phá vỡ tương thích.

Mọi hàm ở đây nhận thêm `compact: bool = False`; `compact=True` là chế độ v3.
"""

import time
from typing import Any, Dict


def make_pong_msg(timestamp: Any, compact: bool = False) -> Dict[str, Any]:
    """Tạo gói tin phản hồi Ping/Pong."""
    now = time.time()
    if compact:
        return {"type": "pong", "timestamp": timestamp, "server_time": now}
    return {
        "type": "pong",
        "timestamp": timestamp,
        "server_time": now,
        "serverTime": now,
    }


def make_utterance_update_msg(
    utt_id: str,
    text: str,
    translated: str = "",
    is_final: bool = False,
    stable_text: str = "",
    unstable_text: str = "",
    filtered: bool = False,
    compact: bool = False,
) -> Dict[str, Any]:
    """Tạo gói tin cập nhật kết quả nhận dạng giọng nói ASR trực tiếp."""
    if compact:
        # v3: một tên cho mỗi giá trị. `original`/`ui_text` bị bỏ vì trùng `text`.
        return {
            "type": "utterance_update",
            "utterance_id": utt_id,
            "text": text,
            "stable_text": stable_text,
            "unstable_text": unstable_text,
            "translated": translated,
            "is_final": is_final,
            "filtered": filtered,
        }
    return {
        "type": "utterance_update",
        "utterance_id": utt_id,
        "utteranceId": utt_id,
        "original": text,
        "ui_text": text,
        "text": text,
        "stable_text": stable_text,
        "stableText": stable_text,
        "unstable_text": unstable_text,
        "unstableText": unstable_text,
        "translated": translated,
        "is_final": is_final,
        "isFinal": is_final,
        "filtered": filtered,
    }


def make_translation_msg(
    utt_id: str,
    translated: str,
    elapsed_ms: int,
    target_lang: str,
    partial: bool = False,
    compact: bool = False,
) -> Dict[str, Any]:
    """Tạo gói tin phụ đề bản dịch (hoặc bản dịch dần nếu partial=True).

    LƯU Ý QUAN TRỌNG: `status` LUÔN là "ok", kể cả với partial.
    `subtitle-renderer.js:313` chỉ nhận bản dịch khi `status === "ok"`; nếu ta đặt
    status="partial" thì client hiện tại sẽ hiểu là `null` và XOÁ bản dịch đang hiển thị.
    Client phân biệt bản dần bằng cờ `partial` (và renderer cập nhật tại chỗ theo id).
    """
    if compact:
        return {
            "type": "translation",
            "sentence_id": utt_id,
            "translated": translated,
            "status": "ok",
            "partial": bool(partial),
            "translate_time_ms": elapsed_ms,
            "target_lang": target_lang,
        }
    return {
        "type": "translation",
        "sentence_id": utt_id,
        "sentenceId": utt_id,
        "utterance_id": utt_id,
        "utteranceId": utt_id,
        "translated": translated,
        "text": translated,
        "status": "ok",
        "partial": bool(partial),
        "translate_time_ms": elapsed_ms,
        "translateTimeMs": elapsed_ms,
        "target_lang": target_lang,
        "targetLang": target_lang,
    }


def make_tts_binary_frame(
    utt_id: str,
    text: str,
    wav_bytes: bytes,
    duration_sec: float,
    sample_rate: int,
    compact: bool = False,
) -> bytes:
    """P3.1: đóng gói audio TTS thành MỘT binary frame (không base64, không JSON).

    Định dạng (little-endian):
        [0:4]   magic "BTTS"
        [4]     uint8  version = 1
        [5:7]   uint16 header_len
        [7:7+n] JSON header (utf-8)
        [7+n:]  WAV bytes thô
    """
    if compact:
        header = {
            "type": "tts_audio",
            "utterance_id": utt_id,
            "text": text,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "format": "audio/wav",
            "encoding": "binary",
        }
    else:
        header = {
            "type": "tts_audio",
            "utterance_id": utt_id,
            "utteranceId": utt_id,
            "sentence_id": utt_id,
            "sentenceId": utt_id,
            "text": text,
            "duration_sec": duration_sec,
            "durationSec": duration_sec,
            "sample_rate": sample_rate,
            "sampleRate": sample_rate,
            "format": "audio/wav",
            "encoding": "binary",
        }
    import json as _json

    header_bytes = _json.dumps(header, ensure_ascii=False).encode("utf-8")
    out = bytearray()
    out += b"BTTS"
    out.append(1)
    out += len(header_bytes).to_bytes(2, "little")
    out += header_bytes
    out += wav_bytes
    return bytes(out)


def make_model_status_msg(stage: str, state: str, model: str, message: str = "") -> Dict[str, Any]:
    """P1.7/P1.8: thông báo trạng thái nạp model để client hiển thị và không 'đoán'."""
    return {
        "type": "model_status",
        "stage": stage,
        "state": state,
        "model": model,
        "message": message,
    }


def make_tts_audio_msg(
    utt_id: str,
    text: str,
    audio_b64: str,
    duration_sec: float,
    sample_rate: int,
    compact: bool = False,
) -> Dict[str, Any]:
    """Tạo gói tin âm thanh Voice Cloning TTS (đường base64 cho client cũ)."""
    if compact:
        return {
            "type": "tts_audio",
            "utterance_id": utt_id,
            "text": text,
            "audio": audio_b64,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "format": "audio/wav",
        }
    return {
        "type": "tts_audio",
        "utterance_id": utt_id,
        "utteranceId": utt_id,
        "sentence_id": utt_id,
        "sentenceId": utt_id,
        "text": text,
        "audio": audio_b64,
        "duration_sec": duration_sec,
        "durationSec": duration_sec,
        "sample_rate": sample_rate,
        "sampleRate": sample_rate,
        "format": "audio/wav",
    }
