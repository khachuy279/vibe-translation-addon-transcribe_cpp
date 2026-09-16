"""Trình xử lý kết nối WebSocket chính (WebSocket Handler) kết nối toàn bộ Pipeline.

Chu trình hoạt động:
Audio Stream -> VAD -> ASR Streaming (transcribe.cpp) -> Commit Manager -> Translation -> TTS -> WebSocket Client
"""

import asyncio
import json
import threading
import time
from typing import Optional, Dict, Any, List
from fastapi import WebSocket, WebSocketDisconnect

from concurrent.futures import ThreadPoolExecutor

from backend.config import config
from backend.translation.context import TranslationContextTracker
from backend.translation.dedup import TranslationDeduplicator
from backend.translation.engine import GGUFTranslationEngine
from backend.tts import get_tts_engine, TTSDedupState
from backend.ws.connection import SafeWebSocketConnection
from backend.ws.protocol import parse_audio_frame
from backend.ws.serializers import (
    make_pong_msg,
    make_utterance_update_msg,
    make_translation_msg,
    make_tts_audio_msg,
    make_tts_binary_frame,
)
from backend.core.commit_manager import count_content_tokens
from backend.ws.session import SessionState
from backend.core.metrics import metrics_collector
from backend.utils.logger import get_logger

logger = get_logger("ws.handler")

# Dedicated Single-Thread Worker cho VAD: Triệt tiêu tranh chấp lock và giảm 70% CPU
_VAD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vad_worker")

# P1.10: registry phiên đang hoạt động để REST /api/config áp dụng được cho phiên
# đang chạy (trước đây REST chỉ ghi vào `config` toàn cục và bỏ quên session).
_ACTIVE_SESSIONS: Dict[str, "SessionState"] = {}
_ACTIVE_SESSIONS_LOCK = threading.Lock()


def register_session(session: "SessionState") -> None:
    with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSIONS[session.session_id] = session


def unregister_session(session: "SessionState") -> None:
    with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSIONS.pop(session.session_id, None)


def get_active_sessions() -> List["SessionState"]:
    with _ACTIVE_SESSIONS_LOCK:
        return list(_ACTIVE_SESSIONS.values())


def count_active_sessions() -> int:
    with _ACTIVE_SESSIONS_LOCK:
        return len(_ACTIVE_SESSIONS)


async def _maybe_unload_tts_when_idle(session: "SessionState") -> None:
    """A2-3 (Hy3): giải phóng model OmniVoice khi không còn phiên nào cần TTS.

    Model TTS là singleton chiếm vài GB VRAM (float16). Trước đây tắt TTS chỉ đổi cờ cấu
    hình nên VRAM vẫn bị giữ. Ở đây chỉ gọi `unload_model()` khi **mọi** phiên đang mở đều
    đã tắt TTS — nếu còn phiên khác dùng TTS thì bỏ qua để phiên đó không phải nạp lại model
    giữa chừng. Bật lại TTS sẽ `prewarm()` nạp lại như cũ nên không mất chức năng.

    `unload_model()` có `gc.collect()` + `empty_cache()` nên chạy trong thread riêng để
    KHÔNG chặn event loop (nếu chặn, chính nó lại gây ra một latency spike — đúng thứ A2-3
    muốn tránh).
    """
    try:
        for other in get_active_sessions():
            if other is not session and other.config.get("tts_enabled"):
                return
        engine = get_tts_engine()
        if getattr(engine, "_is_loaded", False) or getattr(engine, "model", None) is not None:
            await asyncio.to_thread(engine.unload_model)
            logger.info(
                "Đã giải phóng TTS (không còn phiên nào bật TTS) để trả VRAM.",
                extra={"module_tag": "WS"},
            )
    except Exception as exc:  # noqa: BLE001
        # Giải phóng VRAM là tối ưu, không được làm hỏng luồng cấu hình.
        logger.debug(f"Bỏ qua giải phóng TTS: {exc}", extra={"module_tag": "WS"})


def shutdown_vad_executor(wait: bool = False) -> None:
    """Giải phóng executor chuyên dụng cho VAD khi máy chủ tắt."""
    try:
        _VAD_EXECUTOR.shutdown(wait=wait, cancel_futures=True)
    except Exception:
        pass


# FIX-09: nhịp tối thiểu giữa hai lần ghi gauge độ sâu hàng đợi (khi queue đứng yên > 0).
_QUEUE_GAUGE_MIN_INTERVAL_S = 0.1


def _record_queue_gauges(session: SessionState) -> None:
    """P0.1: ghi độ sâu queue để bottleneck trở thành quan sát được (F-22).

    FIX-09: KHÔNG ghi ở MỌI item nữa. `MetricsCollector` dùng MUTEX TOÀN CỤC cho mọi
    metric, mà hàm này được gọi theo từng item ở vòng stream ASR và trong cả 2 worker ⇒
    mỗi item là 2 lần tranh mutex trong hot path realtime.

    Nay chỉ ghi khi thông tin THỰC SỰ MỚI:
      - độ sâu ĐỔI (tăng hoặc giảm) — đây là tín hiệu backlog chính, hoặc
      - độ sâu tạo high-watermark mới, hoặc
      - queue đứng yên ở mức > 0 và đã quá `_QUEUE_GAUGE_MIN_INTERVAL_S` (nhịp tim, để
        gauge không "cũ" khi hàng đợi kẹt lâu).
    """
    try:
        state = getattr(session, "_queue_gauge_state", None)
        if state is None:
            state = {}
            session._queue_gauge_state = state  # type: ignore[attr-defined]

        now = time.perf_counter()
        for q_name, metric in (
            ("translation_queue", "translation_depth"),
            ("tts_queue", "tts_depth"),
        ):
            q = getattr(session, q_name, None)
            if q is None:
                continue

            depth = q.qsize()
            last_depth, last_ts, high_water = state.get(q_name, (-1, 0.0, -1))
            is_new_high = depth > high_water
            changed = depth != last_depth
            heartbeat = depth > 0 and (now - last_ts) >= _QUEUE_GAUGE_MIN_INTERVAL_S

            if changed or is_new_high or heartbeat:
                metrics_collector.record_gauge("queue", metric, depth)
                state[q_name] = (depth, now, depth if is_new_high else high_water)
    except Exception:
        pass


def _discard_queued(session: SessionState, counter_suffix: str = "cleared") -> int:
    """Bỏ HẾT item còn trong hàng đợi dịch/TTS và cân lại `_unfinished_tasks`.

    F-44b: PHẢI gọi `task_done()` cho mỗi mục bị bỏ. Không gọi thì bộ đếm
    `_unfinished_tasks` của asyncio.Queue không bao giờ về 0 ⇒ mọi `queue.join()` sau đó
    treo vĩnh viễn (một dạng "server treo" rất khó thấy).

    Dùng chung cho 2 đường: tua video (`_reset_session_stream`) và dọn phiên (FIX-13).
    Trả về tổng số item đã bỏ.
    """
    total = 0
    for q_name in ("translation_queue", "tts_queue"):
        q = getattr(session, q_name, None)
        if q is None:
            continue
        dropped = 0
        while True:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break
            except Exception:  # noqa: BLE001
                break
            dropped += 1
            try:
                q.task_done()
            except Exception:  # noqa: BLE001
                pass
        if dropped:
            metrics_collector.increment_counter(f"queue.{q_name}_{counter_suffix}")
        total += dropped
    return total


def _join_texts(a: str, b: str) -> str:
    """Nối hai câu thành một, tránh khoảng trắng thừa."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a:
        return b
    if not b:
        return a
    return f"{a} {b}"


def _coalesce_enqueue(
    queue: "asyncio.Queue",
    item: Dict[str, Any],
    *,
    label: str,
    merged_counter: str,
    dropped_counter: str,
) -> str:
    """Đẩy `item` vào hàng đợi; nếu ĐẦY thì GỘP vào câu cũ nhất thay vì VỨT.

    VÌ SAO (FIX-02/FIX-03 — lỗi mất chất lượng đã xác nhận):
    `translation_queue` và `tts_queue` CHỈ chứa câu FINAL (`is_final`). Bản cũ dùng
    `put_nowait` + `except QueueFull: drop`, nghĩa là khi hàng đợi đầy thì câu đã chốt
    **vĩnh viễn không có bản dịch / không có lồng tiếng** (phụ đề gốc vẫn hiện, kèm dấu
    "..." treo). Đây là mất mát về đúng đắn, không phải tối ưu.

    Cách xử lý (cùng triết lý với `_pending_commits` của ASR: gộp, không vứt):
    - GỘP `text` của câu mới vào câu **MỚI NHẤT đang chờ** — tức hai câu LIỀN KỀ về thời
      gian, nên câu gộp vẫn là một đơn vị ngữ nghĩa hợp lý (gộp với câu CŨ NHẤT sẽ tạo ra
      "câu đầu + câu vừa nói" trong khi các câu giữa vẫn nằm trong hàng đợi ⇒ vô nghĩa);
    - giữ nguyên số phần tử và thứ tự hàng đợi ⇒ trần RAM và FIFO không đổi;
    - ghi lại `merged_utterance_ids` để tầng gửi kết quả còn báo được bản dịch cho những
      câu đã bị gộp (tránh phụ đề treo ở "...").

    Trả về: "queued" | "merged" | "dropped".
    """
    try:
        queue.put_nowait(item)
        return "queued"
    except asyncio.QueueFull:
        pass

    # Đã đầy ⇒ phải gộp. `asyncio.Queue` không có API sửa phần tử, nên rút hết ra rồi đẩy
    # lại. Chi phí O(n) với n <= trần hàng đợi, và chỉ chạy khi đã quá tải.
    # AN TOÀN: đang ở event loop đơn luồng nên không coroutine nào chen giữa được.
    drained: list = []
    while True:
        try:
            drained.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    if not drained:
        # Hàng đợi đầy nhưng không lấy ra được (đua hiếm gặp) — đành chịu.
        metrics_collector.increment_counter(dropped_counter)
        logger.warning(f"Hàng đợi {label} đầy và không gộp được — bỏ câu này.", extra={"module_tag": "WS"})
        return "dropped"

    # `get_nowait()` KHÔNG giảm `_unfinished_tasks`, còn `put_nowait()` thì TĂNG. Cân lại
    # bằng `task_done()` cho đúng số phần tử đã rút, nếu không `queue.join()` trong
    # `drain_queues()` sẽ treo vĩnh viễn.
    for _ in drained:
        try:
            queue.task_done()
        except ValueError:  # pragma: no cover - chỉ xảy ra nếu ai đó đã task_done thừa
            break

    newest = drained[-1]
    merged = dict(newest)
    merged["text"] = _join_texts(newest.get("text", ""), item.get("text", ""))
    # `_queued_at` giữ của câu mới nhất đang chờ: metric queue_wait phản ánh đúng thời điểm
    # nội dung được đưa vào.
    merged["utterance_id"] = item.get("utterance_id") or newest.get("utterance_id", "")
    covered = list(newest.get("merged_utterance_ids") or [newest.get("utterance_id", "")])
    covered.append(item.get("utterance_id", ""))
    merged["merged_utterance_ids"] = [u for u in covered if u]
    merged["_merged_from"] = int(newest.get("_merged_from", 1) or 1) + 1
    drained[-1] = merged

    for element in drained:
        try:
            queue.put_nowait(element)
        except asyncio.QueueFull:  # pragma: no cover - vừa rút ra nên không thể xảy ra
            metrics_collector.increment_counter(dropped_counter)
            logger.warning(f"Hàng đợi {label} đầy trở lại sau khi gộp — bỏ câu này.", extra={"module_tag": "WS"})
            return "dropped"

    metrics_collector.increment_counter(merged_counter)
    logger.warning(
        f"Hàng đợi {label} đầy — đã GỘP {merged['_merged_from']} câu thay vì vứt bỏ. "
        f"Không mất chữ; bản dịch/lồng tiếng sẽ phủ cả {len(merged['merged_utterance_ids'])} câu.",
        extra={"module_tag": "WS"},
    )
    return "merged"


async def handle_ws(ws: WebSocket) -> None:
    """Điểm nhập kết nối WebSocket chính quản lý toàn bộ vòng đời phiên."""
    safe_ws = SafeWebSocketConnection(ws)
    await safe_ws.accept()

    session = SessionState(safe_ws)
    metrics_collector.increment_counter("ws.sessions_connected")
    metrics_collector.record_checkpoint(f"session_start_{session.session_id[:8]}")
    logger.info(f"Session {session.session_id}: Đã kết nối từ client", extra={"module_tag": "WS"})

    session.init_components()
    register_session(session)

    # P3.0: cho client biết server hỗ trợ gì TRƯỚC khi client khai báo phiên bản của nó.
    # Nhờ vậy có thể triển khai backend trước extension mà không phá extension cũ.
    try:
        await session.send_json({
            "type": "connected",
            "protocol_version": int(config.ws.protocol_version),
            "binary_tts": True,          # server có thể gửi TTS dạng binary frame
            "stream_translation": bool(getattr(config.translation, "stream_tokens", False)),
            "session_id": session.session_id[:8],
        })
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Không gửi được gói connected: {e}", extra={"module_tag": "WS"})

    asr_task = asyncio.create_task(_stream_asr_tokens(session), name=f"asr_{session.session_id[:8]}")
    translation_task = asyncio.create_task(_translation_worker(session), name=f"trans_{session.session_id[:8]}")
    tts_task = asyncio.create_task(_tts_worker(session), name=f"tts_{session.session_id[:8]}")
    workers = [asr_task, translation_task, tts_task]

    main_task = asyncio.current_task()
    session_error: Optional[BaseException] = None

    def _on_worker_done(t: asyncio.Task) -> None:
        nonlocal session_error
        if not t.cancelled():
            exc = t.exception()
            if exc is not None:
                session_error = exc
                logger.error(
                    f"Worker {t.get_name()} gặp sự cố bất ngờ: {exc}",
                    exc_info=exc,
                    extra={"module_tag": "WS"},
                )
                if main_task and not main_task.done():
                    main_task.cancel()

    for t in workers:
        t.add_done_callback(_on_worker_done)

    t_cleanup_start = None
    try:
        while True:
            message = await safe_ws.receive()
            msg_type = message.get("type", "")

            if msg_type == "websocket.disconnect":
                logger.info(f"Session {session.session_id}: Client ngắt kết nối", extra={"module_tag": "WS"})
                break
            elif "text" in message:
                await _handle_text_message(session, message["text"])
            elif "bytes" in message:
                await _handle_binary_message(session, message["bytes"])

    except WebSocketDisconnect:
        logger.info(f"Session {session.session_id}: Client ngắt kết nối an toàn", extra={"module_tag": "WS"})
    except asyncio.CancelledError:
        if session_error:
            logger.error(
                f"Session {session.session_id}: Buộc đóng phiên do worker gặp sự cố: {session_error}",
                extra={"module_tag": "WS"},
            )
        else:
            logger.info(f"Session {session.session_id}: Phiên bị hủy", extra={"module_tag": "WS"})
    except Exception as e:
        logger.warning(f"Session {session.session_id}: Kết thúc vòng lặp do lỗi ({e})", exc_info=True, extra={"module_tag": "WS"})
    finally:
        t_cleanup_start = time.perf_counter()

        unregister_session(session)

        # FIX-13: THỨ TỰ DỌN PHIÊN.
        # 1) Dừng nhận việc mới: huỷ + chờ worker kết thúc.
        for t in workers:
            t.remove_done_callback(_on_worker_done)
            t.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # 2) Bỏ hết item còn lại VÀ cân lại `_unfinished_tasks`.
        #    Bản cũ gọi `drain_queues()` SAU khi đã huỷ worker: không còn consumer nào gọi
        #    `task_done()`, nên `join()` chỉ thoát nhờ timeout 0.15 s — vừa vô nghĩa vừa
        #    làm chậm cleanup. Realtime thì nên VỨT NHANH, không phải flush.
        _discard_queued(session, "cleared_on_cleanup")

        await session.cleanup()
        await safe_ws.close()

        metrics_collector.increment_counter("ws.sessions_disconnected")
        metrics_collector.record_checkpoint(f"session_end_{session.session_id[:8]}")
        metrics_collector.record_gauge("ws", "active_sessions", count_active_sessions())
        cleanup_ms = (time.perf_counter() - t_cleanup_start) * 1000.0 if t_cleanup_start else 0.0
        logger.info(f"Session {session.session_id}: Đã đóng và giải phóng tài nguyên hoàn tất ({cleanup_ms:.2f}ms)", extra={"module_tag": "WS"})


async def _handle_text_message(session: SessionState, text: str) -> None:
    """Xử lý thông điệp cấu hình JSON hoặc kiểm tra kết nối ping/pong."""
    try:
        msg = json.loads(text)
        action = msg.get("type") or msg.get("action", "")
        # F-44: `reset_stream` có thể tới dưới 2 dạng:
        #   {"type": "reset_stream", ...}                        (extension mới)
        #   {"type": "set_config", "action": "reset_stream"}      (bản cũ / client khác)
        if msg.get("action") == "reset_stream":
            action = "reset_stream"

        if action in ("set_config", "configure"):
            session.apply_config(msg)
            logger.info(
                f"Session {session.session_id}: Đồng bộ cấu hình từ Extension Popup "
                    f"(vad={session.config.get('vad_engine')}, "
                        f"threshold={session.config.get('vad_threshold')}, "
                        f"silence={session.config.get('silence_duration_ms')}ms, "
                        f"hangover={session.config.get('hangover_ms')}ms, "
                        f"min_words={session.config.get('min_words_to_commit')}, "
                        f"lang: {session.config.get('source_lang')} -> {session.config.get('target_lang')}, "
                        f"tts={session.config.get('tts_enabled')}, "
                        f"voice='{session.config.get('tts_voice')}')",
                extra={"module_tag": "WS"},
            )
            if session.config.get("tts_enabled"):
                prewarm_task = asyncio.create_task(
                    get_tts_engine().prewarm(), name=f"tts_prewarm_{session.session_id[:8]}"
                )
                prewarm_task.add_done_callback(
                    lambda t: logger.error(f"TTS prewarm failed: {t.exception()}", extra={"module_tag": "WS"})
                    if not t.cancelled() and t.exception()
                    else None
                )
            else:
                # A2-3 (Hy3): popup tắt TTS ⇒ trả VRAM của OmniVoice (vài GB float16) thay vì
                # giữ vô ích. Chỉ giải phóng khi KHÔNG còn phiên nào khác đang cần TTS.
                await _maybe_unload_tts_when_idle(session)

        elif action == "reset_stream":
            # F-44: client báo vừa TUA video (hoặc nhảy vị trí) ⇒ xoá audio/trạng thái cũ để
            # phụ đề không trộn nội dung trước-sau khi tua.
            reason = str(msg.get("reason") or "seek")
            await _reset_session_stream(session, reason)

        elif action == "ping":
            pong_payload = make_pong_msg(
                msg.get("timestamp", 0),
                compact=bool(getattr(session, "supports_compact_payload", False)),
            )
            await session.send_json(pong_payload)

    except json.JSONDecodeError:
        pass


def _process_binary_chunk(session: SessionState, data: bytes) -> None:
    """Giải mã tiêu đề khung âm thanh và đưa mẫu PCM vào VADProcessor."""
    metrics_collector.increment_counter("ws.audio_chunks_received")
    pcm_data, capture_ts, chunk_idx = parse_audio_frame(data)

    if pcm_data is None:
        return

    session.record_chunk(chunk_idx)
    if session.vad_processor:
        session.vad_processor.feed_chunk(pcm_data, capture_timestamp=capture_ts)


async def _handle_binary_message(session: SessionState, data: bytes) -> None:
    """Xử lý khung nhị phân âm thanh trên luồng worker VAD chuyên dụng."""
    # FIX-14: `config.ws.max_payload_bytes` trước đây là config CHẾT (không nơi nào đọc).
    # Frame audio thật chỉ ~2 KB, nên một khung vượt trần là bất thường (client lỗi/hỏng
    # giao thức). Chặn ở đây để (a) config có tác dụng thật, (b) không đẩy rác vào VAD/ASR.
    limit = int(getattr(config.ws, "max_payload_bytes", 0) or 0)
    if limit > 0 and len(data) > limit:
        metrics_collector.increment_counter("ws.payload_rejected")
        logger.warning(
            f"Bỏ khung nhị phân {len(data)} byte vượt trần max_payload_bytes={limit}.",
            extra={"module_tag": "WS"},
        )
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_VAD_EXECUTOR, _process_binary_chunk, session, data)


async def _stream_asr_tokens(session: SessionState) -> None:
    """Truyền trực tiếp bản xem trước (live preview) và câu chốt (final) ASR tới Client."""
    engine = session.asr_engine
    if not engine:
        return
    # F-30: client khai báo protocol >= 3 thì nhận payload GỌN (không field trùng lặp).
    compact = bool(getattr(session, "supports_compact_payload", False))

    try:
        async for msg in engine.stream_tokens():
            _record_queue_gauges(session)
            if msg.get("type") == "utterance_update":
                utt_id = msg.get("utterance_id", "")
                text = msg.get("text", "")
                is_final = msg.get("is_final", False)
                stable_text = msg.get("stable_text", "")
                unstable_text = msg.get("unstable_text", "")

                # 1. Nếu là câu chốt final, kiểm tra điều kiện lọc độ dài (min_words_to_commit)
                if is_final and text:
                    val = session.config.get("min_words_to_commit")
                    min_words = int(val if val is not None else config.sentence.min_words_to_commit)
                    token_cnt = count_content_tokens(text)
                    if token_cnt < min_words:
                        logger.info(f"Lọc bỏ câu quá ngắn ({token_cnt} < {min_words} từ): '{text}'", extra={"module_tag": "WS"})
                        metrics_collector.increment_counter("translation.short_words_filtered")
                        # Gửi gói tin filtered=True để Extension xóa bỏ ngay lập tức subtitle draft trên màn hình
                        out_msg = make_utterance_update_msg(
                            utt_id=utt_id,
                            text=text,
                            translated="",
                            is_final=True,
                            filtered=True,
                            compact=compact,
                        )
                        await session.send_json(out_msg)
                        continue

                # 2. Câu hợp lệ hoặc bản xem trước live preview
                out_msg = make_utterance_update_msg(
                    utt_id=utt_id,
                    text=text,
                    translated="..." if is_final else "",
                    is_final=is_final,
                    stable_text=stable_text,
                    unstable_text=unstable_text,
                    filtered=False,
                    compact=compact,
                )

                sent = await session.send_json(out_msg)
                if not sent:
                    return

                # 3. Đưa vào hàng đợi dịch thuật
                if is_final and text and session.translation_queue:
                    target_l = session.config.get("target_lang") or config.translation.target_lang
                    _coalesce_enqueue(
                        session.translation_queue,
                        {
                            "utterance_id": utt_id,
                            "text": text,
                            "source_lang": msg.get("language", session.config.get("source_lang", "auto")),
                            "target_lang": target_l,
                            "_queued_at": time.perf_counter(),
                        },
                        label="Translation",
                        merged_counter="queue.translation_merged",
                        dropped_counter="queue.translation_dropped",
                    )


    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Lỗi worker stream ASR: {e}", exc_info=True, extra={"module_tag": "WS"})


async def _reset_session_stream(session: SessionState, reason: str = "seek") -> None:
    """F-44: reset pipeline ASR/VAD/queue khi video bị tua.

    Xoá audio đang có (kể cả pre-roll), bỏ commit đang chờ, dọn hàng đợi dịch/TTS và thông
    báo cho client xoá phụ đề cũ — nhờ vậy tua video không còn sinh phụ đề trộn nội dung cũ.
    """
    # F-44c: `seeking` và `seeked` bắn liền nhau (~50-100 ms) ⇒ chỉ reset MỘT lần cho mỗi
    # cú tua. Reset hai lần sát nhau làm pipeline khởi động lại giữa chừng (dễ sinh race).
    now = time.monotonic()
    last = getattr(session, "_last_stream_reset_at", 0.0)
    if now - last < 0.5:
        metrics_collector.increment_counter("ws.stream_reset_coalesced")
        return
    session._last_stream_reset_at = now  # type: ignore[attr-defined]

    engine = getattr(session, "asr_engine", None)
    if engine is not None and hasattr(engine, "reset_stream"):
        try:
            await engine.reset_stream(reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Session {session.session_id[:8]}: reset ASR stream lỗi: {exc}",
                           extra={"module_tag": "WS"})

    vad = getattr(session, "vad_processor", None)
    if vad is not None and hasattr(vad, "reset"):
        try:
            vad.reset()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Session {session.session_id[:8]}: reset VAD lỗi: {exc}",
                           extra={"module_tag": "WS"})

    # Hàng đợi dịch/TTS còn câu của đoạn CŨ ⇒ bỏ hết (kết quả sẽ lệch với vị trí mới).
    _discard_queued(session, "cleared_on_seek")

    metrics_collector.increment_counter("ws.stream_reset")
    logger.info(
        f"Session {session.session_id[:8]}: reset stream do {reason} — đã xoá audio cũ, "
        f"hàng đợi dịch/TTS và phụ đề đang hiển thị.",
        extra={"module_tag": "WS"},
    )
    await session.send_json({"type": "stream_reset", "reason": reason, "status": "ok"})


async def _process_translation_item(
    session: SessionState,
    trans_engine: GGUFTranslationEngine,
    ctx_tracker: TranslationContextTracker,
    item: Dict[str, Any],
    dedup: TranslationDeduplicator,
) -> None:
    """Xử lý 1 bản ghi dịch thuật với bộ lọc chống trùng và gửi kết quả về client."""
    text = item.get("text", "")
    utt_id = item.get("utterance_id", "")
    src_lang = item.get("source_lang", "auto")
    tgt_lang = session.config.get("target_lang") or item.get("target_lang", "vi")
    queued_at = item.get("_queued_at", time.perf_counter())

    if not text:
        return

    queue_wait_ms = (time.perf_counter() - queued_at) * 1000.0
    metrics_collector.record_metric("translation", "queue_wait_ms", queue_wait_ms)

    clean_utt = (utt_id or "unknown")[:8]
    # F-30: payload gọn cho client protocol >= 3.
    compact = bool(getattr(session, "supports_compact_payload", False))
    if dedup.is_duplicate(text):
        metrics_collector.increment_counter("translation.dedup_skipped")
        logger.info(f"[utt={clean_utt}] Bỏ qua câu dịch trùng lặp: '{text}'", extra={"module_tag": "WS"})
        return

    start_t = time.monotonic()
    context_str = ctx_tracker.get_context_str() if config.translation.use_context else ""

    translated = ""
    # P3.3: đẩy token dịch dần để bản dịch hiện sớm (K3: < 120ms tới token đầu).
    use_stream = bool(getattr(config.translation, "stream_tokens", False)) and hasattr(
        trans_engine, "translate_stream"
    )
    if use_stream:
        last_sent_t = time.monotonic()
        last_sent_text = ""
        first_token_ms: Optional[float] = None
        min_chars = int(getattr(config.translation, "partial_min_chars", 3) or 0)
        min_interval = float(getattr(config.translation, "partial_min_interval_ms", 120) or 0) / 1000.0
        try:
            async for partial in trans_engine.translate_stream(
                text=text, source_lang=src_lang, target_lang=tgt_lang, context=context_str
            ):
                translated = partial
                if first_token_ms is None:
                    first_token_ms = (time.monotonic() - start_t) * 1000.0
                    metrics_collector.record_metric("translation", "first_token_ms", first_token_ms)
                now = time.monotonic()
                # Throttle kép: đủ ký tự MỚI và đủ thời gian. Không gửi tiền tố rác
                # (ví dụ một khoảng trắng) và không ngập WebSocket.
                if (
                    partial != last_sent_text
                    and len(partial) >= min_chars
                    and (now - last_sent_t) >= min_interval
                ):
                    last_sent_t = now
                    last_sent_text = partial
                    await session.send_json(
                        make_translation_msg(
                            utt_id=utt_id,
                            translated=partial,
                            elapsed_ms=int((now - start_t) * 1000),
                            target_lang=tgt_lang,
                            partial=True,
                            compact=compact,
                        )
                    )
                    metrics_collector.increment_counter("translation.partial_sent")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Streaming thất bại, fallback sang dịch một lần: {e}", extra={"module_tag": "WS"})

    if not translated:
        res = await trans_engine.translate(
            text=text,
            source_lang=src_lang,
            target_lang=tgt_lang,
            context=context_str,
        )
        translated = res.get("translated_text", text)

    elapsed_ms = int((time.monotonic() - start_t) * 1000)
    metrics_collector.record_metric("translation", "infer_ms", float(elapsed_ms))

    logger.info(f"[utt={clean_utt}] ({src_lang} -> {tgt_lang} in {elapsed_ms}ms): '{translated}'", extra={"module_tag": "WS"})
    ctx_tracker.add(text, translated)

    # 1. Gói tin translation chuyên biệt
    trans_msg = make_translation_msg(
        utt_id=utt_id,
        translated=translated,
        elapsed_ms=elapsed_ms,
        target_lang=tgt_lang,
        compact=compact,
    )

    # 2. Gói tin utterance_update cập nhật trạng thái chốt kèm bản dịch
    update_msg = make_utterance_update_msg(
        utt_id=utt_id,
        text=text,
        translated=translated,
        is_final=True,
        compact=compact,
    )

    sent1 = await session.send_json(trans_msg)
    sent2 = await session.send_json(update_msg)
    if not sent1 or not sent2:
        return

    # FIX-02: nếu item này là kết quả của việc GỘP nhiều câu final (hàng đợi đầy), phát
    # bản dịch cho MỌI câu được phủ. Nếu không, các câu bị gộp sẽ treo vĩnh viễn ở dấu "..."
    # vì chúng đã được gửi đi với placeholder `translated="..."`.
    for covered_id in item.get("merged_utterance_ids") or []:
        if not covered_id or covered_id == utt_id:
            continue
        await session.send_json(
            make_translation_msg(
                utt_id=covered_id,
                translated=translated,
                elapsed_ms=elapsed_ms,
                target_lang=tgt_lang,
                compact=compact,
            )
        )

    # Đo đạc End-to-End Latency từ lúc ASR commit đến khi Client nhận phụ đề
    e2e_sub_ms = (time.perf_counter() - queued_at) * 1000.0
    metrics_collector.record_metric("pipeline", "e2e_asr_to_sub_ms", e2e_sub_ms)
    metrics_collector.increment_counter("pipeline.subtitles_delivered")

    # 3. Đưa vào hàng đợi TTS nếu người dùng bật chế độ lồng tiếng
    if session.config.get("tts_enabled"):
        if session.tts_queue and translated:
            _coalesce_enqueue(
                session.tts_queue,
                {
                    "utterance_id": utt_id,
                    "text": translated,
                    "voice": session.config.get("tts_voice"),
                    "speed": float(session.config.get("tts_speed", 1.0)),
                    "_queued_at": time.perf_counter(),
                },
                label="TTS",
                merged_counter="queue.tts_merged",
                dropped_counter="queue.tts_dropped",
            )


async def _translation_worker(session: SessionState) -> None:
    """Worker tuần tự dịch thuật GGUF đảm bảo không tắc nghẽn queue."""
    ctx_tracker = TranslationContextTracker(window_size=config.translation.context_window)
    dedup = TranslationDeduplicator()

    while True:
        try:
            # P1.7: lấy singleton MỖI lần để không giữ tham chiếu cũ nếu bị thay.
            trans_engine = GGUFTranslationEngine.get_instance()
            item = await session.translation_queue.get()
            try:
                _record_queue_gauges(session)
                await _process_translation_item(session, trans_engine, ctx_tracker, item, dedup)
            finally:
                session.translation_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Lỗi Translation worker: {e}", exc_info=True, extra={"module_tag": "WS"})
            await asyncio.sleep(0.05)


async def _process_tts_item(
    session: SessionState,
    tts: Any,
    item: Dict[str, Any],
    dedup_state: TTSDedupState,
) -> None:
    """Xử lý 1 bản ghi TTS, thực hiện Voice Cloning và gửi audio base64 về Client."""
    text = item.get("text", "")
    utt_id = item.get("utterance_id", "")
    voice = item.get("voice") or session.config.get("tts_voice")
    speed = float(item.get("speed") or session.config.get("tts_speed", 1.0))
    queued_at = item.get("_queued_at", time.perf_counter())

    if not text:
        return

    tts_queue_wait_ms = (time.perf_counter() - queued_at) * 1000.0
    metrics_collector.record_metric("tts", "queue_wait_ms", tts_queue_wait_ms)

    if dedup_state.is_duplicate(text):
        metrics_collector.increment_counter("tts.dedup_skipped")
        logger.info(f"Bỏ qua câu phát âm trùng lặp: '{text}'", extra={"module_tag": "WS"})
        return

    try:
        t_tts_start = time.perf_counter()
        # P3.1: nếu client khai báo protocol >= 2 thì gửi WAV thô dạng binary frame
        # (bỏ base64 +33%, bỏ vòng lặp per-byte trên main thread của client).
        use_binary = bool(getattr(session, "supports_binary_tts", False)) and hasattr(
            tts, "synthesize_clone_bytes"
        )
        if use_binary:
            wav_bytes, duration_sec = await tts.synthesize_clone_bytes(
                text=text, voice_id=voice, speed=speed
            )
        else:
            audio_b64, duration_sec = await tts.synthesize_clone(
                text=text,
                voice_id=voice,
                speed=speed,
            )
            wav_bytes = None
        synthesis_ms = (time.perf_counter() - t_tts_start) * 1000.0
        metrics_collector.record_metric("tts", "synthesis_ms", synthesis_ms)
        tts_rtf = (synthesis_ms / 1000.0) / max(0.001, duration_sec)
        metrics_collector.record_metric("tts", "rtf", tts_rtf)
        metrics_collector.increment_counter("tts.synthesized_utterances")

        if use_binary and wav_bytes:
            frame = make_tts_binary_frame(
                utt_id=utt_id,
                text=text,
                wav_bytes=wav_bytes,
                duration_sec=duration_sec,
                sample_rate=tts.sample_rate,
                compact=bool(getattr(session, "supports_compact_payload", False)),
            )
            sent = await session.connection.send_bytes(frame)
            metrics_collector.record_metric("tts", "payload_bytes", float(len(frame)))
            metrics_collector.increment_counter("tts.sent_binary")
            if sent:
                e2e_tts_ms = (time.perf_counter() - queued_at) * 1000.0
                metrics_collector.record_metric("pipeline", "e2e_sub_to_tts_ms", e2e_tts_ms)
                logger.info(
                    f"{duration_sec:.2f}s ({len(frame)} bytes) cho utt "
                        f"'{utt_id}'(synth={synthesis_ms:.1f}ms, RTF={tts_rtf:.2f})", extra={"module_tag": "WS"}
                )
        elif audio_b64:
            out_msg = make_tts_audio_msg(
                utt_id=utt_id,
                text=text,
                audio_b64=audio_b64,
                duration_sec=duration_sec,
                sample_rate=tts.sample_rate,
                compact=bool(getattr(session, "supports_compact_payload", False)),
            )
            sent = await session.send_json(out_msg)
            if sent:
                e2e_tts_ms = (time.perf_counter() - queued_at) * 1000.0
                metrics_collector.record_metric("pipeline", "e2e_sub_to_tts_ms", e2e_tts_ms)
                logger.info(f"Gửi {duration_sec:.2f}s audio về client cho utt '{utt_id}'(synth={synthesis_ms:.1f}ms, RTF={tts_rtf:.2f})", extra={"module_tag": "WS"})
    except Exception as e:
        logger.error(f"Lỗi TTS synthesis: {e}", exc_info=True, extra={"module_tag": "WS"})


async def _tts_worker(session: SessionState) -> None:
    """Worker tuần tự OmniVoice TTS đảm bảo an toàn VRAM và task_done."""
    tts: Optional[Any] = None
    dedup_state = TTSDedupState()

    while True:
        try:
            item = await session.tts_queue.get()
            try:
                _record_queue_gauges(session)
                if not session.config.get("tts_enabled"):
                    metrics_collector.increment_counter("tts.skipped_disabled")
                    continue
                if tts is None:
                    tts = get_tts_engine()
                await _process_tts_item(session, tts, item, dedup_state)
            finally:
                session.tts_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Lỗi TTS worker loop: {e}", exc_info=True, extra={"module_tag": "WS"})
            await asyncio.sleep(0.05)


__all__ = [
    "handle_ws",
    "shutdown_vad_executor",
    "register_session",
    "unregister_session",
    "get_active_sessions",
    "count_active_sessions",
]
