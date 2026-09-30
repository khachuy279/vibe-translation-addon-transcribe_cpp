// Subtitle Renderer Engine — 3-Layer Anchor Focus Architecture
// Layer 1: [Lịch sử cũ + Bản dịch] — Cuộn lên trên (dimmed, font nhỏ hơn một chút)
// Layer 2: [Phụ đề mới nhất + Bản dịch] — CỐ ĐỊNH LÀM TRUNG TÂM TIÊU ĐIỂM (100% focus)
// Layer 3: [Phụ đề đang nhận diện / chưa chốt] — Luôn ở dưới cùng (live typing)

class SubtitleRenderer {
  constructor(container) {
    this.container = container;
    this.mode = "bottom_bar"; // "bottom_bar" | "floating"

    // Completed translated sentences pool (CHỈ CHỨA CÁC CÂU ĐÃ CÓ BẢN DỊCH HOÀN CHỈNH)
    this.completedSentences = [];
    this.maxLines = 2; // Mặc định 2 câu: 1 câu lịch sử Tầng 1 + 1 câu mới nhất Tầng 2
    this.maxHistorySentences = 1; // Số câu lịch sử tối đa ở Layer 1 (= maxLines - 1)

    // Lưu trữ câu gốc (utterance_id -> { id, originalText })
    this.utteranceStore = new Map();

    // Quản lý vòng đời hiển thị theo độ dài câu dịch & fade-out
    this.lifecycleTimer = null;
    this.fadeDurationMs = 500; // 0.5s mờ dần trước khi biến mất

    // Safety fallback auto-clear timeout khi hoàn toàn không có tương tác (30s)
    this.autoClearTimeoutMs = 30000;
    this.autoClearTimer = null;

    // Current live draft (Layer 3) — Đang nhận diện realtime (CHƯA chốt VAD)
    this.currentDraft = null; // { id, originalText, isFinal: false }

    // Pending focus (Layer 2) — Câu đã chốt VAD, đang chờ bản dịch
    // Xuất hiện ngay ở Layer 2 với indicator ・・・, bản dịch điền vào sau
    this.pendingFocus = null; // { id, originalText }

    // Cache state to avoid DOM thrashing
    this.currentFocusId = null;
    this._historyFingerprint = "";

    // DOM Elements for 3 distinct layers
    this._buildDOM();

    // Tự động tải cài đặt maxLines từ chrome.storage
    if (typeof chrome !== "undefined" && chrome?.storage?.local) {
      chrome.storage.local.get("bs_settings", (data) => {
        if (data?.bs_settings) {
          this.applySettings(data.bs_settings);
        }
      });
    }
  }

  _resetAutoClearTimer() {
    if (this.autoClearTimer) {
      clearTimeout(this.autoClearTimer);
      this.autoClearTimer = null;
    }
    if (this.autoClearTimeoutMs > 0) {
      this.autoClearTimer = setTimeout(() => {
        this.clear();
      }, this.autoClearTimeoutMs);
    }
  }

  // Tính thời gian hiển thị (ms) dựa theo độ dài câu dịch (tối thiểu 3.0s, tối đa 10.0s)
  _calculateDuration(translatedText, originalText = "") {
    const text = (translatedText || originalText || "").trim();
    if (!text) return 3000;
    // Base 2500ms, cộng 60ms cho mỗi ký tự (~16 ký tự/giây theo chuẩn đọc phụ đề)
    const baseMs = 2500;
    const msPerChar = 60;
    const duration = baseMs + text.length * msPerChar;
    return Math.max(3000, Math.min(10000, duration));
  }

  // Ràng buộc thời gian: Nếu Tầng 1 (câu cũ) dài hơn Tầng 2 (câu mới nhất),
  // câu Tầng 2 sẽ không mất trước mà được đồng bộ mất cùng lúc với Tầng 1.
  _enforceLayerTimingConstraints() {
    const len = this.completedSentences.length;
    if (len >= 2) {
      const focusSentence = this.completedSentences[len - 1]; // Tầng 2 (câu mới nhất)
      const historySentences = this.completedSentences.slice(0, len - 1); // Tầng 1 (các câu cũ)
      const maxHistoryExpireAt = Math.max(...historySentences.map((s) => s.expireAt));

      if (maxHistoryExpireAt > focusSentence.expireAt) {
        focusSentence.expireAt = maxHistoryExpireAt;
      }
    }
  }

  _applyFadeOutToDOM(sentenceId) {
    if (!this.container) return;
    const el = this.container.querySelector(`[data-sentence-id="${sentenceId}"]`);
    if (el) {
      el.classList.add("bs-fading-out");
    }
  }

  _removeFadeOutFromDOM(sentenceId) {
    if (!this.container) return;
    const el = this.container.querySelector(`[data-sentence-id="${sentenceId}"]`);
    if (el) {
      el.classList.remove("bs-fading-out");
    }
  }

  _scheduleNextLifecycleTick() {
    if (this.lifecycleTimer) {
      clearTimeout(this.lifecycleTimer);
      this.lifecycleTimer = null;
    }

    if (this.completedSentences.length === 0) {
      return;
    }

    const now = Date.now();
    let minWaitMs = Infinity;

    for (const item of this.completedSentences) {
      if (item.isFadingOut) {
        // Đang trong giai đoạn fade-out 0.5s: đợi đủ 500ms để xoá hoàn toàn
        const remainingFade = (item.fadeStartAt + this.fadeDurationMs) - now;
        if (remainingFade < minWaitMs) {
          minWaitMs = remainingFade;
        }
      } else {
        // Chưa fade-out: đợi đến expireAt để bắt đầu mờ dần
        const remainingDisplay = item.expireAt - now;
        if (remainingDisplay < minWaitMs) {
          minWaitMs = remainingDisplay;
        }
      }
    }

    if (minWaitMs === Infinity) return;

    // Giới hạn delay từ 10ms đến 30000ms
    const delay = Math.max(10, Math.min(minWaitMs, 30000));
    this.lifecycleTimer = setTimeout(() => {
      this._onLifecycleTick();
    }, delay);
  }

  _onLifecycleTick() {
    const now = Date.now();
    let stateChanged = false;

    // 1. Kiểm tra câu đã fade-out xong (sau 0.5s) -> xoá khỏi danh sách completedSentences
    const remaining = [];
    for (const item of this.completedSentences) {
      if (item.isFadingOut && (now - item.fadeStartAt >= this.fadeDurationMs)) {
        stateChanged = true;
      } else {
        remaining.push(item);
      }
    }
    this.completedSentences = remaining;

    // 2. Kiểm tra câu đã hết thời gian hiển thị -> kích hoạt hiệu ứng mờ dần (fade-out 0.5s)
    for (const item of this.completedSentences) {
      if (!item.isFadingOut && now >= item.expireAt) {
        item.isFadingOut = true;
        item.fadeStartAt = now;
        this._applyFadeOutToDOM(item.id);
        stateChanged = true;
      }
    }

    // 3. Nếu có thay đổi danh sách phần tử -> render lại DOM
    if (stateChanged) {
      this._renderAll();
    }

    // 4. Lên lịch cho lần tick tiếp theo
    this._scheduleNextLifecycleTick();
  }

  _buildDOM() {
    this.container.textContent = "";

    // TẦNG 1: Lịch sử cũ (Cuộn lên trên)
    this.historyLayer = document.createElement("div");
    this.historyLayer.className = "bs-history-layer";

    // TẦNG 2: Tiêu điểm trung tâm cố định (Anchor Focus)
    this.focusLayer = document.createElement("div");
    this.focusLayer.className = "bs-focus-layer";

    // TẦNG 3: Liveview đang nhận diện / chưa chốt VAD (Cố định ở dưới cùng)
    this.liveLayer = document.createElement("div");
    this.liveLayer.className = "bs-live-layer";

    this.container.appendChild(this.historyLayer);
    this.container.appendChild(this.focusLayer);
    this.container.appendChild(this.liveLayer);
  }

  setMode(mode) {
    this.mode = mode;
    this.container.className = "bs-content-area";
  }

  setMaxLines(maxLines) {
    const val = parseInt(maxLines, 10);
    if (!isNaN(val) && val >= 1) {
      this.maxLines = val;
      this.maxHistorySentences = Math.max(0, val - 1); // Layer 1 chứa (maxLines - 1) câu cũ
      while (this.completedSentences.length > this.maxLines) {
        this.completedSentences.shift();
      }
      this._renderAll();
    }
  }

  applySettings(settings) {
    if (!settings) return;
    if (settings.maxLines !== undefined) {
      this.setMaxLines(settings.maxLines);
    }
    // "Tắt chạy chữ": chỉ hiện bản dịch MỘT LẦN khi đã có bản dịch hoàn chỉnh.
    // Mặc định BẬT (người dùng đã yêu cầu tắt chạy chữ); đặt false để quay lại hành vi cũ.
    if (settings.showTranslationOnce !== undefined) {
      this.showTranslationOnce = !!settings.showTranslationOnce;
    }
  }

  // ── Utterance Update (Realtime Live Stream & Finalization) ─────────────

  onUtteranceUpdate(payload) {
    if (!payload) return;
    // v3 (F-30): payload GỌN — chỉ đọc tên chuẩn snake_case, KHÔNG fallback sang alias.
    const utteranceId = payload.utterance_id || payload.id;

    // Nếu bị lọc hoặc xóa
    if (payload.filtered || payload.is_deleted || payload.deleted) {
      if (this.currentDraft && this.currentDraft.id === utteranceId) {
        this.currentDraft = null;
      }
      if (this.pendingFocus && this.pendingFocus.id === utteranceId) {
        this.pendingFocus = null;
      }
      this.utteranceStore.delete(utteranceId);
      this.completedSentences = this.completedSentences.filter((s) => s.id !== utteranceId);
      this._renderAll();
      return;
    }

    const text = payload.text || "";
    if (!text || !text.trim()) {
      if (this.currentDraft && this.currentDraft.id === utteranceId) {
        this.currentDraft = null;
        this._renderAll();
      }
      return;
    }

    this._resetAutoClearTimer();

    const isFinal = payload.is_final || false;

    // Lưu vào utteranceStore để không bao giờ bị mất câu gốc khi dịch hoàn thành
    this.utteranceStore.set(utteranceId, {
      id: utteranceId,
      originalText: text,
      isFinal: isFinal,
    });

    // Giới hạn kích thước utteranceStore tránh phình bộ nhớ
    if (this.utteranceStore.size > 50) {
      const oldestKey = this.utteranceStore.keys().next().value;
      this.utteranceStore.delete(oldestKey);
    }

    // 1. Nếu câu này đã nằm trong completedSentences (đã có bản dịch) -> cập nhật text
    let existingCompleted = this.completedSentences.find((s) => s.id === utteranceId);
    if (existingCompleted) {
      existingCompleted.originalText = text;
      this._renderAll();
      return;
    }

    // 2. Nếu câu này đang ở pendingFocus (chờ dịch ở Layer 2) -> cập nhật text
    if (this.pendingFocus && this.pendingFocus.id === utteranceId) {
      this.pendingFocus.originalText = text;
      this._renderAll();
      return;
    }

    if (isFinal) {
      // 3a. Câu vừa CHỐT VAD -> đẩy lên Layer 2 ngay (pendingFocus), xóa Layer 3
      // Nếu đã có pendingFocus cũ (bị timeout không có dịch) -> đẩy xuống completedSentences không có dịch
      if (this.pendingFocus) {
        this._promotePendingToCompleted(null);
      }
      this.pendingFocus = { id: utteranceId, originalText: text };
      this.currentDraft = null;
    } else {
      // 3b. Câu đang nhận diện dở (chưa chốt) -> Layer 3
      this.currentDraft = { id: utteranceId, originalText: text, isFinal: false };
    }

    this._renderAll();
  }

  // ── Translation Update ──────────────────────────────────────────

  onTranslation(sentenceId, translatedText, status) {
    let id = sentenceId;
    let text = translatedText;
    let st = status || "ok";
    let payload = null;

    if (typeof sentenceId === "object" && sentenceId !== null) {
      payload = sentenceId;
      // v3 (F-30): chỉ đọc tên chuẩn snake_case.
      id = sentenceId.sentence_id || sentenceId.utterance_id;
      text = sentenceId.translated || translatedText;
      st = sentenceId.status || status || "ok";
    }

    if (!id) return;

    // "Tắt chạy chữ" (mặc định): BỎ QUA các mảnh dịch dở (`partial`) và chỉ hiện bản dịch
    // MỘT LẦN khi backend gửi bản hoàn chỉnh. Có thể tắt bằng `showTranslationOnce=false`.
    const policy = (typeof BSSubtitlePolicy !== "undefined" && BSSubtitlePolicy)
      || (typeof globalThis !== "undefined" && globalThis.BSSubtitlePolicy)
      || null;
    const showOnce = this.showTranslationOnce !== undefined ? this.showTranslationOnce : true;
    if (policy && typeof policy.shouldApplyTranslation === "function") {
      const candidate = payload || { status: st, partial: false };
      if (!policy.shouldApplyTranslation(candidate, { showOnce })) {
        return;
      }
    }

    const cleanTranslation = (st === "ok" && text) ? text : null;

    // Lấy câu gốc đã lưu trong utteranceStore
    const stored = this.utteranceStore.get(id) || {};

    // 1. Nếu câu đang ở pendingFocus (Layer 2, chờ dịch) -> điền bản dịch vào, chuyển sang completedSentences
    if (this.pendingFocus && this.pendingFocus.id === id) {
      this._promotePendingToCompleted(cleanTranslation);
      this._renderAll();
      return;
    }

    const originalText = stored.originalText || "";

    // 2. Kiểm tra xem câu đã có trong completedSentences chưa
    const existing = this.completedSentences.find((s) => s.id === id);
    if (existing) {
      existing.translatedText = cleanTranslation;
      if (!existing.originalText && originalText) {
        existing.originalText = originalText;
      }
      existing.isFinal = true;
      existing.duration = this._calculateDuration(cleanTranslation, existing.originalText);
      existing.createdAt = Date.now();
      existing.expireAt = existing.createdAt + existing.duration;
      if (existing.isFadingOut) {
        existing.isFadingOut = false;
        existing.fadeStartAt = 0;
        this._removeFadeOutFromDOM(existing.id);
      }
    } else {
      // 3. Tạo completedItem mới đầy đủ cả originalText và translatedText
      const now = Date.now();
      const duration = this._calculateDuration(cleanTranslation, originalText);

      // Nếu có câu trước đó ở Tầng 2 nay bị đẩy lên Tầng 1 do có câu mới => RESET BỘ ĐẾM
      if (this.completedSentences.length > 0) {
        const prevSentence = this.completedSentences[this.completedSentences.length - 1];
        prevSentence.createdAt = now;
        prevSentence.expireAt = now + prevSentence.duration;
        if (prevSentence.isFadingOut) {
          prevSentence.isFadingOut = false;
          prevSentence.fadeStartAt = 0;
          this._removeFadeOutFromDOM(prevSentence.id);
        }
      }

      const completedItem = {
        id: id,
        originalText: originalText,
        translatedText: cleanTranslation,
        isFinal: true,
        duration: duration,
        createdAt: now,
        expireAt: now + duration,
        isFadingOut: false,
        fadeStartAt: 0,
      };
      this.completedSentences.push(completedItem);
    }

    // Nếu câu vừa dịch trùng với currentDraft ở Layer 3 -> Xóa Layer 3
    if (this.currentDraft && this.currentDraft.id === id) {
      this.currentDraft = null;
    }

    // Giới hạn số lượng câu lưu trữ theo Max Lines (Focus + History = maxLines)
    const maxTotal = this.maxLines || (this.maxHistorySentences + 1) || 2;
    while (this.completedSentences.length > maxTotal) {
      this.completedSentences.shift();
    }

    // Ràng buộc thời gian: Nếu Tầng 1 dài hơn Tầng 2 => Tầng 2 mất cùng lúc Tầng 1
    this._enforceLayerTimingConstraints();

    this._renderAll();
    this._scheduleNextLifecycleTick();
  }

  // Chuyển pendingFocus thành completedSentence với bản dịch cho sẵn (hoặc null nếu không có)
  _promotePendingToCompleted(translatedText) {
    if (!this.pendingFocus) return;
    const now = Date.now();
    const duration = this._calculateDuration(translatedText, this.pendingFocus.originalText);

    // 1. Trong trường hợp câu đang ở tầng 2 bị đẩy lên tầng 1 do có câu mới => RESET BỘ ĐẾM
    if (this.completedSentences.length > 0) {
      const prevSentence = this.completedSentences[this.completedSentences.length - 1];
      prevSentence.createdAt = now;
      prevSentence.expireAt = now + prevSentence.duration;
      if (prevSentence.isFadingOut) {
        prevSentence.isFadingOut = false;
        prevSentence.fadeStartAt = 0;
        this._removeFadeOutFromDOM(prevSentence.id);
      }
    }

    // 2. Thêm câu mới vào completedSentences (Tầng 2)
    const item = {
      id: this.pendingFocus.id,
      originalText: this.pendingFocus.originalText,
      translatedText: translatedText,
      isFinal: true,
      duration: duration,
      createdAt: now,
      expireAt: now + duration,
      isFadingOut: false,
      fadeStartAt: 0,
    };
    this.completedSentences.push(item);
    this.pendingFocus = null;

    // Giới hạn số lượng câu lưu trữ theo maxLines
    const maxTotal = this.maxLines || (this.maxHistorySentences + 1) || 2;
    while (this.completedSentences.length > maxTotal) {
      this.completedSentences.shift();
    }

    // 3. Ràng buộc thời gian: Nếu Tầng 1 dài hơn Tầng 2 => Tầng 2 mất cùng lúc Tầng 1
    this._enforceLayerTimingConstraints();

    // 4. Lên lịch kiểm tra lifecycle
    this._scheduleNextLifecycleTick();
  }

  // ── Render 3 Layers ───────────────────────────────────────────────

  _renderAll() {
    this._renderHistoryLayer();
    this._renderFocusLayer();
    this._renderLiveLayer();
  }

  // TẦNG 1: Lịch sử cũ (Chỉ re-render khi danh sách câu hoàn thành thay đổi)
  _renderHistoryLayer() {
    const len = this.completedSentences.length;
    const histStart = Math.max(0, this.pendingFocus ? (len - this.maxHistorySentences) : (len - 1 - this.maxHistorySentences));
    const histEnd = this.pendingFocus ? len : Math.max(0, len - 1);
    const histItems = (len > 0 && histEnd > histStart) ? this.completedSentences.slice(histStart, histEnd) : [];

    // Tạo fingerprint để kiểm tra xem history có thực sự thay đổi không
    const fp = histItems.map((s) => `${s.id}:${s.originalText || ""}:${s.translatedText || ""}`).join("|");
    if (fp === this._historyFingerprint) {
      return; // Không có thay đổi, giữ nguyên DOM tránh layout thrashing
    }
    this._historyFingerprint = fp;

    const newIds = new Set(histItems.map((item) => String(item.id)));

    // 1. Tìm các câu cũ đang hiển thị nhưng không còn trong histItems:
    //    Thay vì xóa đột ngột, thêm class .bs-evicting để trôi lên và mờ dần mượt mà trong 0.35s
    const currentChildren = Array.from(this.historyLayer.children);
    for (const child of currentChildren) {
      if (child.classList.contains("bs-evicting")) continue;
      const childId = child.getAttribute("data-sentence-id");
      if (!newIds.has(childId)) {
        child.classList.add("bs-evicting");
        setTimeout(() => {
          if (child.parentNode) {
            child.remove();
          }
        }, 380);
      }
    }

    // Dọn dẹp nếu có quá nhiều phần tử đang evicting dồn ứ (khi nói nhanh liên tục)
    const evictingNodes = this.historyLayer.querySelectorAll(".bs-evicting");
    if (evictingNodes.length > 2) {
      for (let i = 0; i < evictingNodes.length - 1; i++) {
        evictingNodes[i].remove();
      }
    }

    // 2. Thêm hoặc cập nhật các câu trong histItems
    for (const item of histItems) {
      let existingEl = this.historyLayer.querySelector(`[data-sentence-id="${item.id}"]:not(.bs-evicting)`);
      if (existingEl) {
        // Cập nhật in-place nội dung nếu câu đã có trong DOM
        const origEl = existingEl.querySelector(".bs-original");
        if (origEl && origEl.textContent !== item.originalText) {
          origEl.textContent = item.originalText || "";
        }
        const transEl = existingEl.querySelector(".bs-translated");
        if (item.translatedText) {
          if (!transEl) {
            // F-52: câu có thể vào TẦNG 1 khi CHƯA có bản dịch (bị câu mới đẩy lên từ
            // TẦNG 2) ⇒ node `.bs-translated` chưa từng được tạo. Bản dịch tới muộn
            // PHẢI được vẽ ra, không chỉ cập nhật node đã có.
            const newTransEl = document.createElement("div");
            newTransEl.className = "bs-translated";
            newTransEl.textContent = item.translatedText;
            existingEl.appendChild(newTransEl);
          } else if (transEl.textContent !== item.translatedText) {
            transEl.textContent = item.translatedText;
          }
        } else if (transEl) {
          // Bản dịch bị rút (ví dụ backend gửi bản rỗng) ⇒ bỏ node để không hiện chữ cũ.
          transEl.remove();
        }
        if (item.isFadingOut) {
          existingEl.classList.add("bs-fading-out");
        } else {
          existingEl.classList.remove("bs-fading-out");
        }
      } else {
        // Câu mới bước vào Tầng 1 -> tạo mới và gắn vào DOM
        const historyEl = this._createSentenceElement(item, "bs-sentence bs-history-item");
        this.historyLayer.appendChild(historyEl);
      }
    }
  }

  // TẦNG 2: Tiêu điểm trung tâm cố định (Anchor Focus) — Update in-place
  _renderFocusLayer() {
    const len = this.completedSentences.length;
    const isPending = !!this.pendingFocus;
    const targetItem = isPending ? this.pendingFocus : (len > 0 ? this.completedSentences[len - 1] : null);

    if (!targetItem) {
      if (this.currentFocusId !== null) {
        this.focusLayer.textContent = "";
        this.currentFocusId = null;
      }
      return;
    }

    const currentEl = this.focusLayer.firstElementChild;
    // Nếu cùng câu đang hiển thị -> cập nhật in-place textContent thay vì destroy/recreate
    if (this.currentFocusId === targetItem.id && currentEl) {
      if (targetItem.isFadingOut) {
        currentEl.classList.add("bs-fading-out");
      } else {
        currentEl.classList.remove("bs-fading-out");
      }

      const origEl = currentEl.querySelector(".bs-original");
      if (origEl && origEl.textContent !== targetItem.originalText) {
        origEl.textContent = targetItem.originalText || "";
      }

      let transEl = currentEl.querySelector(".bs-translated");
      let pendingIndicator = currentEl.querySelector(".bs-translating");

      if (isPending) {
        if (transEl) transEl.remove();
        if (!pendingIndicator) {
          pendingIndicator = document.createElement("div");
          pendingIndicator.className = "bs-translating";
          pendingIndicator.textContent = "・・・";
          currentEl.appendChild(pendingIndicator);
        }
      } else {
        if (pendingIndicator) pendingIndicator.remove();
        if (targetItem.translatedText) {
          if (!transEl) {
            transEl = document.createElement("div");
            transEl.className = "bs-translated";
            currentEl.appendChild(transEl);
          }
          if (transEl.textContent !== targetItem.translatedText) {
            transEl.textContent = targetItem.translatedText;
          }
        } else if (transEl) {
          transEl.remove();
        }
      }
      return;
    }

    // Nếu chuyển sang câu mới
    this.currentFocusId = targetItem.id;
    const focusEl = isPending
      ? this._createPendingFocusElement(targetItem)
      : this._createSentenceElement(targetItem, "bs-sentence bs-focus-item");
    this.focusLayer.replaceChildren(focusEl);
  }

  // TẦNG 3: Liveview nhận diện chưa chốt (Dưới cùng) — Update in-place
  _renderLiveLayer() {
    if (this.currentDraft && this.currentDraft.originalText && !this.currentDraft.isFinal) {
      const text = this.currentDraft.originalText;
      let liveEl = this.liveLayer.firstElementChild;
      if (!liveEl) {
        liveEl = document.createElement("div");
        liveEl.className = "bs-sentence bs-live-item";
        const originalSpan = document.createElement("span");
        originalSpan.className = "bs-original";
        originalSpan.textContent = text;
        liveEl.appendChild(originalSpan);
        this.liveLayer.replaceChildren(liveEl);
      } else {
        const originalSpan = liveEl.querySelector(".bs-original");
        if (originalSpan && originalSpan.textContent !== text) {
          originalSpan.textContent = text;
        }
      }
    } else {
      if (this.liveLayer.firstElementChild) {
        this.liveLayer.textContent = "";
      }
    }
  }

  // Tạo element cho pendingFocus (Layer 2, chờ dịch)
  _createPendingFocusElement(item) {
    const sentenceDiv = document.createElement("div");
    sentenceDiv.className = "bs-sentence bs-focus-item bs-focus-pending";
    sentenceDiv.setAttribute("data-sentence-id", item.id);

    const originalDiv = document.createElement("div");
    originalDiv.className = "bs-original";
    originalDiv.textContent = item.originalText || "";
    sentenceDiv.appendChild(originalDiv);

    // Indicator chờ dịch
    const translatingDiv = document.createElement("div");
    translatingDiv.className = "bs-translating";
    translatingDiv.textContent = "・・・";
    sentenceDiv.appendChild(translatingDiv);

    return sentenceDiv;
  }

  _createSentenceElement(item, className) {
    const sentenceDiv = document.createElement("div");
    sentenceDiv.className = className;
    if (item.isFadingOut) {
      sentenceDiv.classList.add("bs-fading-out");
    }
    sentenceDiv.setAttribute("data-sentence-id", item.id);
    sentenceDiv.setAttribute("data-utterance-id", item.id);

    const originalDiv = document.createElement("div");
    originalDiv.className = "bs-original";
    originalDiv.textContent = item.originalText || "";
    sentenceDiv.appendChild(originalDiv);

    // Bản dịch
    if (item.translatedText) {
      const translatedDiv = document.createElement("div");
      translatedDiv.className = "bs-translated";
      translatedDiv.textContent = item.translatedText;
      sentenceDiv.appendChild(translatedDiv);
    }

    return sentenceDiv;
  }

  // ── Legacy Handlers ───────────────────────────────────────

  onPartialTranscript(tokens) {
    if (!tokens || tokens.length === 0) return;
    const text = tokens.map((t) => t.text).join("");
    if (text) {
      this.onUtteranceUpdate({
        utterance_id: "legacy-draft",
        ui_text: text,
        is_final: false,
      });
    }
  }

  onSentenceComplete(sentence) {
    if (!sentence) return;
    this.onUtteranceUpdate({
      utterance_id: sentence.sentence_id || sentence.id,
      ui_text: sentence.ui_text || sentence.text || "",
      is_final: true,
    });
  }

  clear() {
    if (this.lifecycleTimer) {
      clearTimeout(this.lifecycleTimer);
      this.lifecycleTimer = null;
    }
    if (this.autoClearTimer) {
      clearTimeout(this.autoClearTimer);
      this.autoClearTimer = null;
    }
    this.completedSentences = [];
    this.utteranceStore.clear();
    this.currentDraft = null;
    this.pendingFocus = null;
    this.currentFocusId = null;
    this._historyFingerprint = "";
    if (this.historyLayer) this.historyLayer.textContent = "";
    if (this.focusLayer) this.focusLayer.textContent = "";
    if (this.liveLayer) this.liveLayer.textContent = "";
  }
}

// Attach to global for content script use
if (typeof window !== "undefined") {
  window.SubtitleRenderer = SubtitleRenderer;
}
