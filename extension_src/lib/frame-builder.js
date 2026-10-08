// Shared binary audio packet builder for Content Script & Service Worker (DRY)
(function (global) {
  const defaultTextEncoder = new TextEncoder();

  /**
   * Build binary audio frame: [4-byte uint32 LE header length][header JSON bytes][PCM bytes]
   * @param {Object} header - Metadata object (type, captureTimestamp, chunkIndex, ...)
   * @param {ArrayBuffer|ArrayBufferView} pcmData - Raw PCM bytes or typed array
   * @param {TextEncoder} [encoder] - Optional cached TextEncoder instance
   * @returns {ArrayBuffer}
   */
  function buildBinaryAudioPacket(header, pcmData, encoder) {
    const enc = encoder || defaultTextEncoder;
    const headerStr = JSON.stringify(header);
    const headerBytes = enc.encode(headerStr);

    let pcmBytes;
    if (pcmData instanceof Uint8Array) {
      pcmBytes = pcmData;
    } else if (ArrayBuffer.isView(pcmData)) {
      pcmBytes = new Uint8Array(pcmData.buffer, pcmData.byteOffset, pcmData.byteLength);
    } else if (pcmData && typeof pcmData === "object") {
      try {
        // Robust against cross-realm ArrayBuffer or objects holding a buffer
        pcmBytes = new Uint8Array(pcmData.buffer ? pcmData.buffer : pcmData);
      } catch (e) {
        pcmBytes = new Uint8Array(0);
      }
    } else {
      pcmBytes = new Uint8Array(0);
    }

    const totalSize = 4 + headerBytes.length + pcmBytes.byteLength;
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);

    // 4-byte little-endian header length
    view.setUint32(0, headerBytes.length, true);
    new Uint8Array(buffer, 4, headerBytes.length).set(headerBytes);
    new Uint8Array(buffer, 4 + headerBytes.length).set(pcmBytes);

    return buffer;
  }

  global.buildBinaryAudioPacket = buildBinaryAudioPacket;
})(typeof globalThis !== "undefined" ? globalThis : this);
