"""MJPEG multipart parser that preserves part headers split across chunks.

The ESP32 firmware sends (see camera_stream.ino):
  Content-Type: image/jpeg
  Content-Length: <n>
  X-Frame-Id: <device frame counter>
  X-Timestamp-Ms: <device millis()>

OpenCV VideoCapture does not expose those headers. This client is the
canonical way to keep device_frame_id and device_timestamp_ms.

Clock domains:
  device_timestamp_ms  -> ESP32 millis() since boot (not Unix time)
  received_timestamp_ms -> laptop Unix epoch milliseconds
These clocks are NOT synchronized. Do not subtract them to claim capture latency.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, Optional


DEFAULT_BOUNDARY = b"123456789000000000000987654321"


@dataclass
class MjpegPart:
    jpeg: bytes
    headers: dict
    device_frame_id: Optional[int]
    device_timestamp_ms: Optional[int]
    received_timestamp_ms: int
    metadata_status: str  # OK | PARTIAL | MISSING
    camera_clock_domain: str
    gateway_clock_domain: str = "unix_epoch_ms"
    parse_error: Optional[str] = None


def _header_map(raw: bytes) -> dict:
    text = raw.decode("latin-1", errors="replace")
    out = {}
    for line in text.split("\r\n"):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip().lower()] = value.strip()
    return out


def classify_metadata(headers: dict) -> tuple[Optional[int], Optional[int], str]:
    fid = headers.get("x-frame-id")
    ts = headers.get("x-timestamp-ms")
    frame_id = None
    timestamp = None
    try:
        if fid not in (None, ""):
            frame_id = int(fid)
    except ValueError:
        frame_id = None
    try:
        if ts not in (None, ""):
            timestamp = int(ts)
    except ValueError:
        timestamp = None
    if frame_id is not None and timestamp is not None:
        status = "OK"
    elif frame_id is None and timestamp is None:
        status = "MISSING"
    else:
        status = "PARTIAL"
    return frame_id, timestamp, status


class MjpegStreamParser:
    """Incremental parser. Headers or JPEG bodies may be split across feed() calls."""

    def __init__(self, boundary: bytes = DEFAULT_BOUNDARY):
        b = boundary.strip()
        if b.startswith(b"--"):
            b = b[2:]
        self.boundary = b"--" + b
        self._buf = b""
        self._in_body = False
        self._headers = {}
        self._body_needed = None
        self._body = bytearray()

    def feed(self, chunk: bytes, received_timestamp_ms: Optional[int] = None) -> list[MjpegPart]:
        if not chunk:
            return []
        self._buf += chunk
        parts: list[MjpegPart] = []
        while True:
            part = self._try_one(received_timestamp_ms)
            if part is None:
                break
            parts.append(part)
        return parts

    def _try_one(self, received_timestamp_ms: Optional[int]) -> Optional[MjpegPart]:
        if not self._in_body:
            header_end = self._buf.find(b"\r\n\r\n")
            if header_end < 0:
                return None
            header_block = self._buf[:header_end]
            # Skip preamble / boundary lines
            if self.boundary in header_block:
                # Keep the last header section after the latest boundary
                idx = header_block.rfind(self.boundary)
                header_block = header_block[idx + len(self.boundary):]
                if header_block.startswith(b"\r\n"):
                    header_block = header_block[2:]
            self._headers = _header_map(header_block)
            self._buf = self._buf[header_end + 4:]
            length = self._headers.get("content-length")
            try:
                self._body_needed = int(length) if length is not None else None
            except ValueError:
                self._body_needed = None
            self._in_body = True
            self._body = bytearray()

        if self._body_needed is not None:
            have = len(self._body)
            need = self._body_needed - have
            if len(self._buf) < need:
                self._body.extend(self._buf)
                self._buf = b""
                return None
            self._body.extend(self._buf[:need])
            self._buf = self._buf[need:]
            jpeg = bytes(self._body)
            # Optional trailing CRLF before next boundary
            if self._buf.startswith(b"\r\n"):
                self._buf = self._buf[2:]
            return self._finish(jpeg, received_timestamp_ms)

        # No Content-Length: read until next boundary
        bound = self._buf.find(self.boundary)
        if bound < 0:
            # Keep a tail so a split boundary can complete later
            keep = len(self.boundary) + 8
            if len(self._buf) > keep:
                self._body.extend(self._buf[:-keep])
                self._buf = self._buf[-keep:]
            return None
        jpeg = bytes(self._body) + self._buf[:bound]
        if jpeg.endswith(b"\r\n"):
            jpeg = jpeg[:-2]
        self._buf = self._buf[bound:]
        return self._finish(jpeg, received_timestamp_ms)

    def _finish(self, jpeg: bytes, received_timestamp_ms: Optional[int]) -> MjpegPart:
        self._in_body = False
        fid, ts, meta_status = classify_metadata(self._headers)
        rx = int(time.time() * 1000) if received_timestamp_ms is None else received_timestamp_ms
        clock = "esp32_millis" if ts is not None else "unknown"
        headers = dict(self._headers)
        self._headers = {}
        self._body_needed = None
        self._body = bytearray()
        return MjpegPart(
            jpeg=jpeg,
            headers=headers,
            device_frame_id=fid,
            device_timestamp_ms=ts,
            received_timestamp_ms=rx,
            metadata_status=meta_status,
            camera_clock_domain=clock,
        )


def boundary_from_content_type(content_type: str) -> bytes:
    if not content_type:
        return DEFAULT_BOUNDARY
    lower = content_type.lower()
    marker = "boundary="
    idx = lower.find(marker)
    if idx < 0:
        return DEFAULT_BOUNDARY
    raw = content_type[idx + len(marker):].strip().strip('"')
    # boundary may be followed by more parameters
    raw = raw.split(";")[0].strip().strip('"')
    return raw.encode("ascii", errors="replace") or DEFAULT_BOUNDARY


def iter_mjpeg_url(
    url: str,
    stop_flag=None,
    chunk_size: int = 4096,
    timeout: float = 5.0,
) -> Iterator[MjpegPart]:
    req = urllib.request.Request(url, headers={"Accept": "multipart/x-mixed-replace"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type") or resp.headers.get("content-type") or ""
        parser = MjpegStreamParser(boundary_from_content_type(ctype))
        while True:
            if stop_flag is not None and stop_flag.is_set():
                return
            chunk = resp.read(chunk_size)
            if not chunk:
                return
            rx = int(time.time() * 1000)
            for part in parser.feed(chunk, received_timestamp_ms=rx):
                yield part
