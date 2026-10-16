"""In-process MJPEG HTTP server for tests. Synthetic JPEG only (no faces)."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

BOUNDARY = "123456789000000000000987654321"


def synthetic_jpeg(width=64, height=48, seed=0) -> bytes:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def build_part(jpeg: bytes, frame_id: int, timestamp_ms: int, include_headers=True) -> bytes:
    if include_headers:
        header = (
            f"--{BOUNDARY}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n"
            f"X-Frame-Id: {frame_id}\r\n"
            f"X-Timestamp-Ms: {timestamp_ms}\r\n"
            f"\r\n"
        ).encode("ascii")
    else:
        header = (
            f"--{BOUNDARY}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n"
            f"\r\n"
        ).encode("ascii")
    return header + jpeg + b"\r\n"


class FakeMjpegServer:
    def __init__(self, parts: list[bytes] | None = None, n_frames: int = 8, split_chunks: int | None = None):
        self.parts = parts
        self.n_frames = n_frames
        self.split_chunks = split_chunks
        self.httpd = HTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.busy = False

    def _payload(self) -> bytes:
        if self.parts is not None:
            return b"".join(self.parts)
        blobs = []
        for i in range(self.n_frames):
            jpeg = synthetic_jpeg(seed=i)
            blobs.append(build_part(jpeg, frame_id=100 + i, timestamp_ms=1000 + i * 100))
        return b"".join(blobs)

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path.split("?")[0] != "/stream":
                    self.send_error(404)
                    return
                if server.busy:
                    self.send_error(503, "Stream busy: one client at a time")
                    return
                server.busy = True
                try:
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        f"multipart/x-mixed-replace;boundary={BOUNDARY}",
                    )
                    self.end_headers()
                    payload = server._payload()
                    if server.split_chunks:
                        n = server.split_chunks
                        for i in range(0, len(payload), n):
                            self.wfile.write(payload[i:i + n])
                            self.wfile.flush()
                    else:
                        self.wfile.write(payload)
                        self.wfile.flush()
                finally:
                    server.busy = False

        return Handler

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}/stream"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
