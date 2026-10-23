"""Shape a multipart/x-mixed-replace MJPEG stream for a remote viewer.

The frame publisher writes every published frame at full rate for a LAN
viewer. A browser behind a tunnel (Tailscale Funnel, a hotel uplink) cannot
drink that: four tiles of 20 fps / 640 px JPEG are ~3 MB/s against a path
that carries ~0.5 MB/s, so every tile starves and looks frozen (CHI, 23 Oct
2026). The API's proxy feeds the upstream bytes through a shaper that keeps
at most `max_fps` frames a second and re-encodes each kept frame to
`width` pixels at JPEG `quality`. Pure bytes in, bytes out; no asyncio.

The upstream part format (cvti/serving/frame_publisher.py):

    --argusframe\\r\\n
    Content-Type: image/jpeg\\r\\n
    Content-Length: N\\r\\n\\r\\n
    <N bytes of JPEG>\\r\\n
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable

DEFAULT_REMOTE_FPS = 8.0
DEFAULT_REMOTE_WIDTH = 480
DEFAULT_REMOTE_QUALITY = 55

_PART_HEAD = re.compile(rb"--(?P<boundary>[^\r\n]+)\r\n(?P<headers>(?:[^\r\n]+\r\n)*)\r\n")


def remote_profile(*, fps: float | None = None, width: int | None = None,
                   quality: int | None = None) -> dict | None:
    """The shaping profile for one request: query values win, then the
    environment (ARGUS_REMOTE_STREAM_FPS / _WIDTH / _QUALITY), then the
    defaults. Returns None when nothing is to be shaped (fps and width both
    0), which is what a LAN viewer asks for."""
    def pick(value, env, default, cast):
        if value is not None:
            return cast(value)
        raw = os.environ.get(env)
        if raw not in (None, ""):
            try:
                return cast(raw)
            except ValueError:
                pass
        return default
    max_fps = pick(fps, "ARGUS_REMOTE_STREAM_FPS", DEFAULT_REMOTE_FPS, float)
    max_width = pick(width, "ARGUS_REMOTE_STREAM_WIDTH", DEFAULT_REMOTE_WIDTH, int)
    jpeg_q = pick(quality, "ARGUS_REMOTE_STREAM_QUALITY", DEFAULT_REMOTE_QUALITY, int)
    if max_fps <= 0 and max_width <= 0:
        return None
    return {"max_fps": max(0.0, max_fps), "width": max(0, max_width),
            "quality": min(95, max(10, jpeg_q))}


class MjpegShaper:
    def __init__(self, *, max_fps: float = DEFAULT_REMOTE_FPS,
                 width: int = DEFAULT_REMOTE_WIDTH, quality: int = DEFAULT_REMOTE_QUALITY,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_fps = float(max_fps)
        self.width = int(width)
        self.quality = int(quality)
        self._clock = clock
        self._buf = b""
        self._next_due = -1e9        # accumulator: average rate == max_fps
        self.kept = 0
        self.dropped = 0

    # -- the stream ---------------------------------------------------------
    def feed(self, chunk: bytes) -> bytes:
        """Take the next upstream bytes; return the bytes to send downstream
        (possibly empty: a dropped frame or an incomplete part)."""
        self._buf += chunk
        out = []
        while True:
            m = _PART_HEAD.search(self._buf)
            if m is None:
                break
            length = self._content_length(m.group("headers"))
            start = m.end()
            if length is None:
                # Not a part we understand: forward it untouched up to the
                # next boundary once we see one, otherwise wait for more.
                nxt = self._buf.find(b"--" + m.group("boundary"), start)
                if nxt < 0:
                    break
                out.append(self._buf[m.start():nxt])
                self._buf = self._buf[nxt:]
                continue
            end = start + length
            if len(self._buf) < end + 2:
                break                              # the JPEG is still arriving
            jpeg = self._buf[start:end]
            self._buf = self._buf[end + 2:]        # past the trailing CRLF
            shaped = self._shape(jpeg)
            if shaped is not None:
                out.append(b"--" + m.group("boundary") + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           + f"Content-Length: {len(shaped)}\r\n\r\n".encode()
                           + shaped + b"\r\n")
        # Drop any junk before the first boundary so the buffer cannot grow
        # without bound on a stream we never recognise.
        if len(self._buf) > 4 * 1024 * 1024:
            self._buf = self._buf[-1024:]
        return b"".join(out)

    # -- one frame ----------------------------------------------------------
    def _shape(self, jpeg: bytes) -> bytes | None:
        now = self._clock()
        if self.max_fps > 0:
            # "At least a period since the last send" rounds up to the next
            # upstream frame every time (20 fps in, 8 asked -> every 3rd
            # frame = 6.7 out). Schedule the next due time instead and let a
            # late frame pull it forward, so the average is what was asked.
            if now < self._next_due:
                self.dropped += 1
                return None
            period = 1.0 / self.max_fps
            self._next_due = max(self._next_due + period, now - period) if self._next_due > -1e8 else now + period
        self.kept += 1
        if self.width <= 0:
            return jpeg
        return self._reencode(jpeg)

    def _reencode(self, jpeg: bytes) -> bytes:
        try:
            import cv2
            import numpy as np
        except ImportError:                        # a slim API host: pass through
            return jpeg
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jpeg
        h, w = img.shape[:2]
        if w > self.width:
            img = cv2.resize(img, (self.width, max(1, int(h * self.width / float(w)))),
                             interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return enc.tobytes() if ok else jpeg

    @staticmethod
    def _content_length(headers: bytes) -> int | None:
        for line in headers.split(b"\r\n"):
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                try:
                    return int(value.strip())
                except ValueError:
                    return None
        return None
