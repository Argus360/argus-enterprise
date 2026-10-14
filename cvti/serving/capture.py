"""One way to open a video source, on every platform (27 Aug).

Three call sites each opened captures their own way, and all three inherited
OpenCV's defaults — which differ per platform in ways that matter:

- **Buffering.** OpenCV queues frames internally. On a live source that queue
  IS latency: you decode the past while the present piles up behind it. The
  queue is deeper on Windows than macOS, which is why the same code feels
  laggier there. `CAP_PROP_BUFFERSIZE = 1` asks for the present.
- **Webcams on Windows** default to the MSMF backend, which routinely takes
  several seconds to open a camera and negotiates poor frame rates. DirectShow
  opens immediately and is what every Windows CV app uses.
- **Network streams** need explicit FFmpeg plus low-latency flags, or they
  buffer whole segments before yielding a frame (HLS measured at 22.5s to
  first frame, with stalls to 5.5s).

Everything here degrades safely: an unsupported property or backend falls back
to OpenCV's default rather than failing to open a camera.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Low-latency FFmpeg options for live network sources. `nobuffer` and
# `low_delay` stop the demuxer holding frames back; `reorder_queue_size=0`
# stops RTSP reordering adding a queue of its own; the TCP transport is what
# most cameras and every NAT actually work with.
_FFMPEG_LIVE_OPTS = (
    "rtsp_transport;tcp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|reorder_queue_size;0"
)


def is_live_source(source: Any) -> bool:
    """A network stream (rtsp/http/https) — as opposed to a file or a webcam."""
    s = str(source)
    return not s.isdigit() and "://" in s


def _hw_decode_params(cv2) -> list:
    """Ask FFmpeg to decode on the GPU when one exists (3 Sep, 'the RTSP is
    slow — I need it fast').

    A CPU-only pilot box decoding a 1080p25 RTSP main stream spends real
    cores on work its integrated GPU does for near-free: D3D11 on Windows,
    VideoToolbox on macOS, VAAPI on Linux. VIDEO_ACCELERATION_ANY negotiates
    whatever exists and falls back to software BY ITSELF, so this is a free
    upgrade on capable machines and a no-op elsewhere. Measured on the dev
    Mac: 0.51 -> 0.21 ms/frame on a test clip; the win on a Windows iGPU at
    1080p is an order of magnitude.
    """
    if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
        return [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
    return []          # ancient OpenCV: software decode, as before


def _live_decode_params(cv2) -> list:
    """Single-threaded decode for LIVE sources (7 Sep, found by the latency
    baseline's first run).

    OpenCV defaults the FFmpeg decoder to thread_count = CPU count, and
    H.264 frame-threading is a pipeline: output is delayed ~one frame per
    thread. On an 11-core dev Mac that measured 810ms of standing latency on
    a 15fps stream (11 x 66.7ms — the arithmetic is exact); on the pilot's
    4-core box it is ~250ms baked into every frame, alert evidence included.
    A camera stream is one frame at a time — the throughput frame-threading
    buys is worthless here, and the latency it costs is the product's whole
    complaint. threads=1 measured 810 -> 145ms p50; each extra thread adds
    exactly one frame period. Offline work keeps the default: an eval run
    wants throughput, and nobody is waiting at the glass.
    """
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        return [cv2.CAP_PROP_N_THREADS, 1]
    return []          # older OpenCV: no such knob, keep the old behaviour


# Set the first time the hardware-decode open fails and the software open
# succeeds: a box with no usable GPU (the pilot server) paid a failed D3D11
# attempt plus a warning on EVERY reconnect of every camera (13 Oct 2026).
_HW_DECODE_FAILED = False


def rtsp_alive(source: Any, timeout: float = 4.0) -> bool:
    """Is the camera reachable, without decoding anything: one TCP connect
    and one RTSP OPTIONS. Any RTSP reply (200, 401, 404...) means a camera
    is answering; no reply or a refused connection means it is not. For a
    non-RTSP network source, the TCP connect alone decides."""
    import socket
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(str(source))
        host, port = parts.hostname, parts.port or (554 if parts.scheme.startswith("rtsp") else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout) as sock:
            if not parts.scheme.startswith("rtsp"):
                return True
            bare = parts._replace(netloc=f"{host}:{port}").geturl()
            sock.sendall(f"OPTIONS {bare} RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: argus-engine/1\r\n\r\n".encode())
            sock.settimeout(timeout)
            reply = sock.recv(256)
            return reply.startswith(b"RTSP/")
    except (OSError, ValueError):
        return False


def open_capture(source: Any, *, low_latency: bool = True):
    """Open `source` with the right backend and the least buffering.

    `low_latency=False` for offline work (evaluation over files) where every
    frame matters more than freshness.
    """
    import cv2

    global _HW_DECODE_FAILED
    src = int(source) if str(source).isdigit() else source
    hw = [] if _HW_DECODE_FAILED else _hw_decode_params(cv2)
    live_params = _live_decode_params(cv2) if low_latency else []

    if isinstance(src, int):
        # Webcam. DirectShow on Windows: MSMF can take seconds to open and
        # often negotiates a worse mode. Everywhere else the default is right.
        # (No hw params: webcams hand over raw frames — nothing to decode.)
        if sys.platform == "win32":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():
                log.warning("DirectShow could not open webcam %s; falling back", src)
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
    elif is_live_source(src):
        if low_latency:
            # setdefault so an operator's explicit env var always wins.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FFMPEG_LIVE_OPTS)
        # FFmpeg applies these at open time, not through cap.set(). Bound
        # failed opens and reads so reconnect/preview shutdown can progress.
        timeouts = []
        for name, milliseconds in (("CAP_PROP_OPEN_TIMEOUT_MSEC", 8000),
                                   ("CAP_PROP_READ_TIMEOUT_MSEC", 5000)):
            if hasattr(cv2, name):
                timeouts.extend([getattr(cv2, name), milliseconds])
        params = hw + live_params + timeouts
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, params) if params \
            else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if hw and not cap.isOpened():
            # ANY should fall back internally; if a broken driver still
            # refuses the open, software decode beats no camera. Retry keeps
            # the single-thread ask — it is a latency fix, not an accelerator.
            cap.release()
            software_params = live_params + timeouts
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, software_params) if software_params \
                else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if cap.isOpened():
                _HW_DECODE_FAILED = True
                log.warning("hardware decode is not available on this machine; "
                            "software decode from now on (no further attempts)")
            else:
                log.warning("network capture open failed with and without hardware decode")
    else:
        # A file: same free decode upgrade (demo clips are video too).
        cap = cv2.VideoCapture(src, cv2.CAP_ANY, hw) if hw else cv2.VideoCapture(src)
        if hw and not cap.isOpened():
            cap = cv2.VideoCapture(src)

    if low_latency and (isinstance(src, int) or is_live_source(src)):
        try:
            # Ask for a queue of one. Not every backend honours it; the
            # decoder's live-edge drain covers the ones that do not.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - a tuning hint must never break capture
            log.debug("BUFFERSIZE not supported by this backend", exc_info=True)
    return cap
