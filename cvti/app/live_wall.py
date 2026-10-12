"""Live wall — decode several video sources concurrently and keep the latest
JPEG frame of each, for the app's multi-camera grid.

Qt-free (OpenCV + threads only) so it runs in the app process without the
detection stack, and is unit-testable headless. One daemon thread per source
reads frames in a loop (file sources loop at EOF so they play "live"),
downscales, JPEG-encodes, and stores the latest as a base64 data URI the web UI
can drop straight into an <img src>.
"""
from __future__ import annotations

import base64
import os
import threading
import time

from cvti.logging_setup import get_logger
log = get_logger(__name__)

import cv2


class LiveWall:
    def __init__(self, sources: list[dict], width: int = 400, fps: float = 8.0,
                 quality: int = 65, retain_snapshot: bool = False) -> None:
        # sources: [{"id": str, "source": str|int}, ...]
        self.sources = sources
        self.width = width
        self.interval = 1.0 / max(1.0, fps)
        self.quality = quality
        self.retain_snapshot = retain_snapshot
        self._raw = {}
        self._latest: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._diagnostics = {}
        self._timings = {}

    def _timed(self, cam_id, stage, started):
        elapsed = (time.perf_counter() - started) * 1000
        with self._lock:
            rec = self._timings.setdefault(cam_id, {}).setdefault(stage,
                {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            rec["count"] += 1
            rec["total_ms"] += elapsed
            rec["max_ms"] = max(rec["max_ms"], elapsed)

    def _record(self, cam_id, state=None, **values):
        with self._lock:
            rec = self._diagnostics.setdefault(cam_id, {"started_at": time.time(),
                "open_attempts": 0, "decoded_count": 0, "published_count": 0,
                "read_failures": 0, "events": []})
            if state is not None and state != rec.get("state"):
                rec["events"] = (rec["events"] + [{"at": time.time(), "state": state}])[-20:]
                rec["state"] = state
            rec.update(values)

    def diagnostics(self):
        from copy import deepcopy
        with self._lock:
            records = deepcopy(self._diagnostics)
            timings = deepcopy(self._timings)
        now = time.time()
        for cam_id, rec in records.items():
            rec["stage_wall_times"] = {stage: dict(values,
                mean_ms=round(values["total_ms"] / values["count"], 2))
                for stage, values in timings.get(cam_id, {}).items()}
            last = rec.get("last_decoded_at")
            rec["last_decoded_age_s"] = round(now - last, 2) if last else None
            rec["average_published_fps"] = round(rec["published_count"] / max(.001, now - rec["started_at"]), 2)
            rec["protocol_error_detail"] = "unavailable from OpenCV capture API"
        return records

    def _open_observed(self, cam_id, source):
        self._record(cam_id, "opening")
        with self._lock:
            self._diagnostics[cam_id]["open_attempts"] += 1
        started = time.perf_counter()
        cap = self._open(source)
        self._record(cam_id, "waiting_for_frame", last_open_ms=round((time.perf_counter() - started) * 1000, 1))
        return cap

    def start(self) -> "LiveWall":
        self._stop.clear()
        for s in self.sources:
            t = threading.Thread(target=self._decode, args=(s["id"], s["source"]),
                                 name=f"live-{s['id']}", daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def _open(self, source):
        from cvti.serving.capture import open_capture
        return open_capture(source)

    def _decode(self, cam_id: str, source) -> None:
        captures = []
        try:
            self._decode_frames(cam_id, source, captures)
        except Exception:
            self._record(cam_id, "failed", failure="capture_exception")
            self._set(cam_id, ok=False, error="camera preview failed")
            log.debug("preview capture failed", exc_info=True)
        finally:
            for cap in captures:
                cap.release()

    def _decode_frames(self, cam_id: str, source, captures) -> None:
        cap = self._open_observed(cam_id, source)
        captures.append(cap)
        src = str(source)
        is_net = "://" in src                     # RTSP/HTTP camera, not a clip
        is_file = not (isinstance(source, int) or src.isdigit() or is_net)
        is_webcam = isinstance(source, int) or src.isdigit()
        first_frame_at = None
        next_publish_at = 0.0
        n = 0
        dead_since = 0.0
        while not self._stop.is_set():
            started = time.perf_counter()
            ok, frame = cap.read()
            self._timed(cam_id, "read", started)
            if not ok:
                if is_file:                       # loop the clip so it plays continuously
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok:
                    self._record(cam_id, "retrying", failure="no_decoded_frame")
                    with self._lock:
                        self._diagnostics[cam_id]["read_failures"] += 1
                    # A dropped network stream must be REOPENED — a VideoCapture
                    # never revives on its own, so this loop used to hold a dead
                    # handle forever and the pre-monitoring view stayed frozen
                    # after any camera blip (proven against mediamtx, 29 Aug:
                    # frame counter identical before/after the stream returned).
                    # Same disease the rules scanner had on 23 Aug.
                    self._set(cam_id, ok=False, error="stream dropped — reconnecting")
                    dead_since = dead_since or time.time()
                    if not is_file and time.time() - dead_since >= 3.0:
                        try:
                            cap.release()
                        except Exception:  # noqa: BLE001
                            log.debug("releasing the dead capture failed", exc_info=True)
                        cap = self._open_observed(cam_id, source)
                        captures[:] = [cap]
                        first_frame_at = None
                        dead_since = time.time()
                    self._stop.wait(0.3)
                    continue
            dead_since = 0.0
            observed = time.time()
            self._record(cam_id, "receiving", last_decoded_at=observed, failure=None)
            with self._lock:
                rec = self._diagnostics[cam_id]
                rec["decoded_count"] += 1
                if "first_decoded_at" not in rec:
                    rec["first_decoded_at"] = observed
                    rec["first_decode_delay_s"] = round(observed - rec["started_at"], 3)
            now = time.monotonic()
            if is_webcam:
                # First USB/AVFoundation frames can be almost black while auto
                # exposure settles. Drain at capture speed before publishing.
                if first_frame_at is None:
                    first_frame_at = now
                if now - first_frame_at < 2.0:
                    self._stop.wait(0.005)
                    continue
            n += 1
            # Live captures must be drained at source cadence. Sleeping after
            # each read queues old video; limit JPEG encoding, not camera reads.
            if not is_file and now < next_publish_at:
                continue
            next_publish_at = now + self.interval
            if self.retain_snapshot:
                with self._lock:
                    self._raw[cam_id] = frame.copy()
            started = time.perf_counter()
            frame = self._downscale(frame)
            self._timed(cam_id, "resize", started)
            started = time.perf_counter()
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            self._timed(cam_id, "jpeg_encode", started)
            if ok2:
                with self._lock:
                    self._diagnostics[cam_id]["published_count"] += 1
                    self._diagnostics[cam_id]["last_published_at"] = time.time()
                # Store RAW jpeg bytes — the frame server streams these natively
                # over localhost so the browser never marshals base64 through
                # QWebChannel (that was the FPS ceiling).
                self._set(cam_id, jpeg=buf.tobytes(), w=int(frame.shape[1]),
                          h=int(frame.shape[0]), frame=n, ok=True)
            else:
                self._record(cam_id, "encode_failed", failure="jpeg_encode_failed")
            if is_file:
                self._stop.wait(self.interval)
        cap.release()

    def _downscale(self, frame):
        h, w = frame.shape[:2]
        if w > self.width:
            nh = max(1, int(h * self.width / w))
            frame = cv2.resize(frame, (self.width, nh))
        return frame

    def _set(self, cam_id: str, **kw) -> None:
        with self._lock:
            rec = self._latest.setdefault(cam_id, {})
            rec.update(kw)

    def frames(self) -> dict:
        # Metadata only (no image bytes) — small + cheap to poll for status.
        with self._lock:
            return {k: {kk: vv for kk, vv in v.items() if kk != "jpeg"} for k, v in self._latest.items()}

    def jpeg(self, cam_id: str) -> bytes | None:
        with self._lock:
            rec = self._latest.get(cam_id)
            return rec.get("jpeg") if rec and rec.get("ok") else None

    def snapshot(self, cam_id):
        with self._lock:
            rec = self._latest.get(cam_id, {})
            frame = self._raw.get(cam_id) if rec.get("ok") else None
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return {"uri": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode(),
                "w": int(frame.shape[1]), "h": int(frame.shape[0])}

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.5)
        self._threads = [t for t in self._threads if t.is_alive()]


class FrameServer:
    """Serves the LiveWall's latest JPEG per camera over localhost, so the UI's
    <img> tags fetch frames natively (fast) instead of base64-over-QWebChannel."""

    def __init__(self, wall: "LiveWall", token: str = "") -> None:
        import secrets
        self.wall = wall
        # Same rule as the engine's publisher: no route to a camera frame is
        # unauthenticated, including this local fallback.
        self.token = token or secrets.token_urlsafe(24)
        self.port = 0
        self._httpd = None
        self._thread = None
        self._stopped = threading.Event()

    @staticmethod
    def expired(elapsed, received_image):
        return elapsed > (5 if received_image else 30)

    def start(self) -> int:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        wall = self.wall
        token = self.token

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence request logging
                pass

            # Chromium refuses to progressively render multipart over
            # HTTP/1.0 — the same lesson the ENGINE's publisher learned the
            # hard way (blank-wall bug, 27 Aug).
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                import hmac
                import urllib.parse
                parsed = urllib.parse.urlparse(self.path)
                supplied = self.headers.get("X-Argus-Token", "") or (
                    urllib.parse.parse_qs(parsed.query).get("token") or [""])[0]
                if not hmac.compare_digest(supplied, token):
                    self.send_error(401)
                    return
                cam = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
                if parsed.path.startswith("/stream/"):
                    self._mjpeg(cam)
                    return
                jpg = wall.jpeg(cam)
                if not jpg:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                try:
                    self.wfile.write(jpg)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _mjpeg(self, cam):
                """Push frames until the client leaves — the UI's tiles and the
                zoom modal use /stream/ URLs, and this server used to answer
                them with ONE static JPEG: the fallback 'live' view froze on a
                single frame ('shows for a while and cut off', field report
                29 Aug — the moving part was the engine before it died; this
                is the still part after). With a real stream here, Watch is
                live even before monitoring starts."""
                acquire = getattr(wall, "acquire", None)
                if acquire is not None and not acquire(cam):
                    self.send_error(404)
                    return
                self.connection.settimeout(3)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=arguswall")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                except OSError:
                    release = getattr(wall, "release", None)
                    if release is not None:
                        release(cam)
                    return
                import time as _t
                last = None
                last_frame = _t.monotonic()
                try:
                    while not stopped.is_set():
                        jpg = wall.jpeg(cam)
                        if jpg is not None and jpg is not last:
                            last_frame = _t.monotonic()
                            last = jpg
                            self.wfile.write(b"--arguswall\r\n"
                                             b"Content-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                                             + jpg + b"\r\n")
                            self.wfile.flush()
                        if FrameServer.expired(_t.monotonic() - last_frame, last is not None):
                            break  # no stale frame masquerading as live video
                        _t.sleep(0.12)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass          # viewer navigated away — normal
                finally:
                    release = getattr(wall, "release", None)
                    if release is not None:
                        release(cam)

        stopped = self._stopped
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)   # port 0 = OS picks
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="frame-server", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        self._stopped.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
