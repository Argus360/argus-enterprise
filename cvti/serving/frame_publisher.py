"""Publish the engine's frames so the UI doesn't decode the same video twice.

The engine and the app are separate processes. Until now each opened every camera
independently: the engine to detect, the app's live wall to display. That doubled
decode cost (the dominant per-camera cost) and capped the wall at ~4 cameras — and
the app, having no detection state, could never draw a box on anyone.

The engine already decodes every stream and knows where tracked movement is, so
it publishes the latest raw frame plus an on-demand tracking variant over a
localhost HTTP port. The app just fetches images.

    GET /frame/<camera_id>   -> image/jpeg  (latest raw frame)
    GET /stream/<camera_id>?tracking=1 -> MJPEG (movement overlays on demand)
    GET /cameras             -> {"cameras": [...], "tracks": {...},
                                  "object_tracks": {...}}

The port is written to <output_dir>/frames.json so the app can find it without
being told. Loopback only — frames never leave the box.
"""
from __future__ import annotations

import copy
import hmac
import json
import math
import os
import secrets
import select
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, unquote, urlparse
from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Boxes are drawn in the engine (it owns the tracks), so the UI stays a dumb viewer.
_BOX_COLOUR = (0, 200, 0)        # BGR green for ordinary scenario-4 movement
_ALERT_COLOUR = (60, 60, 220)    # red once that track is part of an active alert


@dataclass(frozen=True)
class FrameOverlay:
    track_id: int
    bbox: tuple[int, int, int, int]
    label: str
    colour: tuple[int, int, int]
    namespace: str = "person"


class FramePublisher:
    """Holds the latest JPEG per camera and serves it over loopback HTTP."""

    def __init__(self, *, quality: int = 70, draw_boxes: bool = True,
                 max_width: int = 640, token: str = "",
                 health_provider=None) -> None:
        # /health serves whatever this returns — the engine passes its live
        # health-doc builder in. None means the route 404s honestly.
        self.health_provider = health_provider
        # Generated per run, handed to the app through frames.json. Not a
        # user credential — a capability for the one process that needs frames.
        self.token = token or secrets.token_urlsafe(24)
        self.quality = quality
        self.draw_boxes = draw_boxes
        self.max_width = max_width
        self._raw_frames: dict[str, bytes] = {}
        self._tracking_frames: dict[str, bytes] = {}
        self._tracking_person_frames: dict[str, bytes] = {}
        self._tracks: dict[str, list] = {}
        self._object_tracks: dict[str, dict | None] = {}
        self._object_overlay_expires_at: dict[str, float] = {}
        self._object_generation: dict[str, int] = {}
        self._alerting: dict[str, set] = {}
        self._raw_seq: dict[str, int] = {}
        self._tracking_seq: dict[str, int] = {}
        self._issued_seq: dict[str, int] = {}
        # Who is being WATCHED right now. Every tile on the Watch grid holds
        # one long-lived /stream connection, so an open stream IS a viewer —
        # no config, no heartbeat file, works for any client including the
        # phone. The decode loop reads this to boost display fps only where
        # eyes actually are (3 Sep). `viewer_linger` keeps the boost through a
        # tab switch or MJPEG reconnect instead of flapping the decoder.
        self._viewers: dict[str, int] = {}
        self._tracking_viewers: dict[str, int] = {}
        self._tracking_generation: dict[str, int] = {}
        self._viewer_last: dict[str, float] = {}
        self.viewer_linger = 5.0
        self._lock = threading.Lock()
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._marker: Path | None = None    # <output_dir>/frames.json, ours to remove
        self.port = 0
        self.published = 0
        # One extra loopback port per camera (W1 wall). A browser opens at
        # most six connections to one host:port, and every MJPEG tile holds
        # one open for as long as it is watched — so on one port the seventh
        # tile waited for one of the first six to close, which on a wall
        # never happens (CHI, 11 Oct 2026: 7 of 12 cameras stream HEVC, which
        # Chromium cannot play over WebRTC, so they all took this path). Same
        # handler, same token, same frames; only the port differs, and the
        # shared port keeps serving everything for any client that uses it.
        self._handler: Any = None
        self._camera_servers: dict[str, Any] = {}
        self.camera_ports: dict[str, int] = {}

    # --- engine side ------------------------------------------------------
    def publish(self, camera_id: str, frame: Any,
                overlays: Sequence[FrameOverlay | tuple] = (), *,
                object_tracks: dict | None = None,
                object_generation: int | None = None) -> None:
        """Store raw glass and, on demand, an annotated variant of the same frame."""
        import cv2
        publication, current_object_generation = self._begin_publish(camera_id)
        if object_generation is None:
            object_generation = current_object_generation
        overlays, object_expiry = self._validated_overlays(overlays, object_tracks)
        img = frame
        h, w = img.shape[:2]
        if self.max_width and w > self.max_width:      # the wall is small; don't ship 1080p
            scale = self.max_width / float(w)
            img = cv2.resize(img, (self.max_width, max(1, int(h * scale))))
        else:
            scale = 1.0
        ok, raw_buf = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        if not ok:
            return

        with self._lock:
            tracking_watched = self._tracking_viewers.get(camera_id, 0) > 0
            tracking_generation = self._tracking_generation.get(camera_id, 0)
            alerting = set(self._alerting.get(camera_id, set()))

        tracking_jpeg = None
        person_tracking_jpeg = None
        if tracking_watched:
            person_overlays = self._person_overlays(overlays)
            object_overlays = self._object_overlays(overlays)
            if not self.draw_boxes:
                tracking_jpeg = raw_buf.tobytes()
                person_tracking_jpeg = tracking_jpeg
            elif not object_overlays:
                tracking_jpeg = (
                    self._encode_tracking(img, person_overlays, scale, alerting)
                    if person_overlays else raw_buf.tobytes()
                )
                person_tracking_jpeg = tracking_jpeg
            else:
                person_tracking_jpeg = (
                    self._encode_tracking(img, person_overlays, scale, alerting)
                    if person_overlays else raw_buf.tobytes()
                )
                tracking_jpeg = self._encode_tracking(img, overlays, scale, alerting)
        self._commit(
            camera_id, raw_buf.tobytes(), tracking_jpeg, overlays,
            tracking_generation if tracking_watched else None, publication,
            object_tracks, person_tracking_jpeg, object_generation,
            object_expiry,
        )

    def publish_jpeg(self, camera_id: str, jpeg: bytes,
                     overlays: Sequence[FrameOverlay | tuple] = (),
                     source_size: tuple[int, int] | None = None, *,
                     object_tracks: dict | None = None,
                     object_generation: int | None = None) -> None:
        """Store paced raw bytes; decode a tracking copy only while requested."""
        publication, current_object_generation = self._begin_publish(camera_id)
        if object_generation is None:
            object_generation = current_object_generation
        overlays, object_expiry = self._validated_overlays(overlays, object_tracks)
        with self._lock:
            tracking_watched = self._tracking_viewers.get(camera_id, 0) > 0
            tracking_generation = self._tracking_generation.get(camera_id, 0)
            alerting = set(self._alerting.get(camera_id, set()))

        tracking_jpeg = None
        person_tracking_jpeg = None
        if tracking_watched:
            if not self.draw_boxes or not overlays:
                tracking_jpeg = jpeg
            else:
                import cv2
                import numpy as np
                image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    scale = (image.shape[1] / float(source_size[1])
                             if source_size and source_size[1] else 1.0)
                    person_overlays = self._person_overlays(overlays)
                    if not self._object_overlays(overlays):
                        tracking_jpeg = self._encode_tracking(
                            image, person_overlays, scale, alerting
                        )
                        person_tracking_jpeg = tracking_jpeg
                    else:
                        tracking_jpeg = self._encode_tracking(
                            image, overlays, scale, alerting
                        )
                        person_tracking_jpeg = (
                            self._encode_tracking(image, person_overlays, scale, alerting)
                            if person_overlays else jpeg
                        )
            if person_tracking_jpeg is None:
                person_tracking_jpeg = jpeg

        self._commit(
            camera_id, jpeg, tracking_jpeg, overlays,
            tracking_generation if tracking_watched else None, publication,
            object_tracks, person_tracking_jpeg, object_generation,
            object_expiry,
        )

    def _begin_publish(self, camera_id: str) -> tuple[int, int]:
        with self._lock:
            publication = self._issued_seq.get(camera_id, 0) + 1
            self._issued_seq[camera_id] = publication
            return publication, self._object_generation.get(camera_id, 0)

    def object_generation(self, camera_id: str) -> int:
        """Token callers capture before collecting object overlays/snapshots."""
        with self._lock:
            return self._object_generation.get(camera_id, 0)

    def _commit(self, camera_id: str, raw_jpeg: bytes,
                tracking_jpeg: bytes | None,
                overlays: Sequence[FrameOverlay | tuple],
                tracking_generation: int | None, publication: int,
                object_tracks: dict | None,
                person_tracking_jpeg: bytes | None,
                object_generation: int,
                object_expiry: float | None) -> None:
        with self._lock:
            if publication <= self._raw_seq.get(camera_id, 0):
                return
            object_current = (
                self._object_generation.get(camera_id, 0) == object_generation
            )
            if not object_current:
                # A source reset won while this JPEG was encoding. Publish the
                # raw/person frame, but never resurrect its old-source objects.
                tracking_jpeg = person_tracking_jpeg
                overlays = self._person_overlays(overlays)
            sequence = publication
            self._raw_frames[camera_id] = raw_jpeg
            self._raw_seq[camera_id] = sequence
            tracking_current = (
                tracking_generation is not None
                and self._tracking_generation.get(camera_id, 0)
                    == tracking_generation
                and self._tracking_viewers.get(camera_id, 0) > 0
            )
            if tracking_current and tracking_jpeg is not None:
                self._tracking_frames[camera_id] = tracking_jpeg
                if person_tracking_jpeg is not None:
                    self._tracking_person_frames[camera_id] = person_tracking_jpeg
                self._tracking_seq[camera_id] = sequence
            else:
                self._tracking_frames.pop(camera_id, None)
                self._tracking_person_frames.pop(camera_id, None)
                self._tracking_seq.pop(camera_id, None)
            self._tracks[camera_id] = [
                int(o.track_id if isinstance(o, FrameOverlay) else o[0])
                for o in overlays
                if not isinstance(o, FrameOverlay) or o.namespace == "person"
            ]
            if object_current:
                self._object_tracks[camera_id] = copy.deepcopy(object_tracks)
                if object_expiry is None:
                    self._object_overlay_expires_at.pop(camera_id, None)
                else:
                    self._object_overlay_expires_at[camera_id] = object_expiry
            self.published += 1

    @staticmethod
    def _person_overlays(overlays: Sequence[FrameOverlay | tuple]) -> list:
        return [overlay for overlay in overlays
                if not isinstance(overlay, FrameOverlay)
                or overlay.namespace in ("person", "person_detection")]

    @staticmethod
    def _object_overlays(overlays: Sequence[FrameOverlay | tuple]) -> list:
        return [overlay for overlay in overlays
                if isinstance(overlay, FrameOverlay)
                and overlay.namespace == "object"]

    @classmethod
    def _validated_overlays(
        cls, overlays: Sequence[FrameOverlay | tuple], snapshot: dict | None,
    ) -> tuple[list, float | None]:
        """Fail closed: every rendered object must have matching fresh evidence."""
        person = cls._person_overlays(overlays)
        objects = cls._object_overlays(overlays)
        if not objects:
            return list(overlays), None
        if not snapshot or snapshot.get("timestamp_clock") != "monotonic":
            return person, None
        now = time.monotonic()
        try:
            snapshot_at = float(snapshot["timestamp"])
            max_age = float((snapshot.get("freshness") or {})[
                "overlay_max_age_seconds"
            ])
            if not math.isfinite(snapshot_at) or snapshot_at > now \
                    or not math.isfinite(max_age) or max_age <= 0:
                return person, None
        except (KeyError, TypeError, ValueError):
            return person, None

        observed: dict[int, float] = {}
        tracks = snapshot.get("tracks", [])
        if not isinstance(tracks, (list, tuple)):
            return person, None
        for track in tracks:
            if not isinstance(track, dict):
                continue
            if track.get("state") != "observed":
                continue
            try:
                counter = int(str(track.get("id", "")).rsplit("/", 1)[-1])
                last_seen = float(track["last_seen"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(last_seen) and last_seen <= snapshot_at:
                observed[counter] = last_seen

        valid_objects = []
        deadlines = []
        for overlay in objects:
            counter = -overlay.track_id
            last_seen = observed.get(counter)
            if last_seen is None:
                continue
            deadline = last_seen + max_age
            if math.isfinite(deadline) and now <= deadline:
                valid_objects.append(overlay)
                deadlines.append(deadline)
        allowed = {id(overlay) for overlay in valid_objects}
        filtered = [
            overlay for overlay in overlays
            if not isinstance(overlay, FrameOverlay)
            or overlay.namespace != "object"
            or id(overlay) in allowed
        ]
        return filtered, min(deadlines) if deadlines else None

    def _expire_object_overlay_locked(self, camera_id: str, now: float) -> None:
        expiry = self._object_overlay_expires_at.get(camera_id)
        if expiry is None or now <= expiry:
            return
        person_frame = self._tracking_person_frames.get(camera_id)
        if person_frame is not None and camera_id in self._tracking_frames:
            self._tracking_frames[camera_id] = person_frame
            sequence = self._issued_seq.get(camera_id, 0) + 1
            self._issued_seq[camera_id] = sequence
            self._tracking_seq[camera_id] = sequence
        self._object_overlay_expires_at.pop(camera_id, None)

    @staticmethod
    def _project_object_snapshot(snapshot: dict | None, now: float) -> dict | None:
        projected = copy.deepcopy(snapshot)
        if not projected or projected.get("timestamp_clock") != "monotonic":
            return projected
        freshness = projected.get("freshness") or {}
        try:
            lost_after = float(freshness["lost_after_seconds"])
            retention = float(freshness["ended_retention_seconds"])
        except (KeyError, TypeError, ValueError):
            return projected
        kept = []
        for track in projected.get("tracks", []):
            last_seen = float(track.get("last_seen", now))
            if track.get("state") != "ended" and now - last_seen > lost_after:
                track["state"] = "ended"
                track["ended_at"] = last_seen + lost_after
                track["end_reason"] = "lost_timeout"
            ended_at = track.get("ended_at")
            if track.get("state") != "ended" or ended_at is None \
                    or now - float(ended_at) <= retention:
                kept.append(track)
        projected["tracks"] = kept
        last_success = projected.get("last_success_at")
        if last_success is not None and now - float(last_success) > lost_after:
            projected["status"] = "stale"
        projected["timestamp"] = now
        return projected

    def invalidate_object_tracking(self, camera_id: str,
                                   object_tracks: dict | None = None) -> None:
        """Atomically replace metadata and remove only object annotations."""
        with self._lock:
            self._object_generation[camera_id] = \
                self._object_generation.get(camera_id, 0) + 1
            self._object_tracks[camera_id] = copy.deepcopy(object_tracks)
            self._object_overlay_expires_at.pop(camera_id, None)
            person_frame = self._tracking_person_frames.get(camera_id)
            if person_frame is not None and camera_id in self._tracking_frames:
                self._tracking_frames[camera_id] = person_frame
                sequence = self._issued_seq.get(camera_id, 0) + 1
                self._issued_seq[camera_id] = sequence
                self._tracking_seq[camera_id] = sequence

    def _encode_tracking(self, image: Any,
                         overlays: Sequence[FrameOverlay | tuple], scale: float,
                         alerting: set) -> bytes | None:
        import cv2
        annotated = image.copy()
        for overlay in overlays:
            if isinstance(overlay, FrameOverlay):
                tid = overlay.track_id
                x1, y1, x2, y2 = overlay.bbox
                label = overlay.label
                colour = overlay.colour
                namespace = overlay.namespace
            else:
                tid, x1, y1, x2, y2 = overlay
                label = f"#{tid}"
                colour = _BOX_COLOUR
                namespace = "person"
            if namespace == "person" and tid in alerting:
                colour = _ALERT_COLOUR
            p1 = (int(x1 * scale), int(y1 * scale))
            p2 = (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(annotated, p1, p2, colour, 2)
            cv2.putText(
                annotated, label, (p1[0] + 3, max(12, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
            )
        ok, buf = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        return buf.tobytes() if ok else None

    def mark_alerting(self, camera_id: str, track_ids) -> None:
        """Tracks to draw in alert colour (cleared by passing an empty set)."""
        with self._lock:
            self._alerting[camera_id] = set(track_ids or ())

    def _viewer_started(self, camera_id: str, tracking: bool = False) -> None:
        with self._lock:
            self._viewers[camera_id] = self._viewers.get(camera_id, 0) + 1
            if tracking:
                if self._tracking_viewers.get(camera_id, 0) == 0:
                    self._tracking_generation[camera_id] = \
                        self._tracking_generation.get(camera_id, 0) + 1
                    self._tracking_frames.pop(camera_id, None)
                    self._tracking_person_frames.pop(camera_id, None)
                    self._tracking_seq.pop(camera_id, None)
                self._tracking_viewers[camera_id] = \
                    self._tracking_viewers.get(camera_id, 0) + 1
            self._viewer_last[camera_id] = time.time()

    def _viewer_stopped(self, camera_id: str, tracking: bool = False) -> None:
        with self._lock:
            self._viewers[camera_id] = max(0, self._viewers.get(camera_id, 0) - 1)
            if tracking:
                previous = self._tracking_viewers.get(camera_id, 0)
                remaining = max(0, previous - 1)
                self._tracking_viewers[camera_id] = remaining
                if previous > 0 and remaining == 0:
                    self._tracking_generation[camera_id] = \
                        self._tracking_generation.get(camera_id, 0) + 1
                    self._tracking_frames.pop(camera_id, None)
                    self._tracking_person_frames.pop(camera_id, None)
                    self._tracking_seq.pop(camera_id, None)
            self._viewer_last[camera_id] = time.time()

    def has_tracking_viewers(self, camera_id: str) -> bool:
        with self._lock:
            return self._tracking_viewers.get(camera_id, 0) > 0

    def has_viewers(self, camera_id: str) -> bool:
        """True while someone holds this camera's stream open (or did within
        the linger window — a reconnecting tile must not flap the decoder)."""
        with self._lock:
            if self._viewers.get(camera_id, 0) > 0:
                return True
            last = self._viewer_last.get(camera_id, 0.0)
        return bool(last and (time.time() - last) < self.viewer_linger)

    def frame(self, camera_id: str, tracking: bool = False) -> bytes | None:
        with self._lock:
            if tracking:
                self._expire_object_overlay_locked(camera_id, time.monotonic())
            frames = self._tracking_frames if tracking else self._raw_frames
            return frames.get(camera_id)

    def frame_seq(self, camera_id: str, tracking: bool = False) -> tuple:
        """(jpeg, seq) — seq bumps per publish so a stream sends only new frames."""
        with self._lock:
            if tracking:
                self._expire_object_overlay_locked(camera_id, time.monotonic())
            frames = self._tracking_frames if tracking else self._raw_frames
            sequences = self._tracking_seq if tracking else self._raw_seq
            return frames.get(camera_id), sequences.get(camera_id, 0)

    def snapshot(self) -> dict:
        with self._lock:
            now = time.monotonic()
            for camera_id in tuple(self._object_overlay_expires_at):
                self._expire_object_overlay_locked(camera_id, now)
            return {"cameras": sorted(self._raw_frames),
                    "tracks": copy.deepcopy(self._tracks),
                    "object_tracks": {
                        camera_id: self._project_object_snapshot(snapshot, now)
                        for camera_id, snapshot in self._object_tracks.items()
                    },
                    "published": self.published}

    # --- server -----------------------------------------------------------
    def start(self, output_dir: str | Path | None = None,
              cameras: Sequence[str] = ()) -> "FramePublisher":
        """Serve. `cameras` get a port each (see camera_ports); the rest use
        the shared one. frames.json is written once every port is listening,
        so a reader never sees a half-made map."""
        pub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):        # keep the engine log readable
                return

            # HTTP/1.1, not the BaseHTTPRequestHandler default of 1.0. A
            # browser will not progressively render multipart/x-mixed-replace
            # over HTTP/1.0 — the live wall showed broken-image icons on every
            # tile while the engine was publishing perfectly (27 Aug). A raw
            # socket client reads it either way, which is exactly why a test
            # that was not a browser missed this. Safe because every non-stream
            # response below sends Content-Length.
            protocol_version = "HTTP/1.1"

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                # No wildcard CORS. This serves live camera frames; letting any
                # origin read them is the browser-side version of leaving the
                # port open.
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass                        # viewer navigated away mid-write

            def _authorised(self) -> bool:
                """Every route. There is no unauthenticated path to a frame."""
                supplied = self.headers.get("X-Argus-Token", "")
                if not supplied:
                    from urllib.parse import parse_qs, urlparse
                    supplied = (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
                # Constant-time: this endpoint is reachable from the network the
                # moment EP-06 exposes it.
                return hmac.compare_digest(supplied, pub.token)

            def do_GET(self):                   # noqa: N802 - BaseHTTPRequestHandler API
                path = self.path.split("?", 1)[0]
                if not self._authorised():
                    log.warning("[frames] unauthenticated request to %s from %s",
                                path, self.client_address[0])
                    self._send(401, b"unauthorised", "text/plain")
                    return
                if path.startswith("/stream/"):
                    # MJPEG (multipart/x-mixed-replace): ONE long-lived
                    # connection per camera, frames pushed as they are
                    # published. The UI used to poll /frame every 90ms —
                    # 11 requests/second/camera, each a fresh HTTP round-trip,
                    # JPEG decode and repaint, which is what "the stream is
                    # sluggish" was. The browser decodes this natively with no
                    # JavaScript in the loop at all. (26 Aug.)
                    cam = unquote(path[len("/stream/"):])
                    tracking = (parse_qs(urlparse(self.path).query).get("tracking")
                                or [""])[0].lower() in ("1", "true")
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=argusframe")
                    self.send_header("Cache-Control", "no-store")
                    # Never keep-alive a response that has no end.
                    self.send_header("Connection", "close")
                    self.end_headers()
                    last = -1
                    last_probe = time.monotonic()
                    pub._viewer_started(cam, tracking)   # an open stream IS a viewer
                    try:
                        while True:
                            data, seq = pub.frame_seq(cam, tracking)
                            if data is not None and seq != last:
                                last = seq
                                self.wfile.write(b"--argusframe\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(
                                    f"Content-Length: {len(data)}\r\n\r\n".encode())
                                self.wfile.write(data)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                            else:
                                time.sleep(0.02)
                                # A closed viewer only surfaces on the next
                                # WRITE — and a camera with no fresh frames
                                # never writes, leaving a ghost viewer pinning
                                # the display boost forever. The client never
                                # sends after its request, so readable == EOF:
                                # probe once a second.
                                if time.monotonic() - last_probe >= 1.0:
                                    last_probe = time.monotonic()
                                    ready, _, _ = select.select(
                                        [self.connection], [], [], 0)
                                    if ready and not self.connection.recv(
                                            1, socket.MSG_PEEK):
                                        break          # viewer hung up
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass          # the viewer navigated away; drop the stream
                    finally:
                        pub._viewer_stopped(cam, tracking)
                    return
                if path.startswith("/frame/"):
                    cam = unquote(path[len("/frame/"):])
                    data = pub.frame(cam)
                    if data is None:
                        self._send(404, b"no frame", "text/plain")
                    else:
                        self._send(200, data, "image/jpeg")
                elif path == "/health":
                    if pub.health_provider is None:
                        self._send(404, b"no health provider", "text/plain")
                    else:
                        try:
                            doc = pub.health_provider()
                            self._send(200, json.dumps(doc, default=str).encode(),
                                       "application/json")
                        except Exception:  # noqa: BLE001 - a health probe must not kill the server
                            log.error("[frames] /health provider failed", exc_info=True)
                            self._send(500, b"health provider failed", "text/plain")
                elif path in ("/cameras", "/"):
                    self._send(200, json.dumps(pub.snapshot()).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

        self._handler = Handler
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="frame-publisher", daemon=True)
        self._thread.start()
        for camera_id in cameras:
            self._ensure_camera_server(str(camera_id))
        if output_dir:                          # so the app can find us
            self._write_marker(Path(output_dir) / "frames.json")
        log.info(f"[frames] publishing on http://127.0.0.1:{self.port} "
              f"(boxes={'on' if self.draw_boxes else 'off'}, "
              f"{len(self.camera_ports)} per-camera port(s))")
        return self

    def _ensure_camera_server(self, camera_id: str) -> int:
        with self._lock:
            port = self.camera_ports.get(camera_id)
        if port:
            return port
        server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        threading.Thread(target=server.serve_forever,
                         name=f"frame-publisher:{camera_id}", daemon=True).start()
        with self._lock:
            self._camera_servers[camera_id] = server
            self.camera_ports[camera_id] = server.server_address[1]
            return self.camera_ports[camera_id]

    def port_for(self, camera_id: str) -> int:
        """The port this camera's tile should open; the shared port when the
        camera was not given one of its own."""
        return self.camera_ports.get(camera_id) or self.port

    def _write_marker(self, path: Path) -> None:
        self._marker = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"port": self.port, "token": self.token,
                                    "ports": dict(self.camera_ports)}))
        try:
            # The token is a credential. Anyone who can read this file can
            # read the cameras, so do not let that be everyone on the box.
            os.chmod(path, 0o600)
        except OSError:
            log.warning("[frames] could not restrict permissions on %s", path,
                        exc_info=True)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with self._lock:
            servers, self._camera_servers = self._camera_servers, {}
            self.camera_ports = {}
        for server in servers.values():
            try:
                server.shutdown()
                server.server_close()
            except OSError:
                pass
        # Like stream_gateway.json: the file existing IS the publisher being
        # alive. It used to outlive the engine, and every run listens on a new
        # port — so after Stop the app resolved a dead URL and every tile read
        # "fallback stream could not be loaded" until something remounted it
        # (12 Sep). Remove it, and the app gets an honest 503 instead.
        if self._marker is not None:
            try:
                self._marker.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                log.warning("[frames] could not remove %s", self._marker, exc_info=True)
            self._marker = None
