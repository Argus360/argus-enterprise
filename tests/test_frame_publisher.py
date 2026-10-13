"""The engine publishes frames so the UI never decodes the same stream twice.

Decode is the dominant per-camera cost, and the app used to pay it a second time
just to display. It also had no detection state, so it could never draw a box.
These cover the transport and the fallback — the app must still work when no
engine is running.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving import frame_publisher
from cvti.serving.frame_publisher import FramePublisher

from _backend_helper import signed_in

JPEG_MAGIC = b"\xff\xd8"


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.pub = FramePublisher(max_width=320).start(self.d)
        self.frame = np.full((480, 640, 3), 60, np.uint8)

    def tearDown(self):
        self.pub.stop()

    def _get(self, path, token=None):
        """Every route authenticates — this serves live camera frames."""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.pub.port}{path}",
            headers={"X-Argus-Token": self.pub.token if token is None else token})
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.read()

    def _status(self, path, token=None):
        url = f"http://127.0.0.1:{self.pub.port}{path}"
        headers = {} if token is None else {"X-Argus-Token": token}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=3) as r:
                return r.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_serves_the_latest_frame_as_jpeg(self):
        self.pub.publish("cam1", self.frame, [(7, 100, 50, 200, 300)])
        body = self._get("/frame/cam1")
        self.assertEqual(body[:2], JPEG_MAGIC)

    def test_publishes_port_so_the_app_can_find_it(self):
        self.assertEqual(json.loads((self.d / "frames.json").read_text(encoding="utf-8"))["port"], self.pub.port)

    def test_cameras_endpoint_lists_cameras_and_tracks(self):
        self.pub.publish("cam1", self.frame, [(7, 10, 10, 50, 90), (9, 60, 10, 90, 90)])
        meta = json.loads(self._get("/cameras"))
        self.assertEqual(meta["cameras"], ["cam1"])
        self.assertEqual(meta["tracks"]["cam1"], [7, 9])

    def test_unknown_camera_is_404_not_a_crash(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/frame/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_tracking_boxes_change_the_image_and_never_mutate_the_source(self):
        original = self.frame.copy()
        self.pub._viewer_started("a", tracking=True)
        self.addCleanup(self.pub._viewer_stopped, "a", True)
        self.pub.publish("a", self.frame, [(1, 10, 10, 100, 200)])
        self.assertNotEqual(self.pub.frame("a", tracking=True), self.pub.frame("a"))
        self.assertTrue(np.array_equal(self.frame, original))

    def test_unassociated_person_box_renders_without_publishing_a_track_id(self):
        overlay = frame_publisher.FrameOverlay(
            -1, (10, 10, 100, 200), "PERSON", (200, 180, 100), "person_detection")
        self.pub._viewer_started("a", tracking=True)
        self.addCleanup(self.pub._viewer_stopped, "a", True)
        self.assertEqual(self.pub._person_overlays([overlay]), [overlay])
        self.pub.publish("a", self.frame, [overlay])
        self.assertNotEqual(self.pub.frame("a", tracking=True), self.pub.frame("a"))
        self.assertEqual(self.pub._tracks["a"], [])

    def test_raw_and_tracking_frames_are_built_from_the_same_source_on_demand(self):
        self.assertTrue(hasattr(frame_publisher, "FrameOverlay"))
        overlay = frame_publisher.FrameOverlay(
            track_id=7,
            bbox=(40, 40, 200, 260),
            label="#7 MOVING",
            colour=(0, 255, 0),
        )
        original = self.frame.copy()

        self.pub.publish("cam1", self.frame, [overlay])
        self.assertIsNone(self.pub.frame("cam1", tracking=True))

        self.pub._viewer_started("cam1", tracking=True)
        self.addCleanup(self.pub._viewer_stopped, "cam1", True)
        self.pub.publish("cam1", self.frame, [overlay])

        import cv2
        raw = cv2.imdecode(
            np.frombuffer(self.pub.frame("cam1"), np.uint8), cv2.IMREAD_COLOR
        )
        tracking = cv2.imdecode(
            np.frombuffer(self.pub.frame("cam1", tracking=True), np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertLess(np.abs(raw.astype(int) - 60).max(), 5)
        self.assertGreater(np.abs(tracking.astype(int) - 60).max(), 100)
        self.assertTrue(np.array_equal(self.frame, original))

        clean = np.full_like(self.frame, 80)
        self.pub.publish("cam1", clean, [])
        self.assertEqual(
            self.pub.frame("cam1", tracking=True),
            self.pub.frame("cam1"),
            "a stopped track left its old box frozen on the tracking stream",
        )

    def test_large_frames_are_downscaled_for_the_wall(self):
        self.pub.publish("big", np.full((1080, 1920, 3), 90, np.uint8), [])
        import cv2
        img = cv2.imdecode(np.frombuffer(self.pub.frame("big"), np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(img.shape[1], 320)

    def test_paced_jpeg_gets_an_on_demand_tracking_variant(self):
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg", cv2.resize(self.frame, (320, 240)),
            [cv2.IMWRITE_JPEG_QUALITY, self.pub.quality],
        )
        self.assertTrue(ok)
        raw = encoded.tobytes()
        overlay = frame_publisher.FrameOverlay(
            track_id=4,
            bbox=(80, 80, 400, 400),
            label="#4 MOVING",
            colour=(0, 200, 255),
        )
        self.pub._viewer_started("paced", tracking=True)
        self.addCleanup(self.pub._viewer_stopped, "paced", True)

        self.pub.publish_jpeg(
            "paced", raw, [overlay], source_size=(480, 640)
        )

        self.assertEqual(self.pub.frame("paced"), raw)
        self.assertNotEqual(self.pub.frame("paced", tracking=True), raw)

    def test_tracking_cache_is_invalidated_across_viewer_sessions(self):
        overlay = frame_publisher.FrameOverlay(
            track_id=8,
            bbox=(40, 40, 200, 260),
            label="#8 MOVING",
            colour=(0, 200, 255),
        )
        self.pub._viewer_started("cam1", tracking=True)
        self.pub.publish("cam1", self.frame, [overlay])
        self.assertIsNotNone(self.pub.frame("cam1", tracking=True))

        self.pub._viewer_stopped("cam1", tracking=True)
        self.assertEqual(self.pub.frame_seq("cam1", tracking=True), (None, 0))
        self.pub.publish("cam1", np.full_like(self.frame, 70), [])
        self.pub.publish("cam1", np.full_like(self.frame, 80), [])

        self.pub._viewer_started("cam1", tracking=True)
        self.addCleanup(self.pub._viewer_stopped, "cam1", True)
        self.assertEqual(
            self.pub.frame_seq("cam1", tracking=True),
            (None, 0),
            "a new tracking session received the prior session's annotation",
        )
        self.pub.publish("cam1", np.full_like(self.frame, 90), [overlay])
        self.assertIsNotNone(self.pub.frame("cam1", tracking=True))
        _, raw_sequence = self.pub.frame_seq("cam1")
        _, tracking_sequence = self.pub.frame_seq("cam1", tracking=True)
        self.assertEqual(tracking_sequence, raw_sequence)

    def test_disconnected_generation_rejects_in_flight_annotation(self):
        import threading

        overlay = frame_publisher.FrameOverlay(
            track_id=10,
            bbox=(40, 40, 200, 260),
            label="#10 MOVING",
            colour=(0, 200, 255),
        )
        encoding = threading.Event()
        release = threading.Event()
        real_encode = self.pub._encode_tracking

        def blocked_encode(*args):
            encoding.set()
            self.assertTrue(release.wait(2))
            return real_encode(*args)

        self.pub._viewer_started("cam1", tracking=True)
        self.pub._encode_tracking = blocked_encode
        worker = threading.Thread(
            target=self.pub.publish, args=("cam1", self.frame, [overlay])
        )
        worker.start()
        self.assertTrue(encoding.wait(2))
        self.pub._viewer_stopped("cam1", tracking=True)
        self.pub._viewer_started("cam1", tracking=True)
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.addCleanup(self.pub._viewer_stopped, "cam1", True)

        self.assertEqual(self.pub.frame_seq("cam1", tracking=True), (None, 0))

    def test_paced_annotation_failure_commits_raw_and_invalidates_tracking(self):
        import cv2

        def jpeg(value):
            ok, encoded = cv2.imencode(
                ".jpg", np.full((240, 320, 3), value, np.uint8)
            )
            self.assertTrue(ok)
            return encoded.tobytes()

        overlay = frame_publisher.FrameOverlay(
            track_id=11,
            bbox=(20, 20, 120, 160),
            label="#11 MOVING",
            colour=(0, 200, 255),
        )
        first_raw = jpeg(30)
        next_raw = jpeg(100)
        self.pub._viewer_started("paced", tracking=True)
        self.addCleanup(self.pub._viewer_stopped, "paced", True)
        self.pub.publish_jpeg(
            "paced", first_raw, [overlay], source_size=(240, 320)
        )
        self.assertIsNotNone(self.pub.frame("paced", tracking=True))

        def fail_annotation(*_args):
            self.assertEqual(
                self.pub.frame("paced"), first_raw,
                "raw advanced before its tracking variant was prepared",
            )
            return None

        self.pub._encode_tracking = fail_annotation
        self.pub.publish_jpeg(
            "paced", next_raw, [overlay], source_size=(240, 320)
        )

        self.assertEqual(self.pub.frame("paced"), next_raw)
        self.assertEqual(self.pub.frame_seq("paced", tracking=True), (None, 0))

    def test_alerting_tracks_are_coloured_differently(self):
        self.pub._viewer_started("c", tracking=True)
        self.addCleanup(self.pub._viewer_stopped, "c", True)
        self.pub.publish("c", self.frame, [(3, 20, 20, 120, 220)])
        normal = self.pub.frame("c", tracking=True)
        self.pub.mark_alerting("c", {3})
        self.pub.publish("c", self.frame, [(3, 20, 20, 120, 220)])
        self.assertNotEqual(normal, self.pub.frame("c", tracking=True))


    def test_late_completion_cannot_overwrite_a_newer_same_session_publish(self):
        import threading

        pub = FramePublisher(max_width=0)
        pub._viewer_started("cam1", tracking=True)
        first_started = threading.Event()
        release_first = threading.Event()
        real_encode = pub._encode_tracking

        def ordered_encode(image, *args):
            if int(image[0, 0, 0]) == 10:
                first_started.set()
                self.assertTrue(release_first.wait(2))
            return real_encode(image, *args)

        pub._encode_tracking = ordered_encode
        overlay = frame_publisher.FrameOverlay(1, (5, 5, 30, 30), "#1", (0, 255, 0))
        older = threading.Thread(
            target=pub.publish,
            args=("cam1", np.full((48, 64, 3), 10, np.uint8), [overlay]),
        )
        newer = threading.Thread(
            target=pub.publish,
            args=("cam1", np.full((48, 64, 3), 200, np.uint8), [overlay]),
        )

        older.start()
        self.assertTrue(first_started.wait(2))
        newer.start()
        newer.join(2)
        release_first.set()
        older.join(2)
        self.assertFalse(older.is_alive())
        self.assertFalse(newer.is_alive())

        import cv2
        raw = cv2.imdecode(np.frombuffer(pub.frame("cam1"), np.uint8), cv2.IMREAD_COLOR)
        tracking = cv2.imdecode(
            np.frombuffer(pub.frame("cam1", tracking=True), np.uint8), cv2.IMREAD_COLOR
        )
        self.assertGreater(float(raw.mean()), 180.0)
        self.assertGreater(float(tracking.mean()), 150.0)
        self.assertEqual(
            pub.frame_seq("cam1")[1], pub.frame_seq("cam1", tracking=True)[1]
        )


    # --- EP-03-T1: no unauthenticated route to a camera ---------------------
    def test_every_route_rejects_an_unauthenticated_request(self):
        self.pub.publish("cam1", self.frame, [])
        for path in (
            "/frame/cam1",
            "/stream/cam1",
            "/stream/cam1?tracking=1",
            "/cameras",
            "/",
            "/frame/nonexistent",
            "/anything",
        ):
            self.assertEqual(self._status(path), 401,
                             f"{path} served without a token")

    def test_a_wrong_token_is_rejected(self):
        self.pub.publish("cam1", self.frame, [])
        self.assertEqual(self._status("/frame/cam1", "not-the-token"), 401)
        self.assertEqual(self._status("/stream/cam1", "not-the-token"), 401)
        self.assertEqual(
            self._status("/stream/cam1?tracking=1", "not-the-token"), 401
        )

    def test_the_token_also_works_as_a_query_parameter(self):
        # An <img> tag cannot set a header.
        self.pub.publish("cam1", self.frame, [])
        self.assertEqual(self._status(f"/frame/cam1?token={self.pub.token}"), 200)

    def test_the_token_is_random_per_publisher(self):
        from cvti.serving.frame_publisher import FramePublisher
        self.assertNotEqual(FramePublisher().token, FramePublisher().token)

    def test_frames_json_carries_the_token_and_is_not_world_readable(self):
        import json
        import os
        import tempfile
        from pathlib import Path

        from cvti.serving.frame_publisher import FramePublisher
        with tempfile.TemporaryDirectory() as tmp:
            pub = FramePublisher().start(tmp)
            try:
                info = json.loads((Path(tmp) / "frames.json").read_text(encoding="utf-8"))
                self.assertEqual(info["token"], pub.token)
                if os.name == "nt":
                    self.skipTest("NTFS has no POSIX mode bits; the file lives in "
                                  "the per-user output dir, where ACLs do the job")
                mode = os.stat(Path(tmp) / "frames.json").st_mode & 0o777
                self.assertEqual(mode, 0o600, "the frame token was world-readable")
            finally:
                pub.stop()

    def test_no_wildcard_cors_on_camera_frames(self):
        # Letting any origin read these is the browser-side open port.
        self.pub.publish("cam1", self.frame, [])
        req = urllib.request.Request(f"http://127.0.0.1:{self.pub.port}/frame/cam1",
                                     headers={"X-Argus-Token": self.pub.token})
        with urllib.request.urlopen(req, timeout=3) as r:
            self.assertIsNone(r.headers.get("Access-Control-Allow-Origin"))

class AppFallbackTests(unittest.TestCase):
    def test_app_decodes_for_itself_when_no_engine_is_publishing(self):
        from cvti.app.console_backend import ConsoleBackend
        d = Path(tempfile.mkdtemp())
        be = signed_in(site_path=str(d / "s.json"), db_path=str(d / "e.db"),
                            enable_demo=False)
        self.assertEqual(be._engine_frame_port(), (0, ""))   # nothing published
        res = be.live_start(2)
        try:
            # falls back to its own decode rather than showing nothing
            self.assertIn(res.get("source"), ("app", None))
        finally:
            be.live_stop()

    def test_stale_port_file_is_not_trusted(self):
        from cvti.app.console_backend import ConsoleBackend
        d = Path(tempfile.mkdtemp())
        (d / "frames.json").write_text(json.dumps({"port": 9}))   # nothing listening
        be = signed_in(site_path=str(d / "s.json"), db_path=str(d / "e.db"),
                            enable_demo=False)
        self.assertEqual(be._engine_frame_port(), (0, ""))


if __name__ == "__main__":
    unittest.main()


class TokenThreadingTests(unittest.TestCase):
    """The demo-day broken-tile bug: EP-03 put a token on every frame route,
    but live_start's probe checked /cameras WITHOUT it (so engine frames were
    never used) and neither return path handed the UI a token (so every <img>
    got 401). The UI cannot render what it cannot authenticate to."""

    def test_engine_probe_sends_the_token_and_returns_it(self):
        from cvti.serving.frame_publisher import FramePublisher
        import numpy as np
        d = Path(tempfile.mkdtemp())
        pub = FramePublisher().start(d)
        try:
            pub.publish("cam1", np.zeros((8, 8, 3), np.uint8), [])
            be = signed_in(site_path=str(d / "s.json"), db_path=str(d / "e.db"),
                           enable_demo=False)
            port, token = be._engine_frame_port()
            self.assertEqual(port, pub.port)
            self.assertEqual(token, pub.token, "the UI is useless without the token")
        finally:
            pub.stop()

    def test_live_start_app_path_returns_the_frameserver_token(self):
        import inspect
        from cvti.app.console_backend import ConsoleBackend
        src = inspect.getsource(ConsoleBackend.live_start)
        self.assertIn('"token"', src)
        self.assertIn("self._fs.token", src)


class MjpegStreamTests(unittest.TestCase):
    """The live wall streams instead of polling (26 Aug). The UI used to fire
    11 requests/second/camera at /frame — a fresh HTTP round-trip, JPEG decode
    and repaint each time, which is what 'the stream is sluggish' was."""

    def setUp(self):
        import numpy as np
        from cvti.serving.frame_publisher import FramePublisher
        self.d = Path(tempfile.mkdtemp())
        self.pub = FramePublisher(max_width=64).start(self.d)
        self.np = np
        self.addCleanup(self.pub.stop)

    def _url(self, path):
        return f"http://127.0.0.1:{self.pub.port}{path}?token={self.pub.token}"

    def test_the_stream_pushes_successive_frames(self):
        import threading
        import urllib.request
        self.pub.publish("cam1", self.np.zeros((48, 64, 3), self.np.uint8))

        chunks = []

        def read_some():
            r = urllib.request.urlopen(self._url("/stream/cam1"), timeout=10)
            self.assertIn("multipart/x-mixed-replace", r.headers["Content-Type"])
            deadline = time.time() + 6
            while time.time() < deadline and len(chunks) < 2:
                data = r.read(4096)
                if not data:
                    break
                if b"\xff\xd8" in data:
                    chunks.append(data)
            r.close()

        t = threading.Thread(target=read_some, daemon=True)
        t.start()
        for i in range(1, 12):          # keep publishing new frames
            time.sleep(0.15)
            img = self.np.full((48, 64, 3), i * 10, self.np.uint8)
            self.pub.publish("cam1", img)
        t.join(timeout=8)
        self.assertGreaterEqual(len(chunks), 2,
                                "the MJPEG stream sent fewer than two frames")

    def test_the_stream_is_authenticated_like_every_other_route(self):
        import urllib.error
        import urllib.request
        self.pub.publish("cam1", self.np.zeros((48, 64, 3), self.np.uint8))
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.pub.port}/stream/cam1", timeout=5)
        self.assertEqual(ctx.exception.code, 401,
                         "the stream route served video without a token")

    def test_the_stream_speaks_http_1_1(self):
        """Chromium — which QtWebEngine is — will not progressively render
        multipart/x-mixed-replace over HTTP/1.0. Every tile showed a broken
        image while the engine published perfectly (27 Aug). A socket client
        reads it either way, which is why the earlier test missed it."""
        import urllib.request
        self.pub.publish("cam1", self.np.zeros((48, 64, 3), self.np.uint8))
        r = urllib.request.urlopen(self._url("/stream/cam1"), timeout=5)
        try:
            self.assertEqual(r.version, 11,
                             "HTTP/1.0 multipart — a browser will show a broken image")
            self.assertEqual(r.headers.get("Connection", "").lower(), "close",
                             "a response with no end must not be kept alive")
        finally:
            r.close()

    def test_normal_responses_still_carry_content_length(self):
        # HTTP/1.1 without Content-Length would hang a keep-alive client.
        import urllib.request
        r = urllib.request.urlopen(self._url("/cameras"), timeout=5)
        try:
            self.assertTrue(r.headers.get("Content-Length"))
        finally:
            r.close()

    def test_a_publish_bumps_the_sequence(self):
        self.pub.publish("cam1", self.np.zeros((48, 64, 3), self.np.uint8))
        _, s1 = self.pub.frame_seq("cam1")
        self.pub.publish("cam1", self.np.ones((48, 64, 3), self.np.uint8))
        _, s2 = self.pub.frame_seq("cam1")
        self.assertGreater(s2, s1, "streams would never send a second frame")

    def test_raw_and_tracking_viewer_counts_are_released_on_disconnect(self):
        import urllib.request

        self.pub.viewer_linger = 0
        self.pub.publish("cam1", self.np.zeros((48, 64, 3), self.np.uint8))
        raw = urllib.request.urlopen(self._url("/stream/cam1"), timeout=5)
        tracking = urllib.request.urlopen(
            self._url("/stream/cam1") + "&tracking=1", timeout=5
        )
        deadline = time.time() + 2
        while time.time() < deadline and (
            self.pub._viewers.get("cam1") != 2
            or self.pub._tracking_viewers.get("cam1") != 1
        ):
            time.sleep(0.02)
        self.assertEqual(self.pub._viewers.get("cam1"), 2)
        self.assertEqual(self.pub._tracking_viewers.get("cam1"), 1)

        raw.close()
        tracking.close()
        deadline = time.time() + 3
        while time.time() < deadline and self.pub._viewers.get("cam1"):
            time.sleep(0.05)
        self.assertEqual(self.pub._viewers.get("cam1"), 0)
        self.assertEqual(self.pub._tracking_viewers.get("cam1"), 0)


class UiStreamWiringTests(unittest.TestCase):
    UI = (Path(__file__).resolve().parents[1] / "cvti/app/web/index.html").read_text(encoding="utf-8")

    def test_tiles_use_the_stream_not_a_polled_still(self):
        self.assertIn("streamUrl(c.id)", self.UI)
        live = self.UI.split("function startLivePoll")[1].split("\nfunction ")[0]
        self.assertNotIn("img.src=frameUrl", live,
                         "the wall still re-fetches a still image on a timer")

    def test_leaving_the_wall_closes_the_streams(self):
        stop = self.UI.split("function stopLive()")[1].split("\nfunction ")[0]
        self.assertIn('img.src=""', stop,
                      "an <img> on an MJPEG stream holds the socket (and an "
                      "engine thread) open forever")


class PerCameraPortTests(unittest.TestCase):
    """W1 wall: a browser allows six connections per host:port and every
    MJPEG tile holds one. Each camera the engine names gets a port of its
    own; the shared port still serves everything."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.pub = FramePublisher(draw_boxes=False).start(self.d, cameras=["cam1", "cam2"])
        self.frame = np.full((48, 64, 3), 60, np.uint8)

    def tearDown(self):
        self.pub.stop()

    def _get(self, port, path):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     headers={"X-Argus-Token": self.pub.token})
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.read()

    def test_every_named_camera_gets_a_distinct_port(self):
        ports = self.pub.camera_ports
        self.assertEqual(set(ports), {"cam1", "cam2"})
        self.assertEqual(len({self.pub.port, *ports.values()}), 3)
        self.assertEqual(self.pub.port_for("cam1"), ports["cam1"])
        self.assertEqual(self.pub.port_for("unknown"), self.pub.port)

    def test_frames_json_carries_the_map_and_every_port_serves_frames(self):
        info = json.loads((self.d / "frames.json").read_text(encoding="utf-8"))
        self.assertEqual(info["ports"], self.pub.camera_ports)
        self.pub.publish("cam1", self.frame)
        self.assertEqual(self._get(info["ports"]["cam1"], "/frame/cam1")[:2], JPEG_MAGIC)
        self.assertEqual(self._get(info["port"], "/frame/cam1")[:2], JPEG_MAGIC)

    def test_stop_closes_the_extra_ports(self):
        port = self.pub.camera_ports["cam1"]
        self.pub.stop()
        self.assertEqual(self.pub.camera_ports, {})
        with self.assertRaises(OSError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/cameras", timeout=1)


class MarkerLifecycleTests(unittest.TestCase):
    def test_stop_removes_frames_json(self):
        # The file existing IS the publisher being alive (the contract
        # stream_gateway.json already keeps). It used to outlive the engine,
        # and every run listens on a new port — so after Stop the app resolved
        # a dead URL and every tile read "fallback stream could not be
        # loaded" (12 Sep).
        with tempfile.TemporaryDirectory() as d:
            pub = FramePublisher(draw_boxes=False).start(d)
            marker = Path(d) / "frames.json"
            self.assertTrue(marker.exists())
            pub.stop()
            self.assertFalse(marker.exists())
            pub.stop()          # idempotent: a second stop must not raise
