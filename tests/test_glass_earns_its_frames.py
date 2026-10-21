"""The live wall is smooth where eyes are, cheap where none are (3 Sep).

Smooth-publish used to run EVERY decoder at max(target, publish) fps around
the clock — 12fps of decode per camera for a wall nobody was watching — while
the operator still called the watched view laggy. The contract now:

- an open /stream connection on the frame publisher IS a viewer (every Watch
  tile holds one); no config, no heartbeat file, any client counts;
- a watched camera's decoder steps up to publish_fps (24 since 4 Sep — the
  wall spent a day capped at the CLI's forgotten default of 12 while every
  layer above could do more; the sustainable-fps cap still protects starved
  machines) within one frame period, and drops back to target_fps when the
  last viewer leaves (with a linger so a reconnecting tile doesn't flap the
  decoder);
- detection keeps sampling at target_fps regardless (the _due_for_detection
  gate) — smooth glass never means more model load.
"""
from __future__ import annotations

import socket
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.frame_publisher import FramePublisher
from cvti.serving.streams import StreamDecoder


class PacingTests(unittest.TestCase):
    def _decoder(self, target=4.0) -> StreamDecoder:
        d = StreamDecoder.__new__(StreamDecoder)   # pacing math needs no stream
        d.target_fps = target
        d.display_fps = 0.0
        d._fps = 30.0
        d.sustainable_fps = 30.0                   # adaptive ingest: no deficit
        return d

    def test_unwatched_paces_at_detection_rate(self):
        d = self._decoder()
        stride, period = d._pacing()
        self.assertEqual(stride, 8)                # 30fps source / 4fps target
        self.assertAlmostEqual(period, 0.25)

    def test_a_viewer_boost_changes_pacing_immediately(self):
        d = self._decoder()
        d.display_fps = 15.0
        stride, period = d._pacing()
        self.assertEqual(stride, 2)                # 30 / 15
        self.assertAlmostEqual(period, 1 / 15)

    def test_the_boost_never_slows_a_faster_target(self):
        d = self._decoder(target=20.0)
        d.display_fps = 15.0
        self.assertEqual(d._effective_fps(), 20.0)

    def test_clearing_the_boost_returns_to_target(self):
        d = self._decoder()
        d.display_fps = 15.0
        d.display_fps = 0.0
        self.assertEqual(d._effective_fps(), 4.0)


class ViewerTrackingTests(unittest.TestCase):
    def setUp(self):
        self.pub = FramePublisher(draw_boxes=False)
        self.pub.viewer_linger = 0.3               # fast tests; prod keeps 5s
        self.pub.start()
        self.pub.publish_jpeg("cam1", b"\xff\xd8fakejpeg")

    def tearDown(self):
        try:
            self.pub._server.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def _open_stream(self, cam: str) -> socket.socket:
        s = socket.create_connection(("127.0.0.1", self.pub.port), timeout=5)
        s.sendall((f"GET /stream/{cam}?token={self.pub.token} HTTP/1.1\r\n"
                   f"Host: 127.0.0.1\r\n\r\n").encode())
        s.recv(1024)                               # headers + first frame part
        return s

    def _wait(self, predicate, timeout=3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def test_an_open_stream_is_a_viewer(self):
        self.assertFalse(self.pub.has_viewers("cam1"))
        s = self._open_stream("cam1")
        try:
            self.assertTrue(self._wait(lambda: self.pub.has_viewers("cam1")),
                            "an open /stream connection did not register as a viewer")
        finally:
            s.close()

    def test_the_viewer_lingers_then_expires(self):
        s = self._open_stream("cam1")
        self.assertTrue(self._wait(lambda: self.pub.has_viewers("cam1")))
        s.close()
        # Within the linger window the boost holds (a reconnecting tile).
        self.assertTrue(self.pub.has_viewers("cam1"))
        self.assertTrue(self._wait(lambda: not self.pub.has_viewers("cam1"),
                                   timeout=3.0),
                        "the viewer never expired after disconnect + linger")

    def test_an_unwatched_camera_reports_no_viewers(self):
        self.assertFalse(self.pub.has_viewers("never-streamed"))


class WiringPins(unittest.TestCase):
    def test_decoders_start_at_detection_rate(self):
        import inspect
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.MultiStreamPipeline.start)
        # detection cameras start at detection rate; view-only glass idles at 1fps
        self.assertIn("target_fps=(1.0 if vo else self.target_fps)", src)
        self.assertNotIn("max(self.target_fps, self.publish_fps)", src)

    def test_the_smooth_loop_boosts_only_watched_cameras(self):
        import inspect
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.MultiStreamPipeline._smooth_publish_loop)
        self.assertIn("has_viewers", src)
        self.assertIn("d.display_fps", src)

    def test_detection_stays_gated_at_target_fps(self):
        """Smooth glass must never mean more model load.

        Detection is gated at the camera's effective rate — the adaptive
        per-camera target when the controller is on, and self.target_fps as the
        fallback when it is off. The gate itself (0.95 / rate) and that fixed
        fallback are both pinned so faster decode can never lift model load."""
        import inspect
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.MultiStreamPipeline._due_for_detection)
        self.assertIn("0.95 / rate", src)
        self.assertIn("or self.target_fps", src)


class PublishRateDefaults(unittest.TestCase):
    def test_every_publish_fps_default_is_24(self):
        """The wall ran at 12fps for a day because the CLI default lagged the
        class default and nobody passed the flag. All three defaults move
        together now — a mismatch is a regression, not a preference."""
        import inspect
        from cvti.serving.pipeline import MultiStreamPipeline, run_site
        self.assertEqual(inspect.signature(MultiStreamPipeline.__init__)
                         .parameters["publish_fps"].default, 24.0)
        self.assertEqual(inspect.signature(run_site)
                         .parameters["publish_fps"].default, 24.0)
        import cvti.serving.pipeline as pipeline_mod
        src = inspect.getsource(pipeline_mod)
        self.assertIn('"--publish-fps", type=float, default=24.0', src)


if __name__ == "__main__":
    unittest.main()
