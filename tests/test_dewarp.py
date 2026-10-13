"""Fisheye dewarp: the circle is found, the maps are right, and every frame
path (engine decoder, stopped-monitoring preview, zone snapshot) is flattened.
Seven of twelve CHI cameras are fisheye (survey, 13 Oct 2026)."""
from __future__ import annotations

import inspect
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.dewarp import Dewarp, dewarps_for  # noqa: E402

CX, CY, R = 410, 290, 250


def fisheye_frame(w=800, h=600, cx=CX, cy=CY, r=R):
    """A black frame with a bright image circle; a red dot at the centre, a
    green dot at the rim on the +x axis."""
    import cv2
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (cx, cy), r, (120, 120, 120), -1)
    cv2.circle(img, (cx, cy), 12, (0, 0, 255), -1)
    cv2.circle(img, (cx + r - 14, cy), 8, (0, 255, 0), -1)
    return img


class CircleTests(unittest.TestCase):
    def test_the_image_circle_is_found_from_the_frame(self):
        cx, cy, r = Dewarp.find_circle(fisheye_frame())
        self.assertAlmostEqual(cx, CX, delta=2)
        self.assertAlmostEqual(cy, CY, delta=2)
        self.assertAlmostEqual(r, R, delta=3)

    def test_a_frame_with_no_border_is_not_a_circle(self):
        self.assertIsNone(Dewarp.find_circle(np.full((300, 400, 3), 200, np.uint8)))

    def test_given_centre_and_radius_are_used_as_they_are(self):
        dw = Dewarp.from_config({"mode": "panorama", "center": [100, 90], "radius": 80})
        dw.apply(fisheye_frame())
        self.assertEqual(dw.describe()["center"], (100.0, 90.0))
        self.assertEqual(dw.describe()["radius"], 80.0)
        self.assertFalse(dw.describe()["auto_circle"])


class PanoramaTests(unittest.TestCase):
    def test_output_is_the_configured_strip_and_the_rim_lands_on_the_top_row(self):
        dw = Dewarp.from_config({"mode": "panorama", "output": [720, 240]})
        out = dw.apply(fisheye_frame())
        self.assertEqual(out.shape, (240, 720, 3))
        # seam 0: column 0 is the +x direction; the green rim dot is near the top-left.
        top_left = out[:20, :12]
        self.assertGreater(int(top_left[..., 1].max()), 200, "rim pixel missing from the strip's top row")
        # the centre of the circle (red dot) is inside the hole: it must NOT appear
        red_only = (out[..., 2].astype(int) > 200) & (out[..., 1].astype(int) < 100)
        self.assertEqual(int(red_only.sum()), 0, "the centre dot leaked into the strip")

    def test_maps_are_rebuilt_when_the_source_size_changes(self):
        dw = Dewarp.from_config({"mode": "panorama"})
        dw.apply(fisheye_frame())
        first = dw.describe()["center"]
        dw.apply(fisheye_frame(w=1600, h=1200, cx=820, cy=580, r=500))
        self.assertNotEqual(dw.describe()["center"], first)
        self.assertAlmostEqual(dw.describe()["center"][0], 820, delta=3)


class PlaceholderFirstFrameTests(unittest.TestCase):
    """A decoder's first frames can be flat grey (HEVC before its first
    keyframe). The pilot server locked a centred guess on those (13 Oct)."""

    def test_a_flat_frame_is_not_a_circle(self):
        self.assertIsNone(Dewarp.find_circle(np.full((600, 800, 3), 128, np.uint8)))

    def test_the_circle_is_found_on_a_later_frame_and_the_maps_rebuilt(self):
        from cvti.serving import dewarp as dw_mod
        dw = Dewarp.from_config({"mode": "panorama", "output": [720, 240]})
        grey = np.full((600, 800, 3), 128, np.uint8)
        out = dw.apply(grey)
        self.assertEqual(out.shape, (240, 720, 3))
        self.assertFalse(dw.describe()["circle_settled"])
        self.assertEqual(dw.describe()["center"], (400.0, 300.0))   # provisional
        with mock.patch.object(dw_mod.time, "monotonic", side_effect=[10.0, 10.0, 20.0, 20.0, 30.0, 30.0]):
            dw.apply(grey)                        # still a placeholder: keeps trying
            self.assertFalse(dw.describe()["circle_settled"])
            dw.apply(fisheye_frame())             # a real frame: settles and rebuilds
        self.assertTrue(dw.describe()["circle_settled"])
        self.assertAlmostEqual(dw.describe()["center"][0], CX, delta=2)
        self.assertAlmostEqual(dw.describe()["radius"], R, delta=3)
        # and it stops looking once settled
        before = dw._circle_attempts
        dw.apply(fisheye_frame())
        self.assertEqual(dw._circle_attempts, before)

    def test_it_gives_up_after_the_attempt_budget(self):
        from cvti.serving import dewarp as dw_mod
        dw = Dewarp.from_config({"mode": "panorama", "output": [720, 240]})
        grey = np.full((600, 800, 3), 128, np.uint8)
        clock = iter(float(t) for t in range(0, 10000, 2))
        with mock.patch.object(dw_mod.time, "monotonic", side_effect=lambda: next(clock)):
            for _ in range(dw_mod.CIRCLE_ATTEMPTS + 5):
                dw.apply(grey)
        self.assertFalse(dw.describe()["circle_settled"])
        self.assertEqual(dw._circle_attempts, dw_mod.CIRCLE_ATTEMPTS)


class PerspectiveTests(unittest.TestCase):
    def test_the_view_centre_is_the_circle_centre(self):
        dw = Dewarp.from_config({"mode": "perspective", "output": [640, 400], "fov": 100})
        out = dw.apply(fisheye_frame())
        self.assertEqual(out.shape, (400, 640, 3))
        centre = out[195:205, 315:325]
        self.assertGreater(int(centre[..., 2].mean()), 200, "the red centre dot should sit mid-view")

    def test_pitch_moves_the_view(self):
        flat = Dewarp.from_config({"mode": "perspective", "output": [640, 400]}).apply(fisheye_frame())
        tilted = Dewarp.from_config({"mode": "perspective", "output": [640, 400], "pitch": 30}).apply(fisheye_frame())
        self.assertFalse(np.array_equal(flat, tilted))


class ConfigTests(unittest.TestCase):
    def test_missing_or_off_means_no_dewarp(self):
        self.assertIsNone(Dewarp.from_config(None))
        self.assertIsNone(Dewarp.from_config({}))
        self.assertIsNone(Dewarp.from_config({"mode": "off"}))

    def test_bad_settings_are_errors_not_silent(self):
        for cfg in ({"mode": "banana"}, {"mode": "panorama", "fov": 500}, {"mode": "panorama", "centre": [1, 2]},
                    {"mode": "perspective", "output": [10, 10]}, "panorama"):
            with self.assertRaises(ValueError, msg=repr(cfg)):
                Dewarp.from_config(cfg)

    def test_site_cameras_map_to_dewarps_by_id(self):
        cams = [{"id": "a", "dewarp": {"mode": "panorama"}}, {"id": "b"}, {"id": "c", "dewarp": {"mode": "off"}}]
        got = dewarps_for(cams)
        self.assertEqual(set(got), {"a"})
        with self.assertRaises(ValueError) as ctx:
            dewarps_for([{"id": "z", "dewarp": {"mode": "nope"}}])
        self.assertIn("camera z", str(ctx.exception))


class FramePathTests(unittest.TestCase):
    """Every path a frame takes is flattened."""

    class Cap:
        def __init__(self): self.n = 0
        def get(self, *_a): return 10.0
        def grab(self): return True
        def read(self):
            self.n += 1
            return True, fisheye_frame()
        def release(self): pass
        def set(self, *_a): pass
        def isOpened(self): return True  # noqa: N802

    def test_the_engine_decoder_hands_out_dewarped_frames(self):
        from cvti.serving import streams as st
        dw = Dewarp.from_config({"mode": "panorama", "output": [720, 240]})
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1/x", target_fps=30.0, dewarp=dw)
        with mock.patch("cvti.serving.capture.open_capture", return_value=self.Cap()):
            d.start()
            deadline = time.time() + 5
            frame = None
            while time.time() < deadline and frame is None:
                frame = d.read_latest()
                time.sleep(0.02)
            d.stop()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.image.shape, (240, 720, 3))
        self.assertEqual((d.stream_width, d.stream_height), (720, 240))

    def test_the_preview_wall_and_its_zone_snapshot_are_dewarped(self):
        from cvti.app.live_wall import LiveWall
        wall = LiveWall([{"id": "cam", "source": "fake-source", "dewarp": {"mode": "panorama", "output": [720, 240]}}],
                        width=960, fps=8, quality=65, retain_snapshot=True)
        with mock.patch.object(LiveWall, "_open", return_value=self.Cap()):
            wall.start()
            deadline = time.time() + 5
            snap = None
            while time.time() < deadline and not snap:
                snap = wall.snapshot("cam")
                time.sleep(0.02)
            wall.stop()
        self.assertTrue(snap and "uri" in snap, snap)
        self.assertEqual((snap["w"], snap["h"]), (720, 240))

    def test_the_site_file_reaches_both_paths(self):
        from cvti.app import console_backend, preview
        from cvti.serving import pipeline
        self.assertIn("camera_dewarp=", inspect.getsource(pipeline.run_site))
        self.assertIn("dewarp=self.camera_dewarp.get(cam_id)", inspect.getsource(pipeline.MultiStreamPipeline.start))
        self.assertIn('dewarp=cam.get("dewarp")', inspect.getsource(console_backend.ConsoleBackend.camera_preview))
        self.assertIn('"dewarp": self._dewarp.get(camera_id)', inspect.getsource(preview.CameraPreview.acquire))


if __name__ == "__main__":
    unittest.main()
