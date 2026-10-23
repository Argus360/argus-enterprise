"""Main-thread cost cuts for whole-frame work (CHI, 23 Oct 2026)."""
from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from cvti.detector.tamper import TamperDetector
from cvti.serving.camera import PerCameraState


def _state(**kw):
    return PerCameraState("cam", Mock(), **kw)


class ClipReplayOnlyWhenEvidenceIsStored(unittest.TestCase):
    def test_off_records_nothing(self):
        s = _state(clip_replay=False)
        s._record_clip_frame(np.zeros((48, 64, 3), np.uint8), 1.0)
        self.assertEqual(len(s._clip_buffer), 0)

    def test_on_records_a_jpeg(self):
        s = _state()
        s._record_clip_frame(np.zeros((48, 64, 3), np.uint8), 1.0)
        self.assertEqual(len(s._clip_buffer), 1)
        self.assertEqual(s._clip_buffer[0][0], 1.0)


class SceneDetectorsAreRateLimited(unittest.TestCase):
    def test_due_at_most_once_per_interval(self):
        s = _state(scene_detector_interval_s=0.5)
        self.assertTrue(s._scene_due("tamper", 10.0))
        self.assertFalse(s._scene_due("tamper", 10.2))
        self.assertFalse(s._scene_due("tamper", 10.49))
        self.assertTrue(s._scene_due("tamper", 10.5))

    def test_detectors_do_not_share_a_clock(self):
        s = _state()
        self.assertTrue(s._scene_due("tamper", 10.0))
        self.assertTrue(s._scene_due("fire", 10.0))


class TamperMeasuresOnAThumbnail(unittest.TestCase):
    def test_brightness_agrees_between_sizes(self):
        rng = np.random.default_rng(0)
        big = rng.integers(0, 255, (938, 1600, 3), dtype=np.uint8)
        import cv2
        small = cv2.resize(big, (320, 187), interpolation=cv2.INTER_AREA)
        b_big, _ = TamperDetector._measure(big)
        b_small, _ = TamperDetector._measure(small)
        self.assertAlmostEqual(b_big, b_small, delta=2.0)

    def test_still_fires_on_blackout(self):
        # Textured frames: a flat colour has zero sharpness and would read as
        # "obscured" on its own, which is not what a real camera looks like.
        rng = np.random.default_rng(1)
        d = TamperDetector(warmup=3, min_frames=2)
        bright = rng.integers(100, 200, (938, 1600, 3), dtype=np.uint8)
        dark = rng.integers(0, 10, (938, 1600, 3), dtype=np.uint8)
        for _ in range(4):
            self.assertIsNone(d.update(bright))
        self.assertIsNone(d.update(dark))            # first dark frame: streak 1 of 2
        self.assertEqual(d.update(dark)["kind"], "blackout")


if __name__ == "__main__":
    unittest.main()


class HeavyPassHasATimeBudget(unittest.TestCase):
    """The pose/weapon pass runs on a time budget, not just a frame stride."""

    def _decide(self, state, present, ts):
        # Mirror the decision in process(): stride gate AND time gate, with a
        # person's first appearance always running.
        first = present and not state._person_seen_last_frame
        state._person_seen_last_frame = present
        state._heavy_tick += 1
        run = present and (first or state._heavy_tick % max(1, state.heavy_stride) == 0)
        if run and not first and state.heavy_min_interval_s > 0 \
                and ts - state._heavy_last_at < state.heavy_min_interval_s:
            run = False
        if run:
            state._heavy_last_at = ts
        return run

    def test_decision_logic_matches_process(self):
        import inspect
        from cvti.serving.camera import PerCameraState as P
        src = inspect.getsource(P.process)
        self.assertIn("timestamp - self._heavy_last_at < self.heavy_min_interval_s", src)
        self.assertIn("self._heavy_last_at = timestamp", src)

    def test_first_appearance_always_runs(self):
        s = _state(heavy_stride=2, heavy_min_interval_s=0.5)
        self.assertTrue(self._decide(s, True, 10.0))
        s._person_seen_last_frame = False          # person left and came back
        self.assertTrue(self._decide(s, True, 10.1))

    def test_at_most_twice_a_second_with_people_present(self):
        s = _state(heavy_stride=1, heavy_min_interval_s=0.5)
        self._decide(s, True, 0.0)
        runs = sum(self._decide(s, True, 0.0 + i / 20) for i in range(1, 41))   # 2s at 20 fps
        self.assertLessEqual(runs, 4)
        self.assertGreaterEqual(runs, 3)

    def test_zero_interval_restores_stride_only(self):
        s = _state(heavy_stride=2, heavy_min_interval_s=0.0)
        self._decide(s, True, 0.0)
        runs = sum(self._decide(s, True, i / 20) for i in range(1, 41))
        self.assertEqual(runs, 20)
