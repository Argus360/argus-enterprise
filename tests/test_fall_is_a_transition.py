"""A fall is a transition from upright, not a posture (cvti/detector/fall.py)."""
from __future__ import annotations

import unittest

from cvti.detector.fall import FallDetector

AREA = 1000.0 * 1000.0
UP = (1, 100, 100, 160, 300)       # w/h 0.3  — upright
DOWN = (1, 100, 100, 400, 220)     # w/h 2.5  — horizontal


class FallNeedsATransition(unittest.TestCase):
    def test_person_already_lying_down_is_not_a_fall(self):
        d = FallDetector(min_frames=2)
        for i in range(6):
            self.assertIsNone(d.update([DOWN], AREA, timestamp=10.0 + i * 0.5))

    def test_upright_then_down_fast_is_a_fall(self):
        d = FallDetector(min_frames=2)
        self.assertIsNone(d.update([UP], AREA, timestamp=10.0))
        self.assertIsNone(d.update([UP], AREA, timestamp=10.5))
        self.assertIsNone(d.update([DOWN], AREA, timestamp=11.0))   # streak 1
        got = d.update([DOWN], AREA, timestamp=11.5)                # streak 2 -> fires
        self.assertEqual(got and got["kind"], "fall")

    def test_slowly_settling_down_is_not_a_fall(self):
        d = FallDetector(min_frames=2, upright_within_s=2.5)
        d.update([UP], AREA, timestamp=10.0)
        # crouched / bending (in between) for a while, then horizontal
        mid = (1, 100, 100, 300, 300)  # w/h 1.0 — neither upright nor down
        for i in range(8):
            self.assertIsNone(d.update([mid], AREA, timestamp=10.5 + i * 0.5))
        self.assertIsNone(d.update([DOWN], AREA, timestamp=15.0))
        self.assertIsNone(d.update([DOWN], AREA, timestamp=15.5))

    def test_person_walking_under_an_overhead_camera_is_not_a_fall(self):
        # Foreshortened from above the box goes wide, but it keeps travelling.
        d = FallDetector(min_frames=3)
        d.update([UP], AREA, timestamp=10.0)
        for i in range(6):
            x = 100 + i * 120                       # moves ~a box width per frame
            box = (1, x, 100, x + 300, 220)         # w/h 2.5 — "down"
            self.assertIsNone(d.update([box], AREA, timestamp=10.2 + i * 0.2))

    def test_person_down_and_still_is_a_fall(self):
        d = FallDetector(min_frames=3)
        d.update([UP], AREA, timestamp=10.0)
        got = None
        for i in range(4):
            box = (1, 100 + i * 5, 100, 400 + i * 5, 220)   # a wobble, not travel
            got = d.update([box], AREA, timestamp=10.2 + i * 0.2) or got
        self.assertEqual(got and got["kind"], "fall")

    def test_old_static_behaviour_is_opt_in(self):
        d = FallDetector(min_frames=2, upright_within_s=0)
        d.update([DOWN], AREA, timestamp=10.0)
        self.assertEqual(d.update([DOWN], AREA, timestamp=10.5)["kind"], "fall")


if __name__ == "__main__":
    unittest.main()
