"""Unit tests for the fall / person-collapsed CV detector.

(Fire, panic-running, and crowd-formation now live in cvti.detector.situational
and are covered by tests/test_situational_hse.py.)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.detector.fall import FallDetector


class FallTests(unittest.TestCase):
    def test_horizontal_person_fires(self):
        # A fall is a transition: the person was upright, then went down.
        det = FallDetector(min_frames=3)
        res = None
        det.update([(1, 80, 10, 120, 180)], frame_area=200 * 200, timestamp=-0.25)
        for i in range(6):
            res = det.update([(1, 10, 100, 210, 160)], frame_area=200 * 200, timestamp=i * 0.25) or res
        self.assertIsNotNone(res)
        self.assertEqual(res["kind"], "fall")
        self.assertEqual(res["track_id"], 1)

    def test_upright_person_is_quiet(self):
        det = FallDetector(min_frames=3)
        res = None
        for i in range(6):
            res = det.update([(1, 80, 10, 120, 180)], frame_area=200 * 200, timestamp=i * 0.25) or res
        self.assertIsNone(res)

    def test_brief_bend_is_quiet(self):
        det = FallDetector(min_frames=4)
        det.update([(1, 10, 100, 210, 160)], frame_area=200 * 200, timestamp=0.0)
        res = det.update([(1, 80, 10, 120, 180)], frame_area=200 * 200, timestamp=0.25)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
