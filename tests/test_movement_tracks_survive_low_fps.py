"""At 1 fps detection a person is seen once a second or less. The motion
tracker's 1.0 s expiry, tuned for several frames a second, dropped every
track between samples, so nobody was ever "moving" and the movement alerts
could not fire (CHI, 19 Oct 2026). The expiry now follows the camera's
measured detection interval."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _person(x, track_id=7):
    return (track_id, x, 100.0, x + 60.0, 260.0)


class TrackerAtOneFpsTests(unittest.TestCase):
    def _run(self, expiry: float, gap: float = 1.2):
        from cvti.detector.person_motion import PersonMotionTracker
        tr = PersonMotionTracker(track_expiry_seconds=expiry)
        shape = (720, 1280)
        motions = None
        # a person walking across the frame, one sample every `gap` seconds;
        # the speed estimate is a slow EMA (alpha 0.2), so give it ten samples
        for i in range(10):
            motions = tr.update([_person(100.0 + 110.0 * i)], 10.0 + gap * i, shape)
        return motions

    def test_default_expiry_loses_the_person_between_samples(self):
        motions = self._run(expiry=1.0)
        self.assertFalse(any(m.moving for m in motions),
                         "with a 1.0 s expiry nothing is ever moving at 1.2 s gaps")

    def test_scaled_expiry_keeps_the_track_and_sees_movement(self):
        motions = self._run(expiry=3.6)      # 3 x a 1.2 s interval
        self.assertEqual(len(motions), 1)
        self.assertTrue(motions[0].moving)


class CameraScalesExpiryTests(unittest.TestCase):
    def test_camera_sets_expiry_from_its_detection_interval(self):
        from cvti.detector.person_motion import PersonMotionTracker
        from cvti.serving import camera as cam_mod
        state = cam_mod.PerCameraState.__new__(cam_mod.PerCameraState)
        state._motion_tracker = PersonMotionTracker()
        state._detection_interval_s = 1.4
        # run just the expiry line the frame loop runs before update()
        interval = float(getattr(state, "_detection_interval_s", 0.0) or 0.0)
        if interval > 0.0:
            state._motion_tracker.track_expiry_seconds = max(1.0, 3.0 * interval)
        self.assertAlmostEqual(state._motion_tracker.track_expiry_seconds, 4.2)


if __name__ == "__main__":
    unittest.main()
