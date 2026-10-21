"""Detection is a number now, not a walk test (CHI, 19 Oct 2026): each
camera counts the frames the model looked at and the frames with a person,
health carries the counts, and the heartbeat whitelists them."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DetectionStatusTests(unittest.TestCase):
    def test_counts_and_last_seen(self):
        from cvti.serving.camera import detection_status
        class S:  # noqa: D401 - a bare state
            frames_analysed = 120
            frames_with_people = 7
            people_in_last_frame = 2
            people_last_seen_at = 1000.0
        d = detection_status(S(), now=1012.5)
        self.assertEqual(d["frames_analysed"], 120)
        self.assertEqual(d["frames_with_people"], 7)
        self.assertEqual(d["people_in_last_frame"], 2)
        self.assertEqual(d["people_last_seen_s"], 12.5)

    def test_never_seen_is_none_not_a_huge_number(self):
        from cvti.serving.camera import detection_status
        class S:
            frames_analysed = 5
        d = detection_status(S(), now=1e9)
        self.assertEqual(d["frames_with_people"], 0)
        self.assertIsNone(d["people_last_seen_s"])


class HeartbeatCarriesCountsTests(unittest.TestCase):
    def test_counts_and_version_travel_nothing_else_does(self):
        from cvti.serving.heartbeat import heartbeat_payload
        health = {"status": "ok", "reasons": [], "uptime_s": 1.0, "version": "1.10.14",
                  "cameras": [{"camera_id": "bay", "state": "connected", "last_frame_age_s": 0.2,
                               "reconnects": 0,
                               "detection": {"frames_analysed": 50, "frames_with_people": 3,
                                             "people_in_last_frame": 1, "people_last_seen_s": 4.0,
                                             "secret_pixels": "nope"}}],
                  "gate": {}, "disk": {}, "memory": {}, "components": {"components": []},
                  "engine": {}, "self_test": {}}
        p = heartbeat_payload(health, site_id="s", now=1.0)
        cam = p["cameras"][0]
        self.assertEqual(cam["frames_analysed"], 50)
        self.assertEqual(cam["frames_with_people"], 3)
        self.assertEqual(cam["people_last_seen_s"], 4.0)
        self.assertEqual(p["version"], "1.10.14")
        self.assertNotIn("secret_pixels", str(p))
        self.assertNotIn("people_in_last_frame", cam)


if __name__ == "__main__":
    unittest.main()
