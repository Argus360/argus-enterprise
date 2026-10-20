"""store_evidence: false in the site file: an alert is recorded (what, where,
when, why) and notified, but no image of any kind is written: no frames, no
clip, no full-resolution snapshot. The event's metadata file is kept. For an
engine hosted off-premises under terms that forbid recording (21 Oct 2026)."""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.contracts import CandidateAlert  # noqa: E402
from cvti.serving.alert_queue import QueuedAlert  # noqa: E402
from cvti.serving.alert_sink import AlertSink  # noqa: E402


class _Result:
    confirmed, confidence, reason = True, 0.9, "lingering"


class _Recorder:
    def __init__(self): self.events = []
    def notify(self, event): self.events.append(event)


def _fire(sink):
    frame = np.full((240, 320, 3), 40, np.uint8)
    cand = CandidateAlert(rule_name="loitering_watch", detector="presence", title="PERSON IN ZONE",
                          person_id=7, object_label=None, priority="medium", timestamp=1.0)
    sink.handle(QueuedAlert(camera_id="cam1", rule_name="loitering_watch", priority="medium",
                            title="T", timestamp=1.0, track_id=7,
                            payload={"candidate": cand, "frames": [frame], "bbox": (10, 10, 50, 90),
                                     "clip_frames": [], "clip_fps": 0}), _Result())


class StoreEvidenceOffTests(unittest.TestCase):
    def test_no_images_are_written_but_the_event_and_notification_are(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _Recorder()
            sink = AlertSink(d, notifier=rec, save_evidence=False, routing_path=None)
            _fire(sink)
            sink.close() if hasattr(sink, "close") else None
            self.assertFalse(list(Path(d).rglob("*.jpg")) + list(Path(d).rglob("*.mp4")),
                             "no frames, clip or snapshot")
            folders = list((Path(d) / "events").iterdir())
            self.assertEqual([p.name for p in folders[0].iterdir()], ["event.json"])
            con = sqlite3.connect(Path(d) / "events.db")
            rows = con.execute("SELECT camera_id, rule FROM events").fetchall()
            con.close()
            self.assertEqual(rows, [("cam1", "loitering_watch")])
            self.assertEqual(len(rec.events), 1)

    def test_full_resolution_snapshot_is_not_written_when_off(self):
        with tempfile.TemporaryDirectory() as d:
            sink = AlertSink(d, notifier=_Recorder(), save_evidence=False, routing_path=None)
            sink.full_frame_provider = lambda cam: b"\xff\xd8fakejpeg"
            _fire(sink)
            self.assertFalse(list(Path(d).rglob("evidence_full.jpg")))

    def test_default_still_writes_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            sink = AlertSink(d, notifier=_Recorder(), routing_path=None)
            _fire(sink)
            sink.close() if hasattr(sink, "close") else None
            self.assertTrue(list(Path(d).rglob("*.jpg")))

    def test_pipeline_reads_the_site_key(self):
        src = (Path(__file__).resolve().parents[1] / "cvti" / "serving" / "pipeline.py").read_text()
        self.assertIn('site.get("store_evidence", True) is not False', src)
        self.assertIn("save_evidence=store_evidence", src)


if __name__ == "__main__":
    unittest.main()
