"""A provisional critical the VLM rejected is not an incident (cvti/api/sources.py).

The row is kept for audit, marked retracted, with review left NULL on purpose.
It must not be listed, must not count as awaiting review, and a live update of
it must read as settled — on the CHI pilot these filled the queue with forklift
work the VLM had already cleared."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from cvti.api import sources


def _db(rows):
    d = Path(tempfile.mkdtemp()) / "events.db"
    c = sqlite3.connect(d)
    c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts REAL, iso TEXT, camera_id TEXT,"
              " rule TEXT, priority TEXT, confidence REAL, reason TEXT, zone TEXT,"
              " evidence_dir TEXT, review TEXT, retracted INTEGER DEFAULT 0)")
    for r in rows:
        c.execute("INSERT INTO events (ts, rule, priority, review, retracted) VALUES (?,?,?,?,?)", r)
    c.commit(); c.close()
    return str(d)


class RetractedRowsAreNotIncidents(unittest.TestCase):
    def setUp(self):
        self.db = _db([
            (1.0, "weapon_sighting", "critical", None, 0),   # real, awaiting review
            (2.0, "violence", "critical", None, 1),          # VLM rejected
            (3.0, "violence", "critical", None, 1),          # VLM rejected
            (4.0, "camera_offline", "high", "resolved", 0),
        ])

    def test_not_listed(self):
        rules = [e["rule"] for e in sources.read_events(self.db)["events"]]
        self.assertEqual(sorted(rules), ["camera_offline", "weapon_sighting"])

    def test_not_counted_as_awaiting_review(self):
        t = sources.read_triage(self.db)
        self.assertEqual(t["to_review"], 1)
        self.assertEqual(t["total"], 2)
        self.assertEqual(t["by_priority"].get("critical"), 1)

    def test_live_update_reads_as_settled(self):
        ev = sources._to_api_event({"id": 2, "rule": "violence", "review": None, "retracted": 1})
        self.assertEqual(ev["triage"]["state"], "resolved")
        self.assertTrue(ev["retracted"])

    def test_old_database_without_the_column(self):
        d = Path(tempfile.mkdtemp()) / "events.db"
        c = sqlite3.connect(d)
        c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, ts REAL, rule TEXT,"
                  " priority TEXT, review TEXT)")
        c.execute("INSERT INTO events (ts, rule, priority) VALUES (1, 'x', 'high')")
        c.commit(); c.close()
        self.assertEqual(len(sources.read_events(str(d))["events"]), 1)
        self.assertEqual(sources.read_triage(str(d))["to_review"], 1)


if __name__ == "__main__":
    unittest.main()
