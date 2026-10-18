"""The heartbeat receiver's diagnostics inbox (18 Oct 2026): a site operator
can push a redacted diagnostics zip with the same site key the heartbeat uses;
it lands on disk, is listed on the dashboard, and downloads byte-for-byte.
Nothing is accepted without the key, over the cap, non-zip, or outside the
inbox directory."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import heartbeat_receiver as hr  # noqa: E402


def _zip(text: bytes = b"hello") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.txt", text)
    return buf.getvalue()


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.recv = hr.Receiver(keys={"chi": "k1"}, db_path=root / "hb.db",
                                diag_dir=root / "diag", dashboard_token="tok")
        self.addCleanup(self.recv.close)
        self.addCleanup(self.tmp.cleanup)

    def test_right_key_stores_lists_and_downloads(self):
        code, msg = self.recv.handle_diagnostic("k1", "chi", "argus-diag-CHI.zip", _zip())
        self.assertEqual(code, 200, msg)
        items = self.recv.store.diagnostics()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["site_id"], "chi")
        self.assertTrue(items[0]["filename"].endswith("-argus-diag-CHI.zip"))
        path = self.recv.diagnostic_path("chi", items[0]["filename"])
        self.assertIsNotNone(path)
        self.assertEqual(path.read_bytes(), _zip())
        page = hr.render_dashboard(self.recv.views(), diagnostics=items, token="tok")
        self.assertIn("Diagnostics uploads", page)
        self.assertIn(items[0]["filename"], page)
        self.assertIn("?token=tok", page)

    def test_wrong_key_unknown_site_and_bad_bodies_are_refused(self):
        self.assertEqual(self.recv.handle_diagnostic("bad", "chi", "x.zip", _zip())[0], 401)
        self.assertEqual(self.recv.handle_diagnostic("k1", "nope", "x.zip", _zip())[0], 401)
        self.assertEqual(self.recv.handle_diagnostic("k1", "chi", "x.zip", b"")[0], 413)
        self.assertEqual(self.recv.handle_diagnostic("k1", "chi", "x.zip", b"not a zip")[0], 415)
        self.assertEqual(self.recv.store.diagnostics(), [])

    def test_names_are_sanitised_and_paths_cannot_escape(self):
        code, _ = self.recv.handle_diagnostic("k1", "chi", "../../etc/passwd", _zip())
        self.assertEqual(code, 200)
        name = self.recv.store.diagnostics()[0]["filename"]
        self.assertNotIn("/", name)
        self.assertTrue(name.endswith(".zip"))
        self.assertIsNone(self.recv.diagnostic_path("chi", "../hb.db"))
        self.assertIsNone(self.recv.diagnostic_path("../", name))
        self.assertIsNone(self.recv.diagnostic_path("chi", "missing.zip"))

    def test_inbox_off_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            r = hr.Receiver(keys={"chi": "k1"}, db_path=Path(d) / "hb.db")
            self.addCleanup(r.close)
            self.assertEqual(r.handle_diagnostic("k1", "chi", "x.zip", _zip())[0], 404)
            self.assertNotIn("Diagnostics uploads", hr.render_dashboard(r.views()))


if __name__ == "__main__":
    unittest.main()
