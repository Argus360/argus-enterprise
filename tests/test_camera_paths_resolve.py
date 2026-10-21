"""A camera's zones/rules path that does not exist where the engine runs must
not kill the engine (field report, 20 Oct 2026: Live EarthCams feed,
FileNotFoundError on configs/zones/live_watch.json at every Start)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ResolveCameraPathsTests(unittest.TestCase):
    def test_missing_relative_path_is_found_in_the_bundle(self):
        from cvti.serving import camera as c
        with tempfile.TemporaryDirectory() as d:
            bundled = Path(d) / "configs" / "zones" / "x.json"
            bundled.parent.mkdir(parents=True)
            bundled.write_text(json.dumps({"zones": []}))
            with mock.patch("cvti.utils.resource_path", side_effect=lambda r: Path(d) / r):
                cam = c.resolve_camera_paths({"id": "a", "zones": "configs/zones/x.json"})
        self.assertEqual(cam["zones"], str(bundled))

    def test_zones_that_exist_nowhere_are_dropped_not_fatal(self):
        from cvti.serving import camera as c
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("cvti.utils.resource_path", side_effect=lambda r: Path(d) / r):
                with self.assertLogs("cvti.serving.camera", level="WARNING"):
                    cam = c.resolve_camera_paths({"id": "a", "zones": "configs/zones/nope.json",
                                                  "config": "configs/rules/nope.json"})
        self.assertNotIn("zones", cam)
        self.assertEqual(cam["config"], "configs/rules/nope.json")   # rules loader warns itself

    def test_existing_paths_are_left_alone(self):
        from cvti.serving import camera as c
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            cam = c.resolve_camera_paths({"id": "a", "zones": f.name})
            self.assertEqual(cam["zones"], f.name)

    def test_the_live_watch_files_ship_in_the_installer(self):
        spec = (Path(__file__).resolve().parents[1] / "packaging" / "argus.spec").read_text()
        self.assertIn('("configs/zones/*.json", "configs/zones")', spec)
        self.assertIn('("configs/rules/*.json", "configs/rules")', spec)


if __name__ == "__main__":
    unittest.main()
