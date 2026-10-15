"""The wall shares a decode budget among watched cameras by measured cost
(Tolu's v1.10.9 audit, 15 Oct 2026): a fixed per-camera floor is not a
budget."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.wall_budget import (  # noqa: E402
    MIN_WALL_FPS, WALL_DECODE_BUDGET_MS, WallCamera, allocate_wall_fps, estimate_cost_ms,
)


def cam(i, watched=True, busy=None, w=2592, h=1944, kf=False):
    return WallCamera(f"c{i}", watched, busy, w, h, kf)


class AllocationTests(unittest.TestCase):
    def test_nobody_watching_means_no_boost(self):
        rates = allocate_wall_fps([cam(1, watched=False), cam(2, watched=False)],
                                  ceiling_fps=8.0, cpu_only=True)
        self.assertEqual(rates, {"c1": 0.0, "c2": 0.0})

    def test_fifteen_expensive_tiles_fit_one_core_instead_of_costing_120_fps(self):
        # 41 ms per 5-megapixel HEVC frame, as measured on the pilot server.
        cams = [cam(i, busy=41.0) for i in range(15)]
        rates = allocate_wall_fps(cams, ceiling_fps=8.0, cpu_only=True)
        total_ms = sum(rates[c.camera_id] * 41.0 for c in cams)
        self.assertLessEqual(total_ms, WALL_DECODE_BUDGET_MS * 1.5 + 1e-6,
                             "fifteen watched tiles must stay near one core of decode")
        for c in cams:
            self.assertGreaterEqual(rates[c.camera_id], MIN_WALL_FPS)
            self.assertLess(rates[c.camera_id], 8.0)

    def test_cheap_cameras_get_the_ceiling(self):
        # recorder substreams at 3 ms a frame and a keyframe-decoded camera
        cams = [cam(1, busy=3.0, w=640, h=480), cam(2, busy=None, w=640, h=512), cam(3, kf=True)]
        rates = allocate_wall_fps(cams, ceiling_fps=24.0, cpu_only=True)
        self.assertEqual(rates, {"c1": 24.0, "c2": 24.0, "c3": 24.0})

    def test_the_budget_is_shared_only_among_watched_cameras(self):
        cams = [cam(1, busy=41.0), cam(2, busy=41.0, watched=False),
                cam(3, busy=41.0, watched=False), cam(4, busy=41.0, watched=False)]
        rates = allocate_wall_fps(cams, ceiling_fps=24.0, cpu_only=True)
        # one watched 41 ms camera alone gets the whole core: 1000/41 = 24.4 -> ceiling
        self.assertEqual(rates["c1"], 24.0)
        self.assertEqual(rates["c2"], 0.0)

    def test_a_gpu_box_keeps_the_ceiling_for_every_watched_tile(self):
        cams = [cam(i, busy=41.0) for i in range(15)]
        rates = allocate_wall_fps(cams, ceiling_fps=24.0, cpu_only=False)
        self.assertEqual(set(rates.values()), {24.0})

    def test_a_site_budget_scales_the_answer(self):
        cams = [cam(i, busy=41.0) for i in range(6)]
        one_core = allocate_wall_fps(cams, ceiling_fps=24.0, cpu_only=True, budget_ms=1000.0)
        two_cores = allocate_wall_fps(cams, ceiling_fps=24.0, cpu_only=True, budget_ms=2000.0)
        self.assertAlmostEqual(two_cores["c0"], min(24.0, one_core["c0"] * 2), places=6)

    def test_unknown_cost_uses_a_size_prior(self):
        self.assertAlmostEqual(estimate_cost_ms(cam(1, busy=None, w=2592, h=1944)), 50.4, places=1)
        self.assertAlmostEqual(estimate_cost_ms(cam(1, busy=None, w=640, h=480)), 3.072, places=6)
        self.assertEqual(estimate_cost_ms(cam(1, busy=None, kf=True)), 1.0)
        self.assertEqual(estimate_cost_ms(cam(1, busy=37.5)), 37.5)


class WiringTests(unittest.TestCase):
    def test_the_smooth_loop_allocates_by_cost_and_health_reports_the_rate(self):
        from cvti.serving import pipeline, streams
        loop = inspect.getsource(pipeline.MultiStreamPipeline._smooth_publish_loop)
        self.assertIn("allocate_wall_fps(", loop)
        self.assertIn("self.publisher.has_viewers(cam_id)", loop)
        self.assertIn('getattr(d, "_busy_ms", None)', loop)
        self.assertIn('"wall_fps"', inspect.getsource(streams.StreamDecoder.ingest_status))
        self.assertIn("wall_decode_budget_ms=", inspect.getsource(pipeline.run_site))


if __name__ == "__main__":
    unittest.main()
