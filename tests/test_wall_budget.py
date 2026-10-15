"""The wall shares a decode budget among watched cameras by measured CPU cost,
after the detection baseline is reserved (Tolu's v1.10.9 audit and his review
of the first version, 15 Oct 2026)."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.wall_budget import (  # noqa: E402
    MIN_WALL_FPS, WALL_DECODE_BUDGET_MS, WallCamera, allocate_wall_fps, estimate_cost_ms,
)


def cam(i, watched=True, cost=None, w=2592, h=1944, kf=False, baseline=0.0):
    return WallCamera(f"c{i}", watched, cost, w, h, kf, baseline)


def spent(alloc, cams):
    """Decode ms/s the allocation implies, with the decoder's max(baseline, display)."""
    return sum(max(c.baseline_fps, alloc.rates[c.camera_id]) * estimate_cost_ms(c) for c in cams)


class BudgetTests(unittest.TestCase):
    def test_nobody_watching_means_no_boost_and_only_the_baseline_is_spent(self):
        cams = [cam(1, watched=False, cost=41.0, baseline=2.0), cam(2, watched=False, cost=41.0)]
        a = allocate_wall_fps(cams, ceiling_fps=8.0)
        self.assertEqual(a.rates, {"c1": 0.0, "c2": 0.0})
        self.assertAlmostEqual(a.reserved_ms, 82.0)
        self.assertTrue(a.fits)

    def test_the_allocation_never_exceeds_the_budget_when_it_can_fit(self):
        # Review: the first version permitted 150% of the nominal budget.
        cams = [cam(i, cost=41.0, baseline=1.0) for i in range(4)]
        a = allocate_wall_fps(cams, ceiling_fps=8.0)
        self.assertTrue(a.fits, a.summary())
        self.assertLessEqual(spent(a, cams), WALL_DECODE_BUDGET_MS + 1e-6)
        for c in cams:
            self.assertGreater(a.rates[c.camera_id], 1.0)      # some wall work was granted
            self.assertLess(a.rates[c.camera_id], 8.0)

    def test_the_detection_baseline_is_reserved_and_reported_when_it_does_not_fit(self):
        # Review P1: 15 analysed cameras at 41 ms, 4 fps detection = 2460 ms/s
        # before the wall does anything. The old code "allocated" 2 fps and
        # the decoders ran at 4 anyway.
        cams = [cam(i, cost=41.0, baseline=4.0) for i in range(15)]
        a = allocate_wall_fps(cams, ceiling_fps=8.0)
        self.assertAlmostEqual(a.reserved_ms, 15 * 4 * 41.0)
        self.assertFalse(a.fits)
        self.assertAlmostEqual(a.over_budget_ms, a.reserved_ms - WALL_DECODE_BUDGET_MS)
        self.assertEqual(a.wall_ms, 0.0, "no wall work is granted above a baseline that already overflows")
        for c in cams:
            self.assertEqual(a.rates[c.camera_id], 4.0, "a watched tile shows at the detection rate it already pays for")

    def test_a_watched_minimum_that_does_not_fit_is_granted_but_reported(self):
        # 15 watched view-only cameras at 41 ms: the 2 fps floor alone is 1230 ms/s.
        cams = [cam(i, cost=41.0) for i in range(15)]
        a = allocate_wall_fps(cams, ceiling_fps=8.0)
        self.assertEqual(set(a.rates.values()), {MIN_WALL_FPS})
        self.assertAlmostEqual(a.over_budget_ms, 15 * 2 * 41.0 - WALL_DECODE_BUDGET_MS)

    def test_a_ceiling_below_the_floor_wins(self):
        # Review P2: a 1 fps wall must not be turned into 2.
        a = allocate_wall_fps([cam(1, cost=41.0)], ceiling_fps=1.0)
        self.assertEqual(a.rates["c1"], 1.0)
        a = allocate_wall_fps([cam(1, cost=41.0)], ceiling_fps=0.0)
        self.assertEqual(a.rates["c1"], 0.0)

    def test_cheap_cameras_reach_the_ceiling_on_any_box(self):
        cams = [cam(1, cost=3.0, w=640, h=480), cam(2, cost=None, w=640, h=512), cam(3, kf=True)]
        a = allocate_wall_fps(cams, ceiling_fps=24.0)
        self.assertEqual(a.rates, {"c1": 24.0, "c2": 24.0, "c3": 24.0})
        self.assertTrue(a.fits)

    def test_a_site_budget_scales_the_answer(self):
        cams = [cam(i, cost=41.0) for i in range(6)]
        one = allocate_wall_fps(cams, ceiling_fps=24.0, budget_ms=1000.0)
        two = allocate_wall_fps(cams, ceiling_fps=24.0, budget_ms=2000.0)
        self.assertAlmostEqual(two.rates["c0"], min(24.0, one.rates["c0"] * 2), places=6)

    def test_unknown_cost_uses_a_size_prior(self):
        self.assertAlmostEqual(estimate_cost_ms(cam(1, cost=None, w=2592, h=1944)), 50.4, places=1)
        self.assertAlmostEqual(estimate_cost_ms(cam(1, cost=None, w=640, h=480)), 3.072, places=6)
        self.assertEqual(estimate_cost_ms(cam(1, cost=None, kf=True)), 1.0)
        self.assertEqual(estimate_cost_ms(cam(1, cost=37.5)), 37.5)


class NetworkPacedReadsTests(unittest.TestCase):
    """Review P1: a decoder at the live edge waits for its camera; a wall-time
    number would feed back on itself. The loop must hand the allocator CPU
    time, and the allocator's answer must not depend on the rate granted."""

    def test_the_loop_passes_cpu_time_not_wall_time(self):
        from cvti.serving import pipeline
        loop = inspect.getsource(pipeline.MultiStreamPipeline._smooth_publish_loop)
        self.assertIn('getattr(d, "frame_cpu_ms", None)', loop)
        self.assertNotIn("_busy_ms", loop)
        self.assertIn("0.0 if cam_id in view_only else baseline", loop)
        self.assertIn("self.wall_budget_status = allocation.summary()", loop)

    def test_cost_does_not_change_with_the_rate_granted(self):
        cams = [cam(i, cost=41.0) for i in range(8)]
        first = allocate_wall_fps(cams, ceiling_fps=8.0)
        again = allocate_wall_fps(cams, ceiling_fps=8.0)    # same CPU cost, same answer
        self.assertEqual(first.rates, again.rates)


class HealthTests(unittest.TestCase):
    def test_health_says_when_the_budget_does_not_fit_and_why(self):
        from cvti.serving.health_doc import derive_status
        status, reasons = derive_status(
            cameras=[], gate={}, disk={}, memory={}, components={},
            engine={"wall_budget": {"budget_ms": 1000, "reserved_ms": 2460, "wall_ms": 0,
                                    "over_budget_ms": 1460, "watched": 15}})
        hit = [r for r in reasons if r.startswith("wall decode budget exceeded by 1460")]
        self.assertEqual(len(hit), 1)
        self.assertIn("detection alone needs 2460 of 1000", hit[0])
        _, reasons = derive_status(cameras=[], gate={}, disk={}, memory={}, components={},
                                   engine={"wall_budget": {"budget_ms": 1000, "reserved_ms": 100,
                                                           "wall_ms": 1130, "over_budget_ms": 230,
                                                           "watched": 15}})
        self.assertTrue(any("15 watched tiles at their minimum rate" in r for r in reasons))
        _, reasons = derive_status(cameras=[], gate={}, disk={}, memory={}, components={},
                                   engine={"wall_budget": {"over_budget_ms": 0}})
        self.assertFalse(any("wall decode budget" in r for r in reasons))
        self.assertIn("engine=engine", inspect.getsource(__import__("cvti.serving.health_doc", fromlist=["x"]).build_health_doc))


if __name__ == "__main__":
    unittest.main()
