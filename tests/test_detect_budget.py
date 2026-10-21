"""Adaptive detection-rate budgeting (cvti/serving/detect_budget.py)."""
from __future__ import annotations

import unittest

from cvti.serving.detect_budget import (AdaptiveConfig, DetectCamera,
                                        DetectionRateController,
                                        allocate_detection_fps)


class AllocateDetectionFps(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(allocate_detection_fps([], budget_fps=50, floor_fps=6, ceiling_fps=30), {})

    def test_every_camera_gets_at_least_the_floor(self):
        cams = [DetectCamera("a", 0.0), DetectCamera("b", 0.0), DetectCamera("c", 0.0)]
        # Budget below the combined floor: everyone still gets the floor.
        rates = allocate_detection_fps(cams, budget_fps=3, floor_fps=6, ceiling_fps=30)
        self.assertEqual(rates, {"a": 6.0, "b": 6.0, "c": 6.0})

    def test_all_quiet_splits_surplus_evenly(self):
        cams = [DetectCamera("a", 0.0), DetectCamera("b", 0.0)]
        rates = allocate_detection_fps(cams, budget_fps=24, floor_fps=6, ceiling_fps=30)
        self.assertAlmostEqual(rates["a"], 12.0)
        self.assertAlmostEqual(rates["b"], 12.0)

    def test_busy_camera_gets_more(self):
        cams = [DetectCamera("busy", 9.0), DetectCamera("quiet", 1.0)]
        rates = allocate_detection_fps(cams, budget_fps=32, floor_fps=6, ceiling_fps=30)
        # floors take 12; surplus 20 split 9:1 -> +18 / +2
        self.assertAlmostEqual(rates["busy"], 24.0)
        self.assertAlmostEqual(rates["quiet"], 8.0)
        self.assertAlmostEqual(sum(rates.values()), 32.0)

    def test_ceiling_caps_and_redistributes(self):
        cams = [DetectCamera("busy", 100.0), DetectCamera("quiet", 1.0)]
        # Huge budget: busy would blow past the ceiling; the overflow goes to quiet.
        rates = allocate_detection_fps(cams, budget_fps=100, floor_fps=6, ceiling_fps=30)
        self.assertAlmostEqual(rates["busy"], 30.0)   # capped
        self.assertAlmostEqual(rates["quiet"], 30.0)  # soaked up the rest, also capped
        self.assertLessEqual(max(rates.values()), 30.0 + 1e-9)

    def test_never_exceeds_ceiling(self):
        cams = [DetectCamera(c, a) for c, a in (("a", 5), ("b", 2), ("c", 0))]
        rates = allocate_detection_fps(cams, budget_fps=1000, floor_fps=6, ceiling_fps=20)
        for r in rates.values():
            self.assertLessEqual(r, 20.0 + 1e-9)
            self.assertGreaterEqual(r, 6.0 - 1e-9)


class Config(unittest.TestCase):
    def test_absent_or_false_is_disabled(self):
        self.assertFalse(AdaptiveConfig.from_site(None).enabled)
        self.assertFalse(AdaptiveConfig.from_site(False).enabled)
        self.assertFalse(AdaptiveConfig.from_site({"enabled": False}).enabled)

    def test_true_enables_with_defaults(self):
        cfg = AdaptiveConfig.from_site(True)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.floor_fps, 6.0)
        self.assertEqual(cfg.ceiling_fps, 120.0)   # GPU-bound, not a fixed cap

    def test_dict_overrides_and_clamps(self):
        cfg = AdaptiveConfig.from_site({"floor_fps": 10, "ceiling_fps": 40,
                                        "target_busy": 5.0, "window_s": 0.1})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.floor_fps, 10.0)
        self.assertEqual(cfg.ceiling_fps, 40.0)
        self.assertLessEqual(cfg.target_busy, 0.95)   # clamped from 5.0
        self.assertGreaterEqual(cfg.window_s, 0.5)    # clamped up from 0.1


class Controller(unittest.TestCase):
    def _cams(self):
        return [DetectCamera("a", 1.0), DetectCamera("b", 1.0)]

    def test_grows_budget_when_gpu_has_headroom(self):
        cfg = AdaptiveConfig(enabled=True, floor_fps=6, ceiling_fps=30,
                             target_busy=0.7, max_step=1.5)
        ctrl = DetectionRateController(cfg, budget_fps=16)
        before = ctrl.budget_fps
        ctrl.update(busy_fraction=0.2, cameras=self._cams())   # well under target
        self.assertGreater(ctrl.budget_fps, before)

    def test_shrinks_budget_when_gpu_saturating(self):
        cfg = AdaptiveConfig(enabled=True, floor_fps=6, ceiling_fps=30,
                             target_busy=0.7, max_step=1.5)
        ctrl = DetectionRateController(cfg, budget_fps=40)
        before = ctrl.budget_fps
        ctrl.update(busy_fraction=0.95, cameras=self._cams())  # over target
        self.assertLess(ctrl.budget_fps, before)

    def test_budget_bounded_by_floor_and_ceiling(self):
        cfg = AdaptiveConfig(enabled=True, floor_fps=6, ceiling_fps=30,
                             target_busy=0.7, max_step=10.0)
        ctrl = DetectionRateController(cfg, budget_fps=100)
        ctrl.update(busy_fraction=0.99, cameras=self._cams())
        self.assertGreaterEqual(ctrl.budget_fps, 6 * 2 - 1e-9)   # floor * n
        ctrl2 = DetectionRateController(cfg, budget_fps=1)
        ctrl2.update(busy_fraction=0.01, cameras=self._cams())
        self.assertLessEqual(ctrl2.budget_fps, 30 * 2 + 1e-9)    # ceiling * n

    def test_step_is_clamped(self):
        cfg = AdaptiveConfig(enabled=True, floor_fps=1, ceiling_fps=1000,
                             target_busy=0.7, max_step=1.5)
        ctrl = DetectionRateController(cfg, budget_fps=20)
        ctrl.update(busy_fraction=0.001, cameras=self._cams())   # would want huge growth
        self.assertLessEqual(ctrl.budget_fps, 20 * 1.5 + 1e-9)   # but clamped to max_step


if __name__ == "__main__":
    unittest.main()


class LoopPeriodFollowsTheController(unittest.TestCase):
    """_loop_period: the run loop must keep up with the adaptive target, so the
    rate is bounded by GPU/stream, not a fixed 0.15s tick (the old hidden cap)."""

    def _pipe(self, adaptive, cam_targets):
        from cvti.serving.pipeline import MultiStreamPipeline
        p = MultiStreamPipeline.__new__(MultiStreamPipeline)
        p.tick_seconds = 0.15
        p._adaptive = adaptive
        p._cam_target_fps = cam_targets
        return p

    def test_fixed_tick_when_adaptive_off(self):
        self.assertAlmostEqual(self._pipe(None, {})._loop_period(), 0.15)

    def test_fixed_tick_until_controller_sets_rates(self):
        ctrl = object()
        self.assertAlmostEqual(self._pipe(ctrl, {})._loop_period(), 0.15)

    def test_speeds_up_to_serve_the_highest_target(self):
        ctrl = object()
        # want 25 fps -> period 0.04s, well under the 0.15 fixed tick
        self.assertAlmostEqual(self._pipe(ctrl, {"a": 25.0})._loop_period(), 0.04)

    def test_never_busy_spins_below_the_floor(self):
        ctrl = object()
        # want 1000 fps -> clamped to the 0.01s (100 loops/s) floor
        self.assertAlmostEqual(self._pipe(ctrl, {"a": 1000.0})._loop_period(), 0.01)

    def test_never_slower_than_the_fixed_tick(self):
        ctrl = object()
        # a low target must not make the loop sleep longer than tick_seconds
        self.assertAlmostEqual(self._pipe(ctrl, {"a": 2.0})._loop_period(), 0.15)
