"""
# These tests pin the frame STRIDE of the heavy pass; the separate time budget
# (heavy_min_interval_s, tests/test_scene_detector_cost.py) is switched off here.
Pose and weapon run only when a frame earns them (latency audit 1 Sep, D3).

The two heaviest per-camera models ran unconditionally, every frame — on
empty corridors, on parked cars, all night. Everything they feed (concealment,
violence, theft, weapons) is about a person doing something. The contract:

- nobody in the shared detector's frame → neither model runs;
- with people present they run on a stride (default every 2nd frame), EXCEPT
  the first frame a person appears, which always runs;
- both models infer at the engine's own imgsz — until 3 Sep run_site's
  --imgsz 512 never reached them and they silently paid for the 640 default;
- the cheap always-on detectors (tamper, fire, fall, zones) are untouched by
  the gate — a covered camera or a fire needs no person.
"""
from __future__ import annotations

import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.detector.core import PosePersonState
from cvti.rules.customization import CustomizationEngine
from cvti.serving.camera import PerCameraState, build_camera_states


def _person(x: float = 10.0) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[x, 10.0, x + 40.0, 90.0]]),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
    )


def _empty() -> sv.Detections:
    return sv.Detections.empty()


def _frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


def _pose_person(wrist: tuple[float, float]) -> PosePersonState:
    return PosePersonState(
        track_id=1,
        bbox=(60, 10, 140, 110),
        timestamp=0.0,
        left_shoulder=(80.0, 20.0),
        right_shoulder=(120.0, 20.0),
        left_elbow=None,
        right_elbow=None,
        left_wrist=wrist,
        right_wrist=None,
        max_wrist_speed=0.0,
        max_wrist_accel=0.0,
        max_arm_extension_ratio=0.0,
        weapon_labels=[],
        left_hip=(85.0, 100.0),
        right_hip=(115.0, 100.0),
    )


class _GateHarness(unittest.TestCase):
    """A camera with pose+weapon signals on, models faked at the two seams:
    _compute_pose (the pose forward pass) and predict_with_model (the weapon
    forward pass inside _merged_detections)."""

    def _state(self, **kw) -> PerCameraState:
        eng = CustomizationEngine("configs/all_threats_v1.json",
                                  baseline_path="configs/baseline_critical_v1.json")
        state = PerCameraState("cam1", eng, person_filter=False, heavy_min_interval_s=0.0,
                               pose_model=object(), weapon_model=object(),
                               concealment=True, weapons=True, **kw)
        return state

    def _run(self, state: PerCameraState, detections_per_frame: list) -> tuple[int, int]:
        """Process the frames; return (pose_calls, weapon_calls)."""
        weapon_calls = 0

        def fake_weapon(*a, **k):
            nonlocal weapon_calls
            weapon_calls += 1
            return []

        with mock.patch.object(PerCameraState, "_compute_pose",
                               return_value=[]) as pose, \
             mock.patch("cvti.detector.core.predict_with_model", fake_weapon), \
             mock.patch("cvti.detector.core.merge_detections",
                        side_effect=lambda a, b: list(a) + list(b)):
            for i, det in enumerate(detections_per_frame):
                state.process(det, _frame(), timestamp=100.0 + i * 0.2)
        return pose.call_count, weapon_calls


class PersonGateTests(_GateHarness):
    def test_an_empty_scene_runs_neither_heavy_model(self):
        pose, weapon = self._run(self._state(), [_empty()] * 6)
        self.assertEqual((pose, weapon), (0, 0))

    def test_a_person_makes_both_run(self):
        pose, weapon = self._run(self._state(), [_person()])
        self.assertEqual((pose, weapon), (1, 1))

    def test_first_appearance_always_runs_even_off_stride(self):
        """The opening moment of a threat must never be the skipped frame:
        empty frames may leave the tick anywhere, and the person's first
        frame runs regardless."""
        for padding in (1, 2, 3, 4):
            state = self._state()
            pose, _ = self._run(state, [_empty()] * padding + [_person()])
            self.assertEqual(pose, 1, f"first appearance skipped after {padding} empty frames")

    def test_with_people_present_the_stride_halves_the_model_calls(self):
        state = self._state()          # heavy_stride defaults to 2
        frames = [_person(10.0 + i) for i in range(10)]
        pose, weapon = self._run(state, frames)
        self.assertEqual(pose, weapon)
        # first appearance + every 2nd tick thereafter: strictly between
        # "every frame" (10) and "starved" — and roughly half.
        self.assertLessEqual(pose, 6)
        self.assertGreaterEqual(pose, 5)

    def test_stride_one_restores_every_frame(self):
        state = self._state(heavy_stride=1)
        pose, _ = self._run(state, [_person(10.0 + i) for i in range(6)])
        self.assertEqual(pose, 6)

    def test_concealment_candidate_behavior_matches_at_pose_stride_one_and_two(self):
        for stride, frame_count in ((1, 13), (2, 24)):
            with self.subTest(heavy_stride=stride):
                engine = CustomizationEngine()
                engine.rules = [
                    {"name": "shoplifting", "trigger": {"detector": "concealment"},
                     "priority": "high"}
                ]
                state = PerCameraState(
                    "cam1", engine, person_filter=False, heavy_min_interval_s=0.0, pose_model=object(),
                    concealment=True, heavy_stride=stride,
                )
                pose_sequence = (
                    [[_pose_person((200.0, 20.0))] for _ in range(4)]
                    + [[_pose_person((105.0, 95.0))] for _ in range(9)]
                )

                alerts = []
                with mock.patch.object(
                    PerCameraState, "_compute_pose", side_effect=pose_sequence,
                ) as pose:
                    for i in range(frame_count):
                        alerts.extend(state.process(
                            _person(), _frame(), timestamp=100.0 + i * 0.1
                        ))

                self.assertEqual(pose.call_count, len(pose_sequence))
                self.assertTrue(
                    any(alert.rule_name == "shoplifting" for alert in alerts),
                    "reach-then-waist samples should reach concealment candidate state",
                )

    def test_pose_failure_does_not_bypass_concealment_expiration(self):
        state = PerCameraState(
            "cam1", CustomizationEngine(), person_filter=False, heavy_min_interval_s=0.0,
            pose_model=object(), concealment=True, heavy_stride=1,
        )
        with mock.patch.object(
            PerCameraState,
            "_compute_pose",
            side_effect=[[_pose_person((200.0, 20.0))], RuntimeError("pose failed")],
        ):
            state.process(_person(), _frame(), timestamp=100.0)
            self.assertIn(1, state._conceal._buffers)
            state.process(_person(), _frame(), timestamp=101.51)

        self.assertNotIn(1, state._conceal._buffers)
        self.assertNotIn(1, state._conceal._over_threshold)
        self.assertNotIn(1, state._conceal._last_seen)

    def test_cheap_detectors_ignore_the_gate(self):
        """Tamper works on an EMPTY scene — that's its whole point."""
        state = self._state(tamper=True)
        with mock.patch.object(PerCameraState, "_compute_pose", return_value=[]):
            covered = np.zeros((120, 160, 3), dtype=np.uint8)   # black = blocked
            for i in range(40):
                alerts = state.process(_empty(), covered, timestamp=100.0 + i * 0.2)
        self.assertIsNotNone(state._tamper_det)                 # it ran, personless


class ImgszPlumbingTests(unittest.TestCase):
    def test_first_pose_timing_excludes_lazy_performance_import(self):
        observed = []

        class _Board:
            def observe(self, stage, unit, duration_ms):
                observed.append(duration_ms)

        class _DelayedPerf(types.ModuleType):
            def __getattribute__(self, name):
                if name == "BOARD":
                    time.sleep(0.06)
                return super().__getattribute__(name)

        delayed = _DelayedPerf("cvti.serving.perf")
        delayed.BOARD = _Board()
        state = PerCameraState(
            "pose-import-contract", CustomizationEngine(), person_filter=False, heavy_min_interval_s=0.0,
            pose_model=object(), concealment=True,
        )
        with mock.patch.dict(sys.modules, {"cvti.serving.perf": delayed}), \
                mock.patch("cvti.detector.core.extract_pose_people", return_value=[]):
            state._compute_pose(_frame(), timestamp=100.0)

        self.assertEqual(len(observed), 1)
        self.assertLess(observed[0], 40.0)

    def test_pose_forward_pass_reports_per_camera_throughput(self):
        from cvti.serving.perf import BOARD

        state = PerCameraState(
            "pose-perf-contract", CustomizationEngine(), person_filter=False, heavy_min_interval_s=0.0,
            pose_model=object(), concealment=True,
        )
        # BOARD is a process-wide singleton: measure the delta, so running this
        # file alongside others that touch the board cannot break it.
        before = BOARD.snapshot().get("pose_infer", {}).get("pose-perf-contract", {})
        with mock.patch("cvti.detector.core.extract_pose_people", return_value=[]):
            state._compute_pose(_frame(), timestamp=100.0)

        pose = BOARD.snapshot()["pose_infer"]["pose-perf-contract"]
        self.assertEqual(pose["count"] - before.get("count", 0), 1)
        self.assertEqual(pose["units"] - before.get("units", 0), 1)

    def test_build_camera_states_hands_the_engines_imgsz_to_every_camera(self):
        site = {"cameras": [{"id": "front", "source": "x",
                             "config": "configs/all_threats_v1.json"}]}
        cams = build_camera_states(site, imgsz=512)
        self.assertEqual(cams["front"]["state"].imgsz, 512)

    def test_run_site_passes_its_imgsz_through(self):
        import inspect
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.run_site)
        self.assertIn("imgsz=imgsz", src)

    def test_heavy_stride_is_a_per_camera_config_knob(self):
        site = {"cameras": [{"id": "front", "source": "x",
                             "config": "configs/all_threats_v1.json",
                             "heavy_stride": 3}]}
        cams = build_camera_states(site)
        self.assertEqual(cams["front"]["state"].heavy_stride, 3)


if __name__ == "__main__":
    unittest.main()
