from __future__ import annotations

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.alert_queue import AlertQueue, QueuedAlert
from cvti.serving.batcher import collect_batch
from cvti.serving.streams import Frame


def _alert(cam="cam0", rule="loitering", prio="medium", ts=0.0, track=1, zone="shelf_right"):
    return QueuedAlert(camera_id=cam, rule_name=rule, priority=prio, title="T",
                       timestamp=ts, track_id=track, zone=zone)


class AlertQueueTests(unittest.TestCase):
    def test_duplicate_within_cooldown_is_suppressed(self):
        q = AlertQueue(cooldown_seconds=8.0)
        self.assertTrue(q.add(_alert(ts=0.0)))
        self.assertFalse(q.add(_alert(ts=3.0)))       # same cam/rule/track/zone, < cooldown
        self.assertEqual(q.dropped_duplicates, 1)
        self.assertEqual(q.pending_count, 1)

    def test_refires_after_cooldown(self):
        now = [0.0]
        q = AlertQueue(cooldown_seconds=8.0, clock=lambda: now[0])
        q.add(_alert(ts=0.0))
        now[0] = 9.0
        self.assertTrue(q.add(_alert(ts=9.0)))         # cooldown elapsed
        self.assertEqual(q.pending_count, 2)

    def test_different_track_not_deduped(self):
        q = AlertQueue(cooldown_seconds=8.0)
        q.add(_alert(ts=0.0, track=1))
        self.assertTrue(q.add(_alert(ts=1.0, track=2)))  # concurrent second person
        self.assertEqual(q.pending_count, 2)

    def test_drain_is_priority_ordered_and_throttled(self):
        q = AlertQueue(cooldown_seconds=0.0)           # no dedup for this test
        q.add(_alert(prio="medium", ts=1.0, track=1))
        q.add(_alert(prio="critical", ts=2.0, track=2))
        q.add(_alert(prio="high", ts=3.0, track=3))
        first = q.drain(max_per_drain=2)
        self.assertEqual([a.priority for a in first], ["critical", "high"])
        self.assertEqual(q.pending_count, 1)           # throttled: one left
        self.assertEqual(q.drain()[0].priority, "medium")

    def test_concurrent_cameras_both_queued(self):
        q = AlertQueue()
        self.assertTrue(q.add(_alert(cam="cam0", ts=0.0)))
        self.assertTrue(q.add(_alert(cam="cam7", ts=0.0)))  # different camera, same rule
        self.assertEqual(q.pending_count, 2)


class _FakeDecoder:
    def __init__(self, frames):
        self._frames = list(frames)

    def read_latest(self):
        return self._frames.pop(0) if self._frames else None


class BatcherTests(unittest.TestCase):
    def test_collect_only_fresh_frames_one_per_camera(self):
        f0 = Frame("cam0", object(), 1, 0.0)
        f1 = Frame("cam1", object(), 1, 0.0)
        decoders = {
            "cam0": _FakeDecoder([f0, None]),   # fresh, then nothing new
            "cam1": _FakeDecoder([f1]),
            "cam2": _FakeDecoder([]),           # idle camera contributes nothing
        }
        batch = collect_batch(decoders)
        self.assertEqual({f.camera_id for f in batch}, {"cam0", "cam1"})
        self.assertEqual(collect_batch(decoders), [])   # nothing new next tick

    def test_max_batch_caps_size(self):
        decoders = {f"cam{i}": _FakeDecoder([Frame(f"cam{i}", object(), 1, 0.0)])
                    for i in range(10)}
        self.assertEqual(len(collect_batch(decoders, max_batch=4)), 4)


class GatePoolTests(unittest.TestCase):
    def test_pool_drains_queue_and_confirms_via_mock_gate(self):
        import time
        import numpy as np
        from cvti.contracts import CandidateAlert
        from cvti.serving.gate_pool import GatePool
        from cvti.verification.gate import VerificationGate

        q = AlertQueue(cooldown_seconds=0.0)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        candidate = CandidateAlert(rule_name="loitering_at_shelf", priority="medium",
                                   detector="presence", title="PERSON IN ZONE",
                                   person_id=6, object_label=None, timestamp=1.0)
        qa = QueuedAlert(camera_id="cam0", rule_name="loitering_at_shelf", priority="medium",
                         title="PERSON IN ZONE", timestamp=1.0, track_id=6, zone="shelf_right",
                         payload={"candidate": candidate, "frames": [frame],
                                  "scene": {"environment_type": "retail_shop"}})
        q.add(qa)

        verdicts = []
        pool = GatePool(q, gate_factory=lambda: VerificationGate(provider="mock"),
                        on_verdict=lambda a, r: verdicts.append((a, r))).start()
        try:
            deadline = time.time() + 5.0
            while pool.verified < 1 and time.time() < deadline:
                time.sleep(0.05)
        finally:
            pool.stop()

        self.assertEqual(pool.verified, 1)
        self.assertEqual(pool.confirmed, 1)          # mock auto-confirms
        self.assertEqual(q.pending_count, 0)
        self.assertEqual(verdicts[0][0].camera_id, "cam0")


class ReconnectTests(unittest.TestCase):
    """A live stream that drops must be reopened and recover — without a real camera."""

    def test_decoder_recovers_after_drop(self):
        import time
        import numpy as np
        from cvti.serving.streams import StreamDecoder

        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        class _DropCap:      # simulates a dropped stream (read always fails)
            def get(self, p): return 0.0
            def grab(self): return True
            def read(self): return (False, None)
            def release(self): pass

        class _LiveCap:      # healthy stream after reconnect
            def get(self, p): return 0.0
            def grab(self): return True
            def read(self): return (True, frame)
            def release(self): pass

        state = {"n": 0}

        def fake_open():
            state["n"] += 1
            return _DropCap() if state["n"] == 1 else _LiveCap()

        d = StreamDecoder("cam", "rtsp://fake/stream", target_fps=5, reconnect_backoff=0.01)
        d._open = fake_open       # avoid real cv2.VideoCapture
        d.start()
        got = None
        for _ in range(60):
            got = d.read_latest()
            if got is not None:
                break
            time.sleep(0.05)
        d.stop()
        self.assertGreaterEqual(d.reconnects, 1)   # it detected the drop + reopened
        self.assertIsNotNone(got)                  # and recovered a live frame


class Phase1ConcurrentAlertsTests(unittest.TestCase):
    """Two simultaneous RawEvents must yield two CandidateAlerts that are BOTH
    queued and processable — not just candidate_alerts[0]."""

    def test_two_concurrent_events_both_reach_the_gate(self):
        from cvti.contracts import RawEvent
        from cvti.rules.customization import CustomizationEngine

        engine = CustomizationEngine("configs/all_threats_v1.json")
        events = [
            RawEvent(detector="weapons", active=True, title="GUN", level="critical",
                     object_label="gun", timestamp=1.0),
            RawEvent(detector="violence", active=True, title="FIGHT", level="critical",
                     person_id=3, timestamp=1.0),
        ]
        candidates = engine.evaluate(events)
        self.assertGreaterEqual(len(candidates), 2)  # both rules fire, not just one

        q = AlertQueue(cooldown_seconds=8.0)
        for c in candidates:
            q.add(QueuedAlert(camera_id="main", rule_name=c.rule_name, priority=c.priority,
                              title=c.title, timestamp=1.0, track_id=c.person_id,
                              object_label=c.object_label, payload=c))
        drained = q.drain(max_per_drain=10)
        self.assertEqual(len(drained), len(candidates))  # every concurrent alert processed

    def test_object_label_distinguishes_weapon_alerts(self):
        q = AlertQueue(cooldown_seconds=8.0)
        base = dict(camera_id="main", rule_name="weapon_sighting", priority="critical",
                    title="WEAPON", timestamp=0.0, track_id=None)
        self.assertTrue(q.add(QueuedAlert(**base, object_label="gun")))
        self.assertTrue(q.add(QueuedAlert(**base, object_label="knife")))   # different object
        self.assertFalse(q.add(QueuedAlert(**base, object_label="gun")))    # dup within cooldown
        self.assertEqual(q.pending_count, 2)


class CameraMappingTests(unittest.TestCase):
    def test_candidate_maps_to_queued_with_evidence(self):
        from cvti.contracts import CandidateAlert
        from cvti.serving.camera import _to_queued

        cand = CandidateAlert(rule_name="after_hours_in_aisle", priority="high",
                              detector="presence", title="PERSON IN ZONE SHELF_RIGHT",
                              person_id=8, object_label=None, timestamp=2.5)
        qa = _to_queued("aisle_cam_2", cand, 2.5, "shelf_right", frames=["F"],
                        scene={"environment_type": "retail_shop"})
        self.assertEqual(qa.camera_id, "aisle_cam_2")
        self.assertEqual(qa.track_id, 8)
        self.assertEqual(qa.zone, "shelf_right")
        self.assertEqual(qa.payload["candidate"], cand)
        self.assertEqual(qa.payload["frames"], ["F"])


class MovementPipelineTests(unittest.TestCase):
    class _Tracker:
        def update_with_detections(self, detections):
            return detections

    class _Zones:
        def update(self, detections, timestamp, frame_hw=None):
            return [
                SimpleNamespace(
                    tracker_id=int(track_id), zones=[zone], dwell_seconds={zone: timestamp},
                    loitering=False, entered_zones=[],
                )
                for track_id, zone in zip(detections.tracker_id, ("gate", "yard"))
            ]

        def drain_exits(self):
            return []

    @staticmethod
    def _detections(xs):
        boxes = np.array([[x, 20.0, x + 20.0, 80.0] for x in xs], dtype=float).reshape(-1, 4)
        return sv.Detections(
            xyxy=boxes,
            class_id=np.array([0] * len(xs)),
            confidence=np.array([0.95] * len(xs)),
            tracker_id=np.array([1, 2][:len(xs)]),
        )

    def _state(self, permitted):
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState

        state = PerCameraState(
            "chi_gate", CustomizationEngine(), zone_monitor=self._Zones(),
            person_filter=False, normal_movement=True, multiple_people_moving=True,
            movement_enter_speed_ratio=0.05, movement_exit_speed_ratio=0.02,
            movement_min_track_seconds=0.4, movement_min_people=2,
            movement_persistence_seconds=0.5,
            permitted_movement_zones=permitted,
        )
        state._tracker = self._Tracker()
        return state

    def test_permitted_zones_filter_overlays_and_simultaneous_count(self):
        state = self._state(("gate",))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        state.process(self._detections((0.0, 10.0)), frame, 0.0)
        state.process(self._detections((60.0, 70.0)), frame, 0.5)
        alerts = state.process(self._detections((70.0, 80.0)), frame, 1.0)

        self.assertEqual([item["track_id"] for item in state._motion_overlays], [1])
        self.assertEqual(state._motion_overlays[0]["colour"], (0, 200, 0))
        self.assertEqual(alerts, [], "one permitted moving track must not fire scenario 5")

    def test_absent_permitted_zones_counts_the_whole_camera(self):
        state = self._state(None)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        state.process(self._detections((0.0, 10.0)), frame, 0.0)
        state.process(self._detections((60.0, 70.0)), frame, 0.5)

        self.assertEqual([item["track_id"] for item in state._motion_overlays], [1, 2])

    def test_new_tracks_visible_without_claiming_movement(self):
        state = self._state(None)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        alerts = state.process(self._detections((0.0, 10.0)), frame, 0.0)
        self.assertEqual([item["label"] for item in state._motion_overlays],
                         ["#1 TRACKED", "#2 TRACKED"])
        self.assertEqual(alerts, [])
        state.process(self._detections(()), frame, 0.25)
        self.assertEqual(state._motion_overlays, [])

    def test_unassociated_detection_visible_but_not_counted_as_moving(self):
        state = self._state(None)
        state._tracker.update_with_detections = lambda detections: self._detections(())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        alerts = state.process(self._detections((0.0, 40.0)), frame, 0.0)
        self.assertEqual([o["label"] for o in state._motion_overlays], ["PERSON", "PERSON"])
        self.assertEqual(state._box_by_track, {})
        self.assertEqual(alerts, [])
        state.process(self._detections(()), frame, 0.5)
        self.assertEqual(state._motion_overlays, [])

    def test_unassociated_detection_cannot_bypass_permitted_zones(self):
        state = self._state(("gate",))
        state._tracker.update_with_detections = lambda detections: self._detections(())
        state.process(self._detections((0.0,)), np.zeros((100, 100, 3), dtype=np.uint8), 0.0)
        self.assertEqual(state._motion_overlays, [])

    def test_two_spread_out_movers_emit_one_latched_group_candidate(self):
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState

        state = PerCameraState(
            "chi_gate", CustomizationEngine("configs/chi_pilot_v1.json"),
            zone_monitor=self._Zones(), person_filter=False,
            multiple_people_moving=True,
            movement_enter_speed_ratio=0.05, movement_exit_speed_ratio=0.02,
            movement_min_track_seconds=0.4, movement_min_people=2,
            movement_persistence_seconds=0.5,
        )
        state._tracker = self._Tracker()
        frame = np.zeros((600, 1000, 3), dtype=np.uint8)

        self.assertEqual(state.process(self._detections((0.0, 900.0)), frame, 0.0), [])
        self.assertEqual(state.process(self._detections((300.0, 600.0)), frame, 0.5), [])
        alerts = state.process(self._detections((500.0, 400.0)), frame, 1.0)

        self.assertEqual(len(alerts), 1)
        candidate = alerts[0].payload["candidate"]
        self.assertEqual(candidate.detector, "multiple_people_moving")
        self.assertEqual(candidate.metadata["track_ids"], [1, 2])
        self.assertEqual(candidate.metadata["group_bbox"], (400.0, 20.0, 520.0, 80.0))
        self.assertEqual(len(candidate.metadata["motions"]), 2)
        self.assertEqual(alerts[0].payload["bbox"], (400, 20, 520, 80))
        self.assertEqual(len(alerts[0].payload["frames"]), 3)

        duplicate = state.process(self._detections((600.0, 300.0)), frame, 1.5)
        self.assertEqual(duplicate, [], "a sustained interval must emit only once")

    def test_scenario_five_only_draws_active_constituents_in_amber(self):
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState

        state = PerCameraState(
            "chi_gate", CustomizationEngine("configs/chi_pilot_v1.json"),
            zone_monitor=self._Zones(), person_filter=False,
            normal_movement=False, multiple_people_moving=True,
            movement_enter_speed_ratio=0.05, movement_exit_speed_ratio=0.02,
            movement_min_track_seconds=0.01, movement_min_people=2,
            movement_persistence_seconds=0.01,
        )
        state._tracker = self._Tracker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state.process(self._detections((0.0, 10.0)), frame, 0.0)

        state.process(self._detections((20.0, 30.0)), frame, 0.1)

        self.assertEqual([item["track_id"] for item in state._motion_overlays], [1, 2])
        self.assertEqual(
            {item["colour"] for item in state._motion_overlays}, {(0, 200, 255)}
        )

    def test_sub_expiry_pipeline_dropout_does_not_emit_a_duplicate_incident(self):
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState

        state = PerCameraState(
            "chi_gate", CustomizationEngine("configs/chi_pilot_v1.json"),
            zone_monitor=self._Zones(), person_filter=False,
            multiple_people_moving=True,
            movement_enter_speed_ratio=0.05, movement_exit_speed_ratio=0.02,
            movement_min_track_seconds=0.01, movement_min_people=2,
            movement_persistence_seconds=0.01,
        )
        state._tracker = self._Tracker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        state.process(self._detections((0.0, 10.0)), frame, 0.0)
        self.assertEqual(state.process(self._detections((20.0, 30.0)), frame, 0.1), [])
        self.assertEqual(len(state.process(self._detections((22.0, 32.0)), frame, 0.12)), 1)

        self.assertEqual(state.process(self._detections(()), frame, 0.3), [])
        self.assertEqual(
            state.process(self._detections((40.0, 50.0)), frame, 0.4),
            [],
            "the same scenario-5 incident duplicated after one detector miss",
        )

    def test_stationary_people_emit_no_movement_candidate(self):
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState

        state = PerCameraState(
            "chi_gate", CustomizationEngine("configs/chi_pilot_v1.json"),
            zone_monitor=self._Zones(), person_filter=False,
            multiple_people_moving=True, movement_persistence_seconds=0.5,
        )
        state._tracker = self._Tracker()
        frame = np.zeros((600, 1000, 3), dtype=np.uint8)

        for timestamp in (0.0, 0.5, 1.0, 1.5):
            self.assertEqual(
                state.process(self._detections((100.0, 800.0)), frame, timestamp),
                [],
            )


class ConcealmentBagWiringTests(unittest.TestCase):
    def test_shared_bag_detection_is_grounded_to_only_the_matching_pose_track(self):
        from cvti.detector.core import PosePersonState
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState

        def pose(track_id, bbox, wrist):
            x1, y1, x2, y2 = bbox
            return PosePersonState(
                track_id=track_id,
                bbox=bbox,
                timestamp=0.0,
                left_shoulder=(x1 + 20.0, y1 + 10.0),
                right_shoulder=(x2 - 20.0, y1 + 10.0),
                left_elbow=None,
                right_elbow=None,
                left_wrist=wrist,
                right_wrist=None,
                max_wrist_speed=0.0,
                max_wrist_accel=0.0,
                max_arm_extension_ratio=0.0,
                weapon_labels=[],
                left_hip=(x1 + 25.0, y2 - 10.0),
                right_hip=(x2 - 25.0, y2 - 10.0),
            )

        detections = sv.Detections(
            xyxy=np.array([
                [60.0, 10.0, 140.0, 110.0],
                [150.0, 10.0, 230.0, 110.0],
                [130.0, 55.0, 145.0, 105.0],
            ]),
            class_id=np.array([0, 0, 26]),
            confidence=np.array([0.95, 0.94, 0.88]),
        )
        pose_people = [
            pose(1, (60, 10, 140, 110), (125.0, 80.0)),
            pose(2, (150, 10, 230, 110), (155.0, 80.0)),
        ]
        state = PerCameraState(
            "cam1",
            CustomizationEngine(),
            person_filter=False,
            pose_model=object(),
            concealment=True,
            heavy_stride=1,
        )

        with mock.patch.object(PerCameraState, "_compute_pose", return_value=pose_people):
            state.process(detections, np.zeros((160, 500, 3), dtype=np.uint8), timestamp=1.0)

        first = state._conceal._buffers[1][-1]
        second = state._conceal._buffers[2][-1]
        self.assertGreater(first.hand_to_bag, 0.0)
        self.assertLess(first.hand_to_bag, 0.6)
        self.assertIsNone(second.hand_to_bag)


if __name__ == "__main__":
    unittest.main()


class AuditHardeningTests(unittest.TestCase):
    """Pins for the 23 Aug head-to-toe audit's fixed findings."""

    def test_watchdog_treats_long_lived_exit_as_scheduled(self):
        # The spawn caps --seconds (~28h); an exit after an hour+ of uptime is
        # a scheduled restart and must not burn crash budget. (finding #1)
        import inspect
        from cvti.app.console_backend import ConsoleBackend
        src = inspect.getsource(ConsoleBackend._start_watchdog)
        self.assertIn("self._restarts = 0", src)
        self.assertIn("3600", src)

    def test_gate_base_url_reaches_the_ollama_call(self):
        # --gate-base-url was silently ignored by the default provider (finding #4)
        import inspect
        from cvti.verification import gate as g
        self.assertIn("base_url", inspect.signature(g._call_ollama).parameters)
        src = inspect.getsource(g.VerificationGate._call_provider)
        self.assertIn("base_url=self.base_url", src)

    def test_a_preset_change_keeps_zone_and_english_rules(self):
        # Applying a template used to clobber cam["config"], dropping the
        # loitering + custom-English rules silently. (finding #9)
        import json, os, tempfile
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from _backend_helper import signed_in
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        (tmp / "site.json").write_text(json.dumps({"cameras": [{"id": "c1", "source": "0"}]}))
        cwd = os.getcwd()
        os.chdir(tmp)     # zones/rules regen writes under configs/ relative
        try:
            be = signed_in("owner", site_path=str(tmp / "site.json"),
                           db_path=str(tmp / "events.db"), enable_demo=False)
            be.set_custom_rule("c1", "Is anyone wearing a black hoodie?")
            be.apply_template("office")
            cam = json.loads((tmp / "site.json").read_text(encoding="utf-8"))["cameras"][0]
            from cvti.serving.custom_rules import _rules_for
            descs = [t["description"] for t in _rules_for(cam)]
            self.assertTrue(any("hoodie" in d for d in descs),
                            "template application dropped the customer's English rule")
        finally:
            os.chdir(cwd)


class SmoothPublishTest(unittest.TestCase):
    """The live wall is decoupled from detection (24 Aug): frames publish at
    stream cadence, the model samples at target_fps, and neither starves the
    other."""

    def test_peek_never_consumes_the_detection_frame(self):
        from cvti.serving.streams import Frame, StreamDecoder
        d = StreamDecoder.__new__(StreamDecoder)     # no stream: state only
        import threading
        d._lock = threading.Lock()
        d._latest = Frame("c", object(), 1, 0.1)
        d._seq = 7
        d._consumed = False
        f1, s1 = d.peek_latest()
        f2, s2 = d.peek_latest()
        self.assertIs(f1, f2); self.assertEqual((s1, s2), (7, 7))
        self.assertIsNotNone(d.read_latest(), "peek consumed the frame")
        self.assertIsNone(d.read_latest())
        f3, _ = d.peek_latest()
        self.assertIs(f3, f1, "peek must still see a consumed frame")

    def test_detection_gate_holds_the_model_to_target_fps(self):
        from cvti.serving.pipeline import MultiStreamPipeline
        pipe = MultiStreamPipeline.__new__(MultiStreamPipeline)
        pipe.target_fps = 4.0
        pipe._last_detect = {}
        hits = sum(1 for i in range(120)                    # 12 fps for 10s
                   if pipe._due_for_detection("cam", i / 12.0))
        self.assertLessEqual(hits, 45, "faster decode leaked into model load")
        self.assertGreaterEqual(hits, 35, "gate starves detection")

    def test_decoders_pace_to_publish_rate_when_decoupled(self):
        # Since 3 Sep the pacing is viewer-aware: decoders START at detection
        # rate and the smooth loop raises display_fps on exactly the cameras
        # someone is watching — the always-on max(target, publish) decode is
        # deliberately gone (it cost 12fps of decode per camera for a wall
        # nobody had open).
        import inspect
        from cvti.serving.pipeline import MultiStreamPipeline
        src = inspect.getsource(MultiStreamPipeline.start)
        # detection cameras start at detection rate; view-only glass idles at 1fps
        self.assertIn("target_fps=(1.0 if vo else self.target_fps)", src)
        self.assertIn("_smooth_publish_loop", src)
        loop = inspect.getsource(MultiStreamPipeline._smooth_publish_loop)
        self.assertIn("has_viewers", loop)

    def test_the_live_wall_ships_raw_glass(self):
        # Boxes live with the alert (the sink's annotated subject shot); the
        # wall draws nothing — no per-frame copy, no trailing rectangles.
        import inspect
        from cvti.serving import pipeline as pl
        src = inspect.getsource(pl.run_site)
        self.assertIn("FramePublisher().start(", src)       # default publisher: no boxes
        self.assertIn("cameras=", src)                      # one port per camera (W1 wall)
        loop = inspect.getsource(pl.MultiStreamPipeline._smooth_publish_loop)
        self.assertNotIn("latest_boxes", loop)

    def test_smooth_publish_passes_the_latest_motion_overlays(self):
        from cvti.serving.frame_publisher import FrameOverlay
        from cvti.serving.pipeline import MultiStreamPipeline

        class StopLoop(BaseException):
            pass

        class Publisher:
            def __init__(self):
                self.call = None

            def has_viewers(self, _camera_id):
                return True

            def publish(self, camera_id, frame, overlays=()):
                self.call = (camera_id, frame, overlays)
                raise StopLoop

        decoder = SimpleNamespace(
            display_fps=0,
            playout=None,
            peek_latest=lambda: (SimpleNamespace(image="decoded-frame"), 1),
        )
        publisher = Publisher()
        pipe = MultiStreamPipeline.__new__(MultiStreamPipeline)
        pipe.publish_fps = 12
        pipe.publisher = publisher
        pipe._decoders = {"cam": decoder}
        pipe.view_only = set()
        pipe._camera_states = {
            "cam": SimpleNamespace(
                _motion_overlays=[{
                    "track_id": 9,
                    "bbox": (10, 20, 30, 40),
                    "label": "#9 MOVING",
                }]
            )
        }

        with self.assertRaises(StopLoop):
            pipe._smooth_publish_loop()

        camera_id, frame, overlays = publisher.call
        self.assertEqual((camera_id, frame), ("cam", "decoded-frame"))
        self.assertEqual(overlays, [FrameOverlay(
            track_id=9,
            bbox=(10, 20, 30, 40),
            label="#9 MOVING",
            colour=(0, 200, 0),
        )])

    def test_motion_overlay_records_are_mapped_for_every_publish_path(self):
        from cvti.serving.frame_publisher import FrameOverlay
        from cvti.serving import pipeline

        self.assertTrue(hasattr(pipeline, "_frame_overlays"))
        state = SimpleNamespace(_motion_overlays=[{
            "track_id": 3,
            "bbox": (1, 2, 30, 40),
            "label": "#3 MOVING",
        }])
        self.assertEqual(pipeline._frame_overlays(state), [FrameOverlay(
            track_id=3,
            bbox=(1, 2, 30, 40),
            label="#3 MOVING",
            colour=(0, 200, 0),
        )])
        self.assertEqual(pipeline._frame_overlays(None), [])

    def test_motion_overlay_mapper_preserves_semantic_colour(self):
        from cvti.serving.frame_publisher import FrameOverlay
        from cvti.serving.pipeline import _frame_overlays

        state = SimpleNamespace(_motion_overlays=[{
            "track_id": 4,
            "bbox": (1, 2, 30, 40),
            "label": "#4 MOVING",
            "colour": (0, 200, 0),
        }])

        self.assertEqual(
            _frame_overlays(state),
            [FrameOverlay(4, (1, 2, 30, 40), "#4 MOVING", (0, 200, 0))],
        )

    def test_smooth_person_boxes_expire_from_observation_not_completion(self):
        from cvti.serving.pipeline import _frame_overlays
        state = SimpleNamespace(
            _motion_overlay_observed_at=10.0,
            _motion_overlays=[{"track_id": 1, "bbox": (1, 2, 30, 40), "label": "PERSON"}],
        )
        self.assertEqual(len(_frame_overlays(state, 10.5)), 1)
        self.assertEqual(_frame_overlays(state, 12.0), [])
        self.assertEqual(len(_frame_overlays(state, 12.0, exact_frame=True)), 1)

    def test_aggregate_track_ids_reach_publisher_alert_state(self):
        from cvti.contracts import CandidateAlert
        from cvti.serving.pipeline import _alert_track_ids

        aggregate = CandidateAlert(
            rule_name="chi_multiple_people_moving",
            priority="high",
            detector="multiple_people_moving",
            title="MULTIPLE PEOPLE MOVING",
            person_id=None,
            object_label=None,
            timestamp=1.0,
            metadata={"track_ids": [7, 9]},
        )
        singular = CandidateAlert(
            rule_name="weapon",
            priority="critical",
            detector="weapons",
            title="WEAPON",
            person_id=3,
            object_label="knife",
            timestamp=1.0,
        )

        queued = [
            QueuedAlert(
                camera_id="cam", rule_name=aggregate.rule_name, priority="high",
                title=aggregate.title, timestamp=1.0,
                payload={"candidate": aggregate},
            ),
            QueuedAlert(
                camera_id="cam", rule_name=singular.rule_name, priority="critical",
                title=singular.title, timestamp=1.0, track_id=3,
                payload={"candidate": singular},
            ),
        ]

        self.assertEqual(_alert_track_ids(queued), {3, 7, 9})

    def test_paced_playout_passes_motion_overlays_and_source_size(self):
        from cvti.serving.frame_publisher import FrameOverlay
        from cvti.serving.pipeline import MultiStreamPipeline

        class StopLoop(BaseException):
            pass

        class Publisher:
            def has_viewers(self, _camera_id):
                return True

            def publish_jpeg(self, camera_id, jpeg, overlays=(), source_size=None):
                self.call = (camera_id, jpeg, overlays, source_size)
                raise StopLoop

        decoded = np.zeros((480, 640, 3), np.uint8)
        decoder = SimpleNamespace(
            display_fps=0,
            playout=SimpleNamespace(pop_due=lambda _now: b"paced-jpeg"),
            peek_latest=lambda: (SimpleNamespace(image=decoded), 1),
        )
        publisher = Publisher()
        pipe = MultiStreamPipeline.__new__(MultiStreamPipeline)
        pipe.publish_fps = 12
        pipe.publisher = publisher
        pipe._decoders = {"cam": decoder}
        pipe.view_only = set()
        pipe._camera_states = {
            "cam": SimpleNamespace(_motion_overlays=[{
                "track_id": 5,
                "bbox": (10, 20, 30, 40),
                "label": "#5 MOVING",
            }])
        }

        with self.assertRaises(StopLoop):
            pipe._smooth_publish_loop()

        self.assertEqual(publisher.call, (
            "cam",
            b"paced-jpeg",
            [FrameOverlay(5, (10, 20, 30, 40), "#5 MOVING", (0, 200, 0))],
            (480, 640),
        ))

    def test_live_sources_play_from_the_playout_buffer(self):
        """Two prior designs both failed the operator: pacing through an HLS
        burst fell progressively behind live ('sluggish', 26 Aug); draining to
        the newest frame replayed each burst at decode speed ('too fast',
        27 Aug). The contract now: live URL sources ingest at arrival speed
        into a BOUNDED playout buffer and the wall pops at content rate. Files
        must still pace themselves to real time."""
        import inspect
        from cvti.serving.streams import StreamDecoder
        src = inspect.getsource(StreamDecoder._loop)
        self.assertIn("self.playout.push(image)", src,
                      "live frames no longer reach the playout buffer")
        self.assertIn("elif self._min_period:", src,
                      "files must still be paced to real time")
        self.assertIn("stale_dropped", src)

    def test_ingest_is_governed_by_fill_level_not_sleep(self):
        """Sleeping through a burst is how the source-side lag grew. The decode
        loop may only sleep while the playout buffer is comfortably empty."""
        import inspect
        from cvti.serving.streams import StreamDecoder
        src = inspect.getsource(StreamDecoder._loop)
        self.assertIn("len(self.playout) <=", src,
                      "the pacing sleep is no longer gated on buffer depth")

    def test_skipped_frames_are_reported_not_hidden(self):
        import inspect
        from cvti.serving.streams import StreamDecoder
        self.assertIn("stale_dropped", inspect.getsource(StreamDecoder.link_status))


class WallFpsCpuCapTests(unittest.TestCase):
    """Twelve watched tiles at 24 fps decoded 288 frames a second of 5-megapixel
    HEVC on the five-vCPU pilot server, for display alone (13 Oct 2026)."""

    def test_a_cpu_box_shares_a_decode_budget_across_cameras(self):
        from cvti.serving.pipeline import wall_fps_for
        self.assertEqual(wall_fps_for({}, 24.0, 12, "cpu"), 8.0)     # 96 // 12 = 8
        self.assertEqual(wall_fps_for({}, 24.0, 4, "cpu"), 24.0)     # 96 // 4 = 24: no cap
        self.assertEqual(wall_fps_for({}, 24.0, 6, "cpu"), 16.0)
        self.assertEqual(wall_fps_for({}, 24.0, 40, "cpu"), 8.0)     # floor

    def test_a_gpu_box_and_an_explicit_site_value_are_left_alone(self):
        from cvti.serving.pipeline import wall_fps_for
        self.assertEqual(wall_fps_for({}, 24.0, 12, "cuda"), 24.0)
        self.assertEqual(wall_fps_for({"wall_fps": 12}, 24.0, 12, "cpu"), 12.0)
        self.assertEqual(wall_fps_for({"wall_fps": 30}, 24.0, 12, "cpu"), 30.0)

    def test_run_site_applies_it_before_building_the_pipeline(self):
        import inspect
        from cvti.serving import pipeline
        self.assertIn("publish_fps = wall_fps_for(site, publish_fps, len(sources)", inspect.getsource(pipeline.run_site))
