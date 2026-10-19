"""Per-camera state for the multi-stream pipeline.

The detector model is shared and batched across all cameras (stateless). The
STATEFUL work is per camera and lives here: each camera keeps its own ByteTrack
tracker, pose-track state, violence/theft detectors, zone monitor, rules engine,
and scene context. Detections for a camera are associated to that camera's
tracks, turned into RawEvents, evaluated against that camera's threat policy,
and emitted as QueuedAlerts for the gate.

This runs the FULL single-stream detector per camera (opt-in via the site
config): zones, pose-based concealment, violence, weapons (needs a shared weapon
model), the theft state machine, and the fine-tuned video-action model (gated,
shared instance). It reuses the exact `cvti/detector/core.py` functions so
behaviour matches single-stream. Shared stateless models (object/pose/weapon/
video) are injected by the pipeline; everything stateful is per camera.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import cv2

from cvti.contracts import RawEvent
from cvti.event_adapters import zone_states_to_events
from cvti.rules.customization import CustomizationEngine
from cvti.serving.alert_queue import QueuedAlert
from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Tuned defaults mirrored from cvti/detector/core.py argparse so the multi-stream
# path behaves like single-stream. Keep in sync if those defaults change.
_VIOLENCE_DISTANCE_RATIO = 1.1
_VIOLENCE_WRIST_SPEED = 120.0
_VIOLENCE_ARM_EXTENSION_RATIO = 0.35
_VIOLENCE_WRIST_ACCEL = 800.0
_WEAPON_HAND_DISTANCE_RATIO = 0.20
_ASSAULT_DISTANCE_RATIO = 1.2
_WEAPON_MIN_AREA_RATIO = 0.002
_WEAPON_MAX_AREA_RATIO = 0.18
_WEAPON_BORDER_MARGIN_RATIO = 0.03
_MIN_THREAT_FRAMES = 3


def _person_boxes(tracked: Any) -> list:
    """Extract (track_id, x1, y1, x2, y2) person boxes from an sv.Detections.

    COCO person class is 0; if no class_id is present (already person-filtered) we
    keep every box. track_id falls back to the row index when the tracker hasn't
    assigned one yet."""
    boxes: list = []
    xyxy = getattr(tracked, "xyxy", None)
    if xyxy is None:
        return boxes
    cls = getattr(tracked, "class_id", None)
    tids = getattr(tracked, "tracker_id", None)
    for i in range(len(xyxy)):
        if cls is not None and cls[i] is not None and int(cls[i]) != 0:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
        tid = int(tids[i]) if tids is not None and tids[i] is not None else i
        boxes.append((tid, x1, y1, x2, y2))
    return boxes


def _to_queued(camera_id: str, alert: Any, timestamp: float, zone: str | None,
               frames: list, scene: dict | None,
               clip_frames: list | None = None, clip_fps: float = 0.0,
               bbox: tuple | None = None, object_watch_token: Any = None,
               object_watch_result: Any = None) -> QueuedAlert:
    # Evidence frames are captured NOW because the async gate verifies later,
    # by which point the live frame is gone.
    #  * `frames`      — a few sharp stills, for the VLM gate + thumbnails.
    #  * `clip_frames` — the continuous JPEG-encoded window (~last N seconds) so the
    #                    sink can write a REAL video of the event, not a slideshow.
    frozen_frames = []
    for frame in frames:
        try:
            frame = frame.copy()
            frame.setflags(write=False)
        except (AttributeError, TypeError, ValueError):
            pass
        frozen_frames.append(frame)
    return QueuedAlert(
        camera_id=camera_id,
        rule_name=alert.rule_name,
        priority=alert.priority,
        title=alert.title,
        timestamp=timestamp,
        track_id=((getattr(alert, "metadata", None) or {}).get("track_id")
                  if getattr(alert, "detector", "") == "object_watch"
                  else alert.person_id),
        zone=zone,
        object_label=alert.object_label,
        payload={"candidate": alert, "frames": frozen_frames, "scene": scene,
                 "clip_frames": clip_frames or [], "clip_fps": clip_fps,
                 "enqueued_at": time.time(),    # wall-clock, for verify-latency
                 # where the subject was when this fired, so evidence can point at
                 # WHO — an alert with no box makes the operator hunt the frame.
                 "bbox": bbox,
                 "object_watch_token": object_watch_token,
                 "object_watch_result": object_watch_result},
    )


# Replay clips are context, not forensic evidence — the gate's frames and the
# saved evidence bundle stay full-resolution elsewhere. 640w is plenty to see
# what led up to an alert, and it is the difference between the per-camera
# replay buffer costing ~7 MB + a full-1080p JPEG encode EVERY frame, and
# ~1.5 MB + an encode a fraction of that size (audit 1 Sep, D4).
CLIP_BUFFER_WIDTH = 640


def encode_clip_frame(image: Any, max_width: int = CLIP_BUFFER_WIDTH) -> bytes | None:
    """One replay-buffer frame: downscaled to <=max_width, JPEG q80.

    Returns None when encoding fails — the replay window simply misses that
    frame, which is the old behaviour for a failed encode."""
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        image = cv2.resize(image, (max_width, max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return enc.tobytes() if ok else None



def _box_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PerCameraState:
    camera_id: str
    engine: CustomizationEngine
    zone_monitor: Any = None          # RetailZoneMonitor | None
    vehicle_zone_monitor: Any = None  # RetailZoneMonitor | None — vehicle presence/dwell in an area
    vehicle_line: Any = None          # dict {name,start,end,normalized,flip} — directional entry/exit tripwire
    scene_context: dict | None = None
    monitoring_scope: str = "full"
    # Human-reviewed context is the ONLY context allowed to suppress alerts
    # (coverage-first hybrid, 1 Sep): an unreviewed AI scene guess still
    # informs the gate and the English rules, but never silences a rule.
    # field(default=...) on purpose: this is state, not a detector toggle —
    # the toggle-coverage scanner matches the `: bool = False` shape.
    scene_reviewed: bool = field(default=False)
    active_zone_roles: set[str] = field(default_factory=set)
    person_filter: bool = True
    # The zone lane's plausibility filter was born in retail (mannequin heads,
    # reflections) and demanded a person be >=1.2% of the frame — on a wide
    # outdoor camera a real person at distance is ~0.3%, so nobody could EVER
    # enter a zone (12 Sep, VIRAT campus demo: raw=1 person, filtered=0,
    # zero presence events). None = derive from the scene: retail keeps the
    # tight gate, everything else gets a far-field one. Site key:
    # "zone_min_person_area_ratio".
    zone_min_person_area_ratio: float | None = None
    # Shared (stateless) models, injected by the pipeline.
    pose_model: Any = None            # LoadedModel | None — needed by concealment/violence/theft
    weapon_model: Any = None          # LoadedModel | None — needed by weapons
    # Per-camera opt-in signals (from the site config).
    concealment: bool = False
    violence: bool = False
    weapons: bool = False
    theft: bool = False
    tamper: bool = False              # camera block/tamper detection (pure CV)
    fall: bool = False                # person collapsed / on the ground (person boxes)
    # HSE situational detectors (cvti.detector.situational) — cheap candidate
    # generators the VLM gate then confirms.
    fire_smoke: bool = False          # fire / smoke visual candidate
    running: bool = False             # sustained fast person movement (panic)
    crowd_formation: bool = False     # tight group formation
    normal_movement: bool = False     # telemetry-only moving-person state
    multiple_people_moving: bool = False
    # Detection counters for health / heartbeat (see detection_status()).
    frames_analysed: int = 0
    frames_with_people: int = 0
    people_in_last_frame: int = 0
    people_last_seen_at: float = 0.0
    object_watch: bool = False
    general_object_tracking: bool = False
    general_object_tracking_overlays: bool = False
    running_min_speed_ratio: float = 0.18
    running_min_frames: int = 3
    crowd_min_people: int = 4
    crowd_min_frames: int = 3
    crowd_max_cluster_ratio: float = 0.24
    fire_min_frames: int = 3
    fire_min_hot_area_ratio: float = 0.012
    fire_rearm_seconds: float = 300.0      # one fire candidate per camera per window unless it doubles
    movement_enter_speed_ratio: float = 0.05
    movement_exit_speed_ratio: float = 0.02
    movement_min_track_seconds: float = 0.4
    movement_min_people: int = 2
    movement_persistence_seconds: float = 0.5
    permitted_movement_zones: tuple[str, ...] | None = None
    object_watch_library: str | None = None
    object_watch_sample_fps: float = 1.0
    object_watch_max_candidates_per_frame: int = 24
    object_watch_min_similarity: float | None = None
    object_watch_open_vocab_provider: str = "disabled"
    video_action: bool = False
    video_action_model: Any = None    # shared VideoMAEActionModel instance
    # Shared AsyncVideoActionRunner (one worker thread per site). When set,
    # inference leaves the frame loop entirely: this camera submits clips and
    # drains verdicts. None = the synchronous legacy path (single-stream CLI).
    va_runner: Any = None
    # Shared worker for the HEAVY per-camera models (pose, weapon). When set,
    # a frame that earns them hands a copy to the worker and returns at once;
    # verdicts are drained on later frames. None = synchronous (CLI, tests).
    # On the pilot server a pose pass took 50-72 s per call on a saturated
    # CPU and, run inline, stalled tracking on every camera (Demi, 13 Oct
    # 2026). The lock keeps the worker and the frame loop off this camera's
    # assessors at the same time; the frame loop never waits for it.
    heavy_runner: Any = None
    _heavy_lock: Any = field(default_factory=threading.Lock, repr=False, compare=False)
    pose_conf: float = 0.35
    weapon_conf: float = 0.40
    # Inference size for the per-camera pose/weapon passes. run_site plumbs
    # its own --imgsz here (audit 1 Sep, D3): the engine spawns with 512, and
    # until 3 Sep this default silently made pose and weapon the only models
    # still paying for 640 — a leak, not a choice.
    imgsz: int = 640
    # The heavy per-camera models (pose, weapon) run every Nth frame instead
    # of every frame — their consumers are temporal accumulators plus the VLM
    # gate, and half the samples still trigger them; the model cost halves.
    heavy_stride: int = 2
    va_fps: float = 5.0
    va_window_seconds: float = 4.0
    va_frames: int = 16
    va_cooldown: float = 2.0
    # --- per-camera stateful bits (not constructor args) ---
    _tracker: Any = field(default=None, init=False, repr=False)
    _vehicle_tracker: Any = field(default=None, init=False, repr=False)
    _vehicle_line_zone: Any = field(default=None, init=False, repr=False)
    # (timestamp, direction, box) of recent tripwire crossings, so a vehicle
    # the tracker re-acquires under a new id is not counted again. See
    # _vehicle_line_events.
    _vehicle_line_recent: list = field(default_factory=list, init=False, repr=False)
    _conceal: Any = field(default=None, init=False, repr=False)
    _violence_gate: Any = field(default=None, init=False, repr=False)
    _theft: Any = field(default=None, init=False, repr=False)
    _prev_pose: list = field(default_factory=list, init=False, repr=False)
    _next_pose_id: int = field(default=1, init=False, repr=False)
    _pose_history: dict = field(default_factory=dict, init=False, repr=False)
    _object_threat_frames: int = field(default=0, init=False, repr=False)
    _weapon_classes: Any = field(default=None, init=False, repr=False)
    _person_classes: Any = field(default=None, init=False, repr=False)
    _video_runtime: Any = field(default=None, init=False, repr=False)
    _va_index: int = field(default=0, init=False, repr=False)
    _heavy_tick: int = field(default=0, init=False, repr=False)
    _person_seen_last_frame: bool = field(default=False, init=False, repr=False)
    _tamper_det: Any = field(default=None, init=False, repr=False)
    _fall_det: Any = field(default=None, init=False, repr=False)
    _fire_det: Any = field(default=None, init=False, repr=False)
    _first_seen_ts: float = field(default=-1.0, init=False, repr=False)
    # Cameras auto-adjust exposure/IR for the first seconds after a stream
    # opens; the fire detector's hot-area heuristic reads that white-out as
    # flame, so EVERY engine start opened with a critical fire alert on the
    # pilot's night camera ('anytime the software starts i get this alert of
    # fire', 30 Aug). Situational detectors sit out the settle window.
    settle_seconds: float = 8.0
    object_state_zones: list = field(default_factory=list)
    registered_objects: list = field(default_factory=list)
    registered_source: str = ""
    _registered_monitor: Any = field(default=None, init=False, repr=False)
    _object_state_monitor: Any = field(default=None, init=False, repr=False)
    _running_det: Any = field(default=None, init=False, repr=False)
    _crowd_det: Any = field(default=None, init=False, repr=False)
    _motion_tracker: Any = field(default=None, init=False, repr=False)
    _simultaneous_movement_det: Any = field(default=None, init=False, repr=False)
    _object_matcher: Any = field(default=None, init=False, repr=False)
    _object_state_tracker: Any = field(default=None, init=False, repr=False)
    _object_watch_runtime: Any = field(default=None, init=False, repr=False)
    _object_watch_rule_path: str | None = field(default=None, init=False, repr=False)
    _object_watch_generation: int = field(default=0, init=False, repr=False)
    _object_watch_sequence: int = field(default=0, init=False, repr=False)
    _general_object_tracker: Any = field(default=None, init=False, repr=False)
    _next_object_watch_sample_ts: float = field(default=-1.0, init=False, repr=False)
    object_watch_skipped_over_budget: int = field(default=0, init=False)
    _motion_overlays: list[dict] = field(default_factory=list, init=False, repr=False)
    context_decisions: list[dict] = field(default_factory=list, init=False)
    context_suppression_count: int = field(default=0, init=False)
    # Rolling recent frames (~2s at 5 FPS) so the gate gets per-rule evidence
    # (motion-peak span for violence, sharpest single frame for weapons).
    _frame_buffer: deque = field(default_factory=lambda: deque(maxlen=10), init=False, repr=False)
    # Longer CONTINUOUS window, JPEG-encoded (kept light for RAM), so a confirmed
    # alert can be replayed as a real video of the event lead-up, not a slideshow.
    # ~48 frames ≈ 8-12s at typical pipeline fps. Holds (timestamp, jpeg_bytes).
    _clip_buffer: deque = field(default_factory=lambda: deque(maxlen=48), init=False, repr=False)

    def activate_scene_context(self, context: dict) -> None:
        """Apply reviewed semantic context without rebuilding detector state."""
        self.scene_context = dict(context)
        self.monitoring_scope = "full"
        self.scene_reviewed = True

    def __post_init__(self) -> None:
        import warnings
        import supervision as sv
        from cvti.health import component
        # Per camera, not per detector class: an operator asks "is camera 4 ok?",
        # and a camera whose detectors throw on every frame looks exactly like a
        # camera watching a quiet corridor.
        self._health = component(f"detector.{self.camera_id}")
        from cvti.detector.core import normalize_threat_classes
        # sv.ByteTrack is deprecation-proxied in supervision 0.28 (removed in
        # 0.30). It still works; silence the per-camera warning spam for now.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack()
            # Vehicles get their OWN tracker: mixing them with the person track
            # ids (and the person-size filter) would corrupt both. Only built
            # when the camera actually has vehicle zones.
            if self.vehicle_zone_monitor is not None or self.vehicle_line is not None:
                self._vehicle_tracker = sv.ByteTrack()
        self._weapon_classes = normalize_threat_classes("gun,knife")
        self._person_classes = normalize_threat_classes("person")
        if self.concealment and self.pose_model is not None:
            from cvti.retail.concealment import ConcealmentDetector
            self._conceal = ConcealmentDetector()
        if self.violence and self.pose_model is not None:
            from cvti.detector.core import ViolenceTemporalGate
            self._violence_gate = ViolenceTemporalGate()
        if self.theft and self.pose_model is not None:
            from cvti.detector.core import TheftDetector
            self._theft = TheftDetector()
        if self.video_action and self.video_action_model is not None:
            from cvti.video_action_runtime import VideoActionRuntime
            self._video_runtime = VideoActionRuntime(
                model=self.video_action_model, backend="videomae",
                model_name=getattr(self.video_action_model, "model_name", "videomae"),
                fps=self.va_fps, window_seconds=self.va_window_seconds,
                frame_count=self.va_frames, cooldown_seconds=self.va_cooldown)
        if self.normal_movement or self.multiple_people_moving:
            from cvti.detector.person_motion import (
                PersonMotionTracker,
                SimultaneousMovementDetector,
            )
            self._motion_tracker = PersonMotionTracker(
                perspective_compensation=True,
                enter_speed_ratio=self.movement_enter_speed_ratio,
                exit_speed_ratio=self.movement_exit_speed_ratio,
                min_track_seconds=self.movement_min_track_seconds,
            )
            self._simultaneous_movement_det = SimultaneousMovementDetector(
                min_people=self.movement_min_people,
                persistence_seconds=self.movement_persistence_seconds,
            )
        # Object recognition is attached by MultiStreamPipeline as one shared
        # site runtime.  Never construct a backend/model in camera setup or in
        # this camera's frame loop.

    def _movement_zone_allows(self, zone_names: tuple[str, ...]) -> bool:
        if self.permitted_movement_zones is None:
            return True
        return bool(set(zone_names).intersection(self.permitted_movement_zones))

    def ensure_general_object_tracker(self, names: Any, expected_fps: float) -> None:
        """Create this camera's tracker once the shared model names are known."""
        if not self.general_object_tracking or self._general_object_tracker is not None:
            return
        from cvti.detector.object_tracks import GeneralObjectTracker

        if not isinstance(names, dict):
            names = {index: str(label) for index, label in enumerate(names)}
        self._general_object_tracker = GeneralObjectTracker(
            camera_id=self.camera_id,
            session_id=uuid.uuid4().hex,
            names={int(class_id): str(label) for class_id, label in names.items()},
            expected_fps=expected_fps,
        )

    def update_general_object_tracks(self, detections: Any, timestamp: float) -> None:
        if detections is None and self._registered_monitor is not None:
            self._registered_monitor.invalidate("person_detection_unavailable")
        if detections is None and self._object_state_monitor is not None:
            self._object_state_monitor.reset()
        if self._general_object_tracker is not None:
            self._general_object_tracker.update(detections, timestamp)

    def general_object_snapshot(self, timestamp: float) -> dict | None:
        if self._general_object_tracker is None:
            return None
        snapshot = self._general_object_tracker.snapshot(timestamp)
        # Lifecycle values are monotonic durations, deliberately immune to NTP
        # and manual wall-clock corrections. Consumers must not interpret them
        # as Unix timestamps; freshness bounds let the publisher age a cached
        # snapshot even when no more frames arrive.
        snapshot["timestamp_clock"] = "monotonic"
        snapshot["freshness"] = {
            "overlay_max_age_seconds": self._general_object_tracker.overlay_max_age_seconds,
            "lost_after_seconds": self._general_object_tracker.lost_after_seconds,
            "ended_retention_seconds": self._general_object_tracker.ended_retention_seconds,
        }
        return snapshot

    def general_object_overlays(self, timestamp: float) -> list[dict]:
        if self._general_object_tracker is None or not self.general_object_tracking_overlays:
            return []
        return self._general_object_tracker.overlays(timestamp)

    def reset_general_object_tracks(self, timestamp: float,
                                    reason: str = "source_reset") -> None:
        if self._general_object_tracker is not None:
            self._general_object_tracker.reset(timestamp, reason)

    def _needs_pose(self) -> bool:
        return self.pose_model is not None and (self.concealment or self.violence or self.theft)

    def _compute_pose(self, image: Any, timestamp: float) -> list:
        """Shared pose model on this camera's frame + per-camera track state, so
        wrist speed/dwell history is not mixed across cameras."""
        from cvti.detector.core import (
            assign_pose_tracks, enrich_pose_people_with_history, extract_pose_people,
        )
        from cvti.serving.perf import BOARD
        started = time.perf_counter()
        try:
            pose_people = extract_pose_people(
                self.pose_model, image, self.pose_conf, self.imgsz
            )
        finally:
            BOARD.observe(
                "pose_infer", self.camera_id,
                (time.perf_counter() - started) * 1000.0,
            )
        pose_people, self._next_pose_id = assign_pose_tracks(
            pose_people, previous_people=self._prev_pose, next_track_id=self._next_pose_id)
        pose_people = enrich_pose_people_with_history(pose_people, self._pose_history)
        self._prev_pose = list(pose_people)
        return pose_people

    def _heavy_pass(self, image: Any, timestamp: float, bag_boxes: list,
                    object_detections: list | None, *, run_heavy: bool) -> list:
        """Pose (+ concealment) and the weapon model for one frame. With
        run_heavy False this is the light path: rule logic on the shared
        detections only, no model call."""
        raw_events: list = []
        pose_ran = self._needs_pose() and run_heavy
        pose_people = self._compute_pose(image, timestamp) if pose_ran else []
        if self._conceal is not None and pose_ran:
            from cvti.detector.core import pose_people_to_concealment_frames
            from cvti.event_adapters import concealment_to_events
            pose_frames = pose_people_to_concealment_frames(pose_people, timestamp)
            assessments = self._conceal.update_with_bag_detections(
                pose_frames, timestamp, bag_boxes
            )
            raw_events += concealment_to_events(assessments, timestamp)
        if self.violence or self.weapons or self.theft:
            merged = self._merged_detections(object_detections, image,
                                             include_weapon=run_heavy)
            raw_events += self._assessment_events(pose_people, merged, image, timestamp)
        return raw_events

    def _heavy_submit_or_light(self, run_heavy: bool, image: Any, timestamp: float,
                               bag_boxes: list, object_detections: list | None) -> list:
        """With a heavy_runner: a frame that earned the heavy models hands a
        copy to the worker and returns at once. Any other frame runs the
        light path here, unless the worker is mid-pass on this camera."""
        wants_heavy = run_heavy and (self._needs_pose() or (
            self.weapon_model is not None and (self.violence or self.weapons or self.theft)))
        if wants_heavy:
            snapshot = (image.copy(), timestamp, list(bag_boxes or []),
                        list(object_detections or []))
            lock = self._heavy_lock

            def job(snap=snapshot):
                with lock:
                    return self._heavy_pass(*snap, run_heavy=True)
            self.heavy_runner.submit(self.camera_id, job)
            return []
        if not (self.violence or self.weapons or self.theft):
            return []
        if not self._heavy_lock.acquire(blocking=False):
            return []                 # the worker holds this camera's assessors
        try:
            return self._heavy_pass(image, timestamp, bag_boxes, object_detections,
                                    run_heavy=False)
        finally:
            self._heavy_lock.release()

    def _merged_detections(self, object_detections: list | None, image: Any,
                           include_weapon: bool = True) -> list:
        """Shared object detections + (optional) per-camera weapon-model detections.

        `include_weapon=False` skips the weapon model's forward pass (the
        person-gate/cadence below decided this frame doesn't earn one) while
        still handing the assessments the shared detections."""
        merged = list(object_detections or [])
        if include_weapon and self.weapon_model is not None:
            from cvti.detector.core import merge_detections, predict_with_model
            weap = predict_with_model(self.weapon_model, image, self.weapon_conf, self.imgsz,
                                      self._weapon_classes, source_model="weapon")
            merged = merge_detections(merged, weap)
        return merged

    def _assessment_events(self, pose_people: list, merged: list, image: Any,
                           timestamp: float) -> list:
        """Run weapons / violence / theft assessments and adapt to RawEvents,
        reusing the single-stream core.py logic."""
        from cvti.detector.core import (
            ThreatAssessment, assess_threat, assess_violence, gate_assessment,
            validate_weapon_detections,
        )
        from cvti.event_adapters import assessments_to_events

        validated_weapons: list = []
        if (self.violence or self.weapons):
            validated_weapons = validate_weapon_detections(
                detections=merged, weapon_classes=self._weapon_classes,
                person_classes=self._person_classes, pose_people=pose_people,
                frame_shape=image.shape, weapon_min_area_ratio=_WEAPON_MIN_AREA_RATIO,
                weapon_max_area_ratio=_WEAPON_MAX_AREA_RATIO,
                weapon_border_margin_ratio=_WEAPON_BORDER_MARGIN_RATIO,
                weapon_hand_distance_ratio=_WEAPON_HAND_DISTANCE_RATIO,
                allow_unattached_weapons=False)

        object_assessment = violence_assessment = theft_assessment = None
        if self.weapons:
            raw = assess_threat(merged, self._weapon_classes, self._person_classes,
                                validated_weapons, _ASSAULT_DISTANCE_RATIO)
            self._object_threat_frames = self._object_threat_frames + 1 if raw.active else 0
            object_assessment = gate_assessment(raw, self._object_threat_frames, _MIN_THREAT_FRAMES)
        if self.violence:
            raw_v = assess_violence(
                pose_people=pose_people, validated_weapon_detections=validated_weapons,
                violence_distance_ratio=_VIOLENCE_DISTANCE_RATIO,
                violence_wrist_speed=_VIOLENCE_WRIST_SPEED,
                violence_arm_extension_ratio=_VIOLENCE_ARM_EXTENSION_RATIO,
                weapon_hand_distance_ratio=_WEAPON_HAND_DISTANCE_RATIO,
                violence_wrist_accel=_VIOLENCE_WRIST_ACCEL)
            violence_assessment = self._violence_gate.update(raw_v)
        if self.theft:
            theft_assessment = self._theft.update(pose_people, merged, timestamp)

        if object_assessment is None and violence_assessment is None and theft_assessment is None:
            return []
        return assessments_to_events(object_assessment, violence_assessment, theft_assessment,
                                     timestamp=timestamp, theft_detector=self._theft)

    def _vehicle_events(self, detections: Any, frame_hw: tuple, timestamp: float) -> list:
        """Track vehicles (car/truck/bus/motorcycle) and emit vehicle_* events.

        Two independent mechanisms, either or both per camera:
        - `vehicle_line`  — a directional TRIPWIRE (sv.LineZone). A vehicle
          crossing the line one way is an ENTRY, the other way an EXIT, counted
          once per track. The robust default for a gate/driveway: a PARKED car
          never crosses it, and detector flicker cannot fabricate a crossing
          (the track must appear on both sides). Replaces the polygon-dwell
          approach that spammed on parked cars and missed passing traffic
          (14 Sep field: 'strengthen the vehicle entry and exit').
        - `vehicle_zone_monitor` — a polygon for 'a vehicle is IN this area /
          parked too long' (dwell), kept for that separate use case.
        """
        import numpy as np
        import supervision as sv
        from cvti.event_adapters import (vehicle_states_to_events,
                                         vehicle_exits_to_events)
        VEHICLE_CLASSES = {2, 3, 5, 7}   # COCO: car, motorcycle, bus, truck
        cls = getattr(detections, "class_id", None)
        if cls is None or len(detections) == 0:
            veh = sv.Detections.empty()
        else:
            keep = np.array([int(c) in VEHICLE_CLASSES for c in cls], dtype=bool)
            veh = detections[keep]
        tracked = self._vehicle_tracker.update_with_detections(veh)
        events: list = []
        if self.vehicle_line is not None:
            events += self._vehicle_line_events(tracked, frame_hw, timestamp)
        if self.vehicle_zone_monitor is not None:
            states = self.vehicle_zone_monitor.update(tracked, timestamp, frame_hw=frame_hw)
            events += vehicle_states_to_events(states, timestamp=timestamp)
            exits = self.vehicle_zone_monitor.drain_exits()
            if exits:
                events += vehicle_exits_to_events(exits, timestamp=timestamp)
        return events

    # A stop-and-go gate is adversarial for a tripwire: a car waiting ~20s at a
    # barrier arm is lost and re-acquired by the tracker several times, and
    # every fresh id that then inches over the line is a 'new' crossing.
    # Replayed on the barrier clip, 21 Sep: ONE car -> 6 tracker ids -> 5
    # same-direction crossings, all within a few px of the same spot. Time
    # and track-age guards barely helped (5 -> 4): the gaps were 7-8s. What
    # identifies the same vehicle is WHERE it is: a new id whose box overlaps
    # where a recently-crossed track was last seen is that vehicle again.
    # With this guard the same replay counts 1. (The #145 idea -- a fresh id
    # at that spot inherits -- applied to lines.)
    VEHICLE_LINE_MEMORY_S = 30.0
    VEHICLE_LINE_SAME_IOU = 0.3

    def _vehicle_line_events(self, tracked: Any, frame_hw: tuple, timestamp: float) -> list:
        import supervision as sv
        if self._vehicle_line_zone is None:
            cfg = self.vehicle_line
            h, w = frame_hw
            s, e = cfg["start"], cfg["end"]
            norm = bool(cfg.get("normalized", max(s[0], s[1], e[0], e[1]) <= 1.0))
            sx, sy = (s[0] * w, s[1] * h) if norm else (s[0], s[1])
            ex, ey = (e[0] * w, e[1] * h) if norm else (e[0], e[1])
            # A vehicle's CENTRE crossing the line is the crossing — not its box
            # corners (the default), which fire early/twice on a large vehicle.
            self._vehicle_line_zone = sv.LineZone(
                start=sv.Point(int(sx), int(sy)), end=sv.Point(int(ex), int(ey)),
                triggering_anchors=(sv.Position.CENTER,))
        if tracked is None or len(tracked) == 0:
            return []
        crossed_in, crossed_out = self._vehicle_line_zone.trigger(tracked)
        name = self.vehicle_line.get("name", "gate")
        flip = bool(self.vehicle_line.get("flip", False))
        tids = tracked.tracker_id
        out: list = []
        # forget crossings older than the memory window. getattr, not the
        # attribute: tests and rebinds build PerCameraState with __new__ and
        # skip the dataclass init, the same reason the object-watch fields
        # are read this way.
        recent = getattr(self, "_vehicle_line_recent", None) or []
        self._vehicle_line_recent = [r for r in recent
                                     if timestamp - r[0] <= self.VEHICLE_LINE_MEMORY_S]
        for i in range(len(tracked)):
            tid = int(tids[i]) if tids is not None and tids[i] is not None else None
            ci, co = bool(crossed_in[i]), bool(crossed_out[i])
            if flip:
                ci, co = co, ci
            if not (ci or co):
                continue
            box = tuple(float(v) for v in tracked.xyxy[i][:4])
            direction = "in" if ci else "out"
            if any(d == direction and _box_iou(box, b) >= self.VEHICLE_LINE_SAME_IOU
                   for _, d, b in self._vehicle_line_recent):
                # the same vehicle, re-acquired under a new id: already counted
                continue
            self._vehicle_line_recent.append((timestamp, direction, box))
            if ci:
                out.append(RawEvent(detector="vehicle_entry", active=True,
                    title=f"VEHICLE ENTERED VIA {name.upper()}", level="high",
                    person_id=tid, object_label="vehicle", timestamp=timestamp,
                    extra={"zone": name, "via": "line"}))
            if co:
                out.append(RawEvent(detector="vehicle_exit", active=True,
                    title=f"VEHICLE EXITED VIA {name.upper()}", level="medium",
                    person_id=tid, object_label="vehicle", timestamp=timestamp,
                    extra={"zone": name, "via": "line"}))
        return out
    def _object_watch_due(self, timestamp: float) -> bool:
        next_sample = getattr(self, "_next_object_watch_sample_ts", -1.0)
        sample_fps = float(getattr(self, "object_watch_sample_fps", 1.0) or 1.0)
        if next_sample < 0:
            self._next_object_watch_sample_ts = timestamp + (1.0 / sample_fps)
            return True
        if timestamp + 1e-9 < next_sample:
            return False
        self._next_object_watch_sample_ts = timestamp + (1.0 / sample_fps)
        return True

    def attach_object_watch_runtime(self, runtime: Any) -> None:
        self._object_watch_runtime = runtime

    def reset_object_watch(self, source_generation: int) -> None:
        if self._registered_monitor is not None:
            self._registered_monitor.invalidate()
        if self._object_state_monitor is not None:
            self._object_state_monitor.reset()
        self._object_watch_generation = int(source_generation)
        self._object_watch_sequence = 0
        self._next_object_watch_sample_ts = -1.0
        runtime = getattr(self, "_object_watch_runtime", None)
        if runtime is not None:
            runtime.reset_camera(self.camera_id, self._object_watch_generation)

    def object_watch_status(self) -> dict:
        runtime = getattr(self, "_object_watch_runtime", None)
        if not getattr(self, "object_watch", False):
            return {"status": "disabled"}
        if runtime is None:
            return {"status": "unavailable", "reason": "object watch runtime unavailable"}
        return dict(runtime.status())

    def _object_candidates(
        self,
        object_detections: list | None,
        frame_hw: tuple[int, int],
    ) -> list:
        from cvti.object_watch.matcher import ObjectCandidate

        candidates = []
        for detection in object_detections or []:
            label = str(getattr(detection, "label", "") or "")
            if label.lower() == "person":
                continue
            bbox = tuple(int(v) for v in getattr(detection, "bbox", ()))
            if len(bbox) != 4:
                continue
            candidates.append(ObjectCandidate(
                bbox=bbox,
                label_hint=label,
                confidence=float(getattr(detection, "confidence", 0.0) or 0.0),
                track_id=getattr(detection, "track_id", None),
                zone_id=self._zone_for_bbox(bbox, frame_hw),
            ))
        return candidates

    def _zone_for_bbox(self, bbox: tuple[int, int, int, int],
                       frame_hw: tuple[int, int]) -> str | None:
        if self.zone_monitor is None:
            return None
        try:
            import cv2
            self.zone_monitor._fit_to_frame(frame_hw)
            x1, y1, x2, y2 = bbox
            point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            for spec in self.zone_monitor.zones:
                if cv2.pointPolygonTest(spec.polygon, point, False) >= 0:
                    return spec.name
        except Exception:  # noqa: BLE001 - zone hinting must not block object matching
            log.debug("object zone hinting failed for %s", self.camera_id,
                      exc_info=True)
            return None
        return None
    def _object_watch_zone_snapshot(self, frame_hw: tuple[int, int]) -> tuple:
        """Copy fitted zone polygons for the background object-watch worker."""
        if self.zone_monitor is None:
            return ()
        try:
            from cvti.object_watch.runtime import WatchZone

            self.zone_monitor._fit_to_frame(frame_hw)
            return tuple(
                WatchZone(
                    str(spec.name),
                    tuple((int(point[0]), int(point[1])) for point in spec.polygon),
                )
                for spec in self.zone_monitor.zones
            )
        except Exception:  # noqa: BLE001 - zone hinting must not block object matching
            log.debug("object zone snapshot failed for %s", self.camera_id,
                      exc_info=True)
            return ()

    def process(self, detections: Any, image: Any, timestamp: float,
                object_detections: list | None = None, *,
                object_watch_source_generation: int | None = None,
                object_watch_observed_at: float | None = None) -> list[QueuedAlert]:
        """Track + run all enabled signals (zones, concealment, violence, weapons,
        theft) + rules; return candidate alerts with per-rule evidence frames.

        `detections` is sv.Detections (tracking/zones); `object_detections` is the
        core.py Detection list from the same frame (weapons/violence/theft)."""
        from cvti.retail.concealment import personal_bag_boxes
        from cvti.retail.zones import filter_person_detections

        bag_boxes = personal_bag_boxes(detections) if self._conceal is not None else []
        frame_hw = image.shape[:2]
        if self._clip_buffer and timestamp < self._clip_buffer[-1][0]:
            self._frame_buffer.clear()
            self._clip_buffer.clear()
            self._prev_pose.clear()
            self._pose_history.clear()
            if self._conceal is not None:
                self._conceal.expire(timestamp)
        self._frame_buffer.append(image)
        # Continuous replay buffer: a rolling JPEG window with timestamps so a
        # confirmed alert replays as real video of the lead-up.
        clip_jpeg = encode_clip_frame(image)
        if clip_jpeg is not None:
            self._clip_buffer.append((timestamp, clip_jpeg))

        # Camera tamper/block runs on the raw frame — independent of any person,
        # since a covered camera shows nothing. Cheap CV, every frame.
        raw_events: list = []
        object_state_evidence = {}
        registered_candidates = []
        if self.registered_objects:
            from cvti.serving.registered_objects import RegisteredObjects
            if self._registered_monitor is None:
                self._registered_monitor = RegisteredObjects(self.camera_id, self.registered_source, self.registered_objects)
            try:
                for candidate, panel in self._registered_monitor.update(image, object_detections, timestamp):
                    registered_candidates.append(candidate)
                    object_state_evidence[(candidate.metadata["zone"], candidate.metadata["state"])] = panel
            except Exception as exc:
                self._registered_monitor.invalidate("monitor_error")
                self._health.failed(exc, log, "registered object monitoring")
        if self.object_state_zones:
            from cvti.detector.object_state import ObjectZoneMonitor
            if self._object_state_monitor is None:
                self._object_state_monitor = ObjectZoneMonitor(self.object_state_zones)
            try:
                state_events = self._object_state_monitor.update(
                    image, object_detections, timestamp,
                    self._object_watch_zone_snapshot(frame_hw),
                )
            except Exception as exc:  # a failed optional detector must not stop camera processing
                self._object_state_monitor.reset()
                self._health.failed(exc, log, "processing object state")
                state_events = []
            for event, panel in state_events:
                raw_events.append(event)
                object_state_evidence[(event.extra["zone"], event.state)] = panel
        if self.tamper:
            if self._tamper_det is None:
                from cvti.detector.tamper import TamperDetector
                self._tamper_det = TamperDetector()
            t = self._tamper_det.update(image)
            if t is not None:
                raw_events.append(RawEvent(
                    detector="camera_tampering", active=True,
                    title=f"CAMERA BLOCKED ({t['kind']})", level="high",
                    timestamp=timestamp, extra=t))

        if self._first_seen_ts < 0:
            self._first_seen_ts = timestamp
        settled = (timestamp - self._first_seen_ts) >= self.settle_seconds

        # Fire + smoke — pure CV pre-filter on the raw frame (no person needed).
        if self.fire_smoke and settled:
            if self._fire_det is None:
                from cvti.detector.situational import FireSmokeCandidateDetector
                self._fire_det = FireSmokeCandidateDetector(
                    min_frames=self.fire_min_frames,
                    min_hot_area_ratio=self.fire_min_hot_area_ratio,
                    rearm_seconds=self.fire_rearm_seconds)
            f = self._fire_det.update(image, timestamp)
            if f is not None:
                raw_events.append(RawEvent(
                    detector="fire", active=True, title="POSSIBLE FIRE OR SMOKE",
                    level="critical", timestamp=timestamp, extra=f))

        if self._video_runtime is not None:
            self._video_runtime.add_frame(image, frame_index=self._va_index)
            self._va_index += 1
        # Vehicles first, from the RAW detections — the person filter below
        # reassigns `detections` to people only, so vehicle boxes must be taken
        # before that. COCO: car=2, motorcycle=3, bus=5, truck=7.
        if self.vehicle_zone_monitor is not None or self.vehicle_line is not None:
            veh_events = self._vehicle_events(detections, frame_hw, timestamp)
            raw_events += veh_events

        if self.person_filter and self.zone_monitor is not None:
            ratio = self.zone_min_person_area_ratio
            if ratio is None:
                env = (self.scene_context or {}).get("environment_type", "")
                ratio = 0.012 if env in ("retail_shop", "mall_corridor") else 0.002
            detections = filter_person_detections(detections, frame_hw,
                                                  min_area_ratio=ratio)
        tracked = self._tracker.update_with_detections(detections)

        # Track -> box for EVERY frame (cheap): lets an alert record where its
        # subject was, so evidence can point at who rather than at a whole frame.
        self._box_by_track = {tid: (int(x1), int(y1), int(x2), int(y2))
                              for (tid, x1, y1, x2, y2) in _person_boxes(tracked)}
        # What the model actually saw, counted: "is detection working on this
        # camera" has only ever been answerable by someone walking in front
        # of it while someone else watched (CHI, 19 Oct 2026). These travel in
        # health and in the heartbeat; counts only, never pixels.
        self.frames_analysed += 1
        people_now = len(self._box_by_track)
        self.people_in_last_frame = people_now
        if people_now:
            self.frames_with_people += 1
            self.people_last_seen_at = time.time()

        # Fall / panic-running / crowd-formation work off tracked PERSON boxes.
        if self.fall or self.running or self.crowd_formation:
            person_boxes = _person_boxes(tracked)   # [(tid, x1, y1, x2, y2), ...]
            frame_area = float(frame_hw[0] * frame_hw[1])
            people = [{"track_id": tid, "bbox": (int(x1), int(y1), int(x2), int(y2))}
                      for (tid, x1, y1, x2, y2) in person_boxes]
            if self.fall:
                if self._fall_det is None:
                    from cvti.detector.fall import FallDetector
                    self._fall_det = FallDetector()
                fl = self._fall_det.update(person_boxes, frame_area, timestamp)
                if fl is not None:
                    raw_events.append(RawEvent(
                        detector="person_fall", active=True, title="PERSON COLLAPSED",
                        person_id=fl.get("track_id"), level="critical",
                        timestamp=timestamp, extra=fl))
            if self.running and settled:
                if self._running_det is None:
                    from cvti.detector.situational import RunningPanicDetector
                    self._running_det = RunningPanicDetector(
                        min_speed_ratio=self.running_min_speed_ratio,
                        min_frames=self.running_min_frames)
                for p in people:
                    r = self._running_det.update(p["track_id"], p["bbox"], timestamp, image.shape)
                    if r is not None:
                        raw_events.append(RawEvent(
                            detector="running", active=True, title="PANIC RUNNING DETECTED",
                            person_id=p["track_id"], level="high", timestamp=timestamp, extra=r))
            if self.crowd_formation and settled:
                # Count from the RAW detections, not the tracked ones: ByteTrack
                # exists to give people stable identities, and it drops
                # low-confidence boxes to do that. In a packed scene those dropped
                # boxes ARE the crowd, so tracking starved the count to zero.
                raw_people = [{"track_id": i, "bbox": (int(x1), int(y1), int(x2), int(y2))}
                              for i, (_t, x1, y1, x2, y2) in enumerate(_person_boxes(detections))]
                if self._crowd_det is None:
                    from cvti.detector.situational import CrowdFormationDetector
                    self._crowd_det = CrowdFormationDetector(
                        min_people=self.crowd_min_people, min_frames=self.crowd_min_frames,
                        max_cluster_ratio=self.crowd_max_cluster_ratio)
                c = self._crowd_det.update(raw_people or people, timestamp, image.shape)
                if c is not None:
                    raw_events.append(RawEvent(
                        detector="crowd_formation", active=True, title="UNSAFE CROWD FORMATION",
                        level="medium", timestamp=timestamp, extra=c))

        zone_by_pid: dict[Any, str | None] = {}   # person_id -> zone, for presence alerts
        zones_by_track: dict[int, tuple[str, ...]] = {}
        if self.zone_monitor is not None:
            # frame_hw lets normalized (0..1) zone polygons fit THIS camera's
            # resolution, and tells the monitor where the frame edges are.
            states = self.zone_monitor.update(tracked, timestamp, frame_hw=frame_hw)
            zone_events = zone_states_to_events(states, timestamp=timestamp)
            # Exits are debounced inside the monitor; drain them each frame so a
            # person leaving a restricted zone is its own event, not silence.
            exits = self.zone_monitor.drain_exits()
            if exits:
                from cvti.event_adapters import zone_exits_to_events
                zone_events = zone_events + zone_exits_to_events(exits, timestamp=timestamp)
            raw_events += zone_events
            zone_by_pid = {e.person_id: e.extra.get("zone") for e in zone_events
                           if e.extra.get("zone")}
            zones_by_track = {
                int(state.tracker_id): tuple(state.zones)
                for state in states
                if state.tracker_id is not None
            }
        # Published per frame like _box_by_track: the off-path scanners (PPE
        # compliance) need WHICH ZONE each person stands in to know what the
        # site requires of them there.
        self._zones_by_track = zones_by_track

        if self._motion_tracker is not None:
            motions = self._motion_tracker.update(
                _person_boxes(tracked), timestamp, frame_hw,
                zones_by_track=zones_by_track,
            )
            permitted_motions = [
                motion for motion in motions
                if self._movement_zone_allows(motion.zone_names)
            ]
            if self.multiple_people_moving:
                movement_event = self._simultaneous_movement_det.update(
                    permitted_motions, timestamp
                )
                if movement_event is not None:
                    from cvti.event_adapters import simultaneous_movement_to_event
                    raw_events.append(
                        simultaneous_movement_to_event(movement_event, timestamp)
                    )
            active_track_ids = (
                set(self._simultaneous_movement_det.active_track_ids)
                if self.multiple_people_moving else set()
            )
            self._motion_overlays = [
                {
                    "track_id": motion.track_id,
                    "bbox": tuple(int(v) for v in motion.bbox),
                    "label": f"#{motion.track_id} {'MOVING' if motion.moving else 'TRACKED'}",
                    "zone_names": list(motion.zone_names),
                    "speed_ratio": motion.speed_ratio,
                    "colour": ((0, 200, 255) if motion.track_id in active_track_ids
                               else (0, 200, 0) if motion.moving else (200, 180, 100)),
                }
                for motion in permitted_motions
                if motion.observed
                and (self.normal_movement or (motion.moving and motion.track_id in active_track_ids))
            ]
            # Tracking association can take several samples or fail at low FPS.
            # Show current detections without inventing identities or motion.
            # Explicit permitted-zone views still require tracked zone membership.
            if self.normal_movement and self.permitted_movement_zones is None:
                tracked_boxes = list(self._box_by_track.values())
                for index, (box, class_id) in enumerate(zip(
                        detections.xyxy, detections.class_id if detections.class_id is not None else [])):
                    if class_id != 0 or any(_box_iou(box, other) >= 0.5 for other in tracked_boxes):
                        continue
                    self._motion_overlays.append({
                        "track_id": -(index + 1), "namespace": "person_detection",
                        "bbox": tuple(int(v) for v in box), "label": "PERSON",
                        "colour": (200, 180, 100), "zone_names": [], "speed_ratio": 0.0,
                    })

        try:
            if self._conceal is not None and self._heavy_lock.acquire(blocking=False):
                try:
                    self._conceal.expire(timestamp)
                finally:
                    self._heavy_lock.release()
            # Person-gate + cadence for the HEAVY per-camera models (audit
            # 1 Sep, D3). Pose and weapon are full forward passes per camera
            # per frame, and they ran unconditionally — on empty corridors,
            # on parked cars, all night. Everything they feed is about a
            # person doing something, so: no person in the shared detector's
            # frame, no heavy models. With people present they run on a
            # stride (every Nth frame — their consumers are temporal
            # accumulators and the VLM gate, which survive half the samples),
            # except the first frame a person APPEARS, which always runs so a
            # threat's opening moment is never the one we skipped. The gate
            # reads the RAW detections, not the tracked ones — ByteTrack
            # drops low-confidence boxes, and a half-seen person is exactly
            # who the weapon model should look at.
            person_present = bool(_person_boxes(detections))
            first_appearance = person_present and not self._person_seen_last_frame
            self._person_seen_last_frame = person_present
            self._heavy_tick += 1
            run_heavy = person_present and (
                first_appearance or self._heavy_tick % max(1, self.heavy_stride) == 0)
            if self.heavy_runner is not None:
                raw_events += self.heavy_runner.drain(self.camera_id)
                raw_events += self._heavy_submit_or_light(
                    run_heavy, image, timestamp, bag_boxes, object_detections)
            else:
                raw_events += self._heavy_pass(image, timestamp, bag_boxes, object_detections,
                                               run_heavy=run_heavy)
            # Video-action runs on a CADENCE, not only when another signal fired.
            # Gating it behind concealment meant theft that isn't concealment-
            # shaped (e.g. a grab-and-go) was never seen by the fine-tuned model —
            # even when it would catch it at high confidence. The runtime's own
            # cooldown_seconds throttles how often clips are queued, so calling
            # every frame is cheap.
            if self._video_runtime is not None:
                if self.va_runner is not None:
                    # Off the frame loop (audit 1 Sep, D1): the 2–6s forward
                    # pass used to run HERE, stalling every camera on the
                    # site. Now this frame only collects verdicts that
                    # finished since last frame and queues a clip snapshot;
                    # the shared worker does the waiting.
                    raw_events += self.va_runner.drain(self.camera_id)
                    clip = self._video_runtime.prepare_analysis(
                        center_frame_index=self._va_index - 1, timestamp=timestamp)
                    if clip is not None:
                        self.va_runner.submit(self.camera_id, clip)
                else:
                    raw_events += self._video_runtime.analyze_event(
                        center_frame_index=self._va_index - 1, timestamp=timestamp)
            self._health.ok()
        except Exception as exc:  # noqa: BLE001 - a detector hiccup must not kill the camera
            # Rate-limited: a detector throwing on every frame must not fill the
            # disk with its own traceback. The counter carries the true scale.
            self._health.failed(exc, log, "processing a frame")

        object_watch_results: dict[tuple[str, Any, int], Any] = {}
        runtime = getattr(self, "_object_watch_runtime", None)
        generation = (getattr(self, "_object_watch_generation", 0)
                      if object_watch_source_generation is None
                      else int(object_watch_source_generation))
        observed_at = (time.monotonic() if object_watch_observed_at is None
                       else float(object_watch_observed_at))
        if self.object_watch and runtime is not None:
            try:
                from cvti.object_watch.runtime import WatchSample
                from cvti.serving.event_adapters import object_watch_result_events

                # Drain every frame, independently of sampling cadence, so a
                # finished worker result reaches rules promptly.
                for result in runtime.drain(self.camera_id, generation, time.monotonic()):
                    matches = tuple(getattr(result, "matches", ()) or ())
                    if self.object_watch_min_similarity is not None:
                        matches = tuple(m for m in matches
                                        if m.similarity >= self.object_watch_min_similarity)
                    if not matches:
                        continue
                    if matches != tuple(result.matches):
                        from dataclasses import replace
                        result = replace(result, matches=matches)
                    events = object_watch_result_events(result)
                    if events:
                        for event in events:
                            event.extra["runtime_config_stamp"] = (
                                result._config_signature[0]
                                if result._config_signature else None
                            )
                        match = matches[0]
                        object_watch_results[(
                            str(match.object_id), match.track_id,
                            int(result.sample_sequence),
                        )] = result
                        raw_events.extend(events)
                if self._object_watch_due(timestamp):
                    self._object_watch_sequence = getattr(
                        self, "_object_watch_sequence", 0
                    ) + 1
                    candidates = tuple(self._object_candidates(object_detections, frame_hw))
                    zones = self._object_watch_zone_snapshot(frame_hw)
                    runtime.submit(WatchSample(
                        camera_id=self.camera_id,
                        source_generation=generation,
                        sample_sequence=self._object_watch_sequence,
                        observed_at_monotonic=observed_at,
                        event_timestamp=timestamp,
                        frame=image,
                        candidates=candidates,
                        zones=zones,
                        proposal_provider=self.object_watch_open_vocab_provider,
                        max_candidates=self.object_watch_max_candidates_per_frame,
                    ))
            except Exception as exc:  # noqa: BLE001 - object watch is optional
                self._health.failed(exc, log, "processing object watch")

        if not raw_events and not registered_candidates:
            return []

        alerts = self.engine.evaluate(
            raw_events,
            scene_context=self.scene_context,
            active_zone_roles=self.active_zone_roles,
            monitoring_scope=self.monitoring_scope,
            scene_reviewed=self.scene_reviewed,
        )
        if self.object_watch:
            from cvti.serving.event_adapters import (
                current_object_watch_rule, object_watch_rule_signature,
            )
            for alert in alerts:
                if alert.detector != "object_watch":
                    continue
                rule = current_object_watch_rule(self, alert, disk=False)
                if rule is not None:
                    alert.metadata["object_watch_rule_signature"] = \
                        object_watch_rule_signature(rule)
        self.context_decisions = list(self.engine.context_decisions)
        self.context_suppression_count += sum(
            decision.get("decision") == "context_incompatible"
            for decision in self.context_decisions
        )
        alerts.extend(registered_candidates)
        if not alerts:
            return []
        from cvti.verification.frame_select import select_evidence_frames

        recent = list(self._frame_buffer)
        # Snapshot the continuous replay window once (shared by all alerts this frame).
        clip_snap = list(self._clip_buffer)
        clip_frames = [j for _, j in clip_snap]
        clip_fps = 0.0
        if len(clip_snap) >= 2:
            span = clip_snap[-1][0] - clip_snap[0][0]
            if span > 0:
                clip_fps = (len(clip_snap) - 1) / span
        out = []
        for a in alerts:
            # A verdict may have completed immediately before a live disable.
            # Never let that already-drained object event enter the queue.
            if a.detector == "object_watch" and not self.object_watch:
                continue
            token = None
            watch_result = None
            # Preserve zones for spatial rules so separate positions deduplicate independently.
            zone = (
                a.metadata.get("zone")
                if a.detector in {"object_watch", "object_state"}
                else zone_by_pid.get(a.person_id) if a.detector == "presence" else None
            )
            frames, _ = select_evidence_frames(recent, a.rule_name)
            if a.detector == "concealment":
                # Preserve a time-based lead-up, regardless of sampled FPS.
                import cv2
                import numpy as np
                history = [(ts, jpeg) for ts, jpeg in clip_snap
                           if timestamp - 4.0 <= ts <= timestamp]
                if history:
                    indices = sorted({0, len(history) // 2, len(history) - 1})
                    decoded = [cv2.imdecode(np.frombuffer(history[i][1], dtype=np.uint8),
                                            cv2.IMREAD_COLOR) for i in indices]
                    frames = [f for f in decoded if f is not None] or frames
            # Whole-frame detectors (video-action, fire) carry no person_id, so an
            # alert would arrive with nothing to point at. If exactly one person is
            # tracked, that's who it's about; with several, box the most prominent
            # (largest) one. An empty scene stays unboxed — correct for fire.
            # Pose-based detectors (concealment) number people in their own ID
            # space, so a person_id may not exist in the ByteTrack map either —
            # fall back for any unresolved id, not just a missing one.
            boxes = getattr(self, "_box_by_track", {}) or {}
            bbox = (a.metadata.get("group_bbox")
                    if a.detector == "multiple_people_moving" else None)
            # Pose IDs and ByteTrack IDs are independent, even when numbers match.
            if a.detector == "concealment":
                bbox = a.metadata.get("subject_bbox")
            if bbox is None and a.detector in {"object_watch", "object_state"}:
                bbox = a.metadata.get("bbox")
            if bbox is None and a.detector not in {"object_watch", "object_state", "concealment"}:
                bbox = boxes.get(a.person_id)
            if bbox is None and boxes and a.detector not in {"object_watch", "object_state", "concealment"}:
                bbox = max(boxes.values(),
                           key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            # Evidence upgrade: full frames give the gate context; a zoomed
            # crop of the flagged subject gives it hands and held objects —
            # its own rejections say "no people visible" on full CCTV frames.
            from cvti.verification.frame_select import append_subject_crop
            if a.detector == "object_watch":
                watch_result = object_watch_results.get((
                    str(a.metadata.get("object_id")), a.metadata.get("track_id"),
                    int(a.metadata.get("sample_sequence", -1)),
                ))
                rule_key = a.metadata.get("object_watch_rule_signature") or a.rule_name
                token = (runtime.reserve(watch_result, rule_key)  # type: ignore[union-attr]
                         if watch_result is not None else None)
                evidence = list(getattr(watch_result, "evidence", ()) or ())
                if token is None or not evidence:
                    continue
            elif a.detector == "object_state":
                panel = object_state_evidence.get((a.metadata.get("zone"), a.metadata.get("state")))
                if panel is None:
                    continue
                evidence = [panel]
            elif a.detector == "multiple_people_moving":
                evidence = frames or [image]
            else:
                evidence = append_subject_crop(frames or [image], image, bbox)
            out.append(_to_queued(self.camera_id, a, timestamp, zone,
                                  evidence, self.scene_context,
                                  clip_frames=clip_frames, clip_fps=clip_fps,
                                  bbox=bbox,
                                  object_watch_token=(token
                                                      if a.detector == "object_watch" else None),
                                  object_watch_result=(watch_result
                                                       if a.detector == "object_watch" else None)))
        return out


# A camera added through the UI carries only {id, source, area_id}: the rules
# config is written later, by apply_template on the wizard's "Use case" step.
# The engine used to index that key directly, so pressing Start before choosing
# a use case killed it instantly with KeyError: 'config' -- and the operator saw
# a camera that tested fine, no video, and no error anywhere (20 Sep field
# report). Seeing your camera must not depend on having decided what to detect
# on it, so an unconfigured camera falls back to the general live-camera rule
# set; the template still overwrites this the moment it runs.
DEFAULT_RULES_CONFIG = "configs/all_threats_v1.json"


def rules_config_for(cam: dict) -> str:
    """The camera's rules config, or the default when none has been chosen."""
    return str(cam.get("config") or DEFAULT_RULES_CONFIG)


def build_camera_states(site_config: dict, *, pose_model: Any = None, weapon_model: Any = None,
                        video_action_model: Any = None,
                        va_runner: Any = None,
                        heavy_runner: Any = None,
                        baseline_config: str | None = None,
                        scene_contexts: dict[str, dict] | None = None,
                        monitoring_scopes: dict[str, str] | None = None,
                        reviewed_camera_ids: set | None = None,
                        imgsz: int = 640,
                        output_dir: str | Path | None = None) -> dict[str, dict]:
    """Parse a site config into {camera_id: {"source": ..., "state": PerCameraState}}.

    Site config per-camera keys: id, source, config, plus optional zones,
    scene_description, and the signal toggles concealment / violence / weapons /
    theft (all default false). `pose_model` / `weapon_model` are shared instances;
    cameras that enable a pose/weapon signal reuse them. `baseline_config` (if
    given) is merged into every camera's engine (always-on critical rules).
    """
    from cvti.retail.zones import RetailZoneMonitor, load_zone_config

    out: dict[str, dict] = {}
    for cam in site_config["cameras"]:
        cam_id = cam["id"]
        from cvti.detector.object_state import ObjectZoneMonitor
        object_state_zones = cam.get("object_state_zones", [])
        ObjectZoneMonitor(object_state_zones)  # validate policies at startup
        detector_flags = (
            "concealment", "violence", "weapons", "theft", "tamper", "fall",
            "fire_smoke", "running", "crowd_formation", "normal_movement",
            "multiple_people_moving", "video_action", "object_watch",
            "object_watch_enabled",
            "general_object_tracking", "general_object_tracking_overlays",
        )
        for flag in detector_flags:
            if flag in cam and not isinstance(cam[flag], bool):
                raise ValueError(f"camera {cam_id}: {flag} must be a boolean")
        if cam.get("view_only"):
            # A view-only camera is glass, not a detector (4 Sep, pilot): it
            # streams to the wall and runs nothing — no state, no rules, no
            # baseline. The pipeline still decodes and publishes it.
            continue
        float_movement_fields = (
            "movement_enter_speed_ratio",
            "movement_exit_speed_ratio",
            "movement_min_track_seconds",
            "movement_persistence_seconds",
        )
        for field_name in float_movement_fields:
            if isinstance(cam.get(field_name), bool):
                raise ValueError(
                    f"camera {cam_id}: {field_name} must be a number, not boolean"
                )
        try:
            movement_enter = float(cam.get("movement_enter_speed_ratio", 0.05))
            movement_exit = float(cam.get("movement_exit_speed_ratio", 0.02))
            movement_min_track = float(cam.get("movement_min_track_seconds", 0.4))
            movement_persistence = float(cam.get("movement_persistence_seconds", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera {cam_id}: invalid movement configuration: {exc}") from exc
        movement_min_people_raw = cam.get("movement_min_people", 2)
        if isinstance(movement_min_people_raw, bool):
            raise ValueError(f"camera {cam_id}: movement_min_people must be an integer")
        try:
            movement_min_people_number = float(movement_min_people_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera {cam_id}: invalid movement configuration: {exc}") from exc
        if not isfinite(movement_min_people_number) or not movement_min_people_number.is_integer():
            raise ValueError(f"camera {cam_id}: movement_min_people must be an integer")
        movement_min_people = int(movement_min_people_number)
        if not all(isfinite(value) for value in (
            movement_enter, movement_exit, movement_min_track, movement_persistence,
        )):
            raise ValueError(f"camera {cam_id}: movement thresholds must be finite")
        if movement_enter <= 0:
            raise ValueError(f"camera {cam_id}: movement_enter_speed_ratio must be positive")
        if movement_exit <= 0 or movement_exit >= movement_enter:
            raise ValueError(
                f"camera {cam_id}: movement_exit_speed_ratio must be positive and lower "
                "than movement_enter_speed_ratio"
            )
        if movement_min_track <= 0:
            raise ValueError(f"camera {cam_id}: movement_min_track_seconds must be positive")
        if movement_min_people < 2:
            raise ValueError(f"camera {cam_id}: movement_min_people must be at least 2")
        if movement_persistence <= 0:
            raise ValueError(f"camera {cam_id}: movement_persistence_seconds must be positive")
        for field_name in ("object_watch_sample_fps", "object_watch_min_similarity"):
            if isinstance(cam.get(field_name), bool):
                raise ValueError(f"camera {cam_id}: {field_name} must be a number, not boolean")
        max_candidates_raw = cam.get("object_watch_max_candidates_per_frame", 24)
        if isinstance(max_candidates_raw, bool):
            raise ValueError(
                f"camera {cam_id}: object_watch_max_candidates_per_frame must be an integer"
            )
        try:
            object_watch_sample_fps = float(cam.get("object_watch_sample_fps", 1.0))
            object_watch_min_similarity = (
                None if cam.get("object_watch_min_similarity") is None
                else float(cam.get("object_watch_min_similarity"))
            )
            max_candidates_number = float(max_candidates_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera {cam_id}: invalid object watch configuration: {exc}") from exc
        if not isfinite(object_watch_sample_fps) or object_watch_sample_fps <= 0:
            raise ValueError(f"camera {cam_id}: object_watch_sample_fps must be positive")
        if not isfinite(max_candidates_number) or not max_candidates_number.is_integer() \
                or max_candidates_number < 1:
            raise ValueError(
                f"camera {cam_id}: object_watch_max_candidates_per_frame must be a positive integer"
            )
        if object_watch_min_similarity is not None and (
            not isfinite(object_watch_min_similarity) or not 0.0 <= object_watch_min_similarity <= 1.0
        ):
            raise ValueError(f"camera {cam_id}: object_watch_min_similarity must be between 0 and 1")
        permitted_raw = cam.get("permitted_movement_zones")
        permitted_zones: tuple[str, ...] | None = None
        if permitted_raw is not None:
            if not isinstance(permitted_raw, (list, tuple)):
                raise ValueError(
                    f"camera {cam_id}: permitted_movement_zones must be a list or tuple"
                )
            if any(not isinstance(zone, str) or not zone.strip() for zone in permitted_raw):
                raise ValueError(
                    f"camera {cam_id}: permitted_movement_zones must contain non-empty strings"
                )
            if not cam.get("zones"):
                raise ValueError(
                    f"camera {cam_id}: permitted_movement_zones requires a zone config"
                )
            permitted_zones = tuple(permitted_raw)
        engine = CustomizationEngine(rules_config_for(cam), baseline_path=baseline_config)
        zone_monitor = None
        if cam.get("zones"):
            zone_specs = load_zone_config(cam["zones"])
            if permitted_zones is not None:
                configured_names = {spec.name for spec in zone_specs}
                unknown_names = sorted(set(permitted_zones) - configured_names)
                if unknown_names:
                    raise ValueError(
                        f"camera {cam_id}: unknown permitted movement zones: {unknown_names}"
                    )
            zone_monitor = RetailZoneMonitor(zone_specs)
        vehicle_zone_monitor = None
        if object_state_zones:
            names = {str(z.name) for z in zone_monitor.zones} if zone_monitor is not None else set()
            if any(policy["zone"] not in names for policy in object_state_zones):
                raise ValueError(f"camera {cam_id}: object-state policy refers to a missing zone")
        if cam.get("vehicle_zones"):
            vehicle_zone_monitor = RetailZoneMonitor(load_zone_config(cam["vehicle_zones"]))
        vehicle_line = cam.get("vehicle_line")   # directional entry/exit tripwire
        scene = (scene_contexts or {}).get(cam_id)
        if scene is None and cam.get("scene_description"):
            scene = {"environment_type": cam.get("environment_type", "unknown"),
                     "scene_description": cam["scene_description"]}
        active_zone_roles = set(cam.get("accepted_zone_roles") or [])
        if cam.get("zones"):
            try:
                zone_data = json.loads(Path(cam["zones"]).read_text())
                kind_roles = {"shelf": "merchandise", "checkout": "checkout"}
                for zone in zone_data.get("zones", []):
                    role = zone.get("context_role") or kind_roles.get(zone.get("kind"))
                    if role:
                        active_zone_roles.add(str(role))
            except (OSError, ValueError, TypeError):
                log.warning("unable to read accepted zone roles for %s", cam_id)
        object_watch_enabled = bool(cam.get("object_watch_enabled", cam.get("object_watch", False)))
        configured_library = cam.get("object_watch_library")
        canonical_library = str((Path(output_dir) / "object_library").resolve()) if output_dir else None
        if object_watch_enabled and configured_library and canonical_library \
                and Path(configured_library).expanduser().resolve() != Path(canonical_library):
            raise ValueError(
                f"camera {cam_id}: object_watch_library conflicts with canonical site library"
            )
        object_watch_library = canonical_library or configured_library
        out[cam_id] = {
            "source": cam["source"],
            "state": PerCameraState(
                cam_id, engine, zone_monitor=zone_monitor,
                object_state_zones=object_state_zones,
                registered_objects=cam.get("registered_objects", []),
                registered_source=str(cam["source"]),
                vehicle_zone_monitor=vehicle_zone_monitor, vehicle_line=vehicle_line,
                scene_context=scene,
                monitoring_scope=(monitoring_scopes or {}).get(cam_id, "full"),
                scene_reviewed=cam_id in (reviewed_camera_ids or set()),
                active_zone_roles=active_zone_roles,
                pose_model=pose_model, weapon_model=weapon_model,
                video_action_model=video_action_model,
                va_runner=va_runner,
                heavy_runner=heavy_runner,
                # The engine's one inference size: pose/weapon run at the
                # same imgsz as the shared detector, not a leaked default.
                imgsz=imgsz,
                heavy_stride=int(cam.get("heavy_stride", 2)),
                concealment=cam.get("concealment", False),
                violence=cam.get("violence", False),
                weapons=cam.get("weapons", False), theft=cam.get("theft", False),
                tamper=cam.get("tamper", False), fall=cam.get("fall", False),
                fire_smoke=cam.get("fire_smoke", False),
                running=cam.get("running", False),
                crowd_formation=cam.get("crowd_formation", False),
                normal_movement=cam.get("normal_movement", False),
                multiple_people_moving=cam.get("multiple_people_moving", False),
                running_min_speed_ratio=float(cam.get("running_min_speed_ratio", 0.18)),
                running_min_frames=int(cam.get("running_min_frames", 3)),
                crowd_min_people=int(cam.get("crowd_min_people", 4)),
                crowd_min_frames=int(cam.get("crowd_min_frames", 3)),
                crowd_max_cluster_ratio=float(cam.get("crowd_max_cluster_ratio", 0.24)),
                fire_min_frames=int(cam.get("fire_min_frames", 3)),
                fire_min_hot_area_ratio=float(cam.get("fire_min_hot_area_ratio", 0.012)),
                fire_rearm_seconds=float(cam.get("fire_rearm_seconds", 300.0)),
                movement_enter_speed_ratio=movement_enter,
                movement_exit_speed_ratio=movement_exit,
                movement_min_track_seconds=movement_min_track,
                movement_min_people=movement_min_people,
                movement_persistence_seconds=movement_persistence,
                permitted_movement_zones=permitted_zones,
                object_watch=object_watch_enabled,
                general_object_tracking=cam.get("general_object_tracking", False),
                general_object_tracking_overlays=cam.get(
                    "general_object_tracking_overlays", False
                ),
                object_watch_library=object_watch_library,
                object_watch_sample_fps=object_watch_sample_fps,
                object_watch_max_candidates_per_frame=int(max_candidates_number),
                object_watch_min_similarity=object_watch_min_similarity,
                object_watch_open_vocab_provider=str(
                    cam.get("object_watch_open_vocab_provider", "disabled")
                ),
                video_action=cam.get("video_action", False),
                zone_min_person_area_ratio=cam.get("zone_min_person_area_ratio"),
            ),
        }
        out[cam_id]["state"]._object_watch_rule_path = rules_config_for(cam)
        _apply_object_watch_settings(out[cam_id]["state"], cam)
    return out


def refresh_camera_rules(state: "PerCameraState", cam: dict,
                         baseline_config: str | None = None) -> None:
    """Hot-swap a camera's rules engine and zone monitor from its config files.

    Rules and zones are JSON — reloading them is cheap and touches no model,
    so a plain-English rule typed in the app takes effect on the RUNNING
    engine within seconds instead of waiting for a restart nobody was told to
    do. Attribute swaps are atomic in CPython: a frame in flight sees either
    the old engine or the new one, never a half-built one.
    """
    from cvti.retail.zones import RetailZoneMonitor, load_zone_config
    from cvti.detector.object_state import ObjectZoneMonitor
    policies = cam.get("object_state_zones", [])
    monitor = ObjectZoneMonitor(policies)
    state.engine = CustomizationEngine(rules_config_for(cam), baseline_path=baseline_config)
    state.object_state_zones = policies
    if state.registered_objects != cam.get("registered_objects", []) or state.registered_source != str(cam["source"]):
        from cvti.serving.registered_objects import RegisteredObjects
        previous = state._registered_monitor
        state.registered_objects = cam.get("registered_objects", [])
        state.registered_source = str(cam["source"])
        updated = RegisteredObjects(state.camera_id, state.registered_source, state.registered_objects)
        if previous is not None and previous.source == updated.source:
            for entry in updated.entries:
                if entry in previous.entries:
                    key = entry["id"]
                    if key in previous.monitors:
                        updated.monitors[key] = previous.monitors[key]
                    if key in previous.samples:
                        updated.samples[key] = previous.samples[key]
                    if key in previous.last_sample:
                        updated.last_sample[key] = previous.last_sample[key]
        state._registered_monitor = updated
    state._object_state_monitor = monitor
    state._object_watch_rule_path = rules_config_for(cam)
    if cam.get("zones"):
        state.zone_monitor = RetailZoneMonitor(load_zone_config(cam["zones"]))
    else:
        state.zone_monitor = None
    if cam.get("vehicle_zones"):
        state.vehicle_zone_monitor = RetailZoneMonitor(load_zone_config(cam["vehicle_zones"]))
    if cam.get("vehicle_line") is not None:
        state.vehicle_line = cam.get("vehicle_line")
        state._vehicle_line_zone = None   # rebuilt lazily against the frame size
    if (cam.get("vehicle_zones") or cam.get("vehicle_line") is not None) and state._vehicle_tracker is None:
        import supervision as sv
        state._vehicle_tracker = sv.ByteTrack()
    _apply_object_watch_settings(state, cam)


def _apply_object_watch_settings(state: "PerCameraState", cam: dict) -> bool:
    """Apply canonical runtime defaults plus only explicit camera overrides."""
    old = (
        state.object_watch, state.object_watch_sample_fps,
        state.object_watch_max_candidates_per_frame,
        state.object_watch_min_similarity, state.object_watch_open_vocab_provider,
    )
    canonical = None
    library = getattr(state, "object_watch_library", None)
    if library:
        try:
            from cvti.object_watch.runtime_config import resolve_config
            canonical = resolve_config(Path(library).resolve().parent)
        except (OSError, TypeError, ValueError):
            canonical = None
    state.object_watch = bool(cam.get("object_watch_enabled", cam.get("object_watch", False)))
    state.object_watch_sample_fps = float(
        cam["object_watch_sample_fps"] if "object_watch_sample_fps" in cam
        else getattr(canonical, "sample_fps", 1.0)
    )
    state.object_watch_max_candidates_per_frame = int(
        cam["object_watch_max_candidates_per_frame"]
        if "object_watch_max_candidates_per_frame" in cam
        else getattr(canonical, "max_candidates", 24)
    )
    state.object_watch_min_similarity = (
        None if cam.get("object_watch_min_similarity") is None
        else float(cam["object_watch_min_similarity"])
    )
    provider = (cam["object_watch_open_vocab_provider"]
                if "object_watch_open_vocab_provider" in cam
                else getattr(canonical, "proposal_provider", "none"))
    state.object_watch_open_vocab_provider = str(provider)
    return old != (
        state.object_watch, state.object_watch_sample_fps,
        state.object_watch_max_candidates_per_frame,
        state.object_watch_min_similarity, state.object_watch_open_vocab_provider,
    )

def load_site_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def detection_status(state: Any, now: float | None = None) -> dict:
    """Counts only: frames the model looked at, how many had a person, how
    long ago one was last seen. None where the camera runs no detection."""
    now = time.time() if now is None else now
    analysed = int(getattr(state, "frames_analysed", 0) or 0)
    with_people = int(getattr(state, "frames_with_people", 0) or 0)
    last = float(getattr(state, "people_last_seen_at", 0.0) or 0.0)
    return {
        "frames_analysed": analysed,
        "frames_with_people": with_people,
        "people_in_last_frame": int(getattr(state, "people_in_last_frame", 0) or 0),
        "people_last_seen_s": (round(max(0.0, now - last), 1) if last else None),
    }
