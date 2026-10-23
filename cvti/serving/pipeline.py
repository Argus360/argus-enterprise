"""Multi-stream pipeline orchestrator (plan.md Phase 8.1).

Ties the pieces together in ONE process with a SHARED detector model:

    decoders (latest-frame-drop) -> collect_batch -> ONE batched YOLO pass
      -> scatter per camera -> per-camera handler (tracker/rules seam)

The batched detector is stateless, so frames from all cameras go through one
forward pass. Stateful work (ByteTrack association, zones, rules, the alert
queue) is per camera and lives in the handler — that is the seam where the
existing cvti detector/rules logic gets wired in next (8.1 continued / 8.3).

Run it directly to prove cross-camera batching:

    python -m cvti.serving.pipeline --sources clipA.mp4 clipB.mp4 --seconds 15
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cvti.contracts import LOCAL_VLM_MODEL
from cvti.serving.alert_queue import QueuedAlert
from cvti.serving.batcher import collect_batch
from cvti.serving.detect_budget import (AdaptiveConfig, DetectCamera,
                                        DetectionRateController)
from cvti.serving.frame_publisher import FrameOverlay
from cvti.serving.streams import Frame, StreamDecoder
from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Handler signature: (frame, ultralytics_result) -> None
ResultHandler = Callable[[Frame, Any], None]


def _frame_overlays(state: Any, timestamp: float | None = None, *,
                    exact_frame: bool = False) -> list[FrameOverlay]:
    if state is None:
        return []
    records = list(getattr(state, "_motion_overlays", ()) or ())
    observed_at = getattr(state, "_motion_overlay_observed_at", None)
    now = time.monotonic() if timestamp is None else timestamp
    # Smooth publication must not paint seconds-old coordinates onto live video.
    # Exact-frame publication deliberately displays the inference frame itself.
    # "Seconds-old" is relative to how often this camera is detected on: a
    # fixed one-second limit on a box that is refreshed every 4.7 s (CHI,
    # 13 Oct 2026) meant boxes expired before they were ever drawn, and
    # "person detection is not working" was the wall hiding working
    # detections. The limit is twice the camera's measured detection interval,
    # never under a second, never over ten.
    if not exact_frame and observed_at is not None and now - observed_at > _overlay_max_age(state):
        records = []
    object_overlay_provider = getattr(state, "general_object_overlays", None)
    if object_overlay_provider is not None:
        try:
            records.extend(
                object_overlay_provider(time.monotonic() if timestamp is None else timestamp)
            )
        except Exception:  # noqa: BLE001 - optional boxes cannot block person rendering
            log.warning("general object overlay snapshot failed", exc_info=True)
    return [
        FrameOverlay(
            track_id=int(item["track_id"]),
            bbox=tuple(item["bbox"]),
            label=str(item["label"]),
            colour=tuple(item.get("colour", (0, 200, 0))),
            namespace=str(item.get("namespace", "person")),
        )
        for item in records
    ]


OVERLAY_MIN_AGE_S = 1.0
OVERLAY_MAX_AGE_S = 10.0


def _overlay_max_age(state: Any) -> float:
    """How long the last detection's boxes stay on the wall for this camera."""
    interval = float(getattr(state, "_detection_interval_s", 0.0) or 0.0)
    return max(OVERLAY_MIN_AGE_S, min(OVERLAY_MAX_AGE_S, 2.0 * interval))


def _object_track_snapshot(state: Any, timestamp: float | None = None) -> dict | None:
    provider = getattr(state, "general_object_snapshot", None) if state is not None else None
    if provider is None:
        return None
    try:
        return provider(time.monotonic() if timestamp is None else timestamp)
    except Exception:  # noqa: BLE001 - raw/person publication must continue
        log.warning("general object metadata snapshot failed", exc_info=True)
        return None


def _publish_frame(publisher: Any, camera_id: str, frame: Any, state: Any, *,
                   exact_frame: bool = False) -> None:
    generation_provider = getattr(publisher, "object_generation", None)
    object_generation = generation_provider(camera_id) if generation_provider else None
    overlays = _frame_overlays(state, exact_frame=exact_frame)
    snapshot = _object_track_snapshot(state)
    if generation_provider is None:
        publisher.publish(camera_id, frame, overlays)
    else:
        publisher.publish(
            camera_id, frame, overlays, object_tracks=snapshot,
            object_generation=object_generation,
        )


def _publish_jpeg(publisher: Any, camera_id: str, jpeg: bytes, state: Any,
                  source_size: tuple[int, int] | None) -> None:
    generation_provider = getattr(publisher, "object_generation", None)
    object_generation = generation_provider(camera_id) if generation_provider else None
    overlays = _frame_overlays(state)
    snapshot = _object_track_snapshot(state)
    if generation_provider is None:
        publisher.publish_jpeg(camera_id, jpeg, overlays, source_size)
    else:
        publisher.publish_jpeg(
            camera_id, jpeg, overlays, source_size, object_tracks=snapshot,
            object_generation=object_generation,
        )


def _alert_track_ids(alerts: Any) -> set[int]:
    track_ids: set[int] = set()
    for alert in alerts:
        scalar = getattr(alert, "track_id", getattr(alert, "person_id", None))
        if scalar is not None:
            track_ids.add(int(scalar))
        payload = getattr(alert, "payload", {}) or {}
        candidate = payload.get("candidate") if isinstance(payload, dict) else None
        metadata = getattr(candidate or alert, "metadata", {}) or {}
        for track_id in metadata.get("track_ids", ()):
            track_ids.add(int(track_id))
    return track_ids


def _gate_workers_for(requested: int, n_cameras: int, *, provider: str = "",
                      device: str = "", slots: int | None = None) -> int:
    """How many VLM workers to run. 0 = derive; an explicit request always wins.

    A verdict takes seconds, so on a multi-camera site alerts queue behind a
    single worker: measured on a 5-camera run, median detection->verified
    latency was 46.5s with one worker and 28.0s with two, for no extra memory.

    For LOCAL providers the ceiling is the server's real parallelism (audit
    1 Sep, V2): the old camera-derived cap of 3 put three workers on a server
    running OLLAMA_NUM_PARALLEL=2 — the third worker queued server-side and
    read as gate latency. Workers now never exceed the configured slots, and
    a CPU-only box runs exactly one: concurrent CPU inference makes every
    verdict slower than a queue would. Cloud gates keep the camera-derived
    sizing — their parallelism is not ours to budget.
    """
    if requested > 0:
        return requested
    derived = max(1, min(3, n_cameras // 2))
    if provider not in ("ollama", "local"):
        return derived
    if (device or _auto_device()) == "cpu":
        return 1
    if slots is None:
        from cvti.verification.ollama import configured_parallel_slots
        slots = configured_parallel_slots()
    return max(1, min(derived, slots))


# Decoded frames per second the wall may ask of a CPU-only box, across all
# cameras. Watched tiles decode at the wall rate; with twelve tiles open at
# 24 fps the pilot server decoded 288 frames a second of 5-megapixel HEVC
# for display alone and sat at 100% CPU (bundle, 13 Oct 2026 22:37).
CPU_WALL_DECODE_BUDGET_FPS = 96
# The static per-camera ceiling keeps a floor so a tile is never asked for
# less than this by the CEILING; the per-tick budget in wall_budget.py may
# take a watched camera lower when its measured decode cost demands it.
WALL_FPS_FLOOR = 8.0


def wall_fps_for(site: dict, requested: float, n_cameras: int, device: str) -> float:
    """The wall's publish rate for this site. A `wall_fps` key in the site
    file always wins. Otherwise a CPU-only box shares CPU_WALL_DECODE_BUDGET_FPS
    across its cameras, never below WALL_FPS_FLOOR; a GPU box keeps the
    requested rate."""
    explicit = site.get("wall_fps") if isinstance(site, dict) else None
    if isinstance(explicit, (int, float)) and explicit > 0:
        return float(explicit)
    if device != "cpu" or n_cameras <= 0:
        return requested
    capped = max(WALL_FPS_FLOOR, float(CPU_WALL_DECODE_BUDGET_FPS // n_cameras))
    if capped < requested:
        log.info("[serving] CPU-only box with %d cameras: wall rate %.0f fps instead of %.0f "
                 "(set wall_fps in the site file to override)", n_cameras, capped, requested)
        return capped
    return requested


def compute_device(device: str = "") -> str:
    """Where detection actually runs: "cpu" only when torch would run on the
    CPU AND onnxruntime has no better provider. The installed Windows build
    ships a CPU torch and does its detection on the GPU through DirectML, so
    the torch device alone called an RTX box "CPU-only" and would have put it
    on keyframe decode and the 8 fps wall (14 Oct 2026)."""
    torch_device = device or _auto_device()
    if torch_device != "cpu":
        return torch_device
    try:
        from cvti.detector.accel import best_available_provider
        provider = best_available_provider()
    except Exception:  # noqa: BLE001 - a probe must never change the answer to worse than cpu
        log.debug("provider probe failed; treating the box as CPU-only", exc_info=True)
        provider = None
    return "cpu" if provider is None else provider.replace("ExecutionProvider", "").lower()


def _auto_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def scanner_endpoint_for(gate_provider: str, gate_base_url: str) -> tuple[str, str] | None:
    """Where the English-rules scanner and the watch runner send their frames,
    as (base_url, api_key_env), or None when the gate's provider has no
    OpenAI-compatible endpoint for them (mock, anthropic).

    They used to start only for the local runtime, so choosing a cloud
    verifier silently switched off every English rule and every watch."""
    if gate_provider in ("ollama", "local"):
        return (gate_base_url or "http://localhost:11434/v1", "OLLAMA_API_KEY")
    if gate_provider == "openrouter":
        return ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    if gate_provider == "openai_compatible" and gate_base_url:
        return (gate_base_url, "OPENAI_API_KEY")
    return None


def resolve_mapper_settings(
    *,
    gate_provider: str,
    gate_model: str,
    gate_base_url: str,
    mapper_provider: str,
    mapper_model: str,
    mapper_base_url: str,
) -> tuple[str, str, str]:
    supported = {"mock", "ollama", "local", "anthropic", "openai_compatible"}
    provider = mapper_provider.strip() if mapper_provider else ""
    if not provider and gate_provider == "openrouter":
        # The mapper has no OpenRouter row, but OpenRouter speaks the OpenAI
        # chat API: route it there on the gate's model and key (the app sets
        # OPENAI_API_KEY alongside OPENROUTER_API_KEY for this reason).
        # Without this, a cloud gate silently sent scene mapping back to a
        # local Ollama that a CPU-only server cannot run.
        return ("openai_compatible", mapper_model or gate_model,
                mapper_base_url or "https://openrouter.ai/api/v1")
    if not provider:
        provider = gate_provider if gate_provider in supported else "ollama"
    model = mapper_model or (gate_model if provider == gate_provider else "")
    base_url = mapper_base_url or (
        gate_base_url if provider == gate_provider else ""
    )
    if provider in {"ollama", "local"} and not base_url:
        base_url = "http://localhost:11434/v1"
    return provider, model, base_url


def build_mapper_service(
    *,
    output_dir: str,
    gate_provider: str,
    gate_model: str,
    gate_base_url: str,
    mapper_provider: str = "",
    mapper_model: str = "",
    mapper_base_url: str = "",
):
    from cvti.scene.agent_mapper import AgentMapper
    from cvti.serving.scene_map import FullAgentMapperService

    provider, model, base_url = resolve_mapper_settings(
        gate_provider=gate_provider,
        gate_model=gate_model,
        gate_base_url=gate_base_url,
        mapper_provider=mapper_provider,
        mapper_model=mapper_model,
        mapper_base_url=mapper_base_url,
    )
    mapper = AgentMapper(provider=provider, model=model, base_url=base_url)
    return FullAgentMapperService(output_dir, mapper)


def prepare_scene_mapping(
    site: dict,
    *,
    output_dir: str,
    gate_provider: str,
    gate_model: str,
    gate_base_url: str,
    mapper_provider: str = "",
    mapper_model: str = "",
    mapper_base_url: str = "",
):
    service = build_mapper_service(
        output_dir=output_dir,
        gate_provider=gate_provider,
        gate_model=gate_model,
        gate_base_url=gate_base_url,
        mapper_provider=mapper_provider,
        mapper_model=mapper_model,
        mapper_base_url=mapper_base_url,
    )
    return service.prepare(
        list(site.get("cameras") or []),
        site.get("scene_context_policy", "auto"),
    )


def active_cameras_after_preflight(cameras: list[dict], preflight) -> list[dict]:
    _ = preflight
    # "Active" means running detection: a view-only camera is glass, not a
    # detector (4 Sep) — it never needs scene context, rules, or models, so
    # it neither blocks startup nor enters the detection roster.
    return [dict(camera) for camera in cameras if not camera.get("view_only")]


def monitoring_scopes_from_preflight(preflight, policy: str = "auto") -> dict[str, str]:
    """Which rule set each camera runs, per the site's chosen philosophy.

    Coverage-first is the default and the decision is deliberate (1 Sep,
    after a pilot spent two days dark behind a review gate): under 'auto',
    EVERY camera runs the full rule set immediately — an unreviewed context
    informs the gate and the English rules but never turns detection off.
    What review unlocks under auto is context-driven SUPPRESSION (see
    CustomizationEngine.evaluate's scene_reviewed): the one power a wrong AI
    guess must not have is silencing a real alert. Sites that explicitly
    choose require_reviewed/manual get certainty-first: critical baseline
    rules until a human approves — running critical-only is still strictly
    more coverage than the hard block those policies used to mean."""
    from cvti.scene.context_store import normalize_scene_context_policy

    if normalize_scene_context_policy(policy) == "auto":
        return {str(camera_id): "full" for camera_id in preflight.statuses}
    return {
        str(camera_id): (
            "full" if status.get("status") == "ready_reviewed" else "critical_only"
        )
        for camera_id, status in preflight.statuses.items()
    }


def activate_reviewed_scene_contexts(states: dict, output_dir: str | Path) -> list[str]:
    """Hot-apply approvals written by the console's separate process."""
    from cvti.scene.context_store import SceneContextStore

    changed: list[str] = []
    context_root = Path(output_dir) / "context"
    for camera_id, state in states.items():
        if getattr(state, "monitoring_scope", "full") == "full":
            continue
        store = SceneContextStore(context_root, camera_id)
        if store.load_status().status != "ready_reviewed":
            continue
        context = store._load_context()
        if context is not None:
            state.activate_scene_context(context)
            changed.append(camera_id)
    return changed


def _mapping_health_rows(preflight) -> list[dict]:
    return [
        {"camera_id": camera_id, **status}
        for camera_id, status in sorted(preflight.statuses.items())
    ]


def write_starting_health(output_dir: str, site: dict, *,
                          gate_provider: str, gate_model: str,
                          phase: str = "starting — mapping camera scenes",
                          scene_mapping: list | None = None) -> None:
    """The engine's first words, written BEFORE anything slow.

    The scene preflight and cold model loads run before the first real
    heartbeat — minutes on a fresh Windows install (Defender scanning the
    bundle, torch loading, the mapper timing out against a model still
    downloading). For all of it the console said 'engine not running' under
    a green Monitoring dot (field screenshot, 1 Sep). This heartbeat makes
    the warm-up visible: every camera reports 'starting', and engine.phase
    says what the engine is actually doing."""
    from pathlib import Path

    from cvti.serving.health_doc import build_health_doc

    cameras = [{"camera_id": str(c.get("id")), "state": "starting",
                "time_in_state": 0, "reconnects": 0}
               for c in site.get("cameras") or []]
    doc = build_health_doc(
        started_at=time.time(),
        cameras=cameras,
        gate={"provider": gate_provider, "model": gate_model, "reachable": None},
        disk={},
        memory={},
        components={"degraded": []},
        engine={"phase": phase, "frames_processed": 0, "alerts_queued": 0,
                "cameras": len(cameras)},
        scene_mapping=scene_mapping or [],
    )
    target = Path(output_dir) / "gate_health.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(doc))
    temporary.replace(target)


def _write_mapping_only_health(
    output_dir: str,
    preflight,
    *,
    gate_provider: str,
    gate_model: str,
) -> None:
    from pathlib import Path

    from cvti.serving.health_doc import build_health_doc

    rows = _mapping_health_rows(preflight)
    doc = build_health_doc(
        started_at=time.time(),
        cameras=[],
        gate={"provider": gate_provider, "model": gate_model, "reachable": None},
        disk={},
        memory={},
        components={"degraded": []},
        engine={"frames_processed": 0, "alerts_queued": 0, "cameras": 0},
        scene_mapping=rows,
    )
    target = Path(output_dir) / "gate_health.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(doc))
    temporary.replace(target)


class MultiStreamPipeline:
    def __init__(self, sources: dict[str, int | str], *, weights: str = "models/yolov8n.pt",
                 target_fps: float = 5.0, tick_seconds: float = 0.15, imgsz: int = 640,
                 conf: float = 0.4, device: str = "", half: bool = False,
                 max_batch: int = 32, on_result: ResultHandler | None = None,
                 camera_states: dict[str, Any] | None = None, alert_queue: Any = None,
                 publisher: Any = None, on_link_change=None,
                 publish_fps: float = 24.0,
                 view_only: set[str] | None = None,
                 fallback_sources: dict | None = None,
                 detector_backend: str = "torch",
                 object_watch_runtime: Any = None,
                 camera_inference: dict | None = None,
                 synchronized_box_cameras: set[str] | None = None,
                 camera_dewarp: dict | None = None,
                 keyframe_decode: bool | None = None,
                 keyframe_min_width: int = 2000,
                 detector_threads: int | None = None,
                 keyframe_cost_share: float = 0.25,
                 wall_decode_budget_ms: float | None = None,
                 keyframe_min_fps: float = 0.5,
                 adaptive_fps: "AdaptiveConfig | None" = None) -> None:
        # CPU share of one core, measured over fixed windows, above which an
        # eligible stream switches to keyframe-only decode under the "auto"
        # policy (see StreamDecoder).
        self.keyframe_cost_share = float(keyframe_cost_share)
        # Decode milliseconds per second the wall may spend, shared by watched
        # cameras after the detection baseline (cvti.serving.wall_budget).
        self.wall_decode_budget_ms = wall_decode_budget_ms
        self.keyframe_min_fps = float(keyframe_min_fps)
        # Threads for the CPU detector session (cvti.detector.accel). None =
        # detector_threads_for(): one core kept free on a CPU-only box.
        self.detector_threads = detector_threads
        self.synchronized_box_cameras = set(synchronized_box_cameras or ())
        # Keyframe-only decode for big live streams: None = on for CPU-only
        # boxes, off with a GPU. A site can force it either way.
        self.keyframe_decode = keyframe_decode
        self.keyframe_min_width = int(keyframe_min_width)
        # camera_id -> Dewarp (cvti.serving.dewarp): fisheye cameras are
        # flattened in the decoder, before any consumer sees a frame.
        self.camera_dewarp: dict = dict(camera_dewarp or {})
        from cvti.serving.camera_inference import CameraInference
        self._camera_inference = CameraInference(camera_inference)
        self.sources = sources
        # W1.3: camera_id -> the camera's OWN url, for cameras whose `source`
        # is the go2rtc restream. A dead gateway costs a reconnect, not
        # coverage: the decoder swaps to this after two failed reopens.
        self.fallback_sources = fallback_sources or {}
        # W2: torch | auto | onnx — how start() picks the detection backend.
        # torch by default ON PURPOSE: eval, tests, and the bare CLI construct
        # this class directly and must behave exactly as always; only run_site
        # (the shipped engine) passes the site file's choice, which defaults
        # to auto. A backend change is a site decision, never a side effect of
        # which test exported an .onnx first.
        self.detector_backend = detector_backend
        self.detector_info: dict = {}
        self.weights = weights
        self.target_fps = target_fps
        self.tick_seconds = tick_seconds
        self.imgsz = imgsz
        self.conf = conf
        self.device = device or _auto_device()
        # "cuda:0" is the device the service passes; an exact == "cuda" match
        # silently dropped --half, so FP16 never switched on (CHI T4, 23 Oct).
        self.half = half and str(self.device).startswith("cuda")
        self.max_batch = max_batch
        self._camera_states = camera_states
        self._object_watch_runtime = object_watch_runtime
        self._object_watch_lifecycle_lock = threading.Lock()
        self._started = False
        for state in (camera_states or {}).values():
            if getattr(state, "object_watch", False):
                attach = getattr(state, "attach_object_watch_runtime", None)
                if attach is not None:
                    attach(object_watch_runtime)
        self._alert_queue = alert_queue
        # Cameras that stream glass only — their frames never enter detection.
        self.view_only: set[str] = set(view_only or ())
        # Publishes frames (with boxes) so the UI never decodes the stream twice.
        self.publisher = publisher
        self.publish_fps = publish_fps
        self.smooth_publish = bool(publisher is not None and publish_fps > 0)
        self.latest_boxes: dict = {}       # camera_id -> last detection's boxes
        self.latest_zones: dict = {}       # camera_id -> {track_id: (zone, ...)} this frame
        self._smooth_thread = None
        self._last_detect: dict = {}       # camera_id -> last model-sample time
        # Adaptive detection rate (off unless the site enables it). When on, a
        # controller rewrites a per-camera target each window from GPU headroom
        # and per-camera activity; _due_for_detection reads _cam_target_fps and
        # falls back to the fixed self.target_fps for any camera not yet sized.
        self._adaptive_cfg = adaptive_fps or AdaptiveConfig(enabled=False)
        self._adaptive = DetectionRateController(
            self._adaptive_cfg, budget_fps=max(1.0, float(target_fps)) * 1.0,
        ) if self._adaptive_cfg.enabled else None
        self._cam_target_fps: dict[str, float] = {}
        self._activity_count: Counter = Counter()   # detections per cam this window
        self._adapt_last_wall = 0.0
        self._adapt_last_detect_ms = 0.0
        # General-object lifecycle uses monotonic observation time. Generation
        # and tracker update/reset share this lock so an inference started on an
        # old source can never re-enter continuity after a reconnect reset.
        self._source_lock = threading.RLock()
        self._source_generation = {camera_id: 0 for camera_id in sources}
        self._inference_context: dict[int, tuple[int, float]] = {}
        self.on_link_change = on_link_change
        # Called the moment an alert is accepted onto the queue — the two-tier
        # fast path hangs off this, ahead of any verification.
        self.on_queued = None
        # When per-camera states + a queue are supplied, route detections through
        # them (track -> zones -> rules -> alert queue). Otherwise just count.
        self._routes_to_queue = on_result is None \
            and camera_states is not None and alert_queue is not None
        if on_result is not None:
            self.on_result = on_result
        elif camera_states is not None and alert_queue is not None:
            self.on_result = self._route_to_queue
        else:
            self.on_result = self._default_handler
        self._decoders: dict[str, StreamDecoder] = {}
        self._model = None
        # stats
        self.batches = 0
        self.frames_processed = 0
        self.batch_hist: Counter = Counter()
        self.per_cam: Counter = Counter()
        self.alerts_queued = 0
        self._detect_ms_total = 0.0

    def start(self) -> None:
        self._started = True
        if self._object_watch_runtime is not None:
            self._object_watch_runtime.start()
        # W2: the detector loads through the accelerator selector — .onnx on
        # the provider this box offers when torch would be CPU-bound, torch
        # exactly as before everywhere else. detector_info says which and why;
        # health carries it.
        from cvti.detector import accel
        from cvti.detector.accel import load_detector
        accel.DETECTOR_THREADS = (self.detector_threads if self.detector_threads is not None
                                  else accel.detector_threads_for(None, compute_device(self.device)))
        self._model, self.detector_info = load_detector(
            self.weights, backend=self.detector_backend, device=self.device)
        self._camera_inference.load(self._model)
        for camera_id, profile in self._camera_inference.profiles.items():
            log.info("[detect] camera=%s calibrated rotation=%s confidence=%s model=%s",
                     camera_id, profile.get("rotation_degrees", 0),
                     profile.get("confidence", self.conf), profile.get("weights", self.weights))
        if self.detector_info["backend"] == "onnx":
            # The ONNX session owns acceleration; handing torch's device to
            # ultralytics here would re-trigger its own provider ideas (CoreML
            # on mps — ruled out) and half is a torch-CUDA concept.
            self.device, self.half = "cpu", False
            log.info("[detect] onnx backend: %s on %s",
                     self.detector_info["weights"].rsplit("/", 1)[-1],
                     self.detector_info["provider"])
        elif self.detector_info.get("reason"):
            log.info("[detect] torch backend (%s)", self.detector_info["reason"])
        # For the per-camera detector path we also need core.py's Detection list
        # (weapons/violence/theft), built from the same shared-model result.
        self._names = self._model.names
        for state in (self._camera_states or {}).values():
            try:
                state.ensure_general_object_tracker(self._names, self.target_fps)
            except Exception:  # noqa: BLE001 - optional tracking cannot block safety
                log.warning("general object tracker setup failed for %s",
                            getattr(state, "camera_id", "unknown"), exc_info=True)
        from cvti.detector.core import normalize_threat_classes
        self._threat_classes = normalize_threat_classes("gun,knife")
        # Decoders start at DETECTION rate. Until 3 Sep smooth-publish ran
        # every decoder at max(target, publish) fps around the clock — 12fps
        # of decode per camera for a wall nobody was watching. The boost is
        # now viewer-aware: the smooth loop raises display_fps on exactly the
        # cameras with an open /stream connection and drops it when the last
        # viewer leaves. Smoother where eyes are, ~3x cheaper where none are.
        # Keyframe-only decode is decided per camera by MEASURED decode cost
        # (streams.StreamDecoder.decode_cpu_ms), not by whether detection has
        # a GPU: a GPU box can still decode on the CPU, and a hardware decoder
        # makes a wide stream cheap (Tolu's audit, 15 Oct 2026). The site can
        # still force it off or on.
        # Three settings (Tolu's review): unset = measure and decide per
        # camera; true = force every eligible wide stream to keyframes; false
        # = never. Measured cost is a CPU share over fixed windows, so our own
        # sampling rate cannot change the verdict.
        policy = ("auto" if self.keyframe_decode is None
                  else "force" if self.keyframe_decode else "off")
        if policy != "off":
            log.info("[serving] keyframe-only decode policy=%s for live streams %d px wide or "
                     "more (auto switches when decoding measures %.0f%% of a core or more; "
                     "keyframe_decode / keyframe_min_width / keyframe_cost_share in the site file)",
                     policy, self.keyframe_min_width, self.keyframe_cost_share * 100)
        for cam_id, src in self.sources.items():
            vo = cam_id in self.view_only
            # A view-only camera decodes just enough to keep its link state
            # honest when nobody watches; the viewer boost carries it to
            # publish rate the moment the wall shows it. Detection cameras
            # decode at detection rate as before.
            self._decoders[cam_id] = StreamDecoder(
                cam_id, src, target_fps=(1.0 if vo else self.target_fps),
                on_state_change=self._handle_link_change, view_only=vo,
                fallback_source=self.fallback_sources.get(cam_id),
                dewarp=self.camera_dewarp.get(cam_id),
                idle_when_unwatched=vo,
                probe_source=self.fallback_sources.get(cam_id, src),
                keyframes_only_min_width=(self.keyframe_min_width if policy != "off" else 0),
                keyframe_policy=policy,
                keyframe_cost_share=self.keyframe_cost_share,
                keyframe_min_fps=self.keyframe_min_fps).start()
        if self.smooth_publish:
            import threading as _th
            self._smooth_thread = _th.Thread(target=self._smooth_publish_loop,
                                             name="smooth-publish", daemon=True)
            self._smooth_thread.start()
            log.info("[serving] live wall decoupled: watched cameras publish at "
                     "%.0f fps, detection samples at %.0f fps", self.publish_fps,
                     self.target_fps)
        log.info(f"[serving] {len(self._decoders)} camera(s) | device={self.device} "
              f"half={self.half} target_fps={self.target_fps} | model={self.weights}")

    def _refresh_camera_state(self, camera: dict,
                              baseline_config: str | None = None) -> None:
        """Hot-apply rules plus object-watch configuration and lifecycle."""
        from cvti.serving.camera import refresh_camera_rules

        camera_id = str(camera.get("id"))
        state = (self._camera_states or {}).get(camera_id)
        if state is None:
            return
        was_enabled = bool(getattr(state, "object_watch", False))
        old_settings = (
            getattr(state, "object_watch_sample_fps", None),
            getattr(state, "object_watch_max_candidates_per_frame", None),
            getattr(state, "object_watch_min_similarity", None),
            getattr(state, "object_watch_open_vocab_provider", None),
        )
        refresh_camera_rules(state, camera, baseline_config)
        enabled = bool(getattr(state, "object_watch", False))
        new_settings = (
            state.object_watch_sample_fps,
            state.object_watch_max_candidates_per_frame,
            state.object_watch_min_similarity,
            state.object_watch_open_vocab_provider,
        )
        with self._object_watch_lifecycle_lock:
            runtime = self._object_watch_runtime
            if enabled and runtime is None:
                library = getattr(state, "object_watch_library", None)
                if library is None:
                    state.object_watch = False
                    return
                try:
                    from cvti.object_watch.runtime import ObjectWatchRuntime
                    from cvti.object_watch.runtime_config import resolve_config
                    runtime = ObjectWatchRuntime(resolve_config(Path(library).resolve().parent))
                    self._object_watch_runtime = runtime
                    if self._started:
                        runtime.start()
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    log.warning("[object-watch] hot enable unavailable for %s: %s",
                                camera_id, exc)
                    state.object_watch = False
                    return
            if enabled:
                for sibling in (self._camera_states or {}).values():
                    if getattr(sibling, "object_watch", False):
                        sibling.attach_object_watch_runtime(runtime)
                if runtime is not None and (not was_enabled or old_settings != new_settings):
                    runtime.reset_camera(camera_id,
                                         self._source_generation.get(camera_id, 0))
            else:
                if runtime is not None:
                    runtime.reset_camera(camera_id,
                                         self._source_generation.get(camera_id, 0))
                state.attach_object_watch_runtime(None)
                if runtime is not None and not any(
                    getattr(sibling, "object_watch", False)
                    for sibling in (self._camera_states or {}).values()
                ):
                    runtime.stop()
                    self._object_watch_runtime = None

    def _synchronized_boxes(self, camera_id: str) -> bool:
        return (camera_id not in self.view_only
                and camera_id in getattr(self, "synchronized_box_cameras", ())
                and self.publisher.has_tracking_viewers(camera_id))

    def _smooth_publish_loop(self) -> None:
        """Ship the newest decoded frame per camera at publish_fps, overlaying
        the LAST detection's boxes — the live wall stops being chained to the
        model's cadence (user ask, 24 Aug: 'live stream smooth and fast, the
        detection on the other end'). Peek never consumes, so detection loses
        nothing. Legacy smooth boxes lag by sampling AND processing latency;
        opted-in exact-frame cameras instead publish from the inference path
        while any annotated viewer is connected."""
        from cvti.serving.wall_budget import WALL_DECODE_BUDGET_MS, WallCamera, allocate_wall_fps
        last_seq: dict = {}
        period = 1.0 / self.publish_fps
        # getattr: tests build pipelines raw, without __init__.
        budget_ms = getattr(self, "wall_decode_budget_ms", None) or WALL_DECODE_BUDGET_MS
        view_only = getattr(self, "view_only", set()) or set()
        baseline = float(getattr(self, "target_fps", 0.0) or 0.0)
        while True:
            t0 = time.perf_counter()
            # Boost decode only for cameras actually watched, sharing a decode
            # budget among them by MEASURED CPU cost after the detection
            # baseline is reserved (Tolu's audit and review, 15 Oct 2026).
            # Allocated once per tick for every camera; the accounting is
            # kept for health.
            try:
                allocation = allocate_wall_fps(
                    [WallCamera(cam_id, self.publisher.has_viewers(cam_id),
                                getattr(d, "frame_cpu_ms", None),
                                getattr(d, "source_width", 0) or getattr(d, "stream_width", 0),
                                getattr(d, "source_height", 0) or getattr(d, "stream_height", 0),
                                bool(getattr(d, "_cap_keyframes", False)),
                                0.0 if cam_id in view_only else baseline)
                     for cam_id, d in list(self._decoders.items())],
                    ceiling_fps=self.publish_fps, budget_ms=budget_ms)
                rates = allocation.rates
                self.wall_budget_status = allocation.summary()
            except Exception:  # noqa: BLE001 - the wall must keep publishing
                log.debug("wall budget allocation failed; keeping last rates", exc_info=True)
                rates = {}
            for cam_id, d in list(self._decoders.items()):
                try:
                    if cam_id in rates:
                        d.display_fps = rates[cam_id]
                    if self._synchronized_boxes(cam_id):
                        # Do not paint stale coordinates onto newer video.
                        # Frames with boxes are published by _route_to_queue.
                        continue
                    if d.playout is not None:
                        # Live URL source: paced playout — content plays at its
                        # own rate a bounded lag behind live, instead of
                        # freeze-then-fast-forward at every burst.
                        jpeg = d.playout.pop_due(time.perf_counter())
                        if jpeg is not None:
                            state = ((self._camera_states or {}).get(cam_id)
                                     if cam_id not in self.view_only else None)
                            latest, _ = d.peek_latest()
                            source_size = (latest.image.shape[:2]
                                           if latest is not None and state is not None
                                           else None)
                            _publish_jpeg(
                                self.publisher, cam_id, jpeg, state, source_size
                            )
                        continue
                    frame, seq = d.peek_latest()
                except Exception:  # noqa: BLE001 - one camera must not stop the wall
                    log.debug("peek failed for %s", cam_id, exc_info=True)
                    continue
                if frame is None or last_seq.get(cam_id) == seq:
                    continue
                last_seq[cam_id] = seq
                try:
                    state = ((self._camera_states or {}).get(cam_id)
                             if cam_id not in self.view_only else None)
                    _publish_frame(self.publisher, cam_id, frame.image, state)
                except Exception:  # noqa: BLE001
                    log.debug("smooth publish failed for %s", cam_id, exc_info=True)
            sleep = period - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _due_for_detection(self, camera_id: str, now: float) -> bool:
        """Detection samples at the camera's target fps even when decoders run
        faster for the live wall — without this gate, faster decode = 2-3x model
        load. The rate is the adaptive per-camera target when the controller is
        on, else the fixed self.target_fps."""
        cam_rates = getattr(self, "_cam_target_fps", None)
        rate = (cam_rates.get(camera_id) if cam_rates else None) or self.target_fps
        if rate <= 0:
            rate = self.target_fps
        last = self._last_detect.get(camera_id, 0.0)
        if now - last >= 0.95 / rate:
            self._last_detect[camera_id] = now
            return True
        return False

    # Never spin faster than this (100 loops/s) — a floor on the loop period so
    # a high adaptive target can't turn the run loop into a busy-wait.
    _MIN_LOOP_PERIOD_S = 0.01

    # Decoders keep a little more than the detection rate so the detection gate
    # (0.95 of the period) always finds a fresh frame, never one tick stale.
    _DECODE_HEADROOM = 1.25

    def _sync_decoder_rates(self) -> None:
        """Decode each camera at ITS adaptive detection rate.

        Decoders start at the fixed target_fps (detection rate, not publish rate:
        an unwatched camera must not be decoded flat out). Once the controller
        sizes a camera, its decoder follows that rate — otherwise the service's
        --target-fps 8 capped every camera at ~8-10 fps and the controller's
        floor could never be met (CHI, 22 Oct 2026). The decoder's own
        measured-sustainable cap and the viewer boost still apply."""
        for cam_id, rate in self._cam_target_fps.items():
            decoder = self._decoders.get(cam_id)
            if decoder is not None and rate > 0:
                want = rate * self._DECODE_HEADROOM
                # Never ask a decoder for more than the camera sends: a 30fps
                # stream asked for 150 reads as "this machine can't keep up"
                # and logs a false decode-limit warning on every window.
                source_fps = float(getattr(decoder, "_fps", 0.0) or 0.0)
                if source_fps > 1.0:
                    want = min(want, source_fps)
                decoder.target_fps = max(float(self.target_fps), want)

    def _loop_period(self) -> float:
        """How long the run loop sleeps between batches.

        Fixed at tick_seconds normally. With the adaptive controller on, the loop
        keeps up with the SUM of per-camera targets, not the max: a tick mostly
        carries one camera's newest frame (cameras at ~20fps produce a frame every
        ~50ms, staggered), so to serve N cameras each at its target the loop must
        tick at the combined rate. Sizing to the max under-ticked by a factor of N
        — four cameras each wanting 20fps got ~5fps apiece while the GPU sat idle
        (CHI, 22 Oct 2026). The controller still holds GPU use near target_busy,
        so ticking at the sum cannot run the card away, and MIN_LOOP_PERIOD caps
        the spin."""
        if self._adaptive is None or not self._cam_target_fps:
            return self.tick_seconds
        want = sum(v for v in self._cam_target_fps.values() if v > 0)
        if want <= 0:
            return self.tick_seconds
        return max(self._MIN_LOOP_PERIOD_S, min(self.tick_seconds, 1.0 / want))

    def _tick_adaptive(self, now: float) -> None:
        """Re-size the per-camera detection rates once per window.

        Busy fraction = detection milliseconds / real milliseconds since the last
        tick. The controller grows the shared budget when the GPU has headroom
        and shrinks it when detection is eating most of real time, then splits it
        across cameras by the detections each produced this window (the activity
        signal). Cheap and best-effort: any error leaves the current rates in
        place rather than disturbing the frame loop."""
        if self._adaptive is None:
            return
        if self._adapt_last_wall == 0.0:          # first call: start the window
            self._adapt_last_wall = now
            self._adapt_last_detect_ms = self._detect_ms_total
            return
        elapsed = now - self._adapt_last_wall
        if elapsed < self._adaptive_cfg.window_s:
            return
        try:
            detect_ms = self._detect_ms_total - self._adapt_last_detect_ms
            busy = detect_ms / (elapsed * 1000.0) if elapsed > 0 else 0.0
            cam_ids = [c for c in self._decoders if c not in self.view_only]
            cameras = [DetectCamera(cid, float(self._activity_count.get(cid, 0)) / elapsed)
                       for cid in cam_ids]
            if cameras:
                self._cam_target_fps = self._adaptive.update(
                    busy_fraction=busy, cameras=cameras)
                self._sync_decoder_rates()
        except Exception:  # noqa: BLE001 - never let budgeting break detection
            log.debug("adaptive fps tick failed", exc_info=True)
        finally:
            self._adapt_last_wall = now
            self._adapt_last_detect_ms = self._detect_ms_total
            self._activity_count.clear()

    def _handle_link_change(self, camera_id: str, previous: str,
                            state_name: str, held: float) -> None:
        """Fence in-flight object results before exposing a source reset."""
        source_reset = (
            state_name == "reconnecting" and previous != "reconnecting"
        ) or (
            state_name == "offline" and previous not in ("reconnecting", "offline")
        )
        if source_reset:
            now = time.monotonic()
            with self._source_lock:
                self._source_generation[camera_id] = \
                    self._source_generation.get(camera_id, 0) + 1
                camera_state = (self._camera_states or {}).get(camera_id)
                if camera_state is not None:
                    try:
                        reset_watch = getattr(camera_state, "reset_object_watch", None)
                        if reset_watch is not None:
                            reset_watch(self._source_generation[camera_id])
                        camera_state.reset_general_object_tracks(now, "source_reset")
                        snapshot = camera_state.general_object_snapshot(now)
                    except Exception:  # noqa: BLE001 - link handling must continue
                        snapshot = None
                        log.warning("general object source reset failed for %s",
                                    camera_id, exc_info=True)
                    if self.publisher is not None:
                        invalidate = getattr(
                            self.publisher, "invalidate_object_tracking", None
                        )
                        if invalidate is not None:
                            try:
                                invalidate(camera_id, snapshot)
                            except Exception:  # noqa: BLE001 - link handling continues
                                log.warning("object publisher reset failed for %s",
                                            camera_id, exc_info=True)
        if self.on_link_change is not None:
            self.on_link_change(camera_id, previous, state_name, held)

    def _object_sample_unavailable(self, camera_id: str, source_generation: int,
                                   observed_at: float) -> None:
        """Record a failed shared sample only if its source generation is live."""
        state = (self._camera_states or {}).get(camera_id)
        if state is None:
            return
        snapshot = None
        with self._source_lock:
            if self._source_generation.get(camera_id, 0) != source_generation:
                return
            try:
                update = getattr(state, "update_general_object_tracks", None)
                if update is not None:
                    update(None, observed_at)
                provider = getattr(state, "general_object_snapshot", None)
                if provider is not None:
                    snapshot = provider(observed_at)
            except Exception:  # noqa: BLE001 - optional lane stays isolated
                log.debug("general object unavailable update failed", exc_info=True)
                snapshot = None
            if self.publisher is not None:
                invalidate = getattr(self.publisher, "invalidate_object_tracking", None)
                if invalidate is not None:
                    try:
                        invalidate(camera_id, snapshot)
                    except Exception:  # noqa: BLE001 - publication remains best effort
                        log.debug("object metadata invalidation failed", exc_info=True)

    def _default_handler(self, frame: Frame, result: Any) -> None:
        n = 0 if result.boxes is None else len(result.boxes)
        self.per_cam[frame.camera_id] += n

    def _route_to_queue(self, frame: Frame, result: Any) -> None:
        import supervision as sv
        from cvti.detector.core import extract_detections
        state = self._camera_states.get(frame.camera_id)
        if state is None:
            return
        with self._source_lock:
            current_generation = self._source_generation.get(frame.camera_id, 0)
        source_generation, tracker_timestamp = self._inference_context.pop(
            id(frame), (current_generation, time.monotonic())
        )
        try:
            detections = sv.Detections.from_ultralytics(result)      # unfiltered shared result
        except Exception as exc:
            # Conversion failure is a known unavailable sample, not a successful
            # empty one. Preserve the old safety failure reporting and stop here.
            self._object_sample_unavailable(
                frame.camera_id, source_generation, tracker_timestamp
            )
            state._health.failed(exc, log, "converting shared detections")
            return
        try:
            with self._source_lock:
                # Observation time and generation were captured when this frame
                # was admitted to inference, not after a potentially slow model
                # pass. A reset either wins before this check or waits and then
                # clears this update; no old result can cross generations.
                if self._source_generation.get(frame.camera_id, 0) == source_generation:
                    ensure_tracker = getattr(state, "ensure_general_object_tracker", None)
                    update_tracker = getattr(state, "update_general_object_tracks", None)
                    if ensure_tracker is not None:
                        ensure_tracker(self._names, self.target_fps)
                    if update_tracker is not None:
                        update_tracker(detections, tracker_timestamp)
        except Exception:  # noqa: BLE001 - never block person tracking/rules
            log.warning("general object tracking failed for %s", frame.camera_id,
                        exc_info=True)
            self._object_sample_unavailable(
                frame.camera_id, source_generation, tracker_timestamp
            )
        object_detections = extract_detections(result, self._names, self._threat_classes)  # weapons/violence/theft
        # Invalidate the previous coordinates before processing. The timestamp
        # is inference admission time, not completion of a slow model call.
        state._motion_overlays = []
        previous = getattr(state, "_motion_overlay_observed_at", None)
        if previous is not None and tracker_timestamp > previous:
            # Smoothed interval between this camera's detections; the wall's
            # box lifetime follows it (see _overlay_max_age).
            last = float(getattr(state, "_detection_interval_s", 0.0) or 0.0)
            gap = tracker_timestamp - previous
            state._detection_interval_s = gap if not last else 0.5 * last + 0.5 * gap
        state._motion_overlay_observed_at = tracker_timestamp
        # `process` guards its detector section, but everything around it —
        # tracking, zones, rule evaluation, evidence selection — was unguarded,
        # so a failure there propagated out through run() and stopped EVERY
        # camera. The comment inside promised one bad detector could not kill the
        # camera loop; this is what makes that true.
        try:
            if getattr(state, "object_watch", False):
                alerts = state.process(
                    detections, frame.image, frame.timestamp,
                    object_detections=object_detections,
                    object_watch_source_generation=source_generation,
                    object_watch_observed_at=tracker_timestamp,
                )
            else:
                alerts = state.process(detections, frame.image, frame.timestamp,
                                       object_detections=object_detections)
        except Exception as exc:  # noqa: BLE001 - one camera must not stop the rest
            state._health.failed(exc, log, "processing a frame")
            return
        if os.environ.get("ARGUS_TRACKING_DIAGNOSTICS") == "1":
            count = getattr(state, "_diagnostic_frame_count", 0) + 1
            state._diagnostic_frame_count = count
            if count % 20 == 0:
                log.info("[tracking-diagnostic] camera=%s raw_people=%d tracked=%d overlays=%d normal_movement=%s",
                         frame.camera_id, sum(d.label == "person" for d in object_detections),
                         len(getattr(state, "_box_by_track", {})),
                         len(getattr(state, "_motion_overlays", [])), state.normal_movement)
        for alert in alerts:
            runtime = None
            candidate = (getattr(alert, "payload", None) or {}).get("candidate")
            if getattr(candidate, "detector", "") == "object_watch":
                watch_result = (alert.payload or {}).get("object_watch_result")
                runtime = getattr(state, "_object_watch_runtime", None)
                from cvti.serving.event_adapters import object_watch_alert_snapshot_current
                if runtime is None or watch_result is None \
                        or not object_watch_alert_snapshot_current(
                            alert, state, runtime, watch_result, time.monotonic()
                        ):
                    continue
            if self._alert_queue.add(alert):
                self.alerts_queued += 1
                if getattr(candidate, "detector", "") == "object_watch":
                    runtime.commit((alert.payload or {}).get("object_watch_token"))  # type: ignore[union-attr]
                if self.on_queued is not None:
                    try:
                        self.on_queued(alert)
                    except Exception:  # noqa: BLE001 - fast path must not stop routing
                        log.error("on_queued callback failed", exc_info=True)
        # The live wall is DECOUPLED from detection (user ask, 24 Aug): the
        # smooth-publish thread ships frames at stream cadence; detection only
        # refreshes its boxes. Exact-frame mode instead publishes this very
        # inference frame, trading display cadence for spatial alignment.
        # Per-track zone membership for the off-path scanners (PPE policy is
        # per zone). Cheap: the camera state already computed it this frame.
        self.latest_zones[frame.camera_id] = dict(getattr(state, "_zones_by_track", None) or {})
        # The PPE scanner reads these boxes too. They used to be recorded only
        # when a frame publisher existed, so an engine started with
        # --no-publish-frames (the CI KPI harness, any headless run) fed the
        # scanner frames with no people in them: "people_seen: 0" on a clip
        # full of workers, and never a violation (5 Oct, KPI 12 run).
        boxes = [(tid, *box) for tid, box in
                 (getattr(state, "_box_by_track", None) or {}).items()]
        self.latest_boxes[frame.camera_id] = boxes
        if self.publisher is not None:
            if alerts:
                self.publisher.mark_alerting(
                    frame.camera_id, _alert_track_ids(alerts))
            if not self.smooth_publish or self._synchronized_boxes(frame.camera_id):
                _publish_frame(self.publisher, frame.camera_id, frame.image, state,
                               exact_frame=True)

    def _all_ended(self) -> bool:
        return all(d.ended and not d.read_latest() for d in self._decoders.values())

    def run(self, max_seconds: float = 20.0) -> None:
        assert self._model is not None, "call start() first"
        t_end = time.perf_counter() + max_seconds
        last_report = time.perf_counter()
        while time.perf_counter() < t_end:
            tick = time.perf_counter()
            batch = collect_batch(self._decoders, max_batch=self.max_batch)
            if batch and self.view_only:
                batch = [f for f in batch if f.camera_id not in self.view_only]
            if batch and self.smooth_publish:
                now = time.perf_counter()
                batch = [f for f in batch if self._due_for_detection(f.camera_id, now)]
            if batch:
                observed_at = time.monotonic()
                if self._routes_to_queue:
                    with self._source_lock:
                        for frame in batch:
                            self._inference_context[id(frame)] = (
                                self._source_generation.get(frame.camera_id, 0), observed_at
                            )
                t0 = time.perf_counter()
                try:
                    results = self._camera_inference.predict(
                        self._model, batch, imgsz=self.imgsz, conf=self.conf,
                        device=self.device, half=self.half, verbose=False,
                    )
                except Exception:  # noqa: BLE001 - mark unavailable, keep engine alive
                    log.error("shared detection inference failed", exc_info=True)
                    for frame in batch if self._routes_to_queue else ():
                        generation, frame_observed_at = self._inference_context.pop(
                            id(frame), (
                                self._source_generation.get(frame.camera_id, 0),
                                observed_at,
                            ),
                        )
                        self._object_sample_unavailable(
                            frame.camera_id, generation, frame_observed_at
                        )
                    results = []
                batch_ms = (time.perf_counter() - t0) * 1000.0
                self._detect_ms_total += batch_ms
                from cvti.serving.perf import BOARD
                # units=len(batch): one observation, N frames' worth of work.
                # Without it per-frame detection cost reads as the whole batch's
                # and the readout ranks detection ~4x its true share.
                BOARD.observe("detect_batch", "engine", batch_ms, units=len(batch))
                for frame, result in zip(batch, results):
                    self.on_result(frame, result)
                    if self._adaptive is not None:
                        # Activity = detections seen this window; drives the
                        # per-camera split in _tick_adaptive.
                        boxes = getattr(result, "boxes", None)
                        self._activity_count[frame.camera_id] += (
                            0 if boxes is None else len(boxes))
                for frame in batch[len(results):]:
                    self._inference_context.pop(id(frame), None)
                self.batches += 1
                self.frames_processed += len(batch)
                self.batch_hist[len(batch)] += 1
            if self._adaptive is not None:
                self._tick_adaptive(time.perf_counter())
            if time.perf_counter() - last_report >= 3.0:
                self._report()
                last_report = time.perf_counter()
            if self._all_ended():
                log.info("[serving] all streams ended")
                break
            slept = self._loop_period() - (time.perf_counter() - tick)
            if slept > 0:
                time.sleep(slept)
        self._report(final=True)

    def _report(self, final: bool = False) -> None:
        avg_batch = (self.frames_processed / self.batches) if self.batches else 0.0
        avg_ms = (self._detect_ms_total / self.batches) if self.batches else 0.0
        fps = (self.frames_processed / (self._detect_ms_total / 1000.0)) if self._detect_ms_total else 0.0
        tag = "FINAL" if final else "stats"
        log.info(f"[{tag}] batches={self.batches} frames={self.frames_processed} "
              f"avg_batch={avg_batch:.1f} detect={avg_ms:.1f}ms/batch "
              f"model_capacity={fps:.0f} frames/s | batch_sizes={dict(sorted(self.batch_hist.items()))}")
        # capacity = frames / MODEL time (1/latency), not a wall-clock rate —
        # the old label "throughput" read as if detection ran that fast.
        if final:
            if self._camera_states is not None:
                log.info(f"[FINAL] alerts_queued={self.alerts_queued}")
            else:
                log.info(f"[FINAL] detections/camera={dict(self.per_cam)}")

    def stop(self) -> None:
        self._started = False
        for d in self._decoders.values():
            d.stop()
        if self._object_watch_runtime is not None:
            self._object_watch_runtime.stop()


@dataclass
class _LinkEvent:
    """A camera-link change, shaped like a candidate alert so it routes normally."""
    detector: str
    rule_name: str
    priority: str
    title: str
    person_id: Any = None
    object_label: Any = None
    track_id: Any = None
    zone: Any = None


def gate_reachable(stats: dict, now: float | None = None):
    """True/False/None — with hysteresis, because one bad verdict is not an outage.

    The old rule was 'whichever came last wins': a single timed-out verify
    flipped the status to UNAVAILABLE until the next success — minutes later
    on slow hardware. The pilot's laptop hit exactly this while screen-sharing
    a Meet call over a 90s-median verify (29 Aug): the AI was fine, merely
    busy, and the banner declared it dead and the English rules paused.

    False now requires sustained evidence: failures more recent than the last
    success AND no success within a patience window scaled to the machine's
    own measured latency. A genuinely absent server still reads False fast,
    because it has no last_success_at at all.
    """
    now = now or time.time()
    if not stats.get("verified") and not stats.get("unverified") \
            and not stats.get("errors"):
        return None
    last_ok = stats.get("last_success_at") or 0
    last_bad = stats.get("last_unverified_at") or 0
    if last_bad <= last_ok:
        return True
    patience = max(240.0, 4.0 * float(stats.get("median_latency_s") or 30.0))
    return (now - last_ok) <= patience


def warm_gate_model_async(gate_provider: str, gate_model: str,
                          gate_base_url: str = "") -> "threading.Thread | None":
    """Start loading the local VLM into memory while the rest of startup runs.

    Cold-start reality on pilot hardware (audit 1 Sep, V3): the first alert
    after engine start paid the full ~3 GB model load INSIDE its verify call
    — tens of seconds billed to "the gate is slow". Scene mapping hits the
    same wall. So the moment the engine knows it will need the local model,
    a background thread brings the server up (best-effort, same as the
    console does before spawning us) and issues Ollama's documented preload.
    Local providers only; cloud gates have nothing to warm. Never blocks and
    never fails startup — a warmup that loses just means the old behaviour.
    """
    if gate_provider not in ("ollama", "local"):
        return None

    def _warm() -> None:
        try:
            from cvti.verification import ollama as _ollama
            host = _ollama.host_from_base_url(gate_base_url)
            if _ollama.ensure_server(host):
                _ollama.warmup_model(gate_model or LOCAL_VLM_MODEL, host)
        except Exception:  # noqa: BLE001 - warmup is an optimisation, never a failure
            log.warning("VLM warmup thread failed; first verdict pays the load",
                        exc_info=True)

    thread = threading.Thread(target=_warm, name="vlm-warmup", daemon=True)
    thread.start()
    return thread


def _apply_gate_frame_cap(site: dict, gate_max_frames: int | None) -> dict:
    """A provider's image limit becomes the gate's frame cap unless the site
    already chose one. Groq refuses requests with more than three images
    (qwen3.8-27b, Oct 2026); the evidence builder sends four to six, so
    without this every verdict there would be a 400 and land UNVERIFIED."""
    if gate_max_frames and not site.get("gate_max_frames"):
        site["gate_max_frames"] = int(gate_max_frames)
    return site


def run_site(site_config_path: str, *, weights: str = "models/yolov8n.pt",
             target_fps: float = 5.0, imgsz: int = 640, conf: float = 0.4,
             device: str = "", half: bool = False, seconds: float = 90.0,
             gate_provider: str = "mock", gate_model: str = "", gate_base_url: str = "",
             mapper_provider: str = "", mapper_model: str = "", mapper_base_url: str = "",
             gate_sensitivity: str = "balanced", publish_frames: bool = True,
             publish_fps: float = 24.0, security_dir: str | None = None,
             memory_guard: bool = True, memory_warn_gb: float = 2.0,
             memory_critical_gb: float = 1.0,
             pose_weights: str = "models/yolov8n-pose.pt",
             weapon_weights: str = "models/weapon_best.pt", yolov5_repo: str = "external/yolov5",
             video_action_model_path: str = "runs/video_finetune/videomae",
             baseline_config: str | None = "configs/baseline_critical_v1.json",
             notify: str = "console", output_dir: str = "runs/serving",
             gate_workers: int = 0, gate_drain: float = 180.0,
             mobile_port: int = 8710, gate_max_frames: int | None = None) -> None:
    """End-to-end multi-camera run: shared batched detector + shared pose/weapon
    models -> per-camera track/zones/concealment/violence/weapons/theft/rules ->
    shared alert queue -> async VLM gate."""
    from pathlib import Path

    from cvti.serving.alert_queue import AlertQueue
    from cvti.serving.camera import build_camera_states, load_site_config
    from cvti.serving.gate_pool import GatePool, bypass_from_site
    from cvti.verification.gate import MOCK_GATE_BANNER, VerificationGate, assert_engine_gate_allowed

    # Before anything expensive loads: a mock gate confirms every alert without
    # looking at it. Refuse unless the operator asked for that by name.
    mock_gate = assert_engine_gate_allowed(gate_provider)
    if mock_gate:
        log.info(f"[site] *** {MOCK_GATE_BANNER} *** every alert will be confirmed WITHOUT verification.")

    site = load_site_config(site_config_path)
    _apply_gate_frame_cap(site, gate_max_frames)
    # Heartbeat THROUGH the slow parts, not just before them: the preflight
    # alone can run minutes on a fresh install (mapper timing out against a
    # model still downloading), and a single stale write would read as a
    # stalled engine. Ticks until the real health loop takes over.
    _starting = {"phase": "starting — mapping camera scenes"}
    _starting_stop = threading.Event()

    def _starting_beat() -> None:
        while not _starting_stop.wait(10.0):
            try:
                write_starting_health(output_dir, site,
                                      gate_provider=gate_provider,
                                      gate_model=gate_model,
                                      phase=_starting["phase"],
                                      scene_mapping=_starting.get("scene"))
            except Exception:  # noqa: BLE001 - the warm-up beat must never kill startup
                log.debug("starting heartbeat write failed", exc_info=True)

    write_starting_health(output_dir, site,
                          gate_provider=gate_provider, gate_model=gate_model)
    threading.Thread(target=_starting_beat, name="starting-beat",
                     daemon=True).start()
    # Start the model load NOW, in parallel with the preflight and the torch
    # loads below — by the time the first candidate alert needs a verdict the
    # weights are resident instead of cold on disk.
    warm_gate_model_async(gate_provider, gate_model, gate_base_url)
    mapping_service = build_mapper_service(
        output_dir=output_dir,
        gate_provider=gate_provider,
        gate_model=gate_model,
        gate_base_url=gate_base_url,
        mapper_provider=mapper_provider,
        mapper_model=mapper_model,
        mapper_base_url=mapper_base_url,
    )

    def _preflight():
        # inspect() reads caches and hierarchy only — no VLM inference — so
        # startup is fast and the wait-loop below can re-check cheaply. The
        # actual mapping happens in the background coordinator once running.
        # View-only cameras are glass: no context, no mapping, no VLM ever.
        return mapping_service.inspect(
            [c for c in (site.get("cameras") or []) if not c.get("view_only")],
            site.get("scene_context_policy", "auto"),
        )

    mapping_preflight = _preflight()
    cams_cfg = active_cameras_after_preflight(
        list(site.get("cameras") or []), mapping_preflight
    )
    mapping_health = _mapping_health_rows(mapping_preflight)
    # A strict policy with nothing approved used to EXIT here — and the
    # console's watchdog dutifully respawned the corpse forever. A pilot's
    # fresh install spent two days in that loop reading as 'the engine never
    # starts' (1 Sep). A policy block is a WAITING state, not a death: stay
    # alive, keep the heartbeat honest about why, and re-run the preflight —
    # the moment a human approves a context in the app (or a remap succeeds),
    # cameras start without anyone restarting anything.
    _view_only_cams = [c for c in (site.get("cameras") or []) if c.get("view_only")]
    while not cams_cfg and not _view_only_cams:
        _starting["phase"] = (
            "waiting — add a camera to begin monitoring"
            if not site.get("cameras") else
            "waiting — this site's policy requires reviewed "
            "scene context before cameras may start")
        _starting["scene"] = mapping_health
        log.warning("[site] no camera can start until scene context is ready — "
                    "retrying the preflight in 60s (approve or remap in the app)")
        time.sleep(60.0)
        # Reload the config: a camera added (or a context approved) from the
        # app while we wait must start cameras WITHOUT an engine restart.
        site = load_site_config(site_config_path)
        _apply_gate_frame_cap(site, gate_max_frames)
        mapping_preflight = _preflight()
        cams_cfg = active_cameras_after_preflight(
            list(site.get("cameras") or []), mapping_preflight
        )
        _view_only_cams = [c for c in (site.get("cameras") or [])
                           if c.get("view_only")]
        mapping_health = _mapping_health_rows(mapping_preflight)
    _starting["phase"] = "starting — loading detection models"
    _starting["scene"] = mapping_health
    # View-only cameras ride along: build_camera_states skips them, but the
    # decoder/publisher wiring below needs them in the roster to stream glass.
    site = {**site, "cameras": cams_cfg + _view_only_cams}
    # Load ONE shared pose model iff any camera enables a pose-based signal.
    pose_model = None
    if any(c.get(k) for c in cams_cfg for k in ("concealment", "violence", "theft")):
        from cvti.detector.core import load_ultralytics_model
        pose_model = load_ultralytics_model(pose_weights)
        log.info(f"[site] shared pose model loaded ({pose_weights})")
    # Detectors the operator configured that could NOT actually run. Feeds the
    # health doc: "weapons: true" with a model that failed to load used to be
    # one log line and then permanent silent non-coverage. (Audit 23 Aug, #10.)
    model_failures: list = []
    # Load ONE shared weapon model iff any camera enables weapons (best-effort).
    weapon_model = None
    if any(c.get("weapons") for c in cams_cfg):
        try:
            from cvti.detector.core import load_detection_model
            repo = yolov5_repo
            if not Path(repo).exists():                    # frozen app: resolve inside the bundle
                from cvti.utils import resource_path
                repo = str(resource_path(yolov5_repo))
            weapon_model = load_detection_model(weapon_weights, repo, preferred_kind="yolov5")
            log.info(f"[site] shared weapon model loaded ({weapon_weights})")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[site] weapon model unavailable ({str(exc)[:80]}); weapons disabled", exc_info=True)
            model_failures.append(f"weapons detector configured but its model failed to load: {str(exc)[:90]}")
            # Saying 'weapons disabled' must MAKE IT TRUE: cameras kept their
            # weapons flag with a None model and threw on every frame — 509
            # detector.cam1 errors in 28 minutes on the pilot's install while
            # health claimed the detector was merely 'configured but failed'.
            for c in cams_cfg:
                c["weapons"] = False
    # Load ONE shared video-action model iff any camera enables it (best-effort).
    video_action_model = None
    va_runner = None
    if any(c.get("video_action") for c in cams_cfg):
        try:
            from cvti.video_action_model import VideoMAEActionModel
            video_action_model = VideoMAEActionModel(video_action_model_path)
            # Eagerly, in the phase whose heartbeat says 'loading detection
            # models' — the wrapper is lazy, so until 2 Sep this log line
            # celebrated a load that had not happened, and the FIRST
            # theft-shaped event paid the whole torch+transformers load
            # inside the frame loop, on top of its inference (audit D1).
            video_action_model.load()
            log.info(f"[site] shared video-action model loaded ({video_action_model_path})")
        except Exception as exc:  # noqa: BLE001
            video_action_model = None
            log.warning(f"[site] video-action model unavailable ({str(exc)[:80]}); disabled", exc_info=True)
            model_failures.append(f"video-action detector configured but its model failed to load: {str(exc)[:90]}")
    if video_action_model is not None:
        from cvti.video_action_runtime import AsyncVideoActionRunner
        va_runner = AsyncVideoActionRunner().start()
    # Pose and weapon leave the frame loop the same way (Demi, 13 Oct 2026:
    # 50-72 s pose calls inline stalled every camera on the pilot server).
    heavy_runner = None
    if pose_model is not None or weapon_model is not None:
        from cvti.video_action_runtime import AsyncVideoActionRunner
        heavy_runner = AsyncVideoActionRunner(name="heavy-models").start()
    cams = build_camera_states(site, pose_model=pose_model, weapon_model=weapon_model,
                               video_action_model=video_action_model,
                               va_runner=va_runner,
                               heavy_runner=heavy_runner,
                               imgsz=imgsz,
                               baseline_config=baseline_config,
                               scene_contexts=mapping_preflight.contexts,
                               monitoring_scopes=monitoring_scopes_from_preflight(
                                   mapping_preflight,
                                   site.get("scene_context_policy", "auto"),
                               ),
                               reviewed_camera_ids={
                                   str(cid) for cid, s in
                                   mapping_preflight.statuses.items()
                                   if s.get("status") == "ready_reviewed"
                               }, output_dir=output_dir)
    sources = {cid: c["source"] for cid, c in cams.items()}
    states = {cid: c["state"] for cid, c in cams.items()}
    canonical_object_library = (Path(output_dir) / "object_library").resolve()
    for state in states.values():
        if not getattr(state, "object_watch", False):
            continue
        configured_library = getattr(state, "object_watch_library", None)
        if configured_library is not None \
                and Path(configured_library).expanduser().resolve() != canonical_object_library:
            raise ValueError(
                f"camera {state.camera_id}: object_watch_library conflicts with canonical site library"
            )
        state.object_watch_library = str(canonical_object_library)
    object_watch_runtime = None
    if any(getattr(state, "object_watch", False) for state in states.values()):
        try:
            from cvti.object_watch.runtime import ObjectWatchRuntime
            from cvti.object_watch.runtime_config import resolve_config
            object_watch_runtime = ObjectWatchRuntime(resolve_config(Path(output_dir)))
        except (OSError, TypeError, ValueError) as exc:
            model_failures.append(f"object watch unavailable: {str(exc)[:120]}")
            log.warning("[object-watch] unavailable; baseline monitoring continues: %s", exc)
    # View-only cameras stream to the wall with NO detection (4 Sep, pilot ask:
    # "just streaming the video"): they get a decoder and a publisher slot,
    # never a camera state, a detector, a scanner entry, or a scene mapping.
    view_only_ids = {str(c.get("id")) for c in (site.get("cameras") or [])
                     if c.get("view_only")}
    for c in site.get("cameras") or []:
        if str(c.get("id")) in view_only_ids and c.get("source") is not None:
            sources[str(c.get("id"))] = c["source"]
    queue = AlertQueue()

    # The go2rtc stream gateway (W1.3): every network camera opened ONCE by a
    # tool whose only job is streams, the engine reading the local restream —
    # and, for a camera with a `detect_source`, the cheap substream. An
    # upgrade, never a requirement: no binary / refused launch / site opt-out
    # ("stream_gateway": false) leaves every camera exactly where it is today,
    # and the fallback map below covers a gateway that dies mid-run.
    from cvti.serving.go2rtc import Go2rtcGateway
    from cvti.serving.onboarding import get_site_meta
    from cvti.detector.accel import backend_from_site
    _detector_backend = backend_from_site(get_site_meta(site_config_path))
    gateway = Go2rtcGateway(site.get("cameras") or [], output_dir)
    _gw_fallbacks: dict = {}
    if get_site_meta(site_config_path).get("stream_gateway", True):
        if gateway.start():
            for _cid in list(sources):
                _r = gateway.restream_url(_cid)
                if _r is not None:
                    _gw_fallbacks[_cid] = sources[_cid]   # the camera's own URL
                    sources[_cid] = _r
            gateway.write_descriptor()                    # W1.4: the API reads this
    else:
        gateway.disabled_reason = "disabled in the site file"
        log.info("[go2rtc] %s", gateway.disabled_reason)

    # Confirmed alerts are persisted (SQLite + evidence bundle) and notified,
    # instead of only printed. sink.handle is the gate's verdict callback.
    from cvti.serving.alert_sink import AlertSink, build_notifier
    # store_evidence: false in the site file = no frames or clips are written
    # for alerts (sites whose terms forbid recording off-premises).
    store_evidence = site.get("store_evidence", True) is not False
    if not store_evidence:
        log.info("[site] store_evidence=false: alerts are recorded without images or clips")
    sink = AlertSink(output_dir, notifier=build_notifier(notify), save_evidence=store_evidence)
    for state in states.values():
        # No evidence on disk means no replay clip to build: skip encoding one.
        state.clip_replay = store_evidence
    queue.on_generated = sink.audit_candidate_generated
    queue.on_admission = sink.audit_candidate_admission
    # W1.6: when detection rides a substream, the sink can still attach one
    # full-resolution mainstream frame per alert — go2rtc hands it over
    # without the engine decoding that stream at all.
    sink.full_frame_provider = gateway.snapshot_jpeg

    save_dir = Path(output_dir) / "gate"
    # Feedback loop: give the gate this site's recent operator-labeled examples for
    # each (camera, rule) as few-shot memory (cached ~60s). Empty on a fresh DB.
    from cvti.feedback.store import FeedbackStore
    _fb_store = FeedbackStore(str(Path(output_dir) / "events.db"))
    _fb_cache: dict = {}

    def _examples_provider(camera: str, rule: str) -> list:
        import time as _t
        key = (camera, rule)
        ent = _fb_cache.get(key)
        if ent and (_t.time() - ent[0]) < 60:
            return ent[1]
        ex = _fb_store.examples(camera, rule, k=4)
        _fb_cache[key] = (_t.time(), ex)
        return ex

    def _object_watch_candidate_current(candidate) -> bool:
        if (getattr(candidate, "metadata", {}) or {}).get("state") == "registered_object_changed":
            from cvti.serving.registered_objects import candidate_current
            try:
                return candidate_current(candidate, load_site_config(site_config_path))
            except (OSError, ValueError, TypeError):
                return False
        if getattr(candidate, "detector", "") != "object_watch":
            return True
        from types import SimpleNamespace
        from cvti.serving.event_adapters import object_watch_alert_current
        metadata = getattr(candidate, "metadata", None) or {}
        camera_id = str(metadata.get("camera_id") or "")
        state = states.get(camera_id)
        try:
            fresh_site = load_site_config(site_config_path)
            camera = next(c for c in fresh_site.get("cameras", ())
                          if str(c.get("id")) == camera_id)
            enabled = bool(camera.get("object_watch_enabled",
                                      camera.get("object_watch", False)))
        except (OSError, ValueError, StopIteration, TypeError):
            enabled = False
        library = getattr(state, "object_watch_library", None)
        alert = SimpleNamespace(payload={"candidate": candidate})
        return bool(enabled and library is not None
                    and object_watch_alert_current(alert, state, library))

    from cvti.serving.concealment_notices import ConcealmentNotices
    concealment_notices = ConcealmentNotices()

    def _deliver_current(alert, result):
        candidate = (getattr(alert, "payload", None) or {}).get("candidate")
        if not _object_watch_candidate_current(candidate):
            if (getattr(candidate, "metadata", {}) or {}).get("state") == "registered_object_changed":
                return None
            if getattr(candidate, "detector", "") == "object_watch":
                log.info("[object-watch] suppressed stale/disabled queued verdict for %s",
                         alert.camera_id)
                return None

        delivered = sink.handle(alert, result)
        concealment_notices.verdict(alert, result)
        return delivered

    class _FreshnessGuardedGate:
        def __init__(self):
            self._gate = VerificationGate(provider=gate_provider, model=gate_model,
                                          base_url=gate_base_url, save_dir=save_dir,
                                          sensitivity=gate_sensitivity,
                                          max_frames=site.get("gate_max_frames"))

        def verify(self, frames, candidate, scene, examples=None):
            if not _object_watch_candidate_current(candidate):
                return None
            return self._gate.verify(frames, candidate, scene, examples=examples)

        def __getattr__(self, name):
            return getattr(self._gate, name)

    gate_pool = GatePool(
        queue,
        gate_factory=_FreshnessGuardedGate,
        workers=_gate_workers_for(gate_workers, len(cams_cfg),
                                  provider=gate_provider, device=device),
        on_verdict=_deliver_current,
        examples_provider=_examples_provider,
        bypass=bypass_from_site(site),
        # site kill switch, same shape as the tier's: "enrich_bypassed": false
        enrich_bypassed=bool(site.get("enrich_bypassed", True)),
    ).start()

    # The UI reads frames from here instead of opening every stream a second time
    # (decode is the dominant per-camera cost) — and gets live boxes for free.
    from cvti.serving.frame_publisher import FramePublisher
    # Raw Watch tiles stay pure glass. Tracking viewers explicitly select the
    # separately cached annotated variant; they never alter the canonical JPEG.
    publisher = (FramePublisher().start(
        output_dir, cameras=[str(c.get("id")) for c in site.get("cameras") or []])
        if publish_frames else None)

    _offline_alerted: set[str] = set()   # cameras we've raised an OFFLINE alert for

    def _on_link_change(cam_id: str, previous: str, state: str, held: float) -> None:
        """A camera going offline raises its own alert through normal routing.

        Without this the customer believes they have coverage they do not have:
        an unreachable camera produces no alerts, which is indistinguishable
        from a camera watching a quiet area.

        Only a REAL outage is worth an alert. A camera reaches OFFLINE only after
        offline_grace (60s) of failed reconnects, so that fires once per genuine
        outage. "Recovered" then fires once to clear it. A brief WAN blip that
        reconnects before OFFLINE is not an outage and must stay silent — firing
        "recovered" on every >1s blink is what buried the feed in hundreds of
        auto-confirmed, unverified INFO alerts (CHI, 22 Oct 2026).
        """
        from cvti.serving.streams import CONNECTED, OFFLINE
        if state == OFFLINE:
            if cam_id in _offline_alerted:
                return                      # already alerted this outage
            _offline_alerted.add(cam_id)
            offline = True
        elif state == CONNECTED:
            if cam_id not in _offline_alerted:
                return                      # a blip that never reached OFFLINE — silent
            _offline_alerted.discard(cam_id)
            offline = False
        else:
            return
        queue.add(QueuedAlert(
            camera_id=cam_id,
            rule_name="camera_offline" if offline else "camera_recovered",
            priority="high" if offline else "info",
            title=(f"CAMERA OFFLINE — unreachable for {held:.0f}s"
                   if offline else "CAMERA RECOVERED — stream is back"),
            timestamp=time.time(),
            payload={"candidate": _LinkEvent(
                detector="camera_offline",
                rule_name="camera_offline" if offline else "camera_recovered",
                priority="high" if offline else "info",
                title=(f"Camera {cam_id} unreachable for {held:.0f}s" if offline
                       else f"Camera {cam_id} is back online")),
                     "frames": [], "scene": {}, "enqueued_at": time.time()}))

    from cvti.serving.dewarp import dewarps_for as _dewarps_for
    publish_fps = wall_fps_for(site, publish_fps, len(sources), compute_device(device))
    pipe = MultiStreamPipeline(sources, weights=weights, target_fps=target_fps, imgsz=imgsz,
                               conf=conf, device=device, half=half, camera_states=states,
                               alert_queue=queue, publisher=publisher,
                               on_link_change=_on_link_change, publish_fps=publish_fps,
                               view_only=view_only_ids,
                               fallback_sources=_gw_fallbacks,
                               detector_backend=_detector_backend,
                               object_watch_runtime=object_watch_runtime,
                               camera_inference={str(c["id"]): c["detection_inference"]
                                                 for c in site.get("cameras", [])
                                                 if "detection_inference" in c},
                               synchronized_box_cameras={str(c["id"]) for c in site.get("cameras", [])
                                                         if c.get("box_display") == "synchronized"},
                               camera_dewarp=_dewarps_for(site.get("cameras", [])),
                               keyframe_decode=(bool(site["keyframe_decode"])
                                                if isinstance(site.get("keyframe_decode"), bool) else None),
                               keyframe_min_width=int(site.get("keyframe_min_width", 2000) or 2000),
                               keyframe_cost_share=float(site.get("keyframe_cost_share", 0.25) or 0.25),
                               keyframe_min_fps=float(site.get("keyframe_min_fps", 0.5) or 0.5),
                               detector_threads=(int(site["detector_threads"])
                                                 if isinstance(site.get("detector_threads"), int)
                                                 and not isinstance(site.get("detector_threads"), bool)
                                                 and site["detector_threads"] > 0 else None),
                               wall_decode_budget_ms=(float(site["wall_decode_budget_ms"])
                                                      if isinstance(site.get("wall_decode_budget_ms"), (int, float))
                                                      and not isinstance(site.get("wall_decode_budget_ms"), bool)
                                                      and site["wall_decode_budget_ms"] > 0 else None),
                               adaptive_fps=AdaptiveConfig.from_site(site.get("adaptive_fps")))

    def _fast_path(alert) -> None:
        """Two-tier alerting (EP-06-T4): criticals are shown provisionally the
        moment the detector fires; the verdict updates the same row in place."""
        concealment_notices.candidate(alert)
        if getattr((alert.payload or {}).get("candidate"), "detector", "") == "concealment":
            return  # transient warning first; persist only a verdict or disputed review
        if alert.priority != "critical":
            return
        candidate = (alert.payload or {}).get("candidate")
        if candidate is not None and getattr(candidate, "detector", "") == "camera_offline":
            return                      # deterministic alerts confirm instantly anyway
        event_id = sink.provisional(alert)
        if event_id is not None and alert.payload is not None:
            alert.payload["provisional_event_id"] = event_id

    pipe.on_queued = _fast_path
    pipe.start()

    # The mapper reuses the ENGINE's frames once decoding runs (4 Sep): many
    # cameras cap concurrent RTSP sessions (the pilot's Tapo allows two), and
    # the mapper opening its own capture while the engine holds one got
    # refused — 'scene mapping failed: Unable to open source' on a camera
    # that was demonstrably streaming (detector.cam1: 178 ok, same screen).
    def _mapper_frame(cam_id: str):
        d = pipe._decoders.get(cam_id)
        if d is None:
            return None
        f, _seq = d.peek_latest()
        return f.image if f is not None else None

    mapping_service.frame_source = _mapper_frame

    # Cameras start immediately in critical-only mode while one persistent,
    # bounded worker maps missing/stale scenes in the background. This keeps a
    # 100-camera commissioning run from becoming 100 blocking Ollama calls.
    from cvti.scene.coordinator import SceneMappingCoordinator
    from cvti.scene.context_store import SceneContextStore

    mapping_coordinator = SceneMappingCoordinator(
        mapping_service, site_config_path, output_dir, max_workers=1,
        policy=site.get("scene_context_policy", "auto"),
    )
    initial_mapping_ids = [
        camera_id for camera_id, status in mapping_preflight.statuses.items()
        if status.get("status") in {"pending", "stale", "failed"}
    ]
    if initial_mapping_ids:
        mapping_coordinator.enqueue(initial_mapping_ids)
    _mapping_stop = threading.Event()

    def _mapping_worker() -> None:
        retry_due: dict[str, float] = {}
        retry_delays = (120.0, 300.0, 600.0)

        def _sync_states_and_health() -> None:
            """Fold finished mappings into the RUNNING cameras and the health
            rows. Reviewed contexts activate fully (suppression unlocked);
            unreviewed ones inform but keep scene_reviewed False. The rows
            use the same policy-aware semantics as the startup preflight —
            the worker used to fork them ('usable' flipping meaning after
            the first background job) and gate_health.json contradicted
            itself."""
            policy = mapping_coordinator.policy
            for camera_id, state in states.items():
                store = SceneContextStore(Path(output_dir) / "context", camera_id)
                status = store.load_status()
                context = store._load_context()
                if context is not None:
                    if status.status == "ready_reviewed":
                        state.activate_scene_context(context)
                    else:
                        state.scene_context = context
                usable = status.status == "ready_reviewed" or (
                    status.status == "ready_unreviewed" and policy == "auto")
                for index, row in enumerate(mapping_health):
                    if str(row.get("camera_id")) == str(camera_id):
                        mapping_health[index] = {
                            "camera_id": camera_id,
                            **status.to_dict(),
                            "usable": usable,
                            "review_required": policy == "require_reviewed",
                            "environment_type": (
                                context or {}
                            ).get("environment_type", "unknown"),
                        }

        while not _mapping_stop.wait(0.25):
            # Don't burn mapping attempts against a gate that is PROVABLY
            # down — the 3.3GB first-run model pull is exactly when every
            # attempt would fail. None means merely untested: a quiet site
            # with no alerts must still map.
            if gate_reachable(gate_pool.stats()) is False:
                if _mapping_stop.wait(15.0):
                    return
                continue
            try:
                ran = mapping_coordinator.run_next()
            except Exception:  # noqa: BLE001 - one bad job must never kill the
                # mapping thread; the queue is persistent and the next tick
                # moves on. (A deleted camera used to end scene mapping for
                # the life of the process.)
                log.exception("[agent-map] mapping job failed unexpectedly")
                if _mapping_stop.wait(5.0):
                    return
                continue
            if ran:
                _sync_states_and_health()
                continue
            now = time.monotonic()
            due = []
            for job in mapping_coordinator.jobs:
                if job.state != "failed" or job.attempts >= 4:
                    retry_due.pop(job.camera_id, None)
                    continue
                deadline = retry_due.setdefault(
                    job.camera_id,
                    now + retry_delays[min(job.attempts - 1, len(retry_delays) - 1)],
                )
                if now >= deadline:
                    due.append(job.camera_id)
            if due:
                mapping_coordinator.requeue_failed(due, max_attempts=4)
                for camera_id in due:
                    retry_due.pop(camera_id, None)
                continue
            if _mapping_stop.wait(2.0):
                return
            try:
                stale_ids = []
                for camera_id in states:
                    status = SceneContextStore(
                        Path(output_dir) / "context", camera_id
                    ).load_status()
                    if status.status == "stale":
                        stale_ids.append(camera_id)
                if stale_ids:
                    mapping_coordinator.enqueue(stale_ids, force=True)
                # Approvals land from the console's separate process; fold
                # them in even when no mapping job ran.
                _sync_states_and_health()
            except Exception:  # noqa: BLE001 - housekeeping must not kill the thread
                log.debug("[agent-map] stale/approval sweep failed", exc_info=True)

    threading.Thread(
        target=_mapping_worker, name="scene-mapping", daemon=True
    ).start()
    custom_scanner = None
    watch_runner = None      # defined here: teardown references it on every path
    scanner_endpoint = scanner_endpoint_for(gate_provider, gate_base_url)
    if scanner_endpoint is not None:
        scanner_base_url, scanner_key_env = scanner_endpoint
        # Watches: plain-English subjects to FOLLOW. Binds a description to a
        # tracked person (via numbered boxes) and keeps a case open for them.
        if any(c.get("watches") for c in cams_cfg):
            from cvti.serving.watch_runner import WatchRunner

            def _latest_frame(cam_id: str):
                # peek, never read: read_latest() CONSUMES the frame, and this
                # helper was quietly stealing frames from detection on every
                # watch check (same family as the scanner's double-decode).
                d = pipe._decoders.get(cam_id)
                if d is None:
                    return None
                f, _seq = d.peek_latest()
                return f.image if f is not None else None

            watch_runner = WatchRunner(
                cams_cfg, states, sink, model=gate_model or LOCAL_VLM_MODEL,
                base_url=scanner_base_url, api_key_env=scanner_key_env,
                frame_source=_latest_frame).start()

        # Customer-written English rules run as a slow VLM scan — the VLM IS the
        # detector, no person-trigger required (an aeroplane on an empty apron
        # counts). Always started: it watches the site file and begins scanning
        # a camera within one cycle of a sentence being typed in the app.
        from cvti.serving.custom_rules import CustomRuleScanner

        def _scanner_frame(cam_id: str):
            d = pipe._decoders.get(cam_id)
            if d is None:
                return None
            f, _seq = d.peek_latest()
            return f.image if f is not None else None

        custom_scanner = CustomRuleScanner(
            cams_cfg, sink, model=gate_model or LOCAL_VLM_MODEL,
            base_url=scanner_base_url, api_key_env=scanner_key_env,
            site_config_path=site_config_path,
            frame_source=_scanner_frame,
            # The tracker's person boxes ground a person claim's evidence box
            # — the VLM says what, the detector says where (4 Sep).
            boxes_source=lambda cam_id: pipe.latest_boxes.get(cam_id),
            context_provider=lambda camera_id: getattr(
                states.get(camera_id), "scene_context", None
            ))
        custom_scanner.status_path = Path(output_dir) / "english_rules_status.json"
        custom_scanner.start()
    # PPE compliance (per-person, three-state, policy per zone) runs as its own
    # bounded worker off the camera path — a grounded detector at ~1 Hz per
    # camera, no VLM, so it does not depend on the gate provider. Started when
    # any camera carries a `ppe` block; it watches the site file for new ones.
    ppe_scanner = None
    if any(c.get("ppe") for c in cams_cfg) or site_config_path:
        from cvti.ppe.scanner import PPEScanner

        def _ppe_frame(cam_id: str):
            d = pipe._decoders.get(cam_id)
            if d is None:
                return None
            f, _seq = d.peek_latest()
            return f.image if f is not None else None

        ppe_scanner = PPEScanner(
            cams_cfg, sink,
            frame_source=_ppe_frame,
            boxes_source=lambda cam_id: pipe.latest_boxes.get(cam_id),
            zones_source=lambda cam_id: pipe.latest_zones.get(cam_id),
            site_config_path=site_config_path,
            status_path=Path(output_dir) / "ppe_status.json",
        ).start()
    # Retention. Storage limitation is not optional, and an edge box with no
    # purge fills its disk and stops recording evidence exactly when it matters.
    from cvti.serving.onboarding import get_site_meta
    from cvti.serving.retention import RetentionManager, RetentionPolicy
    try:
        _policy = RetentionPolicy.from_site(get_site_meta(site_config_path))
    except Exception:  # noqa: BLE001 - a bad setting must not stop monitoring
        log.warning("retention: could not read site policy; using defaults", exc_info=True)
        _policy = RetentionPolicy()
    retention = RetentionManager(output_dir, _policy).start()

    # Watch our own footprint: on an edge box, running out means swapping, and a
    # swapping box misses every camera. Shed load on purpose instead.
    mem_guard = None
    if memory_guard:
        from cvti.serving.memory_guard import MemoryGuard, build_default_actions
        warn_actions, crit_actions = build_default_actions(pipe, states)
        mem_guard = MemoryGuard(warn_available_gb=memory_warn_gb,
                                critical_available_gb=memory_critical_gb,
                                warn_actions=warn_actions,
                                critical_actions=crit_actions).start()

    log.info(f"[site] {len(states)} camera(s) | gate={gate_provider} | "
             f"notify={notify_channels(notify)} | rules per camera")

    # Escalation ticker: re-notify alerts nobody acknowledged in time. Cheap poll,
    # daemon so it never holds shutdown open.
    import threading as _threading
    _esc_stop = _threading.Event()

    def _escalation_loop() -> None:
        while not _esc_stop.wait(30.0):
            try:
                sink.run_escalations()
            except Exception as exc:  # noqa: BLE001
                log.error(f"[routing] escalation tick failed: {str(exc)[:90]}", exc_info=True)

    _esc_thread = _threading.Thread(target=_escalation_loop, name="escalations", daemon=True)
    _esc_thread.start()

    # Gate health, published for the UI's System panel. The engine is a separate
    # process from the app, so a file next to events.db is the channel — same
    # trick the frame publisher uses.
    _health_path = Path(output_dir) / "gate_health.json"
    # Last totals written to the ledger, so each tick contributes only its delta.
    _ledger_seen = {"confirmed": 0, "rejected": 0, "deduped": 0, "errors": 0}

    _started_at = time.time()
    assurance = None            # constructed below, once the decoders exist
    heartbeat = None            # constructed below, iff the site opted in

    def _gate_reachable(stats: dict):
        return gate_reachable(stats)

    def _detector_health() -> dict:
        from cvti.detector.accel import detector_health
        return detector_health(pipe.detector_info, pipe._model)

    def _build_health() -> dict:
        """The /health document (EP-04-T1): six signal classes, one status.
        Written to gate_health.json for the app, served over HTTP by the
        publisher, and the future heartbeat sends exactly this."""
        from cvti.health import snapshot as health_snapshot
        from cvti.serving.health_doc import build_health_doc
        from cvti.serving.memory_guard import sample_memory
        stats = gate_pool.stats()
        # Link state per camera, so the UI shows coverage rather than inferring
        # it from the absence of alerts.
        cameras = [d.link_status() for d in pipe._decoders.values()]
        # What detection saw per camera (counts only), so "is it detecting"
        # is a number in health and in the heartbeat, not a walk test.
        from cvti.serving.camera import detection_status
        for row in cameras:
            st = states.get(row.get("camera_id"))
            row["detection"] = detection_status(st) if st is not None else None
        # Per-component counters. Without these, "this detector found nothing"
        # and "this detector has thrown on every frame for a week" are the same
        # silence.
        components = health_snapshot()
        ret = retention.status()
        mem = sample_memory()
        gate_doc = {"provider": gate_provider, "model": gate_model,
                    "reachable": _gate_reachable(stats), **stats}
        doc = build_health_doc(
            started_at=_started_at, cameras=cameras, gate=gate_doc,
            disk=ret.get("disk") or {},
            memory={"available_gb": round(mem.available_gb, 2),
                    "rss_gb": round(mem.rss_gb, 2),
                    "level": mem.level(memory_warn_gb, memory_critical_gb)},
            components=components,
            engine={"frames_processed": pipe.frames_processed,
                    "alerts_queued": pipe.alerts_queued,
                    "target_fps": target_fps, "cameras": len(pipe._decoders),
                    "wall_budget": getattr(pipe, "wall_budget_status", None),
                    "context_suppressions": sum(
                        state.context_suppression_count for state in states.values()
                    )},
            self_test=(assurance.last_result if assurance else {}),
            scene_mapping=mapping_health)
        if model_failures:
            # Configured coverage that is NOT running is at least degraded,
            # and the reason is named — never inferred from silence.
            doc["reasons"] = list(doc.get("reasons") or []) + model_failures
            if doc.get("status") == "ok":
                doc["status"] = "degraded"
        # Legacy keys the System panel already reads — kept at the top level so
        # the app needs no migration.
        doc.update(stats)
        doc.update({"provider": gate_provider, "model": gate_model,
                    "mock": mock_gate, "banner": MOCK_GATE_BANNER if mock_gate else "",
                    "updated_at": time.time(), "health": components,
                    "retention": ret,
                    # Detection -> operator-visible latency per priority tier.
                    # "Criticals alert in under a second" carries its measurement.
                    "alert_latency": sink.latency_stats(),
                    # W1.5: the gateway is a component like any other — running
                    # or not, and when not, WHY. Decoders that fell back to
                    # their camera's own URL already say so per-camera in
                    # cameras[].gateway_fallback.
                    "stream_gateway": gateway.status(),
                    # W2: which backend detection runs on, on what provider,
                    # and when it is not the accelerator — why not.
                    "detector": _detector_health(),
                    "heartbeat": heartbeat.status() if heartbeat else {"enabled": False}})
        for camera in doc.get("cameras", []):
            camera["concealment_notice"] = concealment_notices.snapshot(
                str(camera.get("camera_id")))
            state = states.get(str(camera.get("camera_id")))
            if state is not None and getattr(state, "object_watch", False):
                camera["object_watch"] = state.object_watch_status()
        return doc

    def _write_health() -> None:
        st = _build_health()
        try:
            from cvti.serving.health_history import record as _record_health
            _record_health(output_dir, st)
        except Exception:  # noqa: BLE001 - history must never hurt monitoring
            log.debug("health history write failed", exc_info=True)
        # Stage-by-stage timing percentiles, same cadence as health. This file
        # rides the diagnostics bundle so a support zip says WHERE the time
        # goes on the customer's machine instead of inviting a guess.
        from cvti.serving.perf import write_report as _write_perf
        _write_perf(output_dir)
        try:
            _health_path.write_text(json.dumps(st))
        except OSError:
            pass   # health reporting must never take the engine down
        # The value surface needs these to outlive the process — see
        # AlertSink.record_suppression.
        try:
            delta = {k: (st.get(k) or 0) - v for k, v in _ledger_seen.items()}
            sink.record_suppression(shown=delta["confirmed"], rejected=delta["rejected"],
                                    deduped=delta["deduped"], errors=delta["errors"])
            for k in _ledger_seen:
                _ledger_seen[k] = st.get(k) or 0
        except Exception as exc:  # noqa: BLE001 - bookkeeping never stops the engine
            log.error(f"[value] suppression ledger write failed: {str(exc)[:90]}", exc_info=True)

    def _health_loop() -> None:
        while not _esc_stop.wait(3.0):
            activated = activate_reviewed_scene_contexts(states, output_dir)
            if activated:
                log.info("[agent-map] full monitoring activated for: %s",
                         ", ".join(activated))
            _write_health()

    _starting_stop.set()          # the real heartbeat owns the file from here
    _write_health()
    _threading.Thread(target=_health_loop, name="gate-health", daemon=True).start()

    if publisher is not None:
        # /health on the publisher's authenticated server (EP-04-T1).
        publisher.health_provider = _build_health

    # Mobile response view (EP-06-T3): the guard's phone, on the site network.
    # Authenticated on every route; Telegram alerts deep-link into it.
    mobile = None
    if mobile_port:
        from cvti.serving.mobile import MobileServer
        # Port taken? Walk forward a few before giving up — and if it still
        # fails, say so in /health instead of only the log: a dead mobile view
        # silently strips the response link from every notification.
        # (Audit 23 Aug, #11.)
        for _try_port in range(mobile_port, mobile_port + 5):
            try:
                mobile = MobileServer(output_dir, port=_try_port,
                                      security_dir=security_dir).start()
                sink.mobile_base = mobile.base_url()
                if _try_port != mobile_port:
                    log.warning("mobile view: port %s taken, serving on %s instead",
                                mobile_port, _try_port)
                break
            except OSError as exc:
                _mobile_err = exc
        if mobile is None:
            log.warning("mobile view could not start on ports %s-%s: %s — alerts "
                        "will not carry a response link", mobile_port,
                        mobile_port + 4, _mobile_err)
            model_failures.append(
                f"mobile response view failed to start (ports {mobile_port}-{mobile_port + 4} busy) "
                "— notifications carry no respond link")

    # Heartbeat (EP-04-T2): OFF unless the site configured a URL. Sends the
    # whitelisted health payload outbound only; docs/HEARTBEAT.md is the schema.
    _site_meta = get_site_meta(site_config_path)
    if _site_meta.get("heartbeat_url"):
        from cvti.serving.heartbeat import Heartbeat
        heartbeat = Heartbeat(
            url=_site_meta["heartbeat_url"], site_key=_site_meta.get("heartbeat_key", ""),
            site_id=(_site_meta.get("name") or "site").strip().lower().replace(" ", "-"),
            health_provider=_build_health, output_dir=output_dir).start()

    # Live re-read of operator-editable site meta (audit 23 Aug, #5/#6/#13).
    # The UI edits notify / retention / heartbeat in the site file and implies
    # immediate effect; the engine used to read them once at spawn, so a test
    # alert would arrive on the NEW channel while real alerts kept going to the
    # old one until a restart nobody knew to do — and the UI claimed a
    # retention period the purger was not enforcing. The engine now watches
    # the file (mtime, 5s) and applies those three live. Cameras/zones/rules
    # still require a restart: they rebuild models, which is a real restart.
    def _rule_fingerprint():
        """mtimes of everything that shapes rules per camera: the site file's
        camera entries plus each camera's rules + zones files."""
        out = {}
        try:
            fresh = load_site_config(site_config_path)
        except Exception:  # noqa: BLE001 - half-written file: keep the old view
            log.debug("site re-read failed", exc_info=True)
            return None, None
        for cam in fresh.get("cameras", []):
            sig = []
            for key in ("config", "zones"):
                path = cam.get(key)
                try:
                    sig.append((key, path, Path(path).stat().st_mtime if path else 0))
                except OSError:
                    sig.append((key, path, 0))
            sig.append(("camera", json.dumps(cam, sort_keys=True, default=str)))
            runtime_path = Path(output_dir) / "object_library" / "runtime.json"
            try:
                runtime_stat = runtime_path.stat()
                sig.append(("object_runtime", runtime_stat.st_mtime_ns,
                            runtime_stat.st_size))
            except OSError:
                sig.append(("object_runtime",))
            out[cam.get("id")] = tuple(sig)
        return fresh, out

    def _watch_site_meta():
        state = {"mtime": 0.0, "notify": notify,
                 "hb_url": _site_meta.get("heartbeat_url", ""),
                 "hb_key": _site_meta.get("heartbeat_key", "")}
        try:
            state["mtime"] = Path(site_config_path).stat().st_mtime
        except OSError:
            pass
        _, state["rules_fp"] = _rule_fingerprint()
        while True:
            time.sleep(5)
            # English rules and zones are JSON, not models: hot-swap them into
            # the running per-camera states so "describe it in English" takes
            # effect in seconds, not at the next restart. (User feedback,
            # 23 Aug: "it should kick off automatically".)
            fresh_site, fp = _rule_fingerprint()
            if fp is not None and fp != state.get("rules_fp"):
                for cam in fresh_site.get("cameras", []):
                    cid = cam.get("id")
                    if cid in states and fp.get(cid) != (state.get("rules_fp") or {}).get(cid):
                        try:
                            pipe._refresh_camera_state(cam, baseline_config)
                            log.info("[site] rules/zones hot-reloaded for camera %s", cid)
                        except Exception:  # noqa: BLE001 - keep the old rules over none
                            log.error("[site] rules hot-reload failed for %s; keeping "
                                      "current rules", cid, exc_info=True)
                state["rules_fp"] = fp
            try:
                mtime = Path(site_config_path).stat().st_mtime
                if mtime == state["mtime"]:
                    continue
                state["mtime"] = mtime
                meta = get_site_meta(site_config_path)
            except Exception:  # noqa: BLE001 - a half-written file is retried next tick
                log.debug("site meta re-read failed; retrying", exc_info=True)
                continue
            new_notify = (meta.get("notify") or "console").strip()
            if new_notify != state["notify"]:
                try:
                    sink.notifier = build_notifier(new_notify)
                    state["notify"] = new_notify
                    log.info("[site] notifier changed live -> %s", new_notify)
                except Exception:  # noqa: BLE001 - keep the old channel over none
                    log.error("[site] new notifier %r failed to build; keeping the old one",
                              new_notify, exc_info=True)
            try:
                new_policy = RetentionPolicy.from_site(meta)
                if new_policy != retention.policy:
                    retention.policy = new_policy
                    log.info("[site] retention policy changed live -> %s days",
                             new_policy.days)
            except Exception:  # noqa: BLE001 - a bad setting must not stop the watcher
                log.warning("[site] new retention setting unreadable; keeping current",
                            exc_info=True)
            hb_url = (meta.get("heartbeat_url") or "").strip()
            hb_key = (meta.get("heartbeat_key") or "").strip()
            if (hb_url, hb_key) != (state["hb_url"], state["hb_key"]):
                nonlocal heartbeat
                if heartbeat:
                    heartbeat.stop()
                    heartbeat = None
                if hb_url:
                    from cvti.serving.heartbeat import Heartbeat
                    heartbeat = Heartbeat(
                        url=hb_url, site_key=hb_key,
                        site_id=(meta.get("name") or "site").strip().lower().replace(" ", "-"),
                        health_provider=_build_health, output_dir=output_dir).start()
                    log.info("[site] heartbeat %s live", "reconfigured" if state["hb_url"] else "enabled")
                else:
                    log.info("[site] heartbeat disabled live")
                state["hb_url"], state["hb_key"] = hb_url, hb_key

    threading.Thread(target=_watch_site_meta, name="site-meta-watch", daemon=True).start()

    # Weekly owner summary (EP-08-T2): automatic, no action required. Checked
    # hourly; fires Monday 08:00+ at most once per ISO week, survives restarts
    # via summaries/state.json, and delivers through the same notifier alerts
    # use — the person deciding renewal hears from the product weekly.
    def _weekly_summary_loop():
        from cvti.owner_summary import due, mark_sent, weekly_summary
        while True:
            time.sleep(3600)
            try:
                if not due(output_dir):
                    continue
                meta = get_site_meta(site_config_path)
                s = weekly_summary(Path(output_dir) / "events.db", meta, output_dir)
                w = s["week"]
                sink.notifier.notify({
                    "ts": time.time(), "iso": s["generated_at"],
                    "camera_id": "weekly_summary", "rule": "weekly_summary",
                    "priority": "low", "confidence": None, "zone": None,
                    "track_id": None, "object_label": None, "evidence_dir": None,
                    "reason": (f"Weekly summary {s['window']['from']}->{s['window']['to']}: "
                               f"{w['incidents']} incidents ({w['real']} real), "
                               f"{w['noise_removed']} false alarms filtered, "
                               f"~{s['hours_saved']}h of attention given back. "
                               f"Full report: {s['pdf']}")})
                mark_sent(output_dir)
            except Exception:  # noqa: BLE001 - the summary must never hurt monitoring
                log.warning("weekly summary failed; will retry next hour", exc_info=True)

    threading.Thread(target=_weekly_summary_loop, name="weekly-summary", daemon=True).start()

    # Daily proof of life (EP-04-T4): the self-test exercises a real frame ->
    # the real gate -> a real notification, and the all-normal message makes
    # silence stop being the success signal. Skipped entirely for a mock gate —
    # a self-test against a gate that confirms everything proves nothing.
    if not mock_gate:
        from cvti.serving.assurance import Assurance

        def _any_latest_frame():
            for d in pipe._decoders.values():
                f, _seq = d.peek_latest()      # peek: detection owns read_latest
                if f is not None:
                    return f.image
            return None

        assurance = Assurance(
            latest_frame=_any_latest_frame,
            gate_factory=lambda: VerificationGate(provider=gate_provider, model=gate_model,
                                                  base_url=gate_base_url,
                                                  sensitivity=gate_sensitivity),
            notifier=sink.notifier,
            status_provider=_build_health,
            daily_normal=bool(get_site_meta(site_config_path).get("daily_normal", True)),
        ).start()


    try:
        pipe.run(max_seconds=seconds)
    finally:
        _mapping_stop.set()
        _esc_stop.set()
        pipe.stop()
        gateway.stop()
        if heavy_runner is not None:
            heavy_runner.stop()
            if heavy_runner.dropped or heavy_runner.failed:
                log.info(f"[heavy-models] passes={heavy_runner.completed} "
                         f"replaced_stale={heavy_runner.dropped} failed={heavy_runner.failed}")
        if va_runner is not None:
            va_runner.stop()
            if va_runner.dropped or va_runner.failed:
                log.info(f"[VideoAction] clips analysed={va_runner.completed} "
                         f"replaced_stale={va_runner.dropped} failed={va_runner.failed}")
        if custom_scanner is not None:
            custom_scanner.stop()
        if ppe_scanner is not None:
            ppe_scanner.stop()
        if watch_runner is not None:
            watch_runner.stop()
            act = watch_runner.active_cases()
            if act or watch_runner.opened:
                log.info(f"[watch] cases opened={watch_runner.opened} still_active={len(act)}")
        # Let in-flight VLM verdicts finish (a real gate is ~12s/verify, so the
        # queue keeps draining after the streams end) before tearing down.
        pending = queue.pending_count
        if pending or gate_pool._active:
            log.info(f"[site] draining gate: {pending} queued verdict(s) (up to {gate_drain:.0f}s)…")
            drained = gate_pool.drain(timeout=gate_drain)
            log.info(f"[site] gate drained cleanly={drained}")
        gate_pool.stop()
        if mem_guard is not None:
            mem_guard.stop()
            if mem_guard.mitigations:
                log.warning(f"[memory] shed load {len(mem_guard.mitigations)} time(s): "
                      f"{'; '.join(mem_guard.mitigations)}")
        retention.stop()
        if assurance is not None:
            assurance.stop()
        if heartbeat is not None:
            heartbeat.beat()      # a final send so "last seen" reflects shutdown time
            heartbeat.stop()
        if mobile is not None:
            mobile.stop()
        if publisher is not None:
            log.info(f"[frames] published {publisher.published} frame(s)")
            publisher.stop()
        _write_health()          # final totals, while the sink's DB is still open
        sink.close()
    log.info(f"[site] alerts_queued={pipe.alerts_queued} gate={gate_pool.stats()}")
    log.info(f"[site] persisted {sink.persisted} confirmed event(s) -> {output_dir}/events.db")
    if getattr(sink, "routing", None) and sink.routing.rules:
        log.info(f"[site] routed={sink.routed} escalated={sink.escalated} "
              f"pending_escalation={sink.escalations.pending_count}")


def notify_channels(spec: str) -> str:
    """Channel names only -- never the credentials inside them.

    A telegram spec carries a bot token and a whatsapp spec a Twilio auth
    token. This banner goes to monitor.log, which cvti/diagnostics.py packs
    into the Diagnose zip a customer emails us, so printing the raw spec would
    put live credentials in our inbox. It only ever said "console" before the
    site config could reach it, which is the only reason it was safe.
    """
    names = []
    for part in (spec or "console").split(","):
        part = part.strip()
        if part:
            names.append(part.split(":", 1)[0])
    return ",".join(names) or "console"


def _notify_for(args) -> str:
    """--notify wins; otherwise the site config decides; otherwise console."""
    if args.notify:
        return args.notify
    try:
        from cvti.serving.onboarding import get_site_meta
        spec = (get_site_meta(args.site_config).get("notify") or "").strip()
        if spec:
            log.info("[site] notify taken from the site config: %s",
                     notify_channels(spec))
            return spec
    except Exception:  # noqa: BLE001 - an unreadable site is not a reason to die
        log.debug("could not read notify from the site config", exc_info=True)
    return "console"


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 8.1 multi-stream pipeline.")
    p.add_argument("--sources", nargs="+", help="Video files / RTSP URLs / webcam indices (demo mode).")
    p.add_argument("--site-config", default="", help="Site JSON: per-camera source + rules + zones.")
    p.add_argument("--weights", default="models/yolov8n.pt")
    p.add_argument("--target-fps", type=float, default=5.0)
    # 24 not 12 (4 Sep): the wall was capped at 12fps by THIS default
    # while every layer above could do more. The decoder still bows to what
    # the machine measurably sustains, and only WATCHED cameras pay for it.
    p.add_argument("--publish-fps", type=float, default=24.0,
                   help="live-wall frame rate, decoupled from detection; 0 = old coupled behavior")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", default="")
    p.add_argument("--half", action="store_true")
    p.add_argument("--seconds", type=float, default=90.0,
                   help="Run cap; file sources stop automatically at end-of-stream.")
    p.add_argument("--gate-provider", default="mock")
    p.add_argument("--gate-model", default="")
    p.add_argument("--gate-base-url", default="")
    p.add_argument("--gate-max-frames", type=int, default=0,
                   help="Most images per verdict the provider accepts (0 = no cap). "
                        "The site's own gate_max_frames wins when set.")
    p.add_argument("--mapper-provider", default="",
                   help="Scene mapper provider; empty inherits the local gate provider.")
    p.add_argument("--mapper-model", default="",
                   help="Scene mapper model; empty inherits the gate model when compatible.")
    p.add_argument("--mapper-base-url", default="",
                   help="Scene mapper API base URL; empty inherits the local gate URL.")
    p.add_argument("--no-memory-guard", action="store_true",
                   help="Do not shed load under memory pressure (may swap).")
    p.add_argument("--memory-warn-gb", type=float, default=2.0)
    p.add_argument("--memory-critical-gb", type=float, default=1.0)
    p.add_argument("--no-publish-frames", action="store_true",
                   help="Do not serve frames to the UI (it will decode streams itself).")
    p.add_argument("--gate-sensitivity", choices=("sensitive", "balanced", "strict"),
                   default="balanced",
                   help="Verification strictness. Measured on held-out clips: balanced = "
                        "recall 89%%/FPR 26%%, strict = recall 78%%/FPR 15%%.")
    p.add_argument("--gate-workers", type=int, default=0,
                   help="Concurrent VLM gate workers; 0 = auto from camera count.")
    p.add_argument("--gate-drain", type=float, default=180.0,
                   help="Seconds to let in-flight verdicts finish after streams end.")
    # No default here on purpose: an unset --notify means "use what the site
    # config says", resolved below. A literal default of "console" silently
    # overrode a site that asked for Telegram, and the only clue was one word
    # in the banner ("notify=console"). The desktop app always passes --notify
    # explicitly so it was never affected, but --site-config claims to describe
    # a whole deployment and did not (20 Sep).
    p.add_argument("--notify", default=None,
                   help="Alert notifier: console | webhook:<url> | telegram:<token>:<chat_id> "
                        "| whatsapp:<sid>:<token>:<from>:<to> | whatsapp (Twilio creds from env). "
                        "Unset: whatever the site config's notify field says.")
    p.add_argument("--security-dir", default="",
                   help="where auth.db/audit.db live — GLOBAL to the install, not "
                        "per-feed (defaults to the output dir for standalone runs)")
    p.add_argument("--mobile-port", type=int, default=8710,
                   help="Port for the phone response view on the site network; 0 disables.")
    p.add_argument("--output-dir", default="runs/serving",
                   help="Where confirmed events + evidence + events.db are written.")
    p.add_argument("--check-clip-encoder", action="store_true",
                   help="Write one synthetic evidence clip, print its codec and exit "
                        "(0 = H.264, playable in the app). CI runs this against the "
                        "shipped binary; nothing else can prove the Windows bundle's "
                        "ffmpeg works.")
    args = p.parse_args()
    if args.check_clip_encoder:
        import tempfile

        from cvti.serving.alert_sink import probe_clip_encoder
        got = probe_clip_encoder(tempfile.mkdtemp(prefix="argus-clip-probe-"))
        print(f"clip encoder: {got['codec']} (ffmpeg: {got['ffmpeg'] or 'not bundled'})")
        if got["codec"] != "h264":
            print("FAIL: evidence clips would not play in the app's player on this machine")
            raise SystemExit(1)
        raise SystemExit(0)

    # Before anything else: a failure during model loading is exactly the kind
    # that used to vanish. Component-scoped because the app runs this as a
    # subprocess pointed at the same output dir.
    from cvti.logging_setup import setup_logging
    setup_logging(args.output_dir, component="argus-engine")

    # Fail fast and readably — a traceback here reads as a crash, and this is a
    # deliberate refusal the operator needs to act on.
    from cvti.verification.gate import MockGateRefused, assert_engine_gate_allowed
    try:
        assert_engine_gate_allowed(args.gate_provider)
    except MockGateRefused as exc:
        log.info(f"[site] {exc}")
        raise SystemExit(2)

    if args.site_config:
        run_site(args.site_config, weights=args.weights, target_fps=args.target_fps,
                 imgsz=args.imgsz, conf=args.conf, device=args.device, half=args.half,
                 seconds=args.seconds, gate_provider=args.gate_provider,
                 gate_sensitivity=args.gate_sensitivity,
                 publish_frames=not args.no_publish_frames,
                 publish_fps=args.publish_fps,
                 memory_guard=not args.no_memory_guard,
                 memory_warn_gb=args.memory_warn_gb,
                 memory_critical_gb=args.memory_critical_gb,
                 mobile_port=args.mobile_port,
                 security_dir=args.security_dir or None,
                 gate_model=args.gate_model, gate_base_url=args.gate_base_url,
                 mapper_provider=args.mapper_provider,
                 mapper_model=args.mapper_model,
                 mapper_base_url=args.mapper_base_url,
                 notify=_notify_for(args), output_dir=args.output_dir,
                 gate_workers=args.gate_workers, gate_drain=args.gate_drain,
                 gate_max_frames=args.gate_max_frames or None)
        return

    if not args.sources:
        raise SystemExit("pass --sources <files...> (demo) or --site-config <site.json>")
    sources = {f"cam{i}": s for i, s in enumerate(args.sources)}
    pipe = MultiStreamPipeline(sources, weights=args.weights, target_fps=args.target_fps,
                               imgsz=args.imgsz, conf=args.conf, device=args.device, half=args.half)
    pipe.start()
    try:
        pipe.run(max_seconds=args.seconds)
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
