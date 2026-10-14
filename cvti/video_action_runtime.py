"""Runtime bridge for hybrid video-action evidence in the live detector."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import cv2

from cvti.contracts import RawEvent
from cvti.video_action_hybrid import predictions_to_events
from cvti.video_action_model import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_VIDEOMAE_MODEL,
    DEFAULT_X3D_MODEL,
    FrameWindow,
    SampledFrame,
    VideoMAEActionModel,
    X3DActionModel,
    sample_evenly_with_indices,
)
from cvti.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class BufferedFrame:
    index: int
    frame: np.ndarray


@dataclass
class _PendingClip:
    submitted_at: float
    run: Callable[[], list[RawEvent]]


class AsyncVideoActionRunner:
    """One shared inference thread for every camera's video-action clips.

    VideoMAE ran INSIDE the per-frame detection path: when its cooldown let it
    fire, one 2–6s transformer forward pass stalled every camera on the site
    (latency audit 1 Sep, D1). The frame loop's only jobs now are O(1): drop a
    prepared clip in this runner's slot and drain finished verdicts.

    One worker on purpose — the model is one shared instance and a second
    concurrent forward pass on the same CPU/GPU only slows the first.

    Latest-clip-wins, per camera: a clip that waited behind a slow inference
    describes a scene that no longer exists, so a newer clip from the same
    camera REPLACES its pending one (counted in `dropped`, never silent).
    Cameras keep one slot each, and the worker always takes the oldest
    submission across cameras, so a busy camera cannot starve a quiet one.
    """

    def __init__(self, name: str = "video-action") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingClip] = {}
        self._results: dict[str, list[RawEvent]] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0      # stale clips replaced before they ran
        self.completed = 0
        self.failed = 0

    def start(self) -> "AsyncVideoActionRunner":
        self._thread = threading.Thread(target=self._loop, name=self.name,
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def submit(self, camera_id: str, run: Callable[[], list[RawEvent]]) -> None:
        """Queue a clip for analysis. Never blocks; replaces this camera's
        pending clip if the worker hasn't reached it yet."""
        with self._lock:
            if camera_id in self._pending:
                self.dropped += 1
            self._pending[camera_id] = _PendingClip(time.monotonic(), run)
        self._wake.set()

    def drain(self, camera_id: str) -> list[RawEvent]:
        """Finished verdicts for this camera since the last drain."""
        with self._lock:
            return self._results.pop(camera_id, [])

    def _next_job(self) -> tuple[str, _PendingClip] | None:
        with self._lock:
            if not self._pending:
                return None
            camera_id = min(self._pending, key=lambda c: self._pending[c].submitted_at)
            return camera_id, self._pending.pop(camera_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self._next_job()
            if job is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            camera_id, clip = job
            try:
                events = clip.run()
            except Exception:  # noqa: BLE001 - one bad clip must not kill the worker
                self.failed += 1
                log.warning("[%s] analysis failed for %s", self.name, camera_id,
                            exc_info=True)
                continue
            with self._lock:
                self._results.setdefault(camera_id, []).extend(events)
                self.completed += 1


class VideoActionRuntime:
    """Keeps recent frames and runs a weak video classifier around trigger frames."""

    def __init__(
        self,
        *,
        model: Any,
        backend: str,
        model_name: str,
        fps: float,
        window_seconds: float = 4.0,
        frame_count: int = DEFAULT_FRAME_COUNT,
        top_k: int = 5,
        cooldown_seconds: float = 2.0,
        buffer_short_side: int = 256,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.backend = backend
        self.model_name = model_name
        self.fps = fps if fps > 0 else 30.0
        self.window_seconds = window_seconds
        self.frame_count = frame_count
        self.top_k = top_k
        self.cooldown_seconds = cooldown_seconds
        # Buffer at the size the model actually consumes (audit 1 Sep, D4):
        # VideoMAE's processor works from 224px crops and X3D from 256 — yet
        # this buffer held full decoded frames, ~250 MB per 1080p camera, all
        # of it resized away at inference. Shrink on the way IN: the resize
        # cost moves to one small op per frame, and the buffer drops to ~14 MB.
        self.buffer_short_side = buffer_short_side
        self.verbose = verbose
        self._last_analysis_timestamp = -1_000_000.0
        buffer_size = max(frame_count, int(round(self.fps * window_seconds * 2)) + frame_count)
        self._frames: deque[BufferedFrame] = deque(maxlen=buffer_size)

    def _shrink(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        short = min(height, width)
        if short <= self.buffer_short_side:
            return frame
        scale = self.buffer_short_side / float(short)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        # INTER_LINEAR, not INTER_AREA: at this shrink factor AREA costs
        # ~6.4 ms per 1080p frame on the detection hot path where LINEAR
        # costs 0.16 ms (measured 3 Sep) — and LINEAR is the exact filter
        # the model's own preprocessing applied to these frames anyway, so
        # the classifier sees what it always saw.
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

    def add_frame(self, frame: np.ndarray, *, frame_index: int) -> None:
        # Shrink BEFORE the colour conversion so cvtColor also runs on the
        # small frame. Store RGB because VideoMAE/X3D wrappers expect RGB.
        small = self._shrink(frame)
        self._frames.append(BufferedFrame(index=frame_index,
                                          frame=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))

    def prepare_analysis(self, *, center_frame_index: int,
                         timestamp: float) -> "Callable[[], list[RawEvent]] | None":
        """Snapshot the clip window NOW; return the heavy part as a callable.

        The split is the whole point (audit 1 Sep, D1): the frame loop pays
        only for sampling ~16 frame references out of the buffer, and hands
        the returned callable — the actual model forward pass — to the
        AsyncVideoActionRunner. The snapshot must happen here, not in the
        worker: the buffer keeps mutating under the frame loop, and the clip
        must be the frames around THIS trigger. The cooldown stamp also moves
        here, to submission time — it throttles how often clips are queued,
        not how often the model finishes.

        Returns None when inside the cooldown window or nothing is buffered.
        The sync path (analyze_event) runs the same callable inline.
        """
        if timestamp - self._last_analysis_timestamp < self.cooldown_seconds:
            return None
        if not self._frames:
            return None

        window = self._build_buffered_window(center_frame_index)
        self._last_analysis_timestamp = timestamp

        def run() -> list[RawEvent]:
            predictions = self.model.predict_frames(
                [item.frame for item in window.sampled], top_k=self.top_k)
            if self.verbose:
                top = predictions[0] if predictions else None
                if top is not None:
                    log.info(f"[VideoAction] {self.backend}:{self.model_name} "
                             f"{window.start_index}..{window.end_index} "
                             f"top={top.label} ({top.confidence:.3f})")
            return predictions_to_events(
                predictions,
                backend=self.backend,
                model_name=self.model_name,
                window_name=window.name,
                sampled_frame_indices=[item.index for item in window.sampled],
                timestamp=timestamp,
            )

        return run

    def analyze_event(self, *, center_frame_index: int, timestamp: float) -> list[RawEvent]:
        """Synchronous analysis — the single-stream CLI path. The engine uses
        prepare_analysis + AsyncVideoActionRunner instead."""
        clip = self.prepare_analysis(center_frame_index=center_frame_index,
                                     timestamp=timestamp)
        return clip() if clip is not None else []

    def _build_buffered_window(self, center_frame_index: int) -> FrameWindow:
        radius = max(0, int(round((self.window_seconds * self.fps) / 2)))
        start = center_frame_index - radius
        end = center_frame_index + radius

        candidates = [item for item in self._frames if start <= item.index <= end]
        if not candidates:
            candidates = list(self._frames)

        sampled_local = sample_evenly_with_indices([item.frame for item in candidates], count=self.frame_count)
        sampled = [
            SampledFrame(index=candidates[item.index].index, frame=item.frame)
            for item in sampled_local
        ]
        return FrameWindow(
            name="event",
            start_index=candidates[0].index,
            end_index=candidates[-1].index,
            sampled=sampled,
        )


def build_video_action_runtime(
    *,
    backend: str,
    model_name: str,
    fps: float,
    window_seconds: float,
    frame_count: int,
    top_k: int,
    cooldown_seconds: float,
    device: str | None,
    verbose: bool,
) -> VideoActionRuntime:
    if backend == "videomae":
        resolved_model = model_name or DEFAULT_VIDEOMAE_MODEL
        model = VideoMAEActionModel(resolved_model, device=device, frame_count=frame_count, verbose=verbose)
    elif backend == "x3d":
        resolved_model = model_name or DEFAULT_X3D_MODEL
        model = X3DActionModel(resolved_model, device=device, frame_count=frame_count, verbose=verbose)
    else:
        raise ValueError(f"Unsupported video action backend: {backend}")

    return VideoActionRuntime(
        model=model,
        backend=backend,
        model_name=resolved_model,
        fps=fps,
        window_seconds=window_seconds,
        frame_count=frame_count,
        top_k=top_k,
        cooldown_seconds=cooldown_seconds,
        verbose=verbose,
    )
