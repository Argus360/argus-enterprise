"""The wall's decode budget, shared by the cameras people are watching.

A watched tile makes its camera decode at the wall rate. v1.10.7 capped that
rate at a fixed number per camera with an 8 fps floor, which Tolu's audit
(15 Oct 2026) showed is not a budget: fifteen open tiles at the floor cost
120 fps, over the nominal 96, and four tiles could still ask 24 each.

This replaces the floor with a budget in decode time: a CPU-only box gives
the wall WALL_DECODE_BUDGET_MS of decode per wall second (one core by
default), split equally across the cameras that have a viewer right now.
Each camera's share is turned into a frame rate by what one of ITS frames
measurably costs to decode (the decoder's own busy_ms, or a size-based
estimate before the first frames), then clamped between MIN_WALL_FPS and
the wall's ceiling. Cheap cameras (recorder substreams, keyframe-only
decode) get the ceiling; expensive ones get what fits. Nobody watching
means no decode boost at all, as before. GPU boxes keep the ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

WALL_DECODE_BUDGET_MS = 1000.0     # decode milliseconds per second: one core
MIN_WALL_FPS = 2.0                 # a watched tile never drops below this
_ESTIMATE_MS_PER_MEGAPIXEL = 10.0  # software decode prior before measurements


@dataclass(frozen=True)
class WallCamera:
    camera_id: str
    watched: bool
    busy_ms: float | None          # measured decode cost of one kept frame
    width: int = 0
    height: int = 0
    keyframes_only: bool = False   # decoded by ffmpeg out of process: ~free here


def estimate_cost_ms(camera: WallCamera) -> float:
    """What one frame of this camera costs the wall, in decode milliseconds."""
    if camera.keyframes_only:
        return 1.0                 # a pipe read and a copy
    if camera.busy_ms is not None and camera.busy_ms > 0:
        return float(camera.busy_ms)
    megapixels = (camera.width * camera.height) / 1_000_000.0
    return max(1.0, megapixels * _ESTIMATE_MS_PER_MEGAPIXEL)


def allocate_wall_fps(cameras: list[WallCamera], *, ceiling_fps: float,
                      cpu_only: bool, budget_ms: float = WALL_DECODE_BUDGET_MS,
                      min_fps: float = MIN_WALL_FPS) -> dict[str, float]:
    """camera_id -> display fps for this tick. Unwatched cameras get 0."""
    out = {c.camera_id: 0.0 for c in cameras}
    watched = [c for c in cameras if c.watched]
    if not watched:
        return out
    if not cpu_only or ceiling_fps <= 0:
        for c in watched:
            out[c.camera_id] = float(ceiling_fps)
        return out
    share_ms = budget_ms / len(watched)
    for c in watched:
        affordable = share_ms / estimate_cost_ms(c)
        out[c.camera_id] = max(min_fps, min(float(ceiling_fps), affordable))
    return out
