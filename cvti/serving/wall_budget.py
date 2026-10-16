"""The wall's decode budget, shared by the cameras people are watching.

A watched tile makes its camera decode at the wall rate. v1.10.7 capped that
rate per camera with an 8 fps floor, which Tolu's audit (15 Oct 2026) showed
is not a budget: fifteen open tiles at the floor cost 120 fps, over the
nominal 96, and four tiles could still ask 24 each.

Tolu's review of the first version of this module added three rules:

1. Cost is CPU time, never wall time. A decoder at the live edge WAITS for
   its camera, and lowering its rate lengthens the wait, so a wall-time
   number feeds back on itself. StreamDecoder.frame_cpu_ms (decode plus
   dewarp, measured with time.thread_time) is what this module takes.
2. Detection comes first. An analysed camera decodes at its detection rate
   whatever the wall says (the decoder uses max(target, display)), so that
   baseline is RESERVED out of the budget before any wall work is granted,
   and the allocation reports when the baseline alone does not fit instead
   of pretending.
3. The floor never beats the ceiling. A tile is never granted more than
   the configured wall rate, even when the floor is higher.
"""
from __future__ import annotations

from dataclasses import dataclass, field

WALL_DECODE_BUDGET_MS = 1000.0     # decode milliseconds per second: one core
MIN_WALL_FPS = 2.0                 # a watched tile never drops below this (ceiling permitting)
_ESTIMATE_MS_PER_MEGAPIXEL = 10.0  # software decode prior before measurements


@dataclass(frozen=True)
class WallCamera:
    camera_id: str
    watched: bool
    cost_ms: float | None          # measured CPU cost of one decoded frame (frame_cpu_ms)
    width: int = 0
    height: int = 0
    keyframes_only: bool = False   # decoded by ffmpeg out of process: ~free here
    baseline_fps: float = 0.0      # detection rate: decoded regardless of the wall


@dataclass
class WallAllocation:
    rates: dict[str, float] = field(default_factory=dict)   # display fps per camera; 0 = no boost
    budget_ms: float = 0.0
    reserved_ms: float = 0.0       # detection baseline: sum(baseline_fps * cost) over all cameras
    wall_ms: float = 0.0           # decode granted to watched tiles ABOVE their baseline
    over_budget_ms: float = 0.0    # by how much reserved + wall exceeds the budget; 0 when it fits
    watched: int = 0

    @property
    def fits(self) -> bool:
        return self.over_budget_ms <= 0.0

    def summary(self) -> dict:
        return {"budget_ms": round(self.budget_ms, 1), "reserved_ms": round(self.reserved_ms, 1),
                "wall_ms": round(self.wall_ms, 1), "over_budget_ms": round(self.over_budget_ms, 1),
                "watched": self.watched}


def estimate_cost_ms(camera: WallCamera) -> float:
    """What one frame of this camera costs, in decode CPU milliseconds."""
    if camera.keyframes_only:
        return 1.0                 # a pipe read and a copy
    if camera.cost_ms is not None and camera.cost_ms > 0:
        return float(camera.cost_ms)
    megapixels = (camera.width * camera.height) / 1_000_000.0
    return max(1.0, megapixels * _ESTIMATE_MS_PER_MEGAPIXEL)


def allocate_wall_fps(cameras: list[WallCamera], *, ceiling_fps: float,
                      budget_ms: float = WALL_DECODE_BUDGET_MS,
                      min_fps: float = MIN_WALL_FPS,
                      budgeted: bool = True) -> WallAllocation:
    """Display fps per camera for this tick, with the budget accounting.

    Unwatched cameras get 0 (their decoder runs at its detection baseline).
    Watched cameras get at least min(ceiling, min_fps) and at most the
    ceiling; between those, the budget left after the detection baseline is
    split equally and converted by each camera's own frame cost. A cheap
    decode (hardware, keyframe-only, small substream) reaches the ceiling by
    itself, so the budget is applied on every box; `budgeted=False` is the
    old unconditional ceiling, kept for comparison."""
    alloc = WallAllocation(rates={c.camera_id: 0.0 for c in cameras}, budget_ms=float(budget_ms))
    ceiling = max(0.0, float(ceiling_fps))
    costs = {c.camera_id: estimate_cost_ms(c) for c in cameras}
    alloc.reserved_ms = sum(max(0.0, c.baseline_fps) * costs[c.camera_id] for c in cameras)
    watched = [c for c in cameras if c.watched]
    alloc.watched = len(watched)
    if not watched or ceiling <= 0:
        alloc.over_budget_ms = max(0.0, alloc.reserved_ms - alloc.budget_ms)
        return alloc
    if not budgeted:
        for c in watched:
            alloc.rates[c.camera_id] = ceiling
        alloc.wall_ms = sum(max(0.0, ceiling - c.baseline_fps) * costs[c.camera_id] for c in watched)
        alloc.over_budget_ms = max(0.0, alloc.reserved_ms + alloc.wall_ms - alloc.budget_ms)
        return alloc
    floor = min(ceiling, float(min_fps))                       # rule 3
    remaining = max(0.0, alloc.budget_ms - alloc.reserved_ms)   # rule 2
    share_ms = remaining / len(watched)
    for c in watched:
        cost = costs[c.camera_id]
        affordable_extra_fps = share_ms / cost
        wanted = max(floor, c.baseline_fps + affordable_extra_fps)
        alloc.rates[c.camera_id] = min(ceiling, wanted)
    alloc.wall_ms = sum(max(0.0, alloc.rates[c.camera_id] - c.baseline_fps) * costs[c.camera_id]
                        for c in watched)
    alloc.over_budget_ms = max(0.0, alloc.reserved_ms + alloc.wall_ms - alloc.budget_ms)
    return alloc
