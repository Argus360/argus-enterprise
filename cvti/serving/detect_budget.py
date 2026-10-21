"""Adaptive detection-rate budgeting for the shared batched detector.

The detector samples each camera at a target fps. A fixed target (the `8` the
CHI pilot shipped) wastes a strong GPU when scenes are quiet and can overload a
weak one when every camera is busy at once. This module turns that fixed rate
into a closed loop:

  * a controller watches how much of REAL time the detector actually spends
    doing inference — its "busy fraction" — and grows or shrinks a shared fps
    budget to hold that near a target (e.g. 70%, leaving headroom for bursts);
  * the budget is then split across cameras by how ACTIVE each one is — a camera
    seeing people/motion gets more fps, a quiet one keeps a guaranteed floor and
    no camera exceeds a ceiling.

Everything here is pure and side-effect free so it is unit-testable; the pipeline
owns the clock, the GPU timing and the activity counts, and calls `update()`
once per window. Mirrors the shape of wall_budget.allocate_wall_fps, which does
the same job for the live-view wall.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectCamera:
    """One camera the detector is sampling, with a recent activity weight.

    `activity` is any non-negative number that rises with how much is happening
    in the scene — the pipeline passes recent detections/second. Only the
    relative sizes matter; all-equal (including all-zero) splits the surplus
    evenly."""
    camera_id: str
    activity: float = 0.0


def allocate_detection_fps(cameras: list[DetectCamera], *, budget_fps: float,
                           floor_fps: float, ceiling_fps: float) -> dict[str, float]:
    """Split a shared detection-fps budget across cameras.

    Rules, in order:
      1. every camera gets at least `floor_fps` (the guarantee — nothing is ever
         under-sampled), capped so the floor itself never exceeds the ceiling;
      2. whatever budget remains after the floors is shared in proportion to
         `activity` — busy cameras pull more; an all-quiet site splits evenly;
      3. no camera exceeds `ceiling_fps`, and budget a capped camera cannot use
         is handed back to the others (water-filling).
    """
    n = len(cameras)
    if n == 0:
        return {}
    floor = max(0.0, float(floor_fps))
    ceiling = max(floor, float(ceiling_fps))
    rates = {c.camera_id: floor for c in cameras}
    remaining = max(0.0, float(budget_fps) - floor * n)

    open_cams = [c for c in cameras if ceiling - rates[c.camera_id] > 1e-9]
    # At most n passes: each pass either exhausts the budget or saturates at
    # least one more camera at the ceiling, so it always terminates.
    for _ in range(n):
        if remaining <= 1e-9 or not open_cams:
            break
        total_w = sum(max(0.0, c.activity) for c in open_cams)
        if total_w <= 0:
            share = {c.camera_id: remaining / len(open_cams) for c in open_cams}
        else:
            share = {c.camera_id: remaining * (max(0.0, c.activity) / total_w)
                     for c in open_cams}
        given = 0.0
        for c in open_cams:
            room = ceiling - rates[c.camera_id]
            add = min(share[c.camera_id], room)
            rates[c.camera_id] += add
            given += add
        remaining -= given
        open_cams = [c for c in open_cams if ceiling - rates[c.camera_id] > 1e-9]
        if given <= 1e-9:
            break
    return rates


@dataclass
class AdaptiveConfig:
    """Operator-facing knobs, read from the site file's `adaptive_fps` block.

    Defaults are tuned for a single modern GPU (e.g. a T4): a 6 fps floor so no
    camera is ever starved, a high ceiling so the real upper bound is GPU
    headroom and the stream's own frame rate rather than a fixed number (the
    controller holds GPU use near target_busy, so it self-limits long before the
    ceiling), and a 70% busy target that keeps the GPU well-fed without
    saturating. The ceiling only matters as a guard in busy multi-camera sites;
    lower it there if one camera should never monopolise."""
    enabled: bool = False
    floor_fps: float = 6.0
    ceiling_fps: float = 120.0
    target_busy: float = 0.70
    window_s: float = 2.0
    # Largest multiplicative change to the budget per window — damps oscillation
    # when the busy fraction swings (a burst of motion, a batch that stalled).
    max_step: float = 1.5

    @classmethod
    def from_site(cls, raw: object) -> "AdaptiveConfig":
        """Build from the site file. Accepts a bool (`true` = defaults on) or a
        dict of overrides; anything else, or absent, stays disabled."""
        if raw is True:
            return cls(enabled=True)
        if not isinstance(raw, dict):
            return cls(enabled=False)
        if not raw.get("enabled", True):
            return cls(enabled=False)
        base = cls(enabled=True)
        return cls(
            enabled=True,
            floor_fps=_num(raw.get("floor_fps"), base.floor_fps),
            ceiling_fps=_num(raw.get("ceiling_fps"), base.ceiling_fps),
            target_busy=_clamp(_num(raw.get("target_busy"), base.target_busy), 0.1, 0.95),
            window_s=max(0.5, _num(raw.get("window_s"), base.window_s)),
            max_step=max(1.05, _num(raw.get("max_step"), base.max_step)),
        )


@dataclass
class DetectionRateController:
    """Holds the shared budget between windows and re-allocates each window."""
    cfg: AdaptiveConfig
    budget_fps: float
    _rates: dict[str, float] = field(default_factory=dict)

    def update(self, *, busy_fraction: float,
               cameras: list[DetectCamera]) -> dict[str, float]:
        """Advance one window and return the new per-camera fps.

        `busy_fraction` is detection time / real time over the last window. Above
        target we shrink the budget, below target we grow it; the change is
        clamped to `max_step` each way, and the budget is bounded so the floors
        and ceiling always hold for the current camera count."""
        n = max(1, len(cameras))
        bf = max(1e-3, float(busy_fraction))
        factor = _clamp(self.cfg.target_busy / bf, 1.0 / self.cfg.max_step, self.cfg.max_step)
        self.budget_fps = _clamp(self.budget_fps * factor,
                                 self.cfg.floor_fps * n, self.cfg.ceiling_fps * n)
        self._rates = allocate_detection_fps(
            cameras, budget_fps=self.budget_fps,
            floor_fps=self.cfg.floor_fps, ceiling_fps=self.cfg.ceiling_fps)
        return self._rates


def _num(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
