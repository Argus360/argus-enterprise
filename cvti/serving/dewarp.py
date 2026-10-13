"""Dewarp fisheye frames before anything else sees them.

Seven of the twelve cameras at the CHI pilot are fisheye (survey, 13 Oct
2026): three ceiling-mounted 360° lenses streaming a full circle at
2592x1944, four wall-mounted ones. The detector was trained on ordinary
lenses, zones drawn on a warped floor are curves, and the verifier is asked
to judge a distorted scene. So a camera may carry a `dewarp` block in the
site file and every consumer — detection, the wall, the zone editor's
snapshot, the verifier's crops — gets the flattened frame instead:

    "dewarp": {"mode": "panorama"}                       # ceiling 360: unwrap to a strip
    "dewarp": {"mode": "perspective", "fov": 110,
               "yaw": 0, "pitch": 15}                     # wall mount: flat view of the centre
    optional: "center": [cx, cy], "radius": r             # else found from the first frame
              "output": [w, h], "inner": 0.08, "seam": 0  # panorama: hole and seam angle

One remap per frame from a precomputed map: a few milliseconds at the
default output sizes, cheaper than decoding the source frame.
"""
from __future__ import annotations

import math
from typing import Any

from cvti.logging_setup import get_logger

log = get_logger(__name__)

MODES = ("off", "panorama", "perspective")
PANORAMA_OUTPUT = (1600, 500)
PERSPECTIVE_OUTPUT = (1280, 800)
# Pixels darker than this (0-255 grey) count as the black border around the
# image circle when the circle has to be found from a frame.
_BORDER_LEVEL = 18


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"dewarp: {name} must be a number between {low} and {high}")
    return float(value)


class Dewarp:
    """A per-camera dewarp. Build with from_config; call apply(frame)."""

    def __init__(self, mode: str = "off", *, center=None, radius=None, output=None,
                 fov: float = 110.0, yaw: float = 0.0, pitch: float = 0.0,
                 inner: float = 0.08, seam: float = 0.0) -> None:
        if mode not in MODES:
            raise ValueError(f"dewarp: mode must be one of {', '.join(MODES)}")
        self.mode = mode
        self.center = None
        if center is not None:
            if not (isinstance(center, (list, tuple)) and len(center) == 2):
                raise ValueError("dewarp: center must be [x, y]")
            self.center = (_number(center[0], "center x", 0, 100000), _number(center[1], "center y", 0, 100000))
        self.radius = _number(radius, "radius", 1, 100000) if radius is not None else None
        default = PANORAMA_OUTPUT if mode == "panorama" else PERSPECTIVE_OUTPUT
        if output is None:
            output = default
        if not (isinstance(output, (list, tuple)) and len(output) == 2):
            raise ValueError("dewarp: output must be [width, height]")
        self.output = (int(_number(output[0], "output width", 64, 4096)),
                       int(_number(output[1], "output height", 64, 4096)))
        self.fov = _number(fov, "fov", 30, 170)
        self.yaw = _number(yaw, "yaw", -180, 180)
        self.pitch = _number(pitch, "pitch", -90, 90)
        self.inner = _number(inner, "inner", 0, 0.5)
        self.seam = _number(seam, "seam", -360, 360)
        self.auto_circle = self.center is None or self.radius is None
        self._maps = None            # (mx, my) for the current source size
        self._source_size = None     # (w, h) the maps were built for

    # --- construction -------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Any) -> "Dewarp | None":
        """None for no dewarp (missing, empty, or mode off); raises ValueError
        on a malformed block so a typo is a start-up error, not a silent no-op."""
        if not cfg:
            return None
        if not isinstance(cfg, dict):
            raise ValueError("dewarp: must be an object")
        unknown = set(cfg) - {"mode", "center", "radius", "output", "fov", "yaw", "pitch", "inner", "seam"}
        if unknown:
            raise ValueError(f"dewarp: unknown setting(s) {', '.join(sorted(unknown))}")
        mode = str(cfg.get("mode") or "off")
        if mode == "off":
            return None
        return cls(mode, center=cfg.get("center"), radius=cfg.get("radius"), output=cfg.get("output"),
                   fov=cfg.get("fov", 110.0), yaw=cfg.get("yaw", 0.0), pitch=cfg.get("pitch", 0.0),
                   inner=cfg.get("inner", 0.08), seam=cfg.get("seam", 0.0))

    def describe(self) -> dict:
        return {"mode": self.mode, "center": self.center, "radius": self.radius,
                "output": list(self.output), "auto_circle": self.auto_circle}

    # --- the circle ---------------------------------------------------------
    @staticmethod
    def find_circle(frame) -> tuple[float, float, float] | None:
        """Centre and radius of the bright image circle inside the dark border,
        or None when the frame has no such border (not a fisheye frame)."""
        import cv2
        import numpy as np
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        grey = cv2.GaussianBlur(grey, (0, 0), 3)
        mask = (grey > _BORDER_LEVEL).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count < 2:
            return None
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        h, w = grey.shape[:2]
        if stats[biggest, cv2.CC_STAT_AREA] > 0.95 * w * h:
            return None                     # no border: the whole frame is image
        ys, xs = np.where(labels == biggest)
        cx, cy = float(xs.mean()), float(ys.mean())
        r = float(max(xs.max() - cx, cx - xs.min(), ys.max() - cy, cy - ys.min()))
        return cx, cy, r

    def _settle_circle(self, frame) -> None:
        if not self.auto_circle:
            return
        h, w = frame.shape[:2]
        found = self.find_circle(frame)
        if found is None:
            log.warning("[dewarp] no image circle found in a %dx%d frame; assuming the "
                        "lens fills the frame", w, h)
            found = (w / 2.0, h / 2.0, min(w, h) / 2.0)
        cx, cy, r = found
        self.center = self.center or (cx, cy)
        self.radius = self.radius or r
        log.info("[dewarp] image circle centre=(%.0f, %.0f) radius=%.0f in %dx%d",
                 self.center[0], self.center[1], self.radius, w, h)

    # --- the maps -----------------------------------------------------------
    def _build_maps(self) -> None:
        import numpy as np
        cx, cy = self.center
        r = self.radius
        out_w, out_h = self.output
        if self.mode == "panorama":
            # Polar unwrap: columns are angle (a full turn), rows run from the
            # rim (walls, top) to the hole at the centre (bottom).
            start = math.radians(self.seam)
            theta = np.linspace(start, start + 2 * math.pi, out_w, endpoint=False, dtype=np.float32)
            rho = np.linspace(r, r * self.inner, out_h, dtype=np.float32)
            T, R = np.meshgrid(theta, rho)
            mx = cx + R * np.cos(T)
            my = cy + R * np.sin(T)
        else:
            # Equidistant fisheye (r = f * theta, 180 degree lens) projected
            # onto a flat view of `fov` degrees, turned by yaw and pitch.
            f_fish = r / (math.pi / 2)
            f_out = (out_w / 2) / math.tan(math.radians(self.fov) / 2)
            xs = (np.arange(out_w, dtype=np.float32) - out_w / 2) / f_out
            ys = (np.arange(out_h, dtype=np.float32) - out_h / 2) / f_out
            X, Y = np.meshgrid(xs, ys)
            Z = np.ones_like(X)
            cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
            cyw, syw = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
            Y2 = Y * cp - Z * sp
            Z2 = Y * sp + Z * cp
            X3 = X * cyw + Z2 * syw
            Z3 = -X * syw + Z2 * cyw
            theta = np.arctan2(np.sqrt(X3 ** 2 + Y2 ** 2), Z3)
            phi = np.arctan2(Y2, X3)
            rr = f_fish * theta
            mx = cx + rr * np.cos(phi)
            my = cy + rr * np.sin(phi)
        self._maps = (mx.astype(np.float32), my.astype(np.float32))

    # --- per frame ----------------------------------------------------------
    def apply(self, frame):
        """The dewarped frame. Maps are built on the first frame (and again if
        the source size changes); the circle is found then when not given."""
        import cv2
        h, w = frame.shape[:2]
        if self._maps is None or self._source_size != (w, h):
            if self._source_size is not None and self.auto_circle:
                self.center = None          # a new source size: find the circle again
                self.radius = None
            self._settle_circle(frame)
            self._build_maps()
            self._source_size = (w, h)
        mx, my = self._maps
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def dewarps_for(cameras) -> dict[str, Dewarp]:
    """camera_id -> Dewarp for every camera in a site list that asks for one."""
    out = {}
    for cam in cameras or []:
        cfg = cam.get("dewarp") if isinstance(cam, dict) else None
        if not cfg:
            continue
        try:
            dw = Dewarp.from_config(cfg)
        except ValueError as exc:
            raise ValueError(f"camera {cam.get('id')}: {exc}") from exc
        if dw is not None:
            out[str(cam.get("id"))] = dw
    return out
