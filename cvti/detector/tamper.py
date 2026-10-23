"""Camera tamper / block detection — pure CV, no model, near-zero cost.

A blocked or sabotaged camera is a threat in itself (covering the lens is a
common first move). Two cheap signals catch the common cases:

  • blackout  — mean brightness collapses (lens covered, spray-painted, bag over
                it, or the feed cut to black)
  • obscured  — image sharpness collapses while it's still bright (defocused,
                smeared, fogged, or a translucent cover)

We learn a slow rolling baseline of brightness + sharpness while the scene looks
normal, and fire when the current value drops well below that baseline for a few
consecutive frames. Baseline only adapts when normal, so gradual lighting drift
doesn't false-fire but a sudden cover does. One alert per episode (latched until
the view recovers).
"""
from __future__ import annotations


class TamperDetector:
    def __init__(self, *, warmup: int = 12, drop_ratio: float = 0.4,
                 dark_floor: float = 45.0, min_frames: int = 6, ema: float = 0.03) -> None:
        self.warmup = warmup
        self.drop_ratio = drop_ratio      # fire below this fraction of baseline
        self.dark_floor = dark_floor      # absolute brightness that always counts as dark
        self.min_frames = min_frames      # consecutive suspicious frames before firing
        self.ema = ema                    # baseline adaptation rate (normal frames only)
        self._n = 0
        self._bright: float | None = None
        self._sharp: float | None = None
        self._streak = 0
        self._firing = False

    MEASURE_WIDTH = 320

    @classmethod
    def _measure(cls, frame_bgr) -> tuple[float, float]:
        import cv2
        # Brightness and sharpness are compared to this detector's own baseline,
        # measured the same way, so a 320px thumbnail answers as well as a 4MP
        # frame at a fraction of the cost (a Laplacian over 1600x938 every frame
        # was 12% of the CHI engine's main thread).
        h, w = frame_bgr.shape[:2]
        if w > cls.MEASURE_WIDTH:
            frame_bgr = cv2.resize(frame_bgr, (cls.MEASURE_WIDTH, max(1, h * cls.MEASURE_WIDTH // w)),
                                   interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return brightness, sharpness

    def update(self, frame_bgr) -> dict | None:
        """Feed one frame. Returns a tamper dict once per episode, else None."""
        bright, sharp = self._measure(frame_bgr)
        self._n += 1
        if self._bright is None:
            self._bright, self._sharp = bright, sharp
            return None

        blackout = bright < self.drop_ratio * self._bright and bright < self.dark_floor
        obscured = self._n > self.warmup and sharp < self.drop_ratio * max(self._sharp, 1.0)
        suspicious = blackout or obscured

        if self._n <= self.warmup:                    # still learning the baseline
            self._bright += self.ema * (bright - self._bright)
            self._sharp += self.ema * (sharp - self._sharp)
            return None

        if suspicious:
            self._streak += 1
            if self._streak >= self.min_frames and not self._firing:
                self._firing = True
                kind = "blackout" if blackout else "obscured"
                return {"kind": kind, "brightness": round(bright, 1), "sharpness": round(sharp, 1),
                        "baseline_brightness": round(self._bright, 1),
                        "baseline_sharpness": round(self._sharp, 1)}
            return None

        # normal view: reset the latch and adapt the baseline slowly
        self._streak = 0
        self._firing = False
        self._bright += self.ema * (bright - self._bright)
        self._sharp += self.ema * (sharp - self._sharp)
        return None
