"""Fall / person-collapsed detection — cheap CV on tracked person boxes, no model.

A person lying on the ground (medical emergency, assault aftermath, faint) reads as
a person box that goes *horizontal* (wider than tall) and stays that way. A brief
bend or crouch shouldn't fire, so we require the horizontal posture to persist for
several consecutive frames on the same track. One alert per track per episode
(latched until they stand back up), with a cooldown so a persistent body on the
floor doesn't re-fire endlessly.

This is a trigger; TrueSight confirms whether the person is actually collapsed vs.
sitting/crouching/lying down normally.
"""
from __future__ import annotations


class FallDetector:
    def __init__(
        self,
        *,
        down_aspect: float = 1.15,    # w/h at/above this = horizontal (fallen)
        up_aspect: float = 0.85,      # w/h at/below this = upright (reset latch)
        min_frames: int = 4,          # consecutive down frames before firing
        min_box_frac: float = 0.004,  # ignore tiny/far boxes (noise)
        cooldown_seconds: float = 90.0,
        # A fall is a TRANSITION: the same person was upright and went down
        # fast. Someone already lying, crouching, bending over a pallet or seen
        # foreshortened from above is not a fall (CHI, 23 Oct 2026: "laying
        # down doesn't mean the person is falling"). The track must have been
        # seen upright within `upright_within_s` of its first down frame.
        # 0 disables the transition requirement (old static behaviour).
        upright_within_s: float = 2.5,
    ) -> None:
        self.down_aspect = down_aspect
        self.up_aspect = up_aspect
        self.min_frames = min_frames
        self.min_box_frac = min_box_frac
        self.cooldown = cooldown_seconds
        self.upright_within_s = upright_within_s
        self._streak: dict = {}       # track_id -> consecutive down frames
        self._fired: dict = {}        # track_id -> last-fired timestamp
        self._last_upright: dict = {} # track_id -> timestamp last seen clearly upright
        self._down_since: dict = {}   # track_id -> timestamp of the first down frame

    def update(self, person_boxes, frame_area: float, timestamp: float = 0.0) -> dict | None:
        """`person_boxes`: list of (track_id, x1, y1, x2, y2). Returns a fall dict or None."""
        seen = set()
        result = None
        for tid, x1, y1, x2, y2 in person_boxes:
            seen.add(tid)
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            if (w * h) / max(frame_area, 1.0) < self.min_box_frac:
                continue
            aspect = w / h
            if aspect >= self.down_aspect:
                if self._streak.get(tid, 0) == 0:
                    self._down_since[tid] = timestamp
                self._streak[tid] = self._streak.get(tid, 0) + 1
                last = self._fired.get(tid, -1e9)
                fell = (self.upright_within_s <= 0 or
                        self._down_since.get(tid, timestamp) - self._last_upright.get(tid, -1e9)
                        <= self.upright_within_s)
                if (self._streak[tid] >= self.min_frames and fell
                        and timestamp - last >= self.cooldown):
                    self._fired[tid] = timestamp
                    if result is None:      # one candidate per call; prefer the first
                        result = {"kind": "fall", "track_id": int(tid), "aspect": round(aspect, 2)}
            elif aspect <= self.up_aspect:
                self._streak[tid] = 0        # clearly upright → reset latch
                self._last_upright[tid] = timestamp
                self._down_since.pop(tid, None)
        # forget tracks that vanished — INCLUDING their fired latches: ids never
        # return, so a latch for a departed track is a pure leak. (Audit #4.)
        for tid in [t for t in self._streak if t not in seen]:
            self._streak.pop(tid, None)
        for tid in [t for t in self._fired if t not in seen]:
            self._fired.pop(tid, None)
        for store in (self._last_upright, self._down_since):
            for tid in [t for t in store if t not in seen]:
                store.pop(tid, None)
        return result
