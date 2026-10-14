"""Per-camera decode thread with latest-frame-drop + FPS governor.

Each camera gets one background thread that reads its RTSP/file/webcam source
and keeps only the MOST RECENT frame. If inference falls behind, stale frames
are discarded rather than queued — so the box stays near real-time instead of
drifting further behind on every camera.

A target-FPS governor throttles the decode loop so a 30 FPS source does not
burn CPU decoding frames the pipeline will only sample at ~5 FPS.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from cvti.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Frame:
    camera_id: str
    image: Any          # numpy BGR array
    index: int
    timestamp: float    # seconds since decoder start


# An offline camera produces no alerts — which looks exactly like a camera
# watching a quiet corridor. The tamper detector answers "did someone cover
# this?"; nothing answered "has this been unreachable since Tuesday?". In a
# security product false confidence is worse than visible failure, so the link
# state is explicit and reported rather than inferred from silence.
CONNECTED = "connected"
RECONNECTING = "reconnecting"
OFFLINE = "offline"

# How long a camera may be reconnecting before we call it offline and tell
# someone. Short enough to catch a real outage, long enough that a camera
# reboot or a network blip does not page anyone.
DEFAULT_OFFLINE_GRACE = 60.0



class PushGovernor:
    """Thins playout pushes when the buffer is under SUSTAINED pressure.

    A camera that reports no FPS metadata (the pilot's Tapo over RTSP on
    Windows, 30 Aug) defeats stride sampling: every frame (~25/s) is pushed
    into a playout that pops at 12/s, so the buffer lives near full and the
    2x catch-up runs in a permanent lag-sprint-lag cycle — 'kinda slow and
    moves fast forwarded some times'. Sustained fullness means input
    structurally outruns output: skip every 2nd push, then every 3rd, until
    pressure clears. An HLS burst never trips it — its spike-drain rhythm
    keeps fullness intermittent, and intermittent pressure resets the clock.
    """

    def __init__(self, high_frac: float = 0.6, sustain_s: float = 8.0) -> None:
        self.high_frac = high_frac
        self.sustain_s = sustain_s
        self.skip = 0          # push every (skip+1)th frame
        self._over_since = 0.0
        self._calm_since = 0.0
        self._i = 0

    def admit(self, fullness: float, now: float) -> bool:
        """True if this frame may be pushed."""
        if fullness >= self.high_frac:
            self._calm_since = 0.0
            self._over_since = self._over_since or now
            if now - self._over_since >= self.sustain_s and self.skip < 3:
                self.skip += 1
                self._over_since = now
        else:
            self._over_since = 0.0
            if self.skip:
                self._calm_since = self._calm_since or now
                if now - self._calm_since >= self.sustain_s:
                    self.skip -= 1
                    self._calm_since = now
        self._i += 1
        return self.skip == 0 or (self._i % (self.skip + 1)) == 0


class PlayoutBuffer:
    """Paced playout for live URL sources: almost real-time, never fast-forward.

    The live wall's problem is shaped like this: HLS (and a reconnecting RTSP
    camera) delivers video in multi-second bursts. Draining to the newest frame
    made the wall freeze for a segment and then replay it at decode speed — a
    fast-forward lurch every five seconds (user report, 27 Aug). Pacing the
    decode loop instead made the source-side buffer grow without bound (the
    26 Aug lag bug). Both problems are the same mistake at different ends:
    playback speed coupled to arrival speed.

    This decouples them. Decode ingests as fast as the source delivers, frames
    land here as bounded JPEGs, and the publisher POPS at content rate — so a
    burst plays out smoothly a few seconds behind live. When the buffer runs
    high the pop rate doubles: a gentle 2x catch-up that drains backlog without
    looking like fast-forward. When the source outruns everything, the deque
    evicts oldest — lag is hard-capped at the buffer's depth.

    Frames are stored resized + JPEG-encoded (~60 KB), not raw (~6 MB): eight
    seconds of 1080p raw would be half a gigabyte per camera.
    """

    def __init__(self, rate: float, depth_seconds: float = 8.0,
                 max_width: int = 640, quality: int = 70) -> None:
        import collections
        self.rate = max(rate, 1.0)
        self.max_width = max_width
        self.quality = quality
        self._frames = collections.deque(maxlen=max(4, int(depth_seconds * self.rate)))
        self._lock = threading.Lock()
        self._next_due = 0.0
        self.evicted = 0                     # frames lost to the depth cap

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def push(self, image) -> bool:
        """Encode and queue a frame. Returns True if an old frame was evicted."""
        import cv2
        h, w = image.shape[:2]
        if self.max_width and w > self.max_width:
            image = cv2.resize(image, (self.max_width, max(1, int(h * self.max_width / w))))
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return False
        with self._lock:
            evict = len(self._frames) == self._frames.maxlen
            self._frames.append(buf.tobytes())
            if evict:
                self.evicted += 1
        return evict

    def pop_due(self, now: float) -> bytes | None:
        """The next frame if one is due at `now`, else None. Never bursts."""
        with self._lock:
            if not self._frames or now < self._next_due:
                return None
            period = 1.0 / self.rate
            if len(self._frames) > (self._frames.maxlen * 3) // 4:
                period /= 2.0                # quietly play 2x until we catch up
            self._next_due = now + period
            return self._frames.popleft()


class StreamDecoder:
    def __init__(self, camera_id: str, source: int | str, *, target_fps: float = 5.0,
                 reconnect: bool = True, reconnect_backoff: float = 1.0,
                 loop_files: bool = True, offline_grace_seconds: float = DEFAULT_OFFLINE_GRACE,
                 on_state_change=None, view_only: bool = False,
                 fallback_source: int | str | None = None,
                 relay_retry_seconds: float = 60.0,
                 dewarp: Any = None,
                 idle_when_unwatched: bool = False,
                 probe_source: int | str | None = None,
                 keyframes_only_min_width: int = 0) -> None:
        self.camera_id = camera_id
        self.source = source
        # Keyframe-only decode (capture.FFmpegKeyframeCapture) for a live
        # stream at least this wide; 0 = never. The first open is a normal
        # one, which tells us the width; a stream that qualifies is reopened
        # keyframes-only on the next loop. One frame per GOP, about one a
        # second, at a fraction of the decode cost: the seven 5-megapixel
        # HEVC cameras on the pilot server (13 Oct 2026). Falls back to full
        # decode by itself when the camera is moved to a smaller substream.
        self.keyframes_only_min_width = int(keyframes_only_min_width or 0)
        self._cap_keyframes = False
        self._cap: Any = None
        # A view-only camera nobody is watching decodes NOTHING. Its link
        # state comes from a cheap RTSP handshake every idle_probe_seconds
        # against the camera's own URL (probe_source), and the capture is
        # opened the moment a viewer appears (display_fps > 0) and released
        # again when the last one leaves. On the pilot server eight view-only
        # cameras, seven of them 5-megapixel HEVC, were being fully decoded
        # for nobody (Demi, 13 Oct 2026).
        self.idle_when_unwatched = bool(idle_when_unwatched)
        self.probe_source = probe_source if probe_source is not None else source
        self.idle_probe_seconds = 30.0
        self.idle = False
        self.idle_probes = 0
        # A fisheye camera's Dewarp (cvti.serving.dewarp). Applied to every
        # frame before it is stored, so detection, the wall, zones and the
        # verifier all see the flattened frame and agree on its geometry.
        self.dewarp = dewarp
        # W1.3: when `source` is the go2rtc restream, this is the camera's own
        # URL. A gateway that dies must cost this camera a reconnect, never
        # its coverage — after two straight failed reopens the decoder swaps
        # to the fallback and says so.
        #
        # The swap used to be one-way ("flapping is worse than settling").
        # On the pilot server that turned every relay hiccup into a second
        # RTSP session per camera for the rest of the run: the camera then
        # served both the gateway and this decoder, its network load doubled,
        # and cameras that cap their viewers refused the next one (11 Oct
        # 2026). So a fallen-back decoder now asks the restream again every
        # `relay_retry_seconds` — off-thread, so a slow open never stalls
        # this camera's frames — and moves back the moment it answers.
        self.fallback_source = fallback_source
        self.fell_back = False
        self._relay_source = source
        self.relay_retry_seconds = relay_retry_seconds
        self._relay_retry_at = 0.0
        self._relay_probe: threading.Thread | None = None
        self._relay_ready: Any = None       # an opened relay capture, handed over
        self.relay_returns = 0
        self.target_fps = target_fps
        # A view-only camera streams glass and runs NO detection (4 Sep, pilot):
        # the flag rides here so link/ingest status can say so honestly.
        self.view_only = view_only
        # Viewer-aware display boost (3 Sep): while someone is actually
        # WATCHING this camera (an open /stream connection on the publisher),
        # the decode rate steps up to this so the glass is smooth; the moment
        # nobody watches it drops back to target_fps, which is all detection
        # samples anyway. 0 = no boost. Written by the publisher thread, read
        # by the decode loop — a plain float, atomic enough in CPython, and a
        # one-loop-late boost costs one frame period.
        self.display_fps: float = 0.0
        self.reconnect = reconnect
        self.reconnect_backoff = reconnect_backoff   # seconds * attempt, capped at 5s
        # File sources loop by default so demo footage streams continuously
        # (real RTSP/webcam sources are live and never hit EOF).
        self.loop_files = loop_files
        self._min_period = 1.0 / target_fps if target_fps > 0 else 0.0
        # URL sources deliver in bursts; the wall plays them from here, paced.
        # Files pace themselves; a webcam is inherently real-time. Neither needs it.
        self.playout = PlayoutBuffer(rate=target_fps) if self._is_live() else None
        self._governor = PushGovernor() if self.playout is not None else None
        self._latest: Frame | None = None
        self._seq = 0                      # bumps per decoded frame (peek dedup)
        self.stale_dropped = 0             # buffered frames skipped to stay live
        self._consumed = True          # True once the current latest was read
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self._t0 = 0.0
        self._fps = 0.0
        self.ended = False
        self.reconnects = 0            # how many times we've re-opened a live stream
        self.offline_grace_seconds = offline_grace_seconds
        self.on_state_change = on_state_change
        self.state = CONNECTED
        self.state_since = time.time()
        self.last_frame_at = 0.0
        # Adaptive ingest state (see _observe_ingest).
        self.stream_width = 0
        self.stream_height = 0
        self._busy_ms: float | None = None
        self.sustainable_fps = 30.0
        self._ingest_limited = False
        # Bounded: the useful signal is "it has been flapping", not every attempt
        # since install.
        self.attempt_history: deque = deque(maxlen=20)

    # --- adaptive ingest (3 Sep) --------------------------------------------
    # "Slow stream" on the pilot's box was the machine, not the camera: a
    # CPU-only laptop decoding a 1080p25 RTSP main stream while the VLM pegs
    # every core cannot hit even 4fps — and nothing measured it, adapted to
    # it, or SAID it. The decoder now measures what one kept frame actually
    # costs on THIS machine (grabs + decode, wall time), derives the rate it
    # can sustain while spending at most half its thread on ingest, caps the
    # effective fps there, and reports all of it through link_status so the
    # health doc and the Cameras screen can tell the operator the truth
    # ("this machine sustains ~2.7fps of 1920x1080@25 — use the camera's
    # substream") instead of just looking slow.
    ADAPT_FLOOR_FPS = 1.0
    ADAPT_TOLERANCE = 1.15   # 15% over pace is jitter; beyond it is deficit

    @property
    def time_in_state(self) -> float:
        return time.time() - self.state_since

    def _observe_ingest(self, elapsed_seconds: float, stride: int) -> None:
        """Is this machine keeping the pace this stream asks of it?

        `elapsed` is one kept frame's wall time (stride grabs + decode). At
        the live edge grab() WAITS for frames at source rate, so raw elapsed
        punishes healthy machines for the camera's own pacing — the honest
        deficit signal is elapsed exceeding what the iteration SHOULD take:
        the larger of our own sampling period and the source's arrival time
        for `stride` frames. Only a machine falling behind that deadline gets
        capped, at the rate it actually achieves; headroom recovers the cap.
        """
        ms = elapsed_seconds * 1000.0
        self._busy_ms = ms if self._busy_ms is None else 0.1 * ms + 0.9 * self._busy_ms
        from cvti.serving.perf import BOARD
        BOARD.observe("decode", self.camera_id, ms)
        deadline_ms = self._min_period * 1000.0
        if self._fps > 1e-3:
            deadline_ms = max(deadline_ms, (stride / self._fps) * 1000.0)
        if deadline_ms <= 0:
            return
        if self._busy_ms > deadline_ms * self.ADAPT_TOLERANCE:
            achieved = max(self.ADAPT_FLOOR_FPS, 1000.0 / self._busy_ms)
            self.sustainable_fps = 0.9 * self.sustainable_fps + 0.1 * achieved
        else:
            self.sustainable_fps = min(30.0, 0.9 * self.sustainable_fps + 0.1 * 30.0)
        limited = self.sustainable_fps < min(self.target_fps, 30.0) - 0.1
        if limited != self._ingest_limited:
            self._ingest_limited = limited
            if limited:
                log.warning(
                    "[decode %s] this machine sustains ~%.1ffps of %dx%d@%.0f — "
                    "sampling reduced (asked %.0f). A lower-resolution stream "
                    "(the camera's substream) would restore full rate.",
                    self.camera_id, self.sustainable_fps, self.stream_width,
                    self.stream_height, self._fps, self.target_fps)
            else:
                log.info("[decode %s] ingest recovered — full sampling rate restored",
                         self.camera_id)
        # The same verdict the log line carries, into the file the customer
        # actually sends us. Whether this machine sustains the rate asked of it
        # is what separates "the box is too slow" from "the camera is pacing
        # us" — and decode's raw milliseconds cannot tell them apart, because
        # at the live edge grab() waits for the source (see the docstring).
        try:
            BOARD.note("decode", self.camera_id, **self.ingest_status())
        except Exception:  # noqa: BLE001 - a metric must never stop a decoder
            log.debug("ingest note dropped for %s", self.camera_id, exc_info=True)

    def ingest_status(self) -> dict:
        out = {
            "width": self.stream_width, "height": self.stream_height,
            "source_fps": round(self._fps, 1),
            "busy_ms_per_frame": round(self._busy_ms or 0.0, 1),
            "sustainable_fps": round(self.sustainable_fps, 1),
            "sampling_fps": round(self._effective_fps(), 1),
            "limited": self._ingest_limited,
        }
        if getattr(self, "view_only", False):   # tests build decoders raw
            out["view_only"] = True
        return out

    def _set_state(self, state: str, detail: str = "") -> None:
        if state == self.state:
            return
        previous, held = self.state, self.time_in_state
        self.state, self.state_since = state, time.time()
        level = log.warning if state != CONNECTED else log.info
        level("[link %s] %s -> %s after %.0fs%s", self.camera_id, previous, state,
              held, f" ({detail})" if detail else "")
        if self.on_state_change is not None:
            try:
                self.on_state_change(self.camera_id, previous, state, held)
            except Exception:  # noqa: BLE001
                log.warning("[link %s] state-change callback failed", self.camera_id,
                            exc_info=True)

    def _wants_idle(self) -> bool:
        return self.idle_when_unwatched and self._is_live() and self.display_fps <= 0

    def _probe_alive(self) -> bool:
        from cvti.serving.capture import rtsp_alive
        return rtsp_alive(self.probe_source)

    def _idle_until_watched(self) -> None:
        """Hold here, decoding nothing, until a viewer appears or stop().
        Link state is kept honest with a periodic reachability probe."""
        self.idle = True
        log.info("[decode %s] idle: view-only and unwatched; probing reachability "
                 "every %.0fs instead of decoding", self.camera_id, self.idle_probe_seconds)
        next_probe = 0.0
        try:
            while not self._stop.is_set() and self._wants_idle():
                now = time.monotonic()
                if now >= next_probe:
                    next_probe = now + self.idle_probe_seconds
                    self.idle_probes += 1
                    if self._probe_alive():
                        self.last_frame_at = time.time()
                        if self.state != CONNECTED:
                            self._set_state(CONNECTED, "camera reachable")
                    elif self.state == CONNECTED:
                        self._set_state(RECONNECTING, "camera not answering")
                    elif (self.state == RECONNECTING
                          and self.time_in_state >= self.offline_grace_seconds):
                        self._set_state(OFFLINE, f"unreachable for {self.time_in_state:.0f}s")
                self._stop.wait(0.25)
        finally:
            self.idle = False
        if not self._stop.is_set():
            log.info("[decode %s] viewer arrived; opening the stream", self.camera_id)

    def link_status(self) -> dict:
        return {"camera_id": self.camera_id, "state": self.state,
                "idle": self.idle,
                "decode": "keyframes" if self._cap_keyframes else "full",
                "time_in_state": round(self.time_in_state, 1),
                "reconnects": self.reconnects,
                "last_frame_at": self.last_frame_at,
                # The liveness number the UI shows. A connected camera whose
                # frames stopped is exactly the ambiguity this exists to remove.
                "last_frame_age_s": (round(time.time() - self.last_frame_at, 1)
                                     if self.last_frame_at else None),
                # How far behind live we would have been: buffered frames
                # skipped to stay at the present. A big number on a camera
                # means its stream bursts (HLS), not that anything is wrong.
                "stale_dropped": self.stale_dropped,
                "ingest": self.ingest_status(),
                # Running, but on the camera's own URL because the gateway
                # went away — coverage held, the upgrade did not (W1.3).
                "gateway_fallback": self.fell_back,
                "attempts": list(self.attempt_history)}

    def _is_live(self) -> bool:
        return not str(self.source).isdigit() and "://" in str(self.source)

    def _effective_fps(self) -> float:
        """target_fps or the display boost — capped at what THIS machine has
        measurably sustained for this stream (adaptive ingest, 3 Sep): asking
        a starved CPU for 15fps it cannot decode only makes every consumer
        later. The cap never goes below the floor, so detection always gets
        something."""
        boost = self.display_fps
        desired = max(self.target_fps, boost) if boost > 0 else self.target_fps
        return max(self.ADAPT_FLOOR_FPS, min(desired, self.sustainable_fps))

    def _pacing(self) -> tuple[int, float]:
        """(stride, min_period) for the CURRENT effective fps — recomputed
        every loop iteration so a boost takes effect within one frame."""
        fps = self._effective_fps()
        stride = max(1, round(self._fps / fps)) if (self._fps > 1e-3 and fps > 0) else 1
        return stride, (1.0 / fps if fps > 0 else 0.0)

    def start(self) -> "StreamDecoder":
        self._thread = threading.Thread(target=self._loop, name=f"decode-{self.camera_id}",
                                        daemon=True)
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def _wants_keyframes(self) -> bool:
        return (self.keyframes_only_min_width > 0 and self._is_live()
                and self.stream_width >= self.keyframes_only_min_width)

    def _open(self):
        # One capture path for every platform (cvti/serving/capture.py): the
        # right backend, low-latency flags, and a one-frame buffer request.
        from cvti.serving.capture import open_capture
        want = self._wants_keyframes()
        cap = open_capture(self.source, keyframes_only=want)
        self._cap_keyframes = bool(getattr(cap, "keyframes_only", False))
        self._cap = cap
        if want and not self._cap_keyframes:
            # No ffmpeg or it refused: do not ask again for this stream.
            self.keyframes_only_min_width = 0
        return cap

    def _loop(self) -> None:
        import cv2
        if self._wants_idle():
            self._idle_until_watched()
            if self._stop.is_set():
                return
        cap = self._open()
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        # Sample the effective fps by SKIPPING frames, not by slowing playback:
        # keep 1 frame every `stride`. This makes a file advance through video
        # at real time while only decoding the frames we need (and drops a
        # 30fps live source to the same sampled rate). Recomputed per
        # iteration: the display boost changes the answer while running.
        orig_index = 0
        attempt = 0
        while not self._stop.is_set():
            if self._wants_idle():
                # The last viewer left: release the stream and wait for the next.
                try:
                    cap.release()
                except Exception:  # noqa: BLE001 - the capture is being discarded
                    log.debug("[decode %s] release before idle failed", self.camera_id, exc_info=True)
                self._idle_until_watched()
                if self._stop.is_set():
                    return
                cap = self._open()
                self._fps = cap.get(cv2.CAP_PROP_FPS) or self._fps
                attempt = 0
                continue
            if self._wants_keyframes() != self._cap_keyframes:
                # The stream turned out big enough for keyframe-only decode
                # (or was moved to a small substream): reopen the right way.
                log.info("[decode %s] %dx%d stream: switching to %s decode", self.camera_id,
                         self.stream_width, self.stream_height,
                         "keyframe-only" if self._wants_keyframes() else "full")
                try:
                    cap.release()
                except Exception:  # noqa: BLE001 - the capture is being discarded
                    log.debug("[decode %s] release before reopen failed", self.camera_id, exc_info=True)
                cap = self._open()
                self._fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                attempt = 0
                continue
            if self.fell_back:
                cap = self._maybe_return_to_relay(cap)
            stride, self._min_period = self._pacing()
            if self.playout is not None:
                # Playout pops at content rate; a boosted camera's content
                # rate is the boosted one, or the wall stays at 4fps anyway.
                self.playout.rate = max(self._effective_fps(), 1.0)
            loop_start = time.perf_counter()
            ok = True
            for _ in range(stride - 1):        # cheaply skip intermediate frames
                ok = cap.grab()
                orig_index += 1
                if not ok:
                    break
            if ok:
                ok, image = cap.read()
                orig_index += 1
            if ok and self.dewarp is not None:
                try:
                    image = self.dewarp.apply(image)
                except Exception:  # noqa: BLE001 - a bad frame must not stop the camera
                    log.warning("[decode %s] dewarp failed on a frame; passing it through",
                                self.camera_id, exc_info=True)
            if ok:
                # What one kept frame COSTS on this machine (grabs + decode,
                # wall time) — feeds the adaptive ingest cap.
                self._observe_ingest(time.perf_counter() - loop_start, stride)
                if not self.stream_width:
                    self.stream_height, self.stream_width = image.shape[:2]
            if not ok:
                if self.reconnect and self._is_live():
                    # Live source dropped (camera reboot / network blip). Keep
                    # reopening with an exponential, capped backoff until it
                    # comes back — linear retry hammers a camera that is down
                    # for an hour, and the log with it.
                    self.reconnects += 1
                    attempt += 1
                    backoff = min(self.reconnect_backoff * (2 ** (attempt - 1)), 30.0)
                    if self.state == CONNECTED:
                        self._set_state(RECONNECTING, "stream dropped")
                    elif (self.state == RECONNECTING
                          and self.time_in_state >= self.offline_grace_seconds):
                        self._set_state(OFFLINE,
                                        f"unreachable for {self.time_in_state:.0f}s")
                    self.attempt_history.append(
                        {"at": time.time(), "attempt": attempt, "backoff": backoff})
                    if (self.fallback_source is not None and not self.fell_back
                            and attempt >= 2):
                        # The restream is refusing us while the camera itself
                        # may be fine — go direct. Logged as a warning: the
                        # gateway's death is a support fact, not a mystery.
                        log.warning("[decode %s] restream unreachable after %d "
                                    "attempts — falling back to the camera's "
                                    "own source", self.camera_id, attempt)
                        self.source = self.fallback_source
                        self.fell_back = True
                        self._relay_retry_at = time.monotonic() + self.relay_retry_seconds
                        self.playout = (PlayoutBuffer(rate=self.target_fps)
                                        if self._is_live() else None)
                        self._governor = (PushGovernor()
                                          if self.playout is not None else None)
                        cap.release()
                        cap = self._open()
                        attempt = 0
                        continue
                    log.warning("[decode %s] stream dropped; reopening in %.0fs (attempt %d)",
                                self.camera_id, backoff, attempt)
                    cap.release()
                    self._stop.wait(backoff)   # interruptible so stop() is prompt
                    cap = self._open()
                    continue
                is_file = not self._is_live() and not str(self.source).isdigit()
                if self.loop_files and is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind: stream footage continuously
                    orig_index = 0
                    continue
                self.ended = True          # webcam exhausted / looping disabled
                break
            attempt = 0                    # a healthy read resets the backoff
            self.last_frame_at = time.time()
            if self.state != CONNECTED:
                # Recovery is announced, not silent — an operator who was told
                # the camera went down has to be told it came back.
                self._set_state(CONNECTED, "stream recovered")
            # VIDEO time from the original frame index so dwell/loiter thresholds
            # are correct regardless of sample rate. Wall-clock fallback if no fps.
            ts = (orig_index / self._fps) if self._fps > 1e-3 else (time.perf_counter() - self._t0)
            with self._lock:
                self._index += 1
                self._latest = Frame(self.camera_id, image, orig_index, ts)
                self._seq += 1
                self._consumed = False
            # Pace each kept frame to target_fps — for FILES, which must play at
            # real time. A live source must NOT be paced this way: HLS (and a
            # reconnecting RTSP camera) hands OpenCV a burst of buffered frames
            # at a segment boundary, and reading one per period consumes that
            # backlog instead of the present, falling further behind live the
            # longer it runs. Measured 26 Aug on a reference HLS CDN: bursts at
            # ~39 fps punctuated by stalls up to 5.5s — pacing turns that into
            # permanent, growing lag. Live sources therefore DRAIN to the live
            # edge instead: grab (decode-free) whatever is already buffered,
            # bounded in both count and time, and keep only the newest.
            if self.playout is not None:
                # Live URL source: the wall plays from the playout buffer, so
                # KEEP the sampled frames instead of draining to the newest.
                # Ingest speed is governed by fill level, not by sleep: at the
                # live edge cap.grab() blocks at source rate anyway, and during
                # a burst we chew the backlog flat-out — sleeping through a
                # burst is exactly how the source-side lag grew before. The
                # deque's depth cap bounds how far behind the wall can fall.
                fullness = len(self.playout) / max(1, self.playout._frames.maxlen)
                if self._governor.admit(fullness, time.perf_counter()):
                    if self.playout.push(image):
                        self.stale_dropped += 1
                else:
                    self.stale_dropped += 1      # thinned under sustained pressure
                if len(self.playout) <= (self.playout._frames.maxlen // 2) \
                        and self._min_period:
                    elapsed = time.perf_counter() - loop_start
                    if elapsed < self._min_period:
                        time.sleep(self._min_period - elapsed)
            elif self._min_period:
                elapsed = time.perf_counter() - loop_start
                if elapsed < self._min_period:
                    time.sleep(self._min_period - elapsed)
        cap.release()

    def read_latest(self) -> Frame | None:
        """Return the newest unread frame, or None if nothing new since last read."""
        with self._lock:
            if self._latest is None or self._consumed:
                return None
            self._consumed = True
            return self._latest

    def peek_latest(self) -> tuple:
        """(frame, seq) WITHOUT consuming — the smooth-publish path reads here.

        read_latest() marks frames consumed for the DETECTION loop; a second
        consumer would steal frames from the models. The live wall doesn't need
        to own the frame, only to see the newest one, so it peeks. `seq` lets
        the publisher skip frames it has already shipped.
        """
        with self._lock:
            return self._latest, self._seq

    def stop(self) -> None:
        self._stop.set()
        cap = getattr(self, "_cap", None)
        if cap is not None and hasattr(cap, "interrupt"):
            try:
                cap.interrupt()          # a keyframe capture may be mid-read on its pipe
            except Exception:  # noqa: BLE001 - teardown
                log.debug("[decode %s] interrupt failed", self.camera_id, exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        ready, self._relay_ready = self._relay_ready, None
        if ready is not None:
            try:
                ready.release()
            except Exception:  # noqa: BLE001 - teardown
                log.debug("[decode %s] relay probe release failed", self.camera_id, exc_info=True)

    # --- back to the relay after a fallback ----------------------------------
    def _maybe_return_to_relay(self, cap):
        """Called once per loop while on the fallback source. Swaps to a
        relay capture the probe handed over, else starts a probe when due."""
        ready, self._relay_ready = self._relay_ready, None
        if ready is not None:
            log.info("[decode %s] restream answering again — back on the gateway",
                     self.camera_id)
            try:
                cap.release()
            except Exception:  # noqa: BLE001 - the old capture is being discarded
                log.debug("[decode %s] fallback capture release failed", self.camera_id, exc_info=True)
            self.source = self._relay_source
            self.fell_back = False
            self.relay_returns += 1
            self.playout = (PlayoutBuffer(rate=self.target_fps)
                            if self._is_live() else None)
            self._governor = PushGovernor() if self.playout is not None else None
            return ready
        if (self.relay_retry_seconds > 0
                and time.monotonic() >= self._relay_retry_at
                and (self._relay_probe is None or not self._relay_probe.is_alive())):
            self._relay_retry_at = time.monotonic() + self.relay_retry_seconds
            self._relay_probe = threading.Thread(
                target=self._probe_relay, name=f"relay-probe:{self.camera_id}",
                daemon=True)
            self._relay_probe.start()
        return cap

    def _probe_relay(self) -> None:
        """Open the restream and read one frame. Success hands the capture
        to the decode loop; failure is silent — it will try again later."""
        from cvti.serving.capture import open_capture
        probe = None
        try:
            probe = open_capture(self._relay_source)
            opened = getattr(probe, "isOpened", lambda: True)()
            ok = bool(probe is not None and opened and probe.grab())
        except Exception:  # noqa: BLE001 - a probe must never take the decoder down
            log.debug("[decode %s] relay probe failed", self.camera_id, exc_info=True)
            ok = False
        if ok and not self._stop.is_set() and self.fell_back:
            self._relay_ready = probe
            return
        if probe is not None:
            try:
                probe.release()
            except Exception:  # noqa: BLE001 - teardown
                log.debug("[decode %s] probe capture release failed", self.camera_id, exc_info=True)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

