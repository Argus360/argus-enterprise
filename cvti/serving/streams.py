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

# How long a LIVE stream may return failed reads before we treat it as dropped
# and reopen the RTSP connection. A lossy 4MP HEVC stream over a WAN returns a
# bad read on every corrupted frame while the connection itself is fine; tearing
# down and reopening on the first one makes the camera flap (CHI, 21 Oct 2026:
# 7 reconnects/min on the heaviest stream). We instead skip the bad frame and
# keep reading, and only reconnect once no good frame has arrived for this long.
DEFAULT_READ_FAIL_GRACE = 3.0



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
                 read_fail_grace_seconds: float = DEFAULT_READ_FAIL_GRACE,
                 on_state_change=None, view_only: bool = False,
                 fallback_source: int | str | None = None,
                 relay_retry_seconds: float = 60.0,
                 dewarp: Any = None,
                 idle_when_unwatched: bool = False,
                 probe_source: int | str | None = None,
                 keyframes_only_min_width: int = 0,
                 keyframe_policy: str = "auto",
                 keyframe_cost_share: float = 0.25,
                 cost_window_s: float = 2.0,
                 playout_idle_period: float = 2.0,
                 keyframe_min_fps: float = 0.5) -> None:
        self.camera_id = camera_id
        self.source = source
        # What decoding this stream costs THIS machine, in CPU time on the
        # decode thread (time.thread_time), not wall time: at the live edge a
        # read waits for the camera, so wall time says 40 ms for a 25 fps
        # camera on an idle box. Keyframe-only decode is chosen by this
        # measurement, not by whether the detector has a GPU (Tolu's audit,
        # 15 Oct 2026): a GPU box can still decode on the CPU, and a hardware
        # decoder makes a wide stream cheap.
        #
        # Two numbers, both independent of OUR sampling rate (Tolu's review:
        # timing a whole stride made a 4 ms camera look like 32 ms when
        # detection dropped from 30 to 4 fps, because OpenCV decodes every
        # skipped frame too):
        #   decode_cpu_ms     CPU per SOURCE frame decoded (stride grabs + read,
        #                     divided by the frames that iteration consumed)
        #   decode_cpu_share  CPU consumed by decoding this stream per wall
        #                     second, measured over cost_window_s windows:
        #                     0.25 = a quarter of one core. This is what the
        #                     keyframe decision uses.
        # frame_cpu_ms adds the dewarp; it is per KEPT frame, which is what
        # the wall's decode budget pays per displayed frame.
        policy = str(keyframe_policy or "auto").lower()
        if policy not in ("auto", "force", "off"):
            raise ValueError(f"keyframe_policy must be auto, force or off, not {keyframe_policy!r}")
        self.keyframe_policy = policy
        self.keyframe_cost_share = float(keyframe_cost_share)
        self.cost_window_s = float(cost_window_s)
        self.decode_cpu_ms: float | None = None    # EWMA, per source frame, full decode only
        self.decode_cpu_share: float | None = None # EWMA over windows, full decode only
        self.frame_cpu_ms: float | None = None     # EWMA, decode + dewarp, per kept frame
        self._cost_windows = 0                     # completed windows since (re)measurement
        self._window_cpu_s = 0.0
        self._window_wall_s = 0.0
        self._window_frames = 0
        self._window_started_at = 0.0
        # The playout buffer holds resized, JPEG-encoded frames for the MJPEG
        # wall. It was fed on every kept frame whether or not anyone was
        # watching, so a camera whose tile is closed or on WebRTC still paid
        # a resize and a JPEG encode per frame (Tolu's audit, 15 Oct 2026).
        # With no MJPEG viewer (display_fps == 0, set by the wall from the
        # publisher's viewer count) the decoder now encodes one keep-alive
        # frame every playout_idle_period seconds, so a tile that opens has a
        # fresh picture at once, and skips the rest. Detection and evidence
        # never used this buffer; they read the raw frame.
        self.playout_idle_period = float(playout_idle_period)
        self.playout_skipped = 0
        self._last_idle_push = 0.0
        # Codec awareness and the achieved rate (Tolu's audit, 15 Oct 2026).
        # Keyframe-only decode only pays off on inter-frame codecs: on MJPEG
        # every frame is a keyframe, so it would save nothing and is never
        # chosen. In keyframe mode the rate detection actually gets is one
        # frame per GOP; below keyframe_min_fps that can miss a two-second
        # gate crossing, so health says so and names the fix (a detection
        # substream, or a shorter I-frame interval on the camera).
        self.codec = ""
        self.keyframe_min_fps = float(keyframe_min_fps)
        self.achieved_fps: float | None = None     # EWMA of delivered frames/s
        self._sparse_warned = False
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
        # The size the camera SENDS, before any dewarp. stream_width is what
        # consumers see (dewarped); this is what decoding costs. The three
        # ceiling fisheyes on the pilot server were judged by their 1600 px
        # panorama and never switched to keyframe decode (14 Oct 2026).
        self.source_width = 0
        self.source_height = 0
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
        self.read_fail_grace_seconds = read_fail_grace_seconds
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
            "codec": getattr(self, "codec", ""),
            "achieved_fps": (round(self.achieved_fps, 2)
                             if getattr(self, "achieved_fps", None) is not None else None),
            "keyframe_min_fps": getattr(self, "keyframe_min_fps", 0.5),
            "decode_cpu_ms": (round(self.decode_cpu_ms, 1)
                              if getattr(self, "decode_cpu_ms", None) is not None else None),
            "decode_cpu_share": (round(self.decode_cpu_share, 3)
                                 if getattr(self, "decode_cpu_share", None) is not None else None),
            "keyframe_policy": getattr(self, "keyframe_policy", "auto"),
            "frame_cpu_ms": (round(self.frame_cpu_ms, 1)
                             if getattr(self, "frame_cpu_ms", None) is not None else None),
            "wall_fps": round(float(getattr(self, "display_fps", 0.0) or 0.0), 1),
            "playout_skipped": getattr(self, "playout_skipped", 0),
            "source_width": getattr(self, "source_width", 0),      # tests build decoders raw
            "source_height": getattr(self, "source_height", 0),
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

    KEYFRAME_DECISION_WINDOWS = 2
    # Codecs where decoding only keyframes saves real work. MJPEG and raw
    # formats are all keyframes already.
    INTERFRAME_CODECS = frozenset({"h264", "avc1", "avc", "x264", "hevc", "hvc1", "hev1",
                                   "h265", "x265", "av01", "av1", "vp9", "vp09", "vp8",
                                   "mpeg4", "mp4v", "mpg4", "mpeg2video", "mpg2", "xvid"})

    @staticmethod
    def _codec_of(cap) -> str:
        """The stream's codec as a short lowercase name, or "" if unknown."""
        name = getattr(cap, "codec", "")
        if name:
            return str(name).lower()
        try:
            raw = int(cap.get(6))              # cv2.CAP_PROP_FOURCC, no cv2 import here
        except Exception:  # noqa: BLE001 - a capture that cannot answer is "unknown"
            log.debug("codec probe failed; treating the codec as unknown", exc_info=True)
            return ""
        if raw <= 0:
            return ""
        chars = [chr((raw >> (8 * i)) & 0xFF) for i in range(4)]
        name = "".join(c for c in chars if 32 < ord(c) < 127).lower()
        return name if len(name) >= 2 else ""    # a capture that answers nonsense is "unknown"

    def _wants_keyframes(self) -> bool:
        """Keyframe-only decode for a live stream that is wide enough and,
        under the "auto" policy, whose full decode has measured expensive on
        this machine over KEYFRAME_DECISION_WINDOWS windows. "force" switches
        every eligible stream without measuring; "off" never switches. Once
        chosen it sticks (nothing is measured in keyframe mode) until the
        source changes size, when it is measured again."""
        if self.keyframe_policy == "off":
            return False
        if not (self.keyframes_only_min_width > 0 and self._is_live()
                and self.source_width >= self.keyframes_only_min_width):
            return False
        if self._cap_keyframes or self.keyframe_policy == "force":
            return True
        if self.codec and self.codec not in self.INTERFRAME_CODECS:
            return False                       # MJPEG etc.: nothing to skip
        return (self._cost_windows >= self.KEYFRAME_DECISION_WINDOWS
                and self.decode_cpu_share is not None
                and self.decode_cpu_share >= self.keyframe_cost_share)

    def _observe_decode_cost(self, cpu_s: float, wall_s: float, source_frames: int) -> None:
        """Account one full-decode iteration: `source_frames` frames went
        through the decoder (stride grabs + the read) in cpu_s of CPU and
        wall_s of wall time (grabs + read only; the pacing sleep is not
        included). The share is CPU per second OF STREAM: per-source-frame
        cost times the source frame rate when the capture reports one, so
        neither our stride nor our pacing sleep can move it; without a
        reported rate it falls back to CPU over decode wall time. Windows
        close on real elapsed time, so a slowly paced loop still decides."""
        per_frame_ms = cpu_s * 1000.0 / max(1, source_frames)
        self.decode_cpu_ms = (per_frame_ms if self.decode_cpu_ms is None
                              else 0.2 * per_frame_ms + 0.8 * self.decode_cpu_ms)
        now = time.perf_counter()
        if not self._window_started_at:
            self._window_started_at = now
        self._window_cpu_s += cpu_s
        self._window_wall_s += wall_s
        self._window_frames += max(1, source_frames)
        if now - self._window_started_at >= self.cost_window_s:
            if self._fps > 1e-3:
                share = (self._window_cpu_s / self._window_frames) * self._fps
            else:
                share = self._window_cpu_s / max(1e-6, self._window_wall_s)
            self.decode_cpu_share = (share if self.decode_cpu_share is None
                                     else 0.5 * share + 0.5 * self.decode_cpu_share)
            self._cost_windows += 1
            self._window_cpu_s = self._window_wall_s = 0.0
            self._window_frames = 0
            self._window_started_at = now

    def _reset_decode_cost(self) -> None:
        self.decode_cpu_ms = self.decode_cpu_share = None
        self._cost_windows = 0
        self._window_cpu_s = self._window_wall_s = 0.0
        self._window_frames = 0
        self._window_started_at = 0.0

    def _open(self):
        # One capture path for every platform (cvti/serving/capture.py): the
        # right backend, low-latency flags, and a one-frame buffer request.
        from cvti.serving.capture import open_capture
        want = self._wants_keyframes()
        cap = open_capture(self.source, keyframes_only=want)
        self._cap_keyframes = bool(getattr(cap, "keyframes_only", False))
        # An ffmpeg-fed capture (GPU decode) pushes frames at the stream's own
        # rate; the loop must read as they arrive and never sleep between reads.
        self._cap_push = bool(getattr(cap, "push_source", False))
        self._cap = cap
        self.codec = self._codec_of(cap) or self.codec
        self.achieved_fps = None               # a new capture: measure its rate afresh
        self._sparse_warned = False
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
                         self.source_width, self.source_height,
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
            if self._cap_keyframes:
                # A keyframe-only capture already IS the sample: it hands over
                # one frame per GOP and nothing in between. Stride-skipping
                # here made every grab() read and throw away a whole keyframe
                # (12 MB of raw pixels on a 2592x1520 camera), so a stride of
                # 25 to 100 cost 1 to 1.5 s of CPU per frame kept and starved
                # detection to one frame every 3 to 12 s (CHI, 18 Oct 2026:
                # frame_cpu_ms 870 to 1550 while decode_cpu_ms was 13 to 29).
                # No pacing sleep either: ffmpeg writes raw frames into a
                # pipe, and sleeping between reads backs that pipe up until
                # ffmpeg stalls on its RTSP input and the stream drops (the
                # same cameras logged "stream dropped; reopening" every few
                # minutes). Every keyframe is read the moment it arrives;
                # the detector still samples at its own rate downstream.
                stride = 1
                self._min_period = 0.0
            elif getattr(self, "_cap_push", False):
                # GPU decode via ffmpeg: every frame is already decoded and
                # pushed at stream rate. Keep the stride (grab() just drains the
                # socket), but never sleep — a paused reader backs the socket up
                # until ffmpeg stalls on its RTSP input and the stream drops.
                self._min_period = 0.0
            if self.playout is not None:
                # Playout pops at content rate; a boosted camera's content
                # rate is the boosted one, or the wall stays at 4fps anyway.
                self.playout.rate = max(self._effective_fps(), 1.0)
            loop_start = time.perf_counter()
            cpu_start = time.thread_time()
            ok = True
            for _ in range(stride - 1):        # cheaply skip intermediate frames
                ok = cap.grab()
                orig_index += 1
                if not ok:
                    break
            if ok:
                ok, image = cap.read()
                orig_index += 1
            if ok and not self._cap_keyframes:
                self._observe_decode_cost(time.thread_time() - cpu_start,
                                          time.perf_counter() - loop_start, stride)
            if ok and not self.source_width:
                self.source_height, self.source_width = image.shape[:2]
            elif ok and (self.source_width, self.source_height) != tuple(image.shape[1::-1]) \
                    and not self._cap_keyframes:
                # The camera was moved to another stream size: measure again.
                self.source_height, self.source_width = image.shape[:2]
                self._reset_decode_cost()
            if ok and self.dewarp is not None:
                try:
                    image = self.dewarp.apply(image)
                except Exception:  # noqa: BLE001 - a bad frame must not stop the camera
                    log.warning("[decode %s] dewarp failed on a frame; passing it through",
                                self.camera_id, exc_info=True)
            if ok:
                frame_cpu = (time.thread_time() - cpu_start) * 1000.0
                self.frame_cpu_ms = (frame_cpu if self.frame_cpu_ms is None
                                     else 0.2 * frame_cpu + 0.8 * self.frame_cpu_ms)
                # What one kept frame COSTS on this machine (grabs + decode,
                # wall time) — feeds the adaptive ingest cap. Not in keyframe
                # mode: there the wall time is the camera's GOP, not our work,
                # and treating it as cost shrank the sustainable rate, grew
                # the stride, and discarded even more keyframes.
                if not self._cap_keyframes:
                    self._observe_ingest(time.perf_counter() - loop_start, stride)
                if not self.stream_width:
                    self.stream_height, self.stream_width = image.shape[:2]
            if not ok:
                # Tolerate transient decode failures on a live stream. A lossy
                # HEVC feed returns a bad read on a corrupted frame while the
                # connection is fine; as long as a good frame arrived within the
                # grace window, skip this one and keep reading rather than tear
                # down and reopen RTSP (which is what made the WAN cameras flap).
                # A genuine drop stops delivering good frames, so last_frame_at
                # ages past the grace and we fall through to the reconnect below.
                if (self.reconnect and self._is_live() and self.last_frame_at
                        and time.time() - self.last_frame_at
                        < self.read_fail_grace_seconds):
                    self._stop.wait(0.03)   # don't busy-spin on repeated bad reads
                    continue
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
            previous_frame_at = self.last_frame_at
            self.last_frame_at = time.time()
            if previous_frame_at:
                gap = self.last_frame_at - previous_frame_at
                if 0.0 < gap < 60.0:
                    rate = 1.0 / gap
                    self.achieved_fps = (rate if self.achieved_fps is None
                                         else 0.2 * rate + 0.8 * self.achieved_fps)
                    if (self._cap_keyframes and not self._sparse_warned
                            and self.achieved_fps < self.keyframe_min_fps):
                        self._sparse_warned = True
                        log.warning("[decode %s] keyframes arrive about every %.1fs (%.2f/s): "
                                    "detection may miss short events. Give this camera a "
                                    "detection substream (detect_source) or shorten its "
                                    "I-frame interval.", self.camera_id,
                                    1.0 / max(self.achieved_fps, 1e-6), self.achieved_fps)
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
                watched = self.display_fps > 0
                if not watched and (time.perf_counter() - self._last_idle_push
                                    < self.playout_idle_period):
                    self.playout_skipped += 1    # nobody watching: no encode
                elif self._governor.admit(fullness, time.perf_counter()):
                    if not watched:
                        self._last_idle_push = time.perf_counter()
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

