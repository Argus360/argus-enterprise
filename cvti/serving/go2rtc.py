"""The go2rtc stream gateway — one camera session, every consumer (W1).

go2rtc opens each network camera ONCE and fans it out: an RTSP restream on
localhost for the engine's decoder, and WebRTC for wall clients that want the
camera's own H.264 passed through with no re-encode. What it buys, in the
order that survived measurement (see tools/latency_baseline.py — the speed
case largely died on 8 Sep when single-threaded decode met the 300ms SLO over
plain MJPEG):

- cameras the current decoder cannot open at all (H.265 and assorted RTSP
  quirks — discovery.py already turns these away as `unsupported-codec`);
- H.264 passthrough to the Electron wall, shedding the per-frame JPEG encode
  on boxes where CPU is the whole complaint;
- reconnects and transport quirks handled by a tool whose only job that is.

Deliberately shaped like the Ollama vendoring (verification/ollama.py):
build scripts drop the binary at vendor/go2rtc/<platform>/, the spec copies
it into the frozen bundle, `go2rtc_binary()` resolves bundled-then-PATH, and
everything degrades loudly-but-safely to the direct decode path when the
binary is missing or the process dies — go2rtc is an upgrade, never a
requirement. Every listener binds 127.0.0.1: go2rtc's API has no auth, and
the house rule is that no unauthenticated route to a frame exists ("tokens on
every frame route"). Exposing WebRTC beyond this box is a deliberate later
step with the Electron work, not a default.
"""
from __future__ import annotations

import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from cvti.logging_setup import get_logger

log = get_logger(__name__)


def bundled_binary() -> str | None:
    """The go2rtc binary shipped inside the app bundle, if present."""
    exe = "go2rtc.exe" if sys.platform == "win32" else "go2rtc"
    plat = {"win32": "windows", "darwin": "darwin"}.get(sys.platform, "linux")
    try:
        from cvti.utils import resource_path
        candidate = resource_path(f"vendor/go2rtc/{plat}/{exe}")
    except Exception:  # noqa: BLE001 - resource resolution is best-effort
        log.debug("go2rtc resource path failed", exc_info=True)
        return None
    return str(candidate) if candidate.exists() else None


PIDFILE = "go2rtc.pid"


def reap_stale(output_dir: str | Path) -> int:
    """Stop a go2rtc left behind by an engine that was killed, not stopped.

    Stop monitoring ends the engine with a hard terminate; on Windows that is
    TerminateProcess, so the engine's own clean-up never runs and the gateway
    it launched lives on. Each Stop/Start leaked one relay: the pilot server
    showed three go2rtc processes from three restarts (8 Oct 2026), each
    holding its camera connections. The gateway records its PID beside its
    config; this ends that process only if it is still go2rtc and still the
    one using this output directory's config, so a recycled PID is never
    touched. Returns how many it stopped. Best-effort, never raises."""
    pidfile = Path(output_dir) / PIDFILE
    # Whatever happens to the process, its advertisement must not outlive
    # it: a stale stream_gateway.json offers WebRTC on a dead port.
    try:
        (Path(output_dir) / "stream_gateway.json").unlink()
    except OSError:
        pass
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return 0
    stopped = 0
    try:
        import psutil
        proc = psutil.Process(pid)
        name = (proc.name() or "").lower()
        args = proc.cmdline()
        ours = os.path.normcase(os.path.abspath(str(Path(output_dir) / "go2rtc.yaml")))
        theirs = [os.path.normcase(os.path.abspath(a)) for a in args[args.index("-config") + 1:][:1]] \
            if "-config" in args else []
        if "go2rtc" in name and ours in theirs:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except psutil.TimeoutExpired:
                proc.kill()
            stopped = 1
            log.info("[go2rtc] stopped a gateway left over from a killed engine (pid %d)", pid)
    except Exception:  # noqa: BLE001 - gone already, access denied, or no psutil
        log.debug("stale go2rtc check skipped", exc_info=True)
    try:
        pidfile.unlink()
    except OSError:
        pass
    return stopped


def go2rtc_binary() -> str | None:
    """Bundled binary first, then PATH, else None (gateway disabled)."""
    import shutil
    return bundled_binary() or shutil.which("go2rtc")


def _sanitize(camera_id: str, taken: set) -> str:
    """A go2rtc stream name for a camera id: URL- and YAML-safe.

    Camera ids are display names ("Main Corridor") — fine as dict keys,
    hostile inside an rtsp:// path. Sanitized deterministically; collisions
    ("Door #1" / "Door 1" both -> door_1) get a numeric suffix rather than
    silently sharing a stream.
    """
    name = re.sub(r"[^A-Za-z0-9]+", "_", camera_id).strip("_").lower() or "cam"
    base, n = name, 2
    while name in taken:
        name = f"{base}_{n}"
        n += 1
    taken.add(name)
    return name


def restreamable(source: Any) -> bool:
    """Only RTSP cameras go through the gateway. Files and webcams are local
    — no session cap to consolidate, no transport to fix — and looping a demo
    clip is the decoder's existing job, not go2rtc's. HTTP(S) sources (HLS,
    MJPEG, a public feed) stay on the direct decode path too: go2rtc cannot
    pass those through, so it spawned ffmpeg and TRANSCODED them to H.264 to
    serve the engine an RTSP restream — one stream cost a laptop 44% CPU in
    the relay plus the engine's own decode (13 Oct 2026). The engine decodes
    HTTP sources itself; the wall shows them over MJPEG as before."""
    from cvti.serving.capture import is_live_source
    s = str(source).lower()
    return is_live_source(source) and (s.startswith("rtsp://") or s.startswith("rtsps://"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Go2rtcGateway:
    """Owns the go2rtc process: config, launch, health, restream URLs.

    Lifecycle mirrors the site: built from the cameras list, started before
    the decoders, stopped with the engine. `start()` returning False means
    "no gateway" — callers keep the direct decode path and say so in health,
    exactly like a missing Ollama runtime.
    """

    def __init__(self, cameras: list, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        taken: set = set()
        # {camera_id: (stream_name, source)} for every restreamable camera.
        self.streams: dict = {}
        # {camera_id: (sub_stream_name, detect_source)} — W1.6. A camera may
        # name its SUBSTREAM as `detect_source`: detection decodes those cheap
        # pixels while the mainstream keeps serving the wall (WebRTC
        # passthrough) and the full-resolution evidence snapshot. Never a
        # blanket downscale — the substream is the camera's own second encode.
        self.detect_streams: dict = {}
        for cam in cameras:
            src = cam.get("source")
            if src is None or not restreamable(src):
                continue
            self.streams[cam["id"]] = (_sanitize(cam["id"], taken), str(src))
            sub = cam.get("detect_source")
            if sub and restreamable(sub) and str(sub) != str(src):
                self.detect_streams[cam["id"]] = (
                    _sanitize(f"{cam['id']}_sub", taken), str(sub))
        self.api_port = 0
        self.rtsp_port = 0
        self.webrtc_port = 0
        self._proc: subprocess.Popen | None = None
        self._log_file = None
        self._lock = threading.Lock()
        self.started_at = 0.0
        self.restarts = 0
        self.disabled_reason = ""       # non-empty == not running, and why

    # --- config -------------------------------------------------------------

    def build_config(self) -> dict:
        """The go2rtc config document. Every listener is loopback-only: the
        API is unauthenticated by design upstream, and the house rule is no
        unauthenticated route to a frame. WebRTC beyond this box arrives
        with the Electron work, as a decision — not as a default."""
        return {
            "api": {"listen": f"127.0.0.1:{self.api_port}"},
            "rtsp": {"listen": f"127.0.0.1:{self.rtsp_port}"},
            "webrtc": {"listen": f"127.0.0.1:{self.webrtc_port}"},
            "log": {"level": "info"},
            "streams": {
                **{name: src for name, src in self.streams.values()},
                **{name: src for name, src in self.detect_streams.values()},
            },
        }

    def write_config(self) -> Path:
        import yaml
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / "go2rtc.yaml"
        target.write_text(yaml.safe_dump(self.build_config(), sort_keys=False))
        return target

    # --- lifecycle ----------------------------------------------------------

    def start(self, *, wait_ready_s: float = 8.0) -> bool:
        """Launch the gateway. False = disabled (reason in disabled_reason),
        and the caller keeps the direct decode path."""
        if not self.streams:
            self.disabled_reason = "no network cameras to restream"
            return False
        binary = go2rtc_binary()
        if binary is None:
            self.disabled_reason = "go2rtc binary not found (bundle or PATH)"
            log.warning("[go2rtc] %s — cameras stay on the direct decode path",
                        self.disabled_reason)
            return False
        # A bundled binary may lose its executable bit through packaging.
        try:
            os.chmod(binary, os.stat(binary).st_mode
                     | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
        # A previous engine that was killed rather than stopped leaves its
        # gateway running; end it before starting this one.
        reap_stale(self.output_dir)
        self.api_port = self.api_port or _free_port()
        self.rtsp_port = self.rtsp_port or _free_port()
        self.webrtc_port = self.webrtc_port or _free_port()
        config = self.write_config()
        # Its own log beside the engine's, and into the Diagnose zip's log
        # sweep — a gateway that died must explain itself in the support zip.
        self._log_file = open(self.output_dir / "go2rtc.log", "a")  # noqa: SIM115 - lives with the subprocess
        try:
            self._proc = subprocess.Popen(
                [binary, "-config", str(config)],
                stdout=self._log_file, stderr=subprocess.STDOUT)
        except OSError as exc:
            self.disabled_reason = f"go2rtc failed to launch: {exc}"
            log.warning("[go2rtc] %s", self.disabled_reason)
            self._close_log()
            return False
        if not self._wait_api(wait_ready_s):
            self.disabled_reason = f"go2rtc API not ready within {wait_ready_s:.0f}s"
            log.warning("[go2rtc] %s — falling back to direct decode",
                        self.disabled_reason)
            self.stop()
            return False
        try:
            (self.output_dir / PIDFILE).write_text(str(self._proc.pid))
        except OSError:
            log.debug("could not record the go2rtc pid", exc_info=True)
        self.started_at = time.time()
        self.disabled_reason = ""
        log.info("[go2rtc] gateway up: %d stream(s), rtsp restream :%d, "
                 "api :%d (loopback only)", len(self.streams),
                 self.rtsp_port, self.api_port)
        return True

    def _wait_api(self, budget_s: float) -> bool:
        """The process existing is not the gateway working — the API answering
        is. Same lesson as gate_health.json: liveness is demonstrated, never
        inferred from a PID."""
        deadline = time.monotonic() + budget_s
        url = f"http://127.0.0.1:{self.api_port}/api"
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False                      # died during startup
            try:
                with urllib.request.urlopen(url, timeout=1.0) as r:
                    if r.status == 200:
                        return True
            except OSError:
                time.sleep(0.2)
        return False

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def restream_url(self, camera_id: str) -> str | None:
        """The localhost RTSP URL the DETECTION decoder opens instead of the
        camera — or None (not a gateway camera / gateway down): caller uses
        the original source. None is the fallback, never an exception.

        A camera with a `detect_source` gets its SUBSTREAM here (W1.6):
        detection wants cheap pixels; the mainstream stays reserved for the
        wall and the evidence snapshot."""
        if not self.alive():
            return None
        entry = self.detect_streams.get(camera_id) or self.streams.get(camera_id)
        if entry is None:
            return None
        return f"rtsp://127.0.0.1:{self.rtsp_port}/{entry[0]}"

    def wall_stream_name(self, camera_id: str) -> str | None:
        """The MAINSTREAM's go2rtc name — what a WebRTC wall tile plays."""
        entry = self.streams.get(camera_id)
        return entry[0] if entry else None

    def snapshot_jpeg(self, camera_id: str, timeout: float = 2.0) -> bytes | None:
        """One full-resolution mainstream frame, on demand (W1.6 evidence).

        When detection rides the substream, alert evidence would otherwise be
        360p. go2rtc can hand back a mainstream frame without the engine
        decoding that stream at all. Only answers for cameras where detection
        is NOT already seeing the mainstream — everyone else's evidence is
        already full-resolution. Best-effort by contract: None, never a raise.
        """
        if camera_id not in self.detect_streams or not self.alive():
            return None
        name = self.wall_stream_name(camera_id)
        if name is None:
            return None
        try:
            url = f"http://127.0.0.1:{self.api_port}/api/frame.jpeg?src={name}"
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            return data if data[:2] == b"\xff\xd8" else None
        except OSError:
            return None

    def write_descriptor(self) -> Path | None:
        """stream_gateway.json beside frames.json — how the API learns the
        wall can speak WebRTC (W1.4). Written only while the gateway is up;
        removed on stop, so a stale file never advertises a dead gateway."""
        if not self.alive():
            return None
        doc = {
            "api_port": self.api_port,
            "rtsp_port": self.rtsp_port,
            "webrtc_port": self.webrtc_port,
            "streams": {cam_id: name for cam_id, (name, _s) in self.streams.items()},
            "generated_at": time.time(),
        }
        target = self.output_dir / "stream_gateway.json"
        try:
            target.write_text(json.dumps(doc, indent=1))
            return target
        except OSError:
            log.debug("stream_gateway.json write failed", exc_info=True)
            return None

    def api_streams(self) -> dict | None:
        """go2rtc's own view of its streams (consumer counts included), for
        health and the one-session-per-camera check. None = unreachable."""
        if not self.alive():
            return None
        try:
            url = f"http://127.0.0.1:{self.api_port}/api/streams"
            with urllib.request.urlopen(url, timeout=2.0) as r:
                return json.loads(r.read().decode())
        except (OSError, ValueError):
            return None

    def status(self) -> dict:
        """For gate_health.json: running or not, and when not, WHY."""
        return {
            "running": self.alive(),
            "streams": len(self.streams),
            "detect_substreams": len(self.detect_streams),
            "rtsp_port": self.rtsp_port if self.alive() else None,
            "api_port": self.api_port if self.alive() else None,
            "restarts": self.restarts,
            "disabled_reason": self.disabled_reason or None,
        }

    def _close_log(self) -> None:
        handle, self._log_file = self._log_file, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def stop(self) -> None:
        try:
            (self.output_dir / "stream_gateway.json").unlink()
        except OSError:
            pass                       # absent already — nothing advertised
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            (self.output_dir / PIDFILE).unlink()
        except OSError:
            pass
        self._close_log()
