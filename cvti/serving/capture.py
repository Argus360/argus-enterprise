"""One way to open a video source, on every platform (27 Aug).

Three call sites each opened captures their own way, and all three inherited
OpenCV's defaults — which differ per platform in ways that matter:

- **Buffering.** OpenCV queues frames internally. On a live source that queue
  IS latency: you decode the past while the present piles up behind it. The
  queue is deeper on Windows than macOS, which is why the same code feels
  laggier there. `CAP_PROP_BUFFERSIZE = 1` asks for the present.
- **Webcams on Windows** default to the MSMF backend, which routinely takes
  several seconds to open a camera and negotiates poor frame rates. DirectShow
  opens immediately and is what every Windows CV app uses.
- **Network streams** need explicit FFmpeg plus low-latency flags, or they
  buffer whole segments before yielding a frame (HLS measured at 22.5s to
  first frame, with stalls to 5.5s).

Everything here degrades safely: an unsupported property or backend falls back
to OpenCV's default rather than failing to open a camera.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Low-latency FFmpeg options for live network sources. `nobuffer` and
# `low_delay` stop the demuxer holding frames back; `reorder_queue_size=0`
# stops RTSP reordering adding a queue of its own; the TCP transport is what
# most cameras and every NAT actually work with.
_FFMPEG_LIVE_OPTS = (
    "rtsp_transport;tcp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|reorder_queue_size;0"
)


def is_live_source(source: Any) -> bool:
    """A network stream (rtsp/http/https) — as opposed to a file or a webcam."""
    s = str(source)
    return not s.isdigit() and "://" in s


def _hw_decode_params(cv2) -> list:
    """Ask FFmpeg to decode on the GPU when one exists (3 Sep, 'the RTSP is
    slow — I need it fast').

    A CPU-only pilot box decoding a 1080p25 RTSP main stream spends real
    cores on work its integrated GPU does for near-free: D3D11 on Windows,
    VideoToolbox on macOS, VAAPI on Linux. VIDEO_ACCELERATION_ANY negotiates
    whatever exists and falls back to software BY ITSELF, so this is a free
    upgrade on capable machines and a no-op elsewhere. Measured on the dev
    Mac: 0.51 -> 0.21 ms/frame on a test clip; the win on a Windows iGPU at
    1080p is an order of magnitude.
    """
    if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
        return [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
    return []          # ancient OpenCV: software decode, as before


def _live_decode_params(cv2) -> list:
    """Single-threaded decode for LIVE sources (7 Sep, found by the latency
    baseline's first run).

    OpenCV defaults the FFmpeg decoder to thread_count = CPU count, and
    H.264 frame-threading is a pipeline: output is delayed ~one frame per
    thread. On an 11-core dev Mac that measured 810ms of standing latency on
    a 15fps stream (11 x 66.7ms — the arithmetic is exact); on the pilot's
    4-core box it is ~250ms baked into every frame, alert evidence included.
    A camera stream is one frame at a time — the throughput frame-threading
    buys is worthless here, and the latency it costs is the product's whole
    complaint. threads=1 measured 810 -> 145ms p50; each extra thread adds
    exactly one frame period. Offline work keeps the default: an eval run
    wants throughput, and nobody is waiting at the glass.
    """
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        return [cv2.CAP_PROP_N_THREADS, 1]
    return []          # older OpenCV: no such knob, keep the old behaviour


# Set the first time the hardware-decode open fails and the software open
# succeeds: a box with no usable GPU (the pilot server) paid a failed D3D11
# attempt plus a warning on EVERY reconnect of every camera (13 Oct 2026).
_HW_DECODE_FAILED = False


def rtsp_alive(source: Any, timeout: float = 4.0) -> bool:
    """Is the camera reachable, without decoding anything: one TCP connect
    and one RTSP OPTIONS. Any RTSP reply (200, 401, 404...) means a camera
    is answering; no reply or a refused connection means it is not. For a
    non-RTSP network source, the TCP connect alone decides."""
    import socket
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(str(source))
        host, port = parts.hostname, parts.port or (554 if parts.scheme.startswith("rtsp") else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout) as sock:
            if not parts.scheme.startswith("rtsp"):
                return True
            bare = parts._replace(netloc=f"{host}:{port}").geturl()
            sock.sendall(f"OPTIONS {bare} RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: argus-engine/1\r\n\r\n".encode())
            sock.settimeout(timeout)
            reply = sock.recv(256)
            return reply.startswith(b"RTSP/")
    except (OSError, ValueError):
        return False


def ffmpeg_exe() -> "str | None":
    """The static ffmpeg the imageio-ffmpeg wheel ships (bundled in every
    build for evidence clips), else one on PATH, else None."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        log.debug("imageio-ffmpeg unavailable; looking for ffmpeg on PATH", exc_info=True)
    import shutil
    return shutil.which("ffmpeg")


# cv2.CAP_PROP_* values the keyframe capture answers, without importing cv2 here.
_PROP_FRAME_WIDTH, _PROP_FRAME_HEIGHT, _PROP_FPS = 3, 4, 5


class FFmpegKeyframeCapture:
    """A VideoCapture look-alike that decodes ONLY keyframes.

    OpenCV must decode every frame of a stream to hand over one, and on the
    pilot server seven 5-megapixel HEVC cameras decoded in full ate the whole
    box before detection saw a frame (13 Oct 2026). FFmpeg can skip
    everything but keyframes before decoding (`-skip_frame nokey`), one
    frame every GOP, typically one or two seconds, at a small fraction of
    the cost. OpenCV cannot pass that option through its capture options
    (measured: same frame count), so this runs ffmpeg itself and reads raw
    BGR frames from its stdout. Same methods the decoder uses on cv2:
    isOpened, grab, read, get, set, release. get(FPS) is 0: the decoder then
    paces by wall clock, which is right for frames that arrive per GOP.
    """
    keyframes_only = True

    def __init__(self, source: Any, *, binary: str, open_timeout_s: float = 15.0,
                 io_timeout_s: float = 10.0) -> None:
        import re
        import subprocess
        import threading
        self.source = str(source)
        self.width = 0
        self.height = 0
        self.codec = ""
        self._frame_bytes = 0
        self._proc = None
        self._sock = None
        self._buf = None
        self._stderr_tail: list = []
        self._size_ready = threading.Event()
        # Raw frames travel over a loopback TCP socket, not ffmpeg's stdout.
        # A 2592x1520 BGR frame is 12 MB; through an anonymous pipe on
        # Windows that arrives in kilobyte-sized reads, thousands of system
        # calls and context switches per frame, and on CHI's 5-vCPU server
        # one keyframe cost about a second of CPU to receive (18 Oct 2026,
        # frame_cpu_ms 800 to 1400 while the decode itself is 10 to 30 ms).
        # A socket with a large receive buffer takes the same frame in a
        # handful of recv_into calls straight into a reusable buffer.
        import socket
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(open_timeout_s)
        except OSError as exc:
            log.warning("[keyframes] could not open a loopback socket: %s", exc)
            listener.close()
            return
        port = listener.getsockname()[1]
        cmd = [binary, "-hide_banner", "-nostdin", "-loglevel", "info"]
        if self.source.lower().startswith("rtsp"):
            cmd += ["-rtsp_transport", "tcp", "-timeout", str(int(io_timeout_s * 1_000_000))]
        cmd += ["-skip_frame", "nokey", "-i", self.source, "-an", "-sn",
                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24",
                f"tcp://127.0.0.1:{port}"]
        flags = {}
        if sys.platform == "win32":
            flags["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE, bufsize=0, **flags)
        except OSError as exc:
            log.warning("[keyframes] could not start ffmpeg: %s", exc)
            listener.close()
            return
        self._listener = listener
        pattern = re.compile(r"Video: (\w+).*?,\s*(\d{2,5})x(\d{2,5})")

        def pump() -> None:
            # Keep stderr drained (ffmpeg blocks when it fills) and learn the
            # stream's size from its "Stream #0:0: Video: ..." line.
            try:
                for raw in iter(self._proc.stderr.readline, b""):
                    line = raw.decode("utf-8", "replace").rstrip()
                    self._stderr_tail = (self._stderr_tail + [line])[-8:]
                    if not self._size_ready.is_set():
                        m = pattern.search(line)
                        if m:
                            self.codec = m.group(1)
                            self.width, self.height = int(m.group(2)), int(m.group(3))
                            self._frame_bytes = self.width * self.height * 3
                            self._size_ready.set()
            except (OSError, ValueError):
                pass
            finally:
                self._size_ready.set()

        threading.Thread(target=pump, name="ffmpeg-stderr", daemon=True).start()
        if not self._size_ready.wait(open_timeout_s) or not self._frame_bytes:
            log.warning("[keyframes] %s: no video stream within %.0fs (%s)",
                        _redact(self.source), open_timeout_s, " | ".join(self._stderr_tail[-2:]))
            self.release()
            return
        # ffmpeg connects once it has opened the input; accept that one client.
        try:
            conn, _peer = listener.accept()
        except (OSError, socket.timeout) as exc:
            log.warning("[keyframes] %s: ffmpeg did not connect within %.0fs (%s)",
                        _redact(self.source), open_timeout_s, exc)
            self.release()
            return
        finally:
            listener.close()
            self._listener = None
        try:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        except OSError:
            log.debug("[keyframes] SO_RCVBUF not honoured", exc_info=True)
        conn.settimeout(max(io_timeout_s, 60.0))   # a GOP can be 12 s on CHI's cameras
        self._sock = conn
        self._buf = bytearray(self._frame_bytes)

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API
        return (self._proc is not None and self._sock is not None and self._frame_bytes > 0
                and self._proc.poll() is None)

    def _read_frame(self) -> bool:
        """Fill the frame buffer from the socket. False at end of stream."""
        if not self.isOpened():
            return False
        sock, buf = self._sock, self._buf
        view = memoryview(buf)
        want = self._frame_bytes
        got = 0
        try:
            while got < want:
                n = sock.recv_into(view[got:], want - got)
                if n <= 0:
                    return False
                got += n
        except (OSError, ValueError):
            return False
        return True

    def grab(self) -> bool:
        return self._read_frame()

    def read(self):
        if not self._read_frame():
            return False, None
        import numpy as np
        frame = np.frombuffer(self._buf, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame.copy()           # the buffer is reused by the next read

    def get(self, prop: int) -> float:
        if prop == _PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == _PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0                       # FPS unknown: one frame per GOP

    def set(self, *_args) -> bool:
        return False

    def interrupt(self) -> None:
        """Unblock a read from another thread (the decoder's stop())."""
        self.release()

    def release(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()                # unblocks a recv_into in another thread
            except OSError:
                pass
        listener = getattr(self, "_listener", None)
        if listener is not None:
            self._listener = None
            try:
                listener.close()
            except OSError:
                pass
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except Exception:  # noqa: BLE001 - then kill
                    proc.kill()
        except OSError:
            log.debug("[keyframes] terminate failed", exc_info=True)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is None:
                continue
            try:
                pipe.close()
            except OSError:
                pass


_NVDEC_BINARY: "str | None | bool" = False     # False = not probed yet
_NVDEC_FAILED_SOURCES: set = set()
import threading as _threading
# Every camera's decoder opens at once on start-up; without the lock the first
# caller's in-progress marker (None) read as "no GPU" to the rest, so only one
# camera got NVDEC and the others silently stayed on the CPU (22 Oct 2026).
_NVDEC_LOCK = _threading.Lock()
NVDEC_CODECS = {"h264": "h264_cuvid", "hevc": "hevc_cuvid"}


def nvdec_ffmpeg() -> "str | None":
    """An ffmpeg that can decode on an NVIDIA GPU (NVDEC/CUVID), or None.

    OpenCV's hardware-decode request means VAAPI/D3D11/VideoToolbox; pip
    OpenCV has no CUDA, so on a cloud NVIDIA box (CHI's T4) every camera was
    decoded on the CPU while NVDEC sat at 0% — four 4MP HEVC streams held at
    ~7.7 fps each, and falling behind real time stalled RTSP and dropped
    streams (22 Oct 2026). The system ffmpeg usually ships the *_cuvid
    decoders; the imageio static build does not, so it is not considered.
    Disable with ARGUS_NVDEC=0. Probed once per process."""
    with _NVDEC_LOCK:
        return _nvdec_ffmpeg_locked()


def _nvdec_ffmpeg_locked() -> "str | None":
    global _NVDEC_BINARY
    if _NVDEC_BINARY is not False:
        return _NVDEC_BINARY or None
    _NVDEC_BINARY = None
    if os.environ.get("ARGUS_NVDEC", "1").strip() in ("0", "false", "no", "off"):
        return None
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    # The decode library ships separately from the compute driver: Azure's
    # headless "-server" driver has CUDA but no libnvcuvid, so ffmpeg's cuvid
    # decoders fail to open ("Cannot load libnvcuvid.so.1"). Check it loads,
    # or every camera would wait out an ffmpeg timeout before falling back.
    import ctypes
    try:
        ctypes.CDLL("libnvcuvid.so.1")
    except OSError:
        log.info("[nvdec] NVIDIA GPU present but libnvcuvid is not installed (install "
                 "libnvidia-decode-<driver>-server); decoding on the CPU")
        return None
    binary = shutil.which("ffmpeg")
    if not binary:
        return None
    try:
        out = subprocess.run([binary, "-hide_banner", "-decoders"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if all(dec in out for dec in NVDEC_CODECS.values()):
        _NVDEC_BINARY = binary
        log.info("[nvdec] GPU video decode available via %s", binary)
    return _NVDEC_BINARY or None


def _nvdec_max_width() -> int:
    """Width NVDEC scales frames to on the GPU (0 = native). Detection runs at
    imgsz ~960 and the VLM's vision encoder at ~900, so 1600 keeps every pixel
    either uses while cutting what crosses into Python by ~2.6x versus 4MP."""
    try:
        return max(0, int(os.environ.get("ARGUS_NVDEC_MAX_WIDTH", "1600")))
    except ValueError:
        return 1600


def _probe_stream(source: str, binary: str, timeout_s: float = 12.0) -> "dict | None":
    """codec / width / height / fps of the first video stream, via ffprobe."""
    import json
    import shutil
    import subprocess
    probe = shutil.which("ffprobe") or os.path.join(os.path.dirname(binary), "ffprobe")
    cmd = [probe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
           "-of", "json"]
    if source.lower().startswith("rtsp"):
        cmd += ["-rtsp_transport", "tcp"]
    try:
        out = subprocess.run(cmd + [source], capture_output=True, text=True,
                             timeout=timeout_s).stdout
        stream = (json.loads(out or "{}").get("streams") or [{}])[0]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

    def rate(text: str) -> float:
        try:
            num, _, den = str(text).partition("/")
            return float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            return 0.0
    if not stream.get("width") or not stream.get("height"):
        return None
    return {"codec": str(stream.get("codec_name") or "").lower(),
            "width": int(stream["width"]), "height": int(stream["height"]),
            "fps": rate(stream.get("avg_frame_rate")) or rate(stream.get("r_frame_rate"))}


class FFmpegNvdecCapture(FFmpegKeyframeCapture):
    """Full-rate decode on the GPU's NVDEC, frames delivered like the keyframe
    capture (ffmpeg subprocess -> raw BGR over a loopback socket).

    Every frame is decoded (on the GPU, near-free) and optionally scaled down
    on the GPU too, then handed over as it arrives. `push_source` tells the
    decoder loop not to sleep between reads: ffmpeg pushes at the stream's own
    rate, and pausing the reader backs the socket up until ffmpeg stalls on
    its RTSP input — the same failure the keyframe path avoids. Detection
    still samples at its own rate downstream."""
    keyframes_only = False
    push_source = True

    def __init__(self, source: Any, *, binary: str, max_width: int = 1600,
                 open_timeout_s: float = 15.0, io_timeout_s: float = 10.0) -> None:
        import socket
        import subprocess
        import threading
        self.source = str(source)
        self.width = self.height = 0
        self.codec = ""
        self.fps = 0.0
        self._frame_bytes = 0
        self._proc = None
        self._sock = None
        self._buf = None
        self._listener = None
        self._stderr_tail: list = []
        self._size_ready = threading.Event()
        info = _probe_stream(self.source, binary)
        # Reachable-but-refused (codec, CUDA) is worth remembering; an
        # unreachable camera is not: it is the reconnect loop's job, not ours.
        self.probe_ok = info is not None
        if not info or info["codec"] not in NVDEC_CODECS:
            log.info("[nvdec] %s: %s — leaving it to the CPU decoder", _redact(self.source),
                     "probe failed" if not info else f"codec {info['codec']} not supported")
            return
        src_w, src_h = info["width"], info["height"]
        out_w, out_h = src_w, src_h
        if max_width and src_w > max_width:
            out_w = max_width - (max_width % 2)
            out_h = max(2, int(round(src_h * out_w / src_w / 2.0)) * 2)
        self.codec, self.fps = info["codec"], info["fps"]
        self.width, self.height = out_w, out_h
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(open_timeout_s)
        except OSError as exc:
            log.warning("[nvdec] could not open a loopback socket: %s", exc)
            listener.close()
            return
        port = listener.getsockname()[1]
        cmd = [binary, "-hide_banner", "-nostdin", "-loglevel", "error",
               "-fflags", "nobuffer", "-flags", "low_delay"]
        if self.source.lower().startswith("rtsp"):
            cmd += ["-rtsp_transport", "tcp", "-timeout", str(int(io_timeout_s * 1_000_000))]
        cmd += ["-c:v", NVDEC_CODECS[self.codec]]
        if (out_w, out_h) != (src_w, src_h):
            cmd += ["-resize", f"{out_w}x{out_h}"]          # scaled on the GPU, in the decoder
        cmd += ["-i", self.source, "-an", "-sn", "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "bgr24", f"tcp://127.0.0.1:{port}"]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE, bufsize=0)
        except OSError as exc:
            log.warning("[nvdec] could not start ffmpeg: %s", exc)
            listener.close()
            return
        self._listener = listener

        def pump() -> None:                     # keep stderr drained; remember the tail
            try:
                for raw in iter(self._proc.stderr.readline, b""):
                    line = raw.decode("utf-8", "replace").rstrip()
                    self._stderr_tail = (self._stderr_tail + [line])[-8:]
            except (OSError, ValueError, AttributeError):
                pass

        threading.Thread(target=pump, name="ffmpeg-nvdec-stderr", daemon=True).start()
        try:
            conn, _peer = listener.accept()
        except (OSError, socket.timeout) as exc:
            log.warning("[nvdec] %s: ffmpeg did not connect within %.0fs (%s; %s)",
                        _redact(self.source), open_timeout_s, exc,
                        " | ".join(self._stderr_tail[-2:]))
            self.release()
            return
        finally:
            listener.close()
            self._listener = None
        try:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        except OSError:
            log.debug("[nvdec] SO_RCVBUF not honoured", exc_info=True)
        conn.settimeout(max(io_timeout_s, 15.0))
        self._sock = conn
        self._frame_bytes = out_w * out_h * 3
        self._buf = bytearray(self._frame_bytes)
        log.info("[nvdec] %s: %s %dx%d@%.0f decoding on GPU%s", _redact(self.source),
                 self.codec, src_w, src_h, self.fps,
                 f", scaled to {out_w}x{out_h}" if (out_w, out_h) != (src_w, src_h) else "")

    def get(self, prop: int) -> float:
        if prop == _PROP_FPS:
            return float(self.fps)
        return super().get(prop)


def _redact(url: str) -> str:
    import re
    return re.sub(r"://[^@/]*@", "://***@", url)


def open_capture(source: Any, *, low_latency: bool = True, keyframes_only: bool = False):
    """Open `source` with the right backend and the least buffering.

    `low_latency=False` for offline work (evaluation over files) where every
    frame matters more than freshness.
    """
    import cv2

    global _HW_DECODE_FAILED
    src = int(source) if str(source).isdigit() else source
    if keyframes_only and not isinstance(src, int):
        binary = ffmpeg_exe()
        if binary:
            cap = FFmpegKeyframeCapture(src, binary=binary)
            if cap.isOpened():
                return cap
            log.warning("[keyframes] ffmpeg keyframe capture failed for %s; decoding in full",
                        _redact(str(src)))
        else:
            log.warning("[keyframes] no ffmpeg available; decoding %s in full", _redact(str(src)))
    if (low_latency and not keyframes_only and not isinstance(src, int)
            and is_live_source(src) and str(src) not in _NVDEC_FAILED_SOURCES):
        binary = nvdec_ffmpeg()
        if binary:
            cap = FFmpegNvdecCapture(src, binary=binary, max_width=_nvdec_max_width())
            if cap.isOpened():
                return cap
            cap.release()
            # Reachable but refused (unsupported codec, CUDA error): don't
            # re-probe on every reconnect; the CPU path keeps the camera working.
            # An unreachable camera is not remembered, so GPU decode resumes
            # once it is back.
            if getattr(cap, "probe_ok", False):
                _NVDEC_FAILED_SOURCES.add(str(src))
            log.warning("[nvdec] %s: GPU decode unavailable; using the CPU decoder",
                        _redact(str(src)))
    hw = [] if _HW_DECODE_FAILED else _hw_decode_params(cv2)
    live_params = _live_decode_params(cv2) if low_latency else []

    if isinstance(src, int):
        # Webcam. DirectShow on Windows: MSMF can take seconds to open and
        # often negotiates a worse mode. Everywhere else the default is right.
        # (No hw params: webcams hand over raw frames — nothing to decode.)
        if sys.platform == "win32":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():
                log.warning("DirectShow could not open webcam %s; falling back", src)
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
    elif is_live_source(src):
        if low_latency:
            # setdefault so an operator's explicit env var always wins.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FFMPEG_LIVE_OPTS)
        # FFmpeg applies these at open time, not through cap.set(). Bound
        # failed opens and reads so reconnect/preview shutdown can progress.
        timeouts = []
        for name, milliseconds in (("CAP_PROP_OPEN_TIMEOUT_MSEC", 8000),
                                   ("CAP_PROP_READ_TIMEOUT_MSEC", 5000)):
            if hasattr(cv2, name):
                timeouts.extend([getattr(cv2, name), milliseconds])
        params = hw + live_params + timeouts
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, params) if params \
            else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if hw and not cap.isOpened():
            # ANY should fall back internally; if a broken driver still
            # refuses the open, software decode beats no camera. Retry keeps
            # the single-thread ask — it is a latency fix, not an accelerator.
            cap.release()
            software_params = live_params + timeouts
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, software_params) if software_params \
                else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if cap.isOpened():
                _HW_DECODE_FAILED = True
                log.warning("hardware decode open failed; retrying software decode worked, "
                            "so software decode from now on (no further hardware attempts)")
            else:
                log.warning("network capture open failed with and without hardware decode")
    else:
        # A file: same free decode upgrade (demo clips are video too).
        cap = cv2.VideoCapture(src, cv2.CAP_ANY, hw) if hw else cv2.VideoCapture(src)
        if hw and not cap.isOpened():
            cap = cv2.VideoCapture(src)

    if low_latency and (isinstance(src, int) or is_live_source(src)):
        try:
            # Ask for a queue of one. Not every backend honours it; the
            # decoder's live-edge drain covers the ones that do not.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 - a tuning hint must never break capture
            log.debug("BUFFERSIZE not supported by this backend", exc_info=True)
    return cap
