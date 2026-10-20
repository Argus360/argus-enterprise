"""Alert sink + notifier (plan.md Phase 9 / P0).

Confirmed alerts in the serving path used to only `print()` and vanish. This
persists every CONFIRMED alert to a SQLite event store WITH an evidence bundle
(the frames the gate saw + a metadata.json), and pushes a notification
(console / webhook / Telegram). Wire it into `GatePool(on_verdict=sink.handle)`.

Stdlib + cv2 only (sqlite3, urllib) so it runs on an offline edge box with no
extra pip deps.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from cvti.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Notifiers — each exposes notify(event: dict) -> None
# ---------------------------------------------------------------------------

def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Build a multipart/form-data body (stdlib only). files: {name: (filename, bytes)}."""
    boundary = "----cvti" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    for name, (filename, data) in files.items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: image/jpeg\r\n\r\n').encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"

class ConsoleNotifier:
    def notify(self, event: dict) -> None:
        log.info(f"[NOTIFY] {event['priority'].upper()} {event['rule']} on {event['camera_id']} "
              f"(conf {event['confidence']:.2f}) — evidence: {event['evidence_dir']}")


class WebhookNotifier:
    """POST the event as JSON to any webhook (Slack/Discord/custom endpoint)."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout

    def notify(self, event: dict) -> None:
        import urllib.request
        text = (f"⚠️ {event['priority'].upper()} — {event['rule']} on "
                f"{event['camera_id']} (conf {event['confidence']:.2f}): {event['reason']}")
        payload = {"text": text, "event": event}
        try:
            req = urllib.request.Request(
                self.url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 - a notify failure must not kill the gate
            log.error(f"[notify webhook error] {str(exc)[:120]}", exc_info=True)


class TelegramNotifier:
    """Send to a Telegram chat via the bot API (token + chat_id). Attaches the
    evidence frames as a photo album so the alert arrives WITH pictures.

    Paced and retried (16 Sep): every multi-camera demo run lost alerts to
    `HTTP 429 Too Many Requests` — five looping cameras fire in bursts, each
    alert is two uploads, and Telegram allows roughly one message a second to
    a person and twenty a minute to a group. Calls to one chat are now
    serialized with a minimum gap, and a 429 waits the `retry_after` Telegram
    names and tries again instead of dropping the alert. In production
    (`background=True`, set by build_notifier) delivery runs on its own
    thread so the gate's verdict callback never waits on an upload; the
    default stays synchronous for callers and tests that expect it.
    """

    # Telegram's published limits: ~1 message/second to a private chat,
    # 20 messages/minute to a group. A little margin on each.
    GAP_PRIVATE = 1.1
    GAP_GROUP = 3.1
    MAX_RETRIES = 3
    MAX_RETRY_WAIT = 30.0
    QUEUE_MAX = 200

    def __init__(self, token: str, chat_id: str, timeout: float = 8.0, *,
                 background: bool = False) -> None:
        self.base = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id
        self.timeout = timeout
        self.min_gap = self.GAP_GROUP if str(chat_id).startswith("-") else self.GAP_PRIVATE
        self._lock = threading.Lock()          # one API call at a time per chat
        self._last_call = 0.0
        self.retried = 0
        self.dropped = 0
        self._queue = None
        self._worker: threading.Thread | None = None
        if background:
            import queue
            self._queue = queue.Queue(maxsize=self.QUEUE_MAX)
            self._worker = threading.Thread(target=self._drain, name=f"telegram-{chat_id}",
                                            daemon=True)
            self._worker.start()

    # --- pacing + retry --------------------------------------------------------

    def _call(self, req: Any, timeout: float | None = None) -> Any:
        """One Telegram API call: paced to the chat's rate, retried on 429."""
        import urllib.error
        import urllib.request
        timeout = self.timeout if timeout is None else timeout
        for attempt in range(self.MAX_RETRIES + 1):
            with self._lock:
                gap = self.min_gap - (time.monotonic() - self._last_call)
                if gap > 0:
                    time.sleep(gap)
                try:
                    resp = urllib.request.urlopen(req, timeout=timeout)
                    self._last_call = time.monotonic()
                    return resp
                except urllib.error.HTTPError as exc:
                    self._last_call = time.monotonic()
                    if exc.code != 429 or attempt >= self.MAX_RETRIES:
                        raise
                    wait = self._retry_after(exc, default=2.0 * (attempt + 1))
                    self.retried += 1
                    log.info(f"[notify telegram] 429 for chat {self.chat_id}; "
                             f"retrying in {wait:.0f}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                time.sleep(wait)   # outside the lock: the other chat may proceed
        raise RuntimeError("unreachable")   # pragma: no cover

    def _retry_after(self, exc: Any, default: float) -> float:
        """Telegram says how long to wait, in the JSON body (`parameters.
        retry_after`) and/or a Retry-After header. Fall back to a short backoff."""
        wait = None
        try:
            body = exc.read()
            if body:
                wait = json.loads(body).get("parameters", {}).get("retry_after")
        except Exception:  # noqa: BLE001 - body is optional
            log.debug("[notify telegram] 429 body unreadable", exc_info=True)
        if wait is None:
            try:
                hdr = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
                wait = float(hdr) if hdr else None
            except (TypeError, ValueError):
                wait = None
        return min(self.MAX_RETRY_WAIT, float(wait) if wait is not None else default)

    # --- background delivery ---------------------------------------------------

    def _drain(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                self._queue.task_done()
                return
            try:
                self._deliver(event)
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for queued alerts to go out (engine shutdown)."""
        if self._queue is None:
            return
        deadline = time.monotonic() + timeout
        while not self._queue.empty() or self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                log.warning(f"[notify telegram] {self._queue.qsize()} alert(s) still queued "
                            f"for chat {self.chat_id} at shutdown")
                return
            time.sleep(0.05)

    def close(self) -> None:
        self.flush()
        if self._queue is not None and self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=2.0)

    def _caption(self, event: dict) -> str:
        link = event.get("link")
        base = self._caption_base(event)
        return f"{base}\n\u27a1 Respond: {link}" if link else base

    def _caption_base(self, event: dict) -> str:
        return (f"⚠️ {event['priority'].upper()} — {event['rule']} on "
                f"{event['camera_id']} (conf {event['confidence']:.2f})\n{event['reason']}")

    def _frames(self, event: dict, cap: int = 3) -> list[Path]:
        ev = event.get("evidence_dir")
        if not ev:
            return []            # no evidence yet (provisional): not the cwd
        d = Path(ev)
        if not d.exists():
            return []
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        # Lead with the annotated subject shot when there is one — on a phone the
        # first picture is what gets looked at, and it should point at the person.
        subject = [p for p in imgs if p.name == "subject.jpg"]
        rest = [p for p in imgs if p.name != "subject.jpg"]
        return (subject + rest)[:cap]

    def _clip(self, event: dict, cap_mb: float = 45.0) -> "Path | None":
        """The event's evidence clip, if one was written and Telegram will take it.

        Bot API uploads cap at 50MB; stay under it with margin. No clip (a
        provisional alert, clip disabled) just means photos-only, as before.
        """
        ev = event.get("evidence_dir")
        if not ev:
            return None
        clip = Path(ev) / "clip.mp4"
        try:
            if clip.exists() and clip.stat().st_size <= cap_mb * 1024 * 1024:
                return clip
        except OSError:
            return None
        return None

    def notify(self, event: dict) -> None:
        if self._queue is not None:
            import queue
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                # A burst deeper than QUEUE_MAX alerts is a runaway feed, not a
                # backlog worth an hour of uploads. Drop, count, say so.
                self.dropped += 1
                log.warning(f"[notify telegram] queue full ({self.QUEUE_MAX}) for chat "
                            f"{self.chat_id}; dropped {event.get('rule')} on {event.get('camera_id')}")
            return
        self._deliver(event)

    def _deliver(self, event: dict) -> None:
        """Send one alert, and SAY SO. Success used to be completely silent,
        so a log showed the console notifier's [NOTIFY] line either way and
        there was no way to tell a delivered alert from one that died in the
        queue. 20 Sep: hours of runs were reported as 'sent' on the strength
        of that line, while a stale build dropped every message on exit.
        Delivery is the product; it gets a log line."""
        import urllib.parse
        import urllib.request
        frames = self._frames(event)
        clip = self._clip(event)
        what = f"{event.get('rule')} on {event.get('camera_id')}"
        try:
            if not frames and not clip:
                data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": self._caption(event)}).encode()
                self._call(urllib.request.Request(f"{self.base}/sendMessage", data=data))
                log.info(f"[notify telegram] delivered {what} (text only) to chat {self.chat_id}")
                return
            if frames:
                self._send_photos(frames, self._caption(event))
            if clip:
                # The clip follows the album ('i need the videos landing on the
                # group chat', 12 Sep) — its own message, so a slow video upload
                # can never delay the photos that page someone.
                self._send_video(clip, self._caption_base(event))
            log.info(f"[notify telegram] delivered {what} to chat {self.chat_id} "
                     f"({len(frames)} photo(s){', video' if clip else ''})")
        except Exception as exc:  # noqa: BLE001 - a notify failure must not kill the gate
            log.error(f"[notify telegram error] {what}: {str(exc)[:140]}", exc_info=True)

    def _send_video(self, clip: "Path", caption: str) -> None:
        import urllib.request
        fields = {"chat_id": self.chat_id, "caption": f"🎬 {caption}",
                  "supports_streaming": "true"}
        files = {"video": (clip.name, clip.read_bytes())}
        body, ctype = _multipart(fields, files)
        req = urllib.request.Request(f"{self.base}/sendVideo", data=body,
                                     headers={"Content-Type": ctype})
        # video uploads are heavier than photos; give them a longer leash
        self._call(req, timeout=max(self.timeout, 30.0))

    def _send_photos(self, frames: list[Path], caption: str) -> None:
        """One photo -> sendPhoto; several -> sendMediaGroup (album)."""
        import urllib.request
        if len(frames) == 1:
            fields = {"chat_id": self.chat_id, "caption": caption}
            files = {"photo": (frames[0].name, frames[0].read_bytes())}
            body, ctype = _multipart(fields, files)
            req = urllib.request.Request(f"{self.base}/sendPhoto", data=body, headers={"Content-Type": ctype})
            self._call(req)
            return
        media, files = [], {}
        for i, fr in enumerate(frames):
            key = f"photo{i}"
            item = {"type": "photo", "media": f"attach://{key}"}
            if i == 0:
                item["caption"] = caption          # caption goes on the first item
            media.append(item)
            files[key] = (fr.name, fr.read_bytes())
        body, ctype = _multipart({"chat_id": self.chat_id, "media": json.dumps(media)}, files)
        req = urllib.request.Request(f"{self.base}/sendMediaGroup", data=body, headers={"Content-Type": ctype})
        self._call(req)


class WhatsAppNotifier:
    """Send to WhatsApp via Twilio's API. Credentials come from env vars (never
    committed): TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM
    (defaults to the Twilio sandbox number), WHATSAPP_TO (e.g. +234...)."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str,
                 to_number: str, timeout: float = 6.0) -> None:
        import base64
        self.url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        self.auth = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
        self.from_ = from_number if from_number.startswith("whatsapp:") else f"whatsapp:{from_number}"
        self.to = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "WhatsAppNotifier":
        import os
        sid = os.environ.get("TWILIO_ACCOUNT_SID")
        tok = os.environ.get("TWILIO_AUTH_TOKEN")
        frm = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")  # Twilio sandbox
        to = os.environ.get("WHATSAPP_TO")
        if not (sid and tok and to):
            raise RuntimeError("WhatsApp needs TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, WHATSAPP_TO env vars")
        return cls(sid, tok, frm, to)

    def notify(self, event: dict) -> None:
        import urllib.parse
        import urllib.request
        text = (f"⚠️ {event['priority'].upper()} — {event['rule']} on "
                f"{event['camera_id']} (conf {event['confidence']:.2f})\n{event['reason']}")
        data = urllib.parse.urlencode({"From": self.from_, "To": self.to, "Body": text}).encode()
        try:
            req = urllib.request.Request(self.url, data=data,
                                         headers={"Authorization": f"Basic {self.auth}"}, method="POST")
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            log.error(f"[notify whatsapp error] {str(exc)[:140]}", exc_info=True)


class MultiNotifier:
    """Fan an alert out to several notifiers; one failing never blocks the rest."""
    def __init__(self, notifiers: list) -> None:
        self.notifiers = notifiers

    def notify(self, event: dict) -> None:
        for n in self.notifiers:
            try:
                n.notify(event)
            except Exception as exc:  # noqa: BLE001
                log.error(f"[alert-sink] {type(n).__name__} failed: {str(exc)[:100]}", exc_info=True)

    def close(self) -> None:
        """Let every channel finish what it has queued (engine shutdown)."""
        for n in self.notifiers:
            fn = getattr(n, "close", None)
            if callable(fn):
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001
                    log.error(f"[alert-sink] {type(n).__name__} close failed: {str(exc)[:100]}",
                              exc_info=True)


def _build_one(spec: str) -> Any:
    spec = spec.strip()
    if not spec or spec == "console":
        return ConsoleNotifier()
    if spec.startswith("webhook:"):
        return WebhookNotifier(spec[len("webhook:"):])
    if spec.startswith("telegram:"):
        # A real bot token CONTAINS a colon (123456789:AAH...), so the chat id
        # is the piece after the LAST colon — split(':', 2) handed half the
        # token to chat_id and no message ever left the building. Found the
        # first time a real token was wired (11 Sep).
        token, _, chat_id = spec[len("telegram:"):].rpartition(":")
        # Production delivery is paced + retried on its own thread; the gate's
        # verdict callback must never wait on a photo upload or a 429 backoff.
        return TelegramNotifier(token, chat_id, background=True)
    if spec.startswith("whatsapp:"):
        # whatsapp:<account_sid>:<auth_token>:<from>:<to>. The UI writes this so
        # an installed customer can set WhatsApp up at all -- before, the only
        # source was environment variables, which nobody can set on a machine
        # that runs an installer, so ticking WhatsApp silently did nothing
        # (20 Sep). Bare "whatsapp" still reads the env, for the dev path.
        fields = spec[len("whatsapp:"):].split(":")
        if len(fields) == 4 and all(f.strip() for f in fields):
            sid, token, from_number, to_number = (f.strip() for f in fields)
            return WhatsAppNotifier(sid, token, from_number, to_number)
        log.warning("[alert-sink] whatsapp spec needs sid:token:from:to; using console")
        return ConsoleNotifier()
    if spec.startswith("email:"):
        # email:<url-encoded key=value pairs> — see cvti/serving/notify_email.py.
        # URL-encoding keeps a password's colon or a recipient's comma out of
        # this colon- and comma-delimited string.
        from cvti.serving.notify_email import EmailNotifier
        notifier = EmailNotifier.from_spec(spec[len("email:"):], background=True)
        if notifier is not None:
            return notifier
        log.warning("[alert-sink] email spec needs host, from and at least one to; using console")
        return ConsoleNotifier()
    if spec == "whatsapp":
        try:
            return WhatsAppNotifier.from_env()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[alert-sink] whatsapp unavailable ({exc}); using console", exc_info=True)
            return ConsoleNotifier()
    log.warning(f"[alert-sink] unknown notifier '{spec}', using console")
    return ConsoleNotifier()


def build_notifier(spec: str) -> Any:
    """spec: one channel, or several comma-separated (e.g. 'console,whatsapp').
    Each channel: 'console' | 'webhook:<url>' | 'telegram:<token>:<chat_id>' |
    'whatsapp' (Twilio creds from env) | 'email:<url-encoded smtp settings>'."""
    parts = [s for s in (spec or "console").split(",") if s.strip()]
    if len(parts) <= 1:
        return _build_one(parts[0] if parts else "console")
    return MultiNotifier([_build_one(p) for p in parts])


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, iso TEXT, camera_id TEXT, rule TEXT, priority TEXT,
    confidence REAL, reason TEXT, track_id INTEGER, zone TEXT,
    object_label TEXT, evidence_dir TEXT,
    review TEXT,          -- NULL=new, 'ack', 'true', 'false' (operator label)
    reviewed_at TEXT,
    latency_s REAL,       -- detection -> TrueSight-verified wall-clock seconds
    bbox TEXT,            -- "x1,y1,x2,y2" of the subject when it fired (may be NULL)
    unverified INTEGER DEFAULT 0,  -- 1 = the gate never reached a verdict (fail-visible)
    gate_error TEXT,      -- why, when unverified
    legal_hold INTEGER DEFAULT 0,  -- 1 = exempt from retention purge, set by an operator
    state TEXT DEFAULT 'new',      -- new -> acknowledged -> resolved (cvti/triage.py)
    owner TEXT,                    -- who claimed it; visible to every operator
    acknowledged_at REAL,
    resolved_at REAL,
    outcome TEXT,                  -- real | false_alarm | inconclusive
    note TEXT,                     -- free text captured at resolution
    provisional INTEGER DEFAULT 0, -- 1 = shown before verification (critical tier)
    retracted INTEGER DEFAULT 0    -- 1 = provisional later verified as NOT a threat
);

-- What the gate threw away, per day. Only confirmed alerts become rows in
-- `events`, so without this the product's central claim — "raw detectors would
-- have shown you 201 alerts, we showed you 28" — has no persistent evidence
-- behind it once the engine restarts.
CREATE TABLE IF NOT EXISTS suppression_daily (
    day TEXT PRIMARY KEY,        -- YYYY-MM-DD, local time
    shown INTEGER DEFAULT 0,     -- verified, confirmed, put in front of a human
    rejected INTEGER DEFAULT 0,  -- verified and thrown away
    deduped INTEGER DEFAULT 0,   -- same event already queued; never verified
    errors INTEGER DEFAULT 0,    -- the gate failed; counted, never hidden
    updated_at REAL
);

-- Every generated concealment candidate, including rows the queue suppresses
-- or evicts before TrueSight. RetentionManager purges this operational evidence
-- on the same site policy as event evidence.
CREATE TABLE IF NOT EXISTS concealment_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version INTEGER NOT NULL DEFAULT 2,
    generated_at REAL NOT NULL,
    camera_id TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    candidate_timestamp REAL NOT NULL,
    track_id INTEGER,
    destination TEXT,
    peak_score REAL,
    components_json TEXT NOT NULL DEFAULT '{}',
    limited INTEGER,
    associated_bag_json TEXT,
    reasons_json TEXT NOT NULL DEFAULT '[]',
    enqueued_at REAL,
    admission_status TEXT NOT NULL DEFAULT 'generated',
    admitted_at REAL,
    verdict_at REAL,
    gate_result_timestamp REAL,
    gate_latency_s REAL,
    verdict TEXT,
    confirmed INTEGER,
    confidence REAL,
    reason TEXT,
    gate_error TEXT,
    prompt_version TEXT,
    audit_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_concealment_audit_generated
    ON concealment_audit(generated_at);

-- Scenario-5's acceptance scorer reads these six leading fields directly.
-- Extra columns retain the runtime lifecycle without changing alert policy.
CREATE TABLE IF NOT EXISTS motion_candidate_audit (
    candidate_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    timestamp_s REAL NOT NULL,
    admission_status TEXT NOT NULL DEFAULT 'generated',
    gate_status TEXT NOT NULL DEFAULT 'pending',
    persisted_event_id TEXT NOT NULL DEFAULT '',
    generated_at REAL NOT NULL,
    camera_id TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    track_ids_json TEXT NOT NULL DEFAULT '[]',
    admitted_at REAL,
    verdict_at REAL,
    confidence REAL,
    reason TEXT,
    gate_error TEXT,
    prompt_version TEXT,
    persistence_status TEXT NOT NULL DEFAULT 'pending',
    persistence_error TEXT,
    user_event_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_motion_candidate_audit_generated
    ON motion_candidate_audit(generated_at);

CREATE TABLE IF NOT EXISTS object_watch_audit (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    object_label TEXT NOT NULL,
    state TEXT NOT NULL,
    generated_at REAL NOT NULL,
    timestamp_s REAL NOT NULL,
    zone TEXT,
    similarity REAL,
    admission_status TEXT NOT NULL DEFAULT 'generated',
    admitted_at REAL,
    gate_status TEXT NOT NULL DEFAULT 'pending',
    verdict_at REAL,
    persisted_event_id TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_object_watch_audit_generated
    ON object_watch_audit(generated_at);
"""


def _ffmpeg_exe() -> "str | None":
    """The static ffmpeg imageio-ffmpeg ships, or None if the wheel is absent."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency; the clip still exists
        log.debug("imageio-ffmpeg unavailable; evidence clips keep OpenCV's codec", exc_info=True)
        return None


def ensure_h264(path: "Path", timeout: float = 60.0) -> str:
    """Re-encode an evidence clip to H.264 so a browser can play it. Returns the
    codec the file ends up in ("h264", or the fourcc OpenCV managed).

    OpenCV tries avc1 first, but the opencv-python-headless wheel that every
    build installs carries NO H.264 encoder on Windows, so the writer falls
    back to mp4v -- MPEG-4 Part 2 -- which Chromium (and therefore the Electron
    renderer) cannot decode in <video>. On a Mac the OS supplies an encoder, so
    clips played on every developer machine and on none of the pilot's
    ("the replay videos weren't playing", Windows, 21 Sep). Telegram's own
    player was more forgiving, which is why phones showed video and the app
    did not.

    imageio-ffmpeg bundles a static ffmpeg per platform and PyInstaller has a
    hook for it, so the fix is one transcode, atomically replacing the file.
    """
    import subprocess
    codec = _clip_codec(path)
    if codec == "h264":
        return codec
    exe = _ffmpeg_exe()
    if not exe:
        log.warning("[evidence] clip %s is %s and no ffmpeg is bundled; "
                    "the app's player may not decode it", path.name, codec)
        return codec
    tmp = path.with_suffix(".h264.tmp.mp4")
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(path),
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        tmp.replace(path)
        return "h264"
    except Exception as exc:  # noqa: BLE001 - keep the playable-somewhere original
        log.warning("[evidence] H.264 transcode of %s failed: %s", path.name, str(exc)[:120])
        try:
            tmp.unlink()
        except OSError:
            log.debug("could not remove %s after a failed transcode", tmp.name, exc_info=True)
        return codec


def probe_clip_encoder(dest_dir: Path | str) -> dict:
    """Write a synthetic evidence clip exactly the way the sink does and report
    what came out. The one question that could not be answered from a Mac:
    does the SHIPPED Windows engine produce an H.264 clip the app can play?

    Returns {"codec", "ffmpeg", "path"}. codec == "h264" is the pass; anything
    else means the bundle's ffmpeg did not run and replays will be black boxes
    in the app (pilot, 21-22 Sep). Runs in seconds, needs no camera or model,
    so CI can ask the real binary on every build (`--check-clip-encoder`)."""
    import cv2
    import numpy as np
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "clip_probe.mp4"
    h, w = 96, 160
    written = None
    for fourcc in ("avc1", "mp4v"):        # the same preference order as evidence
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), 24, (w, h))
        if vw.isOpened():
            for i in range(24):
                im = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.rectangle(im, (i * 5, 20), (i * 5 + 30, 70), (0, 200, 255), -1)
                vw.write(im)
            vw.release()
            written = fourcc
            break
    if written is None or not path.exists():
        return {"codec": "none", "ffmpeg": _ffmpeg_exe(), "path": str(path)}
    codec = ensure_h264(path)
    return {"codec": codec, "ffmpeg": _ffmpeg_exe(), "path": str(path)}


def _clip_codec(path: "Path") -> str:
    """What OpenCV wrote, read back from the container's fourcc."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        raw = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()
        tag = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
        return {"avc1": "h264", "H264": "h264", "h264": "h264",
                "mp4v": "mp4v", "FMP4": "mp4v"}.get(tag, tag.lower() or "unknown")
    except Exception:  # noqa: BLE001 - an unreadable container is reported, not raised
        log.debug("could not read the codec of %s", path, exc_info=True)
        return "unknown"



class AlertSink:
    """Persist + notify CONFIRMED alerts. `handle(alert, result)` is the
    GatePool on_verdict callback (also prints every verdict, confirmed or not)."""

    def __init__(self, output_dir: str | Path, *, notifier: Any = None,
                 save_evidence: bool = True, iso_now: str | None = None,
                 routing_path: str | Path | None = "configs/routing.json") -> None:
        self.root = Path(output_dir)
        self.events_dir = self.root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "events.db"
        self.notifier = notifier or ConsoleNotifier()
        self.save_evidence = save_evidence
        # W1.6: camera_id -> full-resolution JPEG bytes | None. When detection
        # rides a camera's substream, evidence frames are substream-sized; this
        # callable (the gateway's mainstream snapshot) lets each alert also
        # carry ONE frame at the camera's full resolution. Best-effort and
        # None-safe by contract — evidence never waits on it and never fails
        # because of it.
        self.full_frame_provider = None
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        # Migrate older DBs in place. ONE column list, shared with the
        # console's migration (cvti.triage.EVENT_COLUMNS) — the engine's and
        # the console's lists drifted once, and the console crashed on a
        # column only this side knew to add (2 Sep, 'no such column:
        # retracted' on the Now screen).
        from cvti.triage import ensure_columns
        ensure_columns(self._db)
        self.persisted = 0
        # Routing: send each alert to the channels its rule names (severity, camera,
        # time-of-day…), falling back to the site-wide notifier. Escalation re-sends
        # anything still unacknowledged after the rule's deadline.
        from cvti.serving.routing import EscalationTracker, RoutingPolicy
        self.routing = RoutingPolicy.load(routing_path) if routing_path else RoutingPolicy()
        self.escalations = EscalationTracker(is_acknowledged=self._is_acknowledged)
        self._notifier_cache: dict = {}
        self.routed = 0
        self.escalated = 0
        if self.routing.rules:
            log.info(f"[routing] {len(self.routing.rules)} rule(s) loaded "
                  f"(default: {self.routing.default})")
        # Feedback loop: a calibration file (written by the FeedbackManager from
        # operator labels) tells us which (camera, rule) pairs are chronically wrong.
        # Demoted pairs are still stored (so the operator keeps correcting them) but
        # don't page anyone. Hot-reloaded when the file changes.
        from cvti.feedback.calibration import Calibration
        # When the mobile response view is up, every notification carries a
        # deep-link so the phone that receives the alert can also act on it.
        self.mobile_base = ""
        # Latency per priority tier (detection -> the operator could see it),
        # and detection -> provisional for the critical fast path. Published in
        # /health: "we alert critical threats in under a second" is a claim
        # that has to carry its measurement.
        from collections import deque
        self._tier_latency: dict = {}
        self._provisional_latency = deque(maxlen=50)
        self._calib_path = self.root / "calibration.json"
        self._calib_mtime = 0.0
        self.calibration = Calibration()
        self._reload_calibration()

    # --- suppression ledger ----------------------------------------------
    def record_suppression(self, *, shown: int = 0, rejected: int = 0,
                           deduped: int = 0, errors: int = 0, day: str | None = None) -> None:
        """Add a delta to today's suppression row.

        Deltas, not totals: the gate pool's counters reset with the process, and
        a day's figure has to survive a restart to be worth quoting.
        """
        if not (shown or rejected or deduped or errors):
            return
        day = day or time.strftime("%Y-%m-%d")
        with self._lock:
            self._db.execute(
                "INSERT INTO suppression_daily (day, shown, rejected, deduped, errors, updated_at) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(day) DO UPDATE SET "
                "shown=shown+excluded.shown, rejected=rejected+excluded.rejected, "
                "deduped=deduped+excluded.deduped, errors=errors+excluded.errors, "
                "updated_at=excluded.updated_at",
                (day, shown, rejected, deduped, errors, time.time()))
            self._db.commit()

    def _reload_calibration(self) -> None:
        from cvti.feedback.calibration import Calibration
        try:
            mtime = self._calib_path.stat().st_mtime
        except OSError:
            return
        if mtime != self._calib_mtime:
            self.calibration = Calibration.load(self._calib_path)
            self._calib_mtime = mtime
            demoted = self.calibration.demoted_keys()
            if demoted:
                log.info(f"[calibration] loaded — demoting (no page): {', '.join(demoted)}")

    # --- two-tier critical alerting (EP-06-T4) -----------------------------
    def provisional(self, alert: Any) -> int | None:
        """Persist + notify a critical candidate IMMEDIATELY, before the gate.

        Detection is ~163ms; the 20s latency is entirely verification. For
        theft that trade is right; for a weapon or a fire it is not. The
        operator gets a provisional warning in under a second, and the verdict
        updates this same row ~20s later — confirmation, or an explicit
        retraction. Never silence.
        """
        ts = time.time()
        payload = alert.payload or {}
        try:
            with self._lock:
                cur = self._db.execute(
                    "INSERT INTO events (ts, iso, camera_id, rule, priority, "
                    "confidence, reason, track_id, zone, object_label, provisional, "
                    "unverified) VALUES (?,?,?,?,?,?,?,?,?,?,1,1)",
                    (ts, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
                     alert.camera_id, alert.rule_name, alert.priority, 0.0,
                     "PROVISIONAL — the detector fired; verification is in "
                     "progress. Treat as unconfirmed.",
                     alert.track_id, alert.zone, alert.object_label))
                event_id = cur.lastrowid
                self._db.commit()
        except Exception:  # noqa: BLE001 - the fast path must never kill detection
            log.error("provisional persist failed", exc_info=True)
            return None
        enq = payload.get("enqueued_at")
        if isinstance(enq, (int, float)):
            self._provisional_latency.append(ts - enq)
        event = {"ts": ts, "iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "id": event_id,
                 "camera_id": alert.camera_id, "rule": alert.rule_name,
                 "priority": alert.priority, "confidence": 0.0, "zone": alert.zone,
                 "track_id": alert.track_id, "object_label": alert.object_label,
                 "reason": "⚠️ PROVISIONAL (unconfirmed) — verification in progress, "
                           "verdict follows in ~20s.",
                 "evidence_dir": None}
        if self.mobile_base:
            event["link"] = f"{self.mobile_base}/alert/{event_id}"
        self._dispatch(event, event_id)
        log.warning("[PROVISIONAL] %s :: %s (%s) — shown unverified, verdict pending",
                    alert.camera_id, alert.rule_name, alert.priority.upper())
        return event_id

    def _settle_provisional(self, event_id: int, alert: Any, result: Any) -> None:
        """The verdict arrives: update the provisional row IN PLACE."""
        ts = time.time()
        payload = alert.payload or {}
        enq = payload.get("enqueued_at")
        latency = round(ts - enq, 2) if isinstance(enq, (int, float)) else None
        errored = bool(getattr(result, "errored", False))
        if result.confirmed or errored:
            # Confirmed (or fail-visible unverified): the provisional was right
            # to show. Evidence is written now, exactly as a normal confirm.
            ev_dir = None
            frames = payload.get("frames") or []
            if self.save_evidence and frames:
                stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
                ev_dir = self.events_dir / f"{stamp}_{alert.camera_id}_{alert.rule_name}"
                ev_dir.mkdir(parents=True, exist_ok=True)
                try:
                    self._write_evidence(ev_dir, payload)
                except Exception:  # noqa: BLE001
                    log.error("evidence write failed for provisional %s", event_id,
                              exc_info=True)
            with self._lock:
                self._db.execute(
                    "UPDATE events SET provisional = 0, confidence = ?, reason = ?, "
                    "unverified = ?, gate_error = ?, latency_s = ?, evidence_dir = ? "
                    "WHERE id = ?",
                    (float(result.confidence), result.reason,
                     1 if errored else 0, getattr(result, "error", "") or None,
                     latency, str(ev_dir) if ev_dir else None, event_id))
                self._db.commit()
            self.persisted += 1
            self._note_latency(alert.priority, latency)
            verdict = "UNVERIFIED — review manually" if errored else "CONFIRMED"
            event = {"ts": ts, "iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "id": event_id,
                     "camera_id": alert.camera_id, "rule": alert.rule_name,
                     "priority": alert.priority, "confidence": float(result.confidence),
                     "zone": alert.zone, "track_id": alert.track_id,
                     "object_label": alert.object_label,
                     "reason": f"{verdict}: {result.reason}",
                     "evidence_dir": str(ev_dir) if ev_dir else None}
            if self.mobile_base:
                event["link"] = f"{self.mobile_base}/alert/{event_id}"
            self._dispatch(event, event_id)
        else:
            # Retraction: explicit and visible, never a quiet deletion. The row
            # stays, marked, so "why did my phone buzz?" always has an answer.
            # review stays NULL: TrueSight's own rejection is NOT an operator
            # label and must not feed the training data as one.
            with self._lock:
                self._db.execute(
                    "UPDATE events SET provisional = 0, retracted = 1, "
                    "confidence = ?, reason = ?, latency_s = ? WHERE id = ?",
                    (float(result.confidence),
                     f"RETRACTED — verified as not a threat: {result.reason}",
                     latency, event_id))
                self._db.commit()
            event = {"ts": ts, "iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "id": event_id,
                     "camera_id": alert.camera_id, "rule": alert.rule_name,
                     "priority": "low", "confidence": float(result.confidence),
                     "zone": alert.zone, "track_id": alert.track_id,
                     "object_label": alert.object_label,
                     "reason": f"❎ RETRACTED — the provisional {alert.rule_name} alert "
                               f"on {alert.camera_id} was verified as NOT a threat. "
                               f"({result.reason})",
                     "evidence_dir": None}
            self._dispatch(event, event_id)
            log.info("[RETRACTED] %s :: %s — provisional verified as not a threat",
                     alert.camera_id, alert.rule_name)

    def _note_latency(self, priority: str, latency) -> None:
        if latency is None:
            return
        from collections import deque
        self._tier_latency.setdefault(priority, deque(maxlen=50)).append(latency)

    def latency_stats(self) -> dict:
        """Detection -> operator-visible, per tier. Published in /health."""
        def med(values):
            ordered = sorted(values)
            return round(ordered[len(ordered) // 2], 2) if ordered else None
        out = {tier: {"median_s": med(dq), "n": len(dq)}
               for tier, dq in self._tier_latency.items()}
        out["provisional"] = {"median_s": med(self._provisional_latency),
                              "n": len(self._provisional_latency)}
        return out

    def triage_state(self, camera_id: str, rule_name: str) -> str | None:
        """The newest matching event's effective triage state, or None.

        The incident lifecycle (custom_rules, 3 Sep) reads this to decide
        whether an ongoing condition still needs reminding: an alert nobody
        has touched does, an acknowledged or resolved one does not — the
        human owns it now. Same review->state projection triage uses.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT state, review FROM events WHERE camera_id = ? AND rule = ? "
                "ORDER BY id DESC LIMIT 1", (camera_id, rule_name)).fetchone()
        if row is None:
            return None
        state, review = row
        if state in ("new", "acknowledged", "resolved"):
            return state
        if review in ("true", "false"):
            return "resolved"
        return "acknowledged" if review == "ack" else "new"

    @staticmethod
    def _audit_json(value: Any, fallback: Any, field: str) -> tuple[str | None, str | None]:
        if value is None:
            return None, None
        try:
            return json.dumps(value, separators=(",", ":")), None
        except (TypeError, ValueError, RecursionError) as exc:
            return json.dumps(fallback, separators=(",", ":")), f"{field}: {exc}"

    def audit_candidate_generated(self, alert: Any) -> int | str | None:
        """Persist auditable pilot candidates before queue admission is decided."""
        payload = alert.payload or {}
        candidate = payload.get("candidate")
        detector = getattr(candidate, "detector", None)
        if detector == "object_watch":
            metadata = getattr(candidate, "metadata", {}) or {}
            audit_id = f"object-{uuid.uuid4().hex}"
            payload_json, payload_error = self._audit_json(metadata, {}, "payload")
            try:
                with self._lock:
                    self._db.execute(
                        "INSERT INTO object_watch_audit ("
                        "id, candidate_id, camera_id, object_id, object_label, state, "
                        "generated_at, timestamp_s, zone, similarity, payload_json"
                        ") VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            audit_id,
                            audit_id,
                            alert.camera_id,
                            str(metadata.get("object_id") or ""),
                            str(getattr(candidate, "object_label", None) or ""),
                            str(metadata.get("state") or ""),
                            time.time(),
                            float(alert.timestamp),
                            metadata.get("zone") or alert.zone,
                            metadata.get("similarity"),
                            payload_json if payload_error is None else payload_json,
                        ),
                    )
                    self._db.commit()
                return audit_id
            except Exception:  # noqa: BLE001 - audit I/O must not stop detection
                log.error("object watch candidate audit write failed", exc_info=True)
                return None
        if detector == "multiple_people_moving":
            metadata = getattr(candidate, "metadata", {}) or {}
            candidate_id = f"motion-{uuid.uuid4().hex}"
            track_ids, track_error = self._audit_json(
                metadata.get("track_ids") or [], [], "track_ids"
            )
            try:
                with self._lock:
                    self._db.execute(
                        "INSERT INTO motion_candidate_audit ("
                        "candidate_id, case_id, timestamp_s, generated_at, camera_id, "
                        "rule_name, track_ids_json, persistence_error"
                        ") VALUES (?,?,?,?,?,?,?,?)",
                        (
                            candidate_id,
                            str(metadata.get("case_id") or alert.camera_id),
                            float(alert.timestamp),
                            time.time(),
                            alert.camera_id,
                            alert.rule_name,
                            track_ids,
                            track_error,
                        ),
                    )
                    self._db.commit()
                return candidate_id
            except Exception:  # noqa: BLE001 - audit I/O must not stop detection
                log.error("motion candidate audit write failed", exc_info=True)
                return None
        if detector != "concealment":
            return None
        metadata = getattr(candidate, "metadata", {}) or {}
        components, components_error = self._audit_json(
            metadata.get("components") or {}, {}, "components"
        )
        associated_bag, bag_error = self._audit_json(
            metadata.get("associated_bag"), None, "associated_bag"
        )
        reasons, reasons_error = self._audit_json(
            metadata.get("reasons") or [], [], "reasons"
        )
        errors = [error for error in (components_error, bag_error, reasons_error) if error]
        generated_at = time.time()
        try:
            with self._lock:
                cursor = self._db.execute(
                    "INSERT INTO concealment_audit ("
                    "generated_at, camera_id, rule_name, candidate_timestamp, track_id, "
                    "destination, peak_score, components_json, limited, "
                    "associated_bag_json, reasons_json, enqueued_at, audit_error"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        generated_at, alert.camera_id, alert.rule_name,
                        float(alert.timestamp), alert.track_id,
                        metadata.get("destination"), metadata.get("score"), components,
                        None if metadata.get("limited") is None
                        else int(bool(metadata.get("limited"))),
                        associated_bag, reasons, payload.get("enqueued_at"),
                        "; ".join(errors) or None,
                    ),
                )
                self._db.commit()
                return int(cursor.lastrowid)
        except Exception:  # noqa: BLE001 - audit I/O must not stop detection
            log.error("concealment candidate audit write failed", exc_info=True)
            return None

    def audit_candidate_admission(self, alert: Any, status: str) -> None:
        """Record whether a generated row entered, duplicated, or left the queue."""
        payload = alert.payload or {}
        object_id = payload.get("object_watch_audit_id")
        if object_id is not None:
            gate_status = "pending" if status == "admitted" else "not_gated"
            try:
                with self._lock:
                    self._db.execute(
                        "UPDATE object_watch_audit SET admission_status = ?, "
                        "gate_status = ?, admitted_at = "
                        "CASE WHEN ? = 'admitted' THEN COALESCE(admitted_at, ?) "
                        "ELSE admitted_at END WHERE id = ?",
                        (status, gate_status, status, time.time(), object_id),
                    )
                    self._db.commit()
            except Exception:  # noqa: BLE001 - audit I/O must not stop detection
                log.error("object watch admission audit write failed", exc_info=True)
            return
        motion_id = payload.get("motion_candidate_audit_id")
        if motion_id is not None:
            gate_status = "pending" if status == "admitted" else "not_gated"
            persistence_status = "pending" if status == "admitted" else "not_applicable"
            try:
                with self._lock:
                    self._db.execute(
                        "UPDATE motion_candidate_audit SET admission_status = ?, "
                        "gate_status = ?, persistence_status = ?, admitted_at = "
                        "CASE WHEN ? = 'admitted' THEN COALESCE(admitted_at, ?) "
                        "ELSE admitted_at END WHERE candidate_id = ?",
                        (
                            status, gate_status, persistence_status, status,
                            time.time(), motion_id,
                        ),
                    )
                    self._db.commit()
            except Exception:  # noqa: BLE001 - audit I/O must not stop detection
                log.error("motion admission audit write failed", exc_info=True)
            return
        audit_id = payload.get("concealment_audit_id")
        if audit_id is None:
            return
        try:
            with self._lock:
                self._db.execute(
                    "UPDATE concealment_audit SET admission_status = ?, admitted_at = "
                    "CASE WHEN ? = 'admitted' THEN COALESCE(admitted_at, ?) "
                    "ELSE admitted_at END "
                    "WHERE id = ?",
                    (status, status, time.time(), audit_id),
                )
                self._db.commit()
        except Exception:  # noqa: BLE001 - audit I/O must not stop detection
            log.error("concealment admission audit write failed", exc_info=True)

    def _audit_object_verdict(self, alert: Any, result: Any) -> None:
        payload = alert.payload or {}
        candidate = payload.get("candidate")
        if getattr(candidate, "detector", None) != "object_watch":
            return
        audit_id = payload.get("object_watch_audit_id")
        if audit_id is None:
            audit_id = self.audit_candidate_generated(alert)
            if audit_id is None:
                return
            payload["object_watch_audit_id"] = audit_id
            self.audit_candidate_admission(alert, "admitted")
        if result is None or getattr(result, "errored", False):
            gate_status = "unverified"
        elif result.confirmed:
            gate_status = "confirmed"
        else:
            gate_status = "rejected"
        try:
            with self._lock:
                self._db.execute(
                    "UPDATE object_watch_audit SET gate_status = ?, verdict_at = ? "
                    "WHERE id = ?",
                    (gate_status, time.time(), audit_id),
                )
                self._db.commit()
        except Exception:  # noqa: BLE001 - audit I/O must not stop the gate
            log.error("object watch verdict audit write failed", exc_info=True)

    def _audit_object_persistence(
        self, alert: Any, event_id: int | None, error: str = ""
    ) -> None:
        audit_id = (alert.payload or {}).get("object_watch_audit_id")
        if audit_id is None:
            return
        try:
            with self._lock:
                self._db.execute(
                    "UPDATE object_watch_audit SET persisted_event_id = ? WHERE id = ?",
                    (str(event_id) if event_id is not None else "", audit_id),
                )
                self._db.commit()
        except Exception:  # noqa: BLE001 - audit I/O must not stop alert delivery
            log.error("object watch persistence audit write failed", exc_info=True)

    def _audit_concealment(self, alert: Any, result: Any) -> None:
        """Attach the eventual gate outcome to its pre-admission audit row."""
        payload = alert.payload or {}
        candidate = payload.get("candidate")
        if getattr(candidate, "detector", None) != "concealment":
            return
        audit_id = payload.get("concealment_audit_id")
        if audit_id is None:
            audit_id = self.audit_candidate_generated(alert)
            if audit_id is None:
                return
            payload["concealment_audit_id"] = audit_id
            self.audit_candidate_admission(alert, "admitted")
        verdict_at = time.time()
        enqueued_at = payload.get("enqueued_at")
        latency = (
            round(verdict_at - enqueued_at, 3)
            if isinstance(enqueued_at, (int, float)) else None
        )
        if result is None or getattr(result, "errored", False):
            verdict = "unverified"
        elif getattr(result, "review_required", False):
            verdict = "review_required"
        elif result.confirmed:
            verdict = "confirmed"
        else:
            verdict = "rejected"
        try:
            with self._lock:
                self._db.execute(
                    "UPDATE concealment_audit SET verdict_at = ?, gate_result_timestamp = ?, "
                    "gate_latency_s = ?, verdict = ?, confirmed = ?, confidence = ?, "
                    "reason = ?, gate_error = ?, prompt_version = ? WHERE id = ?",
                    (
                        verdict_at, getattr(result, "timestamp", None), latency, verdict,
                        None if result is None else int(bool(getattr(result, "confirmed", False))),
                        getattr(result, "confidence", None),
                        getattr(result, "reason", "") if result is not None else "",
                        (getattr(result, "error", "") if result is not None
                         else "missing verification result"),
                        getattr(result, "prompt_version", "") if result else "",
                        audit_id,
                    ),
                )
                self._db.commit()
        except Exception:  # noqa: BLE001 - audit I/O must not stop the gate
            log.error("concealment audit write failed", exc_info=True)

    def _audit_motion_verdict(self, alert: Any, result: Any) -> None:
        payload = alert.payload or {}
        candidate = payload.get("candidate")
        if getattr(candidate, "detector", None) != "multiple_people_moving":
            return
        audit_id = payload.get("motion_candidate_audit_id")
        if audit_id is None:
            audit_id = self.audit_candidate_generated(alert)
            if audit_id is None:
                return
            payload["motion_candidate_audit_id"] = audit_id
            self.audit_candidate_admission(alert, "admitted")
        if result is None or getattr(result, "errored", False):
            gate_status = "unverified"
        elif result.confirmed:
            gate_status = "confirmed"
        else:
            gate_status = "rejected"
        persistence_status = "pending" if result is not None and result.confirmed \
            else "not_applicable"
        try:
            with self._lock:
                self._db.execute(
                    "UPDATE motion_candidate_audit SET gate_status = ?, verdict_at = ?, "
                    "confidence = ?, reason = ?, gate_error = ?, prompt_version = ?, "
                    "persistence_status = ? WHERE candidate_id = ?",
                    (
                        gate_status, time.time(), getattr(result, "confidence", None),
                        getattr(result, "reason", "") if result is not None else "",
                        (getattr(result, "error", "") if result is not None
                         else "missing verification result"),
                        getattr(result, "prompt_version", "") if result else "",
                        persistence_status, audit_id,
                    ),
                )
                self._db.commit()
        except Exception:  # noqa: BLE001 - audit I/O must not stop the gate
            log.error("motion verdict audit write failed", exc_info=True)

    def _audit_motion_persistence(
        self, alert: Any, event_id: int | None, error: str = ""
    ) -> None:
        audit_id = (alert.payload or {}).get("motion_candidate_audit_id")
        if audit_id is None:
            return
        status = "persisted" if event_id is not None else "failed"
        try:
            with self._lock:
                gate_row = self._db.execute(
                    "SELECT gate_status FROM motion_candidate_audit WHERE candidate_id = ?",
                    (audit_id,),
                ).fetchone()
                gate_status = gate_row[0] if gate_row else ""
                self._db.execute(
                    "UPDATE motion_candidate_audit SET persistence_status = ?, "
                    "persistence_error = ?, user_event_id = ?, persisted_event_id = ? "
                    "WHERE candidate_id = ?",
                    (
                        status, error or None, str(event_id or ""),
                        str(event_id) if event_id is not None and gate_status == "confirmed" else "",
                        audit_id,
                    ),
                )
                self._db.commit()
        except Exception:  # noqa: BLE001 - audit I/O must not stop alert delivery
            log.error("motion persistence audit write failed", exc_info=True)

    def handle(self, alert: Any, result: Any) -> None:
        self._audit_concealment(alert, result)
        self._audit_motion_verdict(alert, result)
        self._audit_object_verdict(alert, result)
        if result is None:
            return
        provisional_id = (alert.payload or {}).get("provisional_event_id")
        if provisional_id:
            self._settle_provisional(provisional_id, alert, result)
            return
        # Three outcomes, not two. UNVERIFIED means the gate never reached a
        # verdict — logging it as CONFIRMED would overstate what we know, and as
        # REJECTED would claim a judgement nobody made.
        if getattr(result, "errored", False):
            tag, level = "UNVERIFIED", log.warning
        elif getattr(result, "review_required", False):
            tag, level = "NEEDS REVIEW", log.info
        elif result.confirmed:
            tag, level = "CONFIRMED", log.info
        else:
            tag, level = "REJECTED ", log.info
        level("[%s] %s :: %s (%s) — %s | conf=%.2f | %s", tag, alert.camera_id,
              alert.rule_name, alert.priority.upper(), alert.title,
              result.confidence, result.reason)
        if getattr(result, "errored", False) and result.error:
            log.warning("[gate unavailable] %s :: %s — %s",
                        alert.camera_id, alert.rule_name, result.error)
        if not result.confirmed and not getattr(result, "review_required", False):
            return None
        try:
            event_id = self._persist(alert, result)
            self._audit_motion_persistence(alert, event_id)
            self._audit_object_persistence(alert, event_id)
            return event_id
        except Exception as exc:  # noqa: BLE001 - persistence must not kill the gate
            self._audit_motion_persistence(alert, None, str(exc))
            self._audit_object_persistence(alert, None, str(exc))
            log.error(f"[alert-sink error] {str(exc)[:140]}", exc_info=True)

    # --- routing ---------------------------------------------------------
    def _is_acknowledged(self, event_id: Any) -> bool:
        """Has an operator handled this alert yet? (any review label clears it)"""
        try:
            with self._lock:
                row = self._db.execute(
                    "SELECT review FROM events WHERE id=?", (event_id,)).fetchone()
            return bool(row and row[0] and row[0] != "new")
        except Exception as exc:  # noqa: BLE001
            log.warning("acknowledgement lookup failed; treating as unacknowledged", exc_info=True)
            return False

    def _notifier_for(self, spec: str) -> Any:
        """Build (and cache) a notifier per channel spec — building a Telegram or
        Twilio client per alert would be wasteful."""
        if spec not in self._notifier_cache:
            self._notifier_cache[spec] = build_notifier(spec)
        return self._notifier_cache[spec]

    def _dispatch(self, event: dict, event_id: Any = None) -> None:
        """Send an event to whichever channels its routing rule names.

        A routing table routes what it MATCHES. An alert no rule claims goes
        to the site-wide notifier the operator configured — not to the routing
        file's own default, which used to send a Telegram site's daytime
        alerts to the engine log because one night-shift rule existed.
        (Audit 23 Aug, #2.)
        """
        spec, rule_name = self.routing.channels_for(event)
        matched = self.routing.rules and rule_name != "default"
        target = self._notifier_for(spec) if matched else self.notifier
        try:
            target.notify(event)
            self.routed += 1
        except Exception as exc:  # noqa: BLE001 - a channel failing must not kill the sink
            log.error(f"[routing] '{rule_name}' -> {spec} failed: {str(exc)[:110]}", exc_info=True)
        if event_id is not None:
            rule = self.routing.match(event)
            if rule:
                self.escalations.register(event_id, event, rule)

    def run_escalations(self) -> int:
        """Re-send alerts nobody acknowledged in time. Call periodically."""
        sent = 0
        for item in self.escalations.due():
            ev = dict(item["event"])
            ev["escalated"] = True
            ev["reason"] = f"ESCALATED (unacknowledged): {ev.get('reason', '')}"
            try:
                self._notifier_for(item["to"]).notify(ev)
                sent += 1
                self.escalated += 1
                log.info(f"[routing] escalated {ev.get('camera_id')}::{ev.get('rule')} "
                      f"-> {item['to']} (rule '{item['rule']}')")
            except Exception as exc:  # noqa: BLE001
                log.error(f"[routing] escalation to {item['to']} failed: {str(exc)[:110]}", exc_info=True)
        return sent

    @staticmethod
    def _annotate(frame: Any, bbox: tuple | None, label: str, priority: str = "high") -> Any:
        """Draw the subject's box on a copy of the frame, so evidence shows WHO."""
        if not bbox:
            return frame
        import cv2
        colours = {"critical": (60, 60, 220), "high": (60, 140, 240),
                   "medium": (60, 200, 240), "low": (160, 160, 160)}   # BGR
        col = colours.get(str(priority).lower(), colours["high"])
        out = frame.copy()
        h, w = out.shape[:2]
        x1, y1, x2, y2 = (max(0, min(int(v), lim)) for v, lim in
                          zip(bbox, (w, h, w, h)))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = y1 - 8 if y1 > th + 12 else min(h - 4, y2 + th + 8)
            cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), col, -1)
            cv2.putText(out, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _write_evidence(self, ev_dir: Path, payload: dict) -> None:
        """Write the event's frames + a clip.mp4.

        The app plays a smooth IMAGE cine-loop (no video-codec dependency — QtWebEngine
        can't always decode the mp4), so we save the whole CONTINUOUS window as
        individual JPEGs (they're already JPEG bytes -> written straight to disk). The
        mp4 is still written for archival / Telegram."""
        import cv2
        frames = payload.get("frames") or []
        clip_frames = payload.get("clip_frames") or []
        if clip_frames:
            for i, jb in enumerate(clip_frames):
                (ev_dir / f"frame_{i:03d}.jpg").write_bytes(jb)
        else:
            for i, f in enumerate(frames):
                cv2.imwrite(str(ev_dir / f"frame_{i:03d}.jpg"), f)
        # A single annotated "who" shot: the frame closest to the moment it fired,
        # with the subject boxed. Deliberately NOT drawn on every clip frame — the
        # subject moves, so a fixed box across the window would point at the wrong
        # place and mislead the operator.
        bbox = payload.get("bbox")
        if bbox and frames:
            cand = payload.get("candidate")
            label = f"#{getattr(cand, 'person_id', '')} {getattr(cand, 'title', '')}".strip()
            shot = self._annotate(frames[-1], bbox, label[:40],
                                  getattr(cand, "priority", "high"))
            cv2.imwrite(str(ev_dir / "subject.jpg"), shot)

        clip_fps = float(payload.get("clip_fps") or 0.0)
        if len(clip_frames) >= 6:
            fps = clip_fps if 1.0 <= clip_fps <= 60.0 else 4.0
            self._write_video_clip(ev_dir / "clip.mp4", clip_frames, fps)
        elif frames:
            self._write_clip(ev_dir / "clip.mp4", frames)

    def annotate_event(self, event_id: int, description: str) -> bool:
        """Append a late-arriving English description to a stored event.

        The async enrichment path (W5): bypassed alerts fire instantly with a
        mechanical reason; the VLM's description arrives afterwards and lands
        here. The API's stream loop watches the reason column, so this UPDATE
        is what becomes the client's alert.update. Never touches review,
        priority, or evidence — it adds words, nothing else."""
        if not description:
            return False
        try:
            with self._lock:
                row = self._db.execute(
                    "SELECT reason FROM events WHERE id=?", (event_id,)).fetchone()
                if row is None:
                    return False
                base = (row[0] or "").split(" · TrueSight:")[0]
                self._db.execute(
                    "UPDATE events SET reason=? WHERE id=?",
                    (f"{base} · TrueSight: {description}", event_id))
                self._db.commit()
            return True
        except Exception:  # noqa: BLE001 - annotation must never cost anything
            log.debug("event annotation failed for id %s", event_id, exc_info=True)
            return False

    def _persist(self, alert: Any, result: Any) -> int | None:
        ts = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        payload = alert.payload or {}
        ev_dir = self.events_dir / f"{stamp}_{alert.camera_id}_{alert.rule_name}"
        ev_dir.mkdir(parents=True, exist_ok=True)

        frames = payload.get("frames") or []
        if self.save_evidence and frames:
            self._write_evidence(ev_dir, payload)

        # detection -> verified wall-clock latency (queue wait + TrueSight time).
        enq = payload.get("enqueued_at")
        latency_s = round(ts - enq, 2) if isinstance(enq, (int, float)) else None
        self._note_latency(alert.priority, latency_s)

        event = {
            "ts": ts, "iso": iso, "camera_id": alert.camera_id, "rule": alert.rule_name,
            "priority": alert.priority, "confidence": float(result.confidence),
            "reason": result.reason, "track_id": alert.track_id, "zone": alert.zone,
            # ABSOLUTE. A relative path is only meaningful next to the working
            # directory that produced it, and retention resolves paths against
            # the OUTPUT root — so a relative row could never be matched, and
            # storage limitation silently did nothing for the life of the
            # install. (27 Aug: 483 events, 3.6 GB, unpurgeable.)
            "object_label": alert.object_label, "evidence_dir": str(ev_dir.resolve()),
            "latency_s": latency_s,
            "bbox": ",".join(str(int(v)) for v in payload["bbox"]) if payload.get("bbox") else None,
            "unverified": 1 if (getattr(result, "errored", False)
                                 or getattr(result, "review_required", False)) else 0,
            "gate_error": getattr(result, "error", "") or None,
            # The capture rate of the evidence frames. Without it the viewer
            # guessed — a hard-coded 140ms/frame that replayed 4fps loitering
            # evidence at ~7fps: 1.75x fast-forward, and the person 'skips'
            # across the window (field report, 30 Aug).
            "clip_fps": float((alert.payload or {}).get("clip_fps") or 0.0) or None,
        }
        # save_evidence=False (site key store_evidence: false) means NO image of
        # any kind: this full-resolution snapshot used to be written even then.
        # The event's metadata (event.json) is not footage and is kept.
        if self.save_evidence and self.full_frame_provider is not None:
            try:
                full = self.full_frame_provider(alert.camera_id)
                if full:
                    (ev_dir / "evidence_full.jpg").write_bytes(full)
            except Exception:  # noqa: BLE001 - an extra frame must never cost the alert
                log.debug("full-frame evidence fetch failed", exc_info=True)
        (ev_dir / "event.json").write_text(json.dumps(event, indent=2))

        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason,"
                "track_id,zone,object_label,evidence_dir,latency_s,bbox,unverified,gate_error,"
                "prompt_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts, iso, alert.camera_id, alert.rule_name, alert.priority,
                 float(result.confidence), result.reason, alert.track_id, alert.zone,
                 alert.object_label, str(ev_dir), latency_s, event["bbox"],
                 event["unverified"], event["gate_error"],
                 getattr(result, "prompt_version", "") or None))
            event_id = cur.lastrowid
            self._db.commit()
        self.persisted += 1
        event["id"] = event_id
        if getattr(result, "review_required", False):
            # Disputed evidence stays in the local review inbox, not paging or escalation.
            return event_id
        if self.mobile_base:
            event["link"] = f"{self.mobile_base}/alert/{event_id}"
        # Feedback loop: chronically-wrong (camera, rule) pairs are stored but not paged.
        self._reload_calibration()
        if self.calibration.demoted(alert.camera_id, alert.rule_name):
            log.warning(f"[calibrated] {alert.camera_id} :: {alert.rule_name} demoted by feedback "
                  f"— stored, NOT notified")
        else:
            # Routed to the channels this alert's rule names; registers escalation
            # if nobody acknowledges it in time.
            self._dispatch(event, event_id)
        return event_id

    def _write_video_clip(self, path: Path, jpeg_frames: list, src_fps: float,
                          container_fps: int = 24) -> None:
        """Write a REAL video of the event from the continuous JPEG window.

        The pipeline captures frames at `src_fps` (e.g. ~4). We re-time them onto a
        player-friendly `container_fps` by repeating each source frame, so the clip
        plays at TRUE real-time speed (duration = n/src_fps) but decodes smoothly in
        any player — real footage of the event, not a slideshow of stills.
        """
        import cv2
        import numpy as np
        if not jpeg_frames:
            return
        imgs = []
        for jb in jpeg_frames:
            im = cv2.imdecode(np.frombuffer(jb, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                imgs.append(im)
        if not imgs:
            return
        h, w = imgs[0].shape[:2]
        repeat = max(1, int(round(container_fps / max(src_fps, 0.1))))
        for fourcc in ("avc1", "mp4v"):        # prefer H.264, fall back to MPEG-4
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), container_fps, (w, h))
            if vw.isOpened():
                for im in imgs:
                    out = im if im.shape[:2] == (h, w) else cv2.resize(im, (w, h))
                    for _ in range(repeat):
                        vw.write(out)
                vw.release()
                ensure_h264(path)
                return

    def _write_clip(self, path: Path, frames: list, fps: int = 12,
                    hold_seconds: float = 0.5) -> None:
        """Encode the evidence frames into a short MP4 (archival + Telegram video).

        Evidence is only a handful of stills, so a naive low-fps clip flashes past
        too fast to read. Instead we hold each frame for ~`hold_seconds` at a normal
        container fps — a slower, smoother playback that any player handles cleanly.
        """
        import cv2
        if not frames:
            return
        h, w = frames[0].shape[:2]
        repeat = max(1, int(round(fps * hold_seconds)))   # frames to hold each still
        for fourcc in ("avc1", "mp4v"):        # prefer H.264, fall back to MPEG-4
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if vw.isOpened():
                for f in frames:
                    out = f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h))
                    for _ in range(repeat):
                        vw.write(out)
                vw.release()
                ensure_h264(path)
                return

    def close(self) -> None:
        # Queued Telegram uploads first — an alert raised in the engine's last
        # seconds must still leave the building before the process does.
        fn = getattr(getattr(self, "notifier", None), "close", None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - shutdown must finish
                log.error(f"[alert-sink] notifier close failed: {str(exc)[:100]}", exc_info=True)
        with self._lock:
            self._db.close()
