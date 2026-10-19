"""Opt-in heartbeat: the site tells you it is alive (EP-04-T2).

Without this, a dead engine, a dead camera, a full disk or a stalled VLM at a
site you cannot physically reach is discovered by the customer — probably
Monday, probably after a missed incident.

Three properties are non-negotiable, and each is enforced in code rather than
policy:

**Off by default.** A site sends nothing until someone deliberately configures
a heartbeat URL. There is no phone-home in the default install.

**Health only, by whitelist.** The payload is built by copying named fields out
of the health document — a whitelist, not a redaction — so a future field added
to /health cannot leak into the heartbeat by forgetting to strip it. No frames,
no event content, nothing identifying a person. The schema is public:
docs/HEARTBEAT.md.

**Inspectable.** Every payload actually sent is also written to
`<output>/heartbeat_last.json`, so the customer can see exactly what leaves
their machine, at any time, without trusting the documentation.

Outbound-only HTTPS POST, so it works through typical NAT/firewalls with no
inbound port on the site.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from cvti.health import component
from cvti.logging_setup import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 1
DEFAULT_INTERVAL_S = 300.0


def heartbeat_payload(health: dict, *, site_id: str, now: float | None = None) -> dict:
    """The wire payload, copied field-by-field from the health document.

    Whitelist by construction: anything not named here does not travel.
    """
    gate = health.get("gate") or {}
    disk = health.get("disk") or {}
    memory = health.get("memory") or {}
    engine = health.get("engine") or {}
    self_test = health.get("self_test") or {}
    return {
        "schema": SCHEMA_VERSION,
        "site_id": site_id,
        "sent_at": now if now is not None else time.time(),
        "status": health.get("status"),
        "reasons": list(health.get("reasons") or [])[:10],
        "uptime_s": health.get("uptime_s"),
        "version": health.get("version"),
        "cameras": [{"id": c.get("camera_id"), "state": c.get("state"),
                     "last_frame_age_s": c.get("last_frame_age_s"),
                     "reconnects": c.get("reconnects"),
                     # detection counts: frames looked at, frames with a person,
                     # seconds since a person was last seen. Counts, no pixels.
                     "frames_analysed": (c.get("detection") or {}).get("frames_analysed"),
                     "frames_with_people": (c.get("detection") or {}).get("frames_with_people"),
                     "people_last_seen_s": (c.get("detection") or {}).get("people_last_seen_s")}
                    for c in (health.get("cameras") or [])],
        "gate": {"provider": gate.get("provider"), "reachable": gate.get("reachable"),
                 "verified": gate.get("verified"), "unverified": gate.get("unverified"),
                 "errors": gate.get("errors"),
                 "median_latency_s": gate.get("median_latency_s")},
        "disk": {"used_pct": disk.get("used_pct"), "free_gb": disk.get("free_gb"),
                 "level": disk.get("level")},
        "memory": {"available_gb": memory.get("available_gb"),
                   "level": memory.get("level")},
        "components": [{"name": c.get("name"), "processed": c.get("processed"),
                        "errors": c.get("errors"), "degraded": c.get("degraded")}
                       for c in (health.get("components") or {}).get("components", [])],
        "engine": {"frames_processed": engine.get("frames_processed"),
                   "alerts_queued": engine.get("alerts_queued"),
                   "cameras": engine.get("cameras")},
        "self_test": {"ok": self_test.get("ok"), "at": self_test.get("at")},
    }


class Heartbeat:
    """Sends the payload every `interval` seconds. Failures are counted and
    rate-limit-logged; a broken heartbeat must never affect detection."""

    def __init__(self, *, url: str, site_key: str, site_id: str,
                 health_provider: Callable[[], dict],
                 output_dir: str | Path | None = None,
                 interval: float = DEFAULT_INTERVAL_S,
                 clock: Callable[[], float] = time.time,
                 transport: Callable[..., Any] | None = None) -> None:
        self.url = url.rstrip("/")
        self.site_key = site_key
        self.site_id = site_id
        self.health_provider = health_provider
        self.output_dir = Path(output_dir) if output_dir else None
        self.interval = interval
        self.clock = clock
        self._transport = transport or self._post
        self._health = component("heartbeat")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.sent = 0
        self.last_sent_at = 0.0
        self.last_error = ""

    def _post(self, payload: dict) -> int:
        req = urllib.request.Request(
            f"{self.url}/heartbeat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "X-Argus-Site-Key": self.site_key},
            method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status

    def beat(self) -> bool:
        """One send. Writes heartbeat_last.json whether or not the send works —
        what the customer can inspect is what we *tried* to send."""
        payload = heartbeat_payload(self.health_provider() or {},
                                    site_id=self.site_id, now=self.clock())
        if self.output_dir:
            try:
                (self.output_dir / "heartbeat_last.json").write_text(
                    json.dumps(payload, indent=2, default=str))
            except OSError:
                log.debug("could not write heartbeat_last.json", exc_info=True)
        try:
            status = self._transport(payload)
            if int(status) >= 400:
                raise RuntimeError(f"receiver answered {status}")
            self.sent += 1
            self.last_sent_at = self.clock()
            self.last_error = ""
            self._health.ok()
            return True
        except Exception as exc:  # noqa: BLE001 - a broken heartbeat must not affect detection
            self.last_error = f"{type(exc).__name__}: {str(exc)[:140]}"
            self._health.failed(exc, log, "sending the heartbeat")
            return False

    def status(self) -> dict:
        return {"enabled": True, "url": self.url, "site_id": self.site_id,
                "interval_s": self.interval, "sent": self.sent,
                "last_sent_at": self.last_sent_at, "last_error": self.last_error}

    def start(self) -> "Heartbeat":
        def loop() -> None:
            while not self._stop.wait(self.interval):
                self.beat()
        self._thread = threading.Thread(target=loop, name="heartbeat", daemon=True)
        self._thread.start()
        log.info("heartbeat: ON — %s every %.0fs as site %r (health only; see "
                 "docs/HEARTBEAT.md and heartbeat_last.json)",
                 self.url, self.interval, self.site_id)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
