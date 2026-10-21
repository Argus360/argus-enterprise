# Heartbeat — what a site transmits, exactly

*Public on purpose. A monitoring feature in a privacy-first product only works
if anyone can check what it sends.*

## The rules

1. **Off by default.** A fresh install sends nothing anywhere. It turns on only
   when a site owner enters a monitoring URL and site key in
   **System → Remote monitoring**.
2. **Health only, by whitelist.** The payload is built by copying the named
   fields below out of the site's health document. Anything not listed does not
   travel — a field added to `/health` in the future cannot leak into the
   heartbeat by omission. **No frames, no video, no event content, nothing
   identifying a person.** A test enforces the whitelist.
3. **Inspectable.** Every payload sent is also written to
   `heartbeat_last.json` beside the site's database, and shown in
   **System → Remote monitoring → "View exactly what was last transmitted."**
4. **Outbound only.** One HTTPS POST every 5 minutes. No inbound port is opened
   on the site; it works through ordinary NAT and firewalls.

## The payload (schema version 1)

```json
{
  "schema": 1,
  "site_id": "deluxe-paints-ikeja",
  "sent_at": 1787150000.0,
  "status": "ok | degraded | critical",
  "reasons": ["camera backroom offline 360s"],
  "uptime_s": 86400.0,
  "cameras": [
    {"id": "aisle_1", "state": "connected", "last_frame_age_s": 0.3, "reconnects": 0}
  ],
  "gate":   {"provider": "ollama", "reachable": true, "verified": 412,
             "unverified": 0, "errors": 0, "median_latency_s": 11.2},
  "disk":   {"used_pct": 61.4, "free_gb": 180.2, "level": "ok"},
  "memory": {"available_gb": 6.1, "level": "ok"},
  "components": [
    {"name": "detector.aisle_1", "processed": 84211, "errors": 0, "degraded": false}
  ],
  "engine": {"frames_processed": 84211, "alerts_queued": 9, "cameras": 6},
  "self_test": {"ok": true, "at": 1787143200.0}
}
```

| Field | What it is — and is not |
|---|---|
| `site_id` | The site's name, slugified. Chosen by the owner. |
| `status`, `reasons` | The health verdict and why. Reasons name cameras and components, never people or events. |
| `cameras[]` | Link state, frame freshness and detection counts per camera: frames the model looked at, frames with a person, seconds since a person was last seen. **No image data.** |
| `version` | The engine build that wrote the heartbeat. |
| `gate` | Verification counts and latency. **Not** what was verified. |
| `disk`, `memory` | Headroom, so a filling disk is caught before recording stops. |
| `components[]` | Error counters per detector. Counts only. |
| `self_test` | Whether the daily end-to-end test passed, and when. |

## Authentication

Each site sends `X-Argus-Site-Key`, a random per-site secret issued when the
site is enrolled and checked in constant time by the receiver. A leaked key's
blast radius is fake health pings — never footage, which does not travel.

## The receiver

`tools/heartbeat_receiver.py` — standard library + SQLite, one file, no
dependencies. Shows every site's state on one page, worst first; a site that
misses ~2.5 intervals is flagged **MISSED** regardless of what it last claimed,
and transitions (missed / degraded / critical / recovered) send a Telegram
message **once**, not per check.

```bash
# on any always-on machine you control
echo '{"deluxe-paints-ikeja": "<random-site-key>"}' > sites.json
python tools/heartbeat_receiver.py --keys sites.json --port 8900 \
    --telegram <bot_token>:<chat_id> --dashboard-token <secret>
```
