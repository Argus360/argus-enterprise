"""The heartbeat receiver — every site's health on one page (EP-04-T2).

    python tools/heartbeat_receiver.py --keys sites.json --port 8900 \
        [--telegram <bot_token>:<chat_id>] [--dashboard-token <secret>]

Runs anywhere with Python 3.9+: standard library and SQLite only, one file, no
dependencies. Sites POST /heartbeat with their site key; you open / for the
dashboard. A site that stops reporting, or reports itself degraded, sends you a
Telegram message on the transition — once, not every check.

`sites.json` is {"site-id": "site-key", ...}. Issue one random key per site and
put the same key in that site's Argus settings. An unknown or wrong key is a
401 and is logged; heartbeats carry health only (see docs/HEARTBEAT.md), so the
blast radius of a leaked key is spam, not footage.

This is deliberately boring. It is the machine that must be up when everything
else is down, and every dependency is a way for it not to be.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# A site is "missed" after this many intervals of silence. 2.5 tolerates one
# lost packet and a slow cycle without crying wolf; a truly dead box is named
# within ~13 minutes at the default 5-minute interval.
MISSED_AFTER_INTERVALS = 2.5
DEFAULT_INTERVAL_S = 300.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id TEXT NOT NULL,
    received_at REAL NOT NULL,
    status TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS hb_site_time ON heartbeats (site_id, received_at DESC);
CREATE TABLE IF NOT EXISTS diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id TEXT NOT NULL,
    received_at REAL NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER NOT NULL
);
"""
# A diagnostics upload is a zip the site operator chose to send (logs, health,
# redacted site file - see tools/send_diagnostics.ps1). Never pushed by the
# engine on its own; capped so a bad client cannot fill the disk.
DIAG_MAX_BYTES = 30 * 1024 * 1024
DIAG_LIST_LIMIT = 50


class Store:
    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def record(self, site_id: str, payload: dict, now: float | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO heartbeats (site_id, received_at, status, payload) "
                "VALUES (?,?,?,?)",
                (site_id, now if now is not None else time.time(),
                 payload.get("status"), json.dumps(payload)))
            self._db.commit()

    def close(self) -> None:
        self._db.close()

    def record_diagnostic(self, site_id: str, filename: str, size: int,
                          now: float | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO diagnostics (site_id, received_at, filename, size_bytes) "
                "VALUES (?, ?, ?, ?)", (site_id, now or time.time(), filename, int(size)))
            self._db.commit()

    def diagnostics(self, limit: int = DIAG_LIST_LIMIT) -> list:
        with self._lock:
            rows = self._db.execute(
                "SELECT site_id, received_at, filename, size_bytes FROM diagnostics "
                "ORDER BY received_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def latest(self) -> dict:
        """site_id -> its most recent heartbeat row (parsed)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT site_id, received_at, status, payload FROM heartbeats h "
                "WHERE received_at = (SELECT MAX(received_at) FROM heartbeats "
                "                     WHERE site_id = h.site_id)").fetchall()
        out = {}
        for r in rows:
            try:
                payload = json.loads(r["payload"])
            except ValueError:
                payload = {}
            out[r["site_id"]] = {"received_at": r["received_at"],
                                 "status": r["status"], "payload": payload}
        return out


def site_view(site_id: str, latest: dict | None, known: bool,
              now: float | None = None, interval: float = DEFAULT_INTERVAL_S) -> dict:
    """One site's dashboard row. `missed` beats whatever the site last said —
    a box that reported 'ok' and then went silent is not ok."""
    now = now if now is not None else time.time()
    if latest is None:
        return {"site_id": site_id, "state": "never-reported", "known": known,
                "age_s": None, "status": None, "reasons": []}
    age = now - latest["received_at"]
    payload = latest.get("payload") or {}
    if age > MISSED_AFTER_INTERVALS * interval:
        state = "missed"
    else:
        state = payload.get("status") or "unknown"
    return {"site_id": site_id, "state": state, "known": known,
            "age_s": round(age, 1), "status": payload.get("status"),
            "reasons": (payload.get("reasons") or [])[:5],
            "cameras": payload.get("cameras") or [],
            "uptime_s": payload.get("uptime_s")}


class Alerter:
    """Tells you ONCE per transition — into missed/degraded/critical, and back.

    Repeating the same bad news every check trains the reader to ignore it,
    which is the alert-fatigue failure this whole product exists to avoid.
    """

    def __init__(self, notify=None):
        self.notify = notify or (lambda text: print(f"[ALERT] {text}", flush=True))
        self._last: dict = {}

    def observe(self, view: dict) -> bool:
        site, state = view["site_id"], view["state"]
        previous = self._last.get(site)
        self._last[site] = state
        if previous == state or previous is None and state in ("ok", "never-reported"):
            return False
        if state == "missed":
            self.notify(f"🔴 {site} has stopped heartbeating "
                        f"(last seen {view['age_s']:.0f}s ago). The box may be down.")
        elif state in ("critical", "degraded"):
            why = "; ".join(view.get("reasons") or [])[:200] or "no reason given"
            icon = "🔴" if state == "critical" else "🟠"
            self.notify(f"{icon} {site} reports {state}: {why}")
        elif state == "ok" and previous in ("missed", "critical", "degraded"):
            self.notify(f"🟢 {site} is back to normal.")
        else:
            return False
        return True


def telegram_notifier(token: str, chat_id: str):
    def send(text: str) -> None:
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=10)
    return send


def render_diagnostics(items: list, token: str = "", now: float | None = None) -> str:
    """The uploads table under the sites table: newest first, one link each."""
    if not items:
        return "<p class='mut'>No diagnostics uploaded yet.</p>"
    now = now or time.time()
    q = f"?token={urllib.parse.quote(token)}" if token else ""
    rows = []
    for it in items:
        age_s = now - float(it["received_at"])
        when = time.strftime("%d %b %H:%M", time.localtime(float(it["received_at"])))
        href = (f"/diagnostics/{urllib.parse.quote(it['site_id'])}/"
                f"{urllib.parse.quote(it['filename'])}{q}")
        rows.append(f"<tr><td><b>{html.escape(it['site_id'])}</b></td>"
                    f"<td>{html.escape(when)} ({age_s / 3600:.1f}h ago)</td>"
                    f"<td>{int(it['size_bytes']) // 1024} KB</td>"
                    f"<td><a href='{href}'>{html.escape(it['filename'])}</a></td></tr>")
    return ("<h2>Diagnostics uploads</h2><table><tr><th>Site</th><th>Received</th>"
            "<th>Size</th><th>File</th></tr>" + "".join(rows) + "</table>")


def render_dashboard(views: list, now: float | None = None, diagnostics: list | None = None,
                     token: str = "") -> str:
    """One HTML page, no assets, refreshes itself. Worst sites first."""
    order = {"missed": 0, "critical": 1, "degraded": 2, "never-reported": 3,
             "unknown": 4, "ok": 5}
    views = sorted(views, key=lambda v: (order.get(v["state"], 4), v["site_id"]))
    colours = {"ok": "#22c55e", "degraded": "#f59e0b", "critical": "#ef4444",
               "missed": "#ef4444", "never-reported": "#5f6b84", "unknown": "#8d94a6"}

    def age(sec):
        if sec is None:
            return "—"
        if sec < 90:
            return f"{sec:.0f}s ago"
        if sec < 5400:
            return f"{sec / 60:.0f}m ago"
        return f"{sec / 3600:.1f}h ago"

    rows = []
    for v in views:
        colour = colours.get(v["state"], "#8d94a6")
        cams = v.get("cameras") or []
        live = sum(1 for c in cams if c.get("state") == "connected")
        reasons = "; ".join(v.get("reasons") or [])
        rows.append(
            f"<tr><td><span style='color:{colour}'>●</span> "
            f"<b>{html.escape(v['site_id'])}</b></td>"
            f"<td style='color:{colour};font-weight:600'>{html.escape(v['state'].upper())}</td>"
            f"<td>{age(v.get('age_s'))}</td>"
            f"<td>{live}/{len(cams) if cams else '—'}</td>"
            f"<td class='mut'>{html.escape(reasons)}</td></tr>")
    body = "".join(rows) or "<tr><td colspan=5 class='mut'>No sites have reported yet.</td></tr>"
    return f"""<!doctype html><meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>Argus sites</title>
<style>
 body{{background:#0a1326;color:#edf1f8;font:14px system-ui;margin:0;padding:28px}}
 h1{{font-size:16px;margin:0 0 4px}} h2{{font-size:13px;margin:28px 0 0;color:#8d94a6}} a{{color:#7dd3fc}} .mut{{color:#8d94a6;font-size:12px}}
 table{{border-collapse:collapse;margin-top:18px;width:100%;max-width:980px}}
 td,th{{text-align:left;padding:9px 14px;border-bottom:1px solid #1f2c48;font-size:13px}}
 th{{color:#5f6b84;font-size:10px;text-transform:uppercase;letter-spacing:.08em}}
</style>
<h1>Argus — deployed sites</h1>
<div class="mut">Health signals only; no footage leaves any site. Refreshes every 30s.</div>
<table><tr><th>Site</th><th>State</th><th>Last heartbeat</th><th>Cameras live</th><th>Why</th></tr>
{body}</table>
{render_diagnostics(diagnostics or [], token, now) if diagnostics is not None else ""}"""


class Receiver:
    def __init__(self, *, keys: dict, db_path: str | Path = "heartbeats.db",
                 alerter: Alerter | None = None, interval: float = DEFAULT_INTERVAL_S,
                 dashboard_token: str = "", diag_dir: str | Path | None = None) -> None:
        self.keys = keys
        self.store = Store(db_path)
        self.diag_dir = Path(diag_dir) if diag_dir else None
        self.alerter = alerter or Alerter()
        self.interval = interval
        self.dashboard_token = dashboard_token
        self._server = None

    # --- request handling, separated from HTTP for tests -------------------

    def close(self) -> None:
        """Release the store — Windows cannot delete a database that is open."""
        self.store.close()
    def handle_heartbeat(self, site_key: str, body: bytes) -> tuple:
        import hmac
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return 400, "not json"
        site_id = str(payload.get("site_id") or "")
        expected = self.keys.get(site_id)
        if not expected or not hmac.compare_digest(site_key, expected):
            return 401, "unknown site or wrong key"
        self.store.record(site_id, payload)
        return 200, "ok"

    def handle_diagnostic(self, site_key: str, site_id: str, name: str, body: bytes) -> tuple:
        """Store one uploaded zip for a known site. Same key as the heartbeat."""
        import hmac
        import re
        if self.diag_dir is None:
            return 404, "diagnostics inbox not enabled"
        expected = self.keys.get(site_id or "")
        if not expected or not hmac.compare_digest(site_key, expected):
            return 401, "unknown site or wrong key"
        if not body or len(body) > DIAG_MAX_BYTES:
            return 413, f"empty or over {DIAG_MAX_BYTES // (1024 * 1024)} MB"
        if body[:2] != b"PK":
            return 415, "not a zip"
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name or "diagnostics.zip")[:120]
        if not safe.lower().endswith(".zip"):
            safe += ".zip"
        filename = time.strftime("%Y%m%d-%H%M%S") + "-" + safe
        folder = self.diag_dir / re.sub(r"[^A-Za-z0-9._-]", "_", site_id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_bytes(body)
        self.store.record_diagnostic(site_id, filename, len(body))
        return 200, f"stored {filename} ({len(body) // 1024} KB)"

    def diagnostic_path(self, site_id: str, filename: str) -> Path | None:
        """Resolve a download, refusing anything that is not a plain file name."""
        if self.diag_dir is None or "/" in site_id or "/" in filename or ".." in filename:
            return None
        path = (self.diag_dir / site_id / filename).resolve()
        try:
            path.relative_to(self.diag_dir.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None

    def views(self, now: float | None = None) -> list:
        latest = self.store.latest()
        all_sites = set(self.keys) | set(latest)
        return [site_view(sid, latest.get(sid), sid in self.keys, now, self.interval)
                for sid in sorted(all_sites)]

    def check_alerts(self, now: float | None = None) -> int:
        return sum(1 for v in self.views(now) if self.alerter.observe(v))

    # --- serving ------------------------------------------------------------
    def serve(self, host: str = "0.0.0.0", port: int = 8900) -> None:
        recv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                return

            def _send(self, code, body, ctype="text/plain"):
                data = body.encode() if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path.rstrip("/") == "/diagnostics":
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > DIAG_MAX_BYTES:
                        self._send(413, "too large")
                        return
                    qs = urllib.parse.parse_qs(parsed.query)
                    code, msg = recv.handle_diagnostic(
                        self.headers.get("X-Argus-Site-Key", ""),
                        (qs.get("site_id") or [""])[0], (qs.get("name") or [""])[0],
                        self.rfile.read(length))
                    print(f"[receiver] diagnostics from {self.client_address[0]}: {code} {msg}",
                          flush=True)
                    self._send(code, msg)
                    return
                if parsed.path.rstrip("/") != "/heartbeat":
                    self._send(404, "not found")
                    return
                length = int(self.headers.get("Content-Length") or 0)
                code, msg = recv.handle_heartbeat(
                    self.headers.get("X-Argus-Site-Key", ""), self.rfile.read(length))
                if code == 401:
                    print(f"[receiver] refused heartbeat from {self.client_address[0]}: {msg}",
                          flush=True)
                self._send(code, msg)

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                if recv.dashboard_token:
                    supplied = (urllib.parse.parse_qs(parsed.query).get("token") or [""])[0]
                    if supplied != recv.dashboard_token:
                        self._send(401, "dashboard token required (?token=...)")
                        return
                if parsed.path in ("/", "/index.html"):
                    diags = recv.store.diagnostics() if recv.diag_dir is not None else None
                    self._send(200, render_dashboard(recv.views(), diagnostics=diags,
                                                     token=recv.dashboard_token), "text/html")
                elif parsed.path.startswith("/diagnostics/"):
                    parts = parsed.path.split("/", 3)
                    path = recv.diagnostic_path(urllib.parse.unquote(parts[2]),
                                                urllib.parse.unquote(parts[3])) if len(parts) == 4 else None
                    if path is None:
                        self._send(404, "not found")
                    else:
                        data = path.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/zip")
                        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                elif parsed.path == "/sites.json":
                    self._send(200, json.dumps(recv.views(), default=str),
                               "application/json")
                else:
                    self._send(404, "not found")

        def alert_loop():
            while True:
                time.sleep(60)
                try:
                    recv.check_alerts()
                except Exception as exc:  # noqa: BLE001 - the watcher must keep watching
                    print(f"[receiver] alert check failed: {exc}", flush=True)

        threading.Thread(target=alert_loop, daemon=True).start()
        self._server = ThreadingHTTPServer((host, port), Handler)
        print(f"[receiver] dashboard on http://{host}:{self._server.server_address[1]}/  "
              f"({len(self.keys)} site key(s) loaded)", flush=True)
        self._server.serve_forever()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--keys", required=True, help='JSON file: {"site-id": "site-key"}')
    p.add_argument("--db", default="heartbeats.db")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8900)
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                   help="the senders' interval; 'missed' = 2.5x this of silence")
    p.add_argument("--telegram", default="",
                   help="<bot_token>:<chat_id> for transition alerts")
    p.add_argument("--dashboard-token", default="",
                   help="require ?token=... on the dashboard")
    p.add_argument("--diag-dir", default="",
                   help="enable the diagnostics inbox: zips POSTed to /diagnostics land here")
    args = p.parse_args()

    keys = json.loads(Path(args.keys).read_text())
    notify = None
    if args.telegram:
        token, _, chat = args.telegram.rpartition(":")
        notify = telegram_notifier(token, chat)
    receiver = Receiver(keys=keys, db_path=args.db, interval=args.interval,
                        alerter=Alerter(notify), dashboard_token=args.dashboard_token,
                        diag_dir=args.diag_dir or None)
    receiver.serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
