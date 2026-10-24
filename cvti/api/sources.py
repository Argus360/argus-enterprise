"""Read helpers over the engine's own outputs.

Everything the read-only API serves comes from files the engine already
writes — the events database, gate_health.json, and the site config. Reading
them here (rather than through the console's permission-gated methods) keeps
the API decoupled from the console's single-session model and adds zero load
to the detection path.
"""

from __future__ import annotations

import json
import sqlite3
import time
from copy import deepcopy
from pathlib import Path

from cvti.logging_setup import get_logger
from cvti.utils import redact_credentials

log = get_logger(__name__)


# ---- health -----------------------------------------------------------------

def read_health(db_path: str) -> dict:
    """The engine's health doc, or a truthful 'unknown' when it hasn't run."""
    path = Path(db_path).parent / "gate_health.json"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"status": "unknown", "reasons": ["engine has not reported yet"],
                "cameras": [], "gate": {}, "engine": {"phase": "stopped"}}
    return doc


def engine_log_tail(db_path: str) -> tuple[str, str]:
    """(log path, the last telling line) from the engine's own stdout log.

    The retired PyQt console reported this on every stopped poll, after a
    pilot machine "spent a day as a photo of a black wall" (29 Aug). The API
    that replaced it reported freshness only, so an engine that died a second
    after Start looked exactly like one the operator had stopped on purpose —
    which is how a KeyError on camera config became "the camera connects but
    there is no picture" with nothing on screen to explain it (20 Sep).
    """
    log_path = Path(db_path).parent / "monitor.log"
    try:
        lines = [line.strip() for line
                 in log_path.read_text(errors="replace").splitlines()[-40:]
                 if line.strip()]
    except OSError:
        return str(log_path), ""
    telling = [line for line in lines
               if any(key in line for key in ("Error", "ERROR", "Traceback",
                                              "error:", "Exception", "denied",
                                              "Permission", "No such"))]
    candidates = telling or lines
    return str(log_path), (candidates[-1][:300] if candidates else "")


def monitor_state(db_path: str) -> dict:
    """Derive running/starting/stopped from the health doc's freshness — the
    same heartbeat truth the console uses, without owning the subprocess."""
    doc = read_health(db_path)
    generated = float(doc.get("generated_at") or 0)
    age = time.time() - generated if generated else None
    phase = str((doc.get("engine") or {}).get("phase") or "")
    fresh = age is not None and age < 30
    starting = phase.startswith("starting") and (age is not None and age < 90)
    state = {
        "running": bool(fresh or starting),
        "starting": bool(starting),
        "phase": phase or ("stopped" if not fresh else "monitoring"),
        "health_age_s": round(age, 1) if age is not None else None,
    }
    if not state["running"]:
        # Stopped is not a diagnosis. Carry why, so the UI can say it.
        log_path, last_error = engine_log_tail(db_path)
        state["log_path"] = log_path
        state["last_error"] = last_error
    return state


def read_link_events(db_path: str, *, now: float | None = None) -> list[dict]:
    """Recent per-camera link transitions for operations health surfaces.

    Link transitions are operational state, not incidents. Keep a compact
    24-hour summary here so operators can see outages and recoveries without
    polluting the incident queue.
    """
    if not Path(db_path).exists():
        return []
    cutoff = (time.time() if now is None else now) - 24 * 60 * 60
    try:
        con = _connect(db_path)
        rows = con.execute(
            "SELECT camera_id, rule, ts, reason FROM events "
            "WHERE rule IN ('camera_offline', 'camera_recovered') AND ts >= ? "
            "ORDER BY ts DESC, id DESC", (cutoff,)
        ).fetchall()
        con.close()
    except sqlite3.OperationalError:
        return []

    summary: dict[str, dict] = {}
    for row in rows:
        camera_id = str(row["camera_id"] or "")
        item = summary.setdefault(camera_id, {
            "camera_id": camera_id,
            "offline_count": 0,
            "recovered_count": 0,
            "last_event_ts": row["ts"],
            "last_event_rule": row["rule"],
            "last_reason": row["reason"] or "",
        })
        key = "offline_count" if row["rule"] == "camera_offline" else "recovered_count"
        item[key] += 1
    return list(summary.values())


# ---- cameras ----------------------------------------------------------------

def _zone_count(zones_path) -> int:
    """How many zones a camera's zones file defines; 0 when there is none."""
    if not zones_path:
        return 0
    try:
        from pathlib import Path as _P
        data = json.loads(_P(str(zones_path)).read_text())
        return len([z for z in data.get("zones", []) if z.get("polygon")])
    except (OSError, ValueError, AttributeError):
        return 0


def _custom_rules(cam: dict) -> list[dict]:
    """The camera's plain-English rules as the app shows them ({question, dwell}).

    Same normalisation as ConsoleBackend._custom_rules: the `custom_rules`
    list plus the wizard-era single `custom_rule`, blanks dropped. Until
    22 Sep the camera list carried neither field, so the Rules tab rendered
    `camera.custom_rules || []` -- always empty -- and a sentence that HAD
    been written to the site file looked like it was never saved (pilot,
    Windows: "describe in English doesn't save")."""
    rules = [dict(r) for r in (cam.get("custom_rules") or []) if isinstance(r, dict)]
    legacy = cam.get("custom_rule")
    if isinstance(legacy, dict) and (legacy.get("question") or "").strip() and \
            legacy["question"] not in [r.get("question") for r in rules]:
        rules.insert(0, dict(legacy))
    return [{"question": r["question"].strip(), "dwell": float(r.get("dwell") or 0.0)}
            for r in rules if (r.get("question") or "").strip()]


# The detector toggles the Detectors tab draws, mirroring
# ConsoleBackend.RULE_FLAGS. They must round-trip: the tab reads
# camera[key] for the switch's position AND sends `not camera[key]` when it
# is clicked, so a camera list that omits them does not merely draw the
# switches grey -- every click computes `not None` and sends True, which
# means a detector can be switched ON and never OFF (reported 25 Sep:
# "the toggles are working but it's not turning green when clicked").
DETECTOR_FLAGS: tuple = (
    "concealment", "video_action", "violence", "weapons", "theft", "object_watch",
    "tamper", "fire_smoke", "running", "crowd_formation", "normal_movement",
    "multiple_people_moving", "fall",
)


def _detectors(cam: dict) -> dict:
    """Every detector toggle's state, explicitly true or false — never absent."""
    return {flag: bool(cam.get(flag)) for flag in DETECTOR_FLAGS}


def read_cameras(site_path: str, db_path: str) -> list[dict]:
    """Configured cameras (credential-redacted) merged with live link state
    from the health doc."""
    try:
        from cvti.serving.onboarding import list_cameras
        cams = list_cameras(site_path)
    except Exception:  # noqa: BLE001 - a missing/invalid site is empty, not a crash
        log.debug("could not read site cameras; returning empty", exc_info=True)
        cams = []
    try:
        from cvti.serving.onboarding import normalized_hierarchy
        hierarchy = normalized_hierarchy(site_path)
    except Exception:  # noqa: BLE001 - location metadata must not hide cameras
        log.debug("could not resolve camera hierarchy", exc_info=True)
        hierarchy = {"branches": [], "unassigned_areas": [],
                     "unassigned_cameras": []}
    locations = {}
    for branch in hierarchy["branches"]:
        for area in branch["areas"]:
            for camera in area["cameras"]:
                locations[str(camera.get("id"))] = {
                    "area_id": area["id"], "branch_id": branch["id"],
                }
    for area in hierarchy["unassigned_areas"]:
        for camera in area["cameras"]:
            locations[str(camera.get("id"))] = {"area_id": area["id"]}
    for camera in hierarchy["unassigned_cameras"]:
        area_id = str(camera.get("area_id", "")).strip()
        locations[str(camera.get("id"))] = (
            {"area_id": area_id} if area_id else {}
        )
    health = read_health(db_path)
    link = {c.get("camera_id"): c for c in (health.get("cameras") or [])}
    out = []
    for c in cams:
        cid = str(c.get("id"))
        live = link.get(cid, {})
        item = {
            "id": cid,
            "source": redact_credentials(str(c.get("source", ""))),
            # Loitering, intrusion and restricted-area alerts only exist inside
            # a zone (PerCameraState.process gates them on zone_monitor). The
            # UI needs this number to say so, because until 21 Sep nothing did
            # and a pilot's "loitering isn't working" was a camera with no zone.
            "zone_count": _zone_count(c.get("zones")),
            "custom_rules": _custom_rules(c),
            **_detectors(c),
            # The drawn tripwire, so a line editor opens on the current one
            # rather than a blank frame. None when the camera has none.
            "vehicle_line": c.get("vehicle_line"),
            "view_only": bool(c.get("view_only")),
            "state": live.get("state", "unknown"),
            "last_frame_age_s": live.get("last_frame_age_s"),
            "reconnects": live.get("reconnects"),
            "ingest": live.get("ingest"),
        }
        item.update(locations.get(cid, {}))
        out.append(item)
    return out


def read_hierarchy(site_path: str) -> dict:
    """Normalized hierarchy with every nested camera credential-redacted."""
    from cvti.serving.onboarding import normalized_hierarchy

    hierarchy = deepcopy(normalized_hierarchy(site_path))
    camera_groups = [hierarchy["unassigned_cameras"]]
    for branch in hierarchy["branches"]:
        camera_groups.extend(area["cameras"] for area in branch["areas"])
    camera_groups.extend(
        area["cameras"] for area in hierarchy["unassigned_areas"]
    )
    for cameras in camera_groups:
        for camera in cameras:
            for field in ("source", "detect_source"):
                if field in camera:
                    camera[field] = redact_credentials(str(camera[field]))
    return hierarchy


# ---- events -----------------------------------------------------------------

_EVENT_COLUMNS = ("id", "ts", "iso", "camera_id", "rule", "priority",
                  "confidence", "reason", "track_id", "zone", "object_label",
                  "evidence_dir", "review", "reviewed_at")


def _verdict_from_row(row: dict) -> str:
    """The event's verification verdict for the API's flat shape."""
    review = row.get("review")
    if review in ("true", "ack"):
        return "confirmed"
    if review == "false":
        return "rejected"
    if str(row.get("reason") or "").startswith("NEEDS REVIEW:"):
        return "review_required"
    # An engine-confirmed alert with no operator label is still 'confirmed' by
    # TrueSight; unverified alerts are written with a marker in reason/priority.
    return "confirmed"


def _to_api_event(row: dict) -> dict:
    eid = f"evt_{row['id']}"
    return {
        "id": eid,
        "ts": row.get("ts"),
        "iso": row.get("iso"),
        "camera_id": row.get("camera_id"),
        "rule": row.get("rule"),
        "priority": row.get("priority"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
        "zone": row.get("zone"),
        "verdict": _verdict_from_row(row),
        "evidence": {
            "dir": row.get("evidence_dir"),
            "thumb": f"/api/v1/events/{eid}/evidence/thumb" if row.get("evidence_dir") else None,
            "clip": bool(row.get("evidence_dir")),
        },
        "triage": {
            # A provisional the VLM later rejected is settled, not awaiting an
            # operator: report it resolved so a live alert.update drops it from
            # the review queue instead of leaving it there as "new".
            "state": ("resolved" if row.get("retracted")
                      else "new" if not row.get("review") else row.get("review")),
        },
        "retracted": bool(row.get("retracted")),
    }


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    con.row_factory = sqlite3.Row
    return con


def _live_clause(con: sqlite3.Connection) -> str:
    """SQL predicate for incidents an operator should see.

    A critical alert is shown provisionally before the VLM verdict; when the
    verdict is "not a threat" the row is kept (audit: "why did it flash?") but
    marked retracted. Retracted rows are not incidents — listing them, or
    counting them as awaiting review (their `review` stays NULL on purpose, so
    TrueSight's rejection never becomes an operator label), filled the CHI
    queue with forklift work the VLM had already cleared. Older databases
    without the column have nothing to hide. Camera link transitions have
    their own health surface and are never incidents."""
    incident_rule = (
        "COALESCE(rule, '') NOT IN ('camera_offline', 'camera_recovered')"
    )
    try:
        con.execute("SELECT retracted FROM events LIMIT 0")
    except sqlite3.OperationalError:
        return incident_rule
    return f"COALESCE(retracted, 0) = 0 AND {incident_rule}"


def read_events(db_path: str, *, limit: int = 50, cursor: int | None = None,
                camera: str | None = None, priority: str | None = None) -> dict:
    """Cursor-paged events, newest first. Cursor is the last id seen."""
    if not Path(db_path).exists():
        return {"events": [], "next_cursor": None}
    where, params = [], []
    if cursor is not None:
        where.append("id < ?"); params.append(cursor)
    if camera:
        where.append("camera_id = ?"); params.append(camera)
    if priority:
        where.append("priority = ?"); params.append(priority)
    params.append(max(1, min(limit, 200)))
    try:
        con = _connect(db_path)
    except sqlite3.OperationalError:
        return {"events": [], "next_cursor": None, "error": "events database unavailable"}
    try:
        where.append(_live_clause(con))
        clause = " WHERE " + " AND ".join(where)
        sql = f"SELECT * FROM events{clause} ORDER BY id DESC LIMIT ?"
        rows = [dict(r) for r in con.execute(sql, params).fetchall()]
    except sqlite3.OperationalError as exc:
        con.close()
        if "no such table" in str(exc):
            return {"events": [], "next_cursor": None}   # fresh site, genuinely quiet
        return {"events": [], "next_cursor": None, "error": "events database unavailable"}
    con.close()
    events = [_to_api_event(r) for r in rows]
    next_cursor = rows[-1]["id"] if len(rows) == params[-1] else None
    return {"events": events, "next_cursor": next_cursor}


def read_event(db_path: str, event_id: str) -> dict | None:
    raw_id = event_id.removeprefix("evt_")
    if not raw_id.isdigit() or not Path(db_path).exists():
        return None
    try:
        con = _connect(db_path)
        row = con.execute("SELECT * FROM events WHERE id = ?", (int(raw_id),)).fetchone()
        con.close()
    except sqlite3.OperationalError:
        return None
    return _to_api_event(dict(row)) if row else None


def review_states(db_path: str, limit: int = 300) -> dict[int, str]:
    """id -> review label for the most recent events — the alert.update diff.

    A change in an event's review column (operator ack/resolve, or the fast
    path settling a provisional verdict in place) is exactly what a WS client
    cannot see from alert.new alone; the stream loop diffs this map and emits
    alert.update for rows that changed."""
    if not Path(db_path).exists():
        return {}
    try:
        con = _connect(db_path)
        rows = con.execute(
            "SELECT id, review, reason FROM events ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        con.close()
        # review AND reason: an async enrichment (annotate_event) changes only
        # the reason — that must reach WS clients as alert.update too.
        return {int(r[0]): f"{r[1] or ''}|{len(r[2] or '')}" for r in rows}
    except sqlite3.OperationalError:
        return {}


def read_triage(db_path: str) -> dict:
    """Counts for the triage header: to-review, total, and a priority split."""
    if not Path(db_path).exists():
        return {"to_review": 0, "total": 0, "by_priority": {}}
    try:
        con = _connect(db_path)
    except sqlite3.OperationalError:
        return {"to_review": 0, "total": 0, "by_priority": {}}
    try:
        live = _live_clause(con)
        total = con.execute(f"SELECT COUNT(*) FROM events WHERE {live}").fetchone()[0]
        to_review = con.execute(
            f"SELECT COUNT(*) FROM events WHERE review IS NULL AND {live}").fetchone()[0]
        rows = con.execute(
            f"SELECT priority, COUNT(*) FROM events WHERE {live} GROUP BY priority").fetchall()
        by_priority = {r[0]: r[1] for r in rows}
    except sqlite3.OperationalError:
        con.close()
        return {"to_review": 0, "total": 0, "by_priority": {}}
    con.close()
    return {"to_review": to_review, "total": total, "by_priority": by_priority}


def max_event_id(db_path: str) -> int:
    """Highest event id, for the WebSocket's new-alert poll. 0 when none."""
    if not Path(db_path).exists():
        return 0
    try:
        con = _connect(db_path)
        row = con.execute("SELECT MAX(id) FROM events").fetchone()
        con.close()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0
