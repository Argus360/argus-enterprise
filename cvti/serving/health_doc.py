"""The /health document — one answer to "is this site OK right now?" (EP-04-T1).

Ask the question directly: a customer's box dies at 3am Saturday — how do you
find out? Today, they tell you, probably Monday, probably after a missed
incident. This document is the foundation for changing that: it is what the
System panel reads, what `/health` serves, and what the heartbeat will send.

Six signal classes, per the plan: per-camera status, gate reachability and
latency, disk headroom, memory level, per-component error counters, uptime.

Pure assembly — no I/O in here, so the status rules are testable without an
engine. The privacy constraint is structural: nothing in this module ever
touches a frame, an event row, or anything identifying a person. Counts,
states, and durations only.
"""

from __future__ import annotations

import time

OK = "ok"
DEGRADED = "degraded"
CRITICAL = "critical"


def derive_status(*, cameras: list, gate: dict, disk: dict, memory: dict,
                  components: dict, scene_mapping: list | None = None,
                  engine: dict | None = None) -> tuple:
    """(status, reasons). The reasons are the point — a bare 'degraded' with no
    'why' just moves the 3am question one hop along."""
    reasons = []

    wall = (engine or {}).get("wall_budget") or {}
    if wall and float(wall.get("over_budget_ms") or 0) > 0:
        # The wall's decode budget does not fit (Tolu's review, 15 Oct 2026):
        # say whether detection alone is the reason, and what to change.
        reserved, budget = float(wall.get("reserved_ms") or 0), float(wall.get("budget_ms") or 0)
        detail = (f"detection alone needs {reserved:.0f} of {budget:.0f} ms/s"
                  if reserved > budget else
                  f"{wall.get('watched', 0)} watched tiles at their minimum rate exceed the "
                  f"{budget:.0f} ms/s left after detection")
        reasons.append((DEGRADED, f"wall decode budget exceeded by {float(wall['over_budget_ms']):.0f} ms/s: "
                                  f"{detail} — fewer analysed cameras, a lower target_fps, "
                                  f"substreams, or a higher wall_decode_budget_ms"))

    for cam in cameras or []:
        if cam.get("state") == "offline":
            reasons.append((CRITICAL, f"camera {cam.get('camera_id')} offline "
                                      f"{cam.get('time_in_state', 0):.0f}s"))
        elif cam.get("state") == "reconnecting":
            reasons.append((DEGRADED, f"camera {cam.get('camera_id')} reconnecting"))
        ingest = cam.get("ingest") or {}
        if ingest.get("limited"):
            # Adaptive ingest (3 Sep): the machine, not the camera, is the
            # bottleneck — say so with numbers and the one action that fixes it.
            reasons.append((DEGRADED,
                            f"camera {cam.get('camera_id')}: this machine sustains "
                            f"~{ingest.get('sustainable_fps')}fps of "
                            f"{ingest.get('width')}x{ingest.get('height')}"
                            f"@{ingest.get('source_fps')} — sampling reduced; the "
                            "camera's lower-resolution substream would restore full rate"))

    if gate.get("reachable") is False:
        # The gate down means alerts arrive UNVERIFIED — the product is running
        # on fail-visible, not verifying. That is a critical condition even
        # though alerts still flow.
        reasons.append((CRITICAL, "verification gate unreachable — alerts are "
                                  "arriving unverified"))

    if disk.get("level") == "critical":
        reasons.append((CRITICAL, f"disk {disk.get('used_pct')}% used"))
    elif disk.get("level") == "warning":
        reasons.append((DEGRADED, f"disk {disk.get('used_pct')}% used"))

    if memory.get("level") == "critical":
        reasons.append((CRITICAL, f"memory: {memory.get('available_gb')}GB available"))
    elif memory.get("level") == "warn":
        reasons.append((DEGRADED, f"memory: {memory.get('available_gb')}GB available"))

    for name in (components or {}).get("degraded", []):
        reasons.append((DEGRADED, f"component {name} failing >10% of attempts"))

    for mapping in scene_mapping or []:
        camera_id = mapping.get("camera_id", "unknown")
        mapping_status = mapping.get("status")
        if mapping_status == "failed":
            # Defensive second layer: whatever produced this error, no
            # credential embedded in a source URL reaches a screen from here.
            from cvti.utils import redact_credentials
            detail = redact_credentials(str(mapping.get("error") or "unknown error"))
            reasons.append((DEGRADED, f"camera {camera_id} scene mapping failed: {detail}"))
        elif mapping_status == "stale":
            reasons.append((DEGRADED, f"camera {camera_id} scene context is stale"))
        elif mapping_status == "pending":
            reasons.append((DEGRADED, f"camera {camera_id} scene mapping is pending"))
        elif (mapping_status == "ready_unreviewed"
              and mapping.get("review_required")):
            reasons.append((DEGRADED, f"camera {camera_id} scene context awaits review"))

    if any(level == CRITICAL for level, _ in reasons):
        status = CRITICAL
    elif reasons:
        status = DEGRADED
    else:
        status = OK
    return status, [text for _, text in reasons]


def build_health_doc(*, started_at: float, cameras: list, gate: dict, disk: dict,
                     memory: dict, components: dict, engine: dict | None = None,
                     self_test: dict | None = None,
                     scene_mapping: list | None = None,
                     now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    status, reasons = derive_status(cameras=cameras, gate=gate, disk=disk,
                                    memory=memory, components=components,
                                    scene_mapping=scene_mapping, engine=engine)
    # Every heartbeat says which build wrote it: support triage starts with
    # "which version are you on?", and the honest answer is what the ENGINE
    # is running, not what the console's sidebar shows.
    from cvti.utils import argus_version
    return {
        "status": status,
        "version": argus_version(),
        "reasons": reasons,                    # empty when ok, by design
        "uptime_s": round(max(0.0, now - started_at), 1),
        "generated_at": now,
        "cameras": cameras,
        "gate": gate,
        "disk": disk,
        "memory": memory,
        "components": components,
        "engine": engine or {},
        "self_test": self_test or {},
        "scene_mapping": scene_mapping or [],
    }
