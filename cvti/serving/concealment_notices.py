"""Transient camera warnings; never a substitute for persisted incidents."""
import threading
import time
import uuid


class ConcealmentNotices:
    def __init__(self, clock=time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._notices = {}

    def candidate(self, alert):
        candidate = (alert.payload or {}).get("candidate")
        if getattr(candidate, "detector", "") != "concealment":
            return
        # Remember the candidate so its verdict can find it, but show nothing
        # yet: an unverified gesture on the live tile ("Possible product
        # concealment - Verifying") read as an accusation the VLM then usually
        # withdrew (CHI, 22 Oct 2026). Only a confirmed or inconclusive verdict
        # puts a notice on the camera; a rejected one never appears.
        token = uuid.uuid4().hex
        alert.payload["concealment_notice_id"] = token

    def verdict(self, alert, result):
        token = (alert.payload or {}).get("concealment_notice_id")
        if not token:
            return
        if result is not None and getattr(result, "review_required", False):
            self._set(alert.camera_id, token, "inconclusive")
        elif result is not None and result.confirmed and not getattr(result, "errored", False):
            self._set(alert.camera_id, token, "review")
        else:
            with self._lock:
                current = self._notices.get(alert.camera_id)
                if current and current["id"] == token:
                    self._notices.pop(alert.camera_id, None)

    def _set(self, camera_id, token, phase):
        now = self._clock()
        with self._lock:
            current = self._notices.get(camera_id)
            if (phase == "verifying" and current and current["phase"] in ("review", "inconclusive")
                    and current["expires_at"] > now):
                return
            self._notices[camera_id] = {
                "id": token, "phase": phase, "expires_at": now + 8,
                "created_at": now,
            }

    def snapshot(self, camera_id):
        with self._lock:
            notice = self._notices.get(camera_id)
            if notice and notice["expires_at"] > self._clock():
                return dict(notice)
            self._notices.pop(camera_id, None)
            return None
