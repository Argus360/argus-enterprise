"""Viewer-owned camera captures, independent of the inference engine."""
from __future__ import annotations

import threading
from urllib.parse import quote, urlencode

from cvti.app.errors import PreviewBusy
from cvti.app.live_wall import FrameServer, LiveWall


class CameraPreview:
    def __init__(self):
        self._lock = threading.RLock()
        self._sources = {}
        self._walls = {}
        self._viewers = {}
        self._history = {}
        self._stopping = set()
        self._closed = False
        self._server = FrameServer(self)
        self._server.start()

    def descriptor(self, camera_id, source):
        with self._lock:
            if self._closed:
                raise RuntimeError("preview is closing")
            if camera_id in self._sources and self._sources[camera_id] != source:
                raise RuntimeError("camera source changed; reopen preview")
            self._sources[camera_id] = source
            return {"kind": "mjpeg", "preview": True,
                    "url": f"http://127.0.0.1:{self._server.port}/stream/"
                    f"{quote(camera_id, safe='')}?{urlencode({'token': self._server.token})}"}

    def acquire(self, camera_id):
        with self._lock:
            if self._closed or camera_id not in self._sources:
                return False
            if camera_id in self._stopping:
                wall = self._walls[camera_id]
                if any(t.is_alive() for t in wall._threads):
                    return False
                self._remember(camera_id, wall)
                del self._walls[camera_id]
                self._stopping.discard(camera_id)
            if camera_id not in self._walls:
                self._walls[camera_id] = LiveWall(
                    [{"id": camera_id, "source": self._sources[camera_id]}],
                    width=960, fps=8, quality=65, retain_snapshot=True).start()
            self._viewers[camera_id] = self._viewers.get(camera_id, 0) + 1
            return True

    def release(self, camera_id):
        wall = None
        with self._lock:
            self._viewers[camera_id] = max(0, self._viewers.get(camera_id, 0) - 1)
            if (not self._viewers[camera_id] and camera_id in self._walls
                    and camera_id not in self._stopping):
                wall = self._walls[camera_id]
                self._stopping.add(camera_id)
        # A blocked camera must not hold the manager lock for every other feed.
        if wall is not None:
            wall.stop()
            with self._lock:
                if self._walls.get(camera_id) is not wall:
                    return
                if not any(t.is_alive() for t in wall._threads):
                    self._remember(camera_id, wall)
                    del self._walls[camera_id]
                    self._stopping.discard(camera_id)

    def _remember(self, camera_id, wall):
        self._history[camera_id] = wall.diagnostics().get(camera_id, {})
        while len(self._history) > 128:
            del self._history[next(iter(self._history))]

    def diagnostics(self):
        import time
        with self._lock:
            records = {key: dict(value, active=False) for key, value in self._history.items()}
            for key, wall in self._walls.items():
                records[key] = dict(wall.diagnostics().get(key, {}),
                                    active=not self._closed and key not in self._stopping,
                                    stopping=key in self._stopping,
                                    reader_alive=any(t.is_alive() for t in wall._threads),
                                    viewers=self._viewers.get(key, 0))
            for key in self._sources:
                records.setdefault(key, {"active": False, "state": "not_started", "viewers": 0})
            for record in records.values():
                last = record.get("last_decoded_at")
                record["last_decoded_age_s"] = round(time.time() - last, 2) if last else None
            return {"captured_at": time.time(), "mode": "preview", "cameras": records}

    def jpeg(self, camera_id):
        with self._lock:
            wall = self._walls.get(camera_id)
            return wall.jpeg(camera_id) if wall and not self._closed and camera_id not in self._stopping else None

    def snapshot(self, camera_id):
        with self._lock:
            wall = self._walls.get(camera_id)
            if wall is None or self._closed or camera_id in self._stopping:
                return None
        return wall.snapshot(camera_id) or {"error": "preview is connecting; retry the snapshot shortly"}

    def close(self):
        with self._lock:
            self._closed = True
            for wall in self._walls.values():
                wall.stop()
            stopped = all(not t.is_alive() for wall in self._walls.values()
                          for t in wall._threads)
        self._server.stop()
        if not stopped:
            raise PreviewBusy("Camera preview is still releasing its capture; retry monitoring shortly")
