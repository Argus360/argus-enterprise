"""The engine API is the contract Demi's Electron app builds against (6 Sep).

Pins the read-only surface of contract v0.2: auth gates every route, the data
endpoints return the agreed shapes from a real events.db + gate_health.json,
credentials never appear in a camera source, the live WebSocket hydrates on
connect and pushes new alerts, and the mock server answers the same shapes so
the frontend can build with no engine.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from cvti.api.app import create_app
from cvti.api.mock import create_mock_app

PREFIX = "/api/v1"


def _make_site(tmp: Path) -> tuple[str, str]:
    """A site config + an events.db with one account and two events."""
    db = tmp / "events.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, iso TEXT, camera_id TEXT, rule TEXT, priority TEXT,
        confidence REAL, reason TEXT, track_id INTEGER, zone TEXT,
        object_label TEXT, evidence_dir TEXT, review TEXT, reviewed_at TEXT)""")
    con.execute("INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason,evidence_dir) "
                "VALUES (1,'2026-09-06T10:00:00Z','Dublin Street','custom:hoodie','high',0.95,'a hoodie','/e/1')")
    con.execute("INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason,review) "
                "VALUES (2,'2026-09-06T10:01:00Z','Loading Bay','video_theft','critical',0.9,'theft','false')")
    con.commit(); con.close()

    from cvti.security.accounts import AccountStore
    from cvti.security import permissions as perms
    store = AccountStore(tmp / "auth.db")
    store.create_user("ayo", "Argus-Fresh-2026", role=perms.OWNER)

    site = tmp / "site.json"
    site.write_text(json.dumps({"cameras": [
        {"id": "Dublin Street", "source": "rtsp://user:secret@10.0.0.9/stream1", "config": "x"},
    ]}))
    (tmp / "gate_health.json").write_text(json.dumps({
        "status": "ok", "generated_at": 9e12, "version": "test",
        "engine": {"phase": "monitoring"},
        "cameras": [{"camera_id": "Dublin Street", "state": "connected",
                     "last_frame_age_s": 0.2, "ingest": {"width": 1280, "sampling_fps": 12}}],
    }))
    return str(db), str(site)


class RealApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        db, site = _make_site(tmp)
        self.app = create_app(db_path=db, site_path=site)
        self.client = TestClient(self.app)
        self.auth_db = tmp / "auth.db"
        self.addCleanup(self._tmp.cleanup)

    def _token(self) -> str:
        r = self.client.post(f"{PREFIX}/auth/session",
                             json={"username": "ayo", "password": "Argus-Fresh-2026"})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["token"]

    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self._token()}"}

    def test_unauthenticated_is_refused(self):
        self.assertEqual(self.client.get(f"{PREFIX}/events").status_code, 401)
        self.assertEqual(self.client.get(f"{PREFIX}/system/health").status_code, 401)

    def test_slash_camera_can_be_read_previewed_and_deleted(self):
        from unittest.mock import PropertyMock, patch
        camera_id = "SMB BAY 3/4"
        site = Path(self.app.state.site_path)
        site.write_text(json.dumps({"cameras": [
            {"id": camera_id, "source": "rtsp://example/stream"},
            {"id": "KC Production 2", "source": "rtsp://example/other"},
        ]}))
        headers = self._auth()
        params = {"camera_id": camera_id}
        self.assertEqual(self.client.get(f"{PREFIX}/camera-by-id", params=params).status_code, 401)
        read = self.client.get(f"{PREFIX}/camera-by-id", params=params, headers=headers)
        self.assertEqual(read.status_code, 200, read.text)
        self.assertEqual(read.json()["id"], camera_id)
        host = self.app.state.backend_host
        with patch.object(type(host), "engine_alive", new_callable=PropertyMock, return_value=False), \
             patch.object(host, "call", return_value={"kind": "mjpeg", "url": "http://127.0.0.1:9000/stream/test"}) as call:
            preview = self.client.get(f"{PREFIX}/camera-by-id/stream", params=params, headers=headers)
            self.assertEqual(preview.status_code, 200, preview.text)
            self.assertEqual(call.call_args.kwargs["camera_id"], camera_id)
        deleted = self.client.delete(f"{PREFIX}/camera-by-id", params=params, headers=headers)
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual([c["id"] for c in json.loads(site.read_text())["cameras"]], ["KC Production 2"])

    def test_query_camera_operations_require_an_id(self):
        response = self.client.delete(f"{PREFIX}/camera-by-id", headers=self._auth())
        self.assertEqual(response.status_code, 400)

    def test_failed_rtsp_probe_is_an_api_error_not_success(self):
        from unittest.mock import patch
        headers = self._auth()
        with patch("cvti.serving.discovery.probe_rtsp", return_value={
            "ok": False, "kind": "auth", "message": "Wrong camera credentials",
        }), patch("cvti.serving.onboarding.test_url") as decode:
            response = self.client.post(f"{PREFIX}/cameras/probe", headers=headers,
                                        json={"url": "rtsp://example/stream"})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["error"]["message"], "Wrong camera credentials")
        decode.assert_not_called()

    def test_all_camera_query_aliases_preserve_id_and_authorization(self):
        from unittest.mock import patch
        from cvti.api.writes import ROUTES
        headers = self._auth()
        for route in ROUTES:
            if not route.path.startswith("/cameras/{camera_id}"):
                continue
            alias = route.path.replace("/cameras/{camera_id}", "/camera-by-id", 1)
            for name in route.path_map:
                alias = alias.replace("{" + name + "}", "example")
            with self.subTest(route=route.path, method=route.verb), \
                 patch.object(self.app.state.backend_host, "call", return_value={"ok": True}) as call:
                denied = self.client.request(route.verb, PREFIX + alias, params={"camera_id": "SMB BAY 3/4"})
                self.assertEqual(denied.status_code, 401)
                call.assert_not_called()
                response = self.client.request(route.verb, PREFIX + alias, params={"camera_id": "SMB BAY 3/4"}, headers=headers)
                self.assertEqual(response.status_code, route.status, response.text)
                if route.path_map.get("camera_id") is not None:
                    self.assertEqual(call.call_args.kwargs["camera_id"], "SMB BAY 3/4")

    def test_health_and_monitor_agree_before_first_heartbeat_and_after_stop(self):
        from unittest.mock import PropertyMock, patch
        headers = self._auth()
        Path(self._tmp.name, "gate_health.json").unlink()
        host = self.app.state.backend_host
        for alive, phase in ((True, "starting"), (False, "stopped")):
            with patch.object(type(host), "engine_alive", new_callable=PropertyMock, return_value=alive):
                health = self.client.get(f"{PREFIX}/system/health", headers=headers).json()
                monitor = self.client.get(f"{PREFIX}/monitor", headers=headers).json()
            self.assertEqual(health["engine"]["running"], alive)
            self.assertEqual(health["engine"]["phase"], phase)
            self.assertEqual(health["engine"]["starting"], monitor["starting"])

    def test_index_is_self_describing_and_needs_no_auth(self):
        for path in ("/", PREFIX):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, path)
            body = r.json()
            self.assertEqual(body["name"], "Argus Engine API")
            self.assertEqual(body["docs"], "/docs")
            self.assertFalse(body["mock"])
            self.assertIn(f"{PREFIX}/system/health", body["endpoints"])

    def test_sign_in_bad_password(self):
        r = self.client.post(f"{PREFIX}/auth/session",
                             json={"username": "ayo", "password": "wrong"})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["error"]["code"], "unauthorized")

    def test_sign_in_returns_token_and_role(self):
        r = self.client.post(f"{PREFIX}/auth/session",
                             json={"username": "ayo", "password": "Argus-Fresh-2026"})
        body = r.json()
        self.assertIn("token", body)
        self.assertEqual(body["user"]["role"], "owner")
        self.assertIn("view_alerts", body["user"]["permissions"])

    def test_health_shape(self):
        r = self.client.get(f"{PREFIX}/system/health", headers=self._auth())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertEqual(r.json()["engine"]["phase"], "monitoring")

    def test_cameras_redact_credentials(self):
        r = self.client.get(f"{PREFIX}/cameras", headers=self._auth())
        cams = r.json()
        self.assertEqual(len(cams), 1)
        self.assertNotIn("secret", cams[0]["source"])
        self.assertIn("rtsp://***@10.0.0.9/stream1", cams[0]["source"])
        self.assertEqual(cams[0]["state"], "connected")   # merged from health

    def test_sign_out_returns_empty_204_not_crashing_null(self):
        # DELETE /auth/session is a 204: the body MUST be empty. Returning
        # JSONResponse(None) wrote "null" against Content-Length 0 and crashed
        # the uvicorn worker on every logout (12 Sep field). Empty 204 now.
        r = self.client.delete(f"{PREFIX}/auth/session", headers=self._auth())
        self.assertEqual(r.status_code, 204)
        self.assertEqual(r.content, b"")

    def test_cameras_presets_is_not_swallowed_by_the_camera_id_route(self):
        # /cameras/presets must resolve to its own route, not fall into
        # /cameras/{camera_id} as camera_id='presets' (the 12 Sep field UI
        # error "no such camera 'presets'"). Empty dict matches the mock.
        site = Path(self.app.state.site_path)
        site.write_text(json.dumps({"cameras": [{
            "id": "presets", "source": "rtsp://example/presets",
        }]}))
        r = self.client.get(f"{PREFIX}/cameras/presets", headers=self._auth())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {})

    def test_slash_camera_name_reaches_proxy_stream_and_mjpeg(self):
        """An encoded slash becomes a path separator before Starlette matches."""
        from unittest.mock import patch

        camera_id = "SMB Bay 3/4"
        site = Path(self.app.state.site_path)
        site.write_text(json.dumps({"cameras": [{
            "id": camera_id, "source": "rtsp://example/smb-bay",
        }]}))
        Path(self._tmp.name, "frames.json").write_text(json.dumps({
            "port": 8765, "token": "publisher-secret",
        }))

        class LoopbackResponse:
            status_code = 200
            headers = {"content-type": "multipart/x-mixed-replace; boundary=frame"}

            async def aiter_raw(self):
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nframe\r\n"

            async def aclose(self):
                pass

        class LoopbackClient:
            requests = []

            def __init__(self, **_kwargs):
                pass

            def build_request(self, method, url):
                self.requests.append((method, url))
                return (method, url)

            async def send(self, request, stream=False):
                self.requests.append((request, stream))
                return LoopbackResponse()

            async def aclose(self):
                pass

        headers = self._auth()
        encoded = "SMB%20Bay%203%2F4"
        with patch("httpx.AsyncClient", LoopbackClient):
            detail = self.client.get(f"{PREFIX}/camera-by-id", params={
                "camera_id": camera_id,
            }, headers=headers)
            self.assertEqual(detail.status_code, 200, detail.text)
            self.assertEqual(detail.json()["id"], camera_id)
            descriptor = self.client.get(
                f"{PREFIX}/cameras/{encoded}/stream?proxy=1", headers=headers)
            self.assertEqual(descriptor.status_code, 200, descriptor.text)
            self.assertEqual(descriptor.json()["kind"], "mjpeg")
            self.assertEqual(
                descriptor.json()["url"],
                f"{PREFIX}/cameras/{encoded}/mjpeg?token=publisher-secret",
            )
            proxied = self.client.get(descriptor.json()["url"])

        self.assertEqual(proxied.status_code, 200, proxied.text)
        self.assertIn(b"frame", proxied.content)
        self.assertEqual(LoopbackClient.requests[0][0], "GET")
        self.assertIn("/stream/SMB%20Bay%203%2F4?token=publisher-secret",
                      LoopbackClient.requests[0][1])

    def test_reads_follow_the_active_feed_not_the_boot_site(self):
        # switch_feed moves the backend to another feed's config + event store;
        # every data route must serve THAT feed. Before this, they read the
        # db/site frozen at boot: Live EarthCams selected, engine running on
        # it, and the wall + incidents still showed the webcam's DB (12 Sep).
        tmp = Path(self._tmp.name)
        other_db = tmp / "feeds" / "live" / "events.db"
        other_db.parent.mkdir(parents=True)
        con = sqlite3.connect(other_db)
        con.execute("""CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, iso TEXT, camera_id TEXT, rule TEXT, priority TEXT,
            confidence REAL, reason TEXT, track_id INTEGER, zone TEXT,
            object_label TEXT, evidence_dir TEXT, review TEXT, reviewed_at TEXT)""")
        con.execute("INSERT INTO events (ts,iso,camera_id,rule,priority,confidence,reason) "
                    "VALUES (3,'2026-09-12T14:00:00Z','Dublin Street','loitering_watch','medium',0.99,'zone')")
        con.commit(); con.close()
        other_site = tmp / "live_camera.json"
        other_site.write_text(json.dumps({"cameras": [
            {"id": "Dublin Street", "source": "https://cdn/live.m3u8", "config": "x"},
        ]}))

        host = self.app.state.backend_host
        host._backend = host._build()               # what the first write call does
        host._backend.site_path = str(other_site)   # what switch_feed does
        host._backend.db_path = str(other_db)

        headers = self._auth()                       # auth still on the HOME store
        evs = self.client.get(f"{PREFIX}/events", headers=headers).json()["events"]
        self.assertEqual([e["camera_id"] for e in evs], ["Dublin Street"])
        cams = self.client.get(f"{PREFIX}/cameras", headers=headers).json()
        self.assertEqual([c["id"] for c in cams], ["Dublin Street"])
        self.assertEqual(self.client.get(f"{PREFIX}/triage", headers=headers).json()["total"], 1)

    def test_monitor_and_stream_follow_the_engine_owner_not_the_stale_heartbeat(self):
        # gate_health.json here is fresh, so the heartbeat alone says
        # "monitoring". The backend that OWNS the engine process says nothing
        # is running: Stop must read as stopped at once, not up to 30s later
        # ("I pressed stop monitoring but it didn't stop", 12 Sep), and a tile
        # must get preview, not a stale publisher URL. Descriptor lookup alone
        # must not open a camera or load inference models.
        host = self.app.state.backend_host
        host._backend = host._build()
        host._backend._engine_owned = True           # what Start/Stop leave behind
        host._backend._monitor = None
        headers = self._auth()
        mon = self.client.get(f"{PREFIX}/monitor", headers=headers).json()
        self.assertFalse(mon["running"])
        self.assertEqual(mon["phase"], "stopped")
        (Path(self._tmp.name) / "frames.json").write_text('{"port": 1, "token": "t"}')
        from unittest.mock import patch
        with patch("cvti.app.preview.FrameServer.start", return_value=12345):
            r = self.client.get(f"{PREFIX}/cameras/Dublin Street/stream", headers=headers)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["preview"])
        self.assertNotIn("127.0.0.1:1/", r.json()["url"])
        self.assertEqual(host._backend._preview._walls, {})
        host._backend.close()

    def test_unowned_stale_engine_uses_preview_not_old_stream_address(self):
        tmp = Path(self._tmp.name)
        (tmp / "gate_health.json").write_text(json.dumps({"generated_at": 1}))
        (tmp / "frames.json").write_text('{"port": 1, "token": "t"}')
        from unittest.mock import patch
        with patch("cvti.app.preview.FrameServer.start", return_value=12345):
            response = self.client.get(
                f"{PREFIX}/cameras/Dublin%20Street/stream", headers=self._auth())
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["preview"])
        self.assertNotIn("127.0.0.1:1/", response.json()["url"])
        self.app.state.backend_host._backend.close()

    def test_tracking_stream_forces_authenticated_local_mjpeg(self):
        tmp = Path(self._tmp.name)
        (tmp / "frames.json").write_text(json.dumps({
            "port": 5599,
            "token": "publisher-secret",
        }))
        (tmp / "stream_gateway.json").write_text(json.dumps({
            "api_port": 1984,
            "streams": {"Dublin Street": "dublin-street"},
        }))
        headers = self._auth()

        raw = self.client.get(
            f"{PREFIX}/cameras/Dublin%20Street/stream", headers=headers
        )
        tracking = self.client.get(
            f"{PREFIX}/cameras/Dublin%20Street/stream?tracking=true",
            headers=headers,
        )

        self.assertEqual(raw.status_code, 200, raw.text)
        self.assertEqual(raw.json()["kind"], "webrtc")
        self.assertEqual(tracking.status_code, 200, tracking.text)
        self.assertEqual(tracking.json(), {
            "kind": "mjpeg",
            "url": (
                "http://127.0.0.1:5599/stream/Dublin%20Street"
                "?tracking=1&token=publisher-secret"
            ),
        })

    def test_monitor_says_running_while_the_owned_engine_lives_through_a_long_preflight(self):
        # Heartbeat stale (scene mapping ran minutes without one), process
        # alive: the button must keep reading "Stop monitoring", not flip to
        # Start and invite a click that kills a half-started engine (12 Sep).
        import types
        (Path(self._tmp.name) / "gate_health.json").write_text(json.dumps({
            "generated_at": 1.0, "engine": {"phase": "starting — mapping camera scenes"}}))
        host = self.app.state.backend_host
        host._backend = host._build()
        host._backend._engine_owned = True
        host._backend._monitor = types.SimpleNamespace(poll=lambda: None, pid=4242)
        mon = self.client.get(f"{PREFIX}/monitor", headers=self._auth()).json()
        self.assertTrue(mon["running"])
        self.assertTrue(mon["starting"])
        self.assertEqual(mon["phase"], "starting — mapping camera scenes")

    def test_switching_feeds_while_monitoring_restarts_the_engine_as_the_caller(self):
        # switch_feed works on a background thread, after the API has cleared
        # its per-request impersonation. The stop/start inside it used to hit
        # the permission gate as '<anonymous>' and fail every time, so from
        # Demi's app a feed switch while monitoring left the engine on the old
        # feed with a permission error in the switcher (12 Sep).
        import types
        host = self.app.state.backend_host
        host._backend = host._build()
        b = host._backend

        class FakeEngine:
            def __init__(self): self.alive, self.pid = True, 4242
            def poll(self): return None if self.alive else 0
            def terminate(self): self.alive = False
            def wait(self, timeout=None): return 0
            def kill(self): self.alive = False

        first, second = FakeEngine(), FakeEngine()
        b._monitor, b._engine_owned = first, True
        b._spawn_engine = lambda: second
        principal = types.SimpleNamespace(username="ayo", role="owner")
        out = host.call(principal, "switch_feed", key="stage")
        self.assertTrue(out["ok"], out)
        deadline = time.time() + 30
        while time.time() < deadline and host.call(principal, "feed_switch_status")["busy"]:
            time.sleep(0.1)
        st = host.call(principal, "feed_switch_status")
        self.assertIsNone(st["error"], st)
        self.assertTrue(st["engine_restarted"], st)
        self.assertFalse(first.alive)              # old engine stopped
        self.assertIs(b._monitor, second)          # new one running on the new feed
        b._monitor_should_run = False              # let the watchdog thread exit

    def test_start_while_a_preview_is_still_releasing_is_a_retryable_503(self):
        # Preview a webcam, press Start: the engine closes the preview first,
        # and if the capture thread is still inside a blocked read the start is
        # refused. That refusal used to escape the bridge as a bare 500 — the
        # app showed nothing the operator could act on (#198, 30 Sep).
        from unittest.mock import MagicMock
        from cvti.app.errors import PreviewBusy
        host = self.app.state.backend_host
        host._backend = host._build()
        b = host._backend
        b._preview = MagicMock()
        b._preview.close.side_effect = PreviewBusy(
            "Camera preview is still releasing its capture; retry monitoring shortly")
        b._bundled_engine = lambda: object()        # not the viewer-only demo branch
        r = self.client.post(f"{PREFIX}/engine/start", headers=self._auth())
        self.assertEqual(r.status_code, 503, r.text)
        self.assertEqual(r.json()["error"]["code"], "preview_releasing")
        self.assertIn("retry", r.json()["error"]["message"])
        self.assertIsNone(b._preview)              # Watch gets a fresh preview next
        self.assertIsNone(b._monitor)              # and no engine was spawned
        b._monitor_should_run = False

    def test_monitor_trusts_the_heartbeat_when_nobody_owns_an_engine(self):
        # No Start/Stop has gone through this API (a headless engine from a
        # terminal, say): the heartbeat file still decides, as before.
        mon = self.client.get(f"{PREFIX}/monitor", headers=self._auth()).json()
        self.assertTrue(mon["running"])

    def test_events_list_and_shape(self):
        r = self.client.get(f"{PREFIX}/events", headers=self._auth())
        body = r.json()
        self.assertEqual(len(body["events"]), 2)
        newest = body["events"][0]
        self.assertTrue(newest["id"].startswith("evt_"))
        self.assertIn(newest["verdict"], ("confirmed", "rejected"))
        self.assertIn("thumb", newest["evidence"])

    def test_events_cursor_paginates(self):
        first = self.client.get(f"{PREFIX}/events?limit=1", headers=self._auth()).json()
        self.assertEqual(len(first["events"]), 1)
        self.assertIsNotNone(first["next_cursor"])
        nxt = self.client.get(f"{PREFIX}/events?limit=1&cursor={first['next_cursor']}",
                              headers=self._auth()).json()
        self.assertEqual(len(nxt["events"]), 1)
        self.assertNotEqual(first["events"][0]["id"], nxt["events"][0]["id"])

    def test_event_detail_and_404(self):
        got = self.client.get(f"{PREFIX}/events/evt_1", headers=self._auth())
        self.assertEqual(got.status_code, 200)
        self.assertEqual(got.json()["camera_id"], "Dublin Street")
        miss = self.client.get(f"{PREFIX}/events/evt_999", headers=self._auth())
        self.assertEqual(miss.status_code, 404)

    def test_triage_counts(self):
        t = self.client.get(f"{PREFIX}/triage", headers=self._auth()).json()
        self.assertEqual(t["total"], 2)
        self.assertEqual(t["to_review"], 1)          # one has review=NULL

    def test_websocket_hydrates_and_pushes(self):
        token = self._token()
        with self.client.websocket_connect(
            f"{PREFIX}/stream",
            subprotocols=["argus.v1", f"argus.token.{token}"],
        ) as ws:
            first = ws.receive_json()
            self.assertEqual(first["type"], "health")
            second = ws.receive_json()
            self.assertEqual(second["type"], "triage")

    def test_websocket_rejects_bad_token(self):
        with self.assertRaises(WebSocketDisconnect) as caught:
            with self.client.websocket_connect(
                f"{PREFIX}/stream",
                subprotocols=["argus.v1", "argus.token.nope"],
            ) as ws:
                ws.receive_json()
        self.assertEqual(caught.exception.code, 4401)

    def test_websocket_closes_when_its_token_expires(self):
        token = self._token()
        principal = self.app.state.tokens.resolve(token)
        with self.client.websocket_connect(
            f"{PREFIX}/stream",
            subprotocols=["argus.v1", f"argus.token.{token}"],
        ) as ws:
            ws.receive_json()
            ws.receive_json()
            principal.expires_at = time.time() - 1
            with self.assertRaises(WebSocketDisconnect) as caught:
                ws.receive_json()
        self.assertEqual(caught.exception.code, 4401)

    def test_websocket_closes_when_its_token_is_revoked(self):
        token = self._token()
        with self.client.websocket_connect(
            f"{PREFIX}/stream",
            subprotocols=["argus.v1", f"argus.token.{token}"],
        ) as ws:
            ws.receive_json()
            ws.receive_json()
            self.app.state.tokens.revoke(token)
            with self.assertRaises(WebSocketDisconnect) as caught:
                ws.receive_json()
        self.assertEqual(caught.exception.code, 4401)

    def test_websocket_closes_when_the_account_role_changes(self):
        from cvti.security.accounts import AccountStore

        token = self._token()
        with self.client.websocket_connect(
            f"{PREFIX}/stream",
            subprotocols=["argus.v1", f"argus.token.{token}"],
        ) as ws:
            ws.receive_json()
            ws.receive_json()
            store = AccountStore(self.auth_db)
            store.set_role("ayo", "installer")
            store.close()
            with self.assertRaises(WebSocketDisconnect) as caught:
                ws.receive_json()
        self.assertEqual(caught.exception.code, 4401)

    def test_websocket_closes_when_the_account_is_deleted(self):
        from cvti.security.accounts import AccountStore

        token = self._token()
        with self.client.websocket_connect(
            f"{PREFIX}/stream",
            subprotocols=["argus.v1", f"argus.token.{token}"],
        ) as ws:
            ws.receive_json()
            ws.receive_json()
            store = AccountStore(self.auth_db)
            store.delete_user("ayo")
            store.close()
            with self.assertRaises(WebSocketDisconnect) as caught:
                ws.receive_json()
        self.assertEqual(caught.exception.code, 4401)


class MockApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_mock_app())

    def test_any_login_works_and_shapes_match(self):
        tok = self.client.post(f"{PREFIX}/auth/session",
                               json={"username": "demo", "password": "x"}).json()["token"]
        h = {"Authorization": f"Bearer {tok}"}
        self.assertEqual(self.client.get(f"{PREFIX}/system/health", headers=h).json()["gate"]["model"],
                         "gemma3:4b")
        cams = self.client.get(f"{PREFIX}/cameras", headers=h).json()
        self.assertTrue(any(c["view_only"] for c in cams))
        for c in cams:
            self.assertNotIn("secret", c["source"])
        evs = self.client.get(f"{PREFIX}/events", headers=h).json()["events"]
        self.assertTrue(evs and evs[0]["id"].startswith("evt_"))

    def test_mock_requires_a_token_too(self):
        self.assertEqual(self.client.get(f"{PREFIX}/events").status_code, 401)

    def test_mock_index_says_it_is_mock(self):
        body = self.client.get("/").json()
        self.assertTrue(body["mock"])
        self.assertEqual(body["docs"], "/docs")


if __name__ == "__main__":
    unittest.main()
