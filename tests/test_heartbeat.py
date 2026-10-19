"""Heartbeat sender + receiver (EP-04-T2).

The three properties that make an opt-in heartbeat compatible with a
privacy-first product, each enforced here rather than promised: off by default,
health-only by whitelist, and inspectable.
"""
import json
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from cvti.serving.heartbeat import SCHEMA_VERSION, Heartbeat, heartbeat_payload

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from heartbeat_receiver import (  # noqa: E402
    MISSED_AFTER_INTERVALS,
    Alerter,
    Receiver,
    render_dashboard,
    site_view,
)

FULL_HEALTH = {
    "status": "degraded",
    "reasons": ["camera backroom reconnecting"],
    "uptime_s": 3600.0,
    "cameras": [{"camera_id": "aisle_1", "state": "connected", "last_frame_age_s": 0.3,
                 "reconnects": 0, "attempts": [{"at": 1, "backoff": 2}]}],
    "gate": {"provider": "ollama", "model": "gemma3:4b", "reachable": True,
             "verified": 10, "unverified": 0, "errors": 0, "median_latency_s": 11.0,
             "last_error": "aisle_1::weapons — timeout"},
    "disk": {"used_pct": 61.0, "free_gb": 180.0, "level": "ok", "total_gb": 460.0},
    "memory": {"available_gb": 6.0, "rss_gb": 3.0, "level": "ok"},
    "components": {"components": [{"name": "detector.aisle_1", "processed": 100,
                                   "errors": 1, "degraded": False,
                                   "last_error": "ValueError: boom"}],
                   "degraded": []},
    "engine": {"frames_processed": 100, "alerts_queued": 2, "cameras": 1,
               "target_fps": 4},
    "self_test": {"ok": True, "at": 123.0, "steps": {"frame": "ok"}},
    # Things that exist in the health doc but must NOT travel:
    "retention": {"held": {"legal_hold": 1}},
    "banner": "", "mock": False,
}


class WhitelistTest(unittest.TestCase):
    def test_payload_carries_the_health_signals(self):
        p = heartbeat_payload(FULL_HEALTH, site_id="test-site")
        self.assertEqual(p["schema"], SCHEMA_VERSION)
        self.assertEqual(p["site_id"], "test-site")
        self.assertEqual(p["status"], "degraded")
        self.assertEqual(p["cameras"][0]["id"], "aisle_1")
        self.assertEqual(p["gate"]["median_latency_s"], 11.0)
        self.assertEqual(p["disk"]["used_pct"], 61.0)
        self.assertEqual(p["components"][0]["errors"], 1)
        self.assertTrue(p["self_test"]["ok"])

    def test_nothing_outside_the_whitelist_travels(self):
        # Whitelist, not redaction: unknown fields must vanish by construction.
        health = dict(FULL_HEALTH)
        health["surprise_new_field"] = {"frames": ["<jpeg bytes>"]}
        blob = json.dumps(heartbeat_payload(health, site_id="s"))
        self.assertNotIn("surprise_new_field", blob)
        self.assertNotIn("jpeg", blob)
        # Error strings and retention details stay home too.
        self.assertNotIn("last_error", blob)
        self.assertNotIn("ValueError", blob)
        self.assertNotIn("legal_hold", blob)
        self.assertNotIn("attempts", blob)

    def test_payload_top_level_keys_are_exactly_the_documented_schema(self):
        p = heartbeat_payload(FULL_HEALTH, site_id="s")
        self.assertEqual(sorted(p), sorted([
            "schema", "site_id", "sent_at", "status", "reasons", "uptime_s",
            "cameras", "gate", "disk", "memory", "components", "engine",
            "self_test", "version"]))


class SenderTest(unittest.TestCase):
    def test_off_by_default_in_site_meta(self):
        import tempfile as tf
        from cvti.serving.onboarding import get_site_meta
        with tf.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site.json"
            site.write_text('{"cameras": []}')
            meta = get_site_meta(site)
            self.assertEqual(meta["heartbeat_url"], "", "heartbeat was on by default")

    def test_beat_sends_and_records(self):
        sent = []
        with tempfile.TemporaryDirectory() as tmp:
            hb = Heartbeat(url="https://x", site_key="k", site_id="s",
                           health_provider=lambda: FULL_HEALTH, output_dir=tmp,
                           transport=lambda p: sent.append(p) or 200)
            self.assertTrue(hb.beat())
            self.assertEqual(hb.sent, 1)
            self.assertEqual(sent[0]["site_id"], "s")

    def test_the_customer_can_inspect_exactly_what_was_sent(self):
        with tempfile.TemporaryDirectory() as tmp:
            sent = []
            hb = Heartbeat(url="https://x", site_key="k", site_id="s",
                           health_provider=lambda: FULL_HEALTH, output_dir=tmp,
                           transport=lambda p: sent.append(p) or 200)
            hb.beat()
            on_disk = json.loads((Path(tmp) / "heartbeat_last.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk, json.loads(json.dumps(sent[0], default=str)))

    def test_a_failed_send_is_recorded_and_does_not_raise(self):
        def broken(_p):
            raise ConnectionError("no route to host")
        with tempfile.TemporaryDirectory() as tmp:
            hb = Heartbeat(url="https://x", site_key="k", site_id="s",
                           health_provider=lambda: FULL_HEALTH, output_dir=tmp,
                           transport=broken)
            self.assertFalse(hb.beat())
            self.assertIn("ConnectionError", hb.last_error)
            # ...and the inspectable file still shows what we TRIED to send.
            self.assertTrue((Path(tmp) / "heartbeat_last.json").exists())

    def test_a_4xx_from_the_receiver_counts_as_failure(self):
        hb = Heartbeat(url="https://x", site_key="wrong", site_id="s",
                       health_provider=lambda: FULL_HEALTH, transport=lambda p: 401)
        self.assertFalse(hb.beat())


class ReceiverTest(unittest.TestCase):
    def _receiver(self, tmp, **kw):
        return Receiver(keys={"site-a": "key-a", "site-b": "key-b"},
                        db_path=Path(tmp) / "hb.db", **kw)

    def _payload(self, site="site-a", status="ok", **extra):
        return json.dumps({"schema": 1, "site_id": site, "status": status,
                           "reasons": [], "cameras": [], **extra}).encode()

    def test_a_valid_heartbeat_is_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._receiver(tmp)
            code, _ = r.handle_heartbeat("key-a", self._payload())
            self.assertEqual(code, 200)
            self.assertIn("site-a", r.store.latest())
            r.close()                  # Windows will not delete an open database

    def test_a_wrong_or_unknown_key_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._receiver(tmp)
            self.assertEqual(r.handle_heartbeat("nope", self._payload())[0], 401)
            self.assertEqual(
                r.handle_heartbeat("key-a", self._payload(site="intruder"))[0], 401)
            self.assertEqual(r.store.latest(), {})
            r.close()

    def test_garbage_is_a_400_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._receiver(tmp)
            self.assertEqual(r.handle_heartbeat("key-a", b"\xff{")[0], 400)
            r.close()

    def test_a_silent_site_is_missed_regardless_of_what_it_last_said(self):
        now = time.time()
        view = site_view("site-a",
                         {"received_at": now - MISSED_AFTER_INTERVALS * 300 - 60,
                          "status": "ok", "payload": {"status": "ok"}},
                         known=True, now=now, interval=300)
        self.assertEqual(view["state"], "missed")

    def test_a_recent_site_shows_its_own_status(self):
        now = time.time()
        view = site_view("site-a", {"received_at": now - 30, "status": "degraded",
                                    "payload": {"status": "degraded",
                                                "reasons": ["disk 91% used"]}},
                         known=True, now=now)
        self.assertEqual(view["state"], "degraded")
        self.assertEqual(view["reasons"], ["disk 91% used"])

    def test_an_enrolled_site_that_never_reported_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._receiver(tmp)
            views = {v["site_id"]: v for v in r.views()}
            self.assertEqual(views["site-a"]["state"], "never-reported")
            r.close()

    def test_end_to_end_over_real_http(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            r = self._receiver(tmp, dashboard_token="dash")
            # serve() blocks; drive the HTTP layer on a thread with a real socket
            thread = threading.Thread(target=r.serve, args=("127.0.0.1", 0), daemon=True)
            thread.start()
            for _ in range(100):
                if r._server:
                    break
                time.sleep(0.02)
            port = r._server.server_address[1]

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/heartbeat", data=self._payload(),
                headers={"X-Argus-Site-Key": "key-a", "Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=3) as resp:
                self.assertEqual(resp.status, 200)

            # dashboard requires its token
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3)
                self.fail("dashboard served without a token")
            except urllib.error.HTTPError as exc:
                self.assertEqual(exc.code, 401)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/?token=dash",
                                        timeout=3) as resp:
                body = resp.read().decode()
            self.assertIn("site-a", body)
            r._server.shutdown()
            r.close()


class AlerterTest(unittest.TestCase):
    def _view(self, state, site="site-a", reasons=None):
        return {"site_id": site, "state": state, "age_s": 900.0,
                "reasons": reasons or [], "known": True}

    def test_alerts_once_per_transition_not_per_check(self):
        sent = []
        a = Alerter(notify=sent.append)
        a.observe(self._view("ok"))
        a.observe(self._view("missed"))
        a.observe(self._view("missed"))
        a.observe(self._view("missed"))
        self.assertEqual(len(sent), 1)
        self.assertIn("stopped heartbeating", sent[0])

    def test_degradation_names_its_reasons(self):
        sent = []
        a = Alerter(notify=sent.append)
        a.observe(self._view("ok"))
        a.observe(self._view("critical", reasons=["camera backroom offline 400s"]))
        self.assertIn("backroom", sent[0])

    def test_recovery_is_announced(self):
        sent = []
        a = Alerter(notify=sent.append)
        a.observe(self._view("missed"))
        a.observe(self._view("ok"))
        self.assertEqual(len(sent), 2)
        self.assertIn("back to normal", sent[1])

    def test_a_site_that_starts_healthy_is_silent(self):
        sent = []
        Alerter(notify=sent.append).observe(self._view("ok"))
        self.assertEqual(sent, [])


class DashboardTest(unittest.TestCase):
    def test_worst_sites_sort_first_and_render(self):
        html_out = render_dashboard([
            {"site_id": "fine", "state": "ok", "age_s": 20.0, "reasons": [],
             "cameras": [{"state": "connected"}], "known": True},
            {"site_id": "dead", "state": "missed", "age_s": 4000.0, "reasons": [],
             "cameras": [], "known": True},
        ])
        self.assertLess(html_out.index("dead"), html_out.index("fine"))
        self.assertIn("MISSED", html_out)

    def test_site_names_are_escaped(self):
        html_out = render_dashboard([
            {"site_id": "<script>alert(1)</script>", "state": "ok", "age_s": 1.0,
             "reasons": [], "cameras": [], "known": True}])
        self.assertNotIn("<script>alert", html_out)


if __name__ == "__main__":
    unittest.main()
