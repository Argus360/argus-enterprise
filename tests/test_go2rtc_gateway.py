"""The go2rtc gateway is an upgrade, never a requirement (W1).

One camera session, every consumer — but the pilot lesson behind every pin
here is the fallback: a missing binary, a refused launch, or a dead process
must leave every camera on the direct decode path with the reason in health,
never a black wall. And every listener stays on 127.0.0.1: go2rtc's API has
no auth, and the house rule is no unauthenticated route to a frame.

No real go2rtc process anywhere in this file — launch is a mocked Popen, so
CI needs no binary. The one test that talks to a live gateway is the manual
smoke at the bottom, skipped unless ARGUS_GO2RTC_SMOKE=1.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvti.serving import go2rtc as g2r  # noqa: E402
from cvti.serving.go2rtc import Go2rtcGateway, _sanitize, restreamable  # noqa: E402

CAMS = [
    {"id": "Main Corridor", "source": "rtsp://user:pw@10.0.0.7/stream1"},
    {"id": "Loading Bay", "source": "data/test_clips/empty_warehouse.mp4"},
    {"id": "Front Door", "source": "rtsp://user:pw@10.0.0.8/stream1"},
    {"id": "Reception", "source": 0},
    {"id": "Public Feed", "source": "https://cdn.example/live.m3u8"},
]


class StreamSelectionTests(unittest.TestCase):
    def test_only_network_cameras_are_restreamed(self):
        """Files and webcams are local: no session cap, no transport quirks —
        and looping a demo clip is the decoder's job, not go2rtc's."""
        gw = Go2rtcGateway(CAMS, tempfile.mkdtemp())
        self.assertEqual(set(gw.streams), {"Main Corridor", "Front Door"})
        self.assertTrue(restreamable("rtsp://cam/1"))
        self.assertTrue(restreamable("RTSPS://cam/1"))
        self.assertFalse(restreamable("clip.mp4"))
        self.assertFalse(restreamable(0))
        self.assertFalse(restreamable("0"))

    def test_http_sources_decode_direct_and_are_never_transcoded(self):
        """go2rtc cannot pass HLS/MJPEG through: it spawned ffmpeg and
        transcoded to H.264 for the engine's restream, 44% CPU for one
        public feed on a laptop (13 Oct 2026). The engine decodes those."""
        self.assertFalse(restreamable("https://cdn.example/live.m3u8"))
        self.assertFalse(restreamable("http://10.0.0.9/mjpeg"))
        gw = Go2rtcGateway(CAMS, tempfile.mkdtemp())
        self.assertNotIn("Public Feed", gw.streams)

    def test_stream_names_are_url_safe(self):
        """Camera ids are display names ('Main Corridor') — hostile inside an
        rtsp:// path. Names are sanitized deterministically."""
        gw = Go2rtcGateway(CAMS, tempfile.mkdtemp())
        name, src = gw.streams["Main Corridor"]
        self.assertEqual(name, "main_corridor")
        self.assertEqual(src, "rtsp://user:pw@10.0.0.7/stream1")

    def test_colliding_names_never_share_a_stream(self):
        taken: set = set()
        self.assertEqual(_sanitize("Door #1", taken), "door_1")
        self.assertEqual(_sanitize("Door 1", taken), "door_1_2")
        self.assertEqual(_sanitize("door_1", taken), "door_1_3")

    def test_a_hostile_id_still_yields_a_name(self):
        self.assertEqual(_sanitize("///", set()), "cam")


class ConfigTests(unittest.TestCase):
    def test_every_listener_binds_loopback_only(self):
        """go2rtc's API is unauthenticated upstream. The house rule is no
        unauthenticated route to a frame — so nothing listens off-box."""
        gw = Go2rtcGateway(CAMS, tempfile.mkdtemp())
        gw.api_port, gw.rtsp_port, gw.webrtc_port = 1984, 8554, 8555
        cfg = gw.build_config()
        for section in ("api", "rtsp", "webrtc"):
            self.assertTrue(cfg[section]["listen"].startswith("127.0.0.1:"),
                            f"{section} must bind loopback, got {cfg[section]}")

    def test_the_config_carries_exactly_the_network_streams(self):
        gw = Go2rtcGateway(CAMS, tempfile.mkdtemp())
        gw.api_port = gw.rtsp_port = gw.webrtc_port = 1
        streams = gw.build_config()["streams"]
        self.assertEqual(streams, {
            "main_corridor": "rtsp://user:pw@10.0.0.7/stream1",
            "front_door": "rtsp://user:pw@10.0.0.8/stream1",
        })

    def test_write_config_is_valid_yaml_on_disk(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            gw = Go2rtcGateway(CAMS, tmp)
            gw.api_port = gw.rtsp_port = gw.webrtc_port = 1
            path = gw.write_config()
            self.assertEqual(path.name, "go2rtc.yaml")
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertIn("main_corridor", doc["streams"])


class BinaryResolutionTests(unittest.TestCase):
    def test_bundled_binary_wins_over_path(self):
        with mock.patch.object(g2r, "bundled_binary", return_value="/bundle/go2rtc"), \
             mock.patch("shutil.which", return_value="/usr/local/bin/go2rtc"):
            self.assertEqual(g2r.go2rtc_binary(), "/bundle/go2rtc")

    def test_path_binary_is_the_dev_fallback(self):
        with mock.patch.object(g2r, "bundled_binary", return_value=None), \
             mock.patch("shutil.which", return_value="/usr/local/bin/go2rtc"):
            self.assertEqual(g2r.go2rtc_binary(), "/usr/local/bin/go2rtc")

    def test_no_binary_anywhere_means_none(self):
        with mock.patch.object(g2r, "bundled_binary", return_value=None), \
             mock.patch("shutil.which", return_value=None):
            self.assertIsNone(g2r.go2rtc_binary())


class FallbackTests(unittest.TestCase):
    """The pilot lesson: a gateway that cannot run must leave the cameras on
    the direct path and say WHY — never a black wall, never an exception."""

    def test_no_binary_disables_with_the_reason_in_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            gw = Go2rtcGateway(CAMS, tmp)
            with mock.patch.object(g2r, "go2rtc_binary", return_value=None):
                self.assertFalse(gw.start())
        self.assertIn("not found", gw.disabled_reason)
        self.assertFalse(gw.status()["running"])
        self.assertEqual(gw.status()["disabled_reason"], gw.disabled_reason)

    def test_no_network_cameras_disables_without_launching(self):
        with tempfile.TemporaryDirectory() as tmp:
            gw = Go2rtcGateway([{"id": "clip", "source": "a.mp4"}], tmp)
            with mock.patch.object(g2r, "subprocess") as sub:
                self.assertFalse(gw.start())
                sub.Popen.assert_not_called()
        self.assertIn("no network cameras", gw.disabled_reason)

    def test_a_launch_that_dies_before_the_api_answers_is_disabled(self):
        dead = mock.MagicMock()
        dead.poll.return_value = 1                     # exited immediately
        with tempfile.TemporaryDirectory() as tmp:
            # A real file, not a patched os.stat: g2r.os IS the global os
            # module, and patching stat there breaks pathlib's mkdir on 3.11
            # — which is exactly how this test failed on CI while passing on
            # a 3.9 dev box (8 Sep). Mock at the seam, never the stdlib.
            fake_bin = Path(tmp) / "go2rtc"
            fake_bin.write_text("#!/bin/sh\n")
            gw = Go2rtcGateway(CAMS, tmp)
            with mock.patch.object(g2r, "go2rtc_binary",
                                   return_value=str(fake_bin)), \
                 mock.patch.object(g2r.subprocess, "Popen", return_value=dead):
                self.assertFalse(gw.start(wait_ready_s=0.5))
        self.assertIn("not ready", gw.disabled_reason)
        self.assertFalse(gw.alive())

    def test_restream_url_is_none_when_the_gateway_is_down(self):
        """None is the fallback contract: the decoder keeps the camera's own
        URL. An exception here would take the camera down with the gateway."""
        gw = Go2rtcGateway(CAMS, tempfile.mkdtemp())
        self.assertIsNone(gw.restream_url("Main Corridor"))
        self.assertIsNone(gw.restream_url("no-such-camera"))

    def test_stop_is_idempotent_and_safe_before_start(self):
        gw = Go2rtcGateway(CAMS, tempfile.mkdtemp())
        gw.stop()
        gw.stop()


class RestreamUrlTests(unittest.TestCase):
    def _live_gateway(self):
        gw = Go2rtcGateway(CAMS, tempfile.mkdtemp())
        gw.rtsp_port = 8554
        proc = mock.MagicMock()
        proc.poll.return_value = None                  # running
        gw._proc = proc
        return gw

    def test_a_gateway_camera_gets_a_localhost_rtsp_url(self):
        gw = self._live_gateway()
        self.assertEqual(gw.restream_url("Main Corridor"),
                         "rtsp://127.0.0.1:8554/main_corridor")

    def test_a_local_camera_stays_on_its_own_source(self):
        gw = self._live_gateway()
        self.assertIsNone(gw.restream_url("Loading Bay"))   # a file — not gatewayed


@unittest.skipUnless(os.environ.get("ARGUS_GO2RTC_SMOKE") == "1",
                     "manual smoke: needs the real binary (ARGUS_GO2RTC_SMOKE=1)")
class LiveSmokeTest(unittest.TestCase):
    def test_the_real_binary_starts_answers_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            gw = Go2rtcGateway(CAMS, tmp)
            self.assertTrue(gw.start(), gw.disabled_reason)
            try:
                self.assertTrue(gw.alive())
                self.assertIsNotNone(gw.api_streams())
                self.assertIn("main_corridor", gw.api_streams())
            finally:
                gw.stop()
            self.assertFalse(gw.alive())


if __name__ == "__main__":
    unittest.main()
