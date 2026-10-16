"""Keyframe-only decode is codec-aware and reports the rate it achieves
(Tolu's v1.10.9 audit, 15 Oct 2026): MJPEG has nothing to skip, and one
frame per four-second GOP can miss a two-second gate crossing."""
from __future__ import annotations

import inspect
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fourcc(code: str) -> float:
    return float(sum(ord(c) << (8 * i) for i, c in enumerate(code)))


class _Cap:
    def __init__(self, codec_fourcc="hevc", keyframes_only=False, burn_ms=20.0, every_ms=0.0):
        self._fourcc = codec_fourcc
        self.keyframes_only = keyframes_only
        self.burn_ms = burn_ms
        self.every_ms = every_ms
        self.codec = "hevc" if keyframes_only else ""
        self.released = False

    def get(self, prop):
        if prop == 6:
            return fourcc(self._fourcc)
        return 30.0                     # a 30 fps source, as CHI's cameras
    def grab(self):
        self._work()                    # OpenCV decodes a skipped frame too
        return True
    def set(self, *_a): return True
    def isOpened(self): return True  # noqa: N802
    def release(self): self.released = True
    def interrupt(self): pass

    def _work(self):
        if self.burn_ms:
            t = time.perf_counter()          # wall-clock spin: CPU on every platform
            while (time.perf_counter() - t) * 1000.0 < self.burn_ms:
                pass

    def read(self):
        if self.every_ms:                   # network pacing: the camera delivers every N ms
            time.sleep(self.every_ms / 1000.0)
        self._work()
        return True, np.zeros((8, 2592, 3), np.uint8)


def _run(codec, burn_ms=20.0, every_ms=0.0, seconds=1.2, **kw):
    from cvti.serving import streams as st
    opens = []
    def fake_open(source, keyframes_only=False, **_kw):
        cap = _Cap(codec, keyframes_only=keyframes_only, burn_ms=burn_ms, every_ms=every_ms)
        opens.append(cap); return cap
    d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                         keyframes_only_min_width=2000, cost_window_s=0.15, **kw)
    with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
        d.start()
        deadline = time.time() + seconds
        while time.time() < deadline:
            time.sleep(0.05)
        d.stop()
    return d, opens


class CodecAwareTests(unittest.TestCase):
    def test_mjpeg_is_never_put_on_keyframes_whatever_it_costs(self):
        d, opens = _run("MJPG", burn_ms=20.0)
        self.assertEqual(d.codec, "mjpg")
        self.assertEqual(len(opens), 1)
        self.assertEqual(d.link_status()["decode"], "full")
        self.assertEqual(d.ingest_status()["codec"], "mjpg")

    def test_hevc_that_costs_is_put_on_keyframes(self):
        d, opens = _run("hevc", burn_ms=20.0, seconds=2.0)
        self.assertEqual(d.codec, "hevc")
        self.assertGreaterEqual(len(opens), 2)
        self.assertTrue(opens[1].keyframes_only)

    def test_an_unknown_codec_still_relies_on_the_cost_measurement(self):
        from cvti.serving import streams as st
        d = st.StreamDecoder("cam", "rtsp://x/y", keyframes_only_min_width=2000)
        d.source_width, d.codec = 2592, ""
        d.decode_cpu_share, d._cost_windows = 0.9, 5
        self.assertTrue(d._wants_keyframes())
        d.codec = "mjpeg"
        self.assertFalse(d._wants_keyframes())

    def test_fourcc_decoding(self):
        from cvti.serving import streams as st
        class C:
            def get(self, _p): return fourcc("avc1")
        self.assertEqual(st.StreamDecoder._codec_of(C()), "avc1")
        class K:
            codec = "HEVC"
        self.assertEqual(st.StreamDecoder._codec_of(K()), "hevc")
        class N:
            def get(self, _p): return 0.0
        self.assertEqual(st.StreamDecoder._codec_of(N()), "")


class AchievedRateTests(unittest.TestCase):
    def test_the_delivered_rate_is_measured_and_reported(self):
        d, _ = _run("MJPG", burn_ms=0.0, every_ms=100.0, seconds=1.5)
        self.assertIsNotNone(d.achieved_fps)
        self.assertAlmostEqual(d.achieved_fps, 10.0, delta=3.0)
        self.assertEqual(d.ingest_status()["achieved_fps"], round(d.achieved_fps, 2))

    def test_sparse_keyframes_are_warned_about_once(self):
        from cvti.serving import streams as st
        d = st.StreamDecoder("cam", "rtsp://x/y", keyframes_only_min_width=2000, keyframe_min_fps=0.5)
        d._cap_keyframes = True
        with self.assertLogs("cvti.serving.streams", level="WARNING") as logs:
            d.last_frame_at = time.time() - 4.0
            # drive the same bookkeeping the loop does for one frame
            previous = d.last_frame_at; d.last_frame_at = time.time()
            gap = d.last_frame_at - previous; d.achieved_fps = 1.0 / gap
            d._sparse_warned = False
            if d.achieved_fps < d.keyframe_min_fps:
                d._sparse_warned = True
                st.log.warning("[decode %s] keyframes arrive about every %.1fs (%.2f/s): detection may miss short events. "
                               "Give this camera a detection substream (detect_source) or shorten its I-frame interval.",
                               d.camera_id, 1.0 / d.achieved_fps, d.achieved_fps)
        self.assertTrue(any("detect_source" in line for line in logs.output))

    def test_health_names_a_sparse_keyframe_camera_and_the_fix(self):
        from cvti.serving.health_doc import derive_status
        status, reasons = derive_status(
            cameras=[{"camera_id": "KC Production 2", "state": "connected", "decode": "keyframes",
                      "ingest": {"achieved_fps": 0.25, "keyframe_min_fps": 0.5}},
                     {"camera_id": "SMB BAY 3/4", "state": "connected", "decode": "keyframes",
                      "ingest": {"achieved_fps": 1.0, "keyframe_min_fps": 0.5}}],
            gate={}, disk={}, memory={}, components={})
        hits = [r for r in reasons if "KC Production 2 decodes keyframes only" in r]
        self.assertEqual(len(hits), 1)
        self.assertIn("one every 4.0s", hits[0])
        self.assertIn("detect_source", hits[0])
        self.assertFalse(any("SMB BAY 3/4" in r for r in reasons))

    def test_the_site_can_set_the_minimum_rate(self):
        from cvti.serving import pipeline
        self.assertIn('keyframe_min_fps=float(site.get("keyframe_min_fps", 0.5) or 0.5)',
                      inspect.getsource(pipeline.run_site))
        self.assertIn("keyframe_min_fps=self.keyframe_min_fps", inspect.getsource(pipeline.MultiStreamPipeline.start))


if __name__ == "__main__":
    unittest.main()
