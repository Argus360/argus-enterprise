"""Big live streams on CPU-only boxes decode keyframes only (13 Oct 2026):
seven 5-megapixel HEVC cameras decoded in full ate the pilot server."""
from __future__ import annotations

import inspect
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLIP = Path("data/test_clips/theft_shop_01.mp4")


class FFmpegKeyframeCaptureTests(unittest.TestCase):
    def setUp(self):
        from cvti.serving import capture
        self.binary = capture.ffmpeg_exe()
        if not self.binary or not CLIP.exists():
            self.skipTest("ffmpeg or the demo clip is not available")

    def test_only_keyframes_come_out_with_the_right_shape(self):
        from cvti.serving.capture import FFmpegKeyframeCapture
        cap = FFmpegKeyframeCapture(str(CLIP), binary=self.binary)
        try:
            self.assertTrue(cap.isOpened())
            self.assertEqual((cap.get(3), cap.get(4), cap.get(5)), (360.0, 640.0, 0.0))
            frames = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                self.assertEqual(frame.shape, (640, 360, 3))
            # the clip has 931 frames and 6 keyframes
            self.assertGreaterEqual(frames, 3)
            self.assertLess(frames, 40)
        finally:
            cap.release()
        self.assertFalse(cap.isOpened())
        cap.release()                          # idempotent

    def test_open_capture_hands_back_the_keyframe_reader_when_asked(self):
        from cvti.serving import capture
        cap = capture.open_capture(str(CLIP), keyframes_only=True)
        try:
            self.assertTrue(getattr(cap, "keyframes_only", False))
        finally:
            cap.release()
        with mock.patch.object(capture, "ffmpeg_exe", return_value=None):
            cap = capture.open_capture(str(CLIP), keyframes_only=True)
            try:
                self.assertFalse(getattr(cap, "keyframes_only", False), "no ffmpeg: full decode")
                self.assertTrue(cap.isOpened())
            finally:
                cap.release()

    def test_a_source_that_is_not_video_closes_itself(self):
        from cvti.serving.capture import FFmpegKeyframeCapture
        cap = FFmpegKeyframeCapture(str(Path("README.md")), binary=self.binary, open_timeout_s=10)
        self.assertFalse(cap.isOpened())


class _Cap:
    def __init__(self, width, keyframes_only=False):
        self.width = width
        self.keyframes_only = keyframes_only
        self.released = False
        self.interrupted = False

    def get(self, *_a): return 10.0
    def grab(self): return True
    def set(self, *_a): return True
    def isOpened(self): return True  # noqa: N802
    def release(self): self.released = True
    def interrupt(self): self.interrupted = True

    def read(self):
        return True, np.zeros((8, self.width, 3), np.uint8)


class DecoderPolicyTests(unittest.TestCase):
    def _run(self, width, min_width):
        from cvti.serving import streams as st
        opens = []
        def fake_open(source, keyframes_only=False, **_kw):
            cap = _Cap(width, keyframes_only=keyframes_only)
            opens.append(cap)
            return cap
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             keyframes_only_min_width=min_width)
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
            d.start()
            deadline = time.time() + 3.0
            while time.time() < deadline and len(opens) < 2 and d.peek_latest()[0] is None:
                time.sleep(0.02)
            time.sleep(0.2)
            d.stop()
        return d, opens

    def test_a_big_stream_is_reopened_keyframes_only_once_its_width_is_known(self):
        d, opens = self._run(width=2592, min_width=2000)
        self.assertGreaterEqual(len(opens), 2)
        self.assertFalse(opens[0].keyframes_only, "the first open learns the width in full decode")
        self.assertTrue(opens[1].keyframes_only)
        self.assertTrue(opens[0].released)
        self.assertEqual(d.link_status()["decode"], "keyframes")
        self.assertTrue(opens[-1].interrupted, "stop() unblocks the keyframe reader")

    def test_a_small_stream_keeps_full_decode(self):
        d, opens = self._run(width=1280, min_width=2000)
        self.assertEqual(len(opens), 1)
        self.assertFalse(opens[0].keyframes_only)
        self.assertEqual(d.link_status()["decode"], "full")

    def test_disabled_means_never(self):
        d, opens = self._run(width=2592, min_width=0)
        self.assertEqual(len(opens), 1)
        self.assertFalse(opens[0].keyframes_only)

    def test_a_refusal_is_not_retried(self):
        from cvti.serving import streams as st
        opens = []
        def fake_open(source, keyframes_only=False, **_kw):
            opens.append(keyframes_only)
            return _Cap(2592, keyframes_only=False)       # ffmpeg missing: full decode back
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             keyframes_only_min_width=2000)
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
            d.start(); time.sleep(0.6); d.stop()
        self.assertEqual(opens[:2], [False, True])
        self.assertEqual(len(opens), 2, "after one refusal the decoder stops asking")

    def test_the_pipeline_turns_it_on_for_cpu_boxes_and_the_site_can_override(self):
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.MultiStreamPipeline.start)
        self.assertIn('else self.device == "cpu"', src)
        self.assertIn("keyframes_only_min_width=(self.keyframe_min_width if keyframes else 0)", src)
        self.assertIn('keyframe_decode=(bool(site["keyframe_decode"])', inspect.getsource(pipeline.run_site))


if __name__ == "__main__":
    unittest.main()
