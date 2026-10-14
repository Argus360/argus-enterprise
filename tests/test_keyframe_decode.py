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
        self.assertIn('else compute_device(self.device) == "cpu"', src)
        self.assertIn("keyframes_only_min_width=(self.keyframe_min_width if keyframes else 0)", src)
        self.assertIn('keyframe_decode=(bool(site["keyframe_decode"])', inspect.getsource(pipeline.run_site))


if __name__ == "__main__":
    unittest.main()


class SourceWidthTests(unittest.TestCase):
    """The three ceiling fisheyes were judged by their 1600 px panorama and
    never switched to keyframe decode (14 Oct 2026)."""

    def test_a_dewarped_big_stream_still_switches_to_keyframes(self):
        from cvti.serving import streams as st
        from cvti.serving.dewarp import Dewarp
        opens = []
        def fake_open(source, keyframes_only=False, **_kw):
            cap = _Cap(2592, keyframes_only=keyframes_only); opens.append(cap); return cap
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             keyframes_only_min_width=2000,
                             dewarp=Dewarp.from_config({"mode": "panorama", "output": [1600, 500]}))
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
            d.start()
            deadline = time.time() + 3.0
            while time.time() < deadline and len(opens) < 2:
                time.sleep(0.02)
            d.stop()
        self.assertEqual((d.source_width, d.stream_width), (2592, 1600))
        self.assertGreaterEqual(len(opens), 2)
        self.assertTrue(opens[1].keyframes_only)
        self.assertEqual(d.ingest_status()["source_width"], 2592)


class DetectorThreadsTests(unittest.TestCase):
    def test_a_cpu_box_keeps_one_core_free_and_the_site_can_override(self):
        from cvti.detector import accel
        with mock.patch("os.cpu_count", return_value=5):
            self.assertEqual(accel.detector_threads_for({}, "cpu"), 4)
            self.assertEqual(accel.detector_threads_for({"detector_threads": 2}, "cpu"), 2)
            self.assertEqual(accel.detector_threads_for({"detector_threads": True}, "cpu"), 4)
            self.assertEqual(accel.detector_threads_for({}, "cuda"), 0)
        with mock.patch("os.cpu_count", return_value=2):
            self.assertEqual(accel.detector_threads_for({}, "cpu"), 2)

    def test_the_cpu_session_is_rebuilt_with_the_cap_when_no_better_provider_exists(self):
        from cvti.detector import accel
        inst = mock.Mock()
        inst.session.get_providers.return_value = ["CPUExecutionProvider"]
        fake_ort = mock.Mock()
        fake_ort.InferenceSession.return_value.get_outputs.return_value = []
        with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}), \
             mock.patch.object(accel, "best_available_provider", return_value=None), \
             mock.patch.object(accel, "DETECTOR_THREADS", 4):
            self.assertTrue(accel._ensure_preferred_provider(inst, "m.onnx"))
        self.assertEqual(fake_ort.SessionOptions.return_value.intra_op_num_threads, 4)
        args, kwargs = fake_ort.InferenceSession.call_args
        self.assertEqual(kwargs.get("providers"), ["CPUExecutionProvider"])

    def test_no_cap_and_no_provider_leaves_the_session_alone(self):
        from cvti.detector import accel
        inst = mock.Mock()
        inst.session.get_providers.return_value = ["CPUExecutionProvider"]
        with mock.patch.object(accel, "best_available_provider", return_value=None), \
             mock.patch.object(accel, "DETECTOR_THREADS", 0):
            self.assertFalse(accel._ensure_preferred_provider(inst, "m.onnx"))

    def test_the_pipeline_sets_the_cap_before_loading(self):
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.MultiStreamPipeline.start)
        self.assertIn("accel.DETECTOR_THREADS = ", src)
        self.assertIn('detector_threads=(int(site["detector_threads"])', inspect.getsource(pipeline.run_site))


class ComputeDeviceTests(unittest.TestCase):
    """The installed Windows build ships a CPU torch and detects on the GPU
    through DirectML: the torch device alone called an RTX box CPU-only."""

    def test_a_directml_box_is_not_cpu_only(self):
        from cvti.serving import pipeline
        with mock.patch("cvti.detector.accel.best_available_provider", return_value="DmlExecutionProvider"):
            self.assertEqual(pipeline.compute_device("cpu"), "dml")
            self.assertEqual(pipeline.wall_fps_for({}, 24.0, 12, pipeline.compute_device("cpu")), 24.0)
        with mock.patch("cvti.detector.accel.best_available_provider", return_value=None):
            self.assertEqual(pipeline.compute_device("cpu"), "cpu")
        self.assertEqual(pipeline.compute_device("cuda"), "cuda")

    def test_the_three_cpu_only_decisions_use_it(self):
        from cvti.serving import pipeline
        start = inspect.getsource(pipeline.MultiStreamPipeline.start)
        self.assertIn('compute_device(self.device) == "cpu"', start)
        self.assertIn("detector_threads_for(None, compute_device(self.device))", start)
        self.assertIn("wall_fps_for(site, publish_fps, len(sources), compute_device(device))",
                      inspect.getsource(pipeline.run_site))
