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
    def __init__(self, width, keyframes_only=False, burn_ms=0.0, wait_ms=0.0):
        self.width = width
        self.keyframes_only = keyframes_only
        self.released = False
        self.interrupted = False
        self.burn_ms = burn_ms        # CPU spent per read, like software decode
        self.wait_ms = wait_ms        # wall time spent waiting, like a live camera

    def get(self, *_a): return 30.0
    def set(self, *_a): return True
    def isOpened(self): return True  # noqa: N802
    def release(self): self.released = True
    def interrupt(self): self.interrupted = True

    def _work(self):
        if self.wait_ms:
            time.sleep(self.wait_ms / 1000.0)
        if self.burn_ms:
            # Spin on the wall clock: a busy loop is CPU time on every
            # platform. Polling time.thread_time() here instead made a 1 ms
            # burn last 15 ms on Windows, whose thread clock ticks that
            # coarsely, and a cheap camera looked expensive in CI.
            t = time.perf_counter()
            while (time.perf_counter() - t) * 1000.0 < self.burn_ms:
                pass

    def grab(self):
        self._work()                   # OpenCV decodes a skipped frame too
        return True

    def read(self):
        self._work()
        return True, np.zeros((8, self.width, 3), np.uint8)


class DecoderPolicyTests(unittest.TestCase):
    def _run(self, width, min_width, burn_ms=20.0, wait_ms=0.0):
        from cvti.serving import streams as st
        opens = []
        def fake_open(source, keyframes_only=False, **_kw):
            cap = _Cap(width, keyframes_only=keyframes_only, burn_ms=burn_ms, wait_ms=wait_ms)
            opens.append(cap)
            return cap
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             keyframes_only_min_width=min_width, cost_window_s=0.15)
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
            d.start()
            deadline = time.time() + 4.0
            while time.time() < deadline and len(opens) < 2:
                time.sleep(0.02)
            time.sleep(0.2)
            d.stop()
        return d, opens

    def test_a_big_expensive_stream_is_reopened_keyframes_only_once_measured(self):
        d, opens = self._run(width=2592, min_width=2000, burn_ms=20.0)
        self.assertGreaterEqual(len(opens), 2)
        self.assertFalse(opens[0].keyframes_only, "the first open measures in full decode")
        self.assertGreaterEqual(d.decode_cpu_share, 0.25)
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
            return _Cap(2592, keyframes_only=False, burn_ms=20.0)   # ffmpeg missing: full decode back
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             keyframes_only_min_width=2000, cost_window_s=0.15)
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
            d.start(); time.sleep(1.2); d.stop()
        self.assertEqual(opens[:2], [False, True])
        self.assertEqual(len(opens), 2, "after one refusal the decoder stops asking")

    def test_the_pipeline_turns_it_on_for_cpu_boxes_and_the_site_can_override(self):
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.MultiStreamPipeline.start)
        self.assertIn('"auto" if self.keyframe_decode is None', src)
        self.assertIn('keyframes_only_min_width=(self.keyframe_min_width if policy != "off" else 0)', src)
        self.assertIn('keyframe_decode=(bool(site["keyframe_decode"])', inspect.getsource(pipeline.run_site))
        self.assertIn('keyframe_cost_share=float(site.get("keyframe_cost_share", 0.25) or 0.25)', inspect.getsource(pipeline.run_site))


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
            cap = _Cap(2592, keyframes_only=keyframes_only, burn_ms=20.0); opens.append(cap); return cap
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             keyframes_only_min_width=2000, cost_window_s=0.15,
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
        self.assertNotIn('compute_device(self.device) == "cpu"', start,
                         "keyframe decode is decided by measured cost, not the detector's device")
        self.assertIn("detector_threads_for(None, compute_device(self.device))", start)
        self.assertIn("wall_fps_for(site, publish_fps, len(sources), compute_device(device))",
                      inspect.getsource(pipeline.run_site))


class MeasuredCostTests(unittest.TestCase):
    """Decode cost is CPU time on the decode thread, not wall time (Tolu's
    audit, 15 Oct 2026): a live camera makes every read WAIT, and a hardware
    decoder makes a wide stream cheap. Neither is a reason for keyframes."""

    def test_waiting_for_a_live_camera_is_not_decode_cost(self):
        d, opens = DecoderPolicyTests()._run(width=2592, min_width=2000, burn_ms=0.0, wait_ms=15.0)
        self.assertEqual(len(opens), 1, "a cheap decode never switches, however wide")
        self.assertFalse(opens[0].keyframes_only)
        self.assertIsNotNone(d.decode_cpu_ms)
        self.assertLess(d.decode_cpu_share or 0.0, 0.25)
        self.assertEqual(d.link_status()["decode"], "full")
        self.assertIn("decode_cpu_ms", d.ingest_status())
        self.assertIn("frame_cpu_ms", d.ingest_status())

    def test_the_site_threshold_is_honoured(self):
        from cvti.serving import streams as st
        opens = []
        def fake_open(source, keyframes_only=False, **_kw):
            cap = _Cap(2592, keyframes_only=keyframes_only, burn_ms=20.0); opens.append(cap); return cap
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             keyframes_only_min_width=2000, keyframe_cost_share=5.0,
                             cost_window_s=0.15)
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
            d.start(); time.sleep(0.8); d.stop()
        self.assertEqual(len(opens), 1, "the measured share is under a five-core threshold")

    def test_the_pipeline_passes_the_threshold_from_the_site(self):
        from cvti.serving import pipeline
        self.assertIn('keyframe_cost_share=float(site.get("keyframe_cost_share", 0.25) or 0.25)',
                      inspect.getsource(pipeline.run_site))
        self.assertIn("keyframe_cost_share=self.keyframe_cost_share", inspect.getsource(pipeline.MultiStreamPipeline.start))


class SamplingIndependenceTests(unittest.TestCase):
    """Review of #244: timing a whole stride made a 4 ms camera look like
    32 ms when detection dropped from 30 to 4 fps. The verdict must not
    depend on our sampling rate; the fake capture burns CPU in grab() too."""

    def _share_at(self, target_fps, burn_ms):
        from cvti.serving import streams as st
        opens = []
        def fake_open(source, keyframes_only=False, **_kw):
            cap = _Cap(2592, keyframes_only=keyframes_only, burn_ms=burn_ms); opens.append(cap); return cap
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=target_fps,
                             keyframes_only_min_width=2000, cost_window_s=0.15)
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
            d.start(); time.sleep(1.2); d.stop()
        return d, opens

    def test_a_cheap_camera_stays_on_full_decode_at_any_sampling_rate(self):
        # 30 fps source at ~1 ms of CPU a frame: a few percent of a core,
        # whether we keep every frame or one in eight.
        for target in (30.0, 4.0):
            d, opens = self._share_at(target_fps=target, burn_ms=1.0)
            self.assertEqual(len(opens), 1, f"target {target}: switched on a cheap camera")
            self.assertLess(d.decode_cpu_share or 0.0, 0.25)
            self.assertLess(d.decode_cpu_ms, 3.0, "cost is per SOURCE frame, not per stride")

    def test_an_expensive_camera_switches_at_any_sampling_rate(self):
        for target in (30.0, 4.0):
            d, opens = self._share_at(target_fps=target, burn_ms=20.0)
            self.assertGreaterEqual(len(opens), 2, f"target {target}: an expensive camera must switch")


class PolicyTests(unittest.TestCase):
    def _run(self, policy, burn_ms):
        from cvti.serving import streams as st
        opens = []
        def fake_open(source, keyframes_only=False, **_kw):
            cap = _Cap(2592, keyframes_only=keyframes_only, burn_ms=burn_ms); opens.append(cap); return cap
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             keyframes_only_min_width=2000, keyframe_policy=policy, cost_window_s=0.15)
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open):
            d.start(); time.sleep(0.8); d.stop()
        return opens

    def test_force_switches_a_cheap_wide_stream_without_measuring(self):
        opens = self._run("force", burn_ms=0.0)
        self.assertGreaterEqual(len(opens), 2)
        self.assertTrue(opens[1].keyframes_only)

    def test_off_never_switches_an_expensive_stream(self):
        opens = self._run("off", burn_ms=20.0)
        self.assertEqual(len(opens), 1)

    def test_an_unknown_policy_is_rejected(self):
        from cvti.serving import streams as st
        with self.assertRaises(ValueError):
            st.StreamDecoder("cam", "rtsp://x/y", keyframe_policy="maybe")

    def test_the_pipeline_maps_the_site_setting_to_three_policies(self):
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.MultiStreamPipeline.start)
        self.assertIn('"force" if self.keyframe_decode else "off"', src)
        self.assertIn("keyframe_policy=policy", src)
