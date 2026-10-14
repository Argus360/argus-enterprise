"""The three engine changes from Demi's read of the 13 Oct CHI bundle:
pose/weapon off the frame loop, unwatched view-only cameras decode nothing,
and a failed hardware-decode open is remembered."""
from __future__ import annotations

import inspect
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _person():
    import supervision as sv
    return sv.Detections(xyxy=np.array([[10.0, 10.0, 50.0, 90.0]]),
                         class_id=np.array([0]), confidence=np.array([0.9]))


class HeavyModelsOffTheLoopTests(unittest.TestCase):
    def _state(self, runner):
        from cvti.rules.customization import CustomizationEngine
        from cvti.serving.camera import PerCameraState
        eng = CustomizationEngine("configs/all_threats_v1.json",
                                  baseline_path="configs/baseline_critical_v1.json")
        return PerCameraState("cam1", eng, person_filter=False, pose_model=object(),
                              concealment=True, heavy_runner=runner)

    def test_a_slow_pose_pass_never_stalls_the_frame_loop(self):
        """50-72 s pose calls ran inline on the pilot server and stalled every
        camera. With the worker, process() returns at once and the pass
        completes in the background."""
        from cvti.serving.camera import PerCameraState
        from cvti.video_action_runtime import AsyncVideoActionRunner
        runner = AsyncVideoActionRunner(name="test-heavy").start()
        self.addCleanup(runner.stop)
        state = self._state(runner)
        started = threading.Event()

        def slow_pose(self_, image, timestamp):
            started.set()
            time.sleep(0.4)
            return []
        with mock.patch.object(PerCameraState, "_compute_pose", slow_pose):
            t = time.perf_counter()
            state.process(_person(), np.zeros((120, 160, 3), np.uint8), timestamp=100.0)
            elapsed = time.perf_counter() - t
            self.assertLess(elapsed, 0.15, f"process() waited for the pose pass ({elapsed:.2f}s)")
            self.assertTrue(started.wait(1.0), "the worker never ran the pass")
            deadline = time.time() + 3.0
            while time.time() < deadline and runner.completed < 1:
                time.sleep(0.02)
        self.assertEqual(runner.completed, 1)
        self.assertEqual(runner.failed, 0)

    def test_the_light_path_steps_aside_while_the_worker_holds_the_camera(self):
        state = self._state(runner=mock.Mock())
        state.violence = True
        state._heavy_lock.acquire()
        try:
            with mock.patch.object(type(state), "_heavy_pass") as heavy:
                out = state._heavy_submit_or_light(False, np.zeros((8, 8, 3), np.uint8), 1.0, [], [])
            self.assertEqual(out, [])
            heavy.assert_not_called()
        finally:
            state._heavy_lock.release()

    def test_without_a_runner_the_pass_is_synchronous_as_before(self):
        from cvti.serving.camera import PerCameraState
        state = self._state(runner=None)
        with mock.patch.object(PerCameraState, "_compute_pose", return_value=[]) as pose:
            state.process(_person(), np.zeros((120, 160, 3), np.uint8), timestamp=100.0)
        self.assertEqual(pose.call_count, 1)

    def test_run_site_wires_one_heavy_worker_per_site(self):
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.run_site)
        self.assertIn('AsyncVideoActionRunner(name="heavy-models")', src)
        self.assertIn("heavy_runner=heavy_runner", src)


class _Cap:
    def __init__(self):
        self.released = False
        self.n = 0

    def get(self, *_a): return 10.0
    def grab(self): return True
    def set(self, *_a): return True
    def isOpened(self): return True  # noqa: N802
    def release(self): self.released = True

    def read(self):
        self.n += 1
        return True, np.zeros((8, 8, 3), np.uint8)


class IdleViewOnlyDecoderTests(unittest.TestCase):
    def _decoder(self, alive=True):
        from cvti.serving import streams as st
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             view_only=True, idle_when_unwatched=True,
                             probe_source="rtsp://cam7/main", offline_grace_seconds=0.1,
                             reconnect_backoff=0.01)
        d.idle_probe_seconds = 0.05
        return d

    def test_unwatched_means_no_capture_and_a_probe_keeps_the_link_state(self):
        from cvti.serving import streams as st
        d = self._decoder()
        caps = []
        def fake_open(source, **_kw):
            caps.append(_Cap()); return caps[-1]
        with mock.patch("cvti.serving.capture.open_capture", side_effect=fake_open), \
             mock.patch.object(st.StreamDecoder, "_probe_alive", return_value=True):
            d.start()
            time.sleep(0.4)
            self.assertEqual(caps, [], "an unwatched view-only camera must not open its stream")
            self.assertTrue(d.link_status()["idle"])
            self.assertEqual(d.state, st.CONNECTED)
            self.assertGreaterEqual(d.idle_probes, 2)
            # a viewer arrives: the stream opens and frames flow
            d.display_fps = 8.0
            deadline = time.time() + 3.0
            while time.time() < deadline and (not caps or d.peek_latest()[0] is None):
                time.sleep(0.02)
            self.assertEqual(len(caps), 1)
            self.assertIsNotNone(d.peek_latest()[0])
            self.assertFalse(d.link_status()["idle"])
            # the viewer leaves: the stream is released and the decoder idles again
            d.display_fps = 0.0
            deadline = time.time() + 3.0
            while time.time() < deadline and not d.idle:
                time.sleep(0.02)
            self.assertTrue(d.idle)
            self.assertTrue(caps[0].released)
            d.stop()

    def test_an_unreachable_camera_goes_reconnecting_then_offline_while_idle(self):
        from cvti.serving import streams as st
        d = self._decoder()
        with mock.patch("cvti.serving.capture.open_capture", side_effect=AssertionError("no open")), \
             mock.patch.object(st.StreamDecoder, "_probe_alive", return_value=False):
            d.start()
            deadline = time.time() + 3.0
            while time.time() < deadline and d.state != st.OFFLINE:
                time.sleep(0.02)
            d.stop()
        self.assertEqual(d.state, st.OFFLINE)

    def test_detection_cameras_are_never_idled(self):
        from cvti.serving import streams as st
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", view_only=False,
                             idle_when_unwatched=True)
        self.assertTrue(d._wants_idle(), "the flag alone decides; the pipeline passes it for view-only")
        d2 = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", view_only=False)
        self.assertFalse(d2._wants_idle())

    def test_the_pipeline_idles_view_only_decoders_and_probes_the_camera_itself(self):
        from cvti.serving import pipeline
        src = inspect.getsource(pipeline.MultiStreamPipeline.start)
        self.assertIn("idle_when_unwatched=vo", src)
        self.assertIn("probe_source=self.fallback_sources.get(cam_id, src)", src)


class RtspAliveTests(unittest.TestCase):
    def test_any_rtsp_reply_means_alive_and_a_refused_port_means_not(self):
        from cvti.serving.capture import rtsp_alive
        server = socket.socket()
        server.bind(("127.0.0.1", 0)); server.listen(1)
        port = server.getsockname()[1]

        def serve():
            conn, _ = server.accept()
            conn.recv(512)
            conn.sendall(b"RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\n\r\n")
            conn.close()
        threading.Thread(target=serve, daemon=True).start()
        self.assertTrue(rtsp_alive(f"rtsp://user:pw@127.0.0.1:{port}/1", timeout=2.0))
        server.close()
        dead = socket.socket(); dead.bind(("127.0.0.1", 0)); dead_port = dead.getsockname()[1]; dead.close()
        self.assertFalse(rtsp_alive(f"rtsp://127.0.0.1:{dead_port}/1", timeout=1.0))
        self.assertFalse(rtsp_alive("not a url"))


class HardwareDecodeMemoTests(unittest.TestCase):
    def test_a_failed_hardware_open_is_not_retried_on_the_next_open(self):
        from cvti.serving import capture
        capture._HW_DECODE_FAILED = False
        self.addCleanup(setattr, capture, "_HW_DECODE_FAILED", False)
        constructed = []

        class FakeCap:
            def __init__(self, src, backend=None, params=None):
                constructed.append(list(params or []))
                self._ok = not (params and 999 in params)
            def isOpened(self): return self._ok  # noqa: N802
            def release(self): pass
            def set(self, *_a): return True
        fake_cv2 = mock.Mock()
        fake_cv2.VideoCapture = FakeCap
        fake_cv2.CAP_FFMPEG = 1900
        fake_cv2.CAP_PROP_BUFFERSIZE = 38
        with mock.patch.dict(sys.modules, {"cv2": fake_cv2}), \
             mock.patch.object(capture, "_hw_decode_params", return_value=[999, 1]), \
             mock.patch.object(capture, "_live_decode_params", return_value=[]):
            capture.open_capture("rtsp://cam/1")
            first = len(constructed)
            capture.open_capture("rtsp://cam/1")
        self.assertEqual(first, 2, "first open: hardware attempt then software")
        self.assertEqual(len(constructed) - first, 1, "second open: software only")
        self.assertNotIn(999, constructed[-1])
        self.assertTrue(capture._HW_DECODE_FAILED)


if __name__ == "__main__":
    unittest.main()
