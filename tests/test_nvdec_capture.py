"""GPU (NVDEC) decode for live streams (cvti/serving/capture.py)."""
from __future__ import annotations

import unittest
from unittest import mock

from cvti.serving import capture


class NvdecDetection(unittest.TestCase):
    def setUp(self):
        capture._NVDEC_BINARY = False
        capture._NVDEC_FAILED_SOURCES.clear()

    def tearDown(self):
        capture._NVDEC_BINARY = False
        capture._NVDEC_FAILED_SOURCES.clear()

    def test_can_be_disabled(self):
        with mock.patch.dict("os.environ", {"ARGUS_NVDEC": "0"}):
            self.assertIsNone(capture.nvdec_ffmpeg())

    def test_needs_an_nvidia_gpu(self):
        with mock.patch.dict("os.environ", {"ARGUS_NVDEC": "1"}), \
             mock.patch("shutil.which", side_effect=lambda n: None if n == "nvidia-smi" else "/x"):
            self.assertIsNone(capture.nvdec_ffmpeg())

    def test_found_when_ffmpeg_has_cuvid(self):
        out = mock.Mock(stdout=" V..... h264_cuvid\n V..... hevc_cuvid\n")
        with mock.patch.dict("os.environ", {"ARGUS_NVDEC": "1"}), \
             mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             mock.patch("subprocess.run", return_value=out), \
             mock.patch("ctypes.CDLL"):
            self.assertEqual(capture.nvdec_ffmpeg(), "/usr/bin/ffmpeg")

    def test_skipped_without_the_decode_library(self):
        out = mock.Mock(stdout=" V..... h264_cuvid\n V..... hevc_cuvid\n")
        with mock.patch.dict("os.environ", {"ARGUS_NVDEC": "1"}), \
             mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             mock.patch("subprocess.run", return_value=out), \
             mock.patch("ctypes.CDLL", side_effect=OSError("libnvcuvid.so.1")):
            self.assertIsNone(capture.nvdec_ffmpeg())


class NvdecCaptureRefusals(unittest.TestCase):
    def test_unsupported_codec_is_not_opened_but_reachable(self):
        with mock.patch.object(capture, "_probe_stream",
                               return_value={"codec": "mjpeg", "width": 1280, "height": 720, "fps": 15}):
            cap = capture.FFmpegNvdecCapture("rtsp://cam/x", binary="/usr/bin/ffmpeg")
        self.assertFalse(cap.isOpened())
        self.assertTrue(cap.probe_ok)

    def test_unreachable_camera_is_not_remembered(self):
        capture._NVDEC_FAILED_SOURCES.clear()
        with mock.patch.object(capture, "nvdec_ffmpeg", return_value="/usr/bin/ffmpeg"), \
             mock.patch.object(capture, "_probe_stream", return_value=None), \
             mock.patch("cv2.VideoCapture") as vc:
            vc.return_value.isOpened.return_value = False
            capture.open_capture("rtsp://cam/x")
        self.assertNotIn("rtsp://cam/x", capture._NVDEC_FAILED_SOURCES)

    def test_reachable_but_refused_is_remembered(self):
        capture._NVDEC_FAILED_SOURCES.clear()
        with mock.patch.object(capture, "nvdec_ffmpeg", return_value="/usr/bin/ffmpeg"), \
             mock.patch.object(capture, "_probe_stream",
                               return_value={"codec": "mjpeg", "width": 640, "height": 480, "fps": 10}), \
             mock.patch("cv2.VideoCapture") as vc:
            vc.return_value.isOpened.return_value = True
            capture.open_capture("rtsp://cam/y")
        self.assertIn("rtsp://cam/y", capture._NVDEC_FAILED_SOURCES)
        capture._NVDEC_FAILED_SOURCES.clear()

    def test_files_and_keyframe_mode_never_use_nvdec(self):
        with mock.patch.object(capture, "nvdec_ffmpeg") as nv, \
             mock.patch("cv2.VideoCapture"):
            capture.open_capture("clip.mp4")
            capture.open_capture("rtsp://cam/z", low_latency=False)
        nv.assert_not_called()


class DecodersFollowTheAdaptiveRate(unittest.TestCase):
    def _pipe(self, targets):
        from cvti.serving.pipeline import MultiStreamPipeline
        p = MultiStreamPipeline.__new__(MultiStreamPipeline)
        p.target_fps = 8.0
        p._cam_target_fps = targets
        p._decoders = {cid: mock.Mock(target_fps=8.0, _fps=0.0) for cid in targets}
        return p

    def test_decoder_rises_to_the_detection_target(self):
        p = self._pipe({"a": 16.0})
        p._sync_decoder_rates()
        self.assertAlmostEqual(p._decoders["a"].target_fps, 16.0 * 1.25)

    def test_never_above_the_cameras_own_rate(self):
        p = self._pipe({"a": 120.0})
        p._decoders["a"]._fps = 30.0
        p._sync_decoder_rates()
        self.assertEqual(p._decoders["a"].target_fps, 30.0)

    def test_never_below_the_fixed_target(self):
        p = self._pipe({"a": 2.0})
        p._sync_decoder_rates()
        self.assertEqual(p._decoders["a"].target_fps, 8.0)


if __name__ == "__main__":
    unittest.main()


class ConcurrentStartup(unittest.TestCase):
    def test_every_camera_sees_the_gpu_when_opened_at_once(self):
        import threading
        import time as _t
        capture._NVDEC_BINARY = False
        out = mock.Mock(stdout=" V..... h264_cuvid\n V..... hevc_cuvid\n")

        def slow_run(*a, **k):
            _t.sleep(0.2)               # the probe takes time; others arrive meanwhile
            return out
        results = []
        with mock.patch.dict("os.environ", {"ARGUS_NVDEC": "1"}), \
             mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             mock.patch("subprocess.run", side_effect=slow_run), \
             mock.patch("ctypes.CDLL"):
            threads = [threading.Thread(target=lambda: results.append(capture.nvdec_ffmpeg()))
                       for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        capture._NVDEC_BINARY = False
        self.assertEqual(results, ["/usr/bin/ffmpeg"] * 4)


class HalfPrecisionOnIndexedCudaDevice(unittest.TestCase):
    def test_cuda_index_keeps_half(self):
        from cvti.serving.pipeline import MultiStreamPipeline
        with mock.patch("cvti.serving.pipeline._auto_device", return_value="cuda:0"):
            p = MultiStreamPipeline({"a": "rtsp://x"}, device="cuda:0", half=True)
        self.assertTrue(p.half)

    def test_cpu_never_half(self):
        from cvti.serving.pipeline import MultiStreamPipeline
        p = MultiStreamPipeline({"a": "rtsp://x"}, device="cpu", half=True)
        self.assertFalse(p.half)
