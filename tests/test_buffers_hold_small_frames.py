"""The rolling buffers hold small frames (latency audit 1 Sep, D4).

Two per-camera buffers hoarded full-resolution pixels that nothing ever used
at full resolution:

- the video-action buffer fed a model that works from 224–256px crops, yet
  held raw decoded frames — ~250 MB per 1080p camera, resized away at
  inference. It now shrinks to short-side 256 on the way in.
- the replay clip buffer JPEG-encoded the full 1080p frame EVERY frame on the
  detection hot path, holding ~7 MB per camera for a replay that is context,
  not forensic evidence. It now downscales to <=640w first.

The gate's evidence frames and the saved evidence bundle are untouched — they
keep full resolution.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.video_action_runtime import VideoActionRuntime


class _CapturingModel:
    def __init__(self):
        self.seen_frames: list = []

    def predict_frames(self, frames, *, top_k=5):
        self.seen_frames = list(frames)
        return []


def _runtime(model=None) -> VideoActionRuntime:
    return VideoActionRuntime(model=model or _CapturingModel(), backend="videomae",
                              model_name="fake", fps=5.0, cooldown_seconds=0.0)


class VideoActionBufferTests(unittest.TestCase):
    def test_1080p_frames_shrink_to_short_side_256(self):
        rt = _runtime()
        rt.add_frame(np.zeros((1080, 1920, 3), dtype=np.uint8), frame_index=0)
        stored = rt._frames[0].frame
        h, w = stored.shape[:2]
        self.assertEqual(min(h, w), 256)
        self.assertAlmostEqual(w / h, 1920 / 1080, places=2)   # aspect preserved

    def test_portrait_frames_shrink_on_their_short_side(self):
        rt = _runtime()
        rt.add_frame(np.zeros((1920, 1080, 3), dtype=np.uint8), frame_index=0)
        h, w = rt._frames[0].frame.shape[:2]
        self.assertEqual(w, 256)

    def test_already_small_frames_pass_through(self):
        rt = _runtime()
        rt.add_frame(np.zeros((240, 320, 3), dtype=np.uint8), frame_index=0)
        self.assertEqual(rt._frames[0].frame.shape[:2], (240, 320))

    def test_the_model_receives_the_small_frames(self):
        model = _CapturingModel()
        rt = _runtime(model)
        for i in range(20):
            rt.add_frame(np.zeros((1080, 1920, 3), dtype=np.uint8), frame_index=i)
        clip = rt.prepare_analysis(center_frame_index=10, timestamp=100.0)
        self.assertIsNotNone(clip)
        clip()
        self.assertTrue(model.seen_frames)
        for frame in model.seen_frames:
            self.assertLessEqual(min(frame.shape[:2]), 256)


class ClipBufferTests(unittest.TestCase):
    def _decoded_hw(self, jpeg: bytes):
        import cv2
        img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img.shape[:2]

    def test_replay_frames_are_capped_at_640_wide(self):
        from cvti.serving.camera import encode_clip_frame
        jpeg = encode_clip_frame(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertIsNotNone(jpeg)
        h, w = self._decoded_hw(jpeg)
        self.assertEqual(w, 640)
        self.assertEqual(h, 360)                                # aspect preserved

    def test_small_sources_are_not_upscaled(self):
        from cvti.serving.camera import encode_clip_frame
        jpeg = encode_clip_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        h, w = self._decoded_hw(jpeg)
        self.assertEqual((h, w), (240, 320))

    def test_the_frame_loop_fills_the_buffer_through_the_capped_encoder(self):
        import inspect
        from cvti.serving import camera
        # The frame loop records through _record_clip_frame (which can skip
        # the encode when evidence is off); the encode itself stays capped.
        src = inspect.getsource(camera.PerCameraState.process)
        self.assertIn("self._record_clip_frame(image, timestamp)", src)
        rec = inspect.getsource(camera.PerCameraState._record_clip_frame)
        self.assertIn("encode_clip_frame", rec)
        # The raw full-res encode must not sneak back onto the hot path.
        self.assertNotIn('cv2.imencode(".jpg", image', src)
        self.assertNotIn('cv2.imencode(".jpg", image', rec)

    def test_gate_evidence_stays_full_resolution(self):
        """D4 shrinks REPLAY, never evidence: the frames the gate judges and
        the saved bundle come from _frame_buffer, which must keep the raw
        decoded frame object untouched."""
        import inspect
        from cvti.serving import camera
        src = inspect.getsource(camera.PerCameraState.process)
        self.assertIn("self._frame_buffer.append(image)", src)


if __name__ == "__main__":
    unittest.main()
