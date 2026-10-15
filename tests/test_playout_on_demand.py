"""The wall's JPEG playout is fed only while someone is watching (Tolu's
v1.10.9 audit, 15 Oct 2026): a closed or WebRTC tile used to cost a resize
and a JPEG encode per decoded frame."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Cap:
    def get(self, *_a): return 30.0
    def grab(self): return True
    def set(self, *_a): return True
    def isOpened(self): return True  # noqa: N802
    def release(self): pass
    def read(self): return True, np.zeros((48, 64, 3), np.uint8)


class PlayoutOnDemandTests(unittest.TestCase):
    def _run(self, display_fps_after: float | None):
        from cvti.serving import streams as st
        d = st.StreamDecoder("cam", "rtsp://127.0.0.1:8554/cam", target_fps=30.0,
                             playout_idle_period=2.0)
        pushes = []
        real_push = d.playout.push
        def counting_push(image):
            pushes.append(time.perf_counter()); return real_push(image)
        d.playout.push = counting_push
        with mock.patch("cvti.serving.capture.open_capture", return_value=_Cap()):
            d.start()
            time.sleep(1.0)
            unwatched_pushes = len(pushes)
            unwatched_skipped = d.playout_skipped
            if display_fps_after is not None:
                d.display_fps = display_fps_after
                time.sleep(1.0)
            d.stop()
        return d, unwatched_pushes, unwatched_skipped, len(pushes)

    def test_nobody_watching_means_one_keepalive_encode_every_two_seconds(self):
        d, pushes, skipped, _ = self._run(None)
        self.assertLessEqual(pushes, 1, f"{pushes} encodes in a second with no viewer")
        self.assertGreater(skipped, 10, "decoded frames must be skipped, not encoded")
        self.assertEqual(d.ingest_status()["playout_skipped"], d.playout_skipped)

    def test_a_viewer_turns_encoding_back_on(self):
        d, before, _, after = self._run(display_fps_after=8.0)
        self.assertLessEqual(before, 1)
        self.assertGreater(after - before, 5, "a watched camera must feed the playout again")

    def test_evidence_and_detection_never_depended_on_the_playout(self):
        import inspect
        from cvti.serving import camera, pipeline
        self.assertNotIn("playout", inspect.getsource(camera))
        import re
        src = inspect.getsource(pipeline)
        calls = re.findall(r"\.playout\.(\w+)\(", src)
        self.assertEqual(set(calls), {"pop_due"}, "only the smooth-publish loop reads the playout")


if __name__ == "__main__":
    unittest.main()
