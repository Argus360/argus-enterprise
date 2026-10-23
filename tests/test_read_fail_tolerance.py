"""Transient decode failures on a live stream must not tear down RTSP
(cvti/serving/streams.py). A lossy 4MP HEVC feed over a WAN returns a bad read
on every corrupted frame; reconnecting on the first one made the CHI cameras
flap. The decoder now skips the bad frame and only reconnects once no good frame
has arrived for read_fail_grace_seconds."""
from __future__ import annotations

import unittest

from cvti.serving.streams import DEFAULT_READ_FAIL_GRACE, StreamDecoder


class ReadFailGraceKnob(unittest.TestCase):
    def test_default_is_a_few_seconds(self):
        d = StreamDecoder("cam", "rtsp://x/stream")
        self.assertEqual(d.read_fail_grace_seconds, DEFAULT_READ_FAIL_GRACE)
        self.assertGreaterEqual(d.read_fail_grace_seconds, 1.0)

    def test_override(self):
        d = StreamDecoder("cam", "rtsp://x/stream", read_fail_grace_seconds=8.0)
        self.assertEqual(d.read_fail_grace_seconds, 8.0)

    def test_tolerance_window_vs_offline_grace(self):
        # Skipping a glitch is a sub-second-to-seconds call; declaring OFFLINE is
        # a much longer one. The read grace must sit well below the offline grace
        # so a flaky feed is smoothed, not hidden when it truly goes down.
        d = StreamDecoder("cam", "rtsp://x/stream")
        self.assertLess(d.read_fail_grace_seconds, d.offline_grace_seconds)


if __name__ == "__main__":
    unittest.main()
