"""The MJPEG proxy shapes a stream for a remote viewer (cvti/api/mjpeg_shaper.py)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from cvti.api.mjpeg_shaper import MjpegShaper, remote_profile


def _jpeg(width: int = 640, height: int = 360) -> bytes:
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, : width // 2] = (0, 0, 255)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    return enc.tobytes()


def _part(jpeg: bytes) -> bytes:
    return (b"--argusframe\r\nContent-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")


def _parts(stream: bytes) -> list[bytes]:
    out = []
    buf = stream
    while True:
        i = buf.find(b"\r\n\r\n")
        if i < 0:
            break
        head, buf = buf[:i], buf[i + 4:]
        n = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        out.append(buf[:n]); buf = buf[n + 2:]
    return out


def _decoded_width(jpeg: bytes) -> int:
    import cv2
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR).shape[1]


class ShaperTests(unittest.TestCase):
    def test_frames_above_max_fps_are_dropped(self):
        t = [0.0]
        s = MjpegShaper(max_fps=5, width=0, clock=lambda: t[0])
        jpeg = _jpeg()
        sent = 0
        for i in range(20):                       # 20 frames in one second
            t[0] = i * 0.05
            sent += len(_parts(s.feed(_part(jpeg))))
        self.assertEqual(sent, 5)
        self.assertEqual(s.dropped, 15)

    def test_the_average_rate_is_the_rate_asked_for(self):
        # 20 fps in, 8 asked: a naive "a period since the last send" gives
        # every 3rd frame (6.7 fps). The accumulator gives 8.
        t = [0.0]
        s = MjpegShaper(max_fps=8, width=0, clock=lambda: t[0])
        jpeg = _jpeg()
        sent = 0
        for i in range(100):                      # 5 seconds at 20 fps
            t[0] = i * 0.05
            sent += len(_parts(s.feed(_part(jpeg))))
        self.assertGreaterEqual(sent, 39)
        self.assertLessEqual(sent, 41)

    def test_kept_frames_are_reencoded_smaller(self):
        s = MjpegShaper(max_fps=0, width=320, quality=50)
        jpeg = _jpeg(640, 360)
        out = _parts(s.feed(_part(jpeg)))
        self.assertEqual(len(out), 1)
        self.assertEqual(_decoded_width(out[0]), 320)
        self.assertLess(len(out[0]), len(jpeg))

    def test_a_frame_split_across_chunks_is_reassembled(self):
        s = MjpegShaper(max_fps=0, width=0)
        part = _part(_jpeg())
        cut = len(part) // 3
        first = s.feed(part[:cut])
        self.assertEqual(first, b"")
        second = s.feed(part[cut: 2 * cut])
        self.assertEqual(second, b"")
        third = s.feed(part[2 * cut:])
        self.assertEqual(_parts(third), [_jpeg()])

    def test_passthrough_keeps_the_part_format(self):
        s = MjpegShaper(max_fps=0, width=0)
        part = _part(_jpeg())
        self.assertEqual(s.feed(part), part)

    def test_small_frames_are_not_upscaled(self):
        s = MjpegShaper(max_fps=0, width=640)
        out = _parts(s.feed(_part(_jpeg(320, 180))))
        self.assertEqual(_decoded_width(out[0]), 320)


class ProfileTests(unittest.TestCase):
    def test_defaults_shape_for_a_tunnel(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("ARGUS_REMOTE_STREAM_FPS", "ARGUS_REMOTE_STREAM_WIDTH",
                      "ARGUS_REMOTE_STREAM_QUALITY"):
                os.environ.pop(k, None)
            p = remote_profile()
        self.assertEqual(p, {"max_fps": 8.0, "width": 480, "quality": 55})

    def test_query_zero_means_as_published(self):
        self.assertIsNone(remote_profile(fps=0, width=0))

    def test_environment_tunes_a_box_without_a_rebuild(self):
        with mock.patch.dict(os.environ, {"ARGUS_REMOTE_STREAM_FPS": "4",
                                          "ARGUS_REMOTE_STREAM_WIDTH": "360"}):
            p = remote_profile()
        self.assertEqual((p["max_fps"], p["width"]), (4.0, 360))

    def test_query_beats_environment(self):
        with mock.patch.dict(os.environ, {"ARGUS_REMOTE_STREAM_FPS": "4"}):
            p = remote_profile(fps=12)
        self.assertEqual(p["max_fps"], 12.0)


if __name__ == "__main__":
    unittest.main()
