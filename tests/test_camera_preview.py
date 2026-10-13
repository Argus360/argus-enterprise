from unittest.mock import MagicMock, patch

import pytest

from cvti.app.preview import CameraPreview


@pytest.fixture
def preview():
    with patch("cvti.app.preview.FrameServer") as server:
        server.return_value.port = 12345
        server.return_value.token = "secret"
        p = CameraPreview()
        yield p
        p.close()


def test_descriptor_is_lazy_and_encoded(preview):
    result = preview.descriptor("front / door", 0)
    assert result["preview"]
    assert "front%20%2F%20door?token=secret" in result["url"]
    assert preview._walls == {}
    assert not preview.acquire("unknown")


def test_startup_deadline_is_distinct_from_live_stall():
    from cvti.app.live_wall import FrameServer
    assert not FrameServer.expired(12, False)
    assert not FrameServer.expired(30, False)
    assert FrameServer.expired(31, False)
    assert FrameServer.expired(6, True)


def test_twelve_camera_descriptors_have_distinct_origins():
    from urllib.parse import urlsplit
    p = CameraPreview()
    try:
        descriptors = [p.descriptor(str(i), "rtsp://example/live") for i in range(12)]
        assert len({urlsplit(d["url"]).netloc for d in descriptors}) == 12
        assert p.descriptor("0", "rtsp://example/live") == descriptors[0]
        assert not p._walls
        assert all(r["stream_request_count"] == 0 for r in p.diagnostics()["cameras"].values())
    finally:
        p.close()


def test_viewers_share_capture_and_last_viewer_releases(preview):
    preview.descriptor("cam", 0)
    with patch("cvti.app.preview.LiveWall") as factory:
        wall = factory.return_value.start.return_value
        wall._threads = []
        assert preview.acquire("cam")
        assert preview.acquire("cam")
        factory.assert_called_once()
        preview.release("cam")
        wall.stop.assert_not_called()
        preview.release("cam")
        wall.stop.assert_called_once()
        assert preview._walls == {}


def test_snapshot_reuses_capture_and_dimensions(preview):
    preview.descriptor("cam", 0)
    with patch("cvti.app.preview.LiveWall") as factory:
        wall = factory.return_value.start.return_value
        wall._threads = []
        wall.snapshot.return_value = {"uri": "data:image/jpeg;base64,anBn", "w": 1920, "h": 1080}
        preview.acquire("cam")
        snapshot = preview.snapshot("cam")
        assert (snapshot["w"], snapshot["h"]) == (1920, 1080)
        factory.assert_called_once()
        preview.release("cam")


def test_handover_refuses_to_overlap_blocked_capture(preview):
    preview.descriptor("cam", 0)
    with patch("cvti.app.preview.LiveWall") as factory:
        wall = factory.return_value.start.return_value
        thread = MagicMock()
        thread.is_alive.return_value = True
        wall._threads = [thread]
        preview.acquire("cam")
        with pytest.raises(RuntimeError, match="releasing"):
            preview.close()
        assert not preview.acquire("cam")
        thread.is_alive.return_value = False


def test_disconnected_capture_does_not_serve_old_jpeg():
    from cvti.app.live_wall import LiveWall
    wall = LiveWall([])
    wall._set("cam", jpeg=b"old", ok=True)
    assert wall.jpeg("cam") == b"old"
    wall._set("cam", ok=False)
    assert wall.jpeg("cam") is None


def test_stopping_capture_is_not_reused_and_replaced_after_exit(preview):
    preview.descriptor("cam", 0)
    with patch("cvti.app.preview.LiveWall") as factory:
        old, new = MagicMock(), MagicMock()
        thread = MagicMock()
        thread.is_alive.return_value = True
        old._threads = [thread]
        new._threads = []
        factory.return_value.start.side_effect = [old, new]
        assert preview.acquire("cam")
        preview.release("cam")
        assert not preview.acquire("cam")
        assert preview.jpeg("cam") is None
        assert factory.call_count == 1
        thread.is_alive.return_value = False
        assert preview.acquire("cam")
        assert factory.call_count == 2
        assert preview._walls["cam"] is new
        preview.release("cam")


def test_preview_scales_jpeg_but_snapshot_keeps_source_resolution():
    import cv2
    import numpy as np
    import base64
    from cvti.app.live_wall import LiveWall
    wall = LiveWall([], width=960, retain_snapshot=True)
    cap = MagicMock()

    def read():
        wall._stop.set()
        return True, np.zeros((1080, 1920, 3), dtype=np.uint8)

    cap.read.side_effect = read
    with patch.object(wall, "_open", return_value=cap):
        wall._decode("cam", "rtsp://test/live")
    preview_frame = cv2.imdecode(np.frombuffer(wall.jpeg("cam"), dtype=np.uint8), 1)
    assert preview_frame.shape[:2] == (540, 960)
    timings = wall.diagnostics()["cam"]["stage_wall_times"]
    assert set(timings) == {"read", "resize", "jpeg_encode"}
    assert all(row["count"] == 1 and row["mean_ms"] >= 0 for row in timings.values())
    snapshot = wall.snapshot("cam")
    assert (snapshot["w"], snapshot["h"]) == (1920, 1080)
    image = cv2.imdecode(np.frombuffer(base64.b64decode(snapshot["uri"].split(",")[1]), dtype=np.uint8), 1)
    assert image.shape[:2] == (1080, 1920)


@pytest.mark.parametrize("source", [0, "0", "clip.mp4", "rtsp://camera/live"])
def test_webcam_warmup_discards_initial_frames_only(source):
    import numpy as np
    from cvti.app.live_wall import LiveWall

    wall = LiveWall([])
    cap = MagicMock()
    frames = [np.full((16, 16, 3), value, dtype=np.uint8)
              for value in (2, 90, 130)]
    cap.read.side_effect = [(True, frame) for frame in frames]
    published = []

    def publish(frame):
        published.append(int(frame.mean()))
        wall._stop.set()
        return frame

    with patch.object(wall, "_open", return_value=cap), \
         patch.object(wall, "_downscale", side_effect=publish), \
         patch("cvti.app.live_wall.time.monotonic", side_effect=[10, 11, 12]):
        wall._decode("cam", source)
    assert published == ([130] if source in (0, "0") else [2])
    cap.release.assert_called()


def test_engine_does_not_start_until_preview_releases(tmp_path):
    from cvti.app.console_backend import ConsoleBackend
    backend = ConsoleBackend.__new__(ConsoleBackend)
    backend.site_path = tmp_path / "site.json"
    backend._preview = MagicMock()
    backend._preview.close.side_effect = RuntimeError("capture still releasing")
    with patch("cvti.app.console_backend.subprocess.Popen") as spawn:
        with pytest.raises(RuntimeError, match="releasing"):
            backend._spawn_engine()
        spawn.assert_not_called()
    # The refused close must not leave the dead preview behind: Watch would
    # read "preview unavailable" until somebody pressed Start again.
    assert backend._preview is None


def test_a_refused_handover_is_a_retryable_503_not_a_500():
    from cvti.app.errors import PreviewBusy
    from cvti.app.console_backend import ConsoleBackend
    backend = ConsoleBackend.__new__(ConsoleBackend)
    backend._preview = MagicMock()
    backend._preview.close.side_effect = PreviewBusy("Camera preview is still releasing its capture; retry monitoring shortly")
    with pytest.raises(PreviewBusy):
        backend._close_preview()
    assert backend._preview is None


def test_real_stream_moves_and_releases_capture(tmp_path):
    import http.client
    import time
    import cv2
    import numpy as np
    from urllib.parse import urlsplit

    clip = str(tmp_path / "preview.avi")
    writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"MJPG"), 8, (64, 48))
    assert writer.isOpened()
    for value in range(0, 240, 20):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    p = CameraPreview()
    connection = None
    try:
        url = urlsplit(p.descriptor("test", clip)["url"])
        assert not p._walls
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
        connection.request("GET", url.path + "?" + url.query)
        response = connection.getresponse()
        assert response.status == 200
        frames = []
        for _ in range(3):
            assert response.readline().strip() == b"--arguswall"
            assert response.readline().strip() == b"Content-Type: image/jpeg"
            size = int(response.readline().split(b":")[1])
            response.readline()
            frames.append(response.read(size))
            response.readline()
        assert len(set(frames)) > 1
        assert p.snapshot("test")["w"] == 64
        response.close()
        connection.close()
        deadline = time.monotonic() + 6
        while p._walls and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not p._walls
    finally:
        if connection:
            connection.close()
        p.close()
