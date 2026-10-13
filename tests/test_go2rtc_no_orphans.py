"""Stop/Start must not leave a go2rtc behind for every restart.

Stop monitoring terminates the engine; on Windows that is TerminateProcess, so
the engine's own `gateway.stop()` never runs. The pilot server showed three
go2rtc processes from three restarts (8 Oct 2026).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving import go2rtc as g


def _fake_gateway(out: Path, *, name_hint: str = "go2rtc") -> subprocess.Popen:
    """A long-running stand-in whose command line names the gateway config."""
    script = out / f"{name_hint}_stub.py"
    script.write_text("import time\nwhile True: time.sleep(0.2)\n")
    return subprocess.Popen([sys.executable, str(script), "-config", str(out / "go2rtc.yaml")])


class ReapStaleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_pidfile_is_a_noop(self):
        self.assertEqual(g.reap_stale(self.out), 0)

    def test_reap_stale_retracts_a_dead_gateways_advertisement(self):
        (self.out / "stream_gateway.json").write_text('{"api_port": 2}')
        g.reap_stale(self.out)
        self.assertFalse((self.out / "stream_gateway.json").exists())

    def test_a_leftover_gateway_is_stopped_and_the_pidfile_removed(self):
        proc = _fake_gateway(self.out)
        try:
            (self.out / g.PIDFILE).write_text(str(proc.pid))
            import psutil
            real = psutil.Process.name
            with patch.object(psutil.Process, "name", lambda self: "go2rtc.exe" if self.pid == proc.pid else real(self)):
                self.assertEqual(g.reap_stale(self.out), 1)
            proc.wait(timeout=5)
            self.assertIsNotNone(proc.poll(), "the leftover gateway is still running")
            self.assertFalse((self.out / g.PIDFILE).exists())
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_a_recycled_pid_that_is_not_go2rtc_is_left_alone(self):
        proc = _fake_gateway(self.out, name_hint="unrelated")
        try:
            (self.out / g.PIDFILE).write_text(str(proc.pid))
            # Its process name is python, not go2rtc: never touched.
            self.assertEqual(g.reap_stale(self.out), 0)
            time.sleep(0.3)
            self.assertIsNone(proc.poll(), "an unrelated process was killed")
        finally:
            proc.kill()

    def test_another_sites_gateway_is_left_alone(self):
        other = Path(tempfile.mkdtemp())
        proc = _fake_gateway(other)           # its config lives elsewhere
        try:
            (self.out / g.PIDFILE).write_text(str(proc.pid))
            import psutil
            real = psutil.Process.name
            with patch.object(psutil.Process, "name", lambda self: "go2rtc.exe" if self.pid == proc.pid else real(self)):
                self.assertEqual(g.reap_stale(self.out), 0)
            time.sleep(0.3)
            self.assertIsNone(proc.poll(), "a gateway serving a different output dir was killed")
        finally:
            proc.kill()

    def test_a_garbage_pidfile_is_cleared_without_error(self):
        (self.out / g.PIDFILE).write_text("not-a-pid")
        self.assertEqual(g.reap_stale(self.out), 0)


class StopEngineReapsTest(unittest.TestCase):
    def test_stop_monitoring_cleans_up_the_gateway(self):
        from cvti.app.console_backend import ConsoleBackend
        be = ConsoleBackend.__new__(ConsoleBackend)
        be._monitor = None
        be.db_path = str(Path(tempfile.mkdtemp()) / "events.db")
        with patch("cvti.serving.go2rtc.reap_stale") as reap:
            be._stop_engine()
        reap.assert_called_once_with(Path(be.db_path).parent)

    def test_stop_monitoring_removes_the_markers_a_killed_engine_left(self):
        """frames.json and stream_gateway.json say "a publisher/gateway is
        alive here". After a hard terminate the engine never removes them, so
        every tile resolved a dead port until the next Start (11 Oct 2026)."""
        from cvti.app.console_backend import ConsoleBackend
        be = ConsoleBackend.__new__(ConsoleBackend)
        be._monitor = None
        out = Path(tempfile.mkdtemp())
        be.db_path = str(out / "events.db")
        (out / "frames.json").write_text('{"port": 1, "token": "t"}')
        (out / "stream_gateway.json").write_text('{"api_port": 2}')
        with patch("cvti.serving.go2rtc.reap_stale"):
            be._stop_engine()
        self.assertFalse((out / "frames.json").exists())
        self.assertFalse((out / "stream_gateway.json").exists())
        be._stop_engine()                     # nothing left: still fine


if __name__ == "__main__":
    unittest.main()
