"""Check the profiler's measurement and stale-health boundaries."""
import json
import time

from tools.argus_profiler.profile import cpu_cores, fresh_run_dir, go_stack_port, read_engine, role_for


def test_cpu_cores_rejects_counter_outlier_and_reset():
    assert cpu_cores(3.0, 1.0, 2.0, 4) == 1.0
    assert cpu_cores(20.0, 1.0, 2.0, 4) is None
    assert cpu_cores(1.0, 3.0, 2.0, 4) is None


def test_classification_keeps_command_lines_out_of_results():
    assert role_for("argus-engine.exe") == "engine"
    assert role_for("go2rtc.exe") == "go2rtc"
    assert role_for("python.exe", ["python", "-m", "cvti.serving.pipeline", "rtsp://secret"]) == "engine"
    assert role_for("python.exe", ["python", "unrelated.py"]) is None


def test_engine_metrics_marks_stale_health(tmp_path):
    path = tmp_path / "gate_health.json"
    path.write_text(json.dumps({"generated_at": time.time() - 120,
                                "engine": {"frames_processed": 24}}))
    stale = read_engine(tmp_path)
    assert stale["health_fresh"] is False
    assert stale["health_age_s"] >= 100
    path.write_text(json.dumps({"generated_at": time.time(),
                                "engine": {"frames_processed": 25}}))
    fresh = read_engine(tmp_path)
    assert fresh["health_fresh"] is True
    assert fresh["frames_processed"] == 25


def test_engine_metrics_omits_stale_stage_timings(tmp_path):
    (tmp_path / "gate_health.json").write_text(json.dumps({"generated_at": time.time()}))
    (tmp_path / "perf_report.json").write_text(json.dumps({
        "generated_at": time.time() - 120,
        "stages": {"detect_batch": {"engine": {"p95_ms": 999}}}}))
    metrics = read_engine(tmp_path)
    assert metrics["health_fresh"] is True
    assert metrics["perf_fresh"] is False
    assert metrics["detect_p95_ms"] is None


def test_discovers_fresh_feed_when_requested_site_is_stale(tmp_path, monkeypatch):
    root = tmp_path / "Argus"
    site = root / "site"
    feed = root / "runs" / "feeds" / "live"
    site.mkdir(parents=True)
    feed.mkdir(parents=True)
    (site / "gate_health.json").write_text(json.dumps({"generated_at": time.time() - 3600}))
    (feed / "gate_health.json").write_text(json.dumps({"generated_at": time.time()}))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert fresh_run_dir(site) == feed


def test_go_stack_port_requires_matching_live_pid(tmp_path):
    (tmp_path / "go2rtc.pid").write_text("1234")
    (tmp_path / "stream_gateway.json").write_text(json.dumps({"api_port": 62943}))
    assert go_stack_port(tmp_path, {1234}) == 62943
    assert go_stack_port(tmp_path, {9999}) is None
