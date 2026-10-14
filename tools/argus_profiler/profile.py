"""Standalone, read-only sampler for a running Argus installation."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean

try:
    import psutil
except ImportError as exc:
    raise SystemExit("Install psutil first: python -m pip install psutil") from exc


PROCESS_FIELDS = ("utc", "elapsed_s", "pid", "ppid", "role", "name", "cpu_cores",
                  "machine_pct", "rss_mb", "threads", "read_mb_s", "write_mb_s")
SYSTEM_FIELDS = ("utc", "elapsed_s", "cpu_pct", "memory_pct", "available_gb",
                 "gpu_backend", "gpu_util_pct", "gpu_used_mb", "gpu_total_mb")
ENGINE_FIELDS = ("utc", "elapsed_s", "health_age_s", "health_fresh",
                 "perf_age_s", "perf_fresh", "version",
                 "cameras", "connected", "offline",
                 "frames_processed", "target_fps", "alerts_queued", "detect_backend",
                 "detect_provider", "detect_mean_ms", "detect_p95_ms",
                 "detect_busy_fraction", "decode_mean_ms", "gateway_running",
                 "gateway_streams", "gate_pending")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def role_for(name: str, cmdline: list[str] | None = None) -> str | None:
    name = name.lower().removesuffix(".exe")
    direct = {"argus-engine": "engine", "argus-api": "api", "argus": "desktop",
              "go2rtc": "go2rtc", "ollama": "ollama"}
    if name in direct:
        return direct[name]
    if name.startswith(("ollama_llama", "ollama-runner")):
        return "ollama"
    if name.startswith("python") and cmdline:
        # The command line is only inspected in memory; it may contain secrets.
        command = " ".join(cmdline).lower().replace("\\", "/")
        if "cvti.serving.pipeline" in command or "serving/pipeline.py" in command:
            return "engine"
        if "cvti.api" in command or "api/__main__.py" in command:
            return "api"
    return None


def cpu_cores(new: float, old: float, seconds: float, logical: int) -> float | None:
    if seconds <= 0 or new < old:
        return None
    cores = (new - old) / seconds
    # Some Windows initial process counters are stale; >host capacity is invalid.
    return cores if 0 <= cores <= logical + 0.05 else None


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def fresh_run_dir(requested: Path | None) -> Path | None:
    """Prefer the active installed feed over a stale site output directory."""
    choices = [requested] if requested else []
    appdata = os.environ.get("APPDATA")
    if appdata:
        root = Path(appdata) / "Argus"
        choices.extend((root / "runs" / "feeds").glob("*/gate_health.json"))
        choices.append(root / "site" / "gate_health.json")
    candidates = []
    for choice in choices:
        if choice is None:
            continue
        directory = choice if choice.is_dir() else choice.parent
        health = read_json(directory / "gate_health.json")
        stamp = health.get("generated_at")
        if isinstance(stamp, (int, float)) and 0 <= time.time() - stamp <= 30:
            candidates.append((stamp, directory))
    return max(candidates, default=(0, requested))[1]


def pyspy_binary() -> str | None:
    sibling = Path(sys.executable).with_name("py-spy.exe" if os.name == "nt" else "py-spy")
    return str(sibling) if sibling.is_file() else shutil.which("py-spy")


def go_stack_port(run_dir: Path, running_pids: set[int]) -> int | None:
    """Use only the loopback API descriptor belonging to a sampled gateway PID."""
    try:
        pid = int((run_dir / "go2rtc.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if pid not in running_pids:
        return None
    descriptor = read_json(run_dir / "stream_gateway.json")
    port = descriptor.get("api_port")
    return port if isinstance(port, int) and 1 <= port <= 65535 else None


def save_go_stack(port: int, target: Path, stamp: str, pid: int) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stack", timeout=3) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("go2rtc stack response exceeded 2 MB")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"Captured {stamp}; go2rtc PID {pid}\n\n" +
                      data.decode("utf-8", errors="replace"), encoding="utf-8")


def read_engine(run_dir: Path) -> dict:
    health = read_json(run_dir / "gate_health.json")
    perf = read_json(run_dir / "perf_report.json")
    cameras = health.get("cameras") or []
    stages = perf.get("stages") or {}
    detect = (stages.get("detect_batch") or {}).get("engine") or {}
    decodes = (stages.get("decode") or {}).values()
    decode_ms = [d["mean_ms"] for d in decodes if isinstance(d, dict)
                 and isinstance(d.get("mean_ms"), (int, float))]
    engine = health.get("engine") or {}
    detector = health.get("detector") or {}
    gateway = health.get("stream_gateway") or {}
    gate = health.get("gate") or {}
    generated = health.get("generated_at")
    age = max(0.0, time.time() - generated) if isinstance(generated, (int, float)) else None
    fresh = age is not None and age <= 30
    perf_generated = perf.get("generated_at")
    perf_age = max(0.0, time.time() - perf_generated) if isinstance(perf_generated, (int, float)) else None
    perf_fresh = perf_age is not None and perf_age <= 30
    return {"health_age_s": round(age, 1) if age is not None else None,
            "health_fresh": fresh,
            "perf_age_s": round(perf_age, 1) if perf_age is not None else None,
            "perf_fresh": perf_fresh,
            "version": health.get("version"), "cameras": engine.get("cameras"),
            "connected": sum(c.get("state") == "connected" for c in cameras if isinstance(c, dict)) if health else None,
            "offline": sum(c.get("state") == "offline" for c in cameras if isinstance(c, dict)) if health else None,
            "frames_processed": engine.get("frames_processed"),
            "target_fps": engine.get("target_fps"), "alerts_queued": engine.get("alerts_queued"),
            "detect_backend": detector.get("backend"), "detect_provider": detector.get("provider"),
            "detect_mean_ms": detect.get("mean_ms") if perf_fresh else None,
            "detect_p95_ms": detect.get("p95_ms") if perf_fresh else None,
            "detect_busy_fraction": detect.get("busy_fraction") if perf_fresh else None,
            "decode_mean_ms": round(mean(decode_ms), 2) if perf_fresh and decode_ms else None,
            "gateway_running": gateway.get("running"), "gateway_streams": gateway.get("streams"),
            "gate_pending": gate.get("pending")}


class JetsonStats:
    def __init__(self) -> None:
        self.latest: dict = {}
        self.proc: subprocess.Popen | None = None

    def start(self) -> bool:
        binary = shutil.which("tegrastats")
        if not binary:
            return False
        try:
            self.proc = subprocess.Popen([binary, "--interval", "1000"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True)
        except OSError:
            return False
        threading.Thread(target=self._read, daemon=True).start()
        return True

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            gpu = re.search(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
            # Jetson RAM is shared; report it via system memory, never as VRAM.
            self.latest = {"gpu_util_pct": float(gpu.group(1)) if gpu else None,
                           "gpu_used_mb": None, "gpu_total_mb": None}

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def nvidia_gpu() -> dict | None:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        result = subprocess.run([binary, "--query-gpu=utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, timeout=4, check=True)
        rows = [[float(v.strip()) for v in line.split(",")]
                for line in result.stdout.splitlines() if line.strip()]
        if not rows or any(len(row) != 3 for row in rows):
            return None
        return {"gpu_util_pct": round(mean(r[0] for r in rows), 1),
                "gpu_used_mb": sum(r[1] for r in rows), "gpu_total_mb": sum(r[2] for r in rows)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def select_gpu(request: str) -> tuple[str, JetsonStats | None]:
    if request == "none":
        return "none", None
    if request in ("auto", "nvidia") and nvidia_gpu() is not None:
        return "nvidia-smi", None
    if request in ("auto", "jetson"):
        jetson = JetsonStats()
        if jetson.start():
            return "tegrastats", jetson
    return "unavailable", None


def snapshot() -> dict[tuple[int, float], dict]:
    candidates = []
    for proc in psutil.process_iter(["pid", "ppid", "name"]):
        try:
            name = proc.info["name"] or ""
            cmd = proc.cmdline() if name.lower().startswith("python") else None
            role = role_for(name, cmd)
            if role or name.lower().removesuffix(".exe") == "ffmpeg":
                candidates.append((proc, name, role))
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    gateways = {p.pid for p, _, role in candidates if role == "go2rtc"}
    found = {}
    for proc, name, role in candidates:
        try:
            if not role:
                if proc.ppid() not in gateways:
                    continue
                role = "go2rtc-child"
            times = proc.cpu_times()
            try:
                io = proc.io_counters()
            except (psutil.AccessDenied, AttributeError, OSError):
                io = None
            found[(proc.pid, proc.create_time())] = {
                "pid": proc.pid, "ppid": proc.ppid(), "role": role, "name": name,
                "cpu_s": times.user + times.system,
                "rss_mb": proc.memory_info().rss / 2**20, "threads": proc.num_threads(),
                "read_bytes": io.read_bytes if io else None,
                "write_bytes": io.write_bytes if io else None}
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return found


def percentile(values: list[float], fraction: float) -> float | None:
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)] if values else None


def write_stacks(raw: Path, output: Path) -> None:
    deepest, leaves = defaultdict(int), defaultdict(int)
    total = 0
    try:
        for line in raw.read_text(encoding="utf-8").splitlines():
            stack, count_text = line.rsplit(" ", 1)
            count = int(count_text)
            frames = stack.split(";")
            total += count
            leaves[frames[-1]] += count
            own = [f for f in frames if "(cvti\\" in f or "(cvti/" in f]
            if own:
                deepest[own[-1]] += count
    except (OSError, ValueError):
        return
    if not total:
        return
    with output.open("w", encoding="utf-8") as handle:
        handle.write(f"{total} sampled stacks; frequencies are not CPU shares.\n\n")
        for title, counter in (("Deepest Argus frame", deepest), ("Leaf frame", leaves)):
            handle.write(f"{title}:\n")
            for frame, count in sorted(counter.items(), key=lambda pair: -pair[1])[:25]:
                handle.write(f"{count:6d} {count / total:6.1%} {frame}\n")
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--label", default="run")
    parser.add_argument("--run-dir", type=Path, help="engine output directory")
    parser.add_argument("--output-dir", type=Path, help="parent for timestamped output")
    parser.add_argument("--gpu", choices=("auto", "none", "nvidia", "jetson"), default="auto")
    parser.add_argument("--gpu-interval", type=float, default=5)
    parser.add_argument("--stacks", dest="stacks", action="store_true", default=True,
                        help="capture Python and go2rtc stacks (default)")
    parser.add_argument("--no-stacks", dest="stacks", action="store_false",
                        help="disable stack capture for clean capacity benchmarks")
    parser.add_argument("--stack-rate", type=int, default=50)
    parser.add_argument("--go-stack-interval", type=float, default=15)
    args = parser.parse_args()
    if min(args.duration, args.interval, args.gpu_interval, args.stack_rate,
           args.go_stack_interval) <= 0:
        parser.error("duration, intervals, and stack rate must be positive")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.label).strip(".-")[:60] or "run"
    root = args.output_dir or Path(__file__).with_name("runs")
    output = root / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + label)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Writing profile to {output.resolve()}", flush=True)
    logical = psutil.cpu_count(logical=True) or 1
    run_dir = fresh_run_dir(args.run_dir)
    if args.run_dir and run_dir != args.run_dir:
        print(f"Using active engine output: {run_dir} (requested directory is stale)", flush=True)
    elif args.run_dir and not read_engine(args.run_dir)["health_fresh"]:
        print(f"Warning: {args.run_dir} has no fresh engine health yet; looking for an active feed.", flush=True)
    manifest = {"version": 2, "label": args.label, "started_utc": utc_now(),
                "host": socket.gethostname(), "os": platform.platform(),
                "logical_cpus": logical, "physical_cpus": psutil.cpu_count(logical=False),
                "memory_total_gb": round(psutil.virtual_memory().total / 2**30, 2),
                "interval_s": args.interval, "requested_duration_s": args.duration,
                "requested_run_dir": str(args.run_dir) if args.run_dir else None,
                "stacks_requested": args.stacks}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    gpu_backend, jetson = select_gpu(args.gpu)
    psutil.cpu_percent(interval=None)
    previous = snapshot()
    started = previous_at = time.monotonic()
    next_gpu_at = 0.0
    gpu_value = {}
    process_rows, system_rows, engine_rows = [], [], []
    excluded = 0
    go_stack_count = 0
    go_stack_error = None
    last_go_stack = -args.go_stack_interval
    stack_proc = stack_handle = None
    stack_status = {"requested": args.stacks,
                    "status": "no engine observed" if args.stacks else "not requested"}
    pyspy = pyspy_binary() if args.stacks else None
    if args.stacks and not pyspy:
        stack_status["status"] = "py-spy not found beside Python or on PATH"
        print("Warning: py-spy not found; Python source-line stacks will be unavailable.", flush=True)
    try:
        with (output / "processes.csv").open("w", newline="", encoding="utf-8") as pf, \
             (output / "system.csv").open("w", newline="", encoding="utf-8") as sf, \
             (output / "engine_metrics.csv").open("w", newline="", encoding="utf-8") as ef:
            pw, sw, ew = (csv.DictWriter(f, fieldnames=fields) for f, fields in
                          ((pf, PROCESS_FIELDS), (sf, SYSTEM_FIELDS), (ef, ENGINE_FIELDS)))
            for writer in (pw, sw, ew):
                writer.writeheader()
            while time.monotonic() - started < args.duration:
                time.sleep(min(args.interval, max(0, args.duration - (time.monotonic() - started))))
                current = snapshot()
                now = time.monotonic()
                seconds = now - previous_at
                elapsed = round(now - started, 2)
                stamp = utc_now()
                if pyspy and stack_proc is None:
                    engine_pid = next((r["pid"] for r in current.values() if r["role"] == "engine"), None)
                    remaining = int(args.duration - elapsed)
                    if engine_pid and remaining >= 2:
                        stack_handle = (output / "py-spy.log").open("w", encoding="utf-8")
                        try:
                            stack_proc = subprocess.Popen(
                                [pyspy, "record", "--pid", str(engine_pid), "--format", "raw",
                                 "--duration", str(remaining), "--rate", str(args.stack_rate),
                                 "--output", str(output / "engine.raw.txt")],
                                stdout=stack_handle, stderr=subprocess.STDOUT)
                            stack_status = {"requested": True, "status": "running", "pid": engine_pid}
                        except OSError as exc:
                            stack_status = {"requested": True, "status": f"launch failed: {exc}"}
                            pyspy = None
                for key, row in current.items():
                    old = previous.get(key)
                    if old is None:
                        continue
                    cores = cpu_cores(row["cpu_s"], old["cpu_s"], seconds, logical)
                    if cores is None:
                        excluded += 1
                        continue
                    def io_rate(field: str) -> float | None:
                        new, before = row[field], old[field]
                        return round((new - before) / seconds / 2**20, 3) if (
                            new is not None and before is not None and new >= before) else None
                    record = {"utc": stamp, "elapsed_s": elapsed, "pid": row["pid"],
                              "ppid": row["ppid"], "role": row["role"], "name": row["name"],
                              "cpu_cores": round(cores, 3), "machine_pct": round(cores * 100 / logical, 1),
                              "rss_mb": round(row["rss_mb"], 1), "threads": row["threads"],
                              "read_mb_s": io_rate("read_bytes"), "write_mb_s": io_rate("write_bytes")}
                    pw.writerow(record)
                    process_rows.append(record)
                previous, previous_at = current, now
                if not run_dir or not read_engine(run_dir)["health_fresh"]:
                    found = fresh_run_dir(args.run_dir)
                    if found and found != run_dir:
                        run_dir = found
                        print(f"Using active engine output: {run_dir}", flush=True)
                if args.stacks and run_dir and elapsed - last_go_stack >= args.go_stack_interval:
                    gateways = {r["pid"] for r in current.values() if r["role"] == "go2rtc"}
                    hot = any(r["role"] == "go2rtc" and r["cpu_cores"] >= 0.2
                              for r in process_rows if r["utc"] == stamp)
                    port = go_stack_port(run_dir, gateways) if hot else None
                    if port:
                        last_go_stack = elapsed
                        try:
                            save_go_stack(port, output / "go2rtc_stacks" /
                                          f"{go_stack_count + 1:03d}.txt", stamp, next(iter(gateways)))
                            go_stack_count += 1
                        except (OSError, ValueError, urllib.error.URLError) as exc:
                            go_stack_error = type(exc).__name__ + ": " + str(exc)
                if gpu_backend == "nvidia-smi" and now >= next_gpu_at:
                    gpu_value = nvidia_gpu() or {}
                    next_gpu_at = now + args.gpu_interval
                elif gpu_backend == "tegrastats" and jetson:
                    gpu_value = jetson.latest.copy()
                memory = psutil.virtual_memory()
                system = {"utc": stamp, "elapsed_s": elapsed,
                          "cpu_pct": psutil.cpu_percent(interval=None), "memory_pct": memory.percent,
                          "available_gb": round(memory.available / 2**30, 2),
                          "gpu_backend": gpu_backend,
                          **{k: gpu_value.get(k) for k in ("gpu_util_pct", "gpu_used_mb", "gpu_total_mb")}}
                sw.writerow(system)
                system_rows.append(system)
                if run_dir:
                    engine = {"utc": stamp, "elapsed_s": elapsed, **read_engine(run_dir)}
                    ew.writerow(engine)
                    engine_rows.append(engine)
                pf.flush(); sf.flush(); ef.flush()
    except KeyboardInterrupt:
        print("Stopped early; writing summary.", flush=True)
    finally:
        if jetson:
            jetson.stop()
        if stack_proc:
            try:
                code = stack_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stack_proc.terminate()
                code = stack_proc.wait(timeout=3)
            stack_status["status"] = "complete" if code == 0 else f"exit {code}"
        if stack_handle:
            stack_handle.close()
        if (output / "engine.raw.txt").exists():
            write_stacks(output / "engine.raw.txt", output / "stacks.txt")
        (output / "stack_status.json").write_text(json.dumps(stack_status, indent=2), encoding="utf-8")
        grouped = defaultdict(list)
        for row in process_rows:
            grouped[(row["role"], row["pid"])].append(row)
        processes = [{"role": role, "pid": pid, "samples": len(rows),
                      "mean_cpu_cores": round(mean(r["cpu_cores"] for r in rows), 3),
                      "p95_cpu_cores": percentile([r["cpu_cores"] for r in rows], .95),
                      "peak_cpu_cores": max(r["cpu_cores"] for r in rows),
                      "peak_rss_mb": max(r["rss_mb"] for r in rows)}
                     for (role, pid), rows in grouped.items()]
        processes.sort(key=lambda p: -p["mean_cpu_cores"])
        valid = [(r["elapsed_s"], r["frames_processed"]) for r in engine_rows
                 if r["health_fresh"] and isinstance(r["frames_processed"], (int, float))]
        fps = (round((valid[-1][1] - valid[0][1]) / (valid[-1][0] - valid[0][0]), 2)
               if len(valid) > 1 and valid[-1][0] > valid[0][0] else None)
        gpu_values = [r["gpu_util_pct"] for r in system_rows if r["gpu_util_pct"] is not None]
        summary = {"label": args.label, "started_utc": manifest["started_utc"],
                   "finished_utc": utc_now(), "elapsed_s": round(time.monotonic() - started, 1),
                   "resolved_run_dir": str(run_dir) if run_dir else None,
                   "go2rtc_stack_snapshots": go_stack_count,
                   "go2rtc_stack_error": go_stack_error,
                   "logical_cpus": logical, "sample_count": len(system_rows),
                   "excluded_impossible_process_samples": excluded,
                   "mean_system_cpu_pct": round(mean(r["cpu_pct"] for r in system_rows), 1)
                   if system_rows else None,
                   "mean_gpu_util_pct": round(mean(gpu_values), 1) if gpu_values else None,
                   "observed_frames_per_second": fps,
                   "last_engine_metrics": engine_rows[-1] if engine_rows else None,
                   "processes": processes, "stack_status": stack_status}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output.resolve()), "samples": len(system_rows),
                          "processes": processes, "observed_frames_per_second": fps,
                          "stack_status": stack_status["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
