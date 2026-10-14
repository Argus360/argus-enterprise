# Argus profiler

This read-only tool samples a running Argus installation. It works with the
packaged executables or a source launch. It never starts monitoring, connects
to a camera, or changes site settings.

## Quick start

Install Python 3.10+ and `psutil` (`requirements.txt` already includes it).
Start Argus monitoring, then run:

```powershell
python -m pip install psutil
python tools/argus_profiler/profile.py --duration 120 --label phone-2-streams
```

To include aggregate camera health and stage timings, pass the engine output
directory containing `gate_health.json` and `perf_report.json`:

```powershell
python tools/argus_profiler/profile.py --duration 120 --label phone-2-streams --run-dir "C:\path\to\site-output"
```

Optional source-line stack sampling requires `py-spy`. On Windows, attaching
to a packaged engine may require an elevated terminal. Process sampling still
runs if stack attachment fails; inspect `stack_status.json`.

```powershell
python -m pip install py-spy
python tools/argus_profiler/profile.py --duration 120 --label phone-2-streams --run-dir "C:\path\to\site-output" --stacks
```

The tool prints a timestamped output folder. Ctrl+C finishes early and writes
the summary. `--gpu auto` (the default) reads NVIDIA `nvidia-smi`, or Jetson
`tegrastats`, if available. `--gpu none` disables GPU collection.

## Files and interpretation

- `processes.csv`: Argus engine, API, desktop, go2rtc, Ollama, and go2rtc child
  process CPU core equivalents, RSS, thread count, and I/O rate.
- `system.csv`: whole-machine CPU and memory, GPU utilization and memory when
  available. Whole-machine CPU includes unrelated programs.
- `engine_metrics.csv`: aggregate frames, connected/offline count, model
  backend, detection/decode timing and gateway status when `--run-dir` is set.
  `health_fresh`/`perf_fresh` flag files over 30 seconds old. Throughput
  excludes stale health; stale stage timings are omitted.
- `summary.json`: per-process mean/peak CPU and memory, observed frames per
  second, and the last engine metrics.
- With `--stacks`: `engine.raw.txt`, `stacks.txt`, `stack_status.json`.

One CPU core equivalent is a full logical CPU; `machine_pct` divides by the
host's logical CPU count. Stack percentages are sampled stack frequency,
**not CPU percentages**. Native ONNX/OpenCV worker threads require an OS/native
profiler for exact CPU attribution.

For capacity tests, hold model, camera codec/resolution/FPS, target FPS, wall
state, and duration constant. Take an idle baseline, then repeat at 2, 4, 8,
etc. streams on each machine. Use labels such as `jetson-4-wall-open`, run each
point more than once, and compare throughput, CPU cores, GPU load, memory,
stage p95, and connected camera count. A phone camera is useful for the first
run but may differ from a customer's RTSP camera.

The tool does not save command lines, camera URLs or IDs, credentials, frames,
or publisher tokens. Review labels/hostnames and model names before sharing.
