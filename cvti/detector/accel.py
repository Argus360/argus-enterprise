"""Detection rides the accelerator the box actually has (W2).

Shipped builds run CPU-torch while customer GPUs sit idle — the pilot's
Windows box has an iGPU that DirectML would use for near-free. Same YOLO
weights, exported once to ONNX at build time (scripts/export_onnx.py,
opset 17: the newest opset the shipping onnxruntime actually loads), run on
whatever accelerator exists. Parity is not hoped for, it is measured: the
exported model produced IDENTICAL boxes to torch on real footage (0.00px,
9 Sep spike), and tests/test_detection_rides_the_accelerator.py holds that.

The provider ladder, most-preferred first:

    DmlExecutionProvider        Windows iGPU/GPU (onnxruntime-directml)
    OpenVINOExecutionProvider   Intel CPU/iGPU (onnxruntime-openvino)
    CUDAExecutionProvider       NVIDIA (ultralytics selects this itself)
    CPUExecutionProvider        the floor — always works

CoreML is deliberately ABSENT: the 9 Sep spike showed onnxruntime's CoreML EP
partitions our dynamic-batch graph and fails at runtime — and the Mac never
needed it, torch-MPS already runs 2.7x there. Dev Macs keep torch; the ONNX
path exists for the boxes customers actually run.

Everything degrades to torch, loudly: no onnxruntime, no .onnx beside the
weights, a corrupt export, or the site saying `"detector_backend": "torch"` —
each lands on exactly today's path with the reason carried into health. An
accelerator is an upgrade, never a requirement (the go2rtc rule, again).

One deliberate transgression, fenced: ultralytics builds its ONNX provider
list internally and knows CUDA, CoreML, and CPU — nothing else. To get
DirectML/OpenVINO in front we wrap its backend's load_model and, when the
session came up CPU-only while a better provider is available, rebuild that
session. ultralytics is PINNED (requirements.txt); a version bump that moves
this seam fails the pin test rather than silently dropping the accelerator.
"""

from __future__ import annotations

from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

# Better-than-CPU providers we know how to ask for, most preferred first.
# CUDA is absent on purpose: ultralytics' own selection already handles it.
PREFERRED_PROVIDERS = ("DmlExecutionProvider", "OpenVINOExecutionProvider")

_PATCHED = False


def onnxruntime_or_none():
    """The onnxruntime module, or None — its absence is a supported state."""
    try:
        import onnxruntime
        return onnxruntime
    except ImportError:
        return None


def best_available_provider() -> str | None:
    """The best provider this box offers beyond plain CPU, or None."""
    ort = onnxruntime_or_none()
    if ort is None:
        return None
    try:
        available = set(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - a probe must never take detection down
        log.debug("provider probe failed", exc_info=True)
        return None
    for provider in PREFERRED_PROVIDERS:
        if provider in available:
            return provider
    return None


def _patch_ultralytics_providers() -> None:
    """Teach ultralytics' ONNX backend about DirectML/OpenVINO, once.

    Its load_model picks CUDA / CoreML / CPU and nothing else, so on the
    pilot's Windows box (onnxruntime-directml installed, no CUDA) it lands on
    CPU with the iGPU idle — the exact failure W2 exists to end. After the
    original runs, a CPU-only session on a box offering a preferred provider
    is rebuilt with that provider in front. Anything unexpected leaves the
    original session standing: worst case is today's CPU speed, never a
    broken detector.
    """
    global _PATCHED
    if _PATCHED:
        return
    from ultralytics.nn.backends import onnx as _onnx_backend

    original = _onnx_backend.ONNXBackend.load_model

    def load_model(self, weight, *args, **kwargs):
        out = original(self, weight, *args, **kwargs)
        _ensure_preferred_provider(self, weight)
        return out

    _onnx_backend.ONNXBackend.load_model = load_model
    _PATCHED = True


# Threads the CPU detector session may use. 0 = onnxruntime's default, every
# core. On the pilot server the detector ran back to back on all five vCPUs
# and the wall, the relay and the verifier got the scraps (14 Oct 2026):
# run_site sets this from the site file or detector_threads_for().
DETECTOR_THREADS = 0


def detector_threads_for(site: dict | None, device: str) -> int:
    """How many threads the detector gets. `detector_threads` in the site
    file wins; otherwise a CPU-only box with three or more cores keeps one
    core free for everything that is not detection; a GPU box is unlimited."""
    import os
    explicit = (site or {}).get("detector_threads") if isinstance(site, dict) else None
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit > 0:
        return explicit
    if device != "cpu":
        return 0
    cores = os.cpu_count() or 1
    return max(1, cores - 1) if cores >= 3 else cores


def _ensure_preferred_provider(backend_instance, weight) -> bool:
    """Rebuild a CPU-only session on the box's preferred provider, or with
    the thread cap when there is no better provider. True when a rebuild
    happened. Split out of the wrapper so the seam is testable without
    constructing a real ultralytics backend. Anything unexpected leaves the
    original session standing: worst case is today's CPU speed, never a
    broken detector."""
    try:
        session = getattr(backend_instance, "session", None)
        provider = best_available_provider()
        if session is None or session.get_providers()[0] != "CPUExecutionProvider":
            return False
        import onnxruntime
        if provider is not None:
            backend_instance.session = onnxruntime.InferenceSession(
                str(weight), providers=[provider, "CPUExecutionProvider"])
            backend_instance.output_names = [
                x.name for x in backend_instance.session.get_outputs()]
            log.info("[detect] session rebuilt on %s (ultralytics chose CPU)", provider)
            return True
        if DETECTOR_THREADS > 0:
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = int(DETECTOR_THREADS)
            backend_instance.session = onnxruntime.InferenceSession(
                str(weight), options, providers=["CPUExecutionProvider"])
            backend_instance.output_names = [
                x.name for x in backend_instance.session.get_outputs()]
            log.info("[detect] CPU session rebuilt with %d thread(s) so the wall, relay "
                     "and verifier keep a core", DETECTOR_THREADS)
            return True
        return False
    except Exception:  # noqa: BLE001 - the original session keeps working
        log.warning("[detect] session rebuild failed — staying on the original session",
                    exc_info=True)
        return False


def select_detection_weights(weights: str, *, backend: str = "auto",
                             device: str = "") -> dict:
    """Which weights file detection should load, and why — the W2 decision.

    Returns {"weights", "backend", "provider", "reason"}. `backend`:
      auto   .onnx when torch would run on CPU and onnxruntime can do better
      onnx   force the ONNX path (bench/tests) — absence is still a fallback
      torch  today's path, untouched — the site file's kill switch

    `device` is the torch device detection WOULD use. The rule that keeps
    auto honest: ONNX exists to rescue CPU-bound boxes. A box where torch is
    already on an accelerator (mps on the dev Macs at 2.7x, cuda anywhere)
    keeps torch — switching those to ONNX is a downgrade, and on mps it walks
    straight into the CoreML EP the module docstring rules out.

    Never raises; the torch path is always a valid answer.
    """
    from cvti.detector.core import resolve_weights

    resolved = resolve_weights(weights)
    torch_answer = {"weights": resolved, "backend": "torch", "provider": None,
                    "reason": ""}
    if backend == "torch":
        torch_answer["reason"] = "detector_backend=torch (site file)"
        return torch_answer
    if backend == "auto" and device in ("mps", "cuda"):
        torch_answer["reason"] = f"torch already accelerated ({device})"
        return torch_answer

    ort = onnxruntime_or_none()
    if ort is None:
        torch_answer["reason"] = "onnxruntime not installed"
        return torch_answer

    onnx_path = Path(resolved).with_suffix(".onnx")
    if not onnx_path.exists():
        # The bundle may carry the .onnx even when the .pt resolved elsewhere.
        candidate = resolve_weights(str(Path(weights).with_suffix(".onnx")))
        onnx_path = Path(candidate)
    if not onnx_path.exists() or onnx_path.suffix != ".onnx":
        torch_answer["reason"] = f"no ONNX export beside {Path(resolved).name}"
        return torch_answer

    _patch_ultralytics_providers()
    provider = best_available_provider() or "CPUExecutionProvider"
    return {"weights": str(onnx_path), "backend": "onnx",
            "provider": provider, "reason": ""}


def load_detector(weights: str, *, backend: str = "auto", device: str = ""):
    """(YOLO model, info dict) — the one call the pipeline makes.

    A failure ANYWHERE on the ONNX path answers with torch and the reason in
    the info dict, which health then names. The floor is today's behaviour.
    """
    from ultralytics import YOLO

    choice = select_detection_weights(weights, backend=backend, device=device)
    if choice["backend"] == "onnx":
        try:
            model = YOLO(choice["weights"], task="detect")
            # YOLO() is lazy: the ONNX session only materializes at the first
            # predict. Force it NOW with one dummy frame, so a corrupt export
            # fails here — where the torch fallback is — instead of taking the
            # engine down at its first live batch. Doubles as session warmup.
            import numpy as np
            model.predict(np.zeros((64, 64, 3), dtype=np.uint8),
                          imgsz=640, conf=0.9, verbose=False)
            return model, choice
        except Exception as exc:  # noqa: BLE001 - the export may be corrupt; the .pt is not
            log.warning("[detect] ONNX load failed (%s) — falling back to torch",
                        str(exc)[:120])
            choice = {"weights": None, "backend": "torch", "provider": None,
                      "reason": f"ONNX load failed: {str(exc)[:80]}"}
    from cvti.detector.core import resolve_weights
    resolved = resolve_weights(weights)
    model = YOLO(resolved)
    info = {"weights": resolved, "backend": "torch", "provider": None,
            "reason": choice.get("reason", "")}
    return model, info


def detector_health(info: dict, model=None) -> dict:
    """The health row: which backend detection is running, on what, and —
    when it is not the accelerator — why not. The Diagnose zip carries this,
    so 'is the GPU actually being used' stops being a guess."""
    doc = {"backend": info.get("backend"),
           "model": Path(info.get("weights") or "?").name,
           "provider": info.get("provider"),
           "reason": info.get("reason") or None}
    # The provider the session ACTUALLY came up on can differ from the ask
    # (a rebuild that failed, ORT quietly falling back) — report the truth.
    try:
        session = getattr(getattr(getattr(model, "predictor", None), "model", None),
                          "backend", None)
        session = getattr(session, "session", None)
        if session is not None:
            doc["provider"] = session.get_providers()[0]
    except Exception:  # noqa: BLE001 - a health probe must never hurt detection
        log.debug("provider introspection failed", exc_info=True)
    return doc


# Site-file kill switch value -> backend argument, tolerant of nonsense.
def backend_from_site(meta: dict) -> str:
    raw = str((meta or {}).get("detector_backend", "auto")).strip().lower()
    return raw if raw in ("auto", "onnx", "torch") else "auto"
