# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Argus — ONE bundle containing the whole product (EP-05-T1).

    python packaging/build.py            # builds with this spec
    pyinstaller packaging/argus.spec     # same thing, by hand

Produces (onedir):
    dist/Argus/            Windows / Linux
    dist/Argus.app         macOS

Two executables ship side by side in the same bundle:

    Argus          the operator console (PyQt6/QtWebEngine GUI shell)
    argus-engine   the detection pipeline (YOLO + VideoMAE + the TrueSight gate)

The console finds and launches argus-engine next to its own executable
(console_backend._bundled_engine), so "Start monitoring" works on a machine
with no Python installed. This replaces the earlier lean viewer-only spec —
the audit's words for that artifact were "a complete product failure at first
contact": an installer whose intelligence did not come with it.

What is bundled vs. pulled on first run:
    bundled     detector weights (models/*.pt, the VideoMAE fine-tune if built),
                configs/prompts/schemas, the web UI, the Ollama runtime if
                scripts/fetch_ollama.* has been run
    first run   the TrueSight verifier model (~3.3 GB) — downloaded in-app with
                progress, and Ollama resumes partial pulls natively

PyInstaller does not cross-compile: each OS builds its own artifact
(.github/workflows/build-app.yml runs the three-OS matrix on tags).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# The real version comes from the release tag (Actions sets GITHUB_REF_NAME);
# a hardcoded "1.0.0" shipped in every bundle's plist while the product was on
# v1.6.x (pilot log, 1 Sep). Local builds say dev, loudly.
_ref = os.environ.get("GITHUB_REF_NAME", "")
APP_VERSION = _ref[1:] if _ref.startswith("v") else \
    os.environ.get("ARGUS_VERSION", "0.0.0-dev")


def _tree(src_rel, dest_rel):
    """Recursively collect a directory into datas, if it exists."""
    out = []
    src = os.path.join(ROOT, src_rel)
    for base, _dirs, files in os.walk(src):
        for f in files:
            full = os.path.join(base, f)
            rel_dir = os.path.relpath(base, src)
            dest = dest_rel if rel_dir == "." else os.path.join(dest_rel, rel_dir)
            out.append((full, dest))
    return out


# ---------------------------------------------------------------------------
# Shared data files — bundled once, visible to both executables.
# ---------------------------------------------------------------------------
datas = [(os.path.join(ROOT, "cvti", "app", "web"), "cvti/app/web")]

# Bake the version into the bundle so the app can SAY which build it is —
# the sidebar claimed 'v0.9' from a hardcoded string forever.
_version_file = os.path.join(ROOT, "build", "VERSION")
os.makedirs(os.path.dirname(_version_file), exist_ok=True)
with open(_version_file, "w") as _vf:
    _vf.write(APP_VERSION)
datas.append((_version_file, "."))

# models/*.onnx: the W2 exports (scripts/export_onnx.py, run by CI before this
# spec). The .pt files STILL ship — the ONNX path falls back to torch when
# anything is off, and a fallback that is not in the bundle is a lie.
for pattern, dest in (("models/*.pt", "models"), ("models/*.onnx", "models"),
                      ("configs/*.json", "configs"),
                      # rules and zones the app's built-in feeds point at
                      # (Live EarthCams: configs/rules|zones/live_watch.json);
                      # absent, the engine started without them or not at all.
                      ("configs/rules/*.json", "configs/rules"),
                      ("configs/zones/*.json", "configs/zones"),
                      ("configs/*.yaml", "configs"), ("prompts/*.txt", "prompts"),
                      ("schemas/*.json", "schemas")):
    import glob as _glob
    for f in _glob.glob(os.path.join(ROOT, pattern)):
        datas.append((f, dest))

# VideoMAE fine-tune (fights/falls) — large, and only present on a machine
# that has trained or copied it. Bundle when there; the engine degrades
# loudly (logs, /health) when absent.
if os.path.isdir(os.path.join(ROOT, "runs", "video_finetune", "videomae")):
    datas += _tree("runs/video_finetune/videomae", "runs/video_finetune/videomae")

# The Ollama runtime (~binary + GPU runners), if a build script fetched it.
# Model weights are NOT here — they pull on first run into the user data dir.
_plat = {"win32": "windows", "darwin": "darwin"}.get(sys.platform, "linux")
if os.path.isdir(os.path.join(ROOT, "vendor", "ollama", _plat)):
    datas += _tree(f"vendor/ollama/{_plat}", f"vendor/ollama/{_plat}")

# The go2rtc stream gateway (W1), if scripts/fetch_go2rtc.* has run. One ~9 MB
# static binary; the engine degrades to direct decode without it, so its
# absence never breaks a build — only removes the gateway from that bundle.
if os.path.isdir(os.path.join(ROOT, "vendor", "go2rtc", _plat)):
    datas += _tree(f"vendor/go2rtc/{_plat}", f"vendor/go2rtc/{_plat}")
if os.path.isdir(os.path.join(ROOT, "vendor", "clip")):
    datas += _tree("vendor/clip", "vendor/clip")


# Self-contained playback demo (clips + recorded alerts), if built
# (packaging/build_demo_data.py) — lets the app demo itself anywhere.
if os.path.isdir(os.path.join(ROOT, "packaging", "demo_data")):
    datas += _tree("packaging/demo_data", "demo_data")
# The weapon detector loads through torch.hub from this vendored repo; without
# it every install failed with "No module named 'hubconf'" (pilot, 29 Aug) —
# tracked in git all along, never bundled.
if os.path.isdir(os.path.join(ROOT, "external", "yolov5")):
    datas += _tree("external/yolov5", "external/yolov5")


# ---------------------------------------------------------------------------
# The console (GUI shell). Keeps the detection stack out of ITS import graph;
# the shared COLLECT below still carries the engine's libraries.
# ---------------------------------------------------------------------------
app_a = Analysis(
    [os.path.join(ROOT, "cvti", "app", "shell.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=[
        "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebEngineCore", "PyQt6.QtWebChannel",
        "PyQt6.QtNetwork", "PyQt6.QtPrintSupport", "PyQt6.QtQml", "PyQt6.QtQuick",
        "PyQt6.QtGui",
        # Imported inside the live-feed resolver, so PyInstaller cannot see it
        # statically. Without it, Live EarthCams exists as a button and fails
        # as a feature on every install.
        "yt_dlp",
        # Stdlib, but only imported dynamically (the yolov5 weapon-model
        # loader) — the frozen engine failed with "No module named
        # 'logging.config'" and the weapons detector was dead on every
        # install, cascading per-frame errors on any camera with weapons
        # enabled (pilot health panel, 30 Aug: detector.cam1 480 errors).
        "logging.config", "logging.handlers",
    ],
    excludes=["torch", "torchvision", "ultralytics", "transformers", "pytorchvideo",
              "matplotlib", "scipy", "pandas", "tkinter", "polars",
              "PyQt5", "PySide6", "PySide2", "IPython", "pytest", "notebook"],
    noarchive=False,
)

# ---------------------------------------------------------------------------
# The engine. Everything the pipeline imports lazily or via strings.
# ---------------------------------------------------------------------------
# onnxruntime is imported LAZILY inside ultralytics' autobackend, so
# PyInstaller's static analysis never sees it — the 1.8.8 Windows bundle
# shipped WITHOUT it, the engine could not load the W2 ONNX path, and
# ultralytics' AutoUpdate tried to pip-install into the frozen app at every
# start (field diagnostics, 10 Sep). collect_all drags the package AND its
# native DLLs (incl. the DirectML provider on Windows) into the bundle.
from PyInstaller.utils.hooks import collect_all as _collect_all
_ort_datas, _ort_bins, _ort_hidden = _collect_all("onnxruntime")
# The W3 open-vocab pair (offline object rules): the clip PACKAGE carries a
# BPE vocab data file plain hiddenimports would drop — collect_all or the
# text encoder dies at first tokenize. Its ViT-B/32 checkpoint rides in
# vendor/clip (CI fetches it) and engine_entry pre-seeds ~/.cache/clip so
# ultralytics' parameterless clip.load() finds it with no network.
try:
    _clip_datas, _clip_bins, _clip_hidden = _collect_all("clip")
except Exception:                              # dev box without the pkg
    _clip_datas, _clip_bins, _clip_hidden = [], [], []
_ort_datas += _clip_datas
_ort_bins += _clip_bins
_ort_hidden += _clip_hidden

engine_a = Analysis(
    [os.path.join(ROOT, "packaging", "engine_entry.py")],
    pathex=[ROOT],
    datas=_ort_datas,              # shared datas ride with the app Analysis
    binaries=_ort_bins,
    hiddenimports=_ort_hidden + [
        # Static ffmpeg for H.264 evidence clips. The headless OpenCV wheel
        # has no H.264 encoder on Windows; without this the app's player
        # got mp4v it cannot decode (21 Sep). hooks-contrib collects the binary.
        "imageio_ffmpeg",
        "cvti.serving.pipeline",
        # Imported inside functions (pipeline starts the PPE worker only for
        # cameras with a `ppe` block; camera.py builds the motion tracker on
        # demand). PyInstaller's analysis does follow function-level imports,
        # but a name here costs nothing and a miss costs a field release.
        "cvti.ppe", "cvti.ppe.scanner", "cvti.ppe.assess", "cvti.ppe.policy",
        "cvti.detector.person_motion", "cvti.detector.openvocab",
        # ultralytics internals reached by name
        "ultralytics", "ultralytics.models.yolo", "ultralytics.models.yolo.detect",
        "ultralytics.models.yolo.classify", "ultralytics.models.yolo.pose",
        "ultralytics.nn.tasks", "ultralytics.data.loaders",
        "ultralytics.utils", "ultralytics.utils.plotting",
        "ultralytics.trackers.byte_tracker", "ultralytics.trackers.utils",
        # VideoMAE
        "transformers", "torch", "torchvision",
        # ByteTrack assignment + image plumbing
        "scipy.optimize", "scipy.spatial", "scipy.linalg",
        "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont", "PIL.ImageOps",
        "cv2",
        # Stdlib, but only imported dynamically (yolov5's utils/general.py
        # does `import logging.config` at weapon-model load). The 30 Aug fix
        # added these to the APP's hiddenimports — but the weapons loader
        # runs in THIS process, so the frozen engine still died with
        # "No module named 'logging.config'" (pilot screenshot, 3 Sep) and
        # the weapons detector was disabled on every Windows install.
        "logging.config", "logging.handlers",
        # transformers alone doesn't guarantee its model modules travel;
        # name the VideoMAE ones the video-action loader actually touches.
        "transformers.models.videomae", "safetensors",
        # SigLIP (object recognition): loaded by the smoke test and object watch.
        "transformers.models.siglip", "cvti.object_watch", "cvti.object_watch.smoke",
        "cvti.object_watch.embeddings", "cvti.object_watch.runtime_config",
    ],
    # polars rides in via an optional pandas/arrow path nothing here uses:
    # 156 MB of a customer's download for a dependency the product never calls.
    excludes=["PyQt6", "PyQt5", "PySide6", "PySide2", "tkinter", "polars",
              "IPython", "pytest", "notebook", "matplotlib.backends.backend_qtagg"],
    noarchive=False,
)

# ---------------------------------------------------------------------------
# The Engine API — the backend the Electron desktop UI spawns and talks to.
# An installed customer has no repo and no venv, so `python -m cvti.api` (the
# dev path) cannot exist on their machine; the shell spawns THIS binary.
# Deliberately not the detection stack: this process serves HTTP, reads the
# event store and starts/stops the engine, which is its own executable.
# ---------------------------------------------------------------------------
api_a = Analysis(
    [os.path.join(ROOT, "packaging", "api_entry.py")],
    pathex=[ROOT],
    datas=[],                      # shared datas ride with the app Analysis
    hiddenimports=[
        "cvti.api", "cvti.api.app", "cvti.api.sources", "cvti.api.tokens",
        "cvti.api.writes", "cvti.app.console_backend",
        # uvicorn resolves its loop and protocol implementations by STRING
        # name at startup ("auto" becomes one of these), so static analysis
        # sees none of them and a frozen API dies on its first request.
        "uvicorn", "uvicorn.config", "uvicorn.main", "uvicorn.server",
        "uvicorn.loops", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
        "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.protocols.websockets.wsproto_impl",
        "uvicorn.lifespan", "uvicorn.lifespan.on", "uvicorn.lifespan.off",
        "uvicorn.logging",
        "fastapi", "starlette", "websockets", "wsproto", "h11", "httptools",
        "anyio", "sniffio",
        # Dynamically imported, same as the app and engine need them.
        "logging.config", "logging.handlers",
    ],
    # The API never runs a detector; keeping the heavy stack out of ITS import
    # graph keeps the process small and its start fast. The libraries still
    # ship — the engine's Analysis carries them into the shared COLLECT below.
    excludes=["PyQt6", "PyQt5", "PySide6", "PySide2", "tkinter", "polars",
              "IPython", "pytest", "notebook", "torch", "torchvision",
              "ultralytics", "transformers", "pytorchvideo", "matplotlib"],
    noarchive=False,
)

app_pyz = PYZ(app_a.pure)
engine_pyz = PYZ(engine_a.pure)
api_pyz = PYZ(api_a.pure)

app_exe = EXE(
    app_pyz,
    app_a.scripts,
    [],
    exclude_binaries=True,
    name="Argus",
    debug=False,
    strip=False,
    upx=False,
    console=False,               # windowed GUI app (no terminal window)
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

engine_exe = EXE(
    engine_pyz,
    engine_a.scripts,
    [],
    exclude_binaries=True,
    name="argus-engine",
    debug=False,
    strip=False,
    upx=False,
    console=True,                # headless subprocess; stdout goes to monitor.log
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

api_exe = EXE(
    api_pyz,
    api_a.scripts,
    [],
    exclude_binaries=True,
    name="argus-api",
    debug=False,
    strip=False,
    upx=False,
    console=True,                # headless subprocess; the shell pipes its output
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# One COLLECT: all three executables share one set of libraries and data files.
def _strip_dead_weight(datas):
    """Drop what a customer downloads but never uses (bundle audit, 25 Aug):
    QtWebEngine's devtools DEBUG resources are 76 MB of symbols for a devtools
    panel the app never opens."""
    drop = ("qtwebengine_devtools_resources.debug.pak",)
    return [d for d in datas if not str(d[0]).endswith(drop)]


app_a.datas = _strip_dead_weight(app_a.datas)
engine_a.datas = _strip_dead_weight(engine_a.datas)

coll = COLLECT(
    app_exe,
    engine_exe,
    api_exe,
    app_a.binaries,
    app_a.datas,
    engine_a.binaries,
    engine_a.datas,
    api_a.binaries,
    api_a.datas,
    strip=False,
    upx=False,
    name="Argus",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Argus.app",
        icon=None,
        bundle_identifier="com.argus.desktop",
        info_plist={
            # PyInstaller stamps LSBackgroundOnly=true because the bundled
            # engine EXE is console=True — and a background-only app can NEVER
            # take keyboard focus, so every keystroke fell through to whatever
            # window was behind Argus (field report, 31 Aug: "I am typing into
            # the app but it's typing here"). Force it off.
            "LSBackgroundOnly": False,
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "NSLocalNetworkUsageDescription":
                "Argus connects to the security cameras on your local network.",
        },
    )
