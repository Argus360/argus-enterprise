from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from cvti.verification import vlm_slot

import cv2
import numpy as np
from cvti.contracts import LOCAL_VLM_MODEL
from cvti.logging_setup import get_logger

log = get_logger(__name__)


# --- The VLM request budget (latency audit 1 Sep, V4a) ----------------------
# gemma3's vision tower works on 896x896 tiles: a 1080p or 4K frame sent whole
# is downscaled server-side anyway, after we paid to base64 and ship megabytes
# per call — and on multi-frame verifies that cost repeats per frame. Shrink
# on OUR side of the wire: long edge <=896, JPEG q80. Detection never sees
# these frames; only what travels to the model does.
VLM_MAX_SIDE = 896
VLM_JPEG_QUALITY = 80
# Output-side budget for the mapper's own scene-context JSON: the largest
# valid answer (description + zones + candidates) fits in well under 1024
# tokens; without a cap a rambling model generates until the context runs out
# at ~10 tok/s on pilot CPUs.
MAPPER_MAX_TOKENS = 1024

DEFAULT_OUTPUT_ROOT = Path("runs/context")
DEFAULT_PROMPT_PATH = Path("prompts/agent_mapper_prompt.txt")
DEFAULT_SCHEMA_PATH = Path("schemas/scene_context.schema.json")

ALLOWED_SOURCE_TYPES = {"webcam", "rtsp", "video_file", "image_file"}
ALLOWED_ENVIRONMENT_TYPES = {
    "estate_gate",
    "estate_street",
    "perimeter_fence",
    "retail_shop",
    "mall_corridor",
    "office_lobby",
    "office_floor",
    "parking_lot",
    "banking_hall",
    "atm_area",
    "warehouse_floor",
    "generator_area",
    "residential_interior",
    "residential_exterior",
    "unknown",
}
ALLOWED_ZONE_ROLES = {
    "entry",
    "exit",
    "transition",
    "safe",
    "restricted",
    "merchandise",
    "checkout",
    "parking",
    "perimeter",
    "asset",
    "unknown",
}


@dataclass
class SampledFrame:
    image: Any
    timestamp_seconds: float
    score: float


@dataclass
class MappingPaths:
    output_dir: Path
    frame_path: Path
    context_path: Path
    raw_response_path: Path


@dataclass
class MappingResult:
    context: dict[str, Any]
    selected_frame: Any
    raw_response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent Mapper for scene context generation from webcam, RTSP, image, or video input."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source camera index, RTSP URL, local image path, or local video path.",
    )
    parser.add_argument(
        "--camera-id",
        default="",
        help="Stable camera identifier used for output paths and scene_context metadata.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="How many frames to sample from a live or video source.",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=4.0,
        help="Seconds between samples for live or video sources when frame-accurate seeking is unavailable.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where scene context artifacts are written.",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        choices=("mock", "anthropic", "openai_compatible"),
        help="VLM provider. `mock` is useful for local testing without network/API keys.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Provider model name. If omitted, a provider-specific default is used.",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_PATH),
        help="Prompt template file used for VLM requests.",
    )
    parser.add_argument(
        "--schema-file",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Scene-context schema file used for validation reference.",
    )
    parser.add_argument(
        "--save-frame",
        action="store_true",
        help="Always save the selected representative frame. Enabled implicitly for normal runs.",
    )
    parser.add_argument(
        "--dump-raw-response",
        action="store_true",
        help="Save the raw provider response alongside the validated scene_context.json.",
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Optional environment-variable override for provider API key.",
    )
    parser.add_argument(
        "--api-base-url",
        default="",
        help="Optional base URL override for `openai_compatible` providers.",
    )
    parser.add_argument(
        "--max-zone-suggestions",
        type=int,
        default=4,
        help="Hint passed into the prompt for the maximum number of zone suggestions.",
    )
    return parser.parse_args()


def normalize_source(raw_source: str) -> int | str:
    return int(raw_source) if raw_source.isdigit() else raw_source


def detect_source_type(source: int | str) -> str:
    if isinstance(source, int):
        return "webcam"
    lowered = str(source).lower()
    if lowered.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    suffix = Path(lowered).suffix
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return "image_file"
    return "video_file"


def build_camera_id(raw_camera_id: str, source: int | str) -> str:
    if raw_camera_id.strip():
        return sanitize_name(raw_camera_id.strip())
    if isinstance(source, int):
        return f"webcam_{source}"
    return sanitize_name(Path(str(source)).stem or "source")


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "source"


def build_output_paths(output_root: Path, camera_id: str) -> MappingPaths:
    output_dir = output_root / camera_id
    return MappingPaths(
        output_dir=output_dir,
        frame_path=output_dir / "source_frame.jpg",
        context_path=output_dir / "scene_context.json",
        raw_response_path=output_dir / "raw_response.txt",
    )


def score_frame(frame: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Favor readable, moderately bright frames with detail.
    brightness_penalty = abs(brightness - 128.0) / 128.0
    detail_score = min(blur / 250.0, 4.0)
    visibility_floor = 0.0 if brightness < 15.0 else 0.5
    return visibility_floor + detail_score - brightness_penalty


def capture_sample_frames(
    source: int | str,
    source_type: str,
    sample_count: int,
    sample_interval_seconds: float,
) -> list[SampledFrame]:
    if source_type == "image_file":
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Unable to read image source: {source}")
        return [SampledFrame(image=frame, timestamp_seconds=0.0, score=score_frame(frame))]

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        from cvti.utils import redact_credentials
        # The source may carry rtsp credentials; this string ends up on the
        # System screen and in the Diagnose zip (pilot's password did, 4 Sep).
        raise RuntimeError(f"Unable to open source: {redact_credentials(str(source))}")

    try:
        if source_type == "video_file":
            return sample_video_frames(capture, sample_count, sample_interval_seconds)
        return sample_live_frames(capture, sample_count, sample_interval_seconds)
    finally:
        capture.release()


def sample_video_frames(
    capture: cv2.VideoCapture,
    sample_count: int,
    sample_interval_seconds: float,
) -> list[SampledFrame]:
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    samples: list[SampledFrame] = []

    if frame_total > 0 and fps > 0:
        target_indices = evenly_spaced_indices(frame_total, sample_count)
        for frame_index in target_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            ts = frame_index / fps
            samples.append(SampledFrame(image=frame, timestamp_seconds=ts, score=score_frame(frame)))
        if samples:
            return samples

    # Fallback for codecs/containers where seeking is unreliable.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return sample_sequential_frames(capture, sample_count, sample_interval_seconds)


def sample_live_frames(
    capture: cv2.VideoCapture,
    sample_count: int,
    sample_interval_seconds: float,
) -> list[SampledFrame]:
    return sample_sequential_frames(capture, sample_count, sample_interval_seconds)


def sample_sequential_frames(
    capture: cv2.VideoCapture,
    sample_count: int,
    sample_interval_seconds: float,
) -> list[SampledFrame]:
    samples: list[SampledFrame] = []
    for index in range(max(1, sample_count)):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        timestamp_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if timestamp_seconds <= 0.0:
            timestamp_seconds = index * max(sample_interval_seconds, 0.0)
        samples.append(SampledFrame(image=frame, timestamp_seconds=timestamp_seconds, score=score_frame(frame)))
        if index < sample_count - 1 and sample_interval_seconds > 0:
            time.sleep(sample_interval_seconds)
    return samples


def evenly_spaced_indices(frame_total: int, sample_count: int) -> list[int]:
    if frame_total <= 1:
        return [0]
    sample_count = max(1, sample_count)
    if sample_count == 1:
        return [frame_total // 2]
    start = max(0, int(frame_total * 0.15))
    end = max(start, int(frame_total * 0.85))
    return sorted({min(frame_total - 1, int(round(start + i * (end - start) / (sample_count - 1)))) for i in range(sample_count)})


def choose_representative_frame(samples: list[SampledFrame]) -> SampledFrame:
    if not samples:
        raise RuntimeError("No frames were captured from the source.")
    return max(samples, key=lambda sample: sample.score)


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(
    template: str,
    camera_id: str,
    source_type: str,
    source_frame_path: str,
    max_zone_suggestions: int,
    operator_hints: dict[str, Any] | None = None,
) -> str:
    # Imported lazily because hierarchy imports the camera-level environment
    # vocabulary from this module. Keeping one validator-owned source prevents
    # the prompt and accepted schema values from drifting apart.
    from cvti.scene.hierarchy import AREA_TYPES, SITE_TYPES

    dynamic_suffix = (
        f"\n\nCaller-provided metadata:\n"
        f'- camera_id: "{camera_id}"\n'
        f'- source_type: "{source_type}"\n'
        f'- source_frame_path: "{source_frame_path}"\n'
        f"- max_zone_suggestions: {max_zone_suggestions}\n"
        "Use the provided camera_id, source_type, and source_frame_path exactly in the JSON output.\n"
        f"Suggest no more than {max_zone_suggestions} zones.\n"
        "For site_type_candidate, use exactly one of: "
        f"{json.dumps(sorted(SITE_TYPES))}.\n"
        "For area_type_candidate, use exactly one of: "
        f"{json.dumps(sorted(AREA_TYPES))}."
    )
    # The operator's onboarding answers are PRIORS, not a bypass: one frame
    # can lie (empty forecourt at 3am reads as "parking lot"), but the person
    # who mounted the camera knows what it watches. Anchoring the model to
    # their answers is the hallucination reduction this mapper exists for.
    if operator_hints:
        lines = ["\n\nThe site operator described this camera during onboarding.",
                 "Treat these answers as fallible priors, not visual facts.",
                 "Make an independent visual observation for this camera; prefer",
                 "the prior unless the frame clearly contradicts it, and return",
                 "unknown when the image is insufficient."]
        site_type = str(operator_hints.get("site_type") or "").strip()
        if site_type and site_type != "unknown":
            lines.append(f'- site type prior: "{site_type}"')
        area_id = str(operator_hints.get("area_id") or "").strip()
        if area_id:
            lines.append(f'- operator area id: "{area_id}"')
        area_type = str(operator_hints.get("area_type") or "").strip()
        if area_type and area_type != "unknown":
            lines.append(f'- area type prior: "{area_type}"')
        environment = str(operator_hints.get("environment_type") or "").strip()
        if environment and environment != "unknown":
            lines.append(f'- environment: "{environment}"')
        actors = [str(a).strip() for a in operator_hints.get("expected_actors") or []]
        actors = [a for a in actors if a]
        if actors:
            lines.append(f"- people and vehicles that belong here: {', '.join(actors)}")
        note = str(operator_hints.get("note") or "").strip()
        if note:
            lines.append(f'- operator note: "{note}"')
        if len(lines) > 3:
            dynamic_suffix += "\n".join(lines)
    dynamic_suffix += (
        "\nReturn site_type_candidate, area_type_candidate, and view_description "
        "as independent visual observations in addition to the existing fields."
    )
    return template + dynamic_suffix


def downscale_for_vlm(frame: Any, max_side: int = VLM_MAX_SIDE) -> Any:
    """The frame as the VLM should receive it: long edge capped at `max_side`.

    Shrink-only (a small frame passes through untouched), aspect preserved,
    INTER_AREA — the correct filter for downscaling. Callers keep using the
    original frame for detection and evidence; this copy exists to travel.
    """
    height, width = frame.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return frame
    scale = max_side / float(long_side)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def encode_frame_as_jpeg_bytes(frame: Any, max_side: int = VLM_MAX_SIDE,
                               quality: int = VLM_JPEG_QUALITY) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", downscale_for_vlm(frame, max_side),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError("Failed to encode selected frame as JPEG.")
    return encoded.tobytes()


def call_provider(
    provider: str,
    prompt: str,
    frame_bytes: bytes,
    frame_shape: tuple[int, int, int],
    camera_id: str,
    source_type: str,
    model: str,
    api_key_env: str,
    api_base_url: str,
) -> str:
    if provider == "mock":
        return mock_scene_context_json(camera_id, source_type, frame_shape)
    if provider == "anthropic":
        return call_anthropic(
            prompt=prompt,
            frame_bytes=frame_bytes,
            model=model or "claude-3-5-sonnet-latest",
            api_key_env=api_key_env or "ANTHROPIC_API_KEY",
        )
    if provider == "openai_compatible":
        return call_openai_compatible(
            prompt=prompt,
            frame_bytes=frame_bytes,
            model=model or "gpt-4.1-mini",
            api_key_env=api_key_env or "OPENAI_API_KEY",
            api_base_url=api_base_url or "https://api.openai.com/v1",
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


def call_anthropic(prompt: str, frame_bytes: bytes, model: str, api_key_env: str) -> str:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable: {api_key_env}")

    payload = {
        "model": model,
        "max_tokens": 1200,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(frame_bytes).decode("ascii"),
                        },
                    },
                ],
            }
        ],
    }
    req = urlrequest.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urlerror.URLError as exc:
        raise RuntimeError(f"Anthropic request failed: {exc}") from exc

    content = body.get("content", [])
    text_chunks = [item.get("text", "") for item in content if item.get("type") == "text"]
    if not text_chunks:
        raise RuntimeError("Anthropic response did not contain any text content.")
    return "\n".join(text_chunks).strip()


def read_http_error_body(exc: urlerror.HTTPError) -> str:
    """Best-effort extraction of a human-readable message from an HTTP error body.

    urllib raises HTTPError without surfacing the response body, so a bare
    `HTTP 429` tells you nothing about *why*. OpenRouter (and most OpenAI-compatible
    backends) return a JSON `{"error": {"message": ...}}` that is the actual reason.
    """
    try:
        raw = exc.read().decode("utf-8", errors="replace").strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics only, never mask the original error
        log.debug("scene field extraction failed", exc_info=True)
        return ""
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        error = data.get("error", data)
        if isinstance(error, dict):
            return str(error.get("message") or raw)[:400]
        return str(error or raw)[:400]
    except (json.JSONDecodeError, AttributeError):
        return raw[:400]


def _post_chat_with_retries(base: str, headers: dict, payload: dict,
                            max_retries: int, timeout: float) -> dict[str, Any]:
    """POST /chat/completions with the retry/backoff policy; returns the body."""
    body: dict[str, Any] | None = None
    last_error: RuntimeError | None = None
    for attempt in range(max_retries + 1):
        req = urlrequest.Request(
            f"{base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            # Default 360s, not 180: the same pilot laptop that measured 90s
            # median verify latency produces ~180s VLM answers under load — and
            # a timeout AT the slowest machine's typical latency failed every
            # scene mapping on it ('cam1 failed: timed out', 1 Sep). Mapping
            # runs once per camera at startup; patience there is nearly free.
            # Callers on a live path (the gate) pass their own tighter,
            # latency-derived budget instead.
            with urlrequest.urlopen(req, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            # Free tiers sometimes return HTTP 200 with an empty/no-choices body under load —
            # treat that like a transient error and retry rather than failing the call.
            if not parsed.get("choices"):
                last_error = RuntimeError("OpenAI-compatible response did not contain choices.")
                if attempt < max_retries:
                    time.sleep(2.0 * (2**attempt))
                    continue
                raise last_error
            body = parsed
            break
        except urlerror.HTTPError as exc:
            detail = read_http_error_body(exc)
            last_error = RuntimeError(
                f"OpenAI-compatible request failed: HTTP {exc.code} {exc.reason}."
                + (f" {detail}" if detail else "")
            )
            # 429 (rate limit) and 5xx (transient upstream) are worth retrying;
            # 401/402/4xx are terminal (bad key, no credits) — fail fast.
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if retryable and attempt < max_retries:
                retry_after = exc.headers.get("retry-after") if exc.headers else None
                delay = (
                    float(retry_after)
                    if retry_after and str(retry_after).replace(".", "", 1).isdigit()
                    else 2.0 * (2**attempt)
                )
                time.sleep(min(delay, 30.0))
                continue
            raise last_error from exc
        except urlerror.URLError as exc:
            last_error = RuntimeError(f"OpenAI-compatible request failed: {exc}")
            if attempt < max_retries:
                time.sleep(2.0 * (2**attempt))
                continue
            raise last_error from exc

    if body is None:
        raise last_error or RuntimeError("OpenAI-compatible request failed after retries.")
    return body


def call_openai_compatible(
    prompt: str,
    frame_bytes: bytes | list[bytes],
    model: str,
    api_key_env: str,
    api_base_url: str,
    max_retries: int = 3,
    require_key: bool = True,
    max_tokens: int | None = None,
    timeout: float = 360.0,
    slot_mode: str = "wait",
) -> str:
    api_key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
    if require_key and not api_key:
        raise RuntimeError(f"Missing API key in environment variable: {api_key_env}")

    # frame_bytes may be a single JPEG or a list of JPEGs (multi-frame verification).
    frames = frame_bytes if isinstance(frame_bytes, list) else [frame_bytes]
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for fb in frames:
        url = f"data:image/jpeg;base64,{base64.b64encode(fb).decode('ascii')}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    # Output budget (latency audit 1 Sep, V1). Every caller knows the shape of
    # the answer it needs — a verdict JSON, a threat list, a scene context —
    # and none of them needs an unbounded one. On a CPU box generating ~10
    # tok/s, an uncapped answer that rambles for 800 tokens IS the missing 80
    # seconds of gate latency. Ollama's /v1 endpoint maps this to num_predict.
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if "api.groq.com" in api_base_url:
        # Groq's Qwen models think out loud before answering, and the
        # thinking counts against max_tokens: with a 96-token verdict budget
        # the JSON never arrived and the gate logged "Unterminated JSON"
        # (CHI daily self-test, 18 and 19 Oct 2026). Ask for the answer only.
        payload["reasoning_format"] = "hidden"
        payload["reasoning_effort"] = "none"
    base = api_base_url.rstrip("/")
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
        # OpenRouter uses these for routing/attribution; ignored by other backends.
        "http-referer": "https://github.com/DEMILADE07/cv-threat-intelligence",
        "x-title": "CV Threat Intelligence Agent Mapper",
        # urllib's default agent is refused by some CDN-fronted clouds.
        "user-agent": "argus-engine/1",
    }

    # Every local VLM caller in the process funnels through here, which makes
    # this the one place serialization can be enforced: concurrent vision
    # requests on a CPU box make every one of them miss its deadline (the 11
    # Sep pilot collapse — verify busy-fraction 1.49, zero verdicts ever).
    # slot_mode="skip" lets periodic callers yield instead of piling on.
    # Cloud endpoints are not serialized; their parallelism is not ours.
    if vlm_slot.is_local(api_base_url):
        with vlm_slot.slot(slot_mode, who=model):
            body = _post_chat_with_retries(base, headers, payload, max_retries, timeout)
    else:
        body = _post_chat_with_retries(base, headers, payload, max_retries, timeout)

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI-compatible response did not contain choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        content = "\n".join(text_parts)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenAI-compatible response did not contain text content.")
    return content.strip()


def mock_scene_context_json(camera_id: str, source_type: str, frame_shape: tuple[int, int, int]) -> str:
    height, width = frame_shape[:2]
    environment = "unknown"
    description = "Camera view with outdoor or indoor scene not yet classified by a live VLM provider."
    lowered = camera_id.lower()

    if any(token in lowered for token in ("estate", "gate")):
        environment = "estate_gate"
        description = "Estate-style entrance scene with gate-like access area visible."
    elif any(token in lowered for token in ("shop", "retail", "mall")):
        environment = "retail_shop"
        description = "Retail-like scene with customer-facing floor space and likely merchandise areas."
    elif any(token in lowered for token in ("office", "lobby")):
        environment = "office_lobby"
        description = "Office-like entry or lobby scene with managed access characteristics."

    mock = {
        "camera_id": camera_id,
        "source_type": source_type,
        "environment_type": environment,
        "scene_description": description,
        "expected_actors": ["people", "vehicles"] if environment == "estate_gate" else ["people", "staff", "visitors"],
        "zones": [
            {
                "id": "center_zone",
                "label": "center_zone",
                "role": "transition",
                "bbox": [width // 4, height // 4, (width * 3) // 4, (height * 3) // 4],
            }
        ],
        "confidence": 0.35,
        "generated_at": utc_now_iso(),
        "source_frame_path": f"runs/context/{camera_id}/source_frame.jpg",
        "notes": "Mock provider output for local testing. Replace with a live VLM provider for real scene understanding.",
    }
    return json.dumps(mock, indent=2)


def extract_first_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    first = stripped.find("{")
    if first < 0:
        raise RuntimeError("Provider response did not contain a JSON object.")

    depth = 0
    for index in range(first, len(stripped)):
        char = stripped[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[first : index + 1]
    raise RuntimeError("Provider response contained an unterminated JSON object.")


def parse_and_validate_scene_context(
    raw_response: str,
    schema: dict[str, Any],
    camera_id: str,
    source_type: str,
    source_frame_path: str,
) -> dict[str, Any]:
    _ = schema  # schema file is loaded for contract parity and future extension.
    payload = json.loads(extract_first_json_object(raw_response))

    if not isinstance(payload, dict):
        raise RuntimeError("Scene context payload must be a JSON object.")

    # These four are facts the caller owns — never let the VLM set or override them.
    # (A VLM will happily hallucinate a timestamp; we stamp the real one here.)
    payload["camera_id"] = camera_id
    payload["source_type"] = source_type
    payload["source_frame_path"] = source_frame_path
    payload["generated_at"] = utc_now_iso()

    environment_type = str(payload.get("environment_type", "unknown"))
    if environment_type not in ALLOWED_ENVIRONMENT_TYPES:
        environment_type = "unknown"
    payload["environment_type"] = environment_type

    payload["scene_description"] = str(payload.get("scene_description", "")).strip() or "No scene description provided."
    payload["expected_actors"] = normalize_string_list(payload.get("expected_actors"))
    payload["zones"] = normalize_zones(payload.get("zones"))
    payload["confidence"] = normalize_confidence(payload.get("confidence"))
    payload["notes"] = str(payload.get("notes", "")).strip()

    from cvti.scene.hierarchy import normalize_area_type, normalize_site_type

    payload["site_type_candidate"] = normalize_site_type(
        payload.get("site_type_candidate")
    )
    payload["area_type_candidate"] = normalize_area_type(
        payload.get("area_type_candidate")
    )
    payload["view_description"] = str(
        payload.get("view_description", "")
    ).strip()

    payload.pop("risk_hints", None)
    payload.pop("suggested_preset", None)

    if payload["source_type"] not in ALLOWED_SOURCE_TYPES:
        raise RuntimeError(f"Invalid source_type in scene context: {payload['source_type']}")
    ensure_iso_timestamp(payload["generated_at"])
    return payload


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if text not in seen:
            result.append(text)
            seen.add(text)
    return result


def normalize_zones(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        zone_id = str(item.get("id", "")).strip() or "zone"
        label = str(item.get("label", "")).strip() or zone_id
        role = str(item.get("role", "unknown")).strip()
        if role not in ALLOWED_ZONE_ROLES:
            role = "unknown"
        bbox = item.get("bbox", [])
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            normalized_bbox = [max(0, int(number)) for number in bbox]
        except (TypeError, ValueError):
            continue
        normalized.append(
            {
                "id": zone_id,
                "label": label,
                "role": role,
                "bbox": normalized_bbox,
            }
        )
    return normalized


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def ensure_iso_timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"Invalid generated_at timestamp: {value}") from exc


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def save_outputs(
    paths: MappingPaths,
    frame: Any,
    context: dict[str, Any],
    raw_response: str,
    save_frame: bool,
    dump_raw_response: bool,
) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if save_frame:
        cv2.imwrite(str(paths.frame_path), frame)
    paths.context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    if dump_raw_response:
        paths.raw_response_path.write_text(raw_response, encoding="utf-8")


class AgentMapper:
    """Object API over the scene-mapping functions, used by the desktop app.

    Providers: "mock", "anthropic", and "local"/"ollama" (an Ollama server on its
    OpenAI-compatible endpoint, no API key required).
    """

    _DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-6",
        "local": LOCAL_VLM_MODEL,
        "ollama": LOCAL_VLM_MODEL,
        "openai_compatible": "gpt-4.1-mini",
    }

    def __init__(
        self,
        provider: str = "mock",
        model: str = "",
        api_key_env: str = "",
        base_url: str = "",
        prompt_path: Path | None = None,
        schema_path: Path | None = None,
        max_zone_suggestions: int = 4,
    ) -> None:
        self.provider = provider
        self.model = model or self._DEFAULT_MODELS.get(provider, "")
        self.api_key_env = api_key_env
        self.base_url = base_url or (
            "http://localhost:11434/v1" if provider in ("local", "ollama") else ""
        )
        self.max_zone_suggestions = max_zone_suggestions
        # Resolve bundled defaults so this works from a frozen app too.
        try:
            from cvti.utils import resource_path
            self._prompt_path = prompt_path or resource_path(str(DEFAULT_PROMPT_PATH))
            self._schema_path = schema_path or resource_path(str(DEFAULT_SCHEMA_PATH))
        except Exception as exc:  # noqa: BLE001 - fall back to CWD-relative defaults in dev
            log.debug("prompt path unreadable; using the default", exc_info=True)
            self._prompt_path = prompt_path or DEFAULT_PROMPT_PATH
            self._schema_path = schema_path or DEFAULT_SCHEMA_PATH

    def map_result(
        self,
        source_raw: str,
        camera_id: str = "",
        sample_count: int = 3,
        source_frame_path: str = "",
        operator_hints: dict[str, Any] | None = None,
        frames: list | None = None,
    ) -> MappingResult:
        source = normalize_source(str(source_raw))
        source_type = detect_source_type(source)
        cam = build_camera_id(camera_id, source)
        frame_path = source_frame_path or f"runs/context/{cam}/source_frame.jpg"

        if frames:
            # Caller-supplied frames (the engine's own decoded view): no
            # second capture session against a camera that may cap them.
            samples = [SampledFrame(image=f, timestamp_seconds=0.0,
                                    score=score_frame(f)) for f in frames]
        else:
            samples = capture_sample_frames(source, source_type, max(1, sample_count), 2.0)
        selected = choose_representative_frame(samples)
        frame_bytes = encode_frame_as_jpeg_bytes(selected.image)

        prompt = build_prompt(
            template=load_text_file(Path(self._prompt_path)),
            camera_id=cam,
            source_type=source_type,
            source_frame_path=frame_path,
            max_zone_suggestions=self.max_zone_suggestions,
            operator_hints=operator_hints,
        )

        if self.provider == "mock":
            raw = mock_scene_context_json(cam, source_type, selected.image.shape)
        elif self.provider == "anthropic":
            raw = call_anthropic(
                prompt=prompt, frame_bytes=frame_bytes,
                model=self.model or "claude-sonnet-4-6",
                api_key_env=self.api_key_env or "ANTHROPIC_API_KEY",
            )
        elif self.provider in ("local", "ollama"):
            raw = call_openai_compatible(
                prompt=prompt, frame_bytes=frame_bytes,
                model=self.model or LOCAL_VLM_MODEL,
                api_key_env="", api_base_url=self.base_url,
                require_key=False,
                max_tokens=MAPPER_MAX_TOKENS,
            )
        elif self.provider == "openai_compatible":
            raw = call_openai_compatible(
                prompt=prompt, frame_bytes=frame_bytes,
                model=self.model or "gpt-4.1-mini",
                api_key_env=self.api_key_env or "OPENAI_API_KEY",
                api_base_url=self.base_url or "https://api.openai.com/v1",
                max_tokens=MAPPER_MAX_TOKENS,
            )
        else:
            raise RuntimeError(f"Unsupported provider: {self.provider}")

        context = parse_and_validate_scene_context(
            raw_response=raw,
            schema=load_schema(Path(self._schema_path)),
            camera_id=cam,
            source_type=source_type,
            source_frame_path=frame_path,
        )
        area_id = str((operator_hints or {}).get("area_id") or "").strip()
        if area_id:
            context["area_id"] = area_id
        return MappingResult(
            context=context,
            selected_frame=selected.image,
            raw_response=raw,
        )

    def map(self, source_raw: str, camera_id: str = "", sample_count: int = 3) -> dict[str, Any]:
        return self.map_result(source_raw, camera_id, sample_count).context


def main() -> None:
    # Entrypoint: configure logging before anything can fail.
    from cvti.logging_setup import setup_logging
    setup_logging(component="argus-mapper")
    args = parse_args()
    source = normalize_source(args.source)
    source_type = detect_source_type(source)
    camera_id = build_camera_id(args.camera_id, source)
    output_root = Path(args.output_dir)
    paths = build_output_paths(output_root, camera_id)

    prompt_template = load_text_file(Path(args.prompt_file))
    schema = load_schema(Path(args.schema_file))

    samples = capture_sample_frames(
        source=source,
        source_type=source_type,
        sample_count=max(1, args.sample_count),
        sample_interval_seconds=max(0.0, args.sample_interval_seconds),
    )
    selected = choose_representative_frame(samples)
    frame_bytes = encode_frame_as_jpeg_bytes(selected.image)

    prompt = build_prompt(
        template=prompt_template,
        camera_id=camera_id,
        source_type=source_type,
        source_frame_path=str(paths.frame_path).replace("\\", "/"),
        max_zone_suggestions=max(1, args.max_zone_suggestions),
    )
    raw_response = call_provider(
        provider=args.provider,
        prompt=prompt,
        frame_bytes=frame_bytes,
        frame_shape=selected.image.shape,
        camera_id=camera_id,
        source_type=source_type,
        model=args.model,
        api_key_env=args.api_key_env,
        api_base_url=args.api_base_url,
    )
    context = parse_and_validate_scene_context(
        raw_response=raw_response,
        schema=schema,
        camera_id=camera_id,
        source_type=source_type,
        source_frame_path=str(paths.frame_path).replace("\\", "/"),
    )
    save_outputs(
        paths=paths,
        frame=selected.image,
        context=context,
        raw_response=raw_response,
        save_frame=True or args.save_frame,
        dump_raw_response=args.dump_raw_response,
    )

    log.info(f"Selected frame timestamp: {selected.timestamp_seconds:.2f}s")
    log.info(f"Scene context saved to: {paths.context_path}")
    log.info(f"Representative frame saved to: {paths.frame_path}")
    log.info("Summary: "
        f"environment_type={context['environment_type']}, "
        f"confidence={context['confidence']:.2f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
