"""The VLM request budget (latency audit 1 Sep: V1, V4a, F6).

Three promises, each measured on a pilot CPU box before being written down:

- OUTPUT is capped: every local VLM call names the most tokens its answer can
  need. Uncapped generation at ~10 tok/s was the dominant share of the 90s
  median verify latency.
- INPUT is shrunk: frames travel at <=896 long edge, JPEG q80 — the model's
  vision tower tiles at 896 anyway, so full-res 1080p/4K uploads bought
  nothing but base64 megabytes per call (times N on multi-frame verifies).
- ONE model tag: the mapper, the gate, the pull, and the eval all name
  gemma3:4b. The 4b / 4b-it-qat split made a default-configured mapper
  request a model nobody had pulled.
"""
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.contracts import LOCAL_VLM_MODEL, CandidateAlert


def _frame(width: int, height: int):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _decoded_shape(jpeg_bytes: bytes):
    import cv2
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img.shape[:2]  # (h, w)


class DownscaleTests(unittest.TestCase):
    def test_landscape_1080p_shrinks_to_896_long_edge(self):
        from cvti.scene.agent_mapper import downscale_for_vlm
        out = downscale_for_vlm(_frame(1920, 1080))
        h, w = out.shape[:2]
        self.assertEqual(w, 896)
        self.assertEqual(h, 504)  # aspect preserved: 1080 * (896/1920)

    def test_portrait_frames_shrink_on_their_long_edge(self):
        from cvti.scene.agent_mapper import downscale_for_vlm
        out = downscale_for_vlm(_frame(1080, 1920))
        h, w = out.shape[:2]
        self.assertEqual(h, 896)
        self.assertEqual(w, 504)

    def test_small_frames_pass_through_untouched(self):
        from cvti.scene.agent_mapper import downscale_for_vlm
        frame = _frame(640, 480)
        self.assertIs(downscale_for_vlm(frame), frame)  # no copy, no resize

    def test_mapper_encode_ships_a_downscaled_jpeg(self):
        from cvti.scene.agent_mapper import encode_frame_as_jpeg_bytes
        h, w = _decoded_shape(encode_frame_as_jpeg_bytes(_frame(3840, 2160)))
        self.assertLessEqual(max(h, w), 896)

    def test_gate_encode_ships_a_downscaled_jpeg(self):
        from cvti.verification.gate import _encode_frame
        h, w = _decoded_shape(_encode_frame(_frame(1920, 1080)))
        self.assertLessEqual(max(h, w), 896)


class _CapturingResponse(io.BytesIO):
    """Stand-in for urlopen's response object (context manager + read)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _openai_body(text: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


class TokenCapWireTests(unittest.TestCase):
    """The cap must actually reach the JSON that goes over the wire."""

    def _payload_sent(self, **kwargs) -> dict:
        from cvti.scene import agent_mapper
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _CapturingResponse(_openai_body("ok"))

        with mock.patch.object(agent_mapper.urlrequest, "urlopen", fake_urlopen):
            agent_mapper.call_openai_compatible(
                prompt="p", frame_bytes=b"jpg", model="m",
                api_key_env="", api_base_url="http://localhost:11434/v1",
                require_key=False, **kwargs)
        return captured["payload"]

    def test_max_tokens_lands_in_the_payload(self):
        self.assertEqual(self._payload_sent(max_tokens=96)["max_tokens"], 96)

    def test_no_cap_means_no_field(self):
        # None must OMIT the field, not send max_tokens: null — some backends
        # reject explicit nulls.
        self.assertNotIn("max_tokens", self._payload_sent())


class GateCapTests(unittest.TestCase):
    """Each gate mode names its own budget: CoT reasons briefly, JSON doesn't."""

    def _alert(self) -> CandidateAlert:
        return CandidateAlert(
            rule_name="theft_attempt", priority="high", detector="video_action",
            title="t", person_id=1, object_label=None, timestamp=0.0)

    def _gate_call_kwargs(self, cot: bool) -> dict:
        from cvti.verification.gate import VerificationGate
        gate = VerificationGate(provider="ollama", cot=cot)
        verdict = json.dumps({"confirmed": False, "confidence": 0.9,
                              "reason": "empty scene", "alert_priority": "high"})
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible",
                        return_value=verdict) as called:
            result = gate.verify(_frame(640, 480), self._alert())
        self.assertFalse(result.errored)      # the stubbed verdict parsed cleanly
        return called.call_args.kwargs

    def test_cot_verdicts_are_capped_at_256(self):
        self.assertEqual(self._gate_call_kwargs(cot=True)["max_tokens"], 256)

    def test_json_only_verdicts_are_capped_at_192(self):
        # 96 cut real verdicts mid-object once a reason was included (CHI
        # self-test, 18 Oct 2026); 192 is still a bounded one-second answer.
        self.assertEqual(self._gate_call_kwargs(cot=False)["max_tokens"], 192)

    def test_scanner_caps_and_downscales_what_it_sends(self):
        from cvti.serving.custom_rules import CustomRuleScanner
        cam = {"id": "front", "source": "x",
               "custom_threats": [{"name": "hoodie", "description": "a hoodie"}]}
        scanner = CustomRuleScanner([cam], sink=None, model=LOCAL_VLM_MODEL)
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible",
                        return_value='{"threats": []}') as called:
            scanner._check(cam, _frame(1920, 1080))
        kwargs = called.call_args.kwargs
        # 320 since 3 Sep: each claim also carries confidence, target and a
        # box. Still a hard cap — the point is bounded, not unbounded.
        self.assertEqual(kwargs["max_tokens"], 320)
        h, w = _decoded_shape(kwargs["frame_bytes"])
        self.assertLessEqual(max(h, w), 896)


class OneModelTagTests(unittest.TestCase):
    """gemma3:4b everywhere a default lives — the tag every number was measured on."""

    def test_gate_defaults_use_the_shipped_tag(self):
        from cvti.verification.gate import VerificationGate
        self.assertEqual(VerificationGate(provider="ollama").model, LOCAL_VLM_MODEL)
        self.assertEqual(VerificationGate(provider="local").model, LOCAL_VLM_MODEL)

    def test_mapper_defaults_use_the_shipped_tag(self):
        from cvti.scene.agent_mapper import AgentMapper
        self.assertEqual(AgentMapper(provider="ollama").model, LOCAL_VLM_MODEL)
        self.assertEqual(AgentMapper(provider="local").model, LOCAL_VLM_MODEL)

    def test_operational_helpers_and_first_run_pull_agree(self):
        from cvti.serving import vlm
        from cvti.verification import ollama
        self.assertEqual(ollama.DEFAULT_MODEL, LOCAL_VLM_MODEL)
        self.assertEqual(vlm.DEFAULT_MODEL, LOCAL_VLM_MODEL)

    def test_the_qat_split_is_gone_from_every_default(self):
        """Regression pin for the 4b vs 4b-it-qat drift itself: no DEFAULT
        anywhere may reintroduce a tag the first-run pull doesn't download."""
        from cvti.scene.agent_mapper import AgentMapper
        from cvti.verification.gate import VerificationGate
        defaults = (list(VerificationGate._DEFAULT_MODELS.values())
                    + list(AgentMapper._DEFAULT_MODELS.values()))
        self.assertFalse([m for m in defaults if "it-qat" in m])


if __name__ == "__main__":
    unittest.main()
