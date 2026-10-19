"""CHI's daily self-test failed on 18 and 19 Oct 2026 with "Unterminated JSON
in VLM response": Groq's Qwen model thinks before it answers and the thinking
ate the 96-token verdict budget. The Groq request now asks for the answer
only, the verdict budget has room for a reason, and a <think> block that does
arrive is skipped rather than parsed as the verdict."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Resp:
    def __init__(self, body: dict):
        self._b = json.dumps(body).encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _capture_payload(base_url: str) -> dict:
    from cvti.scene import agent_mapper as am
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["payload"] = json.loads(req.data.decode())
        return _Resp({"choices": [{"message": {"content": "{\"confirmed\": true}"}}]})

    with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "k"}), \
            mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        am.call_openai_compatible(prompt="p", frame_bytes=b"\xff\xd8", model="m",
                                  api_key_env="OPENAI_API_KEY", api_base_url=base_url,
                                  max_tokens=192, max_retries=1, timeout=5)
    return seen["payload"]


class GroqRequestTests(unittest.TestCase):
    def test_groq_is_asked_for_the_answer_only(self):
        p = _capture_payload("https://api.groq.com/openai/v1")
        self.assertEqual(p.get("reasoning_format"), "hidden")
        self.assertEqual(p.get("reasoning_effort"), "none")
        self.assertEqual(p.get("max_tokens"), 192)

    def test_other_endpoints_get_a_plain_request(self):
        p = _capture_payload("https://openrouter.ai/api/v1")
        self.assertNotIn("reasoning_format", p)
        self.assertNotIn("reasoning_effort", p)


class ThinkBlockTests(unittest.TestCase):
    def test_a_closed_think_block_is_skipped(self):
        from cvti.verification.gate import _extract_json
        raw = "<think>is this {a person}? yes {}</think>\n{\"confirmed\": true, \"confidence\": 0.9}"
        self.assertEqual(_extract_json(raw)["confirmed"], True)

    def test_a_cut_off_think_block_is_still_unterminated(self):
        from cvti.verification.gate import _extract_json
        with self.assertRaises(RuntimeError):
            _extract_json("<think>reasoning that never ends {")

    def test_verdict_budget_has_room_for_a_reason(self):
        from cvti.verification.gate import VerificationGate
        self.assertGreaterEqual(VerificationGate.MAX_TOKENS_JSON, 160)


if __name__ == "__main__":
    unittest.main()
