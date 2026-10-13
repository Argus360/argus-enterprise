"""The verifier is a choice, not a hardcoded Ollama (CHI pilot, 8 Oct 2026).

A five-vCPU server with no GPU spent 30 to 90 s per verdict on the local
model and the breaker opened after two of them. The engine could already
talk to OpenRouter, Anthropic and any OpenAI-compatible cloud; the app never
let anyone pick one. These tests pin down the pieces that let it:

  - the provider catalogue turns a site choice into engine flags and env;
  - the launch passes the key through the environment only;
  - the local runtime is not started for a cloud provider;
  - gate_status and setup_check describe a cloud verifier truthfully;
  - the openai_compatible path sends every frame, with retries;
  - scene mapping, English rules and watches follow the gate to the cloud;
  - the secrets file is private and never inside site.json.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cvti.app.secrets import VERIFIER_KEY, SecretStore  # noqa: E402
from cvti.verification import providers  # noqa: E402
from _backend_helper import signed_in  # noqa: E402


def _backend(case):
    """A signed-in backend in its own temp dir. Cleanups run last-in first-out:
    the backend's databases and log are closed BEFORE the directory goes,
    which is what Windows needs to delete it."""
    tmp = tempfile.mkdtemp()
    case.addCleanup(shutil.rmtree, tmp, True)
    site = Path(tmp) / "site"
    site.mkdir(exist_ok=True)
    (site / "site.json").write_text(json.dumps({"name": "Test", "cameras": [
        {"id": "cam1", "name": "Gate", "source": "rtsp://127.0.0.1:1/x"}]}))
    cb = signed_in(site_path=str(site / "site.json"), db_path=str(site / "events.db"),
                   enable_demo=False)
    case.addCleanup(cb.close)
    case.addCleanup(cb._close_engine_log)
    return cb


class ProviderCatalogue(unittest.TestCase):
    def test_unknown_or_empty_means_the_local_runtime(self):
        self.assertEqual(providers.get_provider(None).id, "ollama")
        self.assertEqual(providers.get_provider("nonsense").id, "ollama")
        self.assertTrue(providers.get_provider("ollama").local)

    def test_normalise_fills_the_default_model_and_knows_each_base_url(self):
        s = providers.normalize_settings({"provider": "groq"})
        self.assertEqual(s["model"], providers.PROVIDERS["groq"].default_model)
        self.assertEqual(s["base_url"], providers.GROQ_BASE_URL)
        custom = providers.normalize_settings({"provider": "custom", "model": "m", "base_url": "https://h/v1"})
        self.assertEqual(custom["base_url"], "https://h/v1")
        # a non-custom provider ignores a stray base_url
        self.assertEqual(providers.normalize_settings({"provider": "openrouter", "base_url": "https://x"})["base_url"], "")

    def test_engine_args_per_provider(self):
        args = providers.engine_args(providers.normalize_settings({"provider": "ollama"}))
        self.assertEqual(args[:2], ["--gate-provider", "ollama"])
        self.assertNotIn("--gate-base-url", args)

        args = providers.engine_args(providers.normalize_settings({"provider": "groq"}))
        self.assertEqual(args[args.index("--gate-provider") + 1], "openai_compatible")
        self.assertEqual(args[args.index("--gate-base-url") + 1], providers.GROQ_BASE_URL)
        self.assertNotIn("--mapper-provider", args)   # same provider, same URL: nothing to override
        # Groq accepts at most three images per request: the cap rides along.
        self.assertEqual(args[args.index("--gate-max-frames") + 1], "3")
        self.assertNotIn("--gate-max-frames", providers.engine_args(providers.normalize_settings({"provider": "openrouter"})))

        args = providers.engine_args(providers.normalize_settings({"provider": "openrouter"}))
        self.assertEqual(args[args.index("--gate-provider") + 1], "openrouter")
        self.assertEqual(args[args.index("--mapper-provider") + 1], "openai_compatible")
        self.assertEqual(args[args.index("--mapper-base-url") + 1], providers.OPENROUTER_BASE_URL)

        args = providers.engine_args(providers.normalize_settings({"provider": "anthropic"}))
        self.assertEqual(args[args.index("--gate-provider") + 1], "anthropic")
        self.assertNotIn("--mapper-provider", args)

    def test_key_fans_out_to_every_variable_the_engine_reads(self):
        env = providers.key_environment({"provider": "openrouter"}, "sk-x")
        self.assertEqual(env, {"OPENROUTER_API_KEY": "sk-x", "OPENAI_API_KEY": "sk-x"})
        self.assertEqual(providers.key_environment({"provider": "ollama"}, "sk-x"), {})
        self.assertEqual(providers.key_environment({"provider": "groq"}, ""), {})

    def test_public_rows_never_carry_a_key(self):
        for row in providers.PROVIDERS.values():
            public = row.public()
            self.assertNotIn("key_env", public)
            self.assertNotIn("key_envs", public)
            self.assertNotIn("api_key", public)


class SecretsFile(unittest.TestCase):
    def test_private_file_round_trip_and_delete(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        store = SecretStore(tmp)
        self.assertEqual(store.get("k"), "")
        store.set("k", "sk-secret")
        self.assertEqual(store.get("k"), "sk-secret")
        if os.name != "nt":
            mode = stat.S_IMODE(os.stat(store.path).st_mode)
            self.assertEqual(mode, stat.S_IRUSR | stat.S_IWUSR, oct(mode))
        store.delete("k")
        self.assertFalse(store.has("k"))


class LaunchingTheEngine(unittest.TestCase):
    def _launch(self, cb):
        """Spawn with Popen and ensure_server stubbed; return (argv, kwargs, ensure_calls)."""
        calls = {"ensure": 0}

        def ensure():
            calls["ensure"] += 1

        with mock.patch("cvti.verification.ollama.ensure_server", ensure), \
             mock.patch("subprocess.Popen") as popen:
            popen.return_value = mock.Mock(poll=lambda: None)
            try:
                cb._spawn_engine()
            finally:
                # The spawn opens monitor.log for the (mocked) subprocess; on
                # Windows an open handle blocks the temp directory's removal.
                cb._close_engine_log()
        argv = popen.call_args.args[0]
        return argv, popen.call_args.kwargs, calls["ensure"]

    def test_default_is_the_local_runtime_and_it_is_started(self):
        cb = _backend(self)
        argv, kwargs, ensure = self._launch(cb)
        self.assertEqual(argv[argv.index("--gate-provider") + 1], "ollama")
        self.assertEqual(ensure, 1)
        self.assertNotIn("env", kwargs)

    def test_cloud_provider_skips_ollama_and_passes_the_key_in_env_only(self):
        cb = _backend(self)
        cb.set_verifier(provider="openrouter", model="google/gemini-2.5-flash-lite", api_key="sk-or-123")
        argv, kwargs, ensure = self._launch(cb)
        self.assertEqual(ensure, 0, "a cloud verifier must not start the local runtime")
        self.assertEqual(argv[argv.index("--gate-provider") + 1], "openrouter")
        self.assertEqual(argv[argv.index("--gate-model") + 1], "google/gemini-2.5-flash-lite")
        self.assertNotIn("sk-or-123", " ".join(argv), "the key must never be on the command line")
        self.assertEqual(kwargs["env"]["OPENROUTER_API_KEY"], "sk-or-123")
        self.assertEqual(kwargs["env"]["OPENAI_API_KEY"], "sk-or-123")
        site = json.loads(Path(cb.site_path).read_text())
        self.assertNotIn("sk-or-123", json.dumps(site), "the key must never be in site.json")
        self.assertEqual(site["gate"]["provider"], "openrouter")

    def test_custom_endpoint_needs_url_and_model(self):
        cb = _backend(self)
        with self.assertRaises(ValueError):
            cb.set_verifier(provider="custom", model="m")
        with self.assertRaises(ValueError):
            cb.set_verifier(provider="custom", base_url="https://h/v1")
        cb.set_verifier(provider="custom", model="m", base_url="https://h/v1", api_key="k")
        argv, kwargs, _ = self._launch(cb)
        self.assertEqual(argv[argv.index("--gate-base-url") + 1], "https://h/v1")
        self.assertEqual(kwargs["env"]["OPENAI_API_KEY"], "k")

    def test_saving_without_a_key_keeps_the_stored_one(self):
        cb = _backend(self)
        cb.set_verifier(provider="groq", api_key="first")
        cb.set_verifier(provider="groq", model="other-model", api_key="")
        self.assertEqual(cb._secret_store().get(VERIFIER_KEY), "first")
        self.assertTrue(cb.verifier_settings()["key_set"])
        cb.clear_verifier_key()
        self.assertFalse(cb.verifier_settings()["key_set"])
        self.assertNotIn("first", json.dumps(cb.verifier_settings()))


class StatusAndChecks(unittest.TestCase):
    def test_gate_status_for_a_cloud_provider_is_live_once_the_key_is_saved(self):
        cb = _backend(self)
        cb.set_verifier(provider="gemini")
        s = cb.gate_status()
        self.assertTrue(s["cloud"])
        self.assertEqual(s["mode"], "no-key")
        cb.set_verifier(provider="gemini", api_key="AIza")
        s = cb.gate_status()
        self.assertEqual(s["mode"], "live")
        self.assertEqual(s["provider"], "gemini")
        self.assertFalse(s["ollama"])

    def test_gate_status_for_the_local_runtime_still_probes_ollama(self):
        cb = _backend(self)
        with mock.patch("cvti.serving.vlm.gate_status",
                        return_value={"ollama": True, "model_present": True, "mode": "live",
                                      "model": "gemma3:4b", "models": []}) as probe:
            s = cb.gate_status()
        self.assertTrue(probe.called)
        self.assertFalse(s["cloud"])
        self.assertEqual(s["mode"], "live")

    def test_setup_check_names_the_missing_key(self):
        cb = _backend(self)
        cb.set_verifier(provider="groq")
        with mock.patch.object(cb, "_probe_stream", return_value=(True, "ok")):
            rows = {c["id"]: c for c in cb.setup_check()}
        self.assertFalse(rows["verifier"]["ok"])
        self.assertIn("API key", rows["verifier"]["fix"])
        cb.set_verifier(provider="groq", api_key="gsk")
        with mock.patch.object(cb, "_probe_stream", return_value=(True, "ok")):
            rows = {c["id"]: c for c in cb.setup_check()}
        self.assertTrue(rows["verifier"]["ok"])
        self.assertIn("Groq", rows["verifier"]["detail"])

    def test_test_verifier_reports_a_bad_key_without_raising(self):
        cb = _backend(self)
        cb.set_verifier(provider="groq", api_key="bad")
        with mock.patch("cvti.app.console_backend._probe_cloud_verifier",
                        side_effect=RuntimeError("HTTP 401: invalid api key")):
            r = cb.test_verifier()
        self.assertFalse(r["ok"])
        self.assertIn("401", r["detail"])
        with mock.patch("cvti.app.console_backend._probe_cloud_verifier", return_value="OK"):
            r = cb.test_verifier()
        self.assertTrue(r["ok"])
        self.assertIn("Groq", r["detail"])

    def test_probe_restores_the_environment(self):
        spec = providers.PROVIDERS["groq"]
        os.environ.pop("OPENAI_API_KEY", None)
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible", return_value="OK") as call:
            from cvti.app.console_backend import _probe_cloud_verifier
            out = _probe_cloud_verifier(spec, providers.normalize_settings({"provider": "groq"}), "gsk")
        self.assertEqual(out, "OK")
        self.assertEqual(call.call_args.kwargs["api_base_url"], providers.GROQ_BASE_URL)
        self.assertNotIn("OPENAI_API_KEY", os.environ)


class TheGateOnACloud(unittest.TestCase):
    def test_openai_compatible_sends_every_frame_with_retries(self):
        from cvti.verification.gate import VerificationGate
        gate = VerificationGate(provider="openai_compatible", model="m", base_url="https://h/v1")
        self.assertEqual(gate.api_key_env, "OPENAI_API_KEY")
        alert = mock.Mock(detector="loitering")
        frames = [b"a", b"b", b"c"]
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible", return_value="{}") as call:
            gate._call_provider("prompt", frames, alert)
        kw = call.call_args.kwargs
        self.assertEqual(kw["frame_bytes"], frames)
        self.assertEqual(kw["api_base_url"], "https://h/v1")
        self.assertEqual(kw["api_key_env"], "OPENAI_API_KEY")
        self.assertEqual(kw["max_retries"], VerificationGate.CLOUD_MAX_RETRIES)
        self.assertLessEqual(kw["timeout"], VerificationGate.CLOUD_TIMEOUT_S)

    def test_local_still_sends_one_frame_and_no_key(self):
        from cvti.verification.gate import VerificationGate
        gate = VerificationGate(provider="local", model="m")
        alert = mock.Mock(detector="loitering")
        with mock.patch("cvti.verification.gate._call_openai_compatible", return_value="{}") as call:
            gate._call_provider("prompt", [b"a", b"b"], alert)
        self.assertEqual(call.call_args.kwargs["frame_bytes"], b"a")
        self.assertEqual(call.call_args.kwargs["api_key_env"], "")


class ProviderFrameCap(unittest.TestCase):
    def test_provider_cap_applies_unless_the_site_chose_its_own(self):
        from cvti.serving.pipeline import _apply_gate_frame_cap
        self.assertEqual(_apply_gate_frame_cap({}, 3)["gate_max_frames"], 3)
        self.assertEqual(_apply_gate_frame_cap({"gate_max_frames": 2}, 3)["gate_max_frames"], 2)
        self.assertNotIn("gate_max_frames", _apply_gate_frame_cap({}, None))
        self.assertNotIn("gate_max_frames", _apply_gate_frame_cap({}, 0))


class EverythingFollowsTheGate(unittest.TestCase):
    def test_scene_mapping_goes_to_openrouter_not_a_local_ollama(self):
        from cvti.serving.pipeline import resolve_mapper_settings
        provider, model, base = resolve_mapper_settings(
            gate_provider="openrouter", gate_model="google/x", gate_base_url="",
            mapper_provider="", mapper_model="", mapper_base_url="")
        self.assertEqual(provider, "openai_compatible")
        self.assertEqual(model, "google/x")
        self.assertEqual(base, providers.OPENROUTER_BASE_URL)

    def test_scanner_endpoint_per_provider(self):
        from cvti.serving.pipeline import scanner_endpoint_for
        self.assertEqual(scanner_endpoint_for("ollama", ""), ("http://localhost:11434/v1", "OLLAMA_API_KEY"))
        self.assertEqual(scanner_endpoint_for("openrouter", ""), (providers.OPENROUTER_BASE_URL, "OPENROUTER_API_KEY"))
        self.assertEqual(scanner_endpoint_for("openai_compatible", "https://h/v1"), ("https://h/v1", "OPENAI_API_KEY"))
        self.assertIsNone(scanner_endpoint_for("anthropic", ""))
        self.assertIsNone(scanner_endpoint_for("mock", ""))

    def test_scanner_and_watch_runner_use_the_key_variable_they_were_given(self):
        from cvti.serving.custom_rules import CustomRuleScanner
        from cvti.serving.watch_runner import WatchRunner
        scanner = CustomRuleScanner([], sink=None, model="m", base_url="https://h/v1", api_key_env="OPENAI_API_KEY")
        self.assertEqual(scanner.api_key_env, "OPENAI_API_KEY")
        runner = WatchRunner([], {}, None, model="m", base_url="https://h/v1", api_key_env="OPENROUTER_API_KEY")
        with mock.patch("cvti.scene.agent_mapper.call_openai_compatible", return_value="[]") as call:
            runner._ask("p", b"jpg")
        self.assertEqual(call.call_args.kwargs["api_key_env"], "OPENROUTER_API_KEY")
        self.assertEqual(call.call_args.kwargs["api_base_url"], "https://h/v1")


if __name__ == "__main__":
    unittest.main()
