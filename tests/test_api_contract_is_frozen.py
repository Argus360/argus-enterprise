"""The API contract file is held equal to both of its sources (9 Sep).

docs/api-v1.md is the frozen agreement between the engine and its clients.
"Frozen" is enforceable only if drift is mechanical to catch, so:

  1. every route the FastAPI app actually serves appears in the contract;
  2. every backend operation Demi's shipped UI invokes (Frontend/bridge.py
     METHODS — the de-facto client surface) has a row in the contract;
  3. the generated docs/openapi.json matches the implemented app.

A failure here means someone changed the code or the client surface without
updating the agreement — which is exactly the afternoon-losing mistake the
freeze exists to prevent.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOC = (ROOT / "docs" / "api-v1.md").read_text(encoding="utf-8")


class ImplementedRoutesAppearInTheContract(unittest.TestCase):
    def test_every_served_route_is_documented(self):
        from cvti.api.app import create_app
        app = create_app()
        missing = []
        for route in app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/api/v1"):
                continue
            tail = path.removeprefix("/api/v1") or "/"
            # the doc writes {id}-style params; the app writes {camera_id}
            pattern = re.sub(r"\{[^}]+\}", r"\\{[^}]+\\}", re.escape(tail)
                             ).replace(r"\\\{[^}]+\\\}", r"\{[^}]+\}")
            # Starlette route converters are part of the implementation, not
            # the public URL contract: {camera_id:path} stays {camera_id} in
            # the documented OpenAPI path.
            normalized = re.sub(r"\{\{?[a-z_:]+\}?\}", "{X}", tail)
            doc_normalized = re.sub(r"\{[a-z_]+\}", "{X}", DOC)
            if normalized not in doc_normalized:
                missing.append(path)
        self.assertEqual(missing, [],
                         f"served but absent from docs/api-v1.md: {missing}")

    def test_hierarchy_extension_routes_are_served(self):
        from cvti.api.app import create_app
        served = {
            (method, route.path)
            for route in create_app().routes
            for method in getattr(route, "methods", set())
        }
        expected = {
            ("GET", "/api/v1/organization"),
            ("PUT", "/api/v1/organization"),
            ("GET", "/api/v1/branches"),
            ("POST", "/api/v1/branches"),
            ("PUT", "/api/v1/branches/{branch_id}"),
            ("DELETE", "/api/v1/branches/{branch_id}"),
            ("GET", "/api/v1/hierarchy"),
        }
        self.assertEqual(expected - served, set())


class TheClientSurfaceIsFullyMapped(unittest.TestCase):
    def test_every_bridge_method_has_a_contract_row(self):
        bridge = (ROOT / "Frontend" / "bridge.py").read_text(encoding="utf-8")
        m = re.search(r"METHODS = set\('([^']+)'", bridge)
        self.assertIsNotNone(m, "bridge METHODS moved — update this test")
        methods = set(m.group(1).split())
        methods.add("live_stop")
        unmapped = sorted(meth for meth in methods if f"`{meth}`" not in DOC)
        self.assertEqual(unmapped, [],
                         f"UI operations with no contract row: {unmapped}")


class GeneratedSpecMatchesTheApp(unittest.TestCase):
    def test_openapi_json_is_current(self):
        from cvti.api.app import create_app
        live = set(create_app().openapi().get("paths", {}))
        committed = set(json.loads(
            (ROOT / "docs" / "openapi.json").read_text(encoding="utf-8")).get("paths", {}))
        self.assertEqual(live, committed,
                         "docs/openapi.json is stale — regenerate it in the "
                         "same PR that changed the routes")


class TheGapsTheContractExists(unittest.TestCase):
    def test_alert_update_is_promised_before_ws_triage(self):
        """The fast path settles provisional criticals in place; a WS client
        without alert.update shows stale verdicts forever. The contract must
        keep carrying that promise until the message ships."""
        self.assertIn("alert.update", DOC)

    def test_package_description_does_not_claim_writes_are_pending(self):
        import cvti.api
        package_contract = (cvti.api.__doc__ or "").lower()
        self.assertNotIn("read-only", package_contract)
        self.assertNotIn("write and config endpoints", package_contract)


if __name__ == "__main__":
    unittest.main()
