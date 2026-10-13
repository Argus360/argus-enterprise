"""Console backend core — the logic behind the desktop app's UI (Phase 9).

Qt-free and unit-tested. The QWebChannel bridge (cvti/app/bridge.py) is a thin
wrapper that JSON-serializes these methods for the web frontend; the desktop
shell hosts the frontend in a native window. Keeping the logic here means the
part that matters is verifiable without a display.

Covers the two core operator flows:
  • Cameras  — list / scan / test / add / remove (reuses cvti.serving.onboarding).
  • Alerts   — list confirmed events (with evidence as data URIs) + review them
               true / false / acknowledged (the label that feeds retraining).
"""
from __future__ import annotations

import base64
import binascii
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from cvti.app.secrets import VERIFIER_KEY, SecretStore
from cvti.contracts import LOCAL_VLM_MODEL
from cvti.logging_setup import get_logger
from cvti.security import permissions as perms
from cvti.security.accounts import AccountStore, AuthError
from cvti.security.audit import AuditLog
from cvti.serving import onboarding, vlm
from cvti.utils import resource_path
from cvti.verification import providers as verifier_providers

log = get_logger(__name__)

_REVIEW_VALUES = {"ack", "true", "false", "new"}

# monitor.log (the engine subprocess's stdout/stderr) is append-only across
# every run and respawn — on a 24/7 site it grew without bound, and the crash
# loops that make it valuable are exactly what makes it grow fastest.
MONITOR_LOG_CAP_BYTES = 5 * 1024 * 1024
OBJECT_EXAMPLE_MAX_BYTES = 8 * 1024 * 1024


def _probe_cloud_verifier(spec, settings: dict, api_key: str) -> str:
    """One text-only request to a cloud verifier, returning its reply.

    Runs in the API process, so the key is handed to the HTTP helpers through
    a scoped environment and removed again: the engine subprocess is the only
    place the key is meant to live for long."""
    import os

    from cvti.scene.agent_mapper import call_openai_compatible
    from cvti.verification.gate import _call_anthropic

    prompt = "Reply with the single word OK."
    env_names = spec.key_envs or (spec.key_env,)
    saved = {name: os.environ.get(name) for name in env_names}
    try:
        for name in env_names:
            os.environ[name] = api_key
        if spec.engine_provider == "anthropic":
            return _call_anthropic(prompt, [], settings["model"], spec.key_env)
        base_url = settings.get("base_url") or spec.base_url or verifier_providers.OPENROUTER_BASE_URL
        if spec.engine_provider == "openrouter":
            base_url = verifier_providers.OPENROUTER_BASE_URL
        return call_openai_compatible(prompt=prompt, frame_bytes=[], model=settings["model"],
                                      api_key_env=spec.key_env, api_base_url=base_url,
                                      max_retries=0, max_tokens=8, timeout=30.0)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _desktop_inference_args(site: dict) -> list[str]:
    """Allow a site to trade detector throughput for small-person coverage."""
    settings = site.get("inference", {})
    if not isinstance(settings, dict):
        raise ValueError("inference must be an object")
    size = settings.get("imgsz", 512)
    fps = settings.get("target_fps", 4)
    confidence = settings.get("confidence", 0.4)
    if type(size) is not int or not 320 <= size <= 1280 or size % 32:
        raise ValueError("inference.imgsz must be a multiple of 32 between 320 and 1280")
    for name, value, low, high in (("target_fps", fps, 1, 30),
                                   ("confidence", confidence, 0.1, 0.9)):
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError(f"inference.{name} must be between {low} and {high}")
    return ["--target-fps", str(fps), "--imgsz", str(size), "--conf", str(confidence)]


def _rotate_monitor_log(path: Path, cap_bytes: int = MONITOR_LOG_CAP_BYTES) -> None:
    """Rotate monitor.log at spawn time when it has outgrown the cap.

    One predecessor is kept (monitor.log.1) so a crash's final words survive
    the restart that follows it. Rotation happens between spawns — the only
    moment no process holds the file — and is best-effort: a failed rename
    (say, an old handle still open on Windows) just means we append on.
    """
    try:
        if path.is_file() and path.stat().st_size > cap_bytes:
            path.replace(path.with_name(path.name + ".1"))
    except OSError:
        log.warning("could not rotate %s; appending instead", path, exc_info=True)


def _has_siglip_files(path) -> bool:
    """A model directory with the files SigLIP needs — judged by presence only,
    so it holds without transformers installed (the API process ships without it)."""
    from pathlib import Path as _P
    d = _P(path)
    if not d.is_dir() or not (d / "config.json").is_file() or not (d / "preprocessor_config.json").is_file():
        return False
    return any(d.glob("*.safetensors")) or any(d.glob("pytorch_model*.bin"))


class ConsoleBackend:
    def __init__(self, site_path: str = "configs/site_live.json",
                 db_path: str = "runs/site/events.db", enable_demo: bool = True) -> None:
        self.site_path = site_path
        self.db_path = db_path
        self._live = None       # LiveWall instance while the Live screen is open
        self._fs = None         # localhost FrameServer serving live JPEGs
        self._monitor = None    # detection-engine subprocess (Start monitoring)
        self._monitor_should_run = False  # watchdog: respawn engine if it dies
        self._restarts = 0
        # The HOME site keeps the original store; every other feed gets its own
        # (quick-win #2, 25 Aug): demo/EarthCam alerts used to pollute the real
        # triage queue — 95 junk Dublin Street events in one sitting — because
        # every feed wrote into one events.db. Auth and audit stay pinned to
        # the home directory below regardless of feed (accounts are global).
        self._home_site = str(Path(site_path).resolve())
        self._home_db = self.db_path
        # bundled playback demo (for machines w/o the engine); off in tests
        self._demo = self._locate_demo() if enable_demo else None

        # --- identity, roles, audit (EP-03) ---
        # Credentials and the audit trail sit beside the events database, never
        # inside it: one deletion must not take both the footage and the record
        # of who touched it.
        _sec_dir = Path(self.db_path).parent
        self.accounts = AccountStore(_sec_dir / "auth.db")
        self.audit = AuditLog(_sec_dir / "audit.db")
        # EP-08-T1: a corrupt events.db is quarantined LOUDLY at startup (a
        # fresh store follows), and the config is backed up daily, versioned.
        from cvti import backup as _backup
        try:
            self.db_check = _backup.check_events_db(self.db_path)
        except Exception:  # noqa: BLE001 - the check must not stop the app
            log.error("events.db startup check failed", exc_info=True)
            self.db_check = {"ok": True, "state": "unchecked"}
        try:
            self._maybe_auto_backup()
        except Exception:  # noqa: BLE001
            log.warning("auto-backup failed", exc_info=True)
        self._session: str = ""          # token of the signed-in operator

    # --- session ----------------------------------------------------------
    @property
    def current_user(self):
        return self.accounts.session_user(self._session) if self._session else None

    def _role(self):
        user = self.current_user
        return user.role if user else None

    def _require(self, permission: str) -> None:
        """Server-side authorisation. Hiding a button changes what is easy;
        this changes what is possible, which is the question that matters."""
        perms.require(self._role(), permission)

    def auth_state(self) -> dict:
        """What the UI needs to pick a screen. Never a permission source — the
        backend re-checks on every call."""
        user = self.current_user
        return {
            "configured": self.accounts.any_users(),
            "signed_in": user is not None,
            "username": user.username if user else "",
            "role": user.role if user else "",
            "must_change_password": bool(user.must_change) if user else False,
            "landing": perms.landing_for(user.role) if user else "",
            "permissions": sorted(perms.permissions_for(user.role)) if user else [],
        }

    def app_version(self) -> str:
        """Which build this actually is. The sidebar said 'v0.9' from a
        hardcoded string while three releases shipped in two days; now every
        surface reads the one source (cvti.utils.argus_version)."""
        from cvti.utils import argus_version
        return argus_version()

    def english_rules_status(self) -> dict:
        """The scanner's own account of itself — written every cycle by the
        engine. Distinguishes the three silences that all looked identical
        from the operator's chair ('the describe in english hasn't fired',
        three reports in two days): the model answering none, the calls
        failing, and nothing scanning at all."""
        path = Path(self.db_path).parent / "english_rules_status.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return {"available": False}
        data["available"] = True
        data["age_s"] = round(time.time() - float(data.get("generated_at") or 0), 1)
        return data

    def auth_recovery(self) -> dict:
        """What the sign-in screen shows a locked-out owner (pre-auth, read-only).

        'Invalid credentials' used to be a dead end: no dev account exists by
        design, so an owner who lost the password — or inherited a machine
        with an account someone else created — had no path forward and no way
        to know one existed (field report, 28 Aug). Recovery stays an
        OS-level act on purpose: deleting the store needs access to the
        machine's files, not just this window, so a passer-by at the console
        cannot seize the site. This only tells the operator where that file
        is on THIS machine; viewing it is audit-logged.
        """
        path = Path(self.accounts.db_path if hasattr(self.accounts, "db_path")
                    else Path(self._home_db).parent / "auth.db")
        try:
            self.audit.record("<unknown>", "auth_recovery_viewed",
                              detail={"outcome": "instructions shown"})
        except Exception:  # noqa: BLE001 - showing help must not depend on audit
            log.debug("audit write failed for recovery view", exc_info=True)
        return {"auth_db": str(path),
                "evidence_untouched": True}

    def auth_accounts(self) -> dict:
        """Who already holds this site — shown on the Create tab pre-auth, so
        'invalid credentials' is never a mystery about WHETHER an account
        exists, only which. Names and roles only, never anything secret."""
        return {"users": [{"username": u.username, "role": u.role}
                          for u in self.accounts.list_users()]}

    def create_owner_override(self, username: str, password: str) -> dict:
        """Replace every existing account with a fresh owner — from the login
        screen, unauthenticated.

        PILOT-PHASE TRADEOFF, at the owner's explicit and reaffirmed decision
        (28 Aug: "for now when we do create account just override it"): during
        the pilot, lockouts on inherited machines cost more than the intrusion
        risk of an on-console reset. The compensating controls: the screen
        lists exactly which accounts will be replaced, the act is confirmed,
        every replaced name goes into the audit log, and the audit log and
        evidence live in files this cannot touch — a hostile reset is loud
        and attributable, never invisible. Revisit before any deployment
        where the console sits in a public spot.
        """
        replaced = [u.username for u in self.accounts.list_users()]
        try:
            for name in replaced:
                self.accounts.delete_user(name)          # also revokes sessions
            self.accounts.create_user(username, password, role=perms.OWNER)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self._session = ""
        self.audit.record(username, "accounts_override_via_login",
                          detail={"replaced": replaced, "new_owner": username})
        log.warning("[auth] accounts overridden from the login screen: %s -> %s",
                    replaced, username)
        return self.sign_in(username, password)

    def create_first_owner(self, username: str, password: str) -> dict:
        """First run. Nothing ships with a known credential because nothing
        ships with an account at all — the first one is created here."""
        if self.accounts.any_users():
            return {"ok": False, "error": "this site already has accounts"}
        try:
            self.accounts.create_user(username, password, role=perms.OWNER)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self.audit.record(username, "role_change", f"user:{username}",
                          {"created": True, "role": perms.OWNER, "first_run": True})
        return self.sign_in(username, password)

    def sign_in(self, username: str, password: str) -> dict:
        try:
            user = self.accounts.authenticate(username, password)
        except AuthError as exc:
            self.audit.record(username or "<unknown>", "login",
                              detail={"outcome": "refused", "reason": str(exc)})
            return {"ok": False, "error": str(exc)}
        self._session = self.accounts.open_session(user.username)
        self.audit.record(user.username, "login",
                          detail={"outcome": "success", "role": user.role})
        return {"ok": True, **self.auth_state()}

    def sign_out(self) -> dict:
        self._close_preview()
        user = self.current_user
        if user:
            self.audit.record(user.username, "login", detail={"outcome": "signed out"})
        self.accounts.close_session(self._session)
        self._session = ""
        return {"ok": True}


    def close(self) -> None:
        """Release the databases this backend owns.

        Windows will not let anyone move or delete a file a process holds
        open, and auth.db/audit.db were held for the life of the backend with
        nothing ever closing them. In the product that blocks an in-place
        upgrade or an uninstall; in the suite it failed 400+ assertions on
        Windows alone, because POSIX unlinks open files without complaint.
        Safe to call twice, and the backend is still readable afterwards for
        anything that does not need its stores.
        """
        self._close_preview()
        for store in ("accounts", "audit"):
            closer = getattr(getattr(self, store, None), "close", None)
            if closer is None:
                continue
            try:
                closer()
            except Exception:  # teardown must not raise
                log.debug("closing %s failed", store, exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def change_own_password(self, current: str, new: str) -> dict:
        user = self.current_user
        if user is None:
            return {"ok": False, "error": "not signed in"}
        try:
            self.accounts.authenticate(user.username, current)
        except AuthError:
            return {"ok": False, "error": "current password is incorrect"}
        try:
            self.accounts.set_password(user.username, new)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self.audit.record(user.username, "role_change", f"user:{user.username}",
                          {"password_changed": True})
        self._session = self.accounts.open_session(user.username)   # set_password revoked it
        return {"ok": True}

    # --- user administration (owner only) ---------------------------------
    def list_users(self) -> list:
        self._require(perms.MANAGE_USERS)
        return [{"username": u.username, "role": u.role, "must_change": u.must_change,
                 "last_login": u.last_login} for u in self.accounts.list_users()]

    def add_user(self, username: str, password: str, role: str = "operator") -> dict:
        self._require(perms.MANAGE_USERS)
        try:
            self.accounts.create_user(username, password, role=role)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self.audit.record(self.current_user.username, "role_change", f"user:{username}",
                          {"created": True, "role": role})
        return {"ok": True}

    def set_user_role(self, username: str, role: str) -> dict:
        self._require(perms.MANAGE_USERS)
        before = self.accounts.user(username)
        try:
            self.accounts.set_role(username, role)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self.audit.record(self.current_user.username, "role_change", f"user:{username}",
                          {"from": before.role if before else None, "to": role})
        return {"ok": True}

    def remove_user(self, username: str) -> dict:
        self._require(perms.MANAGE_USERS)
        me = self.current_user
        if me and me.username == username:
            return {"ok": False, "error": "you cannot remove your own account"}
        owners = [u for u in self.accounts.list_users() if u.role == perms.OWNER]
        if len(owners) <= 1 and any(u.username == username for u in owners):
            # A site with no owner has nobody who can grant access to anyone again.
            return {"ok": False, "error": "the last owner cannot be removed"}
        self.accounts.delete_user(username)
        self.audit.record(me.username if me else "?", "role_change", f"user:{username}",
                          {"removed": True})
        return {"ok": True}

    # --- audit trail (owner only) -----------------------------------------
    def audit_entries(self, limit: int = 200) -> list:
        self._require(perms.VIEW_AUDIT)
        return [e.to_dict() for e in self.audit.entries(limit=limit)]

    def audit_verify(self) -> dict:
        self._require(perms.VIEW_AUDIT)
        return self.audit.verify()

    def audit_export(self) -> dict:
        self._require(perms.VIEW_AUDIT)
        path = self.audit.export(Path(self.db_path).parent / f"argus-audit-{int(time.time())}.json")
        self.audit.record(self.current_user.username, "evidence_export", "audit_log",
                          {"path": str(path)})
        return {"ok": True, "path": str(path)}

    def heartbeat_status(self) -> dict:
        """Config + exactly what was last transmitted, so 'what leaves my
        machine?' is answered by looking, not by trusting the docs."""
        meta = self.get_site()
        out = {"enabled": bool(meta.get("heartbeat_url")),
               "url": meta.get("heartbeat_url", ""),
               "has_key": bool(meta.get("heartbeat_key"))}
        engine = self._gate_health() or {}
        out["live"] = engine.get("heartbeat") or {}
        try:
            out["last_payload"] = json.loads(
                (Path(self.db_path).parent / "heartbeat_last.json").read_text())
        except (OSError, ValueError):
            out["last_payload"] = None
        return out

    def set_heartbeat(self, url: str = "", key: str = "") -> dict:
        self._require(perms.CONFIGURE_SITE)
        onboarding.set_site_meta(self.site_path, heartbeat_url=(url or "").strip(),
                                 heartbeat_key=(key or "").strip())
        self.audit.record(self.current_user.username, "config_change", "heartbeat",
                          {"enabled": bool((url or "").strip())})
        return self.heartbeat_status()

    def disk_encryption(self) -> dict:
        """Is the evidence on this machine readable if the machine is taken?"""
        from cvti.security.disk import encryption_status, requirement_message
        status = encryption_status()
        status["message"] = requirement_message(status)
        return status

    def role_table(self) -> dict:
        """The whole permission table — for the UI, and for procurement."""
        return perms.describe()

    def _writable_config(self, rel: str) -> Path:
        """A per-user copy of a bundled config that the app needs to WRITE.

        Feed switching regenerates live_camera.json; English rules and zones
        edit the active site config. In dev those writes land in the repo —
        fine. Installed, they would land inside the bundle: silently wrong on
        macOS, PermissionError under Program Files on Windows (the installer
        made this failure certain, not just likely). First use copies the
        bundled file next to the user's site data and edits that copy.
        """
        if not getattr(sys, "frozen", False):
            return resource_path(rel)
        from cvti.utils import user_data_dir
        dst = user_data_dir() / "feeds" / Path(rel).name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = resource_path(rel)
            if src.exists():
                dst.write_text(src.read_text())
        return dst

    @staticmethod
    def _locate_demo():
        """Self-contained playback demo (clips + recorded alerts) shipped in the
        bundle, so the app shows the live wall + alert system on any machine
        without the engine/Ollama/clips installed."""
        cands = []
        if getattr(sys, "frozen", False):
            cands.append(Path(getattr(sys, "_MEIPASS", ".")) / "demo_data")
        cands.append(Path(__file__).resolve().parents[2] / "packaging" / "demo_data")
        for c in cands:
            if (c / "events.db").exists():
                return c
        return None

    # --- cameras (delegate to onboarding) ---
    def list_cameras(self) -> list[dict]:
        return onboarding.list_cameras(self.site_path)

    def scan(self, cidr: str) -> dict:
        return {"hosts": onboarding.scan_subnet(cidr)}

    def detect_subnet(self) -> dict:
        return {"cidr": onboarding.detect_subnet()}

    def test(self, url: str) -> dict:
        """Stream test with a NAME for every failure (EP-05-T4).

        For RTSP, a raw protocol probe first: it distinguishes unreachable /
        wrong credentials / wrong path / unsupported codec — cv2 collapses all
        four into "could not open". Only a passing probe pays for the cv2 open
        that produces the preview snapshot."""
        if str(url).startswith("rtsp"):
            from cvti.serving import discovery
            probe = discovery.probe_rtsp(url)
            if not probe["ok"]:
                return {"ok": False, "error": probe["message"], "kind": probe["kind"]}
            out = onboarding.test_url(url)
            if out.get("error"):
                out.setdefault("kind", "open-failed")
            else:
                out["codec"] = probe.get("codec")
            return out
        return onboarding.test_url(url)

    def discover_cameras(self) -> dict:
        """ONVIF WS-Discovery sweep of the local segment (EP-05-T4)."""
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.serving import discovery
        cams = discovery.discover()
        return {"cameras": cams, "count": len(cams)}

    def add_camera(self, camera: dict) -> list[dict]:
        self._require(perms.CONFIGURE_CAMERAS)
        self._close_preview()
        return onboarding.add_camera(self.site_path, camera)

    def remove_camera(self, camera_id: str) -> list[dict]:
        self._require(perms.CONFIGURE_CAMERAS)
        self._close_preview()
        return onboarding.remove_camera(self.site_path, camera_id)

    def presets(self) -> dict:
        return onboarding.RULE_PRESETS

    # Detectors the operator can toggle per camera (drive which models run).
    # Must stay in sync with PerCameraState's flags in cvti/serving/camera.py.
    RULE_FLAGS = ("concealment", "video_action", "violence", "weapons", "theft", "tamper",
                  "fire_smoke", "running", "crowd_formation", "fall", "normal_movement",
                  "multiple_people_moving", "object_watch")

    # Tuning params a detector needs to behave sensibly. Applied when it is first
    # switched on so a toggle "just works" without hand-editing the site config;
    # never overwrite a value the operator has already set.
    DETECTOR_DEFAULTS = {
        "running": {"running_min_speed_ratio": 0.08, "running_min_frames": 3},
        "crowd_formation": {"crowd_min_people": 5, "crowd_min_frames": 3,
                            "crowd_max_cluster_ratio": 0.32},
        "normal_movement": {"movement_enter_speed_ratio": 0.05,
                            "movement_exit_speed_ratio": 0.02,
                            "movement_min_track_seconds": 0.4},
        "multiple_people_moving": {"movement_enter_speed_ratio": 0.05,
                                   "movement_exit_speed_ratio": 0.02,
                                   "movement_min_track_seconds": 0.4,
                                   "movement_min_people": 2,
                                   "movement_persistence_seconds": 0.5},
    }

    # --- first-run use-case templates (EP-05-T3) ---------------------------
    # Where the CustomizationEngine's flags finally meet the interface: a
    # non-technical installer picks what the site IS, not which of ten
    # detectors to enable. Keys are RULE_FLAGS; anything absent is OFF.
    USE_CASE_TEMPLATES = {
        "retail": {
            "label": "Retail / Store",
            "blurb": "Shoplifting, concealment, weapons and fire — the shop-floor set.",
            "detectors": {"concealment": True, "video_action": True, "theft": True,
                          "weapons": True, "violence": True, "tamper": True,
                          "fire_smoke": True},
            "config": "configs/all_threats_video_v1.json",
        },
        "warehouse": {
            "label": "Warehouse / HSE",
            "blurb": "People down, fire, panic and crowding — safety first, theft off.",
            "detectors": {"fall": True, "fire_smoke": True, "running": True,
                          "crowd_formation": True, "tamper": True, "weapons": True},
            "config": "configs/all_threats_v1.json",
        },
        "office": {
            "label": "Office",
            "blurb": "After-hours intrusion essentials: violence, weapons, fire, tamper.",
            "detectors": {"violence": True, "weapons": True, "fire_smoke": True,
                          "tamper": True},
            "config": "configs/all_threats_v1.json",
        },
    }

    # Validation status per detector flag (EP-07-T4). Mirrors the coverage
    # table in docs/NUMBERS.md — a test asserts the two agree, so this cannot
    # quietly drift from the published numbers. A detector without a measured
    # row is EXPERIMENTAL at the point of configuration, and excluded from
    # marketing claims (NUMBERS.md is the claims sheet).
    DETECTOR_VALIDATION = {
        "fire_smoke": {"measured": True,
                       "summary": "100% caught (n=9) · 6.7% false alarms"},
        "crowd_formation": {"measured": True,
                            "summary": "75% caught (n=8) · 13.3% false alarms"},
        "concealment": {"measured": True,
                        "summary": "88.9% caught (n=9) · 25.9% false alarms"},
        "video_action": {"measured": True,
                         "summary": "88.9% caught (n=9) · 25.9% false alarms"},
        "running": {"measured": False}, "fall": {"measured": False},
        "normal_movement": {"measured": False},
        "multiple_people_moving": {"measured": False},
        "object_watch": {"measured": False},
        "weapons": {"measured": False}, "violence": {"measured": False},
        "tamper": {"measured": False}, "theft": {"measured": False},
    }

    def detector_validation(self) -> dict:
        """Per-detector measurement status for the configuration surfaces."""
        out = {}
        for flag in self.RULE_FLAGS:
            v = self.DETECTOR_VALIDATION.get(flag, {"measured": False})
            out[flag] = {"measured": bool(v.get("measured")),
                         "summary": v.get("summary") or "EXPERIMENTAL — built and demonstrable, not yet validated"}
        return out

    def use_case_templates(self) -> dict:
        """The three templates, detector labels resolved — read-only, any role."""
        return {k: {"label": t["label"], "blurb": t["blurb"],
                    "detectors": {f: bool(t["detectors"].get(f)) for f in self.RULE_FLAGS}}
                for k, t in self.USE_CASE_TEMPLATES.items()}

    def apply_template(self, key: str) -> dict:
        """Apply a use-case template to EVERY configured camera.

        Sets each RULE_FLAG explicitly (on or off) so switching templates is
        deterministic, then lets set_camera_rules seed detector defaults for
        anything newly enabled. Per-camera fine-tuning stays possible after —
        the template is a starting point, not a lock.
        """
        self._require(perms.CONFIGURE_DETECTORS)
        tpl = self.USE_CASE_TEMPLATES.get(key)
        if tpl is None:
            return {"error": f"unknown template '{key}'"}
        rules = {f: bool(tpl["detectors"].get(f)) for f in self.RULE_FLAGS}
        rules["config"] = tpl["config"]
        cams = onboarding.list_cameras(self.site_path)
        for cam in cams:
            self.set_camera_rules(cam["id"], rules)
        self.audit.record(self.current_user.username, "config_change",
                          "site:template", detail={"template": key, "cameras": len(cams)})
        return {"ok": True, "template": key, "cameras": len(cams),
                "detectors_on": sorted(k for k, v in tpl["detectors"].items() if v)}

    # --- setup self-test (EP-05-T3) -----------------------------------------
    @staticmethod
    def _probe_stream(source) -> tuple:
        """(ok, detail) — fast reachability, never a blocking cv2 open.

        A wrong-credentials RTSP URL still connects at the TCP layer; the full
        stream test is the wizard's per-camera Test button. This check answers
        the self-test's question — is anything answering at all — in <1s.
        """
        import socket
        from urllib.parse import urlparse
        src = str(source)
        if src.isdigit():
            return True, "built-in webcam"
        if "://" not in src:
            return (True, "video file on disk") if Path(src).exists()                 else (False, f"file not found: {src}")
        u = urlparse(src)
        port = u.port or {"rtsp": 554, "http": 80, "https": 443}.get(u.scheme, 554)
        try:
            with socket.create_connection((u.hostname, port), timeout=1.5):
                return True, f"{u.hostname}:{port} answers"
        except OSError as exc:
            return False, f"{u.hostname}:{port} unreachable ({exc})"

    def setup_check(self) -> list:
        """Every component, verified, in plain English (EP-05-T3 acceptance:
        'names exactly what is missing'). Each item: ok True/False/None(warn),
        what was checked, what was found, and the one action that fixes it."""
        checks = []

        cams = [c for c in onboarding.list_cameras(self.site_path) if c.get("source")]
        checks.append({"id": "cameras", "ok": bool(cams),
                       "label": "Cameras configured",
                       "detail": f"{len(cams)} camera(s) in the site config" if cams
                       else "no cameras yet",
                       "fix": None if cams else "Add a camera in the Cameras step."})

        for c in cams[:8]:
            ok, detail = self._probe_stream(c.get("source"))
            checks.append({"id": f"stream:{c['id']}", "ok": ok,
                           "label": f"Camera '{c['id']}' reachable",
                           "detail": detail,
                           "fix": None if ok else "Check the camera's power, network "
                           "cable and IP address, then use Test on the Cameras step."})

        g = self.gate_status()
        if g.get("cloud"):
            label = g.get("label") or "cloud verifier"
            if g.get("mode") == "live":
                checks.append({"id": "verifier", "ok": True, "label": "AI verifier (TrueSight)",
                               "detail": f"cloud: {label}, model {g.get('model')}, key saved",
                               "fix": None})
            else:
                checks.append({"id": "verifier", "ok": False, "label": "AI verifier (TrueSight)",
                               "detail": f"cloud: {label} is selected but no API key is saved",
                               "fix": "Enter the API key under Settings → AI verification, "
                                      "then press Test."})
        elif g.get("mode") == "live":
            checks.append({"id": "verifier", "ok": True, "label": "AI verifier (TrueSight)",
                           "detail": "running, model installed", "fix": None})
        elif g.get("mode") == "no-model":
            checks.append({"id": "verifier", "ok": False, "label": "AI verifier (TrueSight)",
                           "detail": "runtime is up but the vision model is not downloaded",
                           "fix": "Download the model in the Verification step (~3.3 GB, resumes)."})
        else:
            checks.append({"id": "verifier", "ok": False, "label": "AI verifier (TrueSight)",
                           "detail": "not running",
                           # Naming a control that exists: the Verification
                           # step's Download button starts the bundled runtime
                           # before it pulls, so one action covers both.
                           "fix": "Click Download model in the Verification step "
                                  "— it starts the bundled AI runtime first."
                           if g.get("runtime_bundled")
                           else "Install Ollama from ollama.com, then recheck."})

        det = resource_path("models/yolov8n.pt")
        checks.append({"id": "detector", "ok": det.exists(), "label": "Detection models",
                       "detail": "YOLO weights present" if det.exists()
                       else "models/yolov8n.pt is missing from this install",
                       "fix": None if det.exists() else "Reinstall Argus — the bundle is incomplete."})
        vm = resource_path("runs/video_finetune/videomae")
        if not vm.exists():
            checks.append({"id": "video_model", "ok": None, "label": "Video action model",
                           "detail": "VideoMAE fine-tune not bundled — video theft "
                           "detection will be off; everything else still runs",
                           "fix": None})

        meta = onboarding.get_site_meta(self.site_path)
        notify = (meta.get("notify") or "console").strip()
        if notify and notify != "console":
            checks.append({"id": "notify", "ok": True, "label": "Notifications",
                           "detail": f"alerts go to {notify}",
                           "fix": "Confirm delivery with the test alert step."})
        else:
            checks.append({"id": "notify", "ok": None, "label": "Notifications",
                           "detail": "alerts only appear inside the app (console)",
                           "fix": "Pick WhatsApp/Telegram/webhook in Site settings "
                           "to be reached when you're not at this screen."})

        import shutil as _shutil
        try:
            du = _shutil.disk_usage(str(Path(self.db_path).parent))
            pct = du.used * 100.0 / max(1, du.total)
            free_gb = du.free / 1e9
            low = free_gb < 5
            checks.append({"id": "disk", "ok": (None if low else True) if free_gb >= 1 else False,
                           "label": "Disk space for evidence",
                           "detail": f"{free_gb:.0f} GB free ({pct:.0f}% used)",
                           "fix": "Free some space — evidence recording needs room."
                           if low else None})
        except OSError:
            checks.append({"id": "disk", "ok": None, "label": "Disk space for evidence",
                           "detail": "could not be measured", "fix": None})
        return checks

    def set_camera_rules(self, camera_id: str, rules: dict) -> dict:
        """Update which threat detectors run on a camera (+ optional rule preset).
        Takes effect on the next Start monitoring."""
        self._require(perms.CONFIGURE_DETECTORS)
        cams = onboarding.list_cameras(self.site_path)
        cam = next((c for c in cams if c.get("id") == camera_id), None)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        for k in self.RULE_FLAGS:
            if k in rules:
                turning_on = bool(rules[k]) and not cam.get(k)
                cam[k] = bool(rules[k])
                if k == "object_watch":
                    # CameraRuntime prefers the newer flag when it is present.
                    # Keep the legacy API toggle and its persisted alias atomic
                    # so disabling object watch cannot leave it effectively on.
                    cam["object_watch_enabled"] = bool(rules[k])
                if turning_on:      # seed this detector's tuning params (don't clobber)
                    for pk, pv in self.DETECTOR_DEFAULTS.get(k, {}).items():
                        cam.setdefault(pk, pv)
        if rules.get("config"):
            # A preset choice is a new BASE, not a raw config pointer: cameras
            # with zones or an English rule keep them — regen layers the base
            # under the generated rules instead of clobbering the pointer.
            # (Audit 23 Aug, #9: applying a template used to silently drop the
            # camera's loitering + custom-English rules, and the next zone edit
            # reverted the camera to the pre-template preset.)
            cam["_base_config"] = rules["config"]
            zones = self.list_zones(camera_id)
            if zones or self._custom_rules(cam) or cam.get("object_watch_rules"):
                self._regen_zone_rules(camera_id, cam, zones)
            else:
                cam["config"] = rules["config"]
        onboarding.add_camera(self.site_path, cam)   # upsert by id
        return {"ok": True, "camera": cam}

    # --- object watchlists -------------------------------------------------
    def _object_library_root(self) -> Path:
        """Site-local object library. Stored beside events.db, not in Git."""
        return Path(self.db_path).parent

    @staticmethod
    def _redact_object_example(example) -> dict:
        return {
            "id": example.id,
            "source": example.source,
            "bbox": list(example.bbox),
            "sha256": example.sha256,
            "reviewed": bool(example.reviewed),
            "bbox_format": example.bbox_format,
        }

    @classmethod
    def _redact_object_target(cls, target) -> dict:
        return {
            "id": target.id,
            "label": target.label,
            "category": target.category,
            "aliases": list(target.aliases),
            "review_state": target.review_state,
            "min_similarity": target.min_similarity,
            "allowed_zone_ids": list(target.allowed_zone_ids),
            "grounding_description": target.grounding_description,
            "examples": [cls._redact_object_example(e) for e in target.examples],
            "negative_examples": [
                cls._redact_object_example(e) for e in target.negative_examples
            ],
        }

    def object_targets(self) -> dict:
        self._require(perms.VIEW_LIVE)
        from types import SimpleNamespace
        from cvti.object_watch.runtime_config import preflight, resolve_config
        from cvti.object_watch.store import load_targets, target_readiness

        config = resolve_config(self._object_library_root())
        readiness = preflight(config)
        runtime = {
            "status": readiness.status,
            "backend": readiness.backend,
            "fingerprint": readiness.model_fingerprint,
            "reason_codes": [self._object_watch_reason_code(v) for v in readiness.reasons],
        }
        descriptor = None
        if readiness.model_fingerprint:
            descriptor = SimpleNamespace(
                name=config.backend, fingerprint=readiness.model_fingerprint,
                preprocessing_version=1,
            )
        targets = []
        for target in load_targets(self._object_library_root()):
            row = self._redact_object_target(target)
            reasons = list(runtime["reason_codes"])
            if descriptor is not None:
                target_status = target_readiness(self._object_library_root(), target, descriptor)
                reasons.extend(self._object_watch_reason_code(v) for v in target_status.reasons)
            row["can_activate"] = not reasons
            row["reasons"] = list(dict.fromkeys(reasons))
            targets.append(row)
        return {"runtime": runtime, "targets": targets}

    @staticmethod
    def _object_watch_reason_code(reason: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")

    def create_object_target(
        self,
        object_id: str,
        label: str,
        category: str,
        aliases: list[str] | None = None,
        min_similarity: float = 0.72,
        allowed_zone_ids: list[str] | None = None,
        grounding_description: str = "",
    ) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.object_watch.store import ObjectTarget, save_target

        saved = save_target(self._object_library_root(), ObjectTarget(
            id=str(object_id).strip(), label=str(label).strip(),
            category=str(category).strip(), aliases=tuple(aliases or ()),
            review_state="draft", min_similarity=float(min_similarity),
            allowed_zone_ids=tuple(allowed_zone_ids or ()),
            examples=(),
            negative_examples=(),
            grounding_description=str(grounding_description or "").strip(),
        ))
        self.audit.record(
            self._actor(), "config_change", f"object_target:{saved.id}",
            {"action": "create", "label": saved.label, "category": saved.category},
        )
        return {"target": self._redact_object_target(saved)}

    @staticmethod
    def _decode_example_image(image_b64: str) -> bytes:
        raw = str(image_b64 or "")
        if "," in raw and raw.split(",", 1)[0].startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_b64 must be valid base64 image data") from exc
        if not data or len(data) > OBJECT_EXAMPLE_MAX_BYTES:
            raise ValueError("image is empty or exceeds the upload size limit")
        if not (
            data.startswith(b"\xff\xd8\xff")
            or data.startswith(b"\x89PNG\r\n\x1a\n")
            or data.startswith(b"RIFF") and data[8:12] == b"WEBP"
        ):
            raise ValueError("image_b64 must contain a JPEG, PNG, or WebP image")
        return data

    def add_object_example(
        self,
        object_id: str,
        image_b64: str,
        bbox: list[float | int],
        source: str,
        bbox_format: str = "legacy",
        negative: bool = False,
    ) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.object_watch.store import add_example, load_targets

        image_bytes = self._decode_example_image(image_b64)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must have four coordinates")
        # Keep normalized coordinates as floats until the image helper has
        # interpreted their declared format. Converting 0.75 to 0 here used to
        # turn valid HTTP crops into empty/full-image legacy boxes.
        raw_bbox = tuple(float(v) for v in bbox)
        example = add_example(
            self._object_library_root(),
            object_id,
            image_bytes,
            (raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]),
            str(source or "upload"),
            bbox_format=str(bbox_format or "legacy"),
            negative=bool(negative),
            reviewed=False,
        )
        target = next(t for t in load_targets(self._object_library_root()) if t.id == object_id)
        self.audit.record(
            self._actor(), "config_change", f"object_target:{object_id}",
            {"action": "add_example", "label": target.label,
             "example_id": example.id, "sha256": example.sha256},
        )
        return {
            "target": self._redact_object_target(target),
            "example": self._redact_object_example(example),
        }

    def review_object_example(
        self, object_id: str, example_id: str, reviewed: bool = True,
    ) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.object_watch.store import load_targets, review_example

        example = review_example(
            self._object_library_root(), object_id, example_id, reviewed=bool(reviewed),
        )
        target = next(t for t in load_targets(self._object_library_root()) if t.id == object_id)
        self.audit.record(
            self._actor(), "config_change", f"object_target:{object_id}",
            {"action": "review_example", "example_id": example_id,
             "reviewed": bool(reviewed)},
        )
        return {"target": self._redact_object_target(target),
                "example": self._redact_object_example(example)}

    def object_example_preview(self, object_id: str, example_id: str) -> dict:
        self._require(perms.VIEW_LIVE)
        from cvti.object_watch.store import load_targets

        target = next((t for t in load_targets(self._object_library_root())
                       if t.id == object_id), None)
        if target is None:
            raise ValueError(f"unknown object target: {object_id}")
        example = next((e for e in target.examples + target.negative_examples
                        if e.id == example_id), None)
        if example is None:
            raise ValueError(f"unknown object example: {example_id}")
        library = (self._object_library_root() / "object_library").resolve()
        path = (library / example.path).resolve()
        if library not in path.parents or not path.is_file() or path.suffix.lower() != ".png":
            raise ValueError("object example preview path is invalid")
        data = path.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("object example preview is not a canonical crop")
        return {"object_id": object_id, "example_id": example_id,
                "mime_type": "image/png", "image_b64": base64.b64encode(data).decode()}

    def _object_watch_backend(self):
        """Load only the configured local backend; tests may replace this seam."""
        from cvti.object_watch.runtime_config import load_configured_backend, resolve_config
        try:
            return load_configured_backend(resolve_config(self._object_library_root()))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    def _object_watch_metadata(self):
        """Return the configured model descriptor without constructing torch."""
        from cvti.object_watch.runtime_config import configured_backend_metadata, resolve_config

        config = resolve_config(self._object_library_root())
        try:
            return configured_backend_metadata(config)
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc

    def activate_object_target(self, object_id: str) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.object_watch.store import activate_target, load_targets

        target = next((item for item in load_targets(self._object_library_root())
                       if item.id == object_id), None)
        if target is None:
            raise ValueError(f"unknown object target: {object_id}")
        if not any(example.reviewed for example in target.examples):
            raise ValueError("active target requires at least one reviewed example")
        metadata = self._object_watch_metadata()
        saved = activate_target(self._object_library_root(), object_id, metadata)
        self.audit.record(
            self._actor(), "config_change", f"object_target:{saved.id}",
            {"action": "activate", "label": saved.label},
        )
        return {"target": self._redact_object_target(saved)}

    def deactivate_object_target(self, object_id: str) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.object_watch.store import deactivate_target

        saved = deactivate_target(self._object_library_root(), object_id)
        self.audit.record(self._actor(), "config_change", f"object_target:{saved.id}",
                          {"action": "deactivate", "label": saved.label})
        return {"target": self._redact_object_target(saved)}

    def reembed_object_targets(self, model: str | None = None) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.object_watch.runtime_config import resolve_config
        from cvti.object_watch.enrollment_jobs import JOBS

        config = resolve_config(self._object_library_root())
        requested = str(model or config.backend).strip().lower()
        if requested == "hash" or requested != config.backend:
            raise ValueError("reembedding must use the configured production backend")
        queued = JOBS.enqueue(self._object_library_root(), config)
        self.audit.record(
            self._actor(), "config_change", "object_targets",
            {"action": "reembed_queued", "model": config.backend,
             "job_id": queued["job_id"]},
        )
        return queued

    def object_watch_job_status(self, job_id: str) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.object_watch.enrollment_jobs import JOBS

        return JOBS.status(self._object_library_root(), job_id)

    def set_object_watch_runtime_config(self, config: dict) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        if not isinstance(config, dict):
            raise ValueError("object watch config must be an object")
        allowed = {"backend", "model_path", "device", "world_weights", "clip_weights",
                   "sample_fps", "max_candidates", "result_ttl_seconds"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError("unknown object watch config fields: " + ", ".join(sorted(unknown)))
        from dataclasses import replace
        from cvti.object_watch.runtime_config import resolve_config, write_config

        current = resolve_config(self._object_library_root())
        backend = str(config.get("backend", current.backend)).strip().lower()
        if backend != "siglip":
            raise ValueError("object watch backend must be siglip; hash is test-only")
        root = self._object_library_root().resolve()

        def local_path(key, current_value):
            value = config.get(key, current_value)
            if value in (None, ""):
                return None
            raw = str(value)
            if "://" in raw or raw.lower().startswith(("http:/", "https:/")):
                raise ValueError("model paths must be local filesystem paths")
            path = Path(raw).expanduser()
            return (path if path.is_absolute() else root / path).resolve()

        sample_fps = float(config.get("sample_fps", current.sample_fps))
        ttl = float(config.get("result_ttl_seconds", current.result_ttl_seconds))
        maximum = config.get("max_candidates", current.max_candidates)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("object watch max_candidates must be a positive integer")
        import math
        if not math.isfinite(sample_fps) or not math.isfinite(ttl) or sample_fps <= 0 or ttl <= 0:
            raise ValueError("object watch runtime limits must be positive")
        candidate = replace(
            current, backend=backend,
            model_path=local_path("model_path", current.model_path),
            world_weights=local_path("world_weights", current.world_weights),
            clip_weights=local_path("clip_weights", current.clip_weights),
            device=str(config.get("device", current.device)).strip() or "cpu",
            sample_fps=sample_fps, max_candidates=maximum, result_ttl_seconds=ttl,
            library_path=None,
        )
        path = write_config(self._object_library_root(), candidate)
        self.audit.record(self._actor(), "config_change", "object_watch_runtime",
                          {"fields": sorted(config), "path": str(path)})
        return self.object_targets()

    def set_object_watch_rule(
        self, camera_id: str, object_id: str, enabled: bool,
        zone_id: str | None = None,
    ) -> dict:
        self._require(perms.CONFIGURE_DETECTORS)
        from cvti.object_watch.store import load_targets

        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            raise ValueError(f"camera '{camera_id}' not found")
        if not any(target.id == object_id for target in load_targets(self._object_library_root())):
            raise ValueError(f"unknown object target: {object_id}")
        if zone_id is not None and zone_id not in {z.get("name") for z in self.list_zones(camera_id)}:
            raise ValueError(f"zone '{zone_id}' not found for camera '{camera_id}'")
        key = f"object_watch:{object_id}:{zone_id or '*'}"
        rules = [r for r in (cam.get("object_watch_rules") or []) if r.get("key") != key]
        if enabled:
            trigger = {"detector": "object_watch", "state": "object_seen",
                       "object_id": object_id}
            if zone_id is not None:
                trigger["zone"] = zone_id
            rules.append({"key": key, "name": key.replace(":", "_"),
                          "trigger": trigger, "priority": "medium"})
        cam["object_watch_rules"] = rules
        cam["object_watch"] = bool(rules)
        cam["object_watch_enabled"] = bool(rules)
        cam["object_watch_library"] = str(
            (self._object_library_root() / "object_library").resolve())
        self._regen_zone_rules(camera_id, cam, self.list_zones(camera_id))
        onboarding.add_camera(self.site_path, cam)
        self.audit.record(self._actor(), "config_change", f"camera:{camera_id}",
                          {"object_watch_rule": key, "enabled": bool(enabled)})
        return {"ok": True, "camera_id": camera_id, "object_id": object_id,
                "zone_id": zone_id, "enabled": bool(enabled), "rules": rules}

    def registered_objects(self, camera_id: str) -> list:
        self._require(perms.VIEW_LIVE)
        cam = self._cam(onboarding.list_cameras(self.site_path), camera_id)
        if cam is None:
            raise ValueError("camera not found")
        result = []
        for entry in cam.get("registered_objects", []):
            state = "pending_monitoring"
            reason = None
            status_path = Path(entry["reference_path"]).with_suffix(".status.json")
            if status_path.exists():
                try:
                    status = json.loads(status_path.read_text())
                    state = status["state"]
                    reason = status.get("reason")
                except (OSError, ValueError, KeyError):
                    state = "status_unavailable"
            result.append({k: v for k, v in entry.items() if k not in {"reference_path", "source_fingerprint"}} | {"state": state, "reason": reason})
        return result

    def register_object_region(self, camera_id: str, name: str, region: list,
                               frame_hw: list, confirm_seconds: float = 8) -> dict:
        self._require(perms.CONFIGURE_DETECTORS)
        import math
        import uuid
        from cvti.serving.registered_objects import source_key
        cam = self._cam(onboarding.list_cameras(self.site_path), camera_id)
        if cam is None:
            raise ValueError("camera not found")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("object name must be 1-80 characters")
        if (not isinstance(frame_hw, list) or len(frame_hw) != 2 or
                any(type(v) is not int or not 1 <= v <= 16384 for v in frame_hw)):
            raise ValueError("invalid frame dimensions")
        if (not isinstance(region, list) or len(region) != 4 or
                any(type(v) is not int for v in region)):
            raise ValueError("region requires four integer coordinates")
        x1, y1, x2, y2 = region
        h, w = frame_hw
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and x2-x1 >= 12 and y2-y1 >= 12):
            raise ValueError("region must be at least 12 pixels across and inside the frame")
        if isinstance(confirm_seconds, bool) or not isinstance(confirm_seconds, (int, float)) or not math.isfinite(confirm_seconds) or not 2 <= confirm_seconds <= 300:
            raise ValueError("confirmation time must be 2-300 seconds")
        entries = cam.get("registered_objects", [])
        if len(entries) >= 16:
            raise ValueError("maximum 16 registered objects per camera")
        key = uuid.uuid4().hex
        root = Path(self.site_path).resolve().parent / "registered_objects"
        entry = {"id": key, "name": name.strip(), "region": region,
                 "frame_hw": frame_hw, "confirm_seconds": confirm_seconds,
                 "source_fingerprint": source_key(cam["source"]),
                 "reference_path": str(root / (key + ".npz"))}
        cam["registered_objects"] = [*entries, entry]
        onboarding.add_camera(self.site_path, cam)
        self.audit.record(self._actor(), "config_change", f"camera:{camera_id}", {"registered_object": key})
        return {"ok": True, "id": key}

    def remove_registered_object(self, camera_id: str, object_id: str) -> dict:
        self._require(perms.CONFIGURE_DETECTORS)
        cam = self._cam(onboarding.list_cameras(self.site_path), camera_id)
        if cam is None:
            raise ValueError("camera not found")
        cam["registered_objects"] = [e for e in cam.get("registered_objects", []) if e["id"] != object_id]
        onboarding.add_camera(self.site_path, cam)
        self.audit.record(self._actor(), "config_change", f"camera:{camera_id}", {"removed_registered_object": object_id})
        return {"ok": True}

    def recapture_registered_object(self, camera_id: str, object_id: str) -> dict:
        self._require(perms.CONFIGURE_DETECTORS)
        import uuid
        from cvti.serving.registered_objects import source_key
        cam = self._cam(onboarding.list_cameras(self.site_path), camera_id)
        entry = next((e for e in (cam or {}).get("registered_objects", []) if e["id"] == object_id), None)
        if entry is None:
            raise ValueError("registered object not found")
        if entry["source_fingerprint"] != source_key(cam["source"]):
            raise ValueError("camera source changed; draw a new object region")
        entry["reference_path"] = str(Path(self.site_path).resolve().parent / "registered_objects" / (uuid.uuid4().hex + ".npz"))
        onboarding.add_camera(self.site_path, cam)
        self.audit.record(self._actor(), "config_change", f"camera:{camera_id}", {"recapture_registered_object": object_id})
        return {"ok": True}

    # --- zones (draw in-app -> geometry + a loitering rule the engine runs) ---
    def camera_snapshot(self, camera_id: str) -> dict:
        """A still from the camera to draw zones on — plus its ORIGINAL pixel
        size so the UI can map canvas coords back to real zone coordinates."""
        import cv2
        monitoring = self._monitor is not None and self._monitor.poll() is None
        if not monitoring:
            result = self.camera_preview(camera_id)
            if result.get("error"):
                return result
            preview = self._preview
            if not preview.acquire(camera_id):
                return {"error": "preview is closing; retry shortly"}
            try:
                for _ in range(60):
                    snapshot = preview.snapshot(camera_id)
                    if snapshot and "uri" in snapshot:
                        return snapshot
                    time.sleep(0.1)
                return {"error": "could not read a frame from this camera"}
            finally:
                preview.release(camera_id)
        preview = getattr(self, "_preview", None)
        if preview is not None:
            snapshot = preview.snapshot(camera_id)
            if snapshot is not None:
                return snapshot
        cam = next((c for c in self.list_cameras() if c.get("id") == camera_id), None)
        if cam is None or not cam.get("source"):
            return {"error": "camera not found"}
        src = cam["source"]
        cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        try:
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if n > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.3))
            ok, fr = cap.read()
        finally:
            cap.release()
        if not ok:
            return {"error": "could not read a frame from this camera"}
        h, w = fr.shape[:2]
        ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok2:
            return {"error": "encode failed"}
        return {"uri": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                "w": int(w), "h": int(h)}

    def _zones_file(self, camera_id: str) -> Path:
        if getattr(sys, "frozen", False):
            from cvti.utils import user_data_dir
            d = user_data_dir() / "zones"
            d.mkdir(parents=True, exist_ok=True)
            return d / f"{camera_id}.json"
        return Path("configs/zones") / f"{camera_id}.json"

    def list_zones(self, camera_id: str) -> list[dict]:
        f = self._zones_file(camera_id)
        if f.exists():
            try:
                return json.loads(f.read_text()).get("zones", [])
            except (ValueError, OSError):
                return []
        return []

    def _regen_zone_rules(self, camera_id: str, cam: dict, zones: list[dict]) -> None:
        """Per-camera rule config = the camera's base preset + one loitering rule
        per zone. Keeps the shared preset untouched."""
        base_cfg = cam.get("_base_config") or cam.get("config") or "configs/all_threats_v1.json"
        cam["_base_config"] = base_cfg
        # A relative preset path resolves against the CWD — the repo when
        # developing, anywhere at all on an install. The silent except below
        # then produced a rules file with NO baseline rules and nothing said so.
        base_path = Path(base_cfg)
        if not base_path.exists():
            base_path = resource_path(base_cfg)
        try:
            rules = list(json.loads(base_path.read_text()).get("rules", []))
        except (ValueError, OSError):
            rules = []
        for z in zones:
            dw = z.get("dwell_alert_seconds", 5)
            rules.append({"name": f"loitering_{z['name']}", "trigger": {"detector": "presence"},
                          "context_filter": f"zone == '{z['name']}' and dwell_seconds >= {dw}",
                          "priority": "medium"})
        # A tripwire without rules is a detector whose events nothing listens
        # for, which the engine discards in silence — the same way crowd and
        # panic-running went unnoticed (see configs/baseline_critical_v1.json).
        # No shipped preset carries vehicle rules, so the camera's own config
        # grows them the moment a line exists, and loses them when it goes.
        if cam.get("vehicle_line"):
            rules.append({"name": "vehicle_entered", "trigger": {"detector": "vehicle_entry"},
                          "priority": "high"})
            rules.append({"name": "vehicle_exited", "trigger": {"detector": "vehicle_exit"},
                          "priority": "medium"})
        # Keep scoped object-watch rules through every zone/preset regeneration.
        for object_rule in cam.get("object_watch_rules") or []:
            rules.append({k: v for k, v in object_rule.items() if k != "key"})
        # English rules do NOT regenerate into presence-gated rules any more:
        # that path only fired when a PERSON lingered, so "detect the white
        # aeroplane" never ran (user report, 23 Aug). The engine's
        # CustomRuleScanner is the single path — it reads cam["custom_rules"]
        # from the site file directly, scans every ~12s person-or-not, and
        # hot-picks-up new sentences within one cycle.
        # Same rule as _zones_file: generated per-camera configs are USER data.
        # Path('configs/rules') on an install is the bundle dir — saving a zone
        # on the pilot's Windows machine died with [WinError 5] Access is
        # denied: 'configs' (field screenshot, 29 Aug).
        if getattr(sys, "frozen", False):
            from cvti.utils import user_data_dir
            rdir = user_data_dir() / "rules"
        else:
            rdir = Path("configs/rules")
        rdir.mkdir(parents=True, exist_ok=True)
        rfile = rdir / f"{camera_id}.json"
        rfile.write_text(json.dumps({"use_case_id": f"{camera_id}_zones", "rules": rules}, indent=2))
        cam["config"] = str(rfile)

    @staticmethod
    def _custom_rules(cam: dict) -> list:
        """The camera's English rules, migrating the legacy single-rule field."""
        rules = list(cam.get("custom_rules") or [])
        legacy = cam.get("custom_rule")
        if legacy and legacy.get("question") and \
                legacy["question"] not in [r.get("question") for r in rules]:
            rules.insert(0, legacy)
        return [r for r in rules if (r.get("question") or "").strip()]

    def _save_custom_rules(self, camera_id: str, cam: dict, rules: list) -> dict:
        cam["custom_rules"] = rules
        cam.pop("custom_rule", None)               # migrated to the list
        f = self._zones_file(camera_id)
        data = json.loads(f.read_text()) if f.exists() else {"zones": []}
        if rules and not data["zones"]:
            # Without a zone the presence detector never runs and the sentences
            # would sit there looking configured while detecting nothing.
            f.parent.mkdir(parents=True, exist_ok=True)
            data["zones"] = [{"name": "everywhere", "kind": "restricted",
                              "anchors": ["BOTTOM_CENTER"], "dwell_alert_seconds": 999999,
                              "polygon": [[0, 0], [99999, 0], [99999, 99999], [0, 99999]]}]
            f.write_text(json.dumps(data, indent=2))
            cam["zones"] = str(f)
        self._regen_zone_rules(camera_id, cam, data["zones"])
        onboarding.add_camera(self.site_path, cam)
        return {"ok": True, "rules": [r["question"] for r in rules],
                "note": "scanned about every 12s, person or not — live within one scan"}

    def add_custom_rule(self, camera_id: str, question: str, dwell: float = 4.0) -> dict:
        """Add one plain-English rule to a camera. Sentences accumulate — a new
        description never overwrites an old one. While monitoring runs, the
        engine hot-reloads the camera's rules within seconds; no restart."""
        self._require(perms.CONFIGURE_DETECTORS)
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        question = (question or "").strip()
        if not question:
            return {"error": "describe the event in a sentence first"}
        rules = self._custom_rules(cam)
        if question in [r["question"] for r in rules]:
            return {"ok": True, "rules": [r["question"] for r in rules],
                    "note": "already watching for that"}
        rules.append({"question": question, "dwell": float(dwell)})
        out = self._save_custom_rules(camera_id, cam, rules)
        self.audit.record(self.current_user.username, "config_change",
                          f"camera:{camera_id}", detail={"custom_rule_added": question[:120]})
        # Say which engine will answer, at the moment of writing (W3): an
        # appearance sentence goes to the grounded detector — which needs
        # subjects near the camera — while behaviour stays with the VLM.
        # Dublin, 10 Sep: a hoodie rule on a far 360p street camera sat
        # silent for good physical reasons nobody had told the author about.
        try:
            from cvti.detector.openvocab import route_rule
            if isinstance(out, dict) and out.get("ok"):
                engine = ("yolo-world" if route_rule(question) == "openvocab"
                          else "vlm")
                out["routing"] = engine
                if engine == "yolo-world":
                    out["routing_note"] = (
                        "appearance rule — answered by the grounded detector "
                        "in ~50ms, but it needs subjects reasonably close to "
                        "the camera (roughly 80+ pixels tall); on far "
                        "wide-angle cameras it may honestly never fire")
        except Exception:  # noqa: BLE001 - the note must never break the save
            log.debug("routing note failed", exc_info=True)
        return out

    def remove_custom_rule(self, camera_id: str, question: str) -> dict:
        self._require(perms.CONFIGURE_DETECTORS)
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        rules = [r for r in self._custom_rules(cam) if r["question"] != question]
        out = self._save_custom_rules(camera_id, cam, rules)
        self.audit.record(self.current_user.username, "config_change",
                          f"camera:{camera_id}", detail={"custom_rule_removed": question[:120]})
        return out

    def set_custom_rule(self, camera_id: str, question: str, dwell: float = 4.0) -> dict:
        """Legacy single-rule entry point (wizard-era). Adds when given a
        sentence; clears ALL English rules when given an empty one."""
        if (question or "").strip():
            return self.add_custom_rule(camera_id, question, dwell)
        self._require(perms.CONFIGURE_DETECTORS)
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        out = self._save_custom_rules(camera_id, cam, [])
        self.audit.record(self.current_user.username, "config_change",
                          f"camera:{camera_id}", detail={"custom_rule": "(all removed)"})
        return out

    def set_view_only(self, camera_id: str, enabled: bool) -> dict:
        """Glass mode per camera (4 Sep, pilot): stream the video, run NO
        detection on it. The engine gives such a camera a decoder and a wall
        slot and nothing else — no models, no scene mapping, no English scans
        — so a starved machine spends its cycles on the cameras that detect.
        Applies on the next engine start; if monitoring is running we restart
        it here so the toggle means what it says."""
        self._require(perms.CONFIGURE_DETECTORS)
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        if enabled:
            cam["view_only"] = True
        else:
            cam.pop("view_only", None)
        onboarding.add_camera(self.site_path, cam)      # upsert by id
        self.audit.record(self.current_user.username, "config_change",
                          f"camera:{camera_id}", detail={"view_only": bool(enabled)})
        restarted = False
        if self._monitor and self._monitor.poll() is None:
            self.stop_monitoring()
            restarted = bool(self.start_monitoring().get("running"))
        return {"ok": True, "view_only": bool(enabled), "engine_restarted": restarted}

    def add_zone(self, camera_id: str, name: str, points: list, dwell_seconds: float = 5.0) -> dict:
        """Save a drawn zone (>=3 [x,y] points in ORIGINAL pixels) + wire a
        loitering rule for it. The running engine hot-reloads it within seconds."""
        self._require(perms.CONFIGURE_CAMERAS)
        pts = [[int(p[0]), int(p[1])] for p in (points or []) if len(p) == 2]
        if len(pts) < 3:
            return {"error": "a zone needs at least 3 points"}
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        f = self._zones_file(camera_id)
        f.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(f.read_text()) if f.exists() else {"zones": []}
        data["zones"] = [z for z in data.get("zones", []) if z.get("name") != name]
        data["zones"].append({"name": name or "zone", "kind": "restricted",
                              "anchors": ["BOTTOM_CENTER"], "dwell_alert_seconds": float(dwell_seconds),
                              "polygon": pts})
        f.write_text(json.dumps(data, indent=2))
        cam["zones"] = str(f)
        self._regen_zone_rules(camera_id, cam, data["zones"])
        onboarding.add_camera(self.site_path, cam)
        return {"ok": True, "zones": data["zones"]}

    def set_vehicle_line(self, camera_id: str, start: list, end: list,
                         flip: bool = False, name: str = "gate") -> dict:
        """Save the directional tripwire a vehicle crosses to count as an
        entry or an exit, and wire the rules that listen for it.

        Two points, not a polygon: a car crossing the line one way is an
        ENTRY and the other way an EXIT, counted once per track. A parked car
        never crosses it, which is why this replaced the polygon-dwell
        approach that reported 30 entries for one car through a barrier.

        Points may be given in frame FRACTIONS (0..1, portable across
        resolutions, what a drawn line should send) or in original pixels;
        which it is, is recorded rather than guessed later. `flip` swaps
        which side counts as entering — the one thing that cannot be read off
        the geometry, because only the operator knows which way is in.
        """
        self._require(perms.CONFIGURE_CAMERAS)
        try:
            s = [float(start[0]), float(start[1])]
            e = [float(end[0]), float(end[1])]
        except (TypeError, ValueError, IndexError):
            return {"ok": False, "error": "a line needs a start and an end, each [x, y]"}
        if s == e:
            return {"ok": False, "error": "the line's start and end are the same point"}
        normalized = max(s[0], s[1], e[0], e[1]) <= 1.0
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"ok": False, "error": f"camera '{camera_id}' not found"}
        cam["vehicle_line"] = {"name": name or "gate", "start": s, "end": e,
                               "normalized": bool(normalized), "flip": bool(flip)}
        self._regen_zone_rules(camera_id, cam, self.list_zones(camera_id))
        onboarding.add_camera(self.site_path, cam)
        self.audit.record(self.current_user.username, "config_change",
                          f"camera:{camera_id}", detail={"vehicle_line": cam["vehicle_line"]})
        return {"ok": True, "vehicle_line": cam["vehicle_line"],
                "note": "vehicle entry/exit alerts apply on the next monitoring start"}

    def remove_vehicle_line(self, camera_id: str) -> dict:
        """Drop the tripwire and the rules that listened for it."""
        self._require(perms.CONFIGURE_CAMERAS)
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"ok": False, "error": f"camera '{camera_id}' not found"}
        cam.pop("vehicle_line", None)
        self._regen_zone_rules(camera_id, cam, self.list_zones(camera_id))
        onboarding.add_camera(self.site_path, cam)
        self.audit.record(self.current_user.username, "config_change",
                          f"camera:{camera_id}", detail={"vehicle_line": "(removed)"})
        return {"ok": True, "vehicle_line": None}

    def vehicle_line(self, camera_id: str) -> dict:
        """The camera's tripwire, or None — what a line editor opens with."""
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"ok": False, "error": f"camera '{camera_id}' not found"}
        return {"vehicle_line": cam.get("vehicle_line")}

    def remove_zone(self, camera_id: str, name: str) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        f = self._zones_file(camera_id)
        data = json.loads(f.read_text()) if f.exists() else {"zones": []}
        data["zones"] = [z for z in data.get("zones", []) if z.get("name") != name]
        f.write_text(json.dumps(data, indent=2))
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam:
            if data["zones"]:
                self._regen_zone_rules(camera_id, cam, data["zones"])
            else:                                   # no zones left -> restore base preset
                cam["config"] = cam.get("_base_config", cam.get("config"))
                cam.pop("zones", None)
            onboarding.add_camera(self.site_path, cam)
        return {"ok": True, "zones": data["zones"]}

    # --- scene context + custom (customer-defined) threats ---
    _SCENE_CONTEXT_FIELDS = {
        "camera_id",
        "source_type",
        "environment_type",
        "scene_description",
        "expected_actors",
        "zones",
        "confidence",
        "generated_at",
        "source_frame_path",
        "notes",
        "area_id",
        "site_type_candidate",
        "area_type_candidate",
        "view_description",
    }

    def _scene_camera(self, camera_id: str) -> dict | None:
        return next(
            (camera for camera in self.list_cameras() if camera.get("id") == camera_id),
            None,
        )

    def _scene_store(self, camera_id: str):
        from cvti.scene.context_store import SceneContextStore

        return SceneContextStore(Path(self.db_path).parent / "context", camera_id)

    def _scene_context_payload(self, context: dict) -> dict:
        return {
            key: value
            for key, value in dict(context or {}).items()
            if key in self._SCENE_CONTEXT_FIELDS
        }

    def _scene_context_response(self, camera_id: str, context: dict) -> dict:
        store = self._scene_store(camera_id)
        response = dict(context)
        status = store.load_status()
        response["mapping"] = {
            **status.to_dict(),
            "provenance": "cache",
        }
        try:
            frame_bytes = store.frame_path.read_bytes()
        except OSError:
            frame_bytes = b""
        response["source_frame_uri"] = (
            "data:image/jpeg;base64," + base64.b64encode(frame_bytes).decode()
            if frame_bytes
            else ""
        )
        return response

    def scene_context(self, camera_id: str) -> dict | None:
        """Return site-scoped canonical context plus lifecycle and frame preview."""
        store = self._scene_store(camera_id)
        try:
            context = json.loads(store.context_path.read_text())
        except (ValueError, OSError):
            context = None
        if isinstance(context, dict):
            return self._scene_context_response(camera_id, context)
        cam = self._scene_camera(camera_id)
        if (
            cam
            and cam.get("scene_description")
            and str(cam.get("scene_context_mode", "")).strip().lower() == "manual"
        ):
            return {
                "camera_id": camera_id,
                "environment_type": cam.get("environment_type", "unknown"),
                "scene_description": cam["scene_description"],
                "expected_actors": list(cam.get("expected_actors") or []),
                "zones": [],
                "confidence": 1.0,
                "mapping": {
                    # Authored is not reviewed (4 Sep): config text shows as
                    # awaiting review, so Scene Review tells the truth.
                    "status": "ready_unreviewed",
                    "provenance": "manual",
                    "reviewed_by": "",
                },
                "source_frame_uri": "",
            }
        if store.status_path.exists():
            return {
                "camera_id": camera_id,
                "environment_type": "unknown",
                "scene_description": "",
                "expected_actors": [],
                "zones": [],
                "confidence": 0.0,
                "mapping": {
                    **store.load_status().to_dict(),
                    "provenance": "mapper",
                },
                "source_frame_uri": "",
            }
        return None

    def _hierarchy_store(self):
        from cvti.scene.hierarchy import HierarchyContextStore

        return HierarchyContextStore(Path(self.db_path).parent / "context")

    def list_areas(self) -> list[dict]:
        self._require(perms.VIEW_LIVE)
        return onboarding.normalized_areas(self.site_path)

    def organization(self) -> dict:
        self._require(perms.VIEW_LIVE)
        return onboarding.normalized_organization(self.site_path)

    def update_organization(self, organization: dict) -> dict:
        self._require(perms.CONFIGURE_SITE)
        updated = onboarding.set_organization(self.site_path, organization)
        self.audit.record(
            self.current_user.username, "config_change",
            f"organization:{updated['id']}", detail={"organization": "updated"},
        )
        return updated

    def list_branches(self) -> list[dict]:
        self._require(perms.VIEW_LIVE)
        return onboarding.normalized_branches(self.site_path)

    def create_branch(self, branch: dict) -> list[dict]:
        self._require(perms.CONFIGURE_CAMERAS)
        from cvti.serving.onboarding import HierarchyConflict

        if isinstance(branch, dict) and branch.get("id") in {
            item["id"] for item in onboarding.normalized_branches(self.site_path)
        }:
            raise HierarchyConflict(
                f"branch {branch.get('id')!r} already exists; use PUT to update it"
            )
        branches = onboarding.upsert_branch(self.site_path, branch)
        self.audit.record(
            self.current_user.username, "config_change",
            f"branch:{branch.get('id', '')}", detail={"branch": "created"},
        )
        return branches

    def update_branch(self, branch_id: str, branch: dict) -> list[dict]:
        self._require(perms.CONFIGURE_CAMERAS)
        if branch_id not in {
            item["id"] for item in onboarding.normalized_branches(self.site_path)
        }:
            raise LookupError(f"unknown branch: {branch_id}")
        if not isinstance(branch, dict):
            raise ValueError("branch must be an object")
        branches = onboarding.upsert_branch(
            self.site_path, {**branch, "id": branch_id}
        )
        self.audit.record(
            self.current_user.username, "config_change", f"branch:{branch_id}",
            detail={"branch": "updated"},
        )
        return branches

    def remove_branch(self, branch_id: str) -> list[dict]:
        self._require(perms.CONFIGURE_CAMERAS)
        if branch_id not in {
            item["id"] for item in onboarding.normalized_branches(self.site_path)
        }:
            raise LookupError(f"unknown branch: {branch_id}")
        branches = onboarding.remove_branch(self.site_path, branch_id)
        self.audit.record(
            self.current_user.username, "config_change", f"branch:{branch_id}",
            detail={"branch": "removed"},
        )
        return branches

    def hierarchy(self) -> dict:
        self._require(perms.VIEW_LIVE)
        from cvti.api.sources import read_hierarchy
        return read_hierarchy(self.site_path)

    def create_area(self, area: dict) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        areas = onboarding.upsert_area(self.site_path, area)
        self.audit.record(self.current_user.username, "config_change",
                          f"area:{area.get('id', '')}", detail={"area": "created"})
        return {"ok": True, "areas": areas}

    def assign_camera_area(self, camera_id: str, area_id: str) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        camera = onboarding.assign_camera_area(self.site_path, camera_id, area_id)
        self.audit.record(self.current_user.username, "config_change",
                          f"camera:{camera_id}", detail={"area_id": area_id})
        return {"ok": True, "camera": camera}

    def area_context(self, area_id: str) -> dict | None:
        self._require(perms.VIEW_LIVE)
        return self._hierarchy_store().load_area(area_id)

    def _area_review_record(self, area: dict, context: dict | None) -> dict:
        evidence_ids = {
            str(camera_id)
            for camera_id in (context or {}).get("evidence_camera_ids", [])
        }
        cameras = []
        missing_evidence = []
        for camera_id in area["camera_ids"]:
            camera_context = self.scene_context(camera_id)
            frame_uri = (camera_context or {}).get("source_frame_uri", "")
            evidence_ready = bool(
                camera_context and frame_uri and camera_id in evidence_ids
            )
            if not evidence_ready:
                missing_evidence.append(camera_id)
            cameras.append({
                "camera_id": camera_id,
                "mapping": (camera_context or {}).get(
                    "mapping", {"status": "pending"}
                ),
                "source_frame_uri": frame_uri,
                "environment_type": (camera_context or {}).get(
                    "environment_type", "unknown"
                ),
                "confidence": float((camera_context or {}).get("confidence", 0.0)),
                "evidence_ready": evidence_ready,
            })
        bulk_reviewable = bool(
            context
            and cameras
            and not context.get("conflicts")
            and not missing_evidence
        )
        return {
            **area,
            "context": context,
            "cameras": cameras,
            "missing_evidence_camera_ids": missing_evidence,
            "bulk_reviewable": bulk_reviewable,
        }

    def approve_area_context(self, area_id: str, context: dict) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        payload = dict(context)
        payload["area_id"] = area_id
        area = next(
            (item for item in onboarding.normalized_areas(self.site_path)
             if item["id"] == area_id),
            None,
        )
        if area is None:
            raise ValueError(f"unknown area: {area_id}")
        review = self._area_review_record(area, payload)
        if not review["bulk_reviewable"]:
            missing = ", ".join(review["missing_evidence_camera_ids"])
            detail = f" Missing: {missing}." if missing else ""
            raise ValueError(
                "Area approval requires complete camera evidence and no conflicts."
                + detail
            )
        saved = self._hierarchy_store().approve_area(
            payload, self.current_user.username
        )
        approved_cameras: list[str] = []
        cameras = {str(camera["id"]): camera for camera in self.list_cameras()}
        for camera_id in area["camera_ids"]:
            camera = cameras.get(camera_id)
            store = self._scene_store(camera_id)
            camera_context = store._load_context()
            if camera is None or camera_context is None:
                continue
            camera_context["area_id"] = area_id
            store.approve(
                camera_context, self.current_user.username, camera["source"]
            )
            approved_cameras.append(camera_id)
        self.audit.record(self.current_user.username, "config_change",
                          f"area:{area_id}", detail={
                              "scene_context": "approved",
                              "camera_ids": approved_cameras,
                          })
        return {"ok": True, "context": saved, "camera_ids": approved_cameras}

    def update_site_context(self, context: dict) -> dict:
        self._require(perms.CONFIGURE_SITE)
        saved = self._hierarchy_store().save_site_proposal(context)
        return {"ok": True, "context": saved}

    def approve_site_context(self, context: dict) -> dict:
        self._require(perms.CONFIGURE_SITE)
        saved = self._hierarchy_store().approve_site(
            context, self.current_user.username
        )
        self.audit.record(self.current_user.username, "config_change", "site:scene",
                          detail={"scene_context": "approved"})
        return {"ok": True, "context": saved}

    def scene_mapping_progress(self) -> dict:
        self._require(perms.VIEW_LIVE)
        path = Path(self.db_path).parent / "context/mapping_queue.json"
        try:
            jobs = json.loads(path.read_text()).get("jobs", [])
        except (OSError, TypeError, ValueError):
            jobs = []
        states = ("pending", "running", "ready", "failed")
        return {"total": len(jobs), **{
            state: sum(job.get("state") == state for job in jobs) for state in states
        }}

    def enqueue_scene_mapping(self, camera_ids: list[str]) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        results = [self.request_scene_remap(str(camera_id)) for camera_id in camera_ids]
        return {"ok": all(result.get("ok") for result in results),
                "camera_ids": [str(value) for value in camera_ids]}

    def scene_review_summary(self) -> dict:
        self._require(perms.VIEW_LIVE)
        areas = []
        for area in onboarding.normalized_areas(self.site_path):
            context = self._hierarchy_store().load_area(area["id"])
            areas.append(self._area_review_record(area, context))
        statuses = [camera["mapping"].get("status")
                    for area in areas for camera in area["cameras"]]
        return {
            "site": self._hierarchy_store().load_site(),
            "areas": areas,
            "counts": {
                "total_cameras": len(statuses),
                "reviewed": sum(status == "ready_reviewed" for status in statuses),
                "ready": sum(status == "ready_unreviewed" for status in statuses),
                "failed": sum(status == "failed" for status in statuses),
            },
            "mapping": self.scene_mapping_progress(),
        }

    def update_scene_context(self, camera_id: str, context: dict) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        camera = self._scene_camera(camera_id)
        if camera is None:
            return {"error": f"camera '{camera_id}' not found"}
        resolution = self._scene_store(camera_id).save_unreviewed(
            self._scene_context_payload(context),
            camera["source"],
            provenance="manual_edit",
        )
        self.audit.record(
            self.current_user.username,
            "config_change",
            f"camera:{camera_id}",
            detail={"scene_context": "edited"},
        )
        return {
            "ok": True,
            "context": self._scene_context_response(camera_id, resolution.context),
        }

    def approve_scene_context(self, camera_id: str, context: dict) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        camera = self._scene_camera(camera_id)
        if camera is None:
            return {"error": f"camera '{camera_id}' not found"}
        resolution = self._scene_store(camera_id).approve(
            self._scene_context_payload(context),
            self.current_user.username,
            camera["source"],
        )
        self.audit.record(
            self.current_user.username,
            "config_change",
            f"camera:{camera_id}",
            detail={"scene_context": "approved"},
        )
        return {
            "ok": True,
            "context": self._scene_context_response(camera_id, resolution.context),
        }

    def request_scene_remap(self, camera_id: str) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        camera = self._scene_camera(camera_id)
        if camera is None:
            return {"error": f"camera '{camera_id}' not found"}
        status = self._scene_store(camera_id).mark_stale(camera["source"])
        self.audit.record(
            self.current_user.username,
            "config_change",
            f"camera:{camera_id}",
            detail={"scene_context": "remap_requested"},
        )
        return {"ok": True, "mapping": status.to_dict()}

    def accept_suggested_zone(
        self, camera_id: str, zone_id: str, dwell_seconds: float = 5.0
    ) -> dict:
        self._require(perms.CONFIGURE_CAMERAS)
        camera = self._scene_camera(camera_id)
        context = self.scene_context(camera_id)
        if camera is None or not context:
            return {"error": "camera scene context not found"}
        suggestion = next(
            (zone for zone in context.get("zones", []) if zone.get("id") == zone_id),
            None,
        )
        if suggestion is None:
            return {"error": f"suggested zone '{zone_id}' not found"}
        import cv2

        frame = cv2.imread(str(self._scene_store(camera_id).frame_path))
        if frame is None:
            return {"error": "representative scene frame is unavailable"}
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(value) for value in suggestion["bbox"])
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        result = self.add_zone(camera_id, zone_id, polygon, dwell_seconds)
        if result.get("error"):
            return result

        role = str(suggestion.get("role", "unknown"))
        zone_file = self._zones_file(camera_id)
        zone_data = json.loads(zone_file.read_text())
        accepted = next(
            zone for zone in zone_data.get("zones", []) if zone.get("name") == zone_id
        )
        accepted["kind"] = role
        accepted["context_role"] = role
        zone_file.write_text(json.dumps(zone_data, indent=2))

        cameras = onboarding.list_cameras(self.site_path)
        camera = self._cam(cameras, camera_id)
        roles = set(camera.get("accepted_zone_roles") or [])
        roles.add(role)
        camera["accepted_zone_roles"] = sorted(roles)
        onboarding.add_camera(self.site_path, camera)
        self.audit.record(
            self.current_user.username,
            "config_change",
            f"camera:{camera_id}",
            detail={"scene_zone_accepted": zone_id, "role": role},
        )
        return {"ok": True, "zone": accepted, "zones": zone_data["zones"]}

    def _cam(self, cams: list, camera_id: str):
        return next((c for c in cams if c.get("id") == camera_id), None)

    def add_custom_threat(self, camera_id: str, name: str, description: str) -> dict:
        """A customer-defined threat in plain English — the VLM gate evaluates it
        in this camera's scene context. Native detectors are kept alongside."""
        if not (description or "").strip():
            return {"error": "describe what to watch for"}
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        threats = cam.get("custom_threats") or []
        threats.append({"name": (name or "custom").strip(), "description": description.strip()})
        cam["custom_threats"] = threats
        onboarding.add_camera(self.site_path, cam)
        return {"ok": True, "custom_threats": threats}

    def remove_custom_threat(self, camera_id: str, index: int) -> dict:
        cams = onboarding.list_cameras(self.site_path)
        cam = self._cam(cams, camera_id)
        if cam is None:
            return {"error": f"camera '{camera_id}' not found"}
        threats = cam.get("custom_threats") or []
        if 0 <= index < len(threats):
            threats.pop(index)
        cam["custom_threats"] = threats
        onboarding.add_camera(self.site_path, cam)
        return {"ok": True, "custom_threats": threats}

    # --- first-run setup wizard ---
    def get_site(self) -> dict:
        meta = onboarding.get_site_meta(self.site_path)
        # Playback demo with no real cameras: skip the wizard, land on the dash.
        if meta["camera_count"] == 0 and self._demo:
            meta["configured"] = True
            meta["name"] = "Demo Store"
            meta["camera_count"] = len(self._live_sources(99))
        return meta

    def set_site(self, name: str | None = None, notify: str | None = None) -> dict:
        self._require(perms.CONFIGURE_SITE)
        return onboarding.set_site_meta(self.site_path, name=name, notify=notify)

    def _home_notify(self) -> str:
        """The site-wide delivery target (Telegram etc.), read from the HOME
        site config. The generated live config used to hard-code "console",
        so switching to Live EarthCams silently turned phone alerts off
        (12 Sep: "alert should send to telegram and the app both")."""
        try:
            return (onboarding.get_site_meta(self._home_site).get("notify") or "console").strip()
        except Exception:  # noqa: BLE001 - a missing home config must not block a spawn
            log.debug("home site notify unreadable; console only", exc_info=True)
            return "console"

    def mark_configured(self) -> dict:
        return onboarding.complete_first_run(self.site_path)

    def send_test_notification(self) -> dict:
        """Fire a synthetic alert through the site's configured notifier so the
        operator can confirm Telegram/WhatsApp/webhook actually reaches their phone."""
        from cvti.serving.alert_sink import build_notifier
        meta = onboarding.get_site_meta(self.site_path)
        notify = (meta.get("notify") or "console").strip()
        event = {
            "ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "camera_id": "test_camera", "rule": "test_alert", "priority": "high",
            "confidence": 0.99, "zone": None, "track_id": None, "object_label": None,
            "reason": "✅ Test alert from Argus — your notifications are working.",
            "evidence_dir": None,
        }
        try:
            notifier = build_notifier(notify)
            notifier.notify(event)
            # Background channels (Telegram, email) queue the send. Wait for
            # it here — this is the one place the operator is watching — and
            # surface a channel's own failure, so a wrong SMTP password reads
            # "authentication failed" in the app rather than "Test alert sent".
            closer = getattr(notifier, "close", None)
            if callable(closer):
                closer()
            parts = getattr(notifier, "notifiers", [notifier])
            errors = [e for e in (getattr(n, "last_error", "") for n in parts) if e]
            if errors:
                return {"ok": False, "via": notify, "error": "; ".join(errors)[:300]}
            return {"ok": True, "via": notify}
        except Exception as exc:  # noqa: BLE001
            log.warning("test notification failed", exc_info=True)
            return {"ok": False, "via": notify, "error": str(exc)[:200]}

    # --- diagnostics ---
    # --- retention / legal hold -------------------------------------------
    def set_legal_hold(self, event_id: int, hold: bool = True) -> dict:
        """Exempt an event's evidence from retention purge, or release it."""
        self._require(perms.MANAGE_LEGAL_HOLD)
        db, _ = self._effective_db()
        try:
            con = self._connect(db)
            con.execute("UPDATE events SET legal_hold = ? WHERE id = ?",
                        (1 if hold else 0, int(event_id)))
            con.commit()
            con.close()
        except sqlite3.OperationalError as exc:
            log.warning("legal hold update failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:200]}
        log.info("legal hold %s for event %s", "set" if hold else "released", event_id)
        return {"ok": True, "id": int(event_id), "legal_hold": bool(hold)}

    def retention_status(self) -> dict:
        """Policy, disk, and what is being retained past expiry and why."""
        from cvti.serving.retention import RetentionManager, RetentionPolicy
        meta = self.get_site()
        policy = RetentionPolicy.from_site(meta)
        out_dir = Path(self.db_path).parent
        db, _ = self._effective_db()
        status = RetentionManager(out_dir, policy, db_path=db).status()
        # The engine's own view wins when it is running — it is the process that
        # actually purges.
        engine = (self._gate_health() or {}).get("retention")
        if engine:
            status["last_run"] = engine.get("last_run") or status["last_run"]
        return status

    def set_retention(self, days: float = None) -> dict:
        self._require(perms.CONFIGURE_SITE)
        return onboarding.set_site_meta(self.site_path, retention_days=days)

    def export_evidence(self, event_ids: str = "", dest: str = "") -> dict:
        """Zip an event's evidence so a customer can keep it past expiry.

        Deliberately the opposite of the diagnostics bundle: that one excludes
        evidence because it is going to us; this one IS the evidence, and is
        going to the person who owns it.
        """
        self._require(perms.EXPORT_EVIDENCE)
        import zipfile
        db, base = self._effective_db()
        ids = [int(x) for x in str(event_ids).split(",") if str(x).strip().isdigit()]
        out_dir = Path(self.db_path).parent
        target = Path(dest) if dest else (out_dir / f"argus-evidence-{int(time.time())}.zip")
        try:
            con = self._connect(db)
            sql = "SELECT * FROM events"
            if ids:
                sql += f" WHERE id IN ({','.join('?' * len(ids))})"
            rows = con.execute(sql, ids).fetchall()
            con.close()
        except sqlite3.OperationalError as exc:
            log.warning("evidence export query failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:200]}

        exported = 0
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for row in rows:
                ev = row["evidence_dir"]
                if not ev:
                    continue
                path = Path(ev)
                if not path.is_absolute() and base:
                    path = base / path
                if not path.exists():
                    continue
                for f in sorted(path.rglob("*")):
                    if f.is_file():
                        zf.write(f, f"event_{row['id']}/{f.relative_to(path)}")
                exported += 1
            zf.writestr("MANIFEST.txt",
                        "Argus evidence export\n"
                        f"events: {exported}\n\n"
                        "CONTAINS camera images and video of identifiable people.\n"
                        "Handle under the same data-protection terms as the system itself.\n")
        log.info("exported evidence for %d event(s) -> %s", exported, target)
        return {"ok": True, "events": exported, "path": str(target),
                "size_kb": round(target.stat().st_size / 1024, 1)}

    def camera_links(self) -> list[dict]:
        """Per-camera link state from the running engine.

        Deliberately reports "unknown" when the engine is not running rather
        than "connected": claiming coverage we cannot observe is the exact
        failure this exists to prevent.
        """
        health = self._gate_health()
        if not health:
            return [{"camera_id": c["id"], "state": "unknown", "time_in_state": 0,
                     "reconnects": 0} for c in self.list_cameras()]
        return health.get("cameras") or []

    def download_diagnostics(self) -> dict:
        """Zip logs + a health snapshot for support. Never includes evidence.

        Returns the path rather than the bytes: the operator sends us a file,
        and keeping it on disk means they can inspect it before they do.
        """
        self._require(perms.VIEW_DIAGNOSTICS)
        from cvti.diagnostics import build_bundle
        out_dir = Path(self.db_path).parent
        try:
            preview = getattr(self, "_preview", None)
            path = build_bundle(out_dir, site_path=self.site_path,
                                preview=preview.diagnostics() if preview else
                                {"mode": "preview", "cameras": {}, "status": "not_started"})
        except Exception as exc:  # noqa: BLE001 - support tooling must not crash the app
            log.exception("diagnostics bundle failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:200]}
        return {"ok": True, "path": str(path),
                "size_kb": round(path.stat().st_size / 1024, 1)}

    # --- config backup / restore (EP-08-T1) --------------------------------
    def _backup_dir(self):
        meta = onboarding.get_site_meta(self.site_path)
        d = (meta.get("backup_dir") or "").strip() if isinstance(meta, dict) else ""
        return d or None

    def _maybe_auto_backup(self) -> None:
        """One versioned backup per calendar day, on app start."""
        from cvti import backup as _backup
        dest = Path(self._backup_dir() or _backup._default_dir())
        today = time.strftime("%Y%m%d")
        if any(dest.glob(f"argus-config-{today}_*.zip")):
            return
        _backup.backup_config(self.site_path, dest)

    def backup_now(self) -> dict:
        self._require(perms.CONFIGURE_SITE)
        from cvti import backup as _backup
        out = _backup.backup_config(self.site_path, self._backup_dir())
        self.audit.record(self.current_user.username, "config_change", "backup",
                          detail={"path": out.get("path")})
        return out

    def list_backups(self) -> list:
        self._require(perms.CONFIGURE_SITE)
        from cvti import backup as _backup
        return _backup.list_backups(self._backup_dir())

    def restore_backup(self, zip_path: str) -> dict:
        self._require(perms.CONFIGURE_SITE)
        from cvti import backup as _backup
        out = _backup.restore_config(zip_path, self.site_path)
        self.audit.record(self.current_user.username, "config_change", "restore",
                          detail={"path": zip_path, "ok": out.get("ok")})
        return out

    def set_backup_dir(self, path: str) -> dict:
        self._require(perms.CONFIGURE_SITE)
        onboarding.set_site_meta(self.site_path, backup_dir=(path or "").strip())
        return {"ok": True, "backup_dir": (path or "").strip()}

    def weekly_summary(self) -> dict:
        """On-demand build of the owner summary (the engine also auto-sends
        one every Monday morning through the site notifier)."""
        self._require(perms.CONFIGURE_SITE)
        from cvti.owner_summary import weekly_summary
        db, _ = self._effective_db()
        meta = onboarding.get_site_meta(self.site_path)
        return weekly_summary(db, meta, Path(self.db_path).parent)

    # --- verifier provider (Settings → AI verification) ----------------------
    def _secret_store(self) -> SecretStore:
        return SecretStore(Path(self._home_db).parent)

    def _verifier_settings(self) -> dict:
        """The site's `gate` block, normalised: provider id, model, base_url."""
        try:
            raw = onboarding.load_site(self.site_path).get("gate")
        except (OSError, ValueError):
            raw = None
        return verifier_providers.normalize_settings(raw)

    def verifier_settings(self) -> dict:
        """What the UI needs to show the verifier form. Never the key itself."""
        settings = self._verifier_settings()
        spec = verifier_providers.get_provider(settings["provider"])
        return {**settings, "label": spec.label, "needs_key": spec.needs_key,
                "key_set": bool(spec.needs_key and self._secret_store().has(VERIFIER_KEY)),
                "local": spec.local,
                "providers": [p.public() for p in verifier_providers.PROVIDERS.values()]}

    def set_verifier(self, provider: str | None = None, model: str | None = None,
                     base_url: str | None = None, api_key: str | None = None) -> dict:
        """Choose the verifier. An empty api_key leaves the stored key alone;
        a non-empty one replaces it. Takes effect on the next Start monitoring."""
        self._require(perms.CONFIGURE_SITE)
        current = self._verifier_settings()
        switching = provider is not None and provider != current["provider"]
        if switching and model is None:
            model = ""   # a new provider starts on its own default model, not the old one's
        wanted = verifier_providers.normalize_settings({
            "provider": provider if provider is not None else current["provider"],
            "model": model if model is not None else current["model"],
            "base_url": base_url if base_url is not None else current["base_url"]})
        spec = verifier_providers.get_provider(wanted["provider"])
        if spec.id == "custom" and not wanted["base_url"]:
            raise ValueError("A custom endpoint needs its base URL, e.g. https://host/v1")
        if spec.id == "custom" and not wanted["model"]:
            raise ValueError("A custom endpoint needs a model name")
        onboarding.set_site_meta(self.site_path, gate=wanted)
        if api_key:
            self._secret_store().set(VERIFIER_KEY, api_key.strip())
        self.audit.record(self._actor(), "config_change", "verifier",
                          {"provider": spec.id, "model": wanted["model"],
                           "key_changed": bool(api_key)})
        return self.verifier_settings()

    def clear_verifier_key(self) -> dict:
        self._require(perms.CONFIGURE_SITE)
        self._secret_store().delete(VERIFIER_KEY)
        self.audit.record(self._actor(), "config_change", "verifier", {"key_cleared": True})
        return self.verifier_settings()

    def test_verifier(self) -> dict:
        """One tiny, text-only request to the configured provider with the
        stored key, so an operator learns in seconds whether the key, the
        model name and the endpoint are right — not on the first real alert."""
        self._require(perms.CONFIGURE_SITE)
        settings = self._verifier_settings()
        spec = verifier_providers.get_provider(settings["provider"])
        if spec.local:
            status = vlm.gate_status(settings["model"])
            return {"ok": status.get("mode") == "live", "provider": spec.id,
                    "detail": {"live": "The on-device model is installed and answering.",
                               "no-model": "The runtime is up but the model is not downloaded.",
                               "offline": "The on-device runtime is not running."
                               }.get(status.get("mode"), status.get("mode", ""))}
        key = self._secret_store().get(VERIFIER_KEY)
        if spec.needs_key and not key:
            return {"ok": False, "provider": spec.id, "detail": "No API key saved yet."}
        started = time.monotonic()
        try:
            reply = _probe_cloud_verifier(spec, settings, key)
        except Exception as exc:  # noqa: BLE001 - the whole point is to report it
            log.info("verifier test against %s failed: %s", spec.id, str(exc)[:200])
            return {"ok": False, "provider": spec.id, "detail": str(exc)[:400],
                    "latency_ms": int((time.monotonic() - started) * 1000)}
        return {"ok": True, "provider": spec.id, "model": settings["model"],
                "latency_ms": int((time.monotonic() - started) * 1000),
                "detail": f"{spec.label} answered with {settings['model']}: {reply[:80]}"}

    def gate_status(self, model: str = vlm.DEFAULT_MODEL) -> dict:
        """Verifier readiness + the running engine's own view of the gate.

        Two different failures look identical from the operator's chair: the
        verifier being unreachable, and the gate erroring on every alert while
        it is up. The first comes from probing it, the second only the engine
        knows — it publishes it to gate_health.json.

        A cloud provider is "live" once its key is saved: there is nothing to
        download and nothing local to probe. `cloud`, `provider` and `label`
        let the UI say which one.
        """
        settings = self._verifier_settings()
        spec = verifier_providers.get_provider(settings["provider"])
        if spec.cloud:
            key_set = bool(self._secret_store().has(VERIFIER_KEY)) if spec.needs_key else True
            status = {"ollama": False, "model_present": True, "model": settings["model"],
                      "models": [], "mode": "live" if key_set else "no-key",
                      "cloud": True, "provider": spec.id, "label": spec.label}
        else:
            status = vlm.gate_status(model if model != vlm.DEFAULT_MODEL else settings["model"])
            status.update({"cloud": False, "provider": spec.id, "label": spec.label})
        status["engine"] = self._gate_health()
        # Whether an Ollama binary ships inside this app: the UI's "offline"
        # advice differs — "click Download" beats "go install ollama.com".
        try:
            from cvti.verification import ollama as _ollama
            status["runtime_bundled"] = _ollama.ollama_binary() is not None
        except Exception:  # noqa: BLE001
            log.debug("could not resolve a local ollama binary", exc_info=True)
            status["runtime_bundled"] = False
        return status

    def _gate_health(self) -> dict:
        """Gate stats published by the engine subprocess. Empty when it isn't running."""
        path = Path(self.db_path).parent / "gate_health.json"
        try:
            health = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        # Stale file from a previous run is worse than no file — it would show a
        # green gate for an engine that exited an hour ago.
        if time.time() - float(health.get("updated_at") or 0) > 30:
            return {}
        return health

    def pull_model(self, model: str = vlm.DEFAULT_MODEL) -> dict:
        # The user clicked Download: if no server is answering, start the
        # bundled/installed one now so the pull has somewhere to go. Ollama's
        # pull resumes partial downloads natively, so a killed 3 GB download
        # continues instead of restarting.
        try:
            from cvti.verification import ollama as _ollama
            _ollama.ensure_server()
        except Exception:  # noqa: BLE001
            log.warning("could not start the local VLM server", exc_info=True)
        # One click, both models. The recognition install runs on its own
        # thread and reports through recognition_model_status(); a failure
        # there never blocks the verification model.
        try:
            self.pull_recognition_model()
        except Exception:  # noqa: BLE001
            log.warning("could not start the recognition model install", exc_info=True)
        return vlm.start_pull(model)

    def pull_progress(self, model: str = vlm.DEFAULT_MODEL) -> dict:
        return vlm.pull_progress(model)

    # --- object-recognition model (SigLIP) -----------------------------------
    # Download AI installs two components: Gemma through Ollama for alert
    # verification, and SigLIP for matching camera crops against a customer's
    # reference photos. Object watch loads SigLIP from a local directory with
    # downloads disabled, so until this existed a fresh install had the
    # recognition code and none of its files (Demi's handoff, 5 Oct 2026).
    def pull_recognition_model(self) -> dict:
        from cvti.object_watch import model_install
        out = model_install.start_install()
        self._adopt_recognition_model()
        return out

    def recognition_model_status(self) -> dict:
        from cvti.object_watch import model_install
        out = model_install.status()
        if out.get("state") == "ready":
            self._adopt_recognition_model()
        out["object_watch"] = self._recognition_readiness()
        return out

    def _recognition_readiness(self) -> dict:
        """Object watch's own view, so a SigLIP failure is reported as an
        object-recognition dependency and nothing else."""
        try:
            from cvti.object_watch.runtime_config import preflight, resolve_config
            readiness = preflight(resolve_config(self._object_library_root()))
            return {"status": readiness.status, "reasons": list(readiness.reasons),
                    "model_path": str(resolve_config(self._object_library_root()).model_path or "")}
        except Exception as exc:  # noqa: BLE001 - a broken runtime.json is itself the status
            log.debug("object watch runtime config unreadable", exc_info=True)
            return {"status": "unavailable", "reasons": [str(exc)[:160]], "model_path": ""}

    def _adopt_recognition_model(self) -> None:
        """Point this site's object watch at the installed model when it has no
        usable model of its own. Never overrides a model directory that exists:
        a site that placed its own checkpoint keeps it."""
        from dataclasses import replace
        from cvti.object_watch import model_install
        from cvti.object_watch.runtime_config import resolve_config, write_config
        installed = model_install.default_install_dir()
        if not model_install.is_installed(installed):
            return
        try:
            current = resolve_config(self._object_library_root())
        except ValueError:
            log.warning("object watch runtime.json is invalid; not adopting the installed model")
            return
        if current.model_path and _has_siglip_files(current.model_path):
            return      # the site brought its own checkpoint; never override it
        write_config(self._object_library_root(), replace(current, model_path=installed.resolve()))
        log.info("object watch now uses the installed recognition model at %s", installed)

    # --- live wall (multi-camera video grid) ---
    def _close_preview(self):
        preview = getattr(self, "_preview", None)
        if preview is None:
            return
        # Let go of the object BEFORE close() can refuse. A refused close has
        # already stopped the frame server and set the decode thread's stop
        # flag, so the capture is released the moment its blocked read
        # returns. Holding on to that closed object left every Watch tile
        # on "preview unavailable" until the operator pressed Start again;
        # the next descriptor request now simply opens a fresh preview.
        self._preview = None
        preview.close()

    def camera_preview(self, camera_id: str) -> dict:
        self._require(perms.VIEW_LIVE)
        if self._monitor is not None and self._monitor.poll() is None:
            return {"error": "monitoring is starting; waiting for its stream"}
        cam = next((c for c in self.list_cameras() if c.get("id") == camera_id), None)
        if cam is None or cam.get("source") in (None, ""):
            return {"error": "camera not found"}
        from cvti.app.preview import CameraPreview
        if getattr(self, "_preview", None) is None:
            self._preview = CameraPreview()
        return self._preview.descriptor(camera_id, cam["source"])

    def _live_sources(self, count: int) -> list[dict]:
        """Sources for the live grid: the site's file/RTSP cameras if configured,
        otherwise fall back to demo clips in data/test_clips/."""
        cams = [c for c in self.list_cameras() if c.get("source")]
        if cams:
            return [{"id": c["id"], "source": c["source"]} for c in cams[:count]]
        if self._demo and (self._demo / "clips").exists():
            clips = sorted((self._demo / "clips").glob("*.mp4"))[:count]
            return [{"id": p.stem, "source": str(p)} for p in clips]
        clips = sorted(Path("data/test_clips").glob("*.mp4"))[:count]
        return [{"id": p.stem, "source": str(p)} for p in clips]

    def _engine_frame_port(self) -> tuple:
        """The engine's frame-publisher port, if it's running and serving.

        When the engine is up it has already decoded every stream and knows where
        everyone is, so we display ITS frames (with boxes) rather than opening the
        same videos again — decode is the dominant per-camera cost."""
        try:
            info = json.loads((Path(self.db_path).parent / "frames.json").read_text())
            port = int(info.get("port") or 0)
            token = str(info.get("token") or "")
        except Exception as exc:  # noqa: BLE001
            log.debug("engine frame port unreadable", exc_info=True)
            return 0, ""
        if not port:
            return 0, ""
        try:      # only trust it if it actually answers — WITH the token: the
            # publisher 401s tokenless probes, which made this check always
            # fail and the app silently re-decode every stream itself.
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{port}/cameras",
                                         headers={"X-Argus-Token": token})
            with urllib.request.urlopen(req, timeout=1.5) as r:
                cams = json.loads(r.read().decode()).get("cameras") or []
            return (port, token) if cams else (0, "")
        except Exception as exc:  # noqa: BLE001
            log.debug("engine frame port unreachable", exc_info=True)
            return 0, ""

    def live_start(self, count: int = 6) -> dict:
        self._require(perms.VIEW_LIVE)
        from cvti.app.live_wall import FrameServer, LiveWall
        self.live_stop()
        # Prefer the engine's already-decoded frames (no second decode, live boxes).
        port, token = self._engine_frame_port()
        if port:
            try:
                import urllib.request
                req = urllib.request.Request(f"http://127.0.0.1:{port}/cameras",
                                             headers={"X-Argus-Token": token})
                with urllib.request.urlopen(req, timeout=1.5) as r:
                    cams = json.loads(r.read().decode()).get("cameras") or []
                # The token MUST travel with the port: the UI's <img> tags are
                # refused without it, which is exactly the broken-tile bug.
                return {"cameras": [{"id": c} for c in cams], "port": port,
                        "token": token, "source": "engine"}
            except Exception as exc:  # noqa: BLE001
                log.debug("engine frames unavailable; decoding locally", exc_info=True)
                pass          # fall through to decoding ourselves
        sources = self._live_sources(count)
        if not sources:
            return {"cameras": [], "port": 0}
        # 15fps: this wall exists ONLY while the Watch screen is open,
        # so it is already viewer-gated by its own lifecycle — spend the
        # frames on smoothness (3 Sep, 'the stream seems laggy').
        self._live = LiveWall(sources, fps=15).start()
        self._fs = FrameServer(self._live)
        port = self._fs.start()
        # cameras + the localhost port the UI fetches JPEG frames from
        return {"cameras": [{"id": s["id"]} for s in sources], "port": port,
                "token": self._fs.token, "source": "app"}

    def live_frames(self) -> dict:
        return self._live.frames() if self._live else {}

    def live_stop(self) -> dict:
        if self._fs:
            self._fs.stop()
            self._fs = None
        if self._live:
            self._live.stop()
            self._live = None
        return {"stopped": True}

    # --- monitoring engine (Start/Stop) ---
    # Launches the full detection pipeline (YOLO + VideoMAE + Gemma gate) as a
    # subprocess pointed at this site, writing confirmed alerts into events.db.
    # In dev that is `python -m cvti.serving.pipeline`; in the installed app it
    # is the argus-engine executable shipped INSIDE the bundle (EP-05-T1) — the
    # installer that "cannot detect anything on its own" is exactly what the
    # audit said we must stop shipping.
    @staticmethod
    def _bundled_engine() -> "Path | None":
        """The engine executable next to this app's own, if we're a bundle."""
        if not getattr(sys, "frozen", False):
            return None
        exe = "argus-engine.exe" if sys.platform == "win32" else "argus-engine"
        candidate = Path(sys.executable).parent / exe
        return candidate if candidate.exists() else None

    def _engine_command(self) -> list:
        engine = self._bundled_engine()
        if engine is not None:
            return [str(engine)]
        return [sys.executable, "-m", "cvti.serving.pipeline"]

    def _close_engine_log(self) -> None:
        """Close the previous spawn's monitor.log handle. Every watchdog
        respawn used to open a fresh handle and leak the old one — six
        respawns in a crash loop meant six open handles on one file, which
        also blocks rotating it on Windows."""
        handle = getattr(self, "_engine_log_file", None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                log.debug("previous monitor.log handle would not close", exc_info=True)
        self._engine_log_file = None

    def _spawn_engine(self) -> "subprocess.Popen":
        inference_args = _desktop_inference_args(onboarding.load_site(self.site_path))
        self._close_preview()
        out_dir = Path(self.db_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        notify = (self.get_site().get("notify") or "console").strip()
        if notify == "console":
            # Feed configs are derived views of ONE site; the delivery
            # preference (Telegram etc.) is the site's, not the feed's.
            notify = self._home_notify()
        self._close_engine_log()
        _rotate_monitor_log(out_dir / "monitor.log")
        log_file = self._engine_log_file = open(out_dir / "monitor.log", "a")  # noqa: SIM115 - lives with the subprocess
        # Lean defaults conserve compute; distant people can require a larger
        # site-specific inference size at the cost of throughput.
        # Which verifier: the site's choice (Settings → AI verification), the
        # bundled Ollama runtime by default. The app used to hardcode Ollama,
        # which on a five-vCPU server with no GPU meant 30 to 90 s per verdict
        # and a breaker open most of the day (CHI pilot, 8 Oct 2026).
        verifier = self._verifier_settings()
        spec = verifier_providers.get_provider(verifier["provider"])
        if spec.local:
            # The gate needs the local Ollama server; in the bundled app nobody
            # has run `ollama serve` in a terminal — that is the point — so bring
            # up the bundled runtime if nothing is answering. Best-effort: if it
            # still is not up, the gate stays fail-visible and alerts arrive
            # UNVERIFIED.
            try:
                from cvti.verification import ollama as _ollama
                _ollama.ensure_server()
            except Exception:  # noqa: BLE001 - engine start must not die on this
                log.warning("could not ensure the local VLM server", exc_info=True)
        cmd = self._engine_command() + [
               "--site-config", self.site_path,
               # Identity is global: with per-feed event stores the engine's
               # mobile view would otherwise build an EMPTY account store and
               # nobody could sign in from a phone.
               "--security-dir", str(Path(self._home_db).parent)] + \
              verifier_providers.engine_args(verifier) + [
               "--notify", notify, "--output-dir", str(out_dir),
               "--seconds", "100000", "--gate-drain", "60"] + inference_args
        if sys.platform == "darwin" and spec.local:
            # Apple silicon shares ONE pool of memory between CPU and GPU, and
            # Ollama runs the gate model on Metal. A torch-MPS detector in the
            # same pool wedged the whole engine mid-graph under memory pressure
            # (12 Sep, main thread sampled inside MPSGraph mutex waits; log and
            # heartbeat silent while the process spun at 96% CPU). Detection at
            # 4fps/512px costs ~40ms on the CPU — give the GPU to the VLM.
            # A cloud verifier leaves the GPU to the detector.
            cmd += ["--device", "cpu"]
        kwargs = {}
        # The key reaches the engine through its environment and nowhere else:
        # not in site.json, not on the command line a process list would show.
        key_env = verifier_providers.key_environment(verifier, self._secret_store().get(VERIFIER_KEY))
        if key_env:
            import os
            kwargs["env"] = {**os.environ, **key_env}
        if sys.platform == "win32":
            # The engine is a console-mode exe; from the windowed app that would
            # flash a terminal at the user. Same flag is a no-op run from a shell.
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._engine_started_at = time.time()
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, **kwargs)

    def start_monitoring(self) -> dict:
        self._require(perms.CONTROL_ENGINE)
        return self._start_engine()

    def _start_engine(self) -> dict:
        """start_monitoring without the permission gate — for callers that
        already checked on the REQUEST thread. switch_feed restarts the engine
        on a background thread, where the API's per-request impersonation has
        already been cleared: the gate saw '<anonymous>' and refused, so a
        feed switch while monitoring failed every time from Demi's app and the
        engine stayed on the old feed (12 Sep)."""
        # A packaged app has no engine (torch/Ollama) inside it — it's a playback
        # demo. Don't try to spawn; the recorded alerts are already shown.
        if getattr(sys, "frozen", False) and self._bundled_engine() is None:
            # A lean viewer-only build (no engine inside). The full installer
            # ships argus-engine and never takes this branch.
            return {"running": False, "demo": True,
                    "note": "Playback demo — alerts are pre-recorded. This build has no detection engine inside."}
        if self._monitor and self._monitor.poll() is None:
            return {"running": True, "pid": self._monitor.pid, "already": True}
        # From here on this backend OWNS the engine's lifecycle, and the API
        # may answer "is it running" from the process instead of the heartbeat
        # file (which lags a Stop by up to 30s). Never set for a headless
        # engine somebody ran from a terminal — that one is heartbeat-judged.
        self._engine_owned = True
        self._monitor_should_run = True
        self._restarts = 0
        self._crash_looped = False
        self._monitor = self._spawn_engine()
        self._start_watchdog()
        return {"running": True, "pid": self._monitor.pid}

    def _start_watchdog(self, max_restarts: int = 5) -> None:
        """Respawn the engine if it dies unexpectedly (crash / OOM), up to a cap
        so a genuinely broken config can't loop forever."""
        import threading
        if getattr(self, "_watchdog", None) and self._watchdog.is_alive():
            return

        def loop():
            started = time.time()
            while getattr(self, "_monitor_should_run", False):
                time.sleep(3)
                if not getattr(self, "_monitor_should_run", False):
                    break
                if self._monitor and self._monitor.poll() is not None:   # died
                    # An engine that ran for an hour+ did not crash-loop: the
                    # spawn caps --seconds (~28h), so long-lived exits are
                    # SCHEDULED. Without this reset, a 24/7 site burned one
                    # restart per day and the watchdog gave up inside a week —
                    # silently, on a monitoring product. (Audit 23 Aug, #1.)
                    if time.time() - started > 3600:
                        self._restarts = 0
                    self._last_exit_code = self._monitor.poll()
                    self._last_death_at = time.time()
                    started = time.time()
                    if self._restarts < max_restarts:
                        self._restarts += 1
                        # Exponential backoff (W7): a crash-looping engine used
                        # to respawn every 3s — five failures in fifteen
                        # seconds, each one thrashing the box (model loads,
                        # Ollama bring-up) and making the NEXT crash likelier.
                        # 3s -> 6 -> 12 -> 24 -> 48 gives a transient cause
                        # (port in use, camera rebooting) time to clear.
                        delay = min(3 * (2 ** (self._restarts - 1)), 60)
                        log.warning(
                            f"[watchdog] engine exited unexpectedly (code "
                            f"{self._last_exit_code}) — restart "
                            f"{self._restarts}/{max_restarts} in {delay}s")
                        deadline = time.time() + delay
                        while (time.time() < deadline
                               and getattr(self, "_monitor_should_run", False)):
                            time.sleep(0.5)
                        if not getattr(self, "_monitor_should_run", False):
                            break
                        self._monitor = self._spawn_engine()
                    else:
                        # The circuit breaker LATCHES and says why — a
                        # monitoring product must never fail into silence
                        # (the pilot's black-wall day, 29 Aug).
                        log.error(
                            f"[watchdog] engine crash loop: {max_restarts} "
                            f"unexpected exits, last code "
                            f"{self._last_exit_code} — GIVING UP. "
                            "monitoring_status carries the exit code and the "
                            "log tail; the Diagnose bundle has the rest.")
                        self._crash_looped = True
                        self._monitor_should_run = False

        self._watchdog = threading.Thread(target=loop, name="engine-watchdog", daemon=True)
        self._watchdog.start()

    def stop_monitoring(self) -> dict:
        self._require(perms.CONTROL_ENGINE)
        return self._stop_engine()

    def _stop_engine(self) -> dict:
        """stop_monitoring without the permission gate — see _start_engine."""
        self._engine_owned = True
        self._monitor_should_run = False   # tell the watchdog this is intentional
        if self._monitor and self._monitor.poll() is None:
            self._monitor.terminate()
            try:
                self._monitor.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._monitor.kill()
                self._monitor.wait(timeout=8)
        self._monitor = None
        self._close_engine_log()
        # The terminate above skips the engine's own shutdown on Windows, so
        # the video gateway it launched would outlive it. End it here.
        try:
            from cvti.serving.go2rtc import reap_stale
            reap_stale(Path(self.db_path).parent)
        except Exception:  # noqa: BLE001 - stopping must not fail over a helper
            log.debug("go2rtc clean-up after stop failed", exc_info=True)
        # The engine removes frames.json and stream_gateway.json when it exits
        # on its own; the terminate above never gives it the chance, and a
        # stale marker points every tile at a dead port until the next Start
        # — the wall after Stop on the pilot server (11 Oct 2026). The process
        # is gone, so the files are ours to remove.
        self._remove_engine_markers()
        return {"running": False}

    def _remove_engine_markers(self) -> None:
        for name in ("frames.json", "stream_gateway.json"):
            try:
                (Path(self.db_path).parent / name).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                log.debug("could not remove stale %s after stop", name, exc_info=True)

    @property
    def crash_looped(self) -> bool:
        """The watchdog's circuit breaker latched: N unexpected exits."""
        return bool(getattr(self, "_crash_looped", False))

    def monitoring_status(self) -> dict:
        """Running or not — and when not, WHY, on screen.

        A pilot machine spent a day as a photo of a black wall because the
        engine died at startup and nothing anywhere said so, let alone why
        (29 Aug). If the engine exited, this now carries the exit code, the
        log's last meaningful line, and the log path — so 'it doesn't work
        on my machine' arrives as an error message instead of a screenshot.
        """
        running = bool(self._monitor and self._monitor.poll() is None)
        out = {"running": running, "pid": (self._monitor.pid if running else None)}
        if not running and self._monitor is not None:
            out["exit_code"] = self._monitor.poll()
            out["gave_up"] = not getattr(self, "_monitor_should_run", False)
            # the latched breaker is its own named state: 'stopped' can mean
            # the operator clicked Stop; 'crash_looped' never does
            out["crash_looped"] = self.crash_looped
            out["restarts"] = getattr(self, "_restarts", 0)
            out["log_path"], out["last_error"] = self._engine_log_tail()
        # A PID is not monitoring. The engine proves it is working by writing
        # gate_health.json every few seconds; a process that exists but has
        # stopped writing (hung model load, deadlock, zombie) used to look
        # exactly like a healthy engine to this check — while the header,
        # reading the stale heartbeat, said 'Engine not running'. Three status
        # strings, three sources of truth ('Noo.. it's saying not monitoring',
        # 29 Aug). The heartbeat decides now.
        if running:
            hb = Path(self.db_path).parent / "gate_health.json"
            age = None
            try:
                import json as _json
                doc = _json.loads(hb.read_text())
                age = time.time() - float(doc.get("generated_at") or 0)
                # The engine's first heartbeat lands BEFORE the slow parts
                # (scene preflight, cold model loads) and names its phase —
                # so a long warm-up reads 'Starting…', never 'engine not
                # running' under a green dot (field screenshot, 1 Sep).
                phase = str((doc.get("engine") or {}).get("phase") or "")
                if phase.startswith("starting") and age < 90:
                    out["starting"] = True
                    out["phase"] = phase
            except (OSError, ValueError):
                pass
            out["heartbeat_age_s"] = round(age, 1) if age is not None else None
            warming = time.time() - getattr(self, "_engine_started_at", 0) < 120
            if not warming and (age is None or age > 60):
                out["stalled"] = True
                lp, le = self._engine_log_tail()
                out["log_path"], out["last_error"] = lp, le
        # A crash LOOP hides from the check above: the watchdog respawns fast
        # enough that nearly every poll lands on a just-born process that looks
        # alive — so the UI said 'Stop monitoring' over an engine that had died
        # four times, and the promised reason never appeared (pilot screenshot,
        # 29 Aug). Restart churn is a first-class state now.
        restarts = getattr(self, "_restarts", 0)
        if restarts and time.time() - getattr(self, "_last_death_at", 0) < 900:
            out["restarts"] = restarts
            out["crash_looping"] = restarts >= 2
            out["last_exit_code"] = getattr(self, "_last_exit_code", None)
            lp, le = self._engine_log_tail()
            out.setdefault("log_path", lp)
            out.setdefault("last_error", le)
        return out

    def _engine_log_tail(self) -> tuple:
        log_path = Path(self.db_path).parent / "monitor.log"
        try:
            lines = [l.strip() for l in
                     log_path.read_text(errors="replace").splitlines()[-40:]
                     if l.strip()]
            telling = [l for l in lines
                       if any(k in l for k in ("Error", "ERROR", "Traceback",
                                               "error:", "Exception", "denied",
                                               "Permission", "No such"))]
            last = (telling or lines)[-1][:300] if (telling or lines) else ""
        except OSError:
            last = ""
        return str(log_path), last

    # --- feed source switcher: flip between demo videos and live cameras ---
    def _feeds_registry(self) -> dict:
        # resource_path, NOT a bare relative Path: the installed app is
        # launched with cwd anywhere, and a cwd-relative read here means the
        # feed toggle simply never appears on customer machines.
        p = resource_path("configs/feeds.json")
        if not p.exists():
            return {"sources": []}
        try:
            return json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            log.warning("feeds registry unreadable; no feeds offered", exc_info=True)
            return {"sources": []}

    def feed_sources(self) -> dict:
        """The switchable feed sources + which one is active (matched by config path)."""
        reg = self._feeds_registry()
        active = None
        srcs = []
        for s in reg.get("sources", []):
            cand = {str(Path(s.get("config", "")).resolve()),
                    str(self._writable_config(s.get("config", "")).resolve())}
            is_active = str(Path(self.site_path).resolve()) in cand
            if is_active:
                active = s["key"]
            srcs.append({"key": s["key"], "label": s["label"], "kind": s.get("kind", "demo")})
        return {"sources": srcs, "active": active}

    def switch_feed(self, key: str) -> dict:
        """Start switching the app (and, if running, the engine) to another feed.

        Returns IMMEDIATELY and does the work on a background thread — resolving
        live stream URLs takes seconds per feed and restarting the engine takes
        more, and doing that inline would freeze the Qt UI thread. Poll
        feed_switch_status() for progress."""
        self._require(perms.CONFIGURE_SITE)
        if self._monitor and self._monitor.poll() is None:
            # The switch will stop and restart the engine — check THAT right
            # here, on the request thread, where the caller's identity is
            # still attached. The worker thread below runs anonymous.
            self._require(perms.CONTROL_ENGINE)
        import threading
        st = getattr(self, "_switch_state", None)
        if st and st.get("busy"):
            return {"ok": False, "busy": True, "status": st.get("status", "switching…")}
        reg = self._feeds_registry()
        src = next((s for s in reg.get("sources", []) if s["key"] == key), None)
        if not src:
            return {"ok": False, "error": f"unknown feed source '{key}'"}
        self._switch_state = {"busy": True, "status": "starting…", "error": None,
                              "active": None, "done": False}
        threading.Thread(target=self._do_switch, args=(src, key),
                         name="feed-switch", daemon=True).start()
        return {"ok": True, "busy": True, "status": "switching…"}

    def feed_switch_status(self) -> dict:
        """Progress of an in-flight switch_feed (the UI polls this)."""
        return dict(getattr(self, "_switch_state", {"busy": False, "done": True}))

    def _do_switch(self, src: dict, key: str) -> None:
        self._close_preview()
        st = self._switch_state
        try:
            if src.get("kind") == "live":
                st["status"] = "resolving live streams…"
                src = dict(src, config=str(self._writable_config(src["config"])))
                res = self._resolve_live_config(src)
                if not res.get("ok"):
                    st.update(busy=False, done=True, error=res.get("error", "resolve failed"))
                    return
            was_running = bool(self._monitor and self._monitor.poll() is None)
            if was_running:
                st["status"] = "stopping engine…"
                self._stop_engine()
            cfg = str(self._writable_config(src["config"]))
            self.site_path = cfg
            self.db_path = self._db_for_feed(key, cfg)
            restarted = False
            # The frozen guard predated EP-05: the installed bundle SHIPS an
            # engine now, so a feed switch there used to stop monitoring and
            # report "done" — permanently dark. start_monitoring itself knows
            # how to refuse on a truly engine-less lean build.
            # (Audit 23 Aug, #3.)
            if was_running:
                st["status"] = "restarting engine…"
                out = self._start_engine()
                restarted = bool(out.get("running"))
            st.update(busy=False, done=True, active=key, error=None,
                      kind=src.get("kind", "demo"), config=src["config"],
                      engine_restarted=restarted, status="done")
        except Exception as exc:  # noqa: BLE001 - a failed switch must not wedge the UI
            log.error("feed switch failed", exc_info=True)
            st.update(busy=False, done=True, error=str(exc)[:200], status="failed")

    def _resolve_live_config(self, src: dict) -> dict:
        """Resolve each YouTube id to a fresh HLS URL and write the live config
        with running + the shared loitering watch-zone.

        The yt_dlp LIBRARY does the resolving — it is a normal dependency and
        travels inside the installed bundle. This used to shell out to a
        yt-dlp EXECUTABLE and refuse if the machine did not have one: true on
        this laptop, false on every customer install, so the pilot's Windows
        box showed the Live feed button and answered every click with an
        instruction to install developer tooling. A field report ('it didn't
        come on when it was switched', 28 Aug) is what surfaced it. The executable remains as a
        dev-environment fallback only.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _resolve_one(entry):
            name, vid = entry
            url = f"https://www.youtube.com/watch?v={vid}"
            try:
                import yt_dlp
                opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
                        "format": "best[height<=720]/best",
                        "extractor_args": {"youtube": {"player_client": ["android"]}},
                        "socket_timeout": 20}
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                return name, str(info.get("url") or "")
            except ImportError:
                pass                       # dev env without the package — try the CLI
            except Exception as exc:  # noqa: BLE001 - a dead feed shouldn't sink the others
                log.debug("camera place lookup failed", exc_info=True)
                return name, ""
            try:
                out = subprocess.run(
                    ["yt-dlp", "-g", "--extractor-args", "youtube:player_client=android",
                     "-f", "best[height<=720]/best", url],
                    capture_output=True, text=True, timeout=25)
                return name, ((out.stdout or "").strip().splitlines() or [""])[0]
            except Exception as exc:  # noqa: BLE001
                log.debug("camera place lookup failed", exc_info=True)
                return name, ""

        feeds = list(src.get("youtube", []))
        cams = []
        # Resolve every feed CONCURRENTLY — sequential resolves made the switch
        # take ~4x longer than it needed to.
        with ThreadPoolExecutor(max_workers=min(6, max(1, len(feeds)))) as pool:
            for name, url in pool.map(_resolve_one, feeds):
                if url:
                    # Per-camera text, not one template pasted four times —
                    # 'Venice Canal' described as "a street/plaza/terminal"
                    # was the operator's 'the descriptions are wrong' (3 Sep).
                    # No environment_type claim: "public area" was not even in
                    # the vocabulary; the mapper infers it, with this text as
                    # its prior, and a human reviews it in Scene Review.
                    cams.append({
                        "id": name, "source": url,
                        "config": "configs/rules/live_watch.json",
                        "zones": "configs/zones/live_watch.json",
                        "scene_description": (
                            f"Public live camera: {name}. A busy public view — "
                            "people passing through is normal; someone camped "
                            "in view or running may signal an incident."),
                        "running": True, "running_min_speed_ratio": 0.08,
                        "running_min_frames": 3})
        if not cams:
            return {"ok": False, "error": "could not reach the public demo feeds — "
                              "check this machine's internet connection and try again"}
        Path(src["config"]).write_text(json.dumps(
            {"name": "Live Dashboard", "notify": self._home_notify(), "configured": True, "cameras": cams}, indent=2))
        return {"ok": True, "resolved": len(cams)}

    def setup_state(self) -> dict:
        """Everything the wizard needs to decide whether to show + where to resume."""
        meta = self.get_site()
        return {
            "configured": meta["configured"],
            "site_name": meta["name"],
            "notify": meta["notify"],
            "cameras": meta["camera_count"],
            "gate": self.gate_status(),
            # playback demo (bundled, no real cameras) -> app opens on the Live wall
            "demo": bool(self._demo and not self.list_cameras()),
        }

    # --- events / review ---
    def _connect(self, path: str | None = None) -> sqlite3.Connection:
        con = sqlite3.connect(path or self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _db_for_feed(self, key: str, config: str) -> str:
        """Each feed's events live in their own store; the home site keeps the
        original path (existing data, and auth/audit stay beside it)."""
        try:
            if str(Path(config).resolve()) == self._home_site:
                return self._home_db
        except OSError:
            pass
        # Frozen apps live somewhere read-only (Program Files, /Applications).
        # This used to be Path("runs/feeds") — the CURRENT DIRECTORY, which on
        # a Start-Menu launch is the install dir, so the engine's first mkdir
        # was a PermissionError and it died before its first frame. Same
        # laptop-only-path disease, engine-side.
        if getattr(sys, "frozen", False):
            from cvti.utils import user_data_dir
            d = user_data_dir() / "runs" / "feeds" / key
        else:
            d = Path("runs/feeds") / key
        d.mkdir(parents=True, exist_ok=True)
        return str(d / "events.db")

    def _effective_db(self) -> tuple[str, "Path | None"]:
        """The DB to read + a base dir to resolve evidence frames against.

        Real DB always wins. The bundled playback demo is used ONLY when there
        are no real cameras configured — so a live site opens empty and fills as
        detection happens (never shows pre-recorded alerts before monitoring).
        """
        if Path(self.db_path).exists():
            return self.db_path, None
        if self._demo and not self.list_cameras():
            return str(self._demo / "events.db"), self._demo
        return self.db_path, None

    def list_events(self, limit: int = 100, embed_frames: bool = False) -> list[dict]:
        """The alert list — METADATA ONLY by default.

        Embedding evidence used to base64 every event's JPEGs into the reply:
        100 events x ~8 frames measured at ~189 MB through the QWebChannel, on
        every alerts render AND every poll cycle that saw a change. That is the
        'switching tabs is slow' report (25 Aug). Frames now load for the ONE
        alert being looked at, via event_clip(), which the UI already caches.
        """
        self._require(perms.VIEW_ALERTS)
        db, frame_base = self._effective_db()
        try:
            con = self._connect(db)
        except sqlite3.OperationalError as exc:
            # "No alerts" and "the events database is unreadable" must never
            # look the same on an operator's screen. (Audit 23 Aug, #12.)
            log.error("events db unreadable", exc_info=True)
            return {"error": f"events database unavailable: {str(exc)[:120]}"}
        try:
            rows = con.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.OperationalError as exc:
            con.close()
            if "no such table" in str(exc):
                return []          # a fresh site with no events yet IS quiet
            log.error("events query failed", exc_info=True)
            return {"error": f"events database unavailable: {str(exc)[:120]}"}
        con.close()
        out = []
        for r in rows:
            e = dict(r)
            e["review"] = e.get("review") or "new"
            # NO images in the list. The rows are text; only the detail and Now
            # screens show evidence, and they fetch it for the ONE alert being
            # looked at via event_clip(). Measured on a real store: embedding
            # frames + subject shots made listEvents(100) ~189 MB over the
            # QWebChannel on every render and every poll that saw a change —
            # the 'switching tabs is slow' report. (25 Aug.)
            if embed_frames:
                e["frames"] = self._frames_as_data_uris(e.get("evidence_dir"), frame_base)
                e["subject"] = self._subject_uri(e.get("evidence_dir"), frame_base)
            out.append(e)
        return out

    def event_clip(self, evidence_dir: str | None) -> dict:
        """Return the event's real-video clip.mp4 as a data URI (lazy, per-selection).

        Works for both a live run (absolute evidence_dir) and the bundled playback
        demo (evidence_dir relative to the demo bundle)."""
        self._require(perms.VIEW_ALERTS)
        _, frame_base = self._effective_db()
        if not evidence_dir:
            return {"uri": None, "frames": [], "subject": None}
        d = Path(evidence_dir)
        if not d.exists() and frame_base:
            d = frame_base / evidence_dir
        # Return ALL the event's frames (for the smooth image cine-loop the app plays)
        # plus the mp4 as a data URI (archival / download).
        frames = self._frames_as_data_uris(evidence_dir, frame_base, cap=120)
        subject = self._subject_uri(evidence_dir, frame_base)
        clip = d / "clip.mp4"
        uri = None
        codec = None
        if clip.exists():
            uri = "data:video/mp4;base64," + base64.b64encode(clip.read_bytes()).decode()
            # Say what the file is. A clip OpenCV wrote as mp4v is a black box
            # in Chromium with no error anyone can read; with the codec in the
            # reply the card can fall back to the frames and SAY why.
            try:
                from cvti.serving.alert_sink import _clip_codec
                codec = _clip_codec(clip)
            except Exception:  # diagnostics never block the clip
                log.debug("clip codec probe failed", exc_info=True)
        fps = None
        try:
            fps = json.loads((d / "event.json").read_text()).get("clip_fps")
        except (OSError, ValueError):
            pass
        return {"uri": uri, "frames": frames, "subject": subject, "fps": fps,
                "codec": codec}

    def search_events(self, query: str, limit: int = 200) -> dict:
        """Ask-your-cameras: natural-language search over past events.

        TrueSight reads a compact catalogue of events and returns the ones that
        match the plain-English query (e.g. 'anyone near the till after 6pm').
        Falls back to keyword matching if the local model isn't reachable."""
        query = (query or "").strip()
        if not query:
            return {"query": "", "matches": [], "answer": "", "engine": "none"}
        events = self.list_events(limit, embed_frames=False)
        if not events:
            return {"query": query, "matches": [], "answer": "No events recorded yet.", "engine": "none"}
        catalogue = "\n".join(
            f"[{e['id']}] {e.get('iso','')} cam={e.get('camera_id')} rule={e.get('rule')} "
            f"zone={e.get('zone') or '-'} :: {e.get('reason','')}" for e in events)
        ids, answer, engine = self._vlm_search(query, catalogue)
        if ids is None:                     # local model unavailable -> keyword fallback
            terms = [t for t in query.lower().split() if len(t) > 2]
            ids = [e["id"] for e in events if any(
                t in (str(e.get("reason", "")) + " " + str(e.get("rule", "")) + " "
                      + str(e.get("camera_id", "")) + " " + str(e.get("zone", ""))).lower()
                for t in terms)]
            answer, engine = "", "keyword"
        idset = set(ids)
        matches = [e for e in events if e["id"] in idset]
        _, frame_base = self._effective_db()
        for e in matches[:24]:              # attach evidence only to the matches shown
            e["frames"] = self._frames_as_data_uris(e.get("evidence_dir"), frame_base)
        return {"query": query, "matches": matches, "answer": answer, "engine": engine}

    def _vlm_search(self, query: str, catalogue: str):
        """Ask the local model which event IDs match. Returns (ids, answer, engine)
        or (None, '', '') if the model is unreachable (caller does keyword fallback)."""
        import urllib.error
        import urllib.request
        prompt = (
            "You are a security-footage search assistant. Below is a catalogue of past "
            "CCTV events (one per line, prefixed with [id]). Return ONLY the events that "
            "match the user's query. Reason over the description, camera, zone, and time.\n"
            "Respond with a single JSON object and nothing else:\n"
            '{"ids": [matching ids as integers], "answer": "one short sentence summarising what you found"}\n\n'
            f"EVENTS:\n{catalogue}\n\nQUERY: {query}\n")
        payload = {"model": LOCAL_VLM_MODEL, "temperature": 0,
                   # The answer is a JSON of ids + one sentence; cap it so a
                   # search can't monopolise an Ollama slot for a minute.
                   "max_tokens": 256,
                   "messages": [{"role": "user", "content": prompt}]}
        try:
            req = urllib.request.Request(
                "http://localhost:11434/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=40) as r:
                body = json.loads(r.read().decode("utf-8"))
            txt = body["choices"][0]["message"]["content"]
            data = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
            ids = [int(i) for i in data.get("ids", []) if str(i).strip().lstrip("-").isdigit()]
            return ids, str(data.get("answer", "")).strip(), "TrueSight"
        except Exception as exc:  # noqa: BLE001 - unreachable/parse error -> keyword fallback
            log.debug("evidence lookup failed", exc_info=True)
            return None, "", ""

    def _frames_as_data_uris(self, evidence_dir: str | None,
                             frame_base: "Path | None" = None, cap: int = 5) -> list[str]:
        # Path(None or "") is Path(".") — the CURRENT DIRECTORY. A provisional
        # alert legitimately has no evidence_dir yet (evidence is written when
        # the verdict settles), so this used to scan the working directory and
        # could hand back whatever JPEGs happened to be lying in it. Nearly
        # cost the repo during a cleanup, 27 Aug.
        if not evidence_dir:
            return []
        d = Path(evidence_dir)
        if not d.exists() and frame_base and evidence_dir:
            d = frame_base / evidence_dir       # bundled demo: paths are relative
        if not d.exists():
            return []
        uris = []
        for p in sorted(d.iterdir()):
            # subject.jpg is the annotated "who" shot, shown on its own — including
            # it here would make a box flash at the end of every cine-loop.
            if p.name == "subject.jpg":
                continue
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and len(uris) < cap:
                b64 = base64.b64encode(p.read_bytes()).decode()
                uris.append(f"data:image/jpeg;base64,{b64}")
        return uris

    def _subject_uri(self, evidence_dir: str | None,
                     frame_base: "Path | None" = None) -> str | None:
        """The annotated frame with the subject boxed, if one was saved."""
        if not evidence_dir:
            return {"ok": False, "error": "this alert has no evidence yet "
                                          "(verification is still in progress)"}
        d = Path(evidence_dir)
        if not d.exists() and frame_base and evidence_dir:
            d = frame_base / evidence_dir
        p = d / "subject.jpg"
        if not p.exists():
            return None
        return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()

    def set_review(self, event_id: int | str, label: str) -> dict:
        """Legacy label entry point — routed through the state machine so every
        transition carries an owner and lands in the audit trail."""
        self._require(perms.REVIEW_ALERTS)
        if label not in _REVIEW_VALUES:
            raise ValueError(f"review must be one of {_REVIEW_VALUES}")
        mapping = {"true": ("resolve", "real"), "false": ("resolve", "false_alarm"),
                   "ack": ("acknowledge", None)}
        action, outcome = mapping[label]
        if action == "acknowledge":
            return self.acknowledge_alert(event_id)
        return self.resolve_alert(event_id, outcome)

    def _triage_connect(self):
        """Write to the SAME db we read from. The bundled read-only demo can't
        be written — callers degrade gracefully rather than showing a modal."""
        db, _ = self._effective_db()
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        con = self._connect(db)
        from cvti import triage
        triage.ensure_columns(con)
        return con

    def _actor(self) -> str:
        user = self.current_user
        return user.username if user else "<unknown>"

    def acknowledge_alert(self, event_id: int | str) -> dict:
        """Claim it. Everyone else sees your name against it."""
        self._require(perms.REVIEW_ALERTS)
        from cvti import triage
        try:
            con = self._triage_connect()
        except (sqlite3.OperationalError, OSError):
            return {"ok": False, "persisted": False}
        try:
            result = triage.acknowledge(con, int(event_id), self._actor())
        except triage.TriageError as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            con.close()
        self.audit.record(self._actor(), "alert_resolution", f"event:{event_id}",
                          {"transition": "acknowledged"})
        return {"id": event_id, "persisted": True, **result}

    def resolve_alert(self, event_id: int | str, outcome: str, note: str = "") -> dict:
        """Conclude it. The outcome feeds the model's feedback loop; the note is
        for the humans on the next shift."""
        self._require(perms.REVIEW_ALERTS)
        from cvti import triage
        try:
            con = self._triage_connect()
        except (sqlite3.OperationalError, OSError):
            return {"ok": False, "persisted": False}
        try:
            result = triage.resolve(con, int(event_id), self._actor(), outcome, note)
        except triage.TriageError as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            con.close()
        self.audit.record(self._actor(), "alert_resolution", f"event:{event_id}",
                          {"transition": "resolved", "outcome": outcome,
                           "note": bool(note)})
        return {"id": event_id, "persisted": True, **result}

    def export_incident_pdf(self, event_id: int | str) -> dict:
        """The incident record as a file that leaves the building intact —
        what a manager reviews, what goes to an insurer or the police."""
        self._require(perms.EXPORT_EVIDENCE)
        db, frame_base = self._effective_db()
        try:
            con = self._connect(db)
            row = con.execute("SELECT * FROM events WHERE id = ?",
                              (int(event_id),)).fetchone()
            con.close()
        except sqlite3.OperationalError as exc:
            return {"ok": False, "error": str(exc)[:200]}
        if row is None:
            return {"ok": False, "error": f"no incident #{event_id}"}
        event = dict(row)

        frames = []
        ev_dir = event.get("evidence_dir")
        if ev_dir:
            path = Path(ev_dir)
            if not path.is_absolute() and frame_base:
                path = frame_base / path
            if path.exists():
                for f in sorted(path.glob("frame_*.jpg"))[:8]:
                    try:
                        frames.append((f.name, f.read_bytes()))
                    except OSError:
                        log.debug("unreadable evidence frame %s", f, exc_info=True)
                subject = path / "subject.jpg"
                if subject.exists():
                    frames.insert(0, ("subject — boxed at the moment it fired",
                                      subject.read_bytes()))

        from cvti.incident_pdf import build_incident_pdf
        dest = Path(self.db_path).parent / f"argus-incident-{event['id']}.pdf"
        try:
            build_incident_pdf(event, frames, dest)
        except Exception as exc:  # noqa: BLE001 - export must not crash the app
            log.error("incident PDF failed", exc_info=True)
            return {"ok": False, "error": str(exc)[:200]}
        self.audit.record(self._actor(), "evidence_export", f"event:{event_id}",
                          {"format": "pdf", "frames": len(frames)})
        return {"ok": True, "path": str(dest),
                "size_kb": round(dest.stat().st_size / 1024, 1)}

    def handover(self, hours: float = 8.0) -> dict:
        """What the incoming shift needs: what fired, what was concluded and by
        whom, and — flagged loudest — what is still open. Context must not
        reset at shift change."""
        self._require(perms.VIEW_ALERTS)
        db, _ = self._effective_db()
        since = time.time() - max(1.0, float(hours)) * 3600
        try:
            con = self._connect(db)
        except sqlite3.OperationalError:
            return {"hours": hours, "fired": [], "resolved": [], "open": [],
                    "counts": {}}
        state_expr = ("COALESCE(state, CASE WHEN review IN ('true','false') "
                      "THEN 'resolved' WHEN review='ack' THEN 'acknowledged' "
                      "ELSE 'new' END)")
        try:
            fired = [dict(r) for r in con.execute(
                "SELECT id, ts, iso, camera_id, rule, priority FROM events "
                "WHERE ts >= ? ORDER BY ts DESC", (since,))]
            resolved = [dict(r) for r in con.execute(
                f"SELECT id, iso, camera_id, rule, owner, outcome, note, resolved_at "
                f"FROM events WHERE {state_expr} = 'resolved' AND resolved_at >= ? "
                f"ORDER BY resolved_at DESC", (since,))]
            # Open items are NOT windowed: an incident from three shifts ago
            # that nobody concluded is precisely what must not be forgotten.
            open_items = [dict(r) for r in con.execute(
                f"SELECT id, ts, iso, camera_id, rule, priority, owner, "
                f"{state_expr} AS state FROM events "
                f"WHERE {state_expr} != 'resolved' ORDER BY ts ASC")]
        except sqlite3.OperationalError as exc:
            con.close()
            return {"hours": hours, "error": str(exc)[:200]}
        con.close()
        now = time.time()
        for item in open_items:
            item["age_h"] = round((now - (item.get("ts") or now)) / 3600, 1)
            item["carried_over"] = (item.get("ts") or now) < since
        outcomes = {}
        for r in resolved:
            outcomes[r.get("outcome") or "?"] = outcomes.get(r.get("outcome") or "?", 0) + 1
        return {"hours": float(hours), "since": since,
                "fired": fired[:100], "resolved": resolved[:50],
                "open": open_items[:50],
                "counts": {"fired": len(fired), "resolved": len(resolved),
                           "open": len(open_items), "outcomes": outcomes}}

    def needs_attention(self, min_priority: str = "medium") -> dict:
        """The Now view: one alert first, the queue behind it, who holds what."""
        self._require(perms.VIEW_ALERTS)
        from cvti import triage
        try:
            con = self._triage_connect()
        except (sqlite3.OperationalError, OSError):
            return {"now": None, "then": [], "waiting": 0, "held": []}
        try:
            out = triage.needs_attention(con, min_priority=min_priority)
        finally:
            con.close()
        db, frame_base = self._effective_db()
        if out["now"]:
            out["now"]["review"] = out["now"].get("review") or "new"
            out["now"]["frames"] = self._frames_as_data_uris(
                out["now"].get("evidence_dir"), frame_base)
            out["now"]["subject"] = self._subject_uri(
                out["now"].get("evidence_dir"), frame_base)
        return out

    def learning_stats(self) -> dict:
        """Feedback / reinforcement-training status for the Learning screen."""
        from cvti.feedback.manager import FeedbackManager
        db, _ = self._effective_db()
        return FeedbackManager(db).status()

    def learning_calibrate(self) -> dict:
        """Re-run online calibration from the operator's labels (writes calibration.json;
        the running engine hot-reloads it and stops paging on chronically-wrong rules)."""
        from cvti.feedback.manager import FeedbackManager
        db, _ = self._effective_db()
        return FeedbackManager(db).calibrate()

    # --- value surface ----------------------------------------------------
    def set_value_inputs(self, incident_value: "float | None" = None,
                         guard_hourly_cost: "float | None" = None,
                         review_minutes: "float | None" = None) -> dict:
        """The site's own money figures. Blank stays blank — see value_summary."""
        self._require(perms.CONFIGURE_SITE)
        return onboarding.set_site_meta(
            self.site_path, incident_value=incident_value,
            guard_hourly_cost=guard_hourly_cost, review_minutes=review_minutes)

    def value_summary(self, days: int = 30) -> dict:
        """What the system was worth over `days`, in the buyer's terms.

        Suppression percentage is an engineering metric. What a buyer is actually
        deciding about is: how many incidents did I get told about, how many
        false alarms did I not have to look at, and how much of my guards' shift
        did that give back.

        Every figure here is a count of real rows — `incidents` from the events
        table, the rest from the suppression ledger — so any number on the screen
        can be walked back to the events behind it. Nothing is modelled or
        extrapolated; the money figures are simply those counts multiplied by
        rates the site typed in, and are omitted entirely when it hasn't.
        """
        db, _ = self._effective_db()
        since = time.time() - max(1, int(days)) * 86400
        since_day = time.strftime("%Y-%m-%d", time.localtime(since))

        incidents = reviewed_true = reviewed_false = unverified = 0
        shown = rejected = deduped = errors = 0
        try:
            con = self._connect(db)
            try:
                row = con.execute(
                    "SELECT COUNT(*), "
                    "SUM(CASE WHEN review='true' THEN 1 ELSE 0 END), "
                    "SUM(CASE WHEN review='false' THEN 1 ELSE 0 END) "
                    "FROM events WHERE ts >= ? AND COALESCE(retracted,0)=0 "
                    "AND COALESCE(provisional,0)=0", (since,)).fetchone()
                incidents = row[0] or 0
                reviewed_true = row[1] or 0
                reviewed_false = row[2] or 0
            except sqlite3.OperationalError:
                pass
            try:
                # An unverified alert reached the operator because the gate could
                # NOT decide. Counting it as a detection would credit the product
                # for work it did not do.
                unverified = con.execute(
                    "SELECT COUNT(*) FROM events WHERE ts >= ? AND unverified = 1",
                    (since,)).fetchone()[0] or 0
            except sqlite3.OperationalError:
                unverified = 0
            try:
                row = con.execute(
                    "SELECT SUM(shown), SUM(rejected), SUM(deduped), SUM(errors) "
                    "FROM suppression_daily WHERE day >= ?", (since_day,)).fetchone()
                shown, rejected, deduped, errors = (v or 0 for v in row)
            except sqlite3.OperationalError:
                pass       # ledger only exists once an engine has run
            con.close()
        except sqlite3.OperationalError:
            pass

        # The gate only ever sees candidates the detectors produced, so raw is
        # exactly what an operator would have faced without verification.
        raw_alerts = shown + rejected + deduped
        # Kept apart on purpose. A rejected alert is one the AI looked at and
        # judged wrong; a deduped one is a repeat of an event already queued.
        # Both cost an operator attention, but only the first is a false alarm,
        # and rolling them together would overstate the claim that matters most.
        noise_removed = rejected + deduped
        meta = self.get_site()
        review_minutes = float(meta.get("review_minutes") or 0.0)
        hours_saved = noise_removed * review_minutes / 60.0

        money = {}
        guard_rate = float(meta.get("guard_hourly_cost") or 0.0)
        incident_value = float(meta.get("incident_value") or 0.0)
        if guard_rate > 0:
            money["attention_saved"] = round(hours_saved * guard_rate, 2)
        if incident_value > 0:
            money["incidents_value"] = round(incidents * incident_value, 2)

        return {
            "days": int(days),
            "incidents": max(0, incidents - unverified),   # gate actually confirmed these
            "unverified": unverified,               # surfaced because the gate could not decide
            "false_alarms_prevented": rejected,     # AI looked and said no
            "duplicates_collapsed": deduped,        # repeat of an event already queued
            "noise_removed": noise_removed,
            "raw_alerts": raw_alerts,               # what you would have seen
            "shown": shown,                         # what you were actually shown
            "suppression_pct": round(noise_removed / raw_alerts, 4) if raw_alerts else None,
            "attention_hours_saved": round(hours_saved, 2),
            "gate_errors": errors,                  # alerts the gate could not verify
            "operator_labels": {"true": reviewed_true, "false": reviewed_false},
            "inputs": {"review_minutes": review_minutes, "guard_hourly_cost": guard_rate,
                       "incident_value": incident_value},
            "money": money,
            # Three states, not two. A database with incidents but no ledger —
            # the bundled playback demo, or any run from before suppression was
            # recorded — would otherwise render "0 false alarms prevented, 0.0
            # hours saved", which reads as the product doing nothing rather than
            # as a measurement we never took.
            "has_data": bool(raw_alerts or incidents),
            "has_verification_history": bool(raw_alerts),
        }

    def counts(self) -> dict:
        """Header/nav summary numbers."""
        cams = self.list_cameras()
        n_cams = len(cams) if cams else len(self._live_sources(99))
        db, _ = self._effective_db()
        pending = 0
        try:
            con = self._connect(db)
            try:
                # "to review" = not yet handled. Ack/True/False all clear it.
                pending = con.execute(
                    "SELECT COUNT(*) FROM events WHERE review IS NULL "
                    "AND COALESCE(retracted, 0) = 0").fetchone()[0]
            except sqlite3.OperationalError:
                pending = 0
            con.close()
        except sqlite3.OperationalError:
            pending = 0
        return {"cameras": n_cams, "pending_alerts": pending}
