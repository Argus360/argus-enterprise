import { app, BrowserWindow, ipcMain, dialog, net, protocol } from "electron";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";
import readline from "node:readline";
import { API_OPERATIONS, type ArgusApiClient } from "./api-client.js";
import { createOwnedApiClient, createSmokeStateWriter } from "./api-runtime.js";
import { startOwnedApi, type OwnedApi } from "./api-supervisor.js";
import { createSupportLog } from "./support-log.js";
import { resolveLayout, type InstallLayout } from "./install-layout.js";
import {
  createBridgeTransport,
  registerBridgeStreamProtocol,
} from "./bridge-transport.js";

protocol.registerSchemesAsPrivileged([
  {
    scheme: "argus-stream",
    privileges: { standard: true, secure: true, stream: true },
  },
]);

const dir = path.dirname(fileURLToPath(import.meta.url));
const root = process.env.ARGUS_REPO || path.resolve(dir, "../..");
/**
 * Development runs out of the repo against a .venv interpreter. An installed
 * app has neither, and spawns the frozen `argus-api` binary shipped beside
 * the engine. Resolved once, lazily, so a layout problem surfaces as a real
 * error in the UI rather than a blank window at startup.
 */
let layoutCache: InstallLayout | undefined;
function layout(): InstallLayout {
  if (!layoutCache)
    layoutCache = resolveLayout({
      packaged: app.isPackaged,
      dir,
      resourcesPath: process.resourcesPath,
    });
  return layoutCache;
}
let window: BrowserWindow | null = null;
let worker: ChildProcessWithoutNullStreams | undefined;
let sequence = 0;
let exiting = false;
const transport = process.env.ARGUS_TRANSPORT === "bridge" ? "bridge" : "api";
let ownedApi: OwnedApi | undefined;
let apiClient: ArgusApiClient | undefined;
let apiStarting: Promise<ArgusApiClient> | undefined;
const smokeState = createSmokeStateWriter(process.env);
if (process.env.ARGUS_USER_DATA)
  app.setPath("userData", process.env.ARGUS_USER_DATA);
const pending = new Map<
  number,
  {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
  }
>();
const methods = new Set([
  "auth_state",
  "create_first_owner",
  "sign_in",
  "sign_out",
  "get_site",
  "set_site",
  "list_cameras",
  "add_camera",
  "remove_camera",
  "test",
  "discover_cameras",
  "detect_subnet",
  "scan",
  "list_areas",
  "create_area",
  "assign_camera_area",
  "scene_context",
  "scene_review_summary",
  "area_context",
  "approve_area_context",
  "approve_site_context",
  "update_scene_context",
  "approve_scene_context",
  "request_scene_remap",
  "enqueue_scene_mapping",
  "scene_mapping_progress",
  "accept_suggested_zone",
  "camera_snapshot",
  "registered_objects",
  "recapture_registered_object",
  "register_object_region",
  "remove_registered_object",
  "list_zones",
  "add_zone",
  "remove_zone",
  "set_camera_rules",
  "object_targets",
  "create_object_target",
  "set_object_watch_runtime_config",
  "reembed_object_targets",
  "add_object_example",
  "review_object_example",
  "object_example_preview",
  "activate_object_target",
  "deactivate_object_target",
  "object_watch_job_status",
  "set_object_watch_rule",
  "presets",
  "use_case_templates",
  "apply_template",
  "add_custom_rule",
  "remove_custom_rule",
  "english_rules_status",
  "list_events",
  "event_clip",
  "acknowledge_alert",
  "resolve_alert",
  "search_events",
  "live_start",
  "camera_links",
  "start_monitoring",
  "stop_monitoring",
  "monitoring_status",
  "feed_sources",
  "switch_feed",
  "feed_switch_status",
  "setup_state",
  "setup_check",
  "mark_configured",
  "send_test_notification",
  "gate_status",
  "pull_model",
  "pull_progress",
  "retention_status",
  "set_retention",
  "list_users",
  "add_user",
  "remove_user",
  "audit_entries",
  "backup_now",
  "download_diagnostics",
  "value_summary",
  "role_table",
  "disk_encryption",
  // parity with the retired PyQt console (17 Sep)
  "auth_accounts",
  "auth_recovery",
  "change_own_password",
  "set_user_role",
  "create_owner_override",
  "counts",
  "needs_attention",
  "export_incident_pdf",
  "export_evidence",
  "set_legal_hold",
  "weekly_summary",
  "handover",
  "list_backups",
  "restore_backup",
  "set_backup_dir",
  "audit_export",
  "audit_verify",
  "app_version",
  "detector_validation",
  "heartbeat_status",
  "set_heartbeat",
  "learning_stats",
  "learning_calibrate",
  "set_value_inputs",
  "live_frames",
  "set_custom_rule",
  "add_custom_threat",
  "remove_custom_threat",
  "update_site_context",
]);

methods.add("live_stop");
for (const method of [
  "camera_stream",
  "organization",
  "update_organization",
  "list_branches",
  "create_branch",
  "update_branch",
  "remove_branch",
  "hierarchy",
])
  methods.add(method);
// Every route the API client knows is callable. The hand-written list above
// drifted from the client twice: the cloud verifier (v1.10.0) and the vehicle
// line shipped with their routes but not here, so the packaged app answered
// "Invalid engine request" to its own settings panel (CHI, 13 Oct 2026).
for (const method of API_OPERATIONS) methods.add(method);
function failPending(message: string) {
  for (const item of pending.values()) {
    clearTimeout(item.timer);
    item.reject(new Error(message));
  }
  pending.clear();
}
function ensureWorker() {
  if (worker) return worker;
  const install = layout();
  // The stdio bridge is the DEVELOPMENT transport: it runs bridge.py through
  // an interpreter, which an installed app does not have. Installed builds
  // use the API transport (the default), so say so plainly rather than
  // failing later with a confusing missing-Python error.
  if (install.packaged)
    throw new Error(
      "ARGUS_TRANSPORT=bridge is a development-only transport; the installed app talks to the bundled Engine API.",
    );
  const bridge = path.resolve(dir, "../bridge.py");
  worker = spawn(
    install.apiCommand,
    [
      "-u",
      bridge,
      "--repo",
      install.engineRoot,
      "--site",
      install.site,
      "--db",
      install.db,
    ],
    {
      cwd: install.engineRoot,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      stdio: "pipe",
    },
  );
  worker.stderr.on("data", (data) => process.stderr.write(data));
  readline.createInterface({ input: worker.stdout }).on("line", (line) => {
    try {
      const data = JSON.parse(line);
      const item = pending.get(data.id);
      if (!item) return;
      clearTimeout(item.timer);
      pending.delete(data.id);
      if (data.error) item.reject(new Error(data.error));
      else item.resolve(data.result);
    } catch {
      console.error("Invalid engine protocol response");
    }
  });
  worker.on("error", (error) => {
    failPending(error.message);
    worker = undefined;
  });
  worker.on("exit", () => {
    failPending("Local backend disconnected. Reconnect to try again.");
    worker = undefined;
  });
  return worker;
}
function invoke(method: string, args: unknown[]) {
  return new Promise((resolve, reject) => {
    try {
      const child = ensureWorker();
      const id = ++sequence;
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(
          new Error(
            "Backend request timed out. Check engine health before retrying a change.",
          ),
        );
      }, 150000);
      pending.set(id, { resolve, reject, timer });
      child.stdin.write(JSON.stringify({ id, method, args }) + "\n");
    } catch (error) {
      reject(error);
    }
  });
}

const bridgeTransport = createBridgeTransport(invoke, (url) => net.fetch(url));

async function ensureApi() {
  if (apiClient) return apiClient;
  if (apiStarting) return apiStarting;
  apiStarting = (async () => {
    const install = layout();
    // Installed, the writable directory may not exist yet (first launch, or a
    // machine upgrading from a version that never had one).
    fs.mkdirSync(install.dataDir, { recursive: true });
    const port = Number(process.env.ARGUS_API_PORT || 8787);
    if (!Number.isInteger(port) || port < 1 || port > 65535)
      throw new Error("ARGUS_API_PORT must be an integer from 1 to 65535.");
    const supportLog = createSupportLog(install.supportLog);
    const writeOutput = (data: Buffer) => {
      const safe = supportLog.write(data);
      process.stderr.write(safe);
    };
    const owner = await startOwnedApi({
      python: install.apiCommand,
      moduleArgs: install.apiArgs,
      root: install.engineRoot,
      port,
      site: install.site,
      db: install.db,
      // A frozen binary unpacks itself before it serves anything: measured
      // ~12s on a developer Mac from warm cache, and the pilot's box is a
      // 4-core Windows machine reading a ~1 GB tree off disk for the first
      // time. The 30s development default would read that as a failed start.
      timeoutMs: install.packaged ? 180_000 : 30_000,
      onStdout: writeOutput,
      onStderr: writeOutput,
    });
    ownedApi = owner;
    if (smokeState) {
      if (!Number.isInteger(owner.process.pid))
        throw new Error("Owned Argus API did not expose a process ID.");
      smokeState.recordPid(owner.process.pid!);
    }
    const client = createOwnedApiClient(
      owner,
      smokeState
        ? { onToken: (token) => smokeState.recordToken(token) }
        : undefined,
    );
    client.subscribe((event) =>
      window?.webContents.send("engine:event", event),
    );
    owner.onExit(() => {
      const remaining = supportLog.close();
      if (remaining) process.stderr.write(remaining);
      if (!exiting) {
        void client.close();
        apiClient = undefined;
        if (ownedApi === owner) ownedApi = undefined;
      }
    });
    owner.assertAlive();
    apiClient = client;
    return client;
  })().finally(() => {
    apiStarting = undefined;
  });
  return apiStarting;
}

app.whenReady().then(() => {
  if (transport === "bridge")
    void registerBridgeStreamProtocol(protocol, bridgeTransport);
  ipcMain.handle("engine:environment", (event) => {
    if (event.sender !== window?.webContents) throw new Error("Unknown caller");
    const install = layout();
    const quote = (value: string) =>
      process.platform === "win32"
        ? "'" + value.replaceAll("'", "''") + "'"
        : "'" + value.replaceAll("'", "'\"'\"'") + "'";
    // Installed, recovery runs through the bundled binary's own flag — there
    // is no repo, no Python and no loose script on a customer's machine.
    const recovery = install.packaged
      ? [install.apiCommand, "--recover-account", "--db", install.db]
      : [
          install.apiCommand,
          path.join(install.engineRoot, "Frontend/scripts/recover_account.py"),
          "--db",
          install.db,
        ];
    return {
      recovery_command:
        (process.platform === "win32" ? "& " : "") +
        recovery.map(quote).join(" "),
      repo: install.engineRoot,
      site: install.site,
      db: install.db,
    };
  });
  ipcMain.handle("engine:invoke", (event, method, args) => {
    if (
      event.sender !== window?.webContents ||
      event.senderFrame !== window.webContents.mainFrame
    )
      throw new Error("Unknown caller");
    if (
      typeof method !== "string" ||
      !methods.has(method) ||
      !Array.isArray(args) ||
      JSON.stringify(args).length > 1000000
    )
      throw new Error("Invalid engine request");
    if (transport === "bridge") return bridgeTransport.invoke(method, args);
    return ensureApi().then((client) => client.invoke(method, args));
  });
  const create = () => {
    window = new BrowserWindow({
      width: 1480,
      height: 960,
      minWidth: 860,
      minHeight: 640,
      backgroundColor: "#f5f6f4",
      title: "ARGUS",
      webPreferences: {
        preload: path.join(dir, "preload.cjs"),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
      },
    });
    window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    window.webContents.on("will-navigate", (event) => event.preventDefault());
    // Packaged: the built renderer rides in the asar beside this file. The
    // --production flag keeps working for a source checkout that wants the
    // built assets instead of the Vite dev server.
    if (app.isPackaged || process.argv.includes("--production"))
      void window.loadFile(path.join(dir, "../dist/index.html"));
    else void window.loadURL("http://127.0.0.1:5173");
  };
  create();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) create();
  });
});
app.on("before-quit", (event) => {
  if (exiting) return;
  if (transport === "bridge") {
    if (!worker) return;
    event.preventDefault();
    exiting = true;
    const timeout = setTimeout(() => {
      worker?.kill();
      app.quit();
    }, 10000);
    void invoke("shutdown", []).finally(() => {
      clearTimeout(timeout);
      worker?.kill();
      app.quit();
    });
    return;
  }
  if (!apiClient && !ownedApi && !apiStarting) return;
  event.preventDefault();
  exiting = true;
  void (async () => {
    try {
      const client = apiClient ?? (await apiStarting);
      await client?.close();
    } finally {
      await ownedApi?.stop();
      app.quit();
    }
  })();
});
app.on("window-all-closed", () => app.quit());
process.on("uncaughtException", (error) => {
  dialog.showErrorBox("Argus desktop", error.message);
});
