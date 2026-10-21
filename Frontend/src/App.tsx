import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  LayoutGrid,
  Camera as CameraIcon,
  ShieldAlert,
  SlidersHorizontal,
  Settings,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Plus,
  Play,
  Square,
  Search,
  RefreshCw,
  ShieldCheck,
  Radio,
  ArrowUpRight,
  Check,
  LogOut,
  Moon,
  Sun,
  Menu,
  Layers,
  Trash2,
  ScanLine,
  Maximize2,
} from "lucide-react";
import { client } from "./lib/client";
import { incidentAssessment } from "./lib/incident-assessment";
import {
  DETECTORS,
  type Camera,
  type Incident,
  type Json,
  type Mode,
  type View,
  type Workspace,
} from "./lib/types";
import {
  Badge,
  CameraMedia,
  Drawer,
  Empty,
  Notice,
  Spinner,
} from "./components/common";
import CameraDetails from "./components/CameraDetails";
import IncidentDetails from "./components/IncidentDetails";
import AddCamera from "./components/AddCamera";
import Setup from "./components/Setup";
import FeedSwitcher from "./components/FeedSwitcher";
import HierarchyReview from "./components/HierarchyReview";
import SettingsPanel from "./components/SettingsPanel";
import { AccountHelp } from "./components/AccountAccess";
import CameraStream from "./components/CameraStream";
import StreamsWall from "./components/StreamsWall";
import { useVisibleStreams } from "./hooks/useVisibleStreams";
import { applyPushEvent } from "./lib/push";
import { initialMode, rememberMode } from "./lib/mode";
import {
  cameraTrackingPreference,
  loadGlobalTrackingPreference,
  saveGlobalTrackingPreference,
  trackingVisible,
  updateTrackingOverride,
  type TrackingOverrides,
  type TrackingPreference,
} from "./lib/tracking-overlay";
import {
  ALL_LOCATIONS,
  areaOptions,
  branchOptions,
  decodeLocationSelection,
  encodeLocationSelection,
  filterCameras,
  loadWallFilterPreference,
  reconcileAreaSelection,
  reconcileBranchSelection,
  sameLocationSelection,
  saveWallFilterPreference,
  type LocationSelection,
} from "./lib/hierarchy";

const blank: Workspace = {
  cameras: [],
  events: [],
  areas: [],
  hierarchy: {
    organization: { id: "", name: "" },
    branches: [],
    unassigned_cameras: [],
  },
  site: {},
  monitor: {},
  english: {},
  auth: {
    configured: true,
    signed_in: false,
    username: "",
    role: "",
    permissions: [],
  },
};
const names: Record<View, string> = {
  watch: "Overview",
  incidents: "Incidents",
  cameras: "Cameras",
  rules: "Intelligence",
  settings: "Settings",
  setup: "Site setup",
};
const resolved = (e: Incident) =>
  e.triage_state === "resolved" || ["true", "false"].includes(e.review);
const title = (e: Incident) => e.rule === "product_concealment"
  ? "Possible product concealment"
  : e.title || e.rule.replaceAll("_", " ");

type TrackingPreferenceState = {
  operatorId: string;
  globalVisible: boolean;
  overrides: TrackingOverrides;
};

const emptyTrackingPreferences = (): TrackingPreferenceState => ({
  operatorId: "",
  globalVisible: false,
  overrides: {},
});

// True when this bundle is the one the engine serves over the tunnel (set by
// `build:web`). Such a build has a real backend at its own origin even though
// there is no Electron bridge, so it behaves like the desktop app for mode and
// sign-in rather than falling back to the demo sandbox.
const SERVED_BY_ENGINE = import.meta.env.VITE_ARGUS_SERVED === "1";

export default function App() {
  const [mode, setMode] = useState<Mode>(() =>
    initialMode(localStorage, Boolean(window.argusDesktop) || SERVED_BY_ENGINE),
  );
  const api = useMemo(() => client(mode), [mode]);
  const [view, setView] = useState<View>("watch");
  const [ws, setWs] = useState<Workspace>(blank);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [query, setQuery] = useState("");
  const [branch, setBranch] = useState<LocationSelection>(ALL_LOCATIONS);
  const [area, setArea] = useState<LocationSelection>(ALL_LOCATIONS);
  const [preferenceUser, setPreferenceUser] = useState("");
  const [trackingPreferences, setTrackingPreferences] =
    useState<TrackingPreferenceState>(emptyTrackingPreferences);
  const [filter, setFilter] = useState("open");
  const [dark, setDark] = useState(
    localStorage.getItem("argus.theme") === "dark",
  );
  const [nav, setNav] = useState(false);
  const [selected, setSelected] = useState<{ id: string; tab: string } | null>(
    null,
  );
  const [eventId, setEventId] = useState<string | null>(null);
  const [add, setAdd] = useState(false);
  const [hierarchyReview, setHierarchyReview] = useState(false);
  const [cameraDirty, setCameraDirty] = useState(false);
  const [streamsOnly, setStreamsOnly] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const generation = useRef(0);
  const firstRunRouted = useRef(false);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const canConfigureCameras = ws.auth.permissions.includes("configure_cameras");
  const trackingPreferencesReady =
    ws.auth.signed_in &&
    Boolean(ws.auth.username) &&
    trackingPreferences.operatorId === ws.auth.username;
  const trackingGlobal = trackingPreferences.globalVisible;
  const trackingOverrides = trackingPreferences.overrides;
  const refresh = useCallback(async () => {
    const current = generation.current;
    const auth = await api.invoke("auth_state");
    if (current !== generation.current) return;
    if (!auth.signed_in) {
      setTrackingPreferences(emptyTrackingPreferences());
      setWs({ ...blank, auth });
      return;
    }
    setTrackingPreferences((preferences) =>
      preferences.operatorId === auth.username
        ? preferences
        : {
            operatorId: auth.username,
            globalVisible: loadGlobalTrackingPreference(
              localStorage,
              auth.username,
            ),
            overrides: {},
          },
    );
    const [cameras, events, areas, hierarchy, site, monitor, english] =
      await Promise.all([
        api.invoke("list_cameras"),
        auth.permissions.includes("view_alerts")
          ? api.invoke("list_events", [100])
          : Promise.resolve([]),
        api.invoke("list_areas"),
        api.invoke("hierarchy"),
        api.invoke("get_site"),
        api.invoke("monitoring_status"),
        api.invoke("english_rules_status"),
      ]);
    if (current !== generation.current) return;
    setWs({ cameras, events, areas, hierarchy, site, monitor, english, auth });
    setLastSync(new Date());
  }, [api]);
  useEffect(() => {
    generation.current++;
    firstRunRouted.current = false;
    setWs(blank);
    setLoading(true);
    setError("");
    setSelected(null);
    setEventId(null);
    let active = true;
    let inFlight = false;
    const sync = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        await refresh();
        if (active) setError("");
      } catch (e) {
        if (active) setError((e as Error).message);
      } finally {
        inFlight = false;
        if (active) setLoading(false);
      }
    };
    void sync();
    const timer = setInterval(() => void sync(), 8000);
    return () => {
      active = false;
      generation.current++;
      clearInterval(timer);
    };
  }, [refresh]);
  useEffect(() => {
    if (!api.subscribe) return;
    return api.subscribe((event) => {
      setWs((current) => applyPushEvent(current, event));
      setLastSync(new Date());
    });
  }, [api]);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("argus.theme", dark ? "dark" : "light");
  }, [dark]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 4500);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    if (canConfigureCameras) return;
    if (add) setAdd(false);
    if (view === "setup") setView("settings");
  }, [add, canConfigureCameras, view]);
  // A site nobody has set up yet opens on the wizard, not on an empty Overview.
  // Once only: after that the operator navigates where they like.
  useEffect(() => {
    if (firstRunRouted.current) return;
    if (loading || !ws.auth.signed_in || !canConfigureCameras) return;
    firstRunRouted.current = true;
    if (!ws.site.configured) setView("setup");
  }, [canConfigureCameras, loading, ws.auth.signed_in, ws.site.configured]);
  useEffect(() => {
    if (!ws.auth.signed_in || !ws.auth.username) {
      setPreferenceUser("");
      setBranch(ALL_LOCATIONS);
      setArea(ALL_LOCATIONS);
      return;
    }
    const saved = loadWallFilterPreference(localStorage, ws.auth.username);
    setBranch(saved.branch);
    setArea(saved.area);
    setPreferenceUser(ws.auth.username);
  }, [ws.auth.signed_in, ws.auth.username]);
  useEffect(() => {
    if (preferenceUser !== ws.auth.username || !ws.auth.signed_in) return;
    saveWallFilterPreference(localStorage, ws.auth.username, {
      branch,
      area,
    });
  }, [area, branch, preferenceUser, ws.auth.signed_in, ws.auth.username]);
  useEffect(() => {
    if (!trackingPreferencesReady) return;
    saveGlobalTrackingPreference(
      localStorage,
      trackingPreferences.operatorId,
      trackingPreferences.globalVisible,
    );
  }, [trackingPreferences, trackingPreferencesReady]);
  useEffect(() => {
    const compatibleBranch = reconcileBranchSelection(ws.hierarchy, branch);
    if (!sameLocationSelection(compatibleBranch, branch)) {
      setBranch(ALL_LOCATIONS);
      setArea(ALL_LOCATIONS);
      return;
    }
    const compatibleArea = reconcileAreaSelection(ws.hierarchy, branch, area);
    if (!sameLocationSelection(compatibleArea, area)) setArea(compatibleArea);
  }, [area, branch, ws.hierarchy]);
  async function action(
    method: string,
    args: unknown[] = [],
    message?: string,
  ) {
    setBusy(true);
    setError("");
    try {
      const r = await api.invoke(method, args);
      if (r?.ok === false)
        throw new Error(r.message || "The operation could not be completed.");
      await refresh();
      if (message) setToast(message);
      return r;
    } catch (e) {
      setError((e as Error).message);
      return null;
    } finally {
      setBusy(false);
    }
  }
  async function toggleMonitoring() {
    await action(ws.monitor.running ? "stop_monitoring" : "start_monitoring");
  }
  const wallCameras = filterCameras(
    ws.cameras,
    ws.hierarchy,
    branch,
    area,
    query,
  );
  const {
    activeIds: overviewActiveIds,
    visibleIds: overviewVisibleIds,
    page: overviewPage,
    pageCount: overviewPageCount,
    setPage: setOverviewPage,
    observeWall: observeOverviewWall,
  } = useVisibleStreams(
    wallCameras.map((camera) => camera.id),
    16,
  );
  const overviewActive = new Set(overviewActiveIds);
  const overviewCameras = overviewVisibleIds
    .map((id) => wallCameras.find((camera) => camera.id === id))
    .filter((camera): camera is Camera => Boolean(camera));
  const cameraMatches = ws.cameras.filter((camera) =>
    `${camera.id} ${camera.area || ""}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const wallBranches = branchOptions(ws.cameras, ws.hierarchy);
  const wallAreas = areaOptions(ws.cameras, ws.hierarchy, branch);
  const hierarchyAreas = ws.hierarchy.branches.flatMap((item) => item.areas);
  const open = ws.events.filter((e) => !resolved(e));
  const events = ws.events.filter(
    (e) =>
      (filter === "all" || (filter === "open" ? !resolved(e) : resolved(e))) &&
      `${e.camera_id} ${title(e)}`.toLowerCase().includes(query.toLowerCase()),
  );
  const currentCamera = ws.cameras.find((c) => c.id === selected?.id);
  const currentEvent = ws.events.find((e) => String(e.id) === eventId);
  const choose = (v: View) => {
    setView(v);
    setQuery("");
    setNav(false);
  };
  const configure = (c: Camera, tab = "scene") =>
    setSelected({ id: c.id, tab });
  const openCameraOnboarding = () => {
    if (canConfigureCameras) setAdd(true);
  };
  const setCameraTracking = (
    cameraId: string,
    preference: TrackingPreference,
  ) => {
    setTrackingPreferences((current) =>
      current.operatorId === ws.auth.username
        ? {
            ...current,
            overrides: updateTrackingOverride(
              current.overrides,
              cameraId,
              preference,
            ),
          }
        : current,
    );
  };
  const setTrackingGlobal = (visible: boolean) => {
    setTrackingPreferences((current) =>
      current.operatorId === ws.auth.username
        ? { ...current, globalVisible: visible, overrides: {} }
        : current,
    );
  };
  const changeMode = (nextMode: Mode) => {
    if (nextMode === mode) return;
    generation.current++;
    setWs(blank);
    setLoading(true);
    setError("");
    setSelected(null);
    setEventId(null);
    setStreamsOnly(false);
    setPreferenceUser("");
    setBranch(ALL_LOCATIONS);
    setArea(ALL_LOCATIONS);
    setTrackingPreferences(emptyTrackingPreferences());
    setLastSync(null);
    rememberMode(localStorage, nextMode);
    setMode(nextMode);
  };
  if (streamsOnly && trackingPreferencesReady) {
    return (
      <StreamsWall
        hierarchy={ws.hierarchy}
        cameras={ws.cameras}
        api={api}
        mode={mode}
        running={Boolean(ws.monitor.running)}
        trackingGlobal={trackingGlobal}
        trackingOverrides={trackingOverrides}
        onTrackingGlobalChange={setTrackingGlobal}
        onTrackingOverrideChange={setCameraTracking}
        branch={branch}
        area={area}
        onBranchChange={(nextBranch) => {
          setBranch(nextBranch);
          setArea((current) =>
            reconcileAreaSelection(ws.hierarchy, nextBranch, current),
          );
        }}
        onAreaChange={setArea}
        onExit={() => setStreamsOnly(false)}
      />
    );
  }
  return (
    <div className="app-shell">
      <aside className={`sidebar ${nav ? "mobile-open" : ""}`}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            choose("watch");
          }}
        >
          <img src="argus-mark.png" alt="" />
          <span>
            ARGUS<small>INDEPENDENT INTELLIGENCE</small>
          </span>
        </a>
        <button className="site-picker" onClick={() => choose("settings")}>
          <span className="site-monogram">
            {(ws.site.name || "A").slice(0, 1)}
          </span>
          <span>
            {ws.site.name || "Your workspace"}
            <small>
              {mode === "demo" ? "Demo site" : "Local installation"}
            </small>
          </span>
          <ChevronDown size={15} />
        </button>
        <div className="nav-caption">WORKSPACE</div>
        <nav>
          {(
            [
              ["watch", LayoutGrid],
              ["incidents", ShieldAlert],
              ["cameras", CameraIcon],
              ["rules", ScanLine],
            ] as const
          ).map(([v, Icon]) => (
            <button
              key={v}
              className={view === v ? "active" : ""}
              onClick={() => choose(v)}
            >
              <Icon size={19} />
              {names[v]}
              {v === "incidents" && open.length > 0 && (
                <span className="count">{open.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="engine-card">
            <ShieldCheck size={19} />
            <strong>
              {mode === "demo" ? "Demo workspace" : "Local backend"}
            </strong>
            <p>
              {mode === "demo"
                ? "Sample footage. No AI inference."
                : ws.monitor.starting
                  ? "Monitoring is starting."
                  : ws.monitor.running
                    ? "Monitoring process running."
                    : "Monitoring is stopped."}
            </p>
            <span className="engine-line">
              <span
                className={`status-dot ${mode === "demo" ? "sample" : ws.monitor.running ? "" : "off"}`}
              />
              {mode === "demo"
                ? "Disconnected from engine"
                : error
                  ? "Connection needs attention"
                  : ws.auth.signed_in
                    ? "Backend connected"
                    : "Sign in required"}
            </span>
          </div>
          <button
            className={
              view === "settings" ? "side-setting active" : "side-setting"
            }
            onClick={() => choose("settings")}
          >
            <Settings size={18} />
            Settings
          </button>
          <div className="operator">
            <span className="avatar">
              {ws.auth.username?.slice(0, 2).toUpperCase() || "--"}
            </span>
            <span>
              {ws.auth.username || "Not signed in"}
              <small>{ws.auth.role || "Local workspace"}</small>
            </span>
            {mode === "engine" && ws.auth.signed_in && (
              <button
                className="icon-button"
                title="Sign out"
                onClick={() => void action("sign_out")}
              >
                <LogOut size={17} />
              </button>
            )}
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            title="Open navigation"
            onClick={() => setNav(!nav)}
          >
            <Menu size={20} />
          </button>
          <div className="breadcrumb">
            Workspace
            <ChevronRight size={13} />
            <strong>{names[view]}</strong>
          </div>
          <div className="topbar-actions">
            <div
              className="mode-switch"
              role="group"
              aria-label="Workspace mode"
            >
              <button
                className={mode === "demo" ? "active" : ""}
                onClick={() => changeMode("demo")}
              >
                Demo
              </button>
              <button
                className={mode === "engine" ? "active" : ""}
                onClick={() => changeMode("engine")}
              >
                Local engine
              </button>
            </div>
            <button
              className="icon-button"
              title={dark ? "Use light theme" : "Use dark theme"}
              aria-label="Toggle theme"
              onClick={() => setDark(!dark)}
            >
              {dark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </header>
        <main>
          {error && (
            <Notice error>
              {error}{" "}
              <button
                className="text-button"
                onClick={() =>
                  void refresh()
                    .then(() => setError(""))
                    .catch((e) => setError(e.message))
                }
              >
                Retry connection
              </button>
            </Notice>
          )}
          {loading ? (
            <div className="loading">
              <Spinner />
              Connecting workspace...
            </div>
          ) : !ws.auth.signed_in ? (
            <div className="auth-layout">
              <span className="eyebrow">YOUR LOCAL WORKSPACE</span>
              <h1>
                {ws.auth.configured
                  ? "Welcome back."
                  : "Make this workspace yours."}
              </h1>
              <p>
                Sign in to connect the cameras, scene mapper and verification
                engine.
              </p>
              {!window.argusDesktop && !SERVED_BY_ENGINE ? (
                <Notice>
                  The browser preview supports demo mode only. Launch the
                  Electron app to use the local engine.
                </Notice>
              ) : (
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    void action(
                      ws.auth.configured ? "sign_in" : "create_first_owner",
                      [username, password],
                    ).then(() => setPassword(""));
                  }}
                >
                  <label>
                    Username
                    <input
                      autoComplete="username"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      required
                    />
                  </label>
                  <label>
                    Password
                    <input
                      type="password"
                      autoComplete={
                        ws.auth.configured ? "current-password" : "new-password"
                      }
                      minLength={ws.auth.configured ? 1 : 12}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                  </label>
                  <button className="button primary" disabled={busy}>
                    {busy ? <Spinner /> : <ShieldCheck size={17} />}{" "}
                    {ws.auth.configured ? "Sign in" : "Create owner account"}
                  </button>
                </form>
              )}
              {window.argusDesktop && ws.auth.configured && <AccountHelp />}
            </div>
          ) : (
            <>
              <div className="page-heading">
                <div>
                  <span className="eyebrow">
                    {mode === "demo"
                      ? "RECORDED DEMO / NO INFERENCE"
                      : "ARGUS / LOCAL INTELLIGENCE"}
                  </span>
                  <h1>
                    {view === "watch"
                      ? "Every camera. One clear picture."
                      : names[view]}
                  </h1>
                  <p>
                    {view === "watch"
                      ? "Your site, its activity, and what needs your attention."
                      : view === "incidents"
                        ? "Review the evidence. Make the call. Keep the record."
                        : view === "cameras"
                          ? "Camera connections and the places they watch."
                          : view === "rules"
                            ? "Scene-aware detection, reviewed by you."
                            : view === "setup"
                              ? "Connect your site and prepare it for monitoring."
                              : "Manage this installation and its verification engine."}
                  </p>
                </div>
                <div className="heading-actions">
                  {view === "watch" ? (
                    <button
                      className={`button ${ws.monitor.running ? "" : "primary"}`}
                      disabled={
                        busy ||
                        (mode === "engine" &&
                          !ws.auth.permissions.includes("control_engine"))
                      }
                      onClick={() => void toggleMonitoring()}
                    >
                      {busy ? (
                        <Spinner />
                      ) : ws.monitor.running ? (
                        <Square size={15} />
                      ) : (
                        <Play size={15} />
                      )}{" "}
                      {mode === "demo"
                        ? ws.monitor.running
                          ? "Pause footage"
                          : "Play footage"
                        : ws.monitor.running
                          ? "Stop monitoring"
                          : "Start monitoring"}
                    </button>
                  ) : ["cameras", "rules"].includes(view) &&
                    canConfigureCameras ? (
                    <button
                      className="button primary"
                      onClick={openCameraOnboarding}
                    >
                      <Plus size={16} />
                      Add camera
                    </button>
                  ) : view === "settings" && canConfigureCameras ? (
                    <button className="button" onClick={() => choose("setup")}>
                      <Layers size={16} />
                      Site setup
                    </button>
                  ) : null}
                </div>
              </div>
              {view === "watch" && (
                <>
                  {mode === "engine" && (
                    <FeedSwitcher
                      api={api}
                      onChange={refresh}
                      disabled={!ws.auth.permissions.includes("configure_site")}
                    />
                  )}{" "}
                  {(ws.monitor.stalled ||
                    ws.monitor.crash_looping ||
                    (!ws.monitor.running && ws.monitor.last_error)) && (
                    <Notice error>
                      {ws.monitor.stalled || ws.monitor.crash_looping
                        ? "Monitoring needs attention."
                        : "Monitoring is stopped. Last recorded log message (may be historical):"}{" "}
                      {ws.monitor.last_error ||
                        "The engine is not reporting a healthy heartbeat."}
                    </Notice>
                  )}
                  <div className="metrics">
                    <div>
                      <span>Cameras configured</span>
                      <strong>
                        {ws.cameras.length}
                        <CameraIcon size={20} />
                      </strong>
                      <small>{ws.areas.length} areas in this site</small>
                    </div>
                    <div>
                      <span>Awaiting review</span>
                      <strong>
                        {open.length}
                        <ShieldAlert size={20} />
                      </strong>
                      <small>
                        {mode === "demo"
                          ? "Sample incident queue"
                          : "Unresolved incidents"}
                      </small>
                    </div>
                    <div>
                      <span>Monitoring</span>
                      <strong className="metric-word">
                        {mode === "demo"
                          ? "Demo"
                          : ws.monitor.starting
                            ? "Starting"
                            : ws.monitor.running
                              ? "Running"
                              : "Stopped"}
                        <Radio size={20} />
                      </strong>
                      <small>
                        {mode === "demo"
                          ? "No live detection claims"
                          : ws.monitor.phase || "Engine status"}
                      </small>
                    </div>
                    <div>
                      <span>Last refreshed</span>
                      <strong className="metric-word">
                        {lastSync?.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        }) || "--"}
                        <RefreshCw size={18} />
                      </strong>
                      <small>
                        {mode === "demo"
                          ? "Local sample data"
                          : "Backend status, refreshed every 8s"}
                      </small>
                    </div>
                  </div>
                  <div className="overview-layout">
                    <section className="camera-section">
                      <div className="section-bar">
                        <div className="section-title-action">
                          <h2>
                            Camera wall <span>{wallCameras.length}</span>
                          </h2>
                          <button
                            className="icon-button"
                            aria-label="Open streams wall"
                            title="Open streams wall"
                            onClick={() => setStreamsOnly(true)}
                          >
                            <Maximize2 size={16} />
                          </button>
                        </div>
                        <div className="section-tools">
                          <label className="person-boxes-control">
                            <input type="checkbox" role="switch" aria-label="Person boxes: all cameras"
                              checked={trackingGlobal} onChange={e => setTrackingGlobal(e.target.checked)} />
                            Person boxes: all cameras
                          </label>
                          <div className="search-field wall-search">
                            <Search size={14} />
                            <input
                              aria-label="Search camera wall"
                              placeholder="Search cameras"
                              value={query}
                              onChange={(event) => setQuery(event.target.value)}
                            />
                          </div>
                          <select
                            aria-label="Filter branch"
                            value={encodeLocationSelection(branch)}
                            onChange={(event) => {
                              const nextBranch = decodeLocationSelection(
                                event.target.value,
                              );
                              if (!nextBranch) return;
                              setBranch(nextBranch);
                              setArea((current) =>
                                reconcileAreaSelection(
                                  ws.hierarchy,
                                  nextBranch,
                                  current,
                                ),
                              );
                            }}
                          >
                            {wallBranches.map((option) => (
                              <option
                                key={encodeLocationSelection(option.selection)}
                                value={encodeLocationSelection(
                                  option.selection,
                                )}
                              >
                                {option.name} ({option.count})
                              </option>
                            ))}
                          </select>
                          <select
                            aria-label="Filter area"
                            value={encodeLocationSelection(area)}
                            onChange={(event) => {
                              const selection = decodeLocationSelection(
                                event.target.value,
                              );
                              if (selection) setArea(selection);
                            }}
                          >
                            {wallAreas.map((option) => (
                              <option
                                key={encodeLocationSelection(option.selection)}
                                value={encodeLocationSelection(
                                  option.selection,
                                )}
                              >
                                {option.name} ({option.count})
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                      <div className="camera-grid" ref={observeOverviewWall}>
                        {overviewCameras.map((c) => (
                          <article className="camera-tile" key={c.id}>
                            {mode === "demo" ? (
                              <CameraMedia
                                camera={c}
                                paused={!ws.monitor.running}
                                onOpen={() => configure(c)}
                              />
                            ) : (
                              <CameraStream
                                camera={c}
                                api={api}
                                active={
                                  trackingPreferencesReady &&
                                  overviewActive.has(c.id)
                                }
                                tracking={trackingVisible(
                                  trackingGlobal,
                                  cameraTrackingPreference(
                                    trackingOverrides,
                                    c.id,
                                  ),
                                )}
                                onOpen={() => configure(c)}
                              />
                            )}
                            <div className="camera-caption">
                              <div>
                                <strong>{c.id}</strong>
                                <small>
                                  {hierarchyAreas.find(
                                    (item) => item.id === c.area_id,
                                  )?.name || "Unassigned"}
                                </small>
                              </div>
                              <div className="camera-actions">
                                <label
                                  className="person-boxes-control"
                                  data-preference={cameraTrackingPreference(
                                    trackingOverrides,
                                    c.id,
                                  )}
                                  title={`Tracking: ${cameraTrackingPreference(
                                    trackingOverrides,
                                    c.id,
                                  )}`}
                                >
                                  <input type="checkbox" role="switch"
                                    aria-label={`Person boxes: ${c.id}`}
                                    checked={trackingVisible(trackingGlobal, cameraTrackingPreference(trackingOverrides, c.id))}
                                    onChange={e => setCameraTracking(c.id, e.target.checked ? "show" : "hide")} />
                                  Person boxes
                                </label>
                                <button
                                  className="icon-button"
                                  title={`Configure ${c.id}`}
                                  aria-label={`Configure ${c.id}`}
                                  onClick={() => configure(c)}
                                >
                                  <SlidersHorizontal size={16} />
                                </button>
                              </div>
                            </div>
                          </article>
                        ))}
                      </div>
                      {overviewPageCount > 1 && (
                        <div
                          className="overview-pager"
                          aria-label="Overview camera pages"
                        >
                          <button
                            className="icon-button"
                            aria-label="Previous overview camera page"
                            title="Previous camera page"
                            disabled={overviewPage <= 1}
                            onClick={() => setOverviewPage(overviewPage - 1)}
                          >
                            <ChevronLeft size={16} />
                          </button>
                          <span>
                            {overviewPage} / {overviewPageCount}
                          </span>
                          <button
                            className="icon-button"
                            aria-label="Next overview camera page"
                            title="Next camera page"
                            disabled={overviewPage >= overviewPageCount}
                            onClick={() => setOverviewPage(overviewPage + 1)}
                          >
                            <ChevronRight size={16} />
                          </button>
                        </div>
                      )}
                      {wallCameras.length === 0 && (
                        <Empty title="No matching cameras">
                          Change the branch, area, or search filter.
                        </Empty>
                      )}
                      <div className="wall-footer">
                        <span>
                          <ShieldCheck size={14} />
                          {mode === "demo"
                            ? "Recorded clips, clearly separated from live monitoring."
                            : "Camera preview alone does not indicate active detection."}
                        </span>
                        <button
                          className="text-button"
                          onClick={() => choose("cameras")}
                        >
                          Manage cameras
                          <ArrowUpRight size={14} />
                        </button>
                      </div>
                    </section>
                    <aside className="activity">
                      <div className="section-bar">
                        <h2>Needs attention</h2>
                        <Badge tone="amber">{open.length}</Badge>
                      </div>
                      {open.length ? (
                        open.slice(0, 5).map((e) => (
                          <button
                            className="activity-item"
                            key={e.id}
                            onClick={() => setEventId(String(e.id))}
                          >
                            <div className="activity-top">
                              <Badge
                                tone={
                                  e.priority === "critical" ? "red" : "amber"
                                }
                              >
                                {e.priority}
                              </Badge>
                              <small>
                                {mode === "demo"
                                  ? "SAMPLE"
                                  : new Date(e.ts * 1000).toLocaleTimeString(
                                      [],
                                      { hour: "2-digit", minute: "2-digit" },
                                    )}
                              </small>
                            </div>
                            <strong>{title(e)}</strong>
                            <span>{e.camera_id}</span>
                            <p>{incidentAssessment(e).summary}</p>
                            <span className="activity-link">
                              Review incident
                              <ArrowUpRight size={15} />
                            </span>
                          </button>
                        ))
                      ) : (
                        <div className="quiet">
                          <ShieldCheck size={27} />
                          <h3>Queue clear</h3>
                          <p>No unresolved incidents in this workspace.</p>
                        </div>
                      )}
                      <button
                        className="button full"
                        onClick={() => choose("incidents")}
                      >
                        All incidents
                        <ChevronRight size={15} />
                      </button>
                      <div className="review-nudge">
                        <ScanLine size={22} />
                        <h3>Context comes first.</h3>
                        <p>
                          Check what each camera sees before trusting its scene
                          interpretation.
                        </p>
                        <button
                          className="text-button"
                          onClick={() => choose("rules")}
                        >
                          Review scene context
                          <ArrowUpRight size={15} />
                        </button>
                      </div>
                    </aside>
                  </div>
                </>
              )}
              {view === "incidents" && (
                <>
                  <div className="list-toolbar">
                    <div className="tabs">
                      {[
                        ["open", "Needs review"],
                        ["resolved", "Resolved"],
                        ["all", "All incidents"],
                      ].map(([k, l]) => (
                        <button
                          key={k}
                          className={filter === k ? "active" : ""}
                          onClick={() => setFilter(k)}
                        >
                          {l}
                        </button>
                      ))}
                    </div>
                    <div className="search-field">
                      <Search size={16} />
                      <input
                        aria-label="Search incidents"
                        placeholder="Search camera or incident"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="incident-table">
                    <div className="table-header">
                      <span>Incident</span>
                      <span>Camera</span>
                      <span>Severity</span>
                      <span>Review</span>
                      <span>Time</span>
                    </div>
                    {events.map((e) => (
                      <button
                        className="incident-row"
                        key={e.id}
                        onClick={() => setEventId(String(e.id))}
                      >
                        <span>
                          <strong>{title(e)}</strong>
                          <small>
                            {mode === "demo"
                              ? "Sample incident"
                              : `Incident #${e.id}`}
                          </small>
                        </span>
                        <span>{e.camera_id}</span>
                        <span>
                          <Badge
                            tone={e.priority === "critical" ? "red" : "amber"}
                          >
                            {e.priority}
                          </Badge>
                        </span>
                        <span>
                          {resolved(e)
                            ? "Resolved"
                            : e.triage_state === "acknowledged"
                              ? "Acknowledged"
                              : "Needs review"}
                        </span>
                        <span>
                          {new Date(e.ts * 1000).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                          <ChevronRight size={15} />
                        </span>
                      </button>
                    ))}
                  </div>
                  {events.length === 0 && (
                    <Empty title="Nothing in this view">
                      Try another filter or wait for the next incident.
                    </Empty>
                  )}
                </>
              )}
              {["cameras", "rules"].includes(view) && (
                <>
                  <div className="list-toolbar">
                    <div className="search-field">
                      <Search size={16} />
                      <input
                        aria-label="Search cameras"
                        placeholder="Search cameras"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                      />
                    </div>
                    {mode === "engine" && (
                      <button
                        className="button"
                        onClick={() => setHierarchyReview(true)}
                      >
                        <ScanLine size={16} />
                        Site & area review
                      </button>
                    )}
                  </div>
                  <div className="management-list">
                    {cameraMatches.map((c) => (
                      <article className="management-row" key={c.id}>
                        <img
                          src={c.snapshot}
                          alt=""
                          className={c.snapshot ? "" : "no-snapshot"}
                        />
                        <div className="management-info">
                          <h3>{c.id}</h3>
                          <p>
                            {ws.areas.find((a) => a.id === c.area_id)?.name ||
                              "Ungrouped area"}
                          </p>
                          <small>
                            {DETECTORS.filter((d) => c[d.key]).length} detectors
                            enabled · {(c.custom_rules || []).length} English
                            rules
                          </small>
                        </div>
                        <div className="management-actions">
                          <button
                            className="button"
                            onClick={() => configure(c, "scene")}
                          >
                            <ScanLine size={16} />
                            Scene review
                          </button>
                          <button
                            className="button"
                            onClick={() =>
                              configure(
                                c,
                                view === "rules" ? "detectors" : "zones",
                              )
                            }
                          >
                            {view === "rules" ? (
                              <SlidersHorizontal size={16} />
                            ) : (
                              <Layers size={16} />
                            )}{" "}
                            {view === "rules" ? "Detectors" : "Zones"}
                          </button>
                          {ws.auth.permissions.includes(
                            "configure_cameras",
                          ) && (
                            <button
                              className="icon-button"
                              title={`Remove ${c.id}`}
                              disabled={busy}
                              onClick={() => {
                                if (
                                  confirm(
                                    `Remove camera ${c.id} from this site?`,
                                  )
                                )
                                  void action(
                                    "remove_camera",
                                    [c.id],
                                    "Camera removed",
                                  );
                              }}
                            >
                              <Trash2 size={16} />
                            </button>
                          )}
                        </div>
                      </article>
                    ))}
                  </div>
                  {!cameraMatches.length && (
                    <Empty title="No matching cameras">
                      Add a camera or change the search.
                    </Empty>
                  )}
                </>
              )}
              {view === "settings" && (
                <SettingsPanel
                  api={api}
                  mode={mode}
                  site={ws.site}
                  auth={ws.auth}
                  hierarchy={ws.hierarchy}
                  cameras={ws.cameras}
                  onChange={refresh}
                  notify={setToast}
                />
              )}
              {view === "setup" && (
                <Setup
                  api={api}
                  mode={mode}
                  cameras={ws.cameras}
                  canConfigureCameras={canConfigureCameras}
                  auth={ws.auth}
                  hierarchy={ws.hierarchy}
                  site={ws.site}
                  notify={setToast}
                  onAdd={openCameraOnboarding}
                  onConfigure={configure}
                  onChange={refresh}
                  onFinish={() => choose("watch")}
                />
              )}
            </>
          )}
        </main>
        <footer className="app-footer">
          <span>
            ARGUS / {mode === "demo" ? "DEMO WORKSPACE" : "LOCAL WORKSPACE"}
          </span>
          <span>Intelligence behind every lens.</span>
        </footer>
      </div>
      {toast && (
        <div className="toast" role="status">
          <Check size={17} />
          {toast}
        </div>
      )}
      {hierarchyReview && (
        <Drawer
          title="Site & area review"
          subtitle="AGENT MAPPER"
          onClose={() => setHierarchyReview(false)}
        >
          <HierarchyReview
            api={api}
            onChange={refresh}
            onCamera={(id) => {
              setHierarchyReview(false);
              setSelected({ id, tab: "scene" });
            }}
          />
        </Drawer>
      )}
      {currentCamera && selected && (
        <Drawer
          expanded={selected.tab === "zones" || selected.tab === "objects"}
          title={currentCamera.id}
          subtitle="CAMERA INTELLIGENCE"
          onClose={() => {
            if (cameraDirty && !confirm("Discard unsaved camera edits?"))
              return;
            setSelected(null);
            setCameraDirty(false);
          }}
        >
          <CameraDetails
            editable={
              mode === "demo" ||
              ws.auth.permissions.includes("configure_cameras")
            }
            camera={currentCamera}
            api={api}
            initialTab={selected.tab}
            onTabChange={(tab) => setSelected({ id: currentCamera.id, tab })}
            onChange={refresh}
            notify={setToast}
            onDirtyChange={setCameraDirty}
          />
        </Drawer>
      )}
      {currentEvent && (
        <Drawer
          title={title(currentEvent)}
          subtitle={`INCIDENT / ${currentEvent.id}`}
          onClose={() => setEventId(null)}
        >
          <IncidentDetails event={currentEvent} api={api} onChange={refresh} />
        </Drawer>
      )}
      {add && canConfigureCameras && (
        <Drawer
          title="Connect a camera"
          subtitle="CAMERA SETUP"
          onClose={() => setAdd(false)}
        >
          <AddCamera
            api={api}
            mode={mode}
            hierarchy={ws.hierarchy}
            authorized={canConfigureCameras}
            onAdded={async () => {
              await refresh();
              setAdd(false);
              setToast("Camera added");
            }}
          />
        </Drawer>
      )}
    </div>
  );
}
