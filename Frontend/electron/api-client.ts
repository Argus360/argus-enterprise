export type ApiError = Error & {
  status?: number;
  code?: string;
  permission?: string;
};
export type PushEvent = {
  type: "health" | "triage" | "alert.new" | "alert.update";
  data: unknown;
};

type Operation = {
  method: "GET" | "POST" | "PUT" | "DELETE";
  path: (args: unknown[]) => string;
  body?: (args: unknown[]) => unknown;
  query?: (args: unknown[]) => Record<string, unknown>;
  normalize?: (value: any, client: ArgusApiClient) => unknown;
};
const fixed = (path: string) => () => path;
const item =
  (prefix: string, suffix = "") =>
  (args: unknown[]) =>
    `${prefix}/${encodeURIComponent(String(args[0]))}${suffix}`;

function normalizeIncident(value: any) {
  const state =
    value?.triage?.state ?? value?.triage_state ?? value?.review ?? "new";
  return {
    ...value,
    evidence_dir: value?.evidence?.dir ?? value?.evidence_dir,
    review: value?.review ?? state,
    triage_state:
      state === "ack"
        ? "acknowledged"
        : ["true", "false"].includes(state)
          ? "resolved"
          : state,
  };
}

function loopbackUrl(value: unknown, protocols: string[]): string {
  if (typeof value !== "string")
    throw new Error("unsafe stream descriptor URL");
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error("unsafe stream descriptor URL");
  }
  if (
    !protocols.includes(parsed.protocol) ||
    !["127.0.0.1", "localhost", "::1", "[::1]"].includes(parsed.hostname) ||
    parsed.username ||
    parsed.password
  )
    throw new Error("unsafe stream descriptor URL");
  return parsed.toString();
}

// Remote-browser (proxy) mode: the API returns a same-origin, root-relative
// MJPEG path (/api/v1/cameras/<id>/mjpeg?token=...) that the browser resolves
// against its own origin — the tunnel host — instead of a 127.0.0.1 URL. Only
// a root-relative path is accepted so a hostile descriptor can't aim the <img>
// at another origin. WebRTC never reaches here: proxy mode asks for MJPEG.
function sameOriginStreamUrl(value: unknown): string {
  if (typeof value !== "string" || !value.startsWith("/") ||
      value.startsWith("//"))
    throw new Error("unsafe stream descriptor URL");
  return value;
}

function normalizeProxyStreamDescriptor(value: any) {
  if (!value || typeof value !== "object")
    throw new Error("unsafe stream descriptor shape");
  if (value.kind === "mjpeg")
    return { kind: "mjpeg", url: sameOriginStreamUrl(value.url),
      ...(value.preview === true ? { preview: true } : {}) };
  throw new Error("unsafe stream descriptor kind");
}

function normalizeStreamDescriptor(value: any) {
  if (!value || typeof value !== "object")
    throw new Error("unsafe stream descriptor shape");
  if (value.kind === "mjpeg")
    return { kind: "mjpeg", url: loopbackUrl(value.url, ["http:"]),
      ...(value.preview === true ? { preview: true } : {}) };
  if (value.kind !== "webrtc") throw new Error("unsafe stream descriptor kind");
  const normalized: {
    kind: "webrtc";
    url: string;
    mjpeg_fallback?: string;
  } = {
    kind: "webrtc",
    url: loopbackUrl(value.url, ["http:"]),
  };
  if (value.mjpeg_fallback !== undefined && value.mjpeg_fallback !== null)
    normalized.mjpeg_fallback = loopbackUrl(value.mjpeg_fallback, ["http:"]);
  if (value.ws !== undefined && value.ws !== null)
    loopbackUrl(value.ws, ["ws:"]);
  return normalized;
}

const operations: Record<string, Operation> = {
  // --- parity with the retired PyQt console (17 Sep) ---------------------
  // Each of these had a backend method and no way to reach it from this UI.
  // Routes: cvti/api/writes.py, same order.
  auth_accounts: { method: "GET", path: fixed("/auth/accounts") },
  auth_recovery: { method: "GET", path: fixed("/auth/recovery") },
  change_own_password: {
    method: "POST",
    path: fixed("/auth/password"),
    body: ([current, next]) => ({ current, new: next }),
  },
  set_user_role: {
    method: "PUT",
    path: item("/users", "/role"),
    body: ([, role]) => ({ role }),
  },
  create_owner_override: {
    method: "POST",
    path: fixed("/auth/owner-override"),
    body: ([username, password]) => ({ username, password }),
  },
  counts: { method: "GET", path: fixed("/reports/counts") },
  needs_attention: {
    method: "GET",
    path: fixed("/reports/needs-attention"),
    query: ([min_priority]) => ({ min_priority }),
  },
  export_incident_pdf: { method: "POST", path: item("/events", "/report") },
  export_evidence: {
    method: "POST",
    path: fixed("/events/export"),
    body: ([event_ids, dest]) => ({ event_ids, dest }),
  },
  set_legal_hold: {
    method: "PUT",
    path: item("/events", "/legal-hold"),
    body: ([, hold]) => ({ hold }),
  },
  weekly_summary: { method: "GET", path: fixed("/reports/weekly") },
  handover: {
    method: "GET",
    path: fixed("/reports/handover"),
    query: ([hours]) => ({ hours }),
  },
  list_backups: { method: "GET", path: fixed("/backups") },
  restore_backup: {
    method: "POST",
    path: fixed("/backups/restore"),
    body: ([zip_path]) => ({ zip_path }),
  },
  set_backup_dir: {
    method: "PUT",
    path: fixed("/backups/directory"),
    body: ([path]) => ({ path }),
  },
  audit_export: { method: "POST", path: fixed("/audit/export") },
  audit_verify: { method: "GET", path: fixed("/audit/verify") },
  app_version: { method: "GET", path: fixed("/system/version") },
  detector_validation: { method: "GET", path: fixed("/system/detectors") },
  heartbeat_status: { method: "GET", path: fixed("/system/heartbeat") },
  set_heartbeat: {
    method: "PUT",
    path: fixed("/system/heartbeat"),
    body: ([url, key]) => ({ url, key }),
  },
  learning_stats: { method: "GET", path: fixed("/system/learning") },
  learning_calibrate: {
    method: "POST",
    path: fixed("/system/learning/calibrate"),
  },
  set_value_inputs: {
    method: "PUT",
    path: fixed("/value/inputs"),
    body: ([incident_value, guard_hourly_cost, review_minutes]) => ({
      incident_value,
      guard_hourly_cost,
      review_minutes,
    }),
  },
  live_frames: { method: "GET", path: fixed("/engine/frames") },
  set_custom_rule: {
    method: "PUT",
    path: item("/cameras", "/rules/question"),
    body: ([, question, dwell]) => ({ question, dwell }),
  },
  add_custom_threat: {
    method: "POST",
    path: item("/cameras", "/threats"),
    body: ([, name, description]) => ({ name, description }),
  },
  remove_custom_threat: {
    method: "DELETE",
    path: ([camera, index]) =>
      `/cameras/${encodeURIComponent(String(camera))}/threats/${encodeURIComponent(String(index))}`,
  },
  update_site_context: {
    method: "PUT",
    path: fixed("/site/context"),
    body: ([context]) => ({ context }),
  },
  get_site: { method: "GET", path: fixed("/site") },
  set_site: {
    method: "PUT",
    path: fixed("/site"),
    body: ([name, notify]) => ({ name, notify }),
  },
  list_cameras: { method: "GET", path: fixed("/cameras") },
  add_camera: {
    method: "POST",
    path: fixed("/cameras"),
    body: ([camera]) => ({ camera }),
  },
  remove_camera: { method: "DELETE", path: item("/cameras") },
  test: {
    method: "POST",
    path: fixed("/cameras/probe"),
    body: ([source]) => ({ source }),
  },
  discover_cameras: { method: "GET", path: fixed("/discovery/cameras") },
  detect_subnet: { method: "GET", path: fixed("/cameras/discovery/subnet") },
  scan: {
    method: "POST",
    path: fixed("/cameras/discovery/scan"),
    body: ([cidr]) => ({ cidr }),
  },
  list_areas: { method: "GET", path: fixed("/areas") },
  create_area: {
    method: "POST",
    path: fixed("/areas"),
    body: ([area]) => ({ area }),
  },
  assign_camera_area: {
    method: "PUT",
    path: item("/cameras", "/area"),
    body: ([, area_id]) => ({ area_id }),
  },
  scene_context: { method: "GET", path: item("/cameras", "/scene") },
  scene_review_summary: { method: "GET", path: fixed("/scene-mapping/review") },
  area_context: { method: "GET", path: item("/areas", "/context") },
  approve_area_context: {
    method: "POST",
    path: item("/areas", "/context/approve"),
    body: ([, context]) => ({ context }),
  },
  approve_site_context: {
    method: "POST",
    path: fixed("/site/context/approve"),
    body: ([context]) => ({ context }),
  },
  update_scene_context: {
    method: "PUT",
    path: item("/cameras", "/scene"),
    body: ([, context]) => ({ context }),
  },
  approve_scene_context: {
    method: "POST",
    path: item("/cameras", "/scene/approve"),
    body: ([, context]) => ({ context }),
  },
  request_scene_remap: {
    method: "POST",
    path: item("/cameras", "/scene/remap"),
  },
  enqueue_scene_mapping: {
    method: "POST",
    path: fixed("/scene-mapping/queue"),
    body: ([camera_ids]) => ({ camera_ids }),
  },
  scene_mapping_progress: {
    method: "GET",
    path: fixed("/scene-mapping/progress"),
  },
  accept_suggested_zone: {
    method: "POST",
    path: ([camera, name]) =>
      `/cameras/${encodeURIComponent(String(camera))}/zones/suggestions/${encodeURIComponent(String(name))}/accept`,
    body: ([, , dwell_seconds]) => ({ dwell_seconds }),
  },
  camera_snapshot: { method: "GET", path: item("/cameras", "/snapshot") },
  registered_objects: { method: "GET", path: item("/cameras", "/registered-objects") },
  recapture_registered_object: {
    method: "POST", path: ([camera, object]) => `/cameras/${encodeURIComponent(String(camera))}/registered-objects/${encodeURIComponent(String(object))}/recapture`,
  },
  register_object_region: {
    method: "POST", path: item("/cameras", "/registered-objects"),
    body: ([, name, region, frame_hw, confirm_seconds]) => ({ name, region, frame_hw, confirm_seconds }),
  },
  remove_registered_object: {
    method: "DELETE", path: ([camera, object]) => `/cameras/${encodeURIComponent(String(camera))}/registered-objects/${encodeURIComponent(String(object))}`,
  },
  camera_stream: {
    method: "GET",
    path: item("/cameras", "/stream"),
    query: ([, tracking = false]) => ({ tracking: tracking === true }),
    normalize: normalizeStreamDescriptor,
  },
  list_zones: { method: "GET", path: item("/cameras", "/zones") },
  add_zone: {
    method: "POST",
    path: item("/cameras", "/zones"),
    body: ([, name, points, dwell_seconds]) => ({
      name,
      points,
      dwell_seconds,
    }),
  },
  remove_zone: {
    method: "DELETE",
    path: ([camera, name]) =>
      `/cameras/${encodeURIComponent(String(camera))}/zones/${encodeURIComponent(String(name))}`,
  },
  // The vehicle tripwire (KPI 2). One per camera, so the path has no name.
  vehicle_line: { method: "GET", path: item("/cameras", "/vehicle-line") },
  set_vehicle_line: {
    method: "POST",
    path: item("/cameras", "/vehicle-line"),
    body: ([, start, end, flip = false, name = "gate"]) => ({
      start,
      end,
      flip,
      name,
    }),
  },
  remove_vehicle_line: {
    method: "DELETE",
    path: item("/cameras", "/vehicle-line"),
  },
  set_camera_rules: {
    method: "PUT",
    path: item("/cameras", "/rules"),
    body: ([, rules]) => ({ rules }),
  },
  object_targets: { method: "GET", path: fixed("/object-targets") },
  create_object_target: {
    method: "POST",
    path: fixed("/object-targets"),
    body: ([
      object_id,
      label,
      category,
      aliases,
      min_similarity,
      allowed_zone_ids,
      grounding_description,
    ]) =>
      typeof object_id === "object" && object_id !== null
        ? { target: object_id }
        : {
            object_id,
            label,
            category,
            aliases,
            min_similarity,
            allowed_zone_ids,
            grounding_description,
          },
  },
  set_object_watch_runtime_config: {
    method: "PUT",
    path: fixed("/object-targets/runtime"),
    body: ([config]) => ({ config }),
  },
  add_object_example: {
    method: "POST",
    path: item("/object-targets", "/examples"),
    body: ([, image_b64, bbox, source, bbox_format, negative]) => ({
      image_b64,
      bbox,
      source,
      bbox_format,
      negative,
    }),
  },
  review_object_example: {
    method: "PUT",
    path: ([objectId, exampleId]) =>
      `/object-targets/${encodeURIComponent(String(objectId))}/examples/${encodeURIComponent(String(exampleId))}/review`,
    body: ([, , reviewed]) => ({ reviewed }),
  },
  object_example_preview: {
    method: "GET",
    path: ([objectId, exampleId]) =>
      `/object-targets/${encodeURIComponent(String(objectId))}/examples/${encodeURIComponent(String(exampleId))}/preview`,
  },
  activate_object_target: {
    method: "POST",
    path: item("/object-targets", "/activate"),
  },
  deactivate_object_target: {
    method: "POST",
    path: item("/object-targets", "/deactivate"),
  },
  reembed_object_targets: {
    method: "POST",
    path: fixed("/object-targets/reembed"),
    body: ([model]) => ({ model }),
  },
  object_watch_job_status: {
    method: "GET",
    path: ([jobId]) =>
      `/object-targets/jobs/${encodeURIComponent(String(jobId))}`,
  },
  set_object_watch_rule: {
    method: "PUT",
    path: ([cameraId, objectId]) =>
      `/cameras/${encodeURIComponent(String(cameraId))}/object-watch/${encodeURIComponent(String(objectId))}`,
    body: ([, , enabled, zone_id]) => ({ enabled, zone_id }),
  },
  presets: { method: "GET", path: fixed("/cameras/presets") },
  use_case_templates: { method: "GET", path: fixed("/site/templates") },
  apply_template: {
    method: "POST",
    path: ([name]) =>
      `/site/templates/${encodeURIComponent(String(name))}/apply`,
  },
  add_custom_rule: {
    method: "POST",
    path: item("/cameras", "/rules/custom"),
    body: ([, question, dwell]) => ({ question, dwell }),
  },
  remove_custom_rule: {
    method: "DELETE",
    path: ([camera, name]) =>
      `/cameras/${encodeURIComponent(String(camera))}/rules/custom/${encodeURIComponent(String(name))}`,
  },
  english_rules_status: { method: "GET", path: fixed("/rules/english/status") },
  list_events: {
    method: "GET",
    path: fixed("/events"),
    query: ([limit]) => ({ limit: limit ?? 100 }),
    normalize: (value, client) =>
      (Array.isArray(value) ? value : (value?.events ?? [])).map((event: any) =>
        client.rememberIncident(event),
      ),
  },
  acknowledge_alert: { method: "POST", path: item("/events", "/acknowledge") },
  resolve_alert: {
    method: "POST",
    path: item("/events", "/resolve"),
    body: ([, outcome, note]) => ({ outcome, note }),
  },
  search_events: {
    method: "GET",
    path: fixed("/events"),
    query: ([q, limit]) => ({ q, limit: limit ?? 200 }),
  },
  camera_links: { method: "GET", path: item("/cameras", "/links") },
  start_monitoring: { method: "POST", path: fixed("/engine/start") },
  stop_monitoring: { method: "POST", path: fixed("/engine/stop") },
  monitoring_status: { method: "GET", path: fixed("/monitor") },
  feed_sources: { method: "GET", path: fixed("/engine/feeds") },
  switch_feed: {
    method: "POST",
    path: fixed("/engine/feeds/switch"),
    body: ([key]) => ({ key }),
  },
  feed_switch_status: { method: "GET", path: fixed("/engine/feeds/switch") },
  setup_state: { method: "GET", path: fixed("/setup/state") },
  setup_check: { method: "GET", path: fixed("/setup/check") },
  mark_configured: { method: "POST", path: fixed("/setup/configured") },
  send_test_notification: {
    method: "POST",
    path: fixed("/site/notifications/test"),
  },
  gate_status: {
    method: "GET",
    path: fixed("/engine/gate"),
    query: ([model]) => ({ model }),
  },
  pull_model: {
    method: "POST",
    path: fixed("/engine/models/pull"),
    body: ([model]) => ({ model }),
  },
  verifier_settings: { method: "GET", path: fixed("/engine/verifier") },
  set_verifier: {
    method: "PUT",
    path: fixed("/engine/verifier"),
    body: ([provider, model, base_url, api_key]) => ({ provider, model, base_url, api_key }),
  },
  clear_verifier_key: { method: "DELETE", path: fixed("/engine/verifier/key") },
  test_verifier: { method: "POST", path: fixed("/engine/verifier/test") },
  pull_progress: {
    method: "GET",
    path: fixed("/engine/models/pull"),
    query: ([model]) => ({ model }),
  },
  retention_status: { method: "GET", path: fixed("/retention") },
  set_retention: {
    method: "PUT",
    path: fixed("/retention"),
    body: ([days]) => ({ days }),
  },
  list_users: { method: "GET", path: fixed("/users") },
  add_user: {
    method: "POST",
    path: fixed("/users"),
    body: ([username, password, role]) => ({ username, password, role }),
  },
  remove_user: { method: "DELETE", path: item("/users") },
  audit_entries: {
    method: "GET",
    path: fixed("/audit"),
    query: ([limit]) => ({ limit }),
  },
  backup_now: { method: "POST", path: fixed("/backups") },
  download_diagnostics: { method: "POST", path: fixed("/diagnostics/bundle") },
  value_summary: {
    method: "GET",
    path: fixed("/value/summary"),
    query: ([days]) => ({ days }),
  },
  role_table: { method: "GET", path: fixed("/roles") },
  disk_encryption: { method: "GET", path: fixed("/system/disk-encryption") },
  organization: { method: "GET", path: fixed("/organization") },
  update_organization: {
    method: "PUT",
    path: fixed("/organization"),
    body: ([organization]) => ({ organization }),
  },
  list_branches: { method: "GET", path: fixed("/branches") },
  create_branch: {
    method: "POST",
    path: fixed("/branches"),
    body: ([branch]) => ({ branch }),
  },
  update_branch: {
    method: "PUT",
    path: item("/branches"),
    body: ([, branch]) => ({ branch }),
  },
  remove_branch: { method: "DELETE", path: item("/branches") },
  hierarchy: { method: "GET", path: fixed("/hierarchy") },
};

export const API_OPERATIONS = Object.freeze(Object.keys(operations));

export class ArgusApiClient {
  private readonly fetchImpl: typeof fetch;
  private readonly WebSocketImpl?: typeof WebSocket;
  private readonly listeners = new Set<(event: PushEvent) => void>();
  private readonly evidenceIds = new Map<string, string>();
  private token?: string;
  private socket?: WebSocket;
  private reconnectTimer?: ReturnType<typeof setTimeout>;
  private reconnectAttempt = 0;
  private closed = false;

  constructor(
    private readonly baseUrl: string,
    options: {
      fetch?: typeof fetch;
      WebSocket?: typeof WebSocket;
      onToken?: (token: string) => void;
      // "loopback" (default, Electron on the same box) hands the UI 127.0.0.1
      // stream URLs; "proxy" (remote browser) hands it same-origin MJPEG URLs
      // routed through the API, since 127.0.0.1 would mean the viewer's machine.
      streamMode?: "loopback" | "proxy";
      // A bearer token kept from an earlier page load, so a browser refresh
      // doesn't force a re-login while the 12h token is still valid.
      initialToken?: string;
    } = {},
  ) {
    this.fetchImpl = options.fetch ?? fetch;
    this.WebSocketImpl = options.WebSocket ?? globalThis.WebSocket;
    this.onToken = options.onToken;
    this.streamMode = options.streamMode ?? "loopback";
    this.token = options.initialToken || undefined;
  }

  private readonly onToken?: (token: string) => void;
  private readonly streamMode: "loopback" | "proxy";

  async invoke<T>(method: string, args: unknown[] = []): Promise<T> {
    if (!Array.isArray(args)) throw new Error("Invalid operation arguments");
    if (method === "auth_state") return (await this.authState()) as T;
    if (method === "sign_in") return (await this.signIn(args)) as T;
    if (method === "create_first_owner") {
      await this.request(
        {
          method: "POST",
          path: fixed("/auth/first-owner"),
          body: ([username, password]) => ({ username, password }),
        },
        args,
        false,
      );
      return (await this.signIn(args)) as T;
    }
    if (method === "sign_out") {
      const result = this.token
        ? await this.request(
            { method: "DELETE", path: fixed("/auth/session") },
            [],
            true,
          )
        : { ok: true };
      this.clearSession();
      return (result ?? { ok: true }) as T;
    }
    if (method === "event_clip") {
      // The card asks by event id. An evidence path is still honoured for
      // callers that remember one, but an empty key must never become
      // GET /events//clip -- that was a 404 dressed up as "no evidence".
      const key = String(args[0] ?? "");
      if (!key) throw new Error("event_clip needs an event id");
      const id = key.startsWith("evt_")
        ? key
        : (this.evidenceIds.get(key) ?? key);
      return (await this.request(
        { method: "GET", path: item("/events", "/clip") },
        [id],
        true,
      )) as T;
    }
    if (method === "live_start" || method === "live_stop")
      throw new Error(
        "Live streams are resolved per camera through camera_stream.",
      );
    if (method === "camera_stream" && this.streamMode === "proxy") {
      // Ask the API for a same-origin proxy URL (proxy=1) and accept the
      // relative MJPEG path it returns, rather than a loopback descriptor.
      return (await this.request(
        {
          method: "GET",
          path: item("/cameras", "/stream"),
          query: ([, tracking = false]) => ({
            tracking: tracking === true,
            proxy: true,
          }),
          normalize: normalizeProxyStreamDescriptor,
        },
        args,
        true,
      )) as T;
    }
    const operation = operations[method];
    if (!operation) throw new Error(`Unsupported engine operation: ${method}`);
    return (await this.request(operation, args, true)) as T;
  }

  subscribe(listener: (event: PushEvent) => void): () => void {
    this.listeners.add(listener);
    this.connectSocket();
    return () => {
      this.listeners.delete(listener);
      if (!this.listeners.size) this.stopSocket();
    };
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.token) {
      try {
        await this.request(
          { method: "DELETE", path: fixed("/auth/session") },
          [],
          true,
        );
      } catch {
        /* API may already be gone. */
      }
    }
    this.clearSession();
    this.listeners.clear();
  }

  rememberIncident(value: any) {
    const event = normalizeIncident(value);
    if (event.evidence_dir)
      this.evidenceIds.set(String(event.evidence_dir), String(event.id));
    return event;
  }

  private async authState() {
    const state = (await this.request(
      { method: "GET", path: fixed("/auth/state") },
      [],
      false,
    )) as any;
    if (!this.token)
      return {
        ...state,
        signed_in: false,
        username: "",
        role: "",
        permissions: [],
      };
    try {
      const user = (await this.request(
        { method: "GET", path: fixed("/auth/me") },
        [],
        true,
      )) as any;
      return { ...state, signed_in: true, ...user };
    } catch (error) {
      if ((error as ApiError).status !== 401) throw error;
      this.clearSession();
      return {
        ...state,
        signed_in: false,
        username: "",
        role: "",
        permissions: [],
      };
    }
  }

  private async signIn(args: unknown[]) {
    const value = (await this.request(
      {
        method: "POST",
        path: fixed("/auth/session"),
        body: ([username, password]) => ({ username, password }),
      },
      args,
      false,
    )) as any;
    this.token = value.token;
    this.onToken?.(value.token);
    this.closed = false;
    this.connectSocket();
    return {
      configured: true,
      signed_in: true,
      username: value.user.username,
      role: value.user.role,
      permissions: value.user.permissions,
    };
  }

  private async request(
    operation: Operation,
    args: unknown[],
    authenticated: boolean,
  ) {
    let path = operation.path(args);
    const cameraSegment = /^\/cameras\/([^/]+)(.*)$/.exec(path);
    let cameraId: string | undefined;
    if (cameraSegment && /%2f/i.test(cameraSegment[1])) {
      // Encoded slashes are decoded by ASGI before matching path parameters.
      cameraId = decodeURIComponent(cameraSegment[1]);
      path = `/camera-by-id${cameraSegment[2]}`;
    }
    const url = new URL(this.baseUrl + path);
    if (cameraId !== undefined) url.searchParams.set("camera_id", cameraId);
    for (const [key, value] of Object.entries(operation.query?.(args) ?? {}))
      if (value !== undefined && value !== null && value !== "")
        url.searchParams.set(key, String(value));
    const headers = new Headers();
    if (authenticated) {
      if (!this.token)
        throw Object.assign(new Error("Sign in to access the local engine"), {
          status: 401,
          code: "unauthorized",
        });
      headers.set("authorization", `Bearer ${this.token}`);
    }
    const body = operation.body?.(args);
    if (body !== undefined) headers.set("content-type", "application/json");
    const response = await this.fetchImpl(url, {
      method: operation.method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await response.text();
    let value: any = null;
    if (text) {
      try {
        value = JSON.parse(text);
      } catch {
        value = text;
      }
    }
    if (!response.ok) {
      const envelope = value?.error ?? {};
      const permission = envelope?.detail?.permission;
      const message =
        envelope.message ??
        value?.detail ??
        response.statusText ??
        "API request failed";
      throw Object.assign(
        new Error(
          permission ? `${message} (requires ${permission})` : String(message),
        ),
        { status: response.status, code: envelope.code, permission },
      ) as ApiError;
    }
    return operation.normalize ? operation.normalize(value, this) : value;
  }

  private connectSocket() {
    if (
      this.closed ||
      !this.token ||
      !this.listeners.size ||
      this.socket ||
      !this.WebSocketImpl
    )
      return;
    const url = new URL(this.baseUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = `${url.pathname.replace(/\/$/, "")}/stream`;
    url.search = "";
    const socket = new this.WebSocketImpl(url.toString(), [
      "argus.v1",
      `argus.token.${this.token}`,
    ]);
    this.socket = socket;
    socket.onopen = () => {
      this.reconnectAttempt = 0;
    };
    socket.onmessage = (message) => {
      try {
        const parsed = JSON.parse(String(message.data));
        if (!isPushEvent(parsed)) return;
        const event = parsed.type.startsWith("alert.")
          ? ({
              type: parsed.type,
              data: this.rememberIncident(parsed.data),
            } as PushEvent)
          : ({ type: parsed.type, data: parsed.data } as PushEvent);
        for (const listener of this.listeners) listener(event);
      } catch {
        /* Recovery polling covers malformed push data. */
      }
    };
    socket.onerror = () => socket.close();
    socket.onclose = (event) => {
      if (this.socket === socket) this.socket = undefined;
      if ([4401, 4403].includes(event.code)) {
        if (event.code === 4401) {
          this.token = undefined;
          this.evidenceIds.clear();
        }
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        this.reconnectTimer = undefined;
        return;
      }
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect() {
    if (
      this.closed ||
      !this.token ||
      !this.listeners.size ||
      this.reconnectTimer
    )
      return;
    const delay = Math.min(1000 * 2 ** this.reconnectAttempt++, 30000);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined;
      this.connectSocket();
    }, delay);
  }

  private stopSocket() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = undefined;
    const socket = this.socket;
    this.socket = undefined;
    socket?.close();
  }

  private clearSession() {
    this.token = undefined;
    this.evidenceIds.clear();
    this.stopSocket();
  }
}

function isPushEvent(value: any): value is PushEvent {
  return Boolean(
    value &&
    typeof value === "object" &&
    ["health", "triage", "alert.new", "alert.update"].includes(value.type) &&
    "data" in value,
  );
}
