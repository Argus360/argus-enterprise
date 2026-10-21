import { ArgusApiClient } from "../../electron/api-client";
import type { PushEvent, Transport } from "./types";

// Browser transport: the same engine API the Electron app speaks, but reached
// over HTTP/WS from a web browser instead of through the Electron IPC bridge.
// Used when the app is served from the engine box itself (API + built UI on one
// origin, behind the tunnel) and there is no window.argusDesktop. ArgusApiClient
// is already pure browser code (fetch/WebSocket/URL) — we only point it at this
// page's own origin and ask for same-origin proxied streams.

const TOKEN_KEY = "argus.session.token";

function readToken(): string | undefined {
  try {
    return sessionStorage.getItem(TOKEN_KEY) || undefined;
  } catch {
    // Private mode or blocked storage: fall back to a login per page load.
    return undefined;
  }
}

function writeToken(token: string): void {
  try {
    sessionStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* Non-fatal: the token still lives in memory for this page load. */
  }
}

function clearToken(): void {
  try {
    sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* Nothing persisted; nothing to clear. */
  }
}

export function createHttpTransport(): Transport {
  // Same-origin: "/api/v1" on whatever host served this page (the tunnel). Same
  // origin means the bearer token and WebSocket reach the API with no CORS.
  const baseUrl = `${window.location.origin}/api/v1`;
  const api = new ArgusApiClient(baseUrl, {
    streamMode: "proxy",
    initialToken: readToken(),
    onToken: writeToken,
  });
  return {
    invoke: async <T>(method: string, args: unknown[] = []): Promise<T> => {
      const result = await api.invoke<T>(method, args);
      // The client clears its own in-memory token on sign-out; drop the
      // persisted copy too so a refresh doesn't resurrect a dead session.
      if (method === "sign_out") clearToken();
      return result;
    },
    subscribe: (listener: (event: PushEvent) => void) =>
      api.subscribe(listener),
  };
}
