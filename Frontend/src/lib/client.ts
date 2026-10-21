import { createDemo } from "./demo";
import { createHttpTransport } from "./http-transport";
import type { Mode, Transport } from "./types";
const demo = createDemo(localStorage);

let httpTransport: Transport | undefined;
function browserTransport(): Transport {
  // One client per page load, so the session token and WebSocket are shared
  // across every hook that calls client().
  return (httpTransport ??= createHttpTransport());
}

export function client(mode: Mode): Transport {
  if (mode === "demo") return demo;
  // Electron desktop: talk to the local engine over the IPC bridge. Served in a
  // plain browser (API + UI on one origin, behind the tunnel): talk to the API
  // over HTTP/WS instead of throwing "open the desktop app".
  if (!window.argusDesktop) return browserTransport();
  return {
    invoke: async (method, args = []) => {
      return window.argusDesktop!.invoke(method, args);
    },
    subscribe: (listener) => {
      return window.argusDesktop!.subscribe(listener);
    },
  };
}
