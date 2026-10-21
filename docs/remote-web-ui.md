# Serving the Argus UI from the engine box (remote browser)

Normally the Argus UI is the Electron desktop app talking to an engine on the
**same machine** over `127.0.0.1`. When the engine runs on a server (e.g. the
Azure GPU box) and people need to open it in a **browser over a tunnel**, serve
the built web UI and the API from that one box, same-origin.

Same-origin is the whole trick: the browser's bearer token and the live
WebSocket reach the API with no CORS, and live video is proxied through the API
so the frame publisher stays bound to loopback.

## What changed (all additive — the desktop app is untouched)

- **API serves the built UI.** `python -m cvti.api --webui-dir <dir>` (or
  `ARGUS_WEBUI_DIR=<dir>`) mounts the SPA at `/`. Without it, the API serves
  JSON at `/` exactly as before.
- **MJPEG proxy.** `GET /api/v1/cameras/{id}/mjpeg?token=…` streams the loopback
  frame publisher back to the browser. The publisher never leaves `127.0.0.1`.
- **Proxy stream descriptor.** `GET /api/v1/cameras/{id}/stream?proxy=1` returns
  a same-origin MJPEG URL instead of a `127.0.0.1` one. WebRTC is skipped in
  proxy mode (no off-box TURN); MJPEG is the transport.
- **Browser transport.** When there is no `window.argusDesktop`, the renderer
  talks to the API over HTTP/WS at `window.location.origin` instead of throwing
  "open the desktop app". It asks for proxied streams and keeps the session
  token in `sessionStorage` so a refresh doesn't force a re-login.

## Build and run

```bash
# 1. Build the web bundle (absolute /asset paths, its own output dir)
cd Frontend && npm ci && npm run build:web      # -> Frontend/dist-web

# 2. Run the API pointed at the engine's output dir, serving the UI
python -m cvti.api \
  --host 127.0.0.1 --port 8787 \
  --db   <engine-output>/events.db \
  --site <site-config>.json \
  --webui-dir Frontend/dist-web
```

The API and the engine share only the output directory (events.db, frames.json,
auth.db beside it) — run the API as its own service alongside the engine.

## Expose it (behind the login wall)

Front the API port with the existing tunnel (e.g. a Tailscale funnel/serve
route to `127.0.0.1:8787`). The app's own account login gates every request and
the WebSocket; the tunnel just makes the origin reachable. Do **not** bind the
frame publisher or go2rtc to a public interface — only the API is exposed, and
it proxies video itself.

## Known limits (follow-ups)

- **Live view needs the engine running.** The config-time preview (engine
  stopped) uses a separate loopback server that isn't proxied yet; in proxy mode
  the tile waits (retryable 503) until the engine is up.
- **WebRTC is loopback-only.** Remote video is MJPEG. WebRTC off-box would need
  TURN/ICE — a later step if higher frame rates are needed.
