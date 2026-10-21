/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Set to "1" only by the `build:web` script, which produces the bundle the
  // engine serves over the tunnel. It tells the app it is served BY a real
  // backend (not the Electron shell, not the demo preview), so it defaults to
  // engine mode and shows the real sign-in instead of the demo-only notice.
  readonly VITE_ARGUS_SERVED?: string;
}
