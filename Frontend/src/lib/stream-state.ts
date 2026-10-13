export type StreamTransport = "none" | "webrtc" | "mjpeg" | "mjpeg-fallback";
export type MediaEvidence = "none" | "webrtc-track" | "mjpeg-frame";

export type StreamPresentation = {
  phase: "inactive" | "loading" | "live" | "degraded" | "offline";
  label: string;
};

export function cameraStreamPresentation({
  active,
  cameraState,
  transport,
  evidence,
  failed,
}: {
  active: boolean;
  cameraState?: string;
  transport: StreamTransport;
  evidence: MediaEvidence;
  failed: boolean;
}): StreamPresentation {
  if (!active) return { phase: "inactive", label: "INACTIVE" };

  const health = cameraState?.toLowerCase();
  if (failed) return { phase: "offline", label: "OFFLINE" };
  const hasMedia =
    (transport === "webrtc" && evidence === "webrtc-track") ||
    ((transport === "mjpeg" || transport === "mjpeg-fallback") &&
      evidence === "mjpeg-frame");
  if (health === "offline" || health === "failed" || health === "error")
    // The engine's decoder state is a badge, not a kill switch. A tile that
    // is still showing frames stays mounted with its last image and an
    // OFFLINE label; unmounting it on every decoder hiccup dropped its
    // stream and remounted it five seconds later, which was the wall
    // flapping on the pilot server (W1 wall, 13 Oct 2026). With no media
    // yet the tile is offline and keeps asking.
    return hasMedia
      ? { phase: "degraded", label: "OFFLINE" }
      : { phase: "offline", label: "OFFLINE" };
  if (health === "reconnecting")
    return { phase: "degraded", label: "RECONNECTING" };
  if (health === "degraded" || health === "stalled")
    return { phase: "degraded", label: "DEGRADED" };
  if (!hasMedia) return { phase: "loading", label: "CONNECTING" };
  if (transport === "webrtc") return { phase: "live", label: "WEBRTC FEED" };
  if (transport === "mjpeg") return { phase: "live", label: "MJPEG FEED" };
  return { phase: "degraded", label: "MJPEG FALLBACK" };
}
