import { useEffect, useRef, useState } from "react";
import { Maximize2, TriangleAlert } from "lucide-react";
import { activeConcealmentNotice, type ConcealmentNotice } from "../lib/concealment-notice";
import {
  cameraStreamPresentation,
  type MediaEvidence,
  type StreamTransport,
} from "../lib/stream-state";
import type { Camera, StreamDescriptor, Transport } from "../lib/types";
import { resolveCameraStream, type ResolvedCameraStream } from "../lib/whep";
import { Empty, Spinner } from "./common";

const RETRY_OFFLINE_MS = 5_000;
const FIRST_FRAME_TIMEOUT_MS = 35_000;

type PlayerState =
  | { kind: "loading" }
  | ResolvedCameraStream
  | { kind: "offline"; message: string };

export default function CameraStream({
  camera,
  api,
  active,
  tracking = false,
  onOpen,
  onLiveChange,
}: {
  camera: Camera;
  api: Transport;
  active: boolean;
  tracking?: boolean;
  onOpen?: () => void;
  onLiveChange?: (live: boolean) => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<PlayerState>({
    kind: active ? "loading" : "inactive",
  });
  const [imageFailed, setImageFailed] = useState(false);
  const [evidence, setEvidence] = useState<MediaEvidence>("none");
  const [retry, setRetry] = useState(0);
  const [notice, setNotice] = useState<ConcealmentNotice | null>(null);

  useEffect(() => {
    setNotice(null);
    if (!active || !api.subscribe) return;
    return api.subscribe((event) => {
      if (event.type !== "health") return;
      const health = event.data as { cameras?: { camera_id: string; concealment_notice?: unknown }[] };
      const row = health.cameras?.find((item) => item.camera_id === camera.id);
      setNotice(activeConcealmentNotice(row?.concealment_notice));
    });
  }, [active, api, camera.id]);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(null), Math.max(0, notice.expires_at * 1000 - Date.now()));
    return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    const controller = new AbortController();
    setImageFailed(false);
    setEvidence("none");
    if (!active) {
      setState({ kind: "inactive" });
      return () => controller.abort();
    }
    setState({ kind: "loading" });
    if (video.current)
      void resolveCameraStream({
        cameraId: camera.id,
        tracking,
        active,
        api,
        video: video.current,
        signal: controller.signal,
        onWebRtcTrack: () => setEvidence("webrtc-track"),
      })
        .then((next) => !controller.signal.aborted && setState(next))
        .catch((error) => {
          if (!controller.signal.aborted)
            setState({
              kind: "offline",
              message: (error as Error).message,
            });
        });
    return () => controller.abort();
  }, [active, api, camera.id, retry, tracking]);

  const transport: StreamTransport =
    state.kind === "webrtc"
      ? "webrtc"
      : state.kind === "mjpeg"
        ? state.degraded
          ? "mjpeg-fallback"
          : "mjpeg"
        : "none";
  const presentation = cameraStreamPresentation({
    active,
    cameraState: state.kind === "mjpeg" && state.preview ? undefined : camera.state,
    transport,
    evidence,
    failed: state.kind === "offline" || imageFailed,
  });

  useEffect(() => {
    onLiveChange?.(presentation.phase === "live");
    return () => onLiveChange?.(false);
  }, [onLiveChange, presentation.phase]);

  // A multipart connection can stay open without delivering any image or
  // firing onError. Give it a deadline so the existing retry path can recover.
  useEffect(() => {
    if (!active || evidence !== "none" || imageFailed ||
        state.kind === "inactive" || state.kind === "offline") return;
    const timer = setTimeout(() => setImageFailed(true), FIRST_FRAME_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [active, evidence, imageFailed, state, retry]);

  // Refresh the descriptor when capture ownership changes. A stopped MJPEG
  // publisher can leave its last image visible without firing an image error.
  useEffect(() => {
    if (!active || (state.kind !== "mjpeg" && state.kind !== "webrtc")) return;
    let cancelled = false;
    const timer = setInterval(() => {
      void api.invoke<StreamDescriptor>("camera_stream", [camera.id, tracking])
        .then((next) => {
          const sameFallback = state.kind === "mjpeg" && state.degraded &&
            next.kind === "webrtc" && next.mjpeg_fallback === state.url;
          if (!cancelled && !sameFallback && (next.kind !== state.kind || next.url !== state.url))
            setRetry((n) => n + 1);
        }).catch(() => {
          if (!cancelled) setImageFailed(true);
        });
    }, RETRY_OFFLINE_MS);
    return () => { cancelled = true; clearInterval(timer); };
  }, [active, api, camera.id, tracking, state]);

  // A tile resolved its stream once and then sat on the result. Open the wall
  // before pressing Start monitoring and every tile stayed "offline" until
  // something remounted it — and each engine run publishes on a new port, so
  // the stale URL could never come back on its own (12 Sep). While offline,
  // ask again every few seconds; one small request per tile.
  useEffect(() => {
    if (!active || presentation.phase !== "offline") return;
    const timer = setTimeout(() => setRetry((n) => n + 1), RETRY_OFFLINE_MS);
    return () => clearTimeout(timer);
  }, [active, presentation.phase, retry]);

  return (
    <div className="camera-media">
      <video
        ref={video}
        className={
          state.kind === "webrtc" && presentation.phase !== "offline"
            ? ""
            : "stream-hidden"
        }
        autoPlay
        muted
        playsInline
      />
      {state.kind === "mjpeg" && presentation.phase !== "offline" ? (
        <img
          src={state.url}
          alt={`${camera.id} live feed`}
          onLoad={() => setEvidence("mjpeg-frame")}
          onError={() => {
            setEvidence("none");
            setImageFailed(true);
          }}
        />
      ) : presentation.phase === "loading" ? (
        <div className="loading">
          <Spinner />
          Connecting feed...
        </div>
      ) : presentation.phase === "offline" ? (
        <Empty title="Preview unavailable">
          {state.kind === "offline"
            ? state.message
            : imageFailed
              ? "No camera image received. Retrying the connection."
              : "Camera health reports that this feed is offline."}
        </Empty>
      ) : presentation.phase === "inactive" ? (
        <Empty title="Preview inactive">
          This camera preview is not visible.
        </Empty>
      ) : presentation.phase === "degraded" && evidence === "none" ? (
        <Empty title="Camera degraded">
          Camera health reports that this feed is reconnecting.
        </Empty>
      ) : null}
      <div className="media-label">
        <span
          className={`status-dot ${
            presentation.phase === "offline"
              ? "off"
              : presentation.phase === "degraded"
                ? "sample"
                : ""
          }`}
        />
        {state.kind === "mjpeg" && state.preview && presentation.phase === "live"
          ? "LIVE PREVIEW" : presentation.label}
      </div>
      {notice && active && presentation.phase !== "offline" && (
        <div className={`concealment-notice ${notice.phase}`} role="status">
          <TriangleAlert size={18} aria-hidden="true" />
          <div><strong>Possible product concealment</strong>
            <span>{notice.phase === "verifying" ? "Unverified gesture · Verifying"
              : notice.phase === "inconclusive" ? "Earlier activity · AI inconclusive · Needs review"
              : "Earlier activity · Review required"}</span>
          </div>
        </div>
      )}
      {onOpen && (
        <button
          className="media-open"
          title={`Open ${camera.id}`}
          aria-label={`Open ${camera.id}`}
          onClick={onOpen}
        >
          <Maximize2 size={17} />
        </button>
      )}
    </div>
  );
}
