import { createRoot } from "react-dom/client";
import "../../src/styles.css";
import CameraStream from "../../src/components/CameraStream";
import type { PushEvent, StreamDescriptor, Transport } from "../../src/lib/types";
const listeners = new Set<(event: PushEvent) => void>();

let descriptor: StreamDescriptor;
let failOffer = false;
let latestPeer: FakePeer | undefined;
let peersMade = 0;

class FakePeer extends EventTarget {
  ontrack: RTCPeerConnection["ontrack"] = null;
  localDescription: RTCSessionDescriptionInit | null = null;
  connectionState: RTCPeerConnectionState = "new";
  constructor() {
    super();
    latestPeer = this;
    peersMade += 1;
  }
  addTransceiver() {}
  async createOffer() {
    if (failOffer) throw new Error("WHEP unavailable");
    return { type: "offer" as const, sdp: "offer" };
  }
  async setLocalDescription(value: RTCSessionDescriptionInit) {
    this.localDescription = value;
  }
  async setRemoteDescription() {}
  close() {}
}

window.RTCPeerConnection = FakePeer as any;
const transport: Transport = {
  subscribe(listener) {
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  },
  async invoke<T>() {
    return descriptor as T;
  },
};
const root = createRoot(document.getElementById("root")!);

(window as any).cameraHarness = {
  push(event: PushEvent) { listeners.forEach((listener) => listener(event)); },
  render(options: {
    state: string;
    descriptor: StreamDescriptor;
    active?: boolean;
    failWhep?: boolean;
  }) {
    descriptor = options.descriptor;
    failOffer = options.failWhep ?? false;
    latestPeer = undefined;
    root.render(
      <CameraStream
        camera={{
          id: "Front Door",
          source: "rtsp://front",
          state: options.state,
        }}
        api={transport}
        active={options.active ?? true}
      />,
    );
  },
  peersMade() {
    return peersMade;
  },
  setConnection(state: RTCPeerConnectionState) {
    if (!latestPeer) return false;
    latestPeer.connectionState = state;
    latestPeer.dispatchEvent(new Event("connectionstatechange"));
    return true;
  },
  emitTrack() {
    if (!latestPeer?.ontrack) return false;
    const stream = new MediaStream();
    latestPeer.ontrack({
      streams: [stream],
      track: {} as MediaStreamTrack,
    } as RTCTrackEvent);
    return true;
  },
};
