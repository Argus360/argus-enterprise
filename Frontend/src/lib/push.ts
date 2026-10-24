import type { Incident, Json, PushEvent, Workspace } from "./types";

export function applyPushEvent(
  workspace: Workspace,
  event: PushEvent,
): Workspace {
  if (event.type === "alert.new" || event.type === "alert.update") {
    const incident = event.data as Incident;
    const existing = workspace.events.findIndex(
      (item) => String(item.id) === String(incident.id),
    );
    if (existing < 0)
      return { ...workspace, events: [incident, ...workspace.events] };
    const events = [...workspace.events];
    events[existing] = incident;
    return { ...workspace, events };
  }
  if (event.type !== "health") return workspace;
  const health = event.data as Json;
  const cameraHealth = new Map(
    ((health.cameras as Json[]) ?? []).map((camera) => [
      String(camera.camera_id),
      camera,
    ]),
  );
  const engine = (health.engine as Json) ?? {};
  const phase = String(engine.phase ?? workspace.monitor.phase ?? "stopped");
  return {
    ...workspace,
    monitor: {
      ...workspace.monitor,
      ...engine,
      link_events: health.link_events ?? workspace.monitor.link_events ?? [],
      status: health.status,
      generated_at: health.generated_at,
      running: typeof engine.running === "boolean"
        ? engine.running : phase !== "stopped" && phase !== "",
      starting: typeof engine.starting === "boolean"
        ? engine.starting : phase.startsWith("starting"),
    },
    cameras: workspace.cameras.map((camera) => ({
      ...camera,
      ...(cameraHealth.get(String(camera.id)) ?? {}),
    })),
  };
}
