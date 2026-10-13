import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { API_OPERATIONS } from "../electron/api-client";

// The renderer can only call what the Electron main process allows, and the
// source-run bridge keeps a list of its own. Both drifted from the API client
// (cloud verifier, vehicle line): the packaged app answered "Invalid engine
// request" to its own settings panel on the pilot server (13 Oct 2026).

function mainAllowlist(): Set<string> {
  const src = readFileSync(new URL("../electron/main.ts", import.meta.url), "utf8");
  const names = new Set<string>();
  const block = src.match(/const methods = new Set\(\[([\s\S]*?)\]\);/)?.[1] ?? "";
  for (const m of block.matchAll(/"([a-z_]+)"/g)) names.add(m[1]);
  for (const m of src.matchAll(/methods\.add\("([a-z_]+)"\)/g)) names.add(m[1]);
  for (const loop of src.matchAll(/for \(const method of \[([\s\S]*?)\]\)\s*methods\.add\(method\)/g))
    for (const m of loop[1].matchAll(/"([a-z_]+)"/g)) names.add(m[1]);
  if (/for \(const method of API_OPERATIONS\) methods\.add\(method\)/.test(src))
    for (const name of API_OPERATIONS) names.add(name);
  return names;
}

function bridgeAllowlist(): Set<string> {
  const src = readFileSync(new URL("../bridge.py", import.meta.url), "utf8");
  const names = new Set<string>();
  const block = src.match(/METHODS = set\('([^']+)'/)?.[1] ?? "";
  for (const name of block.split(/\s+/)) if (name) names.add(name);
  for (const m of src.matchAll(/METHODS\.add\('([a-z_]+)'\)/g)) names.add(m[1]);
  for (const loop of src.matchAll(/for _m in \(([^)]*)\):\s*METHODS\.add\(_m\)/g))
    for (const m of loop[1].matchAll(/'([a-z_]+)'/g)) names.add(m[1]);
  return names;
}

describe("engine method allowlists", () => {
  it("the Electron main process allows every API client route", () => {
    const allowed = mainAllowlist();
    expect(API_OPERATIONS.filter((name) => !allowed.has(name))).toEqual([]);
  });

  it("the source-run bridge allows the routes the console backend serves", () => {
    // The bridge speaks straight to ConsoleBackend; routes that only the API
    // layer implements (organization, hierarchy, camera_stream) are not its
    // business. The ones that drifted are.
    const allowed = bridgeAllowlist();
    const expected = [
      "verifier_settings", "set_verifier", "clear_verifier_key", "test_verifier",
      "vehicle_line", "set_vehicle_line", "remove_vehicle_line",
    ];
    expect(expected.filter((name) => !allowed.has(name))).toEqual([]);
  });
});
