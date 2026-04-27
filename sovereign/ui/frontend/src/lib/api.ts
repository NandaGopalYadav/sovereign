// Thin client over the FastAPI backend. All paths go through the Vite proxy in dev
// and through the bundled FastAPI static-mount in production.

export type Territory = {
  id: number;
  name: string;
  home_of: number;
  resource_value: number;
  strategic_value: number;
};

export type MapInfo = {
  name: string;
  n: number;
  territories: Territory[];
  edges: [number, number][];
};

export type LiveState = {
  turn: number;
  controller: number[];
  invader_units: number[];
  defender_units: number[];
  neutral_units: number[];
  legitimacy: number;
  economy: number;
  theta: number;
  t_occ: number;
  sanctions_active: boolean;
  supply_routes_open: boolean;
  formal_alliance: boolean;
  thresholds: {
    sanctions_on: number;
    sanctions_off: number;
    neutral_joins_defender: number;
    supply_routes_open: number;
    formal_alliance: number;
  };
  reward?: number;
  reward_components?: Record<string, number>;
  terminated?: boolean;
  truncated?: boolean;
  terminal_reason?: string | null;
  last_action?: { pol: number; mil: number; target: number };
  map_name?: string;
};

export type EpisodeSummary = {
  episode: number;
  length: number;
  return: number;
  mean_theta: number;
  min_legitimacy: number;
  terminal_reason: string | null;
};

export type TraceFrame = {
  t: number;
  reward: number;
  theta: number;
  legitimacy: number;
  action: number;
  reward_components: Record<string, number>;
};

const j = async <T>(url: string, init?: RequestInit): Promise<T> => {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json() as Promise<T>;
};

export const api = {
  maps: () => j<MapInfo[]>("/api/maps"),
  actions: () => j<{ political: string[]; military: string[] }>("/api/actions"),
  runs: () => j<{ id: string; path: string; size: number }[]>("/api/runs"),
  runEpisodes: (id: string) =>
    j<EpisodeSummary[]>(`/api/runs/${encodeURIComponent(id)}/episodes`),
  runTrace: (id: string, ep: number) =>
    j<TraceFrame[]>(`/api/runs/${encodeURIComponent(id)}/trace/${ep}`),
  liveReset: (req: { map_name: string; regime: string; seed?: number | null }) =>
    j<LiveState>("/api/live/reset", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    }),
  liveStep: (req: { pol: number; mil: number; target: number }) =>
    j<LiveState>("/api/live/step", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    }),
  liveState: () => j<LiveState>("/api/live/state"),
  liveHistory: () => j<LiveState[]>("/api/live/history"),
  compare: () =>
    j<
      {
        regime: string;
        return_mean: number;
        return_std: number;
        settlement_rate: number;
        conquest_rate: number;
        legitimacy_collapse_rate: number;
        destroyed_rate: number;
        timeout_rate: number;
        mean_theta: number;
        mean_legitimacy: number;
      }[]
    >("/api/compare"),
};
