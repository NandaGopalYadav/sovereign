import { useEffect, useState } from "react";
import { api, LiveState, MapInfo } from "../lib/api";
import Gauge from "./Gauge";
import MapGraph from "./MapGraph";
import RewardStack from "./RewardStack";

const REGIMES = [
  "full",
  "no_legitimacy",
  "no_occupation_cost",
  "no_neutral",
  "baseline",
] as const;

export default function LivePanel() {
  const [maps, setMaps] = useState<MapInfo[]>([]);
  const [actions, setActions] = useState<{ political: string[]; military: string[] }>({
    political: [],
    military: [],
  });
  const [mapName, setMapName] = useState<string>("rulebook9");
  const [regime, setRegime] = useState<string>("full");
  const [pol, setPol] = useState<number>(0);
  const [mil, setMil] = useState<number>(0);
  const [target, setTarget] = useState<number>(0);
  const [state, setState] = useState<LiveState | null>(null);
  const [history, setHistory] = useState<LiveState[]>([]);

  useEffect(() => {
    api.maps().then(setMaps);
    api.actions().then(setActions);
  }, []);

  const reset = async () => {
    const s = await api.liveReset({ map_name: mapName, regime, seed: 0 });
    setState(s);
    setHistory([]);
  };

  const step = async () => {
    if (!state || state.terminated || state.truncated) return;
    const s = await api.liveStep({ pol, mil, target });
    setState(s);
    setHistory((h) => [...h, s]);
  };

  const currentMap = maps.find((m) => m.name === (state?.map_name ?? mapName));

  return (
    <div className="grid grid-cols-12 gap-3 p-3 h-full overflow-hidden">
      <section className="col-span-3 flex flex-col gap-3 overflow-y-auto pr-1">
        <div className="panel p-3">
          <div className="axis-label mb-2">session</div>
          <div className="grid grid-cols-2 gap-2">
            <label className="text-xxs text-ink-mute uppercase tracking-wider">
              map
              <select
                className="mt-1 w-full bg-bg-elev border border-line text-ink text-xs px-2 py-1"
                value={mapName}
                onChange={(e) => setMapName(e.target.value)}
              >
                {maps.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-xxs text-ink-mute uppercase tracking-wider">
              regime
              <select
                className="mt-1 w-full bg-bg-elev border border-line text-ink text-xs px-2 py-1"
                value={regime}
                onChange={(e) => setRegime(e.target.value)}
              >
                {REGIMES.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <button
            onClick={reset}
            className="mt-3 w-full text-xs uppercase tracking-wider border border-accent text-accent
                       px-3 py-1.5 hover:bg-accent/10"
          >
            reset
          </button>
        </div>

        <div className="panel p-3">
          <div className="axis-label mb-2">action</div>
          <label className="text-xxs text-ink-mute uppercase tracking-wider block">
            political
            <select
              className="mt-1 w-full bg-bg-elev border border-line text-ink text-xs px-2 py-1"
              value={pol}
              onChange={(e) => setPol(parseInt(e.target.value))}
            >
              {actions.political.map((p, i) => (
                <option key={p} value={i}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xxs text-ink-mute uppercase tracking-wider block mt-2">
            military
            <select
              className="mt-1 w-full bg-bg-elev border border-line text-ink text-xs px-2 py-1"
              value={mil}
              onChange={(e) => setMil(parseInt(e.target.value))}
            >
              {actions.military.map((m, i) => (
                <option key={m} value={i}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xxs text-ink-mute uppercase tracking-wider block mt-2">
            target
            <select
              className="mt-1 w-full bg-bg-elev border border-line text-ink text-xs px-2 py-1"
              value={target}
              onChange={(e) => setTarget(parseInt(e.target.value))}
            >
              {currentMap?.territories.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.id} · {t.name}
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={step}
            disabled={!state || state.terminated || state.truncated}
            className="mt-3 w-full text-xs uppercase tracking-wider border border-line
                       px-3 py-1.5 disabled:opacity-40 hover:border-accent hover:text-accent"
          >
            step
          </button>
          {state && (state.terminated || state.truncated) && (
            <div className="mt-2 text-xxs text-warn uppercase tracking-wider">
              episode ended · {state.terminal_reason ?? "timeout"}
            </div>
          )}
        </div>

        <div className="panel p-3">
          <div className="axis-label mb-2">state</div>
          <div className="grid grid-cols-2 gap-y-2 text-xs">
            <span className="text-ink-mute">turn</span>
            <span className="num text-right">{state?.turn ?? 0}</span>
            <span className="text-ink-mute">t_occ</span>
            <span className="num text-right">{state?.t_occ ?? 0}</span>
            <span className="text-ink-mute">sanctions</span>
            <span className={"text-right num " + (state?.sanctions_active ? "text-warn" : "text-ink-mute")}>
              {state?.sanctions_active ? "active" : "—"}
            </span>
            <span className="text-ink-mute">supply route</span>
            <span className={"text-right num " + (state?.supply_routes_open ? "text-ok" : "text-ink-mute")}>
              {state?.supply_routes_open ? "open" : "—"}
            </span>
            <span className="text-ink-mute">alliance</span>
            <span className={"text-right num " + (state?.formal_alliance ? "text-defender" : "text-ink-mute")}>
              {state?.formal_alliance ? "formed" : "—"}
            </span>
          </div>
        </div>
      </section>

      <section className="col-span-6 grid grid-rows-[1fr_auto_220px] gap-3 overflow-hidden">
        <div className="panel overflow-hidden relative">
          {currentMap ? <MapGraph map={currentMap} state={state} /> : null}
          <div className="absolute top-2 right-2 flex gap-2">
            <span className="pill"><span className="w-1.5 h-1.5 bg-invader mr-1" />invader</span>
            <span className="pill"><span className="w-1.5 h-1.5 bg-defender mr-1" />defender</span>
            <span className="pill"><span className="w-1.5 h-1.5 bg-neutral mr-1" />neutral</span>
            <span className="pill"><span className="w-1.5 h-1.5 bg-contested mr-1" />contested</span>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <Gauge
            label="θ — neutral posture"
            value={state?.theta ?? 0}
            min={-1}
            max={+1}
            bands={[
              { from: 0.6, to: 1, label: "sanctions", color: "warn" },
              { from: 0.85, to: 1, label: "joins defender", color: "danger" },
              { from: -0.6, to: -0.85, label: "supply routes", color: "ok" },
              { from: -0.85, to: -1, label: "alliance", color: "ok" },
            ]}
          />
          <Gauge label="L — legitimacy" value={state?.legitimacy ?? 1} min={0} max={1} />
          <Gauge label="E — economy" value={state?.economy ?? 1} min={0} max={1} />
        </div>
        <div className="panel p-3 overflow-hidden">
          <div className="axis-label mb-1">reward decomposition</div>
          <div className="h-[170px]">
            <RewardStack history={history.map((h) => ({ t: h.turn, reward_components: h.reward_components ?? {} }))} />
          </div>
        </div>
      </section>

      <section className="col-span-3 panel p-3 overflow-y-auto">
        <div className="axis-label mb-2">turn log</div>
        <ol className="space-y-1 text-xxs num">
          {history.length === 0 && <li className="text-ink-mute">— no turns yet</li>}
          {history.map((h, i) => (
            <li key={i} className="flex justify-between border-b border-line/60 py-1">
              <span className="text-ink-mute">t{h.turn}</span>
              <span className="text-ink-dim">
                {actions.political[h.last_action?.pol ?? 0]}/
                {actions.military[h.last_action?.mil ?? 0]}@{h.last_action?.target}
              </span>
              <span className={(h.reward ?? 0) >= 0 ? "text-ok" : "text-invader"}>
                {h.reward !== undefined ? (h.reward >= 0 ? "+" : "") + h.reward.toFixed(2) : "—"}
              </span>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
