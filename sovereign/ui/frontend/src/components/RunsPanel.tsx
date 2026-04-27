import { useEffect, useMemo, useState } from "react";
import { api, EpisodeSummary, TraceFrame } from "../lib/api";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import RewardStack from "./RewardStack";

export default function RunsPanel() {
  const [runs, setRuns] = useState<{ id: string }[]>([]);
  const [activeRun, setActiveRun] = useState<string | null>(null);
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [activeEp, setActiveEp] = useState<number | null>(null);
  const [trace, setTrace] = useState<TraceFrame[]>([]);

  useEffect(() => {
    api.runs().then((rs) => {
      setRuns(rs);
      if (rs.length > 0) setActiveRun(rs[0].id);
    });
  }, []);

  useEffect(() => {
    if (!activeRun) return;
    api.runEpisodes(activeRun).then((eps) => {
      setEpisodes(eps);
      const traceable = eps.find((e) => e.episode % 25 === 0);
      if (traceable) setActiveEp(traceable.episode);
    });
  }, [activeRun]);

  useEffect(() => {
    if (!activeRun || activeEp === null) {
      setTrace([]);
      return;
    }
    api.runTrace(activeRun, activeEp).then(setTrace).catch(() => setTrace([]));
  }, [activeRun, activeEp]);

  const returnSeries = useMemo(
    () => episodes.map((e) => ({ ep: e.episode, ret: e.return })),
    [episodes]
  );

  return (
    <div className="grid grid-cols-12 gap-3 p-3 h-full overflow-hidden">
      <aside className="col-span-3 panel p-3 overflow-y-auto">
        <div className="axis-label mb-2">runs</div>
        {runs.length === 0 && (
          <div className="text-xxs text-ink-mute">
            no runs found — train a model and refresh
          </div>
        )}
        <ul className="space-y-1">
          {runs.map((r) => (
            <li key={r.id}>
              <button
                onClick={() => setActiveRun(r.id)}
                className={
                  "w-full text-left text-xs num px-2 py-1 border " +
                  (r.id === activeRun
                    ? "border-accent text-ink"
                    : "border-line text-ink-dim hover:text-ink")
                }
              >
                {r.id}
              </button>
            </li>
          ))}
        </ul>
        <div className="axis-label mt-4 mb-2">episodes ({episodes.length})</div>
        <ul className="space-y-1 max-h-[40vh] overflow-y-auto">
          {episodes.map((e) => (
            <li key={e.episode}>
              <button
                onClick={() => setActiveEp(e.episode)}
                disabled={e.episode % 25 !== 0}
                className={
                  "w-full text-left text-xxs num px-2 py-0.5 flex justify-between " +
                  (e.episode === activeEp
                    ? "bg-accent/15 text-ink"
                    : "text-ink-dim hover:text-ink") +
                  (e.episode % 25 !== 0 ? " opacity-40 cursor-not-allowed" : "")
                }
              >
                <span>#{e.episode}</span>
                <span>{e.return.toFixed(2)}</span>
                <span className="text-ink-mute">{e.terminal_reason ?? "—"}</span>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <section className="col-span-9 grid grid-rows-[230px_230px_1fr] gap-3 overflow-hidden">
        <div className="panel p-3 overflow-hidden">
          <div className="axis-label mb-1">return per episode</div>
          <div className="h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={returnSeries} margin={{ left: 8, right: 8, top: 4, bottom: 0 }}>
                <CartesianGrid stroke="#23262d" strokeDasharray="2 2" />
                <XAxis dataKey="ep" stroke="#6b7280" tick={{ fontSize: 10, fill: "#9ba0aa" }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 10, fill: "#9ba0aa" }} />
                <Tooltip
                  contentStyle={{ background: "#14161b", border: "1px solid #2a2d35", color: "#e6e7ea", fontSize: 12 }}
                  labelStyle={{ color: "#9ba0aa" }}
                />
                <Line type="monotone" dataKey="ret" stroke="#d39459" dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel p-3 overflow-hidden">
          <div className="axis-label mb-1">
            θ &amp; L for episode {activeEp ?? "—"}
          </div>
          <div className="h-[180px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trace} margin={{ left: 8, right: 8, top: 4, bottom: 0 }}>
                <CartesianGrid stroke="#23262d" strokeDasharray="2 2" />
                <XAxis dataKey="t" stroke="#6b7280" tick={{ fontSize: 10, fill: "#9ba0aa" }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 10, fill: "#9ba0aa" }} />
                <Tooltip contentStyle={{ background: "#14161b", border: "1px solid #2a2d35", color: "#e6e7ea", fontSize: 12 }} />
                <Line type="monotone" dataKey="theta" stroke="#d39459" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="legitimacy" stroke="#5b8eda" dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel p-3 overflow-hidden">
          <div className="axis-label mb-1">reward decomposition over episode</div>
          <div className="h-full pb-2">
            <RewardStack history={trace} />
          </div>
        </div>
      </section>
    </div>
  );
}
