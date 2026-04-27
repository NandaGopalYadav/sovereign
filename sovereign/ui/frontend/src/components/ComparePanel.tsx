import { useEffect, useState } from "react";
import { api } from "../lib/api";

type Row = Awaited<ReturnType<typeof api.compare>>[number];

export default function ComparePanel() {
  const [rows, setRows] = useState<Row[]>([]);

  useEffect(() => {
    api.compare().then(setRows).catch(() => setRows([]));
  }, []);

  if (rows.length === 0) {
    return (
      <div className="p-6 text-xs text-ink-dim">
        no ablation eval results yet · run{" "}
        <span className="num text-accent">python -m sovereign.experiments.ablations</span>
      </div>
    );
  }

  const maxReturn = Math.max(...rows.map((r) => r.return_mean));
  const minReturn = Math.min(...rows.map((r) => r.return_mean));

  return (
    <div className="p-3 grid grid-cols-1 md:grid-cols-5 gap-3 h-full overflow-y-auto">
      {rows.map((r) => {
        const span = maxReturn - minReturn || 1;
        const norm = (r.return_mean - minReturn) / span;
        return (
          <div key={r.regime} className="panel p-3 flex flex-col gap-2">
            <div className="flex items-baseline justify-between">
              <span className="text-xs uppercase tracking-wider text-ink">{r.regime}</span>
              <span className="text-xxs text-ink-mute">n={r.n_episodes}</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="num text-2xl">
                {r.return_mean >= 0 ? "+" : ""}{r.return_mean.toFixed(1)}
              </span>
              <span className="text-xxs text-ink-mute num">±{r.return_std.toFixed(1)}</span>
            </div>
            <div className="h-1 bg-line">
              <div
                className="h-full bg-accent"
                style={{ width: `${Math.max(8, norm * 100)}%` }}
              />
            </div>
            <div className="grid grid-cols-2 text-xxs gap-y-1 text-ink-dim">
              <span>settle</span><span className="num text-right">{(r.settlement_rate * 100).toFixed(0)}%</span>
              <span>conquest</span><span className="num text-right">{(r.conquest_rate * 100).toFixed(0)}%</span>
              <span>destroyed</span><span className="num text-right">{(r.destroyed_rate * 100).toFixed(0)}%</span>
              <span>collapse</span><span className="num text-right">{(r.legitimacy_collapse_rate * 100).toFixed(0)}%</span>
              <span>timeout</span><span className="num text-right">{(r.timeout_rate * 100).toFixed(0)}%</span>
            </div>
            <div className="grid grid-cols-2 text-xxs gap-y-1 mt-2 text-ink-dim">
              <span>L̄</span><span className="num text-right">{r.mean_legitimacy.toFixed(2)}</span>
              <span>θ̄</span>
              <span className={"num text-right " + (r.mean_theta >= 0 ? "text-warn" : "text-ok")}>
                {r.mean_theta >= 0 ? "+" : ""}{r.mean_theta.toFixed(2)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
