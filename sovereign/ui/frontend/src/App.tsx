import { useState } from "react";
import LivePanel from "./components/LivePanel";
import RunsPanel from "./components/RunsPanel";
import ComparePanel from "./components/ComparePanel";

const TABS = [
  { id: "live", label: "Live" },
  { id: "runs", label: "Runs" },
  { id: "compare", label: "Compare" },
] as const;
type TabId = (typeof TABS)[number]["id"];

export default function App() {
  const [tab, setTab] = useState<TabId>("live");

  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-line bg-bg-elev">
        <div className="px-5 py-3 flex items-center gap-6">
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-accent text-sm tracking-wide">
              sovereign
            </span>
            <span className="text-xxs text-ink-mute uppercase tracking-widest">
              v0.1 · research console
            </span>
          </div>
          <nav className="flex gap-1 ml-6">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={
                  "px-3 py-1 text-xs uppercase tracking-wider border " +
                  (tab === t.id
                    ? "border-accent text-ink"
                    : "border-transparent text-ink-dim hover:text-ink")
                }
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className="ml-auto text-xxs text-ink-mute">
            three-nation conflict / dominated-strategy probe
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-hidden">
        {tab === "live" && <LivePanel />}
        {tab === "runs" && <RunsPanel />}
        {tab === "compare" && <ComparePanel />}
      </main>
    </div>
  );
}
