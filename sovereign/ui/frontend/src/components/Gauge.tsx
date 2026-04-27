// A bipolar threshold gauge. Designed for θ ∈ [-1, +1] where bands annotate the
// hysteretic events. For unipolar metrics (L, E ∈ [0,1]) pass `min=0, max=1` and
// no bands.

type Band = {
  from: number;
  to: number;
  label: string;
  color: "warn" | "ok" | "danger";
};

type Props = {
  label: string;
  value: number;
  min: number;
  max: number;
  bands?: Band[];
  unit?: string;
};

const COLOR: Record<Band["color"], string> = {
  warn: "rgba(211,148,89,0.22)",
  ok: "rgba(107,163,107,0.22)",
  danger: "rgba(201,87,87,0.22)",
};

export default function Gauge({ label, value, min, max, bands = [], unit }: Props) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="panel p-3">
      <div className="flex items-baseline justify-between mb-2">
        <span className="axis-label">{label}</span>
        <span className="num text-sm">
          {value.toFixed(2)}
          {unit ? <span className="text-ink-mute ml-1">{unit}</span> : null}
        </span>
      </div>
      <div className="relative h-4 gauge-track border border-line">
        {bands.map((b, i) => {
          const left = ((b.from - min) / (max - min)) * 100;
          const width = ((b.to - b.from) / (max - min)) * 100;
          return (
            <div
              key={i}
              className="absolute top-0 bottom-0"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                background: COLOR[b.color],
              }}
              title={b.label}
            />
          );
        })}
        <div
          className="absolute top-[-2px] bottom-[-2px] w-px bg-accent"
          style={{ left: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between mt-1 text-xxs text-ink-mute num">
        <span>{min.toFixed(1)}</span>
        <span>{max.toFixed(1)}</span>
      </div>
    </div>
  );
}
