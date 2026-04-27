import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

type Frame = {
  t: number;
  reward_components: Record<string, number>;
};

type Props = {
  history: Frame[];
};

const KEYS = [
  { key: "territory_gain", color: "#6ba36b" },
  { key: "resource_gain", color: "#5b8eda" },
  { key: "occupation_cost", color: "#d3a14b" },
  { key: "legitimacy_cost", color: "#c95757" },
  { key: "sanction_cost", color: "#a259c9" },
  { key: "insurgency_cost", color: "#7388a3" },
  { key: "terminal_bonus", color: "#d39459" },
];

export default function RewardStack({ history }: Props) {
  const data = history.map((h, i) => ({ t: i, ...(h.reward_components ?? {}) }));
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ left: 8, right: 8, top: 4, bottom: 0 }}>
        <CartesianGrid stroke="#23262d" strokeDasharray="2 2" />
        <XAxis
          dataKey="t"
          stroke="#6b7280"
          tick={{ fontSize: 10, fill: "#9ba0aa" }}
        />
        <YAxis stroke="#6b7280" tick={{ fontSize: 10, fill: "#9ba0aa" }} />
        <Tooltip
          contentStyle={{
            background: "#14161b",
            border: "1px solid #2a2d35",
            color: "#e6e7ea",
            fontSize: 12,
          }}
          labelStyle={{ color: "#9ba0aa" }}
        />
        <ReferenceLine y={0} stroke="#3a3d44" />
        {KEYS.map((k) => (
          <Area
            key={k.key}
            type="monotone"
            dataKey={k.key}
            stackId="1"
            stroke={k.color}
            fill={k.color + "40"}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
