"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MonteCarloResult } from "@/lib/monteCarlo";

type Props = {
  result: MonteCarloResult;
  ticker: string;
};

function percent(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value);
}

function money(value: number) {
  return `$${value.toFixed(2)}`;
}

export function MonteCarloPanel({ result, ticker }: Props) {
  const chartData = result.percentiles.map((point) => ({
    day: point.day,
    outerRange: [point.p10, point.p90],
    innerRange: [point.p25, point.p75],
    median: point.median,
  }));

  return (
    <section className="panel reveal probability-panel">
      <div className="panel-heading">
        <p className="eyebrow">Historical simulation</p>
        <h2>
          {ticker} after {result.horizonDays} trading day
          {result.horizonDays === 1 ? "" : "s"}
        </h2>
        <p>
          {result.simulations.toLocaleString()} paths sampled in{" "}
          {result.blockSize}-day blocks from the latest{" "}
          {result.lookbackDays.toLocaleString()} daily returns.
        </p>
      </div>

      <div className="probability-grid">
        <article className="metric-card positive">
          <span>Up at least {result.thresholdPercent}%</span>
          <strong>{percent(result.probabilityUp)}</strong>
        </article>
        <article className="metric-card negative">
          <span>Down at least {result.thresholdPercent}%</span>
          <strong>{percent(result.probabilityDown)}</strong>
        </article>
        <article className="metric-card neutral">
          <span>Within ±{result.thresholdPercent}%</span>
          <strong>{percent(result.probabilityWithin)}</strong>
        </article>
        <article className="metric-card">
          <span>Median simulated return</span>
          <strong>{percent(result.medianReturn)}</strong>
        </article>
      </div>

      <div className="range-summary">
        <span>Current close: {money(result.currentPrice)}</span>
        <span>
          80% simulated range: {money(result.terminalP10)}–
          {money(result.terminalP90)}
        </span>
      </div>

      <div className="chart-frame">
        <ResponsiveContainer width="100%" height={340}>
          <AreaChart data={chartData}>
            <CartesianGrid stroke="rgba(18,32,26,0.08)" vertical={false} />
            <XAxis
              dataKey="day"
              tickFormatter={(value) => `Day ${value}`}
              minTickGap={36}
              tick={{ fill: "#4a5c52", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={["auto", "auto"]}
              tickFormatter={(value) => `$${Number(value).toFixed(0)}`}
              width={58}
              tick={{ fill: "#4a5c52", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              labelFormatter={(value) => `Trading day ${value}`}
              formatter={(value, name) => {
                if (Array.isArray(value)) {
                  return [`${money(value[0])} – ${money(value[1])}`, name];
                }
                return [money(Number(value)), name];
              }}
              contentStyle={tooltipStyle}
            />
            <Area
              type="monotone"
              dataKey="outerRange"
              name="10th–90th percentile"
              stroke="none"
              fill="#2a6f97"
              fillOpacity={0.14}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="innerRange"
              name="25th–75th percentile"
              stroke="none"
              fill="#1f8a6e"
              fillOpacity={0.22}
              isAnimationActive={false}
            />
            <ReferenceLine
              y={result.currentPrice}
              stroke="#4a5c52"
              strokeDasharray="4 4"
            />
            <Line
              type="monotone"
              dataKey="median"
              name="Median price"
              stroke="#146b55"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <p className="disclaimer">
        This historical simulation assumes recent return patterns may recur. It
        does not predict news, market regimes, or future performance and is not
        investment advice.
      </p>
    </section>
  );
}

const tooltipStyle = {
  background: "#12201a",
  border: "none",
  borderRadius: 8,
  color: "#f3f6f2",
  fontSize: 12,
};
