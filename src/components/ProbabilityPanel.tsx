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
import { money, percent, signedPercent } from "@/lib/format";

type Props = {
  result: MonteCarloResult;
  ticker: string;
};

export function ProbabilityPanel({ result, ticker }: Props) {
  const { assumptions } = result;
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
          {ticker} over the next {result.horizonDays} trading day
          {result.horizonDays === 1 ? "" : "s"}
        </h2>
        <p>
          {assumptions.simulations.toLocaleString()} paths, resampled in{" "}
          {assumptions.blockSize}-day blocks from the latest{" "}
          {assumptions.lookbackDays.toLocaleString()} daily returns.
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
          <strong>{signedPercent(result.medianReturn)}</strong>
        </article>
      </div>

      <dl className="assumptions">
        <div>
          <dt>Current close</dt>
          <dd>{money(result.currentPrice)}</dd>
        </div>
        <div>
          <dt>80% simulated range</dt>
          <dd>
            {money(result.terminalP10)} – {money(result.terminalP90)}
          </dd>
        </div>
        <div>
          <dt>Assumed drift (annualized)</dt>
          <dd>
            {assumptions.driftMode === "zero"
              ? "0% (drift removed)"
              : signedPercent(assumptions.annualizedDrift)}
          </dd>
        </div>
        <div>
          <dt>Volatility (annualized)</dt>
          <dd>{percent(assumptions.annualizedVolatility)}</dd>
        </div>
      </dl>

      {assumptions.driftMode === "historical" ? (
        <p className="callout">
          These paths inherit the average drift of the lookback window, so a
          period of strong gains tilts the upside probability regardless of
          today&apos;s conditions. Switch drift to <strong>zero</strong> to see
          the volatility-only answer.
        </p>
      ) : (
        <p className="callout">
          Drift has been removed, so up and down probabilities reflect
          volatility and the shape of the return distribution alone.
        </p>
      )}

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
              formatter={(value, name) =>
                Array.isArray(value)
                  ? [`${money(value[0])} – ${money(value[1])}`, name]
                  : [money(Number(value)), name]
              }
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
        This is a historical simulation, not a prediction. It assumes recent
        return patterns may recur and cannot account for news, earnings, or
        regime changes. Not investment advice.
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
