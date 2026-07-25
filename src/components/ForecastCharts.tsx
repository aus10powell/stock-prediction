"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  ForecastPoint,
  ModelFit,
  PricePoint,
  SeasonalityPoint,
} from "@/lib/forecast";
import { money } from "@/lib/format";

type Props = {
  history: PricePoint[];
  forecast: ForecastPoint[];
  weekly: SeasonalityPoint[];
  yearly: SeasonalityPoint[];
  trend: SeasonalityPoint[];
  fit: ModelFit;
  years: number;
  ticker: string;
};

export function ForecastCharts({
  history,
  forecast,
  weekly,
  yearly,
  trend,
  fit,
  years,
  ticker,
}: Props) {
  const historyChart = history.map((point) => ({
    date: point.date,
    open: point.open,
    close: point.close,
  }));

  const forecastChart = forecast.map((point) => ({
    date: point.date,
    actual: point.actualClose,
    yhat: point.yhat,
    range: [point.yhatLower, point.yhatUpper],
  }));

  return (
    <div className="charts-stack">
      <section className="panel reveal" style={{ animationDelay: "80ms" }}>
        <div className="panel-heading">
          <h2>Price history</h2>
          <p>Open and close for {ticker} since 2015.</p>
        </div>
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={historyChart}>
              <CartesianGrid stroke="rgba(18,32,26,0.08)" vertical={false} />
              <XAxis
                dataKey="date"
                minTickGap={48}
                tick={{ fill: "#4a5c52", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tickFormatter={(value) => `$${value}`}
                width={56}
                tick={{ fill: "#4a5c52", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                formatter={(value) => money(Number(value))}
                contentStyle={tooltipStyle}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="open"
                name="Open"
                stroke="#2a6f97"
                dot={false}
                strokeWidth={1.75}
              />
              <Line
                type="monotone"
                dataKey="close"
                name="Close"
                stroke="#1f8a6e"
                dot={false}
                strokeWidth={1.75}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel reveal" style={{ animationDelay: "160ms" }}>
        <div className="panel-heading">
          <p className="eyebrow">Model fit diagnostic</p>
          <h2>Trend and seasonality fit</h2>
          <p>
            How well a straight trend plus weekday and yearly seasonal terms
            describe {ticker}, extended {years * 252} trading days past the last
            close. Prices do not actually follow a deterministic trend, so read
            the extension as the shape of the fitted line, not a forecast — the
            probability panel above is the forward-looking view.
          </p>
        </div>

        <dl className="assumptions compact">
          <div>
            <dt>R²</dt>
            <dd>{fit.rSquared.toFixed(3)}</dd>
          </div>
          <div>
            <dt>Residual std</dt>
            <dd>{money(fit.residualStd)}</dd>
          </div>
          <div>
            <dt>Trend per trading day</dt>
            <dd>{money(fit.slopePerTradingDay)}</dd>
          </div>
          <div>
            <dt>Residual autocorrelation</dt>
            <dd>{fit.residualAutocorrelation.toFixed(3)}</dd>
          </div>
        </dl>

        <div className="chart-frame">
          <ResponsiveContainer width="100%" height={360}>
            <AreaChart data={forecastChart}>
              <CartesianGrid stroke="rgba(18,32,26,0.08)" vertical={false} />
              <XAxis
                dataKey="date"
                minTickGap={56}
                tick={{ fill: "#4a5c52", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tickFormatter={(value) => `$${value}`}
                width={56}
                tick={{ fill: "#4a5c52", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                formatter={(value, name) =>
                  Array.isArray(value)
                    ? [`${money(value[0])} – ${money(value[1])}`, name]
                    : [money(Number(value)), name]
                }
                contentStyle={tooltipStyle}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="range"
                name="95% prediction interval"
                stroke="none"
                fill="#1f8a6e"
                fillOpacity={0.16}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="actual"
                name="Actual close"
                stroke="#12201a"
                dot={false}
                strokeWidth={1.5}
              />
              <Line
                type="monotone"
                dataKey="yhat"
                name="Fitted"
                stroke="#1f8a6e"
                dot={false}
                strokeWidth={2}
                strokeDasharray="6 4"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <p className="disclaimer">
          The interval widens with the horizon. Residual autocorrelation of{" "}
          {fit.residualAutocorrelation.toFixed(2)} means deviations from the
          trend persist rather than reset each day, so uncertainty accumulates
          the further out the line is extended.
        </p>
      </section>

      <section
        className="components-grid reveal"
        style={{ animationDelay: "240ms" }}
      >
        <div className="panel">
          <div className="panel-heading">
            <h2>Trend</h2>
            <p>Long-run direction in the fitted series.</p>
          </div>
          <div className="chart-frame compact">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={trend}>
                <CartesianGrid stroke="rgba(18,32,26,0.08)" vertical={false} />
                <XAxis dataKey="label" hide />
                <YAxis
                  tickFormatter={(value) => `$${value}`}
                  width={48}
                  tick={{ fill: "#4a5c52", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  formatter={(value) => money(Number(value))}
                  contentStyle={tooltipStyle}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#2a6f97"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <h2>Weekday effect</h2>
            <p>Average weekday offset, jointly estimated.</p>
          </div>
          <div className="chart-frame compact">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={weekly}>
                <CartesianGrid stroke="rgba(18,32,26,0.08)" vertical={false} />
                <XAxis
                  dataKey="label"
                  tick={{ fill: "#4a5c52", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "#4a5c52", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={40}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" fill="#1f8a6e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <h2>Yearly seasonality</h2>
            <p>Month-level Fourier terms. Largely noise for equities.</p>
          </div>
          <div className="chart-frame compact">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={yearly}>
                <CartesianGrid stroke="rgba(18,32,26,0.08)" vertical={false} />
                <XAxis
                  dataKey="label"
                  tick={{ fill: "#4a5c52", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: "#4a5c52", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={40}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" fill="#2a6f97" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </div>
  );
}

const tooltipStyle = {
  background: "#12201a",
  border: "none",
  borderRadius: 8,
  color: "#f3f6f2",
  fontSize: 12,
};
