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
  PricePoint,
  SeasonalityPoint,
} from "@/lib/forecast";

type Props = {
  history: PricePoint[];
  forecast: ForecastPoint[];
  weekly: SeasonalityPoint[];
  yearly: SeasonalityPoint[];
  trend: SeasonalityPoint[];
  years: number;
  ticker: string;
};

function money(value: number) {
  return `$${value.toFixed(2)}`;
}

export function ForecastCharts({
  history,
  forecast,
  weekly,
  yearly,
  trend,
  years,
  ticker,
}: Props) {
  const historyChart = history.map((h) => ({
    date: h.date,
    open: h.open,
    close: h.close,
  }));

  const forecastChart = forecast.map((f) => ({
    date: f.date,
    actual: f.actualClose,
    yhat: f.yhat,
    lower: f.yhatLower,
    upper: f.yhatUpper,
  }));

  return (
    <div className="charts-stack">
      <section className="panel reveal" style={{ animationDelay: "80ms" }}>
        <div className="panel-heading">
          <h2>Price history</h2>
          <p>Open and close for {ticker} with a rangeslider-friendly view.</p>
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
                tickFormatter={(v) => `$${v}`}
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
          <h2>Forecast plot · {years} year{years > 1 ? "s" : ""}</h2>
          <p>
            Trend plus weekly and yearly seasonality, with a 95% uncertainty
            band.
          </p>
        </div>
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height={360}>
            <AreaChart data={forecastChart}>
              <defs>
                <linearGradient id="band" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#1f8a6e" stopOpacity={0.22} />
                  <stop offset="100%" stopColor="#1f8a6e" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(18,32,26,0.08)" vertical={false} />
              <XAxis
                dataKey="date"
                minTickGap={56}
                tick={{ fill: "#4a5c52", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tickFormatter={(v) => `$${v}`}
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
              <Area
                type="monotone"
                dataKey="upper"
                stroke="none"
                fill="url(#band)"
                name="Upper"
              />
              <Area
                type="monotone"
                dataKey="lower"
                stroke="none"
                fill="#f3f6f2"
                name="Lower"
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
                name="Forecast"
                stroke="#1f8a6e"
                dot={false}
                strokeWidth={2}
                strokeDasharray="6 4"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="components-grid reveal" style={{ animationDelay: "240ms" }}>
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
                  tickFormatter={(v) => `$${v}`}
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
            <h2>Weekly seasonality</h2>
            <p>Average weekday lift after removing trend.</p>
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
            <p>Month-level Fourier seasonality in the residual.</p>
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
