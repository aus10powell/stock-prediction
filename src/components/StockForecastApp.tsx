"use client";

import { FormEvent, useEffect, useState, useTransition } from "react";
import { ForecastCharts } from "@/components/ForecastCharts";
import { MonteCarloPanel } from "@/components/MonteCarloPanel";
import type {
  ForecastPoint,
  PricePoint,
  SeasonalityPoint,
} from "@/lib/forecast";
import type { MonteCarloResult } from "@/lib/monteCarlo";

type ForecastResponse = {
  ticker: string;
  years: number;
  days: number;
  history: PricePoint[];
  forecast: ForecastPoint[];
  weekly: SeasonalityPoint[];
  yearly: SeasonalityPoint[];
  trend: SeasonalityPoint[];
  rawTail: PricePoint[];
  forecastTail: ForecastPoint[];
  monteCarlo: MonteCarloResult;
  error?: string;
};

export function StockForecastApp() {
  const [ticker, setTicker] = useState("GME");
  const [years, setYears] = useState(1);
  const [days, setDays] = useState(20);
  const [feedback, setFeedback] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [isPending, startTransition] = useTransition();

  async function fetchForecast(
    nextTicker: string,
    nextYears: number,
    nextDays: number,
  ) {
    const params = new URLSearchParams({
      ticker: nextTicker,
      years: String(nextYears),
      days: String(nextDays),
    });
    const res = await fetch(`/api/forecast?${params.toString()}`);
    const json = (await res.json()) as ForecastResponse;
    if (!res.ok) {
      throw new Error(json.error ?? "Forecast failed.");
    }
    return json;
  }

  function runForecast(
    nextTicker = ticker,
    nextYears = years,
    nextDays = days,
  ) {
    startTransition(async () => {
      try {
        const json = await fetchForecast(nextTicker, nextYears, nextDays);
        setError(null);
        setData(json);
      } catch (err) {
        setData(null);
        setError(
          err instanceof Error ? err.message : "Network error while loading forecast.",
        );
      }
    });
  }

  useEffect(() => {
    let cancelled = false;
    startTransition(async () => {
      try {
        const json = await fetchForecast("GME", 1, 20);
        if (cancelled) return;
        setError(null);
        setData(json);
      } catch (err) {
        if (cancelled) return;
        setData(null);
        setError(
          err instanceof Error ? err.message : "Network error while loading forecast.",
        );
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    runForecast();
  }

  return (
    <div className="app-shell">
      <div className="atmosphere" aria-hidden="true" />

      <header className="hero">
        <p className="byline">Austin Powell</p>
        <h1 className="brand">Stock Forecast</h1>
        <p className="lede">
          Explore trend forecasts and the historical probability of a stock
          moving at least 1% over your chosen horizon.
        </p>

        <form className="controls" onSubmit={onSubmit}>
          <label className="field">
            <span>Ticker</span>
            <input
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="GME"
              maxLength={10}
              aria-label="Stock ticker symbol"
            />
          </label>

          <label className="field slider-field">
            <span>
              Years ahead <strong>{years}</strong>
            </span>
            <input
              type="range"
              min={1}
              max={4}
              step={1}
              value={years}
              onChange={(e) => setYears(Number(e.target.value))}
              aria-label="Years ahead to predict"
            />
          </label>

          <label className="field slider-field">
            <span>
              Probability horizon <strong>{days} days</strong>
            </span>
            <input
              type="range"
              min={1}
              max={252}
              step={1}
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              aria-label="Trading days for probability simulation"
            />
          </label>

          <button className="cta" type="submit" disabled={isPending}>
            {isPending ? "Simulating…" : "Run analysis"}
          </button>
        </form>

        {error ? <p className="error">{error}</p> : null}
        {isPending ? (
          <p className="status">
            Loading market data, fitting the model, and simulating 10,000
            paths…
          </p>
        ) : null}
      </header>

      {data ? (
        <main className="results">
          <section className="panel reveal">
            <div className="panel-heading row">
              <div>
                <h2>Raw data</h2>
                <p>Latest closes for {data.ticker}</p>
              </div>
              <label className="feedback">
                <span>Any feedback on the app?</span>
                <textarea
                  rows={3}
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  placeholder="Love it! Suggested improvements?"
                />
              </label>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Open</th>
                    <th>Close</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rawTail.map((row) => (
                    <tr key={row.date}>
                      <td>{row.date}</td>
                      <td>${row.open.toFixed(2)}</td>
                      <td>${row.close.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <MonteCarloPanel result={data.monteCarlo} ticker={data.ticker} />

          <ForecastCharts
            history={data.history}
            forecast={data.forecast}
            weekly={data.weekly}
            yearly={data.yearly}
            trend={data.trend}
            years={data.years}
            ticker={data.ticker}
          />

          <section className="panel reveal" style={{ animationDelay: "280ms" }}>
            <div className="panel-heading">
              <h2>Forecast data</h2>
              <p>Tail of the predicted series</p>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>yhat</th>
                    <th>Lower</th>
                    <th>Upper</th>
                  </tr>
                </thead>
                <tbody>
                  {data.forecastTail.map((row) => (
                    <tr key={row.date}>
                      <td>{row.date}</td>
                      <td>${row.yhat.toFixed(2)}</td>
                      <td>${row.yhatLower.toFixed(2)}</td>
                      <td>${row.yhatUpper.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </main>
      ) : null}

      <footer className="footer">
        Migrated from the original Streamlit + Prophet Heroku app to Next.js on
        Vercel.
      </footer>
    </div>
  );
}
