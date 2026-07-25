"use client";

import { FormEvent, useState } from "react";
import type { ScreenRow } from "@/app/api/screen/route";
import { money, percent, signedPercent } from "@/lib/format";

type ScreenResponse = {
  horizonDays: number;
  thresholdPercent: number;
  simulations: number;
  rows: ScreenRow[];
  error?: string;
};

type Props = {
  horizonDays: number;
  thresholdPercent: number;
  driftMode: string;
  seed: number;
};

type SortKey = "probabilityUp" | "probabilityDown" | "annualizedVolatility";

const DEFAULT_WATCHLIST = "AAPL, MSFT, NVDA, GME, TSLA";

export function WatchlistPanel({
  horizonDays,
  thresholdPercent,
  driftMode,
  seed,
}: Props) {
  const [input, setInput] = useState(DEFAULT_WATCHLIST);
  const [sortKey, setSortKey] = useState<SortKey>("probabilityUp");
  const [data, setData] = useState<ScreenResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function runScreen(event: FormEvent) {
    event.preventDefault();
    setIsLoading(true);
    try {
      const params = new URLSearchParams({
        tickers: input,
        days: String(horizonDays),
        threshold: String(thresholdPercent),
        drift: driftMode,
        seed: String(seed),
      });
      const response = await fetch(`/api/screen?${params.toString()}`);
      const json = (await response.json()) as ScreenResponse;
      if (!response.ok) throw new Error(json.error ?? "Screening failed.");
      setError(null);
      setData(json);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Screening failed.");
    } finally {
      setIsLoading(false);
    }
  }

  const rows = data
    ? [...data.rows].sort((a, b) => {
        if (a.error) return 1;
        if (b.error) return -1;
        return (b[sortKey] ?? 0) - (a[sortKey] ?? 0);
      })
    : [];

  return (
    <section className="panel reveal">
      <div className="panel-heading">
        <p className="eyebrow">Watchlist screen</p>
        <h2>Compare several tickers at once</h2>
        <p>
          Runs the same simulation across a list at the current horizon of{" "}
          {horizonDays} trading day{horizonDays === 1 ? "" : "s"} and a{" "}
          {thresholdPercent}% threshold.
        </p>
      </div>

      <form className="watchlist-controls" onSubmit={runScreen}>
        <label className="field">
          <span>Tickers (comma separated, up to 12)</span>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={DEFAULT_WATCHLIST}
            aria-label="Watchlist tickers"
          />
        </label>
        <label className="field">
          <span>Sort by</span>
          <select
            value={sortKey}
            onChange={(event) => setSortKey(event.target.value as SortKey)}
            aria-label="Sort watchlist by"
          >
            <option value="probabilityUp">Probability up</option>
            <option value="probabilityDown">Probability down</option>
            <option value="annualizedVolatility">Volatility</option>
          </select>
        </label>
        <button className="cta" type="submit" disabled={isLoading}>
          {isLoading ? "Screening…" : "Run screen"}
        </button>
      </form>

      {error ? <p className="error">{error}</p> : null}

      {rows.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Price</th>
                <th>Up ≥ {data?.thresholdPercent}%</th>
                <th>Down ≥ {data?.thresholdPercent}%</th>
                <th>Median</th>
                <th>Volatility</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.ticker}>
                  <td>
                    <strong>{row.ticker}</strong>
                  </td>
                  {row.error ? (
                    <td colSpan={5} className="muted-cell">
                      {row.error}
                    </td>
                  ) : (
                    <>
                      <td>{money(row.currentPrice ?? 0)}</td>
                      <td>{percent(row.probabilityUp ?? 0)}</td>
                      <td>{percent(row.probabilityDown ?? 0)}</td>
                      <td>{signedPercent(row.medianReturn ?? 0)}</td>
                      <td>{percent(row.annualizedVolatility ?? 0)}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
