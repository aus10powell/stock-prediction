"use client";

import { FormEvent, useState } from "react";
import { CalibrationPanel } from "@/components/CalibrationPanel";
import { ForecastCharts } from "@/components/ForecastCharts";
import { ProbabilityPanel } from "@/components/ProbabilityPanel";
import { WatchlistPanel } from "@/components/WatchlistPanel";
import {
  ANALYSIS_SEED,
  type Analysis,
  type CalibrationPayload,
  type ForecastPayload,
  type ProbabilityPayload,
} from "@/lib/analysisTypes";
import type { DriftMode } from "@/lib/monteCarlo";
import { settingsToQuery, type Settings } from "@/lib/settings";

type Props = {
  settings: Settings;
  analysis: Analysis;
};

async function getJson<T>(
  path: string,
  params: URLSearchParams,
): Promise<T> {
  const response = await fetch(`${path}?${params.toString()}`);
  const json = (await response.json()) as T & { error?: string };
  if (!response.ok) {
    throw new Error(json.error ?? "Request failed.");
  }
  return json;
}

export function StockForecastApp({ settings: initial, analysis }: Props) {
  const [settings, setSettings] = useState<Settings>(initial);
  const [probability, setProbability] = useState<ProbabilityPayload | undefined>(
    analysis.probability,
  );
  const [forecast, setForecast] = useState<ForecastPayload | undefined>(
    analysis.forecast,
  );
  const [calibration, setCalibration] = useState<CalibrationPayload | undefined>(
    analysis.calibration,
  );
  const [error, setError] = useState<string | null>(analysis.error ?? null);
  const [calibrationError, setCalibrationError] = useState<string | null>(
    analysis.calibrationError ?? null,
  );
  const [isLoading, setIsLoading] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    window.history.replaceState(null, "", `?${settingsToQuery(settings)}`);

    setIsLoading(true);
    setCalibrationError(null);

    const shared = new URLSearchParams({
      ticker: settings.ticker,
      days: String(settings.days),
      threshold: String(settings.threshold),
      drift: settings.drift,
      seed: String(ANALYSIS_SEED),
    });

    try {
      const [nextProbability, nextForecast] = await Promise.all([
        getJson<ProbabilityPayload>("/api/probability", shared),
        getJson<ForecastPayload>(
          "/api/forecast",
          new URLSearchParams({
            ticker: settings.ticker,
            years: String(settings.years),
          }),
        ),
      ]);
      setError(null);
      setProbability(nextProbability);
      setForecast(nextForecast);
    } catch (err) {
      setProbability(undefined);
      setForecast(undefined);
      setCalibration(undefined);
      setError(err instanceof Error ? err.message : "Unable to load data.");
      return;
    } finally {
      setIsLoading(false);
    }

    // Backtesting is the slowest step, so it resolves after the main results.
    setIsCalibrating(true);
    try {
      setCalibration(
        await getJson<CalibrationPayload>("/api/calibration", shared),
      );
    } catch (err) {
      setCalibration(undefined);
      setCalibrationError(
        err instanceof Error ? err.message : "Calibration failed.",
      );
    } finally {
      setIsCalibrating(false);
    }
  }

  function update<K extends keyof Settings>(key: K, value: Settings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  return (
    <div className="app-shell">
      <div className="atmosphere" aria-hidden="true" />

      <header className="hero">
        <p className="byline">Austin Powell</p>
        <h1 className="brand">Stock Odds</h1>
        <p className="lede">
          Simulated probabilities that a stock moves by at least a chosen
          percentage over a chosen number of trading days, backtested against
          what actually happened.
        </p>

        <form className="controls" onSubmit={onSubmit}>
          <label className="field">
            <span>Ticker</span>
            <input
              value={settings.ticker}
              onChange={(event) =>
                update("ticker", event.target.value.toUpperCase())
              }
              placeholder="GME"
              maxLength={10}
              aria-label="Stock ticker symbol"
            />
          </label>

          <label className="field slider-field">
            <span>
              Horizon <strong>{settings.days} trading days</strong>
            </span>
            <input
              type="range"
              min={1}
              max={252}
              step={1}
              value={settings.days}
              onChange={(event) => update("days", Number(event.target.value))}
              aria-label="Trading days for the probability simulation"
            />
          </label>

          <label className="field slider-field">
            <span>
              Move threshold <strong>±{settings.threshold}%</strong>
            </span>
            <input
              type="range"
              min={0.5}
              max={20}
              step={0.5}
              value={settings.threshold}
              onChange={(event) =>
                update("threshold", Number(event.target.value))
              }
              aria-label="Percentage move threshold"
            />
          </label>

          <label className="field">
            <span>Drift</span>
            <select
              value={settings.drift}
              onChange={(event) =>
                update("drift", event.target.value as DriftMode)
              }
              aria-label="Drift assumption"
            >
              <option value="historical">Historical</option>
              <option value="zero">Zero</option>
            </select>
          </label>

          <label className="field">
            <span>Fit window</span>
            <select
              value={settings.years}
              onChange={(event) => update("years", Number(event.target.value))}
              aria-label="Years to extend the model fit"
            >
              {[1, 2, 3, 4].map((value) => (
                <option key={value} value={value}>
                  {value} year{value > 1 ? "s" : ""}
                </option>
              ))}
            </select>
          </label>

          <button className="cta" type="submit" disabled={isLoading}>
            {isLoading ? "Simulating…" : "Run analysis"}
          </button>
        </form>

        {error ? <p className="error">{error}</p> : null}
        {isLoading ? (
          <p className="status">
            Loading market data and simulating 10,000 paths…
          </p>
        ) : null}
      </header>

      <main className="results">
        {probability ? (
          <ProbabilityPanel result={probability} ticker={probability.ticker} />
        ) : null}

        {calibration ? (
          <CalibrationPanel result={calibration} ticker={calibration.ticker} />
        ) : null}
        {isCalibrating ? (
          <p className="status">Backtesting historical calibration…</p>
        ) : null}
        {calibrationError ? (
          <p className="status">Calibration unavailable: {calibrationError}</p>
        ) : null}

        <WatchlistPanel
          horizonDays={settings.days}
          thresholdPercent={settings.threshold}
          driftMode={settings.drift}
          seed={ANALYSIS_SEED}
        />

        {forecast ? (
          <>
            <ForecastCharts
              history={forecast.history}
              forecast={forecast.forecast}
              weekly={forecast.weekly}
              yearly={forecast.yearly}
              trend={forecast.trend}
              fit={forecast.fit}
              years={forecast.years}
              ticker={forecast.ticker}
            />

            <section className="panel reveal">
              <div className="panel-heading">
                <h2>Raw data</h2>
                <p>Latest closes for {forecast.ticker}</p>
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
                    {forecast.rawTail.map((row) => (
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
          </>
        ) : null}
      </main>

      <footer className="footer">
        Migrated from the original Streamlit + Prophet Heroku app to Next.js on
        Vercel. Historical simulation only — not investment advice.
      </footer>
    </div>
  );
}
