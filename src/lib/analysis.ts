import {
  ANALYSIS_SEED,
  type Analysis,
  type CalibrationPayload,
  type ForecastPayload,
  type ProbabilityPayload,
} from "./analysisTypes";
import { runCalibrationBacktest } from "./calibration";
import { buildForecast } from "./forecast";
import { buildMonteCarloForecast } from "./monteCarlo";
import type { Settings } from "./settings";
import { loadStockHistory, normalizeTicker } from "./stocks";
import { TRADING_DAYS_PER_YEAR } from "./tradingCalendar";

function message(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

/**
 * Server-side entry point for the initial render. Calibration is reported
 * separately because it is the slowest and least essential part: a horizon too
 * long to backtest should still render probabilities.
 */
export async function runAnalysis(settings: Settings): Promise<Analysis> {
  let ticker: string;
  try {
    ticker = normalizeTicker(settings.ticker);
  } catch (error) {
    return { error: message(error, "Invalid ticker.") };
  }

  try {
    const history = await loadStockHistory(ticker);

    const probability: ProbabilityPayload = {
      ticker,
      ...buildMonteCarloForecast(history, {
        horizonDays: settings.days,
        thresholdPercent: settings.threshold,
        driftMode: settings.drift,
        seed: ANALYSIS_SEED,
      }),
    };

    const forecastResult = buildForecast(
      history,
      settings.years * TRADING_DAYS_PER_YEAR,
    );
    const forecast: ForecastPayload = {
      ticker,
      years: settings.years,
      ...forecastResult,
      rawTail: history.slice(-8),
    };

    let calibration: CalibrationPayload | undefined;
    let calibrationError: string | undefined;
    try {
      calibration = {
        ticker,
        ...runCalibrationBacktest(history, {
          horizonDays: settings.days,
          thresholdPercent: settings.threshold,
          driftMode: settings.drift,
          seed: ANALYSIS_SEED,
        }),
      };
    } catch (error) {
      calibrationError = message(error, "Calibration failed.");
    }

    return { probability, forecast, calibration, calibrationError };
  } catch (error) {
    return { error: message(error, "Unable to load market data.") };
  }
}
