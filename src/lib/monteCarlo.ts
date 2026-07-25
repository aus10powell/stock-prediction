import type { PricePoint } from "./forecast";
import { TRADING_DAYS_PER_YEAR } from "./tradingCalendar";

export type DriftMode = "historical" | "zero";

export type MonteCarloPoint = {
  day: number;
  p10: number;
  p25: number;
  median: number;
  p75: number;
  p90: number;
};

export type MonteCarloAssumptions = {
  /** Compounded drift of the sampled returns, annualized. */
  annualizedDrift: number;
  annualizedVolatility: number;
  driftMode: DriftMode;
  lookbackDays: number;
  blockSize: number;
  simulations: number;
  seed: number;
};

export type MonteCarloResult = {
  currentPrice: number;
  horizonDays: number;
  thresholdPercent: number;
  probabilityUp: number;
  probabilityDown: number;
  probabilityWithin: number;
  medianReturn: number;
  terminalP10: number;
  terminalP90: number;
  percentiles: MonteCarloPoint[];
  assumptions: MonteCarloAssumptions;
};

export type MonteCarloOptions = {
  horizonDays: number;
  simulations?: number;
  lookbackDays?: number;
  blockSize?: number;
  seed?: number;
  thresholdPercent?: number;
  driftMode?: DriftMode;
};

export const MONTE_CARLO_DEFAULTS = {
  simulations: 10_000,
  lookbackDays: 504,
  blockSize: 5,
  seed: 42,
  thresholdPercent: 1,
  driftMode: "historical" as DriftMode,
};

export const MAX_HORIZON_DAYS = 252;
export const MIN_HISTORY_DAYS = 30;

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

/** Linear-interpolated quantile of an already-sorted sample. */
function quantile(sorted: ArrayLike<number>, probability: number): number {
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  const weight = index - lower;
  if (lower + 1 >= sorted.length) return sorted[lower];
  return sorted[lower] * (1 - weight) + sorted[lower + 1] * weight;
}

function assertIntegerInRange(
  value: number,
  name: string,
  minimum: number,
  maximum: number,
) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(
      `${name} must be an integer between ${minimum} and ${maximum}.`,
    );
  }
}

/** Adjusted-close log returns, so splits and dividends do not look like moves. */
export function logReturns(history: PricePoint[], lookbackDays: number): number[] {
  const recent = history.slice(-(lookbackDays + 1));
  const prices = recent.map((point) => point.adjustedClose ?? point.close);
  if (prices.some((price) => !Number.isFinite(price) || price <= 0)) {
    throw new Error("Price history contains invalid adjusted close values.");
  }
  return prices.slice(1).map((price, index) => Math.log(price / prices[index]));
}

/**
 * Simulates future prices by resampling blocks of recent adjusted log returns.
 * Sampling in blocks preserves some short-term volatility clustering, and
 * avoids assuming returns are normally distributed.
 *
 * With `driftMode: "historical"` the simulation inherits whatever average drift
 * the lookback window contains, which for a strong bull run bakes a meaningful
 * upward tilt into the probabilities. `driftMode: "zero"` removes the mean so
 * the result reflects volatility alone.
 */
export function buildMonteCarloForecast(
  history: PricePoint[],
  options: MonteCarloOptions,
): MonteCarloResult {
  const {
    horizonDays,
    simulations = MONTE_CARLO_DEFAULTS.simulations,
    lookbackDays = MONTE_CARLO_DEFAULTS.lookbackDays,
    blockSize = MONTE_CARLO_DEFAULTS.blockSize,
    seed = MONTE_CARLO_DEFAULTS.seed,
    thresholdPercent = MONTE_CARLO_DEFAULTS.thresholdPercent,
    driftMode = MONTE_CARLO_DEFAULTS.driftMode,
  } = options;

  assertIntegerInRange(horizonDays, "Horizon days", 1, MAX_HORIZON_DAYS);
  assertIntegerInRange(simulations, "Simulations", 100, 25_000);
  assertIntegerInRange(lookbackDays, "Lookback days", MIN_HISTORY_DAYS, 2_520);
  assertIntegerInRange(blockSize, "Block size", 1, 20);

  if (!Number.isFinite(seed)) {
    throw new Error("Seed must be a finite number.");
  }
  if (
    !Number.isFinite(thresholdPercent) ||
    thresholdPercent <= 0 ||
    thresholdPercent >= 100
  ) {
    throw new Error("Threshold percent must be between 0 and 100.");
  }
  if (driftMode !== "historical" && driftMode !== "zero") {
    throw new Error("Drift mode must be 'historical' or 'zero'.");
  }
  if (history.length < MIN_HISTORY_DAYS) {
    throw new Error(
      `Need at least ${MIN_HISTORY_DAYS} trading days of history to simulate.`,
    );
  }

  const rawReturns = logReturns(history, lookbackDays);
  const meanReturn =
    rawReturns.reduce((sum, value) => sum + value, 0) / rawReturns.length;
  const returns =
    driftMode === "zero"
      ? rawReturns.map((value) => value - meanReturn)
      : rawReturns;

  const variance =
    rawReturns.reduce((sum, value) => sum + (value - meanReturn) ** 2, 0) /
    Math.max(1, rawReturns.length - 1);

  const currentPrice = history[history.length - 1]?.close;
  if (!currentPrice || !Number.isFinite(currentPrice) || currentPrice <= 0) {
    throw new Error("Current close price is unavailable.");
  }

  const effectiveBlockSize = Math.min(blockSize, returns.length);
  const blockStarts = returns.length - effectiveBlockSize + 1;
  const random = seededRandom(Math.trunc(seed));

  const pathsByDay = Array.from(
    { length: horizonDays + 1 },
    () => new Float64Array(simulations),
  );
  pathsByDay[0].fill(currentPrice);
  const terminalReturns = new Float64Array(simulations);

  for (let simulation = 0; simulation < simulations; simulation++) {
    let price = currentPrice;
    let day = 1;

    while (day <= horizonDays) {
      const blockStart = Math.floor(random() * blockStarts);
      for (
        let offset = 0;
        offset < effectiveBlockSize && day <= horizonDays;
        offset++, day++
      ) {
        price *= Math.exp(returns[blockStart + offset]);
        pathsByDay[day][simulation] = price;
      }
    }

    terminalReturns[simulation] = price / currentPrice - 1;
  }

  const percentiles = pathsByDay.map((prices, day) => {
    prices.sort();
    return {
      day,
      p10: quantile(prices, 0.1),
      p25: quantile(prices, 0.25),
      median: quantile(prices, 0.5),
      p75: quantile(prices, 0.75),
      p90: quantile(prices, 0.9),
    };
  });

  terminalReturns.sort();
  const threshold = thresholdPercent / 100;
  let upCount = 0;
  let downCount = 0;
  for (const value of terminalReturns) {
    if (value >= threshold) upCount++;
    else if (value <= -threshold) downCount++;
  }

  const terminal = percentiles[percentiles.length - 1];
  const simulatedDrift = driftMode === "zero" ? 0 : meanReturn;

  return {
    currentPrice,
    horizonDays,
    thresholdPercent,
    probabilityUp: upCount / simulations,
    probabilityDown: downCount / simulations,
    probabilityWithin: (simulations - upCount - downCount) / simulations,
    medianReturn: quantile(terminalReturns, 0.5),
    terminalP10: terminal.p10,
    terminalP90: terminal.p90,
    percentiles,
    assumptions: {
      annualizedDrift:
        Math.exp(simulatedDrift * TRADING_DAYS_PER_YEAR) - 1,
      annualizedVolatility: Math.sqrt(variance * TRADING_DAYS_PER_YEAR),
      driftMode,
      lookbackDays: Math.min(lookbackDays, rawReturns.length),
      blockSize: effectiveBlockSize,
      simulations,
      seed: Math.trunc(seed),
    },
  };
}
