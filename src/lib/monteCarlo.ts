import type { PricePoint } from "./forecast";

export type MonteCarloPoint = {
  day: number;
  p10: number;
  p25: number;
  median: number;
  p75: number;
  p90: number;
};

export type MonteCarloResult = {
  currentPrice: number;
  horizonDays: number;
  simulations: number;
  lookbackDays: number;
  blockSize: number;
  seed: number;
  thresholdPercent: number;
  probabilityUp: number;
  probabilityDown: number;
  probabilityWithin: number;
  medianReturn: number;
  terminalP10: number;
  terminalP90: number;
  percentiles: MonteCarloPoint[];
};

type MonteCarloOptions = {
  horizonDays: number;
  simulations?: number;
  lookbackDays?: number;
  blockSize?: number;
  seed?: number;
  thresholdPercent?: number;
};

const DEFAULT_SIMULATIONS = 10_000;
const DEFAULT_LOOKBACK_DAYS = 504;
const DEFAULT_BLOCK_SIZE = 5;
const DEFAULT_SEED = 42;
const DEFAULT_THRESHOLD_PERCENT = 1;

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

function percentile(sorted: number[], probability: number): number {
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  const weight = index - lower;
  const upper = sorted[lower + 1];
  return upper === undefined
    ? sorted[lower]
    : sorted[lower] * (1 - weight) + upper * weight;
}

function assertIntegerInRange(
  value: number,
  name: string,
  minimum: number,
  maximum: number,
) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}.`);
  }
}

/**
 * Simulates future prices by sampling five-day blocks of recent adjusted
 * log returns. Block sampling retains some short-term volatility clustering
 * without assuming returns follow a normal distribution.
 */
export function buildMonteCarloForecast(
  history: PricePoint[],
  options: MonteCarloOptions,
): MonteCarloResult {
  const {
    horizonDays,
    simulations = DEFAULT_SIMULATIONS,
    lookbackDays = DEFAULT_LOOKBACK_DAYS,
    blockSize = DEFAULT_BLOCK_SIZE,
    seed = DEFAULT_SEED,
    thresholdPercent = DEFAULT_THRESHOLD_PERCENT,
  } = options;

  assertIntegerInRange(horizonDays, "Horizon days", 1, 252);
  assertIntegerInRange(simulations, "Simulations", 100, 25_000);
  assertIntegerInRange(lookbackDays, "Lookback days", 30, 2_520);
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
  if (history.length < 30) {
    throw new Error("Need at least 30 trading days of history to simulate.");
  }

  const recent = history.slice(-(lookbackDays + 1));
  const adjustedPrices = recent.map((point) => point.adjustedClose ?? point.close);
  if (adjustedPrices.some((price) => !Number.isFinite(price) || price <= 0)) {
    throw new Error("Price history contains invalid adjusted close values.");
  }

  const returns = adjustedPrices.slice(1).map((price, index) => {
    return Math.log(price / adjustedPrices[index]);
  });
  const effectiveBlockSize = Math.min(blockSize, returns.length);
  const lastBlockStart = returns.length - effectiveBlockSize;
  const currentPrice = history.at(-1)?.close;
  if (!currentPrice || !Number.isFinite(currentPrice) || currentPrice <= 0) {
    throw new Error("Current close price is unavailable.");
  }

  const random = seededRandom(Math.trunc(seed));
  const pricesByDay = Array.from(
    { length: horizonDays + 1 },
    (_, day) => (day === 0 ? [currentPrice] : ([] as number[])),
  );
  const terminalReturns = new Array<number>(simulations);

  for (let simulation = 0; simulation < simulations; simulation++) {
    let price = currentPrice;
    let day = 1;

    while (day <= horizonDays) {
      const blockStart = Math.floor(random() * (lastBlockStart + 1));
      for (
        let offset = 0;
        offset < effectiveBlockSize && day <= horizonDays;
        offset++, day++
      ) {
        price *= Math.exp(returns[blockStart + offset]);
        pricesByDay[day].push(price);
      }
    }

    terminalReturns[simulation] = price / currentPrice - 1;
  }

  const percentiles = pricesByDay.map((prices, day) => {
    prices.sort((a, b) => a - b);
    return {
      day,
      p10: percentile(prices, 0.1),
      p25: percentile(prices, 0.25),
      median: percentile(prices, 0.5),
      p75: percentile(prices, 0.75),
      p90: percentile(prices, 0.9),
    };
  });

  terminalReturns.sort((a, b) => a - b);
  const threshold = thresholdPercent / 100;
  const upCount = terminalReturns.filter((value) => value >= threshold).length;
  const downCount = terminalReturns.filter((value) => value <= -threshold).length;
  const terminal = percentiles.at(-1)!;

  return {
    currentPrice,
    horizonDays,
    simulations,
    lookbackDays: Math.min(lookbackDays, returns.length),
    blockSize: effectiveBlockSize,
    seed: Math.trunc(seed),
    thresholdPercent,
    probabilityUp: upCount / simulations,
    probabilityDown: downCount / simulations,
    probabilityWithin: (simulations - upCount - downCount) / simulations,
    medianReturn: percentile(terminalReturns, 0.5),
    terminalP10: terminal.p10,
    terminalP90: terminal.p90,
    percentiles,
  };
}
