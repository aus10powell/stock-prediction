import type { PricePoint } from "./forecast";
import {
  buildMonteCarloForecast,
  MAX_HORIZON_DAYS,
  MIN_HISTORY_DAYS,
  MONTE_CARLO_DEFAULTS,
  type DriftMode,
} from "./monteCarlo";

export type ReliabilityBin = {
  lowerBound: number;
  upperBound: number;
  count: number;
  meanPredicted: number;
  observedFrequency: number;
};

export type CalibrationOutcome = {
  /** Mean probability the model assigned across all evaluated origins. */
  meanPredicted: number;
  /** How often the move actually happened. */
  observedFrequency: number;
  brierScore: number;
  /** Brier score of always predicting the historical base rate. */
  baselineBrierScore: number;
  /** Positive means the model beat that base-rate baseline. */
  skillScore: number;
  bins: ReliabilityBin[];
};

export type CalibrationResult = {
  horizonDays: number;
  thresholdPercent: number;
  driftMode: DriftMode;
  samples: number;
  simulationsPerSample: number;
  firstOrigin: string;
  lastOrigin: string;
  up: CalibrationOutcome;
  down: CalibrationOutcome;
};

export type CalibrationOptions = {
  horizonDays: number;
  thresholdPercent?: number;
  driftMode?: DriftMode;
  lookbackDays?: number;
  blockSize?: number;
  seed?: number;
  /** Simulations per evaluated origin; lower than a one-off forecast for speed. */
  simulationsPerSample?: number;
  /** Upper bound on evaluated origins, controlling runtime. */
  maxSamples?: number;
};

const DEFAULT_SIMULATIONS_PER_SAMPLE = 400;
const DEFAULT_MAX_SAMPLES = 220;
const BIN_COUNT = 10;

function summarize(
  predictions: number[],
  outcomes: number[],
): CalibrationOutcome {
  const count = predictions.length;
  const meanPredicted =
    predictions.reduce((sum, value) => sum + value, 0) / count;
  const observedFrequency =
    outcomes.reduce((sum, value) => sum + value, 0) / count;

  const brierScore =
    predictions.reduce(
      (sum, value, index) => sum + (value - outcomes[index]) ** 2,
      0,
    ) / count;
  const baselineBrierScore =
    outcomes.reduce(
      (sum, value) => sum + (observedFrequency - value) ** 2,
      0,
    ) / count;

  const bins: ReliabilityBin[] = Array.from({ length: BIN_COUNT }, (_, i) => ({
    lowerBound: i / BIN_COUNT,
    upperBound: (i + 1) / BIN_COUNT,
    count: 0,
    meanPredicted: 0,
    observedFrequency: 0,
  }));

  predictions.forEach((prediction, index) => {
    const slot = Math.min(BIN_COUNT - 1, Math.floor(prediction * BIN_COUNT));
    const bin = bins[slot];
    bin.count += 1;
    bin.meanPredicted += prediction;
    bin.observedFrequency += outcomes[index];
  });

  for (const bin of bins) {
    if (bin.count > 0) {
      bin.meanPredicted /= bin.count;
      bin.observedFrequency /= bin.count;
    }
  }

  return {
    meanPredicted,
    observedFrequency,
    brierScore,
    baselineBrierScore,
    skillScore:
      baselineBrierScore === 0 ? 0 : 1 - brierScore / baselineBrierScore,
    bins: bins.filter((bin) => bin.count > 0),
  };
}

/**
 * Walk-forward check of whether the simulated probabilities mean anything: at
 * many historical origins, simulate using only data available at that point,
 * then compare the predicted probability against what actually happened over
 * the following horizon.
 *
 * A well-calibrated model puts roughly 40% of its "40% likely" calls in the
 * column where the move occurred, which is what the reliability bins show.
 */
export function runCalibrationBacktest(
  history: PricePoint[],
  options: CalibrationOptions,
): CalibrationResult {
  const {
    horizonDays,
    thresholdPercent = MONTE_CARLO_DEFAULTS.thresholdPercent,
    driftMode = MONTE_CARLO_DEFAULTS.driftMode,
    lookbackDays = MONTE_CARLO_DEFAULTS.lookbackDays,
    blockSize = MONTE_CARLO_DEFAULTS.blockSize,
    seed = MONTE_CARLO_DEFAULTS.seed,
    simulationsPerSample = DEFAULT_SIMULATIONS_PER_SAMPLE,
    maxSamples = DEFAULT_MAX_SAMPLES,
  } = options;

  if (!Number.isInteger(horizonDays) || horizonDays < 1 || horizonDays > MAX_HORIZON_DAYS) {
    throw new Error(
      `Horizon days must be an integer between 1 and ${MAX_HORIZON_DAYS}.`,
    );
  }

  const minTrainingDays = Math.max(MIN_HISTORY_DAYS, Math.min(lookbackDays, 252));
  const firstOrigin = minTrainingDays - 1;
  const lastOrigin = history.length - 1 - horizonDays;
  const availableOrigins = lastOrigin - firstOrigin + 1;

  if (availableOrigins < 20) {
    throw new Error(
      "Not enough history to backtest this horizon. Try a shorter horizon.",
    );
  }

  const stride = Math.max(1, Math.ceil(availableOrigins / maxSamples));
  const adjusted = history.map((point) => point.adjustedClose ?? point.close);

  const upPredictions: number[] = [];
  const upOutcomes: number[] = [];
  const downPredictions: number[] = [];
  const downOutcomes: number[] = [];
  const originDates: string[] = [];

  for (let origin = firstOrigin; origin <= lastOrigin; origin += stride) {
    const training = history.slice(0, origin + 1);
    const simulation = buildMonteCarloForecast(training, {
      horizonDays,
      simulations: simulationsPerSample,
      lookbackDays: Math.min(lookbackDays, training.length - 1),
      blockSize,
      seed: seed + origin,
      thresholdPercent,
      driftMode,
    });

    const realizedReturn = adjusted[origin + horizonDays] / adjusted[origin] - 1;
    const threshold = thresholdPercent / 100;

    upPredictions.push(simulation.probabilityUp);
    upOutcomes.push(realizedReturn >= threshold ? 1 : 0);
    downPredictions.push(simulation.probabilityDown);
    downOutcomes.push(realizedReturn <= -threshold ? 1 : 0);
    originDates.push(history[origin].date);
  }

  return {
    horizonDays,
    thresholdPercent,
    driftMode,
    samples: originDates.length,
    simulationsPerSample,
    firstOrigin: originDates[0],
    lastOrigin: originDates[originDates.length - 1],
    up: summarize(upPredictions, upOutcomes),
    down: summarize(downPredictions, downOutcomes),
  };
}
