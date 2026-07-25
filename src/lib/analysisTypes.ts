import type { CalibrationResult } from "./calibration";
import type { ForecastResult } from "./forecast";
import type { MonteCarloResult } from "./monteCarlo";

/** Fixed so a given URL always reproduces the same simulation. */
export const ANALYSIS_SEED = 42;

export type ProbabilityPayload = MonteCarloResult & { ticker: string };

export type ForecastPayload = ForecastResult & {
  ticker: string;
  years: number;
  rawTail: ForecastResult["history"];
};

export type CalibrationPayload = CalibrationResult & { ticker: string };

export type Analysis = {
  probability?: ProbabilityPayload;
  forecast?: ForecastPayload;
  calibration?: CalibrationPayload;
  error?: string;
  calibrationError?: string;
};
