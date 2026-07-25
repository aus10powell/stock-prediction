import type { DriftMode } from "./monteCarlo";

export type Settings = {
  ticker: string;
  days: number;
  threshold: number;
  drift: DriftMode;
  years: number;
};

export const DEFAULT_SETTINGS: Settings = {
  ticker: "GME",
  days: 20,
  threshold: 1,
  drift: "historical",
  years: 1,
};

export type RawSearchParams = Record<string, string | string[] | undefined>;

function first(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function integer(
  value: string | string[] | undefined,
  min: number,
  max: number,
  fallback: number,
): number {
  const parsed = Number(first(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, Math.round(parsed)));
}

/** Tolerant parser: bad query values fall back to defaults rather than erroring. */
export function parseSettings(params: RawSearchParams): Settings {
  const ticker = first(params.ticker)?.trim().toUpperCase().slice(0, 10);
  const threshold = Number(first(params.threshold));

  return {
    ticker: ticker && /^[A-Z.\-]{1,10}$/.test(ticker)
      ? ticker
      : DEFAULT_SETTINGS.ticker,
    days: integer(params.days, 1, 252, DEFAULT_SETTINGS.days),
    threshold:
      Number.isFinite(threshold) && threshold > 0 && threshold <= 50
        ? threshold
        : DEFAULT_SETTINGS.threshold,
    drift: first(params.drift) === "zero" ? "zero" : DEFAULT_SETTINGS.drift,
    years: integer(params.years, 1, 4, DEFAULT_SETTINGS.years),
  };
}

export function settingsToQuery(settings: Settings): string {
  return new URLSearchParams({
    ticker: settings.ticker,
    days: String(settings.days),
    threshold: String(settings.threshold),
    drift: settings.drift,
    years: String(settings.years),
  }).toString();
}
