import { NextResponse } from "next/server";
import { MAX_HORIZON_DAYS, MONTE_CARLO_DEFAULTS, type DriftMode } from "./monteCarlo";
import { clientKey, createRateLimiter } from "./rateLimit";

export class RequestError extends Error {
  constructor(
    message: string,
    readonly status = 400,
  ) {
    super(message);
  }
}

const limiter = createRateLimiter(30, 0.5);

export function enforceRateLimit(request: Request): NextResponse | null {
  const verdict = limiter(clientKey(request));
  if (verdict.allowed) return null;

  return NextResponse.json(
    { error: "Too many requests. Please slow down and retry shortly." },
    {
      status: 429,
      headers: { "retry-after": String(verdict.retryAfterSeconds) },
    },
  );
}

export function errorResponse(error: unknown, fallback: string): NextResponse {
  if (error instanceof RequestError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  const message = error instanceof Error ? error.message : fallback;
  return NextResponse.json({ error: message }, { status: 400 });
}

export function cacheHeaders(seconds: number): Record<string, string> {
  return {
    "cache-control": `public, s-maxage=${seconds}, stale-while-revalidate=${seconds * 2}`,
  };
}

function integerParam(
  params: URLSearchParams,
  name: string,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const raw = params.get(name);
  if (raw === null || raw === "") return fallback;

  const value = Number(raw);
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RequestError(
      `${name} must be a whole number between ${minimum} and ${maximum}.`,
    );
  }
  return value;
}

export function horizonParam(params: URLSearchParams): number {
  return integerParam(params, "days", 20, 1, MAX_HORIZON_DAYS);
}

export function yearsParam(params: URLSearchParams): number {
  return integerParam(params, "years", 1, 1, 4);
}

export function seedParam(params: URLSearchParams): number {
  return integerParam(params, "seed", MONTE_CARLO_DEFAULTS.seed, 0, 1_000_000);
}

export function thresholdParam(params: URLSearchParams): number {
  const raw = params.get("threshold");
  if (raw === null || raw === "") return MONTE_CARLO_DEFAULTS.thresholdPercent;

  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0 || value > 50) {
    throw new RequestError(
      "threshold must be a percentage greater than 0 and no more than 50.",
    );
  }
  return value;
}

export function driftParam(params: URLSearchParams): DriftMode {
  const raw = params.get("drift") ?? MONTE_CARLO_DEFAULTS.driftMode;
  if (raw !== "historical" && raw !== "zero") {
    throw new RequestError("drift must be either 'historical' or 'zero'.");
  }
  return raw;
}

export function tickerParam(params: URLSearchParams, fallback = "GME"): string {
  return params.get("ticker")?.trim() || fallback;
}

export function tickerListParam(
  params: URLSearchParams,
  maximum = 12,
): string[] {
  const raw = params.get("tickers") ?? "";
  const symbols = [
    ...new Set(
      raw
        .split(",")
        .map((symbol) => symbol.trim().toUpperCase())
        .filter(Boolean),
    ),
  ];

  if (symbols.length === 0) {
    throw new RequestError("Provide at least one ticker.");
  }
  if (symbols.length > maximum) {
    throw new RequestError(`Provide at most ${maximum} tickers.`);
  }
  return symbols;
}
