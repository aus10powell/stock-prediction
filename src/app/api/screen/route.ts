import { NextRequest, NextResponse } from "next/server";
import {
  cacheHeaders,
  driftParam,
  enforceRateLimit,
  errorResponse,
  horizonParam,
  seedParam,
  thresholdParam,
  tickerListParam,
} from "@/lib/apiSupport";
import { buildMonteCarloForecast } from "@/lib/monteCarlo";
import { loadStockHistory, normalizeTicker } from "@/lib/stocks";

export const runtime = "nodejs";
export const maxDuration = 60;

const CONCURRENCY = 4;
/** Fewer paths than a single-ticker run so a full watchlist stays responsive. */
const SIMULATIONS = 4_000;

export type ScreenRow = {
  ticker: string;
  currentPrice?: number;
  probabilityUp?: number;
  probabilityDown?: number;
  probabilityWithin?: number;
  medianReturn?: number;
  annualizedVolatility?: number;
  error?: string;
};

export async function GET(request: NextRequest) {
  const limited = enforceRateLimit(request);
  if (limited) return limited;

  try {
    const params = request.nextUrl.searchParams;
    const tickers = tickerListParam(params);
    const horizonDays = horizonParam(params);
    const thresholdPercent = thresholdParam(params);
    const driftMode = driftParam(params);
    const seed = seedParam(params);

    const rows: ScreenRow[] = new Array(tickers.length);
    let cursor = 0;

    async function worker() {
      while (cursor < tickers.length) {
        const index = cursor++;
        const requested = tickers[index];
        try {
          const ticker = normalizeTicker(requested);
          const history = await loadStockHistory(ticker);
          const simulation = buildMonteCarloForecast(history, {
            horizonDays,
            thresholdPercent,
            driftMode,
            seed,
            simulations: SIMULATIONS,
          });
          rows[index] = {
            ticker,
            currentPrice: simulation.currentPrice,
            probabilityUp: simulation.probabilityUp,
            probabilityDown: simulation.probabilityDown,
            probabilityWithin: simulation.probabilityWithin,
            medianReturn: simulation.medianReturn,
            annualizedVolatility: simulation.assumptions.annualizedVolatility,
          };
        } catch (error) {
          rows[index] = {
            ticker: requested,
            error:
              error instanceof Error ? error.message : "Simulation failed.",
          };
        }
      }
    }

    await Promise.all(
      Array.from({ length: Math.min(CONCURRENCY, tickers.length) }, worker),
    );

    return NextResponse.json(
      {
        horizonDays,
        thresholdPercent,
        driftMode,
        simulations: SIMULATIONS,
        rows,
      },
      { headers: cacheHeaders(300) },
    );
  } catch (error) {
    return errorResponse(error, "Unable to screen the watchlist.");
  }
}
