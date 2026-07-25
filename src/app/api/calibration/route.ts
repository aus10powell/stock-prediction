import { NextRequest, NextResponse } from "next/server";
import {
  cacheHeaders,
  driftParam,
  enforceRateLimit,
  errorResponse,
  horizonParam,
  seedParam,
  thresholdParam,
  tickerParam,
} from "@/lib/apiSupport";
import { runCalibrationBacktest } from "@/lib/calibration";
import { loadStockHistory, normalizeTicker } from "@/lib/stocks";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function GET(request: NextRequest) {
  const limited = enforceRateLimit(request);
  if (limited) return limited;

  try {
    const params = request.nextUrl.searchParams;
    const ticker = normalizeTicker(tickerParam(params));
    const horizonDays = horizonParam(params);
    const thresholdPercent = thresholdParam(params);
    const driftMode = driftParam(params);
    const seed = seedParam(params);

    const history = await loadStockHistory(ticker);
    const calibration = runCalibrationBacktest(history, {
      horizonDays,
      thresholdPercent,
      driftMode,
      seed,
    });

    return NextResponse.json(
      { ticker, ...calibration },
      { headers: cacheHeaders(900) },
    );
  } catch (error) {
    return errorResponse(error, "Unable to run the calibration backtest.");
  }
}
