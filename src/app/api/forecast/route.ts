import { NextRequest, NextResponse } from "next/server";
import { buildForecast } from "@/lib/forecast";
import { loadStockHistory } from "@/lib/stocks";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function GET(request: NextRequest) {
  const ticker = request.nextUrl.searchParams.get("ticker") ?? "GME";
  const years = Number(request.nextUrl.searchParams.get("years") ?? "1");

  if (!Number.isFinite(years) || years < 1 || years > 4) {
    return NextResponse.json(
      { error: "Years must be between 1 and 4." },
      { status: 400 },
    );
  }

  try {
    const history = await loadStockHistory(ticker);
    const periods = Math.round(years * 365);
    const result = buildForecast(history, periods);

    return NextResponse.json({
      ticker: ticker.trim().toUpperCase(),
      years,
      ...result,
      rawTail: history.slice(-8),
      forecastTail: result.forecast.slice(-8),
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unable to build forecast.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
