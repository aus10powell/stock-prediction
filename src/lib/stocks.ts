import YahooFinance from "yahoo-finance2";
import { createTtlCache } from "./cache";
import type { PricePoint } from "./forecast";

const yahooFinance = new YahooFinance({
  suppressNotices: ["ripHistorical"],
});

const START = "2015-01-01";
const TICKER_PATTERN = /^[A-Z.\-]{1,10}$/;
export const HISTORY_TTL_MS = 15 * 60 * 1000;

export function normalizeTicker(ticker: string): string {
  const symbol = ticker.trim().toUpperCase();
  if (!TICKER_PATTERN.test(symbol)) {
    throw new Error("Enter a valid ticker symbol (e.g. GME, AAPL, TSLA).");
  }
  return symbol;
}

export async function fetchStockHistory(ticker: string): Promise<PricePoint[]> {
  const symbol = normalizeTicker(ticker);

  const rows = await yahooFinance.chart(symbol, {
    period1: START,
    period2: new Date(),
    interval: "1d",
  });

  const history: PricePoint[] = (rows.quotes ?? [])
    .filter(
      (quote) =>
        quote.date &&
        typeof quote.open === "number" &&
        typeof quote.close === "number" &&
        Number.isFinite(quote.open) &&
        Number.isFinite(quote.close),
    )
    .map((quote) => ({
      date: new Date(quote.date).toISOString().slice(0, 10),
      open: quote.open as number,
      close: quote.close as number,
      adjustedClose:
        typeof quote.adjclose === "number" && Number.isFinite(quote.adjclose)
          ? quote.adjclose
          : undefined,
    }));

  if (history.length === 0) {
    throw new Error(`No price history found for ${symbol}.`);
  }

  return history;
}

const withCache = createTtlCache<PricePoint[]>(HISTORY_TTL_MS);

/** Cached per warm instance; daily bars do not change within the TTL. */
export function loadStockHistory(ticker: string): Promise<PricePoint[]> {
  const symbol = normalizeTicker(ticker);
  return withCache(symbol, () => fetchStockHistory(symbol));
}
