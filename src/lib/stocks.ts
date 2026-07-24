import YahooFinance from "yahoo-finance2";
import type { PricePoint } from "./forecast";

const yahooFinance = new YahooFinance({
  suppressNotices: ["ripHistorical"],
});

const START = "2015-01-01";

export async function loadStockHistory(ticker: string): Promise<PricePoint[]> {
  const symbol = ticker.trim().toUpperCase();
  if (!/^[A-Z.\-]{1,10}$/.test(symbol)) {
    throw new Error("Enter a valid ticker symbol (e.g. GME, AAPL, TSLA).");
  }

  const today = new Date();
  const rows = await yahooFinance.chart(symbol, {
    period1: START,
    period2: today,
    interval: "1d",
  });

  const quotes = rows.quotes ?? [];
  const history: PricePoint[] = quotes
    .filter(
      (q) =>
        q.date &&
        typeof q.open === "number" &&
        typeof q.close === "number" &&
        Number.isFinite(q.open) &&
        Number.isFinite(q.close),
    )
    .map((q) => ({
      date: new Date(q.date).toISOString().slice(0, 10),
      open: q.open as number,
      close: q.close as number,
      adjustedClose:
        typeof q.adjclose === "number" && Number.isFinite(q.adjclose)
          ? q.adjclose
          : undefined,
    }));

  if (history.length === 0) {
    throw new Error(`No price history found for ${symbol}.`);
  }

  return history;
}
