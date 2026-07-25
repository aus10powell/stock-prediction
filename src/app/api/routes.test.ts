import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";
import type { PricePoint } from "@/lib/forecast";

const loadStockHistory = vi.hoisted(() => vi.fn());

vi.mock("@/lib/stocks", async () => {
  const actual = await import("@/lib/stocks");
  return { ...actual, loadStockHistory };
});

const { GET: getProbability } = await import("./probability/route");
const { GET: getForecast } = await import("./forecast/route");
const { GET: getScreen } = await import("./screen/route");

function syntheticHistory(days: number): PricePoint[] {
  let price = 100;
  const points: PricePoint[] = [];
  for (let i = 0; i < days; i++) {
    price *= Math.exp(Math.sin(i * 0.7) * 0.015 + 0.0004);
    points.push({
      date: new Date(Date.UTC(2018, 0, 1 + i)).toISOString().slice(0, 10),
      open: price,
      close: price,
      adjustedClose: price,
    });
  }
  return points;
}

const history = syntheticHistory(700);

function call(path: string, query: Record<string, string> = {}) {
  const url = new URL(`http://localhost${path}`);
  Object.entries(query).forEach(([key, value]) =>
    url.searchParams.set(key, value),
  );
  return new NextRequest(url);
}

beforeEach(() => {
  loadStockHistory.mockReset();
  loadStockHistory.mockResolvedValue(history);
});

describe("GET /api/probability", () => {
  it("returns probabilities for the requested threshold and horizon", async () => {
    const response = await getProbability(
      call("/api/probability", {
        ticker: "aapl",
        days: "15",
        threshold: "2.5",
        drift: "zero",
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(response.headers.get("cache-control")).toContain("s-maxage");
    expect(body.ticker).toBe("AAPL");
    expect(body.horizonDays).toBe(15);
    expect(body.thresholdPercent).toBe(2.5);
    expect(body.assumptions.driftMode).toBe("zero");
    expect(body.percentiles).toHaveLength(16);
    expect(
      body.probabilityUp + body.probabilityDown + body.probabilityWithin,
    ).toBeCloseTo(1, 12);
  });

  it.each([
    [{ days: "0" }, "days must be a whole number"],
    [{ days: "300" }, "days must be a whole number"],
    [{ days: "7.5" }, "days must be a whole number"],
    [{ threshold: "0" }, "threshold must be a percentage"],
    [{ threshold: "80" }, "threshold must be a percentage"],
    [{ drift: "sideways" }, "drift must be either"],
    [{ ticker: "not a ticker" }, "valid ticker symbol"],
  ])("rejects invalid input %j", async (query, message) => {
    const response = await getProbability(call("/api/probability", query));
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.error).toContain(message);
  });

  it("surfaces upstream failures as a client error", async () => {
    loadStockHistory.mockRejectedValue(new Error("No price history found."));
    const response = await getProbability(call("/api/probability"));

    expect(response.status).toBe(400);
    expect((await response.json()).error).toBe("No price history found.");
  });
});

describe("GET /api/forecast", () => {
  it("extends the fit by whole trading years", async () => {
    const response = await getForecast(
      call("/api/forecast", { ticker: "GME", years: "2" }),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.years).toBe(2);
    expect(body.fit.tradingDaysAhead).toBe(504);
    expect(body.forecast).toHaveLength(history.length + 504);
    expect(body.rawTail).toHaveLength(8);
  });

  it("rejects an out-of-range fit window", async () => {
    const response = await getForecast(call("/api/forecast", { years: "9" }));
    expect(response.status).toBe(400);
    expect((await response.json()).error).toContain("years must be");
  });
});

describe("GET /api/screen", () => {
  it("simulates each requested ticker", async () => {
    const response = await getScreen(
      call("/api/screen", { tickers: "aapl, msft ,aapl", days: "10" }),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.rows.map((row: { ticker: string }) => row.ticker)).toEqual([
      "AAPL",
      "MSFT",
    ]);
    for (const row of body.rows) {
      expect(row.error).toBeUndefined();
      expect(row.probabilityUp).toBeGreaterThanOrEqual(0);
    }
  });

  it("reports per-ticker failures without failing the request", async () => {
    loadStockHistory.mockImplementation(async (ticker: string) => {
      if (ticker === "BAD") throw new Error("No price history found for BAD.");
      return history;
    });

    const response = await getScreen(
      call("/api/screen", { tickers: "AAPL,BAD" }),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.rows[0].error).toBeUndefined();
    expect(body.rows[1].error).toContain("BAD");
  });

  it("requires a non-empty, bounded ticker list", async () => {
    const empty = await getScreen(call("/api/screen", { tickers: " , " }));
    expect(empty.status).toBe(400);
    expect((await empty.json()).error).toContain("at least one ticker");

    const tooMany = await getScreen(
      call("/api/screen", {
        tickers: Array.from({ length: 13 }, (_, i) => `T${i}`).join(","),
      }),
    );
    expect(tooMany.status).toBe(400);
    expect((await tooMany.json()).error).toContain("at most 12");
  });
});
