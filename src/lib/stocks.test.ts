import { beforeEach, describe, expect, it, vi } from "vitest";

const chart = vi.hoisted(() => vi.fn());

vi.mock("yahoo-finance2", () => ({
  default: class {
    chart = chart;
  },
}));

const { fetchStockHistory, loadStockHistory, normalizeTicker } = await import(
  "./stocks"
);

function quote(date: string, close: number, adjclose?: number | null) {
  return { date: new Date(`${date}T00:00:00.000Z`), open: close - 1, close, adjclose };
}

beforeEach(() => {
  chart.mockReset();
});

describe("normalizeTicker", () => {
  it("uppercases and trims valid symbols", () => {
    expect(normalizeTicker(" aapl ")).toBe("AAPL");
    expect(normalizeTicker("brk-b")).toBe("BRK-B");
  });

  it("rejects symbols that cannot be tickers", () => {
    for (const invalid of ["", "   ", "TOOLONGSYMBOL", "AA PL", "A$PL"]) {
      expect(() => normalizeTicker(invalid)).toThrow("valid ticker symbol");
    }
  });
});

describe("fetchStockHistory", () => {
  it("maps adjusted closes and drops incomplete rows", async () => {
    chart.mockResolvedValue({
      quotes: [
        quote("2024-01-02", 100, 95),
        quote("2024-01-03", 101, null),
        { date: new Date("2024-01-04T00:00:00.000Z"), open: null, close: 102 },
        { date: new Date("2024-01-05T00:00:00.000Z"), open: 102, close: null },
      ],
    });

    const history = await fetchStockHistory("aapl");

    expect(history).toEqual([
      { date: "2024-01-02", open: 99, close: 100, adjustedClose: 95 },
      { date: "2024-01-03", open: 100, close: 101, adjustedClose: undefined },
    ]);
    expect(chart).toHaveBeenCalledWith(
      "AAPL",
      expect.objectContaining({ interval: "1d", period1: "2015-01-01" }),
    );
  });

  it("throws when the provider returns nothing usable", async () => {
    chart.mockResolvedValue({ quotes: [] });
    await expect(fetchStockHistory("AAPL")).rejects.toThrow(
      "No price history found for AAPL.",
    );
  });

  it("validates before calling the provider", async () => {
    await expect(fetchStockHistory("nope!")).rejects.toThrow(
      "valid ticker symbol",
    );
    expect(chart).not.toHaveBeenCalled();
  });
});

describe("loadStockHistory", () => {
  it("caches repeated requests for the same symbol", async () => {
    chart.mockResolvedValue({ quotes: [quote("2024-01-02", 100, 100)] });

    await loadStockHistory("MSFT");
    await loadStockHistory("msft");

    expect(chart).toHaveBeenCalledTimes(1);
  });
});
