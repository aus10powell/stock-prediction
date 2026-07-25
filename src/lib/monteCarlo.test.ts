import { describe, expect, it } from "vitest";
import type { PricePoint } from "./forecast";
import { buildMonteCarloForecast } from "./monteCarlo";

function historyFromReturns(returns: number[]): PricePoint[] {
  let adjustedClose = 100;
  return [
    { date: "2025-01-01", open: 100, close: 100, adjustedClose },
    ...returns.map((value, index) => {
      adjustedClose *= Math.exp(value);
      return {
        date: `2025-01-${String(index + 2).padStart(2, "0")}`,
        open: adjustedClose,
        close: adjustedClose,
        adjustedClose,
      };
    }),
  ];
}

const variedHistory = historyFromReturns(
  Array.from({ length: 80 }, (_, index) => {
    return Math.sin(index * 1.7) * 0.018 + (index % 9 === 0 ? -0.012 : 0.001);
  }),
);

describe("buildMonteCarloForecast", () => {
  it("returns exhaustive threshold probabilities and ordered percentiles", () => {
    const result = buildMonteCarloForecast(variedHistory, {
      horizonDays: 20,
      simulations: 1_000,
      seed: 123,
    });

    expect(result.percentiles).toHaveLength(21);
    expect(
      result.probabilityUp + result.probabilityDown + result.probabilityWithin,
    ).toBeCloseTo(1, 12);
    expect(result.percentiles[0]).toEqual({
      day: 0,
      p10: result.currentPrice,
      p25: result.currentPrice,
      median: result.currentPrice,
      p75: result.currentPrice,
      p90: result.currentPrice,
    });
    for (const point of result.percentiles) {
      expect(point.p10).toBeLessThanOrEqual(point.p25);
      expect(point.p25).toBeLessThanOrEqual(point.median);
      expect(point.median).toBeLessThanOrEqual(point.p75);
      expect(point.p75).toBeLessThanOrEqual(point.p90);
    }
  });

  it("is reproducible for the same seed", () => {
    const options = { horizonDays: 10, simulations: 500, seed: 7 };
    expect(buildMonteCarloForecast(variedHistory, options)).toEqual(
      buildMonteCarloForecast(variedHistory, options),
    );
  });

  it("honours the requested threshold", () => {
    const options = { horizonDays: 20, simulations: 2_000, seed: 5 };
    const onePercent = buildMonteCarloForecast(variedHistory, {
      ...options,
      thresholdPercent: 1,
    });
    const tenPercent = buildMonteCarloForecast(variedHistory, {
      ...options,
      thresholdPercent: 10,
    });

    expect(tenPercent.probabilityUp).toBeLessThan(onePercent.probabilityUp);
    expect(tenPercent.probabilityWithin).toBeGreaterThan(
      onePercent.probabilityWithin,
    );
    expect(tenPercent.thresholdPercent).toBe(10);
  });

  it("removes drift in zero-drift mode and reports the assumption", () => {
    const trending = historyFromReturns(
      Array.from({ length: 300 }, (_, index) =>
        index % 2 === 0 ? 0.014 : 0.006,
      ),
    );

    const historical = buildMonteCarloForecast(trending, {
      horizonDays: 20,
      simulations: 2_000,
      seed: 11,
    });
    const zeroDrift = buildMonteCarloForecast(trending, {
      horizonDays: 20,
      simulations: 2_000,
      seed: 11,
      driftMode: "zero",
    });

    expect(historical.assumptions.annualizedDrift).toBeGreaterThan(1);
    expect(historical.probabilityUp).toBe(1);

    expect(zeroDrift.assumptions.annualizedDrift).toBe(0);
    expect(zeroDrift.medianReturn).toBeCloseTo(0, 6);
    expect(zeroDrift.assumptions.annualizedVolatility).toBeCloseTo(
      historical.assumptions.annualizedVolatility,
      12,
    );
  });

  it("uses adjusted closes to avoid split-driven returns", () => {
    const history = historyFromReturns(Array.from({ length: 40 }, () => 0));
    history[history.length - 1].close = 50;

    const result = buildMonteCarloForecast(history, {
      horizonDays: 20,
      simulations: 500,
    });

    expect(result.currentPrice).toBe(50);
    expect(result.probabilityWithin).toBe(1);
    expect(result.medianReturn).toBe(0);
  });

  it("rejects unsupported inputs", () => {
    expect(() =>
      buildMonteCarloForecast(variedHistory, { horizonDays: 0 }),
    ).toThrow("Horizon days must be an integer between 1 and 252.");
    expect(() =>
      buildMonteCarloForecast(variedHistory.slice(0, 20), { horizonDays: 5 }),
    ).toThrow("at least 30 trading days");
    expect(() =>
      buildMonteCarloForecast(variedHistory, {
        horizonDays: 5,
        thresholdPercent: 0,
      }),
    ).toThrow("Threshold percent");
  });
});
