import { describe, expect, it } from "vitest";
import { buildForecast, type PricePoint } from "./forecast";
import { isTradingDay, parseDateKey } from "./tradingCalendar";

/** Trading-day series starting 2021-01-04 with a fixed per-day price step. */
function linearHistory(days: number, step: number, start = 100): PricePoint[] {
  const points: PricePoint[] = [];
  let cursor = parseDateKey("2021-01-04");
  let price = start;

  while (points.length < days) {
    if (isTradingDay(cursor)) {
      points.push({
        date: cursor.toISOString().slice(0, 10),
        open: price,
        close: price,
        adjustedClose: price,
      });
      price += step;
    }
    cursor = new Date(cursor.getTime() + 86_400_000);
  }
  return points;
}

describe("buildForecast", () => {
  it("advances the trend one slope step per projected trading day", () => {
    const step = 0.5;
    const history = linearHistory(300, step);
    const horizon = 60;
    const result = buildForecast(history, horizon);

    const projected = result.forecast.slice(history.length);
    expect(projected).toHaveLength(horizon);
    expect(result.fit.slopePerTradingDay).toBeCloseTo(step, 6);

    const lastClose = history[history.length - 1].close;
    // A horizon of N trading days must extrapolate exactly N steps of slope,
    // regardless of how many calendar days those trading days span.
    expect(projected[horizon - 1].yhat).toBeCloseTo(lastClose + step * horizon, 4);
    expect(projected[0].yhat).toBeCloseTo(lastClose + step, 4);
  });

  it("projects onto trading days only", () => {
    const history = linearHistory(120, 0.25);
    const result = buildForecast(history, 30);

    const projectedDates = result.forecast
      .slice(history.length)
      .map((point) => parseDateKey(point.date));

    expect(projectedDates.every(isTradingDay)).toBe(true);
    for (let i = 1; i < projectedDates.length; i++) {
      expect(projectedDates[i].getTime()).toBeGreaterThan(
        projectedDates[i - 1].getTime(),
      );
    }
  });

  it("widens the prediction interval roughly with the square root of horizon", () => {
    // Persistent deviations from trend, so the accumulating term dominates.
    const base = linearHistory(600, 0.2);
    let wander = 0;
    const history = base.map((point, index) => {
      wander += Math.sin(index * 0.05) * 0.5;
      return { ...point, close: point.close + wander };
    });

    const result = buildForecast(history, 200);
    const projected = result.forecast.slice(history.length);
    const width = (index: number) =>
      projected[index].yhatUpper - projected[index].yhatLower;

    for (let i = 1; i < projected.length; i++) {
      expect(width(i)).toBeGreaterThan(width(i - 1));
    }

    // Quadrupling the horizon should roughly double the interval width.
    const ratio = width(199) / width(49);
    expect(ratio).toBeGreaterThan(1.5);
    expect(ratio).toBeLessThan(2.5);
    expect(result.fit.residualStepStd).toBeGreaterThan(0);
  });

  it("distinguishes persistent residuals from independent noise", () => {
    const base = linearHistory(400, 0.3);

    const wandering = base.map((point, index) => ({
      ...point,
      close: point.close + Math.sin(index * 0.015) * 8,
    }));
    // Deterministic alternating noise: no persistence from one day to the next.
    const choppy = base.map((point, index) => ({
      ...point,
      close: point.close + (index % 2 === 0 ? 4 : -4),
    }));

    expect(
      buildForecast(wandering, 10).fit.residualAutocorrelation,
    ).toBeGreaterThan(0.8);
    expect(buildForecast(choppy, 10).fit.residualAutocorrelation).toBeLessThan(
      -0.8,
    );
  });

  it("recovers known seasonal amplitude with jointly fitted terms", () => {
    // Correlated sine and cosine terms at the same frequency are only
    // recovered correctly when the design matrix is solved as a whole.
    const base = linearHistory(760, 0);
    const history = base.map((point) => {
      const date = parseDateKey(point.date);
      const dayOfYear = Math.floor(
        (date.getTime() - Date.UTC(date.getUTCFullYear(), 0, 0)) / 86_400_000,
      );
      const t = (2 * Math.PI * dayOfYear) / 365.25;
      const seasonal = 10 * Math.sin(t) + 6 * Math.cos(t);
      return { ...point, close: point.close + seasonal };
    });

    const result = buildForecast(history, 0);
    const amplitudes = result.yearly.map((entry) => entry.value);
    const peak = Math.max(...amplitudes);
    const trough = Math.min(...amplitudes);

    // True amplitude is sqrt(10² + 6²) ≈ 11.66, so peak-to-trough ≈ 23.3.
    expect(peak - trough).toBeGreaterThan(21);
    expect(peak - trough).toBeLessThan(25);
    expect(result.fit.rSquared).toBeGreaterThan(0.97);
  });

  it("rejects short history and invalid horizons", () => {
    expect(() => buildForecast(linearHistory(20, 1), 10)).toThrow(
      "at least 30 trading days",
    );
    expect(() => buildForecast(linearHistory(60, 1), -1)).toThrow(
      "non-negative integer",
    );
  });
});
