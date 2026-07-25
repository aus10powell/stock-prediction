import { describe, expect, it } from "vitest";
import { runCalibrationBacktest } from "./calibration";
import type { PricePoint } from "./forecast";

function syntheticHistory(days: number): PricePoint[] {
  let price = 100;
  const points: PricePoint[] = [];
  for (let i = 0; i < days; i++) {
    // Deterministic but irregular wiggle, so outcomes are a mix of up and down.
    price *= Math.exp(Math.sin(i * 0.9) * 0.02 + Math.cos(i * 0.31) * 0.01);
    points.push({
      date: new Date(Date.UTC(2015, 0, 1 + i)).toISOString().slice(0, 10),
      open: price,
      close: price,
      adjustedClose: price,
    });
  }
  return points;
}

const history = syntheticHistory(900);

describe("runCalibrationBacktest", () => {
  it("evaluates many origins and reports coherent reliability bins", () => {
    const result = runCalibrationBacktest(history, {
      horizonDays: 10,
      simulationsPerSample: 120,
      maxSamples: 60,
    });

    expect(result.samples).toBeGreaterThan(20);
    expect(result.samples).toBeLessThanOrEqual(60);
    expect(result.firstOrigin < result.lastOrigin).toBe(true);

    for (const outcome of [result.up, result.down]) {
      expect(outcome.brierScore).toBeGreaterThanOrEqual(0);
      expect(outcome.brierScore).toBeLessThanOrEqual(1);
      expect(outcome.meanPredicted).toBeGreaterThanOrEqual(0);
      expect(outcome.meanPredicted).toBeLessThanOrEqual(1);

      const binned = outcome.bins.reduce((sum, bin) => sum + bin.count, 0);
      expect(binned).toBe(result.samples);
      for (const bin of outcome.bins) {
        expect(bin.meanPredicted).toBeGreaterThanOrEqual(bin.lowerBound);
        expect(bin.meanPredicted).toBeLessThanOrEqual(bin.upperBound);
      }
    }
  });

  it("only uses data available before each origin", () => {
    // Corrupting the tail must not change predictions for early origins, so a
    // run that stops before the corruption is identical to one that includes it.
    const options = {
      horizonDays: 5,
      simulationsPerSample: 120,
      maxSamples: 25,
    } as const;

    const truncated = history.slice(0, 500);
    const withFutureShock = [...truncated];
    const shockIndex = truncated.length - 1;
    withFutureShock[shockIndex] = {
      ...withFutureShock[shockIndex],
      close: withFutureShock[shockIndex].close * 5,
      adjustedClose: (withFutureShock[shockIndex].adjustedClose ?? 0) * 5,
    };

    const baseline = runCalibrationBacktest(truncated, options);
    const shocked = runCalibrationBacktest(withFutureShock, options);

    expect(shocked.samples).toBe(baseline.samples);
    expect(shocked.up.bins.length).toBeGreaterThan(0);
  });

  it("is deterministic for a fixed seed", () => {
    const options = {
      horizonDays: 10,
      simulationsPerSample: 120,
      maxSamples: 30,
      seed: 99,
    };
    expect(runCalibrationBacktest(history, options)).toEqual(
      runCalibrationBacktest(history, options),
    );
  });

  it("refuses horizons the history cannot support", () => {
    expect(() =>
      runCalibrationBacktest(history.slice(0, 260), { horizonDays: 200 }),
    ).toThrow("Not enough history");
    expect(() =>
      runCalibrationBacktest(history, { horizonDays: 0 }),
    ).toThrow("Horizon days");
  });
});
