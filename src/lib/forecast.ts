import { fitLeastSquares, leverage } from "./leastSquares";
import { futureTradingDays, parseDateKey, toDateKey } from "./tradingCalendar";

export type PricePoint = {
  date: string;
  open: number;
  close: number;
  adjustedClose?: number;
};

export type ForecastPoint = {
  date: string;
  yhat: number;
  yhatLower: number;
  yhatUpper: number;
  actualClose?: number;
};

export type SeasonalityPoint = {
  label: string;
  value: number;
};

export type ModelFit = {
  rSquared: number;
  residualStd: number;
  slopePerTradingDay: number;
  observations: number;
  parameters: number;
  tradingDaysAhead: number;
};

export type ForecastResult = {
  history: PricePoint[];
  forecast: ForecastPoint[];
  weekly: SeasonalityPoint[];
  yearly: SeasonalityPoint[];
  trend: SeasonalityPoint[];
  fit: ModelFit;
};

const FOURIER_ORDER = 3;
const WEEKDAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const Z_95 = 1.959964;

function dayOfYear(date: Date): number {
  const start = Date.UTC(date.getUTCFullYear(), 0, 0);
  return Math.floor((date.getTime() - start) / 86_400_000);
}

function fourierTerms(date: Date): number[] {
  const t = (2 * Math.PI * dayOfYear(date)) / 365.25;
  const terms: number[] = [];
  for (let k = 1; k <= FOURIER_ORDER; k++) {
    terms.push(Math.sin(k * t), Math.cos(k * t));
  }
  return terms;
}

/**
 * Least-squares fit of price against a linear trend, weekday effects, and
 * yearly Fourier terms, all estimated jointly.
 *
 * The trend is indexed by trading-day position, and forward projection advances
 * that same index one step per generated trading day, so a horizon of N trading
 * days extrapolates exactly N steps of slope.
 *
 * This is a description of the fitted history, not a market prediction: prices
 * are not well modelled by a deterministic trend. Use the simulation in
 * `monteCarlo.ts` for forward-looking probabilities.
 */
export function buildForecast(
  history: PricePoint[],
  tradingDaysAhead: number,
): ForecastResult {
  if (history.length < 30) {
    throw new Error("Need at least 30 trading days of history to forecast.");
  }
  if (!Number.isInteger(tradingDaysAhead) || tradingDaysAhead < 0) {
    throw new Error("Trading days ahead must be a non-negative integer.");
  }

  const closes = history.map((point) => point.close);
  const dates = history.map((point) => parseDateKey(point.date));
  const lastIndex = history.length - 1;
  const scale = lastIndex;

  const presentWeekdays = [...new Set(dates.map((d) => d.getUTCDay()))].sort(
    (a, b) => a - b,
  );
  const baselineWeekday = presentWeekdays[0];
  const dummyWeekdays = presentWeekdays.slice(1);

  const designRow = (date: Date, index: number): number[] => [
    1,
    index / scale,
    ...dummyWeekdays.map((day) => (date.getUTCDay() === day ? 1 : 0)),
    ...fourierTerms(date),
  ];

  const design = dates.map((date, index) => designRow(date, index));
  const fit = fitLeastSquares(design, closes);
  const { beta, covarianceUnscaled, residualStd } = fit;

  const slopePerTradingDay = beta[1] / scale;
  const weekdayOffset = 2;
  const fourierOffset = weekdayOffset + dummyWeekdays.length;

  const weekdayEffectRaw = new Map<number, number>([[baselineWeekday, 0]]);
  dummyWeekdays.forEach((day, i) => {
    weekdayEffectRaw.set(day, beta[weekdayOffset + i]);
  });
  const weekdayValues = [...weekdayEffectRaw.values()];
  const weekdayMean =
    weekdayValues.reduce((sum, v) => sum + v, 0) / weekdayValues.length;

  const yearlyAt = (date: Date): number => {
    const terms = fourierTerms(date);
    return terms.reduce(
      (sum, term, i) => sum + term * beta[fourierOffset + i],
      0,
    );
  };
  const yearlySamples = Array.from({ length: 365 }, (_, i) =>
    yearlyAt(new Date(Date.UTC(2024, 0, 1 + i))),
  );
  const yearlyMean =
    yearlySamples.reduce((sum, v) => sum + v, 0) / yearlySamples.length;

  const predict = (date: Date, index: number) => {
    const row = designRow(date, index);
    const yhat = row.reduce((sum, value, i) => sum + value * beta[i], 0);
    // Widen with distance from the fitted data: σ√(1 + leverage).
    const band = Z_95 * residualStd * Math.sqrt(1 + leverage(row, covarianceUnscaled));
    return { yhat, band };
  };

  const forecast: ForecastPoint[] = history.map((point, index) => {
    const { yhat, band } = predict(dates[index], index);
    return {
      date: point.date,
      yhat,
      yhatLower: yhat - band,
      yhatUpper: yhat + band,
      actualClose: point.close,
    };
  });

  futureTradingDays(dates[lastIndex], tradingDaysAhead).forEach((date, step) => {
    const { yhat, band } = predict(date, lastIndex + step + 1);
    forecast.push({
      date: toDateKey(date),
      yhat,
      yhatLower: yhat - band,
      yhatUpper: yhat + band,
    });
  });

  const weekly: SeasonalityPoint[] = WEEKDAY_NAMES.map((label, day) => ({
    label,
    value: weekdayEffectRaw.has(day)
      ? (weekdayEffectRaw.get(day) as number) - weekdayMean
      : 0,
  })).filter((_, day) => weekdayEffectRaw.has(day));

  const yearly: SeasonalityPoint[] = Array.from({ length: 12 }, (_, month) => {
    const sample = new Date(Date.UTC(2024, month, 15));
    return {
      label: sample.toLocaleString("en-US", {
        month: "short",
        timeZone: "UTC",
      }),
      value: yearlyAt(sample) - yearlyMean,
    };
  });

  const trendStep = Math.max(1, Math.floor(history.length / 24));
  const trend: SeasonalityPoint[] = history
    .map((point, index) => ({ point, index }))
    .filter(({ index }) => index % trendStep === 0 || index === lastIndex)
    .map(({ point, index }) => ({
      label: point.date.slice(0, 7),
      value: beta[0] + slopePerTradingDay * index,
    }));

  return {
    history,
    forecast,
    weekly,
    yearly,
    trend,
    fit: {
      rSquared: fit.rSquared,
      residualStd,
      slopePerTradingDay,
      observations: history.length,
      parameters: beta.length,
      tradingDaysAhead,
    },
  };
}
