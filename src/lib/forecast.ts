export type PricePoint = {
  date: string;
  open: number;
  close: number;
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

export type ForecastResult = {
  history: PricePoint[];
  forecast: ForecastPoint[];
  weekly: SeasonalityPoint[];
  yearly: SeasonalityPoint[];
  trend: SeasonalityPoint[];
};

function toDateKey(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function addDays(d: Date, days: number): Date {
  const next = new Date(d);
  next.setUTCDate(next.getUTCDate() + days);
  return next;
}

function linearRegression(xs: number[], ys: number[]) {
  const n = xs.length;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - meanX) * (ys[i] - meanY);
    den += (xs[i] - meanX) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = meanY - slope * meanX;
  return { slope, intercept };
}

function dayOfYear(d: Date): number {
  const start = Date.UTC(d.getUTCFullYear(), 0, 0);
  const diff = d.getTime() - start;
  return Math.floor(diff / 86_400_000);
}

/**
 * Lightweight additive forecast inspired by Prophet:
 * trend + weekly seasonality + yearly Fourier terms.
 */
export function buildForecast(
  history: PricePoint[],
  periods: number,
): ForecastResult {
  if (history.length < 30) {
    throw new Error("Need at least 30 trading days of history to forecast.");
  }

  const closes = history.map((h) => h.close);
  const dates = history.map((h) => new Date(`${h.date}T00:00:00.000Z`));
  const xs = dates.map((_, i) => i);

  const { slope, intercept } = linearRegression(xs, closes);
  const trendSeries = xs.map((x) => intercept + slope * x);
  const detrended = closes.map((y, i) => y - trendSeries[i]);

  const weeklyBuckets = Array.from({ length: 7 }, () => [] as number[]);
  detrended.forEach((v, i) => {
    weeklyBuckets[dates[i].getUTCDay()].push(v);
  });
  const weeklyEffect = weeklyBuckets.map((bucket) =>
    bucket.length ? bucket.reduce((a, b) => a + b, 0) / bucket.length : 0,
  );
  const weeklyMean =
    weeklyEffect.reduce((a, b) => a + b, 0) / weeklyEffect.length;
  const weeklyCentered = weeklyEffect.map((v) => v - weeklyMean);

  const afterWeekly = detrended.map(
    (v, i) => v - weeklyCentered[dates[i].getUTCDay()],
  );

  const order = 3;
  const features = afterWeekly.map((_, i) => {
    const t = (2 * Math.PI * dayOfYear(dates[i])) / 365.25;
    const row: number[] = [];
    for (let k = 1; k <= order; k++) {
      row.push(Math.sin(k * t), Math.cos(k * t));
    }
    return row;
  });

  const coeffs = Array.from({ length: order * 2 }, () => 0);
  for (let j = 0; j < coeffs.length; j++) {
    const xsJ = features.map((f) => f[j]);
    const { slope: s } = linearRegression(xsJ, afterWeekly);
    coeffs[j] = s;
  }

  const yearlyAt = (d: Date) => {
    const t = (2 * Math.PI * dayOfYear(d)) / 365.25;
    let sum = 0;
    for (let k = 1; k <= order; k++) {
      sum +=
        coeffs[(k - 1) * 2] * Math.sin(k * t) +
        coeffs[(k - 1) * 2 + 1] * Math.cos(k * t);
    }
    return sum;
  };

  const fitted = dates.map((d, i) => {
    return (
      intercept +
      slope * i +
      weeklyCentered[d.getUTCDay()] +
      yearlyAt(d)
    );
  });

  const residuals = closes.map((y, i) => y - fitted[i]);
  const residualStd = Math.sqrt(
    residuals.reduce((a, b) => a + b * b, 0) / residuals.length,
  );
  const band = 1.96 * residualStd;

  const forecast: ForecastPoint[] = history.map((h, i) => ({
    date: h.date,
    yhat: fitted[i],
    yhatLower: fitted[i] - band,
    yhatUpper: fitted[i] + band,
    actualClose: h.close,
  }));

  const lastDate = dates[dates.length - 1];
  const lastIndex = xs[xs.length - 1];
  for (let step = 1; step <= periods; step++) {
    const d = addDays(lastDate, step);
    // Skip weekends for a more realistic trading calendar
    const dow = d.getUTCDay();
    if (dow === 0 || dow === 6) continue;
    const x = lastIndex + step;
    const yhat =
      intercept + slope * x + weeklyCentered[dow] + yearlyAt(d);
    forecast.push({
      date: toDateKey(d),
      yhat,
      yhatLower: yhat - band,
      yhatUpper: yhat + band,
    });
  }

  const weekdayNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const weekly: SeasonalityPoint[] = weekdayNames.map((label, i) => ({
    label,
    value: weeklyCentered[i],
  }));

  const yearly: SeasonalityPoint[] = Array.from({ length: 12 }, (_, month) => {
    const sample = new Date(Date.UTC(2024, month, 15));
    return {
      label: sample.toLocaleString("en-US", { month: "short", timeZone: "UTC" }),
      value: yearlyAt(sample),
    };
  });

  const trendStep = Math.max(1, Math.floor(history.length / 24));
  const trend: SeasonalityPoint[] = history
    .filter((_, i) => i % trendStep === 0 || i === history.length - 1)
    .map((h, idx) => {
      const i =
        idx === 0
          ? 0
          : history.findIndex((p) => p.date === h.date);
      return {
        label: h.date.slice(0, 7),
        value: intercept + slope * i,
      };
    });

  return { history, forecast, weekly, yearly, trend };
}
