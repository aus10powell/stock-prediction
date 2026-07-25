const MS_PER_DAY = 86_400_000;

export function toDateKey(date: Date): string {
  return date.toISOString().slice(0, 10);
}

export function parseDateKey(key: string): Date {
  return new Date(`${key}T00:00:00.000Z`);
}

function utc(year: number, month: number, day: number): Date {
  return new Date(Date.UTC(year, month, day));
}

function addCalendarDays(date: Date, days: number): Date {
  return new Date(date.getTime() + days * MS_PER_DAY);
}

function nthWeekdayOfMonth(
  year: number,
  month: number,
  weekday: number,
  n: number,
): Date {
  const first = utc(year, month, 1);
  const offset = (weekday - first.getUTCDay() + 7) % 7;
  return utc(year, month, 1 + offset + (n - 1) * 7);
}

function lastWeekdayOfMonth(
  year: number,
  month: number,
  weekday: number,
): Date {
  const last = utc(year, month + 1, 0);
  const offset = (last.getUTCDay() - weekday + 7) % 7;
  return utc(year, month, last.getUTCDate() - offset);
}

/** Meeus/Jones/Butcher algorithm for the Gregorian date of Easter Sunday. */
function easterSunday(year: number): Date {
  const a = year % 19;
  const b = Math.floor(year / 100);
  const c = year % 100;
  const d = Math.floor(b / 4);
  const e = b % 4;
  const f = Math.floor((b + 8) / 25);
  const g = Math.floor((b - f + 1) / 3);
  const h = (19 * a + b - d - g + 15) % 30;
  const i = Math.floor(c / 4);
  const k = c % 4;
  const l = (32 + 2 * e + 2 * i - h - k) % 7;
  const m = Math.floor((a + 11 * h + 22 * l) / 451);
  const month = Math.floor((h + l - 7 * m + 114) / 31) - 1;
  const day = ((h + l - 7 * m + 114) % 31) + 1;
  return utc(year, month, day);
}

/**
 * NYSE moves a holiday falling on Saturday to the preceding Friday and one
 * falling on Sunday to the following Monday. New Year's Day is the exception:
 * a Saturday January 1 is not observed on the previous trading day.
 */
function observed(date: Date): Date | null {
  const day = date.getUTCDay();
  if (day === 6) return addCalendarDays(date, -1);
  if (day === 0) return addCalendarDays(date, 1);
  return date;
}

const holidayCache = new Map<number, Set<string>>();

export function usMarketHolidays(year: number): Set<string> {
  const cached = holidayCache.get(year);
  if (cached) return cached;

  const dates: Array<Date | null> = [];

  const newYear = utc(year, 0, 1);
  dates.push(newYear.getUTCDay() === 6 ? null : observed(newYear));

  dates.push(nthWeekdayOfMonth(year, 0, 1, 3)); // Martin Luther King Jr. Day
  dates.push(nthWeekdayOfMonth(year, 1, 1, 3)); // Washington's Birthday
  dates.push(addCalendarDays(easterSunday(year), -2)); // Good Friday
  dates.push(lastWeekdayOfMonth(year, 4, 1)); // Memorial Day
  if (year >= 2022) {
    dates.push(observed(utc(year, 5, 19))); // Juneteenth
  }
  dates.push(observed(utc(year, 6, 4))); // Independence Day
  dates.push(nthWeekdayOfMonth(year, 8, 1, 1)); // Labor Day
  dates.push(nthWeekdayOfMonth(year, 10, 4, 4)); // Thanksgiving
  dates.push(observed(utc(year, 11, 25))); // Christmas

  const keys = new Set(
    dates.filter((date): date is Date => date !== null).map(toDateKey),
  );
  holidayCache.set(year, keys);
  return keys;
}

export function isTradingDay(date: Date): boolean {
  const day = date.getUTCDay();
  if (day === 0 || day === 6) return false;
  return !usMarketHolidays(date.getUTCFullYear()).has(toDateKey(date));
}

export function nextTradingDay(date: Date): Date {
  let candidate = addCalendarDays(date, 1);
  while (!isTradingDay(candidate)) {
    candidate = addCalendarDays(candidate, 1);
  }
  return candidate;
}

/** The next `count` trading days strictly after `from`. */
export function futureTradingDays(from: Date, count: number): Date[] {
  const days: Date[] = [];
  let cursor = from;
  for (let i = 0; i < count; i++) {
    cursor = nextTradingDay(cursor);
    days.push(cursor);
  }
  return days;
}

export const TRADING_DAYS_PER_YEAR = 252;
