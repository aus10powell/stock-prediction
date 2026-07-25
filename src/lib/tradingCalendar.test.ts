import { describe, expect, it } from "vitest";
import {
  futureTradingDays,
  isTradingDay,
  parseDateKey,
  toDateKey,
  usMarketHolidays,
} from "./tradingCalendar";

describe("usMarketHolidays", () => {
  it("includes fixed, floating, and Easter-derived closures", () => {
    const holidays = usMarketHolidays(2024);

    expect(holidays).toContain("2024-01-01"); // New Year's Day
    expect(holidays).toContain("2024-01-15"); // Martin Luther King Jr. Day
    expect(holidays).toContain("2024-02-19"); // Washington's Birthday
    expect(holidays).toContain("2024-03-29"); // Good Friday
    expect(holidays).toContain("2024-05-27"); // Memorial Day
    expect(holidays).toContain("2024-06-19"); // Juneteenth
    expect(holidays).toContain("2024-07-04"); // Independence Day
    expect(holidays).toContain("2024-09-02"); // Labor Day
    expect(holidays).toContain("2024-11-28"); // Thanksgiving
    expect(holidays).toContain("2024-12-25"); // Christmas
  });

  it("shifts weekend holidays the way the exchange does", () => {
    // July 4 2020 fell on Saturday, observed the preceding Friday.
    expect(usMarketHolidays(2020)).toContain("2020-07-03");
    // Christmas 2022 fell on Sunday, observed the following Monday.
    expect(usMarketHolidays(2022)).toContain("2022-12-26");
    // A Saturday January 1 is not pulled back to December 31.
    expect(usMarketHolidays(2022)).not.toContain("2021-12-31");
    expect(usMarketHolidays(2022)).not.toContain("2022-01-03");
  });

  it("only observes Juneteenth once it became a market holiday", () => {
    expect(usMarketHolidays(2021)).not.toContain("2021-06-18");
    expect(usMarketHolidays(2022)).toContain("2022-06-20");
  });
});

describe("isTradingDay", () => {
  it("rejects weekends and holidays but accepts ordinary weekdays", () => {
    expect(isTradingDay(parseDateKey("2024-07-03"))).toBe(true);
    expect(isTradingDay(parseDateKey("2024-07-04"))).toBe(false);
    expect(isTradingDay(parseDateKey("2024-07-06"))).toBe(false);
  });
});

describe("futureTradingDays", () => {
  it("skips weekends and holidays when stepping forward", () => {
    const days = futureTradingDays(parseDateKey("2024-07-03"), 3).map(toDateKey);
    expect(days).toEqual(["2024-07-05", "2024-07-08", "2024-07-09"]);
  });

  it("returns exactly the requested number of trading days", () => {
    const days = futureTradingDays(parseDateKey("2024-12-20"), 10);
    expect(days).toHaveLength(10);
    expect(days.every(isTradingDay)).toBe(true);
  });
});
