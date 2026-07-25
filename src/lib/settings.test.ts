import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS, parseSettings, settingsToQuery } from "./settings";

describe("parseSettings", () => {
  it("reads a full query string", () => {
    expect(
      parseSettings({
        ticker: "aapl",
        days: "45",
        threshold: "2.5",
        drift: "zero",
        years: "3",
      }),
    ).toEqual({
      ticker: "AAPL",
      days: 45,
      threshold: 2.5,
      drift: "zero",
      years: 3,
    });
  });

  it("falls back to defaults for missing or unusable values", () => {
    expect(parseSettings({})).toEqual(DEFAULT_SETTINGS);
    expect(
      parseSettings({
        ticker: "not a ticker",
        days: "abc",
        threshold: "-4",
        drift: "sideways",
        years: "0",
      }),
    ).toEqual({ ...DEFAULT_SETTINGS, years: 1 });
  });

  it("clamps numeric values into range", () => {
    expect(parseSettings({ days: "9999" }).days).toBe(252);
    expect(parseSettings({ days: "-5" }).days).toBe(1);
    expect(parseSettings({ years: "12" }).years).toBe(4);
    expect(parseSettings({ threshold: "99" }).threshold).toBe(
      DEFAULT_SETTINGS.threshold,
    );
  });

  it("uses the first value when a parameter repeats", () => {
    expect(parseSettings({ ticker: ["MSFT", "AAPL"] }).ticker).toBe("MSFT");
  });
});

describe("settingsToQuery", () => {
  it("round-trips through parseSettings", () => {
    const settings = {
      ticker: "BRK-B",
      days: 63,
      threshold: 5,
      drift: "zero" as const,
      years: 2,
    };
    const parsed = parseSettings(
      Object.fromEntries(new URLSearchParams(settingsToQuery(settings))),
    );
    expect(parsed).toEqual(settings);
  });
});
