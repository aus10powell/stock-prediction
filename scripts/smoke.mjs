#!/usr/bin/env node
// Ad hoc smoke test against a running server. Not part of the unit test suite.
const base = process.env.BASE_URL ?? "http://localhost:3000";

let failures = 0;

function check(label, condition, detail = "") {
  const status = condition ? "pass" : "FAIL";
  if (!condition) failures++;
  console.log(`  [${status}] ${label}${detail ? ` — ${detail}` : ""}`);
}

async function getJson(path) {
  const started = Date.now();
  const response = await fetch(`${base}${path}`);
  const body = await response.json();
  return { response, body, ms: Date.now() - started };
}

console.log(`Smoke testing ${base}\n`);

{
  console.log("GET /api/probability (AAPL, 20d, 1%)");
  const { response, body, ms } = await getJson(
    "/api/probability?ticker=AAPL&days=20&threshold=1",
  );
  check("status 200", response.status === 200, String(response.status));
  check("ticker echoed", body.ticker === "AAPL");
  check("21 percentile points", body.percentiles?.length === 21);
  check(
    "probabilities sum to 1",
    Math.abs(
      body.probabilityUp + body.probabilityDown + body.probabilityWithin - 1,
    ) < 1e-9,
  );
  check("volatility reported", body.assumptions?.annualizedVolatility > 0);
  console.log(
    `  up=${(body.probabilityUp * 100).toFixed(1)}% down=${(body.probabilityDown * 100).toFixed(1)}% within=${(body.probabilityWithin * 100).toFixed(1)}% vol=${(body.assumptions.annualizedVolatility * 100).toFixed(1)}% drift=${(body.assumptions.annualizedDrift * 100).toFixed(1)}% (${ms}ms)`,
  );
}

{
  console.log("\nGET /api/probability (drift=zero)");
  const { body } = await getJson(
    "/api/probability?ticker=AAPL&days=20&threshold=1&drift=zero",
  );
  check("drift removed", body.assumptions?.annualizedDrift === 0);
  check("median near zero", Math.abs(body.medianReturn) < 0.02);
  console.log(
    `  up=${(body.probabilityUp * 100).toFixed(1)}% down=${(body.probabilityDown * 100).toFixed(1)}%`,
  );
}

{
  console.log("\nGET /api/probability (threshold=5)");
  const { body } = await getJson(
    "/api/probability?ticker=AAPL&days=20&threshold=5",
  );
  check("threshold echoed", body.thresholdPercent === 5);
  console.log(
    `  up=${(body.probabilityUp * 100).toFixed(1)}% within=${(body.probabilityWithin * 100).toFixed(1)}%`,
  );
}

{
  console.log("\nGET /api/calibration (AAPL, 20d)");
  const { response, body, ms } = await getJson(
    "/api/calibration?ticker=AAPL&days=20",
  );
  check("status 200", response.status === 200, String(response.status));
  check("samples evaluated", body.samples > 50, `${body.samples} samples`);
  check("reliability bins present", body.up?.bins?.length > 0);
  console.log(
    `  ${body.samples} origins ${body.firstOrigin}..${body.lastOrigin} (${ms}ms)`,
  );
  console.log(
    `  up: predicted=${(body.up.meanPredicted * 100).toFixed(1)}% actual=${(body.up.observedFrequency * 100).toFixed(1)}% brier=${body.up.brierScore.toFixed(4)} skill=${(body.up.skillScore * 100).toFixed(1)}%`,
  );
  console.log(
    `  down: predicted=${(body.down.meanPredicted * 100).toFixed(1)}% actual=${(body.down.observedFrequency * 100).toFixed(1)}% brier=${body.down.brierScore.toFixed(4)} skill=${(body.down.skillScore * 100).toFixed(1)}%`,
  );
}

{
  console.log("\nGET /api/screen (5 tickers)");
  const { response, body, ms } = await getJson(
    "/api/screen?tickers=AAPL,MSFT,NVDA,GME,TSLA&days=20&threshold=1",
  );
  check("status 200", response.status === 200, String(response.status));
  check("all rows returned", body.rows?.length === 5);
  check(
    "no row errors",
    body.rows?.every((row) => !row.error),
    body.rows
      ?.filter((row) => row.error)
      .map((row) => `${row.ticker}: ${row.error}`)
      .join("; "),
  );
  console.log(`  ${ms}ms`);
  for (const row of body.rows ?? []) {
    console.log(
      `  ${row.ticker.padEnd(5)} up=${((row.probabilityUp ?? 0) * 100).toFixed(1)}% down=${((row.probabilityDown ?? 0) * 100).toFixed(1)}% vol=${((row.annualizedVolatility ?? 0) * 100).toFixed(1)}%`,
    );
  }
}

{
  console.log("\nGET /api/forecast (2 years)");
  const { response, body, ms } = await getJson(
    "/api/forecast?ticker=AAPL&years=2",
  );
  check("status 200", response.status === 200, String(response.status));
  check("504 trading days ahead", body.fit?.tradingDaysAhead === 504);
  const projected = body.forecast.slice(body.history.length);
  check("projected point count", projected.length === 504);
  const firstWidth = projected[0].yhatUpper - projected[0].yhatLower;
  const lastWidth =
    projected[projected.length - 1].yhatUpper -
    projected[projected.length - 1].yhatLower;
  check(
    "interval widens materially with horizon",
    lastWidth > firstWidth * 1.5,
    `${firstWidth.toFixed(1)} → ${lastWidth.toFixed(1)}`,
  );
  const weekdays = new Set(
    projected.map((point) => new Date(`${point.date}T00:00:00Z`).getUTCDay()),
  );
  check("no weekend dates", !weekdays.has(0) && !weekdays.has(6));
  console.log(
    `  R²=${body.fit.rSquared.toFixed(3)} slope/day=$${body.fit.slopePerTradingDay.toFixed(4)} band ${firstWidth.toFixed(2)}→${lastWidth.toFixed(2)} (${ms}ms)`,
  );
}

{
  console.log("\nValidation");
  for (const [path, expected] of [
    ["/api/probability?days=0", 400],
    ["/api/probability?days=999", 400],
    ["/api/probability?threshold=0", 400],
    ["/api/probability?drift=sideways", 400],
    ["/api/probability?ticker=not%20a%20ticker", 400],
    ["/api/forecast?years=9", 400],
    ["/api/screen?tickers=", 400],
  ]) {
    const { response, body } = await getJson(path);
    check(
      `${path} → ${expected}`,
      response.status === expected,
      body.error ?? "",
    );
  }
}

{
  console.log("\nPage render");
  const response = await fetch(
    `${base}/?ticker=MSFT&days=30&threshold=2&drift=zero&years=2`,
  );
  // React separates adjacent text nodes with comment markers.
  const html = (await response.text()).replaceAll("<!-- -->", "");
  check("status 200", response.status === 200, String(response.status));
  check("ticker in markup", html.includes("MSFT"));
  check("horizon in markup", html.includes("30 trading days"));
  check("threshold in markup", html.includes("±2%"));
  check("drift note rendered", html.includes("Drift has been removed"));
  check("calibration section rendered", html.includes("trustworthy"));
  check("watchlist section rendered", html.includes("Watchlist screen"));
  check("disclosure present", html.includes("not investment advice"));
  check(
    "no dead feedback field",
    !html.includes("Any feedback on the app?"),
  );
}

console.log(
  failures === 0 ? "\nAll smoke checks passed." : `\n${failures} check(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
