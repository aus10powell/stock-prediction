# Stock Odds

Interactive stock probability app by **Austin Powell**.

Pick a ticker, a horizon in trading days, and a move threshold, then see the
simulated probability that the stock rises or falls by at least that much —
along with a backtest of whether those probabilities have historically been
trustworthy.

Every control is encoded in the URL, so any view can be shared or bookmarked:
`/?ticker=AAPL&days=20&threshold=1&drift=zero&years=1`

## Live

**https://workspace-lemon-two.vercel.app**

## Deploy from any machine (including remote / cloud agents)

### 1. Automatic (recommended)

The repo is linked to the Vercel project `stock-prediction`.  
**Push to `main` → production deploys.** No local Vercel login required.

```bash
git push origin main
```

### 2. CLI (when you need an explicit deploy)

```bash
npm install
npm run deploy
```

`scripts/deploy.sh` uses the committed `.vercel/project.json` so the project is
known without interactive prompts.

Auth options for remotes:

| Method | How |
|--------|-----|
| Git push | Push `main` (uses Vercel Git integration) |
| Token | Set `VERCEL_TOKEN` from https://vercel.com/account/tokens, then `npm run deploy` |
| Device login | `npx vercel login` once, then `npm run deploy` |

Optional GitHub Action: **Actions → Deploy to Vercel → Run workflow**  
(requires repo secret `VERCEL_TOKEN`).

## Local development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Stack

- **Next.js** (App Router) on Vercel, with the first render computed on the server
- **Yahoo Finance** market history via `yahoo-finance2`, cached per instance
- **Recharts** for interactive plots
- **Vitest** for unit and route tests (`npm test`)

## How the numbers are produced

### Probability simulation

A seeded 10,000-path block bootstrap. Five-day blocks of recent adjusted-close
log returns are resampled to build each path, which preserves some short-term
volatility clustering and avoids assuming returns are normally distributed.
Adjusted closes keep splits and dividends from registering as real moves.

Drift is an explicit choice. In **historical** mode the paths inherit the
average drift of the lookback window, so a strong bull run tilts the upside
probability. **Zero** mode removes the mean so the answer reflects volatility
alone. The assumed annualized drift and volatility are always displayed.

### Calibration backtest

Probabilities are only useful if they are calibrated, so the app replays the
simulation at many historical starting points using only the data available at
each one, then compares the predicted probability against what actually
happened. It reports reliability bins, a Brier score, and skill relative to
always quoting the historical base rate. Overlapping horizons make the samples
correlated, so this is a sanity check rather than a precise measurement.

### Model fit diagnostic

A joint least-squares fit of price against a linear trend, weekday effects, and
yearly Fourier terms. All terms are estimated together, since fitting
correlated sine and cosine terms separately biases them. The trend is indexed by
trading-day position and extrapolated one slope step per projected trading day,
using a US market calendar that accounts for weekends and holidays. Prediction
intervals widen with distance from the fitted data.

This panel describes the fitted history; it is not a forecast. Prices do not
follow a deterministic trend, so the probability panel is the forward-looking
view.

## API

| Route | Purpose |
|-------|---------|
| `GET /api/probability` | Simulated probabilities for one ticker |
| `GET /api/calibration` | Walk-forward calibration backtest |
| `GET /api/screen` | Same simulation across a watchlist (up to 12 tickers) |
| `GET /api/forecast` | Trend and seasonality fit diagnostic |

Shared parameters: `ticker`, `days` (1–252), `threshold` (percent), `drift`
(`historical` or `zero`), `seed`. Responses are CDN-cacheable and rate limited
per caller.

All of this is historical simulation. It is not investment advice, and it cannot
account for news, earnings, or changing market regimes.

## Original Streamlit app

The previous Heroku Streamlit + `fbprophet` version lives in [`legacy/`](./legacy/).
