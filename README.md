# Stock Forecast

Interactive stock forecasting app by **Austin Powell**.

Enter a ticker, choose forecast and trading-day horizons, and explore price
history, trend / seasonal forecasts, and Monte Carlo probabilities for a move
of at least 1% in either direction.

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

- **Next.js** (App Router) on Vercel
- **Yahoo Finance** market history via `yahoo-finance2`
- Additive forecast (trend + weekly + yearly seasonality), inspired by the original Prophet model
- Seeded 10,000-path block-bootstrap simulation using recent adjusted returns
- **Recharts** for interactive plots

The probability simulation is based on historical returns. It is not investment
advice and does not account for future news or changing market regimes.

## Original Streamlit app

The previous Heroku Streamlit + `fbprophet` version lives in [`legacy/`](./legacy/).
