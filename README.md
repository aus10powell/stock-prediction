# Stock Forecast

Interactive stock forecasting app by **Austin Powell**.

Enter a ticker, choose forecast and trading-day horizons, and explore price
history, trend / seasonal forecasts, and Monte Carlo probabilities for a move
of at least 1% in either direction.

## Live on Vercel

Deploy with one click or from the CLI:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/aus10powell/stock-prediction)

```bash
npm install
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

```bash
npx vercel --prod
```

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
