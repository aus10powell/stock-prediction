# Stock Forecast

Interactive stock forecasting app by **Austin Powell**.

Enter a ticker, choose how many years ahead to project, and explore price history plus trend / seasonal forecast charts.

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
- **Recharts** for interactive plots

## Original Streamlit app

The previous Heroku Streamlit + `fbprophet` version lives in [`legacy/`](./legacy/).
