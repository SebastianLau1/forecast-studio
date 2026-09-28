# Forecast Studio

Interactive forecasting with CSV uploads, chronological validation, and model comparisons.

[Live app](https://sebastianlau1.github.io/forecast-studio/) · [Portfolio](https://sebastianlau.is-a.dev)

## Run

```sh
npm run dev
# Open http://localhost:8080
npm test
```

## What it does

Upload (or drop) a `date,value` CSV, or select one of three reproducible synthetic datasets. Compare linear regression trend, seasonal-naive forecasting, and Holt’s linear smoothing. Model selection uses MAE on the last 20% of observations, then refits the chosen model on the full series.

The chart shades the holdout window and overlays the selected model's backtest against what actually happened, so you can see why a model won. Hover for exact values and ranges. The leaderboard ranks models by MAE; click any row to forecast with it instead of the auto pick. Exports include the forecast and heuristic range.

All processing runs locally in the browser. The upload parser validates increasing, evenly spaced ISO dates, finite numeric values, 21–5,000 rows, and a 500 KB file limit. The model tests cover linear extrapolation, seasonal repetition, holdout scoring, and invalid CSVs.

Shaded ranges are `±1.96 × holdout RMSE × sqrt(1 + step / holdout length)`. They are **heuristic uncertainty ranges**, not calibrated 95% prediction intervals. Samples are synthetic; results are baselines for demonstration, not financial or operational advice. Monthly series with varying day counts are not supported by the current evenly spaced date parser.

## Deployment

Every push to `main` runs the tests and publishes `web/` to GitHub Pages (`.github/workflows/deploy.yml`). The app can also be deployed as a Cloudflare Workers static-asset app with `wrangler.jsonc`.

No credentials are stored in source.
