# Forecast Studio

Interactive forecasting with CSV uploads, chronological validation, and model comparisons.

[Live app](https://sebastianlau.is-a.dev/projects/forecast-studio/index.html) · [Portfolio](https://sebastianlau.is-a.dev)

## Run

```sh
npm run dev
# Open http://localhost:8080
npm test
```

## What it does

Upload a `date,value` CSV or select one of three reproducible synthetic datasets. Compare linear regression trend, seasonal-naive forecasting, and Holt’s linear smoothing. Model selection uses MAE on the last 20% of observations, then refits the chosen model on the full series. Exports include the forecast and heuristic range.

All processing runs locally in the browser. The upload parser validates increasing, evenly spaced ISO dates, finite numeric values, 21–5,000 rows, and a 500 KB file limit. The model tests cover linear extrapolation, seasonal repetition, holdout scoring, and invalid CSVs.

Shaded ranges are `±1.96 × holdout RMSE × sqrt(1 + step / holdout length)`. They are **heuristic uncertainty ranges**, not calibrated 95% prediction intervals. Samples are synthetic; results are baselines for demonstration, not financial or operational advice. Monthly series with varying day counts are not supported by the current evenly spaced date parser.

## Deployment

The app is independently deployable as a Cloudflare Worker/Pages static asset app using `wrangler.jsonc`. Its public demo is also deployed with the portfolio under `/projects/forecast-studio/`. Static app files are committed into the portfolio; to refresh them, run `python3 scripts/sync_projects.py` from the portfolio checkout with sibling project checkouts present.

No credentials are stored in source.
