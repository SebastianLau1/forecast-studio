# Forecast Studio

Interactive forecasting on real public data (NYC subway ridership, Wikipedia traffic, Mauna Loa CO₂) or your own CSV, with chronological validation and model comparisons.

[Live app](https://sebastianlau1.github.io/forecast-studio/) · [Portfolio](https://sebastianlau.is-a.dev)

## Run

```sh
npm run dev
# Open http://localhost:8080
npm test
```

## What it does

Pick one of three real datasets or upload (or drop) a `date,value` CSV. Compare linear regression trend, seasonal-naive forecasting, and Holt’s linear smoothing. Model selection uses MAE on the last 20% of observations, then refits the chosen model on the full series.

The chart shades the holdout window and overlays the selected model's backtest against what actually happened, so you can see why a model won. Hover for exact values and ranges. The leaderboard ranks models by MAE; click any row to forecast with it instead of the auto pick. Exports include the forecast and heuristic range.

All processing runs locally in the browser. The upload parser validates increasing, evenly spaced ISO dates, finite numeric values, 21–5,000 rows, and a 500 KB file limit. The model tests cover linear extrapolation, seasonal repetition, holdout scoring, and invalid CSVs.

Shaded ranges are `±1.96 × holdout RMSE × sqrt(1 + step / holdout length)`. They are **heuristic uncertainty ranges**, not calibrated 95% prediction intervals. Results are baselines for demonstration, not financial or operational advice. Monthly series with varying day counts are not supported by the current evenly spaced date parser.

## Data

The bundled series live in `web/data/` as `date,value` CSVs with their sources in `datasets.json`. Refresh them with `node scripts/fetch-data.mjs` (Node 18+, no dependencies); the script checks that every series is evenly spaced before writing it.

| Series | Cadence | Window | Source | License |
| --- | --- | --- | --- | --- |
| NYC subway ridership | Daily | Last 26 weeks | [MTA Daily Ridership and Traffic](https://data.ny.gov/Transportation/MTA-Daily-Ridership-and-Traffic-Beginning-2020/sayj-mze2), data.ny.gov | NY Open Data terms of use |
| Wikipedia views: Machine learning | Daily | Last 26 weeks | [Wikimedia Pageviews API](https://pageviews.wmcloud.org/?project=en.wikipedia.org&pages=Machine_learning) (user traffic only) | CC0 |
| Mauna Loa CO₂ | Weekly | Last 5 years | [NOAA Global Monitoring Laboratory](https://gml.noaa.gov/ccgg/trends/data.html) | Public domain |

NOAA flags weeks with too few valid measurements as `-999.99`; the script fills those (2 in the current window) by linear interpolation and says so in the app. Each dataset sets a default season length: 7 for the daily series, 52 for weekly CO₂.

## Deployment

Every push to `main` runs the tests and publishes `web/` to GitHub Pages (`.github/workflows/deploy.yml`). The app can also be deployed as a Cloudflare Workers static-asset app with `wrangler.jsonc`.

No credentials are stored in source.
