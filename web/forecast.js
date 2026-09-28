// Baseline forecasting models, chronological holdout scoring, and CSV parsing.

export const names = {
  trend: "Linear trend",
  seasonal: "Seasonal naive",
  holt: "Holt’s linear smoothing",
  holtWinters: "Holt-Winters seasonal",
};

const HOLDOUT_SHARE = 0.2;
const HOLT_ALPHA = 0.35; // level smoothing
const HOLT_BETA = 0.15; // trend smoothing
// Holt-Winters smoothing weights are tuned per series on its own history (see fitHoltWinters).
const HW_GRID = {
  alpha: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  beta: [0, 0.02, 0.05, 0.1, 0.2],
  gamma: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
};
const DAY = 86_400_000;

/** Models that need whole seasons of history, and how many. */
const seasonsNeeded = { seasonal: 1, holtWinters: 2 };
const mean = (list) => list.reduce((sum, v) => sum + v, 0) / list.length;

/**
 * Forecast `horizon` steps. `weights` ({ alpha, beta, gamma }) fixes the Holt-Winters smoothing;
 * omit it to auto-tune on `values`.
 */
export function predict(values, horizon, model, period = 7, weights) {
  if (values.length < 3 || !values.every(Number.isFinite) || !Number.isInteger(horizon) || horizon < 1) {
    throw Error("A valid numeric series and positive horizon are required.");
  }
  const n = values.length;

  if (model === "seasonal") {
    if (n < period) throw Error("Not enough observations for this season length.");
    return Array.from({ length: horizon }, (_, i) => values[n - period + (i % period)]);
  }

  if (model === "trend") {
    const xMean = (n - 1) / 2;
    const yMean = values.reduce((sum, y) => sum + y, 0) / n;
    let num = 0;
    let den = 0;
    values.forEach((y, x) => {
      num += (x - xMean) * (y - yMean);
      den += (x - xMean) ** 2;
    });
    const slope = num / den;
    return Array.from({ length: horizon }, (_, i) => yMean + slope * (n + i - xMean));
  }

  if (model === "holt") {
    let level = values[0];
    let trend = values[1] - values[0];
    for (let i = 1; i < n; i++) {
      const previous = level;
      level = HOLT_ALPHA * values[i] + (1 - HOLT_ALPHA) * (level + trend);
      trend = HOLT_BETA * (level - previous) + (1 - HOLT_BETA) * trend;
    }
    return Array.from({ length: horizon }, (_, i) => level + (i + 1) * trend);
  }

  if (model === "holtWinters") {
    if (n < 2 * period) throw Error("Holt-Winters needs at least two full seasons.");
    const { level, trend, season } = runHoltWinters(values, period, weights ?? fitHoltWinters(values, period));
    return Array.from({ length: horizon }, (_, i) => level + (i + 1) * trend + season[(n + i) % period]);
  }

  throw Error("Unknown model.");
}

/**
 * Additive Holt-Winters: level + trend + a repeating seasonal offset, initialized from the
 * first two seasons. Also returns the one-step-ahead squared error used to tune the weights.
 */
function runHoltWinters(values, period, { alpha, beta, gamma }) {
  const first = mean(values.slice(0, period));
  let level = first;
  let trend = (mean(values.slice(period, 2 * period)) - first) / period;
  const season = values.slice(0, period).map((v) => v - first);
  let sse = 0;
  for (let t = period; t < values.length; t++) {
    const s = season[t % period];
    sse += (values[t] - (level + trend + s)) ** 2;
    const previous = level;
    level = alpha * (values[t] - s) + (1 - alpha) * (level + trend);
    trend = beta * (level - previous) + (1 - beta) * trend;
    season[t % period] = gamma * (values[t] - level) + (1 - gamma) * s;
  }
  return { level, trend, season, sse };
}

/** Grid-search the smoothing weights that best predict each next step of the series it is given. */
export function fitHoltWinters(values, period) {
  if (values.length < 2 * period) return null;
  let best = null;
  for (const alpha of HW_GRID.alpha) {
    for (const beta of HW_GRID.beta) {
      for (const gamma of HW_GRID.gamma) {
        const { sse } = runHoltWinters(values, period, { alpha, beta, gamma });
        if (!best || sse < best.sse) best = { alpha, beta, gamma, sse };
      }
    }
  }
  return best;
}

/**
 * Train each model on the first 80% of the series and score it on the last 20%.
 * Results are sorted by MAE; `backtest` holds each model's holdout predictions.
 */
export function evaluate(values, period = 7, weights) {
  const cut = Math.floor(values.length * (1 - HOLDOUT_SHARE));
  const train = values.slice(0, cut);
  const test = values.slice(cut);
  return Object.keys(names)
    .filter((model) => train.length >= (seasonsNeeded[model] || 0) * period)
    .map((model) => {
      // Auto-tuning only ever sees the training slice, so the holdout stays unseen.
      const used = model === "holtWinters" ? weights ?? fitHoltWinters(train, period) : undefined;
      const backtest = predict(train, test.length, model, period, used);
      const errors = test.map((value, i) => value - backtest[i]);
      return {
        model,
        mae: errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length,
        rmse: Math.sqrt(errors.reduce((sum, e) => sum + e * e, 0) / errors.length),
        holdout: test.length,
        start: cut,
        backtest,
        ...(used && { weights: { alpha: used.alpha, beta: used.beta, gamma: used.gamma } }),
      };
    })
    .sort((a, b) => a.mae - b.mae);
}

export function parseCSV(text) {
  const lines = text.replace(/^﻿/, "").trim().split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 22) throw Error("Upload at least 21 observations plus the date,value header.");
  if (lines.length > 5001) throw Error("Please use at most 5,000 observations.");

  const header = lines.shift().toLowerCase().replace(/\s/g, "");
  if (header !== "date,value") throw Error("The first row must be date,value.");

  const rows = lines.map((line, i) => {
    const cells = line.split(",").map((cell) => cell.trim().replace(/^"|"$/g, ""));
    if (cells.length !== 2 || !/^\d{4}-\d{2}-\d{2}$/.test(cells[0]) || !cells[1]) {
      throw Error(`Row ${i + 2}: use YYYY-MM-DD and a numeric value.`);
    }
    const time = Date.parse(`${cells[0]}T00:00:00Z`);
    const value = Number(cells[1]);
    if (!Number.isFinite(time) || new Date(time).toISOString().slice(0, 10) !== cells[0] || !Number.isFinite(value)) {
      throw Error(`Row ${i + 2}: invalid date or value.`);
    }
    return { date: cells[0], time, value };
  });

  const step = rows[1].time - rows[0].time;
  if (step <= 0) throw Error("Dates must be unique and in increasing order.");
  if (rows.some((row, i) => i > 0 && row.time - rows[i - 1].time !== step)) {
    throw Error("Dates must be evenly spaced. Fill missing dates before uploading.");
  }
  return rows;
}

export const stepDays = (rows) => Math.round((rows[1].time - rows[0].time) / DAY);
