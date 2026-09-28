// Baseline forecasting models, chronological holdout scoring, and CSV parsing.

export const names = {
  trend: "Linear trend",
  seasonal: "Seasonal naive",
  holt: "Holt’s linear smoothing",
};

const HOLDOUT_SHARE = 0.2;
const HOLT_ALPHA = 0.35; // level smoothing
const HOLT_BETA = 0.15; // trend smoothing
const DAY = 86_400_000;

export function predict(values, horizon, model, period = 7) {
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

  throw Error("Unknown model.");
}

/**
 * Train each model on the first 80% of the series and score it on the last 20%.
 * Results are sorted by MAE; `backtest` holds each model's holdout predictions.
 */
export function evaluate(values, period = 7) {
  const cut = Math.floor(values.length * (1 - HOLDOUT_SHARE));
  const train = values.slice(0, cut);
  const test = values.slice(cut);
  return Object.keys(names)
    .filter((model) => model !== "seasonal" || train.length >= period)
    .map((model) => {
      const backtest = predict(train, test.length, model, period);
      const errors = test.map((value, i) => value - backtest[i]);
      return {
        model,
        mae: errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length,
        rmse: Math.sqrt(errors.reduce((sum, e) => sum + e * e, 0) / errors.length),
        holdout: test.length,
        start: cut,
        backtest,
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

/** Reproducible synthetic daily series: 84 days with weekly seasonality and deterministic noise. */
export function sample(kind) {
  return Array.from({ length: 84 }, (_, i) => {
    const time = Date.UTC(2026, 0, 1 + i);
    const weekly = Math.sin((i * 2 * Math.PI) / 7);
    const wave = Math.sin(i * 1.91) * 3 + Math.cos(i * 0.63) * 2;
    const value =
      kind === "demand" ? 180 + i * 0.7 + 32 * weekly + wave
      : kind === "energy" ? 420 + 85 * weekly + wave * 4
      : 1200 + i * 12 + 140 * weekly + wave * 10;
    return { time, date: new Date(time).toISOString().slice(0, 10), value: Math.round(value * 100) / 100 };
  });
}

export const stepDays = (rows) => Math.round((rows[1].time - rows[0].time) / DAY);
