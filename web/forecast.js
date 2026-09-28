// Forecasting models, chronological holdout scoring, and CSV parsing. No dependencies.

/** Every model, in display order. `about` is shown in the leaderboard. */
export const models = [
  { id: "naive", name: "Naive (last value)", family: "Baselines", about: "Repeats the most recent value." },
  { id: "mean", name: "Historical mean", family: "Baselines", about: "Forecasts the average of all history." },
  { id: "drift", name: "Random walk with drift", family: "Baselines", about: "Extends the line from the first value to the last." },
  { id: "movingAverage", name: "Moving average", family: "Baselines", about: "Averages the most recent season of values." },
  { id: "seasonal", name: "Seasonal naive", family: "Seasonal baselines", about: "Repeats the last full season." },
  { id: "seasonalDrift", name: "Seasonal naive + drift", family: "Seasonal baselines", about: "Repeats the last season, shifted by the average season-over-season change." },
  { id: "trend", name: "Linear trend", family: "Regression", about: "Least-squares straight line through time." },
  { id: "seasonalTrend", name: "Trend + seasonal means", family: "Regression", about: "Linear trend plus the average offset at each point in the season." },
  { id: "autoregression", name: "Seasonal autoregression", family: "Regression", about: "Least-squares fit on the previous two values and the value one season ago." },
  { id: "ses", name: "Simple exponential smoothing", family: "Exponential smoothing", about: "A weighted average that favors recent values; weight tuned on history." },
  { id: "holt", name: "Holt’s linear smoothing", family: "Exponential smoothing", about: "A smoothed level plus a smoothed trend." },
  { id: "damped", name: "Damped trend", family: "Exponential smoothing", about: "Holt’s method with a trend that levels off; weights tuned on history." },
  { id: "holtWinters", name: "Holt-Winters seasonal", family: "Exponential smoothing", about: "Level, trend, and an additive seasonal pattern; weights tunable." },
  { id: "holtWintersMult", name: "Holt-Winters multiplicative", family: "Exponential smoothing", about: "Like Holt-Winters, but the seasonal swing scales with the level." },
  { id: "theta", name: "Theta method", family: "Exponential smoothing", about: "Seasonally adjusted smoothing plus half the long-run trend; a classic competition winner." },
];
export const names = Object.fromEntries(models.map((m) => [m.id, m.name]));

const HOLDOUT_SHARE = 0.2;
const HOLT_ALPHA = 0.35; // Holt level smoothing
const HOLT_BETA = 0.15; // Holt trend smoothing
const DAY = 86_400_000;

// Smoothing weights for the tuned models are chosen per series by grid search on one-step-ahead error.
const GRID = {
  alpha: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
  beta: [0, 0.02, 0.05, 0.1, 0.2],
  gamma: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  phi: [0.8, 0.9, 0.95, 0.98],
  ses: Array.from({ length: 19 }, (_, i) => (i + 1) / 20),
};

const mean = (list) => list.reduce((sum, v) => sum + v, 0) / list.length;
const repeat = (value, horizon) => Array.from({ length: horizon }, () => value);

/** Whether a model has enough (and suitable) history to run. */
function available(model, values, period) {
  const n = values.length;
  if (model === "seasonal") return n >= period;
  if (["seasonalDrift", "seasonalTrend", "holtWinters"].includes(model)) return n >= 2 * period;
  if (model === "holtWintersMult") return n >= 2 * period && values.every((v) => v > 0);
  if (model === "autoregression") return n >= period + 20;
  return n >= 3;
}

/**
 * Forecast `horizon` steps. `weights` ({ alpha, beta, gamma }) fixes the additive Holt-Winters
 * smoothing; omit it to auto-tune on `values`.
 */
export function predict(values, horizon, model, period = 7, weights) {
  if (values.length < 3 || !values.every(Number.isFinite) || !Number.isInteger(horizon) || horizon < 1) {
    throw Error("A valid numeric series and positive horizon are required.");
  }
  if (!models.some((m) => m.id === model)) throw Error("Unknown model.");
  if (!available(model, values, period)) throw Error(`${names[model]} needs more history for this season length.`);
  const n = values.length;
  const steps = (fn) => Array.from({ length: horizon }, (_, i) => fn(i));

  switch (model) {
    case "naive":
      return repeat(values[n - 1], horizon);
    case "mean":
      return repeat(mean(values), horizon);
    case "drift": {
      const slope = (values[n - 1] - values[0]) / (n - 1);
      return steps((i) => values[n - 1] + (i + 1) * slope);
    }
    case "movingAverage":
      return repeat(mean(values.slice(-Math.max(2, Math.min(n, period)))), horizon);
    case "seasonal":
      return steps((i) => values[n - period + (i % period)]);
    case "seasonalDrift": {
      const drift = mean(values.slice(period).map((v, t) => v - values[t]));
      return steps((i) => values[n - period + (i % period)] + (Math.floor(i / period) + 1) * drift);
    }
    case "trend": {
      const { intercept, slope } = linearFit(values);
      return steps((i) => intercept + slope * (n + i));
    }
    case "seasonalTrend": {
      const { intercept, slope } = linearFit(values);
      const offsets = seasonalOffsets(values.map((v, t) => v - (intercept + slope * t)), period);
      return steps((i) => intercept + slope * (n + i) + offsets[(n + i) % period]);
    }
    case "autoregression":
      return autoregression(values, horizon, period);
    case "ses":
      return repeat(runSES(values, fitSES(values)).level, horizon);
    case "holt": {
      let level = values[0];
      let trend = values[1] - values[0];
      for (let t = 1; t < n; t++) {
        const previous = level;
        level = HOLT_ALPHA * values[t] + (1 - HOLT_ALPHA) * (level + trend);
        trend = HOLT_BETA * (level - previous) + (1 - HOLT_BETA) * trend;
      }
      return steps((i) => level + (i + 1) * trend);
    }
    case "damped": {
      const w = fitDamped(values);
      const { level, trend } = runDamped(values, w);
      let damping = 0;
      return steps(() => level + (damping = damping * w.phi + w.phi) * trend);
    }
    case "holtWinters": {
      const { level, trend, season } = runHoltWinters(values, period, weights ?? fitHoltWinters(values, period));
      return steps((i) => level + (i + 1) * trend + season[(n + i) % period]);
    }
    case "holtWintersMult": {
      const { level, trend, season } = runHoltWintersMult(values, period, fitGrid(values, period, runHoltWintersMult));
      return steps((i) => (level + (i + 1) * trend) * season[(n + i) % period]);
    }
    case "theta":
      return theta(values, horizon, period);
  }
}

/* ---------- Regression helpers ---------- */

function linearFit(values) {
  const n = values.length;
  const xMean = (n - 1) / 2;
  const yMean = mean(values);
  let num = 0;
  let den = 0;
  values.forEach((y, x) => {
    num += (x - xMean) * (y - yMean);
    den += (x - xMean) ** 2;
  });
  const slope = num / den;
  return { slope, intercept: yMean - slope * xMean };
}

/** Average value at each position in the season, centered so the offsets sum to zero. */
function seasonalOffsets(residuals, period) {
  const sums = new Array(period).fill(0);
  const counts = new Array(period).fill(0);
  residuals.forEach((r, t) => {
    sums[t % period] += r;
    counts[t % period] += 1;
  });
  const raw = sums.map((sum, k) => (counts[k] ? sum / counts[k] : 0));
  const center = mean(raw);
  return raw.map((v) => v - center);
}

/** Solve A·x = b with Gaussian elimination and partial pivoting (A is small and square). */
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = col + 1; r < n; r++) {
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let sum = M[r][n];
    for (let c = r + 1; c < n; c++) sum -= M[r][c] * x[c];
    x[r] = sum / M[r][r];
  }
  return x;
}

/** AR model on lags 1, 2, and one season, fitted by least squares on standardized values. */
function autoregression(values, horizon, period) {
  const lags = [...new Set([1, 2, period])].sort((a, b) => a - b);
  const mu = mean(values);
  const sigma = Math.sqrt(mean(values.map((v) => (v - mu) ** 2))) || 1;
  const z = values.map((v) => (v - mu) / sigma);
  const maxLag = lags.at(-1);
  const size = lags.length + 1;
  const XtX = Array.from({ length: size }, () => new Array(size).fill(0));
  const Xty = new Array(size).fill(0);
  for (let t = maxLag; t < z.length; t++) {
    const x = [1, ...lags.map((lag) => z[t - lag])];
    for (let i = 0; i < size; i++) {
      Xty[i] += x[i] * z[t];
      for (let j = 0; j < size; j++) XtX[i][j] += x[i] * x[j];
    }
  }
  for (let i = 1; i < size; i++) XtX[i][i] += 1e-6; // tiny ridge keeps the system well-posed
  const coef = solve(XtX, Xty);
  const series = [...z];
  for (let h = 0; h < horizon; h++) {
    const t = series.length;
    series.push(coef[0] + lags.reduce((sum, lag, i) => sum + coef[i + 1] * series[t - lag], 0));
  }
  return series.slice(-horizon).map((v) => v * sigma + mu);
}

/* ---------- Exponential smoothing ---------- */

function runSES(values, alpha) {
  let level = values[0];
  let sse = 0;
  for (let t = 1; t < values.length; t++) {
    sse += (values[t] - level) ** 2;
    level = alpha * values[t] + (1 - alpha) * level;
  }
  return { level, sse };
}

function fitSES(values) {
  let best = { alpha: GRID.ses[0], sse: Infinity };
  for (const alpha of GRID.ses) {
    const { sse } = runSES(values, alpha);
    if (sse < best.sse) best = { alpha, sse };
  }
  return best.alpha;
}

function runDamped(values, { alpha, beta, phi }) {
  let level = values[0];
  let trend = values[1] - values[0];
  let sse = 0;
  for (let t = 1; t < values.length; t++) {
    sse += (values[t] - (level + phi * trend)) ** 2;
    const previous = level;
    level = alpha * values[t] + (1 - alpha) * (level + phi * trend);
    trend = beta * (level - previous) + (1 - beta) * phi * trend;
  }
  return { level, trend, sse };
}

function fitDamped(values) {
  let best = null;
  for (const alpha of GRID.alpha) {
    for (const beta of GRID.beta) {
      for (const phi of GRID.phi) {
        const { sse } = runDamped(values, { alpha, beta, phi });
        if (!best || sse < best.sse) best = { alpha, beta, phi, sse };
      }
    }
  }
  return best;
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

/** Multiplicative Holt-Winters: the seasonal pattern is a ratio, so swings grow with the level. */
function runHoltWintersMult(values, period, { alpha, beta, gamma }) {
  const first = mean(values.slice(0, period));
  let level = first;
  let trend = (mean(values.slice(period, 2 * period)) - first) / period;
  const season = values.slice(0, period).map((v) => v / first);
  let sse = 0;
  for (let t = period; t < values.length; t++) {
    const s = season[t % period];
    sse += (values[t] - (level + trend) * s) ** 2;
    const previous = level;
    level = alpha * (values[t] / s) + (1 - alpha) * (level + trend);
    trend = beta * (level - previous) + (1 - beta) * trend;
    season[t % period] = gamma * (values[t] / level) + (1 - gamma) * s;
  }
  return { level, trend, season, sse };
}

function fitGrid(values, period, run) {
  let best = null;
  for (const alpha of GRID.alpha) {
    for (const beta of GRID.beta) {
      for (const gamma of GRID.gamma) {
        const { sse } = run(values, period, { alpha, beta, gamma });
        if (Number.isFinite(sse) && (!best || sse < best.sse)) best = { alpha, beta, gamma, sse };
      }
    }
  }
  return best;
}

/** Grid-search the additive Holt-Winters weights that best predict each next step of `values`. */
export function fitHoltWinters(values, period) {
  if (values.length < 2 * period) return null;
  return fitGrid(values, period, runHoltWinters);
}

/**
 * Theta method (Hyndman & Billah form): simple exponential smoothing plus half the
 * least-squares trend, on seasonally adjusted data when there are two full seasons.
 */
function theta(values, horizon, period) {
  const n = values.length;
  const seasonal = period > 1 && n >= 2 * period;
  const { intercept, slope } = linearFit(values);
  const offsets = seasonal ? seasonalOffsets(values.map((v, t) => v - (intercept + slope * t)), period) : null;
  const adjusted = seasonal ? values.map((v, t) => v - offsets[t % period]) : values;
  const b = linearFit(adjusted).slope;
  const alpha = fitSES(adjusted);
  const { level } = runSES(adjusted, alpha);
  const tail = 1 / alpha - (1 - alpha) ** n / alpha;
  return Array.from({ length: horizon }, (_, i) => level + (b / 2) * (i + tail) + (seasonal ? offsets[(n + i) % period] : 0));
}

/* ---------- Evaluation ---------- */

/**
 * Train each model on the first 80% of the series and score it on the last 20%.
 * Results are sorted by MAE; `backtest` holds each model's holdout predictions.
 */
export function evaluate(values, period = 7, weights) {
  const cut = Math.floor(values.length * (1 - HOLDOUT_SHARE));
  const train = values.slice(0, cut);
  const test = values.slice(cut);
  return models
    .map((m) => m.id)
    .filter((model) => available(model, train, period))
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
    .filter((score) => Number.isFinite(score.mae))
    .sort((a, b) => a.mae - b.mae);
}

/* ---------- Data ---------- */

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
