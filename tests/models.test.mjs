import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { models, predict, evaluate, fitHoltWinters, parseCSV, stepDays } from "../web/forecast.js";

const data = (name) => readFile(new URL(`../web/data/${name}`, import.meta.url), "utf8");
const toCSV = (rows) => `date,value\n${rows.map((row) => `${row.date},${row.value}`).join("\n")}`;
const pattern = [8, 6, 5, 6, 7, -14, -18];
const trendingWeeks = Array.from({ length: 84 }, (_, i) => 100 + 0.5 * i + pattern[i % 7]);

test("linear trend extrapolates an exact line", () => {
  assert.deepEqual(predict([2, 4, 6, 8], 3, "trend"), [10, 12, 14]);
});

test("seasonal baseline repeats the final season", () => {
  assert.deepEqual(predict([1, 3, 2, 1, 3, 2], 5, "seasonal", 3), [1, 3, 2, 1, 3]);
});

test("holdout scores an exact linear signal perfectly", () => {
  const result = evaluate(Array.from({ length: 40 }, (_, i) => i * 3 + 2));
  assert.equal(result.find((r) => r.model === "trend").mae, 0);
  assert.equal(result[0].mae, 0);
  assert.equal(result[0].holdout, 8);
});

test("baseline models behave as documented", () => {
  const v = [2, 4, 6, 8, 10];
  assert.deepEqual(predict(v, 2, "naive"), [10, 10]);
  assert.deepEqual(predict(v, 2, "mean"), [6, 6]);
  assert.deepEqual(predict(v, 2, "drift"), [12, 14]);
  assert.deepEqual(predict(v, 1, "movingAverage", 2), [9]);
});

test("seasonal naive + drift is exact on a trending seasonal series", () => {
  const forecast = predict(trendingWeeks, 14, "seasonalDrift", 7);
  forecast.forEach((value, i) => {
    const t = 84 + i;
    assert.ok(Math.abs(value - (100 + 0.5 * t + pattern[t % 7])) < 1e-9, `step ${i + 1}`);
  });
});

test("multiplicative Holt-Winters is skipped for series with zero or negative values", () => {
  const withZero = trendingWeeks.map((v, i) => (i === 3 ? 0 : v));
  assert.ok(!evaluate(withZero, 7).some((s) => s.model === "holtWintersMult"));
  assert.ok(evaluate(trendingWeeks, 7).some((s) => s.model === "holtWintersMult"));
});

test("evaluation exposes each model's holdout backtest for charting", () => {
  const values = Array.from({ length: 40 }, (_, i) => i * 3 + 2);
  const [best] = evaluate(values);
  assert.equal(best.start, 32);
  assert.equal(best.backtest.length, best.holdout);
  assert.deepEqual(best.backtest.map(Math.round), values.slice(32));
});

test("seasonal model is skipped when training data is shorter than the season", () => {
  const models = evaluate(Array.from({ length: 25 }, (_, i) => i), 24).map((r) => r.model);
  assert.ok(!models.includes("seasonal"));
});

test("auto-tuned Holt-Winters fits a trending weekly series and reports its weights", () => {
  const scores = evaluate(trendingWeeks, 7);
  const hw = scores.find((s) => s.model === "holtWinters");
  assert.ok(hw.mae < 0.5);
  for (const key of ["alpha", "beta", "gamma"]) assert.ok(hw.weights[key] >= 0 && hw.weights[key] <= 1, key);
  assert.equal(scores.find((s) => s.model === "trend").weights, undefined);
});

test("manual Holt-Winters weights are used as given", () => {
  const weights = { alpha: 0.9, beta: 0.3, gamma: 0.9 };
  const manual = evaluate(trendingWeeks, 7, weights).find((s) => s.model === "holtWinters");
  const auto = evaluate(trendingWeeks, 7).find((s) => s.model === "holtWinters");
  assert.deepEqual(manual.weights, weights);
  assert.ok(manual.mae > auto.mae, "hand-picked extreme weights do worse than tuned ones");
  assert.deepEqual(predict(trendingWeeks, 3, "holtWinters", 7, weights), predict(trendingWeeks, 3, "holtWinters", 7, weights));
});

test("Holt-Winters needs two full seasons", () => {
  assert.equal(fitHoltWinters([1, 2, 3, 4, 5], 7), null);
  assert.ok(!evaluate(Array.from({ length: 30 }, (_, i) => i), 24).some((s) => s.model === "holtWinters"));
});

test("every bundled real dataset parses, matches its metadata, and supports its season", async () => {
  const datasets = JSON.parse(await data("datasets.json"));
  assert.deepEqual(datasets.map((d) => d.id), ["subway", "wikipedia", "co2"]);
  for (const meta of datasets) {
    const rows = parseCSV(await data(meta.file));
    assert.equal(rows.length, meta.count, meta.id);
    assert.equal(rows[0].date, meta.first, meta.id);
    assert.equal(rows.at(-1).date, meta.last, meta.id);
    assert.equal(stepDays(rows), meta.cadence === "weekly" ? 7 : 1, meta.id);
    assert.ok(rows.every((row) => row.value > 0), `${meta.id} has no gaps or sentinel values`);
    assert.match(meta.sourceUrl, /^https:\/\//);
    assert.ok(meta.pattern.length > 40, `${meta.id} explains its pattern in plain language`);
    const values = rows.map((row) => row.value);
    const scored = evaluate(values, meta.season);
    assert.equal(scored.length, models.length, `all ${models.length} models score ${meta.id}`);
    for (const { id } of models) {
      const forecast = predict(values, 14, id, meta.season);
      assert.ok(forecast.length === 14 && forecast.every(Number.isFinite), `${id} forecasts ${meta.id}`);
    }
  }
});

test("CSV rejects missing dates, duplicate dates, empty and nonfinite values", async () => {
  const rows = parseCSV(await data("subway.csv")).slice(0, 25);
  assert.equal(parseCSV(toCSV(rows)).length, 25);
  assert.throws(() => parseCSV(toCSV(rows.filter((_, i) => i !== 8))), /evenly/);
  assert.throws(() => parseCSV(toCSV(rows.map((r, i) => (i === 1 ? rows[0] : r)))), /unique/);
  assert.throws(() => parseCSV(toCSV(rows.map((r, i) => (i === 0 ? { ...r, value: "Infinity" } : r)))), /invalid/);
  assert.throws(() => parseCSV("date,value\n2026-01-01,2"), /21 observations/);
});
