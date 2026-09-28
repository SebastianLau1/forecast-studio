import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { predict, evaluate, fitHoltWinters, parseCSV, stepDays } from "../web/forecast.js";

const data = (name) => readFile(new URL(`../web/data/${name}`, import.meta.url), "utf8");
const toCSV = (rows) => `date,value\n${rows.map((row) => `${row.date},${row.value}`).join("\n")}`;

test("linear trend extrapolates an exact line", () => {
  assert.deepEqual(predict([2, 4, 6, 8], 3, "trend"), [10, 12, 14]);
});

test("seasonal baseline repeats the final season", () => {
  assert.deepEqual(predict([1, 3, 2, 1, 3, 2], 5, "seasonal", 3), [1, 3, 2, 1, 3]);
});

test("holdout favors the exact linear signal", () => {
  const result = evaluate(Array.from({ length: 40 }, (_, i) => i * 3 + 2));
  assert.equal(result[0].model, "trend");
  assert.equal(result[0].mae, 0);
  assert.equal(result[0].holdout, 8);
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

const trendingWeeks = Array.from({ length: 84 }, (_, i) => 100 + 0.5 * i + [8, 6, 5, 6, 7, -14, -18][i % 7]);

test("auto-tuned Holt-Winters wins on a trending weekly series and reports its weights", () => {
  const scores = evaluate(trendingWeeks, 7);
  assert.equal(scores[0].model, "holtWinters");
  assert.ok(scores[0].mae < 0.5);
  for (const key of ["alpha", "beta", "gamma"]) assert.ok(scores[0].weights[key] >= 0 && scores[0].weights[key] <= 1, key);
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
    const models = evaluate(rows.map((row) => row.value), meta.season).map((r) => r.model);
    assert.ok(models.includes("seasonal"), `${meta.id} has enough history for a ${meta.season}-step season`);
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
