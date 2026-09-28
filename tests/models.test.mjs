import test from "node:test";
import assert from "node:assert/strict";
import { predict, evaluate, parseCSV, sample, stepDays } from "../web/forecast.js";

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

test("CSV rejects missing dates, duplicate dates, empty and nonfinite values", () => {
  const rows = sample("demand").slice(0, 25);
  assert.equal(parseCSV(toCSV(rows)).length, 25);
  assert.throws(() => parseCSV(toCSV(rows.filter((_, i) => i !== 8))), /evenly/);
  assert.throws(() => parseCSV(toCSV(rows.map((r, i) => (i === 1 ? rows[0] : r)))), /unique/);
  assert.throws(() => parseCSV(toCSV(rows.map((r, i) => (i === 0 ? { ...r, value: "Infinity" } : r)))), /invalid/);
  assert.throws(() => parseCSV("date,value\n2026-01-01,2"), /21 observations/);
});

test("step size is reported in days", () => {
  assert.equal(stepDays(sample("energy")), 1);
});
