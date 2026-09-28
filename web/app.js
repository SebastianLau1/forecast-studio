import { names, predict, evaluate, parseCSV, stepDays } from "./forecast.js";

const $ = (id) => document.getElementById(id);
const MAX_UPLOAD_BYTES = 500_000;
const COLORS = { observed: "#0f1115", backtest: "#2f5bff", forecast: "#ff4f1f" };

const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const oneDecimal = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const compactPrecise = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 });
const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
// Millions read as "4.26M", thousands as whole numbers ("1,769"), small values keep a decimal ("424.3").
const fmt = (n) => (Math.abs(n) >= 1_000_000 ? compactPrecise.format(n) : Math.abs(n) >= 1000 ? whole.format(n) : number.format(n));
const err = (n) => (Math.abs(n) >= 1000 ? fmt(n) : oneDecimal.format(n));
const axis = (n) => (Math.abs(n) >= 10_000 ? compact.format(n) : number.format(n));
const shortDate = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
const monthYear = new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
const longDate = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });

const state = {
  rows: [],
  title: "",
  meta: null, // metadata for a bundled dataset; null for uploads
  datasets: [],
  cache: new Map(),
  uploaded: false,
  model: "auto",
  result: null,
  animate: true,
  chart: null,
  chartWidth: 0,
};

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

function download(text, name) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const link = el("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function cadence(rows) {
  const days = stepDays(rows);
  if (days === 1) return "Daily";
  if (days === 7) return "Weekly";
  return `Every ${days} days`;
}

function periodWord(count) {
  const days = state.rows.length > 1 ? stepDays(state.rows) : 1;
  const word = days === 1 ? "day" : days === 7 ? "week" : "period";
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/* ---------- Forecast ---------- */

function run() {
  const started = performance.now();
  try {
    const horizon = Number($("horizon").value);
    const period = Number($("period").value);
    const { rows } = state;
    const values = rows.map((row) => row.value);
    const scores = evaluate(values, period);
    const model = state.model === "auto" ? scores[0].model : state.model;
    const score = scores.find((s) => s.model === model);
    if (!score) throw Error("This season length needs more training data. Pick a shorter season or another model.");

    const step = rows[1].time - rows[0].time;
    const future = predict(values, horizon, model, period).map((value, i) => {
      const time = rows.at(-1).time + (i + 1) * step;
      return {
        time,
        date: new Date(time).toISOString().slice(0, 10),
        value,
        spread: 1.96 * score.rmse * Math.sqrt(1 + (i + 1) / score.holdout),
      };
    });

    state.result = { values, scores, model, score, future, horizon };
    renderSummary();
    renderChart();
    renderLeaderboard();
    renderInsight();
    state.animate = false;
    status(`Forecast updated in ${Math.max(1, Math.round(performance.now() - started))} ms, computed in your browser.`);
  } catch (error) {
    status(error.message, true);
  }
}

function renderSummary() {
  const { values, score, model, future, horizon } = state.result;
  const last = values.at(-1);
  const next = future[0].value;
  const change = last !== 0 ? (next / last - 1) * 100 : null;

  $("next").textContent = fmt(next);
  $("change").textContent = change === null ? "Versus a zero baseline" : `${change >= 0 ? "+" : ""}${change.toFixed(1)}% vs last observation`;
  $("change").className = change === null ? "" : change >= 0 ? "up" : "down";
  $("end").textContent = fmt(future.at(-1).value);
  $("end-date").textContent = `${longDate.format(future.at(-1).time)} · ${periodWord(horizon)} out`;
  $("mae").textContent = err(score.mae);
  $("mae-note").textContent = `RMSE ${err(score.rmse)} · lower is better`;
  $("observations").textContent = number.format(state.rows.length);
  $("data-label").textContent = state.meta ? `Real data · ${state.meta.unit}` : "Your upload, kept in this tab";

  $("chart-title").textContent = state.title;
  $("chart-meta").textContent = `${cadence(state.rows)} · ${state.meta ? state.meta.unit : "uploaded data"}`;
  renderSource();
  $("range").textContent = `${longDate.format(state.rows[0].time)} → ${longDate.format(future.at(-1).time)}`;
  $("selected-model").textContent = `${names[model]}${state.model === "auto" ? " · auto-selected" : ""}`;
  $("holdout-meta").textContent = `Last ${score.holdout} obs. held out`;
}

function renderSource() {
  const source = $("source");
  if (!state.meta) {
    source.textContent = "Your uploaded data. It never leaves this tab.";
    return;
  }
  const { note, source: name, sourceUrl, license, retrieved } = state.meta;
  const link = el("a", name);
  link.href = sourceUrl;
  link.target = "_blank";
  link.rel = "noreferrer";
  source.replaceChildren(`${note} Source: `, link, ` (${license}), retrieved ${longDate.format(Date.parse(retrieved))}.`);
}

/* ---------- Chart ---------- */

function niceTicks(lo, hi, count) {
  const raw = (hi - lo) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw);
  const ticks = [];
  for (let k = Math.ceil(lo / step); k * step <= hi; k++) ticks.push(k * step);
  return ticks;
}

function renderChart() {
  const { values, score, future } = state.result;
  const box = $("chart");
  const width = Math.max(300, box.clientWidth);
  const small = width < 640;
  const height = small ? 270 : 390;
  const pad = { l: 52, r: small ? 46 : 78, t: 30, b: 30 };
  const n = values.length;
  const total = n + future.length - 1;

  const all = [...values, ...score.backtest, ...future.flatMap((f) => [f.value - f.spread, f.value + f.spread])];
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  const margin = (hi - lo || 1) * 0.08;
  lo -= margin;
  hi += margin;

  const x = (i) => pad.l + (i / total) * (width - pad.l - pad.r);
  const y = (v) => height - pad.b - ((v - lo) / (hi - lo)) * (height - pad.t - pad.b);
  const points = (pairs) => pairs.map(([i, v]) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const base = height - pad.b;
  const dateAt = (i) => (i < n ? state.rows[i].time : future[i - n].time);
  const draw = state.animate ? " draw" : "";
  const fade = state.animate ? " fade" : "";

  const observed = points(values.map((v, i) => [i, v]));
  const backtest = points(score.backtest.map((v, i) => [score.start + i, v]));
  const forecast = points([[n - 1, values.at(-1)], ...future.map((f, i) => [n + i, f.value])]);
  const band = points([
    [n - 1, values.at(-1)],
    ...future.map((f, i) => [n + i, f.value + f.spread]),
    ...future.map((f, i) => [n + i, f.value - f.spread]).reverse(),
  ]);

  let svg = `<svg viewBox="0 0 ${width} ${height}" height="${height}" xmlns="http://www.w3.org/2000/svg">
    <defs><linearGradient id="observedFill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="${COLORS.observed}" stop-opacity=".08"/><stop offset="1" stop-color="${COLORS.observed}" stop-opacity="0"/></linearGradient></defs>
    <rect class="holdout-zone" x="${x(score.start)}" y="${pad.t}" width="${x(n - 1) - x(score.start)}" height="${base - pad.t}"/>
    ${x(n - 1) - x(score.start) > 84 ? `<text class="zone-label" x="${x(score.start) + 7}" y="${pad.t + 15}">HOLDOUT</text>` : ""}
    <g class="grid">`;
  for (const tick of niceTicks(lo, hi, small ? 4 : 5)) {
    svg += `<line x1="${pad.l}" x2="${width - pad.r}" y1="${y(tick)}" y2="${y(tick)}"/><text x="${pad.l - 10}" y="${y(tick) + 4}" text-anchor="end">${axis(tick)}</text>`;
  }
  svg += `</g>
    <g class="series${draw}">
      <polygon class="area" points="${x(0)},${base} ${observed} ${x(n - 1)},${base}"/>
      <polyline class="observed" points="${observed}"/>
    </g>
    <polyline class="backtest${fade}" points="${backtest}"/>
    <line class="now" x1="${x(n - 1)}" x2="${x(n - 1)}" y1="${pad.t}" y2="${base}"/>
    <text class="now-label" x="${x(n - 1) + 7}" y="${pad.t + 15}">FORECAST →</text>
    <g class="series${fade}">
      <polygon class="band" points="${band}"/>
      <polyline class="forecast" points="${forecast}"/>
      <circle class="end-dot" cx="${x(total)}" cy="${y(future.at(-1).value)}" r="5"/>
      <text class="end-label" x="${x(total) + 10}" y="${y(future.at(-1).value) + 4}">${axis(future.at(-1).value)}</text>
    </g>`;

  // Date labels at start, middle, last observation, and horizon end, skipping any that would collide.
  const candidates = [[0, "start"], [Math.floor((n - 1) / 2), "middle"], [n - 1, "middle"], [total, "end"]];
  const labels = [candidates[0]];
  for (const candidate of candidates.slice(1, -1)) {
    if (x(candidate[0]) - x(labels.at(-1)[0]) > 70 && x(total) - x(candidate[0]) > 70) labels.push(candidate);
  }
  labels.push(candidates.at(-1));
  const dateLabel = dateAt(total) - dateAt(0) > 300 * 86_400_000 ? monthYear : shortDate;
  for (const [i, anchor] of labels) {
    svg += `<text x="${x(i)}" y="${height - 8}" text-anchor="${anchor}">${dateLabel.format(dateAt(i))}</text>`;
  }
  svg += `<line class="cursor" id="cursor" y1="${pad.t}" y2="${base}" visibility="hidden"/><g id="cursor-dots"></g></svg>`;

  box.innerHTML = svg;
  state.chart = { x, y, n, total, pad, width, dateAt };
  state.chartWidth = box.clientWidth;
}

function showTooltip(event) {
  if (!state.chart || !state.result) return;
  const { x, y, n, total, pad, width, dateAt } = state.chart;
  const { values, score, future } = state.result;
  const bounds = $("chart").getBoundingClientRect();
  const px = event.clientX - bounds.left;
  const i = Math.max(0, Math.min(total, Math.round(((px - pad.l) / (width - pad.l - pad.r)) * total)));

  const rows = [];
  if (i < n) rows.push(["Observed", values[i], COLORS.observed]);
  if (i >= score.start && i < n) rows.push(["Backtest", score.backtest[i - score.start], COLORS.backtest]);
  if (i >= n) {
    const f = future[i - n];
    rows.push(["Forecast", f.value, COLORS.forecast]);
    rows.push(["Range", `${fmt(f.value - f.spread)} to ${fmt(f.value + f.spread)}`, null]);
  }

  const cursor = $("cursor");
  cursor.setAttribute("x1", x(i));
  cursor.setAttribute("x2", x(i));
  cursor.setAttribute("visibility", "visible");
  $("cursor-dots").innerHTML = rows
    .filter(([, value, color]) => color && typeof value === "number")
    .map(([, value, color]) => `<circle class="cursor-dot" cx="${x(i)}" cy="${y(value)}" r="4.5" fill="${color}"/>`)
    .join("");

  const tip = $("tooltip");
  tip.replaceChildren(el("b", longDate.format(dateAt(i))), ...rows.map(([label, value, color]) => {
    const row = el("div");
    const name = el("span");
    if (color) {
      const dot = el("i");
      dot.style.background = color;
      name.append(dot);
    }
    name.append(label);
    row.append(name, el("em", typeof value === "number" ? fmt(value) : value));
    return row;
  }));
  tip.hidden = false;
  const left = x(i) + 16;
  tip.style.left = `${left + tip.offsetWidth > width ? x(i) - 16 - tip.offsetWidth : left}px`;
  tip.style.top = `${pad.t}px`;
}

function hideTooltip() {
  $("tooltip").hidden = true;
  $("cursor")?.setAttribute("visibility", "hidden");
  const dots = $("cursor-dots");
  if (dots) dots.innerHTML = "";
}

/* ---------- Leaderboard and narrative ---------- */

function renderLeaderboard() {
  const { scores, model } = state.result;
  const worst = Math.max(...scores.map((s) => s.mae)) || 1;
  $("comparison").replaceChildren(...scores.map((s, rank) => {
    const item = el("li");
    const button = el("button", undefined, s.model === model ? "selected" : "");
    button.type = "button";
    button.setAttribute("aria-label", `Forecast with ${names[s.model]}: MAE ${err(s.mae)}, RMSE ${err(s.rmse)}`);
    const name = el("span", names[s.model], "name");
    const badge = s.model === model ? (state.model === "auto" ? "Selected · auto" : "Selected") : rank === 0 ? "Lowest MAE" : "";
    if (badge) name.append(el("span", badge, "badge"));
    const bar = el("span", undefined, "bar");
    const fill = el("i");
    fill.style.width = `${Math.max(4, (s.mae / worst) * 100)}%`;
    if (!state.animate) fill.style.animation = "none";
    bar.append(fill);
    const mae = el("span", err(s.mae), "num");
    mae.append(el("small", "MAE"));
    const rmse = el("span", err(s.rmse), "num");
    rmse.append(el("small", "RMSE"));
    button.append(el("span", String(rank + 1).padStart(2, "0"), "rank"), name, bar, mae, rmse);
    button.onclick = () => selectModel(s.model);
    item.append(button);
    return item;
  }));
}

function renderInsight() {
  const { values, scores, model, future, horizon } = state.result;
  const [best, runner] = scores;
  const end = future.at(-1).value;
  const last = values.at(-1);
  const parts = [
    { b: names[best.model] },
    ` had the lowest error on the ${best.holdout} held-out observations (MAE ${err(best.mae)}`,
    runner ? `, ${Math.max(0, (1 - best.mae / runner.mae) * 100).toFixed(0)}% below ${names[runner.model]}). ` : "). ",
  ];
  if (model !== best.model) parts.push(`You’re forecasting with `, { b: names[model] }, " instead. ");
  parts.push(
    `Refit on all ${number.format(values.length)} observations, it projects `,
    { b: fmt(end) },
    ` at period ${horizon}`,
    trendPhrase(last, end),
    "Treat it as a baseline, not a guarantee.",
  );
  $("insight").replaceChildren(...parts.map((part) => (typeof part === "string" ? part : el("b", part.b))));
}

function trendPhrase(last, end) {
  if (last === 0) return ". ";
  const change = (end / last - 1) * 100;
  if (Math.abs(change) < 0.05) return ", level with the last value. ";
  return `, ${change > 0 ? "up" : "down"} ${Math.abs(change).toFixed(1)}% from the last value. `;
}

/* ---------- Inputs ---------- */

function selectModel(model) {
  state.model = model;
  document.querySelectorAll(".segmented button").forEach((button) => {
    button.setAttribute("aria-checked", String(button.dataset.model === model));
  });
  state.animate = true;
  run();
}

async function loadFile(file) {
  if (!file) return;
  try {
    if (file.size > MAX_UPLOAD_BYTES) throw Error("CSV must be smaller than 500 KB.");
    state.rows = parseCSV(await file.text());
    state.title = file.name.replace(/\.csv$/i, "");
    state.meta = null;
    state.uploaded = true;
    state.animate = true;
    const option = $("dataset").querySelector('[value="upload"]');
    option.hidden = false;
    option.disabled = false;
    option.textContent = `Your upload · ${file.name}`;
    $("dataset").value = "upload";
    $("horizon-value").textContent = periodWord(Number($("horizon").value));
    run();
  } catch (error) {
    status(error.message, true);
  }
}

async function loadDataset(id) {
  const meta = state.datasets.find((d) => d.id === id);
  if (!meta) return;
  try {
    if (!state.cache.has(id)) {
      const response = await fetch(`data/${meta.file}`);
      if (!response.ok) throw Error(`Could not load ${meta.title} (HTTP ${response.status}).`);
      state.cache.set(id, parseCSV(await response.text()));
    }
    state.rows = state.cache.get(id);
    state.title = meta.title;
    state.meta = meta;
    state.uploaded = false;
    state.animate = true;
    $("period").value = String(meta.season);
    $("horizon-value").textContent = periodWord(Number($("horizon").value));
    run();
  } catch (error) {
    status(error.message, true);
  }
}

async function init() {
  try {
    const response = await fetch("data/datasets.json");
    if (!response.ok) throw Error(`Could not load datasets (HTTP ${response.status}).`);
    state.datasets = await response.json();
    const select = $("dataset");
    select.querySelector('option[value=""]')?.remove();
    const upload = select.querySelector('[value="upload"]');
    for (const meta of state.datasets) {
      const option = el("option", `${meta.title} · ${meta.cadence}`);
      option.value = meta.id;
      select.insertBefore(option, upload);
    }
    select.value = state.datasets[0].id;
    await loadDataset(state.datasets[0].id);
  } catch (error) {
    status(`${error.message} You can still upload a CSV.`, true);
  }
}

$("dataset").onchange = () => {
  const id = $("dataset").value;
  if (id !== "upload") loadDataset(id);
};

document.querySelectorAll(".segmented button").forEach((button) => {
  button.onclick = () => selectModel(button.dataset.model);
});

$("horizon").oninput = () => {
  $("horizon-value").textContent = periodWord(Number($("horizon").value));
  if (state.rows.length) run();
};

$("period").onchange = () => {
  state.animate = true;
  if (state.rows.length) run();
};

$("upload-button").onclick = () => $("upload").click();
$("upload").onchange = (event) => {
  loadFile(event.target.files[0]);
  event.target.value = "";
};

window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer?.files[0];
  if (file) loadFile(file);
});

$("sample-csv").onclick = () => {
  if (!state.rows.length) return;
  const name = state.meta ? state.meta.file : "series.csv";
  download(`date,value\n${state.rows.map((row) => `${row.date},${row.value}`).join("\n")}\n`, name);
};

$("export").onclick = () => {
  if (!state.result) return;
  const { future, model } = state.result;
  download(
    "date,forecast,lower_heuristic,upper_heuristic,model\n" +
      future.map((f) => [f.date, f.value.toFixed(3), (f.value - f.spread).toFixed(3), (f.value + f.spread).toFixed(3), model].join(",")).join("\n"),
    "forecast.csv",
  );
};

const chart = $("chart");
chart.addEventListener("pointermove", showTooltip);
chart.addEventListener("pointerdown", showTooltip);
chart.addEventListener("pointerleave", hideTooltip);

new ResizeObserver(() => {
  if (!state.result || chart.clientWidth === state.chartWidth) return;
  hideTooltip();
  renderChart();
}).observe(chart);

init();
