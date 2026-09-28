// Refresh the real datasets bundled with Forecast Studio.
// Usage: node scripts/fetch-data.mjs   (Node 18+, no dependencies)
//
// Writes web/data/<id>.csv (date,value) and web/data/datasets.json (sources and notes).
import { mkdir, writeFile } from "node:fs/promises";

const OUT = new URL("../web/data/", import.meta.url);
const DAYS = 182; // 26 weeks of daily data
const WEEKS = 260; // 5 years of weekly data
const UA = "ForecastStudio/1.0 (https://github.com/SebastianLau1/forecast-studio)";
const DAY = 86_400_000;

const iso = (time) => new Date(time).toISOString().slice(0, 10);
const compactDate = (time) => iso(time).replaceAll("-", "");

async function get(url, as = "json") {
  const response = await fetch(url, { headers: { "User-Agent": UA } });
  if (!response.ok) throw Error(`${url} returned HTTP ${response.status}`);
  return as === "json" ? response.json() : response.text();
}

function checkSpacing(id, rows, stepDays) {
  for (let i = 1; i < rows.length; i++) {
    const gap = (Date.parse(rows[i].date) - Date.parse(rows[i - 1].date)) / DAY;
    if (gap !== stepDays) throw Error(`${id}: ${rows[i - 1].date} → ${rows[i].date} is ${gap} days, expected ${stepDays}.`);
  }
  if (!rows.every((row) => Number.isFinite(row.value))) throw Error(`${id}: non-numeric value.`);
  return rows;
}

async function subway() {
  const since = iso(Date.now() - (DAYS + 45) * DAY);
  const where = encodeURIComponent(`mode='Subway' AND date>='${since}'`);
  const rows = await get(`https://data.ny.gov/resource/sayj-mze2.json?$where=${where}&$order=date&$limit=5000`);
  return checkSpacing("subway", rows.map((r) => ({ date: r.date.slice(0, 10), value: Number(r.count) })).slice(-DAYS), 1);
}

async function wikipedia() {
  const start = compactDate(Date.now() - (DAYS + 45) * DAY);
  const end = compactDate(Date.now() - DAY);
  const { items } = await get(`https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Machine_learning/daily/${start}/${end}`);
  const rows = items.map((item) => ({
    date: `${item.timestamp.slice(0, 4)}-${item.timestamp.slice(4, 6)}-${item.timestamp.slice(6, 8)}`,
    value: item.views,
  }));
  return checkSpacing("wikipedia", rows.slice(-DAYS), 1);
}

async function co2() {
  const text = await get("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.csv", "text");
  const rows = text.split("\n")
    .filter((line) => /^\d{4},/.test(line))
    .map((line) => {
      const [year, month, day, , average] = line.split(",");
      return { date: `${year}-${month.padStart(2, "0")}-${day.padStart(2, "0")}`, value: Number(average) };
    })
    .slice(-WEEKS);

  // NOAA marks weeks without enough valid measurements as -999.99. Fill them by linear interpolation.
  let filled = 0;
  rows.forEach((row, i) => {
    if (row.value > 0) return;
    const before = rows.slice(0, i).findLast((r) => r.value > 0);
    const afterIndex = rows.findIndex((r, j) => j > i && r.value > 0);
    if (!before || afterIndex < 0) throw Error(`co2: cannot interpolate ${row.date}.`);
    const beforeIndex = rows.indexOf(before);
    const t = (i - beforeIndex) / (afterIndex - beforeIndex);
    row.value = Math.round((before.value + t * (rows[afterIndex].value - before.value)) * 100) / 100;
    filled++;
  });
  return { rows: checkSpacing("co2", rows, 7), filled };
}

// Plain-language descriptions of each series, computed from the data itself.
const average = (list) => list.reduce((sum, v) => sum + v, 0) / list.length;
const short = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : Math.round(n).toLocaleString("en-US"));

function weekly(rows, noun) {
  const isWeekend = (row) => [0, 6].includes(new Date(`${row.date}T00:00:00Z`).getUTCDay());
  const weekdays = average(rows.filter((r) => !isWeekend(r)).map((r) => r.value));
  const weekends = average(rows.filter(isWeekend).map((r) => r.value));
  return `Each dip is a weekend: about ${short(weekdays)} ${noun} on an average weekday vs ${short(weekends)} on Saturdays and Sundays. The gray line is the 7-day average.`;
}

function yearly(rows) {
  const values = rows.map((r) => r.value);
  const rise = average(values.slice(-52)) - average(values.slice(-104, -52));
  return `Rises about ${rise.toFixed(1)} ppm a year, with a yearly cycle: CO₂ falls each Northern Hemisphere summer as plants grow, then climbs back.`;
}

const retrieved = iso(Date.now());
const [subwayRows, wikiRows, co2Result] = await Promise.all([subway(), wikipedia(), co2()]);

const datasets = [
  {
    id: "subway",
    title: "NYC subway ridership",
    cadence: "daily",
    unit: "riders",
    season: 7,
    source: "Metropolitan Transportation Authority, via data.ny.gov",
    sourceUrl: "https://data.ny.gov/Transportation/MTA-Daily-Ridership-and-Traffic-Beginning-2020/sayj-mze2",
    license: "NY Open Data terms of use",
    note: "Estimated daily subway ridership across New York City.",
    pattern: weekly(subwayRows, "riders"),
    rows: subwayRows,
  },
  {
    id: "wikipedia",
    title: "Wikipedia views: Machine learning",
    cadence: "daily",
    unit: "views",
    season: 7,
    source: "Wikimedia Pageviews API",
    sourceUrl: "https://pageviews.wmcloud.org/?project=en.wikipedia.org&pages=Machine_learning",
    license: "CC0",
    note: "Daily views of the English Wikipedia article by people (automated traffic excluded).",
    pattern: weekly(wikiRows, "views"),
    rows: wikiRows,
  },
  {
    id: "co2",
    title: "Mauna Loa CO₂",
    cadence: "weekly",
    unit: "ppm",
    season: 52,
    source: "NOAA Global Monitoring Laboratory",
    sourceUrl: "https://gml.noaa.gov/ccgg/trends/data.html",
    license: "public domain, U.S. government work",
    note: `Weekly mean atmospheric CO₂ at Mauna Loa Observatory, Hawaii.${co2Result.filled ? ` ${co2Result.filled} week${co2Result.filled === 1 ? "" : "s"} without enough measurements ${co2Result.filled === 1 ? "was" : "were"} filled by linear interpolation.` : ""}`,
    pattern: yearly(co2Result.rows),
    rows: co2Result.rows,
  },
];

await mkdir(OUT, { recursive: true });
for (const { rows, ...meta } of datasets) {
  await writeFile(new URL(`${meta.id}.csv`, OUT), `date,value\n${rows.map((r) => `${r.date},${r.value}`).join("\n")}\n`);
  meta.file = `${meta.id}.csv`;
  meta.retrieved = retrieved;
  meta.first = rows[0].date;
  meta.last = rows.at(-1).date;
  meta.count = rows.length;
  Object.assign(datasets.find((d) => d.id === meta.id), meta);
}
await writeFile(new URL("datasets.json", OUT), `${JSON.stringify(datasets.map(({ rows, ...meta }) => meta), null, 2)}\n`);
for (const d of datasets) console.log(`${d.id.padEnd(10)} ${d.count} rows  ${d.first} → ${d.last}`);
