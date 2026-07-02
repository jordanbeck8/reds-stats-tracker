# Compare Players

```js
import * as Plot from "npm:@observablehq/plot";
import {CalendarBar} from "./components/calendarBar.js";
import {radar, percentile} from "./components/radar.js";
const schedule = FileAttachment("./data/schedule.json").json();
const latest = FileAttachment("./data/latest.json").json();
```

```js
display(CalendarBar(schedule));
```

```js
const mode = view(Inputs.radio(["Hitters", "Pitchers"], {value: "Hitters", label: "Player type"}));
```

```js
const pool = mode === "Hitters"
  ? latest.hitting.filter((p) => p.PA >= 30)
  : latest.pitching.filter((p) => (p.IP_f ?? 0) >= 5);
const byName = new Map(pool.map((p) => [p.Name, p]));
const defaults = [...pool].sort((a, b) => (b.bWAR ?? -9) - (a.bWAR ?? -9)).slice(0, 2).map((p) => p.Name);
```

```js
const names = view(Inputs.checkbox(
  [...byName.keys()].sort(),
  {value: defaults, label: "Players (pick 2+)", className: "player-picker"}
));
```

```js
const players = names.map((n) => byName.get(n)).filter(Boolean);

// Team-percentile-normalized radar axes; "inv" = lower is better.
const AXES = mode === "Hitters"
  ? [["AVG"], ["OBP"], ["SLG"], ["HR"], ["SB"], ["BB%"], ["K%", true], ["bWAR"]]
  : [["ERA", true], ["WHIP", true], ["K/9"], ["BB/9", true], ["IP_f"], ["bWAR"]];

const radarData = players.flatMap((p) =>
  AXES.map(([axis, inv]) => ({
    name: p.Name,
    axis: axis === "IP_f" ? "IP" : axis,
    raw: p[axis],
    value: percentile(p[axis], pool.map((q) => q[axis]), inv ?? false)
  }))
);
```

```js
if (players.length < 2) {
  display(html`<p><em>Select at least two players to compare.</em></p>`);
} else {
  display(radar(radarData, {width: Math.min(width, 560)}));
}
```

## Side by side

```js
const STATS = mode === "Hitters"
  ? ["G", "PA", "AB", "H", "HR", "RBI", "R", "SB", "AVG", "OBP", "SLG", "OPS", "BB%", "K%", "bWAR", "fWAR"]
  : ["G", "GS", "IP", "W", "L", "SV", "ERA", "WHIP", "K/9", "BB/9", "HR/9", "K%", "BB%", "bWAR", "fWAR"];
const LOWER_BETTER = new Set(["K%", "ERA", "WHIP", "BB/9", "HR/9", "BB%"]);
const hitterLowerBetter = new Set(["K%"]);
const lowerBetter = mode === "Hitters" ? hitterLowerBetter : LOWER_BETTER;

const fmt = (stat, v) => {
  if (v == null) return "-";
  if (["AVG", "OBP", "SLG", "OPS"].includes(stat)) return v.toFixed(3).replace(/^0\./, ".");
  if (["BB%", "K%"].includes(stat)) return `${(v * 100).toFixed(1)}%`;
  if (["ERA", "WHIP", "K/9", "BB/9", "HR/9"].includes(stat)) return v.toFixed(2);
  if (["bWAR", "fWAR"].includes(stat)) return v.toFixed(1);
  return v;
};

if (players.length >= 2) {
  display(html`<table class="compare-table">
    <thead><tr><th>Stat</th>${players.map((p) => html`<th>${p.Name}</th>`)}</tr></thead>
    <tbody>${STATS.map((stat) => {
      const vals = players.map((p) => (stat === "IP" ? p.IP_f : p[stat]));
      const numeric = vals.filter((v) => typeof v === "number");
      const best = numeric.length >= 2
        ? (lowerBetter.has(stat) ? Math.min(...numeric) : Math.max(...numeric))
        : undefined;
      return html`<tr><td><b>${stat}</b></td>${players.map((p, i) => {
        const v = vals[i];
        return html`<td class=${v === best && numeric.length >= 2 ? "best" : ""}>${fmt(stat, p[stat] ?? v)}</td>`;
      })}</tr>`;
    })}</tbody>
  </table>`);
}
```

## WAR: Baseball-Reference vs FanGraphs

```js
if (players.length >= 2) {
  display(Plot.plot({
    height: 90 + players.length * 56,
    marginLeft: 130,
    x: {label: "WAR"},
    y: {label: null},
    color: {domain: ["bWAR", "fWAR"], range: ["#c6011f", "#000000"], legend: true},
    marks: [
      Plot.barX(players.flatMap((p) => [
        {name: p.Name, source: "bWAR", war: p.bWAR},
        {name: p.Name, source: "fWAR", war: p.fWAR}
      ]).filter((d) => d.war != null),
      {y: "name", x: "war", fill: "source", fy: "name", tip: true}),
      Plot.ruleX([0])
    ]
  }));
}
```
