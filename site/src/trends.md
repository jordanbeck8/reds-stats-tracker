# Trends

```js
import * as Plot from "npm:@observablehq/plot";
import {CalendarBar} from "./components/calendarBar.js";
const schedule = FileAttachment("./data/schedule.json").json();
const snapshots = FileAttachment("./data/snapshots.parquet").parquet();
const team = FileAttachment("./data/team.json").json();
```

```js
display(CalendarBar(schedule));
```

```js
const rows = [...snapshots].map((r) => ({...r, dateD: new Date(`${r.date}T12:00:00Z`)}));
const firstLive = rows.filter((r) => r.source === "live")
  .reduce((min, r) => (min == null || r.date < min ? r.date : min), null);
```

## Player trajectories

```js
const group = view(Inputs.radio(["hitting", "pitching"], {value: "hitting", label: "Group"}));
```

```js
const groupRows = rows.filter((r) => r.group === group);
const playerNames = [...new Set(groupRows.map((r) => r.Name))].sort();
const topDefault = groupRows
  .filter((r) => r.date === groupRows.reduce((m, x) => (x.date > m ? x.date : m), ""))
  .sort((a, b) => (b.bWAR ?? -9) - (a.bWAR ?? -9)).slice(0, 3).map((r) => r.Name);
```

```js
const selected = view(Inputs.select(playerNames, {multiple: 5, value: topDefault, label: "Players"}));
```

```js
const STAT_CHOICES = group === "hitting"
  ? ["OPS", "AVG", "OBP", "SLG", "HR", "RBI", "SB", "BB%", "K%", "bWAR", "fWAR"]
  : ["ERA", "WHIP", "K/9", "BB/9", "IP_f", "SO", "bWAR", "fWAR"];
const stat = view(Inputs.select(STAT_CHOICES, {value: group === "hitting" ? "OPS" : "ERA", label: "Stat"}));
```

```js
const series = rows
  .filter((r) => r.group === group && selected.includes(r.Name) && r[stat] != null)
  .sort((a, b) => a.date.localeCompare(b.date));
const warStat = stat === "bWAR" || stat === "fWAR";

display(Plot.plot({
  height: 360,
  x: {type: "utc", label: null},
  y: {label: stat, grid: true},
  color: {legend: true},
  marks: [
    Plot.lineY(series, {x: "dateD", y: stat, stroke: "Name", strokeWidth: 2, curve: "monotone-x", tip: true}),
    ...(warStat && firstLive
      ? [
          Plot.ruleX([new Date(`${firstLive}T12:00:00Z`)], {stroke: "#c6011f", strokeDasharray: "4,3"}),
          Plot.text([{x: new Date(`${firstLive}T12:00:00Z`)}], {
            x: "x", frameAnchor: "top", dy: 6, dx: 6, textAnchor: "start",
            text: () => `WAR history begins ${firstLive}`, fill: "#c6011f", fontSize: 11
          })
        ]
      : [])
  ]
}));
```

${firstLive ? html`<p><small>Counting stats before <b>${firstLive}</b> are reconstructed from MLB game logs (<code>source: backfill</code>). bWAR/fWAR only exist from <b>${firstLive}</b> onward, when daily live snapshots began.</small></p>` : ""}

## Team WAR accumulation

```js
const teamSeries = team.flatMap((d) => [
  {date: new Date(`${d.date}T12:00:00Z`), source: "bWAR", war: d.bWAR},
  {date: new Date(`${d.date}T12:00:00Z`), source: "fWAR", war: d.fWAR}
]);

display(Plot.plot({
  height: 300,
  x: {type: "utc", label: null},
  y: {label: "Team WAR", grid: true},
  color: {domain: ["bWAR", "fWAR"], range: ["#c6011f", "#000000"], legend: true},
  marks: [
    Plot.lineY(teamSeries, {x: "date", y: "war", stroke: "source", strokeWidth: 2, curve: "monotone-x"}),
    Plot.dot(teamSeries, {x: "date", y: "war", fill: "source", r: 3, tip: true}),
    Plot.ruleY([0])
  ]
}));
```

<p><small>Team WAR = sum of player bWAR/fWAR across hitting and pitching on each live snapshot date. The series lengthens daily.</small></p>
