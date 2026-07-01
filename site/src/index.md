# Cincinnati Reds Dashboard

```js
import {CalendarBar, teamRecord} from "./components/calendarBar.js";
const schedule = FileAttachment("./data/schedule.json").json();
const latest = FileAttachment("./data/latest.json").json();
```

```js
display(CalendarBar(schedule));
```

```js
const rec = teamRecord(schedule);
const played = schedule.filter((g) => g.state.startsWith("Final")).length;
display(html`<div class="grid grid-cols-3">
  <div class="card"><h2>Record</h2><span class="big">${rec.w}–${rec.l}</span></div>
  <div class="card"><h2>Games Played</h2><span class="big">${played}</span></div>
  <div class="card"><h2>Stats as of</h2><span class="big">${latest.asOf}</span></div>
</div>`);
```

## WAR Leaders

```js
const leaders = [...latest.hitting, ...latest.pitching]
  .filter((p) => p.bWAR != null || p.fWAR != null)
  .sort((a, b) => (b.bWAR ?? -9) - (a.bWAR ?? -9))
  .slice(0, 5);

display(html`<div style="display:flex; gap:12px; flex-wrap:wrap;">
  ${leaders.map((p) => html`<div class="war-card">
    <div class="war-name">${p.Name}</div>
    <div class="war-values">${p.bWAR?.toFixed(1) ?? "-"} <span style="font-size:14px;color:#666">/ ${p.fWAR?.toFixed(1) ?? "-"}</span></div>
    <div class="war-label">bWAR / fWAR</div>
  </div>`)}
</div>`);
```

## Team WAR by source

```js
import * as Plot from "npm:@observablehq/plot";
const hit = latest.hitting.reduce((s, p) => s + (p.bWAR ?? 0), 0);
const hitF = latest.hitting.reduce((s, p) => s + (p.fWAR ?? 0), 0);
const pit = latest.pitching.reduce((s, p) => s + (p.bWAR ?? 0), 0);
const pitF = latest.pitching.reduce((s, p) => s + (p.fWAR ?? 0), 0);
display(Plot.plot({
  height: 260,
  marginLeft: 70,
  x: {label: "WAR"},
  y: {label: null},
  color: {domain: ["bWAR", "fWAR"], range: ["#c6011f", "#000000"], legend: true},
  marks: [
    Plot.barX([
      {side: "Hitting", source: "bWAR", war: hit},
      {side: "Hitting", source: "fWAR", war: hitF},
      {side: "Pitching", source: "bWAR", war: pit},
      {side: "Pitching", source: "fWAR", war: pitF}
    ], {y: "side", x: "war", fill: "source", fy: "source", tip: true}),
    Plot.ruleX([0])
  ]
}));
```
