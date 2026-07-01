# Hitting

```js
import {CalendarBar} from "./components/calendarBar.js";
import {statTable, HITTING_COLS} from "./components/statTable.js";
const schedule = FileAttachment("./data/schedule.json").json();
const latest = FileAttachment("./data/latest.json").json();
```

```js
display(CalendarBar(schedule));
```

*Sorted by bWAR — click any column header to re-sort. As of ${latest.asOf}.*

```js
display(statTable(latest.hitting, HITTING_COLS));
```
