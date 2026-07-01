# Fielding

```js
import {CalendarBar} from "./components/calendarBar.js";
import {statTable, FIELDING_COLS} from "./components/statTable.js";
const schedule = FileAttachment("./data/schedule.json").json();
const latest = FileAttachment("./data/latest.json").json();
```

```js
display(CalendarBar(schedule));
```

*PO = putouts · A = assists · E = errors · FP = fielding %. As of ${latest.asOf}.*

```js
display(statTable(latest.fielding, FIELDING_COLS, {sort: "E", reverse: true}));
```
