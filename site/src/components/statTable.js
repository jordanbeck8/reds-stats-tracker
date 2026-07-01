import * as Inputs from "npm:@observablehq/inputs";

const fmt3 = (v) => (v == null ? "-" : v.toFixed(3).replace(/^0\./, "."));
const fmt2 = (v) => (v == null ? "-" : v.toFixed(2));
const fmt1 = (v) => (v == null ? "-" : v.toFixed(1));
const pct = (v) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`);

export const HITTING_COLS = {
  Name: {}, G: {}, PA: {}, AB: {}, H: {}, HR: {}, RBI: {}, R: {}, SB: {},
  AVG: {format: fmt3}, OBP: {format: fmt3}, SLG: {format: fmt3}, OPS: {format: fmt3},
  "BB%": {format: pct}, "K%": {format: pct},
  bWAR: {format: fmt1}, fWAR: {format: fmt1}
};

export const PITCHING_COLS = {
  Name: {}, G: {}, GS: {}, IP: {}, W: {}, L: {}, SV: {},
  ERA: {format: fmt2}, WHIP: {format: fmt2},
  "K/9": {format: fmt2}, "BB/9": {format: fmt2}, "HR/9": {format: fmt2},
  "K%": {format: pct}, "BB%": {format: pct},
  bWAR: {format: fmt1}, fWAR: {format: fmt1}
};

export const FIELDING_COLS = {
  Name: {}, Pos: {}, G: {}, GS: {}, Inn: {format: fmt1}, PO: {}, A: {}, E: {},
  FP: {format: fmt3}
};

export function statTable(rows, cols, {sort = "bWAR", reverse = true} = {}) {
  const columns = Object.keys(cols).filter((c) => rows.some((r) => r[c] != null));
  return Inputs.table(rows, {
    columns,
    format: Object.fromEntries(
      columns.filter((c) => cols[c].format).map((c) => [c, cols[c].format])
    ),
    sort,
    reverse,
    rows: 40,
    layout: "auto"
  });
}
