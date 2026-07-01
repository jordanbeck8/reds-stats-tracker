"""Durable daily snapshot store — NDJSON, one file per stat group per season.

Append is idempotent: rows are keyed by (date, player_id[, Pos]); re-running
on the same day replaces that day's rows, so the nightly and post-game runs
never duplicate. Files stay sorted by (date, name) for clean git diffs.
"""

import json
import math

import pandas as pd

from .config import SNAPSHOT_DIR


def _clean(value):
    """JSON-safe value: NaN → None, numpy scalars → Python scalars."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _load(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _key(row: dict, group: str):
    key = (row["date"], row["player_id"])
    return key + (row.get("Pos"),) if group == "fielding" else key


def append_daily(group: str, df: pd.DataFrame, date: str, season: int, source: str = "live") -> int:
    """Replace `date` rows for `group` with the rows of df. Returns row count written."""
    path = SNAPSHOT_DIR / f"{group}-{season}.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)

    fresh = []
    for rec in df.to_dict(orient="records"):
        row = {"date": date, "season": season, "source": source}
        row.update({k: _clean(v) for k, v in rec.items()})
        fresh.append(row)

    existing = [r for r in _load(path) if r["date"] != date]
    merged = existing + fresh
    merged.sort(key=lambda r: (r["date"], r.get("Name") or "", r.get("Pos") or ""))

    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    return len(fresh)


def append_rows(group: str, rows: list[dict], season: int) -> int:
    """Bulk-insert pre-built rows (backfill). Existing (date, player_id) keys win.

    Live rows are never overwritten by backfill: any key already present in
    the file is kept and the incoming row dropped.
    """
    path = SNAPSHOT_DIR / f"{group}-{season}.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load(path)
    seen = {_key(r, group) for r in existing}
    added = [r for r in rows if _key(r, group) not in seen]

    merged = existing + added
    merged.sort(key=lambda r: (r["date"], r.get("Name") or "", r.get("Pos") or ""))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    return len(added)
