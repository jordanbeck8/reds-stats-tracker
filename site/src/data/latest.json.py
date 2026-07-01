#!/usr/bin/env python3
"""Loader: newest live snapshot per stat group → {hitting, pitching, fielding, asOf}."""

import json
import sys
from pathlib import Path

SNAPSHOTS = Path(__file__).resolve().parents[3] / "data" / "snapshots"


def latest(group: str) -> tuple[list, str | None]:
    path = next(iter(sorted(SNAPSHOTS.glob(f"{group}-*.ndjson"), reverse=True)), None)
    if path is None:
        return [], None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return [], None
    max_date = max(r["date"] for r in rows)
    return [r for r in rows if r["date"] == max_date], max_date


hitting, as_of = latest("hitting")
pitching, _ = latest("pitching")
fielding, _ = latest("fielding")

json.dump(
    {"hitting": hitting, "pitching": pitching, "fielding": fielding, "asOf": as_of},
    sys.stdout,
)
