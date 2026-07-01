#!/usr/bin/env python3
"""Loader: per-date team WAR sums (live snapshot dates only) for trend charts."""

import json
import sys
from collections import defaultdict
from pathlib import Path

SNAPSHOTS = Path(__file__).resolve().parents[3] / "data" / "snapshots"

by_date = defaultdict(lambda: {"bWAR": 0.0, "fWAR": 0.0})
for group in ("hitting", "pitching"):
    for path in SNAPSHOTS.glob(f"{group}-*.ndjson"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("source") != "live":
                continue
            d = by_date[r["date"]]
            d["bWAR"] += r.get("bWAR") or 0.0
            d["fWAR"] += r.get("fWAR") or 0.0

out = [
    {"date": date, "bWAR": round(v["bWAR"], 2), "fWAR": round(v["fWAR"], 2)}
    for date, v in sorted(by_date.items())
]
json.dump(out, sys.stdout)
