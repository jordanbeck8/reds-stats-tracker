#!/usr/bin/env python3
"""Loader: full hitting+pitching snapshot history → one parquet for the client."""

import io
import json
import sys
from pathlib import Path

import pandas as pd

SNAPSHOTS = Path(__file__).resolve().parents[3] / "data" / "snapshots"

frames = []
for group in ("hitting", "pitching"):
    for path in SNAPSHOTS.glob(f"{group}-*.ndjson"):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        df["group"] = group
        frames.append(df)

out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
# pyarrow doesn't reliably flush sys.stdout.buffer — write via BytesIO.
buf = io.BytesIO()
out.to_parquet(buf, index=False)
sys.stdout.buffer.write(buf.getvalue())
sys.stdout.buffer.flush()
