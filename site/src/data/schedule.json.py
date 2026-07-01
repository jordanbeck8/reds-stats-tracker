#!/usr/bin/env python3
"""Loader: current-season schedule passthrough."""

import sys
from pathlib import Path

SCHEDULE = Path(__file__).resolve().parents[3] / "data" / "schedule"

path = next(iter(sorted(SCHEDULE.glob("*.json"), reverse=True)), None)
sys.stdout.write(path.read_text() if path else "[]")
