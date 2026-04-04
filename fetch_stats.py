#!/usr/bin/env python3
"""
Cincinnati Reds Stats Tracker
Fetches hitting, pitching, and fielding statistics from FanGraphs (via pybaseball)
and Baseball-Reference, then writes formatted markdown tables to README.md.
"""

import json
import sys
import unicodedata
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import pybaseball
import requests

TEAM = "CIN"
YEAR = datetime.now().year

# BRef raw CSV endpoints — programmatic access intended by BRef
_BWAR_URLS = {
    "bat":   "https://www.baseball-reference.com/data/war_daily_bat.txt",
    "pitch": "https://www.baseball-reference.com/data/war_daily_pitch.txt",
}
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# bWAR cache — updated when BRef is reachable (local runs), read-only fallback on CI
_BWAR_CACHE = Path(__file__).parent / "bwar_cache.json"


def normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy merge across FanGraphs/BRef naming differences."""
    name = str(name)
    name = "".join(
        c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn"
    )
    for suffix in (" Jr.", " Sr.", " II", " III", " IV"):
        name = name.replace(suffix, "")
    return name.strip().lower()


def _fmt(series: pd.Series, spec: str, multiply: float = 1.0, suffix: str = "") -> pd.Series:
    """Format a numeric series with a format spec, replacing NaN with '-'."""
    return series.apply(
        lambda x: f"{x * multiply:{spec}}{suffix}" if pd.notna(x) else "-"
    )


def _get_bwar(source: str, label: str, year: int) -> pd.DataFrame:
    """Fetch bWAR from Baseball-Reference for CIN in the given year.

    Strategy:
      1. Direct HTTP request with browser UA (BRef blocks default pybaseball UA).
      2. pybaseball fallback.
      3. If both fail (e.g. GitHub Actions IP blocks), load bwar_cache.json.
    On success, writes result to bwar_cache.json so CI always has fresh data.
    """
    cache_key = f"{source}_{year}"

    def _parse(df: pd.DataFrame) -> pd.DataFrame:
        required = {"year_ID", "team_ID", "name_common", "WAR"}
        if not required.issubset(df.columns):
            raise ValueError(f"Unexpected columns: {df.columns.tolist()[:6]}")
        cin = df[(df["year_ID"] == year) & (df["team_ID"] == TEAM)].copy()
        cin = cin.groupby("name_common", as_index=False)["WAR"].sum()
        cin.columns = ["Name", "bWAR"]
        cin["name_key"] = cin["Name"].apply(normalize_name)
        return cin

    def _cache_write(result: pd.DataFrame) -> None:
        data = json.loads(_BWAR_CACHE.read_text()) if _BWAR_CACHE.exists() else {}
        data[cache_key] = result[["Name", "bWAR"]].to_dict(orient="records")
        _BWAR_CACHE.write_text(json.dumps(data, indent=2))

    print(f"  Fetching bWAR ({label}) from Baseball-Reference…")

    # Attempt 1: direct request with browser User-Agent
    try:
        resp = requests.get(
            _BWAR_URLS[source], headers={"User-Agent": _BROWSER_UA}, timeout=30
        )
        resp.raise_for_status()
        result = _parse(pd.read_csv(StringIO(resp.text)))
        _cache_write(result)
        return result
    except Exception as exc:
        print(f"  Direct fetch failed ({exc}), trying pybaseball…")

    # Attempt 2: pybaseball
    pybaseball_func = pybaseball.bwar_bat if source == "bat" else pybaseball.bwar_pitch
    try:
        result = _parse(pybaseball_func(return_all=False))
        _cache_write(result)
        return result
    except Exception as exc:
        print(f"  pybaseball failed ({exc}), loading cache…")

    # Attempt 3: local cache (committed to repo, always available on CI)
    if _BWAR_CACHE.exists():
        try:
            data = json.loads(_BWAR_CACHE.read_text())
            if cache_key in data:
                print(f"  Using cached bWAR ({label}) — live fetch unavailable.")
                df = pd.DataFrame(data[cache_key])
                df["name_key"] = df["Name"].apply(normalize_name)
                return df
        except Exception:
            pass

    print(f"  Warning: bWAR ({label}) unavailable — no live data or cache.")
    return pd.DataFrame(columns=["Name", "bWAR", "name_key"])


def _merge_bwar(df: pd.DataFrame, bwar: pd.DataFrame) -> pd.DataFrame:
    """Left-join bWAR onto df via normalized name key."""
    df["name_key"] = df["Name"].apply(normalize_name)
    if not bwar.empty:
        df = df.merge(bwar[["name_key", "bWAR"]], on="name_key", how="left")
    else:
        df["bWAR"] = None
    return df.drop("name_key", axis=1)


# ---------------------------------------------------------------------------
# Stat fetchers
# ---------------------------------------------------------------------------

def get_hitting_stats(year: int) -> pd.DataFrame:
    """Fetch hitting stats from FanGraphs and merge bWAR."""
    print(f"Fetching hitting stats from FanGraphs ({year})…")
    raw = pybaseball.batting_stats(year, qual=0)
    reds = raw[raw["Team"] == TEAM].copy()

    cols = ["Name", "G", "PA", "AB", "H", "HR", "RBI", "R", "SB",
            "AVG", "OBP", "SLG", "OPS", "BB%", "K%", "wRC+", "WAR"]
    reds = reds[[c for c in cols if c in reds.columns]].copy()
    reds = reds[reds["PA"] > 0]
    reds.rename(columns={"WAR": "fWAR"}, inplace=True)

    bwar = _get_bwar("bat", "hitting", year)
    reds = _merge_bwar(reds, bwar)
    reds.sort_values("fWAR", ascending=False, inplace=True)

    for col in ["AVG", "OBP", "SLG", "OPS"]:
        reds[col] = _fmt(reds[col], ".3f")
    for col in ["BB%", "K%"]:
        reds[col] = _fmt(reds[col], ".1f", multiply=100, suffix="%")
    reds["wRC+"] = reds["wRC+"].apply(lambda x: str(int(x)) if pd.notna(x) else "-")
    reds["fWAR"] = _fmt(reds["fWAR"], ".1f")
    reds["bWAR"] = _fmt(reds["bWAR"], ".1f")

    return reds


def get_pitching_stats(year: int) -> pd.DataFrame:
    """Fetch pitching stats from FanGraphs and merge bWAR."""
    print(f"Fetching pitching stats from FanGraphs ({year})…")
    raw = pybaseball.pitching_stats(year, qual=0)
    reds = raw[raw["Team"] == TEAM].copy()

    cols = ["Name", "G", "GS", "IP", "W", "L", "SV",
            "ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "HR/9",
            "K%", "BB%", "WAR"]
    reds = reds[[c for c in cols if c in reds.columns]].copy()
    reds.rename(columns={"WAR": "fWAR"}, inplace=True)

    bwar = _get_bwar("pitch", "pitching", year)
    reds = _merge_bwar(reds, bwar)
    reds.sort_values("fWAR", ascending=False, inplace=True)

    for col in ["ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "HR/9"]:
        if col in reds.columns:
            reds[col] = _fmt(reds[col], ".2f")
    reds["IP"]   = _fmt(reds["IP"], ".1f")
    reds["K%"]   = _fmt(reds["K%"],  ".1f", multiply=100, suffix="%")
    reds["BB%"]  = _fmt(reds["BB%"], ".1f", multiply=100, suffix="%")
    reds["fWAR"] = _fmt(reds["fWAR"], ".1f")
    reds["bWAR"] = _fmt(reds["bWAR"], ".1f")

    return reds


def get_fielding_stats(year: int) -> pd.DataFrame:
    """Fetch fielding stats from FanGraphs."""
    print(f"Fetching fielding stats from FanGraphs ({year})…")
    raw = pybaseball.fielding_stats(year, qual=0)
    reds = raw[raw["Team"] == TEAM].copy()

    preferred = ["Name", "Pos", "G", "GS", "Inn", "PO", "A", "E",
                 "FP", "DRS", "UZR", "UZR/150", "OAA"]
    reds = reds[[c for c in preferred if c in reds.columns]].copy()

    if "DRS" in reds.columns:
        reds.sort_values("DRS", ascending=False, inplace=True)

    for col in ["UZR", "UZR/150", "DRS", "OAA"]:
        if col in reds.columns:
            reds[col] = _fmt(reds[col], ".1f")
    if "FP" in reds.columns:
        reds["FP"] = _fmt(reds["FP"], ".3f")
    if "Inn" in reds.columns:
        reds["Inn"] = _fmt(reds["Inn"], ".1f")

    return reds


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a GitHub-flavored markdown table."""
    if df.empty:
        return "_No data available yet for the current season._\n"
    cols = df.columns.tolist()
    header    = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, separator] + rows) + "\n"


def generate_readme(
    hitting: pd.DataFrame,
    pitching: pd.DataFrame,
    fielding: pd.DataFrame,
    year: int,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""\
# 🔴 Cincinnati Reds Stats Tracker — {year}

> **Last updated:** {timestamp}
> Data sources: [FanGraphs](https://www.fangraphs.com) · [Baseball-Reference](https://www.baseball-reference.com)
> **fWAR** = FanGraphs WAR · **bWAR** = Baseball-Reference WAR

---

## ⚾ Hitting

*Sorted by fWAR · wRC+ = Weighted Runs Created Plus (100 = league avg)*

{df_to_markdown(hitting)}
---

## 🔥 Pitching

*Sorted by fWAR · All rate stats per 9 innings*

{df_to_markdown(pitching)}
---

## 🧤 Fielding

*Sorted by DRS (Defensive Runs Saved) · UZR/150 = UZR per 150 games · OAA = Outs Above Average*

{df_to_markdown(fielding)}
---

*Auto-updated daily via [GitHub Actions](../../actions) · [View source](fetch_stats.py)*
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Cincinnati Reds Stats Tracker — {YEAR}")
    print("=" * 50)

    errors = []

    def fetch(name, fn):
        try:
            df = fn()
            print(f"  ✓ {name.capitalize()}: {len(df)} players")
            return df
        except Exception as exc:
            print(f"  ✗ {name.capitalize()} stats failed: {exc}")
            errors.append(f"{name}: {exc}")
            return pd.DataFrame()

    hitting  = fetch("hitting",  lambda: get_hitting_stats(YEAR))
    pitching = fetch("pitching", lambda: get_pitching_stats(YEAR))
    fielding = fetch("fielding", lambda: get_fielding_stats(YEAR))

    readme = generate_readme(hitting, pitching, fielding, YEAR)

    with open("README.md", "w", encoding="utf-8") as fh:
        fh.write(readme)

    print("\n✓ README.md updated successfully.")

    if errors:
        print(f"\nWarnings ({len(errors)} non-fatal errors):")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
