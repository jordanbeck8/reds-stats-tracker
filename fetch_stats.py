#!/usr/bin/env python3
"""
Cincinnati Reds Stats Tracker
Fetches hitting, pitching, and fielding statistics from the MLB Stats API
and Baseball-Reference (bWAR), then writes formatted markdown tables to README.md.
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

# MLB Stats API — Reds team ID is 113
_MLB_TEAM_ID = 113
_MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"

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
    """Normalize a player name for fuzzy merge across data source naming differences."""
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


def _fetch_mlb_stats(group: str, year: int) -> list:
    """Fetch per-player stats from the MLB Stats API for the Reds."""
    params = {
        "stats": "season",
        "season": str(year),
        "group": group,
        "gameType": "R",
        "teamId": str(_MLB_TEAM_ID),
        "playerPool": "All",
        "limit": "100",
    }
    resp = requests.get(_MLB_STATS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["stats"][0]["splits"]


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
        # BRef serves UTF-8 but may advertise no charset — decode explicitly.
        result = _parse(pd.read_csv(StringIO(resp.content.decode("utf-8", errors="replace"))))
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
    """Fetch hitting stats from MLB Stats API and merge bWAR."""
    print(f"Fetching hitting stats from MLB Stats API ({year})…")
    splits = _fetch_mlb_stats("hitting", year)

    rows = []
    for s in splits:
        stat = s["stat"]
        pa = stat.get("plateAppearances") or 0
        bb = stat.get("baseOnBalls") or 0
        so = stat.get("strikeOuts") or 0
        rows.append({
            "Name": s["player"]["fullName"],
            "G":    stat.get("gamesPlayed"),
            "PA":   pa,
            "AB":   stat.get("atBats"),
            "H":    stat.get("hits"),
            "HR":   stat.get("homeRuns"),
            "RBI":  stat.get("rbi"),
            "R":    stat.get("runs"),
            "SB":   stat.get("stolenBases"),
            "AVG":  stat.get("avg", "-"),
            "OBP":  stat.get("obp", "-"),
            "SLG":  stat.get("slg", "-"),
            "OPS":  stat.get("ops", "-"),
            "BB%":  bb / pa if pa > 0 else None,
            "K%":   so / pa if pa > 0 else None,
        })

    reds = pd.DataFrame(rows)
    reds = reds[reds["PA"] > 0]

    bwar = _get_bwar("bat", "hitting", year)
    reds = _merge_bwar(reds, bwar)
    reds.sort_values("bWAR", ascending=False, na_position="last", inplace=True)

    reds["BB%"]  = _fmt(reds["BB%"], ".1f", multiply=100, suffix="%")
    reds["K%"]   = _fmt(reds["K%"],  ".1f", multiply=100, suffix="%")
    reds["bWAR"] = _fmt(reds["bWAR"], ".1f")

    return reds


def get_pitching_stats(year: int) -> pd.DataFrame:
    """Fetch pitching stats from MLB Stats API and merge bWAR."""
    print(f"Fetching pitching stats from MLB Stats API ({year})…")
    splits = _fetch_mlb_stats("pitching", year)

    rows = []
    for s in splits:
        stat = s["stat"]
        bf = stat.get("battersFaced") or 0
        bb = stat.get("baseOnBalls") or 0
        so = stat.get("strikeOuts") or 0
        rows.append({
            "Name": s["player"]["fullName"],
            "G":    stat.get("gamesPlayed"),
            "GS":   stat.get("gamesStarted"),
            "IP":   stat.get("inningsPitched", "-"),
            "W":    stat.get("wins"),
            "L":    stat.get("losses"),
            "SV":   stat.get("saves"),
            "ERA":  stat.get("era", "-"),
            "WHIP": stat.get("whip", "-"),
            "K/9":  stat.get("strikeoutsPer9Inn", "-"),
            "BB/9": stat.get("walksPer9Inn", "-"),
            "HR/9": stat.get("homeRunsPer9", "-"),
            "K%":   so / bf if bf > 0 else None,
            "BB%":  bb / bf if bf > 0 else None,
        })

    reds = pd.DataFrame(rows)

    bwar = _get_bwar("pitch", "pitching", year)
    reds = _merge_bwar(reds, bwar)
    reds.sort_values("bWAR", ascending=False, na_position="last", inplace=True)

    reds["K%"]   = _fmt(reds["K%"],  ".1f", multiply=100, suffix="%")
    reds["BB%"]  = _fmt(reds["BB%"], ".1f", multiply=100, suffix="%")
    reds["bWAR"] = _fmt(reds["bWAR"], ".1f")

    return reds


def get_fielding_stats(year: int) -> pd.DataFrame:
    """Fetch fielding stats from MLB Stats API."""
    print(f"Fetching fielding stats from MLB Stats API ({year})…")
    splits = _fetch_mlb_stats("fielding", year)

    rows = []
    for s in splits:
        stat = s["stat"]
        pos = s.get("position", {}).get("abbreviation", "?")
        if pos == "P":
            continue
        rows.append({
            "Name": s["player"]["fullName"],
            "Pos":  pos,
            "G":    stat.get("gamesPlayed"),
            "GS":   stat.get("gamesStarted"),
            "Inn":  stat.get("innings", "-"),
            "PO":   stat.get("putOuts"),
            "A":    stat.get("assists"),
            "E":    stat.get("errors"),
            "FP":   stat.get("fielding", "-"),
        })

    reds = pd.DataFrame(rows)
    if not reds.empty and "E" in reds.columns:
        reds.sort_values("E", ascending=False, inplace=True)

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
> Data sources: [MLB Stats API](https://statsapi.mlb.com) · [Baseball-Reference](https://www.baseball-reference.com)
> **bWAR** = Baseball-Reference WAR

---

## ⚾ Hitting

*Sorted by bWAR · BB% and K% calculated from plate appearances*

{df_to_markdown(hitting)}
---

## 🔥 Pitching

*Sorted by bWAR · All rate stats per 9 innings*

{df_to_markdown(pitching)}
---

## 🧤 Fielding

*Standard fielding stats · PO = Putouts · A = Assists · E = Errors · FP = Fielding %*

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
