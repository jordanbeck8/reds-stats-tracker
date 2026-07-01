"""Render the README.md artifact from raw stat DataFrames.

Formatting lives here (and in the site layer) — the pipeline stores raw numerics.
"""

from datetime import datetime, timezone

import pandas as pd


def _fmt(series: pd.Series, spec: str, multiply: float = 1.0, suffix: str = "") -> pd.Series:
    return series.apply(
        lambda x: f"{x * multiply:{spec}}{suffix}" if pd.notna(x) else "-"
    )


def df_to_markdown(df: pd.DataFrame) -> str:
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


def _format_hitting(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy().sort_values("bWAR", ascending=False, na_position="last")
    for col, spec in (("AVG", ".3f"), ("OBP", ".3f"), ("SLG", ".3f"), ("OPS", ".3f")):
        out[col] = _fmt(out[col], spec).str.replace("0.", ".", regex=False)
    out["BB%"]  = _fmt(out["BB%"], ".1f", multiply=100, suffix="%")
    out["K%"]   = _fmt(out["K%"],  ".1f", multiply=100, suffix="%")
    out["bWAR"] = _fmt(out["bWAR"], ".1f")
    out["fWAR"] = _fmt(out["fWAR"], ".1f")
    cols = ["Name", "G", "PA", "AB", "H", "HR", "RBI", "R", "SB",
            "AVG", "OBP", "SLG", "OPS", "BB%", "K%", "bWAR", "fWAR"]
    return out[cols]


def _format_pitching(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy().sort_values("bWAR", ascending=False, na_position="last")
    for col in ("ERA", "WHIP", "K/9", "BB/9", "HR/9"):
        out[col] = _fmt(out[col], ".2f")
    out["K%"]   = _fmt(out["K%"],  ".1f", multiply=100, suffix="%")
    out["BB%"]  = _fmt(out["BB%"], ".1f", multiply=100, suffix="%")
    out["bWAR"] = _fmt(out["bWAR"], ".1f")
    out["fWAR"] = _fmt(out["fWAR"], ".1f")
    cols = ["Name", "G", "GS", "IP", "W", "L", "SV", "ERA", "WHIP",
            "K/9", "BB/9", "HR/9", "K%", "BB%", "bWAR", "fWAR"]
    return out[cols]


def _format_fielding(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy().sort_values("E", ascending=False)
    out["Inn"] = _fmt(out["Inn"], ".1f")
    out["FP"]  = _fmt(out["FP"], ".3f").str.replace("0.", ".", regex=False)
    return out[["Name", "Pos", "G", "GS", "Inn", "PO", "A", "E", "FP"]]


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
> Data sources: [MLB Stats API](https://statsapi.mlb.com) · [Baseball-Reference](https://www.baseball-reference.com) · [FanGraphs](https://www.fangraphs.com)
> **bWAR** = Baseball-Reference WAR · **fWAR** = FanGraphs WAR

---

## ⚾ Hitting

*Sorted by bWAR · BB% and K% calculated from plate appearances*

{df_to_markdown(_format_hitting(hitting))}
---

## 🔥 Pitching

*Sorted by bWAR · All rate stats per 9 innings*

{df_to_markdown(_format_pitching(pitching))}
---

## 🧤 Fielding

*Standard fielding stats · PO = Putouts · A = Assists · E = Errors · FP = Fielding %*

{df_to_markdown(_format_fielding(fielding))}
---

*Auto-updated nightly + post-game · [View source](pipeline/) · Dashboard on the tailnet*
"""
