"""Fetch Reds season stats from the MLB Stats API as raw numeric DataFrames.

All values are numeric (no display formatting) — formatting happens in the
README renderer and the site layer. Rows carry the MLB person id as the
stable join key.
"""

import pandas as pd

from .mlb_api import fetch_team_stats


def _num(value):
    """Parse an MLB API stat value ('.270', '2.36', '-.--') to float, else None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def innings_to_float(ip) -> float | None:
    """Convert baseball innings notation (91.2 = 91⅔) to a real float."""
    if ip is None:
        return None
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole) + int(frac or 0) / 3.0
    except ValueError:
        return None


def get_hitting_stats(year: int) -> pd.DataFrame:
    print(f"Fetching hitting stats from MLB Stats API ({year})…")
    splits = fetch_team_stats("hitting", year)

    rows = []
    for s in splits:
        stat = s["stat"]
        pa = stat.get("plateAppearances") or 0
        bb = stat.get("baseOnBalls") or 0
        so = stat.get("strikeOuts") or 0
        rows.append({
            "player_id": s["player"]["id"],
            "Name": s["player"]["fullName"],
            "G":    stat.get("gamesPlayed"),
            "PA":   pa,
            "AB":   stat.get("atBats"),
            "H":    stat.get("hits"),
            "2B":   stat.get("doubles"),
            "3B":   stat.get("triples"),
            "HR":   stat.get("homeRuns"),
            "RBI":  stat.get("rbi"),
            "R":    stat.get("runs"),
            "SB":   stat.get("stolenBases"),
            "BB":   bb,
            "SO":   so,
            "AVG":  _num(stat.get("avg")),
            "OBP":  _num(stat.get("obp")),
            "SLG":  _num(stat.get("slg")),
            "OPS":  _num(stat.get("ops")),
            "BB%":  bb / pa if pa > 0 else None,
            "K%":   so / pa if pa > 0 else None,
        })

    reds = pd.DataFrame(rows)
    return reds[reds["PA"] > 0]


def get_pitching_stats(year: int) -> pd.DataFrame:
    print(f"Fetching pitching stats from MLB Stats API ({year})…")
    splits = fetch_team_stats("pitching", year)

    rows = []
    for s in splits:
        stat = s["stat"]
        bf = stat.get("battersFaced") or 0
        bb = stat.get("baseOnBalls") or 0
        so = stat.get("strikeOuts") or 0
        rows.append({
            "player_id": s["player"]["id"],
            "Name": s["player"]["fullName"],
            "G":    stat.get("gamesPlayed"),
            "GS":   stat.get("gamesStarted"),
            "IP":   stat.get("inningsPitched"),
            "IP_f": innings_to_float(stat.get("inningsPitched")),
            "W":    stat.get("wins"),
            "L":    stat.get("losses"),
            "SV":   stat.get("saves"),
            "BF":   bf,
            "ER":   stat.get("earnedRuns"),
            "BB":   bb,
            "SO":   so,
            "ERA":  _num(stat.get("era")),
            "WHIP": _num(stat.get("whip")),
            "K/9":  _num(stat.get("strikeoutsPer9Inn")),
            "BB/9": _num(stat.get("walksPer9Inn")),
            "HR/9": _num(stat.get("homeRunsPer9")),
            "K%":   so / bf if bf > 0 else None,
            "BB%":  bb / bf if bf > 0 else None,
        })

    return pd.DataFrame(rows)


def get_fielding_stats(year: int) -> pd.DataFrame:
    print(f"Fetching fielding stats from MLB Stats API ({year})…")
    splits = fetch_team_stats("fielding", year)

    rows = []
    for s in splits:
        stat = s["stat"]
        pos = s.get("position", {}).get("abbreviation", "?")
        if pos == "P":
            continue
        rows.append({
            "player_id": s["player"]["id"],
            "Name": s["player"]["fullName"],
            "Pos":  pos,
            "G":    stat.get("gamesPlayed"),
            "GS":   stat.get("gamesStarted"),
            "Inn":  innings_to_float(stat.get("innings")),
            "PO":   stat.get("putOuts"),
            "A":    stat.get("assists"),
            "E":    stat.get("errors"),
            "FP":   _num(stat.get("fielding")),
        })

    return pd.DataFrame(rows)
