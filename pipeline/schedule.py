"""Write the Reds schedule (matchups, home/away, scores, status) to data/schedule/."""

import json

from .config import MLB_TEAM_ID, SCHEDULE_DIR
from .mlb_api import fetch_schedule


def write_schedule(year: int) -> int:
    print(f"Fetching schedule from MLB Stats API ({year})…")
    games = fetch_schedule(year)

    out = []
    for g in games:
        home = g["teams"]["home"]
        away = g["teams"]["away"]
        is_reds_home = home["team"]["id"] == MLB_TEAM_ID
        opponent = away["team"] if is_reds_home else home["team"]
        out.append({
            "gamePk": g["gamePk"],
            "date": g["officialDate"],
            "gameDateUTC": g["gameDate"],
            "home": home["team"].get("abbreviation") or home["team"]["name"],
            "away": away["team"].get("abbreviation") or away["team"]["name"],
            "homeId": home["team"]["id"],
            "awayId": away["team"]["id"],
            "homeScore": home.get("score"),
            "awayScore": away.get("score"),
            "state": g["status"]["detailedState"],
            "isRedsHome": is_reds_home,
            "opponentId": opponent["id"],
            "opponentAbbr": opponent.get("abbreviation") or opponent["name"],
        })

    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    path = SCHEDULE_DIR / f"{year}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"  ✓ Schedule: {len(out)} games → {path.name}")
    return len(out)
