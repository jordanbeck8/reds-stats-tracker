"""MLB Stats API access — season stats, schedule, per-player game logs."""

import requests

from .config import MLB_PEOPLE_URL, MLB_SCHEDULE_URL, MLB_STATS_URL, MLB_TEAM_ID


def fetch_team_stats(group: str, year: int) -> list:
    """Fetch per-player season stats from the MLB Stats API for the Reds.

    group: "hitting" | "pitching" | "fielding"
    """
    params = {
        "stats": "season",
        "season": str(year),
        "group": group,
        "gameType": "R",
        "teamId": str(MLB_TEAM_ID),
        "playerPool": "All",
        "limit": "100",
    }
    resp = requests.get(MLB_STATS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["stats"][0]["splits"]


def fetch_schedule(year: int) -> list:
    """Fetch the Reds' full regular-season schedule with scores and status."""
    params = {
        "sportId": "1",
        "teamId": str(MLB_TEAM_ID),
        "startDate": f"{year}-03-01",
        "endDate": f"{year}-11-15",
        "gameType": "R",
        "hydrate": "team,linescore",
    }
    resp = requests.get(MLB_SCHEDULE_URL, params=params, timeout=30)
    resp.raise_for_status()
    games = []
    for day in resp.json().get("dates", []):
        games.extend(day.get("games", []))
    return games


def fetch_game_log(player_id: int, group: str, year: int) -> list:
    """Fetch a player's per-game log for one season (used by backfill)."""
    params = {"stats": "gameLog", "season": str(year), "group": group}
    resp = requests.get(f"{MLB_PEOPLE_URL}/{player_id}/stats", params=params, timeout=30)
    resp.raise_for_status()
    stats = resp.json().get("stats", [])
    if not stats:
        return []
    return stats[0].get("splits", [])
