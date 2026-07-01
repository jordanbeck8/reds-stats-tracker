"""Shared configuration and paths."""

from datetime import datetime
from pathlib import Path

TEAM = "CIN"
MLB_TEAM_ID = 113          # MLB Stats API team id for Cincinnati
FG_TEAM_ID = 18            # FanGraphs leaderboard team id for Cincinnati
YEAR = datetime.now().year

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
SCHEDULE_DIR = DATA_DIR / "schedule"
CACHE_DIR = DATA_DIR / "caches"

MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"

BWAR_URLS = {
    "bat":   "https://www.baseball-reference.com/data/war_daily_bat.txt",
    "pitch": "https://www.baseball-reference.com/data/war_daily_pitch.txt",
}
FG_LEADERS_URL = "https://www.fangraphs.com/api/leaders/major-league/data"

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
