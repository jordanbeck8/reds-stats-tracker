"""Pipeline orchestrator: fetch stats + WAR + schedule, snapshot, render README.

Usage: python -m pipeline.run
Exits non-zero only if the MLB Stats API itself fails (WAR degrades to null).
"""

import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .config import REPO_ROOT, YEAR
from .readme import generate_readme
from .schedule import write_schedule
from .snapshots import append_daily
from .stats import get_fielding_stats, get_hitting_stats, get_pitching_stats
from .war import get_bwar, get_fwar, merge_war


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

    if not hitting.empty:
        hitting = merge_war(hitting, get_bwar("bat", "hitting", YEAR),
                            get_fwar("bat", "hitting", YEAR))
    if not pitching.empty:
        pitching = merge_war(pitching, get_bwar("pitch", "pitching", YEAR),
                             get_fwar("pitch", "pitching", YEAR))

    # Snapshot date = today in Cincinnati's timezone (post-game runs after
    # midnight UTC still belong to the local game day).
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    for group, df in (("hitting", hitting), ("pitching", pitching), ("fielding", fielding)):
        if df.empty:
            continue
        n = append_daily(group, df, today, YEAR)
        print(f"  ✓ Snapshot {group}: {n} rows @ {today}")

    try:
        write_schedule(YEAR)
    except Exception as exc:
        print(f"  ✗ Schedule failed: {exc}")
        errors.append(f"schedule: {exc}")

    readme = generate_readme(hitting, pitching, fielding, YEAR)
    (REPO_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("\n✓ README.md updated successfully.")

    if errors:
        print(f"\nWarnings ({len(errors)} non-fatal errors):")
        for e in errors:
            print(f"  - {e}")
    if hitting.empty and pitching.empty and fielding.empty:
        print("FATAL: MLB Stats API returned no data for any group.")
        sys.exit(1)


if __name__ == "__main__":
    main()
