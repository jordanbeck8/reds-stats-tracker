"""One-time season backfill: reconstruct daily cumulative stats from game logs.

For every player in the current season stat pool, fetches the per-game log
from the MLB Stats API and cumulative-sums counting stats by date, deriving
rate stats. Rows are marked source="backfill" with bWAR/fWAR null — WAR
history only exists from the date live snapshots began (label this on the
trends page). Fielding is not backfilled (low trend value, per-position rows).

Existing (date, player_id) keys in the snapshot files always win, so running
this never clobbers live rows.

Usage: python -m pipeline.backfill
"""

import time

from .config import YEAR
from .mlb_api import fetch_game_log, fetch_team_stats
from .snapshots import append_rows
from .stats import innings_to_float


def _backfill_hitting(year: int) -> list[dict]:
    pool = [(s["player"]["id"], s["player"]["fullName"])
            for s in fetch_team_stats("hitting", year)
            if (s["stat"].get("plateAppearances") or 0) > 0]
    rows = []
    for pid, name in pool:
        log = fetch_game_log(pid, "hitting", year)
        c = dict.fromkeys(
            ("G", "PA", "AB", "H", "2B", "3B", "HR", "RBI", "R", "SB",
             "BB", "SO", "HBP", "SF"), 0)
        for game in sorted(log, key=lambda s: s["date"]):
            st = game["stat"]
            c["G"]   += 1
            c["PA"]  += st.get("plateAppearances") or 0
            c["AB"]  += st.get("atBats") or 0
            c["H"]   += st.get("hits") or 0
            c["2B"]  += st.get("doubles") or 0
            c["3B"]  += st.get("triples") or 0
            c["HR"]  += st.get("homeRuns") or 0
            c["RBI"] += st.get("rbi") or 0
            c["R"]   += st.get("runs") or 0
            c["SB"]  += st.get("stolenBases") or 0
            c["BB"]  += st.get("baseOnBalls") or 0
            c["SO"]  += st.get("strikeOuts") or 0
            c["HBP"] += st.get("hitByPitch") or 0
            c["SF"]  += st.get("sacFlies") or 0

            tb = c["H"] + c["2B"] + 2 * c["3B"] + 3 * c["HR"]
            obp_denom = c["AB"] + c["BB"] + c["HBP"] + c["SF"]
            avg = c["H"] / c["AB"] if c["AB"] else None
            obp = (c["H"] + c["BB"] + c["HBP"]) / obp_denom if obp_denom else None
            slg = tb / c["AB"] if c["AB"] else None
            rows.append({
                "date": game["date"], "season": year, "source": "backfill",
                "player_id": pid, "Name": name,
                "G": c["G"], "PA": c["PA"], "AB": c["AB"], "H": c["H"],
                "2B": c["2B"], "3B": c["3B"], "HR": c["HR"], "RBI": c["RBI"],
                "R": c["R"], "SB": c["SB"], "BB": c["BB"], "SO": c["SO"],
                "AVG": round(avg, 3) if avg is not None else None,
                "OBP": round(obp, 3) if obp is not None else None,
                "SLG": round(slg, 3) if slg is not None else None,
                "OPS": round(obp + slg, 3) if obp is not None and slg is not None else None,
                "BB%": round(c["BB"] / c["PA"], 4) if c["PA"] else None,
                "K%":  round(c["SO"] / c["PA"], 4) if c["PA"] else None,
                "bWAR": None, "fWAR": None,
            })
        print(f"  {name}: {len(log)} games")
        time.sleep(0.3)  # politeness to the free API
    return rows


def _backfill_pitching(year: int) -> list[dict]:
    pool = [(s["player"]["id"], s["player"]["fullName"])
            for s in fetch_team_stats("pitching", year)]
    rows = []
    for pid, name in pool:
        log = fetch_game_log(pid, "pitching", year)
        c = dict.fromkeys(
            ("G", "GS", "outs", "W", "L", "SV", "BF", "ER", "H", "BB", "SO", "HR"), 0)
        for game in sorted(log, key=lambda s: s["date"]):
            st = game["stat"]
            c["G"]  += 1
            c["GS"] += st.get("gamesStarted") or 0
            ip = innings_to_float(st.get("inningsPitched"))
            c["outs"] += round((ip or 0) * 3)
            c["W"]  += st.get("wins") or 0
            c["L"]  += st.get("losses") or 0
            c["SV"] += st.get("saves") or 0
            c["BF"] += st.get("battersFaced") or 0
            c["ER"] += st.get("earnedRuns") or 0
            c["H"]  += st.get("hits") or 0
            c["BB"] += st.get("baseOnBalls") or 0
            c["SO"] += st.get("strikeOuts") or 0
            c["HR"] += st.get("homeRuns") or 0

            ip_f = c["outs"] / 3.0
            rows.append({
                "date": game["date"], "season": year, "source": "backfill",
                "player_id": pid, "Name": name,
                "G": c["G"], "GS": c["GS"],
                "IP": f"{c['outs'] // 3}.{c['outs'] % 3}", "IP_f": round(ip_f, 2),
                "W": c["W"], "L": c["L"], "SV": c["SV"], "BF": c["BF"],
                "ER": c["ER"], "BB": c["BB"], "SO": c["SO"],
                "ERA":  round(c["ER"] * 9 / ip_f, 2) if ip_f else None,
                "WHIP": round((c["BB"] + c["H"]) / ip_f, 2) if ip_f else None,
                "K/9":  round(c["SO"] * 9 / ip_f, 2) if ip_f else None,
                "BB/9": round(c["BB"] * 9 / ip_f, 2) if ip_f else None,
                "HR/9": round(c["HR"] * 9 / ip_f, 2) if ip_f else None,
                "K%":  round(c["SO"] / c["BF"], 4) if c["BF"] else None,
                "BB%": round(c["BB"] / c["BF"], 4) if c["BF"] else None,
                "bWAR": None, "fWAR": None,
            })
        print(f"  {name}: {len(log)} games")
        time.sleep(0.3)
    return rows


def main() -> None:
    print(f"Backfilling {YEAR} season from MLB game logs…")
    print("Hitting:")
    n = append_rows("hitting", _backfill_hitting(YEAR), YEAR)
    print(f"✓ hitting: {n} backfill rows added")
    print("Pitching:")
    n = append_rows("pitching", _backfill_pitching(YEAR), YEAR)
    print(f"✓ pitching: {n} backfill rows added")


if __name__ == "__main__":
    main()
