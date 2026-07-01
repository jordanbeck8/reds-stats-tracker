"""WAR fetchers — bWAR from Baseball-Reference, fWAR from FanGraphs.

Both sources are flaky from datacenter IPs, so each has a committed JSON cache
as the final fallback tier. Caches are refreshed on every successful live fetch.
"""

import json
import time
from io import StringIO

import pandas as pd
import requests

from .config import BROWSER_UA, BWAR_URLS, CACHE_DIR, FG_LEADERS_URL, FG_TEAM_ID, TEAM
from .names import normalize_name

_BWAR_CACHE = CACHE_DIR / "bwar_cache.json"
_FWAR_CACHE = CACHE_DIR / "fwar_cache.json"


def _cache_write(cache_path, cache_key: str, result: pd.DataFrame, cols: list) -> None:
    data = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    data[cache_key] = result[cols].to_dict(orient="records")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2))


def _cache_read(cache_path, cache_key: str) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
    except Exception:
        return None
    if cache_key not in data:
        return None
    return pd.DataFrame(data[cache_key])


# ---------------------------------------------------------------------------
# bWAR — Baseball-Reference
# ---------------------------------------------------------------------------

def get_bwar(source: str, label: str, year: int) -> pd.DataFrame:
    """Fetch bWAR from Baseball-Reference for CIN in the given year.

    Strategy:
      1. Direct HTTP request with browser UA (BRef blocks default pybaseball UA).
      2. pybaseball fallback.
      3. If both fail (e.g. GitHub Actions IP blocks), load bwar_cache.json.
    On success, writes result to bwar_cache.json so CI always has fresh data.
    Returns columns: Name, bWAR, name_key.
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

    print(f"  Fetching bWAR ({label}) from Baseball-Reference…")

    # Attempt 1: direct request with browser User-Agent
    try:
        resp = requests.get(
            BWAR_URLS[source], headers={"User-Agent": BROWSER_UA}, timeout=30
        )
        resp.raise_for_status()
        # BRef serves UTF-8 but may advertise no charset — decode explicitly.
        result = _parse(pd.read_csv(StringIO(resp.content.decode("utf-8", errors="replace"))))
        _cache_write(_BWAR_CACHE, cache_key, result, ["Name", "bWAR"])
        return result
    except Exception as exc:
        print(f"  Direct fetch failed ({exc}), trying pybaseball…")

    # Attempt 2: pybaseball
    try:
        import pybaseball
        pybaseball_func = pybaseball.bwar_bat if source == "bat" else pybaseball.bwar_pitch
        result = _parse(pybaseball_func(return_all=False))
        _cache_write(_BWAR_CACHE, cache_key, result, ["Name", "bWAR"])
        return result
    except Exception as exc:
        print(f"  pybaseball failed ({exc}), loading cache…")

    # Attempt 3: local cache (committed to repo, always available on CI)
    cached = _cache_read(_BWAR_CACHE, cache_key)
    if cached is not None:
        print(f"  Using cached bWAR ({label}) — live fetch unavailable.")
        cached["name_key"] = cached["Name"].apply(normalize_name)
        return cached

    print(f"  Warning: bWAR ({label}) unavailable — no live data or cache.")
    return pd.DataFrame(columns=["Name", "bWAR", "name_key"])


# ---------------------------------------------------------------------------
# fWAR — FanGraphs
# ---------------------------------------------------------------------------

def get_fwar(source: str, label: str, year: int) -> pd.DataFrame:
    """Fetch fWAR from the FanGraphs leaderboard JSON API for CIN.

    FanGraphs sits behind Cloudflare and 403s plain requests AND pybaseball
    (verified 2026-07-01), so the primary route is cloudscraper with retries.
    Falls back to the committed fwar_cache.json.
    Returns columns: Name, fWAR, name_key, and player_id when the API
    provides xMLBAMID (MLB person id — preferred join key).
    """
    cache_key = f"{source}_{year}"
    params = {
        "pos": "all",
        "stats": "bat" if source == "bat" else "pit",
        "lg": "all",
        "qual": "0",
        "season": str(year),
        "season1": str(year),
        "month": "0",
        "team": str(FG_TEAM_ID),
        "pageitems": "200",
        "pagenum": "1",
        "ind": "0",
        "type": "8",
    }

    print(f"  Fetching fWAR ({label}) from FanGraphs…")

    result = None
    try:
        import cloudscraper
        for attempt in range(4):
            try:
                # Fresh scraper per attempt — Cloudflare flags reused sessions
                # after the first request (observed 2026-07-01).
                scraper = cloudscraper.create_scraper()
                resp = scraper.get(FG_LEADERS_URL, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
                records = []
                for r in rows:
                    if r.get("WAR") is None:
                        continue
                    records.append({
                        "Name": r.get("PlayerName"),
                        "fWAR": float(r["WAR"]),
                        "player_id": r.get("xMLBAMID"),
                    })
                result = pd.DataFrame(records)
                break
            except Exception as exc:
                print(f"  fWAR attempt {attempt + 1} failed ({exc}), retrying…")
                time.sleep(5 * (attempt + 1))
    except ImportError:
        print("  cloudscraper not installed — skipping live fWAR fetch.")

    if result is not None and not result.empty:
        _cache_write(_FWAR_CACHE, cache_key, result, ["Name", "fWAR", "player_id"])
        result["name_key"] = result["Name"].apply(normalize_name)
        return result

    cached = _cache_read(_FWAR_CACHE, cache_key)
    if cached is not None:
        print(f"  Using cached fWAR ({label}) — live fetch unavailable.")
        cached["name_key"] = cached["Name"].apply(normalize_name)
        return cached

    print(f"  Warning: fWAR ({label}) unavailable — no live data or cache.")
    return pd.DataFrame(columns=["Name", "fWAR", "name_key", "player_id"])


def merge_war(df: pd.DataFrame, bwar: pd.DataFrame, fwar: pd.DataFrame) -> pd.DataFrame:
    """Join bWAR (by normalized name) and fWAR (by MLB id, else name) onto df.

    df must have Name and player_id columns. Adds bWAR and fWAR columns.
    """
    df = df.copy()
    df["name_key"] = df["Name"].apply(normalize_name)

    if not bwar.empty:
        df = df.merge(bwar[["name_key", "bWAR"]], on="name_key", how="left")
    else:
        df["bWAR"] = None

    if not fwar.empty and fwar["player_id"].notna().any():
        fw = fwar.dropna(subset=["player_id"]).copy()
        fw["player_id"] = fw["player_id"].astype(int)
        df = df.merge(fw[["player_id", "fWAR"]], on="player_id", how="left")
        # Names without an id match get a second chance via name key.
        missing = df["fWAR"].isna()
        if missing.any():
            by_name = dict(zip(fwar["name_key"], fwar["fWAR"]))
            df.loc[missing, "fWAR"] = df.loc[missing, "name_key"].map(by_name)
    elif not fwar.empty:
        df = df.merge(fwar[["name_key", "fWAR"]], on="name_key", how="left")
    else:
        df["fWAR"] = None

    return df.drop(columns=["name_key"])
