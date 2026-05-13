"""
Fetch multi-season historical game logs from ESPN for model training.
Uses the team schedule endpoint with season-year parameter.
Each team+season is cached independently so incremental fetches are fast.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Callable

import httpx

from data.daily_cache import load as cache_load, save as cache_save
from data.espn_api import ESPN_BASE, SPORT_MAP, _norm, _to_int

logger = logging.getLogger(__name__)

# How many past seasons to walk back by default
DEFAULT_YEARS_BACK = 3

# ESPN uses "end-year" convention for multi-sport leagues (NBA 2023-24 → season=2024)
# MLB uses calendar year (2024 → season=2024)
# We pass current_year, current_year-1, etc. which works across all sports.


def _season_years(years_back: int) -> list[int]:
    """Return list of season years to fetch (most recent first)."""
    yr = date.today().year
    return [yr - i for i in range(years_back)]


def _fetch_team_season_games(
    http: httpx.Client,
    sport: str,
    team_id: int,
    season_year: int,
) -> list[dict]:
    """
    Fetch all completed regular-season + playoff games for one team in one season.
    Returns list of game dicts: {date, home_team, away_team, home_score, away_score, home_winner}.
    """
    cache_key = f"hist_sched_{sport}_{team_id}_{season_year}"
    cached = cache_load(cache_key)
    if cached is not None:
        return cached

    sport_slug, league_slug = SPORT_MAP[sport.upper()]
    url = f"{ESPN_BASE}/{sport_slug}/{league_slug}/teams/{team_id}/schedule"
    games: list[dict] = []

    for seasontype in (2, 3):  # 2=regular, 3=postseason
        try:
            r = http.get(url, params={"season": season_year, "seasontype": seasontype}, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.debug("ESPN season fetch team=%d year=%d type=%d: %s", team_id, season_year, seasontype, exc)
            continue

        for ev in data.get("events", []):
            comps = ev.get("competitions", [{}])
            if not comps:
                continue
            comp = comps[0]
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home_c or not away_c:
                continue
            h_name = home_c.get("team", {}).get("displayName", "")
            a_name = away_c.get("team", {}).get("displayName", "")
            if not h_name or not a_name:
                continue
            games.append({
                "date":        ev.get("date", ""),
                "home_team":   h_name,
                "away_team":   a_name,
                "home_score":  _to_int(home_c.get("score", "0")),
                "away_score":  _to_int(away_c.get("score", "0")),
                "home_winner": bool(home_c.get("winner", False)),
            })

    cache_save(cache_key, games)
    return games


def fetch_all_historical_games(
    sport: str,
    years_back: int = DEFAULT_YEARS_BACK,
    progress_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    """
    Collect all unique completed games for a sport across the past N seasons.
    Deduplicates by (date-prefix, home, away).
    Returns games sorted chronologically oldest → newest.
    Returns [] for sports not covered by ESPN (triggers consensus-only mode).
    """
    if sport.upper() not in SPORT_MAP:
        logger.info("%s has no ESPN historical data — consensus-only mode will be used.", sport)
        return []

    agg_cache_key = f"hist_all_{sport}_{years_back}y"
    cached = cache_load(agg_cache_key)
    if cached is not None:
        logger.info("Loaded %d cached historical %s games (%d seasons)", len(cached), sport, years_back)
        return cached

    from data.espn_api import ESPNClient
    espn_client = ESPNClient()
    team_map = espn_client._fetch_teams(sport)  # {display_name: team_id}

    seen: set[tuple] = set()
    all_games: list[dict] = []
    season_years = _season_years(years_back)
    team_ids = list(team_map.values())

    http = httpx.Client(timeout=20)
    try:
        for i, tid in enumerate(team_ids):
            if progress_cb:
                progress_cb(f"{sport}: team {i + 1}/{len(team_ids)}")
            for year in season_years:
                games = _fetch_team_season_games(http, sport, tid, year)
                for g in games:
                    key = (g["date"][:10], _norm(g["home_team"]), _norm(g["away_team"]))
                    if key not in seen:
                        seen.add(key)
                        all_games.append(g)
    finally:
        http.close()
        espn_client.close()

    def _dt(g: dict) -> datetime:
        try:
            return datetime.fromisoformat(g["date"].rstrip("Z"))
        except Exception:
            return datetime.min

    all_games.sort(key=_dt)
    cache_save(agg_cache_key, all_games)
    logger.info("Fetched %d %s games across %d seasons", len(all_games), sport, years_back)
    return all_games
