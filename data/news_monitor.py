"""
news_monitor.py — fetch injury/roster news from ESPN, cached daily.

Returns a dict of {normalised_team_name: [injury_items]} so EV Picks cards
can display a warning badge when a key player is out or doubtful.

Statuses we surface (ordered by severity):
  OUT, DOUBTFUL, QUESTIONABLE, PROBABLE, DAY-TO-DAY
"""
from __future__ import annotations

import logging
import re
from datetime import date

import httpx

from data.daily_cache import load as cache_load, save as cache_save
from data.espn_api import SPORT_MAP, ESPN_BASE

logger = logging.getLogger(__name__)

_ALERT_STATUSES = {"out", "doubtful", "questionable"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip()))


def fetch_injuries(sport: str) -> dict[str, list[dict]]:
    """
    Return {team_name: [injury_dict]} for a sport.
    injury_dict keys: player, position, status, detail
    Cached daily — one call per sport per day.
    """
    cache_key = f"injuries_{sport}_{date.today().isoformat()}"
    cached = cache_load(cache_key)
    if cached is not None:
        return cached

    sport_slug, league_slug = SPORT_MAP.get(sport.upper(), ("", ""))
    if not sport_slug:
        return {}

    url = f"{ESPN_BASE}/{sport_slug}/{league_slug}/injuries"
    try:
        r = httpx.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("ESPN injuries fetch failed %s: %s", sport, exc)
        return {}

    result: dict[str, list[dict]] = {}

    # ESPN injury response structure varies slightly by sport.
    # Top-level key is usually "injuries" containing a list, or it may be
    # nested under "items". We handle both.
    raw_items = data.get("injuries") or data.get("items") or []
    if not raw_items and isinstance(data, list):
        raw_items = data

    for item in raw_items:
        athlete  = item.get("athlete") or item.get("player") or {}
        team_obj = (
            item.get("team")
            or athlete.get("team")
            or {}
        )
        status_raw = (
            item.get("status")
            or (item.get("type") or {}).get("description")
            or ""
        )
        detail_obj = item.get("details") or {}
        detail_str = detail_obj.get("detail") or detail_obj.get("type") or ""

        player_name = athlete.get("displayName") or athlete.get("fullName") or ""
        position    = (athlete.get("position") or {}).get("abbreviation") or ""
        team_name   = team_obj.get("displayName") or team_obj.get("name") or ""
        status      = status_raw.strip()

        if not player_name or not team_name:
            continue
        if status.lower() not in _ALERT_STATUSES:
            continue

        entry = {
            "player":   player_name,
            "position": position,
            "status":   status,
            "detail":   detail_str,
        }
        key = _norm(team_name)
        result.setdefault(key, []).append(entry)

    cache_save(cache_key, result)
    logger.info("ESPN injuries %s: %d teams with news", sport, len(result))
    return result


def fetch_all_injuries(sports: list[str] | None = None) -> dict[str, list[dict]]:
    """
    Merge injuries across all (or specified) sports into one dict.
    Useful for a single lookup across the whole card list.
    """
    from config import SUPPORTED_SPORTS
    target = sports or list(SUPPORTED_SPORTS.keys())
    merged: dict[str, list[dict]] = {}
    for sp in target:
        for team, items in fetch_injuries(sp).items():
            merged.setdefault(team, []).extend(items)
    return merged


def injuries_for_teams(
    home_team: str,
    away_team: str,
    injury_map: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    """
    Look up injury news for both teams in a game.
    Returns (home_injuries, away_injuries) — each a (possibly empty) list.
    """
    def _lookup(team: str) -> list[dict]:
        k = _norm(team)
        # Exact normalised match
        if k in injury_map:
            return injury_map[k]
        # Last-word (nickname) fallback
        nick = k.split()[-1]
        for key, val in injury_map.items():
            if key.split()[-1] == nick:
                return val
        return []

    return _lookup(home_team), _lookup(away_team)
