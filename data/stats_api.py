"""Stats fetchers: BallDontLie (NBA) + SportsData.io (NFL/MLB/NHL)."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)

_BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"
_SPORTSDATA_BASE = "https://api.sportsdata.io/v3"


# ── BallDontLie — NBA ─────────────────────────────────────────────────────────

class BallDontLieClient:
    def __init__(self, api_key: str = config.BALLDONTLIE_API_KEY):
        headers = {"Authorization": api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=_BALLDONTLIE_BASE,
            headers=headers,
            timeout=15,
        )

    def _get(self, path: str, params: dict | None = None) -> Any:
        r = self._client.get(path, params=params or {})
        r.raise_for_status()
        return r.json()

    @lru_cache(maxsize=32)
    def get_teams(self) -> list[dict]:
        return self._get("/teams")["data"]

    def get_season_averages(self, season: int, player_ids: list[int]) -> list[dict]:
        params = {"season": season}
        for pid in player_ids:
            params.setdefault("player_ids[]", [])
            params["player_ids[]"].append(pid)  # type: ignore[arg-type]
        return self._get("/season_averages", params).get("data", [])

    def get_team_stats(self, season: int) -> list[dict]:
        """Returns per-game offensive stats per team."""
        raw = self._get("/games", {"seasons[]": season, "per_page": 100})
        return raw.get("data", [])

    def get_recent_games(self, team_id: int, last_n: int = 10) -> list[dict]:
        raw = self._get("/games", {
            "team_ids[]": team_id,
            "per_page": last_n,
        })
        return raw.get("data", [])

    def close(self):
        self._client.close()


# ── SportsData.io — NFL / MLB / NHL ───────────────────────────────────────────

class SportsDataClient:
    SPORT_CONFIG = {
        "NFL": ("nfl", config.SPORTSDATA_NFL_KEY),
        "MLB": ("mlb", config.SPORTSDATA_MLB_KEY),
        "NHL": ("nhl", config.SPORTSDATA_NHL_KEY),
    }

    def __init__(self, sport: str):
        sport = sport.upper()
        if sport not in self.SPORT_CONFIG:
            raise ValueError(f"Unsupported sport: {sport}")
        path_prefix, api_key = self.SPORT_CONFIG[sport]
        if not api_key:
            logger.warning("No SportsData API key for %s — stats will be unavailable.", sport)
        self._sport = sport
        self._prefix = path_prefix
        # SportsData.io URL structure: /v3/{sport}/stats/json/...
        self._client = httpx.Client(
            base_url=f"{_SPORTSDATA_BASE}/{path_prefix}/stats/json",
            params={"key": api_key},
            timeout=15,
        )

    def _get(self, path: str, params: dict | None = None) -> Any:
        r = self._client.get(path, params=params or {})
        r.raise_for_status()
        return r.json()

    def get_standings(self, season: str) -> list[dict]:
        return self._get(f"/Standings/{season}")

    def get_team_season_stats(self, season: str) -> list[dict]:
        return self._get(f"/TeamSeasonStats/{season}REG")

    def get_schedule(self, season: str) -> list[dict]:
        return self._get(f"/Games/{season}")

    def get_scores_by_week(self, season: str, week: int) -> list[dict]:
        """NFL-specific week scores."""
        return self._get(f"/ScoresByWeek/{season}REG/{week}")

    def close(self):
        self._client.close()


# ── Unified stats loader ───────────────────────────────────────────────────────

def load_team_stats(sport: str, season: str | int) -> dict[str, dict]:
    """
    Returns {team_name: {stat_key: value}} for downstream models.
    Falls back to empty dict if API keys are missing.
    """
    sport = sport.upper()
    stats: dict[str, dict] = {}

    if sport == "NBA":
        client = BallDontLieClient()
        try:
            games = client.get_team_stats(int(season))
            for g in games:
                for side in ("home_team", "visitor_team"):
                    team = g.get(side, {}).get("full_name", "")
                    if team:
                        stats.setdefault(team, {"games": 0, "pts": 0})
                        key = "home_team_score" if side == "home_team" else "visitor_team_score"
                        pts = g.get(key, 0) or 0
                        stats[team]["pts"] += pts
                        stats[team]["games"] += 1
            # compute averages
            for team_data in stats.values():
                g_count = team_data["games"] or 1
                team_data["pts_per_game"] = round(team_data["pts"] / g_count, 2)
        except Exception as exc:
            logger.warning("NBA stats unavailable: %s", exc)
        finally:
            client.close()
    else:
        client = SportsDataClient(sport)
        try:
            raw = client.get_team_season_stats(str(season))
            for row in raw:
                name = row.get("Team", row.get("Name", ""))
                if name:
                    stats[name] = row
        except Exception as exc:
            logger.warning("%s stats unavailable: %s", sport, exc)
        finally:
            client.close()

    return stats
