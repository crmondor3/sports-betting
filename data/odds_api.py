"""
The Odds API v4 client — DraftKings only, pulled once per day.
Subsequent calls within the same calendar day return cached data.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

import config
from data.daily_cache import load as cache_load, save as cache_save, is_cached

logger = logging.getLogger(__name__)

REGIONS = "us"
ODDS_FORMAT = "american"
TARGET_BOOK = config.TARGET_BOOK   # "draftkings"


@dataclass
class OddsGame:
    game_id: str
    sport: str
    commence_time: datetime
    home_team: str
    away_team: str
    # market_key → list of outcomes for DraftKings only
    markets: dict[str, list[dict]] = field(default_factory=dict)

    def get_line(self, market: str, side: str) -> float | None:
        """Return American odds for a side in a given market, or None."""
        for o in self.markets.get(market, []):
            if o.get("name", "").lower() == side.lower():
                return float(o["price"])
        return None

    def get_spread_line(self, team: str) -> tuple[float, float] | None:
        """Return (spread_point, american_price) for a team, or None."""
        for o in self.markets.get("spreads", []):
            if o.get("name", "") == team:
                return float(o.get("point", 0)), float(o["price"])
        return None

    def get_total_line(self) -> float | None:
        """Return the over/under total point from DraftKings."""
        for o in self.markets.get("totals", []):
            if o.get("name", "").lower() == "over":
                return o.get("point")
        return None


class OddsAPIClient:
    def __init__(self, api_key: str = config.ODDS_API_KEY):
        if not api_key:
            raise ValueError("ODDS_API_KEY not set. Copy .env.template → .env and add your key.")
        self._key = api_key
        self._client = httpx.Client(
            base_url=config.ODDS_API_BASE,
            params={"apiKey": self._key},
            timeout=15,
        )
        self.requests_remaining: int | None = None
        self.requests_used: int | None = None

    def _get(self, path: str, params: dict | None = None, retries: int = 3) -> Any:
        for attempt in range(retries):
            try:
                r = self._client.get(path, params=params or {})
                self.requests_remaining = int(r.headers.get("x-requests-remaining", -1))
                self.requests_used = int(r.headers.get("x-requests-used", -1))
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Rate limited — retrying in %ss", wait)
                    time.sleep(wait)
                else:
                    raise
            except httpx.RequestError as exc:
                logger.error("Request error: %s", exc)
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed after {retries} retries")

    def _raw_to_games(self, raw: list[dict], sport_key: str) -> list[OddsGame]:
        games: list[OddsGame] = []
        for item in raw:
            game = OddsGame(
                game_id=item["id"],
                sport=sport_key,
                commence_time=datetime.fromisoformat(item["commence_time"].rstrip("Z")),
                home_team=item["home_team"],
                away_team=item["away_team"],
            )
            # Extract only DraftKings markets
            for bm in item.get("bookmakers", []):
                if bm["key"] != TARGET_BOOK:
                    continue
                for market in bm.get("markets", []):
                    game.markets[market["key"]] = market["outcomes"]
                break   # only one bookmaker we care about
            # Only include games where DraftKings has odds
            if game.markets:
                games.append(game)
        return games

    def get_odds(
        self,
        sport_key: str,
        force_refresh: bool = False,
    ) -> list[OddsGame]:
        """
        Fetch DraftKings odds for a sport.
        Returns cached data if already pulled today, unless force_refresh=True.
        """
        cache_key = f"dk_odds_{sport_key}"
        if not force_refresh:
            cached = cache_load(cache_key)
            if cached is not None:
                games = [
                    OddsGame(
                        game_id=g["game_id"],
                        sport=g["sport"],
                        commence_time=datetime.fromisoformat(g["commence_time"]),
                        home_team=g["home_team"],
                        away_team=g["away_team"],
                        markets=g["markets"],
                    )
                    for g in cached
                ]
                logger.info("Loaded %d %s games from daily cache", len(games), sport_key)
                return games

        # Fetch fresh from API
        params = {
            "regions": REGIONS,
            "markets": ",".join(config.MARKETS),
            "oddsFormat": ODDS_FORMAT,
            "dateFormat": "iso",
            "bookmakers": TARGET_BOOK,
        }
        raw = self._get(f"/sports/{sport_key}/odds", params)
        games = self._raw_to_games(raw, sport_key)

        # Persist to daily cache as plain dicts
        serialisable = [
            {
                "game_id": g.game_id,
                "sport": g.sport,
                "commence_time": g.commence_time.isoformat(),
                "home_team": g.home_team,
                "away_team": g.away_team,
                "markets": g.markets,
            }
            for g in games
        ]
        cache_save(cache_key, serialisable)

        logger.info(
            "Fetched %d %s games from DraftKings — %s requests remaining",
            len(games), sport_key, self.requests_remaining,
        )
        return games

    def get_all_sports_odds(self, force_refresh: bool = False) -> dict[str, list[OddsGame]]:
        result: dict[str, list[OddsGame]] = {}
        for label, key in config.SUPPORTED_SPORTS.items():
            try:
                result[label] = self.get_odds(key, force_refresh=force_refresh)
            except Exception as exc:
                logger.error("Could not fetch %s: %s", label, exc)
                result[label] = []
        return result

    def get_scores(self, sport_key: str, days_from: int = 2) -> list[dict]:
        """
        Fetch completed game scores for the past `days_from` days.
        Cached daily — one API call per sport per day.
        """
        cache_key = f"scores_{sport_key}_{days_from}d"
        cached = cache_load(cache_key)
        if cached is not None:
            return cached

        raw = self._get(
            f"/sports/{sport_key}/scores",
            {"daysFrom": days_from, "dateFormat": "iso"},
        )
        completed = [g for g in raw if g.get("completed")]
        cache_save(cache_key, completed)
        logger.info(
            "Fetched %d completed %s scores (daysFrom=%d) — %s requests remaining",
            len(completed), sport_key, days_from, self.requests_remaining,
        )
        return completed

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
