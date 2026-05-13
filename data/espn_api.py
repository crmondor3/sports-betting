"""
ESPN unofficial API — free, no key required.
Provides: team records, recent form, H2H history, rest days, scoring averages.
All data is cached daily so it only hits ESPN once per day.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import get_close_matches
from typing import Any

import httpx

from data.daily_cache import load as cache_load, save as cache_save

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

SPORT_MAP: dict[str, tuple[str, str]] = {
    "NBA":   ("basketball", "nba"),
    "MLB":   ("baseball", "mlb"),
    "NHL":   ("hockey", "nhl"),
    "NFL":   ("football", "nfl"),
    "WNBA":  ("basketball", "wnba"),
    "NCAAF": ("football", "college-football"),
    "NCAAB": ("basketball", "mens-college-basketball"),
}


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class TeamForm:
    team_name: str = ""
    wins: int = 0
    losses: int = 0
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    last_10_wins: int = 0
    last_10_losses: int = 0
    streak: str = ""           # "W3" or "L2"
    ppg: float = 0.0           # points per game (scored)
    papg: float = 0.0          # points allowed per game
    last_game_date: date | None = None
    rest_days: int = 1

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total else 0.5

    @property
    def home_win_pct(self) -> float:
        total = self.home_wins + self.home_losses
        return self.home_wins / total if total else 0.5

    @property
    def away_win_pct(self) -> float:
        total = self.away_wins + self.away_losses
        return self.away_wins / total if total else 0.5

    @property
    def l10_label(self) -> str:
        return f"{self.last_10_wins}-{self.last_10_losses} (L10)"

    @property
    def record_label(self) -> str:
        return f"{self.wins}-{self.losses}"


@dataclass
class HeadToHead:
    home_team: str = ""
    away_team: str = ""
    home_wins: int = 0
    away_wins: int = 0
    meetings: int = 0
    home_ppg: float = 0.0
    away_ppg: float = 0.0

    @property
    def home_h2h_win_pct(self) -> float:
        return self.home_wins / self.meetings if self.meetings else 0.5

    @property
    def label(self) -> str:
        if self.meetings == 0:
            return "No recent H2H data"
        return f"{self.home_team} {self.home_wins}-{self.away_wins} in last {self.meetings}"


@dataclass
class MatchupData:
    home_team: str
    away_team: str
    sport: str
    home_form: TeamForm = field(default_factory=TeamForm)
    away_form: TeamForm = field(default_factory=TeamForm)
    h2h: HeadToHead = field(default_factory=HeadToHead)
    data_quality: str = "cold"   # "full" | "partial" | "cold"


# ── ESPN client ────────────────────────────────────────────────────────────────

class ESPNClient:
    def __init__(self):
        self._http = httpx.Client(timeout=15)
        self._team_id_cache: dict[str, dict[str, int]] = {}  # sport → {name: id}

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _get(self, url: str, params: dict | None = None) -> Any:
        r = self._http.get(url, params=params or {})
        r.raise_for_status()
        return r.json()

    def _sport_path(self, sport: str) -> str:
        s, l = SPORT_MAP[sport.upper()]
        return f"{ESPN_BASE}/{s}/{l}"

    # ── Team ID lookup ─────────────────────────────────────────────────────────

    def _fetch_teams(self, sport: str) -> dict[str, int]:
        cache_key = f"espn_teams_{sport}"
        cached = cache_load(cache_key)
        if cached:
            return cached

        url = f"{self._sport_path(sport)}/teams"
        data = self._get(url, {"limit": 100})
        teams: dict[str, int] = {}
        for item in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
            t = item.get("team", {})
            name = t.get("displayName", "")
            tid = int(t.get("id", 0))
            if name and tid:
                teams[name] = tid
        cache_save(cache_key, teams)
        return teams

    def _resolve_team_id(self, sport: str, team_name: str) -> int | None:
        teams = self._fetch_teams(sport)
        # Exact match
        if team_name in teams:
            return teams[team_name]
        # Fuzzy match
        normalised = {_norm(k): v for k, v in teams.items()}
        matches = get_close_matches(_norm(team_name), normalised.keys(), n=1, cutoff=0.55)
        if matches:
            return normalised[matches[0]]
        # Last-word match (team nickname only)
        nick = team_name.split()[-1].lower()
        for k, v in teams.items():
            if k.split()[-1].lower() == nick:
                return v
        return None

    # ── Schedule / results ─────────────────────────────────────────────────────

    def _fetch_team_schedule(self, sport: str, team_id: int) -> list[dict]:
        cache_key = f"espn_sched_{sport}_{team_id}"
        cached = cache_load(cache_key)
        if cached is not None:
            return cached

        url = f"{self._sport_path(sport)}/teams/{team_id}/schedule"
        try:
            data = self._get(url)
        except Exception as exc:
            logger.warning("ESPN schedule fetch failed team_id=%d: %s", team_id, exc)
            return []

        events = data.get("events", [])
        games = []
        for ev in events:
            comps = ev.get("competitions", [{}])
            if not comps:
                continue
            comp = comps[0]
            status = comp.get("status", {}).get("type", {}).get("completed", False)
            if not status:
                continue   # skip unplayed games
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            games.append({
                "date": ev.get("date", ""),
                "home_team": home.get("team", {}).get("displayName", ""),
                "away_team": away.get("team", {}).get("displayName", ""),
                "home_score": _to_int(home.get("score", "0")),
                "away_score": _to_int(away.get("score", "0")),
                "home_winner": home.get("winner", False),
            })
        cache_save(cache_key, games)
        return games

    # ── Public: get team form ──────────────────────────────────────────────────

    def get_team_form(self, sport: str, team_name: str, today: date | None = None) -> TeamForm:
        today = today or date.today()
        form = TeamForm(team_name=team_name)
        tid = self._resolve_team_id(sport, team_name)
        if tid is None:
            logger.warning("ESPN: could not resolve '%s' (%s)", team_name, sport)
            return form

        games = self._fetch_team_schedule(sport, tid)
        if not games:
            return form

        scored: list[float] = []
        allowed: list[float] = []
        recent: list[bool] = []   # True = win
        last_date: date | None = None

        for g in games:
            try:
                gdate = datetime.fromisoformat(g["date"].rstrip("Z")).date()
            except Exception:
                continue
            if gdate > today:
                continue

            is_home = _norm(g["home_team"]) == _norm(team_name)
            won = g["home_winner"] if is_home else not g["home_winner"]
            pts = g["home_score"] if is_home else g["away_score"]
            opp_pts = g["away_score"] if is_home else g["home_score"]

            if won:
                form.wins += 1
                if is_home: form.home_wins += 1
                else: form.away_wins += 1
            else:
                form.losses += 1
                if is_home: form.home_losses += 1
                else: form.away_losses += 1

            scored.append(pts)
            allowed.append(opp_pts)
            recent.append(won)

            if last_date is None or gdate > last_date:
                last_date = gdate

        # Last 10 games
        last10 = recent[-10:]
        form.last_10_wins = sum(last10)
        form.last_10_losses = len(last10) - form.last_10_wins

        # Streak
        if recent:
            last = recent[-1]
            count = 0
            for r in reversed(recent):
                if r == last:
                    count += 1
                else:
                    break
            form.streak = f"{'W' if last else 'L'}{count}"

        # Scoring averages (last 15 games for recency)
        last15_scored = scored[-15:]
        last15_allowed = allowed[-15:]
        form.ppg = round(sum(last15_scored) / len(last15_scored), 1) if last15_scored else 0.0
        form.papg = round(sum(last15_allowed) / len(last15_allowed), 1) if last15_allowed else 0.0

        # Rest days
        form.last_game_date = last_date
        if last_date:
            form.rest_days = max((today - last_date).days, 0)

        return form

    # ── Public: H2H ───────────────────────────────────────────────────────────

    def get_h2h(self, sport: str, home_team: str, away_team: str,
                last_n: int = 10) -> HeadToHead:
        h2h = HeadToHead(home_team=home_team, away_team=away_team)
        tid = self._resolve_team_id(sport, home_team)
        if tid is None:
            return h2h

        games = self._fetch_team_schedule(sport, tid)
        home_scores: list[float] = []
        away_scores: list[float] = []

        for g in games[-50:]:  # search last 50 in schedule
            g_home = _norm(g["home_team"])
            g_away = _norm(g["away_team"])
            hn = _norm(home_team)
            an = _norm(away_team)
            if not ((g_home == hn and g_away == an) or (g_home == an and g_away == hn)):
                continue

            h2h.meetings += 1
            home_actual = _norm(g["home_team"]) == hn
            if (home_actual and g["home_winner"]) or (not home_actual and not g["home_winner"]):
                h2h.home_wins += 1
            else:
                h2h.away_wins += 1

            if home_actual:
                home_scores.append(g["home_score"])
                away_scores.append(g["away_score"])
            else:
                home_scores.append(g["away_score"])
                away_scores.append(g["home_score"])

            if h2h.meetings >= last_n:
                break

        if home_scores:
            h2h.home_ppg = round(sum(home_scores) / len(home_scores), 1)
            h2h.away_ppg = round(sum(away_scores) / len(away_scores), 1)

        return h2h

    # ── Public: scoreboard by date ───────────────────────────────────────────

    def get_scoreboard(self, sport: str, game_date: date) -> list[dict]:
        """
        Return completed games for a sport on a given calendar date.
        Each dict has: event_id, date, home_team, away_team, home_score,
        away_score, home_winner.
        Results are cached so the same date is only fetched once.
        """
        cache_key = f"espn_scoreboard_{sport}_{game_date.isoformat()}"
        cached = cache_load(cache_key)
        if cached is not None:
            return cached

        sport_slug, league_slug = SPORT_MAP.get(sport.upper(), ("", ""))
        if not sport_slug:
            return []

        url = f"{ESPN_BASE}/{sport_slug}/{league_slug}/scoreboard"
        try:
            data = self._get(url, {"dates": game_date.strftime("%Y%m%d"), "limit": 100})
        except Exception as exc:
            logger.warning("ESPN scoreboard failed %s %s: %s", sport, game_date, exc)
            return []

        games = []
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
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            games.append({
                "event_id":   ev.get("id", ""),
                "date":       ev.get("date", game_date.isoformat()),
                "home_team":  home.get("team", {}).get("displayName", ""),
                "away_team":  away.get("team", {}).get("displayName", ""),
                "home_score": _to_int(home.get("score", "0")),
                "away_score": _to_int(away.get("score", "0")),
                "home_winner": home.get("winner", False),
            })

        cache_save(cache_key, games)
        logger.info("ESPN scoreboard %s %s: %d completed games", sport, game_date, len(games))
        return games

    # ── Public: full matchup ──────────────────────────────────────────────────

    def get_matchup(self, sport: str, home_team: str, away_team: str) -> MatchupData:
        cache_key = f"matchup_{sport}_{_norm(home_team)}_{_norm(away_team)}"
        cached = cache_load(cache_key)
        if cached:
            # Reconstruct from dict
            m = MatchupData(
                home_team=cached["home_team"],
                away_team=cached["away_team"],
                sport=cached["sport"],
                data_quality=cached.get("data_quality", "cold"),
            )
            m.home_form = TeamForm(**cached["home_form"])
            m.away_form = TeamForm(**cached["away_form"])
            m.h2h = HeadToHead(**cached["h2h"])
            return m

        try:
            home_form = self.get_team_form(sport, home_team)
            away_form = self.get_team_form(sport, away_team)
            h2h = self.get_h2h(sport, home_team, away_team)

            games_found = (home_form.wins + home_form.losses) > 0
            quality = "full" if games_found else "cold"

            m = MatchupData(
                home_team=home_team,
                away_team=away_team,
                sport=sport,
                home_form=home_form,
                away_form=away_form,
                h2h=h2h,
                data_quality=quality,
            )
        except Exception as exc:
            logger.warning("ESPN matchup failed %s vs %s: %s", home_team, away_team, exc)
            m = MatchupData(home_team=home_team, away_team=away_team, sport=sport)

        # Persist as plain dict
        cache_save(cache_key, {
            "home_team": m.home_team,
            "away_team": m.away_team,
            "sport": m.sport,
            "data_quality": m.data_quality,
            "home_form": m.home_form.__dict__,
            "away_form": m.away_form.__dict__,
            "h2h": m.h2h.__dict__,
        })
        return m

    def close(self):
        self._http.close()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip()))


def _to_int(val: Any) -> int:
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return 0
