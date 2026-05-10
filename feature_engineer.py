"""
feature_engineer.py — build predictive feature matrices from historical game data.

Extends the 15-feature set in models/trainer.py with:
  • Rolling averages at windows 5, 10, season
  • Pythagorean win expectation
  • DraftKings implied probability + vig (when odds snapshots are available)
  • Line-movement magnitude and direction
  • Expanded H2H and rest features

All features are computed using ONLY data available before each game (no leakage).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Sport-specific constants ───────────────────────────────────────────────────

_SPORT_ELO_K  = {"NFL": 28, "NBA": 20, "MLB": 10, "NHL": 16, "WNBA": 20}
_SPORT_ELO_HA = {"NFL": 70, "NBA": 35, "MLB": 18, "NHL": 28, "WNBA": 22}
# Pythagorean exponent per sport (Morey's formula for NBA: 13.91, NFL: 2.37, etc.)
_PYTHAG_EXP   = {"NFL": 2.37, "NBA": 13.91, "MLB": 1.83, "NHL": 2.15, "WNBA": 11.5}

FEATURE_NAMES: list[str] = [
    # Elo
    "elo_diff",
    "home_elo",
    "away_elo",
    # Rest / scheduling
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "home_b2b",
    "away_b2b",
    # Recent form (rolling wins)
    "home_l5_wp",
    "home_l10_wp",
    "away_l5_wp",
    "away_l10_wp",
    "form_advantage_l5",
    "form_advantage_l10",
    # Scoring differentials (rolling PPG diff)
    "home_ppg_diff_l5",
    "home_ppg_diff_l10",
    "home_ppg_diff_season",
    "away_ppg_diff_l5",
    "away_ppg_diff_l10",
    "away_ppg_diff_season",
    "ppg_diff_advantage",
    # Home/away splits
    "home_home_wp",
    "away_away_wp",
    # H2H
    "h2h_home_wp",
    # Pythagorean expectation
    "home_pythag",
    "away_pythag",
    "pythag_advantage",
    # Odds-based features (set to 0.0 when no odds data available)
    "dk_home_implied",      # DK moneyline implied prob for home (no-vig)
    "dk_away_implied",
    "dk_home_spread",       # DraftKings spread (home perspective, e.g. -3.5)
    "dk_total",             # over/under total
    "dk_over_implied",      # no-vig probability of Over
    "dk_vig_h2h",           # vig on moneyline market
    "dk_line_move_ml",      # signed change in home implied prob (open→current)
    "dk_line_move_spread",  # signed change in home spread point (open→current)
    "dk_sharp_signal",      # 1 if any sharp move detected
]


# ── FeatureEngineer ────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Builds a feature matrix from a chronological list of game dicts.
    Optionally enriches features with stored DraftKings odds snapshots.

    Usage:
        fe = FeatureEngineer(sport="NFL")
        df = fe.build(games, odds_lookup)   # odds_lookup is optional
    """

    def __init__(self, sport: str = "NFL") -> None:
        self.sport = sport.upper()
        self._elo_k  = _SPORT_ELO_K.get(self.sport, 20)
        self._elo_ha = _SPORT_ELO_HA.get(self.sport, 50)
        self._pythag = _PYTHAG_EXP.get(self.sport, 2.5)

    # ── Public entry point ─────────────────────────────────────────────────────

    def build(
        self,
        games: list[dict],
        odds_lookup: dict[str, dict] | None = None,
    ) -> pd.DataFrame:
        """
        Build the feature matrix.

        Args:
            games:       Chronologically sorted list of game dicts with keys:
                         date, home_team, away_team, home_score, away_score, home_winner
            odds_lookup: Optional dict keyed by game_id → {market: snapshot_dict}
                         (from DataCollector.get_odds_snapshot)

        Returns:
            DataFrame with columns = FEATURE_NAMES + ['home_win', 'date', 'home_team', 'away_team']
        """
        odds_lookup = odds_lookup or {}
        elo: dict[str, float] = defaultdict(lambda: 1500.0)
        team_hist: dict[str, list[dict]] = defaultdict(list)
        rows: list[dict] = []

        for g in games:
            if not self._is_complete(g):
                continue

            dt = self._parse_dt(g["date"])
            home, away = g["home_team"], g["away_team"]
            h_pts = float(g.get("home_score") or 0)
            a_pts = float(g.get("away_score") or 0)
            h_won = bool(g.get("home_winner", h_pts > a_pts))

            hh = team_hist[home]
            ah = team_hist[away]

            feat = self._compute_row(home, away, dt, hh, ah, elo)
            feat.update(self._odds_features(g, odds_lookup))
            feat["home_win"]  = 1 if h_won else 0
            feat["home_pts"]  = h_pts
            feat["away_pts"]  = a_pts
            feat["date"]      = dt.date().isoformat()
            feat["home_team"] = home
            feat["away_team"] = away
            feat["game_id"]   = g.get("game_id", "")
            feat["season"]    = g.get("season", dt.year)

            rows.append(feat)

            # Update state AFTER computing features (no leakage)
            e_h = 1.0 / (1 + 10 ** ((elo[away] - elo[home] - self._elo_ha) / 400))
            elo[home] += self._elo_k * ((1 if h_won else 0) - e_h)
            elo[away] += self._elo_k * ((0 if h_won else 1) - (1 - e_h))

            hh.append({"dt": dt, "won": h_won, "is_home": True,
                        "scored": h_pts, "allowed": a_pts, "opponent": away})
            ah.append({"dt": dt, "won": not h_won, "is_home": False,
                        "scored": a_pts, "allowed": h_pts, "opponent": home})

        if not rows:
            return pd.DataFrame(columns=FEATURE_NAMES + ["home_win", "date", "home_team", "away_team"])

        df = pd.DataFrame(rows)
        # Ensure all feature columns are present (fill missing odds cols with 0)
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def build_prediction_row(
        self,
        home: str,
        away: str,
        historical_games: list[dict],
        game_id: str = "",
        odds_snapshot: dict | None = None,
    ) -> dict[str, float]:
        """
        Compute a single feature row for an UPCOMING game (no outcome).
        historical_games must be sorted chronologically and include all prior games
        for both teams.  Used at prediction time.
        """
        elo: dict[str, float] = defaultdict(lambda: 1500.0)
        team_hist: dict[str, list[dict]] = defaultdict(list)

        for g in historical_games:
            if not self._is_complete(g):
                continue
            dt = self._parse_dt(g["date"])
            h, a = g["home_team"], g["away_team"]
            h_pts = float(g.get("home_score") or 0)
            a_pts = float(g.get("away_score") or 0)
            h_won = bool(g.get("home_winner", h_pts > a_pts))

            e_h = 1.0 / (1 + 10 ** ((elo[a] - elo[h] - self._elo_ha) / 400))
            elo[h] += self._elo_k * ((1 if h_won else 0) - e_h)
            elo[a] += self._elo_k * ((0 if h_won else 1) - (1 - e_h))

            team_hist[h].append({"dt": dt, "won": h_won, "is_home": True,
                                  "scored": h_pts, "allowed": a_pts, "opponent": a})
            team_hist[a].append({"dt": dt, "won": not h_won, "is_home": False,
                                  "scored": a_pts, "allowed": h_pts, "opponent": h})

        now = datetime.utcnow()
        feat = self._compute_row(home, away, now, team_hist[home], team_hist[away], elo)
        odds_lu = {game_id: odds_snapshot} if (game_id and odds_snapshot) else {}
        feat.update(self._odds_features({"game_id": game_id}, odds_lu))
        return feat

    # ── Core feature computation ───────────────────────────────────────────────

    def _compute_row(
        self,
        home: str,
        away: str,
        dt: datetime,
        hh: list[dict],
        ah: list[dict],
        elo: dict[str, float],
    ) -> dict[str, float]:
        h_elo = elo[home]
        a_elo = elo[away]

        feat: dict[str, float] = {
            "elo_diff":   h_elo + self._elo_ha - a_elo,
            "home_elo":   h_elo,
            "away_elo":   a_elo,
        }
        feat.update(self._rest_features(home, away, dt, hh, ah))
        feat.update(self._rolling_features(home, away, dt, hh, ah))
        feat.update(self._ha_split_features(home, away, dt, hh, ah))
        feat.update(self._h2h_features(home, away, dt, hh))
        feat.update(self._pythag_features(home, away, dt, hh, ah))
        return feat

    # ── Rest / scheduling ─────────────────────────────────────────────────────

    def _rest_features(
        self, home: str, away: str, dt: datetime,
        hh: list[dict], ah: list[dict],
    ) -> dict[str, float]:
        def rest(hist: list[dict]) -> int:
            prev = [x for x in hist if x["dt"] < dt]
            if not prev:
                return 3
            return min(int((dt - max(x["dt"] for x in prev)).total_seconds() / 86400), 14)

        h_r = rest(hh)
        a_r = rest(ah)
        return {
            "home_rest_days":  float(h_r),
            "away_rest_days":  float(a_r),
            "rest_advantage":  float(h_r - a_r),
            "home_b2b":        1.0 if h_r <= 1 else 0.0,
            "away_b2b":        1.0 if a_r <= 1 else 0.0,
        }

    # ── Rolling win% + PPG diff ───────────────────────────────────────────────

    def _rolling_features(
        self, home: str, away: str, dt: datetime,
        hh: list[dict], ah: list[dict],
    ) -> dict[str, float]:
        def _wp(hist: list[dict], n: int) -> float:
            past = [x for x in hist if x["dt"] < dt][-n:]
            return sum(x["won"] for x in past) / len(past) if past else 0.5

        def _ppg(hist: list[dict], n: int | None) -> float:
            past = [x for x in hist if x["dt"] < dt]
            if n:
                past = past[-n:]
            if not past:
                return 0.0
            return sum(x["scored"] - x["allowed"] for x in past) / len(past)

        h_l5  = _wp(hh, 5)
        h_l10 = _wp(hh, 10)
        a_l5  = _wp(ah, 5)
        a_l10 = _wp(ah, 10)

        return {
            "home_l5_wp":           h_l5,
            "home_l10_wp":          h_l10,
            "away_l5_wp":           a_l5,
            "away_l10_wp":          a_l10,
            "form_advantage_l5":    h_l5  - a_l5,
            "form_advantage_l10":   h_l10 - a_l10,
            "home_ppg_diff_l5":     _ppg(hh, 5),
            "home_ppg_diff_l10":    _ppg(hh, 10),
            "home_ppg_diff_season": _ppg(hh, None),
            "away_ppg_diff_l5":     _ppg(ah, 5),
            "away_ppg_diff_l10":    _ppg(ah, 10),
            "away_ppg_diff_season": _ppg(ah, None),
            "ppg_diff_advantage":   _ppg(hh, 10) - _ppg(ah, 10),
        }

    # ── Home / away splits ────────────────────────────────────────────────────

    def _ha_split_features(
        self, home: str, away: str, dt: datetime,
        hh: list[dict], ah: list[dict],
    ) -> dict[str, float]:
        def _ha_wp(hist: list[dict], is_home: bool, n: int = 30) -> float:
            past = [x for x in hist if x["dt"] < dt and x["is_home"] == is_home][-n:]
            return sum(x["won"] for x in past) / len(past) if past else 0.5

        return {
            "home_home_wp": _ha_wp(hh, True),
            "away_away_wp": _ha_wp(ah, False),
        }

    # ── H2H ──────────────────────────────────────────────────────────────────

    def _h2h_features(
        self, home: str, away: str, dt: datetime,
        hh: list[dict],
    ) -> dict[str, float]:
        meetings = [x for x in hh if x["dt"] < dt and x["opponent"] == away][-5:]
        wp = (sum(x["won"] for x in meetings) / len(meetings)) if meetings else 0.5
        return {"h2h_home_wp": wp}

    # ── Pythagorean expectation ───────────────────────────────────────────────

    def _pythag_features(
        self, home: str, away: str, dt: datetime,
        hh: list[dict], ah: list[dict],
    ) -> dict[str, float]:
        exp = self._pythag

        def pythag(hist: list[dict]) -> float:
            past = [x for x in hist if x["dt"] < dt][-20:]
            if not past:
                return 0.5
            pf = sum(x["scored"]  for x in past) / len(past)
            pa = sum(x["allowed"] for x in past) / len(past)
            if pf == 0 and pa == 0:
                return 0.5
            denom = pf ** exp + pa ** exp
            return (pf ** exp / denom) if denom > 0 else 0.5

        h_p = pythag(hh)
        a_p = pythag(ah)
        return {
            "home_pythag":    h_p,
            "away_pythag":    a_p,
            "pythag_advantage": h_p - a_p,
        }

    # ── DraftKings odds features ──────────────────────────────────────────────

    def _odds_features(self, game: dict, odds_lookup: dict) -> dict[str, float]:
        """
        Extract DK odds-based features from stored snapshots.
        Returns zeros when no snapshot is available (model degrades gracefully).
        """
        base: dict[str, float] = {
            "dk_home_implied":    0.0,
            "dk_away_implied":    0.0,
            "dk_home_spread":     0.0,
            "dk_total":           0.0,
            "dk_over_implied":    0.0,
            "dk_vig_h2h":         0.0,
            "dk_line_move_ml":    0.0,
            "dk_line_move_spread": 0.0,
            "dk_sharp_signal":    0.0,
        }

        gid = game.get("game_id", "")
        if not gid or gid not in odds_lookup:
            return base

        snap = odds_lookup[gid]

        # Moneyline
        h2h = snap.get("h2h", {})
        h_price = h2h.get("home_price")
        a_price = h2h.get("away_price")
        if h_price and a_price:
            raw_h = _impl_prob(h_price)
            raw_a = _impl_prob(a_price)
            total = raw_h + raw_a
            base["dk_home_implied"] = raw_h / total
            base["dk_away_implied"] = raw_a / total
            base["dk_vig_h2h"]      = (total - 1.0) * 100

        # Spread
        spr = snap.get("spreads", {})
        if spr.get("home_point") is not None:
            base["dk_home_spread"] = float(spr["home_point"])

        # Totals
        tot = snap.get("totals", {})
        if tot.get("over_under") is not None:
            base["dk_total"] = float(tot["over_under"])
            o_price = tot.get("over_price")
            u_price = tot.get("under_price")
            if o_price and u_price:
                raw_o = _impl_prob(o_price)
                raw_u = _impl_prob(u_price)
                base["dk_over_implied"] = raw_o / (raw_o + raw_u)

        # Line movement (open → current)
        h2h_open = snap.get("h2h_open", {})
        if h2h_open.get("home_price") and h_price:
            open_impl = _impl_prob(h2h_open["home_price"])
            curr_impl = _impl_prob(h_price)
            base["dk_line_move_ml"] = curr_impl - open_impl

        spr_open = snap.get("spreads_open", {})
        if spr_open.get("home_point") is not None and spr.get("home_point") is not None:
            base["dk_line_move_spread"] = spr["home_point"] - spr_open["home_point"]

        # Sharp signal flag
        base["dk_sharp_signal"] = float(snap.get("sharp_signal", False))

        return base

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_complete(g: dict) -> bool:
        return (
            g.get("home_score") is not None
            and g.get("away_score") is not None
        )

    @staticmethod
    def _parse_dt(date_str: str) -> datetime:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str[:19].rstrip("Z"), fmt[:len(date_str[:19])])
            except ValueError:
                continue
        return datetime.utcnow()

    # ── Convenience: target construction ─────────────────────────────────────

    @staticmethod
    def make_spread_target(df: pd.DataFrame) -> pd.Series:
        """
        Binary target: 1 if home team covered the DK spread, 0 otherwise.
        Requires 'home_pts', 'away_pts', 'dk_home_spread' columns.
        Rows where spread = 0 (no odds data) are NaN.
        """
        mask = df["dk_home_spread"] != 0
        margin = df["home_pts"] - df["away_pts"]
        covered = (margin + df["dk_home_spread"]) > 0
        result = covered.astype(float)
        result[~mask] = float("nan")
        return result

    @staticmethod
    def make_totals_target(df: pd.DataFrame) -> pd.Series:
        """
        Binary target: 1 if Over hit, 0 if Under.
        Rows where total = 0 (no odds data) are NaN.
        """
        mask = df["dk_total"] != 0
        total_pts = df["home_pts"] + df["away_pts"]
        over_hit = (total_pts > df["dk_total"]).astype(float)
        over_hit[~mask] = float("nan")
        return over_hit


def _impl_prob(american: float) -> float:
    """Raw implied probability (vig included)."""
    if american >= 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)
