"""Elo + logistic regression hybrid model for moneyline win probability."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 50.0  # Elo points added to home team

# Sport-specific tuning (empirically motivated):
#   NFL: ~3.5 points ≈ 57% home win rate, fewer games → high K
#   NBA: ~3 pts ≈ 59% home, rest/travel matters → moderate K+adv
#   MLB: ~54% home, extreme game variance → low K
#   NHL: ~55% home, puck luck → moderate K
_SPORT_K:        dict[str, float] = {"NFL": 28, "NBA": 20, "MLB": 10, "NHL": 16}
_SPORT_HOME_ADV: dict[str, float] = {"NFL": 70, "NBA": 35, "MLB": 18, "NHL": 28}


@dataclass
class MoneylinePrediction:
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    home_elo: float
    away_elo: float

    @property
    def favourite(self) -> str:
        return self.home_team if self.home_win_prob >= 0.5 else self.away_team

    @property
    def favourite_prob(self) -> float:
        return max(self.home_win_prob, self.away_win_prob)

    def to_dict(self) -> dict:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win_prob": round(self.home_win_prob, 4),
            "away_win_prob": round(self.away_win_prob, 4),
            "home_elo": round(self.home_elo, 1),
            "away_elo": round(self.away_elo, 1),
        }


class EloModel:
    """
    Pure Elo for moneyline prediction with optional logistic regression layer
    that can incorporate additional features (rest days, travel, injuries).
    """

    def __init__(
        self,
        k: float = K_FACTOR,
        home_adv: float = HOME_ADVANTAGE,
        sport: str | None = None,
    ):
        # Sport-specific overrides take priority over generic defaults
        self.k        = _SPORT_K.get(sport.upper(), k) if sport else k
        self.home_adv = _SPORT_HOME_ADV.get(sport.upper(), home_adv) if sport else home_adv
        self._ratings: dict[str, float] = {}
        self._lr: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._use_lr = False

    # ── Elo core ───────────────────────────────────────────────────────────────

    def _rating(self, team: str) -> float:
        return self._ratings.setdefault(team, DEFAULT_ELO)

    @staticmethod
    def expected_score(elo_a: float, elo_b: float) -> float:
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def update(self, home: str, away: str, home_won: bool) -> None:
        r_h = self._rating(home)
        r_a = self._rating(away)
        e_h = self.expected_score(r_h + self.home_adv, r_a)
        s_h = 1.0 if home_won else 0.0
        self._ratings[home] = r_h + self.k * (s_h - e_h)
        self._ratings[away] = r_a + self.k * ((1 - s_h) - (1 - e_h))

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, use_logistic: bool = False) -> "EloModel":
        """
        Fit on DataFrame with columns: home_team, away_team, home_won (bool).
        Optionally trains a logistic layer on top of Elo diff + extra features.
        """
        # Reset ratings
        self._ratings = {}

        # Chronological pass to build Elo ratings
        elo_diffs, targets = [], []
        for _, row in df.iterrows():
            home, away = row["home_team"], row["away_team"]
            r_h = self._rating(home)
            r_a = self._rating(away)
            elo_diffs.append(r_h + self.home_adv - r_a)
            targets.append(int(bool(row["home_won"])))
            self.update(home, away, bool(row["home_won"]))

        if use_logistic and len(df) >= 50:
            X = np.array(elo_diffs).reshape(-1, 1)
            y = np.array(targets)
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            self._lr = LogisticRegression(C=1.0, max_iter=500)
            self._lr.fit(X_scaled, y)
            self._use_lr = True
            logger.info("EloModel fitted logistic layer on %d games", len(df))
        else:
            logger.info("EloModel fitted (Elo-only) on %d games", len(df))

        return self

    def fit_from_standings(self, standings: list[dict], win_key: str = "Wins",
                            loss_key: str = "Losses", name_key: str = "Team") -> "EloModel":
        """Bootstrap Elo from win/loss records when full game log unavailable."""
        self._ratings = {}
        for row in standings:
            name = row.get(name_key, "")
            wins = int(row.get(win_key, 0))
            losses = int(row.get(loss_key, 0))
            total = wins + losses
            if total == 0:
                continue
            win_rate = wins / total
            # Map win rate to Elo: 50% → 1500, linear stretch
            self._ratings[name] = 1500 + 400 * math.log10(max(win_rate, 0.01) / max(1 - win_rate, 0.01))
        logger.info("Bootstrapped Elo for %d teams from standings", len(self._ratings))
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, home_team: str, away_team: str) -> MoneylinePrediction:
        r_h = self._rating(home_team)
        r_a = self._rating(away_team)

        if self._use_lr and self._lr and self._scaler:
            elo_diff = np.array([[r_h + self.home_adv - r_a]])
            X_scaled = self._scaler.transform(elo_diff)
            home_prob = float(self._lr.predict_proba(X_scaled)[0, 1])
        else:
            home_prob = self.expected_score(r_h + self.home_adv, r_a)

        return MoneylinePrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=round(home_prob, 4),
            away_win_prob=round(1 - home_prob, 4),
            home_elo=r_h,
            away_elo=r_a,
        )

    def get_ratings(self) -> dict[str, float]:
        return dict(sorted(self._ratings.items(), key=lambda x: -x[1]))
