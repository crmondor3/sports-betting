"""Poisson regression model for game totals (over/unders)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class PoissonPrediction:
    home_team: str
    away_team: str
    home_goals_mean: float
    away_goals_mean: float
    over_prob: float
    under_prob: float
    push_prob: float
    predicted_total: float

    def to_dict(self) -> dict:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "predicted_total": round(self.predicted_total, 2),
            "over_prob": round(self.over_prob, 4),
            "under_prob": round(self.under_prob, 4),
            "push_prob": round(self.push_prob, 4),
        }


class PoissonModel:
    """
    Dixon-Coles style Poisson regression.
    Estimates attack/defence parameters per team via MLE.
    """

    def __init__(self, home_advantage: float = 0.1):
        self.home_advantage = home_advantage
        self._attack: dict[str, float] = {}
        self._defence: dict[str, float] = {}
        self._home_adv: float = home_advantage
        self._fitted = False

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "PoissonModel":
        """
        Fit on a DataFrame with columns:
          home_team, away_team, home_score, away_score
        """
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        if len(teams) < 2:
            raise ValueError("Need at least 2 teams to fit.")

        # index → team mapping
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        def neg_log_likelihood(params: np.ndarray) -> float:
            attack = params[:n]
            defence = params[n:2 * n]
            home_adv = params[2 * n]
            ll = 0.0
            for _, row in df.iterrows():
                hi = idx[row["home_team"]]
                ai = idx[row["away_team"]]
                lam_h = np.exp(attack[hi] - defence[ai] + home_adv)
                lam_a = np.exp(attack[ai] - defence[hi])
                ll += poisson.logpmf(int(row["home_score"]), lam_h)
                ll += poisson.logpmf(int(row["away_score"]), lam_a)
            return -ll

        x0 = np.zeros(2 * n + 1)
        x0[2 * n] = self.home_advantage
        # Constraint: mean attack = 0
        constraints = [{"type": "eq", "fun": lambda p: np.mean(p[:n])}]

        result = minimize(neg_log_likelihood, x0, method="SLSQP", constraints=constraints,
                          options={"maxiter": 500, "ftol": 1e-9})
        if not result.success:
            logger.warning("Poisson optimisation: %s", result.message)

        params = result.x
        for team in teams:
            i = idx[team]
            self._attack[team] = params[i]
            self._defence[team] = params[n + i]
        self._home_adv = params[2 * n]
        self._fitted = True
        logger.info("PoissonModel fitted on %d games / %d teams", len(df), n)
        return self

    def fit_from_averages(self, team_stats: dict[str, dict]) -> "PoissonModel":
        """
        Lightweight fit from pre-computed scoring averages when full game logs
        are unavailable.  team_stats: {team: {"pts_per_game": X, ...}}
        """
        if not team_stats:
            raise ValueError("team_stats is empty — check your stats API.")
        avg = np.mean([v.get("pts_per_game", 100) for v in team_stats.values()])
        for team, stats in team_stats.items():
            ppg = stats.get("pts_per_game", avg)
            self._attack[team] = np.log(max(ppg / avg, 0.1))
            self._defence[team] = 0.0
        self._home_adv = self.home_advantage
        self._fitted = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def _lambda(self, attack: float, defence: float, is_home: bool) -> float:
        base = np.exp(attack - defence)
        return base * np.exp(self._home_adv) if is_home else base

    def predict(
        self,
        home_team: str,
        away_team: str,
        line: float,
        max_goals: int = 20,
    ) -> PoissonPrediction:
        """
        Predict over/under probabilities for a total line.
        max_goals: upper bound for probability summation.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_from_averages() first.")

        # Fall back to league-average attack/defence if team unseen
        mean_atk = np.mean(list(self._attack.values())) if self._attack else 0.0
        mean_def = np.mean(list(self._defence.values())) if self._defence else 0.0

        atk_h = self._attack.get(home_team, mean_atk)
        def_h = self._defence.get(home_team, mean_def)
        atk_a = self._attack.get(away_team, mean_atk)
        def_a = self._defence.get(away_team, mean_def)

        lam_h = self._lambda(atk_h, def_a, is_home=True)
        lam_a = self._lambda(atk_a, def_h, is_home=False)

        # Joint probability table
        goals = np.arange(0, max_goals + 1)
        p_home = poisson.pmf(goals, lam_h)
        p_away = poisson.pmf(goals, lam_a)
        joint = np.outer(p_home, p_away)  # [home_goals, away_goals]

        totals = goals[:, None] + goals[None, :]  # shape (n, n)
        over_prob = float(joint[totals > line].sum())
        under_prob = float(joint[totals < line].sum())
        push_prob = float(joint[totals == line].sum())

        return PoissonPrediction(
            home_team=home_team,
            away_team=away_team,
            home_goals_mean=round(lam_h, 2),
            away_goals_mean=round(lam_a, 2),
            over_prob=over_prob,
            under_prob=under_prob,
            push_prob=push_prob,
            predicted_total=round(lam_h + lam_a, 2),
        )
