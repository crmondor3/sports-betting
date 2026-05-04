"""
Multi-signal matchup model.
Combines Elo, ESPN form/H2H/rest into a single probability estimate
and a 0-100 confidence score. Fewer, higher-conviction bets only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

from data.espn_api import MatchupData, TeamForm

logger = logging.getLogger(__name__)

# ── Signal weights ─────────────────────────────────────────────────────────────
# How much each signal can move the probability away from 50%
W_ELO         = 0.45   # Elo differential
W_FORM        = 0.20   # L10 win rate differential
W_HOME_AWAY   = 0.15   # actual home vs away records
W_H2H         = 0.10   # head-to-head record
W_REST        = 0.10   # rest days advantage


@dataclass
class Signal:
    name: str
    label: str       # human-readable value, e.g. "BOS 7-3 vs NYK 4-6"
    direction: int   # +1 favours home, -1 favours away, 0 neutral
    magnitude: float # 0.0–1.0 strength of signal


@dataclass
class MatchupPrediction:
    home_team: str
    away_team: str
    sport: str
    # Win probabilities
    home_win_prob: float
    away_win_prob: float
    # Total points
    predicted_total: float
    over_prob: float
    under_prob: float
    # Confidence (0-100)
    confidence: int
    # Individual signals
    signals: list[Signal] = field(default_factory=list)
    # Raw Elo
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    data_quality: str = "cold"

    @property
    def favourite(self) -> str:
        return self.home_team if self.home_win_prob >= 0.5 else self.away_team

    @property
    def underdog(self) -> str:
        return self.away_team if self.home_win_prob >= 0.5 else self.home_team

    @property
    def favourite_prob(self) -> float:
        return max(self.home_win_prob, self.away_win_prob)

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 75: return "High"
        if self.confidence >= 55: return "Medium"
        return "Low"

    @property
    def signal_summary(self) -> list[str]:
        lines = []
        for s in self.signals:
            icon = "+" if s.direction > 0 else ("-" if s.direction < 0 else "~")
            lines.append(f"{icon} {s.name}: {s.label}")
        return lines


class MatchupAnalyzer:
    """
    Takes an EloModel (pre-fitted) + ESPN matchup data and produces
    a MatchupPrediction with confidence score and signal breakdown.
    """

    def __init__(self, elo_model=None):
        self._elo = elo_model

    def analyse(self, matchup: MatchupData) -> MatchupPrediction:
        home = matchup.home_team
        away = matchup.away_team
        sport = matchup.sport
        hf = matchup.home_form
        af = matchup.away_form
        h2h = matchup.h2h

        signals: list[Signal] = []

        # ── 1. Elo signal ──────────────────────────────────────────────────────
        if self._elo:
            pred = self._elo.predict(home, away)
            elo_prob_home = pred.home_win_prob
            home_elo = pred.home_elo
            away_elo = pred.away_elo
        else:
            elo_prob_home = 0.5
            home_elo = away_elo = 1500.0

        elo_diff = home_elo - away_elo
        elo_dir = 1 if elo_diff > 0 else (-1 if elo_diff < 0 else 0)
        elo_label = f"{home} {home_elo:.0f} vs {away} {away_elo:.0f}"
        signals.append(Signal("Elo rating", elo_label, elo_dir, min(abs(elo_diff) / 400, 1.0)))

        # ── 2. Recent form signal ──────────────────────────────────────────────
        if (hf.wins + hf.losses) > 0 and (af.wins + af.losses) > 0:
            home_l10 = hf.last_10_wins / max(hf.last_10_wins + hf.last_10_losses, 1)
            away_l10 = af.last_10_wins / max(af.last_10_wins + af.last_10_losses, 1)
            form_diff = home_l10 - away_l10
            form_dir = 1 if form_diff > 0 else (-1 if form_diff < 0 else 0)
            signals.append(Signal(
                "Recent form",
                f"{home} {hf.l10_label} vs {away} {af.l10_label}",
                form_dir,
                abs(form_diff),
            ))
        else:
            form_diff = 0.0

        # ── 3. Home/away split ─────────────────────────────────────────────────
        if (hf.home_wins + hf.home_losses) > 5 and (af.away_wins + af.away_losses) > 5:
            h_home_pct = hf.home_win_pct
            a_away_pct = af.away_win_pct
            split_diff = h_home_pct - a_away_pct
            split_dir = 1 if split_diff > 0.05 else (-1 if split_diff < -0.05 else 0)
            signals.append(Signal(
                "Home/Away record",
                f"{home} {hf.home_wins}-{hf.home_losses} home | {away} {af.away_wins}-{af.away_losses} away",
                split_dir,
                abs(split_diff),
            ))
        else:
            split_diff = 0.0
            h_home_pct = 0.5
            a_away_pct = 0.5

        # ── 4. H2H signal ─────────────────────────────────────────────────────
        if h2h.meetings >= 3:
            h2h_diff = h2h.home_h2h_win_pct - 0.5
            h2h_dir = 1 if h2h_diff > 0 else (-1 if h2h_diff < 0 else 0)
            signals.append(Signal(
                "Head-to-head",
                h2h.label,
                h2h_dir,
                abs(h2h_diff) * 2,  # scale to 0-1
            ))
        else:
            h2h_diff = 0.0

        # ── 5. Rest advantage ─────────────────────────────────────────────────
        rest_diff = hf.rest_days - af.rest_days
        if abs(rest_diff) >= 1:
            rest_dir = 1 if rest_diff > 0 else -1
            rest_label = (
                f"{home} +{rest_diff}d rest advantage"
                if rest_diff > 0
                else f"{away} +{abs(rest_diff)}d rest advantage"
            )
            signals.append(Signal("Rest days", rest_label, rest_dir, min(abs(rest_diff) / 3, 1.0)))
        else:
            rest_diff = 0.0

        # ── 6. Streak signal ──────────────────────────────────────────────────
        if hf.streak.startswith("W") and int(hf.streak[1:] or 1) >= 3:
            n = int(hf.streak[1:] or 1)
            signals.append(Signal("Hot streak", f"{home} on {n}-game win streak", 1, min(n / 7, 1.0)))
        elif af.streak.startswith("W") and int(af.streak[1:] or 1) >= 3:
            n = int(af.streak[1:] or 1)
            signals.append(Signal("Hot streak", f"{away} on {n}-game win streak", -1, min(n / 7, 1.0)))

        # ── Blend probabilities ────────────────────────────────────────────────
        # Start from Elo, then nudge with each signal
        p_home = elo_prob_home

        if (hf.wins + hf.losses) > 0:
            form_contrib = (form_diff * W_FORM)
            p_home = _clamp(p_home + form_contrib)

        if (hf.home_wins + hf.home_losses) > 5:
            split_contrib = (split_diff * W_HOME_AWAY)
            p_home = _clamp(p_home + split_contrib)

        if h2h.meetings >= 3:
            h2h_contrib = (h2h_diff * W_H2H)
            p_home = _clamp(p_home + h2h_contrib)

        if abs(rest_diff) >= 1:
            rest_contrib = (min(rest_diff / 3, 1.0) * 0.03 * (1 if rest_diff > 0 else -1))
            p_home = _clamp(p_home + rest_contrib)

        # ── Confidence score ──────────────────────────────────────────────────
        # Base: 40. Add up to 60 based on signal strength + agreement.
        favourite_dir = 1 if p_home >= 0.5 else -1
        agreeing = [s for s in signals if s.direction == favourite_dir]
        disagreeing = [s for s in signals if s.direction == -favourite_dir]
        agreement_score = sum(s.magnitude for s in agreeing) - sum(s.magnitude * 0.5 for s in disagreeing)
        data_bonus = 15 if matchup.data_quality == "full" else 5
        confidence = int(_clamp(0.4 + agreement_score * 0.35) * 100) + data_bonus
        confidence = max(0, min(100, confidence))

        # ── Totals prediction ─────────────────────────────────────────────────
        home_ppg = hf.ppg if hf.ppg > 0 else _league_avg_ppg(sport)
        away_ppg = af.ppg if af.ppg > 0 else _league_avg_ppg(sport)
        home_papg = hf.papg if hf.papg > 0 else _league_avg_ppg(sport)
        away_papg = af.papg if af.papg > 0 else _league_avg_ppg(sport)

        # Predicted total = average of offensive and defensive perspectives
        pred_total = round(((home_ppg + away_papg) / 2 + (away_ppg + home_papg) / 2), 1)
        # H2H scoring average override if enough samples
        if h2h.meetings >= 4:
            h2h_total = h2h.home_ppg + h2h.away_ppg
            pred_total = round((pred_total * 0.6 + h2h_total * 0.4), 1)

        return MatchupPrediction(
            home_team=home,
            away_team=away,
            sport=sport,
            home_win_prob=round(p_home, 4),
            away_win_prob=round(1 - p_home, 4),
            predicted_total=pred_total,
            over_prob=0.5,    # placeholder — Poisson model fills this
            under_prob=0.5,
            confidence=confidence,
            signals=signals,
            home_elo=home_elo,
            away_elo=away_elo,
            data_quality=matchup.data_quality,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clamp(p: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return max(lo, min(hi, p))


def _league_avg_ppg(sport: str) -> float:
    return {"NBA": 113.0, "MLB": 4.5, "NHL": 3.0, "NFL": 23.0}.get(sport.upper(), 100.0)
