"""
ModelCalibrator — continuously learns from settled bet outcomes.

Theory:
  Under Kelly criterion, bankroll growth rate = sum(p * log(1 + b*f) + q * log(1-f)).
  This is maximised only when model probabilities are perfectly calibrated.
  A 5% systematic overestimation of win probability causes the same Kelly fraction
  to erode bankroll instead of growing it. This calibrator corrects that.

Approach:
  1. Multiplicative correction per (sport, bet_type) — adjusts raw model_prob
     toward the historically-observed actual win rate.
  2. Brier Skill Score (BSS) vs the book's implied probability — measures whether
     our model adds any real information beyond what the market already prices in.
     BSS drives the Kelly confidence multiplier: we size down when the model is
     adding little value above the book.
  3. Profitable segment analysis — tracks ROI, hit rate, and avg EV by sport and
     bet type so we know exactly which areas are generating real edge.
  4. Persistence — calibration is saved to cache/calibration.json so learning
     compounds across sessions rather than resetting each day.

Calibration blending:
  raw_factor  = actual_win_rate / model_expected_rate
  blend_weight = min((n - MIN_SAMPLE) / (FULL_TRUST - MIN_SAMPLE), 1) * MAX_WEIGHT
  final_factor = 1.0 + (raw_factor - 1.0) * blend_weight
  → No adjustment at small n; full correction at FULL_TRUST samples.
  → Capped at [MIN_FACTOR, MAX_FACTOR] to prevent extreme overcorrection.

Brier Skill Score:
  BSS = 1 - (BS_model / BS_baseline)
  BS_model    = mean((model_prob - outcome)^2) across settled bets
  BS_baseline = mean((implied_prob - outcome)^2) across settled bets
  BSS = 0.0  → model adds nothing over book price
  BSS = 1.0  → perfect model
  Kelly_mult  = clip(0.5 + BSS * 0.5, 0.40, 1.0)
  → At zero skill, use 40% of standard Kelly (very conservative).
  → At full skill, use 100% of standard Kelly.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).parent.parent / "cache" / "calibration.json"
_MIN_SAMPLE  = 5     # bets needed before applying any per-group correction
_FULL_TRUST  = 40    # samples for full calibration weight
_MAX_WEIGHT  = 0.85  # maximum calibration blend weight (retain 15% prior)
_MAX_FACTOR  = 1.40  # cap upward correction
_MIN_FACTOR  = 0.65  # cap downward correction


def _blend_factor(raw: float, n: int) -> float:
    """Blend raw correction factor toward 1.0 for small samples."""
    w = min(max((n - _MIN_SAMPLE) / (_FULL_TRUST - _MIN_SAMPLE), 0.0), 1.0) * _MAX_WEIGHT
    blended = 1.0 + (raw - 1.0) * w
    return max(_MIN_FACTOR, min(_MAX_FACTOR, blended))


class ModelCalibrator:
    """
    Continuously updated probability calibration from settled model bets.
    All public methods are safe to call even with zero historical bets
    (they return identity / neutral values in that case).
    """

    def __init__(self):
        self._sport_factor: dict[str, float] = {}
        self._type_factor:  dict[str, float] = {}
        self._kelly_mult:   dict[str, float] = {}  # sport → Kelly confidence
        self._segment_roi:  dict[tuple, float] = {}  # (sport, bet_type) → ROI %
        self._segment_n:    dict[tuple, int]   = {}  # (sport, bet_type) → sample count
        self.n_settled:     int              = 0
        self.brier_score:   float | None     = None
        self.log_loss:      float | None     = None
        self.segment_report: list[dict]      = []

    # ─── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, bets: list[Any]) -> "ModelCalibrator":
        """
        Fit calibration from a list of settled Bet ORM objects.
        Expects each bet to have: model_prob, implied_prob, sport, bet_type, won,
        ev_pct, stake_kelly, pnl_kelly.
        """
        if not bets:
            return self
        self.n_settled = len(bets)

        self._sport_factor = self._fit_groups(bets, lambda b: b.sport, "sport")
        self._type_factor  = self._fit_groups(bets, lambda b: b.bet_type, "bet_type")
        self._kelly_mult   = self._compute_kelly_multipliers(bets)
        self.brier_score   = self._brier(bets)
        self.log_loss      = self._log_loss_score(bets)
        self._build_segment_report(bets)

        logger.info(
            "Calibrator fitted: n=%d Brier=%.4f LogLoss=%.4f",
            self.n_settled, self.brier_score or 0.0, self.log_loss or 0.0,
        )
        return self

    def _fit_groups(self, bets, key_fn, label: str) -> dict[str, float]:
        groups: dict[str, list] = defaultdict(list)
        for b in bets:
            groups[key_fn(b)].append(b)

        factors: dict[str, float] = {}
        for k, grp in groups.items():
            n = len(grp)
            if n < _MIN_SAMPLE:
                factors[k] = 1.0
                continue
            actual   = sum(1 for b in grp if b.won) / n
            expected = sum(b.model_prob or 0.5 for b in grp) / n
            if expected <= 0.0:
                factors[k] = 1.0
                continue
            raw = actual / expected
            factors[k] = _blend_factor(raw, n)
            logger.info(
                "Cal[%s=%s] n=%d actual=%.1f%% expected=%.1f%% → factor=%.3f",
                label, k, n, actual * 100, expected * 100, factors[k],
            )
        return factors

    def _compute_kelly_multipliers(self, bets) -> dict[str, float]:
        """
        Brier Skill Score vs book's implied prob per sport.
        Drives Kelly confidence: a model with no skill vs the market sizes down.
        """
        mults: dict[str, float] = {}
        groups: dict[str, list] = defaultdict(list)
        for b in bets:
            groups[b.sport].append(b)

        for sport, grp in groups.items():
            if len(grp) < _MIN_SAMPLE:
                mults[sport] = 1.0
                continue
            bs_model    = sum((b.model_prob - (1 if b.won else 0)) ** 2 for b in grp) / len(grp)
            bs_baseline = sum((b.implied_prob - (1 if b.won else 0)) ** 2 for b in grp) / len(grp)
            bss  = max(0.0, 1.0 - bs_model / bs_baseline) if bs_baseline > 0 else 0.0
            mult = max(0.40, min(1.0, 0.50 + bss * 0.50))
            mults[sport] = mult
            logger.info(
                "Kelly mult [%s]: BSS=%.3f → %.2f× Kelly", sport, bss, mult
            )
        return mults

    @staticmethod
    def _brier(bets) -> float | None:
        valid = [(b.model_prob, 1 if b.won else 0) for b in bets if b.model_prob]
        if not valid:
            return None
        return sum((p - y) ** 2 for p, y in valid) / len(valid)

    @staticmethod
    def _log_loss_score(bets) -> float | None:
        eps   = 1e-9
        valid = [(b.model_prob, 1 if b.won else 0) for b in bets if b.model_prob]
        if not valid:
            return None
        return -sum(
            y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
            for p, y in valid
        ) / len(valid)

    def _build_segment_report(self, bets) -> None:
        groups: dict[tuple, list] = defaultdict(list)
        for b in bets:
            groups[(b.sport, b.bet_type)].append(b)

        self._segment_roi = {}
        self._segment_n   = {}
        rows = []
        for (sport, bt), grp in sorted(groups.items()):
            n      = len(grp)
            wins   = sum(1 for b in grp if b.won)
            pnl    = sum(b.pnl_kelly or 0 for b in grp)
            staked = sum(b.stake_kelly or 0 for b in grp)
            act    = wins / n if n else 0.0
            exp    = sum(b.model_prob or 0.5 for b in grp) / n if n else 0.0
            avg_ev = sum(b.ev_pct or 0 for b in grp) / n if n else 0.0
            roi    = pnl / staked * 100 if staked > 0 else 0.0
            self._segment_roi[(sport, bt)] = roi
            self._segment_n[(sport, bt)]   = n
            factor = self.get_combined_factor(sport, bt)
            km     = self._kelly_mult.get(sport, 1.0)
            rows.append({
                "Sport":      sport,
                "Bet Type":   bt.replace("total_", "").replace("_", " ").title(),
                "Bets":       n,
                "W-L":        f"{wins}-{n - wins}",
                "Hit %":      f"{act:.1%}",
                "Model %":    f"{exp:.1%}",
                "Avg EV":     f"{avg_ev:+.1f}%",
                "ROI":        f"{roi:+.1f}%",
                "P&L":        f"${pnl:+.2f}",
                "Cal Factor": f"{factor:.3f}",
                "Kelly ×":    f"{km:.2f}",
                "Status":     (
                    "✅ profitable" if roi > 2 and n >= 5
                    else "🔍 small sample" if n < _MIN_SAMPLE
                    else "⚠️ break-even" if abs(roi) <= 2
                    else "❌ losing"
                ),
            })
        self.segment_report = rows

    # ─── Inference ────────────────────────────────────────────────────────────

    def get_combined_factor(self, sport: str, bet_type: str) -> float:
        """Geometric mean of sport and bet_type correction factors."""
        sf = self._sport_factor.get(sport, 1.0)
        tf = self._type_factor.get(bet_type, 1.0)
        return max(_MIN_FACTOR, min(_MAX_FACTOR, (sf * tf) ** 0.5))

    def calibrate_prob(self, prob: float, sport: str, bet_type: str) -> float:
        """Apply calibration and clip to [0.02, 0.98]."""
        if self.n_settled < _MIN_SAMPLE:
            return prob
        return max(0.02, min(0.98, prob * self.get_combined_factor(sport, bet_type)))

    def kelly_multiplier(self, sport: str) -> float:
        """Kelly confidence for a sport (0.40–1.0). 1.0 = full trust."""
        return self._kelly_mult.get(sport, 1.0)

    def is_profitable_segment(self, sport: str, bet_type: str) -> bool:
        """
        True if this (sport, bet_type) segment should be bet.
        Small-sample segments get the benefit of the doubt (True).
        Confirmed losing segments (n >= _MIN_SAMPLE and ROI < -5%) return False.
        """
        key = (sport, bet_type)
        n = self._segment_n.get(key, 0)
        if n < _MIN_SAMPLE:
            return True  # insufficient data — don't filter
        roi = self._segment_roi.get(key, 0.0)
        return roi > -5.0  # allow break-even; kill only confirmed losers

    def model_quality_label(self) -> str:
        if self.n_settled < _MIN_SAMPLE:
            return "Insufficient data"
        bs = self.brier_score or 0.5
        if bs < 0.18: return "Excellent calibration"
        if bs < 0.22: return "Good calibration"
        if bs < 0.25: return "Fair calibration"
        return "Needs improvement"

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        try:
            _CACHE_PATH.parent.mkdir(exist_ok=True)
            _CACHE_PATH.write_text(
                json.dumps({
                    "n_settled":    self.n_settled,
                    "brier_score":  self.brier_score,
                    "log_loss":     self.log_loss,
                    "sport_factor": self._sport_factor,
                    "type_factor":  self._type_factor,
                    "kelly_mult":   self._kelly_mult,
                }, default=str),
                encoding="utf-8",
            )
            logger.info("Calibration saved (%d settled bets)", self.n_settled)
        except Exception as exc:
            logger.warning("Calibration save failed: %s", exc)

    @classmethod
    def load(cls) -> "ModelCalibrator":
        """Load persisted calibration or return blank instance."""
        c = cls()
        if not _CACHE_PATH.exists():
            return c
        try:
            d = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            c.n_settled      = d.get("n_settled", 0)
            c.brier_score    = d.get("brier_score")
            c.log_loss       = d.get("log_loss")
            c._sport_factor  = d.get("sport_factor", {})
            c._type_factor   = d.get("type_factor", {})
            c._kelly_mult    = d.get("kelly_mult", {})
            logger.info("Calibration loaded (%d settled bets)", c.n_settled)
        except Exception as exc:
            logger.warning("Calibration load failed: %s", exc)
        return c

    @classmethod
    def rebuild_from_db(cls) -> "ModelCalibrator":
        """Refit from all settled bets and return. No auto_tracked filter —
        since manual logging is removed, every bet in the DB is a model pick."""
        from tracker.database import session_scope
        from tracker.models import Bet
        with session_scope() as s:
            bets = (
                s.query(Bet)
                .filter(Bet.settled == True)
                .all()
            )
        c = cls()
        if bets:
            c.fit(bets)
            c.save()
        return c
