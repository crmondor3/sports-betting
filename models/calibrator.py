"""
ModelCalibrator — continuously learns from settled bet outcomes.

Theory:
  Under Kelly criterion, bankroll growth rate = sum(p * log(1 + b*f) + q * log(1-f)).
  This is maximised only when model probabilities are perfectly calibrated.
  A 5% systematic overestimation causes Kelly to erode bankroll instead of growing it.

Calibration approach (two-stage):
  Stage 1 — IsotonicRegression (non-parametric, monotone):
    Fits a non-linear calibration curve directly from (model_prob → actual_win_rate).
    Captures patterns like "model is overconfident between 0.60-0.70 but accurate below 0.60".
    Requires 20+ samples globally, 15+ per sport.

  Stage 2 — Multiplicative fallback (small samples):
    raw_factor  = actual_win_rate / model_expected_rate
    blend_weight = min((n - MIN_SAMPLE) / (FULL_TRUST - MIN_SAMPLE), 1) * MAX_WEIGHT
    final_factor = 1.0 + (raw_factor - 1.0) * blend_weight
    No adjustment at small n; full correction at FULL_TRUST samples.
    Capped at [MIN_FACTOR, MAX_FACTOR] to prevent extreme overcorrection.

Kelly multiplier (sport-level Brier Skill Score):
  BSS = 1 - (BS_model / BS_baseline)
  Kelly_mult = clip(0.5 + BSS * 0.5, 0.40, 1.0)
  At zero skill vs the market → 40% of standard Kelly (conservative).
  At full skill → 100%.

Persistence:
  - Simple stats saved to cache/calibration.json (human-readable)
  - IsotonicRegression objects saved to cache/calibration_ir.pkl
  Learning compounds across sessions; both files restored on cold starts.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_PATH  = Path(__file__).parent.parent / "cache" / "calibration.json"
_IR_CACHE    = Path(__file__).parent.parent / "cache" / "calibration_ir.pkl"
_MIN_SAMPLE  = 5     # bets before any per-group correction
_FULL_TRUST  = 40    # samples for full multiplicative weight
_MAX_WEIGHT  = 0.85
_MAX_FACTOR  = 1.40
_MIN_FACTOR  = 0.65
_IR_MIN      = 20    # global IR threshold
_IR_SPORT    = 15    # per-sport IR threshold


def _blend_factor(raw: float, n: int) -> float:
    w = min(max((n - _MIN_SAMPLE) / (_FULL_TRUST - _MIN_SAMPLE), 0.0), 1.0) * _MAX_WEIGHT
    blended = 1.0 + (raw - 1.0) * w
    return max(_MIN_FACTOR, min(_MAX_FACTOR, blended))


class ModelCalibrator:
    """
    Self-learning probability calibrator.
    Safe to call with zero historical data (returns identity / neutral values).
    """

    def __init__(self):
        # Multiplicative fallback factors
        self._sport_factor: dict[str, float] = {}
        self._type_factor:  dict[str, float] = {}
        self._kelly_mult:   dict[str, float] = {}
        self._segment_roi:  dict[tuple, float] = {}
        self._segment_n:    dict[tuple, int]   = {}
        # IsotonicRegression calibrators (primary when enough data)
        self._ir_global = None
        self._ir_sport: dict[str, Any] = {}
        # Brier tracking
        self._brier_before: float | None = None
        self._brier_after:  float | None = None
        # Public
        self.n_settled:      int             = 0
        self.brier_score:    float | None    = None
        self.log_loss:       float | None    = None
        self.segment_report: list[dict]      = []

    # ─── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, bets: list[Any]) -> "ModelCalibrator":
        if not bets:
            return self
        self.n_settled = len(bets)

        self._sport_factor = self._fit_groups(bets, lambda b: b.sport, "sport")
        self._type_factor  = self._fit_groups(bets, lambda b: b.bet_type, "bet_type")
        self._kelly_mult   = self._compute_kelly_multipliers(bets)
        self.brier_score   = self._brier(bets)
        self.log_loss      = self._log_loss_score(bets)
        self._fit_isotonic(bets)
        self._build_segment_report(bets)

        logger.info(
            "Calibrator fitted: n=%d Brier=%.4f→%.4f (%.1f%% improvement) LogLoss=%.4f",
            self.n_settled,
            self._brier_before or 0.0,
            self._brier_after or (self.brier_score or 0.0),
            self.brier_improvement,
            self.log_loss or 0.0,
        )
        return self

    def _fit_isotonic(self, bets: list[Any]) -> None:
        """Fit IsotonicRegression globally and per-sport. Silent on failure."""
        valid = [(b.model_prob, 1 if b.won else 0) for b in bets if b.model_prob is not None]
        if not valid:
            return

        probs    = np.array([x[0] for x in valid], dtype=float)
        outcomes = np.array([x[1] for x in valid], dtype=float)

        if len(valid) >= _IR_MIN:
            try:
                from sklearn.isotonic import IsotonicRegression
                self._brier_before = float(np.mean((probs - outcomes) ** 2))
                ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
                ir.fit(probs, outcomes)
                self._ir_global = ir
                cal = ir.predict(probs)
                self._brier_after = float(np.mean((cal - outcomes) ** 2))
            except Exception as exc:
                logger.debug("Global IR fit failed: %s", exc)

        # Per-sport
        self._ir_sport = {}
        sports = set(b.sport for b in bets)
        for sport in sports:
            sb = [(b.model_prob, 1 if b.won else 0)
                  for b in bets if b.sport == sport and b.model_prob is not None]
            if len(sb) >= _IR_SPORT:
                try:
                    from sklearn.isotonic import IsotonicRegression
                    sp = np.array([x[0] for x in sb], dtype=float)
                    so = np.array([x[1] for x in sb], dtype=float)
                    ir_s = IsotonicRegression(out_of_bounds="clip", increasing=True)
                    ir_s.fit(sp, so)
                    self._ir_sport[sport] = ir_s
                except Exception as exc:
                    logger.debug("Sport IR fit [%s] failed: %s", sport, exc)

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
            factors[k] = _blend_factor(actual / expected, n)
        return factors

    def _compute_kelly_multipliers(self, bets) -> dict[str, float]:
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
                "Status": (
                    "✅ profitable" if roi > 2 and n >= 5
                    else "🔍 small sample" if n < _MIN_SAMPLE
                    else "⚠️ break-even" if abs(roi) <= 2
                    else "❌ losing"
                ),
            })
        self.segment_report = rows

    # ─── Inference ────────────────────────────────────────────────────────────

    def get_combined_factor(self, sport: str, bet_type: str) -> float:
        sf = self._sport_factor.get(sport, 1.0)
        tf = self._type_factor.get(bet_type, 1.0)
        return max(_MIN_FACTOR, min(_MAX_FACTOR, (sf * tf) ** 0.5))

    def calibrate_prob(self, prob: float, sport: str, bet_type: str) -> float:
        """
        Apply learned calibration. Uses IsotonicRegression when available
        (non-linear, more accurate), falls back to multiplicative correction.
        """
        p = max(0.02, min(0.98, float(prob)))

        # Stage 1: isotonic regression (sport-specific preferred, then global)
        ir = self._ir_sport.get(sport) or self._ir_global
        if ir is not None:
            return max(0.02, min(0.98, float(ir.predict([p])[0])))

        # Stage 2: multiplicative fallback
        if self.n_settled < _MIN_SAMPLE:
            return p
        return max(0.02, min(0.98, p * self.get_combined_factor(sport, bet_type)))

    def kelly_multiplier(self, sport: str) -> float:
        return self._kelly_mult.get(sport, 1.0)

    def is_profitable_segment(self, sport: str, bet_type: str) -> bool:
        key = (sport, bet_type)
        n   = self._segment_n.get(key, 0)
        if n < _MIN_SAMPLE:
            return True
        return self._segment_roi.get(key, 0.0) > -5.0

    @property
    def brier_improvement(self) -> float:
        """Percentage Brier score reduction from isotonic calibration."""
        if self._brier_before and self._brier_after and self._brier_before > 0:
            return round((self._brier_before - self._brier_after) / self._brier_before * 100, 1)
        return 0.0

    @property
    def using_isotonic(self) -> bool:
        return self._ir_global is not None or bool(self._ir_sport)

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
                    "n_settled":     self.n_settled,
                    "brier_score":   self.brier_score,
                    "brier_before":  self._brier_before,
                    "brier_after":   self._brier_after,
                    "log_loss":      self.log_loss,
                    "sport_factor":  self._sport_factor,
                    "type_factor":   self._type_factor,
                    "kelly_mult":    self._kelly_mult,
                }, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Calibration JSON save failed: %s", exc)
        # Pickle IR objects separately
        try:
            with open(_IR_CACHE, "wb") as f:
                pickle.dump({
                    "ir_global":    self._ir_global,
                    "ir_sport":     self._ir_sport,
                }, f)
        except Exception as exc:
            logger.warning("Calibration IR save failed: %s", exc)
        logger.info("Calibration saved (%d settled bets, isotonic=%s)", self.n_settled, self.using_isotonic)

    @classmethod
    def load(cls) -> "ModelCalibrator":
        c = cls()
        if _CACHE_PATH.exists():
            try:
                d = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
                c.n_settled      = d.get("n_settled", 0)
                c.brier_score    = d.get("brier_score")
                c._brier_before  = d.get("brier_before")
                c._brier_after   = d.get("brier_after")
                c.log_loss       = d.get("log_loss")
                c._sport_factor  = d.get("sport_factor", {})
                c._type_factor   = d.get("type_factor", {})
                c._kelly_mult    = d.get("kelly_mult", {})
            except Exception as exc:
                logger.warning("Calibration JSON load failed: %s", exc)
        if _IR_CACHE.exists():
            try:
                with open(_IR_CACHE, "rb") as f:
                    ir_data = pickle.load(f)
                c._ir_global = ir_data.get("ir_global")
                c._ir_sport  = ir_data.get("ir_sport", {})
            except Exception as exc:
                logger.warning("Calibration IR load failed: %s", exc)
        return c

    @classmethod
    def rebuild_from_db(cls) -> "ModelCalibrator":
        """Refit from all settled bets and persist."""
        from tracker.database import session_scope
        from tracker.models import Bet
        with session_scope() as s:
            bets = s.query(Bet).filter(Bet.settled == True).all()
        c = cls()
        if bets:
            c.fit(bets)
            c.save()
        return c
