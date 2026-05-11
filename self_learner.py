"""
self_learner.py — meta-learning controller for continuous model improvement.

Each cycle:
  1. Settle recent game outcomes via OutcomeTracker
  2. Compute rolling performance metrics (ROI, calibration, win rate)
  3. Adapt min_edge + kelly_fraction based on whether the model is profitable
  4. Trigger model retraining when:
       • N new bets have been settled since last retrain
       • Brier score degrades by more than a threshold (model miscalibrated)
       • Caller forces retrain

Strategy adaptation rules
  ROI < -2%   → raise min_edge 10%, lower kelly 5%   (be more selective)
  ROI  2-3%   → no change                             (hold steady)
  ROI > 3%    → lower min_edge 5%, raise kelly 5%    (capture more value)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Strategy parameter bounds
_MIN_EDGE_LO = 0.020   # floor — never require less than 2% edge
_MIN_EDGE_HI = 0.080   # ceiling — never require more than 8% edge
_KELLY_LO    = 0.10
_KELLY_HI    = 0.40

# Retraining triggers
_RETRAIN_AFTER_N_BETS = 50    # accumulate this many new settled bets before retraining
_BRIER_DEGRADATION    = 0.05  # absolute Brier worsening that triggers immediate retrain
_MIN_BETS_FOR_ADAPT   = 20    # minimum settled bets before adapting params

# Evaluation windows
_WINDOW_SHORT = 25
_WINDOW_LONG  = 100


class SelfLearner:
    """
    Meta-learning controller.

    Usage:
        learner = SelfLearner(sport="NFL")
        result  = learner.run_cycle(games, markets=["h2h"])
        learner.print_summary(result)
    """

    def __init__(self, sport: str = "NFL", db_path: Path | None = None) -> None:
        from outcome_tracker import OutcomeTracker
        self.sport   = sport.upper()
        self.tracker = OutcomeTracker() if db_path is None else OutcomeTracker(db_path)
        self._last_brier: dict[str, float] = {}

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_cycle(
        self,
        games: list[dict],
        markets: list[str] | None = None,
        force_retrain: bool = False,
    ) -> dict:
        """
        Full self-learning cycle.

        Args:
            games:         Historical game list from DataCollector — used for
                           building the retraining feature matrix only.
            markets:       Markets to evaluate/retrain (default: ["h2h"]).
            force_retrain: Retrain regardless of trigger conditions.

        Returns dict with keys: sport, params, metrics, retrained, settled.
        """
        markets = markets or ["h2h"]
        logger.info("Self-learning cycle starting for %s…", self.sport)

        # 1. Settle outcomes using OddsAPI recent scores (fresh, not cached ESPN)
        settled = self._settle_recent(markets)
        logger.info("Settled %d new outcomes.", settled)

        # 2. Evaluate each market
        all_metrics: dict[str, dict] = {}
        needs_retrain = force_retrain or (settled >= _RETRAIN_AFTER_N_BETS)

        for market in markets:
            df = self.tracker.get_settled_df(self.sport, n_recent=_WINDOW_LONG * 3)
            mdf = df[df["market"] == market] if not df.empty else df

            if len(mdf) < _MIN_BETS_FOR_ADAPT:
                logger.info(
                    "%s %s: only %d settled bets — skipping metric evaluation.",
                    self.sport, market, len(mdf),
                )
                continue

            metrics = self._compute_metrics(mdf)
            all_metrics[market] = metrics
            self.tracker.log_performance(self.sport, market, mdf)

            logger.info(
                "%s %s | n=%d | ROI(long)=%+.1f%% | ROI(short)=%+.1f%% | "
                "WR=%.1f%% | Brier=%.4f",
                self.sport, market,
                metrics["n"], metrics["roi"], metrics["short_roi"],
                metrics["win_rate"] * 100, metrics["brier"],
            )

            # Calibration degradation check
            if market in self._last_brier:
                degradation = metrics["brier"] - self._last_brier[market]
                if degradation > _BRIER_DEGRADATION:
                    logger.warning(
                        "%s %s: Brier degraded %.4f → %.4f — triggering retrain.",
                        self.sport, market,
                        self._last_brier[market], metrics["brier"],
                    )
                    needs_retrain = True
            self._last_brier[market] = metrics["brier"]

        # 3. Adapt strategy parameters from overall settled history
        overall_df = self.tracker.get_settled_df(self.sport, n_recent=_WINDOW_LONG)
        params = self._adapt_params(overall_df)
        logger.info(
            "Strategy params for %s: min_edge=%.3f  kelly=%.3f",
            self.sport, params["min_edge"], params["kelly_frac"],
        )

        # 4. Retrain if warranted
        retrained = False
        if needs_retrain:
            logger.info("Retraining models for %s on %d games…", self.sport, len(games))
            self._retrain(games, markets)
            retrained = True

        return {
            "sport":     self.sport,
            "params":    params,
            "metrics":   all_metrics,
            "retrained": retrained,
            "settled":   settled,
        }

    # ── Settlement ────────────────────────────────────────────────────────────

    def _settle_recent(self, markets: list[str]) -> int:
        """Fetch OddsAPI scores for the past 3 days and settle unsettled picks."""
        import config
        from data.odds_api import OddsAPIClient

        sport_key = config.SUPPORTED_SPORTS.get(self.sport)
        if not sport_key:
            logger.warning("No sport key for %s — skipping settlement.", self.sport)
            return 0

        try:
            with OddsAPIClient() as client:
                scores = client.get_scores(sport_key, days_from=3)
            return self.tracker.settle_from_scores_api(self.sport, scores)
        except Exception as exc:
            logger.warning("Scores fetch failed for %s: %s", self.sport, exc)
            return 0

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        long_df  = df.tail(_WINDOW_LONG)
        short_df = df.tail(_WINDOW_SHORT)

        def _roi(subset: pd.DataFrame) -> float:
            won    = subset["won"].astype(float)
            stakes = subset["kelly_stake"].astype(float)
            odds   = subset["dk_odds"].astype(float)
            pnls   = [_pnl(o, s, bool(w)) for o, s, w in zip(odds, stakes, won)]
            w      = stakes.sum()
            return (sum(pnls) / w * 100) if w > 0 else 0.0

        brier    = float(((long_df["model_prob"].astype(float) - long_df["won"].astype(float)) ** 2).mean())
        win_rate = float(long_df["won"].astype(float).mean())
        avg_edge = float(long_df["edge_pct"].astype(float).mean()) if "edge_pct" in long_df.columns else 0.0

        return {
            "n":         len(long_df),
            "roi":       round(_roi(long_df),  2),
            "short_roi": round(_roi(short_df), 2),
            "win_rate":  round(win_rate,  4),
            "brier":     round(brier,     5),
            "avg_edge":  round(avg_edge,  3),
        }

    # ── Strategy adaptation ───────────────────────────────────────────────────

    def _adapt_params(self, df: pd.DataFrame) -> dict:
        """
        Adjust min_edge and kelly_fraction from overall settled history.
        Only updates if we have enough data to be confident.
        """
        current = self.tracker.get_strategy_params(self.sport)
        min_edge   = current["min_edge"]
        kelly_frac = current["kelly_frac"]

        if len(df) < _MIN_BETS_FOR_ADAPT:
            return current

        won    = df["won"].astype(float)
        stakes = df["kelly_stake"].astype(float)
        odds   = df["dk_odds"].astype(float)
        pnls   = [_pnl(o, s, bool(w)) for o, s, w in zip(odds, stakes, won)]
        wagered = stakes.sum()
        roi_frac = (sum(pnls) / wagered) if wagered > 0 else 0.0

        if roi_frac < -0.02:
            min_edge   = min(min_edge * 1.10, _MIN_EDGE_HI)
            kelly_frac = max(kelly_frac * 0.95, _KELLY_LO)
            logger.info(
                "ROI=%+.1f%% below breakeven — raising min_edge to %.3f, "
                "lowering kelly to %.3f.",
                roi_frac * 100, min_edge, kelly_frac,
            )
        elif roi_frac >= 0.03:
            min_edge   = max(min_edge * 0.95, _MIN_EDGE_LO)
            kelly_frac = min(kelly_frac * 1.05, _KELLY_HI)
            logger.info(
                "ROI=%+.1f%% profitable — lowering min_edge to %.3f, "
                "raising kelly to %.3f.",
                roi_frac * 100, min_edge, kelly_frac,
            )

        min_edge   = round(min_edge,   4)
        kelly_frac = round(kelly_frac, 4)
        self.tracker.save_strategy_params(self.sport, min_edge, kelly_frac)
        return {"min_edge": min_edge, "kelly_frac": kelly_frac}

    # ── Retraining ────────────────────────────────────────────────────────────

    def _retrain(self, games: list[dict], markets: list[str]) -> None:
        from feature_engineer import FeatureEngineer
        from model_trainer import ModelTrainer

        fe = FeatureEngineer(sport=self.sport)
        df = fe.build(games)
        if df.empty:
            logger.error("Feature matrix empty — cannot retrain.")
            return

        if "dk_home_spread" in df.columns:
            df["spread_win"] = fe.make_spread_target(df)
        if "dk_total" in df.columns:
            df["over_hit"] = fe.make_totals_target(df)

        trainer = ModelTrainer(sport=self.sport)
        for market in markets:
            metrics = trainer.fit(df, market=market, use_optuna=True)
            if "error" not in metrics:
                logger.info(
                    "Retrained %s %s — AUC xgb=%.3f lgb=%.3f lr=%.3f",
                    self.sport, market,
                    metrics.get("xgboost_auc", 0),
                    metrics.get("lightgbm_auc", 0),
                    metrics.get("logreg_auc", 0),
                )

    # ── Summary output ────────────────────────────────────────────────────────

    def print_summary(self, result: dict) -> None:
        print(f"\n{'=' * 58}")
        print(f"  SELF-LEARNING CYCLE — {result['sport']}")
        print(f"{'=' * 58}")
        print(f"  Outcomes settled : {result['settled']}")
        print(f"  Models retrained : {'Yes' if result['retrained'] else 'No'}")
        params = result["params"]
        print(
            f"  Active params    : min_edge={params['min_edge']:.3f}  "
            f"kelly={params['kelly_frac']:.3f}"
        )
        if result["metrics"]:
            print()
            for market, m in result["metrics"].items():
                print(
                    f"  [{market.upper():7s}]  n={m['n']:4d}  "
                    f"ROI={m['roi']:+6.1f}%  short={m['short_roi']:+6.1f}%  "
                    f"WR={m['win_rate']:.1%}  Brier={m['brier']:.4f}"
                )
        print(f"{'=' * 58}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pnl(dk_odds: float, stake: float, won: bool) -> float:
    if won:
        net = dk_odds / 100.0 if dk_odds >= 0 else 100.0 / abs(dk_odds)
        return stake * net
    return -stake
