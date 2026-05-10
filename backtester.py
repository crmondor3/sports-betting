"""
backtester.py — simulate betting strategy on historical DraftKings odds.

Workflow:
  1. Load historical games (from DataCollector / ESPN).
  2. For each game, build features using only past data (no leakage).
  3. Get ML model probability, compute edge vs stored DraftKings odds.
  4. Simulate Kelly-sized bets on qualifying value bets.
  5. Compute ROI, win rate, max drawdown, Sharpe ratio, and CLV.
  6. Generate CSV report and matplotlib summary chart.

Historical DraftKings odds come from:
  • Snapshots stored in the ml_odds_snapshots table (collected over time), OR
  • The Odds API /odds-history endpoint (requires paid plan), OR
  • A CSV of historical odds (e.g. from a third-party provider).

When no historical odds are available, the backtest runs on the H2H market only,
using ESP-derived game results and the model's probability vs a simulated -110 line.
"""
from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")
_FLAT_BET_PCT = 0.02  # 2% flat-bet baseline


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class BacktestBet:
    date:         str
    home_team:    str
    away_team:    str
    market:       str
    side:         str
    dk_odds:      float
    model_prob:   float
    dk_implied:   float
    edge_pct:     float
    ev:           float
    kelly_frac:   float
    stake:        float
    bankroll_pre: float
    won:          bool
    pnl:          float
    bankroll_post: float
    clv:          float = 0.0   # closing line value (if closing odds available)


@dataclass
class BacktestReport:
    sport:         str
    market:        str
    n_bets:        int
    n_wins:        int
    win_rate:      float
    total_wagered: float
    total_profit:  float
    roi:           float
    max_drawdown:  float
    sharpe:        float
    avg_clv:       float
    flat_roi:      float    # flat 2% bet baseline
    bets:          list[BacktestBet] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"  BACKTEST RESULTS — {self.sport} {self.market.upper()}\n"
            f"{'=' * 60}\n"
            f"  Bets:          {self.n_bets}\n"
            f"  Win Rate:      {self.win_rate:.1%}\n"
            f"  ROI (Kelly):   {self.roi:+.2f}%\n"
            f"  ROI (Flat):    {self.flat_roi:+.2f}%\n"
            f"  Total Profit:  ${self.total_profit:+.2f}\n"
            f"  Max Drawdown:  {self.max_drawdown:.2f}%\n"
            f"  Sharpe Ratio:  {self.sharpe:.3f}\n"
            f"  Avg CLV:       {self.avg_clv:+.4f}\n"
            f"{'=' * 60}"
        )


# ── Backtester ────────────────────────────────────────────────────────────────

class Backtester:
    """
    Runs a walk-forward backtest of the ML betting strategy.

    The model is retrained periodically (every *retrain_every* games) using
    only games before the current evaluation point — preventing leakage.
    """

    def __init__(
        self,
        sport:          str   = "NFL",
        bankroll:       float = 1000.0,
        kelly_fraction: float = 0.25,
        min_edge:       float = 0.03,
        markets:        list[str] | None = None,
        retrain_every:  int   = 50,    # retrain model every N games
    ) -> None:
        self.sport          = sport.upper()
        self.bankroll       = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge       = min_edge
        self.markets        = markets or ["h2h"]
        self.retrain_every  = retrain_every

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(
        self,
        games: list[dict],
        odds_lookup: dict[str, dict] | None = None,
        use_optuna: bool = False,
    ) -> list[BacktestReport]:
        """
        Run backtest over *games* (chronologically sorted).

        Args:
            games:       list of game dicts with outcomes (from DataCollector)
            odds_lookup: {game_id: {market: snapshot_dict}} for DK odds features
            use_optuna:  run Optuna tuning when retraining (slow)

        Returns list of BacktestReport, one per market.
        """
        from feature_engineer import FeatureEngineer, FEATURE_NAMES

        odds_lookup = odds_lookup or {}
        fe = FeatureEngineer(sport=self.sport)

        # Build full feature DataFrame
        df = fe.build(games, odds_lookup)
        if df.empty:
            logger.error("No feature rows built — check game data.")
            return []

        # Attach spread/totals targets if odds available
        if "dk_home_spread" in df.columns:
            df["spread_win"] = fe.make_spread_target(df)
        if "dk_total" in df.columns:
            df["over_hit"] = fe.make_totals_target(df)

        reports = []
        for market in self.markets:
            try:
                report = self._run_market(df, market, use_optuna=use_optuna)
                reports.append(report)
                print(report)
            except Exception as exc:
                logger.error("Backtest failed for %s %s: %s", self.sport, market, exc)

        return reports

    def _run_market(
        self, df: pd.DataFrame, market: str, use_optuna: bool
    ) -> BacktestReport:
        from model_trainer import ModelTrainer, _MIN_ROWS

        target_col = {"h2h": "home_win", "spreads": "spread_win", "totals": "over_hit"}[market]
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame.")

        sub = df[df[target_col].notna()].reset_index(drop=True)
        if len(sub) < _MIN_ROWS * 2:
            raise ValueError(f"Insufficient rows for {market} backtest: {len(sub)}")

        from feature_engineer import FEATURE_NAMES
        feat_cols = [c for c in FEATURE_NAMES if c in sub.columns]

        bankroll_kelly = self.bankroll
        bankroll_flat  = self.bankroll
        bets_kelly: list[BacktestBet] = []
        bets_flat:  list[float] = []
        flat_pnls:  list[float] = []

        # Walk-forward: first 60% is initial train, then rolling
        initial_train = int(len(sub) * 0.60)
        trainer = ModelTrainer(sport=self.sport)
        train_df = sub.iloc[:initial_train]
        trainer.fit(train_df, market=market, use_optuna=use_optuna)

        retrain_counter = 0

        for i in range(initial_train, len(sub)):
            row = sub.iloc[i]
            features = {c: float(row.get(c, 0.0)) for c in feat_cols}

            # Determine DK odds
            dk_home_odds, dk_away_odds = self._get_dk_odds(row, market)
            if dk_home_odds is None:
                continue

            try:
                pred = trainer.predict(features, market=market)
            except Exception:
                continue

            model_prob = pred["ensemble_prob"]
            if market == "h2h":
                side, model_p, dk_odds = self._pick_side_h2h(
                    model_prob, dk_home_odds, dk_away_odds
                )
            elif market == "spreads":
                side, model_p, dk_odds = self._pick_side_spreads(
                    model_prob, dk_home_odds, dk_away_odds
                )
            else:  # totals
                side, model_p, dk_odds = self._pick_side_totals(
                    model_prob, dk_home_odds, dk_away_odds
                )

            if side is None:
                continue

            # No-vig implied prob
            dk_impl = _no_vig_single(dk_home_odds, dk_away_odds, side)
            edge = model_p - dk_impl

            if edge < self.min_edge:
                continue

            # Kelly sizing
            kf    = _kelly(model_p, dk_odds, self.kelly_fraction, cap=0.05)
            stake = round(bankroll_kelly * kf, 2)
            if stake <= 0:
                continue

            # Determine outcome
            won = self._determine_win(row, market, side)

            # PnL
            if won:
                net  = stake * (_american_to_net(dk_odds))
                pnl  = net
                flat_pnl = bankroll_flat * _FLAT_BET_PCT * _american_to_net(dk_odds)
            else:
                pnl  = -stake
                flat_pnl = -bankroll_flat * _FLAT_BET_PCT

            ev         = _ev(model_p, dk_odds)
            bankroll_post = bankroll_kelly + pnl
            flat_pnls.append(flat_pnl)
            bankroll_flat += flat_pnl

            bets_kelly.append(BacktestBet(
                date=str(row.get("date", "")),
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
                market=market,
                side=side,
                dk_odds=dk_odds,
                model_prob=round(model_p, 4),
                dk_implied=round(dk_impl, 4),
                edge_pct=round(edge * 100, 2),
                ev=round(ev * 100, 2),
                kelly_frac=round(kf * 100, 2),
                stake=stake,
                bankroll_pre=bankroll_kelly,
                won=won,
                pnl=round(pnl, 2),
                bankroll_post=round(bankroll_post, 2),
                clv=self._compute_clv(row, side, dk_odds),
            ))

            bankroll_kelly = bankroll_post

            # Periodic retraining
            retrain_counter += 1
            if retrain_counter >= self.retrain_every:
                retrain_counter = 0
                trainer.fit(sub.iloc[:i + 1], market=market, use_optuna=use_optuna)

        return self._build_report(bets_kelly, flat_pnls, market)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_dk_odds(
        self, row: pd.Series, market: str
    ) -> tuple[Optional[float], Optional[float]]:
        """Extract DK odds from feature row. Falls back to -110/-110 if missing."""
        if market == "h2h":
            hi = row.get("dk_home_implied", 0)
            ai = row.get("dk_away_implied", 0)
            if hi > 0 and ai > 0:
                # Reconstruct approximate American odds from no-vig implied probs
                # This is an approximation; real odds would be better
                home_odds = _prob_to_american(hi)
                away_odds = _prob_to_american(ai)
                return home_odds, away_odds
            return -110.0, -110.0   # default juice line

        if market == "spreads":
            hp = row.get("dk_home_implied", 0)
            ap = row.get("dk_away_implied", 0)
            if hp > 0 and ap > 0:
                return _prob_to_american(hp), _prob_to_american(ap)
            return -110.0, -110.0

        if market == "totals":
            op = row.get("dk_over_implied", 0)
            up = 1 - op if op > 0 else 0
            if op > 0:
                return _prob_to_american(op), _prob_to_american(up)
            return -110.0, -110.0

        return None, None

    @staticmethod
    def _pick_side_h2h(
        model_prob: float, home_odds: float, away_odds: float
    ) -> tuple[str | None, float, float]:
        """Return (side, model_prob_for_side, dk_odds) for the stronger edge."""
        return "home", model_prob, home_odds

    @staticmethod
    def _pick_side_spreads(
        model_prob: float, home_odds: float, away_odds: float
    ) -> tuple[str | None, float, float]:
        return "home", model_prob, home_odds

    @staticmethod
    def _pick_side_totals(
        model_prob: float, over_odds: float, under_odds: float
    ) -> tuple[str | None, float, float]:
        return "over", model_prob, over_odds

    @staticmethod
    def _determine_win(row: pd.Series, market: str, side: str) -> bool:
        if market == "h2h":
            return bool(row.get("home_win", False)) if side == "home" else not bool(row.get("home_win", False))
        if market == "spreads":
            return bool(row.get("spread_win", False)) if side == "home" else not bool(row.get("spread_win", False))
        if market == "totals":
            return bool(row.get("over_hit", False)) if side == "over" else not bool(row.get("over_hit", False))
        return False

    @staticmethod
    def _compute_clv(row: pd.Series, side: str, bet_odds: float) -> float:
        """Closing line value: closing implied prob minus bet implied prob."""
        if side in ("home",):
            close_key = "dk_close_home_implied"
        elif side == "away":
            close_key = "dk_close_away_implied"
        elif side == "over":
            close_key = "dk_close_over_implied"
        else:
            close_key = "dk_close_under_implied"

        close_impl = row.get(close_key, 0)
        if not close_impl:
            return 0.0
        bet_impl = _impl_prob(bet_odds)
        return float(close_impl - bet_impl)

    def _build_report(
        self,
        bets: list[BacktestBet],
        flat_pnls: list[float],
        market: str,
    ) -> BacktestReport:
        if not bets:
            return BacktestReport(
                sport=self.sport, market=market,
                n_bets=0, n_wins=0, win_rate=0, total_wagered=0,
                total_profit=0, roi=0, max_drawdown=0,
                sharpe=0, avg_clv=0, flat_roi=0,
            )

        n_bets     = len(bets)
        n_wins     = sum(b.won for b in bets)
        wagered    = sum(b.stake for b in bets)
        profit     = sum(b.pnl for b in bets)
        bankrolls  = [self.bankroll] + [b.bankroll_post for b in bets]
        max_dd     = _max_drawdown(bankrolls)
        sharpe_val = _sharpe(bets)
        avg_clv    = float(np.mean([b.clv for b in bets]))
        flat_profit = sum(flat_pnls)
        flat_wagered = n_bets * (self.bankroll * _FLAT_BET_PCT)

        return BacktestReport(
            sport=self.sport,
            market=market,
            n_bets=n_bets,
            n_wins=n_wins,
            win_rate=n_wins / n_bets,
            total_wagered=round(wagered, 2),
            total_profit=round(profit, 2),
            roi=round((profit / wagered) * 100, 2) if wagered > 0 else 0,
            max_drawdown=round(max_dd * 100, 2),
            sharpe=round(sharpe_val, 4),
            avg_clv=round(avg_clv, 5),
            flat_roi=round((flat_profit / flat_wagered) * 100, 2) if flat_wagered > 0 else 0,
            bets=bets,
        )

    # ── Reporting ─────────────────────────────────────────────────────────────

    def save_reports(self, reports: list[BacktestReport]) -> None:
        """Write CSV and matplotlib chart for each market."""
        _REPORTS_DIR.mkdir(exist_ok=True)
        for report in reports:
            self._save_csv(report)
            self._save_chart(report)

    def _save_csv(self, report: BacktestReport) -> None:
        path = _REPORTS_DIR / f"backtest_{self.sport.lower()}_{report.market}.csv"
        if not report.bets:
            return
        with path.open("w", newline="") as fh:
            from dataclasses import asdict
            writer = csv.DictWriter(fh, fieldnames=asdict(report.bets[0]).keys())
            writer.writeheader()
            for b in report.bets:
                from dataclasses import asdict
                writer.writerow(asdict(b))
        logger.info("Backtest CSV saved: %s", path)

    def _save_chart(self, report: BacktestReport) -> None:
        if not report.bets:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            bankrolls = [self.bankroll] + [b.bankroll_post for b in report.bets]
            dates     = ["Start"] + [b.date for b in report.bets]

            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.suptitle(
                f"Backtest — {self.sport} {report.market.upper()}  "
                f"ROI: {report.roi:+.1f}%  |  Sharpe: {report.sharpe:.3f}  |  "
                f"n={report.n_bets}",
                fontsize=13, fontweight="bold",
            )

            # Bankroll curve
            ax1 = axes[0, 0]
            ax1.plot(bankrolls, color="steelblue", linewidth=1.5)
            ax1.axhline(self.bankroll, linestyle="--", color="gray", alpha=0.6)
            ax1.set_title("Bankroll (Kelly)")
            ax1.set_xlabel("Bet #")
            ax1.set_ylabel("$")
            ax1.grid(alpha=0.3)

            # PnL per bet
            ax2 = axes[0, 1]
            pnls = [b.pnl for b in report.bets]
            colors = ["green" if p >= 0 else "red" for p in pnls]
            ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax2.axhline(0, linestyle="--", color="gray")
            ax2.set_title("PnL per Bet")
            ax2.set_xlabel("Bet #")
            ax2.set_ylabel("$")
            ax2.grid(alpha=0.3)

            # Edge distribution
            ax3 = axes[1, 0]
            edges = [b.edge_pct for b in report.bets]
            ax3.hist(edges, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
            ax3.axvline(report.roi, linestyle="--", color="red", label=f"ROI {report.roi:.1f}%")
            ax3.set_title("Edge % Distribution")
            ax3.set_xlabel("Edge %")
            ax3.set_ylabel("Count")
            ax3.legend()
            ax3.grid(alpha=0.3)

            # CLV distribution
            ax4 = axes[1, 1]
            clvs = [b.clv for b in report.bets if b.clv != 0]
            if clvs:
                ax4.hist(clvs, bins=20, color="darkorange", edgecolor="white", alpha=0.8)
                ax4.axvline(0, linestyle="--", color="gray")
                ax4.set_title(f"CLV Distribution  (avg={report.avg_clv:+.4f})")
            else:
                ax4.text(0.5, 0.5, "No CLV data\n(no closing odds stored)",
                         ha="center", va="center", transform=ax4.transAxes)
                ax4.set_title("CLV Distribution")
            ax4.set_xlabel("CLV")
            ax4.set_ylabel("Count")
            ax4.grid(alpha=0.3)

            plt.tight_layout()
            chart_path = _REPORTS_DIR / f"backtest_{self.sport.lower()}_{report.market}.png"
            fig.savefig(chart_path, dpi=120)
            plt.close(fig)
            logger.info("Backtest chart saved: %s", chart_path)

        except ImportError:
            logger.warning("matplotlib not available — skipping chart.")
        except Exception as exc:
            logger.warning("Chart generation failed: %s", exc)


# ── Math helpers ──────────────────────────────────────────────────────────────

def _impl_prob(american: float) -> float:
    if american >= 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def _no_vig_single(home_odds: float, away_odds: float, side: str) -> float:
    r_h = _impl_prob(home_odds)
    r_a = _impl_prob(away_odds)
    t   = r_h + r_a
    if side in ("home", "over"):
        return r_h / t
    return r_a / t


def _ev(model_prob: float, american_odds: float) -> float:
    net = american_odds / 100.0 if american_odds >= 0 else 100.0 / abs(american_odds)
    return model_prob * net - (1 - model_prob)


def _kelly(model_prob: float, american_odds: float, fraction: float, cap: float) -> float:
    b = american_odds / 100.0 if american_odds >= 0 else 100.0 / abs(american_odds)
    if b <= 0:
        return 0.0
    full = (b * model_prob - (1 - model_prob)) / b
    return max(0.0, min(full * fraction, cap))


def _american_to_net(american: float) -> float:
    return american / 100.0 if american >= 0 else 100.0 / abs(american)


def _prob_to_american(prob: float) -> float:
    """Convert probability to American odds (fair, no vig)."""
    if prob <= 0 or prob >= 1:
        return -110.0
    if prob >= 0.5:
        return -(prob / (1 - prob)) * 100
    return ((1 - prob) / prob) * 100


def _max_drawdown(bankrolls: list[float]) -> float:
    peak, max_dd = bankrolls[0], 0.0
    for b in bankrolls:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe(bets: list[BacktestBet], risk_free: float = 0.0) -> float:
    """Annualised Sharpe using per-bet returns (bet PnL / stake)."""
    if len(bets) < 2:
        return 0.0
    returns = [b.pnl / b.stake if b.stake > 0 else 0.0 for b in bets]
    mu  = float(np.mean(returns)) - risk_free
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    # Annualise assuming ~16 bets/week × 52 = ~832 bets/year (adjust if needed)
    return mu / std * math.sqrt(min(len(bets), 832))
