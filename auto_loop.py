"""
auto_loop.py — daily self-learning automation loop across all supported sports.

Each cycle (per sport):
  1. Fetch recent OddsAPI scores → settle past picks
  2. Self-learning: evaluate metrics → adapt params → retrain if needed
  3. Fetch today's DraftKings odds
  4. Generate picks with adapted parameters
  5. Persist picks to DB for future outcome tracking

Usage:
  python auto_loop.py                          # all sports, one cycle
  python auto_loop.py --sport NBA              # single sport
  python auto_loop.py --daemon                 # all sports, loop every 24h
  python auto_loop.py --settle                 # settle outcomes only
  python auto_loop.py --retrain                # force retrain all sports
  python auto_loop.py --status                 # print performance summary
  python auto_loop.py --sport NFL --backtest   # run backtest for one sport
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("auto_loop.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("auto_loop")

_CYCLE_HOURS   = 24
_ALL_SPORTS    = list(config.SUPPORTED_SPORTS.keys())
_DEFAULT_MARKETS = ["h2h", "spreads", "totals"]


# ── Per-sport pipeline ────────────────────────────────────────────────────────

def run_sport(
    sport: str,
    bankroll: float,
    markets: list[str],
    top_n: int,
    force_retrain: bool,
) -> bool:
    """
    Run one full cycle for a single sport.
    Returns True if picks were generated, False if sport is off-season/no games.
    """
    from data_collector import DataCollector
    from data.odds_api import OddsAPIClient
    from outcome_tracker import OutcomeTracker
    from self_learner import SelfLearner
    from bet_selector import BetSelector

    sport_key = config.SUPPORTED_SPORTS[sport]
    has_espn  = sport in config.ESPN_SPORTS
    logger.info("--- %s cycle start (mode=%s) ---", sport, "ML+consensus" if has_espn else "consensus-only")

    # 1. Fetch historical training data (ESPN sports only; others return [] immediately)
    dc = DataCollector()
    games = dc.fetch_historical_games(sport)
    if not games and has_espn:
        logger.warning("%s: no historical game data, skipping.", sport)
        return False

    # 2. Snapshot today's DraftKings lines (for line-movement features)
    try:
        dc.fetch_and_store_odds(sport)
    except Exception as exc:
        logger.debug("%s: odds snapshot skipped (%s)", sport, exc)

    # 3. Self-learning: settle → evaluate → adapt → maybe retrain (ESPN sports only)
    if has_espn and games:
        learner = SelfLearner(sport=sport)
        result  = learner.run_cycle(games, markets=markets, force_retrain=force_retrain)
        learner.print_summary(result)

    # 4. Load adapted params for this sport
    tracker    = OutcomeTracker()
    params     = tracker.get_strategy_params(sport)
    min_edge   = params["min_edge"]
    kelly_frac = params["kelly_frac"]
    logger.info(
        "%s: using min_edge=%.3f  kelly=%.3f", sport, min_edge, kelly_frac
    )

    # 5. Fetch today's odds
    try:
        with OddsAPIClient() as client:
            odds_games = client.get_odds(sport_key)
    except Exception as exc:
        logger.error("%s: could not fetch odds — %s", sport, exc)
        return False

    if not odds_games:
        logger.info("%s: no upcoming games with DraftKings odds today.", sport)
        return False

    # 6. Generate picks
    selector = BetSelector(
        sport=sport,
        bankroll=bankroll,
        kelly_fraction=kelly_frac,
        min_edge=min_edge,
        markets=markets,
    )
    bets = selector.analyze(odds_games, games)
    selector.print_top(bets, n=top_n)
    selector.save_csv(bets)       # also auto-persists to ml_picks via BetSelector

    logger.info("%s: %d qualifying bets found.", sport, len([b for b in bets if b.ev > 0]))
    return True


def run_all_sports(
    bankroll: float,
    markets: list[str],
    top_n: int,
    force_retrain: bool,
    sports: list[str] | None = None,
) -> None:
    """Run cycle for every sport, collecting per-sport errors without aborting."""
    target = sports or _ALL_SPORTS
    results: dict[str, str] = {}

    for sport in target:
        try:
            found = run_sport(
                sport=sport,
                bankroll=bankroll,
                markets=markets,
                top_n=top_n,
                force_retrain=force_retrain,
            )
            results[sport] = "picks generated" if found else "no games"
        except Exception as exc:
            logger.error("%s: cycle failed — %s", sport, exc, exc_info=True)
            results[sport] = f"ERROR: {exc}"

    # Summary
    print(f"\n{'=' * 50}")
    print("  CYCLE SUMMARY")
    print(f"{'=' * 50}")
    for sp, status in results.items():
        print(f"  {sp:<6} {status}")
    print(f"{'=' * 50}\n")


# ── Utility commands ──────────────────────────────────────────────────────────

def settle_only(sports: list[str]) -> None:
    """Settle outcomes for each sport using OddsAPI scores."""
    from data.odds_api import OddsAPIClient
    from outcome_tracker import OutcomeTracker

    tracker = OutcomeTracker()
    for sport in sports:
        sport_key = config.SUPPORTED_SPORTS.get(sport)
        if not sport_key:
            continue
        try:
            with OddsAPIClient() as client:
                scores = client.get_scores(sport_key, days_from=3)
            n = tracker.settle_from_scores_api(sport, scores)
            print(f"  {sport}: settled {n} outcomes")
        except Exception as exc:
            logger.error("%s: settlement failed — %s", sport, exc)


def show_status(sports: list[str]) -> None:
    """Print performance summary for each sport."""
    from outcome_tracker import OutcomeTracker

    tracker = OutcomeTracker()
    for sport in sports:
        params = tracker.get_strategy_params(sport)
        print(f"\n{'=' * 50}")
        print(f"  {sport}  |  min_edge={params['min_edge']:.3f}  kelly={params['kelly_frac']:.3f}")
        print(f"{'=' * 50}")
        for market in ("h2h", "spreads", "totals"):
            df = tracker.get_settled_df(sport, n_recent=500)
            mdf = df[df["market"] == market] if not df.empty else df
            if mdf.empty:
                continue
            won    = mdf["won"].astype(float)
            stakes = mdf["kelly_stake"].astype(float)
            odds   = mdf["dk_odds"].astype(float)
            pnls   = [
                (s * (o / 100 if o >= 0 else 100 / abs(o)) if w else -s)
                for o, s, w in zip(odds, stakes, won)
            ]
            wagered = stakes.sum()
            roi = (sum(pnls) / wagered * 100) if wagered > 0 else 0
            brier = float(((mdf["model_prob"].astype(float) - won) ** 2).mean())
            print(
                f"  [{market.upper():7}]  n={len(mdf):4d}  ROI={roi:+.1f}%  "
                f"WR={won.mean():.1%}  Brier={brier:.4f}"
            )


def run_backtest(sport: str, bankroll: float, markets: list[str]) -> None:
    from data_collector import DataCollector
    from backtester import Backtester

    dc    = DataCollector()
    games = dc.fetch_historical_games(sport)
    if not games:
        logger.error("No historical game data for %s.", sport)
        return

    bt = Backtester(
        sport=sport,
        bankroll=bankroll,
        markets=markets,
        retrain_every=50,
    )
    reports = bt.run(games, use_optuna=False)
    bt.save_reports(reports)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="auto_loop.py",
        description="Daily self-learning sports betting loop — all sports",
    )
    p.add_argument(
        "--sport", metavar="SPORT",
        help=f"Single sport to target. Default: all sports. Options: {', '.join(sorted(_ALL_SPORTS))}",
    )
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--top-n",   type=int,   default=5)
    p.add_argument(
        "--market", nargs="+",
        choices=["h2h", "spreads", "totals"],
        default=_DEFAULT_MARKETS,
    )
    p.add_argument(
        "--daemon", action="store_true",
        help="Run continuously, cycling every 24 hours",
    )
    p.add_argument(
        "--settle", action="store_true",
        help="Settle outcomes only — no picks generated",
    )
    p.add_argument(
        "--retrain", action="store_true",
        help="Force model retraining this cycle",
    )
    p.add_argument(
        "--status", action="store_true",
        help="Print performance summary and exit",
    )
    p.add_argument(
        "--backtest", action="store_true",
        help="Run historical backtest (requires --sport)",
    )
    return p


def main() -> None:
    args   = _build_parser().parse_args()
    sports = [args.sport.upper()] if args.sport else _ALL_SPORTS

    # Validate sport names
    for s in sports:
        if s not in config.SUPPORTED_SPORTS:
            print(f"Unknown sport '{s}'. Supported: {_ALL_SPORTS}")
            sys.exit(1)

    if args.status:
        show_status(sports)
        return

    if args.settle:
        print("Settling outcomes for:", sports)
        settle_only(sports)
        return

    if args.backtest:
        if len(sports) > 1:
            print("--backtest requires --sport to specify a single sport.")
            sys.exit(1)
        run_backtest(sports[0], args.bankroll, args.market)
        return

    if args.daemon:
        logger.info("Daemon mode — running %s every %dh.", sports, _CYCLE_HOURS)
        while True:
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            logger.info("=== Daemon cycle: %s ===", ts)
            try:
                run_all_sports(
                    bankroll=args.bankroll,
                    markets=args.market,
                    top_n=args.top_n,
                    force_retrain=args.retrain,
                    sports=sports,
                )
            except Exception as exc:
                logger.error("Daemon cycle error: %s", exc, exc_info=True)

            next_run = datetime.utcnow() + timedelta(hours=_CYCLE_HOURS)
            logger.info(
                "Next cycle at %s UTC. Sleeping %dh…",
                next_run.strftime("%Y-%m-%d %H:%M"), _CYCLE_HOURS,
            )
            time.sleep(_CYCLE_HOURS * 3600)
    else:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        logger.info("=== Single cycle: %s  sports=%s ===", ts, sports)
        run_all_sports(
            bankroll=args.bankroll,
            markets=args.market,
            top_n=args.top_n,
            force_retrain=args.retrain,
            sports=sports,
        )


if __name__ == "__main__":
    main()
