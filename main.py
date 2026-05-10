"""
main.py — ML sports betting pipeline CLI.

Usage examples:
  python main.py --sport NFL --train
  python main.py --sport NBA --bankroll 2000 --min-edge 0.04
  python main.py --sport NFL --backtest --seasons 3
  python main.py --sport NFL --market h2h spreads --kelly-fraction 0.2
  python main.py --sport NFL --date 2024-11-10
"""
from __future__ import annotations

import argparse
import logging
import sys

# Guard: prevent Streamlit from executing the CLI pipeline.
# Streamlit sets __name__ == "__main__" when it runs any file, so the
# normal `if __name__ == "__main__"` guard is not enough.
# If this file is being run by `streamlit run`, exit immediately with
# a clear message pointing to the correct entry point.
if any("streamlit" in arg for arg in sys.argv):
    print(
        "\n[ERROR] Streamlit is running main.py instead of app.py.\n"
        "Fix: go to your Streamlit Cloud app → Settings → "
        "change 'Main file path' to app.py\n",
        file=sys.stderr,
    )
    sys.exit(1)
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sports_betting.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main_ml")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="ML sports betting model — DraftKings value bet finder",
    )
    p.add_argument("--sport", default="NFL",
                   help="Sport to analyse (NFL, NBA, MLB, NHL, WNBA). Default: NFL")
    p.add_argument("--bankroll", type=float, default=1000.0,
                   help="Current bankroll in dollars. Default: 1000")
    p.add_argument("--kelly-fraction", type=float, default=0.25,
                   help="Fractional Kelly multiplier (0–1). Default: 0.25")
    p.add_argument("--min-edge", type=float, default=0.03,
                   help="Minimum model edge vs DK implied prob to flag a bet (fraction). Default: 0.03")
    p.add_argument("--market", nargs="+", choices=["h2h", "spreads", "totals"],
                   default=["h2h"],
                   help="Markets to analyse. Default: h2h")
    p.add_argument("--date",
                   help="Target date YYYY-MM-DD. Default: today's upcoming games")
    p.add_argument("--train", action="store_true",
                   help="Force re-train ML models before generating picks")
    p.add_argument("--backtest", action="store_true",
                   help="Run historical backtest instead of live picks")
    p.add_argument("--seasons", type=int, default=3,
                   help="Number of seasons of historical data to fetch/use. Default: 3")
    p.add_argument("--no-optuna", action="store_true",
                   help="Skip Optuna hyperparameter tuning (faster, less accurate)")
    p.add_argument("--top-n", type=int, default=5,
                   help="Number of top bets to print. Default: 5")
    p.add_argument("--refresh", action="store_true",
                   help="Force-refresh today's odds cache")
    return p


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_collect(sport: str, refresh: bool = False) -> list[dict]:
    """Fetch historical ESPN game data and store DK odds snapshots."""
    from data_collector import DataCollector
    dc = DataCollector()

    logger.info("Collecting historical games for %s…", sport)
    games = dc.fetch_historical_games(sport)

    logger.info("Fetching DraftKings odds snapshot for %s…", sport)
    try:
        dc.fetch_and_store_odds(sport, force_refresh=refresh)
    except Exception as exc:
        logger.warning("Could not fetch live odds: %s", exc)

    return games


def step_train(
    sport: str,
    games: list[dict],
    markets: list[str],
    use_optuna: bool = True,
) -> None:
    """Build feature matrix and train ML models."""
    from feature_engineer import FeatureEngineer
    from model_trainer import ModelTrainer

    logger.info("Building feature matrix for %s (%d games)…", sport, len(games))
    fe = FeatureEngineer(sport=sport)
    df = fe.build(games)

    if df.empty:
        logger.error("Feature matrix is empty — cannot train.")
        return

    # Attach spread/totals targets when odds columns are available
    if "dk_home_spread" in df.columns and "spreads" in markets:
        df["spread_win"] = fe.make_spread_target(df)
    if "dk_total" in df.columns and "totals" in markets:
        df["over_hit"] = fe.make_totals_target(df)

    trainer = ModelTrainer(sport=sport)
    for market in markets:
        logger.info("Training %s model for %s…", market, sport)
        metrics = trainer.fit(df, market=market, use_optuna=use_optuna)
        if "error" not in metrics:
            logger.info(
                "%s %s — AUC xgb=%.3f lgb=%.3f lr=%.3f",
                sport, market,
                metrics.get("xgboost_auc", 0),
                metrics.get("lightgbm_auc", 0),
                metrics.get("logreg_auc", 0),
            )


def step_picks(
    sport: str,
    games: list[dict],
    bankroll: float,
    kelly_fraction: float,
    min_edge: float,
    markets: list[str],
    top_n: int,
    refresh: bool,
) -> None:
    """Fetch live odds, generate picks, print and save."""
    import config
    from data.odds_api import OddsAPIClient
    from bet_selector import BetSelector

    sport_key = config.SUPPORTED_SPORTS.get(sport.upper())
    if not sport_key:
        logger.error("Unknown sport '%s'", sport)
        return

    with OddsAPIClient() as client:
        odds_games = client.get_odds(sport_key, force_refresh=refresh)

    if not odds_games:
        logger.warning("No upcoming %s games with DraftKings odds.", sport)
        return

    selector = BetSelector(
        sport=sport,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        min_edge=min_edge,
        markets=markets,
    )

    logger.info("Analysing %d upcoming games…", len(odds_games))
    bets = selector.analyze(odds_games, games)

    selector.print_top(bets, n=top_n)
    selector.save_csv(bets)

    logger.info(
        "Found %d qualifying value bets (edge > %.0f%%).",
        len([b for b in bets if b.ev > 0]),
        min_edge * 100,
    )


def step_backtest(
    sport: str,
    games: list[dict],
    bankroll: float,
    kelly_fraction: float,
    min_edge: float,
    markets: list[str],
    use_optuna: bool,
) -> None:
    """Run historical backtest and save reports."""
    from backtester import Backtester

    backtester = Backtester(
        sport=sport,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        min_edge=min_edge,
        markets=markets,
    )

    logger.info("Running backtest on %d games…", len(games))
    reports = backtester.run(games, use_optuna=use_optuna)
    backtester.save_reports(reports)

    for r in reports:
        print(r)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    sport         = args.sport.upper()
    markets       = args.market
    use_optuna    = not args.no_optuna

    logger.info(
        "Starting ML pipeline — sport=%s markets=%s bankroll=%.0f kelly=%.2f min_edge=%.1f%%",
        sport, markets, args.bankroll, args.kelly_fraction, args.min_edge * 100,
    )

    # 1. Collect / refresh data
    games = step_collect(sport, refresh=args.refresh)
    if not games:
        logger.error("No historical game data available. Exiting.")
        sys.exit(1)

    # 2. Optionally train (or check if models need training)
    from model_trainer import ModelTrainer
    trainer = ModelTrainer(sport=sport)
    needs_train = args.train or not trainer.is_trained("h2h")
    if needs_train:
        step_train(sport, games, markets, use_optuna=use_optuna)

    # 3. Backtest OR live picks
    if args.backtest:
        step_backtest(sport, games, args.bankroll, args.kelly_fraction,
                      args.min_edge, markets, use_optuna)
    else:
        step_picks(sport, games, args.bankroll, args.kelly_fraction,
                   args.min_edge, markets, args.top_n, args.refresh)


if __name__ == "__main__":
    main()
