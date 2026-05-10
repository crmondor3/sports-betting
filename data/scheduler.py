"""APScheduler jobs — odds refresh, daily pipeline, CLV settlement, auto ML learning."""
from __future__ import annotations

import logging
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import config

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone="America/New_York")
    return _scheduler


def add_odds_refresh_job(callback: Callable, interval_seconds: int = config.ODDS_REFRESH_INTERVAL):
    sched = get_scheduler()
    sched.add_job(
        callback,
        trigger=IntervalTrigger(seconds=interval_seconds),
        id="odds_refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )


def add_daily_summary_job(callback: Callable, hour: int = 8, minute: int = 0):
    sched = get_scheduler()
    sched.add_job(
        callback,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="daily_summary",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )


def add_daily_picks_alert_job(hour: int = 8, minute: int = 30, ev_min: float = 15.0) -> None:
    """
    Daily 8:30 AM ET job: run the picks pipeline and send qualifying picks
    via Telegram. Silently skips if Telegram is not configured.
    """
    def _alert() -> None:
        try:
            from alerts.telegram_alert import TelegramAlert
            bot = TelegramAlert()
            if not bot.enabled:
                return
            logger.info("Daily picks alert starting…")
            import config as cfg
            cfg.EV_THRESHOLD  = -1.0
            cfg.MIN_CONFIDENCE = 0
            cfg.MAX_PICKS_PER_DAY = 500
            from run import run_pipeline
            from tracker.bankroll import BankrollTracker
            picks, _, _ = run_pipeline()
            picks_dicts = [p.to_dict() for p in picks]
            bankroll = BankrollTracker().get_current_bankroll()
            sent = bot.send_picks(picks_dicts, bankroll, ev_min=ev_min)
            logger.info("Daily picks alert sent=%s (%d picks)", sent, len(picks_dicts))
        except Exception as exc:
            logger.error("Daily picks alert failed: %s", exc)

    sched = get_scheduler()
    sched.add_job(
        _alert,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="daily_picks_alert",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    logger.info("Daily picks alert scheduled at %02d:%02d ET", hour, minute)


def add_clv_settlement_job(callback: Callable, hour: int = 23, minute: int = 55):
    sched = get_scheduler()
    sched.add_job(
        callback,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="clv_settlement",
        replace_existing=True,
    )


def add_ml_update_job(hour: int = 3, minute: int = 0) -> None:
    """
    Daily nightly job: fetch new ESPN results → rebuild calibrator.
    Runs at 3 AM ET by default so fresh game results from the evening are included.
    """
    def _ml_update() -> None:
        try:
            logger.info("Nightly ML update starting…")
            from data.historical_backfill import run_backfill
            n = run_backfill(days=7)
            logger.info("Backfill added %d new rows", n)
            from models.calibrator import ModelCalibrator
            ModelCalibrator.rebuild_from_db()
            logger.info("Nightly ML update complete")
        except Exception as exc:
            logger.error("Nightly ML update failed: %s", exc)

    sched = get_scheduler()
    sched.add_job(
        _ml_update,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="ml_update",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    logger.info("Nightly ML update scheduled at %02d:%02d ET", hour, minute)


def add_weekly_retrain_job(day_of_week: int = 6, hour: int = 4, minute: int = 0) -> None:
    """
    Weekly job (Sunday 4 AM ET by default): retrain GB model on fresh ESPN data
    then rebuild the calibrator with the new predictions.
    day_of_week: 0=Mon … 6=Sun
    """
    def _weekly_retrain() -> None:
        try:
            logger.info("Weekly GB retrain starting…")
            from models.trainer import train_all
            results = train_all(years_back=3)
            ok = [r for r in results if "error" not in r]
            logger.info("Weekly retrain: %d models trained", len(ok))
            from models.calibrator import ModelCalibrator
            ModelCalibrator.rebuild_from_db()
            logger.info("Weekly retrain complete")
        except Exception as exc:
            logger.error("Weekly retrain failed: %s", exc)

    sched = get_scheduler()
    sched.add_job(
        _weekly_retrain,
        trigger=CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
        id="weekly_retrain",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    logger.info("Weekly retrain scheduled day_of_week=%d at %02d:%02d ET", day_of_week, hour, minute)


def start() -> None:
    sched = get_scheduler()
    if not sched.running:
        sched.start()
        logger.info("Scheduler started")


def stop() -> None:
    sched = get_scheduler()
    if sched.running:
        sched.shutdown(wait=False)
        logger.info("Scheduler stopped")
