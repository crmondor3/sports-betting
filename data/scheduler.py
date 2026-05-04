"""APScheduler jobs — odds refresh, daily pipeline, CLV settlement."""
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
    """Refresh odds every N seconds (default 5 min)."""
    sched = get_scheduler()
    sched.add_job(
        callback,
        trigger=IntervalTrigger(seconds=interval_seconds),
        id="odds_refresh",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    logger.info("Odds refresh job scheduled every %ds", interval_seconds)


def add_daily_summary_job(callback: Callable, hour: int = 8, minute: int = 0):
    """Run the daily summary alert at a fixed time each morning."""
    sched = get_scheduler()
    sched.add_job(
        callback,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="daily_summary",
        replace_existing=True,
    )
    logger.info("Daily summary job scheduled at %02d:%02d ET", hour, minute)


def add_clv_settlement_job(callback: Callable, hour: int = 23, minute: int = 55):
    """Settle CLV by comparing pre-game lines to closing lines each night."""
    sched = get_scheduler()
    sched.add_job(
        callback,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="clv_settlement",
        replace_existing=True,
    )
    logger.info("CLV settlement job scheduled at %02d:%02d ET", hour, minute)


def start():
    sched = get_scheduler()
    if not sched.running:
        sched.start()
        logger.info("Scheduler started")


def stop():
    sched = get_scheduler()
    if sched.running:
        sched.shutdown(wait=False)
        logger.info("Scheduler stopped")
