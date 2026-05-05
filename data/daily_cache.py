"""
File-based daily cache.
Data is fetched once per cache slot and stored as JSON in cache/.
The cache slot advances at 07:00 America/New_York each day — so a fresh fetch
happens automatically every morning at 7 AM ET regardless of server timezone.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_REFRESH_HOUR_ET = 7  # 07:00 America/New_York


def _cache_slot() -> str:
    """
    Returns the active cache-slot date string (YYYY-MM-DD).
    The slot flips forward at 07:00 ET — before that hour we still use the
    previous day's slot so overnight data stays valid until morning.
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.hour >= _REFRESH_HOUR_ET:
        return now_et.strftime("%Y-%m-%d")
    return (now_et - timedelta(days=1)).strftime("%Y-%m-%d")


def _path(key: str, for_date: date | None = None) -> Path:
    d = for_date.isoformat() if for_date else _cache_slot()
    return CACHE_DIR / f"{d}_{key}.json"


def load(key: str, for_date: date | None = None) -> Any | None:
    """Return cached data for today, or None if not cached yet."""
    p = _path(key, for_date)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            logger.info("Cache HIT  — %s", p.name)
            return data
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", p.name, exc)
    return None


def save(key: str, data: Any, for_date: date | None = None) -> None:
    """Persist data to today's cache file."""
    p = _path(key, for_date)
    try:
        p.write_text(json.dumps(data, default=str), encoding="utf-8")
        logger.info("Cache WRITE — %s", p.name)
    except Exception as exc:
        logger.warning("Cache write failed for %s: %s", p.name, exc)


def bust(key: str, for_date: date | None = None) -> None:
    """Delete a specific cache entry (force refresh on next load)."""
    p = _path(key, for_date)
    if p.exists():
        p.unlink()
        logger.info("Cache BUSTED — %s", p.name)


def bust_all() -> int:
    """Delete all cache files. Returns count deleted."""
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    logger.info("Busted %d cache files", count)
    return count


def is_cached(key: str, for_date: date | None = None) -> bool:
    return _path(key, for_date).exists()
