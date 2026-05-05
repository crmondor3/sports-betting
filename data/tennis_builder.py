"""
Build / refresh the TennisPlayer database from Jeff Sackmann's CSV data.

Usage (standalone):
    python -m data.tennis_builder          # both tours
    python -m data.tennis_builder --tour atp

Usage (from app):
    from data.tennis_builder import build_tennis_db
    n = build_tennis_db("atp", progress_cb=st.progress)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, date
from io import StringIO
from typing import Callable

import httpx
import pandas as pd

from tracker.database import session_scope, init_db
from tracker.models import TennisPlayer

logger = logging.getLogger(__name__)

SACKMANN_ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
SACKMANN_WTA = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"


# ── CSV download ───────────────────────────────────────────────────────────────

def _fetch_csv(tour: str, year: int) -> pd.DataFrame:
    url_tmpl = SACKMANN_ATP if tour == "atp" else SACKMANN_WTA
    url = url_tmpl.format(year=year)
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        if resp.status_code != 200:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
        if "surface" in df.columns:
            df["surface"] = df["surface"].str.lower().fillna("hard")
        return df
    except Exception as exc:
        logger.warning("Sackmann fetch failed %s %d: %s", tour, year, exc)
        return pd.DataFrame()


def _load_sackmann(tour: str) -> pd.DataFrame:
    """Download current + prior two years and concat (deduplicated)."""
    today = date.today()
    frames: list[pd.DataFrame] = []
    for yr in [today.year, today.year - 1, today.year - 2]:
        df = _fetch_csv(tour, yr)
        if not df.empty:
            df["_year"] = yr
            frames.append(df)
            logger.info("Loaded %d %s matches for %d", len(df), tour.upper(), yr)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Drop exact duplicates if the same file appears twice
    key_cols = [c for c in ["tourney_id", "match_num", "winner_name", "loser_name"]
                if c in combined.columns]
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols)
    return combined


# ── Per-player stat aggregation ────────────────────────────────────────────────

def _safe(val, fallback=None):
    try:
        v = float(val)
        import math
        return v if not math.isnan(v) else fallback
    except (TypeError, ValueError):
        return fallback


def _avg(lst: list) -> float | None:
    clean = [x for x in lst if x is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _compute_player_stats(name: str, df: pd.DataFrame) -> dict:
    won_mask  = df.get("winner_name", pd.Series(dtype=str)) == name
    lost_mask = df.get("loser_name",  pd.Series(dtype=str)) == name
    won_rows  = df[won_mask]
    lost_rows = df[lost_mask]

    if won_rows.empty and lost_rows.empty:
        return {}

    records: list[dict] = []

    def _parse_first_set(score: str | None, player_won: bool) -> bool | None:
        try:
            if not score or str(score).strip() in ("", "nan"):
                return None
            fs = str(score).strip().split()[0].split("(")[0]
            g1, g2 = int(fs.split("-")[0]), int(fs.split("-")[1])
            winner_won_fs = g1 > g2
            return winner_won_fs if player_won else not winner_won_fs
        except Exception:
            return None

    def _srv(row, pfx):
        svpt      = _safe(row.get(f"{pfx}svpt"))
        first_in  = _safe(row.get(f"{pfx}1stIn"))
        first_won = _safe(row.get(f"{pfx}1stWon"))
        sec_won   = _safe(row.get(f"{pfx}2ndWon"))
        aces      = _safe(row.get(f"{pfx}ace"))
        dfs       = _safe(row.get(f"{pfx}df"))
        spw = (((first_won or 0) + (sec_won or 0)) / svpt) if svpt else None
        fsp = (first_in / svpt) if (svpt and first_in is not None) else None
        opp_pfx = "l_" if pfx == "w_" else "w_"
        opp_svpt = _safe(row.get(f"{opp_pfx}svpt"))
        opp_fw   = _safe(row.get(f"{opp_pfx}1stWon"))
        opp_sw   = _safe(row.get(f"{opp_pfx}2ndWon"))
        opp_spw  = (((opp_fw or 0) + (opp_sw or 0)) / opp_svpt) if opp_svpt else None
        rpw = (1.0 - opp_spw) if opp_spw is not None else None
        return spw, fsp, aces, dfs, rpw

    for _, row in won_rows.iterrows():
        spw, fsp, aces, dfs, rpw = _srv(row, "w_")
        records.append({
            "date":     str(row.get("tourney_date", "")),
            "surface":  str(row.get("surface", "hard")).lower(),
            "won":      True,
            "spw":      spw, "fsp": fsp, "aces": aces, "dfs": dfs, "rpw": rpw,
            "won_fs":   _parse_first_set(row.get("score"), True),
            "ranking":  _safe(row.get("winner_rank")),
        })

    for _, row in lost_rows.iterrows():
        spw, fsp, aces, dfs, rpw = _srv(row, "l_")
        records.append({
            "date":     str(row.get("tourney_date", "")),
            "surface":  str(row.get("surface", "hard")).lower(),
            "won":      False,
            "spw":      spw, "fsp": fsp, "aces": aces, "dfs": dfs, "rpw": rpw,
            "won_fs":   _parse_first_set(row.get("score"), False),
            "ranking":  _safe(row.get("loser_rank")),
        })

    records.sort(key=lambda r: r["date"], reverse=True)

    # Overall averages
    wins   = sum(1 for r in records if r["won"])
    losses = len(records) - wins

    # Surface stats
    surf_acc: dict[str, dict] = {}
    for r in records:
        s = r["surface"] or "hard"
        if s not in surf_acc:
            surf_acc[s] = {"wins": 0, "losses": 0, "spw": [], "rpw": []}
        sa = surf_acc[s]
        if r["won"]: sa["wins"] += 1
        else:        sa["losses"] += 1
        if r["spw"] is not None: sa["spw"].append(r["spw"])
        if r["rpw"] is not None: sa["rpw"].append(r["rpw"])
    surface_stats = {
        s: {
            "wins": sa["wins"], "losses": sa["losses"],
            "spw": _avg(sa["spw"]), "rpw": _avg(sa["rpw"]),
        }
        for s, sa in surf_acc.items()
    }

    # Day of week
    day_acc: dict[str, dict] = {}
    for r in records:
        try:
            d = r["date"]
            dt = datetime.strptime(d, "%Y%m%d") if (len(d) == 8 and d.isdigit()) else datetime.fromisoformat(d[:10])
            day = dt.strftime("%A")
        except Exception:
            continue
        if day not in day_acc:
            day_acc[day] = {"wins": 0, "losses": 0}
        if r["won"]: day_acc[day]["wins"] += 1
        else:        day_acc[day]["losses"] += 1

    # First-set stats
    won_fs  = [r for r in records if r["won_fs"] is True]
    lost_fs = [r for r in records if r["won_fs"] is False]
    total_fs = len(won_fs) + len(lost_fs)

    def _wr(recs):
        if not recs: return None
        return round(sum(1 for r in recs if r["won"]) / len(recs), 4)

    first_set_stats = {
        "won_fs_win_pct":  _wr(won_fs),
        "lost_fs_win_pct": _wr(lost_fs),
        "fs_win_rate":     round(len(won_fs) / total_fs, 4) if total_fs else None,
    }

    rankings = [r["ranking"] for r in records[:10] if r["ranking"] is not None]

    return {
        "wins":    wins,
        "losses":  losses,
        "ranking": int(min(rankings)) if rankings else None,
        "avg_spw": _avg([r["spw"] for r in records if r["spw"] is not None]),
        "avg_rpw": _avg([r["rpw"] for r in records if r["rpw"] is not None]),
        "avg_fsp": _avg([r["fsp"] for r in records if r["fsp"] is not None]),
        "avg_aces":_avg([r["aces"] for r in records if r["aces"] is not None]),
        "avg_dfs": _avg([r["dfs"] for r in records if r["dfs"] is not None]),
        "surface_stats":    surface_stats,
        "day_stats":        day_acc,
        "first_set_stats":  first_set_stats,
    }


# ── Main build function ────────────────────────────────────────────────────────

def build_tennis_db(
    tour: str = "atp",
    progress_cb: Callable[[float], None] | None = None,
) -> int:
    """
    Download Sackmann CSVs and upsert all players into the TennisPlayer table.
    Returns count of players written.
    """
    init_db()
    tour = tour.lower()

    logger.info("Downloading Sackmann data for %s...", tour.upper())
    df = _load_sackmann(tour)
    if df.empty:
        logger.warning("No Sackmann data found for %s", tour)
        return 0

    # Collect all unique player names
    all_names: set[str] = set()
    for col in ("winner_name", "loser_name"):
        if col in df.columns:
            all_names.update(df[col].dropna().astype(str).unique())
    all_names = {n for n in all_names if n.strip() and n != "nan"}
    total = len(all_names)
    logger.info("Computing stats for %d %s players...", total, tour.upper())

    count = 0
    for i, name in enumerate(sorted(all_names)):
        stats = _compute_player_stats(name, df)
        if not stats:
            continue

        with session_scope() as s:
            existing = s.query(TennisPlayer).filter_by(name=name, tour=tour).first()
            if existing is None:
                existing = TennisPlayer(name=name, tour=tour)
                s.add(existing)

            existing.ranking             = stats["ranking"]
            existing.wins                = stats["wins"]
            existing.losses              = stats["losses"]
            existing.avg_serve_win_pct   = stats["avg_spw"]
            existing.avg_return_win_pct  = stats["avg_rpw"]
            existing.avg_first_serve_pct = stats["avg_fsp"]
            existing.avg_aces            = stats["avg_aces"]
            existing.avg_dfs             = stats["avg_dfs"]
            existing.surface_stats       = json.dumps(stats["surface_stats"])
            existing.day_stats           = json.dumps(stats["day_stats"])
            existing.first_set_stats     = json.dumps(stats["first_set_stats"])
            existing.last_updated        = datetime.utcnow()
            count += 1

        if progress_cb and total:
            progress_cb((i + 1) / total)

    logger.info("Tennis DB: wrote %d %s players", count, tour.upper())
    return count


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--tour", default="both", choices=["atp", "wta", "both"])
    args = p.parse_args()
    tours = ["atp", "wta"] if args.tour == "both" else [args.tour]
    for t in tours:
        n = build_tennis_db(t)
        print(f"  {t.upper()}: {n} players written")
