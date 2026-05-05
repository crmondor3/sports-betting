"""
Tennis data client — ESPN scoreboard + Jeff Sackmann CSV stats.
No API keys required.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from difflib import get_close_matches
from typing import Any

import httpx
import pandas as pd

from data.daily_cache import load as cache_load, save as cache_save

logger = logging.getLogger(__name__)

ESPN_TENNIS_BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"

SACKMANN_ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
SACKMANN_WTA = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"

# ── Surface inference ──────────────────────────────────────────────────────────

TOURNAMENT_SURFACE: dict[str, str] = {
    # Grand Slams
    "australian open": "hard",
    "roland garros": "clay",
    "french open": "clay",
    "wimbledon": "grass",
    "us open": "hard",
    # ATP Masters 1000
    "indian wells": "hard",
    "miami open": "hard",
    "miami": "hard",
    "monte carlo": "clay",
    "monte-carlo": "clay",
    "madrid open": "clay",
    "madrid": "clay",
    "rome": "clay",
    "italian open": "clay",
    "internazionali": "clay",
    "canada": "hard",
    "canadian open": "hard",
    "montreal": "hard",
    "toronto": "hard",
    "cincinnati": "hard",
    "western & southern": "hard",
    "shanghai": "hard",
    "paris": "hard",
    "paris masters": "hard",
    # ATP 500
    "rotterdam": "hard",
    "dubai": "hard",
    "acapulco": "hard",
    "barcelona": "clay",
    "hamburg": "clay",
    "halle": "grass",
    "queen's club": "grass",
    "queens club": "grass",
    "washington": "hard",
    "tokyo": "hard",
    "beijing": "hard",
    "vienna": "hard",
    "basel": "hard",
    # ATP 250 / common
    "doha": "hard",
    "brisbane": "hard",
    "auckland": "hard",
    "montpellier": "hard",
    "rio": "clay",
    "buenos aires": "clay",
    "marrakech": "clay",
    "estoril": "clay",
    "munich": "clay",
    "geneva": "clay",
    "lyon": "clay",
    "stuttgart": "grass",
    "eastbourne": "grass",
    "newport": "grass",
    "nottingham": "grass",
    "'s-hertogenbosch": "grass",
    "hertogenbosch": "grass",
    "gstaad": "clay",
    "bastad": "clay",
    "umag": "clay",
    "kitzbuhel": "clay",
    "winston-salem": "hard",
    "winston salem": "hard",
    "metz": "hard",
    "sofia": "hard",
    "astana": "hard",
    "almaty": "hard",
    "stockholm": "hard",
    "antwerp": "hard",
    "moscow": "hard",
    "st. petersburg": "hard",
    "adelaide": "hard",
    "dallas": "hard",
    "san diego": "hard",
    "los cabos": "hard",
    "atlanta": "hard",
    "kitzbühel": "clay",
    # WTA specific
    "birmingham": "grass",
    "bad homburg": "grass",
    "berlin": "clay",
    "strasbourg": "clay",
    "prague": "clay",
    "palermo": "clay",
    "lausanne": "clay",
    "san jose": "hard",
    "stanford": "hard",
    "guadalajara": "hard",
    "ostrava": "hard",
    "chicago": "hard",
    "wuhan": "hard",
    "hong kong": "hard",
    "linz": "hard",
    "zhuhai": "hard",
    "shenzhen": "hard",
    # Finals / year-end
    "nitto atp finals": "hard",
    "atp finals": "hard",
    "wta finals": "hard",
    "finals": "hard",
    "united cup": "hard",
    "davis cup": "hard",
    "laver cup": "hard",
}

SURFACE_ADJ = {"grass": 0.04, "clay": -0.03, "hard": 0.0, "carpet": 0.02}


def infer_surface(tournament_name: str) -> str:
    """Infer surface from tournament name using keyword lookup."""
    name_lower = tournament_name.lower()
    for keyword, surface in TOURNAMENT_SURFACE.items():
        if keyword in name_lower:
            return surface
    # Fallback heuristics
    if any(w in name_lower for w in ["open", "masters", "cup"]):
        return "hard"
    return "hard"


# ── Name normalisation ─────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip()))


def _fuzzy_match_name(target: str, candidates: list[str], cutoff: float = 0.6) -> str | None:
    """Return best fuzzy match from candidates list, or None."""
    if not candidates:
        return None
    target_norm = _norm(target)
    cand_norm = {_norm(c): c for c in candidates}
    # Exact match first
    if target_norm in cand_norm:
        return cand_norm[target_norm]
    matches = get_close_matches(target_norm, list(cand_norm.keys()), n=1, cutoff=cutoff)
    if matches:
        return cand_norm[matches[0]]
    # Last-name only match
    last = target_norm.split()[-1] if target_norm else ""
    for norm_c, orig_c in cand_norm.items():
        if last and norm_c.split()[-1] == last:
            return orig_c
    return None


# ── Safe numeric helper ────────────────────────────────────────────────────────

def _safe_div(num: float, denom: float, fallback: float = 0.0) -> float:
    return round(num / denom, 4) if denom and denom > 0 else fallback


# ── Main client ────────────────────────────────────────────────────────────────

class TennisDataClient:
    def __init__(self):
        self._http = httpx.Client(timeout=20, follow_redirects=True)

    def _get(self, url: str, params: dict | None = None) -> Any:
        r = self._http.get(url, params=params or {})
        r.raise_for_status()
        return r.json()

    # ── ESPN scoreboard ────────────────────────────────────────────────────────

    def get_upcoming_matches(self, tour: str = "atp") -> list[dict]:
        """
        Fetch ESPN tennis scoreboard for `tour` ('atp' or 'wta').
        Returns list of match dicts.
        """
        tour_key = tour.lower()
        cache_key = f"espn_tennis_{tour_key}"
        cached = cache_load(cache_key)
        if cached is not None:
            return cached

        url = f"{ESPN_TENNIS_BASE}/{tour_key}/scoreboard"
        try:
            data = self._get(url)
        except Exception as exc:
            logger.warning("ESPN tennis fetch failed (%s): %s", tour_key, exc)
            return []

        matches: list[dict] = []
        events = data.get("events", [])

        for ev in events:
            try:
                tournament_name = (
                    ev.get("name", "")
                    or ev.get("season", {}).get("displayName", "")
                    or ev.get("shortName", "")
                )
                # ESPN tennis: competitions live inside groupings, not on the event directly.
                # Structure: event -> groupings[].competitions[]
                competitions: list[dict] = ev.get("competitions", [])
                for grp in ev.get("groupings", []):
                    competitions = competitions + grp.get("competitions", [])
                    # Some nested: groupings[].events[].competitions[]
                    for inner_ev in grp.get("events", []):
                        competitions = competitions + inner_ev.get("competitions", [])

                for comp in competitions:
                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue

                    p1 = competitors[0]
                    p2 = competitors[1]

                    def _player_name(c: dict) -> str:
                        athlete = c.get("athlete", {})
                        if not athlete:
                            athlete = c.get("team", {})
                        return (
                            athlete.get("displayName", "")
                            or athlete.get("fullName", "")
                            or athlete.get("shortName", "")
                        )

                    def _player_id(c: dict) -> str:
                        athlete = c.get("athlete", {})
                        if not athlete:
                            athlete = c.get("team", {})
                        return str(athlete.get("id", ""))

                    p1_name = _player_name(p1)
                    p2_name = _player_name(p2)
                    if not p1_name or not p2_name:
                        continue

                    round_name = (
                        comp.get("round", {}).get("displayName", "")
                        or comp.get("status", {}).get("type", {}).get("description", "")
                    )
                    is_completed = comp.get("status", {}).get("type", {}).get("completed", False)
                    scheduled_time = comp.get("date", ev.get("date", ""))
                    venue = comp.get("venue", {})
                    # ESPN tennis venues use fullName like "Rome, Italy" rather than city field
                    venue_city = (
                        venue.get("city", "")
                        or venue.get("fullName", "")
                        if venue else ""
                    )

                    surface = infer_surface(tournament_name)

                    matches.append({
                        "match_id": comp.get("id", ""),
                        "player1_name": p1_name,
                        "player2_name": p2_name,
                        "player1_id": _player_id(p1),
                        "player2_id": _player_id(p2),
                        "tour": tour_key.upper(),
                        "tournament": tournament_name,
                        "surface": surface,
                        "round_name": round_name,
                        "scheduled_time": scheduled_time,
                        "is_completed": bool(is_completed),
                        "venue_city": venue_city,
                    })
            except Exception as exc:
                logger.debug("Skipping event: %s", exc)
                continue

        cache_save(cache_key, matches)
        return matches

    # ── Sackmann CSV ───────────────────────────────────────────────────────────

    def get_player_df(self, tour: str = "atp") -> pd.DataFrame:
        """
        Download and cache Sackmann match CSV for `tour`.
        Uses current year; falls back to prior year if current year is empty.
        Returns a DataFrame (empty if unavailable).
        """
        tour_key = tour.lower()
        today = date.today()
        year = today.year

        for yr in [year, year - 1, year - 2]:
            cache_key = f"sackmann_{tour_key}_{yr}"
            cached = cache_load(cache_key)
            if cached is not None:
                if cached:  # non-empty list
                    return pd.DataFrame(cached)
                continue  # cached empty — try prior year

            url_template = SACKMANN_ATP if tour_key == "atp" else SACKMANN_WTA
            url = url_template.format(year=yr)
            try:
                resp = self._http.get(url)
                if resp.status_code != 200:
                    logger.warning("Sackmann CSV %s returned %d", url, resp.status_code)
                    cache_save(cache_key, [])
                    continue
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text), low_memory=False)
                if df.empty:
                    cache_save(cache_key, [])
                    continue
                # Normalise surface casing
                if "surface" in df.columns:
                    df["surface"] = df["surface"].str.lower().fillna("hard")
                # Persist as records
                records = df.to_dict(orient="records")
                cache_save(cache_key, records)
                return df
            except Exception as exc:
                logger.warning("Sackmann fetch failed (%s %d): %s", tour_key, yr, exc)
                cache_save(cache_key, [])

        return pd.DataFrame()

    # ── Player stats ───────────────────────────────────────────────────────────

    def get_player_stats(self, player_name: str, df: pd.DataFrame) -> dict:
        """
        Compute stats for `player_name` from the Sackmann DataFrame.
        Returns a structured dict.
        """
        empty = {
            "recent_matches": [],
            "surface_record": {},
            "day_record": {},
            "first_set_stats": {
                "won_first_set_win_pct": None,
                "lost_first_set_win_pct": None,
                "first_set_win_rate": None,
            },
            "overall": {
                "wins": 0, "losses": 0,
                "avg_serve_win_pct": None,
                "avg_return_win_pct": None,
                "avg_first_serve_pct": None,
                "avg_aces": None,
                "avg_dfs": None,
                "ranking": None,
            },
            "data_quality": "no_data",
        }

        if df.empty:
            return empty

        # Build unique player names from both winner and loser columns
        all_names: list[str] = []
        for col in ("winner_name", "loser_name"):
            if col in df.columns:
                all_names.extend(df[col].dropna().unique().tolist())
        all_names = list(set(str(n) for n in all_names))

        matched = _fuzzy_match_name(player_name, all_names)
        if matched is None:
            return empty

        # Rows where player won
        won_mask = df.get("winner_name", pd.Series(dtype=str)) == matched
        lost_mask = df.get("loser_name", pd.Series(dtype=str)) == matched

        won_rows = df[won_mask].copy()
        lost_rows = df[lost_mask].copy()

        if won_rows.empty and lost_rows.empty:
            return empty

        # ── Build unified match list ───────────────────────────────────────────

        def _parse_score_first_set(score: str | None) -> bool | None:
            """Return True if this player won the first set. score is like '6-3 7-5'."""
            if not score or pd.isna(score):
                return None
            try:
                first_set = str(score).strip().split()[0]
                parts = first_set.replace("(", "").split("(")[0].split("-")
                if len(parts) < 2:
                    return None
                g1, g2 = int(parts[0]), int(parts[1])
                return g1 > g2
            except Exception:
                return None

        def _extract_serve_stats(row: pd.Series, prefix: str) -> dict:
            """prefix is 'w_' or 'l_'."""
            svpt = _safe_num(row.get(f"{prefix}svpt"))
            first_in = _safe_num(row.get(f"{prefix}1stIn"))
            first_won = _safe_num(row.get(f"{prefix}1stWon"))
            second_won = _safe_num(row.get(f"{prefix}2ndWon"))
            aces = _safe_num(row.get(f"{prefix}ace"))
            dfs = _safe_num(row.get(f"{prefix}df"))

            second_svpt = svpt - first_in if svpt and first_in else 0

            serve_win_pct = _safe_div(
                (first_won or 0) + (second_won or 0), svpt or 0
            ) if svpt else None

            first_serve_pct = _safe_div(first_in or 0, svpt or 0) if svpt else None

            return {
                "serve_win_pct": serve_win_pct,
                "first_serve_pct": first_serve_pct,
                "aces": aces,
                "dfs": dfs,
            }

        def _extract_return_stats(row: pd.Series, player_prefix: str, opp_prefix: str) -> float | None:
            """Return points won pct = 1 - opp serve win pct."""
            svpt = _safe_num(row.get(f"{opp_prefix}svpt"))
            opp_first_won = _safe_num(row.get(f"{opp_prefix}1stWon"))
            opp_second_won = _safe_num(row.get(f"{opp_prefix}2ndWon"))
            if not svpt:
                return None
            opp_serve_won = (opp_first_won or 0) + (opp_second_won or 0)
            return round(1.0 - _safe_div(opp_serve_won, svpt), 4)

        records: list[dict] = []

        for _, row in won_rows.iterrows():
            score = str(row.get("score", "")) if not pd.isna(row.get("score", "")) else ""
            won_first = _parse_score_first_set(score)
            srv = _extract_serve_stats(row, "w_")
            rtn = _extract_return_stats(row, "w_", "l_")
            surface = str(row.get("surface", "hard")).lower()
            records.append({
                "date": str(row.get("tourney_date", "")),
                "tournament": str(row.get("tourney_name", "")),
                "surface": surface,
                "opponent": str(row.get("loser_name", "")),
                "won": True,
                "score": score,
                "serve_win_pct": srv["serve_win_pct"],
                "return_win_pct": rtn,
                "first_serve_pct": srv["first_serve_pct"],
                "aces": srv["aces"],
                "dfs": srv["dfs"],
                "won_first_set": won_first,
                "ranking": _safe_num(row.get("winner_rank")),
            })

        for _, row in lost_rows.iterrows():
            score = str(row.get("score", "")) if not pd.isna(row.get("score", "")) else ""
            won_first = _parse_score_first_set(score)
            # Loser's first set win = opponent (winner) lost first set → invert
            if won_first is not None:
                won_first = not won_first
            srv = _extract_serve_stats(row, "l_")
            rtn = _extract_return_stats(row, "l_", "w_")
            surface = str(row.get("surface", "hard")).lower()
            records.append({
                "date": str(row.get("tourney_date", "")),
                "tournament": str(row.get("tourney_name", "")),
                "surface": surface,
                "opponent": str(row.get("winner_name", "")),
                "won": False,
                "score": score,
                "serve_win_pct": srv["serve_win_pct"],
                "return_win_pct": rtn,
                "first_serve_pct": srv["first_serve_pct"],
                "aces": srv["aces"],
                "dfs": srv["dfs"],
                "won_first_set": won_first,
                "ranking": _safe_num(row.get("loser_rank")),
            })

        # Sort by date descending
        records.sort(key=lambda r: r["date"], reverse=True)

        recent_matches = records[:10]

        # ── Surface record ─────────────────────────────────────────────────────
        surface_record: dict[str, dict] = {}
        for rec in records:
            s = rec["surface"] or "hard"
            if s not in surface_record:
                surface_record[s] = {
                    "wins": 0, "losses": 0,
                    "serve_win_pcts": [], "return_win_pcts": [],
                }
            sr = surface_record[s]
            if rec["won"]:
                sr["wins"] += 1
            else:
                sr["losses"] += 1
            if rec["serve_win_pct"] is not None:
                sr["serve_win_pcts"].append(rec["serve_win_pct"])
            if rec["return_win_pct"] is not None:
                sr["return_win_pcts"].append(rec["return_win_pct"])

        surface_out: dict[str, dict] = {}
        for surf, sr in surface_record.items():
            surface_out[surf] = {
                "wins": sr["wins"],
                "losses": sr["losses"],
                "serve_win_pct": _avg(sr["serve_win_pcts"]),
                "return_win_pct": _avg(sr["return_win_pcts"]),
            }

        # ── Day of week record ─────────────────────────────────────────────────
        day_record: dict[str, dict] = {}
        for rec in records:
            try:
                d_str = str(rec["date"])
                # Sackmann date format: YYYYMMDD
                if len(d_str) == 8 and d_str.isdigit():
                    dt = datetime.strptime(d_str, "%Y%m%d")
                else:
                    dt = datetime.fromisoformat(d_str[:10])
                day_name = dt.strftime("%A")
            except Exception:
                continue
            if day_name not in day_record:
                day_record[day_name] = {"wins": 0, "losses": 0}
            if rec["won"]:
                day_record[day_name]["wins"] += 1
            else:
                day_record[day_name]["losses"] += 1

        # ── First set stats ────────────────────────────────────────────────────
        won_first_set = [r for r in records if r["won_first_set"] is True]
        lost_first_set = [r for r in records if r["won_first_set"] is False]

        def _win_rate(recs: list[dict]) -> float | None:
            if not recs:
                return None
            return round(sum(1 for r in recs if r["won"]) / len(recs), 4)

        first_set_win_rate = _win_rate(won_first_set + lost_first_set)
        if first_set_win_rate is not None and (won_first_set or lost_first_set):
            total_with_fs = len(won_first_set) + len(lost_first_set)
            first_set_win_rate = round(len(won_first_set) / total_with_fs, 4) if total_with_fs else None

        first_set_stats = {
            "won_first_set_win_pct": _win_rate(won_first_set),
            "lost_first_set_win_pct": _win_rate(lost_first_set),
            "first_set_win_rate": first_set_win_rate,
        }

        # ── Overall stats ──────────────────────────────────────────────────────
        serve_pcts = [r["serve_win_pct"] for r in records if r["serve_win_pct"] is not None]
        return_pcts = [r["return_win_pct"] for r in records if r["return_win_pct"] is not None]
        fs_pcts = [r["first_serve_pct"] for r in records if r["first_serve_pct"] is not None]
        aces_list = [r["aces"] for r in records if r["aces"] is not None]
        dfs_list = [r["dfs"] for r in records if r["dfs"] is not None]
        rankings = [r["ranking"] for r in records[:5] if r["ranking"] is not None]

        total_wins = sum(1 for r in records if r["won"])
        total_losses = len(records) - total_wins

        overall = {
            "wins": total_wins,
            "losses": total_losses,
            "avg_serve_win_pct": _avg(serve_pcts),
            "avg_return_win_pct": _avg(return_pcts),
            "avg_first_serve_pct": _avg(fs_pcts),
            "avg_aces": _avg(aces_list),
            "avg_dfs": _avg(dfs_list),
            "ranking": int(min(rankings)) if rankings else None,
        }

        data_quality = "no_data"
        if len(records) >= 10:
            data_quality = "full"
        elif len(records) > 0:
            data_quality = "partial"

        return {
            "recent_matches": recent_matches,
            "surface_record": surface_out,
            "day_record": day_record,
            "first_set_stats": first_set_stats,
            "overall": overall,
            "data_quality": data_quality,
            "matched_name": matched,
        }

    def close(self):
        self._http.close()


# ── Private helpers ────────────────────────────────────────────────────────────

def _safe_num(val: Any) -> float | None:
    try:
        v = float(val)
        return v if not pd.isna(v) else None
    except (TypeError, ValueError):
        return None


def _avg(lst: list[float]) -> float | None:
    if not lst:
        return None
    return round(sum(lst) / len(lst), 4)
