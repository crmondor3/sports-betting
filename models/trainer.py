"""
Train a gradient-boosting model on multi-season ESPN game history.

Architecture:
  1. Process games chronologically, maintaining per-team rolling state
     (Elo ratings, recent form, rest days, PPG differentials).
  2. For each game, compute a 15-feature vector using ONLY data available
     BEFORE that game (no leakage).
  3. Train GradientBoostingClassifier + isotonic calibration on 80% of data.
  4. Evaluate on held-out 20%.
  5. Save trained bundle (model, scaler, final Elo ratings) to cache/.

At prediction time, MatchupAnalyzer feeds TeamForm / HeadToHead data into
the same feature vector and calls model.predict_proba().
"""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_DIR         = Path(__file__).parent.parent / "cache"
_MIN_GAMES         = 200   # need this many games to bother training
DEFAULT_YEARS_BACK = 3
_ELO_K      = 20.0
_ELO_HOME   = 50.0  # default home advantage in Elo points (overridden per sport)
_SPORT_K    = {"NFL": 28, "NBA": 20, "MLB": 10, "NHL": 16}
_SPORT_HA   = {"NFL": 70, "NBA": 35, "MLB": 18, "NHL": 28}

FEATURE_NAMES = [
    "elo_diff",           # home_elo + home_adv - away_elo  (pre-game)
    "home_rest",          # days since last game, capped 7
    "away_rest",
    "rest_advantage",     # home_rest - away_rest
    "home_b2b",           # 1 if back-to-back
    "away_b2b",
    "home_l10_wp",        # win% last 10 games
    "away_l10_wp",
    "form_advantage",     # home_l10_wp - away_l10_wp
    "home_ppg_diff",      # avg (scored - allowed) last 15 games
    "away_ppg_diff",
    "ppg_diff_advantage", # home_ppg_diff - away_ppg_diff
    "home_ha_wp",         # home team's home win% (last 30 home games)
    "away_ha_wp",         # away team's away win% (last 30 away games)
    "h2h_advantage",      # home h2h win% (last 5 meetings) - 0.5
]


def _model_path(sport: str) -> Path:
    return _CACHE_DIR / f"trained_{sport.lower()}.pkl"


class _Calibrated:
    """Wraps a GBM + IsotonicRegression into a predict_proba interface. Module-level so pickle works."""
    def __init__(self, gb, ir):
        self.gb = gb
        self.ir = ir

    def predict_proba(self, X):
        raw = self.gb.predict_proba(X)[:, 1]
        cal = self.ir.predict(raw)
        return np.column_stack([1 - cal, cal])


# ── Feature computation ────────────────────────────────────────────────────────

def _build_features(games: list[dict], sport: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Process game list chronologically, compute features using only past data.
    Returns (X, y, final_elo_ratings).
    """
    k        = _SPORT_K.get(sport.upper(), _ELO_K)
    home_adv = _SPORT_HA.get(sport.upper(), _ELO_HOME)

    elo: dict[str, float]      = defaultdict(lambda: 1500.0)
    # team_hist[team] = list of {dt, won, is_home, scored, allowed, opponent}
    team_hist: dict[str, list] = defaultdict(list)

    X_rows, y_rows = [], []

    for g in games:
        try:
            dt = datetime.fromisoformat(g["date"].rstrip("Z"))
        except Exception:
            continue

        home   = g["home_team"]
        away   = g["away_team"]
        h_won  = bool(g.get("home_winner", False))
        h_pts  = float(g.get("home_score", 0))
        a_pts  = float(g.get("away_score", 0))

        hh = team_hist[home]
        ah = team_hist[away]

        # ── Elo diff ──────────────────────────────────────────────────
        h_elo = elo[home]
        a_elo = elo[away]
        elo_diff = h_elo + home_adv - a_elo

        # ── Rest ──────────────────────────────────────────────────────
        def _rest(hist: list) -> int:
            prev = [x for x in hist if x["dt"] < dt]
            if not prev:
                return 3
            return min(int((dt - max(x["dt"] for x in prev)).total_seconds() / 86400), 7)

        h_rest = _rest(hh)
        a_rest = _rest(ah)

        # ── Recent form ───────────────────────────────────────────────
        def _l10(hist: list) -> float:
            recent = [x for x in hist if x["dt"] < dt][-10:]
            return sum(x["won"] for x in recent) / len(recent) if recent else 0.5

        h_l10 = _l10(hh)
        a_l10 = _l10(ah)

        # ── PPG differential ─────────────────────────────────────────
        def _ppg_diff(hist: list) -> float:
            recent = [x for x in hist if x["dt"] < dt][-15:]
            if not recent:
                return 0.0
            return sum(x["scored"] - x["allowed"] for x in recent) / len(recent)

        h_ppg = _ppg_diff(hh)
        a_ppg = _ppg_diff(ah)

        # ── Home/away splits ─────────────────────────────────────────
        def _ha_wp(hist: list, is_home: bool) -> float:
            recent = [x for x in hist if x["dt"] < dt and x["is_home"] == is_home][-30:]
            return sum(x["won"] for x in recent) / len(recent) if recent else 0.5

        h_home_wp = _ha_wp(hh, True)
        a_away_wp = _ha_wp(ah, False)

        # ── H2H ──────────────────────────────────────────────────────
        h2h = [x for x in hh if x["dt"] < dt and x["opponent"] == away][-5:]
        h2h_adv = (sum(x["won"] for x in h2h) / len(h2h) - 0.5) if h2h else 0.0

        feat = [
            elo_diff,
            h_rest,         a_rest,
            h_rest - a_rest,
            1 if h_rest <= 1 else 0,
            1 if a_rest <= 1 else 0,
            h_l10,          a_l10,
            h_l10 - a_l10,
            h_ppg,          a_ppg,
            h_ppg - a_ppg,
            h_home_wp,      a_away_wp,
            h2h_adv,
        ]

        X_rows.append(feat)
        y_rows.append(1 if h_won else 0)

        # ── Update state AFTER recording features ─────────────────────
        e_h = 1 / (1 + 10 ** ((a_elo - h_elo - home_adv) / 400))
        elo[home] = h_elo + k * ((1 if h_won else 0) - e_h)
        elo[away] = a_elo + k * ((0 if h_won else 1) - (1 - e_h))

        hh.append({"dt": dt, "won": h_won,      "is_home": True,  "scored": h_pts, "allowed": a_pts, "opponent": away})
        ah.append({"dt": dt, "won": not h_won,  "is_home": False, "scored": a_pts, "allowed": h_pts, "opponent": home})

    return np.array(X_rows, dtype=float), np.array(y_rows), dict(elo)


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    sport: str,
    games: list[dict],
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Train gradient boosting on game history and save to disk.
    Returns metrics dict.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss

    if len(games) < _MIN_GAMES:
        msg = f"{sport}: only {len(games)} games, need {_MIN_GAMES} — skipping"
        logger.warning(msg)
        if progress_cb:
            progress_cb(msg)
        return {"sport": sport, "error": "insufficient data", "n_games": len(games)}

    if progress_cb:
        progress_cb(f"{sport}: computing features for {len(games):,} games…")

    X, y, final_elo = _build_features(games, sport)

    if len(X) < _MIN_GAMES:
        return {"sport": sport, "error": "feature build too small", "n_games": len(X)}

    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    if progress_cb:
        progress_cb(f"{sport}: training on {len(X_tr):,} games, testing on {len(X_te):,}…")

    gb = GradientBoostingClassifier(
        n_estimators   = 300,
        max_depth      = 3,
        learning_rate  = 0.04,
        subsample      = 0.80,
        min_samples_leaf = 20,
        random_state   = 42,
    )
    gb.fit(X_tr_s, y_tr)

    # Manual isotonic calibration on holdout — avoids sklearn cv="prefit" API version issues
    raw_te = gb.predict_proba(X_te_s)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_te, y_te)
    cal = _Calibrated(gb, ir)

    proba = cal.predict_proba(X_te_s)[:, 1]
    bs    = brier_score_loss(y_te, proba)
    ll    = sk_log_loss(y_te, proba)
    base  = brier_score_loss(y_te, [y_tr.mean()] * len(y_te))
    bss   = round(1 - bs / base, 4) if base > 0 else 0.0
    acc   = float(np.mean((proba >= 0.5) == y_te))

    # Feature importance
    importances = sorted(
        zip(FEATURE_NAMES, gb.feature_importances_),
        key=lambda x: -x[1],
    )

    logger.info(
        "%s: n=%d Brier=%.4f BSS=%.3f LogLoss=%.4f Acc=%.3f",
        sport, len(games), bs, bss, ll, acc,
    )

    _CACHE_DIR.mkdir(exist_ok=True)
    bundle = {
        "sport":        sport,
        "model":        cal,
        "scaler":       scaler,
        "feature_names": FEATURE_NAMES,
        "final_elo":    final_elo,
        "n_games":      len(games),
        "brier":        round(bs, 5),
        "bss":          bss,
        "log_loss":     round(ll, 5),
        "accuracy":     round(acc, 4),
        "importances":  importances[:5],
        "trained_at":   datetime.utcnow().isoformat(),
    }
    with open(_model_path(sport), "wb") as f:
        pickle.dump(bundle, f)

    if progress_cb:
        progress_cb(f"{sport} ✓  BSS={bss:.3f}  Brier={bs:.4f}  n={len(games):,}")

    return {k: v for k, v in bundle.items() if k not in ("model", "scaler")}


def load(sport: str) -> dict | None:
    """Load saved trained bundle. Returns None if not found or stale."""
    path = _model_path(sport)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        logger.info(
            "Loaded %s model (n=%d BSS=%.3f trained=%s)",
            sport, bundle.get("n_games", 0),
            bundle.get("bss", 0), bundle.get("trained_at", "?")[:10],
        )
        return bundle
    except Exception as exc:
        logger.warning("Failed to load %s model: %s", sport, exc)
        return None


def train_all(
    years_back: int = DEFAULT_YEARS_BACK,
    progress_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    """Fetch data and train models for all supported sports. Returns metrics list."""
    from data.season_fetcher import fetch_all_historical_games
    import config

    results = []
    for sport_label in config.SUPPORTED_SPORTS:
        try:
            if progress_cb:
                progress_cb(f"Fetching {sport_label} data ({years_back} seasons)…")
            games = fetch_all_historical_games(sport_label, years_back=years_back, progress_cb=progress_cb)
            result = train(sport_label, games, progress_cb=progress_cb)
        except Exception as exc:
            logger.error("Training %s failed: %s", sport_label, exc)
            result = {"sport": sport_label, "error": str(exc)}
        results.append(result)
    return results
