"""
Streamlit app — DraftKings EV-based sports betting model.
Run: streamlit run app.py
"""
from __future__ import annotations

import sys
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="DK Betting Model",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from tracker.bankroll import BankrollTracker
from tracker.database import init_db
from data.daily_cache import bust_all, is_cached

init_db()

# ── Background scheduler (once per process) ───────────────────────────────────
if "scheduler_started" not in st.session_state:
    try:
        from data.scheduler import (
            add_ml_update_job, add_weekly_retrain_job,
            add_daily_picks_alert_job, start as _sched_start,
        )
        add_ml_update_job(hour=3, minute=0)
        add_weekly_retrain_job(day_of_week=6, hour=4, minute=0)
        add_daily_picks_alert_job(hour=8, minute=30, ev_min=5.0)
        _sched_start()
    except Exception:
        pass
    st.session_state["scheduler_started"] = True

# ── Auto-settle completed games once per session ──────────────────────────────
if "auto_settled_done" not in st.session_state:
    try:
        from tracker.auto_settle import auto_settle
        from tracker.bankroll import BankrollTracker as _BRT
        n = auto_settle()
        if n > 0:
            _BRT().auto_backup()
        st.session_state["auto_settled_done"] = True
        st.session_state["auto_settled_count"] = n
        if n > 0 and "calibrator" in st.session_state:
            del st.session_state["calibrator"]
    except Exception as _as_exc:
        st.session_state["auto_settled_done"] = True
        st.session_state["auto_settled_count"] = 0

# ── Calibrator (persists in session, rebuilt after settle) ────────────────────
if "calibrator" not in st.session_state:
    from models.calibrator import ModelCalibrator
    st.session_state["calibrator"] = ModelCalibrator.rebuild_from_db()

_calibrator = st.session_state["calibrator"]

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.bet-card {
    background: #1a2535;
    border: 1px solid #2d4060;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.bet-card-high  { border-left: 5px solid #00e676; }
.bet-card-med   { border-left: 5px solid #ffeb3b; }
.bet-card-low   { border-left: 5px solid #78909c; }
.bet-card-watch { border-left: 5px solid #546e7a; opacity: 0.8; }

.tag { display:inline-block; padding:2px 10px; border-radius:20px;
       font-size:0.78rem; font-weight:600; margin-right:4px; }
.tag-sport { background:#1e3a5f; color:#90caf9; }
.tag-type  { background:#1b3a2b; color:#81c784; }
.tag-watch { background:#2a3040; color:#78909c; }

.signal-pos { color: #00e676; }
.signal-neg { color: #ff5252; }
.signal-neu { color: #90a4ae; }

.action-box {
    background: #0d1f35;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0 4px 0;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ev_color(ev: float) -> str:
    if ev >= 5: return "#00e676"
    if ev >= 2: return "#ffeb3b"
    return "#90a4ae"

def _card_class(ev: float, stake: float) -> str:
    if stake <= 0:                     return "bet-card bet-card-watch"
    if ev >= 5:                        return "bet-card bet-card-high"
    if ev >= 2:                        return "bet-card bet-card-med"
    return "bet-card bet-card-low"

def _odds_str(american: float) -> str:
    return f"+{american:.0f}" if american >= 0 else f"{american:.0f}"

def _american_to_decimal(american: float) -> float:
    return (american / 100 + 1) if american >= 0 else (100 / abs(american) + 1)

def _be_win_rate(american: float) -> float:
    dec = _american_to_decimal(american)
    return 1.0 / dec if dec > 0 else 0.5

def _signal_html(signals: list[str]) -> str:
    parts = []
    for s in signals:
        body = s[2:] if len(s) > 2 else s
        if s.startswith("+"):
            parts.append(f'<span class="signal-pos">&#10003; {body}</span>')
        elif s.startswith("-"):
            parts.append(f'<span class="signal-neg">&#10007; {body}</span>')
        else:
            parts.append(f'<span class="signal-neu">&#8764; {body}</span>')
    return "<br>".join(parts)

def _html_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("No data yet.")
        return
    hcells = "".join(
        f'<th style="padding:8px 14px;background:#1e3a5f;color:#90caf9;'
        f'font-size:0.72rem;text-transform:uppercase;letter-spacing:0.8px;'
        f'border-bottom:2px solid #2d4060;white-space:nowrap;">{c}</th>'
        for c in df.columns
    )
    rows_html = ""
    for idx, row in df.iterrows():
        bg = "#0f1a2e" if idx % 2 == 0 else "#111e30"
        cells = "".join(
            f'<td style="padding:7px 14px;border-bottom:1px solid #1e2d40;'
            f'color:#dde6f0;font-size:0.85rem;">{v}</td>'
            for v in row
        )
        rows_html += f'<tr style="background:{bg};">{cells}</tr>'
    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #2d4060;'
        f'margin-bottom:12px;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{hcells}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


# ── Session state defaults ─────────────────────────────────────────────────────
_KELLY        = 0.25
_BASE_BANKROLL = 100.0

_LEAGUE_DISPLAY = {
    "NFL": "NFL", "NCAAF": "NCAAF", "NBA": "NBA", "WNBA": "WNBA",
    "NCAAB": "NCAAB", "MLB": "MLB", "NHL": "NHL",
    "MLS": "MLS", "EPL": "EPL", "LALIGA": "La Liga",
    "BUNDESLIGA": "Bundesliga", "SERIEA": "Serie A", "LIGUE1": "Ligue 1",
    "UCL": "UCL", "UEL": "UEL", "LIGAMX": "Liga MX",
    "MMA": "MMA", "BOXING": "Boxing",
}

def _init_state():
    defaults = {
        "sport":       "All",
        "ev_min":      3.0,       # 3% — realistic edge threshold for value betting
        "ev_min_odds": -250,
        "ev_max_odds":  400,
        "bankroll":    _BASE_BANKROLL,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Cached pipeline ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _run_ev_pipeline(sport_filter, kelly_frac, bankroll):
    from run import run_pipeline
    import config as cfg
    cfg.EV_THRESHOLD           = -1.0   # no floor — log everything, filter in UI
    cfg.DEFAULT_KELLY_FRACTION = kelly_frac
    cfg.STARTING_BANKROLL      = bankroll
    cfg.MIN_CONFIDENCE         = 0
    cfg.MAX_PICKS_PER_DAY      = 500
    picks, stats, api_rem = run_pipeline(sport_filter or None)
    return [p.to_dict() for p in picks], stats, api_rem


def _auto_log_model_picks(picks: list[dict], bankroll: float, kelly_frac: float) -> int:
    from tracker.database import session_scope
    from tracker.models import Bet as _BetModel
    from models.kelly_criterion import kelly_fraction as _kf, flat_stake

    with session_scope() as s:
        existing = {
            (r.game_id, r.bet_type, r.side)
            for r in s.query(_BetModel.game_id, _BetModel.bet_type, _BetModel.side).all()
        }
        new_count = 0
        for p in picks:
            key = (p["game_id"], p["bet_type"], p["side"])
            if key in existing:
                continue
            try:
                frac  = _kf(p["model_prob"], p["dk_odds"], kelly_frac)
                stake = round(bankroll * frac, 2)
                ct    = datetime.fromisoformat(p["commence_time_iso"])
                bet   = _BetModel(
                    sport=p["sport"],
                    game_id=p["game_id"],
                    home_team=p["home_team"],
                    away_team=p["away_team"],
                    commence_time=ct,
                    bet_type=p["bet_type"],
                    side=p["side"],
                    line=p["line"],
                    bookmaker="draftkings",
                    odds=float(p["dk_odds"]),
                    model_prob=p["model_prob"],
                    implied_prob=p["implied_prob"],
                    ev_pct=p["ev_pct"],
                    kelly_frac=frac,
                    stake_kelly=stake,
                    stake_flat=flat_stake(bankroll),
                    bankroll_at_bet=bankroll,
                    settled=False,
                    auto_tracked=True,
                )
                s.add(bet)
                existing.add(key)
                new_count += 1
            except Exception:
                pass
    return new_count


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## DK Model  ·  EV Focus")
    st.caption(f"Today: {date.today().isoformat()}")
    st.divider()

    page = st.radio(
        "Page",
        ["EV Picks", "Bankroll", "Bet History"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Filters")

    # Build ordered sport list: "All" first, then grouped by category
    _sport_opts = ["All"] + [
        s for cat in config.SPORT_CATEGORIES.values() for s in cat
    ]
    _sport_idx  = _sport_opts.index(st.session_state["sport"]) if st.session_state["sport"] in _sport_opts else 0
    st.session_state["sport"] = st.selectbox(
        "Sport", _sport_opts, index=_sport_idx,
        help="ESPN sports (NFL/NCAAF/NBA/WNBA/NCAAB/MLB/NHL) use ML models. "
             "Others use multi-book consensus edge detection.",
    )

    st.session_state["ev_min"] = st.slider(
        "Min EV %", 0.5, 25.0,
        float(st.session_state["ev_min"]), 0.5, format="%.1f%%",
        help="Hide picks below this EV threshold. Professionals typically target 3–7%.",
    )
    st.session_state["ev_min_odds"] = st.slider(
        "Exclude odds heavier than",
        min_value=-350, max_value=-100,
        value=int(st.session_state["ev_min_odds"]), step=10, format="%d",
        help="-250 excludes -300, -400 etc. Heavy favourites compress upside.",
    )
    st.session_state["ev_max_odds"] = st.slider(
        "Exclude longshots above",
        min_value=150, max_value=600,
        value=int(st.session_state["ev_max_odds"]), step=25, format="+%d",
        help="Caps at +400 by default — very long odds amplify variance.",
    )
    st.session_state["bankroll"] = st.number_input(
        "Bankroll ($)", 10.0,
        value=float(st.session_state["bankroll"]), step=10.0,
        help="Used only for stake sizing — does not affect EV or edge calculations.",
    )

    # Show effective bankroll after real P&L
    try:
        from tracker.database import session_scope as _sbss
        from tracker.models import Bet as _sbBM
        from sqlalchemy import func as _sbf
        with _sbss() as _sbs:
            _sb_pnl = _sbs.query(_sbf.sum(_sbBM.pnl_kelly)).filter(
                _sbBM.settled == True,
                _sbBM.bookmaker != "espn_historical",
            ).scalar() or 0.0
        _sb_eff = max(round(st.session_state["bankroll"] + _sb_pnl, 2),
                      st.session_state["bankroll"] * 0.10)
        _sb_delta = _sb_eff - st.session_state["bankroll"]
        if abs(_sb_delta) > 0.01:
            _sb_color = "#00e676" if _sb_delta >= 0 else "#ff5252"
            st.markdown(
                f'<div style="font-size:0.82rem;color:{_sb_color};margin-top:-4px;">'
                f'Effective: <b>${_sb_eff:,.2f}</b> ({_sb_delta:+.2f} real P&L)</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    # Self-learning / calibration status
    _cal = _calibrator
    if _cal.n_settled >= 10:
        st.success(f"Self-learning active · {_cal.n_settled} settled bets")
        if _cal.brier_score:
            _bl = f"Brier: {_cal.brier_score:.4f} · {_cal.model_quality_label()}"
            if _cal.brier_improvement > 0:
                _bl += f" · IR improved {_cal.brier_improvement:.1f}%"
            st.caption(_bl)
        if _cal.using_isotonic:
            st.caption("Isotonic calibration active")
        st.caption("Auto-updates: nightly 3 AM · retrain Sundays 4 AM")
    elif _cal.n_settled > 0:
        st.info(f"Calibrating… {_cal.n_settled}/20 bets")
    else:
        st.caption("Self-learning activates after first settled bets.")

    st.divider()

    # API / cache status
    if config.ODDS_API_KEY:
        st.success("API Key: configured")
    else:
        st.error("API Key missing — add ODDS_API_KEY to .env or Streamlit secrets")

    if st.button("Force Refresh Odds", use_container_width=True,
                 help="Burns ~4 API requests. Fetches fresh DraftKings lines."):
        if not config.ODDS_API_KEY:
            st.error("Cannot refresh — ODDS_API_KEY not set.")
        else:
            bust_all()
            st.cache_data.clear()
            st.rerun()

    st.caption("Cache refreshes automatically at 7 AM ET daily.")
    st.divider()

    # ML Model training status
    st.markdown("### ML Model Training")
    st.caption("Trains on 3 seasons of ESPN data per sport. Run once, re-run monthly.")

    from models.trainer import load as _load_bundle
    _any_trained = False
    for _sl in sorted(config.ESPN_SPORTS):
        _b = _load_bundle(_sl)
        if _b:
            _any_trained = True
            st.caption(
                f"{_sl}: BSS={_b.get('bss', 0):.3f} · "
                f"n={_b.get('n_games', 0):,} · "
                f"{_b.get('trained_at', '?')[:10]}"
            )
    if not _any_trained:
        st.warning("No trained model — click Train to build it.")
    st.caption("Non-ESPN sports (soccer, MMA, etc.) use consensus-only mode — no training needed.")

    if st.button("Train Model (3 seasons)", use_container_width=True):
        from models.trainer import train_all as _train_all
        _tprog = st.empty()
        try:
            with st.spinner("Fetching ESPN data and training… (2-5 min first run)"):
                _results = _train_all(
                    years_back=3,
                    progress_cb=lambda m: _tprog.caption(m),
                )
            _tprog.empty()
            _ok   = [r for r in _results if "error" not in r]
            _fail = [r for r in _results if "error" in r]
            for r in _ok:
                st.success(
                    f"{r['sport']}: trained on {r.get('n_games', 0):,} games · "
                    f"BSS={r.get('bss', 0):.3f}"
                )
            for r in _fail:
                st.warning(f"{r['sport']}: {r.get('error', 'failed')}")
            if _ok:
                st.cache_data.clear()
                st.rerun()
        except Exception as _te:
            _tprog.empty()
            st.error(f"Training error: {_te}")


# ── Active session values ──────────────────────────────────────────────────────
sport_arg     = None if st.session_state["sport"] == "All" else st.session_state["sport"]
bankroll_val  = st.session_state["bankroll"]
ev_min_filter = st.session_state["ev_min"]
odds_lo       = st.session_state["ev_min_odds"]
odds_hi       = st.session_state["ev_max_odds"]

from models.kelly_criterion import kelly_fraction as _kf
from models.ev_calculator import american_to_decimal as _a2d, decimal_to_american as _d2a, calculate_ev as _cev

# Compound bankroll
try:
    from tracker.database import session_scope as _ss
    from tracker.models import Bet as _BM
    from sqlalchemy import func as _func
    with _ss() as _s:
        _real_pnl = _s.query(_func.sum(_BM.pnl_kelly)).filter(
            _BM.settled == True,
            _BM.bookmaker != "espn_historical",
        ).scalar() or 0.0
    _effective_bankroll = max(round(bankroll_val + _real_pnl, 2), bankroll_val * 0.10)
except Exception:
    _effective_bankroll = bankroll_val

# Injury map
if "injury_map" not in st.session_state:
    try:
        from data.news_monitor import fetch_all_injuries
        st.session_state["injury_map"] = fetch_all_injuries()
    except Exception:
        st.session_state["injury_map"] = {}
_injury_map = st.session_state["injury_map"]

# ── Run pipeline ──────────────────────────────────────────────────────────────
_all_dicts:  list[dict] = []
_ev_stats:   dict       = {}
_api_rem:    int | None = None
_pipe_error: str | None = None

if not config.ODDS_API_KEY:
    _pipe_error = "ODDS_API_KEY not set. Add it to .env or Streamlit secrets."
else:
    try:
        _all_dicts, _ev_stats, _api_rem = _run_ev_pipeline(sport_arg, _KELLY, bankroll_val)

        # Stale cache guard
        if _all_dicts and ("game_id" not in _all_dicts[0] or "n_books" not in _all_dicts[0]):
            st.cache_data.clear()
            st.rerun()

        # Apply calibration + multi-signal Kelly scaling
        from data.news_monitor import injuries_for_teams as _inj2
        for _pd in _all_dicts:
            if not _calibrator.is_profitable_segment(_pd["sport"], _pd["bet_type"]):
                _pd["kelly_frac"]  = 0.0
                continue

            _adj_p = _calibrator.calibrate_prob(_pd["model_prob"], _pd["sport"], _pd["bet_type"])
            _km    = _calibrator.kelly_multiplier(_pd["sport"])

            _nb = _pd.get("n_books", 0)
            _book_mult = (
                min(1.0 + (_nb - config.MIN_BOOKS_FOR_CONSENSUS) * 0.05, 1.25)
                if _nb >= config.MIN_BOOKS_FOR_CONSENSUS else 0.75
            )

            _h_inj, _a_inj = _inj2(_pd["home_team"], _pd["away_team"], _injury_map)
            _bet_inj = _h_inj if _pd["side"] in ("home", "over", "under") else _a_inj
            _inj_mult = 0.60 if any(i["status"].lower() == "out" for i in _bet_inj) else 1.0

            _final_km = _km * _book_mult * _inj_mult
            _pd["model_prob"]  = _adj_p
            _pd["ev_pct"]      = _cev(_adj_p, _pd["dk_odds"]) * 100
            _pd["kelly_frac"]  = _kf(_adj_p, _pd["dk_odds"], _KELLY * _final_km)

        # Portfolio cap applied to TODAY's picks only (ET date).
        # Tomorrow's picks show for planning but don't consume today's budget.
        # Max 25% of the $100 stake bankroll ($25) spread across today's slate.
        _MAX_PORTFOLIO  = 0.25
        _cap_today_et   = datetime.now(_ET).date()

        def _pick_is_today(p: dict) -> bool:
            try:
                ct = datetime.fromisoformat(p.get("commence_time_iso", "")).replace(tzinfo=timezone.utc)
                return ct.astimezone(_ET).date() == _cap_today_et
            except Exception:
                return False

        _today_picks_cap = [p for p in _all_dicts if _pick_is_today(p) and p.get("kelly_frac", 0) > 0]
        _today_frac_sum  = sum(p["kelly_frac"] for p in _today_picks_cap)
        if _today_frac_sum > _MAX_PORTFOLIO:
            _cap_scale = _MAX_PORTFOLIO / _today_frac_sum
            for _pd in _today_picks_cap:
                _pd["kelly_frac"] = round(_pd["kelly_frac"] * _cap_scale, 4)

        # Auto-log picks to DB
        _n_new = _auto_log_model_picks(_all_dicts, _effective_bankroll, _KELLY)
        if _n_new > 0:
            from tracker.auto_settle import auto_settle as _settle_new
            _settle_new()
            if "calibrator" in st.session_state:
                del st.session_state["calibrator"]
            if "telegram_sent_today" not in st.session_state:
                try:
                    from alerts.telegram_alert import TelegramAlert as _TGA
                    _tg = _TGA()
                    if _tg.enabled:
                        _tg.send_picks(_all_dicts, _effective_bankroll, ev_min=5.0)
                    st.session_state["telegram_sent_today"] = True
                except Exception:
                    pass

    except Exception as _pe:
        _pipe_error = str(_pe)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EV Picks
# ══════════════════════════════════════════════════════════════════════════════

if page == "EV Picks":
    st.title("EV Picks")
    st.caption(
        "All picks where the model finds a mathematical edge on DraftKings. "
        "Ranked by EV%. Stale games (already started) shown in Completed tab."
    )

    if _pipe_error:
        st.error(f"Pipeline error: {_pipe_error}")
        st.info("Check that ODDS_API_KEY is set in your .env or Streamlit secrets.")
        st.stop()

    _now = datetime.now(timezone.utc)

    def _commence_dt(p: dict) -> datetime:
        try:
            return datetime.fromisoformat(p["commence_time_iso"]).replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.max.replace(tzinfo=timezone.utc)

    # Filter by EV and odds range, then split by game time.
    # Show ALL positive-EV picks regardless of stake — let the bettor decide.
    _qualifying = [
        p for p in _all_dicts
        if p["ev_pct"] >= ev_min_filter
        and odds_lo <= p["dk_odds"] <= odds_hi
    ]
    # Use ET date for "today/tomorrow" bucketing — an 8 PM ET game is midnight UTC
    # (next calendar day in UTC), so UTC date comparisons mis-bucket evening games.
    _today_et = datetime.now(_ET).date()

    def _et_date(p: dict) -> date:
        return _commence_dt(p).astimezone(_ET).date()

    _upcoming = sorted(
        [p for p in _qualifying if _now < _commence_dt(p) < _now + timedelta(days=2)],
        key=lambda p: ((_et_date(p) - _today_et).days, -p["ev_pct"]),
    )
    _started = sorted(
        [p for p in _qualifying if _commence_dt(p) <= _now],
        key=lambda p: _commence_dt(p),
    )

    # KPI row
    _active_with_stake = [p for p in _upcoming if _pick_is_today(p) and p.get("kelly_frac", 0) > 0]
    _total_stake       = sum(round(_BASE_BANKROLL * p["kelly_frac"], 2) for p in _active_with_stake)
    _avg_ev            = sum(p["ev_pct"] for p in _upcoming) / len(_upcoming) if _upcoming else 0
    _avg_edge          = sum(
        (p["model_prob"] - p["implied_prob"]) * 100 for p in _upcoming
    ) / len(_upcoming) if _upcoming else 0

    _kc1, _kc2, _kc3, _kc4, _kc5 = st.columns(5)
    _kc1.metric("Qualifying Picks", len(_upcoming))
    _kc2.metric("With Stake Rec.", len(_active_with_stake))
    _kc3.metric("Avg EV %", f"{_avg_ev:.1f}%" if _upcoming else "—")
    _kc4.metric("Avg Edge", f"{_avg_edge:.1f}%" if _upcoming else "—")
    _kc5.metric("Total Exposure", f"${_total_stake:.2f}" if _active_with_stake else "$0")

    _tab_active, _tab_done = st.tabs(
        [f"Active ({len(_upcoming)})", f"Completed Today ({len(_started)})"]
    )

    # ═══ ACTIVE TAB ════════════════════════════════════════════════════════════
    with _tab_active:
        if not _upcoming:
            st.warning(
                "No qualifying EV picks right now. "
                f"Try lowering Min EV% (currently {ev_min_filter:.1f}%) "
                "or widening the odds range."
            )
            st.info("Picks refresh daily at 7 AM ET as new games open.")
        else:
            from data.news_monitor import injuries_for_teams as _inj_for

            # Separate actionable bets from monitor-only
            _bet_picks   = [p for p in _upcoming if p.get("kelly_frac", 0) > 0]
            _watch_picks = [p for p in _upcoming if p.get("kelly_frac", 0) <= 0]

            if _bet_picks:
                st.markdown(
                    f'<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1.2px;'
                    f'color:#78909c;margin-bottom:8px;">'
                    f'&#9654; BET — {len(_bet_picks)} picks with stake recommendation</div>',
                    unsafe_allow_html=True,
                )

            for _i, _p in enumerate(_bet_picks + _watch_picks, 1):
                is_watch = _p.get("kelly_frac", 0) <= 0
                if not _p.get("model_prob") or not _p.get("implied_prob"):
                    continue

                _ev      = _p["ev_pct"]
                _prob    = _p["model_prob"]
                _imp     = _p["implied_prob"]
                _edge    = round((_prob - _imp) * 100, 1)
                _odds    = _p["dk_odds"]
                _odds_s  = _odds_str(_odds)
                _frac    = _p.get("kelly_frac", 0)
                _stake   = round(_BASE_BANKROLL * _frac, 2) if _frac > 0 else 0
                _to_win  = round(_stake * (_american_to_decimal(_odds) - 1), 2) if _stake > 0 else 0
                _be_rate = _be_win_rate(_odds)
                _bteam   = _p.get("bet_team", _p["side"])
                _blbl    = _p["bet_type"].replace("total_", "").replace("_", " ").title()
                _sdisp   = _p["side"].upper() + (f" {_p['line']:+g}" if _p["line"] else "")
                _nb      = _p.get("n_books", 0)
                _medge   = _p.get("market_edge", 0.0)
                _cons_p  = _p.get("consensus_prob")
                _sigs    = _signal_html(_p.get("signals", []))
                _ev_col  = _ev_color(_ev)
                _medge_col = "#00e676" if _medge > 0 else "#ff5252"
                _card_cls  = "bet-card bet-card-watch" if is_watch else _card_class(_ev, _stake)

                # Time to game
                _gdt  = _commence_dt(_p)
                _mins = int((_gdt - _now).total_seconds() / 60)
                if _mins < 60:    _tbcol, _tlbl = "#ff5252", f"{_mins}m to tip"
                elif _mins < 180: _tbcol, _tlbl = "#ffeb3b", f"{_mins//60}h {_mins%60}m to tip"
                else:             _tbcol, _tlbl = "#90a4ae", f"{_mins//60}h to tip"

                # Injury display
                _h_inj, _a_inj = _inj_for(_p["home_team"], _p["away_team"], _injury_map)
                _all_inj = [*[(i, "home") for i in _h_inj], *[(i, "away") for i in _a_inj]]
                _inj_html = ""
                if _all_inj:
                    _out = sum(1 for i, _ in _all_inj if i["status"].lower() == "out")
                    _gtd = len(_all_inj) - _out
                    _parts = []
                    if _out:
                        _parts.append(
                            f'<span style="background:#ff525222;color:#ff5252;border:1px solid #ff5252;'
                            f'padding:1px 8px;border-radius:12px;font-size:0.75rem;font-weight:700;">'
                            f'{_out} OUT</span>'
                        )
                    if _gtd:
                        _parts.append(
                            f'<span style="background:#ffeb3b22;color:#ffeb3b;border:1px solid #ffeb3b;'
                            f'padding:1px 8px;border-radius:12px;font-size:0.75rem;font-weight:700;">'
                            f'{_gtd} GTD</span>'
                        )
                    _det = " &nbsp;".join(
                        f'<span style="color:#b0bec5;font-size:0.75rem;">'
                        f'{i["player"]} ({i["position"]}) {i["status"]}</span>'
                        for i, _ in _all_inj[:4]
                    )
                    _inj_html = (
                        f'<div style="margin-top:6px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;">'
                        f'<span style="color:#78909c;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Injuries</span>'
                        + " ".join(_parts) + f'&nbsp;&nbsp;{_det}</div>'
                    )

                # Consensus line
                _cons_html = ""
                if _cons_p:
                    _cons_html = (
                        f'<div style="margin-top:4px;font-size:0.8rem;color:#90a4ae;">'
                        f'Consensus ({_nb} books): <b style="color:#90caf9;">{_cons_p:.1%}</b>'
                        f' &nbsp;·&nbsp; DK no-vig: {_imp:.1%}'
                        f' &nbsp;·&nbsp; Market edge: <b style="color:{_medge_col};">{_medge:+.1f}%</b>'
                        f'</div>'
                    )
                else:
                    _cons_html = (
                        f'<div style="margin-top:4px;font-size:0.75rem;color:#546e7a;">'
                        f'Model only ({_nb} book{"s" if _nb != 1 else ""}) — '
                        f'lower confidence without multi-book consensus</div>'
                    )

                # Separator before watch section
                if is_watch and _p is _watch_picks[0] if _watch_picks else False:
                    st.markdown(
                        f'<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:1.2px;'
                        f'color:#546e7a;margin:18px 0 8px 0;">'
                        f'&#9711; MONITOR — positive EV, no stake recommended yet '
                        f'(insufficient historical data on this segment)</div>',
                        unsafe_allow_html=True,
                    )

                # Action summary line
                if is_watch:
                    _action_html = (
                        f'<div class="action-box" style="color:#78909c;">'
                        f'<b>Watch:</b> {_bteam} ({_blbl} {_sdisp} @ {_odds_s}) — '
                        f'positive EV but model has not yet established profitability on this segment. '
                        f'Track for future reference.</div>'
                    )
                else:
                    _action_html = (
                        f'<div class="action-box">'
                        f'<b style="color:#fff;">Bet {_bteam}</b> '
                        f'<span style="color:#90a4ae;">({_blbl} {_sdisp})</span> '
                        f'@ <b style="color:{_ev_col};">{_odds_s}</b> &nbsp;·&nbsp; '
                        f'Risk <b style="color:#fff;">${_stake:.2f}</b> '
                        f'to win <b style="color:#00e676;">${_to_win:.2f}</b>'
                        f'</div>'
                    )

                st.markdown(f"""
<div class="{_card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <span style="background:#1e3a5f;color:#90caf9;border-radius:50%;width:28px;height:28px;
                   display:inline-flex;align-items:center;justify-content:center;
                   font-weight:800;font-size:0.9rem;flex-shrink:0;">#{_i}</span>
      <span style="font-size:1.15rem;font-weight:800;color:#ffffff;margin-left:4px;">{_p['game']} <span style="color:#90a4ae;font-size:0.88rem;font-weight:500;">({_LEAGUE_DISPLAY.get(_p.get('sport',''), _p.get('sport',''))})</span></span>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <span style="background:{_tbcol}22;color:{_tbcol};border:1px solid {_tbcol};
                   padding:2px 10px;border-radius:20px;font-size:0.8rem;font-weight:600;">{_tlbl}</span>
      <span style="font-size:2rem;font-weight:900;color:{_ev_col if not is_watch else '#78909c'};">{_ev:+.1f}% EV</span>
    </div>
  </div>
  {_inj_html}
  {_cons_html}
  {_action_html}

  <div style="margin-top:12px;display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:14px 20px;">
    <div>
      <div style="color:#78909c;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">EV %</div>
      <div style="font-size:1.35rem;font-weight:900;color:{_ev_col};">{_ev:+.1f}%</div>
      <div style="font-size:0.75rem;color:#78909c;">per dollar risked</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Edge vs DK</div>
      <div style="font-size:1.35rem;font-weight:900;color:{_ev_col};">+{_edge:.1f}%</div>
      <div style="font-size:0.75rem;color:#78909c;">model − no-vig</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Model Prob</div>
      <div style="font-size:1.2rem;font-weight:800;color:#ffffff;">{_prob:.1%}</div>
      <div style="font-size:0.75rem;color:#78909c;">DK no-vig: {_imp:.1%}</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Break-Even</div>
      <div style="font-size:1.2rem;font-weight:700;color:#ff5252;">{_be_rate:.1%}</div>
      <div style="font-size:0.75rem;color:#78909c;">win rate needed</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">DK Odds</div>
      <div style="font-size:1.35rem;font-weight:800;color:#ffffff;">{_odds_s}</div>
      <div style="font-size:0.75rem;color:#78909c;">raw implied {_imp:.1%}</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Kelly Stake</div>
      <div style="font-size:1.2rem;font-weight:800;color:#ffffff;">{'$'+f'{_stake:.2f}' if _stake>0 else '—'}</div>
      <div style="font-size:0.75rem;color:#78909c;">{f'{_frac:.1%} of bankroll' if _frac>0 else 'segment unproven'}</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Books Agreeing</div>
      <div style="font-size:1.2rem;font-weight:800;color:#90caf9;">{_nb}</div>
      <div style="font-size:0.75rem;color:#78909c;">{'consensus ✓' if _nb >= 3 else 'low consensus'}</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Market Edge</div>
      <div style="font-size:1.2rem;font-weight:800;color:{_medge_col};">{_medge:+.1f}%</div>
      <div style="font-size:0.75rem;color:#78909c;">vs consensus line</div>
    </div>
  </div>

  <div style="margin-top:12px;border-top:1px solid #1e2e42;padding-top:10px;
              font-size:0.83rem;line-height:1.9;color:#b0bec5;">
    <span style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;color:#546e7a;">
      Signals — {_bteam}</span><br>
    {_sigs if _sigs else '<span style="color:#546e7a;">No signal data available</span>'}
  </div>
</div>
""", unsafe_allow_html=True)

    # ═══ COMPLETED TODAY TAB ═══════════════════════════════════════════════════
    with _tab_done:
        if not _started:
            st.info("No completed games from today's qualifying picks yet.")
        else:
            from tracker.database import session_scope as _ss2
            from tracker.models import Bet as _BM2
            _sid = list({p["game_id"] for p in _started})
            with _ss2() as _s2:
                _db_bets = {
                    (b.game_id, b.bet_type, b.side): {
                        "settled": b.settled, "won": b.won,
                        "pnl": b.pnl_kelly, "id": b.id,
                    }
                    for b in _s2.query(_BM2).filter(_BM2.game_id.in_(_sid)).all()
                }

            _n_won  = sum(1 for v in _db_bets.values() if v["settled"] and v["won"])
            _n_lost = sum(1 for v in _db_bets.values() if v["settled"] and not v["won"])
            _n_open = sum(1 for v in _db_bets.values() if not v["settled"])
            if _n_won or _n_lost:
                _rc1, _rc2, _rc3 = st.columns(3)
                _rc1.metric("Wins",    _n_won)
                _rc2.metric("Losses",  _n_lost)
                _rc3.metric("Pending", _n_open)
            st.caption("Games that have started. Results auto-settle ~4h after tip.")

            for _p in _started:
                _ev     = _p["ev_pct"]
                _ev_col = _ev_color(_ev)
                _bteam  = _p.get("bet_team", _p["side"])
                _blbl   = _p["bet_type"].replace("total_", "").replace("_", " ").title()
                _sdisp  = _p["side"].upper() + (f" {_p['line']:+g}" if _p["line"] else "")
                _odds_s_c = _odds_str(_p["dk_odds"])
                _db_key = (_p["game_id"], _p["bet_type"], _p["side"])
                _db_rec = _db_bets.get(_db_key)

                if _db_rec and _db_rec["settled"]:
                    _won = _db_rec["won"]
                    _pnl = _db_rec["pnl"] or 0
                    _result_badge = (
                        f'<span style="background:#00e67622;color:#00e676;border:1px solid #00e676;'
                        f'padding:2px 12px;border-radius:20px;font-size:0.85rem;font-weight:700;">WIN  +${_pnl:.2f}</span>'
                        if _won else
                        f'<span style="background:#ff525222;color:#ff5252;border:1px solid #ff5252;'
                        f'padding:2px 12px;border-radius:20px;font-size:0.85rem;font-weight:700;">LOSS  -${abs(_pnl):.2f}</span>'
                    )
                elif _db_rec:
                    _result_badge = (
                        '<span style="background:#ffeb3b22;color:#ffeb3b;border:1px solid #ffeb3b;'
                        'padding:2px 12px;border-radius:20px;font-size:0.8rem;">Settling…</span>'
                    )
                else:
                    _result_badge = (
                        '<span style="background:#546e7a33;color:#90a4ae;border:1px solid #546e7a;'
                        'padding:2px 12px;border-radius:20px;font-size:0.8rem;">Started</span>'
                    )

                _card_bg = (
                    "border-left:5px solid #00e676;" if (_db_rec and _db_rec["settled"] and _db_rec["won"])
                    else "border-left:5px solid #ff5252;" if (_db_rec and _db_rec["settled"] and not _db_rec["won"])
                    else ""
                )

                st.markdown(f"""
<div class="bet-card" style="opacity:0.88;{_card_bg}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <span class="tag tag-sport">{_p['sport']}</span>
      <span class="tag tag-type">{_blbl}</span>
      <span style="font-size:1.05rem;font-weight:700;color:#cfd8dc;">{_p['game']}</span>
    </div>
    {_result_badge}
  </div>
  <div style="margin-top:10px;font-size:0.9rem;color:#b0bec5;">
    <b style="color:#fff;">{_bteam}</b> &nbsp;{_blbl} {_sdisp} &nbsp;
    <b style="color:{_ev_col};">{_odds_s_c}</b> &nbsp;·&nbsp;
    EV at post: <b style="color:{_ev_col};">{_ev:+.1f}%</b> &nbsp;·&nbsp;
    Model: {_p['model_prob']:.1%} &nbsp;·&nbsp;
    Break-even: {_be_win_rate(_p['dk_odds']):.1%}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Bankroll
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Bankroll":
    from collections import defaultdict as _dd
    from datetime import timedelta
    from tracker.database import session_scope as _bss
    from tracker.models import Bet as _BetM

    st.title("Bankroll Tracker")

    _br_hdr, _br_date_col = st.columns([3, 1])
    with _br_date_col:
        _br_start = st.date_input(
            "Track from",
            value=date.today() - timedelta(days=30),
            help="Only bets placed on or after this date are included.",
        )

    with _bss() as _bs:
        _all_settled = _bs.query(_BetM).filter(_BetM.settled == True).all()

    def _bet_date(b) -> datetime:
        return b.settled_at or b.placed_at or b.commence_time or datetime(2000, 1, 1)

    _mbets = sorted(
        [b for b in _all_settled if _bet_date(b).date() >= _br_start],
        key=_bet_date,
    )
    _real_mbets = [
        b for b in _mbets
        if b.bookmaker != "espn_historical" and (b.ev_pct or 0) >= ev_min_filter
    ]

    with _br_hdr:
        st.caption(
            f"**{len(_real_mbets):,}** bets (≥{ev_min_filter:.1f}% EV) from {_br_start.isoformat()} · "
            f"{len([b for b in _mbets if b.bookmaker != 'espn_historical']):,} total real bets in range."
        )

    if not _real_mbets:
        st.info(
            f"No bets with ≥{ev_min_filter:.1f}% EV in the selected range. "
            "Widen the date range or lower Min EV% in the sidebar."
        )
        st.stop()

    # KPIs
    _total_pnl  = sum(b.pnl_kelly or 0 for b in _real_mbets)
    _model_br   = _BASE_BANKROLL + _total_pnl
    _total_ret  = (_model_br - _BASE_BANKROLL) / _BASE_BANKROLL * 100
    _wins_m     = sum(1 for b in _real_mbets if b.won)
    _hit_m      = _wins_m / len(_real_mbets) if _real_mbets else 0
    _wagered_m  = sum(b.stake_kelly or 0 for b in _real_mbets)
    _roi_m      = (_total_pnl / _wagered_m * 100) if _wagered_m > 0 else 0
    _avg_ev_m   = sum(b.ev_pct or 0 for b in _real_mbets) / len(_real_mbets)
    _clv_bets   = [b for b in _real_mbets if b.clv is not None]
    _avg_clv    = sum(b.clv for b in _clv_bets) / len(_clv_bets) if _clv_bets else None
    _beat_close = sum(1 for b in _clv_bets if b.clv > 0) / len(_clv_bets) if _clv_bets else None

    with _bss() as _bs2:
        _open_count = _bs2.query(_BetM).filter(_BetM.settled == False).count()

    _k1, _k2, _k3, _k4, _k5, _k6 = st.columns(6)
    _k1.metric("ROI",          f"{_roi_m:+.1f}%",   f"{_total_pnl:+.2f} P&L")
    _k2.metric("Hit Rate",     f"{_hit_m:.1%}",      f"{_wins_m}W – {len(_real_mbets)-_wins_m}L")
    _k3.metric("Model Bankroll", f"${_model_br:,.2f}", f"{_total_ret:+.1f}% from $100")
    _k4.metric("Avg EV",       f"{_avg_ev_m:.1f}%",  f"{len(_real_mbets)} bets")
    _k5.metric("Avg CLV",      f"{_avg_clv:+.2f}%" if _avg_clv is not None else "—",
               help="Closing line value — the gold standard for long-term edge proof.")
    _k6.metric("Open Bets",    str(_open_count))

    # ROI is the #1 professional metric — explain it
    _roi_col = "#00e676" if _roi_m >= 0 else "#ff5252"
    st.markdown(
        f'<div style="background:#0d1f35;border:1px solid #1e3a5f;border-radius:8px;'
        f'padding:10px 16px;font-size:0.85rem;color:#90a4ae;margin-bottom:8px;">'
        f'<b style="color:#fff;">ROI {_roi_m:+.1f}%</b> on ${_wagered_m:,.0f} wagered. '
        f'This is the only number that matters long-term. '
        f'A sustained ROI > 3% is elite. CLV > +0.5% confirms the edge is real, not variance.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Daily bankroll chart
    st.subheader("Bankroll Growth")
    _daily_pnl: dict  = _dd(float)
    _daily_bets: dict = _dd(list)
    for b in _real_mbets:
        _d = _bet_date(b).date()
        _daily_pnl[_d]  += (b.pnl_kelly or 0)
        _daily_bets[_d].append(b)

    _running = _BASE_BANKROLL
    _daily_rows = []
    for _dt in sorted(_daily_pnl):
        _day_p   = _daily_pnl[_dt]
        _day_br  = _running
        _day_pct = _day_p / _day_br * 100 if _day_br > 0 else 0
        _running += _day_p
        _day_bets = _daily_bets[_dt]
        _day_w    = sum(1 for b in _day_bets if b.won)
        _daily_rows.append({
            "Date":     _dt,
            "Bets":     len(_day_bets),
            "W":        _day_w,
            "L":        len(_day_bets) - _day_w,
            "P&L":      round(_day_p, 2),
            "Daily %":  round(_day_pct, 2),
            "Bankroll": round(_running, 2),
        })

    if _daily_rows:
        _ddf = pd.DataFrame(_daily_rows)

        _fig_br = go.Figure()
        _fig_br.add_hline(y=_BASE_BANKROLL, line_dash="dash", line_color="#546e7a",
                          annotation_text=f"Start ${_BASE_BANKROLL:.0f}")
        _fig_br.add_trace(go.Scatter(
            x=_ddf["Date"], y=_ddf["Bankroll"], mode="lines+markers",
            line=dict(color="#00e676", width=3),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.07)",
            hovertemplate="<b>%{x}</b><br>$%{y:.2f}<extra></extra>",
        ))
        _fig_br.update_layout(
            title="Model Bankroll (from $100 start)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="", yaxis_title="Bankroll ($)", showlegend=False, height=300,
        )
        st.plotly_chart(_fig_br, use_container_width=True)

        _fig_pnl = go.Figure()
        _fig_pnl.add_trace(go.Bar(
            x=_ddf["Date"], y=_ddf["P&L"],
            marker_color=["#00e676" if v >= 0 else "#ff5252" for v in _ddf["P&L"]],
            hovertemplate="<b>%{x}</b><br>%{customdata:.1f}%<br>$%{y:+.2f}<extra></extra>",
            customdata=_ddf["Daily %"],
        ))
        _fig_pnl.update_layout(
            title="Daily P&L",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="", yaxis_title="P&L ($)", height=240,
        )
        st.plotly_chart(_fig_pnl, use_container_width=True)

        _ddf_disp = _ddf.copy()
        _ddf_disp["P&L"]     = _ddf_disp["P&L"].map(lambda x: f"${x:+.2f}")
        _ddf_disp["Daily %"] = _ddf_disp["Daily %"].map(lambda x: f"{x:+.2f}%")
        _ddf_disp["Bankroll"] = _ddf_disp["Bankroll"].map(lambda x: f"${x:,.2f}")
        _html_table(_ddf_disp)

    st.divider()

    # CLV Analysis
    st.subheader("Closing Line Value (CLV)")
    st.caption(
        "CLV is the single best leading indicator of a real edge. "
        "If you consistently get better prices than where DraftKings closes, "
        "you will be profitable long-term even through short-term losing streaks. "
        "Target: avg CLV > +0.5%, beat close rate > 55%."
    )

    if not _clv_bets:
        st.info(
            "No CLV data yet. The model captures closing DK lines within 3 hours of tip. "
            "Populate by running more cycles and checking back after games close."
        )
    else:
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Avg CLV",         f"{_avg_clv:+.2f}%" if _avg_clv is not None else "—")
        _c2.metric("Beat Closing %",  f"{_beat_close:.0%}" if _beat_close is not None else "—")
        try:
            import numpy as _np
            _xs = [b.clv for b in _clv_bets]
            _ys = [b.ev_pct or 0 for b in _clv_bets]
            _corr = float(_np.corrcoef(_xs, _ys)[0, 1]) if len(_xs) >= 5 else None
            _c3.metric("CLV ↔ EV Corr", f"{_corr:.2f}" if _corr is not None else "—",
                       help=">0.3 means the model's EV estimates predict real line movement.")
        except Exception:
            _c3.metric("CLV ↔ EV Corr", "—")

        _clv_vals = [b.clv for b in _clv_bets]
        _fig_clv = go.Figure()
        _fig_clv.add_trace(go.Histogram(
            x=_clv_vals, nbinsx=20,
            marker_color=["#00e676" if v > 0 else "#ff5252" for v in _clv_vals],
            marker_line_width=0.5, marker_line_color="#0e1117",
        ))
        _fig_clv.add_vline(x=0, line_dash="dash", line_color="#546e7a")
        if _avg_clv is not None:
            _c = "#00e676" if _avg_clv >= 0 else "#ff5252"
            _fig_clv.add_vline(x=_avg_clv, line_dash="dot", line_color=_c,
                               annotation_text=f"avg {_avg_clv:+.2f}%")
        _fig_clv.update_layout(
            title="CLV Distribution — positive = beating the closing line",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="CLV (%)", yaxis_title="# Bets", height=260, showlegend=False,
        )
        st.plotly_chart(_fig_clv, use_container_width=True)

        # Per-sport CLV breakdown — tells you which sports to focus on
        _clv_by_sport: dict = _dd(list)
        for b in _clv_bets:
            _clv_by_sport[b.sport].append(b.clv)
        _sport_clv_rows = [
            {
                "Sport":        sp,
                "Bets":         len(vals),
                "Avg CLV":      f"{sum(vals)/len(vals):+.2f}%",
                "Beat Close":   f"{sum(1 for v in vals if v > 0)/len(vals):.0%}",
                "ROI Signal":   (
                    "Strong edge — prioritise" if sum(vals)/len(vals) > 1.5
                    else "Mild edge — monitor" if sum(vals)/len(vals) > 0.3
                    else "Neutral — hold"      if sum(vals)/len(vals) >= -0.3
                    else "Negative CLV — avoid"
                ),
            }
            for sp, vals in sorted(_clv_by_sport.items()) if vals
        ]
        if _sport_clv_rows:
            st.markdown("#### CLV by Sport — where your real edge lives")
            _html_table(pd.DataFrame(_sport_clv_rows))

    st.divider()

    # Model Calibration
    st.subheader("Model Calibration")
    _cal = _calibrator
    if _cal.n_settled < 10:
        st.info(
            f"Calibration needs {10 - _cal.n_settled} more settled bets. "
            "Picks are auto-tracked — check back after games complete."
        )
    else:
        _bscore = _cal.brier_score or 0
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Quality",       _cal.model_quality_label())
        cm2.metric("Brier Score",   f"{_bscore:.4f}",
                   help="Lower = better. Perfect = 0, coin flip = 0.25.")
        cm3.metric("Log Loss",      f"{_cal.log_loss:.4f}" if _cal.log_loss else "—")
        cm4.metric("Training Bets", _cal.n_settled)
        st.caption(
            "Calibration rescales model probabilities based on observed outcomes. "
            "A well-calibrated model's 60% predictions win ~60% of the time. "
            "Kelly sizing is automatically reduced for miscalibrated segments."
        )
        if _cal.segment_report:
            st.markdown("#### Performance by Segment")
            _html_table(pd.DataFrame(_cal.segment_report))
        if _cal._sport_factor:
            st.markdown("#### Calibration Factors by Sport")
            _sf_rows = [
                {
                    "Sport":    sp,
                    "Factor":   f"{f:.3f}",
                    "Kelly ×":  f"{_cal._kelly_mult.get(sp, 1.0):.2f}",
                    "Meaning":  (
                        f"Over-confident: reducing prob {(1-f)*100:.0f}%" if f < 0.95
                        else f"Under-confident: boosting prob {(f-1)*100:.0f}%" if f > 1.05
                        else "Well-calibrated"
                    ),
                }
                for sp, f in sorted(_cal._sport_factor.items())
            ]
            _html_table(pd.DataFrame(_sf_rows))

    st.divider()

    # P&L breakdown
    _col1, _col2 = st.columns(2)
    with _col1:
        _sp_pnl: dict = _dd(float)
        for b in _real_mbets:
            _sp_pnl[b.sport] += (b.pnl_kelly or 0)
        _fig2 = px.bar(
            pd.DataFrame({"Sport": list(_sp_pnl), "P&L ($)": list(_sp_pnl.values())}),
            x="Sport", y="P&L ($)", color="P&L ($)",
            color_continuous_scale=["#ff5252", "#ffeb3b", "#00e676"],
            title="P&L by Sport",
        )
        _fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(_fig2, use_container_width=True)

    with _col2:
        _td: dict = {}
        for b in _real_mbets:
            _td.setdefault(b.bet_type, {"W": 0, "L": 0})
            _td[b.bet_type]["W" if b.won else "L"] += 1
        _fig3 = px.bar(
            pd.DataFrame([{
                "Type":   t.replace("total_", "").replace("_", " ").title(),
                "Wins":   v["W"],
                "Losses": v["L"],
            } for t, v in _td.items()]),
            x="Type", y=["Wins", "Losses"], barmode="group",
            color_discrete_map={"Wins": "#00e676", "Losses": "#ff5252"},
            title="W/L by Bet Type",
        )
        _fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(_fig3, use_container_width=True)

    # ROI by bet type — critical for knowing where to focus
    st.markdown("#### ROI by Bet Type — focus your bankroll here")
    _roi_rows = []
    for _bt, _td_v in _td.items():
        _bt_bets = [b for b in _real_mbets if b.bet_type == _bt]
        _bt_pnl  = sum(b.pnl_kelly or 0 for b in _bt_bets)
        _bt_wag  = sum(b.stake_kelly or 0 for b in _bt_bets)
        _bt_roi  = (_bt_pnl / _bt_wag * 100) if _bt_wag > 0 else 0
        _bt_ev   = sum(b.ev_pct or 0 for b in _bt_bets) / len(_bt_bets) if _bt_bets else 0
        _roi_rows.append({
            "Bet Type": _bt.replace("total_", "").replace("_", " ").title(),
            "Bets":     len(_bt_bets),
            "W/L":      f"{_td_v['W']}/{_td_v['L']}",
            "Wagered":  f"${_bt_wag:.0f}",
            "P&L":      f"${_bt_pnl:+.2f}",
            "ROI":      f"{_bt_roi:+.1f}%",
            "Avg EV":   f"{_bt_ev:.1f}%",
            "Action":   "Focus here" if _bt_roi > 5 else "Monitor" if _bt_roi > 0 else "Reduce exposure",
        })
    _html_table(pd.DataFrame(_roi_rows))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Bet History
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Bet History":
    from tracker.database import session_scope as _hss
    from tracker.models import Bet as _HBet

    st.title("Bet History")
    _ns = st.session_state.get("auto_settled_count", 0)
    if _ns:
        st.success(f"Auto-settled {_ns} bet(s) this session.")

    with _hss() as _hs:
        _hbets = _hs.query(_HBet).order_by(_HBet.placed_at.desc()).all()
        _hrows = [
            {
                "Date":    b.placed_at.strftime("%m/%d %H:%M") if b.placed_at else "",
                "Sport":   b.sport or "",
                "Game":    f"{b.away_team or ''} @ {b.home_team or ''}",
                "Type":    (b.bet_type or "").replace("total_", "").replace("_", " ").title(),
                "Side":    (b.side or "") + (f" {b.line:+g}" if b.line else ""),
                "Odds":    f"{b.odds:+.0f}" if b.odds is not None else "",
                "Model%":  f"{b.model_prob:.1%}" if b.model_prob is not None else "",
                "EV%":     f"{b.ev_pct:+.2f}%" if b.ev_pct is not None else "",
                "BE%":     f"{_be_win_rate(b.odds):.1%}" if b.odds is not None else "",
                "Stake":   f"${b.stake_kelly:.2f}" if b.stake_kelly is not None else "",
                "CLV":     f"{b.clv:+.2f}%" if b.clv is not None else "—",
                "Result":  ("WIN" if b.won else "LOSS") if b.settled else "OPEN",
                "P&L":     f"${b.pnl_kelly:+.2f}" if b.pnl_kelly is not None else "—",
            }
            for b in _hbets
        ]

    if not _hrows:
        st.info(
            "No model bets yet. The pipeline auto-logs picks each time the page loads "
            "(requires ODDS_API_KEY). Use 'Train Model' in the sidebar first."
        )
        st.stop()

    _hdf = pd.DataFrame(_hrows)
    _hc1, _hc2 = st.columns(2)
    _sp_opts = sorted(_hdf["Sport"].unique().tolist())
    _sp_sel  = _hc1.multiselect("Sport",  _sp_opts, default=_sp_opts)
    _res_sel = _hc2.multiselect("Result", ["WIN", "LOSS", "OPEN"], default=["WIN", "LOSS", "OPEN"])
    _hfilt = _hdf[_hdf["Sport"].isin(_sp_sel) & _hdf["Result"].isin(_res_sel)] if (_sp_sel and _res_sel) else _hdf

    if len(_hfilt) <= 200:
        _html_table(_hfilt)
    else:
        st.dataframe(_hfilt, use_container_width=True, hide_index=True, height=600)

    st.caption(f"{len(_hfilt):,} of {len(_hrows):,} bets shown · auto-tracked & auto-settled")

    # Quick stats on filtered set
    if not _hfilt.empty:
        _settled_f = _hfilt[_hfilt["Result"].isin(["WIN", "LOSS"])]
        if not _settled_f.empty:
            _wr_f = (_settled_f["Result"] == "WIN").mean()
            st.caption(
                f"Hit rate: {_wr_f:.1%} on {len(_settled_f)} settled bets · "
                f"CLV in history tracks whether your edges are real vs variance."
            )
