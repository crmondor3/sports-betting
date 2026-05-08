"""
Streamlit app — DraftKings sports betting model.
Focus: quality over quantity. Top picks as bet cards, not raw tables.
Run: streamlit run app.py
"""
from __future__ import annotations

import sys
from datetime import datetime, date, timezone
from pathlib import Path

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

# ── Auto-settle completed games once per session ──────────────────────────────
if "auto_settled_done" not in st.session_state:
    try:
        from tracker.auto_settle import auto_settle
        n = auto_settle()
        st.session_state["auto_settled_done"] = True
        st.session_state["auto_settled_count"] = n
        if n > 0 and "calibrator" in st.session_state:
            del st.session_state["calibrator"]  # force calibration rebuild
    except Exception as _as_exc:
        st.session_state["auto_settled_done"] = True
        st.session_state["auto_settled_count"] = 0

# ── Load / rebuild calibrator (persists in session, rebuilt after settle) ──────
if "calibrator" not in st.session_state:
    from models.calibrator import ModelCalibrator
    st.session_state["calibrator"] = ModelCalibrator.rebuild_from_db()

_calibrator = st.session_state["calibrator"]

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Card layout */
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

.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 4px;
}
.tag-sport { background:#1e3a5f; color:#90caf9; }
.tag-type  { background:#1b3a2b; color:#81c784; }

.signal-pos { color: #00e676; }
.signal-neg { color: #ff5252; }
.signal-neu { color: #90a4ae; }

[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _conf_color(c: int) -> str:
    if c >= 70: return "#00e676"
    if c >= 55: return "#ffeb3b"
    return "#78909c"

def _conf_label(c: int) -> str:
    if c >= 70: return "HIGH"
    if c >= 55: return "MED"
    return "LOW"

def _ev_color(ev: float) -> str:
    if ev >= 8: return "#00e676"
    if ev >= 5: return "#ffeb3b"
    return "#90a4ae"

def _card_class(conf: int) -> str:
    if conf >= 70: return "bet-card bet-card-high"
    if conf >= 55: return "bet-card bet-card-med"
    return "bet-card bet-card-low"

def _signal_html(signals: list[str]) -> str:
    parts = []
    for s in signals:
        if s.startswith("+"):
            parts.append(f'<span class="signal-pos">&#10003; {s[2:]}</span>')
        elif s.startswith("-"):
            parts.append(f'<span class="signal-neg">&#10007; {s[2:]}</span>')
        else:
            parts.append(f'<span class="signal-neu">~ {s[2:]}</span>')
    return "<br>".join(parts)


# ── Session state defaults ─────────────────────────────────────────────────────

_KELLY = 0.25          # Fixed quarter-Kelly — sized by calibrator confidence
_BASE_BANKROLL = 100.0  # All performance tracking starts from this baseline

def _init_state():
    defaults = {
        "sport":       "All",
        "ev_min":      2.0,
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
    """Fetch all +EV candidates (1% floor) — filtering/ranking done in the UI."""
    from run import run_pipeline
    import config as cfg
    cfg.EV_THRESHOLD           = 0.01   # broad net; UI applies the real floor
    cfg.DEFAULT_KELLY_FRACTION = kelly_frac
    cfg.STARTING_BANKROLL      = bankroll
    cfg.MIN_CONFIDENCE         = 0
    cfg.MAX_PICKS_PER_DAY      = 60
    picks, stats, api_rem = run_pipeline(sport_filter or None)
    return [p.to_dict() for p in picks], stats, api_rem


def _auto_log_model_picks(picks: list[dict], bankroll: float, kelly_frac: float) -> int:
    """
    Idempotently persist all pipeline picks as auto-tracked model bets.
    Skips any game_id+bet_type+side that's already in the DB (manual or auto).
    Returns count of newly inserted rows.
    """
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
    st.caption("Apply instantly — no recalculate needed.")

    st.session_state["sport"] = st.selectbox(
        "Sport", ["All", "NBA", "MLB", "NHL", "NFL"],
        index=["All","NBA","MLB","NHL","NFL"].index(st.session_state["sport"]),
    )
    st.session_state["ev_min"] = st.slider(
        "Min EV %", 1.0, 15.0,
        float(st.session_state["ev_min"]), 0.5, format="%.1f%%",
        help="Only show picks where calibrated edge exceeds this threshold.",
    )
    st.session_state["ev_min_odds"] = st.slider(
        "Exclude odds heavier than",
        min_value=-350, max_value=-100,
        value=int(st.session_state["ev_min_odds"]), step=10, format="%d",
        help="-250 excludes -300, -400 etc. Avoids over-relying on heavy favorites.",
    )
    st.session_state["ev_max_odds"] = st.slider(
        "Exclude longshots above",
        min_value=150, max_value=600,
        value=int(st.session_state["ev_max_odds"]), step=25, format="+%d",
    )
    st.session_state["bankroll"] = st.number_input(
        "Today's bankroll ($)", 10.0,
        value=float(st.session_state["bankroll"]), step=10.0,
        help="Bet stakes are sized as quarter-Kelly × this amount.",
    )

    # Show calibration status
    _cal = _calibrator
    if _cal.n_settled >= 10:
        st.success(f"Model calibrated · {_cal.n_settled} settled bets")
        if _cal.brier_score:
            st.caption(f"Brier: {_cal.brier_score:.4f} · {_cal.model_quality_label()}")
    elif _cal.n_settled > 0:
        st.info(f"Calibrating… {_cal.n_settled}/10 bets settled")
    else:
        st.caption("Calibration active after 10 settled bets.")

    st.divider()
    any_cached = any(is_cached(f"dk_odds_{v}") for v in config.SUPPORTED_SPORTS.values())
    if any_cached:
        st.success("Odds: cached today")
    else:
        st.warning("Odds: not fetched yet")

    if st.button("Force Refresh Odds", use_container_width=True,
                 help="Burns ~4 API requests. Fetches fresh odds from DraftKings."):
        bust_all()
        st.cache_data.clear()
        st.rerun()

    st.caption("Cache refreshes automatically at 7 AM ET daily.")


# ── Active session values ──────────────────────────────────────────────────────
sport_arg     = None if st.session_state["sport"] == "All" else st.session_state["sport"]
bankroll_val  = st.session_state["bankroll"]
ev_min_filter = st.session_state["ev_min"]
odds_lo       = st.session_state["ev_min_odds"]
odds_hi       = st.session_state["ev_max_odds"]

from models.kelly_criterion import kelly_fraction as _kf
from models.ev_calculator import american_to_decimal as _a2d, decimal_to_american as _d2a, calculate_ev as _cev

# ── Pipeline: runs on every page load; cached so subsequent calls are instant ──
# Picks are auto-logged here, not on any specific page, so the bankroll is always
# up to date regardless of which tab the user visits first.
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
        if _all_dicts and "game_id" not in _all_dicts[0]:
            st.cache_data.clear()
            st.rerun()

        # Apply calibration (always fresh — calibrator is session-scoped)
        for _pd in _all_dicts:
            _adj_p  = _calibrator.calibrate_prob(_pd["model_prob"], _pd["sport"], _pd["bet_type"])
            _km     = _calibrator.kelly_multiplier(_pd["sport"])
            _pd["model_prob"]  = _adj_p
            _pd["ev_pct"]      = _cev(_adj_p, _pd["dk_odds"]) * 100
            _pd["kelly_frac"]  = _kf(_adj_p, _pd["dk_odds"], _KELLY * _km)
            _pd["_kelly_mult"] = _km

        # Auto-log all picks to DB (idempotent — skips already-logged game+type+side)
        _n_new = _auto_log_model_picks(_all_dicts, bankroll_val, _KELLY)
        if _n_new > 0:
            # Immediately try to settle any that have already completed
            from tracker.auto_settle import auto_settle as _settle_new
            _settle_new()
            # Force calibrator rebuild with fresh data
            if "calibrator" in st.session_state:
                del st.session_state["calibrator"]

    except Exception as _pe:
        _pipe_error = str(_pe)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EV Picks  — ranked by expected value, Active / Completed tabs
# ══════════════════════════════════════════════════════════════════════════════

if page == "EV Picks":
    st.title("EV Picks")
    st.caption(
        "Building bankroll through expected value — every pick shown has a mathematical edge "
        "on DraftKings. Ranked by EV%, updated daily at 7 AM ET."
    )

    if _pipe_error:
        st.error(f"Pipeline error: {_pipe_error}")
        st.info("Check that ODDS_API_KEY is set in your Streamlit secrets or .env")
        st.stop()

    # ── Split: upcoming vs started ─────────────────────────────────────────────
    _now = datetime.now(timezone.utc)

    def _commence_dt(p: dict) -> datetime:
        try:
            return datetime.fromisoformat(p["commence_time_iso"]).replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.max.replace(tzinfo=timezone.utc)

    # Apply user filters then split by time
    _qualifying = [
        p for p in _all_dicts
        if p["ev_pct"] >= ev_min_filter
        and odds_lo <= p["dk_odds"] <= odds_hi
    ]
    _upcoming   = sorted(
        [p for p in _qualifying if _commence_dt(p) > _now],
        key=lambda p: p["ev_pct"], reverse=True
    )
    _started    = sorted(
        [p for p in _qualifying if _commence_dt(p) <= _now],
        key=lambda p: _commence_dt(p)
    )

    _top_10 = _upcoming[:10]

    # ── KPI row ────────────────────────────────────────────────────────────────
    _kc1, _kc2, _kc3, _kc4 = st.columns(4)
    _kc1.metric("Active Picks", f"{len(_top_10)} / 10")
    _kc2.metric("Completed Today", len(_started))
    _kc3.metric(
        "Avg EV",
        f"{sum(p['ev_pct'] for p in _top_10)/len(_top_10):.1f}%" if _top_10 else "—",
    )
    _kc4.metric("API Requests Left", _api_rem if _api_rem is not None else "cached")

    _tab_active, _tab_done = st.tabs(
        [f"Active ({len(_top_10)})", f"Completed Today ({len(_started)})"]
    )

    # ═══════ ACTIVE TAB ════════════════════════════════════════════════════════
    with _tab_active:
        if not _top_10:
            st.warning(
                "No qualifying EV picks for upcoming games right now. "
                "Try lowering Min EV% or widening the odds range in the sidebar."
            )
            st.info(
                "New games are added daily — check back after 7 AM ET for the next batch."
            )
        else:
            if len(_upcoming) < 10:
                st.info(
                    f"Found {len(_upcoming)} qualifying bet(s) today — "
                    f"{10 - len(_upcoming)} short of the target 10. "
                    "Lower Min EV% to surface more candidates."
                )

            for _i, _p in enumerate(_top_10, 1):
                _ev      = _p["ev_pct"]
                _prob    = _p["model_prob"]
                _conf    = _p["confidence"]
                _ev_col  = _ev_color(_ev)
                _odds_s  = f"+{_p['dk_odds']:.0f}" if _p["dk_odds"] >= 0 else f"{_p['dk_odds']:.0f}"
                _edge    = round((_prob - _p["implied_prob"]) * 100, 1)
                _bteam   = _p.get("bet_team", _p["side"])
                _blbl    = _p["bet_type"].replace("total_", "").replace("_", " ").title()
                _sdisp   = _p["side"].upper() + (f" {_p['line']:+g}" if _p["line"] else "")
                _sigs    = _signal_html(_p.get("signals", []))
                _km      = _p.get("_kelly_mult", 1.0)
                _frac    = _p["kelly_frac"]
                _stake   = round(bankroll_val * _frac, 2)
                _payout  = round((_a2d(_p["dk_odds"]) - 1) * 100, 0)
                _be_dec  = 1 / _prob
                _be_amer = _d2a(_be_dec)
                _be_s    = f"+{_be_amer:.0f}" if _be_amer >= 0 else f"{_be_amer:.0f}"

                _gdt    = _commence_dt(_p)
                _mins   = int((_gdt - _now).total_seconds() / 60)
                if _mins < 60:   _tbcol, _tlbl = "#ff5252", f"{_mins}m to tip"
                elif _mins < 180:_tbcol, _tlbl = "#ffeb3b", f"{_mins//60}h {_mins%60}m to tip"
                else:            _tbcol, _tlbl = "#00e676", f"{_mins//60}h to tip"

                _card_cls = ("bet-card bet-card-high" if _ev >= 8
                             else "bet-card bet-card-med" if _ev >= 5
                             else "bet-card bet-card-low")

                st.markdown(f"""
<div class="{_card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <span style="background:#1e3a5f;color:#90caf9;border-radius:50%;width:28px;height:28px;
                   display:inline-flex;align-items:center;justify-content:center;
                   font-weight:800;font-size:0.9rem;flex-shrink:0;">#{_i}</span>
      <span class="tag tag-sport">{_p['sport']}</span>
      <span class="tag tag-type">{_blbl}</span>
      <span style="font-size:1.2rem;font-weight:800;color:#ffffff;margin-left:6px;">{_p['game']}</span>
    </div>
    <div style="display:flex;gap:10px;align-items:center;">
      <span style="background:{_tbcol}22;color:{_tbcol};border:1px solid {_tbcol};
                   padding:2px 10px;border-radius:20px;font-size:0.8rem;font-weight:600;">{_tlbl}</span>
      <span style="font-size:2rem;font-weight:900;color:{_ev_col};">{_ev:+.1f}% EV</span>
    </div>
  </div>

  <div style="margin-top:10px;padding:8px 12px;background:#0d1520;border-radius:6px;font-size:0.9rem;color:#cfd8dc;">
    Betting <b style="color:#fff;">{_bteam}</b> ({_blbl} {_sdisp} @ <b style="color:{_ev_col};">{_odds_s}</b>).
    Model: <b style="color:{_ev_col};">{_prob:.1%}</b> win vs book's {_p['implied_prob']:.1%} implied.
    Edge: <b style="color:{_ev_col};">+{_edge:.1f}%</b>
  </div>

  <div style="margin-top:14px;display:flex;gap:28px;flex-wrap:wrap;">
    <div>
      <div style="color:#78909c;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Bet</div>
      <div style="font-size:1.25rem;font-weight:800;color:#ffffff;">{_bteam}</div>
      <div style="font-size:0.85rem;color:#b0bec5;">{_blbl} &nbsp;{_sdisp}</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">EV</div>
      <div style="font-size:1.6rem;font-weight:900;color:{_ev_col};">{_ev:+.1f}%</div>
      <div style="font-size:0.8rem;color:#b0bec5;">expected value</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Odds</div>
      <div style="font-size:1.4rem;font-weight:800;color:#ffffff;">{_odds_s}</div>
      <div style="font-size:0.8rem;color:#b0bec5;">+${_payout:.0f} per $100</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Model vs Book</div>
      <div style="font-size:1.15rem;font-weight:700;color:{_ev_col};">{_prob:.1%}</div>
      <div style="font-size:0.85rem;color:#b0bec5;">implied: {_p['implied_prob']:.1%}</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Stake (Kelly)</div>
      <div style="font-size:1.25rem;font-weight:800;color:#ffffff;">${_stake:.2f}</div>
      <div style="font-size:0.85rem;color:#b0bec5;">{_frac:.1%} · {_km:.0%} confidence</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Bet-To Line</div>
      <div style="font-size:1.15rem;font-weight:700;color:#ff5252;">{_be_s}</div>
      <div style="font-size:0.8rem;color:#b0bec5;">no edge below this</div>
    </div>
    <div>
      <div style="color:#78909c;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Confidence</div>
      <div style="font-size:1.15rem;font-weight:700;color:{_conf_color(_conf)};">
        {_conf}/100 <span style="font-size:0.8rem;color:#b0bec5;">({_conf_label(_conf)})</span>
      </div>
    </div>
  </div>

  <div style="margin-top:14px;border-top:1px solid #1e2e42;padding-top:10px;
              font-size:0.85rem;line-height:1.9;color:#b0bec5;">
    <span style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;color:#546e7a;">
      Signals for {_bteam}</span><br>
    {_sigs if _sigs else '<span style="color:#546e7a;">No signal data — cold-start model</span>'}
  </div>
</div>
""", unsafe_allow_html=True)

        # All picks are auto-tracked — no manual log needed

    # ═══════ COMPLETED TODAY TAB ════════════════════════════════════════════════
    with _tab_done:
        if not _started:
            st.info("No completed games from today's picks yet.")
        else:
            # Look up W/L for each started pick from the DB
            from tracker.database import session_scope as _ss
            from tracker.models import Bet as _BM
            _started_game_ids = list({p["game_id"] for p in _started})
            with _ss() as _s:
                _db_bets = {
                    (b.game_id, b.bet_type, b.side): {
                        "settled": b.settled, "won": b.won,
                        "pnl": b.pnl_kelly, "id": b.id,
                    }
                    for b in _s.query(_BM).filter(_BM.game_id.in_(_started_game_ids)).all()
                }

            _n_won  = sum(1 for v in _db_bets.values() if v["settled"] and v["won"])
            _n_lost = sum(1 for v in _db_bets.values() if v["settled"] and not v["won"])
            _n_open = sum(1 for v in _db_bets.values() if not v["settled"])
            if _n_won or _n_lost:
                _rc1, _rc2, _rc3 = st.columns(3)
                _rc1.metric("Wins", _n_won)
                _rc2.metric("Losses", _n_lost)
                _rc3.metric("Pending", _n_open)
            st.caption("Games that have started. Results auto-settle ~4 hours after tip.")

            for _p in _started:
                _ev      = _p["ev_pct"]
                _ev_col  = _ev_color(_ev)
                _bteam   = _p.get("bet_team", _p["side"])
                _blbl    = _p["bet_type"].replace("total_", "").replace("_", " ").title()
                _sdisp   = _p["side"].upper() + (f" {_p['line']:+g}" if _p["line"] else "")
                _odds_s  = f"+{_p['dk_odds']:.0f}" if _p["dk_odds"] >= 0 else f"{_p['dk_odds']:.0f}"
                _db_key  = (_p["game_id"], _p["bet_type"], _p["side"])
                _db_rec  = _db_bets.get(_db_key)

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
                        'padding:2px 12px;border-radius:20px;font-size:0.8rem;">In Progress</span>'
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
<div class="bet-card" style="opacity:0.85;{_card_bg}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <span class="tag tag-sport">{_p['sport']}</span>
      <span class="tag tag-type">{_blbl}</span>
      <span style="font-size:1.1rem;font-weight:700;color:#cfd8dc;">{_p['game']}</span>
    </div>
    {_result_badge}
  </div>
  <div style="margin-top:10px;font-size:0.9rem;color:#b0bec5;">
    <b style="color:#fff;">{_bteam}</b> &nbsp;{_blbl} {_sdisp} &nbsp;
    <b style="color:{_ev_col};">{_odds_s}</b> &nbsp;·&nbsp;
    EV at post: <b style="color:{_ev_col};">{_ev:+.1f}%</b> &nbsp;·&nbsp;
    Model: {_p['model_prob']:.1%}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Bankroll
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Bankroll":
    from collections import defaultdict as _dd
    from tracker.database import session_scope as _bss
    from tracker.models import Bet as _BetM

    st.title("Bankroll")
    st.caption(
        f"Model performance tracking from a ${_BASE_BANKROLL:.0f} starting bankroll. "
        "Grows via quarter-Kelly EV bets, calibrated by historical accuracy."
    )

    with _bss() as _bs:
        _mbets = (
            _bs.query(_BetM)
            .filter(_BetM.settled == True)
            .all()
        )
    # Sort by effective date: settled_at → placed_at → commence_time → epoch
    def _bet_date(b) -> datetime:
        return b.settled_at or b.placed_at or b.commence_time or datetime(2000, 1, 1)
    _mbets = sorted(_mbets, key=_bet_date)

    # ── Header KPIs ────────────────────────────────────────────────────────────
    _total_pnl  = sum(b.pnl_kelly or 0 for b in _mbets)
    _model_br   = _BASE_BANKROLL + _total_pnl
    _total_ret  = (_model_br - _BASE_BANKROLL) / _BASE_BANKROLL * 100
    _wins_m     = sum(1 for b in _mbets if b.won)
    _hit_m      = _wins_m / len(_mbets) if _mbets else 0
    _avg_ev_m   = sum(b.ev_pct or 0 for b in _mbets) / len(_mbets) if _mbets else 0
    _theo_ev    = sum((b.ev_pct or 0) / 100 * (b.stake_kelly or 0) for b in _mbets)
    _open_count = 0
    with _bss() as _bs2:
        _open_count = _bs2.query(_BetM).filter(_BetM.settled == False).count()

    _br_col = "#00e676" if _model_br >= _BASE_BANKROLL else "#ff5252"
    st.markdown(f"""
<div style="background:#0d1520;border-radius:12px;padding:20px 24px;margin-bottom:20px;
            display:flex;gap:32px;flex-wrap:wrap;align-items:center;">
  <div>
    <div style="color:#546e7a;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Model Bankroll</div>
    <div style="font-size:2.4rem;font-weight:900;color:{_br_col};">${_model_br:,.2f}</div>
    <div style="color:#78909c;font-size:0.8rem;">started at ${_BASE_BANKROLL:.0f}</div>
  </div>
  <div>
    <div style="color:#546e7a;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Total Return</div>
    <div style="font-size:1.8rem;font-weight:800;color:{_br_col};">{_total_ret:+.1f}%</div>
  </div>
  <div>
    <div style="color:#546e7a;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Hit Rate</div>
    <div style="font-size:1.6rem;font-weight:800;color:#cfd8dc;">{_hit_m:.1%}</div>
    <div style="color:#78909c;font-size:0.8rem;">{_wins_m}W – {len(_mbets)-_wins_m}L</div>
  </div>
  <div>
    <div style="color:#546e7a;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Avg EV / Pick</div>
    <div style="font-size:1.6rem;font-weight:800;color:#cfd8dc;">{_avg_ev_m:+.1f}%</div>
  </div>
  <div>
    <div style="color:#546e7a;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Settled / Open</div>
    <div style="font-size:1.6rem;font-weight:800;color:#cfd8dc;">{len(_mbets)} / {_open_count}</div>
  </div>
  <div>
    <div style="color:#546e7a;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;">Theoretical EV</div>
    <div style="font-size:1.4rem;font-weight:700;color:#90caf9;">${_theo_ev:+.2f}</div>
    <div style="color:#78909c;font-size:0.8rem;">actual: ${_total_pnl:+.2f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    if not _mbets:
        st.info("No settled model bets yet. Results appear here as games complete (~4h after tip).")
        st.stop()

    # ── Daily P&L + running bankroll ───────────────────────────────────────────
    st.subheader("Daily Performance")
    _daily_pnl: dict = _dd(float)
    _daily_bets: dict = _dd(list)
    for b in _mbets:
        _d = _bet_date(b).date()
        _daily_pnl[_d] += (b.pnl_kelly or 0)
        _daily_bets[_d].append(b)

    _running = _BASE_BANKROLL
    _daily_rows = []
    for _dt in sorted(_daily_pnl):
        _day_p   = _daily_pnl[_dt]
        _day_br  = _running  # bankroll at start of day
        _day_pct = _day_p / _day_br * 100 if _day_br > 0 else 0
        _running += _day_p
        _day_bets = _daily_bets[_dt]
        _day_w    = sum(1 for b in _day_bets if b.won)
        _daily_rows.append({
            "Date":      _dt,
            "Bets":      len(_day_bets),
            "W":         _day_w,
            "L":         len(_day_bets) - _day_w,
            "P&L":       round(_day_p, 2),
            "Daily %":   round(_day_pct, 2),
            "Bankroll":  round(_running, 2),
        })

    if _daily_rows:
        _ddf = pd.DataFrame(_daily_rows)

        # Running bankroll line chart
        _fig_br = go.Figure()
        _fig_br.add_hline(y=_BASE_BANKROLL, line_dash="dash", line_color="#546e7a",
                          annotation_text=f"Start ${_BASE_BANKROLL:.0f}")
        _fig_br.add_trace(go.Scatter(
            x=_ddf["Date"], y=_ddf["Bankroll"], mode="lines+markers",
            name="Bankroll", line=dict(color="#00e676", width=3),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.07)",
            hovertemplate="<b>%{x}</b><br>$%{y:.2f}<extra></extra>",
        ))
        _fig_br.update_layout(
            title="Model Bankroll Growth (from $100 start)",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            xaxis_title="", yaxis_title="Bankroll ($)",
            showlegend=False, height=320,
        )
        st.plotly_chart(_fig_br, use_container_width=True)

        # Daily P&L bars
        _fig_pnl = go.Figure()
        _fig_pnl.add_trace(go.Bar(
            x=_ddf["Date"], y=_ddf["P&L"],
            marker_color=["#00e676" if v >= 0 else "#ff5252" for v in _ddf["P&L"]],
            hovertemplate="<b>%{x}</b><br>%{customdata:.1f}%<br>$%{y:+.2f}<extra></extra>",
            customdata=_ddf["Daily %"],
        ))
        _fig_pnl.update_layout(
            title="Daily P&L",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            xaxis_title="", yaxis_title="P&L ($)", height=260,
        )
        st.plotly_chart(_fig_pnl, use_container_width=True)

        # Daily table
        _ddf_disp = _ddf.copy()
        _ddf_disp["P&L"]     = _ddf_disp["P&L"].map(lambda x: f"${x:+.2f}")
        _ddf_disp["Daily %"] = _ddf_disp["Daily %"].map(lambda x: f"{x:+.2f}%")
        _ddf_disp["Bankroll"]= _ddf_disp["Bankroll"].map(lambda x: f"${x:,.2f}")
        st.dataframe(_ddf_disp, use_container_width=True, hide_index=True)

    st.divider()

    # ── Model Calibration Report ───────────────────────────────────────────────
    st.subheader("Model Calibration & Learning")
    _cal = _calibrator
    if _cal.n_settled < 10:
        st.info(f"Calibration needs {10 - _cal.n_settled} more settled bets to activate. "
                "All model picks are auto-tracked — just check back after games complete.")
    else:
        _bscore = _cal.brier_score or 0
        _lloss  = _cal.log_loss or 0
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Calibration Quality", _cal.model_quality_label())
        cm2.metric("Brier Score", f"{_bscore:.4f}",
                   help="Lower = better. Perfect = 0, coin flip = 0.25.")
        cm3.metric("Log Loss", f"{_lloss:.4f}",
                   help="Lower = better probability estimates.")
        cm4.metric("Bets in Training Set", _cal.n_settled)

        st.caption(
            "The model learns from every settled bet. Calibration factors shrink overconfident "
            "predictions and amplify underestimated ones. Kelly confidence (shown on each card) "
            "scales stake size based on how well the model adds value above the book price."
        )

        if _cal.segment_report:
            st.markdown("#### Segment Performance")
            _seg_df = pd.DataFrame(_cal.segment_report)
            st.dataframe(_seg_df, use_container_width=True, hide_index=True)

        # Sport calibration factors
        if _cal._sport_factor:
            st.markdown("#### Calibration Factors by Sport")
            _sf_rows = []
            for _sp, _f in sorted(_cal._sport_factor.items()):
                _km = _cal._kelly_mult.get(_sp, 1.0)
                _sf_rows.append({
                    "Sport":         _sp,
                    "Cal Factor":    f"{_f:.3f}",
                    "Kelly ×":       f"{_km:.2f}",
                    "Interpretation": (
                        f"⬆ inflating prob +{(_f-1)*100:.0f}%" if _f > 1.05
                        else f"⬇ deflating prob {(_f-1)*100:.0f}%" if _f < 0.95
                        else "✓ well-calibrated"
                    ),
                })
            st.dataframe(pd.DataFrame(_sf_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── P&L breakdown ──────────────────────────────────────────────────────────
    _col1, _col2 = st.columns(2)
    with _col1:
        _sp_pnl = _dd(float)
        for b in _mbets:
            _sp_pnl[b.sport] += (b.pnl_kelly or 0)
        _fig2 = px.bar(
            pd.DataFrame({"Sport": list(_sp_pnl), "P&L ($)": list(_sp_pnl.values())}),
            x="Sport", y="P&L ($)", color="P&L ($)",
            color_continuous_scale=["#ff5252", "#ffeb3b", "#00e676"],
            title="P&L by Sport",
        )
        _fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(_fig2, use_container_width=True)

    with _col2:
        _td: dict = {}
        for b in _mbets:
            _td.setdefault(b.bet_type, {"W": 0, "L": 0})
            _td[b.bet_type]["W" if b.won else "L"] += 1
        _fig3 = px.bar(
            pd.DataFrame([{"Type": t.replace("total_","").replace("_"," ").title(),
                           "Wins": v["W"], "Losses": v["L"]} for t, v in _td.items()]),
            x="Type", y=["Wins", "Losses"], barmode="group",
            color_discrete_map={"Wins": "#00e676", "Losses": "#ff5252"},
            title="W/L by Bet Type",
        )
        _fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(_fig3, use_container_width=True)


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
                "Type":    (b.bet_type or "").replace("total_","").replace("_"," ").title(),
                "Side":    (b.side or "") + (f" {b.line:+g}" if b.line else ""),
                "Odds":    f"{b.odds:+.0f}" if b.odds is not None else "",
                "Model%":  f"{b.model_prob:.1%}" if b.model_prob is not None else "",
                "EV%":     f"{b.ev_pct:+.2f}%" if b.ev_pct is not None else "",
                "Stake $": f"${b.stake_kelly:.2f}" if b.stake_kelly is not None else "",
                "Result":  ("WIN" if b.won else "LOSS") if b.settled else "OPEN",
                "P&L":     f"${b.pnl_kelly:+.2f}" if b.pnl_kelly is not None else "—",
            }
            for b in _hbets
        ]

    if not _hrows:
        st.info("No model bets tracked yet — picks are auto-logged when the pipeline runs.")
        st.stop()

    _hdf = pd.DataFrame(_hrows)
    _hc1, _hc2 = st.columns(2)
    _sp_opts = _hdf["Sport"].unique().tolist()
    _sp_sel  = _hc1.multiselect("Sport", _sp_opts, default=_sp_opts)
    _res_sel = _hc2.multiselect("Result", ["WIN","LOSS","OPEN"], default=["WIN","LOSS","OPEN"])
    _hfilt   = _hdf[_hdf["Sport"].isin(_sp_sel) & _hdf["Result"].isin(_res_sel)] if (_sp_sel and _res_sel) else _hdf
    st.dataframe(_hfilt, use_container_width=True, hide_index=True)
    st.caption(f"{len(_hfilt)} of {len(_hrows)} model bets · all auto-tracked, auto-settled")
