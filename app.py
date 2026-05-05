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
    except Exception as _as_exc:
        st.session_state["auto_settled_done"] = True
        st.session_state["auto_settled_count"] = 0

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

def _init_state():
    defaults = {
        "sport":      "All",
        "ev_min":     2.0,     # min EV% to show
        "ev_min_odds": -250,   # exclude bigger favorites than this
        "ev_max_odds":  400,   # exclude longshots beyond this
        "kelly":      0.25,
        "bankroll":   100.0,
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
        ["EV Picks", "Bankroll", "Bet History", "Settle Bets"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Filters")
    st.caption("All filters apply instantly to the current data.")

    st.session_state["sport"] = st.selectbox(
        "Sport", ["All", "NBA", "MLB", "NHL", "NFL"],
        index=["All","NBA","MLB","NHL","NFL"].index(st.session_state["sport"]),
    )
    st.session_state["ev_min"] = st.slider(
        "Min EV %", 1.0, 15.0,
        float(st.session_state["ev_min"]), 0.5, format="%.1f%%",
        help="Only show picks where the model's edge exceeds this threshold.",
    )
    st.session_state["ev_min_odds"] = st.slider(
        "Exclude odds heavier than",
        min_value=-350, max_value=-100,
        value=int(st.session_state["ev_min_odds"]), step=10, format="%d",
        help="e.g. -250 excludes -300, -400. Avoids massive favorites.",
    )
    st.session_state["ev_max_odds"] = st.slider(
        "Exclude longshots above",
        min_value=150, max_value=600,
        value=int(st.session_state["ev_max_odds"]), step=25, format="+%d",
    )
    st.session_state["kelly"] = st.slider(
        "Kelly Fraction", 0.05, 1.0,
        float(st.session_state["kelly"]), 0.05, format="%.0f%%",
    )
    st.session_state["bankroll"] = st.number_input(
        "Bankroll ($)", 10.0,
        value=float(st.session_state["bankroll"]), step=10.0,
    )

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
sport_arg    = None if st.session_state["sport"] == "All" else st.session_state["sport"]
kelly        = st.session_state["kelly"]
bankroll_val = st.session_state["bankroll"]
ev_min_filter = st.session_state["ev_min"]
odds_lo       = st.session_state["ev_min_odds"]
odds_hi       = st.session_state["ev_max_odds"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EV Picks  — ranked by expected value, Active / Completed tabs
# ══════════════════════════════════════════════════════════════════════════════

if page == "EV Picks":
    st.title("EV Picks")
    st.caption(
        "Building bankroll through expected value — every pick shown has a mathematical edge "
        "on DraftKings. Ranked by EV%, updated daily at 7 AM ET."
    )

    from models.kelly_criterion import kelly_fraction as _kf
    from models.ev_calculator import american_to_decimal as _a2d, decimal_to_american as _d2a

    with st.spinner("Fetching picks (odds cached daily)…"):
        try:
            _all_dicts, _ev_stats, _api_rem = _run_ev_pipeline(sport_arg, kelly, bankroll_val)
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.info("Check that ODDS_API_KEY is set in your Streamlit secrets or .env")
            st.stop()

    if _all_dicts and "game_id" not in _all_dicts[0]:
        st.cache_data.clear()
        st.rerun()

    # ── Auto-log ALL pipeline picks as model-tracked bets (idempotent) ─────────
    if "model_picks_logged" not in st.session_state:
        _n_new = _auto_log_model_picks(_all_dicts, bankroll_val, kelly)
        if _n_new > 0:
            # Settle any completed games that were just logged
            from tracker.auto_settle import auto_settle as _settle_new
            _settle_new()
        st.session_state["model_picks_logged"] = True

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
                _edge    = _p.get("edge_pct", round((_prob - _p["implied_prob"]) * 100, 1))
                _bteam   = _p.get("bet_team", _p["side"])
                _blbl    = _p["bet_type"].replace("total_", "").replace("_", " ").title()
                _sdisp   = _p["side"].upper() + (f" {_p['line']:+g}" if _p["line"] else "")
                _sigs    = _signal_html(_p.get("signals", []))
                _frac    = _kf(_prob, _p["dk_odds"], kelly)
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
      <div style="font-size:0.85rem;color:#b0bec5;">{_frac:.1%} of ${bankroll_val:,.0f}</div>
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

        # ── Log bet ────────────────────────────────────────────────────────────
        if _top_10:
            st.divider()
            with st.expander("Log a bet to tracker"):
                from bets.edge_detector import EdgeBet
                _log_labels = [
                    f"#{_i}  {_d['game']}  —  "
                    f"{_d['bet_type'].replace('total_','').title()} {_d['side'].upper()}"
                    + (f" {_d['line']:+g}" if _d['line'] else "")
                    + f"  ({'+' if _d['dk_odds']>=0 else ''}{int(_d['dk_odds'])})"
                    for _i, _d in enumerate(_top_10, 1)
                ]
                _log_idx = st.selectbox(
                    "Pick to log", range(len(_top_10)),
                    format_func=lambda x: _log_labels[x], key="ev_log_sel"
                )
                _sel = _top_10[_log_idx]
                _lc1, _lc2 = st.columns(2)
                _log_odds  = _lc1.number_input(
                    "Odds placed at (American)", value=int(_sel["dk_odds"]),
                    step=1, key="ev_odds_input"
                )
                _log_frac  = _kf(_sel["model_prob"], float(_log_odds), kelly)
                _log_stake = _lc2.number_input(
                    "Stake ($)", value=round(bankroll_val * _log_frac, 2),
                    min_value=0.01, step=0.50, key="ev_stake_input"
                )
                _log_tracker = BankrollTracker(starting_bankroll=bankroll_val)
                if st.button("Log Bet", type="primary", key="ev_log_btn"):
                    _eb = EdgeBet(
                        sport=_sel["sport"], game_id=_sel["game_id"],
                        home_team=_sel["home_team"], away_team=_sel["away_team"],
                        commence_time=datetime.fromisoformat(_sel["commence_time_iso"]),
                        bet_type=_sel["bet_type"], side=_sel["side"], line=_sel["line"],
                        best_book="draftkings", best_odds=float(_log_odds),
                        model_prob=_sel["model_prob"], implied_prob=_sel["implied_prob"],
                        ev_pct=_sel["ev_pct"], kelly_frac=_log_frac,
                        recommended_stake=_log_stake, is_sharp_book=False,
                    )
                    _bid = _log_tracker.log_bet(_eb, bankroll=bankroll_val)
                    _os  = f"+{int(_log_odds)}" if _log_odds >= 0 else str(int(_log_odds))
                    st.success(f"Logged Bet #{_bid} — {_sel['bet_team']} {_os} · Stake ${_log_stake:.2f}")
                    st.rerun()

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
    st.title("Bankroll Performance")
    tracker = BankrollTracker(starting_bankroll=bankroll_val)
    stats = tracker.get_stats()

    start = stats.get("starting_bankroll", bankroll_val)
    curr_k = stats.get("current_bankroll_kelly", bankroll_val)
    curr_f = stats.get("current_bankroll_flat", bankroll_val)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bankroll (Kelly)", f"${curr_k:,.2f}",
              delta=f"${curr_k - start:+.2f}" if stats["total_bets"] > 0 else None)
    c2.metric("Bankroll (Flat)",  f"${curr_f:,.2f}",
              delta=f"${curr_f - start:+.2f}" if stats["total_bets"] > 0 else None)
    c3.metric("Hit Rate",  stats["hit_rate"])
    c4.metric("ROI (Kelly)", stats["roi_kelly"])
    c5.metric("Avg CLV",   stats["avg_clv"])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Starting Bankroll", f"${start:,.2f}")
    col_b.metric("Settled Bets", stats["total_bets"])
    col_c.metric("Open Bets", stats.get("open_bets", 0))

    if stats["total_bets"] == 0:
        st.divider()
        st.info("No settled bets yet — log a pick on Today's Picks and settle it here to see your chart.")
        st.stop()

    st.divider()

    from tracker.database import session_scope
    from tracker.models import Bet
    with session_scope() as s:
        bets = s.query(Bet).filter(Bet.settled == True).order_by(Bet.settled_at).all()

    # ── EV Profile ─────────────────────────────────────────────────────────────
    st.subheader("EV Profile")
    st.caption(
        "Tracking every pick the model surfaces — win rate and edge efficiency across all tracked bets."
    )

    _model_bets  = [b for b in bets if b.auto_tracked]
    _placed_bets = [b for b in bets if not b.auto_tracked]

    def _ev_stats(bet_list):
        if not bet_list:
            return dict(theo=0.0, pnl=0.0, eff=0.0, avg_ev=0.0, staked=0.0,
                        wins=0, total=0, hit="—")
        theo   = sum((b.ev_pct or 0) / 100 * (b.stake_kelly or 0) for b in bet_list)
        pnl    = sum(b.pnl_kelly or 0 for b in bet_list)
        staked = sum(b.stake_kelly or 0 for b in bet_list)
        wins   = sum(1 for b in bet_list if b.won)
        return dict(
            theo=theo, pnl=pnl,
            eff=(pnl / theo * 100) if theo > 0 else 0.0,
            avg_ev=sum(b.ev_pct or 0 for b in bet_list) / len(bet_list),
            staked=staked, wins=wins, total=len(bet_list),
            hit=f"{wins / len(bet_list):.1%}",
        )

    _ms = _ev_stats(_model_bets)
    _ps = _ev_stats(_placed_bets)

    _ev_col_m = "#00e676" if _ms["eff"] >= 80 else "#ffeb3b" if _ms["eff"] >= 40 else "#ff5252"
    _ev_col_p = "#00e676" if _ps["eff"] >= 80 else "#ffeb3b" if _ps["eff"] >= 40 else "#ff5252"

    st.markdown("#### Model Picks (all auto-tracked)")
    ep1, ep2, ep3, ep4, ep5 = st.columns(5)
    ep1.metric("Settled", _ms["total"])
    ep2.metric("Win Rate", _ms["hit"])
    ep3.metric("Theoretical EV", f"${_ms['theo']:+.2f}",
               help="EV% × Kelly stake at bet time.")
    ep4.metric("Actual P&L", f"${_ms['pnl']:+.2f}",
               delta=f"${_ms['pnl'] - _ms['theo']:+.2f} vs edge")
    ep5.metric("EV Efficiency", f"{_ms['eff']:.0f}%",
               help=">100% = running above model expectation.")

    _mc = max(0.0, min(_ms["eff"], 200.0))
    st.markdown(f"""
<div style="margin:6px 0 14px;background:#0d1520;border-radius:6px;overflow:hidden;height:8px;">
  <div style="width:{_mc/2:.1f}%;height:8px;background:{_ev_col_m};border-radius:6px;"></div>
</div>
<div style="font-size:0.8rem;color:#546e7a;margin-bottom:16px;">
  {_ms['total']} settled picks · avg EV <b style="color:#cfd8dc;">{_ms['avg_ev']:+.1f}%</b>
  · total staked <b style="color:#cfd8dc;">${_ms['staked']:,.2f}</b>
</div>
""", unsafe_allow_html=True)

    if _placed_bets:
        st.markdown("#### Manually Placed Bets")
        fp1, fp2, fp3, fp4, fp5 = st.columns(5)
        fp1.metric("Settled", _ps["total"])
        fp2.metric("Win Rate", _ps["hit"])
        fp3.metric("Theoretical EV", f"${_ps['theo']:+.2f}")
        fp4.metric("Actual P&L", f"${_ps['pnl']:+.2f}",
                   delta=f"${_ps['pnl'] - _ps['theo']:+.2f} vs edge")
        fp5.metric("EV Efficiency", f"{_ps['eff']:.0f}%")
        _pc = max(0.0, min(_ps["eff"], 200.0))
        st.markdown(f"""
<div style="margin:6px 0 14px;background:#0d1520;border-radius:6px;overflow:hidden;height:8px;">
  <div style="width:{_pc/2:.1f}%;height:8px;background:{_ev_col_p};border-radius:6px;"></div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # Running bankroll chart — starts at bankroll_val, each settled bet moves it
    running_k, running_f = bankroll_val, bankroll_val
    rows = [{"Date": None, "Kelly": bankroll_val, "Flat": bankroll_val, "Label": "Start"}]
    for b in bets:
        running_k += b.pnl_kelly or 0
        running_f += b.pnl_flat or 0
        rows.append({
            "Date": b.settled_at,
            "Kelly": round(running_k, 2),
            "Flat": round(running_f, 2),
            "Label": f"#{b.id} {'W' if b.won else 'L'} {b.sport}",
        })
    cdf = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Kelly"], mode="lines+markers",
                             name="Kelly", line=dict(color="#00e676", width=2)))
    fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Flat"], mode="lines+markers",
                             name="Flat", line=dict(color="#40c4ff", width=2, dash="dot")))
    fig.add_hline(y=bankroll_val, line_dash="dash", line_color="#555",
                  annotation_text="Starting bankroll")
    fig.update_layout(
        title="Bankroll Over Time — Kelly vs Flat",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
        legend=dict(bgcolor="#1e2a3a"), xaxis_title="Date", yaxis_title="$",
    )
    st.plotly_chart(fig, use_container_width=True)

    # P&L by sport + bet type
    col1, col2 = st.columns(2)
    with col1:
        sport_pnl = {}
        for b in bets:
            sport_pnl.setdefault(b.sport, 0)
            sport_pnl[b.sport] += b.pnl_kelly or 0
        fig2 = px.bar(
            pd.DataFrame({"Sport": list(sport_pnl), "P&L ($)": list(sport_pnl.values())}),
            x="Sport", y="P&L ($)", color="P&L ($)",
            color_continuous_scale=["#ff5252", "#ffeb3b", "#00e676"],
            title="P&L by Sport (Kelly)",
        )
        fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        type_data: dict[str, dict] = {}
        for b in bets:
            type_data.setdefault(b.bet_type, {"W": 0, "L": 0})
            type_data[b.bet_type]["W" if b.won else "L"] += 1
        tdf = pd.DataFrame([{"Type": t, "Wins": v["W"], "Losses": v["L"]} for t, v in type_data.items()])
        fig3 = px.bar(tdf, x="Type", y=["Wins", "Losses"], barmode="group",
                      color_discrete_map={"Wins": "#00e676", "Losses": "#ff5252"},
                      title="W/L by Bet Type")
        fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    if st.button("Export Bets to CSV"):
        path = tracker.export_csv()
        with open(path, "rb") as f:
            st.download_button("Download", f, file_name="bets_export.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Bet History
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Bet History":
    st.title("Bet History")

    n_settled = st.session_state.get("auto_settled_count", 0)
    if n_settled:
        st.success(f"Auto-settled {n_settled} bet(s) from completed games this session.")

    from tracker.database import session_scope
    from tracker.models import Bet

    # Load all attributes inside the session to avoid detached-instance issues
    with session_scope() as s:
        all_bets = s.query(Bet).order_by(Bet.placed_at.desc()).all()
        rows = [
            {
                "ID": b.id,
                "Date": b.placed_at.strftime("%Y-%m-%d %H:%M") if b.placed_at else "",
                "Sport": b.sport or "",
                "Game": f"{b.away_team or ''} @ {b.home_team or ''}",
                "Type": (b.bet_type or "").replace("total_", "").replace("_", " ").title(),
                "Side": (b.side or "") + (f" {b.line:+g}" if b.line else ""),
                "Odds": f"{b.odds:+.0f}" if b.odds is not None else "",
                "Model%": f"{b.model_prob:.1%}" if b.model_prob is not None else "",
                "EV%": f"{b.ev_pct:+.2f}%" if b.ev_pct is not None else "",
                "Stake": f"${b.stake_kelly:.2f}" if b.stake_kelly is not None else "",
                "Result": ("WIN" if b.won else "LOSS") if b.settled else "OPEN",
                "P&L": f"${b.pnl_kelly:+.2f}" if b.pnl_kelly is not None else "—",
                "CLV": f"{b.clv:+.2f}%" if b.clv is not None else "—",
            }
            for b in all_bets
        ]

    if not rows:
        st.info("No bets logged yet. Log a pick from Today's Picks to get started.")
        st.stop()

    df = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    sp_opts = df["Sport"].unique().tolist()
    sp = c1.multiselect("Sport", sp_opts, default=sp_opts)
    res = c2.multiselect("Result", ["WIN", "LOSS", "OPEN"], default=["WIN", "LOSS", "OPEN"])

    filtered = df[df["Sport"].isin(sp) & df["Result"].isin(res)] if (sp and res) else df

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered)} of {len(rows)} bets")

    st.divider()
    with st.expander("Remove a bet"):
        tracker_del = BankrollTracker(starting_bankroll=bankroll_val)
        bet_options = {
            f"#{r['ID']}  {r['Date']}  {r['Game']}  {r['Type']} {r['Side']}  {r['Odds']}  [{r['Result']}]": r["ID"]
            for r in rows
        }
        if not bet_options:
            st.info("No bets to remove.")
        else:
            sel_label = st.selectbox("Select bet to remove", list(bet_options.keys()), key="del_bet_sel")
            sel_id = bet_options[sel_label]
            confirmed = st.checkbox(f"I confirm I want to permanently delete Bet #{sel_id}", key="del_confirm")
            if st.button("Delete Bet", type="primary", disabled=not confirmed, key="del_btn"):
                tracker_del.delete_bet(sel_id)
                st.success(f"Bet #{sel_id} deleted.")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Settle Bets
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Settle Bets":
    st.title("Settle Open Bets")
    tracker = BankrollTracker(starting_bankroll=bankroll_val)
    open_bets = tracker.get_open_bets()

    if not open_bets:
        st.success("No open bets.")
        st.stop()

    st.info(f"{len(open_bets)} open bet(s)")
    for bet in open_bets:
        with st.expander(
            f"#{bet.id} · {bet.sport} · {bet.away_team} @ {bet.home_team} · "
            f"{bet.bet_type} {bet.side} {bet.odds:+.0f}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Stake (Kelly)", f"${bet.stake_kelly:.2f}")
            c2.metric("Stake (Flat)",  f"${bet.stake_flat:.2f}")
            c3.metric("EV at time",    f"{bet.ev_pct:+.2f}%")

            with st.form(f"settle_{bet.id}"):
                result = st.radio("Result", ["WIN", "LOSS"], horizontal=True)
                closing = st.number_input("Closing Odds (optional)", value=0.0, step=1.0)
                if st.form_submit_button("Settle"):
                    tracker.settle_bet(bet.id, won=(result == "WIN"),
                                       closing_odds=closing or None)
                    st.success(f"Bet #{bet.id} settled as {result}")
                    st.rerun()

            st.divider()
            confirmed_del = st.checkbox("Confirm delete", key=f"del_open_{bet.id}")
            if st.button("Delete Bet", key=f"del_open_btn_{bet.id}",
                         disabled=not confirmed_del):
                tracker.delete_bet(bet.id)
                st.success(f"Bet #{bet.id} deleted.")
                st.rerun()
