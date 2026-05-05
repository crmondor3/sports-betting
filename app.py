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
        "ev_min":     5.0,
        "kelly":      0.25,
        "bankroll":   100.0,
        "max_picks":  10,
        "min_conf":   55,
        # committed = what the last recalc used
        "committed_sport":     "All",
        "committed_ev_min":    5.0,
        "committed_kelly":     0.25,
        "committed_bankroll":  100.0,
        "committed_max_picks": 10,
        "committed_min_conf":  55,
        # Smart Picks filters (instant-apply, no recalc needed)
        "sp_min_odds":    -180,
        "sp_max_odds":     300,
        "sp_min_win_prob": 50,
        "sp_max_picks":    10,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Cached pipeline ────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _run_pipeline(sport_filter, ev_thresh, kelly_frac, bankroll, min_conf, max_picks_n):
    from run import run_pipeline
    import config as cfg
    cfg.EV_THRESHOLD    = ev_thresh
    cfg.DEFAULT_KELLY_FRACTION = kelly_frac
    cfg.STARTING_BANKROLL      = bankroll
    cfg.MIN_CONFIDENCE         = int(min_conf)
    cfg.MAX_PICKS_PER_DAY      = int(max_picks_n)
    picks, stats, api_rem = run_pipeline(sport_filter or None)
    return [p.to_dict() for p in picks], picks, stats, api_rem


@st.cache_data(ttl=86400, show_spinner=False)
def _run_winners_pipeline(sport_filter, kelly_frac, bankroll):
    """Low-threshold pipeline for Smart Picks — gets all +EV candidates so we
    can rank and filter by win probability in the UI."""
    from run import run_pipeline
    import config as cfg
    cfg.EV_THRESHOLD           = 0.01   # 1% — just needs to be positive
    cfg.DEFAULT_KELLY_FRACTION = kelly_frac
    cfg.STARTING_BANKROLL      = bankroll
    cfg.MIN_CONFIDENCE         = 0
    cfg.MAX_PICKS_PER_DAY      = 50
    picks, stats, api_rem = run_pipeline(sport_filter or None)
    return [p.to_dict() for p in picks], stats, api_rem


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## DK Model")
    st.caption(f"Today: {date.today().isoformat()}")
    st.divider()

    page = st.radio(
        "Page",
        ["Smart Picks", "Tennis", "Today's Picks", "Bankroll", "Bet History", "Settle Bets"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Settings")
    st.caption("Adjust below, then hit **Recalculate** to apply.")

    sport_sel = st.selectbox(
        "Sport", ["All", "NBA", "MLB", "NHL", "NFL"],
        index=["All","NBA","MLB","NHL","NFL"].index(st.session_state["sport"]),
    )
    ev_min_input = st.slider(
        "Min EV %", 1.0, 15.0,
        float(st.session_state["ev_min"]), 0.5, format="%.1f%%",
    )
    kelly_input = st.slider(
        "Kelly Fraction", 0.05, 1.0,
        float(st.session_state["kelly"]), 0.05, format="%.0f%%",
    )
    bankroll_input = st.number_input(
        "Bankroll ($)", 10.0,
        value=float(st.session_state["bankroll"]), step=10.0,
    )
    max_picks_input = st.slider(
        "Max Picks", 1, 15,
        int(st.session_state["max_picks"]),
    )
    min_conf_input = st.slider(
        "Min Confidence", 0, 100,
        int(st.session_state["min_conf"]),
    )

    st.divider()

    recalc = st.button("Recalculate", type="primary", use_container_width=True)
    if recalc:
        st.session_state["sport"]      = sport_sel
        st.session_state["ev_min"]     = ev_min_input
        st.session_state["kelly"]      = kelly_input
        st.session_state["bankroll"]   = bankroll_input
        st.session_state["max_picks"]  = max_picks_input
        st.session_state["min_conf"]   = min_conf_input
        # Commit to pipeline params
        st.session_state["committed_sport"]     = sport_sel
        st.session_state["committed_ev_min"]    = ev_min_input
        st.session_state["committed_kelly"]     = kelly_input
        st.session_state["committed_bankroll"]  = bankroll_input
        st.session_state["committed_max_picks"] = max_picks_input
        st.session_state["committed_min_conf"]  = min_conf_input
        st.cache_data.clear()
        st.rerun()

    # Smart Picks filters (instant — no recalc button needed)
    if page == "Smart Picks":
        st.divider()
        st.markdown("### Winner Filter")
        st.caption("Applied instantly — no Recalculate needed.")
        st.session_state["sp_min_odds"] = st.slider(
            "Exclude odds heavier than",
            min_value=-350, max_value=0,
            value=st.session_state["sp_min_odds"], step=10,
            format="%d",
            help="e.g. -180 excludes -200, -250, etc. Avoids massive favorites.",
        )
        st.session_state["sp_max_odds"] = st.slider(
            "Exclude longshots above",
            min_value=100, max_value=500,
            value=st.session_state["sp_max_odds"], step=25,
            format="+%d",
            help="Avoids low-probability bets with inflated payouts.",
        )
        st.session_state["sp_min_win_prob"] = st.slider(
            "Min win probability",
            min_value=40, max_value=75,
            value=st.session_state["sp_min_win_prob"], step=1,
            format="%d%%",
        )
        st.session_state["sp_max_picks"] = st.slider(
            "Max picks shown", 1, 15,
            value=st.session_state["sp_max_picks"],
        )

    # Cache status
    any_cached = any(is_cached(f"dk_odds_{v}") for v in config.SUPPORTED_SPORTS.values())
    if any_cached:
        st.success("Odds: cached today")
    else:
        st.warning("Odds: not fetched yet")

    if st.button("Force Refresh Odds", use_container_width=True,
                 help="Burns ~4 API requests. Use sparingly."):
        bust_all()
        st.cache_data.clear()
        st.rerun()

    st.caption("Odds fetched once/day. ESPN stats cached daily.")


# ── Read committed settings ────────────────────────────────────────────────────
sport_arg    = None if st.session_state["committed_sport"] == "All" else st.session_state["committed_sport"]
ev_min       = st.session_state["committed_ev_min"] / 100
kelly        = st.session_state["committed_kelly"]
bankroll_val = st.session_state["committed_bankroll"]
max_picks    = st.session_state["committed_max_picks"]
min_conf     = st.session_state["committed_min_conf"]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Smart Picks  (win-probability first, odds-filtered)
# ══════════════════════════════════════════════════════════════════════════════

if page == "Smart Picks":
    st.title("Smart Picks")
    sp_min_odds  = st.session_state["sp_min_odds"]
    sp_max_odds  = st.session_state["sp_max_odds"]
    sp_min_win   = st.session_state["sp_min_win_prob"] / 100
    sp_max_picks = st.session_state["sp_max_picks"]

    st.caption(
        f"Ranked by win probability × confidence  ·  "
        f"Odds {sp_min_odds:+d} to {sp_max_odds:+d}  ·  "
        f"Min win prob {st.session_state['sp_min_win_prob']}%  ·  "
        f"Kelly {kelly:.0%}  ·  Bankroll ${bankroll_val:,.0f}"
    )

    with st.spinner("Running pipeline (odds cached daily)..."):
        try:
            sp_dicts, sp_stats, sp_api_rem = _run_winners_pipeline(
                sport_arg, kelly, bankroll_val
            )
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.info("Check that ODDS_API_KEY is set in your Streamlit secrets or .env")
            st.stop()

    if sp_dicts and "game_id" not in sp_dicts[0]:
        st.cache_data.clear()
        st.rerun()

    # ── Filter by odds window + minimum win probability ────────────────────────
    filtered = [
        p for p in sp_dicts
        if sp_min_odds <= p["dk_odds"] <= sp_max_odds
        and p["model_prob"] >= sp_min_win
    ]

    # Rank by win_score = model_prob × (confidence / 100)
    filtered.sort(key=lambda p: p["model_prob"] * p["confidence"] / 100, reverse=True)
    filtered = filtered[:sp_max_picks]

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Picks", len(filtered))
    c2.metric("API Requests Left", sp_api_rem if sp_api_rem is not None else "cached")
    if filtered:
        avg_win  = sum(p["model_prob"] for p in filtered) / len(filtered)
        avg_conf = sum(p["confidence"] for p in filtered) / len(filtered)
        c3.metric("Avg Win Prob", f"{avg_win:.1%}")
        c4.metric("Avg Confidence", f"{avg_conf:.0f}/100")

    if not filtered:
        st.warning(
            "No picks match the current filter. "
            "Try widening the odds range or lowering the min win probability."
        )
        st.info(
            "The Smart Picks pipeline uses a 1% EV floor to surface candidates. "
            "If still empty, check that today's odds have been fetched (sidebar)."
        )
        st.stop()

    st.divider()
    st.markdown("### Top Picks by Win Probability")
    st.caption(
        "Odds extremes filtered out — focused on bets the model believes in "
        "at prices that are actually worth taking."
    )

    def _win_prob_color(prob: float) -> str:
        if prob >= 0.63: return "#00e676"
        if prob >= 0.54: return "#ffeb3b"
        return "#90a4ae"

    def _win_card_class(prob: float) -> str:
        if prob >= 0.63: return "bet-card bet-card-high"
        if prob >= 0.54: return "bet-card bet-card-med"
        return "bet-card bet-card-low"

    from models.kelly_criterion import kelly_fraction as _kf_sp
    from models.ev_calculator import american_to_decimal as _a2d_sp

    for i, p in enumerate(filtered, 1):
        prob      = p["model_prob"]
        conf      = p["confidence"]
        prob_col  = _win_prob_color(prob)
        card_cls  = _win_card_class(prob)
        odds_str  = f"+{p['dk_odds']:.0f}" if p["dk_odds"] >= 0 else f"{p['dk_odds']:.0f}"
        edge_pct  = p.get("edge_pct", round((prob - p["implied_prob"]) * 100, 1))
        bet_team  = p.get("bet_team", p["side"])
        bet_lbl   = p["bet_type"].replace("total_", "").replace("_", " ").title()
        side_disp = p["side"].upper() + (f" {p['line']:+g}" if p["line"] else "")
        signals_html = _signal_html(p.get("signals", []))

        live_frac  = _kf_sp(prob, p["dk_odds"], kelly)
        live_stake = round(bankroll_val * live_frac, 2)
        payout_per_100 = round((_a2d_sp(p["dk_odds"]) - 1) * 100, 0)

        try:
            game_dt = datetime.fromisoformat(p["commence_time_iso"]).replace(tzinfo=timezone.utc)
            mins_left = int((game_dt - datetime.now(timezone.utc)).total_seconds() / 60)
            if mins_left < 0:     time_badge_col, time_label = "#546e7a", "Started"
            elif mins_left < 60:  time_badge_col, time_label = "#ff5252", f"{mins_left}m to tip"
            elif mins_left < 180: time_badge_col, time_label = "#ffeb3b", f"{mins_left//60}h {mins_left%60}m to tip"
            else:                 time_badge_col, time_label = "#00e676", f"{mins_left//60}h to tip"
        except Exception:
            time_badge_col, time_label = "#546e7a", p["commence"]

        st.markdown(f"""
<div class="{card_cls}">

  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
      <span style="background:#1e3a5f; color:#90caf9; border-radius:50%; width:28px; height:28px;
                   display:inline-flex; align-items:center; justify-content:center;
                   font-weight:800; font-size:0.9rem; flex-shrink:0;">#{i}</span>
      <span class="tag tag-sport">{p['sport']}</span>
      <span class="tag tag-type">{bet_lbl}</span>
      <span style="font-size:1.2rem; font-weight:800; color:#ffffff; margin-left:6px;">{p['game']}</span>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <span style="background:{time_badge_col}22; color:{time_badge_col}; border:1px solid {time_badge_col};
                   padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600;">
        {time_label}
      </span>
      <span style="font-size:1.8rem; font-weight:900; color:{prob_col};">{prob:.0%} WIN</span>
    </div>
  </div>

  <div style="margin-top:10px; padding:8px 12px; background:#0d1520;
              border-radius:6px; font-size:0.9rem; color:#cfd8dc;">
    Betting <b style="color:#fff;">{bet_team}</b> ({bet_lbl} {side_disp} <b style="color:{prob_col};">{odds_str}</b>).
    Model: <b style="color:{prob_col};">{prob:.1%}</b> win chance vs book's {p['implied_prob']:.1%}.
    Edge: <b style="color:{prob_col};">+{edge_pct:.1f}%</b>
  </div>

  <div style="margin-top:14px; display:flex; gap:28px; flex-wrap:wrap;">
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Bet</div>
      <div style="font-size:1.25rem; font-weight:800; color:#ffffff;">{bet_team}</div>
      <div style="font-size:0.85rem; color:#b0bec5;">{bet_lbl} &nbsp;{side_disp}</div>
    </div>
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Odds</div>
      <div style="font-size:1.4rem; font-weight:800; color:#ffffff;">{odds_str}</div>
      <div style="font-size:0.8rem; color:#b0bec5;">+${payout_per_100:.0f} per $100</div>
    </div>
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Win Prob</div>
      <div style="font-size:1.4rem; font-weight:800; color:{prob_col};">{prob:.1%}</div>
      <div style="font-size:0.85rem; color:#b0bec5;">book: {p['implied_prob']:.1%}</div>
    </div>
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Stake (Kelly)</div>
      <div style="font-size:1.25rem; font-weight:800; color:#ffffff;">${live_stake:.2f}</div>
      <div style="font-size:0.85rem; color:#b0bec5;">{live_frac:.1%} of ${bankroll_val:,.0f}</div>
    </div>
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Confidence</div>
      <div style="font-size:1.25rem; font-weight:800; color:{_conf_color(conf)};">
        {conf}/100 <span style="font-size:0.8rem; color:#b0bec5;">({_conf_label(conf)})</span>
      </div>
      <div style="font-size:0.8rem; color:#b0bec5;">{p.get('data_quality','cold')} data</div>
    </div>
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">EV</div>
      <div style="font-size:1.25rem; font-weight:800; color:{_ev_color(p['ev_pct'])};">{p['ev_pct']:+.1f}%</div>
      <div style="font-size:0.8rem; color:#b0bec5;">expected value</div>
    </div>
  </div>

  <div style="margin-top:14px; border-top:1px solid #1e2e42; padding-top:10px;
              font-size:0.85rem; line-height:1.9; color:#b0bec5;">
    <span style="font-size:0.68rem; text-transform:uppercase; letter-spacing:1px; color:#546e7a;">
      Signals for {bet_team}</span><br>
    {signals_html if signals_html else '<span style="color:#546e7a;">No signal data — cold-start model</span>'}
  </div>

</div>
""", unsafe_allow_html=True)

    # ── Log bet ───────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Log a bet to tracker"):
        from models.kelly_criterion import kelly_fraction as kf
        from bets.edge_detector import EdgeBet

        sp_labels = [
            f"#{i}  {d['game']}  —  "
            f"{d['bet_type'].replace('total_','').title()} {d['side'].upper()}"
            + (f" {d['line']:+g}" if d['line'] else "")
            + f"  ({'+' if d['dk_odds']>=0 else ''}{int(d['dk_odds'])})"
            for i, d in enumerate(filtered, 1)
        ]
        sp_idx = st.selectbox("Pick to log", range(len(filtered)),
                              format_func=lambda x: sp_labels[x], key="sp_log_sel")
        sel = filtered[sp_idx]

        c1, c2 = st.columns(2)
        actual_odds  = c1.number_input("Odds placed at (American)", value=int(sel["dk_odds"]),
                                       step=1, key="sp_odds_input")
        sp_frac      = kf(sel["model_prob"], float(actual_odds), kelly)
        default_stk  = round(bankroll_val * sp_frac, 2)
        actual_stake = c2.number_input("Stake ($)", value=float(default_stk),
                                       min_value=0.01, step=0.50, key="sp_stake_input")

        tracker = BankrollTracker(starting_bankroll=bankroll_val)
        if st.button("Log Bet", type="primary", key="sp_log_btn"):
            eb = EdgeBet(
                sport=sel["sport"], game_id=sel["game_id"],
                home_team=sel["home_team"], away_team=sel["away_team"],
                commence_time=datetime.fromisoformat(sel["commence_time_iso"]),
                bet_type=sel["bet_type"], side=sel["side"], line=sel["line"],
                best_book="draftkings", best_odds=float(actual_odds),
                model_prob=sel["model_prob"], implied_prob=sel["implied_prob"],
                ev_pct=sel["ev_pct"], kelly_frac=sp_frac,
                recommended_stake=actual_stake, is_sharp_book=False,
            )
            bid = tracker.log_bet(eb, bankroll=bankroll_val)
            ods = f"+{int(actual_odds)}" if actual_odds >= 0 else str(int(actual_odds))
            st.success(f"Logged Bet #{bid} — {sel['bet_team']} {ods} · Stake ${actual_stake:.2f}")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Tennis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Tennis":
    st.title("Tennis Analysis")

    from data.tennis_api import TennisDataClient, infer_surface
    from models.tennis_model import TennisModel
    from tracker.database import session_scope as _tsc
    from tracker.models import TennisPlayer as _TP

    _tennis_client = TennisDataClient()
    _SURF_COLOR = {"grass": "#2e7d32", "clay": "#b71c1c", "hard": "#1565c0", "carpet": "#4a148c"}

    # ── Database status ────────────────────────────────────────────────────────
    with _tsc() as _ts:
        _atp_n = _ts.query(_TP).filter_by(tour="atp").count()
        _wta_n = _ts.query(_TP).filter_by(tour="wta").count()
        _latest_row = (
            _ts.query(_TP.last_updated)
            .filter(_TP.last_updated.isnot(None))
            .order_by(_TP.last_updated.desc())
            .first()
        )
        _db_updated = _latest_row[0] if _latest_row else None

    _dc1, _dc2, _dc3 = st.columns(3)
    _dc1.metric("ATP Players in DB", _atp_n)
    _dc2.metric("WTA Players in DB", _wta_n)
    _dc3.metric("DB Last Updated", _db_updated.strftime("%Y-%m-%d") if _db_updated else "Never")

    with st.expander("Build / Refresh Player Database"):
        st.caption(
            "Downloads Jeff Sackmann CSV data (last 3 seasons) and computes stats "
            "for every ATP/WTA player. Takes 1–3 minutes. Run once — data persists in SQLite."
        )
        _build_tour_sel = st.radio("Tour", ["Both", "ATP", "WTA"],
                                   horizontal=True, key="build_tour_sel")
        if st.button("Build Tennis Database", type="primary", key="build_db_btn"):
            from data.tennis_builder import build_tennis_db
            _tours_build = (["atp", "wta"] if _build_tour_sel == "Both"
                            else [_build_tour_sel.lower()])
            for _bt in _tours_build:
                _pb = st.progress(0.0, text=f"Building {_bt.upper()}…")
                _bn = build_tennis_db(
                    _bt,
                    progress_cb=lambda f, pb=_pb, t=_bt: pb.progress(
                        min(float(f), 1.0),
                        text=f"Processing {t.upper()} players ({float(f):.0%})"
                    )
                )
                _pb.progress(1.0, text=f"Done — {_bn} players written.")
                st.success(f"Built {_bn} {_bt.upper()} players.")
            st.rerun()

    st.divider()
    tab_matches, tab_kpi = st.tabs(["Today's Matches", "Player KPIs"])

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_tennis_matches(tours: tuple) -> list:
        client = TennisDataClient()
        out: list = []
        for t in tours:
            out.extend(client.get_upcoming_matches(t))
        return out

    # ── Today's Matches ────────────────────────────────────────────────────────
    with tab_matches:
        _tour_sel = st.radio("Tour", ["ATP", "WTA", "Both"],
                             horizontal=True, key="tennis_tour")
        _tours_fetch = ("atp", "wta") if _tour_sel == "Both" else (_tour_sel.lower(),)

        with st.spinner("Fetching ESPN scoreboard…"):
            _all_matches = _fetch_tennis_matches(_tours_fetch)

        if not _all_matches:
            st.warning(
                "No matches found on the ESPN tennis scoreboard right now. "
                "Check back when tournaments are in progress."
            )
        else:
            st.caption(f"{len(_all_matches)} match(es) found")
            if _atp_n + _wta_n == 0:
                st.info(
                    "Player database is empty — model will use default serve percentages. "
                    "Click **Build / Refresh Player Database** above to populate it with real stats."
                )

            for _match in _all_matches:
                _p1n   = _match["player1_name"]
                _p2n   = _match["player2_name"]
                _tbadge = _match["tour"]
                _tourn  = _match["tournament"] or "Unknown Tournament"
                _surf   = _match["surface"]
                _rnd    = _match["round_name"] or ""
                _done   = _match["is_completed"]
                _tkey   = "atp" if _tbadge == "ATP" else "wta"

                try:
                    _s1 = _tennis_client.get_player_from_db(_p1n, _tkey)
                    _s2 = _tennis_client.get_player_from_db(_p2n, _tkey)
                except Exception:
                    _s1 = {"data_quality": "no_data", "overall": {}, "recent_matches": [],
                            "surface_record": {}, "day_record": {}, "first_set_stats": {}}
                    _s2 = dict(_s1)

                try:
                    _pred = TennisModel.predict(_s1, _s2, _surf, best_of=3)
                    _p1p  = _pred["p1_win_prob"]
                    _p2p  = _pred["p2_win_prob"]
                    _conf = _pred["model_confidence"]
                    _sfav = _pred["surface_favors"]
                except Exception:
                    _p1p, _p2p, _conf, _sfav = 0.5, 0.5, 0, "neutral"

                _sc     = _SURF_COLOR.get(_surf, "#546e7a")
                _sbadge = (
                    '<span style="background:#546e7a33;color:#90a4ae;border:1px solid #546e7a;'
                    'padding:2px 8px;border-radius:20px;font-size:0.75rem;">Completed</span>'
                    if _done else
                    '<span style="background:#00e67622;color:#00e676;border:1px solid #00e676;'
                    'padding:2px 8px;border-radius:20px;font-size:0.75rem;">Upcoming</span>'
                )
                _sfnote = ""
                if _sfav == "player1":
                    _sfnote = f'<span style="color:#00e676;font-size:0.8rem;">Surface favors {_p1n}</span>'
                elif _sfav == "player2":
                    _sfnote = f'<span style="color:#00e676;font-size:0.8rem;">Surface favors {_p2n}</span>'
                _cconf = "#00e676" if _conf >= 60 else ("#ffeb3b" if _conf >= 35 else "#78909c")

                st.markdown(f"""
<div class="bet-card" style="border-left:5px solid {_sc}; margin-bottom:20px;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <span style="background:{_sc}22;color:{_sc};border:1px solid {_sc};
                   padding:2px 10px;border-radius:20px;font-size:0.8rem;font-weight:700;">{_surf.upper()}</span>
      <span class="tag tag-sport">{_tbadge}</span>
      {_sbadge}
      <span style="font-size:1rem;color:#b0bec5;">{_tourn}</span>
      {"<span style='color:#78909c;font-size:0.85rem;'>· " + _rnd + "</span>" if _rnd else ""}
    </div>
    <span style="color:{_cconf};font-size:0.8rem;font-weight:600;">Model confidence: {_conf}%</span>
  </div>
  <div style="margin-top:14px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
    <div style="flex:1;min-width:120px;text-align:right;">
      <div style="font-size:1.3rem;font-weight:800;color:#ffffff;">{_p1n}</div>
      <div style="font-size:1.8rem;font-weight:900;color:#40c4ff;">{_p1p:.0%}</div>
    </div>
    <div style="min-width:60px;text-align:center;">
      <div style="background:#1e2a3a;border-radius:6px;height:12px;overflow:hidden;width:160px;margin:0 auto;">
        <div style="background:linear-gradient(90deg,#40c4ff {_p1p*100:.0f}%,#ff6b6b {_p1p*100:.0f}%);height:100%;border-radius:6px;"></div>
      </div>
      <div style="color:#78909c;font-size:0.75rem;margin-top:4px;">WIN PROBABILITY</div>
      {_sfnote}
    </div>
    <div style="flex:1;min-width:120px;text-align:left;">
      <div style="font-size:1.3rem;font-weight:800;color:#ffffff;">{_p2n}</div>
      <div style="font-size:1.8rem;font-weight:900;color:#ff6b6b;">{_p2p:.0%}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                # Surface stats comparison
                _sr1     = _s1.get("surface_record", {}).get(_surf, {})
                _sr2     = _s2.get("surface_record", {}).get(_surf, {})
                _any_srf = (
                    (_sr1.get("wins", 0) + _sr1.get("losses", 0)) > 0
                    or (_sr2.get("wins", 0) + _sr2.get("losses", 0)) > 0
                )
                if _any_srf:
                    _mc1, _mc2 = st.columns(2)
                    with _mc1:
                        for _pname, _sr, _clr in [(_p1n, _sr1, "#40c4ff"), (_p2n, _sr2, "#ff6b6b")]:
                            _pw  = _sr.get("wins", 0)
                            _pl  = _sr.get("losses", 0)
                            _ptot = _pw + _pl
                            _pwr  = round(_pw / _ptot * 100, 0) if _ptot else 0
                            _pspw = _sr.get("serve_win_pct")
                            _prpw = _sr.get("return_win_pct")
                            st.markdown(
                                f'<div style="background:#0d1520;padding:10px 14px;border-radius:8px;'
                                f'margin-bottom:8px;border-left:3px solid {_clr};">'
                                f'<b style="color:{_clr};">{_pname}</b> on {_surf}'
                                f'<span style="color:#78909c;"> — {_pw}W / {_pl}L ({_pwr:.0f}%)</span><br>'
                                f'Serve win %: <b>{f"{_pspw*100:.1f}%" if _pspw else "N/A"}</b>'
                                f' &nbsp;·&nbsp; '
                                f'Return win %: <b>{f"{_prpw*100:.1f}%" if _prpw else "N/A"}</b>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    with _mc2:
                        _bspw1 = ((_sr1.get("serve_win_pct") or _s1.get("overall", {}).get("avg_serve_win_pct") or 0) * 100)
                        _bspw2 = ((_sr2.get("serve_win_pct") or _s2.get("overall", {}).get("avg_serve_win_pct") or 0) * 100)
                        _brpw1 = ((_sr1.get("return_win_pct") or _s1.get("overall", {}).get("avg_return_win_pct") or 0) * 100)
                        _brpw2 = ((_sr2.get("return_win_pct") or _s2.get("overall", {}).get("avg_return_win_pct") or 0) * 100)
                        _cfig = go.Figure()
                        _cfig.add_trace(go.Bar(
                            name="Serve Win %",
                            x=[_p1n[:12], _p2n[:12]], y=[_bspw1, _bspw2],
                            marker_color=["#40c4ff", "#ff6b6b"],
                        ))
                        _cfig.add_trace(go.Bar(
                            name="Return Win %",
                            x=[_p1n[:12], _p2n[:12]], y=[_brpw1, _brpw2],
                            marker_color=["#00bcd4", "#ef9a9a"],
                        ))
                        _cfig.update_layout(
                            barmode="group",
                            title=f"Serve & Return on {_surf.capitalize()}",
                            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font_color="white", height=220,
                            yaxis=dict(range=[0, 80], ticksuffix="%"),
                            margin=dict(l=10, r=10, t=40, b=20),
                            legend=dict(bgcolor="#1e2a3a"),
                        )
                        st.plotly_chart(_cfig, use_container_width=True,
                                        key=f"comp_{_match['match_id']}")
                else:
                    st.caption(f"No {_surf} surface stats in database for these players.")

                st.divider()

    # ── Player KPIs ────────────────────────────────────────────────────────────
    with tab_kpi:
        _kpi_tour = st.radio("Tour", ["ATP", "WTA"], horizontal=True, key="kpi_tour")
        _kpi_tkey = _kpi_tour.lower()

        with _tsc() as _kts:
            _kpi_rows  = (
                _kts.query(_TP.name)
                .filter_by(tour=_kpi_tkey)
                .order_by(_TP.name)
                .all()
            )
            _kpi_names = [r[0] for r in _kpi_rows]

        if not _kpi_names:
            st.warning(
                f"No {_kpi_tour} players in database yet. "
                "Use **Build / Refresh Player Database** above to populate it."
            )
        else:
            st.caption(f"{len(_kpi_names)} {_kpi_tour} players in database")
            _kpi_player = st.selectbox("Select Player", _kpi_names, key="kpi_player_sel")

            if _kpi_player:
                _kst = _tennis_client.get_player_from_db(_kpi_player, _kpi_tkey)

                if _kst["data_quality"] == "no_data":
                    st.warning(f"No stats found for {_kpi_player}.")
                else:
                    _ov  = _kst.get("overall", {})
                    _fs  = _kst.get("first_set_stats", {})
                    _mn  = _kst.get("matched_name", _kpi_player)
                    if _mn != _kpi_player:
                        st.caption(f"Matched to dataset name: **{_mn}**")

                    _tm  = _ov.get("wins", 0) + _ov.get("losses", 0)
                    _wp  = (_ov.get("wins", 0) / _tm * 100) if _tm else 0
                    _spw = _ov.get("avg_serve_win_pct")
                    _rpw = _ov.get("avg_return_win_pct")
                    _rnk = _ov.get("ranking")

                    _kc1, _kc2, _kc3, _kc4 = st.columns(4)
                    _kc1.metric("Win %", f"{_wp:.1f}%",
                                help=f"{_ov.get('wins',0)}W / {_ov.get('losses',0)}L")
                    _kc2.metric("Avg Serve Win %", f"{_spw*100:.1f}%" if _spw else "N/A")
                    _kc3.metric("Avg Return Win %", f"{_rpw*100:.1f}%" if _rpw else "N/A")
                    _kc4.metric("Ranking", f"#{_rnk}" if _rnk else "N/A")

                    st.divider()
                    _cc1, _cc2 = st.columns(2)

                    _surfrec = _kst.get("surface_record", {})

                    with _cc1:
                        _sord = ["hard", "clay", "grass"]
                        _swd  = {"Surface": [], "Wins": [], "Losses": []}
                        for _s in _sord:
                            if _s in _surfrec:
                                _swd["Surface"].append(_s.capitalize())
                                _swd["Wins"].append(_surfrec[_s].get("wins", 0))
                                _swd["Losses"].append(_surfrec[_s].get("losses", 0))
                        if _swd["Surface"]:
                            _sfig = go.Figure()
                            _sfig.add_trace(go.Bar(name="Wins", x=_swd["Surface"], y=_swd["Wins"],
                                                   marker_color="#00e676"))
                            _sfig.add_trace(go.Bar(name="Losses", x=_swd["Surface"], y=_swd["Losses"],
                                                   marker_color="#ff5252"))
                            _sfig.update_layout(barmode="group", title="Win / Loss by Surface",
                                                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                                font_color="white", legend=dict(bgcolor="#1e2a3a"),
                                                height=300)
                            st.plotly_chart(_sfig, use_container_width=True,
                                            key=f"kpi_surf_{_kpi_player}")
                        else:
                            st.info("No surface breakdown available.")

                    with _cc2:
                        _dayrec = _kst.get("day_record", {})
                        _dord   = ["Monday","Tuesday","Wednesday","Thursday",
                                   "Friday","Saturday","Sunday"]
                        _dwd    = {"Day": [], "Win Rate": [], "Matches": []}
                        for _dn in _dord:
                            if _dn in _dayrec:
                                _dw2 = _dayrec[_dn].get("wins", 0)
                                _dl2 = _dayrec[_dn].get("losses", 0)
                                _dt2 = _dw2 + _dl2
                                _dwd["Day"].append(_dn[:3])
                                _dwd["Win Rate"].append(round(_dw2 / _dt2 * 100, 1) if _dt2 else 0)
                                _dwd["Matches"].append(_dt2)
                        if _dwd["Day"]:
                            _dfig = go.Figure(go.Bar(
                                x=_dwd["Day"], y=_dwd["Win Rate"],
                                marker_color=[
                                    "#00e676" if v >= 60 else ("#ffeb3b" if v >= 45 else "#ff5252")
                                    for v in _dwd["Win Rate"]
                                ],
                                text=[f"{v:.0f}%<br>({n})"
                                      for v, n in zip(_dwd["Win Rate"], _dwd["Matches"])],
                                textposition="outside",
                            ))
                            _dfig.update_layout(
                                title="Win Rate by Day of Week",
                                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                font_color="white",
                                yaxis=dict(range=[0, 110], ticksuffix="%"), height=300,
                            )
                            st.plotly_chart(_dfig, use_container_width=True,
                                            key=f"kpi_day_{_kpi_player}")
                        else:
                            st.info("No day-of-week data available.")

                    st.divider()
                    st.markdown("#### First Set Impact")
                    _wfs = _fs.get("won_first_set_win_pct")
                    _lfs = _fs.get("lost_first_set_win_pct")
                    _fsr = _fs.get("first_set_win_rate")
                    _fc1, _fc2, _fc3 = st.columns(3)
                    _fc1.metric("Wins 1st set → wins match",
                                f"{_wfs*100:.1f}%" if _wfs is not None else "N/A")
                    _fc2.metric("Loses 1st set → wins match",
                                f"{_lfs*100:.1f}%" if _lfs is not None else "N/A")
                    _fc3.metric("1st Set Win Rate",
                                f"{_fsr*100:.1f}%" if _fsr is not None else "N/A")

                    st.divider()
                    st.markdown("#### Serve Averages")
                    _fsp = _ov.get("avg_first_serve_pct")
                    _ace = _ov.get("avg_aces")
                    _dfs = _ov.get("avg_dfs")
                    _ka1, _ka2, _ka3, _ka4 = st.columns(4)
                    _ka1.metric("First Serve %",  f"{_fsp*100:.1f}%" if _fsp else "N/A")
                    _ka2.metric("Serve Win %",     f"{_spw*100:.1f}%" if _spw else "N/A")
                    _ka3.metric("Aces / Match",    f"{_ace:.1f}" if _ace is not None else "N/A")
                    _ka4.metric("Double Faults / Match", f"{_dfs:.1f}" if _dfs is not None else "N/A")

                    if _surfrec:
                        st.divider()
                        st.markdown("#### Stats by Surface")
                        _strows = []
                        for _s in ["hard", "clay", "grass"]:
                            if _s not in _surfrec:
                                continue
                            _sr  = _surfrec[_s]
                            _w2  = _sr.get("wins", 0)
                            _l2  = _sr.get("losses", 0)
                            _t2  = _w2 + _l2
                            _swp = _sr.get("serve_win_pct")
                            _rrp = _sr.get("return_win_pct")
                            _strows.append({
                                "Surface":       _s.capitalize(),
                                "W":             _w2,
                                "L":             _l2,
                                "Win %":         f"{round(_w2/_t2*100,1):.1f}%" if _t2 else "0%",
                                "Serve Win %":   f"{_swp*100:.1f}%" if _swp else "N/A",
                                "Return Win %":  f"{_rrp*100:.1f}%" if _rrp else "N/A",
                            })
                        if _strows:
                            st.dataframe(pd.DataFrame(_strows),
                                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Today's Picks
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Today's Picks":
    st.title("Today's Top Picks")
    st.caption(
        f"DraftKings only  ·  Min EV {ev_min:.0%}  ·  "
        f"Min Confidence {min_conf}  ·  Max {max_picks} picks  ·  "
        f"Kelly {kelly:.0%}  ·  Bankroll ${bankroll_val:,.0f}"
    )

    with st.spinner("Running pipeline (odds fetched once today)..."):
        try:
            dicts, picks_obj, stats, api_rem = _run_pipeline(
                sport_arg, ev_min, kelly, bankroll_val, min_conf, max_picks
            )
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.info("Check that ODDS_API_KEY is set in .env")
            st.stop()

    # If cached dicts are missing game_id (old format), clear cache and rerun
    if dicts and "game_id" not in dicts[0]:
        st.cache_data.clear()
        st.rerun()

    # Apply max_picks override
    dicts = dicts[:max_picks]

    # ── Top-line KPIs ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Picks Today", len(dicts))
    c2.metric("API Requests Left", api_rem if api_rem is not None else "cached")
    tracker = BankrollTracker(starting_bankroll=bankroll_val)
    br_stats = tracker.get_stats()
    c3.metric("Bankroll (Kelly)", f"${br_stats.get('current_bankroll_kelly', bankroll_val):,.2f}")
    c4.metric("Hit Rate", br_stats.get("hit_rate", "—"))

    if not dicts:
        st.warning(
            f"No picks meet the criteria (EV ≥ {ev_min:.0%}, confidence ≥ {config.MIN_CONFIDENCE}).\n\n"
            "Try lowering the EV slider, or check back after today's games are posted."
        )
        st.stop()

    st.divider()

    # ── Bet cards ─────────────────────────────────────────────────────────────
    st.markdown("### Recommended Picks")
    st.caption("Ranked by EV × Confidence — only bets where the model and matchup data agree.")

    for i, p in enumerate(dicts, 1):
        conf = p["confidence"]
        ev = p["ev_pct"]
        card_class = _card_class(conf)
        conf_col = _conf_color(conf)
        ev_col = _ev_color(ev)

        bet_team = p.get("bet_team", p["side"])
        side_display = p["side"].upper()
        if p["line"] is not None:
            side_display += f" {p['line']:+g}"

        bet_type_label = p["bet_type"].replace("total_", "").replace("_", " ").title()
        odds_str = f"+{p['dk_odds']:.0f}" if p['dk_odds'] >= 0 else f"{p['dk_odds']:.0f}"
        edge_pct = p.get("edge_pct", round((p["model_prob"] - p["implied_prob"]) * 100, 1))
        signals_html = _signal_html(p.get("signals", []))

        # Live stake from current bankroll
        from models.kelly_criterion import kelly_fraction
        from models.ev_calculator import decimal_to_american, american_to_decimal
        live_kelly_frac = kelly_fraction(p["model_prob"], p["dk_odds"], kelly)
        live_stake = round(bankroll_val * live_kelly_frac, 2)

        # ── Bet-to line: the worst odds at which this bet stays +EV ──────────
        # EV = 0 when decimal odds = 1 / model_prob
        breakeven_decimal = 1 / p["model_prob"]
        breakeven_american = decimal_to_american(breakeven_decimal)
        be_str = f"+{breakeven_american:.0f}" if breakeven_american >= 0 else f"{breakeven_american:.0f}"

        # Time to game — commence_time_iso is UTC, compare to UTC now
        try:
            game_dt = datetime.fromisoformat(p["commence_time_iso"]).replace(tzinfo=timezone.utc)
            mins_left = int((game_dt - datetime.now(timezone.utc)).total_seconds() / 60)
            if mins_left < 0:
                time_badge_col = "#546e7a"
                time_label = "Started"
            elif mins_left < 60:
                time_badge_col = "#ff5252"
                time_label = f"{mins_left}m to tip"
            elif mins_left < 180:
                time_badge_col = "#ffeb3b"
                time_label = f"{mins_left // 60}h {mins_left % 60}m to tip"
            else:
                time_badge_col = "#00e676"
                time_label = f"{mins_left // 60}h to tip"
        except Exception:
            time_badge_col = "#546e7a"
            time_label = p["commence"]

        # Explain WHY
        why_html = (
            f"Betting <b style='color:#ffffff;'>{bet_team}</b> ({bet_type_label}). "
            f"Model gives them <b style='color:{ev_col};'>{p['model_prob']:.1%}</b> — "
            f"DraftKings prices them at <b>{p['implied_prob']:.1%}</b>. "
            f"Edge: <b style='color:{ev_col};'>+{edge_pct:.1f}%</b>"
        )

        st.markdown(f"""
<div class="{card_class}">

  <!-- Header row -->
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
      <span style="background:#1e3a5f; color:#90caf9; border-radius:50%; width:28px; height:28px;
                   display:inline-flex; align-items:center; justify-content:center;
                   font-weight:800; font-size:0.9rem; flex-shrink:0;">#{i}</span>
      <span class="tag tag-sport">{p['sport']}</span>
      <span class="tag tag-type">{bet_type_label}</span>
      <span style="font-size:1.2rem; font-weight:800; color:#ffffff; margin-left:6px;">
        {p['game']}
      </span>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <span style="background:{time_badge_col}22; color:{time_badge_col}; border:1px solid {time_badge_col};
                   padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600;">
        {time_label}
      </span>
      <span style="font-size:1.6rem; font-weight:800; color:{ev_col};">{ev:+.1f}% EV</span>
    </div>
  </div>

  <!-- Why box -->
  <div style="margin-top:10px; padding:8px 12px; background:#0d1520;
              border-radius:6px; font-size:0.9rem; color:#cfd8dc;">
    {why_html}
  </div>

  <!-- Stats row -->
  <div style="margin-top:14px; display:flex; gap:28px; flex-wrap:wrap;">
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Bet</div>
      <div style="font-size:1.25rem; font-weight:800; color:#ffffff;">{bet_team}</div>
      <div style="font-size:0.85rem; color:#b0bec5;">{bet_type_label} &nbsp;{side_display} &nbsp;
        <span style="color:#ffffff; font-weight:700;">{odds_str}</span>
      </div>
    </div>
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Model vs Book</div>
      <div style="font-size:1.15rem; font-weight:700; color:{ev_col};">{p['model_prob']:.1%}</div>
      <div style="font-size:0.85rem; color:#b0bec5;">book implied: {p['implied_prob']:.1%}</div>
    </div>
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Stake (Kelly)</div>
      <div style="font-size:1.25rem; font-weight:800; color:#ffffff;">${live_stake:.2f}</div>
      <div style="font-size:0.85rem; color:#b0bec5;">{live_kelly_frac:.1%} of ${bankroll_val:,.0f}</div>
    </div>
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Bet-To Line</div>
      <div style="font-size:1.15rem; font-weight:700; color:#ff5252;">{be_str}</div>
      <div style="font-size:0.8rem; color:#b0bec5;">no value below this</div>
    </div>
    <div>
      <div style="color:#78909c; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Confidence</div>
      <div style="font-size:1.15rem; font-weight:700; color:{conf_col};">
        {conf}/100 <span style="font-size:0.8rem; color:#b0bec5;">({_conf_label(conf)})</span>
      </div>
      <div style="font-size:0.8rem; color:#b0bec5;">{p.get('data_quality','cold')} data</div>
    </div>
  </div>

  <!-- Signals -->
  <div style="margin-top:14px; border-top:1px solid #1e2e42; padding-top:10px;
              font-size:0.85rem; line-height:1.9; color:#b0bec5;">
    <span style="font-size:0.68rem; text-transform:uppercase; letter-spacing:1px; color:#546e7a;">
      Signals for {bet_team}</span><br>
    {signals_html if signals_html else '<span style="color:#546e7a;">No signal data — cold-start model</span>'}
  </div>

</div>
""", unsafe_allow_html=True)

    # ── Log bets ──────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Log a bet to tracker"):
        from models.kelly_criterion import kelly_fraction as kf
        from bets.edge_detector import EdgeBet

        pick_labels = [
            f"#{i}  {d['game']}  —  "
            f"{d['bet_type'].replace('total_','').title()} {d['side'].upper()}"
            + (f" {d['line']:+g}" if d['line'] else "")
            + f"  ({'+' if d['dk_odds']>=0 else ''}{int(d['dk_odds'])})"
            for i, d in enumerate(dicts, 1)
        ]
        pick_idx = st.selectbox("Pick to log", range(len(dicts)),
                                format_func=lambda x: pick_labels[x])
        sel = dicts[pick_idx]

        c1, c2 = st.columns(2)
        actual_odds = c1.number_input(
            "Odds placed at (American)",
            value=int(sel["dk_odds"]),
            step=1,
            help="Adjust if the line moved when you placed the bet",
        )
        live_frac   = kf(sel["model_prob"], float(actual_odds), kelly)
        default_stake = round(bankroll_val * live_frac, 2)
        actual_stake = c2.number_input(
            "Stake ($)",
            value=float(default_stake),
            min_value=0.01,
            step=0.50,
            help="Override Kelly stake if desired",
        )

        if st.button("Log Bet", type="primary"):
            eb = EdgeBet(
                sport=sel["sport"],
                game_id=sel["game_id"],
                home_team=sel["home_team"],
                away_team=sel["away_team"],
                commence_time=datetime.fromisoformat(sel["commence_time_iso"]),
                bet_type=sel["bet_type"],
                side=sel["side"],
                line=sel["line"],
                best_book="draftkings",
                best_odds=float(actual_odds),
                model_prob=sel["model_prob"],
                implied_prob=sel["implied_prob"],
                ev_pct=sel["ev_pct"],
                kelly_frac=live_frac,
                recommended_stake=actual_stake,
                is_sharp_book=False,
            )
            bid = tracker.log_bet(eb, bankroll=bankroll_val)
            odds_str = f"+{int(actual_odds)}" if actual_odds >= 0 else str(int(actual_odds))
            st.success(f"Logged Bet #{bid} — {sel['bet_team']} {odds_str} · Stake ${actual_stake:.2f}")
            st.rerun()


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
