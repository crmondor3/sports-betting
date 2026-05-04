"""
Streamlit app — DraftKings sports betting model.
Focus: quality over quantity. Top picks as bet cards, not raw tables.
Run: streamlit run app.py
"""
from __future__ import annotations

import sys
from datetime import datetime, date
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


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## DK Model")
    st.caption(f"Today: {date.today().isoformat()}")
    st.divider()

    page = st.radio(
        "Page",
        ["Today's Picks", "Bankroll", "Bet History", "Settle Bets"],
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
# PAGE: Today's Picks
# ══════════════════════════════════════════════════════════════════════════════

if page == "Today's Picks":
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

        # Time to game
        try:
            game_dt = datetime.strptime(p["commence"], "%m/%d %H:%M").replace(
                year=datetime.now().year)
            now = datetime.now()
            mins_left = int((game_dt - now).total_seconds() / 60)
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
