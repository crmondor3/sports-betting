# Sports Betting Model

A modular, end-to-end positive-EV sports betting pipeline built in Python.

## Features

- **Live odds** via The Odds API (NFL, NBA, MLB, NHL)
- **Poisson regression** model for game totals
- **Elo model** (+ optional logistic layer) for moneylines
- **EV calculator** — only surfaces bets >3% edge by default
- **Kelly Criterion** sizing (fractional Kelly, capped)
- **SQLite bankroll tracker** — ROI, hit rate, CLV
- **Rich CLI dashboard** with colour-coded EV%
- **Telegram + email alerts** (optional)
- **APScheduler** — odds refresh every 5 minutes

---

## Quick Start

### 1. Clone and set up a virtual environment

```bash
cd sports_betting
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.template .env
```

Open `.env` and fill in your keys:

| Key | Where to get it |
|-----|----------------|
| `ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com) — free 500 req/month |
| `SPORTSDATA_NFL_KEY` | [sportsdata.io](https://sportsdata.io) — free tier |
| `SPORTSDATA_MLB_KEY` | same |
| `SPORTSDATA_NHL_KEY` | same |
| `BALLDONTLIE_API_KEY` | [balldontlie.io](https://balldontlie.io) — free (optional, speeds up NBA) |
| `TELEGRAM_BOT_TOKEN` | BotFather on Telegram (optional) |
| `EMAIL_SENDER` / `EMAIL_PASSWORD` | Gmail app password (optional) |

### 4. Run the pipeline

```bash
# Single run — show today's top bets
python run.py

# Live dashboard — refreshes every 60s
python run.py --live

# Filter to one sport
python run.py --sport NBA

# Run + send alerts
python run.py --alert

# Print bankroll stats
python run.py --stats

# Export bets to CSV
python run.py --export

# Settle bet #42 as a win
python run.py --settle 42 W
```

---

## Project Structure

```
sports_betting/
├── .env.template          # Copy to .env and fill in keys
├── config.py              # All config loaded from .env
├── run.py                 # Entry point / CLI
├── requirements.txt
│
├── data/
│   ├── odds_api.py        # The Odds API v4 client
│   ├── stats_api.py       # BallDontLie (NBA) + SportsData.io
│   └── scheduler.py       # APScheduler jobs
│
├── models/
│   ├── ev_calculator.py   # EV, implied probability, CLV
│   ├── kelly_criterion.py # Kelly Criterion sizing
│   ├── poisson_model.py   # Poisson regression for totals
│   └── elo_model.py       # Elo + logistic for moneylines
│
├── bets/
│   └── edge_detector.py   # Surfaces +EV bets across all markets
│
├── tracker/
│   ├── models.py          # SQLAlchemy ORM (Bet, BankrollSnapshot)
│   ├── database.py        # Engine, session, init_db
│   └── bankroll.py        # Log/settle bets, ROI/CLV stats, CSV export
│
├── alerts/
│   ├── telegram_alert.py  # Telegram daily summary
│   └── email_alert.py     # HTML email summary
│
├── dashboard/
│   └── cli.py             # Rich CLI dashboard
│
└── tests/
    ├── test_ev_calculator.py
    └── test_kelly.py
```

---

## Running Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Model Details

### EV Calculator
`EV = (model_prob × net_payout) − (1 − model_prob)`

Bets only surface when `EV% > 3%` (configurable via `EV_THRESHOLD` in `.env`).

### Poisson Model (Totals)
Estimates per-team attack/defence parameters via maximum likelihood. Predicts joint score distribution to compute over/under probabilities at any line.

### Elo Model (Moneylines)
Standard Elo with configurable K-factor and home-court advantage. Can be bootstrapped from win/loss standings when full game logs are unavailable.

### Kelly Criterion
`f* = (bp - q) / b` where `b` = decimal odds − 1, `p` = model probability, `q` = 1 − p.

Default sizing is **quarter Kelly** (25%), capped at 10% of bankroll per bet.

---

## Automating the Pipeline (Windows)

Add a scheduled task or use the built-in APScheduler:

```python
# In a background script
from data.scheduler import add_odds_refresh_job, start
import run

add_odds_refresh_job(lambda: run.run_pipeline())
start()
```

---

## API Quota Tips

- The Odds API free tier: ~500 requests/month. Each `get_all_sports_odds()` call uses ~4 requests (one per sport).
- Refreshing every 5 minutes across 4 sports = ~1,152 requests/day — upgrade to a paid tier for live use.
- Set `ODDS_REFRESH_INTERVAL=900` (15 min) in `.env` to stay within free quota.
