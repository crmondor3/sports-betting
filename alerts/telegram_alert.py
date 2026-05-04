"""Telegram alert — sends daily top picks summary."""
from __future__ import annotations

import logging

import httpx

import config
from bets.edge_detector import EdgeBet

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlert:
    def __init__(
        self,
        token: str = config.TELEGRAM_BOT_TOKEN,
        chat_id: str = config.TELEGRAM_CHAT_ID,
    ):
        self._token = token
        self._chat_id = chat_id
        self._enabled = bool(token and chat_id)
        if not self._enabled:
            logger.info("Telegram alerts disabled — no token/chat_id configured.")

    def _send(self, text: str) -> bool:
        if not self._enabled:
            return False
        url = _TELEGRAM_API.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        try:
            r = httpx.post(url, json=payload, timeout=10)
            r.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    def send_daily_summary(self, top_bets: list[EdgeBet], bankroll: float) -> bool:
        if not top_bets:
            return self._send("🏈 <b>Sports Betting Model</b>\nNo qualifying edges found today.")

        lines = ["🏈 <b>Top Picks — Today</b>", f"Bankroll: ${bankroll:,.2f}\n"]
        for i, bet in enumerate(top_bets[:10], 1):
            game = f"{bet.away_team} @ {bet.home_team}"
            lines.append(
                f"{i}. <b>{bet.sport}</b> | {game}\n"
                f"   {bet.bet_type} → {bet.side.upper()}"
                + (f" {bet.line:+g}" if bet.line else "")
                + f"  [{bet.best_odds:+.0f}]\n"
                f"   EV: <b>{bet.ev_pct:+.2f}%</b>  "
                f"Model: {bet.model_prob:.1%}  "
                f"Stake: ${bet.recommended_stake:.2f} @{bet.best_book}"
            )
        return self._send("\n".join(lines))

    def send_alert(self, message: str) -> bool:
        return self._send(message)
