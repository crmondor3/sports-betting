"""Telegram alert — sends today's top picks to your phone via Telegram bot."""
from __future__ import annotations

import logging

import httpx

import config

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlert:
    def __init__(
        self,
        token: str = config.TELEGRAM_BOT_TOKEN,
        chat_id: str = config.TELEGRAM_CHAT_ID,
    ):
        self._token   = token
        self._chat_id = chat_id
        self._enabled = bool(token and chat_id)
        if not self._enabled:
            logger.info("Telegram alerts disabled — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _send(self, text: str) -> bool:
        if not self._enabled:
            return False
        url = _TELEGRAM_API.format(token=self._token)
        try:
            r = httpx.post(
                url,
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            r.raise_for_status()
            logger.info("Telegram alert sent (%d chars)", len(text))
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    def send_picks(self, picks: list[dict], bankroll: float, ev_min: float = 15.0) -> bool:
        """
        Send today's qualifying picks (EV >= ev_min) as a Telegram message.
        picks: list of Pick.to_dict() dicts from run_pipeline.
        """
        qualifying = [
            p for p in picks
            if p.get("ev_pct", 0) >= ev_min
            and p.get("kelly_frac", 0) > 0
        ]

        if not qualifying:
            return self._send(
                f"🎯 <b>DK Model — No Picks Today</b>\n"
                f"No bets cleared {ev_min:.0f}% EV today.\n"
                f"Bankroll: <b>${bankroll:,.2f}</b>"
            )

        lines = [
            f"🎯 <b>DK Model — {len(qualifying)} Pick{'s' if len(qualifying) != 1 else ''} Today</b>",
            f"Bankroll: <b>${bankroll:,.2f}</b>  ·  Min EV: {ev_min:.0f}%\n",
        ]

        for i, p in enumerate(qualifying[:10], 1):
            sport    = p.get("sport", "")
            game     = p.get("game", "")
            bet_type = p.get("bet_type", "").replace("total_", "").replace("_", " ").title()
            side     = p.get("side", "").upper()
            line     = p.get("line")
            line_str = f" {line:+g}" if line else ""
            odds     = p.get("dk_odds", 0)
            odds_str = f"+{odds:.0f}" if odds >= 0 else f"{odds:.0f}"
            ev       = p.get("ev_pct", 0)
            prob     = p.get("model_prob", 0)
            stake    = round(bankroll * p.get("kelly_frac", 0), 2)
            commence = p.get("commence", "")

            lines.append(
                f"<b>{i}. [{sport}]</b> {game}\n"
                f"   {bet_type} → <b>{side}{line_str}</b>  {odds_str}\n"
                f"   EV: <b>{ev:+.1f}%</b>  Model: {prob:.1%}  Stake: <b>${stake:.2f}</b>\n"
                f"   🕐 {commence}"
            )

        return self._send("\n\n".join(lines))

    def send_alert(self, message: str) -> bool:
        return self._send(message)
