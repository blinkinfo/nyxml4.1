"""Pattern strategy -- 6-candle historical pattern matching on BTC-USD.

Flow:
1. Fetch 10 most recently closed 5-min BTC-USD candles from Coinbase
2. Read directions of N-1 through N-6 (6 most recent fully closed)
3. Build 6-char string (N-1 N-2 ... N-6, left to right) using U for up, D for down
4. Look up string in pattern table
5. If match -> trade Predicted direction for N+1 candle
6. If no match -> skip

Candle direction: close >= open means U (up), close < open means D (down).
"""

from __future__ import annotations

import logging
from typing import Any

import config as cfg
import httpx
from polymarket.markets import get_next_slot_info, get_slot_prices

log = logging.getLogger(__name__)


# Pattern table: 6-char string -> predicted direction for N+1
PATTERN_TABLE = {
    "DDDDDD": "UP",
    "DUUUDU": "DOWN",
    "DUUUUD": "DOWN",
    "UDDUUU": "UP",
    "DUDDUD": "DOWN",
    "DUUUDD": "DOWN",
    "UDDUUD": "UP",
    "DUDUDU": "DOWN",
    "UDDDDU": "UP",
    "UUDUUU": "DOWN",
    "DDUDDU": "UP",
    "UUUDUD": "DOWN",
    "DUDUUU": "UP",
    "UUUUUD": "DOWN",
    "DDDUUD": "DOWN",
    "UDUUDU": "DOWN",
    "DUUDDD": "UP",
    "UDDUDD": "DOWN",
    "DUUUUU": "DOWN",
    "UUDUUD": "UP",
    "DDUDDD": "UP",
    "DUDDDU": "DOWN",
}


async def _fetch_candles(count: int = 10) -> list[dict[str, float]] | None:
    """Fetch *count* most recently closed 5-min BTC-USD candles from Coinbase.

    Requests 300-candle window from Coinbase, returns the *count* most
    recent fully closed candles (oldest first).
    """
    import time as _time

    granularity = 300
    end_ts = int(_time.time())
    start_ts = end_ts - 300 * granularity

    params = {
        "granularity": granularity,
        "start": start_ts,
        "end": end_ts,
    }

    try:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            resp = await client.get(cfg.COINBASE_CANDLE_URL, params=params)
            resp.raise_for_status()
            raw = resp.json()
    except Exception:
        log.exception("Coinbase candle fetch failed")
        return None

    if not raw or not isinstance(raw, list):
        log.error("Coinbase returned empty or invalid response")
        return None

    candles = []
    for row in raw:
        try:
            candles.append({
                "time": float(row[0]),
                "low":  float(row[1]),
                "high": float(row[2]),
                "open": float(row[3]),
                "close": float(row[4]),
            })
        except (IndexError, ValueError, TypeError):
            continue

    candles.reverse()
    return candles


def _build_pattern_string(candles: list[dict[str, float]], depth: int = 6) -> str | None:
    """Build an *depth*-character pattern string from candle directions.

    Takes the *depth* most recent candles (index -depth to -1), oldest to
    newest, and assigns U (close >= open) or D (close < open).

    The returned string is left-to-right: most recent closed candle first,
    then the one before it, etc.

    So for depth=6 with candles list [oldest ... newest]:
      result = direction(candles[-1]) + direction(candles[-2]) + ... + direction(candles[-6])
    """
    if len(candles) < depth:
        log.warning(
            "Not enough closed candles to build pattern: have %d, need %d",
            len(candles), depth
        )
        return None

    pattern = ""
    for i in range(depth):
        candle = candles[-1 - i]
        direction = "U" if candle["close"] >= candle["open"] else "D"
        pattern += direction

    return pattern


class PatternStrategy:
    """6-candle historical pattern matching strategy."""

    async def check_signal(self) -> dict[str, Any] | None:
        """Pattern-based signal for slot N+1.

        1. Fetch 10 latest closed candles
        2. Build 6-char pattern from N-1..N-6
        3. Look up in pattern table
        4. If match -> fetch Polymarket prices for token_id and entry_price
        5. If no match -> skip
        """
        candles = await _fetch_candles(count=10)
        if candles is None:
            log.error("Pattern strategy: could not fetch candles")
            return None

        pattern = _build_pattern_string(candles, depth=6)
        if pattern is None:
            log.error("Pattern strategy: could not build pattern string")
            return None

        prediction = PATTERN_TABLE.get(pattern)
        if prediction is None:
            slot_n1 = get_next_slot_info()
            log.info(
                "Pattern strategy: pattern '%s' not in table -> SKIP",
                pattern
            )
            return {
                "skipped": True,
                "pattern": pattern,
                "candles_used": len(candles[-6:]) if len(candles) >= 6 else len(candles),
                "slot_n1_start_full": slot_n1["slot_start_full"],
                "slot_n1_end_full": slot_n1["slot_end_full"],
                "slot_n1_start_str": slot_n1["slot_start_str"],
                "slot_n1_end_str": slot_n1["slot_end_str"],
                "slot_n1_ts": slot_n1["slot_start_ts"],
            }

        # Normalize prediction to "Up" or "Down"
        side = "Up" if prediction == "UP" else "Down"

        # Fetch Polymarket prices for N+1 slot (needed for token_id and entry_price)
        slot_n1 = get_next_slot_info()
        prices = await get_slot_prices(slot_n1["slug"])
        if prices is None:
            log.error(
                "Pattern strategy: matched pattern '%s' -> %s but "
                "could not fetch Polymarket prices for slot %s",
                pattern, prediction, slot_n1["slug"]
            )
            return None

        # Use actual ask price from Polymarket as entry_price
        entry_price = prices["up_price"] if side == "Up" else prices["down_price"]
        opposite_price = prices["down_price"] if side == "Up" else prices["up_price"]
        token_id = prices["up_token_id"] if side == "Up" else prices["down_token_id"]

        log.info(
            "Pattern strategy: MATCH '%s' -> %s for slot %s-%s UTC  "
            "entry=$%.4f (market ask)  token=%s",
            pattern,
            prediction,
            slot_n1["slot_start_str"],
            slot_n1["slot_end_str"],
            entry_price,
            token_id,
        )

        return {
            "skipped": False,
            "side": side,
            "entry_price": entry_price,
            "opposite_price": opposite_price,
            "token_id": token_id,
            "pattern": pattern,
            "candles_used": len(candles[-6:]) if len(candles) >= 6 else len(candles),
            "slot_n1_start_full": slot_n1["slot_start_full"],
            "slot_n1_end_full": slot_n1["slot_end_full"],
            "slot_n1_start_str": slot_n1["slot_start_str"],
            "slot_n1_end_str": slot_n1["slot_end_str"],
            "slot_n1_ts": slot_n1["slot_start_ts"],
            "slot_n1_slug": slot_n1["slug"],
        }
