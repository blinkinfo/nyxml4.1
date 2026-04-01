"""Trade Manager — pre-trade gate with pluggable filters.

Currently implements two chained filters (both controlled by the
same 'n2_filter_enabled' DB toggle, default: true):

  1. **Diff Side from N-2** — Only take a trade if the current signal
     side DIFFERS from the side traded at slot N-2 (two slots ago).
     If N-2 had no trade (skipped, filtered, or bot was offline), the
     filter passes (we allow the trade — no data = no block).

  2. **N-4 Must Be Win** — Only take a trade if the trade placed at
     slot N-4 (four slots ago) resulted in a WIN.  If N-4 had no
     trade or the trade is still unresolved, the filter BLOCKS
     (conservative — require a confirmed win).

  Both filters must pass for a trade to be allowed.
  Toggle stored in DB settings key 'n2_filter_enabled' (default: true).
  When is_demo=True the filters check demo trade history (is_demo=1)
  instead of real trade history (is_demo=0).

Calling convention:
    result = await TradeManager.check(signal_side, current_slot_ts)
    if result.allowed:
        # proceed with trade
    else:
        # log result.reason, record block in DB, notify Telegram
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from db import queries

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterResult:
    allowed: bool
    reason: str
    filter_name: str | None = None
    n2_side: str | None = None   # what N-2 traded (for logging/notification)
    n4_win: bool | None = None   # whether N-4 was a win (for logging/notification)


class TradeManager:
    """Stateless pre-trade gate.  All methods are async class methods."""

    @classmethod
    async def check(
        cls,
        signal_side: str,
        current_slot_ts: int,
        is_demo: bool = False,
    ) -> FilterResult:
        """Run all active filters. Returns FilterResult(allowed=True/False).

        Pass is_demo=True when the scheduler is running the demo trade path
        so the filter lookups query demo trades rather than real trades.

        Filters run in order. First block wins.
        """
        # --- Filter 1: Diff Side from N-2 ---
        n2_result = await cls._check_n2_filter(signal_side, current_slot_ts, is_demo=is_demo)
        if not n2_result.allowed:
            return n2_result

        # --- Filter 2: N-4 Must Be Win ---
        n4_result = await cls._check_n4_win_filter(current_slot_ts, is_demo=is_demo)
        if not n4_result.allowed:
            # Carry forward the n2_side from Filter 1 for complete logging
            return FilterResult(
                allowed=False,
                reason=n4_result.reason,
                filter_name=n4_result.filter_name,
                n2_side=n2_result.n2_side,
                n4_win=n4_result.n4_win,
            )

        # All filters passed
        return FilterResult(
            allowed=True,
            reason="All filters passed",
            n2_side=n2_result.n2_side,
            n4_win=n4_result.n4_win,
        )

    # ------------------------------------------------------------------
    # Filter implementations
    # ------------------------------------------------------------------

    @classmethod
    async def _check_n2_filter(
        cls,
        signal_side: str,
        current_slot_ts: int,
        is_demo: bool = False,
    ) -> FilterResult:
        """Diff Side from N-2 filter.

        Block if: filter enabled AND N-2 trade side == current signal side.
        Pass if:  filter disabled OR N-2 had no trade OR sides differ.

        When is_demo=True, looks up the N-2 side from demo trades only
        (is_demo=1).  When is_demo=False (default), looks up real trades
        (is_demo=0).
        """
        enabled = await queries.is_n2_filter_enabled()
        if not enabled:
            return FilterResult(
                allowed=True,
                reason="N-2 filter disabled",
                filter_name="n2_diff",
            )

        if is_demo:
            n2_side = await queries.get_n2_demo_trade_side(current_slot_ts)
        else:
            n2_side = await queries.get_n2_trade_side(current_slot_ts)

        if n2_side is None:
            # No N-2 trade data — allow (bot was offline, slot was skipped, etc.)
            log.debug(
                "N-2 filter: no %strade found for N-2 slot (ts=%d) — allowing %s",
                "demo " if is_demo else "",
                current_slot_ts - 600,  # approximate, real value computed in query
                signal_side,
            )
            return FilterResult(
                allowed=True,
                reason="N-2 has no trade — filter passes by default",
                filter_name="n2_diff",
                n2_side=None,
            )

        if n2_side == signal_side:
            log.info(
                "N-2 filter BLOCKED: current=%s matches N-2=%s (demo=%s) — skipping trade",
                signal_side,
                n2_side,
                is_demo,
            )
            return FilterResult(
                allowed=False,
                reason=f"N-2 traded {n2_side} — same as current signal ({signal_side})",
                filter_name="n2_diff",
                n2_side=n2_side,
            )

        log.info(
            "N-2 filter PASSED: current=%s differs from N-2=%s (demo=%s) — allowing trade",
            signal_side,
            n2_side,
            is_demo,
        )
        return FilterResult(
            allowed=True,
            reason=f"N-2 traded {n2_side} — differs from current ({signal_side})",
            filter_name="n2_diff",
            n2_side=n2_side,
        )

    @classmethod
    async def _check_n4_win_filter(
        cls,
        current_slot_ts: int,
        is_demo: bool = False,
    ) -> FilterResult:
        """N-4 Must Be Win filter.

        Block if: filter enabled AND (N-4 had no trade OR N-4 trade lost
                  OR N-4 trade is still unresolved).
        Pass if:  filter disabled OR N-4 trade was a confirmed win.

        Conservative edge case: no data at N-4 = BLOCK (require confirmed win).

        When is_demo=True, looks up the N-4 result from demo trades only
        (is_demo=1).  When is_demo=False (default), looks up real trades
        (is_demo=0).
        """
        enabled = await queries.is_n2_filter_enabled()
        if not enabled:
            return FilterResult(
                allowed=True,
                reason="N-4 win filter disabled (shares N-2 toggle)",
                filter_name="n4_win",
            )

        if is_demo:
            n4_win = await queries.get_n4_demo_trade_win(current_slot_ts)
        else:
            n4_win = await queries.get_n4_trade_win(current_slot_ts)

        if n4_win is None:
            # No N-4 trade data or unresolved — BLOCK (conservative)
            log.info(
                "N-4 win filter BLOCKED: no %strade data at N-4 (ts=%d) — blocking (conservative)",
                "demo " if is_demo else "",
                current_slot_ts - 1200,  # approximate
            )
            return FilterResult(
                allowed=False,
                reason="N-4 has no trade or unresolved — blocked (require confirmed win)",
                filter_name="n4_win",
                n4_win=None,
            )

        if not n4_win:
            log.info(
                "N-4 win filter BLOCKED: N-4 trade was a LOSS (demo=%s) — skipping trade",
                is_demo,
            )
            return FilterResult(
                allowed=False,
                reason="N-4 trade was a loss — blocked (require N-4 win)",
                filter_name="n4_win",
                n4_win=False,
            )

        log.info(
            "N-4 win filter PASSED: N-4 trade was a WIN (demo=%s) — allowing trade",
            is_demo,
        )
        return FilterResult(
            allowed=True,
            reason="N-4 trade was a win",
            filter_name="n4_win",
            n4_win=True,
        )
