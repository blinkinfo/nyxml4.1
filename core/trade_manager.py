"""Trade Manager — pre-trade gate (now a passthrough).

All pre-trade filters (N-2 diff, N-4 win) have been removed.
The new pattern-based strategy handles all entry logic internally.
TradeManager always returns allowed=True for backward compatibility
with the scheduler flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterResult:
    allowed: bool
    reason: str
    filter_name: str | None = None
    n2_side: str | None = None
    n4_win: bool | None = None


class TradeManager:
    """Stateless pre-trade gate — now always passes.

    Kept as a class so the scheduler import chain stays intact.
    The pattern strategy handles all entry decisions internally.
    """

    @classmethod
    async def check(
        cls,
        signal_side: str,
        current_slot_ts: int,
        is_demo: bool = False,
    ) -> FilterResult:
        """Always return allowed=True. Filters removed in favour of
        pattern-based strategy logic."""
        return FilterResult(
            allowed=True,
            reason="No filters — passthrough",
        )
