"""Base strategy interface for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStrategy(ABC):
    """All trading strategies implement check_signal() returning the standard
    signal/skip/error sentinel dicts."""

    @abstractmethod
    async def check_signal(self) -> dict[str, Any] | None:
        """Generate a signal for slot N+1. T-85s before current slot N ends.
        Returns None on hard failure, or a dict with 'skipped' bool + fields."""
        ...
