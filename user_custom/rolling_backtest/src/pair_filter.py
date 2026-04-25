"""Pair availability filter for rolling backtests.

Loads pair_availability.json once and provides a fast lookup to determine
which pairs have data for a given time window and timeframe.

Usage:
    from pair_filter import PairAvailabilityFilter

    pf = PairAvailabilityFilter("/data/freqtrade_data/_meta/pair_availability.json")
    active_pairs = pf.filter_pairs(
        pairs=["BTC/USDT:USDT", "ETH/USDT:USDT", "0G/USDT:USDT"],
        window_start="2024-01-01",
        window_end="2024-06-01",
        timeframe="5m",  # optional, defaults to any
    )
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("/data/freqtrade_data/_meta/pair_availability.json")


class PairAvailabilityFilter:
    """Fast in-memory filter based on pre-computed pair availability data."""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._pairs: dict[str, dict] = data.get("pairs", {})
        self._generated_at = data.get("generated_at", "unknown")
        logger.info(
            "Loaded pair availability: %d pairs (generated %s)",
            len(self._pairs),
            self._generated_at,
        )

    @property
    def all_pairs(self) -> list[str]:
        """Return all known pair names."""
        return list(self._pairs.keys())

    def pair_info(self, pair: str) -> dict | None:
        """Return raw availability info for a pair, or None."""
        return self._pairs.get(pair)

    @staticmethod
    def _parse_dt(s: str | datetime) -> datetime:
        if isinstance(s, datetime):
            if s.tzinfo is None:
                return s.replace(tzinfo=timezone.utc)
            return s
        # Handle ISO format strings
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def is_available(
        self,
        pair: str,
        window_start: str | datetime,
        window_end: str | datetime | None = None,
        timeframe: str | None = None,
    ) -> bool:
        """Check if a pair has data covering (at least overlapping) the window.

        A pair is considered available if its data range overlaps with the
        requested window. Specifically:
            pair.start <= window_end AND pair.end >= window_start

        If timeframe is given, checks that specific timeframe's range.
        Otherwise checks the pair's overall earliest_start / latest_end.
        """
        info = self._pairs.get(pair)
        if info is None:
            return False

        ws = self._parse_dt(window_start)
        we = self._parse_dt(window_end) if window_end else ws

        if timeframe and timeframe in info.get("timeframes", {}):
            tf_info = info["timeframes"][timeframe]
            p_start = self._parse_dt(tf_info["start"])
            p_end = self._parse_dt(tf_info["end"])
        else:
            p_start = self._parse_dt(info["earliest_start"])
            p_end = self._parse_dt(info["latest_end"])

        # Overlap check: pair data must start before window ends,
        # and pair data must end after window starts
        return p_start <= we and p_end >= ws

    def filter_pairs(
        self,
        pairs: Sequence[str],
        window_start: str | datetime,
        window_end: str | datetime | None = None,
        timeframe: str | None = None,
    ) -> list[str]:
        """Return only pairs that have data overlapping the given window."""
        result = [
            p for p in pairs
            if self.is_available(p, window_start, window_end, timeframe)
        ]
        logger.debug(
            "filter_pairs: %d/%d pairs available for %s ~ %s (tf=%s)",
            len(result), len(pairs), window_start, window_end, timeframe,
        )
        return result

    def get_available_pairs(
        self,
        window_start: str | datetime,
        window_end: str | datetime | None = None,
        timeframe: str | None = None,
    ) -> list[str]:
        """Return ALL known pairs that have data overlapping the window."""
        return self.filter_pairs(self.all_pairs, window_start, window_end, timeframe)
