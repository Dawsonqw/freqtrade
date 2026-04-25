from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

from freqtrade.configuration import TimeRange
from freqtrade.exchange import timeframe_to_seconds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Window:
    index: int
    start: datetime
    end: datetime
    timerange: TimeRange


@dataclass(frozen=True)
class WindowPlan:
    """Result of auto_window_days() — documents how window_days was chosen."""
    window_days: int
    num_windows: int
    total_days: int
    num_pairs: int
    mem_per_window_mb: float
    mem_budget_mb: float
    max_window_days: int
    reason: str


# ─── Constants for memory estimation ────────────────────────────────────────
# 5m candles = 288/day; feather in-memory ~4.8 MB/pair/year ≈ 0.0132 MB/pair/day
# Add ~2x headroom for indicators, intermediate copies, etc.
_CANDLES_PER_DAY_5M = 288
_BYTES_PER_CANDLE = 60  # ~60 bytes per row in memory (OHLCV + date + extras)
_INDICATOR_MULTIPLIER = 2.5  # indicators + copies + strategy overhead
_MB = 1024 * 1024


def _floor_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    secs = timeframe_to_seconds(timeframe)
    ts = int(dt.timestamp())
    return datetime.fromtimestamp(ts - (ts % secs), tz=UTC)


def resolve_global_bounds(timerange: TimeRange, timeframe: str) -> tuple[datetime, datetime]:
    if timerange.startdt is None:
        raise ValueError("Rolling backtest requires bounded start timerange.")
    start = _floor_to_timeframe(timerange.startdt, timeframe)
    if timerange.stopdt is None:
        stop = _floor_to_timeframe(datetime.now(tz=UTC), timeframe)
    else:
        stop = _floor_to_timeframe(timerange.stopdt, timeframe)
    if start >= stop:
        raise ValueError(f"Invalid timerange bounds: start={start}, stop={stop}")
    return start, stop


def estimate_window_memory_mb(
    num_pairs: int,
    window_days: int,
    timeframe: str = "5m",
) -> float:
    """Estimate peak memory (MB) for one window of data + indicators."""
    tf_secs = timeframe_to_seconds(timeframe)
    candles_per_day = 86400 / tf_secs
    candles = num_pairs * candles_per_day * window_days
    raw_mb = candles * _BYTES_PER_CANDLE / _MB
    return raw_mb * _INDICATOR_MULTIPLIER


def auto_window_days(
    timerange: TimeRange,
    *,
    timeframe: str = "5m",
    num_pairs: int,
    mem_budget_mb: float = 0,
    preferred_days: int = 0,
    min_window_days: int = 7,
    max_window_days: int = 90,
) -> WindowPlan:
    """Automatically determine optimal window_days.

    Goals (in priority order):
      1. Memory safety — window fits in mem_budget_mb (default: 50% of free RAM)
      2. Full coverage — total_days is evenly divisible by window_days (no tail gap)
      3. Fewest windows — prefer larger windows to reduce overhead

    Algorithm:
      - Compute total_days from timerange
      - Compute max affordable window_days from memory budget
      - If preferred_days given and fits memory, try to find a nearby divisor
      - Otherwise, find the largest divisor of total_days within [min, max ∩ affordable]
      - If no exact divisor exists, pick the value that minimizes the remainder
        while the last short window is at least 50% of window_days

    Args:
        timerange: start/stop time bounds
        timeframe: candle timeframe (default "5m")
        num_pairs: number of trading pairs
        mem_budget_mb: max memory per window in MB (0 = auto-detect from system)
        preferred_days: user hint for window size (0 = auto)
        min_window_days: minimum acceptable window (default 7)
        max_window_days: maximum acceptable window (default 90)

    Returns:
        WindowPlan with chosen window_days and explanation
    """
    start, stop = resolve_global_bounds(timerange, timeframe)
    total_days = (stop - start).days
    if total_days <= 0:
        raise ValueError(f"Timerange too short: {total_days} days")

    # ── Memory budget ──
    if mem_budget_mb <= 0:
        try:
            import psutil
            free_mb = psutil.virtual_memory().available / _MB
            mem_budget_mb = free_mb * 0.5  # use 50% of free RAM
        except ImportError:
            mem_budget_mb = 2048  # conservative 2GB fallback

    # ── Max affordable window from memory ──
    mem_per_day = estimate_window_memory_mb(num_pairs, 1, timeframe)
    if mem_per_day <= 0:
        affordable_days = max_window_days
    else:
        affordable_days = int(mem_budget_mb / mem_per_day)

    effective_max = min(max_window_days, affordable_days, total_days)
    effective_max = max(effective_max, min_window_days)  # ensure at least min

    reason_parts = []

    # ── If preferred_days fits, try to use it ──
    if preferred_days > 0:
        if preferred_days > affordable_days:
            reason_parts.append(
                f"preferred {preferred_days}d exceeds memory limit "
                f"({estimate_window_memory_mb(num_pairs, preferred_days, timeframe):.0f}MB > {mem_budget_mb:.0f}MB budget)"
            )
        elif preferred_days > total_days:
            reason_parts.append(f"preferred {preferred_days}d > total {total_days}d")
        else:
            # Check if it divides evenly or remainder is acceptable
            remainder = total_days % preferred_days
            if remainder == 0:
                return WindowPlan(
                    window_days=preferred_days,
                    num_windows=total_days // preferred_days,
                    total_days=total_days,
                    num_pairs=num_pairs,
                    mem_per_window_mb=estimate_window_memory_mb(num_pairs, preferred_days, timeframe),
                    mem_budget_mb=mem_budget_mb,
                    max_window_days=effective_max,
                    reason=f"preferred {preferred_days}d divides {total_days}d evenly",
                )
            # Try nearby divisors (±5 days)
            for delta in range(1, 6):
                for candidate in [preferred_days - delta, preferred_days + delta]:
                    if min_window_days <= candidate <= effective_max and total_days % candidate == 0:
                        reason_parts.append(
                            f"adjusted from preferred {preferred_days}d to exact divisor {candidate}d "
                            f"(remainder was {remainder}d)"
                        )
                        return WindowPlan(
                            window_days=candidate,
                            num_windows=total_days // candidate,
                            total_days=total_days,
                            num_pairs=num_pairs,
                            mem_per_window_mb=estimate_window_memory_mb(num_pairs, candidate, timeframe),
                            mem_budget_mb=mem_budget_mb,
                            max_window_days=effective_max,
                            reason="; ".join(reason_parts),
                        )
            # No nearby divisor found, preferred is still usable (tail window ok)
            reason_parts.append(
                f"using preferred {preferred_days}d with {remainder}d tail window"
            )

    # ── Find largest exact divisor in [min, effective_max] ──
    best_divisor = None
    for d in range(effective_max, min_window_days - 1, -1):
        if total_days % d == 0:
            best_divisor = d
            break

    if best_divisor is not None:
        reason_parts.append(
            f"largest exact divisor of {total_days}d in [{min_window_days}, {effective_max}]"
        )
        return WindowPlan(
            window_days=best_divisor,
            num_windows=total_days // best_divisor,
            total_days=total_days,
            num_pairs=num_pairs,
            mem_per_window_mb=estimate_window_memory_mb(num_pairs, best_divisor, timeframe),
            mem_budget_mb=mem_budget_mb,
            max_window_days=effective_max,
            reason="; ".join(reason_parts),
        )

    # ── No exact divisor — minimize waste ──
    # Pick the window_days that has the smallest remainder,
    # but ensure tail >= 50% of window_days
    best_candidate = effective_max
    best_score = total_days  # worst case

    for d in range(effective_max, min_window_days - 1, -1):
        remainder = total_days % d
        if remainder == 0:
            # Already handled above, but just in case
            best_candidate = d
            best_score = 0
            break
        # Tail window quality: prefer remainder close to d (bigger tail)
        # Score: lower = better. Weight remainder penalty + window count
        tail_ratio = remainder / d
        if tail_ratio < 0.3:
            # Very short tail — penalize heavily
            score = (d - remainder) * 2
        else:
            score = d - remainder

        if score < best_score:
            best_score = score
            best_candidate = d

    num_full = total_days // best_candidate
    remainder = total_days % best_candidate
    num_windows = num_full + (1 if remainder > 0 else 0)

    reason_parts.append(
        f"no exact divisor found; {best_candidate}d gives {num_full} full + "
        f"1×{remainder}d tail (tail ratio {remainder/best_candidate:.0%})"
    )

    return WindowPlan(
        window_days=best_candidate,
        num_windows=num_windows,
        total_days=total_days,
        num_pairs=num_pairs,
        mem_per_window_mb=estimate_window_memory_mb(num_pairs, best_candidate, timeframe),
        mem_budget_mb=mem_budget_mb,
        max_window_days=effective_max,
        reason="; ".join(reason_parts),
    )


def build_windows(
    timerange: TimeRange,
    *,
    timeframe: str,
    window_days: int = 0,
    num_pairs: int = 0,
    mem_budget_mb: float = 0,
    preferred_days: int = 0,
    min_window_days: int = 7,
    max_window_days: int = 90,
) -> tuple[list[Window], Optional[WindowPlan]]:
    """Build window list for rolling backtest.

    If window_days > 0, uses it directly (legacy behavior).
    If window_days <= 0 (or omitted), auto-calculates optimal window size.

    Returns:
        (windows, plan) — plan is None when window_days was explicit.
    """
    plan: Optional[WindowPlan] = None

    if window_days <= 0:
        if num_pairs <= 0:
            raise ValueError("num_pairs is required when window_days is auto (0)")
        plan = auto_window_days(
            timerange,
            timeframe=timeframe,
            num_pairs=num_pairs,
            mem_budget_mb=mem_budget_mb,
            preferred_days=preferred_days,
            min_window_days=min_window_days,
            max_window_days=max_window_days,
        )
        window_days = plan.window_days
        logger.info(
            "Auto window: %dd × %d windows (%s) | mem %.0f MB/window, budget %.0f MB | %s",
            plan.window_days,
            plan.num_windows,
            f"{plan.total_days}d total",
            plan.mem_per_window_mb,
            plan.mem_budget_mb,
            plan.reason,
        )

    start, stop = resolve_global_bounds(timerange, timeframe)
    delta = timedelta(days=window_days)
    windows: list[Window] = []
    cursor = start
    idx = 0
    while cursor < stop:
        nxt = min(cursor + delta, stop)
        # Skip degenerate windows (< 1 day) unless it's the only window
        if (nxt - cursor).total_seconds() < 86400 and idx > 0:
            # Merge into previous window
            if windows:
                prev = windows[-1]
                windows[-1] = Window(
                    index=prev.index,
                    start=prev.start,
                    end=nxt,
                    timerange=TimeRange("date", "date", int(prev.start.timestamp()), int(nxt.timestamp())),
                )
                logger.info(
                    "Merged short tail (%s → %s) into window %d",
                    cursor.strftime("%Y-%m-%d"),
                    nxt.strftime("%Y-%m-%d"),
                    prev.index,
                )
            break
        windows.append(
            Window(
                index=idx,
                start=cursor,
                end=nxt,
                timerange=TimeRange("date", "date", int(cursor.timestamp()), int(nxt.timestamp())),
            )
        )
        cursor = nxt
        idx += 1
    return windows, plan
