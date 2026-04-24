"""Walk-Forward Optimization — IS/OOS window pair generation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from freqtrade.configuration import TimeRange
from freqtrade.exchange import timeframe_to_seconds


@dataclass(frozen=True)
class WFOFold:
    """One Walk-Forward fold with IS (in-sample) and OOS (out-of-sample) ranges."""
    index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime

    @property
    def is_timerange(self) -> TimeRange:
        return TimeRange("date", "date", int(self.is_start.timestamp()), int(self.is_end.timestamp()))

    @property
    def oos_timerange(self) -> TimeRange:
        return TimeRange("date", "date", int(self.oos_start.timestamp()), int(self.oos_end.timestamp()))

    @property
    def is_days(self) -> float:
        return (self.is_end - self.is_start).total_seconds() / 86400

    @property
    def oos_days(self) -> float:
        return (self.oos_end - self.oos_start).total_seconds() / 86400


def _floor_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    """Floor a datetime to the nearest timeframe boundary."""
    secs = timeframe_to_seconds(timeframe)
    ts = int(dt.timestamp())
    return datetime.fromtimestamp(ts - (ts % secs), tz=UTC)


def build_wfo_folds(
    timerange: TimeRange,
    *,
    timeframe: str,
    is_days: int,
    oos_days: int,
    step_days: int | None = None,
) -> list[WFOFold]:
    """Build Walk-Forward folds from a global timerange.

    Args:
        timerange: Global data timerange.
        timeframe: Candle timeframe (e.g. "5m").
        is_days: In-sample window length in days.
        oos_days: Out-of-sample window length in days.
        step_days: Slide step in days. Defaults to oos_days (non-overlapping OOS).

    Returns:
        List of WFOFold with IS/OOS ranges.

    Raises:
        ValueError: If timerange is too short for even one fold.
    """
    if step_days is None:
        step_days = oos_days

    if timerange.startdt is None:
        raise ValueError("WFO requires bounded start timerange.")
    start = _floor_to_timeframe(timerange.startdt, timeframe)
    if timerange.stopdt is None:
        stop = _floor_to_timeframe(datetime.now(tz=UTC), timeframe)
    else:
        stop = _floor_to_timeframe(timerange.stopdt, timeframe)

    is_delta = timedelta(days=is_days)
    oos_delta = timedelta(days=oos_days)
    step_delta = timedelta(days=step_days)
    min_required = is_delta + oos_delta

    if (stop - start) < min_required:
        raise ValueError(
            f"Timerange too short for WFO: need >= {is_days + oos_days} days, "
            f"got {(stop - start).days} days"
        )

    folds: list[WFOFold] = []
    cursor = start
    idx = 0

    while True:
        is_start = cursor
        is_end = cursor + is_delta
        oos_start = is_end
        oos_end = min(is_end + oos_delta, stop)

        # Need at least 1 day of OOS
        if oos_end <= oos_start or oos_start >= stop:
            break

        folds.append(WFOFold(
            index=idx,
            is_start=_floor_to_timeframe(is_start, timeframe),
            is_end=_floor_to_timeframe(is_end, timeframe),
            oos_start=_floor_to_timeframe(oos_start, timeframe),
            oos_end=_floor_to_timeframe(oos_end, timeframe),
        ))
        cursor += step_delta
        idx += 1

    return folds
