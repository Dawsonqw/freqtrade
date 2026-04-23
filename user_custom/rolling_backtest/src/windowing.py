from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from freqtrade.configuration import TimeRange
from freqtrade.exchange import timeframe_to_seconds


@dataclass(frozen=True)
class Window:
    index: int
    start: datetime
    end: datetime
    timerange: TimeRange


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


def build_windows(
    timerange: TimeRange,
    *,
    timeframe: str,
    window_days: int,
) -> list[Window]:
    start, stop = resolve_global_bounds(timerange, timeframe)
    delta = timedelta(days=window_days)
    windows: list[Window] = []
    cursor = start
    idx = 0
    while cursor < stop:
        nxt = min(cursor + delta, stop)
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
    return windows
