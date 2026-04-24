from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WindowPerformance:
    """Performance metrics for a single backtest window."""

    window_index: int
    window_start: str
    window_end: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit_pct: float = 0.0
    total_profit_abs: float = 0.0
    avg_profit_pct: float = 0.0
    avg_profit_abs: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_abs: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_min: float = 0.0
    carry_over_trades: int = 0  # trades opened before this window
    # Breakdown by direction
    long_trades: int = 0
    short_trades: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _annualized_sharpe(returns: np.ndarray, periods_per_year: float = 365.25) -> float:
    """Calculate annualized Sharpe ratio from trade returns."""
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r == 0:
        return 0.0
    return float(mean_r / std_r * np.sqrt(periods_per_year))


def _annualized_sortino(returns: np.ndarray, periods_per_year: float = 365.25) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if mean_r > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(mean_r / downside_std * np.sqrt(periods_per_year))


def _max_drawdown_from_returns(
    returns: np.ndarray, starting_balance: float
) -> tuple[float, float]:
    """Calculate max drawdown (as ratio and absolute) from trade returns.

    Returns (max_drawdown_ratio, max_drawdown_abs).
    """
    if len(returns) == 0:
        return 0.0, 0.0

    equity = starting_balance
    peak = equity
    max_dd_abs = 0.0

    for r in returns:
        equity += r * starting_balance  # approximate: each trade uses starting balance
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd_abs:
            max_dd_abs = dd

    max_dd_ratio = max_dd_abs / starting_balance if starting_balance > 0 else 0.0
    return max_dd_ratio, max_dd_abs


def compute_window_performance(
    trades: pd.DataFrame,
    window_index: int,
    window_start: datetime,
    window_end: datetime,
    starting_balance: float = 1000.0,
) -> WindowPerformance:
    """Compute performance metrics for trades within a window.

    Trades are assigned to the window where they CLOSE
    (close_date within [window_start, window_end)).
    """
    perf = WindowPerformance(
        window_index=window_index,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
    )

    if trades is None or trades.empty:
        return perf

    # Filter trades that closed in this window
    ws = pd.Timestamp(window_start).tz_localize("UTC") if pd.Timestamp(window_start).tzinfo is None else pd.Timestamp(window_start)
    we = pd.Timestamp(window_end).tz_localize("UTC") if pd.Timestamp(window_end).tzinfo is None else pd.Timestamp(window_end)
    if "close_date" in trades.columns:
        close_dates = pd.to_datetime(trades["close_date"], utc=True)
        mask = (close_dates >= ws) & (close_dates < we)
        window_trades = trades.loc[mask].copy()
    else:
        window_trades = trades.copy()

    if window_trades.empty:
        return perf

    n = len(window_trades)
    perf.total_trades = n

    # Profit ratios
    profit_ratios = window_trades["profit_ratio"].values.astype(float)
    profit_abs = (
        window_trades["profit_abs"].values.astype(float)
        if "profit_abs" in window_trades.columns
        else profit_ratios * starting_balance
    )

    perf.winning_trades = int(np.sum(profit_ratios > 0))
    perf.losing_trades = int(np.sum(profit_ratios < 0))
    perf.win_rate = perf.winning_trades / n if n > 0 else 0.0

    perf.total_profit_pct = float(np.sum(profit_ratios) * 100)
    perf.total_profit_abs = float(np.sum(profit_abs))
    perf.avg_profit_pct = float(np.mean(profit_ratios) * 100)
    perf.avg_profit_abs = float(np.mean(profit_abs))
    perf.best_trade_pct = float(np.max(profit_ratios) * 100)
    perf.worst_trade_pct = float(np.min(profit_ratios) * 100)

    # Risk metrics
    perf.sharpe_ratio = _annualized_sharpe(profit_ratios)
    perf.sortino_ratio = _annualized_sortino(profit_ratios)

    dd_ratio, dd_abs = _max_drawdown_from_returns(profit_ratios, starting_balance)
    perf.max_drawdown = dd_ratio
    perf.max_drawdown_abs = dd_abs

    # Calmar = annualized return / max drawdown
    total_days = max((window_end - window_start).total_seconds() / 86400, 1)
    annual_return = float(np.sum(profit_ratios)) * (365.25 / total_days)
    perf.calmar_ratio = annual_return / dd_ratio if dd_ratio > 0 else 0.0

    # Profit factor
    gross_profit = float(np.sum(profit_abs[profit_abs > 0]))
    gross_loss = float(np.abs(np.sum(profit_abs[profit_abs < 0])))
    perf.profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )

    # Duration
    if "trade_duration" in window_trades.columns:
        perf.avg_trade_duration_min = float(window_trades["trade_duration"].mean())

    # Carry-over: trades whose open_date is before window_start
    if "open_date" in window_trades.columns:
        open_dates = pd.to_datetime(window_trades["open_date"], utc=True)
        perf.carry_over_trades = int((open_dates < ws).sum())

    # Direction breakdown
    if "is_short" in window_trades.columns:
        perf.short_trades = int(window_trades["is_short"].sum())
        perf.long_trades = n - perf.short_trades

    return perf


def compute_all_window_performances(
    all_trades: pd.DataFrame,
    window_boundaries: list[tuple[int, datetime, datetime]],
    starting_balance: float = 1000.0,
) -> list[WindowPerformance]:
    """Compute performance for all windows.

    Args:
        all_trades: All trades from the backtest run.
        window_boundaries: List of (index, start, end) tuples.
        starting_balance: Initial wallet balance.

    Returns:
        List of WindowPerformance, one per window.
    """
    results = []
    for idx, start, end in window_boundaries:
        perf = compute_window_performance(
            trades=all_trades,
            window_index=idx,
            window_start=start,
            window_end=end,
            starting_balance=starting_balance,
        )
        results.append(perf)
    return results
