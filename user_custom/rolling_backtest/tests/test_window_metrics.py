import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from user_custom.rolling_backtest.src.window_metrics import (
    WindowPerformance,
    compute_window_performance,
    compute_all_window_performances,
)


def _make_trades(n_trades: int, win_rate: float = 0.6, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic trade data for testing."""
    rng = np.random.RandomState(seed)
    profits = []
    for _ in range(n_trades):
        if rng.random() < win_rate:
            profits.append(rng.uniform(0.001, 0.05))
        else:
            profits.append(rng.uniform(-0.05, -0.001))

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    records = []
    for i, p in enumerate(profits):
        open_dt = base + timedelta(hours=i * 4)
        close_dt = open_dt + timedelta(hours=2)
        records.append({
            "pair": "BTC/USDT:USDT",
            "open_date": open_dt,
            "close_date": close_dt,
            "profit_ratio": p,
            "profit_abs": p * 1000,
            "trade_duration": 120,
            "is_short": False,
            "exit_reason": "roi" if p > 0 else "stop_loss",
            "stake_amount": 100.0,
        })
    return pd.DataFrame(records)


def test_compute_window_performance_basic():
    trades = _make_trades(20, win_rate=0.7)
    perf = compute_window_performance(
        trades=trades,
        window_index=0,
        window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2024, 1, 15, tzinfo=timezone.utc),
        starting_balance=1000.0,
    )
    assert isinstance(perf, WindowPerformance)
    assert perf.window_index == 0
    assert perf.total_trades == 20
    assert 0 <= perf.win_rate <= 1
    assert perf.sharpe_ratio is not None
    assert perf.sortino_ratio is not None
    assert perf.max_drawdown >= 0


def test_compute_window_performance_no_trades():
    """Empty trades should produce zero metrics, not crash."""
    trades = pd.DataFrame()
    perf = compute_window_performance(
        trades=trades,
        window_index=0,
        window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        starting_balance=1000.0,
    )
    assert perf.total_trades == 0
    assert perf.win_rate == 0.0
    assert perf.sharpe_ratio == 0.0
    assert perf.max_drawdown == 0.0


def test_compute_all_window_performances():
    """Test batch computation across multiple windows."""
    trades = _make_trades(50, seed=123)
    window_boundaries = [
        (0, datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 4, tzinfo=timezone.utc)),
        (1, datetime(2024, 1, 4, tzinfo=timezone.utc), datetime(2024, 1, 7, tzinfo=timezone.utc)),
        (2, datetime(2024, 1, 7, tzinfo=timezone.utc), datetime(2024, 1, 10, tzinfo=timezone.utc)),
    ]
    results = compute_all_window_performances(
        all_trades=trades,
        window_boundaries=window_boundaries,
        starting_balance=1000.0,
    )
    assert len(results) == 3
    total = sum(r.total_trades for r in results)
    assert total <= 50


def test_window_performance_profit_factor():
    trades = _make_trades(30, win_rate=0.8)
    perf = compute_window_performance(
        trades=trades, window_index=0,
        window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        starting_balance=1000.0,
    )
    assert perf.profit_factor > 0


def test_window_performance_carry_over_trades():
    """Trades that started before the window should be flagged as carry-over."""
    trades = _make_trades(10)
    trades.loc[:2, "open_date"] = datetime(2023, 12, 28, tzinfo=timezone.utc)
    perf = compute_window_performance(
        trades=trades, window_index=0,
        window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        starting_balance=1000.0,
    )
    assert perf.carry_over_trades == 3
