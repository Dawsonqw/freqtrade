# tests/test_comparison_report.py
import pytest
from datetime import datetime, timezone

from user_custom.rolling_backtest.src.window_metrics import WindowPerformance
from user_custom.rolling_backtest.src.comparison_report import (
    StrategyWindowComparison,
    build_comparison_report,
    format_comparison_table,
)


def _make_perf(idx: int, trades: int, win_rate: float, profit: float, sharpe: float) -> WindowPerformance:
    return WindowPerformance(
        window_index=idx,
        window_start=f"2024-0{idx+1}-01T00:00:00+00:00",
        window_end=f"2024-0{idx+1}-15T00:00:00+00:00",
        total_trades=trades,
        winning_trades=int(trades * win_rate),
        losing_trades=trades - int(trades * win_rate),
        win_rate=win_rate,
        total_profit_pct=profit,
        sharpe_ratio=sharpe,
        sortino_ratio=sharpe * 1.2,
        max_drawdown=abs(profit) * 0.3 if profit < 0 else 0.05,
        profit_factor=1.5 if profit > 0 else 0.8,
    )


def test_build_comparison_report_basic():
    strat_a_perfs = [_make_perf(0, 20, 0.6, 5.0, 1.2), _make_perf(1, 15, 0.5, -2.0, -0.5)]
    strat_b_perfs = [_make_perf(0, 18, 0.7, 8.0, 2.0), _make_perf(1, 12, 0.4, -5.0, -1.0)]

    report = build_comparison_report({
        "StratA": strat_a_perfs,
        "StratB": strat_b_perfs,
    })
    assert len(report.windows) == 2
    assert len(report.strategies) == 2
    # Window 0: StratB wins on profit
    w0 = report.windows[0]
    assert w0.best_strategy_profit == "StratB"
    assert w0.best_strategy_sharpe == "StratB"


def test_build_comparison_report_single_strategy():
    """Single strategy should still work (no comparison, just report)."""
    perfs = [_make_perf(0, 10, 0.5, 1.0, 0.5)]
    report = build_comparison_report({"Only": perfs})
    assert len(report.windows) == 1
    assert report.windows[0].best_strategy_profit == "Only"


def test_build_comparison_report_overall_ranking():
    strat_a = [_make_perf(0, 20, 0.6, 5.0, 1.2), _make_perf(1, 15, 0.5, 3.0, 0.8)]
    strat_b = [_make_perf(0, 18, 0.7, 8.0, 2.0), _make_perf(1, 12, 0.4, -5.0, -1.0)]

    report = build_comparison_report({"A": strat_a, "B": strat_b})
    # A total profit = 8.0, B total = 3.0
    # Overall ranking by total profit
    assert report.overall_ranking[0].strategy == "A"


def test_format_comparison_table():
    strat_a = [_make_perf(0, 20, 0.6, 5.0, 1.2)]
    strat_b = [_make_perf(0, 18, 0.7, 8.0, 2.0)]
    report = build_comparison_report({"A": strat_a, "B": strat_b})
    table = format_comparison_table(report)
    assert "A" in table
    assert "B" in table
    assert "Window 0" in table or "Win 0" in table
