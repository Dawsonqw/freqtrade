"""Integration tests for enhanced features:
1. Window summary table with per-window trades (not cumulative) + carry-over
2. Comparison report with WinRate/MaxDD sections + delta columns
3. End-to-end: WindowPerformance -> comparison_report -> formatted table
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from user_custom.rolling_backtest.src.window_metrics import (
    WindowPerformance,
    compute_all_window_performances,
    compute_window_performance,
)
from user_custom.rolling_backtest.src.comparison_report import (
    build_comparison_report,
    format_comparison_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trades_df(
    trades: list[dict],
) -> pd.DataFrame:
    """Build a minimal trades DataFrame from dicts with sensible defaults."""
    rows = []
    for i, t in enumerate(trades):
        rows.append({
            "pair": t.get("pair", "BTC/USDT:USDT"),
            "open_date": pd.Timestamp(t["open_date"], tz="UTC"),
            "close_date": pd.Timestamp(t["close_date"], tz="UTC"),
            "profit_ratio": t["profit_ratio"],
            "profit_abs": t.get("profit_abs", t["profit_ratio"] * 1000),
            "trade_duration": t.get("trade_duration", 60),
            "is_short": t.get("is_short", False),
        })
    return pd.DataFrame(rows)


def _make_window_perf(
    idx: int,
    start: str,
    end: str,
    *,
    total_trades: int = 10,
    winning_trades: int = 6,
    win_rate: float = 0.6,
    total_profit_pct: float = 5.0,
    sharpe_ratio: float = 1.5,
    sortino_ratio: float = 2.0,
    max_drawdown: float = 0.05,
    profit_factor: float = 2.0,
    carry_over_trades: int = 0,
) -> WindowPerformance:
    return WindowPerformance(
        window_index=idx,
        window_start=start,
        window_end=end,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=total_trades - winning_trades,
        win_rate=win_rate,
        total_profit_pct=total_profit_pct,
        total_profit_abs=total_profit_pct * 10,
        avg_profit_pct=total_profit_pct / max(total_trades, 1),
        avg_profit_abs=total_profit_pct * 10 / max(total_trades, 1),
        best_trade_pct=3.0,
        worst_trade_pct=-1.5,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=1.0,
        max_drawdown=max_drawdown,
        max_drawdown_abs=max_drawdown * 1000,
        profit_factor=profit_factor,
        avg_trade_duration_min=120,
        carry_over_trades=carry_over_trades,
        long_trades=total_trades,
        short_trades=0,
    )


# ---------------------------------------------------------------------------
# Tests: Window metrics with carry-over trades
# ---------------------------------------------------------------------------


class TestWindowMetricsCarryOver:
    """Test that carry_over_trades counts trades opened before window start."""

    def test_carry_over_counted_correctly(self):
        """Trades opened before window_start but closing in window are carry-over."""
        trades = _make_trades_df([
            # Opened before window, closes in window → carry-over
            {"open_date": "2024-01-10", "close_date": "2024-02-05", "profit_ratio": 0.02},
            {"open_date": "2024-01-20", "close_date": "2024-02-10", "profit_ratio": -0.01},
            # Opened in window, closes in window → NOT carry-over
            {"open_date": "2024-02-01", "close_date": "2024-02-15", "profit_ratio": 0.03},
            {"open_date": "2024-02-05", "close_date": "2024-02-20", "profit_ratio": 0.01},
        ])
        ws = datetime(2024, 2, 1, tzinfo=timezone.utc)
        we = datetime(2024, 3, 1, tzinfo=timezone.utc)

        perf = compute_window_performance(trades, window_index=1, window_start=ws, window_end=we)
        assert perf.total_trades == 4
        assert perf.carry_over_trades == 2  # 2 trades opened before Feb 1

    def test_no_carry_over_when_all_open_in_window(self):
        trades = _make_trades_df([
            {"open_date": "2024-02-01", "close_date": "2024-02-15", "profit_ratio": 0.02},
            {"open_date": "2024-02-10", "close_date": "2024-02-20", "profit_ratio": -0.01},
        ])
        ws = datetime(2024, 2, 1, tzinfo=timezone.utc)
        we = datetime(2024, 3, 1, tzinfo=timezone.utc)

        perf = compute_window_performance(trades, window_index=0, window_start=ws, window_end=we)
        assert perf.carry_over_trades == 0

    def test_all_carry_over(self):
        """All trades opened before window → all are carry-over."""
        trades = _make_trades_df([
            {"open_date": "2024-01-01", "close_date": "2024-02-05", "profit_ratio": 0.05},
            {"open_date": "2024-01-15", "close_date": "2024-02-10", "profit_ratio": -0.02},
        ])
        ws = datetime(2024, 2, 1, tzinfo=timezone.utc)
        we = datetime(2024, 3, 1, tzinfo=timezone.utc)

        perf = compute_window_performance(trades, window_index=2, window_start=ws, window_end=we)
        assert perf.carry_over_trades == 2
        assert perf.total_trades == 2


class TestWindowMetricsPerWindow:
    """Test that window metrics are per-window (not cumulative)."""

    def test_trades_assigned_by_close_date(self):
        """Trades are assigned to the window where they close, not open."""
        trades = _make_trades_df([
            # Window 0: closes in Jan
            {"open_date": "2024-01-01", "close_date": "2024-01-15", "profit_ratio": 0.05},
            {"open_date": "2024-01-10", "close_date": "2024-01-25", "profit_ratio": -0.02},
            # Window 1: closes in Feb
            {"open_date": "2024-01-20", "close_date": "2024-02-10", "profit_ratio": 0.03},
            {"open_date": "2024-02-01", "close_date": "2024-02-20", "profit_ratio": 0.01},
            {"open_date": "2024-02-05", "close_date": "2024-02-25", "profit_ratio": -0.01},
        ])

        boundaries = [
            (0, datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc)),
            (1, datetime(2024, 2, 1, tzinfo=timezone.utc), datetime(2024, 3, 1, tzinfo=timezone.utc)),
        ]

        perfs = compute_all_window_performances(trades, boundaries)
        assert len(perfs) == 2
        assert perfs[0].total_trades == 2  # window 0: 2 trades
        assert perfs[1].total_trades == 3  # window 1: 3 trades

    def test_profit_is_per_window(self):
        """Total profit should be per-window, not cumulative."""
        trades = _make_trades_df([
            {"open_date": "2024-01-01", "close_date": "2024-01-15", "profit_ratio": 0.10},
            {"open_date": "2024-02-01", "close_date": "2024-02-15", "profit_ratio": -0.05},
        ])

        boundaries = [
            (0, datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc)),
            (1, datetime(2024, 2, 1, tzinfo=timezone.utc), datetime(2024, 3, 1, tzinfo=timezone.utc)),
        ]

        perfs = compute_all_window_performances(trades, boundaries)
        assert perfs[0].total_profit_pct == pytest.approx(10.0, abs=0.01)  # 0.10 * 100
        assert perfs[1].total_profit_pct == pytest.approx(-5.0, abs=0.01)  # -0.05 * 100


# ---------------------------------------------------------------------------
# Tests: Enhanced comparison report
# ---------------------------------------------------------------------------


class TestComparisonReportEnhanced:
    """Test that comparison report includes WinRate/MaxDD sections and delta columns."""

    def _build_two_strategy_report(self):
        """Helper: build a 2-strategy, 3-window comparison."""
        strat_perfs = {
            "StratA": [
                _make_window_perf(0, "2024-01-01", "2024-02-01",
                                  total_trades=10, win_rate=0.7, total_profit_pct=8.0,
                                  sharpe_ratio=2.0, max_drawdown=0.03, carry_over_trades=0),
                _make_window_perf(1, "2024-02-01", "2024-03-01",
                                  total_trades=8, win_rate=0.5, total_profit_pct=-2.0,
                                  sharpe_ratio=-0.5, max_drawdown=0.10, carry_over_trades=2),
                _make_window_perf(2, "2024-03-01", "2024-04-01",
                                  total_trades=12, win_rate=0.6, total_profit_pct=5.0,
                                  sharpe_ratio=1.2, max_drawdown=0.05, carry_over_trades=1),
            ],
            "StratB": [
                _make_window_perf(0, "2024-01-01", "2024-02-01",
                                  total_trades=15, win_rate=0.6, total_profit_pct=3.0,
                                  sharpe_ratio=1.0, max_drawdown=0.06, carry_over_trades=0),
                _make_window_perf(1, "2024-02-01", "2024-03-01",
                                  total_trades=12, win_rate=0.6, total_profit_pct=2.0,
                                  sharpe_ratio=0.8, max_drawdown=0.04, carry_over_trades=3),
                _make_window_perf(2, "2024-03-01", "2024-04-01",
                                  total_trades=10, win_rate=0.5, total_profit_pct=4.0,
                                  sharpe_ratio=1.5, max_drawdown=0.08, carry_over_trades=0),
            ],
        }
        return build_comparison_report(strat_perfs)

    def test_report_has_drawdown_best(self):
        """WindowComparison should have best_strategy_drawdown."""
        report = self._build_two_strategy_report()
        assert len(report.windows) == 3
        # Window 0: StratA has 0.03 DD < StratB 0.06 → StratA best
        assert report.windows[0].best_strategy_drawdown == "StratA"
        # Window 1: StratB has 0.04 DD < StratA 0.10 → StratB best
        assert report.windows[1].best_strategy_drawdown == "StratB"

    def test_format_table_has_four_metric_sections(self):
        """Formatted table should have Profit%, WinRate%, Sharpe, and MaxDD sections."""
        report = self._build_two_strategy_report()
        table = format_comparison_table(report)

        assert "--- Profit %" in table
        assert "--- Win Rate %" in table
        assert "--- Sharpe Ratio" in table
        assert "--- Max Drawdown %" in table

    def test_format_table_has_delta_columns(self):
        """Each metric section should show Δ values."""
        report = self._build_two_strategy_report()
        table = format_comparison_table(report)

        # Should contain delta markers
        delta_count = table.count("Δ")
        # 3 windows × 4 metrics = 12 delta values
        assert delta_count == 12, f"Expected 12 Δ markers, got {delta_count}"

    def test_format_table_has_star_markers(self):
        """Best strategy per window should be marked with ★."""
        report = self._build_two_strategy_report()
        table = format_comparison_table(report)
        star_count = table.count("★")
        # 3 windows × 4 metrics = 12 stars
        assert star_count == 12

    def test_overall_ranking_sorted_by_profit(self):
        """Rankings should be sorted by total profit descending."""
        report = self._build_two_strategy_report()
        assert len(report.overall_ranking) == 2
        # StratA total: 8+(-2)+5 = 11, StratB total: 3+2+4 = 9
        assert report.overall_ranking[0].strategy == "StratA"
        assert report.overall_ranking[0].total_profit_pct == pytest.approx(11.0)
        assert report.overall_ranking[1].strategy == "StratB"
        assert report.overall_ranking[1].total_profit_pct == pytest.approx(9.0)

    def test_windows_won_counts(self):
        """Should correctly count how many windows each strategy won."""
        report = self._build_two_strategy_report()
        ranking_map = {r.strategy: r for r in report.overall_ranking}

        # Profit wins: Win0=StratA(8>3), Win1=StratB(2>-2), Win2=StratA(5>4)
        assert ranking_map["StratA"].windows_won_profit == 2
        assert ranking_map["StratB"].windows_won_profit == 1

    def test_consistency_score(self):
        """Consistency score should be std of per-window profits."""
        report = self._build_two_strategy_report()
        ranking_map = {r.strategy: r for r in report.overall_ranking}

        # StratA profits: [8, -2, 5] → std
        expected_std = float(np.std([8.0, -2.0, 5.0]))
        assert ranking_map["StratA"].consistency_score == pytest.approx(expected_std, abs=0.01)


class TestComparisonReportSingleStrategy:
    """Edge case: single strategy comparison (should still work)."""

    def test_single_strategy_no_delta(self):
        strat_perfs = {
            "OnlyStrat": [
                _make_window_perf(0, "2024-01-01", "2024-02-01", total_trades=5, total_profit_pct=3.0),
            ],
        }
        report = build_comparison_report(strat_perfs)
        table = format_comparison_table(report)
        # Single strategy → no delta column (only shows when >= 2 strategies)
        assert "Δ" not in table
        assert "OnlyStrat" in table


class TestComparisonReportThreeStrategies:
    """Test with 3 strategies — the delta should be max-min across all 3."""

    def test_three_strategy_delta(self):
        strat_perfs = {
            "A": [_make_window_perf(0, "2024-01-01", "2024-02-01", total_profit_pct=10.0)],
            "B": [_make_window_perf(0, "2024-01-01", "2024-02-01", total_profit_pct=3.0)],
            "C": [_make_window_perf(0, "2024-01-01", "2024-02-01", total_profit_pct=-2.0)],
        }
        report = build_comparison_report(strat_perfs)
        table = format_comparison_table(report)

        # Profit delta = 10 - (-2) = 12
        assert "Δ+12.00" in table


# ---------------------------------------------------------------------------
# Tests: End-to-end from raw trades to comparison table
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Simulate the full pipeline: trades → window_performances → comparison → table."""

    def test_full_pipeline_two_strategies(self):
        """Build window performances from trades, then generate comparison."""
        # Strategy A trades
        trades_a = _make_trades_df([
            {"open_date": "2024-01-05", "close_date": "2024-01-20", "profit_ratio": 0.05},
            {"open_date": "2024-01-10", "close_date": "2024-01-25", "profit_ratio": 0.02},
            {"open_date": "2024-01-20", "close_date": "2024-02-10", "profit_ratio": -0.03},  # carry-over to window 1
            {"open_date": "2024-02-01", "close_date": "2024-02-15", "profit_ratio": 0.04},
        ])

        # Strategy B trades
        trades_b = _make_trades_df([
            {"open_date": "2024-01-03", "close_date": "2024-01-15", "profit_ratio": -0.01},
            {"open_date": "2024-01-08", "close_date": "2024-01-28", "profit_ratio": 0.03},
            {"open_date": "2024-02-02", "close_date": "2024-02-20", "profit_ratio": 0.06},
        ])

        boundaries = [
            (0, datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc)),
            (1, datetime(2024, 2, 1, tzinfo=timezone.utc), datetime(2024, 3, 1, tzinfo=timezone.utc)),
        ]

        perfs_a = compute_all_window_performances(trades_a, boundaries)
        perfs_b = compute_all_window_performances(trades_b, boundaries)

        # Verify per-window (not cumulative)
        assert perfs_a[0].total_trades == 2  # 2 trades close in Jan
        assert perfs_a[1].total_trades == 2  # 2 trades close in Feb
        assert perfs_a[1].carry_over_trades == 1  # 1 trade opened in Jan

        assert perfs_b[0].total_trades == 2
        assert perfs_b[1].total_trades == 1

        # Build comparison
        report = build_comparison_report({"StratA": perfs_a, "StratB": perfs_b})
        table = format_comparison_table(report)

        # Verify table structure
        assert "Multi-Strategy Window Comparison" in table
        assert "StratA" in table
        assert "StratB" in table
        assert "Profit %" in table
        assert "Win Rate %" in table
        assert "Sharpe Ratio" in table
        assert "Max Drawdown %" in table
        assert "Overall Ranking" in table
        assert "★" in table
        assert "Δ" in table
