# 窗口级增强功能 — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 三项增强：(1) 窗口边界持仓延续改进——扩展数据加载覆盖未平仓位，消除±1交易偏差；(2) 窗口级独立绩效分析（Sharpe/Sortino/胜率/最大回撤等）；(3) 多策略窗口横向对比报告。

**Architecture:**
- 持仓延续：在每个窗口的数据加载阶段，检查当前 open trades，将数据加载的 start 前推到覆盖所有 open trade 的 open_date（确保指标连续），同时记录 carry-over 信息
- 窗口级绩效：新增 `window_metrics.py` 模块，基于每个窗口时间范围内的已关闭交易计算独立指标
- 多策略对比：新增 `comparison_report.py`，在所有策略的窗口级指标上做横向对比

**Tech Stack:** freqtrade Backtesting internals, pandas, numpy, json

---

## 新增/修改文件

```
user_custom/rolling_backtest/
├── src/
│   ├── rolling_runner.py           # MODIFY: carry-over tracking + window metrics integration
│   ├── window_metrics.py           # CREATE: per-window performance calculation
│   ├── comparison_report.py        # CREATE: multi-strategy window comparison
│   └── windowing.py                # MINOR: no changes needed
├── tests/
│   ├── test_window_metrics.py      # CREATE: unit tests
│   └── test_comparison_report.py   # CREATE: unit tests
└── scripts/
    └── run_rolling_backtest.py     # MODIFY: new CLI flags
```

---

## Task 1: 窗口级绩效指标模块

**Objective:** 创建 `window_metrics.py`，输入一个窗口内的交易列表，输出 Sharpe/Sortino/胜率/最大回撤等指标

**Files:**
- Create: `user_custom/rolling_backtest/src/window_metrics.py`
- Create: `user_custom/rolling_backtest/tests/test_window_metrics.py`

### Step 1: 写失败测试

```python
# tests/test_window_metrics.py
import pytest
import pandas as pd
import numpy as np
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
        window_end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        starting_balance=1000.0,
    )
    assert isinstance(perf, WindowPerformance)
    assert perf.window_index == 0
    assert perf.total_trades == 20
    assert 0 <= perf.win_rate <= 1
    assert perf.sharpe_ratio is not None  # may be negative
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
    # All closed trades should be assigned to the window where they closed
    assert total <= 50  # some may fall outside window boundaries


def test_window_performance_profit_factor():
    trades = _make_trades(30, win_rate=0.8)
    perf = compute_window_performance(
        trades=trades, window_index=0,
        window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        starting_balance=1000.0,
    )
    assert perf.profit_factor > 0  # with 80% win rate, should be positive


def test_window_performance_carry_over_trades():
    """Trades that started before the window should be flagged as carry-over."""
    trades = _make_trades(10)
    # Shift open_date of first 3 trades to before window start
    trades.loc[:2, "open_date"] = datetime(2023, 12, 28, tzinfo=timezone.utc)
    perf = compute_window_performance(
        trades=trades, window_index=0,
        window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        starting_balance=1000.0,
    )
    assert perf.carry_over_trades == 3
```

### Step 2: 运行测试确认失败

```bash
cd /root/workspace/freqtrade && source .venv/bin/activate
PYTHONPATH=/root/workspace/freqtrade python -m pytest user_custom/rolling_backtest/tests/test_window_metrics.py -v
# Expected: FAIL — module not found
```

### Step 3: 实现模块

```python
# src/window_metrics.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
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


def _max_drawdown_from_returns(returns: np.ndarray, starting_balance: float) -> tuple[float, float]:
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

    Trades are assigned to the window where they CLOSE (close_date within [window_start, window_end)).
    """
    perf = WindowPerformance(
        window_index=window_index,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
    )

    if trades is None or trades.empty:
        return perf

    # Filter trades that closed in this window
    if "close_date" in trades.columns:
        close_dates = pd.to_datetime(trades["close_date"], utc=True)
        ws = pd.Timestamp(window_start, tz="UTC")
        we = pd.Timestamp(window_end, tz="UTC")
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
    profit_abs = window_trades["profit_abs"].values.astype(float) if "profit_abs" in window_trades.columns else profit_ratios * starting_balance

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
    perf.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

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
```

### Step 4: 运行测试确认通过

```bash
PYTHONPATH=/root/workspace/freqtrade python -m pytest user_custom/rolling_backtest/tests/test_window_metrics.py -v
# Expected: 5 passed
```

### Step 5: Commit

```bash
git add user_custom/rolling_backtest/src/window_metrics.py user_custom/rolling_backtest/tests/test_window_metrics.py
git commit -m "feat(rolling): add per-window performance metrics module (Sharpe/Sortino/drawdown/win rate)"
```

---

## Task 2: 窗口级指标集成到 rolling_runner

**Objective:** 在 `RollingBacktestRunner` 中集成窗口级指标，每个窗口结束后计算并记录绩效

**Files:**
- Modify: `user_custom/rolling_backtest/src/rolling_runner.py`

### Step 1: 修改 `_run_strategy()` 方法

在策略跑完所有窗口后，基于 `window_stats` 中的窗口边界和全部交易计算每个窗口的独立绩效。

关键改动点：

1. 在 `_run_strategy()` 末尾、构建 `strat_result` 之前，调用 `compute_all_window_performances()`
2. 将 `window_performances` 加入 `strat_result`
3. 新增 `_print_window_metrics_table()` 打印窗口级绩效表

```python
# In rolling_runner.py — add import at top
from .window_metrics import compute_all_window_performances, WindowPerformance

# In _run_strategy(), after trades_df is built (line ~376), add:
        # Compute per-window performance metrics
        window_boundaries = [
            (ws.index, datetime.fromisoformat(ws.start), datetime.fromisoformat(ws.end))
            for ws in self.window_stats
            if ws.status == "ok"
        ]
        window_performances = compute_all_window_performances(
            all_trades=trades_df,
            window_boundaries=window_boundaries,
            starting_balance=strat.config.get("dry_run_wallet", 1000.0),
        )

# Add window_performances to strat_result:
        strat_result["window_performances"] = [wp.to_dict() for wp in window_performances]

# Add stability summary
        if window_performances:
            sharpes = [wp.sharpe_ratio for wp in window_performances if wp.total_trades > 0]
            win_rates = [wp.win_rate for wp in window_performances if wp.total_trades > 0]
            drawdowns = [wp.max_drawdown for wp in window_performances if wp.total_trades > 0]
            strat_result["summary"]["window_stability"] = {
                "sharpe_mean": float(np.mean(sharpes)) if sharpes else 0.0,
                "sharpe_std": float(np.std(sharpes)) if len(sharpes) > 1 else 0.0,
                "win_rate_mean": float(np.mean(win_rates)) if win_rates else 0.0,
                "win_rate_std": float(np.std(win_rates)) if len(win_rates) > 1 else 0.0,
                "max_drawdown_mean": float(np.mean(drawdowns)) if drawdowns else 0.0,
                "max_drawdown_worst": float(max(drawdowns)) if drawdowns else 0.0,
            }
```

### Step 2: 新增 `_print_window_metrics_table()` 方法

```python
    def _print_window_metrics_table(
        self, strategy_name: str, performances: list[WindowPerformance]
    ) -> None:
        """Print detailed per-window metrics table."""
        if not performances:
            return

        header = (
            f"\n{'=' * 110}\n"
            f" Window Performance — {strategy_name}\n"
            f"{'=' * 110}\n"
            f" {'Win':>3} | {'Period':^23} | {'Trades':>6} | {'Win%':>5} | "
            f"{'Profit%':>8} | {'Sharpe':>7} | {'Sortino':>8} | "
            f"{'MaxDD%':>7} | {'PF':>5} | {'Carry':>5}\n"
            f"{'-' * 110}"
        )
        lines = [header]
        for wp in performances:
            start_short = wp.window_start[:10]
            end_short = wp.window_end[:10]
            period = f"{start_short} → {end_short}"
            sharpe_s = f"{wp.sharpe_ratio:>7.2f}" if wp.total_trades > 0 else "    N/A"
            sortino_s = f"{wp.sortino_ratio:>8.2f}" if wp.total_trades > 0 else "     N/A"
            pf_s = f"{wp.profit_factor:>5.2f}" if wp.profit_factor < 999 else "  INF"
            lines.append(
                f" {wp.window_index:>3} | {period:^23} | {wp.total_trades:>6} | "
                f"{wp.win_rate * 100:>5.1f} | {wp.total_profit_pct:>+8.2f} | "
                f"{sharpe_s} | {sortino_s} | "
                f"{wp.max_drawdown * 100:>7.2f} | {pf_s} | {wp.carry_over_trades:>5}"
            )
        lines.append(f"{'=' * 110}")
        print("\n".join(lines))
```

### Step 3: 在 `_run_strategy()` 末尾调用打印

```python
        # After _print_window_summary_table
        self._print_window_metrics_table(strategy_name, window_performances)
```

### Step 4: 运行现有测试确认不 break

```bash
PYTHONPATH=/root/workspace/freqtrade python -m pytest user_custom/rolling_backtest/tests/ -v
# Expected: all existing tests pass
```

### Step 5: Commit

```bash
git add user_custom/rolling_backtest/src/rolling_runner.py
git commit -m "feat(rolling): integrate per-window metrics into runner output and console"
```

---

## Task 3: 窗口边界持仓延续改进

**Objective:** 扩展窗口数据加载范围，覆盖所有未平仓位的 open_date，确保指标在窗口边界处连续

**Files:**
- Modify: `user_custom/rolling_backtest/src/rolling_runner.py`

### 设计

当前行为：每个窗口加载 [window_start - startup_candles, window_end]。
问题：如果有 open trade 在 window_start 之前开仓，那个位置的指标值在新窗口重算后可能不同。
改进：检查 `LocalTrade.bt_trades_open`，找到最早的 `open_date`，将数据加载的 timerange start 前推到覆盖那个日期。

### Step 1: 修改 `_run_single_window()`

在调用 `_load_window_data()` 之前，检查 open trades 并扩展 timerange：

```python
    def _get_extended_timerange_for_open_trades(self, window: Window) -> TimeRange:
        """Extend window start to cover all open trades' entry points.

        This ensures indicators are calculated continuously for positions
        that span window boundaries, eliminating ±1 trade deviations.
        """
        open_trades = LocalTrade.bt_trades_open
        if not open_trades:
            return window.timerange

        # Find earliest open_date among current open trades
        earliest_open = min(t.open_date for t in open_trades)

        window_start_dt = window.start
        if earliest_open < window_start_dt:
            # Extend start to cover the earliest open trade
            extended_start_ts = int(earliest_open.timestamp())
            logger.info(
                "[window %s] extending data load start from %s to %s to cover %d open trades",
                window.index,
                window_start_dt.isoformat(),
                earliest_open.isoformat(),
                len(open_trades),
            )
            return TimeRange(
                "date", "date",
                extended_start_ts,
                int(window.end.timestamp()),
            )
        return window.timerange

    # In _run_single_window, replace:
    #   raw_data = self._load_window_data(window.timerange)
    # with:
    #   extended_tr = self._get_extended_timerange_for_open_trades(window)
    #   raw_data = self._load_window_data(extended_tr)
```

### Step 2: 更新 `WindowRunStat` 加 carry_over 字段

```python
@dataclass
class WindowRunStat:
    # ... existing fields ...
    carry_over_trades: int = 0  # ADD: open trades carried from previous window
```

在 `_run_single_window()` 记录 carry-over 数量：

```python
        carry_over = len(LocalTrade.bt_trades_open)
        # ... (at the start of _run_single_window, before processing)
        # Record in WindowRunStat:
        self.window_stats[-1].carry_over_trades = carry_over  # (or set during construction)
```

### Step 3: 运行测试

```bash
PYTHONPATH=/root/workspace/freqtrade python -m pytest user_custom/rolling_backtest/tests/ -v
```

### Step 4: Commit

```bash
git add user_custom/rolling_backtest/src/rolling_runner.py
git commit -m "feat(rolling): extend window data load to cover open trade entry points for indicator continuity"
```

---

## Task 4: 多策略窗口横向对比报告

**Objective:** 创建 `comparison_report.py`，对多个策略的窗口级指标做横向对比

**Files:**
- Create: `user_custom/rolling_backtest/src/comparison_report.py`
- Create: `user_custom/rolling_backtest/tests/test_comparison_report.py`

### Step 1: 写失败测试

```python
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
```

### Step 2: 运行测试确认失败

```bash
PYTHONPATH=/root/workspace/freqtrade python -m pytest user_custom/rolling_backtest/tests/test_comparison_report.py -v
# Expected: FAIL — module not found
```

### Step 3: 实现模块

```python
# src/comparison_report.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

from .window_metrics import WindowPerformance

logger = logging.getLogger(__name__)


@dataclass
class WindowComparison:
    """Cross-strategy comparison for a single window."""
    window_index: int
    window_start: str
    window_end: str
    strategy_metrics: dict[str, dict[str, Any]]  # strategy_name -> metrics dict
    best_strategy_profit: str = ""
    best_strategy_sharpe: str = ""
    best_strategy_win_rate: str = ""
    profit_spread: float = 0.0  # best - worst profit
    sharpe_spread: float = 0.0


@dataclass
class StrategyRanking:
    """Overall ranking entry for a strategy."""
    strategy: str
    total_profit_pct: float
    avg_sharpe: float
    avg_win_rate: float
    avg_max_drawdown: float
    windows_won_profit: int  # how many windows this strategy had best profit
    windows_won_sharpe: int
    consistency_score: float = 0.0  # lower std of profit across windows = more consistent


@dataclass
class StrategyWindowComparison:
    """Full comparison report across strategies and windows."""
    strategies: list[str]
    windows: list[WindowComparison]
    overall_ranking: list[StrategyRanking]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": self.strategies,
            "windows": [asdict(w) for w in self.windows],
            "overall_ranking": [asdict(r) for r in self.overall_ranking],
        }


def build_comparison_report(
    strategy_performances: dict[str, list[WindowPerformance]],
) -> StrategyWindowComparison:
    """Build a cross-strategy, per-window comparison report.

    Args:
        strategy_performances: {strategy_name: [WindowPerformance, ...]}

    Returns:
        StrategyWindowComparison with per-window and overall rankings.
    """
    strategies = list(strategy_performances.keys())
    if not strategies:
        return StrategyWindowComparison(strategies=[], windows=[], overall_ranking=[])

    # Determine number of windows from first strategy
    n_windows = len(next(iter(strategy_performances.values())))

    windows: list[WindowComparison] = []
    # Track wins per strategy
    wins_profit: dict[str, int] = {s: 0 for s in strategies}
    wins_sharpe: dict[str, int] = {s: 0 for s in strategies}

    for wi in range(n_windows):
        # Gather metrics for this window across strategies
        strategy_metrics: dict[str, dict[str, Any]] = {}
        for sname, perfs in strategy_performances.items():
            if wi < len(perfs):
                wp = perfs[wi]
                strategy_metrics[sname] = wp.to_dict()

        if not strategy_metrics:
            continue

        # Find best by different metrics
        best_profit_name = max(strategy_metrics, key=lambda s: strategy_metrics[s].get("total_profit_pct", -999))
        best_sharpe_name = max(strategy_metrics, key=lambda s: strategy_metrics[s].get("sharpe_ratio", -999))
        best_winrate_name = max(strategy_metrics, key=lambda s: strategy_metrics[s].get("win_rate", -999))

        wins_profit[best_profit_name] = wins_profit.get(best_profit_name, 0) + 1
        wins_sharpe[best_sharpe_name] = wins_sharpe.get(best_sharpe_name, 0) + 1

        profits = [strategy_metrics[s].get("total_profit_pct", 0) for s in strategy_metrics]
        sharpes = [strategy_metrics[s].get("sharpe_ratio", 0) for s in strategy_metrics]

        # Get window dates from first available strategy
        first_wp = next(iter(strategy_metrics.values()))
        wc = WindowComparison(
            window_index=wi,
            window_start=first_wp.get("window_start", ""),
            window_end=first_wp.get("window_end", ""),
            strategy_metrics=strategy_metrics,
            best_strategy_profit=best_profit_name,
            best_strategy_sharpe=best_sharpe_name,
            best_strategy_win_rate=best_winrate_name,
            profit_spread=max(profits) - min(profits) if profits else 0.0,
            sharpe_spread=max(sharpes) - min(sharpes) if sharpes else 0.0,
        )
        windows.append(wc)

    # Build overall ranking
    import numpy as np
    rankings: list[StrategyRanking] = []
    for sname, perfs in strategy_performances.items():
        profits = [wp.total_profit_pct for wp in perfs if wp.total_trades > 0]
        sharpes = [wp.sharpe_ratio for wp in perfs if wp.total_trades > 0]
        win_rates = [wp.win_rate for wp in perfs if wp.total_trades > 0]
        drawdowns = [wp.max_drawdown for wp in perfs if wp.total_trades > 0]

        rankings.append(StrategyRanking(
            strategy=sname,
            total_profit_pct=sum(wp.total_profit_pct for wp in perfs),
            avg_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
            avg_win_rate=float(np.mean(win_rates)) if win_rates else 0.0,
            avg_max_drawdown=float(np.mean(drawdowns)) if drawdowns else 0.0,
            windows_won_profit=wins_profit.get(sname, 0),
            windows_won_sharpe=wins_sharpe.get(sname, 0),
            consistency_score=float(np.std(profits)) if len(profits) > 1 else 0.0,
        ))

    # Sort by total profit descending
    rankings.sort(key=lambda r: r.total_profit_pct, reverse=True)

    return StrategyWindowComparison(
        strategies=strategies,
        windows=windows,
        overall_ranking=rankings,
    )


def format_comparison_table(report: StrategyWindowComparison) -> str:
    """Format comparison report as human-readable text table."""
    if not report.windows:
        return "No windows to compare."

    strategies = report.strategies
    col_width = max(len(s) for s in strategies) + 2

    lines = []
    lines.append(f"\n{'=' * (40 + col_width * len(strategies))}")
    lines.append(" Multi-Strategy Window Comparison")
    lines.append(f"{'=' * (40 + col_width * len(strategies))}")

    # Header row
    hdr = f" {'Window':>6} | {'Period':^23} |"
    for s in strategies:
        hdr += f" {s:^{col_width}} |"
    hdr += " Best"
    lines.append(hdr)
    lines.append(f"{'-' * (40 + col_width * len(strategies) + 6)}")

    # Profit comparison per window
    lines.append(" --- Profit % ---")
    for wc in report.windows:
        start_short = wc.window_start[:10]
        end_short = wc.window_end[:10]
        period = f"{start_short}→{end_short}"
        row = f" {f'Win {wc.window_index}':>6} | {period:^23} |"
        for s in strategies:
            val = wc.strategy_metrics.get(s, {}).get("total_profit_pct", 0)
            marker = " ★" if s == wc.best_strategy_profit else "  "
            row += f" {val:>+{col_width - 4}.2f}{marker} |"
        row += f" {wc.best_strategy_profit}"
        lines.append(row)

    # Sharpe comparison per window
    lines.append(f"\n --- Sharpe Ratio ---")
    for wc in report.windows:
        start_short = wc.window_start[:10]
        end_short = wc.window_end[:10]
        period = f"{start_short}→{end_short}"
        row = f" {f'Win {wc.window_index}':>6} | {period:^23} |"
        for s in strategies:
            val = wc.strategy_metrics.get(s, {}).get("sharpe_ratio", 0)
            marker = " ★" if s == wc.best_strategy_sharpe else "  "
            row += f" {val:>+{col_width - 4}.2f}{marker} |"
        row += f" {wc.best_strategy_sharpe}"
        lines.append(row)

    # Overall ranking
    lines.append(f"\n{'=' * (40 + col_width * len(strategies) + 6)}")
    lines.append(" Overall Ranking")
    lines.append(f"{'-' * 80}")
    lines.append(f" {'#':>2} | {'Strategy':<{col_width}} | {'TotalP%':>8} | {'AvgSharpe':>9} | "
                 f"{'AvgWin%':>7} | {'AvgDD%':>7} | {'WinsP':>5} | {'WinsS':>5} | {'Consist':>7}")
    lines.append(f"{'-' * 80}")
    for i, r in enumerate(report.overall_ranking):
        lines.append(
            f" {i+1:>2} | {r.strategy:<{col_width}} | {r.total_profit_pct:>+8.2f} | "
            f"{r.avg_sharpe:>9.2f} | {r.avg_win_rate * 100:>7.1f} | "
            f"{r.avg_max_drawdown * 100:>7.2f} | {r.windows_won_profit:>5} | "
            f"{r.windows_won_sharpe:>5} | {r.consistency_score:>7.2f}"
        )
    lines.append(f"{'=' * 80}")

    return "\n".join(lines)
```

### Step 4: 运行测试确认通过

```bash
PYTHONPATH=/root/workspace/freqtrade python -m pytest user_custom/rolling_backtest/tests/test_comparison_report.py -v
# Expected: 4 passed
```

### Step 5: Commit

```bash
git add user_custom/rolling_backtest/src/comparison_report.py user_custom/rolling_backtest/tests/test_comparison_report.py
git commit -m "feat(rolling): add multi-strategy window comparison report module"
```

---

## Task 5: 集成多策略对比到 runner 和 CLI

**Objective:** 在多策略模式下自动生成对比报告，新增 `--window-metrics` CLI flag

**Files:**
- Modify: `user_custom/rolling_backtest/src/rolling_runner.py`
- Modify: `user_custom/rolling_backtest/scripts/run_rolling_backtest.py`

### Step 1: 修改 `run()` 方法

```python
# In rolling_runner.py run() method, after all strategies finish:

from .comparison_report import build_comparison_report, format_comparison_table

        # Multi-strategy comparison report
        if len(all_strategy_results) > 1:
            # Gather per-strategy WindowPerformance objects
            strat_perfs = {}
            for sr in all_strategy_results:
                sname = sr["strategy"]
                if "window_performances" in sr:
                    from .window_metrics import WindowPerformance
                    perfs = [WindowPerformance(**wp) for wp in sr["window_performances"]]
                    strat_perfs[sname] = perfs

            if strat_perfs:
                comparison = build_comparison_report(strat_perfs)
                table = format_comparison_table(comparison)
                print(table)
                result["comparison"] = comparison.to_dict()
```

### Step 2: 新增 CLI flag

```python
# In run_rolling_backtest.py parse_args():
    p.add_argument(
        "--window-metrics",
        action="store_true",
        help="Enable per-window performance metrics (Sharpe/Sortino/drawdown/win rate)",
    )
```

将 `window_metrics` flag 通过 config 传入 runner：

```python
# In main():
    config["window_metrics"] = ns.window_metrics
```

在 runner 中根据 flag 决定是否计算指标（但默认 always on 更简单——flag 控制是否打印详细表格）。

### Step 3: 运行完整测试

```bash
PYTHONPATH=/root/workspace/freqtrade python -m pytest user_custom/rolling_backtest/tests/ -v
```

### Step 4: Commit

```bash
git add user_custom/rolling_backtest/src/rolling_runner.py user_custom/rolling_backtest/scripts/run_rolling_backtest.py
git commit -m "feat(rolling): integrate comparison report for multi-strategy mode + --window-metrics CLI flag"
```

---

## Task 6: Smoke Test

**Objective:** 端到端验证三个功能

### 测试场景

```bash
cd /root/workspace/freqtrade && source .venv/bin/activate

# 1. 单策略 + 窗口级绩效
python user_custom/rolling_backtest/scripts/run_rolling_backtest.py \
  --config user_custom/rolling_backtest/config/rolling_backtest_config.example.json \
  --strategy RollingSmokeStrategy \
  --timerange 20240101-20240401 \
  --window-days 30 \
  --window-metrics

# 验证: 输出中包含 Window Performance 表格（Sharpe/Sortino/MaxDD 列）
# 验证: JSON 输出中有 window_performances 数组

# 2. 多策略 A/B 对比（需要两个策略）
python user_custom/rolling_backtest/scripts/run_rolling_backtest.py \
  --config user_custom/rolling_backtest/config/rolling_backtest_config.example.json \
  --strategy-list RollingSmokeStrategy WFOSmokeStrategy \
  --timerange 20240101-20240401 \
  --window-days 30 \
  --window-metrics

# 验证: 输出中包含 Multi-Strategy Window Comparison 表格
# 验证: JSON 输出中有 comparison 字段

# 3. 持仓延续验证
# 查看日志中 "extending data load start" 消息（仅当有 carry-over trades 时出现）
```

### Step 1: 运行并验证输出

验证要点：
- `window_performances` 数组长度 == ok_windows 数量
- 每个 WindowPerformance 有 sharpe_ratio, sortino_ratio, max_drawdown, win_rate
- carry_over_trades >= 0
- 多策略模式下 comparison 报告包含 strategies, windows, overall_ranking
- 控制台打印两张表（Window Performance + Multi-Strategy Comparison）

### Step 2: Commit

```bash
git add -A user_custom/rolling_backtest/
git commit -m "feat(rolling): smoke test passed — window metrics + carry-over + comparison report"
```

---

## 验证 Checklist

- [ ] `test_window_metrics.py` — 5 tests pass
- [ ] `test_comparison_report.py` — 4 tests pass
- [ ] 现有测试不 break (`test_parity_tools.py`, `test_dynamic_pairlist_and_signals.py`, `test_wfo_windowing.py`)
- [ ] 单策略 smoke: window_performances 出现在 JSON + 控制台表格
- [ ] 多策略 smoke: comparison 报告输出
- [ ] carry-over trades 字段正确记录
- [ ] 日志中可见 "extending data load" 消息（当有 open trades 跨窗口时）
