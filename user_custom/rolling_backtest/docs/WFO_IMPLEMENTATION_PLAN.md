# Walk-Forward Optimization (WFO) — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 在现有滚动回测框架上构建 Walk-Forward Optimization，实现"训练窗口参数优化 → 测试窗口 out-of-sample 验证"的自动循环，检测参数过拟合并输出每折的 IS/OOS 对比报告。

**Architecture:**
- 复用现有 `RollingBacktestRunner` 的窗口化回测能力
- 复用 freqtrade 内置 `Backtesting.backtest()` 方法（hyperopt 用的单次回测接口）+ optuna 优化器
- 新增 `WalkForwardEngine` 编排 IS/OOS 窗口对、参数优化循环、结果聚合
- 不修改 freqtrade 核心代码，全部在 `user_custom/rolling_backtest/` 下

**Tech Stack:** freqtrade Backtesting internals, optuna (freqtrade 已内置), pandas, json

---

## 核心概念

### Walk-Forward 窗口划分

```
全量数据: |==============================================================|
                                                                          
Fold 1:   |---- IS (训练) ----|-- OOS (验证) --|
Fold 2:        |---- IS (训练) ----|-- OOS (验证) --|
Fold 3:             |---- IS (训练) ----|-- OOS (验证) --|
...

IS = In-Sample: 用于参数优化（hyperopt）
OOS = Out-of-Sample: 用优化后的参数做回测验证
```

### 关键参数
- `is_days`: 训练窗口天数（如 180）
- `oos_days`: 验证窗口天数（如 30-60）
- `step_days`: 窗口滑动步长（默认 = oos_days，无重叠）
- `n_epochs`: 每折的 optuna 优化轮数（如 100）
- `loss_function`: hyperopt loss 名称（如 SharpeHyperOptLoss）

### 输出
- 每折: IS 最优参数 + IS 绩效 + OOS 绩效
- 汇总: OOS 拼接的"伪实盘"权益曲线 + IS/OOS 效率比 + 过拟合检测

---

## 新增文件

```
user_custom/rolling_backtest/
├── src/
│   ├── wfo_engine.py          # Walk-Forward 主引擎
│   ├── wfo_fold.py            # 单折 IS/OOS 执行逻辑
│   ├── wfo_optimizer.py       # 参数优化（复用 freqtrade hyperopt 内部接口）
│   ├── wfo_report.py          # WFO 结果聚合与报告生成
│   └── wfo_windowing.py       # WFO 窗口划分（IS/OOS 对生成）
├── scripts/
│   └── run_wfo.py             # CLI 入口
└── tests/
    ├── test_wfo_windowing.py  # 窗口划分单元测试
    └── test_wfo_smoke.py      # 端到端 smoke test
```

---

## Task 1: WFO 窗口划分模块

**Objective:** 实现 IS/OOS 窗口对的生成逻辑

**Files:**
- Create: `user_custom/rolling_backtest/src/wfo_windowing.py`
- Create: `user_custom/rolling_backtest/tests/test_wfo_windowing.py`

### Step 1: 写失败测试

```python
# tests/test_wfo_windowing.py
import pytest
from datetime import datetime, UTC
from freqtrade.configuration import TimeRange

from user_custom.rolling_backtest.src.wfo_windowing import WFOFold, build_wfo_folds


def test_build_wfo_folds_basic():
    """180d IS + 30d OOS, step=30d, over 1 year should give ~7 folds."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()))
    folds = build_wfo_folds(
        timerange=tr,
        timeframe="5m",
        is_days=180,
        oos_days=30,
        step_days=30,
    )
    assert len(folds) >= 5
    # Each fold has IS + OOS
    for f in folds:
        assert f.is_start < f.is_end
        assert f.oos_start == f.is_end  # OOS immediately follows IS
        assert f.oos_end > f.oos_start
        assert (f.oos_end - f.oos_start).days <= 31  # ~30 days


def test_build_wfo_folds_no_overlap():
    """With step=oos_days, OOS windows should tile without gaps."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()))
    folds = build_wfo_folds(
        timerange=tr, timeframe="5m",
        is_days=90, oos_days=30, step_days=30,
    )
    # OOS end of fold i == OOS start of fold i+1
    for i in range(len(folds) - 1):
        assert folds[i].oos_end == folds[i + 1].oos_start


def test_build_wfo_folds_too_short():
    """Timerange shorter than IS+OOS should raise."""
    tr = TimeRange("date", "date",
                    int(datetime(2024, 1, 1, tzinfo=UTC).timestamp()),
                    int(datetime(2024, 3, 1, tzinfo=UTC).timestamp()))
    with pytest.raises(ValueError, match="too short"):
        build_wfo_folds(tr, timeframe="5m", is_days=180, oos_days=30)
```

### Step 2: 运行测试确认失败

```bash
cd /root/workspace/freqtrade && source .venv/bin/activate
python -m pytest user_custom/rolling_backtest/tests/test_wfo_windowing.py -v
# Expected: FAIL — module not found
```

### Step 3: 实现模块

```python
# src/wfo_windowing.py
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
```

### Step 4: 运行测试确认通过

```bash
python -m pytest user_custom/rolling_backtest/tests/test_wfo_windowing.py -v
# Expected: 3 passed
```

### Step 5: Commit

```bash
git add user_custom/rolling_backtest/src/wfo_windowing.py user_custom/rolling_backtest/tests/test_wfo_windowing.py
git commit -m "feat(wfo): add WFO fold windowing with IS/OOS pair generation"
```

---

## Task 2: WFO 优化器 — 单折参数优化

**Objective:** 封装 freqtrade 的 hyperopt 内部接口，对单个 IS 窗口执行参数优化

**Files:**
- Create: `user_custom/rolling_backtest/src/wfo_optimizer.py`

### Step 1: 实现

核心思路：复用 `Backtesting.backtest()` 方法（hyperopt 内部也是调用它），配合 optuna 做参数搜索。

```python
# src/wfo_optimizer.py
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import get_tick_size_over_time
from freqtrade.data.converter import trim_dataframes
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.hyperopt_tools import HyperoptTools
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver
from freqtrade.optimize.optimize_reports import generate_strategy_stats

import optuna

from .wfo_windowing import WFOFold

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimizing parameters on an IS window."""
    fold_index: int
    best_params: dict[str, Any]
    best_loss: float
    n_trials: int
    is_stats: dict[str, Any]  # strategy stats for IS period
    params_details: dict[str, Any] = field(default_factory=dict)


class WFOOptimizer:
    """Optimize strategy parameters on an IS window using optuna + freqtrade backtest."""

    def __init__(
        self,
        backtesting: Backtesting,
        *,
        n_epochs: int = 100,
        loss_function: str = "SharpeHyperOptLoss",
        spaces: list[str] | None = None,
    ) -> None:
        self.bt = backtesting
        self.n_epochs = n_epochs
        self.loss_function_name = loss_function
        self.spaces = spaces or ["buy", "sell", "roi", "stoploss"]

        # Load loss function
        loss_config = dict(self.bt.config)
        loss_config["hyperopt_loss"] = loss_function
        self.loss_fn = HyperOptLossResolver.load_hyperoptloss(loss_config)

    def _load_and_prepare_data(
        self, timerange: TimeRange
    ) -> tuple[dict[str, DataFrame], datetime, datetime]:
        """Load OHLCV data for a timerange and run indicators."""
        data = history.load_data(
            datadir=self.bt.config["datadir"],
            pairs=self.bt.pairlists.whitelist,
            timeframe=self.bt.timeframe,
            timerange=timerange,
            startup_candles=self.bt.required_startup,
            fail_without_data=False,
            data_format=self.bt.config["dataformat_ohlcv"],
            candle_type=self.bt.config.get("candle_type_def"),
        )
        if not data:
            raise RuntimeError(f"No data for timerange {timerange.timerange_str}")

        # Set price precision
        self.bt.price_pair_prec = {}
        self.bt.available_pairs = []
        for pair in data:
            self.bt.price_pair_prec[pair] = get_tick_size_over_time(data[pair])
            self.bt.available_pairs.append(pair)

        preprocessed = self.bt.strategy.advise_all_indicators(data)
        preprocessed = trim_dataframes(preprocessed, timerange, self.bt.required_startup)

        min_date, max_date = history.get_timerange(preprocessed)
        return preprocessed, min_date, max_date

    def _apply_params(self, params: dict[str, Any]) -> None:
        """Apply parameter dict to strategy (same as hyperopt does)."""
        for attr_name, attr in self.bt.strategy.enumerate_parameters():
            if attr.in_space and attr.optimize and attr_name in params:
                attr.value = params[attr_name]

        if "roi" in self.spaces and HyperoptTools.has_space(self.bt.config, "roi"):
            from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
            auto = HyperOptAuto(self.bt.config)
            auto.strategy = self.bt.strategy
            self.bt.strategy.minimal_roi = auto.generate_roi_table(params)

        if "stoploss" in self.spaces and "stoploss" in params:
            self.bt.strategy.stoploss = params["stoploss"]

        if "trailing" in self.spaces:
            from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
            auto = HyperOptAuto(self.bt.config)
            auto.strategy = self.bt.strategy
            if any(k.startswith("trailing_") for k in params):
                d = auto.generate_trailing_params(params)
                self.bt.strategy.trailing_stop = d.get("trailing_stop", False)
                self.bt.strategy.trailing_stop_positive = d.get("trailing_stop_positive")
                self.bt.strategy.trailing_stop_positive_offset = d.get("trailing_stop_positive_offset", 0.0)
                self.bt.strategy.trailing_only_offset_is_reached = d.get("trailing_only_offset_is_reached", False)

    def _get_dimensions(self) -> dict[str, Any]:
        """Get optuna search space from strategy hyperopt parameters."""
        dimensions = {}
        for attr_name, attr in self.bt.strategy.enumerate_parameters():
            if attr.in_space and attr.optimize:
                # Map freqtrade parameter types to optuna
                if hasattr(attr, "low") and hasattr(attr, "high"):
                    if hasattr(attr, "decimals"):
                        dimensions[attr_name] = {
                            "type": "float",
                            "low": float(attr.low),
                            "high": float(attr.high),
                        }
                    else:
                        dimensions[attr_name] = {
                            "type": "int",
                            "low": int(attr.low),
                            "high": int(attr.high),
                        }
                elif hasattr(attr, "opt_range"):
                    dimensions[attr_name] = {
                        "type": "categorical",
                        "choices": list(attr.opt_range),
                    }
        return dimensions

    def _suggest_params(self, trial: optuna.Trial, dimensions: dict) -> dict[str, Any]:
        """Generate a parameter suggestion from optuna trial."""
        params = {}
        for name, spec in dimensions.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    def optimize_fold(
        self,
        fold: WFOFold,
    ) -> OptimizationResult:
        """Run parameter optimization on a fold's IS window.

        Returns the best parameters and IS performance stats.
        """
        logger.info(
            "[WFO fold %d] Optimizing IS: %s -> %s (%d epochs)",
            fold.index, fold.is_start.isoformat(), fold.is_end.isoformat(), self.n_epochs,
        )

        # Load IS data
        processed, min_date, max_date = self._load_and_prepare_data(fold.is_timerange)
        dimensions = self._get_dimensions()

        if not dimensions:
            raise RuntimeError(
                "No optimizable parameters found. Strategy must define "
                "IntParameter/DecimalParameter/CategoricalParameter with optimize=True."
            )

        best_loss = float("inf")
        best_params: dict[str, Any] = {}
        best_is_stats: dict[str, Any] = {}

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_loss, best_params, best_is_stats

            params = self._suggest_params(trial, dimensions)
            self._apply_params(params)

            # Run backtest (resets state internally)
            bt_results = self.bt.backtest(
                processed=processed, start_date=min_date, end_date=max_date,
            )

            # Calculate loss
            strat_stats = generate_strategy_stats(
                self.bt.pairlists.whitelist,
                self.bt.strategy.get_strategy_name(),
                bt_results,
                min_date, max_date,
                market_change=0.0,
                is_hyperopt=True,
            )

            loss = self.loss_fn.hyperopt_loss_function(
                results=bt_results["results"],
                trade_count=len(bt_results["results"]),
                min_date=min_date,
                max_date=max_date,
                config=self.bt.config,
                processed=processed,
                backtest_stats=strat_stats,
            )

            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
                best_is_stats = strat_stats

            gc.collect()
            return loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_epochs, show_progress_bar=False)

        # Cleanup
        del processed
        gc.collect()

        logger.info(
            "[WFO fold %d] IS optimization done: best_loss=%.6f, n_trials=%d",
            fold.index, best_loss, len(study.trials),
        )

        return OptimizationResult(
            fold_index=fold.index,
            best_params=best_params,
            best_loss=best_loss,
            n_trials=len(study.trials),
            is_stats=best_is_stats,
        )
```

### Step 2: 编译检查

```bash
python -m py_compile user_custom/rolling_backtest/src/wfo_optimizer.py
```

### Step 3: Commit

```bash
git add user_custom/rolling_backtest/src/wfo_optimizer.py
git commit -m "feat(wfo): add WFO optimizer wrapping freqtrade hyperopt internals"
```

---

## Task 3: WFO 单折执行逻辑

**Objective:** 实现单折的完整流程：IS 优化 → 应用最优参数 → OOS 回测

**Files:**
- Create: `user_custom/rolling_backtest/src/wfo_fold.py`

### Step 1: 实现

```python
# src/wfo_fold.py
from __future__ import annotations

import gc
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.data import history
from freqtrade.data.btanalysis import get_tick_size_over_time, trade_list_to_dataframe
from freqtrade.data.converter import trim_dataframes
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.optimize_reports import generate_strategy_stats

from .wfo_optimizer import OptimizationResult, WFOOptimizer
from .wfo_windowing import WFOFold

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Complete result for one WFO fold."""
    fold_index: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    best_params: dict[str, Any]
    best_loss: float
    n_trials: int
    is_metrics: dict[str, Any]   # IS period summary metrics
    oos_metrics: dict[str, Any]  # OOS period summary metrics
    efficiency_ratio: float | None = None  # OOS_profit / IS_profit
    status: str = "ok"
    error: str | None = None


class WFOFoldRunner:
    """Execute a single WFO fold: optimize IS, validate OOS."""

    def __init__(
        self,
        backtesting: Backtesting,
        optimizer: WFOOptimizer,
    ) -> None:
        self.bt = backtesting
        self.optimizer = optimizer

    def _run_oos_backtest(
        self,
        fold: WFOFold,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run OOS backtest with fixed parameters. Returns strategy stats."""
        logger.info(
            "[WFO fold %d] OOS backtest: %s -> %s",
            fold.index, fold.oos_start.isoformat(), fold.oos_end.isoformat(),
        )

        # Apply optimized params
        self.optimizer._apply_params(params)

        # Load OOS data
        data = history.load_data(
            datadir=self.bt.config["datadir"],
            pairs=self.bt.pairlists.whitelist,
            timeframe=self.bt.timeframe,
            timerange=fold.oos_timerange,
            startup_candles=self.bt.required_startup,
            fail_without_data=False,
            data_format=self.bt.config["dataformat_ohlcv"],
            candle_type=self.bt.config.get("candle_type_def"),
        )
        if not data:
            raise RuntimeError(f"No OOS data for fold {fold.index}")

        self.bt.price_pair_prec = {}
        self.bt.available_pairs = []
        for pair in data:
            self.bt.price_pair_prec[pair] = get_tick_size_over_time(data[pair])
            self.bt.available_pairs.append(pair)

        preprocessed = self.bt.strategy.advise_all_indicators(data)
        preprocessed = trim_dataframes(preprocessed, fold.oos_timerange, self.bt.required_startup)
        min_date, max_date = history.get_timerange(preprocessed)

        bt_results = self.bt.backtest(
            processed=preprocessed,
            start_date=min_date,
            end_date=max_date,
        )

        strat_stats = generate_strategy_stats(
            self.bt.pairlists.whitelist,
            self.bt.strategy.get_strategy_name(),
            bt_results,
            min_date, max_date,
            market_change=0.0,
            is_hyperopt=False,
        )

        del data, preprocessed
        gc.collect()

        return strat_stats

    def run_fold(self, fold: WFOFold) -> FoldResult:
        """Execute complete fold: IS optimize → OOS validate."""
        try:
            # Phase 1: IS optimization
            opt_result = self.optimizer.optimize_fold(fold)

            # Phase 2: OOS backtest with best params
            oos_stats = self._run_oos_backtest(fold, opt_result.best_params)

            # Calculate efficiency ratio
            is_profit = opt_result.is_stats.get("profit_total", 0.0)
            oos_profit = oos_stats.get("profit_total", 0.0)
            if is_profit != 0:
                efficiency = oos_profit / is_profit
            else:
                efficiency = None

            # Extract key metrics
            is_metrics = _extract_key_metrics(opt_result.is_stats)
            oos_metrics = _extract_key_metrics(oos_stats)

            return FoldResult(
                fold_index=fold.index,
                is_start=fold.is_start.isoformat(),
                is_end=fold.is_end.isoformat(),
                oos_start=fold.oos_start.isoformat(),
                oos_end=fold.oos_end.isoformat(),
                best_params=opt_result.best_params,
                best_loss=opt_result.best_loss,
                n_trials=opt_result.n_trials,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
                efficiency_ratio=efficiency,
            )

        except Exception as exc:
            logger.exception("[WFO fold %d] failed", fold.index)
            return FoldResult(
                fold_index=fold.index,
                is_start=fold.is_start.isoformat(),
                is_end=fold.is_end.isoformat(),
                oos_start=fold.oos_start.isoformat(),
                oos_end=fold.oos_end.isoformat(),
                best_params={},
                best_loss=float("inf"),
                n_trials=0,
                is_metrics={},
                oos_metrics={},
                status="error",
                error=str(exc),
            )


def _extract_key_metrics(stats: dict) -> dict:
    """Extract key metrics from freqtrade strategy stats for comparison."""
    if not stats:
        return {}
    return {
        "profit_total": stats.get("profit_total", 0.0),
        "profit_total_abs": stats.get("profit_total_abs", 0.0),
        "trade_count": stats.get("trade_count", 0),
        "win_rate": stats.get("winning_trades", 0) / max(stats.get("trade_count", 1), 1),
        "sharpe": stats.get("sharpe", 0.0),
        "sortino": stats.get("sortino", 0.0),
        "max_drawdown": stats.get("max_drawdown", 0.0),
        "max_drawdown_abs": stats.get("max_drawdown_abs", 0.0),
        "profit_factor": stats.get("profit_factor", 0.0),
        "avg_duration": str(stats.get("holding_avg", "")),
    }
```

### Step 2: 编译检查

```bash
python -m py_compile user_custom/rolling_backtest/src/wfo_fold.py
```

### Step 3: Commit

```bash
git add user_custom/rolling_backtest/src/wfo_fold.py
git commit -m "feat(wfo): add WFO fold runner (IS optimize + OOS validate)"
```

---

## Task 4: WFO 报告生成

**Objective:** 聚合所有折的 IS/OOS 结果，生成过拟合分析报告

**Files:**
- Create: `user_custom/rolling_backtest/src/wfo_report.py`

### Step 1: 实现

```python
# src/wfo_report.py
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .wfo_fold import FoldResult

logger = logging.getLogger(__name__)


@dataclass
class WFOSummary:
    """Aggregated WFO results."""
    total_folds: int
    ok_folds: int
    failed_folds: int
    is_total_profit: float
    oos_total_profit: float
    oos_total_trades: int
    avg_efficiency_ratio: float | None
    overfit_score: float  # 0 = perfect, 1 = total overfit
    param_stability: float  # 0 = unstable, 1 = perfectly stable

from dataclasses import dataclass


def generate_wfo_report(
    folds: list[FoldResult],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Generate WFO summary report from fold results.

    Returns a dict with folds detail + summary + overfit analysis.
    """
    ok_folds = [f for f in folds if f.status == "ok"]

    # IS/OOS aggregation
    is_total_profit = sum(f.is_metrics.get("profit_total", 0.0) for f in ok_folds)
    oos_total_profit = sum(f.oos_metrics.get("profit_total", 0.0) for f in ok_folds)
    oos_total_trades = sum(f.oos_metrics.get("trade_count", 0) for f in ok_folds)

    # Efficiency ratios
    efficiencies = [f.efficiency_ratio for f in ok_folds if f.efficiency_ratio is not None]
    avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else None

    # Overfit score: 1 - (OOS_profit / IS_profit), clamped to [0, 1]
    if is_total_profit > 0:
        overfit_score = max(0.0, min(1.0, 1.0 - (oos_total_profit / is_total_profit)))
    elif is_total_profit == 0:
        overfit_score = 0.5  # indeterminate
    else:
        # IS negative — if OOS also negative, not necessarily overfit
        overfit_score = 0.5

    # Parameter stability: how consistent are best_params across folds
    param_stability = _compute_param_stability(ok_folds)

    report = {
        "folds": [asdict(f) for f in folds],
        "summary": {
            "total_folds": len(folds),
            "ok_folds": len(ok_folds),
            "failed_folds": len(folds) - len(ok_folds),
            "is_total_profit_pct": round(is_total_profit * 100, 4),
            "oos_total_profit_pct": round(oos_total_profit * 100, 4),
            "oos_total_trades": oos_total_trades,
            "avg_efficiency_ratio": round(avg_efficiency, 4) if avg_efficiency else None,
            "overfit_score": round(overfit_score, 4),
            "param_stability": round(param_stability, 4),
        },
        "overfit_analysis": _overfit_analysis(ok_folds, overfit_score, avg_efficiency),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("WFO report saved to %s", output_path)

    # Print console summary
    _print_wfo_summary(report)

    return report


def _compute_param_stability(folds: list[FoldResult]) -> float:
    """Compute parameter stability across folds (0=unstable, 1=stable).

    Uses coefficient of variation for numeric params.
    """
    if len(folds) < 2:
        return 1.0

    all_params = [f.best_params for f in folds if f.best_params]
    if not all_params:
        return 0.0

    # Get common numeric keys
    keys = set(all_params[0].keys())
    for p in all_params[1:]:
        keys &= set(p.keys())

    if not keys:
        return 0.0

    stabilities = []
    for key in keys:
        values = [p[key] for p in all_params if isinstance(p.get(key), (int, float))]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        if mean == 0:
            continue
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        cv = (variance ** 0.5) / abs(mean)  # coefficient of variation
        # Convert CV to stability: CV=0 → 1.0, CV>=1 → 0.0
        stabilities.append(max(0.0, 1.0 - cv))

    return sum(stabilities) / len(stabilities) if stabilities else 0.0


def _overfit_analysis(
    folds: list[FoldResult],
    overfit_score: float,
    avg_efficiency: float | None,
) -> dict[str, Any]:
    """Generate human-readable overfit analysis."""
    analysis: dict[str, Any] = {}

    if overfit_score < 0.3:
        analysis["verdict"] = "LOW_OVERFIT"
        analysis["description"] = "策略 OOS 表现接近 IS，过拟合风险低。"
    elif overfit_score < 0.6:
        analysis["verdict"] = "MODERATE_OVERFIT"
        analysis["description"] = "策略 OOS 表现明显弱于 IS，存在一定过拟合。建议增大 IS 窗口或简化参数空间。"
    else:
        analysis["verdict"] = "HIGH_OVERFIT"
        analysis["description"] = "策略 OOS 表现远弱于 IS，严重过拟合。建议减少优化参数数量、使用更长 IS 窗口、或换用更鲁棒的 loss function。"

    # Per-fold IS vs OOS comparison
    analysis["fold_comparison"] = []
    for f in folds:
        is_p = f.is_metrics.get("profit_total", 0.0) * 100
        oos_p = f.oos_metrics.get("profit_total", 0.0) * 100
        analysis["fold_comparison"].append({
            "fold": f.fold_index,
            "is_profit_pct": round(is_p, 2),
            "oos_profit_pct": round(oos_p, 2),
            "efficiency": round(f.efficiency_ratio, 4) if f.efficiency_ratio else None,
        })

    return analysis


def _print_wfo_summary(report: dict) -> None:
    """Print compact WFO summary to console."""
    s = report["summary"]
    a = report["overfit_analysis"]

    lines = [
        "",
        "=" * 80,
        " Walk-Forward Optimization Summary",
        "=" * 80,
        f" Folds: {s['ok_folds']}/{s['total_folds']} ok | "
        f"IS Profit: {s['is_total_profit_pct']:.2f}% | "
        f"OOS Profit: {s['oos_total_profit_pct']:.2f}% | "
        f"OOS Trades: {s['oos_total_trades']}",
        f" Avg Efficiency: {s['avg_efficiency_ratio']:.4f}" if s['avg_efficiency_ratio'] else " Avg Efficiency: N/A",
        f" Overfit Score: {s['overfit_score']:.4f} | "
        f"Param Stability: {s['param_stability']:.4f}",
        f" Verdict: {a['verdict']}",
        "-" * 80,
        f" {'Fold':>4} | {'IS Profit':>10} | {'OOS Profit':>10} | {'Efficiency':>10}",
        "-" * 80,
    ]

    for fc in a.get("fold_comparison", []):
        eff_str = f"{fc['efficiency']:.4f}" if fc["efficiency"] is not None else "N/A"
        lines.append(
            f" {fc['fold']:>4} | {fc['is_profit_pct']:>9.2f}% | "
            f"{fc['oos_profit_pct']:>9.2f}% | {eff_str:>10}"
        )

    lines.append("=" * 80)
    print("\n".join(lines))
```

### Step 2: 编译检查 + Commit

```bash
python -m py_compile user_custom/rolling_backtest/src/wfo_report.py
git add user_custom/rolling_backtest/src/wfo_report.py
git commit -m "feat(wfo): add WFO report generation with overfit analysis"
```

---

## Task 5: WFO 主引擎

**Objective:** 编排整个 WFO 流程：生成折 → 逐折优化+验证 → 聚合报告

**Files:**
- Create: `user_custom/rolling_backtest/src/wfo_engine.py`

### Step 1: 实现

```python
# src/wfo_engine.py
from __future__ import annotations

import gc
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from freqtrade.optimize.backtesting import Backtesting
from freqtrade.util import dt_now

from .wfo_fold import FoldResult, WFOFoldRunner
from .wfo_optimizer import WFOOptimizer
from .wfo_report import generate_wfo_report
from .wfo_windowing import WFOFold, build_wfo_folds

logger = logging.getLogger(__name__)


class WalkForwardEngine:
    """Walk-Forward Optimization orchestrator.

    Flow:
    1. Build IS/OOS fold pairs from global timerange
    2. For each fold:
       a. Optimize parameters on IS window
       b. Backtest OOS window with best IS params
    3. Aggregate results and detect overfit
    """

    def __init__(
        self,
        backtesting: Backtesting,
        *,
        is_days: int = 180,
        oos_days: int = 30,
        step_days: int | None = None,
        n_epochs: int = 100,
        loss_function: str = "SharpeHyperOptLoss",
        spaces: list[str] | None = None,
    ) -> None:
        self.bt = backtesting
        self.is_days = is_days
        self.oos_days = oos_days
        self.step_days = step_days
        self.n_epochs = n_epochs
        self.loss_function = loss_function
        self.spaces = spaces

        self.optimizer = WFOOptimizer(
            backtesting=backtesting,
            n_epochs=n_epochs,
            loss_function=loss_function,
            spaces=spaces,
        )
        self.fold_runner = WFOFoldRunner(
            backtesting=backtesting,
            optimizer=self.optimizer,
        )

    def run(
        self,
        output_path: Path | None = None,
        *,
        fail_fast: bool = False,
        max_failed_folds: int = 0,
    ) -> dict[str, Any]:
        """Run complete Walk-Forward Optimization.

        Args:
            output_path: Path to save JSON results.
            fail_fast: Stop on first fold failure.
            max_failed_folds: Max tolerated failures (0=unlimited).

        Returns:
            WFO report dict.
        """
        strategy_name = self.bt.strategy.get_strategy_name()
        started_at = dt_now()

        # Build folds
        folds = build_wfo_folds(
            self.bt.timerange,
            timeframe=self.bt.timeframe,
            is_days=self.is_days,
            oos_days=self.oos_days,
            step_days=self.step_days,
        )

        logger.info(
            "WFO start: strategy=%s, folds=%d, IS=%dd, OOS=%dd, epochs=%d, loss=%s",
            strategy_name, len(folds), self.is_days, self.oos_days,
            self.n_epochs, self.loss_function,
        )

        # Execute folds
        results: list[FoldResult] = []
        failed_count = 0

        for fold in folds:
            logger.info(
                "[WFO fold %d/%d] IS: %s→%s | OOS: %s→%s",
                fold.index + 1, len(folds),
                fold.is_start.strftime("%Y-%m-%d"), fold.is_end.strftime("%Y-%m-%d"),
                fold.oos_start.strftime("%Y-%m-%d"), fold.oos_end.strftime("%Y-%m-%d"),
            )

            fold_result = self.fold_runner.run_fold(fold)
            results.append(fold_result)

            if fold_result.status == "error":
                failed_count += 1
                if fail_fast:
                    logger.error("Fail-fast: stopping after fold %d error", fold.index)
                    break
                if max_failed_folds > 0 and failed_count > max_failed_folds:
                    logger.error("Too many failed folds (%d), stopping", failed_count)
                    break
            else:
                is_p = fold_result.is_metrics.get("profit_total", 0) * 100
                oos_p = fold_result.oos_metrics.get("profit_total", 0) * 100
                eff = fold_result.efficiency_ratio
                eff_str = f"{eff:.4f}" if eff is not None else "N/A"
                logger.info(
                    "[WFO fold %d] IS=%.2f%% OOS=%.2f%% efficiency=%s",
                    fold.index, is_p, oos_p, eff_str,
                )

            gc.collect()

        ended_at = dt_now()

        # Generate report
        report = generate_wfo_report(results, output_path)
        report["metadata"] = {
            "strategy": strategy_name,
            "is_days": self.is_days,
            "oos_days": self.oos_days,
            "step_days": self.step_days or self.oos_days,
            "n_epochs": self.n_epochs,
            "loss_function": self.loss_function,
            "spaces": self.spaces or ["buy", "sell", "roi", "stoploss"],
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_sec": round((ended_at - started_at).total_seconds(), 3),
        }

        if output_path:
            output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        return report
```

### Step 2: 编译检查 + Commit

```bash
python -m py_compile user_custom/rolling_backtest/src/wfo_engine.py
git add user_custom/rolling_backtest/src/wfo_engine.py
git commit -m "feat(wfo): add WalkForwardEngine orchestrator"
```

---

## Task 6: CLI 入口脚本

**Objective:** 创建 `run_wfo.py` CLI 脚本，复用现有的 config/args 模式

**Files:**
- Create: `user_custom/rolling_backtest/scripts/run_wfo.py`

### Step 1: 实现

```python
#!/usr/bin/env python3
# scripts/run_wfo.py
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from freqtrade.commands.optimize_commands import setup_optimize_configuration
from freqtrade.enums import RunMode
from freqtrade.optimize.backtesting import Backtesting

from user_custom.rolling_backtest.src.wfo_engine import WalkForwardEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("run_wfo")

# Silence optuna's verbose logging
logging.getLogger("optuna").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-Forward Optimization")
    p.add_argument("--config", required=True, help="Freqtrade config path")
    p.add_argument("--strategy", required=True, help="Strategy name")
    p.add_argument("--timerange", required=True, help="Global timerange (e.g. 20220101-20251231)")
    p.add_argument("--datadir", default="/data/freqtrade_data", help="OHLCV data directory")
    p.add_argument("--user-data-dir", default="/root/workspace/freqtrade/user_data")

    # WFO parameters
    p.add_argument("--is-days", type=int, default=180, help="In-sample window days")
    p.add_argument("--oos-days", type=int, default=30, help="Out-of-sample window days")
    p.add_argument("--step-days", type=int, default=None, help="Step days (default=oos-days)")
    p.add_argument("--n-epochs", type=int, default=100, help="Optuna trials per fold")
    p.add_argument("--loss-function", default="SharpeHyperOptLoss", help="Hyperopt loss function")
    p.add_argument("--spaces", nargs="+", default=None, help="Hyperopt spaces (buy sell roi stoploss trailing)")

    # Execution
    p.add_argument("--output-json", default="user_custom/rolling_backtest/output/wfo_result.json")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--max-failed-folds", type=int, default=0)
    p.add_argument("--log-file", default=None)
    return p.parse_args()


def main() -> None:
    ns = parse_args()

    if ns.log_file:
        fp = Path(ns.log_file)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(fp, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
        logging.getLogger().addHandler(fh)

    args: dict[str, Any] = {
        "config": [ns.config],
        "strategy": ns.strategy,
        "strategy_list": [],
        "timerange": ns.timerange,
        "verbosity": 0,
        "timeframe": None,
        "timeframe_detail": None,
        "datadir": ns.datadir,
        "user_data_dir": ns.user_data_dir,
        "export": "none",
        "cache": "none",
        "backtest_breakdown": [],
        "enable_protections": False,
    }

    config = setup_optimize_configuration(args, RunMode.BACKTEST)

    # Ensure hyperopt spaces config
    if ns.spaces:
        config["hyperopt_spaces"] = ns.spaces

    backtesting = Backtesting(config)

    engine = WalkForwardEngine(
        backtesting=backtesting,
        is_days=ns.is_days,
        oos_days=ns.oos_days,
        step_days=ns.step_days,
        n_epochs=ns.n_epochs,
        loss_function=ns.loss_function,
        spaces=ns.spaces,
    )

    report = engine.run(
        output_path=Path(ns.output_json),
        fail_fast=ns.fail_fast,
        max_failed_folds=ns.max_failed_folds,
    )

    logger.info("WFO complete. Report: %s", ns.output_json)


if __name__ == "__main__":
    main()
```

### Step 2: 编译检查 + Commit

```bash
python -m py_compile user_custom/rolling_backtest/scripts/run_wfo.py
git add user_custom/rolling_backtest/scripts/run_wfo.py
git commit -m "feat(wfo): add run_wfo.py CLI entry point"
```

---

## Task 7: Smoke Test — 用 ParityTestStrategy 做端到端验证

**Objective:** 用现有的随机数据 + ParityTestStrategy 跑一个小规模 WFO，验证全流程可用

**Files:**
- Create: `user_custom/rolling_backtest/tests/test_wfo_smoke.py`

### 测试策略

ParityTestStrategy 没有 `optimize=True` 的参数，所以需要创建一个带可优化参数的测试策略：

```python
# tests/parity_test/WFOTestStrategy.py
# 基于 ParityTestStrategy，增加可优化的 EMA 周期参数
```

### 运行命令

```bash
cd /root/workspace/freqtrade && source .venv/bin/activate
python user_custom/rolling_backtest/scripts/run_wfo.py \
  --config user_custom/rolling_backtest/tests/parity_test/config_parity.json \
  --strategy WFOTestStrategy \
  --timerange 20230101-20250101 \
  --is-days 90 \
  --oos-days 30 \
  --n-epochs 10 \
  --output-json /tmp/wfo_smoke_result.json \
  2>&1 | tee /tmp/wfo_smoke.log
```

### 验证点

1. 生成至少 3 个 fold
2. 每个 fold 有 IS/OOS metrics
3. 输出 overfit 分析
4. JSON 文件可解析
5. 无 crash / unhandled exception

### Commit

```bash
git add user_custom/rolling_backtest/tests/
git commit -m "test(wfo): add WFO smoke test with optimizable strategy"
```

---

## Task 8: 更新文档和 Skill

**Objective:** 更新 README 和 SKILL 文件，记录 WFO 功能

**Files:**
- Modify: `user_custom/rolling_backtest/README.md`
- Modify: skill `freqtrade-rolling-backtest-futures`

### 内容

- 新增 WFO 章节说明
- CLI 用法示例
- 参数调优建议（IS 窗口选择、epochs 数量、loss function 选择）
- 过拟合指标解读

### Commit

```bash
git add README.md
git commit -m "docs: add Walk-Forward Optimization documentation"
```

---

## 实施顺序和依赖关系

```
Task 1 (windowing) ──┐
                      ├── Task 2 (optimizer) ──┐
                      │                         ├── Task 3 (fold runner) ──┐
                      │                         │                          ├── Task 5 (engine)
                      │                         │    Task 4 (report) ──────┘        │
                      │                         │                                   │
                      └─────────────────────────┘                                   │
                                                                                    ▼
                                                                    Task 6 (CLI) ── Task 7 (smoke)
                                                                                         │
                                                                                    Task 8 (docs)
```

## 关键设计决策

1. **不修改 freqtrade 核心** — 所有代码在 `user_custom/` 下
2. **复用 `backtest()` 方法** — hyperopt 也是调用这个，保证回测结果一致性
3. **每折独立** — 折之间不保持交易状态（不同于滚动回测），因为每折参数不同
4. **optuna 直接调用** — 不走 freqtrade 的 Hyperopt 类（太重、太多副作用），直接用 optuna study + `backtest()` 组合
5. **效率比是核心指标** — OOS_profit / IS_profit 直接反映过拟合程度
