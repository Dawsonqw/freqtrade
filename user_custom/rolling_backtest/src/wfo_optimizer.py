"""WFO Optimizer — single-fold parameter optimization using optuna + freqtrade backtest."""
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
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver

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
        self.spaces = spaces or ["buy", "sell"]

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
            data_format=self.bt.config.get("dataformat_ohlcv", "feather"),
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

        if "stoploss" in params:
            self.bt.strategy.stoploss = params["stoploss"]

        if "trailing_stop" in params:
            self.bt.strategy.trailing_stop = params.get("trailing_stop", False)
            self.bt.strategy.trailing_stop_positive = params.get("trailing_stop_positive")
            self.bt.strategy.trailing_stop_positive_offset = params.get(
                "trailing_stop_positive_offset", 0.0
            )
            self.bt.strategy.trailing_only_offset_is_reached = params.get(
                "trailing_only_offset_is_reached", False
            )

    def _get_dimensions(self) -> dict[str, Any]:
        """Get optuna search space from strategy hyperopt parameters."""
        dimensions = {}
        for attr_name, attr in self.bt.strategy.enumerate_parameters():
            if attr.in_space and attr.optimize:
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
        starting_balance = self.bt.config.get("dry_run_wallet", 1000.0)

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_loss, best_params, best_is_stats

            params = self._suggest_params(trial, dimensions)
            self._apply_params(params)

            # Run backtest
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
                starting_balance=starting_balance,
            )

            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
                best_is_stats = strat_stats

            return loss

        study = optuna.create_study(direction="minimize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
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
