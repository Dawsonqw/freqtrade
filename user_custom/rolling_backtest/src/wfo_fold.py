"""WFO Fold Runner — execute a single fold: IS optimize → OOS validate."""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.data import history
from freqtrade.data.btanalysis import get_tick_size_over_time
from freqtrade.data.converter import trim_dataframes
from freqtrade.enums import CandleType, TradingMode
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
            data_format=self.bt.config.get("dataformat_ohlcv", "feather"),
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

        # Load futures funding/mark data if in futures mode
        if self.bt.trading_mode == TradingMode.FUTURES:
            from freqtrade.exchange.exchange import timeframe_to_seconds
            funding_fee_tf = self.bt.exchange.get_option("funding_fee_timeframe")
            mark_tf = self.bt.exchange.get_option("mark_ohlcv_timeframe")
            self.bt.funding_fee_timeframe_secs = timeframe_to_seconds(funding_fee_tf)

            funding_rates = history.load_data(
                datadir=self.bt.config["datadir"],
                pairs=self.bt.pairlists.whitelist,
                timeframe=funding_fee_tf,
                timerange=fold.oos_timerange,
                startup_candles=0,
                fail_without_data=False,
                fill_up_missing=False,
                data_format=self.bt.config.get("dataformat_ohlcv", "feather"),
                candle_type=CandleType.FUNDING_RATE,
            )
            mark_rates = history.load_data(
                datadir=self.bt.config["datadir"],
                pairs=self.bt.pairlists.whitelist,
                timeframe=mark_tf,
                timerange=fold.oos_timerange,
                startup_candles=0,
                fail_without_data=False,
                fill_up_missing=False,
                data_format=self.bt.config.get("dataformat_ohlcv", "feather"),
                candle_type=CandleType.from_string(
                    self.bt.exchange.get_option("mark_ohlcv_price")
                ),
            )
            self.bt.futures_data = {}
            funding_rate_cfg = self.bt.config.get("futures_funding_rate", None)
            for pair in self.bt.pairlists.whitelist:
                if pair in funding_rates and pair in mark_rates:
                    self.bt.futures_data[pair] = self.bt.exchange.combine_funding_and_mark(
                        funding_rates=funding_rates[pair],
                        mark_rates=mark_rates[pair],
                        futures_funding_rate=funding_rate_cfg,
                    )
                else:
                    logger.warning(
                        "No funding/mark data for %s — using futures_funding_rate=0", pair,
                    )
                    import pandas as pd
                    ohlcv = data.get(pair)
                    if ohlcv is not None:
                        ff = pd.DataFrame({
                            "date": ohlcv["date"],
                            "open_fund": 0.0,
                            "open_mark": ohlcv["open"],
                        })
                        self.bt.futures_data[pair] = ff
        else:
            self.bt.futures_data = {}

        min_date, max_date = history.get_timerange(preprocessed)

        bt_results = self.bt.backtest(
            processed=preprocessed,
            start_date=min_date,
            end_date=max_date,
        )
        # Inject fields expected by generate_strategy_stats
        bt_results.setdefault("backtest_start_time", int(min_date.timestamp()))
        bt_results.setdefault("backtest_end_time", int(max_date.timestamp()))

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
    total_trades = stats.get("total_trades", 0)
    wins = stats.get("wins", 0)
    return {
        "profit_total": stats.get("profit_total", 0.0),
        "profit_total_abs": stats.get("profit_total_abs", 0.0),
        "trade_count": total_trades,
        "win_rate": wins / max(total_trades, 1),
        "sharpe": stats.get("sharpe", 0.0),
        "sortino": stats.get("sortino", 0.0),
        "max_drawdown": stats.get("max_drawdown_account", 0.0),
        "max_drawdown_abs": stats.get("max_drawdown_abs", 0.0),
        "profit_factor": stats.get("profit_factor", 0.0),
        "avg_duration": str(stats.get("holding_avg", "")),
    }
