from __future__ import annotations

import gc
import json
import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.data import history
from freqtrade.data.btanalysis import get_tick_size_over_time, trade_list_to_dataframe
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.metrics import combined_dataframes_with_rel_mean
from freqtrade.optimize.backtesting import Backtesting, convert_bt_wallet_collection
from freqtrade.optimize.optimize_reports import generate_backtest_stats, store_backtest_results
from freqtrade.optimize.optimize_reports.bt_output import show_backtest_results
from freqtrade.persistence import LocalTrade, PairLocks
from freqtrade.util import dt_now
from freqtrade.util.datetime_helpers import dt_ts

import numpy as np

from .comparison_report import build_comparison_report, format_comparison_table
from .pair_filter import PairAvailabilityFilter
from .parity_tools import compute_trade_digest
from .signal_export import SignalExporter
from .window_metrics import WindowPerformance, compute_all_window_performances
from .windowing import Window, WindowPlan, build_windows


logger = logging.getLogger(__name__)


@dataclass
class WindowRunStat:
    index: int
    start: str
    end: str
    pairs_loaded: int
    candles_total: int
    trades_after_window: int
    open_trades_after_window: int
    duration_sec: float
    status: str
    error: str | None = None
    carry_over_trades: int = 0


class RollingBacktestRunner:
    """
    Chunked backtesting runner reusing freqtrade Backtesting internals.
    - Loads OHLCV by time window.
    - Preserves in-memory trade/wallet state across windows.
    - Frees dataframe memory after each window.
    """

    def __init__(
        self,
        backtesting: Backtesting,
        window_days: int = 0,
        *,
        mem_budget_mb: float = 0,
        preferred_days: int = 0,
        min_window_days: int = 7,
        max_window_days: int = 90,
        parallel_workers: int = 0,
    ) -> None:
        self.bt = backtesting
        self.window_days = window_days
        self.mem_budget_mb = mem_budget_mb
        self.preferred_days = preferred_days
        self.min_window_days = min_window_days
        self.max_window_days = max_window_days
        self.parallel_workers = parallel_workers
        self.window_plan: WindowPlan | None = None
        self.window_stats: list[WindowRunStat] = []
        self._dynamic_pairlist = self.bt.config.get("enable_dynamic_pairlist", False)
        self._pair_filter = PairAvailabilityFilter()
        self._original_whitelist = list(self.bt.pairlists.whitelist)
        # Pre-loaded full-range data cache (populated by _preload_all_data)
        self._preloaded_data: dict[str, DataFrame] | None = None

    # ------------------------------------------------------------------ #
    #  Optimisation: pre-load data once, slice per window                  #
    # ------------------------------------------------------------------ #

    def _preload_all_data(self, full_tr: TimeRange) -> None:
        """Load OHLCV for the full timerange once and cache in memory.

        Subsequent windows use _slice_preloaded_data() instead of disk I/O.
        Also pre-computes tick_size per pair (expensive, avoids per-window re-computation).
        """
        t0 = _time.monotonic()
        pairs = list(self._original_whitelist)
        data = history.load_data(
            datadir=self.bt.config["datadir"],
            pairs=pairs,
            timeframe=self.bt.timeframe,
            timerange=full_tr,
            startup_candles=self.bt.required_startup,
            fail_without_data=False,
            data_format=self.bt.config["dataformat_ohlcv"],
            candle_type=self.bt.config.get("candle_type_def"),
        )
        t1 = _time.monotonic()
        total_rows = sum(len(v) for v in data.values())
        logger.info(
            "Pre-loaded full-range data: %d pairs, %d rows in %.1fs",
            len(data), total_rows, t1 - t0,
        )

        # Lazy tick_size cache — computed per-pair on first encounter in _slice
        self._preloaded_tick_sizes: dict[str, object] = {}

        self._preloaded_data = data

    def _slice_preloaded_data(
        self, tr: TimeRange, window: Window | None = None
    ) -> dict[str, DataFrame]:
        """Slice pre-loaded data for a specific window time range.

        Much faster than loading from disk — just pandas boolean indexing.
        """
        import pandas as pd

        assert self._preloaded_data is not None, "Call _preload_all_data first"

        pairs = list(self._original_whitelist)
        if window is not None and self._pair_filter.loaded:
            original_count = len(pairs)
            pairs = self._pair_filter.filter_pairs(
                pairs,
                window_start=window.start.strftime("%Y-%m-%d"),
                window_end=window.end.strftime("%Y-%m-%d"),
                timeframe=self.bt.timeframe,
            )
            if len(pairs) < original_count:
                logger.info(
                    "[window %s] pair availability filter: %d -> %d pairs",
                    window.index, original_count, len(pairs),
                )

        # Convert timerange to datetime for slicing, extending start by startup candles
        # so indicators have the same warm-up period as _load_window_data (which passes
        # startup_candles=self.bt.required_startup to history.load_data).
        start_dt = pd.Timestamp(tr.startdt) if tr.startdt else None
        end_dt = pd.Timestamp(tr.stopdt) if tr.stopdt else None

        if start_dt is not None and self.bt.required_startup > 0:
            from freqtrade.exchange import timeframe_to_seconds
            startup_secs = self.bt.required_startup * timeframe_to_seconds(self.bt.timeframe)
            start_dt = start_dt - pd.Timedelta(seconds=startup_secs)

        result: dict[str, DataFrame] = {}
        for pair in pairs:
            if pair not in self._preloaded_data:
                continue
            df = self._preloaded_data[pair]
            if df.empty:
                continue
            dates = df["date"]
            if start_dt is not None and end_dt is not None:
                mask = (dates >= start_dt) & (dates <= end_dt)
            elif start_dt is not None:
                mask = dates >= start_dt
            elif end_dt is not None:
                mask = dates <= end_dt
            else:
                mask = slice(None)
            sliced = df.loc[mask]
            if not sliced.empty:
                result[pair] = sliced.copy()

        # Update backtesting pair tracking — lazy tick_size with cache
        self.bt.price_pair_prec = {}
        self.bt.available_pairs = []
        for pair in result:
            if pair not in self._preloaded_tick_sizes:
                # First time seeing this pair — compute and cache
                self._preloaded_tick_sizes[pair] = get_tick_size_over_time(
                    self._preloaded_data[pair].copy()
                )
            self.bt.price_pair_prec[pair] = self._preloaded_tick_sizes[pair]
            self.bt.available_pairs.append(pair)
        if result and self._pair_filter.loaded:
            self.bt.pairlists._whitelist = list(result.keys())

        return result

    # ------------------------------------------------------------------ #
    #  Optimisation: parallel indicator computation                        #
    # ------------------------------------------------------------------ #

    def _parallel_advise_all_indicators(
        self, data: dict[str, DataFrame], workers: int
    ) -> dict[str, DataFrame]:
        """Compute indicators in parallel using threads.

        TA-Lib releases the GIL during C computation, so threads give
        real parallelism for the per-pair populate_indicators() calls.
        The strategy's advise_all_indicators may have a post-processing
        step (e.g. cross-pair ranking) that must run in the main thread.
        We handle this by:
        1. Parallel: call populate_indicators per pair (thread pool)
        2. Serial: call the strategy's post-processing if it overrides
           advise_all_indicators (detected by checking for _precomputed_ranks).
        """
        from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper

        strategy = self.bt.strategy
        pairs = list(data.keys())

        def _compute_one(pair: str) -> tuple[str, DataFrame]:
            pair_data = data[pair].copy()
            result_df = strategy.advise_indicators(pair_data, {"pair": pair}).copy()
            return pair, result_df

        t0 = _time.monotonic()
        result: dict[str, DataFrame] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for pair, df in pool.map(lambda p: _compute_one(p), pairs):
                result[pair] = df
        t1 = _time.monotonic()
        logger.info(
            "Parallel indicators (%d workers): %d pairs in %.1fs",
            workers, len(result), t1 - t0,
        )

        # Post-processing: cross-pair ranking (strategy-specific)
        if hasattr(strategy, '_precomputed_ranks'):
            t2 = _time.monotonic()
            # Call the strategy's ranking logic directly
            strategy._precomputed_ranks = {}
            self._compute_rankings(strategy, result)
            t3 = _time.monotonic()
            logger.info("Cross-pair ranking: %.1fs", t3 - t2)

        return result

    def _compute_rankings(self, strategy, result: dict[str, DataFrame]) -> None:
        """Run the cross-pair ranking logic from FullMarketDynamicStrategy."""
        import pandas as pd

        frames = []
        for pair, df in result.items():
            if df is None or len(df) < 30:
                continue
            sub = df[["date", "volatility_pct", "volume_usd", "adx"]].dropna().copy()
            min_vol = getattr(strategy, 'min_volume_usd', 100_000)
            sub = sub[(sub["volume_usd"] >= min_vol) & (sub["volatility_pct"] > 0)]
            if sub.empty:
                continue
            sub["pair"] = pair
            sub["ts"] = sub["date"].astype("int64") // 10**6
            frames.append(sub[["ts", "pair", "volatility_pct", "volume_usd", "adx"]])

        if not frames:
            return

        scores_df = pd.concat(frames, ignore_index=True)
        top_n = getattr(strategy, 'top_n_pairs', 30)
        w_vol = getattr(strategy, 'weight_volatility', 0.4)
        w_liq = getattr(strategy, 'weight_liquidity', 0.3)
        w_adx = getattr(strategy, 'weight_adx', 0.3)

        for ts, group in scores_df.groupby("ts"):
            if len(group) <= top_n:
                strategy._precomputed_ranks[int(ts)] = set(group["pair"].tolist())
                continue
            g = group.copy()
            g["score"] = (
                w_vol * g["volatility_pct"].rank(pct=True)
                + w_liq * g["volume_usd"].rank(pct=True)
                + w_adx * g["adx"].rank(pct=True)
            )
            top = g.nlargest(top_n, "score")
            strategy._precomputed_ranks[int(ts)] = set(top["pair"].tolist())

        logger.info("Rankings computed: %d timestamps", len(strategy._precomputed_ranks))

    def _refresh_pairlist_for_window(self, window: Window) -> list[str]:
        """Refresh pairlist at the start of each window (dynamic pairlist mode).

        In dynamic mode, calls pairlists.refresh_pairlist() using available data
        for the window so filters like VolumePairList can re-rank pairs.
        Returns the active whitelist for this window.
        """
        if not self._dynamic_pairlist or not self.bt.pairlists:
            return self.bt.pairlists.whitelist

        # Refresh with available pairs so filters can rank/filter
        self.bt.pairlists.refresh_pairlist(pairs=self.bt.available_pairs or None)
        whitelist = self.bt.pairlists.whitelist
        logger.info(
            "[window %s] dynamic pairlist refreshed: %d pairs",
            window.index,
            len(whitelist),
        )
        return whitelist

    def _load_window_data(self, tr: TimeRange, window: Window | None = None) -> dict[str, DataFrame]:
        # Restore full whitelist before filtering for this window
        pairs = list(self._original_whitelist)
        # Filter pairs by availability for this window's time range
        if window is not None and self._pair_filter.loaded:
            original_count = len(pairs)
            pairs = self._pair_filter.filter_pairs(
                pairs,
                window_start=window.start.strftime("%Y-%m-%d"),
                window_end=window.end.strftime("%Y-%m-%d"),
                timeframe=self.bt.timeframe,
            )
            if len(pairs) < original_count:
                logger.info(
                    "[window %s] pair availability filter: %d -> %d pairs",
                    window.index,
                    original_count,
                    len(pairs),
                )
        data = history.load_data(
            datadir=self.bt.config["datadir"],
            pairs=pairs,
            timeframe=self.bt.timeframe,
            timerange=tr,
            startup_candles=self.bt.required_startup,
            fail_without_data=False,
            data_format=self.bt.config["dataformat_ohlcv"],
            candle_type=self.bt.config.get("candle_type_def"),
        )
        self.bt.price_pair_prec = {}
        self.bt.available_pairs = []
        for pair in data:
            self.bt.price_pair_prec[pair] = get_tick_size_over_time(data[pair])
            self.bt.available_pairs.append(pair)
        # Update pairlists whitelist to only include pairs with loaded data,
        # so dp.current_whitelist() in strategies won't query missing pairs
        if data and self._pair_filter.loaded:
            self.bt.pairlists._whitelist = list(data.keys())
        return data

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

    def _run_single_window(
        self,
        *,
        window: Window,
        is_last_window: bool,
        signal_exporter: SignalExporter | None = None,
    ) -> None:
        t0 = dt_now()
        self.bt.timerange = window.timerange

        # Record carry-over trades from previous window
        carry_over = len(LocalTrade.bt_trades_open)

        # Dynamic pairlist: refresh before loading data
        active_pairs = self._refresh_pairlist_for_window(window)

        # Extend data load range to cover open trades' entry points
        effective_tr = self._get_extended_timerange_for_open_trades(window)
        _t_load = _time.monotonic()

        # Use pre-loaded data if available, otherwise load from disk
        if self._preloaded_data is not None:
            raw_data = self._slice_preloaded_data(effective_tr, window=window)
        else:
            raw_data = self._load_window_data(effective_tr, window=window)
        if not raw_data:
            t1 = dt_now()
            self.window_stats.append(
                WindowRunStat(
                    index=window.index,
                    start=window.start.isoformat(),
                    end=window.end.isoformat(),
                    pairs_loaded=0,
                    candles_total=0,
                    trades_after_window=len(LocalTrade.bt_trades),
                    open_trades_after_window=len(LocalTrade.bt_trades_open),
                    duration_sec=round((t1 - t0).total_seconds(), 3),
                    status="no_data",
                    error=None,
                )
            )
            logger.warning(
                "[window %s] no data in %s -> %s",
                window.index,
                window.start.strftime(DATETIME_PRINT_FORMAT),
                window.end.strftime(DATETIME_PRINT_FORMAT),
            )
            return

        self.bt._load_bt_data_detail()
        _t_load_done = _time.monotonic()
        logger.info(
            "[window %s] data loaded: %d pairs, %.1fs",
            window.index, len(raw_data), _t_load_done - _t_load,
        )

        # 1) indicators — parallel if workers > 0
        _t_ind = _time.monotonic()
        if self.parallel_workers > 0:
            preprocessed = self._parallel_advise_all_indicators(raw_data, self.parallel_workers)
        else:
            preprocessed = self.bt.strategy.advise_all_indicators(raw_data)
        _t_ind_done = _time.monotonic()
        logger.info(
            "[window %s] indicators computed: %.1fs",
            window.index, _t_ind_done - _t_ind,
        )

        # Collect signals if exporter is active
        if signal_exporter is not None:
            signal_exporter.collect_window(window.index, preprocessed)

        # 2) trim to the actual window for signal generation (startup is handled by trim)
        preprocessed_tmp = trim_dataframes(preprocessed, window.timerange, self.bt.required_startup)
        if not preprocessed_tmp:
            t1 = dt_now()
            self.window_stats.append(
                WindowRunStat(
                    index=window.index,
                    start=window.start.isoformat(),
                    end=window.end.isoformat(),
                    pairs_loaded=0,
                    candles_total=0,
                    trades_after_window=len(LocalTrade.bt_trades),
                    open_trades_after_window=len(LocalTrade.bt_trades_open),
                    duration_sec=round((t1 - t0).total_seconds(), 3),
                    status="no_data",
                    error="empty after startup trim",
                )
            )
            logger.warning("[window %s] empty after startup trim", window.index)
            return

        min_date, max_date = history.get_timerange(preprocessed_tmp)
        processed_lists = self.bt._get_ohlcv_as_lists(preprocessed)

        _t_bt = _time.monotonic()
        for current_time, pair, row, is_last_row, trade_dir in self.bt.time_pair_generator(
            min_date, max_date, list(processed_lists.keys()), processed_lists
        ):
            if not self.bt._can_short or trade_dir is None:
                self.bt.backtest_loop(row, pair, current_time, trade_dir, not is_last_row)
            else:
                for _ in (0, 1):
                    closed_dir = self.bt.backtest_loop(
                        row, pair, current_time, trade_dir, not is_last_row
                    )
                    if not closed_dir or closed_dir == trade_dir:
                        break

        if is_last_window:
            self.bt.handle_left_open(LocalTrade.bt_trades_open_pp, data=processed_lists)

        _t_bt_done = _time.monotonic()
        self.bt.wallets.update()
        t1 = dt_now()

        candles_total = sum(len(v) for v in processed_lists.values())
        self.window_stats.append(
            WindowRunStat(
                index=window.index,
                start=min_date.isoformat(),
                end=max_date.isoformat(),
                pairs_loaded=len(processed_lists),
                candles_total=candles_total,
                trades_after_window=len(LocalTrade.bt_trades),
                open_trades_after_window=len(LocalTrade.bt_trades_open),
                duration_sec=round((t1 - t0).total_seconds(), 3),
                status="ok",
                error=None,
                carry_over_trades=carry_over,
            )
        )

        logger.info(
            "[window %s] %s -> %s | pairs=%s candles=%s trades=%s open=%s | load=%.1fs ind=%.1fs bt=%.1fs total=%.2fs",
            window.index,
            min_date.strftime(DATETIME_PRINT_FORMAT),
            max_date.strftime(DATETIME_PRINT_FORMAT),
            len(processed_lists),
            candles_total,
            len(LocalTrade.bt_trades),
            len(LocalTrade.bt_trades_open),
            _t_load_done - _t_load,
            _t_ind_done - _t_ind,
            _t_bt_done - _t_bt,
            (t1 - t0).total_seconds(),
        )

        # Aggressive cleanup to avoid OOM
        del raw_data
        del preprocessed
        del preprocessed_tmp
        del processed_lists
        self.bt.detail_data = {}
        self.bt.futures_data = {}
        self.bt.dataprovider.clear_cache()
        gc.collect()

    def run(
        self,
        output_json: Path | None = None,
        *,
        fail_fast: bool = False,
        max_failed_windows: int = 0,
        export: str = "none",
        plot: bool = False,
        export_signals: bool = False,
    ) -> dict:
        """Run rolling backtest for all loaded strategies.

        Returns a dict with ``strategies`` list (one entry per strategy)
        and a combined ``all_bt_content`` for standard report generation.
        When only one strategy is loaded, the top-level keys are identical
        to the single-strategy result for backward compatibility.
        """
        if not self.bt.strategylist:
            raise RuntimeError("No strategy loaded.")

        all_strategy_results: list[dict] = []
        all_bt_content: dict = {}
        global_started_at = dt_now()

        for strat in self.bt.strategylist:
            strat_result, bt_content = self._run_strategy(
                strat=strat,
                fail_fast=fail_fast,
                max_failed_windows=max_failed_windows,
                export_signals=export_signals,
            )
            all_strategy_results.append(strat_result)
            if bt_content is not None:
                all_bt_content[strat.get_strategy_name()] = bt_content

        global_ended_at = dt_now()

        # Build combined result — backward compatible for single strategy
        if len(all_strategy_results) == 1:
            result = all_strategy_results[0]
        else:
            result = {
                "strategies": all_strategy_results,
                "summary": {
                    "total_strategies": len(all_strategy_results),
                    "start_ts": dt_ts(global_started_at),
                    "end_ts": dt_ts(global_ended_at),
                    "duration_sec": round(
                        (global_ended_at - global_started_at).total_seconds(), 3
                    ),
                },
            }

        # Multi-strategy comparison report
        if len(all_strategy_results) > 1:
            strat_perfs = {}
            for sr in all_strategy_results:
                sname = sr["strategy"]
                if "window_performances" in sr:
                    perfs = [WindowPerformance(**wp) for wp in sr["window_performances"]]
                    strat_perfs[sname] = perfs

            if strat_perfs:
                comparison = build_comparison_report(strat_perfs)
                table = format_comparison_table(comparison)
                print(table)
                result["comparison"] = comparison.to_dict()

        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # ---- Standard freqtrade reports (combined across all strategies) ----
        if all_bt_content:
            # Use global timerange from first strategy's windows
            first_result = all_strategy_results[0]
            self._generate_standard_reports(
                all_bt_content=all_bt_content,
                min_date=datetime.fromisoformat(first_result["timerange"]["start"]),
                max_date=datetime.fromisoformat(first_result["timerange"]["end"]),
                started_at=global_started_at,
                ended_at=global_ended_at,
                export=export,
                plot=plot,
            )

        return result

    def _run_strategy(
        self,
        *,
        strat,
        fail_fast: bool,
        max_failed_windows: int,
        export_signals: bool = False,
    ) -> tuple[dict, dict | None]:
        """Run rolling backtest for a single strategy.

        Returns (result_dict, bt_content_or_None).
        """
        strategy_name = strat.get_strategy_name()
        self.bt._set_strategy(strat)
        self.bt.reset_backtest(self.bt.enable_protections)
        self.bt.wallets.update()
        self.window_stats = []  # reset per strategy

        # Signal exporter (optional)
        signal_exporter: SignalExporter | None = None
        if export_signals:
            output_dir = Path(self.bt.config.get("user_data_dir", "user_data")) / "backtest_results"
            signal_exporter = SignalExporter(output_dir / f"signals_{strategy_name}")

        windows, self.window_plan = build_windows(
            self.bt.timerange,
            timeframe=self.bt.timeframe,
            window_days=self.window_days,
            num_pairs=len(self.bt.pairlists.whitelist),
            mem_budget_mb=self.mem_budget_mb,
            preferred_days=self.preferred_days,
            min_window_days=self.min_window_days,
            max_window_days=self.max_window_days,
        )
        if not windows:
            raise RuntimeError(f"No windows generated for strategy {strategy_name}.")

        logger.info(
            "Rolling backtest start: strategy=%s windows=%s timerange=%s",
            strategy_name,
            len(windows),
            self.bt.timerange.timerange_str,
        )

        # Pre-load all data once if we have multiple windows
        if len(windows) > 1:
            logger.info("Pre-loading full-range data for all %d windows...", len(windows))
            self._preload_all_data(self.bt.timerange)
        else:
            self._preloaded_data = None

        started_at = dt_now()
        failed_count = 0
        _window_times: list[float] = []

        for idx, window in enumerate(windows):
            try:
                _wt0 = _time.monotonic()
                self._run_single_window(
                    window=window,
                    is_last_window=(idx == len(windows) - 1),
                    signal_exporter=signal_exporter,
                )
                _wt1 = _time.monotonic()
                _window_times.append(_wt1 - _wt0)
                # Progress + ETA
                pct = (idx + 1) / len(windows) * 100
                avg_time = sum(_window_times) / len(_window_times)
                remaining = avg_time * (len(windows) - idx - 1)
                eta_min = remaining / 60
                logger.info(
                    "[%s] progress: %d/%d (%.1f%%) | avg=%.1fs/window | ETA=%.1fmin",
                    strategy_name, idx + 1, len(windows), pct, avg_time, eta_min,
                )
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                self.window_stats.append(
                    WindowRunStat(
                        index=window.index,
                        start=window.start.isoformat(),
                        end=window.end.isoformat(),
                        pairs_loaded=0,
                        candles_total=0,
                        trades_after_window=len(LocalTrade.bt_trades),
                        open_trades_after_window=len(LocalTrade.bt_trades_open),
                        duration_sec=0.0,
                        status="error",
                        error=str(exc),
                    )
                )
                logger.exception("[%s][window %s] failed", strategy_name, window.index)
                if fail_fast:
                    raise
                if max_failed_windows > 0 and failed_count > max_failed_windows:
                    raise RuntimeError(
                        f"Failed windows exceeded max_failed_windows={max_failed_windows}"
                    )

        ended_at = dt_now()

        # Free pre-loaded data cache after all windows processed
        if self._preloaded_data is not None:
            del self._preloaded_data
            self._preloaded_data = None
            gc.collect()

        trades_df = trade_list_to_dataframe(LocalTrade.bt_trades)
        final_balance = self.bt.wallets.get_total(strat.config["stake_currency"])

        status_counts = {
            "ok": sum(1 for x in self.window_stats if x.status == "ok"),
            "no_data": sum(1 for x in self.window_stats if x.status == "no_data"),
            "error": sum(1 for x in self.window_stats if x.status == "error"),
        }

        strat_result = {
            "strategy": strategy_name,
            "window_days": self.window_days,
            "timerange": {
                "start": windows[0].start.isoformat(),
                "end": windows[-1].end.isoformat(),
            },
            "windows": [asdict(x) for x in self.window_stats],
            "summary": {
                "total_windows": len(windows),
                "ok_windows": status_counts["ok"],
                "no_data_windows": status_counts["no_data"],
                "failed_windows": status_counts["error"],
                "total_trades": int(len(trades_df)),
                "open_trades_end": int(len(LocalTrade.bt_trades_open)),
                "final_balance": float(final_balance),
                "trade_digest": compute_trade_digest(trades_df),
                "start_ts": dt_ts(started_at),
                "end_ts": dt_ts(ended_at),
                "duration_sec": round((ended_at - started_at).total_seconds(), 3),
            },
        }

        # ---- Per-window performance metrics ----
        window_boundaries = [
            (ws.index, datetime.fromisoformat(ws.start), datetime.fromisoformat(ws.end))
            for ws in self.window_stats
            if ws.status == "ok"
        ]
        starting_bal = strat.config.get("dry_run_wallet", 1000.0)
        window_performances = compute_all_window_performances(
            all_trades=trades_df,
            window_boundaries=window_boundaries,
            starting_balance=starting_bal,
        )
        strat_result["window_performances"] = [wp.to_dict() for wp in window_performances]

        # Window stability summary
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

        # Build BacktestContentType for standard report generation
        bt_content = None
        first_ok = next((ws for ws in self.window_stats if ws.status == "ok"), None)
        if first_ok:
            bt_content = {
                "results": trades_df,
                "config": strat.config,
                "locks": PairLocks.get_all_locks(),
                "rejected_signals": self.bt.rejected_trades,
                "timedout_entry_orders": self.bt.timedout_entry_orders,
                "timedout_exit_orders": self.bt.timedout_exit_orders,
                "canceled_trade_entries": self.bt.canceled_trade_entries,
                "canceled_entry_orders": self.bt.canceled_entry_orders,
                "replaced_entry_orders": self.bt.replaced_entry_orders,
                "final_balance": final_balance,
                "backtest_start_time": int(started_at.timestamp()),
                "backtest_end_time": int(ended_at.timestamp()),
                "run_id": "",
                "wallet_summary": convert_bt_wallet_collection(self.bt.wallet_captures),
            }

        logger.info(
            "[%s] completed: %d trades, balance=%.4f, duration=%.1fs",
            strategy_name,
            len(trades_df),
            final_balance,
            (ended_at - started_at).total_seconds(),
        )

        # Print per-window rolling performance table
        self._print_window_summary_table(strategy_name, strat.config.get("stake_currency", ""), window_performances)

        # Export signals and detect deviations if enabled
        if signal_exporter is not None:
            signal_exporter.export()
            signal_exporter.export_deviations()

        return strat_result, bt_content

    def _print_window_summary_table(
        self,
        strategy_name: str,
        stake_currency: str,
        window_performances: list[WindowPerformance] | None = None,
    ) -> None:
        """Print a compact table showing per-window trade counts, performance metrics, and cumulative stats."""
        ok_windows = [ws for ws in self.window_stats if ws.status == "ok"]
        if not ok_windows:
            return

        # Build lookup from window_index -> WindowPerformance
        perf_map: dict[int, WindowPerformance] = {}
        if window_performances:
            perf_map = {wp.window_index: wp for wp in window_performances}

        w = 130  # table width
        header = (
            f"\n{'=' * w}\n"
            f" Rolling Window Summary — {strategy_name} ({stake_currency})\n"
            f"{'=' * w}\n"
            f" {'Win':>4} | {'Period':^23} | {'Trades':>6} | {'Carry':>5} | {'Open':>4} | "
            f"{'Profit%':>8} | {'WinRate':>7} | {'Sharpe':>7} | {'Sortino':>7} | {'MaxDD%':>7} | {'PF':>6} | {'Status':^7}\n"
            f"{'-' * w}"
        )
        lines = [header]
        for ws in self.window_stats:
            start_short = ws.start[:10] if ws.start else "?"
            end_short = ws.end[:10] if ws.end else "?"
            period = f"{start_short} → {end_short}"

            wp = perf_map.get(ws.index)
            # Use per-window trade count from WindowPerformance (not cumulative)
            if wp and wp.total_trades > 0:
                trades_str = f"{wp.total_trades:>6}"
                carry_str = f"{wp.carry_over_trades:>5}"
                profit_str = f"{wp.total_profit_pct:>8.2f}"
                wr_str = f"{wp.win_rate * 100:>6.1f}%"
                sharpe_str = f"{wp.sharpe_ratio:>7.2f}"
                sortino_str = f"{wp.sortino_ratio:>7.2f}" if wp.sortino_ratio < 999 else f"{'inf':>7}"
                dd_str = f"{wp.max_drawdown * 100:>6.2f}%"
                pf_str = f"{wp.profit_factor:>6.2f}" if wp.profit_factor < 999 else f"{'inf':>6}"
            elif wp:
                trades_str = f"{0:>6}"
                carry_str = f"{wp.carry_over_trades:>5}"
                profit_str = f"{'—':>8}"
                wr_str = f"{'—':>7}"
                sharpe_str = f"{'—':>7}"
                sortino_str = f"{'—':>7}"
                dd_str = f"{'—':>7}"
                pf_str = f"{'—':>6}"
            else:
                trades_str = f"{'?':>6}"
                carry_str = f"{getattr(ws, 'carry_over_trades', 0):>5}"
                profit_str = f"{'—':>8}"
                wr_str = f"{'—':>7}"
                sharpe_str = f"{'—':>7}"
                sortino_str = f"{'—':>7}"
                dd_str = f"{'—':>7}"
                pf_str = f"{'—':>6}"

            lines.append(
                f" {ws.index:>4} | {period:^23} | {trades_str} | {carry_str} | "
                f"{ws.open_trades_after_window:>4} | {profit_str} | {wr_str} | "
                f"{sharpe_str} | {sortino_str} | {dd_str} | {pf_str} | {ws.status:^7}"
            )
        lines.append(f"{'=' * w}")

        # Stability footer if we have metrics
        if perf_map:
            active = [wp for wp in window_performances if wp.total_trades > 0]
            if active:
                avg_sharpe = np.mean([wp.sharpe_ratio for wp in active])
                avg_sortino = np.mean([wp.sortino_ratio for wp in active if wp.sortino_ratio < 999])
                avg_wr = np.mean([wp.win_rate for wp in active]) * 100
                worst_dd = max(wp.max_drawdown for wp in active) * 100
                total_profit = sum(wp.total_profit_pct for wp in active)
                total_trades = sum(wp.total_trades for wp in active)
                total_carry = sum(wp.carry_over_trades for wp in active)
                lines.append(
                    f" {'TOTAL':>4} | {'':^23} | {total_trades:>6} | {total_carry:>5} | "
                    f"{'':>4} | {total_profit:>8.2f} | {avg_wr:>6.1f}% | "
                    f"{avg_sharpe:>7.2f} | {avg_sortino:>7.2f} | {worst_dd:>6.2f}% | {'':>6} |"
                )
                lines.append(f"{'=' * w}")

        print("\n".join(lines))

    def _generate_standard_reports(
        self,
        *,
        all_bt_content: dict,
        min_date: datetime,
        max_date: datetime,
        started_at: datetime,
        ended_at: datetime,
        export: str,
        plot: bool = False,
    ) -> None:
        """Generate freqtrade-standard backtest stats, console output, ZIP export, and plots."""

        # Load data for market_change calculation
        full_tr = TimeRange.parse_timerange(
            f"{min_date.strftime('%Y%m%d')}-{max_date.strftime('%Y%m%d')}"
        )
        btdata = history.load_data(
            datadir=self.bt.config["datadir"],
            pairs=self.bt.pairlists.whitelist,
            timeframe=self.bt.timeframe,
            timerange=full_tr,
            startup_candles=0,
            fail_without_data=False,
            data_format=self.bt.config["dataformat_ohlcv"],
            candle_type=self.bt.config.get("candle_type_def"),
        )

        if not btdata:
            logger.warning("Cannot generate standard reports: no data available for market change.")
            return

        try:
            stats = generate_backtest_stats(
                btdata=btdata,
                all_results=all_bt_content,
                min_date=min_date,
                max_date=max_date,
            )

            # Console output — formatted tables
            show_backtest_results(self.bt.config, stats)

            # Export to standard ZIP if requested
            if export in ("trades", "signals"):
                dt_appendix = started_at.strftime("%Y-%m-%d_%H-%M-%S")
                market_change_data = combined_dataframes_with_rel_mean(btdata, min_date, max_date)
                wallet_summary = {
                    s: x["wallet_summary"]
                    for s, x in all_bt_content.items()
                    if "wallet_summary" in x
                }
                # Build strategy_files mapping from loaded strategies
                strategy_files = {}
                for s in self.bt.strategylist:
                    sname = s.get_strategy_name()
                    if sname in all_bt_content and getattr(s, "__file__", None):
                        strategy_files[sname] = s.__file__
                outpath = store_backtest_results(
                    self.bt.config,
                    stats,
                    dt_appendix,
                    market_change_data=market_change_data,
                    wallet_summary=wallet_summary,
                    strategy_files=strategy_files or None,
                )
                logger.info("Standard backtest results stored to: %s", outpath)

            # Generate profit plot if requested
            if plot:
                self._generate_plots(btdata, all_bt_content, min_date, max_date)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to generate standard reports: %s", exc)
        finally:
            # Clean up loaded data
            del btdata
            gc.collect()

    def _generate_plots(
        self,
        btdata: dict[str, DataFrame],
        all_bt_content: dict,
        min_date: datetime,
        max_date: datetime,
    ) -> None:
        """Generate profit plot for each strategy."""
        try:
            from freqtrade.plot.plotting import generate_profit_graph, store_plot_file
        except ImportError:
            logger.warning("plotly not installed — skipping plot generation. pip install plotly")
            return

        plot_dir = Path(self.bt.config.get("user_data_dir", "user_data")) / "plot"
        plot_dir.mkdir(parents=True, exist_ok=True)
        stake_currency = self.bt.config.get("stake_currency", "USDT")

        for strategy_name, content in all_bt_content.items():
            trades_df = content["results"]
            if trades_df.empty:
                logger.info("[%s] No trades — skipping plot.", strategy_name)
                continue

            starting_balance = self.bt.config.get("dry_run_wallet", 1000.0)
            fig = generate_profit_graph(
                pairs=self.bt.pairlists.whitelist,
                data=btdata,
                trades=trades_df,
                timeframe=self.bt.timeframe,
                stake_currency=stake_currency,
                starting_balance=starting_balance,
            )
            filename = f"rolling-profit-{strategy_name}.html"
            store_plot_file(fig, filename, directory=plot_dir, auto_open=False)
            logger.info("[%s] Profit plot saved: %s", strategy_name, plot_dir / filename)
