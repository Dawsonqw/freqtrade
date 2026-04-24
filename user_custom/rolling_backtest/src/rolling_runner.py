from __future__ import annotations

import gc
import json
import logging
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

from .parity_tools import compute_trade_digest
from .windowing import Window, build_windows


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


class RollingBacktestRunner:
    """
    Chunked backtesting runner reusing freqtrade Backtesting internals.
    - Loads OHLCV by time window.
    - Preserves in-memory trade/wallet state across windows.
    - Frees dataframe memory after each window.
    """

    def __init__(self, backtesting: Backtesting, window_days: int) -> None:
        self.bt = backtesting
        self.window_days = window_days
        self.window_stats: list[WindowRunStat] = []

    def _load_window_data(self, tr: TimeRange) -> dict[str, DataFrame]:
        data = history.load_data(
            datadir=self.bt.config["datadir"],
            pairs=self.bt.pairlists.whitelist,
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
        return data

    def _run_single_window(
        self,
        *,
        window: Window,
        is_last_window: bool,
    ) -> None:
        t0 = dt_now()
        self.bt.timerange = window.timerange

        raw_data = self._load_window_data(window.timerange)
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

        # 1) indicators
        preprocessed = self.bt.strategy.advise_all_indicators(raw_data)
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
            )
        )

        logger.info(
            "[window %s] %s -> %s | pairs=%s candles=%s trades=%s open=%s in %.2fs",
            window.index,
            min_date.strftime(DATETIME_PRINT_FORMAT),
            max_date.strftime(DATETIME_PRINT_FORMAT),
            len(processed_lists),
            candles_total,
            len(LocalTrade.bt_trades),
            len(LocalTrade.bt_trades_open),
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
            )

        return result

    def _run_strategy(
        self,
        *,
        strat,
        fail_fast: bool,
        max_failed_windows: int,
    ) -> tuple[dict, dict | None]:
        """Run rolling backtest for a single strategy.

        Returns (result_dict, bt_content_or_None).
        """
        strategy_name = strat.get_strategy_name()
        self.bt._set_strategy(strat)
        self.bt.reset_backtest(self.bt.enable_protections)
        self.bt.wallets.update()
        self.window_stats = []  # reset per strategy

        windows = build_windows(
            self.bt.timerange,
            timeframe=self.bt.timeframe,
            window_days=self.window_days,
        )
        if not windows:
            raise RuntimeError(f"No windows generated for strategy {strategy_name}.")

        logger.info(
            "Rolling backtest start: strategy=%s windows=%s timerange=%s",
            strategy_name,
            len(windows),
            self.bt.timerange.timerange_str,
        )

        started_at = dt_now()
        failed_count = 0

        for idx, window in enumerate(windows):
            try:
                self._run_single_window(window=window, is_last_window=(idx == len(windows) - 1))
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

        return strat_result, bt_content

    def _generate_standard_reports(
        self,
        *,
        all_bt_content: dict,
        min_date: datetime,
        max_date: datetime,
        started_at: datetime,
        ended_at: datetime,
        export: str,
    ) -> None:
        """Generate freqtrade-standard backtest stats, console output, and ZIP export."""

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
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to generate standard reports: %s", exc)
        finally:
            # Clean up loaded data
            del btdata
            gc.collect()
