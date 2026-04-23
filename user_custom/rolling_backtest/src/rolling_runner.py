from __future__ import annotations

import gc
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.data import history
from freqtrade.data.btanalysis import get_tick_size_over_time, trade_list_to_dataframe
from freqtrade.data.converter import trim_dataframes
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import LocalTrade
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
    ) -> dict:
        if not self.bt.strategylist:
            raise RuntimeError("No strategy loaded.")
        if len(self.bt.strategylist) > 1:
            logger.warning("Rolling runner currently executes only first strategy in strategylist.")

        strat = self.bt.strategylist[0]
        self.bt._set_strategy(strat)
        self.bt.reset_backtest(self.bt.enable_protections)
        self.bt.wallets.update()

        windows = build_windows(
            self.bt.timerange,
            timeframe=self.bt.timeframe,
            window_days=self.window_days,
        )
        if not windows:
            raise RuntimeError("No windows generated from timerange.")

        logger.info(
            "Rolling backtest start: strategy=%s windows=%s timerange=%s",
            strat.get_strategy_name(),
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
                logger.exception("[window %s] failed", window.index)
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

        result = {
            "strategy": strat.get_strategy_name(),
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

        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        return result
