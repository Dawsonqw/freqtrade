#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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

from user_custom.rolling_backtest.src.rolling_runner import RollingBacktestRunner

EXCHANGE_CACHE = Path("/data/freqtrade_data/_meta/exchange_cache.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("run_rolling_backtest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run chunked rolling backtest")
    p.add_argument("--config", required=True, help="Freqtrade config path")
    p.add_argument("--strategy", default=None, help="Override strategy name")
    p.add_argument("--strategy-list", nargs="+", default=None, help="List of strategies to compare")
    p.add_argument("--timerange", default=None, help="Override timerange")
    p.add_argument("--window-days", type=int, default=0,
                    help="Chunk size in days (0 = auto-calculate based on memory and coverage)")
    p.add_argument("--mem-budget-mb", type=float, default=0,
                    help="Max memory per window in MB (0 = auto-detect 50%% free RAM)")
    p.add_argument("--preferred-days", type=int, default=0,
                    help="Hint for preferred window size when auto-calculating (0 = no preference)")
    p.add_argument("--min-window-days", type=int, default=7,
                    help="Minimum window size in days for auto mode (default: 7)")
    p.add_argument("--max-window-days", type=int, default=90,
                    help="Maximum window size in days for auto mode (default: 90)")
    p.add_argument("--datadir", default="/data/freqtrade_data", help="OHLCV data directory")
    p.add_argument("--user-data-dir", default="/root/workspace/freqtrade/user_data", help="freqtrade user_data directory")
    p.add_argument("--pairs-file", default=None, help="CSV/TXT with pair list")
    p.add_argument("--output-json", default="user_custom/rolling_backtest/output/rolling_backtest_result.json")
    p.add_argument("--log-file", default=None)
    p.add_argument("--fail-fast", action="store_true", help="Stop immediately when a window fails")
    p.add_argument(
        "--max-failed-windows",
        type=int,
        default=0,
        help="Abort when failed window count exceeds this threshold (0 = unlimited)",
    )
    p.add_argument(
        "--export",
        default="none",
        choices=["none", "trades", "signals"],
        help="Export standard freqtrade backtest ZIP (none/trades/signals)",
    )
    p.add_argument(
        "--enable-protections",
        action="store_true",
        help="Enable protections during backtest",
    )
    p.add_argument(
        "--timeframe-detail",
        default=None,
        help="Detail timeframe for more accurate backtest simulation",
    )
    p.add_argument(
        "--backtest-breakdown",
        nargs="+",
        default=[],
        choices=["day", "week", "month"],
        help="Show profit breakdown per period (day/week/month)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate profit plot (HTML) after backtest completes",
    )
    p.add_argument(
        "--export-signals",
        action="store_true",
        help="Export entry/exit signals per window and detect deviations",
    )
    p.add_argument(
        "--window-metrics",
        action="store_true",
        help="Enable per-window performance metrics (Sharpe/Sortino/drawdown/win rate)",
    )
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="Number of parallel workers for indicator computation (0 = serial, default: 0). "
             "Also enables data pre-loading to avoid repeated disk I/O across windows.",
    )
    return p.parse_args()


def setup_file_logger(log_file: str | None) -> None:
    if not log_file:
        return
    fp = Path(log_file)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(fp, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def inject_cached_markets(backtesting: Backtesting) -> None:
    """Inject cached markets data to avoid slow network requests during init."""
    pass  # Now handled by pre-init patching


def patch_exchange_for_offline_init() -> None:
    """Monkey-patch exchange to skip network calls during Backtesting init.
    
    Injects cached markets data so reload_markets() and fill_leverage_tiers()
    don't need to hit the network.
    """
    if not EXCHANGE_CACHE.exists():
        logger.warning("Exchange cache not found at %s, skipping offline patch", EXCHANGE_CACHE)
        return

    import time as _time
    from freqtrade.exchange.exchange import Exchange

    t0 = _time.time()
    logger.info("Loading cached exchange data for offline init...")
    with EXCHANGE_CACHE.open("r") as f:
        cache = json.load(f)
    cached_markets = cache.get("markets", {})
    if not cached_markets:
        logger.warning("Exchange cache has no markets, skipping offline patch")
        return
    logger.info("Loaded %d cached markets in %.1fs", len(cached_markets), _time.time() - t0)

    _original_reload_markets = Exchange.reload_markets

    def _patched_reload_markets(self, force=False, *, load_leverage_tiers=True):
        """Skip first markets load — inject cached data instead."""
        if self._last_markets_refresh == 0 and cached_markets:
            logger.info("Injecting %d cached markets (skipping network)", len(cached_markets))
            self._api.markets = cached_markets
            self._api_async.markets = cached_markets
            self._api.markets_by_id = self._api.index_by(list(cached_markets.values()), "id")
            self._api_async.markets_by_id = self._api_async.index_by(list(cached_markets.values()), "id")
            self._markets = cached_markets
            self._last_markets_refresh = int(_time.time() * 1000)
            # Still load leverage tiers from cache (handled by freqtrade's own cache mechanism)
            if load_leverage_tiers:
                from freqtrade.enums import TradingMode
                if self.trading_mode == TradingMode.FUTURES:
                    self.fill_leverage_tiers()
            return
        return _original_reload_markets(self, force, load_leverage_tiers=load_leverage_tiers)

    Exchange.reload_markets = _patched_reload_markets
    logger.info("Exchange patched for offline initialization")


def load_pairs(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")
    if p.suffix.lower() == ".csv":
        out = []
        with p.open("r", encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            if "pair" not in rd.fieldnames:
                raise ValueError("CSV must include 'pair' column")
            for row in rd:
                out.append(row["pair"].strip())
        return sorted(set(x for x in out if x))

    # TXT fallback: one pair per line
    lines = [x.strip() for x in p.read_text(encoding="utf-8").splitlines()]
    return sorted(set(x for x in lines if x and not x.startswith("#")))


def build_args(ns: argparse.Namespace) -> dict[str, Any]:
    args: dict[str, Any] = {
        "config": [ns.config],
        "strategy": ns.strategy,
        "strategy_list": ns.strategy_list or [],
        "timerange": ns.timerange,
        "verbosity": 0,
        "timeframe": None,
        "timeframe_detail": ns.timeframe_detail,
        "datadir": ns.datadir,
        "user_data_dir": ns.user_data_dir,
        "export": "none",
        "cache": "none",
        "backtest_breakdown": ns.backtest_breakdown,
        "enable_protections": ns.enable_protections,
    }
    return args


def main() -> None:
    ns = parse_args()
    setup_file_logger(ns.log_file)

    logger.info("=== Rolling Backtest Starting ===")
    logger.info("Parsing configuration...")
    args = build_args(ns)
    config = setup_optimize_configuration(args, RunMode.BACKTEST)
    # Re-add file logger after freqtrade's logging setup (which reconfigures root logger)
    setup_file_logger(ns.log_file)
    logger.info("Configuration loaded: timerange=%s, datadir=%s", ns.timerange, ns.datadir)

    if ns.pairs_file:
        pairs = load_pairs(ns.pairs_file)
        if not pairs:
            raise RuntimeError("Pairs file is empty")
        config["pairs"] = pairs
        config["exchange"]["pair_whitelist"] = pairs
        config["pairlists"] = [{"method": "StaticPairList"}]
        logger.info("Loaded pairs from file: %s", len(pairs))

    # Patch exchange to use cached markets (skip network requests)
    logger.info("Patching exchange for offline initialization...")
    patch_exchange_for_offline_init()

    logger.info("Initializing Backtesting engine (exchange + strategy)...")
    import time as _time
    t0 = _time.time()
    backtesting = Backtesting(config)
    logger.info("Backtesting engine initialized in %.1fs (pairs=%d)", 
                _time.time() - t0, len(backtesting.pairlists.whitelist))

    logger.info("Creating RollingBacktestRunner (window_days=%s)...", ns.window_days)
    runner = RollingBacktestRunner(
        backtesting=backtesting,
        window_days=ns.window_days,
        mem_budget_mb=ns.mem_budget_mb,
        preferred_days=ns.preferred_days,
        min_window_days=ns.min_window_days,
        max_window_days=ns.max_window_days,
        parallel_workers=ns.parallel_workers,
    )
    logger.info("Starting rolling backtest run...")
    result = runner.run(
        output_json=Path(ns.output_json),
        fail_fast=ns.fail_fast,
        max_failed_windows=ns.max_failed_windows,
        export=ns.export,
        plot=ns.plot,
        export_signals=ns.export_signals,
    )

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
