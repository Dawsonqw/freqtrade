#!/usr/bin/env python3
"""
Parity diagnosis: run built-in and rolling backtest on the same config,
then compare trades side-by-side to identify the exact source of deviation.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datetime import datetime

import pandas as pd

from freqtrade.commands.optimize_commands import setup_optimize_configuration
from freqtrade.enums import RunMode
from freqtrade.optimize.backtesting import Backtesting

from user_custom.rolling_backtest.src.rolling_runner import RollingBacktestRunner
from user_custom.rolling_backtest.scripts.run_rolling_backtest import patch_exchange_for_offline_init

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("parity_diagnosis")

# ---- Config ----
CONFIG = {
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "dry_run_wallet": 10000.0,
    "max_open_trades": 1,
    "exchange": {
        "name": "binance",
        "pair_whitelist": ["BTC/USDT:USDT"],
        "pair_blacklist": [],
    },
    "pairlists": [{"method": "StaticPairList"}],
    "datadir": "/data/freqtrade_data/parity_test",
    "user_data_dir": "/root/workspace/freqtrade/user_data",
    "strategy_path": "/root/workspace/freqtrade/user_custom/rolling_backtest/tests/parity_test",
    "dataformat_ohlcv": "feather",
    "candle_type_def": "futures",
    "timeframe": "5m",
    "timerange": "20240101-20240301",  # 2 months for faster diagnosis
    "fee": 0.0004,
    "futures_funding_rate": 0.0,
    "liquidation_buffer": 0.05,
    "enable_protections": False,
    "entry_pricing": {"price_side": "other", "use_order_book": True, "order_book_top": 1},
    "exit_pricing": {"price_side": "other", "use_order_book": True, "order_book_top": 1},
    "internals": {"process_throttle_secs": 0},
}


def _make_config(config: dict) -> dict:
    """Build a freqtrade-compatible config dict directly (no CLI parsing)."""
    from copy import deepcopy
    from freqtrade.configuration import TimeRange

    cfg = deepcopy(config)
    cfg["runmode"] = RunMode.BACKTEST
    cfg["strategy"] = "ParityTestStrategy"
    cfg["strategy_list"] = ["ParityTestStrategy"]
    cfg["export"] = "none"
    cfg["cache"] = "none"
    cfg["backtest_breakdown"] = []
    cfg["pairs"] = cfg["exchange"]["pair_whitelist"]
    cfg["verbosity"] = 0
    cfg["tradable_balance_ratio"] = 1.0
    cfg["available_capital"] = config.get("dry_run_wallet", 10000.0)
    # Ensure paths are Path objects
    cfg["datadir"] = Path(cfg["datadir"])
    cfg["user_data_dir"] = Path(cfg["user_data_dir"])
    cfg["strategy_path"] = Path(cfg["strategy_path"])
    # Parse timerange
    cfg["timerange_parsed"] = TimeRange.parse_timerange(cfg["timerange"])
    return cfg


def run_builtin(config: dict) -> pd.DataFrame:
    """Run standard freqtrade backtest, return trades DataFrame."""
    logger.info("=== Running BUILT-IN backtest ===")
    cfg = _make_config(config)

    t0 = time.time()
    bt = Backtesting(cfg)
    bt.start()
    t1 = time.time()
    logger.info("Built-in backtest done in %.1fs", t1 - t0)

    from freqtrade.data.btanalysis import trade_list_to_dataframe
    from freqtrade.persistence import LocalTrade

    trades_df = trade_list_to_dataframe(LocalTrade.bt_trades)
    logger.info("Built-in trades: %d", len(trades_df))
    return trades_df


def run_rolling(config: dict, window_days: int) -> pd.DataFrame:
    """Run rolling backtest, return trades DataFrame."""
    logger.info("=== Running ROLLING backtest (window_days=%d) ===", window_days)
    cfg = _make_config(config)

    t0 = time.time()
    bt = Backtesting(cfg)
    runner = RollingBacktestRunner(bt, window_days=window_days)
    result = runner.run(fail_fast=True)
    t1 = time.time()
    logger.info("Rolling backtest done in %.1fs", t1 - t0)

    from freqtrade.data.btanalysis import trade_list_to_dataframe
    from freqtrade.persistence import LocalTrade

    trades_df = trade_list_to_dataframe(LocalTrade.bt_trades)
    logger.info("Rolling trades: %d", len(trades_df))
    return trades_df


def compare_trades(builtin_df: pd.DataFrame, rolling_df: pd.DataFrame) -> dict:
    """Compare trades and identify deviations."""
    result = {
        "builtin_count": len(builtin_df),
        "rolling_count": len(rolling_df),
        "count_diff": len(rolling_df) - len(builtin_df),
    }

    if builtin_df.empty or rolling_df.empty:
        logger.warning("One or both trade sets are empty!")
        return result

    # Normalize columns for comparison
    cols = ["pair", "open_date", "close_date", "open_rate", "close_rate",
            "profit_ratio", "is_short", "exit_reason"]
    available = [c for c in cols if c in builtin_df.columns and c in rolling_df.columns]

    b = builtin_df[available].copy()
    r = rolling_df[available].copy()

    # Convert dates to string for comparison
    for c in ["open_date", "close_date"]:
        if c in b.columns:
            b[c] = pd.to_datetime(b[c], utc=True).dt.strftime("%Y-%m-%d %H:%M")
            r[c] = pd.to_datetime(r[c], utc=True).dt.strftime("%Y-%m-%d %H:%M")

    # Match by open_date (primary key for single-pair, max_open_trades=1)
    b = b.sort_values("open_date").reset_index(drop=True)
    r = r.sort_values("open_date").reset_index(drop=True)

    # Find first divergence point
    min_len = min(len(b), len(r))
    first_diff_idx = None
    diff_details = []

    for i in range(min_len):
        diffs = {}
        for c in available:
            bv = b.iloc[i][c]
            rv = r.iloc[i][c]
            if isinstance(bv, float) and isinstance(rv, float):
                if abs(bv - rv) > 1e-8:
                    diffs[c] = {"builtin": bv, "rolling": rv, "diff": rv - bv}
            elif str(bv) != str(rv):
                diffs[c] = {"builtin": str(bv), "rolling": str(rv)}

        if diffs:
            if first_diff_idx is None:
                first_diff_idx = i
            diff_details.append({
                "index": i,
                "builtin_open_date": str(b.iloc[i].get("open_date", "?")),
                "rolling_open_date": str(r.iloc[i].get("open_date", "?")),
                "diffs": diffs,
            })
            if len(diff_details) >= 20:  # Cap at 20 diffs
                break

    result["first_diff_index"] = first_diff_idx
    result["total_diffs_found"] = len(diff_details)
    result["diff_details"] = diff_details

    # Extra trades analysis
    if len(b) != len(r):
        b_dates = set(b["open_date"].tolist())
        r_dates = set(r["open_date"].tolist())
        only_builtin = sorted(b_dates - r_dates)
        only_rolling = sorted(r_dates - b_dates)
        result["trades_only_in_builtin"] = only_builtin[:20]
        result["trades_only_in_rolling"] = only_rolling[:20]

    # Analyze exit reasons distribution
    if "exit_reason" in available:
        b_exits = builtin_df["exit_reason"].value_counts().to_dict()
        r_exits = rolling_df["exit_reason"].value_counts().to_dict()
        result["builtin_exit_reasons"] = b_exits
        result["rolling_exit_reasons"] = r_exits

    return result


def main():
    output_dir = Path("/root/workspace/freqtrade/user_custom/rolling_backtest/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_exchange_for_offline_init()

    # Step 1: Built-in backtest
    builtin_trades = run_builtin(CONFIG.copy())
    builtin_trades.to_json(output_dir / "diag_builtin_trades.json", orient="records", indent=2)

    # Step 2: Rolling backtest with 30-day windows
    rolling_trades = run_rolling(CONFIG.copy(), window_days=30)
    rolling_trades.to_json(output_dir / "diag_rolling_trades.json", orient="records", indent=2)

    # Step 3: Compare
    comparison = compare_trades(builtin_trades, rolling_trades)
    print("\n" + "=" * 80)
    print("PARITY DIAGNOSIS RESULT")
    print("=" * 80)
    print(f"Built-in trades: {comparison['builtin_count']}")
    print(f"Rolling trades:  {comparison['rolling_count']}")
    print(f"Count diff:      {comparison['count_diff']}")
    print(f"First diff at:   index {comparison.get('first_diff_index', 'N/A')}")
    print()

    if comparison.get("diff_details"):
        print("--- First divergences ---")
        for dd in comparison["diff_details"][:10]:
            print(f"\n  Trade #{dd['index']}:")
            print(f"    Builtin open: {dd['builtin_open_date']}")
            print(f"    Rolling open: {dd['rolling_open_date']}")
            for col, vals in dd["diffs"].items():
                print(f"    {col}: {vals}")

    if comparison.get("trades_only_in_builtin"):
        print(f"\n--- Trades ONLY in built-in ({len(comparison['trades_only_in_builtin'])}) ---")
        for d in comparison["trades_only_in_builtin"][:10]:
            print(f"  {d}")

    if comparison.get("trades_only_in_rolling"):
        print(f"\n--- Trades ONLY in rolling ({len(comparison['trades_only_in_rolling'])}) ---")
        for d in comparison["trades_only_in_rolling"][:10]:
            print(f"  {d}")

    if "builtin_exit_reasons" in comparison:
        print(f"\n--- Exit reasons ---")
        print(f"  Built-in: {comparison['builtin_exit_reasons']}")
        print(f"  Rolling:  {comparison['rolling_exit_reasons']}")

    # Save full comparison
    with open(output_dir / "diag_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nFull results saved to {output_dir}/diag_*.json")


if __name__ == "__main__":
    main()
